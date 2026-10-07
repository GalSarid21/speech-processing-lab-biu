import base64
import io
import json

import numpy as np
import pytest
import soundfile as sf
from assertpy import assert_that

from speech_processing.adapters import openai_server
from speech_processing.adapters.openai_server import (
    OpenAIChatServerAdapter,
    audio_file_to_input_parts,
    to_server_messages,
)
from speech_processing.config.core import GenerationParams

SR = 16_000
PARAMS = GenerationParams(temperature=0.5, top_p=0.5, max_new_tokens=4096)


@pytest.fixture
def clip(tmp_path):
    def _write(seconds: float) -> str:
        path = tmp_path / f"clip_{seconds}.wav"
        sf.write(path, np.zeros(int(seconds * SR), dtype=np.float32), SR)
        return str(path)

    return _write


class FakeServer:
    """Stands in for urllib.request.urlopen: records payloads and replies with queued answers."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.payloads = []

    def __call__(self, request, timeout):
        self.payloads.append(json.loads(request.data))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        body = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
        return io.BytesIO(body)


@pytest.fixture
def server(monkeypatch):
    def _install(*answers):
        fake = FakeServer(answers)
        monkeypatch.setattr(openai_server.urllib.request, "urlopen", fake)
        monkeypatch.setattr(openai_server, "RETRY_DELAY_S", 0)
        return fake

    return _install


def test_long_audio_is_sent_as_25_second_wav_chunks(clip):
    parts = audio_file_to_input_parts(clip(60))

    assert_that(parts).is_length(3)
    durations = [sf.info(io.BytesIO(base64.b64decode(p["input_audio"]["data"]))).duration for p in parts]
    assert_that(durations).is_equal_to([25.0, 25.0, 10.0])
    assert_that(parts[0]["input_audio"]["format"]).is_equal_to("wav")


def test_messages_inline_audio_and_drop_reasoning_from_history(clip):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": f"file://{clip(3)}"}},
                {"type": "text", "text": "Question?"},
            ],
        },
        {"role": "assistant", "content": "<think>\nlong reasoning</think>\nB"},
        {"role": "user", "content": "Are you sure?"},
    ]

    messages = to_server_messages(conversation)

    assert_that([part["type"] for part in messages[0]["content"]]).is_equal_to(["input_audio", "text"])
    assert_that(messages[1]["content"]).is_equal_to("B")
    assert_that(messages[2]["content"]).is_equal_to("Are you sure?")


def test_reasoning_prefix_is_restored_and_prefill_continues(server):
    fake = server("thinking...</think>\nB", "C</analysis>")
    adapter = OpenAIChatServerAdapter("http://host/v1/", "step", reasoning_prefix="<think>\n", stop_token_ids=[7])

    free = adapter.generate_batch([[{"role": "user", "content": "Q"}]], PARAMS)
    prefilled = adapter.generate_batch(
        [[{"role": "user", "content": "Q"}, {"role": "assistant", "content": "<analysis>"}]], PARAMS
    )

    assert_that(free).is_equal_to(["<think>\nthinking...</think>\nB"])
    assert_that(prefilled).is_equal_to(["C</analysis>"])
    first, second = fake.payloads
    assert_that(first).contains_entry({"add_generation_prompt": True}, {"continue_final_message": False})
    assert_that(first).contains_entry({"max_tokens": 4096}, {"stop_token_ids": [7]}, {"model": "step"})
    assert_that(second).contains_entry({"add_generation_prompt": False}, {"continue_final_message": True})


def test_failed_requests_are_retried_then_raised(server):
    server(TimeoutError("slow"), "A")
    adapter = OpenAIChatServerAdapter("http://host/v1", "step")
    assert_that(adapter.generate_batch([[{"role": "user", "content": "Q"}]], PARAMS)).is_equal_to(["A"])

    server(*[TimeoutError("down")] * openai_server.MAX_ATTEMPTS)
    with pytest.raises(RuntimeError, match="failed after"):
        adapter.generate_batch([[{"role": "user", "content": "Q"}]], PARAMS)
