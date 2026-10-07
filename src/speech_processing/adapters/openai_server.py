"""Generation through an OpenAI-compatible chat server instead of an in-process vLLM engine.

Some audio models only run on a vendor's own vLLM build (Step-Audio-R1.1 needs StepFun's fork). That
build lives in its own virtual environment and serves the model over HTTP; this adapter lets the
pipeline send it the exact conversations it would hand to the in-process engine, so prompts, few-shot
turns and multi-turn logic stay identical across models.

Audio parts arrive as `audio_url` file:// parts and are sent inline as base64 WAV `input_audio` parts,
cut into chunks of at most `MAX_AUDIO_CHUNK_S` seconds, as StepFun's reference client does: its audio
encoder takes at most 30 s per clip.

Reasoning models: when `reasoning_prefix` is set (the chat template opens a `<think>` block that the
server does not echo), the prefix is put back in front of every free-generation answer, so a cut-off
reasoning block stays recognisable and is scored as unparsed rather than mined for a stray letter.
Reasoning blocks are removed from earlier assistant turns before they are sent back as history.
"""

import base64
import io
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import soundfile as sf
from loguru import logger

from speech_processing.adapters.base import BaseGenerationAdapter
from speech_processing.config.core import GenerationParams
from speech_processing.evaluation.answer_parsing import strip_thinking
from speech_processing.utils.audio import FILE_URL_PREFIX

MAX_AUDIO_CHUNK_S = 25.0
MAX_CONCURRENT_REQUESTS = 64
REQUEST_TIMEOUT_S = 1800
MAX_ATTEMPTS = 3
RETRY_DELAY_S = 5.0
ASSISTANT = "assistant"


def audio_file_to_input_parts(path: str) -> list[dict]:
    """A local audio file as base64 WAV input_audio parts of at most MAX_AUDIO_CHUNK_S seconds each."""
    waveform, sample_rate = sf.read(path, always_2d=False)
    chunk_len = int(MAX_AUDIO_CHUNK_S * sample_rate)
    parts = []
    for start in range(0, max(len(waveform), 1), chunk_len):
        buffer = io.BytesIO()
        sf.write(buffer, waveform[start : start + chunk_len], sample_rate, format="WAV")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        parts.append({"type": "input_audio", "input_audio": {"data": data, "format": "wav"}})
    return parts


def _convert_part(part: dict) -> list[dict]:
    if part.get("type") != "audio_url":
        return [part]
    url = part["audio_url"]["url"]
    if not url.startswith(FILE_URL_PREFIX):
        return [part]
    return audio_file_to_input_parts(url[len(FILE_URL_PREFIX) :])


def to_server_messages(conversation: list[dict]) -> list[dict]:
    """Inline the audio, and drop reasoning blocks from earlier assistant turns."""
    messages = []
    for index, message in enumerate(conversation):
        content = message["content"]
        is_history = message["role"] == ASSISTANT and index < len(conversation) - 1
        if isinstance(content, list):
            content = [converted for part in content for converted in _convert_part(part)]
        elif is_history and isinstance(content, str):
            content = strip_thinking(content)
        messages.append({"role": message["role"], "content": content})
    return messages


class OpenAIChatServerAdapter(BaseGenerationAdapter):
    def __init__(
        self,
        base_url: str,
        model: str,
        reasoning_prefix: str | None = None,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
        stop_token_ids: list[int] | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.reasoning_prefix = reasoning_prefix
        self.max_concurrent = max_concurrent
        self.stop_token_ids = stop_token_ids
        # No local tokenizer: contrastive letter scoring needs prompt log-probs and is not supported here.
        self.tokenizer = None

    def generate_batch(self, prompts: list[Any], sampling_params: GenerationParams) -> list[str]:
        with ThreadPoolExecutor(max_workers=min(self.max_concurrent, max(len(prompts), 1))) as pool:
            return list(pool.map(lambda conversation: self._generate_one(conversation, sampling_params), prompts))

    def _payload(self, conversation: list[dict], sampling_params: GenerationParams) -> dict:
        prefilled = conversation[-1]["role"] == ASSISTANT
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_server_messages(conversation),
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_tokens": sampling_params.max_new_tokens,
            "skip_special_tokens": False,
            "add_generation_prompt": not prefilled,
            "continue_final_message": prefilled,
        }
        if sampling_params.stop:
            payload["stop"] = sampling_params.stop
        if self.stop_token_ids:
            payload["stop_token_ids"] = self.stop_token_ids
        return payload

    def _generate_one(self, conversation: list[dict], sampling_params: GenerationParams) -> str:
        payload = json.dumps(self._payload(conversation, sampling_params)).encode()
        request = urllib.request.Request(self.url, data=payload, headers={"Content-Type": "application/json"})
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                    text = json.loads(response.read())["choices"][0]["message"]["content"] or ""
                break
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as e:
                if attempt == MAX_ATTEMPTS:
                    raise RuntimeError(f"Chat server request failed after {MAX_ATTEMPTS} attempts: {e}") from e
                logger.warning(f"Chat server request failed (attempt {attempt}/{MAX_ATTEMPTS}): {e}")
                time.sleep(RETRY_DELAY_S)

        prefilled = conversation[-1]["role"] == ASSISTANT
        if self.reasoning_prefix and not prefilled and not text.lstrip().startswith(self.reasoning_prefix.strip()):
            text = self.reasoning_prefix + text
        return text
