import os
import tempfile

import pytest
from assertpy import assert_that

from speech_processing.config.core import AudioModelConfig
from speech_processing.data.dtos import AudioRequest, AudioResponse, FewShotTurn, ItemMetadata
from speech_processing.models import audio as audio_module
from speech_processing.models.audio import MULTI_TURN_ERROR_MARKER, VoxtralAudioEngine
from speech_processing.models.contrastive import ContrastiveChoiceScorer
from speech_processing.models.presentation import TwoPassLocalizationPresentation
from speech_processing.utils import audio as audio_utils
from speech_processing.utils.exceptions import AudioProcessingError

TARGET_SR = 16_000
SYSTEM_PROMPT = "SYSTEM."


@pytest.fixture(autouse=True)
def mono_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_utils, "MONO_AUDIO_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def audio_config():
    return AudioModelConfig(
        model_id="mock/voxtral",
        dtype="bfloat16",
        max_num_seqs=4,
        max_new_tokens=64,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        temperature=0.5,
        top_p=0.5,
        target_sr=TARGET_SR,
    )


@pytest.fixture
def engine_factory(mocker, audio_config):
    def _build(outputs_per_call: list[list[str]] | None = None, side_effect=None, **kwargs):
        adapter_class = mocker.patch("speech_processing.models.audio.VLLMAdapter")
        adapter = adapter_class.return_value
        if side_effect is not None:
            adapter.generate_batch.side_effect = side_effect
        else:
            adapter.generate_batch.side_effect = outputs_per_call or [["answer"]]

        engine = VoxtralAudioEngine(audio_config, **kwargs)
        return engine, adapter

    return _build


def sent_prompts(adapter) -> list[list[list[dict]]]:
    return [call.kwargs["prompts"] for call in adapter.generate_batch.call_args_list]


def texts_of(message) -> list[str]:
    return [part["text"] for part in message["content"] if part["type"] == "text"]


def test_full_clip_conversation_carries_audio_then_text(engine_factory, wav_factory):
    engine, adapter = engine_factory()
    req = AudioRequest(instruction="Answer.", audio_path=wav_factory(1.0, "a.wav"), system_prompt=SYSTEM_PROMPT)

    engine.batch_infer([req])

    user_message = sent_prompts(adapter)[0][0][0]
    assert_that([part["type"] for part in user_message["content"]]).is_equal_to(["audio_url", "text"])
    assert_that(user_message["content"][1]["text"]).starts_with(SYSTEM_PROMPT)


def test_only_the_first_few_shot_turn_carries_the_system_prompt(engine_factory, wav_factory):
    engine, adapter = engine_factory()
    shot_audio = wav_factory(1.0, "shot.wav")
    req = AudioRequest(
        instruction="Answer.",
        audio_path=wav_factory(1.0, "target.wav"),
        system_prompt=SYSTEM_PROMPT,
        few_shot_turns=[
            FewShotTurn(audio_path=shot_audio, user_text="shot one", assistant_text="A"),
            FewShotTurn(audio_path=shot_audio, user_text="shot two", assistant_text="B"),
        ],
    )

    engine.batch_infer([req])

    conversation = sent_prompts(adapter)[0][0]
    assert_that(texts_of(conversation[0])[0]).is_equal_to(f"{SYSTEM_PROMPT}\n\nshot one")
    assert_that(texts_of(conversation[2])[0]).is_equal_to("shot two")
    assert_that(texts_of(conversation[4])[0]).is_equal_to("Answer.")


def test_prefill_is_sent_and_reinjected_once(engine_factory, wav_factory):
    engine, adapter = engine_factory(outputs_per_call=[["rest of the analysis"]])
    req = AudioRequest(instruction="Answer.", audio_path=wav_factory(1.0, "a.wav"), assistant_prefill="<analysis>\n")

    responses = engine.batch_infer([req])

    conversation = sent_prompts(adapter)[0][0]
    assert_that(conversation[-1]).is_equal_to({"role": "assistant", "content": "<analysis>\n"})
    assert_that(responses[0].generated_text).is_equal_to("<analysis>\nrest of the analysis")
    assert_that(responses[0].generated_text.count("<analysis>")).is_equal_to(1)


def test_multi_turn_accumulates_and_keeps_the_final_turn(engine_factory, wav_factory):
    engine, adapter = engine_factory(outputs_per_call=[["first reply"], ["second reply"]])
    req = AudioRequest(instruction=["Describe the audio.", "Now answer."], audio_path=wav_factory(1.0, "a.wav"))

    responses = engine.batch_infer([req])

    second_call_last_user = sent_prompts(adapter)[1][0][-1]
    assert_that([part["type"] for part in second_call_last_user["content"]]).is_equal_to(["text"])
    assert_that(responses[0].final_turn_text).is_equal_to("second reply")
    assert_that(responses[0].generated_text).contains("Turn 1")
    assert_that(responses[0].generated_text).contains("Turn 2")


def test_two_pass_crops_on_the_second_turn(engine_factory, audio_config, wav_factory):
    engine, adapter = engine_factory(
        outputs_per_call=[['{"start":1,"end":2}'], ["B"]],
        presentation=TwoPassLocalizationPresentation(audio_config.target_sr),
    )
    req = AudioRequest(instruction=["Locate it.", "Now answer."], audio_path=wav_factory(5.0, "a.wav"))

    engine.batch_infer([req])

    second_call_last_user = sent_prompts(adapter)[1][0][-1]
    assert_that(texts_of(second_call_last_user)[0]).contains("Cropped segment")


def test_too_many_audio_parts_raises(engine_factory, wav_factory):
    engine, _ = engine_factory()
    shot_audio = wav_factory(0.5, "shot.wav")
    req = AudioRequest(
        instruction="Answer.",
        audio_path=wav_factory(0.5, "target.wav"),
        few_shot_turns=[
            FewShotTurn(audio_path=[shot_audio] * 4, user_text="shot", assistant_text="A"),
            FewShotTurn(audio_path=[shot_audio] * 4, user_text="shot", assistant_text="B"),
        ],
    )

    assert_that(engine.batch_infer).raises(AudioProcessingError).when_called_with([req])


def test_generation_failure_yields_the_error_marker(engine_factory, wav_factory):
    engine, _ = engine_factory(side_effect=RuntimeError("cuda oom"))
    req = AudioRequest(instruction="Answer.", audio_path=wav_factory(1.0, "a.wav"))

    responses = engine.batch_infer([req])

    assert_that(responses[0].generated_text).is_equal_to(MULTI_TURN_ERROR_MARKER.strip())


def test_injected_scorer_bypasses_generation(engine_factory, audio_config, wav_factory):
    class StubScorer(ContrastiveChoiceScorer):
        def score(self, backend, requests):
            return [
                AudioResponse(
                    sample_id=req.audio_path,
                    instruction=req.instruction,
                    generated_text="A",
                    final_turn_text="A",
                    metadata=req.metadata,
                )
                for req in requests
            ]

    engine, adapter = engine_factory(scorer=StubScorer(0.0, audio_config.target_sr, 2))
    req = AudioRequest(
        instruction="Answer.",
        audio_path=wav_factory(1.0, "a.wav"),
        metadata=ItemMetadata(item_id="ID1", question="q", choices=["a", "b"]),
    )

    assert_that(engine.batch_infer([req])[0].generated_text).is_equal_to("A")
    adapter.generate_batch.assert_not_called()


def test_temp_files_are_removed_after_batch_infer(monkeypatch, engine_factory, wav_factory):
    created: list[str] = []
    real_temporary_directory = tempfile.TemporaryDirectory

    def tracking_temporary_directory(*args, **kwargs):
        handle = real_temporary_directory(*args, **kwargs)
        created.append(handle.name)
        return handle

    monkeypatch.setattr(audio_module.tempfile, "TemporaryDirectory", tracking_temporary_directory)

    engine, _ = engine_factory()
    engine.batch_infer([AudioRequest(instruction="Answer.", audio_path=wav_factory(1.0, "a.wav"))])

    assert_that(created).is_length(1)
    assert_that(os.path.exists(created[0])).is_false()
