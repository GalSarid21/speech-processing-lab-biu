import pytest
from assertpy import assert_that

from speech_processing.models.presentation import (
    ChunkedPresentation,
    CyclePresentation,
    FullClipPresentation,
    TwoPassLocalizationPresentation,
    parse_time_span,
)
from speech_processing.utils import audio as audio_utils
from speech_processing.utils.audio import chunk_audio
from speech_processing.utils.consts import CHUNK_LENGTH_S, MAX_AUDIO_CHUNKS, VOXTRAL_MAX_AUDIOS_PER_PROMPT

TARGET_SR = 16_000


@pytest.fixture(autouse=True)
def mono_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_utils, "MONO_AUDIO_CACHE_DIR", str(tmp_path / "cache"))


@pytest.mark.parametrize(
    "duration_s, expected_chunks", [(3.0, 1), (12.0, 3), (30.0, MAX_AUDIO_CHUNKS), (61.0, MAX_AUDIO_CHUNKS)]
)
def test_chunk_audio_covers_the_whole_clip(tmp_path, wav_factory, duration_s, expected_chunks):
    clip = wav_factory(duration_s, f"clip_{duration_s}.wav")
    chunks = chunk_audio(clip, CHUNK_LENGTH_S, MAX_AUDIO_CHUNKS, str(tmp_path))

    assert_that(chunks).is_length(expected_chunks)
    assert_that(chunks[0][1]).is_close_to(0.0, 0.001)
    assert_that(chunks[-1][2]).is_close_to(duration_s, 0.01)
    assert_that(sum(end - start for _, start, end in chunks)).is_close_to(duration_s, 0.01)


def test_full_clip_content(tmp_path, wav_factory):
    content = FullClipPresentation(TARGET_SR).first_turn(wav_factory(2.0, "full.wav"), "instruction", str(tmp_path))
    assert_that([part["type"] for part in content]).is_equal_to(["audio_url", "text"])
    assert_that(content[1]["text"]).is_equal_to("instruction")


def test_chunked_labels(tmp_path, wav_factory):
    content = ChunkedPresentation(TARGET_SR).first_turn(wav_factory(12.0, "chunked.wav"), "instruction", str(tmp_path))
    texts = [part["text"] for part in content if part["type"] == "text"]

    assert_that(texts[0]).is_equal_to("Segment 1 (0.0-4.0 s):")
    assert_that(texts).contains("Full audio:")
    assert_that(texts[-1]).is_equal_to("instruction")


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"start": 1.0, "end": 2.0}', (1.0, 2.0)),
        ('{"start": 1.0, "end": 99.0}', (1.0, 3.0)),
        ('{"start": 2.0, "end": 1.0}', None),
        ('{"from": 1.0, "to": 2.0}', None),
        ("no json here", None),
        ('{"start": "abc", "end": 2.0}', None),
    ],
)
def test_parse_time_span(text, expected):
    assert_that(parse_time_span(text, 3.0)).is_equal_to(expected)


def test_two_pass_crops_with_padding(tmp_path, wav_factory):
    presentation = TwoPassLocalizationPresentation(TARGET_SR)
    content = presentation.followup_turn(
        wav_factory(10.0, "two_pass.wav"), "answer now", '{"start": 2.0, "end": 4.0}', str(tmp_path)
    )

    assert_that(content[0]["text"]).is_equal_to("Cropped segment (1.5-4.5 s):")
    assert_that(content[1]["type"]).is_equal_to("audio_url")
    assert_that(content[2]["text"]).is_equal_to("answer now")
    assert_that(presentation.num_fallbacks).is_equal_to(0)


def test_two_pass_falls_back_to_text_only(tmp_path, wav_factory):
    presentation = TwoPassLocalizationPresentation(TARGET_SR)
    content = presentation.followup_turn(
        wav_factory(10.0, "two_pass.wav"), "answer now", "I could not find it", str(tmp_path)
    )

    assert_that(content).is_length(1)
    assert_that(content[0]["text"]).is_equal_to("answer now")
    assert_that(presentation.num_fallbacks).is_equal_to(1)


@pytest.mark.parametrize("duration_s", [12.0, 30.0, 61.0, 120.0])
def test_chunked_layout_never_exceeds_the_voxtral_audio_limit(tmp_path, wav_factory, duration_s):
    """vLLM rejects a 6th clip in one Voxtral prompt; the chunks plus the full clip must fit."""
    content = ChunkedPresentation(TARGET_SR).first_turn(
        wav_factory(duration_s, f"long_{duration_s}.wav"), "instruction", str(tmp_path)
    )
    audios = [part for part in content if part["type"] == "audio_url"]
    assert_that(len(audios)).is_less_than_or_equal_to(VOXTRAL_MAX_AUDIOS_PER_PROMPT)


def test_cycle_layout_never_exceeds_the_voxtral_audio_limit(tmp_path, wav_factory):
    clip = wav_factory(20.0, "cycles.wav")
    many_spans = [(float(i), float(i) + 1.0) for i in range(12)]

    content = CyclePresentation(TARGET_SR, {clip: many_spans}).first_turn(clip, "instruction", str(tmp_path))

    audios = [part for part in content if part["type"] == "audio_url"]
    assert_that(len(audios)).is_equal_to(VOXTRAL_MAX_AUDIOS_PER_PROMPT)
