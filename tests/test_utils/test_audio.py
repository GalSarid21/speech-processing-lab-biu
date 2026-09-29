import numpy as np
import pytest
import soundfile as sf
from assertpy import assert_that

from speech_processing.utils import audio as audio_utils
from speech_processing.utils.audio import crop_audio, force_mono_audio, get_duration_s, write_silence

TARGET_SR = 16_000


@pytest.fixture
def stereo_wav(tmp_path):
    sample_rate = 44_100
    samples = np.zeros((sample_rate * 2, 2), dtype=np.float32)
    samples[:, 0] = 0.5
    path = tmp_path / "stereo.wav"
    sf.write(path, samples, sample_rate)
    return str(path)


def test_force_mono_audio_converts_and_caches(monkeypatch, tmp_path, stereo_wav):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(audio_utils, "MONO_AUDIO_CACHE_DIR", str(cache_dir))

    url = force_mono_audio(stereo_wav, TARGET_SR)
    assert_that(url).starts_with("file://")

    cached = url.removeprefix("file://")
    info = sf.info(cached)
    assert_that(info.channels).is_equal_to(1)
    assert_that(info.samplerate).is_equal_to(TARGET_SR)

    mtime = cache_dir.joinpath(cached.rsplit("/", 1)[-1]).stat().st_mtime_ns
    assert_that(force_mono_audio(stereo_wav, TARGET_SR)).is_equal_to(url)
    assert_that(cache_dir.joinpath(cached.rsplit("/", 1)[-1]).stat().st_mtime_ns).is_equal_to(mtime)


def test_remote_urls_pass_through():
    assert_that(force_mono_audio("http://example.com/a.wav", TARGET_SR)).is_equal_to("http://example.com/a.wav")


def test_write_silence(tmp_path):
    path = write_silence(1.5, TARGET_SR, str(tmp_path))
    samples, sample_rate = sf.read(path)

    assert_that(sample_rate).is_equal_to(TARGET_SR)
    assert_that(len(samples)).is_equal_to(int(1.5 * TARGET_SR))
    assert_that(bool(np.all(samples == 0))).is_true()


@pytest.mark.parametrize(
    "start_s, end_s, expected_duration",
    [(1.0, 2.5, 1.5), (-1.0, 1.0, 1.0), (2.0, 99.0, 1.0)],
)
def test_crop_audio_clamps(tmp_path, wav_factory, start_s, end_s, expected_duration):
    clip = wav_factory(3.0, "clip.wav")
    cropped = crop_audio(clip, start_s, end_s, str(tmp_path))
    assert_that(get_duration_s(cropped)).is_close_to(expected_duration, 0.01)
