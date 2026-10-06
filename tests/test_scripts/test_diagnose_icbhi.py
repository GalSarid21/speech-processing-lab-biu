"""Guards the near-duplicate detector.

Its first version compared clip-averaged MFCC/chroma fingerprints, under which every pair of lung
recordings looked identical: on the real data it flagged 170 of 174 clips, which would have held
most of the dataset out of evaluation.
"""

import importlib.util
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from assertpy import assert_that

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_icbhi.py"
SOURCE_SR = 44_100
CLIP_S = 20.0


@pytest.fixture(scope="module")
def diag():
    spec = importlib.util.spec_from_file_location("diagnose_icbhi", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recording(seed: int, seconds: float) -> np.ndarray:
    """A breathing-like recording: low-passed noise under a slow envelope, unique per seed."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SOURCE_SR * seconds)) / SOURCE_SR
    envelope = (1 - np.cos(2 * np.pi * rng.uniform(0.2, 0.4) * t + rng.uniform(0, 6))) / 2
    noise = np.convolve(rng.standard_normal(t.size), np.ones(40) / 40, mode="same")
    return (0.3 * envelope * noise).astype(np.float32)


def wav_bytes(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, SOURCE_SR, format="WAV")
    return buffer.getvalue()


def clip(samples: np.ndarray, start_sample: int) -> bytes:
    return wav_bytes(samples[start_sample : start_sample + int(CLIP_S * SOURCE_SR)])


@pytest.fixture(scope="module")
def long_recording():
    return recording(seed=1, seconds=40.0)


def min_overlap(diag) -> int:
    return int(diag.MIN_SHARED_AUDIO_S * diag.DUPLICATE_SR)


def test_a_clip_matches_itself(diag, long_recording):
    signal = diag.load_clip_signal(clip(long_recording, 0))
    assert_that(diag.max_overlap_correlation(signal, signal, min_overlap(diag))).is_close_to(1.0, 1e-6)


def test_overlapping_cuts_of_one_recording_are_near_duplicates(diag, long_recording):
    """Cut at an offset that is not a whole number of 2 kHz samples, as real clip boundaries would be."""
    offset = int(10.3 * SOURCE_SR) + 7
    a = diag.load_clip_signal(clip(long_recording, 0))
    b = diag.load_clip_signal(clip(long_recording, offset))

    assert_that(diag.max_overlap_correlation(a, b, min_overlap(diag))).is_greater_than(diag.NEAR_DUPLICATE_THRESHOLD)


@pytest.mark.parametrize("seed_a, seed_b", [(2, 3), (4, 5), (6, 7)])
def test_independent_recordings_are_not_near_duplicates(diag, seed_a, seed_b):
    """The case the old fingerprint got wrong: it scored independent recordings at cosine >= 0.9996."""
    a = diag.load_clip_signal(wav_bytes(recording(seed_a, CLIP_S)))
    b = diag.load_clip_signal(wav_bytes(recording(seed_b, CLIP_S)))

    assert_that(diag.max_overlap_correlation(a, b, min_overlap(diag))).is_less_than(0.2)


def test_a_shared_stretch_shorter_than_the_minimum_does_not_count(diag, long_recording):
    """Clips that only touch at the edges share too little audio to call them one recording."""
    a = diag.load_clip_signal(clip(long_recording, 0))
    b = diag.load_clip_signal(clip(long_recording, int((CLIP_S - 2.0) * SOURCE_SR)))  # 2 s shared

    assert_that(diag.max_overlap_correlation(a, b, min_overlap(diag))).is_less_than(diag.NEAR_DUPLICATE_THRESHOLD)


def test_find_near_duplicates_pairs_only_the_shared_clips(diag, long_recording):
    df = pd.DataFrame(
        {
            "item_id": ["audio1", "audio2", "audio3"],
            "audio": [
                {"bytes": clip(long_recording, 0)},
                {"bytes": clip(long_recording, int(8.0 * SOURCE_SR) + 3)},
                {"bytes": wav_bytes(recording(seed=99, seconds=CLIP_S))},
            ],
        }
    )

    assert_that(diag.find_near_duplicates(df, diag.NEAR_DUPLICATE_THRESHOLD)).is_equal_to(
        {"audio1": ["audio2"], "audio2": ["audio1"]}
    )


def test_one_pair_is_fast_enough_for_the_full_dataset(diag):
    """174 clips are ~15k pairs; at the observed per-pair cost that must stay a few minutes at most."""
    a = diag.load_clip_signal(wav_bytes(recording(10, CLIP_S)))
    b = diag.load_clip_signal(wav_bytes(recording(11, CLIP_S)))

    started = time.perf_counter()
    for _ in range(5):
        diag.max_overlap_correlation(a, b, min_overlap(diag))
    per_pair_s = (time.perf_counter() - started) / 5

    assert_that(per_pair_s * 174 * 173 / 2).is_less_than(300)
