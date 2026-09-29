"""Synthetic-signal tests for the A1 front end.

These are the only guard the detectors have before they are pointed at real recordings: if a tone
burst does not read as one wheeze and a train of clicks does not read as crackles, no downstream
experiment means anything.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from assertpy import assert_that

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "extract_respiratory_features.py"

TONE_HZ = 400.0
TONE_DURATION_S = 0.3
CLICK_DURATION_S = 0.005
CLICK_SPACING_S = 0.06
NUM_CLICKS = 5
BREATHING_HZ = 0.3
EXPECTED_RATE_PER_MIN = 18.0
CLIP_DURATION_S = 20.0
SEED = 0


@pytest.fixture(scope="module")
def features():
    spec = importlib.util.spec_from_file_location("extract_respiratory_features", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sample_rate(features):
    return features.TARGET_SR


def _silence(duration_s: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


def test_tone_burst_reads_as_one_wheeze(features, sample_rate):
    signal = _silence(2.0, sample_rate)
    t = np.arange(int(TONE_DURATION_S * sample_rate)) / sample_rate
    start = int(0.5 * sample_rate)
    signal[start : start + t.size] = 0.5 * np.sin(2 * np.pi * TONE_HZ * t)

    events = features.detect_wheezes(features.preprocess(signal, sample_rate), sample_rate)

    assert_that(events).is_length(1)
    assert_that(events[0].peak_hz[0]).is_close_to(TONE_HZ, 30.0)
    assert_that(events[0].end_s - events[0].start_s).is_greater_than_or_equal_to(features.MIN_WHEEZE_S)
    assert_that(events[0].kind).is_equal_to(features.MONOPHONIC)


def test_two_simultaneous_tones_read_as_polyphonic(features, sample_rate):
    signal = _silence(2.0, sample_rate)
    t = np.arange(int(TONE_DURATION_S * sample_rate)) / sample_rate
    start = int(0.5 * sample_rate)
    signal[start : start + t.size] = 0.4 * np.sin(2 * np.pi * 350 * t) + 0.4 * np.sin(2 * np.pi * 620 * t)

    events = features.detect_wheezes(features.preprocess(signal, sample_rate), sample_rate)

    assert_that(events).is_length(1)
    assert_that(events[0].kind).is_equal_to(features.POLYPHONIC)
    assert_that(events[0].peak_hz).is_length(2)


def test_click_train_reads_as_late_crackles(features, sample_rate):
    rng = np.random.default_rng(SEED)
    signal = (0.001 * rng.standard_normal(int(3.0 * sample_rate))).astype(np.float32)
    click_samples = int(CLICK_DURATION_S * sample_rate)
    first_click_s = 2.2

    for index in range(NUM_CLICKS):
        start = int((first_click_s + index * CLICK_SPACING_S) * sample_rate)
        signal[start : start + click_samples] += 0.6 * np.hanning(click_samples)

    events = features.detect_crackles(features.preprocess(signal, sample_rate), sample_rate)

    assert_that(events).is_length(NUM_CLICKS)
    assert_that([round(event.start_s, 1) for event in events]).is_equal_to([2.2, 2.3, 2.3, 2.4, 2.4])
    assert_that({event.kind for event in events}).is_equal_to({features.FINE})

    # The clicks sit in the last third of a cycle spanning 0-3 s.
    positions = {features.position_in_cycle(event.start_s, (0.0, 3.0)) for event in events}
    assert_that(positions).is_equal_to({features.LATE})


@pytest.mark.parametrize("detector", ["detect_wheezes", "detect_crackles"])
def test_stationary_noise_produces_no_events(features, sample_rate, detector):
    rng = np.random.default_rng(SEED)
    noise = (0.1 * rng.standard_normal(int(5.0 * sample_rate))).astype(np.float32)

    events = getattr(features, detector)(features.preprocess(noise, sample_rate), sample_rate)

    assert_that(events).is_empty()


def test_amplitude_modulated_envelope_gives_the_right_rate(features, sample_rate):
    rng = np.random.default_rng(SEED)
    t = np.arange(int(CLIP_DURATION_S * sample_rate)) / sample_rate
    envelope = (1 - np.cos(2 * np.pi * BREATHING_HZ * t)) / 2
    signal = (0.3 * envelope * rng.standard_normal(t.size)).astype(np.float32)

    cycles = features.detect_cycles(features.preprocess(signal, sample_rate), sample_rate)
    rate = features.respiratory_rate_per_min(cycles, CLIP_DURATION_S)

    assert_that(cycles).is_not_empty()
    assert_that(rate).is_close_to(EXPECTED_RATE_PER_MIN, 1.0)
    assert_that(all(end > start for start, end in cycles)).is_true()


@pytest.mark.parametrize("time_s, expected", [(0.5, "early"), (1.5, "middle"), (2.5, "late"), (2.99, "late")])
def test_position_in_cycle(features, time_s, expected):
    assert_that(features.position_in_cycle(time_s, (0.0, 3.0))).is_equal_to(expected)


def test_extract_item_produces_the_loader_schema(features, sample_rate):
    rng = np.random.default_rng(SEED)
    t = np.arange(int(CLIP_DURATION_S * sample_rate)) / sample_rate
    envelope = (1 - np.cos(2 * np.pi * BREATHING_HZ * t)) / 2
    signal = (0.3 * envelope * rng.standard_normal(t.size)).astype(np.float32)

    evidence = features.extract_item("audio1", signal, sample_rate)

    assert_that(evidence.item_id).is_equal_to("audio1")
    assert_that(evidence.cycles_summary).contains("Respiratory rate ~18/min")
    assert_that(evidence.cycle_spans).is_not_empty()
    assert_that(set(evidence.model_dump())).is_equal_to(
        {"item_id", "cycles_summary", "adventitious_summary", "cycle_spans"}
    )


def test_silent_clip_degrades_gracefully(features, sample_rate):
    evidence = features.extract_item("silent", _silence(5.0, sample_rate), sample_rate)

    assert_that(evidence.cycle_spans).is_empty()
    assert_that(evidence.adventitious_summary).is_equal_to(features.NO_EVENTS_TEXT)
