# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "datasets",
#     "librosa",
#     "loguru",
#     "numpy",
#     "pydantic",
#     "scipy",
#     "soundfile",
# ]
# ///
"""A1: measures respiratory features from ICBHI recordings and renders them as text.

The hypothesis this serves: the language model already knows that polyphonic expiratory wheezes
suggest obstructive disease and that late inspiratory crackles suggest pneumonia. What it lacks is
a way to turn auscultation audio into those words. This script is that front end.

Everything here is deterministic signal processing on the waveform. It never sees a label.

Usage:
    uv run scripts/extract_respiratory_features.py --output-file data/icbhi_respiratory_features.jsonl
"""

import argparse
import io
import itertools
import os
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from loguru import logger
from pydantic import BaseModel
from scipy.signal import butter, find_peaks, medfilt, peak_prominences, peak_widths, sosfiltfilt, stft

DATASET_ID = "DynamicSuperb/RespiratorySoundClassification_ICBHI2017"
DATASET_SPLIT = "test"
DEFAULT_OUTPUT_FILE = "data/icbhi_respiratory_features.jsonl"

# Lung sounds live below ~2 kHz; 4 kHz keeps all of it and makes the transient detectors cheap.
TARGET_SR = 4_000
BANDPASS_LOW_HZ = 100.0
BANDPASS_HIGH_HZ = 1_800.0
BANDPASS_ORDER = 4

ENVELOPE_FRAME_S = 0.050
ENVELOPE_SMOOTH_S = 0.500
MIN_CYCLE_S = 1.2
# A minimum must dip by at least this fraction of the envelope range to count as a cycle boundary;
# without it, ripple in the envelope of a noisy recording doubles the measured respiratory rate.
CYCLE_MINIMUM_PROMINENCE = 0.10
SECONDS_PER_MINUTE = 60.0

STFT_WINDOW_S = 0.064
WHEEZE_FLATNESS_MAX = 0.3
WHEEZE_PEAK_MIN_HZ = 100.0
WHEEZE_PEAK_MAX_HZ = 1_000.0
MIN_WHEEZE_S = 0.100
WHEEZE_ENERGY_GATE = 0.10
POLYPHONIC_PEAK_PROMINENCE = 0.25
MONOPHONIC = "monophonic"
POLYPHONIC = "polyphonic"

# A crackle has to clear three independent bars, because any one of them alone misfires:
#   local   - it must stand above the running median of its own neighbourhood;
#   global  - it must be an outlier of the whole clip (this is what keeps stationary noise at zero);
#   shape   - it must be a prominent, narrow peak, not a plateau of sustained tonal energy.
TEAGER_LOCAL_MULTIPLE = 12.0
TEAGER_GLOBAL_MAD_MULTIPLE = 15.0
TEAGER_FLOOR_FRACTION = 0.02
CRACKLE_PROMINENCE_FRACTION = 0.5
MAD_TO_SIGMA = 0.6745
TEAGER_MEDIAN_WINDOW_S = 0.100
MAX_CRACKLE_S = 0.020
MIN_CRACKLE_SEPARATION_S = 0.010
CRACKLE_WIDTH_REL_HEIGHT = 0.5
FINE_CRACKLE_MAX_S = 0.010
FINE = "fine"
COARSE = "coarse"
EARLY_THIRD = 1 / 3
LATE_THIRD = 2 / 3
EARLY = "early"
MIDDLE = "middle"
LATE = "late"

EPSILON = 1e-12
NO_EVENTS_TEXT = "No wheeze-like or crackle-like events detected."


class RespiratoryEvidence(BaseModel):
    """Matches speech_processing.data.icbhi.RespiratoryEvidence exactly."""

    item_id: str
    cycles_summary: str
    adventitious_summary: str
    cycle_spans: list[tuple[float, float]] = []


@dataclass(frozen=True)
class WheezeEvent:
    start_s: float
    end_s: float
    peak_hz: list[float]

    @property
    def kind(self) -> str:
        return POLYPHONIC if len(self.peak_hz) > 1 else MONOPHONIC


@dataclass(frozen=True)
class CrackleEvent:
    start_s: float
    end_s: float

    @property
    def kind(self) -> str:
        return FINE if (self.end_s - self.start_s) < FINE_CRACKLE_MAX_S else COARSE


def preprocess(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Band-pass away heart sounds (below ~100 Hz) and hiss (above ~1.8 kHz)."""
    nyquist = sample_rate / 2
    high = min(BANDPASS_HIGH_HZ, nyquist * 0.99)
    if BANDPASS_LOW_HZ >= high:
        return waveform
    sos = butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, waveform)


def rms_envelope(waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    """Smoothed RMS envelope plus the hop it was computed at, in seconds."""
    frame_length = max(2, int(ENVELOPE_FRAME_S * sample_rate))
    envelope = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=frame_length)[0]

    smooth_frames = max(1, int(ENVELOPE_SMOOTH_S / ENVELOPE_FRAME_S))
    kernel = np.ones(smooth_frames) / smooth_frames
    smoothed = np.convolve(envelope, kernel, mode="same")
    return smoothed, frame_length / sample_rate


def detect_cycles(waveform: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
    """Breathing cycles as the intervals between envelope minima at least MIN_CYCLE_S apart."""
    envelope, hop_s = rms_envelope(waveform, sample_rate)
    if envelope.size < 3:
        return []

    envelope_range = float(envelope.max() - envelope.min())
    minima, _ = find_peaks(
        -envelope,
        distance=max(1, int(MIN_CYCLE_S / hop_s)),
        prominence=CYCLE_MINIMUM_PROMINENCE * envelope_range if envelope_range > 0 else None,
    )
    return [(float(start * hop_s), float(end * hop_s)) for start, end in itertools.pairwise(minima)]


def respiratory_rate_per_min(cycles: list[tuple[float, float]], duration_s: float) -> float:
    """Extrapolated rate. Measured over the span the cycles actually cover, not the whole clip."""
    if not cycles:
        return 0.0
    span = cycles[-1][1] - cycles[0][0]
    if span <= 0:
        return 0.0
    return SECONDS_PER_MINUTE * len(cycles) / span


def inspiration_expiration_ratio(
    waveform: np.ndarray, sample_rate: int, cycles: list[tuple[float, float]]
) -> float | None:
    """Median inspiration:expiration duration ratio, taking the envelope peak as the turning point."""
    ratios = []
    for start_s, end_s in cycles:
        segment = np.abs(waveform[int(start_s * sample_rate) : int(end_s * sample_rate)])
        if segment.size < 3:
            continue
        peak_index = int(np.argmax(segment))
        inspiration = peak_index / sample_rate
        expiration = (segment.size - peak_index) / sample_rate
        if inspiration > 0 and expiration > 0:
            ratios.append(inspiration / expiration)
    return float(np.median(ratios)) if ratios else None


def detect_wheezes(waveform: np.ndarray, sample_rate: int) -> list[WheezeEvent]:
    """Frames that are tonal (low spectral flatness) with a dominant peak in the wheeze band.

    A wheeze must last at least MIN_WHEEZE_S, which is the clinical minimum and also what keeps
    isolated tonal frames of noise out.
    """
    window = max(8, int(STFT_WINDOW_S * sample_rate))
    frequencies, times, spectrogram = stft(waveform, fs=sample_rate, nperseg=window, noverlap=window // 2)
    magnitude = np.abs(spectrogram)
    if magnitude.shape[1] < 2:
        return []

    frame_energy = magnitude.sum(axis=0)
    energy_gate = frame_energy.max() * WHEEZE_ENERGY_GATE

    band = (frequencies >= WHEEZE_PEAK_MIN_HZ) & (frequencies <= WHEEZE_PEAK_MAX_HZ)
    tonal_flags = np.zeros(magnitude.shape[1], dtype=bool)
    peaks_per_frame: list[list[float]] = []

    for index in range(magnitude.shape[1]):
        column = magnitude[:, index]
        peaks_per_frame.append([])
        if frame_energy[index] < energy_gate:
            continue

        flatness = float(np.exp(np.mean(np.log(column + EPSILON))) / (np.mean(column) + EPSILON))
        if flatness >= WHEEZE_FLATNESS_MAX:
            continue

        band_column = column * band
        if band_column.max() <= 0 or band_column.argmax() == 0:
            continue

        tonal_flags[index] = True
        peak_indices, _ = find_peaks(band_column, prominence=band_column.max() * POLYPHONIC_PEAK_PROMINENCE)
        peaks_per_frame[index] = sorted(float(frequencies[i]) for i in peak_indices) or [
            float(frequencies[int(band_column.argmax())])
        ]

    events: list[WheezeEvent] = []
    for start, end in _flag_runs(tonal_flags):
        start_s, end_s = float(times[start]), float(times[end])
        if end_s - start_s < MIN_WHEEZE_S:
            continue
        run_peaks = [peak for index in range(start, end + 1) for peak in peaks_per_frame[index]]
        events.append(WheezeEvent(start_s, end_s, _cluster_peaks(run_peaks)))
    return events


def detect_crackles(waveform: np.ndarray, sample_rate: int) -> list[CrackleEvent]:
    """Short, isolated, prominent transients in the Teager-Kaiser energy.

    Stationary noise produces no crackles here by construction: its energy has no outliers, so the
    global MAD bar is never cleared. A sustained tone produces none either: its energy is a plateau,
    so the prominence bar is never cleared.
    """
    if waveform.size < 3:
        return []

    teager = np.abs(waveform[1:-1] ** 2 - waveform[:-2] * waveform[2:])
    if teager.size < 3:
        return []

    window = int(TEAGER_MEDIAN_WINDOW_S * sample_rate) | 1
    baseline = medfilt(teager, kernel_size=min(window, teager.size - (teager.size + 1) % 2))

    median = float(np.median(teager))
    mad = float(np.median(np.abs(teager - median)))
    global_threshold = median + TEAGER_GLOBAL_MAD_MULTIPLE * mad / MAD_TO_SIGMA

    threshold = np.maximum(
        np.maximum(TEAGER_LOCAL_MULTIPLE * np.maximum(baseline, median), global_threshold),
        TEAGER_FLOOR_FRACTION * float(teager.max()),
    )

    peaks, _ = find_peaks(teager, height=threshold, distance=max(1, int(MIN_CRACKLE_SEPARATION_S * sample_rate)))
    if peaks.size == 0:
        return []

    prominences = peak_prominences(teager, peaks)[0]
    widths = peak_widths(teager, peaks, rel_height=CRACKLE_WIDTH_REL_HEIGHT)[0] / sample_rate

    events: list[CrackleEvent] = []
    for peak, prominence, width in zip(peaks, prominences, widths, strict=True):
        if width > MAX_CRACKLE_S or prominence < CRACKLE_PROMINENCE_FRACTION * teager[peak]:
            continue
        centre_s = (peak + 1) / sample_rate
        events.append(CrackleEvent(centre_s - width / 2, centre_s + width / 2))
    return events


def _flag_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of every run of True."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _cluster_peaks(peaks: list[float]) -> list[float]:
    """Collapses the per-frame peaks of one run into the distinct tones it carried."""
    if not peaks:
        return []
    unique = sorted(set(round(peak, 1) for peak in peaks))
    clustered = [unique[0]]
    for peak in unique[1:]:
        if peak - clustered[-1] > WHEEZE_PEAK_MIN_HZ:
            clustered.append(peak)
    return clustered


def position_in_cycle(time_s: float, cycle: tuple[float, float]) -> str:
    """Early inspiratory crackles point to obstructive disease, late ones to pneumonic disease."""
    start_s, end_s = cycle
    span = end_s - start_s
    if span <= 0:
        return MIDDLE
    fraction = (time_s - start_s) / span
    if fraction < EARLY_THIRD:
        return EARLY
    if fraction < LATE_THIRD:
        return MIDDLE
    return LATE


def _cycle_of(time_s: float, cycles: list[tuple[float, float]]) -> int | None:
    for index, (start_s, end_s) in enumerate(cycles):
        if start_s <= time_s <= end_s:
            return index
    return None


def render_cycles_summary(cycles: list[tuple[float, float]], rate_per_min: float, ie_ratio: float | None) -> str:
    if not cycles:
        return "No breathing cycles could be segmented from the envelope."

    mean_length = float(np.mean([end - start for start, end in cycles]))
    parts = [
        f"Respiratory rate ~{rate_per_min:.0f}/min over {len(cycles)} segmented cycles",
        f"mean cycle length {mean_length:.1f} s",
    ]
    if ie_ratio is not None:
        parts.append(f"inspiration:expiration ~1:{1 / ie_ratio:.1f}" if ie_ratio > 0 else "")
    return "; ".join(part for part in parts if part) + "."


def render_adventitious_summary(
    cycles: list[tuple[float, float]], wheezes: list[WheezeEvent], crackles: list[CrackleEvent]
) -> str:
    if not wheezes and not crackles:
        return NO_EVENTS_TEXT

    lines: list[str] = []
    for event in wheezes:
        cycle_index = _cycle_of(event.start_s, cycles)
        where = f"Cycle {cycle_index + 1}" if cycle_index is not None else f"{event.start_s:.1f} s"
        tones = " and ".join(f"~{peak:.0f} Hz" for peak in event.peak_hz)
        position = position_in_cycle(event.start_s, cycles[cycle_index]) if cycle_index is not None else ""
        position_text = f" in the {position} third of the cycle" if position else ""
        lines.append(
            f"{where}: {event.kind} wheeze-like tonal energy at {tones} "
            f"for {event.end_s - event.start_s:.1f} s{position_text}."
        )

    by_cycle_and_position: dict[tuple[int | None, str, str], int] = {}
    for event in crackles:
        cycle_index = _cycle_of(event.start_s, cycles)
        position = position_in_cycle(event.start_s, cycles[cycle_index]) if cycle_index is not None else MIDDLE
        key = (cycle_index, position, event.kind)
        by_cycle_and_position[key] = by_cycle_and_position.get(key, 0) + 1

    for (cycle_index, position, kind), count in sorted(
        by_cycle_and_position.items(), key=lambda kv: (kv[0][0] is None, kv[0][0], kv[0][1])
    ):
        where = f"Cycle {cycle_index + 1}" if cycle_index is not None else "Outside the segmented cycles"
        lines.append(f"{where}: {count} {kind} crackle-like transients in the {position} third of the cycle.")

    per_cycle = len(crackles) / len(cycles) if cycles else 0.0
    lines.append(
        f"Total: {len(wheezes)} wheeze-like events, {len(crackles)} crackle-like transients ({per_cycle:.1f} per cycle)."
    )
    return "\n".join(lines)


def extract_item(item_id: str, waveform: np.ndarray, sample_rate: int) -> RespiratoryEvidence:
    filtered = preprocess(waveform, sample_rate)
    cycles = detect_cycles(filtered, sample_rate)
    rate = respiratory_rate_per_min(cycles, waveform.size / sample_rate)
    ie_ratio = inspiration_expiration_ratio(filtered, sample_rate, cycles)

    return RespiratoryEvidence(
        item_id=item_id,
        cycles_summary=render_cycles_summary(cycles, rate, ie_ratio),
        adventitious_summary=render_adventitious_summary(
            cycles, detect_wheezes(filtered, sample_rate), detect_crackles(filtered, sample_rate)
        ),
        cycle_spans=cycles,
    )


def _iter_rows(dataset_id: str, split: str):
    ds = load_dataset(dataset_id, split=split).cast_column("audio", Audio(decode=False))
    for row in ds:
        item_id = os.path.splitext(os.path.basename(str(row["file"])))[0]
        yield item_id, row["audio"]["bytes"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure respiratory features for every ICBHI recording.")
    parser.add_argument("--dataset-id", type=str, default=DATASET_ID)
    parser.add_argument("--split", type=str, default=DATASET_SPLIT)
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    failures: list[str] = []
    written = 0
    with open(args.output_file, "w") as f:
        for item_id, audio_bytes in _iter_rows(args.dataset_id, args.split):
            try:
                waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)
                evidence = extract_item(item_id, waveform, TARGET_SR)
            except (RuntimeError, OSError, ValueError) as e:
                logger.error(f"Failed on {item_id}: {e}")
                failures.append(item_id)
                continue

            f.write(evidence.model_dump_json() + "\n")
            f.flush()
            written += 1

    logger.info(f"Wrote {written} records to {args.output_file}")
    if failures:
        logger.warning(f"{len(failures)} items failed: {failures}")


if __name__ == "__main__":
    main()
