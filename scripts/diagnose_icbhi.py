"""Dataset diagnostics for ICBHI 2017, written before any experiment is run.

The headline question it answers: how much of the label can be predicted from the recording
metadata that the dataset's own instruction states? On ICBHI the acquisition mode is tied to the
clinical site and therefore to the diagnosis, so a prompt that carries it can score well without
listening. The "metadata ceiling" printed here is the number every audio technique has to be read
against.

This is a project script (it imports the package), so it runs in the project environment.
"""

import argparse
import io
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from loguru import logger
from scipy.signal import correlate, correlation_lags

from speech_processing.data.icbhi import (
    HEALTHY_LABEL,
    ICBHI_LABELS,
    ICBHI_PRIMARY_LABELS,
    MULTICHANNEL,
    DemonstrationSplit,
    parse_instruction_metadata,
    select_demonstration_split,
)

DATASET_ID = "DynamicSuperb/RespiratorySoundClassification_ICBHI2017"
DATASET_SPLIT = "test"
DEFAULT_OUTPUT = "results/ICBHI/V4/dataset_report.md"
DEFAULT_NEAR_DUPLICATES = "data/icbhi_near_duplicates.json"
DEFAULT_DEMO_SPLIT = "data/icbhi_demo_ids.json"

COPD_LABEL = "COPD"
# Near-duplicate = two clips that contain the same audio samples, i.e. cut from overlapping stretches
# of one recording. Lung-sound energy sits well below 1 kHz, so 2 kHz keeps the waveform detail that
# makes shared audio match, while keeping ~15k pairwise correlations to about a minute.
DUPLICATE_SR = 2_000
MIN_SHARED_AUDIO_S = 5.0
# Pearson correlation over the shared stretch: shared audio scores near 1, unrelated recordings near 0.
NEAR_DUPLICATE_THRESHOLD = 0.8
PCT = 100.0


def load_frame(dataset_id: str, split: str) -> pd.DataFrame:
    logger.info(f"Loading {dataset_id} [{split}]...")
    ds = load_dataset(dataset_id, split=split).cast_column("audio", Audio(decode=False))
    df = ds.to_pandas()

    metadata = [parse_instruction_metadata(str(instruction)) for instruction in df["instruction"]]
    df["item_id"] = [Path(str(name)).stem for name in df["file"]]
    df["location"] = [location for location, _ in metadata]
    df["mode"] = [mode for _, mode in metadata]
    return df


def contingency(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return pd.crosstab(df[column], df["label"])


def balanced_accuracy_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro recall over the primary classes, matching the experiments' primary metric."""
    recalls = [np.mean(y_pred[y_true == label] == label) for label in ICBHI_PRIMARY_LABELS if np.any(y_true == label)]
    return float(np.mean(recalls)) * PCT if recalls else 0.0


def _loo_cell_majority(df: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Leave-one-out majority vote within each metadata cell, so a cell holding one recording cannot
    predict its own label."""
    cells: dict[tuple, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        cells[tuple(row[k] for k in keys)].append(row["label"])

    global_majority = Counter(df["label"]).most_common(1)[0][0]
    predictions = []
    for _, row in df.iterrows():
        cell = list(cells[tuple(row[k] for k in keys)])
        cell.remove(row["label"])
        predictions.append(Counter(cell).most_common(1)[0][0] if cell else global_majority)
    return np.array(predictions)


def metadata_rules(df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """Every text-only rule worth comparing against. None of them hears the audio."""
    return [
        (
            "Text-only rule: multichannel -> COPD, else -> no disease",
            np.where(df["mode"] == MULTICHANNEL, COPD_LABEL, HEALTHY_LABEL),
        ),
        ("Leave-one-out majority per acquisition mode", _loo_cell_majority(df, ["mode"])),
        ("Leave-one-out majority per chest location", _loo_cell_majority(df, ["location"])),
        ("Leave-one-out majority per (mode, location)", _loo_cell_majority(df, ["mode", "location"])),
    ]


def baseline_rows(df: pd.DataFrame) -> tuple[list[str], float]:
    """Markdown table rows plus the metadata ceiling: the best metadata rule's balanced accuracy.

    The ceiling is a maximum over rules, not one rule picked in advance - on the real data the finest
    (mode, location) rule is *worse* than the plain mode rule, because location splits cells too thin.
    """
    y_true = df["label"].to_numpy()
    majority = Counter(y_true).most_common(1)[0][0]
    always = np.array([majority] * len(df))

    rows = [
        f"| Majority class (always {majority}) | {np.mean(always == y_true) * PCT:.1f} | "
        f"{balanced_accuracy_pct(y_true, always):.1f} |"
    ]
    ceiling = 0.0
    for name, predictions in metadata_rules(df):
        ba = balanced_accuracy_pct(y_true, predictions)
        ceiling = max(ceiling, ba)
        rows.append(f"| {name} | {np.mean(predictions == y_true) * PCT:.1f} | {ba:.1f} |")
    return rows, ceiling


def chance_balanced_accuracy() -> float:
    return PCT / len(ICBHI_PRIMARY_LABELS)


class ClipSignal:
    """A clip resampled for duplicate detection, with the running sums an overlap correlation needs."""

    def __init__(self, waveform: np.ndarray) -> None:
        self.samples = (waveform - waveform.mean()).astype(np.float64)
        self.sums = np.concatenate([[0.0], np.cumsum(self.samples)])
        self.square_sums = np.concatenate([[0.0], np.cumsum(self.samples**2)])

    def __len__(self) -> int:
        return len(self.samples)


def load_clip_signal(audio_bytes: bytes) -> ClipSignal:
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return ClipSignal(librosa.resample(waveform, orig_sr=sample_rate, target_sr=DUPLICATE_SR))


def max_overlap_correlation(a: ClipSignal, b: ClipSignal, min_overlap: int) -> float:
    """The highest Pearson correlation between `a` and `b` over every alignment sharing >= min_overlap samples.

    Clips cut from overlapping stretches of one recording contain the same samples, so at the right
    lag they correlate near 1. Unrelated recordings stay near 0 at every lag. Averaged spectral
    features cannot tell these apart: every lung recording has much the same average spectrum.
    """
    lags = correlation_lags(len(a), len(b), mode="full")
    products = correlate(a.samples, b.samples, mode="full", method="fft")

    # For lag L the shared stretch is a[a_start:a_end] against b[a_start - L : a_end - L].
    a_start = np.maximum(0, lags)
    a_end = np.minimum(len(a), len(b) + lags)
    overlap = a_end - a_start
    valid = overlap >= min_overlap
    if not valid.any():
        return 0.0

    lags, products, a_start, a_end, overlap = (x[valid] for x in (lags, products, a_start, a_end, overlap))
    b_start, b_end = a_start - lags, a_end - lags
    n = overlap.astype(np.float64)

    sum_a = a.sums[a_end] - a.sums[a_start]
    sum_b = b.sums[b_end] - b.sums[b_start]
    var_a = (a.square_sums[a_end] - a.square_sums[a_start]) - sum_a**2 / n
    var_b = (b.square_sums[b_end] - b.square_sums[b_start]) - sum_b**2 / n
    covariance = products - sum_a * sum_b / n

    denominator = np.sqrt(np.clip(var_a * var_b, 0.0, None))
    correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
    return float(correlation.max())


def find_near_duplicates(df: pd.DataFrame, threshold: float) -> dict[str, list[str]]:
    """Pairs of clips that share at least MIN_SHARED_AUDIO_S of the same audio."""
    signals = {
        row["item_id"]: load_clip_signal(row["audio"]["bytes"])
        for _, row in df.iterrows()
        if isinstance(row["audio"], dict) and row["audio"].get("bytes")
    }
    min_overlap = int(MIN_SHARED_AUDIO_S * DUPLICATE_SR)

    pairs: dict[str, list[str]] = defaultdict(list)
    for left, right in combinations(sorted(signals), 2):
        if max_overlap_correlation(signals[left], signals[right], min_overlap) >= threshold:
            pairs[left].append(right)
            pairs[right].append(left)
    return dict(pairs)


def durations_and_rates(df: pd.DataFrame) -> tuple[list[float], set[int]]:
    durations, rates = [], set()
    for _, row in df.iterrows():
        audio = row["audio"]
        if not (isinstance(audio, dict) and audio.get("bytes")):
            continue
        info = sf.info(io.BytesIO(audio["bytes"]))
        durations.append(float(info.duration))
        rates.add(int(info.samplerate))
    return durations, rates


def build_report(df: pd.DataFrame, near_duplicates: dict[str, list[str]], split: DemonstrationSplit) -> str:
    held_out = split.held_out_ids
    eval_df = df[~df["item_id"].isin(held_out)]
    label_counts = (
        pd.DataFrame({"all rows": df["label"].value_counts(), "evaluation set": eval_df["label"].value_counts()})
        .fillna(0)
        .astype(int)
    )
    rule_rows, ceiling = baseline_rows(eval_df)
    durations, rates = durations_and_rates(df)
    labels_by_id = dict(zip(df["item_id"], df["label"], strict=True))

    sections = [
        "# ICBHI 2017 dataset report",
        "",
        f"Rows: **{len(df)}**, of which **{len(eval_df)}** form the evaluation set shared by every "
        f"experiment and **{len(held_out)}** are held out as demonstrations. Distinct labels: "
        f"**{df['label'].nunique()}** of {len(ICBHI_LABELS)} ICBHI diagnoses.",
        "",
        "## Label distribution",
        "",
        label_counts.to_markdown(),
        "",
        f"The primary metric averages over {len(ICBHI_PRIMARY_LABELS)} classes: "
        f"{', '.join(ICBHI_PRIMARY_LABELS)}. Asthma and LRTI remain answer options and count towards "
        "accuracy, but with one and two items they would each move a macro average by 6-12 points.",
        "",
        "## Acquisition mode x label",
        "",
        contingency(df, "mode").to_markdown(),
        "",
        "## Chest location x label",
        "",
        contingency(df, "location").to_markdown(),
        "",
        "## Baselines (evaluation set)",
        "",
        "| Predictor | Accuracy % | Balanced accuracy % |",
        "|---|---|---|",
        *rule_rows,
        "",
        f"Chance balanced accuracy: **{chance_balanced_accuracy():.1f}%**. "
        f"Metadata ceiling (best rule above, balanced accuracy): **{ceiling:.1f}%**.",
        "",
        "Always answering the majority class scores exactly chance on balanced accuracy, however large "
        "its share. The metadata ceiling is the bar for any experiment whose prompt states the recording "
        "conditions: it is reachable without listening to anything.",
        "",
        "## Held-out demonstrations",
        "",
        "| Item | Label |",
        "|---|---|",
        *[f"| {item_id} | {labels_by_id[item_id]} |" for item_id in split.demonstration_ids],
        "",
        f"Also held out as near-duplicates of a demonstration: {split.excluded_near_duplicate_ids or 'none'}. "
        f"Selection seed: {split.seed}.",
        "",
        "These items are never scored, by any experiment. Patient ids are not published in this packaging, "
        "so a demonstration may still share a patient with an evaluation item; see data/README.md.",
        "",
        "## Audio properties",
        "",
        f"- Sample rates present: {sorted(rates)}",
        f"- Duration: min {min(durations):.1f} s, median {np.median(durations):.1f} s, max {max(durations):.1f} s"
        if durations
        else "- Duration: unavailable",
        "",
        "## Near-duplicate clips",
        "",
        f"{len(near_duplicates)} clips share at least {MIN_SHARED_AUDIO_S:.0f} s of identical audio with another "
        f"clip (waveform correlation >= {NEAR_DUPLICATE_THRESHOLD} over the shared stretch), i.e. were cut from "
        "overlapping stretches of one recording. A demonstration's near-duplicates are held out of evaluation "
        "along with it. Clips cut from *non-overlapping* stretches of one recording share no audio and cannot "
        "be detected this way; that is the patient-overlap caveat in data/README.md.",
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the ICBHI dataset before running experiments.")
    parser.add_argument("--dataset-id", type=str, default=DATASET_ID)
    parser.add_argument("--split", type=str, default=DATASET_SPLIT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--near-duplicates-output", type=str, default=DEFAULT_NEAR_DUPLICATES)
    parser.add_argument("--near-duplicate-threshold", type=float, default=NEAR_DUPLICATE_THRESHOLD)
    parser.add_argument("--demo-split-output", type=str, default=DEFAULT_DEMO_SPLIT)
    args = parser.parse_args()

    df = load_frame(args.dataset_id, args.split)

    logger.info("Comparing clip waveforms to find clips that share audio (near-duplicates)...")
    near_duplicates = find_near_duplicates(df, args.near_duplicate_threshold)

    split = select_demonstration_split(dict(zip(df["item_id"], df["label"], strict=True)), near_duplicates)
    split_path = Path(args.demo_split_output)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(split.model_dump_json(indent=2))
    logger.info(
        f"Held out {len(split.held_out_ids)} items as demonstrations ({split.demonstration_ids}); "
        f"wrote the split to {split_path}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(df, near_duplicates, split))
    logger.info(f"Wrote the dataset report to {output_path}")

    duplicates_path = Path(args.near_duplicates_output)
    duplicates_path.parent.mkdir(parents=True, exist_ok=True)
    duplicates_path.write_text(json.dumps(near_duplicates, indent=2))
    logger.info(f"Wrote {len(near_duplicates)} near-duplicate entries to {duplicates_path}")


if __name__ == "__main__":
    main()
