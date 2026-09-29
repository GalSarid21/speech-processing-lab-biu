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

from speech_processing.data.icbhi import (
    HEALTHY_LABEL,
    ICBHI_LABELS,
    MULTICHANNEL,
    parse_instruction_metadata,
)

DATASET_ID = "DynamicSuperb/RespiratorySoundClassification_ICBHI2017"
DATASET_SPLIT = "test"
DEFAULT_OUTPUT = "results/ICBHI/V4/dataset_report.md"
DEFAULT_NEAR_DUPLICATES = "data/icbhi_near_duplicates.json"

COPD_LABEL = "COPD"
FINGERPRINT_SR = 4_000
FINGERPRINT_HOP = 512
NEAR_DUPLICATE_THRESHOLD = 0.98
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


def metadata_rule_accuracy(df: pd.DataFrame) -> float:
    """The trivial text-only rule: multichannel means COPD, anything else means no disease."""
    predictions = np.where(df["mode"] == MULTICHANNEL, COPD_LABEL, HEALTHY_LABEL)
    return float(np.mean(predictions == df["label"].to_numpy())) * PCT


def metadata_ceiling_accuracy(df: pd.DataFrame) -> float:
    """Leave-one-out accuracy of the best decision rule on (mode, location).

    Majority vote per cell, with the item itself held out, so the number is not inflated by a cell
    that contains a single recording.
    """
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        cells[(row["mode"], row["location"])].append(row["label"])

    global_majority = Counter(df["label"]).most_common(1)[0][0]
    correct = 0
    for _, row in df.iterrows():
        cell = list(cells[(row["mode"], row["location"])])
        cell.remove(row["label"])
        prediction = Counter(cell).most_common(1)[0][0] if cell else global_majority
        correct += prediction == row["label"]

    return correct / len(df) * PCT


def chance_balanced_accuracy(df: pd.DataFrame) -> float:
    return PCT / df["label"].nunique()


def fingerprint(audio_bytes: bytes) -> np.ndarray:
    """Mean-normalised chroma+MFCC fingerprint, used only to find clips cut from one recording."""
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=FINGERPRINT_SR)

    mfcc = librosa.feature.mfcc(y=waveform, sr=FINGERPRINT_SR, hop_length=FINGERPRINT_HOP)
    chroma = librosa.feature.chroma_stft(y=waveform, sr=FINGERPRINT_SR, hop_length=FINGERPRINT_HOP)
    vector = np.concatenate([mfcc.mean(axis=1), chroma.mean(axis=1)])
    return vector / (np.linalg.norm(vector) + 1e-12)


def find_near_duplicates(df: pd.DataFrame, threshold: float) -> dict[str, list[str]]:
    """Pairs whose fingerprints are almost identical, i.e. very likely the same source recording."""
    fingerprints = {}
    for _, row in df.iterrows():
        audio = row["audio"]
        if isinstance(audio, dict) and audio.get("bytes"):
            fingerprints[row["item_id"]] = fingerprint(audio["bytes"])

    pairs: dict[str, list[str]] = defaultdict(list)
    for left, right in combinations(sorted(fingerprints), 2):
        if float(np.dot(fingerprints[left], fingerprints[right])) >= threshold:
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


def build_report(df: pd.DataFrame, near_duplicates: dict[str, list[str]]) -> str:
    label_counts = df["label"].value_counts()
    majority_pct = label_counts.iloc[0] / len(df) * PCT
    durations, rates = durations_and_rates(df)

    sections = [
        "# ICBHI 2017 dataset report",
        "",
        f"Rows: **{len(df)}**. Distinct labels: **{df['label'].nunique()}** of {len(ICBHI_LABELS)} ICBHI diagnoses.",
        "",
        "## Label distribution",
        "",
        label_counts.to_frame("n").to_markdown(),
        "",
        "## Acquisition mode x label",
        "",
        contingency(df, "mode").to_markdown(),
        "",
        "## Chest location x label",
        "",
        contingency(df, "location").to_markdown(),
        "",
        "## Baselines",
        "",
        "| Baseline | Accuracy % |",
        "|---|---|",
        f"| Majority class ({label_counts.index[0]}) | {majority_pct:.1f} |",
        f"| Text-only rule: multichannel -> COPD, else -> no disease | {metadata_rule_accuracy(df):.1f} |",
        f"| Metadata ceiling: best (mode, location) rule, leave-one-out | {metadata_ceiling_accuracy(df):.1f} |",
        "",
        f"Chance balanced accuracy over the {df['label'].nunique()} present classes: "
        f"**{chance_balanced_accuracy(df):.1f}%**.",
        "",
        "Every audio technique must be read against the metadata ceiling, not against chance. Any "
        "experiment whose prompt carries the recording metadata can reach that ceiling without "
        "listening to anything.",
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
        f"{len(near_duplicates)} clips have at least one near-duplicate at a cosine similarity of "
        f"{NEAR_DUPLICATE_THRESHOLD}. These are almost certainly cut from the same source recording and "
        "are excluded from each other's few-shot demonstrations.",
        "",
        "Patient ids are not published in this packaging, so patient-level overlap between the evaluation "
        "items and any demonstration pool drawn from the same 174 clips cannot be ruled out. Few-shot "
        "numbers measured that way are an upper bound; see data/README.md.",
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
    args = parser.parse_args()

    df = load_frame(args.dataset_id, args.split)

    logger.info("Fingerprinting clips to find near-duplicates...")
    near_duplicates = find_near_duplicates(df, args.near_duplicate_threshold)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(df, near_duplicates))
    logger.info(f"Wrote the dataset report to {output_path}")

    duplicates_path = Path(args.near_duplicates_output)
    duplicates_path.parent.mkdir(parents=True, exist_ok=True)
    duplicates_path.write_text(json.dumps(near_duplicates, indent=2))
    logger.info(f"Wrote {len(near_duplicates)} near-duplicate entries to {duplicates_path}")


if __name__ == "__main__":
    main()
