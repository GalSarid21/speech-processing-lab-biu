# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "datasets",
#     "librosa",
#     "loguru",
#     "numpy",
#     "pydantic",
#     "soundfile",
#     "torch",
#     "transformers",
# ]
# ///
"""A2: runs a general-purpose AudioSet tagger over each ICBHI recording and renders its output as text.

The tagger was never trained on auscultation, which is the point: it supplies an independent,
vocabulary-level description of what is in the recording ("Breathing", "Wheeze", "Cough") that the
language model can reason over. Tags outside the whitelist are dropped, because the interesting
question is whether the relevant categories fire, not what the tagger guesses overall.

Verify the model id below before the first run.

Usage:
    uv run scripts/extract_audio_tags.py --output-file data/icbhi_audio_tags.jsonl
"""

import argparse
import io
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from loguru import logger
from pydantic import BaseModel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

DATASET_ID = "DynamicSuperb/RespiratorySoundClassification_ICBHI2017"
DATASET_SPLIT = "test"
TAGGER_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEFAULT_OUTPUT_FILE = "data/icbhi_audio_tags.jsonl"

TARGET_SR = 16_000
WINDOW_S = 5.0
WINDOW_OVERLAP = 0.5
TOP_K = 5

# Only AudioSet categories that could plausibly describe a stethoscope recording.
TAG_WHITELIST = frozenset(
    {
        "Breathing",
        "Wheeze",
        "Gasp",
        "Cough",
        "Snoring",
        "Sneeze",
        "Heart sounds, heartbeat",
        "Heart murmur",
        "Rustle",
        "Noise",
        "Speech",
        "Silence",
        "Throat clearing",
    }
)
NO_TAGS_TEXT = "No whitelisted audio-tagger categories fired."


class AudioTagEvidence(BaseModel):
    """Matches speech_processing.data.icbhi.AudioTagEvidence exactly."""

    item_id: str
    tags_summary: str


def window_bounds(num_samples: int, sample_rate: int) -> list[tuple[int, int]]:
    window = int(WINDOW_S * sample_rate)
    hop = max(1, int(window * (1 - WINDOW_OVERLAP)))
    if num_samples <= window:
        return [(0, num_samples)]
    return [(start, min(start + window, num_samples)) for start in range(0, num_samples - window + 1, hop)]


def tag_windows(
    waveform: np.ndarray, sample_rate: int, model, extractor, device: str
) -> list[tuple[float, float, list[tuple[str, float]]]]:
    results = []
    for start, end in window_bounds(waveform.size, sample_rate):
        inputs = extractor(waveform[start:end], sampling_rate=sample_rate, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits[0]

        scores = torch.sigmoid(logits)
        ranked = torch.argsort(scores, descending=True)

        whitelisted: list[tuple[str, float]] = []
        for index in ranked:
            label = model.config.id2label[int(index)]
            if label in TAG_WHITELIST:
                whitelisted.append((label, float(scores[index])))
            if len(whitelisted) == TOP_K:
                break

        results.append((start / sample_rate, end / sample_rate, whitelisted))
    return results


def render_summary(windows: list[tuple[float, float, list[tuple[str, float]]]]) -> str:
    lines = [
        f"{start:.1f}-{end:.1f} s: " + ", ".join(f"{label} {score:.2f}" for label, score in tags)
        for start, end, tags in windows
        if tags
    ]
    return "\n".join(lines) if lines else NO_TAGS_TEXT


def _iter_rows(dataset_id: str, split: str):
    ds = load_dataset(dataset_id, split=split).cast_column("audio", Audio(decode=False))
    for row in ds:
        yield os.path.splitext(os.path.basename(str(row["file"])))[0], row["audio"]["bytes"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag every ICBHI recording with a general audio classifier.")
    parser.add_argument("--dataset-id", type=str, default=DATASET_ID)
    parser.add_argument("--split", type=str, default=DATASET_SPLIT)
    parser.add_argument("--model-id", type=str, default=TAGGER_MODEL_ID)
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading {args.model_id} on {device}...")
    extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    model = AutoModelForAudioClassification.from_pretrained(args.model_id).to(device)
    model.eval()

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
                summary = render_summary(tag_windows(waveform, TARGET_SR, model, extractor, device))
            except (RuntimeError, OSError, ValueError) as e:
                logger.error(f"Failed on {item_id}: {e}")
                failures.append(item_id)
                continue

            f.write(AudioTagEvidence(item_id=item_id, tags_summary=summary).model_dump_json() + "\n")
            f.flush()
            written += 1

    logger.info(f"Wrote {written} records to {args.output_file}")
    if failures:
        logger.warning(f"{len(failures)} items failed: {failures}")


if __name__ == "__main__":
    main()
