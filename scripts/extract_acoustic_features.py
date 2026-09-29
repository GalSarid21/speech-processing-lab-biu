# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "datasets",
#     "librosa",
#     "loguru",
#     "numpy",
#     "openai-whisper",
#     "pyannote.audio>=4.0",
#     "pydantic",
#     "torch",
# ]
# ///
"""Extracts a diarized transcript and per-speaker acoustic measurements for each MMAR item.

Usage:
    HF_TOKEN=... uv run scripts/extract_acoustic_features.py --sample-ids-file data/mmar_en_speech_test_ids.txt

Accept the terms of pyannote/speaker-diarization-3.1 on Hugging Face before running.
The output is JSONL, one AcousticEvidence record per line, matching the loader's schema.
"""

import argparse
import itertools
import os
import sys
from dataclasses import dataclass

import librosa
import numpy as np
import torch
import whisper
from datasets import load_dataset
from loguru import logger
from pyannote.audio import Pipeline
from pydantic import BaseModel

DATASET_ID = "BoJack/MMAR"
DATASET_SPLIT = "test"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_MODEL = "large-v3"
WHISPER_LANGUAGE = "en"
HF_TOKEN_ENV = "HF_TOKEN"

SAMPLE_RATE = 16_000
F0_MIN_HZ = 50.0
F0_MAX_HZ = 500.0
F0_LOW_PCT, F0_HIGH_PCT = 5, 95
MIN_PAUSE_S = 0.25
EPSILON = 1e-10
DB_FACTOR = 20.0

DEFAULT_AUDIO_BASE_DIR = "data/MMAR"
DEFAULT_OUTPUT_FILE = "data/mmar_acoustic_features.jsonl"
DEFAULT_SAMPLE_IDS_FILE = "data/mmar_en_speech_test_ids.txt"

SPEAKER_NAME_TEMPLATE = "SPK{index}"
DIARIZATION_ATTR = "speaker_diarization"


class AcousticEvidence(BaseModel):
    item_id: str
    diarized_transcript: str
    acoustic_features: str


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class SpeakerFeatures:
    f0_mean_hz: float
    f0_low_hz: float
    f0_high_hz: float
    loudness_db: float
    words_per_s: float
    num_pauses: int
    mean_pause_s: float

    def describe(self, name: str) -> str:
        return (
            f"{name}: F0 mean {self.f0_mean_hz:.0f} Hz "
            f"(5-95% range {self.f0_low_hz:.0f}-{self.f0_high_hz:.0f} Hz), "
            f"loudness {self.loudness_db:+.1f} dB vs clip, "
            f"rate {self.words_per_s:.1f} words/s, "
            f"{self.num_pauses} pauses (avg {self.mean_pause_s:.1f} s)"
        )


def load_item_paths(sample_ids_file: str | None, audio_base_dir: str) -> list[tuple[str, str]]:
    """Item ids and audio paths come from the MMAR dataset itself, never from a directory listing."""
    df = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=False).to_pandas()

    if sample_ids_file and os.path.exists(sample_ids_file):
        with open(sample_ids_file) as f:
            wanted = {line.strip() for line in f if line.strip()}
        df = df[df["id"].isin(wanted)]
        logger.info(f"Filtered to {len(df)} items using {sample_ids_file}")

    return [
        (str(row["id"]), os.path.join(audio_base_dir, str(row["audio_path"]).lstrip("./"))) for _, row in df.iterrows()
    ]


def diarize(pipeline, audio_path: str) -> list[tuple[float, float, str]]:
    output = pipeline(audio_path)
    # pyannote 4.x wraps the annotation in a result object; 3.x returns it directly.
    annotation = getattr(output, DIARIZATION_ATTR, output)
    return [(turn.start, turn.end, speaker) for turn, _, speaker in annotation.itertracks(yield_label=True)]


def transcribe_words(model, audio_path: str, use_fp16: bool) -> list[Word]:
    """One Whisper pass over the whole file, so context and word alignment are preserved."""
    result = model.transcribe(audio_path, word_timestamps=True, language=WHISPER_LANGUAGE, fp16=use_fp16)
    words: list[Word] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            text = str(word.get("word", "")).strip()
            if text:
                words.append(Word(text=text, start=float(word["start"]), end=float(word["end"])))
    return words


def assign_speakers(words: list[Word], turns: list[tuple[float, float, str]]) -> list[str]:
    """Each word takes the speaker whose turn overlaps it most, falling back to the nearest turn."""
    labels: list[str] = []
    for word in words:
        best_label, best_overlap = None, 0.0
        for start, end, speaker in turns:
            overlap = min(word.end, end) - max(word.start, start)
            if overlap > best_overlap:
                best_label, best_overlap = speaker, overlap

        if best_label is None and turns:
            midpoint = (word.start + word.end) / 2
            best_label = min(turns, key=lambda t: abs((t[0] + t[1]) / 2 - midpoint))[2]

        labels.append(best_label or SPEAKER_NAME_TEMPLATE.format(index=1))
    return labels


def rename_by_first_appearance(labels: list[str]) -> dict[str, str]:
    """Raw pyannote ids are not ordered, so SPK1 must be defined as the first speaker heard."""
    mapping: dict[str, str] = {}
    for label in labels:
        if label not in mapping:
            mapping[label] = SPEAKER_NAME_TEMPLATE.format(index=len(mapping) + 1)
    return mapping


def diarized_lines(words: list[Word], speakers: list[str]) -> list[str]:
    lines: list[str] = []
    current: list[Word] = []
    current_speaker: str | None = None

    def flush() -> None:
        if current and current_speaker:
            lines.append(
                f"[{current[0].start:.1f}-{current[-1].end:.1f}] {current_speaker}: "
                + " ".join(w.text for w in current)
            )

    for word, speaker in zip(words, speakers, strict=True):
        if speaker != current_speaker:
            flush()
            current, current_speaker = [], speaker
        current.append(word)
    flush()
    return lines


def speaker_features(
    waveform: np.ndarray,
    sample_rate: int,
    turns: list[tuple[float, float]],
    words: list[Word],
    clip_rms: float,
) -> SpeakerFeatures:
    segments = [waveform[int(start * sample_rate) : int(end * sample_rate)] for start, end in turns]
    segments = [s for s in segments if len(s) > 0]
    speaker_audio = np.concatenate(segments) if segments else np.zeros(1, dtype=np.float32)

    f0, voiced_flag, _ = librosa.pyin(speaker_audio, fmin=F0_MIN_HZ, fmax=F0_MAX_HZ, sr=sample_rate)
    voiced_f0 = f0[voiced_flag] if f0 is not None and np.any(voiced_flag) else np.array([])
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)] if voiced_f0.size else voiced_f0

    f0_mean = float(np.mean(voiced_f0)) if voiced_f0.size else 0.0
    f0_low = float(np.percentile(voiced_f0, F0_LOW_PCT)) if voiced_f0.size else 0.0
    f0_high = float(np.percentile(voiced_f0, F0_HIGH_PCT)) if voiced_f0.size else 0.0

    speaker_rms = float(np.sqrt(np.mean(speaker_audio**2)))
    loudness_db = DB_FACTOR * float(np.log10((speaker_rms + EPSILON) / (clip_rms + EPSILON)))

    word_time = sum(w.end - w.start for w in words)
    words_per_s = len(words) / word_time if word_time > 0 else 0.0

    pauses = [
        later.start - earlier.end
        for earlier, later in itertools.pairwise(words)
        if later.start - earlier.end >= MIN_PAUSE_S
    ]

    return SpeakerFeatures(
        f0_mean_hz=f0_mean,
        f0_low_hz=f0_low,
        f0_high_hz=f0_high,
        loudness_db=loudness_db,
        words_per_s=words_per_s,
        num_pauses=len(pauses),
        mean_pause_s=float(np.mean(pauses)) if pauses else 0.0,
    )


def extract_item(item_id: str, audio_path: str, diarizer, whisper_model, use_fp16: bool) -> AcousticEvidence:
    waveform, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    raw_turns = diarize(diarizer, audio_path)
    words = transcribe_words(whisper_model, audio_path, use_fp16)

    raw_labels = assign_speakers(words, raw_turns)
    rename = rename_by_first_appearance(raw_labels)
    for _, _, speaker in raw_turns:
        if speaker not in rename:
            rename[speaker] = SPEAKER_NAME_TEMPLATE.format(index=len(rename) + 1)

    speakers = [rename[label] for label in raw_labels]
    lines = diarized_lines(words, speakers)

    clip_rms = float(np.sqrt(np.mean(waveform**2)))
    descriptions: list[str] = []
    for name in sorted(set(speakers) | set(rename.values()), key=lambda n: int(n.removeprefix("SPK"))):
        speaker_turns = [(s, e) for s, e, label in raw_turns if rename.get(label) == name]
        speaker_words = [w for w, spk in zip(words, speakers, strict=True) if spk == name]
        if not speaker_turns and not speaker_words:
            continue
        features = speaker_features(waveform, sample_rate, speaker_turns, speaker_words, clip_rms)
        descriptions.append(features.describe(name))

    return AcousticEvidence(
        item_id=item_id,
        diarized_transcript="\n".join(lines),
        acoustic_features="\n".join(descriptions),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract diarized transcripts and acoustic features for MMAR.")
    parser.add_argument("--audio-base-dir", type=str, default=DEFAULT_AUDIO_BASE_DIR)
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--sample-ids-file", type=str, default=DEFAULT_SAMPLE_IDS_FILE)
    args = parser.parse_args()

    token = os.environ.get(HF_TOKEN_ENV)
    if not token:
        logger.error(
            f"{HF_TOKEN_ENV} is not set. Accept the terms of {DIARIZATION_MODEL} on Hugging Face "
            f"and export {HF_TOKEN_ENV} before running."
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"

    logger.info(f"Loading {DIARIZATION_MODEL} on {device}...")
    diarizer = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token).to(torch.device(device))

    logger.info(f"Loading Whisper {WHISPER_MODEL} on {device}...")
    whisper_model = whisper.load_model(WHISPER_MODEL, device=device)

    items = load_item_paths(args.sample_ids_file, args.audio_base_dir)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    failures: list[str] = []
    with open(args.output_file, "w") as f:
        for index, (item_id, audio_path) in enumerate(items, start=1):
            logger.info(f"Processing {index}/{len(items)}: {item_id}")
            try:
                evidence = extract_item(item_id, audio_path, diarizer, whisper_model, use_fp16)
            except (RuntimeError, OSError, ValueError) as e:
                logger.error(f"Failed on {item_id}: {e}")
                failures.append(item_id)
                continue

            f.write(evidence.model_dump_json() + "\n")
            f.flush()

    logger.info(f"Wrote {len(items) - len(failures)} records to {args.output_file}")
    if failures:
        logger.warning(f"{len(failures)} items failed: {failures}")


if __name__ == "__main__":
    main()
