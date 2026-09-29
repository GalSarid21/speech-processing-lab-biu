# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "datasets",
#     "librosa",
#     "loguru",
#     "numpy",
#     "soundfile",
#     "torch",
#     "transformers",
# ]
# ///
"""A6: embeds every ICBHI clip and writes, per item, its nearest neighbours and their diagnoses.

Two artifacts come out of one pass:

* `icbhi_neighbor_labels.json` - item -> the neighbours' labels, injected as text evidence (A6a);
* `icbhi_rag_mapping.json`     - item -> the neighbour item ids, used as audio demonstrations (A6b).

It also prints the pure kNN majority-vote accuracy. That is the control the experiments have to
beat: if the audio-language model scores the same, it added nothing to the retrieval.

Neighbours are drawn from the same 174 clips, so patient overlap is possible and near-duplicates are
excluded explicitly. See data/README.md.

Verify the embedding model id before the first run.
"""

import argparse
import io
import json
import os
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from loguru import logger
from transformers import AutoModel, AutoProcessor

DATASET_ID = "DynamicSuperb/RespiratorySoundClassification_ICBHI2017"
DATASET_SPLIT = "test"
EMBEDDING_MODEL_ID = "laion/clap-htsat-unfused"
DEFAULT_LABELS_OUTPUT = "data/icbhi_neighbor_labels.json"
DEFAULT_MAPPING_OUTPUT = "data/icbhi_rag_mapping.json"
DEFAULT_NEAR_DUPLICATES = "data/icbhi_near_duplicates.json"

TARGET_SR = 48_000
TOP_K = 5
PCT = 100.0


def load_clips(dataset_id: str, split: str) -> list[tuple[str, str, np.ndarray]]:
    ds = load_dataset(dataset_id, split=split).cast_column("audio", Audio(decode=False))
    clips = []
    for row in ds:
        item_id = os.path.splitext(os.path.basename(str(row["file"])))[0]
        waveform, sample_rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        clips.append((item_id, str(row["label"]), librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)))
    return clips


def embed(clips, model, processor, device: str) -> np.ndarray:
    vectors = []
    for _, _, waveform in clips:
        inputs = processor(audios=waveform, sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            embedding = model.get_audio_features(**inputs)[0]
        vectors.append((embedding / embedding.norm()).cpu().numpy())
    return np.stack(vectors)


def nearest_neighbors(
    embeddings: np.ndarray, item_ids: list[str], near_duplicates: dict[str, list[str]], top_k: int
) -> dict[str, list[int]]:
    similarity = embeddings @ embeddings.T
    index_of = {item_id: index for index, item_id in enumerate(item_ids)}

    neighbors: dict[str, list[int]] = {}
    for index, item_id in enumerate(item_ids):
        scores = similarity[index].copy()
        scores[index] = -np.inf
        for duplicate in near_duplicates.get(item_id, []):
            if duplicate in index_of:
                scores[index_of[duplicate]] = -np.inf
        neighbors[item_id] = [int(i) for i in np.argsort(-scores)[:top_k]]
    return neighbors


def knn_vote_accuracy(neighbors: dict[str, list[int]], labels: list[str], item_ids: list[str]) -> float:
    """The control: what retrieval alone scores, with no language model involved."""
    correct = 0
    for index, item_id in enumerate(item_ids):
        votes = [labels[i] for i in neighbors[item_id]]
        correct += Counter(votes).most_common(1)[0][0] == labels[index]
    return correct / len(item_ids) * PCT


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ICBHI audio-embedding kNN artifacts.")
    parser.add_argument("--dataset-id", type=str, default=DATASET_ID)
    parser.add_argument("--split", type=str, default=DATASET_SPLIT)
    parser.add_argument("--model-id", type=str, default=EMBEDDING_MODEL_ID)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--labels-output", type=str, default=DEFAULT_LABELS_OUTPUT)
    parser.add_argument("--mapping-output", type=str, default=DEFAULT_MAPPING_OUTPUT)
    parser.add_argument("--near-duplicates", type=str, default=DEFAULT_NEAR_DUPLICATES)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading {args.model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id).to(device)
    model.eval()

    clips = load_clips(args.dataset_id, args.split)
    item_ids = [item_id for item_id, _, _ in clips]
    labels = [label for _, label, _ in clips]
    logger.info(f"Embedding {len(clips)} clips...")
    embeddings = embed(clips, model, processor, device)

    near_duplicates = {}
    if Path(args.near_duplicates).exists():
        near_duplicates = json.loads(Path(args.near_duplicates).read_text())
    else:
        logger.warning(f"No near-duplicate file at {args.near_duplicates}; run scripts/diagnose_icbhi.py first.")

    neighbors = nearest_neighbors(embeddings, item_ids, near_duplicates, args.top_k)

    labels_output = {item_id: [labels[i] for i in indices] for item_id, indices in neighbors.items()}
    mapping_output = {item_id: [item_ids[i] for i in indices] for item_id, indices in neighbors.items()}

    for path, payload in ((args.labels_output, labels_output), (args.mapping_output, mapping_output)):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2))
        logger.info(f"Wrote {len(payload)} entries to {target}")

    logger.info(
        f"Control - pure kNN majority vote accuracy (k={args.top_k}): "
        f"{knn_vote_accuracy(neighbors, labels, item_ids):.1f}%. "
        "An experiment that only matches this number added nothing to retrieval."
    )


if __name__ == "__main__":
    main()
