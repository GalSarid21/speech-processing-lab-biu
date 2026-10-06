import json
import sys
import types
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import soundfile as sf


class DummyMock:
    """Stand-in for the vllm module tree: vllm is not installable on macOS."""

    __path__: ClassVar[list[str]] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, name):
        return DummyMock()

    def __call__(self, *args, **kwargs):
        return DummyMock()

    def __iter__(self):
        return iter([])


dummy_vllm = types.ModuleType("vllm")
dummy_vllm.LLM = DummyMock
dummy_vllm.SamplingParams = DummyMock
sys.modules["vllm"] = dummy_vllm

dummy_sampling_params = types.ModuleType("vllm.sampling_params")
dummy_sampling_params.StructuredOutputsParams = DummyMock
dummy_sampling_params.GuidedDecodingParams = DummyMock
sys.modules["vllm.sampling_params"] = dummy_sampling_params

from speech_processing.config.core import DatasetConfig  # noqa: E402 - needs the vllm stub above

TEST_SAMPLE_RATE = 16_000
SINE_HZ = 220.0

EVAL_IDS = ["ID1", "ID2"]
SHOT_ID = "SHOT1"

MMAR_ROWS = [
    {
        "id": "ID1",
        "question": "Who speaks first?",
        # numpy array, as pandas yields it from the real dataset
        "choices": np.array(["the first speaker", "the second speaker"]),
        "answer": "the second speaker",
        "audio_path": "./audio/id1.wav",
        "modality": "speech",
        "category": "Speaker",
        "sub-category": "Order",
    },
    {
        "id": "ID2",
        # stringified list, the other shape the real dataset uses
        "choices": "['British', 'American', 'Australian']",
        "question": "Which accent?",
        "answer": " American ",
        "audio_path": "./audio/id2.wav",
        "modality": "speech",
        "category": "Accent",
        "sub-category": "Imitation",
    },
    {
        "id": SHOT_ID,
        "question": "Is the speaker serious?",
        "choices": ["yes", "no"],
        "answer": "no",
        "audio_path": "./audio/shot1.wav",
        "modality": "speech",
        "category": "Tone",
        "sub-category": "Playfulness",
    },
]

TRANSCRIPTS = {
    "ID1": "hello there",
    "ID2": "cheers mate",
    SHOT_ID: "oh really",
}


@pytest.fixture
def wav_factory(tmp_path):
    def _make(duration_s: float, name: str) -> str:
        num_samples = max(1, int(duration_s * TEST_SAMPLE_RATE))
        t = np.arange(num_samples) / TEST_SAMPLE_RATE
        path = tmp_path / name
        sf.write(path, (0.5 * np.sin(2 * np.pi * SINE_HZ * t)).astype(np.float32), TEST_SAMPLE_RATE)
        return str(path)

    return _make


ICBHI_INSTRUCTION_TEMPLATES = [
    "This audio is a recording of a respiratory cycle, recorded at the patient's trachea, with the "
    "acquisition mode being multichannel. Detect the disease in this lung sound audio. The answer could be "
    "COPD, Pneumonia, URTI, LRTI, Bronchiectasis, Bronchiolitis, Asthma or healthy.",
    "Listen to the respiratory cycle recorded at the patient's left lower, with the acquisition mode being "
    "single channel, and identify the potential disease.",
    "The following respiratory sound was recorded at the patient's right upper, with the acquisition mode "
    "being single channel. What is the diagnosis?",
]

ICBHI_ROWS = [
    {
        "file": "audio1.wav",
        "label": "COPD",
        "instruction": ICBHI_INSTRUCTION_TEMPLATES[0],
        "audio": {"bytes": b"RIFFfake-copd"},
    },
    {
        "file": "audio2.wav",
        "label": "No potential disease detected",
        "instruction": ICBHI_INSTRUCTION_TEMPLATES[1],
        "audio": {"bytes": b"RIFFfake-healthy"},
    },
    {
        "file": "audio3.wav",
        "label": "Pneumonia",
        "instruction": ICBHI_INSTRUCTION_TEMPLATES[2],
        "audio": {"bytes": b"RIFFfake-pneumonia"},
    },
    # Held out of evaluation by ICBHI_DEMO_SPLIT below: a demonstration, never a scored item.
    {
        "file": "audio4.wav",
        "label": "URTI",
        "instruction": ICBHI_INSTRUCTION_TEMPLATES[1],
        "audio": {"bytes": b"RIFFfake-urti"},
    },
]
ICBHI_DEMO_IDS = ["audio4"]
ICBHI_EVAL_IDS = ["audio1", "audio2", "audio3"]


@pytest.fixture
def icbhi_config(tmp_path):
    features_file = tmp_path / "respiratory.jsonl"
    features_file.write_text(
        "\n".join(
            json.dumps(
                {
                    "item_id": item_id,
                    "cycles_summary": f"Rate ~18/min; 4 cycles for {item_id}.",
                    "adventitious_summary": f"Cycle 2: wheeze-like tonal energy for {item_id}.",
                    "cycle_spans": [[0.0, 1.0], [1.0, 2.0]],
                }
            )
            for item_id in ("audio1", "audio2")
        )
        + "\n"
    )

    tags_file = tmp_path / "tags.jsonl"
    tags_file.write_text(
        "\n".join(
            json.dumps({"item_id": item_id, "tags_summary": f"0.0-5.0 s: Breathing 0.81 for {item_id}"})
            for item_id in ("audio1", "audio2", "audio3")
        )
        + "\n"
    )

    neighbors_file = tmp_path / "neighbors.json"
    neighbors_file.write_text(json.dumps({"audio1": ["COPD", "COPD", "Pneumonia"]}))

    rag_file = tmp_path / "icbhi_rag.json"
    rag_file.write_text(json.dumps({"audio1": ["audio2", "audio3"]}))

    demo_ids_file = tmp_path / "icbhi_demo_ids.json"
    demo_ids_file.write_text(json.dumps({"demonstration_ids": ICBHI_DEMO_IDS, "excluded_near_duplicate_ids": []}))

    return DatasetConfig(
        dataset_id="test/icbhi",
        target_labels=[],
        split="test",
        audio_base_dir=str(tmp_path / "icbhi_audio"),
        respiratory_features_file=str(features_file),
        audio_tags_file=str(tags_file),
        neighbor_labels_file=str(neighbors_file),
        demo_ids_file=str(demo_ids_file),
    )


@pytest.fixture
def patched_icbhi(mocker):
    mocker.patch(
        "speech_processing.data.icbhi._load_icbhi_frame",
        return_value=pd.DataFrame(ICBHI_ROWS),
    )


@pytest.fixture
def mmar_config(tmp_path):
    features_file = tmp_path / "features.jsonl"
    features_file.write_text(
        "\n".join(
            json.dumps(
                {
                    "item_id": item_id,
                    "diarized_transcript": f"[0.0-1.0] SPK1: line for {item_id}",
                    "acoustic_features": f"SPK1: F0 mean 180 Hz for {item_id}",
                }
            )
            for item_id in EVAL_IDS
        )
        + "\n"
    )

    few_shot_file = tmp_path / "few_shot_ids.txt"
    few_shot_file.write_text(f"{SHOT_ID}\n")

    transcripts_file = tmp_path / "transcripts.json"
    transcripts_file.write_text(json.dumps(TRANSCRIPTS))

    return DatasetConfig(
        dataset_id="test/mmar",
        target_labels=[],
        split="test",
        sample_ids=list(EVAL_IDS),
        audio_base_dir="data/MMAR",
        transcripts_file=str(transcripts_file),
        few_shot_ids_file=str(few_shot_file),
        acoustic_features_file=str(features_file),
    )


@pytest.fixture
def patched_mmar(mocker):
    mocker.patch(
        "speech_processing.data.dataset._load_mmar_frame",
        return_value=pd.DataFrame(MMAR_ROWS),
    )
    mocker.patch("speech_processing.data.dataset._load_transcripts", return_value=dict(TRANSCRIPTS))
