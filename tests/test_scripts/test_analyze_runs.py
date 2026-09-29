import importlib.util
import json
from pathlib import Path

import pytest
from assertpy import assert_that
from loguru import logger

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_runs.py"
CHOICES = ["Dog", "Cat"]
ITEM_IDS = ["ID1", "ID2", "ID3", "ID4"]


@pytest.fixture(scope="module")
def analyze():
    spec = importlib.util.spec_from_file_location("analyze_runs", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v2_record(item_id: str, answer: str, ground_truth: str = "B") -> dict:
    return {
        "sample_id": f"data/MMAR/audio/{item_id}.wav",
        "instruction": "Answer.",
        "generated_text": f"reasoning ...\n{answer}",
        "final_turn_text": answer,
        "ground_truth": ground_truth,
        "metadata": {"item_id": item_id, "question": "q", "choices": CHOICES, "category": "Speaker"},
    }


def v1_record(item_id: str, diagnostic: int) -> dict:
    return {
        "sample_id": f"data/MMAR/audio/{item_id}.wav",
        "instruction": "Answer.",
        "generated_text": "Cat",
        "ground_truth": "Cat",
        "evaluation": {"reasoning": "r", "diagnostic_accuracy": diagnostic},
    }


def write_run_dir(base: Path, name: str, per_run_answers: list[dict[str, str]]) -> str:
    run_dir = base / name
    run_dir.mkdir(parents=True)
    for index, answers in enumerate(per_run_answers, start=1):
        lines = [json.dumps(v2_record(item_id, answer)) for item_id, answer in answers.items()]
        (run_dir / f"raw_output_run{index}.jsonl").write_text("\n".join(lines) + "\n")
    return str(run_dir)


@pytest.fixture
def run_dirs(tmp_path):
    # ID1 always right, ID2 right twice, ID3 right once, ID4 never right.
    baseline = write_run_dir(
        tmp_path,
        "mmar_baseline",
        [
            {"ID1": "B", "ID2": "B", "ID3": "B", "ID4": "A"},
            {"ID1": "B", "ID2": "B", "ID3": "A", "ID4": "A"},
            {"ID1": "B", "ID2": "A", "ID3": "A", "ID4": "A"},
        ],
    )
    candidate = write_run_dir(
        tmp_path,
        "mmar_candidate",
        [
            {"ID1": "B", "ID2": "B", "ID3": "B", "ID4": "B"},
            {"ID1": "B", "ID2": "B", "ID3": "B", "ID4": "B"},
            {"ID1": "B", "ID2": "B", "ID3": "B", "ID4": "A"},
        ],
    )
    legacy = tmp_path / "mmar_v1_legacy"
    legacy.mkdir()
    (legacy / "raw_output.jsonl").write_text(
        "\n".join(json.dumps(v1_record(item_id, 10 if item_id != "ID4" else 0)) for item_id in ITEM_IDS) + "\n"
    )
    return {"baseline": baseline, "candidate": candidate, "legacy": str(legacy)}


@pytest.mark.parametrize(
    "record, expected",
    [
        (v2_record("ID1", "B"), "ID1"),
        (v1_record("ID9", 10), "ID9"),
    ],
)
def test_item_key_handles_both_formats(analyze, record, expected):
    assert_that(analyze.item_key(record)).is_equal_to(expected)


@pytest.mark.parametrize(
    "record, expected",
    [
        (v2_record("ID1", "B"), True),
        (v2_record("ID1", "A"), False),
        (v2_record("ID1", "Cat"), True),
        (v1_record("ID1", 10), True),
        (v1_record("ID1", 4), False),
    ],
)
def test_is_correct_handles_both_formats(analyze, record, expected):
    assert_that(analyze.is_correct(record)).is_equal_to(expected)


def test_majority_needs_two_of_three(analyze, run_dirs):
    majority = analyze.majority_correct(analyze.load_run_dir(run_dirs["baseline"]))
    assert_that(majority).is_equal_to({"ID1": True, "ID2": True, "ID3": False, "ID4": False})


def test_legacy_run_dir_is_readable(analyze, run_dirs):
    majority = analyze.majority_correct(analyze.load_run_dir(run_dirs["legacy"]))
    assert_that(majority["ID1"]).is_true()
    assert_that(majority["ID4"]).is_false()


@pytest.mark.parametrize("gained, lost, expected", [(0, 0, 1.0), (10, 0, 0.01)])
def test_mcnemar_exact(analyze, gained, lost, expected):
    pvalue = analyze.mcnemar_exact(gained, lost)
    if (gained, lost) == (0, 0):
        assert_that(pvalue).is_equal_to(expected)
    else:
        assert_that(pvalue).is_less_than(expected)


def test_bootstrap_is_deterministic(analyze):
    outcomes = [True, False, True, True, False, True]
    assert_that(analyze.bootstrap_ci(outcomes)).is_equal_to(analyze.bootstrap_ci(outcomes))


def test_write_split_then_summarize_fills_both_subsets(analyze, tmp_path, run_dirs):
    split_file = tmp_path / "split.json"
    analyze.write_split(run_dirs["baseline"], str(split_file))

    split = json.loads(split_file.read_text())
    assert_that(set(split.values())).is_equal_to({"transcript-solvable", "audio-dependent"})

    summary_df, category_df = analyze.summarize([run_dirs["candidate"]], run_dirs["baseline"], split)
    row = summary_df.iloc[0]

    for subset in analyze.SUBSETS:
        assert_that(row[f"{subset} acc (n)"]).does_not_contain("(0)")
    assert_that(row["gained/lost"]).is_equal_to("2/0")
    assert_that(category_df.index.tolist()).contains("Speaker")


def test_missing_split_entries_warn(analyze, run_dirs):
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        analyze.summarize([run_dirs["candidate"]], None, {"ID1": "audio-dependent"})
    finally:
        logger.remove(sink_id)

    assert_that("".join(messages)).contains("missing from the split file")


ICBHI_CHOICES = [
    "COPD",
    "Pneumonia",
    "URTI",
    "LRTI",
    "Bronchiectasis",
    "Bronchiolitis",
    "Asthma",
    "No potential disease detected",
]


def icbhi_record(item_id: str, answer: str, ground_truth: str = "A") -> dict:
    return {
        "sample_id": f"data/ICBHI/{item_id}.wav",
        "instruction": "Choose.",
        "generated_text": f"reasoning ...\n{answer}",
        "final_turn_text": answer,
        "ground_truth": ground_truth,
        "metadata": {
            "item_id": item_id,
            "question": "What is the patient's diagnosis?",
            "choices": ICBHI_CHOICES,
            "category": "COPD",
            "sub_category": "multichannel|trachea",
        },
    }


@pytest.fixture
def icbhi_run_dirs(tmp_path):
    def write(name: str, answers: dict[str, str]) -> str:
        run_dir = tmp_path / name
        run_dir.mkdir(parents=True)
        (run_dir / "raw_output.jsonl").write_text(
            "\n".join(json.dumps(icbhi_record(item_id, answer)) for item_id, answer in answers.items()) + "\n"
        )
        return str(run_dir)

    return {
        "control": write("icbhi_text_only_metadata", {"audio1": "A", "audio2": "B"}),
        "candidate": write("icbhi_audio_features", {"audio1": "A", "audio2": "A"}),
    }


def test_icbhi_split_uses_the_metadata_solvable_label(analyze, tmp_path, icbhi_run_dirs):
    split_file = tmp_path / "icbhi_split.json"
    analyze.write_split(icbhi_run_dirs["control"], str(split_file), analyze.ICBHI)

    split = json.loads(split_file.read_text())
    assert_that(split).is_equal_to({"audio1": "metadata-solvable", "audio2": "audio-dependent"})


def test_icbhi_summary_uses_the_icbhi_subset_columns(analyze, tmp_path, icbhi_run_dirs):
    split_file = tmp_path / "icbhi_split.json"
    analyze.write_split(icbhi_run_dirs["control"], str(split_file), analyze.ICBHI)
    split = json.loads(split_file.read_text())

    summary_df, _ = analyze.summarize([icbhi_run_dirs["candidate"]], icbhi_run_dirs["control"], split, analyze.ICBHI)
    row = summary_df.iloc[0]

    assert_that(row["Experiment"]).is_equal_to("audio_features")
    assert_that(summary_df.columns.tolist()).contains("metadata-solvable acc (n)", "audio-dependent acc (n)")
    for subset in analyze.subsets_for(analyze.ICBHI):
        assert_that(row[f"{subset} acc (n)"]).does_not_contain("(0)")


@pytest.mark.parametrize("dataset, expected", [("mmar", "transcript-solvable"), ("icbhi", "metadata-solvable")])
def test_subsets_for(analyze, dataset, expected):
    assert_that(analyze.subsets_for(dataset)).is_equal_to((expected, "audio-dependent"))
