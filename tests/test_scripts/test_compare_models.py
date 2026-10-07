import importlib.util
import json
import sys
from pathlib import Path

import pytest
from assertpy import assert_that

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
CHOICES = ["Dog", "Cat"]
ITEM_IDS = [f"ID{i}" for i in range(1, 5)]


@pytest.fixture(scope="module")
def compare():
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("compare_models", SCRIPTS_DIR / "compare_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(SCRIPTS_DIR))


def record(item_id: str, answer: str) -> dict:
    return {
        "sample_id": f"data/MMAR/audio/{item_id}.wav",
        "instruction": "Answer.",
        "generated_text": answer,
        "final_turn_text": answer,
        "ground_truth": "B",
        "metadata": {"item_id": item_id, "question": "q", "choices": CHOICES, "category": "Speaker"},
    }


def write_run(results: Path, name: str, answers: dict[str, str], model_id: str | None, finished: bool = True) -> Path:
    run_dir = results / name
    run_dir.mkdir(parents=True)
    (run_dir / "raw_output.jsonl").write_text("\n".join(json.dumps(record(i, a)) for i, a in answers.items()) + "\n")
    if finished:
        (run_dir / "metrics.txt").write_text("done\n")
    if model_id:
        (run_dir / "run_config.json").write_text(json.dumps({"audio_model_id": model_id}))
    return run_dir


@pytest.fixture
def folders(tmp_path):
    voxtral, qwen = tmp_path / "voxtral", tmp_path / "qwen"
    write_run(voxtral, "mmar_baseline_20261006_100000", dict.fromkeys(ITEM_IDS, "B"), None)
    # Qwen: one wrong answer, one answer written in Chinese, one English answer quoting a Chinese word.
    # The Chinese answer does not parse to a letter, so it also counts as wrong.
    qwen_answers = {"ID1": "A", "ID2": "答案是 B 说话者是猫", "ID3": 'He says "猫".\n\nB. Cat', "ID4": "B"}
    write_run(qwen, "mmar_baseline_20261007_100000", qwen_answers, "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    # A newer crashed run must not replace the finished one.
    write_run(qwen, "mmar_baseline_20261007_120000", dict.fromkeys(ITEM_IDS, "A"), "Qwen/x", finished=False)
    return voxtral, qwen


def test_report_compares_matched_experiments_and_tracks_language(compare, folders):
    voxtral, qwen = folders
    report = compare.build_report(
        "mmar", dir_a=str(voxtral), name_a="voxtral", dir_b=str(qwen), name_b="qwen", num_examples=3
    )

    assert_that(report).contains("| baseline | 4 | 100.0 | 50.0 | -50.0 |")
    assert_that(report).contains("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    assert_that(report).contains("unrecorded")
    assert_that(report).contains("| 0.0 | 25.0 |")
    assert_that(report).contains("`ID2`").contains("`ID3`")


def test_language_check_separates_a_switch_from_a_quote(compare, folders):
    _, qwen = folders
    check = compare.language_check(qwen / "mmar_baseline_20261007_100000")

    assert_that([item for item, _ in check["non_english"]]).is_equal_to(["ID2"])
    assert_that([item for item, _ in check["quotes_cjk"]]).is_equal_to(["ID3"])
    assert_that(check["pct"]).is_equal_to(25.0)


def test_mixed_models_in_one_folder_are_flagged(compare, folders):
    voxtral, qwen = folders
    write_run(qwen, "mmar_cot_20261007_100000", dict.fromkeys(ITEM_IDS, "B"), "mistralai/Voxtral-Small-24B-2507")
    write_run(voxtral, "mmar_cot_20261006_100000", dict.fromkeys(ITEM_IDS, "B"), None)

    report = compare.build_report(
        "mmar", dir_a=str(voxtral), name_a="voxtral", dir_b=str(qwen), name_b="qwen", num_examples=3
    )

    assert_that(report).contains("mixes audio models")
