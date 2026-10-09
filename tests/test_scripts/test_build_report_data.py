import importlib.util
import json
import sys
from pathlib import Path

import pytest
from assertpy import assert_that

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
CHOICES = ["Dog", "Cat"]
TS_ITEMS = ["ID1", "ID2", "ID3"]
AD_ITEMS = ["ID4", "ID5"]
SPLIT = {**dict.fromkeys(TS_ITEMS, "transcript-solvable"), **dict.fromkeys(AD_ITEMS, "audio-dependent")}


@pytest.fixture(scope="module")
def report():
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("build_report_data", SCRIPTS_DIR / "build_report_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(SCRIPTS_DIR))


def write_run(results: Path, experiment: str, correct: set[str]) -> None:
    run_dir = results / f"mmar_{experiment}_20261007_100000"
    run_dir.mkdir(parents=True)
    records = [
        {
            "sample_id": item_id,
            "final_turn_text": "B" if item_id in correct else "A",
            "ground_truth": "B",
            "metadata": {
                "item_id": item_id,
                "choices": CHOICES,
                "category": "Semantic Layer",
                "sub_category": "Content Analysis",
            },
        }
        for item_id in SPLIT
    ]
    (run_dir / "raw_output.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (run_dir / "stability_report.json").write_text("{}")


@pytest.fixture
def data_root(report, tmp_path, monkeypatch):
    folder = tmp_path / "model_a"
    write_run(folder, "baseline", {"ID1", "ID2", "ID4"})
    # Gains two transcript-solvable items, loses the one audio-dependent item the baseline had right.
    write_run(folder, "transcript_augmented", {"ID1", "ID2", "ID3"})
    write_run(folder, "text_only_llm", set(TS_ITEMS))
    monkeypatch.setattr(report, "MMAR_MODELS", [("Model-A", "Family", 7, "model_a")])
    return tmp_path


def row_for(rows: list[dict], model: str, experiment: str) -> dict:
    return next(row for row in rows if row["model"] == model and row["experiment"] == experiment)


def test_changes_are_measured_per_subset_against_the_models_own_baseline(report, data_root):
    summary, _ = report.mmar_tables(data_root, SPLIT)
    row = row_for(summary, "Model-A", "transcript_augmented")

    assert_that(row["acc_ts"]).is_close_to(100.0, 1e-9)
    assert_that(row["d_ts"]).is_close_to(100 / 3, 1e-9)
    assert_that(row["d_ad"]).is_close_to(-50.0, 1e-9)
    assert_that((row["won"], row["lost"])).is_equal_to((1, 1))
    assert_that((row["won_ts"], row["lost_ad"])).is_equal_to((1, 1))


def test_text_only_arms_are_filed_under_the_text_only_control(report, data_root):
    summary, items = report.mmar_tables(data_root, SPLIT)

    text_only = row_for(summary, report.TEXT_ONLY_MODEL, "text_only_llm")
    assert_that(text_only["acc"]).is_close_to(60.0, 1e-9)
    assert_that([row["experiment"] for row in summary if row["model"] == "Model-A"]).does_not_contain("text_only_llm")
    assert_that(items).is_length(3 * len(SPLIT))
    assert_that({item["sub_category"] for item in items}).is_equal_to({"Content Analysis"})
