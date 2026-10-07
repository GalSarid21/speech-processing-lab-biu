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
def scaling():
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("plot_model_scaling", SCRIPTS_DIR / "plot_model_scaling.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(SCRIPTS_DIR))


def write_run(results: Path, experiment: str, num_correct: int) -> None:
    run_dir = results / f"mmar_{experiment}_20261007_100000"
    run_dir.mkdir(parents=True)
    records = [
        {
            "sample_id": item_id,
            "generated_text": "B" if index < num_correct else "A",
            "ground_truth": "B",
            "metadata": {"item_id": item_id, "choices": CHOICES, "category": "Speaker"},
        }
        for index, item_id in enumerate(ITEM_IDS)
    ]
    (run_dir / "raw_output.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (run_dir / "metrics.txt").write_text("done\n")


def test_ranges_cover_only_experiments_every_model_ran(scaling, tmp_path):
    small, large = tmp_path / "small", tmp_path / "large"
    for experiment, correct in (("baseline", 2), ("cot", 1), ("few_shot_audio", 3)):
        write_run(small, experiment, correct)
    for experiment, correct in (("baseline", 3), ("cot", 4), ("role_prompting", 0)):
        write_run(large, experiment, correct)
    models = [
        {"name": "S", "family": "F", "params_b": 3.0, "dir": str(small)},
        {"name": "L", "family": "F", "params_b": 24.0, "dir": str(large)},
    ]

    shared, scored = scaling.score_models(models)

    assert_that(shared).is_equal_to(["baseline", "cot"])
    assert_that(scored[0]["baseline"]).is_equal_to(50.0)
    assert_that(scored[1]["best"]).is_equal_to(("cot", 100.0))
    assert_that(scored[0]["worst"]).is_equal_to(("cot", 25.0))


def test_subset_scores_only_the_given_items(scaling, tmp_path):
    write_run(tmp_path / "m", "baseline", 2)
    models = [{"name": "M", "family": "F", "params_b": 7.0, "dir": str(tmp_path / "m")}]

    _, scored = scaling.score_models(models, {"ID3", "ID4"})

    assert_that(scored[0]["baseline"]).is_equal_to(0.0)
