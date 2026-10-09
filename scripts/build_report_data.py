"""Builds the tables every report figure and number is read from, so text and figures cannot drift apart.

One pass over every finished run folder of every model. Writes, into the output folder:

* `mmar_summary.csv`  - one row per model x experiment: accuracy overall / transcript-solvable (TS) /
  audio-dependent (AD), change against the model's own baseline on each, won / lost items with an exact
  McNemar p-value (Holm-corrected across the model's experiments), unparsed share and top-answer share.
* `mmar_items.csv`    - one row per model x experiment x item: majority-vote correctness, subset, MMAR
  layer and sub-category. Item-level questions (who wins where) are answered from this file.
* `icbhi_summary.csv` - one row per model x experiment: balanced accuracy, accuracy, change against the
  baseline (r0) and against the silence control (c1) with paired permutation p-values, top-answer share.
* `icbhi_answers.csv` - one row per model x experiment x answer: the share of all answers (all runs).
* `mmar_split.json`   - the TS / AD split, rebuilt from the text-only run so the report is self-contained.

Text-only arms (no audio model) are filed under their own model, the text-only control. Each model's newest
finished run of an experiment is used. This is a project script (it imports the package).
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from analyze_runs import (
    AUDIO_DEPENDENT,
    ICBHI,
    MMAR,
    TRANSCRIPT_SOLVABLE,
    balanced_accuracy,
    categories_of,
    load_run_dir,
    load_split,
    majority_correct,
    mcnemar_exact,
    paired_permutation_ba,
    write_split,
)
from compare_models import UNPARSED_ANSWER, _records, answer_text, holm, latest_finished_runs
from loguru import logger

_PCT = 100.0
BASELINE = "baseline"
ICBHI_SILENCE = "silence_prior"
TEXT_ONLY_MODEL = "Text-only control (Gemma-4-26B-A4B)"
TEXT_ONLY_EXPERIMENTS = frozenset(
    {"text_only_llm", "text_diarized_features", "text_only_metadata", "text_only_features_dictionary"}
)
TEXT_ONLY_SPLIT_RUN = "text_only_llm"

# (display name, family, parameters in billions, results folder relative to the data root)
MMAR_MODELS = [
    ("Voxtral-Mini-3B", "Voxtral", 3, "speech-processing-res-mmar-v2-scaling/voxtral_mini"),
    ("Qwen2.5-Omni-7B", "Qwen Omni", 7, "speech-processing-res-mmar-v2-scaling/qwen25omni_7b"),
    ("Voxtral-Small-24B", "Voxtral", 24, "speech-processing-res-mmar-v2"),
    ("Qwen3-Omni-30B-A3B", "Qwen Omni", 30, "speech-processing-res-mmar-v2-qwen3omni"),
    ("Step-Audio-R1.1", "Step-Audio", 32, "speech-processing-res-mmar-v2-scaling/step_audio_r1_1"),
]
ICBHI_MODELS = [
    ("Voxtral-Small-24B", "Voxtral", 24, "speech-processing-res-icbhi-v4"),
    ("Qwen3-Omni-30B-A3B", "Qwen Omni", 30, "speech-processing-res-icbhi-v4-qwen3omni"),
]


def pct(values: list[bool]) -> float:
    return float(np.mean(values)) * _PCT if values else float("nan")


def subset_accuracy(correct: dict[str, bool], items: set[str] | None = None) -> float:
    return pct([ok for item_id, ok in correct.items() if items is None or item_id in items])


def won_lost(before: dict[str, bool], after: dict[str, bool], items: set[str] | None = None) -> tuple[int, int]:
    shared = [i for i in before if i in after and (items is None or i in items)]
    return sum(after[i] and not before[i] for i in shared), sum(before[i] and not after[i] for i in shared)


def answer_shares(run_dir: Path) -> tuple[Counter, int]:
    counts = Counter(answer_text(record) for record in _records(run_dir))
    return counts, sum(counts.values())


def split_runs(data_root: Path, models: list[tuple], dataset: str) -> dict[str, dict[str, Path]]:
    """model name -> {experiment: newest finished run}; text-only arms are moved to the text-only model."""
    runs: dict[str, dict[str, Path]] = {}
    for name, _, _, folder in models:
        for experiment, run_dir in latest_finished_runs(str(data_root / folder), dataset).items():
            owner = TEXT_ONLY_MODEL if experiment in TEXT_ONLY_EXPERIMENTS else name
            runs.setdefault(owner, {})[experiment] = run_dir
    return runs


def mmar_tables(data_root: Path, split: dict[str, str]) -> tuple[list[dict], list[dict]]:
    ts = {i for i, s in split.items() if s == TRANSCRIPT_SOLVABLE}
    ad = {i for i, s in split.items() if s == AUDIO_DEPENDENT}
    meta = {name: (family, params) for name, family, params, _ in MMAR_MODELS}
    summary, items = [], []

    for model, experiments in split_runs(data_root, MMAR_MODELS, MMAR).items():
        family, params = meta.get(model, ("text-only", float("nan")))
        loaded = {experiment: load_run_dir(str(run_dir)) for experiment, run_dir in experiments.items()}
        majority = {experiment: majority_correct(runs) for experiment, runs in loaded.items()}
        base = majority.get(BASELINE)

        rows = []
        for experiment, correct in majority.items():
            layers = {i: r.category for run in loaded[experiment] for i, r in run.items()}
            sub_categories = {
                str(record["metadata"].get("item_id")): record["metadata"].get("sub_category") or ""
                for record in _records(experiments[experiment])
            }
            counts, total = answer_shares(experiments[experiment])
            top, top_count = counts.most_common(1)[0] if counts else (UNPARSED_ANSWER, 0)
            row = {
                "model": model,
                "family": family,
                "params_b": params,
                "experiment": experiment,
                "run_dir": experiments[experiment].name,
                "n": len(correct),
                "acc": subset_accuracy(correct),
                "acc_ts": subset_accuracy(correct, ts),
                "acc_ad": subset_accuracy(correct, ad),
                "unparsed_pct": counts[UNPARSED_ANSWER] / total * _PCT if total else float("nan"),
                "top_answer": top,
                "top_answer_pct": top_count / total * _PCT if total else float("nan"),
            }
            for suffix, subset in (("", None), ("_ts", ts), ("_ad", ad)):
                if base is None or experiment == BASELINE:
                    row.update({f"d{suffix}": float("nan"), f"won{suffix}": 0, f"lost{suffix}": 0, f"p{suffix}": 1.0})
                    continue
                won, lost = won_lost(base, correct, subset)
                row.update(
                    {
                        f"d{suffix}": subset_accuracy(correct, subset) - subset_accuracy(base, subset),
                        f"won{suffix}": won,
                        f"lost{suffix}": lost,
                        f"p{suffix}": mcnemar_exact(won, lost),
                    }
                )
            rows.append(row)
            items += [
                {
                    "model": model,
                    "experiment": experiment,
                    "item_id": item_id,
                    "subset": split.get(item_id, ""),
                    "layer": layers.get(item_id, ""),
                    "sub_category": sub_categories.get(item_id, ""),
                    "correct": int(ok),
                }
                for item_id, ok in correct.items()
            ]

        techniques = [row for row in rows if row["experiment"] != BASELINE and base is not None]
        for row, adjusted in zip(techniques, holm([row["p"] for row in techniques]), strict=True):
            row["p_holm"] = adjusted
        summary += rows
    return summary, items


def icbhi_tables(data_root: Path) -> tuple[list[dict], list[dict]]:
    summary, answers = [], []
    for model, experiments in split_runs(data_root, ICBHI_MODELS, ICBHI).items():
        loaded = {experiment: load_run_dir(str(run_dir)) for experiment, run_dir in experiments.items()}
        majority = {experiment: majority_correct(runs) for experiment, runs in loaded.items()}
        classes = {i: c for runs in loaded.values() for i, c in categories_of(runs).items()}
        references = {"base": majority.get(BASELINE), "silence": majority.get(ICBHI_SILENCE)}

        rows = []
        for experiment, correct in majority.items():
            counts, total = answer_shares(experiments[experiment])
            top, top_count = counts.most_common(1)[0] if counts else (UNPARSED_ANSWER, 0)
            ba = balanced_accuracy(correct, classes)
            row = {
                "model": model,
                "experiment": experiment,
                "run_dir": experiments[experiment].name,
                "n": len(correct),
                "ba": ba,
                "acc": subset_accuracy(correct),
                "unparsed_pct": counts[UNPARSED_ANSWER] / total * _PCT if total else float("nan"),
                "top_answer": top,
                "top_answer_pct": top_count / total * _PCT if total else float("nan"),
            }
            for label, reference in references.items():
                if reference is None or reference is correct:
                    row.update({f"d_ba_vs_{label}": float("nan"), f"p_vs_{label}": 1.0})
                    continue
                reference_ba = balanced_accuracy(reference, classes)
                row[f"d_ba_vs_{label}"] = ba - reference_ba if ba is not None and reference_ba is not None else None
                row[f"p_vs_{label}"] = paired_permutation_ba(correct, reference, classes)
            rows.append(row)
            answers += [
                {"model": model, "experiment": experiment, "answer": answer, "share_pct": count / total * _PCT}
                for answer, count in counts.most_common()
            ]

        techniques = [row for row in rows if row["experiment"] != BASELINE and references["base"] is not None]
        for row, adjusted in zip(techniques, holm([row["p_vs_base"] for row in techniques]), strict=True):
            row["p_vs_base_holm"] = adjusted
        summary += rows
    return summary, answers


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the report's data tables from the run folders.")
    parser.add_argument("--data-root", default="data", help="Folder holding the speech-processing-res-* folders")
    parser.add_argument("--output-dir", default="insights/report/data")
    args = parser.parse_args()

    data_root, output = Path(args.data_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    text_only = split_runs(data_root, MMAR_MODELS, MMAR).get(TEXT_ONLY_MODEL, {}).get(TEXT_ONLY_SPLIT_RUN)
    if text_only is None:
        sys.exit(f"No finished {TEXT_ONLY_SPLIT_RUN} run: the TS / AD split cannot be built.")
    split_file = output / "mmar_split.json"
    write_split(str(text_only), str(split_file))

    mmar_summary, mmar_items = mmar_tables(data_root, load_split(str(split_file)))
    icbhi_summary, icbhi_answers = icbhi_tables(data_root)
    write_csv(output / "mmar_summary.csv", mmar_summary)
    write_csv(output / "mmar_items.csv", mmar_items)
    write_csv(output / "icbhi_summary.csv", icbhi_summary)
    write_csv(output / "icbhi_answers.csv", icbhi_answers)


if __name__ == "__main__":
    main()
