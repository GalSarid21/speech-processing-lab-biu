"""Compares experiment run directories for either dataset.

Majority-vote accuracy across repeated runs, bootstrap CIs, a paired McNemar test against a
baseline, per-category tables, and a two-way subset split:

* MMAR  - transcript-solvable vs audio-dependent, derived from the text-only run;
* ICBHI - metadata-solvable vs audio-dependent, derived from the text-only metadata control.

`--dataset` selects the extra columns. Scoring is the deterministic parser in both cases, so V1
result files (which carry a judge `evaluation` block instead of typed metadata) stay readable.

This is a project script (it imports the package), so it runs with the project environment rather
than a PEP 723 header.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from scipy.stats import binomtest

from speech_processing.data.icbhi import ICBHI_PRIMARY_LABELS
from speech_processing.evaluation.answer_parsing import parse_choice_letter

RAW_OUTPUT_GLOB = "raw_output*.jsonl"

MMAR = "mmar"
ICBHI = "icbhi"

AUDIO_DEPENDENT = "audio-dependent"
TRANSCRIPT_SOLVABLE = "transcript-solvable"
METADATA_SOLVABLE = "metadata-solvable"

# What the text-only control being right means, per dataset.
SOLVABLE_LABEL = {MMAR: TRANSCRIPT_SOLVABLE, ICBHI: METADATA_SOLVABLE}
DEFAULT_OUTPUT = {MMAR: "results/MMAR/V2/comparison.md", ICBHI: "results/ICBHI/V4/comparison.md"}
DEFAULT_SPLIT_FILE = {MMAR: "data/mmar_subset_split.json", ICBHI: "data/icbhi_subset_split.json"}
RUN_PREFIXES = {MMAR: "mmar_", ICBHI: "icbhi_"}

# Kept for backwards compatibility with split files written before --dataset existed.
SUBSETS = (TRANSCRIPT_SOLVABLE, AUDIO_DEPENDENT)


def subsets_for(dataset: str) -> tuple[str, str]:
    return SOLVABLE_LABEL[dataset], AUDIO_DEPENDENT


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
PERMUTATIONS = 10_000
PERMUTATION_SEED = 0
# ICBHI's primary metric averages over these classes only; see ICBHI_PRIMARY_LABELS.
ICBHI_PRIMARY = frozenset(ICBHI_PRIMARY_LABELS)
NOT_AVAILABLE = "-"
ICBHI_TABLE_LEGEND = (
    f"BA = balanced accuracy (macro recall) over the {len(ICBHI_PRIMARY)} primary classes; chance is "
    f"{100 / len(ICBHI_PRIMARY):.1f}%. Asthma and LRTI items are answered and count towards Acc, but not "
    "towards BA. BA 95% CI = stratified bootstrap. p (perm.) = two-sided paired permutation test on the BA "
    "difference against the baseline. Acc must be read against the majority-class rate, not against chance."
)
CI_ALPHA = 0.05
V1_CORRECT_DIAGNOSTIC_SCORE = 10
UNKNOWN_CATEGORY = "Unknown"


class ItemResult(BaseModel):
    item_id: str
    correct: bool
    category: str


def item_key(record: dict) -> str:
    """V2 records carry typed metadata; V1 records only have the audio path as sample_id."""
    metadata = record.get("metadata") or {}
    if metadata.get("item_id"):
        return str(metadata["item_id"])
    return Path(str(record.get("sample_id", ""))).stem


def is_correct(record: dict) -> bool:
    metadata = record.get("metadata") or {}
    choices = metadata.get("choices")

    if choices:
        answer_text = record.get("final_turn_text") or record.get("generated_text") or ""
        return parse_choice_letter(answer_text, list(choices)) == record.get("ground_truth")

    evaluation = record.get("evaluation") or {}
    return evaluation.get("diagnostic_accuracy") == V1_CORRECT_DIAGNOSTIC_SCORE


def load_run_dir(run_dir: str) -> list[dict[str, ItemResult]]:
    run_files = sorted(glob(str(Path(run_dir) / RAW_OUTPUT_GLOB)))
    if not run_files:
        raise SystemExit(f"No {RAW_OUTPUT_GLOB} files found in {run_dir}")

    runs: list[dict[str, ItemResult]] = []
    for run_file in run_files:
        results: dict[str, ItemResult] = {}
        with open(run_file) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = item_key(record)
                metadata = record.get("metadata") or {}
                results[key] = ItemResult(
                    item_id=key,
                    correct=is_correct(record),
                    category=str(metadata.get("category") or UNKNOWN_CATEGORY),
                )
        runs.append(results)
    return runs


def majority_correct(runs: list[dict[str, ItemResult]]) -> dict[str, bool]:
    """An item counts as correct when it is correct in a strict majority of the runs."""
    if not runs:
        return {}
    shared = set(runs[0])
    for run in runs[1:]:
        shared &= set(run)

    return {item_id: sum(run[item_id].correct for run in runs) * 2 > len(runs) for item_id in sorted(shared)}


def bootstrap_ci(correct: list[bool]) -> tuple[float, float]:
    if not correct:
        return 0.0, 0.0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    outcomes = np.asarray(correct, dtype=float)
    draws = rng.integers(0, len(outcomes), size=(BOOTSTRAP_RESAMPLES, len(outcomes)))
    accuracies = outcomes[draws].mean(axis=1)
    lower, upper = np.percentile(accuracies, [CI_ALPHA / 2 * 100, (1 - CI_ALPHA / 2) * 100])
    return float(lower) * 100, float(upper) * 100


def mcnemar_exact(gained: int, lost: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs only."""
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    return float(binomtest(min(gained, lost), discordant, 0.5).pvalue)


def _class_groups(correct: dict[str, bool], classes: dict[str, str], include: frozenset[str]) -> dict[str, list[bool]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for item_id, ok in correct.items():
        label = classes.get(item_id, UNKNOWN_CATEGORY)
        if label in include:
            groups[label].append(ok)
    return groups


def balanced_accuracy(
    correct: dict[str, bool], classes: dict[str, str], include: frozenset[str] = ICBHI_PRIMARY
) -> float | None:
    """Macro recall over the included classes; None when no item belongs to one of them."""
    groups = _class_groups(correct, classes, include)
    if not groups:
        return None
    return float(np.mean([np.mean(values) for values in groups.values()])) * 100


def stratified_bootstrap_ba_ci(
    correct: dict[str, bool], classes: dict[str, str], include: frozenset[str] = ICBHI_PRIMARY
) -> tuple[float, float] | None:
    """Resamples within each class, so every replicate keeps the class mix the metric is defined on."""
    groups = [np.asarray(v, dtype=float) for v in _class_groups(correct, classes, include).values()]
    if not groups:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_class = [g[rng.integers(0, len(g), size=(BOOTSTRAP_RESAMPLES, len(g)))].mean(axis=1) for g in groups]
    replicates = np.mean(per_class, axis=0)
    lower, upper = np.percentile(replicates, [CI_ALPHA / 2 * 100, (1 - CI_ALPHA / 2) * 100])
    return float(lower) * 100, float(upper) * 100


def paired_permutation_ba(
    ours: dict[str, bool],
    baseline: dict[str, bool],
    classes: dict[str, str],
    include: frozenset[str] = ICBHI_PRIMARY,
) -> float:
    """Two-sided paired permutation test on the balanced-accuracy difference.

    McNemar compares plain accuracy, so it cannot back a claim about balanced accuracy. Under the null
    the two systems are exchangeable on every item, so each permutation swaps their outcomes per item.
    """
    shared = sorted(i for i in set(ours) & set(baseline) if classes.get(i, UNKNOWN_CATEGORY) in include)
    if not shared:
        return 1.0

    a = np.array([ours[i] for i in shared], dtype=float)
    b = np.array([baseline[i] for i in shared], dtype=float)
    labels = sorted({classes[i] for i in shared})
    membership = np.array([[classes[i] == label for label in labels] for i in shared], dtype=float)
    counts = membership.sum(axis=0)

    def ba(outcomes: np.ndarray) -> np.ndarray:
        return ((outcomes @ membership) / counts).mean(axis=-1)

    observed = abs(ba(a) - ba(b))
    rng = np.random.default_rng(PERMUTATION_SEED)
    swap = rng.random((PERMUTATIONS, len(shared))) < 0.5
    differences = np.abs(ba(np.where(swap, b, a)) - ba(np.where(swap, a, b)))
    return float((np.sum(differences >= observed - 1e-12) + 1) / (PERMUTATIONS + 1))


def _fmt(value: float | None, pattern: str = "{:.1f}") -> str:
    return NOT_AVAILABLE if value is None else pattern.format(value)


def categories_of(runs: list[dict[str, ItemResult]]) -> dict[str, str]:
    return {item_id: result.category for run in runs for item_id, result in run.items()}


def write_split(text_only_dir: str, split_file: str, dataset: str = MMAR) -> None:
    """Derives the subset split from the text-only control. Never a side effect of a comparison."""
    solvable, dependent = subsets_for(dataset)
    runs = load_run_dir(text_only_dir)
    majority = majority_correct(runs)
    split = {item_id: solvable if correct else dependent for item_id, correct in majority.items()}

    Path(split_file).parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(split, f, indent=2)

    counts = Counter(split.values())
    logger.info(
        f"Wrote {len(split)} items to {split_file} ({counts[solvable]} {solvable}, {counts[dependent]} {dependent})."
    )


def load_split(split_file: str) -> dict[str, str]:
    if not Path(split_file).exists():
        logger.warning(f"Split file {split_file} not found; subset columns will be empty.")
        return {}
    with open(split_file) as f:
        return json.load(f)


def _experiment_name(run_dir: str, dataset: str = MMAR) -> str:
    return Path(run_dir.rstrip("/")).name.removeprefix(RUN_PREFIXES[dataset])


def _accuracy_row(
    name: str,
    *,
    runs: list[dict[str, ItemResult]],
    majority: dict[str, bool],
    baseline_majority: dict[str, bool],
    split: dict[str, str],
    dataset: str,
) -> dict[str, str]:
    """MMAR: plain accuracy, a bootstrap CI, and exact McNemar against the baseline."""
    outcomes = list(majority.values())
    accuracy = float(np.mean(outcomes)) * 100 if outcomes else 0.0
    lower, upper = bootstrap_ci(outcomes)
    run_accuracies = [float(np.mean([r.correct for r in run.values()])) * 100 for run in runs]

    row: dict[str, str] = {
        "Experiment": name,
        "n": str(len(outcomes)),
        "Acc (majority)": f"{accuracy:.1f}",
        "95% CI": f"[{lower:.1f}, {upper:.1f}]",
        "Acc (mean of runs)": f"{np.mean(run_accuracies):.1f}",
    }

    if baseline_majority:
        shared = sorted(set(majority) & set(baseline_majority))
        gained = sum(majority[i] and not baseline_majority[i] for i in shared)
        lost = sum(not majority[i] and baseline_majority[i] for i in shared)
        base_accuracy = float(np.mean([baseline_majority[i] for i in shared])) * 100 if shared else 0.0
        our_accuracy = float(np.mean([majority[i] for i in shared])) * 100 if shared else 0.0
        row["Δ vs base"] = f"{our_accuracy - base_accuracy:+.1f}"
        row["gained/lost"] = f"{gained}/{lost}"
        row["p"] = f"{mcnemar_exact(gained, lost):.3f}"

    for subset in subsets_for(dataset):
        subset_outcomes = [majority[i] for i in majority if split.get(i) == subset]
        subset_accuracy = float(np.mean(subset_outcomes)) * 100 if subset_outcomes else 0.0
        row[f"{subset} acc (n)"] = f"{subset_accuracy:.1f} ({len(subset_outcomes)})"
    return row


def _balanced_accuracy_row(
    name: str,
    *,
    runs: list[dict[str, ItemResult]],
    majority: dict[str, bool],
    classes: dict[str, str],
    baseline_majority: dict[str, bool],
    split: dict[str, str],
    dataset: str,
) -> dict[str, str]:
    """ICBHI: balanced accuracy over the primary classes, a stratified bootstrap CI, and a paired
    permutation test. Plain accuracy stays as a secondary column, read against the majority rate."""
    point = balanced_accuracy(majority, classes)
    if point is None:
        logger.warning(f"{name}: no item carries a primary-class label, so balanced accuracy is unavailable.")
    interval = stratified_bootstrap_ba_ci(majority, classes)
    per_run = [balanced_accuracy({i: r.correct for i, r in run.items()}, classes) for run in runs]
    per_run = [value for value in per_run if value is not None]

    row: dict[str, str] = {
        "Experiment": name,
        "n": str(len(majority)),
        "BA (majority)": _fmt(point),
        "BA 95% CI": NOT_AVAILABLE if interval is None else f"[{interval[0]:.1f}, {interval[1]:.1f}]",
        "BA (mean of runs)": _fmt(float(np.mean(per_run)) if per_run else None),
        "Acc (majority)": f"{float(np.mean(list(majority.values()))) * 100:.1f}" if majority else NOT_AVAILABLE,
    }

    if baseline_majority:
        shared = set(majority) & set(baseline_majority)
        ours = balanced_accuracy({i: majority[i] for i in shared}, classes)
        base = balanced_accuracy({i: baseline_majority[i] for i in shared}, classes)
        gained = sum(majority[i] and not baseline_majority[i] for i in shared)
        lost = sum(not majority[i] and baseline_majority[i] for i in shared)
        row["Δ BA vs base"] = NOT_AVAILABLE if ours is None or base is None else f"{ours - base:+.1f}"
        row["gained/lost"] = f"{gained}/{lost}"
        row["p (perm.)"] = f"{paired_permutation_ba(majority, baseline_majority, classes):.3f}"

    for subset in subsets_for(dataset):
        members = {i: majority[i] for i in majority if split.get(i) == subset}
        row[f"{subset} BA (n)"] = f"{_fmt(balanced_accuracy(members, classes))} ({len(members)})"
    return row


def summarize(
    run_dirs: list[str],
    baseline_dir: str | None,
    split: dict[str, str],
    dataset: str = MMAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_majority = majority_correct(load_run_dir(baseline_dir)) if baseline_dir else {}

    rows: list[dict[str, str]] = []
    per_category: dict[str, dict[str, str]] = {}

    for run_dir in run_dirs:
        name = _experiment_name(run_dir, dataset)
        runs = load_run_dir(run_dir)
        majority = majority_correct(runs)
        categories = categories_of(runs)

        if dataset == ICBHI:
            row = _balanced_accuracy_row(
                name,
                runs=runs,
                majority=majority,
                classes=categories,
                baseline_majority=baseline_majority,
                split=split,
                dataset=dataset,
            )
        else:
            row = _accuracy_row(
                name, runs=runs, majority=majority, baseline_majority=baseline_majority, split=split, dataset=dataset
            )

        missing = [item_id for item_id in majority if item_id not in split]
        if split and missing:
            logger.warning(f"{name}: {len(missing)} items are missing from the split file (e.g. {missing[:3]}).")

        rows.append(row)

        by_category: dict[str, list[bool]] = defaultdict(list)
        for item_id, correct in majority.items():
            by_category[categories.get(item_id, UNKNOWN_CATEGORY)].append(correct)
        per_category[name] = {
            category: f"{float(np.mean(values)) * 100:.1f} ({len(values)})"
            for category, values in sorted(by_category.items())
        }

    return pd.DataFrame(rows), pd.DataFrame(per_category).fillna("-")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare experiment run directories.")
    parser.add_argument("--dataset", type=str, choices=[MMAR, ICBHI], default=MMAR)
    parser.add_argument("--run-dirs", nargs="+", default=[], help="Run directories to compare")
    parser.add_argument("--baseline-dir", type=str, default=None, help="Baseline run directory")
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument(
        "--write-split-from",
        type=str,
        default=None,
        help="Derive the subset split from this text-only control run directory, then exit",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    split_file = args.split_file or DEFAULT_SPLIT_FILE[args.dataset]
    output = args.output or DEFAULT_OUTPUT[args.dataset]

    if args.write_split_from:
        write_split(args.write_split_from, split_file, args.dataset)
        return

    if not args.run_dirs:
        parser.error("--run-dirs is required unless --write-split-from is given")

    split = load_split(split_file)
    summary_df, category_df = summarize(args.run_dirs, args.baseline_dir, split, args.dataset)

    summary_md = summary_df.to_markdown(index=False, disable_numparse=True)
    category_md = category_df.to_markdown(disable_numparse=True)

    logger.info(f"\n{summary_md}")
    logger.info(f"\nPer-category majority accuracy:\n{category_md}")

    if args.dataset == ICBHI:
        summary_md = f"{summary_md}\n\n{ICBHI_TABLE_LEGEND}"

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{summary_md}\n\n## Per-category majority accuracy\n\n{category_md}\n")
    logger.info(f"Saved comparison table to {output_path}")


if __name__ == "__main__":
    sys.exit(main())
