"""Compares two audio models experiment by experiment, on the same items.

Each results folder holds run directories named `<dataset>_<experiment_name>_<timestamp>`; the newest
finished run of each experiment is used. MMAR is compared on majority-vote accuracy with an exact McNemar
test, ICBHI on balanced accuracy with a paired permutation test - the same statistics analyze_runs.py
reports against a baseline - and the p-values are Holm-corrected across experiments.

Every row also reports each model's most common answer (as option text, so shuffled option orders still
count as one answer) and its share of all answers: a model that gives one answer to everything scores
exactly 1/k balanced accuracy, which can look like a significant win over a model that is worse than
constant, and this column makes that visible.

Every row also reports how often each model answered in a CJK script instead of English, and the report
ends with verbatim examples - answers written in Chinese first, then English answers that merely quote a
Chinese word from the audio - so a model drifting out of English (Qwen on reasoning prompts) can be traced
to the exact experiment and item.

This is a project script (it imports the package), so it runs with the project environment.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from analyze_runs import (
    ICBHI,
    MMAR,
    balanced_accuracy,
    categories_of,
    item_key,
    load_run_dir,
    majority_correct,
    mcnemar_exact,
    paired_permutation_ba,
)
from loguru import logger

from speech_processing.evaluation.answer_parsing import parse_choice_letter
from speech_processing.evaluation.language import contains_cjk, is_non_english
from speech_processing.runners.base import RUN_CONFIG_FILE

FINISHED_MARKERS = ("metrics.txt", "stability_report.json")
UNRECORDED_MODEL = "unrecorded (run predates run_config.json)"
UNPARSED_ANSWER = "(unparsed)"
EXCERPT_CHARS = 200
DEFAULT_EXAMPLES = 3
COLLAPSE_PCT = 90.0
_PCT = 100.0


def latest_finished_runs(results_dir: str, dataset: str) -> dict[str, Path]:
    """experiment_name -> newest run folder that finished. Crashed runs (no final marker) are skipped."""
    pattern = re.compile(rf"{dataset}_(.+)_(\d{{8}}_\d{{6}})")
    latest: dict[str, Path] = {}
    for run_dir in sorted(Path(results_dir).iterdir()):
        match = pattern.fullmatch(run_dir.name)
        if match and any((run_dir / marker).exists() for marker in FINISHED_MARKERS):
            latest[match.group(1)] = run_dir
    return latest


def audio_model_of(run_dir: Path) -> str:
    config_file = run_dir / RUN_CONFIG_FILE
    if not config_file.exists():
        return UNRECORDED_MODEL
    return json.loads(config_file.read_text()).get("audio_model_id") or UNRECORDED_MODEL


def _records(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for raw_file in sorted(run_dir.glob("raw_output*.jsonl"))
        for line in raw_file.read_text().splitlines()
        if line.strip()
    ]


def language_check(run_dir: Path) -> dict:
    """Over all runs: share of non-English answers, plus (item_id, excerpt) for every answer with CJK text."""
    total = 0
    non_english: list[tuple[str, str]] = []
    quotes_cjk: list[tuple[str, str]] = []
    for record in _records(run_dir):
        total += 1
        text = record.get("generated_text") or record.get("final_turn_text") or ""
        if not contains_cjk(text):
            continue
        target = non_english if is_non_english(text) else quotes_cjk
        target.append((item_key(record), text.replace("\n", " ")[:EXCERPT_CHARS]))
    return {
        "pct": len(non_english) / total * _PCT if total else 0.0,
        "non_english": non_english,
        "quotes_cjk": quotes_cjk,
    }


def answer_text(record: dict) -> str:
    """The chosen option's text, or UNPARSED_ANSWER. Text, not letter: shuffled arms reorder the options."""
    choices = list((record.get("metadata") or {}).get("choices") or [])
    letter = parse_choice_letter(record.get("final_turn_text") or record.get("generated_text") or "", choices)
    index = ord(letter) - ord("A") if letter else -1
    return choices[index] if 0 <= index < len(choices) else UNPARSED_ANSWER


def top_answer(run_dir: Path) -> tuple[str, float]:
    """Most common answer over all runs, and its share of all answers in percent."""
    counts = Counter(answer_text(record) for record in _records(run_dir))
    if not counts:
        return UNPARSED_ANSWER, 0.0
    answer, count = counts.most_common(1)[0]
    return answer, count / sum(counts.values()) * _PCT


def holm(p_values: list[float]) -> list[float]:
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(np.argsort(p_values)):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def compare_experiment(dataset: str, run_a: Path, run_b: Path) -> dict:
    runs_a, runs_b = load_run_dir(str(run_a)), load_run_dir(str(run_b))
    correct_a, correct_b = majority_correct(runs_a), majority_correct(runs_b)
    shared = sorted(set(correct_a) & set(correct_b))
    correct_a = {i: correct_a[i] for i in shared}
    correct_b = {i: correct_b[i] for i in shared}

    if dataset == ICBHI:
        classes = categories_of(runs_a)
        score_a = balanced_accuracy(correct_a, classes)
        score_b = balanced_accuracy(correct_b, classes)
        p_value = paired_permutation_ba(correct_b, correct_a, classes)
    else:
        score_a = float(np.mean(list(correct_a.values()))) * _PCT
        score_b = float(np.mean(list(correct_b.values()))) * _PCT
        gained = sum(correct_b[i] and not correct_a[i] for i in shared)
        lost = sum(correct_a[i] and not correct_b[i] for i in shared)
        p_value = mcnemar_exact(gained, lost)

    return {
        "n": len(shared),
        "score_a": score_a,
        "score_b": score_b,
        "p": p_value,
        "lang_a": language_check(run_a),
        "lang_b": language_check(run_b),
        "top_a": top_answer(run_a),
        "top_b": top_answer(run_b),
    }


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def build_report(dataset: str, *, dir_a: str, name_a: str, dir_b: str, name_b: str, num_examples: int) -> str:
    latest_a = latest_finished_runs(dir_a, dataset)
    latest_b = latest_finished_runs(dir_b, dataset)
    shared = sorted(set(latest_a) & set(latest_b))
    if not shared:
        raise SystemExit(f"No experiment has a finished run in both {dir_a} and {dir_b}.")

    rows = {name: compare_experiment(dataset, latest_a[name], latest_b[name]) for name in shared}
    for name, adjusted in zip(shared, holm([rows[name]["p"] for name in shared]), strict=True):
        rows[name]["holm"] = adjusted

    metric = "BA" if dataset == ICBHI else "Acc"
    models_a = sorted({audio_model_of(latest_a[name]) for name in shared})
    models_b = sorted({audio_model_of(latest_b[name]) for name in shared})
    lines = [
        f"# {dataset.upper()}: {name_a} vs {name_b}",
        "",
        f"- **{name_a}** ({dir_a}): audio model(s) {', '.join(models_a)}",
        f"- **{name_b}** ({dir_b}): audio model(s) {', '.join(models_b)}",
        "",
    ]
    if len(models_a) > 1 or len(models_b) > 1:
        lines += ["**Warning: a results folder mixes audio models - check run_config.json in each run.**", ""]

    lines += [
        f"| Experiment | n | {metric} {name_a} | {metric} {name_b} | Δ ({name_b} - {name_a}) | p | Holm "
        f"| top answer % {name_a} | top answer % {name_b} | non-English % {name_a} | non-English % {name_b} |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name in shared:
        row = rows[name]
        delta = None if row["score_a"] is None or row["score_b"] is None else row["score_b"] - row["score_a"]
        lines.append(
            f"| {name} | {row['n']} | {_fmt(row['score_a'])} | {_fmt(row['score_b'])} | "
            f"{'-' if delta is None else f'{delta:+.1f}'} | {row['p']:.3f} | {row['holm']:.3f} | "
            f"{row['top_a'][1]:.0f} | {row['top_b'][1]:.0f} | "
            f"{row['lang_a']['pct']:.1f} | {row['lang_b']['pct']:.1f} |"
        )
    lines += [
        "",
        "top answer % = share of all answers (all runs) that are the single most common answer. Near 100% means "
        "the model gave one answer to everything; its score then says nothing about the audio.",
    ]
    collapsed = [
        (name, model, top[0], top[1])
        for name in shared
        for model, top in ((name_a, rows[name]["top_a"]), (name_b, rows[name]["top_b"]))
        if top[1] >= COLLAPSE_PCT
    ]
    if collapsed:
        lines += ["", f"## Arms where one answer makes up at least {COLLAPSE_PCT:.0f}% of all answers", ""]
        lines += [f"- {name} / {model}: {answer!r} ({pct:.0f}%)" for name, model, answer, pct in collapsed]

    only_a, only_b = sorted(set(latest_a) - set(latest_b)), sorted(set(latest_b) - set(latest_a))
    if only_a or only_b:
        lines += [
            "",
            f"Only in {name_a}: {', '.join(only_a) or 'none'}. Only in {name_b}: {', '.join(only_b) or 'none'}.",
        ]

    for title, kind, empty in (
        (
            "Answers written in a CJK script (language switch)",
            "non_english",
            "None: no answer switched out of English.",
        ),
        ("English answers quoting CJK text from the audio (not a switch)", "quotes_cjk", "None."),
    ):
        lines += ["", f"## {title}", ""]
        flagged = [
            (name, model, rows[name][key][kind])
            for name in shared
            for model, key in ((name_a, "lang_a"), (name_b, "lang_b"))
            if rows[name][key][kind]
        ]
        if not flagged:
            lines.append(empty)
        for name, model, examples in flagged:
            lines.append(f"**{name} / {model}**: {len(examples)} answer(s)")
            lines += [f"- `{item_id}`: {excerpt!r}" for item_id, excerpt in examples[:num_examples]]
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two audio models experiment by experiment.")
    parser.add_argument("--dataset", choices=[MMAR, ICBHI], required=True)
    parser.add_argument("--a-dir", required=True, help="Results folder of the reference model")
    parser.add_argument("--a-name", required=True)
    parser.add_argument("--b-dir", required=True, help="Results folder of the model being compared")
    parser.add_argument("--b-name", required=True)
    parser.add_argument("--output", required=True, help="Markdown file to write")
    parser.add_argument(
        "--examples", type=int, default=DEFAULT_EXAMPLES, help="Examples listed per experiment and model"
    )
    args = parser.parse_args()

    report = build_report(
        args.dataset,
        dir_a=args.a_dir,
        name_a=args.a_name,
        dir_b=args.b_dir,
        name_b=args.b_name,
        num_examples=args.examples,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    sys.stdout.write(report)
    logger.info(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
