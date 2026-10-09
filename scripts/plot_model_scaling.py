"""MMAR accuracy against model size: how far apart the models are vs how far prompting moves each one.

For every model (one results folder each) the newest finished run of every experiment is scored by
majority-vote accuracy. Only experiments present for *all* models are used, so each model's prompting
range covers the same techniques. The figure shows, per model:

* a point at the baseline accuracy;
* a vertical bar from the worst to the best technique (the prompting range);
* lines joining models of the same family, so size is compared within a family;
* dashed reference lines for chance and, if given, the text-only model reading the transcript.

With a split file a second panel repeats this on the audio-dependent items only. A markdown table with
the same numbers is written next to the figure.

This is a project script (it imports the package), so it runs with the project environment.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_runs import AUDIO_DEPENDENT, MMAR, load_run_dir, load_split, majority_correct
from compare_models import latest_finished_runs
from loguru import logger

BASELINE = "baseline"
CHANCE_PCT = 32.0
AUDIO_DEPENDENT_CHANCE_PCT = 33.2
_PCT = 100.0
TICK_MERGE_RATIO = 1.25


def accuracy(correct: dict[str, bool], items: set[str] | None = None) -> float:
    values = [ok for item_id, ok in correct.items() if items is None or item_id in items]
    return float(np.mean(values)) * _PCT if values else float("nan")


def score_models(models: list[dict], items_subset: set[str] | None = None) -> tuple[list[str], list[dict]]:
    """Per model: baseline accuracy and accuracy of every experiment shared by all models."""
    latest = [latest_finished_runs(model["dir"], MMAR) for model in models]
    shared = sorted(set.intersection(*(set(runs) for runs in latest)))
    if BASELINE not in shared:
        raise SystemExit(f"Every results folder needs a finished '{BASELINE}' run.")

    scored = []
    for model, runs in zip(models, latest, strict=True):
        per_arm = {name: accuracy(majority_correct(load_run_dir(str(runs[name]))), items_subset) for name in shared}
        techniques = {name: value for name, value in per_arm.items() if name != BASELINE}
        best = max(techniques, key=techniques.get) if techniques else BASELINE
        worst = min(techniques, key=techniques.get) if techniques else BASELINE
        scored.append(
            {
                **model,
                "baseline": per_arm[BASELINE],
                "best": (best, per_arm[best]),
                "worst": (worst, per_arm[worst]),
            }
        )
    return shared, scored


def _size_ticks(sizes: list[float]) -> tuple[list[float], list[str]]:
    """One tick per size; sizes too close to label apart on a log axis share one tick ("30/32B")."""
    groups: list[list[float]] = []
    for size in sizes:
        if groups and size / groups[-1][-1] < TICK_MERGE_RATIO:
            groups[-1].append(size)
        else:
            groups.append([size])
    positions = [float(np.exp(np.mean(np.log(group)))) for group in groups]
    labels = ["/".join(f"{size:g}" for size in group) + "B" for group in groups]
    return positions, labels


def _panel(axis, scored: list[dict], title: str, chance: float, text_only: float | None) -> None:
    families = sorted({model["family"] for model in scored})
    colors = dict(zip(families, plt.rcParams["axes.prop_cycle"].by_key()["color"], strict=False))
    for family in families:
        members = sorted((m for m in scored if m["family"] == family), key=lambda m: m["params_b"])
        xs = [m["params_b"] for m in members]
        axis.plot(xs, [m["baseline"] for m in members], "-o", color=colors[family], label=family, zorder=3)
        for m in members:
            axis.vlines(m["params_b"], m["worst"][1], m["best"][1], color=colors[family], alpha=0.35, lw=8)
            axis.annotate(
                m["name"], (m["params_b"], m["baseline"]), textcoords="offset points", xytext=(8, -4), fontsize=8
            )
    axis.axhline(chance, ls="--", color="grey", lw=1, label=f"chance ({chance:.1f}%)")
    if text_only is not None:
        axis.axhline(text_only, ls=":", color="black", lw=1, label=f"text-only on transcript ({text_only:.1f}%)")
    axis.set_xscale("log")
    sizes = sorted({m["params_b"] for m in scored})
    axis.set_xticks(*_size_ticks(sizes), minor=False)
    axis.tick_params(axis="x", labelrotation=35)
    axis.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axis.set_xlim(min(sizes) / 1.6, max(sizes) * 2.2)
    axis.set_xlabel("parameters (billions, log scale)")
    axis.set_ylabel("accuracy (%)")
    axis.set_title(title)
    axis.legend(fontsize=8, loc="lower right")


def _table(scored: list[dict], scored_ad: list[dict] | None, num_arms: int) -> str:
    header = "| Model | Family | Params (B) | Baseline | Best technique | Worst technique | Range |"
    divider = "|:--|:--|--:|--:|:--|:--|--:|"
    if scored_ad:
        header += " Baseline, audio-dependent |"
        divider += "--:|"
    lines = [header, divider]
    for index, m in enumerate(scored):
        best, worst = m["best"], m["worst"]
        row = (
            f"| {m['name']} | {m['family']} | {m['params_b']:g} | {m['baseline']:.1f} | "
            f"{best[0]} {best[1]:.1f} ({best[1] - m['baseline']:+.1f}) | "
            f"{worst[0]} {worst[1]:.1f} ({worst[1] - m['baseline']:+.1f}) | {best[1] - worst[1]:.1f} |"
        )
        if scored_ad:
            row += f" {scored_ad[index]['baseline']:.1f} |"
        lines.append(row)
    lines += [
        "",
        f"Accuracy = majority vote over runs. Best/worst are taken over the {num_arms - 1} techniques that every "
        "model ran, so each range covers the same experiments. Range = best minus worst technique.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MMAR accuracy against model size.")
    parser.add_argument(
        "--model",
        nargs=4,
        action="append",
        required=True,
        metavar=("NAME", "FAMILY", "PARAMS_B", "RESULTS_DIR"),
        help="One model: display name, family, parameter count in billions, results folder. Repeat per model.",
    )
    parser.add_argument("--text-only-dir", help="Run folder of the text-only transcript model (reference line)")
    parser.add_argument("--split-file", help="MMAR subset split; adds an audio-dependent panel")
    parser.add_argument("--output", required=True, help="PNG to write; the table goes next to it as .md")
    args = parser.parse_args()

    models = [
        {"name": name, "family": family, "params_b": float(params), "dir": results_dir}
        for name, family, params, results_dir in args.model
    ]
    shared, scored = score_models(models)
    text_only = accuracy(majority_correct(load_run_dir(args.text_only_dir))) if args.text_only_dir else None

    scored_ad = None
    if args.split_file:
        split = load_split(args.split_file)
        _, scored_ad = score_models(models, {item for item, subset in split.items() if subset == AUDIO_DEPENDENT})

    figure, axes = plt.subplots(1, 2 if scored_ad else 1, figsize=(13 if scored_ad else 7, 5), squeeze=False)
    _panel(axes[0][0], scored, "MMAR, all items", CHANCE_PCT, text_only)
    if scored_ad:
        _panel(axes[0][1], scored_ad, "MMAR, audio-dependent items", AUDIO_DEPENDENT_CHANCE_PCT, None)
    figure.suptitle("Point = baseline; bar = range over prompting techniques")
    figure.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    table = _table(scored, scored_ad, len(shared))
    output.with_suffix(".md").write_text(table)
    output.with_suffix(".json").write_text(json.dumps({"experiments": shared, "models": scored}, indent=2))
    sys.stdout.write(table)
    logger.info(f"Saved {output} and {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
