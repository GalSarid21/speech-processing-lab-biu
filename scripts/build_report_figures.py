"""Draws the report figures from the tables written by build_report_data.py (never from raw runs).

Figures (PNG, into the output folder):

* `fig_words_vs_sound.png`   - each model's baseline accuracy on transcript-solvable (x) vs audio-dependent (y)
  items, with chance lines: reading the words vs hearing the sound.
* `fig_remove_audio.png`     - v3 (audio + transcript) vs v29 (silence + transcript) per model, on both subsets.
* `fig_technique_heatmap.png`- change against the model's own baseline for every technique, split into the
  transcript-solvable and audio-dependent items; cells with a significant paired change are labelled.
* `fig_support_disturb.png`  - one panel per model: each technique placed by its change on the two subsets.
* `fig_categories.png`       - baseline accuracy per MMAR sub-category with at least MIN_CATEGORY_ITEMS items.
* `fig_icbhi_answers.png`    - ICBHI: share of each diagnosis among all answers, with audio vs silence.

Colour: one hue per model family (three validated categorical slots, never cycled); diverging blue-red with a
grey midpoint for changes. Identity is never colour alone: every point is direct-labelled.
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap

# Reference palette (dataviz skill), validated all-pairs for three slots: one per model family.
FAMILY_COLOURS = {"Voxtral": "#2a78d6", "Qwen Omni": "#eb6834", "Step-Audio": "#1baf7a"}
NEUTRAL = "#8a8985"
GRID = "#e4e3df"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#ffffff"
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#d03b3b", "#f0efec", "#2a78d6"])
MODEL_ORDER = ["Voxtral-Mini-3B", "Qwen2.5-Omni-7B", "Voxtral-Small-24B", "Qwen3-Omni-30B-A3B", "Step-Audio-R1.1"]
ICBHI_MODELS = ["Voxtral-Small-24B", "Qwen3-Omni-30B-A3B"]
CHANCE_TS, CHANCE_AD = 31.5, 33.2
# Label offsets (points) that keep the five direct labels of the words-vs-sound scatter apart.
LABEL_OFFSETS = {
    "Voxtral-Mini-3B": (8, -6),
    "Qwen2.5-Omni-7B": (-10, 12),
    "Voxtral-Small-24B": (-12, -26),
    "Qwen3-Omni-30B-A3B": (-14, 16),
    "Step-Audio-R1.1": (10, -14),
}
HEATMAP_LIMIT = 20.0
SIGNIFICANT = 0.05
MIN_CATEGORY_ITEMS = 14

# Technique families, in reading order: (key shown, experiment name).
TECHNIQUES = {
    "Reasoning": [
        ("v2 CoT", "cot"),
        ("v12 describe, then answer", "multi_turn_decomposition"),
        ("v11 expert persona", "role_prompting"),
    ],
    "Added text": [
        ("v3 transcript", "transcript_augmented"),
        ("v4 transcript + CoT", "transcript_augmented_cot"),
        ("v18 diarized transcript", "audio_diarized_transcript"),
        ("v19 acoustic features", "audio_acoustic_features"),
        ("v20 diarized + features", "audio_diarized_features"),
    ],
    "Examples": [
        ("v5 text examples", "few_shot_text_only"),
        ("v6 text examples + CoT", "few_shot_text_only_cot"),
        ("v7 audio examples", "few_shot_audio"),
        ("v8 audio examples + CoT", "few_shot_audio_cot"),
        ("v9 audio ex. + transcript", "few_shot_audio_transcript"),
        ("v10 audio ex. + transcript + CoT", "few_shot_audio_transcript_cot"),
        ("v14 retrieved examples", "rag_few_shots"),
        ("v16 retrieved ex. + transcript", "rag_few_shots_transcript"),
    ],
    "Presentation": [
        ("v22 5-s segments", "chunked_audio"),
        ("v23 locate, then zoom", "two_pass_localization"),
        ("v28 'listen to HOW'", "audio_first_instruction"),
    ],
    "Scoring & voting": [
        ("v24 letter scoring", "contrastive_alpha_0_0"),
        ("v25 contrastive a=0.5", "contrastive_alpha_0_5"),
        ("v26 contrastive a=1.0", "contrastive_alpha_1_0"),
        ("v27 shuffled-option vote", "shuffled_consistency"),
    ],
    "Control": [("v29 transcript, sound removed", "transcript_silence")],
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.labelcolor": TEXT_SECONDARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def save(figure, output: Path, name: str) -> None:
    figure.savefig(output / name, dpi=200, bbox_inches="tight")
    plt.close(figure)
    logger.info(f"Saved {output / name}")


def short(model: str) -> str:
    return model.replace("-A3B", "")


def words_vs_sound(summary: pd.DataFrame, text_only_acc: float, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.0))
    base = summary[summary.experiment == "baseline"].set_index("model")
    for model in MODEL_ORDER:
        row = base.loc[model]
        colour = FAMILY_COLOURS[row.family]
        axis.scatter(row.acc_ts, row.acc_ad, s=70, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3)
        axis.annotate(
            f"{short(model)}\n{row.acc:.1f}% overall",
            (row.acc_ts, row.acc_ad),
            textcoords="offset points",
            xytext=LABEL_OFFSETS[model],
            ha="right" if LABEL_OFFSETS[model][0] < 0 else "left",
            fontsize=8,
            color=TEXT,
        )
    axis.axvline(CHANCE_TS, color=NEUTRAL, lw=1, ls="--")
    axis.axhline(CHANCE_AD, color=NEUTRAL, lw=1, ls="--")
    axis.text(CHANCE_TS + 0.6, 98, "chance, TS 31.5%", color=TEXT_SECONDARY, fontsize=7.5, va="top")
    axis.text(26, CHANCE_AD + 1, "chance, AD 33.2%", color=TEXT_SECONDARY, fontsize=7.5)
    axis.set_xlim(25, 100)
    axis.set_ylim(0, 100)
    axis.set_xlabel("Reading the words: accuracy on transcript-solvable items (%, n = 143)")
    axis.set_ylabel("Hearing the sound: accuracy on audio-dependent items (%, n = 56)")
    axis.set_title("Words vs sound, baseline prompt", loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=7, label=f) for f, c in FAMILY_COLOURS.items()]
    axis.legend(handles=handles, loc="lower right", fontsize=8, title="family", title_fontsize=8)
    axis.text(
        26,
        3,
        f"Text-only model reading the transcript: {text_only_acc:.1f}% overall\n"
        "(100% on TS and 0% on AD by construction of the split)",
        fontsize=7.5,
        color=TEXT_SECONDARY,
    )
    save(figure, output, "fig_words_vs_sound.png")


def remove_audio(summary: pd.DataFrame, output: Path) -> None:
    models = [m for m in MODEL_ORDER if ((summary.model == m) & (summary.experiment == "transcript_silence")).any()]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for axis, (column, title) in zip(
        axes,
        [("acc_ts", "Transcript-solvable items (n = 143)"), ("acc_ad", "Audio-dependent items (n = 56)")],
        strict=True,
    ):
        for row_index, model in enumerate(models):
            rows = summary[summary.model == model].set_index("experiment")
            with_audio, without = rows.loc["transcript_augmented", column], rows.loc["transcript_silence", column]
            colour = FAMILY_COLOURS[rows.family.iloc[0]]
            axis.plot([with_audio, without], [row_index] * 2, color=GRID, lw=3, zorder=1, solid_capstyle="round")
            axis.scatter(with_audio, row_index, s=60, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3)
            axis.scatter(without, row_index, s=60, facecolor=SURFACE, edgecolor=colour, linewidth=2, zorder=3)
            axis.annotate(
                f"{without - with_audio:+.1f}",
                (min(with_audio, without), row_index),
                textcoords="offset points",
                xytext=(-8, -3),
                ha="right",
                fontsize=8,
            )
        axis.axvline(CHANCE_TS if column == "acc_ts" else CHANCE_AD, color=NEUTRAL, lw=1, ls="--")
        axis.set_title(title, loc="left")
        axis.set_xlim(0, 100)
        axis.set_xlabel("accuracy (%)")
        axis.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(models)), [short(m) for m in models])
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=TEXT_SECONDARY, markersize=7, label="audio + transcript (v3)"),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor=SURFACE,
            color=TEXT_SECONDARY,
            markersize=7,
            label="silence + transcript (v29)",
        ),
    ]
    figure.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), fontsize=8)
    figure.suptitle("What changes when the sound is removed but the transcript stays", y=1.1, fontweight="bold")
    save(figure, output, "fig_remove_audio.png")


def technique_rows() -> list[tuple[str, str, str]]:
    return [(family, label, name) for family, entries in TECHNIQUES.items() for label, name in entries]


def technique_heatmap(summary: pd.DataFrame, output: Path) -> None:
    rows = technique_rows()
    columns = [(model, subset) for subset in ("ts", "ad") for model in MODEL_ORDER]
    values = np.full((len(rows), len(columns)), np.nan)
    significant = np.zeros_like(values, dtype=bool)
    indexed = summary.set_index(["model", "experiment"])
    for r, (_, _, name) in enumerate(rows):
        for c, (model, subset) in enumerate(columns):
            if (model, name) in indexed.index:
                values[r, c] = indexed.loc[(model, name), f"d_{subset}"]
                significant[r, c] = indexed.loc[(model, name), f"p_{subset}"] < SIGNIFICANT

    figure, axis = plt.subplots(figsize=(9.6, 9.4))
    axis.grid(False)
    image = axis.imshow(
        np.clip(values, -HEATMAP_LIMIT, HEATMAP_LIMIT),
        cmap=DIVERGING,
        vmin=-HEATMAP_LIMIT,
        vmax=HEATMAP_LIMIT,
        aspect="auto",
    )
    for r in range(len(rows)):
        for c in range(len(columns)):
            if np.isnan(values[r, c]):
                axis.text(c, r, "n/a", ha="center", va="center", color=NEUTRAL, fontsize=8)
            elif significant[r, c]:
                strong = abs(values[r, c]) > HEATMAP_LIMIT * 0.6
                axis.text(
                    c,
                    r,
                    f"{values[r, c]:+.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color=SURFACE if strong else TEXT,
                )
    axis.set_xticks(range(len(columns)), [short(m) for m, _ in columns], rotation=45, ha="right")
    axis.set_yticks(range(len(rows)), [label for _, label, _ in rows])
    axis.axvline(len(MODEL_ORDER) - 0.5, color=SURFACE, lw=6)
    axis.text(len(MODEL_ORDER) / 2 - 0.5, -1.2, "Transcript-solvable items (words)", ha="center", fontweight="bold")
    axis.text(len(MODEL_ORDER) * 1.5 - 0.5, -1.2, "Audio-dependent items (sound)", ha="center", fontweight="bold")
    family_starts = {}
    for index, (family, _, _) in enumerate(rows):
        family_starts.setdefault(family, index)
    for family, start in family_starts.items():
        if start:
            axis.axhline(start - 0.5, color=SURFACE, lw=4)
        axis.text(len(columns) - 0.3, start, family, fontsize=8, color=TEXT_SECONDARY, va="center")
    colourbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.13)
    colourbar.set_label("change vs the model's own baseline (points; clipped at ±20)")
    axis.set_title(
        "Effect of each technique on each capability\n"
        "numbers = significant paired change (McNemar p < 0.05, uncorrected); n/a = not run",
        loc="left",
        pad=26,
    )
    save(figure, output, "fig_technique_heatmap.png")


def support_disturb(summary: pd.DataFrame, output: Path) -> None:
    rows = technique_rows()
    labels = {name: label.split(" ")[0] for _, label, name in rows}
    figure, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(15, 3.8), sharex=True, sharey=True)
    for axis, model in zip(axes, MODEL_ORDER, strict=True):
        data = summary[(summary.model == model) & summary.experiment.isin(labels)]
        colour = FAMILY_COLOURS[data.family.iloc[0]]
        axis.axhline(0, color=NEUTRAL, lw=1)
        axis.axvline(0, color=NEUTRAL, lw=1)
        axis.scatter(data.d_ts, data.d_ad, s=34, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        extremes = data.assign(size=data.d_ts.abs() + data.d_ad.abs()).nlargest(4, "size")
        for _, row in extremes.iterrows():
            axis.annotate(
                labels[row.experiment], (row.d_ts, row.d_ad), textcoords="offset points", xytext=(4, 3), fontsize=7.5
            )
        axis.set_title(short(model), loc="left")
        axis.text(14, 22, "helps both", fontsize=7, color=TEXT_SECONDARY, ha="right")
        axis.text(-24, -26, "hurts both", fontsize=7, color=TEXT_SECONDARY)
    axes[0].set_ylabel("change on audio-dependent items (points)")
    figure.supxlabel(
        "change on transcript-solvable items (points, vs the model's own baseline)", fontsize=9, color=TEXT_SECONDARY
    )
    axes[0].set_xlim(-26, 16)
    axes[0].set_ylim(-28, 24)
    figure.suptitle(
        "Does a technique help reading and hearing together, or trade one for the other? (each dot = one technique)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    save(figure, output, "fig_support_disturb.png")


def categories(items: pd.DataFrame, families: dict[str, str], output: Path) -> None:
    base = items[items.experiment == "baseline"]
    counts = base[base.model == MODEL_ORDER[0]].sub_category.value_counts()
    keep = [c for c in counts.index if counts[c] >= MIN_CATEGORY_ITEMS] + ["Counting and Statistics"]
    figure, axis = plt.subplots(figsize=(9.6, 3.8))
    width = 0.15
    for index, model in enumerate(MODEL_ORDER):
        rows = base[base.model == model]
        shares = [rows[rows.sub_category == c].correct.mean() * 100 for c in keep]
        family = families[model]
        positions = np.arange(len(keep)) + (index - 2) * width
        bars = axis.bar(
            positions,
            shares,
            width * 0.86,
            color=FAMILY_COLOURS[family],
            alpha=1.0 if "Mini" not in model and "2.5" not in model else 0.55,
            label=short(model),
        )
        for bar, share in zip(bars, shares, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                share + 1,
                f"{share:.0f}",
                ha="center",
                fontsize=6.5,
                color=TEXT_SECONDARY,
            )
    axis.set_xticks(range(len(keep)), [f"{c}\n(n = {counts[c]})" for c in keep])
    axis.set_ylabel("baseline accuracy (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="x", visible=False)
    axis.legend(ncol=5, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    axis.set_title(
        "Baseline accuracy by MMAR sub-category (lighter bar = smaller model of the family)", loc="left", pad=28
    )
    save(figure, output, "fig_categories.png")


def icbhi_answers(answers: pd.DataFrame, output: Path) -> None:
    arms = [("baseline", "audio"), ("silence_prior", "silence")]
    order = [
        "COPD",
        "No potential disease detected",
        "Pneumonia",
        "URTI",
        "Bronchiectasis",
        "Bronchiolitis",
        "LRTI",
        "Asthma",
        "(unparsed)",
    ]
    shades = ["#0d366b", "#9ec5f4", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", NEUTRAL]
    true_share = {
        "COPD": 35.1,
        "No potential disease detected": 20.2,
        "Pneumonia": 13.7,
        "URTI": 13.1,
        "Bronchiectasis": 8.9,
        "Bronchiolitis": 7.1,
        "LRTI": 1.2,
        "Asthma": 0.6,
    }
    bars = [("True labels", true_share)]
    for model in ICBHI_MODELS:
        for experiment, condition in arms:
            rows = answers[(answers.model == model) & (answers.experiment == experiment)]
            bars.append((f"{short(model)}, {condition}", dict(zip(rows.answer, rows.share_pct, strict=True))))
    figure, axis = plt.subplots(figsize=(9.6, 3.2))
    axis.grid(False)
    for row, (_, shares) in enumerate(bars):
        left = 0.0
        for answer, colour in zip(order, shades, strict=True):
            share = shares.get(answer, 0.0)
            if share <= 0:
                continue
            axis.barh(row, share, left=left, height=0.62, color=colour, edgecolor=SURFACE, linewidth=1.5)
            if share >= 7:
                axis.text(
                    left + share / 2,
                    row,
                    f"{share:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=SURFACE if colour in ("#0d366b", "#4a3aa7", "#008300") else TEXT,
                )
            left += share
    axis.set_yticks(range(len(bars)), [label for label, _ in bars])
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("share of all answers (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in shades]
    axis.legend(
        handles,
        [o.replace("No potential disease detected", "healthy") for o in order],
        ncol=5,
        fontsize=7.5,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.28),
    )
    axis.set_title("ICBHI: what each model answers, with the recording and with silence", loc="left")
    save(figure, output, "fig_icbhi_answers.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the report figures from build_report_data.py's tables.")
    parser.add_argument("--data-dir", default="insights/report/data")
    parser.add_argument("--output-dir", default="insights/report/figures")
    args = parser.parse_args()
    data, output = Path(args.data_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    style()

    summary = pd.read_csv(data / "mmar_summary.csv")
    items = pd.read_csv(data / "mmar_items.csv")
    text_only = summary[summary.experiment == "text_only_llm"].acc.iloc[0]
    json.loads((data / "mmar_split.json").read_text())  # fail early if the split is missing

    words_vs_sound(summary, text_only, output)
    remove_audio(summary, output)
    technique_heatmap(summary, output)
    support_disturb(summary, output)
    categories(items, dict(zip(summary.model, summary.family, strict=True)), output)
    icbhi_answers(pd.read_csv(data / "icbhi_answers.csv"), output)


if __name__ == "__main__":
    main()
