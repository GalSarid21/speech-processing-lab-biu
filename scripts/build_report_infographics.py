"""Draws the two explanatory infographics of the report (no data involved, only model facts from §1.1).

* `fig_architecture.png`    - cascade (speech recogniser -> text model) vs end-to-end audio-language model.
* `fig_models_overview.png` - the five models: same-size audio encoder, language models of very different size.
"""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

AUDIO = "#2a78d6"
AUDIO_LIGHT = "#cde2fb"
TEXT_BOX = "#f0efec"
TEXT_EDGE = "#8a8985"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
LOSS = "#d03b3b"
KEEP = "#0ca30c"
SURFACE = "#ffffff"
FAMILY_COLOURS = {"Voxtral": "#2a78d6", "Qwen Omni": "#eb6834", "Step-Audio": "#1baf7a"}


def box(axis, rect, label, *, face, edge, size=9, weight="normal", colour=INK):
    x, y, width, height = rect
    axis.add_patch(
        FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.08", fc=face, ec=edge, lw=1.2)
    )
    axis.text(
        x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=size, fontweight=weight, color=colour
    )


def arrow(axis, start, end):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=INK_SECONDARY, lw=1.2))


def architecture(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.2))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 5.4)
    axis.axis("off")

    # Cascade, top row.
    axis.text(
        0.05,
        5.15,
        "Cascade: speech recogniser, then a text-only language model",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    y = 3.75
    box(axis, (0.1, y, 1.1, 0.8), "audio", face=AUDIO_LIGHT, edge=AUDIO)
    box(axis, (1.6, y, 1.5, 0.8), "Whisper\nencoder", face=AUDIO_LIGHT, edge=AUDIO)
    box(axis, (3.5, y, 1.5, 0.8), "Whisper\ndecoder", face=TEXT_BOX, edge=TEXT_EDGE)
    box(axis, (5.4, y, 1.7, 0.8), 'transcript\n"I\'m fine."', face=TEXT_BOX, edge=TEXT_EDGE)
    box(axis, (7.5, y, 1.5, 0.8), "text-only\nLLM", face=TEXT_BOX, edge=TEXT_EDGE)
    box(axis, (9.3, y + 0.15, 0.55, 0.5), "B", face=SURFACE, edge=INK, weight="bold")
    for x0, x1 in [(1.2, 1.6), (3.1, 3.5), (5.0, 5.4), (7.1, 7.5), (9.0, 9.3)]:
        arrow(axis, (x0, y + 0.4), (x1, y + 0.4))
    axis.text(
        4.25,
        y - 0.32,
        "tone, speaker, emotion, laughter,\nbackground sounds are lost here",
        ha="center",
        va="top",
        fontsize=8.5,
        color=LOSS,
    )

    axis.plot([0.05, 9.95], [2.55, 2.55], color="#e4e3df", lw=1)

    # End-to-end, bottom row.
    axis.text(
        0.05,
        2.25,
        "End-to-end audio-language model (all five models in this study)",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    y = 0.85
    box(axis, (0.1, y, 1.1, 0.8), "audio", face=AUDIO_LIGHT, edge=AUDIO)
    box(axis, (1.6, y, 1.5, 0.8), "audio\nencoder", face=AUDIO_LIGHT, edge=AUDIO)
    box(axis, (3.5, y, 1.1, 0.8), "adapter", face=AUDIO_LIGHT, edge=AUDIO)
    for index in range(5):
        axis.add_patch(Rectangle((5.0 + index * 0.24, y + 0.45), 0.2, 0.25, fc=AUDIO, ec=SURFACE))
    for index in range(4):
        axis.add_patch(Rectangle((5.0 + index * 0.24, y + 0.1), 0.2, 0.25, fc=TEXT_EDGE, ec=SURFACE))
    axis.text(6.25, y + 0.57, "audio tokens", fontsize=8, va="center", color=INK_SECONDARY)
    axis.text(6.25, y + 0.22, "text prompt", fontsize=8, va="center", color=INK_SECONDARY)
    box(axis, (7.5, y, 1.5, 0.8), "language\nmodel", face=AUDIO_LIGHT, edge=AUDIO, weight="bold")
    box(axis, (9.3, y + 0.15, 0.55, 0.5), "B", face=SURFACE, edge=INK, weight="bold")
    for x0, x1 in [(1.2, 1.6), (3.1, 3.5), (4.6, 4.95), (7.1, 7.5), (9.0, 9.3)]:
        arrow(axis, (x0, y + 0.4), (x1, y + 0.4))
    axis.text(
        5.6,
        y - 0.18,
        "no transcript: pitch, voice and timing can still pass",
        ha="center",
        va="top",
        fontsize=8.5,
        color=KEEP,
    )

    figure.savefig(output / "fig_architecture.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)


MODELS = [
    # name, family, LLM size (B), note on the language model, note on the encoder
    ("Voxtral-Mini-3B", "Voxtral", 3, "dense", "from Whisper-large-v3"),
    ("Qwen2.5-Omni-7B", "Qwen Omni", 7, "dense", "from Whisper-large-v3, trained further"),
    ("Voxtral-Small-24B", "Voxtral", 24, "dense", "from Whisper-large-v3"),
    ("Qwen3-Omni-30B-A3B", "Qwen Omni", 30, "MoE: 128 experts, ~3B active per token", "new (AuT), 20M hours"),
    ("Step-Audio-R1.1", "Step-Audio", 32, "dense, trained to reason", "Qwen2-Audio encoder, frozen"),
]


def models_overview(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.4))
    axis.set_xlim(-0.2, 52)
    axis.set_ylim(-0.6, len(MODELS) - 0.2)
    axis.axis("off")
    encoder_width = 0.6 * 1.6  # ~0.6B, drawn wider than scale so it stays visible
    for row, (name, family, size, llm_note, encoder_note) in enumerate(reversed(MODELS)):
        colour = FAMILY_COLOURS[family]
        axis.add_patch(Rectangle((0, row), encoder_width, 0.55, fc=AUDIO_LIGHT, ec=AUDIO, lw=1))
        start = encoder_width + 0.25
        if family == "Qwen Omni" and "MoE" in llm_note:
            cells = 32
            cell = size / cells
            for index in range(cells):
                axis.add_patch(
                    Rectangle(
                        (start + index * cell, row),
                        cell * 0.85,
                        0.55,
                        fc=colour,
                        alpha=1.0 if index % 16 == 0 else 0.22,
                        ec="none",
                    )
                )
        else:
            axis.add_patch(Rectangle((start, row), size, 0.55, fc=colour, alpha=0.85, ec="none"))
        axis.text(-0.4, row + 0.27, name, ha="right", va="center", fontsize=9, fontweight="bold", color=INK)
        inside = "MoE" not in llm_note and size > 12
        axis.text(
            start + 0.3 if inside else start + size + 0.4,
            row + 0.27,
            f"{size}B language model, {llm_note}",
            va="center",
            fontsize=8,
            color=SURFACE if inside else INK,
        )
        axis.text(start, row - 0.12, f"ear: {encoder_note}", va="top", fontsize=7.5, color=INK_SECONDARY)
    axis.text(0, len(MODELS) - 0.15, "audio encoder (~0.6B, same shape in all five)", fontsize=8, color=AUDIO)
    figure.suptitle(
        "Same size of ear, very different brains and ear training", x=0.02, ha="left", fontweight="bold", fontsize=11
    )
    figure.savefig(output / "fig_models_overview.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the report's explanatory infographics.")
    parser.add_argument("--output-dir", default="insights/report/figures")
    output = Path(parser.parse_args().output_dir)
    output.mkdir(parents=True, exist_ok=True)
    architecture(output)
    models_overview(output)


if __name__ == "__main__":
    main()
