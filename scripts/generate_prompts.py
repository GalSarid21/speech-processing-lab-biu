import argparse
from enum import EnumMeta
from pathlib import Path

from loguru import logger

from speech_processing.data.dtos import JudgeRequest
from speech_processing.prompts.templates.judge.qwen import (
    build_icbhi_judge_conversation,
    build_mmar_judge_conversation,
)
from speech_processing.runners.icbhi import ExperimentVersion as IcbhiExperimentVersion
from speech_processing.runners.mmar import ExperimentVersion as MmarExperimentVersion

EXPERIMENT_REGISTRIES: dict[str, EnumMeta] = {
    "icbhi": IcbhiExperimentVersion,
    "mmar": MmarExperimentVersion,
}
JUDGE_TEMPLATES = {
    "icbhi": build_icbhi_judge_conversation,
    "mmar": build_mmar_judge_conversation,
}
SEPARATOR = "=" * 40
TURN_SEPARATOR = "-" * 40


def export_audio_prompts(dataset: str, output_dir: Path) -> None:
    """Exports every active experiment prompt from the dataset's ExperimentVersion enum."""
    logger.info(f"Exporting audio experiment prompts for {dataset}...")
    registry = EXPERIMENT_REGISTRIES[dataset]

    count = 0
    for exp in registry:
        if exp.value.deprecated:
            logger.info(f"Skipping deprecated experiment {exp.name} ({exp.value.experiment_name}).")
            continue

        file_path = output_dir / f"{dataset}_audio_experiment_{exp.name.lower()}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Experiment Name: {exp.value.experiment_name}\n")
            f.write(SEPARATOR + "\n")

            prompt_data = exp.value.prompt
            if isinstance(prompt_data, list):
                for i, turn in enumerate(prompt_data, start=1):
                    f.write(f"Turn {i}: {turn.strip()}\n")
            else:
                f.write(prompt_data.strip() + "\n")

        count += 1
    logger.info(f"Successfully exported {count} audio prompts to {output_dir}/")


def export_judge_prompts(dataset: str, output_dir: Path) -> None:
    """Exports the judge evaluation prompt with placeholder data."""
    logger.info(f"Exporting judge prompt for {dataset}...")

    req = JudgeRequest(
        sample_id="<SAMPLE_ID>",
        instruction="<ORIGINAL_INSTRUCTION>",
        generated_text="<MODEL_ANSWER>",
        final_turn_text="<MODEL_FINAL_TURN>",
        ground_truth="<GROUND_TRUTH_LABEL>",
    )
    conversation = JUDGE_TEMPLATES[dataset](req)

    file_path = output_dir / f"{dataset}_judge_prompt_template.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{dataset.upper()} Judge Evaluation Conversation Template\n")
        f.write(SEPARATOR + "\n\n")
        for turn in conversation:
            f.write(f"[{turn['role'].upper()}]\n")
            f.write(f"{turn['content']}\n")
            f.write(TURN_SEPARATOR + "\n\n")

    logger.info(f"Successfully exported 1 judge prompt to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export prompts dynamically from the actual prompt classes.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(EXPERIMENT_REGISTRIES),
        help="Which dataset prompts to export",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["audio", "judge", "all"],
        default="all",
        help="Which type of prompts to export (audio, judge, or all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/exported_prompts",
        help="Directory to save the exported text files",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.type in ("audio", "all"):
        export_audio_prompts(args.dataset, output_dir)

    if args.type in ("judge", "all"):
        export_judge_prompts(args.dataset, output_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
