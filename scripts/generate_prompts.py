import argparse
import json
from pathlib import Path

from loguru import logger

from speech_processing.data.dtos import JudgeRequest
from speech_processing.prompts.templates.audio.experiments import ExperimentVersion
from speech_processing.prompts.templates.judge.qwen_judge import QwenICBHI2017JudgeTemplate


def export_audio_prompts(output_dir: Path):
    """Exports all audio experiment prompts from the ExperimentVersion Enum."""
    logger.info("Exporting audio experiment prompts...")
    count = 0
    for exp in ExperimentVersion:
        filename = f"audio_experiment_{exp.name.lower()}.txt"
        file_path = output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Experiment Name: {exp.value.experiment_name}\n")
            f.write("=" * 40 + "\n")
            f.write(exp.value.prompt.strip() + "\n")
        count += 1
    logger.info(f"Successfully exported {count} audio prompts to {output_dir}/")


def export_judge_prompts(output_dir: Path):
    """Exports the judge evaluation prompt with placeholder data."""
    logger.info("Exporting judge prompt...")
    template = QwenICBHI2017JudgeTemplate()
    req = JudgeRequest(
        sample_id="<SAMPLE_ID>",
        instruction="<ORIGINAL_INSTRUCTION>",
        generated_text="<MODEL_ANSWER>",
        ground_truth="<GROUND_TRUTH_LABEL>"
    )
    
    conversation = template.build_conversation(req)
    
    file_path = output_dir / "judge_prompt_template.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Judge Evaluation Conversation Template\n")
        f.write("=" * 40 + "\n\n")
        for turn in conversation:
            f.write(f"[{turn['role'].upper()}]\n")
            f.write(f"{turn['content']}\n")
            f.write("-" * 40 + "\n\n")
            
    logger.info(f"Successfully exported 1 judge prompt to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Export prompts dynamically from the actual prompt classes.")
    parser.add_argument(
        "--type", 
        type=str, 
        choices=["audio", "judge", "all"], 
        default="all", 
        help="Which type of prompts to export (audio, judge, or all)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/exported_prompts", 
        help="Directory to save the exported text files"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.type in ("audio", "all"):
        export_audio_prompts(output_dir)
        
    if args.type in ("judge", "all"):
        export_judge_prompts(output_dir)
        
    logger.info("Done!")

if __name__ == "__main__":
    main()
