import argparse
from pathlib import Path
from loguru import logger

from speech_processing.data.dtos import JudgeRequest

def export_audio_prompts(dataset: str, output_dir: Path):
    """Exports all audio experiment prompts from the ExperimentVersion Enum."""
    logger.info(f"Exporting audio experiment prompts for {dataset}...")
    
    if dataset == "icbhi":
        from speech_processing.runners.icbhi import ExperimentVersion
    elif dataset == "mmar":
        from speech_processing.runners.mmar import ExperimentVersion
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    count = 0
    for exp in ExperimentVersion:
        filename = f"{dataset}_audio_experiment_{exp.name.lower()}.txt"
        file_path = output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Experiment Name: {exp.value.experiment_name}\n")
            f.write("=" * 40 + "\n")
            
            prompt_data = exp.value.prompt
            if isinstance(prompt_data, list):
                for i, p in enumerate(prompt_data):
                    f.write(f"Turn {i+1}: {p.strip()}\n")
            else:
                f.write(prompt_data.strip() + "\n")
                
        count += 1
    logger.info(f"Successfully exported {count} audio prompts to {output_dir}/")


def export_judge_prompts(dataset: str, output_dir: Path):
    """Exports the judge evaluation prompt with placeholder data."""
    logger.info(f"Exporting judge prompt for {dataset}...")
    
    if dataset == "icbhi":
        from speech_processing.prompts.templates.judge.qwen import build_icbhi_judge_conversation as build_judge_conversation
    elif dataset == "mmar":
        from speech_processing.prompts.templates.judge.qwen import build_mmar_judge_conversation as build_judge_conversation
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    conversation = build_judge_conversation(req)
    req = JudgeRequest(
        sample_id="<SAMPLE_ID>",
        instruction="<ORIGINAL_INSTRUCTION>",
        generated_text="<MODEL_ANSWER>",
        ground_truth="<GROUND_TRUTH_LABEL>"
    )
    
    
    
    file_path = output_dir / f"{dataset}_judge_prompt_template.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{dataset.upper()} Judge Evaluation Conversation Template\n")
        f.write("=" * 40 + "\n\n")
        for turn in conversation:
            f.write(f"[{turn['role'].upper()}]\n")
            f.write(f"{turn['content']}\n")
            f.write("-" * 40 + "\n\n")
            
    logger.info(f"Successfully exported 1 judge prompt to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Export prompts dynamically from the actual prompt classes.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=["icbhi", "mmar"], 
        help="Which dataset prompts to export"
    )
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
        export_audio_prompts(args.dataset, output_dir)
        
    if args.type in ("judge", "all"):
        export_judge_prompts(args.dataset, output_dir)
        
    logger.info("Done!")

if __name__ == "__main__":
    main()
