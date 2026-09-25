import argparse
import json
import os
from loguru import logger
from speech_processing.config.core import AppConfig, DatasetConfig, TextModelConfig, AudioModelConfig, JudgeConfig, ExperimentMeta

def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", type=str, required=True, choices=["icbhi", "mmar"], help="Which dataset pipeline to run")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save artifacts")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of dataset samples to evaluate")
    parser.add_argument("--experiment", type=str, default="v1", help="Which experiment version to run")
    parser.add_argument("--sample-ids-file", type=str, default=None, help="Path to a previous raw_output.jsonl file")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to run the experiment")
    return parser.parse_args()

def create_experiment_config(args, experiment_meta, dataset_id: str, target_labels: list[str] = None) -> AppConfig:
    is_text_only = getattr(experiment_meta, "experiment_name", "") == "text_only_llm"
    target_labels = target_labels or []
    
    dataset_config: DatasetConfig = _create_experiment_dataset_config(args, dataset_id, target_labels)

    kwargs = {
        "output_dir": args.output_dir,
        "dataset": dataset_config,
    }

    # Configure models at instantiation based on experiment meta
    if is_text_only:
        kwargs["text_model"] = TextModelConfig(
            model_id="google/gemma-4-26B-A4B-it",
            dtype="bfloat16",
            max_num_seqs=getattr(experiment_meta, "batch_size", 4),
            max_new_tokens=getattr(experiment_meta, "max_new_tokens", 512),
            max_model_len=8192,
        )
    else:
        kwargs["audio_model"] = AudioModelConfig(
            model_id="Qwen/Qwen2-Audio-7B-Instruct",
            dtype="bfloat16",
            max_num_seqs=getattr(experiment_meta, "batch_size", 8),
            max_new_tokens=getattr(experiment_meta, "max_new_tokens", 256),
            max_model_len=8192,
        )
    
    return AppConfig(**kwargs)

def _create_experiment_dataset_config(args, dataset_id: str, target_labels: list[str] = None) -> DatasetConfig:
    sample_ids = []
    
    if args.sample_ids_file and os.path.exists(args.sample_ids_file):
        logger.info(f"Loading reference sample IDs from {args.sample_ids_file}")
        with open(args.sample_ids_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if args.sample_ids_file.endswith(".jsonl"):
                    data = json.loads(line)
                    sample_ids.append(data["sample_id"])
                else:
                    sample_ids.append(line)

    return DatasetConfig(
        dataset_id=dataset_id,
        target_labels=target_labels,
        split="test",
        num_samples=args.num_samples,
        sample_ids=sample_ids
    )
