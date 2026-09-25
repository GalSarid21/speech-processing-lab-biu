import json
import os
from loguru import logger

from speech_processing.config.core import (
    AppConfig, 
    DatasetConfig, 
    TextModelConfig, 
    AudioModelConfig, 
    ExperimentMeta
)

def _parse_sample_ids(sample_ids_file: str | None) -> list[str]:
    if not sample_ids_file:
        logger.warning("No sample IDs file provided.")
        return []
    
    sample_ids = []
    if os.path.exists(sample_ids_file):
        logger.info(f"Loading reference sample IDs from {sample_ids_file}")
        
        try:
            with open(sample_ids_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if sample_ids_file.endswith(".jsonl"):
                        data = json.loads(line)
                        sample_ids.append(data["sample_id"])
                    else:
                        sample_ids.append(line)
        except FileNotFoundError:
            logger.error(f"Sample IDs file not found: {sample_ids_file}")
            
        except json.JSONDecodeError:
            logger.error(f"Sample IDs file is not valid JSON: {sample_ids_file}")
        
        except Exception as e:
            logger.error(f"An error occurred while loading sample IDs from {sample_ids_file}: {e}")

    else:
        logger.warning(f"Sample IDs file not found: {sample_ids_file}")
        
    
    return sample_ids
    
def _build_model_kwargs(experiment_meta: ExperimentMeta) -> dict:
    is_text_only = getattr(experiment_meta, "experiment_name", "") == "text_only_llm"
    kwargs = {}
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
    return kwargs

def create_icbhi_config(args, experiment_meta: ExperimentMeta) -> AppConfig:
    """Factory for creating a fully initialized ICBHI configuration."""
    dataset_config = DatasetConfig(
        dataset_id="DynamicSuperb/RespiratorySoundClassification_ICBHI2017",
        target_labels=["COPD", "No potential disease detected", "Healthy"],
        split="test",
        num_samples=args.num_samples,
        sample_ids=_parse_sample_ids(args.sample_ids_file)
    )
    
    kwargs = {
        "output_dir": args.output_dir,
        "dataset": dataset_config,
    }
    kwargs.update(_build_model_kwargs(experiment_meta))
    
    return AppConfig(**kwargs)

def create_mmar_config(args, experiment_meta: ExperimentMeta) -> AppConfig:
    """Factory for creating a fully initialized MMAR configuration."""
    dataset_config = DatasetConfig(
        dataset_id="BoJack/MMAR",
        target_labels=[],
        split="test",
        num_samples=args.num_samples,
        sample_ids=_parse_sample_ids(args.sample_ids_file),
        audio_base_dir="data/MMAR",
        transcripts_file="data/mmar_transcripts.json",
        few_shot_ids_file="data/mmar_en_speech_few_shot_ids.txt"
    )
    
    kwargs = {
        "output_dir": args.output_dir,
        "dataset": dataset_config,
    }
    kwargs.update(_build_model_kwargs(experiment_meta))
    
    return AppConfig(**kwargs)
