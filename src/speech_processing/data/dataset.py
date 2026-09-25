import pandas as pd
from datasets import Audio, load_dataset
from loguru import logger

from speech_processing.config.core import DatasetConfig
from speech_processing.data.dtos import AudioRequest,FewShotTurn, TextRequest, BaseRequest

def load_icbhi_requests(config: DatasetConfig, is_text_only: bool = False) -> list[tuple[BaseRequest, str]]:
    """Loads the ICBHI dataset, filters it, and returns a list of (Request, ground_truth)."""
    logger.info(f"Loading {config.dataset_id} dataset from HuggingFace...")
    ds = load_dataset(config.dataset_id, split=config.split)
    
    # Cast audio to NOT decode so we can extract the raw bytes safely
    ds = ds.cast_column("audio", Audio(decode=False))

    df = ds.to_pandas()

    if config.sample_ids:
        # Filter explicitly by the provided sample IDs
        sample_df = df[df["file"].isin(config.sample_ids)]
        found_ids = set(sample_df["file"].tolist())
        missing_ids = set(config.sample_ids) - found_ids
        
        if missing_ids:
            logger.warning(f"Could not find {len(missing_ids)} specified sample IDs (e.g. {list(missing_ids)[:3]})")
        else:
            logger.info("100% of specified sample IDs were successfully fetched.")
            
        if config.num_samples is not None:
            if config.num_samples > len(sample_df):
                logger.warning(f"Requested {config.num_samples} samples but only {len(sample_df)} available in list. Using all.")
            else:
                sample_df = sample_df.sample(n=config.num_samples, random_state=42)
    else:
        # Filter dataset by labels and pad if necessary
        filtered_df = df[
            df["label"].str.contains("|".join(config.target_labels), case=False, na=False)
        ]

        if config.num_samples is None:
            sample_df = filtered_df
        else:
            num_samples = config.num_samples
            if len(filtered_df) < num_samples:
                logger.warning(
                    f"Only found {len(filtered_df)} matching samples. Padding with disjoint samples."
                )
                # Take all filtered, then pad from the inverse subset to avoid duplicates
                remaining_needed = num_samples - len(filtered_df)
                inverse_df = df[~df.index.isin(filtered_df.index)]
                pad_df = inverse_df.sample(n=min(remaining_needed, len(inverse_df)), random_state=42)
                sample_df = pd.concat([filtered_df, pad_df])
            else:
                sample_df = filtered_df.sample(n=num_samples, random_state=42)

    results = []
    for _, row in sample_df.iterrows():
        if is_text_only:
            req = TextRequest(
                instruction=row["instruction"]
            )
        else:
            audio_bytes = row["audio"]["bytes"] if isinstance(row["audio"], dict) and "bytes" in row["audio"] else None
            req = AudioRequest(
                instruction=row["instruction"], 
                audio_path=row["file"],
                audio_bytes=audio_bytes
            )
        results.append((req, row["label"]))

    logger.info(f"Successfully prepared {len(results)} samples from the dataset.")
    return results

def load_mmar_requests(config: DatasetConfig, experiment_meta=None, is_text_only: bool = False) -> list[tuple[BaseRequest, str]]:
    """Loads the MMAR dataset test split (from text files and HuggingFace)."""
    import os
    import json
    logger.info(f"Loading {config.dataset_id} dataset from HuggingFace...")
    ds = load_dataset(config.dataset_id, split=config.split, streaming=False)
    df = ds.to_pandas()
    
    if config.sample_ids:
        target_ids = set(config.sample_ids)
    else:
        # Fallback to taking N samples
        if config.num_samples is None:
            target_ids = set(df['id'].tolist())
        else:
            target_ids = set(df['id'].head(config.num_samples).tolist())

    sample_df = df[df['id'].isin(target_ids)]

    if config.num_samples is not None:
        if config.num_samples > len(sample_df):
            logger.warning(f"Requested {config.num_samples} samples but only {len(sample_df)} available. Using all.")
        else:
            sample_df = sample_df.sample(n=config.num_samples, random_state=42)

    # Load transcripts if they exist
    transcripts = {}
    if config.transcripts_file and os.path.exists(config.transcripts_file):
        with open(config.transcripts_file, "r") as f:
            transcripts = json.load(f)

    results = []
    for _, row in sample_df.iterrows():
        item_id = row['id']
        question = row['question']
        choices = row['choices']
        if isinstance(choices, str):
            import ast
            try:
                choices = ast.literal_eval(choices)
            except (SyntaxError, ValueError):
                pass
        
        answer = row['answer']
        
        audio_rel_path = row['audio_path']
        audio_base = config.audio_base_dir or ""
        audio_path = os.path.join(audio_base, audio_rel_path.lstrip('./'))
        
        transcript = transcripts.get(item_id, "[NO TRANSCRIPT]")
        
        if experiment_meta:
            formatted_instruction = experiment_meta.prompt
            if "transcript" in experiment_meta.experiment_name or "text_only" in experiment_meta.experiment_name:
                formatted_instruction += f"\n\nTranscript: {transcript}"
            formatted_instruction += f"\n\nQuestion: {question}\nChoices: {choices}"
        else:
            formatted_instruction = f"Question: {question}\nChoices: {choices}"
        
        if is_text_only:
            req = TextRequest(
                instruction=formatted_instruction,
                metadata={
                    "question": question,
                    "choices": choices,
                    "transcript": transcript
                }
            )
        else:
            req = AudioRequest(
                instruction=formatted_instruction,
                audio_path=audio_path,
                metadata={
                    "question": question,
                    "choices": choices,
                    "transcript": transcript
                }
            )
        results.append((req, answer))

    logger.info(f"Successfully prepared {len(results)} MMAR samples.")
    return results
