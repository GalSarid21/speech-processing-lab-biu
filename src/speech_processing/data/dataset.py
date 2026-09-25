import pandas as pd
from datasets import Audio, load_dataset
from loguru import logger

from speech_processing.config.core import DatasetConfig
from speech_processing.data.dtos import AudioRequest,FewShotTurn, TextRequest, BaseRequest

def get_authentic_few_shot_turns(config: DatasetConfig, prompt: str, experiment_name: str) -> list[FewShotTurn]:
    """Fetches real audio samples from the test split to use as authentic few-shot references."""
    logger.info("Fetching authentic few-shot examples from the test split...")
    ds = load_dataset(config.dataset_id, split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    df = ds.to_pandas()

    turns = []
    
    few_shot_configs = [
        {
            "file": "audio100.wav",
            "class_name": "Healthy",
            "assistant_text": "I detect normal vesicular breath sounds without any adventitious sounds like crackles or wheezes. Cross-referencing the dictionary, these features indicate a Healthy patient.\nFinal Diagnosis: Healthy"
        },
        {
            "file": "audio101.wav",
            "class_name": "COPD",
            "assistant_text": "I detect a prolonged expiratory phase along with early inspiratory coarse crackles and expiratory wheezes. Cross-referencing the dictionary, these match the classic acoustic signature of COPD.\nFinal Diagnosis: COPD"
        },
        {
            "file": "audio133.wav",
            "class_name": "Pneumonia",
            "assistant_text": "I detect localized late inspiratory fine crackles and bronchial breath sounds. Cross-referencing the dictionary, these match the classic acoustic signature of Pneumonia.\nFinal Diagnosis: Pneumonia"
        },
        {
            "file": "audio134.wav",
            "class_name": "URTI",
            "assistant_text": "I detect transmitted upper airway sounds, stridor, and coarse transmitted rhonchi. Cross-referencing the dictionary, these match the classic acoustic signature of URTI.\nFinal Diagnosis: URTI"
        },
        {
            "file": "audio131.wav",
            "class_name": "Bronchiectasis",
            "assistant_text": "I detect early and mid-inspiratory coarse crackles. Cross-referencing the dictionary, these match the classic acoustic signature of Bronchiectasis.\nFinal Diagnosis: Bronchiectasis"
        },
        {
            "file": "audio16.wav",
            "class_name": "Bronchiolitis",
            "assistant_text": "I detect high-pitched expiratory wheezes and fine inspiratory crackles. Cross-referencing the dictionary, these match the classic acoustic signature of Bronchiolitis.\nFinal Diagnosis: Bronchiolitis"
        }
    ]

    if "proportional" in experiment_name:
        few_shot_configs.extend([
            {
                "file": "audio109.wav",
                "class_name": "COPD",
                "assistant_text": "I detect a prolonged expiratory phase along with early inspiratory coarse crackles and expiratory wheezes. Cross-referencing the dictionary, these match the classic acoustic signature of COPD.\nFinal Diagnosis: COPD"
            },
            {
                "file": "audio111.wav",
                "class_name": "COPD",
                "assistant_text": "I detect a prolonged expiratory phase along with early inspiratory coarse crackles and expiratory wheezes. Cross-referencing the dictionary, these match the classic acoustic signature of COPD.\nFinal Diagnosis: COPD"
            },
            {
                "file": "audio124.wav",
                "class_name": "COPD",
                "assistant_text": "I detect a prolonged expiratory phase along with early inspiratory coarse crackles and expiratory wheezes. Cross-referencing the dictionary, these match the classic acoustic signature of COPD.\nFinal Diagnosis: COPD"
            },
            {
                "file": "audio118.wav",
                "class_name": "Healthy",
                "assistant_text": "I detect normal vesicular breath sounds without any adventitious sounds like crackles or wheezes. Cross-referencing the dictionary, these features indicate a Healthy patient.\nFinal Diagnosis: Healthy"
            }
        ])

    for config in few_shot_configs:
        sample_df = df[df["file"] == config["file"]]
        if not sample_df.empty:
            row = sample_df.iloc[0]
            audio_bytes = row["audio"]["bytes"] if isinstance(row["audio"], dict) and "bytes" in row["audio"] else None
            turns.append(FewShotTurn(
                audio_bytes=audio_bytes,
                audio_path=row["file"],
                user_text=prompt,
                assistant_text=config["assistant_text"]
            ))

    logger.info(f"Successfully loaded {len(turns)} authentic few-shot turns.")
    return turns


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
    else:
        # Filter dataset by labels and pad if necessary
        filtered_df = df[
            df["label"].str.contains("|".join(config.target_labels), case=False, na=False)
        ]

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
    logger.info("Loading BoJack/MMAR dataset...")
    ds = load_dataset('BoJack/MMAR', split='test', streaming=False)
    df = ds.to_pandas()
    
    # Check if a custom subset is provided via config (or hardcoded for now)
    target_ids_file = "data/mmar_en_speech_test_ids.txt"
    if config.sample_ids:
        target_ids = set(config.sample_ids)
    elif os.path.exists(target_ids_file):
        with open(target_ids_file, "r") as f:
            target_ids = set(line.strip() for line in f if line.strip())
    else:
        # Fallback to taking N samples
        target_ids = set(df['id'].head(config.num_samples).tolist())

    sample_df = df[df['id'].isin(target_ids)]
    
    # If config says num_samples = 100, we can take head(100)
    if len(sample_df) > config.num_samples:
        sample_df = sample_df.sample(n=config.num_samples, random_state=42)

    # Load transcripts if they exist
    transcripts = {}
    transcripts_file = "data/mmar_transcripts.json"
    if os.path.exists(transcripts_file):
        with open(transcripts_file, "r") as f:
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
        audio_path = os.path.join("data/MMAR", audio_rel_path.lstrip('./'))
        
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
