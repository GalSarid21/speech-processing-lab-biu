import pandas as pd
from datasets import Audio, load_dataset
from loguru import logger

from speech_processing.config.core import DatasetConfig
from speech_processing.data.dtos import AudioRequest, FewShotTurn

def get_authentic_few_shot_turns(config: DatasetConfig, prompt: str) -> list[FewShotTurn]:
    """Fetches real audio samples from the train split to use as authentic few-shot references."""
    logger.info("Fetching authentic few-shot examples from the train split...")
    ds = load_dataset(config.dataset_id, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    df = ds.to_pandas()

    turns = []
    
    # 1. Healthy Example (Hardcoded ID for reproducibility)
    healthy_df = df[df["file"] == "audio100.wav"]
    if not healthy_df.empty:
        row = healthy_df.iloc[0]
        audio_bytes = row["audio"]["bytes"] if isinstance(row["audio"], dict) and "bytes" in row["audio"] else None
        turns.append(FewShotTurn(
            audio_bytes=audio_bytes,
            audio_path=row["file"],
            user_text=prompt,
            assistant_text="I detect normal vesicular breath sounds without any adventitious sounds like crackles or wheezes. Cross-referencing the dictionary, these features indicate a Healthy patient.\nFinal Diagnosis: Healthy"
        ))

    # 2. COPD Example (Hardcoded ID for reproducibility)
    copd_df = df[df["file"] == "audio101.wav"]
    if not copd_df.empty:
        row = copd_df.iloc[0]
        audio_bytes = row["audio"]["bytes"] if isinstance(row["audio"], dict) and "bytes" in row["audio"] else None
        turns.append(FewShotTurn(
            audio_bytes=audio_bytes,
            audio_path=row["file"],
            user_text=prompt,
            assistant_text="I detect a prolonged expiratory phase along with early inspiratory coarse crackles and expiratory wheezes. Cross-referencing the dictionary, these match the classic acoustic signature of COPD.\nFinal Diagnosis: COPD"
        ))

    logger.info(f"Successfully loaded {len(turns)} authentic few-shot turns.")
    return turns


def load_icbhi_requests(config: DatasetConfig) -> list[tuple[AudioRequest, str]]:
    """Loads the ICBHI dataset, filters it, and returns a list of (AudioRequest, ground_truth)."""
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
        audio_bytes = row["audio"]["bytes"] if isinstance(row["audio"], dict) and "bytes" in row["audio"] else None
        
        req = AudioRequest(
            instruction=row["instruction"], 
            audio_path=row["file"],
            audio_bytes=audio_bytes
        )
        results.append((req, row["label"]))

    logger.info(f"Successfully prepared {len(results)} samples from the dataset.")
    return results
