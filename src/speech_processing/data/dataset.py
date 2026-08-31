import pandas as pd
from datasets import load_dataset, Audio
from loguru import logger

from speech_processing.config.core import DatasetConfig
from speech_processing.data.dtos import AudioRequest


def load_icbhi_requests(config: DatasetConfig) -> list[tuple[AudioRequest, str]]:
    """Loads the ICBHI dataset, filters it, and returns a list of (AudioRequest, ground_truth)."""
    logger.info(f"Loading {config.dataset_id} dataset from HuggingFace...")
    ds = load_dataset(config.dataset_id, split=config.split)
    
    # Cast audio to NOT decode so we can extract the raw bytes safely
    ds = ds.cast_column("audio", Audio(decode=False))

    df = ds.to_pandas()

    # Filter dataset
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
