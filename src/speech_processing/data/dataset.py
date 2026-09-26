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
                assistant_prefill="<analysis>\n" if "<analysis>" in str(row["instruction"]) else None, 
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
            prompt_def = experiment_meta.prompt
            
            # Create a deep copy to avoid mutating the Enum's singleton config
            if isinstance(prompt_def, list):
                formatted_instruction = prompt_def.copy()
            else:
                formatted_instruction = prompt_def
                
            suffix = ""
            if "transcript" in experiment_meta.experiment_name or "text_only" in experiment_meta.experiment_name:
                suffix += f"\n\nTranscript: {transcript}"
            suffix += f"\n\nQuestion: {question}\nChoices: {choices}"
            
            if isinstance(formatted_instruction, list):
                # Append the question and choices to the LAST turn in the multi-turn list
                formatted_instruction[-1] += suffix
            else:
                formatted_instruction += suffix
        else:
            formatted_instruction = f"Question: {question}\nChoices: {choices}"
        
        if is_text_only:
            req = TextRequest(
                instruction=formatted_instruction,
                assistant_prefill="<analysis>\n" if "<analysis>" in str(formatted_instruction) else None,
                metadata={
                    "question": question,
                    "choices": choices,
                    "transcript": transcript
                }
            )
        else:
            req = AudioRequest(
                instruction=formatted_instruction,
                assistant_prefill="<analysis>\n" if "<analysis>" in str(formatted_instruction) else None,
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

def get_mmar_few_shot_turns(config: DatasetConfig, experiment_meta, is_text_only: bool = False, num_shots: int = 3) -> list[FewShotTurn]:
    """Loads and formats few-shot examples for MMAR experiments."""
    import os
    import json
    
    if not config.few_shot_ids_file or not os.path.exists(config.few_shot_ids_file):
        logger.warning("No few-shot IDs file found. Cannot load few-shot examples.")
        return []
        
    with open(config.few_shot_ids_file, "r") as f:
        few_shot_ids = [line.strip() for line in f if line.strip()][:num_shots]
        
    ds = load_dataset(config.dataset_id, split=config.split, streaming=False)
    df = ds.to_pandas()
    
    sample_df = df[df['id'].isin(few_shot_ids)]
    
    # Load transcripts if they exist
    transcripts = {}
    if config.transcripts_file and os.path.exists(config.transcripts_file):
        with open(config.transcripts_file, "r") as f:
            transcripts = json.load(f)
            
    turns = []
    # Ensure they are added in the exact order requested
    for target_id in few_shot_ids:
        row_subset = sample_df[sample_df['id'] == target_id]
        if row_subset.empty:
            continue
        row = row_subset.iloc[0]
        
        item_id = row['id']
        question = row['question']
        choices = row['choices']
        if isinstance(choices, str):
            import ast
            try: choices = ast.literal_eval(choices)
            except (SyntaxError, ValueError): pass
        
        answer = row['answer']
        audio_rel_path = row['audio_path']
        audio_base = config.audio_base_dir or ""
        audio_path = os.path.join(audio_base, audio_rel_path.lstrip('./'))
        
        transcript = transcripts.get(item_id, "[NO TRANSCRIPT]")
        
        if "transcript" in experiment_meta.experiment_name or "text_only" in experiment_meta.experiment_name:
            user_text = f"Transcript: {transcript}\n\nQuestion: {question}\nChoices: {choices}"
        else:
            user_text = f"Question: {question}\nChoices: {choices}"

        fake_cots = {
            "KodXqxwrFiE_00-00-00_00-00-19": "Logically, I hear a clear instructional voice pointing out a specific type of fruit. Acoustically, the speech is direct, well-articulated, and recorded in a quiet environment. Based on this, the speaker specifically points out blueberries.",
            "BV1Gt42157zM_00-00-00_00-00-17": "Logically, a girl speaks first, followed by a boy imitating her pronunciation. Acoustically, the boy artificially alters his pitch and vowel sounds to perform a mock accent. Based on this, the boy is imitating a British accent.",
            "Scaei6tdU6k_00-00-00_00-00-10": "Logically, a woman is speaking, but her words contrast with her delivery. Acoustically, her tone is highly elevated, laughing, and playful, indicating she is not being serious. Based on this, she was being playful and expressing surprise."
        }

        assistant_text = answer
        if "cot" in experiment_meta.experiment_name:
            reasoning = fake_cots.get(item_id, f"The correct choice is {answer}.")
            assistant_text = f"<analysis>\n{reasoning}\n</analysis>\n<answer>\n{answer}\n</answer>"

        is_few_shot_text_only = "few_shot_text_only" in experiment_meta.experiment_name
            
        turns.append(FewShotTurn(
            audio_path=None if is_few_shot_text_only else audio_path,
            audio_bytes=None,
            user_text=user_text,
            assistant_text=assistant_text
        ))
        
    logger.info(f"Successfully loaded {len(turns)} authentic MMAR few-shot examples.")
    return turns

def get_icbhi_few_shot_turns(config: DatasetConfig, experiment_meta, is_text_only: bool = False, num_shots_per_class: int = 1) -> list[FewShotTurn]:
    """Loads authentic audio few-shot examples from the ICBHI train split."""
    logger.info(f"Dynamically sampling {num_shots_per_class} ICBHI few-shot examples per class from the train split...")
    ds = load_dataset(config.dataset_id, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    df = ds.to_pandas()
    
    turns = []
    for label in config.target_labels:
        label_df = df[df["label"].str.contains(label, case=False, na=False)]
        if label_df.empty:
            continue
            
        sample_df = label_df.sample(n=min(num_shots_per_class, len(label_df)), random_state=42)
        
        for _, row in sample_df.iterrows():
            audio_bytes = row["audio"]["bytes"] if isinstance(row["audio"], dict) and "bytes" in row["audio"] else None
            
            user_text = row["instruction"]
            fake_reasoning = f"The audio presents acoustic signatures indicative of {row['label']}."
            assistant_text = f"<analysis>\n{fake_reasoning}\n</analysis>\n<answer>\n{row['label']}\n</answer>"
            
            turns.append(FewShotTurn(
                audio_path=row["file"],
                audio_bytes=audio_bytes,
                user_text=user_text,
                assistant_text=assistant_text
            ))
            
    logger.info(f"Successfully prepared {len(turns)} authentic ICBHI few-shot examples.")
    return turns
