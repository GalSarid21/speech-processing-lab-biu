import argparse
import os
import json
import torch
import librosa
import numpy as np
from pydantic import BaseModel, Field
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from tqdm import tqdm


class TranscriptModelConfig(BaseModel):
    """Configuration for the batched ASR pipeline."""
    model_id: str = Field(default="openai/whisper-large-v3", description="HuggingFace model ID")
    target_sample_rate: int = Field(default=16000, description="Sample rate required by the model")
    chunk_length_s: int = Field(default=30, description="Chunk length for batched inference")
    batch_size: int = Field(default=64, description="Batch size for GPU throughput")
    attention_impl: str = Field(default="sdpa", description="Attention backend (sdpa = Flash Attention 2)")
    language: str = Field(default="english", description="Target language")
    task: str = Field(default="transcribe", description="Task (transcribe or translate)")
    data_folder: str = Field(default="data", description="Base folder for artifacts")


def audio_generator(items: list[dict], target_sr: int):
    for item in items:
        try:
            audio, sr = librosa.load(item["audio_path"], sr=target_sr)
            yield {"raw": audio, "sampling_rate": sr}
        except Exception as e:
            print(f"Error loading {item['audio_path']}: {e}")
            yield {"raw": np.zeros(target_sr), "sampling_rate": target_sr}


def main():
    parser = argparse.ArgumentParser(description="Preprocess Transcripts")
    parser.add_argument("--dataset", type=str, required=True, choices=["icbhi", "mmar"], help="Which dataset to process")
    args = parser.parse_args()

    if args.dataset != "mmar":
        print(f"Dataset {args.dataset} does not contain speech data requiring Whisper transcription.")
        return

    config = TranscriptModelConfig()
    
    target_ids_file = f"{config.data_folder}/mmar_experiment_sample_ids.txt"
    few_shot_ids_file = f"{config.data_folder}/mmar_en_speech_few_shot_ids.txt"
    output_json = f"{config.data_folder}/mmar_transcripts.json"
    output_md = f"{config.data_folder}/mmar_transcripts_review.md"

    # Load target IDs
    target_ids = set()
    for f_path in [target_ids_file, few_shot_ids_file]:
        if os.path.exists(f_path):
            with open(f_path, "r") as f:
                target_ids.update(line.strip() for line in f if line.strip())

    if not target_ids:
        print(f"No target IDs found. Check if {target_ids_file} exists.")
        return

    print(f"Loaded {len(target_ids)} target IDs.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading {config.model_id} onto {device} in {torch_dtype}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(config.model_id)

    # Configured for maximum throughput
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs={"attn_implementation": config.attention_impl}, 
        chunk_length_s=config.chunk_length_s,
        batch_size=config.batch_size,
        generate_kwargs={"task": config.task, "language": config.language},
    )

    print("Loading BoJack/MMAR dataset...")
    ds = load_dataset('BoJack/MMAR', split='test', streaming=False)
    df = ds.to_pandas()
    sample_df = df[df['id'].isin(target_ids)]

    # Prepare data for batched generator
    dataset_items = []
    for _, row in sample_df.iterrows():
        item_id = row['id']
        audio_rel_path = row['audio_path']
        audio_path = os.path.join(config.data_folder, "MMAR", audio_rel_path.lstrip('./'))
        dataset_items.append({
            "id": item_id,
            "question": row['question'],
            "audio_path": audio_path
        })

    results = {}
    md_lines = [f"# MMAR Transcripts Review ({config.model_id})\n", "| ID | Question | Whisper Transcript |", "|---|---|---|"]

    print("Starting batched transcription...")
    
    # Execute batched inference pipeline
    generator = audio_generator(dataset_items, config.target_sample_rate)
    for item, out in tqdm(zip(dataset_items, pipe(generator)), total=len(dataset_items)):
        item_id = item["id"]
        text = out.get("text", "").strip()
        if not text:
            text = "[TRANSCRIPTION FAILED OR SILENT]"
            
        results[item_id] = text
        md_lines.append(f"| `{item_id}` | {item['question']} | {text} |")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
        
    with open(output_md, "w") as f:
        f.write("\n".join(md_lines) + "\n")
        
    print(f"Saved {len(results)} transcripts to {output_json} and {output_md}")

if __name__ == "__main__":
    main()
