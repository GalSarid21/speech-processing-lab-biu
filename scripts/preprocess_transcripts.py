import argparse
import os
import json
import torch
import librosa
import numpy as np
import yaml
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from tqdm import tqdm


def audio_generator(items: list[dict], target_sr: int, path_col: str):
    for item in items:
        try:
            audio, sr = librosa.load(item[path_col], sr=target_sr)
            yield {"raw": audio, "sampling_rate": sr}
        except Exception as e:
            print(f"Error loading {item[path_col]}: {e}")
            yield {"raw": np.zeros(target_sr), "sampling_rate": target_sr}


def main():
    parser = argparse.ArgumentParser(description="Preprocess Transcripts from YAML configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found.")
        return

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration sections
    ds_conf = config.get("dataset", {})
    paths_conf = config.get("paths", {})
    model_conf = config.get("model", {})

    target_ids_files = paths_conf.get("target_ids_files", [])
    output_json = paths_conf.get("output_json", "transcripts.json")
    output_md = paths_conf.get("output_md", "transcripts_review.md")
    audio_base_dir = paths_conf.get("audio_base_dir", "data")

    # Load target IDs
    target_ids = set()
    for f_path in target_ids_files:
        if os.path.exists(f_path):
            with open(f_path, "r") as f:
                target_ids.update(line.strip() for line in f if line.strip())

    if not target_ids:
        print(f"No target IDs found. Check if your target_ids_files exist.")
        return

    print(f"Loaded {len(target_ids)} target IDs.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = model_conf.get("id", "openai/whisper-large-v3")
    print(f"Loading {model_id} onto {device} in {torch_dtype}...")
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs={"attn_implementation": model_conf.get("attention_impl", "sdpa")}, 
        chunk_length_s=model_conf.get("chunk_length_s", 30),
        batch_size=model_conf.get("batch_size", 64),
        generate_kwargs={
            "task": model_conf.get("task", "transcribe"), 
            "language": model_conf.get("language", "english")
        },
    )

    dataset_id = ds_conf.get("id")
    split = ds_conf.get("split", "test")
    print(f"Loading dataset {dataset_id} [{split}]...")
    ds = load_dataset(dataset_id, split=split, streaming=False)
    df = ds.to_pandas()
    
    id_col = ds_conf.get("id_column", "id")
    audio_col = ds_conf.get("audio_path_column", "audio_path")
    ref_col = ds_conf.get("reference_text_column", "question")
    
    sample_df = df[df[id_col].isin(target_ids)]

    dataset_items = []
    for _, row in sample_df.iterrows():
        item_id = str(row[id_col])
        audio_rel_path = str(row[audio_col])
        audio_path = os.path.join(audio_base_dir, audio_rel_path.lstrip('./'))
        dataset_items.append({
            "id": item_id,
            "ref_text": str(row.get(ref_col, "")),
            "audio_path": audio_path
        })

    results = {}
    md_lines = [f"# Transcripts Review ({model_id})\n", "| ID | Reference Text | Whisper Transcript |", "|---|---|---|"]

    print("Starting batched transcription...")
    
    target_sr = model_conf.get("target_sample_rate", 16000)
    generator = audio_generator(dataset_items, target_sr, "audio_path")
    
    for item, out in tqdm(zip(dataset_items, pipe(generator)), total=len(dataset_items)):
        item_id = item["id"]
        text = out.get("text", "").strip()
        if not text:
            text = "[TRANSCRIPTION FAILED OR SILENT]"
            
        results[item_id] = text
        md_lines.append(f"| `{item_id}` | {item['ref_text']} | {text} |")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
        
    with open(output_md, "w") as f:
        f.write("\n".join(md_lines) + "\n")
        
    print(f"Saved {len(results)} transcripts to {output_json} and {output_md}")

if __name__ == "__main__":
    main()
