import os
import json
import yaml
import argparse
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm
from datasets import load_dataset

from speech_processing.models.audio_embeddings import LCOAudioEmbedder
from datasets import load_dataset
from loguru import logger
from speech_processing.models.audio_embeddings import LCOAudioEmbedder
from tqdm import tqdm
import argparse
import json
import os
import torch
import yaml

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)["rag_indexer"]

    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Initializing Audio Embedder ({config['model_id']})...")
    
    embedder = LCOAudioEmbedder(model_id=config['model_id'])

    few_shot_ids_file = config["io"]["candidate_ids_file"]
    test_ids_file = config["io"]["query_ids_file"]
    output_mapping_file = config["io"]["output_mapping_file"]
    top_k = config["parameters"].get("top_k", 3)
    batch_size = config["parameters"].get("batch_size", 16)
    
    with open(few_shot_ids_file, "r") as f:
        candidate_ids = [line.strip() for line in f if line.strip()]
    with open(test_ids_file, "r") as f:
        query_ids = [line.strip() for line in f if line.strip()]
        
    logger.info(f"Loaded {len(candidate_ids)} candidate IDs and {len(query_ids)} query IDs.")
    
    dataset_id = config["dataset"]["id"]
    dataset_split = config["dataset"]["split"]
    logger.info(f"Loading {dataset_id} [{dataset_split}] dataset...")
    ds = load_dataset(dataset_id, split=dataset_split, streaming=False)
    df = ds.to_pandas()
    
    id_col = config["dataset"]["id_column"]
    path_col = config["dataset"]["audio_path_column"]
    audio_base_dir = config["dataset"].get("audio_base_dir", "")

    def get_audio_path(item_id):
        row = df[df[id_col] == item_id]
        if row.empty:
            return None
        if 'file' in row.columns and isinstance(row.iloc[0]['file'], str):
            return os.path.join(audio_base_dir, str(row.iloc[0]['file']).lstrip('./'))
        if path_col in row.columns:
            return os.path.join(audio_base_dir, str(row.iloc[0][path_col]).lstrip('./'))
        return None

    # PHASE 1: Embed Candidates (Batched)
    logger.info("--- Embedding Candidates (Batched) ---")
    valid_cand_ids = []
    valid_cand_paths = []
    for cid in candidate_ids:
        path = get_audio_path(cid)
        if path and os.path.exists(path):
            valid_cand_ids.append(cid)
            valid_cand_paths.append(path)
        else:
            logger.warning(f"Audio path not found for candidate {cid}")

    if not valid_cand_paths:
        logger.error("No valid candidate paths found! Cannot proceed.")
        return

    # A single batched call across the GPU
    cand_tensor = embedder.embed_audio(valid_cand_paths, batch_size=batch_size)
    logger.info(f"Generated candidate tensor of shape {cand_tensor.shape}")

    # PHASE 2: Embed Queries (Batched)
    logger.info("--- Embedding Queries & Computing Similarity Matrix ---")
    valid_query_ids = []
    valid_query_paths = []
    for qid in query_ids:
        path = get_audio_path(qid)
        if path and os.path.exists(path):
            valid_query_ids.append(qid)
            valid_query_paths.append(path)
        else:
            logger.warning(f"Audio path not found for query {qid}")
            
    if not valid_query_paths:
        logger.error("No valid query paths found!")
        return

    # A single batched call across the GPU
    query_tensor = embedder.embed_audio(valid_query_paths, batch_size=batch_size)
    logger.info(f"Generated query tensor of shape {query_tensor.shape}")

    # PHASE 3: Compute Cross-Similarity and Top-K
    rag_mapping = {}
    
    # query_tensor is [Q, D], cand_tensor is [C, D]
    # SentenceTransformer similarity natively handles (Q, D) vs (C, D) -> (Q, C)
    try:
        sims_matrix = embedder.compute_similarity(query_tensor, cand_tensor)
        
        k = min(top_k, len(valid_cand_ids))
        top_scores, top_indices = torch.topk(sims_matrix, k, dim=-1)
        
        for idx, qid in enumerate(valid_query_ids):
            indices_for_query = top_indices[idx]
            rag_mapping[qid] = [valid_cand_ids[i.item()] for i in indices_for_query]
            
    except Exception as e:
        logger.error(f"Failed to compute similarity matrix: {e}")
        return

    # Save Mapping
    os.makedirs(os.path.dirname(output_mapping_file), exist_ok=True)
    with open(output_mapping_file, "w") as f:
        json.dump(rag_mapping, f, indent=4)
        
    logger.info(f"Successfully computed RAG mappings for {len(rag_mapping)} queries and saved to {output_mapping_file}")

if __name__ == "__main__":
    main()
