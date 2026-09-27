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

def main():
    parser = argparse.ArgumentParser(description="Build Acoustic RAG Index dynamically via YAML config.")
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

    # IO Paths
    few_shot_ids_file = config["io"]["candidate_ids_file"]
    test_ids_file = config["io"]["query_ids_file"]
    output_mapping_file = config["io"]["output_mapping_file"]
    top_k = config["parameters"].get("top_k", 3)
    
    # Load IDs
    with open(few_shot_ids_file, "r") as f:
        candidate_ids = [line.strip() for line in f if line.strip()]
    with open(test_ids_file, "r") as f:
        query_ids = [line.strip() for line in f if line.strip()]
        
    logger.info(f"Loaded {len(candidate_ids)} candidate IDs and {len(query_ids)} query IDs.")
    
    # Load HF Dataset
    dataset_id = config["dataset"]["id"]
    dataset_split = config["dataset"]["split"]
    logger.info(f"Loading {dataset_id} [{dataset_split}] dataset...")
    ds = load_dataset(dataset_id, split=dataset_split, streaming=False)
    df = ds.to_pandas()
    
    id_col = config["dataset"]["id_column"]
    path_col = config["dataset"]["audio_path_column"]

    def get_audio_path(item_id):
        row = df[df[id_col] == item_id]
        if row.empty:
            return None
            
        # Standardize path extraction
        if 'file' in row.columns and isinstance(row.iloc[0]['file'], str):
            return row.iloc[0]['file']
        if path_col in row.columns:
            return str(row.iloc[0][path_col]).lstrip('./')
        return None

    # Phase 1: Embed Candidates
    logger.info("--- Embedding Candidates ---")
    candidate_embeddings = {}
    for cid in tqdm(candidate_ids, desc="Embedding Candidates"):
        path = get_audio_path(cid)
        if path and os.path.exists(path):
            try:
                candidate_embeddings[cid] = embedder.embed_audio(path)
            except Exception as e:
                logger.warning(f"Failed to embed candidate {cid}: {e}")
        else:
            logger.warning(f"Audio path not found for candidate {cid}: {path}")

    if not candidate_embeddings:
        logger.error("No candidate embeddings were generated! Cannot proceed.")
        return

    # Phase 2: Embed Queries and Compute RAG Matrix
    logger.info("--- Embedding Queries & Computing Similarity Matrix ---")
    rag_mapping = {}
    
    cand_ids_list = list(candidate_embeddings.keys())
    cand_tensor = torch.cat([candidate_embeddings[cid] for cid in cand_ids_list], dim=0) 
    
    for qid in tqdm(query_ids, desc="Embedding & Mapping Queries"):
        path = get_audio_path(qid)
        if not path or not os.path.exists(path):
            logger.warning(f"Audio path not found for query {qid}")
            continue
            
        try:
            query_emb = embedder.embed_audio(path)
            
            # Cosine similarity
            sims = embedder.compute_similarity(query_emb, cand_tensor).squeeze()
            
            k = min(top_k, len(cand_ids_list))
            top_scores, top_indices = torch.topk(sims, k)
            
            top_k_ids = [cand_ids_list[i.item()] for i in top_indices]
            rag_mapping[qid] = top_k_ids
            
        except Exception as e:
            logger.warning(f"Failed to process query {qid}: {e}")

    # Phase 3: Save Mapping
    os.makedirs(os.path.dirname(output_mapping_file), exist_ok=True)
    with open(output_mapping_file, "w") as f:
        json.dump(rag_mapping, f, indent=4)
        
    logger.info(f"RAG mapping successfully saved to {output_mapping_file}")
    logger.info(f"Mapped {len(rag_mapping)} queries to their top-{top_k} acoustic neighbors.")

if __name__ == "__main__":
    main()
