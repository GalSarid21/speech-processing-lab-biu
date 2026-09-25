import argparse
import pandas as pd
from datasets import load_dataset, Audio
from loguru import logger

def create_icbhi_subset(n_total: int, output_path: str):
    logger.info("Fetching ICBHI dataset...")
    ds = load_dataset("DynamicSuperb/RespiratorySoundClassification_ICBHI2017", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    df = ds.to_pandas()

    logger.info(f"Original dataset length: {len(df)}")

    # Exclude the few-shot examples we use in V9-V12
    df = df[~df['file'].isin(['audio100.wav', 'audio101.wav'])]
    
    # Exclude Asthma and LRTI as requested (extremely rare classes)
    df = df[~df['label'].isin(['Asthma', 'LRTI'])]

    # Calculate proportional representation
    proportions = df['label'].value_counts(normalize=True)
    
    target_counts = (proportions * n_total).round().astype(int)
    
    # Ensure it exactly sums
    diff = n_total - target_counts.sum()
    if diff != 0:
        target_counts.iloc[0] += diff

    sampled_df = pd.DataFrame()
    for label, count in target_counts.items():
        class_subset = df[df['label'] == label]
        if count > len(class_subset):
            count = len(class_subset)
        sampled_df = pd.concat([sampled_df, class_subset.sample(n=count, random_state=42)])

    with open(output_path, 'w') as f:
        for file_id in sampled_df['file']:
            f.write(file_id + '\n')
            
    logger.info(f"\n--- NEW BALANCED {n_total}-SAMPLE SUBSET FOR ICBHI ---")
    logger.info(f"\n{sampled_df['label'].value_counts()}")


def create_mmar_subset(n_total: int, output_path: str):
    logger.info("Fetching MMAR dataset...")
    ds = load_dataset("BoJack/MMAR", split="test", streaming=False)
    df = ds.to_pandas()
    
    logger.info(f"Original dataset length: {len(df)}")
    
    # Stratify by answer distribution (A, B, C, D)
    proportions = df['answer'].value_counts(normalize=True)
    target_counts = (proportions * n_total).round().astype(int)
    
    diff = n_total - target_counts.sum()
    if diff != 0:
        target_counts.iloc[0] += diff

    sampled_df = pd.DataFrame()
    for label, count in target_counts.items():
        class_subset = df[df['answer'] == label]
        if count > len(class_subset):
            count = len(class_subset)
        sampled_df = pd.concat([sampled_df, class_subset.sample(n=count, random_state=42)])

    with open(output_path, 'w') as f:
        for file_id in sampled_df['id']:
            f.write(file_id + '\n')
            
    logger.info(f"\n--- NEW BALANCED {n_total}-SAMPLE SUBSET FOR MMAR ---")
    logger.info(f"\n{sampled_df['answer'].value_counts()}")


def main():
    parser = argparse.ArgumentParser(description="Create a balanced subset of sample IDs.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=["icbhi", "mmar"], 
        help="Which dataset to create a subset for"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=100, 
        help="Number of samples to extract"
    )
    
    args = parser.parse_args()
    
    output_path = f"data/{args.dataset}_experiment_sample_ids.txt"
    
    if args.dataset == "icbhi":
        create_icbhi_subset(args.num_samples, output_path)
    elif args.dataset == "mmar":
        create_mmar_subset(args.num_samples, output_path)
        
    logger.info(f"Successfully saved sample IDs to {output_path}")

if __name__ == "__main__":
    main()
