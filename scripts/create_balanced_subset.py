"""Builds the stratified 100-item ICBHI subset used by V1 and V3. SUPERSEDED by V4.

V4 evaluates a fixed set - every row except a small held-out demonstration split - identically for
every experiment, and `load_icbhi_requests` ignores `sample_ids` entirely. It does not subsample to
balance either: balanced accuracy already weights every class equally while keeping all the data. That is deliberate: V1 and V3 were run on different item sets (V1 was 60%
COPD, V3 35%), which made every cross-version comparison meaningless. A fixed evaluation set is
what makes the V4 numbers comparable to each other.

This script is kept because `scripts/generate_notebook.py` still reads its output for the V1
research notebook, and because the V1/V3 result files cannot be interpreted without knowing how
their subset was drawn. Do not use it to drive a V4 run.
"""

import argparse

import pandas as pd
from datasets import Audio, load_dataset
from loguru import logger


def create_icbhi_subset(n_total: int, output_path: str):
    logger.info("Fetching ICBHI dataset...")
    ds = load_dataset("DynamicSuperb/RespiratorySoundClassification_ICBHI2017", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    df = ds.to_pandas()

    logger.info(f"Original dataset length: {len(df)}")

    # Exclude the few-shot examples we use in V9-V12
    df = df[~df["file"].isin(["audio100.wav", "audio101.wav"])]

    # Exclude Asthma and LRTI as requested (extremely rare classes)
    df = df[~df["label"].isin(["Asthma", "LRTI"])]

    # Calculate proportional representation
    proportions = df["label"].value_counts(normalize=True)

    target_counts = (proportions * n_total).round().astype(int)

    # Ensure it exactly sums
    diff = n_total - target_counts.sum()
    if diff != 0:
        target_counts.iloc[0] += diff

    sampled_df = pd.DataFrame()
    for label, target_count in target_counts.items():
        class_subset = df[df["label"] == label]
        n_sampled = min(target_count, len(class_subset))
        sampled_df = pd.concat([sampled_df, class_subset.sample(n=n_sampled, random_state=42)])

    with open(output_path, "w") as f:
        for file_id in sampled_df["file"]:
            f.write(file_id + "\n")

    logger.info(f"\n--- NEW BALANCED {n_total}-SAMPLE SUBSET FOR ICBHI ---")
    logger.info(f"\n{sampled_df['label'].value_counts()}")


def main():
    parser = argparse.ArgumentParser(description="Create a balanced subset of sample IDs.")
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["icbhi"], help="Which dataset to create a subset for"
    )
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to extract")

    args = parser.parse_args()

    if args.dataset != "icbhi":
        logger.error(
            "Balancing logic only applies to ICBHI (which has disease classes). MMAR is simply evaluated on its predefined English subset."
        )
        return

    output_path = "data/icbhi_experiment_sample_ids.txt"
    create_icbhi_subset(args.num_samples, output_path)
    logger.info(f"Successfully saved sample IDs to {output_path}")


if __name__ == "__main__":
    main()
