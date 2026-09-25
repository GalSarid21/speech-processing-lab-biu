import argparse
import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import f1_score
from loguru import logger

def parse_metrics(jsonl_path):
    total = 0
    acoustic = 0
    diagnostic = 0
    hallucination = 0
    y_true = []
    y_pred = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            eval_res = data.get("evaluation", {})
            acoustic += eval_res.get("acoustic_accuracy", 0)
            diagnostic += eval_res.get("diagnostic_accuracy", 0)
            hallucination += eval_res.get("hallucination_penalty", 0)
            total += 1
            
            y_true.append(data.get("ground_truth", ""))
            y_pred.append(eval_res.get("extracted_class", ""))
            
    if total == 0:
        return None
        
    def normalize_label(label):
        if label == "No potential disease detected":
            return "Healthy"
        if label == "None" or not label:
            return "Unknown"
        return label
        
    y_true_norm = [normalize_label(l) for l in y_true]
    y_pred_norm = [normalize_label(l) for l in y_pred]
        
    macro_f1 = f1_score(y_true_norm, y_pred_norm, average='macro', zero_division=0) * 100
    weighted_f1 = f1_score(y_true_norm, y_pred_norm, average='weighted', zero_division=0) * 100
    
    return {
        "Acoustic Accuracy (%)": (acoustic / (total * 10)) * 100,
        "Diagnostic Accuracy (%)": (diagnostic / (total * 10)) * 100,
        "Hallucination Rate (%)": (hallucination / total) * 100,
        "Macro F1 (%)": macro_f1,
        "Weighted F1 (%)": weighted_f1
    }

def main():
    parser = argparse.ArgumentParser(description="Compare experiment results.")
    parser.add_argument("--dataset", type=str, required=True, choices=["icbhi", "mmar"], help="Which dataset to analyze")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing the results")
    args = parser.parse_args()

    experiments = {}
    
    for run_dir in glob.glob(os.path.join(args.results_dir, f"{args.dataset}_*/")):
        jsonl_path = os.path.join(run_dir, "raw_output.jsonl")
        if not os.path.exists(jsonl_path):
            jsonl_path = os.path.join(run_dir, "raw_output_run1.jsonl")
        if not os.path.exists(jsonl_path):
            continue
            
        dirname = os.path.basename(os.path.normpath(run_dir))
        
        # New format: {dataset}_{experiment_name}_{timestamp}
        # e.g., icbhi_baseline_20260921_123456
        parts = dirname.split('_')
        # Filter out the dataset prefix and the timestamp (last 2 parts usually, date and time)
        # However, timestamp could be anything. Let's look for a timestamp pattern or just strip the known ones.
        if len(parts) >= 4 and parts[-1].isdigit() and parts[-2].isdigit():
            exp_name = "_".join(parts[1:-2])
        else:
            exp_name = "_".join(parts[1:])
            
        metrics = parse_metrics(jsonl_path)
        if metrics:
            # If we run the same experiment multiple times, take the latest (they are sorted alphabetically by default which implies chronologically)
            experiments[exp_name] = metrics
            
    if not experiments:
        logger.error(f"No completed {args.dataset.upper()} experiments found in {args.results_dir}.")
        return

    df = pd.DataFrame.from_dict(experiments, orient='index')
    
    sns.set_theme(style="whitegrid")
    
    # 1. Bar Chart: Diagnostic Accuracy vs Acoustic Accuracy
    plt.figure(figsize=(14, 8))
    df_melted = df[['Diagnostic Accuracy (%)', 'Acoustic Accuracy (%)']].reset_index().melt(id_vars='index')
    sns.barplot(x='index', y='value', hue='variable', data=df_melted, palette=['#4c72b0', '#55a868'])
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{args.dataset.upper()} - Model Accuracy Comparison Across Strategies')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Experiment Strategy')
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, f"{args.dataset}_accuracy_comparison.png"))
    plt.close()

    # 2. Bar Chart: F1 Scores
    plt.figure(figsize=(14, 8))
    df_f1 = df[['Macro F1 (%)', 'Weighted F1 (%)']].reset_index().melt(id_vars='index')
    sns.barplot(x='index', y='value', hue='variable', data=df_f1, palette=['#c44e52', '#8172b3'])
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{args.dataset.upper()} - F1 Score Comparison Across Strategies')
    plt.ylabel('F1 Score (%)')
    plt.xlabel('Experiment Strategy')
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, f"{args.dataset}_f1_comparison.png"))
    plt.close()

    # 3. Save Markdown Table
    md_path = os.path.join(args.results_dir, f"{args.dataset}_comparison_report.md")
    with open(md_path, 'w') as f:
        f.write(f"# {args.dataset.upper()} Experiment Comparison Report\n\n")
        f.write(df.to_markdown())
        
    logger.info(f"Comparison complete! Generated artifacts in {args.results_dir}/")

if __name__ == "__main__":
    main()
