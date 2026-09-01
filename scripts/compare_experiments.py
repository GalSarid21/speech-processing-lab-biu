import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define the expected order of experiments
EXPERIMENT_ORDER = [
    "baseline",
    "format_strict",
    "symptoms_dict",
    "acoustic_dict",
    "cot",
    "few_shot",
    "cot_and_few_shot",
    "full_optimized", # if it exists
    "authentic_few_shot",
    "authentic_few_shot_holistic",
    "authentic_few_shot_no_guardrails"
]

def parse_metrics(jsonl_path):
    total = 0
    acoustic = 0
    diagnostic = 0
    hallucination = 0
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            eval_res = data.get("evaluation", {})
            acoustic += eval_res.get("acoustic_accuracy", 0)
            diagnostic += eval_res.get("diagnostic_accuracy", 0)
            hallucination += eval_res.get("hallucination_penalty", 0)
            total += 1
    if total == 0:
        return None
    return {
        "Acoustic Accuracy (%)": (acoustic / (total * 10)) * 100,
        "Diagnostic Accuracy (%)": (diagnostic / (total * 10)) * 100,
        "Hallucination Rate (%)": (hallucination / total) * 100
    }

def main():
    results_dir = "results"
    experiments = {}
    
    # Gather metrics for the latest run of each experiment
    for run_dir in glob.glob(os.path.join(results_dir, "*/")):
        jsonl_path = os.path.join(run_dir, "raw_output.jsonl")
        if not os.path.exists(jsonl_path):
            continue
            
        dirname = os.path.basename(os.path.normpath(run_dir))
        
        # Extract experiment name by stripping timestamp if present
        exp_name = dirname
        for order_name in EXPERIMENT_ORDER:
            if dirname.startswith(order_name):
                # Ensure it's not a substring match of a longer name (e.g. authentic_few_shot vs authentic_few_shot_holistic)
                if dirname == order_name or dirname.startswith(order_name + "_2"):
                    exp_name = order_name
                    break
        
        metrics = parse_metrics(jsonl_path)
        if metrics is not None:
            # Calculate composite score (invert hallucination so higher is better)
            comp_score = (metrics["Acoustic Accuracy (%)"] + metrics["Diagnostic Accuracy (%)"] + (100 - metrics["Hallucination Rate (%)"])) / 3
            metrics["Overall Score"] = comp_score
            metrics["Dir"] = run_dir
            
            # Keep latest if multiple
            if exp_name not in experiments or run_dir > experiments[exp_name]["Dir"]:
                experiments[exp_name] = metrics
                
    # Sort experiments by predefined order
    sorted_exps = []
    for exp in EXPERIMENT_ORDER:
        if exp in experiments:
            sorted_exps.append({"Experiment": exp, **experiments[exp]})
            
    df = pd.DataFrame(sorted_exps)
    
    if df.empty:
        print("No data found.")
        return
        
    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Experiment Comparisons (Scatter Plots)", fontsize=16)
    
    metrics_to_plot = [
        ("Acoustic Accuracy (%)", axes[0, 0]),
        ("Diagnostic Accuracy (%)", axes[0, 1]),
        ("Hallucination Rate (%)", axes[1, 0]),
        ("Overall Score", axes[1, 1])
    ]
    
    x = range(len(df))
    for metric, ax in metrics_to_plot:
        ax.scatter(x, df[metric], s=100, color='blue')
        ax.plot(x, df[metric], linestyle='--', alpha=0.5, color='gray') # Connect dots to show progression
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Experiment"], rotation=45, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Percentage")
        
        # Highlight best point
        if metric == "Hallucination Rate (%)":
            best_idx = df[metric].idxmin()
            best_val = df[metric].min()
        else:
            best_idx = df[metric].idxmax()
            best_val = df[metric].max()
            
        ax.scatter([best_idx], [best_val], s=200, color='red', zorder=5, label="Best")
        ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "comparison_scatter_plots.png")
    plt.savefig(plot_path)
    print(f"Saved scatter plots to {plot_path}")
    
    # --- TABLE GENERATION ---
    baseline_row = df[df["Experiment"] == "baseline"].iloc[0]
    
    def format_multiplier(val, baseline_val, is_hallucination=False):
        if baseline_val == 0:
            return f"{val:.1f}% (N/A)"
        
        if is_hallucination:
            # For hallucination, we want drop. 24 -> 12 is 0.5x (or 2x reduction)
            # Let's show it as a multiplier relative to baseline
            mult = val / baseline_val
            return f"{val:.1f}% ({mult:.2f}x)"
        else:
            mult = val / baseline_val
            return f"{val:.1f}% ({mult:.2f}x)"

    md_lines = []
    md_lines.append("## Experiment Comparisons\n")
    md_lines.append("| Experiment | Acoustic Acc | Diagnostic Acc | Hallucination Rate | Overall Score |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    # Find bests for bolding
    best_ac = df["Acoustic Accuracy (%)"].max()
    best_diag = df["Diagnostic Accuracy (%)"].max()
    best_halluc = df["Hallucination Rate (%)"].min()
    best_score = df["Overall Score"].max()
    
    for _, row in df.iterrows():
        ac_str = format_multiplier(row["Acoustic Accuracy (%)"], baseline_row["Acoustic Accuracy (%)"])
        diag_str = format_multiplier(row["Diagnostic Accuracy (%)"], baseline_row["Diagnostic Accuracy (%)"])
        hal_str = format_multiplier(row["Hallucination Rate (%)"], baseline_row["Hallucination Rate (%)"], is_hallucination=True)
        score_str = format_multiplier(row["Overall Score"], baseline_row["Overall Score"])
        
        if row["Acoustic Accuracy (%)"] == best_ac: ac_str = f"**{ac_str}**"
        if row["Diagnostic Accuracy (%)"] == best_diag: diag_str = f"**{diag_str}**"
        if row["Hallucination Rate (%)"] == best_halluc: hal_str = f"**{hal_str}**"
        if row["Overall Score"] == best_score: score_str = f"**{score_str}**"
        
        md_lines.append(f"| `{row['Experiment']}` | {ac_str} | {diag_str} | {hal_str} | {score_str} |")
        
    md_path = os.path.join(results_dir, "comparison_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved comparison table to {md_path}")

if __name__ == "__main__":
    main()
