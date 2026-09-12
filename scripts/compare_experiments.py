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
    "baseline_honest",
    "format_strict",
    "symptoms_dict",
    "acoustic_dict",
    "cot",
    "few_shot",
    "cot_and_few_shot",
    "full_optimized",
    "authentic_few_shot",
    "authentic_few_shot_holistic",
    "authentic_few_shot_no_guardrails",
    "authentic_few_shot_holistic_honest",
    "authentic_few_shot_prior_aware",
    "proportional_few_shot_prior_aware",
    "multi_turn_decomposition"
]

from sklearn.metrics import f1_score

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
            
            y_true.append(data.get("request", {}).get("ground_truth", ""))
            y_pred.append(eval_res.get("extracted_disease_class", ""))
            
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

import sys

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    experiments = {}
    
    # Gather metrics for the latest run of each experiment
    for run_dir in glob.glob(os.path.join(results_dir, "*/")):
        jsonl_path = os.path.join(run_dir, "raw_output.jsonl")
        if not os.path.exists(jsonl_path):
            jsonl_path = os.path.join(run_dir, "raw_output_run1.jsonl")
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
            # Parse stability report
            n_runs = 1
            stability_val = "N/A"
            stability_path = os.path.join(run_dir, "stability_report.json")
            if os.path.exists(stability_path):
                try:
                    with open(stability_path, 'r') as sf:
                        stab_data = json.load(sf)
                        n_runs = stab_data.get("runs", 1)
                        cv = stab_data.get("diagnostic_accuracy", {}).get("coefficient_of_variation_pct", 0.0)
                        stability_val = f"{cv:.2f}%"
                except Exception:
                    pass
            
            # Calculate composite score (60% Diagnostic, 20% Acoustic, 20% Hallucination Inverse)
            comp_score = 0.2 * metrics["Acoustic Accuracy (%)"] + 0.6 * metrics["Diagnostic Accuracy (%)"] + 0.2 * (100 - metrics["Hallucination Rate (%)"])
            metrics["Overall Score"] = comp_score
            metrics["Runs"] = n_runs
            metrics["Diag CV"] = stability_val
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
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Experiment Comparisons (Scatter Plots)", fontsize=16)
    
    metrics_to_plot = [
        ("Acoustic Accuracy (%)", axes[0, 0]),
        ("Diagnostic Accuracy (%)", axes[0, 1]),
        ("Hallucination Rate (%)", axes[0, 2]),
        ("Macro F1 (%)", axes[1, 0]),
        ("Weighted F1 (%)", axes[1, 1]),
        ("Overall Score", axes[1, 2])
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
        if baseline_val == 0 and not is_hallucination:
            return f"{val:.1f}% (N/A)"
        
        if is_hallucination:
            if val == baseline_val:
                return f"{val:.1f}% (Unchanged)"
            elif val > baseline_val:
                mult = val / baseline_val if baseline_val > 0 else val
                return f"{val:.1f}% ({mult:.1f}x Worse 🔴)"
            else:
                mult = baseline_val / val if val > 0 else float('inf')
                if mult == float('inf'):
                    return f"{val:.1f}% (100% Fixed 🟢)"
                return f"{val:.1f}% ({mult:.1f}x Better 🟢)"
        else:
            mult = val / baseline_val
            return f"{val:.1f}% ({mult:.2f}x)"

    md_lines = []
    md_lines.append("## Experiment Comparisons\n")
    md_lines.append("| Experiment | Runs | Diag CV | Acoustic Acc | Diagnostic Acc | Hallucination Rate | Macro F1 | Weighted F1 | Overall Score |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    # Find bests for bolding
    best_ac = df["Acoustic Accuracy (%)"].max()
    best_diag = df["Diagnostic Accuracy (%)"].max()
    best_halluc = df["Hallucination Rate (%)"].min()
    best_macro = df["Macro F1 (%)"].max()
    best_weighted = df["Weighted F1 (%)"].max()
    best_score = df["Overall Score"].max()
    
    for _, row in df.iterrows():
        ac_str = format_multiplier(row["Acoustic Accuracy (%)"], baseline_row["Acoustic Accuracy (%)"])
        diag_str = format_multiplier(row["Diagnostic Accuracy (%)"], baseline_row["Diagnostic Accuracy (%)"])
        hal_str = format_multiplier(row["Hallucination Rate (%)"], baseline_row["Hallucination Rate (%)"], is_hallucination=True)
        macro_str = format_multiplier(row["Macro F1 (%)"], baseline_row["Macro F1 (%)"])
        weighted_str = format_multiplier(row["Weighted F1 (%)"], baseline_row["Weighted F1 (%)"])
        score_str = format_multiplier(row["Overall Score"], baseline_row["Overall Score"])
        
        if row["Acoustic Accuracy (%)"] == best_ac: ac_str = f"**{ac_str}**"
        if row["Diagnostic Accuracy (%)"] == best_diag: diag_str = f"**{diag_str}**"
        if row["Hallucination Rate (%)"] == best_halluc: hal_str = f"**{hal_str}**"
        if row["Macro F1 (%)"] == best_macro: macro_str = f"**{macro_str}**"
        if row["Weighted F1 (%)"] == best_weighted: weighted_str = f"**{weighted_str}**"
        if row["Overall Score"] == best_score: score_str = f"**{score_str}**"
        
        runs_str = str(row["Runs"])
        cv_str = str(row["Diag CV"])
        
        md_lines.append(f"| `{row['Experiment']}` | {runs_str} | {cv_str} | {ac_str} | {diag_str} | {hal_str} | {macro_str} | {weighted_str} | {score_str} |")
        
    md_path = os.path.join(results_dir, "comparison_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved comparison table to {md_path}")

if __name__ == "__main__":
    main()
