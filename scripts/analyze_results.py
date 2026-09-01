import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from speech_processing.data.dtos import JudgeResponse


def generate_markdown_report(metrics_dict, y_true, y_pred, output_file):
    # Calculate global averages
    len(y_true)
    
    # Calculate classification report as dict
    clf_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    
    md = "# Speech Processing Pipeline Analysis\n\n"
    
    md += "## Aggregate Judge Scores\n"
    md += f"- **Acoustic Accuracy**: {metrics_dict['avg_acoustic']:.2f}/10\n"
    md += f"- **Diagnostic Accuracy**: {metrics_dict['avg_diagnostic']:.2f}/10\n"
    md += f"- **Hallucination Rate**: {metrics_dict['avg_hallucination']:.2f}%\n\n"
    
    md += "## Classification Report\n\n"
    
    # Create Markdown Table
    md += "| Class | Precision | Recall | F1-Score | Support |\n"
    md += "|-------|-----------|--------|----------|---------|\n"
    
    for key, value in clf_dict.items():
        if key in ["accuracy", "macro avg", "weighted avg"]:
            continue
        md += f"| {key} | {value['precision']:.2f} | {value['recall']:.2f} | {value['f1-score']:.2f} | {value['support']} |\n"
        
    md += "\n**Averages:**\n"
    md += f"- **Accuracy**: {clf_dict['accuracy']:.2f}\n"
    md += f"- **Macro Avg (F1)**: {clf_dict['macro avg']['f1-score']:.2f}\n"
    md += f"- **Weighted Avg (F1)**: {clf_dict['weighted avg']['f1-score']:.2f}\n\n"
    
    md += "## Generated Visualizations\n"
    md += "The following visualizations have been saved in this directory:\n"
    md += "- `confusion_matrix.png`: Confusion matrix of predicted vs ground truth labels\n"
    md += "- `acoustic_dist.png`: Distribution of Acoustic Accuracy scores\n"
    md += "- `diagnostic_dist.png`: Distribution of Diagnostic Accuracy scores\n"
    md += "- `hallucination_pie.png`: Proportion of hallucinated outputs\n"

    with open(output_file, "w") as f:
        f.write(md)

def analyze_run(run_dir: str):
    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"Error: Directory {run_dir} does not exist.")
        sys.exit(1)
        
    raw_output_path = run_path / "raw_output.jsonl"
    if not raw_output_path.exists():
        print(f"Error: {raw_output_path} not found.")
        sys.exit(1)
        
    analysis_dir = run_path / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    responses = []
    with open(raw_output_path, "r") as f:
        for line in f:
            responses.append(JudgeResponse.model_validate_json(line))
            
    if not responses:
        print("No responses found to analyze.")
        sys.exit(0)
        
    y_true = [resp.request.ground_truth for resp in responses]
    y_pred = [resp.evaluation.extracted_disease_class for resp in responses]
    acoustic_scores = [resp.evaluation.acoustic_accuracy for resp in responses]
    diagnostic_scores = [resp.evaluation.diagnostic_accuracy for resp in responses]
    hallucination_scores = [resp.evaluation.hallucination_penalty for resp in responses]
    
    # Generate Visualizations
    sns.set_theme(style="whitegrid")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Disease Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(analysis_dir / "confusion_matrix.png")
    plt.close()
    
    # 2. Acoustic Score Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(acoustic_scores, bins=10, kde=True, binrange=(0,10))
    plt.title('Acoustic Accuracy Score Distribution')
    plt.xlabel('Score (0-10)')
    plt.ylabel('Count')
    plt.savefig(analysis_dir / "acoustic_dist.png")
    plt.close()
    
    # 3. Diagnostic Score Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(diagnostic_scores, bins=10, kde=True, binrange=(0,10))
    plt.title('Diagnostic Accuracy Score Distribution')
    plt.xlabel('Score (0-10)')
    plt.ylabel('Count')
    plt.savefig(analysis_dir / "diagnostic_dist.png")
    plt.close()
    
    # 4. Hallucination Pie Chart
    plt.figure(figsize=(6, 6))
    hal_count = sum(hallucination_scores)
    no_hal_count = len(hallucination_scores) - hal_count
    plt.pie([hal_count, no_hal_count], labels=['Hallucinated', 'Clean'], autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
    plt.title('Hallucination Rate')
    plt.savefig(analysis_dir / "hallucination_pie.png")
    plt.close()
    
    # 5. Generate Markdown
    metrics_dict = {
        "avg_acoustic": sum(acoustic_scores) / len(acoustic_scores),
        "avg_diagnostic": sum(diagnostic_scores) / len(diagnostic_scores),
        "avg_hallucination": (hal_count / len(hallucination_scores)) * 100
    }
    
    generate_markdown_report(metrics_dict, y_true, y_pred, analysis_dir / "metrics.md")
    
    print(f"Analysis complete! Artifacts saved to: {analysis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze pipeline results")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the run directory")
    args = parser.parse_args()
    analyze_run(args.run_dir)
