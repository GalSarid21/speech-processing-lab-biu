import argparse
import json
import os
import pandas as pd
import numpy as np
from glob import glob
from collections import defaultdict
from statsmodels.stats.contingency_tables import mcnemar

def compute_majority_vote(runs, item_ids):
    # runs: list of dicts mapping item_id -> predicted_choice
    # returns dict item_id -> majority choice
    majority = {}
    for item_id in item_ids:
        votes = [run.get(item_id) for run in runs if item_id in run]
        if not votes:
            continue
        # get most common
        vote_counts = {}
        for v in votes:
            vote_counts[v] = vote_counts.get(v, 0) + 1
        majority[item_id] = max(vote_counts.items(), key=lambda x: x[1])[0]
    return majority

def bootstrap_ci(y_true, y_pred, n_resamples=10000, alpha=0.05):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accs = []
    n = len(y_true)
    for _ in range(n_resamples):
        indices = np.random.randint(0, n, n)
        accs.append(np.mean(y_true[indices] == y_pred[indices]))
    accs = np.sort(accs)
    lower = accs[int((alpha/2) * n_resamples)]
    upper = accs[int((1 - alpha/2) * n_resamples)]
    return lower * 100, upper * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--baseline_dir", type=str)
    parser.add_argument("--split_file", type=str, default="data/mmar_subset_split.json")
    args = parser.parse_args()

    # If recomputing split
    # Wait, the split is based on text_only_llm
    text_only_run = next((d for d in args.run_dirs if "text_only_llm" in d), None)
    subset_split = {}
    if text_only_run:
        # Load text only run
        runs = []
        for run_file in glob(os.path.join(text_only_run, "raw_output*.jsonl")):
            run_data = {}
            with open(run_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    # Get ground truth and prediction
                    gt = data.get("ground_truth")
                    if 'metadata' in data:
                        item_id = data['metadata'].get("item_id")
                    elif 'evaluation' in data and 'sample_id' in data:
                        item_id = data['sample_id']
                    else:
                        continue
                    
                    if 'evaluation' in data:
                        pred = data['evaluation'].get('extracted_choice', '')
                        is_correct = data['evaluation'].get('is_correct', False)
                    else:
                        continue
                    run_data[item_id] = (pred, gt, is_correct)
            runs.append(run_data)
        
        # compute majority correctness
        item_ids = runs[0].keys() if runs else []
        for item_id in item_ids:
            votes = [run[item_id][2] for run in runs if item_id in run]
            is_correct_majority = sum(votes) > len(votes) / 2
            subset_split[item_id] = "transcript-solvable" if is_correct_majority else "audio-dependent"
        
        with open(args.split_file, "w") as f:
            json.dump(subset_split, f, indent=2)
        print(f"Saved subset split to {args.split_file}")
    else:
        if os.path.exists(args.split_file):
            with open(args.split_file, "r") as f:
                subset_split = json.load(f)
        else:
            print(f"Warning: {args.split_file} not found and text_only_llm run not provided.")

    baseline_majority = None
    baseline_true = None
    
    if args.baseline_dir:
        baseline_runs = []
        for run_file in glob(os.path.join(args.baseline_dir, "raw_output*.jsonl")):
            run_data = {}
            baseline_true = {}
            with open(run_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    item_id = data.get('sample_id')
                    pred = data['evaluation'].get('extracted_choice', '')
                    gt = data.get("ground_truth")
                    run_data[item_id] = pred
                    baseline_true[item_id] = gt
            baseline_runs.append(run_data)
        baseline_majority = compute_majority_vote(baseline_runs, baseline_true.keys())

    results = []

    for run_dir in args.run_dirs:
        exp_name = os.path.basename(run_dir).replace("mmar_", "")
        
        runs = []
        true_labels = {}
        subsets = defaultdict(list)
        categories = defaultdict(list)
        
        for run_file in glob(os.path.join(run_dir, "raw_output*.jsonl")):
            run_data = {}
            with open(run_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    item_id = data.get('sample_id')
                    pred = data['evaluation'].get('extracted_choice', '')
                    gt = data.get("ground_truth")
                    
                    run_data[item_id] = pred
                    true_labels[item_id] = gt
                    
                    meta = data.get("metadata", {})
                    cat = meta.get("category", "Unknown")
                    categories[cat].append((pred, gt))
            runs.append(run_data)
            
        if not runs:
            continue
            
        majority = compute_majority_vote(runs, true_labels.keys())
        y_true_list = []
        y_pred_list = []
        subset_results = {"transcript-solvable": {"true": [], "pred": []}, "audio-dependent": {"true": [], "pred": []}}
        
        gained = 0
        lost = 0
        contingency = [[0, 0], [0, 0]]
        
        for item_id, pred in majority.items():
            gt = true_labels[item_id]
            y_true_list.append(gt)
            y_pred_list.append(pred)
            
            sub = subset_split.get(item_id)
            if sub in subset_results:
                subset_results[sub]["true"].append(gt)
                subset_results[sub]["pred"].append(pred)
                
            if baseline_majority:
                base_pred = baseline_majority.get(item_id)
                base_correct = base_pred == gt
                our_correct = pred == gt
                
                if our_correct and not base_correct:
                    gained += 1
                elif not our_correct and base_correct:
                    lost += 1
                    
                if our_correct and base_correct:
                    contingency[0][0] += 1
                elif our_correct and not base_correct:
                    contingency[0][1] += 1
                elif not our_correct and base_correct:
                    contingency[1][0] += 1
                else:
                    contingency[1][1] += 1
                    
        acc = np.mean(np.array(y_true_list) == np.array(y_pred_list)) * 100
        lower, upper = bootstrap_ci(y_true_list, y_pred_list)
        
        ts_acc = np.mean(np.array(subset_results["transcript-solvable"]["true"]) == np.array(subset_results["transcript-solvable"]["pred"])) * 100 if subset_results["transcript-solvable"]["true"] else 0
        ad_acc = np.mean(np.array(subset_results["audio-dependent"]["true"]) == np.array(subset_results["audio-dependent"]["pred"])) * 100 if subset_results["audio-dependent"]["true"] else 0
        
        p_val = "-"
        if baseline_majority:
            res = mcnemar(contingency, exact=True)
            p_val = f"{res.pvalue:.3f}"
            
        results.append({
            "Experiment": exp_name,
            "Acc": f"{acc:.1f}",
            "CI": f"[{lower:.1f}, {upper:.1f}]",
            "Gained/Lost": f"{gained}/{lost}" if baseline_majority else "-",
            "p": p_val,
            "Transcript-solvable Acc": f"{ts_acc:.1f}",
            "Audio-dependent Acc": f"{ad_acc:.1f}"
        })
        
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    
    os.makedirs("results/MMAR/V2", exist_ok=True)
    with open("results/MMAR/V2/comparison.md", "w") as f:
        f.write(df.to_markdown(index=False) + "\n")
    print("Saved comparison table to results/MMAR/V2/comparison.md")

if __name__ == "__main__":
    main()
