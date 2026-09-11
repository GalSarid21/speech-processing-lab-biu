import gc
import json
import os
import time
from datetime import UTC, datetime

import torch
from loguru import logger

from speech_processing.config.core import AppConfig
from speech_processing.data.dataset import load_icbhi_requests
from speech_processing.data.dtos import JudgeRequest
from speech_processing.models.audio_model import QwenAudioEngine
from speech_processing.models.judge import QwenJudge
from speech_processing.prompts.templates.judge.qwen_judge import (
    QwenICBHI2017JudgeTemplate,
)


def run_audio_phase(config: AppConfig, temp_file_paths: list[str], prompt_version: str = "original"):
    logger.info(f"--- [PHASE 1] AUDIO INFERENCE ({len(temp_file_paths)} RUNS) ---")
    audio_engine = QwenAudioEngine(config.audio_model)

    # Load samples from the real dataset based on config
    dataset_items = load_icbhi_requests(config.dataset)
    
    # Optional Prompt Override
    if prompt_version != "original":
        from speech_processing.prompts.templates.audio.experiments import ExperimentVersion
        try:
            experiment_meta = ExperimentVersion.get_version(prompt_version).value
        except ValueError as e:
            logger.error(str(e))
            raise e
            
        custom_instruction = experiment_meta.prompt
        logger.info(f"Overriding dataset instructions with prompt version: {prompt_version} ({experiment_meta.experiment_name})")
        
        # Override the instruction in the DTO
        
        # Load authentic few-shot turns if running an authentic experiment
        few_shot_turns = []
        if "authentic_few_shot" in experiment_meta.experiment_name or "proportional_few_shot" in experiment_meta.experiment_name:
            from speech_processing.data.dataset import get_authentic_few_shot_turns
            few_shot_turns = get_authentic_few_shot_turns(
                config.dataset, 
                custom_instruction,
                experiment_name=experiment_meta.experiment_name
            )

        for req, _ in dataset_items:
            req.instruction = custom_instruction
            req.few_shot_turns = few_shot_turns

    requests = [req for req, _ in dataset_items]
    ground_truths = [gt for _, gt in dataset_items]

    for run_idx, temp_file_path in enumerate(temp_file_paths):
        logger.info(f"Audio Inference: Run {run_idx + 1}/{len(temp_file_paths)}")
        # Evaluate using the maximal batch size internally configured in the engine
        responses = audio_engine.batch_infer(requests)

        with open(temp_file_path, "w") as f:
            f.writelines(
                json.dumps(
                    {
                        "sample_id": req.audio_path,
                        "instruction": req.instruction,
                        "generated_text": resp.generated_text,
                        "ground_truth": gt,
                    }
                )
                + "\n"
                for req, resp, gt in zip(requests, responses, ground_truths)
            )

        logger.info(
            f"Run {run_idx + 1} audio inference complete. Wrote {len(responses)} results to {temp_file_path}."
        )
    # audio_engine goes out of scope here, making it eligible for GC.


def run_judge_phase(config: AppConfig, temp_file_paths: list[str], output_file_paths: list[str], metrics_file_paths: list[str]):
    logger.info(f"--- [PHASE 2] JUDGE EVALUATION ({len(temp_file_paths)} RUNS) ---")

    # Inject template
    template = QwenICBHI2017JudgeTemplate()
    judge_engine = QwenJudge(config.judge, template=template)

    from speech_processing.evaluation.metrics import calculate_metrics
    all_metrics = []

    for run_idx, (temp_file, output_file, metrics_file) in enumerate(zip(temp_file_paths, output_file_paths, metrics_file_paths)):
        logger.info(f"Judge Evaluation: Run {run_idx + 1}/{len(temp_file_paths)}")
        judge_requests = []
        if os.path.exists(temp_file):
            with open(temp_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    judge_requests.append(
                        JudgeRequest(
                            sample_id=data["sample_id"],
                            instruction=data["instruction"],
                            generated_text=data["generated_text"],
                            ground_truth=data["ground_truth"],
                        )
                    )

        if not judge_requests:
            logger.warning(f"No requests found in {temp_file}. Skipping run.")
            all_metrics.append(None)
            continue

        evaluations = judge_engine.batch_evaluate(judge_requests)

        with open(output_file, "w") as f:
            f.writelines(eval_resp.model_dump_json() + "\n" for eval_resp in evaluations)

        metrics = calculate_metrics(evaluations)
        all_metrics.append(metrics)

        print(f"\n==============================================")
        print(f"FINAL EVALUATION REPORT (Run {run_idx + 1})")
        print(f"==============================================")

        for i, eval_item in enumerate(evaluations):
            print(f"\n--- Sample {eval_item.request.sample_id} ---")
            print(f"Instruction: {eval_item.request.instruction}")
            print(f"True Label:  {eval_item.request.ground_truth}")
            print(f"Qwen Output: {eval_item.request.generated_text}")
            print(f"Predicted:   {eval_item.evaluation.extracted_disease_class}")
            print(f"Acoustic:    {eval_item.evaluation.acoustic_accuracy}/10")
            print(f"Diagnostic:  {eval_item.evaluation.diagnostic_accuracy}/10")
            print(f"Hallucinated:{'Yes' if eval_item.evaluation.hallucination_penalty == 1 else 'No'}")
            print(f"Reasoning:   {eval_item.evaluation.reasoning}")
            print("-" * 30)

        if metrics:
            report = (
                f"--- AGGREGATE METRICS (Run {run_idx + 1}) ---\n"
                f"Avg Acoustic Accuracy:   {metrics.avg_acoustic_pct:.2f}%\n"
                f"Avg Diagnostic Accuracy: {metrics.avg_diagnostic_pct:.2f}%\n"
                f"Hallucination Rate:      {metrics.hallucination_rate_pct:.2f}%\n\n"
                f"Classification Report:\n"
                f"{metrics.classification_report}\n"
            )
            print(f"\n{report}")
            
            with open(metrics_file, "w") as f:
                f.write(report)
            logger.info(f"Aggregate metrics for Run {run_idx + 1} saved to {metrics_file}")
        else:
            print("\nAggregate Dataset Accuracy: 0/10 (0%)")

    return all_metrics


def release_vram():
    """Explicitly garbage collect and empty the CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    logger.info("Explicit VRAM clearance complete.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the Speech Processing Pipeline")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save artifacts")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of dataset samples to evaluate")
    parser.add_argument("--experiment", type=str, default="v1", help="Which experiment version to run (e.g., v1, v2, ..., v8)")
    parser.add_argument("--sample-ids-file", type=str, default=None, help="Path to a previous raw_output.jsonl file to enforce exact same samples")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to run the experiment for stability analysis")
    args = parser.parse_args()

    # Pass the CLI arguments to AppConfig
    config = AppConfig(output_dir=args.output_dir)
    config.dataset.num_samples = args.num_samples

    if args.sample_ids_file and os.path.exists(args.sample_ids_file):
        logger.info(f"Loading reference sample IDs from {args.sample_ids_file}")
        ids = []
        with open(args.sample_ids_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if args.sample_ids_file.endswith(".jsonl"):
                    # We can parse the JudgeResponse JSON
                    data = json.loads(line)
                    ids.append(data["request"]["sample_id"])
                else:
                    # Treat it as a raw text file containing one ID per line
                    ids.append(line)
        config.dataset.sample_ids = ids

    # Fetch experiment metadata
    from speech_processing.prompts.templates.audio.experiments import ExperimentVersion
    experiment_meta = ExperimentVersion.get_version(args.experiment).value

    # Override dynamic variables based on experiment meta to prevent OOM / Sequence Length crashes
    config.audio_model.max_new_tokens = experiment_meta.max_new_tokens
    config.audio_model.max_num_seqs = experiment_meta.batch_size

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_meta.experiment_name}_{timestamp}"
    run_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    temp_files = []
    final_outputs = []
    metrics_outputs = []
    for i in range(args.runs):
        suffix = f"_run{i+1}" if args.runs > 1 else ""
        temp_files.append(os.path.join(run_dir, f"temp_audio_preds{suffix}.jsonl"))
        final_outputs.append(os.path.join(run_dir, f"raw_output{suffix}.jsonl"))
        metrics_outputs.append(os.path.join(run_dir, f"metrics{suffix}.txt"))

    # Phase 1: Audio Model
    run_audio_phase(config, temp_files, prompt_version=args.experiment)

    # Free memory explicitly instead of relying on process termination
    release_vram()
    
    logger.info("GPU Memory freed. Waiting 60 seconds before loading Judge model...")
    time.sleep(60)

    # Phase 2: Judge Model
    all_metrics = run_judge_phase(config, temp_files, final_outputs, metrics_outputs)
    
    # Final cleanup (optional but good practice)
    release_vram()

    if args.runs > 1:
        from speech_processing.evaluation.stability import calculate_stability
        stability_report = calculate_stability(all_metrics)
        stability_file = os.path.join(run_dir, "stability_report.json")
        with open(stability_file, "w") as f:
            json.dump(stability_report, f, indent=4)
        logger.info(f"Saved statistical stability report to {stability_file}")


if __name__ == "__main__":
    main()
