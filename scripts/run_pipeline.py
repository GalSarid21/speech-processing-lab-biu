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


def run_audio_phase(config: AppConfig, temp_file_path: str, prompt_version: str = "original"):
    logger.info("--- [PHASE 1] AUDIO INFERENCE ---")
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
        for req, _ in dataset_items:
            req.instruction = custom_instruction

    requests = [req for req, _ in dataset_items]
    ground_truths = [gt for _, gt in dataset_items]

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
        f"Audio inference complete. Wrote {len(responses)} results to {temp_file_path}."
    )
    # audio_engine goes out of scope here, making it eligible for GC.


def run_judge_phase(config: AppConfig, temp_file_path: str, output_file_path: str, metrics_file_path: str):
    logger.info("--- [PHASE 2] JUDGE EVALUATION ---")

    # Inject template
    template = QwenICBHI2017JudgeTemplate()
    judge_engine = QwenJudge(config.judge, template=template)

    judge_requests = []
    if os.path.exists(temp_file_path):
        with open(temp_file_path, "r") as f:
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
        logger.warning(f"No requests found in {temp_file_path}. Exiting.")
        return

    evaluations = judge_engine.batch_evaluate(judge_requests)

    with open(output_file_path, "w") as f:
        f.writelines(eval_resp.model_dump_json() + "\n" for eval_resp in evaluations)

    from speech_processing.evaluation.metrics import calculate_metrics
    metrics = calculate_metrics(evaluations)

    print("\n==============================================")
    print("FINAL EVALUATION REPORT")
    print("==============================================")

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
            f"--- AGGREGATE METRICS ---\n"
            f"Avg Acoustic Accuracy:   {metrics.avg_acoustic_pct:.2f}%\n"
            f"Avg Diagnostic Accuracy: {metrics.avg_diagnostic_pct:.2f}%\n"
            f"Hallucination Rate:      {metrics.hallucination_rate_pct:.2f}%\n\n"
            f"Classification Report:\n"
            f"{metrics.classification_report}\n"
        )
        print(f"\n{report}")
        
        with open(metrics_file_path, "w") as f:
            f.write(report)
        logger.info(f"Aggregate metrics saved to {metrics_file_path}")
    else:
        print("\nAggregate Dataset Accuracy: 0/10 (0%)")


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
    args = parser.parse_args()

    # Pass the CLI arguments to AppConfig
    # We update the default DatasetConfig inside AppConfig to respect --num-samples
    config = AppConfig(output_dir=args.output_dir)
    config.dataset.num_samples = args.num_samples

    # Fetch experiment metadata
    from speech_processing.prompts.templates.audio.experiments import ExperimentVersion
    experiment_meta = ExperimentVersion.get_version(args.experiment).value

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_meta.experiment_name}_{timestamp}"
    run_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    temp_file = os.path.join(run_dir, "temp_audio_preds.jsonl")
    final_output = os.path.join(run_dir, "raw_output.jsonl")
    metrics_output = os.path.join(run_dir, "metrics.txt")

    # Phase 1: Audio Model
    run_audio_phase(config, temp_file, prompt_version=args.experiment)

    # Free memory explicitly instead of relying on process termination
    release_vram()
    
    logger.info("GPU Memory freed. Waiting 60 seconds before loading Judge model...")
    time.sleep(60)

    # Phase 2: Judge Model
    run_judge_phase(config, temp_file, final_output, metrics_output)
    
    # Final cleanup (optional but good practice)
    release_vram()


if __name__ == "__main__":
    main()
