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


def run_audio_phase(config: AppConfig, temp_file_path: str):
    logger.info("--- [PHASE 1] AUDIO INFERENCE ---")
    audio_engine = QwenAudioEngine(config.audio_model)

    # Load exactly 100 samples from the real dataset
    dataset_items = load_icbhi_requests(num_samples=100)
    requests = [req for req, _ in dataset_items]
    ground_truths = [gt for _, gt in dataset_items]

    # Evaluate using the maximal batch size internally configured in the engine
    responses = audio_engine.batch_infer(requests)

    with open(temp_file_path, "w") as f:
        f.writelines(
            json.dumps(
                {
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


def run_judge_phase(config: AppConfig, temp_file_path: str, output_file_path: str):
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

    print("\n==============================================")
    print("FINAL EVALUATION REPORT")
    print("==============================================")

    total_rate = 0.0
    for i, eval_item in enumerate(evaluations):
        score = eval_item.evaluation.rate
        total_rate += score

        print(f"\n--- Sample {i + 1} ---")
        print(f"Instruction: {eval_item.request.instruction}")
        print(f"True Label:  {eval_item.request.ground_truth}")
        print(f"Qwen Output: {eval_item.request.generated_text}")
        print(f"Judge Score: {score}/10")
        print(f"Reasoning:   {eval_item.evaluation.reasoning}")
        print("-" * 30)

    if evaluations:
        avg_rate = total_rate / len(evaluations)
        accuracy_percentage = (avg_rate / 10.0) * 100
        print(
            f"\nAggregate Dataset Accuracy: {avg_rate:.3f}/10 ({accuracy_percentage:.3f}%)"
        )
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
    config = AppConfig()
    os.makedirs("./results", exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    temp_file = f"./results/temp_audio_preds_{timestamp}.jsonl"
    final_output = (
        f"./results/qwen2-audio-baseline-qwen3.6-35b-a3b-judge-{timestamp}.jsonl"
    )

    # Phase 1: Audio Model
    run_audio_phase(config, temp_file)

    # Free memory explicitly instead of relying on process termination
    release_vram()
    
    logger.info("GPU Memory freed. Waiting 60 seconds before loading Judge model...")
    time.sleep(60)

    # Phase 2: Judge Model
    run_judge_phase(config, temp_file, final_output)
    
    # Final cleanup (optional but good practice)
    release_vram()


if __name__ == "__main__":
    main()
