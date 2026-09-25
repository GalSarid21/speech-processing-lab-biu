import json
import os
import time
from datetime import datetime, UTC
from enum import Enum
from loguru import logger

from speech_processing.config.core import ExperimentMeta
from speech_processing.runners.base import parse_args, create_experiment_config
from speech_processing.data.dataset import load_icbhi_requests
from speech_processing.prompts.templates.judge.qwen import build_icbhi_judge_conversation
from speech_processing.models.audio import QwenAudioEngine
from speech_processing.models.text import GemmaTextModel
from speech_processing.models.judge import QwenJudge
from speech_processing.pipelines.inference import InferencePipeline
from speech_processing.pipelines.judge import JudgePipeline
from speech_processing.pipelines.base import release_vram
from speech_processing.evaluation.stability import calculate_stability

class ExperimentVersion(Enum):
    v1 = ExperimentMeta(
        experiment_name="baseline",
        prompt="Describe the respiratory cycle and the acoustic signature. Respond with the diagnosis in the format 'Final Diagnosis: [diagnosis]'.",
        max_new_tokens=256,
        batch_size=8
    )
    v2 = ExperimentMeta(
        experiment_name="cot",
        prompt="Describe the respiratory cycle and the acoustic signature step by step. Then, respond with the diagnosis in the format 'Final Diagnosis: [diagnosis]'.",
        max_new_tokens=512,
        batch_size=4
    )
    v3 = ExperimentMeta(
        experiment_name="few_shot",
        prompt="""Here are some examples of respiratory audio segments and their diagnoses. Listen to these examples and learn the acoustic features of each condition. Then, evaluate the final test audio segment. Respond with the diagnosis in the format 'Final Diagnosis: [diagnosis]'.

Examples:
- Normal vesicular breath sounds without any adventitious sounds indicate a Healthy patient.
- Early inspiratory coarse crackles and expiratory wheezes indicate COPD.
- High-pitched expiratory wheezes and fine inspiratory crackles indicate Bronchiolitis.

Test Audio: Listen to the following audio and provide the diagnosis.""",
        max_new_tokens=256,
        batch_size=8
    )

    @classmethod
    def get_version(cls, version_str: str) -> 'ExperimentVersion':
        try:
            return cls[version_str.lower()]
        except KeyError:
            raise ValueError(f"Unknown prompt version: {version_str}. Available versions: {[e.name for e in cls]}")

def run_icbhi(args):
    experiment_meta = ExperimentVersion.get_version(args.experiment).value
    is_text_only = getattr(experiment_meta, "experiment_name", "") == "text_only_llm"

    config = create_experiment_config(
        args, 
        experiment_meta, 
        dataset_id="DynamicSuperb/RespiratorySoundClassification_ICBHI2017",
        target_labels=["COPD", "No potential disease detected", "Healthy"]
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"icbhi_{experiment_meta.experiment_name}_{timestamp}"
    run_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    logger.info("--- [PHASE 1] DATA LOADING & PREPARATION ---")
    dataset_items = load_icbhi_requests(config.dataset, is_text_only=is_text_only)
    requests = []
    ground_truths = []
    for req, gt in dataset_items:
        req.instruction = experiment_meta.prompt
        requests.append(req)
        ground_truths.append(gt)

    logger.info("--- [PHASE 2] INITIALIZING INFERENCE ENGINE ---")
    if is_text_only:
        inference_engine = GemmaTextModel(config.text_model)
    else:
        inference_engine = QwenAudioEngine(config.audio_model)
    inference_pipeline = InferencePipeline(inference_engine)

    logger.info(f"--- [PHASE 3] RUNNING INFERENCE ({args.runs} RUNS) ---")
    temp_files = []
    final_outputs = []
    metrics_outputs = []
    
    for i in range(args.runs):
        suffix = f"_run{i+1}" if args.runs > 1 else ""
        temp_path = os.path.join(run_dir, f"temp_audio_preds{suffix}.jsonl")
        final_path = os.path.join(run_dir, f"raw_output{suffix}.jsonl")
        metrics_path = os.path.join(run_dir, f"metrics{suffix}.txt")
        
        temp_files.append(temp_path)
        final_outputs.append(final_path)
        metrics_outputs.append(metrics_path)
        
        inference_pipeline.run(requests, ground_truths, temp_path)

    # Garbage Collect Inference Engine
    inference_pipeline = None
    inference_engine = None
    release_vram()
    logger.info("Waiting 30 seconds before loading Judge...")
    time.sleep(30)

    logger.info("--- [PHASE 4] INITIALIZING JUDGE ENGINE ---")
    judge_engine = QwenJudge(config.judge, template=build_icbhi_judge_conversation)
    judge_pipeline = JudgePipeline(judge_engine)

    logger.info(f"--- [PHASE 5] RUNNING JUDGE EVALUATION ({args.runs} RUNS) ---")
    all_metrics = []
    for i in range(args.runs):
        metrics = judge_pipeline.run(temp_files[i], final_outputs[i], metrics_outputs[i], i)
        all_metrics.append(metrics)
        
    judge_pipeline = None
    judge_engine = None
    release_vram()

    if args.runs > 1:
        logger.info("--- [PHASE 6] CALCULATING STABILITY ---")
        stability_report = calculate_stability(all_metrics)
        stability_file = os.path.join(run_dir, "stability_report.json")
        with open(stability_file, "w") as f:
            json.dump(stability_report, f, indent=4)
        logger.info(f"Saved statistical stability report to {stability_file}")

if __name__ == "__main__":
    run_icbhi(parse_args("Run ICBHI Speech Processing Pipeline"))
