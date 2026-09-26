import json
import os
import time
from datetime import datetime, UTC
from enum import Enum
from loguru import logger

from speech_processing.config.core import ExperimentMeta
from speech_processing.runners.base import parse_args
from speech_processing.data.dataset import load_mmar_requests
from speech_processing.prompts.templates.judge.qwen import build_mmar_judge_conversation
from speech_processing.models.audio import VoxtralAudioEngine
from speech_processing.models.text import GemmaTextModel
from speech_processing.models.judge import QwenJudge
from speech_processing.pipelines.inference import InferencePipeline
from speech_processing.pipelines.judge import JudgePipeline
from speech_processing.pipelines.base import release_vram
from speech_processing.evaluation.stability import calculate_stability

class ExperimentVersion(Enum):
    v1 = ExperimentMeta(
        experiment_name="baseline",
        prompt="Listen to the audio and answer the multiple-choice question. You MUST respond entirely in English. Respond with the exact text of the correct choice.",
        max_new_tokens=256,
        batch_size=8
    )
    v2 = ExperimentMeta(
        experiment_name="cot",
        prompt="First, carefully listen to the audio file. You MUST respond entirely in English. Describe what you hear logically (e.g., speakers, environment, spoken content) and acoustically (e.g., pitch, tone, speech quality, background noise). Then, read the question and choices. Finally, select the choice that best matches your analysis.\n\nFormat your response exactly as follows:\n<analysis>\n[Your detailed logical and acoustic analysis in English here]\n</analysis>\n<answer>\n[The exact text of the correct choice]\n</answer>",
        max_new_tokens=256,
        batch_size=8
    )
    v3 = ExperimentMeta(
        experiment_name="transcript_augmented",
        prompt="Listen to the audio and read the provided whisper transcript. Then, answer the multiple-choice question. Respond with the exact text of the correct choice.",
        max_new_tokens=256,
        batch_size=8
    )
    v4 = ExperimentMeta(
        experiment_name="transcript_augmented_cot",
        prompt="First, read the provided whisper transcript and carefully listen to the audio file. Describe what you hear logically (e.g., speakers, environment) and acoustically (e.g., pitch, tone, speech quality). Then, read the question and choices. Finally, select the choice that best matches your analysis.\n\nFormat your response exactly as follows:\n<analysis>\n[Your detailed logical and acoustic analysis here]\n</analysis>\n<answer>\n[The exact text of the correct choice]\n</answer>",
        max_new_tokens=1024,
        batch_size=8
    )
    v5 = ExperimentMeta(
        experiment_name="few_shot_text_only",
        prompt="Here are some examples of questions and answers. Read them, then listen to the final audio and answer the multiple-choice question. Respond with the exact text of the correct choice.",
        max_new_tokens=1024,
        batch_size=8
    )
    v6 = ExperimentMeta(
        experiment_name="few_shot_text_only_cot",
        prompt="Here are examples of questions and reasoning chains. Read them, then listen to the final test audio. Describe the test audio logically (speakers, context) and acoustically (tone, pitch, speech quality). Finally, read the question and choices, select the best match.\n\nFormat your response exactly as follows:\n<analysis>\n[Your detailed logical and acoustic analysis here]\n</analysis>\n<answer>\n[The exact text of the correct choice]\n</answer>",
        max_new_tokens=1024,
        batch_size=8
    )
    v7 = ExperimentMeta(
        experiment_name="few_shot_audio",
        prompt="Here are some examples of audio, questions, and answers. Listen to them, then evaluate the final test audio and answer the multiple-choice question. Respond with the exact text of the correct choice.",
        max_new_tokens=1024,
        batch_size=8
    )
    v8 = ExperimentMeta(
        experiment_name="few_shot_audio_cot",
        prompt="Here are examples of audio, questions, and reasoning chains. Listen to them, then carefully evaluate the final test audio. Describe it logically (speakers, context) and acoustically (tone, pitch, speech quality). Finally, select the best matching choice.\n\nFormat your response exactly as follows:\n<analysis>\n[Your detailed logical and acoustic analysis here]\n</analysis>\n<answer>\n[The exact text of the correct choice]\n</answer>",
        batch_size=4,
        max_new_tokens=1024
    )
    v9 = ExperimentMeta(
        experiment_name="few_shot_audio_transcript",
        prompt="Here are examples with audio, whisper transcripts, questions, and answers. Evaluate the final test audio/transcript and answer the multiple-choice question. Respond with the exact text of the correct choice.",
        max_new_tokens=1024,
        batch_size=8
    )
    v10 = ExperimentMeta(
        experiment_name="few_shot_audio_transcript_cot",
        prompt="Here are examples with audio, transcripts, questions, and reasoning chains. Evaluate the final test audio and transcript. Describe it logically and acoustically. Finally, select the best matching choice.\n\nFormat your response exactly as follows:\n<analysis>\n[Your detailed logical and acoustic analysis here]\n</analysis>\n<answer>\n[The exact text of the correct choice]\n</answer>",
        batch_size=4,
        max_new_tokens=1024
    )
    v11 = ExperimentMeta(
        experiment_name="role_prompting",
        prompt="You are an expert socio-linguist and audio analyst. Evaluate the following audio and answer the multiple-choice question. Respond with the exact text of the correct choice.",
        max_new_tokens=256,
        batch_size=8
    )
    v12 = ExperimentMeta(
        experiment_name="multi_turn_decomposition",
        prompt=["Describe the audio and the speaker's tone.", "Based on your description, answer the multiple-choice question. Respond with the exact text of the correct choice."],
        batch_size=2,
        max_new_tokens=256
    )
    v13 = ExperimentMeta(
        experiment_name="text_only_llm",
        prompt="Read the provided whisper transcript. Answer the multiple-choice question based solely on the transcript. Respond with the exact text of the correct choice.",
        max_new_tokens=256,
        batch_size=8
    )

    @classmethod
    def get_version(cls, version_str: str) -> 'ExperimentVersion':
        try:
            return cls[version_str.lower()]
        except KeyError:
            raise ValueError(f"Unknown prompt version: {version_str}. Available versions: {[e.name for e in cls]}")

from speech_processing.config.factories import create_mmar_config

def run_mmar(args):
    experiment_meta = ExperimentVersion.get_version(args.experiment).value
    is_text_only = getattr(experiment_meta, "experiment_name", "") == "text_only_llm"

    config = create_mmar_config(args, experiment_meta)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"mmar_{experiment_meta.experiment_name}_{timestamp}"
    run_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    logger.info("--- [PHASE 1] DATA LOADING & PREPARATION ---")
    dataset_items = load_mmar_requests(config.dataset, experiment_meta=experiment_meta, is_text_only=is_text_only)
    
    few_shot_turns = []
    if "few_shot" in experiment_meta.experiment_name:
        from speech_processing.data.dataset import get_mmar_few_shot_turns
        few_shot_turns = get_mmar_few_shot_turns(config.dataset, experiment_meta, is_text_only=is_text_only, num_shots=3)

    requests = []
    ground_truths = []
    for req, gt in dataset_items:
        if few_shot_turns:
            req.few_shot_turns = few_shot_turns
        requests.append(req)
        ground_truths.append(gt)

    logger.info("--- [PHASE 2] INITIALIZING INFERENCE ENGINE ---")
    if is_text_only:
        inference_engine = GemmaTextModel(config.text_model)
    else:
        inference_engine = VoxtralAudioEngine(config.audio_model)
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
    judge_engine = QwenJudge(config.judge, template_func=build_mmar_judge_conversation)
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
    run_mmar(parse_args("Run MMAR Speech Processing Pipeline"))
