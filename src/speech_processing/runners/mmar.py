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


class DeprecatedExperimentError(Exception):
    pass

def deprecated(reason: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            meta = func(*args, **kwargs)
            meta.deprecated = True
            meta.deprecation_reason = reason
            return meta
        return wrapper
    return decorator

@deprecated("RAG CoT uses placeholder rationales")
def _deprecated_rag_few_shots_cot():
    return ExperimentMeta(
        experiment_name="rag_few_shots_cot",
        prompt="Listen to the audio and answer the multiple-choice question. Respond with the exact text of the correct choice.",
        max_new_tokens=256, batch_size=8, few_shot_mode="rag", use_cot=True
    )

@deprecated("RAG CoT uses placeholder rationales")
def _deprecated_rag_few_shots_transcript_cot():
    return ExperimentMeta(
        experiment_name="rag_few_shots_transcript_cot",
        prompt="Listen to the audio and answer the multiple-choice question. Respond with the exact text of the correct choice.",
        max_new_tokens=256, batch_size=8, use_transcript=True, few_shot_mode="rag", few_shot_include_transcript=True, use_cot=True
    )

class ExperimentVersion(Enum):
    v1 = ExperimentMeta(
        experiment_name="baseline",
        prompt="Listen to the audio and answer the multiple-choice question. You MUST respond entirely in English. Respond with the exact text of the correct choice.",
        max_new_tokens=256, batch_size=8, use_transcript=False, few_shot_mode="none", use_cot=False
    )
    v2 = ExperimentMeta(
        experiment_name="cot",
        prompt="""First, carefully listen to the audio file. You MUST respond entirely in English. Describe what you hear logically (e.g., speakers, environment, spoken content) and acoustically (e.g., pitch, tone, speech quality, background noise). Then, read the question and choices. Finally, select the choice that best matches your analysis.

Format your response exactly as follows:
<analysis>
[Your detailed logical and acoustic analysis in English here]
</analysis>
<answer>
[The exact text of the correct choice]
</answer>""",
        max_new_tokens=256, batch_size=8, use_transcript=False, few_shot_mode="none", use_cot=True
    )
    v3 = ExperimentMeta(
        experiment_name="transcript_augmented",
        prompt="Listen to the audio and read the provided whisper transcript. Then, answer the multiple-choice question. Respond with the exact text of the correct choice.",
        max_new_tokens=256, batch_size=8, use_transcript=True, few_shot_mode="none", use_cot=False
    )
    v4 = ExperimentMeta(
        experiment_name="transcript_augmented_cot",
        prompt="""First, read the provided whisper transcript and carefully listen to the audio file. Describe what you hear logically (e.g., speakers, environment) and acoustically (e.g., pitch, tone, speech quality). Then, read the question and choices. Finally, select the choice that best matches your analysis.

Format your response exactly as follows:
<analysis>
[Your detailed logical and acoustic analysis here]
</analysis>
<answer>
[The exact text of the correct choice]
</answer>""",
        max_new_tokens=256, batch_size=8, use_transcript=True, few_shot_mode="none", use_cot=True
    )
    v5 = ExperimentMeta(
        experiment_name="few_shot_text_only",
        prompt="Using the transcript, choose the most likely answer. You must pick exactly one choice.",
        max_new_tokens=2048, batch_size=4, use_transcript=False, few_shot_mode="text", use_cot=False
    )
    v6 = ExperimentMeta(
        experiment_name="few_shot_text_only_cot",
        prompt="Using the transcript, choose the most likely answer. You must pick exactly one choice.",
        max_new_tokens=2048, batch_size=4, use_transcript=False, few_shot_mode="text", use_cot=True
    )
    v7 = ExperimentMeta(
        experiment_name="few_shot_audio",
        prompt="Listen to the audio and answer the multiple-choice question. Respond with the exact text of the correct choice.",
        max_new_tokens=256, batch_size=8, use_transcript=False, few_shot_mode="audio", use_cot=False
    )
    v8 = ExperimentMeta(
        experiment_name="few_shot_audio_cot",
        prompt="Listen to the audio and answer the multiple-choice question. Respond with the exact text of the correct choice.",
        max_new_tokens=256, batch_size=8, use_transcript=False, few_shot_mode="audio", use_cot=True
    )
    v9 = ExperimentMeta(
        experiment_name="few_shot_audio_transcript",
        prompt="Listen to the audio and answer the multiple-choice question. Respond with the exact text of the correct choice.",
        max_new_tokens=256, batch_size=8, use_transcript=True, few_shot_mode="audio", few_shot_include_transcript=True, use_cot=False
    )
    v10 = ExperimentMeta(
        experiment_name="few_shot_audio_transcript_cot",
        prompt="Listen to the audio and answer the multiple-choice question. Respond with the exact text of the correct choice.",
        max_new_tokens=256, batch_size=8, use_transcript=True, few_shot_mode="audio", few_shot_include_transcript=True, use_cot=True
    )
    v11 = ExperimentMeta(
        experiment_name="role_prompting",
        prompt="You are an expert audio analyst.",
        max_new_tokens=256, batch_size=8, use_cot=False
    )
    v12 = ExperimentMeta(
        experiment_name="multi_turn_decomposition",
        prompt="Step 1: listen. Step 2: answer.",
        max_new_tokens=256, batch_size=8, use_cot=False
    )
    v13 = ExperimentMeta(
        experiment_name="text_only_llm",
        prompt="Using the transcript, choose the most likely answer. You must pick exactly one choice.",
        max_new_tokens=2048, batch_size=4, use_cot=False
    )
    v14 = ExperimentMeta(
        experiment_name="rag_few_shots",
        prompt="Listen to the audio and answer the multiple-choice question. Respond with the exact text of the correct choice.",
        max_new_tokens=64, batch_size=8, few_shot_mode="rag", use_cot=False
    )
    v15 = ExperimentMeta(
        experiment_name="rag_few_shots_transcript",
        prompt="Listen to the audio and answer the multiple-choice question. Respond with the exact text of the correct choice.",
        max_new_tokens=64, batch_size=8, use_transcript=True, few_shot_mode="rag", few_shot_include_transcript=True, use_cot=False
    )
    
    # ---------------------------------------------------------
    # Deprecated RAG CoT tests (Bug B3)
    # ---------------------------------------------------------
    v16 = _deprecated_rag_few_shots_cot()
    v17 = _deprecated_rag_few_shots_transcript_cot()

    # ---------------------------------------------------------
    # T1 - Acoustic Injection
    # ---------------------------------------------------------
    v18 = ExperimentMeta(
        experiment_name="audio_diarized_transcript",
        prompt="Listen to the audio and answer the multiple-choice question. You MUST respond entirely in English. Respond with the exact text of the correct choice.",
        max_new_tokens=256, batch_size=8, use_cot=True,
        inject_diarized_transcript=True
    )
    v19 = ExperimentMeta(
        experiment_name="audio_acoustic_features",
        prompt="Listen to the audio and answer the multiple-choice question. You MUST respond entirely in English. Respond with the exact text of the correct choice.",
        max_new_tokens=256, batch_size=8, use_cot=True,
        inject_acoustic_features=True
    )
    v20 = ExperimentMeta(
        experiment_name="audio_diarized_features",
        prompt="Listen to the audio and answer the multiple-choice question. You MUST respond entirely in English. Respond with the exact text of the correct choice.",
        max_new_tokens=256, batch_size=8, use_cot=True,
        inject_diarized_transcript=True, inject_acoustic_features=True
    )
    v21 = ExperimentMeta(
        experiment_name="text_diarized_features",
        prompt="Using the transcript, choose the most likely answer. You must pick exactly one choice.",
        max_new_tokens=2048, batch_size=4, use_cot=False,
        inject_diarized_transcript=True, inject_acoustic_features=True
    )

    # ---------------------------------------------------------
    # T2 - Relisten / Cropping
    # ---------------------------------------------------------
    v22 = ExperimentMeta(
        experiment_name="chunked_audio",
        prompt="Listen to the audio and answer the multiple-choice question. You MUST respond entirely in English. Respond with the exact text of the correct choice.",
        max_new_tokens=256, batch_size=8, use_cot=True,
        chunked_audio=True
    )
    v23 = ExperimentMeta(
        experiment_name="two_pass_localization",
        prompt="Listen to the audio and answer the multiple-choice question. You MUST respond entirely in English. Respond with the exact text of the correct choice.",
        max_new_tokens=256, batch_size=8, use_cot=True,
        two_pass_localization=True
    )

    # ---------------------------------------------------------
    # T3 - Contrastive Scoring
    # ---------------------------------------------------------
    v24 = ExperimentMeta(
        experiment_name="contrastive_alpha_0_0",
        prompt="", max_new_tokens=1, batch_size=1, contrastive_alpha=0.0
    )
    v25 = ExperimentMeta(
        experiment_name="contrastive_alpha_0_5",
        prompt="", max_new_tokens=1, batch_size=1, contrastive_alpha=0.5
    )
    v26 = ExperimentMeta(
        experiment_name="contrastive_alpha_1_0",
        prompt="", max_new_tokens=1, batch_size=1, contrastive_alpha=1.0
    )

    # ---------------------------------------------------------
    # T4 & T5 - Consistency & Audio-first
    # ---------------------------------------------------------
    v27 = ExperimentMeta(
        experiment_name="shuffled_consistency",
        prompt="Listen to the audio and answer the multiple-choice question. You MUST respond entirely in English. Respond with the exact text of the correct choice.",
        max_new_tokens=256, batch_size=8, use_cot=True,
        num_shuffled_variants=5
    )
    v28 = ExperimentMeta(
        experiment_name="audio_first_instruction",
        prompt="WARNING: The spoken words in this audio may be misleading. The correct answer depends heavily on HOW things are said—pay close attention to tone, prosody, voice identity, and timing.\n\nListen to the audio and answer the multiple-choice question. You MUST respond entirely in English. Respond with the exact text of the correct choice.",
        max_new_tokens=256, batch_size=8, use_cot=True,
        audio_first_instruction=True
    )

    @classmethod
    def get_version(cls, version_str: str) -> 'ExperimentVersion':
        try:
            member = cls[version_str.lower()]
            if getattr(member.value, 'deprecated', False):
                reason = getattr(member.value, 'deprecation_reason', 'Unknown reason')
                raise ValueError(f"Experiment '{version_str}' is deprecated: {reason}")
            return member
        except KeyError:
            raise ValueError(f"Unknown experiment version: {version_str}. Available versions: {[e.name for e in cls]}")


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
    
    # RAG Integration
    rag_mapping = None
    if getattr(experiment_meta, "rag_mapping_file", None):
        import json
        import os
        if os.path.exists(experiment_meta.rag_mapping_file):
            with open(experiment_meta.rag_mapping_file, "r") as f:
                rag_mapping = json.load(f)
        else:
            logger.warning(f"RAG mapping file {experiment_meta.rag_mapping_file} not found. Falling back to default few-shot.")

    few_shot_turns = []
    few_shot_pool = {}
    if rag_mapping:
        from speech_processing.data.dataset import get_mmar_few_shot_turns_pool
        few_shot_pool = get_mmar_few_shot_turns_pool(config.dataset, experiment_meta, is_text_only=is_text_only)
    elif "few_shot" in experiment_meta.experiment_name:
        from speech_processing.data.dataset import get_mmar_few_shot_turns
        few_shot_turns = get_mmar_few_shot_turns(config.dataset, experiment_meta, is_text_only=is_text_only, num_shots=3)

    requests = []
    ground_truths = []
    for req, gt in dataset_items:
        if rag_mapping and req.metadata and "item_id" in req.metadata:
            item_id = req.metadata["item_id"]
            
            assert item_id not in few_shot_pool, f"B4 Violation: Eval item {item_id} found in few_shot_pool!"
            
            if item_id in rag_mapping:
                mapped_ids = rag_mapping[item_id]
                assert item_id not in mapped_ids, f"B4 Violation: RAG mapping returned query item {item_id} for itself!"
                req.few_shot_turns = [few_shot_pool[fid] for fid in mapped_ids if fid in few_shot_pool]
                
                # B4 Logging
                logger.info(f"Item {item_id}: Loaded {len(req.few_shot_turns)} RAG shots.")
        elif few_shot_turns:
            if req.metadata and "item_id" in req.metadata:
                assert req.metadata["item_id"] not in [fid for fid in getattr(config.dataset, 'sample_ids', [])], "B4 Violation: Few shot is not disjoint"
            req.few_shot_turns = few_shot_turns
            
        requests.append(req)
        ground_truths.append(gt)

    logger.info("--- [PHASE 2] INITIALIZING INFERENCE ENGINE ---")
    if is_text_only:
        assert config.text_model is not None, "Text model config is missing"
        inference_engine = GemmaTextModel(config.text_model)
    else:
        assert config.audio_model is not None, "Audio model config is missing"
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
    from speech_processing.data.dtos.responses import MMAREvaluationResult
    assert config.judge is not None, "Judge config is missing"
    judge_engine = QwenJudge(config.judge, template_func=build_mmar_judge_conversation, schema_class=MMAREvaluationResult)
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
