import json
import os
import time
from datetime import UTC, datetime
from enum import Enum

from loguru import logger

from speech_processing.config.core import AppConfig, ExperimentMeta
from speech_processing.config.factories import create_mmar_config
from speech_processing.data.dataset import (
    get_mmar_few_shot_turns,
    get_mmar_few_shot_turns_pool,
    load_mmar_requests,
    validate_few_shot_disjointness,
)
from speech_processing.data.dtos import BaseRequest, MMAREvaluationResult
from speech_processing.evaluation.metrics import calculate_mmar_metrics
from speech_processing.evaluation.stability import calculate_stability
from speech_processing.models.audio import VoxtralAudioEngine
from speech_processing.models.contrastive import ContrastiveChoiceScorer
from speech_processing.models.judge import QwenJudge
from speech_processing.models.presentation import build_presentation
from speech_processing.models.text import GemmaTextModel
from speech_processing.pipelines.aggregation import (
    IdentityAggregator,
    ResponseAggregator,
    ShuffledMajorityVoteAggregator,
)
from speech_processing.pipelines.base import log_gpu_memory, release_vram
from speech_processing.pipelines.inference import InferencePipeline
from speech_processing.pipelines.judge import JudgePipeline, ParserOnlyPipeline
from speech_processing.prompts.templates.judge.qwen import build_mmar_judge_conversation
from speech_processing.runners.base import parse_args, write_run_config
from speech_processing.utils.consts import (
    COT_END_TAG,
    COT_START_TAG,
    DEFAULT_NUM_FEW_SHOTS,
    JUDGE_LOAD_DELAY_S,
    RAG_STOP_SEQUENCES,
)
from speech_processing.utils.exceptions import DatasetIntegrityError, DeprecatedExperimentError

RAG_MAPPING_FILE = "data/mmar_rag_mapping.json"

ENGLISH_ONLY = "You MUST respond entirely in English."
ANSWER_WITH_LETTER = "Respond with only the letter of the correct choice."
FINAL_ANSWER_LINE = "End your response with a final line of the form 'Final answer: <letter>'."

COT_FORMAT = (
    "Format your response exactly as follows:\n"
    f"{COT_START_TAG}\n[Your detailed logical and acoustic analysis here]\n{COT_END_TAG}\n"
    "<answer>\n[The letter of the correct choice]\n</answer>"
)
COT_FORMAT_ENGLISH = COT_FORMAT.replace("analysis here]", "analysis in English here]")

BASELINE_PROMPT = f"Listen to the audio and answer the multiple-choice question. {ENGLISH_ONLY} {ANSWER_WITH_LETTER}"

_COT_PROMPT = (
    "First, carefully listen to the audio file. You MUST respond entirely in English. Describe what you hear "
    "logically (e.g., speakers, environment, spoken content) and acoustically (e.g., pitch, tone, speech quality, "
    "background noise). Then, read the question and choices. Finally, select the choice that best matches your "
    "analysis.\n\n" + COT_FORMAT_ENGLISH
)
_TRANSCRIPT_PROMPT = (
    "Listen to the audio and read the provided whisper transcript. Then, answer the multiple-choice question. "
    f"{ANSWER_WITH_LETTER}"
)
_TRANSCRIPT_COT_PROMPT = (
    "First, read the provided whisper transcript and carefully listen to the audio file. Describe what you hear "
    "logically (e.g., speakers, environment) and acoustically (e.g., pitch, tone, speech quality). Then, read the "
    "question and choices. Finally, select the choice that best matches your analysis.\n\n" + COT_FORMAT
)
_FEW_SHOT_TEXT_PROMPT = (
    "Here are some examples of questions and answers. Study them, then answer the final multiple-choice question. "
    f"{ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_FEW_SHOT_AUDIO_PROMPT = f"Listen to the audio and answer the multiple-choice question. {ANSWER_WITH_LETTER}"
_ROLE_PROMPT = (
    "You are an expert socio-linguist and audio analyst. Listen to the audio and answer the multiple-choice "
    f"question. {ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_TEXT_ONLY_PROMPT = (
    "Using the transcript, choose the most likely answer. You must pick exactly one choice. " + FINAL_ANSWER_LINE
)
_TEXT_EVIDENCE_PROMPT = (
    "Using the diarized transcript and the measured acoustic features below, choose the most likely answer. "
    "You must pick exactly one choice. " + FINAL_ANSWER_LINE
)
_DIARIZED_PROMPT = (
    "Listen to the audio. A diarized transcript is provided below: use it to track who speaks and when, but judge "
    "HOW things are said from the audio itself. Then answer the multiple-choice question. "
    f"{ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_FEATURES_PROMPT = (
    "Listen to the audio. Per-speaker acoustic measurements are provided below: use them together with what you "
    f"hear, not instead of it. Then answer the multiple-choice question. {ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_DIARIZED_FEATURES_PROMPT = (
    "Listen to the audio. A diarized transcript and per-speaker acoustic measurements are provided below: use both "
    "together with what you hear, not instead of it. Then answer the multiple-choice question. "
    f"{ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_CHUNKED_PROMPT = (
    "The audio is given as consecutive labeled segments followed by the full recording. Use the segments to locate "
    "the evidence, then answer the multiple-choice question about the full recording. "
    f"{ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_LOCALIZATION_TURN_1 = (
    "Listen to the audio and identify the single time span that contains the evidence needed to answer the "
    "question. Do NOT answer the question yet. Respond ONLY with JSON of the form "
    '{"start": <seconds>, "end": <seconds>}.'
)
_LOCALIZATION_TURN_2 = (
    "Here is a cropped segment covering the span you identified. Using it together with what you already heard, "
    f"answer the multiple-choice question. {ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_AUDIO_FIRST_WARNING = (
    "WARNING: The spoken words in this audio may be misleading. The correct answer depends heavily on HOW things "
    "are said—pay close attention to tone, prosody, voice identity, and timing."
)

_RAG_COT_DEPRECATION = (
    "B3: RAG few-shots run with a 64-token budget and the RAG stop sequences, which truncate the analysis block, "
    "so the CoT variant never produced a usable chain."
)
_DEPRECATED_TOKENS = 1
_DEPRECATED_BATCH = 1


# ---------------------------------------------------------------------------------------------
# Scope. V1 already explored the classic-prompting space on this dataset; V2's job is to re-run it
# under valid measurement and to test the new families. The reported set is the `core` tier: the
# baseline, the text-only control that defines the subset split, three classic representatives
# carried over from V1, and one representative of each new family (T1 evidence, T2 presentation,
# T3 contrastive, T4 consistency, T5 framing).
# Everything else is `future_work`: implemented, tested and runnable with --experiment, but not
# part of the reported results. Nothing here is deprecated except v15/v17.
# ---------------------------------------------------------------------------------------------
FUTURE_V1_VARIANT = (
    "Variant of a family V1 already covered; v2, v3 and v7 are the representatives re-run under valid measurement."
)
FUTURE_FEW_SHOT_VARIANT = "Few-shot variant; v7 is the reported arm because it was V1's strongest result."
FUTURE_RAG = "RAG retrieval was characterised in V1 and is not one of the new families this version tests."
FUTURE_T1_VARIANT = "Acoustic-evidence variant; v18 (diarized) and v20 (both sources) are the reported arms."
FUTURE_ALPHA_SWEEP = "Contrastive alpha sweep; v25 is the reported mid-point. Run the full sweep only if v25 moves."
FUTURE_MULTI_TURN = "Multi-turn decomposition overlaps with the T2 two-pass arm (v23), which is the reported one."


class ExperimentVersion(Enum):
    v1 = ExperimentMeta(
        experiment_name="baseline",
        prompt=BASELINE_PROMPT,
        max_new_tokens=256,
        batch_size=8,
    )
    v2 = ExperimentMeta(
        experiment_name="cot",
        prompt=_COT_PROMPT,
        max_new_tokens=256,
        batch_size=8,
        use_cot=True,
    )
    v3 = ExperimentMeta(
        experiment_name="transcript_augmented",
        prompt=_TRANSCRIPT_PROMPT,
        max_new_tokens=256,
        batch_size=8,
        use_transcript=True,
    )
    v4 = ExperimentMeta(
        experiment_name="transcript_augmented_cot",
        prompt=_TRANSCRIPT_COT_PROMPT,
        max_new_tokens=1024,
        batch_size=8,
        use_transcript=True,
        use_cot=True,
        tier="future_work",
        future_work_reason=FUTURE_V1_VARIANT,
    )
    v5 = ExperimentMeta(
        experiment_name="few_shot_text_only",
        prompt=_FEW_SHOT_TEXT_PROMPT,
        max_new_tokens=1024,
        batch_size=8,
        few_shot_mode="text",
        tier="future_work",
        future_work_reason=FUTURE_FEW_SHOT_VARIANT,
    )
    v6 = ExperimentMeta(
        experiment_name="few_shot_text_only_cot",
        prompt=f"{_FEW_SHOT_TEXT_PROMPT}\n\n{COT_FORMAT}",
        max_new_tokens=1024,
        batch_size=8,
        few_shot_mode="text",
        use_cot=True,
        tier="future_work",
        future_work_reason=FUTURE_FEW_SHOT_VARIANT,
    )
    v7 = ExperimentMeta(
        experiment_name="few_shot_audio",
        prompt=_FEW_SHOT_AUDIO_PROMPT,
        max_new_tokens=1024,
        batch_size=8,
        few_shot_mode="audio",
    )
    v8 = ExperimentMeta(
        experiment_name="few_shot_audio_cot",
        prompt=f"{_FEW_SHOT_AUDIO_PROMPT}\n\n{COT_FORMAT}",
        max_new_tokens=1024,
        batch_size=4,
        few_shot_mode="audio",
        use_cot=True,
        tier="future_work",
        future_work_reason=FUTURE_FEW_SHOT_VARIANT,
    )
    v9 = ExperimentMeta(
        experiment_name="few_shot_audio_transcript",
        prompt=_FEW_SHOT_AUDIO_PROMPT,
        max_new_tokens=1024,
        batch_size=8,
        use_transcript=True,
        few_shot_mode="audio",
        few_shot_include_transcript=True,
        tier="future_work",
        future_work_reason=FUTURE_FEW_SHOT_VARIANT,
    )
    v10 = ExperimentMeta(
        experiment_name="few_shot_audio_transcript_cot",
        prompt=f"{_FEW_SHOT_AUDIO_PROMPT}\n\n{COT_FORMAT}",
        max_new_tokens=1024,
        batch_size=4,
        use_transcript=True,
        few_shot_mode="audio",
        few_shot_include_transcript=True,
        use_cot=True,
        tier="future_work",
        future_work_reason=FUTURE_FEW_SHOT_VARIANT,
    )
    v11 = ExperimentMeta(
        experiment_name="role_prompting",
        prompt=_ROLE_PROMPT,
        max_new_tokens=256,
        batch_size=8,
        tier="future_work",
        future_work_reason=FUTURE_V1_VARIANT,
    )
    v12 = ExperimentMeta(
        experiment_name="multi_turn_decomposition",
        prompt=[
            "Describe the audio and the speaker's tone.",
            f"Based on your description, answer the multiple-choice question. {ANSWER_WITH_LETTER}",
        ],
        max_new_tokens=256,
        batch_size=2,
        tier="future_work",
        future_work_reason=FUTURE_MULTI_TURN,
    )
    v13 = ExperimentMeta(
        experiment_name="text_only_llm",
        prompt=_TEXT_ONLY_PROMPT,
        max_new_tokens=2048,
        batch_size=4,
        text_only=True,
        use_transcript=True,
    )
    v14 = ExperimentMeta(
        experiment_name="rag_few_shots",
        prompt=_FEW_SHOT_AUDIO_PROMPT,
        max_new_tokens=64,
        batch_size=8,
        few_shot_mode="rag",
        rag_mapping_file=RAG_MAPPING_FILE,
        stop=RAG_STOP_SEQUENCES,
        tier="future_work",
        future_work_reason=FUTURE_RAG,
    )
    v15 = ExperimentMeta(
        experiment_name="rag_few_shots_cot",
        prompt="",
        max_new_tokens=_DEPRECATED_TOKENS,
        batch_size=_DEPRECATED_BATCH,
        deprecated=True,
        deprecation_reason=_RAG_COT_DEPRECATION,
    )
    v16 = ExperimentMeta(
        experiment_name="rag_few_shots_transcript",
        prompt=_FEW_SHOT_AUDIO_PROMPT,
        max_new_tokens=64,
        batch_size=8,
        use_transcript=True,
        few_shot_mode="rag",
        few_shot_include_transcript=True,
        rag_mapping_file=RAG_MAPPING_FILE,
        stop=RAG_STOP_SEQUENCES,
        tier="future_work",
        future_work_reason=FUTURE_RAG,
    )
    v17 = ExperimentMeta(
        experiment_name="rag_few_shots_transcript_cot",
        prompt="",
        max_new_tokens=_DEPRECATED_TOKENS,
        batch_size=_DEPRECATED_BATCH,
        deprecated=True,
        deprecation_reason=_RAG_COT_DEPRECATION,
    )

    # ---------------------------------------------------------
    # T1 - Acoustic evidence injection
    # ---------------------------------------------------------
    v18 = ExperimentMeta(
        experiment_name="audio_diarized_transcript",
        prompt=_DIARIZED_PROMPT,
        max_new_tokens=256,
        batch_size=8,
        inject_diarized_transcript=True,
    )
    v19 = ExperimentMeta(
        experiment_name="audio_acoustic_features",
        prompt=_FEATURES_PROMPT,
        max_new_tokens=256,
        batch_size=8,
        inject_acoustic_features=True,
        tier="future_work",
        future_work_reason=FUTURE_T1_VARIANT,
    )
    v20 = ExperimentMeta(
        experiment_name="audio_diarized_features",
        prompt=_DIARIZED_FEATURES_PROMPT,
        max_new_tokens=256,
        batch_size=8,
        inject_diarized_transcript=True,
        inject_acoustic_features=True,
    )
    v21 = ExperimentMeta(
        experiment_name="text_diarized_features",
        prompt=_TEXT_EVIDENCE_PROMPT,
        max_new_tokens=2048,
        batch_size=4,
        text_only=True,
        inject_diarized_transcript=True,
        inject_acoustic_features=True,
        tier="future_work",
        future_work_reason=FUTURE_T1_VARIANT,
    )

    # ---------------------------------------------------------
    # T2 - Audio presentation
    # ---------------------------------------------------------
    v22 = ExperimentMeta(
        experiment_name="chunked_audio",
        prompt=_CHUNKED_PROMPT,
        max_new_tokens=256,
        batch_size=8,
        presentation="chunked",
    )
    v23 = ExperimentMeta(
        experiment_name="two_pass_localization",
        prompt=[_LOCALIZATION_TURN_1, _LOCALIZATION_TURN_2],
        max_new_tokens=256,
        batch_size=8,
        presentation="two_pass_localization",
    )

    # ---------------------------------------------------------
    # T3 - Contrastive scoring
    # ---------------------------------------------------------
    v24 = ExperimentMeta(
        experiment_name="contrastive_alpha_0_0",
        prompt=BASELINE_PROMPT,
        max_new_tokens=1,
        batch_size=8,
        contrastive_alpha=0.0,
        tier="future_work",
        future_work_reason=FUTURE_ALPHA_SWEEP,
    )
    v25 = ExperimentMeta(
        experiment_name="contrastive_alpha_0_5",
        prompt=BASELINE_PROMPT,
        max_new_tokens=1,
        batch_size=8,
        contrastive_alpha=0.5,
    )
    v26 = ExperimentMeta(
        experiment_name="contrastive_alpha_1_0",
        prompt=BASELINE_PROMPT,
        max_new_tokens=1,
        batch_size=8,
        contrastive_alpha=1.0,
        tier="future_work",
        future_work_reason=FUTURE_ALPHA_SWEEP,
    )

    # ---------------------------------------------------------
    # T4 & T5 - Choice-order consistency & audio-first framing
    # ---------------------------------------------------------
    v27 = ExperimentMeta(
        experiment_name="shuffled_consistency",
        prompt=BASELINE_PROMPT,
        max_new_tokens=256,
        batch_size=8,
        num_shuffled_variants=5,
    )
    v28 = ExperimentMeta(
        experiment_name="audio_first_instruction",
        prompt=f"{_AUDIO_FIRST_WARNING}\n\n{BASELINE_PROMPT}",
        max_new_tokens=256,
        batch_size=8,
    )

    @classmethod
    def get_version(cls, version_str: str) -> "ExperimentVersion":
        try:
            version = cls[version_str.lower()]
        except KeyError as e:
            raise ValueError(
                f"Unknown experiment version: {version_str}. Available versions: {[e.name for e in cls]}"
            ) from e

        meta = version.value
        if meta.deprecated:
            raise DeprecatedExperimentError(
                f"{version.name} ({meta.experiment_name}) is deprecated: {meta.deprecation_reason}"
            )
        if meta.is_future_work:
            logger.warning(
                f"{version.name} ({meta.experiment_name}) is tier 'future_work' and sits outside the "
                f"reported set: {meta.future_work_reason}"
            )
        return version


def _load_rag_mapping(path: str) -> dict[str, list[str]]:
    if not os.path.exists(path):
        raise DatasetIntegrityError(f"RAG mapping file not found: {path}. Run scripts/build_audio_rag_index.py first.")
    with open(path) as f:
        return json.load(f)


def attach_few_shots(
    items: list[tuple[BaseRequest, str]], config: AppConfig, meta: ExperimentMeta
) -> list[tuple[BaseRequest, str]]:
    """Returns new requests carrying their few-shot turns. Never mutates the inputs."""
    if meta.few_shot_mode == "none":
        return items

    eval_ids = {req.metadata.item_id for req, _ in items if req.metadata}

    if meta.few_shot_mode == "rag":
        mapping = _load_rag_mapping(meta.rag_mapping_file)
        pool = get_mmar_few_shot_turns_pool(config.dataset, meta)
        validate_few_shot_disjointness(eval_ids, set(pool), mapping)

        attached = []
        for req, ground_truth in items:
            item_id = req.metadata.item_id if req.metadata else None
            turns = [pool[shot_id] for shot_id in mapping.get(item_id, []) if shot_id in pool]
            logger.info(f"Item {item_id}: attached {len(turns)} RAG shots.")
            attached.append((req.model_copy(update={"few_shot_turns": turns}), ground_truth))
        return attached

    shot_ids, turns = get_mmar_few_shot_turns(config.dataset, meta, num_shots=DEFAULT_NUM_FEW_SHOTS)
    validate_few_shot_disjointness(eval_ids, set(shot_ids))
    return [(req.model_copy(update={"few_shot_turns": turns}), gt) for req, gt in items]


def build_engine(config: AppConfig, meta: ExperimentMeta) -> GemmaTextModel | VoxtralAudioEngine:
    if meta.text_only:
        return GemmaTextModel(config.text_model)

    audio_config = config.audio_model
    scorer = (
        ContrastiveChoiceScorer(meta.contrastive_alpha, audio_config.target_sr, meta.batch_size)
        if meta.contrastive_alpha is not None
        else None
    )
    return VoxtralAudioEngine(
        audio_config,
        presentation=build_presentation(meta, audio_config.target_sr),
        scorer=scorer,
    )


def build_aggregator(meta: ExperimentMeta) -> ResponseAggregator:
    if meta.num_shuffled_variants > 1:
        return ShuffledMajorityVoteAggregator()
    return IdentityAggregator()


def run_mmar(args):
    meta = ExperimentVersion.get_version(args.experiment).value
    config = create_mmar_config(args, meta)

    num_runs = args.runs
    if meta.is_deterministic and num_runs > 1:
        logger.info(f"{meta.experiment_name} is deterministic; collapsing {num_runs} requested runs to 1.")
        num_runs = 1

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.output_dir, f"mmar_{meta.experiment_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    write_run_config(run_dir, args, config, meta)

    logger.info("--- [PHASE 1] DATA LOADING & PREPARATION ---")
    items = load_mmar_requests(config.dataset, meta, is_text_only=meta.text_only)
    items = attach_few_shots(items, config, meta)
    requests = [req for req, _ in items]
    ground_truths = [gt for _, gt in items]

    logger.info("--- [PHASE 2] INITIALIZING INFERENCE ENGINE ---")
    log_gpu_memory()
    inference_pipeline = InferencePipeline(build_engine(config, meta), build_aggregator(meta))

    logger.info(f"--- [PHASE 3] RUNNING INFERENCE ({num_runs} RUNS) ---")
    temp_files, final_outputs, metrics_outputs = [], [], []
    for i in range(num_runs):
        suffix = f"_run{i + 1}" if num_runs > 1 else ""
        temp_files.append(os.path.join(run_dir, f"temp_audio_preds{suffix}.jsonl"))
        final_outputs.append(os.path.join(run_dir, f"raw_output{suffix}.jsonl"))
        metrics_outputs.append(os.path.join(run_dir, f"metrics{suffix}.txt"))
        inference_pipeline.run(requests, ground_truths, temp_files[i])

    inference_pipeline = None
    release_vram()

    if getattr(args, "no_judge", False):
        # The parser is the primary MMAR metric; the judge only adds a secondary one. Skipping it is required
        # when the audio model is served from the same GPU (--audio-server-url) and the judge would not fit.
        logger.info("--- [PHASE 4] JUDGE SKIPPED: the parser alone scores the answers (judge metrics stay empty) ---")
        judge_pipeline = ParserOnlyPipeline(metrics_calculator=calculate_mmar_metrics)
    else:
        logger.info(f"Waiting {JUDGE_LOAD_DELAY_S} seconds before loading Judge...")
        time.sleep(JUDGE_LOAD_DELAY_S)

        logger.info("--- [PHASE 4] INITIALIZING JUDGE ENGINE ---")
        judge_pipeline = JudgePipeline(
            QwenJudge(
                config.judge,
                template_func=build_mmar_judge_conversation,
                schema_class=MMAREvaluationResult,
            ),
            metrics_calculator=calculate_mmar_metrics,
        )

    logger.info(f"--- [PHASE 5] RUNNING EVALUATION ({num_runs} RUNS) ---")
    all_metrics = [judge_pipeline.run(temp_files[i], final_outputs[i], metrics_outputs[i], i) for i in range(num_runs)]

    judge_pipeline = None
    release_vram()

    if num_runs > 1:
        logger.info("--- [PHASE 6] CALCULATING STABILITY ---")
        stability_file = os.path.join(run_dir, "stability_report.json")
        with open(stability_file, "w") as f:
            json.dump(calculate_stability(all_metrics), f, indent=4)
        logger.info(f"Saved statistical stability report to {stability_file}")


if __name__ == "__main__":
    run_mmar(parse_args("Run MMAR Speech Processing Pipeline"))
