import json
import os

from loguru import logger

from speech_processing.config.core import (
    AppConfig,
    AudioModelConfig,
    DatasetConfig,
    ExperimentMeta,
    JudgeConfig,
    TextModelConfig,
)
from speech_processing.utils.consts import DEFAULT_SAMPLING_RATE

GEMMA_MODEL_ID = "google/gemma-4-26B-A4B-it"
VOXTRAL_MODEL_ID = "mistralai/Voxtral-Small-24B-2507"
# ~45 GiB of bf16 weights, so VOXTRAL_MODEL_ID needs an 80GB card. This 3B sibling fits anywhere and
# is the intended --audio-model-id for validating the pipeline on a smaller GPU. Its scores are NOT
# comparable to any run made with the 24B model.
VOXTRAL_MINI_MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
QWEN_AUDIO_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
JUDGE_MODEL_ID = "Qwen/Qwen3.8-27B-FP8"

DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_GPU_PCT = 0.95
DEFAULT_MAX_SEQS = 256
DEFAULT_TEXT_MAX_NEW_TOKENS = 512
DEFAULT_AUDIO_MAX_NEW_TOKENS = 256
JUDGE_MAX_NEW_TOKENS = 512
JUDGE_MAX_NUM_SEQS = 64

ICBHI_DATASET_ID = "DynamicSuperb/RespiratorySoundClassification_ICBHI2017"
ICBHI_SPLIT = "test"
ICBHI_AUDIO_BASE_DIR = "data/ICBHI"
ICBHI_RESPIRATORY_FEATURES_FILE = "data/icbhi_respiratory_features.jsonl"
ICBHI_AUDIO_TAGS_FILE = "data/icbhi_audio_tags.jsonl"
ICBHI_NEIGHBOR_LABELS_FILE = "data/icbhi_neighbor_labels.json"
ICBHI_RAG_MAPPING_FILE = "data/icbhi_rag_mapping.json"
ICBHI_DEMO_IDS_FILE = "data/icbhi_demo_ids.json"

MMAR_DATASET_ID = "BoJack/MMAR"
MMAR_SPLIT = "test"
MMAR_AUDIO_BASE_DIR = "data/MMAR"
MMAR_TRANSCRIPTS_FILE = "data/mmar_transcripts.json"
MMAR_FEW_SHOT_IDS_FILE = "data/mmar_en_speech_few_shot_ids.txt"
MMAR_ACOUSTIC_FEATURES_FILE = "data/mmar_acoustic_features.jsonl"

SAMPLE_ID_JSONL_KEY = "sample_id"
THINK_BLOCK_PREFIX = "<think>\n"


def _audio_model_config(args, experiment_meta: ExperimentMeta, max_seqs: int, gpu_pct: float) -> AudioModelConfig:
    """The audio model, with the command-line overrides for server-hosted and reasoning models."""
    return create_voxtral_config(
        max_num_seqs=max_seqs,
        max_new_tokens=getattr(args, "audio_max_new_tokens", None) or experiment_meta.max_new_tokens,
        gpu_pct=gpu_pct,
        stop=experiment_meta.stop,
        model_id=getattr(args, "audio_model_id", None) or VOXTRAL_MODEL_ID,
        server_url=getattr(args, "audio_server_url", None),
        reasoning_prefix=THINK_BLOCK_PREFIX if getattr(args, "audio_thinking", False) else None,
        stop_token_ids=getattr(args, "audio_stop_token_ids", None),
    )


def _parse_sample_ids(sample_ids_file: str | None) -> list[str]:
    if not sample_ids_file:
        logger.warning("No sample IDs file provided.")
        return []

    if not os.path.exists(sample_ids_file):
        logger.warning(f"Sample IDs file not found: {sample_ids_file}")
        return []

    logger.info(f"Loading reference sample IDs from {sample_ids_file}")
    sample_ids: list[str] = []
    try:
        with open(sample_ids_file) as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if sample_ids_file.endswith(".jsonl"):
                    sample_ids.append(json.loads(line)[SAMPLE_ID_JSONL_KEY])
                else:
                    sample_ids.append(line)
    except json.JSONDecodeError:
        logger.error(f"Sample IDs file is not valid JSON: {sample_ids_file}")
    except OSError as e:
        logger.error(f"An error occurred while loading sample IDs from {sample_ids_file}: {e}")

    return sample_ids


def create_gemma_config(
    max_num_seqs: int = DEFAULT_MAX_SEQS,
    max_new_tokens: int = DEFAULT_TEXT_MAX_NEW_TOKENS,
    gpu_pct: float = DEFAULT_GPU_PCT,
    stop: list[str] | None = None,
    model_id: str = GEMMA_MODEL_ID,
) -> TextModelConfig:
    return TextModelConfig(
        model_id=model_id,
        dtype="bfloat16",
        max_num_seqs=max_num_seqs,
        max_new_tokens=max_new_tokens,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization=gpu_pct,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        enable_thinking=True,
        stop=stop,
    )


def create_voxtral_config(
    max_num_seqs: int = DEFAULT_MAX_SEQS,
    max_new_tokens: int = DEFAULT_AUDIO_MAX_NEW_TOKENS,
    gpu_pct: float = DEFAULT_GPU_PCT,
    stop: list[str] | None = None,
    model_id: str = VOXTRAL_MODEL_ID,
    *,
    server_url: str | None = None,
    reasoning_prefix: str | None = None,
    stop_token_ids: list[int] | None = None,
) -> AudioModelConfig:
    return AudioModelConfig(
        model_id=model_id,
        server_url=server_url,
        reasoning_prefix=reasoning_prefix,
        stop_token_ids=stop_token_ids,
        dtype="bfloat16",
        max_num_seqs=max_num_seqs,
        max_new_tokens=max_new_tokens,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization=gpu_pct,
        temperature=0.5,
        top_p=0.5,
        target_sr=DEFAULT_SAMPLING_RATE,
        stop=stop,
    )


def create_qwen_audio_config(
    max_num_seqs: int = DEFAULT_MAX_SEQS,
    max_new_tokens: int = DEFAULT_AUDIO_MAX_NEW_TOKENS,
    gpu_pct: float = DEFAULT_GPU_PCT,
    stop: list[str] | None = None,
) -> AudioModelConfig:
    return AudioModelConfig(
        model_id=QWEN_AUDIO_MODEL_ID,
        dtype="bfloat16",
        max_num_seqs=max_num_seqs,
        max_new_tokens=max_new_tokens,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization=gpu_pct,
        temperature=0.5,
        top_p=0.5,
        target_sr=DEFAULT_SAMPLING_RATE,
        stop=stop,
    )


def create_judge_config() -> JudgeConfig:
    return JudgeConfig(
        model_id=JUDGE_MODEL_ID,
        dtype="auto",
        max_num_seqs=JUDGE_MAX_NUM_SEQS,
        max_new_tokens=JUDGE_MAX_NEW_TOKENS,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization=DEFAULT_GPU_PCT,
    )


def create_icbhi_config(args, experiment_meta: ExperimentMeta) -> AppConfig:
    """Factory for creating a fully initialized ICBHI configuration."""
    gpu_pct = getattr(args, "gpu_pct", DEFAULT_GPU_PCT)
    max_seqs = getattr(args, "max_seqs", DEFAULT_MAX_SEQS)
    text_model_id = getattr(args, "text_model_id", None) or GEMMA_MODEL_ID

    dataset_config = DatasetConfig(
        dataset_id=ICBHI_DATASET_ID,
        # The label set is closed and lives in data/icbhi.py, so no target-label filter is needed.
        target_labels=[],
        split=ICBHI_SPLIT,
        num_samples=args.num_samples,
        sample_ids=_parse_sample_ids(args.sample_ids_file),
        audio_base_dir=ICBHI_AUDIO_BASE_DIR,
        respiratory_features_file=ICBHI_RESPIRATORY_FEATURES_FILE,
        audio_tags_file=ICBHI_AUDIO_TAGS_FILE,
        neighbor_labels_file=ICBHI_NEIGHBOR_LABELS_FILE,
        demo_ids_file=ICBHI_DEMO_IDS_FILE,
    )

    text_model = None
    audio_model = None
    if experiment_meta.text_only:
        text_model = create_gemma_config(
            max_num_seqs=max_seqs,
            max_new_tokens=experiment_meta.max_new_tokens,
            gpu_pct=gpu_pct,
            stop=experiment_meta.stop,
            model_id=text_model_id,
        )
    else:
        audio_model = _audio_model_config(args, experiment_meta, max_seqs, gpu_pct)

    return AppConfig(
        output_dir=args.output_dir,
        dataset=dataset_config,
        judge=create_judge_config(),
        text_model=text_model,
        audio_model=audio_model,
    )


def create_mmar_config(args, experiment_meta: ExperimentMeta) -> AppConfig:
    """Factory for creating a fully initialized MMAR configuration."""
    gpu_pct = getattr(args, "gpu_pct", DEFAULT_GPU_PCT)
    max_seqs = getattr(args, "max_seqs", DEFAULT_MAX_SEQS)
    text_model_id = getattr(args, "text_model_id", None) or GEMMA_MODEL_ID

    dataset_config = DatasetConfig(
        dataset_id=MMAR_DATASET_ID,
        target_labels=[],
        split=MMAR_SPLIT,
        num_samples=args.num_samples,
        sample_ids=_parse_sample_ids(args.sample_ids_file),
        audio_base_dir=MMAR_AUDIO_BASE_DIR,
        transcripts_file=MMAR_TRANSCRIPTS_FILE,
        few_shot_ids_file=MMAR_FEW_SHOT_IDS_FILE,
        acoustic_features_file=MMAR_ACOUSTIC_FEATURES_FILE,
    )

    text_model = None
    audio_model = None
    if experiment_meta.text_only:
        text_model = create_gemma_config(
            max_num_seqs=max_seqs,
            max_new_tokens=experiment_meta.max_new_tokens,
            gpu_pct=gpu_pct,
            stop=experiment_meta.stop,
            model_id=text_model_id,
        )
    else:
        audio_model = _audio_model_config(args, experiment_meta, max_seqs, gpu_pct)

    return AppConfig(
        output_dir=args.output_dir,
        dataset=dataset_config,
        judge=create_judge_config(),
        text_model=text_model,
        audio_model=audio_model,
    )
