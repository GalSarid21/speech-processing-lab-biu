import string

import torch

DEFAULT_DTYPE = torch.float32
MODEL_DEVICE_DTYPE_MAPPING = {
    "cuda": torch.bfloat16,
    "mps": torch.float16,
    "cpu": DEFAULT_DTYPE,
}

DEFAULT_SAMPLING_RATE = 16_000

# CoT Parsing Tags
COT_START_TAG = "<analysis>"
COT_END_TAG = "</analysis>"
ANSWER_START_TAG = "<answer>"
ANSWER_END_TAG = "</answer>"

# Multiple-choice rendering and parsing
CHOICE_LETTERS = string.ascii_uppercase
NO_TRANSCRIPT_PLACEHOLDER = "[NO TRANSCRIPT]"
NO_FEATURES_PLACEHOLDER = "[FEATURES UNAVAILABLE]"
UNPARSED_CHOICE = "Unknown"

# Audio staging
MONO_AUDIO_CACHE_DIR = "/tmp/audio_mono_cache"
ICBHI_AUDIO_CACHE_DIR = "/tmp/icbhi_audio_cache"
TEMP_AUDIO_DIR_PREFIX = "speech_processing_audio_"

# Voxtral prompt budget
VOXTRAL_MAX_AUDIOS_PER_PROMPT = 8
VOXTRAL_AUDIO_TOKENS_PER_SECOND = 12.5  # Whisper-style encoder at 50 Hz, downsampled 4x

# T2 presentation
CHUNK_LENGTH_S = 5.0
MAX_AUDIO_CHUNKS = 6
MAX_BREATHING_CYCLES = 6
LOCALIZATION_PADDING_S = 0.5

# T3 contrastive scoring
CONTRASTIVE_PROMPT_LOGPROBS = 1
CONTENT_FREE_SEED = 7

# T4 shuffled variants
SHUFFLE_BASE_SEED = 1234

# Few-shot / RAG
DEFAULT_NUM_FEW_SHOTS = 3
RAG_STOP_SEQUENCES = ["\n\nQuestion:", "\nQuestion:"]

# Runner orchestration
JUDGE_LOAD_DELAY_S = 30
