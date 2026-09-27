import torch

DEFAULT_DTYPE = torch.float32
MODEL_DEVICE_DTYPE_MAPPING = {
    "cuda": torch.bfloat16,
    "mps": torch.float16,
    "cpu": DEFAULT_DTYPE
}

DEFAULT_SAMPLING_RATE = 16_000

# CoT Parsing Tags
COT_START_TAG = "<analysis>"
COT_END_TAG = "</analysis>"
ANSWER_START_TAG = "<answer>"
ANSWER_END_TAG = "</answer>"
