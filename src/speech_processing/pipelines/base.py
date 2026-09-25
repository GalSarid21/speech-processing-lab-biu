import gc
import torch
from loguru import logger

def release_vram():
    """Explicitly garbage collect and empty the CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    logger.info("Explicit VRAM clearance complete.")
