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


def log_gpu_memory() -> None:
    """Logs the device the run will use, before a multi-GB model spends a minute loading into it.

    A 24B model needs roughly 45 GiB of weights; on a smaller card vLLM only says so at the end of a
    200-line traceback, so the size of the card belongs in the log up front.
    """
    if not torch.cuda.is_available():
        logger.warning("No CUDA device visible. The vLLM engine will not start.")
        return

    properties = torch.cuda.get_device_properties(0)
    logger.info(f"GPU: {properties.name}, {properties.total_memory / 1024**3:.1f} GiB total")
