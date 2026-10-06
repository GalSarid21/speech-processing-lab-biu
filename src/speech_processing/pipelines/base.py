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
    """Logs the device, and whether the installed torch was actually built for it.

    Two things stop a run before it starts, and both are cheap to check up front:
    a card too small for the weights, and a card newer than the CUDA kernels torch shipped with.
    The second one otherwise surfaces much later as "no kernel image is available for execution".
    """
    if not torch.cuda.is_available():
        logger.warning("No CUDA device visible. The vLLM engine will not start.")
        return

    properties = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    architecture = f"sm_{major}{minor}"

    logger.info(
        f"GPU: {properties.name}, {properties.total_memory / 1024**3:.1f} GiB, {architecture}, "
        f"torch {torch.__version__} (CUDA {torch.version.cuda})"
    )

    supported = torch.cuda.get_arch_list()
    if supported and architecture not in supported:
        logger.error(
            f"This torch build has no kernels for {architecture} (it supports {supported}). "
            "Install a build matching the card before running anything."
        )
