"""The GPU preflight. Both failures it catches are cheap up front and opaque later."""

import pytest
from assertpy import assert_that
from loguru import logger

from speech_processing.pipelines.base import log_gpu_memory

BLACKWELL = (12, 0)
AMPERE = (8, 0)
GIB = 1024**3


@pytest.fixture
def captured():
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO")
    yield messages
    logger.remove(sink_id)


def patch_cuda(mocker, name, total_gib, capability, arch_list, available=True):
    torch = mocker.patch("speech_processing.pipelines.base.torch")
    torch.cuda.is_available.return_value = available
    torch.cuda.get_device_properties.return_value = mocker.MagicMock(name=name, total_memory=int(total_gib * GIB))
    torch.cuda.get_device_properties.return_value.name = name
    torch.cuda.get_device_capability.return_value = capability
    torch.cuda.get_arch_list.return_value = arch_list
    torch.__version__ = "2.9.0"
    torch.version.cuda = "12.8"
    return torch


def test_logs_the_card_and_its_architecture(mocker, captured):
    patch_cuda(mocker, "RTX PRO 6000 Blackwell", 95.6, BLACKWELL, ["sm_90", "sm_100", "sm_120"])

    log_gpu_memory()

    joined = "".join(captured)
    assert_that(joined).contains("RTX PRO 6000 Blackwell", "95.6 GiB", "sm_120", "CUDA 12.8")
    assert_that(joined).does_not_contain("no kernels")


def test_errors_when_torch_has_no_kernels_for_the_card(mocker, captured):
    """A Blackwell card with an Ampere-era torch: the exact trap this check exists for."""
    patch_cuda(mocker, "RTX PRO 6000 Blackwell", 95.6, BLACKWELL, ["sm_80", "sm_86", "sm_90"])

    log_gpu_memory()

    assert_that("".join(captured)).contains("has no kernels for sm_120")


def test_quiet_when_the_architecture_is_supported(mocker, captured):
    patch_cuda(mocker, "A100-SXM4-80GB", 79.2, AMPERE, ["sm_80", "sm_90"])

    log_gpu_memory()

    assert_that("".join(captured)).does_not_contain("no kernels")


def test_warns_without_a_cuda_device(mocker, captured):
    patch_cuda(mocker, "", 0, AMPERE, [], available=False)

    log_gpu_memory()

    assert_that("".join(captured)).contains("No CUDA device visible")
