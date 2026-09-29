import sys

import pytest
from assertpy import assert_that

from speech_processing.config.core import TextModelConfig
from speech_processing.data.dtos import ItemMetadata, TextRequest
from speech_processing.models.text import BATCH_SAMPLE_ID, GemmaTextModel

STOP = ["\nQuestion:"]


@pytest.fixture
def text_config():
    return TextModelConfig(
        model_id="mock/gemma",
        dtype="bfloat16",
        max_num_seqs=4,
        max_new_tokens=128,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        enable_thinking=True,
        stop=STOP,
    )


@pytest.fixture
def patched_llm(mocker):
    vllm = sys.modules["vllm"]
    llm_class = mocker.patch.object(vllm, "LLM")
    sampling_params = mocker.patch.object(vllm, "SamplingParams")
    llm = llm_class.return_value
    llm.get_tokenizer.return_value.apply_chat_template.return_value = "PROMPT"
    output = mocker.MagicMock()
    output.outputs = [mocker.MagicMock(text="<think>hmm A</think>Final answer: B")]
    llm.generate.return_value = [output]
    return llm, sampling_params


def test_thinking_is_stripped_and_metadata_drives_the_sample_id(text_config, patched_llm):
    _, sampling_params = patched_llm
    request = TextRequest(
        instruction="Answer.",
        metadata=ItemMetadata(item_id="ID1", question="q", choices=["a", "b"]),
    )

    responses = GemmaTextModel(text_config).batch_infer([request])

    assert_that(responses[0].generated_text).is_equal_to("Final answer: B")
    assert_that(responses[0].final_turn_text).is_equal_to("Final answer: B")
    assert_that(responses[0].sample_id).is_equal_to("ID1")
    assert_that(sampling_params.call_args.kwargs["stop"]).is_equal_to(STOP)


def test_requests_without_metadata_fall_back_to_the_batch_id(text_config, patched_llm):
    responses = GemmaTextModel(text_config).batch_infer([TextRequest(instruction="Answer.")])
    assert_that(responses[0].sample_id).is_equal_to(BATCH_SAMPLE_ID)
