import json

import pytest
from assertpy import assert_that

from speech_processing.config.core import JudgeConfig
from speech_processing.data.dtos import (
    ChoiceExtractionResult,
    EvaluationResult,
    ItemMetadata,
    JudgeRequest,
    MMAREvaluationResult,
)
from speech_processing.models.judge import QwenJudge
from speech_processing.prompts.templates.judge.qwen import (
    build_icbhi_choice_judge_conversation,
    build_icbhi_judge_conversation,
    build_mmar_judge_conversation,
)
from speech_processing.utils.consts import UNPARSED_CHOICE

ICBHI_JSON = json.dumps(
    {
        "reasoning": "Clear sounds.",
        "acoustic_accuracy": 9,
        "diagnostic_accuracy": 9,
        "hallucination_penalty": 0,
        "extracted_class": "COPD",
    }
)
MMAR_JSON = json.dumps({"reasoning": "It picked B.", "extracted_choice": "B"})


@pytest.fixture
def judge_config():
    return JudgeConfig(
        model_id="mock/judge",
        dtype="auto",
        max_new_tokens=100,
        max_num_seqs=10,
        max_model_len=1024,
        gpu_memory_utilization=0.95,
    )


@pytest.fixture
def judge_request():
    return JudgeRequest(
        sample_id="test_id",
        instruction="prompt",
        generated_text="Turn 1 ... Turn 2 ... B",
        final_turn_text="B",
        ground_truth="B",
        metadata=ItemMetadata(item_id="ID1", question="q", choices=["Dog", "Cat"]),
    )


@pytest.mark.parametrize(
    "template, schema, raw_output, expected_field, expected_value",
    [
        (build_icbhi_judge_conversation, EvaluationResult, ICBHI_JSON, "extracted_class", "COPD"),
        (build_icbhi_judge_conversation, EvaluationResult, "not json", "extracted_class", UNPARSED_CHOICE),
        (build_mmar_judge_conversation, MMAREvaluationResult, MMAR_JSON, "extracted_choice", "B"),
        (build_mmar_judge_conversation, MMAREvaluationResult, "not json", "extracted_choice", UNPARSED_CHOICE),
    ],
)
def test_judge_parses_or_falls_back(
    mocker, judge_config, judge_request, template, schema, raw_output, expected_field, expected_value
):
    adapter = mocker.patch("speech_processing.models.judge.VLLMAdapter").return_value
    adapter.generate_batch.return_value = [raw_output]

    judge = QwenJudge(judge_config, template_func=template, schema_class=schema)
    responses = judge.batch_evaluate([judge_request])

    assert_that(responses).is_length(1)
    assert_that(responses[0].evaluation).is_instance_of(schema)
    assert_that(getattr(responses[0].evaluation, expected_field)).is_equal_to(expected_value)
    assert_that(responses[0].final_turn_text).is_equal_to("B")

    sent_schema = json.loads(adapter.generate_batch.call_args.kwargs["sampling_params"].json_schema)
    assert_that(sent_schema["properties"]).is_equal_to(schema.model_json_schema()["properties"])


@pytest.mark.parametrize("template", [build_mmar_judge_conversation, build_icbhi_choice_judge_conversation])
def test_judge_prompt_names_every_schema_field(template, judge_request):
    system_prompt = template(judge_request)[0]["content"]
    for field in ChoiceExtractionResult.model_fields:
        assert_that(system_prompt).contains(f'"{field}"')
