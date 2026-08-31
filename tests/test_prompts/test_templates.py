import pytest
from assertpy import assert_that

from speech_processing.data.dtos import JudgeRequest
from speech_processing.prompts.templates.judge.qwen_judge import (
    QwenICBHI2017JudgeTemplate,
)


@pytest.mark.parametrize(
    "instruction, generated, ground_truth, expected_in_prompt",
    [
        ("Find pathology", "Crackle", "Crackle", "Crackle"),
        ("Detect sounds", "Wheeze", "Wheeze", "Wheeze"),
    ],
)
def test_icbhi_template(instruction, generated, ground_truth, expected_in_prompt):
    template = QwenICBHI2017JudgeTemplate()
    req = JudgeRequest(
        instruction=instruction, generated_text=generated, ground_truth=ground_truth
    )
    conversation = template.build_conversation(req)

    assert_that(conversation).is_length(2)
    assert_that(conversation[1]["content"]).contains(expected_in_prompt)
