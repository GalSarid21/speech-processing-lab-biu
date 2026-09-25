import pytest
from assertpy import assert_that

from speech_processing.data.dtos import JudgeRequest
from speech_processing.prompts.templates.judge.qwen import (
    build_icbhi_judge_conversation,
    build_mmar_judge_conversation
)


@pytest.mark.parametrize(
    "instruction, generated, ground_truth, expected_in_prompt",
    [
        ("Find pathology", "Crackle", "Crackle", "Crackle"),
        ("Detect sounds", "Wheeze", "Wheeze", "Wheeze"),
    ],
)
def test_icbhi_template(instruction, generated, ground_truth, expected_in_prompt):
    req = JudgeRequest(
        sample_id="test", instruction=instruction, generated_text=generated, ground_truth=ground_truth
    )
    conversation = build_icbhi_judge_conversation(req)

    assert_that(conversation).is_length(2)
    assert_that(conversation[1]["content"]).contains(expected_in_prompt)

def test_mmar_judge_template():
    req = JudgeRequest(
        sample_id="test", instruction="What sound?", generated_text="Dog", ground_truth="Dog"
    )
    conversation = build_mmar_judge_conversation(req)
    
    # Assert exact required schema keys are requested
    system_prompt = conversation[0]["content"]
    assert_that(system_prompt).contains('"reasoning":')
    assert_that(system_prompt).contains('"acoustic_accuracy":')
    assert_that(system_prompt).contains('"diagnostic_accuracy":')
    assert_that(system_prompt).contains('"hallucination_penalty":')
    assert_that(system_prompt).contains('"extracted_class":')
    
    # Assert it asks to extract the multiple choice string, not disease name
    assert_that(system_prompt).contains("Extract the exact multiple-choice answer")
    assert_that(system_prompt).does_not_contain("COPD")

def test_experiment_version_enums():
    # Test dual enum routing logic
    from speech_processing.runners.icbhi import ExperimentVersion as ICBHIEnum
    from speech_processing.runners.mmar import ExperimentVersion as MMAREnum
    
    # V3 in ICBHI is typically symptomatic
    icbhi_v3 = ICBHIEnum.get_version("v3").value
    assert_that(icbhi_v3.experiment_name).contains("few_shot")
    
    # V3 in MMAR is transcript augmented
    mmar_v3 = MMAREnum.get_version("v3").value
    assert_that(mmar_v3.experiment_name).is_equal_to("transcript_augmented")
