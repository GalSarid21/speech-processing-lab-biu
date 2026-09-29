import pytest
from assertpy import assert_that

from speech_processing.data.dtos import JudgeRequest
from speech_processing.prompts.templates.judge.qwen import (
    build_icbhi_choice_judge_conversation,
    build_icbhi_judge_conversation,
    build_mmar_judge_conversation,
)
from speech_processing.runners.icbhi import ExperimentVersion as IcbhiExperimentVersion
from speech_processing.runners.mmar import ExperimentVersion as MmarExperimentVersion

GROUND_TRUTH = "ZZGROUNDTRUTHZZ"


@pytest.fixture
def judge_request():
    return JudgeRequest(
        sample_id="test",
        instruction="Question: what?\nChoices:\nA. Dog\nB. Cat",
        generated_text="long reasoning ... finally B",
        final_turn_text="B",
        ground_truth=GROUND_TRUTH,
    )


def test_icbhi_system_prompt_keeps_the_task_newline(judge_request):
    system_prompt = build_icbhi_judge_conversation(judge_request)[0]["content"]
    assert_that(system_prompt).contains("### Task\n")


def test_icbhi_choice_judge_extracts_without_the_ground_truth(judge_request):
    conversation = build_icbhi_choice_judge_conversation(judge_request)
    user_content = conversation[1]["content"]

    assert_that(conversation[0]["content"]).contains("Only extract it.")
    assert_that(user_content).does_not_contain(GROUND_TRUTH)
    assert_that(user_content).contains("--- MODEL ANSWER ---\nB")


def test_mmar_user_content_hides_the_ground_truth(judge_request):
    conversation = build_mmar_judge_conversation(judge_request)
    user_content = conversation[1]["content"]

    assert_that(conversation).is_length(2)
    assert_that(user_content).does_not_contain(GROUND_TRUTH)
    assert_that(user_content).contains("--- MODEL ANSWER ---\nB")


def test_mmar_falls_back_to_generated_text_without_a_final_turn(judge_request):
    without_final = judge_request.model_copy(update={"final_turn_text": None})
    user_content = build_mmar_judge_conversation(without_final)[1]["content"]
    assert_that(user_content).contains("--- MODEL ANSWER ---\nlong reasoning ... finally B")


@pytest.mark.parametrize(
    "registry, key, expected_name",
    [
        (IcbhiExperimentVersion, "r0", "baseline"),
        (IcbhiExperimentVersion, "a1a", "audio_features"),
        (MmarExperimentVersion, "v3", "transcript_augmented"),
        (MmarExperimentVersion, "v13", "text_only_llm"),
    ],
)
def test_experiment_enum_routing(registry, key, expected_name):
    assert_that(registry.get_version(key).value.experiment_name).is_equal_to(expected_name)
