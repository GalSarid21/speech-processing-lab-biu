import pytest


@pytest.fixture
def sample_judge_request():
    from speech_processing.data.dtos import JudgeRequest

    return JudgeRequest(
        instruction="Detect pathology",
        generated_text="I detect fine crackles.",
        ground_truth="Fine crackles",
    )
