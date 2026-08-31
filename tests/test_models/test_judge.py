import json
import sys

import pytest
from assertpy import assert_that


# Mock vLLM import since it is not installable on MacOS
class DummyMock:
    def __getattr__(self, name):
        return DummyMock()
    def __call__(self, *args, **kwargs):
        return DummyMock()

sys.modules['vllm'] = DummyMock()

from speech_processing.config.core import JudgeConfig
from speech_processing.data.dtos import JudgeRequest
from speech_processing.models.judge import QwenJudge
from speech_processing.prompts.templates.judge.qwen_judge import (
    QwenICBHI2017JudgeTemplate,
)


@pytest.fixture
def mock_judge_config():
    return JudgeConfig(
        model_id="mock/judge",
        dtype="auto",
        max_new_tokens=100,
        max_num_seqs=10,
        max_model_len=1024,
    )


def test_judge_batch_evaluate_success(mocker, mock_judge_config):
    mocker.patch("speech_processing.models.judge.LLM")
    mock_vllm_adapter_class = mocker.patch("speech_processing.models.judge.VLLMAdapter")
    
    # Setup mock LLM and Adapter
    mock_adapter = mock_vllm_adapter_class.return_value
    
    # Simulate a perfect JSON response from vLLM
    valid_json = json.dumps({
        "reasoning": "Clear sounds.",
        "acoustic_accuracy": 9,
        "diagnostic_accuracy": 9,
        "hallucination_penalty": 0,
        "extracted_disease_class": "COPD"
    })
    mock_adapter.generate_batch.return_value = [valid_json]

    template = QwenICBHI2017JudgeTemplate()
    judge = QwenJudge(config=mock_judge_config, template=template)

    requests = [JudgeRequest(sample_id="test_id", instruction="prompt", generated_text="answer", ground_truth="COPD")]
    responses = judge.batch_evaluate(requests)

    assert_that(responses).is_length(1)
    
    eval_obj = responses[0].evaluation
    assert_that(eval_obj.acoustic_accuracy).is_equal_to(9)
    assert_that(eval_obj.extracted_disease_class).is_equal_to("COPD")
    
    # Ensure adapter was called correctly
    mock_adapter.generate_batch.assert_called_once()


def test_judge_batch_evaluate_json_fallback(mocker, mock_judge_config):
    mocker.patch("speech_processing.models.judge.LLM")
    mock_vllm_adapter_class = mocker.patch("speech_processing.models.judge.VLLMAdapter")
    mock_adapter = mock_vllm_adapter_class.return_value
    
    # Simulate garbage string from vLLM (e.g. if guided_json fails or model hallucinates text)
    mock_adapter.generate_batch.return_value = ["This is not JSON!"]

    template = QwenICBHI2017JudgeTemplate()
    judge = QwenJudge(config=mock_judge_config, template=template)

    requests = [JudgeRequest(sample_id="test_id", instruction="prompt", generated_text="answer", ground_truth="COPD")]
    responses = judge.batch_evaluate(requests)

    assert_that(responses).is_length(1)
    
    # It should seamlessly fallback without crashing!
    eval_obj = responses[0].evaluation
    assert_that(eval_obj.reasoning).is_equal_to("Parse failed.")
    assert_that(eval_obj.acoustic_accuracy).is_equal_to(0)
    assert_that(eval_obj.hallucination_penalty).is_equal_to(1)
    assert_that(eval_obj.extracted_disease_class).is_equal_to("Unknown")
