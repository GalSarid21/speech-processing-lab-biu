from assertpy import assert_that

from speech_processing.data.dtos import EvaluationResult, JudgeRequest, JudgeResponse
from speech_processing.evaluation.metrics import calculate_metrics


def test_calculate_metrics():
    req = JudgeRequest(sample_id="test_id", instruction="", generated_text="", ground_truth="COPD")
    resp1 = JudgeResponse(
        request=req, evaluation=EvaluationResult(reasoning="pass", acoustic_accuracy=8, diagnostic_accuracy=10, hallucination_penalty=0, extracted_disease_class="COPD")
    )
    resp2 = JudgeResponse(
        request=req, evaluation=EvaluationResult(reasoning="fail", acoustic_accuracy=2, diagnostic_accuracy=0, hallucination_penalty=1, extracted_disease_class="Healthy")
    )

    metrics = calculate_metrics([resp1, resp2, resp1])
    assert metrics is not None
    assert_that(metrics.avg_acoustic_pct).is_close_to(60.0, 0.01)
    assert_that(metrics.avg_diagnostic_pct).is_close_to(66.66, 0.01)
    assert_that(metrics.hallucination_rate_pct).is_close_to(33.33, 0.01)


def test_calculate_metrics_empty():
    metrics = calculate_metrics([])
    assert_that(metrics).is_none()
