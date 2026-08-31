from assertpy import assert_that

from speech_processing.data.dtos import EvaluationResult, JudgeRequest, JudgeResponse
from speech_processing.evaluation.metrics import calculate_accuracy


def test_calculate_accuracy():
    req = JudgeRequest(instruction="", generated_text="", ground_truth="")
    eval_pass = JudgeResponse(
        request=req, evaluation=EvaluationResult(reasoning="pass", rate=10.0)
    )
    eval_fail = JudgeResponse(
        request=req, evaluation=EvaluationResult(reasoning="fail", rate=0.0)
    )

    acc = calculate_accuracy([eval_pass, eval_fail, eval_pass])
    assert_that(acc).is_close_to(66.66, 0.01)
