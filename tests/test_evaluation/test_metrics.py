import pytest
from assertpy import assert_that

from speech_processing.data.dtos import (
    AggregateMetrics,
    EvaluationResult,
    ItemMetadata,
    JudgeResponse,
    MMAREvaluationResult,
)
from speech_processing.evaluation.metrics import calculate_metrics, calculate_mmar_metrics
from speech_processing.evaluation.stability import calculate_stability

CHOICES = ["Dog", "Cat"]


def icbhi_response(acoustic: int, diagnostic: int, hallucination: int, predicted: str) -> JudgeResponse:
    return JudgeResponse(
        sample_id="test_id",
        instruction="",
        generated_text="",
        ground_truth="COPD",
        evaluation=EvaluationResult(
            reasoning="r",
            acoustic_accuracy=acoustic,
            diagnostic_accuracy=diagnostic,
            hallucination_penalty=hallucination,
            extracted_class=predicted,
        ),
    )


def mmar_response(
    generated: str,
    ground_truth: str,
    judged: str,
    category: str,
    final_turn_text: str | None = None,
) -> JudgeResponse:
    return JudgeResponse(
        sample_id="test_id",
        instruction="",
        generated_text=generated,
        final_turn_text=final_turn_text,
        ground_truth=ground_truth,
        evaluation=MMAREvaluationResult(reasoning="r", extracted_choice=judged),
        metadata=ItemMetadata(item_id="ID", question="q", choices=CHOICES, category=category),
    )


def test_icbhi_metrics():
    metrics = calculate_metrics(
        [
            icbhi_response(8, 10, 0, "COPD"),
            icbhi_response(2, 0, 1, "Healthy"),
            icbhi_response(8, 10, 0, "COPD"),
        ]
    )
    assert_that(metrics.avg_acoustic_pct).is_close_to(60.0, 0.01)
    assert_that(metrics.avg_diagnostic_pct).is_close_to(66.66, 0.01)
    assert_that(metrics.hallucination_rate_pct).is_close_to(33.33, 0.01)


@pytest.mark.parametrize("calculator", [calculate_metrics, calculate_mmar_metrics])
def test_calculators_return_none_on_empty(calculator):
    assert_that(calculator([])).is_none()


def test_mmar_parser_is_the_primary_metric():
    metrics = calculate_mmar_metrics(
        [
            mmar_response("B", "B", "B", "Speaker"),  # both right
            mmar_response("Dog", "A", "B", "Speaker"),  # parser right via choice text, judge wrong
            mmar_response("no idea", "A", "A", "Accent"),  # unparsed, judge right
            mmar_response("B", "A", "B", "Accent"),  # both wrong
        ]
    )

    assert_that(metrics.accuracy_pct).is_close_to(50.0, 0.01)
    assert_that(metrics.judge_accuracy_pct).is_close_to(50.0, 0.01)
    assert_that(metrics.unparsed_pct).is_close_to(25.0, 0.01)
    assert_that(metrics.parser_judge_agreement_pct).is_close_to(50.0, 0.01)
    assert_that(metrics.classification_report).contains("Speaker")
    assert_that(metrics.classification_report).contains("Accent")


def test_mmar_scores_only_the_final_turn():
    metrics = calculate_mmar_metrics(
        [mmar_response("Turn 1 - Assistant: A\n\nTurn 2 - Assistant: B", "B", "B", "Speaker", final_turn_text="B")]
    )
    assert_that(metrics.accuracy_pct).is_close_to(100.0, 0.01)


def test_stability_covers_every_numeric_field_and_keeps_legacy_keys():
    report = calculate_stability(
        [
            AggregateMetrics(
                avg_acoustic_pct=60.0,
                avg_diagnostic_pct=60.0,
                hallucination_rate_pct=10.0,
                classification_report="ignored",
            ),
            AggregateMetrics(
                avg_acoustic_pct=80.0,
                avg_diagnostic_pct=40.0,
                hallucination_rate_pct=30.0,
                classification_report="ignored",
            ),
        ]
    )

    assert_that(report).contains_key("acoustic_accuracy", "diagnostic_accuracy", "hallucination_rate")
    assert_that(report).does_not_contain_key("classification_report")
    assert_that(report["runs"]).is_equal_to(2)
    assert_that(report["acoustic_accuracy"]["mean"]).is_equal_to(70.0)


def test_stability_generalises_to_mmar_metrics():
    metrics = [
        calculate_mmar_metrics([mmar_response("B", "B", "B", "Speaker")]),
        calculate_mmar_metrics([mmar_response("A", "B", "A", "Speaker")]),
    ]
    report = calculate_stability(metrics)

    assert_that(report).contains_key("accuracy_pct", "judge_accuracy_pct", "unparsed_pct", "parser_judge_agreement_pct")
    assert_that(report["accuracy_pct"]["mean"]).is_equal_to(50.0)


def test_mmar_counts_answers_that_switched_language():
    metrics = calculate_mmar_metrics(
        [
            mmar_response("答案是 B 因为说话者是猫。", "B", "B", "Speaker"),
            mmar_response('He says "猫" (cat). B', "B", "B", "Speaker"),
        ]
    )

    assert_that(metrics.non_english_pct).is_equal_to(50.0)
