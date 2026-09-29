import pytest
from assertpy import assert_that

from speech_processing.data.choices import letter_for_index
from speech_processing.data.dtos import ChoiceExtractionResult, ItemMetadata, JudgeResponse
from speech_processing.data.icbhi import ICBHI_LABELS
from speech_processing.evaluation.metrics import calculate_icbhi_metrics
from speech_processing.evaluation.stability import calculate_stability

COPD = "COPD"
HEALTHY = "No potential disease detected"
PNEUMONIA = "Pneumonia"


def letter_of(label: str) -> str:
    return letter_for_index(ICBHI_LABELS.index(label))


def response(true_label: str, answer: str) -> JudgeResponse:
    return JudgeResponse(
        sample_id="a.wav",
        instruction="Choose.",
        generated_text=answer,
        final_turn_text=answer,
        ground_truth=letter_of(true_label),
        evaluation=ChoiceExtractionResult.fallback(),
        metadata=ItemMetadata(item_id="ID", question="q", choices=list(ICBHI_LABELS), category=true_label),
    )


def test_collapsing_to_the_majority_class_scores_at_chance():
    """4 COPD + 2 healthy, always answering COPD: 66.7% accuracy but 50% balanced accuracy."""
    evaluations = [response(COPD, letter_of(COPD)) for _ in range(4)]
    evaluations += [response(HEALTHY, letter_of(COPD)) for _ in range(2)]

    metrics = calculate_icbhi_metrics(evaluations)

    assert_that(metrics.accuracy_pct).is_close_to(66.67, 0.01)
    assert_that(metrics.majority_baseline_pct).is_close_to(66.67, 0.01)
    assert_that(metrics.balanced_accuracy_pct).is_close_to(50.0, 0.01)
    assert_that(metrics.chance_balanced_acc_pct).is_close_to(50.0, 0.01)
    assert_that(metrics.collapse_index_pct).is_close_to(100.0, 0.01)
    assert_that(metrics.copd_vs_rest_balanced_acc_pct).is_close_to(50.0, 0.01)


def test_perfect_predictions():
    metrics = calculate_icbhi_metrics(
        [
            response(COPD, letter_of(COPD)),
            response(HEALTHY, letter_of(HEALTHY)),
            response(PNEUMONIA, letter_of(PNEUMONIA)),
        ]
    )

    assert_that(metrics.balanced_accuracy_pct).is_close_to(100.0, 0.01)
    assert_that(metrics.macro_f1_pct).is_greater_than(0.0)
    assert_that(metrics.unparsed_pct).is_equal_to(0.0)
    assert_that(metrics.binary_disease_balanced_acc_pct).is_close_to(100.0, 0.01)


def test_off_list_answers_count_as_unparsed():
    metrics = calculate_icbhi_metrics([response(COPD, "Sleep Apnea"), response(HEALTHY, letter_of(HEALTHY))])

    assert_that(metrics.unparsed_pct).is_close_to(50.0, 0.01)
    assert_that(metrics.accuracy_pct).is_close_to(50.0, 0.01)


def test_choice_text_answers_are_parsed():
    metrics = calculate_icbhi_metrics([response(COPD, COPD)])
    assert_that(metrics.accuracy_pct).is_close_to(100.0, 0.01)


def test_empty_returns_none():
    assert_that(calculate_icbhi_metrics([])).is_none()


@pytest.mark.parametrize("field", ["balanced_accuracy_pct", "accuracy_pct", "collapse_index_pct", "unparsed_pct"])
def test_stability_covers_the_icbhi_fields(field):
    metrics = [
        calculate_icbhi_metrics([response(COPD, letter_of(COPD))]),
        calculate_icbhi_metrics([response(COPD, letter_of(HEALTHY))]),
    ]
    report = calculate_stability(metrics)

    assert_that(report).contains_key(field)
    assert_that(report).does_not_contain_key("classification_report")
