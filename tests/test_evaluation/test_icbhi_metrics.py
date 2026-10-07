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


ASTHMA = "Asthma"
LRTI = "LRTI"


def test_rare_classes_are_answered_but_excluded_from_the_primary_average():
    """Two COPD items right, one Asthma item wrong. Asthma has 1 item in the real data, so letting it
    carry 1/K of the macro average would let a single prediction move the primary metric 12.5 pt."""
    evaluations = [response(COPD, letter_of(COPD)), response(COPD, letter_of(COPD)), response(ASTHMA, letter_of(COPD))]

    metrics = calculate_icbhi_metrics(evaluations)

    assert_that(metrics.balanced_accuracy_pct).is_close_to(100.0, 0.01)  # COPD only
    assert_that(metrics.balanced_accuracy_all_classes_pct).is_close_to(50.0, 0.01)  # COPD + Asthma
    assert_that(metrics.accuracy_pct).is_close_to(66.67, 0.01)  # every item still counts here
    assert_that(metrics.chance_balanced_acc_pct).is_close_to(100.0, 0.01)  # one primary class present


def test_always_copd_scores_chance_on_the_primary_metric():
    """The answer to "guessing COPD pays off": on balanced accuracy it pays exactly chance."""
    evaluations = [response(label, letter_of(COPD)) for label in (COPD, COPD, COPD, HEALTHY, PNEUMONIA, LRTI)]

    metrics = calculate_icbhi_metrics(evaluations)

    assert_that(metrics.accuracy_pct).is_close_to(50.0, 0.01)
    assert_that(metrics.balanced_accuracy_pct).is_close_to(metrics.chance_balanced_acc_pct, 0.01)


def test_macro_f1_ignores_classes_that_never_occur():
    """A perfect run over three classes used to report macro-F1 37.5, because the five absent
    labels each contributed an F1 of zero."""
    metrics = calculate_icbhi_metrics([response(label, letter_of(label)) for label in (COPD, HEALTHY, PNEUMONIA)])
    assert_that(metrics.macro_f1_pct).is_close_to(100.0, 0.01)


def test_icbhi_counts_answers_that_switched_language():
    metrics = calculate_icbhi_metrics([response(COPD, "诊断结果是 A 慢性阻塞性肺病"), response(COPD, letter_of(COPD))])

    assert_that(metrics.non_english_pct).is_equal_to(50.0)
