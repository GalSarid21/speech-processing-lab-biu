from sklearn.metrics import classification_report

from speech_processing.data.dtos import (
    AggregateMetrics,
    EvaluationResult,
    JudgeResponse,
)

_PCT: float = 100.0
_DEFAULT_MAX_RATE: float = 10.0


def _get_max_rate(field_name: str) -> float:
    """Dynamically extracts the maximum valid rate from the Pydantic schema."""
    for meta in EvaluationResult.model_fields[field_name].metadata:
        if hasattr(meta, "le"):
            return float(meta.le)
    return _DEFAULT_MAX_RATE


def calculate_metrics(evaluations: list[JudgeResponse]) -> AggregateMetrics | None:
    if not evaluations:
        return None

    max_acoustic = _get_max_rate("acoustic_accuracy")
    max_diagnostic = _get_max_rate("diagnostic_accuracy")

    total_acoustic = sum(eval.evaluation.acoustic_accuracy for eval in evaluations)
    total_diagnostic = sum(eval.evaluation.diagnostic_accuracy for eval in evaluations)
    total_hallucination = sum(eval.evaluation.hallucination_penalty for eval in evaluations)

    num_evals = len(evaluations)

    avg_acoustic = (total_acoustic / num_evals / max_acoustic) * _PCT
    avg_diagnostic = (total_diagnostic / num_evals / max_diagnostic) * _PCT
    avg_hallucination = (total_hallucination / num_evals) * _PCT

    # Classification Metrics
    y_true = [eval.request.ground_truth for eval in evaluations]
    y_pred = [eval.evaluation.extracted_disease_class for eval in evaluations]

    # zero_division=0 to prevent warnings if a class was completely missed
    clf_report = classification_report(y_true, y_pred, zero_division=0)

    return AggregateMetrics(
        avg_acoustic_pct=avg_acoustic,
        avg_diagnostic_pct=avg_diagnostic,
        hallucination_rate_pct=avg_hallucination,
        classification_report=clf_report
    )
