from collections import Counter, defaultdict
from collections.abc import Callable

import numpy as np
from sklearn.metrics import classification_report, f1_score

from speech_processing.data.choices import index_for_letter, letters_for
from speech_processing.data.dtos.metrics import AggregateMetrics, ICBHIAggregateMetrics, MMARAggregateMetrics
from speech_processing.data.dtos.responses import EvaluationResult, JudgeResponse
from speech_processing.data.icbhi import HEALTHY_LABEL, ICBHI_LABELS, ICBHI_PRIMARY_LABELS
from speech_processing.evaluation.answer_parsing import parse_choice_letter
from speech_processing.utils.consts import UNPARSED_CHOICE

MetricsCalculator = Callable[
    [list[JudgeResponse]], AggregateMetrics | MMARAggregateMetrics | ICBHIAggregateMetrics | None
]

COPD_LABEL = "COPD"

_PCT: float = 100.0
_DEFAULT_MAX_RATE: float = 10.0
_UNKNOWN_CATEGORY = "Unknown"


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

    num_evals = len(evaluations)
    total_acoustic = sum(e.evaluation.acoustic_accuracy for e in evaluations)
    total_diagnostic = sum(e.evaluation.diagnostic_accuracy for e in evaluations)
    total_hallucination = sum(e.evaluation.hallucination_penalty for e in evaluations)

    y_true = [e.ground_truth for e in evaluations]
    y_pred = [e.evaluation.extracted_class for e in evaluations]

    return AggregateMetrics(
        avg_acoustic_pct=(total_acoustic / num_evals / max_acoustic) * _PCT,
        avg_diagnostic_pct=(total_diagnostic / num_evals / max_diagnostic) * _PCT,
        hallucination_rate_pct=(total_hallucination / num_evals) * _PCT,
        # zero_division=0 to prevent warnings if a class was completely missed
        classification_report=classification_report(y_true, y_pred, zero_division=0),
    )


def parsed_letter(response: JudgeResponse) -> str | None:
    """The deterministic, primary MMAR prediction for one judged response."""
    if response.metadata is None:
        return None
    return parse_choice_letter(response.final_turn_text or response.generated_text, response.metadata.choices)


def _category_report(rows: list[tuple[str, bool]]) -> str:
    by_category: dict[str, list[bool]] = defaultdict(list)
    for category, correct in rows:
        by_category[category].append(correct)

    lines = ["Category | Acc % | n", "-------- | ----- | -"]
    for category in sorted(by_category):
        outcomes = by_category[category]
        accuracy = sum(outcomes) / len(outcomes) * _PCT
        lines.append(f"{category} | {accuracy:.2f} | {len(outcomes)}")
    return "\n".join(lines)


def calculate_mmar_metrics(evaluations: list[JudgeResponse]) -> MMARAggregateMetrics | None:
    if not evaluations:
        return None

    num_evals = len(evaluations)
    num_parser_correct = 0
    num_judge_correct = 0
    num_unparsed = 0
    num_agree = 0
    category_rows: list[tuple[str, bool]] = []

    for response in evaluations:
        parsed = parsed_letter(response)
        judged = response.evaluation.extracted_choice.strip()

        parser_correct = parsed == response.ground_truth
        num_parser_correct += parser_correct
        num_judge_correct += judged == response.ground_truth
        num_unparsed += parsed is None
        num_agree += (parsed or UNPARSED_CHOICE) == judged

        category = response.metadata.category if response.metadata else _UNKNOWN_CATEGORY
        category_rows.append((category or _UNKNOWN_CATEGORY, parser_correct))

    return MMARAggregateMetrics(
        accuracy_pct=num_parser_correct / num_evals * _PCT,
        judge_accuracy_pct=num_judge_correct / num_evals * _PCT,
        unparsed_pct=num_unparsed / num_evals * _PCT,
        parser_judge_agreement_pct=num_agree / num_evals * _PCT,
        classification_report=_category_report(category_rows),
    )


def _balanced_accuracy_pct(y_true: list[str], y_pred: list[str]) -> float:
    """Macro recall over the classes actually present in the ground truth."""
    recalls = []
    for label in sorted(set(y_true)):
        present = [pred for true, pred in zip(y_true, y_pred, strict=True) if true == label]
        recalls.append(sum(pred == label for pred in present) / len(present))
    return float(np.mean(recalls)) * _PCT if recalls else 0.0


def _binary_balanced_accuracy_pct(y_true: list[str], y_pred: list[str], positive: Callable[[str], bool]) -> float:
    binary_true = ["pos" if positive(label) else "neg" for label in y_true]
    binary_pred = ["pos" if positive(label) else "neg" for label in y_pred]
    return _balanced_accuracy_pct(binary_true, binary_pred)


def _icbhi_prediction(response: JudgeResponse) -> str:
    """The predicted label, or UNPARSED_CHOICE when no letter could be read."""
    letter = parsed_letter(response)
    if letter is None or letter not in letters_for(list(ICBHI_LABELS)):
        return UNPARSED_CHOICE
    return ICBHI_LABELS[index_for_letter(letter)]


def calculate_icbhi_metrics(evaluations: list[JudgeResponse]) -> ICBHIAggregateMetrics | None:
    """Closed-set ICBHI scoring.

    Balanced accuracy is primary, averaged over ICBHI_PRIMARY_LABELS: every supported class counts
    equally, so always answering COPD scores exactly chance however large the COPD share is. Asthma
    and LRTI items are still answered and still count towards accuracy, but not towards the primary
    average, where one or two items would swing it by 6-12 points. Every report carries its own
    baselines. Acoustic accuracy and hallucination rate are deliberately absent: this packaging has
    no cycle-level ground truth, so they would measure the judge's sense of plausibility, not audio.
    """
    if not evaluations:
        return None

    y_true = [ICBHI_LABELS[index_for_letter(e.ground_truth)] for e in evaluations]
    y_pred = [_icbhi_prediction(e) for e in evaluations]
    num_evals = len(evaluations)

    primary = [(t, p) for t, p in zip(y_true, y_pred, strict=True) if t in ICBHI_PRIMARY_LABELS]
    primary_true = [t for t, _ in primary]
    primary_pred = [p for _, p in primary]
    primary_present = sorted(set(primary_true), key=ICBHI_LABELS.index)

    majority_count = Counter(y_true).most_common(1)[0][1]
    prediction_counts = Counter(y_pred)

    return ICBHIAggregateMetrics(
        balanced_accuracy_pct=_balanced_accuracy_pct(primary_true, primary_pred),
        balanced_accuracy_all_classes_pct=_balanced_accuracy_pct(y_true, y_pred),
        # Over the primary classes that occur, so an absent class cannot contribute a free zero.
        macro_f1_pct=(
            f1_score(y_true, y_pred, labels=primary_present, average="macro", zero_division=0) * _PCT
            if primary_present
            else 0.0
        ),
        accuracy_pct=sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / num_evals * _PCT,
        majority_baseline_pct=majority_count / num_evals * _PCT,
        chance_balanced_acc_pct=_PCT / len(primary_present) if primary_present else 0.0,
        collapse_index_pct=prediction_counts.most_common(1)[0][1] / num_evals * _PCT,
        unparsed_pct=prediction_counts.get(UNPARSED_CHOICE, 0) / num_evals * _PCT,
        binary_disease_balanced_acc_pct=_binary_balanced_accuracy_pct(
            y_true, y_pred, lambda label: label != HEALTHY_LABEL
        ),
        copd_vs_rest_balanced_acc_pct=_binary_balanced_accuracy_pct(y_true, y_pred, lambda label: label == COPD_LABEL),
        classification_report=classification_report(y_true, y_pred, labels=list(ICBHI_LABELS), zero_division=0),
    )
