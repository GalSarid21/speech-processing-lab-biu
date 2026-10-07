from pydantic import BaseModel


class AggregateMetrics(BaseModel):
    avg_acoustic_pct: float
    avg_diagnostic_pct: float
    hallucination_rate_pct: float
    classification_report: str


class MMARAggregateMetrics(BaseModel):
    accuracy_pct: float
    judge_accuracy_pct: float
    unparsed_pct: float
    parser_judge_agreement_pct: float
    # Share of answers written mostly in a CJK script - a model drifting out of English (evaluation/language.py).
    non_english_pct: float
    classification_report: str


class ICBHIAggregateMetrics(BaseModel):
    """Closed-set ICBHI metrics. Balanced accuracy is primary; every report carries its own baselines.

    `balanced_accuracy_pct` averages over the primary classes only; the all-classes variant is kept as
    a secondary figure so the effect of the two tiny classes stays visible.
    """

    balanced_accuracy_pct: float
    balanced_accuracy_all_classes_pct: float
    macro_f1_pct: float
    accuracy_pct: float
    majority_baseline_pct: float
    chance_balanced_acc_pct: float
    collapse_index_pct: float
    unparsed_pct: float
    binary_disease_balanced_acc_pct: float
    copd_vs_rest_balanced_acc_pct: float
    non_english_pct: float
    classification_report: str
