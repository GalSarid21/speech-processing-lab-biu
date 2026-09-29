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
    classification_report: str


class ICBHIAggregateMetrics(BaseModel):
    """Closed-set ICBHI metrics. Balanced accuracy is primary; every report carries its own baselines."""

    balanced_accuracy_pct: float
    macro_f1_pct: float
    accuracy_pct: float
    majority_baseline_pct: float
    chance_balanced_acc_pct: float
    collapse_index_pct: float
    unparsed_pct: float
    binary_disease_balanced_acc_pct: float
    copd_vs_rest_balanced_acc_pct: float
    classification_report: str
