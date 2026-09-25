from pydantic import BaseModel, Field


class AggregateMetrics(BaseModel):
    avg_acoustic_pct: float
    avg_diagnostic_pct: float
    hallucination_rate_pct: float
    classification_report: str
