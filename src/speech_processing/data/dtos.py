from pydantic import BaseModel, Field


class AudioRequest(BaseModel):
    instruction: str
    audio_path: str
    audio_bytes: bytes | None = None


class AudioResponse(BaseModel):
    instruction: str
    generated_text: str


class JudgeRequest(BaseModel):
    instruction: str
    generated_text: str
    ground_truth: str


class EvaluationResult(BaseModel):
    # CRITICAL: 'reasoning' must be the FIRST field.
    # This forces the model to generate its thought process before committing to a score.
    reasoning: str = Field(
        description="A brief, 1-2 sentence explanation justifying the rating."
    )
    acoustic_accuracy: int = Field(
        ge=0,
        le=10,
        description="0 to 10 based on how well it detected raw acoustic features like crackles or wheezes.",
    )
    diagnostic_accuracy: int = Field(
        ge=0,
        le=10,
        description="0 to 10 based on how well it deduced the patient-level pathology (e.g. COPD).",
    )
    hallucination_penalty: int = Field(
        ge=0,
        le=1,
        description="1 if the model hallucinated or made up sounds not present, 0 otherwise.",
    )
    extracted_disease_class: str = Field(
        description="The exact disease class the audio model predicted. Must be one of: COPD, Healthy, URTI, Bronchiectasis, Pneumonia, Bronchiolitis, LRTI, Asthma, or 'Unknown'"
    )


class JudgeResponse(BaseModel):
    request: JudgeRequest
    evaluation: EvaluationResult


class AggregateMetrics(BaseModel):
    avg_acoustic_pct: float
    avg_diagnostic_pct: float
    hallucination_rate_pct: float
    classification_report: str
