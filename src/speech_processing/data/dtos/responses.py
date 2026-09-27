from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    sample_id: str
    instruction: str | list[str]
    generated_text: str
    metadata: dict = Field(default_factory=dict)


class AudioResponse(BaseResponse):
    pass


class TextResponse(BaseResponse):
    pass


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
    extracted_class: str = Field(
        description="The exact disease class the audio model predicted. Must be one of: COPD, Healthy, URTI, Bronchiectasis, Pneumonia, Bronchiolitis, LRTI, Asthma, or 'Unknown'"
    )

    @classmethod
    def fallback(cls):
        return cls(reasoning="Parse failed.", acoustic_accuracy=0, diagnostic_accuracy=0, hallucination_penalty=1, extracted_class="Unknown")


class MMAREvaluationResult(BaseModel):
    reasoning: str = Field(description="A brief explanation justifying the choice.")
    is_correct: bool = Field(description="True if the extracted choice matches the ground truth.")
    extracted_choice: str = Field(description="The exact letter of the choice the audio model predicted (A, B, C, D, etc.), or 'Unknown'.")
    
    @classmethod
    def fallback(cls):
        return cls(reasoning="Parse failed.", is_correct=False, extracted_choice="Unknown")


from speech_processing.data.dtos.requests import JudgeRequest

class JudgeResponse(BaseResponse):
    ground_truth: str
    evaluation: EvaluationResult | MMAREvaluationResult
