from pydantic import BaseModel, Field

from speech_processing.data.dtos.metadata import ItemMetadata
from speech_processing.utils.consts import UNPARSED_CHOICE


class BaseResponse(BaseModel):
    sample_id: str
    instruction: str | list[str]
    generated_text: str
    final_turn_text: str | None = Field(default=None, description="The last assistant turn only.")
    metadata: ItemMetadata | None = None


class AudioResponse(BaseResponse):
    pass


class TextResponse(BaseResponse):
    pass


class EvaluationResult(BaseModel):
    # CRITICAL: 'reasoning' must be the FIRST field.
    # This forces the model to generate its thought process before committing to a score.
    reasoning: str = Field(description="A brief, 1-2 sentence explanation justifying the rating.")
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
    def fallback(cls) -> "EvaluationResult":
        return cls(
            reasoning="Parse failed.",
            acoustic_accuracy=0,
            diagnostic_accuracy=0,
            hallucination_penalty=1,
            extracted_class=UNPARSED_CHOICE,
        )


class ChoiceExtractionResult(BaseModel):
    """What a judge returns for any closed-set, lettered-choice benchmark.

    The judge only extracts; correctness is always computed in code, never asserted by the judge.
    """

    reasoning: str = Field(description="One sentence locating the model's final selected choice.")
    extracted_choice: str = Field(
        description="The single letter (A, B, C, ...) of the model's final choice, or 'Unknown'."
    )

    @classmethod
    def fallback(cls) -> "ChoiceExtractionResult":
        return cls(reasoning="Parse failed.", extracted_choice=UNPARSED_CHOICE)


# MMAR-facing name for the same contract, kept so existing MMAR call sites read naturally.
MMAREvaluationResult = ChoiceExtractionResult


class JudgeResponse(BaseResponse):
    ground_truth: str
    evaluation: EvaluationResult | ChoiceExtractionResult
