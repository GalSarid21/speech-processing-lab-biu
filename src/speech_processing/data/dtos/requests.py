from pydantic import BaseModel, Field


class FewShotTurn(BaseModel):
    audio_bytes: bytes | list[bytes] | None = None
    audio_path: str | list[str] | None = None
    user_text: str
    assistant_text: str


class BaseRequest(BaseModel):
    instruction: str | list[str]
    assistant_prefill: str | None = None


class AudioRequest(BaseRequest):
    audio_path: str
    audio_bytes: bytes | None = None
    few_shot_turns: list[FewShotTurn] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class JudgeRequestParseError(Exception):
    """Raised when a JudgeRequest fails to parse from JSON."""
    pass

class JudgeRequest(BaseRequest):
    sample_id: str
    generated_text: str
    ground_truth: str

    @classmethod
    def from_json(cls, json_str: str) -> 'JudgeRequest':
        import json
        try:
            data = json.loads(json_str)
            return cls(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise JudgeRequestParseError(f"Failed to parse JudgeRequest from JSON: {json_str}") from e


class TextRequest(BaseRequest):
    few_shot_turns: list[FewShotTurn] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
