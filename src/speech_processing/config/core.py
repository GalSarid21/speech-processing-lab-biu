from pydantic import BaseModel, Field


class JudgeConfig(BaseModel):
    model_id: str
    dtype: str
    max_new_tokens: int = Field(gt=0)
    max_num_seqs: int = Field(description="Batch size for the judge model")
    max_model_len: int = Field(description="Max context length for the judge model")

class AudioModelConfig(BaseModel):
    model_id: str
    dtype: str
    max_num_seqs: int = Field(description="Batch size for the audio model")
    max_new_tokens: int = Field(gt=0)

class DatasetConfig(BaseModel):
    dataset_id: str
    target_labels: list[str]
    split: str

class AppConfig(BaseModel):
    """The master configuration wrapper. Single source of truth for defaults."""

    judge: JudgeConfig = Field(default_factory=lambda: JudgeConfig(
        model_id="Qwen/Qwen3.8-27B-FP8",
        dtype="auto",
        max_new_tokens=512,
        max_num_seqs=256,
        max_model_len=8192,
    ))
    audio_model: AudioModelConfig = Field(default_factory=lambda: AudioModelConfig(
        model_id="Qwen/Qwen2-Audio-7B-Instruct",
        dtype="bfloat16",
        max_num_seqs=512,
        max_new_tokens=256,
    ))
    dataset: DatasetConfig = Field(default_factory=lambda: DatasetConfig(
        dataset_id="DynamicSuperb/RespiratorySoundClassification_ICBHI2017",
        target_labels=["COPD", "No potential disease detected", "Healthy"],
        split="test",
    ))
