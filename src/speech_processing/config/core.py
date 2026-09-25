from pydantic import BaseModel, Field


class BaseModelConfig(BaseModel):
    model_id: str
    dtype: str
    max_num_seqs: int = Field(description="Batch size for the model")
    max_new_tokens: int = Field(gt=0)


class JudgeConfig(BaseModelConfig):
    max_model_len: int = Field(description="Max context length for the judge model")


class TextModelConfig(BaseModelConfig):
    max_model_len: int = Field(description="Max context length for the text model")
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    enable_thinking: bool = True


class AudioModelConfig(BaseModelConfig):
    max_model_len: int = Field(description="Max context length for the audio model")


class DatasetConfig(BaseModel):
    dataset_id: str
    target_labels: list[str]
    split: str
    num_samples: int | None = Field(default=None, description="Number of samples to evaluate. If None, uses all available.")
    sample_ids: list[str] = Field(default_factory=list, description="Specific sample IDs to load.")
    audio_base_dir: str | None = Field(default=None, description="Base directory for local audio files.")
    transcripts_file: str | None = Field(default=None, description="Path to transcripts JSON.")


class ExperimentMeta(BaseModel):
    experiment_name: str
    prompt: str | list[str]
    max_new_tokens: int = 256
    batch_size: int = 8


class AppConfig(BaseModel):
    """The master configuration wrapper. Single source of truth for defaults."""
    
    output_dir: str = Field(default="./results", description="Destination for pipeline artifacts")

    judge: JudgeConfig = Field(default_factory=lambda: JudgeConfig(
        model_id="Qwen/Qwen3.8-27B-FP8",
        dtype="auto",
        max_new_tokens=512,
        max_num_seqs=64,
        max_model_len=8192,
    ))
    audio_model: AudioModelConfig = Field(default_factory=lambda: AudioModelConfig(
        model_id="Qwen/Qwen2-Audio-7B-Instruct",
        dtype="bfloat16",
        max_num_seqs=8,
        max_new_tokens=256,
        max_model_len=8192,
    ))
    text_model: TextModelConfig = Field(default_factory=lambda: TextModelConfig(
        model_id="google/gemma-4-26B-A4B-it",
        dtype="bfloat16",
        max_num_seqs=4,
        max_new_tokens=512,
        max_model_len=8192,
    ))
    dataset: DatasetConfig = Field(default_factory=lambda: DatasetConfig(
        dataset_id="DynamicSuperb/RespiratorySoundClassification_ICBHI2017",
        target_labels=["COPD", "No potential disease detected", "Healthy"],
        split="test",
        num_samples=100,
    ))
