from pydantic import BaseModel, Field


class BaseModelConfig(BaseModel):
    model_id: str
    dtype: str
    max_num_seqs: int = Field(description="Batch size for the model")
    max_new_tokens: int = Field(gt=0)
    gpu_memory_utilization: float = Field(description="Fraction of GPU memory to allocate for vLLM")


class JudgeConfig(BaseModelConfig):
    max_model_len: int = Field(description="Max context length for the judge model")


class TextModelConfig(BaseModelConfig):
    max_model_len: int = Field(description="Max context length for the text model")
    temperature: float
    top_p: float
    top_k: int
    enable_thinking: bool


class AudioModelConfig(BaseModelConfig):
    max_model_len: int = Field(description="Max context length for the audio model")
    temperature: float = Field(description="Generation temperature")
    top_p: float = Field(description="Top p sampling")


class DatasetConfig(BaseModel):
    dataset_id: str
    target_labels: list[str]
    split: str
    num_samples: int | None = Field(default=None, description="Number of samples to evaluate. If None, uses all available.")
    sample_ids: list[str] = Field(default_factory=list, description="Specific sample IDs to load.")
    audio_base_dir: str | None = Field(default=None, description="Base directory for local audio files.")
    transcripts_file: str | None = Field(default=None, description="Path to transcripts JSON.")
    few_shot_ids_file: str | None = Field(default=None, description="Path to file containing few-shot IDs.")


class ExperimentMeta(BaseModel):
    experiment_name: str
    prompt: str | list[str]
    max_new_tokens: int
    batch_size: int


class AppConfig(BaseModel):
    """The master configuration wrapper. Single source of truth for defaults."""
    
    output_dir: str = Field(default="./results", description="Destination for pipeline artifacts")
    dataset: DatasetConfig

    judge: JudgeConfig | None = None
    audio_model: AudioModelConfig | None = None
    text_model: TextModelConfig | None = None


class GenerationParams(BaseModel):
    temperature: float
    top_p: float
    max_new_tokens: int
    json_schema: str | None = None
