from typing import Literal
from pydantic import BaseModel, Field


class BaseModelConfig(BaseModel):
    model_id: str
    dtype: str
    max_num_seqs: int = Field(description="Batch size for the model")
    max_new_tokens: int = Field(gt=0)
    gpu_memory_utilization: float = Field(description="Fraction of GPU memory to allocate for vLLM")
    stop: list[str] | None = None


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
    target_sr: int = Field(description="Target sample rate for the audio model")
    
    # Flags passed through from ExperimentMeta
    chunked_audio: bool = False
    two_pass_localization: bool = False
    contrastive_alpha: float | None = None


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
    system_prompt: str | None = None
    rag_mapping_file: str | None = None
    use_transcript: bool = False
    few_shot_mode: Literal["none", "text", "audio", "rag"] = "none"
    few_shot_include_transcript: bool = False
    use_cot: bool = False
    
    # T1 Flags
    inject_diarized_transcript: bool = False
    inject_acoustic_features: bool = False
    
    # T2 Flags
    chunked_audio: bool = False
    two_pass_localization: bool = False
    
    # T3 Flags
    contrastive_alpha: float | None = None
    
    # T4 Flags
    num_shuffled_variants: int = 1
    
    # T5 Flags
    audio_first_instruction: bool = False
    
    deprecated: bool = False


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
    stop: list[str] | None = None
