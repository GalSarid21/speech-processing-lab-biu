from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from speech_processing.data.enums import DatasetType
from speech_processing.utils.exceptions import InvalidExperimentConfigError

FewShotMode = Literal["none", "text", "audio", "rag"]
AudioPresentationMode = Literal["full_clip", "chunked", "two_pass_localization", "cycles"]
AudioPreprocessingMode = Literal["none", "bandpass_normalize", "bandpass_normalize_shift"]
CalibrationMode = Literal["none", "silence", "content_free"]
# "core" is the pre-registered set this project reports. "future_work" is implemented, tested and
# runnable, but deliberately outside the reported scope - at n=174 every extra arm compared against
# one baseline buys another chance of a spurious result. It is NOT the same as `deprecated`, which
# means retired and refuses to run.
ExperimentTier = Literal["core", "future_work"]

TWO_PASS_NUM_TURNS = 2


class FrozenConfig(BaseModel):
    """Base for every configuration object: immutable once constructed."""

    model_config = ConfigDict(frozen=True)


class BaseModelConfig(FrozenConfig):
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


class DatasetConfig(FrozenConfig):
    dataset_id: str
    target_labels: list[str]
    split: str
    num_samples: int | None = Field(
        default=None, description="Number of samples to evaluate. If None, uses all available."
    )
    sample_ids: list[str] = Field(default_factory=list, description="Specific sample IDs to load.")
    audio_base_dir: str | None = Field(default=None, description="Base directory for local audio files.")
    transcripts_file: str | None = Field(default=None, description="Path to transcripts JSON.")
    few_shot_ids_file: str | None = Field(default=None, description="Path to file containing few-shot IDs.")
    acoustic_features_file: str | None = Field(
        default=None, description="Path to the diarized/acoustic evidence JSONL."
    )
    respiratory_features_file: str | None = Field(
        default=None, description="Path to the ICBHI measured respiratory features JSONL."
    )
    audio_tags_file: str | None = Field(default=None, description="Path to the general audio-tagger evidence JSONL.")
    neighbor_labels_file: str | None = Field(
        default=None, description="Path to the audio-embedding kNN neighbour labels JSON."
    )
    near_duplicates_file: str | None = Field(
        default=None, description="Path to the near-duplicate pairs JSON used to filter demonstrations."
    )


class ExperimentMeta(FrozenConfig):
    experiment_name: str
    dataset: DatasetType = DatasetType.MMAR
    prompt: str | list[str]
    max_new_tokens: int
    batch_size: int
    system_prompt: str | None = None
    stop: list[str] | None = None
    text_only: bool = False
    use_transcript: bool = False
    few_shot_mode: FewShotMode = "none"
    few_shot_include_transcript: bool = False
    rag_mapping_file: str | None = None
    use_cot: bool = False
    inject_diarized_transcript: bool = False
    inject_acoustic_features: bool = False
    inject_audio_tags: bool = False
    inject_neighbor_labels: bool = False
    include_recording_metadata: bool = False
    presentation: AudioPresentationMode = "full_clip"
    preprocessing: AudioPreprocessingMode = "none"
    replace_audio_with_silence: bool = False
    calibration: CalibrationMode = "none"
    contrastive_alpha: float | None = Field(default=None, ge=0.0)
    num_shuffled_variants: int = Field(default=1, ge=1)
    tier: ExperimentTier = "core"
    future_work_reason: str | None = Field(
        default=None, description="Why this arm is implemented but left out of the reported set."
    )
    deprecated: bool = False
    deprecation_reason: str | None = None

    @property
    def is_deterministic(self) -> bool:
        return self.contrastive_alpha is not None

    @property
    def uses_acoustic_evidence(self) -> bool:
        return (
            self.inject_diarized_transcript
            or self.inject_acoustic_features
            or self.inject_audio_tags
            or self.inject_neighbor_labels
        )

    @property
    def needs_cycle_spans(self) -> bool:
        return self.presentation == "cycles"

    @property
    def is_future_work(self) -> bool:
        return self.tier == "future_work"

    @model_validator(mode="after")
    def _validate_combination(self) -> "ExperimentMeta":
        if self.deprecated and self.is_future_work:
            raise InvalidExperimentConfigError(
                f"Invalid experiment '{self.experiment_name}': R14: deprecated and future_work are "
                "different exits; an experiment cannot be both."
            )
        if self.deprecated:
            return self

        violations: list[str] = []

        if self.is_future_work and not self.future_work_reason:
            violations.append("R15: a future_work experiment must say why it is out of scope.")
        if self.future_work_reason and not self.is_future_work:
            violations.append("R16: future_work_reason only applies to tier='future_work'.")

        if self.few_shot_mode == "rag" and not self.rag_mapping_file:
            violations.append("R1: few_shot_mode='rag' requires rag_mapping_file.")
        if self.rag_mapping_file and self.few_shot_mode != "rag":
            violations.append("R2: rag_mapping_file is only meaningful with few_shot_mode='rag'.")
        if self.presentation == "two_pass_localization" and not (
            isinstance(self.prompt, list) and len(self.prompt) == TWO_PASS_NUM_TURNS
        ):
            violations.append("R3: two_pass_localization requires a prompt of exactly 2 turns.")
        if self.text_only and (self.presentation != "full_clip" or self.few_shot_mode == "audio"):
            violations.append("R4: a text-only experiment cannot use audio presentation or audio few-shots.")
        if self.text_only and not (
            self.use_transcript or self.uses_acoustic_evidence or self.include_recording_metadata
        ):
            violations.append("R5: a text-only experiment must inject some textual evidence.")
        if self.presentation != "full_clip" and self.few_shot_mode in ("audio", "rag"):
            violations.append(
                "R6: non-full-clip presentation plus audio few-shots would exceed the audio-per-prompt limit."
            )
        if self.contrastive_alpha is not None and (
            self.text_only
            or self.presentation != "full_clip"
            or isinstance(self.prompt, list)
            or self.few_shot_mode != "none"
        ):
            violations.append("R7: contrastive scoring requires a single-turn, full-clip, zero-shot audio experiment.")
        if self.num_shuffled_variants > 1 and self.contrastive_alpha is not None:
            violations.append("R8: shuffled variants cannot be combined with contrastive scoring.")
        if self.include_recording_metadata and self.dataset is not DatasetType.ICBHI:
            violations.append("R9: include_recording_metadata only exists for ICBHI recordings.")
        if self.replace_audio_with_silence and (self.text_only or self.presentation != "full_clip"):
            violations.append("R10: the silence control needs a single full-clip audio prompt.")
        if self.preprocessing != "none" and self.text_only:
            violations.append("R11: audio preprocessing is meaningless for a text-only experiment.")
        if self.calibration != "none" and self.contrastive_alpha is None:
            violations.append("R12: calibration only applies to contrastive choice scoring.")
        if self.presentation == "cycles" and self.dataset is not DatasetType.ICBHI:
            violations.append("R13: cycle presentation relies on the ICBHI respiratory feature spans.")

        if violations:
            raise InvalidExperimentConfigError(f"Invalid experiment '{self.experiment_name}': " + " ".join(violations))
        return self


class AppConfig(FrozenConfig):
    """The master configuration wrapper. Single source of truth for defaults."""

    output_dir: str = Field(default="./results", description="Destination for pipeline artifacts")
    dataset: DatasetConfig

    judge: JudgeConfig | None = None
    audio_model: AudioModelConfig | None = None
    text_model: TextModelConfig | None = None


class GenerationParams(FrozenConfig):
    temperature: float
    top_p: float
    max_new_tokens: int
    json_schema: str | None = None
    stop: list[str] | None = None
