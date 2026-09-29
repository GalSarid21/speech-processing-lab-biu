import abc
import json
import re

from loguru import logger

from speech_processing.config.core import AudioPresentationMode, ExperimentMeta
from speech_processing.utils.audio import chunk_audio, crop_audio, force_mono_audio, get_duration_s
from speech_processing.utils.consts import (
    CHUNK_LENGTH_S,
    LOCALIZATION_PADDING_S,
    MAX_AUDIO_CHUNKS,
    MAX_BREATHING_CYCLES,
)

ContentPart = dict[str, str | dict[str, str]]

_JSON_OBJECT_PATTERN = re.compile(r"\{[^{}]*\}")

FULL_AUDIO_LABEL = "Full audio:"
SEGMENT_LABEL_TEMPLATE = "Segment {index} ({start:.1f}-{end:.1f} s):"
CYCLE_LABEL_TEMPLATE = "Cycle {index} ({start:.1f}-{end:.1f} s):"
CROPPED_SEGMENT_LABEL_TEMPLATE = "Cropped segment ({start:.1f}-{end:.1f} s):"
SPAN_START_KEY = "start"
SPAN_END_KEY = "end"


def text_part(text: str) -> ContentPart:
    return {"type": "text", "text": text}


def audio_part(path: str, target_sr: int) -> ContentPart:
    return {"type": "audio_url", "audio_url": {"url": force_mono_audio(path, target_sr)}}


def parse_time_span(text: str, duration_s: float) -> tuple[float, float] | None:
    """Extracts the first well-formed {"start": .., "end": ..} object, clamped to the clip."""
    for match in _JSON_OBJECT_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(0))
            start = max(0.0, min(float(payload[SPAN_START_KEY]), duration_s))
            end = max(0.0, min(float(payload[SPAN_END_KEY]), duration_s))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if start < end:
            return start, end
    return None


class AudioPresentation(abc.ABC):
    """Strategy deciding how a clip is laid out in the conversation (T2)."""

    def __init__(self, target_sr: int) -> None:
        self.target_sr = target_sr

    @abc.abstractmethod
    def first_turn(self, audio_path: str, instruction: str, work_dir: str) -> list[ContentPart]:
        """Content parts for the opening user turn."""

    def followup_turn(
        self, audio_path: str, instruction: str, previous_output: str, work_dir: str
    ) -> list[ContentPart]:
        # The audio is already in the conversation, so later turns are text by default.
        return [text_part(instruction)]

    def log_summary(self) -> None:  # noqa: B027 - optional hook, subclasses may stay silent
        """Hook for strategies that accumulate diagnostics across a batch."""


class FullClipPresentation(AudioPresentation):
    def first_turn(self, audio_path: str, instruction: str, work_dir: str) -> list[ContentPart]:
        return [audio_part(audio_path, self.target_sr), text_part(instruction)]


class ChunkedPresentation(AudioPresentation):
    def __init__(
        self,
        target_sr: int,
        chunk_len_s: float = CHUNK_LENGTH_S,
        max_chunks: int = MAX_AUDIO_CHUNKS,
    ) -> None:
        super().__init__(target_sr)
        self.chunk_len_s = chunk_len_s
        self.max_chunks = max_chunks

    def first_turn(self, audio_path: str, instruction: str, work_dir: str) -> list[ContentPart]:
        content: list[ContentPart] = []
        for index, (chunk_path, start_s, end_s) in enumerate(
            chunk_audio(audio_path, self.chunk_len_s, self.max_chunks, work_dir), start=1
        ):
            content.append(text_part(SEGMENT_LABEL_TEMPLATE.format(index=index, start=start_s, end=end_s)))
            content.append(audio_part(chunk_path, self.target_sr))

        content.append(text_part(FULL_AUDIO_LABEL))
        content.append(audio_part(audio_path, self.target_sr))
        content.append(text_part(instruction))
        return content


class CyclePresentation(AudioPresentation):
    """A4: presents the breathing cycles measured by the A1 feature script, then the full clip.

    Cycle boundaries are injected per audio path rather than read inside the strategy, so the
    presentation stays a pure layout decision and the dataset layer stays the only thing that
    touches the feature artifact. A clip with no measured cycles degrades to the full clip.
    """

    def __init__(
        self,
        target_sr: int,
        spans_by_path: dict[str, list[tuple[float, float]]] | None = None,
        max_cycles: int = MAX_BREATHING_CYCLES,
    ) -> None:
        super().__init__(target_sr)
        self.spans_by_path = spans_by_path or {}
        self.max_cycles = max_cycles
        self.num_with_cycles = 0
        self.num_without_cycles = 0

    def first_turn(self, audio_path: str, instruction: str, work_dir: str) -> list[ContentPart]:
        spans = self.spans_by_path.get(audio_path, [])[: self.max_cycles]
        if not spans:
            self.num_without_cycles += 1
            return [audio_part(audio_path, self.target_sr), text_part(instruction)]

        self.num_with_cycles += 1
        content: list[ContentPart] = []
        for index, (start_s, end_s) in enumerate(spans, start=1):
            content.append(text_part(CYCLE_LABEL_TEMPLATE.format(index=index, start=start_s, end=end_s)))
            content.append(audio_part(crop_audio(audio_path, start_s, end_s, work_dir), self.target_sr))

        content.append(text_part(FULL_AUDIO_LABEL))
        content.append(audio_part(audio_path, self.target_sr))
        content.append(text_part(instruction))
        return content

    def log_summary(self) -> None:
        total = self.num_with_cycles + self.num_without_cycles
        if not total:
            return
        logger.info(
            f"Cycle presentation: {self.num_with_cycles}/{total} clips had measured cycles, "
            f"{self.num_without_cycles} fell back to the full clip."
        )


class TwoPassLocalizationPresentation(AudioPresentation):
    def __init__(self, target_sr: int, padding_s: float = LOCALIZATION_PADDING_S) -> None:
        super().__init__(target_sr)
        self.padding_s = padding_s
        self.num_crops = 0
        self.num_fallbacks = 0

    def first_turn(self, audio_path: str, instruction: str, work_dir: str) -> list[ContentPart]:
        return [audio_part(audio_path, self.target_sr), text_part(instruction)]

    def followup_turn(
        self, audio_path: str, instruction: str, previous_output: str, work_dir: str
    ) -> list[ContentPart]:
        duration_s = get_duration_s(audio_path)
        span = parse_time_span(previous_output, duration_s)

        if span is None:
            logger.warning("No valid evidence span in the first-pass reply; falling back to a text-only turn.")
            self.num_fallbacks += 1
            return [text_part(instruction)]

        start = max(0.0, span[0] - self.padding_s)
        end = min(duration_s, span[1] + self.padding_s)
        cropped_path = crop_audio(audio_path, start, end, work_dir)
        self.num_crops += 1

        return [
            text_part(CROPPED_SEGMENT_LABEL_TEMPLATE.format(start=start, end=end)),
            audio_part(cropped_path, self.target_sr),
            text_part(instruction),
        ]

    def log_summary(self) -> None:
        total = self.num_crops + self.num_fallbacks
        if not total:
            return
        logger.info(
            f"Two-pass localization: cropped {self.num_crops}/{total} "
            f"({self.num_crops / total:.1%}), fell back {self.num_fallbacks}/{total} "
            f"({self.num_fallbacks / total:.1%})."
        )


_PRESENTATION_BY_MODE: dict[AudioPresentationMode, type[AudioPresentation]] = {
    "full_clip": FullClipPresentation,
    "chunked": ChunkedPresentation,
    "two_pass_localization": TwoPassLocalizationPresentation,
}


def build_presentation(
    meta: ExperimentMeta | None,
    target_sr: int,
    cycle_spans_by_path: dict[str, list[tuple[float, float]]] | None = None,
) -> AudioPresentation:
    mode: AudioPresentationMode = meta.presentation if meta else "full_clip"
    if mode == "cycles":
        return CyclePresentation(target_sr, cycle_spans_by_path)
    return _PRESENTATION_BY_MODE[mode](target_sr)
