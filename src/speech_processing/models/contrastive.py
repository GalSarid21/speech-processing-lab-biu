import itertools
import tempfile
from typing import Protocol

from loguru import logger

from speech_processing.config.core import CalibrationMode
from speech_processing.data.choices import letters_for
from speech_processing.data.dtos import AudioRequest, AudioResponse
from speech_processing.models.presentation import audio_part, text_part
from speech_processing.utils.audio import PINK_NOISE, WHITE_NOISE, get_duration_s, write_noise, write_silence
from speech_processing.utils.consts import CONTENT_FREE_SEED, TEMP_AUDIO_DIR_PREFIX
from speech_processing.utils.exceptions import ContrastiveScoringError


class FinalTokenScorer(Protocol):
    """The backend capability ContrastiveChoiceScorer depends on (vLLM in production, a fake in tests)."""

    def score_last_prompt_token(self, conversations: list[list[dict]]) -> list[tuple[int, float]]: ...

    def decode_token(self, token_id: int) -> str: ...


class ContrastiveChoiceScorer:
    """T3 / A5: score(L) = logp(L | real audio) - alpha * mean_r logp(L | content-free reference r).

    The answer is scored as a single-token assistant message, which removes the length bias
    that plagues whole-sentence scoring and makes the answer position unambiguous.

    `calibration` chooses the reference signals. "silence" uses one duration-matched silent clip.
    "content_free" averages over silence, white noise and pink noise, which estimates the model's
    label prior more robustly than any single reference and is the contextual-calibration variant.
    """

    def __init__(
        self,
        alpha: float,
        target_sr: int,
        batch_size: int,
        calibration: CalibrationMode = "silence",
    ) -> None:
        self.alpha = alpha
        self.target_sr = target_sr
        self.batch_size = batch_size
        self.calibration = calibration

    def _reference_paths(self, duration_s: float, work_dir: str) -> list[str]:
        silence = write_silence(duration_s, self.target_sr, work_dir)
        if self.calibration != "content_free":
            return [silence]
        return [
            silence,
            write_noise(duration_s, self.target_sr, work_dir, WHITE_NOISE, CONTENT_FREE_SEED),
            write_noise(duration_s, self.target_sr, work_dir, PINK_NOISE, CONTENT_FREE_SEED),
        ]

    def score(self, backend: FinalTokenScorer, requests: list[AudioRequest]) -> list[AudioResponse]:
        responses: list[AudioResponse] = []
        for batch in itertools.batched(requests, self.batch_size):
            responses.extend(self._score_batch(backend, list(batch)))
        return responses

    def _score_batch(self, backend: FinalTokenScorer, batch: list[AudioRequest]) -> list[AudioResponse]:
        with tempfile.TemporaryDirectory(prefix=TEMP_AUDIO_DIR_PREFIX) as work_dir:
            conversations: list[list[dict]] = []
            keys: list[tuple[int, str]] = []

            for req_index, req in enumerate(batch):
                references = self._reference_paths(get_duration_s(req.audio_path), work_dir)
                instruction = req.instruction if isinstance(req.instruction, str) else req.instruction[-1]

                for letter in letters_for(self._choices_of(req)):
                    keys.append((req_index, letter))
                    for audio_path in (req.audio_path, *references):
                        conversations.append(
                            [
                                {
                                    "role": "user",
                                    "content": [
                                        audio_part(audio_path, self.target_sr),
                                        text_part(instruction),
                                    ],
                                },
                                {"role": "assistant", "content": letter},
                            ]
                        )

            stride = 1 + len(references)
            scored = backend.score_last_prompt_token(conversations)

        scores_by_request: list[dict[str, float]] = [{} for _ in batch]
        for key_index, (req_index, letter) in enumerate(keys):
            group = scored[key_index * stride : (key_index + 1) * stride]
            for token_id, _ in group:
                self._assert_answer_token(backend, token_id, letter)
            real_logprob = group[0][1]
            reference_logprob = sum(logprob for _, logprob in group[1:]) / (stride - 1)
            scores_by_request[req_index][letter] = real_logprob - self.alpha * reference_logprob

        responses = []
        for req, scores in zip(batch, scores_by_request, strict=True):
            best_letter = max(scores, key=scores.__getitem__)
            logger.debug(f"Contrastive scores for {req.audio_path}: {scores}")
            responses.append(
                AudioResponse(
                    sample_id=req.audio_path,
                    instruction=req.instruction,
                    generated_text=best_letter,
                    final_turn_text=best_letter,
                    metadata=req.metadata.model_copy(update={"choice_scores": scores}) if req.metadata else None,
                )
            )
        return responses

    @staticmethod
    def _choices_of(req: AudioRequest) -> list[str]:
        if not req.metadata or not req.metadata.choices:
            raise ContrastiveScoringError(f"Request {req.audio_path} carries no choices to score.")
        return req.metadata.choices

    @staticmethod
    def _assert_answer_token(backend: FinalTokenScorer, token_id: int, letter: str) -> None:
        decoded = backend.decode_token(token_id).strip()
        if decoded != letter:
            raise ContrastiveScoringError(
                f"Scored the token {decoded!r} instead of the answer {letter!r}: the chat template appends "
                "tokens after the assistant message, so the final prompt token is not the answer."
            )
