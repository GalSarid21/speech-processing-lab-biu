import os
import tempfile
from collections.abc import Iterator
from io import BytesIO
from urllib.request import urlopen

import librosa
from loguru import logger

from speech_processing.adapters.openai_server import OpenAIChatServerAdapter
from speech_processing.adapters.transformers import TransformersAdapter
from speech_processing.adapters.vllm import VLLMAdapter
from speech_processing.config.core import AudioModelConfig, GenerationParams
from speech_processing.data.dtos import AudioRequest, AudioResponse, FewShotTurn
from speech_processing.models.base import BaseAudioModel
from speech_processing.models.contrastive import ContrastiveChoiceScorer
from speech_processing.models.preprocessing import AudioPreprocessor, IdentityPreprocessor
from speech_processing.models.presentation import (
    AudioPresentation,
    ContentPart,
    FullClipPresentation,
    audio_part,
    text_part,
)
from speech_processing.utils.audio import FILE_URL_PREFIX, get_duration_s
from speech_processing.utils.consts import (
    COT_START_TAG,
    TEMP_AUDIO_DIR_PREFIX,
    VOXTRAL_AUDIO_TOKENS_PER_SECOND,
    VOXTRAL_MAX_AUDIOS_PER_PROMPT,
)
from speech_processing.utils.exceptions import AudioProcessingError

MULTI_TURN_ERROR_MARKER = "\n[Error processing turn]"
CHARS_PER_TOKEN_ESTIMATE = 4
QWEN_AUDIO_TOKEN = "<|AUDIO|>"


def _iter_content_parts(conversation: list[dict]) -> Iterator[ContentPart]:
    for message in conversation:
        content = message.get("content")
        if isinstance(content, list):
            yield from content
        elif isinstance(content, str):
            yield text_part(content)


class QwenAudioEngine(BaseAudioModel):
    def __init__(self, config: AudioModelConfig) -> None:
        self.config = config
        self.adapter = TransformersAdapter(
            model_id=config.model_id,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
        )

    def batch_infer(self, requests: list[AudioRequest]) -> list[AudioResponse]:
        batch_size = self.config.max_num_seqs
        responses: list[AudioResponse] = []

        for start in range(0, len(requests), batch_size):
            batch_reqs = requests[start : start + batch_size]

            valid_reqs: list[AudioRequest] = []
            batch_conversations: list[list[dict]] = []
            batch_audios: list[list] = []

            for req in batch_reqs:
                conversation: list[dict] = []
                if req.system_prompt:
                    conversation.append({"role": "system", "content": req.system_prompt})
                req_audios: list = []

                try:
                    for turn in req.few_shot_turns:
                        user_text = turn.user_text
                        if isinstance(turn.audio_path, list):
                            for path, audio_bytes in zip(turn.audio_path, turn.audio_bytes, strict=True):
                                user_text = f"{QWEN_AUDIO_TOKEN}\n{user_text}"
                                req_audios.append(self._load_audio(audio_bytes, path))
                        elif turn.audio_path is not None:
                            user_text = f"{QWEN_AUDIO_TOKEN}\n{user_text}"
                            req_audios.append(self._load_audio(turn.audio_bytes, turn.audio_path))

                        conversation.append({"role": "user", "content": user_text})
                        conversation.append({"role": "assistant", "content": turn.assistant_text})

                    req_audios.append(self._load_audio(req.audio_bytes, req.audio_path))

                    batch_conversations.append(conversation)
                    batch_audios.append(req_audios)
                    valid_reqs.append(req)
                except RuntimeError as e:
                    logger.error(f"Failed to load audio {req.audio_path}: {e}")

            if not valid_reqs:
                continue

            responses.extend(self._run_turns(valid_reqs, batch_conversations, batch_audios))

        return responses

    def _load_audio(self, audio_bytes: bytes | None, audio_path: str | None):
        target_sr = self.adapter.processor.feature_extractor.sampling_rate
        if audio_bytes is not None:
            return librosa.load(BytesIO(audio_bytes), sr=target_sr)[0]
        if audio_path is not None and audio_path.startswith("http"):
            return librosa.load(BytesIO(urlopen(audio_path).read()), sr=target_sr)[0]
        return librosa.load(audio_path, sr=target_sr)[0]

    def _run_turns(
        self,
        valid_reqs: list[AudioRequest],
        batch_conversations: list[list[dict]],
        batch_audios: list[list],
    ) -> list[AudioResponse]:
        num_steps = _num_turns(valid_reqs[0])
        accumulated = [""] * len(valid_reqs)
        last_turn_texts = [""] * len(valid_reqs)

        for step in range(num_steps):
            texts = []
            for idx, conv in enumerate(batch_conversations):
                instruction = _turn_instruction(valid_reqs[idx], step)
                content = instruction if step else f"{instruction}\n{QWEN_AUDIO_TOKEN}"
                conv.append({"role": "user", "content": content})

                base_text = self.adapter.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                if COT_START_TAG in instruction:
                    base_text += f"{COT_START_TAG}\n"
                texts.append(base_text)

            logger.info(f"Processing audio batch of size {len(valid_reqs)} (Turn {step + 1}/{num_steps})...")

            try:
                generated_texts = self.adapter.generate_batch(
                    texts=texts,
                    audios=batch_audios,
                    sampling_params=GenerationParams(
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        max_new_tokens=self.config.max_new_tokens,
                    ),
                )
            except RuntimeError as e:
                logger.error(f"Failed to process batch on turn {step + 1}: {e}")
                for idx in range(len(valid_reqs)):
                    accumulated[idx] += MULTI_TURN_ERROR_MARKER
                    last_turn_texts[idx] = MULTI_TURN_ERROR_MARKER
                continue

            for idx, raw_text in enumerate(generated_texts):
                instruction = _turn_instruction(valid_reqs[idx], step)
                turn_text = f"{COT_START_TAG}\n{raw_text}" if COT_START_TAG in instruction else raw_text

                batch_conversations[idx].append({"role": "assistant", "content": turn_text})
                last_turn_texts[idx] = turn_text
                if num_steps > 1:
                    accumulated[idx] += f"Turn {step + 1} - User: {instruction}\nAssistant: {turn_text}\n\n"
                else:
                    accumulated[idx] = turn_text

        return _build_responses(valid_reqs, accumulated, last_turn_texts)


class VoxtralAudioEngine(BaseAudioModel):
    def __init__(
        self,
        config: AudioModelConfig,
        presentation: AudioPresentation | None = None,
        scorer: ContrastiveChoiceScorer | None = None,
        preprocessor: AudioPreprocessor | None = None,
    ) -> None:
        self.config = config
        self.presentation = presentation or FullClipPresentation(config.target_sr)
        self.scorer = scorer
        self.preprocessor = preprocessor or IdentityPreprocessor(config.target_sr)

        if config.server_url:
            if scorer is not None:
                raise AudioProcessingError(
                    "Contrastive letter scoring needs prompt log-probs from an in-process engine; "
                    "it is not supported with --audio-server-url."
                )
            self.adapter: VLLMAdapter | OpenAIChatServerAdapter = OpenAIChatServerAdapter(
                base_url=config.server_url,
                model=config.model_id,
                reasoning_prefix=config.reasoning_prefix,
                stop_token_ids=config.stop_token_ids,
            )
        else:
            self.adapter = VLLMAdapter(
                model=config.model_id,
                trust_remote_code=True,
                max_model_len=config.max_model_len,
                limit_mm_per_prompt={"audio": VOXTRAL_MAX_AUDIOS_PER_PROMPT},
                gpu_memory_utilization=config.gpu_memory_utilization,
                max_num_seqs=config.max_num_seqs,
                allowed_local_media_path="/",
            )
        self.tokenizer = self.adapter.tokenizer

    def batch_infer(self, requests: list[AudioRequest]) -> list[AudioResponse]:
        if not requests:
            return []

        with tempfile.TemporaryDirectory(prefix=TEMP_AUDIO_DIR_PREFIX) as work_dir:
            prepared = self._preprocess(requests, work_dir)
            if self.scorer is not None:
                return _restore_sample_ids(self.scorer.score(self.adapter, prepared), requests)
            responses = _restore_sample_ids(self._generate(prepared, work_dir), requests)

        self.presentation.log_summary()
        return responses

    def _preprocess(self, requests: list[AudioRequest], work_dir: str) -> list[AudioRequest]:
        """Applies the A3 preprocessor to the target clip, keeping the original sample_id stable."""
        if isinstance(self.preprocessor, IdentityPreprocessor):
            return requests
        return [
            req.model_copy(update={"audio_path": self.preprocessor.process(req.audio_path, work_dir)})
            for req in requests
        ]

    def _generate(self, requests: list[AudioRequest], work_dir: str) -> list[AudioResponse]:
        turn_counts = {_num_turns(req) for req in requests}
        if len(turn_counts) > 1:
            raise AudioProcessingError(f"A batch cannot mix conversations of different lengths: {sorted(turn_counts)}.")
        num_steps = turn_counts.pop()

        conversations: list[list[dict]] = [[] for _ in requests]
        accumulated = [""] * len(requests)
        last_turn_texts = [""] * len(requests)

        for step in range(num_steps):
            prompts = []
            for idx, conv in enumerate(conversations):
                req = requests[idx]
                instruction = _turn_instruction(req, step)

                if step == 0:
                    self._append_few_shot_turns(conv, req)
                    prefix = f"{req.system_prompt}\n\n" if req.system_prompt and not req.few_shot_turns else ""
                    content = self.presentation.first_turn(req.audio_path, prefix + instruction, work_dir)
                else:
                    content = self.presentation.followup_turn(
                        req.audio_path, instruction, last_turn_texts[idx], work_dir
                    )

                conv.append({"role": "user", "content": content})
                if req.assistant_prefill:
                    conv.append({"role": "assistant", "content": req.assistant_prefill})

                self._check_prompt_budget(conv)
                prompts.append(list(conv))

            logger.info(f"Processing audio batch of size {len(requests)} (Turn {step + 1}/{num_steps})...")

            try:
                generated_texts = self.adapter.generate_batch(
                    prompts=prompts,
                    sampling_params=GenerationParams(
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        max_new_tokens=self.config.max_new_tokens,
                        stop=self.config.stop,
                    ),
                )
            except RuntimeError as e:
                logger.error(f"Failed to process batch on turn {step + 1}: {e}")
                for idx in range(len(requests)):
                    accumulated[idx] += MULTI_TURN_ERROR_MARKER
                    last_turn_texts[idx] = MULTI_TURN_ERROR_MARKER
                continue

            for idx, raw_text in enumerate(generated_texts):
                req = requests[idx]
                turn_text = raw_text
                if req.assistant_prefill:
                    turn_text = req.assistant_prefill + raw_text
                    conversations[idx].pop()

                conversations[idx].append({"role": "assistant", "content": turn_text})
                last_turn_texts[idx] = turn_text

                if num_steps > 1:
                    instruction = _turn_instruction(req, step)
                    accumulated[idx] += f"Turn {step + 1} - User: {instruction}\nAssistant: {turn_text}\n\n"
                else:
                    accumulated[idx] = turn_text

        return _build_responses(requests, accumulated, last_turn_texts)

    def _append_few_shot_turns(self, conversation: list[dict], req: AudioRequest) -> None:
        system_prefix = f"{req.system_prompt}\n\n" if req.system_prompt else ""
        for index, turn in enumerate(req.few_shot_turns):
            content: list[ContentPart] = [
                audio_part(path, self.config.target_sr) for path in _few_shot_audio_paths(turn)
            ]
            content.append(text_part(system_prefix + turn.user_text if index == 0 else turn.user_text))
            conversation.append({"role": "user", "content": content})
            conversation.append({"role": "assistant", "content": turn.assistant_text})

    def _check_prompt_budget(self, conversation: list[dict]) -> None:
        num_audios = 0
        audio_seconds = 0.0
        text_chars = 0

        for part in _iter_content_parts(conversation):
            if part["type"] == "text":
                text_chars += len(part["text"])
                continue
            num_audios += 1
            audio_seconds += _audio_url_duration_s(part["audio_url"]["url"])

        if num_audios > VOXTRAL_MAX_AUDIOS_PER_PROMPT:
            raise AudioProcessingError(
                f"Prompt carries {num_audios} audio clips, above the Voxtral limit of {VOXTRAL_MAX_AUDIOS_PER_PROMPT}."
            )

        estimate = (
            audio_seconds * VOXTRAL_AUDIO_TOKENS_PER_SECOND
            + text_chars / CHARS_PER_TOKEN_ESTIMATE
            + self.config.max_new_tokens
        )
        if estimate > self.config.max_model_len:
            logger.warning(
                f"Estimated prompt budget {estimate:.0f} tokens exceeds max_model_len "
                f"{self.config.max_model_len} ({num_audios} clips, {audio_seconds:.1f} s audio)."
            )


def _num_turns(req: AudioRequest) -> int:
    return len(req.instruction) if isinstance(req.instruction, list) else 1


def _turn_instruction(req: AudioRequest, step: int) -> str:
    return req.instruction[step] if isinstance(req.instruction, list) else req.instruction


def _few_shot_audio_paths(turn: FewShotTurn) -> list[str]:
    if isinstance(turn.audio_path, list):
        return turn.audio_path
    return [turn.audio_path] if turn.audio_path is not None else []


def _audio_url_duration_s(url: str) -> float:
    path = url[len(FILE_URL_PREFIX) :] if url.startswith(FILE_URL_PREFIX) else url
    if not os.path.exists(path):
        return 0.0
    return get_duration_s(path)


def _restore_sample_ids(responses: list[AudioResponse], requests: list[AudioRequest]) -> list[AudioResponse]:
    """Keeps the sample_id pointing at the original recording, not a preprocessed temp copy."""
    return [
        resp if resp.sample_id == req.audio_path else resp.model_copy(update={"sample_id": req.audio_path})
        for resp, req in zip(responses, requests, strict=True)
    ]


def _build_responses(
    requests: list[AudioRequest], accumulated: list[str], last_turn_texts: list[str]
) -> list[AudioResponse]:
    responses = []
    for req, full_text, last_text in zip(requests, accumulated, last_turn_texts, strict=True):
        instruction = str(req.instruction) if isinstance(req.instruction, list) else req.instruction
        responses.append(
            AudioResponse(
                sample_id=req.audio_path,
                instruction=instruction,
                generated_text=full_text.strip(),
                final_turn_text=last_text.strip(),
                metadata=req.metadata,
            )
        )
    return responses
