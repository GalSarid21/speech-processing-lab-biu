from io import BytesIO
from urllib.request import urlopen

import librosa
import torch
from loguru import logger
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from speech_processing.adapters.transformers_adapter import TransformersAdapter
from speech_processing.config.core import AudioModelConfig
from speech_processing.data.dtos import AudioRequest, AudioResponse
from speech_processing.models.base import BaseAudioModel


class QwenAudioEngine(BaseAudioModel):
    def __init__(self, config: AudioModelConfig):
        self.config = config
        logger.info(
            f"Loading Qwen Audio Model from {config.model_id} via transformers..."
        )
        self.processor = AutoProcessor.from_pretrained(config.model_id)

        torch_dtype: torch.dtype = getattr(torch, config.dtype, torch.bfloat16)

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            config.model_id, torch_dtype=torch_dtype, device_map="auto"
        )
        self.model.eval()

        self.adapter = TransformersAdapter(self.model, self.processor)

    def batch_infer(self, requests: list[AudioRequest]) -> list[AudioResponse]:
        batch_size = self.config.max_num_seqs
        responses = []

        for i in range(0, len(requests), batch_size):
            batch_reqs = requests[i : i + batch_size]
            texts = []
            audios = []
            valid_reqs = []

            for req in batch_reqs:
                conversation = []
                req_audios = []
                
                try:
                    # Helper to load audio
                    def load_audio(audio_bytes, audio_path):
                        if audio_bytes is not None:
                            return librosa.load(BytesIO(audio_bytes), sr=self.processor.feature_extractor.sampling_rate)[0]
                        elif audio_path.startswith("http"):
                            return librosa.load(BytesIO(urlopen(audio_path).read()), sr=self.processor.feature_extractor.sampling_rate)[0]
                        else:
                            return librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)[0]

                    # 1. Add Few-Shot Turns
                    for turn in req.few_shot_turns:
                        conversation.append({
                            "role": "user",
                            "content": [
                                {"type": "audio", "audio_url": turn.audio_path},
                                {"type": "text", "text": turn.user_text},
                            ],
                        })
                        conversation.append({
                            "role": "assistant",
                            "content": [{"type": "text", "text": turn.assistant_text}]
                        })
                        req_audios.append(load_audio(turn.audio_bytes, turn.audio_path))

                    # 2. Add Target Turn
                    conversation.append({
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": req.audio_path},
                            {"type": "text", "text": req.instruction},
                        ],
                    })
                    req_audios.append(load_audio(req.audio_bytes, req.audio_path))

                    text = self.processor.apply_chat_template(
                        conversation, add_generation_prompt=True, tokenize=False
                    )

                    texts.append(text)
                    
                    # Qwen2AudioProcessor expects a flat list if passing a single text, 
                    # but for batched texts it expects a list of lists of arrays
                    # However, if it expects a flat list for batched text too, we will flatten it in the adapter. 
                    # For now, pass list of lists.
                    audios.append(req_audios)
                    valid_reqs.append(req)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to load audio {req.audio_path}: {e}")

            if not texts:
                continue

            logger.info(f"Processing audio batch of size {len(valid_reqs)}...")

            try:
                # Use adapter to generate
                generated_texts = self.adapter.generate_batch(
                    texts=texts, audios=audios, max_new_tokens=self.config.max_new_tokens
                )

                for req, gen_text in zip(valid_reqs, generated_texts):
                    responses.append(
                        AudioResponse(
                            sample_id=req.audio_path,
                            instruction=req.instruction, 
                            generated_text=gen_text
                        )
                    )
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to process batch: {e}")

        return responses
