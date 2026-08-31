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
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": req.audio_path},
                            {"type": "text", "text": req.instruction},
                        ],
                    }
                ]

                text = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )

                try:
                    if req.audio_bytes is not None:
                        audio_data, _ = librosa.load(
                            BytesIO(req.audio_bytes),
                            sr=self.processor.feature_extractor.sampling_rate,
                        )
                    elif req.audio_path.startswith("http"):
                        audio_data, _ = librosa.load(
                            BytesIO(urlopen(req.audio_path).read()),
                            sr=self.processor.feature_extractor.sampling_rate,
                        )
                    else:
                        audio_data, _ = librosa.load(
                            req.audio_path,
                            sr=self.processor.feature_extractor.sampling_rate,
                        )
                    texts.append(text)
                    audios.append(audio_data)
                    valid_reqs.append(req)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to load audio {req.audio_path}: {e}")

            if not texts:
                continue

            logger.info(f"Processing audio batch of size {len(valid_reqs)}...")

            try:
                # Use adapter to generate
                generated_texts = self.adapter.generate_batch(
                    texts=texts, audios=audios, max_length=self.config.max_new_tokens
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
