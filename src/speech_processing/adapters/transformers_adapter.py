from typing import Any

import torch

from speech_processing.adapters.base import BaseGenerationAdapter


class TransformersAdapter(BaseGenerationAdapter):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def generate_batch(
        self, texts: list[str], audios: list[Any], max_length: int = 256
    ) -> list[str]:
        inputs = self.processor(
            text=texts, audios=audios, return_tensors="pt", padding=True
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_length=max_length)

        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

        return self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
