from typing import Any

import torch

from speech_processing.adapters.base import BaseGenerationAdapter


class TransformersAdapter(BaseGenerationAdapter):
    def __init__(self, model, processor, max_model_len: int):
        self.model = model
        self.processor = processor
        self.max_model_len = max_model_len

    def generate_batch(
        self, texts: list[str], audios: list[Any], max_new_tokens: int = 256
    ) -> list[str]:
        # Qwen2AudioProcessor expects a flat list of audios, even for batched text!
        flat_audios = []
        for a in audios:
            if isinstance(a, list):
                flat_audios.extend(a)
            else:
                flat_audios.append(a)
                
        inputs = self.processor(
            text=texts, 
            audio=flat_audios, 
            return_tensors="pt", 
            padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate
        )
        inputs = inputs.to(self.model.device)
        
        # Dimension 1 represents the sequence length of the tokenized inputs
        seq_len = inputs.input_ids.size(1)
        max_len = getattr(self.model.config, "max_position_embeddings", self.max_model_len)
        
        if seq_len + max_new_tokens > max_len:
            from loguru import logger
            logger.error(f"CRITICAL: Prompt length ({seq_len} tokens) + max_new_tokens ({max_new_tokens}) exceeds model's max context window ({max_len})! This causes silent trimming or OOM.")
            raise ValueError(f"Prompt length {seq_len} exceeds max context limit of {max_len}")

        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens
            )

        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

        return self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
