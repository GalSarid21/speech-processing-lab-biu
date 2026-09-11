from typing import Any

import torch

from speech_processing.adapters.base import BaseGenerationAdapter


class TransformersAdapter(BaseGenerationAdapter):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

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

        with torch.no_grad():
            # Explicitly force greedy decoding for determinism
            generate_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None
            )

        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

        return self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
