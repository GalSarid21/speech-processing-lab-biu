from typing import Any
from speech_processing.adapters.base import BaseGenerationAdapter
from speech_processing.config.core import GenerationParams


class TransformersAdapter(BaseGenerationAdapter):
    def __init__(self, model_id: str, dtype: str, max_model_len: int):
        import torch
        from loguru import logger
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        
        logger.info(f"Loading Model from {model_id} via transformers...")
        self.processor = AutoProcessor.from_pretrained(model_id)

        torch_dtype: torch.dtype = getattr(torch, dtype, torch.bfloat16)

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto"
        )
        self.model.eval()
        self.max_model_len = max_model_len

    def generate_batch(
        self, texts: list[str], audios: list[Any], sampling_params: GenerationParams, batch_size: int = 4
    ) -> list[str]:
        import torch
        
        all_outputs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_audios_lists = audios[i:i+batch_size]
            
            flat_audios = []
            for a in batch_audios_lists:
                if isinstance(a, list):
                    flat_audios.extend(a)
                else:
                    flat_audios.append(a)
                    
            inputs = self.processor(
                text=batch_texts, 
                audio=flat_audios, 
                return_tensors="pt", 
                padding=True,
                sampling_rate=self.processor.feature_extractor.sampling_rate
            )
            inputs = inputs.to(self.model.device)
            
            # Dimension 1 represents the sequence length of the tokenized inputs
            seq_len = inputs.input_ids.size(1)
            max_len = getattr(self.model.config, "max_position_embeddings", self.max_model_len)
            
            if seq_len + sampling_params.max_new_tokens > max_len:
                from loguru import logger
                logger.error(f"CRITICAL: Prompt length ({seq_len} tokens) + max_new_tokens ({sampling_params.max_new_tokens}) exceeds model's max context window ({max_len})! This causes silent trimming or OOM.")
                raise ValueError(f"Prompt length {seq_len} exceeds max context limit of {max_len}")

            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=sampling_params.max_new_tokens,
                    do_sample=True,
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p
                )

            generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

            batch_outputs = self.processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            all_outputs.extend(batch_outputs)
            
            del inputs
            del generate_ids
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return all_outputs
