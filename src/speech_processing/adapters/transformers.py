from typing import Any

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2AudioForConditionalGeneration

from speech_processing.adapters.base import BaseGenerationAdapter
from speech_processing.config.core import GenerationParams
from speech_processing.utils.consts import MODEL_DEVICE_DTYPE_MAPPING

OVIS_EMBEDDING_MARKER = "Ovis-Omni-Embedding"
DEFAULT_GENERATION_BATCH_SIZE = 4


class TransformersAdapter(BaseGenerationAdapter):
    def __init__(self, model_id: str, dtype: str, max_model_len: int) -> None:
        logger.info(f"Loading Model from {model_id} via transformers...")
        self.processor = AutoProcessor.from_pretrained(model_id)

        torch_dtype: torch.dtype = getattr(torch, dtype, torch.bfloat16)

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto"
        )
        self.model.eval()
        self.max_model_len = max_model_len

    def generate_batch(
        self,
        texts: list[str],
        audios: list[Any],
        sampling_params: GenerationParams,
        batch_size: int = DEFAULT_GENERATION_BATCH_SIZE,
    ) -> list[str]:
        all_outputs: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_audios_lists = audios[i : i + batch_size]

            flat_audios: list[Any] = []
            for item in batch_audios_lists:
                if isinstance(item, list):
                    flat_audios.extend(item)
                else:
                    flat_audios.append(item)

            inputs = self.processor(
                text=batch_texts,
                audio=flat_audios,
                return_tensors="pt",
                padding=True,
                sampling_rate=self.processor.feature_extractor.sampling_rate,
            )
            inputs = inputs.to(self.model.device)

            # Dimension 1 represents the sequence length of the tokenized inputs
            seq_len = inputs.input_ids.size(1)
            max_len = getattr(self.model.config, "max_position_embeddings", self.max_model_len)

            if seq_len + sampling_params.max_new_tokens > max_len:
                logger.error(
                    f"CRITICAL: Prompt length ({seq_len} tokens) + max_new_tokens "
                    f"({sampling_params.max_new_tokens}) exceeds the model's max context window ({max_len})! "
                    "This causes silent trimming or OOM."
                )
                raise ValueError(f"Prompt length {seq_len} exceeds max context limit of {max_len}")

            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=sampling_params.max_new_tokens,
                    do_sample=True,
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                )

            generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

            all_outputs.extend(
                self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            )

            del inputs
            del generate_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_outputs


class TransformersEmbeddingAdapter:
    def __init__(self, model_id: str) -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        torch_dtype = MODEL_DEVICE_DTYPE_MAPPING.get(self.device, torch.float32)

        logger.info(f"Loading Embedding Model {model_id} onto {self.device} via transformers...")
        # Some models (like Ovis) store everything in a 'model' subfolder
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if OVIS_EMBEDDING_MARKER in model_id:
            kwargs["subfolder"] = "model"

        self.processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=self.device, **kwargs
        )
        self.model.eval()

    def embed_audio(self, audio_array) -> Any:
        inputs = self.processor(audios=[audio_array], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        if getattr(outputs, "hidden_states", None) is not None:
            return outputs.hidden_states[-1][:, -1, :].clone().cpu()
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, -1, :].clone().cpu()
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output.clone().cpu()
        if isinstance(outputs, tuple):
            return outputs[0][:, -1, :].clone().cpu()

        logger.error("Could not parse output embedding format from model.")
        raise ValueError("Unexpected model output format.")
