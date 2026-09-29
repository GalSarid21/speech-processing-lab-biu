from loguru import logger

from speech_processing.config.core import TextModelConfig
from speech_processing.data.dtos.requests import TextRequest
from speech_processing.data.dtos.responses import TextResponse
from speech_processing.evaluation.answer_parsing import strip_thinking
from speech_processing.models.base import BaseTextModel

BATCH_SAMPLE_ID = "batch_text"


class GemmaTextModel(BaseTextModel):
    def __init__(self, config: TextModelConfig) -> None:
        # Scoped import because vllm is not installable on macOS and would crash on import
        from vllm import LLM

        self.config = config

        logger.info(f"Loading Text Model from {config.model_id} via vLLM...")

        dtype = config.dtype if config.dtype != "float8" else "auto"

        self.llm = LLM(
            model=config.model_id,
            dtype=dtype,
            max_model_len=config.max_model_len,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def batch_infer(self, requests: list[TextRequest]) -> list[TextResponse]:
        # Scoped import because vllm is not installable on macOS and would crash on import
        from vllm import SamplingParams

        prompts = []
        for req in requests:
            conversation: list[dict[str, str]] = []

            if self.config.enable_thinking:
                conversation.append({"role": "system", "content": "<|think|>"})

            for turn in req.few_shot_turns:
                conversation.append({"role": "user", "content": turn.user_text})
                # Gemma best practice: multi-turn history should only contain the final answer,
                # not the `<|channel>thought...` block.
                conversation.append({"role": "assistant", "content": turn.assistant_text})

            instruction = req.instruction if isinstance(req.instruction, str) else req.instruction[-1]
            conversation.append({"role": "user", "content": instruction})

            prompts.append(self.tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False))

        logger.info(
            f"Evaluating text batch of size {len(requests)} with thinking="
            f"{'ON' if self.config.enable_thinking else 'OFF'}..."
        )

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_new_tokens,
            stop=self.config.stop,
        )

        try:
            outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params)
            generated_texts = [out.outputs[0].text.strip() for out in outputs]
        except RuntimeError as e:
            logger.error(f"vLLM batch generation failed: {e}")
            return []

        responses = []
        for req, raw_text in zip(requests, generated_texts, strict=True):
            output_text = strip_thinking(raw_text)
            responses.append(
                TextResponse(
                    sample_id=req.metadata.item_id if req.metadata else BATCH_SAMPLE_ID,
                    instruction=req.instruction,
                    generated_text=output_text,
                    final_turn_text=output_text,
                    metadata=req.metadata,
                )
            )

        return responses
