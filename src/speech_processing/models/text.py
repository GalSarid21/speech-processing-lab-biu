from loguru import logger
from vllm import LLM, SamplingParams

from speech_processing.config.core import TextModelConfig
from speech_processing.data.dtos.requests import TextRequest
from speech_processing.data.dtos.responses import TextResponse
from speech_processing.models.base import BaseTextModel


class GemmaTextModel(BaseTextModel):
    def __init__(self, config: TextModelConfig) -> None:
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
        prompts = []
        for req in requests:
            conversation = []
            
            if self.config.enable_thinking:
                conversation.append({"role": "system", "content": "<|think|>"})
            
            # 1. Add Few-Shot Turns
            for turn in req.few_shot_turns:
                conversation.append({"role": "user", "content": turn.user_text})
                # Gemma best practice: multi-turn history should only contain the final answer, not the `<|channel>thought...` block
                conversation.append({"role": "assistant", "content": turn.assistant_text})
                
            # 2. Add Target Instruction
            instruction = req.instruction if isinstance(req.instruction, str) else req.instruction[-1]
            conversation.append({"role": "user", "content": instruction})
            
            prompt_str = self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt_str)

        logger.info(f"Evaluating text batch of size {len(requests)} with thinking={'ON' if self.config.enable_thinking else 'OFF'}...")

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_new_tokens,
        )

        try:
            outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params)
            generated_texts = [out.outputs[0].text.strip() for out in outputs]
        except Exception as e:
            logger.error(f"vLLM batch generation failed: {e}")
            return []

        responses = []
        for req, output_text in zip(requests, generated_texts):
            responses.append(
                TextResponse(
                    sample_id="batch_text", # Sample ID should ideally come from req if we add it
                    instruction=req.instruction,
                    generated_text=output_text
                )
            )

        return responses
