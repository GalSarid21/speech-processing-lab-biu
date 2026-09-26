import json
from collections.abc import Callable

from loguru import logger


from speech_processing.adapters.vllm_adapter import VLLMAdapter
from speech_processing.config.core import JudgeConfig, GenerationParams
from speech_processing.data.dtos import EvaluationResult, JudgeRequest, JudgeResponse
from speech_processing.models.base import BaseJudgeModel


class QwenJudge(BaseJudgeModel):
    def __init__(self, config: JudgeConfig, template_func: Callable[[JudgeRequest], list[dict[str, str]]]) -> None:
        self.config = config
        self.template_func = template_func

        logger.info(f"Loading Judge Model from {config.model_id} via vLLM...")

        # Determine proper dtype for vLLM
        dtype = config.dtype if config.dtype != "float8" else "auto"

        self.adapter = VLLMAdapter(
            model=config.model_id,
            dtype=dtype,
            max_model_len=config.max_model_len,
            trust_remote_code=True,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_seqs=config.max_num_seqs,
        )
        self.tokenizer = self.adapter.tokenizer

    def batch_evaluate(self, requests: list[JudgeRequest]) -> list[JudgeResponse]:
        prompts = []
        for req in requests:
            conversation = self.template_func(req)
            prompt_str = self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt_str)

        logger.info(
            f"Evaluating batch of size {len(requests)} with guided JSON decoding..."
        )

        schema_str = json.dumps(EvaluationResult.model_json_schema())
        sampling_params = GenerationParams(
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=self.config.max_new_tokens,
            json_schema=schema_str
        )

        try:
            generated_texts = self.adapter.generate_batch(
                prompts=prompts, sampling_params=sampling_params
            )
        except RuntimeError as e:
            logger.error(f"vLLM batch generation failed: {e}")
            return []

        responses = []
        for req, output_text in zip(requests, generated_texts):
            try:
                evaluation = EvaluationResult.model_validate_json(output_text)
            except ValueError as e:
                logger.error(
                    f"Failed to parse JSON for request. Raw output: {output_text}. Error: {e}"
                )
                evaluation = EvaluationResult(
                    reasoning="Parse failed.",
                    acoustic_accuracy=0,
                    diagnostic_accuracy=0,
                    hallucination_penalty=1,
                    extracted_class="Unknown"
                )

            responses.append(JudgeResponse(
                sample_id=req.sample_id,
                instruction=req.instruction,
                generated_text=req.generated_text,
                ground_truth=req.ground_truth,
                evaluation=evaluation
            ))

        return responses
