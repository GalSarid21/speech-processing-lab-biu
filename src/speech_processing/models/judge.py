import json

from loguru import logger
from vllm import LLM, SamplingParams

from speech_processing.adapters.vllm_adapter import VLLMAdapter
from speech_processing.config.core import JudgeConfig
from speech_processing.data.dtos import EvaluationResult, JudgeRequest, JudgeResponse
from speech_processing.models.base import BaseJudgeModel
from speech_processing.prompts.base import BasePromptTemplate


class QwenJudge(BaseJudgeModel):
    def __init__(self, config: JudgeConfig, template: BasePromptTemplate) -> None:
        self.config = config
        self.template = template

        logger.info(f"Loading Judge Model from {config.model_id} via vLLM...")

        # Determine proper dtype for vLLM
        dtype = config.dtype if config.dtype != "float8" else "auto"

        self.llm = LLM(
            model=config.model_id,
            dtype=dtype,
            max_model_len=config.max_model_len,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.adapter = VLLMAdapter(self.llm)

    def batch_evaluate(self, requests: list[JudgeRequest]) -> list[JudgeResponse]:
        prompts = []
        for req in requests:
            conversation = self.template.build_conversation(req)
            prompt_str = self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt_str)

        logger.info(
            f"Evaluating batch of size {len(requests)} with guided JSON decoding..."
        )

        schema_str = json.dumps(EvaluationResult.model_json_schema())
        guided_kwargs = {}
        try:
            from vllm.sampling_params import GuidedDecodingParams
            guided_kwargs["guided_decoding"] = GuidedDecodingParams(json=schema_str)
        except ImportError:
            try:
                from vllm.sampling_params import StructuredOutputsParams
                guided_kwargs["structured_outputs"] = StructuredOutputsParams(json=schema_str)
            except ImportError:
                guided_kwargs["guided_json"] = schema_str

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.config.max_new_tokens,
            **guided_kwargs
        )

        try:
            generated_texts = self.adapter.generate_batch(
                prompts=prompts, sampling_params=sampling_params
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"vLLM batch generation failed: {e}")
            return []

        responses = []
        for req, output_text in zip(requests, generated_texts):
            try:
                evaluation = EvaluationResult.model_validate_json(output_text)
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"Failed to parse JSON for request. Raw output: {output_text}. Error: {e}"
                )
                evaluation = EvaluationResult(
                    reasoning="Parse failed.",
                    acoustic_accuracy=0,
                    diagnostic_accuracy=0,
                    hallucination_penalty=1,
                    extracted_disease_class="Unknown"
                )

            responses.append(JudgeResponse(request=req, evaluation=evaluation))

        return responses
