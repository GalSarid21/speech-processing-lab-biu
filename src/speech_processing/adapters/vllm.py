from typing import Any
from speech_processing.adapters.base import BaseGenerationAdapter
from speech_processing.config.core import GenerationParams


class VLLMAdapter(BaseGenerationAdapter):
    def __init__(self, **kwargs) -> None:
        # Scoped import because vllm is not available on MacOS and would crash on import
        from vllm import LLM
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def generate_batch(self, prompts: list[Any], sampling_params: GenerationParams) -> list[str]:
        # Scoped import because vllm is not available on MacOS and would crash on import
        from vllm import SamplingParams
        
        kwargs = {
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_tokens": sampling_params.max_new_tokens
        }
        if hasattr(sampling_params, "stop") and sampling_params.stop:
            kwargs["stop"] = sampling_params.stop
        
        vllm_params = None
        if sampling_params.json_schema:
            try:
                # 1. Try Newest API (vLLM >= 0.6.1)
                from vllm.sampling_params import StructuredOutputsParams
                vllm_params = SamplingParams(
                    **kwargs, 
                    structured_outputs=StructuredOutputsParams(json=sampling_params.json_schema)
                )
            except (ImportError, TypeError, ValueError):
                try:
                    # 2. Try Recent API (vLLM ~ 0.5.x)
                    from vllm.sampling_params import GuidedDecodingParams
                    vllm_params = SamplingParams(
                        **kwargs, 
                        guided_decoding=GuidedDecodingParams(json=sampling_params.json_schema)
                    )
                except (ImportError, TypeError, ValueError):
                    # 3. Try Oldest API (vLLM < 0.5.x)
                    vllm_params = SamplingParams(
                        **kwargs, 
                        guided_json=sampling_params.json_schema
                    )
        else:
            vllm_params = SamplingParams(**kwargs)
        
        # If the input is a list of lists (i.e. batch of OpenAI message dicts)
        if len(prompts) > 0 and isinstance(prompts[0], list):
            # Check if the final message is an assistant prefill
            has_prefill = any(p[-1]["role"] == "assistant" for p in prompts)
            try:
                if has_prefill:
                    # New vLLM API supports continue_final_message. We MUST pass add_generation_prompt=False 
                    # directly at the top level because vLLM has it as a top-level arg with default=True.
                    outputs: list[Any] = self.llm.chat(
                        messages=prompts, 
                        sampling_params=vllm_params, 
                        continue_final_message=True,
                        add_generation_prompt=False
                    )
                else:
                    outputs: list[Any] = self.llm.chat(messages=prompts, sampling_params=vllm_params)
            except TypeError:
                # Older vLLM APIs fallback (if continue_final_message is not a valid arg)
                if has_prefill:
                    outputs: list[Any] = self.llm.chat(
                        messages=prompts, 
                        sampling_params=vllm_params, 
                        add_generation_prompt=False
                    )
                else:
                    outputs: list[Any] = self.llm.chat(messages=prompts, sampling_params=vllm_params)
        else:
            outputs: list[Any] = self.llm.generate(prompts=prompts, sampling_params=vllm_params)
        return [output.outputs[0].text for output in outputs]
