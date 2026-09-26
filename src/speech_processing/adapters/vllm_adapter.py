from typing import Any
from speech_processing.adapters.base import BaseGenerationAdapter
from speech_processing.config.core import GenerationParams


class VLLMAdapter(BaseGenerationAdapter):
    def __init__(self, **kwargs) -> None:
        from vllm import LLM
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def generate_batch(self, prompts: list[Any], sampling_params: GenerationParams) -> list[str]:
        from vllm import SamplingParams
        
        kwargs = {
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_tokens": sampling_params.max_new_tokens
        }
        
        if sampling_params.json_schema:
            try:
                from vllm.sampling_params import GuidedDecodingParams
                kwargs["guided_decoding"] = GuidedDecodingParams(json=sampling_params.json_schema)
            except ImportError:
                try:
                    from vllm.sampling_params import StructuredOutputsParams
                    kwargs["guided_decoding"] = StructuredOutputsParams(json=sampling_params.json_schema)
                except ImportError:
                    pass
                    
        vllm_params = SamplingParams(**kwargs)
        
        # If the input is a list of lists (i.e. batch of OpenAI message dicts)
        if len(prompts) > 0 and isinstance(prompts[0], list):
            outputs: list[Any] = self.llm.chat(messages=prompts, sampling_params=vllm_params)
        else:
            outputs: list[Any] = self.llm.generate(prompts=prompts, sampling_params=vllm_params)
        return [output.outputs[0].text for output in outputs]
