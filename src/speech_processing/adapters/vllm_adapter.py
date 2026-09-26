from typing import Any

from vllm import LLM

from speech_processing.adapters.base import BaseGenerationAdapter


class VLLMAdapter(BaseGenerationAdapter):
    def __init__(self, llm: LLM) -> None:
        self.llm: LLM = llm

    def generate_batch(self, prompts: list[Any], sampling_params: Any) -> list[str]:
        # llm.generate accepts either a list of strings or a list of dictionaries (for multimodal)
        outputs: list[Any] = self.llm.generate(prompts, sampling_params=sampling_params)
        return [output.outputs[0].text for output in outputs]
