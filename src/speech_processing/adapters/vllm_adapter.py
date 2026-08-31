from typing import Any

from vllm import LLM

from speech_processing.adapters.base import BaseGenerationAdapter


class VLLMAdapter(BaseGenerationAdapter):
    def __init__(self, llm: LLM) -> None:
        self.llm: LLM = llm

    def generate_batch(self, prompts: list[str], sampling_params: Any) -> list[str]:
        # llm.generate returns a list of RequestOutput objects
        outputs: list[Any] = self.llm.generate(prompts, sampling_params=sampling_params)
        return [output.outputs[0].text for output in outputs]
