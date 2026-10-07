from typing import Any

from speech_processing.adapters.base import BaseGenerationAdapter
from speech_processing.config.core import GenerationParams
from speech_processing.utils.consts import CONTRASTIVE_PROMPT_LOGPROBS
from speech_processing.utils.exceptions import AudioProcessingError, ContrastiveScoringError


class VLLMAdapter(BaseGenerationAdapter):
    def __init__(self, **kwargs: Any) -> None:
        # Scoped import because vllm is not installable on macOS and would crash on import
        from vllm import LLM

        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def generate_batch(self, prompts: list[Any], sampling_params: GenerationParams) -> list[str]:
        # Scoped import because vllm is not installable on macOS and would crash on import
        from vllm import SamplingParams

        kwargs: dict[str, Any] = {
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_tokens": sampling_params.max_new_tokens,
        }
        if sampling_params.stop:
            kwargs["stop"] = sampling_params.stop

        if sampling_params.json_schema:
            vllm_params = self._structured_params(SamplingParams, kwargs, sampling_params.json_schema)
        else:
            vllm_params = SamplingParams(**kwargs)

        if prompts and isinstance(prompts[0], list):
            prefill_flags = {p[-1]["role"] == "assistant" for p in prompts}
            if len(prefill_flags) > 1:
                raise AudioProcessingError("A batch cannot mix prefilled and non-prefilled conversations.")
            has_prefill = prefill_flags.pop()

            if has_prefill:
                # vLLM exposes add_generation_prompt as a top-level arg defaulting to True, so it must be
                # passed explicitly alongside continue_final_message.
                outputs: list[Any] = self.llm.chat(
                    messages=prompts,
                    sampling_params=vllm_params,
                    continue_final_message=True,
                    add_generation_prompt=False,
                )
            else:
                outputs = self.llm.chat(messages=prompts, sampling_params=vllm_params)
        else:
            outputs = self.llm.generate(prompts=prompts, sampling_params=vllm_params)

        return [output.outputs[0].text for output in outputs]

    @staticmethod
    def _structured_params(sampling_params_cls: Any, kwargs: dict[str, Any], json_schema: str) -> Any:
        try:
            # 1. Newest API (vLLM >= 0.6.1)
            from vllm.sampling_params import StructuredOutputsParams

            # disable_any_whitespace: the grammar otherwise allows unlimited whitespace between JSON tokens,
            # and a judge can then emit blank lines until max_tokens, leaving the object unclosed.
            return sampling_params_cls(
                **kwargs,
                structured_outputs=StructuredOutputsParams(json=json_schema, disable_any_whitespace=True),
            )
        except (ImportError, TypeError, ValueError):
            try:
                # 2. Recent API (vLLM ~ 0.5.x)
                from vllm.sampling_params import GuidedDecodingParams

                return sampling_params_cls(**kwargs, guided_decoding=GuidedDecodingParams(json=json_schema))
            except (ImportError, TypeError, ValueError):
                # 3. Oldest API (vLLM < 0.5.x)
                return sampling_params_cls(**kwargs, guided_json=json_schema)

    def score_last_prompt_token(self, conversations: list[Any]) -> list[tuple[int, float]]:
        """Returns (token_id, logprob) of the final prompt token of each already-completed conversation."""
        # Scoped import because vllm is not installable on macOS and would crash on import
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=1, prompt_logprobs=CONTRASTIVE_PROMPT_LOGPROBS)
        outputs = self.llm.chat(
            messages=conversations,
            sampling_params=params,
            add_generation_prompt=False,
            continue_final_message=True,
        )

        scored: list[tuple[int, float]] = []
        for output in outputs:
            token_id = output.prompt_token_ids[-1]
            position = output.prompt_logprobs[-1] if output.prompt_logprobs else None
            if not position or token_id not in position:
                raise ContrastiveScoringError(f"vLLM returned no log-prob for the final prompt token {token_id}.")
            scored.append((token_id, position[token_id].logprob))
        return scored

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])
