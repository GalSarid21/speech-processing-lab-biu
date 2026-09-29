import pytest
from assertpy import assert_that

from speech_processing.adapters.vllm import VLLMAdapter
from speech_processing.config.core import GenerationParams
from speech_processing.utils.exceptions import AudioProcessingError, ContrastiveScoringError

PROMPT_TOKEN_ID = 7
TOP_TOKEN_ID = 99


def build_adapter(mocker, llm=None) -> VLLMAdapter:
    adapter = VLLMAdapter.__new__(VLLMAdapter)
    adapter.llm = llm if llm is not None else mocker.MagicMock()
    adapter.tokenizer = mocker.MagicMock()
    return adapter


def scoring_output(mocker, position: dict | None, prompt_token_ids=(1, PROMPT_TOKEN_ID)):
    output = mocker.MagicMock()
    output.prompt_token_ids = list(prompt_token_ids)
    output.prompt_logprobs = [None, position] if position is not None else None
    return output


def logprob(mocker, value: float):
    entry = mocker.MagicMock()
    entry.logprob = value
    return entry


def test_score_last_prompt_token_indexes_by_the_actual_token(mocker):
    position = {TOP_TOKEN_ID: logprob(mocker, -0.1), PROMPT_TOKEN_ID: logprob(mocker, -2.3)}
    llm = mocker.MagicMock()
    llm.chat.return_value = [scoring_output(mocker, position)]
    adapter = build_adapter(mocker, llm)

    assert_that(adapter.score_last_prompt_token([[{"role": "user", "content": "x"}]])).is_equal_to(
        [(PROMPT_TOKEN_ID, -2.3)]
    )
    assert_that(llm.chat.call_args.kwargs["continue_final_message"]).is_true()
    assert_that(llm.chat.call_args.kwargs["add_generation_prompt"]).is_false()


@pytest.mark.parametrize("position", [None, {TOP_TOKEN_ID: -0.1}])
def test_missing_logprob_raises(mocker, position):
    llm = mocker.MagicMock()
    llm.chat.return_value = [scoring_output(mocker, position)]
    adapter = build_adapter(mocker, llm)

    assert_that(adapter.score_last_prompt_token).raises(ContrastiveScoringError).when_called_with(
        [[{"role": "user", "content": "x"}]]
    )


def test_mixed_prefill_batch_raises(mocker):
    adapter = build_adapter(mocker)
    prompts = [
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "prefill"}],
        [{"role": "user", "content": "b"}],
    ]

    assert_that(adapter.generate_batch).raises(AudioProcessingError).when_called_with(
        prompts, GenerationParams(temperature=0.5, top_p=0.5, max_new_tokens=8)
    )


@pytest.mark.parametrize(
    "prompts, expects_continue",
    [
        ([[{"role": "user", "content": "a"}, {"role": "assistant", "content": "p"}]], True),
        ([[{"role": "user", "content": "a"}]], False),
    ],
)
def test_continue_final_message_is_set_only_for_prefills(mocker, prompts, expects_continue):
    llm = mocker.MagicMock()
    llm.chat.return_value = []
    adapter = build_adapter(mocker, llm)

    adapter.generate_batch(prompts, GenerationParams(temperature=0.5, top_p=0.5, max_new_tokens=8))

    assert_that("continue_final_message" in llm.chat.call_args.kwargs).is_equal_to(expects_continue)
