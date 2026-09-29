import pytest
from assertpy import assert_that

from speech_processing.data.dtos import AudioRequest, ItemMetadata
from speech_processing.models.contrastive import ContrastiveChoiceScorer
from speech_processing.utils import audio as audio_utils
from speech_processing.utils.exceptions import ContrastiveScoringError

TARGET_SR = 16_000
CHOICES = ["Dog", "Cat", "Bird"]
SILENCE_MARKER = "silence_"

REAL_LOGPROBS = {"A": -0.5, "B": -1.0, "C": -3.0}
SILENCE_LOGPROBS = {"A": -0.2, "B": -2.5, "C": -3.0}
TOKEN_IDS = {"A": 10, "B": 11, "C": 12}


class FakeBackend:
    """Returns log-probs keyed by (is_silence, letter); silence is detected from the audio URL."""

    def __init__(self, decoded_override: str | None = None) -> None:
        self.calls = 0
        self.decoded_override = decoded_override

    def score_last_prompt_token(self, conversations):
        self.calls += 1
        scored = []
        for conversation in conversations:
            letter = conversation[-1]["content"]
            url = conversation[0]["content"][0]["audio_url"]["url"]
            table = SILENCE_LOGPROBS if SILENCE_MARKER in url else REAL_LOGPROBS
            scored.append((TOKEN_IDS[letter], table[letter]))
        return scored

    def decode_token(self, token_id: int) -> str:
        if self.decoded_override is not None:
            return self.decoded_override
        return {v: k for k, v in TOKEN_IDS.items()}[token_id]


@pytest.fixture(autouse=True)
def mono_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_utils, "MONO_AUDIO_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def request_with_choices(wav_factory):
    return AudioRequest(
        instruction="Answer.",
        audio_path=wav_factory(2.0, "real.wav"),
        metadata=ItemMetadata(item_id="ID1", question="q", choices=CHOICES),
    )


@pytest.mark.parametrize("alpha, expected", [(0.0, "A"), (1.0, "B")])
def test_alpha_selects_the_expected_letter(request_with_choices, alpha, expected):
    backend = FakeBackend()
    responses = ContrastiveChoiceScorer(alpha, TARGET_SR, batch_size=4).score(backend, [request_with_choices])

    assert_that(responses).is_length(1)
    assert_that(responses[0].generated_text).is_equal_to(expected)
    assert_that(responses[0].final_turn_text).is_equal_to(expected)
    assert_that(responses[0].metadata.choice_scores).is_length(len(CHOICES))
    assert_that(backend.calls).is_equal_to(1)


def test_decode_mismatch_raises(request_with_choices):
    scorer = ContrastiveChoiceScorer(0.5, TARGET_SR, batch_size=4)
    assert_that(scorer.score).raises(ContrastiveScoringError).when_called_with(
        FakeBackend(decoded_override="</s>"), [request_with_choices]
    )
