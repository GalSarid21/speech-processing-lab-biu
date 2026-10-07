import pytest
from assertpy import assert_that

from speech_processing.evaluation.language import contains_cjk, is_non_english

ENGLISH = "Based on the audio, the speaker sounds surprised.\n\nB. Surprise"
QUOTES_CHINESE = 'Based on the audio, the streamer is repeatedly saying "速度" which translates to "Speed". C. Speed'
CHINESE = "根据音频内容 说话者听起来很惊讶。答案是 B"
JAPANESE = "音声によると、答えは B です"
KOREAN = "오디오에 따르면 정답은 B 입니다"


@pytest.mark.parametrize(
    "text, has_cjk, switched",
    [
        (ENGLISH, False, False),
        (QUOTES_CHINESE, True, False),
        (CHINESE, True, True),
        (JAPANESE, True, True),
        (KOREAN, True, True),
        ("", False, False),
        (None, False, False),
    ],
    ids=["english", "english-quoting-chinese", "chinese", "japanese", "korean", "empty", "none"],
)
def test_language_switch_is_told_apart_from_a_quote(text, has_cjk, switched):
    assert_that(contains_cjk(text)).is_equal_to(has_cjk)
    assert_that(is_non_english(text)).is_equal_to(switched)
