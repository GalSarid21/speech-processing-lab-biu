"""Detects answers that switched out of English into a CJK script.

Qwen audio models are known to answer in Chinese on reasoning prompts even when told to answer in English.
Such an answer may still parse to a letter, so the switch would otherwise go unnoticed. Two cases differ:

* an answer *written* in Chinese - a language switch, counted by `is_non_english`;
* an English answer that *quotes* a Chinese word heard in the audio (MMAR has clips with Chinese speech) -
  not a switch, so it only shows up through `contains_cjk`.

The share is CJK characters over CJK characters plus Latin letters, so punctuation, digits and the
answer letter's formatting do not move it.
"""

import re

# CJK punctuation, Hiragana/Katakana, CJK ideographs (incl. extension A), Hangul syllables, full-width forms.
_CJK_SCRIPT = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")

# An English answer quoting a few heard words stays far below this; an answer written in Chinese is far above.
NON_ENGLISH_MIN_CJK_SHARE = 0.2


def contains_cjk(text: str | None) -> bool:
    return bool(text) and _CJK_SCRIPT.search(text) is not None


def cjk_share(text: str | None) -> float:
    if not text:
        return 0.0
    cjk = len(_CJK_SCRIPT.findall(text))
    letters = cjk + len(_LATIN_LETTER.findall(text))
    return cjk / letters if letters else 0.0


def is_non_english(text: str | None) -> bool:
    return cjk_share(text) >= NON_ENGLISH_MIN_CJK_SHARE
