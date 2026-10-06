import re

from speech_processing.data.choices import letters_for, match_choice_text
from speech_processing.utils.consts import ANSWER_END_TAG, ANSWER_START_TAG

_THINKING_BLOCK_PATTERNS = (
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|channel>thought.*?<channel\|>", re.DOTALL | re.IGNORECASE),
)

_ANSWER_TAG_PATTERN = re.compile(
    rf"{re.escape(ANSWER_START_TAG)}(.*?)(?:{re.escape(ANSWER_END_TAG)}|$)", re.DOTALL | re.IGNORECASE
)

_EXPLICIT_STATEMENT_PATTERN = re.compile(
    r"(?i:final answer|correct (?:answer|choice|option)|answer)\s*(?i:is)?\s*:?\s*[\(\[]?\*{0,2}([A-Z])\b(?![\w'])"
)

_BARE_LETTER_ALONE = re.compile(r"^\s*[\(\[]?\*{0,2}([A-Z])\*{0,2}[\.\):\]]?\s*$")
_BARE_LETTER_PREFIX = re.compile(r"^\s*[\(\[]?\*{0,2}([A-Z])\*{0,2}[\.\):\]]\s+\S")
# "G. Asthma", "**A. COPD**", "(B) Pneumonia": a letter, a delimiter, then text checked against that letter.
_LETTER_WITH_TEXT = re.compile(r"^\s*[\(\[]?\*{0,2}([A-Z])\*{0,2}[\.\):\]]\s*(.+?)\s*$")


def strip_thinking(text: str) -> str:
    """Removes model-specific thinking blocks so only the user-visible answer remains."""
    stripped = text
    for pattern in _THINKING_BLOCK_PATTERNS:
        stripped = pattern.sub("", stripped)
    return stripped.strip()


def _non_empty_lines(text: str) -> list[str]:
    return [line for line in (raw.strip() for raw in text.splitlines()) if line]


def parse_choice_letter(text: str, choices: list[str]) -> str | None:
    """Deterministically extracts the selected choice letter. Returns None when nothing is unambiguous."""
    if not text or not choices:
        return None

    valid_letters = set(letters_for(choices))
    candidate = strip_thinking(text)

    answer_tag = _ANSWER_TAG_PATTERN.search(candidate)
    if answer_tag:
        candidate = answer_tag.group(1).strip()

    explicit = [m for m in _EXPLICIT_STATEMENT_PATTERN.finditer(candidate) if m.group(1) in valid_letters]
    if explicit:
        return explicit[-1].group(1)

    lines = _non_empty_lines(candidate)
    for probe in (candidate, lines[0] if lines else ""):
        letter = match_choice_text(probe, choices)
        if letter:
            return letter

    if not lines:
        return None

    for line in (lines[0], lines[-1]):
        for pattern in (_BARE_LETTER_ALONE, _BARE_LETTER_PREFIX):
            match = pattern.match(line)
            if match and match.group(1) in valid_letters:
                return match.group(1)

    return _single_answer_line(lines, choices, valid_letters)


def _single_answer_line(lines: list[str], choices: list[str], valid_letters: set[str]) -> str | None:
    """The letter named by every answer-shaped line, wherever those lines sit.

    Models often wrap the answer in prose - "The diagnosis is most likely:\n\nG. Asthma\n\nThe presence
    of..." - so it is on neither the first nor the last line. A line counts only if it is a bare letter,
    or a letter followed by that same letter's own choice text. If those lines name more than one
    letter (an enumerated list of the options, say), the answer is ambiguous and stays unparsed.
    """
    named: set[str] = set()
    for line in lines:
        alone = _BARE_LETTER_ALONE.match(line)
        if alone and alone.group(1) in valid_letters:
            named.add(alone.group(1))
            continue
        with_text = _LETTER_WITH_TEXT.match(line)
        if with_text and with_text.group(1) in valid_letters:
            if match_choice_text(with_text.group(2), choices) == with_text.group(1):
                named.add(with_text.group(1))
    return named.pop() if len(named) == 1 else None
