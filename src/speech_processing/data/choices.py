import ast
import random
import re
from collections.abc import Iterable
from typing import Any

from speech_processing.utils.consts import CHOICE_LETTERS, SHUFFLE_BASE_SEED
from speech_processing.utils.exceptions import DatasetIntegrityError

_WHITESPACE_PATTERN = re.compile(r"\s+")
_SURROUNDING_NOISE = "\"'`.*"


def _canonical(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", text).strip().casefold()


def normalize_choices(raw: Any, item_id: str) -> list[str]:
    """Coerces a raw MMAR `choices` cell (numpy array, list, or stringified list) into a list of strings."""
    value = raw
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as e:
            raise DatasetIntegrityError(f"Item {item_id}: choices is not a parsable list: {raw!r}") from e

    if isinstance(value, str) or not isinstance(value, Iterable):
        raise DatasetIntegrityError(f"Item {item_id}: choices is not iterable: {raw!r}")

    choices = [str(choice) for choice in value]
    if not choices:
        raise DatasetIntegrityError(f"Item {item_id}: choices is empty.")
    if len(choices) > len(CHOICE_LETTERS):
        raise DatasetIntegrityError(
            f"Item {item_id}: {len(choices)} choices exceed the {len(CHOICE_LETTERS)} available letters."
        )
    return choices


def letter_for_index(index: int) -> str:
    return CHOICE_LETTERS[index]


def index_for_letter(letter: str) -> int:
    return CHOICE_LETTERS.index(letter)


def letters_for(choices: list[str]) -> list[str]:
    return [letter_for_index(i) for i in range(len(choices))]


def format_lettered_choices(choices: list[str]) -> str:
    return "\n".join(f"{letter_for_index(i)}. {choice}" for i, choice in enumerate(choices))


def answer_to_letter(answer: str, choices: list[str], item_id: str) -> str:
    """Maps a gold answer string onto its choice letter, tolerating case and whitespace differences."""
    target = _canonical(str(answer))
    for i, choice in enumerate(choices):
        if _canonical(choice) == target:
            return letter_for_index(i)
    raise DatasetIntegrityError(f"Item {item_id}: answer {answer!r} is not among the choices {choices!r}.")


def match_choice_text(text: str, choices: list[str]) -> str | None:
    """Returns the letter of the single choice whose text equals `text`, ignoring quotes, periods and case."""
    candidate = _canonical(text.strip().strip(_SURROUNDING_NOISE))
    if not candidate:
        return None

    matches = [i for i, choice in enumerate(choices) if _canonical(choice) == candidate]
    if len(matches) != 1:
        return None
    return letter_for_index(matches[0])


def shuffle_permutations(item_id: str, n_choices: int, n_variants: int) -> list[list[int]]:
    """Deterministic choice-order variants. Variant 0 is always the dataset's own order.

    `permutation[displayed_index] = original_index`, and the seed is derived from the item id so a
    rerun reproduces exactly the same variants.
    """
    identity = list(range(n_choices))
    permutations = [identity]
    for variant in range(1, n_variants):
        rng = random.Random(f"{SHUFFLE_BASE_SEED}:{item_id}:{variant}")
        permutations.append(rng.sample(identity, n_choices))
    return permutations
