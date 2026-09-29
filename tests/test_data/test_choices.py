import numpy as np
import pytest
from assertpy import assert_that

from speech_processing.data.choices import (
    answer_to_letter,
    format_lettered_choices,
    match_choice_text,
    normalize_choices,
)
from speech_processing.utils.exceptions import DatasetIntegrityError

CHOICES = ["Dog", "Cat", "Bird"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        (np.array(["Dog", "Cat"]), ["Dog", "Cat"]),
        (["Dog", "Cat"], ["Dog", "Cat"]),
        ("['Dog', 'Cat']", ["Dog", "Cat"]),
    ],
)
def test_normalize_choices_accepts_every_shape(raw, expected):
    assert_that(normalize_choices(raw, "ID1")).is_equal_to(expected)


@pytest.mark.parametrize("raw", ["not a list [", "[]", 42])
def test_normalize_choices_rejects_bad_input(raw):
    assert_that(normalize_choices).raises(DatasetIntegrityError).when_called_with(raw, "ID1")


def test_format_lettered_choices():
    assert_that(format_lettered_choices(CHOICES)).is_equal_to("A. Dog\nB. Cat\nC. Bird")


@pytest.mark.parametrize("answer, expected", [("Cat", "B"), (" cat ", "B"), ("BIRD", "C")])
def test_answer_to_letter_is_tolerant(answer, expected):
    assert_that(answer_to_letter(answer, CHOICES, "ID1")).is_equal_to(expected)


def test_answer_to_letter_raises_when_answer_is_not_a_choice():
    assert_that(answer_to_letter).raises(DatasetIntegrityError).when_called_with("Fish", CHOICES, "ID1")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Cat", "B"),
        ("'Cat'", "B"),
        ('"Cat".', "B"),
        ("Cat.", "B"),
        ("Fish", None),
        ("", None),
    ],
)
def test_match_choice_text(text, expected):
    assert_that(match_choice_text(text, CHOICES)).is_equal_to(expected)
