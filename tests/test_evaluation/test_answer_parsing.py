import pytest
from assertpy import assert_that

from speech_processing.evaluation.answer_parsing import parse_choice_letter, strip_thinking

CHOICES = ["A dog barks", "Stuttering", "Speech is too fast", "the first speaker"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("B", "B"),
        ("B.", "B"),
        ("(C)", "C"),
        ("**B**", "B"),
        ("B. Stuttering", "B"),
        ("Stuttering", "B"),
        ("'Stuttering'", "B"),
        ("A dog barks", "A"),
        ("The answer is C.", "C"),
        ("First, B looks wrong.\nFinal answer: D", "D"),
        ("<analysis>It could be A or B.</analysis><answer>C</answer>", "C"),
        ("<think>maybe A</think>B", "B"),
        ("reasoning...\nD", "D"),
        ("I think it is the first speaker", None),
        ("E", None),
        ("", None),
    ],
)
def test_parse_choice_letter(text, expected):
    assert_that(parse_choice_letter(text, CHOICES)).is_equal_to(expected)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("<think>hidden</think>visible", "visible"),
        ("<|channel>thought about it<channel|>visible", "visible"),
        ("plain", "plain"),
    ],
)
def test_strip_thinking(text, expected):
    assert_that(strip_thinking(text)).is_equal_to(expected)
