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
        ("<think>cut off while reasoning about B", ""),
        ("<think>done</think>B. Cat<think>cut off", "B. Cat"),
    ],
)
def test_strip_thinking(text, expected):
    assert_that(strip_thinking(text)).is_equal_to(expected)


ICBHI = [
    "COPD",
    "Pneumonia",
    "URTI",
    "LRTI",
    "Bronchiectasis",
    "Bronchiolitis",
    "Asthma",
    "No potential disease detected",
]


@pytest.mark.parametrize(
    "text, expected",
    [
        # Verbatim Voxtral answers from the c4 acceptance run, cut off at the token limit.
        (
            "Based on the provided auscultation signatures and the measured respiratory features, the patient's "
            "diagnosis is most likely:\n\nG. Asthma\n\nThe presence of poly",
            "G",
        ),
        (
            "Based on the auscultation signatures and the measured respiratory features, the patient's diagnosis "
            "is most likely:\n\n**A. COPD**\n\nHere's why:\n\n-",
            "A",
        ),
        ("The most likely diagnosis is:\n\n(B) Pneumonia\n\nbecause", "B"),
        ("After listening, my choice:\n\nH\n\nThe recording sounds clean", "H"),
        # The same answer line twice is still one answer.
        ("Diagnosis:\nA. COPD\nTo confirm:\nA. COPD\nEnd", "A"),
        # A restated option list names every letter: ambiguous, so it must stay unparsed.
        ("The options were:\nA. COPD\nB. Pneumonia\nC. URTI\nI am not sure.", None),
        # A letter paired with the wrong choice text is not trusted from the middle of the text.
        ("It sounds like:\nB. Asthma\nfrom the wheezes", None),
    ],
)
def test_answer_line_in_the_middle_of_prose(text, expected):
    assert_that(parse_choice_letter(text, ICBHI)).is_equal_to(expected)


def test_cut_off_reasoning_is_unparsed_not_mined_for_a_letter():
    assert_that(parse_choice_letter("<think>\nThe answer is B, but wait", CHOICES)).is_none()
    assert_that(parse_choice_letter("<think>\nmaybe A</think>\nB. Stuttering", CHOICES)).is_equal_to("B")
