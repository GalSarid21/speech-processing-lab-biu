import json

import pandas as pd
import pytest
from assertpy import assert_that

from speech_processing.config.core import ExperimentMeta
from speech_processing.data.dataset import (
    MAX_IDENTICAL_TRANSCRIPT_SHARE,
    MIN_TRANSCRIPTS_FOR_DEGENERACY_CHECK,
    MMAR_EXCLUDED_ITEMS,
    MMAR_GOLD_CORRECTIONS,
    get_mmar_few_shot_turns,
    load_mmar_requests,
    validate_few_shot_disjointness,
)
from speech_processing.data.dtos import AudioRequest, TextRequest
from speech_processing.utils.consts import NO_FEATURES_PLACEHOLDER
from speech_processing.utils.exceptions import DatasetIntegrityError, FewShotLeakageError
from tests.conftest import MMAR_ROWS, SHOT_ID

BASE_META_FIELDS = {
    "experiment_name": "test",
    "prompt": "Answer the question.",
    "max_new_tokens": 64,
    "batch_size": 2,
}


def meta(**overrides) -> ExperimentMeta:
    return ExperimentMeta(**(BASE_META_FIELDS | overrides))


# --------------------------------------------------------------------------------------
# MMAR: rendering
# --------------------------------------------------------------------------------------


def test_mmar_requests_are_lettered_and_joined(patched_mmar, mmar_config):
    items = load_mmar_requests(mmar_config, meta())

    assert_that([gt for _, gt in items]).is_equal_to(["B", "B"])

    first = items[0][0]
    assert_that(first).is_instance_of(AudioRequest)
    assert_that(first.audio_path).is_equal_to("data/MMAR/audio/id1.wav")
    assert_that(first.instruction).contains("Choices:\nA. the first speaker\nB. the second speaker")
    assert_that(first.metadata.category).is_equal_to("Speaker")
    assert_that(first.metadata.sub_category).is_equal_to("Order")


@pytest.mark.parametrize(
    "overrides, present, absent",
    [
        ({}, ["Question: Who speaks first?"], ["Transcript:", "Diarized transcript:", "Measured acoustic"]),
        ({"use_transcript": True}, ["Transcript: hello there"], ["Diarized transcript:", "Measured acoustic"]),
        (
            {"inject_diarized_transcript": True},
            ["Diarized transcript:", "line for ID1"],
            ["Transcript: hello there", "Measured acoustic"],
        ),
        (
            {"inject_acoustic_features": True},
            ["Measured acoustic features:", "F0 mean 180 Hz"],
            ["Transcript: hello there", "Diarized transcript:"],
        ),
    ],
)
def test_mmar_context_injection(patched_mmar, mmar_config, overrides, present, absent):
    instruction = load_mmar_requests(mmar_config, meta(**overrides))[0][0].instruction

    for fragment in present:
        assert_that(instruction).contains(fragment)
    for fragment in absent:
        assert_that(instruction).does_not_contain(fragment)


def test_missing_features_file_raises(patched_mmar, mmar_config, tmp_path):
    config = mmar_config.model_copy(update={"acoustic_features_file": str(tmp_path / "nope.jsonl")})
    assert_that(load_mmar_requests).raises(DatasetIntegrityError).when_called_with(
        config, meta(inject_acoustic_features=True)
    )


def test_item_without_evidence_gets_placeholder(patched_mmar, mmar_config, tmp_path):
    partial = tmp_path / "partial.jsonl"
    partial.write_text(json.dumps({"item_id": "ID1", "diarized_transcript": "d", "acoustic_features": "f"}) + "\n")
    config = mmar_config.model_copy(update={"acoustic_features_file": str(partial)})

    items = load_mmar_requests(config, meta(inject_acoustic_features=True))
    assert_that(items[1][0].instruction).contains(NO_FEATURES_PLACEHOLDER)


def test_text_only_builds_text_request_with_transcript(patched_mmar, mmar_config):
    text_meta = meta(text_only=True, use_transcript=True)
    req = load_mmar_requests(mmar_config, text_meta, is_text_only=True)[0][0]

    assert_that(req).is_instance_of(TextRequest)
    assert_that(req.instruction).contains("Transcript: hello there")


def test_two_pass_puts_the_question_in_both_turns(patched_mmar, mmar_config):
    two_pass = meta(prompt=["Locate the evidence span.", "Now answer."], presentation="two_pass_localization")
    instruction = load_mmar_requests(mmar_config, two_pass)[0][0].instruction

    assert_that(instruction).is_length(2)
    assert_that(instruction[0]).contains("Who speaks first?")
    assert_that(instruction[1]).contains("Who speaks first?")
    assert_that(instruction[0]).does_not_contain("Choices:")
    assert_that(instruction[1]).contains("Choices:")


def test_shuffled_variants_are_seeded_and_consistent(patched_mmar, mmar_config):
    config = mmar_config.model_copy(update={"sample_ids": ["ID2"]})
    shuffled = meta(num_shuffled_variants=5)

    items = load_mmar_requests(config, shuffled)
    assert_that(items).is_length(5)

    for variant_idx, (req, ground_truth) in enumerate(items):
        metadata = req.metadata
        assert_that(metadata.variant_idx).is_equal_to(variant_idx)
        assert_that([metadata.original_choices[i] for i in metadata.permutation]).is_equal_to(metadata.choices)
        assert_that(metadata.choices[ord(ground_truth) - ord("A")]).is_equal_to("American")

    assert_that(items[0][0].metadata.permutation).is_equal_to([0, 1, 2])

    repeated = load_mmar_requests(config, shuffled)
    assert_that([r.metadata.permutation for r, _ in repeated]).is_equal_to([r.metadata.permutation for r, _ in items])


def test_answer_not_among_choices_raises(mocker, mmar_config):
    broken = [dict(MMAR_ROWS[0]) | {"answer": "nobody"}]
    mocker.patch("speech_processing.data.dataset._load_mmar_frame", return_value=pd.DataFrame(broken))
    mocker.patch("speech_processing.data.dataset._load_transcripts", return_value={})

    config = mmar_config.model_copy(update={"sample_ids": ["ID1"]})
    assert_that(load_mmar_requests).raises(DatasetIntegrityError).when_called_with(config, meta())


def _patch_frame(mocker, rows):
    mocker.patch("speech_processing.data.dataset._load_mmar_frame", return_value=pd.DataFrame(rows))
    mocker.patch("speech_processing.data.dataset._load_transcripts", return_value={})


def test_known_gold_error_is_corrected(mocker, mmar_config):
    item_id, full_choice = next(iter(MMAR_GOLD_CORRECTIONS.items()))
    row = dict(MMAR_ROWS[0]) | {"id": item_id, "choices": ["Other", full_choice], "answer": "abbreviated gold"}
    _patch_frame(mocker, [row])

    config = mmar_config.model_copy(update={"sample_ids": [item_id]})
    [(_, ground_truth)] = load_mmar_requests(config, meta())

    assert_that(ground_truth).is_equal_to("B")


def test_known_unscoreable_item_is_excluded(mocker, mmar_config):
    excluded_id = next(iter(MMAR_EXCLUDED_ITEMS))
    broken = dict(MMAR_ROWS[0]) | {"id": excluded_id, "answer": "both\nchoices"}
    _patch_frame(mocker, [broken, MMAR_ROWS[0]])

    config = mmar_config.model_copy(update={"sample_ids": [excluded_id, MMAR_ROWS[0]["id"]]})
    items = load_mmar_requests(config, meta())

    assert_that([request.metadata.item_id for request, _ in items]).is_equal_to([MMAR_ROWS[0]["id"]])


def _transcript_config(mmar_config, tmp_path, transcripts: dict[str, str] | None):
    path = tmp_path / "whisper_transcripts_under_test.json"
    if transcripts is not None:
        path.write_text(json.dumps(transcripts))
    return mmar_config.model_copy(update={"transcripts_file": str(path)})


def _degenerate_transcripts() -> dict[str, str]:
    """More than the allowed share of entries are the same stock phrase Whisper emits for silence."""
    total = MIN_TRANSCRIPTS_FOR_DEGENERACY_CHECK * 2
    num_identical = int(total * MAX_IDENTICAL_TRANSCRIPT_SHARE) + 1
    return {f"ID{i}": "you" if i < num_identical else f"real speech {i}" for i in range(total)}


@pytest.mark.parametrize("transcripts", [None, _degenerate_transcripts()], ids=["missing", "degenerate"])
def test_transcript_arm_rejects_unusable_transcripts(mocker, mmar_config, tmp_path, transcripts):
    mocker.patch("speech_processing.data.dataset._load_mmar_frame", return_value=pd.DataFrame(MMAR_ROWS))
    config = _transcript_config(mmar_config, tmp_path, transcripts)

    assert_that(load_mmar_requests).raises(DatasetIntegrityError).when_called_with(config, meta(use_transcript=True))


def test_arm_without_transcripts_ignores_the_transcript_file(mocker, mmar_config, tmp_path):
    mocker.patch("speech_processing.data.dataset._load_mmar_frame", return_value=pd.DataFrame(MMAR_ROWS))
    config = _transcript_config(mmar_config, tmp_path, _degenerate_transcripts())

    assert_that(load_mmar_requests(config, meta())).is_not_empty()


# --------------------------------------------------------------------------------------
# MMAR: few-shot
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, expect_audio, answer_prefix, expect_transcript",
    [
        ({"few_shot_mode": "audio"}, True, "B", False),
        ({"few_shot_mode": "text"}, False, "B", False),
        (
            {
                "few_shot_mode": "audio",
                "use_cot": True,
                "use_transcript": True,
                "few_shot_include_transcript": True,
            },
            True,
            "<analysis>",
            True,
        ),
    ],
)
def test_few_shot_turns(patched_mmar, mmar_config, overrides, expect_audio, answer_prefix, expect_transcript):
    shot_ids, turns = get_mmar_few_shot_turns(mmar_config, meta(**overrides))

    assert_that(shot_ids).is_equal_to([SHOT_ID])
    assert_that(turns).is_length(1)
    assert_that(turns[0].audio_path is not None).is_equal_to(expect_audio)
    assert_that(turns[0].assistant_text).starts_with(answer_prefix)
    assert_that("Transcript: oh really" in turns[0].user_text).is_equal_to(expect_transcript)


def test_text_few_shots_keep_audio_on_the_eval_requests(patched_mmar, mmar_config):
    text_shots = meta(few_shot_mode="text")
    _, turns = get_mmar_few_shot_turns(mmar_config, text_shots)
    items = load_mmar_requests(mmar_config, text_shots)

    assert_that(turns[0].audio_path).is_none()
    assert_that(items[0][0].audio_path).is_equal_to("data/MMAR/audio/id1.wav")


@pytest.mark.parametrize(
    "eval_ids, shot_ids, rag_mapping",
    [
        ({"ID1"}, {"ID1"}, None),  # a shot is also an eval item
        ({"ID1"}, {"SHOT1"}, {"ID1": ["ID1"]}),  # RAG retrieves the query itself
        ({"ID1", "ID2"}, {"SHOT1"}, {"ID1": ["ID2"]}),  # RAG retrieves another eval item
    ],
)
def test_disjointness_violations_raise(eval_ids, shot_ids, rag_mapping):
    assert_that(validate_few_shot_disjointness).raises(FewShotLeakageError).when_called_with(
        eval_ids, shot_ids, rag_mapping
    )


def test_disjointness_passes_when_clean():
    validate_few_shot_disjointness({"ID1", "ID2"}, {"SHOT1"}, {"ID1": ["SHOT1"], "ID2": ["SHOT1"]})
