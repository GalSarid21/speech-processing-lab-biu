import json

import pandas as pd
import pytest
from assertpy import assert_that

from speech_processing.config.core import ExperimentMeta
from speech_processing.data.dtos import AudioRequest, TextRequest
from speech_processing.data.enums import DatasetType
from speech_processing.data.icbhi import (
    ICBHI_LABELS,
    MAX_DEMONSTRATIONS,
    build_icbhi_few_shot_pool,
    filter_demonstrations,
    load_icbhi_requests,
    load_near_duplicates,
    parse_instruction_metadata,
    validate_icbhi_demonstrations,
)
from speech_processing.runners.icbhi import ExperimentVersion
from speech_processing.utils.consts import NO_FEATURES_PLACEHOLDER
from speech_processing.utils.exceptions import DatasetIntegrityError, FewShotLeakageError
from tests.conftest import ICBHI_INSTRUCTION_TEMPLATES, ICBHI_ROWS

BASE_FIELDS = {
    "dataset": DatasetType.ICBHI,
    "experiment_name": "test",
    "prompt": "Choose the diagnosis.",
    "max_new_tokens": 32,
    "batch_size": 2,
}

LOCATION_MARKER = "recorded at"
MODE_MARKER = "acquisition mode"


def meta(**overrides) -> ExperimentMeta:
    return ExperimentMeta(**(BASE_FIELDS | overrides))


@pytest.mark.parametrize(
    "instruction, expected",
    [
        (ICBHI_INSTRUCTION_TEMPLATES[0], ("trachea", "multichannel")),
        (ICBHI_INSTRUCTION_TEMPLATES[1], ("left lower", "single channel")),
        (ICBHI_INSTRUCTION_TEMPLATES[2], ("right upper", "single channel")),
    ],
)
def test_parse_instruction_metadata(instruction, expected):
    assert_that(parse_instruction_metadata(instruction)).is_equal_to(expected)


def test_parse_instruction_metadata_raises_without_metadata():
    assert_that(parse_instruction_metadata).raises(DatasetIntegrityError).when_called_with("no metadata here")


def test_requests_are_lettered_over_the_closed_label_set(patched_icbhi, icbhi_config):
    items = load_icbhi_requests(icbhi_config, meta())

    assert_that(items).is_length(len(ICBHI_ROWS))
    assert_that([gt for _, gt in items]).is_equal_to(["A", "H", "B"])

    first = items[0][0]
    assert_that(first).is_instance_of(AudioRequest)
    assert_that(first.instruction).contains("Choices:\nA. COPD")
    assert_that(first.instruction).contains("H. No potential disease detected")
    assert_that(first.metadata.choices).is_equal_to(list(ICBHI_LABELS))
    assert_that(first.metadata.category).is_equal_to("COPD")
    assert_that(first.metadata.sub_category).is_equal_to("multichannel|trachea")


@pytest.mark.parametrize("key", ["r0", "a1a", "a3a", "a5c"])
def test_blind_track_never_leaks_recording_metadata(patched_icbhi, icbhi_config, key):
    experiment = ExperimentVersion[key].value
    instruction = str(load_icbhi_requests(icbhi_config, experiment)[0][0].instruction)

    assert_that(instruction).does_not_contain(LOCATION_MARKER)
    assert_that(instruction).does_not_contain(MODE_MARKER)
    assert_that(instruction).does_not_contain("trachea")
    assert_that(instruction).does_not_contain("multichannel")


def test_metadata_aware_track_states_the_recording_conditions(patched_icbhi, icbhi_config):
    instruction = load_icbhi_requests(icbhi_config, meta(include_recording_metadata=True))[0][0].instruction
    assert_that(instruction).contains("Recording site: trachea. Acquisition mode: multichannel.")


def test_label_outside_the_closed_set_raises(mocker, icbhi_config):
    broken = [dict(ICBHI_ROWS[0]) | {"label": "COVID-19"}]
    mocker.patch("speech_processing.data.icbhi._load_icbhi_frame", return_value=pd.DataFrame(broken))
    assert_that(load_icbhi_requests).raises(DatasetIntegrityError).when_called_with(icbhi_config, meta())


@pytest.mark.parametrize(
    "overrides, present, absent",
    [
        ({}, ["Question: What is the patient's diagnosis?"], ["Measured respiratory", "Audio tagger"]),
        ({"inject_acoustic_features": True}, ["Measured respiratory features:", "Rate ~18/min"], ["Audio tagger"]),
        ({"inject_audio_tags": True}, ["Audio tagger output:", "Breathing 0.81"], ["Measured respiratory"]),
        ({"inject_neighbor_labels": True}, ["most acoustically similar", "COPD, COPD, Pneumonia"], []),
    ],
)
def test_evidence_injection(patched_icbhi, icbhi_config, overrides, present, absent):
    instruction = load_icbhi_requests(icbhi_config, meta(**overrides))[0][0].instruction

    for fragment in present:
        assert_that(instruction).contains(fragment)
    for fragment in absent:
        assert_that(instruction).does_not_contain(fragment)


def test_item_without_features_gets_a_placeholder(patched_icbhi, icbhi_config):
    # The fixture only has respiratory evidence for audio1 and audio2.
    items = load_icbhi_requests(icbhi_config, meta(inject_acoustic_features=True))
    assert_that(items[2][0].instruction).contains(NO_FEATURES_PLACEHOLDER)


def test_missing_features_file_raises(patched_icbhi, icbhi_config, tmp_path):
    config = icbhi_config.model_copy(update={"respiratory_features_file": str(tmp_path / "nope.jsonl")})
    assert_that(load_icbhi_requests).raises(DatasetIntegrityError).when_called_with(
        config, meta(inject_acoustic_features=True)
    )


def test_cycle_spans_reach_the_metadata(patched_icbhi, icbhi_config):
    items = load_icbhi_requests(icbhi_config, meta(presentation="cycles"))
    assert_that(items[0][0].metadata.cycle_spans).is_equal_to([(0.0, 1.0), (1.0, 2.0)])
    assert_that(items[2][0].metadata.cycle_spans).is_none()


def test_text_only_builds_a_text_request(patched_icbhi, icbhi_config):
    request = load_icbhi_requests(icbhi_config, meta(text_only=True, include_recording_metadata=True))[0][0]
    assert_that(request).is_instance_of(TextRequest)
    assert_that(request.instruction).contains("Acquisition mode: multichannel")


def test_audio_bytes_are_materialised_to_files(patched_icbhi, icbhi_config, tmp_path):
    request = load_icbhi_requests(icbhi_config, meta())[0][0]
    assert_that(request.audio_path).ends_with("audio1.wav")
    assert_that(json.dumps(request.audio_path)).contains("icbhi_audio")


def test_demonstration_pool_respects_the_audio_limit(patched_icbhi, icbhi_config):
    pool = build_icbhi_few_shot_pool(icbhi_config, meta(few_shot_mode="audio"))

    assert_that(len(pool)).is_less_than_or_equal_to(MAX_DEMONSTRATIONS)
    assert_that(sorted(pool)).is_equal_to(["audio1", "audio2", "audio3"])
    assert_that(pool["audio1"].assistant_text).is_equal_to("A")
    assert_that(pool["audio1"].audio_path).is_not_none()


def test_demonstrations_exclude_the_item_and_its_near_duplicates(patched_icbhi, icbhi_config):
    pool = build_icbhi_few_shot_pool(icbhi_config, meta(few_shot_mode="audio"))
    near_duplicates = load_near_duplicates(icbhi_config.near_duplicates_file)

    turns = filter_demonstrations("audio1", pool, near_duplicates)

    assert_that(turns).is_length(1)
    assert_that(turns[0].assistant_text).is_equal_to("H")


@pytest.mark.parametrize("shot_ids", [{"audio1"}, {"audio3"}])
def test_leaking_demonstrations_raise(shot_ids):
    assert_that(validate_icbhi_demonstrations).raises(FewShotLeakageError).when_called_with(
        "audio1", shot_ids, {"audio1": ["audio3"]}
    )


def test_clean_demonstrations_pass():
    validate_icbhi_demonstrations("audio1", {"audio2"}, {"audio1": ["audio3"]})
