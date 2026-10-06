import json

import pandas as pd
import pytest
from assertpy import assert_that

from speech_processing.config.core import ExperimentMeta
from speech_processing.data.dtos import AudioRequest, TextRequest
from speech_processing.data.enums import DatasetType
from speech_processing.data.icbhi import (
    ICBHI_LABELS,
    ICBHI_PRIMARY_LABELS,
    MAX_DEMONSTRATIONS,
    build_icbhi_few_shot_pool,
    build_icbhi_neighbor_pool,
    load_demonstration_split,
    load_icbhi_requests,
    parse_instruction_metadata,
    select_demonstration_split,
    validate_demonstration_disjointness,
)
from speech_processing.runners.icbhi import ExperimentVersion
from speech_processing.utils.consts import NO_FEATURES_PLACEHOLDER
from speech_processing.utils.exceptions import DatasetIntegrityError, FewShotLeakageError
from tests.conftest import ICBHI_DEMO_IDS, ICBHI_EVAL_IDS, ICBHI_INSTRUCTION_TEMPLATES, ICBHI_ROWS

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

    assert_that(items).is_length(len(ICBHI_EVAL_IDS))
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


def test_held_out_demonstrations_are_never_scored(patched_icbhi, icbhi_config):
    """Every experiment scores the same items, and a demonstration is never one of them."""
    for experiment in (meta(), meta(few_shot_mode="audio"), meta(include_recording_metadata=True)):
        scored = [req.metadata.item_id for req, _ in load_icbhi_requests(icbhi_config, experiment)]
        assert_that(scored).is_equal_to(ICBHI_EVAL_IDS)
        assert_that(scored).does_not_contain(*ICBHI_DEMO_IDS)


def test_missing_demonstration_split_is_fatal(patched_icbhi, icbhi_config, tmp_path):
    """Without it the evaluation set would silently change, which is what broke V1 vs V3."""
    config = icbhi_config.model_copy(update={"demo_ids_file": str(tmp_path / "nope.json")})
    assert_that(load_icbhi_requests).raises(DatasetIntegrityError).when_called_with(config, meta())


def test_the_few_shot_pool_is_exactly_the_held_out_split(patched_icbhi, icbhi_config):
    pool = build_icbhi_few_shot_pool(icbhi_config, meta(few_shot_mode="audio"))

    assert_that(list(pool)).is_equal_to(ICBHI_DEMO_IDS)
    assert_that(len(pool)).is_less_than_or_equal_to(MAX_DEMONSTRATIONS)
    assert_that(pool["audio4"].assistant_text).is_equal_to("C")  # URTI
    assert_that(pool["audio4"].audio_path).is_not_none()


def test_retrieval_candidates_come_only_from_the_held_out_split(patched_icbhi, icbhi_config):
    pool = build_icbhi_neighbor_pool(icbhi_config, meta(few_shot_mode="rag", rag_mapping_file="m.json"))
    assert_that(set(pool)).is_equal_to(set(ICBHI_DEMO_IDS))


def test_demonstration_split_round_trips(icbhi_config):
    split = load_demonstration_split(icbhi_config.demo_ids_file)
    assert_that(split.held_out_ids).is_equal_to(set(ICBHI_DEMO_IDS))


LABELS_BY_ID = {
    "copd_clean": "COPD",
    "copd_dup": "COPD",
    "copd_dup_twin": "COPD",
    "pneu": "Pneumonia",
    "urti": "URTI",
    "bronchiect": "Bronchiectasis",
    "bronchiol": "Bronchiolitis",
    "healthy": "No potential disease detected",
    "asthma": "Asthma",
}
NEAR_DUPLICATES = {"copd_dup": ["copd_dup_twin"], "copd_dup_twin": ["copd_dup"]}


def test_selection_takes_one_demonstration_per_primary_class():
    split = select_demonstration_split(LABELS_BY_ID, NEAR_DUPLICATES)

    chosen_labels = [LABELS_BY_ID[item_id] for item_id in split.demonstration_ids]
    assert_that(chosen_labels).is_equal_to(list(ICBHI_PRIMARY_LABELS))
    assert_that(split.demonstration_ids).does_not_contain("asthma")


def test_selection_prefers_clips_without_near_duplicates():
    split = select_demonstration_split(LABELS_BY_ID, NEAR_DUPLICATES)
    assert_that(split.demonstration_ids).contains("copd_clean")
    assert_that(split.excluded_near_duplicate_ids).is_empty()


def test_a_chosen_demonstrations_near_duplicates_are_held_out_too():
    only_duplicated_copd = {k: v for k, v in LABELS_BY_ID.items() if k != "copd_clean"}

    split = select_demonstration_split(only_duplicated_copd, NEAR_DUPLICATES)

    chosen_copd = next(i for i in split.demonstration_ids if LABELS_BY_ID[i] == "COPD")
    assert_that(split.excluded_near_duplicate_ids).is_equal_to(NEAR_DUPLICATES[chosen_copd])
    assert_that(split.held_out_ids).contains("copd_dup", "copd_dup_twin")


def test_selection_is_deterministic():
    first = select_demonstration_split(LABELS_BY_ID, NEAR_DUPLICATES)
    second = select_demonstration_split(LABELS_BY_ID, NEAR_DUPLICATES)
    assert_that(first).is_equal_to(second)


def test_selection_fails_when_a_primary_class_has_no_items():
    without_urti = {k: v for k, v in LABELS_BY_ID.items() if v != "URTI"}
    assert_that(select_demonstration_split).raises(DatasetIntegrityError).when_called_with(without_urti, {})


def test_overlapping_demonstrations_raise():
    assert_that(validate_demonstration_disjointness).raises(FewShotLeakageError).when_called_with(
        {"audio1", "audio2"}, {"audio2"}
    )


def test_disjoint_demonstrations_pass():
    validate_demonstration_disjointness({"audio1", "audio2"}, {"audio4"})


def test_num_samples_counts_scored_items_not_raw_rows(mocker, icbhi_config):
    """With the held-out row first, truncating raw rows would score one item fewer than asked."""
    held_out_first = [ICBHI_ROWS[-1], *ICBHI_ROWS[:-1]]
    mocker.patch("speech_processing.data.icbhi._load_icbhi_frame", return_value=pd.DataFrame(held_out_first))

    items = load_icbhi_requests(icbhi_config.model_copy(update={"num_samples": 2}), meta())

    assert_that([req.metadata.item_id for req, _ in items]).is_equal_to(ICBHI_EVAL_IDS[:2])


def test_a_misfiring_duplicate_detector_cannot_shrink_the_evaluation_set():
    """The first detector flagged 170 of 174 real clips. Holding all their duplicates out would have
    left almost nothing to evaluate, so the selection refuses instead."""
    labels = {f"copd{i}": "COPD" for i in range(40)} | {k: v for k, v in LABELS_BY_ID.items() if v != "COPD"}
    everything_duplicates = {item_id: [other for other in labels if other != item_id] for item_id in labels}

    assert_that(select_demonstration_split).raises(DatasetIntegrityError).when_called_with(
        labels, everything_duplicates
    )


def test_a_demonstration_set_that_cannot_fit_in_one_prompt_fails_before_inference(
    patched_icbhi, icbhi_config, tmp_path
):
    """One demonstration per primary class (6) plus the test clip is 7 clips; Voxtral takes 5."""
    split_file = tmp_path / "six_demos.json"
    split_file.write_text(json.dumps({"demonstration_ids": [f"d{i}" for i in range(len(ICBHI_PRIMARY_LABELS))]}))
    config = icbhi_config.model_copy(update={"demo_ids_file": str(split_file)})

    with pytest.raises(DatasetIntegrityError, match="Voxtral accepts"):
        build_icbhi_few_shot_pool(config, meta(few_shot_mode="audio"))
