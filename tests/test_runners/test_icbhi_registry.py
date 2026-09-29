import pytest
from assertpy import assert_that

from speech_processing.data.dtos import AudioRequest, TextRequest
from speech_processing.data.enums import DatasetType
from speech_processing.data.icbhi import ICBHI_LABELS, load_icbhi_requests
from speech_processing.runners.icbhi import ACOUSTIC_DICTIONARY, ExperimentVersion
from speech_processing.utils.consts import COT_START_TAG

ACTIVE = [v for v in ExperimentVersion if not v.value.deprecated]
ACTIVE_KEYS = [v.name for v in ACTIVE]
NUM_EVAL_ITEMS = 3

LOCATION_MARKER = "recorded at"
MODE_MARKER = "acquisition mode"
METADATA_AWARE_KEYS = {"c2", "r0_aware", "a1d"}
BLIND_KEYS = [k for k in ACTIVE_KEYS if k not in METADATA_AWARE_KEYS]


def last_turn(prompt: str | list[str]) -> str:
    return prompt[-1] if isinstance(prompt, list) else prompt


@pytest.mark.parametrize("key", ACTIVE_KEYS)
def test_every_experiment_is_declared_for_icbhi(key):
    assert_that(ExperimentVersion[key].value.dataset).is_equal_to(DatasetType.ICBHI)


@pytest.mark.parametrize("key", ACTIVE_KEYS)
def test_prompts_ask_for_a_letter(key):
    assert_that(last_turn(ExperimentVersion[key].value.prompt).lower()).contains("letter")


@pytest.mark.parametrize("key", ACTIVE_KEYS)
def test_every_experiment_builds_valid_requests(patched_icbhi, icbhi_config, key):
    experiment = ExperimentVersion[key].value
    items = load_icbhi_requests(icbhi_config, experiment)

    assert_that(items).is_length(NUM_EVAL_ITEMS * experiment.num_shuffled_variants)

    expected_class = TextRequest if experiment.text_only else AudioRequest
    for request, ground_truth in items:
        assert_that(request).is_instance_of(expected_class)
        assert_that(ground_truth).is_in(*[chr(ord("A") + i) for i in range(len(ICBHI_LABELS))])
        assert_that(last_turn(request.instruction)).contains("Choices:\nA. ")
        assert_that(sorted(request.metadata.choices)).is_equal_to(sorted(ICBHI_LABELS))


@pytest.mark.parametrize("key", BLIND_KEYS)
def test_the_blind_track_never_shows_recording_metadata(patched_icbhi, icbhi_config, key):
    """The single most important guard: acquisition mode alone predicts the diagnosis."""
    experiment = ExperimentVersion[key].value
    instruction = str(load_icbhi_requests(icbhi_config, experiment)[0][0].instruction)

    assert_that(instruction).does_not_contain(LOCATION_MARKER)
    assert_that(instruction).does_not_contain(MODE_MARKER)
    assert_that(instruction).does_not_contain("trachea")


@pytest.mark.parametrize("key", sorted(METADATA_AWARE_KEYS))
def test_the_aware_track_does_show_recording_metadata(patched_icbhi, icbhi_config, key):
    experiment = ExperimentVersion[key].value
    instruction = str(load_icbhi_requests(icbhi_config, experiment)[0][0].instruction)
    assert_that(instruction).contains("Acquisition mode: multichannel")


def test_experiment_names_are_unique():
    names = [v.value.experiment_name for v in ExperimentVersion]
    assert_that(sorted(set(names))).is_length(len(names))


@pytest.mark.parametrize(
    "key, marker",
    [
        ("a1a", "Measured respiratory features:"),
        ("a1c", "Measured respiratory features:"),
        ("a2a", "Audio tagger output:"),
        ("a2b", "Measured respiratory features:"),
        ("a6a", "most acoustically similar"),
    ],
)
def test_evidence_reaches_the_prompt(patched_icbhi, icbhi_config, key, marker):
    experiment = ExperimentVersion[key].value
    instruction = str(load_icbhi_requests(icbhi_config, experiment)[0][0].instruction)
    assert_that(instruction).contains(marker)


@pytest.mark.parametrize("key", ["r3", "r4", "a1b", "a1c", "a3c"])
def test_dictionary_experiments_carry_the_dictionary(key):
    assert_that(str(ExperimentVersion[key].value.prompt)).contains(ACOUSTIC_DICTIONARY)


BASELINE_SIGNATURE_FIELDS = (
    "prompt",
    "presentation",
    "preprocessing",
    "contrastive_alpha",
    "calibration",
    "num_shuffled_variants",
    "few_shot_mode",
    "text_only",
    "replace_audio_with_silence",
    "include_recording_metadata",
    "inject_acoustic_features",
    "inject_audio_tags",
    "inject_neighbor_labels",
)


def signature(meta) -> tuple:
    return tuple(str(getattr(meta, field)) for field in BASELINE_SIGNATURE_FIELDS)


@pytest.mark.parametrize("key", [k for k in ACTIVE_KEYS if k != "r0"])
def test_no_experiment_is_the_baseline_in_disguise(key):
    assert_that(signature(ExperimentVersion[key].value)).is_not_equal_to(signature(ExperimentVersion.r0.value))


@pytest.mark.parametrize("key, expects_prefill", [("r2", True), ("r4", True), ("r0", False)])
def test_cot_prefill(patched_icbhi, icbhi_config, key, expects_prefill):
    request = load_icbhi_requests(icbhi_config, ExperimentVersion[key].value)[0][0]
    assert_that(request.assistant_prefill).is_equal_to(f"{COT_START_TAG}\n" if expects_prefill else None)


def test_the_silence_control_is_the_only_one_replacing_audio():
    replacing = [v.name for v in ExperimentVersion if v.value.replace_audio_with_silence]
    assert_that(replacing).is_equal_to(["c1"])


def test_unknown_version_raises_value_error():
    assert_that(ExperimentVersion.get_version).raises(ValueError).when_called_with("v999")


def test_shuffled_variants_keep_the_true_label_behind_the_ground_truth_letter(patched_icbhi, icbhi_config):
    items = load_icbhi_requests(icbhi_config, ExperimentVersion.a7a.value)

    assert_that(items).is_length(NUM_EVAL_ITEMS * 5)
    for request, ground_truth in items:
        metadata = request.metadata
        assert_that(metadata.choices[ord(ground_truth) - ord("A")]).is_equal_to(metadata.category)

    assert_that(items[0][0].metadata.permutation).is_equal_to(list(range(len(ICBHI_LABELS))))
