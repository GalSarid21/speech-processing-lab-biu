import pytest
from assertpy import assert_that

from speech_processing.data.dtos import AudioRequest, TextRequest
from speech_processing.data.enums import DatasetType
from speech_processing.data.icbhi import ICBHI_LABELS, ICBHI_PRIMARY_LABELS, load_icbhi_requests
from speech_processing.runners.base import experiment_keys
from speech_processing.runners.icbhi import ACOUSTIC_DICTIONARY, ExperimentVersion
from speech_processing.utils.consts import COT_START_TAG, VOXTRAL_MAX_AUDIOS_PER_PROMPT
from speech_processing.utils.exceptions import DeprecatedExperimentError

ACTIVE = [v for v in ExperimentVersion if not v.value.deprecated]
ACTIVE_KEYS = [v.name for v in ACTIVE]
NUM_EVAL_ITEMS = 3

LOCATION_MARKER = "recorded at"
MODE_MARKER = "acquisition mode"
METADATA_AWARE_KEYS = {"c2", "c3", "r0_aware", "a1d"}
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


def test_only_the_controls_replace_the_audio():
    replacing = {v.name for v in ExperimentVersion if v.value.replace_audio_with_silence}
    assert_that(replacing).is_equal_to({"c1", "c3", "c4"})


@pytest.mark.parametrize("control, parent", [("c1", "r0"), ("c3", "r0_aware"), ("c4", "a1b")])
def test_each_control_is_an_audio_ablation_of_its_parent(control, parent):
    """Same model, same prompt, same evidence: the only difference is that the audio is silence.
    So the gap between a control and its parent is exactly what listening contributed."""
    ablated = ExperimentVersion[control].value
    original = ExperimentVersion[parent].value
    ignore = {"experiment_name", "replace_audio_with_silence"}

    differing = {
        field
        for field in type(original).model_fields
        if field not in ignore and getattr(ablated, field) != getattr(original, field)
    }
    assert_that(differing).is_empty()
    assert_that(ablated.replace_audio_with_silence).is_true()
    assert_that(original.replace_audio_with_silence).is_false()


def test_r8_is_deprecated_because_its_prior_had_no_source():
    assert_that(ExperimentVersion.get_version).raises(DeprecatedExperimentError).when_called_with("r8")


def test_unknown_version_raises_value_error():
    assert_that(ExperimentVersion.get_version).raises(ValueError).when_called_with("v999")


def test_shuffled_variants_keep_the_true_label_behind_the_ground_truth_letter(patched_icbhi, icbhi_config):
    items = load_icbhi_requests(icbhi_config, ExperimentVersion.a7a.value)

    assert_that(items).is_length(NUM_EVAL_ITEMS * 5)
    for request, ground_truth in items:
        metadata = request.metadata
        assert_that(metadata.choices[ord(ground_truth) - ord("A")]).is_equal_to(metadata.category)

    assert_that(items[0][0].metadata.permutation).is_equal_to(list(range(len(ICBHI_LABELS))))


CORE_KEYS = {"c1", "c3", "c4", "r0", "r0_aware", "r2", "r3", "a1b", "a3a", "a3b", "a5a", "a5c"}


def test_the_core_tier_is_the_pre_registered_set():
    assert_that(set(experiment_keys(ExperimentVersion, "core"))).is_equal_to(CORE_KEYS)


def test_every_pre_registered_comparison_has_both_arms_in_core():
    """(1) a1b vs r0  (2) best a3* vs r0  (3) a5c vs a5a."""
    core = set(experiment_keys(ExperimentVersion, "core"))
    for left, right in [("a1b", "r0"), ("a3a", "r0"), ("a3b", "r0"), ("a5c", "a5a")]:
        assert_that(core).contains(left, right)


def test_the_controls_that_interpret_the_results_are_in_core():
    core = set(experiment_keys(ExperimentVersion, "core"))
    # c1 is the label prior; c3 defines the metadata-solvable split; c4 says whether A1's gain needs
    # the audio at all. All three run on the same model as the arms they interpret.
    assert_that(core).contains("c1", "c3", "c4", "r0_aware")
    assert_that(core).does_not_contain("c2", "a1c")  # cross-model controls are future work


@pytest.mark.parametrize("key", sorted(set(ACTIVE_KEYS) - CORE_KEYS))
def test_every_non_core_experiment_explains_why_it_is_out_of_scope(key):
    meta = ExperimentVersion[key].value
    assert_that(meta.is_future_work).is_true()
    assert_that(meta.future_work_reason).is_not_none()
    assert_that(len(meta.future_work_reason)).is_greater_than(30)


def test_future_work_is_still_runnable():
    """Out of scope is not deprecated: these must still resolve."""
    assert_that(ExperimentVersion.get_version("a7a").value.experiment_name).is_equal_to("choice_shuffling")


def test_tiers_partition_the_runnable_experiments():
    core = experiment_keys(ExperimentVersion, "core")
    future = experiment_keys(ExperimentVersion, "future_work")
    assert_that(set(core) & set(future)).is_empty()
    assert_that(sorted(core + future)).is_equal_to(sorted(experiment_keys(ExperimentVersion)))


def test_no_core_arm_needs_more_audio_clips_than_voxtral_accepts():
    """r6 was core until it met vLLM's 5-clip limit on the card: 6 demonstrations + the test clip."""
    for key in experiment_keys(ExperimentVersion, "core"):
        meta = ExperimentVersion[key].value
        if meta.few_shot_mode in ("audio", "rag"):
            assert_that(len(ICBHI_PRIMARY_LABELS) + 1).is_less_than_or_equal_to(VOXTRAL_MAX_AUDIOS_PER_PROMPT)
