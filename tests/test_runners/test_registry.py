import pytest
from assertpy import assert_that

from speech_processing.data.dataset import load_mmar_requests
from speech_processing.data.dtos import AudioRequest, TextRequest
from speech_processing.runners.base import experiment_keys
from speech_processing.runners.mmar import ExperimentVersion
from speech_processing.utils.consts import COT_START_TAG
from speech_processing.utils.exceptions import DeprecatedExperimentError

ACTIVE = [v for v in ExperimentVersion if not v.value.deprecated]
ACTIVE_KEYS = [v.name for v in ACTIVE]
NUM_EVAL_ITEMS = 2

TRANSCRIPT_MARKER = "Transcript: "
DIARIZED_MARKER = "Diarized transcript:"
FEATURES_MARKER = "Measured acoustic features:"


def last_turn(prompt: str | list[str]) -> str:
    return prompt[-1] if isinstance(prompt, list) else prompt


@pytest.mark.parametrize("key", ACTIVE_KEYS)
def test_active_prompts_ask_for_a_letter(key):
    prompt = ExperimentVersion[key].value.prompt
    assert_that(str(prompt)).does_not_contain("exact text")
    assert_that(last_turn(prompt).lower()).contains("letter")


@pytest.mark.parametrize("key", ACTIVE_KEYS)
def test_active_experiments_build_valid_requests(patched_mmar, mmar_config, key):
    exp = ExperimentVersion[key].value
    items = load_mmar_requests(mmar_config, exp, is_text_only=exp.text_only)

    assert_that(items).is_length(NUM_EVAL_ITEMS * exp.num_shuffled_variants)

    expected_class = TextRequest if exp.text_only else AudioRequest
    for req, ground_truth in items:
        assert_that(req).is_instance_of(expected_class)
        assert_that(ground_truth).is_in("A", "B", "C")
        assert_that(last_turn(req.instruction)).contains("Choices:\nA. ")


def test_experiment_names_are_unique():
    names = [v.value.experiment_name for v in ExperimentVersion]
    assert_that(sorted(set(names))).is_length(len(names))


@pytest.mark.parametrize(
    "key, marker",
    [
        ("v3", TRANSCRIPT_MARKER),
        ("v13", TRANSCRIPT_MARKER),
        ("v18", DIARIZED_MARKER),
        ("v21", DIARIZED_MARKER),
        ("v19", FEATURES_MARKER),
        ("v20", FEATURES_MARKER),
    ],
)
def test_evidence_reaches_the_prompt(patched_mmar, mmar_config, key, marker):
    exp = ExperimentVersion[key].value
    instruction = load_mmar_requests(mmar_config, exp, is_text_only=exp.text_only)[0][0].instruction
    assert_that(last_turn(instruction)).contains(marker)


def test_audio_first_warning_reaches_the_prompt(patched_mmar, mmar_config):
    instruction = load_mmar_requests(mmar_config, ExperimentVersion.v28.value)[0][0].instruction
    assert_that(str(instruction)).contains("WARNING")


@pytest.mark.parametrize("key", ["v1", "v2", "v11", "v18", "v19", "v22", "v27", "v28"])
def test_transcript_free_experiments_carry_no_transcript(patched_mmar, mmar_config, key):
    exp = ExperimentVersion[key].value
    instruction = load_mmar_requests(mmar_config, exp)[0][0].instruction
    assert_that(str(instruction)).does_not_contain(TRANSCRIPT_MARKER)


BASELINE_SIGNATURE_FIELDS = (
    "prompt",
    "presentation",
    "contrastive_alpha",
    "num_shuffled_variants",
    "few_shot_mode",
    "use_transcript",
    "inject_diarized_transcript",
    "inject_acoustic_features",
    "text_only",
)


def signature(meta) -> tuple:
    return tuple(str(getattr(meta, field)) for field in BASELINE_SIGNATURE_FIELDS)


@pytest.mark.parametrize("key", [k for k in ACTIVE_KEYS if k != "v1"])
def test_no_active_experiment_is_the_baseline_in_disguise(key):
    assert_that(signature(ExperimentVersion[key].value)).is_not_equal_to(signature(ExperimentVersion.v1.value))


@pytest.mark.parametrize("key, expects_prefill", [("v2", True), ("v1", False)])
def test_cot_prefill(patched_mmar, mmar_config, key, expects_prefill):
    req = load_mmar_requests(mmar_config, ExperimentVersion[key].value)[0][0]
    expected = f"{COT_START_TAG}\n" if expects_prefill else None
    assert_that(req.assistant_prefill).is_equal_to(expected)


@pytest.mark.parametrize("key", ["v15", "V17"])
def test_deprecated_versions_raise(key):
    assert_that(ExperimentVersion.get_version).raises(DeprecatedExperimentError).when_called_with(key)


def test_unknown_version_raises_value_error():
    assert_that(ExperimentVersion.get_version).raises(ValueError).when_called_with("v999")


CORE_KEYS = {"v1", "v2", "v3", "v7", "v13", "v18", "v20", "v22", "v23", "v25", "v27", "v28"}


def test_the_core_tier_is_the_reported_set():
    assert_that(set(experiment_keys(ExperimentVersion, "core"))).is_equal_to(CORE_KEYS)


def test_core_covers_the_baseline_the_split_and_every_new_family():
    core = set(experiment_keys(ExperimentVersion, "core"))
    assert_that(core).contains("v1")  # baseline
    assert_that(core).contains("v13")  # text-only control, defines the subset split
    assert_that(core).contains("v18", "v20")  # T1 acoustic evidence
    assert_that(core).contains("v22", "v23")  # T2 presentation
    assert_that(core).contains("v25")  # T3 contrastive
    assert_that(core).contains("v27")  # T4 consistency
    assert_that(core).contains("v28")  # T5 framing


@pytest.mark.parametrize("key", sorted(set(ACTIVE_KEYS) - CORE_KEYS))
def test_every_non_core_experiment_explains_why_it_is_out_of_scope(key):
    meta = ExperimentVersion[key].value
    assert_that(meta.is_future_work).is_true()
    assert_that(meta.future_work_reason).is_not_none()
    assert_that(len(meta.future_work_reason)).is_greater_than(30)


def test_deprecated_entries_belong_to_no_tier():
    """v15 and v17 are retired, which is a different exit from being out of scope."""
    runnable = set(experiment_keys(ExperimentVersion))
    assert_that(runnable).does_not_contain("v15", "v17")
    for key in ("v15", "v17"):
        assert_that(ExperimentVersion[key].value.is_future_work).is_false()


def test_tiers_partition_the_runnable_experiments():
    core = experiment_keys(ExperimentVersion, "core")
    future = experiment_keys(ExperimentVersion, "future_work")
    assert_that(set(core) & set(future)).is_empty()
    assert_that(sorted(core + future)).is_equal_to(sorted(experiment_keys(ExperimentVersion)))
