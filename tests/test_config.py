from types import SimpleNamespace

import pytest
from assertpy import assert_that
from pydantic import ValidationError

from speech_processing.config.core import ExperimentMeta
from speech_processing.config.factories import (
    VOXTRAL_MODEL_ID,
    create_icbhi_config,
    create_mmar_config,
)
from speech_processing.utils.consts import RAG_STOP_SEQUENCES
from speech_processing.utils.exceptions import InvalidExperimentConfigError

BASE_FIELDS = {
    "experiment_name": "test",
    "prompt": "Answer.",
    "max_new_tokens": 256,
    "batch_size": 8,
}


def meta(**overrides) -> ExperimentMeta:
    return ExperimentMeta(**(BASE_FIELDS | overrides))


@pytest.fixture
def args():
    return SimpleNamespace(num_samples=5, sample_ids_file=None, output_dir="./test_results", gpu_pct=0.8, max_seqs=32)


def test_factory_builds_the_audio_config_with_stop_at_construction(args):
    config = create_mmar_config(args, meta(few_shot_mode="rag", rag_mapping_file="m.json", stop=RAG_STOP_SEQUENCES))

    assert_that(config.output_dir).is_equal_to("./test_results")
    assert_that(config.text_model).is_none()
    assert_that(config.audio_model.model_id).is_equal_to(VOXTRAL_MODEL_ID)
    assert_that(config.audio_model.gpu_memory_utilization).is_equal_to(0.8)
    assert_that(config.audio_model.stop).is_equal_to(RAG_STOP_SEQUENCES)
    assert_that(config.dataset.acoustic_features_file).ends_with("mmar_acoustic_features.jsonl")


def test_text_only_builds_only_the_text_config(args):
    config = create_mmar_config(args, meta(text_only=True, use_transcript=True, max_new_tokens=2048))

    assert_that(config.audio_model).is_none()
    assert_that(config.text_model.max_new_tokens).is_equal_to(2048)


def test_configs_are_frozen(args):
    config = create_mmar_config(args, meta())
    with pytest.raises(ValidationError):
        config.dataset.num_samples = 1


@pytest.mark.parametrize(
    "rule, overrides",
    [
        ("R1", {"few_shot_mode": "rag"}),
        ("R2", {"rag_mapping_file": "m.json"}),
        ("R3", {"presentation": "two_pass_localization"}),
        ("R3", {"presentation": "two_pass_localization", "prompt": ["only one turn"]}),
        ("R4", {"text_only": True, "use_transcript": True, "presentation": "chunked"}),
        ("R5", {"text_only": True}),
        ("R6", {"presentation": "chunked", "few_shot_mode": "audio"}),
        ("R7", {"contrastive_alpha": 0.5, "few_shot_mode": "text"}),
        ("R8", {"contrastive_alpha": 0.5, "num_shuffled_variants": 3}),
    ],
)
def test_experiment_rules_are_enforced(rule, overrides):
    with pytest.raises(InvalidExperimentConfigError) as excinfo:
        meta(**overrides)
    assert_that(str(excinfo.value)).contains(rule)


@pytest.mark.parametrize("overrides", [{"contrastive_alpha": -0.5}, {"num_shuffled_variants": 0}])
def test_field_constraints(overrides):
    with pytest.raises(ValidationError):
        meta(**overrides)


def test_deprecated_entries_skip_validation():
    deprecated = meta(prompt="", few_shot_mode="rag", deprecated=True, deprecation_reason="B3")
    assert_that(deprecated.deprecated).is_true()


VOXTRAL_MINI = "mistralai/Voxtral-Mini-3B-2507"


def test_audio_model_id_can_be_overridden(args):
    """The 24B default needs an 80GB card; a smaller GPU has to be able to run the pipeline at all."""
    overridden = SimpleNamespace(**vars(args), audio_model_id=VOXTRAL_MINI)

    config = create_mmar_config(overridden, meta())

    assert_that(config.audio_model.model_id).is_equal_to(VOXTRAL_MINI)


def test_text_model_id_can_be_overridden(args):
    overridden = SimpleNamespace(**vars(args), text_model_id="google/gemma-2-2b-it")

    config = create_mmar_config(overridden, meta(text_only=True, use_transcript=True))

    assert_that(config.text_model.model_id).is_equal_to("google/gemma-2-2b-it")


@pytest.mark.parametrize("builder", [create_mmar_config, create_icbhi_config])
def test_model_ids_default_to_the_registry_constants(args, builder):
    config = builder(args, meta())

    assert_that(config.audio_model.model_id).is_equal_to(VOXTRAL_MODEL_ID)


@pytest.mark.parametrize("builder", [create_mmar_config, create_icbhi_config])
def test_max_seqs_and_gpu_pct_reach_the_model_config(args, builder):
    config = builder(args, meta())

    assert_that(config.audio_model.max_num_seqs).is_equal_to(args.max_seqs)
    assert_that(config.audio_model.gpu_memory_utilization).is_equal_to(args.gpu_pct)


@pytest.mark.parametrize(
    "rule, overrides",
    [
        ("R14", {"tier": "future_work", "future_work_reason": "r", "deprecated": True}),
        ("R15", {"tier": "future_work"}),
        ("R16", {"future_work_reason": "set without the tier"}),
    ],
)
def test_tier_rules_are_enforced(rule, overrides):
    with pytest.raises(InvalidExperimentConfigError) as excinfo:
        meta(**overrides)
    assert_that(str(excinfo.value)).contains(rule)


def test_experiments_are_core_by_default():
    assert_that(meta().tier).is_equal_to("core")
    assert_that(meta().is_future_work).is_false()
