from types import SimpleNamespace

import pytest
from assertpy import assert_that
from pydantic import ValidationError

from speech_processing.config.core import ExperimentMeta
from speech_processing.config.factories import VOXTRAL_MODEL_ID, create_mmar_config
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
