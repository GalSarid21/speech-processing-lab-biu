from assertpy import assert_that
from types import SimpleNamespace

from speech_processing.config.factories import create_mmar_config
from speech_processing.config.core import ExperimentMeta


def test_factories_initialization():
    args = SimpleNamespace(
        num_samples=5,
        sample_ids_file=None,
        output_dir="./test_results",
        gpu_pct=0.8,
        max_seqs=32
    )
    meta = ExperimentMeta(
        experiment_name="test_cot",
        prompt="Test",
        max_new_tokens=1024,
        batch_size=8
    )
    
    config = create_mmar_config(args, meta)
    
    assert_that(config.output_dir).is_equal_to("./test_results")
    assert_that(config.audio_model).is_not_none()
    assert_that(config.audio_model.model_id).is_equal_to("mistralai/Voxtral-Small-24B-2507")
    assert_that(config.audio_model.gpu_memory_utilization).is_equal_to(0.8)
