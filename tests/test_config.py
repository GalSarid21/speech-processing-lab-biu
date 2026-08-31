from assertpy import assert_that

from speech_processing.config.core import AppConfig


def test_app_config_initialization():
    config = AppConfig()
    assert_that(config.judge.model_id).is_equal_to("Qwen/Qwen3.8-27B-FP8")
