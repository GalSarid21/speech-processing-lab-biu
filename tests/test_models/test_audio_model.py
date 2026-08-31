from unittest.mock import MagicMock, patch

import pytest
from assertpy import assert_that

from speech_processing.config.core import AudioModelConfig
from speech_processing.data.dtos import AudioRequest
from speech_processing.models.audio_model import QwenAudioEngine


@pytest.fixture
def mock_audio_config():
    return AudioModelConfig(
        model_id="mock/audio",
        dtype="float32",
        max_num_seqs=2,
        max_new_tokens=100,
    )


@patch("speech_processing.models.audio_model.Qwen2AudioForConditionalGeneration")
@patch("speech_processing.models.audio_model.AutoProcessor")
@patch("speech_processing.models.audio_model.TransformersAdapter")
@patch("speech_processing.models.audio_model.librosa")
@patch("speech_processing.models.audio_model.urlopen")
def test_audio_batch_infer(mock_urlopen, mock_librosa, mock_adapter_class, mock_processor_class, mock_model_class, mock_audio_config):
    # Setup mocks
    mock_urlopen.return_value.read.return_value = b"fake audio data"
    mock_librosa.load.return_value = (MagicMock(), 16000)
    mock_adapter = mock_adapter_class.return_value
    mock_adapter.generate_batch.return_value = ["mock answer 1", "mock answer 2"]
    
    # Initialize engine
    engine = QwenAudioEngine(config=mock_audio_config)
    
    # Create 2 requests
    requests = [
        AudioRequest(instruction="prompt1", audio_path="fake1.wav"),
        AudioRequest(instruction="prompt2", audio_path="http://fake2.wav")
    ]
    
    responses = engine.batch_infer(requests)
    
    assert_that(responses).is_length(2)
    assert_that(responses[0].generated_text).is_equal_to("mock answer 1")
    assert_that(responses[1].generated_text).is_equal_to("mock answer 2")
    
    # Ensure librosa loaded both files
    assert_that(mock_librosa.load.call_count).is_equal_to(2)
    
    # Ensure adapter generated exactly one batch (since max_num_seqs is 2)
    mock_adapter.generate_batch.assert_called_once()
