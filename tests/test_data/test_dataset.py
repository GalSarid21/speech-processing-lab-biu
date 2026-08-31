from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from assertpy import assert_that

from speech_processing.config.core import DatasetConfig
from speech_processing.data.dataset import load_icbhi_requests


@pytest.fixture
def mock_dataset_config():
    return DatasetConfig(
        dataset_id="mock/dataset",
        target_labels=["COPD", "Healthy"],
        split="test",
    )


@patch("speech_processing.data.dataset.load_dataset")
def test_dataset_filtering_and_padding(mock_load_dataset, mock_dataset_config):
    # Create a mock huggingface dataset that returns a specific pandas dataframe
    mock_ds = MagicMock()
    
    # 2 COPD, 1 Healthy, 2 Asthma (which should be filtered out)
    df = pd.DataFrame({
        "instruction": ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"],
        "file": ["f1.wav", "f2.wav", "f3.wav", "f4.wav", "f5.wav"],
        "label": ["COPD", "Asthma", "Healthy", "COPD", "Asthma"],
        "audio": [{"bytes": b"1"}, {"bytes": b"2"}, {"bytes": b"3"}, {"bytes": b"4"}, {"bytes": b"5"}]
    })
    
    mock_ds.cast_column.return_value = mock_ds
    mock_ds.to_pandas.return_value = df
    mock_load_dataset.return_value = mock_ds

    # We ask for 5 samples, but only 3 are valid (COPD, Healthy, COPD).
    # It should pad the remaining 2 with disjoint samples (the 2 Asthma ones).
    mock_dataset_config.num_samples = 5
    results = load_icbhi_requests(config=mock_dataset_config)
    
    assert_that(results).is_length(5)
    
    # Ensure all elements are AudioRequest DTOs and string ground truths
    for req, label in results:
        assert_that(req.instruction).starts_with("prompt")
        assert_that(req.audio_path).ends_with(".wav")
        assert_that(req.audio_bytes).is_not_none()
        assert_that(label).is_in("COPD", "Healthy", "Asthma")
    
    # Extract the labels that were returned
    returned_labels = [gt for _, gt in results]
    assert_that(returned_labels.count("COPD")).is_equal_to(2)
    assert_that(returned_labels.count("Healthy")).is_equal_to(1)
    # The padding should have pulled the 2 Asthma samples
    assert_that(returned_labels.count("Asthma")).is_equal_to(2)


@patch("speech_processing.data.dataset.load_dataset")
def test_dataset_no_padding_needed(mock_load_dataset, mock_dataset_config):
    mock_ds = MagicMock()
    df = pd.DataFrame({
        "instruction": ["p1", "p2", "p3", "p4"],
        "file": ["1.wav", "2.wav", "3.wav", "4.wav"],
        "label": ["COPD", "COPD", "Healthy", "Healthy"],
        "audio": [{"bytes": b"1"}, {"bytes": b"2"}, {"bytes": b"3"}, {"bytes": b"4"}]
    })
    mock_ds.cast_column.return_value = mock_ds
    mock_ds.to_pandas.return_value = df
    mock_load_dataset.return_value = mock_ds

    # We only ask for 2 samples out of 4 valid ones
    mock_dataset_config.num_samples = 2
    results = load_icbhi_requests(config=mock_dataset_config)
    
    assert_that(results).is_length(2)
    returned_labels = [gt for _, gt in results]
    for label in returned_labels:
        assert_that(label).is_in("COPD", "Healthy")
