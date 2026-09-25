
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
        num_samples=100
    )


def test_dataset_filtering_and_padding(mocker, mock_dataset_config):
    mock_load_dataset = mocker.patch("speech_processing.data.dataset.load_dataset")
    mock_ds = mocker.MagicMock()
    
    # 2 COPD, 1 Healthy, 2 FakeDisease (which should be filtered out)
    df = pd.DataFrame({
        "instruction": ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"],
        "file": ["f1.wav", "f2.wav", "f3.wav", "f4.wav", "f5.wav"],
        "label": ["COPD", "FakeDisease", "Healthy", "COPD", "FakeDisease"],
        "audio": [{"bytes": b"1"}, {"bytes": b"2"}, {"bytes": b"3"}, {"bytes": b"4"}, {"bytes": b"5"}]
    })
    
    mock_ds.cast_column.return_value = mock_ds
    mock_ds.to_pandas.return_value = df
    mock_load_dataset.return_value = mock_ds

    # We ask for 5 samples, but only 3 are valid (COPD, Healthy, COPD).
    # It should pad the remaining 2 with disjoint samples (the 2 FakeDisease ones).
    mock_dataset_config.num_samples = 5
    results = load_icbhi_requests(config=mock_dataset_config)
    
    assert_that(results).is_length(5)
    
    # Ensure all elements are AudioRequest DTOs and string ground truths
    for req, label in results:
        assert_that(req.instruction).starts_with("prompt")
        assert_that(req.audio_path).ends_with(".wav")
        assert_that(req.audio_bytes).is_not_none()
        assert_that(label).is_in("COPD", "Healthy", "FakeDisease")
    
    # Extract the labels that were returned
    returned_labels = [gt for _, gt in results]
    assert_that(returned_labels.count("COPD")).is_equal_to(2)
    assert_that(returned_labels.count("Healthy")).is_equal_to(1)
    # The padding should have pulled the 2 FakeDisease samples
    assert_that(returned_labels.count("FakeDisease")).is_equal_to(2)


def test_dataset_no_padding_needed(mocker, mock_dataset_config):
    mock_load_dataset = mocker.patch("speech_processing.data.dataset.load_dataset")
    mock_ds = mocker.MagicMock()
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

def test_load_mmar_requests(mocker, mock_dataset_config):
    from speech_processing.data.dataset import load_mmar_requests
    
    mock_load_dataset = mocker.patch("speech_processing.data.dataset.load_dataset")
    mock_ds = mocker.MagicMock()
    
    df = pd.DataFrame({
        "id": ["ID1", "ID2", "ID3"],
        "question": ["What sound is this?", "What accent?", "Is it loud?"],
        # MMAR stores choices as literal string representation of a list
        "choices": ["['Dog', 'Cat']", "['British', 'American']", "['Yes', 'No']"],
        "answer": ["Dog", "British", "Yes"],
        "audio_path": ["./audio/id1.wav", "./audio/id2.wav", "./audio/id3.wav"]
    })
    
    mock_ds.to_pandas.return_value = df
    mock_load_dataset.return_value = mock_ds
    
    # Pass the explicitly requested IDs and transcript path directly in the config
    mock_dataset_config.sample_ids = ["ID1", "ID3"]
    mock_dataset_config.transcripts_file = "dummy_transcripts.json"
    mock_dataset_config.audio_base_dir = "data/MMAR"
    
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("builtins.open", mocker.mock_open())
    
    # Mock json.load for transcripts
    mocker.patch("json.load", return_value={"ID1": "Bark", "ID2": "Hello mate"})

    results = load_mmar_requests(mock_dataset_config)

    # Should only return the 2 IDs we requested
    assert_that(results).is_length(2)
    
    # Check ID1 (Has transcript)
    req1, gt1 = results[0]
    assert_that(gt1).is_equal_to("Dog")
    assert_that(req1.audio_path).is_equal_to("data/MMAR/audio/id1.wav")
    assert_that(req1.metadata["question"]).is_equal_to("What sound is this?")
    assert_that(req1.metadata["choices"]).is_equal_to(["Dog", "Cat"]) # Evaluated list!
    assert_that(req1.metadata["transcript"]).is_equal_to("Bark")
    
    # Check ID3 (No transcript)
    req3, gt3 = results[1]
    assert_that(gt3).is_equal_to("Yes")
    assert_that(req3.metadata["transcript"]).is_equal_to("[NO TRANSCRIPT]")
    assert_that(req3.metadata["choices"]).is_equal_to(["Yes", "No"])
import sys
import os
import pytest
from assertpy import assert_that
import pandas as pd

from speech_processing.config.core import DatasetConfig
from speech_processing.data.dataset import load_mmar_requests
from speech_processing.runners.mmar import ExperimentVersion

def test_mmar_dynamic_prompt_injection(mocker):
    # We want to test the string formatting logic inside load_mmar_requests
    
    mock_config = DatasetConfig(
        dataset_id="test", 
        target_labels=["A", "B"], 
        split="test", 
        num_samples=1, 
        sample_ids=["1"],
        transcripts_file="dummy_transcripts.json"
    )
    
    # Mock huggingface load_dataset to return a tiny dataframe
    mock_df = pd.DataFrame([{
        "id": "1",
        "question": "What is this?",
        "choices": "['A', 'B']",
        "answer": "A",
        "audio_path": "./test.wav"
    }])
    
    mock_ds = mocker.MagicMock()
    mock_ds.to_pandas.return_value = mock_df
    mocker.patch("speech_processing.data.dataset.load_dataset", return_value=mock_ds)
    
    # Mock transcript reading
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("builtins.open", mocker.mock_open(read_data='{"1": "Hello world"}'))
    mocker.patch("json.load", return_value={"1": "Hello world"})
    
    # Run with V1 (Baseline - NO TRANSCRIPT)
    meta_v1 = ExperimentVersion.get_version("v1").value
    items_v1 = load_mmar_requests(mock_config, experiment_meta=meta_v1)
    req_v1, _ = items_v1[0]
    
    assert_that(req_v1.instruction).does_not_contain("Hello world")
    assert_that(req_v1.instruction).contains("What is this?")
    assert_that(req_v1.instruction).contains("['A', 'B']")
    
    # Run with V3 (Transcript Augmented)
    meta_v3 = ExperimentVersion.get_version("v3").value
    items_v3 = load_mmar_requests(mock_config, experiment_meta=meta_v3)
    req_v3, _ = items_v3[0]
    
    assert_that(req_v3.instruction).contains("Hello world")
    assert_that(req_v3.instruction).contains("Transcript: Hello world")
    assert_that(req_v3.instruction).contains("What is this?")

def test_dataset_num_samples_none(mocker, mock_dataset_config):
    mock_load_dataset = mocker.patch("speech_processing.data.dataset.load_dataset")
    mock_ds = mocker.MagicMock()
    df = pd.DataFrame({
        "instruction": ["p1", "p2", "p3", "p4", "p5"],
        "file": ["1.wav", "2.wav", "3.wav", "4.wav", "5.wav"],
        "label": ["COPD", "COPD", "Healthy", "Healthy", "COPD"],
        "audio": [{"bytes": b"1"}, {"bytes": b"2"}, {"bytes": b"3"}, {"bytes": b"4"}, {"bytes": b"5"}]
    })
    mock_ds.cast_column.return_value = mock_ds
    mock_ds.to_pandas.return_value = df
    mock_load_dataset.return_value = mock_ds

    # Set to None, should return all 5 valid samples
    mock_dataset_config.num_samples = None
    results = load_icbhi_requests(config=mock_dataset_config)
    assert_that(results).is_length(5)


def test_dataset_num_samples_larger_than_dataset(mocker, mock_dataset_config):
    mock_load_dataset = mocker.patch("speech_processing.data.dataset.load_dataset")
    mock_ds = mocker.MagicMock()
    df = pd.DataFrame({
        "instruction": ["p1", "p2", "p3"],
        "file": ["1.wav", "2.wav", "3.wav"],
        "label": ["COPD", "COPD", "Healthy"],
        "audio": [{"bytes": b"1"}, {"bytes": b"2"}, {"bytes": b"3"}]
    })
    mock_ds.cast_column.return_value = mock_ds
    mock_ds.to_pandas.return_value = df
    mock_load_dataset.return_value = mock_ds

    # Ask for 100, but only 3 exist (and no disjoint padding is possible since the whole dataset is 3)
    # Wait, the padding logic in ICBHI samples from the inverse subset.
    # If the total dataset has only 3 rows, it pads with up to `min(remaining, len(inverse))`.
    # So it will just return 3!
    mock_dataset_config.num_samples = 100
    results = load_icbhi_requests(config=mock_dataset_config)
    
    assert_that(results).is_length(3)


def test_mmar_num_samples_none_and_truncation(mocker, mock_dataset_config):
    from speech_processing.data.dataset import load_mmar_requests
    mock_load_dataset = mocker.patch("speech_processing.data.dataset.load_dataset")
    mock_ds = mocker.MagicMock()
    df = pd.DataFrame({
        "id": ["ID1", "ID2", "ID3", "ID4"],
        "question": ["Q", "Q", "Q", "Q"],
        "choices": ["['A']", "['A']", "['A']", "['A']"],
        "answer": ["A", "A", "A", "A"],
        "audio_path": ["p1", "p2", "p3", "p4"]
    })
    mock_ds.to_pandas.return_value = df
    mock_load_dataset.return_value = mock_ds
    
    mocker.patch("os.path.exists", return_value=False)

    # Test None -> returns all 4
    mock_dataset_config.num_samples = None
    results = load_mmar_requests(mock_dataset_config)
    assert_that(results).is_length(4)
    
    # Test 2 -> returns 2
    mock_dataset_config.num_samples = 2
    results = load_mmar_requests(mock_dataset_config)
    assert_that(results).is_length(2)
    
    # Test 100 -> returns all 4 safely with warning
    mock_dataset_config.num_samples = 100
    results = load_mmar_requests(mock_dataset_config)
    assert_that(results).is_length(4)
