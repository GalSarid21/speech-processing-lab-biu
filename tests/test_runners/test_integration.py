import os
from argparse import Namespace
import pytest
from assertpy import assert_that

from speech_processing.runners.mmar import run_mmar
from speech_processing.runners.icbhi import run_icbhi
from speech_processing.data.dtos import AudioRequest

@pytest.fixture
def mock_pipeline_dependencies(mocker):
    """Mocks out heavy models and datasets to allow lightning-fast end-to-end runner testing."""
    # 1. Mock the dataset loaders to return a single dummy item (prevents HuggingFace downloads)
    dummy_req = AudioRequest(instruction="Test", audio_path="dummy.wav", audio_bytes=b"0000")
    mocker.patch("speech_processing.runners.mmar.load_mmar_requests", return_value=[(dummy_req, "Answer")])
    mocker.patch("speech_processing.runners.icbhi.load_icbhi_requests", return_value=[(dummy_req, "Healthy")])
    
    # 2. Mock the Engines WITH autospec=True (This enforces signature checking!)
    mock_audio = mocker.patch("speech_processing.runners.mmar.QwenAudioEngine", autospec=True)
    mock_text = mocker.patch("speech_processing.runners.mmar.GemmaTextModel", autospec=True)
    mock_judge = mocker.patch("speech_processing.runners.mmar.QwenJudge", autospec=True)
    
    # 3. Mock the Inference/Judge Pipelines to return dummy responses
    mock_inf_pipe = mocker.patch("speech_processing.runners.mmar.InferencePipeline", autospec=True)
    mock_inf_pipe.return_value.run.return_value = (["temp.jsonl"], [["Final Diagnosis: Healthy"]])
    
    mock_judge_pipe = mocker.patch("speech_processing.runners.mmar.JudgePipeline", autospec=True)
    
    # Do the same for ICBHI runner
    mocker.patch("speech_processing.runners.icbhi.QwenAudioEngine", autospec=True)
    mocker.patch("speech_processing.runners.icbhi.GemmaTextModel", autospec=True)
    mocker.patch("speech_processing.runners.icbhi.QwenJudge", autospec=True)
    mocker.patch("speech_processing.runners.icbhi.InferencePipeline", autospec=True).return_value.run.return_value = (["temp.jsonl"], [["Final Diagnosis: Healthy"]])
    mocker.patch("speech_processing.runners.icbhi.JudgePipeline", autospec=True)
    
    # Mock VRAM clearance to save time
    mocker.patch("speech_processing.runners.mmar.release_vram")
    mocker.patch("speech_processing.runners.mmar.time.sleep")
    mocker.patch("speech_processing.runners.icbhi.release_vram")
    mocker.patch("speech_processing.runners.icbhi.time.sleep")


def test_mmar_runner_end_to_end(mock_pipeline_dependencies, tmp_path):
    """Executes the MMAR runner start-to-finish to catch integration or signature errors."""
    args = Namespace(
        dataset="mmar",
        experiment="v1",
        output_dir=str(tmp_path),
        num_samples=1,
        sample_ids_file=None,
        runs=1
    )
    
    # If the orchestrator passes a wrong kwarg to an engine (like template= instead of template_func=), 
    # the autospec=True on the mock will instantly raise a TypeError here.
    run_mmar(args)
    
    # Verify the output directory was created
    assert_that(os.listdir(tmp_path)).is_not_empty()


def test_icbhi_runner_end_to_end(mock_pipeline_dependencies, tmp_path):
    """Executes the ICBHI runner start-to-finish to catch integration or signature errors."""
    args = Namespace(
        dataset="icbhi",
        experiment="v1",
        output_dir=str(tmp_path),
        num_samples=1,
        sample_ids_file=None,
        runs=1
    )
    
    run_icbhi(args)
    
    assert_that(os.listdir(tmp_path)).is_not_empty()
