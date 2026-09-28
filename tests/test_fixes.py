import numpy as np
import pytest
from speech_processing.config.core import ExperimentMeta
from speech_processing.data.dataset import load_mmar_requests
from speech_processing.config.core import DatasetConfig
import pandas as pd
from pydantic import BaseModel
from speech_processing.data.dtos.requests import TextRequest
from speech_processing.data.dtos.responses import MMAREvaluationResult
from speech_processing.data.dtos.responses import TextResponse
from speech_processing.models.text import GemmaTextModel
import numpy

def test_b1_explicit_flags():
    # Ensuring no substring logic needed
    meta = ExperimentMeta(
        experiment_name="few_shot_text_only_test",
        prompt="Test",
        max_new_tokens=100,
        batch_size=8,
        use_transcript=False,
        few_shot_mode="text"
    )
    assert meta.use_transcript is False
    assert meta.few_shot_mode == "text"

def test_b5_choices_numpy():
    # Numpy choices should be converted
    row = pd.Series({
        'id': 'test1',
        'question': 'What?',
        'choices': np.array(['apple', 'banana']),
        'answer': 'banana',
        'audio_path': 'fake.wav'
    })
    
    choices_list = list(row['choices'])
    assert type(choices_list) == list
    assert choices_list == ['apple', 'banana']
    
    lettered = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices_list)])
    assert "A. apple" in lettered
    assert "B. banana" in lettered
    
    ans = row['answer']
    if ans in choices_list:
        ans = chr(65 + choices_list.index(ans))
    assert ans == "B"

def test_b2_sample_id():
    req = TextRequest(instruction="test", metadata={"item_id": "item123"})
    assert req.metadata["item_id"] == "item123"

def test_b7_judge_schema():
    eval_res = MMAREvaluationResult(reasoning="Because.", is_correct=True, extracted_choice="A")
    assert eval_res.extracted_choice == "A"
    assert eval_res.is_correct is True

