import os
from argparse import Namespace

import pytest
from assertpy import assert_that

from speech_processing.data.dtos import AudioRequest, ItemMetadata, TextRequest
from speech_processing.data.icbhi import ICBHI_LABELS
from speech_processing.models.preprocessing import (
    BandpassNormalizePreprocessor,
    BandpassNormalizeShiftPreprocessor,
    IdentityPreprocessor,
    SilenceReplacementPreprocessor,
)
from speech_processing.models.presentation import (
    CyclePresentation,
    FullClipPresentation,
    TwoPassLocalizationPresentation,
)
from speech_processing.pipelines.aggregation import IdentityAggregator, ShuffledMajorityVoteAggregator
from speech_processing.pipelines.judge import EvaluationPipeline, JudgePipeline, ParserOnlyPipeline
from speech_processing.runners.icbhi import ExperimentVersion, run_icbhi

EVAL_ID = "audio1"
SHOT_ID = "audio2"
ACTIVE_KEYS = [v.name for v in ExperimentVersion if not v.value.deprecated]


def eval_metadata() -> ItemMetadata:
    return ItemMetadata(
        item_id=EVAL_ID,
        question="What is the patient's diagnosis?",
        choices=list(ICBHI_LABELS),
        category="COPD",
        sub_category="multichannel|trachea",
        cycle_spans=[(0.0, 1.0), (1.0, 2.0)],
    )


@pytest.fixture
def mocked_runner(mocker):
    def load_requests(config, meta=None):
        request_cls = TextRequest if meta and meta.text_only else AudioRequest
        kwargs = {"instruction": "Choose.", "metadata": eval_metadata()}
        if request_cls is AudioRequest:
            kwargs["audio_path"] = "audio1.wav"
        return [(request_cls(**kwargs), "A")]

    mocker.patch("speech_processing.runners.icbhi.load_icbhi_requests", side_effect=load_requests)
    mocker.patch(
        "speech_processing.runners.icbhi.build_icbhi_few_shot_pool",
        return_value={SHOT_ID: mocker.MagicMock()},
    )
    mocker.patch(
        "speech_processing.runners.icbhi.build_icbhi_neighbor_pool",
        return_value={SHOT_ID: mocker.MagicMock()},
    )
    mocker.patch("speech_processing.runners.icbhi.load_near_duplicates", return_value={})
    mocker.patch("speech_processing.runners.icbhi._load_icbhi_rag_mapping", return_value={EVAL_ID: [SHOT_ID]})
    mocker.patch(
        "speech_processing.runners.icbhi.load_respiratory_evidence",
        return_value={EVAL_ID: mocker.MagicMock(cycle_spans=[(0.0, 1.0)])},
    )

    patched = {
        "voxtral": mocker.patch("speech_processing.runners.icbhi.VoxtralAudioEngine", autospec=True),
        "gemma": mocker.patch("speech_processing.runners.icbhi.GemmaTextModel", autospec=True),
        "judge": mocker.patch("speech_processing.runners.icbhi.QwenJudge", autospec=True),
        "inference": mocker.patch("speech_processing.runners.icbhi.InferencePipeline", autospec=True),
        "judge_pipeline": mocker.patch("speech_processing.runners.icbhi.JudgePipeline", autospec=True),
        "parser_pipeline": mocker.patch("speech_processing.runners.icbhi.ParserOnlyPipeline", autospec=True),
    }

    mocker.patch("speech_processing.runners.icbhi.release_vram")
    mocker.patch("speech_processing.runners.icbhi.time.sleep")
    return patched


def build_args(experiment: str, output_dir, runs: int = 1, no_judge: bool = True) -> Namespace:
    return Namespace(
        dataset="icbhi",
        experiment=experiment,
        output_dir=str(output_dir),
        num_samples=None,
        sample_ids_file=None,
        runs=runs,
        no_judge=no_judge,
    )


@pytest.mark.parametrize("key", ACTIVE_KEYS)
def test_every_experiment_runs(mocked_runner, tmp_path, key):
    run_icbhi(build_args(key, tmp_path))
    assert_that(os.listdir(tmp_path)).is_not_empty()


@pytest.mark.parametrize(
    "key, presentation_cls, preprocessor_cls, expects_scorer, aggregator_cls",
    [
        ("r0", FullClipPresentation, IdentityPreprocessor, False, IdentityAggregator),
        ("c1", FullClipPresentation, SilenceReplacementPreprocessor, False, IdentityAggregator),
        ("a3a", FullClipPresentation, BandpassNormalizePreprocessor, False, IdentityAggregator),
        ("a3b", FullClipPresentation, BandpassNormalizeShiftPreprocessor, False, IdentityAggregator),
        ("a4a", CyclePresentation, IdentityPreprocessor, False, IdentityAggregator),
        ("a4b", TwoPassLocalizationPresentation, IdentityPreprocessor, False, IdentityAggregator),
        ("a5c", FullClipPresentation, IdentityPreprocessor, True, IdentityAggregator),
        ("a7a", FullClipPresentation, IdentityPreprocessor, False, ShuffledMajorityVoteAggregator),
    ],
)
def test_engine_wiring(
    mocked_runner, tmp_path, key, presentation_cls, preprocessor_cls, expects_scorer, aggregator_cls
):
    run_icbhi(build_args(key, tmp_path))

    kwargs = mocked_runner["voxtral"].call_args.kwargs
    assert_that(kwargs["presentation"]).is_instance_of(presentation_cls)
    assert_that(kwargs["preprocessor"]).is_instance_of(preprocessor_cls)
    assert_that(kwargs["scorer"] is not None).is_equal_to(expects_scorer)
    assert_that(mocked_runner["inference"].call_args.args[1]).is_instance_of(aggregator_cls)


def test_content_free_calibration_reaches_the_scorer(mocked_runner, tmp_path):
    run_icbhi(build_args("a5c", tmp_path))
    assert_that(mocked_runner["voxtral"].call_args.kwargs["scorer"].calibration).is_equal_to("content_free")


@pytest.mark.parametrize("key", ["c2", "a1c"])
def test_text_only_experiments_never_build_voxtral(mocked_runner, tmp_path, key):
    run_icbhi(build_args(key, tmp_path))

    mocked_runner["gemma"].assert_called_once()
    mocked_runner["voxtral"].assert_not_called()


@pytest.mark.parametrize("key, expected_runs", [("r0", 3), ("a5a", 1)])
def test_deterministic_experiments_collapse_repeated_runs(mocked_runner, tmp_path, key, expected_runs):
    run_icbhi(build_args(key, tmp_path, runs=3))
    assert_that(mocked_runner["inference"].return_value.run.call_count).is_equal_to(expected_runs)


@pytest.mark.parametrize("key", ["r6", "a6b"])
def test_demonstrations_are_attached(mocked_runner, tmp_path, key):
    run_icbhi(build_args(key, tmp_path))

    requests = mocked_runner["inference"].return_value.run.call_args.args[0]
    assert_that(requests[0].few_shot_turns).is_length(1)


@pytest.mark.parametrize(
    "no_judge, expected_pipeline, unexpected_pipeline",
    [(True, "parser_pipeline", "judge_pipeline"), (False, "judge_pipeline", "parser_pipeline")],
)
def test_no_judge_flag_selects_the_evaluation_pipeline(
    mocked_runner, tmp_path, no_judge, expected_pipeline, unexpected_pipeline
):
    run_icbhi(build_args("r0", tmp_path, no_judge=no_judge))

    mocked_runner[expected_pipeline].assert_called_once()
    mocked_runner[unexpected_pipeline].assert_not_called()


@pytest.mark.parametrize("pipeline_cls", [JudgePipeline, ParserOnlyPipeline])
def test_both_evaluation_pipelines_share_the_same_contract(pipeline_cls):
    assert_that(issubclass(pipeline_cls, EvaluationPipeline)).is_true()
