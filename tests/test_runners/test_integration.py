import os
from argparse import Namespace

import pytest
from assertpy import assert_that

from speech_processing.data.dtos import AudioRequest, ItemMetadata, TextRequest
from speech_processing.models.presentation import (
    ChunkedPresentation,
    FullClipPresentation,
    TwoPassLocalizationPresentation,
)
from speech_processing.pipelines.aggregation import IdentityAggregator, ShuffledMajorityVoteAggregator
from speech_processing.runners.mmar import ExperimentVersion, run_mmar
from speech_processing.utils.exceptions import DeprecatedExperimentError

EVAL_ID = "EVAL1"
SHOT_ID = "SHOT1"
ACTIVE_KEYS = [v.name for v in ExperimentVersion if not v.value.deprecated]
DEPRECATED_KEYS = [v.name for v in ExperimentVersion if v.value.deprecated]


def eval_metadata() -> ItemMetadata:
    return ItemMetadata(item_id=EVAL_ID, question="q", choices=["Dog", "Cat"], transcript="t")


@pytest.fixture
def mocked_runner(mocker):
    """Patches every heavy dependency so the runner can be exercised end to end."""

    def load_requests(config, experiment_meta=None, is_text_only=False):
        request_cls = TextRequest if is_text_only else AudioRequest
        kwargs = {"instruction": "Answer.", "metadata": eval_metadata()}
        if not is_text_only:
            kwargs["audio_path"] = "dummy.wav"
        return [(request_cls(**kwargs), "B")]

    mocker.patch("speech_processing.runners.mmar.load_mmar_requests", side_effect=load_requests)
    mocker.patch(
        "speech_processing.runners.mmar.get_mmar_few_shot_turns",
        return_value=([SHOT_ID], [mocker.MagicMock()]),
    )
    mocker.patch(
        "speech_processing.runners.mmar.get_mmar_few_shot_turns_pool",
        return_value={SHOT_ID: mocker.MagicMock()},
    )
    mocker.patch("speech_processing.runners.mmar._load_rag_mapping", return_value={EVAL_ID: [SHOT_ID]})

    mocker.patch(
        "speech_processing.runners.icbhi.load_icbhi_requests",
        return_value=[(AudioRequest(instruction="Answer.", audio_path="dummy.wav"), "Healthy")],
    )

    patched = {
        "voxtral": mocker.patch("speech_processing.runners.mmar.VoxtralAudioEngine", autospec=True),
        "gemma": mocker.patch("speech_processing.runners.mmar.GemmaTextModel", autospec=True),
        "judge": mocker.patch("speech_processing.runners.mmar.QwenJudge", autospec=True),
        "inference": mocker.patch("speech_processing.runners.mmar.InferencePipeline", autospec=True),
        "judge_pipeline": mocker.patch("speech_processing.runners.mmar.JudgePipeline", autospec=True),
        "parser_pipeline": mocker.patch("speech_processing.runners.mmar.ParserOnlyPipeline", autospec=True),
    }

    mocker.patch("speech_processing.runners.mmar.release_vram")
    mocker.patch("speech_processing.runners.mmar.time.sleep")

    return patched


def build_args(experiment: str, output_dir, runs: int = 1, dataset: str = "mmar", no_judge: bool = False) -> Namespace:
    return Namespace(
        dataset=dataset,
        experiment=experiment,
        output_dir=str(output_dir),
        num_samples=1,
        sample_ids_file=None,
        runs=runs,
        no_judge=no_judge,
    )


@pytest.mark.parametrize("key", ACTIVE_KEYS)
def test_every_active_version_runs(mocked_runner, tmp_path, key):
    run_mmar(build_args(key, tmp_path))
    assert_that(os.listdir(tmp_path)).is_not_empty()


@pytest.mark.parametrize("key", DEPRECATED_KEYS)
def test_deprecated_versions_raise(mocked_runner, tmp_path, key):
    assert_that(run_mmar).raises(DeprecatedExperimentError).when_called_with(build_args(key, tmp_path))


@pytest.mark.parametrize(
    "key, presentation_cls, expects_scorer, aggregator_cls",
    [
        ("v1", FullClipPresentation, False, IdentityAggregator),
        ("v22", ChunkedPresentation, False, IdentityAggregator),
        ("v23", TwoPassLocalizationPresentation, False, IdentityAggregator),
        ("v25", FullClipPresentation, True, IdentityAggregator),
        ("v27", FullClipPresentation, False, ShuffledMajorityVoteAggregator),
    ],
)
def test_engine_wiring(mocked_runner, tmp_path, key, presentation_cls, expects_scorer, aggregator_cls):
    run_mmar(build_args(key, tmp_path))

    voxtral_kwargs = mocked_runner["voxtral"].call_args.kwargs
    assert_that(voxtral_kwargs["presentation"]).is_instance_of(presentation_cls)
    assert_that(voxtral_kwargs["scorer"] is not None).is_equal_to(expects_scorer)
    assert_that(mocked_runner["inference"].call_args.args[1]).is_instance_of(aggregator_cls)


@pytest.mark.parametrize("key", ["v13", "v21"])
def test_text_only_versions_never_build_voxtral(mocked_runner, tmp_path, key):
    run_mmar(build_args(key, tmp_path))

    mocked_runner["gemma"].assert_called_once()
    mocked_runner["voxtral"].assert_not_called()


@pytest.mark.parametrize("key, expected_runs", [("v1", 3), ("v24", 1)])
def test_deterministic_experiments_collapse_repeated_runs(mocked_runner, tmp_path, key, expected_runs):
    run_mmar(build_args(key, tmp_path, runs=3))
    assert_that(mocked_runner["inference"].return_value.run.call_count).is_equal_to(expected_runs)


@pytest.mark.parametrize("key", ["v7", "v14"])
def test_few_shots_are_attached(mocked_runner, tmp_path, key):
    run_mmar(build_args(key, tmp_path))

    requests = mocked_runner["inference"].return_value.run.call_args.args[0]
    assert_that(requests[0].few_shot_turns).is_length(1)


@pytest.mark.parametrize(
    "no_judge, expected_pipeline, unexpected_pipeline",
    [(True, "parser_pipeline", "judge_pipeline"), (False, "judge_pipeline", "parser_pipeline")],
)
def test_no_judge_flag_selects_the_evaluation_pipeline(
    mocked_runner, tmp_path, no_judge, expected_pipeline, unexpected_pipeline
):
    """A server-hosted audio model holds the GPU, so --no-judge must keep the judge from loading at all."""
    run_mmar(build_args("v1", tmp_path, no_judge=no_judge))

    mocked_runner[expected_pipeline].assert_called_once()
    mocked_runner[unexpected_pipeline].assert_not_called()
    if no_judge:
        mocked_runner["judge"].assert_not_called()
