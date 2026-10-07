import argparse
import json
import os
from datetime import UTC, datetime
from enum import Enum

from speech_processing.config.core import AppConfig, ExperimentMeta, ExperimentTier
from speech_processing.config.factories import DEFAULT_GPU_PCT, DEFAULT_MAX_SEQS

ALL_TIERS = "all"
RUN_CONFIG_FILE = "run_config.json"


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["icbhi", "mmar"], help="Which dataset pipeline to run"
    )
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save artifacts")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of dataset samples to evaluate")
    parser.add_argument("--experiment", type=str, default="v1", help="Which experiment version to run")
    parser.add_argument("--sample-ids-file", type=str, default=None, help="Path to a previous raw_output.jsonl file")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to run the experiment")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Score closed-set answers with the deterministic parser only, without loading the judge model",
    )
    parser.add_argument(
        "--audio-model-id",
        type=str,
        default=None,
        help="Override the audio model, e.g. mistralai/Voxtral-Mini-3B-2507 to validate the pipeline on a "
        "GPU too small for the 24B default. Scores are not comparable across models.",
    )
    parser.add_argument("--text-model-id", type=str, default=None, help="Override the text-only model")
    parser.add_argument(
        "--max-seqs", type=int, default=DEFAULT_MAX_SEQS, help="vLLM max_num_seqs (engine-side batch size)"
    )
    parser.add_argument("--gpu-pct", type=float, default=DEFAULT_GPU_PCT, help="Fraction of GPU memory vLLM may use")
    parser.add_argument(
        "--list-experiments",
        type=str,
        nargs="?",
        const=ALL_TIERS,
        default=None,
        choices=["core", "future_work", ALL_TIERS],
        help="Print the runnable experiment keys of a tier, one per line, then exit",
    )
    return parser.parse_args()


def experiment_keys(registry: type[Enum], tier: ExperimentTier | None = None) -> list[str]:
    """Runnable experiment keys, optionally restricted to one tier.

    Deprecated entries are never runnable, so they belong to no tier.
    """
    return [
        member.name
        for member in registry
        if not member.value.deprecated and (tier is None or member.value.tier == tier)
    ]


def write_run_config(run_dir: str, args: argparse.Namespace, config: AppConfig, meta: ExperimentMeta) -> None:
    """Records which models produced a run folder, so results of different models are never mixed up."""
    record = {
        "dataset": args.dataset,
        "experiment": args.experiment,
        "experiment_name": meta.experiment_name,
        "audio_model_id": config.audio_model.model_id if config.audio_model else None,
        "text_model_id": config.text_model.model_id if config.text_model else None,
        "judge_model_id": config.judge.model_id if config.judge and not getattr(args, "no_judge", False) else None,
        "runs_requested": args.runs,
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with open(os.path.join(run_dir, RUN_CONFIG_FILE), "w") as f:
        json.dump(record, f, indent=2)
