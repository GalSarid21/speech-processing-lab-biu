import argparse
import json
import os
from loguru import logger
from speech_processing.config.core import AppConfig, DatasetConfig, TextModelConfig, AudioModelConfig, JudgeConfig, ExperimentMeta

def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", type=str, required=True, choices=["icbhi", "mmar"], help="Which dataset pipeline to run")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save artifacts")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of dataset samples to evaluate")
    parser.add_argument("--experiment", type=str, default="v1", help="Which experiment version to run")
    parser.add_argument("--sample-ids-file", type=str, default=None, help="Path to a previous raw_output.jsonl file")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to run the experiment")
    return parser.parse_args()


