import os
import sys
from datetime import datetime

from loguru import logger

from speech_processing.runners.base import ALL_TIERS, experiment_keys, parse_args
from speech_processing.runners.icbhi import ExperimentVersion as IcbhiExperimentVersion
from speech_processing.runners.icbhi import run_icbhi
from speech_processing.runners.mmar import ExperimentVersion as MmarExperimentVersion
from speech_processing.runners.mmar import run_mmar
from speech_processing.utils.exceptions import SpeechProcessingError

LOGS_DIR = "logs"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
RUNNERS = {"icbhi": run_icbhi, "mmar": run_mmar}
REGISTRIES = {"icbhi": IcbhiExperimentVersion, "mmar": MmarExperimentVersion}


def main() -> None:
    args = parse_args("Master Runner for Speech Processing Pipelines")

    if args.list_experiments:
        tier = None if args.list_experiments == ALL_TIERS else args.list_experiments
        for key in experiment_keys(REGISTRIES[args.dataset], tier):
            sys.stdout.write(f"{key}\n")
        return

    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"{args.dataset}_{args.experiment}_{timestamp}.log")
    logger.add(log_file, format=LOG_FORMAT, level="INFO")

    logger.info(f"Starting execution for dataset: {args.dataset.upper()} (Logging to {log_file})")

    runner = RUNNERS.get(args.dataset)
    if runner is None:
        logger.error(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    try:
        runner(args)
    except SpeechProcessingError as e:
        logger.exception(f"CRITICAL PIPELINE FAILURE: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
