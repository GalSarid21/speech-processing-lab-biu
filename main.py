import sys
from loguru import logger
from speech_processing.runners.base import parse_args
from speech_processing.runners.icbhi import run_icbhi
from speech_processing.runners.mmar import run_mmar

def main():
    args = parse_args("Master Runner for Speech Processing Pipelines")
    
    logger.info(f"Starting execution for dataset: {args.dataset.upper()}")
    
    if args.dataset == "icbhi":
        run_icbhi(args)
    elif args.dataset == "mmar":
        run_mmar(args)
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

if __name__ == "__main__":
    main()
