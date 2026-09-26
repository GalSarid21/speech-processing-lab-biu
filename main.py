import sys
from loguru import logger
from speech_processing.runners.base import parse_args
from speech_processing.runners.icbhi import run_icbhi
from speech_processing.runners.mmar import run_mmar

def main():
    args = parse_args("Master Runner for Speech Processing Pipelines")
    
    import os
    from datetime import datetime
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{args.dataset}_{args.experiment}_{timestamp}.log"
    logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}", level="INFO")
    
    logger.info(f"Starting execution for dataset: {args.dataset.upper()} (Logging to {log_file})")
    
    try:
        if args.dataset == "icbhi":
            run_icbhi(args)
        elif args.dataset == "mmar":
            run_mmar(args)
        else:
            logger.error(f"Unknown dataset: {args.dataset}")
            sys.exit(1)
    
    except Exception as e:
        logger.exception(f"CRITICAL PIPELINE FAILURE: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
