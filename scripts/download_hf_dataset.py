import argparse
import os
import tarfile

from huggingface_hub import hf_hub_download
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Download and optionally extract a file from a HuggingFace Dataset repository."
    )
    parser.add_argument("--repo-id", type=str, required=True, help="The HuggingFace repository ID (e.g., BoJack/MMAR)")
    parser.add_argument(
        "--filename", type=str, required=True, help="The specific file to download (e.g., mmar-audio.tar.gz)"
    )
    parser.add_argument(
        "--extract-dir", type=str, default=None, help="If provided and file is a tar.gz, extract to this directory"
    )
    args = parser.parse_args()

    logger.info(f"Downloading {args.filename} from {args.repo_id}...")
    try:
        file_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            repo_type="dataset",
            local_dir="data",
            local_dir_use_symlinks=False,
        )
        logger.info(f"Successfully downloaded to {file_path}")

        if args.extract_dir and file_path.endswith(".tar.gz"):
            os.makedirs(args.extract_dir, exist_ok=True)
            logger.info(f"Extracting to {args.extract_dir}...")
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=args.extract_dir)
            logger.info("Extraction complete.")

    except (OSError, ValueError) as e:
        logger.error(e)


if __name__ == "__main__":
    main()
