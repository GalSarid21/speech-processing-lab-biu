import os
import tarfile
import argparse
from huggingface_hub import hf_hub_download

def main():
    parser = argparse.ArgumentParser(description="Download and optionally extract a file from a HuggingFace Dataset repository.")
    parser.add_argument("--repo-id", type=str, required=True, help="The HuggingFace repository ID (e.g., BoJack/MMAR)")
    parser.add_argument("--filename", type=str, required=True, help="The specific file to download (e.g., mmar-audio.tar.gz)")
    parser.add_argument("--extract-dir", type=str, default=None, help="If provided and file is a tar.gz, extract to this directory")
    args = parser.parse_args()
    
    print(f"Downloading {args.filename} from {args.repo_id}...")
    try:
        file_path = hf_hub_download(
            repo_id=args.repo_id, 
            filename=args.filename, 
            repo_type="dataset", 
            local_dir="data", 
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded to {file_path}")
        
        if args.extract_dir and file_path.endswith(".tar.gz"):
            os.makedirs(args.extract_dir, exist_ok=True)
            print(f"Extracting to {args.extract_dir}...")
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=args.extract_dir)
            print("Extraction complete.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
