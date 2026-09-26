import os
import hashlib
import librosa
import soundfile as sf

CACHE_DIR = "/tmp/audio_mono_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_safe_mono_audio_path(path: str) -> str:
    if path.startswith("http"):
        return path
        
    abs_path = os.path.abspath(path)
    
    # Fast path: check if it's already mono without loading the whole file?
    # Actually, we can just load it. librosa.load defaults to mono=True.
    path_hash = hashlib.md5(abs_path.encode()).hexdigest()
    basename = os.path.basename(path)
    cached_path = os.path.join(CACHE_DIR, f"{path_hash}_{basename}")
    
    if not os.path.exists(cached_path):
        y, sr = librosa.load(abs_path, sr=16000, mono=True)
        sf.write(cached_path, y, sr)
        
    return f"file://{cached_path}"
