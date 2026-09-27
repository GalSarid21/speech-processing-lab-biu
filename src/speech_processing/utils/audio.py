import soundfile as sf
import tempfile
import os
import numpy as np

def get_silence_audio(target_sr: int) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, np.zeros(target_sr, dtype=np.float32), target_sr)
    return path

def chunk_audio(audio_path: str, max_chunk_len_s: float = 5.0, max_chunks: int = 6) -> list[tuple[str, float, float]]:
    y, sr = sf.read(audio_path)
    total_len_s = len(y) / sr
    
    chunk_len_samples = int(max_chunk_len_s * sr)
    chunks = []
    
    for i in range(0, len(y), chunk_len_samples):
        if len(chunks) >= max_chunks:
            break
            
        chunk_y = y[i:i+chunk_len_samples]
        start_s = i / sr
        end_s = start_s + (len(chunk_y) / sr)
        
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(temp_path, chunk_y, sr)
        
        chunks.append((temp_path, start_s, end_s))
        
    return chunks

def crop_audio(audio_path: str, start_s: float, end_s: float) -> str:
    y, sr = sf.read(audio_path)
    start_sample = max(0, int(start_s * sr))
    end_sample = min(len(y), int(end_s * sr))
    
    y_cropped = y[start_sample:end_sample]
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(temp_path, y_cropped, sr)
    return temp_path
