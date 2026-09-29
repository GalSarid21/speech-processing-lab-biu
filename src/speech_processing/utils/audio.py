import hashlib
import math
import os

import librosa
import numpy as np
import soundfile as sf

from speech_processing.utils.consts import MONO_AUDIO_CACHE_DIR

REMOTE_AUDIO_PREFIX = "http"
FILE_URL_PREFIX = "file://"

WHITE_NOISE = "white_noise"
PINK_NOISE = "pink_noise"
# -40 dBFS, the level the contextual-calibration controls are specified at.
NOISE_PEAK_AMPLITUDE = 0.01


def force_mono_audio(path: str, target_sr: int) -> str:
    """Converts audio to mono at `target_sr`, caching the result, and returns a file:// URL.

    Required by engines (Voxtral/Mistral) whose internal tokenizers crash on multi-channel audio.
    """
    if path.startswith(REMOTE_AUDIO_PREFIX):
        return path

    abs_path = os.path.abspath(path)
    os.makedirs(MONO_AUDIO_CACHE_DIR, exist_ok=True)

    path_hash = hashlib.md5(abs_path.encode()).hexdigest()
    cached_path = os.path.join(MONO_AUDIO_CACHE_DIR, f"{path_hash}_{target_sr}_{os.path.basename(path)}")

    if not os.path.exists(cached_path):
        waveform, sample_rate = librosa.load(abs_path, sr=target_sr, mono=True)
        sf.write(cached_path, waveform, sample_rate)

    return f"{FILE_URL_PREFIX}{cached_path}"


def get_duration_s(path: str) -> float:
    return float(sf.info(path).duration)


def write_silence(duration_s: float, target_sr: int, output_dir: str) -> str:
    num_samples = max(1, round(duration_s * target_sr))
    output_path = os.path.join(output_dir, f"silence_{duration_s:.3f}s_{target_sr}.wav")
    sf.write(output_path, np.zeros(num_samples, dtype=np.float32), target_sr)
    return output_path


def write_audio_bytes(audio_bytes: bytes, name: str, output_dir: str) -> str:
    """Materialises in-memory audio (as HuggingFace hands it over) to a file an engine can reference."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, name)
    if not os.path.exists(output_path):
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
    return output_path


def write_noise(duration_s: float, target_sr: int, output_dir: str, color: str, seed: int) -> str:
    """Writes a content-free reference signal used to measure the model's label prior."""
    num_samples = max(1, round(duration_s * target_sr))
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(num_samples)

    if color == PINK_NOISE:
        spectrum = np.fft.rfft(samples)
        frequencies = np.arange(1, spectrum.size + 1, dtype=float)
        samples = np.fft.irfft(spectrum / np.sqrt(frequencies), n=num_samples)

    peak = float(np.max(np.abs(samples))) or 1.0
    samples = (samples / peak * NOISE_PEAK_AMPLITUDE).astype(np.float32)

    output_path = os.path.join(output_dir, f"{color}_{duration_s:.3f}s_{target_sr}.wav")
    sf.write(output_path, samples, target_sr)
    return output_path


def crop_audio(path: str, start_s: float, end_s: float, output_dir: str) -> str:
    waveform, sample_rate = sf.read(path)
    start_sample = max(0, int(start_s * sample_rate))
    end_sample = min(len(waveform), int(end_s * sample_rate))
    output_path = os.path.join(output_dir, f"crop_{start_s:.3f}_{end_s:.3f}_{os.path.basename(path)}")
    sf.write(output_path, waveform[start_sample:end_sample], sample_rate)
    return output_path


def chunk_audio(path: str, chunk_len_s: float, max_chunks: int, output_dir: str) -> list[tuple[str, float, float]]:
    """Splits the clip into equal-length chunks covering the WHOLE recording, never more than `max_chunks`."""
    waveform, sample_rate = sf.read(path)
    total_samples = len(waveform)
    duration_s = total_samples / sample_rate

    num_chunks = max(1, min(max_chunks, math.ceil(duration_s / chunk_len_s)))
    basename = os.path.basename(path)

    chunks: list[tuple[str, float, float]] = []
    for index in range(num_chunks):
        start_sample = (total_samples * index) // num_chunks
        end_sample = (total_samples * (index + 1)) // num_chunks
        output_path = os.path.join(output_dir, f"chunk_{index}_of_{num_chunks}_{basename}")
        sf.write(output_path, waveform[start_sample:end_sample], sample_rate)
        chunks.append((output_path, start_sample / sample_rate, end_sample / sample_rate))

    return chunks
