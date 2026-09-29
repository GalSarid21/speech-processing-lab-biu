"""A3: signal preprocessing applied to the clip before the audio-language model sees it.

Auscultation recordings sit almost entirely below ~2 kHz, are recorded at low level, and carry
heart sounds and room noise that web-scale audio encoders were never asked to ignore. These
strategies clean and, optionally, shift the recording towards the ranges the encoder saw in
training. The pitch-shifting variant is exploratory and must be reported as such.
"""

import abc
import os

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from speech_processing.config.core import AudioPreprocessingMode, ExperimentMeta
from speech_processing.utils.audio import get_duration_s, write_silence

BANDPASS_LOW_HZ = 100.0
BANDPASS_HIGH_HZ = 1800.0
BANDPASS_ORDER = 4
TARGET_PEAK = 0.7
PITCH_SHIFT_SEMITONES = 12.0
TIME_STRETCH_RATE = 0.75
EPSILON = 1e-10


def bandpass(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Zero-phase Butterworth band-pass that removes heart sounds below ~100 Hz and hiss above ~1.8 kHz."""
    nyquist = sample_rate / 2
    high = min(BANDPASS_HIGH_HZ, nyquist * 0.99)
    if BANDPASS_LOW_HZ >= high:
        return waveform
    sos = butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, waveform)


def peak_normalize(waveform: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(waveform)))
    if peak < EPSILON:
        return waveform
    return waveform / peak * TARGET_PEAK


class AudioPreprocessor(abc.ABC):
    """Turns a clip into the clip the engine should actually send."""

    def __init__(self, target_sr: int) -> None:
        self.target_sr = target_sr

    @abc.abstractmethod
    def process(self, path: str, work_dir: str) -> str:
        """Return the path of the audio to send. May be `path` itself."""

    def _write(self, waveform: np.ndarray, sample_rate: int, path: str, work_dir: str, tag: str) -> str:
        output_path = os.path.join(work_dir, f"{tag}_{os.path.basename(path)}")
        sf.write(output_path, waveform.astype(np.float32), sample_rate)
        return output_path


class IdentityPreprocessor(AudioPreprocessor):
    def process(self, path: str, work_dir: str) -> str:
        return path


class BandpassNormalizePreprocessor(AudioPreprocessor):
    def process(self, path: str, work_dir: str) -> str:
        waveform, sample_rate = librosa.load(path, sr=self.target_sr, mono=True)
        return self._write(peak_normalize(bandpass(waveform, sample_rate)), sample_rate, path, work_dir, "bp")


class BandpassNormalizeShiftPreprocessor(AudioPreprocessor):
    """Moves 100-1800 Hz content up to roughly 200-3600 Hz and slows transients.

    Exploratory: it makes the recording less faithful in exchange for landing closer to the
    encoder's training distribution. Report any gain as such.
    """

    def process(self, path: str, work_dir: str) -> str:
        waveform, sample_rate = librosa.load(path, sr=self.target_sr, mono=True)
        processed = peak_normalize(bandpass(waveform, sample_rate))
        processed = librosa.effects.pitch_shift(processed, sr=sample_rate, n_steps=PITCH_SHIFT_SEMITONES)
        processed = librosa.effects.time_stretch(processed, rate=TIME_STRETCH_RATE)
        return self._write(processed, sample_rate, path, work_dir, "bpshift")


class SilenceReplacementPreprocessor(AudioPreprocessor):
    """The silence control: the model sees a duration-matched silent clip instead of the recording.

    Whatever it answers is its label prior with no signal at all.
    """

    def process(self, path: str, work_dir: str) -> str:
        return write_silence(get_duration_s(path), self.target_sr, work_dir)


_PREPROCESSOR_BY_MODE: dict[AudioPreprocessingMode, type[AudioPreprocessor]] = {
    "none": IdentityPreprocessor,
    "bandpass_normalize": BandpassNormalizePreprocessor,
    "bandpass_normalize_shift": BandpassNormalizeShiftPreprocessor,
}


def build_preprocessor(meta: ExperimentMeta | None, target_sr: int) -> AudioPreprocessor:
    if meta is not None and meta.replace_audio_with_silence:
        return SilenceReplacementPreprocessor(target_sr)
    mode: AudioPreprocessingMode = meta.preprocessing if meta else "none"
    return _PREPROCESSOR_BY_MODE[mode](target_sr)
