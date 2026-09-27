import torch
import librosa
from loguru import logger
from speech_processing.adapters.transformers import TransformersEmbeddingAdapter
from speech_processing.utils.consts import DEFAULT_SAMPLING_RATE

class OvisAudioEmbedder:
    def __init__(self, model_id: str = "ATH-MaaS/Ovis-Omni-Embedding-3B"):
        self.model_id = model_id
        self.adapter = TransformersEmbeddingAdapter(model_id=model_id)
        
    def embed_audio(self, audio_path: str) -> torch.Tensor:
        """Embeds a single audio file and returns its dense vector."""
        try:
            audio, sr = librosa.load(audio_path, sr=DEFAULT_SAMPLING_RATE, mono=True)
            return self.adapter.embed_audio(audio)
        except Exception as e:
            logger.error(f"Error embedding {audio_path}: {e}")
            raise
