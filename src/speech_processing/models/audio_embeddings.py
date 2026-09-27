import torch
import librosa
from loguru import logger
from speech_processing.adapters.sentence_transformers import SentenceTransformersEmbeddingAdapter
from speech_processing.utils.consts import DEFAULT_SAMPLING_RATE

class LCOAudioEmbedder:
    def __init__(self, model_id: str = "ATH-MaaS/Ovis-Omni-Embedding-3B"):
        self.model_id = model_id
        self.adapter = SentenceTransformersEmbeddingAdapter(model_id=model_id)
        
    def embed_audio(self, audio_source: str | dict | list[str] | list[dict], batch_size: int = 16) -> torch.Tensor:
        """Embeds audio files and returns dense vectors. Passes paths directly to adapter."""
        try:
            return self.adapter.embed_audio(audio_source, batch_size=batch_size)
        except Exception as e:
            logger.error(f"Error embedding {audio_source}: {e}")
            raise

    def compute_similarity(self, query_emb, cand_emb):
        return self.adapter.compute_similarity(query_emb, cand_emb)
