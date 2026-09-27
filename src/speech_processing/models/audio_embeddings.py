import torch
import librosa
from loguru import logger
from speech_processing.adapters.sentence_transformers import SentenceTransformersEmbeddingAdapter
from speech_processing.utils.consts import DEFAULT_SAMPLING_RATE

class LCOAudioEmbedder:
    def __init__(self, model_id: str = "ATH-MaaS/Ovis-Omni-Embedding-3B"):
        self.model_id = model_id
        self.adapter = SentenceTransformersEmbeddingAdapter(model_id=model_id)
        
    def embed_audio(self, audio_path: str | list[str] | dict | list[dict], batch_size: int | None = None) -> torch.Tensor:
        """Embeds a single audio file and returns its dense vector."""
        try:
            audio, sr = librosa.load(audio_path, sr=DEFAULT_SAMPLING_RATE, mono=True)
            return self.adapter.embed_audio(audio, batch_size)
        except Exception as e:
            logger.error(f"Error embedding {audio_path}: {e}")
            raise

    def compute_similarity(self, query_emb, cand_emb):
        return self.adapter.compute_similarity(query_emb, cand_emb)
