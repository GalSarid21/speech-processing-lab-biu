import torch
from loguru import logger

from speech_processing.adapters.sentence_transformers import SentenceTransformersEmbeddingAdapter

DEFAULT_EMBEDDING_MODEL_ID = "ATH-MaaS/Ovis-Omni-Embedding-3B"
DEFAULT_BATCH_SIZE = 16


class LCOAudioEmbedder:
    def __init__(self, model_id: str = DEFAULT_EMBEDDING_MODEL_ID) -> None:
        self.model_id = model_id
        self.adapter = SentenceTransformersEmbeddingAdapter(model_id=model_id)

    def embed_audio(
        self, audio_source: str | dict | list[str] | list[dict], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> torch.Tensor:
        """Embeds audio files and returns dense vectors. Passes paths directly to the adapter."""
        try:
            return self.adapter.embed_audio(audio_source, batch_size=batch_size)
        except (RuntimeError, OSError, ValueError) as e:
            logger.error(f"Error embedding {audio_source}: {e}")
            raise

    def compute_similarity(self, query_emb: torch.Tensor, cand_emb: torch.Tensor) -> torch.Tensor:
        return self.adapter.compute_similarity(query_emb, cand_emb)
