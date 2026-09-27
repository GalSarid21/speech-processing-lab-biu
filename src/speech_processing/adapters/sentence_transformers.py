import torch
class SentenceTransformersEmbeddingAdapter:
    def __init__(self, model_id: str):
        import torch
        from loguru import logger
        from sentence_transformers import SentenceTransformer
        from speech_processing.utils.consts import MODEL_DEVICE_DTYPE_MAPPING
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        torch_dtype = MODEL_DEVICE_DTYPE_MAPPING.get(self.device, torch.float32)
        
        logger.info(f"Loading SentenceTransformers Embedding Model {model_id} onto {self.device}...")
        
        # We explicitly don't use flash_attention_2 by default to keep it stable on mac/colab
        self.model = SentenceTransformer(
            model_id,
            model_kwargs={
                "torch_dtype": torch_dtype,
                "trust_remote_code": True,
            },
            device=self.device
        )
        self.model.eval()
        
    def embed_audio(self, audio_source: str | dict | list[str] | list[dict], batch_size: int = 16) -> torch.Tensor:
        """
        Takes a single audio path/dict OR a list of audio paths/dicts and returns the embedding(s).
        """
        import torch
        
        if isinstance(audio_source, list):
            payload = [{"audio": x} if isinstance(x, str) else x for x in audio_source]
        elif isinstance(audio_source, str):
            payload = {"audio": audio_source}
        else:
            payload = audio_source
            
        with torch.no_grad():
            embedding = self.model.encode(payload, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
            # Ensure it is always 2D: [batch, hidden_dim]
            if embedding.ndim == 1:
                embedding = embedding.unsqueeze(0)
            return embedding.clone().cpu()

    def compute_similarity(self, query_emb: torch.Tensor, cand_emb: torch.Tensor) -> torch.Tensor:
        """
        Uses the internal similarity metric of the SentenceTransformer model.
        """
        return self.model.similarity(query_emb, cand_emb)
