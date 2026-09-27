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
        
    def embed_audio(self, audio_source) -> torch.Tensor:
        """
        Takes an audio path, URL, or multimodal dict and returns the embedding.
        """
        import torch
        
        # sentence-transformers natively handles multimodal dicts for LCO-Embedding
        if isinstance(audio_source, str):
            payload = {"audio": audio_source}
        else:
            payload = audio_source
            
        with torch.no_grad():
            embedding = self.model.encode(payload, convert_to_tensor=True)
            return embedding.clone().cpu()

    def compute_similarity(self, query_emb: torch.Tensor, cand_emb: torch.Tensor) -> torch.Tensor:
        """
        Uses the internal similarity metric of the SentenceTransformer model.
        """
        return self.model.similarity(query_emb, cand_emb)
