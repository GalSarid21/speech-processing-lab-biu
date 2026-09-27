import torch
import librosa
from transformers import AutoModel, AutoProcessor
from loguru import logger

class OvisAudioEmbedder:
    def __init__(self, model_id: str = "ATH-MaaS/Ovis-Omni-Embedding-3B", device: str = "cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {model_id} onto {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        self.model.eval()
        
    def embed_audio(self, audio_path: str) -> torch.Tensor:
        """Embeds a single audio file and returns its dense vector."""
        # Need to verify exactly how Ovis expects audio. 
        # Typically processors take arrays and sampling rates, or just raw bytes/paths depending on the custom code.
        # We will attempt librosa load to standard 16k mono, and let the processor handle it.
        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            # Depending on Ovis processor, it might accept raw audio arrays or lists of dicts
            # Assuming standard multimodal usage:
            inputs = self.processor(audios=[audio], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Usually the embedding is the last hidden state of the last token
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state[:, -1, :].clone().cpu()
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output.clone().cpu()
            elif isinstance(outputs, tuple):
                # Fallback for hidden states if returned as tuple
                return outputs[0][:, -1, :].clone().cpu()
            else:
                logger.error("Could not parse output embedding format from Ovis model.")
                raise ValueError("Unexpected model output format.")
        except Exception as e:
            logger.error(f"Error embedding {audio_path}: {e}")
            raise
