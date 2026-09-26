import argparse
import gc
import torch
from loguru import logger

from speech_processing.config.core import AudioModelConfig
from speech_processing.models.audio import QwenAudioEngine
from speech_processing.data.dtos import AudioRequest, FewShotTurn
import librosa
import numpy as np

def create_dummy_audio(duration_sec: int, sr: int) -> np.ndarray:
    # Create random noise to simulate audio
    return np.random.randn(sr * duration_sec).astype(np.float32)

def run_diagnostic(model_id: str, start_batch_size: int = 8, max_audio_duration_sec: int = 20, few_shots: int = 0):
    logger.info(f"Initializing {model_id} (This takes a moment...)")
    
    config = AudioModelConfig(
        model_id=model_id,
        dtype="bfloat16",
        max_num_seqs=start_batch_size,
        max_new_tokens=256,
        max_model_len=8192,
    )
    
    try:
        engine = QwenAudioEngine(config)
        # Dynamically extract the optimal sampling rate for this specific model!
        sr = engine.processor.feature_extractor.sampling_rate
        logger.info(f"Extracted optimal sampling rate: {sr}Hz")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
        
    current_batch_size = start_batch_size
    
    # We will loop down until we find a batch size that survives
    while current_batch_size > 0:
        logger.info(f"\n--- Testing Batch Size: {current_batch_size} (Few Shots: {few_shots}, Duration: {max_audio_duration_sec}s) ---")
        
        # Free memory before test
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Build dummy requests
        requests = []
        for i in range(current_batch_size):
            turns = []
            for j in range(few_shots):
                turns.append(FewShotTurn(
                    user_text=f"Dummy question {j}",
                    assistant_text=f"Dummy answer {j}",
                    audio_path=f"dummy_{i}_{j}.wav",
                    audio_bytes=create_dummy_audio(max_audio_duration_sec, sr).tobytes()
                ))
            
            req = AudioRequest(
                instruction="Describe this audio carefully.",
                audio_path=f"dummy_target_{i}.wav",
                audio_bytes=None, 
                few_shot_turns=turns
            )
            requests.append(req)
            
        # Mock librosa load inside this context
        import librosa
        original_load = librosa.load
        librosa.load = lambda path, sr_arg: (create_dummy_audio(max_audio_duration_sec, sr), sr)
        
        engine.config.max_num_seqs = current_batch_size
        
        try:
            responses = engine.batch_infer(requests)
            logger.success(f"SUCCESS! Batch size {current_batch_size} fits in VRAM.")
            librosa.load = original_load
            return current_batch_size
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(f"OOM at batch size {current_batch_size}. Reducing...")
                current_batch_size //= 2 # cut batch_size in half after OOM error
            else:
                logger.error(f"Unexpected error: {e}")
                break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
        finally:
            librosa.load = original_load

    logger.error("Failed even at batch size 1! Your audio/few-shots might be too massive.")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--start-batch", type=int, default=8)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--few-shots", type=int, default=0)
    args = parser.parse_args()
    
    run_diagnostic(args.model_id, args.start_batch, args.duration, args.few_shots)

