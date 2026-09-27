from io import BytesIO
from urllib.request import urlopen

import librosa
from loguru import logger

from speech_processing.adapters.transformers import TransformersAdapter
from speech_processing.config.core import AudioModelConfig, GenerationParams
from speech_processing.data.dtos import AudioRequest, AudioResponse
from speech_processing.models.base import BaseAudioModel


def _force_mono_audio(path: str, target_sr: int) -> str:
    """
    Converts audio to mono and target_sr, caching it to avoid re-computation.
    Only required for engines (like Voxtral/Mistral) whose internal tokenizers crash on stereo/multi-channel audio.
    """
    import os
    import hashlib
    import soundfile as sf
    import librosa
    
    if path.startswith("http"):
        return path
        
    abs_path = os.path.abspath(path)
    cache_dir = "/tmp/audio_mono_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    path_hash = hashlib.md5(abs_path.encode()).hexdigest()
    basename = os.path.basename(path)
    cached_path = os.path.join(cache_dir, f"{path_hash}_{target_sr}_{basename}")
    
    if not os.path.exists(cached_path):
        # librosa.load naturally downmixes to mono
        y, sr = librosa.load(abs_path, sr=target_sr, mono=True)
        sf.write(cached_path, y, sr)
        
    return f"file://{cached_path}"



class QwenAudioEngine(BaseAudioModel):
    def __init__(self, config: AudioModelConfig):
        self.config = config
        self.adapter = TransformersAdapter(
            model_id=config.model_id,
            dtype=config.dtype,
            max_model_len=config.max_model_len
        )

    def batch_infer(self, requests: list[AudioRequest]) -> list[AudioResponse]:
        batch_size = self.config.max_num_seqs
        responses = []

        for i in range(0, len(requests), batch_size):
            batch_reqs = requests[i : i + batch_size]
            
            # Setup initial state
            valid_reqs = []
            batch_conversations = []
            batch_audios = []

            for req in batch_reqs:
                conversation = []
                if req.system_prompt:
                    conversation.append({
                        "role": "system",
                        "content": req.system_prompt
                    })
                req_audios = []
                
                try:
                    def load_audio(audio_bytes, audio_path):
                        if audio_bytes is not None:
                            return librosa.load(BytesIO(audio_bytes), sr=self.adapter.processor.feature_extractor.sampling_rate)[0]
                        elif audio_path.startswith("http"):
                            return librosa.load(BytesIO(urlopen(audio_path).read()), sr=self.adapter.processor.feature_extractor.sampling_rate)[0]
                        else:
                            return librosa.load(audio_path, sr=self.adapter.processor.feature_extractor.sampling_rate)[0]

                    # 1. Add Few-Shot Turns
                    for turn in req.few_shot_turns:
                        user_text = turn.user_text
                        if isinstance(turn.audio_path, list):
                            for p, b in zip(turn.audio_path, turn.audio_bytes):
                                user_text = "<|AUDIO|>\n" + user_text
                                req_audios.append(load_audio(b, p))
                        elif turn.audio_path is not None:
                            user_text = "<|AUDIO|>\n" + user_text
                            req_audios.append(load_audio(turn.audio_bytes, turn.audio_path))
                        
                        conversation.append({"role": "user", "content": user_text})
                        conversation.append({"role": "assistant", "content": turn.assistant_text})

                    # Pre-load target audio
                    req_audios.append(load_audio(req.audio_bytes, req.audio_path))

                    batch_conversations.append(conversation)
                    batch_audios.append(req_audios)
                    valid_reqs.append(req)
                except RuntimeError as e:
                    logger.error(f"Failed to load audio {req.audio_path}: {e}")

            if not valid_reqs:
                continue

            first_inst = valid_reqs[0].instruction
            num_steps = len(first_inst) if isinstance(first_inst, list) else 1
            
            final_outputs = [""] * len(valid_reqs)

            for step in range(num_steps):
                texts = []
                for idx, conv in enumerate(batch_conversations):
                    item_insts = valid_reqs[idx].instruction
                    if not isinstance(item_insts, list):
                        item_insts = [item_insts]
                        
                    inst = item_insts[step]
                    
                    if step == 0:
                        content = "<|AUDIO|>\n" + inst
                    else:
                        content = inst
                    
                    conv.append({"role": "user", "content": content})
                    base_text = self.adapter.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                    if "<analysis>" in inst:
                        base_text += "<analysis>\n"
                    texts.append(base_text)

                logger.info(f"Processing audio batch of size {len(valid_reqs)} (Turn {step+1}/{num_steps})...")

                try:
                    generated_texts = self.adapter.generate_batch(
                        texts=texts, audios=batch_audios, max_new_tokens=self.config.max_new_tokens
                    )
                    
                    for idx, gen_text in enumerate(generated_texts):
                        
                        # Re-inject the prefilled tag so the output is well-formed
                        item_inst_check = valid_reqs[idx].instruction[step] if isinstance(valid_reqs[idx].instruction, list) else valid_reqs[idx].instruction
                        if "<analysis>" in item_inst_check:
                            gen_text = "<analysis>\n" + gen_text
                            
                        batch_conversations[idx].append({"role": "assistant", "content": gen_text})
                        if num_steps > 1:
                            item_inst = valid_reqs[idx].instruction[step] if isinstance(valid_reqs[idx].instruction, list) else valid_reqs[idx].instruction
                            final_outputs[idx] += f"Turn {step+1} - User: {item_inst}\nAssistant: {gen_text}\n\n"
                        else:
                            final_outputs[idx] = gen_text
                except RuntimeError as e:
                    logger.error(f"Failed to process batch on turn {step+1}: {e}")
                    for idx in range(len(valid_reqs)):
                        final_outputs[idx] += "\n[Error processing turn]"

            for req, out_text, insts in zip(valid_reqs, final_outputs, [req.instruction for req in valid_reqs]):
                # Store instruction as string for backwards compatibility with reporting
                inst_str = str(insts) if isinstance(insts, list) else insts
                responses.append(
                    AudioResponse(
                        sample_id=req.audio_path,
                        instruction=inst_str, 
                        generated_text=out_text.strip()
                    )
                )

        return responses


class VoxtralAudioEngine(BaseAudioModel):
    def __init__(self, config: AudioModelConfig) -> None:
        self.config = config
        from speech_processing.adapters.vllm import VLLMAdapter
        import os

        # Pass kwargs directly to VLLMAdapter
        self.adapter = VLLMAdapter(
            model=config.model_id,
            trust_remote_code=True,
            max_model_len=config.max_model_len,
            # vLLM defaults to 1 audio per prompt. We hardcode this to 8 across all audio models 
            # to pre-allocate KV cache for few-shot prompting, which injects multiple audios per prompt.
            limit_mm_per_prompt={"audio": 8},
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_seqs=config.max_num_seqs,
            # Whitelist the entire root filesystem ("/") to bypass vLLM's strict local media security sandbox.
            # Required because we pass local dataset files via "file://" URI scheme.
            allowed_local_media_path="/",
        )
        self.tokenizer = self.adapter.tokenizer

    def batch_infer(self, requests: list[AudioRequest]) -> list[AudioResponse]:
        import os
        from loguru import logger
        
        if not requests:
            return []

        first_inst = requests[0].instruction
        num_steps = len(first_inst) if isinstance(first_inst, list) else 1
        final_outputs = [""] * len(requests)
        batch_conversations = [[] for _ in requests]
        


        for step in range(num_steps):
            inputs = []
            for idx, conv in enumerate(batch_conversations):
                req = requests[idx]
                item_insts = req.instruction
                if not isinstance(item_insts, list):
                    item_insts = [item_insts]
                    
                inst = item_insts[step]
                
                # First step logic
                if step == 0:
                    sys_prompt = f"{req.system_prompt}\n\n" if req.system_prompt else ""
                    
                    for i, turn in enumerate(req.few_shot_turns):
                        content = []
                        if isinstance(turn.audio_path, list):
                            for p in turn.audio_path:
                                content.append({"type": "audio_url", "audio_url": {"url": _force_mono_audio(p, self.config.target_sr)}})
                        elif turn.audio_path is not None:
                            content.append({"type": "audio_url", "audio_url": {"url": _force_mono_audio(turn.audio_path, self.config.target_sr)}})
                            
                        # Prepend the system prompt strictly to the FIRST user message to bypass Voxtral tokenizer limitations
                        user_text = (sys_prompt + turn.user_text) if i == 0 else turn.user_text
                        content.append({"type": "text", "text": user_text})
                        conv.append({"role": "user", "content": content})
                        conv.append({"role": "assistant", "content": turn.assistant_text})
                
                # Add the target user turn
                if step == 0:
                    sys_prompt = f"{req.system_prompt}\n\n" if req.system_prompt else ""
                    # If there were no few-shot turns, this target instruction is the first message
                    target_inst = (sys_prompt + inst) if not req.few_shot_turns else inst
                    
                    content = [
                        {"type": "audio_url", "audio_url": {"url": _force_mono_audio(req.audio_path, self.config.target_sr)}},
                        {"type": "text", "text": target_inst},
                    ]
                else:
                    content = [{"type": "text", "text": inst}]
                    
                conv.append({"role": "user", "content": content})
                
                # ASSISTANT PREFILLING (Forced Chain-of-Thought)
                # If the dataset runner specifies an assistant prefill (e.g., forcing an XML tag),
                # we structurally push it here so the engine remains agnostic to the prompt schema.
                if req.assistant_prefill:
                    conv.append({"role": "assistant", "content": req.assistant_prefill})
                
                inputs.append(list(conv))

            logger.info(f"Processing audio batch of size {len(requests)} (Turn {step+1}/{num_steps})...")

            try:
                sampling_params = GenerationParams(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_new_tokens=self.config.max_new_tokens
                )
                
                generated_texts = self.adapter.generate_batch(
                    prompts=inputs, sampling_params=sampling_params
                )
                
                for idx, gen_text in enumerate(generated_texts):
                    item_inst_check = requests[idx].instruction[step] if isinstance(requests[idx].instruction, list) else requests[idx].instruction
                    if req.assistant_prefill:
                        gen_text = req.assistant_prefill + gen_text
                        batch_conversations[idx].pop()
                        
                    batch_conversations[idx].append({"role": "assistant", "content": gen_text})
                    
                    if num_steps > 1:
                        final_outputs[idx] += f"Turn {step+1} - User: {item_inst_check}\nAssistant: {gen_text}\n\n"
                    else:
                        final_outputs[idx] = gen_text
            except RuntimeError as e:
                logger.error(f"Failed to process batch on turn {step+1}: {e}")
                for idx in range(len(requests)):
                    final_outputs[idx] += "\n[Error processing turn]"

        responses = []
        for req, out_text, insts in zip(requests, final_outputs, [r.instruction for r in requests]):
            inst_str = str(insts) if isinstance(insts, list) else insts
            responses.append(
                AudioResponse(
                    sample_id=req.audio_path,
                    instruction=inst_str, 
                    generated_text=out_text.strip()
                )
            )

        return responses
