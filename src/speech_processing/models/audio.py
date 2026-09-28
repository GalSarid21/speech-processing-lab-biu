from io import BytesIO
from urllib.request import urlopen

import librosa
from loguru import logger

from speech_processing.adapters.transformers import TransformersAdapter
from speech_processing.config.core import AudioModelConfig, GenerationParams
from speech_processing.data.dtos import AudioRequest, AudioResponse
from speech_processing.utils.consts import COT_START_TAG, DEFAULT_SAMPLING_RATE

from speech_processing.models.base import BaseAudioModel
from loguru import logger
from speech_processing.config.core import GenerationParams
from speech_processing.utils.audio import chunk_audio, crop_audio, get_silence_audio
from speech_processing.utils.audio import crop_audio
from speech_processing.utils.audio import get_silence_audio
import hashlib
import json
import librosa
import numpy
import os
import re
import soundfile


def _force_mono_audio(path: str, target_sr: int) -> str:
    """
    Converts audio to mono and target_sr, caching it to avoid re-computation.
    Only required for engines (like Voxtral/Mistral) whose internal tokenizers crash on stereo/multi-channel audio.
    """
    
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


    def score_choices(self, requests: list[AudioRequest]) -> list[AudioResponse]:
        # Scoped import because vllm is not available on MacOS and would crash on import
        from vllm import SamplingParams
        
        silence_path = get_silence_audio(self.config.target_sr)
        
        responses = []
        for req in requests:
            # We assume req.metadata["choices"] contains the raw choice strings
            choices = req.metadata.get("choices", [])
            if not choices:
                responses.append(AudioResponse(sample_id=req.audio_path, instruction=req.instruction, generated_text="A", metadata=req.metadata))
                continue
                
            best_choice = "A"
            best_score = -float('inf')
            
            for i, choice_text in enumerate(choices):
                choice_letter = chr(65 + i)
                # Formulate the prompt with the choice text appended
                prompt_text = req.instruction + f" {choice_text}"
                
                # We need to construct the prompt dicts for vllm
                def build_prompt(audio_path):
                    return [
                        {"role": "user", "content": [
                            {"type": "audio_url", "audio_url": {"url": _force_mono_audio(audio_path, self.config.target_sr)}},
                            {"type": "text", "text": req.instruction}
                        ]},
                        {"role": "assistant", "content": choice_text} # Force assistant to say the choice
                    ]
                
                audio_prompt = build_prompt(req.audio_path)
                silence_prompt = build_prompt(silence_path)
                
                sp = SamplingParams(max_tokens=1, prompt_logprobs=1)
                
                audio_out = self.adapter.llm.chat(messages=[audio_prompt], sampling_params=sp, add_generation_prompt=False, continue_final_message=True)[0]
                silence_out = self.adapter.llm.chat(messages=[silence_prompt], sampling_params=sp, add_generation_prompt=False, continue_final_message=True)[0]
                
                # The logprobs for the assistant's turn are at the end of prompt_logprobs
                # We need to sum them. For simplicity, we just sum all valid logprobs in the prompt
                def get_prob(out):
                    if not out.prompt_logprobs: return -100
                    return sum([list(p.values())[0].logprob for p in out.prompt_logprobs if p])
                
                audio_score = get_prob(audio_out)
                silence_score = get_prob(silence_out)
                
                score = audio_score - silence_score
                if score > best_score:
                    best_score = score
                    best_choice = choice_letter
                    
            responses.append(AudioResponse(
                sample_id=req.audio_path,
                instruction=req.instruction,
                generated_text=best_choice,
                metadata=req.metadata
            ))
            
        return responses
    def batch_infer(self, requests: list[AudioRequest]) -> list[AudioResponse]:
        if getattr(self.config, 'use_contrastive_scoring', False):
            return self.score_choices(requests)

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
                        content = ("<|AUDIO|>\n" + inst) if getattr(self.config, 'audio_first', False) else (inst + "\n<|AUDIO|>")
                    else:
                        content = inst
                    
                    conv.append({"role": "user", "content": content})
                    base_text = self.adapter.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                    if COT_START_TAG in inst:
                        base_text += f"{COT_START_TAG}\n"
                    texts.append(base_text)

                logger.info(f"Processing audio batch of size {len(valid_reqs)} (Turn {step+1}/{num_steps})...")

                try:
                    generated_texts = self.adapter.generate_batch(
                        texts=texts, audios=batch_audios, max_new_tokens=self.config.max_new_tokens
                    )
                    
                    for idx, gen_text in enumerate(generated_texts):
                        
                        # Re-inject the prefilled tag so the output is well-formed
                        item_inst_check = valid_reqs[idx].instruction[step] if isinstance(valid_reqs[idx].instruction, list) else valid_reqs[idx].instruction
                        if COT_START_TAG in item_inst_check:
                            gen_text = f"{COT_START_TAG}\n" + gen_text
                            
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
                        generated_text=out_text.strip(),
                        metadata=req.metadata
                    )
                )

        return responses


class VoxtralAudioEngine(BaseAudioModel):
    def __init__(self, config: AudioModelConfig) -> None:
        self.config = config
        # Scoped import because vllm is not available on MacOS and would crash on import
        from speech_processing.adapters.vllm import VLLMAdapter

        # Pass kwargs directly to VLLMAdapter
        self.adapter = VLLMAdapter(
            model=config.model_id,
            trust_remote_code=True,
            max_model_len=config.max_model_len,
            limit_mm_per_prompt={"audio": 8},
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_seqs=config.max_num_seqs,
            allowed_local_media_path="/",
        )
        self.tokenizer = self.adapter.tokenizer

    def batch_infer(self, requests: list[AudioRequest]) -> list[AudioResponse]:
        # Scoped import because vllm is not available on MacOS and would crash on import
        from vllm import SamplingParams
        
        if not requests:
            return []

        # T3: Contrastive Scoring
        if getattr(self.config, 'contrastive_alpha', None) is not None:
            return self.score_choices(requests)

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
                            
                        user_text = (sys_prompt + turn.user_text) if i == 0 else turn.user_text
                        content.append({"type": "text", "text": user_text})
                        conv.append({"role": "user", "content": content})
                        conv.append({"role": "assistant", "content": turn.assistant_text})
                
                # Add the target user turn
                if step == 0:
                    sys_prompt = f"{req.system_prompt}\n\n" if req.system_prompt else ""
                    target_inst = (sys_prompt + inst) if not req.few_shot_turns else inst
                    
                    content = []
                    
                    if getattr(self.config, 'chunked_audio', False):
                        chunks = chunk_audio(req.audio_path, max_chunk_len_s=5.0, max_chunks=6)
                        for c_idx, (c_path, start_s, end_s) in enumerate(chunks):
                            content.append({"type": "text", "text": f"Segment {c_idx+1} ({start_s:.1f}-{end_s:.1f} s):"})
                            content.append({"type": "audio_url", "audio_url": {"url": _force_mono_audio(c_path, self.config.target_sr)}})
                        content.append({"type": "text", "text": "Full Audio:"})
                        content.append({"type": "audio_url", "audio_url": {"url": _force_mono_audio(req.audio_path, self.config.target_sr)}})
                        content.append({"type": "text", "text": target_inst})
                        
                    else:
                        content = [
                            {"type": "audio_url", "audio_url": {"url": _force_mono_audio(req.audio_path, self.config.target_sr)}},
                            {"type": "text", "text": target_inst},
                        ]
                else:
                    if getattr(self.config, 'two_pass_localization', False) and step == 1:
                        # Parse JSON from previous assistant message
                        prev_msg = conv[-1]["content"]
                        try:
                            # Use regex to find JSON
                            match = re.search(r'\{[^{}]*\}', prev_msg)
                            if match:
                                times = json.loads(match.group(0))
                                start = max(0, float(times["start"]) - 0.5)
                                end = float(times["end"]) + 0.5
                                cropped_path = crop_audio(req.audio_path, start, end)
                                
                                content = [
                                    {"type": "text", "text": "Full Audio:"},
                                    {"type": "audio_url", "audio_url": {"url": _force_mono_audio(req.audio_path, self.config.target_sr)}},
                                    {"type": "text", "text": f"Cropped Segment ({start:.1f}-{end:.1f} s):"},
                                    {"type": "audio_url", "audio_url": {"url": _force_mono_audio(cropped_path, self.config.target_sr)}},
                                    {"type": "text", "text": inst},
                                ]
                            else:
                                raise ValueError("No JSON found")
                        except Exception as e:
                            logger.warning(f"Fallback on localization parsing: {e}")
                            content = [
                                {"type": "audio_url", "audio_url": {"url": _force_mono_audio(req.audio_path, self.config.target_sr)}},
                                {"type": "text", "text": inst},
                            ]
                    else:
                        content = [{"type": "text", "text": inst}]
                    
                conv.append({"role": "user", "content": content})
                
                if req.assistant_prefill:
                    conv.append({"role": "assistant", "content": req.assistant_prefill})
                
                inputs.append(list(conv))

            logger.info(f"Processing audio batch of size {len(requests)} (Turn {step+1}/{num_steps})...")

            try:
                # Need generation params
                sp = GenerationParams(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_new_tokens=self.config.max_new_tokens,
                    stop=self.config.stop
                )
                
                generated_texts = self.adapter.generate_batch(
                    prompts=inputs, sampling_params=sp
                )
                
                for idx, gen_text in enumerate(generated_texts):
                    item_inst_check = requests[idx].instruction[step] if isinstance(requests[idx].instruction, list) else requests[idx].instruction
                    if requests[idx].assistant_prefill:
                        gen_text = requests[idx].assistant_prefill + gen_text
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
                    generated_text=out_text.strip(),
                    metadata=req.metadata
                )
            )

        return responses

    def score_choices(self, requests: list[AudioRequest]) -> list[AudioResponse]:
        # Scoped import because vllm is not available on MacOS and would crash on import
        from vllm import SamplingParams
        
        silence_path = get_silence_audio(self.config.target_sr)
        alpha = self.config.contrastive_alpha
        
        responses = []
        for req in requests:
            choices = req.metadata.get("choices", [])
            if not choices:
                responses.append(AudioResponse(sample_id=req.audio_path, instruction=req.instruction, generated_text="A", metadata=req.metadata))
                continue
                
            best_choice = "A"
            best_score = -float('inf')
            
            # Match duration for silence
            y, sr = librosa.load(req.audio_path, sr=self.config.target_sr)
            dur_s = len(y) / sr
            fd, custom_silence = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(custom_silence, np.zeros(int(dur_s * self.config.target_sr), dtype=np.float32), self.config.target_sr)
            
            for i, choice_text in enumerate(choices):
                choice_letter = chr(65 + i)
                
                def build_prompt(audio_path):
                    return [
                        {"role": "user", "content": [
                            {"type": "audio_url", "audio_url": {"url": _force_mono_audio(audio_path, self.config.target_sr)}},
                            {"type": "text", "text": req.instruction}
                        ]},
                        {"role": "assistant", "content": choice_text}
                    ]
                
                audio_prompt = build_prompt(req.audio_path)
                silence_prompt = build_prompt(custom_silence)
                
                sp = SamplingParams(max_tokens=1, prompt_logprobs=1)
                
                audio_out = self.adapter.llm.chat(messages=[audio_prompt], sampling_params=sp, add_generation_prompt=False, continue_final_message=True)[0]
                silence_out = self.adapter.llm.chat(messages=[silence_prompt], sampling_params=sp, add_generation_prompt=False, continue_final_message=True)[0]
                
                def get_prob(out):
                    if not out.prompt_logprobs: return -100
                    return sum([list(p.values())[0].logprob for p in out.prompt_logprobs if p])
                
                audio_score = get_prob(audio_out)
                silence_score = get_prob(silence_out)
                
                score = audio_score - (alpha * silence_score)
                if score > best_score:
                    best_score = score
                    best_choice = choice_letter
                    
            responses.append(AudioResponse(
                sample_id=req.audio_path,
                instruction=req.instruction,
                generated_text=best_choice,
                metadata=req.metadata
            ))
            
        return responses
