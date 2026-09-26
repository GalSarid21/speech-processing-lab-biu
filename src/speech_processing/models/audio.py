from io import BytesIO
from urllib.request import urlopen

import librosa
import torch
from loguru import logger
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from speech_processing.adapters.transformers_adapter import TransformersAdapter
from speech_processing.config.core import AudioModelConfig
from speech_processing.data.dtos import AudioRequest, AudioResponse
from speech_processing.models.base import BaseAudioModel


class QwenAudioEngine(BaseAudioModel):
    def __init__(self, config: AudioModelConfig):
        self.config = config
        logger.info(
            f"Loading Qwen Audio Model from {config.model_id} via transformers..."
        )
        self.processor = AutoProcessor.from_pretrained(config.model_id)

        torch_dtype: torch.dtype = getattr(torch, config.dtype, torch.bfloat16)

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            config.model_id, torch_dtype=torch_dtype, device_map="auto"
        )
        self.model.eval()

        self.adapter = TransformersAdapter(
            self.model, 
            self.processor, 
            max_model_len=self.config.max_model_len,
            temperature=self.config.temperature,
            top_p=self.config.top_p
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
                conversation = [
                    {
                        "role": "system",
                        "content": "You are a helpful and precise reasoning assistant. You MUST think and respond entirely in English. NEVER use Chinese characters."
                    }
                ]
                req_audios = []
                
                try:
                    def load_audio(audio_bytes, audio_path):
                        if audio_bytes is not None:
                            return librosa.load(BytesIO(audio_bytes), sr=self.processor.feature_extractor.sampling_rate)[0]
                        elif audio_path.startswith("http"):
                            return librosa.load(BytesIO(urlopen(audio_path).read()), sr=self.processor.feature_extractor.sampling_rate)[0]
                        else:
                            return librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)[0]

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
                except Exception as e:  # noqa: BLE001
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
                    base_text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
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
                except Exception as e:  # noqa: BLE001
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
    def __init__(self, config: AudioModelConfig):
        self.config = config
        logger.info(
            f"Loading Voxtral Audio Model from {config.model_id} via vLLM..."
        )
        
        try:
            from vllm import LLM, SamplingParams
            from speech_processing.adapters.vllm_adapter import VLLMAdapter
        except ImportError:
            logger.error("vLLM is not installed.")
            raise

        self.llm = LLM(
            model=config.model_id,
            trust_remote_code=True,
            max_model_len=config.max_model_len,
            limit_mm_per_prompt={"audio": 8},
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.adapter = VLLMAdapter(self.llm)

    def batch_infer(self, requests: list[AudioRequest]) -> list[AudioResponse]:
        from vllm import SamplingParams
        batch_size = self.config.max_num_seqs
        responses = []

        for i in range(0, len(requests), batch_size):
            batch_reqs = requests[i : i + batch_size]
            valid_reqs = []
            batch_conversations = []
            batch_audios = []

            for req in batch_reqs:
                conversation = [
                    {
                        "role": "system",
                        "content": "You are a helpful and precise reasoning assistant. You MUST think and respond entirely in English. NEVER use Chinese characters."
                    }
                ]
                req_audios = []
                
                try:
                    def load_audio(audio_bytes, audio_path):
                        if audio_bytes is not None:
                            return librosa.load(BytesIO(audio_bytes), sr=16000)[0]
                        elif audio_path.startswith("http"):
                            return librosa.load(BytesIO(urlopen(audio_path).read()), sr=16000)[0]
                        else:
                            return librosa.load(audio_path, sr=16000)[0]

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

                    req_audios.append(load_audio(req.audio_bytes, req.audio_path))

                    batch_conversations.append(conversation)
                    batch_audios.append(req_audios)
                    valid_reqs.append(req)
                except Exception as e:
                    logger.error(f"Failed to load audio {req.audio_path}: {e}")

            if not valid_reqs:
                continue

            first_inst = valid_reqs[0].instruction
            num_steps = len(first_inst) if isinstance(first_inst, list) else 1
            final_outputs = [""] * len(valid_reqs)

            for step in range(num_steps):
                inputs = []
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
                    
                    # Apply chat template
                                        # Apply chat template with fallback for system role
                    try:
                        base_text = self.tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                    except Exception as e:
                        # Some Mistral templates do not support the 'system' role. 
                        # If it fails, move the system instruction to the beginning of the first user message.
                        if conv[0]["role"] == "system":
                            sys_content = conv[0]["content"] if isinstance(conv[0]["content"], str) else conv[0]["content"][0]["text"]
                            conv.pop(0)
                            if conv[0]["role"] == "user":
                                if isinstance(conv[0]["content"], str):
                                    conv[0]["content"] = sys_content + "\n\n" + conv[0]["content"]
                                else:
                                    conv[0]["content"].insert(0, {"type": "text", "text": sys_content + "\n\n"})
                        base_text = self.tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                        
                    if "<analysis>" in inst:
                        base_text += "<analysis>\n"
                        
                    # Prepare multimodal data for vLLM
                    # vLLM expects a single tuple (audio_array, sampling_rate) or a list of tuples for multiple audios
                    mm_audios = [(a, 16000) for a in batch_audios[idx]]
                    
                    # NOTE: Depending on Voxtral implementation in vLLM, it might need 'audio' or 'audios' key.
                    # Standard vLLM for Qwen2-Audio / Audio models uses 'audio' key with list of tuples if multiple
                    mm_data = {"audio": mm_audios} if len(mm_audios) > 1 else {"audio": mm_audios[0]}
                    
                    inputs.append({
                        "prompt": base_text,
                        "multi_modal_data": mm_data
                    })

                logger.info(f"Processing audio batch of size {len(valid_reqs)} (Turn {step+1}/{num_steps})...")

                try:
                    sampling_params = SamplingParams(
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        max_tokens=self.config.max_new_tokens
                    )
                    
                    generated_texts = self.adapter.generate_batch(
                        prompts=inputs, sampling_params=sampling_params
                    )
                    
                    for idx, gen_text in enumerate(generated_texts):
                        item_inst_check = valid_reqs[idx].instruction[step] if isinstance(valid_reqs[idx].instruction, list) else valid_reqs[idx].instruction
                        if "<analysis>" in item_inst_check:
                            gen_text = "<analysis>\n" + gen_text
                            
                        batch_conversations[idx].append({"role": "assistant", "content": gen_text})
                        if num_steps > 1:
                            item_inst = valid_reqs[idx].instruction[step] if isinstance(valid_reqs[idx].instruction, list) else valid_reqs[idx].instruction
                            final_outputs[idx] += f"Turn {step+1} - User: {item_inst}\nAssistant: {gen_text}\n\n"
                        else:
                            final_outputs[idx] = gen_text
                except Exception as e:
                    logger.error(f"Failed to process batch on turn {step+1}: {e}")
                    for idx in range(len(valid_reqs)):
                        final_outputs[idx] += "\n[Error processing turn]"

            for req, out_text, insts in zip(valid_reqs, final_outputs, [req.instruction for req in valid_reqs]):
                inst_str = str(insts) if isinstance(insts, list) else insts
                responses.append(
                    AudioResponse(
                        sample_id=req.audio_path,
                        instruction=inst_str, 
                        generated_text=out_text.strip()
                    )
                )

        return responses

