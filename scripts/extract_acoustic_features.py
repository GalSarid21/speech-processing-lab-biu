import json
import os
import argparse
import librosa
import numpy as np
import torch
from pyannote.audio import Pipeline
import whisper

def get_pauses(y, sr, threshold=-40, min_duration=0.1):
    # simple pause detection
    intervals = librosa.effects.split(y, top_db=abs(threshold))
    pauses = []
    prev_end = 0
    for start, end in intervals:
        pause_dur = (start - prev_end) / sr
        if pause_dur >= min_duration and prev_end > 0:
            pauses.append(pause_dur)
        prev_end = end
    return pauses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default="data/MMAR/audio")
    parser.add_argument("--output_file", type=str, default="data/mmar_acoustic_features.json")
    parser.add_argument("--sample_ids_file", type=str, default="data/mmar_en_speech_test_ids.txt")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Pyannote pipeline...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=True
    ).to(torch.device(device))
    
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("large-v3", device=device)
    
    audio_files = [f for f in os.listdir(args.audio_dir) if f.endswith(".wav")]
    
    if args.sample_ids_file and os.path.exists(args.sample_ids_file):
        with open(args.sample_ids_file, 'r') as f:
            valid_ids = {line.strip() for line in f if line.strip()}
        audio_files = [f for f in audio_files if os.path.splitext(f)[0] in valid_ids]
        print(f"Filtered down to {len(audio_files)} files using {args.sample_ids_file}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, "w") as f:
        for idx, file_name in enumerate(audio_files):
            print(f"Processing {idx+1}/{len(audio_files)}: {file_name}")
            item_id = os.path.splitext(file_name)[0]
            audio_path = os.path.join(args.audio_dir, file_name)
            
            try:
                diarization = diarization_pipeline(audio_path)
                y, sr = librosa.load(audio_path, sr=16000)
                
                # We need word-level timestamps, but simple segmentation is fine for the scene-graph
                speaker_stats = {}
                transcript_lines = []
                
                annotation = getattr(diarization, "speaker_diarization", diarization)
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    start = turn.start
                    end = turn.end
                    
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    segment_audio = y[start_sample:end_sample]
                    
                    if len(segment_audio) < 160:
                        continue
                        
                    result = whisper_model.transcribe(segment_audio, fp16=(device=="cuda"))
                    text = result["text"].strip()
                    
                    if not text:
                        continue
                        
                    transcript_lines.append(f"[{start:.1f}-{end:.1f}] {speaker}: {text}")
                    
                    f0, voiced_flag, _ = librosa.pyin(segment_audio, fmin=50, fmax=500, sr=sr)
                    valid_f0 = f0[voiced_flag] if f0 is not None and np.any(voiced_flag) else []
                    rms = librosa.feature.rms(y=segment_audio)[0]
                    
                    words = text.split()
                    duration = end - start
                    rate = len(words) / duration if duration > 0 else 0
                    
                    pauses = get_pauses(segment_audio, sr)
                    
                    if speaker not in speaker_stats:
                        speaker_stats[speaker] = {"f0": [], "rms": [], "words": 0, "duration": 0, "pauses": []}
                        
                    speaker_stats[speaker]["f0"].extend(valid_f0)
                    speaker_stats[speaker]["rms"].extend(rms)
                    speaker_stats[speaker]["words"] += len(words)
                    speaker_stats[speaker]["duration"] += duration
                    speaker_stats[speaker]["pauses"].extend(pauses)
                    
                feature_lines = []
                for spk, stats in speaker_stats.items():
                    f0_arr = np.array(stats["f0"])
                    rms_arr = np.array(stats["rms"])
                    
                    f0_mean = np.mean(f0_arr) if len(f0_arr) > 0 else 0
                    f0_min = np.min(f0_arr) if len(f0_arr) > 0 else 0
                    f0_max = np.max(f0_arr) if len(f0_arr) > 0 else 0
                    
                    rms_mean = np.mean(rms_arr) if len(rms_arr) > 0 else 0
                    rms_min = np.min(rms_arr) if len(rms_arr) > 0 else 0
                    rms_max = np.max(rms_arr) if len(rms_arr) > 0 else 0
                    energy_level = "high" if rms_mean > 0.05 else ("low" if rms_mean < 0.01 else "medium")
                    
                    rate = stats["words"] / stats["duration"] if stats["duration"] > 0 else 0
                    pause_count = len(stats["pauses"])
                    pause_avg = np.mean(stats["pauses"]) if pause_count > 0 else 0
                    
                    feature_lines.append(f"{spk}: F0 mean {f0_mean:.0f} Hz (range {f0_min:.0f}-{f0_max:.0f}), energy {energy_level}, rate {rate:.1f} words/s, {pause_count} pauses (avg {pause_avg:.1f} s)")
                
                # Dump JSON line
                data = {
                    "item_id": item_id,
                    "diarized_transcript": "\\n".join(transcript_lines),
                    "acoustic_features": "\\n".join(feature_lines)
                }
                f.write(json.dumps(data) + "\\n")
                f.flush()
                
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    main()
