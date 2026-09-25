# 🫁 Speech Processing Lab BIU: Lung Sound Diagnostics Research

**Course**: Speech Processing, Prof. Sharon Ganot, Bar Ilan University  
**Goal**: Evaluate and improve zero-shot & few-shot diagnostic capabilities of Audio-Language Models (ALMs) on respiratory audio using advanced prompting and context management techniques without fine-tuning.

---

## 🧠 Models & Data Selection

1. **Audio Inference Model**: `Qwen/Qwen2-Audio-7B-Instruct` (A multimodal audio-language model capable of natively processing audio waveforms).
2. **Judge Evaluator Model**: `Qwen/Qwen3.8-27B-FP8` (LLM-as-a-judge to evaluate generated diagnoses).
3. **Dataset**: **ICBHI 2017 Respiratory Sound Database** (`DynamicSuperb/RespiratorySoundClassification_ICBHI2017`), capturing stethoscope audio of healthy subjects and patients with diseases like COPD, Bronchiectasis, and Pneumonia.

> **Note**: The `ICBHI 2017 Respiratory Sound Database` is not mentioned in the `Qwen2-Audio-7B-Instruct` technical report, which guarantees it serves as a true out-of-distribution (OOD) test set for evaluating zero-shot/few-shot reasoning without risk of data contamination.

---

## 🎧 Dataset Overview & Evaluation Subset

The full HuggingFace benchmark consists of only 174 audio samples. Because we needed to preserve uncontaminated "unused room" to act as few-shot examples (ensuring the model was never evaluated on the same audio it was prompted with), we ran our pipeline on a subset of **exactly 100 samples**.

Using stratified random sampling (excluding the statistically negligible Asthma and LRTI outliers), we constructed an evaluation subset that perfectly mirrors the natural imbalance of the full dataset:

| Disease Class | Full Dataset (174) % | Our Subset (100) % |
| :--- | :--- | :--- |
| **COPD** | 34.5% | 35.0% |
| **Healthy** | 20.1% | 20.0% |
| **Pneumonia** | 13.8% | 14.0% |
| **URTI** | 13.2% | 14.0% |
| **Bronchiectasis** | 9.2% | 9.0% |
| **Bronchiolitis** | 7.5% | 8.0% |


---

## ⚙️ Experimental Setup & Metrics

**Hardware**: NVIDIA A100-SXM4 (80GB VRAM)

### Deterministic Sampling
To maximize the likelihood that the Audio-Language Model strictly respects our complex prompt guardrails (e.g., Honesty clauses, Base Rates), we explicitly disabled probabilistic sampling. The ALM runs with **Greedy Decoding (`do_sample=False`)**, ensuring it always takes the highest-probability logical path rather than "creatively" deviating and hallucinating.

### Custom Grading Metrics (LLM-as-a-Judge)
Because respiratory audio classification via LLMs produces generative text rather than strict logits, we developed three custom metrics evaluated by a 27B parameter judge:
1. **Acoustic Accuracy**: Did the model correctly detect and describe the *acoustic features* (e.g., crackles, wheezes)?
2. **Diagnostic Accuracy**: Did the model correctly deduce the final *patient disease* (e.g., COPD)?
3. **Hallucination Rate**: Did the model "hallucinate" sounds not present in the audio to justify a guess?

---

## 🧪 The Prompt Engineering Journey

We ran 12 distinct experiments. Below are the core milestones highlighting our progression:

### 🔴 V1: Baseline
* **Rationale**: Establish a zero-shot baseline. How well does the model perform out-of-the-box?
* **Prompt**: `Detect the disease in this lung sound audio.`
* **Results**: Acoustic Acc: **12.5%** | Diagnostic Acc: **27.2%** | Hallucination: **24.0%**

### 🔴 V3: Symptoms Dictionary
* **Rationale**: Let's inject a clinical symptom dictionary (fever, cough).
* **Results**: Acoustic Acc: **13.7%** | Diagnostic Acc: **7.8%** | Hallucination: **88.0%**
* **Failure**: The model began hallucinating *clinical text* directly from the raw stethoscope audio!

### 🟡 V5: Acoustic Dictionary & Chain-of-Thought
* **Rationale**: Replace textual symptoms with *Acoustic Signatures*. Force the model to describe the audio step-by-step (CoT) before guessing.
* **Results**: Acoustic Acc: **52.7%** | Diagnostic Acc: **37.0%** | Hallucination: **56.0%**
* **Breakthrough**: Grounding reasoning in audio signatures massively boosted accuracy.

### 🟢 V10: The Holy Grail (Authentic Few-Shot Holistic)
* **Rationale**: The "Kitchen Sink". We combined a **Holistic Dictionary** (Clinical + Acoustic separation), **Chain-of-Thought**, and **True Multimodal Few-Shot** (injecting 6 real `.wav` tensors representing every target class directly into the context window).
* **Results**: 
    - Acoustic Acc: **64.0%** (5.12x improvement)
    - Diagnostic Acc: **60.0%** (2.21x improvement)
    - Hallucination: **35.0%** (1.46x DEGRADATION from Baseline)

---

## 📊 Results Comparison & The Hallucination Tradeoff


### Experiment Comparison Table
## Experiment Comparisons

| Experiment | Acoustic Acc | Diagnostic Acc | Hallucination Rate | Overall Score |
| :--- | :--- | :--- | :--- | :--- |
| `baseline` | 12.5% (1.00x) | 27.2% (1.00x) | **24.0% (1.00x)** | 38.6% (1.00x) |
| `format_strict` | 0.0% (0.00x) | 0.0% (0.00x) | 36.0% (1.50x) | 21.3% (0.55x) |
| `symptoms_dict` | 13.7% (1.10x) | 7.8% (0.29x) | 88.0% (3.67x) | 11.2% (0.29x) |
| `acoustic_dict` | 42.5% (3.40x) | 23.2% (0.85x) | 63.0% (2.62x) | 34.2% (0.89x) |
| `cot` | 52.7% (4.22x) | 37.0% (1.36x) | 56.0% (2.33x) | 44.6% (1.16x) |
| `few_shot` | 29.5% (2.36x) | 6.2% (0.23x) | 82.0% (3.42x) | 17.9% (0.46x) |
| `cot_and_few_shot` | 38.0% (3.04x) | 15.8% (0.58x) | 81.0% (3.38x) | 24.3% (0.63x) |
| `authentic_few_shot` | 54.0% (4.32x) | 11.8% (0.43x) | 35.0% (1.46x) | 43.6% (1.13x) |
| `authentic_few_shot_holistic` | **64.0% (5.12x)** | **60.0% (2.21x)** | 35.0% (1.46x) | **63.0% (1.63x)** |
| `authentic_few_shot_no_guardrails` | 52.0% (4.16x) | 0.0% (0.00x) | 35.0% (1.46x) | 39.0% (1.01x) |


### Metric Progression Scatter Plots
![Comparison Plots](assets/comparison_scatter_plots.png)

### The Hallucination Tradeoff

While V10 massively improved diagnostic accuracy (2.21x) and acoustic accuracy (5.12x), **we did not beat the baseline Hallucination Rate** (35% vs 24%). 
* **Why?** In the V1 baseline, the model blindly guessed classes without describing any sounds, thus dodging the hallucination penalty. When we added Chain-of-Thought (forcing it to describe sounds), the hallucination rate spiked. 
* **The Failed Fix (V12)**: We tried to fix this in V12 by adding an "Honesty Clause" (`Do NOT invent or hallucinate sounds... if unsure, explicitly state it`). However, the metrics didn't budge. We discovered that **In-Context Learning overwrote the written prompt**—because the model saw absolute confidence in the few-shot examples, it mimicked that confidence, completely ignoring the honesty instruction.

---

## 🎯 Future Work

1. **Scaling to Larger Datasets**: Because we required uncontaminated 'unused room' for our few-shot examples, we were mathematically constrained to evaluating on a subset of the 174 total samples available in the `DynamicSuperb` benchmark. Future iterations must identify significantly larger OOD evaluation datasets to solidify statistical significance.
2. **Bias & Few-Shot Ordering**: Our V10 prompt uses exactly 6 few-shot examples injected in a static order. Because LLMs are sensitive to few-shot ordering, this almost certainly introduces a positional bias. Future experiments should randomly shuffle the order of few-shot examples to measure the exact effect on prediction bias.
3. **Retrieval-Augmented Few-Shot (RAG)**: Implement dynamic few-shot retrieval where we use a vector database to find the top-K most semantically similar examples to the current test sample (e.g., matching the question text or the audio embeddings) and inject them dynamically. This prevents the systemic positional/semantic bias introduced by static few-shot examples.
4. **Dynamic Contrastive Profiling (Two-Stage Audio RAG)**: Rather than trying to cram all possible disease comparisons into a single prompt (which biases the model or explodes VRAM), split inference into two turns. Turn 1 (Zero-Shot) nominates 2-3 suspected candidate diseases. Turn 2 fetches specific contrastive audio pairs for those candidates and asks the model to compare them to the patient's audio for a final decision.
5. **Fine-Tuning**: Freeze the LLM and parameter-efficiently fine-tune (LoRA) the audio encoder explicitly on stethoscope spectrograms to try and lower the 35% CoT hallucination floor.
6. **Hyperparameter Tuning per Prompting Technique**: Currently, the audio model uses a static generation configuration (its factory defaults: `temperature=0.7`, `top_p=0.5`) across all experiments. However, different prompting techniques require different sampling strategies. Future work should dynamically tune hyperparameters per technique: e.g., using strict greedy decoding (`temperature=0.0`) for Direct Zero-Shot answering, while using higher temperatures for Chain-of-Thought (CoT) to allow diverse reasoning paths. This could easily be combined with our N-runs Stability Analyzer to implement **Self-Consistency Decoding** (taking the majority vote of N high-temperature CoT runs).
7. **Scaling to Larger ALMs**: Evaluate larger, state-of-the-art Audio-Language Models (e.g., `mistralai/Voxtral-Small-24B-2507`). A larger parameter count in the audio encoder and the reasoning layers may naturally bridge some of the modality gap observed in the 7B model.
8. **Quantization Impact on Acoustic Reasoning**: Investigate the effect of weight quantization (e.g., FP8, INT4) on the ALM's zero-shot acoustic capabilities. While text LLMs are notoriously resilient to quantization, degrading the precision of the audio encoder's continuous latent space might have disproportionate effects on fine-grained acoustic feature extraction (like detecting subtle crackles).
9. **Top-Down Semantic Injection**: Test a two-step prompt architecture where a strong text-only LLM pre-processes clinical priors to synthesize a highly loaded "search query" (e.g., *"Listen specifically for early coarse crackles"*). Passing this to the ALM might force its attention mechanism to locate latent features that it would otherwise ignore in a pure zero-shot setting.
