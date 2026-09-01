# Speech Processing Lab BIU: Lung Sound Diagnostics

Codebase for the final project of the speech processing course, taught by Prof. Sharon Ganot at Bar Ilan University.

## 🎯 Project Goal

The primary objective of this project is to improve the zero-shot and few-shot diagnostic capabilities of state-of-the-art **Audio-Language Models (ALMs)** using advanced prompt engineering techniques, *without* the need for resource-intensive fine-tuning.

Specifically, we aim to bridge the gap between abstract audio perception and medical diagnosis by enriching the model's context window with:
- **Medical Dictionaries**: Mapping human symptoms to diseases.
- **Acoustic Signatures**: Teaching the model the exact acoustic characteristics of diseases (e.g., "polyphonic wheezes" vs "fine crackles").
- **Chain-of-Thought (CoT)**: Forcing the model to explicitly describe the raw audio before classifying it.
- **Few-Shot Learning**: Providing concrete examples of acoustic mappings.

## 🧠 Models Used

This repository implements an **LLM-as-a-Judge** pipeline to evaluate the Audio model's outputs.

1. **Audio Inference Model**: [`Qwen/Qwen2-Audio-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
   - A multimodal audio-language model capable of directly ingesting raw audio waveforms and processing them alongside text prompts.
2. **Judge Evaluator Model**: [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
   - A highly capable 27-billion parameter language model used to grade the Audio model's outputs. It evaluates Acoustic Accuracy, Diagnostic Accuracy, and Hallucination Rates, returning structured JSON reports. (Runs on vLLM using FP8 quantization).

## 💻 Hardware Environment

The pipeline was developed and tested on a compute environment with the following specifications:
- **GPU**: NVIDIA A100-SXM4 (80GB VRAM)
- **CPU**: Intel® Xeon® CPU @ 2.20GHz (12 cores)
- **Architecture**: x86_64
- **CUDA Version**: 13.0

*(Note: The 80GB of VRAM is highly recommended when running the 27B FP8 model with a large context window, alongside the 7B multimodal audio model in sequential phases).*

## 📊 Dataset

We use the **ICBHI 2017 Respiratory Sound Database**, fetched dynamically via HuggingFace (`DynamicSuperb/RespiratorySoundClassification_ICBHI2017`). The dataset includes recordings of patients with various respiratory diseases (COPD, Asthma, Pneumonia, Bronchiolitis, etc.) and healthy subjects.

## 🧪 The Prompt Engineering Journey (Experiments V1-V11)

We structured a graduated prompt engineering experiment spanning 11 versions to systematically measure which techniques yield the highest performance. Here is the documentation of our findings and the motivations behind each step.

### 1. The Baselines & Formatting
*   **V1 (`baseline`)**: Bare-minimum instruction ("Detect the disease in this lung sound audio.").
    *   **Goal**: Establish zero-shot capability.
    *   **Result**: High hallucinations (24%), moderate diagnostic accuracy (27.2%), poor acoustic accuracy (12.5%).
*   **V2 (`format_strict`)**: Baseline + Strict output formatting rules ("Output exactly as...").
    *   **Goal**: Force a structured JSON/string output.
    *   **Result**: **Total failure (0% accuracy)**. Strict text formatting caused the multimodal model to hyper-fixate on the string, crashing its reasoning.

### 2. The Language of Audio
*   **V3 (`symptoms_dict`)**: Added a medical dictionary describing the *symptoms* of each disease (e.g., "fever", "cough").
    *   **Goal**: Give the model medical knowledge.
    *   **Result**: **Hallucinations skyrocketed to 88%**. The model started "hearing" textual symptoms like chest tightness in raw audio. Diagnostic accuracy crashed to 7.8%.
*   **V4 (`acoustic_dict`)**: Replaced symptoms with an **Acoustic Signatures** dictionary (e.g., "polyphonic wheezes", "coarse crackles").
    *   **Goal**: Provide audio-specific grounding.
    *   **Result**: **Massive improvement.** Acoustic accuracy jumped to 42.5%. We proved we must "speak to the model in audio terms".

### 3. Chain-of-Thought (CoT)
*   **V5 (`cot`)**: Added step-by-step reasoning instructions forcing the model to describe the raw audio *before* classifying.
    *   **Goal**: Prevent impulsive guessing by forcing logical audio grounding.
    *   **Result**: Acoustic accuracy hit 52.7%, and Diagnostic accuracy jumped to 37.0%.

### 4. The Multimodal Few-Shot Challenge
*   **V6 (`few_shot`) & V7 (`cot_and_few_shot`)**: Added text-based few-shot examples using pseudo-text placeholders like `[Audio containing crackles]`.
    *   **Goal**: Teach the model format and reasoning via examples.
    *   **Result**: **Total failure (6% diag acc, 82% hallucinations)**. Pseudo-text examples broke the multimodal alignment, causing the model to spam-guess diseases blindly.
*   **V8 (`full_optimized`)**: Skipped few-shot entirely, utilizing V5 (CoT) + aggressive anti-hallucination guardrails.
*   **V9 (`authentic_few_shot`)**: Engineered a complex **True Multimodal Few-Shot** pipeline. We dynamically loaded real `.wav` files of a Healthy and a COPD patient from the train split and passed them as raw audio tensors alongside CoT examples. Added strict anti-hallucination guardrails.
    *   **Goal**: Restore multimodal alignment using authentic audio.
    *   **Result**: Hallucinations crashed to 35%, but Diagnostic Accuracy plummeted to 11.8% because the guardrails were too aggressive, scaring the model into defaulting to "Healthy".

### 5. The Holy Grail (V10)
*   **V10 (`authentic_few_shot_holistic`)**: The Kitchen Sink. We created a **Holistic Dictionary** (combining both Clinical Context and Acoustic Signatures), combined with Authentic Few-Shot Audio Tensors, Chain-of-Thought, and Guardrails.
    *   **Goal**: Bridge the model's text-LLM brain with its audio-encoder brain without causing it to panic.
    *   **Result**: **"THE MOTHER OF ALL BOOMS"**. 
        *   Avg Acoustic Accuracy: **64.00%**
        *   Avg Diagnostic Accuracy: **60.00%**
        *   Hallucination Rate: **35.00%**
*   **V11 (`authentic_few_shot_no_guardrails`)**: V10 but removed the Holistic Dictionary and Guardrails to verify their necessity.
    *   **Goal**: Ablation study.
    *   **Result**: Diagnostic Accuracy crashed to 0%, proving the Holistic Dictionary and Guardrails in V10 were the ultimate missing link.

## 🚀 Setup & Usage

### 1. Environment Setup
This project uses `uv` for modern, blazing-fast Python package management. 

*(Note: The Judge model relies on `vllm`, which requires a Linux environment and an NVIDIA GPU (e.g., A100) to run natively).*

```bash
# Install dependencies
uv sync
```

### 2. Running an Experiment
Execute the end-to-end pipeline (Audio Inference -> VRAM Cleanup -> Judge Evaluation) in a single command. Use the `--experiment` flag to choose the prompt version.

```bash
uv run python scripts/run_pipeline.py --experiment v5 --num-samples 100
```
*Results will be saved in a timestamped folder, e.g., `results/cot_20260901_183000/`.*

### 3. Analyzing Results
Generate a comprehensive markdown report, confusion matrices, and distribution graphs using the analysis script:

```bash
uv run python scripts/analyze_results.py --run-dir results/cot_20260901_183000
```
The graphs and tables will be output into the `analysis/` subfolder within your run directory.
