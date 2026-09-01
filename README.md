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

## 🧪 Experiments Pipeline

We have structured a graduated prompt engineering experiment spanning 8 versions, to systematically measure which techniques yield the highest performance:

| Version | Experiment Name | Description |
| :--- | :--- | :--- |
| **v1** | `baseline` | Bare-minimum instruction ("Detect the disease"). |
| **v2** | `format_strict` | Baseline + Strict output formatting rules. |
| **v3** | `symptoms_dict` | Adds a medical dictionary describing the *symptoms* of each disease. |
| **v4** | `acoustic_dict` | Replaces symptoms with an **Acoustic Signatures** dictionary. |
| **v5** | `cot` | Adds **Chain-of-Thought** reasoning instructions. |
| **v6** | `few_shot` | Removes CoT, adds **Few-Shot Examples**. |
| **v7** | `cot_and_few_shot` | Combines CoT + Few-Shot Examples. |
| **v8** | `full_optimized` | v7 + explicit edge-case handling rules (e.g., corrupted audio). |

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
