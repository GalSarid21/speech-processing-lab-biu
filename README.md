# Do Audio-Language Models Hear, or Only Read?

**Prompting, scale and the limits of zero-shot audio understanding**

| | |
|---|---|
| **Course** | Digital Speech Processing |
| **Lecturer** | Prof. Sharon Ganot |
| **Institution** | Bar-Ilan University, Faculty of Engineering |
| **Semester** | Fall 2026 |

---

## Summary

Audio-language models answer questions about recordings, but it is rarely measured whether they use the
**sound** or only the **words** spoken in it. We test five open models of 3B to 32B parameters on two lanes:

- **In-Knowledge:** 199 English-speech questions from **MMAR**, the kind of audio the models were trained on.
- **Out-of-Knowledge:** 168 lung-sound recordings from **ICBHI 2017**, a domain none of them saw.

A text-only model reading a Whisper transcript splits the MMAR questions into **143 transcript-solvable**
and **56 audio-dependent** ones, so *reading the words* and *hearing the sound* are measured separately.
Each model is tested with about **24 prompting techniques** (chain-of-thought, transcripts, acoustic
evidence, few-shot and retrieved examples, audio presentation, calibrated scoring, option voting) plus a
control that removes the sound but keeps the transcript.

### Key findings

1. **Model choice matters more than prompting.** Baselines span 17.6 points; the best technique adds at
   most 7.5; no prompting gain survives Holm correction (the only two effects that do are losses).
2. **Bigger is better within a family, not across families.** Qwen2.5-Omni-7B ties Voxtral-Small-24B; the
   largest model (Step-Audio-R1.1, 32B) does not beat Qwen3-Omni (~3B active parameters per token).
3. **Size buys reading, not hearing.** Step-Audio-R1.1 reads best (88.1%) and hears at chance (37.5%).
4. **Hearing follows how the audio encoder was trained.** Only the two Qwen models hear above chance;
   removing the sound takes the best listener, Qwen3-Omni, to chance.
5. **The audio does not hurt reading; written text hurts hearing.** Adding transcripts or measured
   acoustics to a model that already hears costs it up to 14 points on audio-dependent questions.
6. **On unfamiliar audio every model fails.** No model and no technique extracts a diagnosis from lung
   sounds; the models fall back on answer priors and on words in the prompt.

| Model | Overall | Reading the words | Hearing the sound |
|---|---|---|---|
| Voxtral-Mini-3B | 59.3 | 73.4 | 23.2 (below chance) |
| Qwen2.5-Omni-7B | 65.8 | 73.4 | 46.4 |
| Voxtral-Small-24B | 65.8 | 75.5 | 41.1 |
| Qwen3-Omni-30B-A3B | **76.9** | 87.4 | **50.0** |
| Step-Audio-R1.1 | 73.9 | **88.1** | 37.5 |

*MMAR baseline accuracy (%), majority of three runs. Chance: 32.0% overall, 33.2% on hearing questions.*

![Words vs sound](insights/report/figures/fig_words_vs_sound.png)

![Model scaling](insights/report/figures/fig_scaling.png)

The report also lists **what not to do** (e.g. chain-of-thought on small non-reasoning models: −13.1
points) and proposes a **late-fusion** design of a listener and a reader, whose upper bound is 83–86%
against 76.9% for the best single model.

---

## Models and data

| Role | Model / dataset |
|---|---|
| Audio-language models | Voxtral-Mini-3B-2507, Qwen2.5-Omni-7B, Voxtral-Small-24B-2507, Qwen3-Omni-30B-A3B-Instruct, Step-Audio-R1.1 |
| Text-only control | Gemma-4-26B-A4B-it |
| LLM judge (secondary metric) | Qwen3.8-27B-FP8 |
| Evidence extraction | Whisper-large-v3 (transcripts), pyannote speaker-diarization-3.1, AST (AudioSet tagger), LAION CLAP and LCO-Embedding-Omni-7B (retrieval) |
| Datasets | MMAR (English-speech subset, 199 questions); ICBHI 2017 (168 evaluation recordings) |

## Hardware and software

- **Hardware:** one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU (96 GB VRAM). BF16 weights, which caps
  the models at about 32B parameters.
- **Inference:** [vLLM](https://github.com/vllm-project/vllm) (in-process for four models; Step-Audio-R1.1
  served through StepFun's vLLM build over an OpenAI-compatible HTTP API).
- **Environment:** Python 3.12, managed with [uv](https://github.com/astral-sh/uv); pandas, NumPy, SciPy,
  Matplotlib for analysis; Pydantic configuration; pytest and ruff.
- **Runs:** Google Colab with the GPU above; results written to Google Drive and copied into `data/`.

## Repository layout

| Path | Contents |
|---|---|
| `src/speech_processing/` | pipeline: data loading, audio engines, prompts, scoring, statistics |
| `src/speech_processing/runners/` | experiment definitions (`mmar.py`, `icbhi.py`) |
| `main.py` | entry point |
| `scripts/` | evidence extraction, analysis, report tables and figures |
| `data/` | raw outputs of every run (`speech-processing-res-*`) |
| `insights/report/` | the report in Markdown, its data tables and figures |
| `insights/*_ANALYSIS.md` | detailed per-dataset and cross-model analyses (appendices) |
| `tests/` | unit and integration tests |

## Reproducing

```bash
uv sync                                                    # install (vLLM installs on Linux only)
uv run python main.py --dataset mmar --experiment v1 \
    --sample-ids-file data/mmar_en_speech_test_ids.txt --runs 3 --output-dir results/mmar
uv run python main.py --dataset icbhi --experiment r0 --runs 3 --output-dir results/icbhi
uv run python main.py --dataset mmar --experiment v1 --audio-model-id Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --sample-ids-file data/mmar_en_speech_test_ids.txt --runs 3 --output-dir results/mmar_qwen3

uv run python scripts/build_report_data.py       # run folders in data/ -> insights/report/data/*.csv
uv run python scripts/build_report_figures.py    # tables -> insights/report/figures/*.png
uv run pytest                                    # tests
```

The MMAR id files (`mmar_en_speech_test_ids.txt`, `mmar_en_speech_few_shot_ids.txt`) are not committed.

---

## References

1. Z. Ma et al. **MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix.** NeurIPS 2025 Datasets and Benchmarks Track. arXiv:2505.13032.
2. B. M. Rocha et al. **An open access database for the evaluation of respiratory sound classification algorithms.** *Physiological Measurement* 40(3):035001, 2019.
3. A. Radford et al. **Robust Speech Recognition via Large-Scale Weak Supervision.** ICML 2023.
4. A. H. Liu et al. **Voxtral.** arXiv:2507.13264, 2025.
5. J. Xu et al. **Qwen2.5-Omni Technical Report.** arXiv:2503.20215, 2025.
6. J. Xu et al. **Qwen3-Omni Technical Report.** arXiv:2509.17765, 2025.
7. F. Tian et al. **Step-Audio-R1 Technical Report.** arXiv:2511.15848, 2025.
8. Gemma Team. **Gemma 4 Technical Report.** arXiv:2607.02770, 2026.
9. W. Kwon et al. **Efficient Memory Management for Large Language Model Serving with PagedAttention.** SOSP 2023.
10. A. Plaquet, H. Bredin. **Powerset multi-class cross entropy loss for neural speaker diarization.** Interspeech 2023.
11. H. Bredin. **pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe.** Interspeech 2023.
12. Y. Gong, Y.-A. Chung, J. Glass. **AST: Audio Spectrogram Transformer.** Interspeech 2021.
13. Y. Wu et al. **Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation.** ICASSP 2023.
14. C. Xiao et al. **Scaling Language-Centric Omnimodal Representation Learning.** arXiv:2510.11693, 2025.
15. Q. McNemar. **Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages.** *Psychometrika* 12(2):153–157, 1947.
16. S. Holm. **A Simple Sequentially Rejective Multiple Test Procedure.** *Scandinavian Journal of Statistics* 6(2):65–70, 1979.
17. Qwen Team. **Qwen3.8-Max: A New Bar for Coding and Cowork.** Blog post, 2026 (model: Qwen/Qwen3.8-27B-FP8).

<details>
<summary><b>BibTeX</b></summary>

```bibtex
@inproceedings{ma2025mmar,
  title     = {{MMAR}: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix},
  author    = {Ma, Ziyang and Ma, Yinghao and Zhu, Yanqiao and Yang, Chen and Chao, Yi-Wen and Xu, Ruiyang and Chen, Wenxi and Chen, Yuanzhe and Chen, Zhuo and Cong, Jian and Li, Kai and Li, Keliang and Li, Siyou and Li, Xinfeng and Li, Xiquan and Lian, Zheng and Liang, Yuzhe and Liu, Minghao and Niu, Zhikang and Wang, Tianrui and Wang, Yuping and Wang, Yuxuan and Wu, Yihao and Yang, Guanrou and Yu, Jianwei and Yuan, Ruibin and Zheng, Zhisheng and Zhou, Ziya and Zhu, Haina and Xue, Wei and Benetos, Emmanouil and Yu, Kai and Chng, Eng-Siong and Chen, Xie},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  volume    = {38},
  doi       = {10.52202/085713-2252},
  year      = {2025},
  note      = {arXiv:2505.13032}
}

@article{rocha2019icbhi,
  title   = {An open access database for the evaluation of respiratory sound classification algorithms},
  author  = {Rocha, Bruno M. and Filos, Dimitris and Mendes, Lu{\'i}s and Serbes, Gorkem and Ulukaya, Sezer and Kahya, Yasemin P. and Jakovljevi{\'c}, Nik{\v{s}}a and Turukalo, Tatjana Lon{\v{c}}ar and Vogiatzis, Ioannis M. and Perantoni, Eleni and Kaimakamis, Evangelos and Natsiavas, Pantelis and Oliveira, Ana and J{\'a}come, Cristina and Marques, Alda and Maglaveras, Nicos and Paiva, Rui Pedro and Chouvarda, Ioanna and de Carvalho, Paulo},
  journal = {Physiological Measurement},
  volume  = {40},
  number  = {3},
  pages   = {035001},
  year    = {2019},
  doi     = {10.1088/1361-6579/ab03ea}
}

@inproceedings{radford2023whisper,
  title     = {Robust Speech Recognition via Large-Scale Weak Supervision},
  author    = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages     = {28492--28518},
  year      = {2023},
  volume    = {202},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}

@misc{liu2025voxtral,
  title         = {Voxtral},
  author        = {Liu, Alexander H. and Ehrenberg, Andy and Lo, Andy and others},
  year          = {2025},
  eprint        = {2507.13264},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD}
}

@misc{xu2025qwen25omni,
  title         = {Qwen2.5-Omni Technical Report},
  author        = {Jin Xu and Zhifang Guo and Jinzheng He and Hangrui Hu and Ting He and Shuai Bai and Keqin Chen and Jialin Wang and Yang Fan and Kai Dang and Bin Zhang and Xiong Wang and Yunfei Chu and Junyang Lin},
  year          = {2025},
  eprint        = {2503.20215},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

@misc{xu2025qwen3omni,
  title         = {Qwen3-Omni Technical Report},
  author        = {Xu, Jin and Guo, Zhifang and Hu, Hangrui and others},
  year          = {2025},
  eprint        = {2509.17765},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

@misc{tian2025stepaudior1,
  title         = {Step-Audio-R1 Technical Report},
  author        = {Fei Tian and Xiangyu Tony Zhang and Yuxin Zhang and Haoyang Zhang and Yuxin Li and Daijiao Liu and Yayue Deng and Donghang Wu and Jun Chen and Liang Zhao and Chengyuan Yao and Hexin Liu and Eng Siong Chng and Xuerui Yang and Xiangyu Zhang and Daxin Jiang and Gang Yu},
  year          = {2025},
  eprint        = {2511.15848},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

@misc{gemmateam2026gemma4,
  title         = {Gemma 4 Technical Report},
  author        = {{Gemma Team}},
  year          = {2026},
  eprint        = {2607.02770},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

@inproceedings{kwon2023efficient,
  title     = {Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author    = {Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle = {Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year      = {2023}
}

@inproceedings{plaquet2023powerset,
  author    = {Alexis Plaquet and Herv{\'e} Bredin},
  title     = {{Powerset multi-class cross entropy loss for neural speaker diarization}},
  booktitle = {Proc. INTERSPEECH 2023},
  pages     = {3222--3226},
  year      = {2023},
  doi       = {10.21437/Interspeech.2023-205}
}

@inproceedings{bredin2023pyannote,
  author    = {Herv{\'e} Bredin},
  title     = {{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}},
  booktitle = {Proc. INTERSPEECH 2023},
  pages     = {1983--1987},
  year      = {2023},
  doi       = {10.21437/Interspeech.2023-105}
}

@inproceedings{gong2021ast,
  author    = {Yuan Gong and Yu-An Chung and James Glass},
  title     = {{AST: Audio Spectrogram Transformer}},
  booktitle = {Proc. Interspeech 2021},
  pages     = {571--575},
  year      = {2021},
  doi       = {10.21437/Interspeech.2021-698}
}

@inproceedings{wu2023clap,
  title     = {Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  author    = {Wu, Yusong and Chen, Ke and Zhang, Tianyu and Hui, Yuchen and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {1--5},
  year      = {2023},
  doi       = {10.1109/ICASSP49357.2023.10095969}
}

@article{xiao2025scaling,
  title   = {Scaling Language-Centric Omnimodal Representation Learning},
  author  = {Xiao, Chenghao and Chan, Hou Pong and Zhang, Hao and Xu, Weiwen and Aljunied, Mahani and Rong, Yu},
  journal = {arXiv preprint arXiv:2510.11693},
  year    = {2025}
}

@article{mcnemar1947,
  title   = {Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages},
  author  = {McNemar, Quinn},
  journal = {Psychometrika},
  volume  = {12},
  number  = {2},
  pages   = {153--157},
  year    = {1947},
  doi     = {10.1007/BF02295996}
}

@article{holm1979,
  title   = {A Simple Sequentially Rejective Multiple Test Procedure},
  author  = {Holm, Sture},
  journal = {Scandinavian Journal of Statistics},
  volume  = {6},
  number  = {2},
  pages   = {65--70},
  year    = {1979},
  url     = {https://www.jstor.org/stable/4615733}
}

@misc{qwen38,
  title  = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
  author = {{Qwen Team}},
  url    = {https://qwen.ai/blog?id=qwen3.8},
  month  = {August},
  year   = {2026}
}
```

</details>
