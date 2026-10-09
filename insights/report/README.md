# Report: do audio-language models hear, or only read?

Read in order:

1. **[Methodology](01_METHODOLOGY.md):** how audio reaches a language model, the five models, the two
   datasets, every experiment and how it is scored.
2. **[Model profiles](02_MODEL_PROFILES.md):** each model's reading and hearing, and how each technique
   moves them.
3. **[Conclusions](03_CONCLUSIONS.md):** the main findings, what not to do, limitations and future work.

## Reproducing the numbers and figures

```bash
uv run python scripts/build_report_data.py      # run folders in data/ -> report/data/*.csv
uv run python scripts/build_report_figures.py   # report/data/*.csv -> report/figures/*.png
```

The scaling figure (`figures/fig_scaling.png`) comes from `scripts/plot_model_scaling.py` with
`--split-file insights/report/data/mmar_split.json`.

## Detailed appendices

The per-run analyses the report is built on, with full arm-by-arm tables:

- `../MMAR_V2_ANALYSIS.md`: MMAR, Voxtral-Small and Qwen3-Omni.
- `../ICBHI_V4_ANALYSIS.md`: ICBHI, both models, including the label-shuffle tests.
- `../CROSS_MODEL_ANALYSIS.md`: model-vs-model comparisons and the scaling study.
