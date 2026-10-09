# MMAR V2 — Results and Conclusions (In-Knowledge lane)

**Model:** Voxtral-Small-24B-2507 (vLLM), on one NVIDIA RTX PRO 6000 Blackwell (96 GB). Text-only arms
use Gemma-4-26B; the judge (secondary metric only) is Qwen3.8-27B-FP8.
**Data:** the 200-item English-speech MMAR test set, minus one item whose gold answer names both options
(`BV1NtQuYTELJ…`, see `MMAR_EXCLUDED_ITEMS`): **n = 199**. 142 items have 4 options, 55 have 2, 2 have 3.
**Runs:** 3 per experiment; contrastive arms (v24–v26) are deterministic, so 1 run.
**Source:** Drive `speech-processing-res-mmar-v2/` (local copy: `data/speech-processing-res-mmar-v2/`),
runs of 2026-10-06; the six transcript arms re-run on 2026-10-07 (the newest run of each is used).

MMAR is the *In-Knowledge* lane: everyday English speech, the domain the model was trained on. It is the
counterpart of ICBHI (`ICBHI_V4_ANALYSIS.md`), where the same model and techniques face audio it never saw.

## Verification

- `comparison.md` was recomputed from the raw outputs with `scripts/analyze_runs.py`, using the newest
  run of every experiment; all 25 rows are identical to the copy on Drive.
- The primary metric is the deterministic parser. Unparsed answers are ≤ 4% everywhere except v22
  (8.5%). Parser and judge agree on 94–100% of answers.
- Run-to-run stability is high: the baseline gives the same answer in all 3 runs on 93.5% of items.
- The v12 log line `memory allocation failed with OOM` was a recovered warning; v12 finished all runs.

## Data-quality finding: the first transcript file was empty (fixed and re-run)

The Whisper transcript file used on 2026-10-06 contained the single word `"you"` for every item. It
was built while the audio sat in the wrong folder. Every load failed, the transcription script silently
substituted silence, and Whisper transcribes silence as "you". The six arms that read it (v3, v4,
v9, v10, v13, v16) were re-run on 2026-10-07 with a rebuilt file; the numbers below are from those
re-runs. The transcription script now stops on missing audio, the loader rejects a transcript file whose
entries are mostly identical, and the notebook detects and repeats affected runs.

The broken v13 run is still informative as an **accidental control**. Gemma saw only the question and
options and scored **45.2%** (chance 32%). The question text alone carries some answer signal.

## How to read the numbers

- **Accuracy** = share of questions answered correctly. **Chance is 32.0%** (mean of 1/number of
  options). The baseline **v1** scores **65.8%**.
- **p** = exact McNemar test against v1 on the same 199 items. With 25 comparisons, a p-value must
  be corrected (Holm) before it counts.
- **Split (from v13, a text-only model reading the Whisper transcript):** **TS** = the 143 items v13
  answers correctly from the transcript alone; **AD** = the 56 it does not. Chance is 31.5% on TS and
  33.2% on AD. Because AD is defined as the items one text model got wrong, it also contains questions
  that are simply hard, so treat it as an approximation of "needs more than the words".

## Results

| exp | technique | acc | Δ vs v1 | gained/lost | p | Holm | TS (143) | AD (56) |
|---|---|---|---|---|---|---|---|---|
| v1 | **baseline** | **65.8** | — | — | — | — | 75.5 | 41.1 |
| v16 | retrieved few-shot + transcript | 73.4 | +7.5 | 25/10 | 0.017 | 0.40 | 86.7 | 39.3 |
| v27 | 5 shuffled option orders, majority vote | 71.9 | +6.0 | 16/4 | 0.012 | 0.30 | 81.8 | 46.4 |
| v13 | *text-only model, transcript, no audio* | 71.9 | +6.0 | 35/23 | 0.148 | 1.0 | 100 | 0 |
| v7 | few-shot, 3 audio examples | 71.4 | +5.5 | 21/10 | 0.071 | 1.0 | 80.4 | 48.2 |
| v8 | few-shot audio + CoT | 71.4 | +5.5 | 25/14 | 0.108 | 1.0 | 86.0 | 33.9 |
| v9 | few-shot audio + transcript | 70.9 | +5.0 | 24/14 | 0.143 | 1.0 | 83.2 | 39.3 |
| v5 | few-shot, text-only examples | 70.9 | +5.0 | 21/11 | 0.110 | 1.0 | 82.5 | 41.1 |
| v14 | few-shot, retrieved by audio similarity | 70.4 | +4.5 | 21/12 | 0.163 | 1.0 | 82.5 | 39.3 |
| v10 | few-shot audio + transcript + CoT | 69.8 | +4.0 | 27/19 | 0.302 | 1.0 | 81.8 | 39.3 |
| v3 | audio + transcript | 68.8 | +3.0 | 18/12 | 0.362 | 1.0 | 80.4 | 39.3 |
| v6 | few-shot text + CoT | 68.8 | +3.0 | 29/23 | 0.488 | 1.0 | 81.8 | 35.7 |
| v23 | two-pass: locate the segment, then answer | 67.8 | +2.0 | 16/12 | 0.572 | 1.0 | 79.0 | 39.3 |
| v28 | audio placed before the instruction | 67.3 | +1.5 | 12/9 | 0.664 | 1.0 | 77.6 | 41.1 |
| v24 | letter log-prob scoring (α=0) | 66.8 | +1.0 | 5/3 | 0.727 | 1.0 | 76.9 | 41.1 |
| v18 | audio + diarized transcript | 66.3 | +0.5 | 14/13 | 1.0 | 1.0 | 80.4 | 30.4 |
| v11 | role prompt | 65.8 | 0.0 | 9/9 | 1.0 | 1.0 | 79.0 | 32.1 |
| v21 | *text-only model:* diarized transcript + features | 65.8 | 0.0 | 29/29 | 1.0 | 1.0 | 83.9 | 19.6 |
| v25 | contrastive scoring (α=0.5) | 65.8 | 0.0 | 7/7 | 1.0 | 1.0 | 75.5 | 41.1 |
| v19 | audio + acoustic features | 65.3 | −0.5 | 11/12 | 1.0 | 1.0 | 78.3 | 32.1 |
| v2 | chain-of-thought | 65.3 | −0.5 | 12/13 | 1.0 | 1.0 | 77.6 | 33.9 |
| v4 | audio + transcript + CoT | 65.3 | −0.5 | 21/22 | 1.0 | 1.0 | 79.7 | 28.6 |
| v12 | two-turn: describe, then answer | 64.8 | −1.0 | 16/18 | 0.864 | 1.0 | 77.6 | 32.1 |
| v20 | audio + diarized transcript + features | 64.3 | −1.5 | 17/20 | 0.743 | 1.0 | 78.3 | 28.6 |
| v26 | contrastive scoring (α=1.0) | 61.3 | −4.5 | 11/20 | 0.150 | 1.0 | 70.6 | 37.5 |
| v22 | audio cut into chunks | 59.3 | −6.5 | 13/26 | 0.053 | 1.0 | 69.9 | 32.1 |

## Conclusions

1. **The model's competence here is mostly the spoken words.** Voxtral answers 65.8% against 32%
   chance. But a text-only model given only the Whisper transcript does at least as well (71.9%, +6.0,
   n.s.), and the question text alone already reaches 45.2% (the accidental control above). On the 56
   items the transcript does not answer, Voxtral scores 41.1% against 33.2% chance (p = 0.13). That is
   no reliable evidence that it uses the non-verbal audio (tone, speaker identity, acoustic events)
   beyond what the words carry.

2. **Audio and transcript are complementary, not redundant.** v1 and v13 are both right on 108 items;
   23 are right only with audio and 35 only with the transcript. An oracle that always picked the right
   one would reach 83.4%. Simply adding the transcript to the audio prompt (v3, 68.8%) captures little
   of that, and is no better than the transcript alone (p = 0.48).

3. **Prompting moves the model only at the margin, and nothing survives correction.** The largest gain
   is +7.5 points (v16), and no improvement holds after Holm correction across 25 comparisons (v27:
   0.30; v16: 0.40). With n = 199 the 95% CI on accuracy is about ±6.5 points, so gains of this size
   cannot be resolved.

4. **Few-shot is the one consistent direction, and its benefit is not acoustic.** All eight few-shot
   arms gain +3.0 to +7.5 (pooled +5.0, p = 0.04; a post-hoc grouping, so indicative only).
   Text-only examples (v5, +5.0) help as much as audio examples (v7, +5.5). The examples teach answer
   format and option use, not what the audio sounds like. The best single arm (v16) combines retrieved
   examples with the transcript.

5. **Choice-order voting gives the cleanest gain.** v27 gains 16 items and loses only 4, and does best
   on the AD items together with v7 (46–48%). The baseline over-picks A (82 answers vs 73 gold) and
   under-picks D (27 vs 38). Voting over shuffled option orders evens this out (A 76, D 30), which is
   consistent with removing position bias.

6. **Structured audio evidence does not help.** Diarized transcripts (v18), acoustic features (v19), or
   both (v20) stay within ±1.5 points of the baseline. In text form the diarized-plus-features input
   (v21, 65.8%) is worse than the plain Whisper transcript (v13, 71.9%; p = 0.06).

7. **Some audio-side changes hurt, and the rest have no effect.**
   - Chunking the clip (v22, −6.5, 8.5% unparsed) and full contrastive calibration (v26, −4.5) lose
     accuracy.
   - Scoring the answer letter directly (v24) equals free generation (66.8 vs 65.8).
   - CoT, role prompts, decomposition, two-pass localisation and audio-first ordering: −1.0 to +2.0.

**One-line summary:** in the In-Knowledge lane Voxtral answers about twice as well as chance, but almost
all of that comes from understanding the spoken words: a transcript alone does as well. Prompting adds at
most ~5–7 points, by fixing answer format and position bias, and none of the gains survives correction.

**Across both lanes:** the model's working ability is linguistic. Where the answer is in the words
(MMAR), it performs well and prompting fine-tunes it. Where the answer is in non-verbal acoustics,
it is near chance in both datasets: lung sounds in ICBHI and the not-transcript-answerable MMAR items.
Prompting techniques do not create that acoustic ability.

## Limitations

- n = 199 English-speech items; 95% CI about ±6.5 points. The AD subset has only 56 items, and the
  Signal (2) and Perception (17) layers are too small to compare.
- The TS/AD split is defined by one text model's answers, so AD mixes "needs the audio" with "hard".
- 25 arms against one baseline; individual p-values below 0.05 do not survive correction.
- MMAR gold errors: one test item was excluded (its gold answer names both options), and one
  (`BV1NX4y1p7Xq…`) is scored against a corrected gold, because its stored answer abbreviates option D.
  The other three MMAR gold errors are outside this test set (see `MMAR_GOLD_CORRECTIONS` /
  `MMAR_EXCLUDED_ITEMS`).

## Second model

Qwen3-Omni-30B-A3B scores 76.9% at baseline (+11.1 over Voxtral, Holm p = 0.03) and is the only model above chance on the audio-dependent items (50.0% vs 33.2%), but no prompting technique improves it over its own baseline. Details: `CROSS_MODEL_ANALYSIS.md`.
