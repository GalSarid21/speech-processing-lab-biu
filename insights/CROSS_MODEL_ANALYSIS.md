# Cross-model analysis — Voxtral-Small-24B vs Qwen3-Omni-30B-A3B

**Models:** Voxtral-Small-24B-2507 (dense, 24B) and Qwen3-Omni-30B-A3B-Instruct (mixture of experts:
30B total, about 3B active per token; vLLM loads its "thinker" only). Both run under vLLM on one NVIDIA
RTX PRO 6000 Blackwell (96 GB), with the same prompts, items, 3 runs per arm and the same 5-clip cap.
**Data:** ICBHI n = 168 (Out-of-Knowledge lane), MMAR n = 199 (In-Knowledge lane), same items as
`ICBHI_V4_ANALYSIS.md` and `MMAR_V2_ANALYSIS.md`.
**Source:** Drive `speech-processing-res-{icbhi-v4,mmar-v2}-qwen3omni/` (runs of 2026-10-07; local copies
in `data/`). Cross-model tables: `scripts/compare_models.py`, saved as `cross_model_{icbhi,mmar}.md` in
the Qwen folders.
**Text-only arms** (Gemma-4-26B) do not depend on the audio model and were not repeated.

## Verification

- Every Qwen arm is complete: 168 items (ICBHI) and 199 (MMAR) in every run. The CUDA allocator
  "OOM" warnings during the run were recovered retries; no item was lost.
- Unparsed answers: 0% on every ICBHI arm, ≤ 2% on every MMAR arm except CoT (4.5%).
- `comparison.md` and the cross-model tables were recomputed locally from the raw outputs; the numbers
  match the Colab output.
- **Language tracking:** 0 of all answers, in both models and both datasets, switched out of English,
  CoT arms included. The only Chinese text is quoting: English answers that repeat Mandarin words heard
  in two MMAR clips. The Qwen2 drift into Chinese did not reproduce with Qwen3-Omni on English prompts.

## ICBHI — both models fail, in different ways

| | Voxtral | Qwen3-Omni |
|---|---|---|
| Baseline BA (chance 16.7) | 7.5 | 16.7 |
| Most common answer, baseline | 46% "LRTI" | **100%** "No potential disease detected" |
| Arms with one answer ≥ 90% of the time | 7 of 27 (6 "Asthma", CoT "no disease") | 15 of 27 (all "no disease") |
| Silence (c1) vs real audio (r0) | different answers, both near zero BA | **identical answers on all 168 items** |
| Feature text added (a1a, silence + features) | flips to "Asthma" | flips to "Bronchiectasis" with audio *and* with silence |
| Best arm | knn_text_evidence 23.7 | knn_text_evidence 23.2 |

1. **Qwen's apparent wins are a constant answer, not hearing.** The cross-model table shows Qwen ahead
   by +9 to +17 BA on most arms, many significant after Holm. They are not improvements: Qwen answers
   "No potential disease detected" to nearly everything, which scores exactly 16.7 BA, while Voxtral's
   answers fall into the rare Asthma/LRTI classes and score below that. Shuffled option orders give the
   same answer (99%), so this is a semantic default, not a letter bias. The "top answer %" column in
   `compare_models.py` was added to make this visible.
2. **Qwen ignores the lung sounds entirely.** Silence and the real recording produce the same answer on
   every item. Feature text, not audio, is what changes its answer, as it was for Voxtral.
3. **The one arm with signal is the same for both models:** retrieval of labelled neighbours
   (knn_text_evidence, 23.2–23.7 BA). The information there comes from the retrieved labels, not from
   the model's listening.

**ICBHI conclusion:** a second model from a different vendor confirms the Out-of-Knowledge finding. Neither
model can diagnose from lung sounds zero-shot, and no prompting technique changes that for either one.

## MMAR — Qwen is clearly stronger, prompting adds nothing

| Items | n | Qwen | Voxtral | text-only (Gemma, transcript) | chance | Qwen vs Voxtral (won/lost, p) |
|---|---|---|---|---|---|---|
| All | 199 | **76.9** | 65.8 | 71.9 | 32.0 | 34/12, p = 0.002 (Holm 0.03) |
| Transcript-solvable | 143 | 87.4 | 75.5 | 100 (by definition) | 31.5 | 24/7, p = 0.003 |
| Audio-dependent | 56 | **50.0** | 41.1 | 0 (by definition) | 33.2 | 10/5, p = 0.30 |

1. **The better model gives +11 points; techniques give about 0.** Qwen beats Voxtral on the baseline
   and on all 24 shared arms (+3.5 to +16.6). The largest gap (+16.6, chunked audio) is Voxtral losing
   accuracy there, not Qwen gaining. Against its own baseline, no Qwen technique improves: every arm is
   between −5.0 and +3.0 points and nothing is significant (best: role prompt, 79.9, +3.0, p = 0.18).
2. **The techniques that helped Voxtral were fixing Voxtral's weaknesses.** Few-shot gave Voxtral +5
   (format and option use); for Qwen it gives −1.5 to +0.5. Adding a Whisper transcript gives Qwen +0.0:
   it already gets the words from the audio. The arms that hurt Voxtral (diarized transcript, full
   contrastive scoring) also hurt Qwen (−5.0, −4.0).
3. **Most of Qwen's lead is understanding the words.** Its advantage comes mainly from the
   transcript-solvable items (+11.9 points).
4. **First evidence of hearing beyond the words.** On the 56 items a transcript does not answer, Qwen
   scores 50.0% against 33.2% chance (one-sided p = 0.007); Voxtral did not reach significance (41.1%,
   p = 0.13). Qwen's lead over Voxtral on these items is not significant (p = 0.30), and the subset also
   contains questions that are simply hard.
5. **Qwen equals, but does not beat, reading the transcript.** 76.9 vs 71.9 for the text-only model
   (p = 0.18).

## Background: what "mixture of experts" means for an audio model

Every model here has the same three parts:

1. **Audio encoder (the "ear").** A Whisper-style transformer that turns the waveform into a sequence of
   vectors, roughly 12.5–25 per second of audio.
2. **Projector.** A small layer that maps those vectors into the language model's embedding space.
3. **Language model (the "brain").** It reads the audio vectors as if they were tokens, next to the text
   prompt, and writes the answer.

**Mixture of experts (MoE) is a property of the language model only.** In an MoE layer, the feed-forward
block is split into many "experts", and a learned router sends each token to a few of them (Qwen3-Omni:
about 3B of 30B parameters are active per token). The attention layers stay shared. The audio encoder is
an ordinary dense network: every audio frame passes through all of its parameters, whatever the language
model looks like.

What that means in practice:

- **The ear is not made smaller by MoE.** Qwen3-Omni's "3B active" describes the reasoning part, not how
  much computation goes into the sound. Its encoder (AuT) was trained from scratch on about 20 million
  hours of audio, per the Qwen3-Omni report.
- **MoE does touch the audio after encoding.** Audio vectors are tokens to the language model, so the
  router sends each audio token to its own experts, as it does for text. Some experts can therefore
  specialise in audio-derived tokens. This is a property of training, not something we measured.
- **MoE saves compute, not memory.** All 30B parameters must sit on the GPU; only the work per token
  shrinks. That is why our ceiling was set by GPU memory, not speed.
- **The ears are about the same size in every model we tested:**

  | Model | Language model | Audio encoder |
  |---|---|---|
  | Voxtral (Mini and Small) | 3B / 24B dense | Whisper-large-v3 based |
  | Qwen2.5-Omni-7B | 7B dense | Whisper-large-v3 initialised |
  | Qwen3-Omni-30B-A3B | 30B MoE, ~3B active | AuT, trained from scratch |
  | Step-Audio-R1.1 | 32B dense | Qwen2-Audio encoder, frozen during training |

  All are encoders of roughly 0.6B parameters. So the x-axis of the scaling plot measures the brain, not
  the ear. A model "hears better" mainly through how much and what kind of audio its encoder and projector
  were trained on.

**What our results say about MoE:** a model with ~3B active parameters (Qwen3-Omni) beat a 24B dense one
(Voxtral) by 11 points on MMAR. So the MoE design does not stand in the way of audio understanding;
training data and model generation matter more than active parameter count. On ICBHI the design made no
difference: both architectures ignore the lung sounds.

## Model scaling on MMAR (five models)

Three models were added on MMAR only, with the same items and 3 runs:
- **Voxtral-Mini-3B**, the small sibling of Voxtral-Small-24B;
- **Qwen2.5-Omni-7B**, the previous generation of Qwen3-Omni;
- **Step-Audio-R1.1**, the largest open audio model that fits on one 96 GB GPU. It is a 32B dense
  reasoning model (Qwen2.5-32B language model, Qwen2-Audio encoder), trained on about 8 million hours of
  audio. It runs only on StepFun's own vLLM build, so it was served over HTTP from a separate environment.

Results: Drive `speech-processing-res-mmar-v2-scaling/` (local copy in `data/`).

![MMAR accuracy against model size](figures/mmar_scaling.png)

*Point = baseline; bar = range over the 19 techniques every model ran; lines join models of one family.
Right panel: the 56 audio-dependent items (chance 33.2%).*

| Model | LLM | Baseline | Transcript-solvable (143) | Audio-dependent (56) | Prompting range (worst–best) |
|---|---|---|---|---|---|
| Voxtral-Mini-3B | 3B dense | 59.3 | 73.4 | **23.2** (below chance) | 54.8–65.3 |
| Qwen2.5-Omni-7B | 7B dense | 65.8 | 73.4 | 46.4 (p = 0.02 vs chance) | 52.8–69.3 |
| Voxtral-Small-24B | 24B dense | 65.8 | 75.5 | 41.1 (p = 0.13) | 64.3–73.4 |
| Qwen3-Omni-30B-A3B | 30B MoE, ~3B active | **76.9** | 87.4 | **50.0** (p = 0.006) | 71.9–79.9 |
| Step-Audio-R1.1 | 32B dense, reasoning | 73.9 | **88.1** | 37.5 (p = 0.29) | 66.8–79.9 |

*p vs chance: one-sided, simulated from each item's number of choices. The prompting range covers only the
19 techniques every model ran, so it differs slightly from the per-model reports.*

**Is larger better? Within a family, yes; across families, no.**

- **Within each family, the larger model is better:**
  - Voxtral 3B → 24B: +6.5 points (p = 0.09).
  - Qwen 7B → 30B-A3B: +11.1 points (p = 0.002).
- **Across families, size does not predict accuracy:**
  - Qwen2.5-Omni-7B ties Voxtral-Small-24B (65.8 each) with about a third of the parameters.
  - The largest model, Step-Audio-R1.1 (32B dense), does not beat Qwen3-Omni (~3B active per token):
    73.9 vs 76.9, 20 items won and 26 lost (p = 0.46).
  - Family and training matter as much as size.
- **The families improve in different places:**
  - Voxtral 3B → 24B gains almost only on the audio-dependent items: +17.9 points (14 won, 4 lost, p = 0.03). On the transcript-solvable items it moves only +2.1 (p = 0.72).
  - Qwen 7B → 30B-A3B gains on the transcript-solvable items: +14.0 points (31 won, 11 lost, p = 0.003). On the audio-dependent items it moves only +3.6 (p = 0.82).
  - Step-Audio's lead over the smaller models is likewise all on the words (next section).
- **Model choice moves accuracy more than prompting.** The spread between baselines (59.3–76.9) is larger than any model's prompting range, and no model has a technique that helps after correction.
- **Language check:** no answer from any of the three new models switched to Chinese. The few flagged Step-Audio answers are English reasoning that quotes Chinese heard in the audio.

### Step-Audio-R1.1: the largest model reasons best over the words, not over the sound

- **Best on the transcript-solvable items (88.1%), close to chance on the audio-dependent items (37.5%,
  p = 0.29 vs chance).** Compared on the same items:

  | Step-Audio vs | All items (won/lost, p) | Transcript-solvable | Audio-dependent |
  |---|---|---|---|
  | Voxtral-Small-24B | +8.0 (35/19, p = 0.04) | +12.6 (26/8, p = 0.003) | −3.6 (9/11, p = 0.82) |
  | Qwen2.5-Omni-7B | +8.0 (36/20, p = 0.04) | +14.7 (29/8, p < 0.001) | −8.9 (7/12, p = 0.36) |
  | Qwen3-Omni-30B-A3B | −3.0 (20/26, p = 0.46) | +0.7 (12/11, p = 1.0) | −12.5 (8/15, p = 0.21) |

  Every significant difference is on the transcript-solvable items. 8 million hours of audio training and
  a 32B language model did not make it hear better than a 7B model. Its encoder (from Qwen2-Audio)
  was kept frozen during training, which fits this pattern: the language model got stronger, the ear did not.
- **By category:** best of all models on *Content Analysis* (84.0%, n = 100) and level with Qwen3-Omni on
  *Emotion and Intention* (78.1%). Behind Qwen3-Omni on *Speaker Analysis* (70.0 vs 86.7) and the
  *Perception* layer (41.2 vs 64.7). *Counting* stays near chance (33.3%), as for every model.
- **Prompting:** no technique survives correction (19 techniques, Holm).
  - Best: shuffled-option voting (v27) +6.0 (16 won, 4 lost, p = 0.012, Holm 0.23). It asks every question
    five times with the options reordered, so it also works as five samples of the reasoning.
  - The few-shot arms add +10.7 and +12.5 on the audio-dependent items, not significant on 56 items.
  - Worst: multi-turn decomposition −7.0 (p = 0.04, Holm 0.69).
- **Chain-of-thought does not hurt it** (+1.5), unlike Qwen2.5-Omni (−13.1). It is a reasoning model and
  thinks before every answer anyway.
- **Text evidence does no harm here.** Diarized transcripts (v18) and acoustic features (v19) each change
  the audio-dependent score by +1.8, where they cost Voxtral-Small and Qwen2.5 up to 10.7 points.
- **Output health:** a baseline answer averages ~3,200 characters of reasoning. 1.5–5.4% of answers per
  arm are unparsed (3.2% in the baseline); almost all of them hit the 4,096-token limit while still
  thinking. They count as wrong, so Step-Audio's scores may be up to ~3 points low.
- **Not run:**
  - the contrastive arms (v24–v26), which need the model's letter probabilities and are not available
    through the server;
  - chunked audio (v22), which crashed StepFun's server.

  These arms are excluded from every model's prompting range. In two-pass localization (v23), 93% of
  first turns gave a usable time span after the reasoning; the rest ran out of tokens while thinking.

## Where the audio front-end gets better, and where it doesn't

**The audio encoders are the same shape in all five models.** We checked each model's `config.json` on
Hugging Face. Every audio encoder is a Whisper-large-v3-shaped transformer:
- 32 layers, width 1280, 20 attention heads;
- 128 mel bins;
- 1500 input positions (30 s per window).

That is about 0.6B parameters in every model.

**The differences are in how each encoder was trained:**

| Model | Audio encoder | Language model it feeds |
|---|---|---|
| Voxtral-Mini-3B | Voxtral encoder, Whisper-large-v3 shape | 3B dense (30 layers, width 3072) |
| Voxtral-Small-24B | same architecture as Mini | 24B dense (40 layers, width 5120) |
| Qwen2.5-Omni-7B | initialised from Whisper-large-v3 (per the Qwen2.5-Omni report) | 7B dense (28 layers) |
| Qwen3-Omni-30B-A3B | AuT, trained from scratch on ~20M hours (per the Qwen3-Omni report); its config adds an extra downsampling stage (`downsample_hidden_size` 480) and halves the attention window (`n_window` 50 vs 100) | 30B MoE: 48 layers, 128 experts, 8 active per token |

So the x-axis of the scaling plot measures the language model; the "ear" stays the same size. What changes
between models is how that ear was trained and how much the language model makes of what it hears. The
probes below separate the two where the data allows.

1. **Understanding the words is not the bottleneck.** Every model, the 3B included, answers 73–87% of
   the transcript-solvable items.
   - Giving the model a Whisper transcript on top of the audio (v3) adds at most +4.9 points on those
     items, and only +0.7 for Qwen3-Omni.
   - So each model's own audio front-end already gets the words. Qwen3-Omni's +14 points there over
     Qwen2.5 is better *reasoning* about the words, not better hearing of them.
2. **Hearing beyond the words grows with the language model when the ear is held fixed.** The two Voxtral
   models have the same encoder architecture:
   - Voxtral-Mini scores 23.2% on the audio-dependent items, *below* chance. When the words don't contain
     the answer, it is pulled toward the distractor that matches the words.
   - Voxtral-Small scores 41.1%.
   - The bigger language model makes much better use of the same kind of acoustic input, which fits the
     view that interpreting tone, speakers and events happens partly in the language model, not only in
     the encoder.
3. **A better-trained ear shows up as hearing at small scale.** Qwen2.5-Omni-7B (46.4%) beats
   Voxtral-Small-24B (41.1%) on the audio-dependent items with a third of the parameters. Qwen3-Omni, with
   its new AuT encoder, reaches 50.0%. Step-Audio-R1.1, the largest language model with a frozen
   Whisper-based encoder, reaches only 37.5%. These differences are not significant on 56 items, but the
   order follows encoder training, not language-model size.
4. **By MMAR layer** (small n, read as direction only):
   - *Perception* (n = 17, chance 25%): Qwen2.5-Omni 29.4 → Qwen3-Omni 64.7, the largest jump of any
     category. Voxtral 47.1 → 52.9.
   - *Speaker analysis* (n = 30): Qwen3-Omni 86.7 against 63–70 for the others.
   - *Counting and statistics* (n = 9): every model at or near chance (11–44%).
   - *Emotion and intention* (n = 32): Voxtral-Mini (71.9) beats Voxtral-Small (56.2). Not every acoustic
     skill improves with size.
5. **External acoustic evidence helps only the weakest ear.** Adding measured acoustic features (v19)
   changes the audio-dependent score by:
   - Voxtral-Mini: **+5.4**
   - Voxtral-Small: −8.9
   - Qwen2.5-Omni: −8.9
   - Qwen3-Omni: −1.8

   Diarized transcripts (v18) hurt every model on these items (−3.6 to −10.7). Text descriptions of the
   audio replace hearing only where the model's own hearing is poor; elsewhere they pull the answer toward
   the text.
6. **Chain-of-thought hurts the small Qwen.** Qwen2.5-Omni drops from 65.8 to 52.8 with CoT. That is not a
   formatting problem: only 1.3% of its CoT answers are unparsed. At 7B, reasoning out loud about audio
   leads it astray.

**Summary of the audio analysis:** all five models use the same size of ear. Gains from size come from the
language model, and they land in different places per family:
- Voxtral's bigger model learns to use non-verbal audio.
- Qwen's newer, larger model reasons better over the words.

A better-trained encoder (Qwen) gives above-chance hearing even at 7B. No model is reliable on counting.

## Conclusions

1. **Model choice matters more than prompting.** Across 25+ techniques per dataset, no prompting
   technique produced a gain that survives correction, for any of the five models. Changing the model
   moved the MMAR baseline from 59.3 to 76.9.
   - Within a family, the larger model is better: +6.5 for Voxtral, +11.1 for Qwen.
   - Across families, size does not predict accuracy: a 7B Qwen ties the 24B Voxtral, and the largest
     model (Step-Audio-R1.1, 32B dense) does not beat Qwen3-Omni (~3B active). All five models share the
     same encoder size, so the gains come from the language model and from how the encoder was trained.
   - Size buys reasoning over the words, not hearing. Step-Audio is the best model on the
     transcript-solvable items (88.1%) and close to chance on the audio-dependent ones (37.5%).
2. **Prompting gains are model-specific.** What helped the weaker model (few-shot, transcripts, option
   voting) fixed its format and position biases; the stronger model has no such gap to close.
3. **The lane result holds for both models.** In-Knowledge (English speech): both models perform well,
   mainly from the words, and Qwen shows the first sign of using non-verbal audio. Out-of-Knowledge (lung
   sounds): both are at or below the score of a constant answer, and both ignore the audio. Their
   failures differ: Voxtral scatters into rare classes, Qwen defaults to "no disease".
4. **A constant answer can look like a win.** Balanced accuracy gives a one-answer model exactly
   1/(number of classes). Comparisons between failing models must report the answer distribution
   alongside the score.
5. **Hallucination was not measured directly.** This ICBHI packaging has no per-cycle crackle/wheeze
   labels, so a judge could only score plausibility. Indirect evidence exists: Qwen gives the same
   diagnosis with silence as with sound, and on one Mandarin MMAR clip the two models report different
   Chinese words for the same speech, so at least one of them invents content.

## Limitations and future work

- **Model size was limited by the hardware.** One 96 GB GPU holds models up to about 30B parameters in
  BF16. Qwen3-Omni activates only ~3B parameters per token. The models tested may simply be too small
  for zero-shot medical audio, and the In-Knowledge gap between them suggests scale and training matter
  more than prompting.
- **Recommended next step:** repeat the core arms (baseline, silence controls, CoT, few-shot, retrieval)
  with larger models:
  - larger open audio-language models on multi-GPU or FP8 setups;
  - closed models with native audio input, such as Gemini.
- **Hallucination rate:** run CoT on silence (a "c5" control) and count invented findings; use the
  original ICBHI 2017 database, which has per-cycle crackle/wheeze labels, to score acoustic descriptions
  against ground truth.
- **Step-Audio-R1.1 is not fully comparable:** it ran on StepFun's own server, so the contrastive arms
  (v24–v26) could not run, chunked audio (v22) crashed the server, and 1.5–5.4% of its answers per arm
  ran out of tokens while thinking. The scaling plot uses only the 19 techniques every model ran.
- **Sample size:** n = 199 / 168 gives about ±6.5 points of uncertainty on accuracy; the MMAR
  audio-dependent subset has only 56 items, and five models (two on ICBHI) are a small basis for general claims.
