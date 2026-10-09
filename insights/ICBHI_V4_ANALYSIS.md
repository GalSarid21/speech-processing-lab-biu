# ICBHI V4 — Results and Conclusions (Out-of-Knowledge lane)

**Model:** Voxtral-Small-24B-2507 (vLLM), zero-shot, on one NVIDIA RTX PRO 6000 Blackwell (96 GB). **Data:** ICBHI 2017 packaging, 168 evaluation
recordings, plus 6 held-out reference recordings (one per primary class) that are never scored.
**Runs:** 3 per experiment; contrastive/letter-scoring arms (a5\*) are deterministic, so 1 run.
**Source:** Drive `speech-processing-res-icbhi-v4/` (local copy: `data/speech-processing-res-icbhi-v4/`),
29 run folders, 2026-10-06.

ICBHI is the *Out-of-Knowledge* lane: lung auscultation is a domain the model never saw in
pre-training, so these are authentic zero-shot measurements. A model that cannot do the task is a
result, not a failure of the experiment.

## Verification

- `comparison.md` was recomputed from the raw outputs (`raw_output*.jsonl`) with
  `scripts/analyze_runs.py`; all 28 rows are identical to the copy on Drive.
- An independent recomputation (own parsing, own majority vote) agrees to ±0.1 BA.
- **Parser gap found:** an answer of the form `D. LRTI (Lower Respiratory Tract Infection)` followed
  by an explanation is left unparsed, because the parenthetical stops the line from matching the
  option text. Re-scoring with that form accepted changes one arm (a2a: 13.9 → 14.9 BA) and no
  conclusion; the hidden answers were almost all LRTI, which BA does not count. Remaining unparsed
  answers are real: c2 (19%) refuses to choose from metadata alone, and a4b (23%) is cut off by its
  64-token budget mid-sentence (`"...the patient's diagnosis is"`).

## How to read the numbers

- **BA (balanced accuracy)** = recall per class, averaged over the 6 primary classes (COPD,
  Pneumonia, URTI, Bronchiectasis, Bronchiolitis, healthy). Asthma (1 item) and LRTI (2 items) are
  still offered as answers but are not averaged.
- **16.7%** is what *any* recording-blind strategy that answers only primary classes scores
  (e.g. always "COPD": 35.1% accuracy, 16.7% BA). A blind strategy that also answers Asthma/LRTI
  scores *below* 16.7%, in proportion to how often it does so. BA below 16.7% therefore does not
  mean "worse than guessing"; it means answers were spent on the two near-empty options.
- **Metadata ceiling 33.3%**: the text rule "multichannel → COPD, else → healthy" reaches it without
  any audio, because acquisition device is confounded with diagnosis in this packaging.
- **Label-shuffle test** (added in this re-evaluation): the true labels are shuffled across items
  10,000 times. If BA does not beat the shuffled BA, the predictions carry no information about the
  diagnosis, whatever their level. A second version shuffles only *within* acquisition mode,
  asking whether there is information beyond the recording device.

## Results

Baseline **r0** ("listen and choose the diagnosis"): **BA 7.4%, accuracy 10.1%**. Answers: LRTI 46%,
COPD 34%, Pneumonia 21%.

| exp | technique | BA | Δ vs r0 | p vs r0 | carries label info?¹ | answers mostly |
|---|---|---|---|---|---|---|
| a6a | diagnoses of the 5 most similar reference recordings, as text | **23.7** | **+16.3** | <0.001 | **yes** (p=0.001; within device p=0.02) | COPD 46%, Pneumonia 38% |
| a5a | most-probable answer letter (log-prob) | 17.6 | +10.2 | <0.001 | no (p=0.31) | COPD 75%, Pneumonia 25% |
| a5d | contrastive scoring + features | 17.6 | +10.1 | 0.006 | no (p=0.09) | healthy 40%, URTI 18% |
| a6b | similar reference recordings played as examples | 16.9 | +9.4 | 0.004 | no (p=0.17) | COPD 58% |
| r2 | chain-of-thought | 16.7 | +9.2 | 0.007 | no (p=1.0) | healthy 100% |
| a5b | contrastive vs silence | 15.9 | +8.4 | 0.070 | no | Bronchiolitis 36% |
| a5c | contrastive vs silence + noise | 15.1 | +7.6 | 0.065 | no | Asthma 30% |
| a2a | audio-tagger labels as text | 13.9 | +6.4 | 0.023 | weak (p=0.018, n.s. corrected) | COPD 33% |
| a3b | band-pass + normalise + pitch shift | 13.3 | +5.9 | 0.011 | no (p=0.93) | COPD 79% |
| c2 | *control:* text-only model, metadata only | 13.2 | +5.8 | 0.070 | no | healthy 80% |
| a1c | *control:* text-only model reads the features | 13.1 | +5.6 | 0.152 | weak (p=0.04, n.s. corrected) | Asthma 29% |
| r3 | acoustic dictionary | 7.9 | +0.5 | 0.852 | no | Asthma 48%, COPD 44% |
| a7a | 5 shuffled option orders, majority vote | 5.7 | −1.7 | 0.472 | no | LRTI 45% |
| r1 | pulmonologist role | 5.2 | −2.3 | 0.275 | no | LRTI 62% |
| a3a | band-pass + normalise | 4.4 | −3.1 | 0.097 | no | LRTI 45%, COPD 44% |
| a2b | tagger + features | 4.2 | −3.3 | 0.142 | no | Asthma 68% |
| r5 | three-turn decomposition | 3.9 | −3.6 | 0.164 | no | Asthma 66% |
| c3 | *control:* silence + metadata | 3.3 | −4.2 | 0.086 | no | LRTI 66% |
| r0_aware | baseline + metadata | 2.8 | −4.7 | 0.053 | no | LRTI 46% |
| c1 | *control:* silence, baseline prompt | 0.7 | −6.7 | 0.001 | no | LRTI 74% |
| a4b | locate worst cycle, then decide | 0.7 | −6.7 | 0.001 | no | 54% unparsed |
| c4 | *control:* silence + features + dictionary | 0.6 | −6.9 | <0.001 | no | Asthma 90% |
| a1d | features + dictionary + metadata | 0.3 | −7.2 | <0.001 | no | Asthma 92% |
| a3c | band-pass + features | 0.3 | −7.2 | <0.001 | no | Asthma 91% |
| a1a | measured features | 0.0 | −7.5 | <0.001 | no | Asthma 94% |
| a1b | features + dictionary | 0.0 | −7.5 | <0.001 | no | Asthma 91% |
| a4a | labelled breathing cycles | 0.0 | −7.5 | <0.001 | no | Asthma 90% |
| r4 | dictionary + chain-of-thought | 0.0 | −7.5 | <0.001 | no | Asthma 99% |

¹ Label-shuffle test, one-sided. Only a6a survives Holm correction across all 29 arms (p=0.029).
Not run: r6/r7 (one reference per class + test = 7 clips; Voxtral accepts 5 per prompt).

**Pre-registered comparisons**

| comparison | result | p |
|---|---|---|
| a1b (features) vs r0 | worse, 0.0 vs 7.4 | <0.001 |
| best of a3a/a3b vs r0 | a3b better, 13.3 vs 7.4 | 0.011 (0.022 for best-of-two) |
| a5c (content-free calibration) vs a5a | no difference, 15.1 vs 17.6 | 0.57 |

## Conclusions

1. **Voxtral cannot diagnose respiratory disease from auscultation zero-shot.** In 28 of 29 arms the
   predictions carry no detectable information about the true diagnosis (label-shuffle test,
   Holm-corrected). Every
   "improvement" over r0 except a6a moves answers away from the near-empty Asthma/LRTI options
   toward common classes; none of them sorts patients by disease.

2. **The model hears the recordings, but not the disease.** Real audio changes the answers relative
   to silence (r0 vs c1, p=0.001), and the change follows the recording device: on multichannel
   recordings (74% COPD) r0 answers LRTI 62% of the time, on single-channel recordings (no COPD at
   all) it answers COPD 38%. The acoustic difference is perceived and mapped to the wrong labels.

3. **Text cues dominate audio.** Prompts that add measured features, labelled cycles, or the
   dictionary with CoT make the model answer Asthma 90–99% of the time — a class with 1 recording;
   the dictionary alone already pushes it to 48%. Features
   with real audio (a1b) and features with silence (c4) are indistinguishable (p=0.51): the audio
   adds nothing once text evidence is present. A text-only model reading the same features (a1c)
   spreads its answers and scores 13.1, so the collapse is Voxtral's reading of the word, not the
   measurements.

4. **Prompt engineering does not open the lane.** Role, CoT, dictionary, decomposition, choice
   shuffling and cycle presentation all stay at or below the blind level. CoT's significant +9.2
   is answering "healthy" for every recording.

5. **The model cannot even exploit trivially predictive metadata.** Given the acquisition mode, a
   one-line rule reaches 33.3% BA; r0_aware (2.8), c3 (3.3) and the text-only c2 (13.2) do not
   approach it. The device–diagnosis link is itself out of the model's knowledge.

6. **The only signal comes from labelled examples, which is no longer zero-shot.** a6a (23.7% BA,
   +16.3) is the single arm with real label information, and some of it survives within
   acquisition mode (p=0.02), so it is not device matching alone. It works by giving the model the
   diagnoses of acoustically similar reference recordings — external labelled knowledge supplied at
   inference time. Caveats: it rests on 6 references, patient ids are not published so a reference
   may share a patient with test items, and it stays below the 33.3% metadata ceiling.

**One-line summary:** in the Out-of-Knowledge lane, zero-shot prompting, audio preprocessing,
injected measurements and calibrated scoring all fail to produce diagnostic information; the model's
answers are driven by text cues and option priors, and only retrieval of labelled examples yields a
small, real signal.

## Limitations

- n=168, with 12–59 items per primary class; the BA 95% CI is about ±4–6 points.
- No patient ids in this packaging, so recordings of one patient may appear on both sides of the
  reference/evaluation split (affects a6a/a6b only).
- 29 arms were compared against one baseline. Uncorrected p<0.05 results other than a6a, a5a and the
  arms significantly *worse* than r0 do not survive Holm correction.
- The a4b result is partly a budget artefact (64 tokens), and r6/r7 could not be run on this model.

## Second model

Qwen3-Omni-30B-A3B reproduces the result: it answers "No potential disease detected" to nearly every recording (exactly 16.7 BA), gives identical answers with silence and with sound, and its only arm with signal is the same retrieval arm (knn_text_evidence, 23.2 BA). Details: `CROSS_MODEL_ANALYSIS.md`.
