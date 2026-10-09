# 1. Methodology

What we tested, on which models and data, and how each result is scored. Results are in
`02_MODEL_PROFILES.md` and the conclusions in `03_CONCLUSIONS.md`.

**Research question:** how well do open audio-language models *understand* audio, as opposed to the words
spoken in it, and can prompting change that? We compare two settings:

- **In-Knowledge lane (MMAR):** everyday English speech, the kind of audio these models were trained on.
- **Out-of-Knowledge lane (ICBHI):** lung-sound recordings, a domain none of them saw in training, so
  every result there is an authentic zero-shot measurement.

---

## 1.1 Background: how a language model hears

### Two ways to connect audio to a language model

**Whisper has two halves, and the two designs use different halves.** Whisper is a speech recogniser
built as an encoder-decoder:
- its **encoder** turns the sound (a spectrogram) into a sequence of vectors, one per 20 ms. These vectors
  are not words: they are a numeric description of the sound, and can still carry pitch, voice quality,
  loudness and timing;
- its **decoder** reads those vectors and writes the transcript, word by word.

1. **Cascade (speech recogniser → text model).** The full Whisper, encoder *and* decoder, produces a
   transcript, and a text-only language model reads it. Strong on *what was said*. Everything that is not
   a word is thrown away at the decoder: tone of voice, who is speaking, emotion, laughter, background
   sounds, music, timing.
2. **End-to-end audio-language model.** The decoder is dropped. Three parts are joined and trained
   together:
   - an **audio encoder** (in all five models here, a network the shape of Whisper-large-v3's encoder:
     128 mel bins, 30-second windows, 25–50 vectors per second);
   - an **adapter** (projector) that compresses those vectors (to 12.5–25 per second here) and maps them
     into the language model's input space, where they act as "audio tokens" next to the text tokens;
   - a **language model** that reads the audio tokens and the text prompt together and writes the answer.

   The language model never sees a transcript. It sees the encoder's vectors, so it *can* use non-verbal
   information. Whether it *does* depends on what the encoder and adapter were trained to keep.

**All five models are end-to-end; every one has a real audio encoder.** "Whisper-large-v3 based" in the
table below means the model took Whisper's *encoder* as its starting point and trained it further inside
the audio-language model. It does not mean the model transcribes first. The models differ in where their
encoder came from and how much it was trained afterwards:

| Model | Encoder starting point | Then |
|---|---|---|
| Voxtral-Mini-3B, Voxtral-Small-24B | Whisper-large-v3 encoder | frozen during a short warm-up that trains only the adapter; the report does not say whether it is trained later |
| Qwen2.5-Omni-7B | Whisper-large-v3 encoder | trained further on Qwen's audio data |
| Qwen3-Omni-30B-A3B | none: a new encoder (AuT) trained from scratch on 20M hours | trained further with the language model |
| Step-Audio-R1.1 | the Qwen2-Audio encoder | **kept frozen**; only the adapter and language model were trained |

A Whisper encoder was originally trained only to help produce transcripts, so it may have learned to
ignore what does not help transcription (background sounds, voice quality). Training it further on other
audio tasks, or training a new encoder on broader audio as Qwen3-Omni did, is how a model can learn to
keep that information.

**How this differs from our transcript experiments.** In v3 the model gets the audio tokens *plus* a
Whisper transcript as text, i.e. both designs at once. In v13 a text-only model gets only the transcript:
a pure cascade. Comparing the baseline (audio tokens only), v3 (both) and v13 (words only) shows whether
the model's own ear already gets the words, and whether written words help or crowd out the sound.

> **Figure note for the report — infographic F1 (architectures).** Prompt for an image model:
> "Clean, flat technical infographic, white background, two horizontal pipelines stacked vertically.
> Top, titled 'Cascade': waveform icon → box 'Whisper encoder' → box 'Whisper decoder' → speech bubble
> with text 'I'm fine.' → box 'Text LLM' → answer card 'B'. Under the decoder, a red label 'tone, speaker,
> emotion, background sounds lost here'. Bottom, titled 'End-to-end audio-LLM': waveform icon → box
> 'Audio encoder' → small box 'Adapter' → a row of colored squares labelled 'audio tokens' merging with
> grey squares labelled 'text prompt' → large box 'Language model' → answer card 'B'. Under the audio
> tokens, a green label 'pitch, voice, timing can still pass'. Minimal colors (blue for audio, grey for
> text), sans-serif labels, no decorative elements."
>
> **Figure note — infographic F1b (our five models).** Prompt: "Five horizontal bars, one per model
> (Voxtral-Mini-3B, Qwen2.5-Omni-7B, Voxtral-Small-24B, Qwen3-Omni-30B-A3B, Step-Audio-R1.1). Each bar
> has a small fixed-size block 'audio encoder ~0.6B' on the left and a language-model block whose width is
> proportional to its size (3, 7, 24, 30, 32 B). Qwen3-Omni's language-model block is drawn as 128 small
> cells with 8 highlighted, labelled 'MoE: ~3B active per token'. Encoder blocks are annotated: 'from
> Whisper' (Voxtral, Qwen2.5), 'new, 20M h' (Qwen3), 'frozen' with a lock icon (Step-Audio). Caption: 'Same
> size of ear, very different brains and ear training.'"

Our experiments mix the two designs on purpose: several arms give an end-to-end model a cascade-style
transcript or text measurements as well (§1.4), to see whether text helps the audio or replaces it.

### What "mixture of experts" changes

A mixture-of-experts (MoE) language model has many feed-forward "experts" per layer and routes each token
to a few of them. Qwen3-Omni-30B-A3B holds 30B parameters but uses about 3B per token. MoE is a property
of the **language model only**: the audio encoder in front of it is an ordinary dense network, and every
audio frame passes through all of it. So "30B" in an MoE model and "32B" in a dense model are not the same
amount of computation per token, but the ear is the same kind of network.

### The models

| | Voxtral-Mini-3B | Qwen2.5-Omni-7B | Voxtral-Small-24B | Qwen3-Omni-30B-A3B | Step-Audio-R1.1 |
|---|---|---|---|---|---|
| Released | Jul 2025 | Mar 2025 | Jul 2025 | Sep 2025 | Jan 2026 (R1: Nov 2025) |
| Language model | Ministral 3B | Qwen2.5-7B | Mistral Small 3.1 24B ¹ | Qwen3-30B-A3B | Qwen2.5-32B |
| Dense / MoE | dense | dense | dense | **MoE**: 128 experts, 8 active, ~3B active per token | dense |
| Audio encoder (sound → vectors, never text) | started from Whisper-large-v3's encoder, 640M | started from Whisper-large-v3's encoder | same as Mini, 640M | **AuT, new, trained from scratch**, ~0.6B | Qwen2-Audio's encoder |
| Encoder during training | frozen only in a short warm-up (adapter only); later stages not reported | adapter first, then encoder, with the LLM frozen; then everything trained | as Mini | adapter first, then encoder, with the LLM frozen; then everything trained | **frozen throughout** |
| Audio tokens per second | 12.5 | ~25 | 12.5 | 12.5 | 12.5 |
| Audio training data | hours not reported | 300B audio tokens in pre-training; hours not reported | hours not reported | encoder: **20M hours** of supervised audio; LLM pre-training: 0.77T audio tokens | **8M hours** (inherited from Step-Audio 2) + 4B audio tokens of reasoning data |
| Post-training | SFT, then DPO / online DPO | instruction SFT | as Mini | SFT, distillation, RL (GSPO) | reasoning SFT + RL (PPO, self-distillation) |
| Trained to reason | no | no | no | no (a separate "Thinking" variant exists; we used Instruct) | **yes**, always thinks before answering |
| MMAR, full benchmark (published) | not reported | 56.7 (MMAR paper) | not reported | not reported ² | not reported |

¹ The Voxtral paper says Mistral Small 3.1; the Hugging Face card says Mistral Small 3.
² The MMAR leaderboard lists the *Thinking* variant at 66.4; no score is published for the Instruct
model we used. Published MMAR scores cover all 1,000 items and all audio types, so they are not
comparable with our 199 English-speech items.

**Sources:** Voxtral, arXiv 2507.13264 (§2.2, §3, Table 1); Qwen2.5-Omni, arXiv 2503.20215
(architecture, pre-training stages); Qwen3-Omni, arXiv 2509.17765 (AuT, pre-training S1–S3,
post-training); Step-Audio-R1, arXiv 2511.15848, and Step-Audio 2, arXiv 2507.16632 (architecture,
training); MMAR, arXiv 2505.13032 (Table 2); each model's Hugging Face card and `config.json`. Step-Audio
R1.1 has no paper of its own: its card describes a "dual-brain" design for reasoning while speaking, and
its training data is not reported.

**What the table says about hearing.** The models differ far more in *how the ear was trained* than in
its size:
- Qwen3-Omni built a new encoder from scratch on 20M hours and then trained it further with the language
  model.
- Step-Audio-R1.1 had the most audio overall (8M hours), but its encoder stayed **frozen**: all that
  training went into the adapter and the language model.
- Voxtral and Qwen2.5-Omni start from Whisper-large-v3, a speech recogniser, and adapt it.

All five encoders have the same shape (verified from each `config.json`): a Whisper-large-v3-sized
transformer of 32 layers, width 1280, 20 heads, about 0.6B parameters. **So model size in this study is
the size of the language model, not of the ear.**

**Where each model ran.** One NVIDIA RTX PRO 6000 Blackwell (96 GB), BF16 weights, which caps the models
at about 32B parameters. Voxtral, Qwen2.5-Omni and Qwen3-Omni ran in-process with vLLM 0.30.
Step-Audio-R1.1 only runs on StepFun's own vLLM build, so it was served over an OpenAI-compatible HTTP
API from a separate environment and received the identical conversations (§1.6).

---

## 1.2 Datasets

### MMAR (In-Knowledge lane)

MMAR (arXiv 2505.13032) has 1,000 multiple-choice questions about real-world audio from internet videos:
speech, music, sounds and mixtures. Each item belongs to one of four layers, with finer sub-categories.

**Our subset:** 200 English items with the "speech" modality label, minus one item whose gold answer
names both options: **n = 199.** Few-shot and retrieval examples come from a separate pool of
English-speech MMAR items; the code checks that no example is also an evaluation item.

**Item distribution** (TS / AD: see the split below):

| Layer | Tests | Items | TS | AD |
|---|---|---|---|---|
| Signal | raw acoustic properties: pitch, loudness, duration, rhythm | 2 | 0 | 2 |
| Perception | kind of sound; the speaker's voice (age, gender, emotion in the voice) | 17 | 9 | 8 |
| Semantic | the meaning and intent of what is said | 162 | 119 | 43 |
| Cultural | social and world knowledge grounded in the audio | 18 | 15 | 3 |

| Largest sub-categories | Items | TS | AD |
|---|---|---|---|
| Content Analysis (what is said) | 100 | 73 | 27 |
| Emotion and Intention | 32 | 24 | 8 |
| Speaker Analysis | 30 | 22 | 8 |
| Culture of Speaker | 14 | 11 | 3 |
| Counting and Statistics | 9 | 4 | 5 |
| 8 others | 14 | 9 | 5 |

So this subset leans heavily towards understanding *what* is said (81% Semantic).

**Answers:**

| | A | B | C | D |
|---|---|---|---|---|
| Correct answer is… | 73 (37%) | 58 (29%) | 30 (15%) | 38 (19%) |

142 items have 4 options, 55 have 2, and 2 have 3. "A" is the most common correct answer, mostly because
two-option items are more often A (32 of 55).

**Weighting:** none. Plain accuracy, every item counts once.
- **Different numbers of options** are handled through chance: 1 / number of options per item, averaged
  to 32.0% overall.
- **The uneven correct letters** could reward a model that prefers "A". Two checks cover this:
  - "top answer %" (§1.5) flags a model that leans on one answer;
  - v27 asks each question with the options in 5 different orders.

### ICBHI 2017 (Out-of-Knowledge lane)

Respiratory-sound recordings labelled with the patient's diagnosis. The task: listen to one recording and
choose the diagnosis from 8 options. The Hugging Face packaging has 174 recordings in a single split.
One example per main class (6 recordings) is held out to serve as few-shot examples and retrieval
candidates and is never scored. **n = 168** evaluation recordings.

| Diagnosis | Items | Share | In the score (BA)? |
|---|---|---|---|
| COPD | 59 | 35% | yes |
| Healthy ("no potential disease detected") | 34 | 20% | yes |
| Pneumonia | 23 | 14% | yes |
| URTI (upper respiratory tract infection) | 22 | 13% | yes |
| Bronchiectasis | 15 | 9% | yes |
| Bronchiolitis | 12 | 7% | yes |
| LRTI (lower respiratory tract infection) | 2 | 1% | offered as an answer, not averaged |
| Asthma | 1 | 1% | offered as an answer, not averaged |

**Balancing: we balance the metric, not the sample.**
- *The imbalance:* always answering "COPD" gets 35% plain accuracy.
- *What we did:* the main metric is **balanced accuracy**, the recall of each of the 6 main classes,
  averaged. Every class weighs the same, so "always COPD" scores exactly 16.7%, the same as any strategy
  that ignores the recording.
- *Why not subsample to equal classes:* that would mean 12 items per class. It gives the same guarantee
  but throws away 96 recordings, so every score would rest on far fewer items.
- *Why LRTI and Asthma are left out of the average:* with 2 and 1 items, a single prediction would move
  the average by 12.5 points.

**Known problems with this packaging:**
- **Device confound.** The recording device is linked to the diagnosis. The text rule "multichannel
  device → COPD, else → healthy" scores 33.3% balanced accuracy without hearing anything.
- **Patient identity is lost.** The original file names are replaced, so two recordings may come from
  the same patient. A held-out example and a scored recording could therefore share a patient.

### Splitting MMAR into "words" and "sound" items

A text-only language model (Gemma-4-26B-A4B) answered every MMAR question from a Whisper-large-v3 transcript
alone, with no audio (experiment v13, 3 runs). Its majority-vote answers split the items into:

- **Transcript-solvable (TS), 143 items:** the words are enough. Chance 31.5%.
- **Audio-dependent (AD), 56 items:** the words were not enough. Chance 33.2%.

The split is stable: 133 of the 143 TS items were right in all 3 text-only runs, and 49 of the 56 AD items
were wrong in all 3.

**Does AD measure hearing? A check with v29.** v29 gives a model the same transcript as v3 but replaces the
recording with silence. If the AD items depend on the sound, removing it should hurt there and not on the
TS items. For the model that hears best (Qwen3-Omni), that is what happens: AD drops from 48.2% to 30.4%,
about chance (3 items won, 13 lost, p = 0.02), while TS barely moves (88.1 → 83.9, not significant).

This split is the basis for measuring **two capabilities separately**:

| Capability | Measured by |
|---|---|
| Reading the words | Accuracy on the TS items, compared with the text-only model (71.9% overall) |
| Hearing the sound | Accuracy on the AD items, compared with chance |
| Combining both | Overall accuracy, and whether added text moves TS and AD in the same direction |

### How much can 56 items tell us? (limits of the split)

The AD subset is small, and it is the only place MMAR measures hearing. This limits what the study can
claim, and we treat it as a central caveat:

1. **Detectable differences.** Paired tests (§1.5) help, but not enough to rescue small effects. If 20 of
   the 56 items change between two conditions, the split must be at least 15 vs 5 to be significant: a
   net change of 10 items, **about 18 points**. Smaller differences in hearing may be real but cannot be
   shown here.
2. **Not all AD items are about sound.** AD means "the text model failed", not "the answer is in the
   sound":
   - 27 of the 56 are *Content Analysis* items, about what is said. The text model may have failed on
     them because of a transcription error or a hard question, not because the answer is in the voice.
   - Only about 20 AD items are in categories about the voice or the sound itself: Speaker Analysis 8,
     Emotion and Intention 8, and single items on spatial position, anomalies and acoustic quality.

**What follows for the claims:**
- We state hearing results as **directions**, and call a difference significant only when the paired
  test supports it. A few hearing differences are large enough to pass: Voxtral 3B → 24B gains +17.9
  points on AD (p = 0.03).
- Agreement across models and experiments counts as evidence, not single numbers.
- **ICBHI is the pure hearing test.** Lung sounds contain no words, so every one of its 168 items depends
  on the audio. The two lanes complement each other: MMAR measures hearing on a small subset of familiar
  audio, ICBHI on a full set of unfamiliar audio.
- A larger acoustic test is future work: for example, MMAR's non-speech items (music, sounds), or a
  dataset built around paralinguistic labels.

## 1.3 Protocol (identical for every model)

- **Zero-shot, no fine-tuning.** Same prompts, same items, same order of options for every model.
- **3 runs per experiment** with sampling; the item's answer is the **majority** over the runs.
  Deterministic arms (contrastive scoring) run once.
- **Answer parsing:** a deterministic parser reads the chosen letter, or the option text, from the reply.
  Reasoning blocks (`<think>…</think>`) are removed first. A reply with no clear answer, including one cut
  off while still reasoning, counts as **wrong** and is reported as "unparsed".
- **Language check:** every reply is checked for a switch out of English (relevant for the Qwen and Step
  models, which are trained heavily on Chinese). Quoting Chinese words heard in the audio is not counted.
- **Judge model (secondary only):** on MMAR, a separate LLM (Qwen3.8-27B-FP8) also graded the answers of
  Voxtral-Small and Qwen3-Omni. All reported numbers use the parser; the judge was skipped for the three
  scaling models.

---

## 1.4 The experiments

Each experiment changes **one thing** relative to the baseline. "Targets" says which capability the
change is meant to affect, which tells us where to look for its effect (§1.5).

### MMAR

**Baseline and reference**

| ID | Name | What changes | Question it answers | Targets |
|---|---|---|---|---|
| v1 | baseline | "Listen to the audio and answer the multiple-choice question. Respond with only the letter." | Reference point for every comparison | — |
| v13 | text_only_llm | No audio. A text-only model (Gemma-4-26B-A4B) reads the Whisper transcript | How far do the words alone go? Defines the TS / AD split | Words |

**Reasoning**

| ID | Name | What changes | Question it answers | Targets |
|---|---|---|---|---|
| v2 | cot | Describe what you hear (speakers, tone, noise) inside `<analysis>` tags, then answer | Does reasoning out loud about the audio help? | Both |
| v12 | multi_turn_decomposition | Turn 1: "Describe the audio and the speaker's tone." Turn 2: answer | Does describing first, in a separate turn, help? | Sound |
| v11 | role_prompting | "You are an expert socio-linguist and audio analyst." | Does an expert persona change anything? | Both |

**Adding text the model could otherwise get from the audio**

| ID | Name | What changes | Question it answers | Targets |
|---|---|---|---|---|
| v3 | transcript_augmented | Audio + Whisper-large-v3 transcript | Does a transcript help, or does the model already get the words? | Words |
| v4 | transcript_augmented_cot | v3 + reasoning | Do transcript and reasoning combine? | Words |
| v18 | audio_diarized_transcript | Audio + transcript split by speaker (pyannote 3.1 + Whisper). Prompt: use it for *who and when*, judge *how* from the audio | Does knowing who spoke when help hearing? | Sound |
| v19 | audio_acoustic_features | Audio + per-speaker measurements: pitch (mean and range), loudness, speaking rate, pauses | Can measured acoustics stand in for a weak ear? | Sound |
| v20 | audio_diarized_features | Audio + v18 + v19 | Do both sources add up? | Sound |
| v21 | text_diarized_features | No audio. Text model reads v18 + v19 | How far does a full text cascade go? | Sound, via text |
| v29 | transcript_silence | v3 with the recording replaced by silence of the same length: same prompt, same transcript, nothing to hear. Run on the four in-process models (not Step-Audio). *Caveat:* silence is not "no audio"; the model still receives silent audio tokens, an unusual input that may itself disturb it | Does hearing the audio help or **hurt** the model's reading of the words? | Words (interference control) |

**Examples in the prompt**

| ID | Name | What changes | Question it answers | Targets |
|---|---|---|---|---|
| v5 / v6 | few_shot_text_only (+cot) | 3 solved MMAR examples as text only (question and answer, no audio) | Does showing the format help? | Format |
| v7 / v8 | few_shot_audio (+cot) | 3 solved examples with their audio | Do audio examples teach what to listen for? | Sound |
| v9 / v10 | few_shot_audio_transcript (+cot) | v7 with transcripts on examples and question | Audio examples + words | Both |
| v14 | rag_few_shots | The 3 examples are the most similar clips, retrieved by an audio embedding (LCO-Embedding-Omni-7B) | Do *relevant* examples beat fixed ones? | Sound |
| v16 | rag_few_shots_transcript | v14 with transcripts | Retrieval + words | Both |

**How the audio is presented**

| ID | Name | What changes | Question it answers | Targets |
|---|---|---|---|---|
| v22 | chunked_audio | Audio as labelled 5-second segments, then the full clip | Does seeing the timeline in pieces help locate evidence? | Sound |
| v23 | two_pass_localization | Turn 1: return the time span with the evidence (JSON). Turn 2: that span is cropped (±0.5 s) and played again, then answer | Does zooming in on the evidence help? | Sound |
| v28 | audio_first_instruction | Warning: "the spoken words may be misleading; the answer depends on HOW things are said" | Can an instruction shift attention from words to sound? | Sound |

**Decoding and voting (no new information)**

| ID | Name | What changes | Question it answers | Targets |
|---|---|---|---|---|
| v24–v26 | contrastive_alpha_0_0 / 0_5 / 1_0 | No generation. Each option letter is scored by its probability with the real audio minus α × its probability with silence. α = 0 is plain letter scoring | Does removing the model's "prior" answer (what it says with no sound) leave the part that comes from the audio? | Sound |
| v27 | shuffled_consistency | Each question asked 5 times with the options in different orders. Votes are mapped back and the majority wins | Is the model biased by option position, and does voting fix it? | Robustness |

Not used: v15 and v17 (retrieval + reasoning) were dropped because their token budget cut the reasoning
off, so they never produced a usable answer.

### ICBHI

| Group | IDs | What changes | Question it answers |
|---|---|---|---|
| **Controls** | c1 silence_prior | Audio replaced by silence of the same length | What does the model answer when there is nothing to hear? Any arm that does not beat this ignores the audio |
| | c2 text_only_metadata | No audio; text model reads the recording metadata (device, chest location) | How much can the metadata confound alone give? |
| | c3, c4 | Silence + metadata; silence + the acoustic dictionary and measured features | Same, with the extra text the audio arms receive |
| **Classic prompting** | r0 baseline; r0_aware | "Listen and choose the diagnosis"; + metadata | Reference point |
| | r1, r2, r5 | Pulmonologist persona; reasoning; step-by-step decomposition | Do generic prompting techniques help? |
| | r3, r4 | An *acoustic dictionary*: the textbook sound of each disease (e.g. "Pneumonia: late inspiratory fine crackles"), ± reasoning | Can the model match what it hears to a written description? |
| | r6, r7 | Held-out labelled example recordings (up to 4, the per-prompt audio limit), ± reasoning | Do audio examples teach the sounds? |
| **Measured evidence** | a1a–a1d | Signal-processing measurements of the recording as text (breathing cycles, detected wheezes and crackles), ± dictionary, ± metadata; a1c is text-only | Can numbers about the sound replace hearing it? |
| | a2a, a2b | Labels from a general sound tagger (AST on AudioSet: "Breathing", "Wheeze", "Cough"), ± features | Does a generic sound tagger help? |
| **Audio processing** | a3a–a3c | Band-pass filtering to the lung-sound range and loudness normalisation; a3b also shifts the pitch up one octave, into the range speech encoders were trained on | Is the problem that the sounds are too faint or too low for a speech encoder? |
| **Presentation** | a4a, a4b | Individual breathing cycles presented separately; two-pass zoom on an abnormal cycle | Does a lung-specific timeline help? |
| **Scoring** | a5a–a5d | Letter scoring, and contrastive scoring against silence or against silence + white + pink noise | Does removing the answer prior reveal any audio signal? |
| **Retrieval** | a6a, a6b | Diagnoses of the most similar recordings (CLAP embedding) as text, or those recordings as examples | Does retrieval from a (small) labelled pool help? |
| **Voting** | a7a | Options shuffled 5 times, majority vote | Position bias |

---

## 1.5 Metrics

### Main metrics

| Metric | Used for | Definition and why |
|---|---|---|
| **Accuracy (majority vote)** | MMAR | Share of items whose majority answer over 3 runs is correct. Majority voting removes run-to-run noise. |
| **TS / AD accuracy** | MMAR | Accuracy on the 143 transcript-solvable and 56 audio-dependent items separately: reading the words vs hearing the sound (§1.2). |
| **Balanced accuracy (BA)** | ICBHI | Recall per class averaged over the 6 main classes. ICBHI is very unbalanced (COPD is about a third), so plain accuracy would reward always answering "COPD". Any strategy that ignores the recording scores 16.7% BA. |
| **Chance** | Both | MMAR: 1 / number of options per item, averaged (32.0% overall, 31.5% TS, 33.2% AD). ICBHI: 16.7% BA. |

### Statistics

**All comparisons are paired.** When we compare two conditions (an experiment vs the baseline, or one
model vs another), both answered **the same 199 questions**. So instead of comparing two overall scores,
we compare them **question by question**:

| | Baseline right | Baseline wrong |
|---|---|---|
| **Experiment right** | unchanged (no information) | **won** |
| **Experiment wrong** | **lost** | unchanged (no information) |

*Example:* Step-Audio vs Voxtral-Small on the baseline: 35 questions won, 19 lost, 145 unchanged. The
question is whether 35 vs 19 could be a coin-flip split; it could not (p = 0.04).

Because both conditions face the same questions, differences in question difficulty cancel out, and the
test is far more sensitive than comparing two overall accuracies.


- **McNemar test:** the exact test of whether the won / lost split could be chance (the table above).
- **Holm correction:** with ~25 experiments per model, a few p-values below 0.05 are expected by chance.
  Holm adjusts each p-value for the number of comparisons; **an effect is only called significant if it
  survives Holm.**
- **Above chance (AD items):** one-sided test against a simulated guesser that picks uniformly among each
  item's options.
- **ICBHI label-shuffle test:** the true labels are shuffled across recordings 10,000 times. If the real
  BA does not beat the shuffled BAs, the answers carry no information about the diagnosis, whatever their
  level.

### Sanity metrics

| Metric | What it catches |
|---|---|
| **Top answer %** | Share of all answers that are the single most common answer. Near 100% means the model gives one answer to everything; its score then says nothing about the audio. |
| **Unparsed %** | Replies with no readable answer: refusals, format breaks, reasoning cut off by the token limit. |
| **Non-English %** | Replies that switch language. |
| **Fallback rate (v23)** | Share of first turns with no usable time span; those items are answered without the crop. |

### How each experiment is judged

The capability an experiment targets decides where its effect should appear:

| Experiment type | Judged by | "It works" means |
|---|---|---|
| Baseline, text-only (v1, v13) | Accuracy, TS / AD | — (reference points) |
| Reasoning, persona (v2, v11, v12) | Δ accuracy vs v1, on TS and AD | Gain that survives Holm; a gain on AD means it helped hearing |
| Added text (v3, v18–v20) | Δ on the subset the text is about: TS for transcripts, AD for diarization and acoustic features | Gain where targeted, **without** a loss on the other subset |
| Interference control (v29) | v3 vs v29 on the TS items, same model; v29 vs v13 | v29 above v3 on TS = the audio interferes with reading the transcript |
| Examples, retrieval (v5–v10, v14, v16) | Δ vs v1, on TS and AD; top answer % | Gain on AD = examples taught what to listen for; gain only on TS = format help |
| Presentation (v22, v23, v28) | Δ on AD; fallback rate for v23 | Gain on AD |
| Contrastive (v24–v26) | Δ vs plain letter scoring (v24) | Removing the silence prior improves accuracy |
| Shuffled voting (v27) | Δ vs v1; agreement across orders | Gain = position bias or noise removed, not better hearing |
| ICBHI controls (c1–c4) | BA and the answer distribution | An audio arm counts only if it beats its silence control |
| ICBHI audio arms | BA vs 16.7%, vs the matching silence control, and the label-shuffle test | Information about the diagnosis that the silence control lacks |

**Support or disturb.** For each technique and model we plot the change on TS against the change on AD.
Both positive: the technique helps both capabilities. Opposite signs: it trades one for the other (for
example, added text helping the words but pulling answers away from the sound). Both negative: it
disturbs the model.

---

## 1.6 Coverage and deviations

| Model | MMAR arms run | ICBHI | Notes |
|---|---|---|---|
| Voxtral-Small-24B | 27 (incl. text-only v13, v21, and v29) | 29 (incl. text-only c2, a1c) | First model; text-only arms are shared by all models |
| Qwen3-Omni-30B-A3B | 25 (incl. v29) | 27 | Text-only arms not repeated (they do not use the audio model) |
| Voxtral-Mini-3B | 25 (incl. v29) | — | MMAR only (scaling study) |
| Qwen2.5-Omni-7B | 25 (incl. v29) | — | MMAR only (scaling study) |
| Step-Audio-R1.1 | 20 | — | No v24–v26 (the HTTP server gives no letter probabilities); v22 crashed StepFun's server; v29 not run |

- **Cross-model comparisons use only shared arms.** The scaling plot uses the 19 techniques all five
  models ran.
- **Step-Audio-R1.1 is a reasoning model.** It always thinks inside `<think>…</think>` before answering,
  even in the "letter only" arms. It got a 4,096-token budget; 1.5–5.4% of its replies per arm ran out
  while still thinking and count as wrong.
- **Item exclusions:** one MMAR item from the evaluation set (gold answer names both options) and two
  from the few-shot pool (gold answer matches no option, or lists three of them).

---

## 1.7 Reproducibility

- **Code:** `src/speech_processing/` (pipeline), `main.py --dataset mmar|icbhi --experiment <id>`.
- **Experiment definitions:** `runners/mmar.py`, `runners/icbhi.py` (prompts, flags, tiers).
- **Evidence extraction:** `scripts/preprocess_transcripts.py` (Whisper transcripts),
  `scripts/extract_acoustic_features.py` (diarization and speech features),
  `scripts/extract_respiratory_features.py`, `scripts/extract_audio_tags.py`,
  `scripts/build_audio_rag_index.py`, `scripts/build_icbhi_knn_index.py`.
- **Analysis:** `scripts/analyze_runs.py` (per-model tables), `scripts/compare_models.py` (model vs
  model), `scripts/plot_model_scaling.py` (scaling figure).
- **Raw outputs:** every reply of every run, in `data/speech-processing-res-*/<run>/raw_output_run*.jsonl`,
  with the run's configuration in `run_config.json`.
