# Data directory

Nothing here is committed: `/data/` is gitignored. This file records where each artifact comes
from, what generates it, and the licence position.

## MMAR

| File | Produced by |
|---|---|
| `MMAR/` | audio from the `BoJack/MMAR` HuggingFace dataset |
| `mmar_transcripts.json` | `scripts/preprocess_transcripts.py` |
| `mmar_acoustic_features.jsonl` | `scripts/extract_acoustic_features.py` |
| `mmar_rag_mapping.json` | `scripts/build_audio_rag_index.py` |
| `mmar_en_speech_few_shot_ids.txt` | curated by hand |
| `mmar_subset_split.json` | `scripts/analyze_runs.py --dataset mmar --write-split-from <v13 run dir>` |

## ICBHI

| File | Produced by |
|---|---|
| `ICBHI/` | audio materialised from `DynamicSuperb/RespiratorySoundClassification_ICBHI2017` |
| `icbhi_demo_ids.json` | `scripts/diagnose_icbhi.py` — **required by every ICBHI experiment** |
| `icbhi_near_duplicates.json` | `scripts/diagnose_icbhi.py` |
| `icbhi_respiratory_features.jsonl` | `scripts/extract_respiratory_features.py` |
| `icbhi_audio_tags.jsonl` | `scripts/extract_audio_tags.py` |
| `icbhi_neighbor_labels.json`, `icbhi_rag_mapping.json` | `scripts/build_icbhi_knn_index.py` (needs `icbhi_demo_ids.json`) |
| `icbhi_subset_split.json` | `scripts/analyze_runs.py --dataset icbhi --write-split-from <c3 run dir>` |
| `icbhi_experiment_sample_ids.txt` | `scripts/create_balanced_subset.py` — **V1/V3 only, not used by V4** |

### The evaluation set

The HuggingFace packaging has a single `test` split of 174 rows; there is no `train` split to draw
demonstrations from. So V4 holds a few rows out:

* `scripts/diagnose_icbhi.py` picks **one demonstration per primary class** (6 clips), preferring
  clips with no near-duplicate, and also holds out any near-duplicate of a chosen clip. The choice is
  seeded and written to `icbhi_demo_ids.json`.
* Every ICBHI experiment, zero-shot ones included, scores **the remaining rows (168 on the real
  data)**. A demonstration is never scored, and a few-shot vs zero-shot difference is never partly a
  different-items difference. Without the file, every run refuses to start.
* Run `diagnose_icbhi.py` **once**. Re-running it after experiments have started would change the
  evaluation set under them.

### Class balance, and why the evaluation set is not subsampled

The real distribution is COPD 60, no disease 35, Pneumonia 24, URTI 23, Bronchiectasis 16,
Bronchiolitis 13, LRTI 2, Asthma 1. Always answering COPD scores 34.5% plain accuracy.

The fix is the metric, not the sample. The primary metric is **balanced accuracy** (macro recall),
under which every class counts equally, so always-COPD scores exactly chance. Subsampling to 13
items per class would give the same guarantee while discarding 90 items: its 95% CI is about ±10
points against about ±8 for balanced accuracy on the full evaluation set.

Balanced accuracy averages over the **6 primary classes**. Asthma and LRTI stay answer options and
count towards accuracy, but with 1 and 2 items they would each carry 1/8 of a macro average: one
Asthma prediction would move it 12.5 points, and the CI would roughly double.

#### The balanced subset is not part of V4

`scripts/create_balanced_subset.py` (V1/V3) sampled *proportionally* — it kept the 35% COPD share,
so it never balanced anything. V1 and V3 also ended up with different class mixes (60% vs 35% COPD),
which is why no V1-vs-V3 comparison holds. `run_icbhi` rejects `--sample-ids-file` outright rather
than accepting it and quietly evaluating something else. Use `--num-samples` for smoke tests.

### Patient leakage

The packaging replaces the original ICBHI filenames (which encode patient id, chest location,
acquisition mode and device) with `audioN.wav`, so **patient identity is lost**, and the 174 clips
come from far fewer than 174 patients.

**What the code does today.** Demonstrations come only from the held-out split, so no labelled
example is ever a scored item, and near-duplicates of a demonstration are held out with it. What
cannot be ruled out is a demonstration and an evaluation item coming from the same patient through
different recordings. Report every few-shot result with that caveat.

Retrieval (A6) is stricter still: candidates must come from the held-out split, or the model would
be handed other scored items' labels as evidence. Six candidates is too few to retrieve from, so A6
stays future work until a larger disjoint pool exists.

**The full fix, not yet done.** Download the original *ICBHI 2017 Respiratory Sound Database* (920
recordings, 126 patients), which ships a patient-diagnosis file and per-cycle crackle/wheeze
annotations:

1. Fingerprint-match each of the 174 clips against the full database to recover its source
   recording, and therefore its patient.
2. Build the demonstration and retrieval pools only from patients that do not appear in the 174.
3. Bonus: the cycle annotations are real acoustic ground truth. They make an honest crackle/wheeze
   detection accuracy possible, and they are the only way to validate the detectors in
   `scripts/extract_respiratory_features.py` rather than spot-checking them by ear.

The database is distributed for research use from the ICBHI 2017 challenge site. Check its terms
before redistributing anything derived from it. **Do not commit the audio.**

### Why the recording metadata is quarantined

The dataset's own `instruction` states the chest location and the acquisition mode. On ICBHI the
acquisition mode tracks the clinical site and therefore the diagnosis, so a prompt that repeats it
lets a model score well without listening. Every ICBHI experiment therefore declares
`include_recording_metadata`, and the blind track (the default) never shows it. On the real data the
acquisition mode alone reaches about 33% balanced accuracy against 16.7% chance — the metadata
ceiling `scripts/diagnose_icbhi.py` reports, and the bar for every metadata-aware result.
