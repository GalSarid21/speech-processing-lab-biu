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
| `icbhi_near_duplicates.json` | `scripts/diagnose_icbhi.py` |
| `icbhi_respiratory_features.jsonl` | `scripts/extract_respiratory_features.py` |
| `icbhi_audio_tags.jsonl` | `scripts/extract_audio_tags.py` |
| `icbhi_neighbor_labels.json`, `icbhi_rag_mapping.json` | `scripts/build_icbhi_knn_index.py` |
| `icbhi_subset_split.json` | `scripts/analyze_runs.py --dataset icbhi --write-split-from <c2 run dir>` |

### Patient leakage: read this before trusting any ICBHI few-shot number

The HuggingFace packaging replaces the original ICBHI filenames (which encode patient id, chest
location, acquisition mode and device) with `audioN.wav`. **Patient identity is therefore lost.**

The 174 clips come from far fewer than 174 patients. Any demonstration pool drawn from those same
174 clips may share a patient with the evaluation item, which would let the model match the same
person's recording rather than reason about the sound.

**Option A — what the code does today.** Demonstrations exclude the test item itself and anything
`scripts/diagnose_icbhi.py` flagged as a near-duplicate (a clip cut from the same source recording).
Patient overlap beyond that cannot be ruled out, so every few-shot and kNN result is an **upper
bound**, and must be reported as one.

**Option B — the fix, not yet done.** Download the original *ICBHI 2017 Respiratory Sound Database*
(920 recordings, 126 patients), which ships a patient-diagnosis file and per-cycle crackle/wheeze
annotations:

1. Fingerprint-match each of the 174 test clips against the full database to recover its source
   recording, and therefore its patient.
2. Build the demonstration pool only from patients that do not appear in the 174.
3. Bonus: the cycle annotations are real acoustic ground truth. They make an honest
   crackle/wheeze detection accuracy possible, and they are the only way to validate the detectors
   in `scripts/extract_respiratory_features.py` rather than spot-checking them by ear.

The database is distributed for research use from the ICBHI 2017 challenge site. Check its terms
before redistributing anything derived from it. **Do not commit the audio.**

### Why the recording metadata is quarantined

The dataset's own `instruction` states the chest location and the acquisition mode. On ICBHI the
acquisition mode tracks the clinical site and therefore the diagnosis, so a prompt that repeats it
lets a model score well without listening. Every ICBHI experiment therefore declares
`include_recording_metadata`, and the blind track (the default) never shows it. Run
`scripts/diagnose_icbhi.py` first: its report prints the metadata ceiling that the audio techniques
have to be read against.
