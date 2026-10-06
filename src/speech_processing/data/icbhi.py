"""ICBHI 2017 data layer.

Two things make this dataset different from MMAR and drive the design here:

1. The native `instruction` states the chest location and the acquisition mode, and on ICBHI the
   acquisition mode is tied to the recording site and therefore to the diagnosis. A prompt that
   carries it lets a model "diagnose" from metadata alone, so every experiment declares whether it
   runs metadata-blind (the primary track) or metadata-aware (reported with a text-only control).
2. The label set is closed and known, so the answer is a letter and scoring is deterministic.
"""

import json
import os
import random
import re

import pandas as pd
from datasets import Audio, load_dataset
from loguru import logger
from pydantic import BaseModel

from speech_processing.config.core import DatasetConfig, ExperimentMeta
from speech_processing.data.choices import (
    answer_to_letter,
    format_lettered_choices,
    index_for_letter,
    letter_for_index,
    shuffle_permutations,
)
from speech_processing.data.dtos import AudioRequest, BaseRequest, FewShotTurn, ItemMetadata, TextRequest
from speech_processing.utils.audio import write_audio_bytes
from speech_processing.utils.consts import (
    ANSWER_END_TAG,
    ANSWER_START_TAG,
    COT_END_TAG,
    COT_START_TAG,
    ICBHI_AUDIO_CACHE_DIR,
    MAX_EXTRA_AUDIOS_PER_PROMPT,
    NO_FEATURES_PLACEHOLDER,
)
from speech_processing.utils.exceptions import DatasetIntegrityError, FewShotLeakageError

# The 8 ICBHI 2017 patient-level diagnoses, in a fixed order so letters are stable across runs.
ICBHI_LABELS: tuple[str, ...] = (
    "COPD",
    "Pneumonia",
    "URTI",
    "LRTI",
    "Bronchiectasis",
    "Bronchiolitis",
    "Asthma",
    "No potential disease detected",
)
HEALTHY_LABEL = "No potential disease detected"

# The classes the primary metric averages over. Asthma (1 item) and LRTI (2 items) stay in the
# answer options, so the task is still the 8-way ICBHI diagnosis, but they are excluded from the
# macro average: with them in, a single Asthma prediction moves balanced accuracy by 12.5 points and
# the 95% CI roughly doubles (about +-15 points instead of +-8). Fixed here rather than derived
# from support counts, so the choice is pre-registered and cannot drift with the data.
ICBHI_PRIMARY_LABELS: tuple[str, ...] = (
    "COPD",
    "Pneumonia",
    "URTI",
    "Bronchiectasis",
    "Bronchiolitis",
    "No potential disease detected",
)
DEMONSTRATION_SEED = 2017
# A clip cut from a longer recording overlaps at most a few neighbouring cuts. Far more "near-duplicates"
# per demonstration means the detector is misfiring (the first one flagged 170 of 174 real clips), and
# holding them all out would gut the evaluation set.
MAX_NEAR_DUPLICATES_PER_DEMONSTRATION = 5

ICBHI_FILE_COLUMN = "file"
ICBHI_LABEL_COLUMN = "label"
ICBHI_INSTRUCTION_COLUMN = "instruction"
ICBHI_AUDIO_COLUMN = "audio"

ICBHI_QUESTION = "What is the patient's diagnosis?"
RECORDING_METADATA_TEMPLATE = "Recording site: {location}. Acquisition mode: {mode}."
RESPIRATORY_FEATURES_HEADER = "Measured respiratory features:"
AUDIO_TAGS_HEADER = "Audio tagger output:"
NEIGHBOR_LABELS_TEMPLATE = "The {count} most acoustically similar reference recordings were diagnosed as: {labels}."

_METADATA_PATTERN = re.compile(
    r"recorded at the patient'?s\s+(?P<location>[a-z][a-z \-]*?)\s*,\s*"
    r"with the acquisition mode being\s+(?P<mode>single channel|multichannel)",
    re.IGNORECASE,
)

SINGLE_CHANNEL = "single channel"
MULTICHANNEL = "multichannel"


class ICBHIItem(BaseModel):
    item_id: str
    audio_path: str
    audio_bytes: bytes | None = None
    label: str
    label_letter: str
    chest_location: str
    acquisition_mode: str


class RespiratoryEvidence(BaseModel):
    """One line of the A1 measured-feature artifact."""

    item_id: str
    cycles_summary: str
    adventitious_summary: str
    cycle_spans: list[tuple[float, float]] = []


class AudioTagEvidence(BaseModel):
    """One line of the A2 audio-tagger artifact."""

    item_id: str
    tags_summary: str


def parse_instruction_metadata(instruction: str) -> tuple[str, str]:
    """Extracts (chest location, acquisition mode) from a native ICBHI instruction."""
    match = _METADATA_PATTERN.search(instruction)
    if not match:
        raise DatasetIntegrityError(f"No recording metadata in ICBHI instruction: {instruction!r}")
    return match.group("location").strip().lower(), match.group("mode").strip().lower()


def _load_icbhi_frame(config: DatasetConfig, split: str | None = None) -> pd.DataFrame:
    logger.info(f"Loading {config.dataset_id} dataset from HuggingFace...")
    ds = load_dataset(config.dataset_id, split=split or config.split)
    # Do not decode, so the raw bytes can be written out verbatim.
    return ds.cast_column(ICBHI_AUDIO_COLUMN, Audio(decode=False)).to_pandas()


def _audio_bytes_of(row) -> bytes | None:
    audio = row[ICBHI_AUDIO_COLUMN]
    if isinstance(audio, dict) and audio.get("bytes"):
        return audio["bytes"]
    return None


def _item_id_of(file_name: str) -> str:
    return os.path.splitext(os.path.basename(file_name))[0]


def _parse_icbhi_row(row, cache_dir: str) -> ICBHIItem:
    file_name = str(row[ICBHI_FILE_COLUMN])
    item_id = _item_id_of(file_name)
    label = str(row[ICBHI_LABEL_COLUMN]).strip()

    if label not in ICBHI_LABELS:
        raise DatasetIntegrityError(f"Item {item_id}: label {label!r} is not an ICBHI diagnosis.")

    location, mode = parse_instruction_metadata(str(row[ICBHI_INSTRUCTION_COLUMN]))
    audio_bytes = _audio_bytes_of(row)
    audio_path = (
        write_audio_bytes(audio_bytes, file_name, cache_dir) if audio_bytes else os.path.join(cache_dir, file_name)
    )

    return ICBHIItem(
        item_id=item_id,
        audio_path=audio_path,
        audio_bytes=audio_bytes,
        label=label,
        label_letter=answer_to_letter(label, list(ICBHI_LABELS), item_id),
        chest_location=location,
        acquisition_mode=mode,
    )


def _load_jsonl_evidence(path: str | None, schema: type[BaseModel], what: str) -> dict:
    if not path or not os.path.exists(path):
        raise DatasetIntegrityError(f"{what} file not found: {path}. Generate it before running this experiment.")

    evidence = {}
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            record = schema.model_validate_json(line)
            evidence[record.item_id] = record
    logger.info(f"Loaded {what} for {len(evidence)} items from {path}.")
    return evidence


def load_respiratory_evidence(path: str | None) -> dict[str, RespiratoryEvidence]:
    return _load_jsonl_evidence(path, RespiratoryEvidence, "Respiratory features")


def load_audio_tag_evidence(path: str | None) -> dict[str, AudioTagEvidence]:
    return _load_jsonl_evidence(path, AudioTagEvidence, "Audio tags")


def load_neighbor_labels(path: str | None) -> dict[str, list[str]]:
    if not path or not os.path.exists(path):
        raise DatasetIntegrityError(f"Neighbour labels file not found: {path}. Run scripts/build_icbhi_knn_index.py.")
    with open(path) as f:
        return json.load(f)


def _build_question_block(choices: list[str]) -> str:
    return f"Question: {ICBHI_QUESTION}\nChoices:\n{format_lettered_choices(choices)}"


def _build_context_block(
    item: ICBHIItem,
    meta: ExperimentMeta,
    respiratory: RespiratoryEvidence | None,
    tags: AudioTagEvidence | None,
    neighbors: list[str] | None,
) -> str:
    parts: list[str] = []

    if meta.include_recording_metadata:
        parts.append(RECORDING_METADATA_TEMPLATE.format(location=item.chest_location, mode=item.acquisition_mode))
    if meta.inject_acoustic_features:
        body = (
            f"{respiratory.cycles_summary}\n{respiratory.adventitious_summary}"
            if respiratory
            else NO_FEATURES_PLACEHOLDER
        )
        parts.append(f"{RESPIRATORY_FEATURES_HEADER}\n{body}")
    if meta.inject_audio_tags:
        parts.append(f"{AUDIO_TAGS_HEADER}\n{tags.tags_summary if tags else NO_FEATURES_PLACEHOLDER}")
    if meta.inject_neighbor_labels:
        body = (
            NEIGHBOR_LABELS_TEMPLATE.format(count=len(neighbors), labels=", ".join(neighbors))
            if neighbors
            else NO_FEATURES_PLACEHOLDER
        )
        parts.append(body)

    return "\n\n".join(parts)


def _build_instruction(meta: ExperimentMeta, context: str, choices: list[str]) -> str | list[str]:
    question_block = _build_question_block(choices)
    body = f"{context}\n\n{question_block}" if context else question_block

    if isinstance(meta.prompt, str):
        return f"{meta.prompt}\n\n{body}"

    turns = list(meta.prompt)
    turns[-1] = f"{turns[-1]}\n\n{body}"
    return turns


def load_icbhi_requests(config: DatasetConfig, meta: ExperimentMeta | None = None) -> list[tuple[BaseRequest, str]]:
    """Builds one request per evaluation row, in dataset order.

    The evaluation set is every row except the fixed held-out demonstration split, and it is the
    same for every experiment - zero-shot ones included - so a few-shot vs zero-shot difference is
    never partly a different-items difference. `num_samples` only truncates, for smoke tests.
    """
    held_out = load_demonstration_split(config.demo_ids_file).held_out_ids
    df = _load_icbhi_frame(config)
    cache_dir = config.audio_base_dir or ICBHI_AUDIO_CACHE_DIR

    # Remove the held-out split before truncating, so --num-samples N means N scored items.
    df = df[~df[ICBHI_FILE_COLUMN].astype(str).map(_item_id_of).isin(held_out)]
    if config.num_samples is not None and config.num_samples < len(df):
        logger.warning(
            f"Truncating ICBHI to the first {config.num_samples} of {len(df)} evaluation rows. "
            "This is a smoke test, not the evaluation protocol."
        )
        df = df.head(config.num_samples)

    respiratory: dict[str, RespiratoryEvidence] = {}
    tags: dict[str, AudioTagEvidence] = {}
    neighbors: dict[str, list[str]] = {}
    if meta is not None:
        if meta.inject_acoustic_features or meta.needs_cycle_spans:
            respiratory = load_respiratory_evidence(config.respiratory_features_file)
        if meta.inject_audio_tags:
            tags = load_audio_tag_evidence(config.audio_tags_file)
        if meta.inject_neighbor_labels:
            neighbors = load_neighbor_labels(config.neighbor_labels_file)

    num_variants = meta.num_shuffled_variants if meta else 1
    results: list[tuple[BaseRequest, str]] = []
    missing_evidence = 0

    for _, row in df.iterrows():
        item = _parse_icbhi_row(row, cache_dir)
        item_respiratory = respiratory.get(item.item_id)
        item_tags = tags.get(item.item_id)

        if meta is not None and meta.uses_acoustic_evidence and not (item_respiratory or item_tags):
            missing_evidence += 1

        context = (
            _build_context_block(item, meta, item_respiratory, item_tags, neighbors.get(item.item_id))
            if meta is not None
            else ""
        )
        original_answer_index = index_for_letter(item.label_letter)

        for variant_idx, permutation in enumerate(shuffle_permutations(item.item_id, len(ICBHI_LABELS), num_variants)):
            display_choices = [ICBHI_LABELS[p] for p in permutation]
            instruction = (
                _build_instruction(meta, context, display_choices)
                if meta is not None
                else _build_question_block(display_choices)
            )
            ground_truth = letter_for_index(permutation.index(original_answer_index))

            metadata = ItemMetadata(
                item_id=item.item_id,
                question=ICBHI_QUESTION,
                choices=display_choices,
                category=item.label,
                sub_category=f"{item.acquisition_mode}|{item.chest_location}",
                cycle_spans=item_respiratory.cycle_spans if item_respiratory else None,
            )
            if num_variants > 1:
                metadata = metadata.model_copy(
                    update={
                        "original_item_id": item.item_id,
                        "variant_idx": variant_idx,
                        "permutation": permutation,
                        "original_choices": list(ICBHI_LABELS),
                    }
                )

            last_turn = instruction[-1] if isinstance(instruction, list) else instruction
            prefill = f"{COT_START_TAG}\n" if COT_START_TAG in last_turn else None
            system_prompt = meta.system_prompt if meta else None

            if meta is not None and meta.text_only:
                request: BaseRequest = TextRequest(
                    instruction=instruction,
                    assistant_prefill=prefill,
                    system_prompt=system_prompt,
                    metadata=metadata,
                )
            else:
                request = AudioRequest(
                    instruction=instruction,
                    assistant_prefill=prefill,
                    system_prompt=system_prompt,
                    audio_path=item.audio_path,
                    audio_bytes=item.audio_bytes,
                    metadata=metadata,
                )
            results.append((request, ground_truth))

    if missing_evidence:
        logger.warning(f"{missing_evidence} ICBHI items have no evidence; using {NO_FEATURES_PLACEHOLDER}.")
    logger.info(f"Successfully prepared {len(results)} ICBHI requests.")
    return results


class DemonstrationSplit(BaseModel):
    """The fixed set of items held out of evaluation so they can serve as demonstrations.

    Written once by `scripts/diagnose_icbhi.py` and read by every ICBHI experiment, so the
    evaluation set is identical across experiments and auditable after the fact.
    """

    demonstration_ids: list[str]
    excluded_near_duplicate_ids: list[str] = []
    seed: int = DEMONSTRATION_SEED

    @property
    def held_out_ids(self) -> set[str]:
        """Everything kept out of evaluation: the demonstrations and anything cut from the same recording."""
        return set(self.demonstration_ids) | set(self.excluded_near_duplicate_ids)


def select_demonstration_split(
    labels_by_id: dict[str, str],
    near_duplicates: dict[str, list[str]],
    labels: tuple[str, ...] = ICBHI_PRIMARY_LABELS,
    seed: int = DEMONSTRATION_SEED,
) -> DemonstrationSplit:
    """Picks one demonstration per primary class, preferring clips with no near-duplicates.

    A demonstration's near-duplicates are held out of evaluation as well; otherwise the model would
    be scored on a near-copy of a labelled example it was just shown.
    """
    rng = random.Random(seed)
    chosen: list[str] = []

    for label in labels:
        candidates = sorted(item_id for item_id, item_label in labels_by_id.items() if item_label == label)
        if not candidates:
            raise DatasetIntegrityError(f"No item of class {label!r} is available as a demonstration.")
        rng.shuffle(candidates)
        # Stable sort: among equally clean candidates, the seeded shuffle decides.
        candidates.sort(key=lambda item_id: len(near_duplicates.get(item_id, [])))
        chosen.append(candidates[0])

    excluded = {duplicate for item_id in chosen for duplicate in near_duplicates.get(item_id, [])} - set(chosen)
    if len(excluded) > MAX_NEAR_DUPLICATES_PER_DEMONSTRATION * len(chosen):
        raise DatasetIntegrityError(
            f"The {len(chosen)} demonstrations have {len(excluded)} near-duplicates between them, out of "
            f"{len(labels_by_id)} items. That is the near-duplicate detector misfiring, not a property of the "
            "data; refusing to hold them all out of evaluation."
        )
    return DemonstrationSplit(demonstration_ids=chosen, excluded_near_duplicate_ids=sorted(excluded), seed=seed)


def load_demonstration_split(path: str | None) -> DemonstrationSplit:
    """Missing is fatal: running without it would silently evaluate a different item set."""
    if not path or not os.path.exists(path):
        raise DatasetIntegrityError(
            f"Demonstration split not found: {path}. Run scripts/diagnose_icbhi.py first - it fixes "
            "the held-out demonstration items, and with them the evaluation set every experiment uses."
        )
    with open(path) as f:
        return DemonstrationSplit.model_validate_json(f.read())


def validate_demonstration_disjointness(eval_ids: set[str], demonstration_ids: set[str]) -> None:
    """A labelled demonstration may never also be an item that gets scored."""
    leaked = eval_ids & demonstration_ids
    if leaked:
        raise FewShotLeakageError(
            f"{len(leaked)} demonstration items are also evaluation items (e.g. {sorted(leaked)[:3]})."
        )


def _build_icbhi_few_shot_turn(item: ICBHIItem, meta: ExperimentMeta, rationale: str | None) -> FewShotTurn:
    assistant_text = item.label_letter
    if meta.use_cot:
        reasoning = rationale or f"The auscultation pattern is consistent with {item.label}."
        assistant_text = (
            f"{COT_START_TAG}\n{reasoning}\n{COT_END_TAG}\n{ANSWER_START_TAG}\n{item.label_letter}\n{ANSWER_END_TAG}"
        )

    return FewShotTurn(
        audio_path=None if meta.few_shot_mode == "text" else item.audio_path,
        audio_bytes=None,
        user_text=_build_question_block(list(ICBHI_LABELS)),
        assistant_text=assistant_text,
    )


# One prompt slot is the test clip itself, so the demonstrations get everything below the limit.
MAX_DEMONSTRATIONS = MAX_EXTRA_AUDIOS_PER_PROMPT


def _load_items_by_id(config: DatasetConfig) -> dict[str, ICBHIItem]:
    cache_dir = config.audio_base_dir or ICBHI_AUDIO_CACHE_DIR
    items = (_parse_icbhi_row(row, cache_dir) for _, row in _load_icbhi_frame(config).iterrows())
    return {item.item_id: item for item in items}


def build_icbhi_few_shot_pool(config: DatasetConfig, meta: ExperimentMeta) -> dict[str, FewShotTurn]:
    """The held-out demonstrations, one per primary class, in the split's fixed order.

    They are disjoint from the evaluation set by construction. Patient ids are not published in this
    packaging, so a demonstration may still share a patient with an evaluation item; see
    data/README.md for the patient-disjoint alternative.
    """
    split = load_demonstration_split(config.demo_ids_file)
    if len(split.demonstration_ids) > MAX_DEMONSTRATIONS:
        raise DatasetIntegrityError(
            f"{len(split.demonstration_ids)} demonstrations plus the test clip exceed the "
            f"{MAX_DEMONSTRATIONS + 1} audio clips Voxtral accepts in one prompt."
        )

    items = _load_items_by_id(config)
    missing = [item_id for item_id in split.demonstration_ids if item_id not in items]
    if missing:
        raise DatasetIntegrityError(f"Demonstration ids not found in the dataset: {missing}")

    logger.info(f"Using {len(split.demonstration_ids)} held-out ICBHI demonstrations.")
    return {
        item_id: _build_icbhi_few_shot_turn(items[item_id], meta, rationale=None) for item_id in split.demonstration_ids
    }


def build_icbhi_neighbor_pool(config: DatasetConfig, meta: ExperimentMeta) -> dict[str, FewShotTurn]:
    """Every held-out item as a retrieval candidate for the kNN few-shot track (A6b).

    Only held-out items qualify: retrieving from the evaluation set would show the model other scored
    items with their labels attached.
    """
    held_out = load_demonstration_split(config.demo_ids_file).held_out_ids
    items = _load_items_by_id(config)
    return {
        item_id: _build_icbhi_few_shot_turn(items[item_id], meta, rationale=None)
        for item_id in sorted(held_out)
        if item_id in items
    }
