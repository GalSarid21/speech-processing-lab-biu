"""MMAR data layer: dataset loading, prompt rendering, and few-shot construction."""

import json
import os
from collections import Counter

import pandas as pd
from datasets import load_dataset
from loguru import logger
from pydantic import BaseModel

from speech_processing.config.core import DatasetConfig, ExperimentMeta
from speech_processing.data.choices import (
    answer_to_letter,
    format_lettered_choices,
    index_for_letter,
    letter_for_index,
    normalize_choices,
    shuffle_permutations,
)
from speech_processing.data.dtos import (
    AudioRequest,
    BaseRequest,
    FewShotTurn,
    ItemMetadata,
    TextRequest,
)
from speech_processing.utils.consts import (
    ANSWER_END_TAG,
    ANSWER_START_TAG,
    COT_END_TAG,
    COT_START_TAG,
    DEFAULT_NUM_FEW_SHOTS,
    NO_FEATURES_PLACEHOLDER,
    NO_TRANSCRIPT_PLACEHOLDER,
)
from speech_processing.utils.exceptions import DatasetIntegrityError, FewShotLeakageError

RANDOM_STATE = 42

# A transcript file in which most entries are one identical string was built from audio that never
# loaded: Whisper turns silence into a stock phrase ("you"). Feeding it to a transcript arm would
# silently turn that arm into a question-only arm, so it is rejected instead.
MAX_IDENTICAL_TRANSCRIPT_SHARE = 0.5
MIN_TRANSCRIPTS_FOR_DEGENERACY_CHECK = 10


MMAR_ID_COLUMN = "id"
MMAR_SUB_CATEGORY_COLUMN = "sub-category"

# Gold-answer errors in BoJack/MMAR itself. A scan of all 1000 items (2026-10-06) found 5 whose
# `answer` matches none of their `choices`. Each is handled explicitly here; any OTHER mismatch still
# raises in answer_to_letter, because it would be a new data problem rather than a known one.
#   Corrections: the gold answer is a shortened copy of exactly one choice, so the intended answer is
#   unambiguous; the value is that choice's full text.
MMAR_GOLD_CORRECTIONS: dict[str, str] = {
    "BV1NX4y1p7Xq_00-33-16_00-33-45": "Because verbal slip, 'terrierst' and 'terrorist' sound similar",
    "BV16j411E7Z9_00-00-13_00-00-34": "Octave interval leap simulates bell",
}
#   Exclusions: the gold answer cannot be mapped to one choice, so the item cannot be scored. They are
#   dropped from every evaluation and few-shot set, with a warning naming them.
MMAR_EXCLUDED_ITEMS: dict[str, str] = {
    "BV1NtQuYTELJ_00-00-00_00-00-26": "the gold answer names both choices ('American' and 'Colombian')",
    "BV1mifMY4Ebq_00-02-41_00-03-10": "the gold answer lists three of the six choices",
    "BV1P4411677K_0-00_0-20": "the gold answer ('professional performance level') matches no choice exactly",
}

FAKE_COTS = {
    "KodXqxwrFiE_00-00-00_00-00-19": "Logically, I hear a clear instructional voice pointing out a specific type of fruit. Acoustically, the speech is direct, well-articulated, and recorded in a quiet environment. Based on this, the speaker specifically points out blueberries.",
    "BV1Gt42157zM_00-00-00_00-00-17": "Logically, a girl speaks first, followed by a boy imitating her pronunciation. Acoustically, the boy artificially alters his pitch and vowel sounds to perform a mock accent. Based on this, the boy is imitating a British accent.",
    "Scaei6tdU6k_00-00-00_00-00-10": "Logically, a woman is speaking, but her words contrast with her delivery. Acoustically, her tone is highly elevated, laughing, and playful, indicating she is not being serious. Based on this, she was being playful and expressing surprise.",
}

TRANSCRIPT_PREFIX = "Transcript: "
DIARIZED_TRANSCRIPT_HEADER = "Diarized transcript:"
ACOUSTIC_FEATURES_HEADER = "Measured acoustic features:"


class MMARItem(BaseModel):
    item_id: str
    question: str
    choices: list[str]
    answer_letter: str
    audio_path: str
    transcript: str
    modality: str = ""
    category: str = ""
    sub_category: str = ""


class AcousticEvidence(BaseModel):
    """Schema of one line of the T1 acoustic-features JSONL artifact."""

    item_id: str
    diarized_transcript: str
    acoustic_features: str


def _load_mmar_frame(config: DatasetConfig) -> pd.DataFrame:
    logger.info(f"Loading {config.dataset_id} dataset from HuggingFace...")
    return load_dataset(config.dataset_id, split=config.split, streaming=False).to_pandas()


def _load_transcripts(config: DatasetConfig, required: bool) -> dict[str, str]:
    """Loads the Whisper transcripts. When an experiment uses them, a missing or degenerate file is fatal."""
    if not required:
        return {}
    if not config.transcripts_file or not os.path.exists(config.transcripts_file):
        raise DatasetIntegrityError(
            f"Transcripts file not found: {config.transcripts_file}. Run scripts/preprocess_transcripts.py first."
        )
    with open(config.transcripts_file) as f:
        transcripts: dict[str, str] = json.load(f)
    _validate_transcripts(transcripts, config.transcripts_file)
    return transcripts


def _validate_transcripts(transcripts: dict[str, str], path: str) -> None:
    if len(transcripts) < MIN_TRANSCRIPTS_FOR_DEGENERACY_CHECK:
        return
    text, count = Counter(transcripts.values()).most_common(1)[0]
    if count / len(transcripts) > MAX_IDENTICAL_TRANSCRIPT_SHARE:
        raise DatasetIntegrityError(
            f"{count} of {len(transcripts)} transcripts in {path} are the identical text {text!r}, so the file was "
            "built from audio that did not load. Rebuild it with scripts/preprocess_transcripts.py."
        )


def load_acoustic_evidence(path: str | None) -> dict[str, AcousticEvidence]:
    """Loads the T1 artifact. Missing files are fatal: the experiment would silently become the baseline."""
    if not path or not os.path.exists(path):
        raise DatasetIntegrityError(
            f"Acoustic evidence file not found: {path}. Run scripts/extract_acoustic_features.py first."
        )

    evidence: dict[str, AcousticEvidence] = {}
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            record = AcousticEvidence.model_validate_json(line)
            evidence[record.item_id] = record
    logger.info(f"Loaded acoustic evidence for {len(evidence)} items from {path}.")
    return evidence


def _parse_mmar_row(row, config: DatasetConfig, transcripts: dict[str, str]) -> MMARItem:
    item_id = str(row[MMAR_ID_COLUMN])
    choices = normalize_choices(row["choices"], item_id)
    audio_base = config.audio_base_dir or ""
    return MMARItem(
        item_id=item_id,
        question=str(row["question"]),
        choices=choices,
        answer_letter=answer_to_letter(MMAR_GOLD_CORRECTIONS.get(item_id, row["answer"]), choices, item_id),
        audio_path=os.path.join(audio_base, str(row["audio_path"]).lstrip("./")),
        transcript=transcripts.get(item_id, NO_TRANSCRIPT_PLACEHOLDER),
        modality=str(row.get("modality", "") or ""),
        category=str(row.get("category", "") or ""),
        sub_category=str(row.get(MMAR_SUB_CATEGORY_COLUMN, "") or ""),
    )


def _drop_excluded_items(df: pd.DataFrame) -> pd.DataFrame:
    """Removes the items listed in MMAR_EXCLUDED_ITEMS, naming each one that was present."""
    excluded = df[MMAR_ID_COLUMN].isin(MMAR_EXCLUDED_ITEMS)
    for item_id in df.loc[excluded, MMAR_ID_COLUMN]:
        logger.warning(f"Excluding MMAR item {item_id}: {MMAR_EXCLUDED_ITEMS[item_id]}.")
    return df[~excluded]


def _select_eval_rows(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    if config.sample_ids:
        target_ids = set(config.sample_ids)
        missing = target_ids - set(df[MMAR_ID_COLUMN].tolist())
        if missing:
            logger.warning(f"Could not find {len(missing)} requested MMAR ids (e.g. {sorted(missing)[:3]}).")
    elif config.num_samples is None:
        target_ids = set(df[MMAR_ID_COLUMN].tolist())
    else:
        target_ids = set(df[MMAR_ID_COLUMN].head(config.num_samples).tolist())

    sample_df = df[df[MMAR_ID_COLUMN].isin(target_ids)]

    if config.num_samples is not None:
        if config.num_samples > len(sample_df):
            logger.warning(f"Requested {config.num_samples} samples but only {len(sample_df)} available. Using all.")
        else:
            sample_df = sample_df.sample(n=config.num_samples, random_state=RANDOM_STATE)

    return sample_df


def _build_question_block(question: str, choices: list[str]) -> str:
    return f"Question: {question}\nChoices:\n{format_lettered_choices(choices)}"


def _build_context_block(
    item: MMARItem,
    meta: ExperimentMeta | None,
    evidence: AcousticEvidence | None,
    include_transcript: bool,
) -> str:
    parts: list[str] = []
    if include_transcript:
        parts.append(f"{TRANSCRIPT_PREFIX}{item.transcript}")
    if meta is not None and meta.inject_diarized_transcript:
        text = evidence.diarized_transcript if evidence else NO_FEATURES_PLACEHOLDER
        parts.append(f"{DIARIZED_TRANSCRIPT_HEADER}\n{text}")
    if meta is not None and meta.inject_acoustic_features:
        text = evidence.acoustic_features if evidence else NO_FEATURES_PLACEHOLDER
        parts.append(f"{ACOUSTIC_FEATURES_HEADER}\n{text}")
    return "\n\n".join(parts)


def _build_instruction(
    meta: ExperimentMeta | None, context: str, question_block: str, question: str
) -> str | list[str]:
    body = f"{context}\n\n{question_block}" if context else question_block

    if meta is None:
        return body

    if isinstance(meta.prompt, str):
        return f"{meta.prompt}\n\n{body}"

    turns = list(meta.prompt)
    if meta.presentation == "two_pass_localization":
        turns[0] = f"{turns[0]}\n\nQuestion: {question}"
    turns[-1] = f"{turns[-1]}\n\n{body}"
    return turns


def load_mmar_requests(
    config: DatasetConfig,
    experiment_meta: ExperimentMeta | None = None,
    is_text_only: bool = False,
) -> list[tuple[BaseRequest, str]]:
    """Loads the MMAR eval split and renders one request per (item, choice-order variant)."""
    df = _load_mmar_frame(config)
    sample_df = _drop_excluded_items(_select_eval_rows(df, config))
    transcripts = _load_transcripts(config, required=bool(experiment_meta and experiment_meta.use_transcript))

    evidence_by_id: dict[str, AcousticEvidence] = {}
    if experiment_meta is not None and experiment_meta.uses_acoustic_evidence:
        evidence_by_id = load_acoustic_evidence(config.acoustic_features_file)

    num_variants = experiment_meta.num_shuffled_variants if experiment_meta else 1
    results: list[tuple[BaseRequest, str]] = []
    missing_evidence = 0

    for _, row in sample_df.iterrows():
        item = _parse_mmar_row(row, config, transcripts)
        evidence = evidence_by_id.get(item.item_id)
        if experiment_meta is not None and experiment_meta.uses_acoustic_evidence and evidence is None:
            missing_evidence += 1

        original_answer_index = index_for_letter(item.answer_letter)
        context = _build_context_block(
            item,
            experiment_meta,
            evidence,
            include_transcript=bool(experiment_meta and experiment_meta.use_transcript),
        )

        for variant_idx, permutation in enumerate(shuffle_permutations(item.item_id, len(item.choices), num_variants)):
            display_choices = [item.choices[p] for p in permutation]
            question_block = _build_question_block(item.question, display_choices)
            instruction = _build_instruction(experiment_meta, context, question_block, item.question)
            ground_truth = letter_for_index(permutation.index(original_answer_index))

            metadata = ItemMetadata(
                item_id=item.item_id,
                question=item.question,
                choices=display_choices,
                transcript=item.transcript,
                modality=item.modality,
                category=item.category,
                sub_category=item.sub_category,
            )
            if num_variants > 1:
                metadata = metadata.model_copy(
                    update={
                        "original_item_id": item.item_id,
                        "variant_idx": variant_idx,
                        "permutation": permutation,
                        "original_choices": item.choices,
                    }
                )

            last_turn = instruction[-1] if isinstance(instruction, list) else instruction
            prefill = f"{COT_START_TAG}\n" if COT_START_TAG in last_turn else None
            system_prompt = experiment_meta.system_prompt if experiment_meta else None

            if is_text_only:
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
                    metadata=metadata,
                )
            results.append((request, ground_truth))

    if missing_evidence:
        logger.warning(f"{missing_evidence} eval items have no acoustic evidence; using {NO_FEATURES_PLACEHOLDER}.")
    logger.info(f"Successfully prepared {len(results)} MMAR requests.")
    return results


def read_few_shot_ids(config: DatasetConfig) -> list[str]:
    if not config.few_shot_ids_file or not os.path.exists(config.few_shot_ids_file):
        logger.warning("No few-shot IDs file found. Cannot load few-shot examples.")
        return []
    with open(config.few_shot_ids_file) as f:
        return [line.strip() for line in f if line.strip()]


def _load_mmar_items(config: DatasetConfig, ids: list[str], meta: ExperimentMeta) -> list[MMARItem]:
    if not ids:
        return []
    df = _drop_excluded_items(_load_mmar_frame(config))
    transcripts = _load_transcripts(config, required=meta.few_shot_include_transcript)
    by_id = {str(row[MMAR_ID_COLUMN]): row for _, row in df[df[MMAR_ID_COLUMN].isin(ids)].iterrows()}
    return [_parse_mmar_row(by_id[item_id], config, transcripts) for item_id in ids if item_id in by_id]


def _build_few_shot_turn(item: MMARItem, meta: ExperimentMeta) -> FewShotTurn:
    question_block = _build_question_block(item.question, item.choices)
    user_text = (
        f"{TRANSCRIPT_PREFIX}{item.transcript}\n\n{question_block}"
        if meta.few_shot_include_transcript
        else question_block
    )

    assistant_text = item.answer_letter
    if meta.use_cot:
        reasoning = FAKE_COTS.get(item.item_id, f"The correct choice is {item.answer_letter}.")
        assistant_text = (
            f"{COT_START_TAG}\n{reasoning}\n{COT_END_TAG}\n{ANSWER_START_TAG}\n{item.answer_letter}\n{ANSWER_END_TAG}"
        )

    return FewShotTurn(
        audio_path=None if meta.few_shot_mode == "text" else item.audio_path,
        audio_bytes=None,
        user_text=user_text,
        assistant_text=assistant_text,
    )


def get_mmar_few_shot_turns(
    config: DatasetConfig, meta: ExperimentMeta, num_shots: int = DEFAULT_NUM_FEW_SHOTS
) -> tuple[list[str], list[FewShotTurn]]:
    """Returns the shot ids alongside the turns so the caller can assert disjointness."""
    shot_ids = read_few_shot_ids(config)[:num_shots]
    items = _load_mmar_items(config, shot_ids, meta)
    turns = [_build_few_shot_turn(item, meta) for item in items]
    logger.info(f"Successfully loaded {len(turns)} authentic MMAR few-shot examples.")
    return [item.item_id for item in items], turns


def get_mmar_few_shot_turns_pool(config: DatasetConfig, meta: ExperimentMeta) -> dict[str, FewShotTurn]:
    items = _load_mmar_items(config, read_few_shot_ids(config), meta)
    return {item.item_id: _build_few_shot_turn(item, meta) for item in items}


def validate_few_shot_disjointness(
    eval_ids: set[str], shot_ids: set[str], rag_mapping: dict[str, list[str]] | None = None
) -> None:
    """Raises when a few-shot example is also an eval item, or when RAG retrieves one."""
    overlap = eval_ids & shot_ids
    if overlap:
        raise FewShotLeakageError(f"{len(overlap)} few-shot ids are also eval items (e.g. {sorted(overlap)[:3]}).")

    if not rag_mapping:
        return

    for item_id in eval_ids:
        retrieved = set(rag_mapping.get(item_id, []))
        if item_id in retrieved:
            raise FewShotLeakageError(f"RAG mapping retrieves eval item {item_id} for itself.")
        leaked = retrieved & eval_ids
        if leaked:
            raise FewShotLeakageError(
                f"RAG mapping for eval item {item_id} retrieves other eval items: {sorted(leaked)[:3]}."
            )
