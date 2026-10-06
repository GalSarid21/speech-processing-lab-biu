"""ICBHI 2017 V4 runner.

Two tracks run for every family of techniques:

* metadata-blind (primary) - the prompt never states the chest location or the acquisition mode.
  On ICBHI the acquisition mode is tied to the recording site and therefore to the diagnosis, so a
  metadata-aware prompt can be "solved" without listening at all.
* metadata-aware (secondary, marked with the AWARE suffix) - reported only alongside the control
  that measures how much of the score the metadata alone buys.

Every control is an audio ablation of a reported arm: the same model and the same prompt, with the
recording swapped for duration-matched silence. Whatever such a control scores, the model scored
without hearing anything, so the gap to its parent arm is exactly what the audio contributed.

All experiments score the same items: every row except the held-out demonstration split fixed by
scripts/diagnose_icbhi.py.
"""

import json
import os
import time
from datetime import UTC, datetime
from enum import Enum

from loguru import logger

from speech_processing.config.core import AppConfig, ExperimentMeta
from speech_processing.config.factories import ICBHI_RAG_MAPPING_FILE, create_icbhi_config
from speech_processing.data.dtos import BaseRequest, ChoiceExtractionResult
from speech_processing.data.enums import DatasetType
from speech_processing.data.icbhi import (
    ICBHI_LABELS,
    RespiratoryEvidence,
    build_icbhi_few_shot_pool,
    build_icbhi_neighbor_pool,
    load_icbhi_requests,
    load_respiratory_evidence,
    validate_demonstration_disjointness,
)
from speech_processing.evaluation.metrics import calculate_icbhi_metrics
from speech_processing.evaluation.stability import calculate_stability
from speech_processing.models.audio import VoxtralAudioEngine
from speech_processing.models.contrastive import ContrastiveChoiceScorer
from speech_processing.models.judge import QwenJudge
from speech_processing.models.preprocessing import build_preprocessor
from speech_processing.models.presentation import build_presentation
from speech_processing.models.text import GemmaTextModel
from speech_processing.pipelines.aggregation import (
    IdentityAggregator,
    ResponseAggregator,
    ShuffledMajorityVoteAggregator,
)
from speech_processing.pipelines.base import log_gpu_memory, release_vram
from speech_processing.pipelines.inference import InferencePipeline
from speech_processing.pipelines.judge import JudgePipeline, ParserOnlyPipeline
from speech_processing.prompts.templates.judge.qwen import build_icbhi_choice_judge_conversation
from speech_processing.runners.base import parse_args
from speech_processing.utils.consts import (
    COT_END_TAG,
    COT_START_TAG,
    JUDGE_LOAD_DELAY_S,
    MAX_EXTRA_AUDIOS_PER_PROMPT,
)
from speech_processing.utils.exceptions import DatasetIntegrityError, DeprecatedExperimentError

ICBHI = DatasetType.ICBHI

ANSWER_WITH_LETTER = "Respond with only the letter of the correct choice."
ENGLISH_ONLY = "You MUST respond entirely in English."

COT_FORMAT = (
    "Format your response exactly as follows:\n"
    f"{COT_START_TAG}\n[Your description of the breathing cycles and adventitious sounds here]\n{COT_END_TAG}\n"
    "<answer>\n[The letter of the correct choice]\n</answer>"
)

BASELINE_PROMPT = (
    "Listen to this lung auscultation recording and choose the patient's diagnosis. "
    f"{ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)

_ROLE_PROMPT = (
    "You are an experienced pulmonologist auscultating a patient. Listen to the recording and choose the "
    f"patient's diagnosis. {ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_COT_PROMPT = (
    "Listen to this lung auscultation recording. First describe the breathing cycles: the respiratory rate, "
    "the relative length of inspiration and expiration, and any adventitious sounds (crackles, wheezes) with "
    "their timing within the cycle. Then choose the diagnosis that best matches your description. "
    f"{ENGLISH_ONLY}\n\n{COT_FORMAT}"
)

# Classic auscultation signatures, taken from clinical references rather than tuned on this eval set.
ACOUSTIC_DICTIONARY = (
    "Reference auscultation signatures:\n"
    "- COPD: prolonged expiration, expiratory wheezes, early inspiratory crackles, diminished breath sounds.\n"
    "- Pneumonia: late inspiratory fine crackles over the affected lobe, bronchial breathing, egophony.\n"
    "- URTI: largely normal lower-airway sounds, transmitted upper-airway noise, occasional coarse rhonchi.\n"
    "- LRTI: scattered coarse crackles and rhonchi, usually bilateral.\n"
    "- Bronchiectasis: coarse mid-inspiratory crackles, often with wheeze, changing after coughing.\n"
    "- Bronchiolitis: high-pitched expiratory wheezes with fine end-inspiratory crackles, mostly in infants.\n"
    "- Asthma: polyphonic expiratory wheezes, prolonged expiration, no crackles.\n"
    "- No potential disease detected: vesicular breath sounds, no adventitious sounds."
)
_DICTIONARY_PROMPT = (
    f"{ACOUSTIC_DICTIONARY}\n\nListen to this lung auscultation recording and choose the patient's diagnosis "
    f"using the signatures above. {ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_DICTIONARY_COT_PROMPT = (
    f"{ACOUSTIC_DICTIONARY}\n\nListen to this lung auscultation recording. First describe the breathing cycles "
    "and any adventitious sounds, then match your description against the signatures above. "
    f"{ENGLISH_ONLY}\n\n{COT_FORMAT}"
)
_HIERARCHICAL_TURNS = [
    "Listen to this lung auscultation recording. Are there adventitious sounds (crackles or wheezes), "
    f"or is the breathing normal? {ENGLISH_ONLY}",
    "Which type of adventitious sound is it, and where does it fall in the breathing cycle "
    "(early or late inspiration, expiration)?",
    f"Based on your analysis, choose the patient's diagnosis. {ENGLISH_ONLY} {ANSWER_WITH_LETTER}",
]
_FEW_SHOT_PROMPT = (
    "Here are reference lung auscultation recordings with their diagnoses. Listen to them, then choose the "
    f"diagnosis of the final recording. {ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
# Kept only as a record of what r8 asked. These figures have no source, which is why r8 is
# deprecated (see R8_DEPRECATION); do not reuse them.
_PRIOR_PROMPT = (
    "In the reference collection these diagnoses occur at roughly the following rates: COPD 38%, "
    "No potential disease detected 18%, URTI 15%, Pneumonia 13%, Bronchiectasis 10%, Bronchiolitis 5%, "
    "LRTI 1%, Asthma <1%. Listen to this lung auscultation recording and choose the patient's diagnosis. "
    f"{ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_FEATURES_PROMPT = (
    "Measurements taken from this lung auscultation recording by a signal-processing front end are given "
    "below. Use them together with what you hear, then choose the patient's diagnosis. "
    f"{ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_FEATURES_DICTIONARY_PROMPT = (
    f"{ACOUSTIC_DICTIONARY}\n\nMeasurements taken from this recording by a signal-processing front end are "
    "given below. Use them together with what you hear, and match against the signatures above. "
    f"{ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_FEATURES_TEXT_ONLY_PROMPT = (
    f"{ACOUSTIC_DICTIONARY}\n\nYou cannot hear the recording. Measurements taken from it by a signal-processing "
    "front end are given below. Using only those measurements and the signatures above, choose the patient's "
    f"diagnosis. {ANSWER_WITH_LETTER}"
)
_TAGS_PROMPT = (
    "A general-purpose audio tagger was run over this recording; its output is given below. Use it together "
    f"with what you hear, then choose the patient's diagnosis. {ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_TAGS_FEATURES_PROMPT = (
    "An audio tagger and a respiratory signal-processing front end were both run over this recording; their "
    "output is given below. Use them together with what you hear, then choose the patient's diagnosis. "
    f"{ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_CYCLE_PROMPT = (
    "The recording is given as labelled breathing cycles followed by the full recording. Compare the cycles, "
    f"then choose the patient's diagnosis. {ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_LOCALIZATION_TURNS = [
    "Listen to this lung auscultation recording and identify the single breathing cycle that sounds most "
    'abnormal. Do NOT answer the question yet. Respond ONLY with JSON of the form {"start": <seconds>, '
    '"end": <seconds>}.',
    "Here is a cropped segment covering the cycle you identified. Using it together with what you already "
    f"heard, choose the patient's diagnosis. {ENGLISH_ONLY} {ANSWER_WITH_LETTER}",
]
_NEIGHBOR_PROMPT = (
    "The diagnoses of the acoustically most similar reference recordings are given below. Use them as evidence "
    f"together with what you hear, then choose the patient's diagnosis. {ENGLISH_ONLY} {ANSWER_WITH_LETTER}"
)
_TEXT_ONLY_METADATA_PROMPT = (
    "You cannot hear the recording. Using only the recording metadata below, choose the most likely diagnosis. "
    f"You must pick exactly one choice. {ANSWER_WITH_LETTER}"
)

# ---------------------------------------------------------------------------------------------
# Scope. The primary metric's 95% CI is about +-8 points, so every additional arm compared against
# one baseline buys another chance of a spurious "significant" result. The reported set is the
# `core` tier:
#   controls     c1 (silence prior), c3 (silence + metadata), c4 (silence + features)
#   baselines    r0 (metadata-blind), r0_aware (metadata-aware)
#   classic      r2 (CoT), r3 (acoustic dictionary)
#   pre-registered comparisons  (1) a1b vs r0   (2) best of a3a/a3b vs r0   (3) a5c vs a5a
# Everything else is `future_work`: implemented, tested and runnable with --experiment, but not part
# of the reported results. Only r8 is deprecated: as written it was never a valid experiment.
# ---------------------------------------------------------------------------------------------
FUTURE_CLASSIC_VARIANT = "Classic-prompting variant; r2 and r3 already represent this family in the reported set."
FUTURE_COT_FEW_SHOT = (
    "Chain-of-thought variant of r6. CoT is already reported through r2, so this adds a second CoT "
    "comparison without a pre-registered question behind it."
)
FUTURE_FEW_SHOT_AUDIO_LIMIT = (
    "One demonstration per primary class is 6 clips plus the test clip, and Voxtral (in vLLM) accepts at "
    "most 5 audio clips per prompt. Fewer demonstrations would leave classes unshown and bias the balanced "
    "accuracy comparison. Few-shot is reported through MMAR v7 instead (3 demonstrations + test = 4 clips)."
)
FUTURE_CROSS_MODEL_CONTROL = (
    "Text-only control on a different model (Gemma). A gap between it and a Voxtral arm mixes the "
    "contribution of the audio with the difference between two models; the same-model silence "
    "controls c3 and c4 answer the question cleanly."
)
FUTURE_T1_VARIANT = "Measured-feature variant; a1b carries the hypothesis and c4 is its same-model control."
FUTURE_METADATA_AWARE = "Metadata-aware duplicate; r0_aware already quantifies what the recording metadata alone buys."
FUTURE_SECOND_TOOL = (
    "A second evidence source on top of A1. Worth testing once the respiratory detectors have been "
    "validated against real cycle annotations rather than synthetic signals."
)
FUTURE_PRESENTATION = (
    "Alternative audio presentation. Cheap to add later, but it is not one of the three pre-registered comparisons."
)
FUTURE_CALIBRATION_VARIANT = "Calibration variant; a5a and a5c are the two arms of the pre-registered comparison."
FUTURE_RETRIEVAL = (
    "Retrieval needs a candidate pool disjoint from the evaluation set, or the model is shown other "
    "scored items with their labels. In this packaging that pool is only the handful of held-out "
    "demonstrations, too few to retrieve from meaningfully. Needs the original 920-recording "
    "database (see data/README.md)."
)
FUTURE_CONSISTENCY = (
    "Choice-order robustness. A good appendix result, but it multiplies the run by 5 for a "
    "measurement that does not bear on the three pre-registered comparisons."
)
R8_DEPRECATION = (
    "The prompt quoted class frequencies that came from no real source. There is no prior that is "
    "both external to the evaluation set and verified, and quoting the evaluation set's own "
    "frequencies would leak its label prior into the prompt."
)

NUM_ICBHI_LABELS = len(ICBHI_LABELS)
# Budget for arms that ask for a single letter. With the long dictionary/feature prompts Voxtral opens
# with a ~25-token lead-in ("Based on the ... the patient's diagnosis is most likely:") before the
# letter, so a 32-token budget could cut the answer off and score it as wrong. 96 leaves headroom;
# the model stops on its own when it is done, so the cost is negligible.
SINGLE_ANSWER_MAX_TOKENS = 96


class ExperimentVersion(Enum):
    # ---------------------------------------------------------
    # Controls
    # ---------------------------------------------------------
    c1 = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="silence_prior",
        prompt=BASELINE_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        replace_audio_with_silence=True,
    )
    c2 = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="text_only_metadata",
        prompt=_TEXT_ONLY_METADATA_PROMPT,
        max_new_tokens=512,
        batch_size=4,
        text_only=True,
        include_recording_metadata=True,
        tier="future_work",
        future_work_reason=FUTURE_CROSS_MODEL_CONTROL,
    )
    c3 = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="silence_metadata_aware",
        prompt=BASELINE_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        include_recording_metadata=True,
        replace_audio_with_silence=True,
    )
    c4 = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="silence_features_dictionary",
        prompt=_FEATURES_DICTIONARY_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        inject_acoustic_features=True,
        replace_audio_with_silence=True,
    )

    # ---------------------------------------------------------
    # R0-R8: classic text prompting, metadata-blind
    # ---------------------------------------------------------
    r0 = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="baseline",
        prompt=BASELINE_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
    )
    r0_aware = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="baseline_metadata_aware",
        prompt=BASELINE_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        include_recording_metadata=True,
    )
    r1 = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="role_pulmonologist",
        prompt=_ROLE_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        tier="future_work",
        future_work_reason=FUTURE_CLASSIC_VARIANT,
    )
    r2 = ExperimentMeta(
        dataset=ICBHI, experiment_name="cot", prompt=_COT_PROMPT, max_new_tokens=512, batch_size=8, use_cot=True
    )
    r3 = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="acoustic_dictionary",
        prompt=_DICTIONARY_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
    )
    r4 = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="dictionary_cot",
        prompt=_DICTIONARY_COT_PROMPT,
        max_new_tokens=512,
        batch_size=8,
        use_cot=True,
        tier="future_work",
        future_work_reason=FUTURE_CLASSIC_VARIANT,
    )
    r5 = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="hierarchical_decomposition",
        prompt=_HIERARCHICAL_TURNS,
        max_new_tokens=256,
        batch_size=2,
        tier="future_work",
        future_work_reason=FUTURE_CLASSIC_VARIANT,
    )
    r6 = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="few_shot_audio_balanced",
        prompt=_FEW_SHOT_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=4,
        few_shot_mode="audio",
        tier="future_work",
        future_work_reason=FUTURE_FEW_SHOT_AUDIO_LIMIT,
    )
    r7 = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="few_shot_audio_cot",
        prompt=f"{_FEW_SHOT_PROMPT}\n\n{COT_FORMAT}",
        max_new_tokens=512,
        batch_size=2,
        few_shot_mode="audio",
        use_cot=True,
        tier="future_work",
        future_work_reason=FUTURE_COT_FEW_SHOT,
    )
    r8 = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="prior_informed",
        prompt=_PRIOR_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        deprecated=True,
        deprecation_reason=R8_DEPRECATION,
    )

    # ---------------------------------------------------------
    # A1: measured respiratory features as text
    # ---------------------------------------------------------
    a1a = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="audio_features",
        prompt=_FEATURES_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        inject_acoustic_features=True,
        tier="future_work",
        future_work_reason=FUTURE_T1_VARIANT,
    )
    a1b = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="audio_features_dictionary",
        prompt=_FEATURES_DICTIONARY_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        inject_acoustic_features=True,
    )
    a1c = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="text_only_features_dictionary",
        prompt=_FEATURES_TEXT_ONLY_PROMPT,
        max_new_tokens=512,
        batch_size=4,
        text_only=True,
        inject_acoustic_features=True,
        tier="future_work",
        future_work_reason=FUTURE_CROSS_MODEL_CONTROL,
    )
    a1d = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="audio_features_dictionary_metadata_aware",
        prompt=_FEATURES_DICTIONARY_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        inject_acoustic_features=True,
        include_recording_metadata=True,
        tier="future_work",
        future_work_reason=FUTURE_METADATA_AWARE,
    )

    # ---------------------------------------------------------
    # A2: general audio tagger as a tool
    # ---------------------------------------------------------
    a2a = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="audio_tags",
        prompt=_TAGS_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        inject_audio_tags=True,
        tier="future_work",
        future_work_reason=FUTURE_SECOND_TOOL,
    )
    a2b = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="audio_tags_features",
        prompt=_TAGS_FEATURES_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        inject_audio_tags=True,
        inject_acoustic_features=True,
        tier="future_work",
        future_work_reason=FUTURE_SECOND_TOOL,
    )

    # ---------------------------------------------------------
    # A3: signal preprocessing before the ALM
    # ---------------------------------------------------------
    a3a = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="bandpass_normalize",
        prompt=BASELINE_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        preprocessing="bandpass_normalize",
    )
    a3b = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="bandpass_normalize_shift",
        prompt=BASELINE_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        preprocessing="bandpass_normalize_shift",
    )
    a3c = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="bandpass_normalize_features",
        prompt=_FEATURES_DICTIONARY_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        preprocessing="bandpass_normalize",
        inject_acoustic_features=True,
        tier="future_work",
        future_work_reason=FUTURE_T1_VARIANT,
    )

    # ---------------------------------------------------------
    # A4: breathing-cycle presentation
    # ---------------------------------------------------------
    a4a = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="cycle_presentation",
        prompt=_CYCLE_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=4,
        presentation="cycles",
        tier="future_work",
        future_work_reason=FUTURE_PRESENTATION,
    )
    a4b = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="two_pass_abnormal_cycle",
        prompt=_LOCALIZATION_TURNS,
        max_new_tokens=64,
        batch_size=4,
        presentation="two_pass_localization",
        tier="future_work",
        future_work_reason=FUTURE_PRESENTATION,
    )

    # ---------------------------------------------------------
    # A5: calibrated / contrastive choice scoring
    # ---------------------------------------------------------
    a5a = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="letter_scoring_alpha_0_0",
        prompt=BASELINE_PROMPT,
        max_new_tokens=1,
        batch_size=4,
        contrastive_alpha=0.0,
    )
    a5b = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="contrastive_silence_alpha_1_0",
        prompt=BASELINE_PROMPT,
        max_new_tokens=1,
        batch_size=4,
        contrastive_alpha=1.0,
        calibration="silence",
        tier="future_work",
        future_work_reason=FUTURE_CALIBRATION_VARIANT,
    )
    a5c = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="contrastive_content_free",
        prompt=BASELINE_PROMPT,
        max_new_tokens=1,
        batch_size=4,
        contrastive_alpha=1.0,
        calibration="content_free",
    )
    a5d = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="contrastive_content_free_features",
        prompt=_FEATURES_DICTIONARY_PROMPT,
        max_new_tokens=1,
        batch_size=4,
        contrastive_alpha=1.0,
        calibration="content_free",
        inject_acoustic_features=True,
        tier="future_work",
        future_work_reason=FUTURE_CALIBRATION_VARIANT,
    )

    # ---------------------------------------------------------
    # A6: audio-embedding kNN evidence
    # ---------------------------------------------------------
    a6a = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="knn_text_evidence",
        prompt=_NEIGHBOR_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        inject_neighbor_labels=True,
        tier="future_work",
        future_work_reason=FUTURE_RETRIEVAL,
    )
    a6b = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="knn_audio_few_shot",
        prompt=_FEW_SHOT_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=2,
        few_shot_mode="rag",
        rag_mapping_file=ICBHI_RAG_MAPPING_FILE,
        tier="future_work",
        future_work_reason=FUTURE_RETRIEVAL,
    )

    # ---------------------------------------------------------
    # A7: multi-view self-consistency
    # ---------------------------------------------------------
    a7a = ExperimentMeta(
        dataset=ICBHI,
        experiment_name="choice_shuffling",
        prompt=BASELINE_PROMPT,
        max_new_tokens=SINGLE_ANSWER_MAX_TOKENS,
        batch_size=8,
        num_shuffled_variants=5,
        tier="future_work",
        future_work_reason=FUTURE_CONSISTENCY,
    )

    @classmethod
    def get_version(cls, version_str: str) -> "ExperimentVersion":
        try:
            version = cls[version_str.lower()]
        except KeyError as e:
            raise ValueError(
                f"Unknown experiment version: {version_str}. Available versions: {[m.name for m in cls]}"
            ) from e

        meta = version.value
        if meta.deprecated:
            raise DeprecatedExperimentError(
                f"{version.name} ({meta.experiment_name}) is deprecated: {meta.deprecation_reason}"
            )
        if meta.is_future_work:
            logger.warning(
                f"{version.name} ({meta.experiment_name}) is tier 'future_work' and sits outside the "
                f"reported set: {meta.future_work_reason}"
            )
        return version


MAX_RAG_SHOTS = MAX_EXTRA_AUDIOS_PER_PROMPT


def _load_icbhi_rag_mapping(path: str) -> dict[str, list[str]]:
    if not os.path.exists(path):
        raise DatasetIntegrityError(f"kNN mapping file not found: {path}. Run scripts/build_icbhi_knn_index.py first.")
    with open(path) as f:
        return json.load(f)


def attach_few_shots(
    items: list[tuple[BaseRequest, str]], config: AppConfig, meta: ExperimentMeta
) -> list[tuple[BaseRequest, str]]:
    """Attaches demonstrations drawn only from the held-out split, never from the evaluation set.

    Patient ids are absent from this packaging, so a demonstration may still share a patient with an
    evaluation item; few-shot numbers carry that caveat. See data/README.md.
    """
    if meta.few_shot_mode == "none":
        return items

    eval_ids = {req.metadata.item_id for req, _ in items if req.metadata}

    if meta.few_shot_mode == "rag":
        mapping = _load_icbhi_rag_mapping(meta.rag_mapping_file)
        retrieved = {shot_id for item_id in eval_ids for shot_id in mapping.get(item_id, [])}
        # Fail on a mapping that retrieves evaluation items rather than quietly dropping them: a stale
        # kNN file built before the held-out split existed would otherwise just yield fewer shots.
        validate_demonstration_disjointness(eval_ids, retrieved)

        pool = build_icbhi_neighbor_pool(config.dataset, meta)
        attached = []
        for req, ground_truth in items:
            item_id = req.metadata.item_id if req.metadata else ""
            shot_ids = [sid for sid in mapping.get(item_id, []) if sid in pool][:MAX_RAG_SHOTS]
            turns = [pool[sid] for sid in shot_ids]
            logger.info(f"Item {item_id}: attached {len(turns)} retrieved demonstrations.")
            attached.append((req.model_copy(update={"few_shot_turns": turns}), ground_truth))
        return attached

    pool = build_icbhi_few_shot_pool(config.dataset, meta)
    validate_demonstration_disjointness(eval_ids, set(pool))
    turns = list(pool.values())
    return [(req.model_copy(update={"few_shot_turns": turns}), ground_truth) for req, ground_truth in items]


def _cycle_spans_by_path(config: AppConfig, items: list[tuple[BaseRequest, str]]) -> dict[str, list]:
    evidence: dict[str, RespiratoryEvidence] = load_respiratory_evidence(config.dataset.respiratory_features_file)
    spans: dict[str, list] = {}
    for req, _ in items:
        record = evidence.get(req.metadata.item_id) if req.metadata else None
        if record and record.cycle_spans:
            spans[req.audio_path] = record.cycle_spans
    return spans


def build_engine(
    config: AppConfig, meta: ExperimentMeta, items: list[tuple[BaseRequest, str]]
) -> GemmaTextModel | VoxtralAudioEngine:
    if meta.text_only:
        return GemmaTextModel(config.text_model)

    audio_config = config.audio_model
    cycle_spans = _cycle_spans_by_path(config, items) if meta.needs_cycle_spans else None
    scorer = (
        ContrastiveChoiceScorer(
            meta.contrastive_alpha, audio_config.target_sr, meta.batch_size, calibration=meta.calibration
        )
        if meta.contrastive_alpha is not None
        else None
    )
    return VoxtralAudioEngine(
        audio_config,
        presentation=build_presentation(meta, audio_config.target_sr, cycle_spans),
        scorer=scorer,
        preprocessor=build_preprocessor(meta, audio_config.target_sr),
    )


def build_aggregator(meta: ExperimentMeta) -> ResponseAggregator:
    if meta.num_shuffled_variants > 1:
        return ShuffledMajorityVoteAggregator()
    return IdentityAggregator()


SAMPLE_IDS_REJECTED = (
    "ICBHI V4 evaluates a fixed set - every row except the held-out demonstrations - identically for "
    "every experiment, so "
    "--sample-ids-file is not honoured: passing it would silently evaluate a different set than "
    "you asked for. That is exactly what made the V1 and V3 numbers incomparable. Drop the flag, "
    "or use --num-samples for a smoke test."
)


def run_icbhi(args):
    if getattr(args, "sample_ids_file", None):
        raise DatasetIntegrityError(SAMPLE_IDS_REJECTED)

    meta = ExperimentVersion.get_version(args.experiment).value
    config = create_icbhi_config(args, meta)

    num_runs = args.runs
    if meta.is_deterministic and num_runs > 1:
        logger.info(f"{meta.experiment_name} is deterministic; collapsing {num_runs} requested runs to 1.")
        num_runs = 1

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.output_dir, f"icbhi_{meta.experiment_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger.info("--- [PHASE 1] DATA LOADING & PREPARATION ---")
    items = load_icbhi_requests(config.dataset, meta)
    items = attach_few_shots(items, config, meta)
    requests = [req for req, _ in items]
    ground_truths = [gt for _, gt in items]

    logger.info("--- [PHASE 2] INITIALIZING INFERENCE ENGINE ---")
    log_gpu_memory()
    inference_pipeline = InferencePipeline(build_engine(config, meta, items), build_aggregator(meta))

    logger.info(f"--- [PHASE 3] RUNNING INFERENCE ({num_runs} RUNS) ---")
    temp_files, final_outputs, metrics_outputs = [], [], []
    for i in range(num_runs):
        suffix = f"_run{i + 1}" if num_runs > 1 else ""
        temp_files.append(os.path.join(run_dir, f"temp_audio_preds{suffix}.jsonl"))
        final_outputs.append(os.path.join(run_dir, f"raw_output{suffix}.jsonl"))
        metrics_outputs.append(os.path.join(run_dir, f"metrics{suffix}.txt"))
        inference_pipeline.run(requests, ground_truths, temp_files[i])

    inference_pipeline = None
    release_vram()

    if getattr(args, "no_judge", False):
        logger.info("--- [PHASE 4] JUDGE SKIPPED: the parser alone scores the closed label set ---")
        evaluation_pipeline = ParserOnlyPipeline(metrics_calculator=calculate_icbhi_metrics)
    else:
        logger.info(f"Waiting {JUDGE_LOAD_DELAY_S} seconds before loading Judge...")
        time.sleep(JUDGE_LOAD_DELAY_S)

        logger.info("--- [PHASE 4] INITIALIZING JUDGE ENGINE ---")
        evaluation_pipeline = JudgePipeline(
            QwenJudge(
                config.judge,
                template_func=build_icbhi_choice_judge_conversation,
                schema_class=ChoiceExtractionResult,
            ),
            metrics_calculator=calculate_icbhi_metrics,
        )

    logger.info(f"--- [PHASE 5] RUNNING EVALUATION ({num_runs} RUNS) ---")
    all_metrics = [
        evaluation_pipeline.run(temp_files[i], final_outputs[i], metrics_outputs[i], i) for i in range(num_runs)
    ]
    evaluation_pipeline = None
    release_vram()

    if num_runs > 1:
        logger.info("--- [PHASE 6] CALCULATING STABILITY ---")
        stability_file = os.path.join(run_dir, "stability_report.json")
        with open(stability_file, "w") as f:
            json.dump(calculate_stability(all_metrics), f, indent=4)
        logger.info(f"Saved statistical stability report to {stability_file}")


if __name__ == "__main__":
    run_icbhi(parse_args("Run ICBHI Speech Processing Pipeline"))
