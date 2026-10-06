import json

import pytest
from assertpy import assert_that

from speech_processing.data.dtos import AudioRequest, AudioResponse, ItemMetadata, JudgeRequest
from speech_processing.models.base import BaseAudioModel
from speech_processing.pipelines.aggregation import ShuffledMajorityVoteAggregator
from speech_processing.pipelines.inference import InferencePipeline

CHOICES = ["Dog", "Cat"]


class FakeEngine(BaseAudioModel):
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def batch_infer(self, requests: list[AudioRequest]) -> list[AudioResponse]:
        return [
            AudioResponse(
                sample_id=req.audio_path,
                instruction=req.instruction,
                generated_text=f"reasoning ...\n{output}",
                final_turn_text=output,
                metadata=req.metadata,
            )
            for req, output in zip(requests, self.outputs, strict=False)
        ]


def build_request(variant_idx: int | None = None) -> AudioRequest:
    metadata = ItemMetadata(item_id="ID1", question="q", choices=CHOICES)
    if variant_idx is not None:
        metadata = metadata.model_copy(
            update={
                "original_item_id": "ID1",
                "variant_idx": variant_idx,
                "permutation": [0, 1],
                "original_choices": CHOICES,
            }
        )
    return AudioRequest(instruction="Answer.", audio_path="a.wav", metadata=metadata)


def read_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_jsonl_round_trips_into_typed_judge_requests(tmp_path):
    output_path = tmp_path / "out.jsonl"
    pipeline = InferencePipeline(FakeEngine(["B"]))

    pipeline.run([build_request()], ["B"], str(output_path))

    records = read_lines(output_path)
    assert_that(records).is_length(1)
    assert_that(records[0]["final_turn_text"]).is_equal_to("B")
    assert_that(records[0]["metadata"]["item_id"]).is_equal_to("ID1")

    judge_request = JudgeRequest.from_json(json.dumps(records[0]))
    assert_that(judge_request.metadata).is_instance_of(ItemMetadata)
    assert_that(judge_request.metadata.choices).is_equal_to(CHOICES)


def test_shuffled_variants_collapse_to_one_line(tmp_path):
    output_path = tmp_path / "out.jsonl"
    pipeline = InferencePipeline(FakeEngine(["B", "B"]), ShuffledMajorityVoteAggregator())

    pipeline.run([build_request(0), build_request(1)], ["B", "B"], str(output_path))

    records = read_lines(output_path)
    assert_that(records).is_length(1)
    assert_that(records[0]["generated_text"]).is_equal_to("B")


def test_mismatched_response_count_raises(tmp_path):
    pipeline = InferencePipeline(FakeEngine(["B"]))
    with pytest.raises(ValueError):
        pipeline.run([build_request(), build_request()], ["B", "B"], str(tmp_path / "out.jsonl"))


class ScoringEngine(BaseAudioModel):
    """Adds a response-only metadata field, as the contrastive scorer does with choice_scores."""

    def batch_infer(self, requests: list[AudioRequest]) -> list[AudioResponse]:
        return [
            AudioResponse(
                sample_id=req.audio_path,
                instruction=req.instruction,
                generated_text="A",
                final_turn_text="A",
                metadata=req.metadata.model_copy(update={"choice_scores": {"A": 0.5, "B": -0.5}}),
            )
            for req in requests
        ]


def test_metadata_the_engine_adds_reaches_the_jsonl(tmp_path):
    """Writing the request's metadata used to drop the contrastive scores the engine had computed."""
    output_path = tmp_path / "out.jsonl"

    InferencePipeline(ScoringEngine()).run([build_request()], ["A"], str(output_path))

    assert_that(read_lines(output_path)[0]["metadata"]["choice_scores"]).is_equal_to({"A": 0.5, "B": -0.5})
