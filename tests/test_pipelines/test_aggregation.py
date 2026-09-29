from assertpy import assert_that

from speech_processing.data.dtos import AudioRequest, AudioResponse, ItemMetadata
from speech_processing.pipelines.aggregation import ShuffledMajorityVoteAggregator

ORIGINAL_CHOICES = ["Dog", "Cat", "Bird"]
# permutation[displayed_index] = original_index
PERMUTATIONS = [[0, 1, 2], [1, 0, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1]]
# Displayed letters the model produced. Variant 2 is unparsable.
OUTPUTS = ["B", "A", "nothing usable", "B", "C"]


def build_triplet(variant_idx: int, output: str, ground_truth: str = "B"):
    permutation = PERMUTATIONS[variant_idx]
    metadata = ItemMetadata(
        item_id="ID1",
        question="q",
        choices=[ORIGINAL_CHOICES[i] for i in permutation],
        original_item_id="ID1",
        variant_idx=variant_idx,
        permutation=permutation,
        original_choices=ORIGINAL_CHOICES,
    )
    request = AudioRequest(instruction=f"instruction for variant {variant_idx}", audio_path="a.wav", metadata=metadata)
    response = AudioResponse(
        sample_id="a.wav",
        instruction=request.instruction,
        generated_text=output,
        final_turn_text=output,
        metadata=metadata,
    )
    return request, response, ground_truth


def test_majority_vote_maps_back_to_the_original_order():
    triplets = [build_triplet(i, output) for i, output in enumerate(OUTPUTS)]
    aggregated = ShuffledMajorityVoteAggregator().aggregate(triplets)

    assert_that(aggregated).is_length(1)
    request, response, ground_truth = aggregated[0]

    assert_that(response.generated_text).is_equal_to("B")
    assert_that(response.final_turn_text).is_equal_to("B")
    assert_that(ground_truth).is_equal_to("B")
    assert_that(request.instruction).is_equal_to("instruction for variant 0")
    assert_that(request.metadata.votes).is_equal_to(["B", "B", "C", "B"])


def test_ties_resolve_to_variant_zero():
    # Variant 0 displays the original order, so "A" maps to A; variant 1 swaps A and B, so "A" maps to B.
    triplets = [build_triplet(0, "A"), build_triplet(1, "A")]
    _, response, _ = ShuffledMajorityVoteAggregator().aggregate(triplets)[0]

    assert_that(response.generated_text).is_equal_to("A")


def test_no_parsable_variant_yields_an_empty_answer():
    triplets = [build_triplet(0, "???"), build_triplet(1, "???")]
    _, response, _ = ShuffledMajorityVoteAggregator().aggregate(triplets)[0]

    assert_that(response.generated_text).is_empty()
