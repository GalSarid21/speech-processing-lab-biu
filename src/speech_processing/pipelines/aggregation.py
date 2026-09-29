import abc
from collections import Counter, defaultdict

from loguru import logger

from speech_processing.data.choices import index_for_letter, letter_for_index
from speech_processing.data.dtos import BaseRequest, BaseResponse
from speech_processing.evaluation.answer_parsing import parse_choice_letter

Triplet = tuple[BaseRequest, BaseResponse, str]


class ResponseAggregator(abc.ABC):
    """Collapses the raw per-request results into the rows that get scored."""

    @abc.abstractmethod
    def aggregate(self, triplets: list[Triplet]) -> list[Triplet]:
        """Return the triplets to persist and score."""


class IdentityAggregator(ResponseAggregator):
    def aggregate(self, triplets: list[Triplet]) -> list[Triplet]:
        return triplets


class ShuffledMajorityVoteAggregator(ResponseAggregator):
    """T4: votes across choice-order variants, mapped back onto the original ordering."""

    def aggregate(self, triplets: list[Triplet]) -> list[Triplet]:
        groups: dict[str | None, list[Triplet]] = defaultdict(list)
        for triplet in triplets:
            metadata = triplet[0].metadata
            groups[metadata.original_item_id if metadata else None].append(triplet)

        aggregated: list[Triplet] = []
        for original_item_id, group in groups.items():
            ordered = sorted(group, key=lambda t: t[0].metadata.variant_idx if t[0].metadata else 0)
            votes = [vote for vote in (self._vote(req, resp) for req, resp, _ in ordered) if vote]

            # Counter preserves insertion order, so a tie resolves to the earliest variant, i.e. the
            # unshuffled order of variant 0.
            winner = Counter(votes).most_common(1)[0][0] if votes else ""
            if not votes:
                logger.warning(f"No variant of {original_item_id} produced a parsable letter.")

            base_req, base_resp, base_gt = ordered[0]
            metadata = base_req.metadata.model_copy(update={"votes": votes}) if base_req.metadata else None
            aggregated.append(
                (
                    base_req.model_copy(update={"metadata": metadata}),
                    base_resp.model_copy(
                        update={"generated_text": winner, "final_turn_text": winner, "metadata": metadata}
                    ),
                    base_gt,
                )
            )

        return aggregated

    @staticmethod
    def _vote(req: BaseRequest, resp: BaseResponse) -> str | None:
        metadata = req.metadata
        if metadata is None:
            return None

        letter = parse_choice_letter(resp.final_turn_text or resp.generated_text, metadata.choices)
        if letter is None:
            return None
        if metadata.permutation is None:
            return letter
        return letter_for_index(metadata.permutation[index_for_letter(letter)])


class MultiViewMajorityVoteAggregator(ResponseAggregator):
    """A7b: votes across several presentations of the SAME item within one batch.

    Requests are grouped by `metadata.item_id`, so an experiment that submits the same recording
    under two or more views (original, band-passed, per-cycle) collapses to one voted answer.
    Unlike the shuffled aggregator there is no permutation to undo: every view shows the same
    choice order.
    """

    def aggregate(self, triplets: list[Triplet]) -> list[Triplet]:
        groups: dict[str | None, list[Triplet]] = defaultdict(list)
        for triplet in triplets:
            metadata = triplet[0].metadata
            groups[metadata.item_id if metadata else None].append(triplet)

        aggregated: list[Triplet] = []
        for item_id, group in groups.items():
            votes = [vote for vote in (self._vote(req, resp) for req, resp, _ in group) if vote]
            winner = Counter(votes).most_common(1)[0][0] if votes else ""
            if not votes:
                logger.warning(f"No view of {item_id} produced a parsable letter.")

            base_req, base_resp, base_gt = group[0]
            metadata = base_req.metadata.model_copy(update={"votes": votes}) if base_req.metadata else None
            aggregated.append(
                (
                    base_req.model_copy(update={"metadata": metadata}),
                    base_resp.model_copy(
                        update={"generated_text": winner, "final_turn_text": winner, "metadata": metadata}
                    ),
                    base_gt,
                )
            )

        return aggregated

    @staticmethod
    def _vote(req: BaseRequest, resp: BaseResponse) -> str | None:
        if req.metadata is None:
            return None
        return parse_choice_letter(resp.final_turn_text or resp.generated_text, req.metadata.choices)
