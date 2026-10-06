import json

from loguru import logger

from speech_processing.data.dtos.requests import BaseRequest
from speech_processing.models.base import BaseAudioModel, BaseTextModel
from speech_processing.pipelines.aggregation import IdentityAggregator, ResponseAggregator


class InferencePipeline:
    def __init__(
        self,
        engine: BaseAudioModel | BaseTextModel,
        aggregator: ResponseAggregator | None = None,
    ) -> None:
        self.engine = engine
        self.aggregator = aggregator or IdentityAggregator()

    def run(self, requests: list[BaseRequest], ground_truths: list[str], output_path: str):
        logger.info(f"Running Inference Pipeline on {len(requests)} samples...")
        responses = self.engine.batch_infer(requests)

        triplets = self.aggregator.aggregate(list(zip(requests, responses, ground_truths, strict=True)))

        with open(output_path, "w") as f:
            for req, resp, ground_truth in triplets:
                record = {
                    "sample_id": resp.sample_id,
                    "instruction": resp.instruction,
                    "generated_text": resp.generated_text,
                    "final_turn_text": resp.final_turn_text,
                    "ground_truth": ground_truth,
                    # The response's metadata is the request's plus whatever the engine added - e.g. the
                    # contrastive scorer's choice_scores - so writing the request's would drop it.
                    "metadata": metadata.model_dump() if (metadata := resp.metadata or req.metadata) else None,
                }
                f.write(json.dumps(record) + "\n")

        return [resp for _, resp, _ in triplets]
