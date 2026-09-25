import json
from loguru import logger
from speech_processing.data.dtos.requests import BaseRequest

class InferencePipeline:
    def __init__(self, engine):
        self.engine = engine

    def run(self, requests: list[BaseRequest], ground_truths: list[str], output_path: str):
        logger.info(f"Running Inference Pipeline on {len(requests)} samples...")
        responses = self.engine.batch_infer(requests)
        
        with open(output_path, "w") as f:
            f.writelines(
                json.dumps(
                    {
                        "sample_id": resp.sample_id,
                        "instruction": resp.instruction,
                        "generated_text": resp.generated_text,
                        "ground_truth": gt,
                    }
                ) + "\n"
                for resp, gt in zip(responses, ground_truths)
            )
        return responses
