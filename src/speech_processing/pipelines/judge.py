import abc
import os

from loguru import logger

from speech_processing.data.dtos.requests import JudgeRequest, JudgeRequestParseError
from speech_processing.data.dtos.responses import ChoiceExtractionResult, JudgeResponse
from speech_processing.evaluation.metrics import MetricsCalculator, calculate_metrics
from speech_processing.models.base import BaseJudgeModel

CLASSIFICATION_REPORT_FIELD = "classification_report"


def _read_judge_requests(input_path: str) -> list[JudgeRequest]:
    requests: list[JudgeRequest] = []
    if not os.path.exists(input_path):
        return requests

    with open(input_path) as f:
        for line in f:
            try:
                requests.append(JudgeRequest.from_json(line))
            except JudgeRequestParseError as e:
                logger.error(e)
    return requests


class EvaluationPipeline(abc.ABC):
    """Turns one inference JSONL into scored responses plus an aggregate metrics report."""

    def __init__(self, metrics_calculator: MetricsCalculator = calculate_metrics) -> None:
        self.metrics_calculator = metrics_calculator

    @abc.abstractmethod
    def _evaluate(self, requests: list[JudgeRequest]) -> list[JudgeResponse]:
        """Attach an evaluation to every request."""

    def _generate_report(self, metrics, run_idx: int) -> str:
        lines = [f"--- AGGREGATE METRICS (Run {run_idx + 1}) ---"]
        payload = metrics.model_dump()
        report = payload.pop(CLASSIFICATION_REPORT_FIELD, None)

        for field, value in payload.items():
            rendered = f"{value:.2f}" if isinstance(value, float) else str(value)
            lines.append(f"{field}: {rendered}")

        if report is not None:
            lines.append("")
            lines.append("Classification Report:")
            lines.append(report)

        return "\n".join(lines) + "\n"

    def run(self, input_path: str, output_path: str, metrics_path: str, run_idx: int):
        logger.info(f"Running {type(self).__name__} (Run {run_idx + 1})...")

        judge_requests = _read_judge_requests(input_path)
        if not judge_requests:
            logger.warning(f"No valid JudgeRequests found in {input_path}.")
            return None

        evaluations = self._evaluate(judge_requests)

        with open(output_path, "w") as f:
            f.writelines(eval_resp.model_dump_json() + "\n" for eval_resp in evaluations)

        metrics = self.metrics_calculator(evaluations)

        if metrics:
            with open(metrics_path, "w") as f:
                f.write(self._generate_report(metrics, run_idx))
            logger.info(f"Saved metrics to {metrics_path}")

        return metrics


class JudgePipeline(EvaluationPipeline):
    def __init__(
        self,
        engine: BaseJudgeModel,
        metrics_calculator: MetricsCalculator = calculate_metrics,
    ) -> None:
        super().__init__(metrics_calculator)
        self.engine = engine

    def _evaluate(self, requests: list[JudgeRequest]) -> list[JudgeResponse]:
        return self.engine.batch_evaluate(requests)


class ParserOnlyPipeline(EvaluationPipeline):
    """Scores a closed label set without loading a judge at all.

    On a lettered multiple-choice benchmark the deterministic parser already is the metric, and a
    judge only adds a second model's noise. The evaluation field is filled with the fallback so the
    output schema stays identical to a judged run.
    """

    def _evaluate(self, requests: list[JudgeRequest]) -> list[JudgeResponse]:
        return [
            JudgeResponse(
                sample_id=req.sample_id,
                instruction=req.instruction,
                generated_text=req.generated_text,
                final_turn_text=req.final_turn_text,
                ground_truth=req.ground_truth,
                evaluation=ChoiceExtractionResult.fallback(),
                metadata=req.metadata,
            )
            for req in requests
        ]
