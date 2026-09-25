import json
import os
from loguru import logger
from speech_processing.data.dtos.requests import JudgeRequest, JudgeRequestParseError
from speech_processing.models.judge import QwenJudge
from speech_processing.evaluation.metrics import calculate_metrics

class JudgePipeline:
    def __init__(self, engine: QwenJudge):
        self.engine = engine

    def _generate_report(self, metrics, run_idx: int) -> str:
        return (
            f"--- AGGREGATE METRICS (Run {run_idx + 1}) ---\n"
            f"Avg Acoustic Accuracy:   {metrics.avg_acoustic_pct:.2f}%\n"
            f"Avg Diagnostic Accuracy: {metrics.avg_diagnostic_pct:.2f}%\n"
            f"Hallucination Rate:      {metrics.hallucination_rate_pct:.2f}%\n\n"
            f"Classification Report:\n"
            f"{metrics.classification_report}\n"
        )

    def run(self, input_path: str, output_path: str, metrics_path: str, run_idx: int):
        logger.info(f"Running Judge Pipeline (Run {run_idx + 1})...")
        
        judge_requests = []
        if os.path.exists(input_path):
            with open(input_path, "r") as f:
                for line in f:
                    try:
                        judge_requests.append(JudgeRequest.from_json(line))
                    except JudgeRequestParseError as e:
                        logger.error(e)

        if not judge_requests:
            logger.warning(f"No valid JudgeRequests found in {input_path}.")
            return None

        evaluations = self.engine.batch_evaluate(judge_requests)

        with open(output_path, "w") as f:
            f.writelines(eval_resp.model_dump_json() + "\n" for eval_resp in evaluations)

        metrics = calculate_metrics(evaluations)
        
        if metrics:
            report = self._generate_report(metrics, run_idx)
            with open(metrics_path, "w") as f:
                f.write(report)
            logger.info(f"Saved metrics to {metrics_path}")
            
        return metrics
