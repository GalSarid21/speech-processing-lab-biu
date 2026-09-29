import numpy as np
from pydantic import BaseModel

ROUND_DIGITS = 2

# Legacy report keys, kept so older stability_report.json files stay comparable.
LEGACY_FIELD_ALIASES = {
    "avg_acoustic_pct": "acoustic_accuracy",
    "avg_diagnostic_pct": "diagnostic_accuracy",
    "hallucination_rate_pct": "hallucination_rate",
}


def _get_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    mean = float(np.mean(values))
    std = float(np.std(values))
    # Coefficient of Variation (CV) = std / mean. (0% means perfectly stable)
    cv = (std / mean * 100) if mean != 0 else 0.0
    return {
        "mean": round(mean, ROUND_DIGITS),
        "median": round(float(np.median(values)), ROUND_DIGITS),
        "min": round(float(np.min(values)), ROUND_DIGITS),
        "max": round(float(np.max(values)), ROUND_DIGITS),
        "p25": round(float(np.percentile(values, 25)), ROUND_DIGITS),
        "p75": round(float(np.percentile(values, 75)), ROUND_DIGITS),
        "variance": round(float(np.var(values)), ROUND_DIGITS),
        "std_dev": round(std, ROUND_DIGITS),
        "coefficient_of_variation_pct": round(cv, ROUND_DIGITS),
    }


def calculate_stability(metrics_list: list[BaseModel | None]) -> dict:
    """Statistics across N repeated runs of one experiment.

    This measures temperature noise only: the requests, the prompt and the model are identical
    across runs, so the spread here says nothing about generalisation.
    """
    present = [m for m in metrics_list if m is not None]
    if not present:
        return {"runs": 0}

    report: dict = {"runs": len(present)}
    for field, value in present[0].model_dump().items():
        if not isinstance(value, int | float) or isinstance(value, bool):
            continue
        values = [float(getattr(m, field)) for m in present]
        report[LEGACY_FIELD_ALIASES.get(field, field)] = _get_stats(values)

    return report
