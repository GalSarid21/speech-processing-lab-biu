import numpy as np

from speech_processing.data.dtos import AggregateMetrics


def calculate_stability(metrics_list: list[AggregateMetrics]) -> dict:
    """Calculates statistical features across N runs for each core metric."""
    
    acoustic = [m.avg_acoustic_pct for m in metrics_list if m is not None]
    diagnostic = [m.avg_diagnostic_pct for m in metrics_list if m is not None]
    hallucination = [m.hallucination_rate_pct for m in metrics_list if m is not None]
    
    def get_stats(arr):
        if not arr:
            return {}
        mean = np.mean(arr)
        std = np.std(arr)
        # Coefficient of Variation (CV) = std / mean. (0% means perfectly stable)
        cv = (std / mean * 100) if mean != 0 else 0.0
        return {
            "mean": round(float(mean), 2),
            "median": round(float(np.median(arr)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
            "p25": round(float(np.percentile(arr, 25)), 2),
            "p75": round(float(np.percentile(arr, 75)), 2),
            "variance": round(float(np.var(arr)), 2),
            "std_dev": round(float(std), 2),
            "coefficient_of_variation_pct": round(float(cv), 2)
        }
        
    return {
        "runs": len(acoustic),
        "acoustic_accuracy": get_stats(acoustic),
        "diagnostic_accuracy": get_stats(diagnostic),
        "hallucination_rate": get_stats(hallucination)
    }
