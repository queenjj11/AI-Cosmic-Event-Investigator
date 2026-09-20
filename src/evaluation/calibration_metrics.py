"""Calibration evaluation metrics: Expected Calibration Error (ECE), Brier Score, and Reliability Diagrams."""

from typing import Any, Dict, List, Tuple
import numpy as np


def compute_calibration_metrics(confidences: np.ndarray,
                                correctness: np.ndarray,
                                num_bins: int = 10) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE), Brier Score, and reliability diagram binning.
    confidences: (N,) predicted probabilities in [0, 1]
    correctness: (N,) binary ground-truth correctness indicators (1 or 0)
    """
    confidences = np.clip(confidences, 0.0, 1.0)
    correctness = np.array(correctness, dtype=np.float32)

    # 1. Brier score: MSE between confidence and binary outcome
    brier_score = float(np.mean((confidences - correctness)**2))

    # 2. Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    bin_data: List[Dict[str, float]] = []

    N = len(confidences)
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_acc = float(np.mean(correctness[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_count / N) * np.abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = (bin_lower + bin_upper) / 2.0

        bin_data.append({
            "bin_index": i,
            "bin_lower": float(bin_lower),
            "bin_upper": float(bin_upper),
            "confidence": bin_conf,
            "accuracy": bin_acc,
            "count": bin_count
        })

    # Maximum Calibration Error (MCE)
    diffs = [abs(b["accuracy"] - b["confidence"]) for b in bin_data if b["count"] > 0]
    mce = float(max(diffs)) if diffs else 0.0

    return {
        "ece": float(ece),
        "mce": mce,
        "brier_score": brier_score,
        "reliability_bins": bin_data
    }
