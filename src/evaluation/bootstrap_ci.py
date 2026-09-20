"""Non-parametric bootstrap confidence intervals for scientific rigor."""

from typing import Any, Callable, Dict, Tuple
import numpy as np


def bootstrap_confidence_interval(metric_fn: Callable[[np.ndarray, np.ndarray], float],
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  n_bootstraps: int = 1000,
                                  ci_level: float = 0.95,
                                  seed: int = 42) -> Tuple[float, float, float]:
    """
    Computes non-parametric bootstrap confidence intervals:
    Returns: (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)

    point_estimate = float(metric_fn(y_true, y_pred))

    if n_samples < 2:
        return point_estimate, point_estimate, point_estimate

    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, size=n_samples)
        score = metric_fn(y_true[indices], y_pred[indices])
        if np.isfinite(score):
            bootstrapped_scores.append(score)

    if not bootstrapped_scores:
        return point_estimate, point_estimate, point_estimate

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(bootstrapped_scores, 100.0 * alpha))
    upper = float(np.percentile(bootstrapped_scores, 100.0 * (1.0 - alpha)))

    return point_estimate, lower, upper


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        y_t = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_p = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.95, 0.3, 0.1, 0.85, 0.05])
        from sklearn.metrics import roc_auc_score
        pe, lo, hi = bootstrap_confidence_interval(roc_auc_score, y_t, y_p, n_bootstraps=200)
        print(f"Test AUROC Bootstrap CI: {pe:.4f} [{lo:.4f}, {hi:.4f}]")
