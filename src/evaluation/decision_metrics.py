"""Active observation decision quality: Information Gain and Policy Comparisons."""

from typing import Any, Dict, List
import numpy as np


def compute_decision_metrics(recommended_gains: List[float],
                             random_policy_gains: List[float],
                             fixed_policy_gains: List[float]) -> Dict[str, Any]:
    """
    Benchmarks active observation recommendations against baseline policies:
    - mean_recommended_gain: Average entropy reduction under ACEI active policy
    - mean_random_gain: Average entropy reduction under uniform random action selection
    - mean_fixed_gain: Average entropy reduction under fixed single-band policy
    - gain_over_random_pct: Percentage improvement of active recommendation over random
    """
    mean_rec = float(np.mean(recommended_gains)) if recommended_gains else 0.0
    mean_rand = float(np.mean(random_policy_gains)) if random_policy_gains else 0.0
    mean_fixed = float(np.mean(fixed_policy_gains)) if fixed_policy_gains else 0.0

    improvement_vs_rand = float((mean_rec - mean_rand) / (mean_rand + 1e-6)) * 100.0
    improvement_vs_fixed = float((mean_rec - mean_fixed) / (mean_fixed + 1e-6)) * 100.0

    return {
        "mean_recommended_gain": mean_rec,
        "mean_random_gain": mean_rand,
        "mean_fixed_gain": mean_fixed,
        "improvement_over_random_pct": improvement_vs_rand,
        "improvement_over_fixed_pct": improvement_vs_fixed
    }
