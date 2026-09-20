"""Free energy score and max-softmax OOD detectors."""

import numpy as np
import torch
from scipy.special import logsumexp


class EnergyOODDetector:
    """
    Computes free-energy score from classifier logits:
    E(x; T) = -T * logsumexp(logits / T)
    Higher energy (less negative log-sum-exp) indicates an out-of-distribution event.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def score(self, logits: np.ndarray) -> np.ndarray:
        """
        Compute energy anomaly score for logits of shape (N, num_classes).
        Higher value => higher anomaly / OOD score.
        """
        scaled_logits = logits / self.temperature
        # Energy = -T * logsumexp(logits / T)
        # We negate it so higher values indicate anomalousness
        lse = logsumexp(scaled_logits, axis=1)
        energy_score = -self.temperature * lse
        # Standard anomaly scoring: flip sign so less confident / low LSE has high score
        # Note: when all logits are low/uncertain, lse is small, so -lse is large.
        return -lse

    @staticmethod
    def max_softmax_score(logits: np.ndarray) -> np.ndarray:
        """
        Baseline max-softmax probability OOD score: 1.0 - max_c(Softmax(logits)_c).
        Higher value indicates lower model confidence / higher anomaly.
        """
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        max_prob = np.max(probs, axis=1)
        return 1.0 - max_prob
