"""Calculates Shannon entropy and expected Bayesian Information Gain."""

from typing import Any, Dict, List
import numpy as np
from src.data.schema import Hypothesis
from src.recommender.candidate_actions import CandidateAction
from src.recommender.observation_simulator import ObservationSimulator
from src.recommender.hypothesis_update import BayesianHypothesisUpdater


class InformationGainCalculator:
    """
    Evaluates expected Shannon Information Gain (entropy reduction) for candidate actions:
    E[Delta H(a)] = H(P(h)) - E_{o ~ P(o|a)} [ H(P(h | o)) ]
    """

    @staticmethod
    def shannon_entropy(probs: np.ndarray) -> float:
        """Compute Shannon entropy H(P) = -sum(p * ln(p)) in nats."""
        probs = np.clip(probs, 1e-12, 1.0)
        # Re-normalize just in case
        probs = probs / np.sum(probs)
        return float(-np.sum(probs * np.log(probs)))

    @classmethod
    def calculate_expected_information_gain(cls, event: Any,
                                           hypotheses: List[Hypothesis],
                                           action: CandidateAction,
                                           num_mc_samples: int = 15) -> float:
        """
        Estimate expected information gain via Monte Carlo integration over simulated outcomes.
        """
        if len(hypotheses) <= 1:
            return 0.0

        priors = np.array([h.probability for h in hypotheses], dtype=np.float32)
        h_prior = cls.shannon_entropy(priors)

        # Monte Carlo sampling of potential outcomes
        posterior_entropies = []
        for _ in range(num_mc_samples):
            sim_outcome = ObservationSimulator.simulate_outcome(event, action, hypotheses)
            posterior = BayesianHypothesisUpdater.update_posterior(hypotheses, action, sim_outcome)
            h_post = cls.shannon_entropy(posterior)
            posterior_entropies.append(h_post)

        expected_h_post = float(np.mean(posterior_entropies))
        # Information Gain is the reduction in entropy
        info_gain = float(max(0.0, h_prior - expected_h_post))
        return info_gain
