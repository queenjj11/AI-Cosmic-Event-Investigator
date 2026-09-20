"""Bayesian posterior updating of competing hypotheses given simulated observations."""

from typing import Any, Dict, List
import numpy as np
from src.data.schema import Hypothesis
from src.recommender.candidate_actions import CandidateAction


class BayesianHypothesisUpdater:
    """
    Computes posterior probability distribution over hypotheses:
    P(h_i | o) = [P(o | h_i) * P(h_i)] / sum_j [P(o | h_j) * P(h_j)]
    """

    @staticmethod
    def compute_likelihood(outcome: Dict[str, float], hypothesis: Hypothesis, action: CandidateAction) -> float:
        """Evaluate observational likelihood P(o | h_i) based on physical signatures."""
        hyp_name = hypothesis.name.lower()
        band = action.band
        flux = outcome.get("simulated_flux", 1.0)

        # Likelihood evaluation logic matching astrophysics literature
        if band == "NIR":
            # LRNe have extreme NIR dust excess
            if "lrn" in hyp_name or "red nova" in hyp_name:
                return float(np.exp(-0.5 * ((flux - 2.5) / 0.4)**2))
            else:
                return float(np.exp(-0.5 * ((flux - 0.4) / 0.2)**2))

        elif band == "optical_spec":
            # Ejecta velocity differentiation
            if "lrn" in hyp_name:
                # Narrow lines, low expansion velocity (~600 km/s)
                return float(np.exp(-0.5 * ((flux - 600.0) / 150.0)**2))
            elif "slsn" in hyp_name:
                # Broad lines, high velocity (~12,000 km/s)
                return float(np.exp(-0.5 * ((flux - 12000.0) / 1000.0)**2))
            else:
                # Standard supernova (~9,000 km/s)
                return float(np.exp(-0.5 * ((flux - 9000.0) / 1000.0)**2))

        elif band == "g":
            # Blue color retention
            if "slsn" in hyp_name or "tde" in hyp_name:
                return float(np.exp(-0.5 * ((flux - 2.0) / 0.4)**2))
            else:
                return float(np.exp(-0.5 * ((flux - 0.5) / 0.2)**2))

        else:  # r band
            return float(np.exp(-0.5 * ((flux - 1.2) / 0.3)**2))

    @classmethod
    def update_posterior(cls, hypotheses: List[Hypothesis],
                         action: CandidateAction,
                         outcome: Dict[str, float]) -> np.ndarray:
        """
        Compute normalized posterior array P(h_i | o).
        Returns: (num_hypotheses,) numpy array summing to 1.0.
        """
        priors = np.array([h.probability for h in hypotheses], dtype=np.float32)
        likelihoods = np.array([
            cls.compute_likelihood(outcome, h, action) for h in hypotheses
        ], dtype=np.float32)

        # Avoid zero division
        likelihoods = np.clip(likelihoods, 1e-6, None)
        unnormalized = priors * likelihoods
        sum_p = np.sum(unnormalized)

        if sum_p > 0:
            return unnormalized / sum_p
        return priors / np.sum(priors)
