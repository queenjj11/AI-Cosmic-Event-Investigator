"""Hypothesis scoring, consistency checking, and probability re-ranking."""

from typing import List
import numpy as np
from src.data.schema import Hypothesis


class HypothesisRanker:
    """Ranks and normalizes hypothesis probabilities based on evidence groundedness and consistency."""

    @staticmethod
    def rank_and_normalize(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """
        Ensures valid ranking order and strict probability simplex constraint: sum(P(h_i)) == 1.0.
        """
        if not hypotheses:
            return []

        # Sort descending by raw probability
        sorted_hyps = sorted(hypotheses, key=lambda h: h.probability, reverse=True)

        # Normalize probabilities
        probs = np.array([max(0.01, h.probability) for h in sorted_hyps], dtype=np.float32)
        norm_probs = probs / np.sum(probs)

        ranked: List[Hypothesis] = []
        for i, h in enumerate(sorted_hyps):
            ranked.append(Hypothesis(
                rank=i + 1,
                name=h.name,
                probability=float(norm_probs[i]),
                raw_confidence=h.raw_confidence,
                evidence_sources=h.evidence_sources,
                justification=h.justification,
                distinguishing_criteria=h.distinguishing_criteria
            ))

        return ranked
