"""Unit tests for Bayesian Active Observation Recommendation."""

import pytest
import numpy as np

from src.data.schema import Hypothesis, AstronomicalEvent, LightCurve
from src.recommender.candidate_actions import CandidateActionSpace, CandidateAction
from src.recommender.hypothesis_update import BayesianHypothesisUpdater
from src.recommender.information_gain import InformationGainCalculator
from src.recommender.recommender import ActiveObservationRecommender


def test_shannon_entropy():
    # Uniform distribution: highest entropy
    uniform = np.array([0.5, 0.5])
    h_uniform = InformationGainCalculator.shannon_entropy(uniform)

    # Deterministic: zero entropy
    certain = np.array([1.0, 0.0])
    h_certain = InformationGainCalculator.shannon_entropy(certain)

    assert h_uniform > 0.6
    assert h_certain < 1e-4


def test_bayesian_hypothesis_update():
    hyps = [
        Hypothesis(1, "Luminous Red Nova (LRN)", 0.6, 0.6, ["doc1"], "rationale", "test"),
        Hypothesis(2, "Type Ia Supernova", 0.4, 0.4, ["doc2"], "rationale", "test")
    ]
    action = CandidateAction("nir", "NIR Photometry", "NIR", 12.0, 2.5, "desc")
    # Outcome showing high NIR flux (strongly favoring LRN dust)
    outcome = {"simulated_flux": 2.5}

    posterior = BayesianHypothesisUpdater.update_posterior(hyps, action, outcome)
    assert len(posterior) == 2
    assert posterior[0] > 0.6  # LRN probability should increase


def test_recommender_selection():
    recommender = ActiveObservationRecommender()
    dummy_event = AstronomicalEvent(
        object_id="TEST_REC",
        ra=100.0, dec=20.0,
        lightcurve=LightCurve()
    )
    hyps = [
        Hypothesis(1, "Luminous Red Nova", 0.55, 0.6, ["doc1"], "rationale", "test"),
        Hypothesis(2, "Superluminous Supernova", 0.45, 0.5, ["doc2"], "rationale", "test")
    ]
    rec = recommender.recommend_best_action(dummy_event, hyps)

    assert rec is not None
    assert rec.expected_information_gain > 0.0
    assert rec.action_id in [a.id for a in CandidateActionSpace.get_actions()]
