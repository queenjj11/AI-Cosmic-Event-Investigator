"""Decision-theoretic observation recommender selecting actions maximizing expected information gain."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.data.schema import AstronomicalEvent, Hypothesis, RecommendedObservation
from src.recommender.candidate_actions import CandidateActionSpace, CandidateAction
from src.recommender.information_gain import InformationGainCalculator


class ActiveObservationRecommender:
    """
    Ranks telescope follow-up actions by Bayesian Information Gain penalized by observational cost:
    Score(a) = E[Delta H(a)] - lambda * Cost(a)
    """

    def __init__(self, action_space: Optional[List[CandidateAction]] = None,
                 cost_penalty_lambda: float = 0.05):
        self.actions = action_space or CandidateActionSpace.get_actions()
        self.cost_penalty_lambda = cost_penalty_lambda

    def recommend_best_action(self, event: AstronomicalEvent,
                              hypotheses: List[Hypothesis]) -> Optional[RecommendedObservation]:
        """
        Evaluate all candidate actions and select the one with highest net information gain.
        """
        if not hypotheses or len(hypotheses) < 2:
            return None

        ranked_actions: List[Tuple[CandidateAction, float, float]] = []

        for action in self.actions:
            info_gain = InformationGainCalculator.calculate_expected_information_gain(
                event=event,
                hypotheses=hypotheses,
                action=action,
                num_mc_samples=12
            )
            # Net utility with cost regularization
            net_utility = info_gain - (self.cost_penalty_lambda * action.cost)
            ranked_actions.append((action, info_gain, net_utility))

        # Sort descending by net utility
        ranked_actions.sort(key=lambda x: x[2], reverse=True)
        best_action, best_ig, best_util = ranked_actions[0]

        # Formulate scientific justification
        rationale = (
            f"Maximizes expected Shannon Information Gain (Delta H = {best_ig:.4f} nats, "
            f"cost = {best_action.cost:.1f} hrs) to discriminate between competing hypotheses "
            f"('{hypotheses[0].name}' vs '{hypotheses[1].name}')."
        )

        return RecommendedObservation(
            action_id=best_action.id,
            name=best_action.name,
            band=best_action.band,
            delay_hours=best_action.delay_hours,
            cost=best_action.cost,
            expected_information_gain=best_ig,
            rationale=rationale
        )

    def rank_all_actions(self, event: AstronomicalEvent,
                         hypotheses: List[Hypothesis]) -> List[Dict[str, Any]]:
        """Return full ranking table for all candidate actions."""
        results = []
        for action in self.actions:
            ig = InformationGainCalculator.calculate_expected_information_gain(
                event, hypotheses, action, num_mc_samples=10
            )
            net = ig - (self.cost_penalty_lambda * action.cost)
            results.append({
                "action_id": action.id,
                "name": action.name,
                "band": action.band,
                "cost": action.cost,
                "information_gain": float(ig),
                "net_utility": float(net)
            })
        results.sort(key=lambda x: x["net_utility"], reverse=True)
        return results
