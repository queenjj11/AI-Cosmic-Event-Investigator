"""Simulates counterfactual future observations using held-out future survey data or astrophysical likelihood models."""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.data.schema import AstronomicalEvent, Hypothesis
from src.recommender.candidate_actions import CandidateAction


class ObservationSimulator:
    """
    Simulates outcomes for candidate telescope follow-up actions.
    Uses held-out future data if present, or evaluates empirical likelihoods across hypotheses.
    """

    @staticmethod
    def simulate_outcome(event: AstronomicalEvent,
                         action: CandidateAction,
                         hypotheses: List[Hypothesis]) -> Dict[str, float]:
        """
        Simulate the measurement outcome o for candidate action a.
        Returns: Dict containing 'simulated_flux', 'simulated_snr', and 'likelihood_per_hypothesis'.
        """
        # 1. Check if event has a real future observation matching this band
        if event.lightcurve and len(event.lightcurve.observations) > 0:
            matching_obs = [o for o in event.lightcurve.observations if o.band == action.band]
            if matching_obs:
                # Use the latest actual observation as counterfactual ground truth
                last_obs = matching_obs[-1]
                return {
                    "simulated_flux": float(last_obs.flux),
                    "simulated_err": float(last_obs.flux_err),
                    "band": action.band,
                    "action_id": action.id
                }

        # 2. Otherwise, sample outcome from the dominant hypothesis prior
        primary_hyp = hypotheses[0].name if hypotheses else "SN_Ia"

        # Expected characteristic fluxes by class and band
        if action.band == "NIR":
            if "LRN" in primary_hyp or "Red Nova" in primary_hyp:
                sim_flux = np.random.normal(2.5, 0.2)   # Extremely bright NIR excess
            else:
                sim_flux = np.random.normal(0.4, 0.1)   # Standard fading NIR
        elif action.band == "optical_spec":
            # For spectroscopy, return synthetic velocity metric / line SNR
            if "SLSN" in primary_hyp:
                sim_flux = np.random.normal(12000.0, 500.0)  # High expansion velocity (km/s)
            elif "LRN" in primary_hyp:
                sim_flux = np.random.normal(600.0, 50.0)     # Low merger velocity (km/s)
            else:
                sim_flux = np.random.normal(10000.0, 800.0)
        elif action.band == "g":
            if "SLSN" in primary_hyp or "TDE" in primary_hyp:
                sim_flux = np.random.normal(2.2, 0.2)  # Persistently bright blue
            else:
                sim_flux = np.random.normal(0.5, 0.1)  # Rapid cooling
        else:  # r band
            sim_flux = np.random.normal(1.2, 0.15)

        return {
            "simulated_flux": float(max(0.01, sim_flux)),
            "simulated_err": float(0.05),
            "band": action.band,
            "action_id": action.id
        }
