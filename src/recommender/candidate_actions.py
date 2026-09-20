"""Defines the discrete candidate follow-up observation actions and telescope costs."""

from typing import Any, Dict, List
from dataclasses import dataclass


@dataclass
class CandidateAction:
    """A candidate follow-up observation action."""
    id: str
    name: str
    band: str
    delay_hours: float
    cost: float
    description: str


class CandidateActionSpace:
    """Catalog of candidate astronomical telescope follow-up actions."""

    DEFAULT_ACTIONS = [
        CandidateAction(
            id="reobserve_r_band_2h",
            name="Immediate Rapid Re-observation in r-band (2h)",
            band="r",
            delay_hours=2.0,
            cost=1.0,
            description="High-cadence photometric monitoring to constrain fast rise/decay rates."
        ),
        CandidateAction(
            id="reobserve_g_band_tomorrow",
            name="Next-Night Photometry in g-band (24h)",
            band="g",
            delay_hours=24.0,
            cost=0.8,
            description="Constrain blue cooling slope and temperature evolution."
        ),
        CandidateAction(
            id="request_nir_photometry",
            name="Target-of-Opportunity NIR (J/H/K) Photometry",
            band="NIR",
            delay_hours=12.0,
            cost=2.5,
            description="Identify late-time dust condensation or red merger signatures."
        ),
        CandidateAction(
            id="request_optical_spectroscopy",
            name="Target-of-Opportunity Optical Spectroscopy",
            band="optical_spec",
            delay_hours=6.0,
            cost=5.0,
            description="Measure ejecta velocities, absorption lines (Si II, Balmer, O II) for unambiguous physical typing."
        )
    ]

    @classmethod
    def get_actions(cls) -> List[CandidateAction]:
        """Return the default action set."""
        return list(cls.DEFAULT_ACTIONS)
