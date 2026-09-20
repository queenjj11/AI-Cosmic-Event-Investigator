"""Data schemas and standard event representations for ACEI."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Observation:
    """A single photometric observation in an astronomical light curve."""
    time: float          # Modified Julian Date (MJD) or relative days
    band: str           # Passband (e.g., 'g', 'r', 'i')
    flux: float         # Calibrated flux (microJansky or counts)
    flux_err: float     # 1-sigma uncertainty on flux
    mag: Optional[float] = None
    mag_err: Optional[float] = None


@dataclass
class LightCurve:
    """Multiband light curve composed of sequential observations."""
    observations: List[Observation] = field(default_factory=list)

    def add_observation(self, time: float, band: str, flux: float, flux_err: float,
                        mag: Optional[float] = None, mag_err: Optional[float] = None) -> None:
        self.observations.append(Observation(time=time, band=band, flux=flux, flux_err=flux_err,
                                             mag=mag, mag_err=mag_err))

    def sort_chronologically(self) -> None:
        self.observations.sort(key=lambda o: o.time)

    def get_band_data(self, band: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract sorted (times, fluxes, flux_errs) for a given filter band."""
        obs = [o for o in self.observations if o.band == band]
        obs.sort(key=lambda o: o.time)
        times = np.array([o.time for o in obs], dtype=np.float32)
        fluxes = np.array([o.flux for o in obs], dtype=np.float32)
        errs = np.array([o.flux_err for o in obs], dtype=np.float32)
        return times, fluxes, errs

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Return full sequential arrays (times, fluxes, errs, bands)."""
        self.sort_chronologically()
        times = np.array([o.time for o in self.observations], dtype=np.float32)
        fluxes = np.array([o.flux for o in self.observations], dtype=np.float32)
        errs = np.array([o.flux_err for o in self.observations], dtype=np.float32)
        bands = [o.band for o in self.observations]
        return times, fluxes, errs, bands

    def __len__(self) -> int:
        return len(self.observations)


@dataclass
class CutoutImage:
    """Astronomical postage stamp image cutout."""
    data: np.ndarray  # Shape: (C, H, W) e.g., (3, 64, 64) for [science, template, difference]
    channels: List[str] = field(default_factory=lambda: ["science", "template", "difference"])
    pixel_scale_arcsec: float = 1.0


@dataclass
class AstronomicalEvent:
    """Standard normalized entity for any transient or variable cosmic event."""
    object_id: str
    ra: float
    dec: float
    lightcurve: LightCurve
    image: Optional[CutoutImage] = None
    true_label: Optional[str] = None
    is_anomaly: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """A scientific explanation for an anomalous cosmic event."""
    rank: int
    name: str
    probability: float
    raw_confidence: float
    evidence_sources: List[str]
    justification: str
    distinguishing_criteria: str


@dataclass
class RecommendedObservation:
    """An active observation action recommended to reduce hypothesis uncertainty."""
    action_id: str
    name: str
    band: str
    delay_hours: float
    cost: float
    expected_information_gain: float  # Shannon entropy reduction (nats or bits)
    rationale: str


@dataclass
class InvestigationResult:
    """Comprehensive output of the ACEI pipeline for an event."""
    event_id: str
    is_anomaly: bool
    anomaly_score: float
    detector_scores: Dict[str, float]  # {'autoencoder': ..., 'mahalanobis': ..., 'energy': ...}
    classifier_prediction: Optional[str] = None
    classifier_confidence: Optional[float] = None
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    recommended_action: Optional[RecommendedObservation] = None
    execution_time_seconds: float = 0.0
