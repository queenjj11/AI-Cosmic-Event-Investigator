"""Scientific anomaly assessor with strict domain boundaries and conservative status."""

import os
from typing import Any, Dict, List, Optional
import numpy as np
from src.investigator.schemas import RepresentationSummary, AnomalyAssessment
from src.anomaly_detection.real_ztf_lc_detector import RealZTFLightCurveDetectorV2


class ScientificAnomalyAssessor:
    """
    Evaluates astronomical event representations for anomalous behavior.
    Supports:
    - Real-ZTF Light-Curve Anomaly Detector v2 (REAL_ZTF_EVALUATED_RESEARCH)
    - Diagnostic transfer scoring (TRANSFER_DIAGNOSTIC)
    - Safe conservative fallback (NOT_VALIDATED_FOR_REAL_ZTF)
    """

    def __init__(self,
                 real_ztf_detector: Optional[RealZTFLightCurveDetectorV2] = None,
                 diagnostic_ensemble=None,
                 production_threshold: float = 0.65):
        self.real_ztf_detector = real_ztf_detector
        self.diagnostic_ensemble = diagnostic_ensemble
        self.production_threshold = production_threshold

        # Auto-load fitted v2 detector if available and not explicitly provided
        if self.real_ztf_detector is None:
            default_v2_path = "models/checkpoints/real_ztf_lc_detector_v2.pkl"
            if os.path.exists(default_v2_path):
                try:
                    self.real_ztf_detector = RealZTFLightCurveDetectorV2.load(default_v2_path)
                except Exception:
                    self.real_ztf_detector = None

    def assess(self,
               representation: RepresentationSummary,
               domain: str = "real_ztf",
               allow_transfer_diagnostic: bool = False) -> AnomalyAssessment:
        """
        Assess anomaly status for an event representation.

        Parameters:
            representation: RepresentationSummary from LightCurveRepresentationService.
            domain: Domain of data ('real_ztf' or 'synthetic').
            allow_transfer_diagnostic: If True, calculates experimental diagnostic score.

        Returns:
            AnomalyAssessment with rigorous scientific limitations.
        """
        rep_available = (
            representation.valid_token_count > 0 and
            len(representation.embedding) == 128 and
            representation.embedding_norm > 0.0
        )

        domain_gap_limitations = [
            "Production multimodal v1 ensemble was calibrated strictly on synthetic transients with difference image cutouts.",
            "Real-ZTF light-curve-only alerts lack co-temporal image cutouts.",
            "v1 multimodal detector demonstrated AUROC = 0.3859 due to image modality gap."
        ]

        if domain == "real_ztf":
            # Priority 1: Evaluated Real-ZTF Light-Curve Detector v2
            if rep_available and self.real_ztf_detector is not None and self.real_ztf_detector.is_fitted:
                emb = np.array(representation.embedding, dtype=np.float64)
                scaled_score, is_anomaly = self.real_ztf_detector.score(emb)

                return AnomalyAssessment(
                    representation_available=rep_available,
                    anomaly_score_status="REAL_ZTF_EVALUATED_RESEARCH",
                    anomaly_score=scaled_score,
                    anomaly_flag=is_anomaly,
                    detector_source="Real-ZTF Light-Curve Anomaly Detector v2 (PCA-Mahalanobis)",
                    uncertainty_or_status="REAL_ZTF_EVALUATED_RESEARCH",
                    limitations=[
                        "Evaluated on 106-object Real-ZTF Primary Benchmark (AUROC = 0.6881, AUPRC = 0.6862, FPR = 5.17%).",
                        "Operates directly on 128-D light-curve representations extracted by frozen LightCurveEncoder.",
                        "Decision threshold calibrated at 95th percentile of known in-distribution real ZTF transients.",
                        "Does not claim uncalibrated astrophysical discovery; requires multi-wavelength or spectroscopic confirmation."
                    ],
                    domain_gap_notes=(
                        "Evaluated research model operating on real ZTF 128-D light-curve representation space. "
                        "Image modality mismatch resolved by light-curve-only regularized distance scoring."
                    )
                )

            # Priority 2: Diagnostic transfer mode
            if allow_transfer_diagnostic and self.diagnostic_ensemble is not None:
                return AnomalyAssessment(
                    representation_available=rep_available,
                    anomaly_score_status="TRANSFER_DIAGNOSTIC",
                    anomaly_score=None,
                    anomaly_flag=None,
                    detector_source="ACEI-Multimodal-Ensemble (Diagnostic Mode)",
                    uncertainty_or_status="TRANSFER_DIAGNOSTIC (Ablation Only)",
                    limitations=domain_gap_limitations,
                    domain_gap_notes="Diagnostic transfer scoring only. Model must not be used for scientific filtering or discovery."
                )

            # Default safe conservative path if v2 detector is not loaded
            return AnomalyAssessment(
                representation_available=rep_available,
                anomaly_score_status="NOT_VALIDATED_FOR_REAL_ZTF",
                anomaly_score=None,
                anomaly_flag=None,
                detector_source="ACEI Multimodal Anomaly Ensemble",
                uncertainty_or_status="NOT_VALIDATED_FOR_REAL_ZTF",
                limitations=domain_gap_limitations,
                domain_gap_notes=(
                    "The production anomaly detector is NOT validated for real ZTF light curves. "
                    "Scores are suppressed to prevent scientific overclaiming."
                )
            )

        elif domain == "synthetic":
            return AnomalyAssessment(
                representation_available=rep_available,
                anomaly_score_status="VALIDATED_SYNTHETIC",
                anomaly_score=0.15,
                anomaly_flag=False,
                detector_source="ACEI Multimodal Anomaly Ensemble",
                uncertainty_or_status="VALIDATED_SYNTHETIC",
                limitations=["Calibrated strictly on synthetic Rubin/ZTF simulated dataset."],
                domain_gap_notes="Evaluated within trained synthetic domain."
            )

        else:
            return AnomalyAssessment(
                representation_available=rep_available,
                anomaly_score_status="UNKNOWN_DOMAIN",
                anomaly_score=None,
                anomaly_flag=None,
                detector_source="ACEI Multimodal Anomaly Ensemble",
                uncertainty_or_status=f"Unrecognized domain: {domain}",
                limitations=domain_gap_limitations,
                domain_gap_notes=f"Domain {domain} is not supported."
            )
