"""Calibrated 3-signal anomaly ensemble combining autoencoder, Mahalanobis, and energy."""

from typing import Dict, Optional, Tuple
import numpy as np
import torch

from src.anomaly_detection.autoencoder import MultimodalAutoencoder
from src.anomaly_detection.mahalanobis import MahalanobisDetector
from src.anomaly_detection.energy_score import EnergyOODDetector


class AnomalyEnsemble:
    """
    Ensembles 3 complementary anomaly signals:
    1. Multimodal autoencoder reconstruction error
    2. Embedding Mahalanobis distance from known class manifold
    3. Classifier logit free-energy score
    Each signal is calibrated against normal distribution percentiles and combined.
    """

    def __init__(self,
                 autoencoder: MultimodalAutoencoder,
                 mahalanobis: MahalanobisDetector,
                 energy_detector: EnergyOODDetector,
                 weights: Optional[Dict[str, float]] = None,
                 threshold: float = 0.65):
        self.autoencoder = autoencoder
        self.mahalanobis = mahalanobis
        self.energy_detector = energy_detector
        self.threshold = threshold

        default_weights = {"autoencoder": 0.40, "mahalanobis": 0.35, "energy": 0.25}
        self.weights = weights or default_weights

        # Calibration parameters (min, max per signal for min-max scaling to [0, 1])
        self.calibration_params: Dict[str, Tuple[float, float]] = {}
        self.is_calibrated = False

    def calibrate(self, val_embeddings: np.ndarray, val_logits: np.ndarray) -> "AnomalyEnsemble":
        """
        Learn normal-class score distributions on an in-distribution validation set
        using robust statistics (median and astronomical robust standard deviation via MAD).
        """
        tensor_embeds = torch.from_numpy(val_embeddings).float()
        raw_ae = self.autoencoder.compute_reconstruction_error(tensor_embeds)
        raw_mah = self.mahalanobis.score(val_embeddings)
        raw_energy = self.energy_detector.score(val_logits)

        for name, scores in [("autoencoder", raw_ae), ("mahalanobis", raw_mah), ("energy", raw_energy)]:
            scores_arr = np.asarray(scores, dtype=np.float64)
            med_val = float(np.median(scores_arr))
            # Astronomical robust scale: sigma_robust = 1.4826 * MAD
            mad = float(np.median(np.abs(scores_arr - med_val)))
            robust_std = 1.4826 * mad
            sample_std = float(np.std(scores_arr))
            scale_val = robust_std if robust_std > 1e-4 else sample_std
            if scale_val < 1e-4:
                scale_val = 1.0
            self.calibration_params[name] = (med_val, scale_val)

        self.is_calibrated = True
        return self

    def normalize_score(self, raw_score: float, signal_name: str) -> float:
        """
        Convert raw score to [0, 1] anomaly score using calibrated sigmoid of standardized deviation (z-score).
        Normal in-distribution events (z <= 0) receive low scores <= 0.15.
        Moderate deviations (z ~ 1.5 sigma) receive intermediate scores ~ 0.60.
        Strong deviations (z >= 2.5 sigma) receive high scores >= 0.85.
        """
        if not self.is_calibrated or signal_name not in self.calibration_params:
            return float(1.0 / (1.0 + np.exp(-raw_score)))

        med_val, scale_val = self.calibration_params[signal_name]
        z = (raw_score - med_val) / (scale_val + 1e-6)
        # Calibrated logistic mapping: center at +2.0 sigma deviation from normal median
        score = 1.0 / (1.0 + np.exp(-1.3 * (z - 2.0)))
        return float(np.clip(score, 0.0, 1.0))

    def score_event(self, embedding: np.ndarray, logits: np.ndarray) -> Tuple[float, bool, Dict[str, float]]:
        """
        Evaluate a single astronomical event.
        embedding: (D,) or (1, D)
        logits: (num_classes,) or (1, num_classes)
        Returns: (ensemble_score, is_flagged, individual_scores_dict)
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        tensor_embed = torch.from_numpy(embedding).float()

        # 1. Compute raw scores
        raw_ae = float(self.autoencoder.compute_reconstruction_error(tensor_embed)[0])
        raw_mah = float(self.mahalanobis.score(embedding)[0])
        raw_energy = float(self.energy_detector.score(logits)[0])

        # 2. Normalize scores to [0, 1]
        norm_ae = self.normalize_score(raw_ae, "autoencoder")
        norm_mah = self.normalize_score(raw_mah, "mahalanobis")
        norm_energy = self.normalize_score(raw_energy, "energy")

        # 3. Ensemble combining weighted modality consensus and peak detector response
        weighted_score = (
            self.weights.get("autoencoder", 0.40) * norm_ae +
            self.weights.get("mahalanobis", 0.35) * norm_mah +
            self.weights.get("energy", 0.25) * norm_energy
        )
        max_signal = max(norm_ae, norm_mah, norm_energy)
        ensemble_score = 0.80 * weighted_score + 0.20 * max_signal
        ensemble_score = float(np.clip(ensemble_score, 0.0, 1.0))
        is_anomaly = bool(ensemble_score >= self.threshold)

        details = {
            "autoencoder_recon_error": raw_ae,
            "autoencoder_norm": norm_ae,
            "mahalanobis_distance": raw_mah,
            "mahalanobis_norm": norm_mah,
            "energy_score": raw_energy,
            "energy_norm": norm_energy,
            "ensemble_score": ensemble_score
        }

        return ensemble_score, is_anomaly, details
