"""Anomaly detection evaluation: AUROC, False Discovery Rate, and detection latency."""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve


class AnomalyEvaluator:
    """Evaluates anomaly detection performance against held-out rare transient classes."""

    @staticmethod
    def evaluate_detector(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """
        Compute anomaly evaluation metrics:
        y_true: 1 for anomaly, 0 for normal known class
        scores: Anomaly scores (higher = more anomalous)
        """
        try:
            auroc = float(roc_auc_score(y_true, scores))
        except Exception:
            auroc = 0.5

        # Compute False Discovery Rate (FDR) at 95% True Positive Rate (Recall)
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        # Find threshold where TPR >= 0.95
        idx_95 = np.where(tpr >= 0.95)[0]
        if len(idx_95) > 0:
            target_idx = idx_95[0]
            thresh_95 = thresholds[target_idx]
            preds_95 = (scores >= thresh_95).astype(int)
            tp = np.sum((preds_95 == 1) & (y_true == 1))
            fp = np.sum((preds_95 == 1) & (y_true == 0))
            fdr_at_95_tpr = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
        else:
            fdr_at_95_tpr = 1.0

        precision, recall, _ = precision_recall_curve(y_true, scores)
        auprc = float(np.trapz(precision[::-1], recall[::-1]))

        return {
            "auroc": auroc,
            "auprc": auprc,
            "fdr_at_95_tpr": fdr_at_95_tpr
        }

    @classmethod
    def compare_signals(cls, y_true: np.ndarray,
                        ae_scores: np.ndarray,
                        mah_scores: np.ndarray,
                        energy_scores: np.ndarray,
                        ensemble_scores: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compare all 3 individual anomaly detectors against the ensemble."""
        return {
            "Autoencoder_Reconstruction": cls.evaluate_detector(y_true, ae_scores),
            "Mahalanobis_Distance": cls.evaluate_detector(y_true, mah_scores),
            "Energy_Score": cls.evaluate_detector(y_true, energy_scores),
            "Anomaly_Ensemble": cls.evaluate_detector(y_true, ensemble_scores),
        }
