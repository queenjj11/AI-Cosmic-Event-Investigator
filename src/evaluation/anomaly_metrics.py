"""Anomaly detection metrics: AUROC, AUPRC, FDR at 95% TPR, and detection latency."""

from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve


def compute_anomaly_metrics(y_true: np.ndarray, anomaly_scores: np.ndarray) -> Dict[str, float]:
    """
    Computes anomaly detection metrics on held-out rare transient evaluation sets.
    y_true: (N,) binary array where 1 = anomalous class, 0 = in-distribution normal
    anomaly_scores: (N,) continuous scores in [0, 1]
    """
    try:
        auroc = float(roc_auc_score(y_true, anomaly_scores))
    except Exception:
        auroc = 0.5

    # Compute False Discovery Rate (FDR = FP / (TP + FP)) at 95% True Positive Rate
    fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
    idx_95 = np.where(tpr >= 0.95)[0]
    if len(idx_95) > 0:
        thresh = thresholds[idx_95[0]]
        preds = (anomaly_scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fdr_95 = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    else:
        fdr_95 = 1.0

    precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
    auprc = float(np.trapz(precision[::-1], recall[::-1]))

    return {
        "anomaly_auroc": auroc,
        "anomaly_auprc": auprc,
        "fdr_at_95_tpr": fdr_95
    }


def compute_detection_latency(partial_lightcurves_scores: List[np.ndarray],
                              y_true: np.ndarray,
                              threshold: float = 0.65) -> Dict[str, float]:
    """
    Measure detection latency: how early (in days or observation count) the system flags anomalies.
    partial_lightcurves_scores: List of scores evaluated at successive time cutoffs (e.g. Day 5, 10, 15, 20)
    """
    earliest_detections = []
    for i in range(len(y_true)):
        if y_true[i] != 1:
            continue
        detected = False
        for epoch_idx, scores_at_epoch in enumerate(partial_lightcurves_scores):
            if scores_at_epoch[i] >= threshold:
                earliest_detections.append(epoch_idx)
                detected = True
                break
        if not detected:
            earliest_detections.append(len(partial_lightcurves_scores))

    return {
        "mean_detection_step": float(np.mean(earliest_detections)) if earliest_detections else 0.0,
        "median_detection_step": float(np.median(earliest_detections)) if earliest_detections else 0.0
    }
