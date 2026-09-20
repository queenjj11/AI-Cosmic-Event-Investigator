"""Real-ZTF Light-Curve Anomaly Detector v2.

Operates directly on 128-D representations extracted by the frozen production LightCurveEncoder,
eliminating image-modality dependency for real ZTF light-curve-only alerts.
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors


class RealZTFLightCurveDetectorV2:
    """
    Light-Curve-First Anomaly Detector v2 for Real ZTF Data.

    Fits on known in-distribution real ZTF light-curve 128-D representations
    (SN Ia, SN II, Variable Stars) using PCA dimension reduction combined with
    shrinkage Mahalanobis distance, Isolation Forest, or k-NN distance.
    """

    def __init__(self,
                 method: str = "pca_mahalanobis",
                 n_components: int = 10,
                 percentile_threshold: float = 95.0):
        """
        Parameters:
            method: 'pca_mahalanobis', 'isolation_forest', or 'knn'
            n_components: Number of PCA components for dimension reduction (default: 10)
            percentile_threshold: Target percentile of in-distribution scores for thresholding
        """
        self.method = method
        self.n_components = n_components
        self.percentile_threshold = percentile_threshold

        self.pca: Optional[PCA] = None
        self.cov_estimator: Optional[LedoitWolf] = None
        self.mean_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.iso_forest: Optional[IsolationForest] = None
        self.knn: Optional[NearestNeighbors] = None

        self.decision_threshold: float = 0.5
        self.score_min: float = 0.0
        self.score_max: float = 1.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RealZTFLightCurveDetectorV2":
        """
        Fit detector on known in-distribution 128-D embeddings.

        Parameters:
            X: Array of shape (N, 128) containing in-distribution light-curve embeddings.
            y: Ignored (unsupervised fit on in-distribution data).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 128:
            raise ValueError(f"Expected input X of shape (N, 128), got shape {X.shape}")

        n_samples = X.shape[0]
        actual_components = min(self.n_components, n_samples - 1, 128)

        # 1. Fit PCA
        self.pca = PCA(n_components=actual_components, random_state=42)
        X_pca = self.pca.fit_transform(X)

        if self.method == "pca_mahalanobis":
            lw = LedoitWolf()
            lw.fit(X_pca)
            self.cov_estimator = lw
            self.mean_ = lw.location_
            self.precision_ = lw.precision_

        elif self.method == "isolation_forest":
            self.iso_forest = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42
            )
            self.iso_forest.fit(X_pca)

        elif self.method == "knn":
            n_neighbors = min(5, n_samples - 1)
            self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
            self.knn.fit(X_pca)

        else:
            raise ValueError(f"Unsupported method: {self.method}")

        self.is_fitted = True

        # Calibrate default score range on training set
        raw_train_scores = self.compute_raw_score(X)
        self.score_min = float(np.min(raw_train_scores))
        self.score_max = float(np.percentile(raw_train_scores, 99.0))
        if self.score_max <= self.score_min:
            self.score_max = self.score_min + 1.0

        # Set default decision threshold at specified percentile of train scores
        self.decision_threshold = float(np.percentile(raw_train_scores, self.percentile_threshold))
        return self

    def compute_raw_score(self, X: np.ndarray) -> np.ndarray:
        """Compute unscaled distance/outlier score for input 128-D embeddings."""
        if not self.is_fitted:
            raise RuntimeError("Detector is not fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_pca = self.pca.transform(X)

        if self.method == "pca_mahalanobis":
            delta = X_pca - self.mean_
            # Mahalanobis distance squared: delta @ precision @ delta.T
            raw_scores = np.sum(delta @ self.precision_ * delta, axis=1)
            # Take square root to get distance metric
            raw_scores = np.sqrt(np.maximum(raw_scores, 0.0))
            return raw_scores

        elif self.method == "isolation_forest":
            # Isolation forest score_samples returns negative anomaly score (lower = more anomalous)
            # Invert so higher = more anomalous
            return -self.iso_forest.score_samples(X_pca)

        elif self.method == "knn":
            distances, _ = self.knn.kneighbors(X_pca)
            # Mean distance to k nearest neighbors
            return np.mean(distances, axis=1)

        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def calibrate_threshold(self, val_X: np.ndarray, percentile: float = 95.0) -> float:
        """
        Calibrate decision threshold using held-out validation in-distribution embeddings.
        Prevents test label leakage.
        """
        val_scores = self.compute_raw_score(val_X)
        self.decision_threshold = float(np.percentile(val_scores, percentile))
        return self.decision_threshold

    def score(self, X: np.ndarray) -> Union[Tuple[float, bool], List[Tuple[float, bool]]]:
        """
        Compute continuous novelty score and binary decision for 128-D embedding(s).

        Returns:
            If single 1D array: (score, is_anomaly)
            If 2D array: list of (score, is_anomaly) tuples
        """
        is_single = (X.ndim == 1)
        raw_scores = self.compute_raw_score(X)

        # Scale continuous score to [0, 1] relative to score_min/score_max, clipped to [0.0, 1.0]
        scaled_scores = (raw_scores - self.score_min) / (self.score_max - self.score_min)
        scaled_scores = np.clip(scaled_scores, 0.0, 1.0)

        results = []
        for raw_s, scaled_s in zip(raw_scores, scaled_scores):
            is_anomaly = bool(raw_s >= self.decision_threshold)
            results.append((float(np.round(scaled_s, 4)), is_anomaly))

        if is_single:
            return results[0]
        return results

    def save(self, filepath: str) -> None:
        """Serialize fitted model to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "RealZTFLightCurveDetectorV2":
        """Load serialized model from disk."""
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not {cls.__name__}")
        return obj
