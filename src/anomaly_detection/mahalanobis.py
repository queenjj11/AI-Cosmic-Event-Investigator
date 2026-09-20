"""Mahalanobis distance-based OOD detector with covariance shrinkage."""

from typing import Dict, List, Optional
import numpy as np
from sklearn.covariance import LedoitWolf


class MahalanobisDetector:
    """
    Computes class-conditional Mahalanobis distance from known-class manifold centroids:
    D_M(z) = min_c sqrt( (z - mu_c)^T Sigma^{-1} (z - mu_c) )
    Uses Ledoit-Wolf shrinkage for well-conditioned covariance estimation.
    """

    def __init__(self):
        self.class_means: Dict[int, np.ndarray] = {}
        self.precision_matrix: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> "MahalanobisDetector":
        """
        Fit class centroids mu_c and pooled precision matrix Sigma^{-1}.
        embeddings: (N, D)
        labels: (N,) integer class indices
        """
        unique_classes = np.unique(labels)
        dim = embeddings.shape[1]

        # 1. Compute per-class centroids
        centered_diffs = []
        for c in unique_classes:
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            class_embeds = embeddings[idx]
            mu_c = np.mean(class_embeds, axis=0)
            self.class_means[int(c)] = mu_c
            centered_diffs.append(class_embeds - mu_c)

        # 2. Estimate pooled covariance with Ledoit-Wolf shrinkage
        all_centered = np.vstack(centered_diffs)
        lw = LedoitWolf()
        lw.fit(all_centered)
        self.precision_matrix = lw.precision_  # Sigma^{-1}
        self.is_fitted = True
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute minimum Mahalanobis distance across all known classes for each embedding.
        Higher distance corresponds to higher degree of novelty/anomaly.
        embeddings: (N, D)
        Returns: (N,) anomaly scores
        """
        if not self.is_fitted or self.precision_matrix is None:
            raise ValueError("Mahalanobis detector is not fitted.")

        N = embeddings.shape[0]
        min_distances = np.full(N, np.inf)

        for c, mu_c in self.class_means.items():
            diff = embeddings - mu_c  # (N, D)
            # D_M^2 = (diff @ Sigma^{-1} * diff).sum(axis=-1)
            dist_sq = np.sum((diff @ self.precision_matrix) * diff, axis=1)
            dist = np.sqrt(np.clip(dist_sq, 0.0, None))
            min_distances = np.minimum(min_distances, dist)

        return min_distances
