"""Calibrates raw LLM confidence scores using Isotonic Regression and Platt scaling."""

from typing import List, Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import joblib
import os


class ConfidenceCalibrator:
    """
    Transforms raw stated LLM confidence scores into empirically calibrated probabilities.
    Directly addresses LLM overconfidence on scientific triage tasks.
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        self.platt = LogisticRegression(C=1.0, max_iter=1000)
        self.temperature = 1.0
        self.is_fitted = False

    def fit(self, confidences: np.ndarray, correct_indicators: np.ndarray) -> "ConfidenceCalibrator":
        """
        confidences: (N,) raw probabilities stated by LLM (0.0 to 1.0)
        correct_indicators: (N,) binary indicator (1 if hypothesis was correct ground truth, 0 otherwise)
        """
        confidences = np.clip(confidences, 1e-4, 1.0 - 1e-4)

        if self.method == "isotonic":
            self.isotonic.fit(confidences, correct_indicators)
        elif self.method == "platt":
            # Logit transform of probabilities for logistic regression
            logits = np.log(confidences / (1.0 - confidences)).reshape(-1, 1)
            self.platt.fit(logits, correct_indicators)
        elif self.method == "temperature":
            # Grid search for temperature minimizing NLL / Brier score
            best_brier = float("inf")
            best_T = 1.0
            for T in np.linspace(0.5, 3.0, 50):
                logits = np.log(confidences / (1.0 - confidences)) / T
                probs = 1.0 / (1.0 + np.exp(-logits))
                brier = np.mean((probs - correct_indicators)**2)
                if brier < best_brier:
                    best_brier = brier
                    best_T = T
            self.temperature = best_T

        self.is_fitted = True
        return self

    def calibrate(self, confidences: np.ndarray) -> np.ndarray:
        """Map raw confidences to calibrated probabilities."""
        if not self.is_fitted:
            # Empirical shrink towards uniform prior if uncalibrated
            return np.clip(0.8 * confidences + 0.1, 0.01, 0.99)

        confidences = np.clip(confidences, 1e-4, 1.0 - 1e-4)

        if self.method == "isotonic":
            return np.clip(self.isotonic.predict(confidences), 0.01, 0.99)
        elif self.method == "platt":
            logits = np.log(confidences / (1.0 - confidences)).reshape(-1, 1)
            return self.platt.predict_proba(logits)[:, 1]
        elif self.method == "temperature":
            logits = np.log(confidences / (1.0 - confidences)) / self.temperature
            return 1.0 / (1.0 + np.exp(-logits))
        return confidences

    def save(self, filepath: str) -> None:
        """Persist calibration parameters."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "method": self.method,
            "isotonic": self.isotonic,
            "platt": self.platt,
            "temperature": self.temperature,
            "is_fitted": self.is_fitted
        }, filepath)

    def load(self, filepath: str) -> "ConfidenceCalibrator":
        """Load calibration parameters from disk."""
        data = joblib.load(filepath)
        self.method = data["method"]
        self.isotonic = data["isotonic"]
        self.platt = data["platt"]
        self.temperature = data["temperature"]
        self.is_fitted = data["is_fitted"]
        return self
