"""Random Forest baseline trained on hand-engineered astrophysical features."""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

from src.data.schema import AstronomicalEvent
from src.features.lightcurve_features import LightCurveFeatureExtractor
from src.features.image_features import ImageFeatureExtractor


class RandomForestBaseline:
    """Random Forest classifier on tabular light curve and morphology features."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 12, random_state: int = 42):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state
        )
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.lc_extractor = LightCurveFeatureExtractor()
        self.img_extractor = ImageFeatureExtractor()
        self.feature_names: List[str] = []
        self.classes_: np.ndarray = np.array([])
        self.is_fitted = False

    def extract_features_from_event(self, event: AstronomicalEvent) -> np.ndarray:
        """Extract combined feature vector for a single event."""
        lc_feats = self.lc_extractor.extract_features(event.lightcurve)
        img_feats = {}
        if event.image is not None and isinstance(event.image.data, np.ndarray):
            img_feats = self.img_extractor.extract_features(event.image.data)
        else:
            img_feats = {
                "image_fwhm": 2.5, "image_ellipticity": 0.1,
                "image_concentration": 1.0, "image_snr": 10.0,
                "host_transient_offset": 0.0
            }

        merged = {**lc_feats, **img_feats}
        if not self.feature_names:
            self.feature_names = sorted(merged.keys())

        return np.array([merged.get(k, 0.0) for k in self.feature_names], dtype=np.float32)

    def prepare_dataset(self, events: List[AstronomicalEvent]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a list of events into (X, y) matrices."""
        X_list, y_list = [], []
        for e in events:
            feat_vec = self.extract_features_from_event(e)
            X_list.append(feat_vec)
            y_list.append(e.true_label or "Unknown")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        return X, y

    def fit(self, events: List[AstronomicalEvent]) -> "RandomForestBaseline":
        """Train Random Forest baseline."""
        X, y = self.prepare_dataset(events)
        X_imp = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imp)
        self.rf.fit(X_scaled, y)
        self.classes_ = self.rf.classes_
        self.is_fitted = True
        return self

    def predict_proba(self, events: List[AstronomicalEvent]) -> np.ndarray:
        """Predict class probabilities for events."""
        if not self.is_fitted:
            raise ValueError("Random Forest model is not fitted.")
        X, _ = self.prepare_dataset(events)
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        return self.rf.predict_proba(X_scaled)

    def predict(self, events: List[AstronomicalEvent]) -> List[str]:
        """Predict class labels for events."""
        probas = self.predict_proba(events)
        best_indices = np.argmax(probas, axis=1)
        return [str(self.classes_[i]) for i in best_indices]

    def save(self, filepath: str) -> None:
        """Save model and transformers."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "rf": self.rf,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "classes": self.classes_
        }, filepath)

    def load(self, filepath: str) -> None:
        """Load model and transformers."""
        data = joblib.load(filepath)
        self.rf = data["rf"]
        self.imputer = data["imputer"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.classes_ = data["classes"]
        self.is_fitted = True
