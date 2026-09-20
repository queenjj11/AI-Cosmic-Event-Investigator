"""Unified evaluation harness comparing unimodal and classical baselines."""

import os
import json
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from src.data.schema import AstronomicalEvent
from src.baselines.random_forest import RandomForestBaseline
from src.baselines.cnn_baseline import CNNImageBaseline
from src.baselines.lstm_baseline import LSTMLightCurveBaseline


class BaselineEvaluator:
    """Trains and benchmarks all Stage 2 baselines on the same known-class dataset splits."""

    def __init__(self, known_classes: List[str]):
        self.known_classes = sorted(known_classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.known_classes)}

    def evaluate_predictions(self, y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
        """Compute standard classification evaluation metrics."""
        y_pred = np.argmax(y_probs, axis=1)

        acc = float(accuracy_score(y_true, y_pred))
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

        # Multi-class One-vs-Rest AUROC
        try:
            auroc = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
        except Exception:
            auroc = 0.5

        return {
            "accuracy": acc,
            "precision_macro": float(p),
            "recall_macro": float(r),
            "f1_macro": float(f1),
            "auroc_ovr": auroc
        }

    def run_benchmark(self, train_events: List[AstronomicalEvent],
                      test_events: List[AstronomicalEvent],
                      device: str = "cpu",
                      output_file: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Fit and evaluate Random Forest, CNN, and LSTM baselines."""
        y_test_indices = np.array([self.class_to_idx.get(e.true_label, -1) for e in test_events])
        valid_mask = y_test_indices >= 0
        test_events_filtered = [e for i, e in enumerate(test_events) if valid_mask[i]]
        y_test = y_test_indices[valid_mask]

        results: Dict[str, Dict[str, float]] = {}

        # 1. Random Forest Baseline
        print("[Evaluating Baseline 1/3] Random Forest (Features)...")
        rf = RandomForestBaseline()
        rf.fit(train_events)
        rf_probs = rf.predict_proba(test_events_filtered)
        results["RandomForest"] = self.evaluate_predictions(y_test, rf_probs)

        # 2. CNN Image Baseline
        print("[Evaluating Baseline 2/3] CNN (Image Cutouts)...")
        cnn = CNNImageBaseline(num_classes=len(self.known_classes))
        cnn.fit(train_events, self.class_to_idx, epochs=10, device=device)
        cnn_probs = cnn.predict_proba(test_events_filtered, self.class_to_idx, device=device)
        results["CNN_Image"] = self.evaluate_predictions(y_test, cnn_probs)

        # 3. LSTM Light Curve Baseline
        print("[Evaluating Baseline 3/3] Bi-LSTM (Light Curves)...")
        lstm = LSTMLightCurveBaseline(num_classes=len(self.known_classes))
        lstm.fit(train_events, self.class_to_idx, epochs=10, device=device)
        lstm_probs = lstm.predict_proba(test_events_filtered, self.class_to_idx, device=device)
        results["LSTM_LightCurve"] = self.evaluate_predictions(y_test, lstm_probs)

        # Print formatted summary table
        print("\n" + "=" * 65)
        print(f"{'Model Baseline':<20} | {'Accuracy':<10} | {'F1-Macro':<10} | {'AUROC':<10}")
        print("-" * 65)
        for model_name, metrics in results.items():
            print(f"{model_name:<20} | {metrics['accuracy']:<10.4f} | {metrics['f1_macro']:<10.4f} | {metrics['auroc_ovr']:<10.4f}")
        print("=" * 65 + "\n")

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

        return results
