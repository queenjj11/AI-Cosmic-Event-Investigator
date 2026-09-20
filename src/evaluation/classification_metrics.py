"""Classification evaluation metrics: Multi-class Accuracy, Precision, Recall, F1, and AUROC."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


def compute_classification_metrics(y_true: np.ndarray, y_probs: np.ndarray,
                                   class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Computes standard multi-class classification metrics.
    y_true: (N,) integer ground truth labels
    y_probs: (N, C) predicted class probabilities
    """
    y_pred = np.argmax(y_probs, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    try:
        auroc = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
    except Exception:
        auroc = 0.5

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "f1_weighted": float(f1_w),
        "auroc_ovr": auroc,
        "confusion_matrix": cm,
        "class_names": class_names or [str(i) for i in range(y_probs.shape[1])]
    }
