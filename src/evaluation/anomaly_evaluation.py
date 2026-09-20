"""Quantitative anomaly evaluation pipeline for the ACEI synthetic benchmark.

Evaluates generalization across multiple known transient classes (SN_Ia, SN_II,
Stellar_Flare, Variable_Star) and held-out rare anomalous classes (LRN, SLSN, TDE).
Supports both programmatic API and CLI execution (`python -m src.evaluation.anomaly_evaluation`).
"""

import os
import csv
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score

from src.pipeline.acei_pipeline import ACEIPipeline
from src.data.dataset_builder import generate_benchmark_events
from src.data.splitter import ObjectLevelSplitter
from src.data.schema import AstronomicalEvent


@dataclass
class ObjectEvaluationRecord:
    object_id: str
    true_label: str
    ground_truth_anomaly: int
    anomaly_score: float
    predicted_anomaly: bool
    classifier_prediction: str
    classifier_confidence: float
    ae_norm: float
    mahalanobis_norm: float
    energy_norm: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate binary classification / detection metrics at fixed threshold."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    total = tp + tn + fp + fn

    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tpr": recall,
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1
    }


def calculate_curve_metrics(y_true: np.ndarray, continuous_scores: np.ndarray) -> Dict[str, float]:
    """Calculate continuous ranking metrics: AUROC and AUPRC."""
    try:
        auroc = float(roc_auc_score(y_true, continuous_scores))
    except Exception:
        auroc = 0.5

    try:
        precisions, recalls, _ = precision_recall_curve(y_true, continuous_scores)
        auprc = float(auc(recalls, precisions))
        ap = float(average_precision_score(y_true, continuous_scores))
    except Exception:
        auprc = 0.5
        ap = 0.5

    return {
        "auroc": auroc,
        "auprc": auprc,
        "average_precision": ap
    }


def calculate_distribution_stats(scores: List[float]) -> Dict[str, float]:
    """Calculate count, mean, median, std, min, max for a list of scores."""
    if not scores:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }


class AnomalyEvaluatorPipeline:
    """End-to-end evaluation pipeline that runs object-level evaluation on the ACEI benchmark."""

    def __init__(self,
                 pipeline: Optional[ACEIPipeline] = None,
                 threshold: float = 0.65,
                 seed: int = 42):
        self.threshold = threshold
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = ACEIPipeline(anomaly_threshold=threshold)
            self.pipeline.initialize_system()

    def evaluate(self,
                 num_known: int = 120,
                 num_anomalies: int = 30) -> Dict[str, Any]:
        """
        Execute full benchmark evaluation across test partitions.
        Strict constraint: Test events are evaluated through `pipeline.investigate_event`
        without any ground-truth labels influencing inference.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # 1. Generate benchmark events and partition using ObjectLevelSplitter
        benchmark_events = generate_benchmark_events(num_known=num_known, num_anomalies=num_anomalies)
        splitter = ObjectLevelSplitter(self.pipeline.known_classes, ["LRN", "SLSN", "TDE"], seed=self.seed)
        splits = splitter.split_events(benchmark_events)

        test_known_events = [e for e in benchmark_events if e.object_id in splits.test_known_ids]
        test_anom_events = [e for e in benchmark_events if e.object_id in splits.test_anomaly_ids]
        all_test_events = test_known_events + test_anom_events

        # 2. Run inference on every test object (isolated from ground truth)
        records: List[ObjectEvaluationRecord] = []
        for event in all_test_events:
            # Model inference receives only the event observation
            res = self.pipeline.investigate_event(event)

            # Ground truth is collected strictly post-inference for evaluation metrics
            gt_anomaly = 1 if event.is_anomaly else 0
            ae_norm = float(res.detector_scores.get("autoencoder_norm", 0.0))
            mah_norm = float(res.detector_scores.get("mahalanobis_norm", 0.0))
            energy_norm = float(res.detector_scores.get("energy_norm", 0.0))

            records.append(ObjectEvaluationRecord(
                object_id=res.event_id,
                true_label=event.true_label or "Unknown",
                ground_truth_anomaly=gt_anomaly,
                anomaly_score=float(res.anomaly_score),
                predicted_anomaly=bool(res.anomaly_score >= self.threshold),
                classifier_prediction=res.classifier_prediction,
                classifier_confidence=float(res.classifier_confidence),
                ae_norm=ae_norm,
                mahalanobis_norm=mah_norm,
                energy_norm=energy_norm
            ))

        # 3. Overall Metric Calculations
        y_true = np.array([r.ground_truth_anomaly for r in records], dtype=int)
        y_score = np.array([r.anomaly_score for r in records], dtype=float)
        y_pred = np.array([1 if r.predicted_anomaly else 0 for r in records], dtype=int)

        binary_metrics = calculate_binary_metrics(y_true, y_pred)
        curve_metrics = calculate_curve_metrics(y_true, y_score)

        # 4. Detector Ablation Analysis
        detectors = {
            "Autoencoder": np.array([r.ae_norm for r in records], dtype=float),
            "Mahalanobis": np.array([r.mahalanobis_norm for r in records], dtype=float),
            "Energy": np.array([r.energy_norm for r in records], dtype=float),
            "Ensemble": y_score
        }

        ablation_results = {}
        for det_name, scores in detectors.items():
            det_curve = calculate_curve_metrics(y_true, scores)
            det_pred = (scores >= self.threshold).astype(int)
            det_bin = calculate_binary_metrics(y_true, det_pred)
            ablation_results[det_name] = {
                "auroc": det_curve["auroc"],
                "auprc": det_curve["auprc"],
                "fpr": det_bin["fpr"],
                "tpr": det_bin["tpr"],
                "precision": det_bin["precision"],
                "f1": det_bin["f1"]
            }

        # 5. Per-Anomaly-Class Breakdown
        anomaly_classes = ["LRN", "SLSN", "TDE"]
        per_class_results = {}
        for cls_name in anomaly_classes:
            cls_records = [r for r in records if r.true_label == cls_name]
            cls_n = len(cls_records)
            cls_scores = [r.anomaly_score for r in cls_records]
            cls_detected = sum(1 for r in cls_records if r.predicted_anomaly)
            cls_recall = float(cls_detected / cls_n) if cls_n > 0 else 0.0
            dist = calculate_distribution_stats(cls_scores)

            per_class_results[cls_name] = {
                "n": cls_n,
                "detected": cls_detected,
                "recall": cls_recall,
                "mean_score": dist["mean"],
                "median_score": dist["median"],
                "std_score": dist["std"],
                "min_score": dist["min"],
                "max_score": dist["max"]
            }

        # Known in-distribution false-positive rate
        known_records = [r for r in records if r.ground_truth_anomaly == 0]
        known_n = len(known_records)
        known_flagged = sum(1 for r in known_records if r.predicted_anomaly)
        known_fpr = float(known_flagged / known_n) if known_n > 0 else 0.0
        known_dist = calculate_distribution_stats([r.anomaly_score for r in known_records])

        # Held-out anomaly distribution
        anom_records = [r for r in records if r.ground_truth_anomaly == 1]
        anom_dist = calculate_distribution_stats([r.anomaly_score for r in anom_records])

        # 6. Score Distribution Diagnostics
        score_distributions = {
            "known_events": known_dist,
            "all_anomalies": anom_dist,
            "LRN": {
                "count": per_class_results["LRN"]["n"],
                "mean": per_class_results["LRN"]["mean_score"],
                "median": per_class_results["LRN"]["median_score"],
                "std": per_class_results["LRN"]["std_score"],
                "min": per_class_results["LRN"]["min_score"],
                "max": per_class_results["LRN"]["max_score"],
            },
            "SLSN": {
                "count": per_class_results["SLSN"]["n"],
                "mean": per_class_results["SLSN"]["mean_score"],
                "median": per_class_results["SLSN"]["median_score"],
                "std": per_class_results["SLSN"]["std_score"],
                "min": per_class_results["SLSN"]["min_score"],
                "max": per_class_results["SLSN"]["max_score"],
            },
            "TDE": {
                "count": per_class_results["TDE"]["n"],
                "mean": per_class_results["TDE"]["mean_score"],
                "median": per_class_results["TDE"]["median_score"],
                "std": per_class_results["TDE"]["std_score"],
                "min": per_class_results["TDE"]["min_score"],
                "max": per_class_results["TDE"]["max_score"],
            }
        }

        evaluation_data = {
            "threshold": self.threshold,
            "seed": self.seed,
            "num_test_events": len(records),
            "num_known_test_events": known_n,
            "num_anomaly_test_events": len(anom_records),
            "binary_metrics": binary_metrics,
            "curve_metrics": curve_metrics,
            "ablation": ablation_results,
            "per_class": per_class_results,
            "known_test_fpr": known_fpr,
            "known_flagged_count": known_flagged,
            "score_distributions": score_distributions,
            "records": [r.to_dict() for r in records]
        }

        return evaluation_data

    def save_reports(self, eval_data: Dict[str, Any], reports_dir: str = "reports") -> Tuple[str, str, str]:
        """Save evaluation outputs to CSV, JSON, and formatted TXT."""
        os.makedirs(reports_dir, exist_ok=True)
        csv_path = os.path.join(reports_dir, "anomaly_evaluation.csv")
        json_path = os.path.join(reports_dir, "anomaly_metrics.json")
        txt_path = os.path.join(reports_dir, "anomaly_evaluation.txt")

        # 1. Save CSV
        records = eval_data["records"]
        if records:
            fieldnames = list(records[0].keys())
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)

        # 2. Save JSON (exclude full records to keep metrics summary compact)
        metrics_summary = {k: v for k, v in eval_data.items() if k != "records"}
        with open(json_path, mode="w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)

        # 3. Save Formatted TXT Report
        txt_content = self.generate_text_report(eval_data)
        with open(txt_path, mode="w", encoding="utf-8") as f:
            f.write(txt_content)

        return csv_path, json_path, txt_path

    @staticmethod
    def generate_text_report(eval_data: Dict[str, Any]) -> str:
        """Render clean, structured diagnostic report as text."""
        bm = eval_data["binary_metrics"]
        cm = eval_data["curve_metrics"]
        abl = eval_data["ablation"]
        pc = eval_data["per_class"]
        sd = eval_data["score_distributions"]

        lines = []
        lines.append("=" * 80)
        lines.append("       ACEI SYNTHETIC BENCHMARK: QUANTITATIVE ANOMALY EVALUATION")
        lines.append("=" * 80)
        lines.append(f"Fixed Threshold:       {eval_data['threshold']:.2f}")
        lines.append(f"Random Seed:           {eval_data['seed']}")
        lines.append(f"Total Test Events:     {eval_data['num_test_events']}")
        lines.append(f"  - Known Test Events: {eval_data['num_known_test_events']}")
        lines.append(f"  - Anomaly Test Events:{eval_data['num_anomaly_test_events']}")
        lines.append("")

        lines.append("--------------------------------------------------------------------------------")
        lines.append("1. OVERALL ANOMALY ENSEMBLE METRICS (Threshold = 0.65)")
        lines.append("--------------------------------------------------------------------------------")
        lines.append(f"Continuous AUROC:      {cm['auroc']:.4f}")
        lines.append(f"Continuous AUPRC:      {cm['auprc']:.4f}")
        lines.append(f"Average Precision:     {cm['average_precision']:.4f}")
        lines.append(f"Accuracy:              {bm['accuracy']:.4f}")
        lines.append(f"Precision:             {bm['precision']:.4f}")
        lines.append(f"Recall / TPR:          {bm['recall']:.4f} ({bm['tp']} / {bm['tp'] + bm['fn']} anomalies detected)")
        lines.append(f"Specificity:           {bm['specificity']:.4f}")
        lines.append(f"False Positive Rate:   {bm['fpr']:.4f} ({bm['fp']} / {bm['fp'] + bm['tn']} known events flagged)")
        lines.append(f"F1 Score:              {bm['f1']:.4f}")
        lines.append(f"Confusion Matrix:      TP={bm['tp']}, FP={bm['fp']}, TN={bm['tn']}, FN={bm['fn']}")
        lines.append("")

        lines.append("--------------------------------------------------------------------------------")
        lines.append("2. DETECTOR ABLATION REPORT")
        lines.append("--------------------------------------------------------------------------------")
        header = f"{'Detector':<20} | {'AUROC':<7} | {'AUPRC':<7} | {'FPR @0.65':<10} | {'TPR @0.65':<10} | {'F1 @0.65':<8}"
        lines.append(header)
        lines.append("-" * len(header))
        for det_name in ["Autoencoder", "Mahalanobis", "Energy", "Ensemble"]:
            d = abl[det_name]
            lines.append(f"{det_name:<20} | {d['auroc']:<7.4f} | {d['auprc']:<7.4f} | {d['fpr']:<10.4f} | {d['tpr']:<10.4f} | {d['f1']:<8.4f}")
        lines.append("")

        lines.append("--------------------------------------------------------------------------------")
        lines.append("3. PER-ANOMALY-CLASS DETECTION BREAKDOWN")
        lines.append("--------------------------------------------------------------------------------")
        header_pc = f"{'Anomaly Class':<15} | {'N':<4} | {'Mean Score':<11} | {'Median Score':<12} | {'Detected @0.65':<15} | {'Recall':<8}"
        lines.append(header_pc)
        lines.append("-" * len(header_pc))
        for cls_name in ["LRN", "SLSN", "TDE"]:
            p = pc[cls_name]
            lines.append(f"{cls_name:<15} | {p['n']:<4} | {p['mean_score']:<11.4f} | {p['median_score']:<12.4f} | {p['detected']:<15} | {p['recall']*100:<7.1f}%")
        lines.append(f"\nKnown Test Events False Positive Rate: {eval_data['known_test_fpr']*100:.1f}% ({eval_data['known_flagged_count']}/{eval_data['num_known_test_events']})")
        lines.append("")

        lines.append("--------------------------------------------------------------------------------")
        lines.append("4. CALIBRATION & SCORE DISTRIBUTION DIAGNOSTICS")
        lines.append("--------------------------------------------------------------------------------")
        header_dist = f"{'Partition / Class':<20} | {'N':<4} | {'Mean':<8} | {'Median':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8}"
        lines.append(header_dist)
        lines.append("-" * len(header_dist))
        for key, name in [
            ("known_events", "Known Test (In-Dist)"),
            ("all_anomalies", "All Anomalies (OOD)"),
            ("LRN", "LRN (Merger)"),
            ("SLSN", "SLSN (Magnetar)"),
            ("TDE", "TDE (Black Hole)")
        ]:
            s = sd[key]
            lines.append(f"{name:<20} | {s['count']:<4} | {s['mean']:<8.4f} | {s['median']:<8.4f} | {s['std']:<8.4f} | {s['min']:<8.4f} | {s['max']:<8.4f}")
        lines.append("=" * 80)

        return "\n".join(lines)


def run_anomaly_evaluation(reports_dir: str = "reports", seed: int = 42) -> Dict[str, Any]:
    """Convenience function to run evaluation and save reports."""
    evaluator = AnomalyEvaluatorPipeline(seed=seed)
    eval_data = evaluator.evaluate()
    evaluator.save_reports(eval_data, reports_dir=reports_dir)
    return eval_data


def main():
    parser = argparse.ArgumentParser(description="ACEI Quantitative Anomaly Evaluation Pipeline")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory to save evaluation reports")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation determinism")
    args = parser.parse_args()

    print("[ACEI Evaluation] Bootstrapping pipeline and running evaluation benchmark...")
    evaluator = AnomalyEvaluatorPipeline(seed=args.seed)
    eval_data = evaluator.evaluate()
    csv_p, json_p, txt_p = evaluator.save_reports(eval_data, reports_dir=args.reports_dir)

    print("\n" + AnomalyEvaluatorPipeline.generate_text_report(eval_data))
    print(f"\n[ACEI Evaluation] Reports successfully saved:")
    print(f"  - CSV:  {csv_p}")
    print(f"  - JSON: {json_p}")
    print(f"  - TXT:  {txt_p}")


if __name__ == "__main__":
    main()
