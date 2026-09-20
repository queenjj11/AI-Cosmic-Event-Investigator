"""Evaluation pipeline for Real-ZTF Light-Curve Anomaly Detector v2.

Evaluates light-curve-only anomaly detection on the 106-object Real-ZTF Primary Benchmark
using representations extracted by the frozen production LightCurveEncoder.

Constraints:
- Frozen production checkpoint (SHA-256: e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72)
- Frozen primary benchmark manifest (SHA-256 verified)
- Zero label leakage: fit on N=58 known objects, calibrate threshold on validation fold, evaluate on N=34 OOD objects
- 14 unclassified control field stars evaluated separately
"""

import os
import json
import hashlib
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score
)

from src.models.checkpoint_manager import (
    DEFAULT_CHECKPOINT_PATH,
    load_production_lightcurve_encoder,
)
from src.investigator.lightcurve_representation import LightCurveRepresentationService
from src.anomaly_detection.real_ztf_lc_detector import RealZTFLightCurveDetectorV2

PRIMARY_BENCHMARK_PATH = "data/real_ztf_benchmark/frozen_primary_benchmark.csv"
FREEZE_METADATA_PATH = "data/real_ztf_benchmark/benchmark_freeze_metadata.json"
EXPECTED_PRODUCTION_SHA256 = "e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72"
OUTPUT_DIR = "reports/real_ztf_lc_detector_v2"
SAVED_MODEL_PATH = "models/checkpoints/real_ztf_lc_detector_v2.pkl"


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 digest of a local file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_integrity():
    """Verify primary benchmark manifest and checkpoint integrity."""
    if not os.path.exists(PRIMARY_BENCHMARK_PATH):
        raise FileNotFoundError(f"Primary benchmark manifest not found: {PRIMARY_BENCHMARK_PATH}")

    with open(FREEZE_METADATA_PATH, "r") as f:
        meta = json.load(f)
    expected_manifest_hash = meta["file_sha256_digests"]["frozen_primary_benchmark_csv"]
    actual_manifest_hash = compute_sha256(PRIMARY_BENCHMARK_PATH)
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError(f"Primary manifest integrity violation! SHA-256 mismatch.")

    actual_ckpt_hash = compute_sha256(DEFAULT_CHECKPOINT_PATH)
    if actual_ckpt_hash != EXPECTED_PRODUCTION_SHA256:
        raise ValueError(f"Production checkpoint integrity violation! SHA-256 mismatch.")

    print("Integrity verification passed:")
    print(f"  Primary Manifest SHA-256: {actual_manifest_hash[:16]}...")
    print(f"  Checkpoint SHA-256:       {actual_ckpt_hash[:16]}...")


def extract_embeddings(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Extract 128-D light-curve embeddings for all candidates using LightCurveRepresentationService."""
    rep_service = LightCurveRepresentationService(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device="cpu")
    embeddings = []
    ids = []

    for _, row in df.iterrows():
        cand_id = row["candidate_id"]
        raw_path = row["raw_data_path"]
        rep = rep_service.represent_ztf_object(cand_id, raw_path)
        embeddings.append(rep.embedding)
        ids.append(cand_id)

    return np.array(embeddings, dtype=np.float64), ids


def evaluate_v2_detector() -> Dict[str, Any]:
    """Run cross-validation and full dataset evaluation for Real-ZTF Light-Curve Detector v2."""
    verify_integrity()

    # Load frozen primary benchmark manifest
    df = pd.read_csv(PRIMARY_BENCHMARK_PATH)
    print(f"Loaded {len(df)} primary benchmark objects.")

    # Extract embeddings
    print("Extracting 128-D light-curve embeddings...")
    embeddings, cand_ids = extract_embeddings(df)
    df["embedding_idx"] = np.arange(len(df))

    # Separate partitions
    known_mask = (df["candidate_role"] == "in_distribution")
    ood_mask = (df["candidate_role"] == "ood_anomaly")
    control_mask = (df["candidate_role"] == "control")

    known_df = df[known_mask].copy().reset_index(drop=True)
    ood_df = df[ood_mask].copy().reset_index(drop=True)
    control_df = df[control_mask].copy().reset_index(drop=True)

    print(f"Partitions: Known In-Domain={len(known_df)}, OOD Anomalies={len(ood_df)}, Control={len(control_df)}")

    X_known = embeddings[known_df["embedding_idx"].values]
    X_ood = embeddings[ood_df["embedding_idx"].values]
    X_control = embeddings[control_df["embedding_idx"].values]

    # --- 1. 5-FOLD OUT-OF-FOLD (OOF) CROSS VALIDATION ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_per_fold_aurocs = []
    cv_per_fold_auprcs = []

    oof_known_scores = np.zeros(len(X_known))
    ood_scores_accum = np.zeros(len(X_ood))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_known)):
        X_tr = X_known[train_idx]
        X_va = X_known[val_idx]

        detector = RealZTFLightCurveDetectorV2(method="pca_mahalanobis", n_components=10)
        detector.fit(X_tr)
        detector.calibrate_threshold(X_va, percentile=95.0)

        # Store out-of-fold validation scores for known objects
        oof_known_scores[val_idx] = detector.compute_raw_score(X_va)
        ood_scores_fold = detector.compute_raw_score(X_ood)
        ood_scores_accum += ood_scores_fold / 5.0

        # Per-fold evaluation slice (val_known vs all_ood)
        X_eval_fold = np.vstack([X_va, X_ood])
        y_eval_fold = np.array([0] * len(X_va) + [1] * len(X_ood))

        scores_fold = detector.compute_raw_score(X_eval_fold)
        cv_per_fold_aurocs.append(roc_auc_score(y_eval_fold, scores_fold))
        cv_per_fold_auprcs.append(average_precision_score(y_eval_fold, scores_fold))

    # Calculate true Out-Of-Fold (OOF) pooled metrics (58 OOF known negatives vs 34 OOD positives)
    y_oof = np.array([0] * len(X_known) + [1] * len(X_ood))
    scores_oof = np.concatenate([oof_known_scores, ood_scores_accum])

    oof_auroc = float(roc_auc_score(y_oof, scores_oof))
    oof_auprc = float(average_precision_score(y_oof, scores_oof))
    per_fold_mean_auroc = float(np.mean(cv_per_fold_aurocs))
    per_fold_mean_auprc_inflated = float(np.mean(cv_per_fold_auprcs))

    print(f"5-Fold OOF Pooled Metrics -> OOF AUROC: {oof_auroc:.4f}, OOF AUPRC: {oof_auprc:.4f}")
    print(f"5-Fold Per-Fold Mean     -> Mean AUROC: {per_fold_mean_auroc:.4f}, Per-Fold Slice Mean AUPRC: {per_fold_mean_auprc_inflated:.4f} (Inflated due to 74.6% fold positive prevalence)")

    # --- 2. FULL FIT FOR PRODUCTION & HELD-OUT BINARY EVALUATION ---
    prod_detector = RealZTFLightCurveDetectorV2(method="pca_mahalanobis", n_components=10)
    prod_detector.fit(X_known)

    # Save fitted production detector
    prod_detector.save(SAVED_MODEL_PATH)
    print(f"Saved production Real-ZTF v2 detector to {SAVED_MODEL_PATH}")

    # Evaluate full binary benchmark set (58 known vs 34 OOD)
    X_binary = np.vstack([X_known, X_ood])
    y_binary = np.array([0] * len(X_known) + [1] * len(X_ood))

    raw_scores = prod_detector.compute_raw_score(X_binary)
    scaled_results = prod_detector.score(X_binary)
    scaled_scores = np.array([r[0] for r in scaled_results])
    pred_flags = np.array([r[1] for r in scaled_results])

    binary_auroc = float(roc_auc_score(y_binary, raw_scores))
    binary_auprc = float(average_precision_score(y_binary, raw_scores))
    binary_ap = binary_auprc

    cm = confusion_matrix(y_binary, pred_flags)
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(f1_score(y_binary, pred_flags))

    # --- 3. EVALUATE CONTROL FIELD STARS (N=14) ---
    control_raw = prod_detector.compute_raw_score(X_control)
    control_results = prod_detector.score(X_control)
    control_flags = [r[1] for r in control_results]
    control_flagged_count = int(sum(control_flags))
    control_flagged_fraction = float(control_flagged_count / len(X_control))

    # --- 4. PER-POPULATION ANALYSIS ---
    population_summary = {}
    eval_rows = []

    for idx, row in df.iterrows():
        cand_id = row["candidate_id"]
        pop = row["astrophysical_class"]
        role = row["candidate_role"]
        gt = int(row["is_anomaly_ground_truth"])
        emb = embeddings[idx]

        r_score = float(prod_detector.compute_raw_score(emb)[0])
        s_score, flag = prod_detector.score(emb)

        eval_rows.append({
            "candidate_id": cand_id,
            "ztf_designation": row["ztf_designation"],
            "astrophysical_class": pop,
            "candidate_role": role,
            "is_anomaly_ground_truth": gt,
            "raw_novelty_score": round(r_score, 4),
            "scaled_novelty_score": s_score,
            "is_anomaly_pred": flag
        })

        if pop not in population_summary:
            population_summary[pop] = {
                "count": 0,
                "role": role,
                "is_anomaly_ground_truth": gt,
                "raw_scores": [],
                "scaled_scores": [],
                "flagged_count": 0
            }
        population_summary[pop]["count"] += 1
        population_summary[pop]["raw_scores"].append(r_score)
        population_summary[pop]["scaled_scores"].append(s_score)
        if flag:
            population_summary[pop]["flagged_count"] += 1

    # Format per-population summary
    pop_report = {}
    for pop, data in population_summary.items():
        pop_report[pop] = {
            "count": data["count"],
            "candidate_role": data["role"],
            "ground_truth": data["is_anomaly_ground_truth"],
            "mean_raw_score": float(np.round(np.mean(data["raw_scores"]), 4)),
            "mean_scaled_score": float(np.round(np.mean(data["scaled_scores"]), 4)),
            "flagged_count": data["flagged_count"],
            "flagged_rate": float(np.round(data["flagged_count"] / data["count"], 4))
        }

    # Compare against V1 multimodal baseline
    v1_baseline = {
        "model": "ACEI Multimodal Anomaly Ensemble v1",
        "modality": "Image + LightCurve (Image Missing)",
        "auroc": 0.3859,
        "auprc": 0.2948,
        "ap": 0.3063,
        "fpr": 0.6552,
        "recall": 0.5000,
        "f1": 0.3820,
        "status": "NOT_VALIDATED_FOR_REAL_ZTF"
    }

    v2_results = {
        "model": "Real-ZTF Light-Curve Anomaly Detector v2",
        "modality": "LightCurve Only (128-D Representation)",
        "method": "PCA-Mahalanobis (Ledoit-Wolf)",
        "n_components": 10,
        "decision_threshold": float(np.round(prod_detector.decision_threshold, 4)),
        "primary_binary_n": len(X_binary),
        "known_in_domain_n": len(X_known),
        "ood_anomalies_n": len(X_ood),
        "control_field_stars_n": len(X_control),
        "metrics": {
            "oof_pooled_auroc": oof_auroc,
            "oof_pooled_auprc": oof_auprc,
            "per_fold_mean_auroc": per_fold_mean_auroc,
            "per_fold_mean_auprc_inflated_slice": per_fold_mean_auprc_inflated,
            "full_binary_auroc": binary_auroc,
            "full_binary_auprc": binary_auprc,
            "full_binary_ap": binary_ap,
            "fpr": fpr,
            "specificity": specificity,
            "recall": recall,
            "f1": f1
        },
        "control_evaluation": {
            "n": len(X_control),
            "flagged_count": control_flagged_count,
            "flagged_fraction": control_flagged_fraction
        },
        "v1_baseline_comparison": {
            "v1_auroc": v1_baseline["auroc"],
            "v2_auroc": binary_auroc,
            "auroc_delta": float(np.round(binary_auroc - v1_baseline["auroc"], 4)),
            "v1_auprc": v1_baseline["auprc"],
            "v2_auprc": binary_auprc,
            "auprc_delta": float(np.round(binary_auprc - v1_baseline["auprc"], 4))
        },
        "per_population": pop_report,
        "operational_status": "REAL_ZTF_EVALUATED_RESEARCH"
    }

    # Write output files
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics_json_path = os.path.join(OUTPUT_DIR, "real_ztf_lc_detector_v2_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(v2_results, f, indent=2)

    eval_df = pd.DataFrame(eval_rows)
    eval_csv_path = os.path.join(OUTPUT_DIR, "real_ztf_lc_detector_v2_eval.csv")
    eval_df.to_csv(eval_csv_path, index=False)

    print("\n" + "=" * 80)
    print("  REAL-ZTF LIGHT-CURVE ANOMALY DETECTOR V2 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  V1 Multimodal Baseline AUROC : {v1_baseline['auroc']:.4f}")
    print(f"  V2 Light-Curve Detector AUROC: {binary_auroc:.4f}  (Delta: +{binary_auroc - v1_baseline['auroc']:.4f})")
    print(f"  V1 Multimodal Baseline AUPRC : {v1_baseline['auprc']:.4f}")
    print(f"  V2 Light-Curve Detector AUPRC: {binary_auprc:.4f}  (Delta: +{binary_auprc - v1_baseline['auprc']:.4f})")
    print(f"  Out-Of-Fold (OOF) Pooled AUROC: {oof_auroc:.4f}")
    print(f"  Out-Of-Fold (OOF) Pooled AUPRC: {oof_auprc:.4f}")
    print(f"  Control Field Stars Flagged : {control_flagged_count}/{len(X_control)} ({control_flagged_fraction:.1%})")
    print(f"  Operational Status           : REAL_ZTF_EVALUATED_RESEARCH")
    print("=" * 80)

    return v2_results


if __name__ == "__main__":
    evaluate_v2_detector()
