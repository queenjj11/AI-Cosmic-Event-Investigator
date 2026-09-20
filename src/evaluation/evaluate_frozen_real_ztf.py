"""Controlled zero-shot evaluation pipeline for the frozen Real-ZTF Primary Benchmark.

Evaluates the frozen production checkpoint on the 106-object primary benchmark.
Strict constraints:
- Evaluates ONLY data/real_ztf_benchmark/frozen_primary_benchmark.csv
- Verifies manifest SHA-256 against benchmark_freeze_metadata.json
- Verifies production checkpoint SHA-256 before loading
- Frozen evaluation: model.eval(), torch.no_grad()
- ZERO training, backprop, calibration, or threshold tuning on real-ZTF data
- Anomaly detector reference distributions fitted strictly on synthetic train/val data
- Generates 128-D lightcurve embeddings, checks numerical stability, computes metrics
"""

import os
import json
import hashlib
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score
)

from src.models.checkpoint_manager import (
    DEFAULT_CHECKPOINT_PATH,
    load_production_model,
    load_production_lightcurve_encoder,
)
from src.models.multimodal_model import MultimodalTransientModel
from src.data.real_ztf_preprocessing import RealZTFPreprocessor
from src.data.dataset_builder import generate_benchmark_events, AstronomicalDataset
from src.data.splitter import ObjectLevelSplitter
from src.anomaly_detection.autoencoder import MultimodalAutoencoder
from src.anomaly_detection.mahalanobis import MahalanobisDetector
from src.anomaly_detection.energy_score import EnergyOODDetector
from src.anomaly_detection.anomaly_ensemble import AnomalyEnsemble


PRIMARY_BENCHMARK_PATH = "data/real_ztf_benchmark/frozen_primary_benchmark.csv"
FREEZE_METADATA_PATH = "data/real_ztf_benchmark/benchmark_freeze_metadata.json"
EXPECTED_PRODUCTION_SHA256 = "e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72"
SECONDARY_BENCHMARK_PATH = "data/real_ztf_benchmark/frozen_secondary_limited.csv"
HISTORICAL_PILOT_IDS = {
    "ZTF18abukavn",
    "ZTF19abjrhbe",
    "ZTF20aaelulu",
    "ZTF18aabtxvd",
    "Gaia DR3 2028869231302243712",
}

OUTPUT_EMBEDDINGS_CSV = "data/real_ztf_benchmark/real_ztf_primary_embeddings.csv"
OUTPUT_EMBEDDINGS_NPY = "data/real_ztf_benchmark/real_ztf_primary_embeddings.npy"
OUTPUT_EVALUATION_CSV = "data/real_ztf_benchmark/real_ztf_primary_evaluation.csv"
OUTPUT_CONFUSION_CSV = "reports/real_ztf_primary_confusion_matrix.csv"
OUTPUT_ROC_CSV = "reports/real_ztf_primary_roc.csv"
OUTPUT_PR_CSV = "reports/real_ztf_primary_pr.csv"
OUTPUT_REPORT_MD = "reports/real_ztf_primary_evaluation_report.md"


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 digest of a local file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_benchmark_integrity(
    manifest_path: str = PRIMARY_BENCHMARK_PATH,
    metadata_path: str = FREEZE_METADATA_PATH
) -> str:
    """Verify primary benchmark manifest integrity against freeze metadata."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Primary benchmark manifest not found: {manifest_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Freeze metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    expected_manifest_hash = meta["file_sha256_digests"]["frozen_primary_benchmark_csv"]
    actual_manifest_hash = compute_sha256(manifest_path)

    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError(
            f"Benchmark integrity violation! Primary manifest SHA-256 mismatch.\n"
            f"Expected: {expected_manifest_hash}\n"
            f"Actual:   {actual_manifest_hash}"
        )

    return actual_manifest_hash


def verify_checkpoint_integrity(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    expected_sha256: str = EXPECTED_PRODUCTION_SHA256
) -> str:
    """Verify production model checkpoint SHA-256 before loading."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Production checkpoint not found: {checkpoint_path}")

    actual_ckpt_hash = compute_sha256(checkpoint_path)
    if actual_ckpt_hash != expected_sha256:
        raise ValueError(
            f"Checkpoint integrity violation! Checkpoint SHA-256 mismatch.\n"
            f"Expected: {expected_sha256}\n"
            f"Actual:   {actual_ckpt_hash}"
        )

    return actual_ckpt_hash


def setup_frozen_anomaly_pipeline(
    model: MultimodalTransientModel,
    payload: Dict[str, Any],
    seed: int = 42,
    threshold: float = 0.65
) -> AnomalyEnsemble:
    """
    Setup the production anomaly ensemble with reference distributions fitted
    STRICTLY on the synthetic benchmark training and validation partitions.

    Zero real-ZTF data is used for training, calibration, or reference fitting.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    class_to_idx = payload["class_to_idx"]
    known_classes = list(payload["dataset_provenance"]["known_classes"])
    held_out_classes = list(payload["dataset_provenance"]["held_out_classes"])

    # 1. Generate canonical synthetic benchmark events
    benchmark_events = generate_benchmark_events(
        num_known=120,
        num_anomalies=30,
        known_classes=known_classes,
        anomaly_classes=held_out_classes
    )

    # 2. Object-level split (70% train, 15% val, 15% test)
    splitter = ObjectLevelSplitter(
        known_classes=known_classes,
        held_out_classes=held_out_classes,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed
    )
    splits = splitter.split_events(benchmark_events)

    train_events = [e for e in benchmark_events if e.object_id in splits.train_ids]
    val_events = [e for e in benchmark_events if e.object_id in splits.val_ids]

    # 3. Extract synthetic train embeddings with frozen model
    train_dataset = AstronomicalDataset(train_events, class_to_idx)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
    train_batch = next(iter(train_loader))

    with torch.no_grad():
        train_logits, train_embeds = model(
            train_batch["image"], train_batch["lightcurve"], train_batch["mask"]
        )

    np_train_embeds = train_embeds.numpy()
    np_train_labels = train_batch["label"].numpy()

    # 4. Fit Autoencoder strictly on normal synthetic training embeddings
    torch.manual_seed(seed)
    np.random.seed(seed)
    autoencoder = MultimodalAutoencoder(input_dim=256, latent_dim=64)
    autoencoder.fit(np_train_embeds, epochs=25, device="cpu")

    # 5. Fit Mahalanobis strictly on normal synthetic training embeddings and labels
    mahalanobis = MahalanobisDetector()
    mahalanobis.fit(np_train_embeds, np_train_labels)

    # 6. Energy detector (no fitting needed)
    energy = EnergyOODDetector(temperature=1.0)

    # 7. Extract synthetic val embeddings and logits for calibration
    val_dataset = AstronomicalDataset(val_events, class_to_idx)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))
    val_batch = next(iter(val_loader))

    with torch.no_grad():
        val_logits, val_embeds = model(
            val_batch["image"], val_batch["lightcurve"], val_batch["mask"]
        )

    # 8. Calibrate anomaly ensemble on held-out synthetic validation set
    ensemble = AnomalyEnsemble(autoencoder, mahalanobis, energy, threshold=threshold)
    ensemble.calibrate(val_embeds.numpy(), val_logits.numpy())

    return ensemble


def run_frozen_real_ztf_evaluation(
    benchmark_path: str = PRIMARY_BENCHMARK_PATH,
    metadata_path: str = FREEZE_METADATA_PATH,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    raw_data_dir: str = "data/real_ztf_benchmark/raw",
    threshold: float = 0.65,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Execute controlled zero-shot evaluation of the frozen 106-object Real-ZTF Primary Benchmark.
    """
    print("=" * 80)
    print("  CONTROLLED ZERO-SHOT REAL-ZTF MODEL EVALUATION")
    print("=" * 80)

    # --- TASK 1: VERIFY INTEGRITY AND LOAD CHECKPOINT ---
    print("\n[Step 1] Verifying manifest and checkpoint integrity...")
    manifest_hash = verify_benchmark_integrity(benchmark_path, metadata_path)
    ckpt_hash_before = verify_checkpoint_integrity(checkpoint_path, EXPECTED_PRODUCTION_SHA256)
    print(f"  Primary Manifest SHA-256 Verified: {manifest_hash}")
    print(f"  Production Checkpoint SHA-256:      {ckpt_hash_before}")

    # Load frozen model and standalone encoder
    model, payload = load_production_model(checkpoint_path, device="cpu")
    model.eval()
    lc_encoder = model.lc_encoder
    lc_encoder.eval()

    # Verify provenance assertion
    assert payload["real_ztf_data_used"] is False, "Violation: Checkpoint payload indicates real data was used!"
    print("  Checkpoint provenance confirmed: real_ztf_data_used == False")

    # Snapshot initial parameters to verify strict parameter invariance
    initial_model_params = {n: p.clone() for n, p in model.named_parameters()}
    initial_encoder_params = {n: p.clone() for n, p in lc_encoder.named_parameters()}

    # Load primary benchmark manifest
    primary_df = pd.read_csv(benchmark_path)
    num_objects = len(primary_df)
    if num_objects != 106:
        raise ValueError(f"Expected exactly 106 primary objects, but found {num_objects}")
    print(f"  Loaded exactly {num_objects} frozen primary benchmark objects.")

    # Verify no secondary-tier or historical pilot candidates appear
    if os.path.exists(SECONDARY_BENCHMARK_PATH):
        sec_df = pd.read_csv(SECONDARY_BENCHMARK_PATH)
        sec_ids = set(sec_df["candidate_id"].tolist())
        colliding = set(primary_df["candidate_id"]).intersection(sec_ids)
        if colliding:
            raise ValueError(f"Leakage violation: secondary-tier IDs in primary benchmark: {colliding}")

    # Check for pilot object designations
    pilot_collision = set(primary_df["ztf_designation"]).intersection(HISTORICAL_PILOT_IDS)
    if pilot_collision:
        raise ValueError(f"Leakage violation: historical transfer-pilot objects in benchmark: {pilot_collision}")

    # --- TASK 4: SETUP FROZEN ANOMALY DETECTION PIPELINE ---
    print("\n[Step 2] Setting up frozen anomaly ensemble using synthetic reference data...")
    ensemble = setup_frozen_anomaly_pipeline(model, payload, seed=seed, threshold=threshold)
    print(f"  Calibration parameters learned from synthetic validation set:")
    for sig_name, params in ensemble.calibration_params.items():
        print(f"    - {sig_name:<15}: median={params[0]:.6f}, scale={params[1]:.6f}")

    # --- TASK 2 & 3: LIGHT-CURVE EMBEDDING EXTRACTION & STABILITY CHECK ---
    print("\n[Step 3] Extracting embeddings and running inference across 106 objects...")
    preprocessor = RealZTFPreprocessor(max_sequence_length=50)

    embeddings_matrix = np.zeros((num_objects, 128), dtype=np.float32)
    embedding_records = []
    evaluation_records = []

    for idx, row in primary_df.iterrows():
        cid = row["candidate_id"]
        ztf_id = row["ztf_designation"]
        ast_cls = row["astrophysical_class"]
        role = row["dataset_role"]
        split = row["dataset_split"]
        gt_anom = int(row["is_anomaly_ground_truth"])

        raw_csv = os.path.join(raw_data_dir, cid, "raw_irsa.csv")
        if not os.path.exists(raw_csv):
            raise FileNotFoundError(f"Missing raw IRSA photometry for {cid}: {raw_csv}")

        raw_df = pd.read_csv(raw_csv)
        records = raw_df.to_dict("records")
        prep_event = preprocessor.process_records(records, object_id=cid)

        feat_tensor = prep_event.feature_tensor.unsqueeze(0)  # (1, 50, 4)
        mask_tensor = prep_event.mask_tensor.unsqueeze(0)     # (1, 50)
        img_tensor = torch.zeros(1, 3, 64, 64)                # Zero image for photometric transient

        with torch.no_grad():
            emb_tensor = lc_encoder(feat_tensor, mask=mask_tensor)
            logits_tensor, fused_tensor = model(img_tensor, feat_tensor, mask_tensor)

        emb_vec = emb_tensor.squeeze(0).cpu().numpy()
        embeddings_matrix[idx] = emb_vec

        emb_norm = float(np.linalg.norm(emb_vec))
        is_finite = bool(np.isfinite(emb_vec).all())

        embedding_records.append({
            "candidate_id": cid,
            "ztf_designation": ztf_id,
            "astrophysical_class": ast_cls,
            "dataset_role": role,
            "dataset_split": split,
            "is_anomaly_ground_truth": gt_anom,
            "valid_token_count": prep_event.valid_token_count,
            "padding_fraction": round((50 - prep_event.valid_token_count) / 50.0, 4),
            "embedding_norm": emb_norm,
            "finite_embedding": is_finite,
            "embedding_path": OUTPUT_EMBEDDINGS_NPY
        })

        fused_vec = fused_tensor.squeeze(0).cpu().numpy()
        logits_vec = logits_tensor.squeeze(0).cpu().numpy()

        score, is_flagged, det_scores = ensemble.score_event(fused_vec, logits_vec)

        evaluation_records.append({
            "candidate_id": cid,
            "ztf_designation": ztf_id,
            "astrophysical_class": ast_cls,
            "population_family": row["population_family"],
            "dataset_role": role,
            "dataset_split": split,
            "is_anomaly_ground_truth": gt_anom,
            "valid_token_count": prep_event.valid_token_count,
            "padding_fraction": round((50 - prep_event.valid_token_count) / 50.0, 4),
            "ae_score": float(det_scores["autoencoder_norm"]),
            "mahalanobis_score": float(det_scores["mahalanobis_norm"]),
            "energy_score": float(det_scores["energy_norm"]),
            "ensemble_score": float(score),
            "threshold": float(threshold),
            "is_flagged": bool(is_flagged),
            "embedding_norm": emb_norm
        })

    # Save embeddings
    emb_df = pd.DataFrame(embedding_records)
    os.makedirs(os.path.dirname(OUTPUT_EMBEDDINGS_CSV), exist_ok=True)
    emb_df.to_csv(OUTPUT_EMBEDDINGS_CSV, index=False)
    np.save(OUTPUT_EMBEDDINGS_NPY, embeddings_matrix)
    print(f"  Saved {len(emb_df)} embeddings to {OUTPUT_EMBEDDINGS_CSV} and {OUTPUT_EMBEDDINGS_NPY}")

    # Save per-object evaluation results
    eval_df = pd.DataFrame(evaluation_records)
    eval_df.to_csv(OUTPUT_EVALUATION_CSV, index=False)
    print(f"  Saved per-object evaluation results to {OUTPUT_EVALUATION_CSV}")

    # Numerical stability checks
    nan_count = int(np.isnan(embeddings_matrix).sum())
    inf_count = int(np.isinf(embeddings_matrix).sum())
    norms = emb_df["embedding_norm"].values

    min_norm = float(np.min(norms))
    max_norm = float(np.max(norms))
    mean_norm = float(np.mean(norms))
    median_norm = float(np.median(norms))

    if nan_count > 0 or inf_count > 0:
        raise ValueError(f"Numerical instability detected: {nan_count} NaNs, {inf_count} Infs!")

    print(f"\n[Step 4] Numerical Stability Check:")
    print(f"  Objects:              {len(embeddings_matrix)}")
    print(f"  Embedding Dimension:  {embeddings_matrix.shape[1]}")
    print(f"  NaN Count:            {nan_count}")
    print(f"  Inf Count:            {inf_count}")
    print(f"  Min Norm:             {min_norm:.4f}")
    print(f"  Max Norm:             {max_norm:.4f}")
    print(f"  Mean Norm:            {mean_norm:.4f}")
    print(f"  Median Norm:          {median_norm:.4f}")

    # Verify model parameter invariance
    for name, param in model.named_parameters():
        if not torch.equal(param, initial_model_params[name]):
            raise RuntimeError(f"Parameter mutation detected in model parameter {name}!")
    for name, param in lc_encoder.named_parameters():
        if not torch.equal(param, initial_encoder_params[name]):
            raise RuntimeError(f"Parameter mutation detected in encoder parameter {name}!")

    ckpt_hash_after = compute_sha256(checkpoint_path)
    if ckpt_hash_after != ckpt_hash_before:
        raise RuntimeError(f"Checkpoint altered during inference! SHA-256 changed.")
    print(f"  Parameter invariance confirmed: 0 parameter changes. Checkpoint hash identical.")

    # --- TASK 6: SCIENTIFIC METRICS ---
    print("\n[Step 5] Calculating scientific metrics...")
    # Binary subset: known (0) and OOD anomaly (1). Controls (-1) excluded.
    eval_binary_df = eval_df[eval_df["is_anomaly_ground_truth"].isin([0, 1])].copy()
    y_true = eval_binary_df["is_anomaly_ground_truth"].values.astype(int)
    y_score = eval_binary_df["ensemble_score"].values.astype(float)
    y_pred = eval_binary_df["is_flagged"].values.astype(int)

    num_known = int((y_true == 0).sum())
    num_anomaly = int((y_true == 1).sum())

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = float((tp + tn) / len(y_true))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    auroc = float(roc_auc_score(y_true, y_score))
    precisions_curve, recalls_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    auprc = float(auc(recalls_curve, precisions_curve))
    ap = float(average_precision_score(y_true, y_score))

    fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_true, y_score)

    # Save ROC and PR curve data
    roc_df = pd.DataFrame({
        "fpr": fpr_curve,
        "tpr": tpr_curve,
        "threshold": roc_thresholds
    })
    os.makedirs(os.path.dirname(OUTPUT_ROC_CSV), exist_ok=True)
    roc_df.to_csv(OUTPUT_ROC_CSV, index=False)

    pr_df = pd.DataFrame({
        "precision": precisions_curve,
        "recall": recalls_curve,
        "threshold": np.append(pr_thresholds, np.nan)
    })
    pr_df.to_csv(OUTPUT_PR_CSV, index=False)

    # Save confusion matrix
    cm_df = pd.DataFrame([
        {"category": "True Positive (TP)", "count": tp, "rate": recall, "description": "OOD anomaly correctly flagged"},
        {"category": "False Positive (FP)", "count": fp, "rate": fpr, "description": "Known in-distribution falsely flagged"},
        {"category": "True Negative (TN)", "count": tn, "rate": specificity, "description": "Known in-distribution correctly rejected"},
        {"category": "False Negative (FN)", "count": fn, "rate": 1.0 - recall, "description": "OOD anomaly missed"}
    ])
    cm_df.to_csv(OUTPUT_CONFUSION_CSV, index=False)

    # Per-population breakdown
    pop_summaries = {}
    for pop_fam, grp in eval_df.groupby("population_family"):
        scores = grp["ensemble_score"].values
        flagged = grp["is_flagged"].sum()
        pop_summaries[pop_fam] = {
            "n": int(len(grp)),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "flagged_count": int(flagged),
            "flagged_fraction": float(flagged / len(grp))
        }

    # Diagnostics: failure cases
    known_eval_df = eval_df[eval_df["is_anomaly_ground_truth"] == 0]
    highest_known = known_eval_df.sort_values(by="ensemble_score", ascending=False).head(5)

    ood_eval_df = eval_df[eval_df["is_anomaly_ground_truth"] == 1]
    lowest_ood = ood_eval_df.sort_values(by="ensemble_score", ascending=True).head(5)

    control_eval_df = eval_df[eval_df["is_anomaly_ground_truth"] == -1]
    highest_controls = control_eval_df.sort_values(by="ensemble_score", ascending=False).head(5)

    # Load synthetic benchmark metrics for comparison
    with open("reports/anomaly_metrics.json", "r") as f:
        synthetic_metrics = json.load(f)

    results_summary = {
        "num_objects": num_objects,
        "num_binary_evaluated": len(y_true),
        "num_known": num_known,
        "num_anomaly": num_anomaly,
        "num_controls": len(control_eval_df),
        "embedding_dim": 128,
        "numerical_stability": {
            "min_norm": min_norm,
            "max_norm": max_norm,
            "mean_norm": mean_norm,
            "median_norm": median_norm,
            "nan_count": nan_count,
            "inf_count": inf_count
        },
        "metrics": {
            "auroc": auroc,
            "auprc": auprc,
            "average_precision": ap,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "fpr": fpr,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        },
        "per_population": pop_summaries,
        "failure_cases": {
            "highest_known": highest_known[[
                "candidate_id", "ztf_designation", "astrophysical_class", "ensemble_score", "valid_token_count", "padding_fraction"
            ]].to_dict("records"),
            "lowest_ood": lowest_ood[[
                "candidate_id", "ztf_designation", "astrophysical_class", "ensemble_score", "valid_token_count", "padding_fraction"
            ]].to_dict("records"),
            "highest_controls": highest_controls[[
                "candidate_id", "ztf_designation", "astrophysical_class", "ensemble_score", "valid_token_count", "padding_fraction"
            ]].to_dict("records")
        },
        "synthetic_baseline": {
            "auroc": synthetic_metrics["curve_metrics"]["auroc"],
            "auprc": synthetic_metrics["curve_metrics"]["auprc"],
            "f1": synthetic_metrics["binary_metrics"]["f1"],
            "precision": synthetic_metrics["binary_metrics"]["precision"],
            "recall": synthetic_metrics["binary_metrics"]["recall"],
            "fpr": synthetic_metrics["binary_metrics"]["fpr"]
        },
        "checkpoint_hashes": {
            "before": ckpt_hash_before,
            "after": ckpt_hash_after,
            "match": bool(ckpt_hash_before == ckpt_hash_after)
        }
    }

    return results_summary


def main():
    results = run_frozen_real_ztf_evaluation()
    print("\n" + "=" * 80)
    print("  EVALUATION SUMMARY")
    print("=" * 80)
    m = results["metrics"]
    print(f"AUROC:             {m['auroc']:.4f}")
    print(f"AUPRC:             {m['auprc']:.4f}")
    print(f"Average Precision: {m['average_precision']:.4f}")
    print(f"Precision:         {m['precision']:.4f}")
    print(f"Recall:            {m['recall']:.4f}")
    print(f"Specificity:       {m['specificity']:.4f}")
    print(f"FPR:               {m['fpr']:.4f}")
    print(f"F1:                {m['f1']:.4f}")
    print(f"TP / FP / TN / FN: {m['tp']} / {m['fp']} / {m['tn']} / {m['fn']}")


if __name__ == "__main__":
    main()
