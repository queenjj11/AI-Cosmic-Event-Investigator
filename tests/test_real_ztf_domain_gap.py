"""Unit and regression tests for the scientific domain gap diagnosis suite.

Verifies:
1. Production checkpoint and primary manifest SHA-256 integrity.
2. Output diagnostic CSVs and plot exist and are non-empty.
3. Detector component summary table has exactly 12 rows with finite values.
4. Population detector summary covers all 7 population families with correct counts (sum=106).
5. Coverage correlations have valid coefficients in [-1, 1] and p-values in [0, 1].
6. Synthetic vs real distribution summary contains valid comparative statistics.
7. Failure case summary contains exactly 15 diagnosed records.
8. Model parameters and checkpoint remain strictly immutable.
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch

from src.models.checkpoint_manager import (
    DEFAULT_CHECKPOINT_PATH,
    load_production_model,
)
from src.evaluation.diagnose_real_ztf_domain_gap import (
    PRIMARY_BENCHMARK_PATH,
    FREEZE_METADATA_PATH,
    EXPECTED_PRODUCTION_SHA256,
    OUTPUT_DIR,
    compute_sha256,
    verify_integrity,
)

EXPECTED_PRIMARY_MANIFEST_SHA256 = "0e839c8f8ff1b83969e3f959a2be3342a96ac2564d00e73b045a21445cd4501c"


def test_1_verify_manifest_and_checkpoint_hashes():
    """Test 1: Verify cryptographic hashes of manifest and production checkpoint."""
    man_hash, ckpt_hash = verify_integrity(PRIMARY_BENCHMARK_PATH, FREEZE_METADATA_PATH, DEFAULT_CHECKPOINT_PATH)
    assert man_hash == EXPECTED_PRIMARY_MANIFEST_SHA256
    assert ckpt_hash == EXPECTED_PRODUCTION_SHA256


def test_2_diagnostic_artifacts_exist_and_non_empty():
    """Test 2: All 5 diagnostic CSVs and diagnostic plot exist and are non-empty."""
    required_files = [
        "detector_component_summary.csv",
        "population_detector_summary.csv",
        "coverage_score_correlations.csv",
        "synthetic_real_distribution_summary.csv",
        "failure_case_summary.csv",
        "domain_gap_diagnostics.png"
    ]
    for fname in required_files:
        fpath = os.path.join(OUTPUT_DIR, fname)
        assert os.path.exists(fpath), f"Missing diagnostic output: {fpath}"
        assert os.path.getsize(fpath) > 100, f"Diagnostic output is unexpectedly small/empty: {fpath}"


def test_3_detector_component_summary_schema_and_values():
    """Test 3: Detector component summary contains 12 rows with expected splits and finite normalized scores."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "detector_component_summary.csv"))
    assert len(df) == 12
    required_cols = ["detector_name", "dataset_split", "norm_mean", "norm_median", "norm_std", "weight_in_ensemble"]
    for c in required_cols:
        assert c in df.columns

    # Check normalized scores are bounded in [0, 1]
    assert (df["norm_mean"] >= 0.0).all() and (df["norm_mean"] <= 1.0).all()
    assert (df["norm_median"] >= 0.0).all() and (df["norm_median"] <= 1.0).all()

    # Mahalanobis on real ZTF should show high saturation
    real_mah = df[(df["detector_name"] == "Mahalanobis") & (df["dataset_split"].str.contains("Real"))]
    assert len(real_mah) == 1
    assert real_mah["norm_mean"].values[0] > 0.85, "Mahalanobis norm mean on real ZTF expected to be > 0.85"


def test_4_population_detector_summary_coverage_and_counts():
    """Test 4: Population summary covers 7 families, sums to 106 objects, with expected counts."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "population_detector_summary.csv"))
    assert len(df) == 7
    assert df["count"].sum() == 106

    expected_counts = {
        "Variable_Star": 24,
        "SN_Ia": 19,
        "SN_II": 15,
        "Unclassified_Field_Star": 14,
        "TDE": 14,
        "Cataclysmic_Variable": 13,
        "SLSN": 7
    }
    actual_counts = dict(zip(df["population_family"], df["count"]))
    assert actual_counts == expected_counts


def test_5_coverage_correlations_bounds_and_significance():
    """Test 5: Correlation matrix has valid coefficients in [-1, 1] and valid p-values in [0, 1]."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "coverage_score_correlations.csv"))
    assert len(df) == 20
    assert (df["pearson_r"] >= -1.0).all() and (df["pearson_r"] <= 1.0).all()
    assert (df["spearman_rho"] >= -1.0).all() and (df["spearman_rho"] <= 1.0).all()
    assert (df["pearson_pvalue"] >= 0.0).all() and (df["pearson_pvalue"] <= 1.0).all()
    assert (df["spearman_pvalue"] >= 0.0).all() and (df["spearman_pvalue"] <= 1.0).all()

    # Valid token count vs ensemble score must be negatively correlated with p < 0.01
    tok_ens = df[(df["coverage_variable"] == "valid_token_count") & (df["detector_score"] == "ensemble_score")]
    assert len(tok_ens) == 1
    assert tok_ens["pearson_r"].values[0] < -0.20
    assert tok_ens["pearson_pvalue"].values[0] < 0.01


def test_6_synthetic_real_distribution_summary_completeness():
    """Test 6: Distribution summary covers all required measurement categories."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "synthetic_real_distribution_summary.csv"))
    assert len(df) == 22
    categories = set(df["metric_category"])
    assert categories == {"Sampling", "Temporal", "Photometry", "Representation"}
    assert (df["mean"].notna()).all()
    assert (df["median"].notna()).all()


def test_7_failure_case_summary_structure():
    """Test 7: Failure case summary contains exactly 15 cases (5 FP, 5 FN, 5 controls)."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "failure_case_summary.csv"))
    assert len(df) == 15
    cats = df["failure_category"].value_counts().to_dict()
    assert cats["False Positive (Known In-Distribution)"] == 5
    assert cats["False Negative (OOD Anomaly)"] == 5
    assert cats["Control Outlier (Unclassified Field Star)"] == 5


def test_8_model_parameters_and_checkpoint_invariant():
    """Test 8: Checkpoint SHA-256 remains bitwise invariant and model weights have zero grads."""
    current_hash = compute_sha256(DEFAULT_CHECKPOINT_PATH)
    assert current_hash == EXPECTED_PRODUCTION_SHA256

    model, payload = load_production_model(DEFAULT_CHECKPOINT_PATH, device="cpu")
    assert not model.training
    for name, param in model.named_parameters():
        assert param.grad is None
    assert payload["real_ztf_data_used"] is False
