"""Unit and integration tests for full real-ZTF photometric retrieval and benchmark eligibility."""

import os
import hashlib
import pandas as pd
import pytest

from src.data.run_full_retrieval import (
    get_active_candidates,
    CHECKPOINT_PATH,
    EXPECTED_CHECKPOINT_SHA256
)

RESULTS_CSV = "data/real_ztf_benchmark/full_retrieval_results.csv"
ELIGIBILITY_CSV = "data/real_ztf_benchmark/benchmark_eligibility.csv"
RAW_BASE_DIR = "data/real_ztf_benchmark/raw"


def test_active_candidate_registry_integrity():
    """Verify registry extraction returns exactly 180 active candidates with zero pilot contamination."""
    df = get_active_candidates()
    assert len(df) == 180, f"Expected 180 active candidates, got {len(df)}"

    # Provenance verification
    assert (df["provenance_status"] == "VERIFIED").all(), "Non-verified candidates found in active cohort"

    # Verify candidate roles
    roles = df["candidate_role"].value_counts()
    assert roles.get("in_distribution", 0) == 115, f"Expected 115 in-distribution, got {roles.get('in_distribution')}"
    assert roles.get("ood_anomaly", 0) == 50, f"Expected 50 OOD anomaly, got {roles.get('ood_anomaly')}"
    assert roles.get("control", 0) == 15, f"Expected 15 control, got {roles.get('control')}"

    # Quarantine verification
    quarantined_names = {
        "SN 2019np", "ZTF19aacgslb",
        "SN 2020jfo", "ZTF20aaynrrh",
        "AT 2018cow", "ZTF18abukavn",
        "SN 2018zd", "ZTF18aarkpda",
        "ZTF_J195200.60+295217.4", "Field686_Star"
    }
    for name in df["ztf_designation"]:
        assert name not in quarantined_names, f"Quarantined pilot object {name} leaked into active cohort!"


def test_full_retrieval_results_schema_and_counts():
    """Verify full_retrieval_results.csv contains exactly 180 rows and all required fields."""
    if not os.path.exists(RESULTS_CSV):
        pytest.skip("full_retrieval_results.csv has not been generated yet")

    df = pd.read_csv(RESULTS_CSV)
    assert len(df) == 180, f"Expected 180 rows in full retrieval results, got {len(df)}"

    required_cols = [
        "candidate_id", "ztf_designation", "astrophysical_class", "dataset_role",
        "retrieval_status", "matched_oids", "crossmatch_separation_arcsec",
        "raw_observation_count", "clean_observation_count", "window_observation_count",
        "valid_token_count", "padding_fraction", "available_filters", "missing_filters",
        "partial_filter_coverage", "baseline_days", "preprocessing_status",
        "coverage_status", "exclusion_reason"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column in full results: {col}"


def test_coordinate_matching_separation_bound():
    """Verify all successful retrievals obey the <= 1.5 arcsec coordinate bound."""
    if not os.path.exists(RESULTS_CSV):
        pytest.skip("full_retrieval_results.csv has not been generated yet")

    df = pd.read_csv(RESULTS_CSV)
    succ = df[df["retrieval_status"] == "SUCCESS"]
    assert len(succ) > 0, "No successful retrievals found"

    for _, row in succ.iterrows():
        sep = float(row["crossmatch_separation_arcsec"])
        assert sep <= 1.5, f"Object {row['candidate_id']} separation {sep} > 1.5 arcsec!"


def test_token_and_padding_arithmetic_full():
    """Verify mathematical consistency of token counts and padding fractions."""
    if not os.path.exists(RESULTS_CSV):
        pytest.skip("full_retrieval_results.csv has not been generated yet")

    df = pd.read_csv(RESULTS_CSV)
    for _, row in df.iterrows():
        tok = int(row["valid_token_count"])
        pad = float(row["padding_fraction"])

        assert 0 <= tok <= 50, f"Invalid token count {tok} for {row['candidate_id']}"
        assert 0.0 <= pad <= 1.0, f"Invalid padding fraction {pad} for {row['candidate_id']}"

        expected_pad = round((50.0 - tok) / 50.0, 4)
        assert abs(pad - expected_pad) < 1e-4, f"Padding arithmetic mismatch: {pad} vs {expected_pad}"


def test_benchmark_eligibility_file_integrity():
    """Verify benchmark_eligibility.csv structure and strict criterion matching."""
    if not os.path.exists(ELIGIBILITY_CSV):
        pytest.skip("benchmark_eligibility.csv has not been generated yet")

    df = pd.read_csv(ELIGIBILITY_CSV)
    assert len(df) == 180, f"Expected 180 rows in eligibility CSV, got {len(df)}"

    required_cols = [
        "candidate_id", "ztf_designation", "astrophysical_class", "dataset_role",
        "coverage_status", "benchmark_eligible", "eligibility_reason"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column in eligibility CSV: {col}"

    # Verify eligibility rule: benchmark_eligible iff COVERAGE_SUFFICIENT
    for _, row in df.iterrows():
        if row["coverage_status"] == "COVERAGE_SUFFICIENT":
            assert bool(row["benchmark_eligible"]) is True, f"COVERAGE_SUFFICIENT object {row['candidate_id']} marked ineligible!"
            assert "ELIGIBLE" in str(row["eligibility_reason"])
        else:
            assert bool(row["benchmark_eligible"]) is False, f"Non-sufficient object {row['candidate_id']} marked eligible!"
            assert len(str(row["eligibility_reason"])) > 0


def test_raw_data_directory_presence():
    """Verify that every active candidate has a dedicated raw directory with data or log."""
    if not os.path.exists(RESULTS_CSV):
        pytest.skip("full_retrieval_results.csv has not been generated yet")

    df = pd.read_csv(RESULTS_CSV)
    for _, row in df.iterrows():
        cid = row["candidate_id"]
        cand_dir = os.path.join(RAW_BASE_DIR, cid)
        assert os.path.isdir(cand_dir), f"Missing raw directory for candidate {cid}"

        raw_csv = os.path.join(cand_dir, "raw_irsa.csv")
        err_log = os.path.join(cand_dir, "error.log")
        assert os.path.exists(raw_csv) or os.path.exists(err_log), f"No raw data or error log for {cid}"


def test_production_checkpoint_sha256_unmodified():
    """Verify that production checkpoint SHA-256 remains strictly untouched."""
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint missing: {CHECKPOINT_PATH}"
    with open(CHECKPOINT_PATH, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert digest == EXPECTED_CHECKPOINT_SHA256, f"Checkpoint SHA-256 altered! Expected {EXPECTED_CHECKPOINT_SHA256}, got {digest}"
