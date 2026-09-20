"""Automated tests for the small-batch real-ZTF photometric retrieval pilot."""

import os
import hashlib
import pandas as pd
import pytest

from src.data.photometric_pilot import select_pilot_batch, CLASS_PREFIX_MAP

EXPECTED_CHECKPOINT_SHA256 = "e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72"
CHECKPOINT_PATH = "models/checkpoints/acei_multimodal_production.pt"
PILOT_CSV_PATH = "data/real_ztf_benchmark/photometric_pilot.csv"
RAW_BASE_DIR = "data/real_ztf_benchmark/raw"


def test_pilot_candidate_selection():
    """Verify deterministic selection of exactly 35 candidates (5 per class across 7 populations)."""
    df = select_pilot_batch()
    assert len(df) == 35, f"Expected 35 candidates, got {len(df)}"

    # Check 5 objects per class
    counts = df["provisional_class"].value_counts()
    for standard_class in CLASS_PREFIX_MAP.values():
        assert counts.get(standard_class, 0) == 5, f"Expected 5 candidates for {standard_class}, got {counts.get(standard_class, 0)}"


def test_quarantined_pilot_objects_excluded():
    """Verify that exploratory transfer-pilot objects are strictly excluded."""
    df = select_pilot_batch()
    quarantined_names = {
        "SN 2019np", "ZTF19aacgslb",
        "SN 2020jfo", "ZTF20aaynrrh",
        "AT 2018cow", "ZTF18abukavn",
        "SN 2018zd", "ZTF18aarkpda",
        "ZTF_J195200.60+295217.4", "Field686_Star"
    }
    for _, row in df.iterrows():
        assert row["ztf_designation"] not in quarantined_names, f"Quarantined object {row['ztf_designation']} found in pilot batch!"
        assert not str(row["candidate_id"]).startswith("REJ_"), f"Rejected ID {row['candidate_id']} found in pilot batch!"


def test_photometric_pilot_csv_structure_and_counts():
    """Verify photometric_pilot.csv contains exactly 35 rows and required schema."""
    if not os.path.exists(PILOT_CSV_PATH):
        pytest.skip("photometric_pilot.csv has not been generated yet")

    df = pd.read_csv(PILOT_CSV_PATH)
    assert len(df) == 35, f"Expected 35 rows in pilot CSV, got {len(df)}"

    required_columns = [
        "candidate_id", "target_name", "provisional_class", "ra", "dec",
        "crossmatch_separation_arcsec", "retrieval_status", "coverage_status",
        "n_raw_points", "n_clean_points", "n_valid_tokens", "padding_fraction",
        "available_filters", "missing_filters", "partial_filter_coverage",
        "baseline_days", "cadence_median_days", "peak_snr", "raw_data_path", "notes"
    ]
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"

    counts = df["provisional_class"].value_counts()
    for standard_class in CLASS_PREFIX_MAP.values():
        assert counts.get(standard_class, 0) == 5, f"Expected 5 rows for {standard_class} in CSV"


def test_raw_data_storage():
    """Verify all 35 candidates have raw directory and raw_irsa.csv or error.log."""
    if not os.path.exists(PILOT_CSV_PATH):
        pytest.skip("photometric_pilot.csv has not been generated yet")

    df = pd.read_csv(PILOT_CSV_PATH)
    for _, row in df.iterrows():
        cand_id = row["candidate_id"]
        cand_dir = os.path.join(RAW_BASE_DIR, cand_id)
        assert os.path.isdir(cand_dir), f"Raw data directory missing for {cand_id}: {cand_dir}"

        raw_csv = os.path.join(cand_dir, "raw_irsa.csv")
        err_log = os.path.join(cand_dir, "error.log")
        has_file = os.path.exists(raw_csv) or os.path.exists(err_log)
        assert has_file, f"Neither raw_irsa.csv nor error.log found in {cand_dir}"


def test_coordinate_crossmatch_constraint():
    """Verify that all successful associations have angular separation <= 1.5 arcsec."""
    if not os.path.exists(PILOT_CSV_PATH):
        pytest.skip("photometric_pilot.csv has not been generated yet")

    df = pd.read_csv(PILOT_CSV_PATH)
    successes = df[df["retrieval_status"] == "SUCCESS"]
    for _, row in successes.iterrows():
        sep = float(row["crossmatch_separation_arcsec"])
        assert sep <= 1.5, f"Object {row['candidate_id']} has separation {sep} > 1.5 arcsec!"


def test_token_and_padding_arithmetic():
    """Verify token counts in [0, 50], padding in [0, 1], and exact padding fraction arithmetic."""
    if not os.path.exists(PILOT_CSV_PATH):
        pytest.skip("photometric_pilot.csv has not been generated yet")

    df = pd.read_csv(PILOT_CSV_PATH)
    for _, row in df.iterrows():
        tokens = int(row["n_valid_tokens"])
        pad = float(row["padding_fraction"])

        assert 0 <= tokens <= 50, f"Object {row['candidate_id']} token count {tokens} outside [0, 50]"
        assert 0.0 <= pad <= 1.0, f"Object {row['candidate_id']} padding fraction {pad} outside [0.0, 1.0]"

        expected_pad = round((50.0 - tokens) / 50.0, 4)
        assert abs(pad - expected_pad) < 1e-4, f"Padding fraction mismatch for {row['candidate_id']}: {pad} vs {expected_pad}"


def test_production_checkpoint_immutability():
    """Verify that production checkpoint SHA-256 remains strictly untouched."""
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    with open(CHECKPOINT_PATH, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert digest == EXPECTED_CHECKPOINT_SHA256, f"Checkpoint SHA-256 altered! Expected {EXPECTED_CHECKPOINT_SHA256}, got {digest}"
