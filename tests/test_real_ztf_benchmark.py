"""Comprehensive unit and integrity tests for the Real-ZTF Evaluation Benchmark.

Verifies:
1. Manifest schema parsing and serialization.
2. 12 strict benchmark integrity checks (duplicates, coordinate collisions, split leakage).
3. Strict pilot isolation (dataset_role='transfer_pilot', dataset_split='transfer_pilot').
4. Provenance completeness on classified objects.
5. Invariance of the production checkpoint (hash, parameters, architecture).
"""

import os
import copy
import hashlib
import pytest
import torch

from src.data.real_ztf_manifest import (
    RealBenchmarkRecord,
    load_manifest,
    save_manifest,
    MANIFEST_COLUMNS
)
from src.data.benchmark_validator import (
    BenchmarkValidator,
    BenchmarkIntegrityError,
    ValidationReport
)
from src.models.checkpoint_manager import (
    load_production_lightcurve_encoder,
    DEFAULT_CHECKPOINT_PATH
)


EXPECTED_PRODUCTION_CHECKPOINT_SHA256 = "e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72"
EXPECTED_ENCODER_PARAMETER_COUNT = 425072


@pytest.fixture
def pilot_records():
    """Load the verified 5-object pilot manifest records."""
    manifest_path = "data/real_ztf_benchmark/manifest.csv"
    assert os.path.exists(manifest_path), f"Manifest not found: {manifest_path}"
    return load_manifest(manifest_path)


def test_manifest_schema_and_column_completeness(pilot_records):
    """Verify manifest file contains all 30 required columns and loads cleanly."""
    assert len(pilot_records) == 5
    for r in pilot_records:
        d = r.to_dict()
        for col in MANIFEST_COLUMNS:
            assert col in d, f"Missing required column '{col}' in record {r.object_id}"


def test_pilot_records_pass_validator(pilot_records):
    """Verify the 5 verified pilot records pass all validator checks with zero errors."""
    validator = BenchmarkValidator(strict_mode=True)
    report = validator.validate(pilot_records)
    assert report.is_valid is True
    assert len(report.errors) == 0
    assert report.total_objects == 5
    assert report.objects_by_role.get("transfer_pilot") == 5
    assert report.objects_by_split.get("transfer_pilot") == 5


def test_pilot_objects_remain_strictly_isolated(pilot_records):
    """Verify pilot objects are strictly marked transfer_pilot and never in evaluation splits."""
    for r in pilot_records:
        assert r.dataset_role == "transfer_pilot"
        assert r.dataset_split == "transfer_pilot"
        assert r.dataset_split not in ["test_known", "test_anomaly", "val_calibration", "control_unclassified"]


def test_pilot_role_mismatch_fails_validation(pilot_records):
    """Verify validator fails if a pilot object is accidentally placed in an evaluation split."""
    mutated = copy.deepcopy(pilot_records)
    # Violate pilot isolation by placing SN_2019np in test_known
    mutated[0].dataset_split = "test_known"

    validator = BenchmarkValidator(strict_mode=False)
    report = validator.validate(mutated)
    assert report.is_valid is False
    assert any("Pilot isolation violation" in err for err in report.errors)


def test_duplicate_object_id_fails_validation(pilot_records):
    """Verify duplicate object IDs trigger a critical validation error."""
    mutated = copy.deepcopy(pilot_records)
    dup = copy.deepcopy(mutated[0])
    mutated.append(dup)

    validator = BenchmarkValidator(strict_mode=False)
    report = validator.validate(mutated)
    assert report.is_valid is False
    assert any("Duplicate object_id detected" in err for err in report.errors)


def test_duplicate_coordinate_collision_fails_validation(pilot_records):
    """Verify distinct object IDs having celestial separation < 1.5 arcsec trigger collision error."""
    mutated = copy.deepcopy(pilot_records)
    # Create a clone of SN_2019np with a different ID but identical coordinates
    clone = copy.deepcopy(mutated[0])
    clone.object_id = "SN_2019np_DUPLICATE_SOURCE"

    mutated.append(clone)

    validator = BenchmarkValidator(strict_mode=False)
    report = validator.validate(mutated)
    assert report.is_valid is False
    assert any("Duplicate celestial coordinate collision" in err for err in report.errors)


def test_invalid_coordinate_separation_fails_validation(pilot_records):
    """Verify coordinate association distance exceeding 1.5 arcseconds fails validation."""
    mutated = copy.deepcopy(pilot_records)
    # Set separation to 2.1 arcseconds (beyond 1.5'' cone)
    mutated[0].max_angular_sep_arcsec = 2.1

    validator = BenchmarkValidator(strict_mode=False)
    report = validator.validate(mutated)
    assert report.is_valid is False
    assert any("exceeds maximum association separation" in err for err in report.errors)


def test_missing_provenance_citation_fails_validation(pilot_records):
    """Verify classified astronomical targets without provenance citation fail validation."""
    mutated = copy.deepcopy(pilot_records)
    mutated[0].class_authority = ""  # Empty authority on SN Ia

    validator = BenchmarkValidator(strict_mode=False)
    report = validator.validate(mutated)
    assert report.is_valid is False
    assert any("lacks authoritative provenance citation" in err for err in report.errors)


def test_split_leakage_detection(pilot_records):
    """Verify object-level split leakage across multiple splits is detected."""
    # Build two benchmark records in different splits with the same object_id
    r1 = copy.deepcopy(pilot_records[0])
    r1.dataset_role = "test_benchmark"
    r1.dataset_split = "test_known"

    r2 = copy.deepcopy(pilot_records[0])
    r2.dataset_role = "val_calibration"
    r2.dataset_split = "val_calibration"

    validator = BenchmarkValidator(strict_mode=False)
    report = validator.validate([r1, r2])
    assert report.is_valid is False
    assert any("Duplicate object_id detected" in err for err in report.errors)


def test_unclassified_object_ground_truth_integrity(pilot_records):
    """Verify unclassified field objects must have is_anomaly_ground_truth = -1."""
    mutated = copy.deepcopy(pilot_records)
    # The 5th record is ZTF_J195200 (Unclassified Field Star)
    # Force an invalid ground truth label of 1
    mutated[4].is_anomaly_ground_truth = 1

    validator = BenchmarkValidator(strict_mode=False)
    report = validator.validate(mutated)
    assert report.is_valid is False
    assert any("must have is_anomaly_ground_truth = -1" in err for err in report.errors)


def test_impossible_class_assignment_fails_validation(pilot_records):
    """Verify impossible or fabricated class strings trigger taxonomy failure."""
    mutated = copy.deepcopy(pilot_records)
    mutated[0].astrophysical_class = "InventedAlienArtifactClass"

    validator = BenchmarkValidator(strict_mode=False)
    report = validator.validate(mutated)
    assert report.is_valid is False
    assert any("assigned unrecognized astrophysical class" in err for err in report.errors)


def test_preprocessing_metadata_consistency(pilot_records):
    """Verify token count arithmetic and padding fraction consistency."""
    mutated = copy.deepcopy(pilot_records)
    # Set valid tokens = 20, but padding fraction = 0.10 (inconsistent, expected 0.60)
    mutated[0].valid_token_count = 20
    mutated[0].padding_fraction = 0.10

    validator = BenchmarkValidator(strict_mode=False)
    report = validator.validate(mutated)
    assert report.is_valid is False
    assert any("inconsistent padding fraction" in err for err in report.errors)


def test_unresolved_exclusion_reason_fails_validation(pilot_records):
    """Verify failed or ambiguous objects missing exclusion_reason fail validation."""
    mutated = copy.deepcopy(pilot_records)
    mutated[0].retrieval_status = "AMBIGUOUS"
    mutated[0].exclusion_reason = ""  # Missing explanation

    validator = BenchmarkValidator(strict_mode=False)
    report = validator.validate(mutated)
    assert report.is_valid is False
    assert any("lacks an explicit exclusion_reason" in err for err in report.errors)


def test_production_checkpoint_remains_frozen_and_unmutated():
    """Verify production checkpoint file is bitwise identical to reference hash."""
    assert os.path.exists(DEFAULT_CHECKPOINT_PATH), f"Checkpoint missing at {DEFAULT_CHECKPOINT_PATH}"

    with open(DEFAULT_CHECKPOINT_PATH, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    assert file_hash == EXPECTED_PRODUCTION_CHECKPOINT_SHA256, (
        f"Checkpoint hash mutated! Expected {EXPECTED_PRODUCTION_CHECKPOINT_SHA256}, got {file_hash}"
    )


def test_production_encoder_architecture_remains_unmutated():
    """Verify production LightCurveEncoder architecture has exact parameters and layers."""
    encoder, payload = load_production_lightcurve_encoder(DEFAULT_CHECKPOINT_PATH, device="cpu")

    # Architecture configuration checks
    arch = payload["architecture"]["lc_encoder_config"]
    assert arch["num_layers"] == 3
    assert arch["nhead"] == 4
    assert arch["d_model"] == 128
    assert arch["dim_feedforward"] == 256
    assert arch["dropout"] == 0.2
    assert arch["time_embed_dim"] == 32
    assert arch["num_bands"] == 5

    # Parameter count check
    total_params = sum(p.numel() for p in encoder.parameters())
    assert total_params == EXPECTED_ENCODER_PARAMETER_COUNT, (
        f"Encoder parameter count mutated! Expected {EXPECTED_ENCODER_PARAMETER_COUNT}, got {total_params}"
    )

    # Confirm real data was not used in checkpoint
    assert payload["real_ztf_data_used"] is False
