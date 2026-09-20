"""Unit and integration tests for Real-ZTF Light-Curve Anomaly Detector v2."""

import os
import json
import hashlib
import numpy as np
import pytest

from src.anomaly_detection.real_ztf_lc_detector import RealZTFLightCurveDetectorV2
from src.investigator.anomaly_assessor import ScientificAnomalyAssessor
from src.investigator.schemas import RepresentationSummary
from src.models.checkpoint_manager import DEFAULT_CHECKPOINT_PATH

PRIMARY_BENCHMARK_PATH = "data/real_ztf_benchmark/frozen_primary_benchmark.csv"
FREEZE_METADATA_PATH = "data/real_ztf_benchmark/benchmark_freeze_metadata.json"
EXPECTED_PRODUCTION_SHA256 = "e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72"


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 digest of a local file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_frozen_artifacts_integrity():
    """Verify that production checkpoint and benchmark CSVs remain strictly unaltered."""
    assert os.path.exists(DEFAULT_CHECKPOINT_PATH)
    ckpt_hash = compute_sha256(DEFAULT_CHECKPOINT_PATH)
    assert ckpt_hash == EXPECTED_PRODUCTION_SHA256, f"Checkpoint SHA-256 altered! Got {ckpt_hash}"

    assert os.path.exists(PRIMARY_BENCHMARK_PATH)
    with open(FREEZE_METADATA_PATH, "r") as f:
        meta = json.load(f)
    expected_bm_hash = meta["file_sha256_digests"]["frozen_primary_benchmark_csv"]
    actual_bm_hash = compute_sha256(PRIMARY_BENCHMARK_PATH)
    assert actual_bm_hash == expected_bm_hash, f"Primary benchmark SHA-256 altered!"


def test_detector_fit_and_predict():
    """Test fitting, scoring, thresholding, and serialization of RealZTFLightCurveDetectorV2."""
    np.random.seed(42)
    # Generate dummy in-distribution embeddings
    X_train = np.random.randn(30, 128)
    X_val = np.random.randn(10, 128)
    # Generate dummy anomaly embeddings (larger variance/offset)
    X_anom = np.random.randn(10, 128) * 5.0 + 3.0

    detector = RealZTFLightCurveDetectorV2(method="pca_mahalanobis", n_components=5)
    detector.fit(X_train)
    assert detector.is_fitted is True

    # Calibrate threshold on validation fold
    thresh = detector.calibrate_threshold(X_val, percentile=90.0)
    assert isinstance(thresh, float)

    # Score single embedding
    score, flag = detector.score(X_anom[0])
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert isinstance(flag, bool)

    # Score multiple embeddings
    results = detector.score(X_anom)
    assert len(results) == 10
    assert all(isinstance(r[0], float) and isinstance(r[1], bool) for r in results)


def test_detector_serialization(tmp_path):
    """Test saving and loading fitted detector model."""
    X_train = np.random.randn(20, 128)
    detector = RealZTFLightCurveDetectorV2(n_components=4)
    detector.fit(X_train)

    save_path = os.path.join(tmp_path, "test_detector.pkl")
    detector.save(save_path)
    assert os.path.exists(save_path)

    loaded_detector = RealZTFLightCurveDetectorV2.load(save_path)
    assert loaded_detector.is_fitted is True
    assert loaded_detector.n_components == 4

    test_emb = np.random.randn(128)
    orig_score, orig_flag = detector.score(test_emb)
    load_score, load_flag = loaded_detector.score(test_emb)
    assert orig_score == load_score
    assert orig_flag == load_flag


def test_invalid_input_shape():
    """Test that input embeddings with invalid dimension raise ValueError."""
    detector = RealZTFLightCurveDetectorV2()
    with pytest.raises(ValueError):
        detector.fit(np.random.randn(10, 64))  # Wrong dimension (64 instead of 128)


def test_scientific_assessor_v2_integration():
    """Test integration of RealZTFLightCurveDetectorV2 with ScientificAnomalyAssessor."""
    X_train = np.random.randn(20, 128)
    detector = RealZTFLightCurveDetectorV2(n_components=4)
    detector.fit(X_train)

    assessor = ScientificAnomalyAssessor(real_ztf_detector=detector)

    rep = RepresentationSummary(
        object_id="TEST_CAND_001",
        embedding_dimension=128,
        embedding_norm=10.5,
        embedding=np.random.randn(128).tolist(),
        valid_token_count=45,
        padding_fraction=0.1,
        available_filters=["zg", "zr"],
        time_span_days=100.0,
        preprocessing_status="SUCCESS"
    )

    assessment = assessor.assess(rep, domain="real_ztf")
    assert assessment.anomaly_score_status == "REAL_ZTF_EVALUATED_RESEARCH"
    assert assessment.anomaly_score is not None
    assert assessment.anomaly_flag is not None
    assert "Real-ZTF Light-Curve Anomaly Detector v2" in assessment.detector_source
    assert len(assessment.limitations) >= 4
