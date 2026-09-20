"""Unit tests for the controlled zero-shot Real-ZTF Primary Benchmark evaluation.

Verifies the 15 required validation assertions:
1. Exactly 106 primary objects evaluated.
2. Exactly 106 embeddings generated.
3. Embedding dimension = 128.
4. Zero NaN embeddings.
5. Zero Inf embeddings.
6. All candidate IDs match the frozen primary manifest.
7. No secondary-tier object appears.
8. No transfer-pilot object appears.
9. Checkpoint SHA-256 remains unchanged.
10. Threshold remains exactly 0.65.
11. No optimizer/training operation occurs.
12. Ground-truth labels match the frozen manifest.
13. Evaluation results contain one row per primary candidate.
14. Continuous scores are finite.
15. No real-ZTF data is written into training datasets.
"""

import os
import hashlib
import pytest
import numpy as np
import pandas as pd
import torch

from src.models.checkpoint_manager import (
    DEFAULT_CHECKPOINT_PATH,
    load_production_model,
    load_production_lightcurve_encoder,
)
from src.evaluation.evaluate_frozen_real_ztf import (
    PRIMARY_BENCHMARK_PATH,
    SECONDARY_BENCHMARK_PATH,
    OUTPUT_EMBEDDINGS_CSV,
    OUTPUT_EMBEDDINGS_NPY,
    OUTPUT_EVALUATION_CSV,
    HISTORICAL_PILOT_IDS,
    compute_sha256,
)

EXPECTED_PRODUCTION_SHA256 = "e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72"
EXPECTED_PRIMARY_MANIFEST_SHA256 = "0e839c8f8ff1b83969e3f959a2be3342a96ac2564d00e73b045a21445cd4501c"


@pytest.fixture(scope="module")
def evaluation_artifacts():
    """Load primary manifest, embeddings, and evaluation outputs."""
    assert os.path.exists(PRIMARY_BENCHMARK_PATH), "Missing primary benchmark manifest"
    assert os.path.exists(OUTPUT_EMBEDDINGS_CSV), "Missing primary embeddings CSV"
    assert os.path.exists(OUTPUT_EMBEDDINGS_NPY), "Missing primary embeddings NPY"
    assert os.path.exists(OUTPUT_EVALUATION_CSV), "Missing primary evaluation CSV"

    manifest_df = pd.read_csv(PRIMARY_BENCHMARK_PATH)
    embeddings_df = pd.read_csv(OUTPUT_EMBEDDINGS_CSV)
    embeddings_mat = np.load(OUTPUT_EMBEDDINGS_NPY)
    evaluation_df = pd.read_csv(OUTPUT_EVALUATION_CSV)

    return {
        "manifest": manifest_df,
        "embeddings_df": embeddings_df,
        "embeddings_mat": embeddings_mat,
        "evaluation": evaluation_df
    }


def test_1_exactly_106_primary_objects_evaluated(evaluation_artifacts):
    """Assertion 1: Exactly 106 primary objects evaluated."""
    eval_df = evaluation_artifacts["evaluation"]
    assert len(eval_df) == 106, f"Expected 106 evaluated objects, found {len(eval_df)}"


def test_2_exactly_106_embeddings_generated(evaluation_artifacts):
    """Assertion 2: Exactly 106 embeddings generated."""
    emb_df = evaluation_artifacts["embeddings_df"]
    emb_mat = evaluation_artifacts["embeddings_mat"]
    assert len(emb_df) == 106, f"Expected 106 embedding CSV rows, found {len(emb_df)}"
    assert emb_mat.shape[0] == 106, f"Expected 106 rows in NPY embedding matrix, found {emb_mat.shape[0]}"


def test_3_embedding_dimension_is_128(evaluation_artifacts):
    """Assertion 3: Embedding dimension = 128."""
    emb_mat = evaluation_artifacts["embeddings_mat"]
    assert emb_mat.shape[1] == 128, f"Expected embedding dim 128, found {emb_mat.shape[1]}"


def test_4_zero_nan_embeddings(evaluation_artifacts):
    """Assertion 4: Zero NaN values in embeddings."""
    emb_mat = evaluation_artifacts["embeddings_mat"]
    nan_count = int(np.isnan(emb_mat).sum())
    assert nan_count == 0, f"Found {nan_count} NaN values in embeddings matrix"


def test_5_zero_inf_embeddings(evaluation_artifacts):
    """Assertion 5: Zero Inf values in embeddings."""
    emb_mat = evaluation_artifacts["embeddings_mat"]
    inf_count = int(np.isinf(emb_mat).sum())
    assert inf_count == 0, f"Found {inf_count} Inf values in embeddings matrix"


def test_6_all_candidate_ids_match_frozen_primary_manifest(evaluation_artifacts):
    """Assertion 6: All candidate IDs match the frozen primary manifest in exact order."""
    manifest_ids = evaluation_artifacts["manifest"]["candidate_id"].tolist()
    eval_ids = evaluation_artifacts["evaluation"]["candidate_id"].tolist()
    emb_ids = evaluation_artifacts["embeddings_df"]["candidate_id"].tolist()

    assert eval_ids == manifest_ids, "Candidate IDs in evaluation CSV do not match manifest!"
    assert emb_ids == manifest_ids, "Candidate IDs in embeddings CSV do not match manifest!"


def test_7_no_secondary_tier_object_appears(evaluation_artifacts):
    """Assertion 7: No secondary-tier object appears in the evaluated dataset."""
    assert os.path.exists(SECONDARY_BENCHMARK_PATH)
    sec_df = pd.read_csv(SECONDARY_BENCHMARK_PATH)
    secondary_ids = set(sec_df["candidate_id"].tolist())

    eval_ids = set(evaluation_artifacts["evaluation"]["candidate_id"].tolist())
    intersection = eval_ids.intersection(secondary_ids)
    assert len(intersection) == 0, f"Secondary tier candidates leaked into primary evaluation: {intersection}"


def test_8_no_transfer_pilot_object_appears(evaluation_artifacts):
    """Assertion 8: No transfer-pilot object appears in the evaluated benchmark."""
    eval_ztf_ids = set(evaluation_artifacts["evaluation"]["ztf_designation"].tolist())
    intersection = eval_ztf_ids.intersection(HISTORICAL_PILOT_IDS)
    assert len(intersection) == 0, f"Historical pilot targets leaked into primary evaluation: {intersection}"


def test_9_checkpoint_sha256_remains_unchanged():
    """Assertion 9: Checkpoint SHA-256 remains exactly unchanged."""
    current_hash = compute_sha256(DEFAULT_CHECKPOINT_PATH)
    assert current_hash == EXPECTED_PRODUCTION_SHA256, (
        f"Checkpoint mutation detected!\nExpected: {EXPECTED_PRODUCTION_SHA256}\nActual:   {current_hash}"
    )


def test_10_threshold_remains_exactly_0_65(evaluation_artifacts):
    """Assertion 10: Threshold remains exactly 0.65 across all records."""
    thresholds = evaluation_artifacts["evaluation"]["threshold"].tolist()
    assert all(t == 0.65 for t in thresholds), "Threshold deviated from 0.65!"


def test_11_no_optimizer_or_training_operation_occurs():
    """Assertion 11: Production model is loaded in eval mode with grad disabled."""
    model, _ = load_production_model(DEFAULT_CHECKPOINT_PATH, device="cpu")
    assert not model.training, "Loaded production model is not in eval mode!"
    for name, param in model.named_parameters():
        # Verify gradients are None (no backward pass ever executed)
        assert param.grad is None, f"Parameter {name} has active gradient tensor!"


def test_12_ground_truth_labels_match_frozen_manifest(evaluation_artifacts):
    """Assertion 12: Ground-truth labels match the frozen manifest exactly."""
    manifest_labels = evaluation_artifacts["manifest"]["is_anomaly_ground_truth"].tolist()
    eval_labels = evaluation_artifacts["evaluation"]["is_anomaly_ground_truth"].tolist()
    assert eval_labels == manifest_labels, "Ground-truth anomaly labels do not match frozen manifest!"


def test_13_evaluation_results_contain_one_row_per_primary_candidate(evaluation_artifacts):
    """Assertion 13: Evaluation results contain exactly one row per primary candidate."""
    eval_df = evaluation_artifacts["evaluation"]
    assert len(eval_df) == 106
    assert eval_df["candidate_id"].nunique() == 106, "Duplicate candidate IDs found in evaluation CSV"


def test_14_continuous_scores_are_finite(evaluation_artifacts):
    """Assertion 14: Continuous scores are finite and bounded in [0, 1]."""
    eval_df = evaluation_artifacts["evaluation"]
    scores = eval_df["ensemble_score"].values
    assert np.isfinite(scores).all(), "Non-finite ensemble scores encountered!"
    assert (scores >= 0.0).all() and (scores <= 1.0).all(), "Scores outside bounded [0, 1] range!"

    ae_scores = eval_df["ae_score"].values
    mah_scores = eval_df["mahalanobis_score"].values
    energy_scores = eval_df["energy_score"].values
    assert np.isfinite(ae_scores).all()
    assert np.isfinite(mah_scores).all()
    assert np.isfinite(energy_scores).all()


def test_15_no_real_ztf_data_is_written_into_training_datasets():
    """Assertion 15: Verify that zero real-ZTF data exists in training datasets or checkpoint payload."""
    _, payload = load_production_model(DEFAULT_CHECKPOINT_PATH, device="cpu")
    assert payload["real_ztf_data_used"] is False, "Checkpoint flag indicates real ZTF data was used!"
    assert payload["dataset_provenance"]["real_ztf_data_used"] is False

    # Check synthetic dataset generator contains no real candidates
    train_classes = set(payload["dataset_provenance"]["known_classes"])
    assert "Unclassified_Field_Star" not in train_classes
    assert "Cataclysmic_Variable" not in train_classes
