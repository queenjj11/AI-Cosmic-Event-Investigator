"""Unit tests for the ACEI production model checkpoint manager and reload verification."""

import os
import pytest
import numpy as np
import torch

from src.models.checkpoint_manager import (
    DEFAULT_CHECKPOINT_PATH,
    CANONICAL_CLASS_TO_IDX,
    save_production_checkpoint,
    load_production_model,
    load_production_lightcurve_encoder
)
from src.models.multimodal_model import MultimodalTransientModel
from src.models.lightcurve_encoder import LightCurveEncoder


@pytest.fixture(scope="module")
def production_checkpoint():
    """Ensure production checkpoint exists and return path and loaded payloads."""
    path = DEFAULT_CHECKPOINT_PATH
    if not os.path.exists(path):
        save_production_checkpoint(output_path=path, seed=42)

    model, payload = load_production_model(path, device="cpu")
    encoder, _ = load_production_lightcurve_encoder(path, device="cpu")
    return {
        "path": path,
        "payload": payload,
        "model": model,
        "encoder": encoder
    }


def test_checkpoint_file_exists_and_non_empty(production_checkpoint):
    """Test A: Verify checkpoint file exists and has non-zero size."""
    path = production_checkpoint["path"]
    assert os.path.exists(path)
    assert os.path.getsize(path) > 10_000_000  # Expecting ~50MB


def test_checkpoint_payload_contains_all_required_fields(production_checkpoint):
    """Test B: Verify all required architectural, training, and dataset metadata exist."""
    payload = production_checkpoint["payload"]
    required_keys = [
        "model_state_dict",
        "lc_encoder_state_dict",
        "architecture",
        "class_to_idx",
        "idx_to_class",
        "training_provenance",
        "dataset_provenance",
        "metadata",
        "real_ztf_data_used"
    ]
    for key in required_keys:
        assert key in payload, f"Missing required payload key: {key}"


def test_reloaded_full_model_state_dict_bitwise_identical(production_checkpoint):
    """Test C: Verify reloaded full model weights match saved state_dict bitwise."""
    model = production_checkpoint["model"]
    payload = production_checkpoint["payload"]
    saved_sd = payload["model_state_dict"]
    reloaded_sd = model.state_dict()

    assert len(reloaded_sd) == len(saved_sd)
    max_diff = 0.0
    for key in saved_sd:
        assert key in reloaded_sd
        diff = float((saved_sd[key] - reloaded_sd[key]).abs().max().item())
        if diff > max_diff:
            max_diff = diff
        assert torch.equal(saved_sd[key], reloaded_sd[key]), f"Mismatch in {key}"

    assert max_diff == 0.0


def test_reloaded_lightcurve_encoder_state_dict_bitwise_identical(production_checkpoint):
    """Test D: Verify reloaded standalone LightCurveEncoder weights match parent model.lc_encoder."""
    model = production_checkpoint["model"]
    encoder = production_checkpoint["encoder"]

    parent_lc_sd = model.lc_encoder.state_dict()
    standalone_lc_sd = encoder.state_dict()

    assert len(standalone_lc_sd) == len(parent_lc_sd)
    max_diff = 0.0
    for key in parent_lc_sd:
        assert key in standalone_lc_sd
        diff = float((parent_lc_sd[key] - standalone_lc_sd[key]).abs().max().item())
        if diff > max_diff:
            max_diff = diff
        assert torch.equal(parent_lc_sd[key], standalone_lc_sd[key]), f"Mismatch in LC layer {key}"

    assert max_diff == 0.0


def test_identical_input_produces_identical_outputs(production_checkpoint):
    """Test E: Verify parent model.lc_encoder and reloaded standalone LightCurveEncoder produce identical outputs."""
    model = production_checkpoint["model"]
    encoder = production_checkpoint["encoder"]

    model.eval()
    encoder.eval()

    # Synthetic input sequence
    torch.manual_seed(999)
    B, seq_len = 4, 50
    time = torch.linspace(0, 40, seq_len).unsqueeze(0).repeat(B, 1).unsqueeze(-1)
    flux = torch.randn(B, seq_len, 1)
    flux_err = torch.abs(torch.randn(B, seq_len, 1)) * 0.1
    band_idx = torch.randint(0, 3, (B, seq_len, 1)).float()
    tokens = torch.cat([time, flux, flux_err, band_idx], dim=-1)
    mask = torch.ones(B, seq_len, dtype=torch.bool)
    mask[:, 30:] = False  # Partial sequence padding

    with torch.no_grad():
        out_parent = model.lc_encoder(tokens, mask=mask)
        out_standalone = encoder(tokens, mask=mask)

    max_diff = float((out_parent - out_standalone).abs().max().item())
    assert max_diff == pytest.approx(0.0, abs=1e-7)
    assert bool(torch.isfinite(out_standalone).all().item()) is True


def test_class_mapping_is_preserved_exactly(production_checkpoint):
    """Test F: Verify class_to_idx mapping matches the canonical 4 production classes."""
    payload = production_checkpoint["payload"]
    expected_mapping = {
        "SN_Ia": 0,
        "SN_II": 1,
        "Stellar_Flare": 2,
        "Variable_Star": 3
    }
    assert payload["class_to_idx"] == expected_mapping


def test_no_real_data_participates_in_checkpoint(production_checkpoint):
    """Test G: Verify checkpoint explicitly records that no real ZTF data were used."""
    payload = production_checkpoint["payload"]
    assert payload["real_ztf_data_used"] is False
    assert payload["dataset_provenance"]["real_ztf_data_used"] is False


def test_training_object_ids_are_synthetic_benchmark_only(production_checkpoint):
    """Test H: Verify all training and evaluation IDs originate strictly from synthetic benchmark generators."""
    payload = production_checkpoint["payload"]
    train_ids = payload["dataset_provenance"]["train_object_ids"]
    val_ids = payload["dataset_provenance"]["val_object_ids"]
    test_known_ids = payload["dataset_provenance"]["test_known_object_ids"]
    test_anom_ids = payload["dataset_provenance"]["test_anomaly_object_ids"]

    assert len(train_ids) == 84
    assert len(val_ids) == 16
    assert len(test_known_ids) == 20
    assert len(test_anom_ids) == 30

    all_ids = train_ids + val_ids + test_known_ids + test_anom_ids
    for oid in all_ids:
        assert oid.startswith("ZTF_")
        assert any(cls in oid for cls in ["SN_Ia", "SN_II", "Stellar_Flare", "Variable_Star", "ANOMALY"])
        # Confirm no pilot real object designations exist
        assert "ZTF19aacgslb" not in oid
        assert "ZTF20aaynrrh" not in oid
        assert "ZTF18abukavn" not in oid
        assert "ZTF18aarkpda" not in oid
        assert "ZTF_J195200" not in oid


def test_trained_vs_random_initialized_encoder_parameters_diverge(production_checkpoint):
    """Test I: Verify that the restored trained encoder parameters differ from a freshly initialized random encoder."""
    trained_encoder = production_checkpoint["encoder"]
    torch.manual_seed(12345)
    random_encoder = LightCurveEncoder(embedding_dim=128, dropout=0.2)

    trained_sd = trained_encoder.state_dict()
    random_sd = random_encoder.state_dict()

    parameter_differences = []
    for key in trained_sd:
        diff = float((trained_sd[key] - random_sd[key]).abs().max().item())
        parameter_differences.append(diff)

    max_divergence = max(parameter_differences)
    # The trained weights must diverge substantially from random initialization
    assert max_divergence > 0.01, f"Trained encoder did not diverge from random: {max_divergence}"
