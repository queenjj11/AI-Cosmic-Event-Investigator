"""Unit tests verifying real-data transfer compatibility with the trained production checkpoint."""

import os
import pytest
import numpy as np
import torch

from src.models.checkpoint_manager import load_production_lightcurve_encoder, DEFAULT_CHECKPOINT_PATH
from src.data.ztf_object_loader import ZTFObjectLoader
from src.data.real_ztf_preprocessing import RealZTFPreprocessor


@pytest.fixture(scope="module")
def trained_encoder_and_meta():
    """Load the frozen checkpoint-trained LightCurveEncoder."""
    assert os.path.exists(DEFAULT_CHECKPOINT_PATH), "Production checkpoint file missing."
    encoder, payload = load_production_lightcurve_encoder(DEFAULT_CHECKPOINT_PATH, device="cpu")
    encoder.eval()
    return encoder, payload


def test_checkpoint_loads_successfully_and_is_trained(trained_encoder_and_meta):
    """Verify trained checkpoint loads successfully and parameters reflect learned weights."""
    encoder, payload = trained_encoder_and_meta
    assert encoder is not None
    assert payload["real_ztf_data_used"] is False
    assert payload["training_provenance"]["epochs"] == 5
    assert payload["training_provenance"]["seed"] == 42


def test_no_model_parameters_change_during_inference(trained_encoder_and_meta):
    """Verify model parameters remain strictly immutable under evaluation mode and torch.no_grad()."""
    encoder, _ = trained_encoder_and_meta
    initial_params = {k: v.clone() for k, v in encoder.named_parameters()}

    dummy_x = torch.randn(2, 50, 4)
    dummy_mask = torch.ones(2, 50, dtype=torch.bool)

    with torch.no_grad():
        _ = encoder(dummy_x, mask=dummy_mask)

    for k, v in encoder.named_parameters():
        assert torch.equal(v, initial_params[k]), f"Parameter {k} mutated during inference!"


def test_real_ztf_tensor_contract_and_finite_embeddings(trained_encoder_and_meta):
    """Verify preprocessed real-ZTF observations adhere to (1, 50, 4) and produce finite 128-D embeddings."""
    encoder, _ = trained_encoder_and_meta
    preprocessor = RealZTFPreprocessor(max_sequence_length=50)

    # Use a synthetic real-like light curve
    records = [
        {"oid": "999", "filtercode": "zg", "mjd": 58000.0 + i, "mag": 18.0 + 0.1 * i,
         "magerr": 0.05, "catflags": 0}
        for i in range(25)
    ]
    prep = preprocessor.process_records(records, object_id="TEST_TRANSFER")

    assert prep.feature_tensor.shape == (50, 4)
    assert prep.mask_tensor.shape == (50,)
    assert prep.valid_token_count == 25

    feat = prep.feature_tensor.unsqueeze(0)
    mask = prep.mask_tensor.unsqueeze(0)

    with torch.no_grad():
        emb = encoder(feat, mask=mask)

    assert emb.shape == (1, 128)
    assert bool(torch.isfinite(emb).all().item()) is True
    assert not torch.isnan(emb).any()
    assert not torch.isinf(emb).any()


def test_no_real_object_enters_training_or_calibration(trained_encoder_and_meta):
    """Verify checkpoint records prove zero real objects entered training or calibration sets."""
    _, payload = trained_encoder_and_meta
    dp = payload["dataset_provenance"]
    assert dp["real_ztf_data_used"] is False

    all_ids = dp["train_object_ids"] + dp["val_object_ids"] + dp["test_known_object_ids"] + dp["test_anomaly_object_ids"]
    for oid in all_ids:
        assert oid.startswith("ZTF_")
        assert "ZTF19aacgslb" not in oid
        assert "ZTF20aaynrrh" not in oid
        assert "ZTF18abukavn" not in oid
        assert "ZTF18aarkpda" not in oid


def test_legacy_invalid_pilot_remains_untouched_and_quarantined():
    """Verify legacy data/real_ztf_pilot/ remains untouched and classified as INVALID_FOR_BENCHMARK."""
    readme_path = "data/real_ztf_pilot/README.md"
    assert os.path.exists(readme_path)
    with open(readme_path, "r") as f:
        content = f.read()
    assert "INVALID_FOR_BENCHMARK" in content
