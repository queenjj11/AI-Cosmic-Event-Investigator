"""Focused unit and integration tests for real ZTF lightcurve preprocessing and zero-shot encoder compatibility."""

import pytest
import torch
import numpy as np

from src.models.lightcurve_encoder import LightCurveEncoder
from src.data.real_ztf_preprocessing import (
    ZTF_BAND_MAP,
    ZTFQualityFilter,
    PhotometryConverter,
    TransientWindowSelector,
    FluxNormalizer,
    TokenSequenceConstructor,
    RealZTFPreprocessor,
    BAD_CATFLAGS_CLOUD_MOON,
    BAD_CATFLAGS_SEVERE_ARTIFACTS
)


def test_ztf_band_mapping():
    """Verify official ZTF passbands map to exact internal ACEI indices: g->0, r->1, i->2."""
    assert ZTF_BAND_MAP["zg"] == 0
    assert ZTF_BAND_MAP["g"] == 0
    assert ZTF_BAND_MAP[1] == 0

    assert ZTF_BAND_MAP["zr"] == 1
    assert ZTF_BAND_MAP["r"] == 1
    assert ZTF_BAND_MAP[2] == 1

    assert ZTF_BAND_MAP["zi"] == 2
    assert ZTF_BAND_MAP["i"] == 2
    assert ZTF_BAND_MAP[3] == 2


def test_ztf_quality_filtering():
    """Verify catflags bitmasks (cloud/moon and artifacts) and bad uncertainties are correctly filtered."""
    filter_mod = ZTFQualityFilter()
    raw_records = [
        {"mjd": 59000.1, "mag": 17.5, "magerr": 0.05, "filtercode": "zg", "catflags": 0},          # Valid clean
        {"mjd": 59001.1, "mag": 17.6, "magerr": 0.05, "filtercode": "zr", "catflags": 32768},      # Cloud bit (filtered)
        {"mjd": 59002.1, "mag": 17.7, "magerr": 0.05, "filtercode": "zg", "catflags": 15},         # Artifact bits (filtered)
        {"mjd": 59003.1, "mag": 17.8, "magerr": -0.1, "filtercode": "zg", "catflags": 0},          # Negative error (filtered)
        {"mjd": 59004.1, "mag": 17.9, "magerr": 2.5, "filtercode": "zg", "catflags": 0},           # Exploded error (filtered)
        {"mjd": 59005.1, "mag": 99.0, "magerr": 0.05, "filtercode": "zg", "catflags": 0},          # Non-detection/unphysical (filtered)
        {"mjd": 59006.1, "mag": 18.0, "magerr": 0.06, "filtercode": "unsupported", "catflags": 0}, # Invalid band (filtered)
        {"mjd": 59007.1, "mag": 18.1, "magerr": 0.04, "filtercode": "zi", "catflags": 0}           # Valid clean
    ]

    clean, report = filter_mod.filter_records(raw_records)
    assert len(clean) == 2
    assert report.total_raw_observations == 8
    assert report.passed_observations == 2
    assert report.removed_cloud_catflags == 1
    assert report.removed_severe_catflags == 1
    assert report.removed_bad_uncertainty == 2
    assert report.removed_unphysical_mag == 1
    assert report.removed_unsupported_band == 1


def test_negative_difference_flux_preservation():
    """Verify difference fluxes with negative values (subtraction noise) are preserved without clamping."""
    record_neg = {"forcediffimflux": -15.4, "forcediffimfluxunc": 3.2, "clean_mag": 20.0, "clean_magerr": 0.2}
    flux, err, is_neg = PhotometryConverter.process_record_photometry(record_neg)
    assert flux == -15.4
    assert err == 3.2
    assert is_neg is True


def test_flux_normalization_and_error_preservation():
    """Verify peak-based normalization scales peak to 1.5 and maintains error proportions."""
    records = [
        {"clean_mag": 17.5, "clean_flux": 100.0, "clean_flux_err": 5.0},
        {"clean_mag": 18.25, "clean_flux": 50.0, "clean_flux_err": 2.5},
        {"clean_mag": 19.0, "clean_flux": 20.0, "clean_flux_err": 1.0}
    ]
    norm_recs, report = FluxNormalizer.normalize_peak_scaled(records, target_peak_scale=1.5)
    assert len(norm_recs) == 3
    assert pytest.approx(norm_recs[0]["norm_flux"], 1e-4) == 1.5
    assert pytest.approx(norm_recs[1]["norm_flux"], 1e-4) == 0.75
    # Relative error ratio must be invariant: sigma / F == norm_sigma / norm_F
    assert pytest.approx(norm_recs[0]["norm_flux_err"] / norm_recs[0]["norm_flux"], 1e-4) == 5.0 / 100.0


def test_relative_time_conversion():
    """Verify relative time conversion starts at t=0 and is monotonically increasing."""
    records = [
        {"clean_mjd": 59000.0, "norm_flux": 1.0, "norm_flux_err": 0.05, "clean_band": "g"},
        {"clean_mjd": 59002.5, "norm_flux": 1.2, "norm_flux_err": 0.05, "clean_band": "r"},
        {"clean_mjd": 59005.0, "norm_flux": 0.8, "norm_flux_err": 0.05, "clean_band": "g"}
    ]
    constructor = TokenSequenceConstructor(max_length=50)
    feat, mask, _ = constructor.construct_tensor(records, t_zero=59000.0)

    assert feat[0, 0].item() == 0.0
    assert feat[1, 0].item() == 2.5
    assert feat[2, 0].item() == 5.0


def test_sequence_construction_fewer_than_50_tokens():
    """Verify light curves with <50 observations are padded with zeros and mask is False for padded tokens."""
    records = [
        {"clean_mjd": 59000.0 + i, "norm_flux": 1.0, "norm_flux_err": 0.05, "clean_band": "g"}
        for i in range(15)
    ]
    constructor = TokenSequenceConstructor(max_length=50)
    feat, mask, band_counts = constructor.construct_tensor(records, t_zero=59000.0)

    assert feat.shape == (50, 4)
    assert mask.shape == (50,)
    assert int(torch.sum(mask).item()) == 15
    assert torch.all(mask[:15] == True)
    assert torch.all(mask[15:] == False)
    assert torch.all(feat[15:, :] == 0.0)
    assert band_counts["g"] == 15


def test_sequence_construction_more_than_50_tokens():
    """Verify light curves with >50 observations are intelligently binned/subsampled to exactly 50 tokens."""
    records = [
        {"clean_mjd": 59000.0 + i * 0.4, "norm_flux": float(1.0 + np.sin(i * 0.1)), "norm_flux_err": 0.05, "clean_band": "g" if i % 2 == 0 else "r"}
        for i in range(120)
    ]
    constructor = TokenSequenceConstructor(max_length=50)
    feat, mask, band_counts = constructor.construct_tensor(records, t_zero=59000.0)

    assert feat.shape == (50, 4)
    assert mask.shape == (50,)
    assert int(torch.sum(mask).item()) == 50
    assert torch.all(mask == True)
    assert band_counts["g"] > 0
    assert band_counts["r"] > 0


def test_real_ztf_zero_shot_encoder_inference():
    """Verify end-to-end preprocessed real ZTF tensors pass through the existing production LightCurveEncoder."""
    preprocessor = RealZTFPreprocessor(max_sequence_length=50)
    # Synthetic real-like ZTF table records
    sample_records = [
        {"mjd": 58500.0 + i * 1.5, "mag": 17.5 + 0.5 * np.cos(i * 0.3), "magerr": 0.04, "filtercode": "zg" if i % 3 == 0 else ("zr" if i % 3 == 1 else "zi"), "catflags": 0}
        for i in range(40)
    ]
    event = preprocessor.process_records(sample_records, object_id="TEST_REAL_ZTF_01")

    assert event.valid_token_count == event.window_report.obs_in_window
    assert event.valid_token_count > 0
    assert event.feature_tensor.shape == (50, 4)
    assert event.mask_tensor.shape == (50,)

    # Pass through existing production LightCurveEncoder without retraining
    torch.manual_seed(42)
    encoder = LightCurveEncoder(embedding_dim=128)
    encoder.eval()

    with torch.no_grad():
        embedding = encoder(event.feature_tensor.unsqueeze(0), event.mask_tensor.unsqueeze(0))

    assert embedding.shape == (1, 128)
    emb_np = embedding.cpu().numpy()
    assert np.all(np.isfinite(emb_np))
    assert not np.any(np.isnan(emb_np))
    assert not np.any(np.isinf(emb_np))
    assert np.linalg.norm(emb_np) > 0.0
