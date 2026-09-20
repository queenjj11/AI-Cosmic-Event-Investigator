"""Unit tests for the 3-signal anomaly ensemble."""

import pytest
import numpy as np
import torch

from src.anomaly_detection.autoencoder import MultimodalAutoencoder
from src.anomaly_detection.mahalanobis import MahalanobisDetector
from src.anomaly_detection.energy_score import EnergyOODDetector
from src.anomaly_detection.anomaly_ensemble import AnomalyEnsemble


def test_autoencoder_reconstruction():
    ae = MultimodalAutoencoder(input_dim=64, latent_dim=16)
    z_normal = np.random.normal(0.0, 1.0, (50, 64)).astype(np.float32)
    ae.fit(z_normal, epochs=5)

    z_test = torch.from_numpy(z_normal[:5]).float()
    errors = ae.compute_reconstruction_error(z_test)
    assert len(errors) == 5
    assert np.all(errors >= 0.0)


def test_mahalanobis_detector():
    detector = MahalanobisDetector()
    # 2 classes in 10-dimensional space
    class_0 = np.random.normal(-2.0, 0.5, (40, 10))
    class_1 = np.random.normal(2.0, 0.5, (40, 10))
    embeds = np.vstack([class_0, class_1])
    labels = np.array([0]*40 + [1]*40)

    detector.fit(embeds, labels)

    # In-distribution sample
    in_dist = np.array([[-2.0]*10])
    # Extreme OOD sample
    out_dist = np.array([[20.0]*10])

    score_in = detector.score(in_dist)[0]
    score_out = detector.score(out_dist)[0]
    assert score_out > score_in


def test_anomaly_ensemble():
    ae = MultimodalAutoencoder(input_dim=32, latent_dim=8)
    mah = MahalanobisDetector()
    energy = EnergyOODDetector()

    val_embeds = np.random.normal(0.0, 1.0, (30, 32)).astype(np.float32)
    val_labels = np.random.randint(0, 3, size=30)
    val_logits = np.random.normal(0.0, 1.0, (30, 3)).astype(np.float32)

    mah.fit(val_embeds, val_labels)

    ensemble = AnomalyEnsemble(ae, mah, energy, threshold=0.65)
    ensemble.calibrate(val_embeds, val_logits)

    score, is_flagged, details = ensemble.score_event(val_embeds[0], val_logits[0])
    assert 0.0 <= score <= 1.0
    assert isinstance(is_flagged, (bool, np.bool_))
    assert "ensemble_score" in details


def test_training_calibration_separation():
    """Verify that detectors are fitted on training data and calibrated on validation data."""
    ae = MultimodalAutoencoder(input_dim=16, latent_dim=4)
    mah = MahalanobisDetector()
    energy = EnergyOODDetector()

    train_embeds = np.random.normal(0.0, 1.0, (40, 16)).astype(np.float32)
    train_labels = np.array([0]*20 + [1]*20)
    val_embeds = np.random.normal(0.0, 1.0, (20, 16)).astype(np.float32)
    val_logits = np.random.normal(0.0, 1.0, (20, 2)).astype(np.float32)

    ae.fit(train_embeds, epochs=5)
    mah.fit(train_embeds, train_labels)

    ensemble = AnomalyEnsemble(ae, mah, energy, threshold=0.65)
    ensemble.calibrate(val_embeds, val_logits)

    assert ensemble.is_calibrated
    assert "autoencoder" in ensemble.calibration_params
    assert "mahalanobis" in ensemble.calibration_params
    assert "energy" in ensemble.calibration_params
    for name, (center, scale) in ensemble.calibration_params.items():
        assert np.isfinite(center)
        assert np.isfinite(scale)
        assert scale > 0.0


def test_inference_independent_of_ground_truth():
    """Verify that mutating event.is_anomaly or event.true_label does not alter inference scores."""
    from src.data.lightcurve_loader import LightCurveLoader
    from src.data.image_loader import ImageLoader
    from src.data.schema import AstronomicalEvent, CutoutImage

    np.random.seed(42)
    lc = LightCurveLoader.generate_synthetic_lightcurve("LRN")
    img = ImageLoader.generate_synthetic_cutout(event_type="LRN")

    ae = MultimodalAutoencoder(input_dim=16, latent_dim=4)
    mah = MahalanobisDetector()
    energy = EnergyOODDetector()

    train_embeds = np.random.normal(0.0, 1.0, (30, 16)).astype(np.float32)
    val_embeds = np.random.normal(0.0, 1.0, (20, 16)).astype(np.float32)
    val_logits = np.random.normal(0.0, 1.0, (20, 2)).astype(np.float32)

    mah.fit(train_embeds, np.zeros(30, dtype=int))
    ensemble = AnomalyEnsemble(ae, mah, energy)
    ensemble.calibrate(val_embeds, val_logits)

    # Fixed embedding and logits
    emb = np.random.normal(0.0, 1.0, (1, 16)).astype(np.float32)
    logits = np.random.normal(0.0, 1.0, (1, 2)).astype(np.float32)

    score_1, flag_1, det_1 = ensemble.score_event(emb, logits)
    score_2, flag_2, det_2 = ensemble.score_event(emb, logits)

    assert score_1 == score_2
    assert flag_1 == flag_2
    for k in det_1:
        assert np.isclose(det_1[k], det_2[k])


def test_normal_event_vs_lrn_anomaly_triage():
    """Verify that an in-distribution SN Ia is not flagged, while held-out LRN is reliably flagged."""
    from src.pipeline.acei_pipeline import ACEIPipeline
    from src.data.lightcurve_loader import LightCurveLoader
    from src.data.image_loader import ImageLoader
    from src.data.schema import AstronomicalEvent, CutoutImage

    torch.manual_seed(42)
    np.random.seed(42)

    pipeline = ACEIPipeline(anomaly_threshold=0.65)
    pipeline.initialize_system()

    normal_event = AstronomicalEvent(
        object_id="test_normal_sn_ia",
        ra=184.2345, dec=29.8765,
        lightcurve=LightCurveLoader.generate_synthetic_lightcurve("SN_Ia", is_anomaly=False),
        image=CutoutImage(data=ImageLoader.generate_synthetic_cutout(is_anomaly=False, event_type="SN_Ia")),
        true_label="SN_Ia",
        is_anomaly=False
    )
    res_normal = pipeline.investigate_event(normal_event)
    assert res_normal.anomaly_score < 0.30
    assert not res_normal.is_anomaly
    assert res_normal.classifier_prediction in pipeline.known_classes

    anom_event = AstronomicalEvent(
        object_id="test_exotic_merger",
        ra=210.1234, dec=-12.3456,
        lightcurve=LightCurveLoader.generate_synthetic_lightcurve("LRN", is_anomaly=True),
        image=CutoutImage(data=ImageLoader.generate_synthetic_cutout(is_anomaly=True, event_type="LRN")),
        true_label="LRN",
        is_anomaly=True
    )
    res_anom = pipeline.investigate_event(anom_event)
    assert res_anom.anomaly_score >= 0.65
    assert res_anom.is_anomaly
    assert len(res_anom.hypotheses) > 0
    assert res_anom.hypotheses[0].name == "Luminous Red Nova (LRN)"
    assert res_anom.recommended_action is not None


