"""Unit tests for the quantitative anomaly evaluation pipeline."""

import copy
import numpy as np
import pytest
import torch

from src.evaluation.anomaly_evaluation import (
    AnomalyEvaluatorPipeline,
    calculate_binary_metrics,
    calculate_curve_metrics,
    calculate_distribution_stats
)
from src.pipeline.acei_pipeline import ACEIPipeline
from src.data.dataset_builder import generate_benchmark_events
from src.data.schema import AstronomicalEvent, CutoutImage
from src.data.lightcurve_loader import LightCurveLoader
from src.data.image_loader import ImageLoader


@pytest.fixture(scope="module")
def initialized_pipeline():
    """Module-scoped pipeline initialization for fast, shared test execution."""
    torch.manual_seed(42)
    np.random.seed(42)
    pipeline = ACEIPipeline(anomaly_threshold=0.65)
    pipeline.initialize_system()
    return pipeline


def test_evaluation_uses_continuous_scores():
    """Verify that AUROC and AUPRC are calculated using continuous ranking scores."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    # Continuous scores with fine-grained separation
    continuous_scores = np.array([0.10, 0.22, 0.35, 0.68, 0.81, 0.95])
    metrics = calculate_curve_metrics(y_true, continuous_scores)

    assert "auroc" in metrics
    assert "auprc" in metrics
    assert "average_precision" in metrics
    # Continuous scores with perfect rank order should yield AUROC = 1.0
    assert metrics["auroc"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["auprc"] == pytest.approx(1.0, abs=1e-4)

    # Invert one ranking to show it sensitively reflects continuous ordering
    imperfect_scores = np.array([0.10, 0.70, 0.35, 0.68, 0.81, 0.95])
    imperfect_metrics = calculate_curve_metrics(y_true, imperfect_scores)
    assert imperfect_metrics["auroc"] < 1.0


def test_evaluation_ground_truth_isolation(initialized_pipeline):
    """Verify that stripping or mutating ground truth labels on test events does NOT alter inference."""
    torch.manual_seed(42)
    np.random.seed(42)

    event_with_gt = AstronomicalEvent(
        object_id="test_iso_obj",
        ra=100.0, dec=25.0,
        lightcurve=LightCurveLoader.generate_synthetic_lightcurve("LRN", is_anomaly=True),
        image=CutoutImage(data=ImageLoader.generate_synthetic_cutout(is_anomaly=True, event_type="LRN")),
        true_label="LRN",
        is_anomaly=True
    )

    # Create stripped event with masked/contradicting ground truth
    event_without_gt = AstronomicalEvent(
        object_id="test_iso_obj",
        ra=100.0, dec=25.0,
        lightcurve=event_with_gt.lightcurve,
        image=event_with_gt.image,
        true_label="SN_Ia",
        is_anomaly=False
    )

    res1 = initialized_pipeline.investigate_event(event_with_gt)
    res2 = initialized_pipeline.investigate_event(event_without_gt)

    assert res1.anomaly_score == pytest.approx(res2.anomaly_score, abs=1e-6)
    assert res1.is_anomaly == res2.is_anomaly
    assert res1.classifier_prediction == res2.classifier_prediction
    assert res1.classifier_confidence == pytest.approx(res2.classifier_confidence, abs=1e-6)


def test_evaluation_threshold_remains_65(initialized_pipeline):
    """Verify that default evaluation strictly respects the 0.65 threshold."""
    evaluator = AnomalyEvaluatorPipeline(pipeline=initialized_pipeline, threshold=0.65, seed=42)
    eval_data = evaluator.evaluate()

    assert eval_data["threshold"] == 0.65
    for record in eval_data["records"]:
        expected_flag = record["anomaly_score"] >= 0.65
        assert record["predicted_anomaly"] == expected_flag


def test_evaluation_produces_per_class_metrics(initialized_pipeline):
    """Verify that per-anomaly-class and distribution statistics are fully computed."""
    evaluator = AnomalyEvaluatorPipeline(pipeline=initialized_pipeline, threshold=0.65, seed=42)
    eval_data = evaluator.evaluate()

    per_class = eval_data["per_class"]
    for expected_cls in ["LRN", "SLSN", "TDE"]:
        assert expected_cls in per_class
        metrics = per_class[expected_cls]
        assert "n" in metrics
        assert "detected" in metrics
        assert "recall" in metrics
        assert "mean_score" in metrics
        assert "median_score" in metrics
        assert metrics["n"] == 10

    # Score distributions check
    dist = eval_data["score_distributions"]
    assert "known_events" in dist
    assert "all_anomalies" in dist
    assert dist["known_events"]["count"] == 20
    assert dist["all_anomalies"]["count"] == 30


def test_evaluation_deterministic_reproducibility(initialized_pipeline):
    """Verify that evaluation is strictly deterministic across repeated runs with identical seed."""
    evaluator1 = AnomalyEvaluatorPipeline(pipeline=initialized_pipeline, threshold=0.65, seed=42)
    data1 = evaluator1.evaluate()

    evaluator2 = AnomalyEvaluatorPipeline(pipeline=initialized_pipeline, threshold=0.65, seed=42)
    data2 = evaluator2.evaluate()

    assert data1["curve_metrics"]["auroc"] == pytest.approx(data2["curve_metrics"]["auroc"], abs=1e-7)
    assert data1["curve_metrics"]["auprc"] == pytest.approx(data2["curve_metrics"]["auprc"], abs=1e-7)
    assert data1["binary_metrics"]["f1"] == pytest.approx(data2["binary_metrics"]["f1"], abs=1e-7)
    assert data1["per_class"]["LRN"]["mean_score"] == pytest.approx(data2["per_class"]["LRN"]["mean_score"], abs=1e-7)
