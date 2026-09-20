"""Unit and integration tests for data schemas, loaders, and preprocessing."""

import pytest
import numpy as np
import os
import tempfile
import json

from src.data.schema import Observation, LightCurve, CutoutImage, AstronomicalEvent
from src.data.ztf_loader import ZTFLoader
from src.data.image_loader import ImageLoader
from src.data.lightcurve_loader import LightCurveLoader
from src.data.preprocessing import Preprocessor
from src.data.splitter import ObjectLevelSplitter
from src.data.dataset_builder import AstronomicalDataset, generate_benchmark_events


def test_schema_lightcurve():
    lc = LightCurve()
    lc.add_observation(time=59000.1, band="g", flux=10.5, flux_err=0.5)
    lc.add_observation(time=58999.0, band="r", flux=15.2, flux_err=0.7)
    lc.add_observation(time=59001.2, band="g", flux=8.3, flux_err=0.4)

    assert len(lc) == 3
    times, fluxes, errs = lc.get_band_data("g")
    assert len(times) == 2
    assert times[0] < times[1]  # Verify chronological ordering


def test_ztf_loader_parsing():
    alert_mock = {
        "objectId": "ZTF23test",
        "candidate": {
            "ra": 150.5,
            "dec": 25.4,
            "jd": 2459000.5,
            "fid": 1,
            "magpsf": 18.5,
            "sigmapsf": 0.05
        },
        "prv_candidates": [
            {"jd": 2458995.5, "fid": 2, "magpsf": 19.0, "sigmapsf": 0.08}
        ]
    }
    event = ZTFLoader.parse_alert_dict(alert_mock)
    assert event.object_id == "ZTF23test"
    assert len(event.lightcurve) == 2
    assert event.ra == 150.5


def test_image_loader_preprocessing():
    loader = ImageLoader(target_size=(64, 64))
    raw_img = np.random.normal(10.0, 2.0, (3, 64, 64))
    processed = loader.preprocess_image(raw_img)

    assert processed.shape == (3, 64, 64)
    assert np.all(processed >= 0.0)
    assert np.all(processed <= 1.0)
    assert processed.dtype == np.float32


def test_object_level_splitter():
    events = generate_benchmark_events(num_known=40, num_anomalies=10)
    splitter = ObjectLevelSplitter(
        known_classes=["SN_Ia", "SN_II", "Stellar_Flare", "Variable_Star"],
        held_out_classes=["LRN", "SLSN", "TDE"]
    )
    splits = splitter.split_events(events)

    # Verify no object leakage between train, val, and test
    train_set = set(splits.train_ids)
    val_set = set(splits.val_ids)
    test_known_set = set(splits.test_known_ids)
    anomaly_set = set(splits.test_anomaly_ids)

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_known_set)) == 0
    assert len(val_set.intersection(test_known_set)) == 0
    assert len(anomaly_set) == 10  # All 10 anomalies cleanly partitioned
