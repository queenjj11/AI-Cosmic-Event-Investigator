"""Unit tests for coordinate-based ZTF object ingestion and provenance loader."""

import pytest
import math
import numpy as np
import torch

from src.data.ztf_object_loader import (
    calculate_angular_separation_arcsec,
    ZTFObjectLoader,
    AssociatedZTFEvent,
    ZTFObjectMetadata,
    ZTFSourceMatch
)
from src.data.real_ztf_preprocessing import RealZTFPreprocessor
from src.models.lightcurve_encoder import LightCurveEncoder


def test_angular_separation_calculation():
    """Test 1: Verify exact Haversine great-circle angular separation calculation."""
    # 1. Identical positions
    sep_zero = calculate_angular_separation_arcsec(120.0, 30.0, 120.0, 30.0)
    assert sep_zero == pytest.approx(0.0, abs=1e-6)

    # 2. Offset along Declination (1.0 arcsecond)
    sep_dec = calculate_angular_separation_arcsec(120.0, 30.0, 120.0, 30.0 + 1.0 / 3600.0)
    assert sep_dec == pytest.approx(1.0, abs=1e-4)

    # 3. Offset along Right Ascension at Declination = 60 deg (cos(60) = 0.5)
    # A true 1.0 arcsecond on the sphere corresponds to delta_RA = 1.0 / cos(60 deg) = 2.0 arcsec
    sep_ra = calculate_angular_separation_arcsec(120.0, 60.0, 120.0 + (1.0 / 3600.0) / math.cos(math.radians(60.0)), 60.0)
    assert sep_ra == pytest.approx(1.0, abs=1e-4)

    # 4. Boundary wrap around RA = 0 / 360 deg
    sep_wrap = calculate_angular_separation_arcsec(0.0001, 10.0, 359.9999, 10.0)
    expected_arcsec = (0.0002 * 3600.0) * math.cos(math.radians(10.0))
    assert sep_wrap == pytest.approx(expected_arcsec, abs=1e-3)


def test_rejection_of_mismatched_coordinates():
    """Test 2: Ensure observations with angular separation > 1.5 arcsec are rejected."""
    loader = ZTFObjectLoader(max_search_radius_arcsec=1.5)
    target_ra = 150.0
    target_dec = 20.0

    # Create synthetic records: one close (0.3''), one distant (14.6 arcminutes, replicating the flawed OID bug)
    records = [
        {
            "oid": "1001",
            "filtercode": "zg",
            "ra": str(target_ra + (0.3 / 3600.0) / math.cos(math.radians(target_dec))),
            "dec": str(target_dec),
            "mjd": "58500.1",
            "mag": "18.0",
            "magerr": "0.05",
            "catflags": "0"
        },
        {
            "oid": "1002",
            "filtercode": "zr",
            "ra": str(target_ra + 14.6 / 60.0),  # 14.6 arcminutes away!
            "dec": str(target_dec),
            "mjd": "58500.2",
            "mag": "17.5",
            "magerr": "0.04",
            "catflags": "0"
        }
    ]

    event = loader.associate_sources(target_ra, target_dec, records, object_id="TEST_MISMATCH")
    assert event.metadata.retrieval_status == "SUCCESS"
    assert "zg" in event.matched_sources
    assert "zr" not in event.matched_sources
    assert len(event.raw_records) == 1
    assert event.raw_records[0]["oid"] == "1001"

    # Now test with ONLY mismatched records -> should fail retrieval
    mismatched_only = [records[1]]
    failed_event = loader.associate_sources(target_ra, target_dec, mismatched_only, object_id="TEST_ALL_MISMATCH")
    assert failed_event.metadata.retrieval_status == "FAILED"
    assert failed_event.metadata.retrieval_failed is True
    assert "Zero observations within maximum search radius" in failed_event.metadata.failure_reason


def test_correct_multiband_association():
    """Test 3: Verify multi-band observations (zg, zr, zi) within tolerance are associated."""
    loader = ZTFObjectLoader(max_search_radius_arcsec=1.5)
    target_ra = 200.0
    target_dec = -10.0

    records = [
        # zg (sep ~ 0.2'')
        {"oid": "ZG_01", "filtercode": "zg", "ra": str(target_ra), "dec": str(target_dec + 0.2 / 3600.0),
         "mjd": "58600.1", "mag": "19.0", "magerr": "0.08", "catflags": "0", "field": "500"},
        # zr (sep ~ 0.3'')
        {"oid": "ZR_02", "filtercode": "zr", "ra": str(target_ra + (0.3 / 3600.0) / math.cos(math.radians(target_dec))),
         "dec": str(target_dec), "mjd": "58601.2", "mag": "18.5", "magerr": "0.06", "catflags": "0", "field": "500"},
        # zi (sep ~ 0.4'')
        {"oid": "ZI_03", "filtercode": "zi", "ra": str(target_ra), "dec": str(target_dec - 0.4 / 3600.0),
         "mjd": "58602.3", "mag": "18.2", "magerr": "0.09", "catflags": "0", "field": "500"},
    ]

    event = loader.associate_sources(target_ra, target_dec, records, object_id="TEST_MULTIBAND")
    assert event.metadata.retrieval_status == "SUCCESS"
    assert set(event.available_filters) == {"zg", "zr", "zi"}
    assert event.missing_filters == []
    assert event.partial_filter_coverage is False
    assert len(event.raw_records) == 3
    assert event.matched_sources["zg"].oid == "ZG_01"
    assert event.matched_sources["zr"].oid == "ZR_02"
    assert event.matched_sources["zi"].oid == "ZI_03"


def test_ambiguous_source_handling():
    """Test 4: Verify ambiguous distinct candidate sources trigger safe rejection (AMBIGUOUS)."""
    # Use min_ambiguity_gap_arcsec = 0.3
    loader = ZTFObjectLoader(max_search_radius_arcsec=1.5, min_ambiguity_gap_arcsec=0.3, min_source_resolution_arcsec=0.15)
    target_ra = 180.0
    target_dec = 45.0

    # Two distinct stars (Star A at sep=0.4'', Star B at sep=0.55'')
    # Separation gap = 0.15'' < 0.3'' threshold!
    # Mutual distance between them = 0.55'' - 0.40'' = 0.15'' >= 0.15''
    records = [
        {"oid": "STAR_A", "filtercode": "zg", "ra": str(target_ra), "dec": str(target_dec + 0.4 / 3600.0),
         "mjd": "58700.1", "mag": "19.0", "magerr": "0.1", "catflags": "0"},
        {"oid": "STAR_B", "filtercode": "zg", "ra": str(target_ra), "dec": str(target_dec - 0.55 / 3600.0),
         "mjd": "58700.2", "mag": "19.2", "magerr": "0.1", "catflags": "0"}
    ]

    event = loader.associate_sources(target_ra, target_dec, records, object_id="TEST_AMBIGUOUS")
    assert event.metadata.retrieval_status == "AMBIGUOUS"
    assert event.metadata.retrieval_failed is True
    assert "Ambiguous distinct candidate sources in passband" in event.metadata.failure_reason
    assert len(event.raw_records) == 0


def test_missing_source_handling():
    """Test 5: Verify missing sources or empty inputs return retrieval_failed=True with diagnostics."""
    loader = ZTFObjectLoader(max_search_radius_arcsec=1.5)
    target_ra = 100.0
    target_dec = 0.0

    event = loader.associate_sources(target_ra, target_dec, [], object_id="TEST_EMPTY")
    assert event.metadata.retrieval_status == "FAILED"
    assert event.metadata.retrieval_failed is True
    assert event.metadata.failure_reason == "No observations returned from data source."
    assert event.total_raw_observations == 0


def test_provenance_preservation():
    """Test 6: Verify observation columns and object-level metadata are fully preserved."""
    loader = ZTFObjectLoader(max_search_radius_arcsec=1.5)
    target_ra = 157.3415
    target_dec = 29.51067

    records = [
        {
            "oid": "1665105100004224",
            "filtercode": "zg",
            "ra": "157.34151",
            "dec": "29.51068",
            "mjd": "58492.4512",
            "mag": "17.82",
            "magerr": "0.035",
            "catflags": "0",
            "chi": "1.05",
            "sharp": "-0.02",
            "field": "651"
        }
    ]

    event = loader.associate_sources(
        target_ra=target_ra,
        target_dec=target_dec,
        records=records,
        object_id="SN_2019np",
        classification="SN Ia",
        classification_source="IAU Transient Name Server (TNS)",
        source_url="https://www.wis-tns.org/object/2019np"
    )

    assert event.metadata.object_id == "SN_2019np"
    assert event.metadata.classification == "SN Ia"
    assert event.metadata.classification_source == "IAU Transient Name Server (TNS)"
    assert event.metadata.source_url == "https://www.wis-tns.org/object/2019np"
    assert event.metadata.target_ra == target_ra
    assert event.metadata.target_dec == target_dec
    assert event.metadata.query_timestamp != ""

    # Check preserved columns in matched observation record
    rec = event.raw_records[0]
    assert rec["oid"] == "1665105100004224"
    assert rec["filtercode"] == "zg"
    assert rec["mjd"] == "58492.4512"
    assert rec["mag"] == "17.82"
    assert rec["magerr"] == "0.035"
    assert rec["catflags"] == "0"
    assert rec["chi"] == "1.05"
    assert rec["sharp"] == "-0.02"


def test_deterministic_preprocessing():
    """Test 7: Verify that preprocessing the same associated event yields bitwise identical tensors."""
    loader = ZTFObjectLoader(max_search_radius_arcsec=1.5)
    preprocessor = RealZTFPreprocessor(max_sequence_length=50)

    target_ra = 150.0
    target_dec = 20.0
    records = [
        {"oid": "1", "filtercode": "zg", "ra": str(target_ra), "dec": str(target_dec),
         "mjd": str(58000.0 + i * 2.0), "mag": str(18.0 + 0.5 * math.sin(i)), "magerr": "0.05", "catflags": "0"}
        for i in range(20)
    ]

    event = loader.associate_sources(target_ra, target_dec, records, object_id="TEST_DET")
    prep1 = preprocessor.process_records(event.raw_records, object_id="TEST_DET")
    prep2 = preprocessor.process_records(event.raw_records, object_id="TEST_DET")

    assert torch.equal(prep1.feature_tensor, prep2.feature_tensor)
    assert torch.equal(prep1.mask_tensor, prep2.mask_tensor)
    assert prep1.valid_token_count == prep2.valid_token_count


def test_finite_encoder_embedding():
    """Test 8: Verify preprocessed associated events yield strictly finite LightCurveEncoder embeddings."""
    loader = ZTFObjectLoader(max_search_radius_arcsec=1.5)
    preprocessor = RealZTFPreprocessor(max_sequence_length=50)
    encoder = LightCurveEncoder()
    encoder.eval()

    target_ra = 150.0
    target_dec = 20.0
    records = [
        {"oid": "1", "filtercode": "zg", "ra": str(target_ra), "dec": str(target_dec),
         "mjd": str(58000.0 + i * 1.5), "mag": str(17.5 + (i - 10)**2 * 0.02), "magerr": "0.04", "catflags": "0"}
        for i in range(30)
    ]

    event = loader.associate_sources(target_ra, target_dec, records, object_id="TEST_EMBEDDING")
    prep = preprocessor.process_records(event.raw_records, object_id="TEST_EMBEDDING")

    feat = prep.feature_tensor.unsqueeze(0)
    mask = prep.mask_tensor.unsqueeze(0)

    with torch.no_grad():
        emb = encoder(feat, mask=mask)

    assert emb.shape == (1, 128)
    assert bool(torch.isfinite(emb).all().item()) is True
    assert not torch.isnan(emb).any()
    assert not torch.isinf(emb).any()
    assert torch.norm(emb, p=2).item() > 0.0


def test_partial_filter_coverage_handling():
    """Amendment 1 Test: Verify partial filter coverage (e.g. zg + zr only) is supported and tracked."""
    loader = ZTFObjectLoader(max_search_radius_arcsec=1.5, standard_filters=("zg", "zr", "zi"))
    target_ra = 100.0
    target_dec = 50.0

    # Only zg and zr provided; zi is missing
    records = [
        {"oid": "ZG_1", "filtercode": "zg", "ra": str(target_ra), "dec": str(target_dec),
         "mjd": "58001.0", "mag": "18.0", "magerr": "0.05", "catflags": "0"},
        {"oid": "ZR_2", "filtercode": "zr", "ra": str(target_ra), "dec": str(target_dec),
         "mjd": "58002.0", "mag": "17.6", "magerr": "0.04", "catflags": "0"}
    ]

    event = loader.associate_sources(target_ra, target_dec, records, object_id="TEST_PARTIAL")
    assert event.metadata.retrieval_status == "SUCCESS"
    assert event.available_filters == ["zg", "zr"]
    assert event.missing_filters == ["zi"]
    assert event.partial_filter_coverage is True
