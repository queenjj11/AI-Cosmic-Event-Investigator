"""Engineering verification tests for the ACEI Real-ZTF Light-Curve-First Investigator."""

import os
import json
import subprocess
import pytest
import numpy as np

from src.investigator.schemas import (
    InvestigationResult,
    EventCharacterization,
    RepresentationSummary,
    AnomalyAssessment,
    HypothesisItem,
    EvidenceItem,
    ObservationRecommendation,
    CharacterizedValue
)
from src.investigator.lightcurve_representation import LightCurveRepresentationService
from src.investigator.event_characterizer import EventCharacterizer
from src.investigator.anomaly_assessor import ScientificAnomalyAssessor
from src.investigator.investigation_service import RealZTFInvestigationService


DEMO_CANDIDATE = "CAND_SNIa_002"


def test_schema_json_and_markdown_serialization(tmp_path):
    """Verify that InvestigationResult serializes to valid JSON and readable Markdown."""
    char_val = CharacterizedValue(value=42.0, status="MEASURED", unit="days", notes="Test baseline")
    char = EventCharacterization(
        num_observations=CharacterizedValue(value=100, status="MEASURED", unit="obs"),
        num_filters=CharacterizedValue(value=2, status="MEASURED", unit="bands"),
        time_baseline_days=char_val,
        window_duration_days=CharacterizedValue(value=80.0, status="DERIVED", unit="days"),
        peak_normalized_flux=CharacterizedValue(value=2.5, status="MEASURED", unit="flux"),
        minimum_normalized_flux=CharacterizedValue(value=0.1, status="MEASURED", unit="flux"),
        median_normalized_flux=CharacterizedValue(value=0.8, status="DERIVED", unit="flux"),
        flux_std=CharacterizedValue(value=0.4, status="DERIVED", unit="flux"),
        median_flux_err=CharacterizedValue(value=0.02, status="DERIVED", unit="flux"),
        variability_amplitude=CharacterizedValue(value=2.4, status="DERIVED", unit="flux"),
        rise_time_proxy_days=CharacterizedValue(value=15.0, status="PROXY", unit="days"),
        decline_time_proxy_days=CharacterizedValue(value=45.0, status="PROXY", unit="days"),
        rise_decline_asymmetry=CharacterizedValue(value=0.5, status="PROXY", unit="dimensionless"),
        cadence_median_days=CharacterizedValue(value=1.0, status="DERIVED", unit="days"),
        cadence_min_days=CharacterizedValue(value=0.1, status="DERIVED", unit="days"),
        cadence_max_days=CharacterizedValue(value=5.0, status="DERIVED", unit="days"),
        per_band_counts={"g": 50, "r": 50},
        missing_bands=["i"],
        valid_token_count=CharacterizedValue(value=30, status="DERIVED", unit="tokens"),
        padding_fraction=CharacterizedValue(value=0.4, status="DERIVED", unit="fraction")
    )

    rep = RepresentationSummary(
        object_id="TEST_OBJ",
        embedding_dimension=128,
        embedding_norm=11.3,
        embedding=[0.1] * 128,
        valid_token_count=30,
        padding_fraction=0.4,
        available_filters=["g", "r"],
        time_span_days=80.0,
        preprocessing_status="SUCCESS"
    )

    anom = AnomalyAssessment(
        representation_available=True,
        anomaly_score_status="NOT_VALIDATED_FOR_REAL_ZTF",
        anomaly_score=None,
        anomaly_flag=None,
        detector_source="ACEI Multimodal Anomaly Ensemble",
        uncertainty_or_status="NOT_VALIDATED_FOR_REAL_ZTF",
        limitations=["Domain gap: missing images"],
        domain_gap_notes="Domain boundary enforced"
    )

    res = InvestigationResult(
        object_id="TEST_OBJ",
        metadata={"claimed_type": "SN Ia", "ra": 180.0, "dec": 0.0},
        event_characterization=char,
        representation_summary=rep,
        anomaly_assessment=anom,
        candidate_hypotheses=[
            HypothesisItem(
                rank=1,
                name="Type Ia Supernova",
                supporting_evidence=["paper_1"],
                contradictory_evidence=["none"],
                confidence_or_status="UNVALIDATED_PRIOR",
                probability=0.7,
                justification="Matches canonical rise and fall",
                distinguishing_criteria="Spectroscopic confirmation"
            )
        ],
        evidence=[
            EvidenceItem(
                doc_id="paper_1",
                title="Supernova Physics",
                relevance_score=0.85,
                snippet="Type Ia supernovae show nickel decay",
                source_uri="knowledge_base/papers/sn.md"
            )
        ],
        confidence_summary={"representation": "HIGH"},
        recommended_observations=[
            ObservationRecommendation(
                priority=1,
                action="Spectroscopy",
                scientific_rationale="Confirm Si II line",
                information_gap="No spectrum",
                target_band="optical_spectrum"
            )
        ],
        limitations=["Unvalidated on real data"],
        provenance={"raw_count": 100}
    )

    # Test JSON serialization
    json_path = tmp_path / "test_inv.json"
    res.save_json(str(json_path))
    assert os.path.exists(json_path)

    with open(json_path, "r") as f:
        loaded = json.load(f)
    assert loaded["object_id"] == "TEST_OBJ"
    assert len(loaded["representation_summary"]["embedding"]) == 128
    assert loaded["anomaly_assessment"]["anomaly_score_status"] == "NOT_VALIDATED_FOR_REAL_ZTF"

    # Test Markdown generation
    md_path = tmp_path / "test_inv.md"
    res.save_markdown(str(md_path))
    assert os.path.exists(md_path)
    with open(md_path, "r") as f:
        md_text = f.read()
    assert "# ACEI Investigation Report: TEST_OBJ" in md_text
    assert "NOT_VALIDATED_FOR_REAL_ZTF" in md_text
    assert "Type Ia Supernova" in md_text


def test_representation_output_shape_and_finite():
    """Verify LightCurveRepresentationService produces 128-D finite embeddings."""
    service = LightCurveRepresentationService()
    rep = service.represent_ztf_object(DEMO_CANDIDATE)

    assert rep.object_id == DEMO_CANDIDATE
    assert rep.embedding_dimension == 128
    assert len(rep.embedding) == 128
    assert rep.embedding_norm > 0.0
    assert np.all(np.isfinite(rep.embedding))
    assert rep.valid_token_count > 0
    assert 0.0 <= rep.padding_fraction <= 1.0
    assert rep.preprocessing_status == "SUCCESS"


def test_event_characterization_status_labels():
    """Verify EventCharacterizer derives features with strict status labels."""
    characterizer = EventCharacterizer()
    raw_path = f"data/real_ztf_benchmark/raw/{DEMO_CANDIDATE}/raw_irsa.csv"
    import csv
    with open(raw_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    char = characterizer.characterize(records=records, object_id=DEMO_CANDIDATE)

    assert char.num_observations.status == "MEASURED"
    assert char.num_observations.value > 100
    assert char.num_filters.status == "MEASURED"
    assert char.time_baseline_days.status == "MEASURED"
    assert char.window_duration_days.status == "DERIVED"
    assert char.peak_normalized_flux.status == "MEASURED"
    assert char.median_normalized_flux.status == "DERIVED"
    assert char.rise_time_proxy_days.status == "PROXY"
    assert char.decline_time_proxy_days.status == "PROXY"
    assert char.rise_decline_asymmetry.status == "PROXY"
    assert "No astrophysical classification claimed" in char.classification_disclaimer


def test_conservative_anomaly_assessor_status():
    """Verify ScientificAnomalyAssessor marks real ZTF as NOT_VALIDATED_FOR_REAL_ZTF."""
    assessor = ScientificAnomalyAssessor()
    rep = RepresentationSummary(
        object_id="TEST_REAL",
        embedding_dimension=128,
        embedding_norm=11.3,
        embedding=[0.1] * 128,
        valid_token_count=30,
        padding_fraction=0.4,
        available_filters=["g", "r"],
        time_span_days=80.0,
        preprocessing_status="SUCCESS"
    )

    # Default assessor with loaded v2 detector returns REAL_ZTF_EVALUATED_RESEARCH
    assessment = assessor.assess(representation=rep, domain="real_ztf")
    assert assessment.anomaly_score_status == "REAL_ZTF_EVALUATED_RESEARCH"
    assert assessment.anomaly_score is not None
    assert len(assessment.limitations) >= 4

    # Assessor without loaded v2 detector returns NOT_VALIDATED_FOR_REAL_ZTF
    unvalidated_assessor = ScientificAnomalyAssessor(real_ztf_detector=None)
    # Explicitly clear auto-loaded detector for test
    unvalidated_assessor.real_ztf_detector = None
    unvalidated_assessment = unvalidated_assessor.assess(representation=rep, domain="real_ztf")
    assert unvalidated_assessment.anomaly_score_status == "NOT_VALIDATED_FOR_REAL_ZTF"
    assert unvalidated_assessment.anomaly_score is None


def test_hypothesis_insufficient_evidence_gate():
    """Verify that sparse observations explicitly return INSUFFICIENT_EVIDENCE."""
    service = RealZTFInvestigationService()

    # Create dummy sparse records (< 5 points)
    sparse_records = [
        {"mjd": "58000.0", "mag": "18.0", "magerr": "0.1", "filtercode": "zg", "catflags": "0"},
        {"mjd": "58001.0", "mag": "18.1", "magerr": "0.1", "filtercode": "zg", "catflags": "0"}
    ]

    result = service.investigate_candidate(
        candidate_id="CAND_SPARSE_TEST",
        records=sparse_records,
        metadata={"claimed_type": "Unknown"}
    )

    assert len(result.candidate_hypotheses) == 1
    top_hyp = result.candidate_hypotheses[0]
    assert top_hyp.name == "INSUFFICIENT_EVIDENCE"
    assert top_hyp.confidence_or_status == "INSUFFICIENT_EVIDENCE"


def test_full_investigation_orchestration():
    """Verify end-to-end investigation produces complete InvestigationResult."""
    service = RealZTFInvestigationService()
    result = service.investigate_candidate(DEMO_CANDIDATE)

    assert result.object_id == DEMO_CANDIDATE
    assert result.metadata.get("claimed_type") == "SN Ia"
    assert result.event_characterization.num_observations.value > 1000
    assert result.representation_summary.embedding_dimension == 128
    assert result.anomaly_assessment.anomaly_score_status == "REAL_ZTF_EVALUATED_RESEARCH"
    assert len(result.candidate_hypotheses) >= 1
    assert len(result.evidence) >= 1
    assert len(result.recommended_observations) >= 1
    assert len(result.limitations) >= 3
    assert result.provenance["raw_observation_count"] == 1785
    assert result.execution_time_seconds > 0.0


def test_cli_execution():
    """Verify the investigate_ztf CLI command executes cleanly."""
    cmd = [
        ".venv/bin/python",
        "-m", "src.cli.investigate_ztf",
        "--candidate", DEMO_CANDIDATE,
        "--output-dir", "reports/investigations"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "ACEI SCIENTIFIC INVESTIGATION: CAND_SNIa_002" in res.stdout
    assert "REAL_ZTF_EVALUATED_RESEARCH" in res.stdout
    assert os.path.exists("reports/investigations/CAND_SNIa_002_investigation.json")
    assert os.path.exists("reports/investigations/CAND_SNIa_002_investigation.md")
