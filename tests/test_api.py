"""Unit tests for the ACEI FastAPI backend."""

import pytest
from starlette.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["checkpoint_sha256"] == "e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72"
    assert data["primary_benchmark_objects"] == 106
    assert data["real_ztf_data_used_in_training"] is False


def test_api_candidates():
    response = client.get("/api/candidates?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    assert len(data["candidates"]) <= 10
    first = data["candidates"][0]
    assert "candidate_id" in first
    assert "ra" in first
    assert "dec" in first
    assert "class" in first


def test_api_candidate_detail():
    response = client.get("/api/candidates/CAND_SNIa_002")
    assert response.status_code == 200
    data = response.json()
    assert data["candidate_id"] == "CAND_SNIa_002"
    assert "metadata" in data


def test_api_investigation():
    response = client.get("/api/investigations/CAND_SNIa_002")
    assert response.status_code == 200
    data = response.json()
    assert data["object_id"] == "CAND_SNIa_002"
    assert data["anomaly_assessment"]["anomaly_score_status"] == "REAL_ZTF_EVALUATED_RESEARCH"
    assert len(data["representation_summary"]["embedding"]) == 128
    assert len(data["candidate_hypotheses"]) >= 1


def test_api_lightcurve():
    response = client.get("/api/lightcurve/CAND_SNIa_002")
    assert response.status_code == 200
    data = response.json()
    assert data["candidate_id"] == "CAND_SNIa_002"
    assert data["observations_count"] > 0
    assert len(data["points"]) > 0
    pt = data["points"][0]
    assert "rel_time" in pt
    assert "norm_flux" in pt
    assert "band" in pt


def test_api_knowledge_search():
    response = client.get("/api/knowledge/search?q=supernova&top_k=3")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] > 0
    assert len(data["results"]) > 0
    res = data["results"][0]
    assert "title" in res
    assert "text" in res


def test_api_reports():
    response = client.get("/api/reports")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 5
    assert any(r["candidate_id"] == "CAND_SNIa_002" for r in data["reports"])


def test_api_markdown_report():
    response = client.get("/api/reports/CAND_SNIa_002/markdown")
    assert response.status_code == 200
    assert "# ACEI Investigation Report: CAND_SNIa_002" in response.text
