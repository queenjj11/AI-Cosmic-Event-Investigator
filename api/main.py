"""FastAPI backend application serving the ACEI Real-ZTF Investigation System."""

import os
import csv
import json
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from src.investigator.investigation_service import RealZTFInvestigationService
from dashboard.investigator_adapter import RealZTFInvestigatorAdapter


app = FastAPI(
    title="AI Cosmic Event Investigator (ACEI) API",
    description="Scientific API providing real-ZTF astronomical investigation, light curves, representations, and knowledge retrieval.",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singleton service and adapter instances
_service: Optional[RealZTFInvestigationService] = None
_adapter: Optional[RealZTFInvestigatorAdapter] = None


def get_service() -> RealZTFInvestigationService:
    global _service
    if _service is None:
        _service = RealZTFInvestigationService()
    return _service


def get_adapter() -> RealZTFInvestigatorAdapter:
    global _adapter
    if _adapter is None:
        _adapter = RealZTFInvestigatorAdapter(investigation_service=get_service())
    return _adapter


@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    """System health status and frozen checkpoint provenance."""
    return {
        "status": "healthy",
        "service": "ACEI Real-ZTF Investigation Engine",
        "model": "Production LightCurveEncoder (128-D Transformer)",
        "device": "cpu",
        "checkpoint_sha256": "e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72",
        "frozen_benchmark_sha256": "0e839c8f8ff1b83969e3f959a2be3342a96ac2564d00e73b045a21445cd4501c",
        "primary_benchmark_objects": 106,
        "real_ztf_data_used_in_training": False,
        "anomaly_status_default": "REAL_ZTF_EVALUATED_RESEARCH",
        "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }


@app.get("/api/candidates")
def get_candidates(
    classification: Optional[str] = None,
    limit: int = Query(default=120, le=200)
) -> Dict[str, Any]:
    """List all frozen primary benchmark candidates with observational metrics."""
    candidates = []
    primary_path = "data/real_ztf_benchmark/frozen_primary_benchmark.csv"
    if not os.path.exists(primary_path):
        raise HTTPException(status_code=404, detail="Primary benchmark file not found")

    with open(primary_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_name = row.get("astrophysical_class", "Unknown")
            if classification and classification.lower() not in cls_name.lower():
                continue

            cid = row.get("candidate_id", "")
            json_report_path = f"reports/investigations/{cid}_investigation.json"

            candidates.append({
                "candidate_id": cid,
                "ztf_designation": row.get("ztf_designation", ""),
                "class": cls_name,
                "dataset_role": row.get("dataset_role", ""),
                "ra": float(row.get("ra", 0.0)),
                "dec": float(row.get("dec", 0.0)),
                "raw_observations": int(row.get("raw_observation_count", 0)),
                "clean_observations": int(row.get("clean_observation_count", 0)),
                "valid_tokens": int(row.get("valid_token_count", 0)),
                "baseline_days": float(row.get("baseline_days", 0.0)),
                "available_filters": [f.strip() for f in row.get("available_filters", "").split(",") if f.strip()],
                "has_investigation_report": os.path.exists(json_report_path)
            })

    return {
        "total_count": len(candidates),
        "candidates": candidates[:limit]
    }


@app.get("/api/candidates/{candidate_id}")
def get_candidate_detail(candidate_id: str) -> Dict[str, Any]:
    """Retrieve metadata and raw observation audit metrics for a candidate."""
    benchmark_paths = [
        "data/real_ztf_benchmark/frozen_primary_benchmark.csv",
        "data/real_ztf_benchmark/frozen_secondary_limited.csv",
        "data/real_ztf_benchmark/candidate_registry.csv"
    ]
    for path in benchmark_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("candidate_id") == candidate_id:
                        return {
                            "candidate_id": candidate_id,
                            "metadata": row,
                            "raw_csv_exists": os.path.exists(row.get("raw_data_path", f"data/real_ztf_benchmark/raw/{candidate_id}/raw_irsa.csv")),
                            "has_cached_investigation": os.path.exists(f"reports/investigations/{candidate_id}_investigation.json")
                        }

    raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found in benchmark registries")


@app.post("/api/investigate/{candidate_id}")
def run_investigation(candidate_id: str) -> Dict[str, Any]:
    """Execute live end-to-end investigation on a candidate."""
    service = get_service()
    try:
        result = service.investigate_candidate(candidate_id=candidate_id)
        # Save cache
        os.makedirs("reports/investigations", exist_ok=True)
        result.save_json(f"reports/investigations/{candidate_id}_investigation.json")
        result.save_markdown(f"reports/investigations/{candidate_id}_investigation.md")
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investigation failed for {candidate_id}: {str(e)}")


@app.get("/api/investigations/{candidate_id}")
def get_investigation(candidate_id: str, force: bool = False) -> Dict[str, Any]:
    """Fetch cached investigation result or execute on the fly."""
    cache_path = f"reports/investigations/{candidate_id}_investigation.json"
    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return run_investigation(candidate_id)


@app.get("/api/lightcurve/{candidate_id}")
def get_lightcurve_data(candidate_id: str) -> Dict[str, Any]:
    """Return cleaned time-series points formatted for interactive plotting."""
    adapter = get_adapter()
    df = adapter.get_lightcurve_dataframe(candidate_id)
    if df.empty:
        return {
            "candidate_id": candidate_id,
            "observations_count": 0,
            "points": []
        }

    points = df.to_dict(orient="records")
    return {
        "candidate_id": candidate_id,
        "observations_count": len(points),
        "points": points
    }


@app.get("/api/knowledge/search")
def search_knowledge(q: str = Query(..., min_length=2), top_k: int = 5) -> Dict[str, Any]:
    """Semantic literature retrieval over vector store."""
    service = get_service()
    results = service.vector_store.query(query_text=q, top_k=top_k)
    items = []
    for chunk, score in results:
        items.append({
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "score": round(float(score), 4),
            "text": chunk.text,
            "source_type": chunk.source_type,
            "filepath": chunk.filepath
        })
    return {
        "query": q,
        "count": len(items),
        "results": items
    }


@app.get("/api/knowledge/documents")
def get_knowledge_documents() -> Dict[str, Any]:
    """List loaded scientific papers and catalogs in the in-memory knowledge base."""
    service = get_service()
    docs = {}
    for c in service.vector_store.chunks:
        if c.doc_id not in docs:
            docs[c.doc_id] = {
                "doc_id": c.doc_id,
                "title": c.title,
                "source_type": c.source_type,
                "filepath": c.filepath,
                "chunks_count": 0
            }
        docs[c.doc_id]["chunks_count"] += 1

    return {
        "total_documents": len(docs),
        "total_chunks": len(service.vector_store.chunks),
        "documents": list(docs.values())
    }


@app.get("/api/reports")
def list_reports() -> Dict[str, Any]:
    """List all available generated scientific investigation reports."""
    reports_dir = "reports/investigations"
    if not os.path.exists(reports_dir):
        return {"reports": []}

    reports = []
    for filename in sorted(os.listdir(reports_dir)):
        if filename.endswith("_investigation.json"):
            cid = filename.replace("_investigation.json", "")
            filepath = os.path.join(reports_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                reports.append({
                    "candidate_id": cid,
                    "claimed_type": data.get("metadata", {}).get("claimed_type", "Unknown"),
                    "execution_time": data.get("execution_time_seconds"),
                    "top_hypothesis": data.get("candidate_hypotheses", [{}])[0].get("name", "None"),
                    "top_probability": data.get("candidate_hypotheses", [{}])[0].get("probability"),
                    "anomaly_status": data.get("anomaly_assessment", {}).get("anomaly_score_status"),
                    "has_markdown": os.path.exists(os.path.join(reports_dir, f"{cid}_investigation.md")),
                    "has_json": True
                })
            except Exception:
                continue

    return {"count": len(reports), "reports": reports}


@app.get("/api/reports/{candidate_id}/markdown", response_class=PlainTextResponse)
def get_markdown_report(candidate_id: str) -> str:
    """Download or view human-readable Markdown investigation report."""
    md_path = f"reports/investigations/{candidate_id}_investigation.md"
    if not os.path.exists(md_path):
        # Generate on the fly
        run_investigation(candidate_id)

    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()

    raise HTTPException(status_code=404, detail="Markdown report not found")


@app.get("/api/reports/{candidate_id}/json")
def get_json_report(candidate_id: str) -> Dict[str, Any]:
    """Download machine-readable JSON investigation report."""
    return get_investigation(candidate_id)
