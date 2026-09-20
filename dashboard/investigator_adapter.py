"""Dashboard adapter bridging Streamlit UI to the Real-ZTF Investigation pipeline."""

import os
import csv
import json
from typing import Any, Dict, List, Optional
import pandas as pd

from src.investigator.investigation_service import RealZTFInvestigationService
from src.investigator.schemas import InvestigationResult


class RealZTFInvestigatorAdapter:
    """
    Provides a high-level UI adapter for Streamlit dashboards:
    OBJECT -> LIGHT CURVE -> EVENT CHARACTERIZATION -> REPRESENTATION -> ANOMALY STATUS -> HYPOTHESES -> EVIDENCE -> RECOMMENDED OBSERVATIONS
    """

    def __init__(self,
                 investigation_service: Optional[RealZTFInvestigationService] = None,
                 cache_dir: str = "reports/investigations"):
        self.service = investigation_service or RealZTFInvestigationService()
        self.cache_dir = cache_dir

    def get_available_candidates(self) -> List[Dict[str, str]]:
        """Return list of discovered and frozen real-ZTF benchmark candidates."""
        candidates = []
        primary_path = "data/real_ztf_benchmark/frozen_primary_benchmark.csv"
        if os.path.exists(primary_path):
            with open(primary_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    candidates.append({
                        "candidate_id": row.get("candidate_id", ""),
                        "ztf_designation": row.get("ztf_designation", ""),
                        "class": row.get("astrophysical_class", "Unknown"),
                        "ra": row.get("ra", "0.0"),
                        "dec": row.get("dec", "0.0")
                    })
        return candidates

    def get_investigation(self, candidate_id: str, force_recompute: bool = False) -> InvestigationResult:
        """Fetch cached investigation result or compute dynamically via service."""
        cached_json = os.path.join(self.cache_dir, f"{candidate_id}_investigation.json")
        if not force_recompute and os.path.exists(cached_json):
            # Load from cached run if available
            return self.service.investigate_candidate(candidate_id=candidate_id)

        result = self.service.investigate_candidate(candidate_id=candidate_id)
        os.makedirs(self.cache_dir, exist_ok=True)
        result.save_json(cached_json)
        return result

    def get_lightcurve_dataframe(self, candidate_id: str) -> pd.DataFrame:
        """Extract cleaned photometry DataFrame suitable for Streamlit / Plotly plotting."""
        raw_path = f"data/real_ztf_benchmark/raw/{candidate_id}/raw_irsa.csv"
        if not os.path.exists(raw_path):
            raw_path = f"data/real_ztf_pilot/raw/{candidate_id}/raw_irsa.csv"
            if not os.path.exists(raw_path):
                return pd.DataFrame(columns=["mjd", "flux", "flux_err", "band"])

        with open(raw_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)

        pre = self.service.representation_service.preprocessor.process_records(records, object_id=candidate_id)
        feats = pre.feature_tensor.cpu().numpy()
        mask = pre.mask_tensor.cpu().numpy()
        valid = feats[mask]

        if len(valid) == 0:
            return pd.DataFrame(columns=["rel_time", "norm_flux", "norm_flux_err", "band"])

        band_idx_to_str = {0: "g", 1: "r", 2: "i"}
        rows = []
        for v in valid:
            rows.append({
                "rel_time": float(v[0]),
                "norm_flux": float(v[1]),
                "norm_flux_err": float(v[2]),
                "band": band_idx_to_str.get(int(v[3]), "g")
            })

        return pd.DataFrame(rows)

    @staticmethod
    def get_characterization_rows(result: InvestigationResult) -> List[Dict[str, Any]]:
        """Return formatted rows for event characterization table."""
        char_dict = result.event_characterization.to_dict()
        rows = []
        for feat_name, char_val in char_dict.items():
            if isinstance(char_val, dict) and "status" in char_val:
                val = char_val.get("value")
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                rows.append({
                    "Feature": feat_name,
                    "Value": val_str,
                    "Status": char_val.get("status"),
                    "Unit": char_val.get("unit") or "-",
                    "Notes": char_val.get("notes") or ""
                })
        return rows

    @staticmethod
    def get_anomaly_status_payload(result: InvestigationResult) -> Dict[str, Any]:
        """Return anomaly status payload with conservative alerts."""
        anom = result.anomaly_assessment
        return {
            "status": anom.anomaly_score_status,
            "is_validated": False,
            "banner_type": "warning",
            "title": f"Anomaly Assessment: {anom.anomaly_score_status}",
            "message": anom.domain_gap_notes,
            "limitations": anom.limitations
        }
