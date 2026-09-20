"""Machine-readable manifest schema and serialization for the Real-ZTF Evaluation Benchmark.

Defines the core data contract for multi-band coordinate-associated ZTF objects,
enforcing strict provenance, astronomical classification status, and pilot isolation.

CRITICAL PROTOCOL RULES:
- Never manufacture or guess astrophysical classifications.
- Never alter or fabricate ZTF Object IDs (OIDs).
- Never merge observations across different celestial sources.
- Pilot objects must always be tagged dataset_role = 'transfer_pilot'.
"""

import os
import csv
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union


MANIFEST_COLUMNS = [
    "object_id",
    "ztf_designation",
    "target_ra",
    "target_dec",
    "astrophysical_class",
    "class_authority",
    "class_reference",
    "classification_status",
    "dataset_role",
    "dataset_split",
    "is_anomaly_ground_truth",
    "retrieval_status",
    "matched_oid_zg",
    "matched_oid_zr",
    "matched_oid_zi",
    "max_angular_sep_arcsec",
    "angular_sep_zg_arcsec",
    "angular_sep_zr_arcsec",
    "angular_sep_zi_arcsec",
    "available_filters",
    "missing_filters",
    "partial_filter_coverage",
    "raw_observation_count",
    "clean_observation_count",
    "window_observation_count",
    "valid_token_count",
    "padding_fraction",
    "preprocessing_status",
    "exclusion_reason",
    "provenance_hash"
]

VALID_CLASSIFICATION_STATUSES = {
    "spectroscopic",
    "catalog_photometric",
    "unclassified",
    "control"
}

VALID_DATASET_ROLES = {
    "test_benchmark",
    "val_calibration",
    "control",
    "transfer_pilot"
}

VALID_DATASET_SPLITS = {
    "test_known",
    "test_anomaly",
    "val_calibration",
    "control_unclassified",
    "transfer_pilot"
}

VALID_RETRIEVAL_STATUSES = {
    "SUCCESS",
    "AMBIGUOUS",
    "FAILED",
    "PENDING"
}

VALID_PREPROCESSING_STATUSES = {
    "PREPROCESSED",
    "PENDING",
    "INSUFFICIENT_OBSERVATIONS",
    "EXCLUDED"
}


@dataclass
class RealBenchmarkRecord:
    """Individual object record in the Real-ZTF Benchmark Manifest."""
    object_id: str
    ztf_designation: str
    target_ra: float
    target_dec: float
    astrophysical_class: str
    class_authority: str
    class_reference: str
    classification_status: str
    dataset_role: str
    dataset_split: str
    is_anomaly_ground_truth: int
    retrieval_status: str
    matched_oid_zg: str = ""
    matched_oid_zr: str = ""
    matched_oid_zi: str = ""
    max_angular_sep_arcsec: float = 0.0
    angular_sep_zg_arcsec: Optional[float] = None
    angular_sep_zr_arcsec: Optional[float] = None
    angular_sep_zi_arcsec: Optional[float] = None
    available_filters: str = ""
    missing_filters: str = ""
    partial_filter_coverage: bool = False
    raw_observation_count: int = 0
    clean_observation_count: int = 0
    window_observation_count: int = 0
    valid_token_count: int = 0
    padding_fraction: float = 0.0
    preprocessing_status: str = "PENDING"
    exclusion_reason: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a flat dictionary for CSV serialization."""
        d = asdict(self)
        # Format optional float fields cleanly
        for key in ["angular_sep_zg_arcsec", "angular_sep_zr_arcsec", "angular_sep_zi_arcsec"]:
            if d[key] is None:
                d[key] = ""
            else:
                d[key] = round(float(d[key]), 4)
        d["target_ra"] = round(float(d["target_ra"]), 6)
        d["target_dec"] = round(float(d["target_dec"]), 6)
        d["max_angular_sep_arcsec"] = round(float(d["max_angular_sep_arcsec"]), 4)
        d["padding_fraction"] = round(float(d["padding_fraction"]), 4)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RealBenchmarkRecord":
        """Reconstruct record from a manifest row dictionary."""
        def parse_optional_float(val: Any) -> Optional[float]:
            if val is None or val == "" or str(val).strip().lower() in ["nan", "none", "null"]:
                return None
            return float(val)

        def parse_bool(val: Any) -> bool:
            if isinstance(val, bool):
                return val
            return str(val).strip().lower() in ["true", "1", "yes"]

        return cls(
            object_id=str(d.get("object_id", "")).strip(),
            ztf_designation=str(d.get("ztf_designation", "")).strip(),
            target_ra=float(d.get("target_ra", 0.0)),
            target_dec=float(d.get("target_dec", 0.0)),
            astrophysical_class=str(d.get("astrophysical_class", "")).strip(),
            class_authority=str(d.get("class_authority", "")).strip(),
            class_reference=str(d.get("class_reference", "")).strip(),
            classification_status=str(d.get("classification_status", "")).strip(),
            dataset_role=str(d.get("dataset_role", "")).strip(),
            dataset_split=str(d.get("dataset_split", "")).strip(),
            is_anomaly_ground_truth=int(d.get("is_anomaly_ground_truth", -1)),
            retrieval_status=str(d.get("retrieval_status", "PENDING")).strip(),
            matched_oid_zg=str(d.get("matched_oid_zg", "")).strip(),
            matched_oid_zr=str(d.get("matched_oid_zr", "")).strip(),
            matched_oid_zi=str(d.get("matched_oid_zi", "")).strip(),
            max_angular_sep_arcsec=float(d.get("max_angular_sep_arcsec", 0.0) or 0.0),
            angular_sep_zg_arcsec=parse_optional_float(d.get("angular_sep_zg_arcsec")),
            angular_sep_zr_arcsec=parse_optional_float(d.get("angular_sep_zr_arcsec")),
            angular_sep_zi_arcsec=parse_optional_float(d.get("angular_sep_zi_arcsec")),
            available_filters=str(d.get("available_filters", "")).strip(),
            missing_filters=str(d.get("missing_filters", "")).strip(),
            partial_filter_coverage=parse_bool(d.get("partial_filter_coverage", False)),
            raw_observation_count=int(d.get("raw_observation_count", 0) or 0),
            clean_observation_count=int(d.get("clean_observation_count", 0) or 0),
            window_observation_count=int(d.get("window_observation_count", 0) or 0),
            valid_token_count=int(d.get("valid_token_count", 0) or 0),
            padding_fraction=float(d.get("padding_fraction", 0.0) or 0.0),
            preprocessing_status=str(d.get("preprocessing_status", "PENDING")).strip(),
            exclusion_reason=str(d.get("exclusion_reason", "")).strip(),
            provenance_hash=str(d.get("provenance_hash", "")).strip()
        )


def load_manifest(csv_path: str) -> List[RealBenchmarkRecord]:
    """Load and parse records from a benchmark manifest CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Manifest file does not exist: {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(RealBenchmarkRecord.from_dict(row))
    return records


def save_manifest(records: List[RealBenchmarkRecord], csv_path: str) -> None:
    """Save records to a benchmark manifest CSV file atomically."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())
