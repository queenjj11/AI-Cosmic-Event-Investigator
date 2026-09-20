"""Automated integrity validation utility for the Real-ZTF Evaluation Benchmark.

Performs 12 rigorous scientific integrity, provenance, and split-leakage checks.
Fails loudly on critical violations to ensure unverified data never enters evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import math
import numpy as np

from src.data.real_ztf_manifest import (
    RealBenchmarkRecord,
    MANIFEST_COLUMNS,
    VALID_CLASSIFICATION_STATUSES,
    VALID_DATASET_ROLES,
    VALID_DATASET_SPLITS,
    VALID_RETRIEVAL_STATUSES,
    VALID_PREPROCESSING_STATUSES
)
from src.data.ztf_object_loader import calculate_angular_separation_arcsec


# Authoritative recognized astrophysical classes for taxonomy verification
RECOGNIZED_ASTROPHYSICAL_CLASSES = {
    # In-Distribution
    "SN_Ia", "SN Ia", "SNIa", "Type Ia Supernova",
    "SN_II", "SN II", "SN IIP", "SN II-P", "SN IIL", "SN IIb", "Type II Supernova",
    "Stellar_Flare", "Stellar Flare", "Flare Star", "UV Ceti",
    "Variable_Star", "Variable Star", "RR Lyrae", "Cepheid", "Mira", "Delta Scuti", "Eclipsing Binary",
    # OOD Anomaly
    "SLSN", "SLSN-I", "SLSN-II", "Superluminous Supernova",
    "TDE", "Tidal Disruption Event",
    "FBOT", "LFBOT", "Fast Blue Optical Transient", "AT2018cow-like",
    "LRN", "Luminous Red Nova", "ILRT", "Intermediate Luminosity Red Transient",
    "SN_Ibn", "SN Ibn", "SN_Icn", "SN Icn",
    "Cataclysmic_Variable", "Cataclysmic Variable", "Dwarf Nova", "AM CVn",
    # Control / Unclassified
    "Unclassified", "Unclassified Field Star", "Field Star", "Control", "Sparse Control"
}


class BenchmarkIntegrityError(Exception):
    """Raised when a benchmark manifest fails one or more critical scientific integrity checks."""
    pass


@dataclass
class ValidationReport:
    """Comprehensive diagnostic report on benchmark integrity, demographics, and coverage."""
    is_valid: bool
    total_objects: int = 0
    objects_by_class: Dict[str, int] = field(default_factory=dict)
    objects_by_split: Dict[str, int] = field(default_factory=dict)
    objects_by_role: Dict[str, int] = field(default_factory=dict)
    filter_coverage: Dict[str, int] = field(default_factory=dict)
    missing_band_frequency: Dict[str, int] = field(default_factory=dict)
    retrieval_status_counts: Dict[str, int] = field(default_factory=dict)
    preprocessing_status_counts: Dict[str, int] = field(default_factory=dict)
    observation_count_stats: Dict[str, float] = field(default_factory=dict)
    valid_token_stats: Dict[str, float] = field(default_factory=dict)
    padding_distribution: Dict[str, int] = field(default_factory=dict)
    coordinate_match_stats: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Format human-readable audit summary."""
        lines = [
            "=" * 70,
            "REAL-ZTF BENCHMARK VALIDATION AUDIT REPORT",
            "=" * 70,
            f"Overall Status:        {'PASSED' if self.is_valid else 'FAILED'}",
            f"Total Objects:         {self.total_objects}",
            f"Critical Errors:       {len(self.errors)}",
            f"Warnings:              {len(self.warnings)}",
            "",
            "--- OBJECTS BY SPLIT ---"
        ]
        for split, count in sorted(self.objects_by_split.items()):
            lines.append(f"  {split:<22}: {count:>4}")

        lines.append("\n--- OBJECTS BY CLASS ---")
        for cls, count in sorted(self.objects_by_class.items()):
            lines.append(f"  {cls:<25}: {count:>4}")

        lines.append("\n--- FILTER COVERAGE ---")
        for ftr, count in sorted(self.filter_coverage.items()):
            lines.append(f"  {ftr:<22}: {count:>4}")

        lines.append("\n--- PADDING DISTRIBUTION ---")
        for pad_bin, count in sorted(self.padding_distribution.items()):
            lines.append(f"  {pad_bin:<22}: {count:>4}")

        if self.errors:
            lines.append("\n--- CRITICAL ERRORS ---")
            for e in self.errors:
                lines.append(f"  [ERROR] {e}")

        if self.warnings:
            lines.append("\n--- WARNINGS ---")
            for w in self.warnings:
                lines.append(f"  [WARN]  {w}")

        lines.append("=" * 70)
        return "\n".join(lines)


class BenchmarkValidator:
    """Validates real ZTF benchmark manifests against strict astronomical standards."""

    def __init__(self,
                 max_angular_separation_arcsec: float = 1.5,
                 duplicate_coordinate_tolerance_arcsec: float = 1.5,
                 strict_mode: bool = True):
        self.max_angular_sep = max_angular_separation_arcsec
        self.dup_tolerance = duplicate_coordinate_tolerance_arcsec
        self.strict_mode = strict_mode

    def validate(self, records: List[RealBenchmarkRecord]) -> ValidationReport:
        """Execute all 12 validation checks across the given records."""
        report = ValidationReport(is_valid=True, total_objects=len(records))

        if not records:
            report.warnings.append("Manifest contains 0 records (empty template).")
            return report

        # 1. Duplicate Object ID Check
        seen_ids: Set[str] = set()
        for r in records:
            if not r.object_id:
                report.errors.append("Encountered record with empty object_id.")
            elif r.object_id in seen_ids:
                report.errors.append(f"Duplicate object_id detected: '{r.object_id}'.")
            seen_ids.add(r.object_id)

        # 2. Duplicate Coordinate Collision Check
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                r1, r2 = records[i], records[j]
                sep = calculate_angular_separation_arcsec(r1.target_ra, r1.target_dec,
                                                          r2.target_ra, r2.target_dec)
                if sep < self.dup_tolerance:
                    report.errors.append(
                        f"Duplicate celestial coordinate collision ({sep:.3f}'' < {self.dup_tolerance}'') "
                        f"between '{r1.object_id}' and '{r2.object_id}'."
                    )

        # 3. Coordinate Association Distance Check
        for r in records:
            if r.retrieval_status == "SUCCESS":
                if r.max_angular_sep_arcsec > self.max_angular_sep:
                    report.errors.append(
                        f"Object '{r.object_id}' exceeds maximum association separation: "
                        f"{r.max_angular_sep_arcsec:.3f}'' > {self.max_angular_sep}''."
                    )
                for band, sep in [("zg", r.angular_sep_zg_arcsec),
                                  ("zr", r.angular_sep_zr_arcsec),
                                  ("zi", r.angular_sep_zi_arcsec)]:
                    if sep is not None and sep > self.max_angular_sep:
                        report.errors.append(
                            f"Object '{r.object_id}' band {band} separation exceeds tolerance: "
                            f"{sep:.3f}'' > {self.max_angular_sep}''."
                        )

        # 4. Provenance Completeness Check
        for r in records:
            if r.classification_status in ["spectroscopic", "catalog_photometric"]:
                if not r.class_authority or not r.class_reference:
                    report.errors.append(
                        f"Classified object '{r.object_id}' ({r.astrophysical_class}) lacks authoritative "
                        f"provenance citation (authority='{r.class_authority}', reference='{r.class_reference}')."
                    )

        # 5. Split Exclusivity & Object-Level Leakage Check
        split_map: Dict[str, Set[str]] = {}
        for r in records:
            split_map.setdefault(r.dataset_split, set())
            split_map[r.dataset_split].add(r.object_id)

        all_splits = list(split_map.keys())
        for i in range(len(all_splits)):
            for j in range(i + 1, len(all_splits)):
                s1, s2 = all_splits[i], all_splits[j]
                intersection = split_map[s1].intersection(split_map[s2])
                if intersection:
                    report.errors.append(
                        f"Object-level split leakage detected: Objects {sorted(list(intersection))} "
                        f"appear in both '{s1}' and '{s2}'."
                    )

        # 6. Pilot Isolation Check
        for r in records:
            if r.dataset_role == "transfer_pilot":
                if r.dataset_split != "transfer_pilot":
                    report.errors.append(
                        f"Pilot isolation violation: '{r.object_id}' has role 'transfer_pilot' "
                        f"but is assigned to evaluation split '{r.dataset_split}'."
                    )
            elif r.dataset_split == "transfer_pilot":
                if r.dataset_role != "transfer_pilot":
                    report.errors.append(
                        f"Pilot isolation violation: '{r.object_id}' has split 'transfer_pilot' "
                        f"but is assigned role '{r.dataset_role}'."
                    )

        # 7. Unclassified Object Ground-Truth Integrity Check
        for r in records:
            if r.classification_status == "unclassified" or r.astrophysical_class.lower() in ["unclassified", "unknown"]:
                if r.is_anomaly_ground_truth not in [-1, None]:
                    report.errors.append(
                        f"Unclassified object '{r.object_id}' must have is_anomaly_ground_truth = -1, "
                        f"got {r.is_anomaly_ground_truth}."
                    )

        # 8. Recognized Taxonomy Check
        for r in records:
            cls_name = r.astrophysical_class.strip()
            if cls_name and cls_name not in RECOGNIZED_ASTROPHYSICAL_CLASSES:
                # Check for partial matches or prefixes
                matched = any(rec in cls_name for rec in RECOGNIZED_ASTROPHYSICAL_CLASSES)
                if not matched:
                    report.errors.append(
                        f"Object '{r.object_id}' assigned unrecognized astrophysical class '{cls_name}'."
                    )

        # 9. Preprocessing Token Count Consistency
        for r in records:
            if r.preprocessing_status == "PREPROCESSED":
                if r.valid_token_count < 0 or r.valid_token_count > 50:
                    report.errors.append(
                        f"Object '{r.object_id}' has invalid valid_token_count: {r.valid_token_count} (must be in [0, 50])."
                    )
                expected_padding = round(float(50 - r.valid_token_count) / 50.0, 4)
                if abs(r.padding_fraction - expected_padding) > 0.02:
                    report.errors.append(
                        f"Object '{r.object_id}' has inconsistent padding fraction: "
                        f"got {r.padding_fraction}, expected {expected_padding}."
                    )
                if r.clean_observation_count > r.raw_observation_count:
                    report.errors.append(
                        f"Object '{r.object_id}' has clean obs count ({r.clean_observation_count}) "
                        f"> raw obs count ({r.raw_observation_count})."
                    )

        # 10. Filter & OID Structural Consistency Check
        for r in records:
            if r.retrieval_status == "SUCCESS":
                oids_present = []
                if r.matched_oid_zg: oids_present.append("zg")
                if r.matched_oid_zr: oids_present.append("zr")
                if r.matched_oid_zi: oids_present.append("zi")
                if not oids_present:
                    report.errors.append(f"Object '{r.object_id}' is marked SUCCESS but has 0 matched filter OIDs.")

        # 11. Preserved Exclusions Check
        for r in records:
            if r.retrieval_status in ["FAILED", "AMBIGUOUS"] or r.preprocessing_status == "EXCLUDED":
                if not r.exclusion_reason:
                    report.errors.append(
                        f"Object '{r.object_id}' is marked {r.retrieval_status}/{r.preprocessing_status} "
                        f"but lacks an explicit exclusion_reason."
                    )

        # 12. Demographics & Distribution Aggregation
        raw_obs_list = []
        token_list = []
        match_seps = []
        pad_bins = {"0%": 0, "1-30%": 0, "31-60%": 0, "61-80%": 0, "81-100%": 0}

        for r in records:
            # Class distribution
            report.objects_by_class[r.astrophysical_class] = report.objects_by_class.get(r.astrophysical_class, 0) + 1
            # Split distribution
            report.objects_by_split[r.dataset_split] = report.objects_by_split.get(r.dataset_split, 0) + 1
            # Role distribution
            report.objects_by_role[r.dataset_role] = report.objects_by_role.get(r.dataset_role, 0) + 1
            # Filter coverage
            ftr_key = r.available_filters or "none"
            report.filter_coverage[ftr_key] = report.filter_coverage.get(ftr_key, 0) + 1
            # Missing band frequency
            for m in (r.missing_filters.split("+") if r.missing_filters and r.missing_filters != "none" else []):
                report.missing_band_frequency[m] = report.missing_band_frequency.get(m, 0) + 1
            # Retrieval & Preprocessing
            report.retrieval_status_counts[r.retrieval_status] = report.retrieval_status_counts.get(r.retrieval_status, 0) + 1
            report.preprocessing_status_counts[r.preprocessing_status] = report.preprocessing_status_counts.get(r.preprocessing_status, 0) + 1

            raw_obs_list.append(r.raw_observation_count)
            token_list.append(r.valid_token_count)
            if r.retrieval_status == "SUCCESS":
                match_seps.append(r.max_angular_sep_arcsec)

            # Padding bins
            p = r.padding_fraction
            if p == 0.0:
                pad_bins["0%"] += 1
            elif p <= 0.30:
                pad_bins["1-30%"] += 1
            elif p <= 0.60:
                pad_bins["31-60%"] += 1
            elif p <= 0.80:
                pad_bins["61-80%"] += 1
            else:
                pad_bins["81-100%"] += 1

        report.padding_distribution = pad_bins

        if raw_obs_list:
            report.observation_count_stats = {
                "mean": float(np.mean(raw_obs_list)),
                "median": float(np.median(raw_obs_list)),
                "min": float(np.min(raw_obs_list)),
                "max": float(np.max(raw_obs_list))
            }
        if token_list:
            report.valid_token_stats = {
                "mean": float(np.mean(token_list)),
                "median": float(np.median(token_list)),
                "min": float(np.min(token_list)),
                "max": float(np.max(token_list))
            }
        if match_seps:
            report.coordinate_match_stats = {
                "mean_arcsec": float(np.mean(match_seps)),
                "max_arcsec": float(np.max(match_seps)),
                "min_arcsec": float(np.min(match_seps))
            }

        if report.errors:
            report.is_valid = False
            if self.strict_mode:
                err_msg = "\n".join(report.errors)
                raise BenchmarkIntegrityError(f"Benchmark validation failed with {len(report.errors)} errors:\n{err_msg}")

        return report
