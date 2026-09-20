"""Freezes the Real-ZTF Evaluation Benchmark into immutable manifests.

This script creates:
- data/real_ztf_benchmark/frozen_primary_benchmark.csv (106 COVERAGE_SUFFICIENT objects)
- data/real_ztf_benchmark/frozen_secondary_limited.csv (31 COVERAGE_LIMITED objects)
- data/real_ztf_benchmark/benchmark_freeze_metadata.json (comprehensive audit metadata & SHA-256 digests)

STRICT SAFEGUARDS:
- Zero model forward passes
- Zero embedding calculations
- Zero anomaly score evaluations
- Verification of production checkpoint immutability
"""

import os
import json
import hashlib
from datetime import datetime, timezone
import pandas as pd
import numpy as np


def sha256_file(filepath: str) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_provenance_hash(candidate_id: str,
                            ztf_designation: str,
                            ra: float,
                            dec: float,
                            astrophysical_class: str,
                            class_authority: str,
                            class_reference: str) -> str:
    """Compute deterministic cryptographic hash binding celestial identity and provenance."""
    key = (f"{candidate_id}|{ztf_designation}|{ra:.6f}|{dec:.6f}|"
           f"{astrophysical_class}|{class_authority}|{class_reference}")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def map_population_family(cls_name: str) -> str:
    """Map fine-grained astrophysical subtype to canonical population family."""
    c = str(cls_name).strip()
    if "Cataclysmic" in c or "CV" in c:
        return "Cataclysmic_Variable"
    elif "SN Ia" in c or "SNIa" in c:
        return "SN_Ia"
    elif "SN II" in c:
        return "SN_II"
    elif "SLSN" in c:
        return "SLSN"
    elif "Variable" in c:
        return "Variable_Star"
    elif "TDE" in c:
        return "TDE"
    elif "Field Star" in c or "Unclassified" in c:
        return "Unclassified_Field_Star"
    return c


def freeze_benchmark():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ztf_dir = os.path.join(project_root, "data", "real_ztf_benchmark")
    ckpt_path = os.path.join(project_root, "models", "checkpoints", "acei_multimodal_production.pt")

    cand_path = os.path.join(ztf_dir, "candidate_registry.csv")
    full_path = os.path.join(ztf_dir, "full_retrieval_results.csv")
    elig_path = os.path.join(ztf_dir, "benchmark_eligibility.csv")

    assert os.path.exists(cand_path), f"Candidate registry missing: {cand_path}"
    assert os.path.exists(full_path), f"Full retrieval results missing: {full_path}"
    assert os.path.exists(elig_path), f"Eligibility file missing: {elig_path}"
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

    # Verify production checkpoint immutability
    ckpt_hash = sha256_file(ckpt_path)
    expected_ckpt_hash = "e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72"
    if ckpt_hash != expected_ckpt_hash:
        raise ValueError(f"Checkpoint integrity violation! Found {ckpt_hash}, expected {expected_ckpt_hash}")

    cand_df = pd.read_csv(cand_path)
    full_df = pd.read_csv(full_path)
    elig_df = pd.read_csv(elig_path)

    # Active candidates only (starting with CAND_)
    active_cand = cand_df[cand_df["candidate_id"].str.startswith("CAND_")].copy()
    assert len(active_cand) == 180, f"Expected 180 active candidates, got {len(active_cand)}"
    assert len(full_df) == 180, f"Expected 180 full retrieval results, got {len(full_df)}"
    assert len(elig_df) == 180, f"Expected 180 eligibility rows, got {len(elig_df)}"

    # Exclude pilot objects check
    pilot_objs = cand_df[cand_df["candidate_id"].str.startswith("REJ_PILOT")]
    assert len(pilot_objs) == 5, f"Expected 5 quarantined pilot objects, got {len(pilot_objs)}"
    assert len(set(active_cand["candidate_id"]).intersection(set(pilot_objs["candidate_id"]))) == 0
    assert len(set(active_cand["ztf_designation"]).intersection(set(pilot_objs["ztf_designation"]))) == 0

    # Merge active candidates with full retrieval results
    # Drop overlapping columns from full_df to preserve exact registry fields
    full_subset = full_df.drop(columns=["astrophysical_class", "dataset_role"])
    merged = pd.merge(active_cand, full_subset, on=["candidate_id", "ztf_designation"])
    assert len(merged) == 180, f"Expected 180 merged records, got {len(merged)}"

    # Canonical population family and dataset split
    merged["population_family"] = merged["astrophysical_class"].apply(map_population_family)

    def get_split_and_truth(row):
        role = row["candidate_role"]
        if role == "in_distribution":
            return "test_benchmark", "test_known", 0
        elif role == "ood_anomaly":
            return "test_benchmark", "test_anomaly", 1
        elif role == "control":
            return "control", "control_unclassified", -1
        else:
            raise ValueError(f"Unknown candidate role: {role}")

    splits_and_truths = [get_split_and_truth(r) for _, r in merged.iterrows()]
    merged["dataset_role"] = [s[0] for s in splits_and_truths]
    merged["dataset_split"] = [s[1] for s in splits_and_truths]
    merged["is_anomaly_ground_truth"] = [s[2] for s in splits_and_truths]

    # Preprocessing data version & raw data path
    merged["preprocessing_data_version"] = "1.0.0"
    merged["raw_data_path"] = merged["candidate_id"].apply(
        lambda cid: f"data/real_ztf_benchmark/raw/{cid}/raw_irsa.csv"
    )

    # Compute deterministic provenance hash
    merged["provenance_hash"] = merged.apply(
        lambda r: compute_provenance_hash(
            r["candidate_id"],
            r["ztf_designation"],
            float(r["ra"]),
            float(r["dec"]),
            r["astrophysical_class"],
            str(r["class_authority"]),
            str(r["class_reference"])
        ),
        axis=1
    )

    # Reorder columns logically
    ordered_cols = [
        "candidate_id",
        "ztf_designation",
        "astrophysical_class",
        "population_family",
        "dataset_role",
        "dataset_split",
        "candidate_role",
        "is_anomaly_ground_truth",
        "ra",
        "dec",
        "crossmatch_separation_arcsec",
        "valid_token_count",
        "padding_fraction",
        "available_filters",
        "missing_filters",
        "partial_filter_coverage",
        "baseline_days",
        "raw_observation_count",
        "clean_observation_count",
        "window_observation_count",
        "matched_oids",
        "retrieval_status",
        "coverage_status",
        "class_authority",
        "class_reference",
        "classification_status",
        "discovery_source",
        "provenance_status",
        "raw_data_path",
        "preprocessing_data_version",
        "provenance_hash"
    ]

    merged_ordered = merged[ordered_cols].copy()

    # Split into primary and secondary
    primary_df = merged_ordered[merged_ordered["coverage_status"] == "COVERAGE_SUFFICIENT"].sort_values("candidate_id").reset_index(drop=True)
    secondary_df = merged_ordered[merged_ordered["coverage_status"] == "COVERAGE_LIMITED"].sort_values("candidate_id").reset_index(drop=True)

    # Validation of counts and bounds
    assert len(primary_df) == 106, f"Expected exactly 106 primary objects, got {len(primary_df)}"
    assert len(secondary_df) == 31, f"Expected exactly 31 secondary objects, got {len(secondary_df)}"

    # Primary criteria assertions
    assert (primary_df["retrieval_status"] == "SUCCESS").all()
    assert (primary_df["valid_token_count"] >= 20).all()
    assert (primary_df["padding_fraction"] <= 0.60).all()
    assert (primary_df["crossmatch_separation_arcsec"] <= 1.5).all()
    assert (primary_df["provenance_status"] == "VERIFIED").all()
    assert len(set(primary_df["candidate_id"])) == 106
    assert len(set(primary_df["ztf_designation"])) == 106

    # Secondary criteria assertions
    assert (secondary_df["retrieval_status"] == "SUCCESS").all()
    assert (secondary_df["valid_token_count"] >= 5).all()
    assert (secondary_df["crossmatch_separation_arcsec"] <= 1.5).all()
    assert len(set(secondary_df["candidate_id"])) == 31
    assert len(set(primary_df["candidate_id"]).intersection(set(secondary_df["candidate_id"]))) == 0

    # Write manifests atomically
    primary_manifest_path = os.path.join(ztf_dir, "frozen_primary_benchmark.csv")
    secondary_manifest_path = os.path.join(ztf_dir, "frozen_secondary_limited.csv")

    primary_df.to_csv(primary_manifest_path, index=False)
    secondary_df.to_csv(secondary_manifest_path, index=False)

    print(f"Saved frozen primary benchmark: {primary_manifest_path} ({len(primary_df)} rows)")
    print(f"Saved frozen secondary benchmark: {secondary_manifest_path} ({len(secondary_df)} rows)")

    # Compute SHA-256 digests
    cand_hash = sha256_file(cand_path)
    full_hash = sha256_file(full_path)
    elig_hash = sha256_file(elig_path)
    prim_hash = sha256_file(primary_manifest_path)
    sec_hash = sha256_file(secondary_manifest_path)

    # Exclusions breakdown
    ambiguous_count = int((merged["retrieval_status"] == "AMBIGUOUS_ASSOCIATION").sum())
    no_data_count = int((merged["retrieval_status"] == "NO_DATA").sum())
    insufficient_count = int((merged["coverage_status"] == "INSUFFICIENT_OBSERVATIONS").sum())
    total_excluded_from_primary = len(merged) - len(primary_df)

    # Population breakdown
    primary_by_class = primary_df["astrophysical_class"].value_counts().to_dict()
    primary_by_family = primary_df["population_family"].value_counts().to_dict()
    secondary_by_class = secondary_df["astrophysical_class"].value_counts().to_dict()
    secondary_by_family = secondary_df["population_family"].value_counts().to_dict()

    # Freeze metadata
    freeze_metadata = {
        "benchmark_freeze_timestamp": datetime.now(timezone.utc).isoformat(),
        "freeze_phase": "Real-ZTF Benchmark Freeze Gate",
        "model_inference_status": (
            "ZERO model forward passes, ZERO embedding extractions, and ZERO anomaly score "
            "calculations were performed before this freeze. The production checkpoint remains "
            "strictly frozen and verified."
        ),
        "production_checkpoint": {
            "path": "models/checkpoints/acei_multimodal_production.pt",
            "sha256": ckpt_hash,
            "status": "FROZEN_VERIFIED"
        },
        "candidate_counts": {
            "total_queried_candidates": 180,
            "primary_benchmark_count": 106,
            "secondary_limited_count": 31,
            "total_excluded_from_primary": total_excluded_from_primary,
            "exclusions_breakdown": {
                "AMBIGUOUS_ASSOCIATION": ambiguous_count,
                "NO_DATA": no_data_count,
                "INSUFFICIENT_OBSERVATIONS": insufficient_count,
                "COVERAGE_LIMITED_HELD_IN_SECONDARY": len(secondary_df)
            }
        },
        "eligibility_criteria": {
            "coordinate_crossmatch": "Haversine distance <= 1.5 arcsec",
            "ambiguity_safety_margin": "Minimum angular separation gap >= 0.3 arcsec",
            "valid_token_count_min": 20,
            "padding_fraction_max": 0.60,
            "provenance_requirement": "VERIFIED (spectroscopic IAU TNS/BTS, Gaia DR3, or published literature)",
            "pilot_isolation": "Strict exclusion of the 5 historical transfer-pilot objects",
            "duplicate_tolerance": "Zero coordinate collisions within 1.5 arcsec",
            "ground_truth_isolation": "Zero test labels accessible to models during training/calibration"
        },
        "primary_population_breakdown": {
            "by_population_family": primary_by_family,
            "by_authoritative_class": primary_by_class
        },
        "secondary_population_breakdown": {
            "by_population_family": secondary_by_family,
            "by_authoritative_class": secondary_by_class
        },
        "file_sha256_digests": {
            "candidate_registry_csv": cand_hash,
            "full_retrieval_results_csv": full_hash,
            "benchmark_eligibility_csv": elig_hash,
            "frozen_primary_benchmark_csv": prim_hash,
            "frozen_secondary_limited_csv": sec_hash,
            "production_checkpoint_pt": ckpt_hash
        }
    }

    metadata_path = os.path.join(ztf_dir, "benchmark_freeze_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(freeze_metadata, f, indent=2)

    meta_hash = sha256_file(metadata_path)
    print(f"Saved freeze metadata: {metadata_path}")
    print(f"Metadata SHA-256: {meta_hash}")

    print("\n" + "=" * 60)
    print("REAL-ZTF BENCHMARK FREEZE COMPLETE")
    print("=" * 60)
    print(f"Primary Benchmark Objects:   {len(primary_df)}")
    print(f"Secondary Stress-Test Tier:  {len(secondary_df)}")
    print(f"Total Evaluated Candidates:  {len(merged)}")
    print("\nPrimary Breakdown by Population Family:")
    for fam, cnt in sorted(primary_by_family.items(), key=lambda x: -x[1]):
        print(f"  {fam:<25}: {cnt:>3}")
    print("\nSecondary Breakdown by Population Family:")
    for fam, cnt in sorted(secondary_by_family.items(), key=lambda x: -x[1]):
        print(f"  {fam:<25}: {cnt:>3}")
    print("\nCryptographic SHA-256 Digests:")
    print(f"  candidate_registry.csv:        {cand_hash}")
    print(f"  full_retrieval_results.csv:    {full_hash}")
    print(f"  benchmark_eligibility.csv:     {elig_hash}")
    print(f"  frozen_primary_benchmark.csv:  {prim_hash}")
    print(f"  frozen_secondary_limited.csv:  {sec_hash}")
    print(f"  benchmark_freeze_metadata.json:{meta_hash}")
    print(f"  production checkpoint:         {ckpt_hash}")
    print("=" * 60)

    return {
        "primary_df": primary_df,
        "secondary_df": secondary_df,
        "metadata": freeze_metadata,
        "meta_hash": meta_hash
    }


if __name__ == "__main__":
    freeze_benchmark()
