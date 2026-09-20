"""Automated tests for Real-ZTF Evaluation Benchmark Freeze Gate.

Verifies:
1. Primary benchmark contains exactly 106 objects.
2. Secondary limited tier contains exactly 31 objects.
3. Zero candidate ID or ZTF designation overlap between primary and secondary tiers.
4. Every primary object satisfies valid_token_count >= 20.
5. Every primary object satisfies padding_fraction <= 0.60.
6. Zero ambiguous associations included in primary or secondary manifests.
7. Zero NO_DATA / failed objects included in primary or secondary manifests.
8. All primary objects have provenance_status == 'VERIFIED' and non-empty citations.
9. Zero duplicate candidate IDs or celestial coordinate collisions (< 1.5'').
10. Zero transfer-pilot objects present in primary or secondary benchmark.
11. Checkpoint SHA-256 matches the frozen production checkpoint bitwise.
12. Benchmark freeze metadata JSON is internally consistent with on-disk file digests.
"""

import os
import json
import hashlib
import pytest
import pandas as pd
from src.data.ztf_object_loader import calculate_angular_separation_arcsec


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZTF_DIR = os.path.join(PROJECT_ROOT, "data", "real_ztf_benchmark")
PRIMARY_MANIFEST = os.path.join(ZTF_DIR, "frozen_primary_benchmark.csv")
SECONDARY_MANIFEST = os.path.join(ZTF_DIR, "frozen_secondary_limited.csv")
FREEZE_METADATA = os.path.join(ZTF_DIR, "benchmark_freeze_metadata.json")
REGISTRY_PATH = os.path.join(ZTF_DIR, "candidate_registry.csv")
RESULTS_PATH = os.path.join(ZTF_DIR, "full_retrieval_results.csv")
ELIGIBILITY_PATH = os.path.join(ZTF_DIR, "benchmark_eligibility.csv")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "checkpoints", "acei_multimodal_production.pt")

EXPECTED_CHECKPOINT_SHA256 = "e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72"
TRANSFER_PILOT_ZTF = {"ZTF19aacgslb", "ZTF20aaynrrh", "ZTF18abukavn", "ZTF18aarkpda", "Field686_Star"}


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def primary_df():
    assert os.path.exists(PRIMARY_MANIFEST), f"Primary manifest missing: {PRIMARY_MANIFEST}"
    return pd.read_csv(PRIMARY_MANIFEST)


@pytest.fixture(scope="module")
def secondary_df():
    assert os.path.exists(SECONDARY_MANIFEST), f"Secondary manifest missing: {SECONDARY_MANIFEST}"
    return pd.read_csv(SECONDARY_MANIFEST)


@pytest.fixture(scope="module")
def freeze_meta():
    assert os.path.exists(FREEZE_METADATA), f"Freeze metadata missing: {FREEZE_METADATA}"
    with open(FREEZE_METADATA, "r", encoding="utf-8") as f:
        return json.load(f)


def test_primary_benchmark_exact_count(primary_df):
    """Verify primary benchmark contains exactly 106 objects."""
    assert len(primary_df) == 106, f"Expected 106 primary objects, got {len(primary_df)}"
    assert (primary_df["coverage_status"] == "COVERAGE_SUFFICIENT").all()


def test_secondary_limited_exact_count(secondary_df):
    """Verify secondary limited tier contains exactly 31 objects."""
    assert len(secondary_df) == 31, f"Expected 31 secondary objects, got {len(secondary_df)}"
    assert (secondary_df["coverage_status"] == "COVERAGE_LIMITED").all()


def test_no_overlap_between_primary_and_secondary(primary_df, secondary_df):
    """Verify strictly disjoint candidate IDs and ZTF designations."""
    primary_cids = set(primary_df["candidate_id"])
    secondary_cids = set(secondary_df["candidate_id"])
    cid_overlap = primary_cids.intersection(secondary_cids)
    assert len(cid_overlap) == 0, f"Candidate ID overlap between tiers: {cid_overlap}"

    primary_ztf = set(primary_df["ztf_designation"])
    secondary_ztf = set(secondary_df["ztf_designation"])
    ztf_overlap = primary_ztf.intersection(secondary_ztf)
    assert len(ztf_overlap) == 0, f"ZTF designation overlap between tiers: {ztf_overlap}"


def test_primary_coverage_quality_criteria(primary_df):
    """Verify all primary objects satisfy valid_tokens >= 20 and padding <= 0.60."""
    assert (primary_df["valid_token_count"] >= 20).all(), (
        f"Found primary objects with valid_token_count < 20: "
        f"{primary_df[primary_df['valid_token_count'] < 20][['candidate_id', 'valid_token_count']]}"
    )
    assert (primary_df["padding_fraction"] <= 0.60).all(), (
        f"Found primary objects with padding_fraction > 0.60: "
        f"{primary_df[primary_df['padding_fraction'] > 0.60][['candidate_id', 'padding_fraction']]}"
    )
    # Check padding fraction arithmetic consistency
    for _, row in primary_df.iterrows():
        expected_padding = round((50 - row["valid_token_count"]) / 50.0, 4)
        assert abs(row["padding_fraction"] - expected_padding) <= 0.02


def test_no_ambiguous_or_no_data_in_benchmarks(primary_df, secondary_df):
    """Verify neither tier contains ambiguous associations or failed/no-data records."""
    for name, df in [("primary", primary_df), ("secondary", secondary_df)]:
        assert (df["retrieval_status"] == "SUCCESS").all(), (
            f"Non-SUCCESS retrieval status found in {name}: {df['retrieval_status'].unique()}"
        )
        assert not df["retrieval_status"].isin(["AMBIGUOUS_ASSOCIATION", "NO_DATA", "RETRIEVAL_FAILED"]).any()


def test_primary_provenance_verification(primary_df):
    """Verify all primary objects possess verified provenance and non-empty authority."""
    assert (primary_df["provenance_status"] == "VERIFIED").all()
    assert primary_df["class_authority"].str.strip().ne("").all()
    assert primary_df["class_reference"].str.strip().ne("").all()
    assert primary_df["provenance_hash"].str.len().eq(64).all()


def test_no_duplicate_candidate_ids_or_coordinates(primary_df, secondary_df):
    """Verify candidate ID uniqueness and celestial coordinate separation within tiers."""
    assert primary_df["candidate_id"].is_unique, "Duplicate candidate_ids in primary manifest"
    assert secondary_df["candidate_id"].is_unique, "Duplicate candidate_ids in secondary manifest"

    # Mutual celestial separation within primary manifest
    for i in range(len(primary_df)):
        for j in range(i + 1, len(primary_df)):
            r1, r2 = primary_df.iloc[i], primary_df.iloc[j]
            sep = calculate_angular_separation_arcsec(r1["ra"], r1["dec"], r2["ra"], r2["dec"])
            assert sep >= 1.5, (
                f"Celestial coordinate collision ({sep:.3f}'' < 1.5'') between "
                f"{r1['candidate_id']} and {r2['candidate_id']}"
            )


def test_no_transfer_pilot_contamination(primary_df, secondary_df):
    """Verify none of the 5 historical transfer-pilot objects enter either benchmark manifest."""
    for name, df in [("primary", primary_df), ("secondary", secondary_df)]:
        overlap = set(df["ztf_designation"]).intersection(TRANSFER_PILOT_ZTF)
        assert len(overlap) == 0, f"Transfer pilot objects contaminated {name} manifest: {overlap}"
        assert not df["candidate_id"].str.contains("PILOT").any()


def test_production_checkpoint_unmodified():
    """Verify bitwise immutability of the production multimodal checkpoint."""
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint missing: {CHECKPOINT_PATH}"
    actual_hash = compute_sha256(CHECKPOINT_PATH)
    assert actual_hash == EXPECTED_CHECKPOINT_SHA256, (
        f"Production checkpoint modified! Found {actual_hash}, expected {EXPECTED_CHECKPOINT_SHA256}"
    )


def test_freeze_metadata_internal_consistency(freeze_meta):
    """Verify metadata file counts, file hashes, and checkpoint hash match reality."""
    assert freeze_meta["candidate_counts"]["total_queried_candidates"] == 180
    assert freeze_meta["candidate_counts"]["primary_benchmark_count"] == 106
    assert freeze_meta["candidate_counts"]["secondary_limited_count"] == 31
    assert freeze_meta["candidate_counts"]["total_excluded_from_primary"] == 74

    # Checkpoint digest match
    assert freeze_meta["production_checkpoint"]["sha256"] == EXPECTED_CHECKPOINT_SHA256

    # Verify hashes of disk files match metadata
    digests = freeze_meta["file_sha256_digests"]
    assert digests["candidate_registry_csv"] == compute_sha256(REGISTRY_PATH)
    assert digests["full_retrieval_results_csv"] == compute_sha256(RESULTS_PATH)
    assert digests["benchmark_eligibility_csv"] == compute_sha256(ELIGIBILITY_PATH)
    assert digests["frozen_primary_benchmark_csv"] == compute_sha256(PRIMARY_MANIFEST)
    assert digests["frozen_secondary_limited_csv"] == compute_sha256(SECONDARY_MANIFEST)
    assert digests["production_checkpoint_pt"] == compute_sha256(CHECKPOINT_PATH)
