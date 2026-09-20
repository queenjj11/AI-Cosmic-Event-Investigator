"""Unit and integrity tests for the Real-ZTF Candidate Discovery Registry.

Verifies:
1. Candidate IDs are 100% unique.
2. Provenance fields (authority, reference) are present for all verified candidates.
3. Rejected candidates retain explicit, non-empty rejection reasons.
4. Classifications adhere to authorized astrophysical taxonomy.
5. Candidate roles are drawn strictly from valid set (in_distribution, ood_anomaly, control, rejected, unverified).
6. Pilot objects cannot enter the candidate registry as benchmark candidates (must be marked 'rejected').
7. Production checkpoint remains bitwise immutable and frozen.
"""

import os
import csv
import hashlib
import pytest

from src.models.checkpoint_manager import DEFAULT_CHECKPOINT_PATH
from src.data.benchmark_validator import RECOGNIZED_ASTROPHYSICAL_CLASSES


EXPECTED_PRODUCTION_CHECKPOINT_SHA256 = "e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72"
REGISTRY_PATH = "data/real_ztf_benchmark/candidate_registry.csv"

VALID_CANDIDATE_ROLES = {
    "in_distribution",
    "ood_anomaly",
    "control",
    "rejected",
    "unverified"
}

PILOT_OBJECT_IDS = {
    "SN_2019np",
    "SN_2020jfo",
    "AT_2018cow",
    "SN_2018zd",
    "ZTF_J195200.60+295217.4"
}


@pytest.fixture
def candidate_rows():
    """Load all rows from candidate_registry.csv."""
    assert os.path.exists(REGISTRY_PATH), f"Candidate registry missing at: {REGISTRY_PATH}"
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_candidate_registry_file_exists_and_populated(candidate_rows):
    """Verify registry is present and contains expected ~190 candidate records."""
    assert len(candidate_rows) >= 180
    assert len(candidate_rows) <= 250


def test_candidate_ids_are_unique(candidate_rows):
    """Verify every candidate ID in the registry is unique."""
    ids = [r["candidate_id"] for r in candidate_rows]
    assert len(ids) == len(set(ids)), f"Found duplicate candidate IDs! Total: {len(ids)}, Unique: {len(set(ids))}"


def test_provenance_completeness_for_verified_candidates(candidate_rows):
    """Verify all verified candidates possess authoritative class authority and reference citations."""
    for r in candidate_rows:
        if r["provenance_status"] == "VERIFIED":
            assert r["class_authority"], f"Candidate {r['candidate_id']} has empty class_authority"
            assert r["class_reference"], f"Candidate {r['candidate_id']} has empty class_reference"
            assert r["discovery_source"], f"Candidate {r['candidate_id']} has empty discovery_source"


def test_rejected_candidates_have_explicit_reasons(candidate_rows):
    """Verify every rejected candidate has an explicit non-empty rejection reason."""
    rejected = [r for r in candidate_rows if r["candidate_role"] == "rejected"]
    assert len(rejected) > 0, "Expected at least some rejected candidates in registry"
    for r in rejected:
        assert r["rejection_reason"], f"Rejected candidate {r['candidate_id']} lacks a rejection_reason"
        assert len(r["rejection_reason"].strip()) >= 10


def test_candidate_classifications_use_allowed_taxonomy(candidate_rows):
    """Verify all candidate astrophysical classes are drawn from recognized taxonomy."""
    for r in candidate_rows:
        cls_name = r["astrophysical_class"].strip()
        matched = any(rec in cls_name for rec in RECOGNIZED_ASTROPHYSICAL_CLASSES)
        assert matched, f"Candidate {r['candidate_id']} assigned unrecognized class: '{cls_name}'"


def test_candidate_roles_are_valid(candidate_rows):
    """Verify candidate roles belong strictly to allowed role enum."""
    for r in candidate_rows:
        role = r["candidate_role"]
        assert role in VALID_CANDIDATE_ROLES, f"Candidate {r['candidate_id']} has invalid role: '{role}'"


def test_pilot_objects_cannot_enter_as_benchmark_candidates(candidate_rows):
    """Verify the 5 exploratory pilot objects are NEVER active benchmark candidates."""
    for r in candidate_rows:
        # Check notes or designation for pilot IDs
        is_pilot = any(p in r["notes"] or p in r["candidate_id"] or p in r["ztf_designation"] for p in PILOT_OBJECT_IDS)
        if is_pilot:
            assert r["candidate_role"] == "rejected", (
                f"Pilot object {r['candidate_id']} leaked into active candidate role: {r['candidate_role']}"
            )
            assert "transfer-pilot" in r["rejection_reason"]


def test_active_candidate_roles_distribution(candidate_rows):
    """Verify presence of all three primary roles (in_distribution, ood_anomaly, control)."""
    roles = {r["candidate_role"] for r in candidate_rows}
    assert "in_distribution" in roles
    assert "ood_anomaly" in roles
    assert "control" in roles
    assert "rejected" in roles

    in_dist = [r for r in candidate_rows if r["candidate_role"] == "in_distribution"]
    ood = [r for r in candidate_rows if r["candidate_role"] == "ood_anomaly"]
    ctrl = [r for r in candidate_rows if r["candidate_role"] == "control"]

    assert len(in_dist) >= 70, f"Expected >= 70 in-distribution candidates, got {len(in_dist)}"
    assert len(ood) >= 30, f"Expected >= 30 OOD anomaly candidates, got {len(ood)}"
    assert len(ctrl) >= 10, f"Expected >= 10 control candidates, got {len(ctrl)}"


def test_production_checkpoint_remains_frozen_during_discovery():
    """Verify production checkpoint file remains bitwise identical to reference hash."""
    assert os.path.exists(DEFAULT_CHECKPOINT_PATH), f"Checkpoint missing at {DEFAULT_CHECKPOINT_PATH}"

    with open(DEFAULT_CHECKPOINT_PATH, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    assert file_hash == EXPECTED_PRODUCTION_CHECKPOINT_SHA256, (
        f"Checkpoint hash mutated! Expected {EXPECTED_PRODUCTION_CHECKPOINT_SHA256}, got {file_hash}"
    )
