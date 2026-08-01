from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from oph_fpe.cosmology.refined_equal_seam_source_gate import (
    DEFAULT_RECEIPT,
    build_refined_equal_seam_source_gate,
    classify_edge_orbits,
)
from oph_fpe.cosmology.verify_refined_equal_seam_source_gate_independent import (
    VerificationError,
    verify_receipt,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_mutation(tmp_path: Path, receipt: dict) -> Path:
    receipt.pop("payload_sha256", None)
    receipt["payload_sha256"] = (
        "sha256:" + hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    )
    path = tmp_path / "mutated_source_gate.json"
    path.write_bytes(_canonical_bytes(receipt))
    return path


@pytest.fixture(scope="module")
def canonical() -> dict:
    return json.loads(DEFAULT_RECEIPT.read_text(encoding="utf-8"))


def test_registered_mesh_a5_edge_orbit_inventory_passes_residual_gate() -> None:
    rows = classify_edge_orbits(5)
    assert [row["edge_count"] for row in rows] == [30, 120, 480, 1920, 7680, 30720]
    assert [row["edge_orbit_count"] for row in rows] == [1, 2, 8, 32, 128, 512]
    assert rows[0]["edge_orbit_size_multiplicities"] == {"30": 1}
    assert [row["edge_orbit_size_multiplicities"] for row in rows[1:]] == [
        {"60": count} for count in (2, 8, 32, 128, 512)
    ]
    assert all(row["edge_incidence_preserved"] for row in rows)
    assert all(row["registered_mesh_permutation_residual_gate_passed"] for row in rows)
    assert all(row["coordinate_residual_gate"] == 5.0e-11 for row in rows)


def test_committed_receipt_passes_independent_replay() -> None:
    result = verify_receipt()
    assert result == {
        "status": "PASS",
        "checked_levels": 6,
        "checked_edges": 40950,
        "observed_orbit_counts": [1, 2, 8, 32, 128, 512],
    }


def test_rebuilt_receipt_matches_constructive_frontier(canonical: dict) -> None:
    rebuilt = build_refined_equal_seam_source_gate()
    assert rebuilt["status"] == canonical["status"]
    assert rebuilt["payload_sha256"] == canonical["payload_sha256"]
    assert rebuilt["classification_finding"] == canonical["classification_finding"]
    assert rebuilt["selection_decision"] == canonical["selection_decision"]


def test_base_selection_and_refined_source_gate_are_not_conflated(
    canonical: dict,
) -> None:
    decision = canonical["selection_decision"]
    assert decision["base_equal_seam_operator_selected_in_bounded_realization"] is True
    assert (
        decision["registered_mesh_a5_edge_orbits_classified_with_residual_gate"] is True
    )
    assert decision["all_level_complete_atomic_counting_law_source_emitted"] is False
    assert decision["refinement_commuting_diagram_discharged"] is False
    assert decision["continuum_equal_seam_operator_selected"] is False
    assert decision["physical_repair_law_selected"] is False
    assert decision["promotion_allowed"] is False


def test_minimal_clause_keeps_the_three_axiom_route_open(canonical: dict) -> None:
    clause = canonical["minimal_constructive_clause"]
    assert clause["may_be_integrated_as_a1_a2_a3_clause_refinement"] is True
    assert clause["additional_branch_or_source_premise_until_derived"] is True
    assert clause["derived_from_canonical_a1_a3_by_this_packet"] is False
    assert clause["fourth_axiom_logically_required"] is False
    assert clause["canonical_basis_amendment_required_before_unconditional_use"] is True
    assert "exactly one event" in clause["a1_atomic_identity"]
    assert "across all edge orbits" in clause["a3_counting_reference"]
    assert "commuting diagram" in clause["refinement_compatibility"]
    assert canonical["classification_finding"]["framework_wide_no_go"] is False


def test_mutated_orbit_inventory_is_rejected(tmp_path: Path, canonical: dict) -> None:
    mutated = copy.deepcopy(canonical)
    mutated["edge_orbit_rows"][2]["edge_orbit_count"] = 7
    with pytest.raises(VerificationError, match="orbit row drift"):
        verify_receipt(_write_mutation(tmp_path, mutated))


def test_unsupported_promotion_is_rejected(tmp_path: Path, canonical: dict) -> None:
    mutated = copy.deepcopy(canonical)
    mutated["selection_decision"][
        "all_level_complete_atomic_counting_law_source_emitted"
    ] = True
    mutated["selection_decision"]["promotion_allowed"] = True
    with pytest.raises(VerificationError, match="source-emitter gate"):
        verify_receipt(_write_mutation(tmp_path, mutated))
