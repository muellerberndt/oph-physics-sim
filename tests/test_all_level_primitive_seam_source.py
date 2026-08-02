from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oph_fpe.cosmology import all_level_primitive_seam_source as producer
from oph_fpe.cosmology import verify_all_level_primitive_seam_source_independent as independent


REPORT = producer.build_receipt()


def _rehash(report: dict) -> None:
    payload = copy.deepcopy(report)
    payload.pop("payload_sha256", None)
    report["payload_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def test_complete_registered_alphabet_and_declared_unit_counting() -> None:
    assert REPORT["schema"] == producer.SCHEMA
    assert REPORT["status"] == producer.STATUS
    rows = REPORT["level_alphabets"]
    assert [row["level"] for row in rows] == list(range(6))
    assert [row["seam_count"] for row in rows] == [30, 120, 480, 1920, 7680, 30720]
    assert sum(row["seam_count"] for row in rows) == 40950
    assert all(row["one_event_per_registered_seam"] is True for row in rows)
    assert all(
        row["declared_multiplicity_by_a5_orbit"]
        == [1] * row["a5_edge_orbit_count_from_residual_gated_parent"]
        for row in rows
    )
    counting = REPORT["unit_counting_certificate"]
    assert counting["exact_unit_counting_across_a5_orbit_classes_in_declared_source"] is True
    assert counting["unit_counting_forced_by_a5_alone"] is False
    assert counting["unit_counting_derived_from_canonical_a1_a3"] is False
    assert counting["unit_counting_is_additional_source_branch_declaration"] is True


def test_parent_child_lineage_and_refinement_boundaries_are_exact() -> None:
    rows = REPORT["refinement_certificate"]["rows"]
    assert len(rows) == 5
    for row in rows:
        assert row["fine_seam_count"] == 4 * row["coarse_seam_count"]
        assert len(row["fine_parent_seam_indices"]) == row["fine_seam_count"]
        assert row["children_per_parent_seam"] == 4
        assert row["boundary_half_children_per_parent"] == 2
        assert row["parent_face_interior_children_per_parent"] == 2
        assert row["normalized_unit_count_pushforward_exact"] is True
        assert row["child_aggregate_identity_exact_for_every_parent"] is True
        assert row["level_mean_generator_identity_exact"] is True
        assert row["strong_intertwiner"] is False
        assert row["strong_intertwiner_witness_on_coarse_basis_zero"]
        assert row["second_order_inherited_identity"] is False
        assert row["second_order_witness_on_coarse_basis_zero"]
    summary = REPORT["refinement_certificate"]
    assert summary["complete_event_lineage_exact"] is True
    assert summary["normalized_unit_counting_refinement_natural"] is True
    assert summary["raw_unit_counting_refinement_natural_without_rescaling"] is False
    assert summary["repair_semigroup_refinement_natural"] is False


def test_a2_meaning_and_atomic_record_boundaries_stay_separate() -> None:
    a2 = REPORT["a2_reconciliation"]
    assert a2["endpoint_total_preserved_pathwise"] is True
    assert a2["nearest_balanced_shell_reached_pathwise"] is True
    assert a2["expected_endpoint_agreement_exact"] is True
    assert a2["odd_total_pathwise_exact_agreement"] is False
    assert a2["canonical_a2_pathwise_agreement_discharged"] is False
    assert a2["expected_balancing_is_diagnostic_not_canonical_a2_agreement"] is True
    assert a2["rule_selected_by_canonical_a2_alone"] is False
    assert a2["issue_628_atomic_record_write_identification"] is False
    decision = REPORT["selection_decision"]
    assert decision["registered_ladder_complete_primitive_attempt_alphabet_source_emitted"] is True
    assert decision["registered_ladder_exact_unit_counting_source_emitted_on_declared_branch"] is True
    assert decision["infinite_tower_complete_primitive_attempt_alphabet_source_emitted"] is False
    assert decision["infinite_tower_exact_unit_counting_source_emitted_on_declared_branch"] is False
    assert decision["canonical_a1_a3_force_the_emitter"] is False
    assert decision["issue_628_atomic_record_bridge_discharged"] is False
    assert decision["full_refinement_commuting_diagram_discharged"] is False
    assert decision["physical_prediction"] is False
    assert decision["promotion_allowed"] is False


def test_independent_verifier_reconstructs_complete_packet(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    _write(receipt, REPORT)
    verification = independent.verify_receipt(receipt)
    assert verification["receipt"] is True
    assert verification["checked_levels"] == 6
    assert verification["checked_primitive_events"] == 40950
    assert verification["checked_refinement_rows"] == 5
    assert verification["producer_imported"] is False
    assert verification["registered_tower_builder_shared"] is True
    assert verification["source_engine_independently_reimplemented"] is False
    assert (
        verification[
            "event_identity_lineage_and_refinement_algebra_independently_reimplemented"
        ]
        is True
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["level_alphabets"][2][
                "complete_registered_unoriented_seams"
            ].pop(),
            "complete seam alphabet drift",
        ),
        (
            lambda value: value["level_alphabets"][3][
                "declared_multiplicity_by_a5_orbit"
            ].__setitem__(0, 2),
            "unit counting across orbit classes failed",
        ),
        (
            lambda value: value["refinement_certificate"]["rows"][0][
                "fine_parent_seam_indices"
            ].__setitem__(0, 1),
            "serialized seam lineage drift",
        ),
        (
            lambda value: value["selection_decision"].__setitem__(
                "canonical_a1_a3_force_the_emitter", True
            ),
            "forbidden decision promotion",
        ),
        (
            lambda value: value["refinement_certificate"]["rows"][0].__setitem__(
                "strong_intertwiner", True
            ),
            "refinement boundary drift",
        ),
        (
            lambda value: value["a2_reconciliation"].__setitem__(
                "target_value", 137.036
            ),
            "expectation-level balancing diagnostic key set drift",
        ),
        (
            lambda value: value["level_alphabets"][0][
                "target_or_comparison_fields_read"
            ].append("sky_target"),
            "level target boundary drift",
        ),
    ],
)
def test_independent_verifier_rejects_rehashed_semantic_mutations(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    tampered = copy.deepcopy(REPORT)
    mutation(tampered)
    _rehash(tampered)
    receipt = tmp_path / "tampered.json"
    _write(receipt, tampered)
    with pytest.raises(independent.VerificationError, match=message):
        independent.verify_receipt(receipt)


def test_canonical_receipt_is_byte_exact(tmp_path: Path) -> None:
    rebuilt = tmp_path / producer.DEFAULT_RECEIPT.name
    subprocess.run(
        [
            sys.executable,
            "-m",
            "oph_fpe.cosmology.all_level_primitive_seam_source",
            "--output",
            str(rebuilt),
        ],
        cwd=producer.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert rebuilt.read_bytes() == producer.DEFAULT_RECEIPT.read_bytes()
