from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

from oph_fpe.cosmology.angular_refinement_labeled_event_readout import (
    DEFAULT_OUTPUT,
    build_receipt,
    canonical_json,
    self_digest,
    verify_receipt,
    write_receipt,
)
from oph_fpe.cosmology.verify_angular_refinement_labeled_event_readout_independent import (
    verify as verify_independent,
)


def _rehash(report: dict) -> dict:
    report["payload_sha256"] = self_digest(report)
    return report


def test_exact_full_reconstruction_and_protocol_ranks() -> None:
    report = build_receipt()
    exact = report["exact_linear_certificate"]

    assert exact["baseline_Q_shape"] == [12, 42]
    assert exact["baseline_Q_rank_over_Q"] == 12
    assert exact["minimal_selected_scalar_shape"] == [42, 42]
    assert exact["minimal_selected_scalar_rank_over_Q"] == 42
    assert exact["decoder_shape"] == [42, 42]
    assert exact["decoder_times_measurement_is_identity"] is True
    assert exact["all_standard_basis_reconstruction_checks"] == 42
    assert exact["A5_stable_scalar_shape"] == [72, 42]
    assert exact["A5_stable_scalar_rank_over_Q"] == 42
    assert exact["directed_event_rows_without_Q_shape"] == [60, 42]
    assert exact["directed_event_rows_without_Q_rank_over_Q"] == 41
    assert exact["directed_event_rows_without_Q_kernel_dimension"] == 1
    assert exact["selected_full_Q_stack_shape"] == [372, 42]
    assert exact["selected_full_Q_stack_rank_over_Q"] == 42
    assert exact["A5_stable_full_Q_stack_shape"] == [732, 42]
    assert exact["A5_stable_full_Q_stack_rank_over_Q"] == 42


def test_event_inventory_and_A5_scope_are_exact() -> None:
    report = build_receipt()
    instrument = report["labeled_event_instrument"]
    symmetry = report["symmetry_certificate"]

    assert instrument["directed_event_count"] == 60
    assert instrument["minimal_selected_event_count"] == 30
    assert len(instrument["directed_events"]) == 60
    assert len(set(instrument["minimal_selected_event_labels"])) == 30
    assert instrument["averaging_factor"] == "1/2"
    assert instrument["fine_edge_membership_checks"] == 60
    assert instrument["all_directed_events_are_fine_edges"] is True
    assert instrument["same_pre_event_field_required"] is True
    assert instrument["repeated_preparation_or_checkpoint_access_required"] is True
    assert (
        instrument["sequential_destructive_application_without_reset_sufficient"]
        is False
    )
    assert symmetry["proper_rotation_count"] == 60
    assert symmetry["directed_event_covariance_checks"] == 3600
    assert symmetry["representative_orbit_size"] == 60
    assert symmetry["directed_event_family_is_A5_stable"] is True
    assert symmetry["minimal_thirty_event_selector_is_A5_invariant"] is False
    assert symmetry["no_A5_equivariant_one_parent_per_midpoint_section"] is True


def test_passive_rank_29_result_is_preserved_as_protocol_specific() -> None:
    boundary = build_receipt()["protocol_boundary"]
    passive = boundary["existing_passive_rank_29_result"]

    assert passive["matrix"] == "O=(Q,Q L_f,...,Q L_f^41)"
    assert passive["rank_over_Q"] == 29
    assert passive["kernel_dimension"] == 13
    assert passive["status"] == "unchanged and protocol-specific"
    assert boundary["rank_29_is_a_universal_readout_no_go"] is False
    assert boundary["labeled_event_grammar_selected_by_bare_OPH_axioms"] is False
    assert boundary["same_state_instrument_constructed"] is False
    assert boundary["physical_sky_readout_constructed"] is False
    assert boundary["laboratory_observable_constructed"] is False
    assert boundary["issue_closure_authorized"] is False


def test_committed_receipt_and_both_verifiers_pass() -> None:
    report = build_receipt()
    assert DEFAULT_OUTPUT.read_bytes() == canonical_json(report) + b"\n"
    assert verify_receipt(report)["receipt"] is True
    independent = verify_independent(report)
    assert independent["receipt"] is True
    assert independent["independent_implementation"] is True
    assert independent["producer_imported"] is False


def test_independent_verifier_does_not_import_producer() -> None:
    path = (
        Path(__file__).resolve().parents[1] / "oph_fpe/cosmology/"
        "verify_angular_refinement_labeled_event_readout_independent.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(
        name.endswith("angular_refinement_labeled_event_readout") for name in imported
    )


def test_wrong_label_fails_closed_even_after_rehash() -> None:
    report = copy.deepcopy(build_receipt())
    report["labeled_event_instrument"]["directed_events"][0]["label"] = "wrong"
    _rehash(report)

    result = verify_independent(report)
    assert result["receipt"] is False
    assert "event_label_or_inventory_mismatch" in result["reasons"]


def test_omitted_response_fails_closed_even_after_rehash() -> None:
    report = copy.deepcopy(build_receipt())
    report["labeled_event_instrument"]["directed_events"][0].pop("response_sparse")
    _rehash(report)

    result = verify_independent(report)
    assert result["receipt"] is False
    assert "event_response_mismatch" in result["reasons"]


def test_wrong_averaging_factor_fails_closed_even_after_rehash() -> None:
    report = copy.deepcopy(build_receipt())
    report["labeled_event_instrument"]["averaging_factor"] = "1/3"
    report["labeled_event_instrument"]["directed_events"][0]["averaging_factor"] = "1/3"
    _rehash(report)

    result = verify_independent(report)
    assert result["receipt"] is False
    assert "averaging_factor_mismatch" in result["reasons"]


def test_parent_and_implementation_pin_mutations_fail_closed() -> None:
    parent = copy.deepcopy(build_receipt())
    parent["parent_pins"][0]["sha256"] = "sha256:" + "0" * 64
    _rehash(parent)
    result = verify_independent(parent)
    assert result["receipt"] is False
    assert "parent_pin_mismatch" in result["reasons"]

    implementation = copy.deepcopy(build_receipt())
    implementation["implementation_integrity_pins"]["producer"]["bytes"] += 1
    _rehash(implementation)
    result = verify_independent(implementation)
    assert result["receipt"] is False
    assert "implementation_pin_mismatch" in result["reasons"]


def test_same_state_and_physical_promotions_fail_closed() -> None:
    same_state = copy.deepcopy(build_receipt())
    same_state["labeled_event_instrument"]["same_pre_event_field_required"] = False
    _rehash(same_state)
    result = verify_independent(same_state)
    assert result["receipt"] is False
    assert "same_state_protocol_mismatch" in result["reasons"]

    promoted = copy.deepcopy(build_receipt())
    promoted["protocol_boundary"]["physical_sky_readout_constructed"] = True
    _rehash(promoted)
    assert verify_receipt(promoted)["receipt"] is False
    result = verify_independent(promoted)
    assert result["receipt"] is False
    assert "forbidden_protocol_or_physical_promotion" in result["reasons"]


def test_source_scope_promotion_fails_closed_even_after_rehash() -> None:
    report = copy.deepcopy(build_receipt())
    report["source_scope"]["physical_sky_readout_constructed"] = True
    _rehash(report)

    result = verify_independent(report)
    assert result["receipt"] is False
    assert "source_scope_mismatch" in result["reasons"]


def test_claim_boundary_mutation_fails_closed_even_after_rehash() -> None:
    report = copy.deepcopy(build_receipt())
    report["claim_boundary"] = "promoted to a physical sky readout"
    _rehash(report)

    result = verify_independent(report)
    assert result["receipt"] is False
    assert "claim_boundary_mismatch" in result["reasons"]


def test_non_fine_edge_event_fails_closed_even_after_rehash() -> None:
    report = copy.deepcopy(build_receipt())
    event = report["labeled_event_instrument"]["directed_events"][0]
    assert event["parent_edge"] == [0, 11]
    event["parent_inherited_slot"] = 1
    event["selected_Q_coordinate"] = 1
    event["response_sparse"] = {
        "1": "1/2",
        str(event["midpoint_carrier_slot"]): "1/2",
    }
    _rehash(report)

    result = verify_independent(report)
    assert result["receipt"] is False
    assert "fine_edge_membership_mismatch" in result["reasons"]


def test_minimal_selector_mutation_fails_closed_even_after_rehash() -> None:
    report = copy.deepcopy(build_receipt())
    report["labeled_event_instrument"]["minimal_selector"] = (
        "arbitrary undeclared selector"
    )
    _rehash(report)

    result = verify_independent(report)
    assert result["receipt"] is False
    assert "event_instrument_metadata_mismatch" in result["reasons"]


def test_protocol_explanation_mutation_fails_closed_even_after_rehash() -> None:
    report = copy.deepcopy(build_receipt())
    report["protocol_boundary"]["why_no_contradiction"] = (
        "the passive rank-29 result is a universal no-go"
    )
    _rehash(report)

    result = verify_independent(report)
    assert result["receipt"] is False
    assert "protocol_boundary_mismatch" in result["reasons"]


def test_fine_edge_certification_mutation_fails_closed_even_after_rehash() -> None:
    report = copy.deepcopy(build_receipt())
    report["labeled_event_instrument"]["fine_edge_membership_checks"] = 59
    report["labeled_event_instrument"]["all_directed_events_are_fine_edges"] = False
    _rehash(report)

    result = verify_independent(report)
    assert result["receipt"] is False
    assert "fine_edge_membership_mismatch" in result["reasons"]


def test_writer_and_independent_entrypoint(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    report = write_receipt(path)
    assert json.loads(path.read_text(encoding="ascii")) == report
    assert path.read_bytes().endswith(b"\n")
    assert b"\r\n" not in path.read_bytes()

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            (
                "oph_fpe.cosmology."
                "verify_angular_refinement_labeled_event_readout_independent"
            ),
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["receipt"] is True
