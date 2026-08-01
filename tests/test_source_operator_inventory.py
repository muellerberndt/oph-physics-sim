from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oph_fpe.dynamics import source_operator_inventory as inventory
from oph_fpe.dynamics import verify_source_operator_inventory_independent as independent


REPORT = inventory.build_inventory()


def _rehash(report: dict) -> None:
    payload = copy.deepcopy(report)
    payload.pop("receipt_sha256", None)
    report["receipt_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _row(report: dict, path: str) -> dict:
    return next(
        row for row in report["canonical_artifact_rows"] if row["path"] == path
    )


def test_inventory_replays_on_the_indexed_serialized_data_surface() -> None:
    verification = independent.verify(REPORT)
    assert verification["receipt"] is True, verification["reasons"]
    assert REPORT["status"] == inventory.STATUS
    assert REPORT["issue"] == 655
    assert REPORT["scope"] == (
        "Git-indexed tracked paths under data; semantic scan of current canonical "
        "simulator JSON objects excluding the recursive inventory and parent bridge "
        "outputs; legacy, imported, and external/comparison paths counted only"
    )

    catalog = REPORT["tracked_serialized_data_catalog"]
    assert catalog["path_count_including_declared_recursive_outputs"] >= 1000
    assert catalog["content_index_row_count_excluding_recursive_outputs"] >= 998
    assert catalog["provenance_counts"]["LEGACY_EARNED_RUN"] >= 895
    assert catalog["provenance_counts"]["IMPORTED_NONNATIVE"] >= 30
    assert catalog["provenance_counts"]["EXTERNAL_OR_COMPARISON_DATA"] >= 40
    assert catalog["untracked_data_paths_excluding_declared_recursive_outputs"] == []
    assert catalog["unstaged_current_canonical_inputs"] == []


def test_only_current_canonical_json_is_scanned_semantically() -> None:
    scan = REPORT["current_canonical_json_contract_scan"]
    assert scan["current_canonical_json_path_count_excluding_recursive_outputs"] == (
        len(inventory.CANONICAL_CONTRACTS) - len(inventory.DECLARED_OUTPUT_PATHS)
    )
    assert scan["recursive_output_paths_excluded"] == sorted(
        inventory.DECLARED_OUTPUT_PATHS
    )
    assert scan["registered_source_packet_rows_excluding_recursive_outputs"] == []
    assert scan["positive_promotion_signal_rows_excluding_recursive_outputs"] == []

    for label in (
        "LEGACY_EARNED_RUN",
        "IMPORTED_NONNATIVE",
        "EXTERNAL_OR_COMPARISON_DATA",
    ):
        assert REPORT["noncurrent_path_catalog"][label][
            "semantic_payloads_scanned"
        ] is False
    assert all(
        not row["path"].startswith(inventory.NONCURRENT_PREFIXES)
        for row in REPORT["canonical_artifact_rows"]
    )


def test_every_current_canonical_json_has_an_exact_contract() -> None:
    rows = REPORT["canonical_artifact_rows"]
    assert {row["path"] for row in rows} == set(inventory.CANONICAL_CONTRACTS)
    for row in rows:
        contract = inventory.CANONICAL_CONTRACTS[row["path"]]
        assert row["schema"] == contract["schema"]
        assert row["status"] == contract["status"]
        assert row["disposition"] == contract["disposition"]
        assert row["semantic_scan_excluded_as_recursive_output"] == (
            row["path"] in inventory.DECLARED_OUTPUT_PATHS
        )
        if row["path"] in inventory.DECLARED_OUTPUT_PATHS:
            assert "raw_pin" not in row
        else:
            assert row["raw_pin"]["path"] == row["path"]


def test_ordered_port_diagnostic_is_indexed_with_exact_negative_evidence() -> None:
    path = "data/a2_holonomy/ordered_port_response_diagnostic_receipt.json"
    row = _row(REPORT, path)
    assert row["schema"] == "oph.ordered-port-response-diagnostic.v1"
    assert row["status"] == "ATTAINED_BOUNDED_NEGATIVE_CONTROL"
    assert row["disposition"] == (
        "TWELVE_PORT_ADJACENCY_PROPAGATION_OVERSHOOTS_TO_U12__"
        "PHYSICAL_CURRENT_SOURCE_OPEN"
    )
    evidence = row["critical_bridge_evidence"]
    assert evidence == {
        "emitted_port_count": 12,
        "emitted_propagation_generator": "minus i times L, where L = 5 I - A",
        "emitted_generated_algebra_type": "u(12)",
        "emitted_generated_algebra_real_rank": 144,
        "emitted_derived_algebra_type": "su(12)",
        "emitted_derived_algebra_real_rank": 143,
        "emitted_A1_complete_response_receipt": False,
        "emitted_A2_same_current_receipt": False,
        "emitted_physical_current_source_bridge_receipt": False,
        "emitted_u12_is_candidate_oph_current": False,
    }


def test_near_candidates_remain_separate_objects() -> None:
    charged = _row(
        REPORT, "data/common_reserve/charged_response_artifact.json"
    )["critical_bridge_evidence"]
    assert charged["emitted_support_size"] == 12
    assert charged["emitted_source_response_operator"] == (
        "negative_graph_antipode_involution"
    )
    assert charged["emitted_source_bound_impulse_readback"] is True
    assert charged["emitted_current_lift_source_selected"] is False
    assert charged["spatial_translation_binding"] == {
        "classification": "ABSENT_FROM_DECLARED_SCHEMA",
        "key": "spatial_translation_identification",
        "searched_scope": "entire_canonical_json_object",
        "occurrences": [],
    }
    assert charged["same_operator_physical_readout"]["classification"] == (
        "ABSENT_FROM_DECLARED_SCHEMA"
    )

    stage3 = _row(REPORT, "data/local_domain/stage3_receipt.json")[
        "critical_bridge_evidence"
    ]
    assert stage3["emitted_visible_edge_count"] == 11816
    assert stage3["emitted_physical_promotion_allowed"] is False
    assert stage3["vertex12_identity_bridge"]["classification"] == (
        "ABSENT_FROM_DECLARED_SCHEMA"
    )

    gap = _row(REPORT, "data/local_domain/source_gap_receipt.json")[
        "critical_bridge_evidence"
    ]
    assert gap["emitted_operator"] == (
        "signed Laplacian of the observer-visible seam complex"
    )
    assert gap["emitted_physical_promotion_allowed"] is False
    assert gap["physical_reference_transition"]["classification"] == (
        "ABSENT_FROM_DECLARED_SCHEMA"
    )


def test_admission_counts_and_boundary_are_scope_qualified() -> None:
    admission = REPORT["bridge_admission_contract"]
    assert admission["registered_packet_count_excluding_recursive_outputs"] == 0
    assert admission["true_promotion_signal_path_count_excluding_recursive_outputs"] == 0
    assert admission["accepted_bridge_count_excluding_recursive_outputs"] == 0
    assert admission["recursive_parent_bridge_receipt_exclusion"] == {
        "path": inventory.BRIDGE_RELATIVE_PATH,
        "reason": (
            "parent output embeds the current negative source packet and is "
            "excluded to avoid recursive custody"
        ),
        "packet_count_included_in_scan": False,
    }

    boundary = REPORT["epistemic_boundary"]
    assert boundary["local_spatial_or_kinetic_operators_exist"] is True
    assert boundary["twelve_port_response_and_readback_exist"] is True
    assert boundary["claim_that_no_spatial_operator_exists"] is False
    assert boundary[
        "registered_accepted_same_domain_chain_on_scanned_surface_exists"
    ] is False
    assert boundary["unregistered_equivalent_semantics_ruled_out"] is False
    assert boundary["producer_code_or_sibling_repository_absence_claimed"] is False
    assert boundary["physical_prediction_unsealed"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "disposition",
        "critical_evidence",
        "schema",
        "scope",
        "issue",
        "required_chain",
        "raw_pin_path",
        "duplicated_row",
        "qualified_count",
        "recursive_exclusion",
    ],
)
def test_rehashed_semantic_mutations_fail_independent_verification(
    mutation: str,
) -> None:
    report = copy.deepcopy(REPORT)
    charged = _row(report, "data/common_reserve/charged_response_artifact.json")
    if mutation == "disposition":
        charged["disposition"] = "PROMOTED"
    elif mutation == "critical_evidence":
        charged["critical_bridge_evidence"][
            "emitted_current_lift_source_selected"
        ] = True
    elif mutation == "schema":
        charged["schema"] = "oph.mutated.v1"
    elif mutation == "scope":
        report["scope"] = "all files everywhere"
    elif mutation == "issue":
        report["issue"] = 0
    elif mutation == "required_chain":
        report["bridge_admission_contract"]["required_chain"].pop()
    elif mutation == "raw_pin_path":
        charged["raw_pin"]["path"] = (
            "data/local_domain/source_gap_receipt.json"
        )
    elif mutation == "duplicated_row":
        report["canonical_artifact_rows"].append(copy.deepcopy(charged))
    elif mutation == "qualified_count":
        report["bridge_admission_contract"][
            "registered_packet_count_excluding_recursive_outputs"
        ] = 1
    elif mutation == "recursive_exclusion":
        report["bridge_admission_contract"][
            "recursive_parent_bridge_receipt_exclusion"
        ]["packet_count_included_in_scan"] = True
    else:  # pragma: no cover
        raise AssertionError(mutation)
    _rehash(report)

    result = independent.verify(report)
    assert result["receipt"] is False
    assert result["status"] == "FAIL"


def test_untracked_and_unstaged_current_input_gates_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inventory,
        "_untracked_data_paths",
        lambda: ["data/unregistered.json"],
    )
    with pytest.raises(ValueError, match="untracked data paths"):
        inventory.build_inventory()

    monkeypatch.setattr(inventory, "_untracked_data_paths", lambda: [])
    monkeypatch.setattr(
        inventory,
        "_unstaged_current_inputs",
        lambda: ["data/local_domain/source_gap_receipt.json"],
    )
    with pytest.raises(ValueError, match="unstaged current canonical inputs"):
        inventory.build_inventory()


def test_schema_status_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original_load = inventory._load_json

    def load_with_drift(path: str) -> dict:
        value = original_load(path)
        if path == "data/common_reserve/charged_response_artifact.json":
            value["schema"] = "oph.mutated.v1"
        return value

    monkeypatch.setattr(inventory, "_load_json", load_with_drift)
    with pytest.raises(ValueError, match="canonical schema/status drift"):
        inventory.build_inventory()


def test_fresh_process_verifier_observes_the_filesystem(tmp_path: Path) -> None:
    report_path = tmp_path / "source_operator_ancestry_inventory.json"
    report_path.write_text(
        json.dumps(REPORT, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "oph_fpe.dynamics.verify_source_operator_inventory_independent",
            str(report_path),
        ],
        cwd=inventory.REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["receipt"] is True

    mutated = copy.deepcopy(REPORT)
    mutated["issue"] = 0
    _rehash(mutated)
    report_path.write_text(
        json.dumps(mutated, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "oph_fpe.dynamics.verify_source_operator_inventory_independent",
            str(report_path),
        ],
        cwd=inventory.REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert json.loads(completed.stdout)["receipt"] is False


def test_writer_and_canonical_receipt_are_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "source_operator_ancestry_inventory.json"
    written = inventory.write_inventory(output)
    assert written == REPORT
    assert json.loads(output.read_text(encoding="utf-8")) == REPORT
    canonical = json.loads(inventory.OUTPUT_PATH.read_text(encoding="utf-8"))
    assert canonical == REPORT
    assert inventory.verify_inventory(canonical)["receipt"] is True
    assert independent.verify(canonical)["receipt"] is True
