"""Independent verifier for the current #655 bounded-nonselection exit.

This implementation deliberately does not import the bridge producer or the
canonical repair producer.  It checks the receipt digest, every raw source
pin, the finite carrier incidence counts, the repair-law scope flags, the
separation of internal repair from spatial propagation and physical readout,
and the resulting live-issue exit.  Its scope is the bounded negative exit on
the indexed serialized-data surface;
it cannot verify a future positive physical bridge.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.dynamics import verify_source_operator_inventory_independent


REPORT_SCHEMA = "oph.port_repair_propagation_bridge_receipt.v1"
VERIFICATION_SCHEMA = (
    "oph.port_repair_propagation_bridge_independent_verification.v1"
)
SOURCE_PACKET_SCHEMA = "oph.port_repair_propagation_source_packet.v1"
CURRENT_OPERATOR_EXIT = "NO_SOURCE_NATIVE_TRANSLATION_BRIDGE"
CURRENT_ISSUE_EXIT = "BOUNDED_NONSELECTION__FZ11_REMAINS_BRANCH_PREDICTION"
INVENTORY_STATUS = (
    "NO_REGISTERED_ACCEPTED_VERTEX12_BRIDGE_PACKET_ON_TRACKED_SERIALIZED_DATA_SURFACE"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = (
    REPOSITORY_ROOT
    / "data/repair_closure/port_repair_propagation_bridge_receipt.json"
)

CARRIER_PATH = "tests/fixtures/echosahedral_federation_reference.json"
REPAIR_PRODUCER_PATH = "oph_fpe/dynamics/canonical_seam_repair.py"
BOUNDED_REPAIR_PATH = (
    "data/repair_closure/bounded_atomic_self_readback_closure_receipt.json"
)
INVENTORY_PATH = "data/repair_closure/source_operator_ancestry_inventory.json"
BRIDGE_PRODUCER_PATH = "oph_fpe/dynamics/port_repair_propagation_bridge.py"
INDEPENDENT_VERIFIER_PATH = (
    "oph_fpe/dynamics/verify_port_repair_propagation_bridge_independent.py"
)
BRIDGE_TEST_PATH = "tests/test_port_repair_propagation_bridge.py"

EXPECTED_SCOPE_BOUNDARY = {
    "internal_seams_are_spatial_hops": False,
    "equal_support_counts_imply_operator_identity": False,
    "physicalization_claimed": False,
}

EXPECTED_DEPENDENCY_AUDIT = {
    "closed_issue_62_supplies_accepted_physical_repair_law": False,
    "reason": (
        "The current exact repair sources type the scalar and integer seam laws "
        "as candidates and set PHYSICAL_REPAIR_LAW_RECEIPT and "
        "canonical_three_axiom_derivation to false."
    ),
    "geometry_and_orbit_moments_available": True,
    "geometry_implies_physical_translation_operator": False,
    "indexed_tracked_serialized_data_census_complete": True,
    "semantic_scan_limited_to_current_canonical_json": True,
    "legacy_imported_external_payloads_semantically_scanned": False,
    "local_spatial_or_kinetic_operators_exist": True,
    "registered_accepted_vertex12_translation_to_physical_readout_chain_on_scanned_surface_exists": False,
}

EXPECTED_EPISTEMIC_BOUNDARY = {
    "current_result": (
        "The exact vertex, face, and edge rays are classified. On the Git-indexed "
        "tracked simulator data surface, the current canonical JSON scan finds "
        "local spatial and kinetic operators and a twelve-port response, with no "
        "registered accepted packet that binds a complete vertex-orbit translation "
        "operator to a digest-identical physical readout. Recursive outputs are "
        "excluded from the packet count."
    ),
    "claim_that_no_spatial_operator_exists": False,
    "unregistered_equivalent_semantics_ruled_out": False,
    "producer_code_or_sibling_repository_absence_claimed": False,
    "registered_accepted_same_domain_chain_on_scanned_surface_exists": False,
    "internal_thirty_seams_do_not_select_edge30_spatial_hops": True,
    "comparison_data_read": False,
    "physical_prediction_unsealed": False,
    "reopen_condition": (
        "Supply a source-history-replayed translation operator and a digest-bound "
        "physical readout in the same operator domain."
    ),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _safe_pinned_path(relative: Any) -> Path:
    if not isinstance(relative, str):
        raise ValueError("pin path must be a string")
    candidate = (REPOSITORY_ROOT / relative).resolve()
    candidate.relative_to(REPOSITORY_ROOT.resolve())
    if not candidate.is_file():
        raise ValueError("pinned source is not a file")
    return candidate


def _check_raw_pin(
    pin: Any,
    reasons: list[str],
    label: str,
    expected_relative_path: str,
) -> Path | None:
    if not isinstance(pin, Mapping):
        reasons.append(f"{label}_pin_missing")
        return None
    try:
        relative_path = pin.get("repository_relative_path")
        if relative_path != expected_relative_path:
            reasons.append(f"{label}_path_mismatch")
        path = _safe_pinned_path(relative_path)
        raw = path.read_bytes()
        if pin.get("bytes") != len(raw):
            reasons.append(f"{label}_byte_count_mismatch")
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        if pin.get("sha256") != digest:
            reasons.append(f"{label}_sha256_mismatch")
        if set(pin) != {"repository_relative_path", "bytes", "sha256"}:
            reasons.append(f"{label}_pin_has_extra_fields")
        return path
    except (OSError, TypeError, ValueError):
        reasons.append(f"{label}_pin_invalid")
        return None


def _check_carrier(path: Path | None, reasons: list[str]) -> None:
    if path is None:
        return
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "oph.echosahedral_selector_manifest.v1":
            reasons.append("carrier_schema_mismatch")
            return
        carrier = manifest.get("carrier")
        if not isinstance(carrier, Mapping):
            reasons.append("carrier_block_missing")
            return
        ports = carrier.get("ports")
        if not isinstance(ports, list) or len(ports) != 12 or len(set(ports)) != 12:
            reasons.append("carrier_port_count_mismatch")
            return
        port_set = set(ports)
        edges_raw = carrier.get("edges")
        if not isinstance(edges_raw, list):
            reasons.append("carrier_edges_missing")
            return
        edges: set[tuple[str, str]] = set()
        degree = {port: 0 for port in ports}
        for edge in edges_raw:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or edge[0] not in port_set
                or edge[1] not in port_set
                or edge[0] == edge[1]
            ):
                reasons.append("carrier_edge_invalid")
                return
            normalized = tuple(sorted((edge[0], edge[1])))
            if normalized in edges:
                reasons.append("carrier_edge_duplicate")
                return
            edges.add(normalized)
            degree[edge[0]] += 1
            degree[edge[1]] += 1
        if len(edges) != 30 or set(degree.values()) != {5}:
            reasons.append("carrier_incidence_mismatch")
        faces = carrier.get("oriented_faces")
        if not isinstance(faces, list) or len(faces) != 20:
            reasons.append("carrier_face_count_mismatch")
            return
        for face in faces:
            if not isinstance(face, list) or len(face) != 3 or len(set(face)) != 3:
                reasons.append("carrier_face_invalid")
                return
            pairs = {
                tuple(sorted((face[0], face[1]))),
                tuple(sorted((face[0], face[2]))),
                tuple(sorted((face[1], face[2]))),
            }
            if not pairs <= edges:
                reasons.append("carrier_face_not_incident")
                return
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        reasons.append("carrier_manifest_malformed")


def _check_bounded_repair(path: Path | None, reasons: list[str]) -> None:
    if path is None:
        return
    try:
        repair = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema": "oph.bounded_atomic_self_readback_closure.v1",
            "FINITE_DIRECTED_SEAM_TORSOR_RECEIPT": True,
            "PHYSICAL_REPAIR_LAW_RECEIPT": False,
            "GLOBAL_A1_A3_POLICY_UNIQUENESS_RECEIPT": False,
        }
        for field, value in expected.items():
            if repair.get(field) != value:
                reasons.append(f"bounded_repair_{field}_mismatch")
        clauses = repair.get("axiom_clause_specialization")
        if not isinstance(clauses, Mapping):
            reasons.append("bounded_repair_axiom_scope_missing")
        else:
            if clauses.get("basis_status") != "proposed_a1r_a2r_specialization_not_adopted":
                reasons.append("bounded_repair_basis_status_mismatch")
            if clauses.get("canonical_three_axiom_derivation") is not False:
                reasons.append("bounded_repair_axiom_derivation_promoted")
            if clauses.get("full_a1_repair_grammar_certified") is not False:
                reasons.append("bounded_repair_grammar_promoted")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        reasons.append("bounded_repair_receipt_malformed")


def _check_operator_inventory(block: Any, reasons: list[str]) -> None:
    if not isinstance(block, Mapping):
        reasons.append("source_operator_inventory_missing")
        return
    path = _check_raw_pin(
        block.get("raw_pin"),
        reasons,
        "source_operator_inventory",
        INVENTORY_PATH,
    )
    if path is None:
        return
    try:
        inventory = json.loads(path.read_text(encoding="utf-8"))
        verification = verify_source_operator_inventory_independent.verify(inventory)
        if not verification["receipt"]:
            reasons.extend(
                f"source_operator_inventory:{reason}"
                for reason in verification["reasons"]
            )
            return
        expected = {
            "raw_pin": block["raw_pin"],
            "schema": "oph.source_operator_ancestry_inventory.v1",
            "status": INVENTORY_STATUS,
            "receipt_sha256": inventory["receipt_sha256"],
            "indexed_tracked_serialized_data_path_count": inventory[
                "tracked_serialized_data_catalog"
            ]["path_count_including_declared_recursive_outputs"],
            "current_canonical_json_path_count_excluding_recursive_outputs": inventory[
                "current_canonical_json_contract_scan"
            ]["current_canonical_json_path_count_excluding_recursive_outputs"],
            "registered_packet_count_excluding_recursive_outputs": inventory[
                "bridge_admission_contract"
            ]["registered_packet_count_excluding_recursive_outputs"],
            "accepted_bridge_count_excluding_recursive_outputs": inventory[
                "bridge_admission_contract"
            ]["accepted_bridge_count_excluding_recursive_outputs"],
            "recursive_parent_bridge_receipt_exclusion": inventory[
                "bridge_admission_contract"
            ]["recursive_parent_bridge_receipt_exclusion"],
            "local_spatial_or_kinetic_operators_exist": True,
            "claim_that_no_spatial_operator_exists": False,
            "producer_code_or_sibling_repository_absence_claimed": False,
            "unregistered_equivalent_semantics_ruled_out": False,
        }
        if dict(block) != expected:
            reasons.append("source_operator_inventory_summary_mismatch")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        reasons.append("source_operator_inventory_malformed")


def verify_independently(report: Mapping[str, Any]) -> dict[str, Any]:
    """Verify only the indexed-surface bounded-nonselection implication."""

    reasons: list[str] = []
    try:
        received = copy.deepcopy(dict(report))
        digest = received.pop("receipt_sha256", None)
        if report.get("schema") != REPORT_SCHEMA:
            reasons.append("schema_mismatch")
        if report.get("issue") != 655:
            reasons.append("issue_mismatch")
        if digest != _canonical_sha256(received):
            reasons.append("receipt_digest_mismatch")

        packet = report.get("source_packet")
        if not isinstance(packet, Mapping):
            reasons.append("source_packet_missing")
            packet = {}
        if packet.get("schema") != SOURCE_PACKET_SCHEMA:
            reasons.append("source_packet_schema_mismatch")

        carrier_path = _check_raw_pin(
            packet.get("carrier_manifest_pin"),
            reasons,
            "carrier_manifest",
            CARRIER_PATH,
        )
        _check_carrier(carrier_path, reasons)

        internal = packet.get("internal_seam_repair")
        if not isinstance(internal, Mapping):
            reasons.append("internal_repair_block_missing")
            internal = {}
        expected_internal = {
            "domain": "finite_twelve_port_scalar_working_readback",
            "support_kind": "thirty_internal_incidence_seams",
            "port_count": 12,
            "support_count": 30,
            "operator": "T = I - L_icosahedron/60",
            "spatial_translation_identification": False,
            "same_operator_physical_readout": False,
            "physical_repair_law_receipt": False,
        }
        for field, value in expected_internal.items():
            if internal.get(field) != value:
                reasons.append(f"internal_repair_{field}_mismatch")
        _check_raw_pin(
            internal.get("canonical_repair_producer_pin"),
            reasons,
            "canonical_repair_producer",
            REPAIR_PRODUCER_PATH,
        )
        repair_path = _check_raw_pin(
            internal.get("bounded_atomic_receipt_pin"),
            reasons,
            "bounded_atomic_receipt",
            BOUNDED_REPAIR_PATH,
        )
        _check_bounded_repair(repair_path, reasons)
        _check_operator_inventory(
            packet.get("source_operator_ancestry_inventory"), reasons
        )
        if report.get("source_operator_ancestry_inventory") != packet.get(
            "source_operator_ancestry_inventory"
        ):
            reasons.append("duplicated_source_operator_inventory_mismatch")

        if packet.get("spatial_hop_operator") is not None:
            reasons.append("spatial_hop_operator_unexpectedly_present")
        if packet.get("physical_readout") is not None:
            reasons.append("physical_readout_unexpectedly_present")
        boundary = packet.get("scope_boundary")
        if not isinstance(boundary, Mapping):
            reasons.append("scope_boundary_missing")
        else:
            if dict(boundary) != EXPECTED_SCOPE_BOUNDARY:
                reasons.append("scope_boundary_mismatch")

        classification = report.get("classification")
        if not isinstance(classification, Mapping):
            reasons.append("classification_missing")
            classification = {}
        expected_classification = {
            "operator_classifier_exit": CURRENT_OPERATOR_EXIT,
            "issue_certified_exit": CURRENT_ISSUE_EXIT,
            "blockers": ["NO_TWELVE_PORT_ORBIT_TRANSLATION_BINDING"],
            "internal_seam_repair_certified": True,
            "spatial_hop_source_certified": False,
            "same_operator_physical_readout_certified": False,
            "selected_spatial_orbit": None,
            "selected_ray": None,
            "internal_support_count_used_as_spatial_selection": False,
            "physicalization_assumed": False,
            "forced_exclusive_issue_exit_attainable_in_v1": False,
        }
        if dict(classification) != expected_classification:
            reasons.append("current_exit_classification_mismatch")
        if report.get("status") != CURRENT_ISSUE_EXIT:
            reasons.append("current_issue_exit_mismatch")

        receipt_flags = {
            "INTERNAL_SEAM_REPAIR_CERTIFIED": True,
            "SPATIAL_PORT_HOP_SOURCE_RECEIPT": False,
            "SAME_OPERATOR_PHYSICAL_READOUT_RECEIPT": False,
            "FZ11_FORCED_EXCLUSIVE_RECEIPT": False,
        }
        for field, value in receipt_flags.items():
            if report.get(field) != value:
                reasons.append(f"receipt_flag_{field}_mismatch")

        audit = report.get("live_issue_acceptance_audit")
        if not isinstance(audit, Mapping):
            reasons.append("live_issue_acceptance_audit_missing")
        else:
            if audit.get("defensible_exit") != CURRENT_ISSUE_EXIT:
                reasons.append("acceptance_audit_exit_mismatch")
            if audit.get("forced_exclusive_exit_supported") is not False:
                reasons.append("forced_exclusive_exit_promoted")
            if audit.get("comparison_data_consumed") is not False:
                reasons.append("comparison_custody_violated")
            if audit.get("independent_full_bridge_implementation") is not False:
                reasons.append("full_independence_overclaimed")

        dependency = report.get("dependency_audit")
        if not isinstance(dependency, Mapping) or dict(dependency) != (
            EXPECTED_DEPENDENCY_AUDIT
        ):
            reasons.append("dependency_inventory_boundary_mismatch")

        epistemic = report.get("epistemic_boundary")
        if not isinstance(epistemic, Mapping) or dict(epistemic) != (
            EXPECTED_EPISTEMIC_BOUNDARY
        ):
            reasons.append("epistemic_inventory_boundary_mismatch")

        pins = report.get("implementation_pins")
        if not isinstance(pins, Mapping):
            reasons.append("implementation_pins_missing")
        else:
            expected_pin_names = {
                "bridge_producer",
                "independent_current_exit_verifier",
                "mutation_and_independent_orbit_tests",
            }
            if set(pins) != expected_pin_names:
                reasons.append("implementation_pin_set_mismatch")
            expected_paths = {
                "bridge_producer": BRIDGE_PRODUCER_PATH,
                "independent_current_exit_verifier": INDEPENDENT_VERIFIER_PATH,
                "mutation_and_independent_orbit_tests": BRIDGE_TEST_PATH,
            }
            for name, expected_path in expected_paths.items():
                _check_raw_pin(pins.get(name), reasons, name, expected_path)
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
        reasons.append("malformed_or_noncanonical_receipt")

    passed = not reasons
    return {
        "schema": VERIFICATION_SCHEMA,
        "receipt": passed,
        "status": "PASS" if passed else "FAIL",
        "reasons": sorted(set(reasons)),
        "independent_implementation": True,
        "imports_bridge_producer": False,
        "verified_exit": CURRENT_ISSUE_EXIT if passed else None,
        "scope": "current_source_pin_domain_separation_and_bounded_nonselection",
        "positive_physical_bridge_verified": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = verify_independently(report)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["receipt"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
