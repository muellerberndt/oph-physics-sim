"""Independent verifier for the issue-655 serialized-data packet census.

This module does not import either #655 producer.  It reconstructs the indexed
path catalog, clean-input gates, current canonical schema/status contracts,
artifact dispositions, critical evidence, admission policy, boundaries, and
exact implementation pins.  Legacy, imported, and external payloads are
counted by indexed path and are not parsed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from oph_fpe.dynamics import verify_vertex12_atomic_port_transfer_independent


SCHEMA = "oph.source_operator_ancestry_inventory.v1"
VERIFICATION_SCHEMA = "oph.source_operator_ancestry_inventory_independent_verification.v1"
SOURCE_PACKET_SCHEMA = "oph.port_repair_propagation_source_packet.v1"
STATUS = "NO_REGISTERED_ACCEPTED_VERTEX12_BRIDGE_PACKET_ON_TRACKED_SERIALIZED_DATA_SURFACE"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPOSITORY_ROOT / "data/repair_closure/source_operator_ancestry_inventory.json"
INVENTORY_RELATIVE_PATH = "data/repair_closure/source_operator_ancestry_inventory.json"
BRIDGE_RELATIVE_PATH = "data/repair_closure/port_repair_propagation_bridge_receipt.json"
DECLARED_OUTPUT_PATHS = {INVENTORY_RELATIVE_PATH, BRIDGE_RELATIVE_PATH}
PRODUCER_PATH = REPOSITORY_ROOT / "oph_fpe/dynamics/source_operator_inventory.py"
INDEPENDENT_VERIFIER_PATH = Path(__file__).resolve()
TEST_PATH = REPOSITORY_ROOT / "tests/test_source_operator_inventory.py"

NONCURRENT_PREFIXES = (
    "data/earned_runs/",
    "data/oph_cross_repo_current/",
    "data/measurements/",
    "data/flyby/",
    "data/gallium/",
)


def _contract(schema: str | None, status: str | None, disposition: str) -> dict[str, Any]:
    return {"schema": schema, "status": status, "disposition": disposition}


EXPECTED_CONTRACTS: dict[str, dict[str, Any]] = {
    "data/a2_holonomy/a2_holonomy_current_selector_report.json": _contract(
        "oph.a2-holonomy-current-selector/1.0.0", "OPEN_SOURCE_HOLONOMY_BRIDGE", "GAUGE_CURRENT_HOLONOMY_OPEN_NOT_SPATIAL_PROPAGATION"
    ),
    "data/a2_holonomy/ordered_port_response_diagnostic_receipt.json": _contract(
        "oph.ordered-port-response-diagnostic.v1", "ATTAINED_BOUNDED_NEGATIVE_CONTROL", "TWELVE_PORT_ADJACENCY_PROPAGATION_OVERSHOOTS_TO_U12__PHYSICAL_CURRENT_SOURCE_OPEN"
    ),
    "data/capacity_readback/capacity_indexed_source_family_independent_receipt.json": _contract(
        "oph.capacity_indexed_source_family_independent_receipt.v1", "PASS", "CAPACITY_READBACK_NOT_PROPAGATION"
    ),
    "data/capacity_readback/capacity_indexed_source_family_projection.json": _contract(
        "oph.capacity_indexed_source_family_projection.v1", None, "CAPACITY_SOURCE_PROJECTION_NOT_PROPAGATION"
    ),
    "data/common_reserve/charged_response_artifact.json": _contract(
        "oph.charged_response_semantic_artifact.v3", None, "TWELVE_PORT_RESPONSE_NOT_SPATIAL_TRANSLATION"
    ),
    "data/common_reserve/producer_capability_matrix.json": _contract(
        "oph.common-reserve.capability-matrix.v1", "CAPABILITY_PROBE_COMPLETE__SCIENTIFIC_PROMOTION_DISABLED", "RAW_RESPONSE_NATIVE_PHYSICAL_AT_BRIDGE_OPEN"
    ),
    "data/einstein_convergence/manifest.json": _contract(
        "oph.einstein-convergence-ladder.v2", None, "EINSTEIN_LADDER_MANIFEST_NOT_VERTEX12_OPERATOR"
    ),
    "data/einstein_convergence/rung_16384.json": _contract(None, None, "EINSTEIN_LADDER_RUNG_NOT_VERTEX12_OPERATOR"),
    "data/einstein_convergence/rung_65536.json": _contract(None, None, "EINSTEIN_LADDER_RUNG_NOT_VERTEX12_OPERATOR"),
    "data/einstein_convergence/rung_262144.json": _contract(None, None, "EINSTEIN_LADDER_RUNG_NOT_VERTEX12_OPERATOR"),
    "data/einstein_convergence/rung_262144_dense.json": _contract(None, None, "EINSTEIN_LADDER_RUNG_NOT_VERTEX12_OPERATOR"),
    "data/local_domain/classical_realization_receipt.json": _contract(
        "oph.local-domain-classical-realization.v1", "CLASSICAL_REALIZATION_MATCHES_DECLARED_FINITE_SPECTRAL_INTERFACE", "LOCAL_DOMAIN_CLASSICAL_OPERATOR_NO_VERTEX12_IDENTITY_BRIDGE"
    ),
    "data/local_domain/clock_unit_verdict.json": _contract(
        "oph.local-domain-clock-unit-verdict.v1", "PHYSICAL_UNITS_NOT_EVALUABLE_ON_DECLARED_SERIALIZED_INTERFACE", "LOCAL_DOMAIN_PHYSICAL_UNITS_AND_READOUT_NOT_EVALUABLE"
    ),
    "data/local_domain/defect_sector_receipt.json": _contract(
        "oph.local-domain-defect-sector-spectra.v1", "ATTAINED", "LOCAL_TWISTED_OPERATORS_NO_VERTEX12_IDENTITY_BRIDGE"
    ),
    "data/local_domain/manifest.json": _contract(
        "oph.local-domain-stage1.manifest.v1", None, "LOCAL_DOMAIN_AGGREGATE_MANIFEST_NO_NEW_OPERATOR"
    ),
    "data/local_domain/matter_attachment_receipt.json": _contract(
        "oph.local-domain-matter-attachment.v1", "ATTAINED", "DECLARED_TENSOR_OPERATOR_AND_SEPARATE_SPIN_PACKET_UNBRIDGED"
    ),
    "data/local_domain/source_gap_receipt.json": _contract(
        "oph.source-clock-gap.v1", "ATTAINED", "LOCAL_SIGNED_LAPLACIAN_NO_PHYSICAL_READOUT"
    ),
    "data/local_domain/stage1_receipt.json": _contract(
        "oph.local-domain-stage1.v1", "ATTAINED", "PRESCRIBED_FINITE_CHART_NOT_PROPAGATION_OPERATOR"
    ),
    "data/local_domain/stage2_receipt.json": _contract(
        "oph.local-domain-stage2.v1", "ATTAINED", "GF2_SEAM_TRANSPORT_NOT_SPATIAL_TRANSLATION"
    ),
    "data/local_domain/stage3_receipt.json": _contract(
        "oph.local-domain-stage3.v1", "ATTAINED", "LOCAL_DIFFERENCE_OPERATOR_NOT_VERTEX12_PHYSICAL_OPERATOR"
    ),
    "data/local_domain/stage4_receipt.json": _contract(
        "oph.local-domain-stage4.v1", "ATTAINED", "LOCAL_DOMAIN_PROVENANCE_AGGREGATE_NO_NEW_OPERATOR"
    ),
    "data/quantum/icosahedral_chsh_candidate_receipt.json": _contract(
        "oph.icosahedral_chsh_candidate.v1", "EXACT_PROJECTIVE_BRANCH_CANDIDATE__TWO_WING_COMPLETED_RECORD_SOURCE_PRODUCER_MISSING", "PROJECTIVE_QUANTUM_CANDIDATE_COMPLETED_RECORD_PRODUCER_MISSING"
    ),
    "data/refinement/physical_birefinement_preflight.json": _contract(
        "oph.refinement.physical-birefinement-preflight.v1", "SOURCE_PRODUCER_MISSING", "PHYSICAL_BIREFINEMENT_SOURCE_PRODUCER_MISSING"
    ),
    "data/repair_closure/angular_refinement_repair_observability_receipt.json": _contract(
        "oph.angular_refinement_repair_observability.v1", "EXACT_REFINEMENT_REPAIR_COUNTERENSEMBLE__DETAIL_COVARIANCE_UNSELECTED__PHYSICAL_SKY_READOUT_OPEN", "ANGULAR_READOUT_PHYSICAL_SKY_BINDING_OPEN"
    ),
    "data/repair_closure/bounded_atomic_self_readback_closure_receipt.json": _contract(
        "oph.bounded_atomic_self_readback_closure.v1", "BOUNDED_EXPECTATION_LEVEL_REPAIR_FIXED_POINT_UNIQUE_MODULO_CLOCK_IN_THE_FROZEN_ADVERSARIAL_SUITE", "INTERNAL_REPAIR_PHYSICAL_LAW_NOT_SELECTED"
    ),
    BRIDGE_RELATIVE_PATH: _contract(
        "oph.port_repair_propagation_bridge_receipt.v1", "BOUNDED_NONSELECTION__FZ11_REMAINS_BRANCH_PREDICTION", "RECURSIVE_PARENT_BRIDGE_RECEIPT_EXCLUDED_FROM_SEMANTIC_SCAN"
    ),
    "data/repair_closure/record_counting_source_projection.json": _contract(
        "oph.record_counting_source_projection.v1", None, "RECORD_COUNTING_PROJECTION_NOT_PROPAGATION"
    ),
    "data/repair_closure/seam_equalizer_current_control_report.json": _contract(
        "oph.seam-equalizer-current-control/1.0.0", "ATTAINED_NEGATIVE_CONTROL", "SEAM_EQUALIZER_CURRENT_NEGATIVE_CONTROL"
    ),
    "data/repair_closure/vertex12_atomic_port_transfer_receipt.json": _contract(
        "oph.vertex12-atomic-port-transfer-subpacket.v1",
        "INTERNAL_VERTEX12_SEAM_MATCHING_PROJECTORS_AND_IN_PROCESS_SNAPSHOT_REREAD_ATTAINED__SPATIAL_PHYSICAL_BRIDGE_OPEN",
        "INTERNAL_VERTEX12_SEAM_MATCHING_PROJECTORS_AND_IN_PROCESS_SNAPSHOT_REREAD_ATTAINED__SPATIAL_TRANSLATION_AND_PHYSICAL_READOUT_OPEN",
    ),
    INVENTORY_RELATIVE_PATH: _contract(
        SCHEMA, STATUS, "RECURSIVE_INVENTORY_OUTPUT_EXCLUDED_FROM_SEMANTIC_SCAN"
    ),
}

POSITIVE_SIGNAL_KEYS = (
    "source_native_translation_receipt",
    "source_native_spatial_translation_receipt",
    "same_operator_receipt",
    "spatial_translation_identification",
    "same_operator_physical_readout",
    "same_operator_physical_readout_receipt",
    "internal_seam_transfer_is_spatial_translation",
    "directed_antipode_inverse_transport_receipt",
    "noncollapsed_quotient_site_map_receipt",
    "physical_sector_readout",
    "independent_persistence_readback",
    "independent_second_producer_readback",
    "physical_prediction_unsealed",
    "SPATIAL_PORT_HOP_SOURCE_RECEIPT",
    "SAME_OPERATOR_PHYSICAL_READOUT_RECEIPT",
    "FZ11_FORCED_EXCLUSIVE_RECEIPT",
    "PHYSICAL_CURRENT_SOURCE_BRIDGE_RECEIPT",
    "A2_HOLONOMY_SOURCE_BRIDGE_RECEIPT",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": path.relative_to(REPOSITORY_ROOT).as_posix(), "bytes": len(raw), "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}


def _run_git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPOSITORY_ROOT, check=True, capture_output=True).stdout


def _git_index_rows() -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for raw_entry in _run_git("ls-files", "--stage", "-z", "data").split(b"\0"):
        if not raw_entry:
            continue
        header, raw_path = raw_entry.split(b"\t", 1)
        mode, object_id, stage = header.decode("ascii").split()
        if stage != "0":
            raise ValueError("unmerged data path")
        path = raw_path.decode("utf-8")
        rows[path] = {"path": path, "mode": mode, "object_id": object_id}
    for path in DECLARED_OUTPUT_PATHS:
        rows.setdefault(path, {"path": path, "mode": "DECLARED_OUTPUT", "object_id": "EXCLUDED"})
    return [rows[path] for path in sorted(rows)]


def _untracked_data_paths() -> list[str]:
    paths = [p.decode("utf-8") for p in _run_git("ls-files", "--others", "--exclude-standard", "-z", "--", "data").split(b"\0") if p]
    return sorted(path for path in paths if path not in DECLARED_OUTPUT_PATHS)


def _unstaged_current_inputs() -> list[str]:
    inputs = sorted(set(EXPECTED_CONTRACTS) - DECLARED_OUTPUT_PATHS)
    raw = _run_git("diff", "--name-only", "-z", "--", *inputs)
    return sorted(p.decode("utf-8") for p in raw.split(b"\0") if p)


def _provenance(path: str) -> str:
    if path.startswith("data/earned_runs/"):
        return "LEGACY_EARNED_RUN"
    if path.startswith("data/oph_cross_repo_current/"):
        return "IMPORTED_NONNATIVE"
    if path.startswith(("data/measurements/", "data/flyby/", "data/gallium/")):
        return "EXTERNAL_OR_COMPARISON_DATA"
    return "CURRENT_SIMULATOR_ARTIFACT"


def _walk(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: str) -> dict[str, Any]:
    value = json.loads(
        (REPOSITORY_ROOT / path).read_text(encoding="utf-8"),
        object_pairs_hook=_strict_json_object,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def _actual_status(value: Mapping[str, Any]) -> Any:
    if "status" in value:
        return value["status"]
    if "verdict" in value:
        return value["verdict"]
    return None


def _key_paths(value: Any, target: str, path: str = "$") -> list[str]:
    rows: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == target:
                rows.append(child_path)
            rows.extend(_key_paths(child, target, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_key_paths(child, target, f"{path}[{index}]"))
    return rows


def _schema_absence(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    occurrences = _key_paths(value, key)
    if occurrences:
        raise ValueError(f"schema absence changed for {key}: {occurrences}")
    return {"classification": "ABSENT_FROM_DECLARED_SCHEMA", "key": key, "searched_scope": "entire_canonical_json_object", "occurrences": []}


def _evidence(path: str, value: Mapping[str, Any]) -> dict[str, Any] | None:
    if path == "data/common_reserve/charged_response_artifact.json":
        binding = value.get("carrier_binding", {})
        response = value.get("source_response", {})
        lift = value.get("derived", {}).get("current_lift_status", {})
        return {
            "emitted_support_size": len(binding.get("port_order", [])),
            "emitted_source_response_operator": response.get("operator"),
            "emitted_source_bound_impulse_readback": response.get("physical_perturb_readback_source_bound"),
            "emitted_current_lift_source_selected": lift.get("source_selected"),
            "spatial_translation_binding": _schema_absence(value, "spatial_translation_identification"),
            "same_operator_physical_readout": _schema_absence(value, "same_operator_physical_readout"),
        }
    if path == "data/common_reserve/producer_capability_matrix.json":
        probe = value.get("raw_twelve_port_response_probe", {})
        return {
            "emitted_finite_simulator_response_identified": probe.get("finite_simulator_response_identified"),
            "emitted_physical_A_T_identification": probe.get("physical_A_T_identification"),
            "emitted_current_lift_source_selected": probe.get("current_lift_source_selected"),
            "emitted_scientific_promotion_allowed": value.get("scientific_promotion_allowed"),
        }
    if path == "data/a2_holonomy/ordered_port_response_diagnostic_receipt.json":
        receipts = value.get("receipts", {})
        response = value.get("propagation_adjoined_response", {})
        interpretation = value.get("scientific_interpretation", {})
        return {
            "emitted_port_count": value.get("source_projection", {}).get("port_count"),
            "emitted_propagation_generator": value.get("source_projection", {}).get("propagation_generator"),
            "emitted_generated_algebra_type": response.get("generated_algebra_type"),
            "emitted_generated_algebra_real_rank": response.get("generated_algebra_real_rank"),
            "emitted_derived_algebra_type": response.get("derived_algebra_type"),
            "emitted_derived_algebra_real_rank": response.get("derived_algebra_real_rank"),
            "emitted_A1_complete_response_receipt": receipts.get("A1_COMPLETE_TWELVE_DIMENSIONAL_RESPONSE_RECEIPT"),
            "emitted_A2_same_current_receipt": receipts.get("A2_SAME_CURRENT_HOLONOMY_RECEIPT"),
            "emitted_physical_current_source_bridge_receipt": receipts.get("PHYSICAL_CURRENT_SOURCE_BRIDGE_RECEIPT"),
            "emitted_u12_is_candidate_oph_current": interpretation.get("u12_is_candidate_oph_current"),
        }
    if path == "data/repair_closure/bounded_atomic_self_readback_closure_receipt.json":
        return {
            "emitted_finite_internal_seam_torsor": value.get("FINITE_DIRECTED_SEAM_TORSOR_RECEIPT"),
            "emitted_physical_repair_law": value.get("PHYSICAL_REPAIR_LAW_RECEIPT"),
            "emitted_full_universe_closure": value.get("FULL_SELF_READING_UNIVERSE_CLOSURE_RECEIPT"),
        }
    if path == "data/local_domain/stage3_receipt.json":
        return {
            "operator_domain": "observer_visible_local_seam_complex",
            "emitted_visible_edge_count": value.get("covariant_derivative_typing", {}).get("domain_edge_count"),
            "emitted_physical_promotion_allowed": value.get("physical_promotion_allowed"),
            "vertex12_identity_bridge": _schema_absence(value, "rer_exact_flux_12_42_vertex_identity_bridge"),
        }
    if path == "data/local_domain/source_gap_receipt.json":
        hamiltonian = value.get("hamiltonian", {})
        return {
            "emitted_operator": hamiltonian.get("operator"),
            "emitted_carrier_count": hamiltonian.get("carrier_count"),
            "emitted_physical_promotion_allowed": value.get("physical_promotion_allowed"),
            "physical_reference_transition": _schema_absence(value, "physical_reference_transition_selected"),
        }
    if path == "data/local_domain/defect_sector_receipt.json":
        identity = value.get("spectral_interface_identity", {})
        return {
            "operator_domain": "finite_local_domain_twist_sectors",
            "emitted_separate_from_rer_exact_flux_certificate": identity.get("separate_from_rer_exact_flux_certificate"),
            "emitted_rer_exact_flux_vertex_identity_bridge": identity.get("rer_exact_flux_12_42_vertex_identity_bridge"),
            "emitted_physical_promotion_allowed": value.get("physical_promotion_allowed"),
        }
    if path == "data/local_domain/matter_attachment_receipt.json":
        matter = value.get("matter_operator_certificate", {})
        spin = value.get("spin_layer", {})
        return {
            "emitted_matter_operator_source_selected": matter.get("source_selected"),
            "emitted_spin_same_source_domain": spin.get("same_source_domain_certified"),
            "emitted_spin_to_local_domain_bridge": spin.get("spin_to_local_domain_bridge_certified"),
            "emitted_physical_promotion_allowed": value.get("physical_promotion_allowed"),
        }
    if path == "data/local_domain/classical_realization_receipt.json":
        identity = value.get("spectral_interface_identity", {})
        return {
            "operator_domain": "finite_local_domain_classical_harmonic_network",
            "emitted_rer_exact_flux_vertex_identity_bridge": identity.get("rer_exact_flux_12_42_vertex_identity_bridge"),
            "emitted_physical_promotion_allowed": value.get("physical_promotion_allowed"),
        }
    if path == "data/local_domain/clock_unit_verdict.json":
        return {"emitted_verdict": value.get("verdict"), "emitted_physical_promotion_allowed": value.get("physical_promotion_allowed")}
    if path == "data/repair_closure/seam_equalizer_current_control_report.json":
        return {
            "operator_domain": "scalar_twelve_port_seam_equalizers",
            "emitted_desired_current_identification": value.get("current_identification_control", {}).get("repair_equalizers_are_the_desired_12d_compact_current"),
            "emitted_negative_control_status": value.get("status"),
        }
    if path == "data/repair_closure/vertex12_atomic_port_transfer_receipt.json":
        operator = value.get("atomic_transfer_operator", {})
        readback = value.get("post_repair_in_process_snapshot_reread", {})
        boundary = value.get("quotient_and_spatial_boundary", {})
        quotient = boundary.get("quotient_enumeration", {})
        candidate = value.get("candidate_next_typed_source_object", {})
        return {
            "operator_domain": operator.get("domain"),
            "emitted_source_native_internal_seam_partner_operator": operator.get(
                "source_native_internal_seam_partner_operator_receipt"
            ),
            "emitted_exact_symbolic_matching_and_projector_algebra": operator.get(
                "exact_symbolic_matching_and_projector_algebra"
            ),
            "emitted_source_native_spatial_translation": operator.get(
                "source_native_spatial_translation_receipt"
            ),
            "emitted_in_process_snapshot_reread_carrier_count": readback.get(
                "covered_carrier_count"
            ),
            "emitted_readback_mechanism": readback.get("readback_mechanism"),
            "emitted_independent_persistence_readback": readback.get(
                "independent_persistence_readback"
            ),
            "emitted_independent_second_producer_readback": readback.get(
                "independent_second_producer_readback"
            ),
            "emitted_physical_sector_readout": readback.get(
                "physical_sector_readout"
            ),
            "emitted_noncollapsed_inverse_compatible_quotient_count": quotient.get(
                "noncollapsed_antipodal_inverse_compatible_quotient_count"
            ),
            "emitted_same_operator_physical_readout": boundary.get(
                "same_operator_physical_readout_receipt"
            ),
            "emitted_current_fixed_matching_family_has_no_qualifying_carrier_set_quotient": candidate.get(
                "current_fixed_matching_family_has_no_qualifying_carrier_set_quotient"
            ),
        }
    if path == "data/a2_holonomy/a2_holonomy_current_selector_report.json":
        return {
            "emitted_status": value.get("status"),
            "emitted_source_current_receipt": value.get("source_current_receipt"),
            "spatial_translation_binding": _schema_absence(value, "spatial_translation_identification"),
        }
    if path == "data/repair_closure/angular_refinement_repair_observability_receipt.json":
        decision = value.get("selection_decision", {})
        return {
            "emitted_physical_sky_readout_selected": decision.get("physical_sky_readout_selected"),
            "emitted_physical_angular_prediction": decision.get("physical_angular_prediction"),
            "emitted_repair_schedule_source_selected": decision.get("repair_schedule_source_selected"),
        }
    return None


def _current_json_paths(paths: Sequence[str]) -> set[str]:
    return {path for path in paths if path.endswith(".json") and not path.startswith(NONCURRENT_PREFIXES)}


def _rows(paths: Sequence[str]) -> list[dict[str, Any]]:
    current = _current_json_paths(paths)
    if current != set(EXPECTED_CONTRACTS):
        raise ValueError("current canonical path contract drift")
    result: list[dict[str, Any]] = []
    for path in sorted(current):
        contract = EXPECTED_CONTRACTS[path]
        if path in DECLARED_OUTPUT_PATHS:
            result.append({
                "path": path,
                "schema": contract["schema"],
                "status": contract["status"],
                "disposition": contract["disposition"],
                "semantic_scan_excluded_as_recursive_output": True,
                "critical_bridge_evidence": None,
            })
            continue
        value = _load(path)
        schema = value.get("schema")
        status = _actual_status(value)
        if schema != contract["schema"] or status != contract["status"]:
            raise ValueError(f"schema/status drift for {path}")
        if path == "data/repair_closure/vertex12_atomic_port_transfer_receipt.json":
            verification = (
                verify_vertex12_atomic_port_transfer_independent.verify_report(value)
            )
            if verification.get("receipt") is not True:
                raise ValueError(
                    "vertex12 packet failed independent verification: "
                    f"{verification.get('reasons')}"
                )
        result.append({
            "path": path,
            "schema": schema,
            "status": status,
            "raw_pin": _raw_pin(REPOSITORY_ROOT / path),
            "disposition": contract["disposition"],
            "semantic_scan_excluded_as_recursive_output": False,
            "critical_bridge_evidence": _evidence(path, value),
        })
    return result


def _scan(paths: Sequence[str]) -> dict[str, Any]:
    scanned = sorted(_current_json_paths(paths) - DECLARED_OUTPUT_PATHS)
    packet_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    for path in scanned:
        packet_count = 0
        signals: set[str] = set()
        for obj in _walk(_load(path)):
            if obj.get("schema") == SOURCE_PACKET_SCHEMA:
                packet_count += 1
            signals.update(key for key in POSITIVE_SIGNAL_KEYS if obj.get(key) is True)
        if packet_count:
            packet_rows.append({"path": path, "packet_count": packet_count})
        if signals:
            signal_rows.append({"path": path, "true_signal_keys": sorted(signals)})
    return {
        "current_canonical_json_path_count_excluding_recursive_outputs": len(scanned),
        "current_canonical_json_path_list_sha256_excluding_recursive_outputs": _sha(scanned),
        "recursive_output_paths_excluded": sorted(DECLARED_OUTPUT_PATHS),
        "registered_source_packet_rows_excluding_recursive_outputs": packet_rows,
        "positive_promotion_signal_rows_excluding_recursive_outputs": signal_rows,
    }


def _noncurrent(paths: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in ("LEGACY_EARNED_RUN", "IMPORTED_NONNATIVE", "EXTERNAL_OR_COMPARISON_DATA"):
        selected = sorted(path for path in paths if _provenance(path) == label)
        result[label] = {"path_count": len(selected), "path_list_sha256": _sha(selected), "semantic_payloads_scanned": False}
    return result


def _expected_payload() -> dict[str, Any]:
    index_rows = _git_index_rows()
    paths = [row["path"] for row in index_rows]
    if _untracked_data_paths():
        raise ValueError("untracked data paths")
    if _unstaged_current_inputs():
        raise ValueError("unstaged current canonical inputs")
    scan = _scan(paths)
    if scan["registered_source_packet_rows_excluding_recursive_outputs"] or scan["positive_promotion_signal_rows_excluding_recursive_outputs"]:
        raise ValueError("registered packet or positive signal requires review")
    content_rows = [row for row in index_rows if row["path"] not in DECLARED_OUTPUT_PATHS]
    provenance_counts = Counter(_provenance(path) for path in paths)
    return {
        "schema": SCHEMA,
        "issue": 655,
        "status": STATUS,
        "scope": (
            "Git-indexed tracked paths under data; semantic scan of current canonical "
            "simulator JSON objects excluding the recursive inventory and parent bridge "
            "outputs; legacy, imported, and external/comparison paths counted only"
        ),
        "tracked_serialized_data_catalog": {
            "path_count_including_declared_recursive_outputs": len(paths),
            "path_list_sha256_including_declared_recursive_outputs": _sha(paths),
            "content_index_row_count_excluding_recursive_outputs": len(content_rows),
            "content_index_sha256_excluding_recursive_outputs": _sha(content_rows),
            "provenance_counts": dict(sorted(provenance_counts.items())),
            "untracked_data_paths_excluding_declared_recursive_outputs": [],
            "unstaged_current_canonical_inputs": [],
        },
        "noncurrent_path_catalog": _noncurrent(paths),
        "current_canonical_json_contract_scan": scan,
        "canonical_artifact_rows": _rows(paths),
        "bridge_admission_contract": {
            "policy": (
                "Only an independently verified packet with the registered schema may "
                "promote the issue-655 translation/readout bridge on this serialized-data surface."
            ),
            "registered_packet_schema": SOURCE_PACKET_SCHEMA,
            "required_chain": [
                "source-history-replayed complete vertex12 translation operator",
                "digest-identical physical scalar or polarization-independent readout",
                "coherent frame transport and declared boost custody",
            ],
            "registered_packet_count_excluding_recursive_outputs": 0,
            "true_promotion_signal_path_count_excluding_recursive_outputs": 0,
            "accepted_bridge_count_excluding_recursive_outputs": 0,
            "recursive_parent_bridge_receipt_exclusion": {
                "path": BRIDGE_RELATIVE_PATH,
                "reason": "parent output embeds the current negative source packet and is excluded to avoid recursive custody",
                "packet_count_included_in_scan": False,
            },
        },
        "implementation_pins": {
            "producer": _raw_pin(PRODUCER_PATH),
            "independent_verifier": _raw_pin(INDEPENDENT_VERIFIER_PATH),
            "mutation_tests": _raw_pin(TEST_PATH),
        },
        "epistemic_boundary": {
            "local_spatial_or_kinetic_operators_exist": True,
            "twelve_port_internal_seam_response_and_in_process_snapshot_reread_exist": True,
            "claim_that_no_spatial_operator_exists": False,
            "registered_accepted_same_domain_chain_on_scanned_surface_exists": False,
            "unregistered_equivalent_semantics_ruled_out": False,
            "producer_code_or_sibling_repository_absence_claimed": False,
            "physical_prediction_unsealed": False,
            "reopen_condition": (
                "Register and independently verify a source packet binding one complete "
                "twelve-port translation operator to a digest-identical physical readout "
                "with frame and boost custody."
            ),
        },
    }


def verify(report: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        received = copy.deepcopy(dict(report))
        digest = received.pop("receipt_sha256", None)
        if digest != _sha(received):
            reasons.append("receipt_sha256_mismatch")
        expected = _expected_payload()
        for key in sorted(set(received) | set(expected)):
            if _canonical_bytes(received.get(key)) != _canonical_bytes(
                expected.get(key)
            ):
                reasons.append(f"top_level_mismatch:{key}")
    except (AttributeError, OSError, subprocess.SubprocessError, TypeError, ValueError, json.JSONDecodeError):
        reasons.append("malformed_or_unreplayable_inventory")
    passed = not reasons
    return {
        "schema": VERIFICATION_SCHEMA,
        "receipt": passed,
        "status": "PASS" if passed else "FAIL",
        "reasons": sorted(set(reasons)),
        "independent_implementation": True,
        "imports_inventory_producer": False,
        "verified_exit": STATUS if passed else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = verify(report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["receipt"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
