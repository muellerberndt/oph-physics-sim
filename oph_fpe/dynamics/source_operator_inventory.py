"""Tracked serialized-data census for the issue-655 bridge.

The FZ-11 branch needs a registered packet that binds a source-history-replayed
translation operator on the complete twelve-port vertex orbit to a physical
scalar or polarization-independent readout of that exact operator.  This
module checks the narrower custody question needed to keep that branch sealed.

The path census is the Git index under ``data/``.  Semantic scanning is limited
to current canonical simulator JSON objects and excludes the inventory itself
and its parent bridge receipt to avoid recursion.  Legacy earned runs, imported
mirrors, and external/comparison data are counted and hash-bound by path only;
their very large or LFS-backed JSON payloads are not parsed here.  Producer
code, configs, tests, documents, sibling repositories, and unregistered
equivalent semantics are outside this receipt.
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

from oph_fpe.cosmology import verify_a5_biposh_refinement_independent
from oph_fpe.dynamics import verify_vertex12_atomic_port_transfer_independent
from oph_fpe.dynamics import (
    verify_vertex12_directed_transport_feasibility_independent,
)


SCHEMA = "oph.source_operator_ancestry_inventory.v1"
VERIFICATION_SCHEMA = "oph.source_operator_ancestry_inventory_verification.v1"
SOURCE_PACKET_SCHEMA = "oph.port_repair_propagation_source_packet.v1"
STATUS = "NO_REGISTERED_ACCEPTED_VERTEX12_BRIDGE_PACKET_ON_TRACKED_SERIALIZED_DATA_SURFACE"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPOSITORY_ROOT / "data/repair_closure/source_operator_ancestry_inventory.json"
BRIDGE_RECEIPT_PATH = REPOSITORY_ROOT / "data/repair_closure/port_repair_propagation_bridge_receipt.json"
PRODUCER_PATH = Path(__file__).resolve()
INDEPENDENT_VERIFIER_PATH = REPOSITORY_ROOT / "oph_fpe/dynamics/verify_source_operator_inventory_independent.py"
TEST_PATH = REPOSITORY_ROOT / "tests/test_source_operator_inventory.py"

INVENTORY_RELATIVE_PATH = OUTPUT_PATH.relative_to(REPOSITORY_ROOT).as_posix()
BRIDGE_RELATIVE_PATH = BRIDGE_RECEIPT_PATH.relative_to(REPOSITORY_ROOT).as_posix()
DECLARED_OUTPUT_PATHS = {INVENTORY_RELATIVE_PATH, BRIDGE_RELATIVE_PATH}

NONCURRENT_PREFIXES = (
    "data/earned_runs/",
    "data/oph_cross_repo_current/",
    "data/measurements/",
    "data/flyby/",
    "data/gallium/",
)


def _contract(schema: str | None, status: str | None, disposition: str) -> dict[str, Any]:
    return {"schema": schema, "status": status, "disposition": disposition}


# This is a path-, schema-, status-, and disposition-closed catalog.  Any
# current canonical JSON addition or semantic schema/status change stops the
# producer until it receives an explicit review.
CANONICAL_CONTRACTS: dict[str, dict[str, Any]] = {
    "data/a2_holonomy/a2_holonomy_current_selector_report.json": _contract(
        "oph.a2-holonomy-current-selector/1.0.0",
        "OPEN_SOURCE_HOLONOMY_BRIDGE",
        "GAUGE_CURRENT_HOLONOMY_OPEN_NOT_SPATIAL_PROPAGATION",
    ),
    "data/a2_holonomy/ordered_port_response_diagnostic_receipt.json": _contract(
        "oph.ordered-port-response-diagnostic.v1",
        "ATTAINED_BOUNDED_NEGATIVE_CONTROL",
        "TWELVE_PORT_ADJACENCY_PROPAGATION_OVERSHOOTS_TO_U12__PHYSICAL_CURRENT_SOURCE_OPEN",
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
        "oph.common-reserve.capability-matrix.v1",
        "CAPABILITY_PROBE_COMPLETE__SCIENTIFIC_PROMOTION_DISABLED",
        "RAW_RESPONSE_NATIVE_PHYSICAL_AT_BRIDGE_OPEN",
    ),
    "data/einstein_convergence/manifest.json": _contract(
        "oph.einstein-convergence-ladder.v2", None, "EINSTEIN_LADDER_MANIFEST_NOT_VERTEX12_OPERATOR"
    ),
    "data/einstein_convergence/rung_16384.json": _contract(None, None, "EINSTEIN_LADDER_RUNG_NOT_VERTEX12_OPERATOR"),
    "data/einstein_convergence/rung_65536.json": _contract(None, None, "EINSTEIN_LADDER_RUNG_NOT_VERTEX12_OPERATOR"),
    "data/einstein_convergence/rung_262144.json": _contract(None, None, "EINSTEIN_LADDER_RUNG_NOT_VERTEX12_OPERATOR"),
    "data/einstein_convergence/rung_262144_dense.json": _contract(None, None, "EINSTEIN_LADDER_RUNG_NOT_VERTEX12_OPERATOR"),
    "data/local_domain/classical_realization_receipt.json": _contract(
        "oph.local-domain-classical-realization.v1",
        "CLASSICAL_REALIZATION_MATCHES_DECLARED_FINITE_SPECTRAL_INTERFACE",
        "LOCAL_DOMAIN_CLASSICAL_OPERATOR_NO_VERTEX12_IDENTITY_BRIDGE",
    ),
    "data/local_domain/clock_unit_verdict.json": _contract(
        "oph.local-domain-clock-unit-verdict.v1",
        "PHYSICAL_UNITS_NOT_EVALUABLE_ON_DECLARED_SERIALIZED_INTERFACE",
        "LOCAL_DOMAIN_PHYSICAL_UNITS_AND_READOUT_NOT_EVALUABLE",
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
        "oph.icosahedral_chsh_candidate.v1",
        "EXACT_PROJECTIVE_BRANCH_CANDIDATE__TWO_WING_COMPLETED_RECORD_SOURCE_PRODUCER_MISSING",
        "PROJECTIVE_QUANTUM_CANDIDATE_COMPLETED_RECORD_PRODUCER_MISSING",
    ),
    "data/refinement/a5_biposh_dual_operator_coefficients.json": _contract(
        "oph.a5-biposh-dual-operator-coefficients.v1",
        None,
        "FULL_BIPOSH_COEFFICIENT_BUNDLE__SUPPORTING_OPERATOR_FINGERPRINT_ONLY_NOT_PHYSICAL",
    ),
    "data/refinement/a5_biposh_dual_operator_receipt.json": _contract(
        "oph.a5-biposh-dual-operator-refinement.v1",
        "FINITE_DUAL_OPERATOR_FINGERPRINT_ATTAINED__CONTINUUM_RESIDUAL_AND_PHYSICAL_COVARIANCE_OPEN",
        "FINITE_DUAL_OPERATOR_FINGERPRINT__OPERATOR_SELECTION_CONTINUUM_AND_PHYSICAL_COVARIANCE_OPEN",
    ),
    "data/refinement/physical_birefinement_preflight.json": _contract(
        "oph.refinement.physical-birefinement-preflight.v1", "SOURCE_PRODUCER_MISSING", "PHYSICAL_BIREFINEMENT_SOURCE_PRODUCER_MISSING"
    ),
    "data/repair_closure/angular_refinement_repair_observability_receipt.json": _contract(
        "oph.angular_refinement_repair_observability.v1",
        "EXACT_REFINEMENT_REPAIR_COUNTERENSEMBLE__DETAIL_COVARIANCE_UNSELECTED__PHYSICAL_SKY_READOUT_OPEN",
        "ANGULAR_READOUT_PHYSICAL_SKY_BINDING_OPEN",
    ),
    "data/repair_closure/bounded_atomic_self_readback_closure_receipt.json": _contract(
        "oph.bounded_atomic_self_readback_closure.v1",
        "BOUNDED_EXPECTATION_LEVEL_REPAIR_FIXED_POINT_UNIQUE_MODULO_CLOCK_IN_THE_FROZEN_ADVERSARIAL_SUITE",
        "INTERNAL_REPAIR_PHYSICAL_LAW_NOT_SELECTED",
    ),
    BRIDGE_RELATIVE_PATH: _contract(
        "oph.port_repair_propagation_bridge_receipt.v1",
        "BOUNDED_NONSELECTION__FZ11_REMAINS_BRANCH_PREDICTION",
        "RECURSIVE_PARENT_BRIDGE_RECEIPT_EXCLUDED_FROM_SEMANTIC_SCAN",
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
    "data/repair_closure/vertex12_directed_transport_feasibility_receipt.json": _contract(
        "oph.vertex12-directed-transport-feasibility.v1",
        "EXACT_SEMICONJUGATE_COVER_OBSTRUCTION_FOR_CURRENT_SOURCE_MATCHINGS__ORIENTED_SOURCE_TRANSITION_LAW_OPEN",
        "CURRENT_MATCHING_COVER_OBSTRUCTED__DECLARED_ALGEBRAIC_CONTROL_NOT_SOURCE_EMITTED_OR_PHYSICAL",
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

BIPOSH_COEFFICIENT_FORBIDDEN_CLAIM_FIELDS = (
    "status",
    "physical_prediction",
    "physical_covariance_selected",
    "promotion_allowed",
    "continuum_residual_decided",
    "source_selected",
    "physical_promotion_allowed",
    "scientific_promotion_allowed",
    "physical_repair_law_selected",
    "screen_to_sky_readout_selected",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


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
            raise ValueError("unmerged data path in source inventory")
        path = raw_path.decode("utf-8")
        rows[path] = {"path": path, "mode": mode, "object_id": object_id}
    for path in DECLARED_OUTPUT_PATHS:
        rows.setdefault(path, {"path": path, "mode": "DECLARED_OUTPUT", "object_id": "EXCLUDED"})
    return [rows[path] for path in sorted(rows)]


def _untracked_data_paths() -> list[str]:
    paths = [p.decode("utf-8") for p in _run_git("ls-files", "--others", "--exclude-standard", "-z", "--", "data").split(b"\0") if p]
    return sorted(path for path in paths if path not in DECLARED_OUTPUT_PATHS)


def _unstaged_current_inputs() -> list[str]:
    inputs = sorted(set(CANONICAL_CONTRACTS) - DECLARED_OUTPUT_PATHS)
    raw = _run_git("diff", "--name-only", "-z", "--", *inputs)
    return sorted(p.decode("utf-8") for p in raw.split(b"\0") if p)


def _provenance_class(path: str) -> str:
    if path.startswith("data/earned_runs/"):
        return "LEGACY_EARNED_RUN"
    if path.startswith("data/oph_cross_repo_current/"):
        return "IMPORTED_NONNATIVE"
    if path.startswith(("data/measurements/", "data/flyby/", "data/gallium/")):
        return "EXTERNAL_OR_COMPARISON_DATA"
    return "CURRENT_SIMULATOR_ARTIFACT"


def _walk_objects(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _walk_objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_objects(item)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: str) -> dict[str, Any]:
    value = json.loads(
        (REPOSITORY_ROOT / path).read_text(encoding="utf-8"),
        object_pairs_hook=_strict_json_object,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
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
        raise ValueError(f"schema absence contract changed for {key}: {occurrences}")
    return {
        "classification": "ABSENT_FROM_DECLARED_SCHEMA",
        "key": key,
        "searched_scope": "entire_canonical_json_object",
        "occurrences": [],
    }


def _critical_evidence(path: str, value: Mapping[str, Any]) -> dict[str, Any] | None:
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
    if path == "data/repair_closure/vertex12_directed_transport_feasibility_receipt.json":
        obstruction = value.get("exact_semiconjugacy_obstruction", {})
        control = value.get("algebraic_transport_positive_control", {})
        requested = value.get("requested_directed_transport_ledger", {})
        return {
            "emitted_antipodal_pair_count": obstruction.get(
                "antipodal_pair_count"
            ),
            "emitted_semiconjugate_cover_can_satisfy_inverse_law": obstruction.get(
                "semiconjugate_noncollapsed_site_cover_can_satisfy_inverse_law"
            ),
            "emitted_algebraic_control_site_domain": control.get("site_domain"),
            "emitted_algebraic_control_site_count": control.get("site_count"),
            "emitted_algebraic_control_A5_order": control.get(
                "proper_A5_port_action_order"
            ),
            "emitted_algebraic_control_inverse_and_covariance": bool(
                control.get("all_six_antipodal_pairs_are_exact_inverses") is True
                and control.get("exact_covariance_attained") is True
            ),
            "emitted_control_source_transition_event": control.get(
                "source_transition_event_emitted"
            ),
            "emitted_control_repair_generated": control.get("repair_generated"),
            "emitted_control_source_selected_site_completion": control.get(
                "source_selected_site_completion"
            ),
            "emitted_control_spatial_translation": control.get(
                "spatial_translation"
            ),
            "emitted_control_physical_readout": control.get("physical_readout"),
            "emitted_control_physical_prediction": control.get(
                "physical_prediction"
            ),
            "emitted_requested_source_transport_ledger": requested.get(
                "attained_from_current_source_emissions"
            ),
            "emitted_requested_twelve_directed_maps": requested.get(
                "twelve_event_emitted_directed_maps_attained"
            ),
            "emitted_requested_antipodal_inverse": requested.get(
                "exact_T_antipode_p_equals_inverse_T_p_attained"
            ),
            "emitted_requested_A5_covariance": requested.get(
                "site_A5_action_and_exact_covariance_attained"
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
    if path == "data/refinement/a5_biposh_dual_operator_coefficients.json":
        forbidden_claim_fields = sorted(
            key
            for key in BIPOSH_COEFFICIENT_FORBIDDEN_CLAIM_FIELDS
            if key in value
        )
        if forbidden_claim_fields:
            raise ValueError(
                "BipoSH coefficient bundle gained forbidden claim fields: "
                f"{forbidden_claim_fields}"
            )
        return {
            "emitted_coefficient_kind": value.get("coefficient_kind"),
            "emitted_case_count": value.get("case_count"),
            "emitted_coefficient_count_per_case": value.get(
                "coefficient_count_per_case"
            ),
            "forbidden_top_level_claim_fields_checked": list(
                BIPOSH_COEFFICIENT_FORBIDDEN_CLAIM_FIELDS
            ),
            "forbidden_top_level_claim_fields_present": forbidden_claim_fields,
            "emitted_supporting_bundle_has_separate_status": (
                "status" in value
            ),
            "emitted_physical_promotion": bool(forbidden_claim_fields),
        }
    if path == "data/refinement/a5_biposh_dual_operator_receipt.json":
        bridge = value.get("bounded_repair_generator_bridge", {})
        decision = value.get("selection_decision", {})
        source_scope = value.get("source_scope", {})
        return {
            "emitted_base_628_operator_match": bridge.get(
                "base_carrier_operator_matches_bounded_reconstructed_one_atom_mean_generator_up_to_scale"
            ),
            "emitted_base_labelled_face_presentation_matches_parent": bridge.get(
                "base_carrier_labelled_face_presentation_matches_parent"
            ),
            "emitted_base_edge_set_matches_face_presentation": bridge.get(
                "base_carrier_edge_set_matches_face_presentation"
            ),
            "emitted_base_equal_seam_operator_bounded_reconstructed": decision.get(
                "base_equal_seam_operator_bounded_reconstructed"
            ),
            "emitted_continuum_residual_decided": decision.get(
                "continuum_residual_decided"
            ),
            "emitted_equal_seam_operator_source_selected": decision.get(
                "equal_seam_operator_source_selected"
            ),
            "emitted_global_frame_quotient_visible": decision.get(
                "global_frame_quotient_visible"
            ),
            "emitted_physical_covariance_selected": decision.get(
                "physical_covariance_selected"
            ),
            "emitted_physical_prediction": decision.get("physical_prediction"),
            "emitted_physical_release_ensemble_selected": decision.get(
                "physical_release_ensemble_selected"
            ),
            "emitted_physical_repair_law_selected": decision.get(
                "physical_repair_law_selected"
            ),
            "emitted_promotion_allowed": decision.get("promotion_allowed"),
            "emitted_refinement_extension_source_selected": decision.get(
                "refinement_tower_equal_seam_extension_source_selected"
            ),
            "emitted_screen_to_sky_readout_selected": decision.get(
                "screen_to_sky_readout_selected"
            ),
            "emitted_comparison_data_used": bool(
                source_scope.get("external_comparison_data_used") is True
                or source_scope.get("sky_data_used") is True
                or source_scope.get("target_values_used") is True
            ),
        }
    return None


def _current_json_paths(paths: Sequence[str]) -> set[str]:
    return {path for path in paths if path.endswith(".json") and not path.startswith(NONCURRENT_PREFIXES)}


def _canonical_rows(paths: Sequence[str]) -> list[dict[str, Any]]:
    current_json = _current_json_paths(paths)
    expected = set(CANONICAL_CONTRACTS)
    if current_json != expected:
        raise ValueError(f"canonical contract drift: missing={sorted(current_json - expected)}, stale={sorted(expected - current_json)}")
    rows: list[dict[str, Any]] = []
    for path in sorted(current_json):
        contract = CANONICAL_CONTRACTS[path]
        if path in DECLARED_OUTPUT_PATHS:
            rows.append({
                "path": path,
                "schema": contract["schema"],
                "status": contract["status"],
                "disposition": contract["disposition"],
                "semantic_scan_excluded_as_recursive_output": True,
                "critical_bridge_evidence": None,
            })
            continue
        value = _load_json(path)
        actual_schema = value.get("schema")
        actual_status = _actual_status(value)
        if actual_schema != contract["schema"] or actual_status != contract["status"]:
            raise ValueError(
                f"canonical schema/status drift for {path}: "
                f"schema={actual_schema!r}, status={actual_status!r}"
            )
        if path == "data/repair_closure/vertex12_atomic_port_transfer_receipt.json":
            verification = (
                verify_vertex12_atomic_port_transfer_independent.verify_report(value)
            )
            if verification.get("receipt") is not True:
                raise ValueError(
                    "vertex12 packet failed independent verification: "
                    f"{verification.get('reasons')}"
                )
        if path == "data/repair_closure/vertex12_directed_transport_feasibility_receipt.json":
            verification = (
                verify_vertex12_directed_transport_feasibility_independent.verify_report(
                    value
                )
            )
            if verification.get("receipt") is not True:
                raise ValueError(
                    "vertex12 directed-transport feasibility packet failed "
                    "independent verification: "
                    f"{verification.get('reasons')}"
                )
        if path == "data/refinement/a5_biposh_dual_operator_receipt.json":
            verification = (
                verify_a5_biposh_refinement_independent.verify_packet()
            )
            if verification.get("status") != "PASS":
                raise ValueError(
                    "A5 BipoSH dual-operator packet failed independent verification"
                )
        rows.append({
            "path": path,
            "schema": actual_schema,
            "status": actual_status,
            "raw_pin": _raw_pin(REPOSITORY_ROOT / path),
            "disposition": contract["disposition"],
            "semantic_scan_excluded_as_recursive_output": False,
            "critical_bridge_evidence": _critical_evidence(path, value),
        })
    return rows


def _scan_current_json(paths: Sequence[str]) -> dict[str, Any]:
    scanned = sorted(_current_json_paths(paths) - DECLARED_OUTPUT_PATHS)
    packet_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    for path in scanned:
        value = _load_json(path)
        packet_count = 0
        signals: set[str] = set()
        for obj in _walk_objects(value):
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


def _noncurrent_catalog(paths: Sequence[str]) -> dict[str, Any]:
    classes = ("LEGACY_EARNED_RUN", "IMPORTED_NONNATIVE", "EXTERNAL_OR_COMPARISON_DATA")
    result: dict[str, Any] = {}
    for label in classes:
        selected = sorted(path for path in paths if _provenance_class(path) == label)
        result[label] = {"path_count": len(selected), "path_list_sha256": _sha(selected), "semantic_payloads_scanned": False}
    return result


def _payload() -> dict[str, Any]:
    index_rows = _git_index_rows()
    paths = [row["path"] for row in index_rows]
    untracked = _untracked_data_paths()
    unstaged = _unstaged_current_inputs()
    if untracked:
        raise ValueError(f"untracked data paths require staging or explicit recursive-output declaration: {untracked}")
    if unstaged:
        raise ValueError(f"unstaged current canonical inputs: {unstaged}")
    scan = _scan_current_json(paths)
    if scan["registered_source_packet_rows_excluding_recursive_outputs"]:
        raise ValueError("a registered source packet requires explicit bridge review")
    if scan["positive_promotion_signal_rows_excluding_recursive_outputs"]:
        raise ValueError("a positive bridge signal requires explicit bridge review")
    content_rows = [row for row in index_rows if row["path"] not in DECLARED_OUTPUT_PATHS]
    provenance_counts = Counter(_provenance_class(path) for path in paths)
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
        "noncurrent_path_catalog": _noncurrent_catalog(paths),
        "current_canonical_json_contract_scan": scan,
        "canonical_artifact_rows": _canonical_rows(paths),
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


def build_inventory() -> dict[str, Any]:
    payload = _payload()
    report = copy.deepcopy(payload)
    report["receipt_sha256"] = _sha(payload)
    return report


def verify_inventory(report: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        received = copy.deepcopy(dict(report))
        digest = received.pop("receipt_sha256", None)
        if digest != _sha(received):
            reasons.append("receipt_sha256_mismatch")
        if _canonical_bytes(dict(report)) != _canonical_bytes(build_inventory()):
            reasons.append("producer_replay_mismatch")
    except (OSError, subprocess.SubprocessError, TypeError, ValueError, json.JSONDecodeError):
        reasons.append("malformed_or_unreplayable_inventory")
    return {
        "schema": VERIFICATION_SCHEMA,
        "receipt": not reasons,
        "status": "PASS" if not reasons else "FAIL",
        "reasons": sorted(set(reasons)),
        "producer_replay": True,
        "independent_implementation": False,
    }


def write_inventory(path: Path = OUTPUT_PATH) -> dict[str, Any]:
    report = build_inventory()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        report = json.loads(args.verify.read_text(encoding="utf-8"))
        result = verify_inventory(report)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["receipt"] else 1
    report = write_inventory(args.output)
    print(report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
