"""Fail-closed preflight for a physical H3/KMS emergence campaign.

This module is deliberately separate from the numerical H3 fit.  A good fit is
not permission to promote a run: the campaign must first show that its source
objects, clock comparison, controls, event semantics, and frozen ladder match
the paper-side contract.  Missing or malformed evidence is always a blocker.

The preferred input is a mapping with two keys::

    {"config": {...}, "reports": {"refinement": {...}, ...}}

``physical_h3_kms_preflight_report`` also accepts a run directory.  In that
case it reads ``config.yml`` (or its JSON/YAML alternatives) and known report
file names.  Legacy report aliases are loaded only so that the resulting
blockers are diagnostic; legacy pass flags cannot satisfy the primitive
checks below.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from oph_fpe.claims import PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT
from oph_fpe.bulk.bw_native_preflight import native_bw01_bw08_report

SCHEMA_VERSION = "oph_physical_h3_kms_preflight_v3"
DEFAULT_INSTRUMENT_VERSION = "physical-h3-kms-v2"
REQUIRED_CLOCK_LABELS = ("1x", "pi", "2pi", "4pi")
REQUIRED_GEOMETRY_MODELS = ("H3", "S2", "E3", "E4")
REQUIRED_RUNGS = (4_096, 16_384, 65_536, 262_144)
REQUIRED_CARRIER_COUNTS = REQUIRED_RUNGS
MINIMUM_INDEPENDENT_SOURCE_SEEDS = 3
REQUIRED_FAMILY_CONTRACT_KEYS = {
    "instrument_commit",
    "container_digest",
    "schema_versions",
    "source_protocol",
    "feature_contract",
    "model_families",
    "loss_functions",
    "thresholds",
    "control_set",
    "split_algorithm",
    "seed_derivation",
    "rung_scaling_laws",
    "archive_boundary",
    "retirement_rule",
}


class CellStatus(str, Enum):
    """Exact campaign-cell status; invalidity is never a scientific failure."""

    NOT_EVALUATED = "NOT_EVALUATED"
    INSTRUMENT_INVALID = "INSTRUMENT_INVALID"
    INCOMPLETE = "INCOMPLETE"
    VALID_FAIL = "VALID_FAIL"
    VALID_PASS = "VALID_PASS"

_REPORT_FILE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "source_observer": (
        "physical_source_observer_contract_report.json",
        "source_dynamics_repair_record_observer_report.json",
        "theorem_core_receipts.json",
    ),
    "refinement": (
        "physical_h3_kms_refinement_report.json",
        "reference_tower_report.json",
        "strict_neutral_bulk_frontier_report.json",
    ),
    "prime_geometric_state": (
        "prime_geometric_cap_state_report.json",
        "maxent_cap_state_report.json",
        "bw_state_derived_report.json",
    ),
    "independent_geometry": (
        "physical_h3_kms_independent_geometry_report.json",
        "cap_geometry_report.json",
        "bw_state_derived_report.json",
    ),
    "native_bw": (
        "physical_h3_kms_native_bw_payload.json",
        "bw_native_payload.json",
    ),
    "candidate_interventions": (
        "physical_h3_kms_candidate_interventions_report.json",
        "modular_response_kernel_cache_report.json",
        "modular_response_h3_report.json",
    ),
    "geometry_controls": (
        "physical_h3_kms_geometry_controls_report.json",
        "modular_response_h3_report.json",
    ),
    "semantic_event": (
        "semantic_event_reconstruction_report.json",
        "observer_modular_experience_report.json",
        "conformal_h3_spatial_chart_report.json",
    ),
    "campaign": (
        "physical_h3_kms_campaign_manifest.json",
        "campaign_manifest.json",
        "receipt_ladder_report.json",
    ),
}

_ALLOWED_REFINEMENT_FAMILIES = {
    "nested_geodesic_icosahedral",
    "geodesic_icosahedral_refinement",
    "nested_icosahedral_cap_net",
}
_ALLOWED_GEOMETRY_DERIVATIONS = {
    "ordered_bw_frame_cross_ratio",
    "cap_incidence_ordered_frame_cross_ratio",
    "source_ordered_frame_cross_ratio",
}
_FORBIDDEN_CAP_STATE_TOKENS = (
    "record",
    "history",
    "pointer",
    "repair_load",
    "declared",
    "target",
    "kms",
    "2pi",
)
_FORBIDDEN_GEOMETRY_TOKENS = (
    "pi",
    "kms",
    "beta",
    "kappa",
    "temperature",
    "clock_scale",
    "normalization",
    "target",
)


def physical_h3_kms_preflight_report(
    source: Path | str | Mapping[str, Any],
    reports: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit whether a bundle may enter the physical H3/KMS campaign.

    This is a design/evidence firewall, not a successful-emergence receipt.
    It verifies both that the instrument is interpretable and, for completed
    cells, that a claimed H3/2pi outcome came from the frozen target-blind
    comparisons.  Physical promotion and branch-retirement authorization are
    separately derived from exact cell status.
    """

    config, evidence, provenance = _load_bundle(source, reports)
    stages = {
        "P0_source_dynamics_repair_record_observer": _source_observer_stage(
            config, _report(evidence, "source_observer")
        ),
        "P1_nested_refinement_and_expectations": _refinement_stage(
            config, _report(evidence, "refinement")
        ),
        "P2_prime_geometric_cap_state": _prime_geometric_state_stage(
            config, _report(evidence, "prime_geometric_state")
        ),
        "P3_independent_geometric_parameter": _independent_geometry_stage(
            config, _report(evidence, "independent_geometry")
        ),
        "P4_native_bw01_bw08": _native_bw_stage(
            config, _report(evidence, "native_bw")
        ),
        "P5_frozen_candidate_interventions": _candidate_intervention_stage(
            config, _report(evidence, "candidate_interventions")
        ),
        "P6_h3_s2_e3_e4_same_holdout_and_curvature_leverage": _geometry_control_stage(
            config, _report(evidence, "geometry_controls")
        ),
        "P7_semantic_event_e1_e4_and_frame_fiber_separation": _semantic_event_stage(
            config, _report(evidence, "semantic_event")
        ),
        "P8_frozen_multiseed_four_rung_campaign": _campaign_stage(
            config, _report(evidence, "campaign")
        ),
    }
    diagnostic_blockers = [
        f"{stage_name}:{blocker}"
        for stage_name, stage in stages.items()
        for blocker in stage["blockers"]
    ]
    diagnostic_contract_passed = bool(
        stages and all(stage["passed"] for stage in stages.values())
    )
    artifact_admission = _physical_artifact_admission(provenance)
    blockers = [
        *diagnostic_blockers,
        *(f"artifact_admission:{item}" for item in artifact_admission["blockers"]),
    ]
    receipt = bool(
        diagnostic_contract_passed
        and artifact_admission["PHYSICAL_ARTIFACT_REPLAY_ADMISSION_RECEIPT"]
    )
    scientific_failures = [
        f"{stage_name}:{failure}"
        for stage_name, stage in stages.items()
        for failure in _sequence(_mapping(stage.get("evidence")).get("scientific_failures"))
    ]
    hard_blockers = list(
        dict.fromkeys(
            hard_blocker
            for stage in stages.values()
            for hard_blocker in _sequence(stage.get("hard_blockers"))
        )
    )
    campaign_stage = stages["P8_frozen_multiseed_four_rung_campaign"]
    campaign_evidence = _mapping(campaign_stage.get("evidence"))
    reported_campaign_status = str(
        campaign_evidence.get("campaign_status") or CellStatus.INCOMPLETE.value
    )
    instrument_invalidating_blockers = {
        "cap_state_uses_record_or_repair_features",
        "candidate_intervention_hash_mismatch",
        "geometry_control_missing_or_nonfinite",
        "nonnested_regulator_family",
        "frozen_config_family_mismatch",
    }
    if (
        reported_campaign_status == CellStatus.INSTRUMENT_INVALID.value
        or instrument_invalidating_blockers.intersection(hard_blockers)
        or (
            reported_campaign_status == CellStatus.VALID_PASS.value
            and bool(scientific_failures)
        )
    ):
        campaign_status = CellStatus.INSTRUMENT_INVALID.value
    elif not diagnostic_contract_passed and reported_campaign_status in {
        CellStatus.VALID_FAIL.value,
        CellStatus.VALID_PASS.value,
    }:
        campaign_status = CellStatus.INCOMPLETE.value
    else:
        campaign_status = reported_campaign_status
    diagnostic_campaign_status = campaign_status
    if not artifact_admission["PHYSICAL_ARTIFACT_REPLAY_ADMISSION_RECEIPT"]:
        campaign_status = (
            CellStatus.INSTRUMENT_INVALID.value
            if diagnostic_campaign_status == CellStatus.INSTRUMENT_INVALID.value
            else CellStatus.INCOMPLETE.value
        )
    physical_promotion_allowed = bool(
        receipt
        and campaign_status == CellStatus.VALID_PASS.value
        and not scientific_failures
    )
    retirement_counting_allowed = bool(
        receipt
        and campaign_status == CellStatus.VALID_FAIL.value
        and campaign_evidence.get("stable_failure_rule_satisfied") is True
    )
    dependency_graph = _dependency_graph(stages, receipt)
    return {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "mode": "physical_h3_kms_campaign_preflight",
        "instrument_version": campaign_evidence.get("instrument_version")
        or DEFAULT_INSTRUMENT_VERSION,
        "campaign_family_hash": campaign_evidence.get("campaign_family_hash"),
        "campaign_status": campaign_status,
        "reported_campaign_cell_aggregate_status": reported_campaign_status,
        "diagnostic_campaign_status": diagnostic_campaign_status,
        "verdict": "GO" if receipt else "NO_GO",
        PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT: receipt,
        "physical_h3_kms_preflight_receipt": receipt,
        "diagnostic_contract_passed": diagnostic_contract_passed,
        "artifact_replay_admission": artifact_admission,
        "eligible_for_physical_campaign_promotion": physical_promotion_allowed,
        "physical_promotion_allowed": physical_promotion_allowed,
        "retirement_counting_allowed": retirement_counting_allowed,
        "diagnostic_outputs_allowed": True,
        "stages": stages,
        "blockers": blockers,
        "blocker_count": len(blockers),
        "hard_blockers": hard_blockers,
        "scientific_failures": scientific_failures,
        "scientific_outcome": (
            CellStatus.INCOMPLETE.value
            if not receipt
            else CellStatus.VALID_FAIL.value
            if scientific_failures
            else CellStatus.VALID_PASS.value
        ),
        "required_clock_candidates": list(REQUIRED_CLOCK_LABELS),
        "required_geometry_models": list(REQUIRED_GEOMETRY_MODELS),
        "required_rungs": list(REQUIRED_RUNGS),
        "dependency_graph": dependency_graph,
        "upstream_dependencies": dependency_graph["upstream_dependency_ids"],
        "downstream_blockers": dependency_graph["downstream_blockers"],
        "input_provenance": provenance,
        "claim_boundary": (
            "This receipt authorizes only entry into a frozen physical H3/KMS campaign. "
            "Caller-authored mappings and ordinary JSON report directories are diagnostic fixtures, "
            "not physical evidence: promotion and retirement require each source, transaction, observer, "
            "control, and campaign-cell result to be reconstructed by a registered verifier from "
            "content-addressed primitive artifacts. "
            "It is false for a record/history surrogate, a target-bearing clock, candidate-specific "
            "interventions, incomplete controls, an H3-as-event-position shortcut, or an unfrozen "
            "seed/rung ladder. Physical promotion additionally requires exact VALID_PASS status; "
            "retirement counting requires a separately derived stable VALID_FAIL campaign. Invalid "
            "or incomplete cells permit neither. This preflight does not itself establish H3, 2pi, "
            "3+1 spacetime, or a physical law."
        ),
    }


def _physical_artifact_admission(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed until the complete campaign-cell replay chain is integrated.

    The stage functions below are useful contract diagnostics, but their inputs
    are ordinary mappings.  A mapping can repeat a correctly named boolean
    without supplying the primitive artifact from which that boolean should be
    recomputed.  Treating such mappings as campaign evidence would let a caller
    fabricate both a physical pass and a stable branch-retirement failure.

    Registered replay exists for several lower-level artifacts elsewhere in the
    package (federation, repair, and common-source contracts).  The physical
    H3/KMS gate nevertheless remains false until *all* campaign-cell producers,
    including controls and scientific outcome predicates, are replayed and
    bound to one source/family commitment.  This explicit false receipt is the
    safe integration boundary; it must not be replaced by a configuration flag.
    """

    source_kind = str(provenance.get("source_kind") or "unknown")
    return {
        "schema": "oph.physical_h3_kms.artifact_admission.v1",
        "source_kind": source_kind,
        "fixture_or_unregistered_report_input": source_kind
        in {"mapping_bundle", "mapping_config", "file", "run_directory", "unknown"},
        "registered_federation_replay_bound": False,
        "registered_transaction_replay_bound": False,
        "registered_observer_replay_bound": False,
        "registered_common_source_replay_bound": False,
        "per_cell_control_artifacts_replayed": False,
        "per_cell_scientific_predicates_recomputed": False,
        "single_bundle_commitment_verified": False,
        "PHYSICAL_ARTIFACT_REPLAY_ADMISSION_RECEIPT": False,
        "blockers": [
            "complete_registered_artifact_replay_chain_not_implemented",
            "per_cell_status_is_not_recomputed_from_primitive_artifacts",
            "single_source_and_campaign_family_commitment_not_verified",
        ],
        "claim_boundary": (
            "This fail-closed gate prevents ordinary dictionaries, JSON booleans, and declared "
            "cell statuses from authorizing physical promotion or branch retirement."
        ),
    }


def write_physical_h3_kms_preflight_report(
    source: Path | str | Mapping[str, Any],
    out: Path | str,
    *,
    reports: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the preflight and persist the structured JSON report."""

    report = physical_h3_kms_preflight_report(source, reports=reports)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _load_bundle(
    source: Path | str | Mapping[str, Any],
    reports: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if isinstance(source, Mapping):
        raw = dict(source)
        if "config" in raw or "reports" in raw:
            config = _mapping(raw.get("config"))
            evidence = _mapping(raw.get("reports"))
            if reports is not None:
                evidence.update(dict(reports))
            return config, evidence, {"source_kind": "mapping_bundle"}
        return raw, dict(reports or {}), {"source_kind": "mapping_config"}

    path = Path(source)
    if path.is_dir():
        config, config_path = _load_first(path, ("config.yml", "config.yaml", "config.json"))
        evidence: dict[str, Any] = {}
        loaded_reports: dict[str, str] = {}
        for name, candidates in _REPORT_FILE_CANDIDATES.items():
            payload, payload_path = _load_first(path, candidates)
            if payload_path is not None:
                evidence[name] = payload
                loaded_reports[name] = str(payload_path)
        if reports is not None:
            evidence.update(dict(reports))
        return config, evidence, {
            "source_kind": "run_directory",
            "run_directory": str(path),
            "config_path": str(config_path) if config_path else None,
            "loaded_reports": loaded_reports,
        }

    payload = _read_structured_file(path)
    if "config" in payload or "reports" in payload:
        config = _mapping(payload.get("config"))
        evidence = _mapping(payload.get("reports"))
    else:
        config = payload
        evidence = {}
    if reports is not None:
        evidence.update(dict(reports))
    return config, evidence, {"source_kind": "file", "path": str(path)}


def _load_first(root: Path, names: Sequence[str]) -> tuple[dict[str, Any], Path | None]:
    for name in names:
        path = root / name
        if path.is_file():
            return _read_structured_file(path), path
    return {}, None


def _read_structured_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _source_observer_stage(config: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the observer-like source system below the geometry gate."""

    blockers: list[str] = []
    architecture = _mapping(report.get("source_architecture"))
    carrier_count = _integer(architecture.get("carrier_count"))
    local_state_dimension = _integer(architecture.get("local_state_dimension"))
    boundary_port_count = _integer(architecture.get("boundary_port_count"))
    architecture_receipts = (
        "SOURCE_PATCH_ARCHITECTURE_RECEIPT",
        "PATCH_LOCAL_STATE_RECEIPT",
        "PATCH_PORT_BOUNDARY_RECEIPT",
        "PATCH_READBACK_RECEIPT",
        "PATCH_ALL_PORT_READBACK_RECEIPT",
        "RECORD_SIGNATURE_BINDS_ALL_LOCAL_PORT_STATE_RECEIPT",
        "ECHOSAHEDRAL_LOCAL_PATCH_ARCHITECTURE_RECEIPT",
        "ECHOSAHEDRAL_CARRIER_CONFORMANCE",
        "FEDERATION_SEWING_RECEIPT",
        "CARRIER_QUOTIENT_INVARIANCE_RECEIPT",
        "CARRIER_REFINEMENT_NATURALITY_RECEIPT",
    )
    missing_architecture_receipts = [
        name for name in architecture_receipts if report.get(name) is not True
    ]
    expected_materialized_coordinates = (
        carrier_count * 12
        if carrier_count is not None and carrier_count > 0
        else None
    )
    architecture_passed = bool(
        report.get("schema_version")
        == "oph_source_repair_record_observer_contract_v2"
        and architecture.get("bounded_patch_system") is True
        and architecture.get("simulation_native_source") is True
        and architecture.get("carrier_family")
        == "federated_echosahedral_patch_system"
        and architecture.get("one_local_echosahedron_per_carrier") is True
        and architecture.get("carrier_is_not_support_chart_cell") is True
        and architecture.get("carrier_is_not_primitive_observer") is True
        and carrier_count is not None
        and carrier_count > 0
        and local_state_dimension is not None
        and local_state_dimension > 0
        and _integer(architecture.get("local_state_factor_count")) == 12
        and boundary_port_count == 12
        and _integer(architecture.get("materialized_local_state_coordinate_count"))
        == expected_materialized_coordinates
        and architecture.get("all_local_port_readout_maps_materialized") is True
        and architecture.get("all_local_port_states_bound_into_records") is True
        and _nonempty_hash(architecture.get("local_patch_template_hash"))
        and _nonempty_hash(architecture.get("patch_port_state_sha256"))
        and _nonempty_hash(architecture.get("source_architecture_hash"))
        and not missing_architecture_receipts
    )
    source_federation = _mapping(config.get("source_federation"))
    configured_carrier_count = _integer(source_federation.get("carrier_count"))
    if source_federation.get("family") != "federated_echosahedral_carriers":
        architecture_passed = False
        blockers.append("source_federation_family_missing_or_mismatched")
    if (
        carrier_count is not None
        and configured_carrier_count is not None
        and carrier_count != configured_carrier_count
    ):
        architecture_passed = False
        blockers.append("source_architecture_carrier_count_mismatches_config")
    if configured_carrier_count is None:
        architecture_passed = False
        blockers.append("source_federation_carrier_count_missing")
    if not architecture_passed:
        blockers.append("SOURCE_PATCH_ARCHITECTURE_dependency_not_discharged")
    blockers.extend(
        f"{name}_missing_or_false" for name in missing_architecture_receipts
    )

    repair = _mapping(report.get("repair_dynamics"))
    repair_passed = bool(
        repair.get("local_update_rule") is True
        and repair.get("uses_only_local_state_and_ports") is True
        and repair.get("target_free_rule") is True
        and _positive_integer(repair.get("repair_event_count"))
        and _integer(repair.get("nonlocal_write_count")) == 0
        and _nonempty_hash(repair.get("repair_rule_hash"))
        and _nonempty_hash(repair.get("repair_event_log_hash"))
    )
    if not repair_passed:
        blockers.append("LOCAL_REPAIR_DYNAMICS_dependency_not_discharged")
    for receipt_name in (
        "TRANSACTION_VALIDATION_COMPLETE_READ_CONFLICT_SET_RECEIPT",
        "UNION_PAYLOAD_ATOMIC_REVALIDATION_RECEIPT",
    ):
        if report.get(receipt_name) is not True:
            blockers.append(f"{receipt_name}_missing_or_false")

    observer = _mapping(report.get("record_observer"))
    observer_passed = bool(
        _positive_integer(observer.get("observer_count"))
        and _positive_integer(observer.get("committed_record_count"))
        and _positive_integer(observer.get("readback_count"))
        and _positive_integer(observer.get("feedback_event_count"))
        and observer.get("readback_changes_future_local_actions") is True
        and observer.get("records_causally_bound_to_writes") is True
        and observer.get("bounded_interface_verified") is True
        and observer.get("self_prediction_beats_shuffled_control") is True
        and observer.get("feedback_ablation_changes_future_actions") is True
        and observer.get("checkpoint_continuation_verified") is True
        and _integer(observer.get("orphan_read_count")) == 0
        and _nonempty_hash(observer.get("record_readback_feedback_log_hash"))
    )
    if not observer_passed:
        blockers.append("OBSERVER_SELF_READING_RECORD_LOOP_dependency_not_discharged")

    forbidden_hits = [str(value) for value in _sequence(report.get("source_forbidden_target_hits"))]
    if report.get("source_generator_target_free") is not True or forbidden_hits:
        blockers.append("SOURCE_GENERATOR_TARGET_FREE_dependency_not_discharged")

    return _stage(
        blockers,
        {
            "dependency_status": {
                "SOURCE_PATCH_ARCHITECTURE": architecture_passed,
                "LOCAL_REPAIR_DYNAMICS": repair_passed,
                "OBSERVER_SELF_READING_RECORD_LOOP": observer_passed,
                "SOURCE_GENERATOR_TARGET_FREE": bool(
                    report.get("source_generator_target_free") is True and not forbidden_hits
                ),
            },
            "carrier_count": carrier_count,
            "local_state_dimension": local_state_dimension,
            "boundary_port_count": boundary_port_count,
            "expected_boundary_port_count": 12,
            "expected_materialized_local_state_coordinate_count": expected_materialized_coordinates,
            "missing_architecture_receipts": missing_architecture_receipts,
            "config_carrier_count": configured_carrier_count,
            "repair_event_count": _integer(repair.get("repair_event_count")),
            "observer_count": _integer(observer.get("observer_count")),
            "committed_record_count": _integer(observer.get("committed_record_count")),
            "readback_count": _integer(observer.get("readback_count")),
            "feedback_event_count": _integer(observer.get("feedback_event_count")),
            "source_forbidden_target_hits": forbidden_hits,
        },
        hard_blockers=("source_observer_contract_missing",) if blockers else (),
    )


def _refinement_stage(config: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    support_regulator = _mapping(config.get("support_regulator"))
    family = str(
        report.get("mesh_family") or support_regulator.get("family") or ""
    )
    if family not in _ALLOWED_REFINEMENT_FAMILIES:
        blockers.append("mesh_family_is_not_nested_geodesic_icosahedral")
    if support_regulator.get("family") not in _ALLOWED_REFINEMENT_FAMILIES:
        blockers.append("support_regulator_config_is_missing_or_not_icosahedral")

    levels = [dict(row) for row in _sequence(report.get("levels")) if isinstance(row, Mapping)]
    if len(levels) < 2:
        blockers.append("nested_refinement_levels_missing")
    level_ids = [str(row.get("level_id", "")) for row in levels]
    counts = [_integer(row.get("patch_count")) for row in levels]
    if levels and (any(not value for value in level_ids) or len(set(level_ids)) != len(level_ids)):
        blockers.append("refinement_level_ids_missing_or_nonunique")
    if counts and (
        any(value is None or value <= 0 for value in counts)
        or any(
            left is None or right is None or left >= right
            for left, right in zip(counts, counts[1:], strict=False)
        )
    ):
        blockers.append("refinement_patch_counts_not_strictly_ordered")
    for index in range(1, len(levels)):
        parent = str(levels[index].get("parent_level_id", ""))
        if not parent or parent != level_ids[index - 1]:
            blockers.append(f"level_{index}_parent_lineage_mismatch")
        if not _nonempty_hash(levels[index].get("lineage_hash")):
            blockers.append(f"level_{index}_lineage_hash_missing")

    if report.get("nested_lineage_receipt") is not True:
        blockers.append("nested_lineage_receipt_missing")
    expectations = [
        dict(row) for row in _sequence(report.get("conditional_expectations")) if isinstance(row, Mapping)
    ]
    if len(expectations) != max(0, len(levels) - 1):
        blockers.append("one_conditional_expectation_per_refinement_step_required")
    expected_pairs = {
        (level_ids[index], level_ids[index - 1]) for index in range(1, len(level_ids))
    }
    observed_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(expectations):
        pair = (str(row.get("fine_level_id", "")), str(row.get("coarse_level_id", "")))
        observed_pairs.add(pair)
        if not _nonempty_hash(row.get("operator_hash")):
            blockers.append(f"conditional_expectation_{index}_operator_hash_missing")
        for clause in ("unital", "positive", "state_preserving", "cap_isotony_compatible"):
            if row.get(clause) is not True:
                blockers.append(f"conditional_expectation_{index}_{clause}_not_verified")
    if expectations and observed_pairs != expected_pairs:
        blockers.append("conditional_expectation_level_pairs_do_not_match_lineage")
    if report.get("conditional_expectations_receipt") is not True:
        blockers.append("conditional_expectations_receipt_missing")
    if report.get("TRUE_ICOSAHEDRAL_REFINEMENT_TOWER_RECEIPT") is not True:
        blockers.append("true_icosahedral_refinement_tower_receipt_missing")
    if report.get("A5_EQUIVARIANT_REFINEMENT_RECEIPT") is not True:
        blockers.append("a5_equivariant_refinement_receipt_missing")
    if report.get("PAPER_MULTIRESOLUTION_REGULATOR_CERTIFICATE") is not True:
        blockers.append("full_noncommutative_multiresolution_certificate_missing")
    for index, row in enumerate(expectations):
        if row.get("noncommutative_prime_cap_expectation") is not True:
            blockers.append(
                f"conditional_expectation_{index}_not_on_noncommutative_prime_cap_algebra"
            )

    return _stage(
        blockers,
        {
            "mesh_family": family or None,
            "level_count": len(levels),
            "patch_counts": counts,
            "conditional_expectation_count": len(expectations),
            "config_support_regulator_family": support_regulator.get("family"),
            "full_multiresolution_certificate": report.get(
                "PAPER_MULTIRESOLUTION_REGULATOR_CERTIFICATE"
            ),
        },
        hard_blockers=("nonnested_regulator_family",) if blockers else (),
    )


def _prime_geometric_state_stage(
    config: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    blockers: list[str] = []
    bw = _mapping(config.get("bw"))
    state_mode = str(report.get("state_mode") or bw.get("state_mode") or "")
    if any(token in state_mode.lower() for token in ("record", "history", "koopman", "declared")):
        blockers.append("configured_state_mode_is_record_history_or_declared_surrogate")

    if report.get("algebra_scope") != "prime_geometric_cap_interior":
        blockers.append("algebra_scope_is_not_prime_geometric_cap_interior")
    if report.get("state_construction") != "source_maxent_cap_state":
        blockers.append("state_is_not_source_maxent_cap_state")
    if report.get("noncommutative_algebra") is not True:
        blockers.append("noncommutative_cap_algebra_not_verified")
    commutator_norm = _finite_number(report.get("commutator_norm"))
    if commutator_norm is None or commutator_norm <= 1.0e-10:
        blockers.append("cap_algebra_commutator_is_missing_or_central")

    source_fields = [str(value) for value in _sequence(report.get("source_primitive_fields"))]
    if not source_fields:
        blockers.append("source_primitive_fields_missing")
    forbidden_hits = _token_hits(source_fields, _FORBIDDEN_CAP_STATE_TOKENS)
    surrogate_inputs = [str(value) for value in _sequence(report.get("surrogate_inputs"))]
    if forbidden_hits or surrogate_inputs:
        blockers.append("record_history_pointer_repair_or_target_input_contaminates_cap_state")

    rho = _mapping(report.get("rho"))
    rho_dimension = _integer(rho.get("dimension"))
    if rho_dimension is None or rho_dimension < 2:
        blockers.append("rho_dimension_missing_or_trivial")
    if not _near(rho.get("trace"), 1.0, 1.0e-8):
        blockers.append("rho_trace_not_one")
    minimum_eigenvalue = _finite_number(rho.get("minimum_eigenvalue"))
    if minimum_eigenvalue is None or minimum_eigenvalue <= 0.0:
        blockers.append("rho_not_faithful_positive")
    if not _at_most(rho.get("hermiticity_residual"), 1.0e-8):
        blockers.append("rho_hermiticity_not_verified")
    if not _nonempty_hash(rho.get("matrix_hash")):
        blockers.append("rho_matrix_hash_missing")

    generator = _mapping(report.get("modular_generator"))
    if generator.get("construction") != "negative_log_density_matrix":
        blockers.append("modular_generator_is_not_negative_log_rho")
    if rho_dimension is None or _integer(generator.get("dimension")) != rho_dimension:
        blockers.append("modular_generator_dimension_mismatch")
    if not _at_most(generator.get("functional_calculus_residual"), 1.0e-8):
        blockers.append("negative_log_rho_relation_not_verified")
    noncentrality = _finite_number(generator.get("noncentrality_norm"))
    if noncentrality is None or noncentrality <= 1.0e-10:
        blockers.append("modular_generator_is_missing_or_central")
    if not _nonempty_hash(generator.get("matrix_hash")):
        blockers.append("modular_generator_matrix_hash_missing")

    mixed_gns = _mapping(report.get("mixed_gns"))
    for clause in ("constructed", "left_right_representation", "cyclic_separating_support"):
        if mixed_gns.get(clause) is not True:
            blockers.append(f"mixed_gns_{clause}_not_verified")

    hard_blockers: list[str] = []
    if blockers:
        hard_blockers.append("prime_cap_algebra_missing")
    if forbidden_hits or surrogate_inputs or any(
        token in state_mode.lower() for token in ("record", "history", "koopman", "repair", "pointer")
    ):
        hard_blockers.append("cap_state_uses_record_or_repair_features")
    if any(value.startswith("mixed_gns_") for value in blockers):
        hard_blockers.append("mixed_gns_tower_missing")

    return _stage(
        blockers,
        {
            "state_mode": state_mode or None,
            "algebra_scope": report.get("algebra_scope"),
            "rho_dimension": rho_dimension,
            "commutator_norm": commutator_norm,
            "generator_noncentrality_norm": noncentrality,
            "source_primitive_fields": source_fields,
            "forbidden_source_token_hits": forbidden_hits,
            "surrogate_inputs": surrogate_inputs,
        },
        hard_blockers=hard_blockers,
    )


def _independent_geometry_stage(
    config: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    blockers: list[str] = []
    method = str(report.get("derivation_method") or "")
    if method not in _ALLOWED_GEOMETRY_DERIVATIONS:
        blockers.append("geometry_derivation_is_not_source_ordered_frame_cross_ratio")
    primitive_fields = [str(value) for value in _sequence(report.get("source_primitive_fields"))]
    required_primitives = {"cap_incidence", "orientation", "ordered_bw_frame", "cross_ratio"}
    if not required_primitives.issubset(set(primitive_fields)):
        blockers.append("geometry_source_primitives_incomplete")
    expression = str(report.get("derivation_expression") or "")
    if not expression:
        blockers.append("geometry_derivation_expression_missing")
    forbidden_hits = _token_hits([expression, *primitive_fields], _FORBIDDEN_GEOMETRY_TOKENS)
    declared_hits = [str(value) for value in _sequence(report.get("forbidden_token_hits"))]
    if forbidden_hits or declared_hits:
        blockers.append("geometry_derivation_contains_pi_kms_or_target_token")
    for clause in (
        "target_blind_derivation",
        "independent_of_modular_fit",
        "independent_of_kms_target",
        "orientation_fixed_from_source",
    ):
        if report.get(clause) is not True:
            blockers.append(f"{clause}_not_verified")

    values = [_finite_number(value) for value in _sequence(report.get("geometric_parameter_values"))]
    if len(values) < 3 or any(value is None for value in values):
        blockers.append("finite_geometric_parameter_values_missing")
    elif len({round(float(value), 14) for value in values if value is not None}) < 3:
        blockers.append("geometric_parameter_values_degenerate")
    geometry_rows = _id_tuple(report.get("geometry_source_row_ids"))
    kms_rows = _id_tuple(report.get("kms_score_row_ids"))
    if not geometry_rows or not kms_rows:
        blockers.append("geometry_and_kms_row_partitions_missing")
    elif set(geometry_rows).intersection(kms_rows):
        blockers.append("geometry_derivation_rows_overlap_kms_scoring_rows")
    if not _nonempty_hash(report.get("geometry_derivation_hash")):
        blockers.append("geometry_derivation_hash_missing")

    bw = _mapping(config.get("bw"))
    configured_normalization = bw.get("normalization")
    configured_target_scale = bw.get("transition_response_scale")
    return _stage(
        blockers,
        {
            "derivation_method": method or None,
            "source_primitive_fields": primitive_fields,
            "computed_forbidden_token_hits": forbidden_hits,
            "declared_forbidden_token_hits": declared_hits,
            "geometry_row_count": len(geometry_rows),
            "kms_score_row_count": len(kms_rows),
            "configured_bw_normalization": configured_normalization,
            "configured_transition_response_scale": configured_target_scale,
        },
        hard_blockers=("independent_geometric_clock_missing",) if blockers else (),
    )


def _candidate_intervention_stage(
    config: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    blockers: list[str] = []
    scientific_failures: list[str] = []
    rows = _named_rows(report.get("candidates"), "label")
    labels = set(rows)
    if labels != set(REQUIRED_CLOCK_LABELS):
        blockers.append("clock_candidate_set_must_be_exactly_1x_pi_2pi_4pi")
    if report.get("interventions_frozen_before_candidate_scoring") is not True:
        blockers.append("interventions_not_frozen_before_candidate_scoring")
    if report.get("candidate_labels_frozen_before_runs") is not True:
        blockers.append("candidate_labels_not_frozen_before_runs")
    if report.get("source_intervention_target_free") is not True:
        blockers.append("source_intervention_not_proven_target_free")

    intervention_rows: list[tuple[str, ...]] = []
    heldout_rows: list[tuple[str, ...]] = []
    packet_hashes: list[str] = []
    trajectory_hashes: list[str] = []
    response_hashes: list[str] = []
    aggregate_hashes: list[str] = []
    expected_values = {
        "1x": 1.0,
        "pi": math.pi,
        "2pi": 2.0 * math.pi,
        "4pi": 4.0 * math.pi,
    }
    for label in REQUIRED_CLOCK_LABELS:
        row = rows.get(label, {})
        intervention_ids = _id_tuple(row.get("intervention_row_ids"))
        heldout_ids = _id_tuple(row.get("heldout_event_row_ids"))
        packet_hash = str(row.get("intervention_packet_hash") or "")
        trajectory_hash = str(row.get("source_trajectory_hash") or "")
        response_hash = str(row.get("raw_response_hash") or "")
        aggregate_hash = str(row.get("candidate_invariance_aggregate_hash") or "")
        intervention_rows.append(intervention_ids)
        heldout_rows.append(heldout_ids)
        packet_hashes.append(packet_hash)
        trajectory_hashes.append(trajectory_hash)
        response_hashes.append(response_hash)
        aggregate_hashes.append(aggregate_hash)
        if not intervention_ids:
            blockers.append(f"{label}_intervention_rows_missing")
        if not heldout_ids:
            blockers.append(f"{label}_heldout_rows_missing")
        if not _strict_sha256(packet_hash):
            blockers.append(f"{label}_intervention_packet_hash_missing")
        if not _strict_sha256(trajectory_hash):
            blockers.append(f"{label}_source_trajectory_hash_missing")
        if not _strict_sha256(response_hash):
            blockers.append(f"{label}_raw_response_hash_missing")
        if not _strict_sha256(aggregate_hash):
            blockers.append(f"{label}_candidate_invariance_aggregate_hash_missing")
        if row.get("candidate_scale_applied_only_in_scoring") is not True:
            blockers.append(f"{label}_candidate_scale_not_scoring_only")
        if row.get("candidate_scale_enters_intervention") is not False:
            blockers.append(f"{label}_candidate_scale_intervention_entry_not_disproved")
        if row.get("candidate_parameter_name") != "kappa":
            blockers.append(f"{label}_candidate_parameter_name_is_not_kappa")
        if not _near(row.get("candidate_value"), expected_values[label], 1.0e-12):
            blockers.append(f"{label}_candidate_value_mismatch")
        if row.get("candidate_units") != "dimensionless_geometric_flow_parameter":
            blockers.append(f"{label}_candidate_units_mismatch")
        aggregate_payload = {
            "intervention_row_ids": list(intervention_ids),
            "heldout_event_row_ids": list(heldout_ids),
            "intervention_packet_hash": packet_hash,
            "source_trajectory_hash": trajectory_hash,
            "raw_response_hash": response_hash,
        }
        computed_aggregate = _canonical_sha256(aggregate_payload)
        if aggregate_hash != computed_aggregate:
            blockers.append(f"{label}_candidate_invariance_aggregate_hash_mismatch")
    if any(intervention_rows) and len(set(intervention_rows)) != 1:
        blockers.append("candidate_intervention_row_ids_differ")
    if any(heldout_rows) and len(set(heldout_rows)) != 1:
        blockers.append("candidate_heldout_event_row_ids_differ")
    if any(packet_hashes) and len(set(packet_hashes)) != 1:
        blockers.append("candidate_intervention_packet_hashes_differ")
    if any(trajectory_hashes) and len(set(trajectory_hashes)) != 1:
        blockers.append("candidate_source_trajectory_hashes_differ")
    if any(response_hashes) and len(set(response_hashes)) != 1:
        blockers.append("candidate_raw_response_hashes_differ")
    if any(aggregate_hashes) and len(set(aggregate_hashes)) != 1:
        blockers.append("candidate_invariance_aggregate_hashes_differ")
    report_aggregate_hash = str(report.get("candidate_invariance_aggregate_hash") or "")
    if not _strict_sha256(report_aggregate_hash) or (
        aggregate_hashes and report_aggregate_hash != aggregate_hashes[0]
    ):
        blockers.append("report_candidate_invariance_aggregate_hash_mismatch")

    continuous = _mapping(report.get("continuous_clock_fit"))
    interval = [_finite_number(value) for value in _sequence(continuous.get("fitted_kappa_interval"))]
    if len(interval) != 2 or any(value is None for value in interval):
        blockers.append("continuous_clock_interval_missing_or_nonfinite")
    else:
        lower, upper = (float(value) for value in interval if value is not None)
        if not lower < upper:
            blockers.append("continuous_clock_interval_is_not_ordered")
        else:
            if not (lower <= 2.0 * math.pi <= upper):
                scientific_failures.append(
                    "continuous_clock_interval_does_not_contain_2pi"
                )
            if any(
                lower <= candidate <= upper
                for candidate in (1.0, math.pi, 4.0 * math.pi)
            ):
                scientific_failures.append(
                    "continuous_clock_interval_does_not_exclude_1_pi_4pi"
                )
    residual = _finite_number(continuous.get("absolute_residual"))
    residual_limit = _finite_number(continuous.get("frozen_absolute_residual_threshold"))
    if (
        residual is None
        or residual_limit is None
        or residual < 0.0
        or residual_limit < 0.0
    ):
        blockers.append("continuous_clock_absolute_adequacy_inputs_missing")
    elif residual > residual_limit:
        scientific_failures.append("continuous_clock_absolute_adequacy_failed")
    if "wrong_normalization_separation_passed" not in continuous:
        blockers.append("continuous_clock_wrong_normalization_result_missing")
    elif continuous.get("wrong_normalization_separation_passed") is not True:
        scientific_failures.append(
            "continuous_clock_wrong_normalization_separation_failed"
        )
    if "refinement_tail_stable" not in continuous:
        blockers.append("continuous_clock_refinement_tail_result_missing")
    elif continuous.get("refinement_tail_stable") is not True:
        scientific_failures.append("continuous_clock_refinement_tail_not_stable")
    if not _strict_sha256(continuous.get("fit_artifact_hash")):
        blockers.append("continuous_clock_fit_artifact_hash_missing")

    discrete = _mapping(report.get("discrete_clock_comparison"))
    losses_raw = _mapping(discrete.get("paired_losses"))
    losses = {label: _finite_number(losses_raw.get(label)) for label in REQUIRED_CLOCK_LABELS}
    if any(value is None for value in losses.values()):
        blockers.append("discrete_clock_paired_losses_missing_or_nonfinite")
    delta = _finite_number(discrete.get("frozen_delta_clock"))
    uncertainty = _finite_number(discrete.get("paired_uncertainty_upper_bound"))
    if delta is None or delta < 0.0 or uncertainty is None or uncertainty < 0.0:
        blockers.append("discrete_clock_frozen_margin_or_uncertainty_missing")
    elif all(value is not None for value in losses.values()):
        lhs = float(losses["2pi"]) + delta + uncertainty
        rhs = min(float(losses[label]) for label in ("1x", "pi", "4pi"))
        if not lhs < rhs:
            scientific_failures.append(
                "discrete_clock_2pi_does_not_defeat_1_pi_4pi_by_frozen_margin"
            )
    if discrete.get("same_rows_and_packets") is not True:
        blockers.append("discrete_clock_comparison_not_paired_on_same_rows_and_packets")
    if discrete.get("thresholds_from_independent_synthetic_calibration") is not True:
        blockers.append("clock_thresholds_not_from_independent_synthetic_calibration")
    if discrete.get("thresholds_frozen_before_physical_campaign") is not True:
        blockers.append("clock_thresholds_not_prefrozen")
    if not _strict_sha256(discrete.get("calibration_artifact_hash")):
        blockers.append("clock_calibration_artifact_hash_missing")

    h3_config = _mapping(config.get("h3_modular_response"))
    likely_scale_dependent = bool(
        str(h3_config.get("perturb_budget_mode", "")).lower() == "modular_amount"
        or "lambda" in str(h3_config.get("perturb_selection_mode", "")).lower()
        or "transport" in str(h3_config.get("transition_readout_mode", "")).lower()
    )
    if likely_scale_dependent:
        blockers.append("config_indicates_target_or_candidate_scale_dependent_source_intervention")
    clock_blocked = any(
        value.startswith("continuous_clock_")
        or value.startswith("discrete_clock_")
        or value.startswith("clock_thresholds_")
        or value.startswith("clock_calibration_")
        for value in blockers
    )
    candidate_hash_blocked = any(
        "hash" in value
        or "intervention" in value
        or "source_trajectory" in value
        or "heldout_event_row_ids" in value
        for value in blockers
    )
    return _stage(
        blockers,
        {
            "candidate_labels": sorted(labels),
            "common_intervention_rows": bool(intervention_rows and len(set(intervention_rows)) == 1),
            "common_heldout_rows": bool(heldout_rows and len(set(heldout_rows)) == 1),
            "common_intervention_packet_hash": bool(packet_hashes and len(set(packet_hashes)) == 1),
            "common_source_trajectory_hash": bool(
                trajectory_hashes and len(set(trajectory_hashes)) == 1
            ),
            "common_raw_response_hash": bool(response_hashes and len(set(response_hashes)) == 1),
            "common_candidate_invariance_aggregate_hash": bool(
                aggregate_hashes and len(set(aggregate_hashes)) == 1
            ),
            "candidate_invariance_aggregate_hash": report_aggregate_hash or None,
            "continuous_clock_fit_passed": not any(
                value.startswith("continuous_clock_")
                for value in scientific_failures
            ),
            "discrete_clock_comparison_passed": not any(
                value.startswith("discrete_clock_")
                for value in scientific_failures
            ),
            "scientific_outcome": (
                CellStatus.VALID_FAIL.value
                if scientific_failures
                else CellStatus.VALID_PASS.value
            ),
            "scientific_failures": scientific_failures,
            "config_likely_scale_dependent_intervention": likely_scale_dependent,
            "source_intervention_target_free": report.get(
                "source_intervention_target_free"
            )
            is True,
        },
        hard_blockers=[
            *(["candidate_intervention_hash_mismatch"] if candidate_hash_blocked else []),
            *(["independent_geometric_clock_missing"] if clock_blocked else []),
        ],
    )


def _native_bw_stage(config: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """Require a native, primitive-derived BW01--BW08 payload.

    The underlying verifier owns every numerical pass predicate.  This stage
    cannot be satisfied by a producer-supplied clause Boolean or by an archived
    replay/fixture report.
    """

    del config
    verification = native_bw01_bw08_report(report)
    conformance_receipt = (
        verification.get("native_payload_conformance_receipt") is True
    )
    receipt = verification.get("native_payload_receipt") is True
    blockers = (
        []
        if conformance_receipt
        else list(_sequence(verification.get("conformance_blockers")))
    )
    if not conformance_receipt and not blockers:
        blockers = ["native_bw01_bw08_payload_did_not_conform"]
    clause_rows = _mapping(verification.get("clauses"))
    failed_clause_ids = {
        clause_id
        for clause_id in (f"BW{index:02d}" for index in range(1, 9))
        if _mapping(clause_rows.get(clause_id)).get("passed") is not True
    }
    hard_blockers = ["bw_native_payload_missing"] if not conformance_receipt else []
    return _stage(
        blockers,
        {
            "native_payload_conformance_receipt": conformance_receipt,
            "native_payload_receipt": receipt,
            "scientific_outcome": verification.get("scientific_outcome"),
            "scientific_failures": verification.get("scientific_failures", []),
            "failed_clause_ids": sorted(failed_clause_ids),
            "antecedent_hash": verification.get("antecedent_hash"),
            "required_clause_ids": verification.get("required_clause_ids", []),
            "clauses": verification.get("clauses", {}),
            "recomputed_issue308_tier": verification.get("recomputed_issue308_tier"),
        },
        hard_blockers=hard_blockers,
    )


def _geometry_control_stage(
    config: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    del config
    control_blockers: list[str] = []
    scientific_failures: list[str] = []
    rows = _named_rows(report.get("models"), "model")
    if set(rows) != set(REQUIRED_GEOMETRY_MODELS):
        control_blockers.append("geometry_model_set_must_be_exactly_h3_s2_e3_e4")
    if report.get("models_frozen_before_holdout") is not True:
        control_blockers.append("geometry_models_not_frozen_before_holdout")
    if report.get("heldout_excluded_from_model_selection") is not True:
        control_blockers.append("heldout_rows_used_or_not_proven_excluded_from_model_selection")
    heldout_rows: list[tuple[str, ...]] = []
    shared_hash_fields = (
        "heldout_event_matrix_hash",
        "heldout_weights_hash",
        "missingness_mask_hash",
        "preprocessing_hash",
        "source_packet_hash",
        "prediction_target_hash",
        "fit_protocol_hash",
    )
    hashes: dict[str, list[str]] = {field: [] for field in shared_hash_fields}
    capacities: list[int | None] = []
    scores: list[float | None] = []
    for model in REQUIRED_GEOMETRY_MODELS:
        row = rows.get(model, {})
        ids = _id_tuple(row.get("heldout_event_row_ids"))
        heldout_rows.append(ids)
        if not ids:
            control_blockers.append(f"{model}_heldout_rows_missing")
        for field in shared_hash_fields:
            value = str(row.get(field) or "")
            hashes[field].append(value)
            if not _strict_sha256(value):
                control_blockers.append(f"{model}_{field}_missing")
        capacity = _integer(row.get("effective_model_capacity"))
        capacities.append(capacity)
        if capacity is None or capacity <= 0:
            control_blockers.append(f"{model}_effective_model_capacity_missing")
        score = _finite_number(row.get("heldout_score"))
        scores.append(score)
        if score is None:
            control_blockers.append(f"{model}_heldout_score_missing_or_nonfinite")
        if row.get("optimizer_status") != "CONVERGED":
            control_blockers.append(f"{model}_optimizer_not_converged")
        if row.get("required_rows_complete") is not True:
            control_blockers.append(f"{model}_required_rows_incomplete")
    if any(heldout_rows) and len(set(heldout_rows)) != 1:
        control_blockers.append("geometry_models_do_not_share_heldout_event_rows")
    for field, values in hashes.items():
        if any(values) and len(set(values)) != 1:
            control_blockers.append(f"geometry_models_do_not_share_{field}")
    if any(value is not None for value in capacities) and len(set(capacities)) != 1:
        control_blockers.append("geometry_model_effective_capacities_are_not_matched")

    paired = _mapping(report.get("paired_geometry_comparison"))
    geometry_margin = _finite_number(paired.get("frozen_h3_win_margin"))
    geometry_uncertainty = _finite_number(
        paired.get("paired_uncertainty_upper_bound")
    )
    if paired.get("loss_direction") != "lower_is_better":
        control_blockers.append("geometry_loss_direction_missing_or_invalid")
    if geometry_margin is None or geometry_margin < 0.0:
        control_blockers.append("geometry_frozen_h3_win_margin_missing")
    if geometry_uncertainty is None or geometry_uncertainty < 0.0:
        control_blockers.append("geometry_paired_uncertainty_missing")
    if paired.get("thresholds_frozen_before_physical_campaign") is not True:
        control_blockers.append("geometry_thresholds_not_prefrozen")
    if not _strict_sha256(paired.get("calibration_artifact_hash")):
        control_blockers.append("geometry_calibration_artifact_hash_missing")
    if (
        all(value is not None for value in scores)
        and geometry_margin is not None
        and geometry_uncertainty is not None
        and not (
            float(scores[0]) + geometry_margin + geometry_uncertainty
            < min(float(value) for value in scores[1:] if value is not None)
        )
    ):
        scientific_failures.append(
            "h3_does_not_defeat_s2_e3_e4_by_frozen_paired_margin"
        )

    leverage = _mapping(report.get("curvature_leverage"))
    leverage_blockers: list[str] = []
    if leverage.get("calibration_source") != "independent_synthetic_power_suite":
        leverage_blockers.append("curvature_power_calibration_not_independent")
    if leverage.get("frozen_before_physical_campaign") is not True:
        leverage_blockers.append("curvature_power_calibration_not_prefrozen")
    for field in ("calibration_hash", "registered_analysis_hash"):
        if not _strict_sha256(leverage.get(field)):
            leverage_blockers.append(f"curvature_{field}_missing")
    for field in ("domain_diameter", "registered_curvature_radius", "noise_scale"):
        value = _finite_number(leverage.get(field))
        if value is None or value <= 0.0:
            leverage_blockers.append(f"curvature_{field}_missing_or_nonpositive")
    if not _positive_integer(leverage.get("sample_count")):
        leverage_blockers.append("curvature_sample_count_missing")
    power = _finite_number(leverage.get("calibrated_power"))
    minimum_power = _finite_number(leverage.get("minimum_power"))
    if (
        power is None
        or minimum_power is None
        or not (0.0 < minimum_power <= 1.0)
        or not (minimum_power <= power <= 1.0)
    ):
        leverage_blockers.append("curvature_power_requirement_not_met")
    radius_source = str(leverage.get("curvature_radius_source") or "")
    independent_radius = bool(
        radius_source == "independent_source"
        and leverage.get("curvature_radius_frozen_before_h3_fit") is True
    )
    fitted_radius_contract = bool(
        radius_source == "fitted_capacity_charged"
        and leverage.get("curvature_parameter_charged_to_model_capacity") is True
    )
    if not (independent_radius or fitted_radius_contract):
        leverage_blockers.append("h3_curvature_radius_not_identifiable_against_flat_limit")
    if fitted_radius_contract and "flat_limit_excluded" not in leverage:
        leverage_blockers.append("h3_flat_limit_test_result_missing")
    elif fitted_radius_contract and leverage.get("flat_limit_excluded") is not True:
        scientific_failures.append("h3_flat_limit_not_excluded")
    if "h3_e3_distinguishable_at_registered_effect" not in leverage:
        leverage_blockers.append("h3_e3_registered_curvature_test_result_missing")
    elif leverage.get("h3_e3_distinguishable_at_registered_effect") is not True:
        scientific_failures.append(
            "h3_e3_registered_curvature_effect_not_distinguishable"
        )

    blockers = [*control_blockers, *leverage_blockers]
    hard_blockers: list[str] = []
    if control_blockers:
        hard_blockers.append("geometry_control_missing_or_nonfinite")
    if leverage_blockers:
        hard_blockers.append("curvature_leverage_missing")
    return _stage(
        blockers,
        {
            "models": sorted(rows),
            "same_heldout_rows": bool(heldout_rows and len(set(heldout_rows)) == 1),
            "shared_field_checks": {
                field: bool(values and len(set(values)) == 1) for field, values in hashes.items()
            },
            "matched_effective_model_capacity": bool(capacities and len(set(capacities)) == 1),
            "all_scores_finite": bool(scores and all(value is not None for value in scores)),
            "scientific_outcome": (
                CellStatus.VALID_FAIL.value
                if scientific_failures
                else CellStatus.VALID_PASS.value
            ),
            "scientific_failures": scientific_failures,
            "paired_geometry_comparison": {
                "frozen_h3_win_margin": geometry_margin,
                "paired_uncertainty_upper_bound": geometry_uncertainty,
            },
            "curvature_leverage": {
                "passed": not leverage_blockers,
                "blockers": leverage_blockers,
                "curvature_radius_source": radius_source or None,
                "calibrated_power": power,
                "minimum_power": minimum_power,
            },
        },
        hard_blockers=hard_blockers,
    )


def _semantic_event_stage(
    config: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    event_blockers: list[str] = []
    scientific_failures: list[str] = []
    clauses = _mapping(report.get("event_clauses"))
    if set(clauses) != {
        "EVENT_E1_POPULATION",
        "EVENT_E2_SEPARATION",
        "EVENT_E3_RANK_FOUR",
        "EVENT_E4_POINCARE_COCYCLE",
    }:
        event_blockers.append("event_clause_set_must_be_exactly_event_e1_through_event_e4")
    e1 = _mapping(clauses.get("EVENT_E1_POPULATION"))
    e2 = _mapping(clauses.get("EVENT_E2_SEPARATION"))
    e3 = _mapping(clauses.get("EVENT_E3_RANK_FOUR"))
    e4 = _mapping(clauses.get("EVENT_E4_POINCARE_COCYCLE"))

    germ_count = _integer(e1.get("semantic_record_germ_count"))
    box_count = _integer(e1.get("certified_localization_box_count"))
    if germ_count is None:
        event_blockers.append("EVENT_E1_semantic_record_population_missing")
    elif germ_count < 2:
        scientific_failures.append("EVENT_E1_semantic_record_population_failed")
    if box_count is None:
        event_blockers.append("EVENT_E1_certified_box_population_missing")
    elif box_count != germ_count:
        scientific_failures.append("EVENT_E1_certified_box_population_mismatch")
    for field, failure in (
        ("dense_population_verified", "EVENT_E1_dense_population_not_verified"),
        (
            "shrinking_box_sequence_verified",
            "EVENT_E1_shrinking_certified_boxes_not_verified",
        ),
    ):
        if field not in e1:
            event_blockers.append(f"EVENT_E1_{field}_result_missing")
        elif e1.get(field) is not True:
            scientific_failures.append(failure)
    if not _strict_sha256(e1.get("population_artifact_hash")):
        event_blockers.append("EVENT_E1_population_artifact_hash_missing")

    distinct_pairs = _integer(e2.get("distinct_germ_pair_count"))
    separated_pairs = _integer(e2.get("separated_pair_count"))
    if distinct_pairs is None or separated_pairs is None:
        event_blockers.append("EVENT_E2_pair_counts_missing")
    elif distinct_pairs <= 0 or separated_pairs != distinct_pairs:
        scientific_failures.append("EVENT_E2_distinct_germs_not_all_separated")
    gap = _finite_number(e2.get("minimum_localization_gap"))
    if gap is None:
        event_blockers.append("EVENT_E2_positive_localization_gap_missing")
    elif gap <= 0.0:
        scientific_failures.append("EVENT_E2_positive_localization_gap_failed")
    if "disjoint_certified_boxes_verified" not in e2:
        event_blockers.append("EVENT_E2_disjoint_box_test_result_missing")
    elif e2.get("disjoint_certified_boxes_verified") is not True:
        scientific_failures.append(
            "EVENT_E2_disjoint_certified_boxes_not_verified"
        )
    if not _strict_sha256(e2.get("separation_artifact_hash")):
        event_blockers.append("EVENT_E2_separation_artifact_hash_missing")

    for field, expected, failure in (
        (
            "conditioned_spatial_response_rank",
            3,
            "EVENT_E3_conditioned_spatial_rank_is_not_three",
        ),
        (
            "independent_clock_line_rank",
            1,
            "EVENT_E3_independent_clock_line_missing",
        ),
        ("combined_event_rank", 4, "EVENT_E3_combined_event_rank_is_not_four"),
    ):
        value = _integer(e3.get(field))
        if value is None:
            event_blockers.append(f"EVENT_E3_{field}_result_missing")
        elif value != expected:
            scientific_failures.append(failure)
    for field, failure in (
        (
            "clock_line_independent_of_spatial_response",
            "EVENT_E3_clock_line_not_independent",
        ),
        ("independent_clock_receipt", "EVENT_E3_independent_clock_failed"),
    ):
        if field not in e3:
            event_blockers.append(f"EVENT_E3_{field}_result_missing")
        elif e3.get(field) is not True:
            scientific_failures.append(failure)
    if not _strict_sha256(e3.get("rank_four_artifact_hash")):
        event_blockers.append("EVENT_E3_rank_four_artifact_hash_missing")

    overlap_count = _integer(e4.get("overlap_transition_count"))
    if overlap_count is None:
        event_blockers.append("EVENT_E4_overlap_transitions_missing")
    elif overlap_count <= 0:
        scientific_failures.append("EVENT_E4_overlap_transitions_failed")
    for field, failure in (
        ("lorentz_components_present", "EVENT_E4_lorentz_components_missing"),
        (
            "translation_components_present",
            "EVENT_E4_translation_components_missing",
        ),
        ("connected_overlap_atlas", "EVENT_E4_overlap_atlas_not_connected"),
    ):
        if field not in e4:
            event_blockers.append(f"EVENT_E4_{field}_result_missing")
        elif e4.get(field) is not True:
            scientific_failures.append(failure)
    cocycle_residual = _finite_number(e4.get("poincare_cocycle_residual"))
    if cocycle_residual is None:
        event_blockers.append("EVENT_E4_poincare_cocycle_result_missing")
    elif cocycle_residual > 1.0e-8:
        scientific_failures.append("EVENT_E4_poincare_cocycle_not_verified")
    if not _strict_sha256(e4.get("poincare_transition_artifact_hash")):
        event_blockers.append("EVENT_E4_poincare_transition_artifact_hash_missing")

    dag_blockers: list[str] = []
    dag = _mapping(report.get("semantic_event_dag"))
    required_identity_fields = {
        "canonical_semantic_payload",
        "observer_token",
        "visible_footprint",
        "semantic_causal_parents",
    }
    identity_fields = set(str(value) for value in _sequence(dag.get("identity_fields")))
    forbidden_identity_fields = {
        "worker_id",
        "queue_position",
        "retry_number",
        "repair_iteration",
        "wall_clock_timestamp",
        "file_offset",
        "object_uuid",
    }
    if identity_fields != required_identity_fields:
        dag_blockers.append("semantic_event_identity_field_contract_mismatch")
    if identity_fields.intersection(forbidden_identity_fields):
        dag_blockers.append("semantic_event_identity_uses_worker_retry_or_storage_metadata")
    declared_forbidden = set(
        str(value) for value in _sequence(dag.get("forbidden_identity_fields_present"))
    )
    if declared_forbidden:
        dag_blockers.append("semantic_event_identity_declares_forbidden_metadata")
    semantic_parent_edges = _integer(dag.get("semantic_parent_edge_count"))
    if semantic_parent_edges is None:
        dag_blockers.append("semantic_parent_edges_missing")
    elif semantic_parent_edges <= 0:
        scientific_failures.append("semantic_parent_edges_not_populated")
    if "acyclic" not in dag or _integer(dag.get("causal_cycle_count")) is None:
        dag_blockers.append("semantic_event_dag_acyclicity_result_missing")
    elif dag.get("acyclic") is not True or _integer(dag.get("causal_cycle_count")) != 0:
        scientific_failures.append("semantic_event_dag_is_not_acyclic")
    duplicate_count = _integer(dag.get("duplicate_semantic_event_count"))
    if duplicate_count is None:
        dag_blockers.append("semantic_event_duplicate_audit_missing")
    elif duplicate_count != 0:
        dag_blockers.append("semantic_event_dag_contains_duplicate_retry_events")
    if dag.get("preassigned_metric_used_for_identity") is not False:
        dag_blockers.append("preassigned_metric_used_for_semantic_event_identity")
    if not _strict_sha256(dag.get("semantic_event_dag_hash")):
        dag_blockers.append("semantic_event_dag_hash_missing")

    ancestry = _mapping(report.get("causal_ancestry"))
    ancestry_edges = _integer(ancestry.get("committed_read_after_write_edge_count"))
    if ancestry_edges is None:
        dag_blockers.append("committed_read_after_write_ancestry_count_missing")
    elif ancestry_edges <= 0:
        scientific_failures.append("committed_read_after_write_ancestry_absent")
    ancestry_coverage = _finite_number(
        ancestry.get("translation_edge_ancestry_coverage_fraction")
    )
    if ancestry_coverage is None or not (0.0 <= ancestry_coverage <= 1.0):
        dag_blockers.append("translation_ancestry_coverage_result_missing")
    elif ancestry_coverage < 1.0:
        scientific_failures.append("translation_edges_not_bound_to_causal_ancestry")
    if ancestry.get("population_used_as_reachability_surrogate") is not False:
        dag_blockers.append("event_population_used_as_causal_reachability_surrogate")
    if not _strict_sha256(ancestry.get("ancestry_artifact_hash")):
        dag_blockers.append("causal_ancestry_artifact_hash_missing")

    cone = _mapping(report.get("quadratic_event_cone"))
    cone_train_rows = _id_tuple(cone.get("fit_row_ids"))
    cone_holdout_rows = _id_tuple(cone.get("heldout_row_ids"))
    if not cone_train_rows or not cone_holdout_rows:
        event_blockers.append("quadratic_cone_train_and_holdout_rows_missing")
    elif set(cone_train_rows).intersection(cone_holdout_rows):
        event_blockers.append("quadratic_cone_train_holdout_rows_overlap")
    if cone.get("inference_source") != "semantic_event_relations":
        event_blockers.append("quadratic_cone_not_inferred_from_semantic_relations")
    if cone.get("preassigned_lorentz_metric_used") is not False:
        event_blockers.append("quadratic_cone_uses_preassigned_lorentz_metric")
    for field, expected, failure in (
        ("ambient_rank", 4, "quadratic_event_cone_rank_is_not_four"),
        ("negative_eigenvalue_count", 1, "quadratic_event_cone_not_one_timelike"),
        ("positive_eigenvalue_count", 3, "quadratic_event_cone_not_three_spacelike"),
        ("zero_eigenvalue_count", 0, "quadratic_event_cone_is_degenerate"),
    ):
        value = _integer(cone.get(field))
        if value is None:
            event_blockers.append(f"quadratic_cone_{field}_result_missing")
        elif value != expected:
            scientific_failures.append(failure)
    cone_residual = _finite_number(cone.get("heldout_quadratic_residual"))
    cone_threshold = _finite_number(cone.get("frozen_residual_threshold"))
    if cone_residual is None or cone_threshold is None or cone_threshold < 0.0:
        event_blockers.append("quadratic_cone_residual_or_threshold_missing")
    elif cone_residual > cone_threshold:
        scientific_failures.append("heldout_quadratic_cone_residual_failed")
    if "time_orientation_consistent" not in cone:
        event_blockers.append("quadratic_cone_time_orientation_result_missing")
    elif cone.get("time_orientation_consistent") is not True:
        scientific_failures.append("quadratic_cone_time_orientation_failed")
    cone_margin = _finite_number(cone.get("cofinal_normalized_margin_lower_bound"))
    if cone_margin is None:
        event_blockers.append("quadratic_cone_cofinal_margin_result_missing")
    elif cone_margin <= 0.0:
        scientific_failures.append("quadratic_cone_cofinal_margin_not_positive")
    if not _positive_integer(cone.get("cofinal_tail_level_count")):
        event_blockers.append("quadratic_cone_cofinal_tail_levels_missing")
    if not _strict_sha256(cone.get("cone_inference_artifact_hash")):
        event_blockers.append("quadratic_cone_inference_artifact_hash_missing")

    causality = _mapping(report.get("stable_causality"))
    if causality.get("time_function_source") != "source_derived_semantic_ancestry":
        event_blockers.append("stable_causality_time_function_not_source_derived")
    time_increment = _finite_number(causality.get("minimum_causal_edge_increment"))
    if time_increment is None:
        event_blockers.append("stable_causality_edge_increment_result_missing")
    elif time_increment <= 0.0:
        scientific_failures.append("stable_causality_time_function_not_strict")
    perturbation_margin = _finite_number(
        causality.get("time_function_perturbation_margin")
    )
    if perturbation_margin is None:
        event_blockers.append("stable_causality_perturbation_margin_missing")
    elif perturbation_margin <= 0.0:
        scientific_failures.append("stable_causality_perturbation_margin_failed")
    if not _strict_sha256(causality.get("stable_causality_artifact_hash")):
        event_blockers.append("stable_causality_artifact_hash_missing")

    completion = _mapping(report.get("record_cauchy_completion"))
    completion_residual = _finite_number(completion.get("refinement_cauchy_residual"))
    completion_threshold = _finite_number(completion.get("frozen_residual_threshold"))
    if completion_residual is None or completion_threshold is None:
        event_blockers.append("record_cauchy_residual_or_threshold_missing")
    elif completion_residual > completion_threshold:
        scientific_failures.append("record_family_not_cauchy_under_refinement")
    for field, failure in (
        (
            "every_cauchy_filter_has_record_germ",
            "record_germ_completion_surjectivity_failed",
        ),
        (
            "open_image_local_degree_nonzero",
            "event_chart_open_image_degree_failed",
        ),
    ):
        if field not in completion:
            event_blockers.append(f"record_cauchy_{field}_result_missing")
        elif completion.get(field) is not True:
            scientific_failures.append(failure)
    if not _strict_sha256(completion.get("completion_artifact_hash")):
        event_blockers.append("record_cauchy_completion_artifact_hash_missing")

    if report.get("h3_role") != "observer_frame_fiber":
        event_blockers.append("h3_not_typed_as_observer_frame_fiber")
    if report.get("event_base_role") != "event_position_manifold":
        event_blockers.append("event_base_not_typed_as_event_position_manifold")
    if report.get("h3_and_event_base_separate") is not True:
        event_blockers.append("h3_frame_fiber_and_event_base_not_separated")
    if not _strict_sha256(report.get("frame_fiber_construction_hash")):
        event_blockers.append("frame_fiber_construction_hash_missing")
    if not _strict_sha256(report.get("event_base_construction_hash")):
        event_blockers.append("event_base_construction_hash_missing")

    assumptions = _mapping(config.get("simulation_assumptions"))
    assumed = _mapping(assumptions.get("assumed"))
    h3_camera_assumed = bool(
        assumed.get("h3_observer_chart") or assumed.get("screen_observer_to_h3_camera_embedding")
    )
    blockers = [*event_blockers, *dag_blockers]
    hard_blockers: list[str] = []
    if event_blockers:
        hard_blockers.append("event_manifold_e1_e4_missing")
    if dag_blockers:
        hard_blockers.append("semantic_event_dag_missing")
    return _stage(
        blockers,
        {
            "event_clause_names": sorted(clauses),
            "event_clause_status": {
                "EVENT_E1_POPULATION": not any(
                    "EVENT_E1" in value for value in scientific_failures
                ),
                "EVENT_E2_SEPARATION": not any(
                    "EVENT_E2" in value for value in scientific_failures
                ),
                "EVENT_E3_RANK_FOUR": not any(
                    "EVENT_E3" in value for value in scientific_failures
                ),
                "EVENT_E4_POINCARE_COCYCLE": not any(
                    "EVENT_E4" in value for value in scientific_failures
                ),
            },
            "semantic_event_dag_passed": not dag_blockers,
            "scientific_outcome": (
                CellStatus.VALID_FAIL.value
                if scientific_failures
                else CellStatus.VALID_PASS.value
            ),
            "scientific_failures": scientific_failures,
            "heldout_quadratic_cone_passed": not any(
                "quadratic" in value or "cone" in value
                for value in scientific_failures
            ),
            "stable_causality_passed": not any(
                "stable_causality" in value for value in scientific_failures
            ),
            "record_cauchy_completion_passed": not any(
                "cauchy" in value
                or "record_germ_completion" in value
                or "open_image" in value
                for value in scientific_failures
            ),
            "h3_role": report.get("h3_role"),
            "event_base_role": report.get("event_base_role"),
            "config_contains_assumed_h3_camera_bridge": h3_camera_assumed,
        },
        hard_blockers=hard_blockers,
    )


def _campaign_stage(config: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    family_blockers: list[str] = []
    instrument_version = str(report.get("instrument_version") or "")
    campaign_id = str(report.get("campaign_id") or "")
    if instrument_version != DEFAULT_INSTRUMENT_VERSION:
        family_blockers.append("instrument_version_missing_or_mismatched")
    if not campaign_id:
        family_blockers.append("campaign_id_missing")

    family_contract = _mapping(report.get("family_contract"))
    if set(family_contract) != REQUIRED_FAMILY_CONTRACT_KEYS:
        family_blockers.append("family_contract_key_set_mismatch")
    computed_family_hash = _canonical_sha256(family_contract)
    campaign_family_hash = str(report.get("campaign_family_hash") or "")
    if not _strict_sha256(campaign_family_hash) or campaign_family_hash != computed_family_hash:
        family_blockers.append("campaign_family_hash_mismatch")
    instrument_commit = str(family_contract.get("instrument_commit") or "")
    if len(instrument_commit) < 7:
        family_blockers.append("instrument_commit_missing")
    if not _strict_sha256(family_contract.get("container_digest")):
        family_blockers.append("container_digest_missing_or_malformed")
    control_set = _mapping(family_contract.get("control_set"))
    if tuple(control_set.get("clock_candidates", ())) != REQUIRED_CLOCK_LABELS:
        family_blockers.append("family_clock_control_set_mismatch")
    if tuple(control_set.get("geometry_models", ())) != REQUIRED_GEOMETRY_MODELS:
        family_blockers.append("family_geometry_control_set_mismatch")
    archive_boundary = _mapping(family_contract.get("archive_boundary"))
    if archive_boundary != {
        "archived_instrument_status": "FROZEN_NO_RETUNE",
        "archived_16k_failure_preserved": True,
        "new_instrument_is_distinct_family": True,
        "archived_outcomes_used_for_threshold_selection": False,
    }:
        family_blockers.append("archived_16k_no_retune_boundary_mismatch")

    rung_scaling_laws = _mapping(family_contract.get("rung_scaling_laws"))
    if rung_scaling_laws.get("carrier_count") != "exact_federation_cardinality":
        family_blockers.append("carrier_rung_scaling_law_mismatch")
    if "nominal_to_actual_icosahedral_cell_counts" in rung_scaling_laws:
        family_blockers.append("carrier_rungs_must_not_be_remapped_to_s2_cell_counts")

    seeds = [_integer(value) for value in _sequence(report.get("seeds"))]
    seeds = [value for value in seeds if value is not None]
    rungs = [_integer(value) for value in _sequence(report.get("rungs"))]
    rungs = [value for value in rungs if value is not None]
    replicate_ids = [str(value) for value in _sequence(report.get("replicate_ids"))]
    replicate_ids = [value for value in replicate_ids if value]
    if len(set(seeds)) < MINIMUM_INDEPENDENT_SOURCE_SEEDS:
        blockers.append("at_least_three_frozen_independent_source_seeds_required")
    if tuple(sorted(set(rungs))) != REQUIRED_RUNGS:
        blockers.append("frozen_rungs_must_be_4k_16k_64k_256k")
    carrier_counts = [
        value
        for value in (
            _integer(item) for item in _sequence(report.get("carrier_counts"))
        )
        if value is not None
    ]
    if tuple(carrier_counts) != REQUIRED_CARRIER_COUNTS:
        family_blockers.append("exact_carrier_counts_mismatch")
    if not replicate_ids or len(set(replicate_ids)) != len(replicate_ids):
        blockers.append("nonempty_unique_replicate_ids_required")
    if report.get("frozen_before_first_run") is not True:
        family_blockers.append("campaign_not_frozen_before_first_run")
    if report.get("retune_after_freeze") is not False:
        family_blockers.append("retune_after_freeze_not_explicitly_forbidden")
    if _sequence(report.get("retune_events")):
        family_blockers.append("retune_events_present")

    run_rows = [dict(row) for row in _sequence(report.get("run_matrix")) if isinstance(row, Mapping)]
    expected_cells = {
        (seed, rung, replicate_id)
        for seed in set(seeds)
        for rung in set(rungs)
        for replicate_id in set(replicate_ids)
    }
    observed_cells: set[tuple[int, int, str]] = set()
    protocol_hashes: set[str] = set()
    statuses: list[str] = []
    for index, row in enumerate(run_rows):
        seed = _integer(row.get("seed"))
        rung = _integer(row.get("rung"))
        replicate_id = str(row.get("replicate_id") or "")
        if seed is not None and rung is not None and replicate_id:
            observed_cells.add((seed, rung, replicate_id))
        carrier_count = _integer(row.get("carrier_count"))
        if rung is None or carrier_count != rung:
            family_blockers.append(f"run_matrix_{index}_carrier_count_mismatch")
        protocol_hash = str(row.get("protocol_hash") or "")
        if not _strict_sha256(protocol_hash):
            blockers.append(f"run_matrix_{index}_protocol_hash_missing")
        else:
            protocol_hashes.add(protocol_hash)
        if not _strict_sha256(row.get("frozen_config_hash")):
            blockers.append(f"run_matrix_{index}_frozen_config_hash_missing")
        for field, expected in (
            ("campaign_id", campaign_id),
            ("instrument_version", instrument_version),
            ("instrument_commit", instrument_commit),
            ("family_hash", campaign_family_hash),
        ):
            if str(row.get(field) or "") != expected or not expected:
                family_blockers.append(f"run_matrix_{index}_{field}_mismatch")
        status = str(row.get("status") or "")
        statuses.append(status)
        if status not in {value.value for value in CellStatus}:
            family_blockers.append(f"run_matrix_{index}_unknown_cell_status")
        if status == CellStatus.INSTRUMENT_INVALID.value:
            family_blockers.append(f"run_matrix_{index}_instrument_invalid")
        if status == CellStatus.VALID_FAIL.value:
            if not str(row.get("failure_mode") or ""):
                family_blockers.append(f"run_matrix_{index}_valid_fail_mode_missing")
            if not _strict_sha256(row.get("failure_evidence_hash")):
                family_blockers.append(
                    f"run_matrix_{index}_valid_fail_evidence_hash_missing"
                )
        if status in {CellStatus.VALID_FAIL.value, CellStatus.VALID_PASS.value}:
            if row.get("preflight") != "PASS":
                family_blockers.append(f"run_matrix_{index}_valid_cell_without_passed_preflight")
            if row.get("required_controls_complete") is not True:
                family_blockers.append(f"run_matrix_{index}_valid_cell_controls_incomplete")
            if row.get("source_hashes_complete") is not True:
                family_blockers.append(f"run_matrix_{index}_valid_cell_source_hashes_incomplete")
    if observed_cells != expected_cells or not expected_cells:
        blockers.append("run_matrix_does_not_cover_every_seed_rung_replicate_cell")
    if len(run_rows) != len(expected_cells):
        blockers.append("run_matrix_requires_exactly_one_row_per_seed_rung_replicate_cell")
    if len(protocol_hashes) != 1:
        blockers.append("run_matrix_protocol_hash_not_single_and_frozen")

    blockers.extend(family_blockers)
    if family_blockers or any(status == CellStatus.INSTRUMENT_INVALID.value for status in statuses):
        campaign_status = CellStatus.INSTRUMENT_INVALID.value
    elif (
        len(statuses) != len(expected_cells)
        or any(
            status in {CellStatus.NOT_EVALUATED.value, CellStatus.INCOMPLETE.value}
            for status in statuses
        )
    ):
        campaign_status = CellStatus.INCOMPLETE.value
    elif statuses and all(status == CellStatus.VALID_PASS.value for status in statuses):
        campaign_status = CellStatus.VALID_PASS.value
    elif statuses and all(
        status in {CellStatus.VALID_PASS.value, CellStatus.VALID_FAIL.value}
        for status in statuses
    ):
        campaign_status = CellStatus.VALID_FAIL.value
    else:
        campaign_status = CellStatus.INCOMPLETE.value

    retirement_rule = _mapping(family_contract.get("retirement_rule"))
    decisive_rungs = tuple(
        sorted(
            value
            for value in (_integer(item) for item in _sequence(retirement_rule.get("decisive_rungs")))
            if value is not None
        )
    )
    expected_decisive_rungs = (16_384, 65_536, 262_144)
    decisive_rows = [row for row in run_rows if _integer(row.get("rung")) in decisive_rungs]
    failure_modes = {
        str(row.get("failure_mode"))
        for row in decisive_rows
        if str(row.get("failure_mode") or "")
    }
    stable_failure_rule_satisfied = bool(
        campaign_status == CellStatus.VALID_FAIL.value
        and decisive_rungs == expected_decisive_rungs
        and len(decisive_rows)
        == len(set(seeds)) * len(decisive_rungs) * len(set(replicate_ids))
        and all(row.get("status") == CellStatus.VALID_FAIL.value for row in decisive_rows)
        and all(row.get("powered_and_complete") is True for row in decisive_rows)
        and all(_strict_sha256(row.get("failure_evidence_hash")) for row in decisive_rows)
        and len(failure_modes) == 1
        and retirement_rule.get("same_predeclared_failure_mode_required") is True
        and retirement_rule.get("all_cells_powered_and_complete_required") is True
        and retirement_rule.get("frozen_before_first_run") is True
    )

    config_seed = _integer(config.get("seed"))
    source_federation = _mapping(config.get("source_federation"))
    config_rung = _integer(source_federation.get("carrier_count"))
    hard_blockers: list[str] = []
    if family_blockers:
        hard_blockers.append("frozen_config_family_mismatch")
    if len(set(seeds)) < MINIMUM_INDEPENDENT_SOURCE_SEEDS:
        hard_blockers.append("insufficient_independent_source_seeds")
    return _stage(
        blockers,
        {
            "seeds": sorted(set(seeds)),
            "rungs": sorted(set(rungs)),
            "carrier_counts": carrier_counts,
            "required_carrier_counts": list(REQUIRED_CARRIER_COUNTS),
            "replicate_ids": sorted(set(replicate_ids)),
            "run_matrix_count": len(run_rows),
            "expected_run_matrix_count": len(expected_cells),
            "single_protocol_hash": len(protocol_hashes) == 1,
            "instrument_version": instrument_version or None,
            "campaign_id": campaign_id or None,
            "campaign_family_hash": campaign_family_hash or None,
            "computed_campaign_family_hash": computed_family_hash,
            "campaign_status": campaign_status,
            "cell_status_counts": {
                status.value: statuses.count(status.value) for status in CellStatus
            },
            "stable_failure_rule_satisfied": stable_failure_rule_satisfied,
            "decisive_rungs": list(decisive_rungs),
            "stable_failure_modes": sorted(failure_modes),
            "config_seed": config_seed,
            "config_rung": config_rung,
        },
        hard_blockers=hard_blockers,
    )


def _dependency_graph(stages: Mapping[str, Mapping[str, Any]], receipt: bool) -> dict[str, Any]:
    """Expose this preflight as a typed node in the full emergence ladder."""

    source_stage = _mapping(stages.get("P0_source_dynamics_repair_record_observer"))
    source_evidence = _mapping(source_stage.get("evidence"))
    source_status = _mapping(source_evidence.get("dependency_status"))
    upstream_ids = [
        "SOURCE_PATCH_ARCHITECTURE",
        "LOCAL_REPAIR_DYNAMICS",
        "OBSERVER_SELF_READING_RECORD_LOOP",
        "SOURCE_GENERATOR_TARGET_FREE",
    ]
    nodes: dict[str, dict[str, Any]] = {}
    for dependency_id in upstream_ids:
        passed = source_status.get(dependency_id) is True
        nodes[dependency_id] = {
            "dependency_id": dependency_id,
            "layer": "source_observer",
            "passed": passed,
            "status": "satisfied" if passed else "blocked",
            "blockers": [] if passed else [f"{dependency_id}_primitive_receipt_required"],
        }

    nodes[PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT] = {
        "dependency_id": PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT,
        "layer": "geometry",
        "passed": receipt,
        "status": "satisfied" if receipt else "blocked",
        "blockers": []
        if receipt
        else [name for name, stage in stages.items() if stage.get("passed") is not True],
    }

    downstream_specs = {
        "EVENT_MANIFOLD_3P1D_RECEIPT": [
            PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT,
            "independent_event_position_manifold_promotion_receipt",
        ],
        "PHYSICAL_STANDARD_MODEL_PROMOTION_RECEIPT": [
            "EVENT_MANIFOLD_3P1D_RECEIPT",
            "source_derived_gauge_matter_selector_receipt",
            "independent_standard_model_promotion_receipt",
        ],
        "OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1": [
            "EVENT_MANIFOLD_3P1D_RECEIPT",
            "finite_cap_entropy_stress_area_bridge_receipts",
            "independent_einstein_branch_entry_receipt",
        ],
        "PHYSICAL_GRAVITY_PREDICTION_RECEIPT": [
            "OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1",
            "production_gravity_observable_receipt",
        ],
    }
    downstream_blockers: dict[str, list[str]] = {}
    for dependency_id, prerequisites in downstream_specs.items():
        blockers = list(prerequisites)
        if receipt and PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT in blockers:
            blockers.remove(PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT)
        downstream_blockers[dependency_id] = blockers
        nodes[dependency_id] = {
            "dependency_id": dependency_id,
            "layer": "event_manifold"
            if dependency_id == "EVENT_MANIFOLD_3P1D_RECEIPT"
            else "standard_model"
            if dependency_id == "PHYSICAL_STANDARD_MODEL_PROMOTION_RECEIPT"
            else "gravity",
            "passed": False,
            "status": "blocked",
            "blockers": blockers,
            "not_established_by_this_preflight": True,
        }

    edges = [
        {"from": "SOURCE_PATCH_ARCHITECTURE", "to": "LOCAL_REPAIR_DYNAMICS"},
        {"from": "SOURCE_GENERATOR_TARGET_FREE", "to": "LOCAL_REPAIR_DYNAMICS"},
        {"from": "LOCAL_REPAIR_DYNAMICS", "to": "OBSERVER_SELF_READING_RECORD_LOOP"},
        {
            "from": "OBSERVER_SELF_READING_RECORD_LOOP",
            "to": PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT,
        },
        {"from": "SOURCE_GENERATOR_TARGET_FREE", "to": PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT},
        {
            "from": PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT,
            "to": "EVENT_MANIFOLD_3P1D_RECEIPT",
        },
        {
            "from": "EVENT_MANIFOLD_3P1D_RECEIPT",
            "to": "PHYSICAL_STANDARD_MODEL_PROMOTION_RECEIPT",
        },
        {
            "from": "EVENT_MANIFOLD_3P1D_RECEIPT",
            "to": "OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1",
        },
        {
            "from": "OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1",
            "to": "PHYSICAL_GRAVITY_PREDICTION_RECEIPT",
        },
    ]
    return {
        "schema_version": "oph_emergence_dependency_graph_v1",
        "gate_id": PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT,
        "upstream_dependency_ids": upstream_ids,
        "downstream_dependency_ids": list(downstream_specs),
        "nodes": nodes,
        "edges": edges,
        "downstream_blockers": downstream_blockers,
        "claim_order": [
            "source_architecture",
            "local_repair",
            "self_reading_observer",
            "h3_kms_geometry",
            "event_manifold",
            "standard_model_and_gravity",
        ],
    }


def _stage(
    blockers: Sequence[str],
    evidence: Mapping[str, Any],
    *,
    hard_blockers: Sequence[str] = (),
) -> dict[str, Any]:
    unique_blockers = list(dict.fromkeys(str(value) for value in blockers))
    return {
        "passed": not unique_blockers,
        "blockers": unique_blockers,
        "hard_blockers": list(dict.fromkeys(str(value) for value in hard_blockers)),
        "evidence": dict(evidence),
    }


def _report(reports: Mapping[str, Any], name: str) -> dict[str, Any]:
    return _mapping(reports.get(name))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _named_rows(value: Any, name_key: str) -> dict[str, dict[str, Any]]:
    if isinstance(value, Mapping):
        return {str(name): _mapping(row) for name, row in value.items()}
    rows: dict[str, dict[str, Any]] = {}
    for raw in _sequence(value):
        row = _mapping(raw)
        name = str(row.get(name_key) or "")
        if name:
            rows[name] = row
    return rows


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if float(value) != float(number):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return number


def _positive_integer(value: Any) -> bool:
    number = _integer(value)
    return number is not None and number > 0


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _near(value: Any, target: float, tolerance: float) -> bool:
    number = _finite_number(value)
    return number is not None and abs(number - target) <= tolerance


def _at_most(value: Any, upper: float) -> bool:
    number = _finite_number(value)
    return number is not None and 0.0 <= number <= upper


def _id_tuple(value: Any) -> tuple[str, ...]:
    rows = tuple(str(item) for item in _sequence(value))
    if not rows or any(not item for item in rows) or len(rows) != len(set(rows)):
        return ()
    return rows


def _nonempty_hash(value: Any) -> bool:
    text = str(value or "").strip()
    return len(text) >= 8 and bool(re.fullmatch(r"(?:sha256:)?[A-Za-z0-9_.:-]+", text))


def _strict_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        payload = b"__NON_CANONICAL_JSON__"
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _token_hits(values: Sequence[str], tokens: Sequence[str]) -> list[str]:
    hits: list[str] = []
    for value in values:
        normalized = str(value).lower()
        for token in tokens:
            if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", normalized):
                hits.append(f"{token}@{value}")
    return sorted(set(hits))


__all__ = [
    "PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT",
    "CellStatus",
    "DEFAULT_INSTRUMENT_VERSION",
    "MINIMUM_INDEPENDENT_SOURCE_SEEDS",
    "REQUIRED_CARRIER_COUNTS",
    "REQUIRED_CLOCK_LABELS",
    "REQUIRED_GEOMETRY_MODELS",
    "REQUIRED_RUNGS",
    "physical_h3_kms_preflight_report",
    "write_physical_h3_kms_preflight_report",
]
