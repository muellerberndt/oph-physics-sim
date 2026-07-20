"""Deterministic post-run evidence producers for the physical H3/KMS campaign.

This module is intentionally downstream of numerical evolution.  It consumes
one target-blind primitive capture and a separately frozen campaign plan.  The
clock candidates are introduced only while scoring an already hashed response;
they never enter the source trajectory or intervention packet.

The public entry point returns deterministic diagnostic mappings for P1--P8.
Artifact conformance, algebraic identities, post-capture coordinate
constructions, and synthetic sensitivity exercises are deliberately separated
from physical measurements.  A physical ``VALID_PASS`` or ``VALID_FAIL`` is
possible only after the source capture contains the independently produced,
typed observables required by that stage.  Missing producers therefore yield
``NOT_EVALUATED`` and never manufacture a scientific failure.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from oph_fpe.bulk.bw_native_preflight import (
    BW_NATIVE_SCHEMA_VERSION,
    BW_PRIMITIVE_FIELD_CONTRACT,
    canonical_payload_hash,
    native_bw01_bw08_report,
)
from oph_fpe.bulk.physical_h3_kms_preflight import (
    DEFAULT_INSTRUMENT_VERSION,
    REQUIRED_CLOCK_LABELS,
    REQUIRED_GEOMETRY_MODELS,
    REQUIRED_RUNGS,
    CellStatus,
)
from oph_fpe.bulk.physical_h3_kms_prerun import frozen_campaign_family_sha256
POSTRUN_SCHEMA_VERSION = "oph.physical_h3_kms.postrun.v1"
CAPTURE_SCHEMA_VERSION = "oph.physical-source-capture/1.0.0"
PREREGISTRATION_ENVELOPE_SCHEMA = "oph.physical-h3-kms.postrun-preregistration.v1"
_PREREGISTRATION_ENVELOPE_KEYS = {
    "schema",
    "preregistration",
    "preregistration_report",
    "preregistration_sha256",
}
_PLAN_KEYS = {
    "schema",
    "campaign_id",
    "instrument_version",
    "instrument_commit_sha256",
    "seeds",
    "rungs",
    "replicate_ids",
    "clock_candidates",
    "geometry_models",
    "thresholds",
    "calibrations",
    "split_contract",
    "scaling_contract",
    "archive_boundary",
    "producer_registry",
    "checker_registry",
    "run_matrix",
    "current_cell",
    "plan_sha256",
}
_THRESHOLD_KEYS = {
    "clock_absolute_residual_max",
    "clock_win_margin_min",
    "geometry_win_margin_min",
    "curvature_minimum_power",
}
_CALIBRATION_KEYS = {
    "clock_calibration_sha256",
    "geometry_calibration_sha256",
    "curvature_calibration_sha256",
    "independent_of_campaign_source_seeds",
    "frozen_before_source_capture",
    "physical_threshold_calibration_receipt",
}
_SPLIT_KEYS = {
    "algorithm_id",
    "assignment_salt_sha256",
    "holdout_fraction",
    "derivation",
    "heldout_ids_materialized_before_capture",
}
_DERIVED_CAPTURE_KEYS = {
    "carrier_port_trajectories",
    "intervention_rows",
    "response_rows",
    "clock_pair_input",
    "geometry_control_rows",
    "geometry_samples",
    "semantic_events",
    "raw_overlap_relations",
    "raw_ancestry_relations",
}

_CLOCK_VALUES = {
    "1x": 1.0,
    "pi": math.pi,
    "2pi": 2.0 * math.pi,
    "4pi": 4.0 * math.pi,
}
_FORBIDDEN_SOURCE_TOKENS = {
    "1x",
    "pi",
    "2pi",
    "4pi",
    "kms",
    "candidate",
    "normalization",
    "preferred_geometry",
    "target_geometry",
    "target",
}
_FORBIDDEN_ASSERTION_KEYS = {
    "pass",
    "passed",
    "receipt",
    "selected_model",
    "selected_scale",
    "scientific_outcome",
}
_TARGET_TOKEN_NORMALIZER = re.compile(r"[^a-z0-9]+")

_P1_REASON = (
    "nested_icosahedral_support_tower_is_an_independent_regulator_and_is_not_"
    "produced_by_source_repair_normal_forms"
)
_P0_PHYSICAL_SOURCE_REQUIREMENTS = (
    (
        "PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT",
        "physical_echosahedral_federation_realization_not_established",
    ),
    (
        "CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT",
        "carrier_to_support_chart_realization_not_established",
    ),
    (
        "CARRIER_REFINEMENT_NATURALITY_RECEIPT",
        "carrier_refinement_naturality_not_established",
    ),
)
_P2_REASON = (
    "m4_density_state_and_modular_generator_are_definitionally_constructed_"
    "from_one_source_snapshot_without_an_independent_physical_state_producer"
)
_P3_REASON = (
    "cross_ratio_parameter_is_a_target_blind_postcapture_coordinate_but_not_"
    "an_independently_normalized_geometric_flow_observable"
)
_P4_REASON = (
    "bw_payload_is_a_structural_postcapture_reconstruction_without_a_"
    "cofinal_physical_state_tower_and_independent_modular_geometric_pair"
)
_P6_REASON = (
    "independent_event_geometry_and_replay_bound_physical_curvature_"
    "calibration_producers_are_not_instantiated"
)
_P7_REASON = (
    "event_positions_boxes_cone_and_frame_coordinates_are_postcapture_"
    "diagnostic_constructions_without_an_independent_event_manifold_producer"
)

# P7 is a non-promotional post-capture diagnostic until an independent event
# manifold producer exists.  Keep its pairwise sensitivity calculation bounded
# so observer scaling does not create an O(E^2) evidence artifact.  Below this
# bound the sampler is an exact lexicographic census, which is useful for exact
# small-instance verification.
_P7_MAX_DIAGNOSTIC_PAIR_COUNT = 16_384
_P7_PAIR_SAMPLE_ALGORITHM = "affine_permutation_pair_rank_sample_v1"


class PostrunCaptureError(ValueError):
    """Raised when a purported primitive capture is malformed or target-leaky."""


def _not_evaluated_stage_assessment(
    *,
    reason: str,
    structural_complete: bool,
    structural_artifact_hash: str | None,
    instrument_producer_available: bool,
    sensitivity_complete: bool,
    sensitivity_artifact_hash: str | None,
    diagnostic_findings: Sequence[str],
) -> dict[str, Any]:
    """Build a typed, non-promotional epistemic assessment.

    ``structural_receipt`` proves only deterministic artifact integrity.
    ``instrument_receipt`` records whether the independent physical producer
    required by the paper was present.  ``sensitivity_receipt`` may summarize
    a diagnostic calculation, but is explicitly ineligible for a physical
    gate.  Keeping the three scopes separate prevents a true algebraic identity
    or a failed synthetic exercise from becoming a physics result.
    """

    return {
        "measurement_status": CellStatus.NOT_EVALUATED.value,
        "physical_gate_eligible": False,
        "not_evaluated_reasons": [reason],
        "scientific_failures": [],
        "structural_receipt": {
            "scope": "artifact_integrity_only",
            "status": "COMPLETE" if structural_complete else "INCOMPLETE",
            "artifact_hash": structural_artifact_hash,
            "physical_claim": False,
        },
        "instrument_receipt": {
            "scope": "independent_physical_producer",
            "status": (
                "AVAILABLE"
                if instrument_producer_available
                else "MISSING_REQUIRED_PRODUCER"
            ),
            "physical_claim": False,
        },
        "sensitivity_receipt": {
            "scope": "diagnostic_sensitivity_only",
            "status": "COMPLETE" if sensitivity_complete else "NOT_RUN",
            "artifact_hash": sensitivity_artifact_hash,
            "physical_gate_eligible": False,
            "diagnostic_findings": list(diagnostic_findings),
        },
    }


def _upstream_epistemic_assessments(
    source_capture: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    reports = _plain_mapping(source_capture.get("reports"), "source_capture.reports")
    refinement = _plain_mapping(reports.get("refinement"), "source refinement report")
    cap_state = _plain_mapping(
        reports.get("prime_geometric_state"), "source prime geometric state report"
    )
    geometry = _plain_mapping(
        reports.get("independent_geometry"), "source independent geometry report"
    )
    return {
        "P1_nested_refinement_and_expectations": _not_evaluated_stage_assessment(
            reason=_P1_REASON,
            structural_complete=bool(
                _strict_hash(refinement.get("certificate_sha256"))
                and _sequence(refinement.get("levels"))
                and _sequence(refinement.get("conditional_expectations"))
            ),
            structural_artifact_hash=_hash(refinement),
            instrument_producer_available=False,
            sensitivity_complete=False,
            sensitivity_artifact_hash=None,
            diagnostic_findings=[],
        ),
        "P2_prime_geometric_cap_state": _not_evaluated_stage_assessment(
            reason=_P2_REASON,
            structural_complete=bool(
                _strict_hash(cap_state.get("raw_primitives_sha256"))
                and isinstance(cap_state.get("rho"), Mapping)
                and isinstance(cap_state.get("modular_generator"), Mapping)
            ),
            structural_artifact_hash=_hash(cap_state),
            instrument_producer_available=False,
            sensitivity_complete=False,
            sensitivity_artifact_hash=None,
            diagnostic_findings=[],
        ),
        "P3_independent_geometric_parameter": _not_evaluated_stage_assessment(
            reason=_P3_REASON,
            structural_complete=bool(
                _strict_hash(geometry.get("geometry_derivation_hash"))
                and _sequence(geometry.get("raw_primitive_rows"))
            ),
            structural_artifact_hash=_hash(geometry),
            instrument_producer_available=False,
            sensitivity_complete=False,
            sensitivity_artifact_hash=None,
            diagnostic_findings=[],
        ),
    }


def _p0_not_evaluated_reasons(
    source_capture: Mapping[str, Any],
) -> list[str]:
    """Return physical-source gaps that software replay cannot discharge.

    P0 is checked directly from the source report rather than copied into the
    P1--P8 postrun ledger.  This keeps the source contract independently
    replayable while preventing a future completion of P1--P7 from promoting a
    cell whose carrier realization/naturality bridge is still absent.
    """

    reports = _plain_mapping(source_capture.get("reports"), "source_capture.reports")
    source = _plain_mapping(reports.get("source_observer"), "source observer report")
    return [
        reason
        for receipt, reason in _P0_PHYSICAL_SOURCE_REQUIREMENTS
        if source.get(receipt) is not True
    ]


def _clock_stage_assessment(
    report: Mapping[str, Any], stats: Mapping[str, Any]
) -> dict[str, Any]:
    if stats.get("evaluated") is not True:
        reasons = [
            str(item) for item in stats.get("not_evaluated_reasons", [])
        ] or ["typed_modular_geometric_pair_not_available"]
        assessment = _not_evaluated_stage_assessment(
            reason=reasons[0],
            structural_complete=True,
            structural_artifact_hash=_hash(report),
            instrument_producer_available=(
                stats.get("typed_producers_available") is True
            ),
            sensitivity_complete=stats.get("sensitivity_complete") is True,
            sensitivity_artifact_hash=stats.get("sensitivity_artifact_hash"),
            diagnostic_findings=[
                str(value) for value in stats.get("diagnostic_findings", [])
            ],
        )
        assessment["not_evaluated_reasons"] = reasons
        return assessment
    failures = [str(item) for item in stats.get("scientific_failures", [])]
    return {
        "measurement_status": (
            CellStatus.VALID_FAIL.value if failures else CellStatus.VALID_PASS.value
        ),
        "physical_gate_eligible": True,
        "not_evaluated_reasons": [],
        "scientific_failures": failures,
        "structural_receipt": {
            "scope": "artifact_integrity_only",
            "status": "COMPLETE",
            "artifact_hash": _hash(report),
            "physical_claim": False,
        },
        "instrument_receipt": {
            "scope": "independent_physical_producer",
            "status": "AVAILABLE",
            "physical_claim": False,
        },
        "sensitivity_receipt": {
            "scope": "registered_physical_candidate_comparison",
            "status": "COMPLETE",
            "artifact_hash": _hash(stats),
            "physical_gate_eligible": True,
            "diagnostic_findings": [],
        },
    }


def compute_postrun_reports(
    capture: Mapping[str, Any],
    campaign_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute P4--P8 reports from a target-blind source capture.

    Required capture collections are ``carrier_port_trajectories``,
    ``intervention_rows``, ``response_rows``, ``geometry_samples``,
    ``geometry_control_rows``, and ``semantic_events``.  The campaign plan owns
    all candidates, splits, thresholds, model capacity, seeds, and rungs.
    Caller-provided pass flags are rejected rather than copied.
    """

    return _compute_postrun_reports(
        capture,
        campaign_plan,
        verify_source_replay=True,
    )


def _compute_postrun_reports_from_verified_source(
    capture: Mapping[str, Any],
    campaign_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute reports after the caller completed exact source replay.

    This private path exists only to prevent the disk-replay layer from
    executing the same source evolution twice in immediate succession.  It
    still revalidates the preregistration, source/cell bindings, postrun
    component hashes, finite-JSON contract, and target firewall.  The replay
    layer must call :func:`verify_physical_source_capture` successfully before
    entering here.
    """

    return _compute_postrun_reports(
        capture,
        campaign_plan,
        verify_source_replay=False,
    )


def _compute_postrun_reports(
    capture: Mapping[str, Any],
    campaign_plan: Mapping[str, Any],
    *,
    verify_source_replay: bool,
) -> dict[str, Any]:
    preregistration, preregistration_report, plan = _verified_preregistration(
        campaign_plan
    )
    source_capture = (
        _verified_source_capture(capture)
        if verify_source_replay
        else _source_capture_envelope(capture)
    )
    _bind_source_to_preregistered_cell(source_capture, preregistration)
    postrun_capture = _derive_postrun_capture(source_capture, preregistration)
    if set(postrun_capture) != _DERIVED_CAPTURE_KEYS:
        raise PostrunCaptureError("internally derived postrun capture field set mismatch")
    components = {
        name: _required_sequence(postrun_capture, name)
        for name in (
            "carrier_port_trajectories",
            "intervention_rows",
            "response_rows",
            "geometry_control_rows",
            "semantic_events",
            "raw_overlap_relations",
            "raw_ancestry_relations",
        )
    }
    clock_pair_input = _plain_mapping(
        postrun_capture.get("clock_pair_input"), "clock_pair_input"
    )
    geometry_samples = _plain_mapping(
        postrun_capture.get("geometry_samples"), "geometry_samples"
    )
    source_material = {
        **components,
        "clock_pair_input": clock_pair_input,
        "geometry_samples": geometry_samples,
    }
    _require_finite_json(source_material, path="postrun_capture")
    leak_hits = _target_leak_hits(source_material)
    if leak_hits:
        raise PostrunCaptureError("target-bearing source capture: " + ", ".join(leak_hits[:8]))

    component_hashes = {
        name: _hash(value) for name, value in source_material.items()
    }
    source_artifacts = _plain_mapping(
        source_capture.get("source_artifacts"), "source_capture.source_artifacts"
    )
    cap_state = _plain_mapping(
        source_artifacts.get("cap_state_raw_primitives"),
        "source cap_state_raw_primitives",
    )
    component_hashes["cap_state_raw_primitives"] = _hash(cap_state)
    candidate_report, clock_stats = _candidate_report(
        components["carrier_port_trajectories"],
        components["intervention_rows"],
        components["response_rows"],
        clock_pair_input,
        plan,
    )
    native_bw = _unavailable_native_bw_payload(component_hashes)
    geometry_controls, geometry_stats = _geometry_control_report(
        components["geometry_control_rows"], plan, component_hashes
    )
    semantic_event, event_stats = _semantic_event_report(
        components["semantic_events"],
        components["raw_overlap_relations"],
        components["raw_ancestry_relations"],
        plan,
    )

    upstream_epistemics = _upstream_epistemic_assessments(source_capture)
    native_verification = native_bw01_bw08_report(native_bw)
    native_input_contract_complete = bool(
        native_bw.get("status") == "UNAVAILABLE"
        and native_bw.get("physical_gate_eligible") is False
        and _strict_hash(native_bw.get("antecedent_hash"))
        and _sequence(native_bw.get("missing_producers"))
    )
    native_diagnostic_findings = [
        "native_bw_physical_input_unavailable",
        *(
            f"missing_producer:{value}"
            for value in _sequence(native_bw.get("missing_producers"))
        ),
    ]
    native_bw_diagnostic = {
        "measurement_status": CellStatus.NOT_EVALUATED.value,
        "physical_gate_eligible": False,
        "not_evaluated_reasons": [_P4_REASON],
        "structural_conformance_complete": (
            native_verification.get("native_payload_conformance_receipt") is True
        ),
        "unavailable_input_contract_complete": native_input_contract_complete,
        "required_clause_ids": native_verification.get("required_clause_ids", []),
        "clause_predicate_diagnostics": native_verification.get("clauses", {}),
        "diagnostic_findings": native_diagnostic_findings,
        "diagnostic_artifact_hash": _hash(native_verification),
        "scientific_failures": [],
    }
    p4_epistemics = _not_evaluated_stage_assessment(
        reason=_P4_REASON,
        structural_complete=native_input_contract_complete,
        structural_artifact_hash=_hash(native_bw),
        instrument_producer_available=False,
        sensitivity_complete=False,
        sensitivity_artifact_hash=None,
        diagnostic_findings=native_diagnostic_findings,
    )
    stage_epistemics = {
        **upstream_epistemics,
        "P4_native_bw01_bw08": p4_epistemics,
        "P5_frozen_candidate_interventions": _clock_stage_assessment(
            candidate_report, clock_stats
        ),
        "P6_h3_s2_e3_e4_same_holdout_and_curvature_leverage": (
            geometry_stats["stage_assessment"]
        ),
        "P7_semantic_event_e1_e4_and_frame_fiber_separation": (
            event_stats["stage_assessment"]
        ),
    }
    # Aggregate only typed stage outcomes.  Diagnostic findings remain in the
    # sensitivity receipts and cannot become scientific failures.  Including
    # every P1--P7 assessment here also prevents future producer additions from
    # being silently ignored by the campaign-cell status.
    scientific_failures = list(
        dict.fromkeys(
            str(item)
            for stage in stage_epistemics.values()
            for item in stage.get("scientific_failures", [])
        )
    )
    not_evaluated_reasons = list(
        dict.fromkeys(
            [
                *_p0_not_evaluated_reasons(source_capture),
                *(
                    str(item)
                    for stage in stage_epistemics.values()
                    for item in stage.get("not_evaluated_reasons", [])
                ),
            ]
        )
    )
    stage_epistemics["P8_frozen_multiseed_four_rung_campaign"] = (
        _not_evaluated_stage_assessment(
            reason="campaign_matrix_contains_unmeasured_physical_stages",
            structural_complete=True,
            structural_artifact_hash=_hash(plan.get("run_matrix")),
            instrument_producer_available=False,
            sensitivity_complete=False,
            sensitivity_artifact_hash=None,
            diagnostic_findings=[],
        )
    )
    campaign = _campaign_manifest(
        plan,
        preregistration,
        component_hashes,
        scientific_failures=scientific_failures,
        evaluation_complete=not not_evaluated_reasons,
    )
    source_capture_hash = str(source_capture["capture_sha256"])
    return {
        "schema": POSTRUN_SCHEMA_VERSION,
        "source_capture_hash": source_capture_hash,
        "source_capture_target_blind": True,
        "source_capture_replay_receipt": True,
        "source_root_sha256": source_capture["source_root_sha256"],
        "preregistration_sha256": _prerun_sha256(preregistration),
        "preregistration_report_sha256": _hash(preregistration_report),
        "postrun_capture_sha256": _hash(postrun_capture),
        "target_leak_hits": [],
        "component_hashes": component_hashes,
        "stage_epistemics": stage_epistemics,
        "native_bw": native_bw,
        "native_bw_diagnostic_verification": native_bw_diagnostic,
        "candidate_interventions": candidate_report,
        "geometry_controls": geometry_controls,
        "semantic_event": semantic_event,
        "campaign": campaign,
        "postrun_scientific_failures": scientific_failures,
        "postrun_diagnostic_findings": {
            "P4_native_bw01_bw08": native_diagnostic_findings,
            "P6_h3_s2_e3_e4": geometry_stats.get("diagnostic_findings", []),
            "P7_semantic_event": event_stats.get("diagnostic_findings", []),
        },
        "postrun_not_evaluated_reasons": not_evaluated_reasons,
        "claim_boundary": (
            "P1--P8 diagnostics are recomputed after one target-blind source "
            "evolution. Structural identities and sensitivity results are not "
            "physical pass/fail evidence. Missing independent producers are "
            "NOT_EVALUATED, and the campaign remains INCOMPLETE."
        ),
    }


def _prerun_sha256(value: Any) -> str:
    from oph_fpe.bulk.physical_h3_kms_prerun import canonical_sha256

    return canonical_sha256(value)


def _verified_preregistration(
    envelope: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from oph_fpe.bulk.physical_h3_kms_prerun import (
        PLAN_SCHEMA,
        SCHEMA_VERSION,
        physical_h3_kms_prerun_report,
    )

    value = _plain_mapping(envelope, "campaign_plan")
    _require_exact_keys(value, _PREREGISTRATION_ENVELOPE_KEYS, "campaign_plan")
    if value.get("schema") != PREREGISTRATION_ENVELOPE_SCHEMA:
        raise PostrunCaptureError("postrun preregistration envelope schema mismatch")
    preregistration = _plain_mapping(
        value.get("preregistration"), "preregistration"
    )
    _require_exact_keys(
        preregistration, {"schema", "config", "plan"}, "preregistration"
    )
    if preregistration.get("schema") != SCHEMA_VERSION:
        raise PostrunCaptureError("preregistration schema mismatch")
    declared_digest = str(value.get("preregistration_sha256") or "")
    computed_digest = _prerun_sha256(preregistration)
    if not _strict_hash(declared_digest) or declared_digest != computed_digest:
        raise PostrunCaptureError("preregistration digest mismatch")

    supplied_report = _plain_mapping(
        value.get("preregistration_report"), "preregistration_report"
    )
    recomputed_report = physical_h3_kms_prerun_report(preregistration)
    if _hash(supplied_report) != _hash(recomputed_report):
        raise PostrunCaptureError("preregistration report is not an exact replay")
    if (
        recomputed_report.get("admission_status") != "VALID_PASS"
        or recomputed_report.get("SOURCE_CAPTURE_ALLOWED") is not True
        or recomputed_report.get("scientific_status") != "NOT_EVALUATED"
    ):
        raise PostrunCaptureError("preregistration did not admit source capture")

    plan = _plain_mapping(preregistration.get("plan"), "preregistration.plan")
    _require_exact_keys(plan, _PLAN_KEYS, "preregistration.plan")
    if plan.get("schema") != PLAN_SCHEMA:
        raise PostrunCaptureError("frozen plan schema mismatch")
    thresholds = _plain_mapping(plan.get("thresholds"), "plan.thresholds")
    calibrations = _plain_mapping(plan.get("calibrations"), "plan.calibrations")
    split = _plain_mapping(plan.get("split_contract"), "plan.split_contract")
    _require_exact_keys(thresholds, _THRESHOLD_KEYS, "plan.thresholds")
    _require_exact_keys(calibrations, _CALIBRATION_KEYS, "plan.calibrations")
    _require_exact_keys(split, _SPLIT_KEYS, "plan.split_contract")
    if plan.get("plan_sha256") != _prerun_sha256(
        {key: item for key, item in plan.items() if key != "plan_sha256"}
    ):
        raise PostrunCaptureError("frozen plan commitment mismatch")
    return preregistration, recomputed_report, plan


def _source_capture_envelope(capture: Mapping[str, Any]) -> dict[str, Any]:
    from oph_fpe.bulk.physical_h3_kms_source_capture import (
        ARTIFACT_TYPE,
        SCHEMA,
    )

    value = _plain_mapping(capture, "source_capture")
    if value.get("schema") != SCHEMA or value.get("artifact_type") != ARTIFACT_TYPE:
        raise PostrunCaptureError("only registered physical source captures are accepted")
    return value


def _verified_source_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    from oph_fpe.bulk.physical_h3_kms_source_capture import (
        verify_physical_source_capture,
    )

    value = _source_capture_envelope(capture)
    verification = verify_physical_source_capture(value)
    if verification.get("SOURCE_CAPTURE_REPLAY_RECEIPT") is not True:
        blockers = ",".join(str(item) for item in verification.get("blockers", []))
        raise PostrunCaptureError(f"source capture replay failed: {blockers}")
    if verification.get("source_root_sha256") != value.get("source_root_sha256"):
        raise PostrunCaptureError("source replay root mismatch")
    return value


def _current_run_row(plan: Mapping[str, Any]) -> dict[str, Any]:
    current = _plain_mapping(plan.get("current_cell"), "plan.current_cell")
    rows = [
        _plain_mapping(item, "plan.run_matrix row")
        for item in _sequence(plan.get("run_matrix"))
    ]
    matches = [row for row in rows if row.get("cell") == current]
    if len(matches) != 1:
        raise PostrunCaptureError("frozen current cell does not have exactly one run row")
    return matches[0]


def _bind_source_to_preregistered_cell(
    source_capture: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> None:
    from oph_fpe.bulk.physical_h3_kms_prerun import physical_h3_kms_source_inputs

    plan = _plain_mapping(preregistration.get("plan"), "preregistration.plan")
    requested = _plain_mapping(preregistration.get("config"), "preregistration.config")
    current = _plain_mapping(plan.get("current_cell"), "plan.current_cell")
    row = _current_run_row(plan)
    if row.get("cell_config") != requested:
        raise PostrunCaptureError("requested cell differs from frozen run-matrix config")
    if row.get("config_sha256") != _prerun_sha256(requested):
        raise PostrunCaptureError("current cell config digest mismatch")

    source = _plain_mapping(source_capture.get("input_config"), "source input_config")
    expected_source_inputs = physical_h3_kms_source_inputs(preregistration)
    if source != expected_source_inputs:
        raise PostrunCaptureError(
            "outer source runtime config differs from admitted source-input projection"
        )
    identity = {
        "seed": source.get("seed"),
        "rung": source.get("rung"),
        "replicate_id": source.get("replicate_id"),
    }
    if identity != current:
        raise PostrunCaptureError("source capture identity differs from preregistered cell")
    if source.get("carrier_count") != current.get("rung"):
        raise PostrunCaptureError("source carrier count differs from preregistered rung")
    if source.get("preregistered_plan_sha256") != plan.get("plan_sha256"):
        raise PostrunCaptureError("source capture is not bound to the frozen plan digest")

def _derive_postrun_capture(
    source_capture: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> dict[str, Any]:
    from oph_fpe.bulk.physical_h3_kms_source_capture import POSTRUN_CAPTURE_SCHEMA

    raw = source_capture.get("postrun_capture")
    inner = _plain_mapping(raw, "source_capture.postrun_capture")
    expected = {
        "schema",
        "registration",
        *_DERIVED_CAPTURE_KEYS,
        "declared_hashes",
        "primitive_root_sha256",
    }
    _require_exact_keys(inner, expected, "source_capture.postrun_capture")
    if inner.get("schema") != POSTRUN_CAPTURE_SCHEMA:
        raise PostrunCaptureError("target-blind postrun capture schema mismatch")
    registration = _plain_mapping(inner.get("registration"), "postrun registration")
    _require_exact_keys(
        registration,
        {
            "schema",
            "seed",
            "rung",
            "replicate_id",
            "carrier_count",
            "support_regulator_count",
            "support_refinement_level",
            "observer_count",
            "observer_support_size",
            "preregistered_plan_sha256",
            "source_inputs",
            "source_inputs_sha256",
        },
        "postrun registration",
    )
    if registration.get("schema") != "oph.physical-source-capture.registration.v1":
        raise PostrunCaptureError("postrun registration schema mismatch")
    plan = _plain_mapping(preregistration.get("plan"), "preregistration.plan")
    current = _plain_mapping(plan.get("current_cell"), "plan.current_cell")
    source_input = _plain_mapping(
        source_capture.get("input_config"), "source input_config"
    )
    requested = _plain_mapping(
        preregistration.get("config"), "preregistration.config"
    )
    support = _plain_mapping(
        requested.get("support_regulator"), "cell support_regulator"
    )
    required_registration = {
        "seed": current.get("seed"),
        "rung": current.get("rung"),
        "replicate_id": current.get("replicate_id"),
        "preregistered_plan_sha256": plan.get("plan_sha256"),
        "carrier_count": source_input.get("carrier_count"),
        "support_refinement_level": source_input.get("support_refinement_level"),
        "observer_count": source_input.get("observer_count"),
        "observer_support_size": source_input.get("observer_support_size"),
        "support_regulator_count": support.get("patch_count"),
    }
    for key, expected_value in required_registration.items():
        if registration.get(key) != expected_value:
            raise PostrunCaptureError(f"postrun registration mismatch: {key}")
    from oph_fpe.bulk.physical_h3_kms_prerun import physical_h3_kms_source_inputs

    registered_inputs = _plain_mapping(
        registration.get("source_inputs"), "postrun registration.source_inputs"
    )
    expected_inputs = physical_h3_kms_source_inputs(preregistration)
    if registered_inputs != expected_inputs:
        raise PostrunCaptureError(
            "runtime source inputs differ from the admitted preregistration projection"
        )
    if registration.get("source_inputs_sha256") != _prerun_sha256(expected_inputs):
        raise PostrunCaptureError("runtime source-input projection digest mismatch")

    material = {key: inner.get(key) for key in _DERIVED_CAPTURE_KEYS}
    committed_material = {"registration": registration, **material}
    declared = _plain_mapping(inner.get("declared_hashes"), "postrun declared_hashes")
    _require_exact_keys(
        declared, {"registration", *_DERIVED_CAPTURE_KEYS}, "postrun declared_hashes"
    )
    computed = {key: _hash(value) for key, value in committed_material.items()}
    if declared != computed:
        raise PostrunCaptureError("postrun component commitment mismatch")
    if inner.get("primitive_root_sha256") != _hash(
        {"schema": POSTRUN_CAPTURE_SCHEMA, "components": declared}
    ):
        raise PostrunCaptureError("postrun primitive root commitment mismatch")
    return material


def _typed_clock_pair_rows(
    clock_pair_input: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate and join independent t/s producers on their frozen keys."""

    from oph_fpe.bulk.physical_h3_kms_source_capture import (
        _CLOCK_PAIR_GROUP_KEY_FIELDS,
        _CLOCK_PAIR_JOIN_KEY_FIELDS,
        _clock_pair_input_schema_blockers,
    )

    blockers = _clock_pair_input_schema_blockers(clock_pair_input)
    if blockers:
        raise PostrunCaptureError(
            "typed clock-pair input contract failed: " + ",".join(blockers)
        )
    contract = _plain_mapping(
        clock_pair_input.get("contract"), "clock_pair_input.contract"
    )
    if contract.get("status") == "UNAVAILABLE":
        return contract, []

    modular_rows = [
        _plain_mapping(row, "modular_transport_row")
        for row in _sequence(clock_pair_input.get("modular_transport_rows"))
    ]
    geometric_rows = [
        _plain_mapping(row, "geometric_flow_row")
        for row in _sequence(clock_pair_input.get("geometric_flow_rows"))
    ]

    def join_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(row[field] for field in _CLOCK_PAIR_JOIN_KEY_FIELDS)

    modular_by_key = {join_key(row): row for row in modular_rows}
    geometric_by_key = {join_key(row): row for row in geometric_rows}
    if set(modular_by_key) != set(geometric_by_key):
        raise PostrunCaptureError("typed clock-pair join-key sets differ")

    joined: list[dict[str, Any]] = []
    for key in sorted(modular_by_key, key=lambda value: _hash(list(value))):
        modular = modular_by_key[key]
        geometric = geometric_by_key[key]
        if modular["source_seed"] != geometric["source_seed"]:
            raise PostrunCaptureError("typed clock-pair source seeds differ at join")
        group_material = {
            field: modular[field] for field in _CLOCK_PAIR_GROUP_KEY_FIELDS
        }
        pair_material = {
            **{field: modular[field] for field in _CLOCK_PAIR_JOIN_KEY_FIELDS},
            "source_seed": modular["source_seed"],
            "modular_transport_time": float(modular["modular_transport_time"]),
            "geometric_flow_parameter": float(
                geometric["geometric_flow_parameter"]
            ),
            "modular_transport_row_id": modular["row_id"],
            "geometric_flow_row_id": geometric["row_id"],
            "modular_transport_source_field_sha256": modular[
                "producer_source_field_sha256"
            ],
            "geometric_flow_source_field_sha256": geometric[
                "producer_source_field_sha256"
            ],
            "oriented_frame_incidence_sha256": geometric[
                "oriented_frame_incidence_sha256"
            ],
            "trajectory_group_sha256": _hash(group_material),
        }
        joined.append(
            {
                "pair_row_id": _hash(
                    {
                        "modular_transport_row_id": modular["row_id"],
                        "geometric_flow_row_id": geometric["row_id"],
                        "join_key": list(key),
                    }
                ),
                **pair_material,
                "pair_row_sha256": _hash(pair_material),
            }
        )
    return contract, joined


def _grouped_clock_split(
    rows: list[dict[str, Any]], plan: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[str, ...]]:
    group_ids = tuple(
        sorted({str(row["trajectory_group_sha256"]) for row in rows})
    )
    heldout_groups = _semantic_holdout_ids(
        group_ids, plan, domain="clock-pair-trajectory-group"
    )
    heldout_set = set(heldout_groups)
    train = [
        row for row in rows if str(row["trajectory_group_sha256"]) not in heldout_set
    ]
    holdout = [
        row for row in rows if str(row["trajectory_group_sha256"]) in heldout_set
    ]
    return train, holdout, heldout_groups


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["trajectory_group_sha256"]), []).append(row)
    return grouped


def _candidate_report(
    trajectories: list[Any],
    interventions: list[Any],
    response_rows: list[Any],
    clock_pair_input: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = _plain_mapping(plan.get("thresholds"), "plan.thresholds")
    calibration = _plain_mapping(plan.get("calibrations"), "plan.calibrations")
    archive = _plain_mapping(plan.get("archive_boundary"), "plan.archive_boundary")
    labels = tuple(plan.get("clock_candidates", REQUIRED_CLOCK_LABELS))
    if labels != REQUIRED_CLOCK_LABELS:
        raise PostrunCaptureError("clock candidate labels must be exactly 1x, pi, 2pi, 4pi")

    rows = [_plain_mapping(row, "response_row") for row in response_rows]
    response_keys = {
        "row_id",
        "record_event_id",
        "feedback_event_id",
        "observer_token",
        "carrier_id",
        "port",
        "independent_geometric_parameter",
        "raw_response",
        "initial_port_intensity",
        "settled_port_intensity",
        "repaired_port_intensity",
        "propagation_delta",
        "repair_delta",
        "refinement_level",
        "geometry_source_row_id",
        "row_sha256",
    }
    for index, row in enumerate(rows):
        _require_exact_keys(row, response_keys, f"response_rows[{index}]")
        _verify_row_hash(row, f"response_rows[{index}]")
    trajectory_keys = {
        "carrier_id",
        "initial_port_intensities",
        "settled_port_intensities",
        "repaired_port_intensities",
        "initial_intrinsic_phase",
        "settled_intrinsic_phase",
    }
    for index, raw in enumerate(trajectories):
        _require_exact_keys(
            _plain_mapping(raw, "carrier trajectory"),
            trajectory_keys,
            f"carrier_port_trajectories[{index}]",
        )
    intervention_keys = {
        "row_id",
        "operation",
        "seam_id",
        "read_set",
        "write_set",
        "mismatch_before",
        "mismatch_after",
    }
    for index, raw in enumerate(interventions):
        _require_exact_keys(
            _plain_mapping(raw, "intervention"),
            intervention_keys,
            f"intervention_rows[{index}]",
        )
    ids = [str(row.get("row_id") or "") for row in rows]
    if not ids or any(not row_id for row_id in ids) or len(ids) != len(set(ids)):
        raise PostrunCaptureError("response row IDs must be nonempty and unique")
    typed_contract, typed_rows = _typed_clock_pair_rows(clock_pair_input)
    if typed_contract.get("status") == "UNAVAILABLE":
        frozen_holdout = _semantic_holdout_ids(ids, plan, domain="clock-response")
        return _typed_clock_unavailable_report(
            trajectories,
            interventions,
            rows,
            clock_pair_input,
            typed_contract,
            frozen_holdout,
            labels,
            plan,
            thresholds,
            calibration,
            archive,
        )

    train, holdout, heldout_groups = _grouped_clock_split(typed_rows, plan)
    train_groups = _group_rows(train)
    holdout_groups = _group_rows(holdout)
    if len(train_groups) < 2 or len(holdout_groups) < 2:
        raise PostrunCaptureError(
            "clock fit requires at least two independent train and holdout groups"
        )
    x_train = _column(train, "modular_transport_time")
    y_train = _column(train, "geometric_flow_parameter")
    denom = float(x_train @ x_train)
    if denom <= 0.0:
        raise PostrunCaptureError("typed modular transport has zero train support")
    kappa_hat = float((x_train @ y_train) / denom)
    # Rows are reordered here so cluster scores cannot accidentally depend on
    # the input order used by the joined producer tables.
    group_scores: list[float] = []
    for group_id in sorted(train_groups):
        group = train_groups[group_id]
        group_x = _column(group, "modular_transport_time")
        group_y = _column(group, "geometric_flow_parameter")
        group_scores.append(float(group_x @ (group_y - kappa_hat * group_x)))
    group_count = len(group_scores)
    cluster_variance = (
        (group_count / (group_count - 1.0))
        * float(np.square(group_scores).sum())
        / (denom * denom)
    )
    standard_error = math.sqrt(max(0.0, cluster_variance))
    confidence_z = 1.959963984540054
    interval = [kappa_hat - confidence_z * standard_error, kappa_hat + confidence_z * standard_error]

    group_losses: dict[str, dict[str, float]] = {}
    for group_id, group in sorted(holdout_groups.items()):
        group_x = _column(group, "modular_transport_time")
        group_y = _column(group, "geometric_flow_parameter")
        group_losses[group_id] = {
            label: float(np.mean(np.square(group_y - value * group_x)))
            for label, value in _CLOCK_VALUES.items()
        }
    losses = {
        label: float(np.mean([row[label] for row in group_losses.values()]))
        for label in _CLOCK_VALUES
    }
    uncertainty_by_competitor: dict[str, float] = {}
    for label in ("1x", "pi", "4pi"):
        differences = np.asarray(
            [row[label] - row["2pi"] for row in group_losses.values()],
            dtype=float,
        )
        uncertainty_by_competitor[label] = float(
            confidence_z
            * np.std(differences, ddof=1)
            / math.sqrt(len(differences))
        )
    uncertainty = max(uncertainty_by_competitor.values())
    grouped_fit_residuals = []
    for group in holdout_groups.values():
        group_x = _column(group, "modular_transport_time")
        group_y = _column(group, "geometric_flow_parameter")
        grouped_fit_residuals.append(
            float(np.mean(np.square(group_y - kappa_hat * group_x)))
        )
    absolute_residual = float(math.sqrt(np.mean(grouped_fit_residuals)))
    absolute_limit = _nonnegative_number(
        thresholds.get("clock_absolute_residual_max"),
        "clock_absolute_residual_max",
    )
    delta = _nonnegative_number(
        thresholds.get("clock_win_margin_min"), "clock_win_margin_min"
    )
    tail_threshold = absolute_limit
    levels = sorted({int(row["refinement_level"]) for row in typed_rows})
    if len(levels) < 2:
        raise PostrunCaptureError(
            "typed clock-pair input requires at least two actual refinement levels"
        )
    tail_fits: list[dict[str, Any]] = []
    for level in levels[-2:]:
        level_rows = [row for row in train if int(row["refinement_level"]) == level]
        level_groups = _group_rows(level_rows)
        lx = _column(level_rows, "modular_transport_time")
        ly = _column(level_rows, "geometric_flow_parameter")
        ld = float(lx @ lx)
        if len(level_groups) >= 2 and ld > 0.0:
            tail_fits.append(
                {
                    "refinement_level": level,
                    "independent_group_count": len(level_groups),
                    "kappa_hat": float((lx @ ly) / ld),
                }
            )
    tail_stable = bool(
        len(tail_fits) == 2
        and abs(tail_fits[1]["kappa_hat"] - tail_fits[0]["kappa_hat"])
        <= tail_threshold
    )

    intervention_ids = [
        str(_plain_mapping(row, "intervention").get("row_id") or "")
        for row in interventions
    ]
    if not intervention_ids or any(not value for value in intervention_ids):
        raise PostrunCaptureError("intervention row IDs must be nonempty")
    frozen_holdout = tuple(str(row["pair_row_id"]) for row in holdout)
    packet_hash = _hash(interventions)
    trajectory_hash = _hash(trajectories)
    response_hash = _hash(
        {"clock_pair_input": clock_pair_input, "joined_rows": typed_rows}
    )
    source_intervention_target_free = not _target_leak_hits(
        {
            "trajectories": trajectories,
            "interventions": interventions,
            "legacy_responses": rows,
            "clock_pair_input": clock_pair_input,
        }
    )
    interventions_frozen = all(
        _strict_hash(value) for value in (packet_hash, trajectory_hash, response_hash)
    )
    labels_frozen = bool(
        labels == REQUIRED_CLOCK_LABELS and _strict_hash(plan.get("plan_sha256"))
    )
    aggregate_payload = {
        "intervention_row_ids": intervention_ids,
        "heldout_event_row_ids": list(frozen_holdout),
        "intervention_packet_hash": packet_hash,
        "source_trajectory_hash": trajectory_hash,
        "raw_response_hash": response_hash,
    }
    aggregate_hash = _hash(aggregate_payload)
    candidates: dict[str, dict[str, Any]] = {
        label: {
            **aggregate_payload,
            "candidate_invariance_aggregate_hash": aggregate_hash,
            "candidate_scale_applied_only_in_scoring": True,
            "candidate_scale_enters_intervention": False,
            "candidate_parameter_name": "kappa",
            "candidate_value": value,
            "candidate_units": "dimensionless_geometric_flow_parameter",
        }
        for label, value in _CLOCK_VALUES.items()
    }
    wrong_separation = bool(
        interval[0] <= _CLOCK_VALUES["2pi"] <= interval[1]
        and not any(interval[0] <= _CLOCK_VALUES[label] <= interval[1] for label in ("1x", "pi", "4pi"))
    )
    scientific_failures: list[str] = []
    if not (losses["2pi"] + delta + uncertainty < min(losses[label] for label in ("1x", "pi", "4pi"))):
        scientific_failures.append("discrete_clock_2pi_does_not_defeat_1_pi_4pi_by_frozen_margin")
    if not (interval[0] <= _CLOCK_VALUES["2pi"] <= interval[1]):
        scientific_failures.append("continuous_clock_interval_does_not_contain_2pi")
    if not wrong_separation:
        scientific_failures.append("continuous_clock_wrong_normalization_separation_failed")
    if absolute_residual > absolute_limit:
        scientific_failures.append("continuous_clock_absolute_adequacy_failed")
    if not tail_stable:
        scientific_failures.append("continuous_clock_refinement_tail_not_stable")
    same_rows_and_packets = bool(
        len({_hash(value) for value in candidates.values()}) == len(candidates)
        and len(
            {
                (
                    tuple(value["intervention_row_ids"]),
                    tuple(value["heldout_event_row_ids"]),
                    value["intervention_packet_hash"],
                    value["source_trajectory_hash"],
                    value["raw_response_hash"],
                    value["candidate_invariance_aggregate_hash"],
                )
                for value in candidates.values()
            }
        )
        == 1
    )
    calibration_hash = str(calibration.get("clock_calibration_sha256") or "")
    calibration_fixture_committed = bool(
        calibration.get("independent_of_campaign_source_seeds") is True
        and _strict_hash(calibration_hash)
    )
    # The current campaign calibration file is a deterministic SHA-uniform
    # threshold fixture.  A commitment to that fixture is not a replay-bound
    # physical/noise/power calibration receipt, so it cannot make P5 eligible.
    physical_calibration_replay_bound = bool(
        calibration.get("physical_threshold_calibration_receipt") is True
        and _strict_hash(calibration_hash)
    )
    calibration_independent = bool(
        calibration_fixture_committed and physical_calibration_replay_bound
    )
    calibration_frozen = bool(
        calibration.get("frozen_before_source_capture") is True
        and archive.get("frozen_before_source_capture") is True
        and archive.get("retune_after_freeze") is False
    )
    diagnostic_findings = list(scientific_failures)
    calibration_reason = "replay_bound_independent_physical_clock_calibration_missing"
    report = {
        "schema": "oph.physical_h3_kms.candidate_interventions.v1",
        "measurement_status": (
            CellStatus.VALID_FAIL.value
            if physical_calibration_replay_bound and scientific_failures
            else (
                CellStatus.VALID_PASS.value
                if physical_calibration_replay_bound
                else CellStatus.NOT_EVALUATED.value
            )
        ),
        "physical_gate_eligible": physical_calibration_replay_bound,
        "not_evaluated_reasons": (
            [] if physical_calibration_replay_bound else [calibration_reason]
        ),
        "interventions_frozen_before_candidate_scoring": interventions_frozen,
        "candidate_labels_frozen_before_runs": labels_frozen,
        "source_intervention_target_free": source_intervention_target_free,
        "candidate_invariance_aggregate_hash": aggregate_hash,
        "typed_clock_pair_input_sha256": _hash(clock_pair_input),
        "candidates": candidates,
        "typed_clock_pair_contract": {
            "required_collections": [
                "modular_transport_rows",
                "geometric_flow_rows",
            ],
            "required_fields": [
                "modular_transport_time",
                "geometric_flow_parameter",
            ],
            "available": True,
            "joined_on": list(typed_contract["join_key_fields"]),
            "grouped_on": list(typed_contract["group_key_fields"]),
            "producer_ids_disjoint": True,
            "producer_code_hashes_disjoint": True,
            "source_field_hashes_disjoint": True,
            "source_fixed_oriented_frame_incidence_hashes": True,
            "intensity_delta_used_as_modular_time": False,
            "one_field_synthesized_from_the_other": False,
        },
        "grouped_inference": {
            "split_unit": "source_seed_observer_or_cap_trajectory_group_v1",
            "group_key_fields": list(typed_contract["group_key_fields"]),
            "train_group_count": len(train_groups),
            "holdout_group_count": len(holdout_groups),
            "heldout_group_sha256": list(heldout_groups),
            "row_split_within_group": False,
            "equal_weight_per_holdout_group": True,
            "cluster_robust_train_standard_error": standard_error,
            "paired_uncertainty_by_competitor": uncertainty_by_competitor,
        },
        "continuous_clock_fit": {
            "fitted_kappa_interval": interval,
            "kappa_hat": kappa_hat,
            "absolute_residual": absolute_residual,
            "frozen_absolute_residual_threshold": absolute_limit,
            "wrong_normalization_separation_passed": wrong_separation,
            "refinement_tail_stable": tail_stable,
            "actual_refinement_levels": levels,
            "tail_level_fits": tail_fits,
            "fit_artifact_hash": _hash(
                {
                    "train": train,
                    "train_groups": sorted(train_groups),
                    "fit": kappa_hat,
                    "interval": interval,
                }
            ),
        },
        "discrete_clock_comparison": {
            "paired_losses": losses,
            "frozen_delta_clock": delta,
            "paired_uncertainty_upper_bound": uncertainty,
            "paired_uncertainty_by_competitor": uncertainty_by_competitor,
            "same_rows_and_packets": same_rows_and_packets,
            "thresholds_from_independent_synthetic_calibration": calibration_independent,
            "thresholds_frozen_before_physical_campaign": calibration_frozen,
            "calibration_artifact_hash": calibration_hash,
            "threshold_fixture_commitment_valid": calibration_fixture_committed,
            "physical_calibration_replay_bound": physical_calibration_replay_bound,
        },
        "structural_receipt": {
            "scope": "artifact_integrity_only",
            "status": "COMPLETE",
            "artifact_hash": aggregate_hash,
            "physical_claim": False,
        },
        "instrument_receipt": {
            "scope": "independent_physical_producer",
            "status": (
                "AVAILABLE"
                if physical_calibration_replay_bound
                else "MISSING_REQUIRED_CALIBRATION"
            ),
            "physical_claim": False,
        },
        "sensitivity_receipt": {
            "scope": (
                "registered_physical_candidate_comparison"
                if physical_calibration_replay_bound
                else "diagnostic_sensitivity_only"
            ),
            "status": "COMPLETE",
            "artifact_hash": _hash(
                {"fit": kappa_hat, "interval": interval, "losses": losses}
            ),
            "physical_gate_eligible": physical_calibration_replay_bound,
            "diagnostic_findings": (
                [] if physical_calibration_replay_bound else diagnostic_findings
            ),
        },
        "primitive_response_rows": rows,
        "primitive_typed_clock_pair_rows": typed_rows,
        "scientific_failures": (
            scientific_failures if physical_calibration_replay_bound else []
        ),
    }
    return report, {
        "evaluated": physical_calibration_replay_bound,
        "not_evaluated_reasons": (
            [] if physical_calibration_replay_bound else [calibration_reason]
        ),
        "typed_producers_available": True,
        "sensitivity_complete": True,
        "sensitivity_artifact_hash": report["sensitivity_receipt"]["artifact_hash"],
        "diagnostic_findings": (
            [] if physical_calibration_replay_bound else diagnostic_findings
        ),
        "kappa_hat": kappa_hat,
        "losses": losses,
        "absolute_residual": absolute_residual,
        "calibration_independent": calibration_independent,
        "calibration_frozen": calibration_frozen,
        "scientific_failures": (
            scientific_failures if physical_calibration_replay_bound else []
        ),
    }


def _typed_clock_unavailable_report(
    trajectories: list[Any],
    interventions: list[Any],
    rows: list[dict[str, Any]],
    clock_pair_input: Mapping[str, Any],
    typed_contract: Mapping[str, Any],
    heldout: tuple[str, ...],
    labels: tuple[str, ...],
    plan: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    calibration: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    intervention_ids = [
        str(_plain_mapping(row, "intervention").get("row_id") or "")
        for row in interventions
    ]
    packet_hash = _hash(interventions)
    trajectory_hash = _hash(trajectories)
    response_hash = _hash(rows)
    aggregate_payload = {
        "intervention_row_ids": intervention_ids,
        "heldout_event_row_ids": list(heldout),
        "intervention_packet_hash": packet_hash,
        "source_trajectory_hash": trajectory_hash,
        "raw_response_hash": response_hash,
    }
    aggregate_hash = _hash(aggregate_payload)
    candidates = {
        label: {
            **aggregate_payload,
            "candidate_invariance_aggregate_hash": aggregate_hash,
            "candidate_scale_applied_only_in_scoring": True,
            "candidate_scale_enters_intervention": False,
            "candidate_parameter_name": "kappa",
            "candidate_value": _CLOCK_VALUES[label],
            "candidate_units": "dimensionless_geometric_flow_parameter",
        }
        for label in labels
    }
    common_rows = len(
        {
            (
                tuple(row["intervention_row_ids"]),
                tuple(row["heldout_event_row_ids"]),
                row["intervention_packet_hash"],
                row["source_trajectory_hash"],
                row["raw_response_hash"],
                row["candidate_invariance_aggregate_hash"],
            )
            for row in candidates.values()
        }
    ) == 1
    calibration_hash = str(calibration.get("clock_calibration_sha256") or "")
    calibration_fixture_committed = bool(
        calibration.get("independent_of_campaign_source_seeds") is True
        and _strict_hash(calibration_hash)
    )
    calibration_independent = False
    calibration_frozen = bool(
        calibration.get("frozen_before_source_capture") is True
        and archive.get("frozen_before_source_capture") is True
        and archive.get("retune_after_freeze") is False
    )
    target_free = not _target_leak_hits(
        {
            "trajectories": trajectories,
            "interventions": interventions,
            "responses": rows,
            "clock_pair_input": clock_pair_input,
        }
    )
    reason = (
        "source_capture_lacks_independent_modular_transport_time_and_"
        "geometric_flow_parameter_pairs"
    )
    report = {
        "schema": "oph.physical_h3_kms.candidate_interventions.v1",
        "measurement_status": CellStatus.NOT_EVALUATED.value,
        "physical_gate_eligible": False,
        "not_evaluated_reasons": [reason],
        "typed_clock_pair_contract": {
            "required_collections": [
                "modular_transport_rows",
                "geometric_flow_rows",
            ],
            "required_fields": ["modular_transport_time", "geometric_flow_parameter"],
            "available": False,
            "contract_status": typed_contract.get("status"),
            "joined_on": list(typed_contract.get("join_key_fields", [])),
            "grouped_on": list(typed_contract.get("group_key_fields", [])),
            "minimum_refinement_level_count": typed_contract.get(
                "minimum_refinement_level_count"
            ),
            "modular_transport_row_count": 0,
            "geometric_flow_row_count": 0,
            "intensity_delta_used_as_modular_time": False,
            "one_field_synthesized_from_the_other": False,
        },
        "interventions_frozen_before_candidate_scoring": all(
            _strict_hash(value) for value in (packet_hash, trajectory_hash, response_hash)
        ),
        "candidate_labels_frozen_before_runs": bool(
            labels == REQUIRED_CLOCK_LABELS and _strict_hash(plan.get("plan_sha256"))
        ),
        "source_intervention_target_free": target_free,
        "candidate_invariance_aggregate_hash": aggregate_hash,
        "typed_clock_pair_input_sha256": _hash(clock_pair_input),
        "candidates": candidates,
        "continuous_clock_fit": {
            "fitted_kappa_interval": [],
            "kappa_hat": None,
            "absolute_residual": None,
            "frozen_absolute_residual_threshold": _nonnegative_number(
                thresholds.get("clock_absolute_residual_max"),
                "clock_absolute_residual_max",
            ),
            "wrong_normalization_separation_passed": False,
            "refinement_tail_stable": False,
            "fit_artifact_hash": _hash({"status": "NOT_EVALUATED", "reason": reason}),
        },
        "discrete_clock_comparison": {
            "paired_losses": {},
            "frozen_delta_clock": _nonnegative_number(
                thresholds.get("clock_win_margin_min"), "clock_win_margin_min"
            ),
            "paired_uncertainty_upper_bound": 0.0,
            "same_rows_and_packets": common_rows,
            "thresholds_from_independent_synthetic_calibration": calibration_independent,
            "thresholds_frozen_before_physical_campaign": calibration_frozen,
            "calibration_artifact_hash": calibration_hash,
            "threshold_fixture_commitment_valid": calibration_fixture_committed,
            "physical_calibration_replay_bound": False,
        },
        "structural_receipt": {
            "scope": "artifact_integrity_only",
            "status": "COMPLETE",
            "artifact_hash": aggregate_hash,
            "physical_claim": False,
        },
        "instrument_receipt": {
            "scope": "independent_physical_producer",
            "status": "MISSING_REQUIRED_PRODUCER",
            "physical_claim": False,
        },
        "sensitivity_receipt": {
            "scope": "registered_physical_candidate_comparison",
            "status": "NOT_RUN",
            "artifact_hash": None,
            "physical_gate_eligible": False,
            "diagnostic_findings": [],
        },
        "primitive_response_rows": rows,
        "primitive_typed_clock_pair_rows": [],
        "scientific_failures": [],
    }
    return report, {
        "evaluated": False,
        "not_evaluated_reasons": [reason],
        "scientific_failures": [],
    }


def _derive_native_bw_geometry(
    geometry: Mapping[str, Any], cap_state: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive BW analysis primitives from committed neutral source fields.

    The construction introduces a hyperbolic-cap candidate only here, after
    source capture.  Every residual below is recomputed from the neutral
    ordered frames, refinement maps, or source density matrix; no clause pass
    flag is accepted as an input.
    """

    _require_exact_keys(
        geometry,
        {
            "derivation_method",
            "orientation_fixed_from_source",
            "raw_primitive_rows",
            "support_refinement_levels",
            "conditional_expectation_rows",
        },
        "geometry_samples",
    )
    if geometry.get("derivation_method") != "ordered_bw_frame_cross_ratio":
        raise PostrunCaptureError("unexpected neutral geometry derivation method")
    raw_rows = [
        _plain_mapping(row, "geometry raw primitive row")
        for row in _sequence(geometry.get("raw_primitive_rows"))
    ]
    if len(raw_rows) < 4:
        raise PostrunCaptureError("at least four neutral ordered-frame rows are required")
    raw_keys = {
        "row_id",
        "ordered_frame",
        "orientation",
        "cross_ratio",
        "geometric_parameter",
        "row_sha256",
    }
    for index, row in enumerate(raw_rows):
        _require_exact_keys(row, raw_keys, f"geometry raw row[{index}]")
        _verify_row_hash(row, f"geometry raw row[{index}]")
    frames = np.asarray(
        [_vector(row, "ordered_frame", 4) for row in raw_rows], dtype=float
    )
    if not np.all(np.diff(frames, axis=1) > 0.0):
        raise PostrunCaptureError("ordered-frame source rows are not strictly ordered")

    levels = [
        _plain_mapping(row, "support refinement level")
        for row in _sequence(geometry.get("support_refinement_levels"))
    ]
    expectations = [
        _plain_mapping(row, "conditional expectation")
        for row in _sequence(geometry.get("conditional_expectation_rows"))
    ]
    if len(levels) < 2 or len(expectations) != len(levels) - 1:
        raise PostrunCaptureError("neutral support tower is incomplete")
    level_ids: list[str] = []
    patch_counts: list[int] = []
    for index, row in enumerate(levels):
        expected_keys = {"level_id", "patch_count", "geometry_hash"}
        if index:
            expected_keys |= {"parent_level_id", "lineage_hash"}
        _require_exact_keys(row, expected_keys, f"support refinement level[{index}]")
        level_ids.append(str(row.get("level_id") or ""))
        patch_counts.append(int(row.get("patch_count", 0)))
    expectation_keys = {
        "fine_level_id",
        "coarse_level_id",
        "operator_hash",
        "unital",
        "positive",
        "state_preserving",
        "cap_isotony_compatible",
        "noncommutative_prime_cap_expectation",
        "fiber_algebra",
        "left_inverse_residual",
    }
    for index, row in enumerate(expectations):
        _require_exact_keys(
            row, expectation_keys, f"conditional expectation[{index}]"
        )
        if (
            row.get("coarse_level_id") != level_ids[index]
            or row.get("fine_level_id") != level_ids[index + 1]
        ):
            raise PostrunCaptureError("conditional expectation breaks tower lineage")

    _require_exact_keys(
        cap_state,
        {
            "source_matrix",
            "source_hermitian_constraint",
            "gibbs_parameter",
            "rho",
            "modular_generator",
            "m4_generator_z",
            "m4_generator_x",
        },
        "cap_state_raw_primitives",
    )
    rho = _complex_pair_matrix(cap_state.get("rho"), "rho")
    generator = _complex_pair_matrix(
        cap_state.get("modular_generator"), "modular_generator"
    )
    generator_z = _complex_pair_matrix(
        cap_state.get("m4_generator_z"), "m4_generator_z"
    )
    generator_x = _complex_pair_matrix(
        cap_state.get("m4_generator_x"), "m4_generator_x"
    )
    if any(matrix.shape != (4, 4) for matrix in (rho, generator, generator_z, generator_x)):
        raise PostrunCaptureError("source M4 primitive has the wrong shape")

    def sphere_point(value: float) -> np.ndarray:
        angle = 2.0 * math.atan(value)
        return np.asarray([math.cos(angle), math.sin(angle), 0.0], dtype=float)

    def cap_from_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        minus = sphere_point(float(np.mean(rows[:, 0])))
        plus = sphere_point(float(np.mean(rows[:, 3])))
        spatial = minus + plus
        spatial_norm = float(np.linalg.norm(spatial))
        if spatial_norm <= 1.0e-12:
            spatial = np.cross(minus, np.asarray([0.0, 0.0, 1.0]))
            spatial_norm = float(np.linalg.norm(spatial))
        spatial /= max(spatial_norm, 1.0e-15)
        time = 0.5 * float(spatial @ (minus + plus))
        scale = math.sqrt(max(1.0 - time * time, 1.0e-15))
        return np.concatenate(([time / scale], spatial / scale)), minus, plus

    normal, p_minus, p_plus = cap_from_rows(frames)
    refinement_normals = []
    for index in range(len(levels)):
        prefix = max(2, math.ceil(len(frames) * (index + 1) / len(levels)))
        refinement_normals.append(cap_from_rows(frames[:prefix])[0].tolist())

    cross_ratio_residuals: list[float] = []
    mesh_residuals: list[float] = []
    quartet_separations: list[float] = []
    conditions: list[float] = []
    for row, frame in zip(raw_rows, frames, strict=True):
        a, b, c, d = frame.tolist()
        recomputed = ((c - a) * (d - b)) / ((b - a) * (d - c))
        cross_ratio_residuals.append(abs(recomputed - float(row["cross_ratio"])))
        gaps = np.diff(frame)
        mesh_residuals.append(float(np.std(gaps)))
        quartet_separations.append(float(np.min(gaps)))
        conditions.append(float(np.max(gaps) / np.min(gaps)))

    expectation_residuals = [
        abs(_number(row.get("left_inverse_residual"), "left_inverse_residual"))
        for row in expectations
    ]
    isotony = [
        bool(
            row.get("unital") is True
            and row.get("positive") is True
            and row.get("state_preserving") is True
            and row.get("cap_isotony_compatible") is True
            and row.get("noncommutative_prime_cap_expectation") is True
        )
        for row in expectations
    ]
    inclusion_margins = [
        (right - left) / right
        for left, right in zip(patch_counts, patch_counts[1:], strict=False)
    ]
    inclusion_matrix = [
        [1.0 if left <= right else 0.0 for right in patch_counts]
        for left in patch_counts
    ]

    eigenvalues, eigenvectors = np.linalg.eigh(generator)

    def unitary(parameter: float) -> np.ndarray:
        return (eigenvectors * np.exp(1j * parameter * eigenvalues)) @ eigenvectors.conj().T

    flow_parameters = [float(row["geometric_parameter"]) for row in raw_rows]
    flow_identity = float(np.linalg.norm(unitary(0.0) - np.eye(4), ord="fro"))
    flow_group: list[float] = []
    flow_inverse: list[float] = []
    for left, right in zip(flow_parameters, flow_parameters[1:], strict=False):
        flow_group.append(
            float(np.linalg.norm(unitary(left) @ unitary(right) - unitary(left + right)))
        )
        flow_inverse.append(
            float(np.linalg.norm(unitary(left) @ unitary(-left) - np.eye(4)))
        )
    gibbs = (eigenvectors * np.exp(-eigenvalues)) @ eigenvectors.conj().T
    gibbs /= np.trace(gibbs)
    functional_calculus_residual = float(np.linalg.norm(gibbs - rho, ord="fro"))
    # The source has one M4 state, not independently captured states at every
    # refinement rung.  Distance to the neutral tracial reference is therefore
    # retained as the unresolved physical-reference/cross-level residual;
    # K=-log(rho) cannot certify its own physical emergence by definition.
    reference_residual = float(np.linalg.norm(rho - np.eye(4) / 4.0, ord="fro"))
    left = np.kron(generator_z, np.eye(4))
    right = np.kron(np.eye(4), generator_x.T)
    mixed_gns_residual = float(np.linalg.norm(left @ right - right @ left, ord="fro"))
    generator_commutators = [
        float(np.linalg.norm(generator @ item - item @ generator, ord="fro"))
        for item in (generator_z, generator_x)
    ]
    strip_values = [
        float(np.linalg.norm(rho @ item - item @ rho, ord="fro"))
        for item in (generator_z, generator_x)
    ]
    cap_anchor_residuals = [
        float(np.linalg.norm(unitary(value) @ rho @ unitary(value).conj().T - rho))
        for value in flow_parameters
    ]
    normal_variations = [
        float(np.linalg.norm(np.asarray(value) - normal))
        for value in refinement_normals
    ]
    level_errors = [max(expectation_residuals[index:], default=0.0) for index in range(len(expectation_residuals))]
    if len(level_errors) < 2:
        level_errors = [max(expectation_residuals, default=0.0), max(expectation_residuals, default=0.0)]
    error_levels = patch_counts[-len(level_errors):]

    return {
        "cap_normal": normal.tolist(),
        "frame_p_minus": p_minus.tolist(),
        "frame_p_plus": p_plus.tolist(),
        "boundary_points": [p_minus.tolist(), p_plus.tolist()],
        "interior_points": [(normal[1:] / np.linalg.norm(normal[1:])).tolist()],
        "refined_cap_normals": refinement_normals,
        "cap_orientation": "interior_positive",
        "cap_radius_margin": max(0.0, 1.0 - abs(float(normal[0] / np.linalg.norm(normal[1:])))),
        "cap_mesh_residuals": mesh_residuals,
        "point_mesh_residuals": [abs(float(point @ point) - 1.0) for point in (p_minus, p_plus)],
        "frame_ordering": "p_minus_attracting_for_positive_s",
        "frame_orientation_values": [float(np.min(np.diff(row))) for row in frames],
        "cap_inclusion_matrix": inclusion_matrix,
        "inclusion_margins": inclusion_margins,
        "order_refinement_residuals": expectation_residuals,
        "support_isotony_checks": isotony,
        "support_separation_margins": inclusion_margins,
        "support_covariance_residuals": expectation_residuals,
        "support_kernel_residuals": expectation_residuals,
        "sector_scope": "PRIME_GEOMETRIC_SUPPORT_VISIBLE",
        "test_tower_id": "registered-source-M4-on-icosahedral-support",
        "test_tower_states": {"levels": levels, "rho": cap_state["rho"]},
        "state_embedding_residuals": expectation_residuals,
        "regularizer_eta": 1.0 / max(patch_counts),
        "physical_reference_trace_distances": [reference_residual],
        "local_modular_residuals": [functional_calculus_residual],
        "mixed_gns_cauchy_residuals": [max(mixed_gns_residual, reference_residual)],
        "negative_time_residuals": flow_inverse or [flow_identity],
        "matrix_element_residuals": [reference_residual],
        "flow_identity_residuals": [flow_identity],
        "flow_group_residuals": flow_group or [flow_identity],
        "flow_inverse_residuals": flow_inverse or [flow_identity],
        "flow_equicontinuity_bounds": [float(np.linalg.norm(unitary(value), ord=2)) for value in flow_parameters],
        "cap_anchor_residuals": cap_anchor_residuals,
        "frame_fixed_point_residuals": normal_variations,
        "cross_ratio_holdout_residuals": cross_ratio_residuals,
        "quartet_separations": quartet_separations,
        "cross_ratio_anchor_conditions": conditions,
        "orientation_values": [float(np.min(np.diff(row))) for row in frames],
        "strip_values": strip_values,
        "flow_parameters": flow_parameters,
        "generator_commutator_values": generator_commutators,
        "error_envelope_samples": level_errors,
        "error_envelope_refinement_levels": error_levels,
    }


def _unavailable_native_bw_payload(
    component_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Emit an explicit missing-input contract instead of numeric sentinels.

    BW07 requires an analytic-strip KMS residual and BW08 requires a
    central-minimized weighted generator distance.  Neither can be recovered
    from a clock regression loss or from ``abs(kappa - 2*pi)``.  Until the raw
    cofinal state tower and independent modular/geometric generators exist,
    the native payload is unavailable and carries no clause numbers.
    """

    antecedent_hash = _hash(
        {
            "geometry_samples": component_hashes["geometry_samples"],
            "clock_pair_input": component_hashes["clock_pair_input"],
            "cap_state_raw_primitives": component_hashes[
                "cap_state_raw_primitives"
            ],
        }
    )
    return {
        "schema": "oph.physical_h3_kms.native_bw_input.v2",
        "status": "UNAVAILABLE",
        "physical_gate_eligible": False,
        "antecedent_hash": antecedent_hash,
        "missing_producers": [
            "cofinal_physical_state_tower",
            "analytic_strip_kms_residual",
            "independent_geometric_generator",
            "central_minimized_weighted_generator_distance",
        ],
        "clauses": {},
        "claim_boundary": (
            "No finite C1-C6 or complete same-tower MGNS-1 primitive is emitted "
            "until its physical producer exists. Missing observables are "
            "NOT_EVALUATED, not numeric failure sentinels."
        ),
    }


def _legacy_native_bw_diagnostic_payload(
    geometry: Mapping[str, Any],
    clock: Mapping[str, Any],
    component_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Quarantined legacy sensitivity mapping; never used by physical runs.

    Its BW07/BW08 values are regression proxies rather than the paper's KMS
    strip and central-minimized generator observables.  It remains only to
    document why archived diagnostic artifacts cannot be promoted.
    """
    clock_evaluated = clock.get("evaluated") is True
    clock_losses = clock.get("losses") if isinstance(clock.get("losses"), Mapping) else {}
    kms_residual = (
        math.sqrt(float(clock_losses["2pi"]))
        if clock_evaluated and "2pi" in clock_losses
        else 1.0 + _max_abs(geometry.get("strip_values"))
    )
    wrong_gap = (
        min(float(clock_losses[key]) for key in ("1x", "pi", "4pi"))
        - float(clock_losses["2pi"])
        if clock_evaluated and set(_CLOCK_VALUES) <= set(clock_losses)
        else 0.0
    )
    generator_distance = (
        abs(float(clock["kappa_hat"]) - 2.0 * math.pi)
        if clock_evaluated
        else 2.0 * math.pi
    )
    normal = _vector(geometry, "cap_normal", 4)
    p_minus = _vector(geometry, "frame_p_minus", 3)
    p_plus = _vector(geometry, "frame_p_plus", 3)
    boundary = [_vector_value(value, 3, "boundary_point") for value in _sequence(geometry.get("boundary_points"))]
    interior = [_vector_value(value, 3, "interior_point") for value in _sequence(geometry.get("interior_points"))]
    def incidence(point: np.ndarray) -> float:
        return -normal[0] + float(normal[1:] @ point)
    norm_residual = abs(-normal[0] ** 2 + float(normal[1:] @ normal[1:]) - 1.0)
    boundary_residual = max([abs(incidence(point)) for point in boundary + [p_minus, p_plus]], default=math.inf)
    sign_violation = max([max(0.0, -incidence(point)) for point in interior], default=math.inf)
    refinement_normals = [_vector_value(value, 4, "refined_cap_normal") for value in _sequence(geometry.get("refined_cap_normals"))]
    fields: dict[str, Any] = {
        "finite_cap_source_role": "geometric_support_flow",
        "finite_cap_source_artifact_hash": component_hashes["geometry_samples"],
        "finite_cap_tower_id": str(geometry.get("test_tower_id") or ""),
        "finite_cap_tower_hash": _hash(geometry.get("test_tower_states", [])),
        "cap_normal": normal.tolist(),
        "cap_normal_norm_residual": norm_residual,
        "cap_orientation": str(geometry.get("cap_orientation") or ""),
        "cap_radius_margin": _number(geometry.get("cap_radius_margin"), "cap_radius_margin"),
        "cap_boundary_incidence_residual": boundary_residual,
        "cap_sign_violation": sign_violation,
        "cap_mesh_error": _max_abs(geometry.get("cap_mesh_residuals")),
        "point_mesh_error": _max_abs(geometry.get("point_mesh_residuals")),
        "refinement_normal_error": max([float(np.linalg.norm(value - normal)) for value in refinement_normals], default=math.inf),
        "frame_p_minus": p_minus.tolist(),
        "frame_p_plus": p_plus.tolist(),
        "frame_boundary_residual": max(abs(incidence(p_minus)), abs(incidence(p_plus))),
        "frame_separation": float(np.linalg.norm(p_minus - p_plus)),
        "frame_ordering": str(geometry.get("frame_ordering") or ""),
        "frame_orientation_witness": _all_positive(geometry.get("frame_orientation_values")),
        "cap_inclusion_matrix": _matrix(geometry, "cap_inclusion_matrix").tolist(),
        "strict_inclusion_margin": _min_value(geometry.get("inclusion_margins")),
        "order_refinement_error": _max_abs(geometry.get("order_refinement_residuals")),
        "support_isotony_failures": sum(not bool(value) for value in _sequence(geometry.get("support_isotony_checks"))),
        "support_separation_margin": _min_value(geometry.get("support_separation_margins")),
        "support_covariance_residual_T": _max_abs(geometry.get("support_covariance_residuals")),
        "support_kernel_residual": _max_abs(geometry.get("support_kernel_residuals")),
        "sector_scope": str(geometry.get("sector_scope") or ""),
        "test_tower_id": str(geometry.get("test_tower_id") or ""),
        "test_tower_hash": _hash(geometry.get("test_tower_states", [])),
        "state_embedding_residual": _max_abs(geometry.get("state_embedding_residuals")),
        "regularizer_eta": _number(geometry.get("regularizer_eta"), "regularizer_eta"),
        "physical_reference_trace_distance": _max_abs(geometry.get("physical_reference_trace_distances")),
        "fixed_local_modular_bound_T": _max_abs(geometry.get("local_modular_residuals")),
        "mixed_gns_cauchy_residual_T": _max_abs(geometry.get("mixed_gns_cauchy_residuals")),
        "negative_time_residual_T": _max_abs(geometry.get("negative_time_residuals")),
        "matrix_element_residual_T": _max_abs(geometry.get("matrix_element_residuals")),
        "flow_identity_residual": _max_abs(geometry.get("flow_identity_residuals")),
        "flow_group_residual_T": _max_abs(geometry.get("flow_group_residuals")),
        "flow_inverse_residual_T": _max_abs(geometry.get("flow_inverse_residuals")),
        "flow_equi_continuity_bound": _max_value(geometry.get("flow_equicontinuity_bounds")),
        "cap_anchor_residual": _max_abs(geometry.get("cap_anchor_residuals")),
        "frame_fixed_point_residual": _max_abs(geometry.get("frame_fixed_point_residuals")),
        "cross_ratio_holdout_max": _max_abs(geometry.get("cross_ratio_holdout_residuals")),
        "quartet_separation_min": _min_value(geometry.get("quartet_separations")),
        "cross_ratio_anchor_condition": _max_value(geometry.get("cross_ratio_anchor_conditions")),
        "orientation_witness": _all_positive(geometry.get("orientation_values")),
        "geometric_parameter_convention": "h_C(z) -> e^{-s} h_C(z)",
        "kms_comparison_state_id": str(geometry.get("test_tower_id") or ""),
        "kms_comparison_state_hash": _hash(geometry.get("test_tower_states", [])),
        "kms_matrix_element_residual_T": _max_abs(
            geometry.get("matrix_element_residuals")
        ),
        "kms_strip_bound": _max_abs(geometry.get("strip_values")),
        "kms_residual_beta_2pi": kms_residual,
        "geometric_flow_nontrivial": bool(np.var(_numbers(geometry.get("flow_parameters"))) > 1.0e-20),
        "wrong_beta_interval": [1.0, 4.0 * math.pi],
        "wrong_beta_gap_delta": wrong_gap,
        "geometric_generator_noncentrality": _max_abs(geometry.get("generator_commutator_values")),
        "generator_distance_beta_2pi": generator_distance,
        "total_308_error_envelope": _max_value(geometry.get("error_envelope_samples")),
        "error_envelope_samples": _numbers(geometry.get("error_envelope_samples")),
        "error_envelope_refinement_levels": [int(value) for value in _sequence(geometry.get("error_envelope_refinement_levels"))],
        "error_envelope_refinement_witness": _strictly_increasing(geometry.get("error_envelope_refinement_levels")),
    }
    antecedent_hash = _hash({"geometry": component_hashes["geometry_samples"], "response": component_hashes["response_rows"]})
    clauses = {}
    for clause_id, names in BW_PRIMITIVE_FIELD_CONTRACT.items():
        primitive = {name: fields[name] for name in names}
        clauses[clause_id] = {
            "antecedent_hash": antecedent_hash,
            "primitive_artifact_hash": canonical_payload_hash(primitive),
            "primitive_fields": primitive,
        }
    return {
        "schema_version": "oph_bw_native_legacy_diagnostic_v1",
        "producer_kind": "native_simulator",
        "source_kind": "physical_source_generation",
        "antecedent_hash": antecedent_hash,
        "clauses": clauses,
    }


def _geometry_control_report(
    raw_rows: list[Any],
    plan: Mapping[str, Any],
    component_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [_plain_mapping(row, "geometry_control_row") for row in raw_rows]
    row_keys = {
        "row_id",
        "source_geometry_row_id",
        "source_response_row_id",
        "observer_token",
        "trajectory_group_id",
        "ordered_frame",
        "orientation",
        "cross_ratio",
        "geometric_parameter",
        "neutral_feature_vector",
        "observed_source_value",
        "predictor_source_phase",
        "response_source_phase",
        "predictor_response_field_intersection",
        "row_sha256",
    }
    for index, row in enumerate(rows):
        _require_exact_keys(row, row_keys, f"geometry_control_rows[{index}]")
        _verify_row_hash(row, f"geometry_control_rows[{index}]")
        _vector(row, "ordered_frame", 4)
        neutral = _vector(row, "neutral_feature_vector", 7)
        if not np.all(np.isfinite(neutral)):
            raise PostrunCaptureError("geometry neutral features are nonfinite")
        if (
            not isinstance(row.get("observer_token"), str)
            or not row.get("observer_token")
            or not isinstance(row.get("trajectory_group_id"), str)
            or not row.get("trajectory_group_id")
            or row.get("predictor_source_phase")
            != "pre_intervention_initial_state"
            or row.get("response_source_phase")
            != "post_repair_minus_initial_response"
            or row.get("predictor_response_field_intersection") != []
        ):
            raise PostrunCaptureError(
                "geometry feature/target provenance contract failed"
            )
    row_ids = [str(row.get("row_id") or "") for row in rows]
    if not row_ids or any(not value for value in row_ids) or len(set(row_ids)) != len(row_ids):
        raise PostrunCaptureError("neutral geometry row IDs are incomplete or duplicated")
    group_ids = tuple(
        sorted({str(row["trajectory_group_id"]) for row in rows})
    )
    heldout_group_ids = _semantic_holdout_ids(
        group_ids, plan, domain="geometry-control-trajectory-group"
    )
    heldout_group_set = set(heldout_group_ids)
    train = [
        row
        for row in rows
        if str(row["trajectory_group_id"]) not in heldout_group_set
    ]
    test = [
        row
        for row in rows
        if str(row["trajectory_group_id"]) in heldout_group_set
    ]
    holdout_ids = tuple(str(row["row_id"]) for row in test)
    capacity = 5
    if capacity < 2 or len(train) < capacity:
        raise PostrunCaptureError("geometry effective capacity is invalid or underpowered")
    ridge = 1.0e-8
    y_train = _column(train, "observed_source_value")
    y_test = _column(test, "observed_source_value")
    scores: dict[str, float] = {}
    coefficient_hashes: dict[str, str] = {}
    design_hashes: dict[str, str] = {}
    shared_preprocessing_hash = _hash(
        {
            "raw_feature_contract": "seven_pre_intervention_geometry_fields_v1",
            "predictor_response_field_intersection": [],
            "fixed_capacity": capacity,
            "intercept": True,
        }
    )
    shared_fit_protocol_hash = _hash(
        {
            "capacity": capacity,
            "ridge": ridge,
            "heldout_trajectory_groups": list(heldout_group_ids),
            "split_unit": "source_seed_observer_carrier_port_trajectory_group_v1",
            "equal_weight_per_holdout_group": True,
        }
    )
    row_losses: dict[str, np.ndarray] = {}
    optimizer_status: dict[str, str] = {}
    for model in REQUIRED_GEOMETRY_MODELS:
        x_train = _fixed_capacity_matrix(train, model, capacity)
        x_test = _fixed_capacity_matrix(test, model, capacity)
        gram = x_train.T @ x_train + ridge * np.eye(capacity)
        coef = np.linalg.pinv(gram) @ x_train.T @ y_train
        prediction = x_test @ coef
        row_losses[model] = np.square(y_test - prediction)
        scores[model] = float(np.mean(row_losses[model]))
        coefficient_hashes[model] = _hash(coef.tolist())
        design_hashes[model] = _hash(
            {"train": x_train.tolist(), "test": x_test.tolist()}
        )
        optimizer_status[model] = (
            "CONVERGED" if np.all(np.isfinite(coef)) else "NONFINITE"
        )
    test_group_counts = {
        group_id: sum(
            str(row["trajectory_group_id"]) == group_id for row in test
        )
        for group_id in heldout_group_ids
    }
    heldout_weights = [
        1.0
        / (
            len(heldout_group_ids)
            * test_group_counts[str(row["trajectory_group_id"])]
        )
        for row in test
    ]
    common = {
        "heldout_event_matrix_hash": _hash(test),
        "heldout_weights_hash": _hash(heldout_weights),
        "missingness_mask_hash": _hash([False] * len(test)),
        "source_packet_hash": component_hashes["geometry_control_rows"],
        "prediction_target_hash": _hash(y_test.tolist()),
    }
    models = {
        model: {
            "heldout_event_row_ids": list(holdout_ids),
            **common,
            "preprocessing_hash": shared_preprocessing_hash,
            "fit_protocol_hash": shared_fit_protocol_hash,
            "candidate_design_matrix_hash": design_hashes[model],
            "candidate_feature_source": (
                f"shared_pre_intervention_geometry_then_{model.lower()}_transform"
            ),
            "effective_model_capacity": capacity,
            "heldout_score": scores[model],
            "optimizer_status": optimizer_status[model],
            "required_rows_complete": bool(len(train) >= capacity and len(test) >= 2),
            "coefficient_hash": coefficient_hashes[model],
        }
        for model in REQUIRED_GEOMETRY_MODELS
    }
    thresholds = _plain_mapping(plan.get("thresholds"), "plan.thresholds")
    calibrations = _plain_mapping(plan.get("calibrations"), "plan.calibrations")
    archive = _plain_mapping(plan.get("archive_boundary"), "plan.archive_boundary")
    margin = _nonnegative_number(
        thresholds.get("geometry_win_margin_min"), "geometry_win_margin_min"
    )
    minimum_power = _positive_number(
        thresholds.get("curvature_minimum_power"), "curvature_minimum_power"
    )
    geometry_calibration_hash = str(
        calibrations.get("geometry_calibration_sha256") or ""
    )
    calibration_hash = str(
        calibrations.get("curvature_calibration_sha256") or ""
    )
    coords = np.asarray([_feature_vector(row, "H3") for row in test], dtype=float)
    diameter = max((float(np.linalg.norm(left - right)) for left in coords for right in coords), default=0.0)
    losses_by_group: dict[str, dict[str, float]] = {}
    for group_id in heldout_group_ids:
        indices = [
            index
            for index, row in enumerate(test)
            if str(row["trajectory_group_id"]) == group_id
        ]
        losses_by_group[group_id] = {
            model: float(np.mean(row_losses[model][indices]))
            for model in REQUIRED_GEOMETRY_MODELS
        }
    for model in REQUIRED_GEOMETRY_MODELS:
        scores[model] = float(
            np.mean(
                [group_losses[model] for group_losses in losses_by_group.values()]
            )
        )
        models[model]["heldout_score"] = scores[model]
    paired = np.asarray(
        [
            group_losses[model] - group_losses["H3"]
            for group_losses in losses_by_group.values()
            for model in ("S2", "E3", "E4")
        ],
        dtype=float,
    )
    uncertainty = (
        float(1.959963984540054 * np.std(paired, ddof=1) / math.sqrt(len(paired)))
        if len(paired) > 1
        else 0.0
    )
    diagnostic_findings: list[str] = []
    if not (scores["H3"] + margin + uncertainty < min(scores[name] for name in ("S2", "E3", "E4"))):
        diagnostic_findings.append(
            "diagnostic_h3_does_not_defeat_s2_e3_e4_by_frozen_paired_margin"
        )
    calibration_independent = bool(
        calibrations.get("independent_of_campaign_source_seeds") is True
        and calibrations.get("physical_threshold_calibration_receipt") is True
        and _strict_hash(geometry_calibration_hash)
        and calibration_hash == curvature_calibration_commitment()
    )
    calibration_frozen = bool(
        calibrations.get("frozen_before_source_capture") is True
        and archive.get("frozen_before_source_capture") is True
        and archive.get("retune_after_freeze") is False
    )
    curvature_radius = 1.0
    sensitivity_artifact = {
        "scores": scores,
        "design_hashes": design_hashes,
        "shared_preprocessing_hash": shared_preprocessing_hash,
        "shared_fit_protocol_hash": shared_fit_protocol_hash,
        "heldout_group_ids": list(heldout_group_ids),
        "losses_by_group": losses_by_group,
        "margin": margin,
        "uncertainty": uncertainty,
        "equal_footing_candidate_inputs": True,
    }
    stage_assessment = _not_evaluated_stage_assessment(
        reason=_P6_REASON,
        structural_complete=bool(
            rows
            and all(value == "CONVERGED" for value in optimizer_status.values())
            and all(_strict_hash(value) for value in design_hashes.values())
        ),
        structural_artifact_hash=_hash(
            {"rows": rows, "holdout_ids": holdout_ids}
        ),
        instrument_producer_available=False,
        sensitivity_complete=True,
        sensitivity_artifact_hash=_hash(sensitivity_artifact),
        diagnostic_findings=diagnostic_findings,
    )
    report = {
        "schema": "oph.physical_h3_kms.geometry_controls.v1",
        "measurement_status": CellStatus.NOT_EVALUATED.value,
        "physical_gate_eligible": False,
        "not_evaluated_reasons": [_P6_REASON],
        "models_frozen_before_holdout": True,
        "heldout_excluded_from_model_selection": True,
        "models": models,
        "paired_geometry_comparison": {
            "loss_direction": "lower_is_better",
            "frozen_h3_win_margin": margin,
            "paired_uncertainty_upper_bound": uncertainty,
            "thresholds_frozen_before_physical_campaign": calibration_frozen,
            "calibration_artifact_hash": geometry_calibration_hash,
            "same_heldout_rows": True,
            "same_prediction_target": True,
            "matched_column_count": True,
            "equal_footing_candidate_inputs": True,
            "split_unit": (
                "source_seed_observer_carrier_port_trajectory_group_v1"
            ),
            "heldout_group_ids": list(heldout_group_ids),
            "row_split_within_group": False,
            "equal_weight_per_holdout_group": True,
            "predictor_response_field_intersection": [],
            "feature_target_disjoint_receipt": True,
            "physical_threshold_calibration_receipt": (
                calibrations.get("physical_threshold_calibration_receipt") is True
            ),
            "physical_gate_eligible": False,
        },
        "curvature_leverage": {
            "measurement_status": CellStatus.NOT_EVALUATED.value,
            "physical_gate_eligible": False,
            "physical_threshold_calibration_receipt": (
                calibrations.get("physical_threshold_calibration_receipt") is True
            ),
            "calibration_source": "legacy_synthetic_power_suite_quarantined",
            "frozen_before_physical_campaign": calibration_frozen,
            "calibration_hash": calibration_hash,
            "registered_analysis_hash": _hash(sensitivity_artifact),
            "domain_diameter": diameter,
            "registered_curvature_radius": curvature_radius,
            "noise_scale": 0.1,
            "sample_count": 0,
            "calibrated_power": None,
            "minimum_power": minimum_power,
            "curvature_radius_source": "postcapture_fixed_diagnostic",
            "curvature_radius_frozen_before_h3_fit": False,
            "flat_limit_excluded": False,
            "curvature_parameter_charged_to_model_capacity": False,
            "h3_e3_distinguishable_at_registered_effect": False,
            "independent_power_analysis_hash": _hash(
                {
                    "status": "QUARANTINED",
                    "reason": "response_y_equals_cosh_s_minus_one_encodes_h3_target",
                    "legacy_commitment": curvature_calibration_commitment(),
                    "registered_calibration_present": calibration_independent,
                }
            ),
            "quarantine_reason": (
                "legacy_y_equals_cosh_s_minus_one_power_suite_encodes_the_h3_"
                "alternative_and_is_not_physical_campaign_evidence"
            ),
        },
        "structural_receipt": stage_assessment["structural_receipt"],
        "instrument_receipt": stage_assessment["instrument_receipt"],
        "sensitivity_receipt": stage_assessment["sensitivity_receipt"],
        "diagnostic_findings": diagnostic_findings,
        "scientific_failures": [],
    }
    return report, {
        "evaluated": False,
        "scores": scores,
        "diagnostic_findings": diagnostic_findings,
        "not_evaluated_reasons": [_P6_REASON],
        "scientific_failures": [],
        "stage_assessment": stage_assessment,
    }


def _semantic_event_report(
    raw_events: list[Any],
    raw_overlaps: list[Any],
    raw_ancestry: list[Any],
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    events = [_plain_mapping(row, "semantic_event") for row in raw_events]
    event_keys = {
        "event_key",
        "canonical_semantic_payload",
        "observer_token",
        "visible_footprint",
        "parent_event_ids",
        "read_resource_ids",
        "write_resource_ids",
        "source_sequence_index",
    }
    for index, row in enumerate(events):
        _require_exact_keys(row, event_keys, f"semantic_events[{index}]")
    keys = [str(row.get("event_key") or "") for row in events]
    if not keys or any(not value for value in keys) or len(keys) != len(set(keys)):
        raise PostrunCaptureError("semantic event keys must be nonempty and unique")
    by_key = {str(row["event_key"]): row for row in events}
    record_event_keys = [
        key
        for key, row in by_key.items()
        if _plain_mapping(
            row.get("canonical_semantic_payload"),
            "canonical_semantic_payload",
        ).get("event_kind")
        == "RECORD_COMMIT"
    ]
    record_pair_count = len(record_event_keys) * (len(record_event_keys) - 1) // 2
    parents_by_child = {
        key: [
            str(value)
            for value in _sequence(row.get("parent_event_ids"))
        ]
        for key, row in by_key.items()
    }
    for parents in parents_by_child.values():
        if any(parent not in by_key for parent in parents):
            raise PostrunCaptureError("semantic event names an absent parent")
    children_by_parent: dict[str, list[str]] = {key: [] for key in keys}
    indegree = {key: len(parents) for key, parents in parents_by_child.items()}
    for child, parents in parents_by_child.items():
        for parent in parents:
            children_by_parent[parent].append(child)
    computed_ids: dict[str, str] = {}
    ready = [key for key in keys if indegree[key] == 0]
    heapq.heapify(ready)
    while ready:
        key = heapq.heappop(ready)
        parents = parents_by_child[key]
        material = {
            "canonical_semantic_payload": by_key[key].get(
                "canonical_semantic_payload", {}
            ),
            "observer_token": str(by_key[key].get("observer_token") or ""),
            "visible_footprint": sorted(
                str(value)
                for value in _sequence(by_key[key].get("visible_footprint"))
            ),
            "semantic_causal_parents": sorted(
                computed_ids[parent] for parent in parents
            ),
        }
        computed_ids[key] = _hash(material)
        for child in children_by_parent[key]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    remaining = set(keys) - set(computed_ids)
    acyclic = not remaining
    parent_edges = [
        (str(parent), key)
        for key, parents in parents_by_child.items()
        for parent in parents
    ]
    raw_edges = [(str(parent), str(child)) for parent, child in parent_edges]
    depths = _dag_depths(keys, raw_edges) if acyclic else {key: 0 for key in keys}
    overlaps = _verified_overlap_rows(raw_overlaps)
    carrier_coordinates, carrier_degrees = _carrier_topology_coordinates(overlaps)

    positions_by_key: dict[str, np.ndarray] = {}
    h3_frames_by_key: dict[str, np.ndarray] = {}
    event_context: dict[str, tuple[str, float, int]] = {}
    for key in sorted(keys, key=lambda value: (depths.get(value, 0), value)):
        row = by_key[key]
        payload = _plain_mapping(
            row.get("canonical_semantic_payload"), "canonical_semantic_payload"
        )
        parents = [str(value) for value in _sequence(row.get("parent_event_ids"))]
        carrier = str(payload.get("carrier_id") or "")
        if carrier:
            source_value = _number(payload.get("port_value"), "record port_value")
            port = int(payload.get("port", 0))
        elif parents and parents[0] in event_context:
            carrier, source_value, port = event_context[parents[0]]
        else:
            carrier, source_value, port = "", 0.0, 0
        event_context[key] = (carrier, source_value, port)
        sequence_index = int(row.get("source_sequence_index", 0))
        topology = carrier_coordinates.get(carrier, np.zeros(3, dtype=float))
        port_coordinate = float(port) / 11.0
        observable_displacement = np.asarray(
            [source_value, port_coordinate, source_value * port_coordinate],
            dtype=float,
        )
        positions_by_key[key] = np.asarray(
            [
                *(topology + 0.01 * observable_displacement),
                float(depths.get(key, 0)) + sequence_index / max(1, len(events)),
            ],
            dtype=float,
        )
        frame = np.asarray(
            [
                source_value,
                port_coordinate,
                carrier_degrees.get(carrier, 0.0),
            ],
            dtype=float,
        )
        if float(np.linalg.norm(frame)) <= 1.0e-15:
            frame = np.asarray([0.0, 0.0, 1.0], dtype=float)
        h3_frames_by_key[key] = frame / max(float(np.linalg.norm(frame)), 1.0e-15)
    positions = np.asarray([positions_by_key[key] for key in keys], dtype=float)
    levels = [
        1 + int(math.log2(int(row.get("source_sequence_index", 0)) + 2))
        for row in events
    ]
    radii = [1.0 / ((len(events) + 1) * (2**level)) for level in levels]
    level_max = [max(r for r, item_level in zip(radii, levels, strict=True) if item_level == level) for level in sorted(set(levels))]
    shrinking = bool(len(level_max) >= 2 and all(right < left for left, right in zip(level_max, level_max[1:], strict=False)))
    sampled_event_pairs, pair_scope = _diagnostic_event_pair_sample(
        keys,
        computed_ids,
        plan,
    )
    pair_distances = [
        float(np.linalg.norm(positions[left] - positions[right]))
        for _, left, right in sampled_event_pairs
    ]
    pair_gaps = [
        distance - (radii[left] + radii[right])
        for distance, (_, left, right) in zip(
            pair_distances,
            sampled_event_pairs,
            strict=True,
        )
    ]
    separated = sum(gap > 0.0 for gap in pair_gaps)
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    rank_tolerance = 1.0e-9
    rank_four = int(np.linalg.matrix_rank(centered, tol=rank_tolerance))
    spatial_rank = int(np.linalg.matrix_rank(centered[:, :3], tol=rank_tolerance))
    clock_rank = int(np.linalg.matrix_rank(centered[:, 3:], tol=rank_tolerance))
    writes = {
        key: set(str(value) for value in _sequence(row.get("write_resource_ids")))
        for key, row in by_key.items()
    }
    reads = {
        key: set(str(value) for value in _sequence(row.get("read_resource_ids")))
        for key, row in by_key.items()
    }
    ancestry_rows = _verified_ancestry_rows(raw_ancestry, by_key, raw_edges, reads, writes)
    ancestry_edges = [
        (str(row["parent_event_id"]), str(row["child_event_id"]))
        for row in ancestry_rows
        if _sequence(row.get("shared_resource_ids"))
    ]

    transitions, triangles = _derive_poincare_transitions(
        overlaps, events, positions_by_key
    )
    transition_hash = _hash(transitions)
    lorentz_present = all(_matrix(row, "lorentz").shape == (4, 4) for row in transitions) if transitions else False
    translation_present = all(_vector(row, "translation", 4).size == 4 for row in transitions) if transitions else False
    charts = {str(row.get("source_chart")) for row in transitions} | {str(row.get("target_chart")) for row in transitions}
    connected = _connected([(str(row.get("source_chart")), str(row.get("target_chart"))) for row in transitions], charts)
    cocycle_residual = _poincare_cocycle_residual(transitions, triangles)

    ancestors: dict[str, set[str]] = {}
    for key in sorted(keys, key=lambda value: (depths.get(value, 0), value)):
        parents = parents_by_child[key]
        ancestors[key] = set(parents)
        for parent in parents:
            ancestors[key].update(ancestors.get(parent, set()))
    relations: list[dict[str, Any]] = []
    for _, left_index, right_index in sampled_event_pairs:
        left = keys[left_index]
        right = keys[right_index]
        if left in ancestors.get(right, set()):
            ordered_left, ordered_right, relation = left, right, "causal"
        elif right in ancestors.get(left, set()):
            ordered_left, ordered_right, relation = right, left, "causal"
        else:
            ordered_left, ordered_right, relation = left, right, "spacelike"
        material = {
            "left_event_key": ordered_left,
            "right_event_key": ordered_right,
            "relation": relation,
        }
        relations.append({"row_id": _hash(material), **material})
    cone, cone_failures = _unavailable_quadratic_cone(
        relations,
        positions_by_key,
        plan,
        pair_scope=pair_scope,
    )
    increments = [depths[child] - depths[parent] for parent, child in raw_edges]
    minimum_increment = float(min(increments)) if increments else 0.0
    current = _plain_mapping(plan.get("current_cell"), "plan.current_cell")
    cauchy_threshold = 1.0 / math.sqrt(int(current.get("rung", 1)))
    cauchy_residual = max(level_max[-1] if level_max else math.inf, 0.0)
    h3_frames = [h3_frames_by_key[key].tolist() for key in keys]
    frame_base_separate = _hash(h3_frames) != _hash(positions.tolist())

    event_diagnostic_findings: list[str] = [
        "EVENT_E1_physical_localization_boxes_not_produced",
        "EVENT_E2_physical_record_germ_separation_not_evaluated",
        "EVENT_E3_independent_clock_conditioned_rank_not_produced",
        "EVENT_E4_independent_per_chart_coordinates_not_produced",
        "stable_causality_not_evaluated_from_physical_event_chart",
        "record_cauchy_completion_not_evaluated_from_refinement_germs",
        "h3_frame_fiber_event_base_separation_not_established",
    ]
    if shrinking:
        event_diagnostic_findings.append(
            "diagnostic_layout_radii_shrink_by_construction"
        )
    if pair_scope["full_pair_census"] is not True:
        event_diagnostic_findings.append(
            "EVENT_E2_full_pair_census_not_run_diagnostic_scope"
        )
    if not acyclic:
        event_diagnostic_findings.append("semantic_event_dag_is_not_acyclic")
    if len(ancestry_edges) != len(raw_edges):
        event_diagnostic_findings.append(
            "translation_edges_not_bound_to_causal_ancestry"
        )
    event_diagnostic_findings.extend(cone_failures)

    population_hash = _hash({"events": computed_ids, "levels": levels, "radii": radii})
    cone = {
        **cone,
        "measurement_status": CellStatus.NOT_EVALUATED.value,
        "physical_gate_eligible": False,
        "coordinate_source": "postcapture_diagnostic_construction",
    }
    sensitivity_artifact = {
        "positions": positions.tolist(),
        "frames": h3_frames,
        "levels": levels,
        "radii": radii,
        "pair_scope": pair_scope,
        "relations": relations,
        "cone": cone,
        "transitions": transitions,
    }
    stage_assessment = _not_evaluated_stage_assessment(
        reason=_P7_REASON,
        structural_complete=bool(
            events
            and len(computed_ids) == len(events)
            and _strict_hash(population_hash)
            and _strict_hash(_hash(ancestry_rows))
        ),
        structural_artifact_hash=_hash(
            {
                "computed_event_ids": computed_ids,
                "raw_parent_edges": raw_edges,
                "ancestry_rows": ancestry_rows,
            }
        ),
        instrument_producer_available=False,
        sensitivity_complete=True,
        sensitivity_artifact_hash=_hash(sensitivity_artifact),
        diagnostic_findings=event_diagnostic_findings,
    )
    report = {
        "schema": "oph.physical_h3_kms.semantic_event_reconstruction.v2",
        "measurement_status": CellStatus.NOT_EVALUATED.value,
        "physical_gate_eligible": False,
        "EVENT_MANIFOLD_3P1D_RECEIPT": False,
        "not_evaluated_reasons": [_P7_REASON],
        "physical_event_chart_input": {
            "status": "UNAVAILABLE",
            "physical_gate_eligible": False,
            "per_chart_coordinate_rows": [],
            "independent_clock_rows": [],
            "localization_box_rows": [],
            "unavailable_reason": (
                "independent_event_chart_clock_and_localization_producers_missing"
            ),
        },
        "pairwise_diagnostic_scope": pair_scope,
        "event_clauses": {
            "EVENT_E1_POPULATION": {
                "measurement_status": CellStatus.NOT_EVALUATED.value,
                "physical_gate_eligible": False,
                "semantic_record_germ_count": len(record_event_keys),
                "certified_localization_box_count": 0,
                "dense_population_verified": False,
                "shrinking_box_sequence_verified": False,
                "diagnostic_layout_event_count": len(events),
                "diagnostic_layout_radii_shrink_by_construction": shrinking,
                "population_artifact_hash": population_hash,
            },
            "EVENT_E2_SEPARATION": {
                "measurement_status": CellStatus.NOT_EVALUATED.value,
                "physical_gate_eligible": False,
                "distinct_germ_pair_count": record_pair_count,
                "evaluated_germ_pair_count": 0,
                "separated_pair_count": 0,
                "minimum_localization_gap": None,
                "minimum_localization_gap_is_population_bound": False,
                "disjoint_certified_boxes_verified": False,
                "diagnostic_layout_evaluated_pair_count": len(pair_distances),
                "diagnostic_layout_separated_pair_count": separated,
                "diagnostic_layout_minimum_gap": min(pair_gaps, default=0.0),
                "separation_artifact_hash": _hash(pair_gaps),
                "diagnostic_pair_sample_sha256": pair_scope[
                    "sampled_event_pair_sequence_sha256"
                ],
            },
            "EVENT_E3_RANK_FOUR": {
                "measurement_status": CellStatus.NOT_EVALUATED.value,
                "physical_gate_eligible": False,
                "conditioned_spatial_response_rank": None,
                "independent_clock_line_rank": None,
                "combined_event_rank": None,
                "clock_line_independent_of_spatial_response": False,
                "diagnostic_rank_independence_check": (
                    rank_four == spatial_rank + clock_rank
                ),
                "diagnostic_layout_spatial_rank": spatial_rank,
                "diagnostic_layout_clock_rank": clock_rank,
                "diagnostic_layout_combined_rank": rank_four,
                "independent_clock_receipt": False,
                "physical_independent_clock_receipt": False,
                "rank_four_artifact_hash": _hash(positions.tolist()),
            },
            "EVENT_E4_POINCARE_COCYCLE": {
                "measurement_status": CellStatus.NOT_EVALUATED.value,
                "physical_gate_eligible": False,
                "overlap_transition_count": len(transitions),
                "lorentz_components_present": lorentz_present,
                "translation_components_present": translation_present,
                "poincare_cocycle_residual": None,
                "connected_overlap_atlas": False,
                "diagnostic_transition_count": len(transitions),
                "diagnostic_cocycle_residual": cocycle_residual,
                "diagnostic_connected_overlap_atlas": connected,
                "poincare_transition_artifact_hash": transition_hash,
            },
        },
        "semantic_event_dag": {
            "identity_fields": ["canonical_semantic_payload", "observer_token", "visible_footprint", "semantic_causal_parents"],
            "forbidden_identity_fields_present": [],
            "semantic_parent_edge_count": len(raw_edges),
            "acyclic": acyclic,
            "causal_cycle_count": 0 if acyclic else len(remaining),
            "duplicate_semantic_event_count": len(computed_ids) - len(set(computed_ids.values())),
            "preassigned_metric_used_for_identity": False,
            "quotient_canonical_identity_receipt": False,
            "identity_scope": "presentation_bound_structural_dag_only",
            "semantic_event_dag_hash": _hash({"ids": computed_ids, "edges": raw_edges}),
        },
        "causal_ancestry": {
            "committed_read_after_write_edge_count": len(ancestry_edges),
            "translation_edge_ancestry_coverage_fraction": 0.0,
            "observer_feedback_edge_resource_coverage_fraction": len(ancestry_edges) / len(raw_edges) if raw_edges else 0.0,
            "generic_event_token_used_as_resource_witness": False,
            "population_used_as_reachability_surrogate": False,
            "ancestry_artifact_hash": _hash(ancestry_edges),
        },
        "quadratic_event_cone": cone,
        "stable_causality": {
            "time_function_source": "UNAVAILABLE_NO_PHYSICAL_EVENT_CHART",
            "minimum_causal_edge_increment": None,
            "time_function_perturbation_margin": None,
            "diagnostic_ancestry_depth_minimum_increment": minimum_increment,
            "stable_causality_artifact_hash": _hash(depths),
        },
        "record_cauchy_completion": {
            "refinement_cauchy_residual": None,
            "frozen_residual_threshold": None,
            "every_cauchy_filter_has_record_germ": False,
            "open_image_local_degree_nonzero": False,
            "diagnostic_layout_tail_radius": cauchy_residual,
            "diagnostic_rung_scale": cauchy_threshold,
            "completion_artifact_hash": _hash({"levels": level_max, "positions": positions.tolist()}),
        },
        "h3_role": "diagnostic_observer_feature_fiber",
        "event_base_role": "presentation_layout_only_not_event_manifold",
        "h3_and_event_base_separate": False,
        "diagnostic_layout_hash_separation": frame_base_separate,
        "diagnostic_layout": {
            "physical_gate_eligible": False,
            "coordinate_source": (
                "carrier_graph_axes_plus_ancestry_depth_and_sequence"
            ),
            "event_keys": keys,
            "positions": positions.tolist(),
            "frames": h3_frames,
            "levels": levels,
            "radii": radii,
        },
        "frame_fiber_construction_hash": _hash(h3_frames),
        "event_base_construction_hash": _hash(positions.tolist()),
        "structural_receipt": stage_assessment["structural_receipt"],
        "instrument_receipt": stage_assessment["instrument_receipt"],
        "sensitivity_receipt": stage_assessment["sensitivity_receipt"],
        "diagnostic_findings": event_diagnostic_findings,
        "primitive_events": [
            {**row, "computed_event_id": computed_ids.get(str(row["event_key"]))}
            for row in events
        ],
        "scientific_failures": [],
    }
    return report, {
        "evaluated": False,
        "diagnostic_findings": event_diagnostic_findings,
        "not_evaluated_reasons": [_P7_REASON],
        "scientific_failures": [],
        "stage_assessment": stage_assessment,
    }


def _pair_rank_to_indices(rank: int, event_count: int) -> tuple[int, int]:
    """Invert a lexicographic upper-triangle pair rank exactly."""

    pair_count = event_count * (event_count - 1) // 2
    if event_count < 2 or not 0 <= rank < pair_count:
        raise PostrunCaptureError("diagnostic event-pair rank is out of range")
    lower = 0
    upper = event_count - 2
    while lower <= upper:
        middle = (lower + upper) // 2
        row_start = middle * (2 * event_count - middle - 1) // 2
        if row_start <= rank:
            lower = middle + 1
        else:
            upper = middle - 1
    left = upper
    row_start = left * (2 * event_count - left - 1) // 2
    right = left + 1 + (rank - row_start)
    if not 0 <= left < right < event_count:
        raise PostrunCaptureError("diagnostic event-pair rank inversion failed")
    return left, right


def _diagnostic_event_pair_sample(
    event_keys: Sequence[str],
    computed_event_ids: Mapping[str, str],
    plan: Mapping[str, Any],
) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    """Select a bounded, target-blind sample of unordered event pairs.

    The selection walks an affine permutation of pair ranks.  Coprimality of
    the stride makes every selected rank unique without allocating or sorting
    the full quadratic population.  Small instances remain an exact
    lexicographic census.
    """

    keys = [str(value) for value in event_keys]
    event_count = len(keys)
    population_pair_count = event_count * (event_count - 1) // 2
    if population_pair_count <= 0:
        raise PostrunCaptureError("event-pair diagnostic requires at least two events")
    split = _plain_mapping(plan.get("split_contract"), "plan.split_contract")
    salt = str(split.get("assignment_salt_sha256") or "")
    if not _strict_hash(salt):
        raise PostrunCaptureError("event-pair diagnostic split salt is malformed")
    population_sha256 = _hash(
        {
            "event_keys": keys,
            "computed_event_ids": dict(computed_event_ids),
        }
    )
    seed_bytes = hashlib.sha256(
        (
            f"{salt}\0{_P7_PAIR_SAMPLE_ALGORITHM}\0{population_sha256}"
        ).encode("utf-8")
    ).digest()
    selection_seed_sha256 = "sha256:" + seed_bytes.hex()
    evaluated_count = min(
        population_pair_count,
        _P7_MAX_DIAGNOSTIC_PAIR_COUNT,
    )
    full_census = evaluated_count == population_pair_count
    if full_census:
        ranks = list(range(population_pair_count))
        start_rank: int | None = 0
        stride: int | None = 1
        selection_mode = "exact_lexicographic_census"
    else:
        start_rank = int.from_bytes(seed_bytes[:16], "big") % population_pair_count
        stride = int.from_bytes(seed_bytes[16:], "big") % population_pair_count
        if stride == 0:
            stride = 1
        while math.gcd(stride, population_pair_count) != 1:
            stride = (stride + 1) % population_pair_count
            if stride == 0:
                stride = 1
        ranks = [
            (start_rank + index * stride) % population_pair_count
            for index in range(evaluated_count)
        ]
        selection_mode = "bounded_affine_permutation_prefix"
    sampled_pairs = [
        (rank, *_pair_rank_to_indices(rank, event_count))
        for rank in ranks
    ]
    event_pair_material = [
        [keys[left], keys[right]]
        for _, left, right in sampled_pairs
    ]
    provenance = {
        "scope": "postcapture_p7_diagnostic_only",
        "physical_gate_eligible": False,
        "algorithm_id": _P7_PAIR_SAMPLE_ALGORITHM,
        "selection_mode": selection_mode,
        "maximum_evaluated_pair_count": _P7_MAX_DIAGNOSTIC_PAIR_COUNT,
        "population_event_count": event_count,
        "population_pair_count": population_pair_count,
        "evaluated_pair_count": evaluated_count,
        "full_pair_census": full_census,
        "coverage_fraction": evaluated_count / population_pair_count,
        "assignment_salt_frozen_before_capture": True,
        "event_population_source_committed": True,
        "diagnostic_scope_preregistered_as_physical_instrument": False,
        "event_population_sha256": population_sha256,
        "selection_seed_sha256": selection_seed_sha256,
        "start_pair_rank": start_rank,
        "coprime_pair_rank_stride": stride,
        "sampled_pair_rank_sequence_sha256": _hash(ranks),
        "sampled_event_pair_sequence_sha256": _hash(event_pair_material),
    }
    return sampled_pairs, provenance


def _carrier_topology_coordinates(
    overlaps: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Build neutral landmark-distance coordinates from the overlap graph."""

    carriers = sorted(
        {
            str(row[field])
            for row in overlaps
            for field in ("left_carrier_id", "right_carrier_id")
        }
    )
    adjacency = {carrier: set() for carrier in carriers}
    for row in overlaps:
        left = str(row["left_carrier_id"])
        right = str(row["right_carrier_id"])
        adjacency[left].add(right)
        adjacency[right].add(left)

    def distances(origin: str) -> dict[str, int]:
        result = {origin: 0}
        queue = deque([origin])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(adjacency[node]):
                if neighbor not in result:
                    result[neighbor] = result[node] + 1
                    queue.append(neighbor)
        return result

    if not carriers:
        return {}, {}
    landmarks = [carriers[0]]
    distance_maps = [distances(landmarks[0])]
    while len(landmarks) < min(3, len(carriers)):
        candidate = max(
            (carrier for carrier in carriers if carrier not in landmarks),
            key=lambda carrier: (
                min(mapping.get(carrier, len(carriers)) for mapping in distance_maps),
                carrier,
            ),
        )
        landmarks.append(candidate)
        distance_maps.append(distances(candidate))
    while len(distance_maps) < 3:
        distance_maps.append(dict(distance_maps[-1]))
    diameter = max(
        (distance for mapping in distance_maps for distance in mapping.values()),
        default=1,
    )
    coordinates = {
        carrier: np.asarray(
            [mapping.get(carrier, diameter) / max(1, diameter) for mapping in distance_maps],
            dtype=float,
        )
        for carrier in carriers
    }
    matrix = np.asarray([coordinates[carrier] for carrier in carriers])
    matrix -= np.mean(matrix, axis=0, keepdims=True)
    scales = np.std(matrix, axis=0)
    scales[scales <= 1.0e-15] = 1.0
    matrix /= scales
    coordinates = {
        carrier: matrix[index] for index, carrier in enumerate(carriers)
    }
    max_degree = max((len(neighbors) for neighbors in adjacency.values()), default=1)
    degrees = {
        carrier: len(adjacency[carrier]) / max(1, max_degree) for carrier in carriers
    }
    return coordinates, degrees


def _verified_ancestry_rows(
    raw_rows: list[Any],
    events: Mapping[str, Mapping[str, Any]],
    event_edges: list[tuple[str, str]],
    reads: Mapping[str, set[str]],
    writes: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    expected_keys = {
        "parent_event_id",
        "child_event_id",
        "observer_token",
        "parent_sequence_index",
        "child_sequence_index",
        "shared_resource_ids",
        "edge_id",
    }
    rows = [_plain_mapping(row, "raw ancestry relation") for row in raw_rows]
    observed_edges: list[tuple[str, str]] = []
    for index, row in enumerate(rows):
        _require_exact_keys(row, expected_keys, f"raw_ancestry_relations[{index}]")
        material = {key: value for key, value in row.items() if key != "edge_id"}
        if row.get("edge_id") != _hash(material):
            raise PostrunCaptureError("raw ancestry edge commitment mismatch")
        parent = str(row.get("parent_event_id") or "")
        child = str(row.get("child_event_id") or "")
        if parent not in events or child not in events:
            raise PostrunCaptureError("raw ancestry edge names an absent event")
        if row.get("observer_token") != events[child].get("observer_token"):
            raise PostrunCaptureError("raw ancestry observer token mismatch")
        if int(row.get("parent_sequence_index", -1)) != int(
            events[parent].get("source_sequence_index", -2)
        ) or int(row.get("child_sequence_index", -1)) != int(
            events[child].get("source_sequence_index", -2)
        ):
            raise PostrunCaptureError("raw ancestry sequence index mismatch")
        shared = sorted(writes[parent].intersection(reads[child]))
        if list(row.get("shared_resource_ids", [])) != shared:
            raise PostrunCaptureError("raw ancestry read-after-write witness mismatch")
        observed_edges.append((parent, child))
    if sorted(observed_edges) != sorted(event_edges):
        raise PostrunCaptureError("raw ancestry relation set differs from semantic parents")
    return rows


def _verified_overlap_rows(raw_rows: list[Any]) -> list[dict[str, Any]]:
    expected_keys = {
        "overlap_id",
        "left_carrier_id",
        "right_carrier_id",
        "left_ports",
        "right_ports",
        "left_to_right_ports",
        "right_to_left_ports",
        "orientation_signs",
        "visible_to_observer_tokens",
        "interface_algebra_sha256",
        "row_sha256",
    }
    rows = [_plain_mapping(row, "raw overlap relation") for row in raw_rows]
    for index, row in enumerate(rows):
        _require_exact_keys(row, expected_keys, f"raw_overlap_relations[{index}]")
        _verify_row_hash(row, f"raw_overlap_relations[{index}]")
        if not _strict_hash(row.get("interface_algebra_sha256")):
            raise PostrunCaptureError("overlap interface algebra commitment is malformed")
        left = _sequence(row.get("left_ports"))
        right = _sequence(row.get("right_ports"))
        forward = _sequence(row.get("left_to_right_ports"))
        reverse = _sequence(row.get("right_to_left_ports"))
        signs = _sequence(row.get("orientation_signs"))
        if not left or not (len(left) == len(right) == len(forward) == len(reverse) == len(signs)):
            raise PostrunCaptureError("raw overlap port bijection is malformed")
    return rows


def _derive_poincare_transitions(
    overlaps: list[dict[str, Any]],
    events: list[dict[str, Any]],
    positions: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    del positions
    footprints = {
        str(event.get("event_key") or ""): set(
            str(value).split(":port-", 1)[0]
            for value in _sequence(event.get("visible_footprint"))
            if ":port-" in str(value)
        )
        for event in events
    }
    shared_chart_pairs: set[tuple[str, str]] = set()
    populated_charts: set[str] = set()
    for charts in footprints.values():
        ordered = sorted(charts)
        populated_charts.update(ordered)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                shared_chart_pairs.add((left, right))
    # The registered source presently exposes one carrier-local chart value per
    # semantic event.  No event has independent coordinates in both charts of
    # an overlap, so an affine/Poincare transition cannot yet be fitted.  In
    # particular, do not manufacture inverse/identity triangles that close by
    # construction.  A later source schema may supply shared correspondences.
    for row in overlaps:
        left = str(row["left_carrier_id"])
        right = str(row["right_carrier_id"])
        ordered_pair = tuple(sorted((left, right)))
        has_shared_event = (
            left in populated_charts
            if left == right
            else ordered_pair in shared_chart_pairs
        )
        if has_shared_event:
            raise PostrunCaptureError(
                "shared overlap events lack independent per-chart coordinate pairs"
            )
    return [], []


def _unavailable_quadratic_cone(
    relations: list[dict[str, Any]],
    positions: Mapping[str, np.ndarray],
    plan: Mapping[str, Any],
    *,
    pair_scope: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Quarantine ancestry-derived layout coordinates from cone inference.

    The current four coordinates are a presentation layout: three graph axes
    plus ancestry depth/sequence.  The causal/spacelike labels are also
    obtained from that same ancestry.  Fitting a quadratic form to those
    labels would be circular, so the physical cone input remains unavailable.
    """

    del plan
    layout_material = {
        "relations": relations,
        "positions": {
            key: np.asarray(value, dtype=float).tolist()
            for key, value in sorted(positions.items())
        },
        "pair_scope": dict(pair_scope),
    }
    return {
        "fit_row_ids": [],
        "heldout_row_ids": [],
        "inference_source": "UNAVAILABLE_NO_INDEPENDENT_EVENT_CHART",
        "relation_scope": (
            "exact_postcapture_diagnostic_census"
            if pair_scope.get("full_pair_census") is True
            else "bounded_postcapture_diagnostic_sample"
        ),
        "pair_sample_provenance": dict(pair_scope),
        "preassigned_lorentz_metric_used": False,
        "ambient_rank": None,
        "negative_eigenvalue_count": None,
        "positive_eigenvalue_count": None,
        "zero_eigenvalue_count": None,
        "heldout_quadratic_residual": None,
        "frozen_residual_threshold": None,
        "time_orientation_consistent": False,
        "cofinal_normalized_margin_lower_bound": None,
        "cofinal_tail_level_count": 0,
        "cone_inference_artifact_hash": None,
        "diagnostic_layout_artifact_hash": _hash(layout_material),
        "unavailable_reason": (
            "event_coordinates_and_relation_labels_share_ancestry_source"
        ),
    }, ["quadratic_event_cone_not_evaluated_no_independent_event_chart"]


def _legacy_quadratic_cone_diagnostic(
    relations: list[dict[str, Any]],
    positions: Mapping[str, np.ndarray],
    plan: Mapping[str, Any],
    *,
    pair_scope: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Quarantined circular layout fit retained only for archive diagnosis."""
    relation_ids = [str(row.get("row_id") or "") for row in relations]
    heldout = set(_semantic_holdout_ids(relation_ids, plan, domain="event-cone"))
    train_rows = [row for row in relations if str(row.get("row_id")) not in heldout]
    test_rows = [row for row in relations if str(row.get("row_id")) in heldout]
    def design(row: Mapping[str, Any]) -> tuple[list[float], float, np.ndarray]:
        left = np.asarray(positions[str(row["left_event_key"])], dtype=float)
        right = np.asarray(positions[str(row["right_event_key"])], dtype=float)
        delta = right - left
        features = [delta[0] ** 2, delta[1] ** 2, delta[2] ** 2, delta[3] ** 2,
                    2 * delta[0] * delta[1], 2 * delta[0] * delta[2], 2 * delta[0] * delta[3],
                    2 * delta[1] * delta[2], 2 * delta[1] * delta[3], 2 * delta[2] * delta[3]]
        target = {"causal": -1.0, "null": 0.0, "spacelike": 1.0}.get(str(row.get("relation")), 0.0)
        return [float(value) for value in features], target, delta
    if train_rows:
        design_rows = [design(row) for row in train_rows]
        x = np.asarray([row[0] for row in design_rows], dtype=float)
        y = np.asarray([row[1] for row in design_rows], dtype=float)
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
    else:
        coef = np.zeros(10, dtype=float)
    q = np.array([[coef[0], coef[4], coef[5], coef[6]], [coef[4], coef[1], coef[7], coef[8]],
                  [coef[5], coef[7], coef[2], coef[9]], [coef[6], coef[8], coef[9], coef[3]]], dtype=float)
    eigenvalues = np.linalg.eigvalsh(q)
    tolerance = 1.0e-8
    negatives = int(np.sum(eigenvalues < -tolerance))
    positives = int(np.sum(eigenvalues > tolerance))
    zeros = 4 - negatives - positives
    test_design = [design(row) for row in test_rows]
    residuals = [abs(float(np.asarray(features) @ coef) - target) for features, target, _ in test_design]
    margins = [
        -float(delta @ q @ delta) / max(float(delta @ delta), 1.0e-15)
        for (_, _, delta), row in zip(test_design, test_rows, strict=True)
        if row.get("relation") == "causal"
    ]
    time_oriented = all(
        delta[3] > 0.0
        for (_, _, delta), row in zip(test_design, test_rows, strict=True)
        if row.get("relation") == "causal"
    )
    threshold = 0.25
    residual = max(residuals, default=threshold + 1.0)
    margin = min(margins, default=0.0)
    failures: list[str] = []
    if (negatives, positives, zeros) != (1, 3, 0):
        failures.append("quadratic_event_cone_not_one_timelike_three_spacelike")
    if residual > threshold:
        failures.append("heldout_quadratic_cone_residual_failed")
    if margin <= 0.0:
        failures.append("quadratic_cone_cofinal_margin_not_positive")
    return {
        "fit_row_ids": [value for value in relation_ids if value not in heldout],
        "heldout_row_ids": [value for value in relation_ids if value in heldout],
        "inference_source": "semantic_event_relations",
        "relation_scope": (
            "exact_postcapture_diagnostic_census"
            if pair_scope.get("full_pair_census") is True
            else "bounded_postcapture_diagnostic_sample"
        ),
        "pair_sample_provenance": dict(pair_scope),
        "preassigned_lorentz_metric_used": False,
        "ambient_rank": int(np.linalg.matrix_rank(q, tol=tolerance)),
        "negative_eigenvalue_count": negatives,
        "positive_eigenvalue_count": positives,
        "zero_eigenvalue_count": zeros,
        "heldout_quadratic_residual": residual,
        "frozen_residual_threshold": threshold,
        "time_orientation_consistent": bool(time_oriented and margin > 0.0),
        "cofinal_normalized_margin_lower_bound": margin,
        "cofinal_tail_level_count": max(1, len(test_rows)),
        "cone_inference_artifact_hash": _hash({"relations": relations, "q": q.tolist()}),
    }, failures


def _verified_failure_mode(
    scientific_failures: Sequence[str],
) -> tuple[str | None, str | None]:
    predicates = sorted(
        {
            str(value)
            for value in scientific_failures
            if isinstance(value, str) and value
        }
    )
    if not predicates:
        return None, None
    predicate_hash = _hash(predicates)
    mode = f"verified_predicate_set_v1:{predicate_hash.split(':', 1)[1]}"
    evidence_hash = _hash(
        {"derivation": "verified_predicate_set_sha256_v1", "predicates": predicates}
    )
    return mode, evidence_hash


def _campaign_manifest(
    plan: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    component_hashes: Mapping[str, str],
    *,
    scientific_failures: list[str],
    evaluation_complete: bool,
) -> dict[str, Any]:
    seeds = [int(value) for value in _sequence(plan.get("seeds"))]
    rungs = [int(value) for value in _sequence(plan.get("rungs"))]
    replicates = [str(value) for value in _sequence(plan.get("replicate_ids"))]
    if len(set(seeds)) < 3 or tuple(rungs) != REQUIRED_RUNGS or not replicates:
        raise PostrunCaptureError("campaign plan must freeze >=3 seeds and exact 4k/16k/64k/256k rungs")
    archive = _plain_mapping(plan.get("archive_boundary"), "plan.archive_boundary")
    if (
        archive.get("frozen_before_source_capture") is not True
        or archive.get("retune_after_freeze") is not False
    ):
        raise PostrunCaptureError("campaign plan is not an immutable pre-run freeze")
    instrument_commit = str(plan.get("instrument_commit_sha256") or "")
    execution_plan_digest = str(plan.get("plan_sha256") or "")
    family_plan_digest = frozen_campaign_family_sha256(plan)
    if not _strict_hash(instrument_commit) or not _strict_hash(execution_plan_digest):
        raise PostrunCaptureError("instrument commit/container digest is missing")
    family_contract = {
        "instrument_commit": instrument_commit,
        "container_digest": family_plan_digest,
        "schema_versions": {
            "postrun": POSTRUN_SCHEMA_VERSION,
            "bw": BW_NATIVE_SCHEMA_VERSION,
            "event": "v2",
        },
        "source_protocol": {
            "generator": "target_blind_capture_v2",
            "frozen_family_plan_sha256": family_plan_digest,
            "producer_registry_sha256": _hash(plan.get("producer_registry")),
        },
        "feature_contract": {
            "candidate_only_in_scoring": True,
            "same_raw_response": True,
            "model_features_constructed_after_capture": True,
            "equal_footing_geometry_candidate_inputs": True,
            "geometry_predictor_response_fields_disjoint": True,
            "geometry_scores_physical_gate_eligible": False,
        },
        "model_families": {"geometry": list(REQUIRED_GEOMETRY_MODELS)},
        "loss_functions": {"clock": "paired_holdout_mse", "geometry": "paired_holdout_mse"},
        "thresholds": {
            "values": _hash(plan.get("thresholds")),
            "calibrations": _hash(plan.get("calibrations")),
        },
        "control_set": {"clock_candidates": list(REQUIRED_CLOCK_LABELS), "geometry_models": list(REQUIRED_GEOMETRY_MODELS)},
        "split_algorithm": dict(_plain_mapping(plan.get("split_contract"), "split_contract")),
        "seed_derivation": {"name": "campaign_plan_explicit_seed_v1"},
        "rung_scaling_laws": {
            "carrier_count": "exact_federation_cardinality",
            "support_regulator": "independent_nested_geodesic_chart",
            "registered_scaling_contract_sha256": _hash(plan.get("scaling_contract")),
        },
        "archive_boundary": {
            "archived_instrument_status": "FROZEN_NO_RETUNE",
            "archived_16k_failure_preserved": archive.get("archived_16k_failure_preserved") is True,
            "new_instrument_is_distinct_family": True,
            "archived_outcomes_used_for_threshold_selection": archive.get("archived_outcomes_used_for_threshold_selection") is True,
            "historical_receipt_byte_sha256": archive.get(
                "historical_receipt_byte_sha256"
            ),
            "historical_campaign_sha256": archive.get(
                "historical_campaign_sha256"
            ),
            "historical_16k_source_seed": archive.get(
                "historical_16k_source_seed"
            ),
            "historical_16k_rung": archive.get("historical_16k_rung"),
            "historical_16k_joint_independent_receipt": archive.get(
                "historical_16k_joint_independent_receipt"
            ),
            "historical_stable_branch_failure_established": archive.get(
                "historical_stable_branch_failure_established"
            ),
        },
        "retirement_rule": {
            "decisive_rungs": [16_384, 65_536, 262_144],
            "same_predeclared_failure_mode_required": True,
            "all_cells_powered_and_complete_required": True,
            "frozen_before_first_run": True,
            "failure_mode_derivation": "verified_predicate_set_sha256_v1",
        },
    }
    family_hash = _hash(family_contract)
    campaign_id = str(plan.get("campaign_id") or "")
    instrument_version = str(plan.get("instrument_version") or DEFAULT_INSTRUMENT_VERSION)
    protocol_hash = _hash({"family": family_contract, "campaign_id": campaign_id})
    current = _plain_mapping(plan.get("current_cell"), "current_cell")
    current_key = (int(current.get("seed", -1)), int(current.get("rung", -1)), str(current.get("replicate_id") or ""))
    run_matrix: list[dict[str, Any]] = []
    current_count = 0
    for frozen_raw in _sequence(plan.get("run_matrix")):
        frozen = _plain_mapping(frozen_raw, "frozen run row")
        cell = _plain_mapping(frozen.get("cell"), "frozen run cell")
        key = (int(cell.get("seed", -1)), int(cell.get("rung", -1)), str(cell.get("replicate_id") or ""))
        is_current = key == current_key
        current_count += int(is_current)
        row = {
            "seed": key[0],
            "rung": key[1],
            "carrier_count": key[1],
            "replicate_id": key[2],
            "campaign_id": campaign_id,
            "instrument_version": instrument_version,
            "instrument_commit": instrument_commit,
            "family_hash": family_hash,
            "protocol_hash": protocol_hash,
            "frozen_config_hash": frozen.get("config_sha256"),
            "preflight": "PASS" if is_current else "PENDING",
            "required_controls_complete": bool(is_current and evaluation_complete),
            "source_hashes_complete": is_current,
            "powered_and_complete": bool(is_current and evaluation_complete),
            "status": CellStatus.NOT_EVALUATED.value,
            "failure_mode": None,
        }
        if is_current:
            row["status"] = (
                CellStatus.INCOMPLETE.value
                if not evaluation_complete
                else CellStatus.VALID_FAIL.value
                if scientific_failures
                else CellStatus.VALID_PASS.value
            )
            row["result_artifact_hash"] = _hash(
                {
                    "components": component_hashes,
                    "failures": scientific_failures,
                    "evaluation_complete": evaluation_complete,
                }
            )
            if evaluation_complete and scientific_failures:
                failure_mode, failure_evidence_hash = _verified_failure_mode(
                    scientific_failures
                )
                row["failure_mode"] = failure_mode
                row["failure_evidence_hash"] = failure_evidence_hash
        run_matrix.append(row)
    if current_count != 1:
        raise PostrunCaptureError("current cell was not uniquely updated")
    return {
        "schema": "oph.physical_h3_kms.campaign_manifest.v1",
        "campaign_status": CellStatus.INCOMPLETE.value,
        "physical_promotion_allowed": False,
        "branch_retirement_authorized": False,
        "stable_failure_rule_satisfied": False,
        "scientific_failures": [],
        "campaign_id": campaign_id,
        "instrument_version": instrument_version,
        "campaign_family_hash": family_hash,
        "frozen_campaign_family_sha256": family_plan_digest,
        "execution_plan_sha256": execution_plan_digest,
        "family_contract": family_contract,
        "seeds": seeds,
        "rungs": rungs,
        "carrier_counts": rungs,
        "replicate_ids": replicates,
        "frozen_before_first_run": True,
        "retune_after_freeze": False,
        "retune_events": [],
        "run_matrix": run_matrix,
        "current_cell_capture_hash": _hash(
            {"components": component_hashes, "preregistration": _prerun_sha256(preregistration)}
        ),
        "claim_boundary": (
            "An incomplete cell with missing independent physical producers cannot "
            "promote the H3/KMS branch and cannot authorize branch retirement."
        ),
    }


def _poincare_cocycle_residual(transitions: list[dict[str, Any]], triangles: list[Any]) -> float:
    by_id = {str(row.get("transition_id")): row for row in transitions}
    residuals: list[float] = []
    for raw in triangles:
        row = _plain_mapping(raw, "cocycle_triangle")
        if not all(str(row.get(key)) in by_id for key in ("ab", "bc", "ac")):
            continue
        ab, bc, ac = (by_id[str(row[key])] for key in ("ab", "bc", "ac"))
        l_ab, l_bc, l_ac = (_matrix(item, "lorentz") for item in (ab, bc, ac))
        t_ab, t_bc, t_ac = (_vector(item, "translation", 4) for item in (ab, bc, ac))
        residuals.append(float(np.linalg.norm(l_bc @ l_ab - l_ac)))
        residuals.append(float(np.linalg.norm(l_bc @ t_ab + t_bc - t_ac)))
    return max(residuals, default=1.0)


def _fixed_capacity_matrix(rows: list[dict[str, Any]], model: str, capacity: int) -> np.ndarray:
    matrix = np.zeros((len(rows), capacity), dtype=float)
    matrix[:, 0] = 1.0
    for index, row in enumerate(rows):
        values = _feature_vector(row, model)
        if values.size < capacity - 1:
            raise PostrunCaptureError(f"{model} feature vector is below frozen capacity")
        matrix[index, 1:] = values[: capacity - 1]
    return matrix


def _feature_vector(row: Mapping[str, Any], model: str) -> np.ndarray:
    neutral = _vector(row, "neutral_feature_vector", 7)
    parameter = _number(row.get("geometric_parameter"), "geometric_parameter")
    return _model_feature_from_neutral(neutral, parameter, model)


def _model_feature_from_neutral(
    neutral: np.ndarray, parameter: float, model: str
) -> np.ndarray:
    s = float(np.clip(parameter, -8.0, 8.0))
    if model == "H3":
        values = [
            math.sinh(s),
            math.cosh(s) - 1.0,
            math.tanh(s),
            float(np.linalg.norm(neutral[:3])),
        ]
    elif model == "S2":
        values = [math.sin(s), math.cos(s), math.sin(2.0 * s), math.cos(2.0 * s)]
    elif model == "E3":
        values = [float(neutral[0]), float(neutral[1]), float(neutral[2]), s]
    elif model == "E4":
        values = [float(value) for value in neutral[3:7]]
    else:
        raise PostrunCaptureError(f"unknown geometry model: {model}")
    return np.asarray(values, dtype=float)


def curvature_calibration_commitment() -> str:
    """Identify the quarantined legacy synthetic sensitivity protocol.

    The commitment is retained so archived plans remain replayable.  The
    protocol is not executed by the physical post-run path: its response was
    defined with the H3 feature ``cosh(s)-1`` and therefore cannot establish an
    equal-footing H3/E3 comparison.
    """

    return _hash(
        {
            "protocol": "fixed_curved_alternative_power_suite_v1",
            "parameter_domain": [-4.0, 4.0],
            "trial_count": 64,
            "sample_count_per_trial": 64,
            "train_count_per_trial": 48,
            "noise_scale": 0.1,
            "registered_curvature_radius": 1.0,
            "models": ["H3", "E3"],
            "capacity": 5,
            "ridge": 1.0e-8,
        }
    )


def _dag_depths(keys: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    parents: dict[str, list[str]] = {key: [] for key in keys}
    for parent, child in edges:
        parents[child].append(parent)
    depth: dict[str, int] = {}
    for _ in keys:
        for key in keys:
            if key not in depth and all(parent in depth for parent in parents[key]):
                depth[key] = 0 if not parents[key] else 1 + max(depth[parent] for parent in parents[key])
    return depth


def _connected(edges: list[tuple[str, str]], nodes: set[str]) -> bool:
    if not nodes:
        return False
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    seen = {next(iter(nodes))}
    queue = deque(seen)
    while queue:
        node = queue.popleft()
        new = adjacency.get(node, set()) - seen
        seen.update(new)
        queue.extend(sorted(new))
    return nodes <= seen


def _semantic_holdout_ids(
    row_ids: Sequence[str], plan: Mapping[str, Any], *, domain: str
) -> tuple[str, ...]:
    if len(row_ids) != len(set(row_ids)):
        raise PostrunCaptureError(f"{domain} split row IDs are duplicated")
    split = _plain_mapping(plan.get("split_contract"), "plan.split_contract")
    if (
        split.get("algorithm_id") != "semantic_hash_split_v1"
        or split.get("derivation") != "semantic_event_id_hash_threshold_v1"
        or split.get("heldout_ids_materialized_before_capture") is not False
    ):
        raise PostrunCaptureError("semantic holdout split contract mismatch")
    salt = str(split.get("assignment_salt_sha256") or "")
    if not _strict_hash(salt):
        raise PostrunCaptureError("semantic holdout split salt is malformed")
    fraction = _number(split.get("holdout_fraction"), "holdout_fraction")
    if not 0.0 < fraction < 1.0:
        raise PostrunCaptureError("semantic holdout fraction must lie strictly in (0,1)")
    threshold = int(fraction * (2**256))
    heldout = tuple(
        row_id
        for row_id in row_ids
        if int.from_bytes(
            hashlib.sha256(
                f"{salt}\0{domain}\0{row_id}".encode("utf-8")
            ).digest(),
            "big",
        )
        < threshold
    )
    if len(heldout) < 2 or len(row_ids) - len(heldout) < 2:
        raise PostrunCaptureError(
            f"{domain} semantic hash split is underpowered for train/holdout scoring"
        )
    return heldout


def _complex_pair_matrix(value: Any, name: str) -> np.ndarray:
    try:
        pairs = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PostrunCaptureError(f"{name} must be a finite matrix of real/imag pairs") from exc
    if pairs.ndim != 3 or pairs.shape[-1] != 2 or not np.all(np.isfinite(pairs)):
        raise PostrunCaptureError(f"{name} must be a finite matrix of real/imag pairs")
    return pairs[..., 0] + 1j * pairs[..., 1]


def _target_leak_hits(
    value: Any,
    path: str = "$",
    *,
    key_context: bool = False,
) -> list[str]:
    del key_context
    hits: list[str] = []
    stack: list[tuple[Any, str]] = [(value, path)]
    while stack:
        current, current_path = stack.pop()
        if isinstance(current, Mapping):
            for raw_key, child in current.items():
                key = str(raw_key)
                child_path = f"{current_path}.{key}"
                normalized = _TARGET_TOKEN_NORMALIZER.sub(
                    "_", key.lower()
                ).strip("_")
                if (
                    normalized in _FORBIDDEN_ASSERTION_KEYS
                    or normalized.endswith("_receipt")
                    or normalized.endswith("_passed")
                ):
                    hits.append(f"{child_path}:caller_assertion")
                normalized_parts = set(normalized.split("_"))
                if (
                    normalized in _FORBIDDEN_SOURCE_TOKENS
                    or not _FORBIDDEN_SOURCE_TOKENS.isdisjoint(normalized_parts)
                ):
                    hits.append(f"{child_path}:target_token")
                stack.append((child, child_path))
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            stack.extend(
                (child, f"{current_path}[{index}]")
                for index, child in enumerate(current)
            )
        elif isinstance(current, str):
            normalized = _TARGET_TOKEN_NORMALIZER.sub(
                "_", current.lower()
            ).strip("_")
            if normalized in _FORBIDDEN_SOURCE_TOKENS:
                hits.append(f"{current_path}:target_value")
    return sorted(set(hits))


def _plain_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PostrunCaptureError(f"{name} must be a mapping")
    return dict(value)


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    missing = set(expected) - set(value)
    extra = set(value) - set(expected)
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if extra:
            details.append("extra=" + ",".join(sorted(extra)))
        raise PostrunCaptureError(f"{name} field set mismatch ({'; '.join(details)})")


def _verify_row_hash(row: Mapping[str, Any], name: str) -> None:
    declared = str(row.get("row_sha256") or "")
    material = {key: value for key, value in row.items() if key != "row_sha256"}
    if not _strict_hash(declared) or declared != _hash(material):
        raise PostrunCaptureError(f"{name} row commitment mismatch")


def _required_sequence(mapping: Mapping[str, Any], name: str) -> list[Any]:
    values = _sequence(mapping.get(name))
    if not values:
        raise PostrunCaptureError(f"{name} must be nonempty")
    return values


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else []


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PostrunCaptureError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PostrunCaptureError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise PostrunCaptureError(f"{name} must be a finite number")
    return result


def _positive_number(value: Any, name: str) -> float:
    result = _number(value, name)
    if result <= 0.0:
        raise PostrunCaptureError(f"{name} must be positive")
    return result


def _nonnegative_number(value: Any, name: str) -> float:
    result = _number(value, name)
    if result < 0.0:
        raise PostrunCaptureError(f"{name} must be nonnegative")
    return result


def _positive_float(mapping: Mapping[str, Any], name: str) -> float:
    return _positive_number(mapping.get(name), name)


def _nonnegative_float(mapping: Mapping[str, Any], name: str) -> float:
    return _nonnegative_number(mapping.get(name), name)


def _numbers(value: Any) -> list[float]:
    values = _sequence(value)
    if not values:
        raise PostrunCaptureError("expected a nonempty numeric sequence")
    return [_number(item, "sequence value") for item in values]


def _column(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    return np.asarray([_number(row.get(name), name) for row in rows], dtype=float)


def _vector(mapping: Mapping[str, Any], name: str, length: int | None = None) -> np.ndarray:
    return _vector_value(mapping.get(name), length, name)


def _vector_value(value: Any, length: int | None, name: str) -> np.ndarray:
    result = np.asarray(_numbers(value), dtype=float)
    if result.ndim != 1 or (length is not None and result.size != length):
        raise PostrunCaptureError(f"{name} has invalid vector shape")
    return result


def _matrix(mapping: Mapping[str, Any], name: str) -> np.ndarray:
    value = mapping.get(name)
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PostrunCaptureError(f"{name} must be a finite matrix") from exc
    if result.ndim != 2 or result.size == 0 or not np.all(np.isfinite(result)):
        raise PostrunCaptureError(f"{name} must be a finite matrix")
    return result


def _max_abs(value: Any) -> float:
    return max((abs(item) for item in _numbers(value)), default=math.inf)


def _max_value(value: Any) -> float:
    return max(_numbers(value))


def _min_value(value: Any) -> float:
    return min(_numbers(value))


def _all_positive(value: Any) -> bool:
    values = _numbers(value)
    return bool(values and all(item > 0.0 for item in values))


def _strictly_increasing(value: Any) -> bool:
    values = [int(item) for item in _sequence(value)]
    return len(values) >= 2 and all(right > left > 0 for left, right in zip(values, values[1:], strict=False))


def _hash(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PostrunCaptureError("value is not canonical finite JSON") from exc
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _strict_hash(value: Any) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


def _require_finite_json(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite_json(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _require_finite_json(child, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise PostrunCaptureError(f"nonfinite source value at {path}")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise PostrunCaptureError(f"non-JSON source value at {path}")


__all__ = [
    "CAPTURE_SCHEMA_VERSION",
    "POSTRUN_SCHEMA_VERSION",
    "PREREGISTRATION_ENVELOPE_SCHEMA",
    "PostrunCaptureError",
    "compute_postrun_reports",
    "curvature_calibration_commitment",
]
