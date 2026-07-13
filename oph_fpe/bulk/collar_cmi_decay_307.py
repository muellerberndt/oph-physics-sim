from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from oph_fpe.claims import (
    BRANCH_INSTANTIATION_SANITY,
    ISSUE_307_COLLAR_CMI_DECAY_RECEIPT,
    with_claim_metadata,
)


ALGORITHM_ID = "oph-issue307-collar-cmi-decay-v1"
RECEIPT_NAME = ISSUE_307_COLLAR_CMI_DECAY_RECEIPT

FINITE_RANGE_GIBBS_RECEIPT = "FINITE_RANGE_GIBBS_EVIDENCE_RECEIPT"
STRONG_MIXING_RECEIPT = "STRONG_CONDITIONAL_MATRIX_MIXING_RECEIPT"
REGIONAL_CMI_RECEIPT = "REGIONAL_COLLAR_CMI_EVIDENCE_RECEIPT"
CMI_BOUND_RECEIPT = "BOUNDARY_PREFACTORED_CMI_BOUND_RECEIPT"
DOUBLE_SCALING_RECEIPT = "SHARP_DOUBLE_SCALING_RATE_RECEIPT"

_IGNORED_PASS_KEYS = {
    RECEIPT_NAME,
    FINITE_RANGE_GIBBS_RECEIPT,
    STRONG_MIXING_RECEIPT,
    REGIONAL_CMI_RECEIPT,
    CMI_BOUND_RECEIPT,
    DOUBLE_SCALING_RECEIPT,
    "passed",
    "receipt",
    "theorem_grade_parent_collar_ladder",
    "EINSTEIN_BRANCH_ENTRY_RECEIPT",
    "CMI_TO_MODULAR_SOURCE_MATCHING_RECEIPT",
    "MODULAR_SOURCE_CHARGE_RECEIPT",
    "physical_claim",
}

_STRONG_MIXING_KINDS = {
    "strong_conditional_matrix_mixing",
    "uniform_conditional_matrix_mixing",
}
_MATRIX_NORMS = {"operator_norm", "trace_norm", "completely_bounded_norm"}
_STATE_SEMANTICS = {
    "noncommutative_density_matrix",
    "classical_exact_joint_distribution",
}
_CMI_ESTIMATORS = {
    "exact_density_matrix_entropy",
    "certified_matrix_entropy_interval",
    "exact_classical_joint_entropy",
}
_BOUNDARY_MEASURES = {"cut_edge_count", "unique_boundary_cell_count"}
_DELTA_METRICS = {"screen_geodesic", "graph_distance_physical"}


def issue307_collar_cmi_decay_report(
    payload: Mapping[str, Any],
    *,
    numerical_tolerance: float = 1.0e-9,
    bound_log_tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    """Audit a finite instantiation of the issue #307 collar-CMI theorem.

    The verifier consumes primitive, hash-pinned matrix/finite-graph evidence.
    It deliberately does not consume the simulator's spatial packet-triplet
    CMI diagnostic. Caller-provided pass flags are ignored, all collar bounds
    are recomputed in log space, and a positive result remains a finite branch
    instantiation rather than a continuum proof or a source/Einstein receipt.
    """

    tolerance = _nonnegative_tolerance(numerical_tolerance, "numerical_tolerance")
    log_tolerance = _nonnegative_tolerance(bound_log_tolerance, "bound_log_tolerance")
    primitive = _primitive_payload(payload)
    ignored = {key: payload.get(key) for key in _IGNORED_PASS_KEYS if key in payload}

    gibbs = _gibbs_clause(primitive.get("gibbs_evidence"), tolerance)
    mixing = _mixing_clause(
        primitive.get("mixing_evidence"),
        gibbs["details"],
        tolerance,
        log_tolerance,
    )
    scaling_parameters = _scaling_parameters(
        primitive.get("scaling_schedule"), tolerance
    )
    regional = _regional_cmi_clause(
        primitive.get("regional_cmi_evidence"),
        gibbs["details"],
        mixing["details"],
        scaling_parameters,
        tolerance,
        log_tolerance,
    )
    scaling = _double_scaling_clause(
        regional["details"],
        gibbs["details"],
        mixing["details"],
        scaling_parameters,
        tolerance,
    )

    clauses = {
        "finite_range_gibbs": gibbs,
        "strong_conditional_matrix_mixing": mixing,
        "regional_collar_cmi": _clause(
            regional["regional_passed"],
            regional["regional_blockers"],
            regional["details"],
        ),
        "boundary_prefactored_cmi_bound": _clause(
            regional["bound_passed"],
            regional["bound_blockers"],
            regional["details"],
        ),
        "sharp_double_scaling_rate": scaling,
    }
    receipt = all(row["passed"] for row in clauses.values())
    blockers = _unique(
        blocker
        for row in clauses.values()
        for blocker in row.get("blockers", [])
    )
    report = {
        "schema_version": 1,
        "mode": "issue_307_collar_cmi_decay_finite_audit",
        "algorithm": ALGORITHM_ID,
        "paper_provenance": {
            "issue": 307,
            "result_class": "finite_branch_instantiation_receipt",
        },
        "ignored_caller_pass_fields": ignored,
        "clauses": clauses,
        "promotion_blockers": blockers,
        FINITE_RANGE_GIBBS_RECEIPT: gibbs["passed"],
        STRONG_MIXING_RECEIPT: mixing["passed"],
        REGIONAL_CMI_RECEIPT: regional["regional_passed"],
        CMI_BOUND_RECEIPT: regional["bound_passed"],
        DOUBLE_SCALING_RECEIPT: scaling["passed"],
        RECEIPT_NAME: bool(receipt),
        # These are hard non-promotions, not missing optional fields.
        "EINSTEIN_BRANCH_ENTRY_RECEIPT": False,
        "CMI_TO_MODULAR_SOURCE_MATCHING_RECEIPT": False,
        "MODULAR_SOURCE_CHARGE_RECEIPT": False,
        "theorem_grade_parent_collar_ladder": False,
        "claim_boundary": (
            "Finite, hash-pinned branch-instantiation audit for the issue #307 sufficient route. "
            "It verifies declared finite-range Gibbs data, strong conditional matrix mixing, "
            "regional CMI values, the boundary-prefactored exponential bound, and a power-law "
            "double-scaling schedule with a positive rate margin. It is not a proof that OPH "
            "repair output is Gibbs, not a continuum-limit proof, not a CMI-to-stress/source "
            "matching theorem, and never promotes Einstein branch entry or a physical source."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=BRANCH_INSTANTIATION_SANITY,
        receipt=RECEIPT_NAME,
        physical_claim=False,
        observable_id="issue_307_finite_gibbs_collar_cmi_decay",
        fit_objective="verify_predeclared_gibbs_mixing_and_collar_bound_constants",
    )


def write_issue307_collar_cmi_decay_report(
    source: Path | Mapping[str, Any],
    out: Path | None = None,
) -> dict[str, Any]:
    """Write an issue #307 audit, with a path-friendly API for later CLI wiring."""

    if isinstance(source, Path):
        loaded = json.loads(source.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, Mapping) else {}
    else:
        payload = source
    report = issue307_collar_cmi_decay_report(payload)
    out_path = out
    if out_path is None and isinstance(source, Path):
        out_path = source.with_name("issue_307_collar_cmi_decay_report.json")
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _primitive_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("issue_307_primitive_fields", "collar_cmi_decay_307"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {key: value for key, value in payload.items() if key not in _IGNORED_PASS_KEYS}


def _gibbs_clause(value: Any, tolerance: float) -> dict[str, Any]:
    blockers: list[str] = []
    if not isinstance(value, Mapping):
        return _clause(False, ["finite_range_gibbs_evidence_missing"], {})

    semantics = str(value.get("state_semantics", ""))
    family_id = _text(value.get("interaction_family_id"))
    graph_metric_id = _text(value.get("graph_metric_id"))
    range_unit = str(value.get("range_unit", ""))
    beta = _number(value.get("beta"))
    interaction_range = _number(value.get("interaction_range_uv"))
    norm_bound = _number(value.get("max_term_norm_bound"))
    degree_bound = _positive_int(value.get("max_degree_bound"))
    stages_value = value.get("stages")

    if semantics not in _STATE_SEMANTICS:
        blockers.append("explicit_matrix_or_exact_joint_state_semantics_missing")
    if not family_id:
        blockers.append("interaction_family_id_missing")
    if not graph_metric_id:
        blockers.append("graph_metric_id_missing")
    if range_unit != "uv_cell":
        blockers.append("interaction_range_not_declared_in_uv_cells")
    if beta is None or beta <= 0.0:
        blockers.append("positive_finite_beta_missing")
    if interaction_range is None or interaction_range <= 0.0:
        blockers.append("positive_finite_interaction_range_missing")
    if norm_bound is None or norm_bound < 0.0:
        blockers.append("uniform_term_norm_bound_missing")
    if degree_bound is None:
        blockers.append("uniform_degree_bound_missing")
    if not isinstance(stages_value, list) or not stages_value:
        blockers.append("gibbs_stage_evidence_missing")
        stages_value = []

    stages: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_patch_counts: set[int] = set()
    for index, raw in enumerate(stages_value):
        prefix = f"gibbs_stage_{index}"
        if not isinstance(raw, Mapping):
            blockers.append(f"{prefix}_not_an_object")
            continue
        stage_id = _text(raw.get("stage_id"))
        patch_count = _positive_int(raw.get("patch_count"))
        stage_beta = _number(raw.get("beta"))
        stage_range = _number(raw.get("interaction_range_uv"))
        max_diameter = _number(raw.get("max_term_support_diameter_uv"))
        max_norm = _number(raw.get("max_term_norm"))
        max_degree = _positive_int(raw.get("max_degree"))
        term_count = _positive_int(raw.get("hamiltonian_term_count"))
        support_count = _positive_int(raw.get("term_support_count"))
        reconstruction = _number(raw.get("hamiltonian_reconstruction_residual"))
        gibbs_residual = _number(raw.get("gibbs_state_trace_residual"))
        terms_hash = raw.get("hamiltonian_terms_sha256")
        state_hash = raw.get("gibbs_state_sha256")

        if not stage_id or stage_id in seen_ids:
            blockers.append(f"{prefix}_stage_id_missing_or_duplicate")
        else:
            seen_ids.add(stage_id)
        if patch_count is None or patch_count in seen_patch_counts:
            blockers.append(f"{prefix}_patch_count_missing_or_duplicate")
        else:
            seen_patch_counts.add(patch_count)
        if beta is None or stage_beta is None or not _close(stage_beta, beta, tolerance):
            blockers.append(f"{prefix}_beta_not_uniform")
        if (
            interaction_range is None
            or stage_range is None
            or not _close(stage_range, interaction_range, tolerance)
        ):
            blockers.append(f"{prefix}_interaction_range_not_uniform")
        if (
            max_diameter is None
            or max_diameter < 0.0
            or interaction_range is None
            or max_diameter > interaction_range + tolerance
        ):
            blockers.append(f"{prefix}_nonlocal_term_support")
        if max_norm is None or max_norm < 0.0 or norm_bound is None or max_norm > norm_bound + tolerance:
            blockers.append(f"{prefix}_term_norm_exceeds_uniform_bound")
        if max_degree is None or degree_bound is None or max_degree > degree_bound:
            blockers.append(f"{prefix}_degree_exceeds_uniform_bound")
        if term_count is None or support_count != term_count:
            blockers.append(f"{prefix}_term_support_audit_incomplete")
        if reconstruction is None or abs(reconstruction) > tolerance:
            blockers.append(f"{prefix}_hamiltonian_reconstruction_mismatch")
        if gibbs_residual is None or abs(gibbs_residual) > tolerance:
            blockers.append(f"{prefix}_gibbs_state_mismatch")
        if not _sha256(terms_hash):
            blockers.append(f"{prefix}_hamiltonian_matrix_evidence_hash_missing")
        if not _sha256(state_hash):
            blockers.append(f"{prefix}_gibbs_state_matrix_evidence_hash_missing")

        stages.append(
            {
                "stage_id": stage_id,
                "patch_count": patch_count,
                "beta": stage_beta,
                "interaction_range_uv": stage_range,
                "max_term_support_diameter_uv": max_diameter,
                "max_term_norm": max_norm,
                "max_degree": max_degree,
                "hamiltonian_term_count": term_count,
                "term_support_count": support_count,
                "hamiltonian_reconstruction_residual": reconstruction,
                "gibbs_state_trace_residual": gibbs_residual,
                "hamiltonian_terms_sha256": terms_hash,
                "gibbs_state_sha256": state_hash,
            }
        )

    details = {
        "state_semantics": semantics,
        "interaction_family_id": family_id,
        "graph_metric_id": graph_metric_id,
        "range_unit": range_unit,
        "beta": beta,
        "interaction_range_uv": interaction_range,
        "max_term_norm_bound": norm_bound,
        "max_degree_bound": degree_bound,
        "stage_count": len(stages),
        "stages": stages,
    }
    return _clause(not blockers, blockers, details)


def _mixing_clause(
    value: Any,
    gibbs: Mapping[str, Any],
    tolerance: float,
    log_tolerance: float,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not isinstance(value, Mapping):
        return _clause(False, ["strong_conditional_matrix_mixing_evidence_missing"], {})

    kind = str(value.get("kind", ""))
    scope = str(value.get("constants_scope", ""))
    influence_norm = str(value.get("influence_norm", ""))
    c_mix = _number(value.get("c_mix"))
    xi_cells = _number(value.get("xi_uv_cells"))
    stage_values = value.get("stages")
    if kind not in _STRONG_MIXING_KINDS:
        blockers.append("ordinary_clustering_is_not_strong_conditional_matrix_mixing")
    if scope != "uniform_over_declared_stage_cap_boundary_family":
        blockers.append("mixing_constants_not_declared_uniform")
    if influence_norm not in _MATRIX_NORMS:
        blockers.append("conditional_matrix_influence_norm_missing")
    if c_mix is None or c_mix <= 0.0:
        blockers.append("positive_finite_c_mix_missing")
    if xi_cells is None or xi_cells <= 0.0:
        blockers.append("positive_finite_xi_uv_cells_missing")
    if not isinstance(stage_values, list) or not stage_values:
        blockers.append("mixing_stage_evidence_missing")
        stage_values = []

    expected_stages = {
        row.get("stage_id") for row in gibbs.get("stages", []) if row.get("stage_id")
    }
    seen_stages: set[str] = set()
    normalized: list[dict[str, Any]] = []
    interaction_range = _number(gibbs.get("interaction_range_uv"))
    for index, raw in enumerate(stage_values):
        prefix = f"mixing_stage_{index}"
        if not isinstance(raw, Mapping):
            blockers.append(f"{prefix}_not_an_object")
            continue
        stage_id = _text(raw.get("stage_id"))
        row_c = _number(raw.get("c_mix"))
        row_xi = _number(raw.get("xi_uv_cells"))
        boundary_count = _positive_int(raw.get("boundary_condition_count"))
        cap_count = _positive_int(raw.get("cap_count"))
        evidence_hash = raw.get("conditional_matrix_evidence_sha256")
        distance_values = raw.get("distance_rows")
        if not stage_id or stage_id in seen_stages:
            blockers.append(f"{prefix}_stage_id_missing_or_duplicate")
        else:
            seen_stages.add(stage_id)
        if c_mix is None or row_c is None or not _close(row_c, c_mix, tolerance):
            blockers.append(f"{prefix}_c_mix_not_uniform")
        if xi_cells is None or row_xi is None or not _close(row_xi, xi_cells, tolerance):
            blockers.append(f"{prefix}_xi_uv_cells_not_uniform")
        if boundary_count is None or boundary_count < 2:
            blockers.append(f"{prefix}_boundary_condition_coverage_insufficient")
        if cap_count is None:
            blockers.append(f"{prefix}_cap_coverage_missing")
        if not _sha256(evidence_hash):
            blockers.append(f"{prefix}_conditional_matrix_evidence_hash_missing")
        if not isinstance(distance_values, list) or len(distance_values) < 2:
            blockers.append(f"{prefix}_distance_ladder_insufficient")
            distance_values = []

        distances: list[dict[str, Any]] = []
        for distance_index, distance_raw in enumerate(distance_values):
            item_prefix = f"{prefix}_distance_{distance_index}"
            if not isinstance(distance_raw, Mapping):
                blockers.append(f"{item_prefix}_not_an_object")
                continue
            distance = _number(distance_raw.get("distance_uv"))
            influence = _number(distance_raw.get("conditional_matrix_influence_upper"))
            if distance is None or distance <= 0.0:
                blockers.append(f"{item_prefix}_positive_distance_missing")
            if influence is None or influence < 0.0:
                blockers.append(f"{item_prefix}_influence_upper_missing")
            log_bound = (
                math.log(c_mix) - distance / xi_cells
                if c_mix is not None
                and c_mix > 0.0
                and xi_cells is not None
                and xi_cells > 0.0
                and distance is not None
                else None
            )
            log_influence = _log_nonnegative(influence)
            bound_passed = bool(
                log_bound is not None
                and log_influence is not None
                and log_influence <= log_bound + log_tolerance
            )
            if not bound_passed:
                blockers.append(f"{item_prefix}_declared_mixing_bound_violated")
            distances.append(
                {
                    "distance_uv": distance,
                    "conditional_matrix_influence_upper": influence,
                    "recomputed_log_bound": log_bound,
                    "recomputed_log_slack": (
                        log_influence - log_bound
                        if log_influence is not None and log_bound is not None
                        else None
                    ),
                    "bound_passed": bound_passed,
                }
            )
        maximum_distance = max(
            (row["distance_uv"] for row in distances if row["distance_uv"] is not None),
            default=None,
        )
        if (
            interaction_range is None
            or maximum_distance is None
            or maximum_distance <= interaction_range
        ):
            blockers.append(f"{prefix}_mixing_not_checked_beyond_interaction_range")
        normalized.append(
            {
                "stage_id": stage_id,
                "c_mix": row_c,
                "xi_uv_cells": row_xi,
                "boundary_condition_count": boundary_count,
                "cap_count": cap_count,
                "conditional_matrix_evidence_sha256": evidence_hash,
                "distance_rows": distances,
            }
        )

    if expected_stages and seen_stages != expected_stages:
        blockers.append("mixing_stage_coverage_does_not_match_gibbs_family")
    details = {
        "kind": kind,
        "constants_scope": scope,
        "influence_norm": influence_norm,
        "c_mix": c_mix,
        "xi_uv_cells": xi_cells,
        "stage_count": len(normalized),
        "stages": normalized,
    }
    return _clause(not blockers, blockers, details)


def _scaling_parameters(value: Any, tolerance: float) -> dict[str, Any]:
    blockers: list[str] = []
    if not isinstance(value, Mapping):
        return {
            "blockers": ["power_law_scaling_schedule_missing"],
            "kind": None,
        }
    kind = str(value.get("kind", ""))
    screen_measure = _number(value.get("screen_measure"))
    uv_dimension = _positive_int(value.get("uv_dimension"))
    boundary_dimension = _nonnegative_int(value.get("boundary_dimension"))
    prefactor = _number(value.get("delta_prefactor"))
    alpha = _number(value.get("alpha"))
    boundary_prefactor = _number(value.get("boundary_prefactor_bound"))
    min_stage_count = _positive_int(value.get("min_stage_count"))
    final_margin = _number(value.get("min_final_log_decay_margin"))
    delta_metric = str(value.get("delta_metric", ""))
    if kind != "power_law_patch_count":
        blockers.append("scaling_schedule_is_not_a_declared_power_law")
    if screen_measure is None or screen_measure <= 0.0:
        blockers.append("positive_screen_measure_missing")
    if uv_dimension is None:
        blockers.append("positive_uv_dimension_missing")
    if (
        boundary_dimension is None
        or uv_dimension is None
        or boundary_dimension >= uv_dimension
    ):
        blockers.append("boundary_dimension_must_be_below_uv_dimension")
    if prefactor is None or prefactor <= 0.0:
        blockers.append("positive_delta_prefactor_missing")
    if alpha is None or alpha <= 0.0:
        blockers.append("positive_scaling_alpha_missing")
    if boundary_prefactor is None or boundary_prefactor <= 0.0:
        blockers.append("positive_boundary_prefactor_bound_missing")
    if min_stage_count is None or min_stage_count < 3:
        blockers.append("minimum_three_stage_scaling_ladder_required")
    if final_margin is None or final_margin < 0.0:
        blockers.append("nonnegative_final_log_decay_margin_missing")
    if delta_metric not in _DELTA_METRICS:
        blockers.append("declared_delta_metric_missing")
    return {
        "blockers": blockers,
        "kind": kind,
        "screen_measure": screen_measure,
        "uv_dimension": uv_dimension,
        "boundary_dimension": boundary_dimension,
        "delta_prefactor": prefactor,
        "alpha": alpha,
        "boundary_prefactor_bound": boundary_prefactor,
        "min_stage_count": min_stage_count,
        "min_final_log_decay_margin": final_margin,
        "delta_metric": delta_metric,
        "schedule_relative_tolerance": max(tolerance, 1.0e-12),
    }


def _regional_cmi_clause(
    value: Any,
    gibbs: Mapping[str, Any],
    mixing: Mapping[str, Any],
    scaling: Mapping[str, Any],
    tolerance: float,
    log_tolerance: float,
) -> dict[str, Any]:
    regional_blockers: list[str] = []
    bound_blockers: list[str] = []
    if not isinstance(value, Mapping):
        empty = {"rows": [], "scaling_schedule": dict(scaling)}
        return {
            "regional_passed": False,
            "regional_blockers": ["regional_cmi_evidence_missing"],
            "bound_passed": False,
            "bound_blockers": ["regional_cmi_rows_unavailable_for_bound"],
            "details": empty,
        }

    semantics = str(value.get("state_semantics", ""))
    log_unit = str(value.get("log_unit", ""))
    rows_value = value.get("rows")
    if semantics != gibbs.get("state_semantics") or semantics not in _STATE_SEMANTICS:
        regional_blockers.append("regional_state_semantics_do_not_match_gibbs_evidence")
    if log_unit != "nat":
        regional_blockers.append("regional_cmi_log_unit_must_be_nat")
    if not isinstance(rows_value, list) or not rows_value:
        regional_blockers.append("regional_cmi_rows_missing")
        rows_value = []

    stage_map = {
        row.get("stage_id"): row
        for row in gibbs.get("stages", [])
        if row.get("stage_id")
    }
    c_mix = _number(mixing.get("c_mix"))
    xi_cells = _number(mixing.get("xi_uv_cells"))
    screen_measure = _number(scaling.get("screen_measure"))
    dimension = _positive_int(scaling.get("uv_dimension"))
    boundary_dimension = _nonnegative_int(scaling.get("boundary_dimension"))
    delta_prefactor = _number(scaling.get("delta_prefactor"))
    alpha = _number(scaling.get("alpha"))
    boundary_prefactor = _number(scaling.get("boundary_prefactor_bound"))
    schedule_tolerance = _number(scaling.get("schedule_relative_tolerance")) or tolerance

    normalized: list[dict[str, Any]] = []
    seen_row_keys: set[tuple[str, str]] = set()
    for index, raw in enumerate(rows_value):
        prefix = f"regional_cmi_row_{index}"
        if not isinstance(raw, Mapping):
            regional_blockers.append(f"{prefix}_not_an_object")
            continue
        stage_id = _text(raw.get("stage_id"))
        cap_id = _text(raw.get("cap_id"))
        cap_family_id = _text(raw.get("cap_family_id"))
        patch_count = _positive_int(raw.get("patch_count"))
        delta = _number(raw.get("delta"))
        boundary_size = _number(raw.get("boundary_size_uv"))
        boundary_kind = str(raw.get("boundary_measure_kind", ""))
        estimator = str(raw.get("cmi_estimator", ""))
        supplied_cmi = _number(raw.get("regional_cmi_nats"))
        numerical_error = _number(raw.get("regional_cmi_numerical_error_nats"))
        entropy = raw.get("entropy_terms_nats")
        state_hash = raw.get("regional_state_evidence_sha256")
        boundary_hash = raw.get("boundary_evidence_sha256")
        regions = raw.get("regions")

        key = (stage_id or "", cap_id or "")
        if not stage_id or not cap_id or key in seen_row_keys:
            regional_blockers.append(f"{prefix}_stage_cap_key_missing_or_duplicate")
        else:
            seen_row_keys.add(key)
        if not cap_family_id:
            regional_blockers.append(f"{prefix}_cap_family_id_missing")
        stage = stage_map.get(stage_id)
        if stage is None:
            regional_blockers.append(f"{prefix}_stage_not_in_gibbs_family")
        if patch_count is None or stage is None or patch_count != stage.get("patch_count"):
            regional_blockers.append(f"{prefix}_patch_count_does_not_match_gibbs_stage")
        if delta is None or delta <= 0.0:
            regional_blockers.append(f"{prefix}_positive_collar_depth_missing")
        if (
            boundary_size is None
            or boundary_size <= 0.0
            or abs(boundary_size - round(boundary_size)) > tolerance
        ):
            regional_blockers.append(f"{prefix}_finite_boundary_uv_count_missing")
        if boundary_kind not in _BOUNDARY_MEASURES:
            regional_blockers.append(f"{prefix}_boundary_measure_kind_missing")
        if estimator not in _CMI_ESTIMATORS:
            regional_blockers.append(f"{prefix}_regional_cmi_estimator_not_exact_or_certified")
        if regions != ["A_delta", "B_delta", "D_delta"]:
            regional_blockers.append(f"{prefix}_regional_tripartition_not_declared")
        if not _sha256(state_hash):
            regional_blockers.append(f"{prefix}_regional_state_evidence_hash_missing")
        if not _sha256(boundary_hash):
            regional_blockers.append(f"{prefix}_boundary_evidence_hash_missing")
        if numerical_error is None or numerical_error < 0.0:
            regional_blockers.append(f"{prefix}_numerical_error_upper_missing")

        entropy_values: dict[str, float | None] = {}
        if isinstance(entropy, Mapping):
            entropy_values = {
                key_name: _number(entropy.get(key_name))
                for key_name in ("AB", "BD", "B", "ABD")
            }
        if not entropy_values or any(
            item is None or item < 0.0 for item in entropy_values.values()
        ):
            regional_blockers.append(f"{prefix}_regional_entropy_terms_missing")
            recomputed_cmi = None
        else:
            raw_cmi = (
                float(entropy_values["AB"])
                + float(entropy_values["BD"])
                - float(entropy_values["B"])
                - float(entropy_values["ABD"])
            )
            if raw_cmi < -tolerance:
                regional_blockers.append(f"{prefix}_strong_subadditivity_violated")
            recomputed_cmi = max(0.0, raw_cmi)
        if (
            supplied_cmi is None
            or supplied_cmi < -tolerance
            or recomputed_cmi is None
            or abs(max(0.0, supplied_cmi) - recomputed_cmi) > tolerance
        ):
            regional_blockers.append(f"{prefix}_regional_cmi_reconstruction_mismatch")

        ell_uv = (
            (screen_measure / patch_count) ** (1.0 / dimension)
            if screen_measure is not None
            and screen_measure > 0.0
            and patch_count is not None
            and dimension is not None
            else None
        )
        expected_delta = (
            delta_prefactor * patch_count ** (-alpha)
            if delta_prefactor is not None
            and patch_count is not None
            and alpha is not None
            else None
        )
        relative_schedule_residual = (
            abs(delta - expected_delta) / expected_delta
            if delta is not None and expected_delta is not None and expected_delta > 0.0
            else None
        )
        if relative_schedule_residual is None or relative_schedule_residual > schedule_tolerance:
            regional_blockers.append(f"{prefix}_collar_depth_not_on_declared_schedule")

        boundary_log_upper = (
            math.log(boundary_prefactor)
            + (boundary_dimension / dimension) * math.log(patch_count)
            if boundary_prefactor is not None
            and boundary_prefactor > 0.0
            and boundary_dimension is not None
            and dimension is not None
            and patch_count is not None
            else None
        )
        if (
            boundary_size is None
            or boundary_log_upper is None
            or math.log(boundary_size) > boundary_log_upper + log_tolerance
        ):
            regional_blockers.append(f"{prefix}_boundary_growth_exceeds_declared_polynomial_bound")

        cmi_upper = (
            recomputed_cmi + numerical_error
            if recomputed_cmi is not None and numerical_error is not None
            else None
        )
        log_rhs = (
            math.log(c_mix)
            + math.log(boundary_size)
            - delta / (xi_cells * ell_uv)
            if c_mix is not None
            and c_mix > 0.0
            and boundary_size is not None
            and boundary_size > 0.0
            and delta is not None
            and xi_cells is not None
            and xi_cells > 0.0
            and ell_uv is not None
            and ell_uv > 0.0
            else None
        )
        log_cmi_upper = _log_nonnegative(cmi_upper)
        bound_passed = bool(
            log_rhs is not None
            and log_cmi_upper is not None
            and log_cmi_upper <= log_rhs + log_tolerance
        )
        if not bound_passed:
            bound_blockers.append(f"{prefix}_boundary_prefactored_cmi_bound_violated")
        normalized.append(
            {
                "stage_id": stage_id,
                "cap_id": cap_id,
                "cap_family_id": cap_family_id,
                "patch_count": patch_count,
                "ell_uv": ell_uv,
                "delta": delta,
                "expected_delta": expected_delta,
                "delta_schedule_relative_residual": relative_schedule_residual,
                "delta_over_ell_uv": (
                    delta / ell_uv if delta is not None and ell_uv is not None else None
                ),
                "boundary_size_uv": boundary_size,
                "boundary_measure_kind": boundary_kind,
                "cmi_estimator": estimator,
                "regional_cmi_nats": recomputed_cmi,
                "regional_cmi_numerical_error_nats": numerical_error,
                "regional_cmi_upper_nats": cmi_upper,
                "recomputed_log_cmi_upper": log_cmi_upper,
                "recomputed_log_boundary_prefactored_bound": log_rhs,
                "recomputed_log_bound_slack": (
                    log_cmi_upper - log_rhs
                    if log_cmi_upper is not None and log_rhs is not None
                    else None
                ),
                "log_decay_rate_margin": -log_rhs if log_rhs is not None else None,
                "boundary_prefactored_bound_nats": _safe_exp(log_rhs),
                "fawzi_renner_trace_bound_from_cmi_upper": _fawzi_renner(cmi_upper),
                "bound_passed": bound_passed,
                "regional_state_evidence_sha256": state_hash,
                "boundary_evidence_sha256": boundary_hash,
            }
        )

    expected_stage_ids = set(stage_map)
    present_stage_ids = {row["stage_id"] for row in normalized if row["stage_id"]}
    if expected_stage_ids and present_stage_ids != expected_stage_ids:
        regional_blockers.append("regional_cmi_stage_coverage_does_not_match_gibbs_family")
    cap_families = {row["cap_family_id"] for row in normalized if row["cap_family_id"]}
    if len(cap_families) != 1:
        regional_blockers.append("mixed_cap_families_in_regional_cmi_ladder")
    cap_sets = {
        stage_id: {row["cap_id"] for row in normalized if row["stage_id"] == stage_id}
        for stage_id in present_stage_ids
    }
    if cap_sets and len({tuple(sorted(items)) for items in cap_sets.values()}) != 1:
        regional_blockers.append("cap_set_changes_across_regulator_stages")
    if c_mix is None or xi_cells is None:
        bound_blockers.append("uniform_mixing_constants_unavailable_for_cmi_bound")
    if regional_blockers:
        bound_blockers.append("regional_cmi_inputs_not_receipt_grade")

    details = {
        "state_semantics": semantics,
        "log_unit": log_unit,
        "row_count": len(normalized),
        "cap_family_ids": sorted(cap_families),
        "rows": normalized,
        "scaling_schedule": dict(scaling),
    }
    return {
        "regional_passed": not regional_blockers,
        "regional_blockers": _unique(regional_blockers),
        "bound_passed": not bound_blockers,
        "bound_blockers": _unique(bound_blockers),
        "details": details,
    }


def _double_scaling_clause(
    regional: Mapping[str, Any],
    gibbs: Mapping[str, Any],
    mixing: Mapping[str, Any],
    scaling: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    blockers = list(scaling.get("blockers", []))
    rows = list(regional.get("rows", []) or [])
    dimension = _positive_int(scaling.get("uv_dimension"))
    alpha = _number(scaling.get("alpha"))
    minimum_stages = _positive_int(scaling.get("min_stage_count")) or 3
    final_margin_min = _number(scaling.get("min_final_log_decay_margin"))
    power_margin = (
        1.0 / dimension - alpha
        if dimension is not None and alpha is not None
        else None
    )
    if alpha is None or alpha <= 0.0:
        blockers.append("delta_does_not_shrink")
    if power_margin is None or power_margin <= 0.0:
        blockers.append("delta_over_ell_uv_does_not_diverge_with_positive_power_margin")

    patch_counts = sorted(
        {row["patch_count"] for row in rows if row.get("patch_count") is not None}
    )
    if len(patch_counts) < minimum_stages:
        blockers.append("insufficient_distinct_regulator_stages")
    gibbs_counts = sorted(
        {
            row["patch_count"]
            for row in gibbs.get("stages", [])
            if row.get("patch_count") is not None
        }
    )
    if patch_counts and patch_counts != gibbs_counts:
        blockers.append("scaling_patch_counts_do_not_match_gibbs_family")

    cap_ids = sorted({row.get("cap_id") for row in rows if row.get("cap_id")})
    cap_ladders: dict[str, dict[str, Any]] = {}
    for cap_id in cap_ids:
        cap_rows = sorted(
            (row for row in rows if row.get("cap_id") == cap_id),
            key=lambda row: int(row.get("patch_count") or 0),
        )
        deltas = [row.get("delta") for row in cap_rows]
        ratios = [row.get("delta_over_ell_uv") for row in cap_rows]
        margins = [row.get("log_decay_rate_margin") for row in cap_rows]
        log_bounds = [row.get("recomputed_log_boundary_prefactored_bound") for row in cap_rows]
        delta_shrinks = _strictly_decreasing(deltas, tolerance)
        ratio_grows = _strictly_increasing(ratios, tolerance)
        margin_grows = _strictly_increasing(margins, tolerance)
        envelope_shrinks = _strictly_decreasing(log_bounds, tolerance)
        final_margin = margins[-1] if margins else None
        final_margin_passed = bool(
            final_margin is not None
            and final_margin_min is not None
            and final_margin >= final_margin_min - tolerance
        )
        if not delta_shrinks:
            blockers.append(f"cap_{cap_id}_delta_does_not_shrink")
        if not ratio_grows:
            blockers.append(f"cap_{cap_id}_delta_over_ell_does_not_grow")
        if not margin_grows or not envelope_shrinks:
            blockers.append(f"cap_{cap_id}_boundary_prefactor_rate_margin_does_not_improve")
        if not final_margin_passed:
            blockers.append(f"cap_{cap_id}_final_log_decay_margin_below_threshold")
        cap_ladders[cap_id] = {
            "patch_counts": [row.get("patch_count") for row in cap_rows],
            "delta_shrinks": delta_shrinks,
            "delta_over_ell_uv_grows": ratio_grows,
            "boundary_prefactor_rate_margin_grows": margin_grows,
            "log_envelope_shrinks": envelope_shrinks,
            "final_log_decay_rate_margin": final_margin,
            "final_margin_passed": final_margin_passed,
        }

    c_mix = _number(mixing.get("c_mix"))
    xi_cells = _number(mixing.get("xi_uv_cells"))
    if c_mix is None or c_mix <= 0.0 or xi_cells is None or xi_cells <= 0.0:
        blockers.append("uniform_positive_mixing_constants_missing_from_scaling_family")
    if not _text(gibbs.get("interaction_family_id")):
        blockers.append("uniform_interaction_family_missing_from_scaling_family")
    details = {
        "schedule_kind": scaling.get("kind"),
        "patch_counts": patch_counts,
        "stage_count": len(patch_counts),
        "alpha": alpha,
        "uv_dimension": dimension,
        "delta_over_ell_power_margin": power_margin,
        "boundary_dimension": scaling.get("boundary_dimension"),
        "boundary_prefactor_bound": scaling.get("boundary_prefactor_bound"),
        "min_final_log_decay_margin": final_margin_min,
        "cap_ladders": cap_ladders,
        "asymptotic_log_envelope": (
            "log(c_mix*K_boundary) + (boundary_dimension/uv_dimension)*log(N) "
            "- [delta_prefactor/(xi_uv*screen_measure^(1/uv_dimension))]"
            "*N^(1/uv_dimension-alpha)"
        ),
    }
    return _clause(not blockers, _unique(blockers), details)


def _clause(passed: bool, blockers: list[str], details: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "blockers": _unique(blockers),
        "details": dict(details),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive_int(value: Any) -> int | None:
    number = _number(value)
    if number is None or number <= 0.0 or not number.is_integer():
        return None
    return int(number)


def _nonnegative_int(value: Any) -> int | None:
    number = _number(value)
    if number is None or number < 0.0 or not number.is_integer():
        return None
    return int(number)


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _sha256(value: Any) -> bool:
    text = str(value).lower() if value is not None else ""
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _close(left: float, right: float, tolerance: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= tolerance * scale


def _log_nonnegative(value: float | None) -> float | None:
    if value is None or value < 0.0:
        return None
    if value == 0.0:
        return -math.inf
    return math.log(value)


def _safe_exp(log_value: float | None) -> float | None:
    if log_value is None:
        return None
    if log_value <= -745.0:
        return 0.0
    if log_value >= 709.0:
        return math.inf
    return math.exp(log_value)


def _fawzi_renner(cmi_nats: float | None) -> float | None:
    if cmi_nats is None or cmi_nats < 0.0:
        return None
    return min(2.0, 2.0 * math.sqrt(max(0.0, -math.expm1(-cmi_nats))))


def _strictly_increasing(values: list[Any], tolerance: float) -> bool:
    parsed = [_number(value) for value in values]
    return bool(
        len(parsed) >= 2
        and all(value is not None for value in parsed)
        and all(
            float(right) > float(left) + tolerance
            for left, right in zip(parsed, parsed[1:])
        )
    )


def _strictly_decreasing(values: list[Any], tolerance: float) -> bool:
    parsed = [_number(value) for value in values]
    return bool(
        len(parsed) >= 2
        and all(value is not None for value in parsed)
        and all(
            float(right) < float(left) - tolerance
            for left, right in zip(parsed, parsed[1:])
        )
    )


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in result:
            result.append(text)
    return result


def _nonnegative_tolerance(value: float, name: str) -> float:
    parsed = _number(value)
    if parsed is None or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed
