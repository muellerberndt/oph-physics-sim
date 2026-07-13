from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from oph_fpe.bulk.cap_geometry import RoundCap, cap_weights, lambda_cap
from oph_fpe.bulk.modular_probe import geometric_permutation_operator
from oph_fpe.gauge.covariant_overlap import (
    covariant_mismatch_mask,
    gauge_invariant_edge_residual,
    group_multiply_indices,
    repair_covariant_port_pairs,
)


def transition_scale_selection_report(
    points: np.ndarray,
    caps: list[RoundCap],
    raw_fields: dict[str, np.ndarray],
    *,
    times: list[float],
    observables: list[str],
    candidate_scales: list[float] | None = None,
    sources: list[str] | None = None,
    declared_response_scale: float = 2.0 * math.pi,
    max_basis: int = 64,
    seed: int = 1,
    graph_response: dict[str, np.ndarray | int | float] | None = None,
    probe_steps: int = 4,
    probe_repairs_per_source: int = 64,
    probe_max_incident_edges: int = 8,
    kms_response_scale: float | None = None,
    kms_transport_steps: int = 8,
) -> dict[str, Any]:
    """Score whether a cap-response operator selects the BW 2*pi scale.

    `perturb_remeasure_response` builds a support-visible permutation from
    bounded cap-local port perturbations, local repair, and remeasurement.
    `repair_affinity_response` is a cheaper fallback from repair/collar packet
    affinities. Candidate geometric scales are external targets in both cases,
    keeping scale selection separate from the declared KMS/BW automorphism
    sanity path.
    """

    scales = [float(value) for value in (candidate_scales or [1.0, math.pi, 2.0 * math.pi, 4.0 * math.pi])]
    source_names = [str(value) for value in (sources or ["repair_affinity_response"])]
    rows: list[dict[str, Any]] = []
    source_reports: dict[str, Any] = {}
    for source_index, source in enumerate(source_names):
        source_rows: list[dict[str, Any]] = []
        for cap_id, cap in enumerate(caps):
            basis = _cap_basis(points, cap, max_basis=max_basis, seed=seed + cap_id * 1009 + source_index * 7919)
            if source == "declared_geometric_sanity":
                response_pullback, response_eta, response_meta = _declared_geometric_pullback(
                    points,
                    cap,
                    basis,
                    response_time=_reference_time(times),
                    response_scale=declared_response_scale,
                )
            elif source == "repair_affinity_response":
                response_pullback, response_eta, response_meta = _repair_affinity_pullback(points, cap, raw_fields, basis)
            elif source == "perturb_remeasure_response":
                if graph_response is None:
                    raise ValueError("perturb_remeasure_response requires graph_response data")
                response_pullback, response_eta, response_meta = _perturb_remeasure_pullback(
                    points,
                    cap,
                    raw_fields,
                    basis,
                    graph_response=graph_response,
                    seed=seed + cap_id * 1009 + source_index * 7919,
                    probe_steps=probe_steps,
                    probe_repairs_per_source=probe_repairs_per_source,
                    probe_max_incident_edges=probe_max_incident_edges,
                )
            elif source == "kms_collar_transport_response":
                if graph_response is None:
                    raise ValueError("kms_collar_transport_response requires graph_response data")
                response_pullback, response_eta, response_meta = _perturb_remeasure_pullback(
                    points,
                    cap,
                    raw_fields,
                    basis,
                    graph_response=graph_response,
                    seed=seed + cap_id * 1009 + source_index * 7919,
                    probe_steps=probe_steps,
                    probe_repairs_per_source=probe_repairs_per_source,
                    probe_max_incident_edges=probe_max_incident_edges,
                    response_source="kms_collar_transport_response",
                    cap_flow_scale=float(kms_response_scale if kms_response_scale is not None else declared_response_scale),
                    cap_flow_time=_reference_time(times),
                    cap_flow_steps=kms_transport_steps,
                )
            else:
                raise ValueError(f"unsupported transition selection source: {source}")
            for time_value in times:
                for scale in scales:
                    geometric_transition, eta_target = geometric_permutation_operator(
                        points,
                        cap,
                        float(scale) * float(time_value),
                        basis,
                    )
                    target_pullback = geometric_transition.conj().T
                    operator_residual = _relative_frobenius(response_pullback - target_pullback, target_pullback)
                    observable_residuals = []
                    for observable in observables:
                        O = _observable_matrix(raw_fields, observable, basis)
                        O_response = response_pullback @ O @ response_pullback.conj().T
                        O_target = target_pullback @ O @ target_pullback.conj().T
                        observable_residuals.append(_relative_frobenius(O_response - O_target, O))
                    support_visible_residual = (
                        float(np.median(observable_residuals)) if observable_residuals else operator_residual
                    )
                    row = {
                        "source": source,
                        "cap_id": int(cap_id),
                        "time": float(time_value),
                        "candidate_label": _scale_label(scale),
                        "candidate_scale": float(scale),
                        "support_visible_residual": float(support_visible_residual),
                        "operator_residual": float(operator_residual),
                        "eta_response": float(response_eta),
                        "eta_target_interpolation": float(eta_target),
                        "selection_score": float(support_visible_residual + 0.25 * operator_residual),
                        "basis_count": int(basis.size),
                        "observable_count": int(len(observables)),
                        "matrix_element_count": int(basis.size * basis.size),
                        **response_meta,
                    }
                    rows.append(row)
                    source_rows.append(row)
        source_reports[source] = _summarize_source(source_rows, scales)
    primary_source = next(
        (
            source
            for source in ("kms_collar_transport_response", "perturb_remeasure_response", "repair_affinity_response")
            if source in source_reports
        ),
        source_names[0],
    )
    primary = source_reports.get(primary_source, {})
    return {
        "mode": "transition_scale_selection",
        "primary_source": primary_source,
        "candidate_scales": [{"label": _scale_label(scale), "scale": float(scale)} for scale in scales],
        "selected_label": primary.get("selected_label"),
        "selected_scale": primary.get("selected_scale"),
        "two_pi_selected": bool(primary.get("two_pi_selected", False)),
        "two_pi_score": primary.get("two_pi_score"),
        "best_score": primary.get("best_score"),
        "two_pi_over_best": primary.get("two_pi_over_best"),
        "two_pi_minus_best": primary.get("two_pi_minus_best"),
        "response_degenerate": bool(primary.get("response_degenerate", False)),
        "source_reports": source_reports,
        "row_count": int(len(rows)),
        "rows": rows,
        "normalization_source": (
            "kms_bw_collar_transport_for_primary"
            if primary_source == "kms_collar_transport_response"
            else "repair_collar_observer_fields_for_primary; declared source is sanity only"
        ),
        "claim_boundary": (
            "finite transition-scale diagnostic. The perturb_remeasure_response and "
            "repair_affinity_response sources do not build their simulated transition from "
            "lambda_C(2*pi*t); they test whether observer-visible repair and collar responses "
            "select that scale. The kms_collar_transport_response source is a branch-accurate "
            "KMS/BW-normalized collar-transport instantiation, not an endogenous selection proof. "
            "Passing declared_geometric_sanity is only an automorphism plumbing check, not a 3D "
            "bulk or early-universe claim."
        ),
    }


def _cap_basis(points: np.ndarray, cap: RoundCap, *, max_basis: int, seed: int) -> np.ndarray:
    weights = cap_weights(points, cap, soft=True)
    support = np.flatnonzero(weights > 0.05)
    if support.size == 0:
        support = np.arange(points.shape[0], dtype=np.int64)
    if support.size <= max_basis:
        return np.sort(support.astype(np.int64))
    rng = np.random.default_rng(seed)
    probs = weights[support].astype(float)
    probs = probs / max(float(np.sum(probs)), 1e-15)
    return np.sort(rng.choice(support, size=max_basis, replace=False, p=probs).astype(np.int64))


def _declared_geometric_pullback(
    points: np.ndarray,
    cap: RoundCap,
    basis: np.ndarray,
    *,
    response_time: float,
    response_scale: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    transition, eta = geometric_permutation_operator(points, cap, float(response_scale) * float(response_time), basis)
    return transition.conj().T, eta, {
        "response_source": "declared_geometric_sanity",
        "declared_response_scale": float(response_scale),
        "declared_response_time": float(response_time),
        "identity_fraction": float(np.mean(np.argmax(np.abs(transition), axis=1) == np.arange(transition.shape[0]))),
    }


def _repair_affinity_pullback(
    points: np.ndarray,
    cap: RoundCap,
    raw_fields: dict[str, np.ndarray],
    basis: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    cap = cap.normalized()
    basis_points = points[basis]
    spatial = np.linalg.norm(basis_points[:, None, :] - basis_points[None, :, :], axis=2)
    spatial_scale = _positive_median(spatial)
    spatial = spatial / spatial_scale

    features = _response_features(points, cap, raw_fields, basis)
    feature_delta = features[:, None, :] - features[None, :, :]
    feature_dist = np.sqrt(np.mean(feature_delta**2, axis=2)) if features.size else np.zeros_like(spatial)

    response_score = _response_score(raw_fields, basis)
    descent_bonus = np.maximum(response_score[:, None] - response_score[None, :], 0.0)
    collar_distance = np.abs(points[basis] @ cap.axis - math.cos(cap.theta0)) / max(cap.collar_width, 1e-6)
    collar_preference = np.exp(-0.5 * np.square(collar_distance))
    collar_bonus = 0.5 * (collar_preference[:, None] + collar_preference[None, :])

    cost = 0.65 * spatial + 0.45 * feature_dist - 0.18 * descent_bonus - 0.04 * collar_bonus
    np.fill_diagonal(cost, np.diag(cost) + 0.02)
    row_ind, col_ind = linear_sum_assignment(cost)
    transition = np.zeros((basis.size, basis.size), dtype=complex)
    transition[row_ind, col_ind] = 1.0
    pullback = transition.conj().T
    identity_fraction = float(np.mean(col_ind[np.argsort(row_ind)] == np.arange(basis.size))) if basis.size else 1.0
    return pullback, 0.0, {
        "response_source": "repair_affinity_response",
        "identity_fraction": identity_fraction,
        "mean_assignment_cost": float(np.mean(cost[row_ind, col_ind])) if cost.size else 0.0,
        "response_score_std": float(np.std(response_score)) if response_score.size else 0.0,
        "response_feature_count": int(features.shape[1]) if features.ndim == 2 else 0,
    }


def _perturb_remeasure_pullback(
    points: np.ndarray,
    cap: RoundCap,
    raw_fields: dict[str, np.ndarray],
    basis: np.ndarray,
    *,
    graph_response: dict[str, np.ndarray | int | float],
    seed: int,
    probe_steps: int,
    probe_repairs_per_source: int,
    probe_max_incident_edges: int,
    response_source: str = "perturb_remeasure_response",
    cap_flow_scale: float | None = None,
    cap_flow_time: float = 0.1,
    cap_flow_steps: int = 8,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    missing_graph_fields = [
        key
        for key in ("gauge", "group_name", "group_order")
        if key not in graph_response
    ]
    sector_repair_enabled = bool(graph_response.get("production_sector_repair_enabled", False))
    if missing_graph_fields or sector_repair_enabled:
        blockers = [f"gauge_coupled_graph_field_missing:{key}" for key in missing_graph_fields]
        if sector_repair_enabled:
            blockers.append("production_sector_repair_not_replayed_by_response_probe")
        return np.eye(len(basis), dtype=complex), 0.0, {
            "response_source": f"{response_source}_fail_closed",
            "identity_fraction": 1.0,
            "gauge_covariant_probe_receipt": False,
            "proof_blockers": blockers,
        }
    left = np.asarray(graph_response["left"], dtype=np.int64)
    right = np.asarray(graph_response["right"], dtype=np.int64)
    port_left = np.asarray(graph_response["port_left"], dtype=np.int64)
    port_right = np.asarray(graph_response["port_right"], dtype=np.int64)
    gauge = np.asarray(graph_response["gauge"], dtype=np.int16)
    group_name = str(graph_response["group_name"])
    group_order = int(graph_response.get("group_order", 6))
    patch_count = int(graph_response.get("patch_count", points.shape[0]))
    incident_edges = _incident_edges(left, right, patch_count)
    rng = np.random.default_rng(seed)
    node_score = _response_score(raw_fields, np.arange(patch_count, dtype=np.int64))
    before_residual = gauge_invariant_edge_residual(
        port_left,
        port_right,
        gauge,
        group_name=group_name,
        group_order=group_order,
    )
    before_signature = _node_packet_signature(before_residual, before_residual, left, right, patch_count)
    response_matrix = np.zeros((basis.size, basis.size), dtype=float)
    total_perturbed_edges = 0
    total_repaired_edges = 0
    exact_basis_masses: list[float] = []
    projected_basis_masses: list[float] = []
    projection_fallbacks = 0
    source_echo_masses: list[float] = []
    source_echo_suppressed = 0

    for row, node in enumerate(basis):
        pl = port_left.copy()
        pr = port_right.copy()
        incident = np.asarray(incident_edges[int(node)], dtype=np.int64)
        if incident.size > probe_max_incident_edges:
            incident = rng.choice(incident, size=probe_max_incident_edges, replace=False)
        perturb_left = left[incident] == node
        left_edges = incident[perturb_left]
        right_edges = incident[~perturb_left]
        if left_edges.size:
            pl[left_edges] = group_multiply_indices(
                pl[left_edges],
                np.ones(left_edges.size, dtype=np.int16),
                group_name=group_name,
                group_order=group_order,
            )
        if right_edges.size:
            pr[right_edges] = group_multiply_indices(
                pr[right_edges],
                np.ones(right_edges.size, dtype=np.int16),
                group_name=group_name,
                group_order=group_order,
            )
        total_perturbed_edges += int(incident.size)
        repair_count = np.zeros(patch_count, dtype=float)
        for _ in range(max(1, int(probe_steps))):
            active = np.flatnonzero(
                covariant_mismatch_mask(
                    pl,
                    pr,
                    gauge,
                    group_name=group_name,
                    group_order=group_order,
                )
            )
            if active.size == 0:
                break
            if active.size > probe_repairs_per_source:
                active = rng.choice(active, size=probe_repairs_per_source, replace=False)
            left_score = node_score[left[active]]
            right_score = node_score[right[active]]
            repair_left = left_score >= right_score
            repair_covariant_port_pairs(
                pl,
                pr,
                gauge,
                active,
                repair_left,
                group_name=group_name,
                group_order=group_order,
            )
            repair_count += np.bincount(left[active], minlength=patch_count)
            repair_count += np.bincount(right[active], minlength=patch_count)
            total_repaired_edges += int(active.size)
        after_residual = gauge_invariant_edge_residual(
            pl,
            pr,
            gauge,
            group_name=group_name,
            group_order=group_order,
        )
        after_signature = _node_packet_signature(after_residual, after_residual, left, right, patch_count)
        changed = (after_signature != before_signature).astype(float)
        response = repair_count + 0.5 * changed
        exact_basis_masses.append(float(np.sum(response[basis])))
        basis_response = _project_patch_response_to_basis(points, basis, response, source_node=int(node))
        if not np.any(basis_response > 0.0):
            projection_fallbacks += 1
            source_position = int(np.flatnonzero(basis == int(node))[0]) if np.any(basis == int(node)) else 0
            basis_response[source_position] = 1.0
        source_positions = np.flatnonzero(basis == int(node))
        if source_positions.size:
            source_position = int(source_positions[0])
            source_echo = float(basis_response[source_position])
            nonself_mass = float(np.sum(basis_response) - source_echo)
            if nonself_mass > 1.0e-12:
                source_echo_masses.append(source_echo)
                basis_response[source_position] = 0.0
                source_echo_suppressed += 1
        projected_basis_masses.append(float(np.sum(basis_response)))
        response_matrix[row, :] = basis_response

    if cap_flow_scale is not None:
        transition, eta_response, flow_meta = _cap_flow_graph_transition(
            points,
            cap,
            basis,
            flow_s=float(cap_flow_scale) * float(cap_flow_time),
            steps=cap_flow_steps,
        )
        assignment_metric_name = "mean_graph_flow_eta"
        assignment_metric = eta_response
    else:
        basis_points = points[basis]
        spatial = np.linalg.norm(basis_points[:, None, :] - basis_points[None, :, :], axis=2)
        spatial = spatial / _positive_median(spatial)
        affinity = _standardize_rows(response_matrix) - 0.15 * spatial
        row_ind, col_ind = linear_sum_assignment(-affinity)
        transition = np.zeros((basis.size, basis.size), dtype=complex)
        transition[row_ind, col_ind] = 1.0
        eta_response = 0.0
        flow_meta = {}
        assignment_metric_name = "mean_assignment_affinity"
        assignment_metric = float(np.mean(affinity[row_ind, col_ind])) if affinity.size else 0.0
    assigned = np.argmax(np.abs(transition), axis=1) if transition.size else np.zeros(0, dtype=np.int64)
    identity_fraction = float(np.mean(assigned == np.arange(basis.size))) if basis.size else 1.0
    return transition.conj().T, float(eta_response), {
        "response_source": response_source,
        "identity_fraction": identity_fraction,
        assignment_metric_name: float(assignment_metric),
        "mean_perturbed_edges_per_source": float(total_perturbed_edges / max(int(basis.size), 1)),
        "mean_repaired_edges_per_source": float(total_repaired_edges / max(int(basis.size), 1)),
        "exact_basis_response_mass_mean": float(np.mean(exact_basis_masses)) if exact_basis_masses else 0.0,
        "projected_basis_response_mass_mean": (
            float(np.mean(projected_basis_masses)) if projected_basis_masses else 0.0
        ),
        "basis_response_projection_fallback_fraction": float(projection_fallbacks / max(int(basis.size), 1)),
        "source_echo_suppression_fraction": float(source_echo_suppressed / max(int(basis.size), 1)),
        "source_echo_mass_mean_before_suppression": (
            float(np.mean(source_echo_masses)) if source_echo_masses else 0.0
        ),
        "basis_response_projection": "local_full_graph_response_projected_to_sparse_cap_basis",
        "probe_steps": int(probe_steps),
        "probe_repairs_per_source": int(probe_repairs_per_source),
        "probe_max_incident_edges": int(probe_max_incident_edges),
        "response_feature_count": 0,
        "gauge_covariant_probe_receipt": True,
        "perturbation_action": "right_multiplication_in_source_endpoint_frame",
        **flow_meta,
    }


def _project_patch_response_to_basis(
    points: np.ndarray,
    basis: np.ndarray,
    response: np.ndarray,
    *,
    source_node: int,
    projection_k: int = 8,
) -> np.ndarray:
    """Compress all-patch perturb/repair response onto a sparse cap basis."""

    basis = np.asarray(basis, dtype=np.int64)
    response = np.asarray(response, dtype=float)
    if basis.size == 0:
        return np.zeros(0, dtype=float)
    projected = np.asarray(response[basis], dtype=float).copy()
    active = np.flatnonzero(response > 0.0)
    if active.size:
        basis_points = np.asarray(points[basis], dtype=float)
        active_points = np.asarray(points[active], dtype=float)
        distances = np.linalg.norm(active_points[:, None, :] - basis_points[None, :, :], axis=2)
        k = min(max(1, int(projection_k)), basis.size)
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        nearest_distances = np.take_along_axis(distances, nearest, axis=1)
        basis_spacing = _positive_median(np.linalg.norm(basis_points[:, None, :] - basis_points[None, :, :], axis=2))
        active_spacing = _positive_median(nearest_distances[:, 0])
        sigma = max(0.35 * basis_spacing, active_spacing, 1.0e-12)
        for active_row, neighbors in enumerate(nearest):
            local_distances = nearest_distances[active_row]
            weights = np.exp(-0.5 * np.square(local_distances / sigma))
            weight_sum = float(np.sum(weights))
            if weight_sum <= 1.0e-15:
                continue
            projected[neighbors] += float(response[active[active_row]]) * weights / weight_sum
    if not np.any(projected > 0.0) and 0 <= int(source_node) < points.shape[0]:
        source_distances = np.linalg.norm(np.asarray(points[basis], dtype=float) - np.asarray(points[int(source_node)], dtype=float), axis=1)
        k = min(max(1, int(projection_k)), basis.size)
        nearest = np.argpartition(source_distances, kth=k - 1)[:k]
        weights = np.exp(-0.5 * np.square(source_distances[nearest] / max(_positive_median(source_distances), 1.0e-12)))
        weight_sum = float(np.sum(weights))
        projected[nearest] = weights / max(weight_sum, 1.0e-15)
    return projected


def _cap_flow_graph_transition(
    points: np.ndarray,
    cap: RoundCap,
    basis: np.ndarray,
    *,
    flow_s: float,
    steps: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    tree = cKDTree(points)
    current_points = points[basis].copy()
    step_count = max(1, int(steps))
    step_s = float(flow_s) / step_count
    for _ in range(step_count):
        current_points = lambda_cap(current_points, cap, step_s)
    projection_distances, projected_indices = tree.query(current_points, k=1)
    target_points = points[projected_indices]
    basis_points = points[basis]
    distances = np.linalg.norm(target_points[:, None, :] - basis_points[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(distances)
    transition = np.zeros((basis.size, basis.size), dtype=complex)
    transition[row_ind, col_ind] = 1.0
    projection_eta = float(np.mean(projection_distances)) if np.size(projection_distances) else 0.0
    eta = float(np.mean(distances[row_ind, col_ind]) + projection_eta) if distances.size else 0.0
    return transition, eta, {
        "cap_flow_s": float(flow_s),
        "cap_flow_steps": int(step_count),
        "cap_flow_step_s": float(step_s),
        "cap_flow_projection_eta": projection_eta,
        "cap_flow_assignment_eta": float(np.mean(distances[row_ind, col_ind])) if distances.size else 0.0,
    }


def _response_features(
    points: np.ndarray,
    cap: RoundCap,
    raw_fields: dict[str, np.ndarray],
    basis: np.ndarray,
) -> np.ndarray:
    cap = cap.normalized()
    local_signed = points[basis] @ cap.axis - math.cos(cap.theta0)
    local_tangent = points[basis] @ cap.tangent
    local_cross = points[basis] @ np.cross(cap.axis, cap.tangent)
    columns = [
        _standardize(local_signed),
        _standardize(local_tangent),
        _standardize(local_cross),
    ]
    for name in (
        "stable_count",
        "committed_mask",
        "repair_load",
        "cumulative_repair_load",
        "local_mismatch_density",
        "s3_class_density",
        "s3_sector_class",
    ):
        if name in raw_fields:
            columns.append(_standardize(np.asarray(raw_fields[name], dtype=float)[basis]))
    if "record_signature" in raw_fields:
        signature = np.asarray(raw_fields["record_signature"], dtype=np.int64)[basis]
        columns.append(_standardize((signature % 257).astype(float) / 257.0))
        columns.append(_standardize((signature % 17).astype(float) / 17.0))
    return np.column_stack(columns) if columns else np.zeros((basis.size, 0), dtype=float)


def _response_score(raw_fields: dict[str, np.ndarray], basis: np.ndarray) -> np.ndarray:
    score = np.zeros(basis.size, dtype=float)
    terms = {
        "local_mismatch_density": 0.35,
        "repair_load": 0.25,
        "cumulative_repair_load": 0.25,
        "s3_class_density": 0.10,
        "stable_count": -0.15,
        "committed_mask": -0.05,
    }
    for name, weight in terms.items():
        if name in raw_fields:
            score += float(weight) * _standardize(np.asarray(raw_fields[name], dtype=float)[basis])
    return _standardize(score)


def _incident_edges(left: np.ndarray, right: np.ndarray, patch_count: int) -> list[list[int]]:
    incident: list[list[int]] = [[] for _ in range(int(patch_count))]
    for edge, (i, j) in enumerate(zip(left.tolist(), right.tolist(), strict=True)):
        incident[int(i)].append(edge)
        incident[int(j)].append(edge)
    return incident


def _node_packet_signature(
    port_left: np.ndarray,
    port_right: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    patch_count: int,
) -> np.ndarray:
    left_packet = port_left.astype(float) + 1.0
    right_packet = port_right.astype(float) + 1.0
    return (
        131.0 * np.bincount(left, weights=left_packet, minlength=patch_count)
        + 17.0 * np.bincount(left, weights=right_packet, minlength=patch_count)
        + 29.0 * np.bincount(right, weights=right_packet, minlength=patch_count)
        + 7.0 * np.bincount(right, weights=left_packet, minlength=patch_count)
    ).astype(np.int64)


def _standardize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    row_sum = np.sum(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_sum, 1e-12)
    centered = normalized - np.mean(normalized, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    return centered / np.maximum(scale, 1e-12)


def _observable_matrix(raw_fields: dict[str, np.ndarray], observable: str, basis: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_fields.get(observable, raw_fields.get("record_signature")), dtype=float)[basis]
    values = _standardize(values)
    diagonal = np.diag(values.astype(complex))
    transition = np.zeros_like(diagonal)
    if values.size > 1:
        diffs = values[:, None] - values[None, :]
        transition = np.exp(-(diffs**2)) - np.eye(values.size)
        transition = transition / max(float(np.linalg.norm(transition)), 1e-15)
    return _hermitian(diagonal + 0.05 * transition)


def _summarize_source(rows: list[dict[str, Any]], scales: list[float]) -> dict[str, Any]:
    by_scale: dict[str, dict[str, Any]] = {}
    for scale in scales:
        label = _scale_label(scale)
        scale_rows = [row for row in rows if math.isclose(float(row["candidate_scale"]), float(scale), rel_tol=1e-12)]
        scores = np.array([float(row["selection_score"]) for row in scale_rows], dtype=float)
        residuals = np.array([float(row["support_visible_residual"]) for row in scale_rows], dtype=float)
        operators = np.array([float(row["operator_residual"]) for row in scale_rows], dtype=float)
        by_scale[label] = {
            "scale": float(scale),
            "count": int(scores.size),
            "selection_score_median": float(np.median(scores)) if scores.size else float("nan"),
            "support_visible_median": float(np.median(residuals)) if residuals.size else float("nan"),
            "operator_median": float(np.median(operators)) if operators.size else float("nan"),
        }
    finite = {
        label: row
        for label, row in by_scale.items()
        if row["count"] and np.isfinite(float(row["selection_score_median"]))
    }
    selected_label = min(finite, key=lambda label: float(finite[label]["selection_score_median"])) if finite else None
    selected_scale = finite[selected_label]["scale"] if selected_label is not None else None
    best_score = finite[selected_label]["selection_score_median"] if selected_label is not None else None
    two_pi = by_scale.get("2pi")
    two_pi_score = two_pi.get("selection_score_median") if two_pi else None
    identity_values = np.array([float(row.get("identity_fraction", 0.0)) for row in rows], dtype=float)
    identity_fraction_median = float(np.median(identity_values)) if identity_values.size else 0.0
    response_degenerate = bool(identity_fraction_median >= 0.9)
    return {
        "selected_label": selected_label,
        "selected_scale": selected_scale,
        "two_pi_selected": bool(selected_label == "2pi"),
        "best_score": best_score,
        "two_pi_score": two_pi_score,
        "two_pi_over_best": (
            float(two_pi_score) / max(float(best_score), 1e-15)
            if two_pi_score is not None and best_score is not None and np.isfinite(float(two_pi_score))
            else None
        ),
        "two_pi_minus_best": (
            float(two_pi_score) - float(best_score)
            if two_pi_score is not None and best_score is not None and np.isfinite(float(two_pi_score))
            else None
        ),
        "identity_fraction_median": identity_fraction_median,
        "response_degenerate": response_degenerate,
        "degeneracy_reason": (
            "response transition is mostly identity; run has not produced a nontrivial endogenous cap response"
            if response_degenerate
            else None
        ),
        "by_scale": by_scale,
    }


def _reference_time(times: list[float]) -> float:
    return float(times[0]) if times else 0.025


def _scale_label(scale: float) -> str:
    if math.isclose(float(scale), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        return "1x"
    if math.isclose(float(scale), math.pi, rel_tol=1e-12, abs_tol=1e-12):
        return "pi"
    if math.isclose(float(scale), 2.0 * math.pi, rel_tol=1e-12, abs_tol=1e-12):
        return "2pi"
    if math.isclose(float(scale), 4.0 * math.pi, rel_tol=1e-12, abs_tol=1e-12):
        return "4pi"
    return f"{float(scale):.8g}"


def _positive_median(values: np.ndarray) -> float:
    positive = np.asarray(values, dtype=float)[np.asarray(values) > 0.0]
    if positive.size == 0:
        return 1.0
    return max(float(np.median(positive)), 1e-12)


def _relative_frobenius(delta: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(delta, ord="fro") / (np.linalg.norm(reference, ord="fro") + 1e-12))


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values))
    if std < 1e-12:
        return values - float(np.mean(values))
    return (values - float(np.mean(values))) / std


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conj().T) / 2.0
