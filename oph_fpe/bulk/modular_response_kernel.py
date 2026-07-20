from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
from typing import Any, Callable, Iterable, Iterator, TypeVar

import numpy as np
from scipy.spatial import cKDTree

from oph_fpe.bulk.cap_geometry import RoundCap, cap_weights, lambda_cap
from oph_fpe.bulk.record_to_h3 import DEFAULT_RECORD_FIELDS
from oph_fpe.claims import CANDIDATE_SCALE_INTERVENTION_INVARIANCE_RECEIPT
from oph_fpe.cache.geometry_cache import GeometryCache
from oph_fpe.gauge.covariant_overlap import (
    GAUGE_COVARIANT_OVERLAP_SCHEMA,
    covariant_discrepancy,
    covariant_mismatch_mask,
    gauge_invariant_edge_residual,
    group_inverse_indices,
    group_multiply_indices,
    repair_covariant_port_pairs,
    repair_production_sector_links,
)


_T = TypeVar("_T")
_R = TypeVar("_R")


def _bounded_ordered_thread_map(
    function: Callable[[_T], _R],
    items: Iterable[_T],
    *,
    max_workers: int,
) -> Iterator[_R]:
    """Map in input order while retaining at most ``max_workers`` results.

    ``ThreadPoolExecutor.map`` eagerly submits the whole iterable.  A full-graph
    perturbation result can own several patch-sized arrays, so an eager queue
    turns a scientifically modest probe grid into an avoidable memory spike.
    This sliding window keeps the shared graph read-only, bounds the number of
    mutable probe states in flight, and yields deterministic ordered assembly.
    """

    workers = max(1, int(max_workers))
    if workers == 1:
        for item in items:
            yield function(item)
        return

    iterator = iter(items)
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="oph-full-graph-probe",
    ) as executor:
        pending: deque[Future[_R]] = deque()
        for _ in range(workers):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            result = pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass
            yield result


def modular_response_kernel(
    points: np.ndarray,
    caps: list[RoundCap],
    raw_fields: dict[str, np.ndarray],
    observer_views: list[dict[str, Any]],
    *,
    times: list[float] | tuple[float, ...],
    field_names: list[str] | tuple[str, ...] = DEFAULT_RECORD_FIELDS,
    cell_entropy: np.ndarray | None = None,
    transport_scale: float = 2.0 * np.pi,
    k_transport: int = 1,
    geometry_cache: GeometryCache | None = None,
    observable_mode: str = "field_transport",
    transition_observables: list[str] | tuple[str, ...] = (
        "checkpoint_class",
        "stable_flag",
        "record_family",
        "s3_sector_class",
        "repair_load_bucket",
    ),
    transition_feature_types: list[str] | tuple[str, ...] = (
        "class_distribution_delta",
        "change_probability_delta",
    ),
    transition_bins: int = 8,
    record_family_modulus: int = 16,
    transform: str = "sigmoid",
    wrong_scales: list[float] | tuple[float, ...] = (1.0,),
    graph_state: dict[str, Any] | None = None,
    perturb_strength: float = 1.0,
    perturb_budget_mode: str = "modular_amount",
    fixed_perturb_fraction: float | None = None,
    perturb_selection_mode: str = "lambda_displacement",
    transition_readout_mode: str = "same_support",
    repair_steps: int = 4,
    repairs_per_step: int = 128,
    max_full_graph_simulations: int | None = None,
    full_graph_budget_policy: str = "skip_if_exceeded",
    full_graph_n_jobs: int = 1,
    freeze_candidate_interventions: bool = True,
    perturb_seed: int = 1,
) -> dict[str, Any]:
    """Build the support-visible modular response tensor for H3 fitting.

    The matrix rows are patch observers. In legacy `field_transport` mode,
    columns are `(cap, time, field)` scalar mean-change slots. In
    `object_transition` mode, columns are finite observer-visible transition
    deltas `(cap, time, packet-class transition feature)`.
    """

    points = np.asarray(points, dtype=float)
    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    weights = np.asarray(cell_entropy, dtype=float) if cell_entropy is not None else np.ones(points.shape[0], dtype=float)
    normalized_mode = str(observable_mode)
    if normalized_mode in {"collar_operator_transition", "cap_collar_operator", "collar_flow_transition"}:
        return _collar_operator_transition_kernel(
            points,
            caps,
            raw_fields,
            patch_views,
            times=times,
            weights=weights,
            transport_scale=float(transport_scale),
            transition_observables=transition_observables,
            transition_feature_types=transition_feature_types,
            transition_bins=int(transition_bins),
            record_family_modulus=int(record_family_modulus),
            transform=str(transform),
            wrong_scales=wrong_scales,
            k_transport=int(k_transport),
            geometry_cache=geometry_cache,
        )
    if normalized_mode in {"perturb_resettle_transition", "transition_response"}:
        return _perturb_resettle_transition_kernel(
            points,
            caps,
            raw_fields,
            patch_views,
            times=times,
            weights=weights,
            graph_state=graph_state or {},
            transport_scale=float(transport_scale),
            transition_observables=transition_observables,
            transition_feature_types=transition_feature_types,
            transition_bins=int(transition_bins),
            record_family_modulus=int(record_family_modulus),
            transform=str(transform),
            wrong_scales=wrong_scales,
            perturb_strength=float(perturb_strength),
            perturb_budget_mode=str(perturb_budget_mode),
            fixed_perturb_fraction=fixed_perturb_fraction,
            perturb_selection_mode=str(perturb_selection_mode),
            transition_readout_mode=str(transition_readout_mode),
            repair_steps=int(repair_steps),
            repairs_per_step=int(repairs_per_step),
            max_full_graph_simulations=max_full_graph_simulations,
            full_graph_budget_policy=str(full_graph_budget_policy),
            full_graph_n_jobs=int(full_graph_n_jobs),
            freeze_candidate_interventions=bool(freeze_candidate_interventions),
            seed=int(perturb_seed),
        )
    if normalized_mode == "object_transition":
        return _object_transition_kernel(
            points,
            caps,
            raw_fields,
            patch_views,
            times=times,
            weights=weights,
            transport_scale=float(transport_scale),
            k_transport=int(k_transport),
            geometry_cache=geometry_cache,
            transition_observables=transition_observables,
            transition_feature_types=transition_feature_types,
            transition_bins=int(transition_bins),
            record_family_modulus=int(record_family_modulus),
            transform=str(transform),
            wrong_scales=wrong_scales,
        )
    fields = _standardized_fields(raw_fields, field_names)
    if not patch_views or not caps or not fields or not times:
        return _empty_kernel(points, patch_views, caps, fields, times)
    cache = geometry_cache or GeometryCache(points)
    supports = [_valid_support(view.get("support_nodes", []), points.shape[0]) for view in patch_views]
    source_features = np.vstack([_support_feature(support, fields, weights) for support in supports])
    field_scales = _field_scales(source_features)
    feature_rows: list[dict[str, Any]] = []
    columns: list[np.ndarray] = []
    no_flow_columns: list[np.ndarray] = []
    s2_columns: list[np.ndarray] = []
    for cap_index, cap in enumerate(caps):
        s2_profile = _observer_s2_profile(points, cap, supports, weights)
        for time_index, time_value in enumerate(times):
            support_maps = [
                _transported_support(cache, support, cap, float(transport_scale) * float(time_value), k=int(k_transport), cap_id=cap_index)
                for support in supports
            ]
            transported_features = np.vstack([_support_feature(support, fields, weights) for support in support_maps])
            delta = transported_features - source_features
            for field_index, field_name in enumerate(fields):
                scale = max(float(field_scales[field_index]), 1e-9)
                response = 1.0 / (1.0 + np.exp(-np.clip(delta[:, field_index] / scale, -60.0, 60.0)))
                columns.append(response)
                no_flow_columns.append(np.full(response.shape[0], 0.5, dtype=float))
                s2_columns.append(s2_profile)
                feature_rows.append(
                    {
                        "feature_index": len(feature_rows),
                        "cap_index": int(cap_index),
                        "time_index": int(time_index),
                        "time": float(time_value),
                        "field": str(field_name),
                    }
                )
    matrix = np.vstack(columns).T if columns else np.zeros((len(patch_views), 0), dtype=float)
    no_flow = np.vstack(no_flow_columns).T if no_flow_columns else np.zeros_like(matrix)
    s2_boundary = np.vstack(s2_columns).T if s2_columns else np.zeros_like(matrix)
    rng = np.random.default_rng(9449)
    shuffled = matrix.copy()
    if shuffled.size:
        for row in shuffled:
            rng.shuffle(row)
    return {
        "mode": "observer_modular_response_kernel",
        "observable_mode": "field_transport",
        "matrix": matrix,
        "s2_boundary_control": s2_boundary,
        "shuffled_control": shuffled,
        "no_modular_flow_control": no_flow,
        "feature_rows": feature_rows,
        "observer_ids": [int(view.get("observer_id", index)) for index, view in enumerate(patch_views)],
        "cap_count": len(caps),
        "time_count": len(times),
        "field_names": list(fields.keys()),
        "response_summary": _response_summary(matrix),
        "geometry_cache": cache.report(),
        "claim_boundary": (
            "legacy support-visible scalar-mean response tensor R[i,C,t,O]. "
            "This is a kinematic diagnostic, not the theorem-aligned object-transition receipt."
        ),
    }


def kernel_json_summary(kernel: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": kernel.get("mode"),
        "observable_mode": kernel.get("observable_mode"),
        "observer_count": int(np.asarray(kernel.get("matrix", np.zeros((0, 0)))).shape[0]),
        "feature_count": int(np.asarray(kernel.get("matrix", np.zeros((0, 0)))).shape[1]),
        "cap_count": int(kernel.get("cap_count", 0)),
        "time_count": int(kernel.get("time_count", 0)),
        "field_names": list(kernel.get("field_names", [])),
        "feature_rows_sample": list(kernel.get("feature_rows", []))[:128],
        "response_summary": kernel.get("response_summary", {}),
        "raw_response_summary": kernel.get("raw_response_summary", {}),
        "transform_report": kernel.get("transform_report", {}),
        "controls": kernel.get("controls", {}),
        "event_row_ids": list(kernel.get("event_row_ids", [])),
        "geometry_control_event_row_ids": dict(
            kernel.get("geometry_control_event_row_ids", {})
        ),
        "perturb_resettle_report": kernel.get("perturb_resettle_report", {}),
        "geometry_cache": kernel.get("geometry_cache", {}),
        "claim_boundary": kernel.get("claim_boundary"),
    }


def _empty_kernel(
    points: np.ndarray,
    patch_views: list[dict[str, Any]],
    caps: list[RoundCap],
    fields: dict[str, np.ndarray],
    times: list[float] | tuple[float, ...],
) -> dict[str, Any]:
    return {
        "mode": "observer_modular_response_kernel",
        "observable_mode": "empty",
        "matrix": np.zeros((len(patch_views), 0), dtype=float),
        "s2_boundary_control": np.zeros((len(patch_views), 0), dtype=float),
        "shuffled_control": np.zeros((len(patch_views), 0), dtype=float),
        "no_modular_flow_control": np.zeros((len(patch_views), 0), dtype=float),
        "feature_rows": [],
        "observer_ids": [int(view.get("observer_id", index)) for index, view in enumerate(patch_views)],
        "cap_count": len(caps),
        "time_count": len(times),
        "field_names": list(fields.keys()),
        "response_summary": _response_summary(np.zeros((len(patch_views), 0), dtype=float)),
        "geometry_cache": {"point_count": int(points.shape[0]), "transport_map_count": 0},
        "claim_boundary": "empty modular response kernel; no bulk claim",
    }


def _object_transition_kernel(
    points: np.ndarray,
    caps: list[RoundCap],
    raw_fields: dict[str, np.ndarray],
    patch_views: list[dict[str, Any]],
    *,
    times: list[float] | tuple[float, ...],
    weights: np.ndarray,
    transport_scale: float,
    k_transport: int,
    geometry_cache: GeometryCache | None,
    transition_observables: list[str] | tuple[str, ...],
    transition_feature_types: list[str] | tuple[str, ...],
    transition_bins: int,
    record_family_modulus: int,
    transform: str,
    wrong_scales: list[float] | tuple[float, ...],
) -> dict[str, Any]:
    packet_fields = _object_packet_fields(
        raw_fields,
        transition_observables,
        bins=max(2, int(transition_bins)),
        record_family_modulus=max(2, int(record_family_modulus)),
    )
    if not patch_views or not caps or not packet_fields or not times:
        return _empty_kernel(points, patch_views, caps, packet_fields, times)
    cache = geometry_cache or GeometryCache(points)
    supports = [_valid_support(view.get("support_nodes", []), points.shape[0]) for view in patch_views]
    feature_types = _normalized_transition_feature_types(transition_feature_types)
    feature_rows = _object_transition_feature_rows(caps, times, packet_fields, feature_types)
    raw_matrix = _object_transition_matrix(
        cache,
        supports,
        caps,
        times,
        packet_fields,
        weights,
        feature_types,
        scale=float(transport_scale),
        k_transport=int(k_transport),
    )
    no_perturb = np.zeros_like(raw_matrix)
    s2_boundary = _object_s2_boundary_matrix(points, caps, supports, weights, feature_rows)
    wrong_scale_controls: dict[str, np.ndarray] = {}
    for wrong_scale in wrong_scales:
        label = _scale_label(float(wrong_scale))
        wrong_scale_controls[label] = _object_transition_matrix(
            cache,
            supports,
            caps,
            times,
            packet_fields,
            weights,
            feature_types,
            scale=float(wrong_scale),
            k_transport=int(k_transport),
        )
    matrix, transformed_controls, transform_report = _transform_with_controls(
        raw_matrix,
        {
            "no_modular_flow_control": no_perturb,
            "s2_boundary_control": s2_boundary,
            **{f"wrong_scale_{label}": value for label, value in wrong_scale_controls.items()},
        },
        transform=str(transform),
        feature_rows=feature_rows,
    )
    rng = np.random.default_rng(9449)
    shuffled_response = matrix.copy()
    if shuffled_response.size:
        for row in shuffled_response:
            rng.shuffle(row)
    shuffled_observers = matrix.copy()
    if shuffled_observers.shape[0] > 1:
        shuffled_observers = shuffled_observers[rng.permutation(shuffled_observers.shape[0])]
    controls = {
        "s2_boundary_control": transformed_controls.get("s2_boundary_control", np.zeros_like(matrix)),
        "no_modular_flow_control": transformed_controls.get("no_modular_flow_control", np.zeros_like(matrix)),
        "shuffled_response_control": shuffled_response,
        "shuffled_observer_labels_control": shuffled_observers,
        "wrong_scale_controls": {
            label: transformed_controls.get(f"wrong_scale_{label}", np.zeros_like(matrix))
            for label in wrong_scale_controls
        },
    }
    return {
        "mode": "observer_modular_response_kernel",
        "observable_mode": "object_transition",
        "matrix": matrix,
        "s2_boundary_control": controls["s2_boundary_control"],
        "shuffled_control": controls["shuffled_response_control"],
        "shuffled_response_control": controls["shuffled_response_control"],
        "shuffled_observer_labels_control": controls["shuffled_observer_labels_control"],
        "no_modular_flow_control": controls["no_modular_flow_control"],
        "wrong_scale_controls": controls["wrong_scale_controls"],
        "feature_rows": feature_rows,
        "observer_ids": [int(view.get("observer_id", index)) for index, view in enumerate(patch_views)],
        "observer_axes": [[float(value) for value in np.asarray(view.get("axis"), dtype=float)] for view in patch_views],
        "cap_count": len(caps),
        "time_count": len(times),
        "field_names": list(packet_fields.keys()),
        "response_summary": _response_summary(matrix),
        "raw_response_summary": _response_summary(raw_matrix),
        "transform_report": transform_report,
        "controls": {
            "s2_boundary": _response_summary(controls["s2_boundary_control"]),
            "no_modular_flow": _response_summary(controls["no_modular_flow_control"]),
            "shuffled_response": _response_summary(controls["shuffled_response_control"]),
            "shuffled_observer_labels": _response_summary(controls["shuffled_observer_labels_control"]),
            "wrong_scale_labels": list(controls["wrong_scale_controls"].keys()),
        },
        "geometry_cache": cache.report(),
        "claim_boundary": (
            "support-visible observer object-transition tensor R[i,C,t,alpha->beta]. "
            "Rows are observers, columns are cap/time/readout transition deltas. This is closer "
            "to the record/collar/object transition observable requested by the paper audit, but "
            "it is still a finite surrogate and not a bulk claim until H3 beats controls."
        ),
    }


def _collar_operator_transition_kernel(
    points: np.ndarray,
    caps: list[RoundCap],
    raw_fields: dict[str, np.ndarray],
    patch_views: list[dict[str, Any]],
    *,
    times: list[float] | tuple[float, ...],
    weights: np.ndarray,
    transport_scale: float,
    transition_observables: list[str] | tuple[str, ...],
    transition_feature_types: list[str] | tuple[str, ...],
    transition_bins: int,
    record_family_modulus: int,
    transform: str,
    wrong_scales: list[float] | tuple[float, ...],
    k_transport: int,
    geometry_cache: GeometryCache | None,
) -> dict[str, Any]:
    """Support-visible collar operator transition tensor.

    This mode keeps the modular-flow direction in the readout instead of
    perturbing the collar and then letting local repair erase most of the
    transient signal. It is still finite-regulator data: rows are observers,
    columns are cap/time/packet transition features, and the target basis is
    the visible packet algebra transported by lambda_C(s).
    """

    packet_fields = _object_packet_fields(
        raw_fields,
        transition_observables,
        bins=max(2, int(transition_bins)),
        record_family_modulus=max(2, int(record_family_modulus)),
    )
    if not patch_views or not caps or not packet_fields or not times:
        return _empty_kernel(points, patch_views, caps, packet_fields, times)
    cache = geometry_cache or GeometryCache(points)
    supports = [_valid_support(view.get("support_nodes", []), points.shape[0]) for view in patch_views]
    feature_types = _normalized_transition_feature_types(transition_feature_types)
    feature_rows = _object_transition_feature_rows(caps, times, packet_fields, feature_types)
    raw_matrix = _collar_operator_transition_matrix(
        cache,
        supports,
        caps,
        times,
        packet_fields,
        weights,
        feature_types,
        scale=float(transport_scale),
        k_transport=int(k_transport),
    )
    no_flow = np.zeros_like(raw_matrix)
    s2_boundary = _object_s2_boundary_matrix(points, caps, supports, weights, feature_rows)
    wrong_scale_controls: dict[str, np.ndarray] = {}
    for wrong_scale in wrong_scales:
        label = _scale_label(float(wrong_scale))
        wrong_scale_controls[label] = _collar_operator_transition_matrix(
            cache,
            supports,
            caps,
            times,
            packet_fields,
            weights,
            feature_types,
            scale=float(wrong_scale),
            k_transport=int(k_transport),
        )
    matrix, transformed_controls, transform_report = _transform_with_controls(
        raw_matrix,
        {
            "no_modular_flow_control": no_flow,
            "s2_boundary_control": s2_boundary,
            **{f"wrong_scale_{label}": value for label, value in wrong_scale_controls.items()},
        },
        transform=str(transform),
        feature_rows=feature_rows,
    )
    rng = np.random.default_rng(9449)
    shuffled_response = matrix.copy()
    if shuffled_response.size:
        for row in shuffled_response:
            rng.shuffle(row)
    shuffled_observers = matrix.copy()
    if shuffled_observers.shape[0] > 1:
        shuffled_observers = shuffled_observers[rng.permutation(shuffled_observers.shape[0])]
    wrong_controls = {
        label: transformed_controls.get(f"wrong_scale_{label}", np.zeros_like(matrix))
        for label in wrong_scale_controls
    }
    return {
        "mode": "observer_modular_response_kernel",
        "observable_mode": "collar_operator_transition",
        "matrix": matrix,
        "s2_boundary_control": transformed_controls.get("s2_boundary_control", np.zeros_like(matrix)),
        "shuffled_control": shuffled_response,
        "shuffled_response_control": shuffled_response,
        "shuffled_observer_labels_control": shuffled_observers,
        "no_modular_flow_control": transformed_controls.get("no_modular_flow_control", np.zeros_like(matrix)),
        "wrong_scale_controls": wrong_controls,
        "feature_rows": feature_rows,
        "observer_ids": [int(view.get("observer_id", index)) for index, view in enumerate(patch_views)],
        "observer_axes": [[float(value) for value in np.asarray(view.get("axis"), dtype=float)] for view in patch_views],
        "cap_count": len(caps),
        "time_count": len(times),
        "field_names": list(packet_fields.keys()),
        "response_summary": _response_summary(matrix),
        "raw_response_summary": _response_summary(raw_matrix),
        "transform_report": transform_report,
        "controls": {
            "s2_boundary": _response_summary(transformed_controls.get("s2_boundary_control", np.zeros_like(matrix))),
            "no_modular_flow": _response_summary(transformed_controls.get("no_modular_flow_control", np.zeros_like(matrix))),
            "shuffled_response": _response_summary(shuffled_response),
            "shuffled_observer_labels": _response_summary(shuffled_observers),
            "wrong_scale_labels": list(wrong_controls.keys()),
        },
        "collar_operator_report": {
            "mode": "lambda_cap_visible_packet_operator",
            "transport_scale": float(transport_scale),
            "readout": "source_support_to_lambda_transported_support_with_collar_flow_weights",
            "claim_boundary": (
                "support-visible finite collar operator readout. It preserves the cap-flow direction "
                "in visible packet transition matrix elements, but it is still a surrogate for the "
                "paper's modular operator until state-derived rho_C/K_a receipts and refinement pass."
            ),
        },
        "geometry_cache": cache.report(),
        "claim_boundary": (
            "support-visible collar-operator transition tensor from lambda_C(s) acting on finite visible "
            "packet algebras. This is a theorem-aligned diagnostic lane, not a physical 3D bulk claim "
            "until H3, wrong-scale, object, and refinement controls pass."
        ),
    }


def _perturb_resettle_transition_kernel(
    points: np.ndarray,
    caps: list[RoundCap],
    raw_fields: dict[str, np.ndarray],
    patch_views: list[dict[str, Any]],
    *,
    times: list[float] | tuple[float, ...],
    weights: np.ndarray,
    graph_state: dict[str, Any],
    transport_scale: float,
    transition_observables: list[str] | tuple[str, ...],
    transition_feature_types: list[str] | tuple[str, ...],
    transition_bins: int,
    record_family_modulus: int,
    transform: str,
    wrong_scales: list[float] | tuple[float, ...],
    perturb_strength: float,
    perturb_budget_mode: str,
    fixed_perturb_fraction: float | None,
    perturb_selection_mode: str,
    transition_readout_mode: str,
    repair_steps: int,
    repairs_per_step: int,
    max_full_graph_simulations: int | None,
    full_graph_budget_policy: str,
    full_graph_n_jobs: int,
    freeze_candidate_interventions: bool,
    seed: int,
) -> dict[str, Any]:
    packet_fields = _object_packet_fields(
        raw_fields,
        transition_observables,
        bins=max(2, int(transition_bins)),
        record_family_modulus=max(2, int(record_family_modulus)),
    )
    graph = _validated_graph_state(graph_state, points.shape[0])
    if not patch_views or not caps or not packet_fields or not times or graph is None:
        empty = _empty_kernel(points, patch_views, caps, packet_fields, times)
        if graph is None:
            empty["proof_blockers"] = ["gauge_coupled_perturb_resettle_graph_state_unavailable"]
        return empty
    sector_replay = _production_sector_replay_contract(graph)
    if sector_replay["blockers"]:
        empty = _empty_kernel(points, patch_views, caps, packet_fields, times)
        empty["proof_blockers"] = list(sector_replay["blockers"])
        empty["perturb_resettle_report"] = {
            "gauge_covariant_probe_receipt": False,
            "production_sector_repair_enabled": bool(
                sector_replay["production_sector_repair_enabled"]
            ),
            "sector_repair_replayed": False,
            "production_move_contract": sector_replay,
        }
        return empty
    scale_values = [float(value) for value in wrong_scales]
    unique_simulation_scales: list[float] = [float(transport_scale)]
    for value in scale_values:
        if not any(_same_probe_scale(value, existing) for existing in unique_simulation_scales):
            unique_simulation_scales.append(value)
    requested_full_graph_simulations = len(caps) * len(times) * (1 + len(scale_values))
    planned_full_graph_simulations = len(caps) * len(times) * len(unique_simulation_scales)
    requested_n_jobs = max(1, int(full_graph_n_jobs))
    tasks_per_matrix = len(caps) * len(times)
    effective_n_jobs = min(requested_n_jobs, max(1, tasks_per_matrix))
    parallel_execution = {
        "schema": "bounded_ordered_full_graph_thread_execution_v1",
        "requested_n_jobs": int(requested_n_jobs),
        "effective_n_jobs": int(effective_n_jobs),
        "executor": (
            "bounded_ordered_thread_pool" if effective_n_jobs > 1 else "sequential"
        ),
        "matrix_batch_count": int(len(unique_simulation_scales)),
        "probe_task_count": int(planned_full_graph_simulations),
        "max_in_flight_full_graph_states": int(effective_n_jobs),
        "ordered_result_assembly": True,
        "independent_named_rng_streams": True,
        "shared_graph_state_read_only": True,
    }
    budget = (
        None
        if max_full_graph_simulations is None
        else max(0, int(max_full_graph_simulations))
    )
    policy = str(full_graph_budget_policy)
    if policy != "skip_if_exceeded":
        raise ValueError(
            "modular response full_graph_budget_policy must be 'skip_if_exceeded'"
        )
    if budget is not None and planned_full_graph_simulations > budget:
        empty = _empty_kernel(points, patch_views, caps, packet_fields, times)
        empty["proof_blockers"] = ["modular_response_full_graph_simulation_budget_exceeded"]
        empty["perturb_resettle_report"] = {
            "gauge_covariant_probe_receipt": False,
            "production_sector_repair_enabled": bool(
                sector_replay["production_sector_repair_enabled"]
            ),
            "sector_repair_replayed": False,
            "production_move_contract": sector_replay,
            "full_graph_simulation_budget": {
                "schema": "modular_response_full_graph_simulation_budget_v1",
                "policy": policy,
                "max_full_graph_simulations": budget,
                "requested_without_scale_reuse": int(requested_full_graph_simulations),
                "planned_with_scale_reuse": int(planned_full_graph_simulations),
                "executed_full_graph_simulations": 0,
                "receipt": False,
            },
            "parallel_execution": {
                **parallel_execution,
                "effective_n_jobs": 0,
                "max_in_flight_full_graph_states": 0,
            },
        }
        return empty
    supports = [_valid_support(view.get("support_nodes", []), points.shape[0]) for view in patch_views]
    feature_types = _normalized_transition_feature_types(transition_feature_types)
    feature_rows = _object_transition_feature_rows(caps, times, packet_fields, feature_types)
    source_intervention_scale = 1.0 if freeze_candidate_interventions else float(transport_scale)
    raw_matrix, primary_intervention_rows = _perturb_resettle_matrix(
        points,
        caps,
        raw_fields,
        packet_fields,
        supports,
        weights,
        graph,
        feature_types,
        times=times,
        scale=float(transport_scale),
        intervention_scale=source_intervention_scale,
        transition_observables=transition_observables,
        transition_bins=int(transition_bins),
        record_family_modulus=int(record_family_modulus),
        perturb_strength=float(perturb_strength),
        perturb_budget_mode=str(perturb_budget_mode),
        fixed_perturb_fraction=fixed_perturb_fraction,
        perturb_selection_mode=str(perturb_selection_mode),
        transition_readout_mode=str(transition_readout_mode),
        repair_steps=int(repair_steps),
        repairs_per_step=int(repairs_per_step),
        n_jobs=effective_n_jobs,
        seed=int(seed),
    )
    no_perturb = np.zeros_like(raw_matrix)
    s2_boundary = _object_s2_boundary_matrix(points, caps, supports, weights, feature_rows)
    wrong_scale_controls: dict[str, np.ndarray] = {}
    scale_matrix_cache: list[tuple[float, np.ndarray, list[dict[str, Any]]]] = [
        (float(transport_scale), raw_matrix, primary_intervention_rows)
    ]
    wrong_scale_intervention_audit: dict[str, Any] = {}
    for wrong_scale in scale_values:
        label = _scale_label(float(wrong_scale))
        cached_matrix = next(
            (
                (matrix_value, audit_rows)
                for cached_scale, matrix_value, audit_rows in scale_matrix_cache
                if _same_probe_scale(float(wrong_scale), cached_scale)
            ),
            None,
        )
        if cached_matrix is None:
            cached_matrix, cached_audit_rows = _perturb_resettle_matrix(
                points,
                caps,
                raw_fields,
                packet_fields,
                supports,
                weights,
                graph,
                feature_types,
                times=times,
                scale=float(wrong_scale),
                intervention_scale=(
                    source_intervention_scale
                    if freeze_candidate_interventions
                    else float(wrong_scale)
                ),
                transition_observables=transition_observables,
                transition_bins=int(transition_bins),
                record_family_modulus=int(record_family_modulus),
                perturb_strength=float(perturb_strength),
                perturb_budget_mode=str(perturb_budget_mode),
                fixed_perturb_fraction=fixed_perturb_fraction,
                perturb_selection_mode=str(perturb_selection_mode),
                transition_readout_mode=str(transition_readout_mode),
                repair_steps=int(repair_steps),
                repairs_per_step=int(repairs_per_step),
                n_jobs=effective_n_jobs,
                seed=int(seed),
            )
            scale_matrix_cache.append((float(wrong_scale), cached_matrix, cached_audit_rows))
        else:
            cached_matrix, cached_audit_rows = cached_matrix
        audit_matches_primary = bool(cached_audit_rows == primary_intervention_rows)
        wrong_scale_intervention_audit[label] = {
            "event_count": len(cached_audit_rows),
            "intervention_rows_match_primary": audit_matches_primary,
        }
        wrong_scale_controls[label] = cached_matrix
    candidate_scale_intervention_invariance_receipt = bool(
        freeze_candidate_interventions
        and primary_intervention_rows
        and all(
            row.get("intervention_rows_match_primary", False)
            for row in wrong_scale_intervention_audit.values()
        )
    )
    target_free_selection_modes = {
        "source_random",
        "uniform_random",
        "collar_random",
    }
    target_free_source_intervention_receipt = bool(
        str(perturb_budget_mode).lower()
        in {"fixed", "fixed_collar_fraction", "fixed_fraction"}
        and str(perturb_selection_mode).lower().replace("-", "_")
        in target_free_selection_modes
    )
    event_row_ids = _response_event_row_ids(feature_rows)
    matrix, transformed_controls, transform_report = _transform_with_controls(
        raw_matrix,
        {
            "no_modular_flow_control": no_perturb,
            "s2_boundary_control": s2_boundary,
            **{f"wrong_scale_{label}": value for label, value in wrong_scale_controls.items()},
        },
        transform=str(transform),
        feature_rows=feature_rows,
    )
    rng = np.random.default_rng(seed + 9449)
    shuffled_response = matrix.copy()
    if shuffled_response.size:
        for row in shuffled_response:
            rng.shuffle(row)
    shuffled_observers = matrix.copy()
    if shuffled_observers.shape[0] > 1:
        shuffled_observers = shuffled_observers[rng.permutation(shuffled_observers.shape[0])]
    wrong_controls = {
        label: transformed_controls.get(f"wrong_scale_{label}", np.zeros_like(matrix))
        for label in wrong_scale_controls
    }
    return {
        "mode": "observer_modular_response_kernel",
        "observable_mode": "perturb_resettle_transition",
        "matrix": matrix,
        "s2_boundary_control": transformed_controls.get("s2_boundary_control", np.zeros_like(matrix)),
        "shuffled_control": shuffled_response,
        "shuffled_response_control": shuffled_response,
        "shuffled_observer_labels_control": shuffled_observers,
        "no_modular_flow_control": transformed_controls.get("no_modular_flow_control", np.zeros_like(matrix)),
        "wrong_scale_controls": wrong_controls,
        "event_row_ids": event_row_ids,
        "geometry_control_event_row_ids": {"S2": event_row_ids},
        "wrong_scale_event_row_ids": {
            label: event_row_ids for label in wrong_controls
        },
        "feature_rows": feature_rows,
        "observer_ids": [int(view.get("observer_id", index)) for index, view in enumerate(patch_views)],
        "observer_axes": [[float(value) for value in np.asarray(view.get("axis"), dtype=float)] for view in patch_views],
        "cap_count": len(caps),
        "time_count": len(times),
        "field_names": list(packet_fields.keys()),
        "response_summary": _response_summary(matrix),
        "raw_response_summary": _response_summary(raw_matrix),
        "transform_report": transform_report,
        "controls": {
            "s2_boundary": _response_summary(transformed_controls.get("s2_boundary_control", np.zeros_like(matrix))),
            "no_modular_flow": _response_summary(transformed_controls.get("no_modular_flow_control", np.zeros_like(matrix))),
            "shuffled_response": _response_summary(shuffled_response),
            "shuffled_observer_labels": _response_summary(shuffled_observers),
            "wrong_scale_labels": list(wrong_controls.keys()),
        },
        "perturb_resettle_report": {
            "mode": "cap_collar_perturb_then_local_repair",
            "gauge_covariant_probe_receipt": bool(
                sector_replay["gauge_covariant_probe_receipt"]
            ),
            "centered_or_single_intervention_support_receipt": True,
            "production_sector_repair_enabled": bool(
                sector_replay["production_sector_repair_enabled"]
            ),
            "sector_repair_replayed": bool(sector_replay["sector_repair_replayed"]),
            "production_move_contract": sector_replay,
            "full_graph_simulation_budget": {
                "schema": "modular_response_full_graph_simulation_budget_v1",
                "policy": policy,
                "max_full_graph_simulations": budget,
                "requested_without_scale_reuse": int(requested_full_graph_simulations),
                "planned_with_scale_reuse": int(planned_full_graph_simulations),
                "executed_full_graph_simulations": int(planned_full_graph_simulations),
                "scale_reuse_count": int(
                    requested_full_graph_simulations - planned_full_graph_simulations
                ),
                "receipt": True,
            },
            "parallel_execution": parallel_execution,
            "repair_steps": int(repair_steps),
            "repairs_per_step": int(repairs_per_step),
            "perturb_strength": float(perturb_strength),
            "perturb_budget_mode": str(perturb_budget_mode),
            "fixed_perturb_fraction": float(fixed_perturb_fraction) if fixed_perturb_fraction is not None else None,
            "perturb_selection_mode": str(perturb_selection_mode),
            "transition_readout_mode": str(transition_readout_mode),
            "freeze_candidate_interventions": bool(freeze_candidate_interventions),
            "source_intervention_scale": float(source_intervention_scale),
            CANDIDATE_SCALE_INTERVENTION_INVARIANCE_RECEIPT: (
                candidate_scale_intervention_invariance_receipt
            ),
            "candidate_scale_intervention_invariance_receipt": (
                candidate_scale_intervention_invariance_receipt
            ),
            "TARGET_FREE_SOURCE_INTERVENTION_RECEIPT": (
                target_free_source_intervention_receipt
            ),
            "primary_intervention_rows": primary_intervention_rows,
            "wrong_scale_intervention_audit": wrong_scale_intervention_audit,
            "candidate_scale_effect_scope": (
                ["transported_support_readout"]
                if str(transition_readout_mode)
                in {"transported_support", "transported_support_delta", "cap_transported_support"}
                else []
            ),
            "graph_edge_count": int(graph["left"].size),
            "claim_boundary": (
                "Finite cap/collar perturb-resettle surrogate. In the default frozen mode, all "
                "normalization candidates reuse byte-identical intervention-support hashes and RNG "
                "seeds; candidate scale may affect only an explicitly labelled transported-support "
                "readout. Legacy candidate-dependent interventions require an explicit opt-out and "
                "cannot pass the physical campaign preflight."
            ),
        },
        "geometry_cache": {"mode": "not_used_by_perturb_resettle", "point_count": int(points.shape[0])},
        "claim_boundary": (
            "support-visible observer transition tensor from cap/collar perturbation followed by local repair. "
            "This is the current finite surrogate for state-derived record/collar response; it is not a bulk "
            "claim until joint H3 beats S2, shuffled, no-perturbation, and wrong-scale controls."
        ),
    }


def _perturb_resettle_matrix(
    points: np.ndarray,
    caps: list[RoundCap],
    raw_fields: dict[str, np.ndarray],
    base_packet_fields: dict[str, np.ndarray],
    supports: list[np.ndarray],
    weights: np.ndarray,
    graph: dict[str, np.ndarray | int],
    feature_types: tuple[str, ...],
    *,
    times: list[float] | tuple[float, ...],
    scale: float,
    intervention_scale: float | None = None,
    transition_observables: list[str] | tuple[str, ...],
    transition_bins: int,
    record_family_modulus: int,
    perturb_strength: float,
    perturb_budget_mode: str,
    fixed_perturb_fraction: float | None,
    perturb_selection_mode: str,
    transition_readout_mode: str,
    repair_steps: int,
    repairs_per_step: int,
    n_jobs: int = 1,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    points = np.asarray(points, dtype=float)
    readout_mode = str(transition_readout_mode)
    tree: cKDTree | None = None
    if readout_mode in {"transported_support", "transported_support_delta", "cap_transported_support"}:
        tree = cKDTree(points)
    jobs = [
        (cap_index, time_index, cap, float(time_value))
        for cap_index, cap in enumerate(caps)
        for time_index, time_value in enumerate(times)
    ]

    source_scale = float(scale) if intervention_scale is None else float(intervention_scale)

    def probe_columns(
        job: tuple[int, int, RoundCap, float],
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        cap_index, time_index, cap, time_value = job
        transported_supports = (
            [
                _transport_support_direct(
                    points,
                    tree,
                    support,
                    cap,
                    float(scale) * float(time_value),
                )
                for support in supports
            ]
            if tree is not None
            else supports
        )
        post_raw = _simulate_cap_collar_perturb_resettle(
                points,
                cap,
                raw_fields,
                graph,
                scale=source_scale,
                time_value=time_value,
                perturb_strength=float(perturb_strength),
                perturb_budget_mode=str(perturb_budget_mode),
                fixed_perturb_fraction=fixed_perturb_fraction,
                perturb_selection_mode=str(perturb_selection_mode),
                repair_steps=int(repair_steps),
                repairs_per_step=int(repairs_per_step),
                seed=int(seed) + 1009 * int(cap_index) + 9173 * int(time_index),
            )
        post_packets = _object_packet_fields(
            post_raw,
            transition_observables,
            bins=max(2, int(transition_bins)),
            record_family_modulus=max(2, int(record_family_modulus)),
        )
        local_columns: list[np.ndarray] = []
        for observable, source_packets in base_packet_fields.items():
            target_packets = np.asarray(
                post_packets.get(observable, source_packets), dtype=np.int64
            )
            class_count = int(np.max(source_packets)) + 1 if source_packets.size else 1
            feature_values: dict[str, np.ndarray] = {
                feature_type: _empty_transition_feature_array(
                    len(supports), class_count, feature_type
                )
                for feature_type in feature_types
            }
            for row_index, (support, transported) in enumerate(
                zip(supports, transported_supports, strict=False)
            ):
                if tree is not None:
                    values = _transition_features_between_supports(
                        support,
                        transported,
                        np.asarray(source_packets, dtype=np.int64),
                        target_packets,
                        weights,
                        class_count,
                        feature_types=feature_types,
                    )
                else:
                    values = _transition_features_same_support(
                        support,
                        np.asarray(source_packets, dtype=np.int64),
                        target_packets,
                        weights,
                        class_count,
                        feature_types=feature_types,
                    )
                for feature_type in feature_types:
                    feature_values[feature_type][row_index] = values[feature_type]
            local_columns.extend(
                _transition_feature_columns(feature_values, feature_types, class_count)
            )
        return local_columns, {
            "cap_index": int(cap_index),
            "time_index": int(time_index),
            "time": float(time_value),
            "source_intervention_scale": source_scale,
            "intervention_support_hash": post_raw.get("_intervention_support_hash"),
            "probe_rng_stream_schema": post_raw.get("_probe_rng_stream_schema"),
        }

    columns: list[np.ndarray] = []
    intervention_rows: list[dict[str, Any]] = []
    effective_n_jobs = min(max(1, int(n_jobs)), max(1, len(jobs)))
    for local_columns, intervention_row in _bounded_ordered_thread_map(
        probe_columns,
        jobs,
        max_workers=effective_n_jobs,
    ):
        columns.extend(local_columns)
        intervention_rows.append(intervention_row)
    matrix = np.vstack(columns).T if columns else np.zeros((len(supports), 0), dtype=float)
    return matrix, intervention_rows


def _simulate_cap_collar_perturb_resettle(
    points: np.ndarray,
    cap: RoundCap,
    raw_fields: dict[str, np.ndarray],
    graph: dict[str, Any],
    *,
    scale: float,
    time_value: float,
    perturb_strength: float,
    perturb_budget_mode: str,
    fixed_perturb_fraction: float | None,
    perturb_selection_mode: str,
    repair_steps: int,
    repairs_per_step: int,
    seed: int,
) -> dict[str, Any]:
    left = np.asarray(graph["left"], dtype=np.int64)
    right = np.asarray(graph["right"], dtype=np.int64)
    group_name = str(graph["group_name"])
    group_order = int(graph["group_order"])
    port_left = np.asarray(graph["port_left"], dtype=np.int16).copy()
    port_right = np.asarray(graph["port_right"], dtype=np.int16).copy()
    gauge = np.asarray(graph["gauge"], dtype=np.int16).copy()
    patch_count = int(graph["patch_count"])
    degree = np.asarray(graph["degree"], dtype=float)
    sector_replay = _production_sector_replay_contract(graph)
    production_sector_repair_enabled = bool(
        sector_replay["production_sector_repair_enabled"]
    )
    sector_repair_config = dict(graph.get("production_sector_repair_config", {}) or {})
    baseline_edge_residual = gauge_invariant_edge_residual(
        port_left,
        port_right,
        gauge,
        group_name=group_name,
        group_order=group_order,
    )
    baseline_signature = _local_node_signature(
        baseline_edge_residual,
        baseline_edge_residual,
        left,
        right,
        patch_count,
    )
    intervention_rng = _probe_rng(seed, "intervention")
    repair_rng = _probe_rng(seed, "repair")
    sector_rng = _probe_rng(seed, "sector")
    selected_edges, affected_nodes = _cap_collar_edges(points, cap, left, right)
    if selected_edges.size:
        if str(perturb_budget_mode) in {"fixed", "fixed_collar_fraction", "fixed_fraction"}:
            fraction = (
                float(fixed_perturb_fraction)
                if fixed_perturb_fraction is not None
                else float(perturb_strength)
            )
            fraction = min(1.0, max(0.0, fraction))
        else:
            modular_amount = abs(float(scale) * float(time_value))
            fraction = min(1.0, max(0.0, float(perturb_strength) * modular_amount / (2.0 * np.pi)))
        perturb_count = max(1, int(round(fraction * selected_edges.size)))
        perturb_count = min(perturb_count, selected_edges.size)
        chosen = _modular_selected_edges(
            points,
            cap,
            left,
            right,
            selected_edges,
            perturb_count,
            scale=float(scale),
            # A centered +/- probe must use the same edge support. The sign is
            # carried by inverse group moves below, not by a different sample.
            time_value=abs(float(time_value)),
            rng=intervention_rng,
            mode=str(perturb_selection_mode),
        )
        side = _perturb_side(
            points,
            cap,
            left[chosen],
            right[chosen],
            scale=float(scale),
            time_value=abs(float(time_value)),
            mode=str(perturb_selection_mode),
        )
        source_element = np.asarray(
            [1 + (abs(seed) % max(1, group_order - 1))],
            dtype=np.int16,
        )
        if float(time_value) < 0.0:
            source_element = group_inverse_indices(
                source_element,
                group_name=group_name,
                group_order=group_order,
            )
        # Right multiplication is equivariant under the local-frame left
        # action p_i -> h_i p_i, including for non-Abelian S3.
        left_edges = chosen[side]
        right_edges = chosen[~side]
        if left_edges.size:
            port_left[left_edges] = group_multiply_indices(
                port_left[left_edges],
                np.full(left_edges.size, source_element[0], dtype=np.int16),
                group_name=group_name,
                group_order=group_order,
            )
        if right_edges.size:
            port_right[right_edges] = group_multiply_indices(
                port_right[right_edges],
                np.full(right_edges.size, source_element[0], dtype=np.int16),
                group_name=group_name,
                group_order=group_order,
            )
        intervention_support_hash = _intervention_support_hash(chosen, side)
    else:
        intervention_support_hash = _intervention_support_hash(
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=bool),
        )
    repair_incident = np.zeros(patch_count, dtype=float)
    sector_edges_changed = 0
    sector_repair_move_calls = 0
    local_edge_mask = np.zeros(left.size, dtype=bool)
    if affected_nodes.size:
        affected = np.zeros(patch_count, dtype=bool)
        affected[affected_nodes] = True
        local_edge_mask = affected[left] | affected[right]
    for _step in range(max(0, int(repair_steps))):
        active = np.flatnonzero(
            local_edge_mask
            & covariant_mismatch_mask(
                port_left,
                port_right,
                gauge,
                group_name=group_name,
                group_order=group_order,
            )
        )
        if active.size == 0:
            break
        count = min(int(repairs_per_step), active.size)
        chosen = repair_rng.choice(active, size=count, replace=False)
        chosen_delta = covariant_discrepancy(
            port_left[chosen],
            port_right[chosen],
            gauge[chosen],
            group_name=group_name,
            group_order=group_order,
        )
        if production_sector_repair_enabled:
            sector_repair_move_calls += 1
        sector_edges_changed += repair_production_sector_links(
            gauge,
            chosen,
            chosen_delta,
            group_name=group_name,
            group_order=group_order,
            rng=sector_rng,
            config=sector_repair_config,
        )
        direction = repair_rng.random(count) < 0.5
        repair_covariant_port_pairs(
            port_left,
            port_right,
            gauge,
            chosen,
            direction,
            group_name=group_name,
            group_order=group_order,
        )
        repair_incident += np.bincount(left[chosen], minlength=patch_count) + np.bincount(right[chosen], minlength=patch_count)
    mismatch = covariant_mismatch_mask(
        port_left,
        port_right,
        gauge,
        group_name=group_name,
        group_order=group_order,
    )
    incident_mismatch = (
        np.bincount(left, weights=mismatch.astype(float), minlength=patch_count)
        + np.bincount(right, weights=mismatch.astype(float), minlength=patch_count)
    )
    edge_residual = gauge_invariant_edge_residual(
        port_left,
        port_right,
        gauge,
        group_name=group_name,
        group_order=group_order,
    )
    signature = _local_node_signature(edge_residual, edge_residual, left, right, patch_count)
    baseline_stable = np.asarray(raw_fields.get("stable_count", np.zeros(patch_count)), dtype=float)
    baseline_committed = np.asarray(raw_fields.get("committed_mask", np.zeros(patch_count)), dtype=float)
    unchanged = signature == baseline_signature
    stable_count = np.where(unchanged, baseline_stable + max(1, int(repair_steps)), 1.0)
    committed = baseline_committed * unchanged.astype(float)
    repair_load = incident_mismatch / np.maximum(degree, 1.0)
    cumulative = np.asarray(raw_fields.get("cumulative_repair_load", np.zeros(patch_count)), dtype=float) + repair_incident / np.maximum(degree, 1.0)
    post = dict(raw_fields)
    post.update(
        {
            "record_signature": signature.astype(float),
            "stable_count": stable_count.astype(float),
            "committed_mask": committed.astype(float),
            "repair_load": repair_load.astype(float),
            "local_mismatch_density": repair_load.astype(float),
            "cumulative_repair_load": cumulative.astype(float),
            "_gauge_covariant_probe_receipt": bool(
                sector_replay["gauge_covariant_probe_receipt"]
            ),
            "_gauge_covariant_overlap_schema": GAUGE_COVARIANT_OVERLAP_SCHEMA,
            "_centered_inverse_intervention_receipt": True,
            "_intervention_support_hash": intervention_support_hash,
            "_intervention_action": "right_multiplication_in_endpoint_frame",
            "_production_sector_repair_enabled": production_sector_repair_enabled,
            "_sector_repair_replayed": bool(sector_replay["sector_repair_replayed"]),
            "_sector_repair_move_calls": int(sector_repair_move_calls),
            "_sector_edges_changed": int(sector_edges_changed),
            "_production_move_contract": sector_replay,
            "_probe_rng_stream_schema": "oph-modular-response-probe-rng-v1",
        }
    )
    return post


def _cap_collar_edges(points: np.ndarray, cap: RoundCap, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cap = cap.normalized()
    dot = points @ cap.axis
    threshold = float(np.cos(cap.theta0))
    width = max(float(cap.collar_width), 2.5 / max(float(np.sqrt(points.shape[0])), 1.0))
    collar = np.abs(dot - threshold) <= width
    inside = dot >= threshold
    edge_mask = collar[left] | collar[right] | (inside[left] ^ inside[right])
    selected = np.flatnonzero(edge_mask).astype(np.int64)
    if selected.size == 0:
        scores = np.minimum(np.abs(dot[left] - threshold), np.abs(dot[right] - threshold))
        count = min(max(1, points.shape[0] // 64), scores.size)
        selected = np.argsort(scores)[:count].astype(np.int64)
    affected = np.unique(np.concatenate([left[selected], right[selected]])).astype(np.int64) if selected.size else np.zeros(0, dtype=np.int64)
    return selected, affected


def _intervention_support_hash(edges: np.ndarray, side: np.ndarray) -> str:
    selected = np.ascontiguousarray(np.asarray(edges, dtype="<i8").reshape(-1))
    endpoint_side = np.ascontiguousarray(np.asarray(side, dtype=np.uint8).reshape(-1))
    hasher = hashlib.sha256()
    hasher.update(b"oph-paired-cap-intervention-support-v1\0")
    hasher.update(selected.tobytes())
    hasher.update(endpoint_side.tobytes())
    return "sha256:" + hasher.hexdigest()


def _perturb_side(
    points: np.ndarray,
    cap: RoundCap,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    *,
    scale: float = 2.0 * np.pi,
    time_value: float = 0.0,
    mode: str = "lambda_displacement",
) -> np.ndarray:
    normalized_mode = str(mode).lower().replace("-", "_")
    if normalized_mode in {
        "lambda_collar_generator",
        "modular_collar_generator",
        "collar_generator",
        "lambda_signed_collar_generator",
        "modular_signed_collar_generator",
    }:
        cap = cap.normalized()
        midpoint = points[edge_left] + points[edge_right]
        midpoint = midpoint / np.maximum(np.linalg.norm(midpoint, axis=1, keepdims=True), 1e-12)
        vector = _cap_flow_vector(midpoint, cap, float(scale) * float(time_value))
        edge_vector = points[edge_left] - points[edge_right]
        signed = np.sum(edge_vector * vector, axis=1)
        fallback = np.abs(signed) <= 1e-12
        if np.any(fallback):
            dot_left = points[edge_left[fallback]] @ cap.axis
            dot_right = points[edge_right[fallback]] @ cap.axis
            signed[fallback] = dot_left - dot_right
        return signed >= 0.0
    dot_left = points[edge_left] @ cap.axis
    dot_right = points[edge_right] @ cap.axis
    return dot_left >= dot_right


def _modular_selected_edges(
    points: np.ndarray,
    cap: RoundCap,
    left: np.ndarray,
    right: np.ndarray,
    selected_edges: np.ndarray,
    count: int,
    *,
    scale: float,
    time_value: float,
    rng: np.random.Generator,
    mode: str,
) -> np.ndarray:
    if selected_edges.size <= int(count):
        return selected_edges
    cap = cap.normalized()
    mids = points[left[selected_edges]] + points[right[selected_edges]]
    mids = mids / np.maximum(np.linalg.norm(mids, axis=1, keepdims=True), 1e-12)
    normalized_mode = str(mode).lower().replace("-", "_")
    if normalized_mode in {"source_random", "uniform_random", "collar_random"}:
        order = rng.choice(selected_edges.size, size=int(count), replace=False)
        return selected_edges[np.sort(order)]
    if normalized_mode in {
        "lambda_collar_generator",
        "modular_collar_generator",
        "collar_generator",
        "lambda_signed_collar_generator",
        "modular_signed_collar_generator",
    }:
        score = _cap_collar_generator_edge_score(
            points,
            cap,
            left,
            right,
            selected_edges,
            scale=float(scale),
            time_value=float(time_value),
            rng=rng,
            signed=normalized_mode in {"lambda_signed_collar_generator", "modular_signed_collar_generator"},
        )
        order = np.argsort(score)[-int(count) :]
        return selected_edges[order]
    if normalized_mode in {"lambda", "lambda_displacement", "conformal_displacement", "transport_displacement"}:
        mapped = lambda_cap(mids, cap, float(scale) * float(time_value))
        cosine = np.clip(np.sum(mids * mapped, axis=1), -1.0, 1.0)
        displacement = np.arccos(cosine)
        # Break exact ties by the ordered boundary pair, but keep the actual
        # lambda_C displacement as the dominant selection criterion.
        tangent = cap.tangent / max(float(np.linalg.norm(cap.tangent)), 1e-12)
        bitangent = np.cross(cap.axis, tangent)
        bitangent = bitangent / max(float(np.linalg.norm(bitangent)), 1e-12)
        angle = np.arctan2(mids @ bitangent, mids @ tangent)
        score = displacement + 1e-6 * np.sin(angle) + 1e-9 * rng.normal(size=selected_edges.size)
        order = np.argsort(score)[-int(count) :]
        return selected_edges[order]
    tangent = cap.tangent / max(float(np.linalg.norm(cap.tangent)), 1e-12)
    bitangent = np.cross(cap.axis, tangent)
    bitangent = bitangent / max(float(np.linalg.norm(bitangent)), 1e-12)
    angle = np.arctan2(mids @ bitangent, mids @ tangent)
    score = np.sin(angle + float(scale) * float(time_value)) + 1e-9 * rng.normal(size=selected_edges.size)
    order = np.argsort(score)[-int(count) :]
    return selected_edges[order]


def _cap_collar_generator_edge_score(
    points: np.ndarray,
    cap: RoundCap,
    left: np.ndarray,
    right: np.ndarray,
    selected_edges: np.ndarray,
    *,
    scale: float,
    time_value: float,
    rng: np.random.Generator,
    signed: bool,
) -> np.ndarray:
    """Score perturbation edges by collar support and cap modular generator.

    The BW target is a cap-local modular automorphism, not a ranking by the
    largest finite point displacement. This score therefore concentrates on the
    boundary collar and the infinitesimal cap-flow generator while retaining the
    ordered cut-pair phase as a deterministic tie-breaker.
    """

    cap = cap.normalized()
    edge_left = left[selected_edges]
    edge_right = right[selected_edges]
    mids = points[edge_left] + points[edge_right]
    mids = mids / np.maximum(np.linalg.norm(mids, axis=1, keepdims=True), 1e-12)
    dot = mids @ cap.axis
    threshold = float(np.cos(cap.theta0))
    width = max(float(cap.collar_width), 2.5 / max(float(np.sqrt(points.shape[0])), 1.0))
    collar_distance = np.abs(dot - threshold)
    collar_score = np.exp(-np.square(collar_distance / max(width, 1e-12)))
    generator = _cap_flow_vector(mids, cap, 0.0)
    generator_norm = np.linalg.norm(generator, axis=1)
    generator_scale = float(np.percentile(generator_norm[generator_norm > 1e-12], 75)) if np.any(generator_norm > 1e-12) else 1.0
    generator_score = np.clip(generator_norm / max(generator_scale, 1e-12), 0.0, 3.0) / 3.0
    finite_flow = _cap_flow_vector(mids, cap, float(scale) * float(time_value))
    finite_norm = np.linalg.norm(finite_flow, axis=1)
    finite_scale = float(np.percentile(finite_norm[finite_norm > 1e-12], 75)) if np.any(finite_norm > 1e-12) else 1.0
    finite_score = np.clip(finite_norm / max(finite_scale, 1e-12), 0.0, 3.0) / 3.0
    tangent = cap.tangent / max(float(np.linalg.norm(cap.tangent)), 1e-12)
    bitangent = np.cross(cap.axis, tangent)
    bitangent = bitangent / max(float(np.linalg.norm(bitangent)), 1e-12)
    phase = np.arctan2(mids @ bitangent, mids @ tangent)
    orientation = np.cos(phase - float(scale) * float(time_value))
    ordered_pair_score = 0.5 + 0.5 * orientation
    if signed:
        directional = np.sign(orientation)
        directional[directional == 0.0] = 1.0
        ordered_pair_score = ordered_pair_score * directional
    score = (
        0.72 * collar_score
        + 0.18 * collar_score * generator_score
        + 0.08 * collar_score * finite_score
        + 0.02 * ordered_pair_score
    )
    return score + 1e-9 * rng.normal(size=selected_edges.size)


def _cap_flow_vector(points: np.ndarray, cap: RoundCap, s: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if abs(float(s)) > 1e-12:
        mapped = lambda_cap(points, cap, float(s))
        vector = mapped - points
    else:
        eps = 1e-4
        vector = (lambda_cap(points, cap, eps) - lambda_cap(points, cap, -eps)) / (2.0 * eps)
    vector = vector - points * np.sum(vector * points, axis=1, keepdims=True)
    return vector


def _transition_delta_same_support(
    support: np.ndarray,
    source_packets: np.ndarray,
    target_packets: np.ndarray,
    weights: np.ndarray,
    class_count: int,
) -> tuple[np.ndarray, float]:
    support = np.asarray(support, dtype=np.int64)
    valid = (support >= 0) & (support < source_packets.size) & (support < target_packets.size) & (support < weights.size)
    if not np.any(valid):
        return np.zeros(int(class_count), dtype=float), 0.0
    support = support[valid]
    source = np.asarray(source_packets[support], dtype=np.int64)
    target = np.asarray(target_packets[support], dtype=np.int64)
    local_weights = np.asarray(weights[support], dtype=float)
    total = max(float(np.sum(local_weights)), 1e-12)
    base = np.bincount(np.clip(source, 0, class_count - 1), weights=local_weights, minlength=class_count) / total
    pert = np.bincount(np.clip(target, 0, class_count - 1), weights=local_weights, minlength=class_count) / total
    change = float(np.sum(local_weights[source != target]) / total)
    return pert - base, change


def _transition_features_same_support(
    support: np.ndarray,
    source_packets: np.ndarray,
    target_packets: np.ndarray,
    weights: np.ndarray,
    class_count: int,
    *,
    feature_types: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray | float]:
    support = np.asarray(support, dtype=np.int64)
    valid = (support >= 0) & (support < source_packets.size) & (support < target_packets.size) & (support < weights.size)
    if not np.any(valid):
        return _empty_transition_features(class_count)
    support = support[valid]
    return _transition_features_from_indices(
        support,
        support,
        source_packets,
        target_packets,
        weights,
        class_count,
        feature_types=feature_types,
    )


def _transition_delta_between_supports(
    support: np.ndarray,
    transported: np.ndarray,
    source_packets: np.ndarray,
    target_packets: np.ndarray,
    weights: np.ndarray,
    class_count: int,
) -> tuple[np.ndarray, float]:
    support = np.asarray(support, dtype=np.int64)
    transported = np.asarray(transported, dtype=np.int64)
    if support.size == 0 or transported.size == 0:
        return np.zeros(int(class_count), dtype=float), 0.0
    count = min(support.size, transported.size)
    source = support[:count]
    target = transported[:count]
    valid = (
        (source >= 0)
        & (target >= 0)
        & (source < source_packets.size)
        & (target < target_packets.size)
        & (source < weights.size)
    )
    if not np.any(valid):
        return np.zeros(int(class_count), dtype=float), 0.0
    source = source[valid]
    target = target[valid]
    source_values = np.asarray(source_packets[source], dtype=np.int64)
    target_values = np.asarray(target_packets[target], dtype=np.int64)
    local_weights = np.asarray(weights[source], dtype=float)
    total = max(float(np.sum(local_weights)), 1e-12)
    base = np.bincount(np.clip(source_values, 0, class_count - 1), weights=local_weights, minlength=class_count) / total
    pert = np.bincount(np.clip(target_values, 0, class_count - 1), weights=local_weights, minlength=class_count) / total
    change = float(np.sum(local_weights[source_values != target_values]) / total)
    return pert - base, change


def _transition_features_between_supports(
    support: np.ndarray,
    transported: np.ndarray,
    source_packets: np.ndarray,
    target_packets: np.ndarray,
    weights: np.ndarray,
    class_count: int,
    *,
    feature_types: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray | float]:
    support = np.asarray(support, dtype=np.int64)
    transported = np.asarray(transported, dtype=np.int64)
    if support.size == 0 or transported.size == 0:
        return _empty_transition_features(class_count)
    count = min(support.size, transported.size)
    source = support[:count]
    target = transported[:count]
    valid = (
        (source >= 0)
        & (target >= 0)
        & (source < source_packets.size)
        & (target < target_packets.size)
        & (source < weights.size)
    )
    if not np.any(valid):
        return _empty_transition_features(class_count)
    return _transition_features_from_indices(
        source[valid],
        target[valid],
        source_packets,
        target_packets,
        weights,
        class_count,
        feature_types=feature_types,
    )


def _transition_features_from_indices(
    source: np.ndarray,
    target: np.ndarray,
    source_packets: np.ndarray,
    target_packets: np.ndarray,
    weights: np.ndarray,
    class_count: int,
    *,
    feature_types: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray | float]:
    requested = set(_normalized_transition_feature_types(feature_types)) if feature_types is not None else {
        "class_distribution_delta",
        "class_log_odds_delta",
        "transition_matrix_delta",
        "entropy_delta",
        "sector_preservation_delta",
        "change_probability_delta",
    }
    class_count = max(1, int(class_count))
    source_values = np.clip(np.asarray(source_packets[source], dtype=np.int64), 0, class_count - 1)
    target_values = np.clip(np.asarray(target_packets[target], dtype=np.int64), 0, class_count - 1)
    local_weights = np.asarray(weights[source], dtype=float)
    total = max(float(np.sum(local_weights)), 1e-12)
    result: dict[str, np.ndarray | float] = {}
    needs_distribution = bool(
        requested
        & {
            "class_distribution_delta",
            "class_log_odds_delta",
            "transition_matrix_delta",
            "entropy_delta",
        }
    )
    base: np.ndarray | None = None
    pert: np.ndarray | None = None
    class_delta: np.ndarray | None = None
    if needs_distribution:
        base = np.bincount(source_values, weights=local_weights, minlength=class_count) / total
        pert = np.bincount(target_values, weights=local_weights, minlength=class_count) / total
        class_delta = pert - base
    if "class_distribution_delta" in requested and class_delta is not None:
        result["class_distribution_delta"] = class_delta
        result["target_distribution_delta"] = class_delta
    if "class_log_odds_delta" in requested and base is not None and pert is not None:
        eps = 1e-6
        result["class_log_odds_delta"] = (
            np.log((pert + eps) / (1.0 - pert + eps))
            - np.log((base + eps) / (1.0 - base + eps))
        )
    if "transition_matrix_delta" in requested and base is not None:
        joint_index = source_values * class_count + target_values
        joint = np.bincount(joint_index, weights=local_weights, minlength=class_count * class_count)
        joint = joint.reshape((class_count, class_count)) / total
        result["transition_matrix_delta"] = (joint - np.diag(base)).reshape(-1)
    if "entropy_delta" in requested and base is not None and pert is not None:
        result["entropy_delta"] = float(_entropy_from_probabilities(pert) - _entropy_from_probabilities(base))
    if requested & {"sector_preservation_delta", "change_probability_delta"}:
        same_probability = float(np.sum(local_weights[source_values == target_values]) / total)
        if "sector_preservation_delta" in requested:
            result["sector_preservation_delta"] = float(same_probability - 1.0)
        if "change_probability_delta" in requested:
            result["change_probability_delta"] = float(1.0 - same_probability)
    return result


def _empty_transition_features(class_count: int) -> dict[str, np.ndarray | float]:
    return {
        "class_distribution_delta": np.zeros(int(class_count), dtype=float),
        "target_distribution_delta": np.zeros(int(class_count), dtype=float),
        "class_log_odds_delta": np.zeros(int(class_count), dtype=float),
        "transition_matrix_delta": np.zeros(int(class_count) * int(class_count), dtype=float),
        "entropy_delta": 0.0,
        "sector_preservation_delta": 0.0,
        "change_probability_delta": 0.0,
    }


def _entropy_from_probabilities(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values > 0.0]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def _normalized_transition_feature_types(feature_types: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    aliases = {
        "target_distribution_delta": "class_distribution_delta",
        "class_delta": "class_distribution_delta",
        "log_odds_delta": "class_log_odds_delta",
        "transition_delta": "transition_matrix_delta",
        "preservation_delta": "sector_preservation_delta",
        "change": "change_probability_delta",
    }
    supported = {
        "class_distribution_delta",
        "class_log_odds_delta",
        "transition_matrix_delta",
        "entropy_delta",
        "sector_preservation_delta",
        "change_probability_delta",
    }
    requested = feature_types or ("class_distribution_delta", "change_probability_delta")
    normalized: list[str] = []
    for value in requested:
        key = aliases.get(str(value).lower().replace("-", "_"), str(value).lower().replace("-", "_"))
        if key in supported and key not in normalized:
            normalized.append(key)
    if not normalized:
        normalized = ["class_distribution_delta", "change_probability_delta"]
    return tuple(normalized)


def _empty_transition_feature_array(observer_count: int, class_count: int, feature_type: str) -> np.ndarray:
    if feature_type in {"class_distribution_delta", "target_distribution_delta", "class_log_odds_delta"}:
        return np.zeros((int(observer_count), int(class_count)), dtype=float)
    if feature_type == "transition_matrix_delta":
        return np.zeros((int(observer_count), int(class_count) * int(class_count)), dtype=float)
    return np.zeros(int(observer_count), dtype=float)


def _transition_feature_columns(
    feature_values: dict[str, np.ndarray],
    feature_types: tuple[str, ...],
    class_count: int,
) -> list[np.ndarray]:
    columns: list[np.ndarray] = []
    for feature_type in feature_types:
        values = np.asarray(feature_values[feature_type], dtype=float)
        if feature_type in {"class_distribution_delta", "target_distribution_delta", "class_log_odds_delta"}:
            for class_index in range(int(class_count)):
                columns.append(values[:, class_index])
        elif feature_type == "transition_matrix_delta":
            for transition_index in range(int(class_count) * int(class_count)):
                columns.append(values[:, transition_index])
        else:
            columns.append(values)
    return columns


def _transport_support_direct(
    points: np.ndarray,
    tree: cKDTree | None,
    support: np.ndarray,
    cap: RoundCap,
    s: float,
) -> np.ndarray:
    support = np.asarray(support, dtype=np.int64)
    if support.size == 0:
        return np.zeros(0, dtype=np.int64)
    valid = support[(support >= 0) & (support < points.shape[0])]
    if valid.size == 0:
        return np.zeros(0, dtype=np.int64)
    local_tree = tree or cKDTree(points)
    mapped = lambda_cap(points[valid], cap, float(s))
    _distances, indices = local_tree.query(mapped, k=1)
    return np.asarray(indices, dtype=np.int64)


def _validated_graph_state(graph_state: dict[str, Any], patch_count: int) -> dict[str, Any] | None:
    required = [
        "left",
        "right",
        "port_left",
        "port_right",
        "gauge",
        "group_name",
        "group_order",
        "patch_count",
    ]
    if any(key not in graph_state for key in required):
        return None
    left = np.asarray(graph_state["left"], dtype=np.int64)
    right = np.asarray(graph_state["right"], dtype=np.int64)
    port_left = np.asarray(graph_state["port_left"], dtype=np.int16)
    port_right = np.asarray(graph_state["port_right"], dtype=np.int16)
    gauge = np.asarray(graph_state["gauge"], dtype=np.int16)
    if not (left.shape == right.shape == port_left.shape == port_right.shape == gauge.shape):
        return None
    count = int(graph_state.get("patch_count", patch_count))
    degree = np.bincount(np.concatenate([left, right]), minlength=count).astype(float) if left.size else np.ones(count, dtype=float)
    degree = np.maximum(degree, 1.0)
    sector_config_value = graph_state.get("production_sector_repair_config")
    sector_config_available = isinstance(sector_config_value, dict)
    sector_config = dict(sector_config_value) if sector_config_available else {}
    sector_enabled = bool(
        graph_state.get(
            "production_sector_repair_enabled",
            sector_config.get("enabled", False),
        )
    )
    return {
        "left": left,
        "right": right,
        "port_left": port_left,
        "port_right": port_right,
        "gauge": gauge,
        "group_name": str(graph_state["group_name"]),
        "group_order": int(graph_state["group_order"]),
        "patch_count": count,
        "degree": degree,
        "production_sector_repair_enabled": sector_enabled,
        "production_sector_repair_config": sector_config,
        "production_sector_repair_config_available": sector_config_available,
    }


def _production_sector_replay_contract(graph: dict[str, Any]) -> dict[str, Any]:
    """Describe whether a probe can replay the configured production move."""

    requested = bool(graph.get("production_sector_repair_enabled", False))
    config_available = bool(
        graph.get(
            "production_sector_repair_config_available",
            isinstance(graph.get("production_sector_repair_config"), dict),
        )
    )
    config = dict(graph.get("production_sector_repair_config", {}) or {})
    configured_enabled = bool(config.get("enabled", False))
    group_name = str(graph.get("group_name", "")).upper()
    group_order = int(graph.get("group_order", 0) or 0)
    mode = str(config.get("mode", "repair_coupled_group_compose"))
    try:
        probability = float(config.get("probability", 0.0))
    except (TypeError, ValueError):
        probability = float("nan")
    active_link_mutation = bool(
        requested
        and configured_enabled
        and group_name == "S3"
        and group_order == 6
        and np.isfinite(probability)
        and probability > 0.0
    )
    blockers: list[str] = []
    if requested and not config_available:
        blockers.append("production_sector_repair_config_unavailable")
    if requested and config_available and not configured_enabled:
        blockers.append("production_sector_repair_enablement_mismatch")
    if requested and not np.isfinite(probability):
        blockers.append("production_sector_repair_probability_invalid")
    if active_link_mutation and mode != "repair_coupled_group_compose":
        blockers.append("production_sector_repair_mode_not_gauge_covariant")
    exact_move_set_replayed = not blockers
    return {
        "schema": "bw_array_production_overlap_move_contract_v1",
        "mismatch_definition": GAUGE_COVARIANT_OVERLAP_SCHEMA,
        "production_sector_repair_enabled": requested,
        "production_sector_repair_config_available": config_available,
        "production_sector_repair_config": config,
        "active_sector_link_mutation": active_link_mutation,
        "sector_repair_mode": mode,
        "sector_repair_probability": probability if np.isfinite(probability) else None,
        "endpoint_repair_replayed": True,
        "sector_repair_replayed": bool(requested and exact_move_set_replayed),
        "exact_production_move_set_replayed": exact_move_set_replayed,
        "gauge_covariant_probe_receipt": exact_move_set_replayed,
        "blockers": blockers,
    }


def _probe_rng(seed: int, stream_name: str) -> np.random.Generator:
    """Return a stable name-isolated RNG for one perturb/resettle subsystem."""

    seed_u64 = int(seed) % (1 << 64)
    material = f"oph-modular-response-probe-rng-v1\0{stream_name}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    words = [
        seed_u64 & 0xFFFFFFFF,
        (seed_u64 >> 32) & 0xFFFFFFFF,
        *(int(value) for value in np.frombuffer(digest[:16], dtype="<u4")),
    ]
    return np.random.default_rng(np.random.SeedSequence(words))


def _local_node_signature(
    port_left: np.ndarray,
    port_right: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    patch_count: int,
) -> np.ndarray:
    return (
        np.bincount(left, weights=(port_left + 1), minlength=patch_count)
        + np.bincount(right, weights=(port_right + 1), minlength=patch_count)
    ).astype(np.int64)


def _object_packet_fields(
    raw_fields: dict[str, np.ndarray],
    observables: list[str] | tuple[str, ...],
    *,
    bins: int,
    record_family_modulus: int,
) -> dict[str, np.ndarray]:
    packets: dict[str, np.ndarray] = {}
    patch_count = _infer_patch_count(raw_fields)
    if patch_count == 0:
        return packets
    stable = np.asarray(raw_fields.get("stable_count", np.zeros(patch_count)), dtype=float)
    committed = np.asarray(raw_fields.get("committed_mask", np.zeros(patch_count)), dtype=float)
    for name in observables:
        key = str(name)
        if key == "stable_flag":
            threshold = max(1.0, float(np.median(stable)) if stable.size else 1.0)
            packets[key] = (stable >= threshold).astype(np.int64)
        elif key == "checkpoint_class":
            threshold = max(1.0, float(np.median(stable)) if stable.size else 1.0)
            stable_flag = (stable >= threshold).astype(np.int64)
            packets[key] = (2 * (committed > 0.5).astype(np.int64) + stable_flag).astype(np.int64)
        elif key == "record_family":
            values = np.asarray(raw_fields.get("record_signature", np.zeros(patch_count)), dtype=np.int64)
            packets[key] = np.mod(np.abs(values), int(record_family_modulus)).astype(np.int64)
        elif key == "s3_sector_class":
            values = np.asarray(raw_fields.get("s3_sector_class", np.zeros(patch_count)), dtype=np.int64)
            packets[key] = np.mod(np.abs(values), 6).astype(np.int64)
        elif key == "repair_load_bucket":
            packets[key] = _quantile_buckets(np.asarray(raw_fields.get("repair_load", np.zeros(patch_count)), dtype=float), bins)
        elif key == "cumulative_repair_load_bucket":
            packets[key] = _quantile_buckets(
                np.asarray(raw_fields.get("cumulative_repair_load", np.zeros(patch_count)), dtype=float),
                bins,
            )
        elif key == "committed_object_normal_form":
            signature = np.asarray(raw_fields.get("record_signature", np.zeros(patch_count)), dtype=np.int64)
            sector = np.asarray(raw_fields.get("s3_sector_class", np.zeros(patch_count)), dtype=np.int64)
            stable_flag = (stable >= max(1.0, float(np.median(stable)) if stable.size else 1.0)).astype(np.int64)
            packets[key] = np.mod(np.abs(signature) + 7 * np.abs(sector) + 13 * stable_flag, int(record_family_modulus)).astype(np.int64)
        elif key in raw_fields:
            values = np.asarray(raw_fields[key], dtype=float)
            packets[key] = _quantile_buckets(values, bins)
    return packets


def _object_transition_feature_rows(
    caps: list[RoundCap],
    times: list[float] | tuple[float, ...],
    packet_fields: dict[str, np.ndarray],
    feature_types: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    feature_types = _normalized_transition_feature_types(feature_types)
    rows: list[dict[str, Any]] = []
    for cap_index, _cap in enumerate(caps):
        for time_index, time_value in enumerate(times):
            for observable, packets in packet_fields.items():
                class_count = int(np.max(packets)) + 1 if packets.size else 1
                for feature_type in feature_types:
                    if feature_type in {"class_distribution_delta", "target_distribution_delta"}:
                        for target_class in range(class_count):
                            rows.append(
                                {
                                    "feature_index": len(rows),
                                    "cap_index": int(cap_index),
                                    "time_index": int(time_index),
                                    "time": float(time_value),
                                    "field": f"{observable}:class_delta_{target_class}",
                                    "observable": str(observable),
                                    "target_class": int(target_class),
                                    "feature_type": "class_distribution_delta",
                                }
                            )
                    elif feature_type == "class_log_odds_delta":
                        for target_class in range(class_count):
                            rows.append(
                                {
                                    "feature_index": len(rows),
                                    "cap_index": int(cap_index),
                                    "time_index": int(time_index),
                                    "time": float(time_value),
                                    "field": f"{observable}:log_odds_delta_{target_class}",
                                    "observable": str(observable),
                                    "target_class": int(target_class),
                                    "feature_type": "class_log_odds_delta",
                                }
                            )
                    elif feature_type == "transition_matrix_delta":
                        for source_class in range(class_count):
                            for target_class in range(class_count):
                                rows.append(
                                    {
                                        "feature_index": len(rows),
                                        "cap_index": int(cap_index),
                                        "time_index": int(time_index),
                                        "time": float(time_value),
                                        "field": f"{observable}:transition_{source_class}_to_{target_class}",
                                        "observable": str(observable),
                                        "source_class": int(source_class),
                                        "target_class": int(target_class),
                                        "feature_type": "transition_matrix_delta",
                                    }
                                )
                    elif feature_type in {
                        "entropy_delta",
                        "sector_preservation_delta",
                        "change_probability_delta",
                    }:
                        field_suffix = {
                            "entropy_delta": "entropy_delta",
                            "sector_preservation_delta": "sector_preservation",
                            "change_probability_delta": "change",
                        }[feature_type]
                        rows.append(
                            {
                                "feature_index": len(rows),
                                "cap_index": int(cap_index),
                                "time_index": int(time_index),
                                "time": float(time_value),
                                "field": f"{observable}:{field_suffix}",
                                "observable": str(observable),
                                "target_class": None,
                                "feature_type": feature_type,
                            }
                        )
    return rows


def _response_event_row_ids(feature_rows: list[dict[str, Any]]) -> list[str]:
    """Return immutable identifiers for response-event feature columns."""

    identifiers: list[str] = []
    for row in feature_rows:
        payload = "|".join(
            [
                str(int(row.get("cap_index", -1))),
                str(int(row.get("time_index", -1))),
                format(float(row.get("time", 0.0)), ".17g"),
                str(row.get("observable", row.get("field", "unknown"))),
                str(row.get("feature_type", "scalar")),
                str(row.get("source_class")),
                str(row.get("target_class")),
            ]
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        identifiers.append(f"oph-response-event-v1:{digest}")
    return identifiers


def _object_transition_matrix(
    cache: GeometryCache,
    supports: list[np.ndarray],
    caps: list[RoundCap],
    times: list[float] | tuple[float, ...],
    packet_fields: dict[str, np.ndarray],
    weights: np.ndarray,
    feature_types: tuple[str, ...] | None = None,
    *,
    scale: float,
    k_transport: int,
) -> np.ndarray:
    feature_types = _normalized_transition_feature_types(feature_types)
    columns: list[np.ndarray] = []
    for cap_index, cap in enumerate(caps):
        for time_value in times:
            transported_supports = [
                _transported_support(cache, support, cap, float(scale) * float(time_value), k=int(k_transport), cap_id=cap_index)
                for support in supports
            ]
            for _observable, packets in packet_fields.items():
                class_count = int(np.max(packets)) + 1 if packets.size else 1
                feature_values: dict[str, np.ndarray] = {
                    feature_type: _empty_transition_feature_array(len(supports), class_count, feature_type)
                    for feature_type in feature_types
                }
                for row_index, (support, transported) in enumerate(zip(supports, transported_supports, strict=False)):
                    values = _transition_features_for_support(
                        support,
                        transported,
                        packets,
                        weights,
                        class_count,
                        feature_types=feature_types,
                    )
                    for feature_type in feature_types:
                        feature_values[feature_type][row_index] = values[feature_type]
                columns.extend(_transition_feature_columns(feature_values, feature_types, class_count))
    return np.vstack(columns).T if columns else np.zeros((len(supports), 0), dtype=float)


def _collar_operator_transition_matrix(
    cache: GeometryCache,
    supports: list[np.ndarray],
    caps: list[RoundCap],
    times: list[float] | tuple[float, ...],
    packet_fields: dict[str, np.ndarray],
    weights: np.ndarray,
    feature_types: tuple[str, ...] | None = None,
    *,
    scale: float,
    k_transport: int,
) -> np.ndarray:
    feature_types = _normalized_transition_feature_types(feature_types)
    points = np.asarray(cache.points, dtype=float)
    columns: list[np.ndarray] = []
    for cap_index, cap in enumerate(caps):
        for time_value in times:
            s = float(scale) * float(time_value)
            transported_supports = [
                _transported_support(cache, support, cap, s, k=int(k_transport), cap_id=cap_index)
                for support in supports
            ]
            operator_weights = _collar_operator_node_weights(points, cap, s, weights)
            for _observable, packets in packet_fields.items():
                class_count = int(np.max(packets)) + 1 if packets.size else 1
                feature_values: dict[str, np.ndarray] = {
                    feature_type: _empty_transition_feature_array(len(supports), class_count, feature_type)
                    for feature_type in feature_types
                }
                for row_index, (support, transported) in enumerate(
                    zip(supports, transported_supports, strict=False)
                ):
                    values = _transition_features_between_supports(
                        support,
                        transported,
                        np.asarray(packets, dtype=np.int64),
                        np.asarray(packets, dtype=np.int64),
                        operator_weights,
                        class_count,
                        feature_types=feature_types,
                    )
                    for feature_type in feature_types:
                        feature_values[feature_type][row_index] = values[feature_type]
                columns.extend(_transition_feature_columns(feature_values, feature_types, class_count))
    return np.vstack(columns).T if columns else np.zeros((len(supports), 0), dtype=float)


def _collar_operator_node_weights(
    points: np.ndarray,
    cap: RoundCap,
    s: float,
    base_weights: np.ndarray,
) -> np.ndarray:
    cap = cap.normalized()
    points = np.asarray(points, dtype=float)
    base = np.asarray(base_weights, dtype=float)
    dot = points @ cap.axis
    threshold = float(np.cos(cap.theta0))
    width = max(float(cap.collar_width), 2.5 / max(float(np.sqrt(points.shape[0])), 1.0))
    soft_inside = 1.0 / (1.0 + np.exp(-np.clip((dot - threshold) / width, -60.0, 60.0)))
    collar = 4.0 * soft_inside * (1.0 - soft_inside)
    mapped = lambda_cap(points, cap, float(s))
    cosine = np.clip(np.sum(points * mapped, axis=1), -1.0, 1.0)
    displacement = np.arccos(cosine)
    nonzero = displacement[displacement > 1e-12]
    displacement_scale = float(np.percentile(nonzero, 75)) if nonzero.size else 1.0
    displacement_score = np.clip(displacement / max(displacement_scale, 1e-12), 0.0, 3.0) / 3.0
    tangent = cap.tangent / max(float(np.linalg.norm(cap.tangent)), 1e-12)
    bitangent = np.cross(cap.axis, tangent)
    bitangent = bitangent / max(float(np.linalg.norm(bitangent)), 1e-12)
    phase = np.arctan2(points @ bitangent, points @ tangent)
    ordered_pair_score = 0.5 + 0.5 * np.cos(phase)
    operator = 0.10 + 0.55 * collar + 0.25 * displacement_score + 0.10 * ordered_pair_score
    operator = np.maximum(operator, 1e-6)
    return base * operator


def _transition_delta_for_support(
    support: np.ndarray,
    transported: np.ndarray,
    packets: np.ndarray,
    weights: np.ndarray,
    class_count: int,
) -> tuple[np.ndarray, float]:
    support = np.asarray(support, dtype=np.int64)
    transported = np.asarray(transported, dtype=np.int64)
    if support.size == 0 or transported.size == 0:
        return np.zeros(int(class_count), dtype=float), 0.0
    count = min(support.size, transported.size)
    source = support[:count]
    target = transported[:count]
    valid = (
        (source >= 0)
        & (source < packets.size)
        & (target >= 0)
        & (target < packets.size)
        & (source < weights.size)
    )
    if not np.any(valid):
        return np.zeros(int(class_count), dtype=float), 0.0
    source = source[valid]
    target = target[valid]
    source_packets = np.asarray(packets[source], dtype=np.int64)
    target_packets = np.asarray(packets[target], dtype=np.int64)
    local_weights = np.asarray(weights[source], dtype=float)
    total = max(float(np.sum(local_weights)), 1e-12)
    base = np.bincount(np.clip(source_packets, 0, class_count - 1), weights=local_weights, minlength=class_count) / total
    pert = np.bincount(np.clip(target_packets, 0, class_count - 1), weights=local_weights, minlength=class_count) / total
    change = float(np.sum(local_weights[source_packets != target_packets]) / total)
    return pert - base, change


def _transition_features_for_support(
    support: np.ndarray,
    transported: np.ndarray,
    packets: np.ndarray,
    weights: np.ndarray,
    class_count: int,
    *,
    feature_types: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray | float]:
    support = np.asarray(support, dtype=np.int64)
    transported = np.asarray(transported, dtype=np.int64)
    if support.size == 0 or transported.size == 0:
        return _empty_transition_features(class_count)
    count = min(support.size, transported.size)
    source = support[:count]
    target = transported[:count]
    valid = (
        (source >= 0)
        & (source < packets.size)
        & (target >= 0)
        & (target < packets.size)
        & (source < weights.size)
    )
    if not np.any(valid):
        return _empty_transition_features(class_count)
    packets = np.asarray(packets, dtype=np.int64)
    return _transition_features_from_indices(
        source[valid],
        target[valid],
        packets,
        packets,
        weights,
        class_count,
        feature_types=feature_types,
    )


def _object_s2_boundary_matrix(
    points: np.ndarray,
    caps: list[RoundCap],
    supports: list[np.ndarray],
    weights: np.ndarray,
    feature_rows: list[dict[str, Any]],
) -> np.ndarray:
    profiles = [_observer_s2_profile(points, cap, supports, weights) for cap in caps]
    columns = []
    for row in feature_rows:
        cap_index = int(row.get("cap_index", 0))
        profile = profiles[cap_index] if 0 <= cap_index < len(profiles) else np.zeros(len(supports), dtype=float)
        columns.append(profile)
    return np.vstack(columns).T if columns else np.zeros((len(supports), 0), dtype=float)


def _transform_with_controls(
    matrix: np.ndarray,
    controls: dict[str, np.ndarray],
    *,
    transform: str,
    feature_rows: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return matrix, controls, {"mode": str(transform), "empty": True}
    mode = str(transform)
    if mode in {"none", "identity"}:
        return matrix, {name: np.asarray(value, dtype=float) for name, value in controls.items()}, {"mode": "identity"}
    if mode == "sigmoid":
        transformed = 1.0 / (1.0 + np.exp(-np.clip(matrix, -60.0, 60.0)))
        return transformed, {
            name: 1.0 / (1.0 + np.exp(-np.clip(np.asarray(value, dtype=float), -60.0, 60.0)))
            for name, value in controls.items()
        }, {"mode": "sigmoid"}
    if mode in {"signed_robust_zscore", "robust_signed_zscore", "signed_zscore_clip"}:
        center = np.median(matrix, axis=0)
        q25 = np.quantile(matrix, 0.25, axis=0)
        q75 = np.quantile(matrix, 0.75, axis=0)
        robust_scale = (q75 - q25) / 1.349
        std_scale = np.std(matrix, axis=0)
        scale = np.where(robust_scale > 1e-6, robust_scale, std_scale)
        scale = np.where(scale > 0.05, scale, 0.05)
        clip = 6.0
        transformed = np.clip((matrix - center) / scale, -clip, clip)
        transformed_controls = {
            name: np.clip((np.asarray(value, dtype=float) - center) / scale, -clip, clip)
            for name, value in controls.items()
        }
        return transformed, transformed_controls, {
            "mode": "signed_robust_zscore",
            "mean_abs_center": float(np.mean(np.abs(center))),
            "median_scale": float(np.median(scale)),
            "min_scale_floor": 0.05,
            "clip": clip,
            "zero_robust_scale_columns": int(np.sum(robust_scale <= 1e-6)),
            "zero_std_columns": int(np.sum(std_scale <= 1e-9)),
        }
    if mode in {
        "signed_group_robust_zscore",
        "grouped_signed_robust_zscore",
        "time_grouped_robust_zscore",
        "bw_time_grouped_robust_zscore",
    }:
        rows = list(feature_rows or [])
        if len(rows) != matrix.shape[1]:
            transformed, transformed_controls, report = _transform_with_controls(
                matrix,
                controls,
                transform="signed_robust_zscore",
                feature_rows=None,
            )
            report = {
                **report,
                "mode": "signed_group_robust_zscore_fallback_columnwise",
                "fallback_reason": "feature_rows_missing_or_wrong_length",
            }
            return transformed, transformed_controls, report
        group_keys = [_time_group_key(row) for row in rows]
        center = np.zeros(matrix.shape[1], dtype=float)
        scale = np.ones(matrix.shape[1], dtype=float)
        for key in sorted(set(group_keys)):
            indices = np.asarray([index for index, value in enumerate(group_keys) if value == key], dtype=np.int64)
            values = matrix[:, indices].reshape(-1)
            group_center = float(np.median(values)) if values.size else 0.0
            q25 = float(np.quantile(values, 0.25)) if values.size else 0.0
            q75 = float(np.quantile(values, 0.75)) if values.size else 0.0
            robust_scale = (q75 - q25) / 1.349
            std_scale = float(np.std(values)) if values.size else 0.0
            group_scale = robust_scale if robust_scale > 1e-6 else std_scale
            group_scale = group_scale if group_scale > 0.05 else 0.05
            center[indices] = group_center
            scale[indices] = group_scale
        clip = 6.0
        transformed = np.clip((matrix - center[None, :]) / scale[None, :], -clip, clip)
        transformed_controls = {
            name: np.clip((np.asarray(value, dtype=float) - center[None, :]) / scale[None, :], -clip, clip)
            for name, value in controls.items()
        }
        return transformed, transformed_controls, {
            "mode": "signed_group_robust_zscore",
            "group_mode": "cap_observable_feature_class_tied_over_time",
            "group_count": int(len(set(group_keys))),
            "mean_abs_center": float(np.mean(np.abs(center))),
            "median_scale": float(np.median(scale)),
            "min_scale_floor": 0.05,
            "clip": clip,
            "claim_boundary": (
                "grouped support-visible feature normalization. Time-indexed columns sharing the same "
                "cap/observable/feature/class use one robust center and scale, so modular-time amplitude "
                "is not normalized away before BW/H3 fitting."
            ),
        }
    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    transformed = (matrix - center) / scale
    transformed_controls = {
        name: (np.asarray(value, dtype=float) - center) / scale
        for name, value in controls.items()
    }
    return transformed, transformed_controls, {
        "mode": "signed_zscore",
        "mean_abs_center": float(np.mean(np.abs(center))),
        "median_scale": float(np.median(scale)),
        "zero_scale_columns": int(np.sum(np.std(matrix, axis=0) <= 1e-9)),
    }


def _time_group_key(row: dict[str, Any]) -> str:
    return ":".join(
        [
            str(int(row.get("cap_index", 0))),
            str(row.get("observable", row.get("field", "field"))),
            str(row.get("feature_type", "")),
            str(row.get("source_class", "")),
            str(row.get("target_class", "")),
        ]
    )


def _infer_patch_count(raw_fields: dict[str, np.ndarray]) -> int:
    for value in raw_fields.values():
        array = np.asarray(value)
        if array.ndim >= 1:
            return int(array.shape[0])
    return 0


def _quantile_buckets(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    if float(np.std(values)) <= 1e-12:
        return np.zeros(values.shape[0], dtype=np.int64)
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)[1:-1]
    edges = np.unique(np.quantile(values, quantiles))
    if edges.size == 0:
        return np.zeros(values.shape[0], dtype=np.int64)
    return np.digitize(values, edges, right=False).astype(np.int64)


def _scale_label(scale: float) -> str:
    if abs(scale - 1.0) < 1e-12:
        return "1x"
    if abs(scale - np.pi) < 1e-12:
        return "pi"
    if abs(scale - 2.0 * np.pi) < 1e-12:
        return "2pi"
    if abs(scale - 4.0 * np.pi) < 1e-12:
        return "4pi"
    return f"{float(scale):.6g}"


def _same_probe_scale(left: float, right: float) -> bool:
    """Match declared scales tightly enough to reuse a deterministic rerun."""

    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1.0e-15))


def _standardized_fields(raw_fields: dict[str, np.ndarray], field_names: list[str] | tuple[str, ...]) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    for name in field_names:
        if name not in raw_fields:
            continue
        values = np.asarray(raw_fields[name], dtype=float)
        std = float(np.std(values))
        fields[str(name)] = (values - float(np.mean(values))) / std if std > 1e-12 else values - float(np.mean(values))
    return fields


def _valid_support(support_nodes: list[int] | np.ndarray, patch_count: int) -> np.ndarray:
    support = np.asarray(support_nodes, dtype=np.int64)
    return support[(support >= 0) & (support < int(patch_count))]


def _transported_support(cache: GeometryCache, support: np.ndarray, cap: RoundCap, s: float, *, k: int, cap_id: int) -> np.ndarray:
    mapped, weights = cache.transport_support(support, cap, s, k=k, cap_id=cap_id)
    if mapped.size == 0:
        return np.zeros(0, dtype=np.int64)
    if mapped.ndim == 1:
        return np.asarray(mapped, dtype=np.int64)
    if mapped.shape[1] == 1:
        return np.asarray(mapped[:, 0], dtype=np.int64)
    return np.asarray(mapped[np.arange(mapped.shape[0]), np.argmax(weights, axis=1)], dtype=np.int64)


def _support_feature(support: np.ndarray, fields: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    if support.size == 0:
        return np.zeros(len(fields), dtype=float)
    support_w = weights[support]
    total = max(float(np.sum(support_w)), 1e-12)
    return np.asarray([float(np.sum(values[support] * support_w) / total) for values in fields.values()], dtype=float)


def _field_scales(source_features: np.ndarray) -> np.ndarray:
    if source_features.size == 0:
        return np.ones(0, dtype=float)
    scales = np.std(source_features, axis=0)
    return np.maximum(scales, 0.25)


def _observer_s2_profile(points: np.ndarray, cap: RoundCap, supports: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    values = []
    cap_w = cap_weights(points, cap, soft=True)
    for support in supports:
        if support.size == 0:
            values.append(0.0)
            continue
        support_w = weights[support]
        values.append(float(np.sum(cap_w[support] * support_w) / max(float(np.sum(support_w)), 1e-12)))
    return np.asarray(values, dtype=float)


def _response_summary(matrix: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0, "mean_row_std": 0.0, "mean_col_std": 0.0}
    return {
        "min": float(np.min(matrix)),
        "mean": float(np.mean(matrix)),
        "max": float(np.max(matrix)),
        "std": float(np.std(matrix)),
        "mean_row_std": float(np.mean(np.std(matrix, axis=1))),
        "mean_col_std": float(np.mean(np.std(matrix, axis=0))),
    }
