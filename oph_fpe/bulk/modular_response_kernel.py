from __future__ import annotations

from typing import Any

import numpy as np

from oph_fpe.bulk.cap_geometry import RoundCap, cap_weights
from oph_fpe.bulk.record_to_h3 import DEFAULT_RECORD_FIELDS
from oph_fpe.cache.geometry_cache import GeometryCache


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
    transition_bins: int = 8,
    record_family_modulus: int = 16,
    transform: str = "sigmoid",
    wrong_scales: list[float] | tuple[float, ...] = (1.0,),
    graph_state: dict[str, Any] | None = None,
    perturb_strength: float = 1.0,
    repair_steps: int = 4,
    repairs_per_step: int = 128,
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
    if str(observable_mode) == "perturb_resettle_transition":
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
            transition_bins=int(transition_bins),
            record_family_modulus=int(record_family_modulus),
            transform=str(transform),
            wrong_scales=wrong_scales,
            perturb_strength=float(perturb_strength),
            repair_steps=int(repair_steps),
            repairs_per_step=int(repairs_per_step),
            seed=int(perturb_seed),
        )
    if str(observable_mode) == "object_transition":
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
    feature_rows = _object_transition_feature_rows(caps, times, packet_fields)
    raw_matrix = _object_transition_matrix(
        cache,
        supports,
        caps,
        times,
        packet_fields,
        weights,
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
    transition_bins: int,
    record_family_modulus: int,
    transform: str,
    wrong_scales: list[float] | tuple[float, ...],
    perturb_strength: float,
    repair_steps: int,
    repairs_per_step: int,
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
        return _empty_kernel(points, patch_views, caps, packet_fields, times)
    supports = [_valid_support(view.get("support_nodes", []), points.shape[0]) for view in patch_views]
    feature_rows = _object_transition_feature_rows(caps, times, packet_fields)
    raw_matrix = _perturb_resettle_matrix(
        points,
        caps,
        raw_fields,
        packet_fields,
        supports,
        weights,
        graph,
        times=times,
        scale=float(transport_scale),
        transition_observables=transition_observables,
        transition_bins=int(transition_bins),
        record_family_modulus=int(record_family_modulus),
        perturb_strength=float(perturb_strength),
        repair_steps=int(repair_steps),
        repairs_per_step=int(repairs_per_step),
        seed=int(seed),
    )
    no_perturb = np.zeros_like(raw_matrix)
    s2_boundary = _object_s2_boundary_matrix(points, caps, supports, weights, feature_rows)
    wrong_scale_controls: dict[str, np.ndarray] = {}
    for wrong_scale in wrong_scales:
        label = _scale_label(float(wrong_scale))
        wrong_scale_controls[label] = _perturb_resettle_matrix(
            points,
            caps,
            raw_fields,
            packet_fields,
            supports,
            weights,
            graph,
            times=times,
            scale=float(wrong_scale),
            transition_observables=transition_observables,
            transition_bins=int(transition_bins),
            record_family_modulus=int(record_family_modulus),
            perturb_strength=float(perturb_strength),
            repair_steps=int(repair_steps),
            repairs_per_step=int(repairs_per_step),
            seed=int(seed) + 311,
        )
    matrix, transformed_controls, transform_report = _transform_with_controls(
        raw_matrix,
        {
            "no_modular_flow_control": no_perturb,
            "s2_boundary_control": s2_boundary,
            **{f"wrong_scale_{label}": value for label, value in wrong_scale_controls.items()},
        },
        transform=str(transform),
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
            "repair_steps": int(repair_steps),
            "repairs_per_step": int(repairs_per_step),
            "perturb_strength": float(perturb_strength),
            "graph_edge_count": int(graph["left"].size),
            "claim_boundary": (
                "finite cap/collar perturb-resettle surrogate: perturb boundary/collar edge packets, "
                "run local overlap repair, then read observer packet transitions"
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
    *,
    times: list[float] | tuple[float, ...],
    scale: float,
    transition_observables: list[str] | tuple[str, ...],
    transition_bins: int,
    record_family_modulus: int,
    perturb_strength: float,
    repair_steps: int,
    repairs_per_step: int,
    seed: int,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for cap_index, cap in enumerate(caps):
        for time_index, time_value in enumerate(times):
            post_raw = _simulate_cap_collar_perturb_resettle(
                points,
                cap,
                raw_fields,
                graph,
                scale=float(scale),
                time_value=float(time_value),
                perturb_strength=float(perturb_strength),
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
            for observable, source_packets in base_packet_fields.items():
                target_packets = np.asarray(post_packets.get(observable, source_packets), dtype=np.int64)
                class_count = int(np.max(source_packets)) + 1 if source_packets.size else 1
                target_deltas = np.zeros((len(supports), class_count), dtype=float)
                changes = np.zeros(len(supports), dtype=float)
                for row_index, support in enumerate(supports):
                    delta, change = _transition_delta_same_support(
                        support,
                        np.asarray(source_packets, dtype=np.int64),
                        target_packets,
                        weights,
                        class_count,
                    )
                    target_deltas[row_index] = delta
                    changes[row_index] = change
                for class_index in range(class_count):
                    columns.append(target_deltas[:, class_index])
                columns.append(changes)
    return np.vstack(columns).T if columns else np.zeros((len(supports), 0), dtype=float)


def _simulate_cap_collar_perturb_resettle(
    points: np.ndarray,
    cap: RoundCap,
    raw_fields: dict[str, np.ndarray],
    graph: dict[str, np.ndarray | int],
    *,
    scale: float,
    time_value: float,
    perturb_strength: float,
    repair_steps: int,
    repairs_per_step: int,
    seed: int,
) -> dict[str, np.ndarray]:
    left = np.asarray(graph["left"], dtype=np.int64)
    right = np.asarray(graph["right"], dtype=np.int64)
    group_order = int(graph["group_order"])
    port_left = np.asarray(graph["port_left"], dtype=np.int16).copy()
    port_right = np.asarray(graph["port_right"], dtype=np.int16).copy()
    patch_count = int(graph["patch_count"])
    degree = np.asarray(graph["degree"], dtype=float)
    rng = np.random.default_rng(seed)
    selected_edges, affected_nodes = _cap_collar_edges(points, cap, left, right)
    if selected_edges.size:
        fraction = min(1.0, max(0.0, float(perturb_strength) * abs(float(time_value))))
        perturb_count = max(1, int(round(fraction * selected_edges.size)))
        perturb_count = min(perturb_count, selected_edges.size)
        chosen = _phase_selected_edges(points, cap, left, right, selected_edges, perturb_count, scale=float(scale), time_value=float(time_value), rng=rng)
        side = _perturb_side(points, cap, left[chosen], right[chosen])
        delta = np.int16(1 + (abs(seed) % max(1, group_order - 1)))
        port_left[chosen[side]] = ((port_left[chosen[side]].astype(np.int64) + delta) % group_order).astype(np.int16)
        port_right[chosen[~side]] = ((port_right[chosen[~side]].astype(np.int64) + delta) % group_order).astype(np.int16)
    repair_incident = np.zeros(patch_count, dtype=float)
    local_edge_mask = np.zeros(left.size, dtype=bool)
    if affected_nodes.size:
        affected = np.zeros(patch_count, dtype=bool)
        affected[affected_nodes] = True
        local_edge_mask = affected[left] | affected[right]
    for _step in range(max(0, int(repair_steps))):
        active = np.flatnonzero(local_edge_mask & (port_left != port_right))
        if active.size == 0:
            break
        count = min(int(repairs_per_step), active.size)
        chosen = rng.choice(active, size=count, replace=False)
        direction = rng.random(count) < 0.5
        port_left[chosen[direction]] = port_right[chosen[direction]]
        port_right[chosen[~direction]] = port_left[chosen[~direction]]
        repair_incident += np.bincount(left[chosen], minlength=patch_count) + np.bincount(right[chosen], minlength=patch_count)
    mismatch = port_left != port_right
    incident_mismatch = (
        np.bincount(left, weights=mismatch.astype(float), minlength=patch_count)
        + np.bincount(right, weights=mismatch.astype(float), minlength=patch_count)
    )
    signature = _local_node_signature(port_left, port_right, left, right, patch_count)
    baseline_signature = np.asarray(raw_fields.get("record_signature", signature), dtype=np.int64)
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


def _perturb_side(points: np.ndarray, cap: RoundCap, edge_left: np.ndarray, edge_right: np.ndarray) -> np.ndarray:
    dot_left = points[edge_left] @ cap.axis
    dot_right = points[edge_right] @ cap.axis
    return dot_left >= dot_right


def _phase_selected_edges(
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
) -> np.ndarray:
    if selected_edges.size <= int(count):
        return selected_edges
    cap = cap.normalized()
    tangent = cap.tangent / max(float(np.linalg.norm(cap.tangent)), 1e-12)
    bitangent = np.cross(cap.axis, tangent)
    bitangent = bitangent / max(float(np.linalg.norm(bitangent)), 1e-12)
    mids = points[left[selected_edges]] + points[right[selected_edges]]
    mids = mids / np.maximum(np.linalg.norm(mids, axis=1, keepdims=True), 1e-12)
    angle = np.arctan2(mids @ bitangent, mids @ tangent)
    score = np.sin(angle + float(scale) * float(time_value)) + 1e-9 * rng.normal(size=selected_edges.size)
    order = np.argsort(score)[-int(count) :]
    return selected_edges[order]


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


def _validated_graph_state(graph_state: dict[str, Any], patch_count: int) -> dict[str, np.ndarray | int] | None:
    required = ["left", "right", "port_left", "port_right", "group_order", "patch_count"]
    if any(key not in graph_state for key in required):
        return None
    left = np.asarray(graph_state["left"], dtype=np.int64)
    right = np.asarray(graph_state["right"], dtype=np.int64)
    port_left = np.asarray(graph_state["port_left"], dtype=np.int16)
    port_right = np.asarray(graph_state["port_right"], dtype=np.int16)
    if left.shape != right.shape or left.shape != port_left.shape or left.shape != port_right.shape:
        return None
    count = int(graph_state.get("patch_count", patch_count))
    degree = np.bincount(np.concatenate([left, right]), minlength=count).astype(float) if left.size else np.ones(count, dtype=float)
    degree = np.maximum(degree, 1.0)
    return {
        "left": left,
        "right": right,
        "port_left": port_left,
        "port_right": port_right,
        "group_order": int(graph_state["group_order"]),
        "patch_count": count,
        "degree": degree,
    }


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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cap_index, _cap in enumerate(caps):
        for time_index, time_value in enumerate(times):
            for observable, packets in packet_fields.items():
                class_count = int(np.max(packets)) + 1 if packets.size else 1
                for target_class in range(class_count):
                    rows.append(
                        {
                            "feature_index": len(rows),
                            "cap_index": int(cap_index),
                            "time_index": int(time_index),
                            "time": float(time_value),
                            "field": f"{observable}:to_{target_class}",
                            "observable": str(observable),
                            "target_class": int(target_class),
                            "feature_type": "target_distribution_delta",
                        }
                    )
                rows.append(
                    {
                        "feature_index": len(rows),
                        "cap_index": int(cap_index),
                        "time_index": int(time_index),
                        "time": float(time_value),
                        "field": f"{observable}:change",
                        "observable": str(observable),
                        "target_class": None,
                        "feature_type": "change_probability_delta",
                    }
                )
    return rows


def _object_transition_matrix(
    cache: GeometryCache,
    supports: list[np.ndarray],
    caps: list[RoundCap],
    times: list[float] | tuple[float, ...],
    packet_fields: dict[str, np.ndarray],
    weights: np.ndarray,
    *,
    scale: float,
    k_transport: int,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for cap_index, cap in enumerate(caps):
        for time_value in times:
            transported_supports = [
                _transported_support(cache, support, cap, float(scale) * float(time_value), k=int(k_transport), cap_id=cap_index)
                for support in supports
            ]
            for _observable, packets in packet_fields.items():
                class_count = int(np.max(packets)) + 1 if packets.size else 1
                target_deltas = np.zeros((len(supports), class_count), dtype=float)
                changes = np.zeros(len(supports), dtype=float)
                for row_index, (support, transported) in enumerate(zip(supports, transported_supports, strict=False)):
                    delta, change = _transition_delta_for_support(support, transported, packets, weights, class_count)
                    target_deltas[row_index] = delta
                    changes[row_index] = change
                for class_index in range(class_count):
                    columns.append(target_deltas[:, class_index])
                columns.append(changes)
    return np.vstack(columns).T if columns else np.zeros((len(supports), 0), dtype=float)


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
