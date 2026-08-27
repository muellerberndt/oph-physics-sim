"""Target-blind structural diagnostics for finite graph histories.

The routines in this module deliberately know nothing about CMB temperature,
laboratory length, cosmological time, energy, or matter density.  They consume
only a finite carrier graph and fields recorded on that graph.  Consequently,
all lengths below are graph hops or embedding chords and all times are supplied
run-cycle labels.  The report is useful for comparing simulator variants and
refinement levels; it cannot by itself identify a simulated field with a
physical early-universe observable.

The carrier graph is supplied once.  Its return probability, volume growth,
and spectral-dimension curve therefore describe that fixed carrier.  They are
not evidence that geometry evolves.  A time-dependent adjacency/metric would
be required for an evolving geometric dimension.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial import cKDTree


SCHEMA = "oph_structural_observables_v1"
CLASSIFICATION = "TARGET_BLIND_INTERNAL_DIAGNOSTIC_ONLY"


def structural_observables_report(
    points: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    cycles: np.ndarray,
    state_frames: np.ndarray,
    *,
    velocity_frames: np.ndarray | None = None,
    commit_frames: np.ndarray | None = None,
    defect_frames: np.ndarray | None = None,
    paired_delta_frames: np.ndarray | None = None,
    quench_values: np.ndarray | None = None,
    thresholds: Iterable[float] = (0.25, 0.50, 0.75),
    threshold_mode: str = "quantile",
    max_graph_hops: int = 8,
    anchor_count: int = 256,
    spectral_modes: int = 16,
    diffusion_steps: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 16),
    random_walk_probe_count: int = 16,
    correlation_null_draws: int = 4,
    seed: int = 17,
) -> dict[str, Any]:
    """Build a machine-readable structural report from generic graph arrays.

    ``state_frames``, ``velocity_frames``, and ``commit_frames`` have shape
    ``(frame, node)``.  ``defect_frames`` has shape ``(frame, edge)``.
    ``paired_delta_frames`` is the node-wise difference between paired runs;
    without a separately audited intervention contract it is reported only as
    a difference front, never as a causal speed.

    ``thresholds`` are quantiles by default.  Pass ``threshold_mode="value"``
    to use literal field values.  Optional inputs fail closed: their sections
    remain present with ``available=False`` and a precise missing-input reason.
    """

    if int(anchor_count) < 1:
        raise ValueError("anchor_count must be positive")
    if int(spectral_modes) < 1:
        raise ValueError("spectral_modes must be positive")
    if not 2 <= int(random_walk_probe_count) <= 256:
        raise ValueError("random_walk_probe_count must lie in [2, 256]")
    if not 2 <= int(correlation_null_draws) <= 64:
        raise ValueError("correlation_null_draws must lie in [2, 64]")
    if not 1 <= int(max_graph_hops) <= 64:
        raise ValueError("max_graph_hops must lie in [1, 64]")
    effective_spectral_modes = min(int(spectral_modes), 64)

    geometry = _validated_inputs(
        points,
        edge_left,
        edge_right,
        cycles,
        state_frames,
        velocity_frames=velocity_frames,
        commit_frames=commit_frames,
        defect_frames=defect_frames,
        paired_delta_frames=paired_delta_frames,
        quench_values=quench_values,
    )
    points = geometry["points"]
    left = geometry["left"]
    right = geometry["right"]
    cycles = geometry["cycles"]
    state = geometry["state"]
    velocity = geometry["velocity"]
    commits = geometry["commits"]
    defects = geometry["defects"]
    paired_delta = geometry["paired_delta"]
    quench = geometry["quench"]

    node_count = int(points.shape[0])
    edge_count = int(left.size)
    frame_count = int(cycles.size)
    rng = np.random.default_rng(int(seed))
    adjacency = _adjacency_lists(node_count, left, right)
    frame_indices = _sample_indices(frame_count, 64)

    carrier = _carrier_report(
        points,
        left,
        right,
        adjacency,
        diffusion_steps=tuple(int(value) for value in diffusion_steps),
        anchor_count=min(int(anchor_count), node_count),
        random_walk_probe_count=int(random_walk_probe_count),
        rng=rng,
    )
    correlation = _correlation_horizon_report(
        state,
        cycles,
        adjacency,
        frame_indices=frame_indices,
        max_hops=int(max_graph_hops),
        anchor_count=min(int(anchor_count), node_count),
        null_draws=int(correlation_null_draws),
        rng=rng,
    )
    intervention = _paired_difference_front_report(
        paired_delta,
        cycles,
        adjacency,
    )
    phase = _phase_transition_report(state, cycles, quench)
    morphology = _morphology_topology_report(
        state,
        cycles,
        left,
        right,
        adjacency,
        thresholds=tuple(float(value) for value in thresholds),
        threshold_mode=str(threshold_mode),
        frame_indices=frame_indices,
    )
    defect_report = _defect_dynamics_report(
        defects,
        cycles,
        points,
        left,
        right,
    )
    field_statistics = _field_statistics_report(
        state,
        cycles,
        left,
        right,
        frame_indices=frame_indices,
        spectral_modes=effective_spectral_modes,
    )
    symmetry = _symmetry_report(points, state, cycles, frame_indices)
    velocity_report = _velocity_report(state, velocity, cycles)
    observer_records = _commit_report(commits, cycles)

    report = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "input_summary": {
            "node_count": node_count,
            "edge_count": edge_count,
            "frame_count": frame_count,
            "cycle_min": float(cycles[0]),
            "cycle_max": float(cycles[-1]),
            "sampled_frame_indices": [int(value) for value in frame_indices],
            "sampling": {
                "analysis_seed": int(seed),
                "rng": "numpy.default_rng_PCG64",
                "rng_consumption_order": [
                    "carrier_return_probability_probes",
                    "carrier_volume_growth_anchors",
                    "field_correlation_anchors",
                    "field_correlation_permutation_nulls",
                ],
                "frame_method": "all_frames_if_at_most_64_else_evenly_spaced_with_endpoints",
                "sampled_frame_count": int(frame_indices.size),
                "all_frame_count": frame_count,
                "requested_anchor_limit": int(anchor_count),
                "requested_spectral_mode_limit": int(spectral_modes),
                "effective_spectral_mode_limit": int(effective_spectral_modes),
                "spectral_mode_safety_cap": 64,
                "random_walk_probe_count": int(random_walk_probe_count),
                "correlation_permutation_null_draws_per_frame": int(
                    correlation_null_draws
                ),
                "all_nodes_used_for_spatial_moments": True,
                "all_frames_used_for_phase_commit_velocity_and_defect_time_series": True,
            },
            "optional_inputs_present": {
                "velocity_frames": velocity is not None,
                "commit_frames": commits is not None,
                "defect_frames": defects is not None,
                "paired_delta_frames": paired_delta is not None,
                "quench_values": quench is not None,
            },
        },
        "carrier_geometry": carrier,
        "field_correlation_horizon": correlation,
        "paired_difference_front": intervention,
        "phase_transition_proxies": phase,
        "morphology_and_graph_topology": morphology,
        "defect_dynamics": defect_report,
        "graph_field_statistics": field_statistics,
        "symmetry_diagnostics": symmetry,
        "scalar_velocity_diagnostics": velocity_report,
        "observer_record_diagnostics": observer_records,
        "epistemic_gates": {
            "target_data_read": False,
            "public_measurement_comparison": False,
            "source_to_physical_observable_map_supplied": False,
            "graph_hops_identified_with_physical_length": False,
            "run_cycles_identified_with_physical_time": False,
            "state_identified_with_physical_density_or_temperature": False,
            "fixed_carrier_is_dynamic_spacetime": False,
            "paired_difference_is_causal_response": False,
            "physical_early_universe_claim": False,
        },
        "nonclaims": [
            "Carrier dimension diagnostics describe the supplied fixed graph.",
            "Field correlations and difference fronts use graph hops and run cycles only.",
            "Spatial moments from one run are not ensemble thermodynamic observables.",
            "No reported field is identified with temperature, density, matter, or curvature.",
            "No public target or fitted physical parameter is consumed by this API.",
        ],
    }
    return _jsonable(report)


def write_structural_observables_report(
    out_dir: str | Path,
    points: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    cycles: np.ndarray,
    state_frames: np.ndarray,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write the canonical JSON report and lossless input arrays for later use."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = structural_observables_report(
        points,
        edge_left,
        edge_right,
        cycles,
        state_frames,
        **kwargs,
    )
    report_path = out / "structural_observables_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    arrays: dict[str, np.ndarray] = {
        "points": np.asarray(points),
        "edge_left": np.asarray(edge_left),
        "edge_right": np.asarray(edge_right),
        "cycles": np.asarray(cycles),
        "state_frames": np.asarray(state_frames),
    }
    for name in (
        "velocity_frames",
        "commit_frames",
        "defect_frames",
        "paired_delta_frames",
        "quench_values",
    ):
        value = kwargs.get(name)
        if value is not None:
            arrays[name] = np.asarray(value)
    np.savez_compressed(out / "structural_observables_inputs.npz", **arrays)

    input_path = out / "structural_observables_inputs.npz"

    manifest = {
        "schema": "oph_structural_observables_artifacts_v1",
        "classification": CLASSIFICATION,
        "artifacts": [
            _artifact_receipt(report_path),
            _artifact_receipt(input_path),
        ],
        "target_data_included": False,
        "physical_identification_included": False,
    }
    (out / "structural_observables_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _artifact_receipt(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def _validated_inputs(
    points: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    cycles: np.ndarray,
    state_frames: np.ndarray,
    *,
    velocity_frames: np.ndarray | None,
    commit_frames: np.ndarray | None,
    defect_frames: np.ndarray | None,
    paired_delta_frames: np.ndarray | None,
    quench_values: np.ndarray | None,
) -> dict[str, Any]:
    points_array = np.asarray(points, dtype=float)
    left = np.asarray(edge_left, dtype=np.int64).reshape(-1)
    right = np.asarray(edge_right, dtype=np.int64).reshape(-1)
    cycle_array = np.asarray(cycles, dtype=float).reshape(-1)
    state = np.asarray(state_frames, dtype=float)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError("points must have shape (node, 3)")
    node_count = int(points_array.shape[0])
    if node_count < 2:
        raise ValueError("at least two nodes are required")
    if left.shape != right.shape or left.size == 0:
        raise ValueError("edge_left and edge_right must be nonempty equal-length arrays")
    if np.any(left < 0) or np.any(right < 0) or np.any(left >= node_count) or np.any(right >= node_count):
        raise ValueError("edge endpoints are outside the node range")
    if np.any(left == right):
        raise ValueError("self-loop edges are not supported")
    undirected = np.sort(np.column_stack([left, right]), axis=1)
    if np.unique(undirected, axis=0).shape[0] != left.size:
        raise ValueError("duplicate undirected edges are not supported")
    if cycle_array.size < 1 or np.any(np.diff(cycle_array) <= 0.0):
        raise ValueError("cycles must be a nonempty strictly increasing array")
    expected_node_shape = (cycle_array.size, node_count)
    if state.shape != expected_node_shape:
        raise ValueError(f"state_frames has shape {state.shape}, expected {expected_node_shape}")
    if not np.all(np.isfinite(points_array)) or not np.all(np.isfinite(cycle_array)) or not np.all(np.isfinite(state)):
        raise ValueError("points, cycles, and state_frames must be finite")

    def optional_node_frames(name: str, values: np.ndarray | None) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=float)
        if array.shape != expected_node_shape:
            raise ValueError(f"{name} has shape {array.shape}, expected {expected_node_shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        return array

    velocity = optional_node_frames("velocity_frames", velocity_frames)
    commits = optional_node_frames("commit_frames", commit_frames)
    paired = optional_node_frames("paired_delta_frames", paired_delta_frames)
    defects = None
    if defect_frames is not None:
        defects = np.asarray(defect_frames, dtype=float)
        expected_edge_shape = (cycle_array.size, left.size)
        if defects.shape != expected_edge_shape:
            raise ValueError(
                f"defect_frames has shape {defects.shape}, expected {expected_edge_shape}"
            )
        if not np.all(np.isfinite(defects)):
            raise ValueError("defect_frames must be finite")
    quench = None
    if quench_values is not None:
        quench = np.asarray(quench_values, dtype=float).reshape(-1)
        if quench.shape != cycle_array.shape:
            raise ValueError(
                f"quench_values has shape {quench.shape}, expected {cycle_array.shape}"
            )
        if not np.all(np.isfinite(quench)):
            raise ValueError("quench_values must be finite")
    return {
        "points": points_array,
        "left": left,
        "right": right,
        "cycles": cycle_array,
        "state": state,
        "velocity": velocity,
        "commits": commits,
        "defects": defects,
        "paired_delta": paired,
        "quench": quench,
    }


def _adjacency_lists(node_count: int, left: np.ndarray, right: np.ndarray) -> list[np.ndarray]:
    rows: list[list[int]] = [[] for _ in range(node_count)]
    for u, v in zip(left.tolist(), right.tolist()):
        rows[u].append(v)
        rows[v].append(u)
    return [np.asarray(sorted(set(row)), dtype=np.int64) for row in rows]


def _sample_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, count - 1, limit)).astype(np.int64))


def _component_labels(adjacency: list[np.ndarray], active: np.ndarray | None = None) -> tuple[np.ndarray, list[int]]:
    node_count = len(adjacency)
    mask = np.ones(node_count, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    labels = np.full(node_count, -1, dtype=np.int64)
    sizes: list[int] = []
    for start in np.flatnonzero(mask):
        if labels[start] >= 0:
            continue
        label = len(sizes)
        queue: deque[int] = deque([int(start)])
        labels[start] = label
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for neighbor in adjacency[node]:
                other = int(neighbor)
                if mask[other] and labels[other] < 0:
                    labels[other] = label
                    queue.append(other)
        sizes.append(size)
    return labels, sizes


def _carrier_report(
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    adjacency: list[np.ndarray],
    *,
    diffusion_steps: tuple[int, ...],
    anchor_count: int,
    random_walk_probe_count: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    node_count = len(adjacency)
    degree = np.asarray([row.size for row in adjacency], dtype=float)
    _, component_sizes = _component_labels(adjacency)
    betti_zero = len(component_sizes)
    betti_one = int(left.size - node_count + betti_zero)
    if any(value < 1 or value > 256 for value in diffusion_steps):
        raise ValueError("diffusion_steps must lie in [1, 256]")
    steps = sorted(set(diffusion_steps))
    if not steps:
        raise ValueError("at least one diffusion step is required")
    return_probability, return_standard_error = _return_probability(
        adjacency,
        steps,
        rng,
        probe_count=random_walk_probe_count,
    )
    spectral_curve = _spectral_dimension_curve(steps, return_probability)
    volume = _volume_growth(adjacency, anchor_count, max(steps, default=8), rng)
    return {
        "available": True,
        "classification": "FIXED_CARRIER_GRAPH_DIAGNOSTIC",
        "carrier_is_time_dependent": False,
        "dynamic_geometry_inference_available": False,
        "dynamic_geometry_inference_reason": "one adjacency is supplied for all frames",
        "causal_dimension": {
            "available": False,
            "reason": (
                "undirected adjacency and frame labels do not supply a directed causal order or interval-counting ensemble"
            ),
        },
        "connected_component_count": int(betti_zero),
        "component_sizes": [int(value) for value in sorted(component_sizes, reverse=True)],
        "degree": {
            "minimum": int(np.min(degree)),
            "maximum": int(np.max(degree)),
            "mean": float(np.mean(degree)),
        },
        "graph_euler_characteristic": int(node_count - left.size),
        "graph_betti_0": int(betti_zero),
        "graph_betti_1_cycle_rank": int(betti_one),
        "lazy_random_walk_return_probability": [
            {
                "step": int(step),
                "probability": float(probability),
                "standard_error": (
                    float(standard_error)
                    if standard_error is not None
                    else None
                ),
            }
            for step, probability, standard_error in zip(
                steps, return_probability, return_standard_error
            )
        ],
        "return_probability_estimator": {
            "method": (
                "exact_lazy_normalized_adjacency_eigenspectrum"
                if node_count <= 256
                else "seeded_rademacher_hutchinson_trace"
            ),
            "probe_count": None if node_count <= 256 else random_walk_probe_count,
            "exact_trace": bool(node_count <= 256),
            "probe_vectors_have_one_entry_per_node": bool(node_count > 256),
            "standard_error_method": (
                None
                if node_count <= 256
                else "sample_standard_error_across_rademacher_quadratic_forms"
            ),
        },
        "spectral_dimension_curve": spectral_curve,
        "spectral_dimension_caveat": (
            "finite-carrier lazy-walk log-slope; endpoints and saturation are not "
            "a continuum dimension, and return-probability uncertainty is not "
            "propagated through the nonlinear slope"
        ),
        "volume_growth": volume,
        "intrinsic_curvature": {
            "available": False,
            "reason": (
                "an edge list and embedding coordinates do not supply an intrinsic metric, oriented faces, or a discrete connection"
            ),
            "embedding_shape_used_as_intrinsic_curvature": False,
        },
        "embedding_role": "coordinates used only for embedding diagnostics, not laboratory distance",
        "embedding_radius": {
            "minimum": float(np.min(np.linalg.norm(points, axis=1))),
            "maximum": float(np.max(np.linalg.norm(points, axis=1))),
        },
    }


def _sparse_normalized_adjacency(adjacency: list[np.ndarray]) -> sparse.csr_matrix:
    node_count = len(adjacency)
    row: list[int] = []
    col: list[int] = []
    for node, neighbors in enumerate(adjacency):
        row.extend([node] * int(neighbors.size))
        col.extend(int(value) for value in neighbors)
    data = np.ones(len(row), dtype=float)
    matrix = sparse.csr_matrix((data, (row, col)), shape=(node_count, node_count))
    degree = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 0.0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized = sparse.diags(inv_sqrt) @ matrix @ sparse.diags(inv_sqrt)
    isolated = np.flatnonzero(~positive)
    if isolated.size:
        normalized = normalized + sparse.csr_matrix(
            (np.ones(isolated.size), (isolated, isolated)),
            shape=(node_count, node_count),
        )
    return normalized


def _return_probability(
    adjacency: list[np.ndarray],
    steps: list[int],
    rng: np.random.Generator,
    *,
    probe_count: int,
) -> tuple[np.ndarray, list[float | None]]:
    if not steps:
        return np.asarray([], dtype=float), []
    normalized = _sparse_normalized_adjacency(adjacency)
    node_count = normalized.shape[0]
    lazy = 0.5 * (sparse.eye(node_count, format="csr") + normalized)
    if node_count <= 256:
        eigenvalues = np.linalg.eigvalsh(lazy.toarray())
        exact = np.asarray(
            [np.mean(eigenvalues**step) for step in steps], dtype=float
        )
        return exact, [None] * len(steps)
    samples = np.zeros((probe_count, len(steps)), dtype=float)
    step_to_index = {step: index for index, step in enumerate(steps)}
    for probe_index in range(probe_count):
        probe = rng.choice(np.asarray([-1.0, 1.0]), size=node_count)
        evolved = probe.copy()
        for step in range(1, max(steps) + 1):
            evolved = lazy @ evolved
            if step in step_to_index:
                samples[probe_index, step_to_index[step]] = (
                    float(probe @ evolved) / node_count
                )
    estimates = np.mean(samples, axis=0)
    standard_errors = np.std(samples, axis=0, ddof=1) / math.sqrt(probe_count)
    return estimates, [float(value) for value in standard_errors]


def _spectral_dimension_curve(steps: list[int], probabilities: np.ndarray) -> list[dict[str, Any]]:
    if len(steps) < 2:
        return []
    step_values = np.asarray(steps, dtype=float)
    safe = np.maximum(np.asarray(probabilities, dtype=float), np.finfo(float).tiny)
    slopes = -2.0 * np.gradient(np.log(safe), np.log(step_values))
    return [
        {
            "step": int(step),
            "return_probability": float(probability),
            "effective_spectral_dimension": float(slope),
        }
        for step, probability, slope in zip(steps, safe, slopes)
    ]


def _bfs_distances(adjacency: list[np.ndarray], sources: Sequence[int], max_hops: int | None = None) -> np.ndarray:
    distance = np.full(len(adjacency), -1, dtype=np.int32)
    queue: deque[int] = deque()
    for source in sources:
        node = int(source)
        if distance[node] < 0:
            distance[node] = 0
            queue.append(node)
    while queue:
        node = queue.popleft()
        if max_hops is not None and distance[node] >= max_hops:
            continue
        for neighbor in adjacency[node]:
            other = int(neighbor)
            if distance[other] < 0:
                distance[other] = distance[node] + 1
                queue.append(other)
    return distance


def _volume_growth(
    adjacency: list[np.ndarray],
    anchor_count: int,
    max_radius: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    node_count = len(adjacency)
    anchors = np.arange(node_count) if anchor_count >= node_count else rng.choice(node_count, anchor_count, replace=False)
    max_radius = min(max(1, int(max_radius)), 32)
    balls = np.zeros((len(anchors), max_radius + 1), dtype=float)
    for index, anchor in enumerate(anchors):
        distances = _bfs_distances(adjacency, [int(anchor)], max_radius)
        for radius in range(max_radius + 1):
            balls[index, radius] = np.count_nonzero((distances >= 0) & (distances <= radius))
    mean_ball = np.mean(balls, axis=0)
    radii = np.arange(max_radius + 1, dtype=float)
    fit_mask = (radii >= 1.0) & (mean_ball > 1.0) & (mean_ball < 0.8 * node_count)
    exponent = None
    if np.count_nonzero(fit_mask) >= 2:
        exponent = float(np.polyfit(np.log(radii[fit_mask]), np.log(mean_ball[fit_mask]), 1)[0])
    return {
        "anchor_count": int(len(anchors)),
        "anchor_selection": "all_nodes_or_seeded_uniform_without_replacement",
        "anchor_node_indices": [int(value) for value in anchors],
        "rows": [
            {
                "radius_hops": int(radius),
                "mean_ball_nodes": float(mean_ball[radius]),
                "minimum_ball_nodes": int(np.min(balls[:, radius])),
                "maximum_ball_nodes": int(np.max(balls[:, radius])),
            }
            for radius in range(max_radius + 1)
        ],
        "log_log_growth_exponent_proxy": exponent,
        "units": "graph_hops_and_nodes",
    }


def _standardized(values: np.ndarray) -> np.ndarray | None:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    scale = float(np.sqrt(np.mean(centered * centered)))
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    return centered / scale


def _correlation_horizon_report(
    state: np.ndarray,
    cycles: np.ndarray,
    adjacency: list[np.ndarray],
    *,
    frame_indices: np.ndarray,
    max_hops: int,
    anchor_count: int,
    null_draws: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if max_hops < 1:
        raise ValueError("max_graph_hops must be at least one")
    node_count = state.shape[1]
    anchors = np.arange(node_count) if anchor_count >= node_count else np.sort(rng.choice(node_count, anchor_count, replace=False))
    shells: list[list[np.ndarray]] = []
    for anchor in anchors:
        distance = _bfs_distances(adjacency, [int(anchor)], max_hops)
        shells.append([np.flatnonzero(distance == hop) for hop in range(max_hops + 1)])
    rows: list[dict[str, Any]] = []
    for frame in frame_indices:
        field = _standardized(state[int(frame)])
        if field is None:
            rows.append(
                {
                    "frame_index": int(frame),
                    "cycle": float(cycles[int(frame)]),
                    "available": False,
                    "reason": "constant_state_frame",
                }
            )
            continue
        observed = _anchor_shell_statistics(field, anchors, shells, max_hops)
        controls = np.full((null_draws, max_hops + 1), np.nan, dtype=float)
        for draw in range(null_draws):
            permuted = field[rng.permutation(node_count)]
            controls[draw] = _anchor_shell_statistics(
                permuted, anchors, shells, max_hops
            )["correlations"]
        shell_rows: list[dict[str, Any]] = []
        for hop in range(max_hops + 1):
            if int(observed["anchor_counts"][hop]) == 0:
                shell_rows.append(
                    {
                        "hop": int(hop),
                        "available": False,
                        "reason": "no_anchor_has_a_node_at_this_graph_hop",
                        "pair_count": 0,
                        "anchor_count_with_nonempty_shell": 0,
                    }
                )
                continue
            correlation = float(observed["correlations"][hop])
            anchor_standard_error = float(observed["standard_errors"][hop])
            null_mean = float(np.mean(controls[:, hop]))
            null_standard_deviation = float(np.std(controls[:, hop], ddof=1))
            null_excess = correlation - null_mean
            threshold = float(
                max(
                    0.05,
                    2.0 * anchor_standard_error,
                    2.0 * null_standard_deviation,
                )
            )
            shell_rows.append(
                {
                    "hop": int(hop),
                    "available": True,
                    "pair_count": int(observed["pair_counts"][hop]),
                    "anchor_count_with_nonempty_shell": int(
                        observed["anchor_counts"][hop]
                    ),
                    "estimator": "equal_weight_mean_of_anchor_shell_means",
                    "correlation": correlation,
                    "anchor_block_standard_error": anchor_standard_error,
                    "permuted_field_null_mean": null_mean,
                    "permuted_field_null_standard_deviation": null_standard_deviation,
                    "correlation_minus_null_mean": null_excess,
                    "descriptive_threshold": threshold,
                    "threshold_crossing": bool(
                        hop > 0 and abs(null_excess) >= threshold
                    ),
                }
            )
        crossings = [
            row["hop"]
            for row in shell_rows[1:]
            if row.get("threshold_crossing") is True
        ]
        excess_rows = [
            (
                row["hop"],
                max(
                    abs(row["correlation_minus_null_mean"])
                    - row["descriptive_threshold"],
                    0.0,
                ),
            )
            for row in shell_rows[1:]
            if row.get("available") is True
        ]
        denominator = sum(value for _, value in excess_rows)
        weighted_hop = (
            float(sum(hop * value for hop, value in excess_rows) / denominator)
            if denominator > 0.0
            else None
        )
        rows.append(
            {
                "frame_index": int(frame),
                "cycle": float(cycles[int(frame)]),
                "available": True,
                "null_and_threshold_excess_weighted_hop_heuristic": weighted_hop,
                "largest_null_excess_threshold_crossing_hop": max(
                    crossings, default=0
                ),
                "shells": shell_rows,
            }
        )
    return {
        "available": any(row["available"] for row in rows),
        "classification": "FIELD_ON_FIXED_GRAPH",
        "anchor_count": int(len(anchors)),
        "anchor_selection": "all_nodes_or_seeded_uniform_without_replacement",
        "anchor_node_indices": [int(value) for value in anchors],
        "sampled_frame_indices": [int(value) for value in frame_indices],
        "permuted_field_null": {
            "draw_count_per_frame": int(null_draws),
            "method": (
                "seeded uniform node-label permutations preserve the one-point "
                "field values and destroy their graph assignment"
            ),
        },
        "extent_heuristic_definition": (
            "largest sampled positive hop whose absolute correlation-minus-null "
            "exceeds max(0.05, twice the anchor-block standard error, twice the "
            "permutation-null standard deviation)"
        ),
        "formal_significance_claim": False,
        "correlation_length_estimate_available": False,
        "correlation_length_estimate_reason": (
            "a calibrated ensemble and a declared correlation-length estimator are not supplied"
        ),
        "rows": rows,
        "physical_length_claim": False,
        "causal_horizon_claim": False,
    }


def _anchor_shell_statistics(
    field: np.ndarray,
    anchors: np.ndarray,
    shells: list[list[np.ndarray]],
    max_hops: int,
) -> dict[str, np.ndarray]:
    """Return equal-anchor shell estimates and anchor-block standard errors.

    Pair products sharing an anchor are not independent.  Reducing each
    anchor's shell to one mean before estimating uncertainty prevents the raw
    pair count from being used as a fictitious iid sample size.  Different
    anchors can still overlap, so the reported standard error remains a
    descriptive block estimate rather than a formal sampling error.
    """

    correlations = np.full(max_hops + 1, np.nan, dtype=float)
    standard_errors = np.zeros(max_hops + 1, dtype=float)
    pair_counts = np.zeros(max_hops + 1, dtype=np.int64)
    anchor_counts = np.zeros(max_hops + 1, dtype=np.int64)
    for hop in range(max_hops + 1):
        anchor_means: list[float] = []
        for anchor_index, anchor in enumerate(anchors):
            members = shells[anchor_index][hop]
            if members.size:
                anchor_means.append(
                    float(field[int(anchor)] * np.mean(field[members]))
                )
                pair_counts[hop] += int(members.size)
        values = np.asarray(anchor_means, dtype=float)
        anchor_counts[hop] = int(values.size)
        if values.size:
            correlations[hop] = float(np.mean(values))
        if values.size >= 2:
            standard_errors[hop] = float(
                np.std(values, ddof=1) / math.sqrt(values.size)
            )
    return {
        "correlations": correlations,
        "standard_errors": standard_errors,
        "pair_counts": pair_counts,
        "anchor_counts": anchor_counts,
    }


def _paired_difference_front_report(
    paired_delta: np.ndarray | None,
    cycles: np.ndarray,
    adjacency: list[np.ndarray],
) -> dict[str, Any]:
    if paired_delta is None:
        return {
            "available": False,
            "reason": "paired_delta_frames_not_supplied",
            "causal_intervention_claim": False,
        }
    amplitude = np.abs(paired_delta)
    nonzero_frames = np.flatnonzero(np.max(amplitude, axis=1) > 0.0)
    if nonzero_frames.size == 0:
        return {
            "available": False,
            "reason": "paired_delta_frames_are_identically_zero",
            "causal_intervention_claim": False,
        }
    first = int(nonzero_frames[0])
    first_amplitude = amplitude[first]
    source_cut = max(float(np.max(first_amplitude)) * 1.0e-6, np.finfo(float).tiny)
    sources = np.flatnonzero(first_amplitude >= source_cut)
    distance = _bfs_distances(adjacency, sources.tolist())
    global_max = float(np.max(amplitude))
    threshold = max(global_max * 1.0e-6, np.finfo(float).tiny)
    rows: list[dict[str, Any]] = []
    first_arrival = np.full(amplitude.shape[1], -1, dtype=np.int64)
    for frame in range(amplitude.shape[0]):
        active = amplitude[frame] >= threshold
        new = active & (first_arrival < 0)
        first_arrival[new] = frame
        reached = distance[active & (distance >= 0)]
        rows.append(
            {
                "frame_index": int(frame),
                "cycle": float(cycles[frame]),
                "affected_node_count": int(np.count_nonzero(active)),
                "maximum_reached_distance_hops": int(np.max(reached)) if reached.size else None,
                "mean_reached_distance_hops": float(np.mean(reached)) if reached.size else None,
                "delta_l1": float(np.sum(amplitude[frame])),
                "delta_l2": float(np.linalg.norm(amplitude[frame])),
            }
        )
    arrived = np.flatnonzero(first_arrival >= 0)
    arrival_counts: dict[tuple[int, int], int] = {}
    for node in arrived:
        key = (int(distance[node]), int(first_arrival[node]))
        arrival_counts[key] = arrival_counts.get(key, 0) + 1
    arrival_rows = [
        {
            "distance_hops": distance_hops,
            "first_arrival_frame": frame,
            "first_arrival_cycle": float(cycles[frame]),
            "node_count": count,
        }
        for (distance_hops, frame), count in sorted(arrival_counts.items())
    ]
    return {
        "available": True,
        "classification": "PAIRED_DIFFERENCE_ONLY",
        "source_frame_index": first,
        "source_cycle": float(cycles[first]),
        "source_nodes": [int(value) for value in sources],
        "activity_threshold": float(threshold),
        "rows": rows,
        "first_arrival_histogram": arrival_rows,
        "first_arrival_node_count": int(arrived.size),
        "lossless_node_deltas_retained_in_input_artifact": True,
        "causal_intervention_contract_supplied": False,
        "causal_intervention_claim": False,
        "caveat": (
            "A causal front requires same-seed pre-intervention equality and proof that a localized intervention is the sole run difference."
        ),
    }


def _phase_transition_report(
    state: np.ndarray,
    cycles: np.ndarray,
    quench: np.ndarray | None,
) -> dict[str, Any]:
    node_count = state.shape[1]
    rows: list[dict[str, Any]] = []
    for frame, values in enumerate(state):
        mean = float(np.mean(values))
        centered = values - mean
        second = float(np.mean(centered**2))
        fourth = float(np.mean(centered**4))
        binder = None if second <= 0.0 else float(1.0 - fourth / (3.0 * second * second))
        rows.append(
            {
                "frame_index": int(frame),
                "cycle": float(cycles[frame]),
                "spatial_state_mean": mean,
                "spatial_site_variance": second,
                "node_count_times_spatial_site_variance": float(
                    node_count * second
                ),
                "spatial_fluctuation_shape_u4": binder,
            }
        )
    order = np.mean(state, axis=1)
    autocorrelation = _autocorrelation_report(order, cycles)
    quench_report: dict[str, Any]
    if quench is None:
        quench_report = {
            "available": False,
            "reason": "quench_values_not_supplied",
            "critical_exponent_fit_available": False,
        }
    else:
        heterogeneity = np.asarray(
            [row["node_count_times_spatial_site_variance"] for row in rows]
        )
        correlation = _pearson(quench, heterogeneity)
        quench_report = {
            "available": True,
            "rows": [
                {
                    "frame_index": int(index),
                    "cycle": float(cycles[index]),
                    "quench_value": float(quench[index]),
                    "node_count_times_spatial_site_variance": float(
                        heterogeneity[index]
                    ),
                }
                for index in range(cycles.size)
            ],
            "quench_spatial_heterogeneity_correlation": correlation,
            "critical_exponent_fit_available": False,
            "critical_exponent_fit_reason": (
                "independent ensembles at several sizes and quench rates are required"
            ),
        }
    return {
        "available": True,
        "classification": "SINGLE_TRAJECTORY_SPATIAL_HETEROGENEITY",
        "rows": rows,
        "spatial_fluctuation_shape_definition": (
            "1 - spatial_fourth_central_moment / (3 * spatial_variance^2)"
        ),
        "spatial_fluctuation_shape_is_binder_cumulant": False,
        "susceptibility": {
            "available": False,
            "reason": (
                "susceptibility requires fluctuations of a global order parameter "
                "across an ensemble at fixed control parameters; node_count times "
                "one frame's spatial site variance is not that estimator"
            ),
        },
        "binder_cumulant": {
            "available": False,
            "reason": (
                "a Binder cumulant requires ensemble moments of the declared global "
                "order parameter; the reported u4 value is only a centered spatial "
                "shape statistic"
            ),
        },
        "spatial_mean_trajectory_autocorrelation": autocorrelation,
        "stationary_mixing_time_claim": False,
        "dynamic_exponent": {
            "available": False,
            "reason": "one system size and one trajectory cannot identify a dynamic critical exponent",
        },
        "quench_scaling_inputs": quench_report,
        "thermodynamic_claim": False,
    }


def _autocorrelation_report(values: np.ndarray, cycles: np.ndarray) -> dict[str, Any]:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    variance = float(np.dot(centered, centered))
    if centered.size < 4 or variance <= 0.0:
        return {"available": False, "reason": "too_few_or_constant_global_order_samples"}
    max_lag = min(centered.size - 1, max(1, centered.size // 2))
    rows: list[dict[str, Any]] = []
    positive_sum = 0.0
    for lag in range(max_lag + 1):
        value = float(np.dot(centered[: centered.size - lag], centered[lag:]) / variance)
        rows.append(
            {
                "lag_frames": int(lag),
                "mean_cycle_separation": float(np.mean(cycles[lag:] - cycles[: cycles.size - lag])) if lag else 0.0,
                "autocorrelation": value,
            }
        )
        if lag > 0 and value > 0.0:
            positive_sum += value
        elif lag > 0:
            break
    return {
        "available": True,
        "rows": rows,
        "integrated_autocorrelation_time_frames_positive_sequence": float(0.5 + positive_sum),
    }


def _morphology_topology_report(
    state: np.ndarray,
    cycles: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    adjacency: list[np.ndarray],
    *,
    thresholds: tuple[float, ...],
    threshold_mode: str,
    frame_indices: np.ndarray,
) -> dict[str, Any]:
    if threshold_mode not in {"quantile", "value"}:
        raise ValueError("threshold_mode must be 'quantile' or 'value'")
    if not thresholds:
        raise ValueError("at least one morphology threshold is required")
    if not all(math.isfinite(value) for value in thresholds):
        raise ValueError("morphology thresholds must be finite")
    if threshold_mode == "quantile" and any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("quantile thresholds must lie in [0, 1]")
    rows: list[dict[str, Any]] = []
    node_count = state.shape[1]
    for frame in frame_indices:
        values = state[int(frame)]
        for requested in thresholds:
            threshold = float(np.quantile(values, requested)) if threshold_mode == "quantile" else requested
            active = values >= threshold
            inactive = ~active
            _, active_sizes = _component_labels(adjacency, active)
            _, void_sizes = _component_labels(adjacency, inactive)
            active_edges = int(np.count_nonzero(active[left] & active[right]))
            inactive_edges = int(np.count_nonzero(inactive[left] & inactive[right]))
            active_nodes = int(np.count_nonzero(active))
            inactive_nodes = node_count - active_nodes
            b0 = len(active_sizes)
            b1 = active_edges - active_nodes + b0
            void_b0 = len(void_sizes)
            void_b1 = inactive_edges - inactive_nodes + void_b0
            rows.append(
                {
                    "frame_index": int(frame),
                    "cycle": float(cycles[int(frame)]),
                    "requested_threshold": float(requested),
                    "threshold_value": threshold,
                    "active_fraction": float(active_nodes / node_count),
                    "active_component_count": int(b0),
                    "largest_active_component_nodes": max(active_sizes, default=0),
                    "largest_active_component_fraction_of_all_nodes": float(max(active_sizes, default=0) / node_count),
                    "active_graph_euler_characteristic": int(active_nodes - active_edges),
                    "active_graph_betti_0": int(b0),
                    "active_graph_betti_1_cycle_rank": int(b1),
                    "void_component_count": int(void_b0),
                    "largest_void_component_nodes": max(void_sizes, default=0),
                    "void_graph_euler_characteristic": int(inactive_nodes - inactive_edges),
                    "void_graph_betti_0": int(void_b0),
                    "void_graph_betti_1_cycle_rank": int(void_b1),
                    "giant_component_proxy": bool(max(active_sizes, default=0) >= 0.5 * max(active_nodes, 1)),
                }
            )
    persistence_frames = sorted(set([int(frame_indices[0]), int(frame_indices[len(frame_indices) // 2]), int(frame_indices[-1])]))
    persistence = [
        {
            "frame_index": frame,
            "cycle": float(cycles[frame]),
            **_zero_dimensional_persistence(state[frame], adjacency),
        }
        for frame in persistence_frames
    ]
    return {
        "available": True,
        "threshold_mode": threshold_mode,
        "thresholds": [float(value) for value in thresholds],
        "threshold_rows": rows,
        "sampled_frame_indices": [int(value) for value in frame_indices],
        "zero_dimensional_superlevel_persistence": persistence,
        "topology_scope": "induced_graph_1_complex_only",
        "higher_betti_numbers_available": False,
        "higher_betti_numbers_reason": "faces or a higher-dimensional cell complex were not supplied",
        "physical_percolation_claim": False,
    }


def _zero_dimensional_persistence(values: np.ndarray, adjacency: list[np.ndarray]) -> dict[str, Any]:
    node_count = values.size
    order = np.lexsort((np.arange(node_count), -values))
    parent = np.arange(node_count, dtype=np.int64)
    birth = np.asarray(values, dtype=float).copy()
    active = np.zeros(node_count, dtype=bool)
    bars: list[tuple[float, float, int]] = []

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            following = int(parent[node])
            parent[node] = root
            node = following
        return root

    for node_value in order:
        node = int(node_value)
        active[node] = True
        threshold = float(values[node])
        roots = sorted({find(int(other)) for other in adjacency[node] if active[int(other)]})
        for other_root in roots:
            root = find(node)
            other_root = find(other_root)
            if root == other_root:
                continue
            if (birth[root], -root) >= (birth[other_root], -other_root):
                survivor, dying = root, other_root
            else:
                survivor, dying = other_root, root
            bars.append((float(birth[dying]), threshold, int(dying)))
            parent[dying] = survivor
    roots = sorted({find(node) for node in range(node_count)})
    finite = [
        {
            "birth": start,
            "death": end,
            "persistence": float(start - end),
            "component_seed_node": seed,
        }
        for start, end, seed in sorted(bars, key=lambda row: (row[0] - row[1], row[0]), reverse=True)
    ]
    return {
        "finite_bar_count": len(finite),
        "essential_component_count": len(roots),
        "essential_component_births": [float(birth[root]) for root in roots],
        "top_finite_bars": finite[: min(32, len(finite))],
        "filtration": "node_superlevel_with_induced_edges",
    }


def _edge_components(
    mask: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    node_count: int,
) -> list[set[int]]:
    active_edges = np.flatnonzero(mask)
    incident: list[list[int]] = [[] for _ in range(node_count)]
    for edge in active_edges:
        incident[int(left[edge])].append(int(edge))
        incident[int(right[edge])].append(int(edge))
    unseen = set(int(value) for value in active_edges)
    components: list[set[int]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component = {start}
        queue: deque[int] = deque([start])
        while queue:
            edge = queue.popleft()
            for node in (int(left[edge]), int(right[edge])):
                for other in incident[node]:
                    if other in unseen:
                        unseen.remove(other)
                        component.add(other)
                        queue.append(other)
        components.append(component)
    return components


def _defect_dynamics_report(
    defects: np.ndarray | None,
    cycles: np.ndarray,
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    if defects is None:
        return {
            "available": False,
            "reason": "defect_frames_not_supplied",
            "abundance_available": False,
            "lifetime_available": False,
            "motion_fusion_annihilation_available": False,
        }
    active = np.abs(defects) > 0.0
    frame_components = [
        _edge_components(frame, left, right, points.shape[0]) for frame in active
    ]
    rows: list[dict[str, Any]] = []
    edge_midpoints = 0.5 * (points[left] + points[right])
    for frame, components in enumerate(frame_components):
        sizes = [len(component) for component in components]
        rows.append(
            {
                "frame_index": frame,
                "cycle": float(cycles[frame]),
                "active_edge_count": int(np.count_nonzero(active[frame])),
                "active_edge_fraction": float(np.mean(active[frame])),
                "network_component_count": len(components),
                "largest_component_edges": max(sizes, default=0),
                "mean_component_edges": float(np.mean(sizes)) if sizes else 0.0,
            }
        )
    observed_episode_lengths: list[int] = []
    uncensored_episode_lengths: list[int] = []
    observed_cycle_span_lower_bounds: list[float] = []
    left_censored_count = 0
    right_censored_count = 0
    doubly_censored_count = 0
    for edge in range(active.shape[1]):
        start = None
        for frame in range(active.shape[0] + 1):
            on = frame < active.shape[0] and active[frame, edge]
            if on and start is None:
                start = frame
            elif not on and start is not None:
                length = frame - start
                left_censored = start == 0
                right_censored = frame == active.shape[0]
                observed_episode_lengths.append(length)
                observed_cycle_span_lower_bounds.append(
                    float(cycles[frame - 1] - cycles[start])
                    if length >= 2
                    else 0.0
                )
                if left_censored:
                    left_censored_count += 1
                if right_censored:
                    right_censored_count += 1
                if left_censored and right_censored:
                    doubly_censored_count += 1
                if not left_censored and not right_censored:
                    uncensored_episode_lengths.append(length)
                start = None

    births = deaths = fusions = splits = 0
    displacements: list[float] = []
    for frame in range(1, len(frame_components)):
        previous = frame_components[frame - 1]
        current = frame_components[frame]
        old_owner = {
            edge: component_index
            for component_index, component in enumerate(previous)
            for edge in component
        }
        new_owner = {
            edge: component_index
            for component_index, component in enumerate(current)
            for edge in component
        }
        old_to_new: list[set[int]] = [set() for _ in previous]
        new_to_old: list[set[int]] = [set() for _ in current]
        for edge in set(old_owner).intersection(new_owner):
            old_index = old_owner[edge]
            new_index = new_owner[edge]
            old_to_new[old_index].add(new_index)
            new_to_old[new_index].add(old_index)
        births += sum(1 for owners in new_to_old if not owners)
        deaths += sum(1 for owners in old_to_new if not owners)
        fusions += sum(1 for owners in new_to_old if len(owners) >= 2)
        splits += sum(1 for owners in old_to_new if len(owners) >= 2)
        for i, old in enumerate(previous):
            linked = old_to_new[i]
            if len(linked) == 1:
                new_index = next(iter(linked))
                if len(new_to_old[new_index]) != 1:
                    continue
                new = current[new_index]
                old_center = np.mean(edge_midpoints[list(old)], axis=0)
                new_center = np.mean(edge_midpoints[list(new)], axis=0)
                displacements.append(float(np.linalg.norm(new_center - old_center)))
    return {
        "available": True,
        "classification": "NONZERO_EDGE_FIELD_COMPONENT_TRACKING",
        "rows": rows,
        "edge_activity_episodes": {
            "episode_count": len(observed_episode_lengths),
            "left_censored_count": int(left_censored_count),
            "right_censored_count": int(right_censored_count),
            "doubly_censored_count": int(doubly_censored_count),
            "all_observed_run_lengths_frames": {
                "count": len(observed_episode_lengths),
                "minimum": min(observed_episode_lengths, default=None),
                "maximum": max(observed_episode_lengths, default=None),
                "mean": (
                    float(np.mean(observed_episode_lengths))
                    if observed_episode_lengths
                    else None
                ),
            },
            "completed_uncensored_run_lengths_frames": {
                "count": len(uncensored_episode_lengths),
                "minimum": min(uncensored_episode_lengths, default=None),
                "maximum": max(uncensored_episode_lengths, default=None),
                "mean": (
                    float(np.mean(uncensored_episode_lengths))
                    if uncensored_episode_lengths
                    else None
                ),
            },
            "observed_active_span_cycles_lower_bound": {
                "minimum": min(observed_cycle_span_lower_bounds, default=None),
                "maximum": max(observed_cycle_span_lower_bounds, default=None),
                "mean": (
                    float(np.mean(observed_cycle_span_lower_bounds))
                    if observed_cycle_span_lower_bounds
                    else None
                ),
            },
            "exact_lifetime_distribution_available": False,
            "reason": (
                "saved frames interval-censor transitions, and episodes touching "
                "the first or last supplied frame are left- or right-censored"
            ),
        },
        "component_events": {
            "births": int(births),
            "deaths_or_annihilations_proxy": int(deaths),
            "fusions": int(fusions),
            "splits": int(splits),
        },
        "event_identification_caveat": (
            "component overlap between consecutive supplied frames labels "
            "occupancy-pattern births, deaths, fusions, and splits; edge fields "
            "carry no persistent defect identity, and events between saved frames "
            "are unresolved"
        ),
        "unambiguous_individual_defect_motion_available": False,
        "unambiguous_individual_defect_motion_reason": (
            "persistent defect identifiers or a transition/current field were not supplied"
        ),
        "matched_component_embedding_displacement": {
            "count": len(displacements),
            "mean_chord": float(np.mean(displacements)) if displacements else None,
            "maximum_chord": max(displacements, default=None),
        },
        "network_scaling_fit_available": False,
        "network_scaling_fit_reason": "several sizes and sufficiently long scaling windows are required",
        "physical_defect_identification": False,
    }


def _normalized_laplacian(left: np.ndarray, right: np.ndarray, node_count: int) -> sparse.csr_matrix:
    row = np.concatenate([left, right])
    col = np.concatenate([right, left])
    adjacency = sparse.csr_matrix((np.ones(row.size), (row, col)), shape=(node_count, node_count))
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros(node_count, dtype=float)
    positive = degree > 0.0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized = sparse.diags(inv_sqrt) @ adjacency @ sparse.diags(inv_sqrt)
    diagonal = sparse.diags(positive.astype(float))
    return diagonal - normalized


def _low_eigenbasis(laplacian: sparse.csr_matrix, mode_count: int) -> tuple[np.ndarray, np.ndarray]:
    node_count = laplacian.shape[0]
    count = max(1, min(int(mode_count), node_count))
    if node_count <= 256 or count == node_count:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
        return eigenvalues[:count], eigenvectors[:, :count]
    count = min(count, node_count - 1)
    try:
        eigenvalues, eigenvectors = sparse_linalg.eigsh(
            laplacian,
            k=count,
            which="SM",
            tol=1.0e-7,
            maxiter=max(1000, node_count * 3),
        )
    except sparse_linalg.ArpackNoConvergence as error:
        if error.eigenvalues is None or len(error.eigenvalues) < 2:
            raise
        eigenvalues = error.eigenvalues
        eigenvectors = error.eigenvectors
    order = np.argsort(eigenvalues)
    return np.asarray(eigenvalues)[order], np.asarray(eigenvectors)[:, order]


def _field_statistics_report(
    state: np.ndarray,
    cycles: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    frame_indices: np.ndarray,
    spectral_modes: int,
) -> dict[str, Any]:
    if spectral_modes < 1:
        raise ValueError("spectral_modes must be positive")
    laplacian = _normalized_laplacian(left, right, state.shape[1])
    requested_modes = min(int(spectral_modes), state.shape[1])
    oversampled_modes = min(state.shape[1], requested_modes + 8)
    eigenvalues, eigenvectors = _low_eigenbasis(laplacian, oversampled_modes)
    candidate_bands = _eigenvalue_bands(eigenvalues)
    retained_bands = [
        band
        for band in candidate_bands
        if int(band["stop"]) <= requested_modes
        and (
            int(band["stop"]) < eigenvalues.size
            or eigenvalues.size == state.shape[1]
        )
    ]
    retained_indices = np.asarray(
        [
            mode
            for band in retained_bands
            for mode in range(int(band["start"]), int(band["stop"]))
        ],
        dtype=np.int64,
    )
    spectral_rows: list[dict[str, Any]] = []
    moment_rows: list[dict[str, Any]] = []
    cubic_rows: list[dict[str, Any]] = []
    nonconstant_bands = [
        band for band in retained_bands if not bool(band["zero_eigenvalue_band"])
    ][:3]
    for frame in frame_indices:
        values = state[int(frame)]
        centered = values - float(np.mean(values))
        coefficients = eigenvectors.T @ centered
        powers = coefficients * coefficients
        spectral_rows.append(
            {
                "frame_index": int(frame),
                "cycle": float(cycles[int(frame)]),
                "eigenvalue_band_powers": [
                    {
                        "band": int(band_index),
                        "mode_start_inclusive": int(band["start"]),
                        "mode_stop_exclusive": int(band["stop"]),
                        "multiplicity": int(band["stop"] - band["start"]),
                        "normalized_laplacian_eigenvalue_mean": float(
                            band["eigenvalue_mean"]
                        ),
                        "power": float(
                            np.sum(
                                powers[int(band["start"]) : int(band["stop"])]
                            )
                        ),
                    }
                    for band_index, band in enumerate(retained_bands)
                ],
                "captured_power_fraction": (
                    float(
                        np.sum(powers[retained_indices])
                        / np.dot(centered, centered)
                    )
                    if retained_indices.size and np.dot(centered, centered) > 0.0
                    else None
                ),
            }
        )
        scale = float(np.std(values))
        z = centered / scale if scale > 0.0 else np.zeros_like(centered)
        moment_rows.append(
            {
                "frame_index": int(frame),
                "cycle": float(cycles[int(frame)]),
                "mean": float(np.mean(values)),
                "standard_deviation": scale,
                "skewness": float(np.mean(z**3)) if scale > 0.0 else None,
                "excess_kurtosis": float(np.mean(z**4) - 3.0) if scale > 0.0 else None,
            }
        )
        cubic_rows.append(
            {
                "frame_index": int(frame),
                "cycle": float(cycles[int(frame)]),
                "band_triples": _basis_invariant_band_cubic_rows(
                    centered,
                    eigenvectors,
                    nonconstant_bands,
                ),
            }
        )
    retained_mode_count = int(retained_indices.size)
    truncated_band_dropped = bool(
        candidate_bands
        and int(candidate_bands[-1]["stop"]) == eigenvalues.size
        and eigenvalues.size < state.shape[1]
    )
    return {
        "available": True,
        "sampled_frame_indices": [int(value) for value in frame_indices],
        "graph_spectrum": {
            "normalization": "symmetric_normalized_laplacian",
            "requested_mode_limit": int(requested_modes),
            "oversampled_computed_mode_count": int(eigenvalues.size),
            "retained_complete_eigenspace_mode_count": retained_mode_count,
            "terminal_or_limit_crossing_band_dropped": truncated_band_dropped,
            "eigenvalue_band_tolerance": {
                "absolute": 1.0e-8,
                "relative": 1.0e-6,
            },
            "individual_eigenvector_coefficients_reported": False,
            "reason": (
                "individual coefficients are basis-dependent inside degenerate "
                "eigenspaces; powers are summed over complete eigenvalue bands"
            ),
            "rows": spectral_rows,
        },
        "field_moments": moment_rows,
        "basis_invariant_low_band_cubic_couplings": {
            "available": bool(nonconstant_bands),
            "rows": cubic_rows,
            "band_limit": 3,
            "basis_invariant_within_each_retained_degenerate_eigenspace": True,
            "ensemble_bispectrum_estimate": False,
            "reason": (
                "these are cubic products of complete graph-eigenspace projections "
                "in one supplied field, not ensemble expectations of Fourier-mode "
                "products"
            ),
        },
        "physical_power_spectrum_claim": False,
        "primordial_non_gaussianity_claim": False,
    }


def _eigenvalue_bands(eigenvalues: np.ndarray) -> list[dict[str, Any]]:
    """Group numerically degenerate consecutive eigenvalues."""

    values = np.asarray(eigenvalues, dtype=float)
    if values.size == 0:
        return []
    bands: list[dict[str, Any]] = []
    start = 0
    for stop in range(1, values.size + 1):
        if stop < values.size and math.isclose(
            float(values[stop - 1]),
            float(values[stop]),
            rel_tol=1.0e-6,
            abs_tol=1.0e-8,
        ):
            continue
        segment = values[start:stop]
        bands.append(
            {
                "start": int(start),
                "stop": int(stop),
                "eigenvalue_mean": float(np.mean(segment)),
                "eigenvalue_minimum": float(np.min(segment)),
                "eigenvalue_maximum": float(np.max(segment)),
                "zero_eigenvalue_band": bool(
                    np.max(np.abs(segment)) <= 1.0e-8
                ),
            }
        )
        start = stop
    return bands


def _basis_invariant_band_cubic_rows(
    centered: np.ndarray,
    eigenvectors: np.ndarray,
    bands: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Cubic couplings of full eigenspace projections.

    Each projected node field ``P_band f`` is invariant under orthogonal basis
    rotations within its eigenvalue band.  The resulting pointwise cubic mean
    therefore avoids the arbitrary-mode problem of individual graph Fourier
    triples.
    """

    projections: list[np.ndarray] = []
    labels: list[list[int]] = []
    for band in bands:
        start = int(band["start"])
        stop = int(band["stop"])
        basis = eigenvectors[:, start:stop]
        projections.append(basis @ (basis.T @ centered))
        labels.append([start, stop])
    rows: list[dict[str, Any]] = []
    for first in range(len(projections)):
        for second in range(first, len(projections)):
            for third in range(second, len(projections)):
                fields = (
                    projections[first],
                    projections[second],
                    projections[third],
                )
                mean_product = float(np.mean(fields[0] * fields[1] * fields[2]))
                rms_product = float(
                    np.prod(
                        [np.sqrt(np.mean(field * field)) for field in fields]
                    )
                )
                rows.append(
                    {
                        "mode_ranges_stop_exclusive": [
                            labels[first], labels[second], labels[third]
                        ],
                        "mean_projected_field_product": mean_product,
                        "rms_normalized_cubic_coupling": (
                            float(mean_product / rms_product)
                            if rms_product > 0.0
                            else None
                        ),
                    }
                )
    return rows


def _symmetry_report(
    points: np.ndarray,
    state: np.ndarray,
    cycles: np.ndarray,
    frame_indices: np.ndarray,
) -> dict[str, Any]:
    radii = np.linalg.norm(points, axis=1)
    if np.any(radii <= np.finfo(float).eps):
        reason = "embedding contains a node at the origin, so radial axes and antipodes are undefined"
        return {
            "preferred_axis": {"available": False, "reason": reason},
            "antipodal_parity": {"available": False, "reason": reason},
            "chirality": {
                "available": False,
                "reason": (
                    "scalar node fields plus an unoriented edge graph do not define a label-independent pseudoscalar; oriented faces or handed transport data are required"
                ),
                "chirality_claim": False,
            },
        }
    unit = points / np.where(radii[:, None] > 0.0, radii[:, None], 1.0)
    axis_rows: list[dict[str, Any]] = []
    for frame in frame_indices:
        field = _standardized(state[int(frame)])
        if field is None:
            axis_rows.append(
                {
                    "frame_index": int(frame),
                    "cycle": float(cycles[int(frame)]),
                    "available": False,
                    "reason": "constant_state_frame",
                }
            )
            continue
        normalization = float(np.sum(np.abs(field)))
        dipole = np.sum(field[:, None] * unit, axis=0) / normalization
        quadrupole = np.einsum("n,ni,nj->ij", field, unit, unit) / normalization
        quadrupole -= np.eye(3) * np.trace(quadrupole) / 3.0
        eigenvalues, eigenvectors = np.linalg.eigh(quadrupole)
        principal = int(np.argmax(np.abs(eigenvalues)))
        ordered_strengths = np.sort(np.abs(eigenvalues))
        eigenvalue_gap = float(ordered_strengths[-1] - ordered_strengths[-2])
        axis_well_conditioned = bool(
            ordered_strengths[-1] > 1.0e-12 and eigenvalue_gap > 1.0e-12
        )
        axis_rows.append(
            {
                "frame_index": int(frame),
                "cycle": float(cycles[int(frame)]),
                "available": True,
                "dipole_vector": [float(value) for value in dipole],
                "dipole_strength": float(np.linalg.norm(dipole)),
                "quadrupole_eigenvalues": [float(value) for value in eigenvalues],
                "principal_quadrupole_axis_embedding_coordinates": [
                    float(value) for value in eigenvectors[:, principal]
                ] if axis_well_conditioned else None,
                "principal_quadrupole_absolute_eigenvalue": float(
                    abs(eigenvalues[principal])
                ),
                "principal_to_next_absolute_eigenvalue_gap": eigenvalue_gap,
                "principal_axis_numerically_well_conditioned": axis_well_conditioned,
                "principal_axis_sign_is_arbitrary": True,
            }
        )
    parity = _antipodal_parity_report(unit, state, cycles, frame_indices)
    return {
        "preferred_axis": {
            "available": any(row["available"] for row in axis_rows),
            "sampled_frame_indices": [int(value) for value in frame_indices],
            "rows": axis_rows,
            "statistical_preferred_axis_detection_available": False,
            "statistical_preferred_axis_detection_reason": (
                "a finite field always has sample dipole and quadrupole axes; no "
                "carrier-preserving null ensemble or multiple-testing calibration "
                "is supplied"
            ),
            "preferred_axis_detection_claim": False,
            "physical_isotropy_claim": False,
        },
        "antipodal_parity": parity,
        "chirality": {
            "available": False,
            "reason": (
                "scalar node fields plus an unoriented edge graph do not define a label-independent pseudoscalar; oriented faces or handed transport data are required"
            ),
            "chirality_claim": False,
        },
    }


def _antipodal_parity_report(
    unit: np.ndarray,
    state: np.ndarray,
    cycles: np.ndarray,
    frame_indices: np.ndarray,
) -> dict[str, Any]:
    tree = cKDTree(unit)
    distances, partners = tree.query(-unit, k=1)
    involutive = bool(np.all(partners[partners] == np.arange(unit.shape[0])))
    exact = bool(involutive and float(np.max(distances)) <= 1.0e-7)
    if not exact:
        return {
            "available": False,
            "reason": "carrier has no numerically exact involutive antipodal node pairing",
            "maximum_antipode_mismatch_chord": float(np.max(distances)),
        }
    pair_starts = np.flatnonzero(np.arange(unit.shape[0]) <= partners)
    rows: list[dict[str, Any]] = []
    for frame in frame_indices:
        raw_values = state[int(frame)]
        values = raw_values - float(np.mean(raw_values))
        paired = values[partners]
        even = 0.5 * (values + paired)
        odd = 0.5 * (values - paired)
        even_power = float(np.mean(even[pair_starts] ** 2))
        odd_power = float(np.mean(odd[pair_starts] ** 2))
        rows.append(
            {
                "frame_index": int(frame),
                "cycle": float(cycles[int(frame)]),
                "even_power": even_power,
                "odd_power": odd_power,
                "odd_fraction": float(odd_power / (even_power + odd_power)) if even_power + odd_power > 0.0 else None,
            }
        )
    return {
        "available": True,
        "maximum_antipode_mismatch_chord": float(np.max(distances)),
        "global_spatial_mean_removed_before_parity_split": True,
        "rows": rows,
        "statistical_parity_anomaly_claim": False,
        "physical_parity_claim": False,
    }


def _velocity_report(
    state: np.ndarray,
    velocity: np.ndarray | None,
    cycles: np.ndarray,
) -> dict[str, Any]:
    if velocity is None:
        return {"available": False, "reason": "velocity_frames_not_supplied"}
    rows = []
    for frame in range(cycles.size):
        rows.append(
            {
                "frame_index": frame,
                "cycle": float(cycles[frame]),
                "mean": float(np.mean(velocity[frame])),
                "rms": float(np.sqrt(np.mean(velocity[frame] ** 2))),
                "state_velocity_correlation": _pearson(state[frame], velocity[frame]),
            }
        )
    return {
        "available": True,
        "classification": "SUPPLIED_SCALAR_NODE_RATE_FIELD",
        "rows": rows,
        "physical_velocity_or_kinetic_energy_claim": False,
    }


def _commit_report(commits: np.ndarray | None, cycles: np.ndarray) -> dict[str, Any]:
    if commits is None:
        return {"available": False, "reason": "commit_frames_not_supplied"}
    committed = commits > 0.5
    first = np.full(committed.shape[1], -1, dtype=np.int64)
    for frame in range(committed.shape[0]):
        first[(first < 0) & committed[frame]] = frame
    revocations = np.count_nonzero(committed[:-1] & ~committed[1:], axis=0) if committed.shape[0] > 1 else np.zeros(committed.shape[1], dtype=int)
    return {
        "available": True,
        "rows": [
            {
                "frame_index": frame,
                "cycle": float(cycles[frame]),
                "committed_fraction": float(np.mean(committed[frame])),
                "new_commit_count": int(np.count_nonzero(first == frame)),
            }
            for frame in range(committed.shape[0])
        ],
        "never_committed_count": int(np.count_nonzero(first < 0)),
        "revocation_event_count": int(np.sum(revocations)),
        "nodes_with_revocation": int(np.count_nonzero(revocations)),
        "record_production_arrow_claim": bool(
            np.all(np.diff(np.mean(committed, axis=1)) >= -1.0e-15)
            and not np.any(revocations)
        ),
        "record_arrow_scope": "monotonicity of supplied finite commit flags only",
    }


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    x = np.asarray(first, dtype=float) - float(np.mean(first))
    y = np.asarray(second, dtype=float) - float(np.mean(second))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0.0 else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
