"""Target-blind observer and excitation diagnostics on a finite graph history.

The routines in this module intentionally use only graph topology and arrays
produced by a simulation.  Graph hops and simulation cycles remain internal
units.  In particular, a localized component is an *excitation candidate*, a
component encounter is a *channel diagnostic*, and a stable numerical cluster
is a *candidate family*.  None of those objects is thereby a particle, a
mass, a scattering amplitude, a causal light cone, or a laboratory observable.

The public API accepts generic arrays so a run can preserve both a compact
JSON receipt and machine-readable tables/arrays without coupling the analysis
to one simulator backend.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA = "oph_observer_excitation_observables_v1"
CLASSIFICATION = "TARGET_BLIND_INTERNAL_DIAGNOSTIC_ONLY"


def observer_excitation_observables(
    *,
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    cycles: np.ndarray,
    state_frames: np.ndarray,
    velocity_frames: np.ndarray,
    record_frames: np.ndarray,
    commit_frames: np.ndarray,
    defect_frames: np.ndarray,
    intervention_delta: np.ndarray | None = None,
    intervention_origin_cycle: float | None = None,
    intervention_origin_mask: np.ndarray | None = None,
    locality_hops_per_cycle: float | None = None,
    entropy_bins: int = 8,
    excitation_threshold: float = 3.0,
    excitation_min_lifetime_frames: int = 2,
    latent_neighborhood_anchor_count: int = 2048,
    max_graph_hops: int = 8,
    seed: int = 17,
) -> dict[str, Any]:
    """Return a deterministic, target-blind diagnostic report.

    ``state_frames`` and ``velocity_frames`` may be ``(time, node)`` scalar
    arrays or ``(time, node, ...)`` channel arrays.  Multi-channel inputs are
    reduced to their root-mean-square magnitude; that reduction is declared in
    the report and should not be mistaken for a physically selected scalar.
    Record and commit arrays must have shape ``(time, node)``.  Defects may be
    node fields ``(time, node)`` or exact seam fields ``(time, edge)``; seam
    defects are preserved and also projected to an incident-node mask.
    A paired-response cone is available only when ``intervention_delta`` is
    accompanied by its actual injection cycle, injection-node mask, and the
    kernel's declared graph-locality bound in hops per simulation cycle.
    """

    report, _ = _analyze(
        points=points,
        left=left,
        right=right,
        cycles=cycles,
        state_frames=state_frames,
        velocity_frames=velocity_frames,
        record_frames=record_frames,
        commit_frames=commit_frames,
        defect_frames=defect_frames,
        intervention_delta=intervention_delta,
        intervention_origin_cycle=intervention_origin_cycle,
        intervention_origin_mask=intervention_origin_mask,
        locality_hops_per_cycle=locality_hops_per_cycle,
        entropy_bins=entropy_bins,
        excitation_threshold=excitation_threshold,
        excitation_min_lifetime_frames=excitation_min_lifetime_frames,
        latent_neighborhood_anchor_count=latent_neighborhood_anchor_count,
        max_graph_hops=max_graph_hops,
        seed=seed,
    )
    return report


def write_observer_excitation_observables(
    out_dir: str | Path,
    *,
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    cycles: np.ndarray,
    state_frames: np.ndarray,
    velocity_frames: np.ndarray,
    record_frames: np.ndarray,
    commit_frames: np.ndarray,
    defect_frames: np.ndarray,
    intervention_delta: np.ndarray | None = None,
    intervention_origin_cycle: float | None = None,
    intervention_origin_mask: np.ndarray | None = None,
    locality_hops_per_cycle: float | None = None,
    entropy_bins: int = 8,
    excitation_threshold: float = 3.0,
    excitation_min_lifetime_frames: int = 2,
    latent_neighborhood_anchor_count: int = 2048,
    max_graph_hops: int = 8,
    seed: int = 17,
) -> dict[str, Any]:
    """Write the report and lossless analysis products beneath ``out_dir``."""

    report, arrays = _analyze(
        points=points,
        left=left,
        right=right,
        cycles=cycles,
        state_frames=state_frames,
        velocity_frames=velocity_frames,
        record_frames=record_frames,
        commit_frames=commit_frames,
        defect_frames=defect_frames,
        intervention_delta=intervention_delta,
        intervention_origin_cycle=intervention_origin_cycle,
        intervention_origin_mask=intervention_origin_mask,
        locality_hops_per_cycle=locality_hops_per_cycle,
        entropy_bins=entropy_bins,
        excitation_threshold=excitation_threshold,
        excitation_min_lifetime_frames=excitation_min_lifetime_frames,
        latent_neighborhood_anchor_count=latent_neighborhood_anchor_count,
        max_graph_hops=max_graph_hops,
        seed=seed,
    )
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    json_path = root / "observer_excitation_observables.json"
    json_path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_rows(
        root / "observer_information_timeseries.csv",
        report["information_dynamics"]["time_rows"],
    )
    _write_rows(
        root / "observer_autocorrelation.csv",
        report["mixing_and_record_arrow"]["autocorrelation_rows"],
    )
    _write_rows(
        root / "observer_homogeneity_scale.csv",
        report["latent_patch_neighborhoods"]["latent_smoothing_rows"],
    )
    _write_rows(
        root / "observer_excitation_components.csv",
        report["localized_excitations"]["component_catalog"],
    )
    _write_rows(
        root / "observer_excitation_tracks.csv",
        report["localized_excitations"]["track_catalog"],
    )
    _write_rows(
        root / "observer_mode_dispersion.csv",
        report["graph_field_snapshot_proxies"]["time_rows"],
    )
    _write_rows(
        root / "observer_scattering_candidates.csv",
        report["candidate_scattering_channels"]["event_rows"],
    )
    _write_rows(
        root / "observer_candidate_families.csv",
        report["candidate_family_clustering"]["assignment_rows"],
    )
    np.savez_compressed(root / "observer_excitation_analysis_arrays.npz", **arrays)

    readme = root / "OBSERVER_EXCITATION_OBSERVABLES.md"
    readme.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def _analyze(
    *,
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    cycles: np.ndarray,
    state_frames: np.ndarray,
    velocity_frames: np.ndarray,
    record_frames: np.ndarray,
    commit_frames: np.ndarray,
    defect_frames: np.ndarray,
    intervention_delta: np.ndarray | None,
    intervention_origin_cycle: float | None,
    intervention_origin_mask: np.ndarray | None,
    locality_hops_per_cycle: float | None,
    entropy_bins: int,
    excitation_threshold: float,
    excitation_min_lifetime_frames: int,
    latent_neighborhood_anchor_count: int,
    max_graph_hops: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    points = np.asarray(points, dtype=float)
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    cycles = np.asarray(cycles, dtype=float)
    _validate_graph_time(points, left, right, cycles)
    time_count = int(cycles.size)
    node_count = int(points.shape[0])
    if int(excitation_min_lifetime_frames) < 1:
        raise ValueError("excitation_min_lifetime_frames must be positive")
    if int(latent_neighborhood_anchor_count) < 1:
        raise ValueError("latent_neighborhood_anchor_count must be positive")

    state, state_reduction = _scalar_frames(
        state_frames, "state_frames", time_count, node_count, signed_scalar=True
    )
    # Preserve signed scalar velocities.  Taking ``abs(v)`` before estimating
    # its robust scale uses a half-normal MAD and makes ordinary noise appear
    # anomalously large.  Multi-channel inputs retain the declared RMS
    # reduction because no signed scalar projection is supplied for them.
    velocity, velocity_reduction = _scalar_frames(
        velocity_frames, "velocity_frames", time_count, node_count, signed_scalar=True
    )
    records = _two_dimensional(record_frames, "record_frames", time_count, node_count)
    commits = _two_dimensional(commit_frames, "commit_frames", time_count, node_count)
    defects, defect_mask, defect_semantics = _defect_arrays(
        defect_frames, time_count, node_count, left, right
    )
    record_mask = records != 0
    commit_mask = commits != 0

    delta: np.ndarray | None = None
    delta_reduction: str | None = None
    if intervention_delta is not None:
        delta, delta_reduction = _scalar_frames(
            intervention_delta,
            "intervention_delta",
            time_count,
            node_count,
            signed_scalar=False,
        )

    adjacency = _adjacency(node_count, left, right)
    quantized, quantile_edges = _quantize(state, int(entropy_bins))
    information = _information_report(quantized, left, right, cycles, int(seed))
    mixing = _mixing_and_arrow_report(
        state, record_mask, commit_mask, defect_mask, cycles
    )
    local_skies, local_arrays = _local_sky_report(
        state, quantized, records, commits, cycles, adjacency, left, right,
        max_hops=max(1, int(max_graph_hops)),
        anchor_count=int(latent_neighborhood_anchor_count),
        seed=int(seed) + 101,
    )
    intervention = _paired_intervention_report(
        delta,
        cycles,
        adjacency,
        max_hops=max(1, int(max_graph_hops)),
        origin_cycle=intervention_origin_cycle,
        origin_mask=intervention_origin_mask,
        locality_hops_per_cycle=locality_hops_per_cycle,
    )
    amplitude = _excitation_amplitude(state, velocity)
    excitation, excitation_arrays, raw_components, tracks = _excitation_report(
        amplitude,
        points,
        cycles,
        adjacency,
        threshold=float(excitation_threshold),
    )
    mode = _mode_report(state, velocity, left, right, cycles)
    scattering = _scattering_report(
        raw_components, excitation_arrays["component_labels"], adjacency, cycles
    )
    families = _candidate_family_report(
        tracks,
        seed=int(seed) + 211,
        min_lifetime_frames=int(excitation_min_lifetime_frames),
    )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "target_data_read": False,
        "measurement_files_read": [],
        "array_semantics": {
            "graph_distance_unit": "edge_hops",
            "time_unit": "simulation_cycles",
            "state_reduction": state_reduction,
            "velocity_reduction": velocity_reduction,
            "intervention_reduction": delta_reduction,
            "record_mask_rule": "value != 0; signed record values are retained separately",
            "commit_mask_rule": "value != 0",
            "defect_semantics": defect_semantics,
        },
        "dimensions": {
            "time_count": time_count,
            "node_count": node_count,
            "edge_count": int(left.size),
            "point_dimension": int(points.shape[1]),
        },
        "declared_diagnostic_settings": {
            "excitation_threshold": float(excitation_threshold),
            "excitation_min_lifetime_frames": int(
                excitation_min_lifetime_frames
            ),
            "latent_neighborhood_anchor_count_requested": int(
                latent_neighborhood_anchor_count
            ),
            "max_graph_hops": int(max_graph_hops),
        },
        "input_custody": {
            "points": _array_receipt(points),
            "left": _array_receipt(left),
            "right": _array_receipt(right),
            "cycles": _array_receipt(cycles),
            "state_frames": _array_receipt(np.asarray(state_frames)),
            "velocity_frames": _array_receipt(np.asarray(velocity_frames)),
            "record_frames": _array_receipt(np.asarray(record_frames)),
            "commit_frames": _array_receipt(np.asarray(commit_frames)),
            "defect_frames": _array_receipt(np.asarray(defect_frames)),
            "intervention_delta": (
                None if intervention_delta is None
                else _array_receipt(np.asarray(intervention_delta))
            ),
            "intervention_origin_mask": (
                None if intervention_origin_mask is None
                else _array_receipt(np.asarray(intervention_origin_mask))
            ),
            "intervention_origin_cycle": intervention_origin_cycle,
            "locality_hops_per_cycle": locality_hops_per_cycle,
        },
        "quantization": {
            "requested_bins": int(entropy_bins),
            "realized_bins": int(np.unique(quantized).size),
            "global_quantile_edges": quantile_edges.tolist(),
        },
        "information_dynamics": information,
        "mixing_and_record_arrow": mixing,
        "defect_dynamics": _defect_dynamics_report(defects, defect_mask, defect_semantics),
        "latent_patch_neighborhoods": local_skies,
        # Compatibility alias.  Status fields in ``local_skies`` make clear
        # that this is a latent-field aggregate rather than an observer sky.
        "observer_local_skies": {
            **local_skies,
            "deprecated_alias_for": "latent_patch_neighborhoods",
        },
        "paired_intervention_response": intervention,
        "localized_excitations": excitation,
        "graph_field_snapshot_proxies": mode,
        "mode_localization_and_dispersion": {
            **mode,
            "deprecated_alias_for": "graph_field_snapshot_proxies",
        },
        "candidate_scattering_channels": scattering,
        "candidate_family_clustering": families,
        "epistemic_limits": {
            "localized_component_is_particle": False,
            "track_energy_proxy_is_mass": False,
            "component_encounter_is_physical_scattering": False,
            "graph_cycle_cone_is_physical_light_cone": False,
            "numerical_cluster_is_particle_family": False,
            "state_field_is_cmb_temperature_or_matter_density": False,
            "reason": (
                "The analysis has no source-to-laboratory realization map, "
                "metric/time calibration, detector model, or externally frozen "
                "same-quantity comparison."
            ),
        },
        "physical_promotion_gates": {
            "state_to_physical_observable_map": False,
            "graph_metric_to_length_map": False,
            "cycle_to_time_map": False,
            "excitation_to_particle_state_map": False,
            "energy_momentum_calibration": False,
            "asymptotic_in_out_state_construction": False,
            "independent_validation_data_reserved": False,
        },
        "physical_claims": {
            "particle_identification": False,
            "mass_measurement": False,
            "scattering_amplitude": False,
            "causal_speed": False,
            "cosmological_prediction": False,
        },
    }
    arrays = {
        "cycles": cycles,
        "quantized_state": quantized.astype(np.int16),
        "excitation_amplitude": amplitude.astype(np.float32),
        "excitation_component_labels": excitation_arrays["component_labels"],
        "excitation_track_labels": excitation_arrays["track_labels"],
        "local_sky_mean": local_arrays["mean"].astype(np.float32),
        "local_sky_variance": local_arrays["variance"].astype(np.float32),
        "defect_frames_raw": np.asarray(defect_frames),
        "incident_defect_mask": defect_mask.astype(np.uint8),
    }
    if delta is not None:
        arrays["intervention_delta_magnitude"] = delta.astype(np.float32)
    if intervention_origin_mask is not None:
        arrays["intervention_origin_mask"] = np.asarray(
            intervention_origin_mask, dtype=np.uint8
        )
    return report, arrays


def _validate_graph_time(
    points: np.ndarray, left: np.ndarray, right: np.ndarray, cycles: np.ndarray
) -> None:
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 1:
        raise ValueError("points must have shape (node>=2, coordinate>=1)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points contain non-finite coordinates")
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape or left.size == 0:
        raise ValueError("left and right must be nonempty one-dimensional arrays of equal shape")
    if np.any(left < 0) or np.any(right < 0):
        raise ValueError("edge endpoints must be nonnegative")
    if np.any(left >= points.shape[0]) or np.any(right >= points.shape[0]):
        raise ValueError("edge endpoint lies outside points")
    if np.any(left == right):
        raise ValueError("self-loop edges are not supported")
    if cycles.ndim != 1 or cycles.size < 2 or not np.all(np.isfinite(cycles)):
        raise ValueError("cycles must be a finite one-dimensional array of length at least two")
    if np.any(np.diff(cycles) <= 0):
        raise ValueError("cycles must be strictly increasing")


def _scalar_frames(
    values: np.ndarray,
    name: str,
    time_count: int,
    node_count: int,
    *,
    signed_scalar: bool,
) -> tuple[np.ndarray, str]:
    data = np.asarray(values, dtype=float)
    if data.shape[:2] != (time_count, node_count):
        raise ValueError(
            f"{name} has leading shape {data.shape[:2]}, expected {(time_count, node_count)}"
        )
    if data.ndim == 2:
        scalar = data
        reduction = "identity_scalar" if signed_scalar else "absolute_scalar_magnitude"
        if not signed_scalar:
            scalar = np.abs(scalar)
    elif data.ndim >= 3:
        flat = data.reshape(time_count, node_count, -1)
        scalar = np.sqrt(np.mean(np.square(flat), axis=2))
        reduction = "root_mean_square_channel_magnitude"
    else:
        raise ValueError(f"{name} must have at least two dimensions")
    if not np.all(np.isfinite(scalar)):
        raise ValueError(f"{name} contains non-finite values")
    return scalar.astype(float, copy=False), reduction


def _two_dimensional(
    values: np.ndarray, name: str, time_count: int, node_count: int
) -> np.ndarray:
    data = np.asarray(values)
    if data.shape != (time_count, node_count):
        raise ValueError(f"{name} has shape {data.shape}, expected {(time_count, node_count)}")
    if not np.all(np.isfinite(data.astype(float))):
        raise ValueError(f"{name} contains non-finite values")
    return data


def _defect_arrays(
    values: np.ndarray,
    time_count: int,
    node_count: int,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    data = np.asarray(values)
    if data.shape == (time_count, left.size):
        if not np.all(np.isfinite(data.astype(float))):
            raise ValueError("defect_frames contains non-finite values")
        edge_mask = data != 0
        incident = np.zeros((time_count, node_count), dtype=bool)
        # Vectorized frame loop retains the exact edge field in ``data`` while
        # providing the observer-local incident-defect mask used below.
        for time_index in range(time_count):
            active_edges = np.flatnonzero(edge_mask[time_index])
            if active_edges.size:
                incident[time_index, left[active_edges]] = True
                incident[time_index, right[active_edges]] = True
        return data, incident, "edge_defects_with_incident_node_projection"
    if data.shape == (time_count, node_count):
        if not np.all(np.isfinite(data.astype(float))):
            raise ValueError("defect_frames contains non-finite values")
        return data, data != 0, "node_defects_projected_by_identity"
    raise ValueError(
        "defect_frames has shape "
        f"{data.shape}, expected {(time_count, node_count)} or {(time_count, left.size)}"
    )


def _defect_dynamics_report(
    raw: np.ndarray, incident_mask: np.ndarray, semantics: str
) -> dict[str, Any]:
    raw_mask = raw != 0
    counts = np.count_nonzero(raw_mask, axis=1)
    incident_counts = np.count_nonzero(incident_mask, axis=1)
    return {
        "input_semantics": semantics,
        "exact_defect_counts_by_frame": counts.astype(int).tolist(),
        "incident_node_counts_by_frame": incident_counts.astype(int).tolist(),
        "initial_exact_defect_count": int(counts[0]),
        "final_exact_defect_count": int(counts[-1]),
        "initial_incident_node_count": int(incident_counts[0]),
        "final_incident_node_count": int(incident_counts[-1]),
    }


def _adjacency(node_count: int, left: np.ndarray, right: np.ndarray) -> list[np.ndarray]:
    rows: list[list[int]] = [[] for _ in range(node_count)]
    for a, b in zip(left.tolist(), right.tolist(), strict=True):
        rows[a].append(b)
        rows[b].append(a)
    return [np.asarray(sorted(set(row)), dtype=np.int64) for row in rows]


def _quantize(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    if bins < 2:
        raise ValueError("entropy_bins must be at least two")
    probabilities = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    edges = np.unique(np.quantile(values.reshape(-1), probabilities))
    return np.digitize(values, edges, right=False).astype(np.int16), edges


def _entropy(labels: np.ndarray) -> float:
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts.astype(float) / float(labels.size)
    return float(-np.sum(p * np.log(p)))


def _mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((int(ai.max()) + 1, int(bi.max()) + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    pxy = table.astype(float) / float(a.size)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / expected[mask])))


def _conditional_entropy(current: np.ndarray, later: np.ndarray) -> float:
    return float(max(0.0, _entropy(later) - _mutual_information(current, later)))


def _information_report(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    cycles: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    time_rows: list[dict[str, Any]] = []
    for index in range(labels.shape[0]):
        row: dict[str, Any] = {
            "time_index": index,
            "cycle": float(cycles[index]),
            "state_entropy_nats": _entropy(labels[index]),
        }
        if index:
            row["temporal_mutual_information_nats"] = _mutual_information(
                labels[index - 1], labels[index]
            )
            row["conditional_entropy_rate_nats_per_step"] = _conditional_entropy(
                labels[index - 1], labels[index]
            )
        else:
            row["temporal_mutual_information_nats"] = None
            row["conditional_entropy_rate_nats_per_step"] = None
        if left.size:
            row["edge_mutual_information_nats"] = _mutual_information(
                labels[index, left], labels[index, right]
            )
        else:
            row["edge_mutual_information_nats"] = None
        time_rows.append(row)

    temporal_a = labels[:-1].reshape(-1)
    temporal_b = labels[1:].reshape(-1)
    spatial_a = labels[:, left].reshape(-1)
    spatial_b = labels[:, right].reshape(-1)
    rng = np.random.default_rng(seed)
    controls = []
    for _ in range(8):
        permuted = np.empty_like(labels)
        for t in range(labels.shape[0]):
            permuted[t] = labels[t, rng.permutation(labels.shape[1])]
        controls.append(
            _mutual_information(
                permuted[:, left].reshape(-1), permuted[:, right].reshape(-1)
            )
        )
    return {
        "entropy_unit": "natural_log_nats",
        "entropy_rate_estimator": "global_quantile_symbol_conditional_entropy_per_frame",
        "mean_state_entropy_nats": float(np.mean([r["state_entropy_nats"] for r in time_rows])),
        "temporal_mutual_information_nats": _mutual_information(temporal_a, temporal_b),
        "conditional_entropy_rate_nats_per_step": _conditional_entropy(temporal_a, temporal_b),
        "edge_mutual_information_nats": _mutual_information(spatial_a, spatial_b),
        "edge_shuffle_control_mean_nats": float(np.mean(controls)),
        "edge_shuffle_control_std_nats": float(np.std(controls)),
        "time_rows": time_rows,
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size < 2 or b.size != a.size or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _mixing_and_arrow_report(
    state: np.ndarray,
    records: np.ndarray,
    commits: np.ndarray,
    defects: np.ndarray,
    cycles: np.ndarray,
) -> dict[str, Any]:
    max_lag = min(32, state.shape[0] - 1)
    autocorrelation_rows = [{"lag_frames": 0, "lag_cycles": 0.0, "correlation": 1.0}]
    for lag in range(1, max_lag + 1):
        autocorrelation_rows.append(
            {
                "lag_frames": lag,
                "lag_cycles": float(np.median(cycles[lag:] - cycles[:-lag])),
                "correlation": _correlation(state[:-lag], state[lag:]),
            }
        )
    mixing_row = next(
        (
            row for row in autocorrelation_rows[1:]
            if row["correlation"] is not None and abs(row["correlation"]) <= math.e ** -1
        ),
        None,
    )

    record_births = np.count_nonzero(records[1:] & ~records[:-1], axis=1)
    record_losses = np.count_nonzero(~records[1:] & records[:-1], axis=1)
    commit_births = np.count_nonzero(commits[1:] & ~commits[:-1], axis=1)
    commit_revocations = np.count_nonzero(~commits[1:] & commits[:-1], axis=1)
    ever_recorded = np.maximum.accumulate(records.astype(np.uint8), axis=0)
    cumulative_fraction = np.mean(ever_recorded, axis=1)
    current_record_fraction = np.mean(records, axis=1)
    production = int(np.sum(record_births))
    destruction = int(np.sum(record_losses))
    denom = production + destruction
    return {
        "autocorrelation_rows": autocorrelation_rows,
        "mixing_time_proxy_frames": None if mixing_row is None else mixing_row["lag_frames"],
        "mixing_time_proxy_cycles": None if mixing_row is None else mixing_row["lag_cycles"],
        "mixing_time_censored": bool(mixing_row is None),
        "mixing_definition": "first sampled lag with absolute flattened state autocorrelation <= exp(-1)",
        "record_birth_count": production,
        "record_loss_count": destruction,
        "record_arrow_score": 0.0 if denom == 0 else float((production - destruction) / denom),
        "record_births_by_transition": record_births.astype(int).tolist(),
        "record_losses_by_transition": record_losses.astype(int).tolist(),
        "commit_births_by_transition": commit_births.astype(int).tolist(),
        "commit_revocations_by_transition": commit_revocations.astype(int).tolist(),
        "cumulative_ever_recorded_fraction": cumulative_fraction.tolist(),
        "current_record_fraction_by_frame": current_record_fraction.tolist(),
        "record_fraction_monotone": bool(
            np.all(np.diff(current_record_fraction) >= -1e-15)
        ),
        "cumulative_ever_recorded_fraction_monotone_by_construction": True,
        "final_record_fraction": float(np.mean(records[-1])),
        "final_commit_fraction": float(np.mean(commits[-1])),
        "initial_defect_count": int(np.count_nonzero(defects[0])),
        "final_defect_count": int(np.count_nonzero(defects[-1])),
        "interpretation": "record-production ordering in run cycles; no physical thermodynamic arrow is inferred",
    }


def _neighborhoods(
    adjacency: list[np.ndarray], radius: int, origins: Iterable[int] | None = None
) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    selected = range(len(adjacency)) if origins is None else origins
    for origin_value in selected:
        origin = int(origin_value)
        seen = {origin}
        frontier = {origin}
        for _ in range(radius):
            frontier = {
                int(neighbor)
                for node in frontier
                for neighbor in adjacency[node]
                if int(neighbor) not in seen
            }
            seen.update(frontier)
            if not frontier:
                break
        result.append(np.asarray(sorted(seen), dtype=np.int64))
    return result


def _local_sky_report(
    state: np.ndarray,
    quantized: np.ndarray,
    records: np.ndarray,
    commits: np.ndarray,
    cycles: np.ndarray,
    adjacency: list[np.ndarray],
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_hops: int,
    anchor_count: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    final = state[-1]
    one_hop = _neighborhoods(adjacency, 1)
    local_mean = np.asarray([np.mean(final[nodes]) for nodes in one_hop])
    local_variance = np.asarray([np.var(final[nodes]) for nodes in one_hop])
    final_signed_record = np.asarray(records[-1], dtype=float)
    final_commit = np.asarray(commits[-1]) != 0
    public_record = np.where(final_commit, final_signed_record, 0.0)
    local_public_record_mean = np.asarray(
        [np.mean(public_record[nodes]) for nodes in one_hop]
    )
    local_commit_coverage = np.asarray(
        [np.mean(final_commit[nodes]) for nodes in one_hop]
    )
    global_scale = max(float(np.std(final)), np.finfo(float).eps)
    overlap_fractions = []
    estimate_agreements = []
    for a, b in zip(left.tolist(), right.tolist(), strict=True):
        aa = set(one_hop[a].tolist())
        bb = set(one_hop[b].tolist())
        overlap_fractions.append(len(aa & bb) / max(1, len(aa | bb)))
        estimate_agreements.append(math.exp(-abs(local_mean[a] - local_mean[b]) / global_scale))
    both_recorded = (
        final_commit[left]
        & final_commit[right]
        & (final_signed_record[left] != 0)
        & (final_signed_record[right] != 0)
    )
    endpoint_sign_similarity = (
        None if not np.any(both_recorded)
        else float(
            np.mean(
                np.sign(final_signed_record[left[both_recorded]])
                == np.sign(final_signed_record[right[both_recorded]])
            )
        )
    )

    smoothing_rows: list[dict[str, Any]] = []
    smoothing_radius = None
    global_variance = float(np.var(final))
    rng = np.random.default_rng(seed)
    homogeneity_anchors = np.sort(
        rng.choice(
            len(adjacency),
            size=min(int(anchor_count), len(adjacency)),
            replace=False,
        )
    )
    for radius in range(0, max_hops + 1):
        neighborhoods = _neighborhoods(adjacency, radius, homogeneity_anchors)
        means = np.asarray([np.mean(final[nodes]) for nodes in neighborhoods])
        ratio = 0.0 if global_variance == 0 else float(np.var(means) / global_variance)
        smoothing_rows.append(
            {
                "radius_hops": radius,
                "observer_mean_variance": float(np.var(means)),
                "variance_ratio_to_node_field": ratio,
                "mean_patch_size": float(np.mean([len(nodes) for nodes in neighborhoods])),
            }
        )
        if smoothing_radius is None and ratio <= 0.10:
            smoothing_radius = radius

    cone = _operational_cone_dependence(
        quantized, cycles, adjacency, max_hops=max_hops, seed=seed
    )
    return (
        {
            "diagnostic_scope": "latent_patch_neighborhoods",
            "observer_sky_claim": False,
            "same_source_observer_consensus_available": False,
            "same_source_observer_consensus_reason": (
                "the input contains one node field and one patch record per node, not "
                "two observer-indexed reports of the same source quantity"
            ),
            "latent_patch_mean_variance": float(np.var(local_mean)),
            "mean_within_latent_patch_variance": float(np.mean(local_variance)),
            "latent_patch_mean_range": [float(np.min(local_mean)), float(np.max(local_mean))],
            "mean_adjacent_neighborhood_overlap_fraction": float(np.mean(overlap_fractions)),
            "mean_adjacent_sky_overlap_fraction": float(np.mean(overlap_fractions)),
            "mean_adjacent_latent_estimate_similarity": float(np.mean(estimate_agreements)),
            "committed_record_local_mean_variance": float(np.var(local_public_record_mean)),
            "mean_local_commit_coverage": float(np.mean(local_commit_coverage)),
            "legacy_adjacent_committed_endpoint_sign_similarity": endpoint_sign_similarity,
            "adjacent_nonzero_signed_record_sign_agreement": None,
            "overlap_agreement_limitation": (
                "Input supplies one node field, not independent observer readouts; agreement "
                "therefore compares overlapping local aggregate estimates only."
            ),
            "latent_smoothing_threshold_variance_ratio": 0.10,
            "latent_smoothing_anchor_count": int(homogeneity_anchors.size),
            "latent_smoothing_threshold_radius_hops": smoothing_radius,
            "latent_smoothing_threshold_censored": smoothing_radius is None,
            "latent_smoothing_rows": smoothing_rows,
            "homogeneity_claim_available": False,
            "homogeneity_scale_hops": None,
            "homogeneity_scale_censored": True,
            "homogeneity_rows": smoothing_rows,
            "latent_graph_cone_dependence": cone,
            "operational_causal_diamond_dependence": {
                **cone,
                "deprecated_alias_for": "latent_graph_cone_dependence",
            },
        },
        {"mean": local_mean, "variance": local_variance},
    )


def _distances_from(origin: int, adjacency: list[np.ndarray], max_hops: int) -> np.ndarray:
    distance = np.full(len(adjacency), max_hops + 1, dtype=np.int32)
    distance[origin] = 0
    frontier = [origin]
    for hop in range(1, max_hops + 1):
        next_frontier: list[int] = []
        for node in frontier:
            for neighbor in adjacency[node]:
                neighbor = int(neighbor)
                if distance[neighbor] > max_hops:
                    distance[neighbor] = hop
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return distance


def _operational_cone_dependence(
    labels: np.ndarray,
    cycles: np.ndarray,
    adjacency: list[np.ndarray],
    *,
    max_hops: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    anchor_count = min(64, labels.shape[1])
    anchors = np.sort(rng.choice(labels.shape[1], size=anchor_count, replace=False))
    inside_a: list[int] = []
    inside_b: list[int] = []
    outside_a: list[int] = []
    outside_b: list[int] = []
    for anchor in anchors.tolist():
        distance = _distances_from(anchor, adjacency, max_hops)
        future = int(labels[-1, anchor])
        for time_index in range(labels.shape[0] - 1):
            available_hops = min(max_hops, labels.shape[0] - 1 - time_index)
            inside = distance <= available_hops
            outside = (distance > available_hops) & (distance <= max_hops)
            inside_a.extend(labels[time_index, inside].astype(int).tolist())
            inside_b.extend([future] * int(np.count_nonzero(inside)))
            outside_a.extend(labels[time_index, outside].astype(int).tolist())
            outside_b.extend([future] * int(np.count_nonzero(outside)))
    inside_mi = _mutual_information(np.asarray(inside_a), np.asarray(inside_b))
    outside_mi = _mutual_information(np.asarray(outside_a), np.asarray(outside_b))
    return {
        "available": True,
        "diagnostic_scope": "latent_graph_cone_partition",
        "convention": "one graph hop per sampled frame, capped at max_graph_hops",
        "anchor_count": int(anchor_count),
        "inside_sample_count": len(inside_a),
        "outside_sample_count": len(outside_a),
        "inside_mutual_information_nats": inside_mi,
        "outside_mutual_information_nats": outside_mi,
        "inside_minus_outside_mutual_information_nats": float(inside_mi - outside_mi),
        "same_source_observer_causal_diamond_available": False,
        "matched_inside_outside_control_available": False,
        "physical_causality_claim": False,
        "reason": (
            "the cone convention partitions one latent node field and has no "
            "matched inside/outside control or observer-indexed same-source readouts"
        ),
    }


def _multi_source_distances(origins: Iterable[int], adjacency: list[np.ndarray]) -> np.ndarray:
    distance = np.full(len(adjacency), np.iinfo(np.int32).max, dtype=np.int32)
    frontier = sorted(set(int(value) for value in origins))
    for origin in frontier:
        distance[origin] = 0
    hop = 0
    while frontier:
        next_frontier: list[int] = []
        for node in frontier:
            for neighbor in adjacency[node]:
                neighbor = int(neighbor)
                if distance[neighbor] == np.iinfo(np.int32).max:
                    distance[neighbor] = hop + 1
                    next_frontier.append(neighbor)
        frontier = next_frontier
        hop += 1
    return distance


def _paired_intervention_report(
    delta: np.ndarray | None,
    cycles: np.ndarray,
    adjacency: list[np.ndarray],
    *,
    max_hops: int,
    origin_cycle: float | None,
    origin_mask: np.ndarray | None,
    locality_hops_per_cycle: float | None,
) -> dict[str, Any]:
    if delta is None:
        return {
            "available": False,
            "reason": "no same-seed paired intervention delta was supplied",
            "causal_speed_claim": False,
        }
    missing = [
        name
        for name, value in (
            ("intervention_origin_cycle", origin_cycle),
            ("intervention_origin_mask", origin_mask),
            ("locality_hops_per_cycle", locality_hops_per_cycle),
        )
        if value is None
    ]
    if missing:
        return {
            "available": False,
            "reason": (
                "paired response support is present, but the declared injection "
                "metadata required for a cone test is missing: " + ", ".join(missing)
            ),
            "missing_required_metadata": missing,
            "response_support_detected": bool(np.any(delta != 0)),
            "causal_speed_claim": False,
        }

    origin_cycle_value = float(origin_cycle)
    locality_value = float(locality_hops_per_cycle)
    if not np.isfinite(origin_cycle_value):
        raise ValueError("intervention_origin_cycle must be finite")
    if not np.isfinite(locality_value) or locality_value <= 0:
        raise ValueError("locality_hops_per_cycle must be positive and finite")
    origin_mask_array = np.asarray(origin_mask)
    if origin_mask_array.shape != (len(adjacency),):
        raise ValueError(
            "intervention_origin_mask has shape "
            f"{origin_mask_array.shape}, expected {(len(adjacency),)}"
        )
    if not np.all(np.isfinite(origin_mask_array.astype(float))):
        raise ValueError("intervention_origin_mask contains non-finite values")
    origin_mask_bool = origin_mask_array != 0
    origins = np.flatnonzero(origin_mask_bool)
    if origins.size == 0:
        raise ValueError("intervention_origin_mask must select at least one node")

    scale = max(float(np.max(np.abs(delta))), np.finfo(float).eps)
    support_tolerance = max(1e-12, 1e-8 * scale)
    active = np.abs(delta) > support_tolerance
    active_times = np.flatnonzero(np.any(active, axis=1))
    if active_times.size == 0:
        return {
            "available": False,
            "reason": "paired intervention delta is identically zero",
            "causal_speed_claim": False,
        }
    pre_origin = active & (cycles[:, None] < origin_cycle_value - 1e-12)
    if np.any(pre_origin):
        return {
            "available": False,
            "reason": "paired delta is nonzero before the declared intervention cycle",
            "pre_intervention_active_count": int(np.count_nonzero(pre_origin)),
            "causal_speed_claim": False,
        }
    distance = _multi_source_distances(origins, adjacency)
    unreachable_value = np.iinfo(np.int32).max
    rows = []
    total_active = 0
    outside_active = 0
    total_unreachable = 0
    beyond_analysis_radius = 0
    fit_x = []
    fit_y = []
    for time_index in active_times.tolist():
        if cycles[time_index] < origin_cycle_value:
            continue
        nodes = np.flatnonzero(active[time_index])
        radii = distance[nodes]
        reachable = radii != unreachable_value
        reachable_radii = radii[reachable]
        radius = int(np.max(reachable_radii)) if reachable_radii.size else None
        allowed = max(
            0.0,
            float(cycles[time_index] - origin_cycle_value) * locality_value,
        )
        unreachable = int(np.count_nonzero(~reachable))
        outside = unreachable + int(np.count_nonzero(reachable_radii > allowed + 1e-12))
        total_active += int(nodes.size)
        outside_active += outside
        total_unreachable += unreachable
        beyond_analysis_radius += int(np.count_nonzero(reachable_radii > max_hops))
        rows.append(
            {
                "time_index": time_index,
                "cycle": float(cycles[time_index]),
                "active_node_count": int(nodes.size),
                "maximum_response_radius_hops": radius,
                "allowed_response_radius_hops": allowed,
                "declared_locality_cone_leak_count": outside,
                "unreachable_response_node_count": unreachable,
            }
        )
        if radius is not None and radius <= max_hops:
            fit_x.append(float(cycles[time_index] - origin_cycle_value))
            fit_y.append(float(radius))
    slope = None
    if len(set(fit_x)) >= 2:
        slope = float(np.polyfit(np.asarray(fit_x), np.asarray(fit_y), 1)[0])
    return {
        "available": True,
        "metadata_complete": True,
        "origin_cycle": origin_cycle_value,
        "origin_node_count": int(origins.size),
        "locality_hops_per_cycle": locality_value,
        "support_detection_tolerance": support_tolerance,
        "analysis_radius_cap_hops": int(max_hops),
        "response_rows": rows,
        "front_slope_hops_per_cycle_proxy": slope,
        "declared_locality_cone_leakage_fraction": (
            0.0 if total_active == 0 else float(outside_active / total_active)
        ),
        # Compatibility key: its value now uses actual cycles and the declared
        # locality, never the number of stored frames.
        "one_hop_per_frame_cone_leakage_fraction": (
            0.0 if total_active == 0 else float(outside_active / total_active)
        ),
        "unreachable_response_node_count": total_unreachable,
        "response_nodes_beyond_analysis_radius_count": beyond_analysis_radius,
        "causal_speed_claim": False,
        "reason": "response support in paired arrays is diagnostic until a physical metric and clock are derived",
    }


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mad = float(np.median(np.abs(values - np.median(values)))) * 1.4826
    if mad > np.finfo(float).eps:
        return mad
    std = float(np.std(values))
    return std if std > np.finfo(float).eps else 1.0


def _excitation_amplitude(state: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    centered = state - np.median(state, axis=1, keepdims=True)
    centered_velocity = velocity - np.median(velocity, axis=1, keepdims=True)
    state_scale = _robust_scale(centered)
    velocity_scale = _robust_scale(centered_velocity)
    return np.sqrt(
        np.square(centered / state_scale)
        + np.square(centered_velocity / velocity_scale)
    )


def _connected_components(active: np.ndarray, adjacency: list[np.ndarray]) -> list[np.ndarray]:
    remaining = set(np.flatnonzero(active).astype(int).tolist())
    components: list[np.ndarray] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack = [root]
        nodes = [root]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                neighbor = int(neighbor)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    nodes.append(neighbor)
        components.append(np.asarray(sorted(nodes), dtype=np.int64))
    return components


def _component_touch(a: np.ndarray, b: np.ndarray, adjacency: list[np.ndarray]) -> bool:
    b_set = set(b.astype(int).tolist())
    if any(int(node) in b_set for node in a):
        return True
    return any(
        int(neighbor) in b_set
        for node in a
        for neighbor in adjacency[int(node)]
    )


def _excitation_report(
    amplitude: np.ndarray,
    points: np.ndarray,
    cycles: np.ndarray,
    adjacency: list[np.ndarray],
    *,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[list[dict[str, Any]]], list[dict[str, Any]]]:
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("excitation_threshold must be a positive finite number")
    labels = np.full(amplitude.shape, -1, dtype=np.int32)
    raw_by_time: list[list[dict[str, Any]]] = []
    catalog: list[dict[str, Any]] = []
    component_id = 0
    for time_index in range(amplitude.shape[0]):
        active = amplitude[time_index] >= threshold
        components = _connected_components(active, adjacency)
        frame_rows: list[dict[str, Any]] = []
        for nodes in components:
            weights = np.square(amplitude[time_index, nodes])
            energy = float(np.sum(weights))
            centroid = np.average(points[nodes], axis=0, weights=weights)
            squared_distance = np.sum(np.square(points[nodes] - centroid), axis=1)
            radius = float(np.sqrt(np.average(squared_distance, weights=weights)))
            ipr = float(np.sum(np.square(weights)) / max(energy * energy, np.finfo(float).eps))
            peak_offset = int(np.argmax(amplitude[time_index, nodes]))
            row = {
                "component_id": component_id,
                "time_index": time_index,
                "cycle": float(cycles[time_index]),
                "node_count": int(nodes.size),
                "peak_node": int(nodes[peak_offset]),
                "peak_amplitude": float(np.max(amplitude[time_index, nodes])),
                "energy_proxy": energy,
                "radius_coordinate_units": radius,
                "inverse_participation_ratio": ipr,
                "centroid": centroid.tolist(),
                "nodes": nodes,
            }
            labels[time_index, nodes] = component_id
            component_id += 1
            frame_rows.append(row)
            catalog.append(_public_component_row(row))
        raw_by_time.append(frame_rows)

    # Greedy one-to-one tracking uses actual node overlap first and one-edge
    # contact second.  Multi-parent/multi-child relations remain available to
    # the independent encounter diagnostic below.
    track_labels = np.full_like(labels, -1)
    tracks: list[dict[str, Any]] = []
    active_tracks: dict[int, dict[str, Any]] = {}
    component_to_track: dict[int, int] = {}
    next_track_id = 0
    for time_index, components in enumerate(raw_by_time):
        proposals: list[tuple[int, int, int, int]] = []
        if time_index:
            previous_components = raw_by_time[time_index - 1]
            previous_by_id = {
                int(row["component_id"]): row for row in previous_components
            }
            for current_index, current in enumerate(components):
                current_nodes = set(current["nodes"].tolist())
                candidate_ids = _touching_component_ids(
                    current["nodes"], labels[time_index - 1], adjacency
                )
                for previous_id in candidate_ids:
                    previous = previous_by_id[previous_id]
                    previous_nodes = set(previous["nodes"].tolist())
                    overlap = len(current_nodes & previous_nodes)
                    proposals.append(
                        (
                            overlap,
                            -abs(len(current_nodes) - len(previous_nodes)),
                            component_to_track[previous_id],
                            current_index,
                        )
                    )
        assigned_tracks: set[int] = set()
        assigned_components: set[int] = set()
        for _, _, track_id, current_index in sorted(proposals, reverse=True):
            if track_id < 0 or track_id in assigned_tracks or current_index in assigned_components:
                continue
            component = components[current_index]
            active_tracks[track_id]["components"].append(component)
            track_labels[time_index, component["nodes"]] = track_id
            component_to_track[int(component["component_id"])] = track_id
            assigned_tracks.add(track_id)
            assigned_components.add(current_index)
        for current_index, component in enumerate(components):
            if current_index in assigned_components:
                continue
            track = {"track_id": next_track_id, "components": [component]}
            tracks.append(track)
            active_tracks[next_track_id] = track
            track_labels[time_index, component["nodes"]] = next_track_id
            component_to_track[int(component["component_id"])] = next_track_id
            next_track_id += 1

    track_catalog = [_track_summary(track, cycles) for track in tracks]
    return (
        {
            "threshold_in_robust_amplitude_units": threshold,
            "component_count": len(catalog),
            "track_count": len(track_catalog),
            "active_node_fraction": float(np.mean(labels >= 0)),
            "component_catalog": catalog,
            "track_catalog": track_catalog,
            "feature_status": "localized graph-field features only",
            "particle_identification": False,
        },
        {"component_labels": labels, "track_labels": track_labels},
        raw_by_time,
        track_catalog,
    )


def _public_component_row(row: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in row.items() if key not in {"nodes", "centroid"}}
    public["centroid_coordinates"] = ";".join(f"{float(v):.12g}" for v in row["centroid"])
    return public


def _track_summary(track: dict[str, Any], cycles: np.ndarray) -> dict[str, Any]:
    components = track["components"]
    start = int(components[0]["time_index"])
    end = int(components[-1]["time_index"])
    speeds = []
    for earlier, later in zip(components[:-1], components[1:], strict=True):
        dt = float(later["cycle"] - earlier["cycle"])
        if dt > 0:
            speeds.append(
                float(np.linalg.norm(np.asarray(later["centroid"]) - np.asarray(earlier["centroid"])) / dt)
            )
    radii = np.asarray([row["radius_coordinate_units"] for row in components], dtype=float)
    track_cycles = np.asarray([row["cycle"] for row in components], dtype=float)
    radius_slope = None
    if radii.size >= 2 and np.ptp(track_cycles) > 0:
        radius_slope = float(np.polyfit(track_cycles, radii, 1)[0])
    return {
        "track_id": int(track["track_id"]),
        "start_time_index": start,
        "end_time_index": end,
        "start_cycle": float(cycles[start]),
        "end_cycle": float(cycles[end]),
        "lifetime_frames": int(len(components)),
        "lifetime_cycles": float(cycles[end] - cycles[start]),
        "mean_node_count": float(np.mean([row["node_count"] for row in components])),
        "mean_energy_proxy": float(np.mean([row["energy_proxy"] for row in components])),
        "mean_peak_amplitude": float(np.mean([row["peak_amplitude"] for row in components])),
        "mean_coordinate_speed_proxy": None if not speeds else float(np.mean(speeds)),
        "maximum_coordinate_speed_proxy": None if not speeds else float(np.max(speeds)),
        "radius_growth_per_cycle_proxy": radius_slope,
        "mean_inverse_participation_ratio": float(
            np.mean([row["inverse_participation_ratio"] for row in components])
        ),
        "physical_particle_claim": False,
    }


def _mode_report(
    state: np.ndarray,
    velocity: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    cycles: np.ndarray,
) -> dict[str, Any]:
    rows = []
    for time_index in range(state.shape[0]):
        centered = state[time_index] - np.mean(state[time_index])
        norm2 = float(np.sum(np.square(centered)))
        if norm2 <= np.finfo(float).eps:
            k2 = None
            omega2 = None
            ipr = None
        else:
            k2 = float(np.sum(np.square(centered[left] - centered[right])) / norm2)
            omega2 = float(np.sum(np.square(velocity[time_index])) / norm2)
            ipr = float(np.sum(np.power(centered, 4)) / (norm2 * norm2))
        rows.append(
            {
                "time_index": time_index,
                "cycle": float(cycles[time_index]),
                "whole_field_graph_rayleigh_quotient": k2,
                "whole_field_velocity_to_state_ratio": omega2,
                "graph_rayleigh_k_squared_proxy": k2,
                "velocity_to_state_omega_squared_proxy": omega2,
                "state_inverse_participation_ratio": ipr,
                "participation_node_count_proxy": None if ipr in (None, 0.0) else float(1.0 / ipr),
            }
        )
    return {
        "diagnostic_scope": "whole_graph_snapshot_localization_proxies",
        "time_rows": rows,
        "dispersion_fit_available": False,
        "dispersion_fit_suppressed": True,
        "dispersion_fit_reason": (
            "a whole-snapshot Rayleigh quotient and velocity/state ratio across "
            "time do not identify graph modes or their temporal frequencies"
        ),
        "omega_squared_vs_graph_k_squared_slope": None,
        "omega_squared_vs_graph_k_squared_intercept": None,
        "omega_squared_vs_graph_k_squared_r_squared": None,
        "required_for_dispersion": (
            "graph-Laplacian mode or spectral-band projection plus an independent "
            "temporal-frequency estimate in a declared stationary window"
        ),
        "units": "dimensionless graph Rayleigh quotient and simulation-cycle velocity ratio",
        "physical_dispersion_relation_claim": False,
    }


def _touching_component_ids(
    nodes: np.ndarray, frame_labels: np.ndarray, adjacency: list[np.ndarray]
) -> list[int]:
    labels: set[int] = set()
    for node_value in nodes:
        node = int(node_value)
        own_label = int(frame_labels[node])
        if own_label >= 0:
            labels.add(own_label)
        for neighbor in adjacency[node]:
            label = int(frame_labels[int(neighbor)])
            if label >= 0:
                labels.add(label)
    return sorted(labels)


def _scattering_report(
    raw_by_time: list[list[dict[str, Any]]],
    component_labels: np.ndarray,
    adjacency: list[np.ndarray],
    cycles: np.ndarray,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for time_index in range(1, len(raw_by_time)):
        current = raw_by_time[time_index]
        previous_by_id = {
            int(row["component_id"]): row for row in raw_by_time[time_index - 1]
        }
        following_by_id = (
            {
                int(row["component_id"]): row
                for row in raw_by_time[time_index + 1]
            }
            if time_index + 1 < len(raw_by_time) else {}
        )
        for component in current:
            parent_ids = _touching_component_ids(
                component["nodes"], component_labels[time_index - 1], adjacency
            )
            parents = [previous_by_id[value] for value in parent_ids]
            if len(parents) < 2:
                continue
            child_ids = (
                _touching_component_ids(
                    component["nodes"], component_labels[time_index + 1], adjacency
                )
                if time_index + 1 < len(raw_by_time) else []
            )
            children = [following_by_id[value] for value in child_ids]
            incoming_energy = float(sum(row["energy_proxy"] for row in parents))
            outgoing_energy = float(sum(row["energy_proxy"] for row in children))
            events.append(
                {
                    "event_id": len(events),
                    "time_index": time_index,
                    "cycle": float(cycles[time_index]),
                    "incoming_component_count": len(parents),
                    "interaction_component_count": 1,
                    "outgoing_component_count": len(children),
                    "channel_topology": f"{len(parents)}_to_{len(children)}",
                    "incoming_energy_proxy": incoming_energy,
                    "outgoing_energy_proxy": outgoing_energy,
                    "energy_proxy_fractional_change": (
                        None if incoming_energy == 0
                        else float((outgoing_energy - incoming_energy) / incoming_energy)
                    ),
                    "has_post_encounter_split": len(children) >= 2,
                    "physical_scattering_claim": False,
                }
            )
    return {
        # Component adjacency patterns remain archived, but this lane fails
        # closed until their rate exceeds a matched null and a same-noise
        # four-arm counterfactual establishes nonlinear interaction.
        "available": False,
        "candidate_detection_available": bool(events),
        "reason": (
            "component merge/split candidates were found, but matched-null and "
            "four-arm nonlinear evidence were not supplied"
            if events else "no current component had at least two touching predecessor components"
        ),
        "encounter_count": len(events),
        "candidate_encounter_count": len(events),
        "event_rows": events,
        "selection_rule": "two or more prior localized components touch one current component",
        "matched_null_control_available": False,
        "four_arm_nonlinear_evidence_available": False,
        "encounter_excess_over_null_established": False,
        "interaction_promotion_available": False,
        "physical_scattering_claim": False,
        "cross_section_or_amplitude_available": False,
    }


def _candidate_family_report(
    tracks: list[dict[str, Any]], *, seed: int, min_lifetime_frames: int
) -> dict[str, Any]:
    all_eligible = [
        row
        for row in tracks
        if row["lifetime_frames"] >= int(min_lifetime_frames)
    ]
    if len(all_eligible) < 6:
        return {
            "available": False,
            "reason": (
                "fewer than six excitation tracks meet the declared minimum "
                f"lifetime of {int(min_lifetime_frames)} frames"
            ),
            "minimum_lifetime_frames": int(min_lifetime_frames),
            "eligible_track_count": len(all_eligible),
            "clustered_track_count": len(all_eligible),
            "assignment_rows": [],
            "stable_candidate_partition": False,
            "candidate_partition_computed": False,
            "one_family_null_rejected": False,
            "matched_null_control_available": False,
            "cross_seed_replication_available": False,
            "physical_particle_family_claim": False,
        }
    # Bound the quadratic stability/silhouette stage for large runs.  The
    # deterministic ordering favors well-resolved, long-lived tracks and is
    # disclosed in the receipt rather than silently sampling after inspection.
    eligible = sorted(
        all_eligible,
        key=lambda row: (
            -int(row["lifetime_frames"]),
            -float(row["mean_energy_proxy"]),
            int(row["track_id"]),
        ),
    )[:1024]
    features = np.asarray(
        [
            [
                row["lifetime_frames"],
                row["mean_node_count"],
                row["mean_energy_proxy"],
                row["mean_peak_amplitude"],
                row["mean_coordinate_speed_proxy"] or 0.0,
                row["mean_inverse_participation_ratio"],
            ]
            for row in eligible
        ],
        dtype=float,
    )
    scale = np.std(features, axis=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    z = (features - np.mean(features, axis=0)) / scale
    rng = np.random.default_rng(seed)
    candidates = []
    for cluster_count in range(2, min(4, len(eligible) // 2) + 1):
        labels, centers = _kmeans(z, cluster_count, rng)
        silhouette = _silhouette(z, labels)
        stabilities = []
        for _ in range(8):
            perturbed = z + rng.normal(0.0, 0.03, size=z.shape)
            trial_labels, _ = _kmeans(perturbed, cluster_count, rng)
            stabilities.append(_adjusted_rand(labels, trial_labels))
        counts = np.bincount(labels, minlength=cluster_count)
        candidates.append(
            {
                "cluster_count": cluster_count,
                "labels": labels,
                "centers": centers,
                "silhouette": silhouette,
                "mean_perturbation_adjusted_rand": float(np.mean(stabilities)),
                "minimum_cluster_size": int(np.min(counts)),
            }
        )
    best = max(candidates, key=lambda row: (row["silhouette"], row["mean_perturbation_adjusted_rand"]))
    perturbation_stable = bool(
        best["silhouette"] >= 0.25
        and best["mean_perturbation_adjusted_rand"] >= 0.75
        and best["minimum_cluster_size"] >= 2
    )
    assignments = [
        {
            "track_id": int(row["track_id"]),
            "candidate_family_label": int(label),
            "physical_particle_family_claim": False,
        }
        for row, label in zip(eligible, best["labels"].tolist(), strict=True)
    ]
    return {
        # K-means is retained as an exploratory partition for visualization,
        # but it is not an available family result: k=1, matched-null
        # clusterability, and cross-run replication have not been tested.
        "available": False,
        "candidate_partition_computed": True,
        "eligible_track_count": len(all_eligible),
        "minimum_lifetime_frames": int(min_lifetime_frames),
        "clustered_track_count": len(eligible),
        "large_run_selection_rule": "first 1024 by lifetime desc, energy desc, track_id asc",
        "selected_cluster_count": int(best["cluster_count"]),
        "silhouette": float(best["silhouette"]),
        "mean_perturbation_adjusted_rand": float(best["mean_perturbation_adjusted_rand"]),
        "minimum_cluster_size": int(best["minimum_cluster_size"]),
        "stability_thresholds": {"silhouette": 0.25, "adjusted_rand": 0.75, "minimum_size": 2},
        "exploratory_partition_stable_under_small_perturbations": perturbation_stable,
        "stable_candidate_partition": False,
        "one_family_null_rejected": False,
        "matched_null_control_available": False,
        "cross_seed_replication_available": False,
        "assignment_rows": assignments,
        "physical_particle_family_claim": False,
        "reason": (
            "an exploratory k-means partition is archived, but no one-family "
            "null, matched field control, or cross-seed replication was supplied"
        ),
    }


def _kmeans(
    values: np.ndarray, cluster_count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    best_labels = None
    best_centers = None
    best_inertia = math.inf
    for _ in range(12):
        indices = rng.choice(values.shape[0], size=cluster_count, replace=False)
        centers = values[indices].copy()
        labels = np.full(values.shape[0], -1, dtype=np.int32)
        for _ in range(100):
            distances = np.sum(np.square(values[:, None, :] - centers[None, :, :]), axis=2)
            new_labels = np.argmin(distances, axis=1).astype(np.int32)
            if np.array_equal(labels, new_labels):
                labels = new_labels
                break
            labels = new_labels
            for index in range(cluster_count):
                members = values[labels == index]
                if members.size:
                    centers[index] = np.mean(members, axis=0)
                else:
                    centers[index] = values[int(rng.integers(values.shape[0]))]
        inertia = float(np.sum(np.square(values - centers[labels])))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    assert best_labels is not None and best_centers is not None
    return best_labels, best_centers


def _silhouette(values: np.ndarray, labels: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return 0.0
    squared_norm = np.sum(np.square(values), axis=1)
    distances = np.sqrt(
        np.maximum(0.0, squared_norm[:, None] + squared_norm[None, :] - 2.0 * values @ values.T)
    )
    scores = []
    for index in range(values.shape[0]):
        own = labels == labels[index]
        own[index] = False
        if not np.any(own):
            scores.append(0.0)
            continue
        a = float(np.mean(distances[index, own]))
        other = [
            float(np.mean(distances[index, labels == candidate]))
            for candidate in np.unique(labels)
            if candidate != labels[index]
        ]
        if not other:
            scores.append(0.0)
            continue
        b = min(other)
        scores.append(0.0 if max(a, b) == 0 else (b - a) / max(a, b))
    return float(np.mean(scores))


def _adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 2:
        return 1.0
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((int(ai.max()) + 1, int(bi.max()) + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)

    def choose2(values: np.ndarray) -> float:
        values = values.astype(float)
        return float(np.sum(values * (values - 1.0) / 2.0))

    sum_cells = choose2(table)
    sum_a = choose2(table.sum(axis=1))
    sum_b = choose2(table.sum(axis=0))
    total = float(a.size * (a.size - 1) / 2)
    expected = sum_a * sum_b / total
    maximum = 0.5 * (sum_a + sum_b)
    denominator = maximum - expected
    return 1.0 if abs(denominator) <= np.finfo(float).eps else float((sum_cells - expected) / denominator)


def _array_receipt(values: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(values)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_jsonable(value), sort_keys=True)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _markdown_summary(report: dict[str, Any]) -> str:
    excitation = report["localized_excitations"]
    scattering = report["candidate_scattering_channels"]
    families = report["candidate_family_clustering"]
    intervention = report["paired_intervention_response"]
    return (
        "# Observer and excitation observables\n\n"
        f"Classification: `{report['classification']}`.\n\n"
        "This receipt analyzes graph-internal fields without reading a public measurement "
        "table. Distances are graph hops and times are simulation cycles. Localized features, "
        "encounters, and numerical families remain diagnostics; the analysis makes no particle, "
        "mass, scattering, causal-speed, or cosmological claim.\n\n"
        "## Inventory\n\n"
        f"- Nodes: {report['dimensions']['node_count']}\n"
        f"- Frames: {report['dimensions']['time_count']}\n"
        f"- Excitation components: {excitation['component_count']}\n"
        f"- Excitation tracks: {excitation['track_count']}\n"
        f"- Component encounter candidates: {scattering['candidate_encounter_count']}\n"
        f"- Encounter promotion available: {scattering['available']}\n"
        f"- Paired intervention available: {intervention['available']}\n"
        f"- Candidate-family clustering available: {families['available']}\n\n"
        "The adjacent JSON, CSV, and NPZ files are the machine-readable analysis products.\n"
    )


__all__ = [
    "CLASSIFICATION",
    "SCHEMA",
    "observer_excitation_observables",
    "write_observer_excitation_observables",
]
