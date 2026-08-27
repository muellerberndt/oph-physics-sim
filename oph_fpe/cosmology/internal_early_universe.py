"""Target-blind observer-internal diagnostics for finite OPH settling runs.

The module deliberately consumes no public measurement table.  Its fixed
primary scalar is ``first_commit_cycle``: the first finite repair cycle at
which each patch has a completed observer-visible record.  All reported
lengths are screen angles or graph hops and all reported times are run cycles.
The output can therefore diagnose a proposed finite source/scheduler from the
inside without silently identifying the run with the physical early universe.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from oph_fpe.cosmology.angular_power import angular_power_report
from oph_fpe.evidence.bundle import strict_jsonable


SCHEMA = "oph_internal_early_universe_diagnostics_v1"
PRIMARY_OBSERVABLE = "first_commit_cycle"
REQUIRED_ARRAYS = (
    "points",
    "edge_left",
    "edge_right",
    "cell_entropy",
    "initial_incident_mismatch_count",
    "first_commit_cycle",
    "first_repair_cycle",
    "last_repair_cycle",
    "first_quiescence_cycle",
    "cumulative_repair_load",
    "final_mismatch_density",
    "committed_final",
    "cycles",
    "record_commit_cycles",
)
OPTIONAL_NODE_EVENT_ARRAYS = (
    "last_commit_cycle",
    "last_record_change_cycle",
    "last_commit_state_change_cycle",
    "commit_revocation_count",
)


def internal_early_universe_report(
    run_dir: str | Path,
    *,
    ell_max: int = 16,
    pair_samples: int = 100_000,
    shuffle_draws: int = 16,
    thresholds: Iterable[float] = (0.50, 0.75, 0.90),
    seed: int = 17,
    n_jobs: int | str | None = 1,
    harmonic_batch_size: int = 4096,
) -> dict[str, Any]:
    """Compute source/scheduler diagnostics from one event-time artifact."""

    root = Path(run_dir)
    event_path = root / "screen_event_times.npz"
    if not event_path.is_file():
        raise FileNotFoundError(f"missing observer-internal event artifact: {event_path}")

    with np.load(event_path, allow_pickle=False) as payload:
        missing = [name for name in REQUIRED_ARRAYS if name not in payload.files]
        if missing:
            raise ValueError(f"screen event artifact is missing arrays: {missing}")
        points = np.asarray(payload["points"], dtype=float)
        left = np.asarray(payload["edge_left"], dtype=np.int64)
        right = np.asarray(payload["edge_right"], dtype=np.int64)
        weights = np.asarray(payload["cell_entropy"], dtype=float)
        initial_incident_mismatch = np.asarray(
            payload["initial_incident_mismatch_count"], dtype=np.int16
        )
        first_commit = np.asarray(payload["first_commit_cycle"], dtype=np.int32)
        first_repair = np.asarray(payload["first_repair_cycle"], dtype=np.int32)
        last_repair = np.asarray(payload["last_repair_cycle"], dtype=np.int32)
        first_quiescence = np.asarray(
            payload["first_quiescence_cycle"], dtype=np.int32
        )
        cumulative_repair = np.asarray(
            payload["cumulative_repair_load"], dtype=float
        )
        final_mismatch = np.asarray(payload["final_mismatch_density"], dtype=float)
        committed_final = np.asarray(payload["committed_final"], dtype=np.uint8)
        cycles = int(np.asarray(payload["cycles"]).reshape(-1)[0])
        record_commit_cycles = int(
            np.asarray(payload["record_commit_cycles"]).reshape(-1)[0]
        )
        optional_node_events = {
            name: np.asarray(payload[name])
            for name in OPTIONAL_NODE_EVENT_ARRAYS
            if name in payload.files
        }

    _validate_geometry(points, left, right, weights)
    node_count = int(points.shape[0])
    for name, values in {
        "initial_incident_mismatch_count": initial_incident_mismatch,
        "first_commit_cycle": first_commit,
        "first_repair_cycle": first_repair,
        "last_repair_cycle": last_repair,
        "first_quiescence_cycle": first_quiescence,
        "cumulative_repair_load": cumulative_repair,
        "final_mismatch_density": final_mismatch,
        "committed_final": committed_final,
    }.items():
        if values.shape != (node_count,):
            raise ValueError(
                f"event field {name!r} has shape {values.shape}, expected {(node_count,)}"
            )
    for name, values in optional_node_events.items():
        if values.shape != (node_count,):
            raise ValueError(
                f"optional event field {name!r} has shape {values.shape}, "
                f"expected {(node_count,)}"
            )

    normalized_weights = _normalized_weights(weights)
    censored = first_commit < 0
    # Censor unfinished records at the first cycle just outside the executed
    # interval.  The eligibility gate below remains closed whenever this is
    # needed, so a partial run cannot look like a completed freezeout surface.
    primary = np.where(censored, cycles, first_commit).astype(float)
    standardized_primary = _standardize(primary, normalized_weights)
    rng = np.random.default_rng(int(seed))

    distribution = _distribution_report(primary, normalized_weights)
    angular_geometry = _angular_geometry_report(
        points, standardized_primary, normalized_weights
    )
    seam_association = _seam_association_report(
        standardized_primary,
        left,
        right,
        pair_sample_limit=int(pair_samples),
        shuffle_draws=int(shuffle_draws),
        rng=rng,
    )
    graph_shell_correlation = _graph_shell_correlation_report(
        standardized_primary,
        left,
        right,
        anchor_count=min(2048, node_count),
        max_hops=8,
        shuffle_draws=min(16, max(1, int(shuffle_draws))),
        rng=rng,
    )
    angular_correlation = _angular_correlation_report(
        points,
        standardized_primary,
        pair_samples=int(pair_samples),
        rng=rng,
    )
    morphology = _morphology_report(
        primary,
        left,
        right,
        thresholds=tuple(float(value) for value in thresholds),
    )
    multiscale = _multiscale_moments_report(
        standardized_primary,
        left,
        right,
        normalized_weights,
        rng=rng,
    )
    trace = _trace_report(root / "mismatch_trace.csv")
    independent_seam_null = _independent_seam_null_report(
        first_commit,
        initial_incident_mismatch,
        record_commit_cycles=record_commit_cycles,
        trace=trace,
    )
    factorized_graph_control = _factorized_graph_control_report(
        points,
        left,
        right,
        first_commit,
        normalized_weights,
        record_commit_cycles=record_commit_cycles,
        trace=trace,
        analytic_null=independent_seam_null,
        draws=min(16, max(1, int(shuffle_draws))),
        rng=np.random.default_rng(int(seed) + 7919),
    )

    if int(ell_max) >= 2 and float(np.std(primary)) > 0.0:
        angular_spectrum = angular_power_report(
            points,
            {PRIMARY_OBSERVABLE: primary},
            ell_max=int(ell_max),
            pair_samples=0,
            seed=int(seed) + 101,
            controls=["shuffled_field", "random_gaussian"],
            estimator="spherical_harmonic",
            measure_weights=weights,
            harmonic_batch_size=int(harmonic_batch_size),
            n_jobs=n_jobs,
        )
    else:
        angular_spectrum = {
            "computed": False,
            "reason": "disabled_or_degenerate_primary_observable",
        }

    complete_fraction = float(np.mean(first_commit >= 0)) if node_count else 0.0
    primary_distinct = int(np.unique(first_commit[first_commit >= 0]).size)
    final_phi = trace.get("final_phi")
    checks = {
        "artifact_shapes_valid": True,
        "primary_observable_fixed_before_analysis": True,
        "all_patches_committed": bool(complete_fraction == 1.0),
        "primary_observable_nondegenerate": bool(primary_distinct >= 2),
        "final_covariant_mismatch_zero": bool(final_phi == 0),
        "no_measurement_input_interface": True,
    }
    diagnostic_eligible = bool(all(checks.values()))

    input_rows = [
        _input_receipt(event_path, role="run_produced_event_times"),
    ]
    trace_path = root / "mismatch_trace.csv"
    if trace_path.is_file():
        input_rows.append(_input_receipt(trace_path, role="run_produced_global_trace"))
    config_path = root / "config.yml"
    if config_path.is_file():
        input_rows.append(_input_receipt(config_path, role="frozen_run_config"))
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        input_rows.append(_input_receipt(manifest_path, role="run_manifest"))

    report = {
        "schema": SCHEMA,
        "classification": "INTERNAL_DIAGNOSTIC_ONLY",
        "run_dir": str(root),
        "primary_observable": PRIMARY_OBSERVABLE,
        "primary_semantics": (
            "first finite repair cycle in which the patch obtains a completed "
            "observer-visible record under the configured stability rule"
        ),
        "node_count": node_count,
        "seam_count": int(left.size),
        "run_cycle_count": int(cycles),
        "completed_record_fraction": complete_fraction,
        "censored_patch_count": int(np.count_nonzero(censored)),
        "diagnostic_eligible": diagnostic_eligible,
        "eligibility_checks": checks,
        "distribution": distribution,
        "angular_geometry": angular_geometry,
        "seam_association_and_information": seam_association,
        "graph_shell_correlation": graph_shell_correlation,
        "angular_correlation": angular_correlation,
        "late_record_morphology": morphology,
        "multiscale_non_gaussianity": multiscale,
        "global_settling_arrow": trace,
        "independent_seam_null": independent_seam_null,
        "factorized_graph_scheduler_control": factorized_graph_control,
        "angular_spectrum": angular_spectrum,
        "secondary_observables": {
            "first_repair_cycle": _censored_summary(first_repair),
            "last_repair_cycle": _censored_summary(last_repair),
            "first_quiescence_cycle": _censored_summary(first_quiescence),
            "cumulative_repair_load": _finite_summary(cumulative_repair),
            "final_mismatch_density": _finite_summary(final_mismatch),
            **{
                name: (
                    _finite_summary(values)
                    if name == "commit_revocation_count"
                    else _censored_summary(values)
                )
                for name, values in optional_node_events.items()
            },
        },
        "input_custody": {
            "inputs": input_rows,
            "measurement_files_read": [],
            "target_data_read": False,
            "public_data_comparison_performed": False,
        },
        "inside_run_interpretation": (
            "Each event time is recoverable from a patch's bounded record history. "
            "The cross-patch statistics are consensus-level analyses of those public "
            "records; they do not inspect a hidden physical coordinate or target table."
        ),
        "source_and_scheduler_warning": (
            "The current finite engine samples active seam repairs with a global seeded "
            "scheduler and consumes a supplied protected authority order. Spatial "
            "structure in these diagnostics may therefore characterize that source and "
            "scheduler rather than autonomous early-universe dynamics. The shuffle, "
            "Gaussian, refinement, seed, and alternate-authority controls must be read "
            "with the primary result."
        ),
        "physical_bridge_status": {
            "physical_clock": False,
            "redshift_or_scale_factor": False,
            "temperature": False,
            "physical_wavenumber": False,
            "bulk_matter_density": False,
            "particle_identification": False,
            "public_likelihood": False,
        },
        "physical_early_universe_claim": False,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Dimensionless and screen-angular statistics of observer-visible finite "
            "record formation. They can falsify or characterize a proposed finite OPH "
            "source/scheduler. They do not establish cosmological time, temperature, "
            "matter, CMB, or particle observables without separate physical bridges."
        ),
    }
    return strict_jsonable(report)


def write_internal_early_universe_report(
    run_dir: str | Path,
    out: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    report = internal_early_universe_report(run_dir, **kwargs)
    destination = Path(out) if out is not None else Path(run_dir)
    if destination.suffix.lower() == ".json":
        out_dir = destination.parent
        json_path = destination
    else:
        out_dir = destination
        json_path = out_dir / "internal_early_universe_diagnostics.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "internal_early_universe_diagnostics.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    _write_rows(
        out_dir / "internal_angular_correlation.csv",
        list((report.get("angular_correlation") or {}).get("bins") or []),
    )
    _write_rows(
        out_dir / "internal_late_record_morphology.csv",
        list((report.get("late_record_morphology") or {}).get("threshold_rows") or []),
    )
    _write_rows(
        out_dir / "internal_graph_shell_correlation.csv",
        list((report.get("graph_shell_correlation") or {}).get("rows") or []),
    )
    _write_rows(
        out_dir / "internal_independent_seam_null.csv",
        list((report.get("independent_seam_null") or {}).get("cycle_rows") or []),
    )
    return report


def _validate_geometry(
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
) -> None:
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("points must have shape (N, 3) with N > 0")
    if not np.all(np.isfinite(points)):
        raise ValueError("points contain non-finite values")
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("edge endpoint arrays must be matching vectors")
    if weights.shape != (points.shape[0],) or not np.all(np.isfinite(weights)):
        raise ValueError("cell entropy weights must be one finite value per point")
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("cell entropy weights must have positive total mass")
    if left.size and (
        int(min(left.min(), right.min())) < 0
        or int(max(left.max(), right.max())) >= points.shape[0]
    ):
        raise ValueError("edge endpoint outside point array")
    radii = np.linalg.norm(points, axis=1)
    if float(np.max(np.abs(radii - 1.0))) > 1.0e-4:
        raise ValueError("screen points are not on the unit sphere")


def _normalized_weights(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    total = float(np.sum(weights))
    return weights / total


def _standardize(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    field = np.asarray(values, dtype=float)
    mean = float(np.sum(weights * field))
    variance = float(np.sum(weights * (field - mean) ** 2))
    if variance <= 1.0e-30:
        return np.zeros_like(field)
    return (field - mean) / math.sqrt(variance)


def _distribution_report(values: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    standardized = _standardize(values, weights)
    mean = float(np.sum(weights * values))
    variance = float(np.sum(weights * (values - mean) ** 2))
    unique, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(float) / max(1, int(np.sum(counts)))
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return {
        "mean_cycle": mean,
        "std_cycles": math.sqrt(max(0.0, variance)),
        "minimum_cycle": float(np.min(values)),
        "maximum_cycle": float(np.max(values)),
        "median_cycle": float(np.median(values)),
        "quantiles": {
            str(q): float(np.quantile(values, q)) for q in (0.05, 0.25, 0.50, 0.75, 0.95)
        },
        "skewness": float(np.sum(weights * standardized**3)),
        "excess_kurtosis": float(np.sum(weights * standardized**4) - 3.0),
        "discrete_cycle_entropy_nats": entropy,
        "distinct_cycle_count": int(unique.size),
    }


def _angular_geometry_report(
    points: np.ndarray,
    field: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    unit = points / np.linalg.norm(points, axis=1, keepdims=True)
    dipole = np.sum(weights[:, None] * field[:, None] * unit, axis=0)
    identity = np.eye(3) / 3.0
    quadrupole = np.einsum(
        "n,n,nij->ij",
        weights,
        field,
        np.einsum("ni,nj->nij", unit, unit) - identity,
        optimize=True,
    )
    eigenvalues = np.linalg.eigvalsh(quadrupole)
    return {
        "dipole_vector": [float(value) for value in dipole],
        "dipole_amplitude": float(np.linalg.norm(dipole)),
        "quadrupole_eigenvalues": [float(value) for value in eigenvalues],
        "quadrupole_frobenius_amplitude": float(np.linalg.norm(quadrupole)),
        "units": "dimensionless standardized field on the unit screen",
    }


def _seam_association_report(
    field: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    pair_sample_limit: int,
    shuffle_draws: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if left.size == 0:
        return {"available": False, "reason": "no_seams"}
    limit = max(1, min(int(pair_sample_limit), int(left.size)))
    if left.size > limit:
        selected = np.sort(rng.choice(left.size, size=limit, replace=False))
        seam_left = left[selected]
        seam_right = right[selected]
    else:
        seam_left = left
        seam_right = right
    observed_corr = float(np.mean(field[seam_left] * field[seam_right]))
    codes, bin_count = _quantile_codes(field, max_bins=8)
    observed_mi = _pair_mutual_information(
        codes[seam_left], codes[seam_right], bin_count
    )

    random_left = rng.integers(0, field.size, size=seam_left.size)
    random_right = rng.integers(0, field.size, size=seam_left.size)
    random_corr = float(np.mean(field[random_left] * field[random_right]))
    random_mi = _pair_mutual_information(
        codes[random_left], codes[random_right], bin_count
    )
    shuffle_corr: list[float] = []
    shuffle_mi: list[float] = []
    for _ in range(max(1, int(shuffle_draws))):
        shuffled = rng.permutation(field)
        shuffled_codes = codes[rng.permutation(codes.size)]
        shuffle_corr.append(float(np.mean(shuffled[seam_left] * shuffled[seam_right])))
        shuffle_mi.append(
            _pair_mutual_information(
                shuffled_codes[seam_left], shuffled_codes[seam_right], bin_count
            )
        )
    corr_null_mean = float(np.mean(shuffle_corr))
    corr_null_std = float(np.std(shuffle_corr, ddof=1)) if len(shuffle_corr) > 1 else 0.0
    mi_null_mean = float(np.mean(shuffle_mi))
    mi_null_std = float(np.std(shuffle_mi, ddof=1)) if len(shuffle_mi) > 1 else 0.0
    corr_z = (
        (observed_corr - corr_null_mean) / corr_null_std
        if corr_null_std > 0.0
        else None
    )
    mi_z = (
        (observed_mi - mi_null_mean) / mi_null_std if mi_null_std > 0.0 else None
    )
    return {
        "available": True,
        "sampled_seam_count": int(seam_left.size),
        "seam_correlation": observed_corr,
        "random_pair_correlation": random_corr,
        "seam_mutual_information_nats": observed_mi,
        "random_pair_mutual_information_nats": random_mi,
        "quantile_bin_count": int(bin_count),
        "shuffle_draws": int(len(shuffle_corr)),
        "shuffle_seam_correlation_mean": corr_null_mean,
        "shuffle_seam_correlation_std": corr_null_std,
        "seam_correlation_z_vs_shuffle": corr_z,
        "shuffle_mutual_information_mean_nats": mi_null_mean,
        "shuffle_mutual_information_std_nats": mi_null_std,
        "mutual_information_z_vs_shuffle": mi_z,
        "local_structure_detected_at_four_sigma": bool(
            corr_z is not None and mi_z is not None and corr_z >= 4.0 and mi_z >= 4.0
        ),
        "interpretation": (
            "Tests whether neighboring patches complete records together more often "
            "than distribution-matched shuffles. Positive separation is an internal "
            "locality diagnostic, not a physical causal-speed measurement."
        ),
    }


def _quantile_codes(values: np.ndarray, *, max_bins: int) -> tuple[np.ndarray, int]:
    field = np.asarray(values, dtype=float)
    unique = np.unique(field)
    if unique.size <= max_bins:
        return np.searchsorted(unique, field).astype(np.int16), int(unique.size)
    edges = np.unique(np.quantile(field, np.linspace(0.0, 1.0, max_bins + 1)))
    if edges.size <= 2:
        return np.zeros(field.size, dtype=np.int16), 1
    codes = np.searchsorted(edges[1:-1], field, side="right").astype(np.int16)
    return codes, int(edges.size - 1)


def _graph_shell_correlation_report(
    field: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    anchor_count: int,
    max_hops: int,
    shuffle_draws: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Estimate equal-cycle correlation on exact dual-graph hop shells."""

    node_count = int(field.size)
    if node_count == 0 or left.size == 0:
        return {"available": False, "reason": "empty_graph"}
    degree = np.bincount(
        np.concatenate((left, right)), minlength=node_count
    ).astype(np.int16)
    max_degree = int(np.max(degree))
    neighbors = np.full((node_count, max_degree), -1, dtype=np.int32)
    cursor = np.zeros(node_count, dtype=np.int16)
    for source, target in zip(left, right, strict=True):
        a = int(source)
        b = int(target)
        neighbors[a, int(cursor[a])] = b
        cursor[a] += 1
        neighbors[b, int(cursor[b])] = a
        cursor[b] += 1
    anchors = np.sort(
        rng.choice(
            node_count,
            size=min(max(1, int(anchor_count)), node_count),
            replace=False,
        )
    )
    pair_left: list[int] = []
    pair_right: list[int] = []
    pair_hop: list[int] = []
    for anchor in anchors:
        visited = {int(anchor)}
        frontier = {int(anchor)}
        for hop in range(1, max(1, int(max_hops)) + 1):
            next_frontier: set[int] = set()
            for node in frontier:
                for neighbor in neighbors[node]:
                    value = int(neighbor)
                    if value >= 0 and value not in visited:
                        next_frontier.add(value)
            if not next_frontier:
                break
            visited.update(next_frontier)
            pair_left.extend([int(anchor)] * len(next_frontier))
            pair_right.extend(sorted(next_frontier))
            pair_hop.extend([int(hop)] * len(next_frontier))
            frontier = next_frontier
    first = np.asarray(pair_left, dtype=np.int32)
    second = np.asarray(pair_right, dtype=np.int32)
    hops = np.asarray(pair_hop, dtype=np.int16)
    observed_products = field[first] * field[second]
    controls: list[np.ndarray] = []
    control_zero_lag: list[float] = []
    for _ in range(max(1, int(shuffle_draws))):
        shuffled = rng.permutation(field)
        controls.append(shuffled[first] * shuffled[second])
        control_zero_lag.append(float(np.mean(shuffled[anchors] ** 2)))
    observed_zero_lag = float(np.mean(field[anchors] ** 2))
    zero_lag_null_mean = float(np.mean(control_zero_lag))
    zero_lag_null_std = (
        float(np.std(control_zero_lag, ddof=1))
        if len(control_zero_lag) > 1
        else 0.0
    )
    rows = [
        {
            "hop": 0,
            "pair_count": int(anchors.size),
            "correlation": observed_zero_lag,
            "shuffle_mean": zero_lag_null_mean,
            "shuffle_std": zero_lag_null_std,
            "z_vs_shuffle": (
                (observed_zero_lag - zero_lag_null_mean) / zero_lag_null_std
                if zero_lag_null_std > 0.0
                else None
            ),
        }
    ]
    for hop in range(1, max(1, int(max_hops)) + 1):
        mask = hops == hop
        if not np.any(mask):
            continue
        observed = float(np.mean(observed_products[mask]))
        null_values = np.asarray(
            [float(np.mean(control[mask])) for control in controls], dtype=float
        )
        null_mean = float(np.mean(null_values))
        null_std = float(np.std(null_values, ddof=1)) if null_values.size > 1 else 0.0
        rows.append(
            {
                "hop": int(hop),
                "pair_count": int(np.count_nonzero(mask)),
                "correlation": observed,
                "shuffle_mean": null_mean,
                "shuffle_std": null_std,
                "z_vs_shuffle": (
                    (observed - null_mean) / null_std if null_std > 0.0 else None
                ),
            }
        )
    first_one_over_e = next(
        (
            int(row["hop"])
            for row in rows[1:]
            if row["correlation"] is not None
            and float(row["correlation"])
            <= observed_zero_lag * math.exp(-1.0)
        ),
        None,
    )
    first_zero = next(
        (
            int(row["hop"])
            for row in rows[1:]
            if row["correlation"] is not None and float(row["correlation"]) <= 0.0
        ),
        None,
    )
    return {
        "available": True,
        "anchor_count": int(anchors.size),
        "maximum_hops": int(max_hops),
        "sampled_pair_count": int(first.size),
        "rows": rows,
        "first_one_over_e_crossing_hops": first_one_over_e,
        "first_zero_crossing_hops": first_zero,
        "causal_speed_claim": False,
        "interpretation": (
            "Equal-cycle correlation length in exact dual-graph hops. An unpaired "
            "settling trajectory cannot establish a causal front speed; that requires "
            "a same-seed localized intervention and no-intervention control."
        ),
    }


def _pair_mutual_information(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    if bins <= 1 or x.size == 0:
        return 0.0
    joint = np.bincount(
        np.asarray(x, dtype=np.int64) * int(bins) + np.asarray(y, dtype=np.int64),
        minlength=int(bins) * int(bins),
    ).reshape(int(bins), int(bins)).astype(float)
    joint += joint.T
    total = float(np.sum(joint))
    if total <= 0.0:
        return 0.0
    pxy = joint / total
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    expected = px[:, None] * py[None, :]
    mask = (pxy > 0.0) & (expected > 0.0)
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / expected[mask])))


def _angular_correlation_report(
    points: np.ndarray,
    field: np.ndarray,
    *,
    pair_samples: int,
    rng: np.random.Generator,
    bin_count: int = 18,
) -> dict[str, Any]:
    count = max(1, int(pair_samples))
    first = rng.integers(0, points.shape[0], size=count)
    second = rng.integers(0, points.shape[0], size=count)
    dot = np.einsum("ij,ij->i", points[first], points[second])
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    product = field[first] * field[second]
    edges = np.linspace(0.0, math.pi, int(bin_count) + 1)
    which = np.clip(np.digitize(theta, edges) - 1, 0, int(bin_count) - 1)
    rows = []
    for index in range(int(bin_count)):
        mask = which == index
        rows.append(
            {
                "bin": int(index),
                "theta_min_rad": float(edges[index]),
                "theta_max_rad": float(edges[index + 1]),
                "theta_mid_rad": float((edges[index] + edges[index + 1]) / 2.0),
                "pair_count": int(np.count_nonzero(mask)),
                "correlation": float(np.mean(product[mask])) if np.any(mask) else None,
            }
        )
    positive_rows = [
        row
        for row in rows
        if row["correlation"] is not None and float(row["correlation"]) > 0.0
    ]
    one_over_e = next(
        (
            float(row["theta_mid_rad"])
            for row in rows
            if row["correlation"] is not None
            and float(row["correlation"]) <= math.exp(-1.0)
        ),
        None,
    )
    first_zero = next(
        (
            float(row["theta_mid_rad"])
            for row in rows
            if row["correlation"] is not None and float(row["correlation"]) <= 0.0
        ),
        None,
    )
    return {
        "pair_samples": int(count),
        "bins": rows,
        "positive_bin_count": int(len(positive_rows)),
        "first_one_over_e_crossing_rad": one_over_e,
        "first_zero_crossing_rad": first_zero,
        "units": "radians on the declared unit S2 screen",
        "physical_length_claim": False,
    }


def _morphology_report(
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    rows = []
    for quantile in thresholds:
        if not 0.0 < quantile < 1.0:
            raise ValueError("morphology quantiles must lie strictly between zero and one")
        threshold = float(np.quantile(values, quantile))
        active = values >= threshold
        rows.append(
            {
                "quantile": float(quantile),
                "cycle_threshold": threshold,
                **_component_stats(active, left, right),
            }
        )
    return {
        "field": PRIMARY_OBSERVABLE,
        "excursion": "late-completing superlevel patches",
        "threshold_rows": rows,
        "topology_scope": (
            "connected components and V-E of the induced dual graph; this is a "
            "graph-morphology proxy, not the Euler characteristic of a physical "
            "three-dimensional matter excursion set"
        ),
    }


def _component_stats(
    active: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    active = np.asarray(active, dtype=bool)
    node_count = int(active.size)
    active_ids = np.flatnonzero(active)
    active_count = int(active_ids.size)
    induced = active[left] & active[right]
    boundary = active[left] ^ active[right]
    if active_count == 0:
        return {
            "active_patch_count": 0,
            "active_patch_fraction": 0.0,
            "component_count": 0,
            "largest_component_patch_count": 0,
            "largest_component_fraction_of_active": 0.0,
            "boundary_seam_count": 0,
            "induced_dual_graph_euler_proxy_V_minus_E": 0,
        }
    parent = np.arange(node_count, dtype=np.int32)
    size = np.ones(node_count, dtype=np.int32)

    def find(value: int) -> int:
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    for a, b in zip(left[induced], right[induced], strict=True):
        root_a = find(int(a))
        root_b = find(int(b))
        if root_a == root_b:
            continue
        if int(size[root_a]) < int(size[root_b]):
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        size[root_a] += size[root_b]
    roots = np.asarray([find(int(value)) for value in active_ids], dtype=np.int32)
    _, component_sizes = np.unique(roots, return_counts=True)
    largest = int(np.max(component_sizes)) if component_sizes.size else 0
    return {
        "active_patch_count": active_count,
        "active_patch_fraction": float(active_count / node_count),
        "component_count": int(component_sizes.size),
        "largest_component_patch_count": largest,
        "largest_component_fraction_of_active": float(largest / active_count),
        "boundary_seam_count": int(np.count_nonzero(boundary)),
        "boundary_seam_fraction": (
            float(np.mean(boundary)) if boundary.size else 0.0
        ),
        "induced_seam_count": int(np.count_nonzero(induced)),
        "induced_dual_graph_euler_proxy_V_minus_E": int(
            active_count - np.count_nonzero(induced)
        ),
    }


def _multiscale_moments_report(
    field: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    milestones = (0, 1, 2, 4, 8)
    current = np.asarray(field, dtype=float).copy()
    control = rng.permutation(current)
    rows = []
    degree = np.bincount(
        np.concatenate((left, right)), minlength=field.size
    ).astype(float)
    degree = np.maximum(degree, 1.0)
    for step in range(max(milestones) + 1):
        if step in milestones:
            rows.append(
                {
                    "neighbor_average_steps": int(step),
                    "field": _standardized_moments(current, weights),
                    "fixed_shuffle_control": _standardized_moments(control, weights),
                }
            )
        if step == max(milestones):
            break
        current = _neighbor_average(current, left, right, degree)
        control = _neighbor_average(control, left, right, degree)
    return {
        "rows": rows,
        "smoothing": "one-hop dual-seam neighbor average",
        "claim_boundary": (
            "Multi-scale skewness and excess kurtosis of the internal event field "
            "against one fixed distribution-matched shuffle; no primordial or "
            "laboratory non-Gaussianity identification is made."
        ),
    }


def _neighbor_average(
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    degree: np.ndarray,
) -> np.ndarray:
    neighbor_sum = np.bincount(left, weights=values[right], minlength=values.size)
    neighbor_sum += np.bincount(right, weights=values[left], minlength=values.size)
    return neighbor_sum / degree


def _standardized_moments(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    standardized = _standardize(values, weights)
    return {
        "std": float(
            math.sqrt(max(0.0, np.sum(weights * (values - np.sum(weights * values)) ** 2)))
        ),
        "skewness": float(np.sum(weights * standardized**3)),
        "excess_kurtosis": float(np.sum(weights * standardized**4) - 3.0),
    }


def _independent_seam_null_report(
    first_commit_cycle: np.ndarray,
    initial_incident_mismatch_count: np.ndarray,
    *,
    record_commit_cycles: int,
    trace: dict[str, Any],
) -> dict[str, Any]:
    """Exact one-point null for uniform settling of independent active seams.

    Conditional on the initial number of active incident seams at a patch, a
    uniform active-edge scheduler assigns those seams uniformly to positions
    in one random ordering of all initial mismatches.  The distribution of the
    patch's final repair batch is therefore an order-statistic calculation;
    the fixed record-stability delay then translates it to first-commit time.
    """

    first_commit = np.asarray(first_commit_cycle, dtype=np.int64)
    incident = np.asarray(initial_incident_mismatch_count, dtype=np.int64)
    if first_commit.shape != incident.shape:
        raise ValueError("independent-seam null arrays must have matching shapes")
    if not bool(trace.get("available", False)):
        return {"available": False, "reason": "global_trace_unavailable"}

    initial_phi = int(trace.get("initial_phi", -1))
    chosen = np.asarray(trace.get("chosen_edges_by_cycle", []), dtype=np.int64)
    repair_budget = np.asarray(
        trace.get("repair_budget_by_cycle", []), dtype=np.int64
    )
    checks = {
        "all_records_uncensored": bool(np.all(first_commit >= 0)),
        "initial_incidence_counts_each_active_seam_twice": bool(
            initial_phi >= 0 and int(np.sum(incident)) == 2 * initial_phi
        ),
        "selected_repairs_exactly_match_phi_drop": bool(
            trace.get("selected_repairs_exactly_match_phi_drop", False)
        ),
        "no_observer_readback_writes": bool(
            int(trace.get("observer_readback_write_count", 0)) == 0
        ),
        "no_sector_link_writes": bool(
            int(trace.get("sector_link_write_count", 0)) == 0
        ),
        "scheduler_trace_present": bool(chosen.size > 0 and chosen.size == repair_budget.size),
        "all_initial_mismatches_repaired": bool(
            initial_phi >= 0 and int(np.sum(chosen)) == initial_phi
        ),
        "positive_record_stability_threshold": bool(int(record_commit_cycles) >= 1),
    }
    if not all(checks.values()):
        return {
            "available": False,
            "reason": "independent_seam_null_assumptions_not_met",
            "assumption_checks": checks,
        }

    cumulative_selected = np.minimum(np.cumsum(chosen), initial_phi)
    predicted_counts: dict[int, float] = {}
    incident_values, incident_counts = np.unique(incident, return_counts=True)
    for incident_count_raw, patch_count_raw in zip(
        incident_values, incident_counts, strict=True
    ):
        k = int(incident_count_raw)
        patch_count = int(patch_count_raw)
        if k < 0 or k > initial_phi:
            return {
                "available": False,
                "reason": "invalid_initial_incident_mismatch_count",
                "assumption_checks": checks,
            }
        if k == 0:
            commit_cycle = int(record_commit_cycles) - 1
            predicted_counts[commit_cycle] = (
                predicted_counts.get(commit_cycle, 0.0) + patch_count
            )
            continue
        previous_cdf = 0.0
        for repair_cycle, selected_total in enumerate(cumulative_selected):
            cdf = _combination_ratio(int(selected_total), k, initial_phi)
            probability = max(0.0, cdf - previous_cdf)
            if probability > 0.0:
                commit_cycle = repair_cycle + int(record_commit_cycles) - 1
                predicted_counts[commit_cycle] = (
                    predicted_counts.get(commit_cycle, 0.0)
                    + patch_count * probability
                )
            previous_cdf = cdf
        if previous_cdf < 1.0 - 1.0e-12:
            return {
                "available": False,
                "reason": "scheduler_trace_does_not_cover_null_support",
                "assumption_checks": checks,
            }

    observed_cycles, observed_counts_raw = np.unique(first_commit, return_counts=True)
    observed_counts = {
        int(cycle): int(count)
        for cycle, count in zip(observed_cycles, observed_counts_raw, strict=True)
    }
    node_count = int(first_commit.size)
    support = sorted(set(predicted_counts) | set(observed_counts))
    rows: list[dict[str, Any]] = []
    absolute_probability_residuals: list[float] = []
    for cycle in support:
        expected_count = float(predicted_counts.get(cycle, 0.0))
        observed_count = int(observed_counts.get(cycle, 0))
        expected_probability = expected_count / max(1, node_count)
        observed_probability = observed_count / max(1, node_count)
        residual = observed_probability - expected_probability
        absolute_probability_residuals.append(abs(residual))
        rows.append(
            {
                "first_commit_cycle": int(cycle),
                "observed_patch_count": observed_count,
                "null_expected_patch_count": expected_count,
                "observed_probability": observed_probability,
                "null_expected_probability": expected_probability,
                "probability_residual": residual,
            }
        )
    predicted_total = float(sum(predicted_counts.values()))
    predicted_mean = (
        float(
            sum(cycle * count for cycle, count in predicted_counts.items())
            / predicted_total
        )
        if predicted_total > 0.0
        else None
    )
    return {
        "available": True,
        "schema": "oph_independent_seam_order_statistic_null_v1",
        "assumption_checks": checks,
        "initial_active_seam_count": initial_phi,
        "record_commit_cycles": int(record_commit_cycles),
        "initial_incident_mismatch_histogram": {
            str(int(value)): int(count)
            for value, count in zip(incident_values, incident_counts, strict=True)
        },
        "observed_mean_first_commit_cycle": float(np.mean(first_commit)),
        "null_expected_mean_first_commit_cycle": predicted_mean,
        "total_variation_distance": 0.5 * float(sum(absolute_probability_residuals)),
        "maximum_absolute_probability_residual": float(
            max(absolute_probability_residuals, default=0.0)
        ),
        "cycle_rows": rows,
        "one_point_distribution_fully_specified_by_factorized_kernel": True,
        "interpretation": (
            "This is the exact conditional one-point law for the current uniform "
            "active-seam settling scheduler. Agreement means that the completed-record "
            "time histogram needs no collective early-universe dynamics. Spatial tests "
            "remain separate because neighboring patches share seams."
        ),
    }


def _combination_ratio(selected: int, subset_size: int, population: int) -> float:
    """Return C(selected, subset_size) / C(population, subset_size)."""

    n = int(selected)
    k = int(subset_size)
    total = int(population)
    if k == 0:
        return 1.0
    if n < k or total < k or n < 0 or total <= 0:
        return 0.0
    ratio = 1.0
    for offset in range(k):
        ratio *= float(n - offset) / float(total - offset)
    return float(min(1.0, max(0.0, ratio)))


def _factorized_graph_control_report(
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    first_commit_cycle: np.ndarray,
    weights: np.ndarray,
    *,
    record_commit_cycles: int,
    trace: dict[str, Any],
    analytic_null: dict[str, Any],
    draws: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Graph-preserving controls for the independent-seam scheduler model."""

    if not bool(analytic_null.get("available", False)):
        return {"available": False, "reason": "analytic_factorized_null_unavailable"}
    edge_count = int(left.size)
    node_count = int(first_commit_cycle.size)
    initial_phi = int(analytic_null.get("initial_active_seam_count", -1))
    budgets = np.asarray(trace.get("repair_budget_by_cycle", []), dtype=np.int64)
    if (
        edge_count <= 0
        or node_count <= 0
        or initial_phi < 0
        or initial_phi > edge_count
        or budgets.size == 0
        or int(np.sum(budgets)) < initial_phi
    ):
        return {"available": False, "reason": "invalid_factorized_control_inputs"}

    cumulative_capacity = np.cumsum(budgets)
    repair_cycles_by_rank = np.searchsorted(
        cumulative_capacity,
        np.arange(1, initial_phi + 1, dtype=np.int64),
        side="left",
    ).astype(np.int32)
    observed = np.asarray(first_commit_cycle, dtype=float)
    observed_standardized = _standardize(observed, weights)
    observed_geometry = _angular_geometry_report(
        points, observed_standardized, weights
    )
    observed_seam_correlation = float(
        np.mean(observed_standardized[left] * observed_standardized[right])
    )
    expected_probabilities = {
        int(row["first_commit_cycle"]): float(row["null_expected_probability"])
        for row in analytic_null.get("cycle_rows", [])
    }

    seam_correlations: list[float] = []
    dipole_amplitudes: list[float] = []
    quadrupole_amplitudes: list[float] = []
    one_point_tv_distances: list[float] = []
    for _ in range(max(1, int(draws))):
        active_edges = rng.choice(edge_count, size=initial_phi, replace=False)
        active_edges = rng.permutation(active_edges)
        last_repair = np.full(node_count, -1, dtype=np.int32)
        np.maximum.at(last_repair, left[active_edges], repair_cycles_by_rank)
        np.maximum.at(last_repair, right[active_edges], repair_cycles_by_rank)
        control_commit = np.where(
            last_repair >= 0,
            last_repair + int(record_commit_cycles) - 1,
            int(record_commit_cycles) - 1,
        ).astype(float)
        standardized = _standardize(control_commit, weights)
        seam_correlations.append(
            float(np.mean(standardized[left] * standardized[right]))
        )
        geometry = _angular_geometry_report(points, standardized, weights)
        dipole_amplitudes.append(float(geometry["dipole_amplitude"]))
        quadrupole_amplitudes.append(
            float(geometry["quadrupole_frobenius_amplitude"])
        )
        cycles, counts = np.unique(control_commit.astype(np.int64), return_counts=True)
        probabilities = {
            int(cycle): float(count / node_count)
            for cycle, count in zip(cycles, counts, strict=True)
        }
        support = set(probabilities) | set(expected_probabilities)
        one_point_tv_distances.append(
            0.5
            * float(
                sum(
                    abs(
                        probabilities.get(cycle, 0.0)
                        - expected_probabilities.get(cycle, 0.0)
                    )
                    for cycle in support
                )
            )
        )

    observed_tv = float(analytic_null.get("total_variation_distance", 0.0))
    seam_summary = _control_distribution_summary(
        seam_correlations, observed_seam_correlation
    )
    dipole_summary = _control_distribution_summary(
        dipole_amplitudes, float(observed_geometry["dipole_amplitude"])
    )
    quadrupole_summary = _control_distribution_summary(
        quadrupole_amplitudes,
        float(observed_geometry["quadrupole_frobenius_amplitude"]),
    )
    tv_summary = _control_distribution_summary(one_point_tv_distances, observed_tv)
    return {
        "available": True,
        "schema": "oph_factorized_graph_scheduler_control_v1",
        "draw_count": int(len(seam_correlations)),
        "conditioning": {
            "fixed_screen_graph": True,
            "fixed_initial_active_seam_count": initial_phi,
            "uniform_active_edge_subset": True,
            "uniform_scheduler_order": True,
            "fixed_repair_budget_schedule": True,
            "fixed_record_commit_cycles": int(record_commit_cycles),
        },
        "seam_correlation": seam_summary,
        "dipole_amplitude": dipole_summary,
        "quadrupole_frobenius_amplitude": quadrupole_summary,
        "one_point_total_variation_distance": tv_summary,
        "observed_within_four_control_standard_deviations": bool(
            all(
                summary.get("absolute_z_vs_control_mean") is None
                or float(summary["absolute_z_vs_control_mean"]) <= 4.0
                for summary in (
                    seam_summary,
                    dipole_summary,
                    quadrupole_summary,
                    tv_summary,
                )
            )
        ),
        "interpretation": (
            "Controls preserve the exact screen graph and repair-budget schedule, "
            "then resample the independent initial active seams and their uniform "
            "repair order. Agreement shows that shared-edge locality and finite-sample "
            "anisotropy do not require collective propagating dynamics."
        ),
    }


def _control_distribution_summary(
    control_values: Iterable[float], observed: float
) -> dict[str, Any]:
    values = np.asarray(tuple(float(value) for value in control_values), dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    z_value = (float(observed) - mean) / std if std > 0.0 else None
    return {
        "observed": float(observed),
        "control_mean": mean,
        "control_std": std,
        "control_minimum": float(np.min(values)),
        "control_maximum": float(np.max(values)),
        "z_vs_control_mean": z_value,
        "absolute_z_vs_control_mean": abs(z_value) if z_value is not None else None,
    }


def _trace_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "reason": "mismatch_trace_missing"}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"available": False, "reason": "mismatch_trace_empty"}
    phi = np.asarray([float(row.get("phi", 0.0)) for row in rows], dtype=float)
    committed = np.asarray(
        [float(row.get("committed_fraction", 0.0)) for row in rows], dtype=float
    )
    entropy = np.asarray(
        [float(row.get("record_packet_entropy", 0.0)) for row in rows], dtype=float
    )
    cycle = np.asarray([int(float(row.get("cycle", index))) for index, row in enumerate(rows)])
    first_phi_before = rows[0].get("phi_before")
    initial_phi = (
        float(first_phi_before)
        if first_phi_before not in (None, "")
        else float(phi[0])
    )
    chosen_edges = np.asarray(
        [int(float(row.get("chosen_edges", 0))) for row in rows], dtype=np.int64
    )
    repair_budget = np.asarray(
        [int(float(row.get("repair_budget", 0))) for row in rows], dtype=np.int64
    )
    phi_before = np.asarray(
        [float(row.get("phi_before", value)) for row, value in zip(rows, phi, strict=True)],
        dtype=float,
    )
    readback_writes = np.asarray(
        [int(float(row.get("observer_readback_drive_edges", 0))) for row in rows],
        dtype=np.int64,
    )
    sector_writes = np.asarray(
        [int(float(row.get("sector_edges_changed", 0))) for row in rows],
        dtype=np.int64,
    )
    repairs_match_drop = bool(
        np.allclose(phi_before - phi, chosen_edges.astype(float), atol=0.0, rtol=0.0)
    )
    return {
        "available": True,
        "sample_count": int(len(rows)),
        "initial_phi": int(initial_phi),
        "final_phi": int(phi[-1]),
        "phi_nonincreasing": bool(np.all(np.diff(phi) <= 0.0)),
        "phi_increase_count": int(np.count_nonzero(np.diff(phi) > 0.0)),
        "first_zero_phi_cycle": _first_cycle(cycle, phi <= 0.0),
        "first_half_committed_cycle": _first_cycle(cycle, committed >= 0.5),
        "first_95pct_committed_cycle": _first_cycle(cycle, committed >= 0.95),
        "record_entropy_initial": float(entropy[0]),
        "record_entropy_final": float(entropy[-1]),
        "record_entropy_delta": float(entropy[-1] - entropy[0]),
        "record_entropy_nondecreasing": bool(np.all(np.diff(entropy) >= -1.0e-12)),
        "chosen_edges_by_cycle": [int(value) for value in chosen_edges],
        "repair_budget_by_cycle": [int(value) for value in repair_budget],
        "selected_repairs_exactly_match_phi_drop": repairs_match_drop,
        "observer_readback_write_count": int(np.sum(readback_writes)),
        "sector_link_write_count": int(np.sum(sector_writes)),
        "repair_cycle_units_only": True,
    }


def _first_cycle(cycles: np.ndarray, mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(cycles[indices[0]]) if indices.size else None


def _censored_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    valid = array[array >= 0]
    return {
        "observed_count": int(valid.size),
        "censored_count": int(array.size - valid.size),
        "observed_fraction": float(valid.size / array.size) if array.size else 0.0,
        **(_finite_summary(valid.astype(float)) if valid.size else {}),
    }


def _finite_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "minimum": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
        "std": float(np.std(finite)),
        "distinct_value_count": int(np.unique(finite).size),
    }


def _input_receipt(path: Path, *, role: str) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "role": role,
        "byte_count": int(path.stat().st_size),
        "sha256": "sha256:" + digest,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(report: dict[str, Any]) -> str:
    seam = report.get("seam_association_and_information", {}) or {}
    graph_shell = report.get("graph_shell_correlation", {}) or {}
    factorized_null = report.get("independent_seam_null", {}) or {}
    graph_control = report.get("factorized_graph_scheduler_control", {}) or {}
    distribution = report.get("distribution", {}) or {}
    trace = report.get("global_settling_arrow", {}) or {}
    return "\n".join(
        [
            "# Observer-internal early-run diagnostics",
            "",
            f"- Classification: `{report.get('classification')}`",
            f"- Diagnostic eligible: `{report.get('diagnostic_eligible')}`",
            f"- Primary observable: `{report.get('primary_observable')}`",
            f"- Patches / seams: `{report.get('node_count')}` / `{report.get('seam_count')}`",
            f"- Completed-record fraction: `{report.get('completed_record_fraction')}`",
            f"- Primary cycle standard deviation: `{distribution.get('std_cycles')}`",
            f"- Seam correlation z versus shuffle: `{seam.get('seam_correlation_z_vs_shuffle')}`",
            f"- Seam information z versus shuffle: `{seam.get('mutual_information_z_vs_shuffle')}`",
            (
                "- First graph-shell zero crossing (hops): "
                f"`{graph_shell.get('first_zero_crossing_hops')}`"
            ),
            f"- Factorized-kernel null available: `{factorized_null.get('available')}`",
            (
                "- Factorized-kernel null mean cycle: "
                f"`{factorized_null.get('null_expected_mean_first_commit_cycle')}`"
            ),
            (
                "- One-point total-variation distance from null: "
                f"`{factorized_null.get('total_variation_distance')}`"
            ),
            (
                "- Graph/scheduler control passed at four sigma: "
                f"`{graph_control.get('observed_within_four_control_standard_deviations')}`"
            ),
            f"- First zero-mismatch cycle: `{trace.get('first_zero_phi_cycle')}`",
            "",
            (
                "These are dimensionless finite-screen and repair-cycle diagnostics. "
                "No mapping to physical time, temperature, redshift, matter density, "
                "CMB temperature, or particles is asserted."
            ),
            "",
            str(report.get("source_and_scheduler_warning", "")),
            "",
        ]
    )
