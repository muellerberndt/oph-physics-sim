from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import numpy as np

from oph_fpe.bulk.cap_geometry import RoundCap, cap_weights
from oph_fpe.bulk.modular_response_kernel import (
    _bounded_ordered_thread_map,
    _production_sector_replay_contract,
    _simulate_cap_collar_perturb_resettle,
)
from oph_fpe.evidence.hashes import CANONICAL_HASH_SCHEMA, stable_json_hash


DEFAULT_PAIRED_B_A_CONTROLS = (
    "no_perturbation",
    "no_repair_load_channel",
    "baryon_delta_applied_after_record_freezeout",
    "phase_shuffled_baryon_mode",
    "random_collar_labels",
    "wrong_k_label",
)

NO_FULL_GRAPH_SIMULATION_CONTROLS = frozenset(
    {
        "no_perturbation",
        "no_repair_load_channel",
        "baryon_delta_applied_after_record_freezeout",
    }
)


@dataclass(frozen=True)
class PhysicalSourceIntervention:
    background_hash: str
    source_vector_id: str
    tangent_vector: list[float]
    constraint_matrix_hash: str
    retraction_id: str
    delivered_source_vector: list[float]
    constraint_residuals: dict[str, float]
    physical_source_intervention: bool = False


def paired_perturb_resettle_b_a_report(
    points: np.ndarray,
    caps: list[RoundCap],
    raw_fields: dict[str, np.ndarray],
    graph_state: dict[str, Any],
    *,
    cell_entropy: np.ndarray | float | None = None,
    observer_views: list[dict[str, Any]] | None = None,
    a_grid: Iterable[float] | None = None,
    times: Iterable[float] | None = None,
    max_caps: int | None = None,
    modes_per_cap_time: int = 2,
    controls: Iterable[str] | None = DEFAULT_PAIRED_B_A_CONTROLS,
    response_field: str = "cumulative_repair_load",
    perturb_strength: float = 1.0,
    perturb_budget_mode: str = "modular_amount",
    fixed_perturb_fraction: float | None = None,
    perturb_selection_mode: str = "lambda_collar_generator",
    repair_steps: int = 4,
    repairs_per_step: int = 64,
    transition_scale: float = 2.0 * math.pi,
    max_full_graph_simulations: int | None = None,
    full_graph_budget_policy: str = "skip_if_exceeded",
    full_graph_n_jobs: int = 1,
    reuse_dynamics_across_a_grid: bool = True,
    seed: int = 1,
) -> dict[str, Any]:
    """Estimate a non-fit B_A parent from actual paired perturb/resettle probes.

    This is still a diagnostic: the baryon mode is a declared cap/collar source
    on the finite regulator, and the response is a repair-density parent
    functional. The important improvement over report-backed surrogates is that
    every main row is built from plus/minus finite screen perturbation reruns.
    """

    points = np.asarray(points, dtype=float)
    graph = _validated_graph_state(graph_state, points.shape[0])
    cap_values = list(caps)
    if max_caps is not None:
        cap_values = cap_values[: max(0, int(max_caps))]
    time_values = [float(value) for value in (times if times is not None else [0.025, 0.05, 0.1])]
    a_values = [float(value) for value in (a_grid if a_grid is not None else [1.0 / 1100.0, 0.01, 0.1, 1.0])]
    entropy = _cell_entropy(cell_entropy, points.shape[0])
    observer_context = _observer_probe_context(
        observer_views,
        entropy=entropy,
        patch_count=points.shape[0],
    )
    seed_base = int(seed)

    rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    observer_response_rows: list[dict[str, Any]] = []
    observer_response_control_rows: list[dict[str, Any]] = []
    control_values = tuple(DEFAULT_PAIRED_B_A_CONTROLS if controls is None else controls)
    mode_count = max(1, int(modes_per_cap_time))
    live_control_count = sum(
        str(control) not in NO_FULL_GRAPH_SIMULATION_CONTROLS
        for control in control_values
    )
    unique_full_graph_simulations = (
        2
        * len(cap_values)
        * len(time_values)
        * mode_count
        * (1 + live_control_count)
    )
    unreused_full_graph_simulations = unique_full_graph_simulations * len(a_values)
    planned_full_graph_simulations = (
        unique_full_graph_simulations
        if bool(reuse_dynamics_across_a_grid) and a_values
        else unreused_full_graph_simulations
    )
    requested_n_jobs = max(1, int(full_graph_n_jobs))
    unique_simulation_task_count = (
        len(cap_values)
        * len(time_values)
        * (1 + live_control_count)
    )
    planned_simulation_task_count = (
        unique_simulation_task_count
        if bool(reuse_dynamics_across_a_grid) and a_values
        else unique_simulation_task_count * len(a_values)
    )
    effective_n_jobs = min(
        requested_n_jobs,
        max(1, int(planned_simulation_task_count)),
    )
    parallel_execution = {
        "schema": "bounded_ordered_full_graph_thread_execution_v1",
        "requested_n_jobs": int(requested_n_jobs),
        "effective_n_jobs": int(effective_n_jobs),
        "executor": (
            "bounded_ordered_thread_pool" if effective_n_jobs > 1 else "sequential"
        ),
        "simulation_bearing_probe_task_count": int(planned_simulation_task_count),
        "full_graph_simulation_count": int(planned_full_graph_simulations),
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
            "paired B_A full_graph_budget_policy must be 'skip_if_exceeded'"
        )
    sector_replay = _production_sector_replay_contract(graph)
    execution_blockers = list(sector_replay["blockers"])
    if budget is not None and planned_full_graph_simulations > budget:
        execution_blockers.append("paired_full_graph_simulation_budget_exceeded")
    execution_allowed = not execution_blockers
    probe_cache: dict[tuple[Any, ...], dict[str, Any]] | None = (
        {} if bool(reuse_dynamics_across_a_grid) else None
    )
    probe_counter = {"full_graph_simulations": 0, "a_grid_cache_hits": 0}

    cap_contexts: list[dict[str, Any]] = []
    for cap_index, cap in enumerate(cap_values):
        normalized_cap = cap.normalized()
        weights = cap_weights(points, normalized_cap, soft=True) * entropy
        cap_contexts.append(
            {
                "cap_index": int(cap_index),
                "cap": normalized_cap,
                "k_proxy": 1.0 / max(float(normalized_cap.theta0), 1.0e-12),
                "weights": weights,
                "rho_a": _response_scale(raw_fields, response_field, weights),
            }
        )

    def task_specs(grid_values: Iterable[float]) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for a_value in grid_values:
            for cap_context in cap_contexts:
                for time_index, time_value in enumerate(time_values):
                    for control in (None, *control_values):
                        specs.append(
                            {
                                "a_value": float(a_value),
                                "cap_context": cap_context,
                                "time_index": int(time_index),
                                "time_value": float(time_value),
                                "control": control,
                            }
                        )
        return specs

    def execute_task(
        spec: dict[str, Any],
    ) -> tuple[str | None, dict[str, Any], dict[tuple[Any, ...], dict[str, Any]], dict[str, int]]:
        cap_context = spec["cap_context"]
        control = spec["control"]
        local_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        local_counter = {"full_graph_simulations": 0, "a_grid_cache_hits": 0}
        cap_index = int(cap_context["cap_index"])
        time_index = int(spec["time_index"])
        row = _paired_row(
            points,
            cap_context["cap"],
            raw_fields,
            graph,
            cap_context["weights"],
            entropy,
            response_field=response_field,
            rho_a=float(cap_context["rho_a"]),
            a_value=float(spec["a_value"]),
            k_proxy=float(cap_context["k_proxy"]),
            cap_index=cap_index,
            time_index=time_index,
            time_value=float(spec["time_value"]),
            modes_per_cap_time=modes_per_cap_time,
            control=None if control is None else str(control),
            perturb_strength=perturb_strength,
            perturb_budget_mode=perturb_budget_mode,
            fixed_perturb_fraction=fixed_perturb_fraction,
            perturb_selection_mode=perturb_selection_mode,
            repair_steps=repair_steps,
            repairs_per_step=repairs_per_step,
            transition_scale=transition_scale,
            seed=(
                seed_base + 1009 * cap_index + 9173 * time_index
                if control is None
                else seed_base + 31_337 + 1009 * cap_index + 9173 * time_index
            ),
            observer_context=observer_context,
            probe_cache=local_cache if bool(reuse_dynamics_across_a_grid) else None,
            probe_execution_counter=local_counter,
        )
        return control, row, local_cache, local_counter

    def record_task_result(
        result: tuple[
            str | None,
            dict[str, Any],
            dict[tuple[Any, ...], dict[str, Any]],
            dict[str, int],
        ],
    ) -> None:
        control, row, local_cache, local_counter = result
        if probe_cache is not None:
            for cache_key, cache_value in local_cache.items():
                if cache_key in probe_cache:
                    raise RuntimeError("duplicate paired full-graph probe cache key")
                probe_cache[cache_key] = cache_value
        probe_counter["full_graph_simulations"] += int(
            local_counter["full_graph_simulations"]
        )
        probe_counter["a_grid_cache_hits"] += int(local_counter["a_grid_cache_hits"])
        if control is None:
            observer_response_rows.append(_pop_observer_response(row))
            rows.append(row)
        else:
            observer_response_control_rows.append(_pop_observer_response(row))
            control_rows.append(row)

    if execution_allowed:
        first_grid = a_values[:1] if bool(reuse_dynamics_across_a_grid) else a_values
        for result in _bounded_ordered_thread_map(
            execute_task,
            task_specs(first_grid),
            max_workers=effective_n_jobs,
        ):
            record_task_result(result)

        # With reuse enabled, every later a-grid row is a compact deterministic
        # read from the first-grid primitive cache.  This preserves the old row
        # ordering and cache-hit accounting without repeating whole-graph work.
        for a_value in a_values[1:] if bool(reuse_dynamics_across_a_grid) else []:
            for spec in task_specs([a_value]):
                cap_context = spec["cap_context"]
                control = spec["control"]
                cap_index = int(cap_context["cap_index"])
                time_index = int(spec["time_index"])
                row = _paired_row(
                    points,
                    cap_context["cap"],
                    raw_fields,
                    graph,
                    cap_context["weights"],
                    entropy,
                    response_field=response_field,
                    rho_a=float(cap_context["rho_a"]),
                    a_value=float(a_value),
                    k_proxy=float(cap_context["k_proxy"]),
                    cap_index=cap_index,
                    time_index=time_index,
                    time_value=float(spec["time_value"]),
                    modes_per_cap_time=modes_per_cap_time,
                    control=None if control is None else str(control),
                    perturb_strength=perturb_strength,
                    perturb_budget_mode=perturb_budget_mode,
                    fixed_perturb_fraction=fixed_perturb_fraction,
                    perturb_selection_mode=perturb_selection_mode,
                    repair_steps=repair_steps,
                    repairs_per_step=repairs_per_step,
                    transition_scale=transition_scale,
                    seed=(
                        seed_base + 1009 * cap_index + 9173 * time_index
                        if control is None
                        else seed_base + 31_337 + 1009 * cap_index + 9173 * time_index
                    ),
                    observer_context=observer_context,
                    probe_cache=probe_cache,
                    probe_execution_counter=probe_counter,
                )
                if control is None:
                    observer_response_rows.append(_pop_observer_response(row))
                    rows.append(row)
                else:
                    observer_response_control_rows.append(_pop_observer_response(row))
                    control_rows.append(row)

    readiness = _readiness(rows, control_rows)
    observer_geometry = _observer_geometry_response_rows(
        observer_context,
        observer_response_rows,
        observer_response_control_rows,
    )
    report = {
        "mode": "paired_cap_collar_perturb_resettle_B_A_parent_v0",
        "primary_parent_source": "paired_cap_collar_perturb_resettle_rerun",
        "normalization": "EQUILIBRIUM_CONTRAST_DIAGNOSTIC",
        "response_numerator": "paired_delta_response_field",
        "source_variable": "ANOMALY_FRAME_BARYON_CONTRAST_PROXY",
        "denominator": "RHO_A_EQ_BACKGROUND_DIAGNOSTIC",
        "source_report_count": 0,
        "observer_view_source_count": int(observer_geometry["producer_receipt"]),
        "paired_perturbation_source_count": int(bool(rows)),
        "a_grid": a_values,
        "k_grid_proxy_inverse_theta": sorted({float(row["k_proxy_inverse_theta"]) for row in rows}) if rows else [],
        "times": time_values,
        "response_field": str(response_field),
        "perturb_strength": float(perturb_strength),
        "perturb_budget_mode": str(perturb_budget_mode),
        "fixed_perturb_fraction": float(fixed_perturb_fraction) if fixed_perturb_fraction is not None else None,
        "perturb_selection_mode": str(perturb_selection_mode),
        "repair_steps": int(repair_steps),
        "repairs_per_step": int(repairs_per_step),
        "transition_scale": float(transition_scale),
        "modes_per_cap_time": int(modes_per_cap_time),
        "production_move_contract": sector_replay,
        "execution_status": "completed" if execution_allowed else "skipped",
        "execution_blockers": execution_blockers,
        "parallel_execution": (
            parallel_execution
            if execution_allowed
            else {
                **parallel_execution,
                "effective_n_jobs": 0,
                "max_in_flight_full_graph_states": 0,
            }
        ),
        "full_graph_simulation_budget": {
            "schema": "paired_full_graph_simulation_budget_v1",
            "policy": policy,
            "max_full_graph_simulations": budget,
            "requested_without_a_grid_reuse": int(unreused_full_graph_simulations),
            "planned_with_a_grid_reuse": int(planned_full_graph_simulations),
            "unique_centered_probe_simulations": int(unique_full_graph_simulations),
            "executed_full_graph_simulations": int(probe_counter["full_graph_simulations"]),
            "a_grid_cache_hits": int(probe_counter["a_grid_cache_hits"]),
            "reuse_dynamics_across_a_grid": bool(reuse_dynamics_across_a_grid),
            "live_control_count": int(live_control_count),
            "analytic_zero_control_count": int(len(control_values) - live_control_count),
            "receipt": bool(
                execution_allowed
                and int(probe_counter["full_graph_simulations"])
                == int(planned_full_graph_simulations)
            ),
        },
        "rows": rows,
        "control_rows": control_rows,
        "paired_perturbation_rows": rows,
        "paired_perturbation_control_rows": control_rows,
        "source_intervention_schema": list(PhysicalSourceIntervention.__dataclass_fields__),
        "source_intervention_hash_schema": CANONICAL_HASH_SCHEMA,
        "observer_view_rows": observer_geometry["observer_view_rows"],
        "observer_view_control_rows": observer_geometry["observer_view_control_rows"],
        "observer_response_feature_manifest": observer_geometry["feature_manifest"],
        "observer_response_control_manifest": observer_geometry["control_manifest"],
        "observer_response_producer": observer_geometry["producer"],
        "observer_response_producer_blockers": observer_geometry["blockers"],
        "observer_response_cap_relabel_invariance": observer_geometry["cap_relabel_invariance"],
        "observer_response_nondegenerate": observer_geometry["response_nondegenerate"],
        "observer_response_no_perturbation_control_separation_receipt": observer_geometry[
            "no_perturbation_control_separation_receipt"
        ],
        "paired_perturbation_response_producer_receipt": observer_geometry["producer_receipt"],
        "stress_report_surrogate_rows": [],
        "stress_report_surrogate_control_rows": [],
        "readiness": readiness,
        "B_A_PAIRED_DIAGNOSTIC_RECEIPT": bool(readiness.get("B_A_PAIRED_DIAGNOSTIC_RECEIPT", False)),
        "B_A_PARENT_RECEIPT": False,
        "physical_prediction_ready": False,
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "claim_boundary": (
            "Actual paired finite cap/collar perturb-resettle B_A parent diagnostic. "
            "No CMB data are used. Rows exercise finite screen repair dynamics, but "
            "they are not physical Boltzmann kernels until a common source functional, "
            "admissible tangent, lift-independent source vector, calibrated k/a units, "
            "exchange and gauge closure, and derivative-level refinement pass."
        ),
    }
    report["observer_geometry_attachment"] = attach_paired_perturbation_features(
        observer_views,
        report,
    )
    return report


def write_paired_perturb_resettle_b_a_report(
    out_dir: Path,
    points: np.ndarray,
    caps: list[RoundCap],
    raw_fields: dict[str, np.ndarray],
    graph_state: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    report = paired_perturb_resettle_b_a_report(points, caps, raw_fields, graph_state, **kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "paired_b_a_perturbation_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "paired_b_a_perturbation_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "paired_b_a_perturbation_rows.csv", report["rows"])
    _write_csv(out / "paired_b_a_perturbation_control_rows.csv", report["control_rows"])
    # Reuse the existing B_A parent report contract so measurement-pack and
    # comparable-data tooling can consume the stronger source without a
    # separate post-processing command.
    (out / "b_a_parent_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "b_a_parent_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "b_a_parent_rows.csv", report["rows"])
    _write_csv(out / "b_a_parent_control_rows.csv", report["control_rows"])
    return report


def _paired_row(
    points: np.ndarray,
    cap: RoundCap,
    raw_fields: dict[str, np.ndarray],
    graph: dict[str, Any],
    weights: np.ndarray,
    entropy: np.ndarray,
    *,
    response_field: str,
    rho_a: float,
    a_value: float,
    k_proxy: float,
    cap_index: int,
    time_index: int,
    time_value: float,
    modes_per_cap_time: int,
    control: str | None,
    perturb_strength: float,
    perturb_budget_mode: str,
    fixed_perturb_fraction: float | None,
    perturb_selection_mode: str,
    repair_steps: int,
    repairs_per_step: int,
    transition_scale: float,
    seed: int,
    observer_context: dict[str, Any],
    probe_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
    probe_execution_counter: dict[str, int] | None = None,
) -> dict[str, Any]:
    readout_weights = np.asarray(weights, dtype=float)
    readout_cap = cap
    if control == "random_collar_labels":
        # Mislabel the observer readout collar while keeping the actual
        # perturbation on the declared cap. This tests cap-specific response
        # coherence rather than merely asking whether any random collar can
        # excite repair load somewhere on the screen.
        readout_cap = _randomized_cap(cap, seed + 811)
        readout_weights = cap_weights(points, readout_cap, soft=True) * np.asarray(entropy, dtype=float)
    base_value = _weighted_mean(raw_fields.get(response_field), readout_weights)
    observer_count = int(observer_context.get("observer_count", 0) or 0)
    observer_base_values = _observer_weighted_means(raw_fields.get(response_field), observer_context)
    if control in NO_FULL_GRAPH_SIMULATION_CONTROLS:
        estimates = np.zeros(max(1, int(modes_per_cap_time)), dtype=float)
        plus_values = np.full(estimates.size, base_value, dtype=float)
        minus_values = np.full(estimates.size, base_value, dtype=float)
        requested_delta = _delta_baryon(time_value, perturb_strength, transition_scale)
        delivered_delta = 0.0 if control == "no_perturbation" else requested_delta
        deltas = np.full(estimates.size, delivered_delta, dtype=float)
        effective_k_proxy = float(k_proxy)
        cap_used = cap
        sim_scale = float(transition_scale)
        observer_estimates = np.zeros((estimates.size, observer_count), dtype=float)
        observer_even_estimates = np.zeros((estimates.size, observer_count), dtype=float)
        observer_response_simulated = False
    else:
        effective_k_proxy = float(k_proxy) * (1.7 if control == "wrong_k_label" else 1.0)
        cap_used = cap
        sim_scale = float(transition_scale) * (0.5 if control == "wrong_k_label" else 1.0)
        cache_key = (int(cap_index), int(time_index), str(control), int(seed))
        cached = probe_cache.get(cache_key) if probe_cache is not None else None
        if cached is not None:
            estimates = np.asarray(cached["estimates"], dtype=float)
            plus_values = np.asarray(cached["plus_values"], dtype=float)
            minus_values = np.asarray(cached["minus_values"], dtype=float)
            deltas = np.asarray(cached["deltas"], dtype=float)
            observer_estimates = np.asarray(cached["observer_estimates"], dtype=float)
            observer_even_estimates = np.asarray(cached["observer_even_estimates"], dtype=float)
            observer_response_simulated = bool(cached["observer_response_simulated"])
            intervention_support_hashes = list(cached["intervention_support_hashes"])
            if probe_execution_counter is not None:
                probe_execution_counter["a_grid_cache_hits"] = int(
                    probe_execution_counter.get("a_grid_cache_hits", 0)
                ) + 2 * max(1, int(modes_per_cap_time))
        else:
            estimate_values: list[float] = []
            plus_value_rows: list[float] = []
            minus_value_rows: list[float] = []
            delta_values: list[float] = []
            observer_estimate_rows: list[np.ndarray] = []
            observer_even_rows: list[np.ndarray] = []
            paired_probe_receipts: list[bool] = []
            intervention_support_hashes = []
            for mode_index in range(max(1, int(modes_per_cap_time))):
                local_seed = int(seed) + 65_537 * mode_index
                plus_time = float(time_value)
                minus_time = -float(time_value)
                if control == "phase_shuffled_baryon_mode":
                    rng = np.random.default_rng(local_seed + 9001)
                    plus_time *= float(rng.choice([-1.0, 1.0]))
                    minus_time *= float(rng.choice([-1.0, 1.0]))
                post_plus = _simulate_cap_collar_perturb_resettle(
                    points,
                    cap_used,
                    raw_fields,
                    graph,
                    scale=float(sim_scale),
                    time_value=plus_time,
                    perturb_strength=float(perturb_strength),
                    perturb_budget_mode=str(perturb_budget_mode),
                    fixed_perturb_fraction=fixed_perturb_fraction,
                    perturb_selection_mode=str(perturb_selection_mode),
                    repair_steps=int(repair_steps),
                    repairs_per_step=int(repairs_per_step),
                    seed=local_seed,
                )
                plus_support_hash = str(
                    post_plus.get("_intervention_support_hash", "")
                )
                plus_probe_receipt = bool(
                    post_plus.get("_gauge_covariant_probe_receipt", False)
                )
                plus_inverse_receipt = bool(
                    post_plus.get("_centered_inverse_intervention_receipt", False)
                )
                plus = _weighted_mean(
                    post_plus.get(response_field), readout_weights
                )
                observer_plus = _observer_weighted_means(
                    post_plus.get(response_field), observer_context
                )
                # Do not retain two patch-sized post states per worker.  The
                # compact plus readout is sufficient for the centered pair.
                del post_plus
                post_minus = _simulate_cap_collar_perturb_resettle(
                    points,
                    cap_used,
                    raw_fields,
                    graph,
                    scale=float(sim_scale),
                    time_value=minus_time,
                    perturb_strength=float(perturb_strength),
                    perturb_budget_mode=str(perturb_budget_mode),
                    fixed_perturb_fraction=fixed_perturb_fraction,
                    perturb_selection_mode=str(perturb_selection_mode),
                    repair_steps=int(repair_steps),
                    repairs_per_step=int(repairs_per_step),
                    seed=local_seed,
                )
                minus_support_hash = str(
                    post_minus.get("_intervention_support_hash", "")
                )
                minus_probe_receipt = bool(
                    post_minus.get("_gauge_covariant_probe_receipt", False)
                )
                minus_inverse_receipt = bool(
                    post_minus.get("_centered_inverse_intervention_receipt", False)
                )
                minus = _weighted_mean(
                    post_minus.get(response_field), readout_weights
                )
                observer_minus = _observer_weighted_means(
                    post_minus.get(response_field), observer_context
                )
                del post_minus
                if probe_execution_counter is not None:
                    probe_execution_counter["full_graph_simulations"] = int(
                        probe_execution_counter.get("full_graph_simulations", 0)
                    ) + 2
                matched_centered_pair = bool(
                    plus_time == -minus_time
                    and plus_support_hash
                    and plus_support_hash == minus_support_hash
                )
                paired_probe_receipts.append(
                    bool(
                        plus_probe_receipt
                        and minus_probe_receipt
                        and plus_inverse_receipt
                        and minus_inverse_receipt
                        and matched_centered_pair
                    )
                )
                intervention_support_hashes.append(
                    plus_support_hash if matched_centered_pair else ""
                )
                delta = _delta_baryon(time_value, perturb_strength, transition_scale)
                derivative = (plus - minus) / (2.0 * max(delta, 1.0e-12))
                observer_derivative = (observer_plus - observer_minus) / (
                    2.0 * max(delta, 1.0e-12)
                )
                observer_even_response = (
                    observer_plus + observer_minus - 2.0 * observer_base_values
                ) / (2.0 * max(abs(delta), 1.0e-12))
                estimate_values.append(
                    float(derivative / max(abs(float(base_value)), 1.0e-12))
                )
                observer_estimate_rows.append(observer_derivative)
                observer_even_rows.append(observer_even_response)
                plus_value_rows.append(float(plus))
                minus_value_rows.append(float(minus))
                delta_values.append(float(delta))
            estimates = np.asarray(estimate_values, dtype=float)
            plus_values = np.asarray(plus_value_rows, dtype=float)
            minus_values = np.asarray(minus_value_rows, dtype=float)
            deltas = np.asarray(delta_values, dtype=float)
            observer_estimates = np.asarray(observer_estimate_rows, dtype=float)
            observer_even_estimates = np.asarray(observer_even_rows, dtype=float)
            observer_response_simulated = bool(
                paired_probe_receipts and all(paired_probe_receipts)
            )
            if probe_cache is not None:
                probe_cache[cache_key] = {
                    "estimates": estimates,
                    "plus_values": plus_values,
                    "minus_values": minus_values,
                    "deltas": deltas,
                    "observer_estimates": observer_estimates,
                    "observer_even_estimates": observer_even_estimates,
                    "observer_response_simulated": observer_response_simulated,
                    "intervention_support_hashes": intervention_support_hashes,
                }
    requested_delta_baryon = _delta_baryon(time_value, perturb_strength, transition_scale)
    delivered_half_step = float(np.mean(deltas)) if deltas.size else 0.0
    source_intervention = _source_intervention(
        cap=cap,
        a_value=a_value,
        cap_index=cap_index,
        time_index=time_index,
        requested_half_step=requested_delta_baryon,
        delivered_half_step=delivered_half_step,
        control=control,
        graph=graph,
    )

    return {
        "a": float(a_value),
        "k_h_mpc": float(effective_k_proxy),
        "k_proxy_inverse_theta": float(effective_k_proxy),
        "k_units": "inverse_cap_opening_angle_proxy",
        "control": control,
        "cap_index": int(cap_index),
        "time_index": int(time_index),
        "time": float(time_value),
        "theta0": float(cap.theta0),
        "collar_width": float(cap.collar_width),
        "readout_theta0": float(readout_cap.theta0),
        "readout_collar_width": float(readout_cap.collar_width),
        "sim_transition_scale": float(sim_scale),
        "response_field": str(response_field),
        "normalization": "EQUILIBRIUM_CONTRAST_DIAGNOSTIC",
        "response_numerator": "paired_delta_response_field",
        "source_variable": "ANOMALY_FRAME_BARYON_CONTRAST_PROXY",
        "denominator": "RHO_A_EQ_BACKGROUND_DIAGNOSTIC",
        "rho_A": float(rho_a),
        "rho_A_base": float(base_value),
        "rho_A_eq": float(base_value),
        "rho_A_eq_background": float(base_value),
        "rho_A_eq_plus_mean": float(np.mean(plus_values)) if plus_values.size else None,
        "rho_A_eq_minus_mean": float(np.mean(minus_values)) if minus_values.size else None,
        "repair_anomaly_plus_mean": float(np.mean(plus_values - base_value)) if plus_values.size else None,
        "repair_anomaly_minus_mean": float(np.mean(minus_values - base_value)) if minus_values.size else None,
        "requested_delta_baryon": float(requested_delta_baryon),
        "delivered_source_half_step": delivered_half_step,
        "delivered_source_difference": 2.0 * delivered_half_step,
        "delta_baryon": delivered_half_step,
        "source_intervention": asdict(source_intervention),
        "physical_source_intervention": False,
        "source_intervention_type": "CAP_COLLAR_PROXY_NOT_PHYSICAL_SOURCE_INTERVENTION",
        "B_A_mean": float(np.mean(estimates)) if estimates.size else None,
        "B_A_std": float(np.std(estimates, ddof=1)) if estimates.size > 1 else 0.0,
        "B_A_sem": float(np.std(estimates, ddof=1) / math.sqrt(estimates.size)) if estimates.size > 1 else 0.0,
        "mode_count": int(estimates.size),
        "sign_stable": _sign_stable(estimates),
        "source": "paired_cap_collar_perturb_resettle_rerun",
        "parent_source": "finite_screen_plus_minus_cap_collar_perturb_resettle",
        "gauge_covariant_centered_probe_receipt": bool(observer_response_simulated),
        "intervention_support_hashes": intervention_support_hashes if control not in {
            "no_perturbation",
            "no_repair_load_channel",
            "baryon_delta_applied_after_record_freezeout",
        } else [],
        "_observer_response": (
            np.mean(observer_estimates, axis=0).tolist()
            if observer_estimates.ndim == 2 and observer_estimates.shape[1] == observer_count
            else []
        ),
        "_observer_even_response": (
            np.mean(observer_even_estimates, axis=0).tolist()
            if observer_even_estimates.ndim == 2 and observer_even_estimates.shape[1] == observer_count
            else []
        ),
        "_observer_response_simulated": bool(observer_response_simulated),
    }


def _observer_probe_context(
    observer_views: list[dict[str, Any]] | None,
    *,
    entropy: np.ndarray,
    patch_count: int,
) -> dict[str, Any]:
    patch_rows = [
        row
        for row in (observer_views or [])
        if isinstance(row, dict) and row.get("view_type") == "patch_observer"
    ]
    blockers: list[str] = []
    if not patch_rows:
        blockers.append("patch_observer_views_unavailable")
    observer_ids: list[int] = []
    flat_observer_indices: list[int] = []
    flat_support_nodes: list[int] = []
    flat_weights: list[float] = []
    totals = np.zeros(len(patch_rows), dtype=float)
    entropy = np.asarray(entropy, dtype=float).reshape(-1)
    for observer_index, row in enumerate(patch_rows):
        try:
            observer_id = int(row.get("observer_id", observer_index))
        except (TypeError, ValueError):
            observer_id = observer_index
            blockers.append(f"invalid_observer_id:{observer_index}")
        observer_ids.append(observer_id)
        support_value = row.get("support_nodes")
        if not isinstance(support_value, (list, tuple, np.ndarray)):
            blockers.append(f"observer_support_unavailable:{observer_id}")
            continue
        try:
            support = np.asarray(support_value, dtype=np.int64).reshape(-1)
        except (TypeError, ValueError):
            blockers.append(f"observer_support_invalid:{observer_id}")
            continue
        valid = (support >= 0) & (support < int(patch_count))
        if not np.all(valid):
            blockers.append(f"observer_support_out_of_range:{observer_id}")
        support = support[valid]
        if support.size == 0:
            blockers.append(f"observer_support_empty:{observer_id}")
            continue
        local_weights = np.where(
            np.isfinite(entropy[support]) & (entropy[support] > 0.0),
            entropy[support],
            0.0,
        )
        total = float(np.sum(local_weights))
        if total <= 0.0:
            blockers.append(f"observer_support_weight_nonpositive:{observer_id}")
            continue
        totals[observer_index] = total
        flat_observer_indices.extend([observer_index] * int(support.size))
        flat_support_nodes.extend(int(value) for value in support)
        flat_weights.extend(float(value) for value in local_weights)
    if len(set(observer_ids)) != len(observer_ids):
        blockers.append("duplicate_patch_observer_ids")
    return {
        "observer_count": len(patch_rows),
        "observer_ids": np.asarray(observer_ids, dtype=np.int64),
        "flat_observer_indices": np.asarray(flat_observer_indices, dtype=np.int64),
        "flat_support_nodes": np.asarray(flat_support_nodes, dtype=np.int64),
        "flat_weights": np.asarray(flat_weights, dtype=float),
        "weight_totals": totals,
        "available": not blockers,
        "blockers": blockers,
    }


def _observer_weighted_means(values: Any, context: dict[str, Any]) -> np.ndarray:
    observer_count = int(context.get("observer_count", 0) or 0)
    if observer_count <= 0 or values is None:
        return np.zeros(observer_count, dtype=float)
    array = np.asarray(values, dtype=float).reshape(-1)
    support_nodes = np.asarray(context.get("flat_support_nodes", []), dtype=np.int64)
    observer_indices = np.asarray(context.get("flat_observer_indices", []), dtype=np.int64)
    weights = np.asarray(context.get("flat_weights", []), dtype=float)
    totals = np.asarray(context.get("weight_totals", []), dtype=float)
    if not (support_nodes.size == observer_indices.size == weights.size) or totals.size != observer_count:
        return np.zeros(observer_count, dtype=float)
    if support_nodes.size == 0 or int(np.max(support_nodes, initial=-1)) >= array.size:
        return np.zeros(observer_count, dtype=float)
    local_values = np.where(np.isfinite(array[support_nodes]), array[support_nodes], 0.0)
    sums = np.bincount(
        observer_indices,
        weights=local_values * weights,
        minlength=observer_count,
    ).astype(float)
    return np.divide(sums, totals, out=np.zeros_like(sums), where=totals > 0.0)


def _pop_observer_response(row: dict[str, Any]) -> dict[str, Any]:
    response = row.pop("_observer_response", [])
    even_response = row.pop("_observer_even_response", [])
    simulated = bool(row.pop("_observer_response_simulated", False))
    return {
        "a": float(row["a"]),
        "cap_index": int(row["cap_index"]),
        "time_index": int(row["time_index"]),
        "time": float(row["time"]),
        "control": row.get("control"),
        "observer_response": response,
        "observer_even_response": even_response,
        "actual_perturb_resettle_rerun": simulated,
    }


def _observer_geometry_response_rows(
    context: dict[str, Any],
    response_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    observer_count = int(context.get("observer_count", 0) or 0)
    observer_ids = np.asarray(context.get("observer_ids", []), dtype=np.int64)
    blockers = list(context.get("blockers", []))
    feature_manifest: list[dict[str, Any]] = []
    for row in response_rows:
        for component in ("odd_centered_derivative", "even_settling_response"):
            feature_manifest.append(
                {
                    "feature_index": len(feature_manifest),
                    "a": float(row["a"]),
                    "cap_index": int(row["cap_index"]),
                    "time_index": int(row["time_index"]),
                    "time": float(row["time"]),
                    "observable": component,
                    "cap_index_role": "shared_intervention_axis_only",
                }
            )
    if not response_rows:
        blockers.append("paired_observer_response_rows_unavailable")
        matrix = np.zeros((observer_count, 0), dtype=float)
    else:
        response_vectors = [
            np.asarray(row.get(key, []), dtype=float)
            for row in response_rows
            for key in ("observer_response", "observer_even_response")
        ]
        if any(vector.size != observer_count for vector in response_vectors):
            blockers.append("paired_observer_response_width_mismatch")
            matrix = np.zeros((observer_count, 0), dtype=float)
        else:
            matrix = np.column_stack(response_vectors)
    if response_rows and not all(row.get("actual_perturb_resettle_rerun", False) for row in response_rows):
        blockers.append("main_observer_response_not_from_actual_paired_reruns")
    if matrix.size and not np.all(np.isfinite(matrix)):
        blockers.append("paired_observer_response_nonfinite")
        matrix = np.where(np.isfinite(matrix), matrix, 0.0)

    control_names = sorted(
        {
            str(row["control"])
            for row in control_rows
            if row.get("control") is not None
        }
    )
    control_matrices: dict[str, np.ndarray] = {}
    control_manifest: list[dict[str, Any]] = []
    main_keys = [
        (float(row["a"]), int(row["cap_index"]), int(row["time_index"]))
        for row in response_rows
    ]
    for control in control_names:
        by_key = {
            (float(row["a"]), int(row["cap_index"]), int(row["time_index"])): row
            for row in control_rows
            if str(row.get("control")) == control
        }
        vectors = [
            np.asarray(by_key.get(key, {}).get(component, []), dtype=float)
            for key in main_keys
            for component in ("observer_response", "observer_even_response")
        ]
        if not vectors or any(vector.size != observer_count for vector in vectors):
            blockers.append(f"paired_observer_control_width_mismatch:{control}")
            continue
        control_matrix = np.column_stack(vectors)
        if not np.all(np.isfinite(control_matrix)):
            blockers.append(f"paired_observer_control_nonfinite:{control}")
            control_matrix = np.where(np.isfinite(control_matrix), control_matrix, 0.0)
        control_matrices[control] = control_matrix
        control_manifest.append(
            {
                "control": control,
                "feature_count": int(control_matrix.shape[1]),
                "intervention_control": True,
                "presentation_relabel_control": False,
            }
        )
    if not control_matrices:
        blockers.append("paired_observer_response_controls_unavailable")

    response_nondegenerate = bool(
        matrix.size
        and np.any(np.abs(matrix) > 1.0e-12)
        and np.any(np.std(matrix, axis=0) > 1.0e-12)
    )
    if not response_nondegenerate:
        blockers.append("paired_observer_response_geometry_degenerate")
    no_perturbation = control_matrices.get("no_perturbation")
    no_perturbation_separation = bool(
        matrix.size
        and no_perturbation is not None
        and no_perturbation.shape == matrix.shape
        and float(np.mean(np.abs(matrix - no_perturbation))) > 1.0e-12
    )
    if not no_perturbation_separation:
        blockers.append("no_perturbation_observer_response_control_not_separated")

    cap_invariance = _cap_relabel_invariance_report(matrix, feature_manifest)
    if not cap_invariance["receipt"]:
        blockers.extend(cap_invariance["blockers"])
    producer_receipt = bool(
        context.get("available", False)
        and matrix.shape == (observer_count, len(feature_manifest))
        and matrix.shape[1] > 0
        and control_matrices
        and not blockers
    )
    observer_rows = []
    observer_control_rows = []
    for observer_index, observer_id in enumerate(observer_ids):
        observer_rows.append(
            {
                "observer_id": int(observer_id),
                "paired_perturbation_response_tensor": matrix[observer_index].tolist(),
            }
        )
        observer_control_rows.append(
            {
                "observer_id": int(observer_id),
                "paired_perturbation_control_tensors": {
                    name: values[observer_index].tolist()
                    for name, values in sorted(control_matrices.items())
                },
            }
        )
    return {
        "producer": "paired_cap_collar_perturb_resettle.observer_local_support_readout_v1",
        "producer_receipt": producer_receipt,
        "observer_view_rows": observer_rows,
        "observer_view_control_rows": observer_control_rows,
        "feature_manifest": feature_manifest,
        "control_manifest": control_manifest,
        "cap_relabel_invariance": cap_invariance,
        "response_nondegenerate": response_nondegenerate,
        "no_perturbation_control_separation_receipt": no_perturbation_separation,
        "blockers": blockers,
    }


def _cap_relabel_invariance_report(
    matrix: np.ndarray,
    feature_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or len(feature_manifest) != matrix.shape[1]:
        return {
            "receipt": False,
            "blockers": ["cap_relabel_invariance_input_unavailable"],
            "max_pairwise_distance_distortion": None,
        }
    cap_labels = sorted({int(row["cap_index"]) for row in feature_manifest})
    relabel = dict(zip(cap_labels, reversed(cap_labels), strict=True))
    permutation = sorted(
        range(len(feature_manifest)),
        key=lambda index: (
            float(feature_manifest[index]["a"]),
            relabel[int(feature_manifest[index]["cap_index"])],
            int(feature_manifest[index]["time_index"]),
        ),
    )
    sample = matrix[: min(matrix.shape[0], 128)]
    original_distance = _squared_row_distance(sample)
    permuted_distance = _squared_row_distance(sample[:, permutation])
    distortion = float(np.max(np.abs(original_distance - permuted_distance))) if sample.size else 0.0
    receipt = bool(sorted(permutation) == list(range(matrix.shape[1])) and distortion <= 1.0e-10)
    return {
        "control": "shared_cap_column_serialization_permutation",
        "presentation_only": True,
        "expected_geometry_change": False,
        "algebraic_serialization_sanity_only": True,
        "dynamics_rerun": False,
        "tested_observer_count": int(sample.shape[0]),
        "max_pairwise_distance_distortion": distortion,
        "receipt": receipt,
        "blockers": [] if receipt else ["global_cap_relabel_changed_observer_response_geometry"],
    }


def _squared_row_distance(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    squared_norm = np.sum(matrix * matrix, axis=1)
    distance = squared_norm[:, None] + squared_norm[None, :] - 2.0 * matrix @ matrix.T
    return np.maximum(distance, 0.0)


def attach_paired_perturbation_features(
    observer_views: list[dict[str, Any]] | None,
    report: dict[str, Any],
) -> dict[str, Any]:
    patch_rows = [
        row
        for row in (observer_views or [])
        if isinstance(row, dict) and row.get("view_type") == "patch_observer"
    ]
    feature_rows = report.get("observer_view_rows")
    control_rows = report.get("observer_view_control_rows")
    blockers = list(report.get("observer_response_producer_blockers", []))
    if not isinstance(feature_rows, list) or not isinstance(control_rows, list):
        blockers.append("paired_observer_feature_rows_unavailable")
        feature_rows = []
        control_rows = []
    feature_by_id = {
        int(row["observer_id"]): row
        for row in feature_rows
        if isinstance(row, dict) and row.get("observer_id") is not None
    }
    controls_by_id = {
        int(row["observer_id"]): row
        for row in control_rows
        if isinstance(row, dict) and row.get("observer_id") is not None
    }
    manifest = report.get("observer_response_feature_manifest")
    manifest_hash = stable_json_hash(
        manifest if isinstance(manifest, list) else []
    ).removeprefix("sha256:")
    attached = 0
    producer_receipt = bool(report.get("paired_perturbation_response_producer_receipt", False))
    cap_invariance = report.get("observer_response_cap_relabel_invariance")
    for index, row in enumerate(patch_rows):
        observer_id = int(row.get("observer_id", index))
        source = feature_by_id.get(observer_id)
        controls = controls_by_id.get(observer_id)
        response = source.get("paired_perturbation_response_tensor") if source else None
        control_tensors = controls.get("paired_perturbation_control_tensors") if controls else None
        row_receipt = bool(
            producer_receipt
            and isinstance(response, list)
            and bool(response)
            and isinstance(control_tensors, dict)
            and bool(control_tensors)
        )
        if response is not None:
            row["paired_perturbation_response_tensor"] = response
        if control_tensors is not None:
            row["paired_perturbation_control_tensors"] = control_tensors
        row["paired_perturbation_response_feature_schema"] = {
            "version": "observer_local_paired_cap_response_v1",
            "feature_manifest_sha256": manifest_hash,
            "feature_manifest_hash_schema": CANONICAL_HASH_SCHEMA,
            "feature_count": len(manifest) if isinstance(manifest, list) else 0,
            "hash_bucket_geometry": False,
            "screen_coordinates_embedded": False,
            "cap_index_role": "shared_intervention_axis_only",
        }
        row["paired_perturbation_response_provenance"] = {
            "producer": report.get("observer_response_producer"),
            "actual_paired_perturb_resettle": True,
            "observer_local_support_readout": True,
            "cap_column_serialization_sanity": cap_invariance,
            "causal_feature_ancestors": ["s2_cap_axis", "screen_pixel_coordinate"],
            "strict_neutral_eligible": False,
            "diagnostic_reason": "RoundCap interventions are selected from the S2 screen chart",
        }
        row["paired_perturbation_response_producer_receipt"] = row_receipt
        attached += int(row_receipt)
    if len(feature_by_id) != len(patch_rows):
        blockers.append(f"paired_feature_observer_coverage:{len(feature_by_id)}/{len(patch_rows)}")
    receipt = bool(patch_rows and attached == len(patch_rows) and not blockers)
    return {
        "mode": "paired_observer_geometry_attachment_v1",
        "patch_observer_count": len(patch_rows),
        "attached_observer_count": attached,
        "feature_manifest_sha256": manifest_hash,
        "feature_manifest_hash_schema": CANONICAL_HASH_SCHEMA,
        "receipt": receipt,
        "blockers": blockers,
    }


def _validated_graph_state(graph_state: dict[str, Any], patch_count: int) -> dict[str, Any]:
    required = ("left", "right", "port_left", "port_right", "gauge", "group_name", "group_order")
    missing = [name for name in required if name not in graph_state]
    if missing:
        raise ValueError(f"paired B_A perturbation graph_state missing keys: {missing}")
    left = np.asarray(graph_state["left"], dtype=np.int64)
    right = np.asarray(graph_state["right"], dtype=np.int64)
    port_left = np.asarray(graph_state["port_left"], dtype=np.int16)
    port_right = np.asarray(graph_state["port_right"], dtype=np.int16)
    gauge = np.asarray(graph_state["gauge"], dtype=np.int16)
    if not (left.shape == right.shape == port_left.shape == port_right.shape == gauge.shape):
        raise ValueError("paired B_A perturbation graph_state arrays must have matching edge shape")
    count = int(graph_state.get("patch_count", patch_count))
    degree = np.asarray(graph_state.get("degree", np.zeros(0)), dtype=float)
    if degree.size != count:
        degree = np.bincount(np.concatenate([left, right]), minlength=count).astype(float) if left.size else np.ones(count)
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
        "degree": np.maximum(degree, 1.0),
        "production_sector_repair_enabled": sector_enabled,
        "production_sector_repair_config": sector_config,
        "production_sector_repair_config_available": sector_config_available,
    }


def _cell_entropy(cell_entropy: np.ndarray | float | None, patch_count: int) -> np.ndarray:
    if cell_entropy is None:
        return np.ones(int(patch_count), dtype=float)
    values = np.asarray(cell_entropy, dtype=float)
    if values.ndim == 0:
        return np.full(int(patch_count), float(values), dtype=float)
    if values.size != int(patch_count):
        raise ValueError("cell_entropy must be scalar or match patch_count")
    return values.astype(float)


def _weighted_mean(values: Any, weights: np.ndarray) -> float:
    if values is None:
        return 0.0
    array = np.asarray(values, dtype=float)
    if array.size != weights.size:
        return 0.0
    local_weights = np.asarray(weights, dtype=float)
    total = float(np.sum(local_weights))
    if total <= 0.0:
        return 0.0
    return float(np.sum(array * local_weights) / total)


def _response_scale(raw_fields: dict[str, np.ndarray], response_field: str, weights: np.ndarray) -> float:
    base = abs(_weighted_mean(raw_fields.get(response_field), weights))
    repair = abs(_weighted_mean(raw_fields.get("repair_load"), weights))
    cumulative = abs(_weighted_mean(raw_fields.get("cumulative_repair_load"), weights))
    mismatch = abs(_weighted_mean(raw_fields.get("local_mismatch_density"), weights))
    return max(base, repair, cumulative, mismatch, 1.0e-12)


def _delta_baryon(time_value: float, perturb_strength: float, transition_scale: float) -> float:
    amount = abs(float(transition_scale) * float(time_value)) / (2.0 * math.pi)
    return max(abs(float(perturb_strength)) * amount, 1.0e-6)


def _source_intervention(
    *,
    cap: RoundCap,
    a_value: float,
    cap_index: int,
    time_index: int,
    requested_half_step: float,
    delivered_half_step: float,
    control: str | None,
    graph: dict[str, Any],
) -> PhysicalSourceIntervention:
    background_payload = {
        "patch_count": int(graph.get("patch_count", 0)),
        "group_order": int(graph.get("group_order", 0)),
        "a": round(float(a_value), 12),
    }
    constraint_payload = {
        "constraint_family": "cap_collar_proxy",
        "cap_theta0": round(float(cap.theta0), 12),
        "collar_width": round(float(cap.collar_width), 12),
    }
    delivered = float(delivered_half_step)
    requested = float(requested_half_step)
    return PhysicalSourceIntervention(
        background_hash=_hash_payload(background_payload),
        source_vector_id="ANOMALY_FRAME_BARYON_CONTRAST_PROXY",
        tangent_vector=[float(cap.axis[0]), float(cap.axis[1]), float(cap.axis[2]), requested],
        constraint_matrix_hash=_hash_payload(constraint_payload),
        retraction_id="cap_collar_proxy_centered_screen_perturb_resettle",
        delivered_source_vector=[delivered],
        constraint_residuals={
            "delivered_minus_requested_abs": abs(delivered - requested) if control != "no_perturbation" else 0.0,
            "admissible_source_tangent_residual": 1.0,
            "physical_source_vector_residual": 1.0,
        },
        physical_source_intervention=False,
    )


def _hash_payload(payload: dict[str, Any]) -> str:
    return stable_json_hash(payload)


def _randomized_cap(cap: RoundCap, seed: int) -> RoundCap:
    rng = np.random.default_rng(int(seed))
    axis = rng.normal(size=3)
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
    tangent = rng.normal(size=3)
    tangent = tangent - axis * float(np.dot(axis, tangent))
    if float(np.linalg.norm(tangent)) < 1.0e-12:
        tangent = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    tangent = tangent / max(float(np.linalg.norm(tangent)), 1.0e-12)
    return RoundCap(axis=axis, theta0=float(cap.theta0), tangent=tangent, collar_width=float(cap.collar_width)).normalized()


def _sign_stable(values: np.ndarray) -> bool:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False
    mean = float(np.mean(values))
    if abs(mean) < 1.0e-12:
        return False
    nonzero = values[np.abs(values) > 1.0e-12]
    if nonzero.size == 0:
        return False
    return bool(float(np.mean(np.sign(nonzero) == np.sign(mean))) >= 0.8)


def _readiness(rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]) -> dict[str, Any]:
    main_values = [abs(float(row["B_A_mean"])) for row in rows if isinstance(row.get("B_A_mean"), (int, float))]
    main_scale = fmean(main_values) if main_values else 0.0
    grouped: dict[str, list[float]] = {}
    for row in control_rows:
        if isinstance(row.get("B_A_mean"), (int, float)):
            grouped.setdefault(str(row.get("control")), []).append(abs(float(row["B_A_mean"])))
    main_by_key = {
        _row_key(row): float(row["B_A_mean"])
        for row in rows
        if isinstance(row.get("B_A_mean"), (int, float)) and np.isfinite(float(row["B_A_mean"]))
    }
    control_metrics: dict[str, dict[str, float | int | bool | str | None]] = {}
    control_failures: dict[str, bool] = {}
    null_controls = {
        "no_perturbation",
        "no_repair_load_channel",
        "baryon_delta_applied_after_record_freezeout",
    }
    for name, values in grouped.items():
        matched_main: list[float] = []
        matched_control: list[float] = []
        for row in control_rows:
            if str(row.get("control")) != name or not isinstance(row.get("B_A_mean"), (int, float)):
                continue
            key = _row_key(row)
            if key not in main_by_key:
                continue
            matched_main.append(float(main_by_key[key]))
            matched_control.append(float(row["B_A_mean"]))
        mean_abs_control = fmean(values) if values else 0.0
        if matched_main and matched_control:
            main_array = np.asarray(matched_main, dtype=float)
            control_array = np.asarray(matched_control, dtype=float)
            separation = float(np.mean(np.abs(control_array - main_array)) / max(float(np.mean(np.abs(main_array))), 1.0e-300))
            corr = _safe_corr(main_array, control_array)
            sign_agreement = float(np.mean(np.sign(main_array) == np.sign(control_array)))
        else:
            separation = 0.0
            corr = None
            sign_agreement = 0.0
        null_suppressed = bool(mean_abs_control < 0.25 * max(main_scale, 1.0e-300))
        separated = bool(separation >= 0.5 or (corr is not None and corr < 0.5) or sign_agreement < 0.6)
        passed = bool(null_suppressed if name in null_controls else separated)
        control_failures[name] = passed
        control_metrics[name] = {
            "expected_failure_mode": "null_suppression" if name in null_controls else "paired_response_separation",
            "mean_abs_B_A": float(mean_abs_control),
            "main_mean_abs_B_A": float(main_scale),
            "mean_abs_ratio_to_main": float(mean_abs_control / max(main_scale, 1.0e-300)),
            "matched_row_count": int(len(matched_main)),
            "relative_separation_from_main": float(separation),
            "correlation_with_main": corr,
            "sign_agreement_with_main": float(sign_agreement),
            "null_suppressed": bool(null_suppressed),
            "separated_from_main": bool(separated),
            "expected_failure_observed": bool(passed),
        }
    checks = {
        "paired_perturb_resettle_rows_emitted": bool(rows),
        "finite_difference_rows_emitted": bool(rows),
        "control_rows_emitted": bool(control_rows),
        "no_cmb_data_used": True,
        "real_baryon_perturbation_runs_present": bool(rows),
        "full_perturbation_rerun": bool(rows),
        "report_backed_surrogate_parent": False,
        "finite_observer_view_parent_variation": False,
        "controls_fail": bool(control_failures and all(control_failures.values())),
        "sign_stable": bool(rows and all(bool(row.get("sign_stable")) for row in rows)),
        "scale_calibrated_k_h_mpc": False,
        "calibrated_a_evolution": False,
        "rho_A_of_a_physical_emitted": False,
        "rho_A_eq_of_a_physical_emitted": False,
        "Gamma_rec_of_k_a_emitted": False,
        "energy_momentum_exchange_closed": False,
        "gauge_consistency_audited": False,
        "refinement_convergence_passed": False,
        "common_source_functional_receipt": False,
        "admissible_source_tangent_receipt": False,
        "constraint_preserving_retraction_receipt": False,
        "B_A_source_lift_independence_receipt": False,
        "source_vector_sufficiency_receipt": False,
        "finite_difference_order_receipt": False,
        "C1_refinement_receipt": False,
        "order_of_limits_receipt": False,
    }
    checks["paired_B_A_diagnostic_receipt"] = bool(
        checks["paired_perturb_resettle_rows_emitted"]
        and checks["control_rows_emitted"]
        and checks["no_cmb_data_used"]
        and checks["real_baryon_perturbation_runs_present"]
        and checks["full_perturbation_rerun"]
        and checks["controls_fail"]
        and checks["sign_stable"]
    )
    required_gates = (
        "paired_perturb_resettle_rows_emitted",
        "finite_difference_rows_emitted",
        "control_rows_emitted",
        "no_cmb_data_used",
        "real_baryon_perturbation_runs_present",
        "full_perturbation_rerun",
        "controls_fail",
        "sign_stable",
        "scale_calibrated_k_h_mpc",
        "calibrated_a_evolution",
        "rho_A_of_a_physical_emitted",
        "rho_A_eq_of_a_physical_emitted",
        "Gamma_rec_of_k_a_emitted",
        "energy_momentum_exchange_closed",
        "gauge_consistency_audited",
        "refinement_convergence_passed",
        "common_source_functional_receipt",
        "admissible_source_tangent_receipt",
        "constraint_preserving_retraction_receipt",
        "B_A_source_lift_independence_receipt",
        "source_vector_sufficiency_receipt",
        "finite_difference_order_receipt",
        "C1_refinement_receipt",
        "order_of_limits_receipt",
    )
    return {
        "checks": checks,
        "control_failures": control_failures,
        "control_metrics": control_metrics,
        "B_A_PAIRED_DIAGNOSTIC_RECEIPT": bool(checks["paired_B_A_diagnostic_receipt"]),
        "B_A_PARENT_RECEIPT": False,
        "physical_prediction_ready": False,
        "missing_gates": [name for name in required_gates if not checks.get(name, False)],
        "claim_boundary": (
            "Paired finite reruns are present, but physical readiness additionally "
            "requires controls to fail, scale/time calibration, closure equations, "
            "and refinement convergence."
        ),
    }


def _row_key(row: dict[str, Any]) -> tuple[float, int, int]:
    return (
        round(float(row.get("a", 0.0)), 12),
        int(row.get("cap_index", -1)),
        int(row.get("time_index", -1)),
    )


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 2:
        return None
    if float(np.std(left)) <= 1.0e-15 or float(np.std(right)) <= 1.0e-15:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(value) for key, value in row.items()})


def _csv_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, default=str)


def _markdown_report(report: dict[str, Any]) -> str:
    rows = [row for row in report.get("rows", []) if isinstance(row.get("B_A_mean"), (int, float))]
    controls = report.get("readiness", {}).get("control_failures", {})
    control_metrics = report.get("readiness", {}).get("control_metrics", {})
    mean_b = fmean(abs(float(row["B_A_mean"])) for row in rows) if rows else None
    lines = [
        "# Paired B_A Perturb/Resettle Diagnostic",
        "",
        f"- mode: `{report['mode']}`",
        f"- rows: {len(report.get('rows', []))}",
        f"- control rows: {len(report.get('control_rows', []))}",
        f"- mean |B_A|: {mean_b}",
        f"- controls_fail: {report.get('readiness', {}).get('checks', {}).get('controls_fail')}",
        f"- control failures: {controls}",
        f"- B_A_PAIRED_DIAGNOSTIC_RECEIPT: {report.get('B_A_PAIRED_DIAGNOSTIC_RECEIPT')}",
        f"- B_A_PARENT_RECEIPT: {report.get('B_A_PARENT_RECEIPT')}",
        f"- physical_cmb_prediction: {report.get('physical_cmb_prediction')}",
        "",
    ]
    if isinstance(control_metrics, dict) and control_metrics:
        lines.extend(["## Control Metrics", ""])
        for name, metrics in sorted(control_metrics.items()):
            if not isinstance(metrics, dict):
                continue
            lines.append(
                f"- `{name}`: mode={metrics.get('expected_failure_mode')}, "
                f"ratio={metrics.get('mean_abs_ratio_to_main')}, "
                f"separation={metrics.get('relative_separation_from_main')}, "
                f"corr={metrics.get('correlation_with_main')}, "
                f"pass={metrics.get('expected_failure_observed')}"
            )
        lines.append("")
    lines.extend([report.get("claim_boundary", ""), ""])
    return "\n".join(lines)
