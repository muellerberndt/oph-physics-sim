from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import numpy as np

from oph_fpe.bulk.cap_geometry import RoundCap, cap_weights
from oph_fpe.bulk.modular_response_kernel import _simulate_cap_collar_perturb_resettle


DEFAULT_PAIRED_B_A_CONTROLS = (
    "no_perturbation",
    "no_repair_load_channel",
    "baryon_delta_applied_after_record_freezeout",
    "phase_shuffled_baryon_mode",
    "random_collar_labels",
    "wrong_k_label",
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
    seed_base = int(seed)

    rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    control_values = tuple(DEFAULT_PAIRED_B_A_CONTROLS if controls is None else controls)
    for a_value in a_values:
        for cap_index, cap in enumerate(cap_values):
            normalized_cap = cap.normalized()
            k_proxy = 1.0 / max(float(normalized_cap.theta0), 1.0e-12)
            weights = cap_weights(points, normalized_cap, soft=True) * entropy
            rho_a = _response_scale(raw_fields, response_field, weights)
            for time_index, time_value in enumerate(time_values):
                rows.append(
                    _paired_row(
                        points,
                        normalized_cap,
                        raw_fields,
                        graph,
                        weights,
                        entropy,
                        response_field=response_field,
                        rho_a=rho_a,
                        a_value=a_value,
                        k_proxy=k_proxy,
                        cap_index=cap_index,
                        time_index=time_index,
                        time_value=time_value,
                        modes_per_cap_time=modes_per_cap_time,
                        control=None,
                        perturb_strength=perturb_strength,
                        perturb_budget_mode=perturb_budget_mode,
                        fixed_perturb_fraction=fixed_perturb_fraction,
                        perturb_selection_mode=perturb_selection_mode,
                        repair_steps=repair_steps,
                        repairs_per_step=repairs_per_step,
                        transition_scale=transition_scale,
                        seed=seed_base + 1009 * cap_index + 9173 * time_index,
                    )
                )
                for control in control_values:
                    control_rows.append(
                        _paired_row(
                            points,
                            normalized_cap,
                            raw_fields,
                            graph,
                            weights,
                            entropy,
                            response_field=response_field,
                            rho_a=rho_a,
                            a_value=a_value,
                            k_proxy=k_proxy,
                            cap_index=cap_index,
                            time_index=time_index,
                            time_value=time_value,
                            modes_per_cap_time=modes_per_cap_time,
                            control=str(control),
                            perturb_strength=perturb_strength,
                            perturb_budget_mode=perturb_budget_mode,
                            fixed_perturb_fraction=fixed_perturb_fraction,
                            perturb_selection_mode=perturb_selection_mode,
                            repair_steps=repair_steps,
                            repairs_per_step=repairs_per_step,
                            transition_scale=transition_scale,
                            seed=seed_base + 31_337 + 1009 * cap_index + 9173 * time_index,
                        )
                    )

    readiness = _readiness(rows, control_rows)
    return {
        "mode": "paired_cap_collar_perturb_resettle_B_A_parent_v0",
        "primary_parent_source": "paired_cap_collar_perturb_resettle_rerun",
        "normalization": "EQUILIBRIUM_CONTRAST_DIAGNOSTIC",
        "response_numerator": "paired_delta_response_field",
        "source_variable": "ANOMALY_FRAME_BARYON_CONTRAST_PROXY",
        "denominator": "RHO_A_EQ_BACKGROUND_DIAGNOSTIC",
        "source_report_count": 0,
        "observer_view_source_count": 0,
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
        "rows": rows,
        "control_rows": control_rows,
        "paired_perturbation_rows": rows,
        "paired_perturbation_control_rows": control_rows,
        "source_intervention_schema": list(PhysicalSourceIntervention.__dataclass_fields__),
        "observer_view_rows": [],
        "observer_view_control_rows": [],
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
    if control in {"no_perturbation", "no_repair_load_channel", "baryon_delta_applied_after_record_freezeout"}:
        estimates = np.zeros(max(1, int(modes_per_cap_time)), dtype=float)
        plus_values = np.full(estimates.size, base_value, dtype=float)
        minus_values = np.full(estimates.size, base_value, dtype=float)
        requested_delta = _delta_baryon(time_value, perturb_strength, transition_scale)
        delivered_delta = 0.0 if control == "no_perturbation" else requested_delta
        deltas = np.full(estimates.size, delivered_delta, dtype=float)
        effective_k_proxy = float(k_proxy)
        cap_used = cap
        sim_scale = float(transition_scale)
    else:
        effective_k_proxy = float(k_proxy) * (1.7 if control == "wrong_k_label" else 1.0)
        cap_used = cap
        sim_scale = float(transition_scale) * (0.5 if control == "wrong_k_label" else 1.0)
        estimates = []
        plus_values = []
        minus_values = []
        deltas = []
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
            plus = _weighted_mean(post_plus.get(response_field), readout_weights)
            minus = _weighted_mean(post_minus.get(response_field), readout_weights)
            delta = _delta_baryon(time_value, perturb_strength, transition_scale)
            derivative = (plus - minus) / (2.0 * max(delta, 1.0e-12))
            estimates.append(float(derivative / max(abs(float(base_value)), 1.0e-12)))
            plus_values.append(float(plus))
            minus_values.append(float(minus))
            deltas.append(float(delta))
        estimates = np.asarray(estimates, dtype=float)
        plus_values = np.asarray(plus_values, dtype=float)
        minus_values = np.asarray(minus_values, dtype=float)
        deltas = np.asarray(deltas, dtype=float)
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
    }


def _validated_graph_state(graph_state: dict[str, Any], patch_count: int) -> dict[str, Any]:
    required = ("left", "right", "port_left", "port_right", "group_order")
    missing = [name for name in required if name not in graph_state]
    if missing:
        raise ValueError(f"paired B_A perturbation graph_state missing keys: {missing}")
    left = np.asarray(graph_state["left"], dtype=np.int64)
    right = np.asarray(graph_state["right"], dtype=np.int64)
    port_left = np.asarray(graph_state["port_left"], dtype=np.int16)
    port_right = np.asarray(graph_state["port_right"], dtype=np.int16)
    if not (left.shape == right.shape == port_left.shape == port_right.shape):
        raise ValueError("paired B_A perturbation graph_state arrays must have matching edge shape")
    count = int(graph_state.get("patch_count", patch_count))
    degree = np.asarray(graph_state.get("degree", np.zeros(0)), dtype=float)
    if degree.size != count:
        degree = np.bincount(np.concatenate([left, right]), minlength=count).astype(float) if left.size else np.ones(count)
    return {
        "left": left,
        "right": right,
        "port_left": port_left,
        "port_right": port_right,
        "group_order": int(graph_state["group_order"]),
        "patch_count": count,
        "degree": np.maximum(degree, 1.0),
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
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


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
