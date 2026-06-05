from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from oph_fpe.bulk.cap_geometry import (
    RoundCap,
    apply_interpolation,
    cap_weights,
    interpolation_map,
    lambda_cap,
)
from oph_fpe.constants.oph_pixel import P_STAR, cap_area_planck, cap_entropy_capacity


@dataclass(frozen=True)
class BWResidualReport:
    median: float
    mean: float
    p90: float
    by_observable: dict[str, dict[str, float]]
    by_cap_size: dict[str, dict[str, float]]
    rows: list[dict[str, Any]]
    controls: dict[str, dict[str, Any]]

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "median": self.median,
            "mean": self.mean,
            "p90": self.p90,
            "by_observable": self.by_observable,
            "by_cap_size": self.by_cap_size,
            "rows": self.rows,
            "controls": self.controls,
        }


def bw_residual_report(
    points: np.ndarray,
    fields: dict[str, np.ndarray],
    caps: list[RoundCap],
    times: list[float],
    *,
    k_interp: int = 8,
    sim_k_interp: int | None = None,
    target_scale: float = 2.0 * math.pi,
    sim_scale: float = 2.0 * math.pi,
    n_jobs: int = 1,
    controls: list[str] | None = None,
    seed: int = 1,
    cell_entropy: float | np.ndarray | None = None,
    cell_area_planck: float | np.ndarray | None = None,
) -> BWResidualReport:
    tree = cKDTree(points)
    entropy_measure = P_STAR / 4.0 if cell_entropy is None else cell_entropy
    area_measure = P_STAR if cell_area_planck is None else cell_area_planck
    sim_k = int(sim_k_interp if sim_k_interp is not None else k_interp)
    rows = _residual_rows(
        points,
        fields,
        caps,
        times,
        tree=tree,
        target_k_interp=k_interp,
        sim_k_interp=sim_k,
        target_scale=target_scale,
        sim_scale=sim_scale,
        n_jobs=n_jobs,
        mode="transport",
        seed=seed,
        cell_entropy=entropy_measure,
        cell_area_planck=area_measure,
    )
    control_reports: dict[str, dict[str, Any]] = {}
    for control in controls or []:
        if control == "wrong_1x_normalization":
            control_reports[control] = _summary_rows(
                _residual_rows(
                    points,
                    fields,
                    caps,
                    times,
                    tree=tree,
                    target_k_interp=k_interp,
                    sim_k_interp=sim_k,
                    target_scale=target_scale,
                    sim_scale=1.0,
                    n_jobs=n_jobs,
                    mode="transport",
                    seed=seed,
                    cell_entropy=entropy_measure,
                    cell_area_planck=area_measure,
                )
            )
        elif control == "wrong_pi_normalization":
            control_reports[control] = _summary_rows(
                _residual_rows(
                    points,
                    fields,
                    caps,
                    times,
                    tree=tree,
                    target_k_interp=k_interp,
                    sim_k_interp=sim_k,
                    target_scale=target_scale,
                    sim_scale=math.pi,
                    n_jobs=n_jobs,
                    mode="transport",
                    seed=seed,
                    cell_entropy=entropy_measure,
                    cell_area_planck=area_measure,
                )
            )
        elif control == "wrong_4pi_normalization":
            control_reports[control] = _summary_rows(
                _residual_rows(
                    points,
                    fields,
                    caps,
                    times,
                    tree=tree,
                    target_k_interp=k_interp,
                    sim_k_interp=sim_k,
                    target_scale=target_scale,
                    sim_scale=4.0 * math.pi,
                    n_jobs=n_jobs,
                    mode="transport",
                    seed=seed,
                    cell_entropy=entropy_measure,
                    cell_area_planck=area_measure,
                )
            )
        elif control == "no_modular_flow":
            control_reports[control] = _summary_rows(
                _residual_rows(
                    points,
                    fields,
                    caps,
                    times,
                    tree=tree,
                    target_k_interp=k_interp,
                    sim_k_interp=sim_k,
                    target_scale=target_scale,
                    sim_scale=0.0,
                    n_jobs=n_jobs,
                    mode="transport",
                    seed=seed,
                    cell_entropy=entropy_measure,
                    cell_area_planck=area_measure,
                )
            )
        elif control == "shuffled_observables":
            shuffled = _shuffled_fields(fields, seed)
            control_reports[control] = _summary_rows(
                _residual_rows(
                    points,
                    shuffled,
                    caps,
                    times,
                    tree=tree,
                    target_k_interp=k_interp,
                    sim_k_interp=sim_k,
                    target_scale=target_scale,
                    sim_scale=sim_scale,
                    n_jobs=n_jobs,
                    mode="shuffled_observables",
                    seed=seed,
                    target_fields=fields,
                    cell_entropy=entropy_measure,
                    cell_area_planck=area_measure,
                )
            )
        elif control in {"shuffled_caps", "randomized_cap_axes"}:
            shuffled_caps = _shuffled_caps(caps, seed)
            control_reports[control] = _summary_rows(
                _residual_rows(
                    points,
                    fields,
                    shuffled_caps,
                    times,
                    tree=tree,
                    target_k_interp=k_interp,
                    sim_k_interp=sim_k,
                    target_scale=target_scale,
                    sim_scale=sim_scale,
                    n_jobs=n_jobs,
                    mode="transport",
                    seed=seed,
                    target_caps=caps,
                    cell_entropy=entropy_measure,
                    cell_area_planck=area_measure,
                )
            )
    summary = _summary_rows(rows)
    return BWResidualReport(
        median=float(summary["median"]),
        mean=float(summary["mean"]),
        p90=float(summary["p90"]),
        by_observable=_group_summary(rows, "observable"),
        by_cap_size=_group_summary(rows, "theta0"),
        rows=rows,
        controls=control_reports,
    )


def _residual_rows(
    points: np.ndarray,
    sim_fields: dict[str, np.ndarray],
    caps: list[RoundCap],
    times: list[float],
    *,
    tree: cKDTree,
    target_k_interp: int,
    sim_k_interp: int,
    target_scale: float,
    sim_scale: float,
    n_jobs: int,
    mode: str,
    seed: int,
    target_fields: dict[str, np.ndarray] | None = None,
    target_caps: list[RoundCap] | None = None,
    cell_entropy: float | np.ndarray = P_STAR / 4.0,
    cell_area_planck: float | np.ndarray = P_STAR,
) -> list[dict[str, Any]]:
    target_fields = target_fields or sim_fields
    target_caps = target_caps or caps
    tasks = [(index, cap, target_caps[index % len(target_caps)], time) for index, cap in enumerate(caps) for time in times]
    if n_jobs <= 1:
        chunks = [
            _cap_time_rows(
                points,
                sim_fields,
                target_fields,
                cap,
                target_cap,
                cap_index,
                time,
                tree,
                target_k_interp,
                sim_k_interp,
                target_scale,
                sim_scale,
                mode,
                cell_entropy,
                cell_area_planck,
            )
            for cap_index, cap, target_cap, time in tasks
        ]
    else:
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as pool:
            chunks = list(
                pool.map(
                    lambda item: _cap_time_rows(
                        points,
                        sim_fields,
                        target_fields,
                        item[1],
                        item[2],
                        item[0],
                        item[3],
                        tree,
                        target_k_interp,
                        sim_k_interp,
                        target_scale,
                        sim_scale,
                        mode,
                        cell_entropy,
                        cell_area_planck,
                    ),
                    tasks,
                )
            )
    return [row for chunk in chunks for row in chunk]


def _cap_time_rows(
    points: np.ndarray,
    sim_fields: dict[str, np.ndarray],
    target_fields: dict[str, np.ndarray],
    sim_cap: RoundCap,
    target_cap: RoundCap,
    cap_index: int,
    time: float,
    tree: cKDTree,
    target_k_interp: int,
    sim_k_interp: int,
    target_scale: float,
    sim_scale: float,
    mode: str,
    cell_entropy: float | np.ndarray,
    cell_area_planck: float | np.ndarray,
) -> list[dict[str, Any]]:
    target_query = lambda_cap(points, target_cap, target_scale * time)
    sim_query = lambda_cap(points, sim_cap, sim_scale * time)
    target_indices, target_weights = interpolation_map(points, target_query, k=target_k_interp, tree=tree)
    sim_indices, sim_weights = interpolation_map(points, sim_query, k=sim_k_interp, tree=tree)
    cap_w = cap_weights(points, target_cap, soft=True)
    residual_weights = np.asarray(cell_entropy, dtype=float) * cap_w
    cap_area = cap_area_planck(cap_w, cell_area_planck)
    cap_entropy = cap_entropy_capacity(cap_w, cell_entropy)
    rows = []
    for name, target_values in target_fields.items():
        sim_values = sim_fields[name]
        target = apply_interpolation(target_values, target_indices, target_weights)
        simulated = apply_interpolation(sim_values, sim_indices, sim_weights)
        residual = _weighted_bw_residual(target_values, simulated, target, residual_weights)
        rows.append(
            {
                "mode": mode,
                "observable": name,
                "cap_index": cap_index,
                "theta0": f"{target_cap.theta0:.6f}",
                "time": float(time),
                "target_scale": float(target_scale),
                "sim_scale": float(sim_scale),
                "weight_measure": "cell_entropy_capacity",
                "cap_area_planck": cap_area,
                "cap_entropy_capacity": cap_entropy,
                "residual": residual,
            }
        )
    return rows


def _weighted_bw_residual(before: np.ndarray, left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    diff = left - right
    numerator = float(np.sqrt(np.sum(weights * diff * diff)))
    denominator = float(np.sqrt(np.sum(weights * before * before)))
    return numerator / (denominator + 1e-12)


def _summary_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.array([float(row["residual"]) for row in rows], dtype=float)
    if values.size == 0:
        return {"median": float("nan"), "mean": float("nan"), "p90": float("nan"), "count": 0}
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "count": int(values.size),
    }


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return {group: _summary_rows(items) for group, items in sorted(grouped.items())}


def _shuffled_fields(fields: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}
    for name, values in fields.items():
        order = rng.permutation(values.size)
        result[name] = values[order]
    return result


def _shuffled_caps(caps: list[RoundCap], seed: int) -> list[RoundCap]:
    rng = np.random.default_rng(seed)
    axes = [cap.axis.copy() for cap in caps]
    rng.shuffle(axes)
    return [
        RoundCap(axis=axes[index], theta0=cap.theta0, tangent=cap.tangent, collar_width=cap.collar_width).normalized()
        for index, cap in enumerate(caps)
    ]
