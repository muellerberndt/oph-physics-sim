from __future__ import annotations

import math
from typing import Any

import numpy as np


def host_mixture_identifiability(
    rows: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...] = ("SFR", "M_star_old", "M_GC"),
    exposure_key: str = "exposure",
    tol: float = 1.0e-10,
) -> dict[str, Any]:
    matrix = []
    weights = []
    for row in rows:
        matrix.append([float(row.get(name, 0.0)) for name in feature_names])
        weights.append(max(0.0, float(row.get(exposure_key, 1.0))))
    if not matrix:
        return {
            "HOST_MIXTURE_IDENTIFIABILITY_RECEIPT": False,
            "rank": 0,
            "required_rank": len(feature_names),
            "feature_names": list(feature_names),
            "status": "no_host_rows",
        }
    design = np.asarray(matrix, dtype=float)
    weight = np.sqrt(np.asarray(weights, dtype=float))[:, None]
    weighted = design * weight
    rank = int(np.linalg.matrix_rank(weighted, tol=tol))
    return {
        "HOST_MIXTURE_IDENTIFIABILITY_RECEIPT": rank == len(feature_names),
        "rank": rank,
        "required_rank": len(feature_names),
        "feature_names": list(feature_names),
        "row_count": len(rows),
        "status": "full_rank" if rank == len(feature_names) else "exposure_weighted_collinearity",
    }


def repair_reload_waiting_time_shift(
    previous_fluence: float,
    *,
    reservoir_before: float,
    fluence_to_discharge: float,
    reload_rate: float,
    threshold: float,
) -> dict[str, Any]:
    discharge = max(0.0, float(previous_fluence) * float(fluence_to_discharge))
    post_burst = float(reservoir_before) - discharge
    if reload_rate <= 0.0:
        return {
            "FRB_REPAIR_RELOAD_RECEIPT": False,
            "waiting_time_to_threshold": None,
            "post_burst_reservoir": post_burst,
            "status": "nonpositive_reload_rate",
        }
    deficit = max(0.0, float(threshold) - post_burst)
    waiting_time = deficit / float(reload_rate)
    return {
        "FRB_REPAIR_RELOAD_RECEIPT": True,
        "previous_fluence": float(previous_fluence),
        "discharge": discharge,
        "post_burst_reservoir": post_burst,
        "waiting_time_to_threshold": waiting_time,
        "prediction": (
            "after source identity, cadence, exposure, and host conditioning, "
            "larger fluence shifts the next high-fluence waiting time later"
        ),
    }


def repair_reload_control_family() -> dict[str, Any]:
    return {
        "M0": "young_only",
        "M1": "young_plus_old_gc_poisson_or_weibull_timing",
        "M2": "young_plus_old_gc_repair_reload_timing",
        "promotion_condition": "logL_heldout(M2)-max(logL_heldout(M0),logL_heldout(M1))>Delta_min",
    }


def logistic_hazard(reservoir: float, *, lambda0: float, epsilon_c: float, scale: float) -> float:
    if scale == 0.0:
        raise ValueError("scale must be nonzero")
    return float(lambda0) / (1.0 + math.exp(-(float(reservoir) - float(epsilon_c)) / float(scale)))
