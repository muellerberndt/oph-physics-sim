from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Protocol

import numpy as np


DEFAULT_B_A_CONTROLS = [
    "phase_shuffled_baryon_mode",
    "wrong_k_label",
    "random_collar_labels",
    "no_repair_load_channel",
    "baryon_delta_applied_after_record_freezeout",
]


@dataclass(frozen=True)
class BaryonPerturbationMode:
    k_h_mpc: float
    seed: int
    mode_index: int
    phase: float
    control: str | None = None


class CollarResponseParent(Protocol):
    """Minimal callback surface for non-fit B_A finite differences."""

    def make_background(self, a: float) -> Any: ...

    def apply_baryon_delta(self, background: Any, amplitude: float, mode: BaryonPerturbationMode) -> Any: ...

    def rho_A(self, state: Any) -> float: ...

    def rho_A_eq(self, state: Any) -> float: ...

    def baryon_density(self, a: float) -> float: ...


def estimate_b_a_grid(
    collar_parent: CollarResponseParent,
    *,
    a_grid: Iterable[float],
    k_grid_h_mpc: Iterable[float],
    eps: float = 1.0e-3,
    modes_per_k: int = 16,
    seeds: Iterable[int] = range(8),
    controls: Iterable[str] = DEFAULT_B_A_CONTROLS,
) -> dict[str, Any]:
    """Estimate a diagnostic B_A(k,a) parent by finite baryon perturbations.

    This uses controlled plus/minus baryon perturbations and never consults CMB
    data. The emitted rows are diagnostic until refinement convergence, control
    failure, time/scale calibration, and energy-momentum closure are proven.
    """

    a_values = [float(value) for value in a_grid]
    k_values = [float(value) for value in k_grid_h_mpc]
    seed_values = [int(value) for value in seeds]
    control_values = [str(value) for value in controls]
    rows = [
        _estimate_row(collar_parent, a, k, eps=float(eps), modes_per_k=int(modes_per_k), seeds=seed_values)
        for a in a_values
        for k in k_values
    ]
    control_rows: list[dict[str, Any]] = []
    for control in control_values:
        for a in a_values:
            for k in k_values:
                control_rows.append(
                    _estimate_row(
                        collar_parent,
                        a,
                        k,
                        eps=float(eps),
                        modes_per_k=int(modes_per_k),
                        seeds=seed_values,
                        control=control,
                    )
                )
    readiness = _readiness(rows, control_rows)
    return {
        "mode": "finite_difference_baryon_response_B_A_parent_v0",
        "a_grid": a_values,
        "k_grid_h_mpc": k_values,
        "eps": float(eps),
        "modes_per_k": int(modes_per_k),
        "seed_count": len(seed_values),
        "rows": rows,
        "control_rows": control_rows,
        "readiness": readiness,
        "B_A_PARENT_RECEIPT": readiness["B_A_PARENT_RECEIPT"],
        "physical_prediction_ready": False,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Finite-difference collar response to controlled baryon perturbations. "
            "Rows are non-fit B_A(k,a) parent diagnostics; they are not physical "
            "Boltzmann inputs until convergence, controls, scale/time calibration, "
            "and energy-momentum exchange closure pass."
        ),
    }


def _estimate_row(
    parent: CollarResponseParent,
    a: float,
    k_h_mpc: float,
    *,
    eps: float,
    modes_per_k: int,
    seeds: list[int],
    control: str | None = None,
) -> dict[str, Any]:
    background = parent.make_background(float(a))
    rho_a = max(float(parent.rho_A(background)), 1.0e-300)
    rho_b = max(float(parent.baryon_density(float(a))), 1.0e-300)
    estimates: list[float] = []
    for seed in seeds:
        for mode_index in range(int(modes_per_k)):
            mode = _sample_mode(float(k_h_mpc), int(seed), int(mode_index), control=control)
            plus = _apply_delta(parent, background, +float(eps), mode)
            minus = _apply_delta(parent, background, -float(eps), mode)
            derivative = (float(parent.rho_A_eq(plus)) - float(parent.rho_A_eq(minus))) / (2.0 * float(eps))
            k_parent = derivative / rho_b
            estimates.append(float((rho_b / rho_a) * k_parent))
    values = np.asarray(estimates, dtype=float)
    mean = float(np.mean(values)) if values.size else None
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    sign_stable = _sign_stable(values)
    return {
        "a": float(a),
        "k_h_mpc": float(k_h_mpc),
        "control": control,
        "B_A_mean": mean,
        "B_A_std": std,
        "B_A_sem": float(std / np.sqrt(values.size)) if values.size else None,
        "mode_count": int(values.size),
        "sign_stable": bool(sign_stable),
        "source": "finite_collar_baryon_perturbation_derivative",
    }


def _sample_mode(k_h_mpc: float, seed: int, mode_index: int, *, control: str | None) -> BaryonPerturbationMode:
    rng = np.random.default_rng(seed * 1_000_003 + mode_index * 97 + 11)
    phase = 1.0
    k_for_parent = float(k_h_mpc)
    if control == "phase_shuffled_baryon_mode":
        phase = float(rng.choice([-1.0, 1.0]))
    elif control == "wrong_k_label":
        k_for_parent = float(k_h_mpc) * 1.7
    return BaryonPerturbationMode(
        k_h_mpc=k_for_parent,
        seed=int(seed),
        mode_index=int(mode_index),
        phase=phase,
        control=control,
    )


def _apply_delta(
    parent: CollarResponseParent,
    background: Any,
    amplitude: float,
    mode: BaryonPerturbationMode,
) -> Any:
    try:
        return parent.apply_baryon_delta(background, float(amplitude), mode)
    except TypeError:
        return parent.apply_baryon_delta(background, float(amplitude), mode.k_h_mpc)


def _sign_stable(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    mean = float(np.mean(values))
    if abs(mean) < 1.0e-12:
        return False
    signs = np.sign(values[np.abs(values) > 1.0e-12])
    if signs.size == 0:
        return False
    return bool(float(np.mean(signs == np.sign(mean))) >= 0.8)


def _readiness(rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sign_stable = bool(rows and all(bool(row.get("sign_stable")) for row in rows))
    controls_by_name: dict[str, list[dict[str, Any]]] = {}
    for row in control_rows:
        controls_by_name.setdefault(str(row.get("control")), []).append(row)
    main_abs = [abs(float(row["B_A_mean"])) for row in rows if isinstance(row.get("B_A_mean"), (int, float))]
    main_scale = fmean(main_abs) if main_abs else 0.0
    control_failures: dict[str, bool] = {}
    for name, group in controls_by_name.items():
        values = [abs(float(row["B_A_mean"])) for row in group if isinstance(row.get("B_A_mean"), (int, float))]
        control_failures[name] = bool(values and fmean(values) < 0.5 * max(main_scale, 1.0e-300))
    controls_fail = bool(control_failures and all(control_failures.values()))
    checks = {
        "finite_difference_rows_emitted": bool(rows),
        "no_cmb_data_used": True,
        "sign_stable": sign_stable,
        "controls_fail": controls_fail,
        "convergence_64k_256k": False,
        "rho_A_of_a_emitted": False,
        "rho_A_eq_of_a_emitted": False,
        "Gamma_rec_of_k_a_emitted": False,
        "energy_momentum_exchange_closed": False,
        "gauge_consistency_audited": False,
    }
    return {
        "checks": checks,
        "control_failures": control_failures,
        "missing_gates": [name for name, passed in checks.items() if not passed],
        "B_A_PARENT_RECEIPT": False,
        "physical_prediction_ready": False,
    }
