from __future__ import annotations

import pytest

from oph_fpe.cosmology.ba_parent import BaryonPerturbationMode, estimate_b_a_grid


class LinearCollarParent:
    def make_background(self, a: float):
        return {"a": float(a), "delta": 0.0, "mode": None}

    def apply_baryon_delta(self, background, amplitude: float, mode: BaryonPerturbationMode):
        state = dict(background)
        state["delta"] = float(amplitude)
        state["mode"] = mode
        return state

    def rho_A(self, state) -> float:
        return 2.0 * float(state["a"]) ** -3

    def rho_A_eq(self, state) -> float:
        a = float(state["a"])
        mode = state.get("mode")
        if mode is None:
            return a**-3
        if mode.control in {"random_collar_labels", "no_repair_load_channel", "baryon_delta_applied_after_record_freezeout"}:
            return a**-3
        return a**-3 + state["delta"] * mode.phase * mode.k_h_mpc

    def baryon_density(self, a: float) -> float:
        return 0.5 * float(a) ** -3


def test_estimate_b_a_grid_uses_symmetric_finite_difference_without_cmb_fit():
    report = estimate_b_a_grid(
        LinearCollarParent(),
        a_grid=[0.5],
        k_grid_h_mpc=[0.25],
        eps=1.0e-4,
        modes_per_k=4,
        seeds=[1, 2],
        controls=["no_repair_load_channel"],
    )

    row = report["rows"][0]
    # derivative = k, B_A = derivative / rho_A = 0.25 / (2 * 0.5^-3)
    assert row["B_A_mean"] == pytest.approx(0.015625)
    assert row["sign_stable"] is True
    assert report["readiness"]["checks"]["no_cmb_data_used"] is True
    assert report["physical_prediction_ready"] is False
    assert report["B_A_PARENT_RECEIPT"] is False


def test_estimate_b_a_grid_reports_control_failures_but_keeps_physical_gate_closed():
    report = estimate_b_a_grid(
        LinearCollarParent(),
        a_grid=[1.0],
        k_grid_h_mpc=[0.5],
        eps=1.0e-4,
        modes_per_k=2,
        seeds=[1],
        controls=["no_repair_load_channel"],
    )

    assert report["readiness"]["control_failures"]["no_repair_load_channel"] is True
    assert "convergence_64k_256k" in report["readiness"]["missing_gates"]
    assert "energy_momentum_exchange_closed" in report["readiness"]["missing_gates"]
