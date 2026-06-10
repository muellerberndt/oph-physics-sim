from __future__ import annotations

import json

import pytest

from oph_fpe.cosmology.ba_parent import BaryonPerturbationMode, estimate_b_a_grid, write_b_a_parent_report


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


def test_write_b_a_parent_report_uses_stress_report_surrogate_and_keeps_gate_closed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "oph_cmb_stress_report.json").write_text(
        """
{
  "finite_collar_parent": {
    "status": "diagnostic_finite_collar_samples_not_theorem_grade",
    "weighted_collar_repair_defect_R": 1.5,
    "mean_epsilon_cmi": 1.25
  },
  "diagnostic_kernel_proxy": {
    "status": "screen_cap_size_shape_proxy_only",
    "a_grid": [0.5, 1.0],
    "kernel_proxy_rows": [
      {"k_proxy_inverse_theta": 1.0, "B_A_shape_proxy": 0.25},
      {"k_proxy_inverse_theta": 2.0, "B_A_shape_proxy": -0.5}
    ],
    "B_A_k_a_emitted": false
  }
}
""",
        encoding="utf-8",
    )

    out = tmp_path / "out"
    report = write_b_a_parent_report(
        [run],
        out,
        eps=1.0e-4,
        modes_per_k=2,
        seeds=[1],
        controls=["no_repair_load_channel"],
    )

    assert report["source_report_count"] == 1
    assert report["readiness"]["checks"]["finite_difference_rows_emitted"] is True
    assert report["readiness"]["checks"]["no_cmb_data_used"] is True
    assert report["readiness"]["checks"]["real_baryon_perturbation_runs_present"] is False
    assert report["readiness"]["checks"]["report_backed_surrogate_parent"] is True
    assert report["B_A_PARENT_RECEIPT"] is False
    assert report["physical_prediction_ready"] is False
    assert (out / "b_a_parent_report.json").exists()
    assert (out / "b_a_parent_rows.csv").read_text(encoding="utf-8").count("inverse_cap_opening_angle_proxy") >= 1


def test_write_b_a_parent_report_prefers_observer_view_packet_variation(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    caps = [
        {
            "axis": [0.0, 0.0, 1.0],
            "theta0": 0.9,
            "tangent": [1.0, 0.0, 0.0],
            "collar_width": 0.2,
        }
    ]
    (run / "modular_response_kernel_cache.json").write_text(json.dumps({"caps": caps}), encoding="utf-8")
    rows = []
    for index in range(48):
        z = -1.0 + 2.0 * index / 47.0
        x = (1.0 - z * z) ** 0.5
        rows.append(
            {
                "view_type": "patch_observer",
                "observer_id": index,
                "axis": [x, 0.0, z],
                "dominant_record_signature": index % 7,
                "dominant_object_packet": index % 5,
                "modular_response_cluster": index % 11,
                "transition_history_key": index % 13,
                "visible_signature_entropy": float(index % 8),
                "record_stability_mean": float(10 + index % 9),
                "counterfactual_stability": 0.5,
                "transition_history_mean_modal_mass": 0.75,
                "repair_load_mean": float(index % 3),
                "mismatch_density_mean": 0.0,
                "perturb_resettle_signature": [float(index % 4), float(index % 2)],
            }
        )
    (run / "observer_views.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    out = tmp_path / "out"
    report = write_b_a_parent_report(
        [run],
        out,
        a_grid=[1.0],
        k_grid_h_mpc=[1.0],
        eps=1.0e-3,
        modes_per_k=2,
        seeds=[1, 2],
        controls=["no_repair_load_channel", "baryon_delta_applied_after_record_freezeout"],
    )

    assert report["primary_parent_source"] == "observer_view_finite_collar_packet_variation"
    assert report["observer_view_source_count"] == 1
    assert report["readiness"]["checks"]["finite_observer_view_parent_variation"] is True
    assert report["readiness"]["checks"]["real_baryon_perturbation_runs_present"] is False
    assert report["B_A_PARENT_RECEIPT"] is False
    assert report["rows"]
    assert report["rows"][0]["source"] == "observer_view_finite_collar_packet_variation"
    assert (out / "b_a_parent_observer_view_rows.csv").exists()
