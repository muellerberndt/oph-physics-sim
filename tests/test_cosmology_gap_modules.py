from __future__ import annotations

import csv
import json

import numpy as np

from oph_fpe.cosmology.adiabaticity import adiabaticity_report
from oph_fpe.cosmology.h0s8 import h0_from_flat_q_a, h0s8_branch_report
from oph_fpe.cosmology.hot_release import hot_release_report
from oph_fpe.cosmology.sync_gap import synchronization_gap_report


def test_h0s8_branch_report_keeps_theorem_gates_closed():
    report = h0s8_branch_report()
    assert report["flat_q_a_closure"]["H0_km_s_Mpc"] == np.float64(h0_from_flat_q_a(q_a=5.363470441))
    assert report["collar_tracking"]["lambda_collar"] == np.float64(np.exp(-report["inputs"]["P"] / 24.0))
    assert report["branches"]["B_direct_jacobi_repair"]["S8"] == np.float64(0.790)
    assert report["theorem_gates"]["Q_A_from_finite_collar_selector"] is False
    assert report["physical_cmb_prediction"] is False


def test_hot_release_detects_mechanical_surface_but_not_theorem_ready(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with (run / "mismatch_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cycle", "beta", "phi", "committed_fraction", "record_entropy"],
        )
        writer.writeheader()
        writer.writerow({"cycle": 0, "beta": 0.1, "phi": 12, "committed_fraction": 0.0, "record_entropy": 0.0})
        writer.writerow({"cycle": 4, "beta": 0.5, "phi": 0, "committed_fraction": 1.0, "record_entropy": 2.0})
    (run / "freezeout_map_summary.json").write_text(
        json.dumps({"freezeout_cycle": 4, "committed_fraction": 1.0}),
        encoding="utf-8",
    )
    (run / "collar_markov_report.json").write_text(
        json.dumps({"median_epsilon_cmi": 0.4, "p90_epsilon_cmi": 0.5}),
        encoding="utf-8",
    )
    report = hot_release_report([run])
    row = report["rows"][0]
    assert row["mechanical_release_surface_found"] is True
    assert row["release_temperature_proxy_inverse_beta"] == 2.0
    assert row["collar_markov_gate_pass"] is False
    assert report["hot_release_theorem_ready"] is False


def test_sync_gap_final_spectrum_proxy_does_not_establish_gap(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with (run / "mismatch_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cycle", "phi"])
        writer.writeheader()
        writer.writerow({"cycle": 0, "phi": 10})
        writer.writerow({"cycle": 1, "phi": 3})
        writer.writerow({"cycle": 2, "phi": 1})
    spectrum = [{"ell": ell, "D_ell": 1.0 / ell} for ell in range(2, 10)]
    control = [{"ell": ell, "D_ell": 0.1 / ell} for ell in range(2, 10)]
    (run / "cl_comparison_report.json").write_text(
        json.dumps(
            {
                "fields": {"record_signature": {"spectrum": spectrum}},
                "controls": {
                    "record_signature": {
                        "shuffled_field": {"spectrum": control},
                        "random_gaussian": {"spectrum": [{"ell": ell, "D_ell": 0.2 / ell} for ell in range(2, 10)]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    report = synchronization_gap_report([run], min_control_z=0.1)
    assert report["rows"][0]["cached_final_spectrum_gap_proxy_pass"] is True
    assert report["rows"][0]["time_resolved_harmonic_trace_available"] is False
    assert report["low_k_gap_established"] is False


def test_sync_gap_counts_time_resolved_harmonic_trace(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    np.savez_compressed(
        run / "harmonic_time_trace.npz",
        cycles=np.asarray([0, 1, 2, 3], dtype=np.int32),
        ell=np.asarray([0, 1, 2, 3], dtype=np.int32),
        record_signature=np.asarray(
            [
                [0.0, 0.0, 8.0, 4.0],
                [0.0, 0.0, 4.0, 2.0],
                [0.0, 0.0, 2.0, 1.0],
                [0.0, 0.0, 1.0, 0.5],
            ],
            dtype=np.float32,
        ),
    )

    report = synchronization_gap_report([run])

    assert report["aggregate"]["time_resolved_trace_count"] == 1
    assert report["rows"][0]["time_resolved_harmonic_trace_available"] is True
    assert report["rows"][0]["time_resolved_gap"]["available"] is True
    assert report["low_k_gap_established"] is False


def test_adiabaticity_proxy_flags_mismatched_channels(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    x = np.linspace(-1.0, 1.0, 64)
    np.savez(
        run / "freezeout_fields.npz",
        record_signature=x,
        cumulative_repair_load=-x,
        stable_count=np.sin(np.arange(64)),
        s3_class_density=np.cos(np.arange(64)),
    )
    report = adiabaticity_report([run], max_entropy_residual_std=0.1, min_common_clock_corr=0.9)
    row = report["rows"][0]
    assert row["has_adiabaticity_inputs"] is True
    assert row["adiabaticity_proxy_pass"] is False
    assert report["adiabaticity_established"] is False
    assert report["physical_cmb_prediction"] is False
