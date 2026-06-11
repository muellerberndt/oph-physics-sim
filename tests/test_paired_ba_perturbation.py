from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oph_fpe.bulk.cap_geometry import RoundCap
from oph_fpe.core.graph import fibonacci_sphere_points
from oph_fpe.cosmology.comparable_data import comparable_data_report
from oph_fpe.cosmology.paired_ba_perturbation import (
    paired_perturb_resettle_b_a_report,
    write_paired_perturb_resettle_b_a_report,
)
from oph_fpe.scale.array_screen import _knn_edges, _node_signature


def _toy_state(patch_count: int = 96):
    points = fibonacci_sphere_points(patch_count)
    left, right = _knn_edges(points, 4)
    group_order = 6
    port_left = (left % group_order).astype(np.int16)
    port_right = (right % group_order).astype(np.int16)
    signature = _node_signature(port_left, port_right, left, right, patch_count)
    degree = np.bincount(np.concatenate([left, right]), minlength=patch_count).astype(float)
    raw_fields = {
        "record_signature": signature.astype(float),
        "stable_count": np.arange(patch_count, dtype=float) % 8.0,
        "committed_mask": np.ones(patch_count, dtype=float),
        "repair_load": np.zeros(patch_count, dtype=float),
        "local_mismatch_density": np.zeros(patch_count, dtype=float),
        "cumulative_repair_load": np.linspace(0.1, 1.1, patch_count),
        "s3_sector_class": np.mod(signature, 6).astype(np.int64),
        "s3_class_density": np.mod(signature, 3).astype(float),
    }
    graph = {
        "left": left,
        "right": right,
        "port_left": port_left,
        "port_right": port_right,
        "group_order": group_order,
        "patch_count": patch_count,
        "degree": np.maximum(degree, 1.0),
    }
    cap = RoundCap(
        axis=np.array([0.0, 0.0, 1.0]),
        theta0=0.8,
        tangent=np.array([1.0, 0.0, 0.0]),
        collar_width=0.12,
    )
    return points, [cap], raw_fields, graph


def test_paired_perturb_resettle_b_a_emits_real_rerun_rows_but_keeps_gate_closed():
    points, caps, raw_fields, graph = _toy_state()

    report = paired_perturb_resettle_b_a_report(
        points,
        caps,
        raw_fields,
        graph,
        a_grid=[1.0],
        times=[0.05],
        modes_per_cap_time=1,
        controls=["no_perturbation", "no_repair_load_channel"],
        repair_steps=2,
        repairs_per_step=16,
        seed=42,
    )

    assert report["rows"]
    assert report["control_rows"]
    assert report["primary_parent_source"] == "paired_cap_collar_perturb_resettle_rerun"
    assert report["readiness"]["checks"]["real_baryon_perturbation_runs_present"] is True
    assert report["readiness"]["checks"]["report_backed_surrogate_parent"] is False
    assert report["B_A_PARENT_RECEIPT"] is False
    assert report["physical_cmb_prediction"] is False
    assert {row["control"] for row in report["control_rows"]} == {"no_perturbation", "no_repair_load_channel"}
    assert all(row["B_A_mean"] == 0.0 for row in report["control_rows"])


def test_write_paired_perturb_resettle_b_a_report_uses_b_a_parent_contract(tmp_path: Path):
    points, caps, raw_fields, graph = _toy_state()

    report = write_paired_perturb_resettle_b_a_report(
        tmp_path,
        points,
        caps,
        raw_fields,
        graph,
        a_grid=[1.0],
        times=[0.05],
        modes_per_cap_time=1,
        controls=["no_perturbation"],
        repair_steps=1,
        repairs_per_step=8,
        seed=17,
    )

    assert (tmp_path / "paired_b_a_perturbation_report.json").exists()
    assert (tmp_path / "b_a_parent_report.json").exists()
    assert (tmp_path / "paired_b_a_perturbation_rows.csv").exists()
    assert json.loads((tmp_path / "b_a_parent_report.json").read_text(encoding="utf-8"))["mode"] == report["mode"]

    comparable = comparable_data_report([tmp_path])
    lane = comparable["measurement_lanes"]["oph_B_A_parent_finite_difference"]
    assert lane["run_count"] == 1
    assert lane["real_baryon_perturbation_run_count"] == 1
    assert lane["primary_parent_source_counts"]["paired_cap_collar_perturb_resettle_rerun"] == 1
