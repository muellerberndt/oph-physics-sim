from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.defects.gravity_assay import (
    free_two_defect_dynamics_report,
    organic_defect_population_report,
    two_defect_stress_contraction_assay_report,
    write_free_two_defect_dynamics_report,
    write_organic_defect_population_report,
    write_two_defect_stress_contraction_assay_report,
)


def test_two_defect_stress_contraction_assay_beats_controls_without_physical_claim() -> None:
    report = two_defect_stress_contraction_assay_report(
        patch_count=4096,
        steps=16,
        support_node_count=6,
        stress_coupling=0.04,
    )

    stress = report["stress_contraction_summary"]
    no_contraction = report["no_contraction_control_summary"]
    shuffled = report["shuffled_pair_control_summary"]

    assert report["controlled_planted_assay"] is True
    assert report["s3_inverse_identity_pass"] is True
    assert report["two_defect_stress_contraction_assay_receipt"] is True
    assert report["gravity_like_attraction_diagnostic_receipt"] is True
    assert report["control_rejection_receipt"] is True
    assert stress["final_h3_separation"] < stress["initial_h3_separation"]
    assert no_contraction["approach_fraction"] == 0.0
    assert shuffled["approach_fraction"] < 0.0
    assert report["production_gravity_receipt"] is False
    assert report["physical_gravity_prediction"] is False
    assert report["particle_matter_receipt"] is False
    assert "not spontaneous particle" in report["claim_boundary"]


def test_two_defect_stress_contraction_writer_emits_json_and_csv(tmp_path: Path) -> None:
    out = tmp_path / "two_defect_gravity_assay.json"

    report = write_two_defect_stress_contraction_assay_report(out, patch_count=4096, steps=8)
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert loaded["mode"] == "controlled_two_defect_stress_contraction_assay_v0"
    assert loaded["two_defect_stress_contraction_assay_receipt"] == report[
        "two_defect_stress_contraction_assay_receipt"
    ]
    assert (tmp_path / "two_defect_gravity_assay_trajectory.csv").exists()
    assert (tmp_path / "two_defect_gravity_assay_controls.csv").exists()


def test_free_two_defect_dynamics_exports_randomized_contact_bookkeeping() -> None:
    report = free_two_defect_dynamics_report(
        patch_count=4096,
        steps=48,
        support_node_count=6,
        seed=1234,
        initial_speed=0.04,
        transverse_kick=0.012,
    )

    summary = report["free_dynamics_summary"]
    first = report["trajectory_rows"][0]

    assert report["mode"] == "free_two_defect_dynamics_v0"
    assert report["controlled_planted_assay"] is False
    assert report["free_dynamics_diagnostic"] is True
    assert report["free_two_defect_dynamics_receipt"] is True
    assert summary["explicit_contact_outcome"] is True
    assert summary["contact_outcome"] in {"scatter", "bind", "annihilate", "pass_through"}
    assert summary["charge_conservation_pass"] is True
    assert summary["transverse_motion_present"] is True
    assert summary["straight_x_axis_control"] is False
    assert first["support_overlap_node_count"] <= 6
    assert len(report["worldlines"]) == 2
    assert report["worldlines"][0]["worldline_id"] == "free_pair_left"
    assert report["worldlines"][0]["events"][0]["velocity"]
    assert report["production_gravity_receipt"] is False
    assert report["physical_gravity_prediction"] is False
    assert "not production" in report["claim_boundary"]


def test_free_two_defect_dynamics_writer_emits_json_and_csv(tmp_path: Path) -> None:
    out = tmp_path / "free_two_defect_dynamics_report.json"

    report = write_free_two_defect_dynamics_report(out, patch_count=4096, steps=12, support_node_count=5)
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert loaded["mode"] == "free_two_defect_dynamics_v0"
    assert loaded["free_two_defect_dynamics_receipt"] == report["free_two_defect_dynamics_receipt"]
    assert (tmp_path / "free_two_defect_dynamics_report_trajectory.csv").exists()


def test_organic_defect_population_exports_natural_multi_worldline_renderings() -> None:
    report = organic_defect_population_report(
        patch_count=4096,
        steps=48,
        defect_count=14,
        min_defects=10,
        max_defects=20,
        support_node_count=6,
        seed=1234,
    )
    summary = report["organic_population_summary"]

    assert report["mode"] == "organic_defect_population_v0"
    assert report["controlled_planted_assay"] is False
    assert report["organic_defect_population_receipt"] is True
    assert report["organic_proto_worldline_visualization_receipt"] is True
    assert summary["defect_count_in_requested_band"] is True
    assert summary["worldline_count"] == 14
    assert summary["staggered_births_present"] is True
    assert summary["transverse_motion_present"] is True
    assert summary["fixed_left_right_pair"] is False
    assert len(report["worldlines"]) == 14
    assert all(not row["worldline_id"].startswith(("stress_pair", "free_pair")) for row in report["worldlines"])
    assert report["worldlines"][0]["events"][0]["render_modes"] == [
        "h3_point",
        "edge_string",
        "subjective_observer_3d_point",
    ]
    assert report["production_gravity_receipt"] is False
    assert report["physical_gravity_prediction"] is False
    assert "not particle matter" in report["claim_boundary"]


def test_organic_defect_population_writer_emits_json_and_csv(tmp_path: Path) -> None:
    out = tmp_path / "organic_defect_population_report.json"

    report = write_organic_defect_population_report(out, patch_count=4096, steps=24, defect_count=12)
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert loaded["mode"] == "organic_defect_population_v0"
    assert loaded["organic_defect_population_receipt"] == report["organic_defect_population_receipt"]
    assert (tmp_path / "organic_defect_population_report_trajectory.csv").exists()
    assert (tmp_path / "organic_defect_population_report_worldlines.csv").exists()
    assert (tmp_path / "organic_defect_population_report_worldline_events.csv").exists()
