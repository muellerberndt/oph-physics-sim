from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.defects.gravity_assay import (
    two_defect_stress_contraction_assay_report,
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
