from __future__ import annotations

from oph_fpe.homochirality import homochirality_demo_report
from oph_fpe.viz.physics_problem_outputs_viewer import write_physics_problem_outputs_viewer


def test_physics_problem_outputs_viewer_is_compact_and_explicit(tmp_path) -> None:
    report = {
        "schema": "oph_physics_problem_outputs_v1",
        "outputs": {
            "homochirality": homochirality_demo_report(),
            "open_example": {
                "status": "input_gated_contract",
                "physicalClaim": False,
                "claimBoundary": "Open example remains a contract.",
            },
        },
    }
    destination = tmp_path / "physics_problem_outputs_viewer.html"

    summary = write_physics_problem_outputs_viewer(report, destination)
    html = destination.read_text(encoding="utf-8")

    assert summary["output_count"] == 2
    assert summary["physical_claim"] is False
    assert "Assumed for demo" in html
    assert "Homochirality as record-branch selection" in html
    assert "All registered problem contracts" in html
    assert "does not choose the terrestrial hand" in html
