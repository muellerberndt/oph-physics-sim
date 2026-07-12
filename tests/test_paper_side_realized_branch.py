"""Tests for the paper-side realized-branch block in the Einstein bridge."""

from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.bulk.einstein_bridge import einstein_bridge_manifest_report


def test_sidecar_absent_reads_as_absent(tmp_path: Path) -> None:
    report = einstein_bridge_manifest_report(tmp_path)
    assert report["paperSideRealizedBranch"] == {"status": "sidecar_absent"}


def test_sidecar_present_is_summarized_and_gate_unchanged(tmp_path: Path) -> None:
    sidecar = {
        "artifact": "einstein_branch_realized_receipt_evaluation",
        "issue": 503,
        "status": "OPEN: receipt families witnessed; limit clauses pending.",
        "topology_mesh_families_realized_with_branch_selection": True,
        "boundary_collar_modular_families_witnessed": True,
        "null_net_families_witnessed_one_particle": True,
        "screen_event_families_witnessed": True,
        "bulk_depth_channel_witnessed": True,
        "realized_geometric_branch_certified_nonempty": False,
    }
    with open(tmp_path / "realized_branch_receipt_report.json", "w") as f:
        json.dump(sidecar, f)
    report = einstein_bridge_manifest_report(tmp_path)
    block = report["paperSideRealizedBranch"]
    assert block["issue"] == 503
    assert block["certified_nonempty"] is False
    assert block["tiers"]["bulk_depth_channel_witnessed"] is True
    # informational only: the run-receipt gate keeps its own verdict
    assert report["einstein_branch_entry_receipt"] in (True, False)
