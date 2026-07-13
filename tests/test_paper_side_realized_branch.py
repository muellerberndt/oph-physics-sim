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
        "evaluations": {
            "bulk_depth_receipts": {
                "receipts_witnessed": {"cone_signature_1_3_all_seeds": True},
                "receipts_pending": ["cone-margin convergence"],
                "caveats": ["narrow timelike margin"],
                "timelike_eigenvalues_across_seeds": [-0.02],
                "countermodel_signature": [2, 2],
            }
        },
    }
    with open(tmp_path / "realized_branch_receipt_report.json", "w") as f:
        json.dump(sidecar, f)
    report = einstein_bridge_manifest_report(tmp_path)
    block = report["paperSideRealizedBranch"]
    assert block["issue"] == 503
    assert block["certified_nonempty"] is False
    assert block["tiers"]["bulk_depth_channel_witnessed"] is True
    assert block["open_obligations"]["bulk_depth_receipts"] == ["cone-margin convergence"]
    assert block["numerical_diagnostics"]["countermodel_signature"] == [2, 2]
    assert block["artifact_pin"]["manifest_verified"] is False
    # informational only: the run-receipt gate keeps its own verdict
    assert report["einstein_branch_entry_receipt"] in (True, False)


def test_sidecar_truthy_strings_do_not_become_paper_tier_booleans(tmp_path: Path) -> None:
    sidecar = {
        "artifact": "einstein_branch_realized_receipt_evaluation",
        "issue": 503,
        "status": "OPEN",
        "bulk_depth_channel_witnessed": "true",
        "realized_geometric_branch_certified_nonempty": "closed",
    }
    (tmp_path / "realized_branch_receipt_report.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )

    block = einstein_bridge_manifest_report(tmp_path)["paperSideRealizedBranch"]

    assert block["tiers"]["bulk_depth_channel_witnessed"] is False
    assert block["certified_nonempty"] is False
