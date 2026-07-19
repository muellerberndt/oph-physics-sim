from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.black_hole_bridge import (
    FINITE_HORIZON_RECORD_GATES,
    PAGE_CURVE_GATES,
    PHYSICAL_EVAPORATION_GATES,
    QNM_RADIATIVE_GATES,
    BlackHoleBridgeInputs,
    black_hole_bridge_status_report,
    write_black_hole_bridge_status_report,
)
from oph_fpe.measurement_pack import export_measurement_pack


def test_default_black_hole_bridge_report_fails_closed_without_artifact() -> None:
    report = black_hole_bridge_status_report()

    assert report["BLACK_HOLE_BRIDGE_STATUS_CONTRACT_RECEIPT"] is True
    assert report["FINITE_HORIZON_RECORD_REPAIR_DIAGNOSTIC_RECEIPT"] is False
    assert report["BLACK_HOLE_PHYSICAL_EVAPORATION_BRIDGE_RECEIPT"] is False
    assert report["BLACK_HOLE_PHYSICAL_PAGE_CURVE_RECEIPT"] is False
    assert report["BLACK_HOLE_QNM_RADIATIVE_BRIDGE_RECEIPT"] is False
    assert report["physical_claims"]["black_hole_information_problem_solved"] is False
    assert "source_artifact_missing" in report["blockers"]


def test_finite_horizon_record_diagnostic_does_not_promote_page_or_qnm(tmp_path: Path) -> None:
    artifact = tmp_path / "finite.json"
    artifact.write_text(
        json.dumps({"readiness_gates": {name: True for name in FINITE_HORIZON_RECORD_GATES}}),
        encoding="utf-8",
    )

    report = black_hole_bridge_status_report(BlackHoleBridgeInputs(source_artifact=artifact))

    assert report["FINITE_HORIZON_RECORD_REPAIR_DIAGNOSTIC_RECEIPT"] is False
    assert report["BLACK_HOLE_PHYSICAL_EVAPORATION_BRIDGE_RECEIPT"] is False
    assert report["BLACK_HOLE_PHYSICAL_PAGE_CURVE_RECEIPT"] is False
    assert report["BLACK_HOLE_QNM_RADIATIVE_BRIDGE_RECEIPT"] is False
    assert report["physical_claims"]["physical_page_time_claim"] is False
    assert report["declared_readiness_gate_assertions"]["finite_horizon_record"] == {
        name: True for name in FINITE_HORIZON_RECORD_GATES
    }
    assert "caller_gate_assertions_are_not_independent_receipts" in report["blockers"]


def test_complete_caller_authored_artifact_cannot_pass_physical_receipts(tmp_path: Path) -> None:
    artifact = tmp_path / "complete.json"
    gates = {
        **{name: True for name in FINITE_HORIZON_RECORD_GATES},
        **{name: True for name in PHYSICAL_EVAPORATION_GATES},
        **{name: True for name in PAGE_CURVE_GATES},
        **{name: True for name in QNM_RADIATIVE_GATES},
    }
    artifact.write_text(
        json.dumps(
            {
                "readiness_gates": gates,
                "full_microstate_coverage": True,
                "standalone_information_problem_solution_gate": True,
            }
        ),
        encoding="utf-8",
    )

    report = write_black_hole_bridge_status_report(
        tmp_path / "out",
        BlackHoleBridgeInputs(source_artifact=artifact),
    )

    assert report["BLACK_HOLE_PHYSICAL_EVAPORATION_BRIDGE_RECEIPT"] is False
    assert report["BLACK_HOLE_PHYSICAL_PAGE_CURVE_RECEIPT"] is False
    assert report["BLACK_HOLE_QNM_RADIATIVE_BRIDGE_RECEIPT"] is False
    assert report["BLACK_HOLE_PHYSICAL_SIMULATION_RECEIPT"] is False
    assert report["physical_claims"]["black_hole_information_problem_solved"] is False
    assert report["independent_gate_verifier_available"] is False
    assert report["caller_positive_gate_assertions_ignored"]
    assert (tmp_path / "out" / "black_hole_bridge_status_report.json").exists()
    assert (tmp_path / "out" / "black_hole_bridge_status_report.md").exists()


def test_forbidden_target_dependency_blocks_physical_promotion(tmp_path: Path) -> None:
    artifact = tmp_path / "leaky.json"
    gates = {
        **{name: True for name in FINITE_HORIZON_RECORD_GATES},
        **{name: True for name in PHYSICAL_EVAPORATION_GATES},
        **{name: True for name in PAGE_CURVE_GATES},
        **{name: True for name in QNM_RADIATIVE_GATES},
        "uses_target_qnm_frequency": True,
    }
    artifact.write_text(json.dumps({"readiness_gates": gates}), encoding="utf-8")

    report = black_hole_bridge_status_report(BlackHoleBridgeInputs(source_artifact=artifact))

    assert report["FINITE_HORIZON_RECORD_REPAIR_DIAGNOSTIC_RECEIPT"] is False
    assert report["BLACK_HOLE_PHYSICAL_EVAPORATION_BRIDGE_RECEIPT"] is False
    assert report["BLACK_HOLE_QNM_RADIATIVE_BRIDGE_RECEIPT"] is False
    assert "target_leakage_or_forbidden_dependency:uses_target_qnm_frequency" in report["blockers"]


def test_measurement_pack_copies_black_hole_bridge_status_report(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "pack"
    write_black_hole_bridge_status_report(run)

    pack = export_measurement_pack([run], out)

    assert (out / "black_hole_bridge_status_report.json").exists()
    assert (out / "black_hole_bridge_status_report.md").exists()
    assert pack["claims"]["black_hole_bridge_status_written"] is True
    assert pack["claims"]["black_hole_finite_horizon_record_receipt"] is False
    assert pack["claims"]["black_hole_physical_simulation_receipt"] is False
