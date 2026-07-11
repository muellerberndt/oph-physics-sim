from __future__ import annotations

import json

from oph_fpe.claims import (
    CHI_NU_G10_ENERGY_LEDGER_THEOREM_RECEIPT,
    CHI_NU_G9_RECORD_GRAVITY_BRIDGE_RECEIPT,
    FINITE_MODULAR_FLOW_IMPORT_RECEIPT,
    MATSCHEKO_PROOF_CHAIN_IMPORT_RECEIPT,
    P_BRANCH_PROVENANCE_RECEIPT,
    QBFT_BOUNDARY_CAVEAT_RECEIPT,
    RULE90_DIAGONAL_SCREEN_IMPORT_RECEIPT,
    RULE90_PARITY_SPLITTING_IMPORT_RECEIPT,
    RULE90_TWO_POWER_UNIVERSALITY_IMPORT_RECEIPT,
    SCALAR_CHANNEL_BRIDGE_IMPORT_RECEIPT,
    TWELVE_PORT_SURFACE_IMPORT_RECEIPT,
)
from oph_fpe.cli import main
from oph_fpe.consensus import (
    chi_nu_g10_energy_ledger_report,
    chi_nu_g9_bridge_gate_report,
    finite_audit_import_report,
    matscheko_proof_chain_import_report,
    p_branch_provenance_report,
)


def test_p_branch_provenance_keeps_source_and_codata_branches_separate() -> None:
    report = p_branch_provenance_report()

    assert report[P_BRANCH_PROVENANCE_RECEIPT] is True
    assert report["externalInputUsedByPpub"] == "CODATA_ALPHA_INV_2022"
    assert 3.8e-6 < report["Pgap"] < 4.0e-6
    assert 1.4e-7 < report["chiBranchGap"] < 1.6e-7


def test_g9_gate_is_fail_closed_until_record_gravity_calibration_exists() -> None:
    report = chi_nu_g9_bridge_gate_report()

    assert report[CHI_NU_G9_RECORD_GRAVITY_BRIDGE_RECEIPT] is False
    assert report["definitionSideClosed"] is True
    assert report["calibrationClosed"] is False
    assert report["blockers"] == [
        "missing numerical map from apparatus record contrast to gravitational scalar"
    ]


def test_g10_report_separates_cycle_theorem_from_named_convention() -> None:
    report = chi_nu_g10_energy_ledger_report()

    assert report[CHI_NU_G10_ENERGY_LEDGER_THEOREM_RECEIPT] is True
    assert 0.549 < report["benchCycleWorkPerMeterJ"] < 0.550
    assert 3.49e6 < report["namedConventionToggleJ"] < 3.52e6
    assert 5.03e15 < report["sourceCreationCeilingJ"] < 5.04e15
    assert report["conventionIsTheorem"] is False


def test_finite_audit_imports_report_all_non_rule_import_receipts() -> None:
    report = finite_audit_import_report()

    assert report["receipt"] is True
    assert report[FINITE_MODULAR_FLOW_IMPORT_RECEIPT] is True
    assert report[SCALAR_CHANNEL_BRIDGE_IMPORT_RECEIPT] is True
    assert report[TWELVE_PORT_SURFACE_IMPORT_RECEIPT] is True
    assert report[QBFT_BOUNDARY_CAVEAT_RECEIPT] is True
    assert report[RULE90_PARITY_SPLITTING_IMPORT_RECEIPT] is True
    assert report[RULE90_TWO_POWER_UNIVERSALITY_IMPORT_RECEIPT] is True
    assert report[RULE90_DIAGONAL_SCREEN_IMPORT_RECEIPT] is True
    assert report["blockers"] == []

    parity = report["imports"][RULE90_PARITY_SPLITTING_IMPORT_RECEIPT]
    two_power = report["imports"][RULE90_TWO_POWER_UNIVERSALITY_IMPORT_RECEIPT]
    diagonal = report["imports"][RULE90_DIAGONAL_SCREEN_IMPORT_RECEIPT]
    assert parity["status"] == "imported_status"
    assert "T38" in parity["scope"]
    assert "T39" in two_power["scope"]
    assert "T40/T41" in diagonal["scope"]
    assert "finite binary audit fixture only" in parity["physics_residue"]


def test_matscheko_proof_chain_import_report_is_status_receipt_not_physics_claim() -> None:
    report = matscheko_proof_chain_import_report()

    assert report[MATSCHEKO_PROOF_CHAIN_IMPORT_RECEIPT] is True
    assert report["receipt"] is True
    assert report["physical_claim"] is False
    assert report["claim_level"] == "demo"
    assert report["mode"] == "matscheko_proof_chain_import_v2"
    assert report["proofChainProvenance"]["commit"].startswith("0f9e43b")
    assert report["g9RecordGravityBridge"][CHI_NU_G9_RECORD_GRAVITY_BRIDGE_RECEIPT] is False
    assert "G9 record-DeltaS to gravity-DeltaS calibration is open" in report[
        "physicsPromotionBlockers"
    ]


def test_matscheko_proof_chain_import_cli_writes_report(tmp_path) -> None:
    out = tmp_path / "proof_chain_import.json"

    assert main(["matscheko-proof-chain-import", "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload[MATSCHEKO_PROOF_CHAIN_IMPORT_RECEIPT] is True
    assert payload["g9RecordGravityBridge"][CHI_NU_G9_RECORD_GRAVITY_BRIDGE_RECEIPT] is False
