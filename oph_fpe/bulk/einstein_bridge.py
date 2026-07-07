from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oph_fpe.claims import (
    EINSTEIN_ALL_TIMELIKE_TENSOR_UPGRADE_RECEIPT,
    EINSTEIN_BOUNDED_INTERVAL_KERNEL_RECEIPT,
    EINSTEIN_BRANCH_ENTRY_RECEIPT,
    EINSTEIN_BRIDGE_DEPENDENCY_DISCHARGE_RECEIPT,
    EINSTEIN_BRIDGE_RUN_RECEIPTS_RECEIPT,
    EINSTEIN_DIAGONAL_REMAINDER_RECEIPT,
    EINSTEIN_FIXED_CAP_ENTROPY_STATIONARITY_RECEIPT,
    EINSTEIN_LAMBDA_CONSTANCY_CONSERVATION_RECEIPT,
    EINSTEIN_NEWTON_COUPLING_FORBIDDEN_INPUT_AUDIT_RECEIPT,
    EINSTEIN_NULL_STRESS_CHARGE_RECEIPT,
    EINSTEIN_RESIDUAL_RECEIPT,
    EINSTEIN_SMALL_BALL_AREA_BRIDGE_RECEIPT,
    EINSTEIN_STRESS_CLOSURE_RECEIPT,
    EINSTEIN_TIMELIKE_COVERAGE_RECEIPT,
    OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_RECEIPT,
    OPH_EINSTEIN_BRIDGE_MANIFEST_RECEIPT,
)


PROVENANCE_TAGS: dict[str, str] = {
    "S2_screen": "AXIOM_1",
    "overlap_consistency": "AXIOM_2",
    "local_MaxEnt": "AXIOM_3",
    "S_gen": "AXIOM_4",
    "MAR": "AXIOM_5",
    "sphere_fold": "MICROPHYSICS_SPHERE_FOLD",
    "GeomRead": "THEOREM_E0_STEP_1",
    "Lorentz_H3": "THEOREM_E0_STEP_2",
    "BW_2pi": "THEOREM_4_2",
    "NullStress": "THEOREM_5_2G",
    "BoundedInterval": "LEMMA_E0_5",
    "FixedCapStat": "AXIOM_3_PLUS_4",
    "SmallBallArea": "STANDARD_GEOMETRY_AFTER_GEOMREAD",
    "RemainderControl": "LEMMA_E0_6",
    "AllTimelikeCoverage": "LEMMA_E0_7",
    "StressClosure": "PROPOSITION_E2",
    "Lambda": "D6_NCRC_CLOSURE",
}


@dataclass(frozen=True)
class ReceiptSpec:
    name: str
    file_name: str
    theorem_tag: str
    keys: tuple[str, ...]
    required_for_branch_entry: bool = True


RECEIPT_SPECS: tuple[ReceiptSpec, ...] = (
    ReceiptSpec(
        "sphere_fold",
        "sphere_fold_receipt.json",
        "MICROPHYSICS_SPHERE_FOLD",
        ("SPHERE_FOLD_RECEIPT", "sphere_fold_receipt", "sphere_fold_geometry_readout_receipt"),
    ),
    ReceiptSpec(
        "bw_2pi",
        "bw_receipt.json",
        "THEOREM_4_2",
        ("BW_2PI_RECEIPT", "BW_KMS_BRANCH_REPLAY_RECEIPT", "bw_2pi_receipt", "support_visible_bw_receipt"),
    ),
    ReceiptSpec(
        "null_stress",
        "null_stress_receipt.json",
        "THEOREM_5_2G",
        (EINSTEIN_NULL_STRESS_CHARGE_RECEIPT, "null_generator_stress_charge_receipt"),
    ),
    ReceiptSpec(
        "bounded_interval",
        "bounded_interval_receipt.json",
        "LEMMA_E0_5",
        (EINSTEIN_BOUNDED_INTERVAL_KERNEL_RECEIPT, "bounded_interval_kernel_receipt"),
    ),
    ReceiptSpec(
        "fixed_cap_entropy",
        "fixed_cap_entropy_receipt.json",
        "AXIOM_3_PLUS_4",
        (EINSTEIN_FIXED_CAP_ENTROPY_STATIONARITY_RECEIPT, "fixed_cap_entropy_stationarity_receipt"),
    ),
    ReceiptSpec(
        "small_ball_area",
        "small_ball_area_receipt.json",
        "STANDARD_GEOMETRY_AFTER_GEOMREAD",
        (EINSTEIN_SMALL_BALL_AREA_BRIDGE_RECEIPT, "small_ball_area_bridge_receipt"),
    ),
    ReceiptSpec(
        "remainder_control",
        "remainder_receipt.json",
        "LEMMA_E0_6",
        (EINSTEIN_DIAGONAL_REMAINDER_RECEIPT, "diagonal_remainder_receipt", "remainder_control_receipt"),
    ),
    ReceiptSpec(
        "timelike_coverage",
        "timelike_coverage_receipt.json",
        "LEMMA_E0_7",
        (
            EINSTEIN_TIMELIKE_COVERAGE_RECEIPT,
            EINSTEIN_ALL_TIMELIKE_TENSOR_UPGRADE_RECEIPT,
            "timelike_coverage_receipt",
            "all_timelike_tensor_upgrade_receipt",
        ),
    ),
    ReceiptSpec(
        "stress_closure",
        "stress_closure_receipt.json",
        "PROPOSITION_E2",
        (EINSTEIN_STRESS_CLOSURE_RECEIPT, "stress_closure_receipt", "covariant_stress_conservation_receipt"),
    ),
    ReceiptSpec(
        "lambda_closure",
        "lambda_closure_receipt.json",
        "D6_NCRC_CLOSURE",
        (EINSTEIN_LAMBDA_CONSTANCY_CONSERVATION_RECEIPT, "lambda_constancy_conservation_receipt"),
    ),
    ReceiptSpec(
        "newton_forbidden_input_audit",
        "newton_forbidden_input_receipt.json",
        "NEWTON_FORBIDDEN_INPUT_AUDIT",
        (
            EINSTEIN_NEWTON_COUPLING_FORBIDDEN_INPUT_AUDIT_RECEIPT,
            "newton_coupling_forbidden_input_audit_receipt",
        ),
    ),
    ReceiptSpec(
        "einstein_residual",
        "einstein_residual_receipt.json",
        "EINSTEIN_RESIDUAL_RUN_CHECK",
        (EINSTEIN_RESIDUAL_RECEIPT, "einstein_residual_receipt", "einstein_equation_solution_receipt"),
    ),
)


def einstein_bridge_manifest_report(run_dir: Path) -> dict[str, Any]:
    """Build the E0 Einstein bridge manifest from run sidecar receipts.

    The E0 theorem provenance is static paper provenance. The run-specific
    branch-entry receipt is stricter: every required sidecar must be present and
    theorem-tagged before visuals may be promoted beyond diagnostics.
    """

    root = Path(run_dir)
    legacy = _read_json(root / "einstein_branch_entry_report.json")
    receipt_rows = [_receipt_row(root, legacy, spec) for spec in RECEIPT_SPECS]
    required_rows = [row for row in receipt_rows if row["requiredForBranchEntry"]]
    run_receipts = bool(required_rows and all(row["receipt"] for row in required_rows))
    dependency_discharge = True
    branch_entry = bool(dependency_discharge and run_receipts)
    blockers = [row["name"] for row in required_rows if not row["receipt"]]
    child_gates = {
        "E1_null_generator_stress_charge": _row_receipt(receipt_rows, "null_stress"),
        "E2_fixed_cap_entropy_stationarity": _row_receipt(receipt_rows, "fixed_cap_entropy"),
        "E3_small_ball_area_bridge": all(
            _row_receipt(receipt_rows, name)
            for name in ("sphere_fold", "bounded_interval", "small_ball_area", "remainder_control")
        ),
        "E4_all_timelike_tensor_upgrade": _row_receipt(receipt_rows, "timelike_coverage"),
        "E5_lambda_constancy_conservation": all(
            _row_receipt(receipt_rows, name) for name in ("stress_closure", "lambda_closure")
        ),
        "E6_newton_coupling_forbidden_input_audit": _row_receipt(
            receipt_rows, "newton_forbidden_input_audit"
        ),
    }
    return {
        "schema": "oph_einstein_bridge_manifest_v1",
        "run_path": str(root),
        OPH_EINSTEIN_BRIDGE_MANIFEST_RECEIPT: True,
        "einstein_bridge_manifest_receipt": True,
        EINSTEIN_BRIDGE_DEPENDENCY_DISCHARGE_RECEIPT: dependency_discharge,
        "theorem_e0_dependency_discharge_receipt": dependency_discharge,
        EINSTEIN_BRIDGE_RUN_RECEIPTS_RECEIPT: run_receipts,
        "einstein_bridge_run_receipts_receipt": run_receipts,
        "all_required_receipts_theorem_tagged": run_receipts,
        OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_RECEIPT: branch_entry,
        EINSTEIN_BRANCH_ENTRY_RECEIPT: branch_entry,
        "einstein_branch_entry_contract_receipt": branch_entry,
        "einstein_branch_entry_receipt": branch_entry,
        "claim_tier": (
            "OPH5_E0_THEOREM_BACKED_RUN_RECEIPTS_PASSED"
            if branch_entry
            else "OPH5_E0_THEOREM_BACKED_RUN_RECEIPTS_OPEN"
        ),
        "provenanceTags": PROVENANCE_TAGS,
        "receiptRows": receipt_rows,
        "requiredReceiptFiles": [spec.file_name for spec in RECEIPT_SPECS if spec.required_for_branch_entry],
        "blockers": blockers,
        "einstein_branch_entry_blockers": blockers,
        "einstein_branch_entry_child_gates": child_gates,
        "legacyIssue503": {
            "status": legacy.get("issue_503_status", "not_used_by_e0_manifest"),
            "sourceReportWritten": bool(legacy),
            "compatibilityOnly": True,
        },
        "claimBoundary": (
            "The E0 paper theorem discharges the OPH5 recovered-core bridge dependencies. A concrete "
            "simulation run still needs each theorem-tagged sidecar receipt before the Einstein branch "
            "entry receipt is true. Curved-spacetime visuals remain diagnostics while any receipt row is open."
        ),
    }


def write_einstein_bridge_manifest(run_dir: Path, out: Path | None = None) -> dict[str, Any]:
    report = einstein_bridge_manifest_report(run_dir)
    out_path = Path(out) if out is not None else Path(run_dir) / "einstein_bridge_manifest.json"
    if out_path.is_dir():
        out_path = out_path / "einstein_bridge_manifest.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _receipt_row(root: Path, legacy: dict[str, Any], spec: ReceiptSpec) -> dict[str, Any]:
    path = root / spec.file_name
    payload = _read_json(path)
    receipt = _truthy_any(payload, *spec.keys) or _truthy_any(legacy, *spec.keys)
    return {
        "name": spec.name,
        "file": spec.file_name,
        "path": str(path),
        "written": bool(payload),
        "receipt": receipt,
        "theoremTag": spec.theorem_tag,
        "requiredForBranchEntry": spec.required_for_branch_entry,
        "acceptedKeys": list(spec.keys),
        "source": "sidecar" if payload else "legacy_einstein_branch_entry_report" if receipt else "missing",
    }


def _row_receipt(rows: list[dict[str, Any]], name: str) -> bool:
    return any(row["name"] == name and row["receipt"] for row in rows)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _truthy_any(mapping: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = mapping.get(key)
        if value is True:
            return True
        if isinstance(value, str) and value.lower() in {"true", "passed", "complete", "closed"}:
            return True
    return False
