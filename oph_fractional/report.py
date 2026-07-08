from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .compare import demo_fractional_report
from .receipts import FAIL_CLOSED_STATES, PASS_RECEIPTS


SANDBOX_CLAIM = "FRACTIONAL_QUOTIENT_SANDBOX_DIAGNOSTIC"
FIRST_BLOCKED_GATE = "MATERIAL_SPECIFIC_HAMILTONIAN_PROOF_RECEIPT"


def fractional_quotient_report() -> dict[str, Any]:
    """Assemble the Q5 fractional quotient-sector sandbox report."""

    simulation = demo_fractional_report()
    readiness_gates = _readiness_gates(simulation)
    claim_gates = {
        "SIMULATOR_QUOTIENT_CORRECTNESS_RECEIPT": readiness_gates["SIMULATOR_QUOTIENT_CORRECTNESS"],
        "OPTICAL_LINE_FAN_RECEIPT": readiness_gates["LINE_FAN_DECOMPOSITION"],
        "OPTICAL_SECTOR_IDENTIFIABILITY_RECEIPT": readiness_gates["OPTICAL_LINE_FAN_INJECTIVE"],
        "NO_TARGET_LEAK_DAG": readiness_gates["NO_TARGET_LEAK_DAG"],
        "MATERIAL_SPECIFIC_HAMILTONIAN_PROOF_RECEIPT": False,
        "EXPERIMENT_SPECIFIC_SOURCE_LAW_RECEIPT": False,
        "FROZEN_SAMPLE_COMPARISON_RECEIPT": False,
    }
    blockers = [name for name, value in claim_gates.items() if not value]
    return {
        "schema": "oph_fractional_quotient_report_v1",
        "mode": "fractional_quotient_sector_sandbox",
        "problem": "fractionalization_sandbox",
        "claim": SANDBOX_CLAIM,
        "strongest_allowed_claim": SANDBOX_CLAIM,
        "first_blocked_gate": FIRST_BLOCKED_GATE,
        "promotion_allowed": False,
        "material_claim": False,
        "fail_closed_state": "DIAGNOSTIC_ONLY",
        "readiness_gates": readiness_gates,
        "claim_gates": claim_gates,
        "blockers": blockers,
        "failure_states": list(FAIL_CLOSED_STATES),
        "simulation": simulation,
        "source_documents": [
            "reverse-engineering-reality/physics-problems/fractional_quantum_hall.md",
            "reverse-engineering-reality/physics-problems/fractional_excitons_as_oph_quotient_sector_readouts.md",
        ],
        "claim_boundary": (
            "The report verifies quotient-sector, Hamiltonian-promotion, optical line-fan, "
            "identifiability, refinement, and no-target-leak mechanics on a deterministic "
            "sandbox. It is not a proof that a specific twisted TMD or FQAH sample realizes OPH."
        ),
    }


def write_fractional_quotient_bundle(out_dir: Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = fractional_quotient_report()
    (out / "fractional_quotient_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (out / "fractional_quotient_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _readiness_gates(simulation: dict[str, Any]) -> dict[str, bool]:
    receipts = {name: False for name in PASS_RECEIPTS}
    receipts.update(_collect_boolean_receipts(simulation))
    receipts["MATERIAL_QUOTIENT_NORMAL_FORM_RECEIPT"] = bool(
        simulation.get("material_presentation", {}).get("material_quotient_normal_form_receipt", False)
    )
    receipts["SOURCE_LAW_FROZEN"] = receipts["SOURCE_HAMILTONIAN_FROZEN"]
    receipts["NO_TARGET_LEAK_DAG"] = receipts["NO_TARGET_LEAK"]
    receipts["K_MATRIX_READOUT"] = bool(simulation.get("abelian_readout", {}).get("K_MATRIX_READOUT", False))
    receipts["OPTICAL_MODULE_CERTIFICATE"] = bool(
        receipts["OPTICAL_OPERATOR_CERTIFIED"] and receipts["LINE_FAN_DECOMPOSITION"]
    )
    receipts["SIMULATOR_QUOTIENT_CORRECTNESS"] = bool(
        receipts["MATERIAL_QUOTIENT_NORMAL_FORM_RECEIPT"]
        and receipts["CANONICALIZER_IDEMPOTENCE"]
        and receipts["REPRESENTATIVE_INVARIANCE"]
        and receipts["QUOTIENT_LUMPABILITY"]
        and receipts["NO_ORBIT_SIZE_BIAS"]
    )
    return dict(sorted(receipts.items()))


def _collect_boolean_receipts(value: Any) -> dict[str, bool]:
    receipts: dict[str, bool] = {}
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str) and key.upper() == key and isinstance(nested, bool):
                receipts[key] = bool(nested)
            if key == "receipts" and isinstance(nested, dict):
                for receipt_name, receipt_value in nested.items():
                    if isinstance(receipt_name, str):
                        receipts[receipt_name] = bool(receipt_value)
            receipts.update(_collect_boolean_receipts(nested))
    elif isinstance(value, list):
        for item in value:
            receipts.update(_collect_boolean_receipts(item))
    return receipts


def _markdown_report(report: dict[str, Any]) -> str:
    gates = report["readiness_gates"]
    lines = [
        "# Fractional Quotient-Sector Sandbox Report",
        "",
        f"- Claim: {report['claim']}",
        f"- Material claim: {report['material_claim']}",
        f"- Promotion allowed: {report['promotion_allowed']}",
        f"- First blocked gate: {report['first_blocked_gate']}",
        f"- Fail-closed state: {report['fail_closed_state']}",
        "",
        "## Core receipts",
        "",
    ]
    for name in (
        "SIMULATOR_QUOTIENT_CORRECTNESS",
        "SOURCE_LAW_FROZEN",
        "NO_TARGET_LEAK_DAG",
        "CHERN_NUMBER",
        "MANYBODY_GAP",
        "K_MATRIX_READOUT",
        "OPTICAL_MODULE_CERTIFICATE",
        "LINE_FAN_DECOMPOSITION",
        "OPTICAL_LINE_FAN_INJECTIVE",
        "REFINEMENT_COMPATIBILITY",
    ):
        lines.append(f"- {name}: {gates.get(name)}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)
