from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oph_fpe.compact_transients.accuracy import simulator_accuracy_receipt
from oph_fpe.compact_transients.bh_recycling import bh_recycling_control_family
from oph_fpe.compact_transients.frb import repair_reload_control_family
from oph_fpe.compact_transients.receipts import (
    FAIL_CLOSED_RULES,
    GATE_RECEIPTS,
    NONCLAIMS,
    conditional_cr2_receipts,
    default_receipt_payloads,
    promotion_audit,
)


def compact_transient_audit_report(receipts: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = conditional_cr2_receipts()
    if receipts:
        merged.update({str(key): bool(value) for key, value in receipts.items()})
    promotion = promotion_audit(merged)
    error_ledger = {
        "epsilon_mu": None,
        "epsilon_K": None,
        "expected_path_length": None,
        "epsilon_E": None,
        "epsilon_prop": None,
        "epsilon_detector": None,
        "epsilon_canon": None,
        "epsilon_clock": None,
        "epsilon_mc": None,
        "tv_bound": None,
    }
    accuracy_receipt = simulator_accuracy_receipt(error_ledger)
    receipt_payloads = default_receipt_payloads(merged)
    receipt_payloads["SIMULATOR_ACCURACY_RECEIPT"] = accuracy_receipt
    return {
        "schema": "oph_compact_transient_audit_v1",
        "mode": "compact_record_transient_conditional_receipt_ladder",
        "problem": "compact_record_transients",
        "claim": promotion["allowed_claim_label"],
        "strongest_allowed_claim": promotion["allowed_claim_label"],
        "first_blocked_gate": promotion["first_blocked_gate"],
        "promotion_allowed": promotion["CR_READY"],
        "physical_claim": promotion["CR_READY"],
        "cr_ready": promotion["CR_READY"],
        "readiness_gates": dict(sorted(merged.items())),
        "promotion_audit": promotion,
        "error_ledger": error_ledger,
        "receipt_payloads": receipt_payloads,
        "required_gate_receipts": GATE_RECEIPTS,
        "fail_closed_rules": list(FAIL_CLOSED_RULES),
        "nonclaims": list(NONCLAIMS),
        "source_documents": [
            "reverse-engineering-reality/physics-problems/compact_record_transients.md",
        ],
        "implementation_targets": {
            "frb": [
                "young_only_control",
                "young_plus_old_gc_poisson_or_weibull_control",
                "young_plus_old_gc_repair_reload_model",
            ],
            "frb_control_family": repair_reload_control_family(),
            "black_hole_recycling": [
                "genealogy_dag",
                "generation_prior_without_ringdown_residual",
                "frozen_repair_tail_template",
                "stacked_ringdown_likelihood",
            ],
            "black_hole_control_family": bh_recycling_control_family(),
        },
        "claim_boundary": (
            "Conditional compact-transient receipt ladder. CR2 permits phenomenological compact "
            "source, repair-emission, detector, censoring, and likelihood studies. CR3 requires "
            "frozen controls, refinement, hashes, and held-out likelihood. CR4 remains blocked "
            "until compact source action, emission microphysics, physical clock, old-host FRB "
            "source law, and black-hole genealogy prior are OPH-derived without target leakage."
        ),
    }


def write_compact_transient_audit_report(out: Path, receipts: dict[str, Any] | None = None) -> dict[str, Any]:
    destination = Path(out)
    if destination.suffix.lower() != ".json":
        destination = destination / "compact_transient_audit_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = compact_transient_audit_report(receipts)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    destination.with_suffix(".md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    gates = report["readiness_gates"]
    lines = [
        "# Compact Transient Audit",
        "",
        f"- Claim: {report['claim']}",
        f"- Promotion allowed: {report['promotion_allowed']}",
        f"- First blocked gate: {report['first_blocked_gate']}",
        "",
        "## Core receipts",
        "",
    ]
    for name in (
        "COMPACT_QUOTIENT_RECEIPT",
        "COMPACT_SOURCE_LAW_RECEIPT",
        "PACKETIZED_KERNEL_RECEIPT",
        "PHYSICAL_CLOCK_RECEIPT",
        "FINITE_PACKET_PARENT_RECEIPT",
        "DETECTION_THINNING_RECEIPT",
        "CENSORING_AND_UPPER_LIMIT_RECEIPT",
        "POINT_PROCESS_LIKELIHOOD_RECEIPT",
        "CONTROL_MODEL_RECEIPT",
        "REFINEMENT_STABILITY_RECEIPT",
        "FROZEN_HASHES_RECEIPT",
        "HELDOUT_LIKELIHOOD_RECEIPT",
    ):
        lines.append(f"- {name}: {gates.get(name)}")
    lines.extend(["", "## Boundary", "", report["claim_boundary"], ""])
    return "\n".join(lines)
