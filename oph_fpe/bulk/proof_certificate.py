from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oph_fpe.claims import (
    BRANCH_INSTANTIATION_SANITY,
    BW_KMS_BRANCH_INSTANTIATION_RECEIPT,
    CHART_LORENTZ_H3_RECEIPT,
    H3_RESPONSE_CANDIDATE_RECEIPT,
    OBJECT_BULK_POPULATION_RECEIPT,
    PHYSICAL_CMB_RECEIPT,
    PROTO_PARTICLE_RECEIPT,
    RECORD_COMMIT_RECEIPT,
    REPAIR_CORE_RECEIPT,
    SCREEN_PROXY_CMB_RECEIPT,
    with_claim_metadata,
)


def bulk_proof_certificate(run_dir: Path) -> dict[str, Any]:
    """Build a tiered OPH 3D-bulk proof certificate from run receipts.

    This report intentionally separates the paper-side Lorentz/H3 chart result
    from neutral third-person reconstruction, particle emergence, and physical
    CMB prediction. The OPH papers derive the chart route from support-visible
    BW cap modular flow; finite simulation receipts then test whether observer
    records and defects populate that chart under controls.
    """

    root = Path(run_dir)
    emergence = _read_json(root / "emergence_status_report.json")
    ladder = _read_json(root / "receipt_ladder_report.json")
    neutral = _read_json(root / "bulk_reconstruction_report.json")
    cmb_lite = _read_json(root / "cmb_lite_comparison_report.json")
    cl = _read_json(root / "cl_comparison_report.json")
    particle = _read_json(root / "particle_likeness_report.json")
    controlled_particle = _read_json(root / "controlled_defect_particle_assay_report.json")

    repair_core = _ladder_passed(ladder, "R0") or _truthy(emergence, "final_phi_zero")
    record_commit = _ladder_passed(ladder, "R1") or _truthy(emergence, "records_committed")
    bw_kms = _truthy_any(
        emergence,
        "BW_KMS_DIRECT_2PI_RECEIPT",
        "state_derived_correct_beats_controls",
        "state_derived_selected_2pi",
    ) or _ladder_passed(ladder, "R2")
    chart = _truthy_any(
        emergence,
        "PAPER_THEOREM_3D_BULK_CHART_RECEIPT",
        "CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT",
        "CHART_LORENTZ_H3_RECEIPT",
    ) or _ladder_passed(ladder, "R3")
    h3_response = _truthy_any(
        emergence,
        "H3_RESPONSE_CANDIDATE_RECEIPT",
        "H3_RESPONSE_CONTROL_SEPARATION_RECEIPT",
        "modular_response_h3_candidate_receipt",
    ) or _ladder_passed(ladder, "R4")
    h3_object_preview = _truthy_any(
        emergence,
        "THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT",
        "PAPER_THEOREM_ASSISTED_H3_POPULATED_CHART_RECEIPT",
        "observer_chart_object_h3_receipt",
    ) or _ladder_passed(ladder, "R5")
    object_nonboundary_population = _truthy_any(
        emergence,
        "OBJECT_H3_NONBOUNDARY_POPULATION_RECEIPT",
        "OBJECT_BULK_POPULATION_RECEIPT",
        "observer_chart_bulk_population_receipt",
    )
    theorem_assisted_chart_preview = bool(
        chart and bw_kms and h3_response and (h3_object_preview or object_nonboundary_population)
    )
    theorem_assisted_nonboundary_population = bool(
        chart and bw_kms and h3_response and object_nonboundary_population
    )
    strict_neutral_bulk = bool(
        neutral.get("bulk_3d_established", False)
        or emergence.get("strict_blind_observer_bulk_receipt", False)
        or emergence.get("neutral_bulk_3d_established", False)
    )
    screen_cmb = bool(
        emergence.get("SCREEN_PROXY_CMB_RECEIPT", False)
        or _ladder_passed(ladder, "R7")
        or cl
        or cmb_lite
    )
    physical_cmb = bool(
        emergence.get("physical_cmb_prediction", False)
        or cmb_lite.get("physical_cmb_prediction", False)
        or cl.get("physical_cmb_prediction", False)
    )
    production_particle = bool(
        emergence.get("particle_matter_receipt", False)
        or particle.get("particle_matter_receipt", False)
        or controlled_particle.get("physical_particle_emergence", False)
    )

    tiers = {
        "T0_finite_repair_core": _tier(
            REPAIR_CORE_RECEIPT,
            repair_core,
            "Finite overlap repair settled the declared mismatch/normal-form surface.",
        ),
        "T1_record_commit": _tier(
            RECORD_COMMIT_RECEIPT,
            record_commit,
            "Observer-facing records committed and can be read as finite record algebra data.",
        ),
        "T2_bw_kms_2pi_branch": _tier(
            BW_KMS_BRANCH_INSTANTIATION_RECEIPT,
            bw_kms,
            "Finite support-visible cap/collar branch instantiates BW/KMS 2*pi cap transport.",
        ),
        "T3_chart_lorentz_h3": _tier(
            CHART_LORENTZ_H3_RECEIPT,
            chart,
            "Paper-side conformal route Conf+(S2) -> SO+(3,1) -> H3 spatial chart is instantiated.",
        ),
        "T4_h3_response_controls": _tier(
            H3_RESPONSE_CANDIDATE_RECEIPT,
            h3_response,
            "Observer/cap response signal populates the H3 chart better than implemented controls.",
        ),
        "T5a_theorem_assisted_h3_object_preview": _tier(
            "THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT",
            theorem_assisted_chart_preview,
            "Persistent observer-facing objects can be displayed in the theorem-side H3 chart under controls.",
        ),
        "T5b_nonboundary_h3_object_population": _tier(
            OBJECT_BULK_POPULATION_RECEIPT,
            theorem_assisted_nonboundary_population,
            "Persistent observer-facing objects populate the theorem-side H3 chart and are not boundary-dominated.",
        ),
        "T6_strict_neutral_third_person_bulk": _tier(
            "STRICT_NEUTRAL_3D_BULK_RECEIPT",
            strict_neutral_bulk,
            "Neutral observer-record reconstruction establishes a third-person 3D bulk without chart prior.",
        ),
        "T7_production_particles": _tier(
            PROTO_PARTICLE_RECEIPT,
            production_particle,
            "Production defects satisfy localization, transport, fusion/scattering, and bulk-worldline gates.",
        ),
        "T8_screen_cmb_proxy": _tier(
            SCREEN_PROXY_CMB_RECEIPT,
            screen_cmb,
            "Freezeout/screen angular spectra are available for shape-level measurement comparison.",
        ),
        "T9_physical_cmb_prediction": _tier(
            PHYSICAL_CMB_RECEIPT,
            physical_cmb,
            "A physical CMB prediction is emitted through Boltzmann/likelihood-ready finite OPH kernels.",
        ),
    }

    report = {
        "mode": "oph_3d_bulk_and_measurement_proof_certificate_v0",
        "run_path": str(root),
        "proof_tiers": tiers,
        "chart_level_3p1_lorentz_kinematics_established": bool(chart and bw_kms),
        "theorem_assisted_h3_object_preview_established": theorem_assisted_chart_preview,
        "theorem_assisted_h3_nonboundary_population_established": theorem_assisted_nonboundary_population,
        "theorem_assisted_h3_populated_chart_established": theorem_assisted_nonboundary_population,
        "strict_neutral_third_person_bulk_established": strict_neutral_bulk,
        "bulk_3d_established_theorem_assisted": theorem_assisted_nonboundary_population,
        "bulk_3d_established_strict": strict_neutral_bulk,
        "screen_cmb_proxy_available": screen_cmb,
        "physical_cmb_prediction": physical_cmb,
        "production_particle_matter_receipt": production_particle,
        "paper_alignment": {
            "screen_role": "S2 is the observer-facing cap/symmetry chart, not a raw point-cloud proof of dimension.",
            "lorentz_route": "support-visible BW/KMS cap flow with s=2*pi*t gives Conf+(S2) ~= SO+(3,1).",
            "spatial_chart": "H3 is the spatial homogeneous chart SO+(3,1)/SO(3).",
            "finite_gate": "finite runs must separately show observer records/objects/defects populate that chart under controls.",
        },
        "claim_boundary": (
            "Tiered OPH proof/readout certificate. T2-T3 establish the chart-level 3+1D Lorentz/H3 "
            "branch for this run. T5a is theorem-assisted H3 preview evidence from observer objects. "
            "T5b is stricter non-boundary H3 object population evidence. "
            "T6 is stricter neutral third-person bulk reconstruction and may remain false even when T5 passes. "
            "T8 is measurement-facing screen C_l data only. T9 is a physical CMB prediction and remains false "
            "until finite OPH kernels feed a Boltzmann/likelihood-ready pipeline."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=BRANCH_INSTANTIATION_SANITY,
        receipt="OPH_3D_BULK_PROOF_CERTIFICATE",
        physical_claim=False,
        observable_id="tiered_3d_bulk_and_measurement_receipts",
        fit_objective="receipt_gate_summary",
    )


def write_bulk_proof_certificate(run_dir: Path, out: Path | None = None) -> dict[str, Any]:
    report = bulk_proof_certificate(run_dir)
    out_path = Path(out) if out is not None else Path(run_dir) / "bulk_proof_certificate_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return report


def _tier(receipt_name: str, passed: bool, description: str) -> dict[str, Any]:
    return {"receipt_name": receipt_name, "passed": bool(passed), "description": description}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _ladder_passed(ladder: dict[str, Any], key: str) -> bool:
    return bool(((ladder.get("receipts", {}) or {}).get(key, {}) or {}).get("passed", False))


def _truthy(data: dict[str, Any], key: str) -> bool:
    return bool(data.get(key, False))


def _truthy_any(data: dict[str, Any], *keys: str) -> bool:
    return any(_truthy(data, key) for key in keys)


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# OPH 3D Bulk And Measurement Proof Certificate",
        "",
        f"- run: `{report['run_path']}`",
        f"- chart-level 3+1D Lorentz/H3: `{str(report['chart_level_3p1_lorentz_kinematics_established']).lower()}`",
        f"- theorem-assisted H3 object preview: `{str(report['theorem_assisted_h3_object_preview_established']).lower()}`",
        f"- theorem-assisted non-boundary H3 population: `{str(report['theorem_assisted_h3_nonboundary_population_established']).lower()}`",
        f"- strict neutral third-person bulk: `{str(report['strict_neutral_third_person_bulk_established']).lower()}`",
        f"- screen CMB proxy available: `{str(report['screen_cmb_proxy_available']).lower()}`",
        f"- physical CMB prediction: `{str(report['physical_cmb_prediction']).lower()}`",
        f"- production particle matter receipt: `{str(report['production_particle_matter_receipt']).lower()}`",
        "",
        "## Tiers",
        "",
    ]
    for name, tier in report["proof_tiers"].items():
        lines.append(f"- `{name}`: `{str(tier['passed']).lower()}` - {tier['receipt_name']}")
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)
