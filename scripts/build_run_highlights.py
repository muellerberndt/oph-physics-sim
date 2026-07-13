#!/usr/bin/env python3
"""Per-run highlights: milestone status and near-fit scoreboard.

Writes RUN_HIGHLIGHTS.md and run_highlights.json into a run directory.
The milestone rows are read from THIS run's receipts; the public-data
near-fit rows are curated excerpts (source: the experiment tracker
section 0a and docs/BEST_OF_PUBLIC_DATA_COMPARISONS.md) and are
run-independent paper-side lanes unless marked run-derived. Claim
discipline: near-fits are conditional or diagnostic comparisons; the
suite holds zero frozen physical-prediction receipts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUN = Path(sys.argv[1]).resolve()


def read_json(name: str) -> dict:
    path = RUN / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


manifest = read_json("manifest.json")
experience = read_json("observer_modular_experience_report.json")
selection = read_json("transition_scale_selection_report.json")
agreement = read_json("observer_agreement_report.json")
bulk_field = read_json("agreement_bulk_field_summary.json")
organic = read_json("organic_defect_population_report.json")
interaction = read_json("defect_interaction_report.json")
freezeout = read_json("freezeout_map_summary.json")
cmb_lite = read_json("cmb_lite_comparison_report.json")
gate = read_json("cosmology_gate_report.json")
einstein = read_json("einstein_bridge_manifest.json")
neutral_audit = read_json("neutral_3d_bulk_audit_report.json")

patch_count = manifest.get("patch_count") or freezeout.get("point_count")
observer_count = experience.get("observer_count")
gates = experience.get("component_gates") or {}

best_shape = None
if isinstance(cmb_lite.get("best_shape_field"), str):
    fields = cmb_lite.get("fields") or {}
    entry = fields.get(cmb_lite["best_shape_field"]) if isinstance(fields, dict) else None
    if isinstance(entry, dict):
        best_shape = entry.get("shape_correlation")

milestones = [
    {
        "milestone": "Observer-local modular time",
        "receipt": bool(gates.get("observer_modular_time_receipt")),
        "detail": f"{observer_count} materialized observers",
    },
    {
        "milestone": "2 pi KMS clock (gauge-covariant, sector replay)",
        "receipt": bool(gates.get("bw_kms_branch_replay_receipt")),
        "detail": (
            f"two_pi_selected={selection.get('two_pi_selected')}, "
            f"degenerate={selection.get('response_degenerate')}"
        ),
    },
    {
        "milestone": "Conformal H3 spatial chart",
        "receipt": bool(gates.get("conformal_h3_chart_receipt")),
        "detail": "observer-facing chart, support-visible",
    },
    {
        "milestone": "H3 modular response",
        "receipt": bool(gates.get("h3_modular_response_receipt")),
        "detail": "perturb/resettle candidate with controls",
    },
    {
        "milestone": "Observer-facing 3+1D experience (all four gates)",
        "receipt": bool(experience.get("observer_facing_3p1d_h3_experience_receipt")),
        "detail": "integer 3 spatial + 1 modular time; never a fractional bulk",
    },
    {
        "milestone": "Observer mutual agreement (bulk-as-agreement)",
        "receipt": bool(agreement.get("MUTUAL_GAUGE_CHART_AGREEMENT_RECEIPT")),
        "detail": (
            f"pairs {agreement.get('population', {}).get('evaluated_pairs')}, "
            f"median defect {agreement.get('pair_agreement', {}).get('median_defect')}, "
            f"cocycle perfect {agreement.get('cocycle', {}).get('perfect_fraction')}, "
            f"control {round(agreement.get('control', {}).get('median_defect_shuffled') or 0, 3)}"
        ),
    },
    {
        "milestone": "Emergent-bulk agreement field",
        "receipt": bulk_field.get("status") == "evaluated",
        "detail": (
            f"coverage {round((bulk_field.get('covered_patch_fraction') or 0) * 100, 1)} pct, "
            f"pair-certified {round((bulk_field.get('pair_certified_patch_fraction') or 0) * 100, 1)} pct, "
            f"max multiplicity {bulk_field.get('max_pair_multiplicity')}"
        ),
    },
    {
        "milestone": "Proto-particle population (organic)",
        "receipt": bool(organic.get("organic_defect_population_receipt")),
        "detail": (
            f"defects {organic.get('defect_count')}, "
            f"fusion candidates {interaction.get('fusion_candidate_count')}, "
            f"identity fraction {interaction.get('fusion_identity_fraction')}"
        ),
    },
    {
        "milestone": "Strict neutral third-person bulk",
        "receipt": bool(neutral_audit.get("strict_neutral_bulk_ready", False)),
        "detail": "glued objective bulk; blockers in neutral_3d_bulk_audit_report.md",
    },
    {
        "milestone": "Einstein branch entry (E1-E6)",
        "receipt": bool(einstein.get("einstein_branch_entry_receipt", False)),
        "detail": f"blockers: {einstein.get('einstein_branch_entry_blockers')}",
    },
    {
        "milestone": "Physical particle / physical CMB",
        "receipt": False,
        "detail": "fail-closed by contract; screen proxies stay diagnostics",
    },
]

run_derived = [
    {
        "quantity": "Screen C_l proxy vs Planck TT shape (cmb-lite)",
        "value": best_shape,
        "note": "shape correlation of the best screen field; a proxy diagnostic, never a CMB prediction",
    },
    {
        "quantity": "Committed fraction at freezeout",
        "value": freezeout.get("committed_fraction"),
        "note": f"freezeout cycle {freezeout.get('freezeout_cycle')}",
    },
    {
        "quantity": "Cosmology product gate",
        "value": gate.get("allowed"),
        "note": f"kms_bw_pass={((gate.get('checks') or {}).get('kms_bw_pass'))}",
    },
]

near_fits = [
    {"quantity": "Scalar tilt n_s = 1 - P/48", "oph": 0.9660215, "measured": "0.9649(42) (Planck 2018)", "pull": "+0.27 sigma"},
    {"quantity": "Higgs-top criticality relation m_H(m_t)", "oph": 125.72, "measured": "125.13(11) GeV (PDG)", "pull": "+0.47 percent"},
    {"quantity": "Electroweak chart pair M_W / M_Z", "oph": "80.330 / 91.119 GeV", "measured": "80.3692(133) / 91.1880(20)", "pull": "-0.05 / -0.08 percent (chart, pole packet open)"},
    {"quantity": "Strong coupling alpha_s(M_Z)", "oph": 0.11834, "measured": "0.1179(9)", "pull": "+0.5 sigma"},
    {"quantity": "QCD scale Lambda_QCD(3)", "oph": "334.8 [319, 350] MeV", "measured": "338(12) MeV (FLAG-class)", "pull": "inside interval"},
    {"quantity": "Capacity -> Lambda (EW branch)", "oph": "Lambda l_P^2 = 2.668e-122", "measured": "2.845e-122 (Planck display)", "pull": "+6.6 percent over 122 orders of magnitude"},
    {"quantity": "CMB low-l IR filter scale", "oph": "corpus l_IR = 32", "measured": "unbinned PR3 TT optimum l_IR ~ 26 (flat 24-40), q ~ 0.16", "pull": "corpus scale 0.2 chi2 units from the data optimum (net -3.9)"},
    {"quantity": "Cosmic birefringence candidate alpha_U/(2 pi)", "oph": "0.37501 deg", "measured": "0.342 +0.094/-0.091 deg (Planck+WMAP EB)", "pull": "+0.35 sigma (S1 coincidence, 4 trials declared)"},
    {"quantity": "Clock-free ratio m_t/m_W", "oph": 2.1490, "measured": 2.1476, "pull": "+0.07 percent"},
    {"quantity": "SPARC RAR fixed-branch RMS", "oph": "0.13283 dex (Z6)", "measured": "0.13281 dex same-data optimum", "pull": "2e-5 dex from optimal (retrospective)"},
]

banked = [
    "Cassini universal continuation excluded (19.2 sigma raw): the settled-galaxy scope survives, the universal extension died",
    "Weighted-cycle neutrino candidate rejected by NuFIT 6.1",
    "Screen C_l proxy anti-correlates with Planck TT: the raw screen is not the CMB",
]

label = f"{RUN.name} ({patch_count:,} patches / {observer_count:,} observers)" if patch_count and observer_count else RUN.name
payload = {
    "schema": "run_highlights_v1",
    "run_id": RUN.name,
    "label": label,
    "patch_count": patch_count,
    "observer_count": observer_count,
    "milestones": milestones,
    "run_derived_diagnostics": run_derived,
    "near_fits_public_data": near_fits,
    "banked_negatives": banked,
    "claim_boundary": (
        "Milestone receipts are this run's own gate values. Near-fit rows are "
        "curated conditional or diagnostic comparisons from the paper-side "
        "lanes (tracker section 0a); none is a frozen physical-prediction "
        "receipt. Banked negatives stay visible because the gates bite."
    ),
    "sources": [
        "docs/OPH_SIGNATURE_EXPERIMENT_TRACKER.md (section 0a)",
        "docs/BEST_OF_PUBLIC_DATA_COMPARISONS.md",
        "docs/SCALING_MILESTONE_ESTIMATES_2026-07-13.md",
    ],
}
(RUN / "run_highlights.json").write_text(json.dumps(payload, indent=2, sort_keys=False))

lines = [f"# Run highlights: {label}", ""]
lines.append("## Milestones (this run's receipts)")
lines.append("")
lines.append("| Milestone | Receipt | Detail |")
lines.append("|---|---|---|")
for row in milestones:
    mark = "PASS" if row["receipt"] else "open"
    lines.append(f"| {row['milestone']} | {mark} | {row['detail']} |")
lines.append("")
lines.append("## Run-derived diagnostics")
lines.append("")
for row in run_derived:
    lines.append(f"- {row['quantity']}: `{row['value']}` ({row['note']})")
lines.append("")
lines.append("## Near-fits against public data (paper-side lanes, curated)")
lines.append("")
lines.append("| Quantity | OPH | Measured | Pull |")
lines.append("|---|---|---|---|")
for row in near_fits:
    lines.append(f"| {row['quantity']} | {row['oph']} | {row['measured']} | {row['pull']} |")
lines.append("")
lines.append("## Banked negatives (kept visible)")
lines.append("")
for row in banked:
    lines.append(f"- {row}")
lines.append("")
lines.append(f"Claim boundary: {payload['claim_boundary']}")
lines.append("")
(RUN / "RUN_HIGHLIGHTS.md").write_text("\n".join(lines))
print(json.dumps({"run": RUN.name, "milestone_passes": sum(1 for m in milestones if m["receipt"]), "written": ["RUN_HIGHLIGHTS.md", "run_highlights.json"]}, indent=1))
