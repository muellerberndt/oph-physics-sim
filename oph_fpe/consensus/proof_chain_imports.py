from __future__ import annotations

import math
from typing import Any

from oph_fpe.claims import (
    CHI_NU_G10_ENERGY_LEDGER_THEOREM_RECEIPT,
    CHI_NU_G9_RECORD_GRAVITY_BRIDGE_RECEIPT,
    DEMO,
    FINITE_MODULAR_FLOW_IMPORT_RECEIPT,
    MATSCHEKO_PROOF_CHAIN_IMPORT_RECEIPT,
    P_BRANCH_PROVENANCE_RECEIPT,
    QBFT_BOUNDARY_CAVEAT_RECEIPT,
    RULE90_DIAGONAL_SCREEN_IMPORT_RECEIPT,
    RULE90_PARITY_SPLITTING_IMPORT_RECEIPT,
    RULE90_TWO_POWER_UNIVERSALITY_IMPORT_RECEIPT,
    SCALAR_CHANNEL_BRIDGE_IMPORT_RECEIPT,
    TWELVE_PORT_SURFACE_IMPORT_RECEIPT,
    with_claim_metadata,
)

P_ROOT_REPORTED = 1.63097209569
CODATA_ALPHA_INV_2022 = 137.035999177
DESIGN_DELTA_M_KG = 0.056
EARTH_SURFACE_POTENTIAL_J_PER_KG = 6.25e7
STANDARD_GRAVITY = 9.81


def matscheko_proof_chain_import_report() -> dict[str, Any]:
    p_branch = p_branch_provenance_report()
    g10 = chi_nu_g10_energy_ledger_report()
    g9 = chi_nu_g9_bridge_gate_report()
    finite_imports = finite_audit_import_report()

    blockers = []
    if not p_branch[P_BRANCH_PROVENANCE_RECEIPT]:
        blockers.append("p_branch_provenance_failed")
    if not g10[CHI_NU_G10_ENERGY_LEDGER_THEOREM_RECEIPT]:
        blockers.append("g10_energy_ledger_arithmetic_failed")
    if not finite_imports["receipt"]:
        blockers.extend(finite_imports["blockers"])

    report = {
        "mode": "matscheko_proof_chain_import_v2",
        MATSCHEKO_PROOF_CHAIN_IMPORT_RECEIPT: not blockers,
        "receipt": not blockers,
        "pBranch": p_branch,
        "g9RecordGravityBridge": g9,
        "g10EnergyLedger": g10,
        "finiteAuditImports": finite_imports,
        "proofChainProvenance": {
            "repository": "dmatscheko/chi_nu_test",
            "commit": "0f9e43b36386ad15e94947751500bf32ee9ccc58",
            "campaign": "v10",
            "leanModules": [
                "OPHProofChain.Rule90Parity",
                "OPHProofChain.Rule90TwoPower",
                "OPHProofChain.Rule90Diagonal",
            ],
        },
        "physicsPromotionBlockers": [
            "G9 record-DeltaS to gravity-DeltaS calibration is open",
            "G10 full interaction-energy pricing is a named decision-layer convention",
            "SEE, MAR, collar instantiation, BW/geometric identification, and continuum transfer remain paper hypotheses",
        ],
        "blockers": blockers,
        "claim_boundary": (
            "Simulator receipt for importing Matscheko's proof-chain audit status. "
            "It mirrors finite theorem status and keeps physics gates explicit; it is not a physical prediction."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=DEMO,
        receipt=MATSCHEKO_PROOF_CHAIN_IMPORT_RECEIPT,
        physical_claim=False,
        observable_id="matscheko_proof_chain_import_status",
        fit_objective="proof_chain_status_receipt",
    )


def p_branch_provenance_report() -> dict[str, Any]:
    p_pub = (1.0 + math.sqrt(5.0)) / 2.0 + math.sqrt(math.pi) / CODATA_ALPHA_INV_2022
    chi_pub = math.exp(-p_pub / 24.0)
    chi_root = math.exp(-P_ROOT_REPORTED / 24.0)
    p_gap = P_ROOT_REPORTED - p_pub
    chi_gap = chi_pub - chi_root
    receipt = (
        1.630968209403959 < p_pub < 1.630968209403960
        and 3.8e-6 < p_gap < 4.0e-6
        and 0.93430063 < chi_pub < 0.93430066
        and 1.4e-7 < chi_gap < 1.6e-7
    )
    return {
        "mode": "two_p_branch_provenance_v1",
        P_BRANCH_PROVENANCE_RECEIPT: receipt,
        "receipt": receipt,
        "Ppub": p_pub,
        "ProotReported": P_ROOT_REPORTED,
        "Pgap": p_gap,
        "chiCanPub": chi_pub,
        "chiCanRoot": chi_root,
        "chiBranchGap": chi_gap,
        "externalInputUsedByPpub": "CODATA_ALPHA_INV_2022",
        "claim_boundary": (
            "Ppub is the CODATA-calibrated comparison branch by definition; "
            "the zero-input solver branch is distinct and is not a Thomson endpoint derivation."
        ),
    }


def chi_nu_g9_bridge_gate_report() -> dict[str, Any]:
    return {
        "mode": "chi_nu_g9_record_gravity_bridge_gate_v1",
        CHI_NU_G9_RECORD_GRAVITY_BRIDGE_RECEIPT: False,
        "receipt": False,
        "definitionSideClosed": True,
        "calibrationClosed": False,
        "boundsOnlyProduct": "chi_nu * DeltaS",
        "blockers": ["missing numerical map from apparatus record contrast to gravitational scalar"],
        "claim_boundary": (
            "The finite coherent-source generator is a formal object, but the numerical "
            "record-DeltaS to gravity-DeltaS calibration remains open."
        ),
    }


def chi_nu_g10_energy_ledger_report() -> dict[str, Any]:
    bench_cycle_work_per_m = DESIGN_DELTA_M_KG * STANDARD_GRAVITY
    convention_toggle_j = DESIGN_DELTA_M_KG * EARTH_SURFACE_POTENTIAL_J_PER_KG
    creation_ceiling_j = DESIGN_DELTA_M_KG * 299_792_458.0**2
    receipt = (
        0.549 < bench_cycle_work_per_m < 0.550
        and 3.49e6 < convention_toggle_j < 3.52e6
        and 5.03e15 < creation_ceiling_j < 5.04e15
        and bench_cycle_work_per_m < convention_toggle_j < creation_ceiling_j
    )
    return {
        "mode": "chi_nu_g10_energy_ledger_v1",
        CHI_NU_G10_ENERGY_LEDGER_THEOREM_RECEIPT: receipt,
        "receipt": receipt,
        "benchCycleWorkPerMeterJ": bench_cycle_work_per_m,
        "namedConventionToggleJ": convention_toggle_j,
        "sourceCreationCeilingJ": creation_ceiling_j,
        "conventionIsTheorem": False,
        "claim_boundary": (
            "Cycle identities and arithmetic are theorem-side checks; full interaction-energy "
            "toggle pricing is a named convention, not a theorem."
        ),
    }


def finite_audit_import_report() -> dict[str, Any]:
    imports = {
        FINITE_MODULAR_FLOW_IMPORT_RECEIPT: {
            "status": "imported_status",
            "scope": "finite matrix modular flow, KMS boundary condition, and Skolem-Noether uniqueness",
            "physics_residue": "BW/geometric identification and scaling limit",
        },
        SCALAR_CHANNEL_BRIDGE_IMPORT_RECEIPT: {
            "status": "imported_status",
            "scope": "single indexed family bridges record slots and collar slices",
            "physics_residue": "channel identification and G9 calibration",
        },
        TWELVE_PORT_SURFACE_IMPORT_RECEIPT: {
            "status": "imported_status",
            "scope": "triangulated-surface double-counting support for twelve-port bookkeeping",
            "physics_residue": "collar instantiation as that surface",
        },
        QBFT_BOUNDARY_CAVEAT_RECEIPT: {
            "status": "imported_status",
            "scope": "quorum-overlap safety caveat for fixed q=2f+1 outside n=3f+1",
            "physics_residue": "none; this is a finite consensus appendix caveat",
        },
        RULE90_PARITY_SPLITTING_IMPORT_RECEIPT: {
            "status": "imported_status",
            "scope": (
                "T38: on even rings, Rule 90 splits into two independent Rule-60 parity "
                "sectors, and arbitrary-screen failure has a nonzero single-parity ghost witness"
            ),
            "physics_residue": "finite binary audit fixture only; no microscopic-law identification",
        },
        RULE90_TWO_POWER_UNIVERSALITY_IMPORT_RECEIPT: {
            "status": "imported_status",
            "scope": (
                "T39: on power-of-two rings, every adjacent-pair worldline screen decodes "
                "exactly at n <= 2(t+1), including noncausal column paths"
            ),
            "physics_residue": "finite binary audit fixture only; no spacetime-causality promotion",
        },
        RULE90_DIAGONAL_SCREEN_IMPORT_RECEIPT: {
            "status": "imported_status",
            "scope": (
                "T40/T41: one lightlike diagonal is counting-tight on odd rings and never "
                "decodes on even rings; opposite-parity diagonal pairs are sharp on even rings"
            ),
            "physics_residue": "finite binary audit fixture only; no physical observer-screen promotion",
        },
    }
    blockers = [key for key, value in imports.items() if value["status"] != "imported_status"]
    report = {
        "mode": "finite_audit_imports_v1",
        "receipt": not blockers,
        "imports": imports,
        "blockers": blockers,
    }
    for key in imports:
        report[key] = key not in blockers
    return report
