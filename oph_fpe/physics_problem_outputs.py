from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


LTAS_P_SRC = 1.630972095694329
LTAS_ACOUSTIC_PORT_COUNT = 12
LTAS_ORIENTED_REGISTER_SLOT_COUNT = 24


def physics_problem_outputs_report(
    *,
    hall_k_matrix: list[list[int]] | None = None,
    hall_charge_vector: list[int] | None = None,
    high_tc_thresholds: dict[str, float] | None = None,
    fusion_ledger: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Emit theorem-bounded output contracts for adjacent OPH problem notes."""

    hall = (
        fractional_quantum_hall_k_matrix_readout(hall_k_matrix, hall_charge_vector)
        if hall_k_matrix is not None and hall_charge_vector is not None
        else fractional_quantum_hall_contract()
    )
    return {
        "schema": "oph_physics_problem_outputs_v1",
        "mode": "adjacent_physics_problem_output_contracts",
        "source_documents": [
            "reverse-engineering-reality/physics-problems/fractional_quantum_hall.md",
            "reverse-engineering-reality/physics-problems/fractional_excitons_as_oph_quotient_sector_readouts.md",
            "reverse-engineering-reality/physics-problems/hadronic_precision_endpoint.md",
            "reverse-engineering-reality/physics-problems/jwst_compact_object_source_release.md",
            "reverse-engineering-reality/physics-problems/compact_record_transients.md",
            "reverse-engineering-reality/physics-problems/gamma_ray_morphology_claims_in_oph.md",
            "reverse-engineering-reality/physics-problems/high_energy_messenger_coefficients.md",
            "reverse-engineering-reality/physics-problems/cmb_simulation_promotion_to_physical_prediction.md",
            "reverse-engineering-reality/physics-problems/high_temperature_superconductivity.md",
            "reverse-engineering-reality/physics-problems/low_temperature_amorphous_universality.md",
            "reverse-engineering-reality/physics-problems/plasma_fusion.md",
            "reverse-engineering-reality/physics-problems/e8_spin8_triality_alt9_certificate.md",
        ],
        "outputs": {
            "low_temperature_amorphous_universality": ltas_source_only_readout(),
            "fractional_quantum_hall": hall,
            "fractional_exciton_quotient_sector": fractional_exciton_quotient_contract(),
            "hadronic_precision_endpoint": hadronic_precision_endpoint_contract(),
            "jwst_compact_object_source_release": jwst_compact_object_contract(),
            "compact_record_transients": compact_transient_contract(),
            "gamma_ray_morphology_claims": gamma_morphology_contract(),
            "high_energy_messenger_coefficients": uhe_coefficient_emission_contract(),
            "cmb_simulation_promotion": cmb_simulation_promotion_contract(),
            "high_temperature_superconductivity": high_tc_gate_readout(high_tc_thresholds),
            "plasma_fusion": fusion_ledger_readout(fusion_ledger),
            "e8_spin8_triality_alt9_certificate": e8_triality_certificate_contract(),
        },
        "claim_boundary": (
            "This report computes only output values licensed by the supplied source contracts. "
            "LTAS source-only constants are frozen without measured attenuation data. FQH K-matrix "
            "readouts require an explicit Abelian K,t input. High-Tc and fusion values require "
            "candidate material or plant-ledger inputs and otherwise remain fail-closed contracts. "
            "JWST, gamma, compact-transient, CMB, and UHE outputs are receipt ladders, not "
            "confirmation claims. E8/Spin8 triality is a paper-stack finite certificate skeleton, "
            "not a simulator physics prediction."
        ),
    }


def write_physics_problem_outputs_report(
    out_dir: Path,
    *,
    hall_k_matrix: list[list[int]] | None = None,
    hall_charge_vector: list[int] | None = None,
    high_tc_thresholds: dict[str, float] | None = None,
    fusion_ledger: dict[str, float] | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = physics_problem_outputs_report(
        hall_k_matrix=hall_k_matrix,
        hall_charge_vector=hall_charge_vector,
        high_tc_thresholds=high_tc_thresholds,
        fusion_ledger=fusion_ledger,
    )
    (out / "physics_problem_outputs_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "physics_problem_outputs_report.md").write_text(
        _physics_problem_outputs_markdown(report),
        encoding="utf-8",
    )
    return report


def ltas_source_only_readout(*, p_src: float = LTAS_P_SRC) -> dict[str, Any]:
    collar_survival = math.exp(-float(p_src) / LTAS_ORIENTED_REGISTER_SLOT_COUNT)
    eta_source = collar_survival / (LTAS_ACOUSTIC_PORT_COUNT * LTAS_ORIENTED_REGISTER_SLOT_COUNT)
    c_star = eta_source / (math.pi**2)
    q0_inverse = (math.pi / 2.0) * c_star
    lambda_dom_over_ell = 12.5 * c_star
    receipts = {
        "LTAS_SOURCE_OUTPUT_RECEIPT": True,
        "LTAS_SOURCE_BRANCH_RECEIPT": True,
        "QUOTIENT_INVARIANCE_RECEIPT": True,
        "SOURCE_ACTION_RECEIPT": True,
        "RIGID_RECORD_TLS_RECEIPT": True,
        "SCALE_FLAT_REPAIR_ACTION_RECEIPT": True,
        "PORT_12_UNIFORMITY_RECEIPT": True,
        "REGISTER_24_UNIFORMITY_RECEIPT": True,
        "PORT_REGISTER_TENSOR_RECEIPT": True,
        "PROTECTED_RESERVE_UNIFORMITY_RECEIPT": True,
        "MEAN_FREE_PATH_READOUT_RECEIPT": True,
        "EXCEPTION_CLASS_RECEIPT": False,
        "MATERIAL_SPECIFIC_BRANCH_AUDIT_RECEIPT": False,
    }
    return {
        "schema": "oph_ltas_source_only_readout_v1",
        "status": "source_only_value_emitted",
        "sourcePixel": float(p_src),
        "acousticPortCount": LTAS_ACOUSTIC_PORT_COUNT,
        "orientedRegisterSlotCount": LTAS_ORIENTED_REGISTER_SLOT_COUNT,
        "collarSurvival": collar_survival,
        "etaSource": eta_source,
        "lambdaOverEll": eta_source,
        "acousticTunnelingStrengthCStar": c_star,
        "acousticPlateauQ0Inverse": q0_inverse,
        "dominantThermalLambdaOverEll": lambda_dom_over_ell,
        "formulae": {
            "collarSurvival": "exp(-P_src/24)",
            "etaSource": "exp(-P_src/24)/(12*24)",
            "CStar": "etaSource/pi^2",
            "Q0Inverse": "(pi/2)*CStar",
            "dominantThermalLambdaOverEll": "12.5*CStar",
        },
        "receipts": receipts,
        "blockers": [
            "material_specific_exception_classifier_not_run",
            "measured_attenuation_comparison_not_supplied",
        ],
        "claimBoundary": (
            "Source-only OPH value for the saturated ordinary LTAS branch. It is frozen before "
            "experimental attenuation data are loaded and is not a material-by-material fit."
        ),
    }


def fractional_quantum_hall_contract() -> dict[str, Any]:
    return {
        "schema": "oph_fractional_quantum_hall_output_contract_v1",
        "status": "input_gated_contract",
        "computed": False,
        "requiredInputs": {
            "abelianBranch": ["K_matrix", "charge_vector_t"],
            "fiveHalfSelector": [
                "source_action_S",
                "base_measure_m",
                "sector_projectors",
                "charge_heat_edge_tunneling_interferometry_readouts",
                "refinement_maps",
            ],
        },
        "availableOutputsWhenKMatrixSupplied": [
            "filling_fraction_nu",
            "sigma_xy_in_e2_over_h",
            "quasiparticle_group_order_abs_det_K",
            "representative_charge_fractions",
            "representative_self_statistics_theta_over_pi",
        ],
        "canonicalFormulaExamples": {
            "laughlin_1_3": fractional_quantum_hall_k_matrix_readout(
                [[3]],
                [1],
                quasiparticle_vectors=[[1]],
            ),
            "jain_2_5": fractional_quantum_hall_k_matrix_readout(
                [[3, 2], [2, 3]],
                [1, 1],
            ),
        },
        "receipts": {
            "FQH_ABELIAN_K_MATRIX_READOUT_RECEIPT": False,
            "FQH_NONCENTRAL_5_2_SELECTOR_RECEIPT": False,
        },
        "claimBoundary": (
            "The universe simulator does not currently instantiate a Hall collar source branch. "
            "It can compute exact Abelian K-matrix readouts when K,t are supplied, but it must not "
            "select a nu=5/2 material sector without source-side selector data."
        ),
    }


def fractional_exciton_quotient_contract() -> dict[str, Any]:
    try:
        from oph_fractional.report import fractional_quotient_report
    except Exception as exc:
        return {
            "schema": "oph_fractional_exciton_quotient_contract_v1",
            "status": "report_unavailable",
            "computed": False,
            "error": str(exc),
            "receipts": {
                "FRACTIONAL_QUOTIENT_REPORT_WRITTEN": False,
                "SIMULATOR_QUOTIENT_CORRECTNESS_RECEIPT": False,
                "OPTICAL_LINE_FAN_RECEIPT": False,
                "MATERIAL_SPECIFIC_HAMILTONIAN_PROOF_RECEIPT": False,
            },
            "claimBoundary": "The fractional quotient-sector report could not be imported in this environment.",
        }

    report = fractional_quotient_report()
    gates = report.get("readiness_gates") or {}
    return {
        "schema": "oph_fractional_exciton_quotient_contract_v1",
        "status": "sandbox_report_available",
        "computed": True,
        "claim": report.get("claim"),
        "strongestAllowedClaim": report.get("strongest_allowed_claim"),
        "firstBlockedGate": report.get("first_blocked_gate"),
        "promotionAllowed": bool(report.get("promotion_allowed", False)),
        "materialClaim": bool(report.get("material_claim", False)),
        "receipts": {
            "FRACTIONAL_QUOTIENT_REPORT_WRITTEN": True,
            "SIMULATOR_QUOTIENT_CORRECTNESS_RECEIPT": bool(
                gates.get("SIMULATOR_QUOTIENT_CORRECTNESS", False)
            ),
            "OPTICAL_LINE_FAN_RECEIPT": bool(gates.get("LINE_FAN_DECOMPOSITION", False)),
            "OPTICAL_SECTOR_IDENTIFIABILITY_RECEIPT": bool(
                gates.get("OPTICAL_LINE_FAN_INJECTIVE", False)
            ),
            "NO_TARGET_LEAK_DAG": bool(gates.get("NO_TARGET_LEAK_DAG", False)),
            "MATERIAL_SPECIFIC_HAMILTONIAN_PROOF_RECEIPT": False,
        },
        "blockers": list(report.get("blockers") or []),
        "claimBoundary": report.get("claim_boundary"),
    }


def hadronic_precision_endpoint_contract() -> dict[str, Any]:
    return {
        "schema": "oph_hadronic_precision_endpoint_contract_v1",
        "status": "source_open_receipt_contract",
        "computed": False,
        "milestone": "HVP_ALPHA_SOURCE_PROTOTYPE",
        "requiredSourceObject": [
            "OPH-QCD quotient ensemble",
            "source QCD parameter map",
            "Euclidean QCD slab/vacuum transfer",
            "hadronic Hilbert quotient",
            "Ward-normalized electromagnetic current ledger",
            "two-current spectral export d rho_Q^(2)",
            "four-current spectral export d rho_QQQQ^(4)",
            "B/Sigma transition spectral exports",
            "same-scheme endpoint remainder Xi_Q",
            "systematics ledger",
            "no-target-leak DAG",
        ],
        "forbiddenSourceInputs": [
            "CODATA_ALPHA",
            "MUON_G_MINUS_2",
            "EE_TO_HADRONS",
            "RARE_DECAY_DATA",
            "HADRON_MASS_TARGETS",
            "PDG_QCD_FITS",
        ],
        "receipts": {
            "HADRON_SOURCE_BACKEND_REPORT_WRITTEN": False,
            "TWO_CURRENT_HADRONIC_BACKEND_RECEIPT": False,
            "FULL_HADRONIC_PRECISION_BACKEND_RECEIPT": False,
            "FINE_STRUCTURE_ENDPOINT_PROMOTION_RECEIPT": False,
            "HVP_G_MINUS_2_PROMOTION_RECEIPT": False,
            "HLBL_G_MINUS_2_PROMOTION_RECEIPT": False,
            "RARE_DECAY_LONG_DISTANCE_PROMOTION_RECEIPT": False,
        },
        "claimBoundary": (
            "The simulator can emit and validate the OPH-QCD hadron-source receipt bundle. "
            "It must not fit alpha(0), g-2, e+e- hadronic spectra, rare decays, hadron masses, "
            "or PDG QCD fits and call that a source-only OPH derivation."
        ),
    }


def jwst_compact_object_contract() -> dict[str, Any]:
    try:
        from oph_fpe.jwst import strongest_allowed_claim
    except Exception as exc:
        return {
            "schema": "oph_jwst_compact_object_contract_v1",
            "status": "report_unavailable",
            "computed": False,
            "error": str(exc),
            "receipts": {"JWST_COMPACT_OBJECT_PLAN_WRITTEN": False},
            "claimBoundary": "The JWST compact-object planner could not be imported in this environment.",
        }

    claim, first_blocked, missing = strongest_allowed_claim({})
    return {
        "schema": "oph_jwst_compact_object_contract_v1",
        "status": "source_release_receipt_ladder_available",
        "computed": True,
        "strongestAllowedClaim": claim,
        "firstBlockedGate": first_blocked,
        "missingReceipts": missing,
        "receipts": {
            "JWST_COMPACT_OBJECT_PLAN_WRITTEN": True,
            "CATALOG_INGESTION_RECEIPT": False,
            "OBJECT_SOURCE_LAW_RECEIPT": False,
            "NO_TARGET_LEAKAGE_RECEIPT": False,
            "DEGENERACY_AUDIT_RECEIPT": False,
            "FROZEN_CATALOG_LIKELIHOOD_RECEIPT": False,
            "JWST_LIKELIHOOD_EVALUATED_PHYSICAL_PREDICTION_RECEIPT": False,
        },
        "claimBoundary": (
            "JWST compact-object artifacts are source-release and degeneracy-audit receipts. "
            "Catalog rows, compactness, redness, or luminosity do not by themselves promote "
            "mass-age tension or OPH confirmation claims."
        ),
    }


def compact_transient_contract() -> dict[str, Any]:
    try:
        from oph_fpe.compact_transients import compact_transient_audit_report
    except Exception as exc:
        return {
            "schema": "oph_compact_transient_contract_v1",
            "status": "report_unavailable",
            "computed": False,
            "error": str(exc),
            "receipts": {
                "COMPACT_TRANSIENT_AUDIT_REPORT_WRITTEN": False,
                "PROMOTION_AUDIT_RECEIPT": False,
            },
            "claimBoundary": "The compact-transient audit report could not be imported in this environment.",
        }

    report = compact_transient_audit_report()
    gates = report.get("readiness_gates") or {}
    return {
        "schema": "oph_compact_transient_contract_v1",
        "status": "conditional_receipt_ladder_available",
        "computed": True,
        "claim": report.get("claim"),
        "strongestAllowedClaim": report.get("strongest_allowed_claim"),
        "firstBlockedGate": report.get("first_blocked_gate"),
        "promotionAllowed": bool(report.get("promotion_allowed", False)),
        "physicalClaim": bool(report.get("physical_claim", False)),
        "receipts": {
            "COMPACT_TRANSIENT_AUDIT_REPORT_WRITTEN": True,
            "COMPACT_QUOTIENT_RECEIPT": bool(gates.get("COMPACT_QUOTIENT_RECEIPT", False)),
            "COMPACT_SOURCE_LAW_RECEIPT": bool(gates.get("COMPACT_SOURCE_LAW_RECEIPT", False)),
            "DETECTION_THINNING_RECEIPT": bool(gates.get("DETECTION_THINNING_RECEIPT", False)),
            "CENSORING_AND_UPPER_LIMIT_RECEIPT": bool(
                gates.get("CENSORING_AND_UPPER_LIMIT_RECEIPT", False)
            ),
            "CONTROL_MODEL_RECEIPT": bool(gates.get("CONTROL_MODEL_RECEIPT", False)),
            "REFINEMENT_STABILITY_RECEIPT": bool(gates.get("REFINEMENT_STABILITY_RECEIPT", False)),
            "HELDOUT_LIKELIHOOD_RECEIPT": bool(gates.get("HELDOUT_LIKELIHOOD_RECEIPT", False)),
            "PROMOTION_AUDIT_RECEIPT": bool(gates.get("PROMOTION_AUDIT_RECEIPT", False)),
        },
        "claimBoundary": report.get("claim_boundary"),
    }


def gamma_morphology_contract() -> dict[str, Any]:
    try:
        from oph_fpe.cosmology.gamma_morphology import gamma_morphology_report
    except Exception as exc:
        return {
            "schema": "oph_gamma_morphology_contract_v1",
            "status": "report_unavailable",
            "computed": False,
            "error": str(exc),
            "receipts": {"GAMMA_MORPHOLOGY_REPORT_WRITTEN": False},
            "claimBoundary": "The gamma morphology report could not be imported in this environment.",
        }

    report = gamma_morphology_report()
    gates = report.get("readiness_gates") or {}
    return {
        "schema": "oph_gamma_morphology_contract_v1",
        "status": "diagnostic_gamma_morphology_ladder_available",
        "computed": True,
        "milestone": report.get("milestone"),
        "strongestAllowedClaim": report.get("strongest_allowed_claim"),
        "firstBlockedGate": report.get("first_blocked_gate"),
        "promotionAllowed": bool(report.get("promotion_allowed", False)),
        "receipts": {
            "GAMMA_MORPHOLOGY_REPORT_WRITTEN": True,
            "GAMMA_SOURCE_ARTIFACT_RECEIPT": bool(gates.get("GAMMA_SOURCE_ARTIFACT_RECEIPT", False)),
            "GAMMA_ROUTE_DECLARATION_RECEIPT": bool(gates.get("GAMMA_ROUTE_DECLARATION_RECEIPT", False)),
            "GAMMA_NO_DATA_USE_RECEIPT": bool(gates.get("GAMMA_NO_DATA_USE_RECEIPT", False)),
            "PHOTON_RESPONSE_KERNEL_RECEIPT": bool(gates.get("PHOTON_RESPONSE_KERNEL_RECEIPT", False)),
            "GAMMA_IDENTIFIABILITY_RECEIPT": bool(gates.get("GAMMA_IDENTIFIABILITY_RECEIPT", False)),
            "FROZEN_GAMMA_LIKELIHOOD_RECEIPT": bool(gates.get("FROZEN_GAMMA_LIKELIHOOD_RECEIPT", False)),
            "OPH_GAMMA_MORPHOLOGY_PREDICTION_RECEIPT": bool(
                gates.get("OPH_GAMMA_MORPHOLOGY_PREDICTION_RECEIPT", False)
            ),
        },
        "claimBoundary": report.get("claim_boundary"),
    }


def uhe_coefficient_emission_contract() -> dict[str, Any]:
    try:
        from oph_fpe.uhe_coefficients import coefficient_emission_report
    except Exception as exc:
        return {
            "schema": "oph_uhe_coefficient_emission_contract_v1",
            "status": "report_unavailable",
            "computed": False,
            "error": str(exc),
            "receipts": {
                "UHE_COEFFICIENT_EMISSION_REPORT_WRITTEN": False,
                "NO_UHE_DATA_USE": False,
                "COMMON_SOURCE_LOCK": False,
            },
            "claimBoundary": "The UHE coefficient-emission report could not be imported in this environment.",
        }

    report = coefficient_emission_report()
    gates = report.get("readiness_gates") or {}
    return {
        "schema": "oph_uhe_coefficient_emission_contract_v1",
        "status": "source_only_coefficient_emitter_available",
        "computed": True,
        "claimTier": report.get("claim_tier"),
        "strongestAllowedClaim": report.get("strongest_allowed_claim"),
        "sourceOnly": bool(report.get("source_only", False)),
        "physicalClaim": bool(report.get("physical_claim", False)),
        "coefficients": list(report.get("coefficients") or []),
        "receipts": {
            "UHE_COEFFICIENT_EMISSION_REPORT_WRITTEN": True,
            "BASELINE_FULL_SUPPORT": bool(gates.get("BASELINE_FULL_SUPPORT", False)),
            "FEATURE_MINIMALITY": bool(gates.get("FEATURE_MINIMALITY", False)),
            "MOMENT_INTERIOR": bool(gates.get("MOMENT_INTERIOR", False)),
            "SOURCE_LOAD_QUOTIENT_VISIBLE": bool(gates.get("SOURCE_LOAD_QUOTIENT_VISIBLE", False)),
            "NO_UHE_DATA_USE": bool(gates.get("NO_UHE_DATA_USE", False)),
            "REFINEMENT_COMPATIBILITY": bool(gates.get("REFINEMENT_COMPATIBILITY", False)),
            "COEFFICIENT_SOLVE_CONVERGED": bool(gates.get("COEFFICIENT_SOLVE_CONVERGED", False)),
            "COMMON_SOURCE_LOCK": bool(gates.get("COMMON_SOURCE_LOCK", False)),
        },
        "claimBoundary": report.get("claim_boundary"),
    }


def cmb_simulation_promotion_contract() -> dict[str, Any]:
    try:
        from oph_fpe.cosmology.cmb_promotion_ledger import cmb_promotion_ledger_report
    except Exception as exc:
        return {
            "schema": "oph_cmb_simulation_promotion_contract_v1",
            "status": "report_unavailable",
            "computed": False,
            "error": str(exc),
            "receipts": {"CMB_PROMOTION_LEDGER_WRITTEN": False},
            "claimBoundary": "The CMB promotion ledger could not be imported in this environment.",
        }

    report = cmb_promotion_ledger_report([])
    gates = report.get("readiness_gates") or {}
    return {
        "schema": "oph_cmb_simulation_promotion_contract_v1",
        "status": "promotion_ledger_available",
        "computed": True,
        "currentClaimTier": report.get("current_claim_tier"),
        "firstBlockedGate": report.get("first_blocked_gate"),
        "likelihoodEvaluatedPhysicalPrediction": bool(
            report.get("likelihood_evaluated_physical_cmb_prediction", False)
        ),
        "receipts": {
            "CMB_PROMOTION_LEDGER_WRITTEN": True,
            "VISUAL_DIAGNOSTIC_RECEIPT": bool(gates.get("VISUAL_DIAGNOSTIC_RECEIPT", False)),
            "SPECTRUM_DIAGNOSTIC_RECEIPT": bool(gates.get("SPECTRUM_DIAGNOSTIC_RECEIPT", False)),
            "SOURCE_ONLY_FINITE_ARTIFACT_RECEIPT": bool(
                gates.get("SOURCE_ONLY_FINITE_ARTIFACT_RECEIPT", False)
            ),
            "OPH_NATIVE_COSMO_GEOM_READ_RECEIPT": bool(
                gates.get("OPH_NATIVE_COSMO_GEOM_READ_RECEIPT", False)
            ),
            "FROZEN_LIKELIHOOD_RECEIPT": bool(gates.get("FROZEN_LIKELIHOOD_RECEIPT", False)),
            "PHYSICAL_CMB_PREDICTION_RECEIPT": bool(gates.get("PHYSICAL_CMB_PREDICTION_RECEIPT", False)),
        },
        "claimBoundary": report.get("claim_boundary"),
    }


def fractional_quantum_hall_k_matrix_readout(
    k_matrix: list[list[int]],
    charge_vector: list[int],
    *,
    quasiparticle_vectors: list[list[int]] | None = None,
) -> dict[str, Any]:
    k = [[Fraction(int(value), 1) for value in row] for row in k_matrix]
    t = [Fraction(int(value), 1) for value in charge_vector]
    _validate_square_matrix(k)
    if len(t) != len(k):
        raise ValueError("charge_vector length must match K matrix rank")
    if any(k[i][j] != k[j][i] for i in range(len(k)) for j in range(len(k))):
        raise ValueError("K matrix must be symmetric")
    det = _matrix_determinant(k)
    if det == 0:
        raise ValueError("K matrix must be nondegenerate")
    inv = _matrix_inverse(k)
    nu = _quadratic_form(t, inv, t)
    if quasiparticle_vectors is None:
        quasiparticle_vectors = [
            [1 if i == j else 0 for i in range(len(k))]
            for j in range(len(k))
        ]
    quasiparticles = []
    for ell_raw in quasiparticle_vectors:
        ell = [Fraction(int(value), 1) for value in ell_raw]
        if len(ell) != len(k):
            raise ValueError("quasiparticle vector length must match K matrix rank")
        charge = _bilinear_form(t, inv, ell)
        theta_over_pi = _quadratic_form(ell, inv, ell)
        quasiparticles.append(
            {
                "ell": [int(value) for value in ell],
                "chargeFractionQOverE": _fraction_payload(charge),
                "selfStatisticsThetaOverPi": _fraction_payload(theta_over_pi),
            }
        )
    return {
        "schema": "oph_fractional_quantum_hall_k_matrix_readout_v1",
        "status": "abelian_k_matrix_readout_emitted",
        "computed": True,
        "rank": len(k),
        "K": [[int(value) for value in row] for row in k_matrix],
        "t": [int(value) for value in charge_vector],
        "detK": _fraction_payload(det),
        "quasiparticleGroupOrderAbsDetK": abs(int(det)) if det.denominator == 1 else None,
        "fillingFractionNu": _fraction_payload(nu),
        "sigmaXYUnitsE2OverH": _fraction_payload(nu),
        "representativeQuasiparticles": quasiparticles,
        "receipts": {
            "FQH_ABELIAN_K_MATRIX_READOUT_RECEIPT": True,
            "FQH_NONCENTRAL_5_2_SELECTOR_RECEIPT": False,
        },
        "blockers": ["noncentral_5_2_selector_not_supplied"],
        "claimBoundary": (
            "Exact Abelian K-matrix Hall readout only. This does not select Pfaffian, anti-Pfaffian, "
            "PH-Pfaffian, or other nu=5/2 noncentral sectors."
        ),
    }


def high_tc_gate_readout(thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    if not thresholds:
        return {
            "schema": "oph_high_tc_gate_readout_v1",
            "status": "input_gated_contract",
            "computed": False,
            "requiredInputs": ["T_amp", "T_phase", "T_hol"],
            "optionalPenaltyInputs": [
                "P_inst",
                "P_dis",
                "P_tox",
                "P_synth",
                "lambda_inst",
                "lambda_dis",
                "lambda_tox",
                "lambda_synth",
            ],
            "receipts": {
                "HIGH_TC_GATE_OUTPUT_RECEIPT": False,
                "HIGH_TC_MATERIAL_RECEIPT": False,
            },
            "claimBoundary": (
                "No material candidate thresholds were supplied. The simulator can compute the OPH "
                "bottleneck Tc only from explicit amplitude, phase-stiffness, and holonomy thresholds."
            ),
        }
    missing = [key for key in ("T_amp", "T_phase", "T_hol") if key not in thresholds]
    if missing:
        return {
            "schema": "oph_high_tc_gate_readout_v1",
            "status": "missing_required_thresholds",
            "computed": False,
            "missingInputs": missing,
            "receipts": {
                "HIGH_TC_GATE_OUTPUT_RECEIPT": False,
                "HIGH_TC_MATERIAL_RECEIPT": False,
            },
            "claimBoundary": "High-Tc output remains closed until all three gate thresholds are supplied.",
        }
    gate_values = {
        "T_amp": float(thresholds["T_amp"]),
        "T_phase": float(thresholds["T_phase"]),
        "T_hol": float(thresholds["T_hol"]),
    }
    tc_oph = min(gate_values.values())
    bottlenecks = [key for key, value in gate_values.items() if value == tc_oph]
    penalties = _high_tc_penalty(thresholds)
    return {
        "schema": "oph_high_tc_gate_readout_v1",
        "status": "gate_output_emitted",
        "computed": True,
        "gateThresholds": gate_values,
        "TcOPH": tc_oph,
        "bottleneckGates": bottlenecks,
        "penaltyTotal": penalties["total"],
        "TcPredAfterPenalties": tc_oph - penalties["total"],
        "penaltyTerms": penalties["terms"],
        "receipts": {
            "HIGH_TC_GATE_OUTPUT_RECEIPT": True,
            "HIGH_TC_MATERIAL_RECEIPT": False,
        },
        "blockers": ["structural_transport_magnetic_bulk_material_receipts_not_supplied"],
        "claimBoundary": (
            "Computes the OPH high-Tc bottleneck formula for supplied candidate thresholds. "
            "It is not a material discovery or public superconductivity receipt."
        ),
    }


def fusion_ledger_readout(ledger: dict[str, float] | None = None) -> dict[str, Any]:
    if not ledger:
        return {
            "schema": "oph_fusion_ledger_readout_v1",
            "status": "input_gated_contract",
            "computed": False,
            "requiredInputsForPlantGate": [
                "E_load",
                "E_all_inputs",
                "Delta_E_stored",
                "E_startup",
                "E_shutdown",
                "E_aux",
                "E_consumables",
                "E_maintenance",
                "E_waste",
                "u_L",
            ],
            "optionalInputs": ["P_fus", "P_aux", "P_loss", "P_ch", "W", "tau_E"],
            "receipts": {
                "FUSION_LEDGER_OUTPUT_RECEIPT": False,
                "FUSION_NET_PLANT_PROMOTION_RECEIPT": False,
            },
            "claimBoundary": (
                "No plant or plasma ledger inputs were supplied. Fusion product, captured heat, "
                "delivered power, and net plant output remain separate receipt tiers."
            ),
        }
    plant = _plant_ledger(ledger)
    plasma_gain = _ratio_or_none(ledger.get("P_fus"), ledger.get("P_aux"))
    energy_confinement_time = _ratio_or_none(ledger.get("W"), ledger.get("P_loss"))
    fusion_record_survival = None
    if ledger.get("P_loss") is not None and ledger.get("P_aux") is not None and ledger.get("P_ch") is not None:
        fusion_record_survival = float(ledger["P_ch"]) + float(ledger["P_aux"]) >= float(ledger["P_loss"])
    computed = plant["computed"] or plasma_gain is not None or energy_confinement_time is not None
    return {
        "schema": "oph_fusion_ledger_readout_v1",
        "status": "ledger_output_emitted" if computed else "insufficient_ledger_inputs",
        "computed": computed,
        "plasmaGainQ": plasma_gain,
        "tauE": energy_confinement_time,
        "fusionRecordSurvivalGate": fusion_record_survival,
        "plantLedger": plant,
        "receipts": {
            "FUSION_LEDGER_OUTPUT_RECEIPT": computed,
            "FUSION_NET_PLANT_PROMOTION_RECEIPT": bool(plant.get("netPlantPromotionReceipt", False)),
        },
        "claimBoundary": (
            "Typed fusion ledger only. Passing plasma gain or record survival does not promote captured "
            "heat, delivered load power, or net plant output without the corresponding ledger receipt."
        ),
    }


def e8_triality_certificate_contract() -> dict[str, Any]:
    return {
        "schema": "oph_e8_spin8_triality_certificate_contract_v1",
        "status": "paper_stack_receipt_skeleton",
        "computed": False,
        "certificateId": "E8_SPIN8_TRIALITY_ALT9_DOUBLE_COVER_CERTIFICATE",
        "repositoryReceiptStatus": "pending_raw_bundle",
        "mathematicalContent": {
            "rootSubsystem": "A8 inside E8",
            "finiteGroup": "Alt(9)",
            "spinLift": "nonsplit 2.Alt(9) in Spin(8)",
            "vectorMod2Orbits": [9, 36, 84, 126],
            "spinMod2Orbits": [120, 135],
        },
        "requiredPublicBundle": [
            "Sage source",
            "exact matrix data",
            "lattice bases",
            "mod-2 orbit computation",
            "stdout or machine-readable checks",
            "stable hashes under reverse-engineering-reality/code/e8_triality",
        ],
        "receipts": {
            "E8_TRIALITY_CERTIFICATE_STATEMENT_RECEIPT": True,
            "E8_TRIALITY_PUBLIC_RAW_BUNDLE_RECEIPT": False,
            "E8_TRIALITY_STANDARD_MODEL_SELECTOR_RECEIPT": False,
            "E8_TRIALITY_HARDWARE_RECEIPT": False,
        },
        "claimBoundary": (
            "Finite exceptional representation-closure support only. This certificate does not "
            "prove OPH, select the Standard Model quotient, close the heterotic critical-edge "
            "gate, or count as hardware evidence."
        ),
    }


def _high_tc_penalty(values: dict[str, float]) -> dict[str, Any]:
    terms = {}
    total = 0.0
    for name in ("inst", "dis", "tox", "synth"):
        weight = float(values.get(f"lambda_{name}", 0.0) or 0.0)
        penalty = float(values.get(f"P_{name}", 0.0) or 0.0)
        amount = weight * penalty
        terms[name] = {
            "weight": weight,
            "penalty": penalty,
            "amount": amount,
        }
        total += amount
    return {"terms": terms, "total": total}


def _plant_ledger(values: dict[str, float]) -> dict[str, Any]:
    required = [
        "E_load",
        "E_all_inputs",
        "Delta_E_stored",
        "E_startup",
        "E_shutdown",
        "E_aux",
        "E_consumables",
        "E_maintenance",
        "E_waste",
        "u_L",
    ]
    missing = [key for key in required if key not in values]
    if missing:
        return {
            "computed": False,
            "missingInputs": missing,
            "netPlantPromotionReceipt": False,
        }
    burden = (
        float(values["E_all_inputs"])
        + float(values["Delta_E_stored"])
        + float(values["E_startup"])
        + float(values["E_shutdown"])
        + float(values["E_aux"])
        + float(values["E_consumables"])
        + float(values["E_maintenance"])
        + float(values["E_waste"])
    )
    l_plant = float(values["E_load"]) - burden
    threshold = 5.0 * float(values["u_L"])
    return {
        "computed": True,
        "ELoad": float(values["E_load"]),
        "EBurden": burden,
        "LPlant": l_plant,
        "netPlantThreshold5uL": threshold,
        "netPlantPositiveReceipt": l_plant > 0.0,
        "netPlantPromotionReceipt": l_plant > threshold,
    }


def _validate_square_matrix(matrix: list[list[Fraction]]) -> None:
    if not matrix:
        raise ValueError("K matrix must be nonempty")
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("K matrix must be square")


def _matrix_inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    n = len(matrix)
    aug = [
        list(row) + [Fraction(1 if i == j else 0, 1) for j in range(n)]
        for i, row in enumerate(matrix)
    ]
    for col in range(n):
        pivot = next((row for row in range(col, n) if aug[row][col] != 0), None)
        if pivot is None:
            raise ValueError("matrix is singular")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0:
                continue
            aug[row] = [value - factor * aug[col][idx] for idx, value in enumerate(aug[row])]
    return [row[n:] for row in aug]


def _matrix_determinant(matrix: list[list[Fraction]]) -> Fraction:
    n = len(matrix)
    work = [list(row) for row in matrix]
    det = Fraction(1, 1)
    for col in range(n):
        pivot = next((row for row in range(col, n) if work[row][col] != 0), None)
        if pivot is None:
            return Fraction(0, 1)
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
            det *= -1
        pivot_value = work[col][col]
        det *= pivot_value
        for row in range(col + 1, n):
            factor = work[row][col] / pivot_value
            for idx in range(col, n):
                work[row][idx] -= factor * work[col][idx]
    return det


def _bilinear_form(left: list[Fraction], matrix: list[list[Fraction]], right: list[Fraction]) -> Fraction:
    return sum(
        left[i] * matrix[i][j] * right[j]
        for i in range(len(left))
        for j in range(len(right))
    )


def _quadratic_form(left: list[Fraction], matrix: list[list[Fraction]], right: list[Fraction]) -> Fraction:
    return _bilinear_form(left, matrix, right)


def _fraction_payload(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": int(value.numerator),
        "denominator": int(value.denominator),
        "decimal": float(value),
        "text": str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}",
    }


def _ratio_or_none(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_f = float(denominator)
    if denominator_f == 0.0:
        return None
    return float(numerator) / denominator_f


def _physics_problem_outputs_markdown(report: dict[str, Any]) -> str:
    outputs = report["outputs"]
    ltas = outputs["low_temperature_amorphous_universality"]
    fq_hall = outputs["fractional_quantum_hall"]
    jwst = outputs["jwst_compact_object_source_release"]
    compact_transients = outputs["compact_record_transients"]
    gamma = outputs["gamma_ray_morphology_claims"]
    uhe_coefficients = outputs["high_energy_messenger_coefficients"]
    cmb = outputs["cmb_simulation_promotion"]
    high_tc = outputs["high_temperature_superconductivity"]
    fusion = outputs["plasma_fusion"]
    e8 = outputs["e8_spin8_triality_alt9_certificate"]
    lines = [
        "# OPH Physics Problem Outputs",
        "",
        f"- schema: `{report['schema']}`",
        f"- mode: `{report['mode']}`",
        "",
        "## Low-Temperature Amorphous Universality",
        "",
        f"- status: `{ltas['status']}`",
        f"- eta source / lambda over ell: `{ltas['etaSource']:.16g}`",
        f"- C star: `{ltas['acousticTunnelingStrengthCStar']:.16g}`",
        f"- Q0 inverse: `{ltas['acousticPlateauQ0Inverse']:.16g}`",
        f"- dominant thermal lambda over ell: `{ltas['dominantThermalLambdaOverEll']:.16g}`",
        "",
        "## Fractional Quantum Hall",
        "",
        f"- status: `{fq_hall['status']}`",
        f"- Abelian K-matrix readout receipt: `{fq_hall['receipts']['FQH_ABELIAN_K_MATRIX_READOUT_RECEIPT']}`",
        f"- 5/2 noncentral selector receipt: `{fq_hall['receipts']['FQH_NONCENTRAL_5_2_SELECTOR_RECEIPT']}`",
        "",
        "## JWST Compact Objects",
        "",
        f"- status: `{jwst['status']}`",
        f"- strongest allowed claim: `{jwst.get('strongestAllowedClaim')}`",
        f"- first blocked gate: `{jwst.get('firstBlockedGate')}`",
        "",
        "## Compact Record Transients",
        "",
        f"- status: `{compact_transients['status']}`",
        f"- claim: `{compact_transients.get('claim')}`",
        f"- first blocked gate: `{compact_transients.get('firstBlockedGate')}`",
        f"- promotion allowed: `{compact_transients.get('promotionAllowed')}`",
        "",
        "## Gamma-Ray Morphology",
        "",
        f"- status: `{gamma['status']}`",
        f"- strongest allowed claim: `{gamma.get('strongestAllowedClaim')}`",
        f"- first blocked gate: `{gamma.get('firstBlockedGate')}`",
        "",
        "## High-Energy Messenger Coefficients",
        "",
        f"- status: `{uhe_coefficients['status']}`",
        f"- claim tier: `{uhe_coefficients.get('claimTier')}`",
        f"- strongest allowed claim: `{uhe_coefficients.get('strongestAllowedClaim')}`",
        f"- no-UHE-data-use receipt: `{uhe_coefficients['receipts']['NO_UHE_DATA_USE']}`",
        f"- common-source lock: `{uhe_coefficients['receipts']['COMMON_SOURCE_LOCK']}`",
        "",
        "## CMB Simulation Promotion",
        "",
        f"- status: `{cmb['status']}`",
        f"- current claim tier: `{cmb.get('currentClaimTier')}`",
        f"- physical prediction receipt: `{cmb['receipts']['PHYSICAL_CMB_PREDICTION_RECEIPT']}`",
        "",
        "## High-Temperature Superconductivity",
        "",
        f"- status: `{high_tc['status']}`",
        f"- gate output receipt: `{high_tc['receipts']['HIGH_TC_GATE_OUTPUT_RECEIPT']}`",
        "",
        "## Plasma Fusion",
        "",
        f"- status: `{fusion['status']}`",
        f"- ledger output receipt: `{fusion['receipts']['FUSION_LEDGER_OUTPUT_RECEIPT']}`",
        f"- net plant promotion receipt: `{fusion['receipts']['FUSION_NET_PLANT_PROMOTION_RECEIPT']}`",
        "",
        "## E8/Spin8 Triality Certificate",
        "",
        f"- status: `{e8['status']}`",
        f"- certificate statement receipt: `{e8['receipts']['E8_TRIALITY_CERTIFICATE_STATEMENT_RECEIPT']}`",
        f"- public raw bundle receipt: `{e8['receipts']['E8_TRIALITY_PUBLIC_RAW_BUNDLE_RECEIPT']}`",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
    ]
    return "\n".join(lines)
