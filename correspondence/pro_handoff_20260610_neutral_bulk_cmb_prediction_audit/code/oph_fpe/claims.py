from __future__ import annotations

from typing import Any


RECEIPT_SCHEMA_VERSION = "2026-06-06b"


RECOVERED_CORE = "recovered_core"
QUANTITATIVE_BRANCH = "quantitative_branch"
CONTINUATION = "continuation"
PROXY = "proxy"
DEMO = "demo"
DEBUG = "debug"
BRANCH_INSTANTIATION_SANITY = "branch_instantiation_sanity"
DECLARED_SHAPE_SUBSTRATE_WITNESS = "declared_shape_substrate_witness"

CLAIM_LEVELS = {
    RECOVERED_CORE,
    QUANTITATIVE_BRANCH,
    CONTINUATION,
    PROXY,
    DEMO,
    DEBUG,
    BRANCH_INSTANTIATION_SANITY,
    DECLARED_SHAPE_SUBSTRATE_WITNESS,
}

REPAIR_CORE_RECEIPT = "REPAIR_CORE_RECEIPT"
RECORD_COMMIT_RECEIPT = "RECORD_COMMIT_RECEIPT"
BW_KMS_BRANCH_INSTANTIATION_RECEIPT = "BW_KMS_BRANCH_INSTANTIATION_RECEIPT"
CONFORMAL_H3_CHART_RECEIPT = "CONFORMAL_H3_CHART_RECEIPT"
CHART_LORENTZ_H3_RECEIPT = "CHART_LORENTZ_H3_RECEIPT"
PAPER_THEOREM_3D_BULK_CHART_RECEIPT = "PAPER_THEOREM_3D_BULK_CHART_RECEIPT"
BW_KMS_DIRECT_2PI_RECEIPT = "BW_KMS_DIRECT_2PI_RECEIPT"
ENDOGENOUS_MODULAR_GENERATOR_RECEIPT = "ENDOGENOUS_MODULAR_GENERATOR_RECEIPT"
H3_RESPONSE_CONTROL_SEPARATION_RECEIPT = "H3_RESPONSE_CONTROL_SEPARATION_RECEIPT"
H3_RESPONSE_CANDIDATE_RECEIPT = "H3_RESPONSE_CANDIDATE_RECEIPT"
OBJECT_CHART_RECEIPT = "OBJECT_CHART_RECEIPT"
OBJECT_BULK_POPULATION_RECEIPT = "OBJECT_BULK_POPULATION_RECEIPT"
SUPPORT_VISIBLE_H3_POPULATED_BULK_RECEIPT = "SUPPORT_VISIBLE_H3_POPULATED_BULK_RECEIPT"
SCREEN_DEFECT_DYNAMICS_RECEIPT = "SCREEN_DEFECT_DYNAMICS_RECEIPT"
H3_DEFECT_WORLDLINE_RECEIPT = "H3_DEFECT_WORLDLINE_RECEIPT"
PROTO_PARTICLE_RECEIPT = "PROTO_PARTICLE_RECEIPT"
SCREEN_PROXY_CMB_RECEIPT = "SCREEN_PROXY_CMB_RECEIPT"
STATIC_GALAXY_LAW_RECEIPT = "STATIC_GALAXY_LAW_RECEIPT"
STATIC_GALAXY_RAR_BTFR_RECEIPT = "STATIC_GALAXY_RAR_BTFR_RECEIPT"
OPH_STATIC_GALAXY_BRIDGE_RECEIPT = "OPH_STATIC_GALAXY_BRIDGE_RECEIPT"
DYNAMIC_DARK_TRANSPORT_RECEIPT = "DYNAMIC_DARK_TRANSPORT_RECEIPT"
OPH_BOLTZMANN_KERNEL_RECEIPT = "OPH_BOLTZMANN_KERNEL_RECEIPT"
PHYSICAL_CMB_RECEIPT = "PHYSICAL_CMB_RECEIPT"
COSMOLOGY_PERTURBATION_RECEIPT = "COSMOLOGY_PERTURBATION_RECEIPT"
SHAPE_VERTEX_SCATTERING_RECEIPT = "SHAPE_VERTEX_SCATTERING_RECEIPT"
SHAPE_DODECA_CELL_RECEIPT = "SHAPE_DODECA_CELL_RECEIPT"
SHAPE_LOOP_MODE_RECEIPT = "SHAPE_LOOP_MODE_RECEIPT"
SHAPE_SETTLING_RECEIPT = "SHAPE_SETTLING_RECEIPT"
SHAPE_LOOP_PARTICLE_RECEIPT = "SHAPE_LOOP_PARTICLE_RECEIPT"
SHAPE_SCREEN_PROJECTION_RECEIPT = "SHAPE_SCREEN_PROJECTION_RECEIPT"
SHAPE_CMB_CERTIFICATE_INPUT_RECEIPT = "SHAPE_CMB_CERTIFICATE_INPUT_RECEIPT"
SHAPE_VISIBLE_IR_TARGET_RECEIPT = "SHAPE_VISIBLE_IR_TARGET_RECEIPT"
SHAPE_FOUR_SECTOR_IR_RECEIPT = "SHAPE_FOUR_SECTOR_IR_RECEIPT"
DECLARED_SHAPE_SUBSTRATE_WITNESS_RECEIPT = "DECLARED_SHAPE_SUBSTRATE_WITNESS_RECEIPT"

CANONICAL_RECEIPTS = {
    "R0": REPAIR_CORE_RECEIPT,
    "R1": RECORD_COMMIT_RECEIPT,
    "R2": BW_KMS_BRANCH_INSTANTIATION_RECEIPT,
    "R3": CHART_LORENTZ_H3_RECEIPT,
    "R4": H3_RESPONSE_CANDIDATE_RECEIPT,
    "R5": OBJECT_CHART_RECEIPT,
    "R6": OBJECT_BULK_POPULATION_RECEIPT,
    "R7": SCREEN_PROXY_CMB_RECEIPT,
    "R8": STATIC_GALAXY_RAR_BTFR_RECEIPT,
    "R9": DYNAMIC_DARK_TRANSPORT_RECEIPT,
    "R10": COSMOLOGY_PERTURBATION_RECEIPT,
}


def receipt_meta(
    *,
    receipt: str,
    claim_level: str,
    physical_claim: bool,
    observable_id: str,
    fit_objective: str,
) -> dict[str, Any]:
    return {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_name": str(receipt),
        "claim_level": checked_claim_level(claim_level),
        "physical_claim": bool(physical_claim),
        "observable_id": str(observable_id),
        "fit_objective": str(fit_objective),
    }


def checked_claim_level(value: str) -> str:
    level = str(value)
    if level not in CLAIM_LEVELS:
        raise ValueError(f"unknown OPH claim level: {value}")
    return level


def with_claim_metadata(
    report: dict[str, Any],
    *,
    claim_level: str,
    receipt: str | None = None,
    physical_claim: bool | None = None,
    observable_id: str | None = None,
    fit_objective: str | None = None,
) -> dict[str, Any]:
    result = dict(report)
    result["receipt_schema_version"] = RECEIPT_SCHEMA_VERSION
    result["claim_level"] = checked_claim_level(claim_level)
    if receipt is not None:
        result["receipt_name"] = str(receipt)
    result["observable_id"] = str(
        observable_id
        or result.get("observable_id")
        or receipt
        or result.get("mode")
        or "unspecified"
    )
    result["fit_objective"] = str(fit_objective or result.get("fit_objective") or "not_applicable")
    if physical_claim is None:
        result.setdefault("physical_claim", False)
    else:
        result["physical_claim"] = bool(physical_claim)
    return result
