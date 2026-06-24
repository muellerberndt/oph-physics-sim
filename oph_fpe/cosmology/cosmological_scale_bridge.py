from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from oph_fpe.cosmology.claim_tiers import ClaimTier, GeometryOrigin, normalize_claim_tier, normalize_geometry_origin


def validate_physical_scale_bridge_receipts(receipts: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the #372 physical scale-bridge receipt bundle.

    The validator recomputes gate status from evidence fields. A bare
    ``PHYSICAL_K_RECEIPT: true`` flag is ignored unless the corresponding hashes,
    origin, basis, and no-fit receipts are present.
    """

    payload = receipts if isinstance(receipts, dict) else {}
    tier = normalize_claim_tier(payload.get("claim_tier"))
    origin = normalize_geometry_origin(payload.get("geometry_origin"))
    valid_physical_origin = (
        (tier == ClaimTier.CONDITIONAL_PHYSICAL and origin == GeometryOrigin.IMPORTED_FLRW)
        or (tier == ClaimTier.OPH_NATIVE_PHYSICAL and origin == GeometryOrigin.OPH_NATIVE)
    )
    hashes = {
        name: payload.get(name)
        for name in (
            "geometry_hash",
            "source_embedding_hash",
            "scale_certificate_hash",
            "background_hash",
            "clock_hash",
            "operator_hash",
            "boundary_condition_hash",
            "solver_convention_hash",
            "mode_basis_hash",
            "mode_lineage_hash",
            "no_peak_fit_receipt_hash",
        )
    }
    hash_checks = {name: _valid_hash(value) for name, value in hashes.items()}
    same_hashes = _same_hashes(payload)
    eigenvalue_intervals = _valid_intervals(payload.get("eigenvalue_intervals"))
    k_intervals = _valid_intervals(payload.get("k_intervals_Mpc_inverse"), positive=True)
    safe_band = _valid_band(payload.get("safe_resolved_band"))
    nofit = bool(payload.get("NO_POSTHOC_CALIBRATION_RECEIPT")) and hash_checks["no_peak_fit_receipt_hash"]
    refinement = _valid_refinement(payload.get("refinement_errors"))
    mode_normalization = bool(payload.get("PHYSICAL_MODE_NORMALIZATION_RECEIPT")) and _valid_hash(
        payload.get("mode_normalization_hash")
    )
    mode_lineage = bool(payload.get("MODE_LINEAGE_RECEIPT")) and hash_checks["mode_lineage_hash"]
    flrw = bool(payload.get("FLRW_BACKGROUND_REDUCTION_RECEIPT")) and hash_checks["background_hash"]
    source_embedding = bool(payload.get("SOURCE_SCREEN_EMBEDDING_RECEIPT")) and hash_checks[
        "source_embedding_hash"
    ]
    scale = bool(payload.get("DIMENSIONAL_SCALE_RECEIPT")) and hash_checks["scale_certificate_hash"]
    geometry = bool(payload.get("FINITE_COSMOLOGICAL_GEOMETRY_RECEIPT")) and hash_checks["geometry_hash"]
    operator = bool(payload.get("PHYSICAL_MODE_OPERATOR_RECEIPT")) and hash_checks["operator_hash"]
    physical_k = bool(
        valid_physical_origin
        and geometry
        and source_embedding
        and scale
        and flrw
        and operator
        and mode_normalization
        and mode_lineage
        and refinement
        and nofit
        and eigenvalue_intervals
        and k_intervals
        and safe_band
        and payload.get("a0_convention") == "a0=1"
        and str(payload.get("k_unit_convention")) == "Mpc^-1"
    )
    angular = bool(
        valid_physical_origin
        and bool(payload.get("CLOSED_SPHERE_OR_EXTENSION_RECEIPT"))
        and bool(payload.get("ANGULAR_OPERATOR_RECEIPT"))
        and bool(payload.get("ANGULAR_CLUSTER_RECEIPT"))
        and refinement
        and str(payload.get("source_angular_mode_type")) == "ell_src"
        and bool(payload.get("observed_cmb_multipole_is_solver_output"))
        and payload.get("source_angular_mode_type") != payload.get("observed_multipole_type")
    )
    a_evolution = bool(
        valid_physical_origin
        and bool(payload.get("PROPER_TIME_CLOCK_RECEIPT"))
        and bool(payload.get("LINEAGE_RECEIPT"))
        and flrw
        and bool(payload.get("VOLUME_SCALE_FACTOR_RECEIPT"))
        and bool(payload.get("EXPANSION_SCALE_FACTOR_RECEIPT"))
        and _finite_nonnegative(payload.get("a_consistency_error"))
        and _finite_nonnegative(payload.get("flrw_residual"))
        and _monotone_table(payload.get("proper_time_table"))
        and _monotone_table(payload.get("a_volume_table"))
        and _monotone_table(payload.get("redshift_table"), decreasing=True)
        and _monotone_table(payload.get("conformal_time_table"))
        and refinement
    )
    freezeout = bool(
        valid_physical_origin
        and bool(payload.get("SPACELIKE_FREEZEOUT_SLICE_RECEIPT"))
        and bool(payload.get("FREEZEOUT_PERSISTENCE_RECEIPT"))
        and bool(payload.get("CURVATURE_RESIDUAL_RECEIPT"))
        and bool(payload.get("ADIABATIC_MODE_RECEIPT"))
        and bool(payload.get("ISOCURVATURE_BOUND_RECEIPT"))
        and bool(payload.get("GROWING_MODE_RECEIPT"))
        and bool(payload.get("CONSTRAINT_RESIDUAL_RECEIPT"))
        and bool(payload.get("FREEZEOUT_INITIAL_DATA_RECEIPT"))
        and _valid_hash(payload.get("surface_mesh_hash"))
        and _valid_hash(payload.get("state_vector_hash"))
        and _valid_hash(payload.get("normal_derivative_hash"))
        and bool(payload.get("common_surface_passed") or payload.get("mode_dependent_freezeout_map"))
        and refinement
    )
    aggregate = bool(physical_k and angular and a_evolution and freezeout and refinement and nofit and same_hashes)
    gates = {
        "PHYSICAL_K_RECEIPT": physical_k,
        "SOURCE_ANGULAR_MODE_RECEIPT": angular,
        "CALIBRATED_A_EVOLUTION_RECEIPT": a_evolution,
        "PHYSICAL_FREEZEOUT_SURFACE_RECEIPT": freezeout,
        "SCALE_BRIDGE_REFINEMENT_RECEIPT": refinement,
        "NO_POSTHOC_CALIBRATION_RECEIPT": nofit,
        "PHYSICAL_SCALE_BRIDGE_RECEIPT": aggregate,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    if tier == ClaimTier.OPH_NATIVE_PHYSICAL and origin != GeometryOrigin.OPH_NATIVE:
        blockers.append("oph_native_geometry_origin_missing")
    if tier == ClaimTier.CONDITIONAL_PHYSICAL and origin != GeometryOrigin.IMPORTED_FLRW:
        blockers.append("conditional_imported_flrw_origin_missing")
    if not valid_physical_origin:
        blockers.append("claim_tier_geometry_origin_mismatch")
    if not same_hashes:
        blockers.append("cross_receipt_hash_mismatch")
    for name, passed in hash_checks.items():
        if not passed:
            blockers.append(f"{name}_missing")
    return {
        **gates,
        "claim_tier": tier.value,
        "geometry_origin": origin.value,
        "oph_native_k_derivation": bool(physical_k and origin == GeometryOrigin.OPH_NATIVE),
        "hash_checks": hash_checks,
        "cross_consistency": same_hashes,
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "Structured physical scale-bridge validation. Conditional physical receipts may import "
            "frozen FLRW geometry; OPH-native physical receipts require OPH_NATIVE geometry origin."
        ),
    }


def imported_flrw_reference_receipts(*, hash_seed: str = "0") -> dict[str, Any]:
    """Small deterministic valid Route-A receipt fixture for analytic/reference tests."""

    hashes = {
        "geometry_hash": _fixture_hash(hash_seed, "geometry"),
        "source_embedding_hash": _fixture_hash(hash_seed, "embedding"),
        "scale_certificate_hash": _fixture_hash(hash_seed, "scale"),
        "background_hash": _fixture_hash(hash_seed, "background"),
        "clock_hash": _fixture_hash(hash_seed, "clock"),
        "operator_hash": _fixture_hash(hash_seed, "operator"),
        "boundary_condition_hash": _fixture_hash(hash_seed, "boundary"),
        "solver_convention_hash": _fixture_hash(hash_seed, "solver"),
        "mode_basis_hash": _fixture_hash(hash_seed, "basis"),
        "mode_lineage_hash": _fixture_hash(hash_seed, "lineage"),
        "mode_normalization_hash": _fixture_hash(hash_seed, "normalization"),
        "no_peak_fit_receipt_hash": _fixture_hash(hash_seed, "nofit"),
        "surface_mesh_hash": _fixture_hash(hash_seed, "surface"),
        "state_vector_hash": _fixture_hash(hash_seed, "state"),
        "normal_derivative_hash": _fixture_hash(hash_seed, "normal"),
    }
    return {
        "claim_tier": ClaimTier.CONDITIONAL_PHYSICAL.value,
        "geometry_origin": GeometryOrigin.IMPORTED_FLRW.value,
        **hashes,
        "FINITE_COSMOLOGICAL_GEOMETRY_RECEIPT": True,
        "SOURCE_SCREEN_EMBEDDING_RECEIPT": True,
        "DIMENSIONAL_SCALE_RECEIPT": True,
        "FLRW_BACKGROUND_REDUCTION_RECEIPT": True,
        "PHYSICAL_MODE_OPERATOR_RECEIPT": True,
        "PHYSICAL_MODE_NORMALIZATION_RECEIPT": True,
        "MODE_LINEAGE_RECEIPT": True,
        "NO_POSTHOC_CALIBRATION_RECEIPT": True,
        "CLOSED_SPHERE_OR_EXTENSION_RECEIPT": True,
        "ANGULAR_OPERATOR_RECEIPT": True,
        "ANGULAR_CLUSTER_RECEIPT": True,
        "PROPER_TIME_CLOCK_RECEIPT": True,
        "LINEAGE_RECEIPT": True,
        "VOLUME_SCALE_FACTOR_RECEIPT": True,
        "EXPANSION_SCALE_FACTOR_RECEIPT": True,
        "SPACELIKE_FREEZEOUT_SLICE_RECEIPT": True,
        "FREEZEOUT_PERSISTENCE_RECEIPT": True,
        "CURVATURE_RESIDUAL_RECEIPT": True,
        "ADIABATIC_MODE_RECEIPT": True,
        "ISOCURVATURE_BOUND_RECEIPT": True,
        "GROWING_MODE_RECEIPT": True,
        "CONSTRAINT_RESIDUAL_RECEIPT": True,
        "FREEZEOUT_INITIAL_DATA_RECEIPT": True,
        "a0_convention": "a0=1",
        "k_unit_convention": "Mpc^-1",
        "curvature_convention": "K_FLRW=0",
        "source_angular_mode_type": "ell_src",
        "observed_multipole_type": "L_CMB",
        "observed_cmb_multipole_is_solver_output": True,
        "eigenvalue_intervals": [[1.0, 1.0], [4.0, 4.1]],
        "k_intervals_Mpc_inverse": [[0.01, 0.011], [0.02, 0.021]],
        "safe_resolved_band": [0.01, 0.021],
        "proper_time_table": [[0.0, 0.0], [1.0, 1.0]],
        "a_volume_table": [[0.0, 0.5], [1.0, 1.0]],
        "a_expansion_table": [[0.0, 0.5], [1.0, 1.0]],
        "redshift_table": [[0.0, 1.0], [1.0, 0.0]],
        "conformal_time_table": [[0.0, 0.0], [1.0, 1.5]],
        "a_consistency_error": 0.0,
        "flrw_residual": 0.0,
        "common_surface_passed": True,
        "refinement_errors": {
            "uv_error": 0.0,
            "ir_error": 0.0,
            "time_error": 0.0,
            "sampling_error": 0.0,
            "boundary_error": 0.0,
        },
    }


def _valid_hash(value: Any) -> bool:
    return bool(isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-fA-F]{64}", value.strip()))


def _fixture_hash(seed: str, label: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(f"{seed}:{label}".encode("utf-8")).hexdigest()


def _valid_intervals(value: Any, *, positive: bool = False) -> bool:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    if np.any(arr[:, 1] < arr[:, 0]):
        return False
    if positive and np.any(arr <= 0.0):
        return False
    return True


def _valid_band(value: Any) -> bool:
    try:
        lo, hi = value
        lo_f = float(lo)
        hi_f = float(hi)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(lo_f) and np.isfinite(hi_f) and 0.0 < lo_f <= hi_f)


def _valid_refinement(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    required = ("uv_error", "ir_error", "time_error", "sampling_error", "boundary_error")
    return all(_finite_nonnegative(value.get(name)) for name in required)


def _finite_nonnegative(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(parsed) and parsed >= 0.0)


def _monotone_table(value: Any, *, decreasing: bool = False) -> bool:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2 or not np.all(np.isfinite(arr)):
        return False
    y = arr[:, 1]
    diffs = np.diff(y)
    return bool(np.all(diffs <= 1.0e-12) if decreasing else np.all(diffs >= -1.0e-12))


def _same_hashes(payload: dict[str, Any]) -> bool:
    expected = {
        "geometry_hash": payload.get("geometry_hash"),
        "background_hash": payload.get("background_hash"),
        "clock_hash": payload.get("clock_hash"),
        "mode_basis_id": payload.get("mode_basis_id") or payload.get("mode_basis_hash"),
        "source_embedding_hash": payload.get("source_embedding_hash"),
        "regulator_family_id": payload.get("regulator_family_id"),
        "solver_convention_hash": payload.get("solver_convention_hash"),
    }
    for receipt in payload.get("cross_receipts") or []:
        if not isinstance(receipt, dict):
            return False
        for key, expected_value in expected.items():
            if expected_value is not None and receipt.get(key) not in {None, expected_value}:
                return False
    return True
