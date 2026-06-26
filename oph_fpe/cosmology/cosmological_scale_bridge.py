from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import numpy as np

from oph_fpe.cosmology.claim_tiers import ClaimTier, GeometryOrigin, normalize_claim_tier, normalize_geometry_origin


HASH_FIELDS = (
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
    "mode_normalization_hash",
    "angular_operator_hash",
    "radial_kernel_hash",
    "density_of_states_hash",
    "tolerance_ledger_hash",
    "freezeout_surface_hash",
    "source_dag_hash",
    "no_peak_fit_receipt_hash",
    "surface_mesh_hash",
    "state_vector_hash",
    "normal_derivative_hash",
)

CROSS_RECEIPT_FIELDS = (
    "regulator_family_id",
    "generation_id",
    "geometry_hash",
    "background_hash",
    "clock_hash",
    "scale_certificate_hash",
    "source_embedding_hash",
    "mode_basis_hash",
    "mode_lineage_hash",
    "boundary_condition_hash",
    "solver_convention_hash",
    "freezeout_surface_hash",
    "source_dag_hash",
)


def validate_physical_scale_bridge_receipts(receipts: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the #372 physical scale-bridge receipt bundle.

    Gate status is recomputed from primitive residuals, hashes, refinement
    evidence, and cross-receipt identity fields. Bare success booleans in the
    producer payload are treated as producer assertions only.
    """

    payload = receipts if isinstance(receipts, dict) else {}
    tier = normalize_claim_tier(payload.get("claim_tier"))
    origin = normalize_geometry_origin(payload.get("geometry_origin"))
    valid_physical_origin = (
        (tier == ClaimTier.CONDITIONAL_PHYSICAL and origin == GeometryOrigin.IMPORTED_FLRW)
        or (tier == ClaimTier.OPH_NATIVE_PHYSICAL and origin == GeometryOrigin.OPH_NATIVE)
    )
    hash_checks = {name: _valid_hash(payload.get(name)) for name in HASH_FIELDS}

    nofit = bool(payload.get("NO_POSTHOC_CALIBRATION_RECEIPT")) and hash_checks["no_peak_fit_receipt_hash"]
    no_data_dag = bool(payload.get("NO_DATA_USE_DAG_RECEIPT")) and hash_checks["source_dag_hash"]
    refinement = _valid_refinement(payload.get("refinement_errors")) and hash_checks["tolerance_ledger_hash"]
    cross_consistency = _same_hashes(payload)

    geometry = bool(payload.get("FINITE_COSMOLOGICAL_GEOMETRY_RECEIPT")) and hash_checks["geometry_hash"]
    scale = bool(payload.get("DIMENSIONAL_SCALE_RECEIPT")) and hash_checks["scale_certificate_hash"]
    flrw = bool(payload.get("FLRW_BACKGROUND_REDUCTION_RECEIPT")) and hash_checks["background_hash"]
    operator = bool(payload.get("PHYSICAL_MODE_OPERATOR_RECEIPT")) and hash_checks["operator_hash"]
    boundary = bool(payload.get("BOUNDARY_TOPOLOGY_SECTOR_RECEIPT")) and hash_checks["boundary_condition_hash"]
    solver_convention = bool(payload.get("SOLVER_CONVENTION_RECEIPT")) and hash_checks["solver_convention_hash"]
    mode_normalization = bool(payload.get("PHYSICAL_MODE_NORMALIZATION_RECEIPT")) and hash_checks[
        "mode_normalization_hash"
    ]
    mode_lineage = bool(payload.get("MODE_LINEAGE_RECEIPT")) and hash_checks["mode_lineage_hash"]
    density_of_states = bool(payload.get("DENSITY_OF_STATES_RECEIPT")) and hash_checks["density_of_states_hash"]
    quadrature = bool(payload.get("SPECTRAL_QUADRATURE_RECEIPT"))
    projector_invariance = bool(payload.get("DEGENERATE_PROJECTOR_INVARIANCE_RECEIPT"))
    physical_spatial_k = bool(
        valid_physical_origin
        and geometry
        and scale
        and flrw
        and operator
        and boundary
        and solver_convention
        and mode_normalization
        and mode_lineage
        and density_of_states
        and quadrature
        and projector_invariance
        and refinement
        and nofit
        and _valid_intervals(payload.get("eigenvalue_intervals"))
        and _valid_intervals(payload.get("k_intervals_Mpc_inverse"), positive=True)
        and _valid_band(payload.get("safe_resolved_band"))
        and _scalar_below(payload.get("orthogonality_residual"), payload.get("orthogonality_tolerance"))
        and _residuals_below(payload.get("eigenpair_residuals"), payload.get("eigenpair_tolerance"))
        and _residuals_below(payload.get("k_equation_residuals"), payload.get("k_equation_tolerance"))
        and payload.get("a0_convention") == "a0=1"
        and str(payload.get("k_unit_convention")) == "Mpc^-1"
    )

    source_embedding = bool(payload.get("SOURCE_SCREEN_EMBEDDING_RECEIPT")) and hash_checks[
        "source_embedding_hash"
    ]
    screen_to_k = bool(
        physical_spatial_k
        and source_embedding
        and bool(payload.get("RADIAL_WINDOW_RECEIPT"))
        and bool(payload.get("RADIAL_KERNEL_RECEIPT"))
        and hash_checks["radial_kernel_hash"]
        and bool(payload.get("RADIAL_KERNEL_GEOMETRY_COMPATIBILITY_RECEIPT"))
        and bool(payload.get("RADIAL_NULL_SPACE_REPORT_RECEIPT"))
        and bool(payload.get("FORWARD_PROJECTION_RESIDUAL_RECEIPT"))
        and _residuals_below(payload.get("forward_projection_residuals"), payload.get("forward_projection_tolerance"))
    )

    angular_screen_ok = _angular_screen_ok(payload)
    source_angular = bool(
        valid_physical_origin
        and angular_screen_ok
        and bool(payload.get("ANGULAR_OPERATOR_RECEIPT"))
        and bool(payload.get("ANGULAR_CLUSTER_RECEIPT"))
        and bool(payload.get("ANGULAR_MULTIPLICITY_RECEIPT"))
        and hash_checks["angular_operator_hash"]
        and _residuals_below(payload.get("angular_sector_residuals"), payload.get("angular_sector_tolerance"))
        and _residuals_below(payload.get("angular_refinement_errors"), payload.get("angular_refinement_tolerance"))
        and str(payload.get("source_angular_mode_type")) == "ell_src"
        and bool(payload.get("observed_cmb_multipole_is_solver_output"))
        and payload.get("source_angular_mode_type") != payload.get("observed_multipole_type")
    )

    flrw_residuals = _valid_flrw_residuals(payload.get("flrw_residuals"))
    a_evolution = bool(
        valid_physical_origin
        and bool(payload.get("PROPER_TIME_CLOCK_RECEIPT"))
        and bool(payload.get("LINEAGE_RECEIPT"))
        and flrw
        and bool(payload.get("VOLUME_SCALE_FACTOR_RECEIPT"))
        and bool(payload.get("EXPANSION_SCALE_FACTOR_RECEIPT"))
        and bool(payload.get("FLRW_CONSTRAINT_RECEIPT"))
        and _finite_nonnegative(payload.get("a_consistency_error"))
        and flrw_residuals
        and _monotone_table(payload.get("proper_time_table"))
        and _monotone_table(payload.get("a_volume_table"))
        and _monotone_table(payload.get("a_expansion_table"))
        and _monotone_table(payload.get("redshift_table"), decreasing=True)
        and _monotone_table(payload.get("conformal_time_table"))
        and refinement
        and nofit
    )

    mode_freezeout = bool(
        valid_physical_origin
        and bool(payload.get("CURVATURE_RESIDUAL_RECEIPT"))
        and bool(payload.get("ADIABATIC_MODE_RECEIPT"))
        and bool(payload.get("ISOCURVATURE_BOUND_RECEIPT"))
        and bool(payload.get("GROWING_MODE_RECEIPT"))
        and bool(payload.get("REPAIR_FORCING_RESIDUAL_RECEIPT"))
        and bool(payload.get("PHASE_COHERENCE_RECEIPT"))
        and bool(payload.get("CONSTRAINT_RESIDUAL_RECEIPT"))
        and bool(payload.get("FREEZEOUT_PERSISTENCE_RECEIPT"))
        and _residuals_below(payload.get("mode_freezeout_residuals"), payload.get("mode_freezeout_tolerance"))
    )
    common_surface = bool(
        mode_freezeout
        and bool(payload.get("SPACELIKE_FREEZEOUT_SLICE_RECEIPT"))
        and bool(payload.get("FREEZEOUT_INITIAL_DATA_RECEIPT"))
        and bool(payload.get("common_surface_passed"))
        and bool(payload.get("COINCIDENT_FREEZEOUT_RECEIPT") or payload.get("TRANSPORT_TO_COMMON_SLICE_RECEIPT"))
        and hash_checks["freezeout_surface_hash"]
        and hash_checks["surface_mesh_hash"]
        and hash_checks["state_vector_hash"]
        and hash_checks["normal_derivative_hash"]
        and refinement
    )
    common_basis = bool(
        bool(payload.get("COMMON_PRIMORDIAL_ANOMALY_MODE_BASIS_RECEIPT"))
        and bool(payload.get("RESPONSE_PROJECTOR_REDUCTION_RECEIPT"))
        and projector_invariance
        and hash_checks["mode_basis_hash"]
    )
    aggregate = bool(
        physical_spatial_k
        and screen_to_k
        and source_angular
        and a_evolution
        and mode_freezeout
        and common_surface
        and common_basis
        and refinement
        and cross_consistency
        and no_data_dag
        and nofit
    )
    gates = {
        "PHYSICAL_SPATIAL_K_RECEIPT": physical_spatial_k,
        "PHYSICAL_K_RECEIPT": physical_spatial_k,
        "SCREEN_TO_PHYSICAL_K_ASSOCIATION_RECEIPT": screen_to_k,
        "SOURCE_ANGULAR_SECTOR_RECEIPT": source_angular,
        "SOURCE_ANGULAR_MODE_RECEIPT": source_angular,
        "CALIBRATED_A_EVOLUTION_RECEIPT": a_evolution,
        "PHYSICAL_MODE_FREEZEOUT_MAP_RECEIPT": mode_freezeout,
        "PHYSICAL_FREEZEOUT_SURFACE_RECEIPT": common_surface,
        "COMMON_PRIMORDIAL_ANOMALY_MODE_BASIS_RECEIPT": common_basis,
        "SCALE_BRIDGE_REFINEMENT_RECEIPT": refinement,
        "CROSS_RECEIPT_CONSISTENCY_RECEIPT": cross_consistency,
        "NO_DATA_USE_DAG_RECEIPT": no_data_dag,
        "NO_POSTHOC_CALIBRATION_RECEIPT": nofit,
        "PHYSICAL_SCALE_BRIDGE_RECEIPT": aggregate,
    }
    blockers = [name.lower() + "_missing" for name, passed in gates.items() if not passed]
    if tier == ClaimTier.OPH_NATIVE_PHYSICAL and origin != GeometryOrigin.OPH_NATIVE:
        blockers.append("oph_native_geometry_origin_missing")
    if tier == ClaimTier.CONDITIONAL_PHYSICAL and origin != GeometryOrigin.IMPORTED_FLRW:
        blockers.append("conditional_imported_flrw_origin_missing")
    if not valid_physical_origin:
        blockers.append("claim_tier_geometry_origin_mismatch")
    for name, passed in hash_checks.items():
        if not passed:
            blockers.append(f"{name}_missing")
    if payload.get("mode_dependent_freezeout_map") and not common_surface:
        blockers.append("mode_dependent_freezeout_is_not_common_surface")
    return {
        **gates,
        "physical_k_units_calibrated": physical_spatial_k,
        "screen_to_physical_k_association_calibrated": screen_to_k,
        "source_angular_sector_calibrated": source_angular,
        "calibrated_a_evolution": a_evolution,
        "physical_mode_freezeout_map_calibrated": mode_freezeout,
        "common_primordial_initial_surface_calibrated": common_surface,
        "claim_tier": tier.value,
        "geometry_origin": origin.value,
        "oph_native_k_derivation": bool(physical_spatial_k and origin == GeometryOrigin.OPH_NATIVE),
        "hash_checks": hash_checks,
        "cross_consistency": cross_consistency,
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "Structured physical scale-bridge validation. Conditional physical receipts may import "
            "frozen FLRW geometry; OPH-native physical receipts require OPH_NATIVE geometry origin. "
            "A mode-dependent freezeout map is separate from a common primordial initial surface."
        ),
    }


def imported_flrw_reference_receipts(*, hash_seed: str = "0") -> dict[str, Any]:
    """Small deterministic valid imported-geometry receipt fixture for tests."""

    hashes = {name: _fixture_hash(hash_seed, name) for name in HASH_FIELDS}
    ids = {
        "regulator_family_id": f"regulator:{hash_seed}",
        "generation_id": f"generation:{hash_seed}",
    }
    payload: dict[str, Any] = {
        "claim_tier": ClaimTier.CONDITIONAL_PHYSICAL.value,
        "geometry_origin": GeometryOrigin.IMPORTED_FLRW.value,
        **ids,
        **hashes,
        "FINITE_COSMOLOGICAL_GEOMETRY_RECEIPT": True,
        "SOURCE_SCREEN_EMBEDDING_RECEIPT": True,
        "DIMENSIONAL_SCALE_RECEIPT": True,
        "FLRW_BACKGROUND_REDUCTION_RECEIPT": True,
        "PHYSICAL_MODE_OPERATOR_RECEIPT": True,
        "BOUNDARY_TOPOLOGY_SECTOR_RECEIPT": True,
        "SOLVER_CONVENTION_RECEIPT": True,
        "PHYSICAL_MODE_NORMALIZATION_RECEIPT": True,
        "MODE_LINEAGE_RECEIPT": True,
        "DENSITY_OF_STATES_RECEIPT": True,
        "SPECTRAL_QUADRATURE_RECEIPT": True,
        "DEGENERATE_PROJECTOR_INVARIANCE_RECEIPT": True,
        "NO_POSTHOC_CALIBRATION_RECEIPT": True,
        "NO_DATA_USE_DAG_RECEIPT": True,
        "RADIAL_WINDOW_RECEIPT": True,
        "RADIAL_KERNEL_RECEIPT": True,
        "RADIAL_KERNEL_GEOMETRY_COMPATIBILITY_RECEIPT": True,
        "RADIAL_NULL_SPACE_REPORT_RECEIPT": True,
        "FORWARD_PROJECTION_RESIDUAL_RECEIPT": True,
        "source_screen_type": "closed_sphere",
        "CLOSED_SPHERE_OR_EXTENSION_RECEIPT": True,
        "ANGULAR_OPERATOR_RECEIPT": True,
        "ANGULAR_CLUSTER_RECEIPT": True,
        "ANGULAR_MULTIPLICITY_RECEIPT": True,
        "PROPER_TIME_CLOCK_RECEIPT": True,
        "LINEAGE_RECEIPT": True,
        "VOLUME_SCALE_FACTOR_RECEIPT": True,
        "EXPANSION_SCALE_FACTOR_RECEIPT": True,
        "FLRW_CONSTRAINT_RECEIPT": True,
        "CURVATURE_RESIDUAL_RECEIPT": True,
        "ADIABATIC_MODE_RECEIPT": True,
        "ISOCURVATURE_BOUND_RECEIPT": True,
        "GROWING_MODE_RECEIPT": True,
        "REPAIR_FORCING_RESIDUAL_RECEIPT": True,
        "PHASE_COHERENCE_RECEIPT": True,
        "CONSTRAINT_RESIDUAL_RECEIPT": True,
        "FREEZEOUT_PERSISTENCE_RECEIPT": True,
        "SPACELIKE_FREEZEOUT_SLICE_RECEIPT": True,
        "FREEZEOUT_INITIAL_DATA_RECEIPT": True,
        "COINCIDENT_FREEZEOUT_RECEIPT": True,
        "COMMON_PRIMORDIAL_ANOMALY_MODE_BASIS_RECEIPT": True,
        "RESPONSE_PROJECTOR_REDUCTION_RECEIPT": True,
        "a0_convention": "a0=1",
        "k_unit_convention": "Mpc^-1",
        "curvature_convention": "K_FLRW=0",
        "source_angular_mode_type": "ell_src",
        "observed_multipole_type": "L_CMB",
        "observed_cmb_multipole_is_solver_output": True,
        "eigenvalue_intervals": [[1.0, 1.0], [4.0, 4.1]],
        "k_intervals_Mpc_inverse": [[0.01, 0.011], [0.02, 0.021]],
        "safe_resolved_band": [0.01, 0.021],
        "orthogonality_residual": 0.0,
        "orthogonality_tolerance": 1.0e-9,
        "eigenpair_residuals": [0.0, 0.0],
        "eigenpair_tolerance": 1.0e-9,
        "k_equation_residuals": [0.0, 0.0],
        "k_equation_tolerance": 1.0e-9,
        "forward_projection_residuals": [0.0, 0.0],
        "forward_projection_tolerance": 1.0e-9,
        "angular_sector_residuals": [0.0, 0.0],
        "angular_sector_tolerance": 1.0e-9,
        "angular_refinement_errors": [1.0e-5, 5.0e-6],
        "angular_refinement_tolerance": 1.0e-4,
        "proper_time_table": [[0.0, 0.0], [1.0, 1.0]],
        "a_volume_table": [[0.0, 0.5], [1.0, 1.0]],
        "a_expansion_table": [[0.0, 0.5], [1.0, 1.0]],
        "redshift_table": [[0.0, 1.0], [1.0, 0.0]],
        "conformal_time_table": [[0.0, 0.0], [1.0, 1.5]],
        "a_consistency_error": 0.0,
        "flrw_residuals": {
            "shear_over_H": 0.0,
            "vorticity_over_H": 0.0,
            "acceleration_over_H": 0.0,
            "curvature_gradient_over_H3": 0.0,
            "backreaction_over_H2": 0.0,
            "hamiltonian": 0.0,
            "momentum": 0.0,
            "energy": 0.0,
        },
        "mode_freezeout_residuals": [0.0, 0.0],
        "mode_freezeout_tolerance": 1.0e-9,
        "common_surface_passed": True,
        "refinement_errors": {
            "spectral_projector_errors": [1.0e-4, 5.0e-5],
            "spectral_measure_errors": [1.0e-4, 4.0e-5],
            "background_errors": [1.0e-4, 4.0e-5],
            "freezeout_surface_errors": [1.0e-4, 4.0e-5],
            "multi_limit_path_errors": [1.0e-4, 4.0e-5],
            "multi_limit_path_independence_receipt": True,
        },
    }
    payload["cross_receipts"] = [_cross_receipt_row(payload, name) for name in ("spatial_k", "screen_to_k", "freezeout")]
    return payload


def _valid_hash(value: Any) -> bool:
    return bool(isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-fA-F]{64}", value.strip()))


def _fixture_hash(seed: str, label: str) -> str:
    return "sha256:" + hashlib.sha256(f"{seed}:{label}".encode("utf-8")).hexdigest()


def _cross_receipt_row(payload: dict[str, Any], receipt_name: str) -> dict[str, Any]:
    return {"receipt": receipt_name, **{field: payload.get(field) for field in CROSS_RECEIPT_FIELDS}}


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
    required = (
        "spectral_projector_errors",
        "spectral_measure_errors",
        "background_errors",
        "freezeout_surface_errors",
        "multi_limit_path_errors",
    )
    return bool(
        all(_decreasing_nonnegative_sequence(value.get(name)) for name in required)
        and bool(value.get("multi_limit_path_independence_receipt"))
    )


def _valid_flrw_residuals(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    required = (
        "shear_over_H",
        "vorticity_over_H",
        "acceleration_over_H",
        "curvature_gradient_over_H3",
        "backreaction_over_H2",
        "hamiltonian",
        "momentum",
        "energy",
    )
    return all(_finite_nonnegative(value.get(name)) for name in required)


def _angular_screen_ok(payload: dict[str, Any]) -> bool:
    screen_type = str(payload.get("source_screen_type") or "closed_sphere")
    if bool(payload.get("CLOSED_SPHERE_OR_EXTENSION_RECEIPT")) and screen_type == "closed_sphere":
        return True
    if screen_type in {"cap", "masked", "partial"}:
        return bool(payload.get("FULL_SCREEN_EXTENSION_RECEIPT") or payload.get("MODE_COUPLING_OPERATOR_RECEIPT"))
    return bool(payload.get("CLOSED_SPHERE_OR_EXTENSION_RECEIPT"))


def _residuals_below(value: Any, tolerance: Any) -> bool:
    try:
        tol = float(tolerance)
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    if arr.size == 0 or not math.isfinite(tol) or tol < 0.0 or not np.all(np.isfinite(arr)):
        return False
    return bool(np.max(np.abs(arr)) <= tol)


def _scalar_below(value: Any, tolerance: Any) -> bool:
    try:
        parsed = float(value)
        tol = float(tolerance)
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(parsed) and math.isfinite(tol) and tol >= 0.0 and abs(parsed) <= tol)


def _decreasing_nonnegative_sequence(value: Any) -> bool:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    if arr.ndim != 1 or arr.size < 2 or not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        return False
    return bool(arr[-1] <= arr[0])


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
    receipts = payload.get("cross_receipts")
    if not isinstance(receipts, list) or not receipts:
        return False
    expected = {field: payload.get(field) for field in CROSS_RECEIPT_FIELDS}
    if any(value in {None, ""} for value in expected.values()):
        return False
    for receipt in receipts:
        if not isinstance(receipt, dict):
            return False
        for key, expected_value in expected.items():
            if receipt.get(key) != expected_value:
                return False
    return True
