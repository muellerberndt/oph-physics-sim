from __future__ import annotations

import cmath
import math
from typing import Any, Sequence

import numpy as np


def _as_coefficients(values: Sequence[Sequence[float] | complex | float]) -> np.ndarray:
    decoded = []
    for value in values:
        if isinstance(value, complex):
            decoded.append(value)
        elif isinstance(value, (float, int)):
            decoded.append(complex(value))
        else:
            decoded.append(complex(float(value[0]), float(value[1])))
    return np.asarray(decoded, dtype=np.complex128)


def _evaluate(coefficients: np.ndarray, value: np.ndarray | complex) -> np.ndarray | complex:
    return np.polynomial.polynomial.polyval(value, coefficients)


def pole_enclosure_receipt(
    determinant_coefficients: Sequence[Sequence[float] | complex | float],
    *,
    reference_coefficients: Sequence[Sequence[float] | complex | float] | None = None,
    contour_center: Sequence[float] = (0.0, 0.0),
    contour_radius: float,
    contour_samples: int = 2048,
    physical_sheet_verified: bool = False,
    nonzero_residue_verified: bool = False,
    uncertainty_bound_present: bool = False,
    source_block_receipt: bool = False,
) -> dict[str, Any]:
    """Count a polynomial determinant zero inside a circular Rouché contour."""

    full = _as_coefficients(determinant_coefficients)
    reference_defaulted_to_full = reference_coefficients is None
    reference = (
        full.copy()
        if reference_defaulted_to_full
        else _as_coefficients(reference_coefficients)
    )
    if contour_radius <= 0.0 or contour_samples < 32:
        raise ValueError("pole contour requires positive radius and at least 32 samples")
    center = complex(float(contour_center[0]), float(contour_center[1]))
    angles = np.linspace(0.0, 2.0 * math.pi, contour_samples, endpoint=False)
    contour = center + contour_radius * np.exp(1j * angles)
    ref_values = _evaluate(reference, contour)
    delta_values = _evaluate(full, contour) - ref_values
    minimum_reference = float(np.min(np.abs(ref_values)))
    maximum_delta = float(np.max(np.abs(delta_values)))
    rouche_margin = minimum_reference - maximum_delta
    rouche_pass = rouche_margin > 0.0

    roots = np.polynomial.polynomial.polyroots(full)
    reference_roots = np.polynomial.polynomial.polyroots(reference)
    inside = [root for root in roots if abs(root - center) < contour_radius]
    reference_inside = [root for root in reference_roots if abs(root - center) < contour_radius]
    unique_zero = len(inside) == 1 and len(reference_inside) == 1 and rouche_pass
    selected = inside[0] if len(inside) == 1 else None
    derivative = np.polynomial.polynomial.polyder(full)
    derivative_at_root = abs(_evaluate(derivative, selected)) if selected is not None else 0.0
    simple_zero = bool(selected is not None and derivative_at_root > 1.0e-12)

    mass = width = None
    lower_half_plane = False
    if selected is not None:
        square_root = cmath.sqrt(selected)
        if square_root.real < 0.0:
            square_root = -square_root
        lower_half_plane = square_root.imag <= 0.0
        mass = float(square_root.real)
        width = float(-2.0 * square_root.imag)

    declared_candidate = bool(
        source_block_receipt is True
        and physical_sheet_verified is True
        and nonzero_residue_verified is True
        and uncertainty_bound_present is True
        and unique_zero
        and simple_zero
        and lower_half_plane
    )
    blockers = []
    if not source_block_receipt:
        blockers.append("source_brst_block_receipt_missing")
    if not physical_sheet_verified:
        blockers.append("physical_riemann_sheet_not_verified")
    if not nonzero_residue_verified:
        blockers.append("physical_residue_not_verified")
    if not uncertainty_bound_present:
        blockers.append("pole_uncertainty_bound_missing")
    if not unique_zero:
        blockers.append("unique_rouche_zero_not_verified")
    if not simple_zero:
        blockers.append("simple_pole_not_verified")
    if selected is not None and not lower_half_plane:
        blockers.append("selected_square_root_not_on_physical_lower_half_plane")
    if reference_defaulted_to_full:
        blockers.append("forbidden_reference_equals_full_kernel_fallback")
    blockers.extend(
        [
            "polynomial_kernel_is_wzh0_synthetic_control",
            "sampled_contour_has_no_continuous_interval_bound",
            "matrix_laurent_and_physical_current_amplitude_not_verified",
            "artifact_resolving_pole_checker_not_implemented",
        ]
    )

    return {
        "schema": "oph_wzh_pole_enclosure_receipt_v1",
        "contour": {
            "center": [center.real, center.imag],
            "radius": float(contour_radius),
            "samples": int(contour_samples),
        },
        "minimum_reference_modulus": minimum_reference,
        "maximum_perturbation_modulus": maximum_delta,
        "rouche_margin": rouche_margin,
        "rouche_pass": rouche_pass,
        "reference_defaulted_to_full_kernel": reference_defaulted_to_full,
        "zero_count": len(inside),
        "reference_zero_count": len(reference_inside),
        "unique_simple_zero": bool(unique_zero and simple_zero),
        "selected_pole_s": None if selected is None else [float(selected.real), float(selected.imag)],
        "determinant_derivative_modulus": float(derivative_at_root),
        "mass_coordinate": mass,
        "width_coordinate": width,
        "mass_convention": "s_B=(M_B-i*Gamma_B/2)^2",
        "numerical_zero_control_receipt": bool(unique_zero and simple_zero),
        "declared_candidate_conditions_met": declared_candidate,
        "physical_pole_receipt": False,
        "promotion_allowed": False,
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "This polynomial/sampled-contour result is an unconditionally "
            "nonpromoting WZH0 numerical control. A physical W/Z pole requires an "
            "analytic complex-ball kernel, continuous contour proof, matrix Laurent "
            "data, and the same pole in a BRST-invariant current amplitude."
        ),
    }
