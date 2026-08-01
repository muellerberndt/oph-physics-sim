"""Conditional continuum-tail certificate for the equal-seam A5 operator.

This module does not select the equal-seam law.  It asks a narrower
mathematical question: if the unweighted seam form is extended through the
registered geodesic-midpoint tower, does its frozen ``(2,4,6)`` BipoSH
fingerprint disappear under refinement?

The certificate has three deliberately separate layers.

* A rational-arithmetic check proves the one-triangle refinement identity.
* Addition-theorem bounds and the spherical midpoint contraction prove that
  the relevant finite-dimensional form blocks are Cauchy.
* A face-streamed level-nine calculation supplies a finite anchor without
  constructing the multi-million-vertex global mesh.

The finite anchor uses binary64 polynomial spherical harmonics.  Its declared
roundoff envelope is checked against an independent SciPy implementation, but
the elementary-function envelope is still a named numerical premise.  The
receipt therefore does not promote the conditional nonzero result to a source
theorem or a physical covariance prediction.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower
from oph_fpe.cosmology.a5_biposh_refinement import (
    _biposh_rows,
    _biposh_summary,
    _clebsch_gordan,
    _equal_seam_graph_stiffness,
    _harmonic_design,
    _vertex_area_weights,
    _weighted_low_mode_removal,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / "data/refinement/a5_biposh_continuum_tail_receipt.json"
FINITE_PARENT = ROOT / "data/refinement/a5_biposh_dual_operator_receipt.json"

SCHEMA = "oph.a5-biposh-continuum-tail.v1"
STATUS = (
    "CONDITIONAL_EQUAL_SEAM_CONTINUUM_L6_NONZERO_UNDER_DECLARED_NUMERICAL_"
    "ENVELOPE__SOURCE_SELECTION_AND_PHYSICAL_TRANSFER_OPEN"
)
ANCHOR_LEVEL = 9
TAIL_EXACT_STOP = 48
ANCHOR_NUMERICAL_RADIUS = 0.01
HARMONIC_VALUE_ERROR_PREMISE = 1.0e-10
TAIL_ROUNDOFF_INFLATION = 1.05
TARGET_LEVELS = (6, 7, 8, 9)


class ContinuumTailError(ValueError):
    """Raised when the conditional tail packet fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_pin(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(payload),
        "sha256": _sha(payload),
    }


def _local_energy(values_left: tuple, values_right: tuple) -> Any:
    """Half-edge energy on one triangle, valid for exact scalar values."""

    a, b, c = values_left
    x, y, z = values_right
    return Fraction(1, 2) * (
        (a - b) * (x - y) + (b - c) * (y - z) + (c - a) * (z - x)
    )


def _direct_refinement_delta(
    coarse_left: tuple,
    defect_left: tuple,
    coarse_right: tuple,
    defect_right: tuple,
) -> Any:
    """Direct four-child energy minus the parent energy."""

    a, b, c = coarse_left
    u, v, w = defect_left
    A, B, C = coarse_right
    U, V, W = defect_right
    x, y, z = (a + b) / 2 + u, (b + c) / 2 + v, (c + a) / 2 + w
    X, Y, Z = (A + B) / 2 + U, (B + C) / 2 + V, (C + A) / 2 + W
    fine = sum(
        (
            _local_energy(left, right)
            for left, right in (
                ((a, x, z), (A, X, Z)),
                ((b, y, x), (B, Y, X)),
                ((c, z, y), (C, Z, Y)),
                ((x, y, z), (X, Y, Z)),
            )
        ),
        Fraction(0),
    )
    return fine - _local_energy(coarse_left, coarse_right)


def _factored_refinement_delta(
    coarse_left: tuple,
    defect_left: tuple,
    coarse_right: tuple,
    defect_right: tuple,
) -> Any:
    """Factored refinement identity used by the analytic tail estimate."""

    a, b, c = coarse_left
    u, v, w = defect_left
    A, B, C = coarse_right
    U, V, W = defect_right
    return (
        u * (A / 2 + B / 2 - C + 3 * U - V - W)
        + v * (-A + B / 2 + C / 2 - U + 3 * V - W)
        + w * (A / 2 - B + C / 2 - U - V + 3 * W)
        + U * (a / 2 + b / 2 - c)
        + V * (-a + b / 2 + c / 2)
        + W * (a / 2 - b + c / 2)
    )


def exact_refinement_identity_report() -> dict[str, Any]:
    """Verify the bilinear identity on the 36 coefficient basis pairs."""

    zero = Fraction(0)
    cases = 0
    for left_index in range(6):
        for right_index in range(6):
            left = [zero] * 6
            right = [zero] * 6
            left[left_index] = Fraction(1)
            right[right_index] = Fraction(1)
            direct = _direct_refinement_delta(
                tuple(left[:3]), tuple(left[3:]), tuple(right[:3]), tuple(right[3:])
            )
            factored = _factored_refinement_delta(
                tuple(left[:3]), tuple(left[3:]), tuple(right[:3]), tuple(right[3:])
            )
            if direct != factored:
                raise ContinuumTailError(
                    f"refinement identity failed at ({left_index}, {right_index})"
                )
            cases += 1
    return {
        "arithmetic": "fractions.Fraction over Q",
        "coefficient_basis_cases": cases,
        "identity": (
            "four-child half-edge bilinear energy minus parent half-edge energy "
            "equals the declared vertex/midpoint-defect factorization"
        ),
        "identity_verified": cases == 36,
        "bound_consequence": (
            "norm(delta_T)<=3(M_l D_lprime+D_l M_lprime)+15 M_l M_lprime"
        ),
    }


def _refine_local_face(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Red-refine one closed base-face patch without global vertex storage."""

    old = np.asarray(vertices, dtype=float)
    output_vertices = [row.copy() for row in old]
    midpoint_ids: dict[tuple[int, int], int] = {}

    def midpoint(left: int, right: int) -> int:
        key = (min(left, right), max(left, right))
        if key not in midpoint_ids:
            point = old[key[0]] + old[key[1]]
            point /= np.linalg.norm(point)
            midpoint_ids[key] = len(output_vertices)
            output_vertices.append(point)
        return midpoint_ids[key]

    output_faces: list[tuple[int, int, int]] = []
    for a_value, b_value, c_value in faces:
        a, b, c = int(a_value), int(b_value), int(c_value)
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        output_faces.extend(
            ((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))
        )
    return np.asarray(output_vertices, dtype=float), np.asarray(
        output_faces, dtype=np.int64
    )


def _solid_harmonics(points: np.ndarray, ell: int) -> np.ndarray:
    """Polynomial Condon--Shortley harmonics for ell two or four."""

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    q = x + 1.0j * y
    root_pi = math.sqrt(math.pi)
    if ell == 2:
        positive = [
            math.sqrt(5.0) * (3.0 * z**2 - 1.0) / (4.0 * root_pi),
            -math.sqrt(30.0) * z * q / (4.0 * root_pi),
            math.sqrt(30.0) * q**2 / (8.0 * root_pi),
        ]
    elif ell == 4:
        positive = [
            3.0 * (35.0 * z**4 - 30.0 * z**2 + 3.0) / (16.0 * root_pi),
            -3.0
            * math.sqrt(5.0)
            * (7.0 * z**3 - 3.0 * z)
            * q
            / (8.0 * root_pi),
            3.0
            * math.sqrt(10.0)
            * (7.0 * z**2 - 1.0)
            * q**2
            / (16.0 * root_pi),
            -3.0 * math.sqrt(35.0) * z * q**3 / (8.0 * root_pi),
            3.0 * math.sqrt(70.0) * q**4 / (32.0 * root_pi),
        ]
    else:
        raise ValueError("only ell=2 and ell=4 are needed by the frozen primary")
    negative = [(-1) ** order * np.conjugate(positive[order]) for order in range(1, ell + 1)]
    return np.column_stack([*reversed(negative), *positive])


def _accumulate_triangle_form(
    left: np.ndarray,
    right: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    result = np.zeros((left.shape[1], right.shape[1]), dtype=np.complex128)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        left_difference = left[faces[:, first]] - left[faces[:, second]]
        right_difference = right[faces[:, first]] - right[faces[:, second]]
        result += 0.5 * left_difference.conj().T @ right_difference
    return result


def _targeted_biposh_summary(
    form22: np.ndarray,
    form24: np.ndarray,
    form44: np.ndarray,
) -> dict[str, Any]:
    def coefficient(
        form: np.ndarray,
        ell: int,
        ell_prime: int,
        total_l: int,
        total_m: int,
    ) -> complex:
        value = 0.0j
        for m in range(-ell, ell + 1):
            m_prime = m - total_m
            if -ell_prime <= m_prime <= ell_prime:
                value += (
                    (-1) ** m_prime
                    * _clebsch_gordan(
                        ell, m, ell_prime, -m_prime, total_l, total_m
                    )
                    * form[m + ell, m_prime + ell_prime]
                )
        return value

    vector = np.asarray(
        [coefficient(form24, 2, 4, 6, total_m) for total_m in range(-6, 7)]
    )
    a22 = coefficient(form22, 2, 2, 0, 0)
    a44 = coefficient(form44, 4, 4, 0, 0)
    numerator = float(np.linalg.norm(vector))
    denominator = math.sqrt(abs(a22 * a44))
    return {
        "A_22_00": [float(a22.real), float(a22.imag)],
        "A_44_00": [float(a44.real), float(a44.imag)],
        "A_24_6M": [[float(value.real), float(value.imag)] for value in vector],
        "primary_numerator_norm": numerator,
        "primary_denominator": denominator,
        "primary_amplitude_free_statistic": numerator / denominator,
        "cross_block_frobenius_norm": float(np.linalg.norm(form24)),
        "a5_cross_block_power_identity_residual": abs(
            float(np.linalg.norm(form24)) - numerator
        ),
    }


def face_streamed_precision_rows(
    target_levels: Iterable[int] = TARGET_LEVELS,
) -> list[dict[str, Any]]:
    """Compute only the frozen precision blocks, one base face at a time."""

    requested = tuple(sorted({int(value) for value in target_levels}))
    if not requested or requested[0] < 0:
        raise ValueError("target levels must be nonnegative")
    maximum = requested[-1]
    forms = {
        level: {
            "22": np.zeros((5, 5), dtype=np.complex128),
            "24": np.zeros((5, 9), dtype=np.complex128),
            "44": np.zeros((9, 9), dtype=np.complex128),
        }
        for level in requested
    }
    base = build_geodesic_icosahedral_tower(0).levels[0]
    for base_face in base.faces:
        vertices = np.asarray(base.vertices[np.asarray(base_face)], dtype=float)
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        for level in range(maximum + 1):
            if level in forms:
                harmonic_two = _solid_harmonics(vertices, 2)
                harmonic_four = _solid_harmonics(vertices, 4)
                forms[level]["22"] += _accumulate_triangle_form(
                    harmonic_two, harmonic_two, faces
                )
                forms[level]["24"] += _accumulate_triangle_form(
                    harmonic_two, harmonic_four, faces
                )
                forms[level]["44"] += _accumulate_triangle_form(
                    harmonic_four, harmonic_four, faces
                )
            if level < maximum:
                vertices, faces = _refine_local_face(vertices, faces)
    rows: list[dict[str, Any]] = []
    for level in requested:
        summary = _targeted_biposh_summary(
            forms[level]["22"], forms[level]["24"], forms[level]["44"]
        )
        rows.append(
            {
                "level": level,
                "frequency": 2**level,
                "vertex_count_global": 10 * 4**level + 2,
                "edge_count_global": 30 * 4**level,
                "face_count_global": 20 * 4**level,
                "base_face_local_vertex_count": (2**level + 1)
                * (2**level + 2)
                // 2,
                "summary": summary,
            }
        )
    return rows


def _legendre(ell: int, value: float) -> float:
    if ell == 2:
        return (3.0 * value * value - 1.0) / 2.0
    if ell == 4:
        square = value * value
        return (35.0 * square * square - 30.0 * square + 3.0) / 8.0
    raise ValueError("tail packet only bounds ell two and four")


def _harmonic_vector_constant(ell: int) -> float:
    return math.sqrt((2 * ell + 1) / (4.0 * math.pi))


def _exact_difference_bound(ell: int, edge_bound: float) -> float:
    # Stable exact polynomial forms of 2(1-P_l(cos(h))).  Writing these in
    # t=sin(h/2)^2 avoids the catastrophic subtraction that appears in the
    # direct addition-theorem expression at deep levels.
    t = math.sin(float(edge_bound) / 2.0) ** 2
    if ell == 2:
        value = 12.0 * t * (1.0 - t)
    elif ell == 4:
        value = 20.0 * t * (1.0 - t) * (7.0 * t**2 - 7.0 * t + 2.0)
    else:
        raise ValueError("stable tail polynomial is registered only for ell 2 and 4")
    return _harmonic_vector_constant(ell) * math.sqrt(value)


def _exact_midpoint_defect_bound(ell: int, edge_bound: float) -> float:
    # Exact stable forms of
    # 1+(1+P_l(cos h))/2-2P_l(cos(h/2)).
    t = math.sin(float(edge_bound) / 2.0) ** 2
    if ell == 2:
        value = 3.0 * t**2
    elif ell == 4:
        value = 5.0 * t**2 * (28.0 * t**2 - 56.0 * t + 29.0) / 4.0
    else:
        raise ValueError("stable tail polynomial is registered only for ell 2 and 4")
    return _harmonic_vector_constant(ell) * math.sqrt(value)


def _generator_difference_bound(ell: int, edge_bound: float) -> float:
    return (
        math.sqrt(ell * (ell + 1.0))
        * _harmonic_vector_constant(ell)
        * edge_bound
    )


def _generator_midpoint_defect_bound(ell: int, edge_bound: float) -> float:
    return (
        ell
        * (ell + 1.0)
        * _harmonic_vector_constant(ell)
        * edge_bound**2
        / 8.0
    )


def _block_increment_bound(
    ell: int,
    ell_prime: int,
    edge_bound: float,
    level: int,
    *,
    generator_bound: bool,
) -> float:
    if generator_bound:
        difference = _generator_difference_bound
        midpoint = _generator_midpoint_defect_bound
    else:
        difference = _exact_difference_bound
        midpoint = _exact_midpoint_defect_bound
    d_left, d_right = difference(ell, edge_bound), difference(
        ell_prime, edge_bound
    )
    m_left, m_right = midpoint(ell, edge_bound), midpoint(
        ell_prime, edge_bound
    )
    per_face = 3.0 * (m_left * d_right + d_left * m_right) + 15.0 * (
        m_left * m_right
    )
    return 20.0 * 4.0**level * per_face


def mesh_and_tail_report() -> dict[str, Any]:
    """Return explicit shape contraction and Cauchy-tail bounds."""

    base_edge = math.acos(1.0 / math.sqrt(5.0))
    edge_bounds: list[float] = []
    edge = base_edge
    for _ in range(TAIL_EXACT_STOP + 1):
        edge_bounds.append(edge)
        edge *= 1.0 / (2.0 * math.cos(edge / 2.0))
    contraction_rows = [
        {
            "level": level,
            "maximum_edge_upper_bound_radians": value,
            "midpoint_map_lipschitz_upper_bound": 1.0
            / (2.0 * math.cos(value / 2.0)),
        }
        for level, value in enumerate(edge_bounds)
    ]
    anchor_t = math.sin(edge_bounds[ANCHOR_LEVEL] / 2.0) ** 2
    monotonicity_threshold = (7.0 - math.sqrt(21.0)) / 14.0
    block_rows: list[dict[str, Any]] = []
    for ell, ell_prime, block_id in (
        (2, 4, "ell2_by_ell4"),
        (2, 2, "ell2_by_ell2"),
        (4, 4, "ell4_by_ell4"),
    ):
        finite_terms = [
            _block_increment_bound(
                ell,
                ell_prime,
                edge_bounds[level],
                level,
                generator_bound=False,
            )
            for level in range(ANCHOR_LEVEL, TAIL_EXACT_STOP)
        ]
        start = TAIL_EXACT_STOP
        generator_term = _block_increment_bound(
            ell,
            ell_prime,
            edge_bounds[start],
            start,
            generator_bound=True,
        )
        contraction = 1.0 / (
            2.0 * math.cos(edge_bounds[start] / 2.0)
        )
        geometric_ratio = max(4.0 * contraction**3, 4.0 * contraction**4)
        generator_tail = generator_term / (1.0 - geometric_ratio)
        raw_tail = sum(finite_terms) + generator_tail
        inflated_tail = raw_tail * TAIL_ROUNDOFF_INFLATION
        block_rows.append(
            {
                "block_id": block_id,
                "ell": ell,
                "ell_prime": ell_prime,
                "exact_addition_theorem_levels": list(
                    range(ANCHOR_LEVEL, TAIL_EXACT_STOP)
                ),
                "finite_increment_upper_bounds": finite_terms,
                "generator_bound_starts_at_level": start,
                "generator_first_increment_upper_bound": generator_term,
                "geometric_tail_ratio_upper_bound": geometric_ratio,
                "unrounded_tail_upper_bound": raw_tail,
                "roundoff_inflation_factor": TAIL_ROUNDOFF_INFLATION,
                "certified_tail_upper_bound": inflated_tail,
            }
        )
    return {
        "base_edge_exact_description": "acos(1/sqrt(5))",
        "midpoint_contraction_theorem": (
            "On a spherical ball of radius h<pi/2, radial geodesic halving "
            "has Lipschitz constant at most 1/(2 cos(h/2))."
        ),
        "contraction_rows": contraction_rows,
        "harmonic_addition_theorem": (
            "||Y_l(x)-Y_l(y)||^2=c_l^2 2(1-P_l(cos d)); the symmetric "
            "midpoint defect has the declared three-kernel expression"
        ),
        "stable_polynomial_forms": {
            "variable": "t=sin(h/2)^2",
            "D2_squared_over_c2_squared": "12*t*(1-t)",
            "M2_squared_over_c2_squared": "3*t^2",
            "D4_squared_over_c4_squared": "20*t*(1-t)*(7*t^2-7*t+2)",
            "M4_squared_over_c4_squared": "5*t^2*(28*t^2-56*t+29)/4",
            "cancellation_free": True,
            "tail_domain_t_upper_bound": anchor_t,
            "shared_derivative_positive_threshold": monotonicity_threshold,
            "all_squared_bounds_monotone_on_tail_domain": bool(
                0.0 <= anchor_t < monotonicity_threshold
            ),
        },
        "generator_fallback": (
            "||Delta Y_l||<=sqrt(l(l+1)) c_l h and ||midpoint defect||"
            "<=l(l+1) c_l h^2/8"
        ),
        "block_rows": block_rows,
        "cauchy_limit_exists_for_declared_blocks": all(
            row["geometric_tail_ratio_upper_bound"] < 1.0 for row in block_rows
        ),
    }


def _finite_anchor_error_report(anchor: dict[str, Any]) -> dict[str, Any]:
    edge_count = int(anchor["edge_count_global"])
    half_edge_term_count = 2 * edge_count
    edge_bound = mesh_and_tail_report()["contraction_rows"][ANCHOR_LEVEL][
        "maximum_edge_upper_bound_radians"
    ]
    unit_roundoff = np.finfo(float).eps / 2.0
    gamma = half_edge_term_count * unit_roundoff / (
        1.0 - half_edge_term_count * unit_roundoff
    )
    d2 = _generator_difference_bound(2, edge_bound)
    d4 = _generator_difference_bound(4, edge_bound)
    value_error = HARMONIC_VALUE_ERROR_PREMISE

    def form_radius(left: float, right: float, dimension: int) -> float:
        evaluation = edge_count * (
            (left + 2.0 * value_error) * 2.0 * value_error
            + (right + 2.0 * value_error) * 2.0 * value_error
            + (2.0 * value_error) ** 2
        )
        summation = (
            gamma
            * edge_count
            * (left + 2.0 * value_error)
            * (right + 2.0 * value_error)
        )
        return math.sqrt(float(dimension)) * (evaluation + summation)

    calculated = {
        "ell2_by_ell4": form_radius(d2, d4, 5 * 9),
        "ell2_by_ell2": form_radius(d2, d2, 5 * 5),
        "ell4_by_ell4": form_radius(d4, d4, 9 * 9),
    }
    return {
        "anchor_level": ANCHOR_LEVEL,
        "binary64_unit_roundoff": unit_roundoff,
        "global_edge_count": edge_count,
        "accumulated_half_edge_term_count": half_edge_term_count,
        "gamma_n": gamma,
        "declared_per_harmonic_value_error_envelope": value_error,
        "declared_value_error_envelope_is_analytic_library_proof": False,
        "calculated_block_radii": calculated,
        "adopted_common_block_radius": ANCHOR_NUMERICAL_RADIUS,
        "common_radius_dominates_calculated_radii": all(
            value < ANCHOR_NUMERICAL_RADIUS for value in calculated.values()
        ),
        "scope": (
            "The summation propagation is explicit. The per-value envelope is "
            "a conservative declared numerical premise checked against a second "
            "harmonic implementation; it is not a formal proof of SciPy or libm."
        ),
    }


def _interval_from_anchor(
    anchor: dict[str, Any],
    tail: dict[str, Any],
) -> dict[str, Any]:
    summary = anchor["summary"]
    tail_by_id = {row["block_id"]: row for row in tail["block_rows"]}
    radius = ANCHOR_NUMERICAL_RADIUS
    numerator = float(summary["primary_numerator_norm"])
    a22 = float(summary["A_22_00"][0])
    a44 = float(summary["A_44_00"][0])
    numerator_tail = tail_by_id["ell2_by_ell4"]["certified_tail_upper_bound"]
    a22_tail = tail_by_id["ell2_by_ell2"]["certified_tail_upper_bound"]
    a44_tail = tail_by_id["ell4_by_ell4"]["certified_tail_upper_bound"]
    numerator_interval = [
        max(0.0, numerator - radius - numerator_tail),
        numerator + radius + numerator_tail,
    ]
    a22_interval = [a22 - radius - a22_tail, a22 + radius + a22_tail]
    a44_interval = [a44 - radius - a44_tail, a44 + radius + a44_tail]
    if min(a22_interval) <= 0.0 or min(a44_interval) <= 0.0:
        raise ContinuumTailError("denominator interval crosses zero")
    denominator_interval = [
        math.sqrt(a22_interval[0] * a44_interval[0]),
        math.sqrt(a22_interval[1] * a44_interval[1]),
    ]
    statistic_interval = [
        numerator_interval[0] / denominator_interval[1],
        numerator_interval[1] / denominator_interval[0],
    ]
    return {
        "primary_numerator_norm_interval": numerator_interval,
        "A_22_00_interval": a22_interval,
        "A_44_00_interval": a44_interval,
        "primary_denominator_interval": denominator_interval,
        "primary_amplitude_free_statistic_interval": statistic_interval,
        "conditional_interval_excludes_zero": statistic_interval[0] > 0.0,
        "conditioning": [
            "equal-seam stiffness form on every registered refinement level",
            "ideal geodesic-midpoint continuation",
            "declared finite-anchor harmonic evaluation envelope",
        ],
    }


def _inverse_covariance_rows(maximum_level: int = 7) -> list[dict[str, Any]]:
    """Finite conditional A K^-1 diagnostic; no continuum claim is made."""

    tower = build_geodesic_icosahedral_tower(maximum_level)
    rows: list[dict[str, Any]] = []
    for level in range(2, maximum_level + 1):
        mesh = tower.levels[level]
        weights = _vertex_area_weights(mesh)
        design = _harmonic_design(mesh.vertices)
        active, _ = _weighted_low_mode_removal(design, weights)
        stiffness = _equal_seam_graph_stiffness(mesh)
        form = active.conj().T @ (stiffness @ active)
        form = 0.5 * (form + form.conj().T)
        covariance = np.linalg.inv(form)
        coefficients, _ = _biposh_rows(covariance)
        summary = _biposh_summary(coefficients)
        rows.append(
            {
                "level": level,
                "primary_amplitude_free_statistic": summary[
                    "primary_amplitude_free_statistic"
                ],
                "minimum_stiffness_eigenvalue": float(
                    np.min(np.linalg.eigvalsh(form))
                ),
                "inverse_hermiticity_residual": float(
                    np.max(np.abs(covariance - covariance.conj().T))
                ),
            }
        )
    return rows


def build_a5_biposh_continuum_tail_packet() -> dict[str, Any]:
    parent = json.loads(FINITE_PARENT.read_text(encoding="utf-8"))
    if (
        parent.get("schema") != "oph.a5-biposh-dual-operator-refinement.v1"
        or parent.get("selection_decision", {}).get(
            "refinement_tower_equal_seam_extension_source_selected"
        )
        is not False
        or parent.get("selection_decision", {}).get("physical_covariance_selected")
        is not False
    ):
        raise ContinuumTailError("finite parent does not preserve the open boundary")
    identity = exact_refinement_identity_report()
    precision_rows = face_streamed_precision_rows()
    anchor = precision_rows[-1]
    tail = mesh_and_tail_report()
    numerical = _finite_anchor_error_report(anchor)
    interval = _interval_from_anchor(anchor, tail)
    inverse_rows = _inverse_covariance_rows()
    producer = Path(__file__).resolve()
    verifier = producer.with_name("verify_a5_biposh_continuum_tail_independent.py")
    test_path = ROOT / "tests/test_a5_biposh_continuum_tail.py"
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "issue": 659,
        "status": STATUS,
        "parent": {
            "path": FINITE_PARENT.relative_to(ROOT).as_posix(),
            "schema": parent["schema"],
            "status": parent["status"],
            "payload_sha256": parent["payload_sha256"],
        },
        "frozen_statistic": parent["frozen_primary_statistic"],
        "exact_refinement_identity": identity,
        "face_streamed_precision_rows": precision_rows,
        "mesh_and_tail_certificate": tail,
        "a5_cross_block_reduction": {
            "coupling_total_L_range": [2, 3, 4, 5, 6],
            "a5_singlet_total_L_values_in_range": [6],
            "consequence": (
                "For every exactly A5-equivariant level and its blockwise limit, "
                "the ell=2 by ell=4 Frobenius norm equals the norm of A_24^{6M}."
            ),
            "finite_anchor_residual": anchor["summary"][
                "a5_cross_block_power_identity_residual"
            ],
            "finite_anchor_residual_within_numerical_radius": anchor["summary"][
                "a5_cross_block_power_identity_residual"
            ]
            < ANCHOR_NUMERICAL_RADIUS,
        },
        "finite_anchor_numerical_envelope": numerical,
        "conditional_continuum_interval": interval,
        "conditional_inverse_covariance": {
            "branch": (
                "finite continuous quadratic-energy MaxEnt identity covariance=A K^-1"
            ),
            "amplitude_cancels_from_primary": True,
            "rows": inverse_rows,
            "finite_matrix_calculation_attained": True,
            "continuum_tail_enclosed": False,
            "continuum_blocker": (
                "The stiffness-block tail does not control inversion through the "
                "full ell=2..8 matrix. A uniform full-band lower bound and full-matrix "
                "tail are required."
            ),
            "source_ensemble_selected": False,
            "physical_covariance_selected": False,
        },
        "selection_decision": {
            "equal_seam_refinement_extension_source_selected": False,
            "global_a1_a3_policy_uniqueness_receipt": False,
            "physical_repair_law_selected": False,
            "conditional_stiffness_continuum_limit_exists": True,
            "conditional_stiffness_l6_nonzero_under_numerical_envelope": interval[
                "conditional_interval_excludes_zero"
            ],
            "finite_inverse_covariance_diagnostic": True,
            "inverse_covariance_continuum_limit_decided": False,
            "physical_covariance_selected": False,
            "physical_release_ensemble_selected": False,
            "global_frame_quotient_visible": False,
            "screen_to_sky_readout_selected": False,
            "physical_prediction": False,
            "promotion_allowed": False,
        },
        "source_pins": [
            _file_pin(FINITE_PARENT),
            _file_pin(ROOT / "oph_fpe/core/icosahedral.py"),
            _file_pin(producer),
            _file_pin(verifier),
            _file_pin(test_path),
        ],
        "claim_boundary": (
            "Conditional mathematical continuation of the frozen equal-seam "
            "stiffness fingerprint. The rational refinement identity and explicit "
            "Cauchy bound establish a blockwise continuum limit. The level-nine "
            "anchor and declared numerical envelope separate its L=6 norm from "
            "zero. The equal-seam extension is not source-selected. The separately "
            "typed inverse-stiffness rows instantiate only a finite continuous "
            "quadratic-energy branch and have no continuum tail. No source ensemble, "
            "physical covariance, frame, sky readout, comparison, or prediction is "
            "claimed."
        ),
    }
    receipt["payload_sha256"] = _sha(_canonical_bytes(receipt))
    return receipt


def write_a5_biposh_continuum_tail_packet(
    output: Path = DEFAULT_RECEIPT,
) -> dict[str, Any]:
    receipt = build_a5_biposh_continuum_tail_packet()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    receipt = write_a5_biposh_continuum_tail_packet(args.output)
    print(receipt["status"])
    print(receipt["payload_sha256"])


if __name__ == "__main__":
    main()
