"""Independent replay for the conditional A5 BipoSH continuum-tail packet."""

from __future__ import annotations

from fractions import Fraction
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import sph_harm_y

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
EXPECTED_SCHEMA = "oph.a5-biposh-continuum-tail.v1"
EXPECTED_STATUS = (
    "CONDITIONAL_EQUAL_SEAM_CONTINUUM_L6_NONZERO_UNDER_DECLARED_NUMERICAL_"
    "ENVELOPE__SOURCE_SELECTION_AND_PHYSICAL_TRANSFER_OPEN"
)
ANCHOR_LEVEL = 9
TARGET_LEVELS = (6, 7, 8, 9)
TAIL_EXACT_STOP = 48
TAIL_INFLATION = 1.05


class VerificationError(ValueError):
    """Raised when independent replay refuses the packet."""


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


def _check_payload_hash(receipt: dict[str, Any]) -> None:
    claimed = receipt.get("payload_sha256")
    payload = dict(receipt)
    payload.pop("payload_sha256", None)
    if claimed != _sha(_canonical_bytes(payload)):
        raise VerificationError("payload hash mismatch")


def _check_source_pins(receipt: dict[str, Any]) -> None:
    pins = receipt.get("source_pins")
    if not isinstance(pins, list) or len(pins) != 5:
        raise VerificationError("source pin set mismatch")
    for pin in pins:
        path = ROOT / pin["path"]
        payload = path.read_bytes()
        if len(payload) != pin["bytes"] or _sha(payload) != pin["sha256"]:
            raise VerificationError(f"source pin mismatch: {pin['path']}")


def _energy(left: tuple, right: tuple) -> Fraction:
    return sum(
        (
            (left[index] - left[(index + 1) % 3])
            * (right[index] - right[(index + 1) % 3])
            / 2
            for index in range(3)
        ),
        Fraction(0),
    )


def _identity_direct(left: tuple, right: tuple) -> Fraction:
    a, b, c, u, v, w = left
    A, B, C, U, V, W = right
    x, y, z = (a + b) / 2 + u, (b + c) / 2 + v, (c + a) / 2 + w
    X, Y, Z = (A + B) / 2 + U, (B + C) / 2 + V, (C + A) / 2 + W
    children = (
        ((a, x, z), (A, X, Z)),
        ((b, y, x), (B, Y, X)),
        ((c, z, y), (C, Z, Y)),
        ((x, y, z), (X, Y, Z)),
    )
    return sum((_energy(p, q) for p, q in children), Fraction(0)) - _energy(
        (a, b, c), (A, B, C)
    )


def _identity_factored(left: tuple, right: tuple) -> Fraction:
    a, b, c, u, v, w = left
    A, B, C, U, V, W = right
    return (
        u * (A / 2 + B / 2 - C + 3 * U - V - W)
        + v * (-A + B / 2 + C / 2 - U + 3 * V - W)
        + w * (A / 2 - B + C / 2 - U - V + 3 * W)
        + U * (a / 2 + b / 2 - c)
        + V * (-a + b / 2 + c / 2)
        + W * (a / 2 - b + c / 2)
    )


def _verify_identity(row: dict[str, Any]) -> None:
    cases = 0
    for left_index in range(6):
        for right_index in range(6):
            left = [Fraction(0)] * 6
            right = [Fraction(0)] * 6
            left[left_index] = Fraction(1)
            right[right_index] = Fraction(1)
            if _identity_direct(tuple(left), tuple(right)) != _identity_factored(
                tuple(left), tuple(right)
            ):
                raise VerificationError("exact refinement identity failed")
            cases += 1
    if (
        row.get("arithmetic") != "fractions.Fraction over Q"
        or row.get("coefficient_basis_cases") != cases
        or row.get("identity_verified") is not True
    ):
        raise VerificationError("exact refinement identity disclosure mismatch")


def _refine(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(vertices, dtype=float)
    points = [value.copy() for value in original]
    lookup: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = tuple(sorted((a, b)))
        if key not in lookup:
            point = original[key[0]] + original[key[1]]
            point /= math.sqrt(float(np.dot(point, point)))
            lookup[key] = len(points)
            points.append(point)
        return lookup[key]

    children: list[tuple[int, int, int]] = []
    for row in faces.tolist():
        a, b, c = (int(value) for value in row)
        p, q, r = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        children.extend(((a, p, r), (b, q, p), (c, r, q), (p, q, r)))
    return np.asarray(points, dtype=float), np.asarray(children, dtype=np.int64)


def _scipy_harmonics(points: np.ndarray, ell: int) -> np.ndarray:
    theta = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * math.pi)
    return np.column_stack(
        [sph_harm_y(ell, m, theta, phi) for m in range(-ell, ell + 1)]
    )


def _triangle_form(left: np.ndarray, right: np.ndarray, faces: np.ndarray) -> np.ndarray:
    result = np.zeros((left.shape[1], right.shape[1]), dtype=np.complex128)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        dl = left[faces[:, a]] - left[faces[:, b]]
        dr = right[faces[:, a]] - right[faces[:, b]]
        result += dl.conj().T @ dr / 2.0
    return result


def _summary(form22: np.ndarray, form24: np.ndarray, form44: np.ndarray) -> dict:
    def coefficient(form, ell, ell_prime, total_l, total_m):
        output = 0.0j
        for m in range(-ell, ell + 1):
            mp = m - total_m
            if -ell_prime <= mp <= ell_prime:
                output += (
                    (-1) ** mp
                    * _clebsch_gordan(ell, m, ell_prime, -mp, total_l, total_m)
                    * form[m + ell, mp + ell_prime]
                )
        return output

    vector = np.asarray(
        [coefficient(form24, 2, 4, 6, value) for value in range(-6, 7)]
    )
    a22 = coefficient(form22, 2, 2, 0, 0)
    a44 = coefficient(form44, 4, 4, 0, 0)
    numerator = float(np.linalg.norm(vector))
    denominator = math.sqrt(abs(a22 * a44))
    return {
        "A_22_00": [a22.real, a22.imag],
        "A_44_00": [a44.real, a44.imag],
        "A_24_6M": [[value.real, value.imag] for value in vector],
        "primary_numerator_norm": numerator,
        "primary_denominator": denominator,
        "primary_amplitude_free_statistic": numerator / denominator,
        "cross_block_frobenius_norm": float(np.linalg.norm(form24)),
        "a5_cross_block_power_identity_residual": abs(
            float(np.linalg.norm(form24)) - numerator
        ),
    }


def _replay_precision_rows() -> list[dict[str, Any]]:
    forms = {
        level: {
            "22": np.zeros((5, 5), complex),
            "24": np.zeros((5, 9), complex),
            "44": np.zeros((9, 9), complex),
        }
        for level in TARGET_LEVELS
    }
    base = build_geodesic_icosahedral_tower(0).levels[0]
    for face in base.faces:
        vertices = np.asarray(base.vertices[np.asarray(face)], dtype=float)
        triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
        for level in range(ANCHOR_LEVEL + 1):
            if level in forms:
                y2 = _scipy_harmonics(vertices, 2)
                y4 = _scipy_harmonics(vertices, 4)
                forms[level]["22"] += _triangle_form(y2, y2, triangles)
                forms[level]["24"] += _triangle_form(y2, y4, triangles)
                forms[level]["44"] += _triangle_form(y4, y4, triangles)
            if level < ANCHOR_LEVEL:
                vertices, triangles = _refine(vertices, triangles)
    return [
        {
            "level": level,
            "frequency": 2**level,
            "vertex_count_global": 10 * 4**level + 2,
            "edge_count_global": 30 * 4**level,
            "face_count_global": 20 * 4**level,
            "base_face_local_vertex_count": (2**level + 1)
            * (2**level + 2)
            // 2,
            "summary": _summary(
                forms[level]["22"], forms[level]["24"], forms[level]["44"]
            ),
        }
        for level in TARGET_LEVELS
    ]


def _assert_close(left: Any, right: Any, tolerance: float, label: str) -> None:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape or np.max(np.abs(a - b), initial=0.0) > tolerance:
        raise VerificationError(f"{label} replay mismatch")


def _verify_precision_rows(stored: list[dict[str, Any]]) -> None:
    replayed = _replay_precision_rows()
    if [row.get("level") for row in stored] != list(TARGET_LEVELS):
        raise VerificationError("precision level set mismatch")
    for expected, actual in zip(stored, replayed, strict=True):
        for key in (
            "frequency",
            "vertex_count_global",
            "edge_count_global",
            "face_count_global",
            "base_face_local_vertex_count",
        ):
            if expected.get(key) != actual[key]:
                raise VerificationError(f"precision metadata mismatch: {key}")
        for key in (
            "A_22_00",
            "A_44_00",
            "A_24_6M",
            "primary_numerator_norm",
            "primary_denominator",
            "primary_amplitude_free_statistic",
            "cross_block_frobenius_norm",
            "a5_cross_block_power_identity_residual",
        ):
            _assert_close(
                expected["summary"][key],
                actual["summary"][key],
                2.0e-9,
                f"precision {expected['level']} {key}",
            )


def _legendre(ell: int, x: float) -> float:
    if ell == 2:
        return (3.0 * x * x - 1.0) / 2.0
    square = x * x
    return (35.0 * square * square - 30.0 * square + 3.0) / 8.0


def _constant(ell: int) -> float:
    return math.sqrt((2 * ell + 1) / (4.0 * math.pi))


def _bounds(ell: int, h: float, generator: bool) -> tuple[float, float]:
    if generator:
        return (
            math.sqrt(ell * (ell + 1.0)) * _constant(ell) * h,
            ell * (ell + 1.0) * _constant(ell) * h**2 / 8.0,
        )
    t = math.sin(h / 2.0) ** 2
    if ell == 2:
        difference_square = 12.0 * t * (1.0 - t)
        midpoint_square = 3.0 * t**2
    elif ell == 4:
        difference_square = 20.0 * t * (1.0 - t) * (
            7.0 * t**2 - 7.0 * t + 2.0
        )
        midpoint_square = (
            5.0 * t**2 * (28.0 * t**2 - 56.0 * t + 29.0) / 4.0
        )
    else:
        raise VerificationError("tail harmonic outside ell two/four")
    difference = _constant(ell) * math.sqrt(difference_square)
    midpoint = _constant(ell) * math.sqrt(midpoint_square)
    return difference, midpoint


def _increment(ell: int, prime: int, h: float, level: int, generator: bool) -> float:
    dl, ml = _bounds(ell, h, generator)
    dr, mr = _bounds(prime, h, generator)
    return 20.0 * 4.0**level * (3.0 * (ml * dr + dl * mr) + 15.0 * ml * mr)


def _replay_tail() -> dict[str, Any]:
    h = math.acos(1.0 / math.sqrt(5.0))
    edge_bounds = []
    for _ in range(TAIL_EXACT_STOP + 1):
        edge_bounds.append(h)
        h /= 2.0 * math.cos(h / 2.0)
    rows = []
    for ell, prime, identifier in (
        (2, 4, "ell2_by_ell4"),
        (2, 2, "ell2_by_ell2"),
        (4, 4, "ell4_by_ell4"),
    ):
        finite = [
            _increment(ell, prime, edge_bounds[level], level, False)
            for level in range(ANCHOR_LEVEL, TAIL_EXACT_STOP)
        ]
        q = 1.0 / (2.0 * math.cos(edge_bounds[TAIL_EXACT_STOP] / 2.0))
        ratio = max(4.0 * q**3, 4.0 * q**4)
        first = _increment(
            ell, prime, edge_bounds[TAIL_EXACT_STOP], TAIL_EXACT_STOP, True
        )
        raw = sum(finite) + first / (1.0 - ratio)
        rows.append(
            {
                "block_id": identifier,
                "finite_increment_upper_bounds": finite,
                "generator_first_increment_upper_bound": first,
                "geometric_tail_ratio_upper_bound": ratio,
                "unrounded_tail_upper_bound": raw,
                "certified_tail_upper_bound": raw * TAIL_INFLATION,
            }
        )
    return {"edge_bounds": edge_bounds, "block_rows": rows}


def _verify_tail(stored: dict[str, Any]) -> None:
    replayed = _replay_tail()
    contraction = stored.get("contraction_rows", [])
    if len(contraction) != TAIL_EXACT_STOP + 1:
        raise VerificationError("mesh contraction row count mismatch")
    _assert_close(
        [row["maximum_edge_upper_bound_radians"] for row in contraction],
        replayed["edge_bounds"],
        2.0e-15,
        "edge contraction",
    )
    indexed = {row.get("block_id"): row for row in stored.get("block_rows", [])}
    for row in replayed["block_rows"]:
        candidate = indexed.get(row["block_id"])
        if candidate is None:
            raise VerificationError("tail block missing")
        for key, value in row.items():
            if key == "block_id":
                continue
            _assert_close(candidate[key], value, 5.0e-12, f"tail {key}")
        if candidate.get("roundoff_inflation_factor") != TAIL_INFLATION:
            raise VerificationError("tail inflation mismatch")
    if stored.get("cauchy_limit_exists_for_declared_blocks") is not True:
        raise VerificationError("Cauchy limit flag missing")
    stable = stored.get("stable_polynomial_forms", {})
    anchor_t = math.sin(replayed["edge_bounds"][ANCHOR_LEVEL] / 2.0) ** 2
    threshold = (7.0 - math.sqrt(21.0)) / 14.0
    if (
        stable.get("variable") != "t=sin(h/2)^2"
        or stable.get("D2_squared_over_c2_squared") != "12*t*(1-t)"
        or stable.get("M2_squared_over_c2_squared") != "3*t^2"
        or stable.get("D4_squared_over_c4_squared")
        != "20*t*(1-t)*(7*t^2-7*t+2)"
        or stable.get("M4_squared_over_c4_squared")
        != "5*t^2*(28*t^2-56*t+29)/4"
        or stable.get("cancellation_free") is not True
        or abs(float(stable.get("tail_domain_t_upper_bound", math.inf)) - anchor_t)
        > 2.0e-18
        or abs(
            float(stable.get("shared_derivative_positive_threshold", math.inf))
            - threshold
        )
        > 2.0e-16
        or stable.get("all_squared_bounds_monotone_on_tail_domain") is not True
        or not (0.0 <= anchor_t < threshold)
    ):
        raise VerificationError("stable addition-theorem polynomial mismatch")


def _replay_inverse_rows(maximum: int = 7) -> list[dict[str, Any]]:
    tower = build_geodesic_icosahedral_tower(maximum)
    output = []
    for level in range(2, maximum + 1):
        mesh = tower.levels[level]
        weights = _vertex_area_weights(mesh)
        design = _harmonic_design(mesh.vertices)
        active, _ = _weighted_low_mode_removal(design, weights)
        operator = _equal_seam_graph_stiffness(mesh)
        form = active.conj().T @ (operator @ active)
        form = (form + form.conj().T) / 2.0
        inverse = np.linalg.inv(form)
        coefficients, _ = _biposh_rows(inverse)
        summary = _biposh_summary(coefficients)
        output.append(
            {
                "level": level,
                "primary_amplitude_free_statistic": summary[
                    "primary_amplitude_free_statistic"
                ],
                "minimum_stiffness_eigenvalue": float(np.min(np.linalg.eigvalsh(form))),
                "inverse_hermiticity_residual": float(
                    np.max(np.abs(inverse - inverse.conj().T))
                ),
            }
        )
    return output


def _verify_inverse(section: dict[str, Any]) -> None:
    if (
        section.get("finite_matrix_calculation_attained") is not True
        or section.get("continuum_tail_enclosed") is not False
        or section.get("source_ensemble_selected") is not False
        or section.get("physical_covariance_selected") is not False
    ):
        raise VerificationError("inverse covariance boundary mismatch")
    replayed = _replay_inverse_rows()
    stored = section.get("rows", [])
    if len(stored) != len(replayed):
        raise VerificationError("inverse covariance row count mismatch")
    for left, right in zip(stored, replayed, strict=True):
        if left.get("level") != right["level"]:
            raise VerificationError("inverse covariance level mismatch")
        for key in (
            "primary_amplitude_free_statistic",
            "minimum_stiffness_eigenvalue",
            "inverse_hermiticity_residual",
        ):
            _assert_close(left[key], right[key], 3.0e-9, f"inverse {key}")


def _verify_interval(receipt: dict[str, Any]) -> None:
    anchor = receipt["face_streamed_precision_rows"][-1]
    tails = {
        row["block_id"]: row["certified_tail_upper_bound"]
        for row in receipt["mesh_and_tail_certificate"]["block_rows"]
    }
    radius = receipt["finite_anchor_numerical_envelope"][
        "adopted_common_block_radius"
    ]
    if radius != 0.01:
        raise VerificationError("finite anchor radius mismatch")
    summary = anchor["summary"]
    numerator = summary["primary_numerator_norm"]
    a22 = summary["A_22_00"][0]
    a44 = summary["A_44_00"][0]
    expected_numerator = [
        max(0.0, numerator - radius - tails["ell2_by_ell4"]),
        numerator + radius + tails["ell2_by_ell4"],
    ]
    expected_a22 = [
        a22 - radius - tails["ell2_by_ell2"],
        a22 + radius + tails["ell2_by_ell2"],
    ]
    expected_a44 = [
        a44 - radius - tails["ell4_by_ell4"],
        a44 + radius + tails["ell4_by_ell4"],
    ]
    denominator = [
        math.sqrt(expected_a22[0] * expected_a44[0]),
        math.sqrt(expected_a22[1] * expected_a44[1]),
    ]
    statistic = [
        expected_numerator[0] / denominator[1],
        expected_numerator[1] / denominator[0],
    ]
    stored = receipt["conditional_continuum_interval"]
    for key, value in (
        ("primary_numerator_norm_interval", expected_numerator),
        ("A_22_00_interval", expected_a22),
        ("A_44_00_interval", expected_a44),
        ("primary_denominator_interval", denominator),
        ("primary_amplitude_free_statistic_interval", statistic),
    ):
        _assert_close(stored[key], value, 3.0e-12, f"continuum interval {key}")
    if stored.get("conditional_interval_excludes_zero") is not True or statistic[0] <= 0:
        raise VerificationError("conditional continuum interval does not exclude zero")


def _verify_numerical_envelope(receipt: dict[str, Any]) -> None:
    section = receipt["finite_anchor_numerical_envelope"]
    anchor = receipt["face_streamed_precision_rows"][-1]
    edge_count = int(anchor["edge_count_global"])
    term_count = 2 * edge_count
    unit_roundoff = np.finfo(float).eps / 2.0
    gamma = term_count * unit_roundoff / (1.0 - term_count * unit_roundoff)
    contraction = receipt["mesh_and_tail_certificate"]["contraction_rows"]
    h = float(contraction[ANCHOR_LEVEL]["maximum_edge_upper_bound_radians"])
    value_error = float(section["declared_per_harmonic_value_error_envelope"])
    d2 = math.sqrt(2.0 * 3.0) * math.sqrt(5.0 / (4.0 * math.pi)) * h
    d4 = math.sqrt(4.0 * 5.0) * math.sqrt(9.0 / (4.0 * math.pi)) * h

    def radius(left: float, right: float, dimension: int) -> float:
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

    expected = {
        "ell2_by_ell4": radius(d2, d4, 45),
        "ell2_by_ell2": radius(d2, d2, 25),
        "ell4_by_ell4": radius(d4, d4, 81),
    }
    if (
        section.get("global_edge_count") != edge_count
        or section.get("accumulated_half_edge_term_count") != term_count
        or section.get("declared_value_error_envelope_is_analytic_library_proof")
        is not False
        or section.get("common_radius_dominates_calculated_radii") is not True
    ):
        raise VerificationError("finite anchor envelope disclosure mismatch")
    _assert_close(section["binary64_unit_roundoff"], unit_roundoff, 0.0, "unit roundoff")
    _assert_close(section["gamma_n"], gamma, 2.0e-18, "gamma_n")
    for key, value in expected.items():
        _assert_close(
            section["calculated_block_radii"][key],
            value,
            2.0e-14,
            f"numerical envelope {key}",
        )
    adopted = float(section["adopted_common_block_radius"])
    if adopted != 0.01 or not all(value < adopted for value in expected.values()):
        raise VerificationError("finite anchor adopted radius mismatch")


def _verify_a5_reduction(receipt: dict[str, Any]) -> None:
    section = receipt.get("a5_cross_block_reduction", {})
    residual = receipt["face_streamed_precision_rows"][-1]["summary"][
        "a5_cross_block_power_identity_residual"
    ]
    if (
        section.get("coupling_total_L_range") != [2, 3, 4, 5, 6]
        or section.get("a5_singlet_total_L_values_in_range") != [6]
        or section.get("finite_anchor_residual_within_numerical_radius") is not True
        or abs(float(section.get("finite_anchor_residual", math.inf)) - residual)
        > 1.0e-15
        or residual >= 0.01
    ):
        raise VerificationError("A5 cross-block reduction mismatch")


def verify_packet(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != EXPECTED_SCHEMA or receipt.get("status") != EXPECTED_STATUS:
        raise VerificationError("schema or status mismatch")
    _check_payload_hash(receipt)
    _check_source_pins(receipt)
    parent = json.loads(FINITE_PARENT.read_text(encoding="utf-8"))
    if (
        receipt.get("parent", {}).get("payload_sha256") != parent.get("payload_sha256")
        or receipt.get("parent", {}).get("status") != parent.get("status")
    ):
        raise VerificationError("finite parent mismatch")
    decisions = receipt.get("selection_decision", {})
    required_false = (
        "equal_seam_refinement_extension_source_selected",
        "global_a1_a3_policy_uniqueness_receipt",
        "physical_repair_law_selected",
        "inverse_covariance_continuum_limit_decided",
        "physical_covariance_selected",
        "physical_release_ensemble_selected",
        "global_frame_quotient_visible",
        "screen_to_sky_readout_selected",
        "physical_prediction",
        "promotion_allowed",
    )
    if any(decisions.get(key) is not False for key in required_false):
        raise VerificationError("selection boundary was promoted")
    _verify_identity(receipt["exact_refinement_identity"])
    _verify_precision_rows(receipt["face_streamed_precision_rows"])
    _verify_tail(receipt["mesh_and_tail_certificate"])
    _verify_numerical_envelope(receipt)
    _verify_a5_reduction(receipt)
    _verify_interval(receipt)
    _verify_inverse(receipt["conditional_inverse_covariance"])
    if (
        decisions.get("conditional_stiffness_continuum_limit_exists") is not True
        or decisions.get(
            "conditional_stiffness_l6_nonzero_under_numerical_envelope"
        )
        is not True
        or decisions.get("finite_inverse_covariance_diagnostic") is not True
    ):
        raise VerificationError("conditional mathematical result missing")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    receipt = verify_packet(args.receipt)
    print("A5_BIPOSH_CONTINUUM_TAIL_VERIFIED")
    print(receipt["status"])


if __name__ == "__main__":
    main()
