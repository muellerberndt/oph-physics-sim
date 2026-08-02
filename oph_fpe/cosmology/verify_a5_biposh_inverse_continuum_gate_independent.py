"""Correlated-kernel verifier for the A5 BipoSH inverse-continuum gate."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix

from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower
from oph_fpe.cosmology.a5_biposh_refinement import (
    LOW_MODE_DIMENSION,
    _biposh_rows,
    _biposh_summary,
    _equal_seam_graph_stiffness,
    _harmonic_design,
    _vertex_area_weights,
    _weighted_low_mode_removal,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = (
    ROOT / "data/refinement/a5_biposh_inverse_continuum_gate_receipt.json"
)
EXPECTED_SCHEMA = "oph.a5-biposh-inverse-continuum-gate.v1"
EXPECTED_STATUS = (
    "FULL_RAW_STIFFNESS_CAUCHY_TAIL_ATTAINED__UNIFORM_COERCIVITY_"
    "PROJECTED_QUOTIENT_AND_PHYSICAL_RESPONSE_OPEN"
)
ANCHOR_LEVEL = 7
EXPECTED_LEAN_DECLARATIONS = [
    "stiffness_recovered_from_uniform_repair",
    "collapsing_stiffness_positive",
    "collapsing_inverse_exact",
    "collapsing_stiffness_arbitrarily_small",
    "inverse_nuisance_unbounded",
    "copy_readout_first",
    "copy_readout_mixed",
    "copy_mixing_changes_statistic",
    "inverseTailRadius_pos",
]


class VerificationError(ValueError):
    """Raised when the committed packet fails certificate reconstruction."""


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


def _assert_close(left: Any, right: Any, tolerance: float, label: str) -> None:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape or np.max(np.abs(a - b), initial=0.0) > tolerance:
        raise VerificationError(f"{label} mismatch")


def _check_hash_and_pins(packet: dict[str, Any]) -> None:
    claimed = packet.get("payload_sha256")
    payload = dict(packet)
    payload.pop("payload_sha256", None)
    if claimed != _sha(_canonical_bytes(payload)):
        raise VerificationError("payload hash mismatch")
    pins = packet.get("source_pins")
    if not isinstance(pins, list) or len(pins) != 8:
        raise VerificationError("source pin inventory mismatch")
    for pin in pins:
        path = ROOT / str(pin["path"])
        raw = path.read_bytes()
        if len(raw) != pin["bytes"] or _sha(raw) != pin["sha256"]:
            raise VerificationError(f"source pin mismatch: {pin['path']}")


def _fraction(value: dict[str, Any]) -> Fraction:
    result = Fraction(int(value["numerator"]), int(value["denominator"]))
    if Fraction.from_float(float(value["binary64_upper"])) < result:
        raise VerificationError("serialized rational upper view is below value")
    return result


def _edge(level: int) -> tuple[Fraction, Fraction]:
    value = Fraction(10, 9)
    for _ in range(level):
        q = 1 / (2 * (1 - value * value / 8))
        value *= q
    q = 1 / (2 * (1 - value * value / 8))
    return value, q


def _replay_tail(start: int) -> dict[str, Any]:
    h, q = _edge(start)
    r3, r4 = 4 * q**3, 4 * q**4
    if not (r3 < 1 and r4 < 1):
        raise VerificationError("tail ratios do not contract")
    face_count = 20 * 4**start
    rows = []
    bounds = [[Fraction(0) for _ in range(7)] for _ in range(7)]
    c = Fraction(6, 5)
    for i, ell in enumerate(range(2, 9)):
        dl = (ell + 1) * c * h
        ml = ell * (ell + 1) * c * h**2 / 8
        for j, prime in enumerate(range(2, 9)):
            dr = (prime + 1) * c * h
            mr = prime * (prime + 1) * c * h**2 / 8
            first_md = face_count * 3 * (ml * dr + dl * mr)
            first_mm = face_count * 15 * ml * mr
            raw = first_md / (1 - r3) + first_mm / (1 - r4)
            bounds[i][j] = raw
            rows.append(
                {
                    "ell": ell,
                    "ell_prime": prime,
                    "first_md_increment_upper_bound": first_md,
                    "first_mm_increment_upper_bound": first_mm,
                    "block_frobenius_tail_upper_bound": raw,
                }
            )
    row_sum = max(sum(row, Fraction(0)) for row in bounds)
    return {
        "h": h,
        "q": q,
        "r3": r3,
        "r4": r4,
        "rows": rows,
        "row_sum": row_sum,
    }


def _verify_tail(section: dict[str, Any]) -> None:
    if section.get("start_level") != ANCHOR_LEVEL:
        raise VerificationError("tail start mismatch")
    replay = _replay_tail(ANCHOR_LEVEL)
    base = section.get("base_edge_rational_proof", {})
    angle = _fraction(base["angle_upper_bound"])
    cosine_upper = _fraction(
        base["cosine_alternating_upper_at_angle_bound"]
    )
    cosine_square = _fraction(base["upper_polynomial_square"])
    comparison = _fraction(base["comparison_target_one_fifth"])
    expected_cosine_upper = Fraction(1) - angle**2 / 2 + angle**4 / 24
    if (
        angle != Fraction(10, 9)
        or cosine_upper != expected_cosine_upper
        or cosine_square != cosine_upper**2
        or comparison != Fraction(1, 5)
        or not cosine_square < comparison
        or base.get("upper_polynomial_square_below_one_fifth") is not True
        or base.get("angle_bound_below_pi_over_two_from_pi_gt_three") is not True
        or base.get("conclusion") != "acos(1/sqrt(5)) < 10/9"
    ):
        raise VerificationError("base edge rational proof mismatch")
    if (
        _fraction(section["maximum_edge_upper_bound_radians"]) != replay["h"]
        or _fraction(section["future_edge_contraction_upper_bound"]) != replay["q"]
        or _fraction(section["md_geometric_ratio_upper_bound"]) != replay["r3"]
        or _fraction(section["mm_geometric_ratio_upper_bound"]) != replay["r4"]
    ):
        raise VerificationError("rational contraction data mismatch")
    stored_rows = section.get("block_rows")
    if not isinstance(stored_rows, list) or len(stored_rows) != 49:
        raise VerificationError("full block census mismatch")
    for stored, rebuilt in zip(stored_rows, replay["rows"], strict=True):
        if stored.get("ell") != rebuilt["ell"] or stored.get("ell_prime") != rebuilt["ell_prime"]:
            raise VerificationError("block label mismatch")
        for key in (
            "first_md_increment_upper_bound",
            "first_mm_increment_upper_bound",
            "block_frobenius_tail_upper_bound",
        ):
            if _fraction(stored[key]) != rebuilt[key]:
                raise VerificationError(f"tail {key} mismatch")
    if (
        _fraction(
            section[
                "full_matrix_operator_tail_upper_bound_via_symmetric_block_row_sum"
            ]
        )
        != replay["row_sum"]
        or section.get("operator_tail_bound_is_exact_rational_upper_arithmetic")
        is not True
        or section.get("full_raw_stiffness_cauchy_limit_exists") is not True
    ):
        raise VerificationError("Cauchy flag missing")


def _replay_anchor() -> dict[str, Any]:
    mesh = build_geodesic_icosahedral_tower(ANCHOR_LEVEL).levels[ANCHOR_LEVEL]
    design = _harmonic_design(mesh.vertices)
    raw = design[:, LOW_MODE_DIMENSION:]
    weights = _vertex_area_weights(mesh)
    projected, removal = _weighted_low_mode_removal(design, weights)
    stiffness = _equal_seam_graph_stiffness(mesh)
    k_raw = raw.conj().T @ (stiffness @ raw)
    k_projected = projected.conj().T @ (stiffness @ projected)
    k_raw = (k_raw + k_raw.conj().T) / 2.0
    k_projected = (k_projected + k_projected.conj().T) / 2.0
    raw_values = np.linalg.eigvalsh(k_raw)
    projected_values = np.linalg.eigvalsh(k_projected)
    low = design[:, :LOW_MODE_DIMENSION]
    low_gram = low.conj().T @ (weights[:, None] * low)
    overlap = low.conj().T @ (weights[:, None] * raw)
    coefficients = np.linalg.solve(low_gram, overlap)
    inverse_rows, _ = _biposh_rows(np.linalg.inv(k_raw))
    stiffness_rows, _ = _biposh_rows(k_raw)
    return {
        "raw_min": float(raw_values[0]),
        "raw_max": float(raw_values[-1]),
        "projected_min": float(projected_values[0]),
        "projected_max": float(projected_values[-1]),
        "raw_inverse": _biposh_summary(inverse_rows)[
            "primary_amplitude_free_statistic"
        ],
        "raw_stiffness": _biposh_summary(stiffness_rows)[
            "primary_amplitude_free_statistic"
        ],
        "projection_op": float(np.linalg.norm(coefficients, 2)),
        "projection_fro": float(np.linalg.norm(coefficients)),
        "form_diff_op": float(np.linalg.norm(k_raw - k_projected, 2)),
        "form_diff_fro": float(np.linalg.norm(k_raw - k_projected)),
        "removal": removal,
    }


def _verify_anchor(section: dict[str, Any]) -> None:
    replay = _replay_anchor()
    fields = {
        "raw_stiffness_minimum_eigenvalue": "raw_min",
        "raw_stiffness_maximum_eigenvalue": "raw_max",
        "projected_stiffness_minimum_eigenvalue": "projected_min",
        "projected_stiffness_maximum_eigenvalue": "projected_max",
        "raw_inverse_primary": "raw_inverse",
        "raw_stiffness_primary": "raw_stiffness",
    }
    for stored, rebuilt in fields.items():
        _assert_close(section[stored], replay[rebuilt], 3.0e-9, stored)
    diagnostics = section["projection_diagnostics"]
    for stored, rebuilt in (
        ("low_to_active_projection_coefficient_operator_norm", "projection_op"),
        ("low_to_active_projection_coefficient_frobenius_norm", "projection_fro"),
        ("raw_minus_projected_stiffness_operator_norm", "form_diff_op"),
        ("raw_minus_projected_stiffness_frobenius_norm", "form_diff_fro"),
    ):
        _assert_close(diagnostics[stored], replay[rebuilt], 3.0e-9, stored)
    if (
        section.get("raw_stiffness_positive") is not True
        or section.get("projected_stiffness_positive") is not True
        or section.get("finite_eigensolver_is_interval_certified") is not False
    ):
        raise VerificationError("finite anchor disclosure mismatch")


def _cotangent_stiffness(mesh: Any) -> Any:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    points = np.asarray(mesh.vertices, dtype=float)[faces]
    cotangents = []
    for vertex in range(3):
        first = points[:, (vertex + 1) % 3] - points[:, vertex]
        second = points[:, (vertex + 2) % 3] - points[:, vertex]
        cotangents.append(
            np.einsum("ij,ij->i", first, second)
            / np.linalg.norm(np.cross(first, second), axis=1)
        )
    weights: dict[tuple[int, int], float] = {}
    for opposite, slots in enumerate(((1, 2), (2, 0), (0, 1))):
        for face_index, face in enumerate(faces):
            left, right = int(face[slots[0]]), int(face[slots[1]])
            edge = (min(left, right), max(left, right))
            weights[edge] = weights.get(edge, 0.0) + 0.5 * float(
                cotangents[opposite][face_index]
            )
    diagonal = np.zeros(mesh.vertex_count, dtype=float)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for (left, right), weight in weights.items():
        diagonal[left] += weight
        diagonal[right] += weight
        rows += [left, right]
        columns += [right, left]
        values += [-weight, -weight]
    rows += list(range(mesh.vertex_count))
    columns += list(range(mesh.vertex_count))
    values += diagonal.tolist()
    return coo_matrix(
        (values, (rows, columns)),
        shape=(mesh.vertex_count, mesh.vertex_count),
    ).tocsr()


def _verify_geometry(section: dict[str, Any], parent: dict[str, Any]) -> None:
    tower = build_geodesic_icosahedral_tower(ANCHOR_LEVEL)
    stored_shapes = section["finite_shape_regular_diagnostic"]["rows"]
    if len(stored_shapes) != ANCHOR_LEVEL + 1:
        raise VerificationError("shape diagnostic census mismatch")
    for stored, mesh in zip(stored_shapes, tower.levels, strict=True):
        points = np.asarray(mesh.vertices, dtype=float)[mesh.faces]
        sides = [
            np.linalg.norm(points[:, 1] - points[:, 2], axis=1),
            np.linalg.norm(points[:, 2] - points[:, 0], axis=1),
            np.linalg.norm(points[:, 0] - points[:, 1], axis=1),
        ]
        a, b, c = sides
        first = np.arccos(
            np.clip((b * b + c * c - a * a) / (2 * b * c), -1, 1)
        )
        second = np.arccos(
            np.clip((c * c + a * a - b * b) / (2 * c * a), -1, 1)
        )
        angles = np.column_stack((first, second, math.pi - first - second))
        if stored.get("level") != mesh.level:
            raise VerificationError("shape diagnostic level mismatch")
        _assert_close(
            stored["minimum_chord_triangle_angle_degrees"],
            np.degrees(np.min(angles)),
            2.0e-11,
            "minimum chord angle",
        )
        _assert_close(
            stored["maximum_chord_triangle_angle_degrees"],
            np.degrees(np.max(angles)),
            2.0e-11,
            "maximum chord angle",
        )
        ratio = np.max(np.maximum.reduce(sides) / np.minimum.reduce(sides))
        _assert_close(
            stored["maximum_to_minimum_chord_edge_ratio"],
            ratio,
            2.0e-12,
            "chord edge ratio",
        )
    shape = section["finite_shape_regular_diagnostic"]
    if (
        shape.get("all_level_uniform_shape_regularity_proved") is not False
        or shape.get("binary64_rows_are_a_theorem") is not False
    ):
        raise VerificationError("shape diagnostic promoted")

    stored_comparison = section["equal_counting_vs_cotangent_diagnostic"]
    rows = stored_comparison["rows"]
    if len(rows) != 5:
        raise VerificationError("weight comparison census mismatch")
    for stored, level in zip(rows, range(2, 7), strict=True):
        mesh = tower.levels[level]
        active = _harmonic_design(mesh.vertices)[:, LOW_MODE_DIMENSION:]
        equal_form = active.conj().T @ (_equal_seam_graph_stiffness(mesh) @ active)
        cotangent_form = active.conj().T @ (_cotangent_stiffness(mesh) @ active)
        equal_rows, _ = _biposh_rows(equal_form)
        cotangent_rows, _ = _biposh_rows(cotangent_form)
        if stored.get("level") != level:
            raise VerificationError("weight comparison level mismatch")
        _assert_close(
            stored["equal_counting_primary_statistic"],
            _biposh_summary(equal_rows)["primary_amplitude_free_statistic"],
            3.0e-10,
            "equal-counting primary",
        )
        _assert_close(
            stored["cotangent_fem_primary_statistic"],
            _biposh_summary(cotangent_rows)["primary_amplitude_free_statistic"],
            3.0e-10,
            "cotangent primary",
        )
    if (
        stored_comparison.get("equal_counting_is_the_declared_repair_measure")
        is not True
        or stored_comparison.get("cotangent_weights_are_the_declared_repair_measure")
        is not False
        or stored_comparison.get("finite_binary64_contrast_is_a_continuum_proof")
        is not False
    ):
        raise VerificationError("weight comparison boundary mismatch")

    parent_interval = parent["conditional_continuum_interval"]
    stored_interval = section["parent_equal_counting_continuum_interval"]
    if (
        stored_interval.get("primary_amplitude_free_statistic_interval")
        != parent_interval["primary_amplitude_free_statistic_interval"]
        or stored_interval.get("conditional_interval_excludes_zero")
        is not parent_interval["conditional_interval_excludes_zero"]
    ):
        raise VerificationError("parent continuum interval mismatch")
    route = section["shape_regular_coercivity_route"]
    if (
        route.get("candidate_route_identified") is not True
        or route.get("not_excluded_by_current_results") is not True
        or route.get("closed_here") is not False
        or route.get("does_not_force_so3_isotropy_or_zero_l6") is not True
        or len(route.get("missing_uniform_geometry_theorems", [])) != 3
    ):
        raise VerificationError("coercivity route boundary mismatch")


def _verify_counterexamples(section: dict[str, Any]) -> None:
    inverse = section["inverse_discontinuity_without_uniform_coercivity"]
    rows = inverse.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:
        raise VerificationError("inverse counterexample census mismatch")
    for row in rows:
        n = int(row["n"])
        _assert_close(row["K_diagonal"], [1.0, 1.0 / (n + 1)], 0.0, "K_n")
        _assert_close(row["K_inverse_diagonal"], [1.0, float(n + 1)], 0.0, "K_n inverse")
    if (
        inverse.get("every_finite_K_n_positive_definite") is not True
        or inverse.get("full_matrix_limit_exists") is not True
        or inverse.get("inverse_family_bounded") is not False
    ):
        raise VerificationError("inverse counterexample boundary mismatch")
    mixing = section["multiplicity_space_radial_mixing"]
    first = abs(1.0) / math.sqrt(1.0 * 1.0)
    mixed = abs(1.0) / math.sqrt(2.0 * 2.0)
    _assert_close(mixing["first_statistic"], first, 0.0, "first readout")
    _assert_close(mixing["mixed_statistic"], mixed, 0.0, "mixed readout")
    if mixing.get("rotation_equivariance_forces_scalar_on_copy_space") is not False:
        raise VerificationError("copy-space boundary promoted")


def verify_packet(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    packet = json.loads(path.read_text(encoding="utf-8"))
    if packet.get("schema") != EXPECTED_SCHEMA or packet.get("status") != EXPECTED_STATUS:
        raise VerificationError("schema or status mismatch")
    _check_hash_and_pins(packet)
    current_parents: dict[str, dict[str, Any]] = {}
    for name, parent in packet.get("parents", {}).items():
        current = json.loads((ROOT / parent["path"]).read_text(encoding="utf-8"))
        if (
            current.get("schema") != parent.get("schema")
            or current.get("status") != parent.get("status")
            or current.get("payload_sha256") != parent.get("payload_sha256")
        ):
            raise VerificationError("parent pin mismatch")
        current_parents[name] = current
    _verify_tail(packet["full_raw_stiffness_tail"])
    _verify_anchor(packet["finite_anchor"])
    _verify_geometry(
        packet["continuum_geometry_assessment"],
        current_parents["continuum_tail"],
    )
    _verify_counterexamples(packet["exact_counterexamples"])

    admission = packet["inverse_admission_gate"]
    epsilon = packet["full_raw_stiffness_tail"][
        "full_matrix_operator_tail_upper_bound_via_symmetric_block_row_sum"
    ]["binary64_upper"]
    gamma = packet["finite_anchor"]["raw_stiffness_minimum_eigenvalue"]
    _assert_close(
        admission["epsilon_divided_by_finite_anchor_gap"],
        epsilon / gamma,
        3.0e-12,
        "Neumann ratio",
    )
    if (
        admission.get("finite_anchor_neumann_gate_epsilon_lt_gap")
        is not (epsilon < gamma)
        or admission.get("finite_anchor_gap_is_continuum_coercivity_certificate")
        is not False
        or admission.get("full_raw_inverse_continuum_tail_certified") is not False
        or admission.get("projected_inverse_continuum_tail_certified") is not False
    ):
        raise VerificationError("inverse admission gate mismatch")

    required_false = (
        "uniform_continuum_coercivity",
        "projected_quotient_continuum_tail",
        "full_inverse_covariance_continuum_limit",
        "source_ensemble_selected",
        "all_level_stiffness_response_source_selected",
        "physical_response_readout_selected",
        "physical_covariance_selected",
        "physical_prediction",
        "promotion_allowed",
    )
    decision = packet["selection_decision"]
    if decision.get("full_raw_stiffness_cauchy_limit") is not True or any(
        decision.get(key) is not False for key in required_false
    ):
        raise VerificationError("selection boundary mismatch")
    response = packet["operational_stiffness_response"]
    if (
        response.get("stiffness_statistic_can_be_an_operational_response_observable")
        is not True
        or response.get(
            "declared_registered_ladder_primitive_alphabet_source_emitted"
        )
        is not True
        or response.get(
            "declared_registered_ladder_unit_counting_source_emitted"
        )
        is not True
        or response.get("declared_registered_geometry_levels")
        != [0, 1, 2, 3, 4, 5]
        or response.get("inverse_anchor_level") != ANCHOR_LEVEL
        or response.get("declared_ladder_reaches_inverse_anchor_level") is not False
        or response.get("first_order_refinement_readback_discharged") is not True
        or response.get("full_refinement_commuting_diagram_discharged") is not False
        or response.get("physical_repair_law_selected") is not False
        or response.get("all_level_response_law_source_selected") is not False
        or response.get("response_readout_physically_attached") is not False
        or response.get("stiffness_statistic_is_a_current_physical_prediction")
        is not False
    ):
        raise VerificationError("operational response boundary mismatch")
    transfer = packet["transfer_boundary"]
    if (
        transfer.get("scalar_rescaling_cancellation_proved") is not True
        or transfer.get("rotation_equivariant_transfer_proved") is not False
        or transfer.get("multiplicity_one_proved") is not False
        or transfer.get("radial_copy_mixing_excluded") is not False
    ):
        raise VerificationError("transfer boundary mismatch")
    verification_scope = packet["verification_scope"]
    if verification_scope != {
        "exact_tail_arithmetic_reimplemented": True,
        "registered_mesh_builder_shared": True,
        "harmonic_design_stiffness_and_biposh_kernels_shared": True,
        "independent_harmonic_implementation": False,
        "classification": (
            "correlated-kernel replay with separately implemented certificate "
            "and boundary checks"
        ),
    }:
        raise VerificationError("verification-scope disclosure mismatch")
    lean = packet["lean_boundary"]
    if (
        lean.get("path") != "Lean/Screen/BipoSHInverseBoundary.lean"
        or lean.get("required_declarations") != EXPECTED_LEAN_DECLARATIONS
        or lean.get("stiffness_readback_theorem")
        != "stiffness_recovered_from_uniform_repair"
        or lean.get("inverse_counterexample_theorems")
        != [
            "collapsing_stiffness_positive",
            "collapsing_stiffness_arbitrarily_small",
            "inverse_nuisance_unbounded",
        ]
        or lean.get("copy_mixing_counterexample_theorem")
        != "copy_mixing_changes_statistic"
    ):
        raise VerificationError("Lean declaration boundary mismatch")
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    packet = verify_packet(args.receipt)
    print("A5_BIPOSH_INVERSE_CONTINUUM_GATE_VERIFIED")
    print(packet["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
