"""Independent verifier for the issue-659 dual-operator BipoSH packet.

This module intentionally does not import the producer.  It reconstructs the
mesh quadrature, low-mode projection, both stiffness matrices, the complete
BipoSH transform, and the spectral/projector diagnostics from the pinned
icosahedral geometry implementation.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.linalg import eigh, null_space
from scipy.optimize import linear_sum_assignment
from scipy.special import sph_harm_y

from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / "data/refinement/a5_biposh_dual_operator_receipt.json"
DEFAULT_COEFFICIENTS = (
    ROOT / "data/refinement/a5_biposh_dual_operator_coefficients.json"
)
ELL_MIN = 2
ELL_MAX = 8
LOW_DIMENSION = 4
ACTIVE_DIMENSION = 77
FLOAT_ATOL = 2.0e-9
FLOAT_RTOL = 2.0e-10
EXPECTED_REPAIR_STATUS = (
    "BOUNDED_EXPECTATION_LEVEL_REPAIR_FIXED_POINT_UNIQUE_MODULO_CLOCK_IN_THE_"
    "FROZEN_ADVERSARIAL_SUITE"
)
EXPECTED_REPAIR_PAYLOAD_SHA256 = (
    "sha256:9e87c5e4abfb3baed80058ffc832a6dbd3412f386eb383d68fee4ebee10c00d5"
)


class VerificationError(ValueError):
    pass


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


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=FLOAT_RTOL,
        abs_tol=FLOAT_ATOL,
    ):
        raise VerificationError(f"{label}: {actual!r} != {expected!r}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"{path} is not a JSON object")
    return value


def _labels() -> list[tuple[int, int]]:
    return [
        (ell, m)
        for ell in range(ELL_MIN, ELL_MAX + 1)
        for m in range(-ell, ell + 1)
    ]


def _offsets() -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    start = 0
    for ell in range(ELL_MIN, ELL_MAX + 1):
        result[ell] = (start, start + 2 * ell + 1)
        start += 2 * ell + 1
    return result


def _quadrature(mesh) -> np.ndarray:
    result = np.zeros(mesh.vertex_count, dtype=float)
    for face_id in range(mesh.face_count):
        share = float(mesh.spherical_face_areas[face_id]) / 3.0
        for vertex in mesh.faces[face_id]:
            result[int(vertex)] += share
    return result


def _design(mesh) -> np.ndarray:
    points = np.asarray(mesh.vertices, dtype=float)
    polar = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
    azimuth = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * math.pi)
    columns = []
    for ell in range(ELL_MAX + 1):
        for m in range(-ell, ell + 1):
            columns.append(sph_harm_y(ell, m, polar, azimuth))
    return np.asarray(columns, dtype=np.complex128).T


def _project_active(
    design: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    low = design[:, :LOW_DIMENSION]
    high = design[:, LOW_DIMENSION:]
    low_weighted = low.conj().T * weights[None, :]
    coefficients = np.linalg.solve(low_weighted @ low, low_weighted @ high)
    active = high - low @ coefficients
    residual = float(np.max(np.abs(low_weighted @ active)))
    gram = active.conj().T @ (weights[:, None] * active)
    return active, 0.5 * (gram + gram.conj().T), residual


def _raw_graph(mesh) -> sparse.csr_matrix:
    result = sparse.lil_matrix((mesh.vertex_count, mesh.vertex_count), dtype=float)
    for left_value, right_value in mesh.edges:
        left = int(left_value)
        right = int(right_value)
        result[left, left] += 1.0
        result[right, right] += 1.0
        result[left, right] -= 1.0
        result[right, left] -= 1.0
    return result.tocsr()


def _cotangent(mesh) -> sparse.csr_matrix:
    points = np.asarray(mesh.vertices, dtype=float)
    result = sparse.lil_matrix((mesh.vertex_count, mesh.vertex_count), dtype=float)
    for face in mesh.faces:
        a, b, c = (int(value) for value in face)
        for left, right, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            first = points[left] - points[opposite]
            second = points[right] - points[opposite]
            cotangent = float(np.dot(first, second)) / float(
                np.linalg.norm(np.cross(first, second))
            )
            weight = 0.5 * cotangent
            result[left, left] += weight
            result[right, right] += weight
            result[left, right] -= weight
            result[right, left] -= weight
    return result.tocsr()


@lru_cache(maxsize=None)
def _cg(j1: int, m1: int, j2: int, m2: int, J: int, M: int) -> float:
    if M != m1 + m2 or J < abs(j1 - j2) or J > j1 + j2:
        return 0.0
    factorial = math.factorial
    triangle = (
        factorial(J + j1 - j2)
        * factorial(J - j1 + j2)
        * factorial(j1 + j2 - J)
        / factorial(j1 + j2 + J + 1)
    )
    normalization = math.sqrt((2 * J + 1) * triangle)
    normalization *= math.sqrt(
        factorial(J + M)
        * factorial(J - M)
        * factorial(j1 - m1)
        * factorial(j1 + m1)
        * factorial(j2 - m2)
        * factorial(j2 + m2)
    )
    first = max(0, j2 - J - m1, j1 - J + m2)
    last = min(j1 + j2 - J, j1 - m1, j2 + m2)
    terms = []
    for k in range(first, last + 1):
        denominator = math.prod(
            factorial(value)
            for value in (
                k,
                j1 + j2 - J - k,
                j1 - m1 - k,
                j2 + m2 - k,
                J - j2 + m1 + k,
                J - j1 - m2 + k,
            )
        )
        terms.append(((-1) ** k) / denominator)
    return normalization * math.fsum(terms)


def _all_biposh(form: np.ndarray) -> list[list[int | float]]:
    offsets = _offsets()
    output: list[list[int | float]] = []
    for ell in range(ELL_MIN, ELL_MAX + 1):
        left_start = offsets[ell][0]
        for ell_prime in range(ELL_MIN, ELL_MAX + 1):
            right_start = offsets[ell_prime][0]
            for total_l in range(abs(ell - ell_prime), ell + ell_prime + 1):
                for total_m in range(-total_l, total_l + 1):
                    terms: list[complex] = []
                    for m in range(-ell, ell + 1):
                        m_prime = m - total_m
                        if -ell_prime <= m_prime <= ell_prime:
                            terms.append(
                                (-1) ** m_prime
                                * _cg(
                                    ell,
                                    m,
                                    ell_prime,
                                    -m_prime,
                                    total_l,
                                    total_m,
                                )
                                * form[
                                    left_start + m + ell,
                                    right_start + m_prime + ell_prime,
                                ]
                            )
                    value = sum(terms, 0.0j)
                    output.append(
                        [
                            ell,
                            ell_prime,
                            total_l,
                            total_m,
                            float(value.real),
                            float(value.imag),
                        ]
                    )
    return output


def _verify_rows(
    stored: list[list[Any]],
    rebuilt: list[list[int | float]],
    label: str,
) -> dict[str, float]:
    if len(stored) != len(rebuilt) or len(stored) != 5929:
        raise VerificationError(f"{label}: wrong coefficient count")
    maximum_absolute_error = 0.0
    maximum_normalized_error = 0.0
    for index, (left, right) in enumerate(zip(stored, rebuilt, strict=True)):
        if [int(value) for value in left[:4]] != [int(value) for value in right[:4]]:
            raise VerificationError(f"{label}: index drift at row {index}")
        _close(left[4], right[4], f"{label} row {index} real")
        _close(left[5], right[5], f"{label} row {index} imaginary")
        for stored_value, raw_value in zip(left[4:], right[4:], strict=True):
            error = abs(float(stored_value) - float(raw_value))
            maximum_absolute_error = max(maximum_absolute_error, error)
            maximum_normalized_error = max(
                maximum_normalized_error,
                error / max(1.0, abs(float(raw_value))),
            )
    return {
        "maximum_absolute_rounding_error": maximum_absolute_error,
        "maximum_error_divided_by_max_one_abs_raw": maximum_normalized_error,
    }


def _verify_transform_unitarity_and_inverse(
    form: np.ndarray,
    rows: list[list[Any]],
    label: str,
) -> None:
    """Check the CG convention without trusting a second forward transform.

    The BipoSH map must be unitary on every (ell,ell') block.  The inverse
    transform is evaluated explicitly, and the isotropic sign convention is
    checked against A_ll^{00}=(-1)^ell sqrt(2ell+1).
    """

    values = {
        tuple(int(value) for value in row[:4]): complex(float(row[4]), float(row[5]))
        for row in rows
    }
    offsets = _offsets()
    for ell in range(ELL_MIN, ELL_MAX + 1):
        left_start, left_stop = offsets[ell]
        for ell_prime in range(ELL_MIN, ELL_MAX + 1):
            right_start, right_stop = offsets[ell_prime]
            block = form[left_start:left_stop, right_start:right_stop]
            coefficient_power = math.fsum(
                abs(values[(ell, ell_prime, total_l, total_m)]) ** 2
                for total_l in range(abs(ell - ell_prime), ell + ell_prime + 1)
                for total_m in range(-total_l, total_l + 1)
            )
            _close(
                coefficient_power,
                float(np.linalg.norm(block, ord="fro") ** 2),
                f"{label} ({ell},{ell_prime}) transform norm",
            )
            rebuilt = np.zeros_like(block)
            for m in range(-ell, ell + 1):
                for m_prime in range(-ell_prime, ell_prime + 1):
                    total_m = m - m_prime
                    rebuilt[m + ell, m_prime + ell_prime] = sum(
                        (
                            (-1) ** m_prime
                            * _cg(
                                ell,
                                m,
                                ell_prime,
                                -m_prime,
                                total_l,
                                total_m,
                            )
                            * values[(ell, ell_prime, total_l, total_m)]
                            for total_l in range(
                                abs(ell - ell_prime), ell + ell_prime + 1
                            )
                            if abs(total_m) <= total_l
                        ),
                        0.0j,
                    )
            maximum_inverse_residual = float(np.max(np.abs(rebuilt - block)))
            if maximum_inverse_residual > 3.0e-9:
                raise VerificationError(
                    f"{label} ({ell},{ell_prime}) inverse residual "
                    f"{maximum_inverse_residual}"
                )

    for ell in range(ELL_MIN, ELL_MAX + 1):
        isotropic_a00 = math.fsum(
            (-1) ** m
            * _cg(ell, m, ell, -m, 0, 0)
            for m in range(-ell, ell + 1)
        )
        _close(
            isotropic_a00,
            (-1) ** ell * math.sqrt(2 * ell + 1),
            f"CG isotropic convention ell={ell}",
        )


def _summary_from_rows(rows: list[list[Any]]) -> dict[str, Any]:
    values = {
        tuple(int(value) for value in row[:4]): complex(float(row[4]), float(row[5]))
        for row in rows
    }
    vector = [values[(2, 4, 6, m)] for m in range(-6, 7)]
    a22 = values[(2, 2, 0, 0)]
    a44 = values[(4, 4, 0, 0)]
    numerator = float(np.linalg.norm(vector))
    denominator = math.sqrt(abs(a22 * a44))
    power_by_rank: dict[int, float] = {}
    for row in rows:
        rank = int(row[2])
        power_by_rank[rank] = power_by_rank.get(rank, 0.0) + float(row[4]) ** 2 + float(
            row[5]
        ) ** 2
    allowed = {0, 6, 10, 12, 15, 16}
    forbidden_power = sum(
        value for rank, value in power_by_rank.items() if rank not in allowed
    )
    total_power = sum(power_by_rank.values())
    return {
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
        "a22": a22,
        "a44": a44,
        "vector": vector,
        "maximum_forbidden_norm": max(
            math.sqrt(value)
            for rank, value in power_by_rank.items()
            if rank not in allowed
        ),
        "relative_forbidden_norm": math.sqrt(forbidden_power / total_power),
    }


def _verify_spectral(
    form: np.ndarray,
    gram: np.ndarray,
    stored: dict[str, Any],
    label: str,
) -> None:
    gram_values, gram_vectors = eigh(gram)
    half = (gram_vectors * np.sqrt(gram_values)) @ gram_vectors.conj().T
    inverse_half = (
        gram_vectors * (1.0 / np.sqrt(gram_values))
    ) @ gram_vectors.conj().T
    operator = inverse_half @ form @ inverse_half
    operator = 0.5 * (operator + operator.conj().T)
    offsets = _offsets()
    reference: dict[int, np.ndarray] = {}
    for ell, (start, stop) in offsets.items():
        reference[ell] = np.linalg.qr(half[:, start:stop], mode="reduced")[0]
    scale = 6.0 / float(
        np.trace(reference[2].conj().T @ operator @ reference[2]).real / 5.0
    )
    operator *= scale
    eigenvalues, eigenvectors = eigh(operator)
    _close(
        stored["ell_two_mean_eigenvalue_normalization"],
        scale,
        f"{label} normalization",
    )
    _close(
        stored["operator_min_eigenvalue_after_normalization"],
        eigenvalues[0],
        f"{label} min eigenvalue",
    )
    _close(
        stored["operator_max_eigenvalue_after_normalization"],
        eigenvalues[-1],
        f"{label} max eigenvalue",
    )

    stored_rows = stored["rows"]
    if [int(row["ell"]) for row in stored_rows] != list(range(2, 9)):
        raise VerificationError(f"{label}: spectral ell rows are incomplete")
    assigned_all = [
        int(index)
        for row in stored_rows
        for index in row["assigned_eigenvalue_indices"]
    ]
    if sorted(assigned_all) != list(range(ACTIVE_DIMENSION)):
        raise VerificationError(f"{label}: assignment is not a partition")

    slot_labels: list[int] = []
    slot_weights: list[np.ndarray] = []
    for ell in range(2, 9):
        overlap = np.sum(
            np.abs(eigenvectors.conj().T @ reference[ell]) ** 2,
            axis=1,
        )
        for _ in range(2 * ell + 1):
            slot_labels.append(ell)
            slot_weights.append(overlap)
    overlap_matrix = np.column_stack(slot_weights)
    optimal_rows, optimal_columns = linear_sum_assignment(-overlap_matrix)
    optimal_score = float(np.sum(overlap_matrix[optimal_rows, optimal_columns]))
    stored_score = 0.0
    for row in stored_rows:
        ell = int(row["ell"])
        overlap = np.sum(
            np.abs(eigenvectors.conj().T @ reference[ell]) ** 2,
            axis=1,
        )
        stored_score += float(
            np.sum(overlap[np.asarray(row["assigned_eigenvalue_indices"], dtype=int)])
        )
    _close(stored_score, optimal_score, f"{label} assignment optimum")

    for row in stored_rows:
        ell = int(row["ell"])
        basis = reference[ell]
        assigned = np.asarray(row["assigned_eigenvalue_indices"], dtype=int)
        if assigned.size != 2 * ell + 1:
            raise VerificationError(f"{label}: ell={ell} dimension mismatch")
        selected = eigenvectors[:, assigned]
        singular = np.linalg.svd(basis.conj().T @ selected, compute_uv=False)
        sine = math.sqrt(max(0.0, 1.0 - float(np.min(singular)) ** 2))
        complement = null_space(basis.conj().T)
        residual = float(
            np.linalg.norm(complement.conj().T @ operator @ basis, ord=2)
        )
        block_a = eigh(basis.conj().T @ operator @ basis, eigvals_only=True)
        block_d = eigh(
            complement.conj().T @ operator @ complement,
            eigvals_only=True,
        )
        block_gap = float(np.min(np.abs(block_a[:, None] - block_d[None, :])))
        selected_values = eigenvalues[assigned]
        other_values = np.delete(eigenvalues, assigned)
        cluster_gap = float(
            np.min(np.abs(selected_values[:, None] - other_values[None, :]))
        )
        hypothesis = bool(block_gap > 2.0 * residual)
        bound = min(1.0, residual / (block_gap - residual)) if hypothesis else None
        for stored_value, actual_value, suffix in (
            (row["maximum_principal_angle_sine"], sine, "sine"),
            (row["block_off_diagonal_residual_norm"], residual, "residual"),
            (row["unperturbed_block_spectral_gap"], block_gap, "block gap"),
            (row["selected_operator_cluster_gap"], cluster_gap, "cluster gap"),
        ):
            _close(stored_value, actual_value, f"{label} ell={ell} {suffix}")
        if bool(row["davis_kahan_hypothesis_2r_lt_gap"]) != hypothesis:
            raise VerificationError(f"{label}: ell={ell} hypothesis drift")
        if hypothesis:
            _close(
                row["davis_kahan_sine_upper_bound"],
                bound,
                f"{label} ell={ell} bound",
            )
            if sine > float(bound) + 2.0e-9:
                raise VerificationError(f"{label}: ell={ell} bound violated")
        elif row["davis_kahan_sine_upper_bound"] is not None:
            raise VerificationError(f"{label}: ell={ell} invalid bound emitted")


def verify_packet(
    receipt_path: Path = DEFAULT_RECEIPT,
    coefficient_path: Path = DEFAULT_COEFFICIENTS,
    *,
    source_root: Path = ROOT,
) -> dict[str, Any]:
    receipt = _load_json(receipt_path)
    coefficients = _load_json(coefficient_path)
    payload_hash = receipt.get("payload_sha256")
    payload_projection = dict(receipt)
    payload_projection.pop("payload_sha256", None)
    if payload_hash != _sha(_canonical_bytes(payload_projection)):
        raise VerificationError("receipt payload hash mismatch")
    expected_status = (
        "FINITE_DUAL_OPERATOR_FINGERPRINT_ATTAINED__CONTINUUM_RESIDUAL_"
        "AND_PHYSICAL_COVARIANCE_OPEN"
    )
    if receipt.get("schema") != "oph.a5-biposh-dual-operator-refinement.v1":
        raise VerificationError("receipt schema mismatch")
    if receipt.get("status") != expected_status:
        raise VerificationError("receipt status mismatch")
    if coefficients.get("schema") != "oph.a5-biposh-dual-operator-coefficients.v1":
        raise VerificationError("coefficient schema mismatch")
    if coefficients.get("case_count") != 8:
        raise VerificationError("coefficient case count mismatch")

    coefficient_bytes = coefficient_path.read_bytes()
    bundle = receipt["full_coefficient_bundle"]
    if int(bundle["bytes"]) != len(coefficient_bytes):
        raise VerificationError("coefficient byte count mismatch")
    if bundle["sha256"] != _sha(coefficient_bytes):
        raise VerificationError("coefficient bundle hash mismatch")

    for pin in receipt["source_pins"]:
        relative = Path(str(pin["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise VerificationError("unsafe source pin path")
        path = source_root / relative
        payload = path.read_bytes()
        if len(payload) != int(pin["bytes"]) or _sha(payload) != pin["sha256"]:
            raise VerificationError(f"source pin mismatch: {relative.as_posix()}")

    selection = receipt["selection_decision"]
    forbidden_promotions = (
        "equal_seam_operator_source_selected",
        "refinement_tower_equal_seam_extension_source_selected",
        "physical_repair_law_selected",
        "physical_covariance_selected",
        "physical_release_ensemble_selected",
        "global_frame_quotient_visible",
        "screen_to_sky_readout_selected",
        "continuum_residual_decided",
        "physical_prediction",
        "promotion_allowed",
    )
    if any(selection.get(key) is not False for key in forbidden_promotions):
        raise VerificationError("a forbidden physical promotion is enabled")
    if selection.get("base_equal_seam_operator_bounded_reconstructed") is not True:
        raise VerificationError("bounded base equal-seam reconstruction is missing")
    if receipt["source_scope"].get("external_comparison_data_used") is not False:
        raise VerificationError("comparison data flag is not false")

    cases = {
        (int(case["level"]), str(case["operator_id"])): case
        for case in coefficients["cases"]
    }
    if len(cases) != 8:
        raise VerificationError("duplicate or missing coefficient cases")
    tower = build_geodesic_icosahedral_tower(5)
    bridge = receipt.get("bounded_repair_generator_bridge", {})
    repair_path = source_root / str(bridge.get("parent_receipt", ""))
    repair = _load_json(repair_path)
    mean_bridge = repair.get("exact_conditional_mean_bridge", {})
    one_atom = mean_bridge.get("one_atom_restriction", {})
    base_degree = np.bincount(
        tower.levels[0].edges.reshape(-1),
        minlength=tower.levels[0].vertex_count,
    )
    bridge_passed = bool(
        repair.get("schema") == "oph.bounded_atomic_self_readback_closure.v1"
        and repair.get("status") == EXPECTED_REPAIR_STATUS
        and repair.get("certificate_payload_sha256")
        == EXPECTED_REPAIR_PAYLOAD_SHA256
        and repair.get("PHYSICAL_REPAIR_LAW_RECEIPT") is False
        and mean_bridge.get("identity")
        == "E[X_next | X=x] = (I - L_icosahedron/60) x"
        and mean_bridge.get("all_probed_states_exact_identity_verified") is True
        and one_atom.get("one_atom_generator") == "-L_icosahedron/60"
        and one_atom.get("exact_identity_verified") is True
        and one_atom.get("physical_time_scale_selected") is False
        and tower.levels[0].vertex_count == 12
        and tower.levels[0].edge_count == 30
        and np.all(base_degree == 5)
    )
    if (
        bridge.get(
            "base_carrier_operator_matches_bounded_reconstructed_one_atom_mean_generator_up_to_scale"
        )
        is not True
        or not bridge_passed
        or bridge.get("parent_status") != EXPECTED_REPAIR_STATUS
        or bridge.get("parent_certificate_payload_sha256")
        != EXPECTED_REPAIR_PAYLOAD_SHA256
        or bridge.get("refinement_tower_extension_source_selected") is not False
        or bridge.get("global_a1_a3_policy_uniqueness_receipt") is not False
        or bridge.get("physical_repair_law_receipt") is not False
        or bridge.get("physical_time_scale_selected") is not False
    ):
        raise VerificationError("bounded repair generator bridge mismatch")
    level_rows = {int(row["level"]): row for row in receipt["level_rows"]}
    if sorted(level_rows) != list(range(6)):
        raise VerificationError("level rows must cover 0 through 5")

    checked_cases = 0
    for mesh in tower.levels:
        stored_level = level_rows[mesh.level]
        if stored_level["geometry_hash"] != mesh.geometry_hash:
            raise VerificationError(f"level {mesh.level}: geometry hash mismatch")
        design = _design(mesh)
        design_rank = int(np.linalg.matrix_rank(design, tol=2.0e-12))
        if int(stored_level["sampled_harmonic_design_rank"]) != design_rank:
            raise VerificationError(f"level {mesh.level}: rank mismatch")
        if mesh.level < 2:
            if stored_level.get("calculation_performed") is not False:
                raise VerificationError(f"level {mesh.level}: invalid coarse calculation")
            continue

        weights = _quadrature(mesh)
        active, gram, removal_residual = _project_active(design, weights)
        _close(
            stored_level["low_mode_removal"][
                "weighted_low_mode_removal_max_abs_residual"
            ],
            removal_residual,
            f"level {mesh.level} low-mode residual",
        )
        stored_operators = {
            row["operator_id"]: row for row in stored_level["operator_cases"]
        }
        for operator_id, stiffness in (
            ("equal_seam_raw_graph_laplacian", _raw_graph(mesh)),
            ("geometric_cotangent_control", _cotangent(mesh)),
        ):
            form = active.conj().T @ (stiffness @ active)
            form = 0.5 * (form + form.conj().T)
            rebuilt = _all_biposh(form)
            stored_case = cases[(mesh.level, operator_id)]
            case_payload = {
                "level": mesh.level,
                "operator_id": operator_id,
                "index_fields": stored_case["index_fields"],
                "rows": stored_case["rows"],
            }
            stored_operator = stored_operators[operator_id]
            if stored_operator["coefficient_case_sha256"] != _sha(
                _canonical_bytes(case_payload)
            ):
                raise VerificationError(
                    f"level {mesh.level} {operator_id}: case hash mismatch"
                )
            serialization_error = _verify_rows(
                stored_case["rows"],
                rebuilt,
                f"level {mesh.level} {operator_id}",
            )
            stored_error = stored_operator["coefficient_serialization_error"]
            _close(
                stored_error["maximum_absolute_rounding_error"],
                serialization_error["maximum_absolute_rounding_error"],
                f"level {mesh.level} {operator_id} serialization absolute error",
            )
            _close(
                stored_error["maximum_error_divided_by_max_one_abs_raw"],
                serialization_error["maximum_error_divided_by_max_one_abs_raw"],
                f"level {mesh.level} {operator_id} serialization normalized error",
            )
            if serialization_error[
                "maximum_error_divided_by_max_one_abs_raw"
            ] > float(stored_error["normalized_error_gate"]):
                raise VerificationError(
                    f"level {mesh.level} {operator_id}: serialization gate failed"
                )
            _verify_transform_unitarity_and_inverse(
                form,
                stored_case["rows"],
                f"level {mesh.level} {operator_id}",
            )
            summary = _summary_from_rows(rebuilt)
            stored_summary = stored_operator["biposh_summary"]
            _close(
                stored_summary["primary_amplitude_free_statistic"],
                summary["value"],
                f"level {mesh.level} {operator_id} primary",
            )
            _close(
                stored_summary["primary_numerator_norm"],
                summary["numerator"],
                f"level {mesh.level} {operator_id} numerator",
            )
            _close(
                stored_summary["primary_denominator"],
                summary["denominator"],
                f"level {mesh.level} {operator_id} denominator",
            )
            leakage = stored_summary["a5_selection_leakage"]
            _close(
                leakage["maximum_forbidden_total_L_norm"],
                summary["maximum_forbidden_norm"],
                f"level {mesh.level} {operator_id} maximum forbidden leakage",
            )
            _close(
                leakage["relative_forbidden_frobenius_norm"],
                summary["relative_forbidden_norm"],
                f"level {mesh.level} {operator_id} relative forbidden leakage",
            )
            if leakage["gate_passed"] is not (
                summary["maximum_forbidden_norm"] < float(leakage["numerical_gate"])
            ):
                raise VerificationError(
                    f"level {mesh.level} {operator_id}: leakage gate drift"
                )
            _verify_spectral(
                form,
                gram,
                stored_operator["spectral_projector_diagnostics"],
                f"level {mesh.level} {operator_id}",
            )
            checked_cases += 1

    return {
        "schema": "oph.a5-biposh-dual-operator-independent-verification.v1",
        "status": "PASS",
        "checked_levels": 6,
        "checked_operator_cases": checked_cases,
        "checked_full_biposh_coefficients": checked_cases * 5929,
        "comparison_data_used": False,
        "physical_promotion_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS)
    args = parser.parse_args()
    result = verify_packet(args.receipt, args.coefficients)
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
