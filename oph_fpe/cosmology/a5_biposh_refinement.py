"""Target-blind dual-operator BipoSH fingerprint on the icosahedral tower.

The packet produced here is deliberately narrower than a cosmological
covariance calculation.  It compares two finite stiffness forms on the
registered geodesic icosahedral vertex tower:

* the unweighted, equal-seam combinatorial graph Laplacian; and
* the geometric cotangent finite-element stiffness form.

Both forms are compressed into sampled spherical harmonics with ell=2,...,8
after weighted removal of ell=0,1.  The resulting matrices have a complete
bipolar-spherical-harmonic (BipoSH) decomposition.  No stochastic ensemble,
Green kernel, heat time, sky map, or comparison datum is introduced.

Levels zero and one are retained as explicit rank controls.  Their 12 and 42
vertices cannot resolve the 81 spherical harmonics through ell=8, so the
operator calculation starts at level two rather than silently truncating the
requested band.
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
BOUNDED_REPAIR_RECEIPT = (
    ROOT / "data/repair_closure/bounded_atomic_self_readback_closure_receipt.json"
)
EXPECTED_BOUNDED_REPAIR_STATUS = (
    "BOUNDED_EXPECTATION_LEVEL_REPAIR_FIXED_POINT_UNIQUE_MODULO_CLOCK_IN_THE_"
    "FROZEN_ADVERSARIAL_SUITE"
)
EXPECTED_BOUNDED_REPAIR_PAYLOAD_SHA256 = (
    "sha256:9e87c5e4abfb3baed80058ffc832a6dbd3412f386eb383d68fee4ebee10c00d5"
)

ELL_MIN = 2
ELL_MAX = 8
FULL_HARMONIC_DIMENSION = (ELL_MAX + 1) ** 2
LOW_MODE_DIMENSION = ELL_MIN**2
ACTIVE_DIMENSION = FULL_HARMONIC_DIMENSION - LOW_MODE_DIMENSION
PRIMARY_PAIR = (2, 4, 6)
ALLOWED_A5_TOTAL_L = (0, 6, 10, 12, 15, 16)
SERIALIZED_SIGNIFICANT_DIGITS = 12


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_pin(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _complex_pair(value: complex) -> list[float]:
    return [float(np.real(value)), float(np.imag(value))]


def _serialized_float(value: float) -> float:
    """Quantize coefficient payloads under a platform-neutral decimal contract."""

    quantized = float(f"{float(value):.{SERIALIZED_SIGNIFICANT_DIGITS - 1}e}")
    return 0.0 if quantized == 0.0 else quantized


def _harmonic_labels(ell_min: int = ELL_MIN, ell_max: int = ELL_MAX) -> list[tuple[int, int]]:
    return [
        (ell, m)
        for ell in range(int(ell_min), int(ell_max) + 1)
        for m in range(-ell, ell + 1)
    ]


def _harmonic_offsets() -> dict[int, tuple[int, int]]:
    offsets: dict[int, tuple[int, int]] = {}
    start = 0
    for ell in range(ELL_MIN, ELL_MAX + 1):
        stop = start + 2 * ell + 1
        offsets[ell] = (start, stop)
        start = stop
    return offsets


def _vertex_area_weights(mesh) -> np.ndarray:
    weights = np.zeros(mesh.vertex_count, dtype=float)
    for face, area in zip(mesh.faces, mesh.spherical_face_areas, strict=True):
        weights[np.asarray(face, dtype=np.int64)] += float(area) / 3.0
    return weights


def _harmonic_design(points: np.ndarray) -> np.ndarray:
    theta = np.arccos(np.clip(points[:, 2].astype(float), -1.0, 1.0))
    phi = np.mod(
        np.arctan2(points[:, 1].astype(float), points[:, 0].astype(float)),
        2.0 * math.pi,
    )
    return np.column_stack(
        [
            sph_harm_y(ell, m, theta, phi)
            for ell in range(0, ELL_MAX + 1)
            for m in range(-ell, ell + 1)
        ]
    )


def _weighted_low_mode_removal(
    design: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    low = design[:, :LOW_MODE_DIMENSION]
    active = design[:, LOW_MODE_DIMENSION:]
    low_gram = low.conj().T @ (weights[:, None] * low)
    overlap = low.conj().T @ (weights[:, None] * active)
    projected = active - low @ np.linalg.solve(low_gram, overlap)
    residual = low.conj().T @ (weights[:, None] * projected)
    return projected, {
        "weighted_low_mode_removal_max_abs_residual": float(
            np.max(np.abs(residual))
        ),
        "low_mode_gram_min_eigenvalue": float(
            np.min(eigh(low_gram, eigvals_only=True))
        ),
    }


def _equal_seam_graph_stiffness(mesh) -> sparse.csr_matrix:
    left = np.asarray(mesh.edges[:, 0], dtype=np.int64)
    right = np.asarray(mesh.edges[:, 1], dtype=np.int64)
    row = np.concatenate((left, right))
    column = np.concatenate((right, left))
    adjacency = sparse.coo_matrix(
        (np.ones(row.size, dtype=float), (row, column)),
        shape=(mesh.vertex_count, mesh.vertex_count),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    return sparse.diags(degree, format="csr") - adjacency


def _cotangent_stiffness(mesh) -> sparse.csr_matrix:
    """Return the chord-triangle cotangent FEM stiffness matrix."""

    edge_weights: dict[tuple[int, int], float] = {}
    points = np.asarray(mesh.vertices, dtype=float)
    for face in mesh.faces:
        a, b, c = (int(value) for value in face)
        for left, right, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            first = points[left] - points[opposite]
            second = points[right] - points[opposite]
            cross_norm = float(np.linalg.norm(np.cross(first, second)))
            if cross_norm <= 0.0:
                raise ValueError("degenerate triangle in cotangent stiffness")
            contribution = 0.5 * float(np.dot(first, second)) / cross_norm
            key = (min(left, right), max(left, right))
            edge_weights[key] = edge_weights.get(key, 0.0) + contribution

    diagonal = np.zeros(mesh.vertex_count, dtype=float)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for (left, right), weight in sorted(edge_weights.items()):
        rows.extend((left, right))
        columns.extend((right, left))
        values.extend((-weight, -weight))
        diagonal[left] += weight
        diagonal[right] += weight
    rows.extend(range(mesh.vertex_count))
    columns.extend(range(mesh.vertex_count))
    values.extend(float(value) for value in diagonal)
    return sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(mesh.vertex_count, mesh.vertex_count),
    ).tocsr()


@lru_cache(maxsize=None)
def _clebsch_gordan(
    j1: int,
    m1: int,
    j2: int,
    m2: int,
    total_j: int,
    total_m: int,
) -> float:
    """Integer-spin Clebsch-Gordan coefficient via the factorial formula."""

    if (
        total_m != m1 + m2
        or total_j < abs(j1 - j2)
        or total_j > j1 + j2
        or abs(m1) > j1
        or abs(m2) > j2
        or abs(total_m) > total_j
    ):
        return 0.0
    factorial = math.factorial
    prefactor = math.sqrt(
        (2 * total_j + 1)
        * factorial(total_j + j1 - j2)
        * factorial(total_j - j1 + j2)
        * factorial(j1 + j2 - total_j)
        / factorial(j1 + j2 + total_j + 1)
    )
    prefactor *= math.sqrt(
        factorial(total_j + total_m)
        * factorial(total_j - total_m)
        * factorial(j1 - m1)
        * factorial(j1 + m1)
        * factorial(j2 - m2)
        * factorial(j2 + m2)
    )
    lower = max(0, j2 - total_j - m1, j1 - total_j + m2)
    upper = min(j1 + j2 - total_j, j1 - m1, j2 + m2)
    series = 0.0
    for index in range(lower, upper + 1):
        arguments = (
            index,
            j1 + j2 - total_j - index,
            j1 - m1 - index,
            j2 + m2 - index,
            total_j - j2 + m1 + index,
            total_j - j1 - m2 + index,
        )
        series += (-1) ** index / math.prod(
            factorial(argument) for argument in arguments
        )
    return prefactor * series


def _biposh_rows(
    matrix: np.ndarray,
) -> tuple[list[list[int | float]], dict[str, float]]:
    """Transform a Hermitian harmonic-coefficient form into full BipoSH rows."""

    offsets = _harmonic_offsets()
    rows: list[list[int | float]] = []
    maximum_absolute_error = 0.0
    maximum_normalized_error = 0.0
    for ell in range(ELL_MIN, ELL_MAX + 1):
        ell_start, _ = offsets[ell]
        for ell_prime in range(ELL_MIN, ELL_MAX + 1):
            prime_start, _ = offsets[ell_prime]
            for total_l in range(abs(ell - ell_prime), ell + ell_prime + 1):
                for total_m in range(-total_l, total_l + 1):
                    value = 0.0j
                    for m in range(-ell, ell + 1):
                        m_prime = m - total_m
                        if not (-ell_prime <= m_prime <= ell_prime):
                            continue
                        coefficient = _clebsch_gordan(
                            ell,
                            m,
                            ell_prime,
                            -m_prime,
                            total_l,
                            total_m,
                        )
                        value += (
                            (-1) ** m_prime
                            * coefficient
                            * matrix[
                                ell_start + m + ell,
                                prime_start + m_prime + ell_prime,
                            ]
                        )
                    serialized_real = _serialized_float(np.real(value))
                    serialized_imaginary = _serialized_float(np.imag(value))
                    for raw, serialized in (
                        (float(np.real(value)), serialized_real),
                        (float(np.imag(value)), serialized_imaginary),
                    ):
                        error = abs(raw - serialized)
                        maximum_absolute_error = max(maximum_absolute_error, error)
                        maximum_normalized_error = max(
                            maximum_normalized_error,
                            error / max(1.0, abs(raw)),
                        )
                    rows.append(
                        [
                            ell,
                            ell_prime,
                            total_l,
                            total_m,
                            serialized_real,
                            serialized_imaginary,
                        ]
                    )
    return rows, {
        "maximum_absolute_rounding_error": maximum_absolute_error,
        "maximum_error_divided_by_max_one_abs_raw": maximum_normalized_error,
        "normalized_error_gate": 5.1e-12,
    }


def _biposh_summary(rows: list[list[int | float]]) -> dict[str, Any]:
    lookup = {
        (int(row[0]), int(row[1]), int(row[2]), int(row[3])): complex(
            float(row[4]), float(row[5])
        )
        for row in rows
    }
    ell, ell_prime, total_l = PRIMARY_PAIR
    primary_vector = [
        lookup[(ell, ell_prime, total_l, total_m)]
        for total_m in range(-total_l, total_l + 1)
    ]
    a22 = lookup[(2, 2, 0, 0)]
    a44 = lookup[(4, 4, 0, 0)]
    primary_norm = float(np.linalg.norm(primary_vector))
    denominator = math.sqrt(abs(a22 * a44))
    norms_by_total_l: dict[int, float] = {}
    for row in rows:
        rank = int(row[2])
        norms_by_total_l[rank] = norms_by_total_l.get(rank, 0.0) + float(
            row[4]
        ) ** 2 + float(row[5]) ** 2
    allowed_power = sum(
        value
        for rank, value in norms_by_total_l.items()
        if rank in ALLOWED_A5_TOTAL_L
    )
    forbidden_power = sum(
        value
        for rank, value in norms_by_total_l.items()
        if rank not in ALLOWED_A5_TOTAL_L
    )
    forbidden_maximum = max(
        math.sqrt(value)
        for rank, value in norms_by_total_l.items()
        if rank not in ALLOWED_A5_TOTAL_L
    )
    total_power = allowed_power + forbidden_power
    return {
        "primary_definition": (
            "norm_M(A_{2,4}^{6M}) / sqrt(abs(A_{2,2}^{00} "
            "A_{4,4}^{00}))"
        ),
        "primary_amplitude_free_statistic": primary_norm / denominator,
        "primary_numerator_norm": primary_norm,
        "primary_denominator": denominator,
        "A_22_00": _complex_pair(a22),
        "A_44_00": _complex_pair(a44),
        "A_24_6M": [_complex_pair(value) for value in primary_vector],
        "total_L_frobenius_norms": {
            str(rank): math.sqrt(value)
            for rank, value in sorted(norms_by_total_l.items())
        },
        "a5_selection_leakage": {
            "allowed_total_L": list(ALLOWED_A5_TOTAL_L),
            "maximum_forbidden_total_L_norm": forbidden_maximum,
            "relative_forbidden_frobenius_norm": math.sqrt(
                forbidden_power / total_power
            ),
            "numerical_gate": 1.0e-11,
            "gate_passed": bool(forbidden_maximum < 1.0e-11),
            "scope": (
                "floating-point A5-equivariance sanity check for the finite "
                "operator form; the exact selection rule is proved separately"
            ),
        },
    }


def _spectral_projector_diagnostics(
    form: np.ndarray,
    gram: np.ndarray,
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    gram_values, gram_vectors = eigh(gram)
    gram_half = (gram_vectors * np.sqrt(gram_values)) @ gram_vectors.conj().T
    gram_inverse_half = (
        gram_vectors * (1.0 / np.sqrt(gram_values))
    ) @ gram_vectors.conj().T
    operator = gram_inverse_half @ form @ gram_inverse_half
    operator = 0.5 * (operator + operator.conj().T)

    offsets = _harmonic_offsets()
    reference_bases: dict[int, np.ndarray] = {}
    for ell, (start, stop) in offsets.items():
        reference_bases[ell] = np.linalg.qr(
            gram_half[:, start:stop], mode="reduced"
        )[0]
    ell_two_basis = reference_bases[2]
    normalization = 6.0 / float(
        np.trace(ell_two_basis.conj().T @ operator @ ell_two_basis).real / 5.0
    )
    operator *= normalization

    eigenvalues, eigenvectors = eigh(operator)
    slot_labels: list[int] = []
    slot_weights: list[np.ndarray] = []
    for ell, basis in reference_bases.items():
        overlap = np.sum(np.abs(eigenvectors.conj().T @ basis) ** 2, axis=1)
        for _ in range(basis.shape[1]):
            slot_labels.append(ell)
            slot_weights.append(overlap)
    weights = np.column_stack(slot_weights)
    eigen_rows, slot_columns = linear_sum_assignment(-weights)
    assignments: dict[int, list[int]] = {ell: [] for ell in reference_bases}
    for eigen_row, slot_column in zip(eigen_rows, slot_columns, strict=True):
        assignments[slot_labels[int(slot_column)]].append(int(eigen_row))

    rows: list[dict[str, Any]] = []
    projectors: dict[int, np.ndarray] = {}
    for ell, reference in reference_bases.items():
        assigned = sorted(assignments[ell])
        selected = eigenvectors[:, assigned]
        projectors[ell] = selected @ selected.conj().T
        singular_values = np.linalg.svd(
            reference.conj().T @ selected,
            compute_uv=False,
        )
        minimum_cosine = float(np.min(np.clip(singular_values, 0.0, 1.0)))
        sine_max = math.sqrt(max(0.0, 1.0 - minimum_cosine**2))

        complement = null_space(reference.conj().T)
        reference_block = reference.conj().T @ operator @ reference
        complement_block = complement.conj().T @ operator @ complement
        residual = float(
            np.linalg.norm(complement.conj().T @ operator @ reference, ord=2)
        )
        reference_spectrum = eigh(reference_block, eigvals_only=True)
        complement_spectrum = eigh(complement_block, eigvals_only=True)
        block_gap = float(
            np.min(
                np.abs(
                    reference_spectrum[:, None] - complement_spectrum[None, :]
                )
            )
        )
        selected_spectrum = eigenvalues[assigned]
        other_spectrum = np.delete(eigenvalues, assigned)
        selected_gap = float(
            np.min(np.abs(selected_spectrum[:, None] - other_spectrum[None, :]))
        )
        separation_hypothesis = bool(block_gap > 2.0 * residual)
        angle_bound = (
            min(1.0, residual / (block_gap - residual))
            if separation_hypothesis
            else None
        )
        rows.append(
            {
                "ell": ell,
                "dimension": 2 * ell + 1,
                "assignment_rule": (
                    "global maximum-total-projector-overlap assignment with exact "
                    "band dimensions"
                ),
                "assigned_eigenvalue_indices": assigned,
                "assigned_eigenvalues": [
                    float(eigenvalues[index]) for index in assigned
                ],
                "maximum_principal_angle_radians": math.asin(min(1.0, sine_max)),
                "maximum_principal_angle_sine": sine_max,
                "block_off_diagonal_residual_norm": residual,
                "unperturbed_block_spectral_gap": block_gap,
                "selected_operator_cluster_gap": selected_gap,
                "davis_kahan_hypothesis_2r_lt_gap": separation_hypothesis,
                "davis_kahan_sine_upper_bound": angle_bound,
                "bound_contains_measured_angle": bool(
                    angle_bound is not None and sine_max <= angle_bound + 2.0e-12
                ),
            }
        )
    return {
        "symmetric_whitening": "unique positive square root of weighted Gram matrix",
        "ell_two_mean_eigenvalue_normalization": normalization,
        "operator_min_eigenvalue_after_normalization": float(eigenvalues[0]),
        "operator_max_eigenvalue_after_normalization": float(eigenvalues[-1]),
        "rows": rows,
        "bound_scope": (
            "For H0 equal to the reference-band block diagonal of H, the "
            "off-diagonal perturbation has norm r. When gap(H0)>2r, Weyl "
            "separation and Davis-Kahan give sin(theta)<=r/(gap-r). Rows that "
            "fail the hypothesis carry no bound."
        ),
    }, projectors


def _projector_step_rows(
    projectors_by_level: dict[int, dict[int, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    levels = sorted(projectors_by_level)
    for coarse, fine in zip(levels, levels[1:]):
        for ell in range(ELL_MIN, ELL_MAX + 1):
            left_values, left_vectors = eigh(projectors_by_level[coarse][ell])
            right_values, right_vectors = eigh(projectors_by_level[fine][ell])
            dimension = 2 * ell + 1
            left = left_vectors[:, np.argsort(left_values)[-dimension:]]
            right = right_vectors[:, np.argsort(right_values)[-dimension:]]
            singular_values = np.linalg.svd(left.conj().T @ right, compute_uv=False)
            minimum_cosine = float(np.min(np.clip(singular_values, 0.0, 1.0)))
            sine_max = math.sqrt(max(0.0, 1.0 - minimum_cosine**2))
            rows.append(
                {
                    "coarse_level": coarse,
                    "fine_level": fine,
                    "ell": ell,
                    "maximum_principal_angle_radians": math.asin(
                        min(1.0, sine_max)
                    ),
                    "maximum_principal_angle_sine": sine_max,
                    "transport": (
                        "identity on canonical harmonic labels after each level's "
                        "unique symmetric Gram whitening"
                    ),
                }
            )
    return rows


def build_a5_biposh_dual_operator_packet() -> tuple[dict[str, Any], dict[str, Any]]:
    tower = build_geodesic_icosahedral_tower(5)
    bounded_repair = json.loads(BOUNDED_REPAIR_RECEIPT.read_text(encoding="utf-8"))
    mean_bridge = bounded_repair.get("exact_conditional_mean_bridge", {})
    one_atom = mean_bridge.get("one_atom_restriction", {})
    repair_torsor = bounded_repair.get("directed_seam_torsor", {})
    base_faces = [
        [int(value) for value in face] for face in tower.levels[0].faces
    ]
    base_face_sha256 = _sha256_bytes(_canonical_bytes(base_faces))
    base_edges_from_faces = sorted(
        {
            tuple(sorted((face[index], face[(index + 1) % 3])))
            for face in base_faces
            for index in range(3)
        }
    )
    base_mesh_edges = sorted(
        tuple(int(value) for value in edge) for edge in tower.levels[0].edges
    )
    base_face_binding_matches = bool(
        repair_torsor.get("exact_oriented_face_count") == 20
        and repair_torsor.get("undirected_seam_count") == 30
        and repair_torsor.get("exact_oriented_face_sha256") == base_face_sha256
        and base_edges_from_faces == base_mesh_edges
    )
    base_degree = np.bincount(
        tower.levels[0].edges.reshape(-1),
        minlength=tower.levels[0].vertex_count,
    )
    base_generator_matches = bool(
        bounded_repair.get("schema")
        == "oph.bounded_atomic_self_readback_closure.v1"
        and bounded_repair.get("status") == EXPECTED_BOUNDED_REPAIR_STATUS
        and bounded_repair.get("certificate_payload_sha256")
        == EXPECTED_BOUNDED_REPAIR_PAYLOAD_SHA256
        and bounded_repair.get("PHYSICAL_REPAIR_LAW_RECEIPT") is False
        and mean_bridge.get("identity")
        == "E[X_next | X=x] = (I - L_icosahedron/60) x"
        and mean_bridge.get("all_probed_states_exact_identity_verified") is True
        and one_atom.get("one_atom_generator") == "-L_icosahedron/60"
        and one_atom.get("exact_identity_verified") is True
        and one_atom.get("physical_time_scale_selected") is False
        and base_face_binding_matches
        and tower.levels[0].vertex_count == 12
        and tower.levels[0].edge_count == 30
        and np.all(base_degree == 5)
    )
    if not base_generator_matches:
        raise ValueError("bounded repair receipt does not bind the base equal-seam operator")
    level_rows: list[dict[str, Any]] = []
    coefficient_cases: list[dict[str, Any]] = []
    projectors: dict[str, dict[int, dict[int, np.ndarray]]] = {
        "equal_seam_raw_graph_laplacian": {},
        "geometric_cotangent_control": {},
    }
    primary_by_operator: dict[str, list[dict[str, float | int]]] = {
        key: [] for key in projectors
    }

    for mesh in tower.levels:
        weights = _vertex_area_weights(mesh)
        design = _harmonic_design(mesh.vertices)
        design_rank = int(np.linalg.matrix_rank(design, tol=2.0e-12))
        base_row: dict[str, Any] = {
            "level": mesh.level,
            "frequency": mesh.frequency,
            "vertex_count": mesh.vertex_count,
            "edge_count": mesh.edge_count,
            "face_count": mesh.face_count,
            "geometry_hash": mesh.geometry_hash,
            "full_harmonic_dimension_through_ell_8": FULL_HARMONIC_DIMENSION,
            "sampled_harmonic_design_rank": design_rank,
            "vertex_area_sum": float(np.sum(weights)),
            "vertex_area_minimum": float(np.min(weights)),
        }
        if design_rank < FULL_HARMONIC_DIMENSION:
            base_row.update(
                {
                    "status": "INSUFFICIENT_RANK_FOR_FULL_ELL_0_TO_8_BAND",
                    "operator_cases": [],
                    "calculation_performed": False,
                }
            )
            level_rows.append(base_row)
            continue

        active_design, removal = _weighted_low_mode_removal(design, weights)
        gram = active_design.conj().T @ (weights[:, None] * active_design)
        gram = 0.5 * (gram + gram.conj().T)
        gram_values = eigh(gram, eigvals_only=True)
        base_row.update(
            {
                "status": "FULL_ELL_2_TO_8_OPERATOR_BAND_RESOLVED",
                "calculation_performed": True,
                "active_dimension": ACTIVE_DIMENSION,
                "low_mode_removal": removal,
                "active_gram_min_eigenvalue": float(gram_values[0]),
                "active_gram_max_eigenvalue": float(gram_values[-1]),
                "active_gram_condition_number": float(
                    gram_values[-1] / gram_values[0]
                ),
            }
        )

        operator_rows: list[dict[str, Any]] = []
        for operator_id, stiffness in (
            (
                "equal_seam_raw_graph_laplacian",
                _equal_seam_graph_stiffness(mesh),
            ),
            ("geometric_cotangent_control", _cotangent_stiffness(mesh)),
        ):
            form = active_design.conj().T @ (stiffness @ active_design)
            form = 0.5 * (form + form.conj().T)
            form_values = eigh(form, eigvals_only=True)
            coefficients, serialization_error = _biposh_rows(form)
            coefficient_payload = {
                "level": mesh.level,
                "operator_id": operator_id,
                "index_fields": [
                    "ell",
                    "ell_prime",
                    "total_L",
                    "total_M",
                    "real",
                    "imaginary",
                ],
                "rows": coefficients,
            }
            coefficient_hash = _sha256_bytes(_canonical_bytes(coefficient_payload))
            coefficient_cases.append(coefficient_payload)
            summary = _biposh_summary(coefficients)
            spectral, case_projectors = _spectral_projector_diagnostics(form, gram)
            projectors[operator_id][mesh.level] = case_projectors
            primary_by_operator[operator_id].append(
                {
                    "level": mesh.level,
                    "value": float(summary["primary_amplitude_free_statistic"]),
                }
            )
            operator_rows.append(
                {
                    "operator_id": operator_id,
                    "operator_status": (
                        "bounded_reconstructed_on_base_carrier__refinement_extension_not_source_selected"
                        if operator_id == "equal_seam_raw_graph_laplacian"
                        else "geometric_nondynamical_control"
                    ),
                    "form_min_eigenvalue": float(form_values[0]),
                    "form_max_eigenvalue": float(form_values[-1]),
                    "positive_on_active_ell_2_to_8_band": bool(
                        form_values[0] > 1.0e-10
                    ),
                    "form_hermiticity_residual": float(
                        np.max(np.abs(form - form.conj().T))
                    ),
                    "full_biposh_coefficient_count": len(coefficients),
                    "coefficient_case_sha256": coefficient_hash,
                    "coefficient_serialization_error": serialization_error,
                    "biposh_summary": summary,
                    "spectral_projector_diagnostics": spectral,
                }
            )
        base_row["operator_cases"] = operator_rows
        level_rows.append(base_row)

    refinement_rows: list[dict[str, Any]] = []
    for operator_id, values in primary_by_operator.items():
        difference_rows: list[dict[str, Any]] = []
        for coarse, fine in zip(values, values[1:]):
            difference_rows.append(
                {
                    "coarse_level": int(coarse["level"]),
                    "fine_level": int(fine["level"]),
                    "absolute_increment": abs(
                        float(fine["value"]) - float(coarse["value"])
                    ),
                }
            )
        contraction_rows: list[dict[str, Any]] = []
        for prior, current in zip(difference_rows, difference_rows[1:]):
            contraction_rows.append(
                {
                    "fine_level": int(current["fine_level"]),
                    "increment_contraction_ratio": float(
                        current["absolute_increment"] / prior["absolute_increment"]
                    ),
                }
            )
        refinement_rows.append(
            {
                "operator_id": operator_id,
                "primary_values": values,
                "successive_absolute_increments": difference_rows,
                "observed_increment_contractions": contraction_rows,
                "projector_principal_angle_steps": _projector_step_rows(
                    projectors[operator_id]
                ),
                "projector_step_scope": (
                    "canonical symmetric-whitened coefficient-coordinate "
                    "diagnostic; not physical Hilbert-space convergence"
                ),
                "physical_hilbert_space_convergence": False,
                "continuum_inference_authorized": False,
                "reason": (
                    "Four finite admissible levels do not supply an analytic tail "
                    "bound or a source-selected continuum operator."
                ),
            }
        )

    coefficient_document = {
        "schema": "oph.a5-biposh-dual-operator-coefficients.v1",
        "issue": 659,
        "basis": (
            "complex Condon-Shortley spherical harmonics; BipoSH convention "
            "A_ll'^{LM}=sum_mm' (-1)^m' <l m,l' -m'|L M> B_lm,l'm'"
        ),
        "coefficient_kind": "finite stiffness-form operator fingerprint",
        "serialization_contract": {
            "coefficient_real_and_imaginary_significant_decimal_digits": (
                SERIALIZED_SIGNIFICANT_DIGITS
            ),
            "rounding": "IEEE-754 conversion of scientific decimal round-to-nearest",
            "receipt_diagnostics": (
                "stored as binary64 JSON numbers and independently replayed with "
                "declared tolerances; no byte-exact eigensolver replay is claimed"
            ),
        },
        "case_count": len(coefficient_cases),
        "coefficient_count_per_case": 5929,
        "cases": coefficient_cases,
    }
    coefficient_bytes = _canonical_bytes(coefficient_document)

    producer_path = Path(__file__).resolve()
    verifier_path = producer_path.with_name(
        "verify_a5_biposh_refinement_independent.py"
    )
    receipt = {
        "schema": "oph.a5-biposh-dual-operator-refinement.v1",
        "issue": 659,
        "status": (
            "FINITE_DUAL_OPERATOR_FINGERPRINT_ATTAINED__CONTINUUM_RESIDUAL_"
            "AND_PHYSICAL_COVARIANCE_OPEN"
        ),
        "source_scope": {
            "geometry": "registered nested geodesic icosahedral vertex tower",
            "levels": list(range(0, 6)),
            "harmonic_band": {"ell_min": ELL_MIN, "ell_max": ELL_MAX},
            "removed_modes": "weighted ell=0,1 projection",
            "candidate_operator": "unweighted equal-seam combinatorial graph Laplacian",
            "control_operator": "chord-triangle cotangent FEM stiffness",
            "external_comparison_data_used": False,
            "sky_data_used": False,
            "target_values_used": False,
            "stochastic_release_ensemble_used": False,
            "green_kernel_or_heat_time_used": False,
        },
        "frozen_primary_statistic": {
            "ell": 2,
            "ell_prime": 4,
            "total_L": 6,
            "definition": (
                "norm_M(A_{2,4}^{6M}) / sqrt(abs(A_{2,2}^{00} "
                "A_{4,4}^{00}))"
            ),
            "amplitude_free": True,
            "selected_before_any_comparison": True,
        },
        "rank_controls": {
            "full_harmonic_dimension_through_ell_8": FULL_HARMONIC_DIMENSION,
            "level_0_vertex_count": tower.levels[0].vertex_count,
            "level_1_vertex_count": tower.levels[1].vertex_count,
            "levels_0_and_1_are_nonadmissible": True,
        },
        "bounded_repair_generator_bridge": {
            "parent_receipt": BOUNDED_REPAIR_RECEIPT.relative_to(ROOT).as_posix(),
            "parent_schema": bounded_repair["schema"],
            "parent_status": bounded_repair["status"],
            "parent_certificate_payload_sha256": bounded_repair[
                "certificate_payload_sha256"
            ],
            "conditional_mean_identity": mean_bridge["identity"],
            "one_atom_generator": one_atom["one_atom_generator"],
            "base_carrier_vertex_count": tower.levels[0].vertex_count,
            "base_carrier_edge_count": tower.levels[0].edge_count,
            "base_carrier_degree": 5,
            "parent_exact_oriented_face_sha256": repair_torsor[
                "exact_oriented_face_sha256"
            ],
            "base_carrier_oriented_face_sha256": base_face_sha256,
            "base_carrier_labelled_face_presentation_matches_parent": (
                repair_torsor["exact_oriented_face_sha256"] == base_face_sha256
            ),
            "base_carrier_edge_set_matches_face_presentation": (
                base_edges_from_faces == base_mesh_edges
            ),
            "base_carrier_operator_matches_bounded_reconstructed_one_atom_mean_generator_up_to_scale": base_generator_matches,
            "scale_relation": "mean generator = -L_equal_seam/60",
            "bounded_frozen_adversarial_suite_attained": True,
            "refinement_tower_extension_source_selected": False,
            "global_a1_a3_policy_uniqueness_receipt": bool(
                bounded_repair.get("GLOBAL_A1_A3_POLICY_UNIQUENESS_RECEIPT")
            ),
            "physical_repair_law_receipt": bool(
                bounded_repair.get("PHYSICAL_REPAIR_LAW_RECEIPT")
            ),
            "physical_time_scale_selected": bool(
                one_atom.get("physical_time_scale_selected")
            ),
            "scope": (
                "Exact positive link on the twelve-vertex base carrier. It does "
                "not select the equal-seam extension over the refinement tower, "
                "a unique global A1-A3 repair policy, a physical clock, or a "
                "continuum operator."
            ),
        },
        "level_rows": level_rows,
        "refinement_diagnostics": refinement_rows,
        "full_coefficient_bundle": {
            "path": DEFAULT_COEFFICIENTS.relative_to(ROOT).as_posix(),
            "bytes": len(coefficient_bytes),
            "sha256": _sha256_bytes(coefficient_bytes),
            "case_count": len(coefficient_cases),
            "coefficient_count_per_case": 5929,
            "contains_every_llprime_LM_coefficient_for_ell_2_through_8": True,
            "coefficient_significant_decimal_digits": SERIALIZED_SIGNIFICANT_DIGITS,
            "cross_platform_replay": (
                "semantic tolerance replay; coefficient bundle bytes are pinned but "
                "fresh eigensolver output is not required to be byte-identical"
            ),
        },
        "a5_selection_rule_context": {
            "allowed_total_L_through_16": list(ALLOWED_A5_TOTAL_L),
            "role": (
                "diagnostic against the separately proved A5 singlet multiplicities; "
                "this numerical packet does not replace that theorem"
            ),
        },
        "source_pins": [
            _file_pin(ROOT / "oph_fpe/core/icosahedral.py"),
            _file_pin(BOUNDED_REPAIR_RECEIPT),
            _file_pin(producer_path),
            _file_pin(verifier_path),
            _file_pin(ROOT / "tests/test_a5_biposh_refinement.py"),
        ],
        "selection_decision": {
            "base_equal_seam_operator_bounded_reconstructed": True,
            "equal_seam_operator_source_selected": False,
            "refinement_tower_equal_seam_extension_source_selected": False,
            "physical_repair_law_selected": False,
            "physical_covariance_selected": False,
            "physical_release_ensemble_selected": False,
            "global_frame_quotient_visible": False,
            "screen_to_sky_readout_selected": False,
            "continuum_residual_decided": False,
            "physical_prediction": False,
            "promotion_allowed": False,
        },
        "claim_boundary": (
            "Target-blind finite screen-side operator fingerprint. The packet "
            "decomposes two declared stiffness forms, not a covariance. The "
            "equal-seam form matches, up to the exact factor minus one over sixty, "
            "the bounded reconstructed one-atom mean generator on the base "
            "carrier. Its extension over the refinement tower is not selected by "
            "a source receipt or a unique global repair law. The cotangent form is "
            "a geometric control. The "
            "finite level-2 through level-5 trend has no analytic tail bound and "
            "does not decide a continuum residual. A covariance or release "
            "ensemble, quotient-visible global frame, screen-to-sky readout, and "
            "physical comparison remain open. No physical prediction follows."
        ),
    }
    receipt["payload_sha256"] = _sha256_bytes(_canonical_bytes(receipt))
    return receipt, coefficient_document


def write_a5_biposh_dual_operator_packet(
    receipt_path: Path = DEFAULT_RECEIPT,
    coefficient_path: Path = DEFAULT_COEFFICIENTS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt, coefficients = build_a5_biposh_dual_operator_packet()
    coefficient_bytes = _canonical_bytes(coefficients)
    expected = receipt["full_coefficient_bundle"]
    if coefficient_path.resolve() != DEFAULT_COEFFICIENTS.resolve():
        expected["path"] = coefficient_path.name
    expected["bytes"] = len(coefficient_bytes)
    expected["sha256"] = _sha256_bytes(coefficient_bytes)
    receipt.pop("payload_sha256", None)
    receipt["payload_sha256"] = _sha256_bytes(_canonical_bytes(receipt))
    coefficient_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    coefficient_path.write_bytes(coefficient_bytes)
    receipt_path.write_bytes(_canonical_bytes(receipt))
    return receipt, coefficients


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS)
    args = parser.parse_args()
    receipt, _ = write_a5_biposh_dual_operator_packet(
        args.receipt,
        args.coefficients,
    )
    print(receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
