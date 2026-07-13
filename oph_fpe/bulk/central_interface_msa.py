from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np

from oph_fpe.evidence.hashes import canonical_json_bytes


Block = tuple[float, np.ndarray, tuple[int, int, int, int]]
ALGORITHM_ID = "oph-central-interface-msa-v1"


def central_interface_msa_report(
    blocks: Sequence[Block],
    *,
    hamiltonian_blocks: Sequence[np.ndarray] | None = None,
    left_terms: Sequence[Sequence[np.ndarray]] | None = None,
    right_terms: Sequence[Sequence[np.ndarray]] | None = None,
    interface_terms: Sequence[Sequence[np.ndarray]] | None = None,
    beta: float = 1.0,
    tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Verify the finite central-interface/Markov-split alignment receipt.

    ``blocks`` encode the direct-sum center sectors and order each tensor factor as
    ``(A, bL, bR, D)``. A positive receipt additionally requires explicit Hamiltonian
    matrices and a decomposition into left, right, and interface terms. Interface
    terms must be scalar inside every center block; merely small classical collar CMI
    is not sufficient.
    """

    if not np.isfinite(beta) or float(beta) <= 0.0:
        raise ValueError("beta must be finite and positive")
    if not np.isfinite(tolerance) or float(tolerance) < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    checked = _validated_blocks(blocks)
    weights = np.array([block[0] for block in checked], dtype=float)
    minimum_block_eigenvalue = min(
        float(np.min(np.linalg.eigvalsh(block[1]))) for block in checked
    )
    faithful = minimum_block_eigenvalue > float(tolerance)
    center_weight_residual = abs(float(np.sum(weights)) - 1.0)
    entropic_defect = entropic_alignment_defect(checked)
    cmi = collar_cmi_nats(checked)
    modular_defect = modular_splitting_defect(checked) if faithful else None
    takesaki = takesaki_defect(checked) if faithful else None

    controls = _mandatory_negative_controls(float(tolerance))
    controls_pass = bool(
        controls["bell_markov_but_misaligned_control_passed"]
        and controls["noncentral_interface_control_passed"]
    )
    matrices_present = all(
        item is not None
        for item in (hamiltonian_blocks, left_terms, right_terms, interface_terms)
    )
    blockers: list[str] = []
    structure: dict[str, Any]
    if matrices_present:
        assert hamiltonian_blocks is not None
        assert left_terms is not None
        assert right_terms is not None
        assert interface_terms is not None
        structure = _operator_structure_report(
            checked,
            hamiltonian_blocks=hamiltonian_blocks,
            left_terms=left_terms,
            right_terms=right_terms,
            interface_terms=interface_terms,
            beta=float(beta),
        )
    else:
        structure = {
            "explicit_hamiltonian_matrices_present": False,
            "operator_classification": {
                "left_term_count": 0,
                "right_term_count": 0,
                "interface_term_count": 0,
            },
            "hamiltonian_reconstruction_residual": None,
            "left_operator_alignment_residual": None,
            "right_operator_alignment_residual": None,
            "central_interface_residual": None,
            "noncentral_cross_cut_residual": None,
            "gibbs_state_trace_residual": None,
            "matrix_evidence_sha256": None,
        }
        blockers.append("explicit_hamiltonian_and_operator_matrices_missing")

    if center_weight_residual > float(tolerance) or np.any(weights <= 0.0):
        blockers.append("center_sector_weights_not_faithful_and_normalized")
    if not faithful:
        blockers.append("block_state_not_faithful")
    if entropic_defect > float(tolerance):
        blockers.append("ec_alignment_defect_nonzero")
    if cmi > float(tolerance):
        blockers.append("quantum_collar_cmi_nonzero")
    if modular_defect is None or modular_defect > float(tolerance):
        blockers.append("modular_splitting_defect_nonzero_or_undefined")
    if takesaki is None or takesaki > 10.0 * float(tolerance):
        blockers.append("takesaki_invariance_defect_nonzero_or_undefined")
    if matrices_present:
        for field, blocker in (
            ("hamiltonian_reconstruction_residual", "hamiltonian_decomposition_mismatch"),
            ("left_operator_alignment_residual", "left_operator_not_one_sided"),
            ("right_operator_alignment_residual", "right_operator_not_one_sided"),
            ("central_interface_residual", "interface_operator_not_central"),
            ("noncentral_cross_cut_residual", "noncentral_cross_cut_component_present"),
            ("gibbs_state_trace_residual", "state_not_gibbs_of_declared_hamiltonian"),
        ):
            value = structure[field]
            if value is None or not np.isfinite(value) or float(value) > float(tolerance):
                blockers.append(blocker)
    if not controls_pass:
        blockers.append("mandatory_misalignment_controls_failed")

    receipt = not blockers
    return {
        "schema_version": 1,
        "evidence_kind": "finite_matrix_central_interface_msa",
        "algorithm": ALGORITHM_ID,
        "paper_provenance": {
            "source": "reverse-engineering-reality/code/collar_alignment/msa_characterizations.py",
            "issue": 543,
            "result_class": "finite_matrix_structural_receipt",
        },
        "state_semantics": "noncommutative_block_matrix",
        "log_unit": "nat",
        "block_count": len(checked),
        "center_sector_weight_residual": float(center_weight_residual),
        "minimum_block_eigenvalue": float(minimum_block_eigenvalue),
        "faithful_block_state": bool(faithful),
        "entropic_alignment_defect_nats": float(entropic_defect),
        "modular_splitting_operator_norm_defect": (
            None if modular_defect is None else float(modular_defect)
        ),
        "takesaki_operator_norm_defect": None if takesaki is None else float(takesaki),
        "quantum_collar_cmi_nats": float(cmi),
        "tolerance": float(tolerance),
        **structure,
        "mandatory_negative_controls": controls,
        "promotion_blockers": blockers,
        "CENTRAL_INTERFACE_MSA_RECEIPT": bool(receipt),
        "claim_boundary": (
            "Finite bounded-matrix central-interface/Markov-split alignment receipt. "
            "It does not follow from the classical packet CMI diagnostic and does not "
            "establish continuum collar alignment or an Einstein-source theorem."
        ),
    }


def collar_cmi_nats(blocks: Sequence[Block]) -> float:
    total = 0.0
    for weight, state, dims in blocks:
        total += float(weight) * conditional_mutual_information(
            state, list(dims), [0], [1, 2], [3]
        )
    return max(0.0, float(total))


def entropic_alignment_defect(blocks: Sequence[Block]) -> float:
    """Maximum blockwise I(A bL : bR D), zero exactly on aligned blocks."""

    return max(
        (
            mutual_information(state, list(dims), [0, 1], [2, 3])
            for weight, state, dims in blocks
            if weight > 0.0
        ),
        default=0.0,
    )


def modular_splitting_defect(blocks: Sequence[Block]) -> float:
    worst = 0.0
    for weight, state, dims in blocks:
        if weight <= 0.0:
            continue
        d_a, d_l, d_r, d_d = dims
        logarithm = _logm_faithful(state)
        projection = one_sided_projection(logarithm, d_a * d_l, d_r * d_d)
        worst = max(worst, float(np.linalg.norm(logarithm - projection, ord=2)))
    return worst


def takesaki_defect(blocks: Sequence[Block]) -> float:
    """Deterministic finite-basis form of the Takesaki commutator criterion."""

    worst = 0.0
    for weight, state, dims in blocks:
        if weight <= 0.0:
            continue
        d_a, d_l, d_r, d_d = dims
        left_dim = d_a * d_l
        right_dim = d_r * d_d
        logarithm = _logm_faithful(state)
        for row in range(left_dim):
            for column in range(left_dim):
                matrix_unit = np.zeros((left_dim, left_dim), dtype=complex)
                matrix_unit[row, column] = 1.0
                embedded = np.kron(matrix_unit, np.eye(right_dim))
                commutator = logarithm @ embedded - embedded @ logarithm
                projected = _left_projection(commutator, left_dim, right_dim)
                worst = max(
                    worst,
                    float(np.linalg.norm(commutator - projected, ord=2)),
                )
    return worst


def one_sided_projection(matrix: np.ndarray, left_dim: int, right_dim: int) -> np.ndarray:
    tensor = matrix.reshape(left_dim, right_dim, left_dim, right_dim)
    left = np.trace(tensor, axis1=1, axis2=3) / float(right_dim)
    right = np.trace(tensor, axis1=0, axis2=2) / float(left_dim)
    scalar = np.trace(matrix) / float(left_dim * right_dim)
    return (
        np.kron(left, np.eye(right_dim))
        + np.kron(np.eye(left_dim), right)
        - scalar * np.eye(left_dim * right_dim)
    )


def mutual_information(
    state: np.ndarray,
    dims: list[int],
    left: list[int],
    right: list[int],
) -> float:
    entropy_left = von_neumann_entropy(partial_trace(state, dims, left))
    entropy_right = von_neumann_entropy(partial_trace(state, dims, right))
    entropy_joint = von_neumann_entropy(partial_trace(state, dims, sorted(left + right)))
    return max(0.0, float(entropy_left + entropy_right - entropy_joint))


def conditional_mutual_information(
    state: np.ndarray,
    dims: list[int],
    part_a: list[int],
    part_b: list[int],
    part_d: list[int],
) -> float:
    entropy_ab = von_neumann_entropy(partial_trace(state, dims, sorted(part_a + part_b)))
    entropy_bd = von_neumann_entropy(partial_trace(state, dims, sorted(part_b + part_d)))
    entropy_b = von_neumann_entropy(partial_trace(state, dims, part_b))
    entropy_abd = von_neumann_entropy(
        partial_trace(state, dims, sorted(part_a + part_b + part_d))
    )
    return max(0.0, float(entropy_ab + entropy_bd - entropy_b - entropy_abd))


def partial_trace(state: np.ndarray, dims: list[int], keep: list[int]) -> np.ndarray:
    keep = sorted(keep)
    tensor = np.asarray(state, dtype=complex).reshape(dims + dims)
    subsystem_count = len(dims)
    traced = [index for index in range(subsystem_count) if index not in keep]
    for count, index in enumerate(sorted(traced, reverse=True)):
        current_count = subsystem_count - count
        tensor = np.trace(tensor, axis1=index, axis2=index + current_count)
    kept_dimension = int(np.prod([dims[index] for index in keep])) if keep else 1
    return tensor.reshape(kept_dimension, kept_dimension)


def von_neumann_entropy(state: np.ndarray) -> float:
    values = np.linalg.eigvalsh(_hermitian(np.asarray(state, dtype=complex)))
    values = values[values > 1.0e-14]
    return float(-np.sum(values * np.log(values)))


def _operator_structure_report(
    blocks: Sequence[Block],
    *,
    hamiltonian_blocks: Sequence[np.ndarray],
    left_terms: Sequence[Sequence[np.ndarray]],
    right_terms: Sequence[Sequence[np.ndarray]],
    interface_terms: Sequence[Sequence[np.ndarray]],
    beta: float,
) -> dict[str, Any]:
    count = len(blocks)
    if not (
        len(hamiltonian_blocks)
        == len(left_terms)
        == len(right_terms)
        == len(interface_terms)
        == count
    ):
        raise ValueError("one Hamiltonian/decomposition entry is required per center block")
    reconstruction = 0.0
    left_alignment = 0.0
    right_alignment = 0.0
    central_alignment = 0.0
    noncentral = 0.0
    checked_hamiltonians: list[np.ndarray] = []
    all_matrices: list[np.ndarray] = []
    term_counts = {"left_term_count": 0, "right_term_count": 0, "interface_term_count": 0}
    for index, ((_, _, dims), hamiltonian, left, right, interface) in enumerate(
        zip(
            blocks,
            hamiltonian_blocks,
            left_terms,
            right_terms,
            interface_terms,
            strict=True,
        )
    ):
        dimension = int(np.prod(dims))
        left_dim = int(dims[0] * dims[1])
        right_dim = int(dims[2] * dims[3])
        h = _validated_hermitian(hamiltonian, dimension, f"hamiltonian_blocks[{index}]")
        left_checked = [
            _validated_hermitian(term, dimension, f"left_terms[{index}]") for term in left
        ]
        right_checked = [
            _validated_hermitian(term, dimension, f"right_terms[{index}]") for term in right
        ]
        interface_checked = [
            _validated_hermitian(term, dimension, f"interface_terms[{index}]")
            for term in interface
        ]
        decomposition = np.zeros_like(h)
        for term in left_checked:
            decomposition += term
            left_alignment = max(
                left_alignment,
                float(np.linalg.norm(term - _left_projection(term, left_dim, right_dim), ord=2)),
            )
        for term in right_checked:
            decomposition += term
            right_alignment = max(
                right_alignment,
                float(np.linalg.norm(term - _right_projection(term, left_dim, right_dim), ord=2)),
            )
        for term in interface_checked:
            decomposition += term
            central = np.trace(term) / float(dimension) * np.eye(dimension)
            central_alignment = max(
                central_alignment,
                float(np.linalg.norm(term - central, ord=2)),
            )
        reconstruction = max(reconstruction, float(np.linalg.norm(h - decomposition, ord=2)))
        noncentral = max(
            noncentral,
            float(np.linalg.norm(h - one_sided_projection(h, left_dim, right_dim), ord=2)),
        )
        checked_hamiltonians.append(h)
        all_matrices.extend([h, *left_checked, *right_checked, *interface_checked])
        term_counts["left_term_count"] += len(left_checked)
        term_counts["right_term_count"] += len(right_checked)
        term_counts["interface_term_count"] += len(interface_checked)

    global_energy_floor = min(
        float(np.min(np.linalg.eigvalsh(hamiltonian)))
        for hamiltonian in checked_hamiltonians
    )
    unnormalized_gibbs = [
        _matrix_exponential(
            -float(beta)
            * (hamiltonian - global_energy_floor * np.eye(hamiltonian.shape[0]))
        )
        for hamiltonian in checked_hamiltonians
    ]
    partition = sum(float(np.trace(matrix).real) for matrix in unnormalized_gibbs)
    gibbs_residual = 0.0
    for (weight, state, _), unnormalized in zip(blocks, unnormalized_gibbs, strict=True):
        expected_block = unnormalized / partition
        gibbs_residual = max(
            gibbs_residual,
            _trace_norm(float(weight) * state - expected_block),
        )
    evidence_hash = _matrix_evidence_hash(blocks, all_matrices, beta=beta)
    return {
        "explicit_hamiltonian_matrices_present": True,
        "operator_classification": term_counts,
        "hamiltonian_reconstruction_residual": float(reconstruction),
        "left_operator_alignment_residual": float(left_alignment),
        "right_operator_alignment_residual": float(right_alignment),
        "central_interface_residual": float(central_alignment),
        "noncentral_cross_cut_residual": float(noncentral),
        "gibbs_state_trace_residual": float(gibbs_residual),
        "matrix_evidence_sha256": evidence_hash,
    }


def _mandatory_negative_controls(tolerance: float) -> dict[str, Any]:
    bell = _bell_counterexample()
    bell_cmi = collar_cmi_nats(bell)
    bell_alignment = entropic_alignment_defect(bell)

    identity = np.eye(2, dtype=complex)
    pauli_z = np.diag([1.0, -1.0]).astype(complex)
    coupling = 0.8 * np.kron(np.kron(np.kron(identity, pauli_z), pauli_z), identity)
    gibbs = _matrix_exponential(-coupling)
    gibbs /= float(np.trace(gibbs).real)
    noncentral_blocks: list[Block] = [(1.0, gibbs, (2, 2, 2, 2))]
    noncentral_alignment = entropic_alignment_defect(noncentral_blocks)
    noncentral_modular = modular_splitting_defect(noncentral_blocks)
    noncentral_operator = float(
        np.linalg.norm(coupling - one_sided_projection(coupling, 4, 4), ord=2)
    )
    threshold = max(1.0e-4, 100.0 * tolerance)
    return {
        "bell_control_quantum_cmi_nats": float(bell_cmi),
        "bell_control_alignment_defect_nats": float(bell_alignment),
        "bell_markov_but_misaligned_control_passed": bool(
            bell_cmi <= tolerance and bell_alignment > threshold
        ),
        "noncentral_control_alignment_defect_nats": float(noncentral_alignment),
        "noncentral_control_modular_defect": float(noncentral_modular),
        "noncentral_control_operator_residual": float(noncentral_operator),
        "noncentral_interface_control_passed": bool(
            noncentral_alignment > threshold
            and noncentral_modular > threshold
            and noncentral_operator > threshold
        ),
    }


def _bell_counterexample() -> list[Block]:
    vector = np.eye(2, dtype=complex).reshape(-1) / np.sqrt(2.0)
    bell = np.outer(vector, vector.conj())
    state = np.kron(bell, bell).reshape([2] * 8)
    permutation = [0, 2, 1, 3]
    state = state.transpose(permutation + [index + 4 for index in permutation]).reshape(16, 16)
    return [(1.0, state, (2, 2, 2, 2))]


def _validated_blocks(blocks: Sequence[Block]) -> list[Block]:
    if not blocks:
        raise ValueError("at least one center block is required")
    checked: list[Block] = []
    for index, (weight, state, dims) in enumerate(blocks):
        if len(dims) != 4 or any(int(value) <= 0 for value in dims):
            raise ValueError(f"blocks[{index}] must have four positive tensor dimensions")
        if not np.isfinite(weight) or float(weight) < 0.0:
            raise ValueError(f"blocks[{index}] has an invalid center weight")
        dimension = int(np.prod(dims))
        matrix = np.asarray(state, dtype=complex)
        if matrix.shape != (dimension, dimension):
            raise ValueError(f"blocks[{index}] state dimension does not match its factors")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"blocks[{index}] contains non-finite entries")
        if float(np.linalg.norm(matrix - matrix.conj().T, ord="fro")) > 1.0e-9:
            raise ValueError(f"blocks[{index}] state is not Hermitian")
        matrix = _hermitian(matrix)
        if abs(float(np.trace(matrix).real) - 1.0) > 1.0e-9:
            raise ValueError(f"blocks[{index}] state must have unit trace")
        if float(np.min(np.linalg.eigvalsh(matrix))) < -1.0e-10:
            raise ValueError(f"blocks[{index}] state is not positive semidefinite")
        checked.append((float(weight), matrix, tuple(int(value) for value in dims)))
    return checked


def _validated_hermitian(matrix: np.ndarray, dimension: int, name: str) -> np.ndarray:
    checked = np.asarray(matrix, dtype=complex)
    if checked.shape != (dimension, dimension):
        raise ValueError(f"{name} has the wrong dimension")
    if not np.all(np.isfinite(checked)):
        raise ValueError(f"{name} contains non-finite entries")
    if float(np.linalg.norm(checked - checked.conj().T, ord="fro")) > 1.0e-9:
        raise ValueError(f"{name} is not Hermitian")
    return _hermitian(checked)


def _left_projection(matrix: np.ndarray, left_dim: int, right_dim: int) -> np.ndarray:
    tensor = matrix.reshape(left_dim, right_dim, left_dim, right_dim)
    left = np.trace(tensor, axis1=1, axis2=3) / float(right_dim)
    return np.kron(left, np.eye(right_dim))


def _right_projection(matrix: np.ndarray, left_dim: int, right_dim: int) -> np.ndarray:
    tensor = matrix.reshape(left_dim, right_dim, left_dim, right_dim)
    right = np.trace(tensor, axis1=0, axis2=2) / float(left_dim)
    return np.kron(np.eye(left_dim), right)


def _logm_faithful(state: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(_hermitian(state))
    if float(np.min(values)) <= 0.0:
        raise ValueError("state is not faithful; matrix logarithm is undefined")
    return _hermitian((vectors * np.log(values)) @ vectors.conj().T)


def _matrix_exponential(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(_hermitian(matrix))
    maximum = float(np.max(values))
    result = (vectors * np.exp(values - maximum)) @ vectors.conj().T
    return _hermitian(result * np.exp(maximum))


def _trace_norm(matrix: np.ndarray) -> float:
    return float(np.sum(np.abs(np.linalg.eigvalsh(_hermitian(matrix)))))


def _matrix_evidence_hash(
    blocks: Sequence[Block], matrices: Sequence[np.ndarray], *, beta: float
) -> str:
    payload = {
        "schema": "oph_central_interface_matrix_evidence_hash_v2",
        "algorithm": ALGORITHM_ID,
        "beta": float(beta),
        "blocks": [
            {
                "weight": float(weight),
                "dims": tuple(int(value) for value in dims),
                "state": np.ascontiguousarray(state, dtype=np.complex128),
            }
            for weight, state, dims in blocks
        ],
        "matrices": [
            np.ascontiguousarray(matrix, dtype=np.complex128)
            for matrix in matrices
        ],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conj().T) / 2.0
