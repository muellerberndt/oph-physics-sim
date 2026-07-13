from __future__ import annotations

import numpy as np

from oph_fpe.bulk.central_interface_msa import central_interface_msa_report


def _random_hermitian(dimension: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    return (matrix + matrix.conj().T) / 2.0


def _matrix_exponential(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.conj().T) / 2.0)
    return (vectors * np.exp(values)) @ vectors.conj().T


def _central_interface_fixture():
    rng = np.random.default_rng(543)
    dims = (2, 2, 2, 2)
    left_dimension = 4
    right_dimension = 4
    hamiltonians = []
    left_terms = []
    right_terms = []
    interface_terms = []
    unnormalized = []
    for energy in (0.0, 0.7):
        left = np.kron(_random_hermitian(left_dimension, rng), np.eye(right_dimension))
        right = np.kron(np.eye(left_dimension), _random_hermitian(right_dimension, rng))
        interface = energy * np.eye(left_dimension * right_dimension)
        hamiltonian = left + right + interface
        hamiltonians.append(hamiltonian)
        left_terms.append([left])
        right_terms.append([right])
        interface_terms.append([interface])
        unnormalized.append(_matrix_exponential(-hamiltonian))
    partition = sum(float(np.trace(matrix).real) for matrix in unnormalized)
    blocks = []
    for matrix in unnormalized:
        block_partition = float(np.trace(matrix).real)
        blocks.append((block_partition / partition, matrix / block_partition, dims))
    return blocks, hamiltonians, left_terms, right_terms, interface_terms


def test_central_interface_matrices_pass_all_msa_characterizations() -> None:
    blocks, hamiltonians, left, right, interface = _central_interface_fixture()

    report = central_interface_msa_report(
        blocks,
        hamiltonian_blocks=hamiltonians,
        left_terms=left,
        right_terms=right,
        interface_terms=interface,
    )

    assert report["CENTRAL_INTERFACE_MSA_RECEIPT"] is True
    assert report["state_semantics"] == "noncommutative_block_matrix"
    assert report["entropic_alignment_defect_nats"] < 1.0e-8
    assert report["modular_splitting_operator_norm_defect"] < 1.0e-8
    assert report["takesaki_operator_norm_defect"] < 1.0e-8
    assert report["quantum_collar_cmi_nats"] < 1.0e-8
    assert report["matrix_evidence_sha256"]
    assert report["mandatory_negative_controls"][
        "bell_markov_but_misaligned_control_passed"
    ] is True
    assert report["mandatory_negative_controls"][
        "noncentral_interface_control_passed"
    ] is True


def test_aligned_state_without_operator_matrices_does_not_promote() -> None:
    blocks, _, _, _, _ = _central_interface_fixture()

    report = central_interface_msa_report(blocks)

    assert report["entropic_alignment_defect_nats"] < 1.0e-8
    assert report["CENTRAL_INTERFACE_MSA_RECEIPT"] is False
    assert "explicit_hamiltonian_and_operator_matrices_missing" in report["promotion_blockers"]


def test_noncentral_cross_cut_term_is_detected() -> None:
    identity = np.eye(2, dtype=complex)
    pauli_z = np.diag([1.0, -1.0]).astype(complex)
    cross_cut = np.kron(np.kron(np.kron(identity, pauli_z), pauli_z), identity)
    hamiltonian = 0.8 * cross_cut
    unnormalized = _matrix_exponential(-hamiltonian)
    state = unnormalized / float(np.trace(unnormalized).real)

    report = central_interface_msa_report(
        [(1.0, state, (2, 2, 2, 2))],
        hamiltonian_blocks=[hamiltonian],
        left_terms=[[]],
        right_terms=[[]],
        interface_terms=[[hamiltonian]],
    )

    assert report["CENTRAL_INTERFACE_MSA_RECEIPT"] is False
    assert report["central_interface_residual"] > 1.0e-3
    assert report["noncentral_cross_cut_residual"] > 1.0e-3
    assert "interface_operator_not_central" in report["promotion_blockers"]
