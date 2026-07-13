from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from oph_fpe.evidence.hashes import canonical_json_bytes


ALGORITHM_ID = "oph-maxent-i-projection-v1"
EVIDENCE_KIND = "finite_matrix_refinement_closure"


@dataclass(frozen=True)
class IProjectionResult:
    state: np.ndarray
    multipliers: np.ndarray
    moment_residual_l2: float
    moment_residual_linf: float
    hessian_min_eigenvalue: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class MaxEntRefinementResult:
    """Finite-dimensional evidence for one MaxEnt refinement edge.

    The result is deliberately stronger than a fitted-state diagnostic: the
    fine state is passed through the declared coarse-graining callable, the
    coarse moments are matched by an I-projection, and the closure receipt is
    recomputed from the resulting matrices and numerical residuals.
    """

    coarse_state: np.ndarray
    projected_state: np.ndarray
    induced_multipliers: np.ndarray
    fine_dimension: int
    coarse_dimension: int
    fine_constraint_count: int
    coarse_constraint_count: int
    fine_independent_constraint_count: int
    coarse_independent_constraint_count: int
    moment_residual_l2: float
    moment_residual_linf: float
    duhamel_hessian_min_eigenvalue: float
    closure_defect_nats: float
    trace_norm_residual: float
    pinsker_residual_bound: float
    projection_iterations: int
    projection_converged: bool
    input_state_validated: bool
    coarse_state_validated: bool
    refinement_channel_id: str
    evidence_input_sha256: str
    moment_tolerance: float
    closure_defect_tolerance: float
    hessian_floor_tolerance: float
    numerical_tolerance: float

    @property
    def counts_match_displayed_dimension(self) -> bool:
        return bool(
            self.fine_constraint_count
            == self.coarse_constraint_count
            == self.fine_independent_constraint_count
            == self.coarse_independent_constraint_count
        )

    @property
    def projection_unique(self) -> bool:
        return bool(self.duhamel_hessian_min_eigenvalue > self.hessian_floor_tolerance)

    @property
    def pinsker_bound_holds(self) -> bool:
        return bool(
            self.trace_norm_residual
            <= self.pinsker_residual_bound + self.numerical_tolerance
        )

    @property
    def closure_receipt(self) -> bool:
        return bool(
            self.input_state_validated
            and self.coarse_state_validated
            and self.refinement_channel_id
            and self.counts_match_displayed_dimension
            and self.projection_converged
            and self.moment_residual_linf <= self.moment_tolerance
            and self.projection_unique
            and self.closure_defect_nats <= self.closure_defect_tolerance
            and self.pinsker_bound_holds
        )

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "evidence_kind": EVIDENCE_KIND,
            "algorithm": ALGORITHM_ID,
            "paper_provenance": {
                "source": "reverse-engineering-reality/code/maxent/maxent_closure_acceptance.py",
                "issue": 539,
                "result_class": "finite_matrix_structural_receipt",
            },
            "refinement_channel_id": self.refinement_channel_id,
            "coarse_graining_applied": True,
            "evidence_input_sha256": self.evidence_input_sha256,
            "fine_dimension": int(self.fine_dimension),
            "coarse_dimension": int(self.coarse_dimension),
            "fine_constraint_count": int(self.fine_constraint_count),
            "coarse_constraint_count": int(self.coarse_constraint_count),
            "fine_independent_constraint_count": int(self.fine_independent_constraint_count),
            "coarse_independent_constraint_count": int(self.coarse_independent_constraint_count),
            "counts_match_displayed_dimension": self.counts_match_displayed_dimension,
            "induced_map_R_multipliers": [float(value) for value in self.induced_multipliers],
            "moment_matching_residual_l2": float(self.moment_residual_l2),
            "moment_matching_residual_linf": float(self.moment_residual_linf),
            "duhamel_hessian_min_eigenvalue": float(
                self.duhamel_hessian_min_eigenvalue
            ),
            "closure_defect_nats": float(self.closure_defect_nats),
            "trace_norm_residual": float(self.trace_norm_residual),
            "pinsker_residual_bound": float(self.pinsker_residual_bound),
            "projection_iterations": int(self.projection_iterations),
            "projection_converged": bool(self.projection_converged),
            "projection_unique": self.projection_unique,
            "pinsker_residual_bound_holds": self.pinsker_bound_holds,
            "input_state_validated": bool(self.input_state_validated),
            "coarse_state_validated": bool(self.coarse_state_validated),
            "tolerances": {
                "moment": float(self.moment_tolerance),
                "closure_defect_nats": float(self.closure_defect_tolerance),
                "hessian_floor": float(self.hessian_floor_tolerance),
                "numerical": float(self.numerical_tolerance),
            },
            "RG_EXPONENTIAL_FAMILY_CLOSURE_RECEIPT": self.closure_receipt,
            "claim_boundary": (
                "Finite-dimensional one-edge MaxEnt I-projection receipt. A positive "
                "closure defect is quantitative non-closure evidence, not a failed run; "
                "this receipt does not establish regulator-uniform or continuum closure."
            ),
        }


def maxent_refinement_closure_report(
    fine_state: np.ndarray,
    coarse_graining: Callable[[np.ndarray], np.ndarray],
    fine_constraints: Sequence[np.ndarray],
    coarse_constraints: Sequence[np.ndarray],
    *,
    refinement_channel_id: str,
    moment_tolerance: float = 1.0e-9,
    closure_defect_tolerance: float = 1.0e-10,
    hessian_floor_tolerance: float = 1.0e-9,
    numerical_tolerance: float = 1.0e-9,
    max_iterations: int = 200,
) -> MaxEntRefinementResult:
    """Evaluate moment-matching MaxEnt closure on an actual refinement edge."""

    if not callable(coarse_graining):
        raise TypeError("coarse_graining must be callable")
    if not str(refinement_channel_id).strip():
        raise ValueError("refinement_channel_id must name the applied channel")
    for name, value in (
        ("moment_tolerance", moment_tolerance),
        ("closure_defect_tolerance", closure_defect_tolerance),
        ("hessian_floor_tolerance", hessian_floor_tolerance),
        ("numerical_tolerance", numerical_tolerance),
    ):
        if not np.isfinite(value) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    fine = _validated_density_matrix(fine_state, name="fine_state")
    fine_ops = _validated_constraints(
        fine_constraints, dimension=fine.shape[0], name="fine_constraints"
    )
    sigma = _validated_density_matrix(
        coarse_graining(np.array(fine, copy=True)), name="coarse_state"
    )
    coarse_ops = _validated_constraints(
        coarse_constraints, dimension=sigma.shape[0], name="coarse_constraints"
    )

    projection = moment_matching_i_projection(
        sigma,
        coarse_ops,
        tolerance=float(moment_tolerance),
        max_iterations=int(max_iterations),
    )
    defect = max(relative_entropy_nats(sigma, projection.state), 0.0)
    residual = trace_norm(sigma - projection.state)
    pinsker = math.sqrt(2.0 * defect)
    evidence_hash = _evidence_hash(
        fine,
        sigma,
        *fine_ops,
        *coarse_ops,
        metadata={
            "algorithm": ALGORITHM_ID,
            "refinement_channel_id": str(refinement_channel_id),
        },
    )

    return MaxEntRefinementResult(
        coarse_state=sigma,
        projected_state=projection.state,
        induced_multipliers=projection.multipliers,
        fine_dimension=int(fine.shape[0]),
        coarse_dimension=int(sigma.shape[0]),
        fine_constraint_count=len(fine_ops),
        coarse_constraint_count=len(coarse_ops),
        fine_independent_constraint_count=independent_operator_count(fine_ops),
        coarse_independent_constraint_count=independent_operator_count(coarse_ops),
        moment_residual_l2=projection.moment_residual_l2,
        moment_residual_linf=projection.moment_residual_linf,
        duhamel_hessian_min_eigenvalue=projection.hessian_min_eigenvalue,
        closure_defect_nats=defect,
        trace_norm_residual=residual,
        pinsker_residual_bound=pinsker,
        projection_iterations=projection.iterations,
        projection_converged=projection.converged,
        input_state_validated=True,
        coarse_state_validated=True,
        refinement_channel_id=str(refinement_channel_id),
        evidence_input_sha256=evidence_hash,
        moment_tolerance=float(moment_tolerance),
        closure_defect_tolerance=float(closure_defect_tolerance),
        hessian_floor_tolerance=float(hessian_floor_tolerance),
        numerical_tolerance=float(numerical_tolerance),
    )


def moment_matching_i_projection(
    state: np.ndarray,
    constraints: Sequence[np.ndarray],
    *,
    tolerance: float = 1.0e-11,
    max_iterations: int = 200,
) -> IProjectionResult:
    """Return the unique moment-matching I-projection when Newton descent converges."""

    sigma = _validated_density_matrix(state, name="state")
    operators = _validated_constraints(
        constraints, dimension=sigma.shape[0], name="constraints"
    )
    targets = np.array([_expectation(sigma, op) for op in operators], dtype=float)

    def objective(lam: np.ndarray) -> float:
        _, log_z = gibbs_state(operators, lam)
        return float(log_z + lam @ targets)

    lam = np.zeros(len(operators), dtype=float)
    iterations = 0
    converged = False
    for iterations in range(1, max(1, int(max_iterations)) + 1):
        rho, _ = gibbs_state(operators, lam)
        moments = np.array([_expectation(rho, op) for op in operators], dtype=float)
        gradient = targets - moments
        if float(np.linalg.norm(gradient, ord=2)) <= float(tolerance):
            converged = True
            break
        hessian = duhamel_covariance(operators, lam)
        regularized = hessian + 1.0e-14 * np.eye(len(lam))
        try:
            step = np.linalg.solve(regularized, -gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(regularized, -gradient, rcond=None)[0]
        base = objective(lam)
        scale = 1.0
        while scale > 1.0e-10:
            candidate = lam + scale * step
            if objective(candidate) <= base + 1.0e-15:
                lam = candidate
                break
            scale *= 0.5
        else:
            break

    rho, _ = gibbs_state(operators, lam)
    moments = np.array([_expectation(rho, op) for op in operators], dtype=float)
    mismatch = targets - moments
    residual_l2 = float(np.linalg.norm(mismatch, ord=2))
    residual_linf = float(np.linalg.norm(mismatch, ord=np.inf))
    converged = bool(converged or residual_linf <= float(tolerance))
    hessian_floor = float(np.min(np.linalg.eigvalsh(duhamel_covariance(operators, lam))))
    return IProjectionResult(
        state=rho,
        multipliers=lam,
        moment_residual_l2=residual_l2,
        moment_residual_linf=residual_linf,
        hessian_min_eigenvalue=hessian_floor,
        iterations=iterations,
        converged=converged,
    )


def gibbs_state(
    constraints: Sequence[np.ndarray], multipliers: np.ndarray
) -> tuple[np.ndarray, float]:
    operators = [np.asarray(op, dtype=complex) for op in constraints]
    lam = np.asarray(multipliers, dtype=float)
    if len(operators) != lam.size:
        raise ValueError("one multiplier is required for each constraint")
    hamiltonian = np.zeros_like(operators[0], dtype=complex)
    for coefficient, operator in zip(lam, operators, strict=True):
        hamiltonian += float(coefficient) * operator
    energies, vectors = np.linalg.eigh(_hermitian(hamiltonian))
    minimum = float(np.min(energies))
    weights = np.exp(-(energies - minimum))
    partition = float(np.sum(weights))
    state = (vectors * (weights / partition)) @ vectors.conj().T
    log_z = math.log(partition) - minimum
    return _hermitian(state), float(log_z)


def duhamel_covariance(
    constraints: Sequence[np.ndarray], multipliers: np.ndarray
) -> np.ndarray:
    """Kubo--Mori covariance, the Hessian of the MaxEnt dual objective."""

    operators = [np.asarray(op, dtype=complex) for op in constraints]
    lam = np.asarray(multipliers, dtype=float)
    hamiltonian = np.zeros_like(operators[0], dtype=complex)
    for coefficient, operator in zip(lam, operators, strict=True):
        hamiltonian += float(coefficient) * operator
    energies, vectors = np.linalg.eigh(_hermitian(hamiltonian))
    probabilities = np.exp(-(energies - float(np.min(energies))))
    probabilities /= float(np.sum(probabilities))
    rho = (vectors * probabilities) @ vectors.conj().T
    identity = np.eye(rho.shape[0], dtype=complex)
    centered = [
        vectors.conj().T @ (op - _expectation(rho, op) * identity) @ vectors
        for op in operators
    ]
    log_probabilities = np.log(np.clip(probabilities, 1.0e-300, None))
    p_i, p_j = np.meshgrid(probabilities, probabilities, indexing="ij")
    log_i, log_j = np.meshgrid(log_probabilities, log_probabilities, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = np.where(
            np.abs(log_i - log_j) > 1.0e-12,
            (p_i - p_j) / (log_i - log_j),
            p_i,
        )
    covariance = np.empty((len(operators), len(operators)), dtype=float)
    for left in range(len(operators)):
        for right in range(len(operators)):
            covariance[left, right] = float(
                np.real(np.sum(kernel * centered[left] * centered[right].T))
            )
    return (covariance + covariance.T) / 2.0


def relative_entropy_nats(state: np.ndarray, reference: np.ndarray) -> float:
    sigma = _validated_density_matrix(state, name="state")
    rho = _validated_density_matrix(reference, name="reference")
    if sigma.shape != rho.shape:
        raise ValueError("state and reference dimensions differ")
    sigma_values, sigma_vectors = np.linalg.eigh(sigma)
    rho_values, rho_vectors = np.linalg.eigh(rho)
    positive = sigma_values > 1.0e-15
    entropy_term = float(
        np.sum(sigma_values[positive] * np.log(sigma_values[positive]))
    )
    if float(np.min(rho_values)) <= 0.0:
        raise ValueError("reference must be faithful for finite relative entropy")
    log_rho = (rho_vectors * np.log(rho_values)) @ rho_vectors.conj().T
    cross_term = float(np.real(np.trace(sigma @ log_rho)))
    return float(entropy_term - cross_term)


def trace_norm(matrix: np.ndarray) -> float:
    values = np.linalg.eigvalsh(_hermitian(np.asarray(matrix, dtype=complex)))
    return float(np.sum(np.abs(values)))


def independent_operator_count(operators: Sequence[np.ndarray]) -> int:
    matrices = [np.asarray(op, dtype=complex) for op in operators]
    if not matrices:
        return 0
    identity = np.eye(matrices[0].shape[0], dtype=complex)
    basis = [identity, *matrices]
    gram = np.array(
        [[np.trace(left.conj().T @ right) for right in basis] for left in basis]
    )
    return int(np.linalg.matrix_rank(gram, tol=1.0e-9)) - 1


def _validated_density_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    state = np.asarray(matrix, dtype=complex)
    if state.ndim != 2 or state.shape[0] != state.shape[1] or state.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{name} contains non-finite entries")
    if float(np.linalg.norm(state - state.conj().T, ord="fro")) > 1.0e-9:
        raise ValueError(f"{name} is not Hermitian")
    state = _hermitian(state)
    if abs(float(np.trace(state).real) - 1.0) > 1.0e-9:
        raise ValueError(f"{name} must have unit trace")
    if float(np.min(np.linalg.eigvalsh(state))) < -1.0e-10:
        raise ValueError(f"{name} is not positive semidefinite")
    return state


def _validated_constraints(
    constraints: Sequence[np.ndarray], *, dimension: int, name: str
) -> list[np.ndarray]:
    operators = [np.asarray(op, dtype=complex) for op in constraints]
    if not operators:
        raise ValueError(f"{name} must contain at least one operator")
    for index, operator in enumerate(operators):
        if operator.shape != (dimension, dimension):
            raise ValueError(f"{name}[{index}] has the wrong dimension")
        if not np.all(np.isfinite(operator)):
            raise ValueError(f"{name}[{index}] contains non-finite entries")
        if float(np.linalg.norm(operator - operator.conj().T, ord="fro")) > 1.0e-9:
            raise ValueError(f"{name}[{index}] is not Hermitian")
    return [_hermitian(operator) for operator in operators]


def _expectation(state: np.ndarray, operator: np.ndarray) -> float:
    return float(np.real(np.trace(state @ operator)))


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conj().T) / 2.0


def _evidence_hash(*matrices: np.ndarray, metadata: dict[str, Any]) -> str:
    payload = {
        "schema": "oph_maxent_refinement_evidence_hash_v2",
        "metadata": metadata,
        "matrices": [
            np.ascontiguousarray(np.asarray(matrix, dtype=np.complex128))
            for matrix in matrices
        ],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
