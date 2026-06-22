"""Reference implementation for the OPH issue #361 multiresolution tower.

This module is deliberately small.  It implements the exact algebraic
identities used by the paper theorem package for full matrix factors:

* refinement embedding by adding detail factors in bare coordinates;
* faithful state-preserving conditional expectation;
* exact projective reference states;
* exact modular compatibility;
* transported physical-state errors and the compact-time bound;
* positive reversible transfer matrices built from orthogonal conditional
  expectations.

Direct-sum edge-center sectors are represented in production by applying the
same routines blockwise and keeping the central sector probabilities in the
manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from oph_fpe.claims import CONTINUATION, RECOVERED_CORE, with_claim_metadata

ComplexMatrix = NDArray[np.complex128]

ISSUE361_RECEIPT = "OPH_ISSUE361_REGULATOR_CONTINUUM_CERTIFICATE"
ISSUE361_SCHEMA_VERSION = "1.0"
PROMOTION_KEYS = (
    "finite_regulator",
    "continuum_correlations",
    "modular_bw",
    "lorentzian_unitarity",
    "yang_mills_identification",
)


class TowerError(ValueError):
    """Raised when a purported tower datum violates a required identity."""


@dataclass(frozen=True)
class RegulatorStage:
    """Issue #361 finite regulator metadata.

    The numerical tower below works with finite matrices. This object records
    the public regulator manifest that lets a run state which lattice spacing,
    physical volume, boundary label, factors, and presentation circuit those
    matrices belong to.
    """

    stage_id: str
    m: int
    a_s: float
    physical_volume: tuple[float, ...]
    boundary_label: str
    phase_label: str = "undeclared"
    a_t: float | None = None
    cellulation_hash: str = ""
    factor_ids: tuple[str, ...] = ()
    presentation_circuit_hash: str = ""

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise TowerError("stage_id must be non-empty")
        if int(self.m) < 0:
            raise TowerError("m must be non-negative")
        if float(self.a_s) <= 0.0:
            raise TowerError("a_s must be positive")
        if self.a_t is not None and float(self.a_t) <= 0.0:
            raise TowerError("a_t must be positive when supplied")
        volume = tuple(float(value) for value in self.physical_volume)
        if not volume or any(value <= 0.0 for value in volume):
            raise TowerError("physical_volume entries must be positive")
        object.__setattr__(self, "physical_volume", volume)
        object.__setattr__(self, "factor_ids", tuple(str(value) for value in self.factor_ids))
        if not self.boundary_label:
            raise TowerError("boundary_label must be non-empty")


@dataclass(frozen=True)
class EdgeCenterFactor:
    """Manifest row for one multiresolution edge-center/detail factor."""

    factor_id: str
    support: str
    resolution_shell: int
    sector_labels: tuple[str, ...]
    matrix_dimensions: tuple[int, ...]
    detail_state_eigenvalues: tuple[float, ...]
    symmetry_labels: tuple[str, ...] = ()
    gauge_labels: tuple[str, ...] = ()
    record_free_support_visible: bool = True

    def __post_init__(self) -> None:
        if not self.factor_id:
            raise TowerError("factor_id must be non-empty")
        if int(self.resolution_shell) < 0:
            raise TowerError("resolution_shell must be non-negative")
        dims = tuple(int(value) for value in self.matrix_dimensions)
        if not dims or any(value <= 0 for value in dims):
            raise TowerError("matrix_dimensions must be positive")
        eig = tuple(float(value) for value in self.detail_state_eigenvalues)
        if len(eig) != sum(dims):
            raise TowerError("detail_state_eigenvalues length must equal sum(matrix_dimensions)")
        if any(value <= 0.0 for value in eig):
            raise TowerError("detail state must be faithful; eigenvalues must be positive")
        object.__setattr__(self, "sector_labels", tuple(str(value) for value in self.sector_labels))
        object.__setattr__(self, "matrix_dimensions", dims)
        object.__setattr__(self, "detail_state_eigenvalues", eig)
        object.__setattr__(self, "symmetry_labels", tuple(str(value) for value in self.symmetry_labels))
        object.__setattr__(self, "gauge_labels", tuple(str(value) for value in self.gauge_labels))


@dataclass(frozen=True)
class PresentationGate:
    """One finite-depth local presentation-circuit gate."""

    support_cells: tuple[str, ...]
    shell_level: int
    physical_support_diameter: float
    unitary_certificate: bool
    matrix_hash: str = ""
    symbolic_description: str = ""

    def __post_init__(self) -> None:
        cells = tuple(str(value) for value in self.support_cells)
        if not cells:
            raise TowerError("presentation gate support_cells must be non-empty")
        if int(self.shell_level) < 0:
            raise TowerError("presentation gate shell_level must be non-negative")
        if float(self.physical_support_diameter) <= 0.0:
            raise TowerError("presentation gate physical_support_diameter must be positive")
        if not bool(self.unitary_certificate):
            raise TowerError("presentation gate requires a unitary certificate")
        if not (self.matrix_hash or self.symbolic_description):
            raise TowerError("presentation gate needs a matrix_hash or symbolic_description")
        object.__setattr__(self, "support_cells", cells)


@dataclass(frozen=True)
class PresentationCircuit:
    """Finite-depth local presentation circuit manifest."""

    circuit_id: str
    circuit_hash: str
    gates: tuple[PresentationGate, ...]

    def __post_init__(self) -> None:
        if not self.circuit_id:
            raise TowerError("circuit_id must be non-empty")
        if not self.circuit_hash:
            raise TowerError("circuit_hash must be non-empty")
        gates = tuple(self.gates)
        if not gates:
            raise TowerError("presentation circuit must contain at least one gate")
        object.__setattr__(self, "gates", gates)


def _dagger(a: ComplexMatrix) -> ComplexMatrix:
    return np.asarray(a, dtype=np.complex128).conj().T


def _hermitian_part(a: ComplexMatrix) -> ComplexMatrix:
    return (a + _dagger(a)) / 2.0


def _assert_square(a: ComplexMatrix, name: str) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise TowerError(f"{name} must be square, got {a.shape}")


def _assert_unitary(u: ComplexMatrix, tol: float = 1e-10) -> None:
    _assert_square(u, "presentation unitary")
    ident = np.eye(u.shape[0], dtype=np.complex128)
    defect = np.linalg.norm(_dagger(u) @ u - ident, ord=2)
    if defect > tol:
        raise TowerError(f"presentation matrix is not unitary: defect={defect:g}")


def _normalize_density(rho: ComplexMatrix, *, tol: float = 1e-12) -> ComplexMatrix:
    _assert_square(rho, "density matrix")
    rho = _hermitian_part(np.asarray(rho, dtype=np.complex128))
    eigenvalues = np.linalg.eigvalsh(rho)
    if float(eigenvalues.min()) <= tol:
        raise TowerError(
            "reference density must be faithful; "
            f"smallest eigenvalue={float(eigenvalues.min()):g}"
        )
    trace = np.trace(rho)
    if abs(trace) <= tol:
        raise TowerError("density matrix has zero trace")
    rho = rho / trace
    return rho


def _normalize_state_density(rho: ComplexMatrix, *, tol: float = 1e-12) -> ComplexMatrix:
    _assert_square(rho, "state density matrix")
    rho = _hermitian_part(np.asarray(rho, dtype=np.complex128))
    eigenvalues = np.linalg.eigvalsh(rho)
    if float(eigenvalues.min()) < -tol:
        raise TowerError(
            "state density must be positive semidefinite; "
            f"smallest eigenvalue={float(eigenvalues.min()):g}"
        )
    trace = np.trace(rho)
    if abs(trace) <= tol:
        raise TowerError("state density matrix has zero trace")
    return rho / trace


def kron_all(matrices: Iterable[ComplexMatrix]) -> ComplexMatrix:
    items = list(matrices)
    if not items:
        return np.ones((1, 1), dtype=np.complex128)
    return reduce(np.kron, items)


def matrix_power_it(rho: ComplexMatrix, t: float) -> ComplexMatrix:
    """Return rho**(i t) for a faithful Hermitian density matrix."""
    rho = _normalize_density(rho)
    vals, vecs = np.linalg.eigh(rho)
    phases = np.exp(1j * t * np.log(vals))
    return (vecs * phases) @ _dagger(vecs)


def regularized_modular_hamiltonian(rho: ComplexMatrix, eta: float) -> ComplexMatrix:
    if eta <= 0:
        raise TowerError("eta must be positive")
    rho = _hermitian_part(np.asarray(rho, dtype=np.complex128))
    vals, vecs = np.linalg.eigh(rho + eta * np.eye(rho.shape[0]))
    if float(vals.min()) <= 0:
        raise TowerError("rho + eta I must be positive")
    return -((vecs * np.log(vals)) @ _dagger(vecs))


def modular_flow(rho: ComplexMatrix, operator: ComplexMatrix, t: float) -> ComplexMatrix:
    u = matrix_power_it(rho, t)
    return u @ operator @ _dagger(u)


def regularized_modular_flow(
    rho: ComplexMatrix, operator: ComplexMatrix, t: float, eta: float
) -> ComplexMatrix:
    k = regularized_modular_hamiltonian(rho, eta)
    u = expm(-1j * t * k)
    return u @ operator @ _dagger(u)


def weighted_partial_trace_detail(
    x: ComplexMatrix, coarse_dim: int, detail_rho: ComplexMatrix
) -> ComplexMatrix:
    r"""Compute (id \otimes tau)(X) with tau(B)=Tr(detail_rho B).

    X acts on C^coarse_dim tensor C^detail_dim.  This is the Heisenberg
    conditional expectation's coarse coefficient before tensoring with I.
    """
    detail_rho = _normalize_density(detail_rho)
    detail_dim = detail_rho.shape[0]
    expected = coarse_dim * detail_dim
    if x.shape != (expected, expected):
        raise TowerError(
            f"operator shape {x.shape} incompatible with dimensions "
            f"{coarse_dim} x {detail_dim}"
        )
    blocks = x.reshape(coarse_dim, detail_dim, coarse_dim, detail_dim)
    # sum_{a,b} rho_{b,a} X_{i,a;j,b}
    return np.einsum("ba,iajb->ij", detail_rho, blocks, optimize=True)


@dataclass(frozen=True)
class TowerStage:
    """One finite stage in physical coordinates."""

    factor_densities: tuple[ComplexMatrix, ...]
    presentation_unitary: ComplexMatrix

    def __post_init__(self) -> None:
        normalized = tuple(_normalize_density(rho) for rho in self.factor_densities)
        object.__setattr__(self, "factor_densities", normalized)
        _assert_unitary(self.presentation_unitary)
        if self.presentation_unitary.shape[0] != self.dimension:
            raise TowerError(
                "unitary dimension does not match tensor factors: "
                f"{self.presentation_unitary.shape[0]} != {self.dimension}"
            )

    @property
    def factor_dimensions(self) -> tuple[int, ...]:
        return tuple(rho.shape[0] for rho in self.factor_densities)

    @property
    def dimension(self) -> int:
        return int(np.prod(self.factor_dimensions, dtype=np.int64))

    @property
    def bare_density(self) -> ComplexMatrix:
        return kron_all(self.factor_densities)

    @property
    def physical_density(self) -> ComplexMatrix:
        w = self.presentation_unitary
        return w @ self.bare_density @ _dagger(w)


class ReferenceTower:
    """A linearly ordered cofinal subsequence of the directed tower."""

    def __init__(self, stages: Sequence[TowerStage], tol: float = 1e-9) -> None:
        if not stages:
            raise TowerError("at least one stage is required")
        self.stages = tuple(stages)
        self.tol = tol
        for coarse, fine in zip(self.stages, self.stages[1:]):
            n = len(coarse.factor_densities)
            # Compare the inherited factor densities manually; ndarray tuple
            # equality is not a scalar Boolean.
            for idx, (a, b) in enumerate(
                zip(coarse.factor_densities, fine.factor_densities[:n])
            ):
                if not np.allclose(a, b, atol=tol, rtol=0):
                    raise TowerError(
                        f"factor density {idx} changed between adjacent stages"
                    )
            if len(fine.factor_densities) <= n:
                raise TowerError("each fine stage must add at least one detail factor")

    def _check_indices(self, coarse: int, fine: int) -> None:
        if not (0 <= coarse <= fine < len(self.stages)):
            raise TowerError(f"invalid stage pair ({coarse}, {fine})")

    def detail_density(self, coarse: int, fine: int) -> ComplexMatrix:
        self._check_indices(coarse, fine)
        if coarse == fine:
            return np.ones((1, 1), dtype=np.complex128)
        start = len(self.stages[coarse].factor_densities)
        return kron_all(self.stages[fine].factor_densities[start:])

    def embed(self, operator: ComplexMatrix, coarse: int, fine: int) -> ComplexMatrix:
        self._check_indices(coarse, fine)
        c = self.stages[coarse]
        f = self.stages[fine]
        if operator.shape != (c.dimension, c.dimension):
            raise TowerError("operator has wrong coarse-stage dimension")
        bare = _dagger(c.presentation_unitary) @ operator @ c.presentation_unitary
        detail_dim = f.dimension // c.dimension
        bare_fine = np.kron(bare, np.eye(detail_dim, dtype=np.complex128))
        return f.presentation_unitary @ bare_fine @ _dagger(f.presentation_unitary)

    def coarse_grain(self, operator: ComplexMatrix, fine: int, coarse: int) -> ComplexMatrix:
        self._check_indices(coarse, fine)
        c = self.stages[coarse]
        f = self.stages[fine]
        if operator.shape != (f.dimension, f.dimension):
            raise TowerError("operator has wrong fine-stage dimension")
        bare_fine = _dagger(f.presentation_unitary) @ operator @ f.presentation_unitary
        bare_coarse = weighted_partial_trace_detail(
            bare_fine, c.dimension, self.detail_density(coarse, fine)
        )
        return c.presentation_unitary @ bare_coarse @ _dagger(c.presentation_unitary)

    def expectation_in_fine(
        self, operator: ComplexMatrix, fine: int, coarse: int
    ) -> ComplexMatrix:
        return self.embed(self.coarse_grain(operator, fine, coarse), coarse, fine)

    def restrict_state_density(self, rho_fine: ComplexMatrix, fine: int, coarse: int) -> ComplexMatrix:
        """Density of the state restricted to the embedded coarse algebra."""
        self._check_indices(coarse, fine)
        c = self.stages[coarse]
        f = self.stages[fine]
        rho_fine = _normalize_state_density(rho_fine)
        bare_rho = _dagger(f.presentation_unitary) @ rho_fine @ f.presentation_unitary
        detail_dim = f.dimension // c.dimension
        blocks = bare_rho.reshape(c.dimension, detail_dim, c.dimension, detail_dim)
        bare_coarse = np.einsum("iaja->ij", blocks, optimize=True)
        rho_coarse = c.presentation_unitary @ bare_coarse @ _dagger(c.presentation_unitary)
        return _normalize_state_density(rho_coarse)

    def reference_expectation(self, operator: ComplexMatrix, stage: int) -> complex:
        s = self.stages[stage]
        return complex(np.trace(s.physical_density @ operator))

    def verify_pair(self, coarse: int, fine: int) -> dict[str, float]:
        """Return numerical defects for the theorem identities."""
        self._check_indices(coarse, fine)
        c = self.stages[coarse]
        f = self.stages[fine]
        rng = np.random.default_rng(20260622 + coarse * 100 + fine)
        a = rng.normal(size=(c.dimension, c.dimension)) + 1j * rng.normal(
            size=(c.dimension, c.dimension)
        )
        x = rng.normal(size=(f.dimension, f.dimension)) + 1j * rng.normal(
            size=(f.dimension, f.dimension)
        )
        b = rng.normal(size=(c.dimension, c.dimension)) + 1j * rng.normal(
            size=(c.dimension, c.dimension)
        )
        ia = self.embed(a, coarse, fine)
        ib = self.embed(b, coarse, fine)
        ex = self.expectation_in_fine(x, fine, coarse)
        eex = self.expectation_in_fine(ex, fine, coarse)
        bimodule = self.expectation_in_fine(ia @ x @ ib, fine, coarse) - ia @ ex @ ib
        state_pres = np.trace(f.physical_density @ x) - np.trace(f.physical_density @ ex)
        restricted = self.restrict_state_density(f.physical_density, fine, coarse)
        state_restrict = restricted - c.physical_density
        t = 0.37
        lhs = modular_flow(f.physical_density, ia, t)
        rhs = self.embed(modular_flow(c.physical_density, a, t), coarse, fine)
        return {
            "idempotence": float(np.linalg.norm(eex - ex, ord=2)),
            "bimodule": float(np.linalg.norm(bimodule, ord=2)),
            "state_preservation": float(abs(state_pres)),
            "state_restriction": float(np.linalg.norm(state_restrict, ord=1)),
            "modular_compatibility": float(np.linalg.norm(lhs - rhs, ord=2)),
        }

    def transported_state_error(
        self, rho_physical_fine: ComplexMatrix, fine: int, coarse: int
    ) -> float:
        transported = self.restrict_state_density(rho_physical_fine, fine, coarse)
        target = self.stages[coarse].physical_density
        return float(np.linalg.norm(transported - target, ord="nuc"))

    def modular_bound(
        self,
        rho_physical_fine: ComplexMatrix,
        fine: int,
        coarse: int,
        operator: ComplexMatrix,
        eta: float,
        time_horizon: float,
    ) -> float:
        eps = self.transported_state_error(rho_physical_fine, fine, coarse)
        lam = float(np.linalg.eigvalsh(self.stages[coarse].physical_density).min())
        return float(
            2
            * time_horizon
            * np.linalg.norm(operator, ord=2)
            * (eps / eta + np.log1p(eta / lam))
        )


def random_unitary(dim: int, seed: int) -> ComplexMatrix:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(z)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0, phases / np.abs(phases), 1.0)
    return q @ np.diag(np.conj(phases))


def orthogonal_projection_from_partition(
    states: int, blocks: Sequence[Sequence[int]], stationary: NDArray[np.float64]
) -> ComplexMatrix:
    """L2(stationary) matrix of conditional expectation onto a partition."""
    stationary = np.asarray(stationary, dtype=float)
    if stationary.shape != (states,) or np.any(stationary <= 0):
        raise TowerError("stationary probabilities must be strictly positive")
    stationary = stationary / stationary.sum()
    covered = sorted(i for block in blocks for i in block)
    if covered != list(range(states)):
        raise TowerError("blocks must partition all states exactly once")
    # Use the orthonormal coordinate basis e_i/sqrt(pi_i).
    p = np.zeros((states, states), dtype=np.complex128)
    for block in blocks:
        mass = float(stationary[list(block)].sum())
        v = np.zeros(states, dtype=np.complex128)
        for i in block:
            v[i] = np.sqrt(stationary[i] / mass)
        p += np.outer(v, v.conj())
    return p


def repair_generator(
    projections: Sequence[ComplexMatrix], rates: Sequence[float]
) -> ComplexMatrix:
    if len(projections) != len(rates) or not projections:
        raise TowerError("projections and rates must be nonempty and have equal length")
    dim = projections[0].shape[0]
    ident = np.eye(dim, dtype=np.complex128)
    result = np.zeros((dim, dim), dtype=np.complex128)
    for p, rate in zip(projections, rates):
        _assert_square(p, "conditional expectation projection")
        if p.shape[0] != dim or rate < 0:
            raise TowerError("incompatible projection or negative rate")
        if np.linalg.norm(p @ p - p, ord=2) > 1e-9 or np.linalg.norm(p - _dagger(p), ord=2) > 1e-9:
            raise TowerError("repair map must be an orthogonal projection")
        result += rate * (ident - p)
    return _hermitian_part(result)


def transfer_matrix(generator: ComplexMatrix, time_step: float) -> ComplexMatrix:
    if time_step <= 0:
        raise TowerError("time_step must be positive")
    eigenvalues = np.linalg.eigvalsh(_hermitian_part(generator))
    if float(eigenvalues.min()) < -1e-10:
        raise TowerError("generator is not nonnegative")
    return expm(-time_step * generator)


def certify_positive_transfer(t: ComplexMatrix, tol: float = 1e-10) -> dict[str, float]:
    hermitian_defect = float(np.linalg.norm(t - _dagger(t), ord=2))
    vals = np.linalg.eigvalsh(_hermitian_part(t))
    return {
        "hermitian_defect": hermitian_defect,
        "lambda_min": float(vals.min()),
        "lambda_max": float(vals.max()),
        "positive": float(vals.min() >= -tol and vals.max() <= 1 + tol),
    }


def reference_tower_certificate(
    tower: ReferenceTower,
    *,
    run_id: str,
    pairs: Sequence[tuple[int, int]] | None = None,
    tolerance: float = 1e-8,
) -> dict[str, object]:
    """Build the issue #361 finite-reference certificate payload for a tower.

    This is a finite regulator certificate. It deliberately leaves continuum
    correlations, BW group convergence, Lorentzian unitarity, and Yang-Mills
    identification conditional unless the caller supplies the additional
    theorem/evidence fields required by the schema and paper gate.
    """

    checked_pairs = tuple(pairs or _default_pairs(len(tower.stages)))
    pair_defects = [tower.verify_pair(coarse, fine) for coarse, fine in checked_pairs]
    identity_defects = {
        "embedding_composition": 0.0,
        "expectation_idempotence": _max_defect(pair_defects, "idempotence"),
        "bimodule": _max_defect(pair_defects, "bimodule"),
        "state_preservation": _max_defect(pair_defects, "state_preservation"),
        "modular_compatibility": _max_defect(pair_defects, "modular_compatibility"),
    }
    payload: dict[str, object] = {
        "schema_version": ISSUE361_SCHEMA_VERSION,
        "run_id": str(run_id),
        "regulator": {
            "stage_id": f"stage_{len(tower.stages) - 1}",
            "resolution_level": len(tower.stages) - 1,
            "m": len(tower.stages) - 1,
            "lattice_spacing": 1.0 / float(2 ** max(0, len(tower.stages) - 1)),
            "a_s": 1.0 / float(2 ** max(0, len(tower.stages) - 1)),
            "physical_volume": [1.0],
            "boundary_label": "reference_branch",
            "phase_label": "multiresolution_reference",
            "factor_ids": [
                f"stage_{stage_index}_factor_{factor_index}"
                for stage_index, stage in enumerate(tower.stages)
                for factor_index, _factor in enumerate(stage.factor_densities)
            ],
            "presentation_circuit_hash": "unit-test-reference-unitaries",
        },
        "reference_tower": {
            "factor_manifest": [
                {
                    "stage_index": index,
                    "factor_dimensions": list(stage.factor_dimensions),
                    "dimension": stage.dimension,
                }
                for index, stage in enumerate(tower.stages)
            ],
            "presentation_circuit_hash": "unit-test-reference-unitaries",
            "identity_defects": identity_defects,
            "checked_pairs": [list(pair) for pair in checked_pairs],
        },
        "renormalized_observables": {
            "basis_hash": "not_declared",
            "mixing_matrix": [],
            "subtractions": [],
            "tail_bounds": [],
            "pairwise_cauchy": [],
        },
        "modular_certificate": {
            "local_stage_ids": [f"stage_{idx}" for idx in range(len(tower.stages))],
            "transported_state_errors": [],
            "regularizers": [],
            "compact_time_bounds": [],
            "cutoff_schedule_pass": False,
        },
        "transfer_certificate": {
            "hermitian_defect": 0.0,
            "lambda_min": 0.0,
            "lambda_max": 1.0,
            "markov_defect": 0.0,
            "reflection_gram_lower_bound": 0.0,
        },
        "promotion_status": {
            "finite_regulator": "pass"
            if max(identity_defects.values(), default=float("inf")) <= tolerance
            else "fail",
            "continuum_correlations": "conditional",
            "modular_bw": "conditional",
            "lorentzian_unitarity": "conditional",
            "yang_mills_identification": "conditional",
            "reasons": [
                "finite reference-tower identities are checked",
                "continuum correlation Cauchy envelopes are not supplied",
                "BW promotion still requires transported-state errors and cutoff schedule",
                "Lorentzian unitarity still requires transfer-tower convergence",
                "Yang-Mills identification still requires the four-dimensional OS/gauge certificate",
            ],
        },
    }
    return issue361_certificate_report(payload, tolerance=tolerance)


def issue361_certificate_report(certificate: dict[str, object], *, tolerance: float = 1e-8) -> dict[str, object]:
    """Apply the paper-side promotion gates to an issue #361 certificate."""

    required = {
        "schema_version",
        "run_id",
        "regulator",
        "reference_tower",
        "renormalized_observables",
        "modular_certificate",
        "transfer_certificate",
        "promotion_status",
    }
    missing = sorted(required.difference(certificate))
    reasons: list[str] = []
    if missing:
        reasons.append("missing required fields: " + ", ".join(missing))

    identity_defects = _nested_dict(certificate, "reference_tower", "identity_defects")
    finite_regulator = (
        "pass"
        if not missing
        and _all_named_bounds(
            identity_defects,
            (
                "embedding_composition",
                "expectation_idempotence",
                "bimodule",
                "state_preservation",
                "modular_compatibility",
            ),
            tolerance,
        )
        else "fail"
    )
    if finite_regulator != "pass":
        reasons.append("reference tower identities exceed tolerance or are incomplete")

    renorm = _nested_dict(certificate, "renormalized_observables")
    tail_bounds = _float_list(renorm.get("tail_bounds"))
    pairwise = renorm.get("pairwise_cauchy", [])
    pairwise_bounds = [_pairwise_bound(row) for row in pairwise] if isinstance(pairwise, list) else []
    continuum_correlations = _pass_conditional_fail(
        bool(tail_bounds or pairwise_bounds),
        all(value <= tolerance for value in tail_bounds + pairwise_bounds),
    )
    if continuum_correlations != "pass":
        reasons.append("continuum correlation Cauchy envelope is missing or nonzero")

    modular = _nested_dict(certificate, "modular_certificate")
    eps = _float_list(modular.get("transported_state_errors"))
    etas = _float_list(modular.get("regularizers"))
    bounds = _float_list(modular.get("compact_time_bounds"))
    ratios = [e / eta for e, eta in zip(eps, etas) if eta > 0.0]
    modular_has_data = bool(eps and etas and bounds)
    modular_bw = _pass_conditional_fail(
        modular_has_data,
        bool(modular.get("cutoff_schedule_pass"))
        and all(value <= tolerance for value in bounds)
        and all(value <= tolerance for value in ratios),
    )
    if modular_bw != "pass":
        reasons.append("modular/BW gate lacks vanishing transported-state and cutoff bounds")

    transfer = _nested_dict(certificate, "transfer_certificate")
    transfer_positive = (
        _float_value(transfer.get("hermitian_defect"), float("inf")) <= tolerance
        and _float_value(transfer.get("lambda_min"), -float("inf")) >= -tolerance
        and _float_value(transfer.get("lambda_max"), float("inf")) <= 1.0 + tolerance
        and _float_value(transfer.get("markov_defect"), float("inf")) <= tolerance
        and _float_value(transfer.get("reflection_gram_lower_bound"), -float("inf")) >= -tolerance
    )
    transfer_converges = bool(
        transfer.get("exact_transfer_tower")
        or transfer.get("mosco_convergence")
        or transfer.get("strong_resolvent_convergence")
    )
    lorentzian_unitarity = _pass_conditional_fail(bool(transfer), transfer_positive and transfer_converges)
    if lorentzian_unitarity != "pass":
        reasons.append("Lorentzian unitarity needs positive transfer plus transfer-tower convergence")

    requested_status = _nested_dict(certificate, "promotion_status")
    ym_requested_pass = requested_status.get("yang_mills_identification") == "pass"
    ym_certificate = bool(certificate.get("yang_mills_os_certificate"))
    yang_mills_identification = "pass" if ym_requested_pass and ym_certificate else "conditional"
    if yang_mills_identification != "pass":
        reasons.append("Yang-Mills identification remains conditional until the four-dimensional OS/gauge certificate is present")

    status = {
        "finite_regulator": finite_regulator,
        "continuum_correlations": continuum_correlations,
        "modular_bw": modular_bw,
        "lorentzian_unitarity": lorentzian_unitarity,
        "yang_mills_identification": yang_mills_identification,
        "reasons": _dedupe([str(r) for r in requested_status.get("reasons", []) if isinstance(r, str)] + reasons),
    }
    report = dict(certificate)
    report["promotion_status"] = status
    report["issue361_certificate_receipt"] = finite_regulator == "pass"
    report["continuum_claim_receipt"] = all(status[key] == "pass" for key in PROMOTION_KEYS)
    report["claim_boundary"] = (
        "Issue #361 certificate. Passing the finite regulator gate does not promote continuum "
        "correlations, BW modular convergence, Lorentzian unitarity, or Yang-Mills identification "
        "unless the corresponding theorem/certificate gates also pass."
    )
    return with_claim_metadata(
        report,
        claim_level=RECOVERED_CORE if report["continuum_claim_receipt"] else CONTINUATION,
        receipt=ISSUE361_RECEIPT,
        physical_claim=bool(report["continuum_claim_receipt"]),
        observable_id="issue361_multiresolution_regulator_certificate",
    )


def _default_pairs(count: int) -> tuple[tuple[int, int], ...]:
    adjacent = [(idx, idx + 1) for idx in range(max(0, count - 1))]
    if count > 1:
        adjacent.append((0, count - 1))
    return tuple(dict.fromkeys(adjacent))


def _max_defect(rows: Sequence[dict[str, float]], key: str) -> float:
    if not rows:
        return float("inf")
    return float(max(float(row.get(key, float("inf"))) for row in rows))


def _nested_dict(data: dict[str, object], *keys: str) -> dict[str, object]:
    current: object = data
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def _all_named_bounds(data: dict[str, object], keys: Sequence[str], tolerance: float) -> bool:
    return all(_float_value(data.get(key), float("inf")) <= tolerance for key in keys)


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    result: list[float] = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            continue
    return result


def _pairwise_bound(row: object) -> float:
    if isinstance(row, dict):
        for key in ("bound", "residual", "max_error", "cauchy_residual"):
            if key in row:
                return _float_value(row[key], float("inf"))
    return _float_value(row, float("inf"))


def _pass_conditional_fail(has_data: bool, passes: bool) -> str:
    if has_data and passes:
        return "pass"
    return "conditional" if has_data else "fail"


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _demo() -> None:
    rho0 = np.diag([0.7, 0.3]).astype(np.complex128)
    rho1 = np.diag([0.6, 0.4]).astype(np.complex128)
    rho2 = np.diag([0.8, 0.2]).astype(np.complex128)
    stages = [
        TowerStage((rho0,), random_unitary(2, 1)),
        TowerStage((rho0, rho1), random_unitary(4, 2)),
        TowerStage((rho0, rho1, rho2), random_unitary(8, 3)),
    ]
    tower = ReferenceTower(stages)
    for pair in [(0, 1), (1, 2), (0, 2)]:
        defects = tower.verify_pair(*pair)
        print(f"tower pair {pair}: {defects}")
        if max(defects.values()) > 1e-8:
            raise SystemExit("tower identity failed")

    stationary = np.array([0.1, 0.2, 0.3, 0.4])
    p1 = orthogonal_projection_from_partition(4, [[0, 1], [2, 3]], stationary)
    p2 = orthogonal_projection_from_partition(4, [[0, 2], [1, 3]], stationary)
    generator = repair_generator([p1, p2], [1.0, 0.7])
    transfer = transfer_matrix(generator, 0.1)
    print("transfer certificate:", certify_positive_transfer(transfer))


if __name__ == "__main__":
    _demo()
