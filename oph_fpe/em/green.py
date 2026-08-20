"""Exact discrete Green solve and the #733 handoff observables.

Design-only, non-evidential module.  The Green matrix ``G`` is the unique
symmetric solution of ``L G = I - J/12`` with ``G 1 = 0`` on the committed
twelve-port seam Laplacian, computed over ``fractions.Fraction`` as
``(L + J)^{-1} - J/144`` and verified against the defining identities before
use (fail-closed).

The three handoff observables, with the definitions of the committed
handoff block:

* ``port_potential_difference``: ``phi = G . rho`` for a neutral rational
  load ``rho``; ``observable(p, q) = phi(p) - phi(q)``.
* ``seam_flux``: ``coulombField rho e`` on the thirty committed seams,
  the coboundary of the Green potential.
* ``chord_field_strength_component``: ``fieldStrength A`` on the nineteen
  chord seams of the committed spanning tree, with
  ``fieldStrength A = A - d(G(boundary A))`` the exact cycle projection
  (Lean anchor ``fieldStrength_determines_up_to_gauge``); exact rational
  on rational inputs.

Boundaries.  The word Coulomb names the canonical discrete Gauss solution
of the committed finite table only.  No port load is identified with a
physical charge and no seam value with a physical field or flux (PR-53,
PR-54 open).  Nothing here arms an instrument or freezes a decision rule.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Sequence

from oph_fpe.em.base_carrier import (
    CHORD_SEAMS,
    PORTS,
    SEAMS,
    Matrix,
    Vector,
    ZERO,
    ONE,
    boundary,
    coboundary,
    fundamental_cycles,
    laplacian_matrix,
    seam_energy,
    seam_inner,
    tree_solution,
)


class GreenSolveError(ValueError):
    """The exact Green solve failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GreenSolveError(message)


def _invert(matrix: Matrix) -> Matrix:
    n = len(matrix)
    work = [
        list(row) + [ONE if i == j else ZERO for j in range(n)]
        for i, row in enumerate(matrix)
    ]
    for col in range(n):
        pivot_row = next((r for r in range(col, n) if work[r][col] != 0), None)
        _require(pivot_row is not None, "regularized Laplacian is singular")
        work[col], work[pivot_row] = work[pivot_row], work[col]
        pivot = work[col][col]
        work[col] = [x / pivot for x in work[col]]
        for r in range(n):
            if r != col and work[r][col] != 0:
                factor = work[r][col]
                work[r] = [
                    x - factor * y
                    for x, y in zip(work[r], work[col], strict=True)
                ]
    return [row[n:] for row in work]


def _verify_green_identities(lap: Matrix, green: Matrix) -> None:
    for i in range(PORTS):
        _require(sum(green[i], start=ZERO) == 0, "green row sum drift")
        for j in range(PORTS):
            _require(green[i][j] == green[j][i], "green symmetry drift")
            product = sum(
                (lap[i][k] * green[k][j] for k in range(PORTS)), start=ZERO
            )
            expected = (ONE if i == j else ZERO) - Fraction(1, 12)
            _require(product == expected, "green defining identity drift")


@lru_cache(maxsize=1)
def _green_matrix_cached() -> tuple[tuple[Fraction, ...], ...]:
    lap = laplacian_matrix()
    regularized = [
        [lap[i][j] + ONE for j in range(PORTS)] for i in range(PORTS)
    ]
    inverse = _invert(regularized)
    green = [
        [inverse[i][j] - Fraction(1, 144) for j in range(PORTS)]
        for i in range(PORTS)
    ]
    _verify_green_identities(lap, green)
    return tuple(tuple(row) for row in green)


def green_matrix() -> Matrix:
    """The exact Green matrix, identity-verified before return."""
    return [list(row) for row in _green_matrix_cached()]


def _as_neutral_load(rho: Sequence[Fraction]) -> Vector:
    _require(len(rho) == PORTS, "load arity drift")
    load = [Fraction(x) for x in rho]
    _require(sum(load, start=ZERO) == 0, "load is not neutral")
    return load


def green_potential(rho: Sequence[Fraction]) -> Vector:
    """``phi = G . rho`` for a neutral rational load."""
    load = _as_neutral_load(rho)
    green = _green_matrix_cached()
    return [
        sum((green[p][q] * load[q] for q in range(PORTS)), start=ZERO)
        for p in range(PORTS)
    ]


def port_potential_difference(
    rho: Sequence[Fraction], p: int, q: int
) -> Fraction:
    """Handoff observable 1: ``phi(p) - phi(q)`` with ``phi = G . rho``."""
    phi = green_potential(rho)
    return phi[p] - phi[q]


def coulomb_field(rho: Sequence[Fraction]) -> Vector:
    """The canonical discrete Coulomb field ``d(G . rho)`` on the thirty
    committed seams."""
    return coboundary(green_potential(rho))


def seam_flux(rho: Sequence[Fraction]) -> Vector:
    """Handoff observable 2: ``coulombField rho e`` on the thirty committed
    seams."""
    return coulomb_field(rho)


def field_strength(seam_field: Sequence[Fraction]) -> Vector:
    """The exact cycle projection ``A - d(G(boundary A))``: gauge-invariant,
    zero exactly on port gradients, the identity on cycles."""
    _require(len(seam_field) == SEAMS, "field strength arity drift")
    field = [Fraction(x) for x in seam_field]
    load = boundary(field)
    green = _green_matrix_cached()
    potential = [
        sum((green[p][q] * load[q] for q in range(PORTS)), start=ZERO)
        for p in range(PORTS)
    ]
    gradient = coboundary(potential)
    return [a - g for a, g in zip(field, gradient, strict=True)]


def chord_field_strength_components(
    seam_field: Sequence[Fraction],
) -> dict[int, Fraction]:
    """Handoff observable 3: ``fieldStrength A`` on the nineteen chord seams
    of the committed spanning tree, keyed by chord seam index."""
    strength = field_strength(seam_field)
    return {chord: strength[chord] for chord in CHORD_SEAMS}


def thomson_decomposition(rho: Sequence[Fraction]) -> dict:
    """Exact Thomson decomposition of the committed tree Gauss solution:
    ``tree = coulomb + cycle_part`` with orthogonal parts and exactly
    additive energies."""
    load = _as_neutral_load(rho)
    coulomb = coulomb_field(load)
    _require(boundary(coulomb) == load, "coulomb gauss drift")
    tree = tree_solution(load)
    cycle_part = [t - c for t, c in zip(tree, coulomb, strict=True)]
    _require(
        all(x == 0 for x in boundary(cycle_part)),
        "cycle part boundary drift",
    )
    cycles = fundamental_cycles()
    _require(
        all(seam_inner(coulomb, cycle) == 0 for cycle in cycles.values()),
        "coulomb cycle-orthogonality drift",
    )
    coulomb_energy = seam_energy(coulomb)
    tree_energy = seam_energy(tree)
    cycle_energy = seam_energy(cycle_part)
    _require(
        tree_energy == coulomb_energy + cycle_energy,
        "thomson energy decomposition drift",
    )
    return {
        "coulomb": coulomb,
        "tree": tree,
        "cycle_part": cycle_part,
        "coulomb_energy": coulomb_energy,
        "tree_solution_energy": tree_energy,
        "cycle_part_energy": cycle_energy,
    }
