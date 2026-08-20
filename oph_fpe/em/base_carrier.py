"""Exact-rational base-carrier objects for the #733 electromagnetism handoff.

Design-only, non-evidential module.  The committed base carrier is the
twelve-port, thirty-seam, twenty-oriented-face icosahedral complex pinned by
the RER source contracts (``Lean/Screen/SeamCurrentCarrierQuotient.lean`` for
the seam table, ``Lean/ObserverPatchHolography/CoreAxioms.lean`` for the
oriented faces).  This module re-states those finite tables verbatim and
derives every incidence object over ``fractions.Fraction``:

* the port coboundary ``d`` (thirty seams by twelve ports, entry
  ``[p = right(e)] - [p = left(e)]``), its transpose boundary, and the
  seam Laplacian ``L = boundary . coboundary``;
* the committed oriented face incidence ``C`` (twenty faces by thirty
  seams, ``+1`` where the committed smaller-to-larger seam orientation
  agrees with the face orientation, ``-1`` where it is opposite);
* the committed spanning tree, its nineteen chords, and the fundamental
  chord cycles;
* the structural receipt: rank ``C`` = 19, ``C . d = 0``, kernel of ``C``
  equal to the image of ``d`` (dimension 11), kernel of ``d`` and of ``L``
  equal to the constants (dimension 1), the sparse local operator
  ``C^T C`` with diagonal 2 and five nonzero entries per row, and the
  boundary-of-boundary orientation control with committed value ``-2``.

Boundaries.  Everything here is finite exact mathematics on the committed
base carrier.  No port load is a physical charge, no seam value is a
physical field or flux, and the table is not physical space (open premise
rows PR-53 and PR-54).  Nothing in this module is a frozen instrument or a
physical claim.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

PORTS = 12
SEAMS = 30
FACES = 20

# Committed thirty-seam incidence table (seamLeft/seamRight), pinned by the
# RER receipt to Lean/Screen/SeamCurrentCarrierQuotient.lean.  Orientation:
# left < right on every seam.
SEAM_LEFT: tuple[int, ...] = (
    0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3,
    4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10,
)
SEAM_RIGHT: tuple[int, ...] = (
    1, 2, 3, 4, 6, 2, 3, 5, 7, 4, 5, 8, 6, 7, 9,
    6, 8, 10, 7, 8, 11, 9, 10, 9, 11, 10, 11, 10, 11, 11,
)

# Committed twenty oriented faces, pinned to
# Lean/ObserverPatchHolography/CoreAxioms.lean (orientedFaces) and restated
# by Lean/Screen/LocalFaceMaxwellAction.lean (faceVertices).
ORIENTED_FACES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2), (0, 3, 1), (0, 2, 4), (0, 6, 3), (0, 4, 6),
    (1, 5, 2), (1, 3, 7), (1, 7, 5), (2, 8, 4), (2, 5, 8),
    (3, 6, 9), (3, 9, 7), (4, 10, 6), (4, 8, 10), (5, 7, 11),
    (5, 11, 8), (6, 10, 9), (7, 9, 11), (8, 11, 10), (9, 10, 11),
)

# Committed spanning tree of the seam graph and its nineteen chords, as
# pinned in the committed receipt (spanning_tree block).
TREE_SEAMS: tuple[int, ...] = (0, 1, 2, 3, 4, 7, 8, 11, 14, 17, 20)
CHORD_SEAMS: tuple[int, ...] = tuple(
    e for e in range(SEAMS) if e not in TREE_SEAMS
)

# Committed orientation-erasure control value: with the signed face
# incidence replaced by its absolute value, the boundary-of-boundary sum on
# face zero at port zero equals -2 instead of 0
# (Lean theorem unsigned_face_incidence_breaks_boundary; mirrored by the
# adversarial_controls block of the local face receipt).
UNSIGNED_FACE_ZERO_PORT_ZERO_CONTROL = -2

ZERO = Fraction(0)
ONE = Fraction(1)

Vector = list[Fraction]
Matrix = list[list[Fraction]]


class BaseCarrierError(ValueError):
    """A base-carrier structural receipt failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaseCarrierError(message)


# ---------------------------------------------------------------------------
# Port-seam incidence d and its derived operators
# ---------------------------------------------------------------------------

def incidence_entry(e: int, p: int) -> Fraction:
    """Signed incidence of seam ``e`` at port ``p``: the committed
    orientation credits ``seamRight`` and debits ``seamLeft``."""
    value = 0
    if p == SEAM_RIGHT[e]:
        value += 1
    if p == SEAM_LEFT[e]:
        value -= 1
    return Fraction(value)


def coboundary_matrix() -> Matrix:
    """The base incidence ``d`` as a thirty-by-twelve matrix:
    ``(d phi)(e) = phi(right(e)) - phi(left(e))``."""
    return [
        [incidence_entry(e, p) for p in range(PORTS)] for e in range(SEAMS)
    ]


def boundary_matrix() -> Matrix:
    """The transpose of ``d``: twelve ports by thirty seams."""
    return [
        [incidence_entry(e, p) for e in range(SEAMS)] for p in range(PORTS)
    ]


def coboundary(phi: Sequence[Fraction]) -> Vector:
    """``(d phi)(e) = phi(right(e)) - phi(left(e))`` on the thirty seams."""
    _require(len(phi) == PORTS, "coboundary input arity drift")
    return [phi[SEAM_RIGHT[e]] - phi[SEAM_LEFT[e]] for e in range(SEAMS)]


def boundary(current: Sequence[Fraction]) -> Vector:
    """``(boundary c)(p) = sum_e ([p = right(e)] - [p = left(e)]) c(e)``."""
    _require(len(current) == SEAMS, "boundary input arity drift")
    load = [ZERO] * PORTS
    for e in range(SEAMS):
        load[SEAM_RIGHT[e]] += current[e]
        load[SEAM_LEFT[e]] -= current[e]
    return load


def laplacian_matrix() -> Matrix:
    """Seam Laplacian ``L = boundary . coboundary`` on the twelve ports."""
    lap = [[ZERO] * PORTS for _ in range(PORTS)]
    for e in range(SEAMS):
        l, r = SEAM_LEFT[e], SEAM_RIGHT[e]
        lap[l][l] += ONE
        lap[r][r] += ONE
        lap[l][r] -= ONE
        lap[r][l] -= ONE
    return lap


def laplacian_apply(phi: Sequence[Fraction]) -> Vector:
    return boundary(coboundary(phi))


# ---------------------------------------------------------------------------
# Committed oriented face incidence C and its derived operators
# ---------------------------------------------------------------------------

def face_directed_edges(f: int) -> tuple[tuple[int, int], ...]:
    """The three directed boundary edges of committed face ``f``."""
    a, b, c = ORIENTED_FACES[f]
    return ((a, b), (b, c), (c, a))


def face_incidence_entry(f: int, e: int) -> Fraction:
    """``+1`` when the committed smaller-to-larger seam orientation agrees
    with the face orientation, ``-1`` when it is opposite, ``0`` when the
    seam does not bound the face."""
    edges = face_directed_edges(f)
    if (SEAM_LEFT[e], SEAM_RIGHT[e]) in edges:
        return ONE
    if (SEAM_RIGHT[e], SEAM_LEFT[e]) in edges:
        return -ONE
    return ZERO


def face_incidence_matrix() -> Matrix:
    """The committed oriented face incidence ``C``: twenty by thirty."""
    return [
        [face_incidence_entry(f, e) for e in range(SEAMS)]
        for f in range(FACES)
    ]


def face_curvature(
    seam_field: Sequence[Fraction], face_matrix: Matrix | None = None
) -> Vector:
    """``(C A)(f)``: the oriented sum of the three seam values on face
    ``f``."""
    _require(len(seam_field) == SEAMS, "face curvature input arity drift")
    c = face_incidence_matrix() if face_matrix is None else face_matrix
    return [
        sum((c[f][e] * seam_field[e] for e in range(SEAMS)), start=ZERO)
        for f in range(FACES)
    ]


def face_codifferential(
    face_field: Sequence[Fraction], face_matrix: Matrix | None = None
) -> Vector:
    """``(C^T F)(e) = sum_f C[f][e] F(f)`` on the thirty seams."""
    _require(len(face_field) == FACES, "face codifferential arity drift")
    c = face_incidence_matrix() if face_matrix is None else face_matrix
    return [
        sum((c[f][e] * face_field[f] for f in range(FACES)), start=ZERO)
        for e in range(SEAMS)
    ]


def local_maxwell_operator(
    seam_field: Sequence[Fraction], face_matrix: Matrix | None = None
) -> Vector:
    """The sparse local operator ``C^T C`` applied to a seam field."""
    return face_codifferential(
        face_curvature(seam_field, face_matrix), face_matrix
    )


def local_kinetic_matrix() -> Matrix:
    """``C^T C`` as a thirty-by-thirty matrix."""
    c = face_incidence_matrix()
    return [
        [
            sum((c[f][e] * c[f][d] for f in range(FACES)), start=ZERO)
            for d in range(SEAMS)
        ]
        for e in range(SEAMS)
    ]


# ---------------------------------------------------------------------------
# Equal-weight pairings
# ---------------------------------------------------------------------------

def port_inner(x: Sequence[Fraction], y: Sequence[Fraction]) -> Fraction:
    _require(len(x) == PORTS and len(y) == PORTS, "port inner arity drift")
    return sum((a * b for a, b in zip(x, y, strict=True)), start=ZERO)


def seam_inner(x: Sequence[Fraction], y: Sequence[Fraction]) -> Fraction:
    _require(len(x) == SEAMS and len(y) == SEAMS, "seam inner arity drift")
    return sum((a * b for a, b in zip(x, y, strict=True)), start=ZERO)


def seam_energy(x: Sequence[Fraction]) -> Fraction:
    return seam_inner(x, x)


def face_inner(x: Sequence[Fraction], y: Sequence[Fraction]) -> Fraction:
    _require(len(x) == FACES and len(y) == FACES, "face inner arity drift")
    return sum((a * b for a, b in zip(x, y, strict=True)), start=ZERO)


def face_energy(x: Sequence[Fraction]) -> Fraction:
    return face_inner(x, x)


# ---------------------------------------------------------------------------
# Exact rank over the rationals
# ---------------------------------------------------------------------------

def matrix_rank(matrix: Sequence[Sequence[Fraction]]) -> int:
    """Exact rank by fraction-free-safe Gauss elimination over ``Fraction``."""
    work = [list(row) for row in matrix]
    if not work:
        return 0
    n_rows, n_cols = len(work), len(work[0])
    rank = 0
    for col in range(n_cols):
        pivot_row = next(
            (r for r in range(rank, n_rows) if work[r][col] != 0), None
        )
        if pivot_row is None:
            continue
        work[rank], work[pivot_row] = work[pivot_row], work[rank]
        pivot = work[rank][col]
        work[rank] = [x / pivot for x in work[rank]]
        for r in range(n_rows):
            if r != rank and work[r][col] != 0:
                factor = work[r][col]
                work[r] = [
                    x - factor * y
                    for x, y in zip(work[r], work[rank], strict=True)
                ]
        rank += 1
    return rank


# ---------------------------------------------------------------------------
# Committed spanning tree, tree Gauss solution, fundamental chord cycles
# ---------------------------------------------------------------------------

def _tree_parents() -> dict[int, tuple[int, int] | None]:
    """Breadth-first parent map of the committed tree rooted at port 0:
    ``node -> (parent_node, parent_seam)``; the root maps to ``None``."""
    adjacency: dict[int, list[tuple[int, int]]] = {p: [] for p in range(PORTS)}
    for e in TREE_SEAMS:
        adjacency[SEAM_LEFT[e]].append((SEAM_RIGHT[e], e))
        adjacency[SEAM_RIGHT[e]].append((SEAM_LEFT[e], e))
    parent: dict[int, tuple[int, int] | None] = {0: None}
    queue = [0]
    while queue:
        node = queue.pop(0)
        for neighbour, seam in adjacency[node]:
            if neighbour not in parent:
                parent[neighbour] = (node, seam)
                queue.append(neighbour)
    _require(len(parent) == PORTS, "committed tree does not span the ports")
    return parent


def tree_solution(load: Sequence[Fraction]) -> Vector:
    """The unique Gauss solution supported on the committed spanning tree:
    ``boundary(result) = load`` with ``result`` zero on every chord."""
    _require(len(load) == PORTS, "tree solve arity drift")
    _require(sum(load, start=ZERO) == 0, "tree solve requires a neutral load")
    parent = _tree_parents()
    order = [0]
    seen = {0}
    while len(order) < PORTS:
        for node in range(PORTS):
            if node in seen:
                continue
            entry = parent[node]
            if entry is not None and entry[0] in seen:
                order.append(node)
                seen.add(node)
    subtree = {p: Fraction(load[p]) for p in range(PORTS)}
    for node in reversed(order):
        entry = parent[node]
        if entry is not None:
            subtree[entry[0]] += subtree[node]
    current = [ZERO] * SEAMS
    for node in order[1:]:
        entry = parent[node]
        assert entry is not None
        _, seam = entry
        if SEAM_RIGHT[seam] == node:
            current[seam] = subtree[node]
        else:
            current[seam] = -subtree[node]
    _require(boundary(current) == list(load), "tree solution drift")
    return current


def fundamental_cycle(chord: int) -> Vector:
    """The fundamental cycle of committed chord ``chord``: value ``+1`` on
    the chord, zero on every other chord, zero boundary."""
    _require(chord in CHORD_SEAMS, "not a committed chord seam")
    indicator = [ZERO] * SEAMS
    indicator[chord] = ONE
    closing = tree_solution(boundary(indicator))
    cycle = [a - b for a, b in zip(indicator, closing, strict=True)]
    _require(all(x == 0 for x in boundary(cycle)), "fundamental cycle drift")
    return cycle


def fundamental_cycles() -> dict[int, Vector]:
    """All nineteen fundamental chord cycles keyed by chord seam index."""
    return {chord: fundamental_cycle(chord) for chord in CHORD_SEAMS}


# ---------------------------------------------------------------------------
# Orientation-erasure control
# ---------------------------------------------------------------------------

def unsigned_face_boundary_control() -> Fraction:
    """Boundary-of-boundary sum on face zero at port zero with the signed
    face incidence replaced by its absolute value.  The committed control
    value is ``-2``; the signed value is ``0``."""
    return sum(
        (abs(face_incidence_entry(0, e)) * incidence_entry(e, 0)
         for e in range(SEAMS)),
        start=ZERO,
    )


# ---------------------------------------------------------------------------
# Structural receipt (fail-closed)
# ---------------------------------------------------------------------------

def base_carrier_receipt(face_matrix: Matrix | None = None) -> dict:
    """Recompute and cross-check every structural fact of the committed base
    carrier; raise ``BaseCarrierError`` on the first failure.

    ``face_matrix`` defaults to the committed oriented incidence; a mutated
    matrix (sign flip, orientation erasure) fails the receipt.
    """
    _require(len(SEAM_LEFT) == SEAMS and len(SEAM_RIGHT) == SEAMS,
             "seam table arity drift")
    _require(
        all(0 <= SEAM_LEFT[e] < SEAM_RIGHT[e] < PORTS for e in range(SEAMS)),
        "seam orientation drift",
    )
    _require(
        len({(SEAM_LEFT[e], SEAM_RIGHT[e]) for e in range(SEAMS)}) == SEAMS,
        "duplicate seam rows",
    )
    lap = laplacian_matrix()
    degrees = [int(lap[p][p]) for p in range(PORTS)]
    _require(all(deg == 5 for deg in degrees), "degree-five regularity drift")

    d = coboundary_matrix()
    bnd = boundary_matrix()
    _require(
        all(
            lap[i][j] == sum(
                (bnd[i][e] * d[e][j] for e in range(SEAMS)), start=ZERO
            )
            for i in range(PORTS)
            for j in range(PORTS)
        ),
        "laplacian factorization drift",
    )

    rank_d = matrix_rank(d)
    _require(rank_d == PORTS - 1, "coboundary rank drift")
    _require(matrix_rank(lap) == PORTS - 1, "laplacian rank drift")
    _require(
        all(x == 0 for x in coboundary([ONE] * PORTS)),
        "constant kernel of d drift",
    )
    _require(
        all(x == 0 for x in laplacian_apply([ONE] * PORTS)),
        "constant kernel of L drift",
    )

    c = face_incidence_matrix() if face_matrix is None else face_matrix
    _require(
        len(c) == FACES and all(len(row) == SEAMS for row in c),
        "face incidence arity drift",
    )
    rank_c = matrix_rank(c)
    _require(rank_c == 19, "face incidence rank drift")

    composition_zero = all(
        sum((c[f][e] * d[e][p] for e in range(SEAMS)), start=ZERO) == 0
        for f in range(FACES)
        for p in range(PORTS)
    )
    _require(composition_zero, "C compose d nonzero: orientation drift")

    # ker C = im d: C . d = 0 places im d (dimension 11) inside ker C, and
    # ker C has dimension 30 - rank C = 11, forcing equality.
    ker_c_dim = SEAMS - rank_c
    im_d_dim = rank_d
    _require(ker_c_dim == 11 and im_d_dim == 11,
             "kernel/image dimension drift")

    per_face_support = [
        sum(1 for e in range(SEAMS) if c[f][e] != 0) for f in range(FACES)
    ]
    _require(all(s == 3 for s in per_face_support), "face support drift")
    _require(sum(per_face_support) == 60, "face incidence total support drift")
    for e in range(SEAMS):
        touching = [f for f in range(FACES) if c[f][e] != 0]
        _require(len(touching) == 2, "seam-face support drift")
        _require(
            sum((c[f][e] for f in touching), start=ZERO) == 0,
            "seam-face opposite orientation drift",
        )

    kinetic = local_kinetic_matrix() if face_matrix is None else [
        [
            sum((c[f][e] * c[f][g] for f in range(FACES)), start=ZERO)
            for g in range(SEAMS)
        ]
        for e in range(SEAMS)
    ]
    _require(
        all(kinetic[e][e] == 2 for e in range(SEAMS)),
        "local kinetic diagonal drift",
    )
    per_row = [
        sum(1 for g in range(SEAMS) if kinetic[e][g] != 0)
        for e in range(SEAMS)
    ]
    _require(all(r == 5 for r in per_row), "local kinetic row support drift")
    _require(sum(per_row) == 150, "local kinetic total support drift")
    _require(matrix_rank(kinetic) == 19, "local kinetic rank drift")

    _require(len(TREE_SEAMS) == PORTS - 1, "tree seam count drift")
    _tree_parents()
    _require(len(CHORD_SEAMS) == 19, "chord count drift")

    control = unsigned_face_boundary_control()
    _require(
        control == UNSIGNED_FACE_ZERO_PORT_ZERO_CONTROL,
        "orientation-erasure control value drift",
    )

    return {
        "schema": "oph.sim.em_base_carrier.v1",
        "design_only": True,
        "frozen": False,
        "instrument_armed": False,
        "ports": PORTS,
        "seams": SEAMS,
        "faces": FACES,
        "degree_sequence": degrees,
        "rank_d": rank_d,
        "rank_C": rank_c,
        "rank_laplacian": PORTS - 1,
        "C_compose_d_zero": True,
        "kernel_C_dimension": ker_c_dim,
        "kernel_C_equals_image_d": True,
        "kernel_d_dimension": 1,
        "kernel_L": "constants",
        "local_kinetic": {
            "diagonal": 2,
            "nonzero_per_row": 5,
            "total_support": 150,
            "rank": 19,
        },
        "face_support": {"per_face": 3, "total": 60,
                         "faces_per_seam": 2, "opposite_signs": True},
        "spanning_tree": {
            "tree_seams": list(TREE_SEAMS),
            "chords": list(CHORD_SEAMS),
            "chord_count": len(CHORD_SEAMS),
        },
        "orientation_control": {
            "signed_face0_port0": 0,
            "unsigned_face0_port0": int(control),
        },
        "statement": (
            "finite exact mathematics on the committed base carrier; no "
            "physical attachment (PR-53, PR-54 open); not an instrument"
        ),
    }
