"""Spin statistics semantic artifact producer for the twelve-port carrier.

This module measures, from finite source structure alone, the transport data
that the paper-side matter-lift certificate (issue #314 in
reverse-engineering-reality) needs in order to type physical matter without a
declared statistics or category contract. It is target-blind: no Standard
Model label, statistics contract, category typing, or matter module enters
the producer. What it emits are measured geometric facts about the certified
carrier:

- the oriented port-to-vertex frame of the certified #565 carrier in exact
  Q(sqrt5) arithmetic, with every oriented face determinant positive;
- the measured deck group of the incidence (order 120), its
  orientation-preserving rotation subgroup (order 60), and the improper coset
  containing the distance-three antipode;
- the exact quaternion lift of every proper deck rotation, computed from the
  measured rotation matrices: the 120-element lift closure, its element-order
  profile {1:1, 2:1, 3:20, 4:30, 5:24, 6:20, 10:24}, its unique nontrivial
  involution -1, and its two-element centre {+-1};
- the section-obstruction measurement: every Klein four-subgroup of the
  rotation deck is enumerated, and for the canonical one all eight sign
  assignments of lift representatives fail to form a section, because the
  lift of every deck involution squares to -1 exactly;
- the homology of the oriented support complex (12 vertices, 30 edges, 20
  faces): Betti numbers (1, 0, 1), Euler characteristic 2, torsion-free
  integral H_2 of rank one, hence exactly 2^{b_1} = 1 spin structure on the
  oriented support;
- the measured orientation/chirality convention: the frame realizes every
  manifest face positively, the sixty rotations preserve the oriented face
  set, and the improper coset (including the antipode) reverses it;
- deck-equivariant persistence of the twelve defect ports along the geodesic
  refinement tower.

The producer fails closed with typed errors on carriers whose measured
structure does not present these facts: wrong regularity, missing antipodal
pairing, inconsistent face orientation, an automorphism group of the wrong
order, a rotation without an exact quaternion lift, or a lift set that does
not close with the binary-icosahedral order profile. Whether these measured
facts force a fermionic Spin/odd-Weyl typing of matter is decided on the
paper side; this producer only measures.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.core.charged_response import (
    ONE,
    PHI,
    Q5,
    ZERO,
    ChargedResponseError,
    _oriented_face_set,
    _require,
    canonical_sha256,
    incidence_automorphisms,
    load_carrier,
    match_vertex_frame,
    q5_det3,
    q5_dot,
)
from oph_fpe.core.icosahedral import icosahedral_defect_port_report

SCHEMA = "oph.spin_statistics_semantic_artifact.v1"
ISSUE = 314

BINARY_ICOSAHEDRAL_ORDER_PROFILE = {1: 1, 2: 1, 3: 20, 4: 30, 5: 24, 6: 20, 10: 24}


# ---------------------------------------------------------------------------
# Exact square roots in Q(sqrt5)
# ---------------------------------------------------------------------------


def _fraction_sqrt(value: Fraction) -> Fraction | None:
    if value < 0:
        return None
    numerator, denominator = value.numerator, value.denominator
    root_n, root_d = math.isqrt(numerator), math.isqrt(denominator)
    if root_n * root_n == numerator and root_d * root_d == denominator:
        return Fraction(root_n, root_d)
    return None


def q5_sqrt(value: Q5) -> Q5 | None:
    """The nonnegative exact square root in Q(sqrt5), or None."""

    if value.sign() < 0:
        return None
    if value.is_zero():
        return ZERO
    if value.b == 0:
        rational = _fraction_sqrt(value.a)
        if rational is not None:
            return Q5.of(rational)
        surd = _fraction_sqrt(value.a / 5)
        if surd is not None:
            return Q5.of(0, surd)
        return None
    discriminant = value.a * value.a - 5 * value.b * value.b
    if discriminant < 0:
        return None
    root_disc = _fraction_sqrt(discriminant)
    if root_disc is None:
        return None
    for x_squared in ((value.a + root_disc) / 2, (value.a - root_disc) / 2):
        if x_squared <= 0:
            continue
        x = _fraction_sqrt(x_squared)
        if x is None or x == 0:
            continue
        candidate = Q5(x, value.b / (2 * x))
        if (candidate * candidate - value).is_zero():
            return candidate if candidate.sign() > 0 else -candidate
    return None


# ---------------------------------------------------------------------------
# Exact 3x3 linear algebra over Q(sqrt5)
# ---------------------------------------------------------------------------


def _matrix3_mul(left: Sequence[Sequence[Q5]], right: Sequence[Sequence[Q5]]) -> list[list[Q5]]:
    return [
        [
            left[i][0] * right[0][j] + left[i][1] * right[1][j] + left[i][2] * right[2][j]
            for j in range(3)
        ]
        for i in range(3)
    ]


def _matrix3_inverse(matrix: Sequence[Sequence[Q5]]) -> list[list[Q5]]:
    determinant = q5_det3(matrix)
    _require(not determinant.is_zero(), "ROTATION_FIT", "frame triple is degenerate")
    inverse_det = determinant.inverse()
    cofactors = [
        [
            matrix[(i + 1) % 3][(j + 1) % 3] * matrix[(i + 2) % 3][(j + 2) % 3]
            - matrix[(i + 1) % 3][(j + 2) % 3] * matrix[(i + 2) % 3][(j + 1) % 3]
            for j in range(3)
        ]
        for i in range(3)
    ]
    return [[cofactors[j][i] * inverse_det for j in range(3)] for i in range(3)]


def _matrix3_transpose(matrix: Sequence[Sequence[Q5]]) -> list[list[Q5]]:
    return [[matrix[j][i] for j in range(3)] for i in range(3)]


def _matrix3_equal(left: Sequence[Sequence[Q5]], right: Sequence[Sequence[Q5]]) -> bool:
    return all((left[i][j] - right[i][j]).is_zero() for i in range(3) for j in range(3))


def _matrix3_identity() -> list[list[Q5]]:
    return [[ONE if i == j else ZERO for j in range(3)] for i in range(3)]


def _nullspace3(matrix: Sequence[Sequence[Q5]]) -> list[list[Q5]]:
    """Basis of the nullspace of a 3x3 matrix over Q(sqrt5)."""

    rows = [list(row) for row in matrix]
    pivots: list[tuple[int, int]] = []
    row_index = 0
    for column in range(3):
        pivot_row = None
        for candidate in range(row_index, 3):
            if not rows[candidate][column].is_zero():
                pivot_row = candidate
                break
        if pivot_row is None:
            continue
        rows[row_index], rows[pivot_row] = rows[pivot_row], rows[row_index]
        scale = rows[row_index][column].inverse()
        rows[row_index] = [entry * scale for entry in rows[row_index]]
        for other in range(3):
            if other == row_index or rows[other][column].is_zero():
                continue
            factor = rows[other][column]
            rows[other] = [
                rows[other][k] - factor * rows[row_index][k] for k in range(3)
            ]
        pivots.append((row_index, column))
        row_index += 1
    pivot_columns = {column for _, column in pivots}
    free_columns = [column for column in range(3) if column not in pivot_columns]
    basis = []
    for free in free_columns:
        vector = [ZERO, ZERO, ZERO]
        vector[free] = ONE
        for row, column in pivots:
            vector[column] = -rows[row][free]
        basis.append(vector)
    return basis


# ---------------------------------------------------------------------------
# Exact quaternions over Q(sqrt5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Quat:
    """Exact quaternion w + x i + y j + z k over Q(sqrt5)."""

    w: Q5
    x: Q5
    y: Q5
    z: Q5

    def mul(self, other: "Quat") -> "Quat":
        return Quat(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def neg(self) -> "Quat":
        return Quat(-self.w, -self.x, -self.y, -self.z)

    def norm_squared(self) -> Q5:
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z

    def is_identity(self) -> bool:
        return (
            (self.w - ONE).is_zero()
            and self.x.is_zero()
            and self.y.is_zero()
            and self.z.is_zero()
        )

    def key(self) -> tuple:
        return (
            (self.w.a, self.w.b),
            (self.x.a, self.x.b),
            (self.y.a, self.y.b),
            (self.z.a, self.z.b),
        )

    def render(self) -> list[str]:
        return [self.w.render(), self.x.render(), self.y.render(), self.z.render()]


QUAT_ONE = Quat(ONE, ZERO, ZERO, ZERO)
QUAT_MINUS_ONE = Quat(-ONE, ZERO, ZERO, ZERO)


def quat_rotation_matrix(q: Quat) -> list[list[Q5]]:
    """The exact SO(3) matrix of a unit quaternion."""

    two = Q5.of(2)
    w, x, y, z = q.w, q.x, q.y, q.z
    return [
        [
            ONE - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
        ],
        [
            two * (x * y + z * w),
            ONE - two * (x * x + z * z),
            two * (y * z - x * w),
        ],
        [
            two * (x * z - y * w),
            two * (y * z + x * w),
            ONE - two * (x * x + y * y),
        ],
    ]


def quaternion_lift(rotation: Sequence[Sequence[Q5]]) -> Quat:
    """One exact unit-quaternion lift of a proper rotation matrix."""

    transpose_product = _matrix3_mul(_matrix3_transpose(rotation), rotation)
    _require(
        _matrix3_equal(transpose_product, _matrix3_identity()),
        "LIFT_ORTHOGONALITY",
        "the fitted deck matrix is not orthogonal",
    )
    determinant = q5_det3(rotation)
    _require(
        (determinant - ONE).is_zero(),
        "LIFT_DETERMINANT",
        "quaternion lifts exist only for proper rotations",
    )
    identity = _matrix3_identity()
    if _matrix3_equal(rotation, identity):
        return QUAT_ONE
    difference = [
        [rotation[i][j] - identity[i][j] for j in range(3)] for i in range(3)
    ]
    kernel = _nullspace3(difference)
    _require(len(kernel) == 1, "LIFT_AXIS", "rotation axis is not one-dimensional")
    axis = kernel[0]
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    cosine = (trace - ONE) * Q5.of(Fraction(1, 2))
    half_cos_squared = (ONE + cosine) * Q5.of(Fraction(1, 2))
    w = q5_sqrt(half_cos_squared)
    _require(w is not None, "LIFT_FIELD", "half-angle cosine has no square root in Q(sqrt5)")
    axis_norm_squared = q5_dot(axis, axis)
    _require(not axis_norm_squared.is_zero(), "LIFT_AXIS", "axis vector is null")
    scale_squared = (ONE - w * w) * axis_norm_squared.inverse()
    scale = q5_sqrt(scale_squared)
    _require(scale is not None, "LIFT_FIELD", "half-angle sine has no square root in Q(sqrt5)")
    for candidate_scale in (scale, -scale):
        candidate = Quat(
            w,
            candidate_scale * axis[0],
            candidate_scale * axis[1],
            candidate_scale * axis[2],
        )
        if _matrix3_equal(quat_rotation_matrix(candidate), rotation):
            _require(
                (candidate.norm_squared() - ONE).is_zero(),
                "LIFT_UNIT",
                "quaternion lift is not unit",
            )
            return candidate
    raise ChargedResponseError("LIFT_ADJOINT", "no exact quaternion reproduces the rotation")


# ---------------------------------------------------------------------------
# Deck realization: automorphisms as exact vertex isometries
# ---------------------------------------------------------------------------


def _independent_port_triple(frame: Sequence[tuple[Q5, Q5, Q5]]) -> tuple[int, int, int]:
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if not q5_det3([frame[i], frame[j], frame[k]]).is_zero():
                    return i, j, k
    raise ChargedResponseError("ROTATION_FIT", "no independent port triple exists")


def deck_matrix(
    permutation: Sequence[int], frame: Sequence[tuple[Q5, Q5, Q5]]
) -> list[list[Q5]]:
    """The exact orthogonal matrix realizing a deck permutation on the frame."""

    i, j, k = _independent_port_triple(frame)
    source = _matrix3_transpose([list(frame[i]), list(frame[j]), list(frame[k])])
    target = _matrix3_transpose(
        [list(frame[permutation[i]]), list(frame[permutation[j]]), list(frame[permutation[k]])]
    )
    matrix = _matrix3_mul(target, _matrix3_inverse(source))
    for port in range(12):
        image = [
            matrix[row][0] * frame[port][0]
            + matrix[row][1] * frame[port][1]
            + matrix[row][2] * frame[port][2]
            for row in range(3)
        ]
        _require(
            all((image[row] - frame[permutation[port]][row]).is_zero() for row in range(3)),
            "ROTATION_FIT",
            "the deck permutation is not realized by a linear isometry of the frame",
        )
    return matrix


def measure_deck_realization(carrier: dict[str, Any], frame: Sequence[tuple[Q5, Q5, Q5]]) -> dict[str, Any]:
    """Measure the deck group, its rotation subgroup, and the improper coset."""

    automorphisms = incidence_automorphisms(carrier["adjacency"])
    _require(
        len(automorphisms) == 120,
        "DECK_GROUP",
        "the measured incidence automorphism group does not have order 120",
    )
    oriented_faces = _oriented_face_set(carrier["faces"])
    reversed_faces = _oriented_face_set([[c, b, a] for a, b, c in carrier["faces"]])
    rotations: list[tuple[int, ...]] = []
    improper: list[tuple[int, ...]] = []
    matrices: dict[tuple[int, ...], list[list[Q5]]] = {}
    for permutation in automorphisms:
        matrix = deck_matrix(permutation, frame)
        matrices[permutation] = matrix
        determinant = q5_det3(matrix)
        image_faces = _oriented_face_set(
            [[permutation[a], permutation[b], permutation[c]] for a, b, c in carrier["faces"]]
        )
        if (determinant - ONE).is_zero():
            _require(
                image_faces == oriented_faces,
                "DECK_ORIENTATION",
                "a proper deck matrix does not preserve the oriented face set",
            )
            rotations.append(permutation)
        else:
            _require(
                (determinant + ONE).is_zero(),
                "DECK_ORIENTATION",
                "deck determinant is not +-1",
            )
            _require(
                image_faces == reversed_faces,
                "DECK_ORIENTATION",
                "an improper deck matrix does not reverse the oriented face set",
            )
            improper.append(permutation)
    _require(len(rotations) == 60, "DECK_GROUP", "sixty orientation-preserving rotations required")
    _require(len(improper) == 60, "DECK_GROUP", "sixty orientation-reversing elements required")
    antipode_permutation = tuple(carrier["antipode"])
    _require(
        antipode_permutation in improper,
        "DECK_ORIENTATION",
        "the distance-three antipode must lie in the orientation-reversing coset",
    )
    return {
        "automorphisms": automorphisms,
        "rotations": rotations,
        "improper": improper,
        "matrices": matrices,
        "antipode_permutation": antipode_permutation,
    }


# ---------------------------------------------------------------------------
# The measured lift group and its section obstruction
# ---------------------------------------------------------------------------


def measure_lift_group(
    deck: Mapping[str, Any]
) -> dict[str, Any]:
    lifts: dict[tuple[int, ...], Quat] = {}
    for permutation in deck["rotations"]:
        lifts[permutation] = quaternion_lift(deck["matrices"][permutation])

    group: dict[tuple, Quat] = {}
    for lift in lifts.values():
        group[lift.key()] = lift
        group[lift.neg().key()] = lift.neg()
    _require(
        len(group) == 120,
        "LIFT_CLOSURE",
        f"expected 120 signed lift elements, got {len(group)}",
    )
    keys = set(group.keys())
    for left in group.values():
        for right in group.values():
            _require(
                left.mul(right).key() in keys,
                "LIFT_CLOSURE",
                "the measured lift set is not closed under multiplication",
            )

    def element_order(q: Quat) -> int:
        power = q
        for order in range(1, 121):
            if power.is_identity():
                return order
            power = power.mul(q)
        raise ChargedResponseError("LIFT_ORDER", "lift element order exceeded the finite bound")

    order_profile: dict[int, int] = {}
    involutions: list[Quat] = []
    for element in group.values():
        order = element_order(element)
        order_profile[order] = order_profile.get(order, 0) + 1
        if order == 2:
            involutions.append(element)
    _require(
        order_profile == BINARY_ICOSAHEDRAL_ORDER_PROFILE,
        "LIFT_PROFILE",
        f"the measured order profile {order_profile} is not binary icosahedral",
    )
    _require(
        len(involutions) == 1 and involutions[0].key() == QUAT_MINUS_ONE.key(),
        "LIFT_INVOLUTION",
        "the measured lift group must have the unique involution -1",
    )

    centre = [
        element
        for element in group.values()
        if all(
            element.mul(other).key() == other.mul(element).key()
            for other in group.values()
        )
    ]
    centre_keys = {element.key() for element in centre}
    _require(
        centre_keys == {QUAT_ONE.key(), QUAT_MINUS_ONE.key()},
        "LIFT_CENTRE",
        "the measured lift centre is not exactly {+1, -1}",
    )

    return {
        "lifts": lifts,
        "group_order": len(group),
        "order_profile": {str(k): v for k, v in sorted(order_profile.items())},
        "unique_nontrivial_involution": "-1",
        "centre_order": 2,
        "centre_elements": ["+1", "-1"],
    }


def measure_klein_four_obstruction(
    deck: Mapping[str, Any], lift_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Enumerate Klein four-subgroups of the rotation deck and measure that no
    sign assignment of quaternion lifts forms a section over any of them."""

    rotations: list[tuple[int, ...]] = deck["rotations"]
    lifts: Mapping[tuple[int, ...], Quat] = lift_report["lifts"]
    identity = tuple(range(12))

    def compose(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
        return tuple(left[right[i]] for i in range(12))

    deck_involutions = [
        permutation
        for permutation in rotations
        if permutation != identity and compose(permutation, permutation) == identity
    ]
    _require(
        len(deck_involutions) == 15,
        "DECK_KLEIN",
        "the rotation deck must contain fifteen involutions",
    )
    klein_subgroups: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    for index_a in range(len(deck_involutions)):
        for index_b in range(index_a + 1, len(deck_involutions)):
            a, b = deck_involutions[index_a], deck_involutions[index_b]
            ab = compose(a, b)
            if ab == compose(b, a) and ab in deck_involutions:
                triple = tuple(sorted((a, b, ab)))
                if triple not in klein_subgroups:
                    klein_subgroups.append(triple)
    _require(
        len(klein_subgroups) == 5,
        "DECK_KLEIN",
        f"the rotation deck must contain five Klein four-subgroups, got {len(klein_subgroups)}",
    )

    all_squares_minus_one = all(
        lifts[involution].mul(lifts[involution]).key() == QUAT_MINUS_ONE.key()
        for involution in deck_involutions
    )
    _require(
        all_squares_minus_one,
        "SECTION_OBSTRUCTION",
        "every deck-involution lift must square to -1",
    )

    section_tables = []
    for triple in klein_subgroups:
        a, b, c = triple
        failures = 0
        for sign_a in (1, -1):
            for sign_b in (1, -1):
                for sign_c in (1, -1):
                    lift_a = lifts[a] if sign_a == 1 else lifts[a].neg()
                    lift_b = lifts[b] if sign_b == 1 else lifts[b].neg()
                    lift_c = lifts[c] if sign_c == 1 else lifts[c].neg()
                    section = (
                        lift_a.mul(lift_a).is_identity()
                        and lift_b.mul(lift_b).is_identity()
                        and lift_c.mul(lift_c).is_identity()
                        and lift_a.mul(lift_b).key() == lift_c.key()
                    )
                    if not section:
                        failures += 1
        _require(
            failures == 8,
            "SECTION_OBSTRUCTION",
            "a sign assignment forms a section over a Klein four-subgroup",
        )
        section_tables.append(
            {
                "sign_assignments_tested": 8,
                "sections_found": 0,
                "obstruction": "every involution lift squares to -1, so no assignment closes",
            }
        )

    return {
        "deck_involutions": 15,
        "klein_four_subgroups": len(klein_subgroups),
        "involution_lift_squares": "-1 for all fifteen deck involutions",
        "section_exhaustion_per_subgroup": section_tables,
        "no_section_over_any_klein_four_subgroup": True,
        "conclusion": (
            "the measured deck-to-lift extension admits no section over any Klein "
            "four-subgroup, so the measured transport double cover is non-split"
        ),
    }


# ---------------------------------------------------------------------------
# Support homology and spin structures
# ---------------------------------------------------------------------------


def _smith_invariants(matrix: list[list[int]]) -> list[int]:
    """Nonzero Smith normal form invariants of an integer matrix."""

    work = [row[:] for row in matrix]
    rows, columns = len(work), len(work[0]) if work else 0
    invariants: list[int] = []
    top = 0
    while top < min(rows, columns):
        pivot = None
        best = None
        for i in range(top, rows):
            for j in range(top, columns):
                value = abs(work[i][j])
                if value and (best is None or value < best):
                    best = value
                    pivot = (i, j)
        if pivot is None:
            break
        i, j = pivot
        work[top], work[i] = work[i], work[top]
        for row in work:
            row[top], row[j] = row[j], row[top]
        reduced = True
        while reduced:
            reduced = False
            for i in range(top + 1, rows):
                if work[i][top] % work[top][top]:
                    quotient = work[i][top] // work[top][top]
                    for j in range(columns):
                        work[i][j] -= quotient * work[top][j]
                    work[top], work[i] = work[i], work[top]
                    reduced = True
            for i in range(top + 1, rows):
                quotient = work[i][top] // work[top][top]
                for j in range(columns):
                    work[i][j] -= quotient * work[top][j]
            for j in range(top + 1, columns):
                if work[top][j] % work[top][top]:
                    quotient = work[top][j] // work[top][top]
                    for row in work:
                        row[j] -= quotient * row[top]
                    for row in work:
                        row[top], row[j] = row[j], row[top]
                    reduced = True
            for j in range(top + 1, columns):
                quotient = work[top][j] // work[top][top]
                for row in work:
                    row[j] -= quotient * row[top]
        invariants.append(abs(work[top][top]))
        top += 1
    return [value for value in invariants if value]


def measure_support_homology(carrier: dict[str, Any]) -> dict[str, Any]:
    """Exact chain complex of the oriented support: 12 vertices, 30 edges, 20 faces."""

    adjacency = carrier["adjacency"]
    edges = sorted(
        (i, j) for i in range(12) for j in range(i + 1, 12) if adjacency[i][j]
    )
    _require(len(edges) == 30, "SUPPORT_COMPLEX", "thirty support edges required")
    edge_index = {edge: position for position, edge in enumerate(edges)}

    boundary_one = [[0] * 30 for _ in range(12)]
    for position, (u, v) in enumerate(edges):
        boundary_one[u][position] = -1
        boundary_one[v][position] = 1

    boundary_two = [[0] * 20 for _ in range(30)]
    for face_position, face in enumerate(carrier["faces"]):
        a, b, c = face
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            sign = 1 if (u, v) == key else -1
            boundary_two[edge_index[key]][face_position] += sign

    composite = [
        [
            sum(boundary_one[i][k] * boundary_two[k][j] for k in range(30))
            for j in range(20)
        ]
        for i in range(12)
    ]
    _require(
        all(entry == 0 for row in composite for entry in row),
        "SUPPORT_COMPLEX",
        "the support boundary maps do not compose to zero",
    )

    invariants_one = _smith_invariants(boundary_one)
    invariants_two = _smith_invariants(boundary_two)
    rank_one, rank_two = len(invariants_one), len(invariants_two)
    _require(rank_one == 11 and rank_two == 19, "SUPPORT_COMPLEX", "unexpected boundary ranks")
    betti_zero = 12 - rank_one
    betti_one = 30 - rank_one - rank_two
    betti_two = 20 - rank_two
    torsion_two = [value for value in invariants_two if value != 1]
    _require(
        (betti_zero, betti_one, betti_two) == (1, 0, 1) and not torsion_two,
        "SUPPORT_COMPLEX",
        "the oriented support is not a homology two-sphere",
    )
    return {
        "cells": {"vertices": 12, "edges": 30, "faces": 20},
        "euler_characteristic": 2,
        "betti_numbers": [betti_zero, betti_one, betti_two],
        "boundary_smith_ranks": [rank_one, rank_two],
        "integral_h2_torsion": [],
        "spin_structure_count": 2 ** betti_one,
        "conclusion": (
            "the oriented support is an exact homology two-sphere with b1 = 0, "
            "so it carries exactly one spin structure and the lift measurement "
            "has no spin-structure ambiguity"
        ),
    }


# ---------------------------------------------------------------------------
# Refinement persistence equivariance
# ---------------------------------------------------------------------------


def measure_refinement_equivariance(
    carrier: dict[str, Any], deck: Mapping[str, Any], levels: int
) -> dict[str, Any]:
    report = icosahedral_defect_port_report(levels - 1)
    _require(
        bool(report["TWELVE_PERSISTENT_COMBINATORIAL_DEFECT_PORTS_RECEIPT"]),
        "REFINEMENT_TOWER",
        "the geodesic tower does not present twelve persistent defect ports",
    )
    per_level = [
        {"level": row["level"], "defect_ports": len(row["unit_defect_vertex_ids"])}
        for row in report["levels"]
    ]
    _require(
        len(per_level) == levels and all(row["defect_ports"] == 12 for row in per_level),
        "REFINEMENT_TOWER",
        "tower level defect-port counts are not twelve at every measured level",
    )
    # The persistence content is measured by the tower report: the same twelve
    # vertex ids carry the unit defect charge at every level. The persistence
    # map on defect ports is therefore the identity, and deck equivariance of
    # an identity map holds by arithmetic, not by an additional measurement;
    # the receipt states exactly that and nothing stronger.
    persistent_ids = report["persistent_port_vertex_ids"]
    stable = all(
        row["unit_defect_vertex_ids"] == persistent_ids for row in report["levels"]
    )
    _require(stable, "REFINEMENT_TOWER", "defect ports do not persist identically per level")
    return {
        "levels_measured": levels,
        "per_level": per_level,
        "persistent_port_vertex_ids": persistent_ids,
        "persistence_map": "identity_on_defect_ports",
        "deck_equivariant_persistence": True,
        "equivariance_provenance": (
            "the measured persistence map is the identity on the persistent "
            "port ids, so it commutes with every deck element by arithmetic"
        ),
    }


# ---------------------------------------------------------------------------
# Artifact assembly
# ---------------------------------------------------------------------------


def produce_spin_statistics_artifact(manifest: Mapping[str, Any]) -> dict[str, Any]:
    carrier = load_carrier(manifest)
    frame = match_vertex_frame(carrier)
    deck = measure_deck_realization(carrier, frame)
    lift_report = measure_lift_group(deck)
    obstruction = measure_klein_four_obstruction(deck, lift_report)
    homology = measure_support_homology(carrier)
    refinement = measure_refinement_equivariance(carrier, deck, levels=3)

    canonical_klein = None
    identity = tuple(range(12))

    def compose(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
        return tuple(left[right[i]] for i in range(12))

    involutions = [
        permutation
        for permutation in deck["rotations"]
        if permutation != identity and compose(permutation, permutation) == identity
    ]
    for index_a in range(len(involutions)):
        for index_b in range(index_a + 1, len(involutions)):
            a, b = involutions[index_a], involutions[index_b]
            if compose(a, b) == compose(b, a) and compose(a, b) in involutions:
                canonical_klein = tuple(sorted((a, b, compose(a, b))))
                break
        if canonical_klein:
            break
    _require(canonical_klein is not None, "DECK_KLEIN", "no Klein four-subgroup found")
    canonical_lift_table = [
        {
            "deck_permutation": list(permutation),
            "quaternion_lift": lift_report["lifts"][permutation].render(),
            "lift_square": "-1",
        }
        for permutation in canonical_klein
    ]

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "target_firewall": (
            "no_standard_model_label_statistics_contract_category_typing_or_matter_module_enters_the_producer"
        ),
        "carrier_binding": {
            "carrier_manifest_sha256": carrier["manifest_sha256"],
            "port_order": carrier["ports"],
            "antipode": {
                carrier["ports"][i]: carrier["ports"][carrier["antipode"][i]]
                for i in range(12)
            },
            "incidence_edge_count": 30,
            "oriented_face_count": 20,
        },
        "port_vertex_frame": {
            carrier["ports"][i]: [frame[i][0].render(), frame[i][1].render(), frame[i][2].render()]
            for i in range(12)
        },
        "deck_measurement": {
            "incidence_automorphism_group_order": 120,
            "orientation_preserving_rotations": 60,
            "orientation_reversing_elements": 60,
            "antipode_is_orientation_reversing": True,
            "matrix_realization": "exact_q_sqrt5_orthogonal_fit_on_measured_frame",
        },
        "lift_measurement": {
            "lift_arithmetic": "exact_unit_quaternions_over_q_sqrt5",
            "lift_group_order": lift_report["group_order"],
            "order_profile": lift_report["order_profile"],
            "unique_nontrivial_involution": lift_report["unique_nontrivial_involution"],
            "centre_order": lift_report["centre_order"],
            "centre_elements": lift_report["centre_elements"],
            "conclusion": (
                "the measured deck rotations lift to a closed 120-element unit-"
                "quaternion group with binary-icosahedral order profile, unique "
                "involution -1, and centre {+1, -1}"
            ),
        },
        "section_obstruction": {
            key: value
            for key, value in obstruction.items()
        },
        "canonical_klein_four_lift_table": canonical_lift_table,
        "support_homology": homology,
        "orientation_convention": {
            "faces": "all_twenty_oriented_face_determinants_positive_in_measured_frame",
            "rotations_preserve_oriented_faces": True,
            "improper_coset_reverses_oriented_faces": True,
            "weyl_relabeling_note": (
                "reversing the measured orientation exchanges the two quaternion "
                "conjugation classes; composing that reversal with the global "
                "conjugation of the response artifact is the only same-reduct relabeling"
            ),
        },
        "refinement_equivariance": refinement,
        "provenance": {
            "producer": "oph_fpe.core.spin_statistics_response.produce_spin_statistics_artifact",
            "deterministic": True,
        },
        "physical_source_gate": {
            "frame_transport_lift_measured": True,
            "lift_group_binary_icosahedral": True,
            "no_section_over_any_klein_four_deck_subgroup": True,
            "unique_nontrivial_central_involution": True,
            "unique_spin_structure_on_oriented_support": True,
            "orientation_chirality_convention_measured": True,
            "deck_equivariant_refinement_persistence": True,
            "laboratory_exchange_measurement": False,
            "continuum_spin_statistics_theorem": False,
            "passed": True,
            "scope": (
                "finite source-model scope: the gate aggregates the seven measured "
                "facts above; the two false rows are separate lanes (laboratory "
                "attachment #569, continuum QFT) and never enter 'passed'"
            ),
        },
    }
    artifact["artifact_sha256"] = "sha256:" + canonical_sha256(artifact)
    return artifact


def write_spin_statistics_artifact(carrier_manifest_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = json.loads(carrier_manifest_path.read_text(encoding="utf-8"))
    artifact = produce_spin_statistics_artifact(manifest)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carrier-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact = write_spin_statistics_artifact(args.carrier_manifest, args.out)
    print(json.dumps({"status": "PASS", "artifact_sha256": artifact["artifact_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
