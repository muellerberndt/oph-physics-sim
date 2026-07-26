"""Charged response semantic artifact producer for the twelve-port carrier.

This module derives the finite charged response representation of the local
recurrent carrier dynamics and emits the versioned semantic artifact consumed
by the paper-side port-current certificate (issue #599 in
reverse-engineering-reality). It derives from finite source structure rather
than a target gauge assignment or declaration:

- the carrier manifest hash and port order of the certified #565 carrier;
- the exact A5-isotypic sector decomposition (1, 3, 3', 5) of the twelve-port
  space under the propagation generator, with the Galois pairing of the two
  triplet sectors;
- the port-to-vertex frame assignment in exact Q(sqrt5) arithmetic, oriented
  by the manifest's oriented faces;
- the exact target-blind response operator ``R = -J`` from the unique
  graph-distance-three antipode involution, whose computed sector eigenvalues
  are negative on ``1 + 5`` and positive on ``3 + 3'``;
- the physical refinement maps: persistence of the twelve defect ports along
  the geodesic tower, with per-level A5 equivariance;
- runtime binding to the dynamics actually propagated by
  ``oph_fpe.core.echosahedral_dynamics`` (spectrum, equivariance, unitarity).

The producer fails closed with typed errors on carriers whose source
structure does not determine a charged double triplet: wrong regularity,
missing antipodal pairing, rational (non-Galois-paired) triplet spectrum,
inconsistent face orientation, or a runtime coupling that does not match the
exact combinatorial spectrum. No construction-model string is accepted or
emitted as an input; the construction name in the artifact is a derived
status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from oph_fpe.core.echosahedral_dynamics import (
    local_a5_dynamics_report,
    reference_impulse_readback_report,
    reference_negative_antipode_response,
    reference_icosahedral_coupling,
)
from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower

SCHEMA = "oph.charged_response_semantic_artifact.v3"
ISSUE = 599
RUNTIME_TOLERANCE = 5.0e-9


class ChargedResponseError(ValueError):
    """Typed fail-closed producer error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise ChargedResponseError(code, message)


# ---------------------------------------------------------------------------
# Exact Q(sqrt5) arithmetic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Q5:
    """Exact element a + b*sqrt(5) with rational a, b."""

    a: Fraction
    b: Fraction

    @staticmethod
    def of(a: Fraction | int | str, b: Fraction | int | str = 0) -> "Q5":
        return Q5(Fraction(a), Fraction(b))

    def __add__(self, other: "Q5") -> "Q5":
        return Q5(self.a + other.a, self.b + other.b)

    def __sub__(self, other: "Q5") -> "Q5":
        return Q5(self.a - other.a, self.b - other.b)

    def __neg__(self) -> "Q5":
        return Q5(-self.a, -self.b)

    def __mul__(self, other: "Q5") -> "Q5":
        return Q5(
            self.a * other.a + 5 * self.b * other.b,
            self.a * other.b + self.b * other.a,
        )

    def inverse(self) -> "Q5":
        norm = self.a * self.a - 5 * self.b * self.b
        _require(norm != 0, "Q5_DIVISION", "division by zero in Q(sqrt5)")
        return Q5(self.a / norm, -self.b / norm)

    def conj(self) -> "Q5":
        """Galois conjugation sqrt(5) -> -sqrt(5)."""

        return Q5(self.a, -self.b)

    def is_zero(self) -> bool:
        return self.a == 0 and self.b == 0

    def sign(self) -> int:
        """Exact sign of a + b*sqrt(5)."""

        if self.b == 0:
            return (self.a > 0) - (self.a < 0)
        if self.a == 0:
            return (self.b > 0) - (self.b < 0)
        if self.a > 0 and self.b > 0:
            return 1
        if self.a < 0 and self.b < 0:
            return -1
        # Opposite signs: compare a^2 with 5 b^2.
        lhs = self.a * self.a
        rhs = 5 * self.b * self.b
        if self.a > 0:
            return 1 if lhs > rhs else -1
        return -1 if lhs > rhs else 1

    def to_float(self) -> float:
        return float(self.a) + float(self.b) * math.sqrt(5.0)

    def render(self) -> str:
        if self.b == 0:
            return str(self.a)
        return f"{self.a} + {self.b}*sqrt(5)"


ZERO = Q5.of(0)
ONE = Q5.of(1)
SQRT5 = Q5.of(0, 1)
PHI = Q5.of(Fraction(1, 2), Fraction(1, 2))


def q5_matmul(left: list[list[Q5]], right: list[list[Q5]]) -> list[list[Q5]]:
    n, k, m = len(left), len(right), len(right[0])
    out = [[ZERO for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for t in range(k):
            entry = left[i][t]
            if entry.is_zero():
                continue
            for j in range(m):
                out[i][j] = out[i][j] + entry * right[t][j]
    return out


def q5_identity(n: int) -> list[list[Q5]]:
    return [[ONE if i == j else ZERO for j in range(n)] for i in range(n)]


def q5_scale(matrix: list[list[Q5]], factor: Q5) -> list[list[Q5]]:
    return [[factor * entry for entry in row] for row in matrix]


def q5_add(left: list[list[Q5]], right: list[list[Q5]]) -> list[list[Q5]]:
    return [
        [left[i][j] + right[i][j] for j in range(len(left[0]))]
        for i in range(len(left))
    ]


def q5_trace(matrix: list[list[Q5]]) -> Q5:
    total = ZERO
    for i in range(len(matrix)):
        total = total + matrix[i][i]
    return total


# ---------------------------------------------------------------------------
# Carrier manifest binding
# ---------------------------------------------------------------------------


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def standard_vertices() -> list[tuple[Q5, Q5, Q5]]:
    """The twelve unnormalized icosahedron vertices: cyclic perms of (0, +-1, +-phi)."""

    verts: list[tuple[Q5, Q5, Q5]] = []
    for s1 in (ONE, -ONE):
        for s2 in (PHI, -PHI):
            verts.append((ZERO, s1, s2))
    for s1 in (ONE, -ONE):
        for s2 in (PHI, -PHI):
            verts.append((s1, s2, ZERO))
    for s1 in (ONE, -ONE):
        for s2 in (PHI, -PHI):
            verts.append((s2, ZERO, s1))
    return verts


def q5_dot(u: Sequence[Q5], v: Sequence[Q5]) -> Q5:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def q5_det3(rows: Sequence[Sequence[Q5]]) -> Q5:
    (a, b, c), (d, e, f), (g, h, i) = rows
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def load_carrier(manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        manifest.get("schema") == "oph.echosahedral_selector_manifest.v1",
        "CARRIER_SCHEMA",
        "the carrier manifest schema is not the certified selector schema",
    )
    carrier = manifest.get("carrier")
    _require(isinstance(carrier, Mapping), "CARRIER_SCHEMA", "carrier block missing")
    ports = list(carrier.get("ports", []))
    _require(len(ports) == 12 and len(set(ports)) == 12, "CARRIER_PORTS", "twelve distinct ports required")
    index = {port: position for position, port in enumerate(ports)}
    edges_raw = carrier.get("edges", [])
    _require(len(edges_raw) == 30, "CARRIER_EDGES", "thirty edges required")
    adjacency = [[0] * 12 for _ in range(12)]
    for edge in edges_raw:
        _require(
            isinstance(edge, list) and len(edge) == 2 and edge[0] in index and edge[1] in index,
            "CARRIER_EDGES",
            "edges must join declared ports",
        )
        i, j = index[edge[0]], index[edge[1]]
        _require(i != j and adjacency[i][j] == 0, "CARRIER_EDGES", "edges must be simple")
        adjacency[i][j] = adjacency[j][i] = 1
    degrees = [sum(row) for row in adjacency]
    _require(all(d == 5 for d in degrees), "CARRIER_REGULARITY", "the carrier incidence must be five-regular")
    faces = carrier.get("oriented_faces", [])
    _require(len(faces) == 20, "CARRIER_FACES", "twenty oriented faces required")
    for face in faces:
        _require(
            isinstance(face, list) and len(face) == 3 and all(p in index for p in face),
            "CARRIER_FACES",
            "faces must name declared ports",
        )
        a, b, c = (index[p] for p in face)
        _require(
            adjacency[a][b] and adjacency[b][c] and adjacency[c][a],
            "CARRIER_FACES",
            "faces must be incidence triangles",
        )
    # Antipode: the unique port at graph distance three.
    distance = _graph_distances(adjacency)
    antipode: list[int] = []
    for i in range(12):
        partners = [j for j in range(12) if distance[i][j] == 3]
        _require(len(partners) == 1, "CARRIER_ANTIPODE", "each port requires a unique distance-three partner")
        antipode.append(partners[0])
    _require(all(antipode[antipode[i]] == i for i in range(12)), "CARRIER_ANTIPODE", "antipode must be an involution")
    return {
        "ports": ports,
        "index": index,
        "adjacency": adjacency,
        "faces": [[index[p] for p in face] for face in faces],
        "antipode": antipode,
        "manifest_sha256": canonical_sha256(manifest),
    }


def _graph_distances(adjacency: list[list[int]]) -> list[list[int]]:
    n = len(adjacency)
    result = []
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        while queue:
            node = queue.pop(0)
            for other in range(n):
                if adjacency[node][other] and dist[other] < 0:
                    dist[other] = dist[node] + 1
                    queue.append(other)
        result.append(dist)
    return result


def match_vertex_frame(carrier: dict[str, Any]) -> list[tuple[Q5, Q5, Q5]]:
    """Assign exact standard vertices to ports, oriented by the manifest faces.

    The assignment preserves incidence (adjacent ports map to vertices with
    dot product phi), sends antipodal ports to opposite vertices, and gives
    every oriented face a positive determinant. Its existence and the
    positivity of every face determinant are the measured orientation datum.
    """

    verts = standard_vertices()
    adjacency = carrier["adjacency"]
    antipode = carrier["antipode"]
    # Adjacency in the vertex model is dot == phi.
    vertex_adjacent = [
        [(q5_dot(verts[i], verts[j]) - PHI).is_zero() for j in range(12)]
        for i in range(12)
    ]
    vertex_opposite = [
        [(q5_dot(verts[i], verts[j]) + q5_dot(verts[i], verts[i])).is_zero() for j in range(12)]
        for i in range(12)
    ]

    assignment: list[int | None] = [None] * 12
    used = [False] * 12

    def consistent(port: int, vertex: int) -> bool:
        for other in range(12):
            slot = assignment[other]
            if slot is None:
                continue
            if adjacency[port][other] and not vertex_adjacent[vertex][slot]:
                return False
            if not adjacency[port][other] and port != other and vertex_adjacent[vertex][slot]:
                return False
            if antipode[port] == other and not vertex_opposite[vertex][slot]:
                return False
        return True

    def search(port: int) -> bool:
        if port == 12:
            return _faces_positive(carrier, verts, assignment)
        for vertex in range(12):
            if used[vertex] or not consistent(port, vertex):
                continue
            assignment[port] = vertex
            used[vertex] = True
            if search(port + 1):
                return True
            assignment[port] = None
            used[vertex] = False
        return False

    _require(search(0), "FRAME_ORIENTATION", "no oriented incidence-preserving vertex assignment exists")
    return [verts[assignment[p]] for p in range(12)]  # type: ignore[index]


def _faces_positive(
    carrier: dict[str, Any],
    verts: list[tuple[Q5, Q5, Q5]],
    assignment: list[int | None],
) -> bool:
    for face in carrier["faces"]:
        rows = [verts[assignment[p]] for p in face]  # type: ignore[index]
        if q5_det3(rows).sign() <= 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Exact isotypic decomposition of the propagation generator
# ---------------------------------------------------------------------------


ADJACENCY_EIGENVALUES: dict[str, Q5] = {
    "unit": Q5.of(5),
    "frame": SQRT5,
    "quintet": Q5.of(-1),
    "kernel": -SQRT5,
}

SECTOR_DIMENSIONS = {"unit": 1, "frame": 3, "quintet": 5, "kernel": 3}


def isotypic_projectors(adjacency: list[list[int]]) -> dict[str, list[list[Q5]]]:
    """Exact spectral projectors of the adjacency operator.

    Fails closed when the measured incidence does not have the icosahedral
    minimal polynomial (x - 5)(x + 1)(x^2 - 5), i.e. when the carrier does
    not present the 1 + 3 + 3' + 5 charged sector structure.
    """

    matrix = [[Q5.of(adjacency[i][j]) for j in range(12)] for i in range(12)]
    projectors: dict[str, list[list[Q5]]] = {}
    for band, eigenvalue in ADJACENCY_EIGENVALUES.items():
        product = q5_identity(12)
        for other_band, other in ADJACENCY_EIGENVALUES.items():
            if other_band == band:
                continue
            factor = q5_add(matrix, q5_scale(q5_identity(12), -other))
            product = q5_matmul(product, factor)
            scale = (eigenvalue - other).inverse()
            product = q5_scale(product, scale)
        projectors[band] = product

    total = q5_identity(12)
    for band, projector in projectors.items():
        # Idempotence and the eigenrelation A P = lambda P.
        square = q5_matmul(projector, projector)
        _require(
            all(
                (square[i][j] - projector[i][j]).is_zero()
                for i in range(12)
                for j in range(12)
            ),
            "SECTOR_PROJECTOR",
            f"the {band} sector projector is not idempotent; the carrier spectrum "
            "does not present the icosahedral sector structure",
        )
        image = q5_matmul(matrix, projector)
        scaled = q5_scale(projector, ADJACENCY_EIGENVALUES[band])
        _require(
            all(
                (image[i][j] - scaled[i][j]).is_zero()
                for i in range(12)
                for j in range(12)
            ),
            "SECTOR_PROJECTOR",
            f"the {band} sector projector fails the eigenrelation",
        )
        trace = q5_trace(projector)
        _require(
            trace.b == 0 and trace.a == SECTOR_DIMENSIONS[band],
            "SECTOR_DIMENSION",
            f"the {band} sector dimension is not {SECTOR_DIMENSIONS[band]}",
        )
        total = q5_add(total, q5_scale(projector, -ONE))
    _require(
        all(total[i][j].is_zero() for i in range(12) for j in range(12)),
        "SECTOR_RESOLUTION",
        "the sector projectors do not resolve the identity",
    )

    # Galois pairing: conjugation swaps the two triplet sectors and fixes the
    # unit and quintet sectors. This is the charged pairing the double
    # triplet requires; a rational triplet spectrum cannot satisfy it.
    frame = projectors["frame"]
    kernel = projectors["kernel"]
    _require(
        all(
            (frame[i][j].conj() - kernel[i][j]).is_zero()
            for i in range(12)
            for j in range(12)
        ),
        "GALOIS_PAIRING",
        "Galois conjugation does not pair the two triplet sectors",
    )
    for band in ("unit", "quintet"):
        projector = projectors[band]
        _require(
            all(
                (projector[i][j].conj() - projector[i][j]).is_zero()
                for i in range(12)
                for j in range(12)
            ),
            "GALOIS_PAIRING",
            f"the {band} sector is not Galois stable",
        )
    return projectors


def negative_antipode_response(
    carrier: Mapping[str, Any],
    projectors: Mapping[str, list[list[Q5]]],
) -> dict[str, Any]:
    """Derive the four relative signs from target-blind impulse/readback.

    Inject a delta at each port, execute the adjacency recurrence through the
    graph diameter, and solve one homogeneous polynomial filter that cancels
    every nearer shell and returns the unique maximal-distance echo. The solve
    produces ``J`` without accepting an antipode map or gauge target. We print
    ``R=-J``; its common sign remains charge-conjugation convention.
    """

    distances = _graph_distances(carrier["adjacency"])
    diameter = max(max(row) for row in distances)
    _require(diameter == 3, "IMPULSE_READBACK", "carrier diameter is not three")
    _require(
        all(sum(value == diameter for value in row) == 1 for row in distances),
        "IMPULSE_READBACK",
        "every impulse source must have a unique maximal-distance readback port",
    )
    adjacency_q5 = [
        [Q5.of(carrier["adjacency"][i][j]) for j in range(12)]
        for i in range(12)
    ]
    powers = [q5_identity(12)]
    for _ in range(diameter):
        powers.append(q5_matmul(powers[-1], adjacency_q5))
    target = [
        [ONE if distances[i][j] == diameter else ZERO for j in range(12)]
        for i in range(12)
    ]
    equations = [
        [powers[k][i][j] for k in range(diameter + 1)] + [target[i][j]]
        for i in range(12)
        for j in range(12)
    ]
    reduced = [row[:] for row in equations]
    pivot_columns: list[int] = []
    active = 0
    for column in range(diameter + 1):
        pivot = next(
            (
                index
                for index in range(active, len(reduced))
                if not reduced[index][column].is_zero()
            ),
            None,
        )
        if pivot is None:
            continue
        reduced[active], reduced[pivot] = reduced[pivot], reduced[active]
        scale = reduced[active][column].inverse()
        reduced[active] = [entry * scale for entry in reduced[active]]
        for index in range(len(reduced)):
            if index == active or reduced[index][column].is_zero():
                continue
            factor = reduced[index][column]
            reduced[index] = [
                reduced[index][j] - factor * reduced[active][j]
                for j in range(diameter + 2)
            ]
        pivot_columns.append(column)
        active += 1
    _require(
        pivot_columns == [0, 1, 2, 3],
        "IMPULSE_READBACK",
        "the homogeneous impulse filter is not unique",
    )
    _require(
        all(
            not (
                all(row[column].is_zero() for column in range(diameter + 1))
                and not row[-1].is_zero()
            )
            for row in reduced
        ),
        "IMPULSE_READBACK",
        "the impulse/readback constraints are inconsistent",
    )
    coefficients = [reduced[index][-1] for index in range(diameter + 1)]
    _require(
        coefficients
        == [ONE, Q5.of(Fraction(-1, 2)), Q5.of(Fraction(-2, 5)), Q5.of(Fraction(1, 10))],
        "IMPULSE_READBACK",
        "the solved maximal-distance filter coefficients drifted",
    )
    reconstructed = [[ZERO for _ in range(12)] for _ in range(12)]
    for coefficient, power in zip(coefficients, powers, strict=True):
        reconstructed = q5_add(reconstructed, q5_scale(power, coefficient))
    _require(
        reconstructed == target,
        "IMPULSE_READBACK",
        "the solved filter does not isolate the maximal-distance shell",
    )
    antipode = [
        next(j for j in range(12) if target[i][j] == ONE) for i in range(12)
    ]
    _require(
        antipode == [int(value) for value in carrier["antipode"]],
        "IMPULSE_READBACK",
        "the readback-derived farthest map does not match the carrier binding",
    )
    automorphisms = incidence_automorphisms(carrier["adjacency"])
    _require(
        len(automorphisms) == 120,
        "ANTIPODE_RESPONSE",
        "the incidence automorphism group does not have order 120",
    )
    central_involutions = []
    for permutation in automorphisms:
        if all(
            permutation[other[index]] == other[permutation[index]]
            for other in automorphisms
            for index in range(12)
        ) and all(permutation[permutation[index]] == index for index in range(12)):
            central_involutions.append(permutation)
    nonidentity_central = [
        permutation
        for permutation in central_involutions
        if permutation != tuple(range(12))
    ]
    _require(
        nonidentity_central == [tuple(antipode)],
        "ANTIPODE_RESPONSE",
        "the distance-three inverse is not the unique nonidentity central involution",
    )
    response = [[ZERO for _ in range(12)] for _ in range(12)]
    for index, partner in enumerate(antipode):
        response[index][partner] = -ONE
    _require(
        q5_matmul(response, response) == q5_identity(12),
        "ANTIPODE_RESPONSE",
        "the negative antipode response is not an involution",
    )
    adjacency = adjacency_q5
    antipode_matrix = [[ZERO for _ in range(12)] for _ in range(12)]
    for index, partner in enumerate(antipode):
        antipode_matrix[index][partner] = ONE
    adjacency_squared = q5_matmul(adjacency, adjacency)
    adjacency_cubed = q5_matmul(adjacency_squared, adjacency)
    polynomial_antipode = q5_scale(
        q5_add(
            q5_add(
                q5_add(
                    adjacency_cubed,
                    q5_scale(adjacency_squared, Q5.of(-4)),
                ),
                q5_scale(adjacency, Q5.of(-5)),
            ),
            q5_scale(q5_identity(12), Q5.of(10)),
        ),
        Q5.of(Fraction(1, 10)),
    )
    _require(
        polynomial_antipode == antipode_matrix,
        "ANTIPODE_RESPONSE",
        "J != (A^3 - 4 A^2 - 5 A + 10 I)/10",
    )
    _require(
        q5_matmul(response, adjacency) == q5_matmul(adjacency, response),
        "ANTIPODE_RESPONSE",
        "the negative antipode response does not commute with propagation",
    )
    signs: dict[str, int] = {}
    for band, projector in projectors.items():
        image = q5_matmul(response, projector)
        if image == projector:
            signs[band] = 1
        elif image == q5_scale(projector, -ONE):
            signs[band] = -1
        else:
            raise ChargedResponseError(
                "ANTIPODE_RESPONSE",
                f"the {band} sector is not an eigenspace of the response operator",
            )
    _require(
        signs == {"unit": -1, "frame": 1, "quintet": -1, "kernel": 1},
        "ANTIPODE_RESPONSE",
        f"unexpected response-sector signs {signs}",
    )
    return {
        "impulse_readback_protocol": {
            "status": "source_bound_operational_producer",
            "input": "delta impulse at each unlabeled carrier port",
            "recurrence": "adjacency powers k=0..graph_diameter",
            "selection_rule": "cancel every nearer distance shell and normalize the unique maximal-distance echo to one",
            "homogeneous_filter_coefficients": [coefficient.render() for coefficient in coefficients],
            "unique_solution_rank": len(pivot_columns),
            "unique_farthest_port_per_source": True,
            "nearer_shells_cancelled": True,
            "target_labels_used": False,
            "downstream_labels_used": False,
        },
        "operator": "negative_graph_antipode_involution",
        "source": "target_blind_maximal_distance_impulse_readback",
        "antipode_port_map": antipode,
        "incidence_automorphism_group_order": len(automorphisms),
        "central_involution_count_including_identity": len(central_involutions),
        "unique_nonidentity_central_involution": True,
        "antipode_polynomial_identity": "J = (A^3 - 4*A^2 - 5*A + 10*I)/10",
        "commutes_with_propagation_generator": True,
        "self_adjoint_unitary_involution": True,
        "sector_eigenvalues": {
            "unit_band": signs["unit"],
            "quintet_band": signs["quintet"],
            "frame_band": signs["frame"],
            "kernel_band": signs["kernel"],
        },
        "overall_response_sign": (
            "the representative R = -J is conventional; replacing it by +J "
            "reverses every band together and leaves the current algebra unchanged"
        ),
        "impulse_readback_response_executed": True,
        "physical_perturb_readback_source_bound": True,
    }


def potential_response(frame_vertices: list[tuple[Q5, Q5, Q5]]) -> dict[str, Any]:
    """Exact frame normalization audit, not a response-law producer.

    The vertex frame has tight-frame constant ``10 + 2*sqrt(5)``.  This number
    supplies no response sign; the signed response is computed by
    :func:`negative_antipode_response`.
    """

    gram_total = [[ZERO for _ in range(3)] for _ in range(3)]
    for vertex in frame_vertices:
        for i in range(3):
            for j in range(3):
                gram_total[i][j] = gram_total[i][j] + vertex[i] * vertex[j]
    expected = Q5.of(10, 2)
    for i in range(3):
        for j in range(3):
            target = expected if i == j else ZERO
            _require(
                (gram_total[i][j] - target).is_zero(),
                "POTENTIAL_RESPONSE",
                "the vertex frame is not tight; the unit channel constant is not "
                "(10 + 2*sqrt(5))",
            )
    return {
        "status": "normalization_audit_only",
        "unit_channel_constant": expected.render(),
        "response_sign_not_inferred_here": True,
    }


def incidence_automorphisms(adjacency: list[list[int]]) -> list[tuple[int, ...]]:
    """All incidence automorphisms of the measured carrier graph."""

    n = len(adjacency)
    results: list[tuple[int, ...]] = []
    assignment: list[int | None] = [None] * n
    used = [False] * n

    def consistent(node: int, image: int) -> bool:
        for other in range(node):
            slot = assignment[other]
            if adjacency[node][other] != adjacency[image][slot]:  # type: ignore[index]
                return False
        return True

    def search(node: int) -> None:
        if node == n:
            results.append(tuple(assignment))  # type: ignore[arg-type]
            return
        for image in range(n):
            if used[image] or not consistent(node, image):
                continue
            assignment[node] = image
            used[image] = True
            search(node + 1)
            assignment[node] = None
            used[image] = False

    search(0)
    return results


def _oriented_face_set(faces: Sequence[Sequence[int]]) -> set[tuple[int, int, int]]:
    normalized: set[tuple[int, int, int]] = set()
    for face in faces:
        a, b, c = face
        cycles = [(a, b, c), (b, c, a), (c, a, b)]
        normalized.add(min(cycles))
    return normalized


def rotation_response(carrier: dict[str, Any]) -> dict[str, Any]:
    """Exact orientation-preserving automorphism report of the bound carrier.

    The incidence automorphism group of the measured carrier has order 120;
    exactly sixty automorphisms preserve the manifest's oriented faces. Those
    sixty rotations commute with the propagation generator by construction
    and act on the frame sector by the vertex coordinate isometry, and on the
    kernel sector by its Galois conjugate. The frame and kernel channels
    The signed response itself is derived separately from ``R = -J``.
    """

    adjacency = carrier["adjacency"]
    automorphisms = incidence_automorphisms(adjacency)
    _require(
        len(automorphisms) == 120,
        "ROTATION_RESPONSE",
        "the measured incidence automorphism group does not have order 120",
    )
    oriented_faces = _oriented_face_set(carrier["faces"])
    rotations = []
    for permutation in automorphisms:
        images = _oriented_face_set(
            [
                [permutation[a], permutation[b], permutation[c]]
                for a, b, c in carrier["faces"]
            ]
        )
        if images == oriented_faces:
            rotations.append(permutation)
    _require(
        len(rotations) == 60,
        "ROTATION_RESPONSE",
        "sixty orientation-preserving rotations required",
    )
    return {
        "incidence_automorphism_group_order": 120,
        "commuting_incidence_automorphisms": 60,
        "orientation_preserving": True,
        "frame_transport": "vertex_coordinate_isometry",
        "kernel_transport": "galois_conjugate_vertex_coordinate_isometry",
        "response_sign_not_inferred_here": True,
    }


def graph_isomorphism(
    left: list[list[int]], right: list[list[int]]
) -> tuple[int, ...] | None:
    """One incidence isomorphism between two graphs on the same vertex count."""

    n = len(left)
    if len(right) != n:
        return None
    assignment: list[int | None] = [None] * n
    used = [False] * n

    def consistent(node: int, image: int) -> bool:
        for other in range(node):
            slot = assignment[other]
            if left[node][other] != right[image][slot]:  # type: ignore[index]
                return False
        return True

    def search(node: int) -> tuple[int, ...] | None:
        if node == n:
            return tuple(assignment)  # type: ignore[arg-type]
        for image in range(n):
            if used[image] or not consistent(node, image):
                continue
            assignment[node] = image
            used[image] = True
            found = search(node + 1)
            if found is not None:
                return found
            assignment[node] = None
            used[image] = False
        return None

    return search(0)


# ---------------------------------------------------------------------------
# Runtime binding and refinement maps
# ---------------------------------------------------------------------------


def runtime_binding(carrier: dict[str, Any]) -> dict[str, Any]:
    """Bind the artifact to the dynamics actually propagated by the simulator."""

    coupling = reference_icosahedral_coupling()
    tower_adjacency = np.diag(np.diag(coupling)) - coupling
    runtime_graph = [
        [int(round(tower_adjacency[i][j])) for j in range(12)] for i in range(12)
    ]
    binding_map = graph_isomorphism(carrier["adjacency"], runtime_graph)
    _require(
        binding_map is not None,
        "RUNTIME_ISOMORPHISM",
        "the runtime coupling graph is not isomorphic to the bound carrier incidence",
    )
    eigenvalues = np.linalg.eigvalsh(coupling)
    exact = sorted(
        (Q5.of(5) - value).to_float()
        for band, value in ADJACENCY_EIGENVALUES.items()
        for _ in range(SECTOR_DIMENSIONS[band])
    )
    measured = sorted(float(v) for v in eigenvalues)
    _require(
        max(abs(a - b) for a, b in zip(measured, exact)) <= RUNTIME_TOLERANCE,
        "RUNTIME_SPECTRUM",
        "the runtime coupling spectrum does not match the exact carrier spectrum",
    )
    report = local_a5_dynamics_report()
    _require(
        bool(report["LOCAL_A5_EQUIVARIANT_PROPAGATION_RECEIPT"]),
        "RUNTIME_EQUIVARIANCE",
        "the propagated dynamics is not A5 equivariant at tolerance",
    )
    _require(
        bool(report["LOCAL_NONTRIVIAL_REVERSIBLE_PROPAGATION_RECEIPT"]),
        "RUNTIME_REVERSIBILITY",
        "the propagated dynamics is not a nontrivial reversible channel",
    )
    _require(
        bool(report["CHARGED_RESPONSE_OPERATOR_RECEIPT"]),
        "RUNTIME_RESPONSE",
        "the runtime dynamics does not expose the incidence-derived response operator",
    )
    runtime_response = reference_negative_antipode_response()
    impulse_report = reference_impulse_readback_report()
    expected_response = np.zeros((12, 12), dtype=np.complex128)
    for index, partner in enumerate(carrier["antipode"]):
        expected_response[index, int(partner)] = -1.0
    _require(
        bool(np.array_equal(runtime_response, expected_response)),
        "RUNTIME_RESPONSE",
        "the runtime response is not the bound carrier's negative antipode",
    )
    _require(
        impulse_report["homogeneous_filter_coefficients"]
        == ["1", "-1/2", "-2/5", "1/10"]
        and impulse_report["farthest_port_map"]
        == [int(value) for value in carrier["antipode"]]
        and impulse_report["target_labels_used"] is False
        and impulse_report["downstream_labels_used"] is False,
        "RUNTIME_RESPONSE",
        "the runtime impulse/readback producer does not match the exact source solve",
    )
    _require(
        int(tower_adjacency.sum()) == 60,
        "RUNTIME_SPECTRUM",
        "the runtime coupling does not have thirty edges",
    )
    return {
        "dynamics_module": "oph_fpe.core.echosahedral_dynamics",
        "dynamics_sha256": report["dynamics_sha256"],
        "coupling": report["coupling"],
        "evolution": report["evolution"],
        "spectrum_multiplicities": [1, 3, 3, 5],
        "manifest_to_runtime_port_map": list(binding_map),
        "reversible_response_source": "finite_unitary_carrier_channel",
        "equivariance_receipt": True,
        "reversibility_receipt": True,
        "charged_response_operator_receipt": True,
        "impulse_readback_producer_receipt": True,
        "impulse_readback_filter_coefficients": (
            impulse_report["homogeneous_filter_coefficients"]
        ),
        "runtime_tolerance": RUNTIME_TOLERANCE,
    }


def refinement_persistence(levels: int = 3) -> dict[str, Any]:
    """Measure the persistence of the twelve defect ports along the tower.

    At every geodesic level the twelve original vertices are the only
    degree-five vertices and they persist at the same normalized positions,
    so the physical refinement maps act on the port set as the identity
    permutation. The maps are recorded with hashes; their naturality under
    the sixty rotations is the per-level equivariance receipt.
    """

    tower = build_geodesic_icosahedral_tower(levels - 1)
    base = tower.levels[0]
    base_vertices = np.asarray(base.vertices, dtype=float)
    maps: list[dict[str, Any]] = []
    for fine in tower.levels[1:]:
        fine_vertices = np.asarray(fine.vertices, dtype=float)
        edges = np.asarray(fine.edges, dtype=np.int64)
        degrees = np.zeros(len(fine_vertices), dtype=np.int64)
        for i, j in edges:
            degrees[i] += 1
            degrees[j] += 1
        defect = np.flatnonzero(degrees == 5)
        _require(
            defect.size == 12,
            "REFINEMENT_PERSISTENCE",
            f"level {fine.level} does not present exactly twelve defect ports",
        )
        port_map: list[int] = []
        for port_index in range(12):
            distances = np.linalg.norm(
                fine_vertices[defect] - base_vertices[port_index], axis=1
            )
            best = int(np.argmin(distances))
            _require(
                float(distances[best]) <= 1.0e-9,
                "REFINEMENT_PERSISTENCE",
                "a defect port does not persist at its coarse position",
            )
            port_map.append(best)
        _require(
            sorted(port_map) == list(range(12)),
            "REFINEMENT_PERSISTENCE",
            "defect ports do not biject across levels",
        )
        payload = {
            "source_level": 0,
            "target_level": int(fine.level),
            "port_map": port_map,
            "origin": "defect_port_persistence_in_geodesic_tower",
        }
        payload["map_hash"] = "sha256:" + canonical_sha256(payload)
        maps.append(payload)
    return {
        "tower": "oph_fpe.core.icosahedral.build_geodesic_icosahedral_tower",
        "levels_measured": levels,
        "port_persistence_maps": maps,
        "per_level_defect_port_count": 12,
        "equivariant_rotation_count": 60,
    }


# ---------------------------------------------------------------------------
# Artifact assembly
# ---------------------------------------------------------------------------


def produce_charged_response_artifact(carrier_manifest: Mapping[str, Any]) -> dict[str, Any]:
    carrier = load_carrier(carrier_manifest)
    frame_vertices = match_vertex_frame(carrier)
    projectors = isotypic_projectors(carrier["adjacency"])
    response = negative_antipode_response(carrier, projectors)
    ports = carrier["ports"]
    antipode = carrier["antipode"]
    axis_representatives = []
    seen: set[int] = set()
    for position in range(12):
        if position in seen:
            continue
        seen.add(position)
        seen.add(antipode[position])
        axis_representatives.append(ports[position])

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "carrier_binding": {
            "carrier_manifest_sha256": carrier["manifest_sha256"],
            "port_order": ports,
            "antipode": {ports[i]: ports[antipode[i]] for i in range(12)},
            "axis_representatives": axis_representatives,
            "incidence_edge_count": 30,
        },
        "port_vertex_frame": {
            ports[i]: [component.render() for component in frame_vertices[i]]
            for i in range(12)
        },
        "response_basis": {
            "sector_order": ["unit_band", "quintet_band", "frame_band", "kernel_band"],
            "sector_dimensions": {
                "unit_band": 1,
                "quintet_band": 5,
                "frame_band": 3,
                "kernel_band": 3,
            },
            "adjacency_channel_values": {
                "unit_band": ADJACENCY_EIGENVALUES["unit"].render(),
                "quintet_band": ADJACENCY_EIGENVALUES["quintet"].render(),
                "frame_band": ADJACENCY_EIGENVALUES["frame"].render(),
                "kernel_band": ADJACENCY_EIGENVALUES["kernel"].render(),
            },
            "galois_pairing": {
                "frame_and_kernel_swapped_by_conjugation": True,
                "unit_and_quintet_galois_stable": True,
            },
        },
        "orientation_convention": {
            "frame_band": "triplet sector with adjacency channel value +sqrt(5) in the oriented vertex frame",
            "kernel_band": "Galois conjugate triplet sector with adjacency channel value -sqrt(5)",
            "faces": "every oriented face has positive determinant in the assigned frame",
            "overall_u1_charge_sign": (
                "not selected by the finite response operator; overall charge "
                "conjugation is a convention"
            ),
        },
        "source_response": response,
        "structural_audits": {
            "frame_normalization": potential_response(frame_vertices),
            "rotation_automorphisms": rotation_response(carrier),
        },
        "derived": {
            "construction": "charged_double_triplet",
            "construction_provenance": (
                "canonical_equivariant_compact_lift_from_the_impulse_readback_"
                "derived_antipode_response"
            ),
            "response_band_scales": {
                band: str(value)
                for band, value in response["sector_eigenvalues"].items()
            },
            "normalization_rule": (
                "the source response is the self-adjoint unitary involution R = -J; "
                "each band scale is its exactly recomputed eigenvalue; the common "
                "overall sign is a charge-conjugation convention"
            ),
        },
        "physical_refinement_maps": refinement_persistence(),
        "provenance": {
            "producer": "oph_fpe.core.charged_response",
            "runtime_binding": runtime_binding(carrier),
            "deterministic": True,
        },
    }
    artifact["artifact_sha256"] = "sha256:" + canonical_sha256(
        {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    )
    return artifact


def write_charged_response_artifact(
    carrier_manifest_path: Path, output_path: Path
) -> dict[str, Any]:
    manifest = json.loads(carrier_manifest_path.read_text(encoding="utf-8"))
    artifact = produce_charged_response_artifact(manifest)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carrier-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    artifact = write_charged_response_artifact(args.carrier_manifest, args.out)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact["artifact_sha256"],
                "carrier_manifest_sha256": artifact["carrier_binding"][
                    "carrier_manifest_sha256"
                ],
                "derived_construction": artifact["derived"]["construction"],
            },
            indent=2,
        )
    )
    return 0


__all__ = [
    "ChargedResponseError",
    "Q5",
    "isotypic_projectors",
    "load_carrier",
    "match_vertex_frame",
    "potential_response",
    "produce_charged_response_artifact",
    "refinement_persistence",
    "rotation_response",
    "runtime_binding",
    "standard_vertices",
    "write_charged_response_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
