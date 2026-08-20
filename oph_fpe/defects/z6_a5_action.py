"""Icosahedral symmetry action for the lane C6 defect census (exploratory,
non-evidential).

Design record: ``oph_fpe/defects/DESIGN.md`` section 7. The rotation group
is derived from the committed base-carrier tables alone (port adjacency and
oriented faces); the concurrently owned ``oph_fpe/core/icosahedral.py`` is
not imported. The derived group is receipted as A5 by order, element-order
histogram, transitivity, and orientation preservation.

Content: graph automorphism search; the orientation-preserving subgroup;
the signed seam action on configurations; the induced sector action on
chord-holonomy classes; orbit data; fail-closed receipts. This module
contains no committed matter-table values (DESIGN.md section 10).
"""

from __future__ import annotations

from typing import Sequence

from oph_fpe.defects.z6_carrier_defects import (
    MOD,
    CarrierDefectError,
    CarrierSpec,
    Config,
    SectorClass,
    chord_holonomies,
    face_curvature,
    mismatch_energy,
    sector_representative,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CarrierDefectError(message)


# ---------------------------------------------------------------------------
# Graph automorphisms and the orientation-preserving subgroup
# ---------------------------------------------------------------------------

def _adjacency_sets(spec: CarrierSpec) -> list[set[int]]:
    adjacency: list[set[int]] = [set() for _ in range(spec.ports)]
    for l, r in zip(spec.seam_left, spec.seam_right, strict=True):
        adjacency[l].add(r)
        adjacency[r].add(l)
    return adjacency


def graph_automorphisms(spec: CarrierSpec) -> list[tuple[int, ...]]:
    """All port permutations preserving the seam adjacency relation, by
    depth-first extension with adjacency consistency."""
    adjacency = _adjacency_sets(spec)
    n = spec.ports
    results: list[tuple[int, ...]] = []
    assignment: list[int] = [-1] * n
    used = [False] * n

    def consistent(vertex: int, image: int) -> bool:
        for other in range(vertex):
            other_image = assignment[other]
            if (other in adjacency[vertex]) != (
                other_image in adjacency[image]
            ):
                return False
        return True

    def extend(vertex: int) -> None:
        if vertex == n:
            results.append(tuple(assignment))
            return
        for image in range(n):
            if used[image] or not consistent(vertex, image):
                continue
            assignment[vertex] = image
            used[image] = True
            extend(vertex + 1)
            assignment[vertex] = -1
            used[image] = False

    extend(0)
    return results


def _face_index_map(spec: CarrierSpec) -> dict[frozenset[int], int]:
    table = {
        frozenset(face): f for f, face in enumerate(spec.oriented_faces)
    }
    _require(len(table) == spec.faces, "duplicate face vertex sets")
    return table


def _is_orientation_preserving(spec: CarrierSpec,
                               perm: Sequence[int]) -> bool:
    rotations = {
        face: {(a, b, c), (b, c, a), (c, a, b)}
        for face in spec.oriented_faces
        for a, b, c in [face]
    }
    by_vertices = _face_index_map(spec)
    for face in spec.oriented_faces:
        image = (perm[face[0]], perm[face[1]], perm[face[2]])
        key = frozenset(image)
        if key not in by_vertices:
            return False
        target = spec.oriented_faces[by_vertices[key]]
        if image not in rotations[target]:
            return False
    return True


def rotation_group(spec: CarrierSpec) -> list[tuple[int, ...]]:
    """The orientation-preserving automorphisms, sorted lexicographically
    (the identity first); receipted as A5 by ``rotation_group_receipt``."""
    rotations = [
        perm for perm in graph_automorphisms(spec)
        if _is_orientation_preserving(spec, perm)
    ]
    rotations.sort()
    return rotations


def _perm_order(perm: Sequence[int]) -> int:
    order = 1
    current = list(perm)
    identity = list(range(len(perm)))
    while current != identity:
        current = [perm[x] for x in current]
        order += 1
    return order


def rotation_group_receipt(spec: CarrierSpec,
                           rotations: Sequence[Sequence[int]]) -> dict:
    """Fail-closed A5 receipts: order 60, identity, closure, element-order
    histogram {1: 1, 2: 15, 3: 20, 5: 24}, port transitivity."""
    _require(len(rotations) == 60, "rotation group order drift")
    index = {tuple(perm): i for i, perm in enumerate(rotations)}
    _require(tuple(range(spec.ports)) in index, "identity missing")
    for perm in rotations:
        for other in rotations:
            composed = tuple(perm[other[p]] for p in range(spec.ports))
            _require(composed in index, "closure drift")
    histogram: dict[int, int] = {}
    for perm in rotations:
        histogram[_perm_order(perm)] = histogram.get(_perm_order(perm), 0) + 1
    _require(histogram == {1: 1, 2: 15, 3: 20, 5: 24},
             "element-order histogram drift (not the A5 profile)")
    images_of_zero = {perm[0] for perm in rotations}
    _require(len(images_of_zero) == spec.ports, "port transitivity drift")
    return {
        "schema": "oph.sim.defect_census.rotation_group_receipt.v1",
        "exploratory": True,
        "evidential": False,
        "order": len(rotations),
        "element_order_histogram": {
            str(k): v for k, v in sorted(histogram.items())
        },
        "port_transitive": True,
        "orientation_preserving": True,
    }


# ---------------------------------------------------------------------------
# Signed seam action and induced sector action
# ---------------------------------------------------------------------------

def seam_action_table(spec: CarrierSpec,
                      perm: Sequence[int],
                      signed: bool = True
                      ) -> tuple[tuple[int, int], ...]:
    """Per seam ``e``: ``(image_seam, sign)``. The unsigned variant
    (``signed=False``) is the mutation-guard control and drops the
    orientation sign."""
    seam_index = {
        (spec.seam_left[e], spec.seam_right[e]): e
        for e in range(spec.seams)
    }
    table: list[tuple[int, int]] = []
    for e in range(spec.seams):
        a, b = perm[spec.seam_left[e]], perm[spec.seam_right[e]]
        if (a, b) in seam_index:
            table.append((seam_index[(a, b)], 1))
        else:
            _require((b, a) in seam_index, "seam image missing")
            table.append((seam_index[(b, a)], -1 if signed else 1))
    _require(len({image for image, _ in table}) == spec.seams,
             "seam action not a permutation")
    return tuple(table)


def act_on_config(spec: CarrierSpec, perm: Sequence[int],
                  config: Sequence[int], signed: bool = True) -> Config:
    """``(sigma . A)(image(e)) = sign(e) A(e)`` mod 6."""
    table = seam_action_table(spec, perm, signed=signed)
    moved = [0] * spec.seams
    for e in range(spec.seams):
        image, sign = table[e]
        moved[image] = (sign * config[e]) % MOD
    return moved


def act_on_sector(spec: CarrierSpec, perm: Sequence[int],
                  sector: Sequence[int]) -> SectorClass:
    """The induced sector action: transport the tree-trivial representative
    and read chord holonomies (well-defined because gauge moves transport
    to gauge moves; receipted)."""
    rep = sector_representative(spec, sector)
    return chord_holonomies(spec, act_on_config(spec, perm, rep))


def sector_orbit(spec: CarrierSpec, rotations: Sequence[Sequence[int]],
                 sector: Sequence[int]) -> tuple[SectorClass, int]:
    """``(canonical_representative, orbit_size)``: the lexicographically
    smallest image and the image count under all rotations."""
    images = {
        act_on_sector(spec, perm, sector) for perm in rotations
    }
    return min(images), len(images)


# ---------------------------------------------------------------------------
# Action receipts (fail-closed)
# ---------------------------------------------------------------------------

def action_receipt(spec: CarrierSpec, rotations: Sequence[Sequence[int]],
                   perm: Sequence[int],
                   sample_configs: Sequence[Sequence[int]],
                   signed: bool = True) -> bool:
    """DESIGN.md section 7 invariances for one rotation on samples:
    curvature transports by the face permutation with sign +1, energy is
    invariant, and the sector action matches the configuration action.
    Returns False on the first failure (the unsigned mutant fails)."""
    by_vertices = _face_index_map(spec)
    face_image = [
        by_vertices[frozenset(perm[v] for v in face)]
        for face in spec.oriented_faces
    ]
    for config in sample_configs:
        moved = act_on_config(spec, perm, config, signed=signed)
        curvature = face_curvature(spec, config)
        moved_curvature = face_curvature(spec, moved)
        for f in range(spec.faces):
            if moved_curvature[face_image[f]] != curvature[f] % MOD:
                return False
        if mismatch_energy(spec, moved) != mismatch_energy(spec, config):
            return False
        if act_on_sector(
            spec, perm, chord_holonomies(spec, config)
        ) != chord_holonomies(spec, moved):
            return False
    identity = tuple(range(spec.ports))
    for config in sample_configs:
        sector = chord_holonomies(spec, config)
        if act_on_sector(spec, identity, sector) != sector:
            return False
    return True


def declared_generator(rotations: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    """The declared test generator: the lexicographically smallest
    non-identity rotation (DESIGN.md section 11)."""
    for perm in rotations:
        if perm != tuple(range(len(perm))):
            return perm
    raise CarrierDefectError("no non-identity rotation")
