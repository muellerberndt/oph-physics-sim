"""Z/6 link configurations and conserved sector classes on a port-graph
carrier (lane C6, exploratory, non-evidential).

Design record: ``oph_fpe/defects/DESIGN.md``, fixed before this module.
Committed carrier source: ``oph_fpe.em.base_carrier`` (read-only import).
Committed classification surface (RER):
``Lean/Screen/SeamU1HolonomyClassification.lean`` — gauge orbits of seam
connections are classified by the nineteen chord holonomies; this module
discretizes the structure group U(1) to Z/6.

Content: a carrier interface accepting any port-graph with a spanning tree
and chord set; configurations as Z/6 seam labels; the port gauge action;
chord holonomy sector data; tree reduction (constructive classification);
face curvature and the mismatch energy; the conservation and structural
receipts of DESIGN.md sections 4 and 6.

This module contains no committed matter-table values, no descent
congruence, and no reference to expected occupancy (DESIGN.md section 10).

Boundaries: finite exact arithmetic mod 6 on the committed tables; no
physical field, charge, or particle attachment; no frozen instrument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Sequence

from oph_fpe.em import base_carrier

MOD = 6

Config = list[int]
SectorClass = tuple[int, ...]


class CarrierDefectError(ValueError):
    """A lane C6 structural or conservation receipt failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CarrierDefectError(message)


# ---------------------------------------------------------------------------
# Carrier interface: any port-graph with a spanning tree and chord set
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CarrierSpec:
    """A port-graph carrier with oriented seams, oriented faces, a spanning
    tree, and the fundamental chord cycles.

    ``face_rows[f]`` is the signed incidence row of face ``f`` over the
    seams; ``cycles[i]`` is the fundamental cycle of ``chords[i]`` as a
    signed integer vector over the seams. ``oriented_faces`` carries the
    committed vertex triples for symmetry derivation.
    """

    ports: int
    seam_left: tuple[int, ...]
    seam_right: tuple[int, ...]
    oriented_faces: tuple[tuple[int, int, int], ...]
    face_rows: tuple[tuple[int, ...], ...]
    tree_seams: tuple[int, ...]
    chords: tuple[int, ...]
    cycles: tuple[tuple[int, ...], ...]
    seams: int = field(init=False)
    faces: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "seams", len(self.seam_left))
        object.__setattr__(self, "faces", len(self.face_rows))


def base_carrier_spec() -> CarrierSpec:
    """The committed icosahedral base carrier, imported from lane C4 and
    converted to exact integer tables (cycle entries are -1, 0, +1)."""
    face_rows = tuple(
        tuple(int(x) for x in row)
        for row in base_carrier.face_incidence_matrix()
    )
    cycles = tuple(
        tuple(int(x) for x in base_carrier.fundamental_cycle(chord))
        for chord in base_carrier.CHORD_SEAMS
    )
    return CarrierSpec(
        ports=base_carrier.PORTS,
        seam_left=tuple(base_carrier.SEAM_LEFT),
        seam_right=tuple(base_carrier.SEAM_RIGHT),
        oriented_faces=tuple(base_carrier.ORIENTED_FACES),
        face_rows=face_rows,
        tree_seams=tuple(base_carrier.TREE_SEAMS),
        chords=tuple(base_carrier.CHORD_SEAMS),
        cycles=cycles,
    )


# ---------------------------------------------------------------------------
# Configurations, gauge action, sector data
# ---------------------------------------------------------------------------

def zero_config(spec: CarrierSpec) -> Config:
    return [0] * spec.seams


def gauge_move(spec: CarrierSpec, config: Sequence[int],
               gauge: Sequence[int]) -> Config:
    """``A(e) -> A(e) + g(right(e)) - g(left(e))`` mod 6: the additive form
    of the committed endpoint rechart action."""
    _require(len(config) == spec.seams, "gauge move config arity drift")
    _require(len(gauge) == spec.ports, "gauge move gauge arity drift")
    return [
        (config[e] + gauge[spec.seam_right[e]] - gauge[spec.seam_left[e]])
        % MOD
        for e in range(spec.seams)
    ]


def chord_holonomies(spec: CarrierSpec, config: Sequence[int]) -> SectorClass:
    """The conserved sector data: the 19-tuple (chord count in general) of
    chord holonomies mod 6, in committed chord order."""
    _require(len(config) == spec.seams, "holonomy config arity drift")
    return tuple(
        sum(cyc[e] * config[e] for e in range(spec.seams)) % MOD
        for cyc in spec.cycles
    )


def sector_representative(spec: CarrierSpec,
                          sector: Sequence[int]) -> Config:
    """The tree-trivial representative of a sector class: zero on tree
    seams, the class entries on the chords."""
    _require(len(sector) == len(spec.chords), "sector arity drift")
    config = zero_config(spec)
    for value, chord in zip(sector, spec.chords, strict=True):
        config[chord] = value % MOD
    return config


def tree_reduce(spec: CarrierSpec,
                config: Sequence[int]) -> tuple[Config, list[int]]:
    """The constructive classification: a gauge ``g`` with ``A - dg`` zero
    on every tree seam, returned as ``(tree_trivial_config, g)``.

    The tree-trivial configuration equals the chord holonomies on the
    chords (receipted in ``conservation_receipt``)."""
    _require(len(config) == spec.seams, "tree reduce arity drift")
    adjacency: dict[int, list[tuple[int, int]]] = {
        p: [] for p in range(spec.ports)
    }
    for e in spec.tree_seams:
        adjacency[spec.seam_left[e]].append((spec.seam_right[e], e))
        adjacency[spec.seam_right[e]].append((spec.seam_left[e], e))
    gauge: list[int | None] = [None] * spec.ports
    gauge[0] = 0
    queue = [0]
    while queue:
        node = queue.pop(0)
        base = gauge[node]
        assert base is not None
        for neighbour, seam in adjacency[node]:
            if gauge[neighbour] is not None:
                continue
            # Want (dg)(seam) = g(right) - g(left) = A(seam).
            if spec.seam_right[seam] == neighbour:
                gauge[neighbour] = (base + config[seam]) % MOD
            else:
                gauge[neighbour] = (base - config[seam]) % MOD
            queue.append(neighbour)
    _require(all(x is not None for x in gauge),
             "spanning tree does not span the ports")
    filled = [int(x) for x in gauge]  # type: ignore[arg-type]
    inverse = [(-x) % MOD for x in filled]
    reduced = gauge_move(spec, config, inverse)
    _require(all(reduced[e] == 0 for e in spec.tree_seams),
             "tree reduction left a nonzero tree seam")
    return reduced, filled


# ---------------------------------------------------------------------------
# Curvature and mismatch energy
# ---------------------------------------------------------------------------

def face_curvature(spec: CarrierSpec, config: Sequence[int]) -> list[int]:
    """``F(f) = (C A)(f)`` mod 6 on the oriented faces."""
    _require(len(config) == spec.seams, "curvature arity drift")
    return [
        sum(row[e] * config[e] for e in range(spec.seams)) % MOD
        for row in spec.face_rows
    ]


def circle_distance(x: int) -> int:
    """``rho(x) = min(x, 6 - x)``: circular distance to flatness on Z/6."""
    x %= MOD
    return min(x, MOD - x)


def mismatch_energy(spec: CarrierSpec, config: Sequence[int]) -> int:
    return sum(circle_distance(f) for f in face_curvature(spec, config))


@lru_cache(maxsize=8)
def seam_faces(spec: CarrierSpec) -> tuple[tuple[int, int, int], ...]:
    """Per seam: ``(f1, f2, s)`` with ``f1 < f2`` the two incident faces and
    ``s`` the incidence sign of ``f1`` at the seam (``f2`` carries ``-s``,
    the committed opposite-signs property, receipted)."""
    table: list[tuple[int, int, int]] = []
    for e in range(spec.seams):
        touching = [f for f in range(spec.faces) if spec.face_rows[f][e] != 0]
        _require(len(touching) == 2, "seam-face support drift")
        f1, f2 = sorted(touching)
        s = spec.face_rows[f1][e]
        _require(spec.face_rows[f2][e] == -s,
                 "seam-face opposite orientation drift")
        table.append((f1, f2, s))
    return tuple(table)


# ---------------------------------------------------------------------------
# Receipts (fail-closed)
# ---------------------------------------------------------------------------

def _rank_mod_p(rows: Sequence[Sequence[int]], p: int) -> int:
    """Exact rank of an integer matrix over GF(p) by Gauss elimination."""
    work = [[x % p for x in row] for row in rows]
    if not work:
        return 0
    n_rows, n_cols = len(work), len(work[0])
    rank = 0
    for col in range(n_cols):
        pivot = next(
            (r for r in range(rank, n_rows) if work[r][col] % p != 0), None
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inv = pow(work[rank][col], -1, p)
        work[rank] = [(x * inv) % p for x in work[rank]]
        for r in range(n_rows):
            if r != rank and work[r][col] % p != 0:
                factor = work[r][col]
                work[r] = [
                    (x - factor * y) % p
                    for x, y in zip(work[r], work[rank], strict=True)
                ]
        rank += 1
    return rank


def _cycle_boundary(spec: CarrierSpec, cycle: Sequence[int]) -> list[int]:
    """Integer boundary of a signed seam vector at every port."""
    load = [0] * spec.ports
    for e in range(spec.seams):
        load[spec.seam_right[e]] += cycle[e]
        load[spec.seam_left[e]] -= cycle[e]
    return load


def conservation_receipt(spec: CarrierSpec,
                         cycles: Sequence[Sequence[int]] | None = None
                         ) -> dict:
    """The exact gauge-conservation receipt of DESIGN.md section 4, as
    finite integer arithmetic (not sampling); fail-closed.

    ``cycles`` defaults to the committed fundamental cycles; a mutated
    cycle set (broken boundary) fails the receipt.
    """
    used = spec.cycles if cycles is None else tuple(
        tuple(c) for c in cycles
    )
    _require(len(used) == len(spec.chords), "cycle count drift")
    # boundary(cycle_c)(p) = <cycle_c, d delta_p> = 0 for all (c, p): gauge
    # moves are Z-linear combinations of unit port moves, so this enumerates
    # the full conservation identity h(A + dg) = h(A).
    for index, cyc in enumerate(used):
        _require(len(cyc) == spec.seams, "cycle arity drift")
        load = _cycle_boundary(spec, cyc)
        _require(
            all(x == 0 for x in load),
            f"cycle {index} has nonzero boundary: gauge conservation fails",
        )
    # Each fundamental cycle meets exactly its own chord, with entry +1:
    # tree-trivial representatives read their chords as holonomies.
    for index, cyc in enumerate(used):
        for j, chord in enumerate(spec.chords):
            expected = 1 if j == index else 0
            _require(cyc[chord] == expected,
                     "fundamental cycle chord-support drift")
    # Curvature is a sector invariant and classifies sectors mod 6:
    # C d = 0 mod 6 and rank(C) = chords over GF(2) and GF(3).
    for row in spec.face_rows:
        load = _cycle_boundary(spec, row)  # transpose pairing: C d columns
        _require(all(x % MOD == 0 for x in load),
                 "C compose d nonzero mod 6")
    rank2 = _rank_mod_p(spec.face_rows, 2)
    rank3 = _rank_mod_p(spec.face_rows, 3)
    _require(rank2 == len(spec.chords), "face incidence rank over GF(2) drift")
    _require(rank3 == len(spec.chords), "face incidence rank over GF(3) drift")
    return {
        "schema": "oph.sim.defect_census.conservation_receipt.v1",
        "exploratory": True,
        "evidential": False,
        "frozen": False,
        "instrument_armed": False,
        "gauge_conservation": "boundary(cycle_c) = 0 for all chords, exact",
        "checked_pairs": len(used) * spec.ports,
        "tree_trivial_reads_chords": True,
        "curvature_classifies_sectors": {
            "C_compose_d_zero_mod6": True,
            "rank_C_mod2": rank2,
            "rank_C_mod3": rank3,
            "chord_count": len(spec.chords),
        },
    }


def structural_receipt(spec: CarrierSpec) -> dict:
    """Carrier pin for census receipts; fail-closed on arity drift."""
    _require(len(spec.seam_left) == len(spec.seam_right),
             "seam table arity drift")
    _require(len(spec.tree_seams) + len(spec.chords) == spec.seams,
             "tree/chord partition drift")
    _require(
        sorted(spec.tree_seams + spec.chords) == list(range(spec.seams)),
        "tree/chord cover drift",
    )
    seam_faces(spec)
    return {
        "ports": spec.ports,
        "seams": spec.seams,
        "faces": spec.faces,
        "tree_seams": list(spec.tree_seams),
        "chords": list(spec.chords),
        "chord_count": len(spec.chords),
    }


def gauge_covariance_receipt(
    spec: CarrierSpec,
    repair: Callable[[CarrierSpec, Sequence[int]], tuple[Config, list]],
    configs: Sequence[Sequence[int]],
    gauges: Sequence[Sequence[int]],
) -> bool:
    """DESIGN.md section 6 item 2 on seeded samples: identical move traces,
    ``repair(A + dg) = repair(A) + dg``, and equal repaired sectors.

    Returns True when every sample passes; False on the first failure (the
    mutation guard asserts False for the raw-label mutant rule)."""
    for config, gauge in zip(configs, gauges, strict=True):
        base_fixed, base_trace = repair(spec, config)
        moved = gauge_move(spec, config, gauge)
        moved_fixed, moved_trace = repair(spec, moved)
        if base_trace != moved_trace:
            return False
        if moved_fixed != gauge_move(spec, base_fixed, gauge):
            return False
        if chord_holonomies(spec, moved_fixed) != chord_holonomies(
            spec, base_fixed
        ):
            return False
    return True
