"""Family readout v2 for the Z/6 defect census (lane D2, exploratory,
non-evidential).

Design record: ``oph_fpe/defects/DESIGN_V2.md``, fixed before this module.
Committed C6 modules are imported read-only; the committed corpus fixes the
label-lattice shape (triality = a 3-family occupancy character, duality = a
2-family occupancy character, both distinct from the charge; sources quoted
in the design record), and the carrier realization here is the declared
convention ``tait_antipodal_family_readout.v2``:

* ``q`` = total chord holonomy mod 6 (unchanged from v1);
* ``t`` = color-index weighted chord holonomy mod 3, colors from the
  canonical face-rainbow 3-coloring (Tait coloring of the dual graph),
  lexicographically minimal in committed seam order;
* ``d`` = paired-chord indicator weighted chord holonomy mod 2, pairing
  from the derived antipodal seam involution.

Receipts (fail-closed, exact): coloring rainbow and class sizes; antipodal
uniqueness, involution, automorphism, and rotation-group non-membership;
gauge invariance of all three characters as zero port boundary over the
integers at every port; the A5 fixed-character theorem (fixed subspace zero
over GF(3) and GF(2)); the image subgroup with the vacuity detector and the
off-diagonal witness. The A5 quotient object is the per-class orbit label
multiset (design record section 2c).

Boundaries: finite exact arithmetic on the committed tables; a label is a
declared convention, not a committed identification; no physical particle
claim; no instrument is frozen or armed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from oph_fpe.defects.z6_a5_action import act_on_sector
from oph_fpe.defects.z6_carrier_defects import (
    MOD,
    CarrierDefectError,
    CarrierSpec,
    SectorClass,
    chord_holonomies,
    gauge_move,
    sector_representative,
    zero_config,
)

FAMILY_READOUT = "tait_antipodal_family_readout.v2"

Label = tuple[int, int, int]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CarrierDefectError(message)


# ---------------------------------------------------------------------------
# Antipodal structure, derived from committed adjacency
# ---------------------------------------------------------------------------

def _adjacency(spec: CarrierSpec) -> list[set[int]]:
    table: list[set[int]] = [set() for _ in range(spec.ports)]
    for l, r in zip(spec.seam_left, spec.seam_right, strict=True):
        table[l].add(r)
        table[r].add(l)
    return table


def antipodal_port_map(spec: CarrierSpec) -> tuple[int, ...]:
    """Per port: the unique port at graph distance three (receipted
    unique); a fixed-point-free involution (receipted)."""
    adjacency = _adjacency(spec)
    result: list[int] = []
    for start in range(spec.ports):
        dist = {start: 0}
        frontier = [start]
        while frontier:
            step: list[int] = []
            for node in frontier:
                for other in adjacency[node]:
                    if other not in dist:
                        dist[other] = dist[node] + 1
                        step.append(other)
            frontier = step
        far = [p for p, x in dist.items() if x == 3]
        _require(len(dist) == spec.ports, "port graph not connected")
        _require(len(far) == 1, f"port {start}: distance-3 set not unique")
        result.append(far[0])
    _require(
        all(result[result[p]] == p and result[p] != p
            for p in range(spec.ports)),
        "antipodal port map not a fixed-point-free involution",
    )
    return tuple(result)


def antipodal_seam_involution(spec: CarrierSpec) -> tuple[int, ...]:
    """Per seam: the seam whose endpoint set is the antipodal image;
    receipted fixed-point-free involution with fifteen orbits on the
    committed carrier (seams / 2 in general)."""
    ports = antipodal_port_map(spec)
    seam_index = {
        frozenset((spec.seam_left[e], spec.seam_right[e])): e
        for e in range(spec.seams)
    }
    result: list[int] = []
    for e in range(spec.seams):
        key = frozenset(
            (ports[spec.seam_left[e]], ports[spec.seam_right[e]])
        )
        _require(key in seam_index, f"seam {e}: antipodal image not a seam")
        result.append(seam_index[key])
    _require(
        all(result[result[e]] == e and result[e] != e
            for e in range(spec.seams)),
        "antipodal seam map not a fixed-point-free involution",
    )
    return tuple(result)


def antipodal_receipt(spec: CarrierSpec,
                      rotations: Sequence[Sequence[int]]) -> dict:
    """Structure receipt: the antipodal port map is a graph automorphism
    and is not an element of the derived rotation group."""
    ports = antipodal_port_map(spec)
    edges = {
        frozenset((spec.seam_left[e], spec.seam_right[e]))
        for e in range(spec.seams)
    }
    _require(
        all(frozenset((ports[l], ports[r])) in edges
            for l, r in zip(spec.seam_left, spec.seam_right, strict=True)),
        "antipodal port map not a graph automorphism",
    )
    in_rotation_group = tuple(ports) in {tuple(p) for p in rotations}
    _require(not in_rotation_group,
             "antipodal port map inside the rotation group")
    pairs = sorted(
        {tuple(sorted((e, x)))
         for e, x in enumerate(antipodal_seam_involution(spec))}
    )
    return {
        "schema": "oph.sim.defect_census.antipodal_receipt.v2",
        "exploratory": True,
        "evidential": False,
        "port_map": list(ports),
        "seam_pair_count": len(pairs),
        "graph_automorphism": True,
        "in_rotation_group": False,
    }


# ---------------------------------------------------------------------------
# Canonical face-rainbow coloring (Tait coloring of the dual graph)
# ---------------------------------------------------------------------------

def _face_seams(spec: CarrierSpec) -> list[list[int]]:
    table: list[list[int]] = []
    for f in range(spec.faces):
        seams = [e for e in range(spec.seams) if spec.face_rows[f][e] != 0]
        _require(len(seams) == 3, "face seam support drift")
        table.append(seams)
    return table


def canonical_tait_coloring(spec: CarrierSpec) -> tuple[int, ...]:
    """The lexicographically minimal coloring ``seams -> {0, 1, 2}`` whose
    three colors are distinct on every face: deterministic backtracking in
    committed seam order, smallest feasible color first."""
    faces = _face_seams(spec)
    seam_faces: list[list[int]] = [[] for _ in range(spec.seams)]
    for f, seams in enumerate(faces):
        for e in seams:
            seam_faces[e].append(f)
    coloring = [-1] * spec.seams

    def feasible(seam: int, color: int) -> bool:
        for f in seam_faces[seam]:
            for other in faces[f]:
                if other != seam and coloring[other] == color:
                    return False
        return True

    def extend(seam: int) -> bool:
        if seam == spec.seams:
            return True
        for color in range(3):
            if feasible(seam, color):
                coloring[seam] = color
                if extend(seam + 1):
                    return True
                coloring[seam] = -1
        return False

    _require(extend(0), "no face-rainbow 3-coloring exists")
    return tuple(coloring)


def tait_receipt(spec: CarrierSpec, coloring: Sequence[int]) -> dict:
    """Fail-closed coloring receipt: rainbow on every face, three classes
    of equal size (perfect matchings of the dual graph)."""
    _require(len(coloring) == spec.seams, "coloring arity drift")
    _require(all(c in (0, 1, 2) for c in coloring), "coloring value drift")
    for f, seams in enumerate(_face_seams(spec)):
        _require(len({coloring[e] for e in seams}) == 3,
                 f"face {f} not rainbow")
    sizes = [sum(1 for c in coloring if c == v) for v in range(3)]
    _require(len(set(sizes)) == 1, "coloring class sizes unequal")
    return {
        "schema": "oph.sim.defect_census.tait_receipt.v2",
        "exploratory": True,
        "evidential": False,
        "coloring": list(coloring),
        "class_sizes": sizes,
        "rainbow_faces": spec.faces,
    }


# ---------------------------------------------------------------------------
# Chord weights and the v2 label map
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FamilyWeights:
    """Chord coefficient vectors of the three v2 characters: ``w6`` for the
    Z/6 charge, ``w3`` for triality, ``w2`` for duality."""

    w6: tuple[int, ...]
    w3: tuple[int, ...]
    w2: tuple[int, ...]


def family_weights(spec: CarrierSpec,
                   coloring: Sequence[int] | None = None,
                   involution: Sequence[int] | None = None) -> FamilyWeights:
    coloring = (
        canonical_tait_coloring(spec) if coloring is None else coloring
    )
    involution = (
        antipodal_seam_involution(spec) if involution is None
        else involution
    )
    _require(len(coloring) == spec.seams, "coloring arity drift")
    _require(len(involution) == spec.seams, "involution arity drift")
    chords = set(spec.chords)
    return FamilyWeights(
        w6=tuple(1 for _ in spec.chords),
        w3=tuple(coloring[c] % 3 for c in spec.chords),
        w2=tuple(
            1 if involution[c] in chords else 0 for c in spec.chords
        ),
    )


def family_label(weights: FamilyWeights, sector: Sequence[int]) -> Label:
    """``(q, t, d)``: the three declared characters on a sector class."""
    _require(len(sector) == len(weights.w6), "sector arity drift")
    q = sum(w * h for w, h in zip(weights.w6, sector, strict=True)) % MOD
    t = sum(w * h for w, h in zip(weights.w3, sector, strict=True)) % 3
    d = sum(w * h for w, h in zip(weights.w2, sector, strict=True)) % 2
    return (q, t, d)


def diagonal_weights(spec: CarrierSpec) -> FamilyWeights:
    """The v1 readout reimplemented as chord weights (all-ones in every
    slot): ``t = q mod 3`` and ``d = q mod 2``. Mutation-guard input for
    the vacuity detector; not part of the v2 pipeline."""
    ones = tuple(1 for _ in spec.chords)
    return FamilyWeights(w6=ones, w3=ones, w2=ones)


# ---------------------------------------------------------------------------
# Gauge-invariance receipts (exact over generators)
# ---------------------------------------------------------------------------

def seam_weight_vector(spec: CarrierSpec,
                       chord_coeffs: Sequence[int]) -> list[int]:
    """The integer seam weight vector ``sum_c coeff_c * cycle_c`` of a
    chord-coefficient character."""
    _require(len(chord_coeffs) == len(spec.chords), "coefficient arity")
    vector = [0] * spec.seams
    for coeff, cycle in zip(chord_coeffs, spec.cycles, strict=True):
        for e in range(spec.seams):
            vector[e] += coeff * cycle[e]
    return vector


def port_boundary(spec: CarrierSpec, vector: Sequence[int]) -> list[int]:
    """Integer port boundary of a seam weight vector; zero at every port
    is exact gauge invariance of the character (gauge moves are Z-linear
    combinations of unit port moves)."""
    load = [0] * spec.ports
    for e in range(spec.seams):
        load[spec.seam_right[e]] += vector[e]
        load[spec.seam_left[e]] -= vector[e]
    return load


def character_is_gauge_invariant(spec: CarrierSpec,
                                 seam_vector: Sequence[int],
                                 modulus: int) -> bool:
    """Whether a raw seam weight vector has zero port boundary modulo the
    character modulus; the all-seam colored-sum mutant returns False."""
    return all(x % modulus == 0 for x in port_boundary(spec, seam_vector))


def gauge_invariance_receipt(spec: CarrierSpec,
                             weights: FamilyWeights) -> dict:
    """Fail-closed exact receipt: the three character weight vectors have
    zero integer port boundary at every port, and the seam-level label of
    every unit port gauge move is the zero label."""
    checks = {}
    for name, coeffs, modulus in (
        ("q", weights.w6, MOD), ("t", weights.w3, 3), ("d", weights.w2, 2),
    ):
        vector = seam_weight_vector(spec, coeffs)
        load = port_boundary(spec, vector)
        _require(all(x == 0 for x in load),
                 f"character {name}: nonzero port boundary")
        checks[name] = {
            "boundary_zero_ports": spec.ports,
            "integer_boundary": "zero at every port, exact",
        }
    for p in range(spec.ports):
        gauge = [0] * spec.ports
        gauge[p] = 1
        moved = gauge_move(spec, zero_config(spec), gauge)
        label = family_label(weights, chord_holonomies(spec, moved))
        _require(label == (0, 0, 0),
                 f"unit port move {p}: nonzero label")
    return {
        "schema": "oph.sim.defect_census.gauge_invariance_receipt.v2",
        "exploratory": True,
        "evidential": False,
        "characters": checks,
        "unit_port_moves_zero_label": spec.ports,
        "statement": (
            "zero integer port boundary at every port for every character;"
            " gauge moves are Z-linear combinations of unit port moves, so"
            " the labels are invariant for every configuration and every"
            " gauge move"
        ),
    }


def all_seam_colored_vector(spec: CarrierSpec,
                            coloring: Sequence[int]) -> list[int]:
    """The gauge-variance mutant of the design record: color-index weights
    on all seams instead of chord-restricted holonomy weights."""
    _require(len(coloring) == spec.seams, "coloring arity drift")
    return [coloring[e] % 3 for e in range(spec.seams)]


# ---------------------------------------------------------------------------
# A5 receipts: fixed characters and the orbit label multiset
# ---------------------------------------------------------------------------

def sector_action_matrix(spec: CarrierSpec,
                         perm: Sequence[int]) -> list[list[int]]:
    """The matrix of the induced sector action mod 6 (columns are the
    images of the unit sectors)."""
    n = len(spec.chords)
    columns = []
    for i in range(n):
        unit = [0] * n
        unit[i] = 1
        columns.append(act_on_sector(spec, perm, unit))
    return [[columns[c][r] for c in range(n)] for r in range(n)]


def _rank_mod_p(rows: list[list[int]], p: int) -> int:
    work = [[x % p for x in row] for row in rows]
    if not work:
        return 0
    rank, cols = 0, len(work[0])
    for col in range(cols):
        pivot = next(
            (r for r in range(rank, len(work)) if work[r][col] % p != 0),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inv = pow(work[rank][col], -1, p)
        work[rank] = [(x * inv) % p for x in work[rank]]
        for r in range(len(work)):
            if r != rank and work[r][col] % p != 0:
                factor = work[r][col]
                work[r] = [
                    (x - factor * y) % p
                    for x, y in zip(work[r], work[rank], strict=True)
                ]
        rank += 1
    return rank


def a5_fixed_character_receipt(spec: CarrierSpec,
                               rotations: Sequence[Sequence[int]]) -> dict:
    """Fail-closed theorem receipt: the fixed subspace of the character
    space under the induced A5 action is zero over GF(3) and GF(2), by
    exact rank computation on the stacked ``(M^T - I)`` blocks over all
    rotations. Consequence: no nonzero A5-invariant triality or duality
    character on sector space exists, and the declared A5 quotient object
    is the per-class orbit label multiset."""
    n = len(spec.chords)
    matrices = [sector_action_matrix(spec, perm) for perm in rotations]
    dims = {}
    for p in (3, 2):
        stacked: list[list[int]] = []
        for matrix in matrices:
            for r in range(n):
                stacked.append(
                    [(matrix[c][r] - (1 if c == r else 0)) % p
                     for c in range(n)]
                )
        dims[p] = n - _rank_mod_p(stacked, p)
    _require(dims[3] == 0, "GF(3) fixed character space nonzero")
    _require(dims[2] == 0, "GF(2) fixed character space nonzero")
    return {
        "schema": "oph.sim.defect_census.a5_fixed_character_receipt.v2",
        "exploratory": True,
        "evidential": False,
        "rotations": len(matrices),
        "character_space_dim": n,
        "fixed_dim_mod3": dims[3],
        "fixed_dim_mod2": dims[2],
        "statement": (
            "no nonzero A5-invariant Z/3 or Z/2 character on sector space"
            " exists; the declared quotient object is the per-class orbit"
            " label multiset under the fixed convention"
        ),
    }


def orbit_label_multiset(spec: CarrierSpec,
                         rotations: Sequence[Sequence[int]],
                         weights: FamilyWeights,
                         sector: Sequence[int]) -> list[list[int]]:
    """The multiset of v2 labels over the distinct sectors of the A5
    orbit, sorted; identical from any orbit member."""
    orbit = {act_on_sector(spec, perm, sector) for perm in rotations}
    return sorted(list(family_label(weights, s)) for s in orbit)


# ---------------------------------------------------------------------------
# Image subgroup, vacuity detector, non-vacuity receipt
# ---------------------------------------------------------------------------

def label_image_subgroup(weights: FamilyWeights) -> set[Label]:
    """The exact image of the label homomorphism: the subgroup of
    W = Z/6 x Z/3 x Z/2 generated by the unit-chord labels."""
    n = len(weights.w6)
    generators = []
    for i in range(n):
        unit = [0] * n
        unit[i] = 1
        generators.append(family_label(weights, unit))
    image: set[Label] = {(0, 0, 0)}
    frontier: list[Label] = [(0, 0, 0)]
    while frontier:
        q, t, d = frontier.pop()
        for gq, gt, gd in generators:
            candidate = ((q + gq) % MOD, (t + gt) % 3, (d + gd) % 2)
            if candidate not in image:
                image.add(candidate)
                frontier.append(candidate)
    return image


def crt_diagonal() -> set[Label]:
    return {(q, q % 3, q % 2) for q in range(MOD)}


def image_is_diagonal(image: set[Label]) -> bool:
    """The vacuity detector: whether a readout image lies inside the CRT
    diagonal (equal to the committed descent kernel), which forces the
    descent check and blocks the control label structurally."""
    return image <= crt_diagonal()


def non_vacuity_receipt(spec: CarrierSpec, weights: FamilyWeights) -> dict:
    """Fail-closed receipt for the shipped v2 weights: the image subgroup
    is not contained in the diagonal, an off-diagonal unit-chord witness
    exists, and the committed control label is reachable. A diagonal
    image raises: the design record mandates report-and-stop over a
    vacuous comparison."""
    image = label_image_subgroup(weights)
    diagonal = crt_diagonal()
    _require(not image_is_diagonal(image),
             "v2 readout image inside the CRT diagonal: vacuity finding,"
             " comparison stopped")
    witness = None
    for i in range(len(weights.w6)):
        unit = [0] * len(weights.w6)
        unit[i] = 1
        label = family_label(weights, unit)
        if label not in diagonal:
            witness = {
                "chord_index": i,
                "seam": spec.chords[i],
                "label_qtd": list(label),
                "center_char": (2 * label[1] + 3 * label[2] + label[0])
                % MOD,
            }
            break
    _require(witness is not None, "no off-diagonal unit-chord witness")
    return {
        "schema": "oph.sim.defect_census.non_vacuity_receipt.v2",
        "exploratory": True,
        "evidential": False,
        "image_size": len(image),
        "lattice_size": 36,
        "image_is_diagonal": False,
        "image_full_lattice": len(image) == 36,
        "off_diagonal_witness": witness,
        "control_label_reachable": (1, 0, 0) in image,
        "statement": (
            "the descent congruence and the control label are live checks"
            " under the v2 readout; the v1 diagonal reimplementation"
            " triggers the detector (test suite guard)"
        ),
    }


# ---------------------------------------------------------------------------
# Assembled readout
# ---------------------------------------------------------------------------

def build_readout(spec: CarrierSpec,
                  rotations: Sequence[Sequence[int]]) -> dict:
    """Construct the v2 readout and all structure receipts; fail-closed.
    Returns ``{"weights": FamilyWeights, "receipts": dict}``."""
    coloring = canonical_tait_coloring(spec)
    involution = antipodal_seam_involution(spec)
    weights = family_weights(spec, coloring, involution)
    mutant = all_seam_colored_vector(spec, coloring)
    _require(
        not character_is_gauge_invariant(spec, mutant, 3),
        "gauge-variance mutant control unexpectedly invariant",
    )
    receipts = {
        "readout": FAMILY_READOUT,
        "tait": tait_receipt(spec, coloring),
        "antipodal": antipodal_receipt(spec, rotations),
        "gauge_invariance": gauge_invariance_receipt(spec, weights),
        "a5_fixed_characters": a5_fixed_character_receipt(spec, rotations),
        "non_vacuity": non_vacuity_receipt(spec, weights),
        "chord_weights": {
            "w6": list(weights.w6),
            "w3": list(weights.w3),
            "w2": list(weights.w2),
        },
        "mutant_control": {
            "all_seam_colored_vector_invariant": False,
            "note": (
                "the raw all-seam colored sum has nonzero port boundary"
                " modulo three and is rejected by the invariance receipt"
            ),
        },
    }
    return {"weights": weights, "receipts": receipts}


def sector_label_v2(spec: CarrierSpec, weights: FamilyWeights,
                    sector: SectorClass) -> Label:
    """Convenience wrapper: the v2 label of a sector class, computed on
    the class tuple directly (the characters are chord functionals)."""
    _require(len(sector) == len(spec.chords), "sector arity drift")
    return family_label(weights, sector)


def config_label_v2(spec: CarrierSpec, weights: FamilyWeights,
                    config: Sequence[int]) -> Label:
    """The v2 label of a configuration through its chord holonomies."""
    return family_label(weights, chord_holonomies(spec, config))
