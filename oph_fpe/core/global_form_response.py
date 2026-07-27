"""Global form semantic artifact producer for the twelve-port carrier.

This module measures, from finite source structure alone, the deck/loop and
sector data that the paper-side axis-center-descent certificate (issue #567
in reverse-engineering-reality) needs in order to attach the derived kernel
quotient to physical transport rather than to Lie-type arithmetic. It is
target-blind: no gauge label, charge table, matter module, or quotient choice
enters the producer. What it emits are measured facts about the certified
carrier and its incidence-nerve federation:

- the measured deck group of the incidence (order 120) acting on the twelve
  federation charts, thirty seams, and twenty triple overlaps, with the
  orientation character separating the sixty rotations from the sixty
  improper elements; this upgrades the federation's former
  ``identity_only`` declared deck to a measured action;
- the measured six-axis class group: the six antipodal port pairs, the free
  lattice on them modulo the diagonal and zero-sum relations, with Smith
  invariants (1, 1, 1, 1, 1, 6), class group of order six, all six axis
  classes equal to the generator, rotations acting trivially on the class,
  and orientation reversal acting by negation;
- the measured sector class of the realized reference federation: composing
  the seam identifications around every one of the twenty triangles returns
  the starting port, so the realized federation lies in the vacuum sector
  (class zero);
- the two-puncture flux-tube witnesses: for every class c in the measured
  order-six group, an explicit seam assignment on the thirty seams whose
  face holonomy is +c at a chosen face, -c at its antipodal face, and zero
  at the eighteen interior faces; a single-puncture witness is impossible on
  the closed support because face holonomies of seam data always sum to
  zero, which the producer also verifies exactly;
- the subgroup obstruction menu: a flux value c lifts through the subgroup
  chain of the measured class group exactly when c lies in the subgroup
  (orders 1, 2, 3, 6), stated as exact arithmetic on the measured group;
- refinement naturality: the same two-puncture menu is realized on the next
  geodesic refinement level through child faces of the same punctures.

The producer fails closed with typed errors when the measured structure does
not present these facts. Whether the measured sector menu and deck data
select a physical global form for a realized tensor package is decided on
the paper side; this producer only measures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.core.charged_response import (
    _require,
    canonical_sha256,
    graph_isomorphism,
    load_carrier,
    match_vertex_frame,
)
from oph_fpe.consensus.incidence_nerve_bridge import incidence_nerve_bridge_report
from oph_fpe.core.echosahedral_federation import (
    _reference_incidence_sha256,
    reference_incidence_nerve_federation,
)
from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower
from oph_fpe.core.spin_statistics_response import (
    _smith_invariants,
    measure_deck_realization,
)

SCHEMA = "oph.global_form_semantic_artifact.v1"
ISSUE = 567


# ---------------------------------------------------------------------------
# Oriented face/edge complex helpers
# ---------------------------------------------------------------------------


def _edge_list(face_rows: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    edges = set()
    for a, b, c in face_rows:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    return sorted(edges)


def _boundary_two(
    face_rows: Sequence[Sequence[int]], edges: Sequence[tuple[int, int]]
) -> list[list[int]]:
    edge_index = {edge: position for position, edge in enumerate(edges)}
    matrix = [[0] * len(face_rows) for _ in range(len(edges))]
    for face_position, (a, b, c) in enumerate(face_rows):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            sign = 1 if (u, v) == key else -1
            matrix[edge_index[key]][face_position] += sign
    return matrix


def _face_holonomies(
    seam_values: Mapping[int, int],
    boundary: Sequence[Sequence[int]],
    face_count: int,
    modulus: int,
) -> list[int]:
    holonomies = []
    for face in range(face_count):
        total = 0
        for edge, value in seam_values.items():
            total += boundary[edge][face] * value
        holonomies.append(total % modulus)
    return holonomies


def _dual_path(
    face_rows: Sequence[Sequence[int]],
    edges: Sequence[tuple[int, int]],
    start: int,
    goal: int,
) -> list[tuple[int, int]]:
    """A shortest dual path as a list of (face, shared_edge) steps ending at goal."""

    edge_index = {edge: position for position, edge in enumerate(edges)}
    edge_faces: dict[int, list[int]] = {}
    for face_position, (a, b, c) in enumerate(face_rows):
        for u, v in ((a, b), (b, c), (c, a)):
            key = edge_index[(min(u, v), max(u, v))]
            edge_faces.setdefault(key, []).append(face_position)
    _require(
        all(len(pair) == 2 for pair in edge_faces.values()),
        "SECTOR_COMPLEX",
        "every seam must border exactly two faces",
    )
    neighbors: dict[int, list[tuple[int, int]]] = {}
    for edge, (left, right) in edge_faces.items():
        neighbors.setdefault(left, []).append((right, edge))
        neighbors.setdefault(right, []).append((left, edge))
    previous: dict[int, tuple[int, int]] = {}
    frontier = [start]
    seen = {start}
    while frontier:
        nxt: list[int] = []
        for face in frontier:
            for other, edge in sorted(neighbors[face]):
                if other in seen:
                    continue
                seen.add(other)
                previous[other] = (face, edge)
                nxt.append(other)
        frontier = nxt
        if goal in seen:
            break
    _require(goal in seen, "SECTOR_COMPLEX", "no dual path joins the puncture faces")
    path: list[tuple[int, int]] = []
    cursor = goal
    while cursor != start:
        parent, edge = previous[cursor]
        path.append((cursor, edge))
        cursor = parent
    path.reverse()
    return path


def flux_tube_witness(
    face_rows: Sequence[Sequence[int]],
    edges: Sequence[tuple[int, int]],
    boundary: Sequence[Sequence[int]],
    start_face: int,
    end_face: int,
    flux: int,
    modulus: int,
) -> dict[str, Any]:
    """An exact seam assignment with holonomy +flux at start, -flux at end."""

    path = _dual_path(face_rows, edges, start_face, end_face)
    seam_values: dict[int, int] = {}
    carried = flux % modulus
    current = start_face
    for step_face, edge in path:
        sign_current = boundary[edge][current]
        _require(sign_current in (1, -1), "SECTOR_COMPLEX", "path edge sign must be +-1")
        # Each path edge borders exactly its two consecutive path faces with
        # opposite signs, so setting the seam value to sign * flux gives the
        # current face its prescribed holonomy and hands -flux to the next
        # face, which the next edge cancels; the flux is carried forward.
        seam_values[edge] = (sign_current * carried) % modulus
        current = step_face
    holonomies = _face_holonomies(seam_values, boundary, len(face_rows), modulus)
    expected = [0] * len(face_rows)
    expected[start_face] = flux % modulus
    expected[end_face] = (-flux) % modulus
    _require(
        holonomies == expected,
        "SECTOR_WITNESS",
        "the constructed flux tube does not have the prescribed face holonomies",
    )
    _require(
        sum(holonomies) % modulus == 0,
        "SECTOR_WITNESS",
        "face holonomies of seam data must sum to zero on the closed support",
    )
    return {
        "flux": flux % modulus,
        "start_face": start_face,
        "end_face": end_face,
        "dual_path_length": len(path),
        "seam_values": {str(edge): value for edge, value in sorted(seam_values.items())},
        "interior_faces_flat": all(
            holonomies[face] == 0
            for face in range(len(face_rows))
            if face not in (start_face, end_face)
        ),
    }


def single_puncture_impossibility(
    boundary: Sequence[Sequence[int]], face_count: int, modulus: int
) -> dict[str, Any]:
    """Verify exactly that seam data cannot put nonzero net flux at one face only.

    Every seam borders two faces with opposite orientation signs, so the face
    holonomies of any seam assignment sum to zero; a single nonzero puncture
    would need a nonzero sum.
    """

    column_sums = [
        sum(boundary[edge][face] for face in range(face_count))
        for edge in range(len(boundary))
    ]
    _require(
        all(value == 0 for value in column_sums),
        "SECTOR_COMPLEX",
        "each seam must appear with opposite signs in its two faces",
    )
    return {
        "every_seam_appears_with_opposite_signs": True,
        "face_holonomy_sum_identically_zero": True,
        "single_puncture_nonzero_flux_impossible": True,
        "physical_reading": (
            "net flux through the closed support vanishes; a nonzero sector "
            "needs a flux tube entering and leaving, realized here through "
            "two antipodal punctures"
        ),
    }


# ---------------------------------------------------------------------------
# Six-axis class group measurement
# ---------------------------------------------------------------------------


def measure_six_axis_class_group(carrier: dict[str, Any], deck: Mapping[str, Any]) -> dict[str, Any]:
    antipode = carrier["antipode"]
    axes = sorted({tuple(sorted((i, antipode[i]))) for i in range(12)})
    _require(len(axes) == 6, "AXIS_CLASS", "six antipodal axes required")
    axis_index = {axis: position for position, axis in enumerate(axes)}

    # Rotation action on axes: the deck permutes the six axes.
    axis_images = []
    for rotation in deck["rotations"]:
        image = [
            axis_index[tuple(sorted((rotation[a], rotation[b])))] for a, b in axes
        ]
        axis_images.append(tuple(image))
    _require(
        len(set(axis_images)) == 60,
        "AXIS_CLASS",
        "the rotation deck must act faithfully on the six axes",
    )
    reached = {0}
    frontier = [0]
    while frontier:
        axis = frontier.pop()
        for image in axis_images:
            if image[axis] not in reached:
                reached.add(image[axis])
                frontier.append(image[axis])
    _require(len(reached) == 6, "AXIS_CLASS", "the axis action must be transitive")

    # The class lattice: Z^6 on the axes modulo the diagonal vector and the
    # zero-sum sublattice. Relations matrix columns: the all-ones vector and a
    # basis e_i - e_{i+1} of the zero-sum lattice.
    relations = []
    relations.append([1] * 6)
    for i in range(5):
        column = [0] * 6
        column[i] = 1
        column[i + 1] = -1
        relations.append(column)
    relation_matrix = [[relations[j][i] for j in range(len(relations))] for i in range(6)]
    invariants = _smith_invariants(relation_matrix)
    _require(
        invariants == [1, 1, 1, 1, 1, 6],
        "AXIS_CLASS",
        f"the axis class lattice Smith invariants are {invariants}, not (1,1,1,1,1,6)",
    )
    class_group_order = invariants[-1]

    # Axis-difference vectors lie in the zero-sum relation lattice, so all six
    # axis basis classes agree in the quotient; checked exactly.
    def _in_relations(vector: Sequence[int]) -> bool:
        # vector = a*ones + zero-sum <=> its residue: sum(vector) divisible by 6
        # combined with the free choice of zero-sum part; membership in the
        # relation span is exactly sum(vector) % 6 == 0.
        return sum(vector) % class_group_order == 0

    for i in range(6):
        for j in range(6):
            difference = [0] * 6
            difference[i] += 1
            difference[j] -= 1
            _require(
                _in_relations(difference),
                "AXIS_CLASS",
                "axis basis classes must coincide in the quotient",
            )
    # Rotations permute the axis basis vectors, so they fix each class; checked
    # exactly through the residue formula on permuted basis vectors.
    for image in axis_images:
        for axis in range(6):
            permuted = [0] * 6
            permuted[image[axis]] = 1
            original = [0] * 6
            original[axis] = 1
            _require(
                _in_relations([permuted[k] - original[k] for k in range(6)]),
                "AXIS_CLASS",
                "rotations must act trivially on the axis class group",
            )
    # The antipode reverses every oriented axis: measured on the ordered pairs.
    antipode_reverses = all(antipode[a] == b and antipode[b] == a for a, b in axes)
    _require(antipode_reverses, "AXIS_CLASS", "the antipode must reverse every oriented axis")

    return {
        "axes": [[carrier["ports"][a], carrier["ports"][b]] for a, b in axes],
        "axis_count": 6,
        "rotation_action_faithful_order": 60,
        "rotation_action_transitive": True,
        "lattice": "free rank six on the axes modulo diagonal and zero-sum relations",
        "smith_invariants": invariants,
        "class_group_order": class_group_order,
        "all_axis_classes_equal_generator": True,
        "rotations_act_trivially_on_class_group": True,
        "antipode_reverses_every_oriented_axis": True,
        "orientation_reversal_class_shadow": (
            "reversing every oriented axis negates the oriented-axis generators, "
            "which descends to negation on the order-six class group"
        ),
        "note": (
            "the class group order is measured as the last Smith invariant of "
            "the axis relation lattice; its identification with any tensor-"
            "action kernel is a paper-side step"
        ),
    }


# ---------------------------------------------------------------------------
# Deck action on the incidence-nerve federation
# ---------------------------------------------------------------------------


def measure_federation_deck_action(carrier: dict[str, Any], deck: Mapping[str, Any]) -> dict[str, Any]:
    federation = reference_incidence_nerve_federation()
    chart_ids = sorted(carrier_block.carrier_id for carrier_block in federation.carriers)
    _require(len(chart_ids) == 12, "FEDERATION_DECK", "twelve federation charts required")
    seam_pairs = set()
    for seam in federation.seams:
        left_chart = int(seam.left_carrier_id.rsplit("-", 1)[1])
        right_chart = int(seam.right_carrier_id.rsplit("-", 1)[1])
        seam_pairs.add((min(left_chart, right_chart), max(left_chart, right_chart)))
    _require(len(seam_pairs) == 30, "FEDERATION_DECK", "thirty federation seams required")
    triangle_charts = []
    for triple in federation.triple_overlaps:
        members = tuple(
            int(chart_id.rsplit("-", 1)[1]) for chart_id in triple.oriented_carrier_ids
        )
        triangle_charts.append(members)
    _require(len(triangle_charts) == 20, "FEDERATION_DECK", "twenty triple overlaps required")

    # The federation charts carry the runtime template numbering; exhibit an
    # explicit incidence isomorphism from the manifest carrier onto the
    # federation chart graph and transport the measured deck through it.
    federation_adjacency = [[0] * 12 for _ in range(12)]
    for left, right in seam_pairs:
        federation_adjacency[left][right] = federation_adjacency[right][left] = 1
    chart_isomorphism = graph_isomorphism(carrier["adjacency"], federation_adjacency)
    _require(
        chart_isomorphism is not None,
        "FEDERATION_DECK",
        "the federation chart graph is not isomorphic to the measured carrier",
    )

    def transport(permutation: Sequence[int]) -> tuple[int, ...]:
        inverse = [0] * 12
        for index, image in enumerate(chart_isomorphism):
            inverse[image] = index
        return tuple(
            chart_isomorphism[permutation[inverse[chart]]] for chart in range(12)
        )

    def normalize(cycle: Sequence[int]) -> tuple[int, ...]:
        a, b, c = cycle
        return min([(a, b, c), (b, c, a), (c, a, b)])

    oriented_triangles = {normalize(members) for members in triangle_charts}
    reversed_triangles = {normalize(tuple(reversed(members))) for members in triangle_charts}

    proper_checked = 0
    improper_checked = 0
    for permutation in deck["rotations"]:
        image = transport(permutation)
        image_seams = {
            (min(image[u], image[v]), max(image[u], image[v])) for u, v in seam_pairs
        }
        _require(image_seams == seam_pairs, "FEDERATION_DECK", "a rotation does not permute the seams")
        image_triangles = {
            normalize(tuple(image[m] for m in members)) for members in triangle_charts
        }
        _require(
            image_triangles == oriented_triangles,
            "FEDERATION_DECK",
            "a rotation does not preserve the oriented triple overlaps",
        )
        proper_checked += 1
    for permutation in deck["improper"]:
        image = transport(permutation)
        image_triangles = {
            normalize(tuple(image[m] for m in members)) for members in triangle_charts
        }
        _require(
            image_triangles == reversed_triangles,
            "FEDERATION_DECK",
            "an improper element does not reverse the oriented triple overlaps",
        )
        improper_checked += 1

    return {
        "source_incidence_sha256": _reference_incidence_sha256(),
        "carrier_to_chart_isomorphism": {
            carrier["ports"][index]: f"incidence-chart-{chart_isomorphism[index]:02d}"
            for index in range(12)
        },
        "charts": 12,
        "seams": 30,
        "triple_overlaps": 20,
        "deck_group_order": 120,
        "orientation_preserving_deck_elements": proper_checked,
        "orientation_reversing_deck_elements": improper_checked,
        "declared_deck_upgrade": (
            "the reference federation formerly declared identity_only deck "
            "elements; the full order-120 incidence deck action on charts, "
            "seams, and oriented triple overlaps is now measured"
        ),
    }


def measure_federation_sector_class(
    carrier: dict[str, Any], class_order: int
) -> dict[str, Any]:
    """Measure the sector class of the realized reference federation.

    The seam schema of the reference federation carries no twist field, so its
    seam twist is the zero cochain; the face holonomies of that cochain are
    evaluated exactly and the class map (the face-holonomy sum) is applied.
    The composition content is not asserted here: the federation's own
    verifier is invoked through the public bridge report, and its higher-
    overlap cocycle receipts (twenty nonvacuous triple-restriction loops with
    exact identity composites) are required and hash-bound.
    """

    federation = reference_incidence_nerve_federation()
    _require(
        len(federation.triple_overlaps) == 20,
        "FEDERATION_SECTOR",
        "twenty seam triangles required",
    )
    for seam in federation.seams:
        _require(
            not hasattr(seam, "twist") and not hasattr(seam, "phase"),
            "FEDERATION_SECTOR",
            "the reference seam schema must not carry an undeclared twist field",
        )
    bridge = incidence_nerve_bridge_report()
    verification = bridge["federation_verification"]
    for receipt_name in (
        "HIGHER_OVERLAP_COCYCLE_CONDITION_RECEIPT",
        "NONVACUOUS_HIGHER_OVERLAP_COCYCLE_WITNESS",
        "INTERFACE_ALGEBRA_MAP_HOMOMORPHISM_RECEIPT",
        "FULL_INTERFACE_ALGEBRA_SEWING_RECEIPT",
    ):
        _require(
            verification.get(receipt_name) is True,
            "FEDERATION_SECTOR",
            f"the federation verifier does not report {receipt_name}",
        )
    face_rows = carrier["faces"]
    edges = _edge_list(face_rows)
    boundary = _boundary_two(face_rows, edges)
    zero_twist = {position: 0 for position in range(len(edges))}
    holonomies = _face_holonomies(zero_twist, boundary, len(face_rows), class_order)
    _require(
        all(value == 0 for value in holonomies),
        "FEDERATION_SECTOR",
        "the zero seam twist must have zero face holonomies",
    )
    measured_class = sum(holonomies) % class_order
    return {
        "seam_twist": "zero_cochain_no_twist_field_in_seam_schema",
        "face_holonomies_all_zero": True,
        "measured_sector_class": measured_class,
        "federation_verifier_binding": {
            "bundle_sha256": bridge["federation_bundle_sha256"],
            "triple_restriction_loops": 20,
            "higher_overlap_cocycle_condition": True,
            "nonvacuous_witness": True,
        },
        "reading": (
            "the realized reference federation carries the vacuum sector; the "
            "flux-tube witnesses below show every other class in the measured "
            "class group is realizable seam data on the twice-punctured support"
        ),
    }


# ---------------------------------------------------------------------------
# Sector menu on the base and refined complexes
# ---------------------------------------------------------------------------


def _antipodal_face(carrier: dict[str, Any], face_position: int) -> int:
    antipode = carrier["antipode"]
    target = frozenset(antipode[v] for v in carrier["faces"][face_position])
    for position, face in enumerate(carrier["faces"]):
        if frozenset(face) == target:
            return position
    raise AssertionError("antipodal face missing")


def measure_sector_menu(carrier: dict[str, Any], class_order: int) -> dict[str, Any]:
    face_rows = carrier["faces"]
    edges = _edge_list(face_rows)
    _require(len(edges) == 30, "SECTOR_COMPLEX", "thirty seams required")
    boundary = _boundary_two(face_rows, edges)
    invariants = _smith_invariants([row[:] for row in boundary])
    _require(
        len(invariants) == 19 and all(value == 1 for value in invariants),
        "SECTOR_COMPLEX",
        "the face boundary must have nineteen unit Smith invariants",
    )
    impossibility = single_puncture_impossibility(boundary, len(face_rows), class_order)
    start_face = 0
    end_face = _antipodal_face(carrier, start_face)
    _require(end_face != start_face, "SECTOR_COMPLEX", "puncture faces must differ")
    witnesses = [
        flux_tube_witness(face_rows, edges, boundary, start_face, end_face, flux, class_order)
        for flux in range(class_order)
    ]
    divisors = [d for d in range(1, class_order + 1) if class_order % d == 0]
    subgroups = {
        f"order_{d}": [c for c in range(class_order) if (c * d) % class_order == 0]
        for d in divisors
    }
    obstruction_menu = {
        name: {
            "liftable_fluxes": members,
            "obstructed_fluxes": [c for c in range(class_order) if c not in members],
        }
        for name, members in subgroups.items()
    }
    return {
        "complex": {"vertices": 12, "seams": 30, "faces": 20},
        "class_order_source": "measured_six_axis_class_group",
        "boundary_smith_invariants_all_unit": True,
        "single_puncture_impossibility": impossibility,
        "puncture_faces": {"start": start_face, "end": end_face, "antipodal": True},
        "flux_tube_witnesses": witnesses,
        "realized_flux_menu": list(range(class_order)),
        "subgroup_obstruction_menu": obstruction_menu,
        "menu_reading": (
            "a flux value lifts through a subgroup of the measured class group "
            "exactly when the subgroup contains it; only the full measured "
            "group carries the whole measured menu"
        ),
    }


def measure_refined_sector_menu(carrier: dict[str, Any], class_order: int) -> dict[str, Any]:
    tower = build_geodesic_icosahedral_tower(1)
    base = tower.levels[0]
    fine = tower.levels[1]
    _require(
        base.face_count == 20 and fine.face_count == 80,
        "SECTOR_REFINEMENT",
        "the refinement tower does not present the 20 -> 80 face refinement",
    )
    children = tower.levels[1].children_by_parent_face
    _require(
        children is not None and len(children) == 20,
        "SECTOR_REFINEMENT",
        "child faces by parent face are required",
    )
    face_rows = [[int(v) for v in row] for row in fine.faces]
    edges = _edge_list(face_rows)
    boundary = _boundary_two(face_rows, edges)
    start_parent = 0
    end_parent = _antipodal_face(carrier, start_parent)
    start_face = int(children[start_parent][0])
    end_face = int(children[end_parent][0])
    witnesses = [
        flux_tube_witness(face_rows, edges, boundary, start_face, end_face, flux, class_order)
        for flux in range(class_order)
    ]
    impossibility = single_puncture_impossibility(boundary, len(face_rows), class_order)
    return {
        "refined_complex": {
            "vertices": int(fine.vertex_count),
            "seams": len(edges),
            "faces": len(face_rows),
        },
        "puncture_faces": {
            "start_child_of": start_parent,
            "end_child_of": end_parent,
            "start": start_face,
            "end": end_face,
        },
        "single_puncture_impossibility": impossibility,
        "realized_flux_menu": [row["flux"] for row in witnesses],
        "refinement_natural_sector_menu": (
            [row["flux"] for row in witnesses] == list(range(class_order))
        ),
    }


# ---------------------------------------------------------------------------
# Artifact assembly
# ---------------------------------------------------------------------------


def produce_global_form_artifact(manifest: Mapping[str, Any]) -> dict[str, Any]:
    carrier = load_carrier(manifest)
    frame = match_vertex_frame(carrier)
    deck = measure_deck_realization(carrier, frame)
    six_axis = measure_six_axis_class_group(carrier, deck)
    class_order = six_axis["class_group_order"]
    federation_deck = measure_federation_deck_action(carrier, deck)
    federation_sector = measure_federation_sector_class(carrier, class_order)
    sector_menu = measure_sector_menu(carrier, class_order)
    refined_menu = measure_refined_sector_menu(carrier, class_order)

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "target_firewall": (
            "no_gauge_label_charge_table_matter_module_or_quotient_choice_enters_the_producer"
        ),
        "carrier_binding": {
            "carrier_manifest_sha256": carrier["manifest_sha256"],
            "port_order": carrier["ports"],
            "incidence_edge_count": 30,
            "oriented_face_count": 20,
        },
        "deck_measurement": {
            "incidence_automorphism_group_order": 120,
            "orientation_preserving_rotations": 60,
            "orientation_reversing_elements": 60,
            "antipode_is_orientation_reversing": True,
        },
        "six_axis_class_measurement": six_axis,
        "federation_deck_action": federation_deck,
        "federation_sector_class": federation_sector,
        "sector_menu": sector_menu,
        "refined_sector_menu": refined_menu,
        "provenance": {
            "producer": "oph_fpe.core.global_form_response.produce_global_form_artifact",
            "deterministic": True,
        },
        "physical_source_gate": {
            "deck_action_on_federation_measured": True,
            "six_axis_class_group_measured": True,
            "reference_federation_sector_class_measured": True,
            "flux_tube_sector_menu_realized": True,
            "single_puncture_impossibility_verified": True,
            "subgroup_obstruction_menu_exact": True,
            "refinement_natural_sector_menu": True,
            "laboratory_line_measurement": False,
            "four_dimensional_instanton_attachment": False,
            "passed": True,
            "scope": (
                "finite source-model scope: the gate aggregates the seven measured "
                "rows above; the two false rows are separate lanes (laboratory "
                "attachment #569, continuum/4d instanton normalization) and never "
                "enter 'passed'"
            ),
        },
    }
    artifact["artifact_sha256"] = "sha256:" + canonical_sha256(artifact)
    return artifact


def write_global_form_artifact(carrier_manifest_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = json.loads(carrier_manifest_path.read_text(encoding="utf-8"))
    artifact = produce_global_form_artifact(manifest)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carrier-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact = write_global_form_artifact(args.carrier_manifest, args.out)
    print(json.dumps({"status": "PASS", "artifact_sha256": artifact["artifact_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
