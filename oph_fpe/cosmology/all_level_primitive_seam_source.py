"""Target-clean registered-ladder primitive seam source packet for issue 659.

The packet constructs one declared source branch on the registered geodesic
icosahedral tower.  At every stored level the complete nearest-neighbour edge
set is serialized once, and the source grammar assigns one primitive
reconciliation-attempt symbol and one unit of counting measure to every edge.
This is a source declaration, not a derivation of unit counting from A1--A3.

Adjacent refinement levels have an exact four-to-one event lineage.  The two
halves of a coarse edge and the two fine face-interior edges parallel to it
are its four children.  Consequently normalized unit counting pushes forward
exactly.  The declared balanced integer kernel preserves the endpoint total,
lands in the nearest-balanced shell, and has exact endpoint agreement in
expectation.  Odd totals do not agree pathwise, and the packet does not
identify a completed seam attempt with an atomic #628 record write.

The refinement result is deliberately limited.  Inherited-vertex readback of
the average child reconciliation generator obeys the exact first-order
identity ``Q G_f J = G_c / 8``.  The embedded coarse subspace is not invariant
and the second-order identity fails, so no commuting repair semigroup or
continuum dynamics is promoted.  No comparison or physical data are read.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / "data/refinement/all_level_primitive_seam_source_receipt.json"
SOURCE_GATE_RECEIPT = ROOT / "data/refinement/refined_equal_seam_source_gate_receipt.json"
BOUNDED_REPAIR_RECEIPT = (
    ROOT / "data/repair_closure/bounded_atomic_self_readback_closure_receipt.json"
)
RECORD_COUNTING_PROJECTION = (
    ROOT / "data/repair_closure/record_counting_source_projection.json"
)

SCHEMA = "oph.registered-ladder-primitive-seam-source.v1"
STATUS = (
    "TARGET_CLEAN_REGISTERED_LADDER_PRIMITIVE_SEAM_ALPHABET_AND_UNIT_COUNTING_"
    "ATTAINED__EXPECTED_A2_RECONCILIATION_FIRST_ORDER_REFINEMENT_ONLY__INFINITE_"
    "TOWER_CANONICAL_DERIVATION_ATOMIC_RECORD_AND_FULL_SEMIGROUP_OPEN"
)
ISSUE = 659
MAX_LEVEL = 5
EXPECTED_SOURCE_GATE_STATUS = (
    "BASE_EQUAL_SEAM_GENERATOR_EXACT__REGISTERED_MESH_A5_EDGE_ORBITS_"
    "CLASSIFIED_WITH_RESIDUAL_GATE__SOURCE_COUNTING_EMITTER_OPEN"
)
EXPECTED_BOUNDED_STATUS = (
    "BOUNDED_EXPECTATION_LEVEL_REPAIR_FIXED_POINT_UNIQUE_MODULO_CLOCK_IN_THE_"
    "FROZEN_ADVERSARIAL_SUITE"
)
EXPECTED_BOUNDED_CERTIFICATE_SHA256 = (
    "sha256:9e87c5e4abfb3baed80058ffc832a6dbd3412f386eb383d68fee4ebee10c00d5"
)

Edge = tuple[int, int]
Vector = tuple[Fraction, ...]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _edges(mesh: Any) -> tuple[Edge, ...]:
    edges = tuple(
        sorted(
            (min(int(left), int(right)), max(int(left), int(right)))
            for left, right in mesh.edges
        )
    )
    if len(edges) != len(set(edges)):
        raise ValueError("registered seam alphabet contains duplicates")
    return edges


def _immediate_parent_set(fine: Any, vertex: int) -> frozenset[int]:
    support = fine.vertex_parent_support[vertex]
    parents = frozenset(int(parent) for parent, _weight in support)
    if len(parents) not in {1, 2} or len(parents) != len(support):
        raise ValueError("registered vertex has malformed immediate-parent support")
    return parents


def _refinement_lineage(
    coarse: Any,
    fine: Any,
) -> tuple[tuple[Edge, ...], tuple[Edge, ...], tuple[int, ...], tuple[str, ...]]:
    """Map every fine seam to its unique parallel coarse parent seam."""

    coarse_edges = _edges(coarse)
    fine_edges = _edges(fine)
    coarse_index = {edge: index for index, edge in enumerate(coarse_edges)}
    parent_indices: list[int] = []
    kinds: list[str] = []
    for left, right in fine_edges:
        left_support = _immediate_parent_set(fine, left)
        right_support = _immediate_parent_set(fine, right)
        if (
            len(left_support) == 1
            and len(right_support) == 2
            and left_support < right_support
        ):
            parent = tuple(sorted(right_support))
            kind = "boundary_half"
        elif (
            len(right_support) == 1
            and len(left_support) == 2
            and right_support < left_support
        ):
            parent = tuple(sorted(left_support))
            kind = "boundary_half"
        elif (
            len(left_support) == 2
            and len(right_support) == 2
            and len(left_support & right_support) == 1
            and len(left_support | right_support) == 3
        ):
            # The fine interior edge joins two side midpoints of one parent
            # triangle.  Its symmetric-difference endpoints form the parallel
            # opposite coarse seam.
            parent = tuple(sorted(left_support ^ right_support))
            kind = "parent_face_interior"
        else:
            raise ValueError("fine seam has no canonical coarse parent")
        parent_index = coarse_index.get(parent)
        if parent_index is None:
            raise ValueError("fine seam parent is absent from coarse alphabet")
        parent_indices.append(parent_index)
        kinds.append(kind)

    multiplicities = Counter(parent_indices)
    if set(multiplicities) != set(range(len(coarse_edges))):
        raise ValueError("refinement lineage omits a coarse seam")
    if set(multiplicities.values()) != {4}:
        raise ValueError("refinement lineage is not four-to-one")
    for parent_index in range(len(coarse_edges)):
        child_kinds = Counter(
            kind
            for candidate, kind in zip(parent_indices, kinds, strict=True)
            if candidate == parent_index
        )
        if child_kinds != {"boundary_half": 2, "parent_face_interior": 2}:
            raise ValueError("coarse seam does not have the canonical four child types")
    return coarse_edges, fine_edges, tuple(parent_indices), tuple(kinds)


def _event_id(level: int, geometry_hash: str, edge: Edge) -> str:
    return _sha(
        {
            "schema": "oph.primitive-seam-reconciliation-attempt.v1",
            "level": level,
            "geometry_hash": geometry_hash,
            "unoriented_endpoints": list(edge),
            "event_type": "balanced_endpoint_reconciliation_attempt",
            "counting_multiplicity": 1,
        }
    )


def _balanced_integer_kernel(
    left: int,
    right: int,
) -> tuple[tuple[Fraction, tuple[int, int]], ...]:
    total = int(left) + int(right)
    lower = total // 2
    upper = total - lower
    if lower == upper:
        return ((Fraction(1), (lower, upper)),)
    return (
        (Fraction(1, 2), (lower, upper)),
        (Fraction(1, 2), (upper, lower)),
    )


def _a2_reconciliation_certificate() -> dict[str, Any]:
    probes = 0
    odd_pathwise_disagreement_witness: dict[str, Any] | None = None
    for left in range(-8, 9):
        for right in range(-8, 9):
            kernel = _balanced_integer_kernel(left, right)
            if sum((weight for weight, _outcome in kernel), Fraction()) != 1:
                raise ValueError("balanced integer kernel is not normalized")
            if any(sum(outcome) != left + right for _weight, outcome in kernel):
                raise ValueError("balanced integer kernel changes the endpoint total")
            if any(abs(outcome[0] - outcome[1]) > 1 for _weight, outcome in kernel):
                raise ValueError("balanced integer kernel misses nearest agreement")
            target = Fraction(left + right, 2)
            expected_left = sum(
                (weight * outcome[0] for weight, outcome in kernel), Fraction()
            )
            expected_right = sum(
                (weight * outcome[1] for weight, outcome in kernel), Fraction()
            )
            if (expected_left, expected_right) != (target, target):
                raise ValueError("balanced integer kernel fails expected agreement")
            reversed_kernel = tuple(
                (weight, (outcome[1], outcome[0])) for weight, outcome in kernel
            )
            if Counter(reversed_kernel) != Counter(
                _balanced_integer_kernel(right, left)
            ):
                raise ValueError("balanced kernel is not endpoint-reversal natural")
            if (
                odd_pathwise_disagreement_witness is None
                and (left + right) % 2
                and any(outcome[0] != outcome[1] for _weight, outcome in kernel)
            ):
                odd_pathwise_disagreement_witness = {
                    "input": [left, right],
                    "outcomes": [
                        {"probability": str(weight), "value": list(outcome)}
                        for weight, outcome in kernel
                    ],
                }
            probes += 1
    if odd_pathwise_disagreement_witness is None:
        raise ValueError("odd pathwise disagreement control was not exercised")
    return {
        "declared_semantic_rule": "balanced_integer_endpoint_kernel",
        "integer_pair_probes": probes,
        "probe_box": [-8, 8],
        "endpoint_total_preserved_pathwise": True,
        "changes_only_selected_endpoints": True,
        "nearest_balanced_shell_reached_pathwise": True,
        "endpoint_reversal_natural": True,
        "expected_endpoint_agreement_exact": True,
        "expected_rational_action": (
            "(x_i,x_j) maps in expectation to "
            "((x_i+x_j)/2,(x_i+x_j)/2)"
        ),
        "odd_total_pathwise_exact_agreement": False,
        "canonical_a2_pathwise_agreement_discharged": False,
        "odd_total_pathwise_disagreement_witness": (
            odd_pathwise_disagreement_witness
        ),
        "rule_selected_by_canonical_a2_alone": False,
        "issue_628_atomic_record_write_identification": False,
        "issue_628_relation": (
            "The source-bound #628 result supplies the exact base conditional "
            "mean generator. Its atomic signed record writes and conservative "
            "one-unit repair moves are not identified with one completed "
            "balanced seam attempt by this packet."
        ),
    }


def _j_row(fine: Any, vertex: int) -> dict[int, Fraction]:
    support = fine.vertex_parent_support[vertex]
    weight = Fraction(1, len(support))
    return {int(parent): weight for parent, _stored_weight in support}


def _add_scaled(
    target: dict[int, Fraction],
    source: Mapping[int, Fraction],
    scale: Fraction,
) -> None:
    for key, value in source.items():
        target[key] += scale * value
        if target[key] == 0:
            del target[key]


def _child_aggregate_identity(
    coarse: Any,
    fine: Any,
    coarse_edges: Sequence[Edge],
    fine_edges: Sequence[Edge],
    parent_indices: Sequence[int],
) -> None:
    children: list[list[Edge]] = [[] for _ in coarse_edges]
    for edge, parent in zip(fine_edges, parent_indices, strict=True):
        children[parent].append(edge)

    for parent_index, (coarse_left, coarse_right) in enumerate(coarse_edges):
        actual: dict[int, dict[int, Fraction]] = {}
        for fine_left, fine_right in children[parent_index]:
            for endpoint, other in (
                (fine_left, fine_right),
                (fine_right, fine_left),
            ):
                if endpoint >= coarse.vertex_count:
                    continue
                row: defaultdict[int, Fraction] = defaultdict(Fraction)
                _add_scaled(row, _j_row(fine, other), Fraction(1, 2))
                _add_scaled(row, _j_row(fine, endpoint), Fraction(-1, 2))
                actual[endpoint] = dict(row)
        expected = {
            coarse_left: {
                coarse_left: Fraction(-1, 4),
                coarse_right: Fraction(1, 4),
            },
            coarse_right: {
                coarse_left: Fraction(1, 4),
                coarse_right: Fraction(-1, 4),
            },
        }
        if actual != expected:
            raise ValueError("child aggregate reconciliation identity failed")


def _interpolate(values: Sequence[Fraction], fine: Any) -> Vector:
    return tuple(
        sum((values[int(parent)] for parent, _weight in support), Fraction())
        / len(support)
        for support in fine.vertex_parent_support
    )


def _mean_reconciliation_delta(
    values: Sequence[Fraction],
    edges: Sequence[Edge],
) -> Vector:
    output = [Fraction()] * len(values)
    scale = Fraction(1, 2 * len(edges))
    for left, right in edges:
        delta = (values[right] - values[left]) * scale
        output[left] += delta
        output[right] -= delta
    return tuple(output)


def _first_nonzero(values: Sequence[Fraction]) -> dict[str, Any] | None:
    for index, value in enumerate(values):
        if value:
            return {"index": index, "residual": str(value)}
    return None


def _refinement_row(
    coarse: Any,
    fine: Any,
) -> dict[str, Any]:
    coarse_edges, fine_edges, parent_indices, kinds = _refinement_lineage(
        coarse, fine
    )
    _child_aggregate_identity(
        coarse,
        fine,
        coarse_edges,
        fine_edges,
        parent_indices,
    )

    coarse_probe = [Fraction()] * coarse.vertex_count
    coarse_probe[0] = Fraction(1)
    interpolated = _interpolate(coarse_probe, fine)
    fine_first = _mean_reconciliation_delta(interpolated, fine_edges)
    coarse_first = _mean_reconciliation_delta(coarse_probe, coarse_edges)
    interpolated_coarse_first = _interpolate(coarse_first, fine)
    strong_residual = tuple(
        left - Fraction(1, 8) * right
        for left, right in zip(
            fine_first,
            interpolated_coarse_first,
            strict=True,
        )
    )
    strong_witness = _first_nonzero(strong_residual)
    if strong_witness is None:
        raise ValueError("strong refinement control unexpectedly commuted")

    inherited_first = fine_first[: coarse.vertex_count]
    if inherited_first != tuple(Fraction(1, 8) * value for value in coarse_first):
        raise ValueError("first-order inherited reconciliation identity failed")

    fine_second = _mean_reconciliation_delta(fine_first, fine_edges)
    coarse_second = _mean_reconciliation_delta(coarse_first, coarse_edges)
    second_residual = tuple(
        fine_second[index] - Fraction(1, 64) * coarse_second[index]
        for index in range(coarse.vertex_count)
    )
    second_witness = _first_nonzero(second_residual)
    if second_witness is None:
        raise ValueError("second-order refinement control unexpectedly commuted")

    kind_counts = Counter(kinds)
    return {
        "coarse_level": int(coarse.level),
        "fine_level": int(fine.level),
        "coarse_geometry_hash": coarse.geometry_hash,
        "fine_geometry_hash": fine.geometry_hash,
        "coarse_seam_count": len(coarse_edges),
        "fine_seam_count": len(fine_edges),
        "fine_to_coarse_count_ratio": 4,
        "fine_parent_seam_indices": list(parent_indices),
        "child_kind_counts": dict(sorted(kind_counts.items())),
        "children_per_parent_seam": 4,
        "boundary_half_children_per_parent": 2,
        "parent_face_interior_children_per_parent": 2,
        "raw_unit_count_pushforward_multiplier": 4,
        "normalized_unit_count_pushforward_exact": True,
        "normalized_pushforward_identity": (
            "4 / |E_f| = 1 / |E_c| because |E_f| = 4 |E_c|"
        ),
        "child_aggregate_inherited_readback_identity": (
            "Q sum_{child of e} (E_child-I) J = (1/2)(E_e-I)"
        ),
        "child_aggregate_identity_exact_for_every_parent": True,
        "average_child_inherited_readback_identity": (
            "Q (1/4) sum_{child of e} (E_child-I) J = (1/8)(E_e-I)"
        ),
        "level_mean_generator_identity": "Q G_f J = (1/8) G_c",
        "level_mean_generator_identity_exact": True,
        "strong_intertwiner": False,
        "strong_intertwiner_witness_on_coarse_basis_zero": strong_witness,
        "second_order_inherited_identity": False,
        "second_order_witness_on_coarse_basis_zero": second_witness,
    }


def _load_parent_packets() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    gate = json.loads(SOURCE_GATE_RECEIPT.read_text(encoding="utf-8"))
    bounded = json.loads(BOUNDED_REPAIR_RECEIPT.read_text(encoding="utf-8"))
    projection = json.loads(RECORD_COUNTING_PROJECTION.read_text(encoding="utf-8"))
    if gate.get("schema") != "oph.refined-equal-seam-source-selection-gate.v1":
        raise ValueError("unexpected source-gate schema")
    if gate.get("status") != EXPECTED_SOURCE_GATE_STATUS:
        raise ValueError("source-gate status drift")
    gate_payload = copy.deepcopy(gate)
    gate_hash = gate_payload.pop("payload_sha256", None)
    if gate_hash != _sha(gate_payload):
        raise ValueError("source-gate payload hash drift")
    if bounded.get("schema") != "oph.bounded_atomic_self_readback_closure.v1":
        raise ValueError("unexpected bounded-repair schema")
    if bounded.get("status") != EXPECTED_BOUNDED_STATUS:
        raise ValueError("bounded-repair status drift")
    if (
        bounded.get("certificate_payload_sha256")
        != EXPECTED_BOUNDED_CERTIFICATE_SHA256
    ):
        raise ValueError("bounded-repair certificate drift")
    if projection.get("schema") != "oph.record_counting_source_projection.v1":
        raise ValueError("unexpected record-counting projection schema")
    if projection.get("source_issue") != 628:
        raise ValueError("record-counting source issue drift")
    return gate, bounded, projection


def build_receipt(max_level: int = MAX_LEVEL) -> dict[str, Any]:
    if max_level != MAX_LEVEL:
        raise ValueError(f"canonical packet requires max_level={MAX_LEVEL}")
    gate, bounded, projection = _load_parent_packets()
    tower = build_geodesic_icosahedral_tower(max_level)
    orbit_rows = gate["edge_orbit_rows"]
    if len(orbit_rows) != len(tower.levels):
        raise ValueError("source-gate orbit levels do not cover the registered tower")

    level_rows: list[dict[str, Any]] = []
    for mesh, orbit_row in zip(tower.levels, orbit_rows, strict=True):
        edges = _edges(mesh)
        if int(orbit_row["level"]) != int(mesh.level):
            raise ValueError("source-gate orbit level drift")
        if int(orbit_row["edge_count"]) != len(edges):
            raise ValueError("source-gate edge count drift")
        event_ids = [_event_id(mesh.level, mesh.geometry_hash, edge) for edge in edges]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("primitive event identifiers are not unique")
        orbit_count = int(orbit_row["edge_orbit_count"])
        level_rows.append(
            {
                "level": int(mesh.level),
                "frequency": int(mesh.frequency),
                "geometry_hash": mesh.geometry_hash,
                "vertex_count": int(mesh.vertex_count),
                "seam_count": len(edges),
                "complete_registered_unoriented_seams": [list(edge) for edge in edges],
                "primitive_event_type": (
                    "oph.primitive-seam-reconciliation-attempt.v1"
                ),
                "one_event_per_registered_seam": True,
                "event_id_sequence_sha256": _sha(event_ids),
                "a5_edge_orbit_count_from_residual_gated_parent": orbit_count,
                "declared_multiplicity_by_a5_orbit": [1] * orbit_count,
                "all_orbit_multiplicities_equal_to_one": True,
                "normalized_event_probability": f"1/{len(edges)}",
                "target_or_comparison_fields_read": [],
            }
        )

    refinement_rows = [
        _refinement_row(coarse, fine)
        for coarse, fine in zip(tower.levels[:-1], tower.levels[1:], strict=True)
    ]
    a2 = _a2_reconciliation_certificate()

    source_files = [
        ROOT / "oph_fpe/core/icosahedral.py",
        BOUNDED_REPAIR_RECEIPT,
        RECORD_COUNTING_PROJECTION,
        SOURCE_GATE_RECEIPT,
        Path(__file__).resolve(),
        ROOT
        / "oph_fpe/cosmology/verify_all_level_primitive_seam_source_independent.py",
        ROOT / "tests/test_all_level_primitive_seam_source.py",
    ]
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "status": STATUS,
        "source_scope": {
            "registered_geometry_levels": list(range(max_level + 1)),
            "source_emitter_kind": (
                "target-clean declared branch emitter instantiated by this packet"
            ),
            "external_comparison_data_used": False,
            "particle_data_used": False,
            "sky_data_used": False,
            "target_values_used": False,
            "canonical_a1_a3_derivation_of_emitter": False,
        },
        "parent_bindings": {
            "source_gate_receipt": SOURCE_GATE_RECEIPT.relative_to(ROOT).as_posix(),
            "source_gate_payload_sha256": gate["payload_sha256"],
            "bounded_repair_receipt": BOUNDED_REPAIR_RECEIPT.relative_to(ROOT).as_posix(),
            "bounded_repair_certificate_payload_sha256": bounded[
                "certificate_payload_sha256"
            ],
            "record_counting_projection": RECORD_COUNTING_PROJECTION.relative_to(
                ROOT
            ).as_posix(),
            "record_counting_upstream_file_sha256": projection[
                "upstream_file_sha256"
            ],
            "record_counting_upstream_manifest_sha256": projection[
                "upstream_manifest_sha256"
            ],
        },
        "primitive_event_declaration": {
            "event_type": "oph.primitive-seam-reconciliation-attempt.v1",
            "event_identity_fields": [
                "level",
                "geometry_hash",
                "unoriented_endpoints",
                "event_type",
                "counting_multiplicity",
            ],
            "one_serialized_row_is_one_primitive_attempt_symbol": True,
            "same_event_type_at_every_registered_level": True,
            "same_declared_counting_multiplicity_at_every_registered_level": 1,
            "completed_attempt_is_issue_628_atomic_record_event": False,
            "primitive_here_means_irreducible_in_declared_attempt_alphabet": True,
            "primitive_here_does_not_mean_proved_atomic_record_write": True,
        },
        "level_alphabets": level_rows,
        "unit_counting_certificate": {
            "complete_alphabet_enumerated": True,
            "duplicates": 0,
            "omitted_registered_seams": 0,
            "hidden_event_species_declared": 0,
            "event_multiplicity_rule": "one per serialized registered seam",
            "exact_unit_counting_across_a5_orbit_classes_in_declared_source": True,
            "normalized_reference_is_uniform_on_each_complete_level": True,
            "unit_counting_forced_by_a5_alone": False,
            "unit_counting_derived_from_canonical_a1_a3": False,
            "unit_counting_is_additional_source_branch_declaration": True,
        },
        "a2_reconciliation": {
            **a2,
            "declared_rule_covers_every_serialized_event": True,
            "complete_alphabet_coverage_exact": True,
            "expected_balancing_is_diagnostic_not_canonical_a2_agreement": True,
            "a2_does_not_select_the_rule_or_event_alphabet": True,
        },
        "refinement_certificate": {
            "rows": refinement_rows,
            "complete_event_lineage_exact": True,
            "normalized_unit_counting_refinement_natural": True,
            "raw_unit_counting_refinement_natural_without_rescaling": False,
            "raw_count_scale_factor_per_adjacent_level": 4,
            "expected_reconciliation_first_order_inherited_readback_exact": True,
            "strong_event_action_refinement_natural": False,
            "repair_semigroup_refinement_natural": False,
            "continuum_limit_certified": False,
        },
        "selection_decision": {
            "target_clean_registered_ladder_event_emitter_instantiated": True,
            "registered_ladder_complete_primitive_attempt_alphabet_source_emitted": True,
            "registered_ladder_exact_unit_counting_source_emitted_on_declared_branch": True,
            "infinite_tower_complete_primitive_attempt_alphabet_source_emitted": False,
            "infinite_tower_exact_unit_counting_source_emitted_on_declared_branch": False,
            "canonical_a1_a3_force_the_emitter": False,
            "issue_628_atomic_record_bridge_discharged": False,
            "pathwise_exact_a2_agreement_for_odd_integer_totals": False,
            "first_order_refinement_readback_discharged": True,
            "full_refinement_commuting_diagram_discharged": False,
            "physical_repair_law_selected": False,
            "physical_covariance_selected": False,
            "continuum_equal_seam_operator_selected": False,
            "physical_prediction": False,
            "promotion_allowed": False,
        },
        "source_pins": [_raw_pin(path) for path in source_files],
        "reopen_or_advance_condition": (
            "Promote beyond the declared source branch only after the completed "
            "seam-attempt symbols are derived from or identified with the atomic "
            "record/repair instrument, and a refinement dynamics is supplied that "
            "commutes beyond first inherited readback. Physical use additionally "
            "requires the independent continuum and readout bridges."
        ),
        "claim_boundary": (
            "Target-clean construction of the complete registered nearest-neighbour "
            "seam alphabets through level five. The packet declares one primitive "
            "attempt symbol and one unit of counting measure per seam, including "
            "across the multiple residual-gated A5 edge orbits, and proves exact "
            "four-to-one parent lineage with normalized-counting pushforward. The "
            "packet does not construct the seam alphabet beyond level five. The "
            "balanced integer rule reaches pathwise nearest agreement and exact "
            "agreement in expectation. Unit counting is emitted by this declared "
            "source branch rather than derived from canonical A1--A3. A completed "
            "attempt is not identified with a #628 atomic record write. Refinement "
            "commutes only at first inherited readback; the strong and second-order "
            "controls fail exactly. No full repair semigroup, continuum operator, "
            "physical repair law, covariance, readout, or prediction is selected."
        ),
    }
    receipt["payload_sha256"] = _sha(receipt)
    return receipt


def write_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    receipt = build_receipt()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    receipt = write_receipt(args.output)
    print(receipt["status"])
    print(receipt["payload_sha256"])


if __name__ == "__main__":
    main()
