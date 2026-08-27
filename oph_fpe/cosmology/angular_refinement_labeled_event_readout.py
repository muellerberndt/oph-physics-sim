"""Exact labeled-event readout on the level-one icosahedral carrier.

The existing averaged-repair packet observes a fine field through
``(Q, Q L_f, ..., Q L_f^41)`` and proves that this particular passive
observation matrix has rank 29.  This packet keeps that theorem unchanged
and constructs a richer active protocol.

For every level-one midpoint ``m`` and either inherited endpoint ``u`` of
its parent edge, let ``E_(u,m)`` be the declared pair-average intervention.
The inherited readback then contains

    (Q E_(u,m) x)_u = (x_u + x_m) / 2,

and the baseline readback gives ``(Q x)_u = x_u``.  Thus

    x_m = 2 (Q E_(u,m) x)_u - (Q x)_u.

One endpoint event per midpoint yields a 42 by 42 rational measurement
matrix with an explicit inverse.  Keeping both endpoints gives a sixty-event
family stable under the proper A5 action; together with ``Q`` it also has
rank 42.  The minimal thirty-event selector is deliberately not claimed to
be A5 invariant.

Every response must refer to the same pre-event field.  Operationally this
requires repeatable preparation, reset/checkpoint access, parallel prepared
copies, or a nondestructive response instrument.  This packet constructs no
such physical instrument and makes no sky or laboratory identification.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.core.icosahedral import (
    build_geodesic_icosahedral_tower,
    icosahedral_a5_port_permutations,
)


SCHEMA = "oph.angular_refinement_labeled_event_readout.v1"
VERIFICATION_SCHEMA = (
    "oph.angular_refinement_labeled_event_readout_producer_verification.v1"
)
STATUS = "EXACT_LEVEL_ONE_LABELED_EVENT_FULL_RECONSTRUCTION__PHYSICAL_INSTRUMENT_OPEN"
ISSUE = 643
CLAIM_BOUNDARY = (
    "Q plus labeled inherited-endpoint/midpoint pair-average responses "
    "reconstructs every coordinate of the exact 42-slot level-one field. "
    "The complete A5-stable sixty-event family and a minimal thirty-event "
    "subinstrument both have rank 42 once Q is included. The theorem "
    "requires same-pre-event responses and event labels. It does not derive "
    "the event grammar, reset/checkpoint instrument, or physical readout, "
    "and it leaves the passive rank-29 theorem unchanged."
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "data/repair_closure/angular_refinement_labeled_event_readout_receipt.json"
)
PARENT_RECEIPT = (
    REPOSITORY_ROOT
    / "data/repair_closure/angular_refinement_repair_observability_receipt.json"
)
PRODUCER_SOURCE = Path(__file__).resolve()
VERIFIER_SOURCE = (
    Path(__file__).resolve().parent
    / "verify_angular_refinement_labeled_event_readout_independent.py"
)

INHERITED_COUNT = 12
MIDPOINT_COUNT = 30
FINE_COUNT = 42

Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


class LabeledReadoutError(ValueError):
    """The labeled-event packet refused to build."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LabeledReadoutError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def tagged_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def self_digest(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    return tagged_sha256(canonical_json(body))


def raw_pin(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    resolved.relative_to(REPOSITORY_ROOT.resolve())
    payload = resolved.read_bytes()
    return {
        "path": resolved.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": len(payload),
        "sha256": tagged_sha256(payload),
    }


def unit_row(index: int, width: int = FINE_COUNT) -> Vector:
    require(0 <= index < width, "unit row index outside width")
    return tuple(Fraction(int(index == column)) for column in range(width))


def average_response_row(parent: int, midpoint: int) -> Vector:
    require(0 <= parent < INHERITED_COUNT, "parent is not inherited")
    require(INHERITED_COUNT <= midpoint < FINE_COUNT, "slot is not a midpoint")
    row = [Fraction()] * FINE_COUNT
    row[parent] = Fraction(1, 2)
    row[midpoint] = Fraction(1, 2)
    return tuple(row)


def pair_average_event(parent: int, midpoint: int) -> Matrix:
    rows = [list(unit_row(index)) for index in range(FINE_COUNT)]
    response = list(average_response_row(parent, midpoint))
    rows[parent] = response
    rows[midpoint] = response
    return tuple(tuple(row) for row in rows)


def matmul(
    left: Sequence[Sequence[Fraction]],
    right: Sequence[Sequence[Fraction]],
) -> Matrix:
    require(bool(left) and bool(right), "empty matrix product")
    require(len(left[0]) == len(right), "matrix product shape mismatch")
    columns = tuple(zip(*right, strict=True))
    return tuple(
        tuple(
            sum((x * y for x, y in zip(row, column, strict=True)), Fraction())
            for column in columns
        )
        for row in left
    )


def matvec(matrix: Sequence[Sequence[Fraction]], vector: Sequence[Fraction]) -> Vector:
    return tuple(
        sum((x * y for x, y in zip(row, vector, strict=True)), Fraction())
        for row in matrix
    )


def rank_over_q(matrix: Sequence[Sequence[Fraction]]) -> int:
    work = [list(map(Fraction, row)) for row in matrix]
    if not work:
        return 0
    width = len(work[0])
    require(all(len(row) == width for row in work), "rank matrix is ragged")
    rank = 0
    for column in range(width):
        pivot = next(
            (row for row in range(rank, len(work)) if work[row][column]),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        work[rank] = [value / pivot_value for value in work[rank]]
        for row in range(len(work)):
            if row == rank:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [
                    value - factor * pivot_entry
                    for value, pivot_entry in zip(work[row], work[rank], strict=True)
                ]
        rank += 1
        if rank == len(work):
            break
    return rank


def matrix_digest(matrix: Sequence[Sequence[Fraction]]) -> str:
    return tagged_sha256(
        canonical_json([[str(value) for value in row] for row in matrix])
    )


def support_index(fine: Any) -> dict[tuple[int, ...], int]:
    result: dict[tuple[int, ...], int] = {}
    for index, support in enumerate(fine.vertex_parent_support):
        parents = tuple(sorted(int(parent) for parent, _weight in support))
        require(parents not in result, "parent support is not injective")
        result[parents] = index
    return result


def midpoint_rows(coarse: Any, fine: Any) -> list[dict[str, Any]]:
    by_support = support_index(fine)
    rows = []
    for midpoint in range(INHERITED_COUNT, fine.vertex_count):
        support = tuple(
            sorted(
                int(parent) for parent, _weight in fine.vertex_parent_support[midpoint]
            )
        )
        require(len(support) == 2, "a midpoint does not have two parents")
        require(by_support[support] == midpoint, "midpoint support lookup drift")
        require(
            tuple(support) in {tuple(sorted(map(int, edge))) for edge in coarse.edges},
            "midpoint support is not a coarse edge",
        )
        rows.append(
            {
                "midpoint_carrier_slot": midpoint,
                "midpoint_label": f"m{midpoint:02d}",
                "parent_edge": list(support),
            }
        )
    require(len(rows) == MIDPOINT_COUNT, "midpoint row count drift")
    return rows


def event_record(row: Mapping[str, Any], parent: int) -> dict[str, Any]:
    midpoint = int(row["midpoint_carrier_slot"])
    left, right = map(int, row["parent_edge"])
    require(parent in (left, right), "event parent is not on the parent edge")
    other = right if parent == left else left
    return {
        "label": f"avg:p{parent:02d}:m{midpoint:02d}",
        "parent_inherited_slot": parent,
        "other_inherited_slot": other,
        "midpoint_carrier_slot": midpoint,
        "parent_edge": [left, right],
        "averaging_factor": "1/2",
        "selected_Q_coordinate": parent,
        "response_sparse": {str(parent): "1/2", str(midpoint): "1/2"},
    }


def event_families(
    coarse: Any,
    fine: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    midpoints = midpoint_rows(coarse, fine)
    selected: list[dict[str, Any]] = []
    directed: list[dict[str, Any]] = []
    for row in midpoints:
        endpoints = tuple(map(int, row["parent_edge"]))
        selected.append(event_record(row, min(endpoints)))
        directed.extend(event_record(row, parent) for parent in endpoints)
    require(len(selected) == MIDPOINT_COUNT, "selected event count drift")
    require(len(directed) == 2 * MIDPOINT_COUNT, "directed event count drift")
    require(
        len({event["label"] for event in directed}) == len(directed),
        "directed event labels are not unique",
    )
    fine_edges = {tuple(sorted(map(int, edge))) for edge in fine.edges}
    for event in directed:
        pair = tuple(
            sorted(
                (
                    int(event["parent_inherited_slot"]),
                    int(event["midpoint_carrier_slot"]),
                )
            )
        )
        require(pair in fine_edges, "a directed event is not a fine-carrier edge")
    return midpoints, selected, directed


def lift_rotation(fine: Any, base: Sequence[int]) -> tuple[int, ...]:
    by_support = support_index(fine)
    lifted = []
    for support in fine.vertex_parent_support:
        mapped = tuple(sorted(int(base[int(parent)]) for parent, _weight in support))
        require(mapped in by_support, "base rotation did not lift to level one")
        lifted.append(by_support[mapped])
    require(len(set(lifted)) == fine.vertex_count, "lift is not a permutation")
    return tuple(lifted)


def symmetry_certificate(
    fine: Any, directed: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rotations = tuple(icosahedral_a5_port_permutations())
    require(len(rotations) == 60 and len(set(rotations)) == 60, "A5 order drift")
    event_keys = {
        (int(event["parent_inherited_slot"]), int(event["midpoint_carrier_slot"]))
        for event in directed
    }
    require(len(event_keys) == 60, "directed event key count drift")
    representative = min(event_keys)
    orbit = set()
    covariance_checks = 0
    for rotation in rotations:
        lifted = lift_rotation(fine, rotation)
        for parent, midpoint in event_keys:
            image = (int(rotation[parent]), int(lifted[midpoint]))
            require(image in event_keys, "A5 image left directed event family")
            covariance_checks += 1
        orbit.add((int(rotation[representative[0]]), int(lifted[representative[1]])))
    require(len(orbit) == 60, "directed event family is not one A5 orbit")
    return {
        "proper_carrier_group": "A5",
        "proper_rotation_count": len(rotations),
        "directed_event_count": len(event_keys),
        "directed_event_covariance_checks": covariance_checks,
        "representative_directed_event": list(representative),
        "representative_orbit_size": len(orbit),
        "directed_event_family_is_one_A5_orbit": True,
        "directed_event_family_is_A5_stable": True,
        "minimal_thirty_event_selector_is_A5_invariant": False,
        "no_A5_equivariant_one_parent_per_midpoint_section": True,
        "no_section_reason": (
            "the sixty directed endpoint-midpoint events form one A5 orbit; an "
            "invariant nonempty subset cannot contain only one of the two events "
            "over each of the thirty midpoint supports"
        ),
    }


def baseline_rows() -> Matrix:
    return tuple(unit_row(index) for index in range(INHERITED_COUNT))


def scalar_event_rows(events: Sequence[Mapping[str, Any]]) -> Matrix:
    return tuple(
        average_response_row(
            int(event["parent_inherited_slot"]),
            int(event["midpoint_carrier_slot"]),
        )
        for event in events
    )


def stacked_full_q_rows(events: Sequence[Mapping[str, Any]]) -> Matrix:
    rows = list(baseline_rows())
    for event in events:
        matrix = pair_average_event(
            int(event["parent_inherited_slot"]),
            int(event["midpoint_carrier_slot"]),
        )
        rows.extend(matrix[:INHERITED_COUNT])
    return tuple(rows)


def decoder_matrix(selected: Sequence[Mapping[str, Any]]) -> Matrix:
    require(len(selected) == MIDPOINT_COUNT, "decoder event count drift")
    rows: list[list[Fraction]] = []
    for inherited in range(INHERITED_COUNT):
        row = [Fraction()] * FINE_COUNT
        row[inherited] = 1
        rows.append(row)
    for event_offset, event in enumerate(selected):
        row = [Fraction()] * FINE_COUNT
        row[int(event["parent_inherited_slot"])] = -1
        row[INHERITED_COUNT + event_offset] = 2
        rows.append(row)
    return tuple(tuple(row) for row in rows)


def linear_certificate(
    selected: Sequence[Mapping[str, Any]],
    directed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    for event in directed:
        parent = int(event["parent_inherited_slot"])
        midpoint = int(event["midpoint_carrier_slot"])
        matrix = pair_average_event(parent, midpoint)
        require(
            matrix[parent] == average_response_row(parent, midpoint),
            "Q E response identity failed",
        )
        require(
            matmul(matrix, matrix) == matrix, "pair-average event is not idempotent"
        )
        require(
            all(sum(row, Fraction()) == 1 for row in matrix),
            "pair-average event is not row stochastic",
        )

    minimal = baseline_rows() + scalar_event_rows(selected)
    directed_only = scalar_event_rows(directed)
    symmetric = baseline_rows() + directed_only
    decoder = decoder_matrix(selected)
    identity = tuple(unit_row(index) for index in range(FINE_COUNT))
    require(matmul(decoder, minimal) == identity, "decoder identity failed")
    require(rank_over_q(minimal) == FINE_COUNT, "minimal readout rank drift")
    require(
        rank_over_q(directed_only) == FINE_COUNT - 1,
        "directed-only rank diagnostic drift",
    )
    require(rank_over_q(symmetric) == FINE_COUNT, "A5 readout rank drift")

    selected_full = stacked_full_q_rows(selected)
    directed_full = stacked_full_q_rows(directed)
    require(rank_over_q(selected_full) == FINE_COUNT, "selected full-Q rank drift")
    require(rank_over_q(directed_full) == FINE_COUNT, "A5 full-Q rank drift")

    basis_checks = 0
    for index in range(FINE_COUNT):
        basis = unit_row(index)
        require(
            matvec(decoder, matvec(minimal, basis)) == basis,
            f"basis reconstruction failed at slot {index}",
        )
        basis_checks += 1
    return {
        "arithmetic": "exact rational Fraction arithmetic; no floating point",
        "baseline_Q_shape": [INHERITED_COUNT, FINE_COUNT],
        "baseline_Q_rank_over_Q": rank_over_q(baseline_rows()),
        "minimal_selected_scalar_shape": [len(minimal), FINE_COUNT],
        "minimal_selected_scalar_rank_over_Q": rank_over_q(minimal),
        "minimal_selected_scalar_matrix_sha256": matrix_digest(minimal),
        "decoder_shape": [len(decoder), len(decoder[0])],
        "decoder_matrix_sha256": matrix_digest(decoder),
        "decoder_times_measurement_is_identity": True,
        "all_standard_basis_reconstruction_checks": basis_checks,
        "A5_stable_scalar_shape": [len(symmetric), FINE_COUNT],
        "A5_stable_scalar_rank_over_Q": rank_over_q(symmetric),
        "A5_stable_scalar_matrix_sha256": matrix_digest(symmetric),
        "directed_event_rows_without_Q_shape": [len(directed_only), FINE_COUNT],
        "directed_event_rows_without_Q_rank_over_Q": rank_over_q(directed_only),
        "directed_event_rows_without_Q_kernel_dimension": (
            FINE_COUNT - rank_over_q(directed_only)
        ),
        "selected_full_Q_stack_shape": [len(selected_full), FINE_COUNT],
        "selected_full_Q_stack_rank_over_Q": rank_over_q(selected_full),
        "A5_stable_full_Q_stack_shape": [len(directed_full), FINE_COUNT],
        "A5_stable_full_Q_stack_rank_over_Q": rank_over_q(directed_full),
        "inherited_reconstruction": "x_u = (Q x)_u",
        "midpoint_reconstruction": (
            "x_m = 2 (Q E_(u,m) x)_u - (Q x)_u for the event label naming (u,m)"
        ),
    }


def validate_parent() -> dict[str, Any]:
    parent = json.loads(PARENT_RECEIPT.read_text(encoding="ascii"))
    require(
        parent.get("schema") == "oph.angular_refinement_repair_observability.v1",
        "passive parent schema drift",
    )
    refinement = parent.get("refinement_and_repair")
    require(isinstance(refinement, Mapping), "passive parent refinement block missing")
    witness = refinement.get("observability_witness")
    require(isinstance(witness, Mapping), "passive observability witness missing")
    require(
        witness.get("observability_matrix_shape") == [504, 42]
        and witness.get("observability_rank_over_Q") == 29
        and witness.get("repair_invisible_detail_dimension") == 13,
        "passive rank-29 parent drift",
    )
    return parent


def _payload() -> dict[str, Any]:
    validate_parent()
    tower = build_geodesic_icosahedral_tower(1)
    coarse, fine = tower.levels
    require(coarse.vertex_count == INHERITED_COUNT, "coarse vertex count drift")
    require(fine.vertex_count == FINE_COUNT, "fine vertex count drift")
    midpoints, selected, directed = event_families(coarse, fine)
    linear = linear_certificate(selected, directed)
    symmetry = symmetry_certificate(fine, directed)

    return {
        "schema": SCHEMA,
        "issue": ISSUE,
        "status": STATUS,
        "source_scope": {
            "coarse_level": 0,
            "fine_level": 1,
            "inherited_vertex_count": INHERITED_COUNT,
            "midpoint_vertex_count": MIDPOINT_COUNT,
            "fine_vertex_count": FINE_COUNT,
            "base_readout": "Q restricts to the twelve inherited vertices",
            "event_operator": (
                "E_(u,m) replaces the inherited endpoint u and midpoint m by "
                "their exact arithmetic mean and fixes every other fine slot"
            ),
            "external_comparison_data_used": False,
            "target_values_used": False,
        },
        "parent_pins": [raw_pin(PARENT_RECEIPT)],
        "implementation_integrity_pins": {
            "scope": (
                "deterministic byte-integrity pins only; self-digests are not "
                "authenticated external custody"
            ),
            "producer": raw_pin(PRODUCER_SOURCE),
            "independent_verifier": raw_pin(VERIFIER_SOURCE),
        },
        "carrier_index": {
            "inherited_slots": list(range(INHERITED_COUNT)),
            "midpoint_slots": list(range(INHERITED_COUNT, FINE_COUNT)),
            "midpoint_rows": midpoints,
            "index_source": (
                "vertex_parent_support from the canonical level-one geodesic tower"
            ),
        },
        "labeled_event_instrument": {
            "baseline_label": "Q",
            "averaging_factor": "1/2",
            "directed_event_count": len(directed),
            "directed_events": directed,
            "fine_edge_membership_checks": len(directed),
            "all_directed_events_are_fine_edges": True,
            "minimal_selected_event_count": len(selected),
            "minimal_selected_event_labels": [event["label"] for event in selected],
            "minimal_selector": (
                "smaller inherited endpoint label on each parent edge; rank witness only"
            ),
            "same_pre_event_field_required": True,
            "repeated_preparation_or_checkpoint_access_required": True,
            "allowed_same_state_realizations": [
                "reset to the same prepared field",
                "parallel copies from a source-defined preparation",
                "a nondestructive labeled response instrument",
            ],
            "sequential_destructive_application_without_reset_sufficient": False,
            "response_used": (
                "the parent coordinate of each Q E response; the other eleven "
                "coordinates are redundant"
            ),
        },
        "exact_linear_certificate": linear,
        "symmetry_certificate": symmetry,
        "protocol_boundary": {
            "existing_passive_rank_29_result": {
                "parent": PARENT_RECEIPT.relative_to(REPOSITORY_ROOT).as_posix(),
                "matrix": "O=(Q,Q L_f,...,Q L_f^41)",
                "rank_over_Q": 29,
                "kernel_dimension": 13,
                "status": "unchanged and protocol-specific",
            },
            "why_no_contradiction": (
                "the active packet adds separately labeled Q E_(u,m) responses; "
                "they are not rows of the passive averaged-semigroup matrix"
            ),
            "rank_29_is_a_universal_readout_no_go": False,
            "labeled_event_grammar_selected_by_bare_OPH_axioms": False,
            "same_state_instrument_constructed": False,
            "physical_sky_readout_constructed": False,
            "laboratory_observable_constructed": False,
            "issue_closure_authorized": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def build_receipt() -> dict[str, Any]:
    report = _payload()
    report["payload_sha256"] = self_digest(report)
    return report


def verify_receipt(report: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        if report.get("payload_sha256") != self_digest(report):
            reasons.append("payload_hash_mismatch")
        if dict(report) != build_receipt():
            reasons.append("producer_replay_mismatch")
        boundary = report.get("protocol_boundary")
        if not isinstance(boundary, Mapping):
            reasons.append("protocol_boundary_missing")
        elif (
            boundary.get("rank_29_is_a_universal_readout_no_go") is not False
            or boundary.get("labeled_event_grammar_selected_by_bare_OPH_axioms")
            is not False
            or boundary.get("same_state_instrument_constructed") is not False
            or boundary.get("physical_sky_readout_constructed") is not False
            or boundary.get("laboratory_observable_constructed") is not False
            or boundary.get("issue_closure_authorized") is not False
        ):
            reasons.append("forbidden_protocol_or_physical_promotion")
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        reasons.append("malformed_or_unverifiable_payload")
    return {
        "schema": VERIFICATION_SCHEMA,
        "receipt": not reasons,
        "reasons": sorted(set(reasons)),
        "producer_replay": True,
        "independent_implementation": False,
    }


def write_receipt(path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    report = build_receipt()
    if verify_receipt(report)["receipt"] is not True:
        raise LabeledReadoutError("internal labeled-event replay failed")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json(report) + b"\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        report = json.loads(args.verify.read_text(encoding="ascii"))
        result = verify_receipt(report)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["receipt"] is True else 1
    report = write_receipt(args.output)
    print(report["status"])
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
