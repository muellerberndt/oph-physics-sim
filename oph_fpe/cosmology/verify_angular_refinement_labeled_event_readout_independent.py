"""Independent verifier for the level-one labeled-event readout packet.

The verifier does not import the producer.  It independently rebuilds the
canonical level-one carrier, event inventory, proper-A5 transport, exact
measurement matrices, ranks, decoder, and all-basis reconstruction.  Parent
and implementation byte pins are checked before the scientific payload is
accepted.
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
    "oph.angular_refinement_labeled_event_readout_independent_verification.v1"
)
STATUS = "EXACT_LEVEL_ONE_LABELED_EVENT_FULL_RECONSTRUCTION__PHYSICAL_INSTRUMENT_OPEN"
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
DEFAULT_RECEIPT = (
    REPOSITORY_ROOT
    / "data/repair_closure/angular_refinement_labeled_event_readout_receipt.json"
)
PARENT_PATH = "data/repair_closure/angular_refinement_repair_observability_receipt.json"
IMPLEMENTATION_PATHS = {
    "producer": "oph_fpe/cosmology/angular_refinement_labeled_event_readout.py",
    "independent_verifier": (
        "oph_fpe/cosmology/"
        "verify_angular_refinement_labeled_event_readout_independent.py"
    ),
}

N_BASE = 12
N_MID = 30
N_FINE = 42
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


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


def payload_digest(report: Mapping[str, Any]) -> str:
    body = {key: value for key, value in report.items() if key != "payload_sha256"}
    return tagged_sha256(canonical_json(body))


def raw_pin(repo_root: Path, relative: str) -> dict[str, Any]:
    path = (repo_root / relative).resolve(strict=True)
    path.relative_to(repo_root.resolve())
    payload = path.read_bytes()
    return {"path": relative, "bytes": len(payload), "sha256": tagged_sha256(payload)}


def unit(index: int, width: int = N_FINE) -> Vector:
    return tuple(Fraction(int(index == column)) for column in range(width))


def response(parent: int, midpoint: int) -> Vector:
    row = [Fraction()] * N_FINE
    row[parent] = Fraction(1, 2)
    row[midpoint] = Fraction(1, 2)
    return tuple(row)


def pair_event(parent: int, midpoint: int) -> Matrix:
    rows = [list(unit(index)) for index in range(N_FINE)]
    mean = list(response(parent, midpoint))
    rows[parent] = mean
    rows[midpoint] = mean
    return tuple(tuple(row) for row in rows)


def support_map(fine: Any) -> dict[tuple[int, ...], int]:
    result = {}
    for index, support in enumerate(fine.vertex_parent_support):
        parents = tuple(sorted(int(parent) for parent, _weight in support))
        if parents in result:
            raise ValueError("noninjective independent parent support")
        result[parents] = index
    return result


def carrier_and_events() -> tuple[
    Any,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    tower = build_geodesic_icosahedral_tower(1)
    coarse, fine = tower.levels
    if coarse.vertex_count != N_BASE or fine.vertex_count != N_FINE:
        raise ValueError("independent carrier size drift")
    coarse_edges = {tuple(sorted(map(int, edge))) for edge in coarse.edges}
    fine_edges = {tuple(sorted(map(int, edge))) for edge in fine.edges}
    by_support = support_map(fine)
    midpoint_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    directed: list[dict[str, Any]] = []
    for midpoint in range(N_BASE, N_FINE):
        parents = tuple(
            sorted(
                int(parent) for parent, _weight in fine.vertex_parent_support[midpoint]
            )
        )
        if (
            len(parents) != 2
            or parents not in coarse_edges
            or by_support[parents] != midpoint
        ):
            raise ValueError("independent midpoint support drift")
        midpoint_rows.append(
            {
                "midpoint_carrier_slot": midpoint,
                "midpoint_label": f"m{midpoint:02d}",
                "parent_edge": list(parents),
            }
        )
        local = []
        for parent in parents:
            other = parents[1] if parent == parents[0] else parents[0]
            local.append(
                {
                    "label": f"avg:p{parent:02d}:m{midpoint:02d}",
                    "parent_inherited_slot": parent,
                    "other_inherited_slot": other,
                    "midpoint_carrier_slot": midpoint,
                    "parent_edge": list(parents),
                    "averaging_factor": "1/2",
                    "selected_Q_coordinate": parent,
                    "response_sparse": {
                        str(parent): "1/2",
                        str(midpoint): "1/2",
                    },
                }
            )
        directed.extend(local)
        selected.append(local[0])
        for event in local:
            event_edge = tuple(
                sorted(
                    (
                        int(event["parent_inherited_slot"]),
                        int(event["midpoint_carrier_slot"]),
                    )
                )
            )
            if event_edge not in fine_edges:
                raise ValueError("independent directed event is not a fine edge")
    if len(midpoint_rows) != N_MID or len(directed) != 60 or len(selected) != N_MID:
        raise ValueError("independent event inventory size drift")
    return fine, midpoint_rows, selected, directed


def lift(fine: Any, base: Sequence[int]) -> tuple[int, ...]:
    by_support = support_map(fine)
    result = []
    for support in fine.vertex_parent_support:
        image_support = tuple(
            sorted(int(base[int(parent)]) for parent, _weight in support)
        )
        result.append(by_support[image_support])
    if len(set(result)) != N_FINE:
        raise ValueError("independent A5 lift is not a permutation")
    return tuple(result)


def symmetry(fine: Any, directed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rotations = tuple(icosahedral_a5_port_permutations())
    keys = {
        (int(event["parent_inherited_slot"]), int(event["midpoint_carrier_slot"]))
        for event in directed
    }
    if len(rotations) != 60 or len(set(rotations)) != 60 or len(keys) != 60:
        raise ValueError("independent A5 or event order drift")
    representative = min(keys)
    orbit = set()
    checks = 0
    for rotation in rotations:
        lifted = lift(fine, rotation)
        for parent, midpoint in keys:
            if (int(rotation[parent]), int(lifted[midpoint])) not in keys:
                raise ValueError("independent event covariance failed")
            checks += 1
        orbit.add((int(rotation[representative[0]]), int(lifted[representative[1]])))
    if len(orbit) != 60:
        raise ValueError("independent directed event orbit is not transitive")
    return {
        "proper_carrier_group": "A5",
        "proper_rotation_count": 60,
        "directed_event_count": 60,
        "directed_event_covariance_checks": checks,
        "representative_directed_event": list(representative),
        "representative_orbit_size": 60,
        "directed_event_family_is_one_A5_orbit": True,
        "directed_event_family_is_A5_stable": True,
        "minimal_thirty_event_selector_is_A5_invariant": False,
        "no_A5_equivariant_one_parent_per_midpoint_section": True,
    }


def rank(matrix: Sequence[Sequence[Fraction]]) -> int:
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("independent rank matrix is ragged")
    result = 0
    for column in range(width):
        pivot = next(
            (row for row in range(result, len(rows)) if rows[row][column]),
            None,
        )
        if pivot is None:
            continue
        rows[result], rows[pivot] = rows[pivot], rows[result]
        scale = rows[result][column]
        rows[result] = [value / scale for value in rows[result]]
        for row in range(result + 1, len(rows)):
            factor = rows[row][column]
            if factor:
                rows[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(rows[row], rows[result], strict=True)
                ]
        result += 1
        if result == len(rows):
            break
    return result


def matmul(
    left: Sequence[Sequence[Fraction]],
    right: Sequence[Sequence[Fraction]],
) -> Matrix:
    if len(left[0]) != len(right):
        raise ValueError("independent matrix product shape mismatch")
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


def matrix_sha(matrix: Sequence[Sequence[Fraction]]) -> str:
    return tagged_sha256(
        canonical_json([[str(value) for value in row] for row in matrix])
    )


def matrices(
    selected: Sequence[Mapping[str, Any]],
    directed: Sequence[Mapping[str, Any]],
) -> dict[str, Matrix]:
    baseline = tuple(unit(index) for index in range(N_BASE))
    selected_rows = tuple(
        response(
            int(event["parent_inherited_slot"]), int(event["midpoint_carrier_slot"])
        )
        for event in selected
    )
    directed_rows = tuple(
        response(
            int(event["parent_inherited_slot"]), int(event["midpoint_carrier_slot"])
        )
        for event in directed
    )
    minimal = baseline + selected_rows
    symmetric = baseline + directed_rows
    decoder_rows: list[list[Fraction]] = []
    for inherited in range(N_BASE):
        row = [Fraction()] * N_FINE
        row[inherited] = 1
        decoder_rows.append(row)
    for offset, event in enumerate(selected):
        row = [Fraction()] * N_FINE
        row[int(event["parent_inherited_slot"])] = -1
        row[N_BASE + offset] = 2
        decoder_rows.append(row)
    decoder = tuple(tuple(row) for row in decoder_rows)

    selected_full = list(baseline)
    for event in selected:
        matrix = pair_event(
            int(event["parent_inherited_slot"]),
            int(event["midpoint_carrier_slot"]),
        )
        selected_full.extend(matrix[:N_BASE])
    directed_full = list(baseline)
    for event in directed:
        matrix = pair_event(
            int(event["parent_inherited_slot"]),
            int(event["midpoint_carrier_slot"]),
        )
        directed_full.extend(matrix[:N_BASE])
    return {
        "baseline": baseline,
        "minimal": minimal,
        "directed_only": directed_rows,
        "symmetric": symmetric,
        "decoder": decoder,
        "selected_full": tuple(selected_full),
        "directed_full": tuple(directed_full),
    }


def check_parent_and_pins(report: Mapping[str, Any], repo_root: Path) -> list[str]:
    reasons: list[str] = []
    expected_parent = [raw_pin(repo_root, PARENT_PATH)]
    if report.get("parent_pins") != expected_parent:
        reasons.append("parent_pin_mismatch")

    implementation = report.get("implementation_integrity_pins")
    if not isinstance(implementation, Mapping):
        reasons.append("implementation_pins_missing")
    else:
        scope = str(implementation.get("scope"))
        if "not authenticated external custody" not in scope:
            reasons.append("implementation_pin_scope_mismatch")
        for role, relative in IMPLEMENTATION_PATHS.items():
            if implementation.get(role) != raw_pin(repo_root, relative):
                reasons.append("implementation_pin_mismatch")

    parent = json.loads((repo_root / PARENT_PATH).read_text(encoding="ascii"))
    if parent.get("schema") != "oph.angular_refinement_repair_observability.v1":
        reasons.append("passive_parent_schema_mismatch")
    witness = parent.get("refinement_and_repair", {}).get("observability_witness", {})
    if (
        not isinstance(witness, Mapping)
        or witness.get("observability_matrix_shape") != [504, 42]
        or witness.get("observability_rank_over_Q") != 29
        or witness.get("repair_invisible_detail_dimension") != 13
    ):
        reasons.append("passive_parent_rank_mismatch")
    return reasons


def verify(
    report: Mapping[str, Any], repo_root: Path = REPOSITORY_ROOT
) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        expected_keys = {
            "schema",
            "issue",
            "status",
            "source_scope",
            "parent_pins",
            "implementation_integrity_pins",
            "carrier_index",
            "labeled_event_instrument",
            "exact_linear_certificate",
            "symmetry_certificate",
            "protocol_boundary",
            "claim_boundary",
            "payload_sha256",
        }
        if set(report) != expected_keys:
            reasons.append("top_level_inventory_mismatch")
        if report.get("schema") != SCHEMA:
            reasons.append("schema_mismatch")
        if report.get("issue") != 643 or report.get("status") != STATUS:
            reasons.append("identity_or_status_mismatch")
        expected_source_scope = {
            "coarse_level": 0,
            "fine_level": 1,
            "inherited_vertex_count": N_BASE,
            "midpoint_vertex_count": N_MID,
            "fine_vertex_count": N_FINE,
            "base_readout": "Q restricts to the twelve inherited vertices",
            "event_operator": (
                "E_(u,m) replaces the inherited endpoint u and midpoint m by "
                "their exact arithmetic mean and fixes every other fine slot"
            ),
            "external_comparison_data_used": False,
            "target_values_used": False,
        }
        if report.get("source_scope") != expected_source_scope:
            reasons.append("source_scope_mismatch")
        if report.get("claim_boundary") != CLAIM_BOUNDARY:
            reasons.append("claim_boundary_mismatch")
        if report.get("payload_sha256") != payload_digest(report):
            reasons.append("payload_hash_mismatch")
        reasons.extend(check_parent_and_pins(report, Path(repo_root)))

        fine, midpoint_rows, selected, directed = carrier_and_events()
        expected_carrier = {
            "inherited_slots": list(range(N_BASE)),
            "midpoint_slots": list(range(N_BASE, N_FINE)),
            "midpoint_rows": midpoint_rows,
            "index_source": "vertex_parent_support from the canonical level-one geodesic tower",
        }
        if report.get("carrier_index") != expected_carrier:
            reasons.append("carrier_index_mismatch")

        instrument = report.get("labeled_event_instrument")
        reported_directed: Any = None
        if not isinstance(instrument, Mapping):
            reasons.append("event_instrument_missing")
        else:
            expected_instrument = {
                "baseline_label": "Q",
                "averaging_factor": "1/2",
                "directed_event_count": 60,
                "directed_events": directed,
                "fine_edge_membership_checks": 60,
                "all_directed_events_are_fine_edges": True,
                "minimal_selected_event_count": 30,
                "minimal_selected_event_labels": [event["label"] for event in selected],
                "minimal_selector": (
                    "smaller inherited endpoint label on each parent edge; "
                    "rank witness only"
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
            }
            if dict(instrument) != expected_instrument:
                reasons.append("event_instrument_metadata_mismatch")
            reported_directed = instrument.get("directed_events")
            if instrument.get("directed_event_count") != 60:
                reasons.append("event_label_or_inventory_mismatch")
            if (
                instrument.get("fine_edge_membership_checks") != 60
                or instrument.get("all_directed_events_are_fine_edges") is not True
            ):
                reasons.append("fine_edge_membership_mismatch")
            if instrument.get("minimal_selected_event_count") != 30:
                reasons.append("event_label_or_inventory_mismatch")
            if instrument.get("minimal_selected_event_labels") != [
                event["label"] for event in selected
            ]:
                reasons.append("event_label_or_inventory_mismatch")
            if instrument.get("averaging_factor") != "1/2":
                reasons.append("averaging_factor_mismatch")
            if (
                instrument.get("same_pre_event_field_required") is not True
                or instrument.get("repeated_preparation_or_checkpoint_access_required")
                is not True
                or instrument.get(
                    "sequential_destructive_application_without_reset_sufficient"
                )
                is not False
            ):
                reasons.append("same_state_protocol_mismatch")

        if not isinstance(reported_directed, list) or len(reported_directed) != 60:
            reasons.extend(
                ["event_label_or_inventory_mismatch", "event_response_mismatch"]
            )
        else:
            labels = [event.get("label") for event in reported_directed]
            if (
                labels != [event["label"] for event in directed]
                or len(set(labels)) != 60
            ):
                reasons.append("event_label_or_inventory_mismatch")
            for received, expected in zip(reported_directed, directed, strict=True):
                if received.get("averaging_factor") != "1/2":
                    reasons.append("averaging_factor_mismatch")
                if received.get("response_sparse") != expected["response_sparse"]:
                    reasons.append("event_response_mismatch")
                if received != expected:
                    reasons.append("event_semantics_mismatch")
            fine_edges = {tuple(sorted(map(int, edge))) for edge in fine.edges}
            for received in reported_directed:
                try:
                    event_edge = tuple(
                        sorted(
                            (
                                int(received["parent_inherited_slot"]),
                                int(received["midpoint_carrier_slot"]),
                            )
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    reasons.append("fine_edge_membership_mismatch")
                    continue
                if event_edge not in fine_edges:
                    reasons.append("fine_edge_membership_mismatch")

        expected_protocol_boundary = {
            "existing_passive_rank_29_result": {
                "parent": PARENT_PATH,
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
        }
        if report.get("protocol_boundary") != expected_protocol_boundary:
            reasons.append("protocol_boundary_mismatch")

        matrix = matrices(selected, directed)
        identity = tuple(unit(index) for index in range(N_FINE))
        if matmul(matrix["decoder"], matrix["minimal"]) != identity:
            reasons.append("decoder_identity_failed")
        basis_checks = 0
        for index in range(N_FINE):
            basis = unit(index)
            if matvec(matrix["decoder"], matvec(matrix["minimal"], basis)) != basis:
                reasons.append("basis_reconstruction_failed")
            basis_checks += 1
        computed = {
            "baseline_Q_shape": [len(matrix["baseline"]), N_FINE],
            "baseline_Q_rank_over_Q": rank(matrix["baseline"]),
            "minimal_selected_scalar_shape": [len(matrix["minimal"]), N_FINE],
            "minimal_selected_scalar_rank_over_Q": rank(matrix["minimal"]),
            "minimal_selected_scalar_matrix_sha256": matrix_sha(matrix["minimal"]),
            "decoder_shape": [len(matrix["decoder"]), len(matrix["decoder"][0])],
            "decoder_matrix_sha256": matrix_sha(matrix["decoder"]),
            "decoder_times_measurement_is_identity": True,
            "all_standard_basis_reconstruction_checks": basis_checks,
            "A5_stable_scalar_shape": [len(matrix["symmetric"]), N_FINE],
            "A5_stable_scalar_rank_over_Q": rank(matrix["symmetric"]),
            "A5_stable_scalar_matrix_sha256": matrix_sha(matrix["symmetric"]),
            "directed_event_rows_without_Q_shape": [
                len(matrix["directed_only"]),
                N_FINE,
            ],
            "directed_event_rows_without_Q_rank_over_Q": rank(matrix["directed_only"]),
            "directed_event_rows_without_Q_kernel_dimension": (
                N_FINE - rank(matrix["directed_only"])
            ),
            "selected_full_Q_stack_shape": [len(matrix["selected_full"]), N_FINE],
            "selected_full_Q_stack_rank_over_Q": rank(matrix["selected_full"]),
            "A5_stable_full_Q_stack_shape": [len(matrix["directed_full"]), N_FINE],
            "A5_stable_full_Q_stack_rank_over_Q": rank(matrix["directed_full"]),
        }
        exact = report.get("exact_linear_certificate")
        if not isinstance(exact, Mapping):
            reasons.append("linear_certificate_missing")
        else:
            for key, value in computed.items():
                if exact.get(key) != value:
                    reasons.append("reported_linear_certificate_mismatch")
            if (
                exact.get("arithmetic")
                != "exact rational Fraction arithmetic; no floating point"
            ):
                reasons.append("arithmetic_scope_mismatch")
            if exact.get("inherited_reconstruction") != "x_u = (Q x)_u":
                reasons.append("reconstruction_formula_mismatch")
            if "2 (Q E_(u,m) x)_u - (Q x)_u" not in str(
                exact.get("midpoint_reconstruction")
            ):
                reasons.append("reconstruction_formula_mismatch")

        expected_symmetry = symmetry(fine, directed)
        received_symmetry = report.get("symmetry_certificate")
        if not isinstance(received_symmetry, Mapping):
            reasons.append("symmetry_certificate_missing")
        else:
            for key, value in expected_symmetry.items():
                if received_symmetry.get(key) != value:
                    reasons.append("symmetry_certificate_mismatch")
            if "one A5 orbit" not in str(received_symmetry.get("no_section_reason")):
                reasons.append("symmetry_certificate_mismatch")

        boundary = report.get("protocol_boundary")
        if not isinstance(boundary, Mapping):
            reasons.append("protocol_boundary_missing")
        else:
            passive = boundary.get("existing_passive_rank_29_result")
            if not isinstance(passive, Mapping) or (
                passive.get("parent") != PARENT_PATH
                or passive.get("matrix") != "O=(Q,Q L_f,...,Q L_f^41)"
                or passive.get("rank_over_Q") != 29
                or passive.get("kernel_dimension") != 13
                or passive.get("status") != "unchanged and protocol-specific"
            ):
                reasons.append("passive_rank_29_scope_mismatch")
            for key in (
                "rank_29_is_a_universal_readout_no_go",
                "labeled_event_grammar_selected_by_bare_OPH_axioms",
                "same_state_instrument_constructed",
                "physical_sky_readout_constructed",
                "laboratory_observable_constructed",
                "issue_closure_authorized",
            ):
                if boundary.get(key) is not False:
                    reasons.append("forbidden_protocol_or_physical_promotion")
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        reasons.append("malformed_or_unverifiable_payload")

    return {
        "schema": VERIFICATION_SCHEMA,
        "receipt": not reasons,
        "reasons": sorted(set(reasons)),
        "independent_implementation": True,
        "producer_imported": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, nargs="?", default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)
    report = json.loads(args.receipt.read_text(encoding="ascii"))
    result = verify(report)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["receipt"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
