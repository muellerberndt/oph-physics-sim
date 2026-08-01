"""Independent verifier for the issue-643 refinement/repair packet.

This verifier does not import the producer.  It reconstructs the two mesh
levels, the graph Laplacian, the semantic detail vector, all finite repair
readbacks, the proper rotation orbit, and the exact degree-six pair statistic.
"""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.core.icosahedral import (
    build_geodesic_icosahedral_tower,
    icosahedral_a5_port_permutations,
)


SCHEMA = "oph.angular_refinement_repair_observability.v1"
VERIFICATION_SCHEMA = "oph.angular_refinement_repair_observability_independent_verification.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = (
    REPOSITORY_ROOT
    / "data/repair_closure/angular_refinement_repair_observability_receipt.json"
)
DETAIL = {
    (0, 11): -1,
    (5, 11): 1,
    (1, 7): -1,
    (0, 7): 1,
    (5, 9): -1,
    (1, 9): 1,
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _edges(mesh: Any) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted((min(int(left), int(right)), max(int(left), int(right))))
        for left, right in mesh.edges
    )


def _laplacian(vertex_count: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    matrix = [[0] * vertex_count for _ in range(vertex_count)]
    for left, right in edges:
        matrix[left][left] += 1
        matrix[right][right] += 1
        matrix[left][right] -= 1
        matrix[right][left] -= 1
    return matrix


def _matvec(matrix: Sequence[Sequence[int]], vector: Sequence[int]) -> list[int]:
    return [
        sum(entry * value for entry, value in zip(row, vector, strict=True))
        for row in matrix
    ]


def _observability_rank(
    laplacian: Sequence[Sequence[int]],
    inherited_count: int,
) -> int:
    dimension = len(laplacian)
    current = [
        [int(row == column) for column in range(dimension)]
        for row in range(inherited_count)
    ]
    transpose = tuple(zip(*laplacian, strict=True))
    rows: list[list[Fraction]] = []
    for _power in range(dimension):
        rows.extend([list(map(Fraction, row)) for row in current])
        current = [
            [
                sum(left * right for left, right in zip(row, column, strict=True))
                for column in transpose
            ]
            for row in current
        ]
    rank = 0
    for column in range(dimension):
        pivot = next(
            (index for index in range(rank, len(rows)) if rows[index][column]),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][column]
        rows[rank] = [value / scale for value in rows[rank]]
        for index in range(rank + 1, len(rows)):
            factor = rows[index][column]
            if factor:
                rows[index] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(rows[index], rows[rank], strict=True)
                ]
        rank += 1
    return rank


def _support_index(fine: Any) -> dict[tuple[int, ...], int]:
    return {
        tuple(sorted(int(parent) for parent, _weight in support)): index
        for index, support in enumerate(fine.vertex_parent_support)
    }


def _detail(fine: Any) -> list[int]:
    by_support = _support_index(fine)
    vector = [0] * fine.vertex_count
    for edge, coefficient in DETAIL.items():
        vector[by_support[edge]] = coefficient
    return vector


def _readback_is_zero(
    laplacian: Sequence[Sequence[int]],
    vector: Sequence[int],
    inherited_count: int,
) -> bool:
    state = list(vector)
    for _power in range(len(vector)):
        if any(state[:inherited_count]):
            return False
        state = _matvec(laplacian, state)
    return True


def _lift(fine: Any, base_permutation: Sequence[int]) -> tuple[int, ...]:
    by_support = _support_index(fine)
    return tuple(
        by_support[
            tuple(
                sorted(
                    int(base_permutation[int(parent)])
                    for parent, _weight in support
                )
            )
        ]
        for support in fine.vertex_parent_support
    )


def _rotate(vector: Sequence[int], permutation: Sequence[int]) -> tuple[int, ...]:
    result = [0] * len(vector)
    for source, target in enumerate(permutation):
        result[target] = vector[source]
    return tuple(result)


def _distances(vertex_count: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    adjacency = [set() for _ in range(vertex_count)]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    table: list[list[int]] = []
    for source in range(vertex_count):
        row = [-1] * vertex_count
        row[source] = 0
        frontier = [source]
        for node in frontier:
            for target in sorted(adjacency[node]):
                if row[target] < 0:
                    row[target] = row[node] + 1
                    frontier.append(target)
        table.append(row)
    return table


def _add(left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    return (left[0] + right[0], left[1] + right[1])


def _mul(left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    return (
        left[0] * right[0] + 5 * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _scale(value: tuple[Fraction, Fraction], scalar: int | Fraction) -> tuple[Fraction, Fraction]:
    return (value[0] * Fraction(scalar), value[1] * Fraction(scalar))


def _divide(left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    norm = right[0] * right[0] - 5 * right[1] * right[1]
    inverse = (right[0] / norm, -right[1] / norm)
    return _mul(left, inverse)


def _base_dot(distance: int) -> tuple[Fraction, Fraction]:
    return {
        0: (Fraction(1), Fraction()),
        1: (Fraction(), Fraction(1, 5)),
        2: (Fraction(), Fraction(-1, 5)),
        3: (Fraction(-1), Fraction()),
    }[distance]


def _midpoint_dot(
    left: tuple[int, int],
    right: tuple[int, int],
    distances: Sequence[Sequence[int]],
) -> tuple[Fraction, Fraction]:
    numerator = (Fraction(), Fraction())
    for first in left:
        for second in right:
            numerator = _add(numerator, _base_dot(distances[first][second]))
    return _divide(numerator, (Fraction(2), Fraction(2, 5)))


def _p6(value: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    square = _mul(value, value)
    fourth = _mul(square, square)
    sixth = _mul(fourth, square)
    return _scale(
        _add(
            _add(_scale(sixth, 231), _scale(fourth, -315)),
            _add(_scale(square, 105), (Fraction(-5), Fraction())),
        ),
        Fraction(1, 16),
    )


def _angular_value(coarse: Any, fine: Any, vector: Sequence[int]) -> Fraction:
    distances = _distances(coarse.vertex_count, _edges(coarse))
    supports = {
        index: tuple(sorted(int(parent) for parent, _weight in support))
        for index, support in enumerate(fine.vertex_parent_support)
        if len(support) == 2
    }
    total = (Fraction(), Fraction())
    active = [index for index, value in enumerate(vector) if value]
    for left in active:
        for right in active:
            term = _p6(_midpoint_dot(supports[left], supports[right], distances))
            total = _add(total, _scale(term, vector[left] * vector[right]))
    if total != (Fraction(945, 512), Fraction()):
        raise ValueError("unexpected exact degree-six detail pair sum")
    return Fraction(1, 4) * total[0] / (fine.vertex_count * fine.vertex_count)


def _check_pins(report: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    pins = report.get("source_pins")
    if not isinstance(pins, list):
        return ["source_pins_missing"]
    for pin in pins:
        if not isinstance(pin, Mapping) or not isinstance(pin.get("path"), str):
            reasons.append("source_pin_malformed")
            continue
        path = (REPOSITORY_ROOT / pin["path"]).resolve()
        try:
            path.relative_to(REPOSITORY_ROOT.resolve())
            raw = path.read_bytes()
        except (OSError, ValueError):
            reasons.append("source_pin_unreadable")
            continue
        if pin.get("bytes") != len(raw):
            reasons.append("source_pin_byte_count_mismatch")
        if pin.get("sha256") != "sha256:" + hashlib.sha256(raw).hexdigest():
            reasons.append("source_pin_hash_mismatch")
    return reasons


def verify(report: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        if report.get("schema") != SCHEMA:
            reasons.append("schema_mismatch")
        payload = copy.deepcopy(dict(report))
        stated_hash = payload.pop("payload_sha256", None)
        if stated_hash != _sha(payload):
            reasons.append("payload_hash_mismatch")
        reasons.extend(_check_pins(report))

        tower = build_geodesic_icosahedral_tower(1)
        coarse, fine = tower.levels
        laplacian = _laplacian(fine.vertex_count, _edges(fine))
        vector = _detail(fine)
        support_index = _support_index(fine)
        expected_rows = [
            {
                "parent_edge": list(edge),
                "fine_index": support_index[edge],
                "coefficient": coefficient,
            }
            for edge, coefficient in sorted(DETAIL.items())
        ]
        witness = report.get("detail_witness")
        if not isinstance(witness, Mapping) or witness.get("semantic_rows") != expected_rows:
            reasons.append("reported_detail_witness_mismatch")
        rank = _observability_rank(laplacian, coarse.vertex_count)
        if rank != 29 or fine.vertex_count - rank != 13:
            reasons.append("exact_observability_rank_mismatch")
        refinement = report.get("refinement_and_repair")
        observability = (
            refinement.get("observability_witness")
            if isinstance(refinement, Mapping)
            else None
        )
        if not isinstance(observability, Mapping) or (
            observability.get("observability_rank_over_Q") != rank
            or observability.get("repair_invisible_detail_dimension")
            != fine.vertex_count - rank
        ):
            reasons.append("reported_observability_rank_mismatch")
        if not _readback_is_zero(laplacian, vector, coarse.vertex_count):
            reasons.append("finite_repair_readback_witness_failed")
        if sum(vector) != 0 or sum(value * value for value in vector) != 6:
            reasons.append("detail_vector_invariants_failed")

        orbit = {
            _rotate(vector, _lift(fine, permutation))
            for permutation in icosahedral_a5_port_permutations()
        }
        if len(orbit) != 20:
            reasons.append("proper_orbit_size_mismatch")
        if any(sum(member[index] for member in orbit) for index in range(fine.vertex_count)):
            reasons.append("proper_orbit_mean_not_zero")
        if any(
            not _readback_is_zero(laplacian, member, coarse.vertex_count)
            for member in orbit
        ):
            reasons.append("rotated_detail_readback_visible")
        reported_orbit = report.get("A5_invariant_counterensemble")
        if not isinstance(reported_orbit, Mapping) or (
            reported_orbit.get("proper_rotation_count") != 60
            or reported_orbit.get("unique_detail_orbit_size") != len(orbit)
            or reported_orbit.get("uniform_orbit_mean_is_zero") is not True
        ):
            reasons.append("reported_orbit_certificate_mismatch")

        if _angular_value(coarse, fine, vector) != Fraction(15, 57344):
            reasons.append("normalized_degree_six_statistic_mismatch")
        statistic = report.get("normalized_statistic")
        if not isinstance(statistic, Mapping) or statistic.get("completion_b") != "15/57344":
            reasons.append("reported_degree_six_statistic_mismatch")
        elif statistic.get("support_measure_physical") is not False:
            reasons.append("forbidden_measure_promotion")
        decision = report.get("selection_decision")
        if not isinstance(decision, Mapping):
            reasons.append("selection_decision_missing")
        elif (
            decision.get("covariance_selected_by_shared_antecedents") is not False
            or decision.get("repair_schedule_source_selected") is not False
            or decision.get("unit_counting_measure_physical") is not False
            or decision.get("physical_sky_readout_selected") is not False
            or decision.get("physical_angular_prediction") is not False
            or decision.get("issue_closure_authorized") is not False
        ):
            reasons.append("forbidden_selection_or_physical_promotion")
    except (AttributeError, KeyError, OSError, TypeError, ValueError, ZeroDivisionError):
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
    report = json.loads(args.receipt.read_text("utf-8"))
    result = verify(report)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["receipt"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
