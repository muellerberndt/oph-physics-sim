"""Exact refinement/repair observability packet for issue 643.

The static twelve-port counterfamily does not test the registered refinement
and repair structure.  This packet uses the level-zero to level-one geodesic
refinement, its midpoint interpolation ``J``, inherited-vertex restriction
``Q``, and the canonical fine graph repair generator.

An explicit integer detail vector ``v`` is supported on six level-one
midpoints and obeys ``Q L_f^k v = 0`` for every ``0 <= k < 42``.  The
Cayley-Hamilton theorem then gives the identity for every polynomial and the
whole analytic repair semigroup.  The two positive fine fields ``J 1`` and
``J 1 + v/2`` therefore have the same coarse field and the same inherited-port
repair trace at every time.  Their centered degree-six Legendre pair moments
are respectively zero and ``15/57344``.  Averaging the second completion over
its proper A5 orbit leaves the mean field constant and retains the nonzero
covariance statistic.

This is a bounded source-side result.  Inherited-vertex restriction is a
registered refinement scaffold, not a selected physical sky readout.  The
packet identifies the missing object as a source-selected probability law on
the repair-invisible refinement-detail fiber, together with its physical
readout binding.  It does not use comparison data and does not promote an
angular prediction.
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
from oph_fpe.dynamics.canonical_seam_repair import graph_laplacian


SCHEMA = "oph.angular_refinement_repair_observability.v1"
VERIFICATION_SCHEMA = "oph.angular_refinement_repair_observability_verification.v1"
STATUS = (
    "EXACT_REFINEMENT_REPAIR_COUNTERENSEMBLE__"
    "DETAIL_COVARIANCE_UNSELECTED__PHYSICAL_SKY_READOUT_OPEN"
)
ISSUE = 643
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "data/repair_closure/angular_refinement_repair_observability_receipt.json"
)

Q5 = tuple[Fraction, Fraction]
Matrix = tuple[tuple[int, ...], ...]
Vector = tuple[int, ...]

# The witness is named by parent-edge support, not by incidental fine indices.
DETAIL_BY_PARENT_EDGE: dict[tuple[int, int], int] = {
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


def _raw_pin(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    resolved.relative_to(REPOSITORY_ROOT.resolve())
    raw = resolved.read_bytes()
    return {
        "path": resolved.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _matvec(matrix: Sequence[Sequence[int]], vector: Sequence[int]) -> Vector:
    return tuple(
        sum(entry * value for entry, value in zip(row, vector, strict=True))
        for row in matrix
    )


def _observability_rows(
    laplacian: Matrix,
    inherited_count: int,
) -> list[list[int]]:
    dimension = len(laplacian)
    current = [
        [int(row == column) for column in range(dimension)]
        for row in range(inherited_count)
    ]
    rows: list[list[int]] = []
    transposed = tuple(zip(*laplacian, strict=True))
    for _power in range(dimension):
        rows.extend(current)
        current = [
            [
                sum(left * right for left, right in zip(row, column, strict=True))
                for column in transposed
            ]
            for row in current
        ]
    return rows


def _rank_over_q(matrix: Sequence[Sequence[int]]) -> int:
    rows = [[Fraction(value) for value in row] for row in matrix]
    if not rows:
        return 0
    rank = 0
    width = len(rows[0])
    for column in range(width):
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


def _parent_support_index(fine: Any) -> dict[tuple[int, ...], int]:
    result: dict[tuple[int, ...], int] = {}
    for index, support in enumerate(fine.vertex_parent_support):
        parents = tuple(sorted(int(parent) for parent, _weight in support))
        if parents in result:
            raise ValueError("refinement parent support is not injective")
        result[parents] = index
    return result


def _detail_vector(fine: Any) -> Vector:
    by_support = _parent_support_index(fine)
    vector = [0] * fine.vertex_count
    for edge, coefficient in DETAIL_BY_PARENT_EDGE.items():
        index = by_support.get(tuple(sorted(edge)))
        if index is None:
            raise ValueError(f"missing level-one midpoint support {edge}")
        vector[index] = coefficient
    return tuple(vector)


def _refinement_interpolation(fine: Any, coarse_count: int) -> tuple[tuple[Fraction, ...], ...]:
    rows: list[tuple[Fraction, ...]] = []
    for support in fine.vertex_parent_support:
        row = [Fraction()] * coarse_count
        weight = Fraction(1, len(support))
        for parent, _stored_weight in support:
            row[int(parent)] += weight
        if sum(row, Fraction()) != 1:
            raise ValueError("a refinement interpolation row is not normalized")
        rows.append(tuple(row))
    return tuple(rows)


def _apply_fraction_matrix(
    matrix: Sequence[Sequence[Fraction]],
    vector: Sequence[Fraction],
) -> tuple[Fraction, ...]:
    return tuple(
        sum((entry * value for entry, value in zip(row, vector, strict=True)), Fraction())
        for row in matrix
    )


def _repair_observability_witness(
    laplacian: Matrix,
    detail: Vector,
    inherited_count: int,
    *,
    compute_rank: bool = True,
) -> dict[str, Any]:
    observability_rows = (
        _observability_rows(laplacian, inherited_count) if compute_rank else []
    )
    observability_rank = _rank_over_q(observability_rows) if compute_rank else None
    kernel_dimension = (
        len(detail) - observability_rank if observability_rank is not None else None
    )
    if compute_rank and (observability_rank != 29 or kernel_dimension != 13):
        raise ValueError("the exact repair observability rank changed")
    state = detail
    maximum_inherited_absolute_value = 0
    state_norms = []
    for power in range(len(detail)):
        inherited = state[:inherited_count]
        maximum_inherited_absolute_value = max(
            maximum_inherited_absolute_value,
            *(abs(value) for value in inherited),
        )
        state_norms.append(sum(value * value for value in state))
        state = _matvec(laplacian, state)
    if maximum_inherited_absolute_value != 0:
        raise ValueError("the detail vector is visible in a finite repair power")
    if not any(detail):
        raise ValueError("the detail witness is zero")
    return {
        "powers_checked": len(detail),
        "power_range": [0, len(detail) - 1],
        "maximum_inherited_absolute_value": maximum_inherited_absolute_value,
        "all_checked_readbacks_zero": True,
        "observability_matrix_shape": (
            [len(observability_rows), len(detail)] if compute_rank else None
        ),
        "observability_rank_over_Q": observability_rank,
        "repair_invisible_detail_dimension": kernel_dimension,
        "first_state_squared_norm": state_norms[0],
        "last_checked_state_squared_norm": state_norms[-1],
        "all_polynomial_times": True,
        "all_analytic_semigroup_times": True,
        "extension_argument": (
            "Cayley-Hamilton reduces every higher power of the 42 by 42 "
            "fine Laplacian to the checked powers; the exponential series "
            "therefore has zero inherited readback on the detail vector"
        ),
    }


def _q5(value: int | Fraction = 0, radical: int | Fraction = 0) -> Q5:
    return (Fraction(value), Fraction(radical))


def _q5_add(left: Q5, right: Q5) -> Q5:
    return (left[0] + right[0], left[1] + right[1])


def _q5_mul(left: Q5, right: Q5) -> Q5:
    return (
        left[0] * right[0] + 5 * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _q5_inv(value: Q5) -> Q5:
    norm = value[0] * value[0] - 5 * value[1] * value[1]
    if norm == 0:
        raise ZeroDivisionError("zero in Q(sqrt5)")
    return (value[0] / norm, -value[1] / norm)


def _q5_div(left: Q5, right: Q5) -> Q5:
    return _q5_mul(left, _q5_inv(right))


def _q5_scale(value: Q5, scalar: int | Fraction) -> Q5:
    factor = Fraction(scalar)
    return (factor * value[0], factor * value[1])


def _legendre_six(value: Q5) -> Q5:
    # P6(x) = (231 x^6 - 315 x^4 + 105 x^2 - 5) / 16.
    square = _q5_mul(value, value)
    fourth = _q5_mul(square, square)
    sixth = _q5_mul(fourth, square)
    numerator = _q5_add(
        _q5_add(_q5_scale(sixth, 231), _q5_scale(fourth, -315)),
        _q5_add(_q5_scale(square, 105), _q5(-5)),
    )
    return _q5_scale(numerator, Fraction(1, 16))


def _graph_distances(vertex_count: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    adjacency = [set() for _ in range(vertex_count)]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    distances: list[list[int]] = []
    for source in range(vertex_count):
        row = [-1] * vertex_count
        row[source] = 0
        frontier = [source]
        while frontier:
            node = frontier.pop(0)
            for target in sorted(adjacency[node]):
                if row[target] < 0:
                    row[target] = row[node] + 1
                    frontier.append(target)
        distances.append(row)
    return distances


def _base_unit_dot(distance: int) -> Q5:
    if distance == 0:
        return _q5(1)
    if distance == 1:
        return _q5(0, Fraction(1, 5))
    if distance == 2:
        return _q5(0, Fraction(-1, 5))
    if distance == 3:
        return _q5(-1)
    raise ValueError("unexpected icosahedron graph distance")


def _midpoint_dot(
    left_edge: tuple[int, int],
    right_edge: tuple[int, int],
    distances: Sequence[Sequence[int]],
) -> Q5:
    numerator = _q5()
    for left in left_edge:
        for right in right_edge:
            numerator = _q5_add(
                numerator,
                _base_unit_dot(distances[left][right]),
            )
    # Each adjacent endpoint sum has squared norm 2 + 2/sqrt(5).
    denominator = _q5(2, Fraction(2, 5))
    return _q5_div(numerator, denominator)


def _degree_six_statistic(
    coarse: Any,
    fine: Any,
    detail: Vector,
) -> dict[str, Any]:
    edges = tuple(
        sorted((min(int(left), int(right)), max(int(left), int(right))))
        for left, right in coarse.edges
    )
    distances = _graph_distances(coarse.vertex_count, edges)
    support_by_index = {
        index: tuple(sorted(int(parent) for parent, _weight in support))
        for index, support in enumerate(fine.vertex_parent_support)
        if len(support) == 2
    }
    raw = _q5()
    nonzero = [index for index, value in enumerate(detail) if value]
    for left in nonzero:
        for right in nonzero:
            kernel = _legendre_six(
                _midpoint_dot(
                    support_by_index[left],
                    support_by_index[right],
                    distances,
                )
            )
            raw = _q5_add(raw, _q5_scale(kernel, detail[left] * detail[right]))
    if raw[1] != 0 or raw[0] != Fraction(945, 512):
        raise ValueError("the centered degree-six detail power changed")
    epsilon = Fraction(1, 2)
    total = Fraction(fine.vertex_count)
    normalized = epsilon * epsilon * raw[0] / (total * total)
    if normalized != Fraction(15, 57344):
        raise ValueError("the normalized degree-six statistic changed")
    return {
        "definition": (
            "S6(w)=(sum_ab delta_w[a] delta_w[b] "
            "P6(x_a dot x_b))/(sum_a w[a])^2"
        ),
        "field_centering": "delta_w = w - ensemble_mean",
        "support_measure": "unit counting on the 42 registered refinement vertices",
        "support_measure_physical": False,
        "detail_pair_sum": str(raw[0]),
        "completion_a": "0",
        "completion_b": str(normalized),
        "completion_b_strictly_positive": normalized > 0,
        "orientation_invariant": True,
        "statistic_type": "centered two-point covariance power at angular degree six",
    }


def _lift_rotation(
    fine: Any,
    base_permutation: Sequence[int],
) -> tuple[int, ...]:
    by_support = _parent_support_index(fine)
    lifted = []
    for support in fine.vertex_parent_support:
        mapped = tuple(
            sorted(int(base_permutation[int(parent)]) for parent, _weight in support)
        )
        if mapped not in by_support:
            raise ValueError("base rotation did not lift to the refinement")
        lifted.append(by_support[mapped])
    if len(set(lifted)) != fine.vertex_count:
        raise ValueError("lifted refinement rotation is not a permutation")
    return tuple(lifted)


def _permute_vector(vector: Vector, permutation: Sequence[int]) -> Vector:
    result = [0] * len(vector)
    for source, target in enumerate(permutation):
        result[int(target)] = vector[source]
    return tuple(result)


def _orbit_certificate(
    fine: Any,
    laplacian: Matrix,
    detail: Vector,
    inherited_count: int,
) -> dict[str, Any]:
    orbit = []
    for base_permutation in icosahedral_a5_port_permutations():
        lifted = _lift_rotation(fine, base_permutation)
        rotated = _permute_vector(detail, lifted)
        _repair_observability_witness(
            laplacian,
            rotated,
            inherited_count,
            compute_rank=False,
        )
        orbit.append(rotated)
    unique = sorted(set(orbit))
    ensemble_sum = [sum(vector[index] for vector in unique) for index in range(len(detail))]
    if any(ensemble_sum):
        raise ValueError("the proper-orbit detail mean is not zero")
    return {
        "proper_rotation_count": len(orbit),
        "unique_detail_orbit_size": len(unique),
        "uniform_orbit_mean_is_zero": True,
        "every_orbit_member_repair_invisible": True,
        "ensemble_mean_field": "constant one",
        "ensemble_covariance_nonzero": True,
        "ensemble_is_A5_invariant": True,
    }


def _payload() -> dict[str, Any]:
    tower = build_geodesic_icosahedral_tower(1)
    coarse, fine = tower.levels
    fine_edges = tuple(
        sorted((min(int(left), int(right)), max(int(left), int(right))))
        for left, right in fine.edges
    )
    laplacian = graph_laplacian(fine.vertex_count, fine_edges)
    detail = _detail_vector(fine)
    interpolation = _refinement_interpolation(fine, coarse.vertex_count)
    coarse_constant = (Fraction(1),) * coarse.vertex_count
    fine_constant = _apply_fraction_matrix(interpolation, coarse_constant)
    if fine_constant != (Fraction(1),) * fine.vertex_count:
        raise ValueError("midpoint interpolation does not preserve constants")
    if any(detail[index] for index in range(coarse.vertex_count)):
        raise ValueError("the detail vector changes an inherited vertex")
    completion_b = tuple(Fraction(1) + Fraction(value, 2) for value in detail)
    if min(completion_b) != Fraction(1, 2) or sum(completion_b) != fine.vertex_count:
        raise ValueError("the second completion is not positive and mass preserving")

    observability = _repair_observability_witness(
        laplacian,
        detail,
        coarse.vertex_count,
    )
    angular = _degree_six_statistic(coarse, fine, detail)
    orbit = _orbit_certificate(fine, laplacian, detail, coarse.vertex_count)
    support_index = _parent_support_index(fine)
    witness_rows = [
        {
            "parent_edge": list(edge),
            "fine_index": support_index[edge],
            "coefficient": coefficient,
        }
        for edge, coefficient in sorted(DETAIL_BY_PARENT_EDGE.items())
    ]

    return {
        "schema": SCHEMA,
        "issue": ISSUE,
        "status": STATUS,
        "source_scope": {
            "coarse_level": 0,
            "fine_level": 1,
            "coarse_vertex_count": coarse.vertex_count,
            "fine_vertex_count": fine.vertex_count,
            "fine_seam_count": len(fine_edges),
            "interpolation": "midpoint parent average J",
            "coarse_readout": "restriction Q to the twelve inherited vertices",
            "repair_generator": "K_f=L_f/(2|E_f|)",
            "repair_schedule": (
                "declared uniform IID expectation branch; its Markov extension "
                "is not derived by bare A3"
            ),
            "external_comparison_data_used": False,
            "target_values_used": False,
        },
        "source_pins": [
            _raw_pin(REPOSITORY_ROOT / "oph_fpe/core/icosahedral.py"),
            _raw_pin(REPOSITORY_ROOT / "oph_fpe/dynamics/canonical_seam_repair.py"),
            _raw_pin(REPOSITORY_ROOT / "docs/CANONICAL_REPAIR_LAW.md"),
        ],
        "detail_witness": {
            "semantic_rows": witness_rows,
            "nonzero_entry_count": sum(1 for value in detail if value),
            "coefficient_sum": sum(detail),
            "squared_norm": sum(value * value for value in detail),
            "inherited_vertex_entries_zero": True,
        },
        "refinement_and_repair": {
            "Q_after_J_is_identity_on_the_coarse_constant": True,
            "completion_a": "J 1",
            "completion_b": "J 1 + v/2",
            "both_coarse_grain_to_constant_one": True,
            "both_have_total_mass_42": True,
            "completion_b_minimum_value": "1/2",
            "both_produce_identical_inherited_repair_traces": True,
            "repair_trace_type": "uniform-schedule expectation semigroup",
            "iid_schedule_source_selected": False,
            "individual_seam_microhistories_identical": False,
            "individual_seam_microhistory_tested": False,
            "observability_witness": observability,
            "scope_boundary": (
                "the identity is state-specific; the full fine repair semigroup "
                "does not intertwine with midpoint refinement on every field"
            ),
        },
        "A5_invariant_counterensemble": orbit,
        "normalized_statistic": angular,
        "selection_decision": {
            "mean_field_agrees_between_completions": "constant one",
            "mean_field_globally_source_selected": False,
            "covariance_selected_by_shared_antecedents": False,
            "higher_correlation_selected_by_shared_antecedents": False,
            "repair_schedule_source_selected": False,
            "unit_counting_measure_physical": False,
            "smallest_missing_source_object": (
                "a source-selected refinement-detail state/readout instrument "
                "that fixes the probability law on the averaged-semigroup "
                "detail fiber and binds its output to the physical sky field"
            ),
            "physical_sky_readout_selected": False,
            "physical_angular_prediction": False,
            "issue_closure_authorized": False,
        },
        "claim_boundary": (
            "The registered midpoint refinement, inherited-vertex readout, "
            "and declared uniform-schedule repair generator admit two positive completions "
            "with identical coarse repair histories and different exact "
            "degree-six covariance power. This removes the invalid static "
            "base-port witness and locates the open choice in the refinement "
            "detail ensemble for the averaged repair semigroup. It does not "
            "equate individual seam microhistories. Inherited-vertex "
            "restriction and unit counting are not identified with a physical "
            "sky readout or measure, and the IID schedule is not selected by "
            "bare A1--A3. The packet supplies no measured-sky statistic or "
            "comparison contract."
        ),
    }


def build_receipt() -> dict[str, Any]:
    payload = _payload()
    receipt = copy.deepcopy(payload)
    receipt["payload_sha256"] = _sha(payload)
    return receipt


def verify_receipt(report: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        received = copy.deepcopy(dict(report))
        received_hash = received.pop("payload_sha256", None)
        if received_hash != _sha(received):
            reasons.append("payload_hash_mismatch")
        if received != _payload():
            reasons.append("producer_replay_mismatch")
        decision = report.get("selection_decision")
        if not isinstance(decision, Mapping):
            reasons.append("selection_decision_missing")
        elif (
            decision.get("physical_sky_readout_selected") is not False
            or decision.get("physical_angular_prediction") is not False
            or decision.get("issue_closure_authorized") is not False
        ):
            reasons.append("forbidden_physical_promotion")
    except (AttributeError, OSError, TypeError, ValueError, ZeroDivisionError):
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
    verification = verify_receipt(report)
    if verification["receipt"] is not True:
        raise RuntimeError("internal observability verification failed")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical_json(report) + b"\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        report = json.loads(args.verify.read_text("utf-8"))
        verification = verify_receipt(report)
        print(json.dumps(verification, sort_keys=True))
        return 0 if verification["receipt"] is True else 1
    report = write_receipt(args.output)
    print(report["status"])
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
