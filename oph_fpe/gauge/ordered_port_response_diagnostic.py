"""Exact bounded diagnostic for the obvious ordered twelve-port response lift.

The local recurrent carrier stores one complex amplitude at each of twelve
ports.  A direct port-local reversible perturbation on that state is the
phase kick

    D_p(epsilon) = exp(i epsilon |p><p|).

This module asks what Lie algebra those twelve kicks generate before and after
the existing adjacency propagation is admitted.  Every calculation uses
integer matrix units and exact rational rank logic.  No matrix exponential,
finite-difference threshold, physical datum, gauge target, or conditional
current fixture enters the computation.

The result is a bounded negative control.  The twelve first derivatives are
independent and commute.  After adjoining the existing connected propagation
generator they generate all of u(12), with derived algebra su(12).  The
inverse-port response R=-J only permutes the projectors.  This rejects the
obvious diagonal phase-kick lift; it neither rules out another A1 response
lift nor closes issue #566.
"""

from __future__ import annotations

import argparse
from collections import deque
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from oph_fpe.core.echosahedral_dynamics import reference_icosahedral_coupling
from oph_fpe.core.icosahedral import (
    build_geodesic_icosahedral_tower,
    icosahedral_a5_port_permutations,
)


SCHEMA = "oph.ordered-port-response-diagnostic.v1"
ISSUE = 566
VERDICT = "OBVIOUS_DIAGONAL_PORT_LIFT_OVERSHOOTS_TO_U12__SOURCE_CURRENT_OPEN"
PORT_COUNT = 12
U_DIMENSION = PORT_COUNT * PORT_COUNT
SU_DIMENSION = U_DIMENSION - 1

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = (
    REPOSITORY_ROOT / "data/a2_holonomy/ordered_port_response_diagnostic_receipt.json"
)
SOURCE_FILES = (
    REPOSITORY_ROOT / "oph_fpe/core/echosahedral_dynamics.py",
    REPOSITORY_ROOT / "oph_fpe/core/icosahedral.py",
)

FORBIDDEN_SOURCE_TOKENS = (
    "standard model",
    "standard_model",
    "su(3)",
    "su3",
    "electroweak",
    "hypercharge",
    "particle_mass",
    "measured_coupling",
    "gauge_target",
)

IntMatrix = tuple[tuple[int, ...], ...]
PairMatrix = tuple[IntMatrix, IntMatrix]


class DiagnosticError(ValueError):
    """Fail-closed exact diagnostic error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _zero_real() -> IntMatrix:
    return tuple(tuple(0 for _ in range(PORT_COUNT)) for _ in range(PORT_COUNT))


def _identity_real() -> IntMatrix:
    return tuple(
        tuple(int(row == column) for column in range(PORT_COUNT))
        for row in range(PORT_COUNT)
    )


def _real_add(left: IntMatrix, right: IntMatrix) -> IntMatrix:
    return tuple(
        tuple(left[row][column] + right[row][column] for column in range(PORT_COUNT))
        for row in range(PORT_COUNT)
    )


def _real_subtract(left: IntMatrix, right: IntMatrix) -> IntMatrix:
    return tuple(
        tuple(left[row][column] - right[row][column] for column in range(PORT_COUNT))
        for row in range(PORT_COUNT)
    )


def _real_multiply(left: IntMatrix, right: IntMatrix) -> IntMatrix:
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(PORT_COUNT))
            for column in range(PORT_COUNT)
        )
        for row in range(PORT_COUNT)
    )


def _pair(real: IntMatrix | None = None, imag: IntMatrix | None = None) -> PairMatrix:
    return real or _zero_real(), imag or _zero_real()


def _pair_add(left: PairMatrix, right: PairMatrix) -> PairMatrix:
    return _real_add(left[0], right[0]), _real_add(left[1], right[1])


def _pair_subtract(left: PairMatrix, right: PairMatrix) -> PairMatrix:
    return _real_subtract(left[0], right[0]), _real_subtract(left[1], right[1])


def _pair_scale(value: PairMatrix, scale: int) -> PairMatrix:
    return tuple(tuple(scale * entry for entry in row) for row in value[0]), tuple(
        tuple(scale * entry for entry in row) for row in value[1]
    )


def _pair_multiply(left: PairMatrix, right: PairMatrix) -> PairMatrix:
    real = _real_subtract(
        _real_multiply(left[0], right[0]),
        _real_multiply(left[1], right[1]),
    )
    imag = _real_add(
        _real_multiply(left[0], right[1]),
        _real_multiply(left[1], right[0]),
    )
    return real, imag


def _commutator(left: PairMatrix, right: PairMatrix) -> PairMatrix:
    return _pair_subtract(
        _pair_multiply(left, right),
        _pair_multiply(right, left),
    )


def _is_zero(value: PairMatrix) -> bool:
    return not any(entry for component in value for row in component for entry in row)


def _h(port: int) -> PairMatrix:
    imag = [list(row) for row in _zero_real()]
    imag[port][port] = 1
    return _pair(imag=tuple(tuple(row) for row in imag))


def _x(left: int, right: int) -> PairMatrix:
    if left == right:
        raise ValueError("an off-diagonal matrix unit needs two ports")
    a, b = sorted((int(left), int(right)))
    real = [list(row) for row in _zero_real()]
    real[a][b] = 1
    real[b][a] = -1
    return _pair(real=tuple(tuple(row) for row in real))


def _y(left: int, right: int) -> PairMatrix:
    if left == right:
        raise ValueError("an off-diagonal matrix unit needs two ports")
    a, b = sorted((int(left), int(right)))
    imag = [list(row) for row in _zero_real()]
    imag[a][b] = 1
    imag[b][a] = 1
    return _pair(imag=tuple(tuple(row) for row in imag))


def _normalize_sign(value: PairMatrix, expected: PairMatrix, label: str) -> PairMatrix:
    if value == expected:
        return value
    if value == _pair_scale(expected, -1):
        return expected
    raise DiagnosticError(f"{label} did not produce the expected matrix unit")


def _sparse_coordinates(value: PairMatrix) -> tuple[tuple[int, int], ...]:
    rows: list[tuple[int, int]] = []
    for component_index, component in enumerate(value):
        offset = component_index * PORT_COUNT * PORT_COUNT
        for row in range(PORT_COUNT):
            for column in range(PORT_COUNT):
                entry = component[row][column]
                if entry:
                    rows.append((offset + row * PORT_COUNT + column, entry))
    return tuple(rows)


def _basis_sha256(values: Sequence[PairMatrix]) -> str:
    return canonical_sha256(
        [[list(entry) for entry in _sparse_coordinates(value)] for value in values]
    )


def _exact_sparse_rank(values: Sequence[PairMatrix]) -> int:
    """Exact rank over Q of sparse real coordinates of Gaussian matrices."""

    basis: dict[int, dict[int, Fraction]] = {}
    for matrix in values:
        row = {
            coordinate: Fraction(value)
            for coordinate, value in _sparse_coordinates(matrix)
        }
        while row:
            pivot = min(row)
            if pivot not in basis:
                scale = row[pivot]
                normalized = {
                    coordinate: value / scale
                    for coordinate, value in row.items()
                    if value
                }
                basis[pivot] = normalized
                break
            factor = row[pivot]
            for coordinate, value in basis[pivot].items():
                updated = row.get(coordinate, Fraction()) - factor * value
                if updated:
                    row[coordinate] = updated
                else:
                    row.pop(coordinate, None)
    return len(basis)


def _carrier_edges() -> tuple[tuple[int, int], ...]:
    level = build_geodesic_icosahedral_tower(0).levels[0]
    edges = tuple(
        sorted(
            (min(int(left), int(right)), max(int(left), int(right)))
            for left, right in level.edges
        )
    )
    if len(edges) != 30 or len(set(edges)) != 30:
        raise DiagnosticError("reference carrier does not have thirty edges")
    return edges


def _adjacency(edges: Sequence[tuple[int, int]]) -> IntMatrix:
    rows = [list(row) for row in _zero_real()]
    for left, right in edges:
        rows[left][right] = 1
        rows[right][left] = 1
    result = tuple(tuple(row) for row in rows)
    if any(sum(row) != 5 for row in result):
        raise DiagnosticError("reference carrier is not five-regular")
    return result


def _registered_runtime_laplacian(edges: Sequence[tuple[int, int]]) -> IntMatrix:
    """Bind the exact audit to the coupling used by the registered propagator."""

    raw = reference_icosahedral_coupling()
    if tuple(raw.shape) != (PORT_COUNT, PORT_COUNT):
        raise DiagnosticError("registered runtime coupling is not twelve-dimensional")
    rows: list[tuple[int, ...]] = []
    for raw_row in raw.tolist():
        row: list[int] = []
        for value in raw_row:
            integer = int(value)
            if float(value) != float(integer):
                raise DiagnosticError("registered runtime coupling is not integral")
            row.append(integer)
        rows.append(tuple(row))
    laplacian = tuple(rows)
    expected = _real_subtract(
        _pair_scale(_pair(real=_identity_real()), 5)[0], _adjacency(edges)
    )
    if laplacian != expected:
        raise DiagnosticError("registered runtime coupling is not L=5I-A")
    return laplacian


def _adjacency_lists(edges: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    rows = [set() for _ in range(PORT_COUNT)]
    for left, right in edges:
        rows[left].add(right)
        rows[right].add(left)
    return tuple(tuple(sorted(row)) for row in rows)


def _shortest_path(
    adjacency: Sequence[Sequence[int]], start: int, target: int
) -> tuple[int, ...]:
    queue: deque[int] = deque([start])
    parent: dict[int, int | None] = {start: None}
    while queue:
        node = queue.popleft()
        if node == target:
            break
        for neighbor in adjacency[node]:
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    if target not in parent:
        raise DiagnosticError("reference carrier is disconnected")
    path: list[int] = []
    node: int | None = target
    while node is not None:
        path.append(node)
        node = parent[node]
    return tuple(reversed(path))


def _antipode_map(adjacency: Sequence[Sequence[int]]) -> tuple[int, ...]:
    result: list[int] = []
    for source in range(PORT_COUNT):
        distances = [-1] * PORT_COUNT
        distances[source] = 0
        queue: deque[int] = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        partners = [index for index, distance in enumerate(distances) if distance == 3]
        if len(partners) != 1:
            raise DiagnosticError("each port must have one distance-three partner")
        result.append(partners[0])
    answer = tuple(result)
    if any(answer[answer[index]] != index for index in range(PORT_COUNT)):
        raise DiagnosticError("distance-three map is not an involution")
    return answer


def _permutation_conjugate(value: PairMatrix, permutation: Sequence[int]) -> PairMatrix:
    components: list[IntMatrix] = []
    for component in value:
        target = [list(row) for row in _zero_real()]
        for row in range(PORT_COUNT):
            for column in range(PORT_COUNT):
                target[int(permutation[row])][int(permutation[column])] = component[
                    row
                ][column]
        components.append(tuple(tuple(row) for row in target))
    return components[0], components[1]


def _signed_permutation_matrix(
    permutation: Sequence[int], *, sign: int = 1
) -> PairMatrix:
    """Return the real matrix sending e_p to ``sign * e_permutation[p]``."""

    if sorted(map(int, permutation)) != list(range(PORT_COUNT)):
        raise DiagnosticError("port response is not a permutation")
    rows = [list(row) for row in _zero_real()]
    for port, image in enumerate(permutation):
        rows[int(image)][port] = int(sign)
    return _pair(real=tuple(tuple(row) for row in rows))


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_strings(child)
    elif isinstance(value, str):
        yield value


def _source_projection() -> dict[str, Any]:
    edges = _carrier_edges()
    adjacency = _adjacency_lists(edges)
    antipode = _antipode_map(adjacency)
    actions = tuple(tuple(map(int, row)) for row in icosahedral_a5_port_permutations())
    if len(actions) != 60 or len(set(actions)) != 60:
        raise DiagnosticError("proper carrier action does not have order sixty")
    projection = {
        "port_count": PORT_COUNT,
        "carrier_edges": [list(edge) for edge in edges],
        "proper_action_count": len(actions),
        "proper_actions_sha256": canonical_sha256(actions),
        "distance_three_map": list(antipode),
        "primitive_path": "one-port norm-preserving complex phase kick",
        "primitive_generator": "i times the selected port projector",
        "propagation_generator": "minus i times L, where L = 5 I - A",
        "registered_runtime_laplacian_sha256": canonical_sha256(
            _registered_runtime_laplacian(edges)
        ),
        "ordered_readback": "exact Lie brackets and nested commutators",
        "target_labels_used": False,
        "laboratory_data_used": False,
        "conditional_current_fixture_used": False,
        "source_files": {
            path.relative_to(REPOSITORY_ROOT).as_posix(): _file_sha256(path)
            for path in SOURCE_FILES
        },
    }
    hits = sorted(
        {
            token
            for text in _walk_strings(projection)
            for token in FORBIDDEN_SOURCE_TOKENS
            if token in text.lower()
        }
    )
    if hits:
        raise DiagnosticError(f"forbidden source tokens: {hits}")
    return projection


def _algebra_report(edges: Sequence[tuple[int, int]]) -> dict[str, Any]:
    diagonal = tuple(_h(port) for port in range(PORT_COUNT))
    if _exact_sparse_rank(diagonal) != PORT_COUNT:
        raise DiagnosticError("first-order port projector rank is not twelve")
    direct_commutators = tuple(
        _commutator(diagonal[left], diagonal[right])
        for left in range(PORT_COUNT)
        for right in range(left)
    )
    if any(not _is_zero(value) for value in direct_commutators):
        raise DiagnosticError("diagonal phase kicks do not commute")

    adjacency_matrix = _adjacency(edges)
    runtime_laplacian = _registered_runtime_laplacian(edges)
    propagation = _pair(imag=adjacency_matrix)  # iA
    adjacency_lists = _adjacency_lists(edges)

    edge_x: dict[tuple[int, int], PairMatrix] = {}
    edge_y: dict[tuple[int, int], PairMatrix] = {}
    edge_witnesses: list[dict[str, Any]] = []
    for left, right in edges:
        first = _commutator(diagonal[left], propagation)
        mixed = _commutator(diagonal[right], first)
        y_value = _normalize_sign(mixed, _y(left, right), "edge mixed response")
        x_value = _normalize_sign(
            _commutator(diagonal[left], y_value),
            _x(left, right),
            "edge skew response",
        )
        edge_x[(left, right)] = x_value
        edge_y[(left, right)] = y_value
        edge_witnesses.append(
            {
                "edge": [left, right],
                "symmetric_witness": "[H_right,[H_left,G]]",
                "skew_witness": "[H_left,Y_left_right]",
            }
        )

    all_x: list[PairMatrix] = []
    all_y: list[PairMatrix] = []
    path_witnesses: list[dict[str, Any]] = []
    for left in range(PORT_COUNT):
        for right in range(left + 1, PORT_COUNT):
            path = _shortest_path(adjacency_lists, left, right)
            current = edge_x[tuple(sorted(path[:2]))]
            for step in range(2, len(path)):
                next_edge = edge_x[tuple(sorted((path[step - 1], path[step])))]
                current = _normalize_sign(
                    _commutator(current, next_edge),
                    _x(left, path[step]),
                    "path skew response",
                )
            x_value = _normalize_sign(current, _x(left, right), "pair skew response")
            y_value = _normalize_sign(
                _commutator(diagonal[left], x_value),
                _y(left, right),
                "pair symmetric response",
            )
            all_x.append(x_value)
            all_y.append(y_value)
            path_witnesses.append(
                {
                    "pair": [left, right],
                    "shortest_path": list(path),
                }
            )

    full_basis = list(diagonal) + all_x + all_y
    full_rank = _exact_sparse_rank(full_basis)
    if full_rank != U_DIMENSION:
        raise DiagnosticError(f"generated unitary algebra rank is {full_rank}")

    diagonal_differences = tuple(
        _pair_subtract(diagonal[port], diagonal[-1]) for port in range(PORT_COUNT - 1)
    )
    for left, right in edges:
        diagonal_commutator = _commutator(_x(left, right), _y(left, right))
        expected = _pair_scale(_pair_subtract(diagonal[left], diagonal[right]), 2)
        _normalize_sign(
            diagonal_commutator,
            expected,
            "derived diagonal response",
        )
    derived_basis = list(diagonal_differences) + all_x + all_y
    derived_rank = _exact_sparse_rank(derived_basis)
    if derived_rank != SU_DIMENSION:
        raise DiagnosticError(f"derived algebra rank is {derived_rank}")

    identity_generator = _pair(imag=_identity_real())
    h_sum = diagonal[0]
    for value in diagonal[1:]:
        h_sum = _pair_add(h_sum, value)
    if h_sum != identity_generator:
        raise DiagnosticError("sum of port phases is not the central phase")
    runtime_generator = _pair_subtract(propagation, _pair_scale(h_sum, 5))
    if runtime_generator != _pair_scale(_pair(imag=runtime_laplacian), -1):
        raise DiagnosticError("registered runtime tangent is not -iL")

    return {
        "direct_port_response": {
            "primitive_generator_count": len(diagonal),
            "first_order_real_rank": _exact_sparse_rank(diagonal),
            "unordered_port_pair_count": len(direct_commutators),
            "nonzero_direct_commutator_count": sum(
                not _is_zero(value) for value in direct_commutators
            ),
            "derived_algebra_real_rank_before_propagation": 0,
            "algebra_type": "u(1)^12",
            "basis_sha256": _basis_sha256(diagonal),
        },
        "propagation_adjoined_response": {
            "existing_runtime_generator_identity": "-iL = iA - 5iI",
            "runtime_generator_verified": not _is_zero(runtime_generator),
            "edge_mixed_response_nonzero_count": len(edge_y),
            "edge_witnesses": edge_witnesses,
            "all_unordered_port_pairs_reached": len(all_x),
            "path_witnesses": path_witnesses,
            "generated_algebra_real_rank": full_rank,
            "generated_algebra_expected_dimension": U_DIMENSION,
            "generated_algebra_type": "u(12)",
            "derived_algebra_real_rank": derived_rank,
            "derived_algebra_expected_dimension": SU_DIMENSION,
            "derived_algebra_type": "su(12)",
            "center_dimension": 1,
            "full_basis_sha256": _basis_sha256(full_basis),
            "derived_basis_sha256": _basis_sha256(derived_basis),
        },
        "_diagonal": diagonal,
        "_propagation": propagation,
    }


def produce_receipt() -> dict[str, Any]:
    projection = _source_projection()
    edges = tuple(tuple(map(int, edge)) for edge in projection["carrier_edges"])
    algebra = _algebra_report(edges)
    diagonal = algebra.pop("_diagonal")
    propagation = algebra.pop("_propagation")
    actions = tuple(tuple(map(int, row)) for row in icosahedral_a5_port_permutations())

    covariance_checks = 0
    for permutation in actions:
        if _permutation_conjugate(propagation, permutation) != propagation:
            raise DiagnosticError("a proper action changes the propagation generator")
        for port in range(PORT_COUNT):
            if (
                _permutation_conjugate(diagonal[port], permutation)
                != diagonal[permutation[port]]
            ):
                raise DiagnosticError("port phase family is not A5 covariant")
            covariance_checks += 1

    antipode = tuple(map(int, projection["distance_three_map"]))
    inverse_response = _signed_permutation_matrix(antipode, sign=-1)
    if _pair_multiply(inverse_response, inverse_response) != _pair(
        real=_identity_real()
    ):
        raise DiagnosticError("R=-J is not an involution")
    inverse_checks = 0
    for port in range(PORT_COUNT):
        conjugated = _pair_multiply(
            _pair_multiply(inverse_response, diagonal[port]),
            inverse_response,
        )
        if conjugated != diagonal[antipode[port]]:
            raise DiagnosticError("inverse-port response does not permute projectors")
        inverse_checks += 1

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "verdict": VERDICT,
        "source_projection": projection,
        "source_projection_sha256": canonical_sha256(projection),
        "target_firewall": {
            "forbidden_source_tokens": list(FORBIDDEN_SOURCE_TOKENS),
            "forbidden_source_hits": [],
            "target_labels_used": False,
            "laboratory_data_used": False,
            "conditional_current_fixture_used": False,
        },
        "calculation_audit": {
            "matrix_domain": "Gaussian integer anti-Hermitian matrices",
            "rank_domain": "exact rational row reduction",
            "floating_point_rank_threshold_used": False,
            "matrix_exponential_evaluated": False,
        },
        **algebra,
        "inverse_port_response_audit": {
            "operator": "R = -J",
            "conjugation_action": "R H_p R^-1 = H_J(p)",
            "projector_permutation_checks": inverse_checks,
            "adds_continuous_tangent_direction": False,
            "reduces_generated_u12_algebra": False,
        },
        "a5_covariance_audit": {
            "proper_action_count": len(actions),
            "port_generator_conjugation_checks": covariance_checks,
            "propagation_invariance_checks": len(actions),
            "all_checks_exact": True,
        },
        "corrected_source_acceptance_gate": {
            "expected_first_order_real_rank": 12,
            "expected_derived_algebra_real_rank": 11,
            "expected_center_dimension": 1,
            "center_condition": (
                "the constant linear combination of the twelve port generators "
                "spans the one-dimensional center"
            ),
            "obvious_diagonal_lift_satisfies_gate": False,
            "failure_before_propagation": "derived rank 0 rather than 11",
            "failure_after_propagation": "derived rank 143 rather than 11",
        },
        "scientific_interpretation": {
            "u12_is_candidate_oph_current": False,
            "only_obvious_diagonal_port_lift_rejected": True,
            "issue_566_closed": False,
        },
        "verification_surface": {
            "independent_implementation": True,
            "mutation_controls": [
                "receipt_hash",
                "source_projection_hash",
                "carrier_edges",
                "proper_action",
                "distance_three_response",
                "first_order_rank",
                "full_basis_hash",
                "target_firewall",
                "physical_promotion",
            ],
        },
        "receipts": {
            "BOUNDED_ORDERED_PORT_RESPONSE_DIAGNOSTIC_RECEIPT": True,
            "A1_COMPLETE_TWELVE_DIMENSIONAL_RESPONSE_RECEIPT": False,
            "A2_SAME_CURRENT_HOLONOMY_RECEIPT": False,
            "PHYSICAL_CURRENT_SOURCE_BRIDGE_RECEIPT": False,
        },
        "status": "ATTAINED_BOUNDED_NEGATIVE_CONTROL",
        "claim_boundary": (
            "The declared diagonal phase kicks are a target-free reversible "
            "port-local lift on the existing complex twelve-channel state. They "
            "are abelian before propagation and generate the full u(12) after "
            "the connected adjacency propagation is admitted. This rejects only "
            "that obvious lift. It does not range over other A1 response spaces "
            "or port actions, does not weaken the conditional A1/A2 Lie-type "
            "theorem, and does not close issue #566."
        ),
        "next_source_object": (
            "a target-free non-diagonal port lift whose twelve first derivatives "
            "close with derived rank eleven and one-dimensional constant center, "
            "with all proper rechartings generated by the same response"
        ),
        "verifier_command": (
            "python3 -m oph_fpe.gauge.verify_ordered_port_response_diagnostic_independent "
            "--receipt data/a2_holonomy/ordered_port_response_diagnostic_receipt.json"
        ),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def verify_receipt(receipt: Mapping[str, Any]) -> None:
    submitted = dict(receipt)
    received_hash = submitted.pop("receipt_sha256", None)
    if received_hash != canonical_sha256(submitted):
        raise DiagnosticError("receipt hash failed")
    expected = produce_receipt()
    if dict(receipt) != expected:
        raise DiagnosticError("receipt does not replay exactly")


def write_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    receipt = produce_receipt()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        receipt = json.loads(args.out.read_text(encoding="utf-8"))
        verify_receipt(receipt)
        print("ORDERED_PORT_RESPONSE_DIAGNOSTIC_VALID")
        return 0
    receipt = write_receipt(args.out)
    print(receipt["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_RECEIPT",
    "DiagnosticError",
    "SCHEMA",
    "VERDICT",
    "canonical_sha256",
    "produce_receipt",
    "verify_receipt",
    "write_receipt",
]
