"""Independent exact verifier for the bounded ordered-port diagnostic.

The verifier deliberately does not import the receipt producer.  It rebuilds
the twelve-port carrier, its proper action and antipode directly from the
registered finite carrier, reconstructs the relevant Gaussian-integer matrix
units, and checks the receipt's algebra ranks and witness identities over the
rationals.
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
VERIFICATION_SCHEMA = "oph.ordered-port-response-diagnostic-independent-verification.v1"
VERDICT = "OBVIOUS_DIAGONAL_PORT_LIFT_OVERSHOOTS_TO_U12__SOURCE_CURRENT_OPEN"
PORTS = 12
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = (
    REPOSITORY_ROOT / "data/a2_holonomy/ordered_port_response_diagnostic_receipt.json"
)
EXPECTED_SOURCE_FILES = (
    "oph_fpe/core/echosahedral_dynamics.py",
    "oph_fpe/core/icosahedral.py",
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


class VerificationError(ValueError):
    """Fail-closed independent verification error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _zero() -> IntMatrix:
    return tuple(tuple(0 for _ in range(PORTS)) for _ in range(PORTS))


def _identity() -> IntMatrix:
    return tuple(
        tuple(int(row == column) for column in range(PORTS)) for row in range(PORTS)
    )


def _real_add(left: IntMatrix, right: IntMatrix) -> IntMatrix:
    return tuple(
        tuple(left[r][c] + right[r][c] for c in range(PORTS)) for r in range(PORTS)
    )


def _real_sub(left: IntMatrix, right: IntMatrix) -> IntMatrix:
    return tuple(
        tuple(left[r][c] - right[r][c] for c in range(PORTS)) for r in range(PORTS)
    )


def _real_mul(left: IntMatrix, right: IntMatrix) -> IntMatrix:
    return tuple(
        tuple(sum(left[r][k] * right[k][c] for k in range(PORTS)) for c in range(PORTS))
        for r in range(PORTS)
    )


def _pair(real: IntMatrix | None = None, imag: IntMatrix | None = None) -> PairMatrix:
    return (real if real is not None else _zero()), (
        imag if imag is not None else _zero()
    )


def _add(left: PairMatrix, right: PairMatrix) -> PairMatrix:
    return _real_add(left[0], right[0]), _real_add(left[1], right[1])


def _sub(left: PairMatrix, right: PairMatrix) -> PairMatrix:
    return _real_sub(left[0], right[0]), _real_sub(left[1], right[1])


def _scale(value: PairMatrix, factor: int) -> PairMatrix:
    return tuple(tuple(factor * entry for entry in row) for row in value[0]), tuple(
        tuple(factor * entry for entry in row) for row in value[1]
    )


def _mul(left: PairMatrix, right: PairMatrix) -> PairMatrix:
    return (
        _real_sub(_real_mul(left[0], right[0]), _real_mul(left[1], right[1])),
        _real_add(_real_mul(left[0], right[1]), _real_mul(left[1], right[0])),
    )


def _bracket(left: PairMatrix, right: PairMatrix) -> PairMatrix:
    return _sub(_mul(left, right), _mul(right, left))


def _h(port: int) -> PairMatrix:
    rows = [list(row) for row in _zero()]
    rows[port][port] = 1
    return _pair(imag=tuple(tuple(row) for row in rows))


def _x(left: int, right: int) -> PairMatrix:
    a, b = sorted((left, right))
    rows = [list(row) for row in _zero()]
    rows[a][b], rows[b][a] = 1, -1
    return _pair(real=tuple(tuple(row) for row in rows))


def _y(left: int, right: int) -> PairMatrix:
    a, b = sorted((left, right))
    rows = [list(row) for row in _zero()]
    rows[a][b], rows[b][a] = 1, 1
    return _pair(imag=tuple(tuple(row) for row in rows))


def _equal_up_to_sign(value: PairMatrix, expected: PairMatrix) -> bool:
    return value == expected or value == _scale(expected, -1)


def _coordinates(value: PairMatrix) -> list[list[int]]:
    result: list[list[int]] = []
    for component_index, component in enumerate(value):
        offset = component_index * PORTS * PORTS
        for row in range(PORTS):
            for column in range(PORTS):
                entry = component[row][column]
                if entry:
                    result.append([offset + row * PORTS + column, entry])
    return result


def _basis_sha(values: Sequence[PairMatrix]) -> str:
    return _sha([_coordinates(value) for value in values])


def _rank(values: Sequence[PairMatrix]) -> int:
    pivots: dict[int, dict[int, Fraction]] = {}
    for value in values:
        row = {index: Fraction(entry) for index, entry in _coordinates(value)}
        while row:
            pivot = min(row)
            if pivot not in pivots:
                scale = row[pivot]
                pivots[pivot] = {index: entry / scale for index, entry in row.items()}
                break
            factor = row[pivot]
            for index, entry in pivots[pivot].items():
                updated = row.get(index, Fraction()) - factor * entry
                if updated:
                    row[index] = updated
                else:
                    row.pop(index, None)
    return len(pivots)


def _edges() -> tuple[tuple[int, int], ...]:
    level = build_geodesic_icosahedral_tower(0).levels[0]
    edges = tuple(
        sorted((min(map(int, edge)), max(map(int, edge))) for edge in level.edges)
    )
    if len(edges) != 30 or len(set(edges)) != 30:
        raise VerificationError("carrier edge census failed")
    return edges


def _neighbors(edges: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    rows = [set() for _ in range(PORTS)]
    for left, right in edges:
        rows[left].add(right)
        rows[right].add(left)
    answer = tuple(tuple(sorted(row)) for row in rows)
    if any(len(row) != 5 for row in answer):
        raise VerificationError("carrier is not five-regular")
    return answer


def _path(
    neighbors: Sequence[Sequence[int]], start: int, target: int
) -> tuple[int, ...]:
    queue: deque[int] = deque([start])
    parent: dict[int, int | None] = {start: None}
    while queue:
        node = queue.popleft()
        if node == target:
            break
        for child in neighbors[node]:
            if child not in parent:
                parent[child] = node
                queue.append(child)
    if target not in parent:
        raise VerificationError("carrier is disconnected")
    answer: list[int] = []
    node: int | None = target
    while node is not None:
        answer.append(node)
        node = parent[node]
    return tuple(reversed(answer))


def _antipodes(neighbors: Sequence[Sequence[int]]) -> tuple[int, ...]:
    result: list[int] = []
    for source in range(PORTS):
        distance = [-1] * PORTS
        distance[source] = 0
        queue: deque[int] = deque([source])
        while queue:
            node = queue.popleft()
            for child in neighbors[node]:
                if distance[child] < 0:
                    distance[child] = distance[node] + 1
                    queue.append(child)
        candidates = [port for port, value in enumerate(distance) if value == 3]
        if len(candidates) != 1:
            raise VerificationError("antipode is not unique")
        result.append(candidates[0])
    answer = tuple(result)
    if any(answer[answer[port]] != port for port in range(PORTS)):
        raise VerificationError("antipode is not involutive")
    return answer


def _permutation_matrix(permutation: Sequence[int], sign: int = 1) -> PairMatrix:
    if sorted(map(int, permutation)) != list(range(PORTS)):
        raise VerificationError("malformed port permutation")
    rows = [list(row) for row in _zero()]
    for source, target in enumerate(permutation):
        rows[int(target)][source] = sign
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


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def verify(receipt: Mapping[str, Any]) -> dict[str, Any]:
    submitted = dict(receipt)
    recorded_hash = submitted.pop("receipt_sha256", None)
    _expect(recorded_hash == _sha(submitted), "receipt hash failed")
    _expect(receipt.get("schema") == SCHEMA, "schema mismatch")
    _expect(receipt.get("issue") == 566, "issue mismatch")
    _expect(receipt.get("verdict") == VERDICT, "verdict mismatch")

    source = receipt.get("source_projection")
    _expect(isinstance(source, Mapping), "source projection missing")
    _expect(
        receipt.get("source_projection_sha256") == _sha(source), "source hash failed"
    )
    _expect(source.get("port_count") == PORTS, "port count mismatch")

    edges = _edges()
    neighbors = _neighbors(edges)
    antipodes = _antipodes(neighbors)
    _expect(source.get("carrier_edges") == [list(edge) for edge in edges], "edge drift")
    _expect(source.get("distance_three_map") == list(antipodes), "antipode drift")

    actions = tuple(
        tuple(map(int, action)) for action in icosahedral_a5_port_permutations()
    )
    _expect(len(actions) == 60 and len(set(actions)) == 60, "A5 action census failed")
    _expect(source.get("proper_action_count") == 60, "A5 count mismatch")
    _expect(
        source.get("proper_actions_sha256") == _sha(actions), "A5 action hash failed"
    )

    file_pins = source.get("source_files")
    _expect(isinstance(file_pins, Mapping), "source file pins missing")
    _expect(set(file_pins) == set(EXPECTED_SOURCE_FILES), "source file set drift")
    for relative in EXPECTED_SOURCE_FILES:
        _expect(
            file_pins.get(relative) == _file_sha(REPOSITORY_ROOT / relative),
            f"source file pin failed: {relative}",
        )

    source_hits = sorted(
        {
            token
            for text in _walk_strings(source)
            for token in FORBIDDEN_SOURCE_TOKENS
            if token in text.lower()
        }
    )
    _expect(not source_hits, f"target firewall failed: {source_hits}")
    _expect(source.get("target_labels_used") is False, "target labels admitted")
    _expect(source.get("laboratory_data_used") is False, "laboratory data admitted")
    _expect(
        source.get("conditional_current_fixture_used") is False,
        "conditional current fixture admitted",
    )
    firewall = receipt.get("target_firewall", {})
    _expect(firewall.get("forbidden_source_hits") == [], "firewall receipt failed")
    _expect(
        firewall.get("forbidden_source_tokens") == list(FORBIDDEN_SOURCE_TOKENS),
        "firewall vocabulary drift",
    )
    _expect(
        receipt.get("calculation_audit")
        == {
            "matrix_domain": "Gaussian integer anti-Hermitian matrices",
            "rank_domain": "exact rational row reduction",
            "floating_point_rank_threshold_used": False,
            "matrix_exponential_evaluated": False,
        },
        "calculation-domain audit drift",
    )

    diagonal = tuple(_h(port) for port in range(PORTS))
    direct = receipt.get("direct_port_response", {})
    direct_commutators = [
        _bracket(diagonal[left], diagonal[right])
        for left in range(PORTS)
        for right in range(left)
    ]
    _expect(_rank(diagonal) == 12, "independent first-order rank failed")
    _expect(
        all(not any(_coordinates(value)) for value in direct_commutators),
        "diagonal bracket nonzero",
    )
    _expect(direct.get("primitive_generator_count") == 12, "primitive count mismatch")
    _expect(direct.get("first_order_real_rank") == 12, "first-order rank mismatch")
    _expect(direct.get("unordered_port_pair_count") == 66, "pair count mismatch")
    _expect(
        direct.get("nonzero_direct_commutator_count") == 0,
        "direct bracket count mismatch",
    )
    _expect(
        direct.get("derived_algebra_real_rank_before_propagation") == 0,
        "pre-propagation derived rank mismatch",
    )
    _expect(direct.get("algebra_type") == "u(1)^12", "direct algebra label mismatch")
    _expect(
        direct.get("basis_sha256") == _basis_sha(diagonal), "diagonal basis hash failed"
    )

    adjacency_rows = [list(row) for row in _zero()]
    for left, right in edges:
        adjacency_rows[left][right] = adjacency_rows[right][left] = 1
    propagation = _pair(imag=tuple(tuple(row) for row in adjacency_rows))
    runtime_raw = reference_icosahedral_coupling()
    _expect(tuple(runtime_raw.shape) == (PORTS, PORTS), "runtime coupling shape drift")
    runtime_laplacian = tuple(
        tuple(int(value) for value in row) for row in runtime_raw.tolist()
    )
    _expect(
        all(
            float(runtime_raw[row, column]) == float(runtime_laplacian[row][column])
            for row in range(PORTS)
            for column in range(PORTS)
        ),
        "runtime coupling is not integral",
    )
    expected_laplacian = tuple(
        tuple(
            5 * int(row == column) - adjacency_rows[row][column]
            for column in range(PORTS)
        )
        for row in range(PORTS)
    )
    _expect(runtime_laplacian == expected_laplacian, "runtime L=5I-A identity failed")
    _expect(
        source.get("registered_runtime_laplacian_sha256") == _sha(runtime_laplacian),
        "runtime coupling hash failed",
    )

    edge_x: dict[tuple[int, int], PairMatrix] = {}
    for left, right in edges:
        mixed = _bracket(diagonal[right], _bracket(diagonal[left], propagation))
        _expect(_equal_up_to_sign(mixed, _y(left, right)), "edge mixed bracket failed")
        skew = _bracket(diagonal[left], _y(left, right))
        _expect(_equal_up_to_sign(skew, _x(left, right)), "edge skew bracket failed")
        edge_x[(left, right)] = _x(left, right)

    all_x: list[PairMatrix] = []
    all_y: list[PairMatrix] = []
    expected_paths: list[dict[str, Any]] = []
    for left in range(PORTS):
        for right in range(left + 1, PORTS):
            path = _path(neighbors, left, right)
            current = edge_x[tuple(sorted(path[:2]))]
            for step in range(2, len(path)):
                current = _bracket(
                    current,
                    edge_x[tuple(sorted((path[step - 1], path[step])))],
                )
                _expect(
                    _equal_up_to_sign(current, _x(left, path[step])),
                    "path bracket failed",
                )
            _expect(_equal_up_to_sign(current, _x(left, right)), "pair bracket failed")
            all_x.append(_x(left, right))
            all_y.append(_y(left, right))
            expected_paths.append({"pair": [left, right], "shortest_path": list(path)})

    full_basis = list(diagonal) + all_x + all_y
    diagonal_differences = [_sub(diagonal[port], diagonal[-1]) for port in range(11)]
    derived_basis = diagonal_differences + all_x + all_y
    _expect(_rank(full_basis) == 144, "independent u(12) rank failed")
    _expect(_rank(derived_basis) == 143, "independent su(12) rank failed")
    for left, right in edges:
        _expect(
            _equal_up_to_sign(
                _bracket(_x(left, right), _y(left, right)),
                _scale(_sub(diagonal[left], diagonal[right]), 2),
            ),
            "derived diagonal bracket failed",
        )

    propagated = receipt.get("propagation_adjoined_response", {})
    expected_scalars = {
        "existing_runtime_generator_identity": "-iL = iA - 5iI",
        "runtime_generator_verified": True,
        "edge_mixed_response_nonzero_count": 30,
        "all_unordered_port_pairs_reached": 66,
        "generated_algebra_real_rank": 144,
        "generated_algebra_expected_dimension": 144,
        "generated_algebra_type": "u(12)",
        "derived_algebra_real_rank": 143,
        "derived_algebra_expected_dimension": 143,
        "derived_algebra_type": "su(12)",
        "center_dimension": 1,
        "full_basis_sha256": _basis_sha(full_basis),
        "derived_basis_sha256": _basis_sha(derived_basis),
    }
    for key, expected in expected_scalars.items():
        _expect(propagated.get(key) == expected, f"propagation field failed: {key}")
    _expect(propagated.get("path_witnesses") == expected_paths, "path witnesses drift")
    edge_witnesses = propagated.get("edge_witnesses")
    _expect(
        isinstance(edge_witnesses, list) and len(edge_witnesses) == 30,
        "edge witnesses drift",
    )
    _expect(
        [row.get("edge") for row in edge_witnesses] == [list(edge) for edge in edges],
        "edge witness order drift",
    )

    identity_phase = _pair(imag=_identity())
    h_sum = diagonal[0]
    for generator in diagonal[1:]:
        h_sum = _add(h_sum, generator)
    _expect(h_sum == identity_phase, "central phase reconstruction failed")

    edge_set = set(edges)
    covariance_checks = 0
    for action in actions:
        _expect(
            {tuple(sorted((action[left], action[right]))) for left, right in edges}
            == edge_set,
            "proper action does not preserve adjacency",
        )
        matrix = _permutation_matrix(action)
        inverse = _permutation_matrix(
            tuple(action.index(port) for port in range(PORTS))
        )
        _expect(
            _mul(_mul(matrix, propagation), inverse) == propagation,
            "A5 propagation covariance failed",
        )
        for port in range(PORTS):
            _expect(
                _mul(_mul(matrix, diagonal[port]), inverse) == diagonal[action[port]],
                "A5 port covariance failed",
            )
            covariance_checks += 1
    covariance = receipt.get("a5_covariance_audit", {})
    _expect(
        covariance.get("proper_action_count") == 60, "covariance action count mismatch"
    )
    _expect(
        covariance.get("port_generator_conjugation_checks") == covariance_checks,
        "covariance check count mismatch",
    )
    _expect(
        covariance.get("propagation_invariance_checks") == 60,
        "propagation covariance count mismatch",
    )
    _expect(covariance.get("all_checks_exact") is True, "covariance exactness missing")

    response = _permutation_matrix(antipodes, sign=-1)
    _expect(
        _mul(response, response) == _pair(real=_identity()), "R=-J involution failed"
    )
    for port in range(PORTS):
        _expect(
            _mul(_mul(response, diagonal[port]), response) == diagonal[antipodes[port]],
            "R=-J projector conjugation failed",
        )
    inverse_audit = receipt.get("inverse_port_response_audit", {})
    _expect(
        inverse_audit.get("projector_permutation_checks") == 12,
        "R audit count mismatch",
    )
    _expect(
        inverse_audit.get("adds_continuous_tangent_direction") is False,
        "R tangent claim mismatch",
    )
    _expect(
        inverse_audit.get("reduces_generated_u12_algebra") is False,
        "R reduction claim mismatch",
    )

    gate = receipt.get("corrected_source_acceptance_gate", {})
    _expect(
        gate.get("expected_first_order_real_rank") == 12,
        "source-gate first-order rank drift",
    )
    _expect(
        gate.get("expected_derived_algebra_real_rank") == 11,
        "source-gate derived rank drift",
    )
    _expect(gate.get("expected_center_dimension") == 1, "source-gate center drift")
    _expect(
        gate.get("center_condition")
        == "the constant linear combination of the twelve port generators spans the one-dimensional center",
        "source-gate center condition drift",
    )
    _expect(
        gate.get("obvious_diagonal_lift_satisfies_gate") is False,
        "negative-control gate promoted",
    )
    _expect(
        gate.get("failure_before_propagation") == "derived rank 0 rather than 11",
        "pre-propagation failure drift",
    )
    _expect(
        gate.get("failure_after_propagation") == "derived rank 143 rather than 11",
        "post-propagation failure drift",
    )
    interpretation = receipt.get("scientific_interpretation", {})
    _expect(
        interpretation.get("u12_is_candidate_oph_current") is False,
        "u(12) promoted to OPH current",
    )
    _expect(
        interpretation.get("only_obvious_diagonal_port_lift_rejected") is True,
        "bounded rejection missing",
    )
    _expect(
        interpretation.get("issue_566_closed") is False, "issue #566 prematurely closed"
    )
    verification_surface = receipt.get("verification_surface", {})
    _expect(
        verification_surface.get("independent_implementation") is True,
        "independent-verifier declaration missing",
    )
    _expect(
        verification_surface.get("mutation_controls")
        == [
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
        "mutation-control register drift",
    )

    receipts = receipt.get("receipts", {})
    _expect(
        receipts.get("BOUNDED_ORDERED_PORT_RESPONSE_DIAGNOSTIC_RECEIPT") is True,
        "diagnostic receipt absent",
    )
    for key in (
        "A1_COMPLETE_TWELVE_DIMENSIONAL_RESPONSE_RECEIPT",
        "A2_SAME_CURRENT_HOLONOMY_RECEIPT",
        "PHYSICAL_CURRENT_SOURCE_BRIDGE_RECEIPT",
    ):
        _expect(receipts.get(key) is False, f"forbidden promotion: {key}")
    boundary = receipt.get("claim_boundary", "")
    _expect(
        isinstance(boundary, str) and "rejects only" in boundary,
        "bounded claim missing",
    )
    _expect("does not close issue #566" in boundary, "issue boundary missing")

    return {
        "schema": VERIFICATION_SCHEMA,
        "verdict": "VALID",
        "first_order_real_rank": 12,
        "derived_rank_before_propagation": 0,
        "generated_algebra_real_rank": 144,
        "derived_algebra_real_rank": 143,
        "a5_covariance_checks": covariance_checks,
        "inverse_port_checks": 12,
        "physical_current_source_bridge_receipt": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    result = verify(receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_RECEIPT", "VerificationError", "verify"]
