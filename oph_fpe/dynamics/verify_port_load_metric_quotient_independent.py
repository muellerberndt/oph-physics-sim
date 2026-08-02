"""Independent verifier for the integer-load repair-Gram quotient packet.

This verifier does not import the producer.  It reconstructs the exact Gram
factorization, fixed-total images, index-two displacement lattice, conditional
mean intertwiner, and nonlinear pathwise counterexample from pinned parents.
"""

from __future__ import annotations

import argparse
import ast
import copy
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / "data/repair_closure/port_load_metric_quotient_receipt.json"
COMPLETION_RECEIPT = ROOT / "data/repair_closure/port_gram_completion_bridge_receipt.json"
ACTION_RECEIPT = ROOT / "data/repair_closure/port_gram_equivariant_action_receipt.json"
REPAIR_RECEIPT = ROOT / "data/repair_closure/bounded_atomic_self_readback_closure_receipt.json"
PORT_REPAIR_RECEIPT = ROOT / "data/repair_closure/port_repair_propagation_bridge_receipt.json"
CARRIER_MANIFEST = ROOT / "tests/fixtures/echosahedral_federation_reference.json"
PRODUCER_PATH = ROOT / "oph_fpe/dynamics/port_load_metric_quotient.py"
VERIFIER_PATH = Path(__file__).resolve()
TEST_PATH = ROOT / "tests/test_port_load_metric_quotient.py"

SCHEMA = "oph.port-load-repair-gram-metric-quotient.v1"
STATUS = (
    "EXACT_INTEGER_LOAD_METRIC_QUOTIENT_AND_MEAN_INTERTWINER_ATTAINED__"
    "PATHWISE_DESCENT_POSITION_SEMANTICS_AND_PHYSICAL_ACTION_OPEN"
)
COMPLETION_SCHEMA = "oph.port-gram-hausdorff-completion-bridge.v1"
ACTION_SCHEMA = "oph.port-gram-equivariant-completion-action.v1"
REPAIR_SCHEMA = "oph.bounded_atomic_self_readback_closure.v1"
PORT_REPAIR_SCHEMA = "oph.port_repair_propagation_bridge_receipt.v1"
TOP_LEVEL_KEYS = {
    "schema",
    "status",
    "issues",
    "target_data_read",
    "comparison_data_read",
    "parent_pins",
    "exact_integer_load_metric_quotient",
    "fixed_total_source_geometry",
    "exact_seam_current_boundary",
    "exact_mean_quotient_intertwiner",
    "pathwise_descent_counterexample",
    "attainment",
    "claim_boundary",
    "implementation_pins",
    "receipt_sha256",
}

Q5 = tuple[Fraction, Fraction]
ZERO: Q5 = (Fraction(0), Fraction(0))


class IndependentPortLoadQuotientError(RuntimeError):
    """Raised when the independently reconstructed packet fails."""


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentPortLoadQuotientError(message)


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


def _raw_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IndependentPortLoadQuotientError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise IndependentPortLoadQuotientError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _load(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentPortLoadQuotientError(f"cannot load {path}: {error}") from error
    _fail(isinstance(result, dict), f"{path} is not an object")
    return result


def _validate_self_digest(path: Path, schema: str) -> dict[str, Any]:
    receipt = _load(path)
    payload = copy.deepcopy(receipt)
    digest = payload.pop("receipt_sha256", None)
    _fail(digest == _sha(payload), f"{path} self digest")
    _fail(receipt.get("schema") == schema, f"{path} schema")
    return receipt


def _parse_q5(text: str) -> Q5:
    suffix = "*sqrt5"
    _fail(isinstance(text, str) and text.endswith(suffix), "Q(sqrt5) encoding")
    body = text[: -len(suffix)]
    split = body.find("+", 1)
    _fail(split > 0, "Q(sqrt5) split")
    return Fraction(body[:split]), Fraction(body[split + 1 :])


def _parse_control_q5(text: str) -> Q5:
    compact = str(text).replace(" ", "").replace("sqrt(5)", "sqrt5")
    if "*sqrt5" not in compact:
        return Fraction(compact), Fraction()
    body = compact[: -len("*sqrt5")]
    split = body.find("+", 1)
    _fail(split > 0, "control Q(sqrt5) split")
    return Fraction(body[:split]), Fraction(body[split + 1 :])


def _qadd(left: Q5, right: Q5) -> Q5:
    return left[0] + right[0], left[1] + right[1]


def _qmul(left: Q5, right: Q5) -> Q5:
    return (
        left[0] * right[0] + 5 * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _qsub(left: Q5, right: Q5) -> Q5:
    return left[0] - right[0], left[1] - right[1]


def _qneg_vector(vector: Sequence[Q5]) -> tuple[Q5, ...]:
    return tuple((-value[0], -value[1]) for value in vector)


def _qaxis(vector: Sequence[Q5]) -> tuple[Q5, ...]:
    frozen = tuple(vector)
    return min(frozen, _qneg_vector(frozen))


def _qlinear_combination(coefficients: Sequence[int], values: Sequence[Q5]) -> Q5:
    result = ZERO
    for coefficient, value in zip(coefficients, values, strict=True):
        result = _qadd(
            result,
            _qmul((Fraction(coefficient), Fraction()), value),
        )
    return result


def _transpose(matrix: Sequence[Sequence[Any]]) -> list[list[Any]]:
    return [list(column) for column in zip(*matrix, strict=True)]


def _qmatmul(
    left: Sequence[Sequence[Q5]], right: Sequence[Sequence[Q5]]
) -> list[list[Q5]]:
    result = []
    for row in range(len(left)):
        output_row = []
        for column in range(len(right[0])):
            value = ZERO
            for pivot in range(len(right)):
                value = _qadd(value, _qmul(left[row][pivot], right[pivot][column]))
            output_row.append(value)
        result.append(output_row)
    return result


def _rmatmul(
    left: Sequence[Sequence[Fraction]], right: Sequence[Sequence[Fraction]]
) -> list[list[Fraction]]:
    return [
        [
            sum(
                (left[row][pivot] * right[pivot][column] for pivot in range(len(right))),
                Fraction(),
            )
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def _determinant(matrix: Sequence[Sequence[int]]) -> int:
    work = [[Fraction(value) for value in row] for row in matrix]
    sign = 1
    value = Fraction(1)
    for column in range(len(work)):
        pivot = next(
            (row for row in range(column, len(work)) if work[row][column]), None
        )
        if pivot is None:
            return 0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            sign *= -1
        diagonal = work[column][column]
        value *= diagonal
        for row in range(column + 1, len(work)):
            factor = work[row][column] / diagonal
            for index in range(column, len(work)):
                work[row][index] -= factor * work[column][index]
    output = sign * value
    _fail(output.denominator == 1, "integer determinant")
    return output.numerator


def _compositions(total: int, slots: int) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []

    def visit(remaining: int, index: int, row: tuple[int, ...]) -> None:
        if index == slots - 1:
            result.append(row + (remaining,))
            return
        for value in range(remaining + 1):
            visit(remaining - value, index + 1, row + (value,))

    visit(total, 0, ())
    return result


def _apply(difference: Sequence[Sequence[int]], state: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        sum(difference[row][column] * state[column] for column in range(12))
        for row in range(6)
    )


def _formula_image(total: int) -> set[tuple[int, ...]]:
    return {
        row
        for row in itertools.product(range(-total, total + 1), repeat=6)
        if sum(abs(value) for value in row) <= total
        and (sum(row) - total) % 2 == 0
    }


def _balanced(state: Sequence[int], left: int, right: int) -> tuple[int, ...]:
    result = list(state)
    total = state[left] + state[right]
    result[left] = total // 2
    result[right] = total - result[left]
    return tuple(result)


def _distribution(
    state: Sequence[int], edges: Sequence[tuple[int, int]], difference: Sequence[Sequence[int]]
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for left, right in edges:
        for first, second in ((left, right), (right, left)):
            quotient = _apply(difference, _balanced(state, first, second))
            result[quotient] = result.get(quotient, Fraction()) + Fraction(1, 60)
    return result


def _expectation(distribution: Mapping[tuple[int, ...], Fraction]) -> tuple[Fraction, ...]:
    return tuple(
        sum(
            (probability * row[column] for row, probability in distribution.items()),
            Fraction(),
        )
        for column in range(6)
    )


def _serialized_distribution(rows: Any) -> dict[tuple[int, ...], Fraction]:
    _fail(isinstance(rows, list), "serialized distribution")
    result = {}
    for row in rows:
        _fail(isinstance(row, Mapping), "distribution row")
        value = tuple(row.get("z", []))
        probability = Fraction(str(row.get("probability")))
        result[value] = probability
    return result


def _check_pins(receipt: Mapping[str, Any]) -> None:
    pins = receipt.get("implementation_pins")
    _fail(isinstance(pins, list) and len(pins) == 3, "implementation pins")
    for pin, path in zip(
        pins, (PRODUCER_PATH, VERIFIER_PATH, TEST_PATH), strict=True
    ):
        _fail(isinstance(pin, Mapping), "implementation pin object")
        _fail(pin.get("path") == path.relative_to(ROOT).as_posix(), "pin path")
        _fail(pin.get("bytes") == len(path.read_bytes()), "pin bytes")
        _fail(pin.get("sha256") == _raw_sha(path), "pin digest")


def _check_no_producer_import() -> None:
    tree = ast.parse(VERIFIER_PATH.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    _fail(
        "oph_fpe.dynamics.port_load_metric_quotient" not in imported,
        "independent verifier imports producer",
    )


def verify_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, bool]:
    receipt = _load(path)
    _fail(set(receipt) == TOP_LEVEL_KEYS, "top-level schema")
    payload = copy.deepcopy(receipt)
    digest = payload.pop("receipt_sha256", None)
    _fail(digest == _sha(payload), "receipt digest")
    _fail(receipt.get("schema") == SCHEMA, "receipt schema")
    _fail(receipt.get("status") == STATUS, "receipt status")
    _fail(receipt.get("issues") == [655, 663, 666], "issue binding")
    _fail(receipt.get("target_data_read") is False, "target data firewall")
    _fail(receipt.get("comparison_data_read") is False, "comparison data firewall")

    completion = _validate_self_digest(COMPLETION_RECEIPT, COMPLETION_SCHEMA)
    action = _validate_self_digest(ACTION_RECEIPT, ACTION_SCHEMA)
    port_repair = _validate_self_digest(PORT_REPAIR_RECEIPT, PORT_REPAIR_SCHEMA)
    repair = _load(REPAIR_RECEIPT)
    repair_payload = copy.deepcopy(repair)
    repair_digest = repair_payload.pop("certificate_payload_sha256", None)
    _fail(repair_digest == _sha(repair_payload), "repair payload digest")
    _fail(repair.get("schema") == REPAIR_SCHEMA, "repair schema")

    parent_pins = receipt.get("parent_pins")
    _fail(isinstance(parent_pins, Mapping), "parent pins")
    paths = {
        "repair_selected_completion": COMPLETION_RECEIPT,
        "conditional_equivariant_completion_action": ACTION_RECEIPT,
        "bounded_integer_load_repair_source": REPAIR_RECEIPT,
        "edge30_orbit_ray_control": PORT_REPAIR_RECEIPT,
        "carrier_manifest": CARRIER_MANIFEST,
    }
    for key, parent_path in paths.items():
        pin = parent_pins.get(key)
        _fail(isinstance(pin, Mapping), f"parent pin {key}")
        _fail(pin.get("sha256") == _raw_sha(parent_path), f"parent raw pin {key}")
    source_pin = parent_pins["bounded_integer_load_repair_source"]
    _fail(source_pin.get("undirected_seam_attempt_count") == 30, "seam attempt count")
    _fail(source_pin.get("directed_completion_label_count") == 60, "completion label count")

    module = receipt.get("exact_integer_load_metric_quotient")
    _fail(isinstance(module, Mapping), "metric quotient packet")
    positive = tuple(module.get("positive_port_basis", []))
    antipodes = tuple(module.get("antipodal_involution", []))
    difference = module.get("difference_matrix")
    _fail(positive == (0, 1, 4, 5, 8, 9), "positive port basis")
    _fail(len(antipodes) == 12, "antipodal involution")
    _fail(
        all(antipodes[antipodes[index]] == index for index in range(12)),
        "antipodal involution law",
    )
    _fail(
        isinstance(difference, list)
        and difference
        == [
            [
                int(column == port) - int(column == antipodes[port])
                for column in range(12)
            ]
            for port in positive
        ],
        "difference matrix",
    )
    integer_section = module.get("integer_section_matrix")
    _fail(
        isinstance(integer_section, list)
        and integer_section
        == [
            [int(row == port) for _column, port in enumerate(positive)]
            for row in range(12)
        ],
        "integer section matrix",
    )
    split_identity = [
        [
            sum(difference[row][pivot] * integer_section[pivot][column] for pivot in range(12))
            for column in range(6)
        ]
        for row in range(6)
    ]
    _fail(
        split_identity
        == [[int(row == column) for column in range(6)] for row in range(6)],
        "split surjectivity",
    )
    _fail(module.get("difference_after_section_is_identity") is True, "reported split")

    exact = completion["exact_repair_selected_gram"]
    gram12 = [[_parse_q5(value) for value in row] for row in exact["full_gram_qsqrt5"]]
    gram6 = [[gram12[left][right] for right in positive] for left in positive]
    difference_q5 = [[(Fraction(value), Fraction()) for value in row] for row in difference]
    factorized = _qmatmul(
        _transpose(difference_q5), _qmatmul(gram6, difference_q5)
    )
    _fail(factorized == gram12, "exact Gram factorization")
    parent_module = completion["exact_signed_module_completion"]
    _fail(parent_module.get("integer_kernel_is_zero") is True, "signed integer kernel")
    _fail(module.get("integer_metric_kernel_exact") is True, "reported load kernel")
    _fail(module.get("quotient_isomorphic_to_Z6") is True, "reported quotient")
    _fail(module.get("signed_module_independent_input_required") is False, "module input boundary")
    _fail(module.get("real_Gram_radical_dimension") == 9, "real radical dimension")
    _fail(module.get("integer_kernel_rank") == 6, "integer kernel rank")
    _fail(
        module.get("real_Gram_radical_equals_integer_kernel_scalar_extension")
        is False,
        "real/integer radical distinction",
    )

    geometry = receipt.get("fixed_total_source_geometry")
    _fail(isinstance(geometry, Mapping), "fixed-total packet")
    rows = geometry.get("exhaustive_source_rows")
    _fail(isinstance(rows, list) and len(rows) == 5, "exhaustive total rows")
    for total, row in enumerate(rows):
        states = _compositions(total, 12)
        observed = {_apply(difference, state) for state in states}
        expected = _formula_image(total)
        _fail(observed == expected, f"fixed-total formula {total}")
        _fail(row.get("protected_total") == total, f"reported total {total}")
        _fail(row.get("source_state_count") == len(states), f"reported states {total}")
        _fail(
            row.get("distinct_quotient_state_count") == len(observed),
            f"reported quotient states {total}",
        )
    d6_basis = geometry.get("total_one_difference_basis")
    d6_matrix = geometry.get("total_one_difference_basis_matrix")
    _fail(isinstance(d6_basis, list) and len(d6_basis) == 6, "D6 basis")
    _fail(d6_matrix == _transpose(d6_basis), "D6 matrix")
    leading_minors = [
        abs(_determinant([row[:size] for row in d6_matrix[:size]]))
        for size in range(1, 7)
    ]
    _fail(leading_minors == [1, 1, 1, 1, 1, 2], "D6 Smith witness")
    _fail(abs(_determinant(d6_matrix)) == 2, "D6 index determinant")
    _fail(
        geometry.get("leading_unit_minor_and_full_determinant_witness")
        == leading_minors,
        "reported D6 determinant witness",
    )
    _fail(geometry.get("displacement_lattice_index_in_Z6") == 2, "reported D6 index")
    _fail(
        geometry.get("displacement_lattice_Smith_invariants")
        == [1, 1, 1, 1, 1, 2],
        "D6 Smith invariants",
    )
    _fail(geometry.get("one_fixed_total_nonnegative_image_is_finite") is True, "finite sector")
    _fail(geometry.get("one_fixed_total_nonnegative_image_is_dense") is False, "finite sector density")
    _fail(
        geometry.get("positive_total_pairwise_displacement_generated_subgroup")
        == "D6={d in Z^6: sum d_i even}",
        "D6 generated subgroup scope",
    )
    _fail(
        geometry.get("raw_pairwise_displacement_set_at_fixed_total_is_finite")
        is True,
        "finite displacement-set scope",
    )
    _fail(geometry.get("D6_contains_2Z6") is True, "D6 contains 2Z6")
    _fail(
        geometry.get("D6_repair_Gram_image_dense_in_parent_completion") is True,
        "D6 completion density",
    )
    _fail(
        geometry.get("density_uses_displacement_group_completion_not_finite_state_set")
        is True,
        "density scope",
    )

    adjacency = [[0] * 12 for _ in range(12)]
    edges = []
    for left, right in exact["independent_repair_incidence"]["source_edge_list"]:
        adjacency[left][right] = adjacency[right][left] = 1
        edges.append((min(left, right), max(left, right)))
    edges = sorted(set(edges))
    _fail(len(edges) == 30, "source seam count")
    seam_boundary = [[0] * 30 for _ in range(12)]
    for column, (left, right) in enumerate(edges):
        seam_boundary[left][column] = -1
        seam_boundary[right][column] = 1
    seam_current = [
        [
            sum(difference[row][pivot] * seam_boundary[pivot][column] for pivot in range(12))
            for column in range(30)
        ]
        for row in range(6)
    ]
    _fail(
        all(sum(seam_current[row][column] for row in range(6)) % 2 == 0 for column in range(30)),
        "seam currents lie in D6",
    )
    seam_packet = receipt.get("exact_seam_current_boundary")
    _fail(isinstance(seam_packet, Mapping), "seam-current packet")
    _fail(seam_packet.get("source_edges") == [list(edge) for edge in edges], "seam edges")
    _fail(seam_packet.get("boundary_matrix_12_by_30") == seam_boundary, "seam boundary")
    _fail(
        seam_packet.get("signed_seam_current_matrix_D_after_boundary")
        == seam_current,
        "signed seam current",
    )
    witness_columns = tuple(seam_packet.get("index_two_minor_columns", []))
    _fail(len(witness_columns) == 6, "seam minor columns")
    witness = [
        [seam_current[row][column] for column in witness_columns]
        for row in range(6)
    ]
    witness_determinant = _determinant(witness)
    _fail(abs(witness_determinant) == 2, "seam-current index witness")
    _fail(
        seam_packet.get("index_two_minor_determinant") == witness_determinant,
        "reported seam determinant",
    )
    _fail(seam_packet.get("rank") == 6, "seam-current rank")
    _fail(
        seam_packet.get("Smith_invariants") == [1, 1, 1, 1, 1, 2],
        "seam-current Smith invariants",
    )
    _fail(
        seam_packet.get("image_equals_pairwise_displacement_generated_D6")
        is True,
        "seam-current D6 image",
    )

    raw_coordinates = parent_module.get("raw_generator_coordinates_qsqrt5")
    _fail(isinstance(raw_coordinates, list) and len(raw_coordinates) == 6, "raw coordinates")
    source_coordinates: list[tuple[Q5, ...] | None] = [None] * 12
    for port, row in zip(positive, raw_coordinates, strict=True):
        coordinate = tuple(_parse_q5(value) for value in row)
        source_coordinates[port] = coordinate
        source_coordinates[antipodes[port]] = _qneg_vector(coordinate)
    _fail(all(value is not None for value in source_coordinates), "full source coordinates")
    seam_current_coordinates = []
    for column, (left, right) in enumerate(edges):
        chart_row = tuple(
            _qlinear_combination(
                [seam_current[basis][column] for basis in range(6)],
                [
                    source_coordinates[positive[basis]][coordinate]  # type: ignore[index]
                    for basis in range(6)
                ],
            )
            for coordinate in range(3)
        )
        direct_row = tuple(
            _qsub(
                source_coordinates[right][coordinate],  # type: ignore[index]
                source_coordinates[left][coordinate],  # type: ignore[index]
            )
            for coordinate in range(3)
        )
        _fail(chart_row == direct_row, "D-boundary coordinate bridge")
        seam_current_coordinates.append(chart_row)
    _fail(
        seam_packet.get("D_boundary_chart_equals_port_coordinate_difference_exact")
        is True,
        "reported D-boundary coordinate bridge",
    )
    phi: Q5 = (Fraction(1, 2), Fraction(1, 2))
    seam_axes = sorted(
        _qaxis(
            tuple(
                _qmul(phi, seam_current_coordinates[column][coordinate])
                for coordinate in range(3)
            )
        )
        for column in range(30)
    )
    edge30 = port_repair["exact_orbit_ray_table"]["rows"]["edge30"]
    control_axes = sorted(
        _qaxis(tuple(_parse_control_q5(value) for value in row))
        for row in edge30["directions"]
    )
    _fail(seam_axes == control_axes, "edge30 control-axis binding")
    _fail(seam_packet.get("edge30_control_axis_multiset_binding_exact") is True, "reported edge30 binding")
    _fail(seam_packet.get("edge30_control_ray") == edge30["ray"], "edge30 ray")
    _fail(seam_packet.get("edge30_control_ray_digest") == _sha(edge30), "edge30 digest")
    for key in (
        "thirty_seams_are_sixty_or_twelve_translation_events",
        "seam_boundary_columns_are_nonlinear_repair_updates",
        "edge30_orientation_signs_selected",
        "edge30_control_ray_physicalized",
        "spatial_hop_source_certified",
    ):
        _fail(seam_packet.get(key) is False, f"seam boundary {key}")
    _fail(
        seam_packet.get("seam_boundary_columns_are_algebraic_load_currents")
        is True,
        "seam-current algebraic scope",
    )
    laplacian = [
        [5 * int(row == column) - adjacency[row][column] for column in range(12)]
        for row in range(12)
    ]
    t12 = [
        [Fraction(int(row == column)) - Fraction(laplacian[row][column], 60) for column in range(12)]
        for row in range(12)
    ]
    section = [[Fraction()] * 6 for _ in range(12)]
    for column, port in enumerate(positive):
        section[port][column] = Fraction(1)
    difference_fraction = [[Fraction(value) for value in row] for row in difference]
    t6 = _rmatmul(_rmatmul(difference_fraction, t12), section)
    _fail(
        _rmatmul(difference_fraction, t12) == _rmatmul(t6, difference_fraction),
        "mean intertwiner",
    )
    mean = receipt.get("exact_mean_quotient_intertwiner")
    _fail(isinstance(mean, Mapping), "mean packet")
    reported_t6 = [
        [Fraction(value) for value in row] for row in mean.get("induced_operator_q", [])
    ]
    _fail(reported_t6 == t6, "reported induced mean")
    _fail(mean.get("identity_exact") is True, "reported mean identity")
    _fail(mean.get("proper_carrier_equivariant") is True, "mean equivariance")
    _fail(mean.get("mean_readback_only") is True, "mean-only scope")
    _fail(mean.get("unitary_or_physical_propagation_identified") is False, "physical propagation boundary")

    counter = receipt.get("pathwise_descent_counterexample")
    _fail(isinstance(counter, Mapping), "counterexample packet")
    left = tuple(counter.get("state_left", []))
    right = tuple(counter.get("state_right", []))
    _fail(len(left) == len(right) == 12, "counterexample states")
    _fail(sum(left) == sum(right) == 2, "counterexample total")
    _fail(_apply(difference, left) == _apply(difference, right), "counterexample input quotient")
    left_distribution = _distribution(left, edges, difference)
    right_distribution = _distribution(right, edges, difference)
    _fail(left_distribution != right_distribution, "nonlinear descent counterexample")
    _fail(_expectation(left_distribution) == _expectation(right_distribution), "counterexample means")
    _fail(
        _serialized_distribution(counter.get("left_quotient_distribution"))
        == left_distribution,
        "reported left distribution",
    )
    _fail(
        _serialized_distribution(counter.get("right_quotient_distribution"))
        == right_distribution,
        "reported right distribution",
    )
    for key in (
        "one_step_quotient_distributions_equal",
        "full_nonlinear_repair_kernel_descends",
        "individual_seam_events_are_translation_generators",
        "thirty_seam_attempts_identified_with_six_axis_steps",
        "sixty_completion_labels_identified_with_twelve_translations",
    ):
        _fail(counter.get(key) is False, f"counterexample boundary {key}")

    action_completion = action.get("exact_completion_action")
    _fail(isinstance(action_completion, Mapping), "action completion packet")
    _fail(action_completion.get("quotient_action_faithful") is True, "parent action")
    _fail(action_completion.get("all_proper_maps_preserve_antipodal_relations") is True, "parent antipodal action")
    _fail(action_completion.get("source_native_physical_action_promoted") is False, "parent physical action")

    attainment = receipt.get("attainment")
    _fail(isinstance(attainment, Mapping), "attainment packet")
    for key in (
        "source_integer_load_metric_quotient_to_signed_module_derived",
        "conditional_mean_descends_and_is_equivariant",
        "conditional_completion_action_available",
    ):
        _fail(attainment.get(key) is True, f"positive attainment {key}")
    _fail(attainment.get("signed_module_remains_arbitrary_algebraic_control") is False, "module gate")
    for key in (
        "full_pathwise_repair_descent_proved",
        "ordered_history_to_position_descent_proved",
        "operational_position_readback_selected",
        "repair_amendment_adopted",
        "cofinal_refinement_gluing_proved",
        "physical_translation_action_promoted",
        "global_space_promoted",
        "physical_prediction_promoted",
        "comparison_permitted",
    ):
        _fail(attainment.get(key) is False, f"negative attainment {key}")

    boundary = receipt.get("claim_boundary")
    _fail(isinstance(boundary, str), "claim boundary")
    for phrase in (
        "A fixed-total state set itself is finite and is not dense.",
        "The full nonlinear repair kernel does not descend",
        "ordered-history descent",
        "physical action",
    ):
        _fail(phrase in boundary, f"claim boundary phrase {phrase}")

    _check_pins(receipt)
    _check_no_producer_import()
    return {
        "receipt": True,
        "producer_imported": False,
        "integer_metric_quotient": True,
        "fixed_total_images_reconstructed": True,
        "D6_index_and_density": True,
        "mean_intertwiner": True,
        "pathwise_descent": False,
        "physical_position": False,
        "comparison_permitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="?", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    verify_receipt(args.receipt)
    print("PORT_LOAD_METRIC_QUOTIENT_INDEPENDENT_VALID")


if __name__ == "__main__":
    main()
