"""Independent verifier for the conditional port-Gram completion packet.

The verifier does not import the producer.  It reconstructs the exact
``Q(sqrt(5))`` Gram identities, repair-band selection, signed-module rank and
integer injection, and parent boundaries.  It also requires all source,
physical, scale, and comparison promotion flags to remain false.
"""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / "data/repair_closure/port_gram_completion_bridge_receipt.json"
FZ11_RECEIPT = ROOT / "data/repair_closure/fz11_3d_translation_bridge_receipt.json"
PORT_DUAL_RECEIPT = ROOT / "data/repair_closure/primitive_port_dual_measure_receipt.json"
SOURCE_LAW_RECEIPT = ROOT / "data/repair_closure/vertex12_constructive_source_law_receipt.json"
BOUNDED_REPAIR_RECEIPT = (
    ROOT / "data/repair_closure/bounded_atomic_self_readback_closure_receipt.json"
)
PORT_REPAIR_BRIDGE_RECEIPT = (
    ROOT / "data/repair_closure/port_repair_propagation_bridge_receipt.json"
)
CARRIER_MANIFEST = ROOT / "tests/fixtures/echosahedral_federation_reference.json"
PRODUCER_PATH = ROOT / "oph_fpe/dynamics/port_gram_completion_bridge.py"
VERIFIER_PATH = Path(__file__).resolve()
TEST_PATH = ROOT / "tests/test_port_gram_completion_bridge.py"

SCHEMA = "oph.port-gram-hausdorff-completion-bridge.v1"
STATUS = (
    "EXACT_REPAIR_RESPONSE_GRAM_QUOTIENT_AND_3D_COMPLETION_ATTAINED__"
    "A1R_SIGNED_RECORD_MODULE_AND_A2R_POSITION_READBACK_PREMISES_OPEN"
)
FZ11_SCHEMA = "oph.fz11-conditional-3d-translation-bridge.v1"
FZ11_STATUS = (
    "CONDITIONAL_3D_SCALAR_TRANSLATION_ADAPTER_ATTAINED__"
    "CANONICAL_SOURCE_SELECTION_TIME_EVOLUTION_PHOTON_SECTOR_SCALE_FRAME_"
    "BOOST_AND_EXCLUSIVITY_OPEN"
)
PORT_DUAL_SCHEMA = "oph.primitive-port-dual-normalized-measure.v1"
PORT_DUAL_STATUS = (
    "QUOTIENT_VISIBLE_NORMALIZED_PORT_DUAL_MEASURE_ATTAINED__"
    "PHYSICAL_PIXEL_AND_HOP_IDENTITIES_OPEN"
)
SOURCE_LAW_SCHEMA = "oph.vertex12-constructive-source-law-control.v1"
SOURCE_LAW_STATUS = (
    "CONSTRUCTIVE_SOURCE_LAW_CONTROL_ATTAINED__"
    "CANONICAL_A1_A2_A3_SELECTION_AND_PHYSICAL_ATTACHMENT_OPEN"
)
BOUNDED_REPAIR_SCHEMA = "oph.bounded_atomic_self_readback_closure.v1"
BOUNDED_REPAIR_STATUS = (
    "BOUNDED_EXPECTATION_LEVEL_REPAIR_FIXED_POINT_UNIQUE_MODULO_CLOCK_IN_THE_"
    "FROZEN_ADVERSARIAL_SUITE"
)
PORT_REPAIR_BRIDGE_SCHEMA = "oph.port_repair_propagation_bridge_receipt.v1"
PORT_REPAIR_BRIDGE_STATUS = "BOUNDED_NONSELECTION__FZ11_REMAINS_BRANCH_PREDICTION"
POSITIVE_PORTS = (0, 1, 4, 5, 8, 9)
ANTIPODES = (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8)
PORT_COUNT = 12

Q5 = tuple[Fraction, Fraction]
ZERO: Q5 = (Fraction(0), Fraction(0))
ONE: Q5 = (Fraction(1), Fraction(0))
PHI: Q5 = (Fraction(1, 2), Fraction(1, 2))
RAW_COORDINATES: tuple[tuple[Q5, Q5, Q5], ...] = (
    ((-ONE[0], ZERO[1]), PHI, ZERO),
    (ONE, PHI, ZERO),
    ((-ONE[0], ZERO[1]), (-PHI[0], -PHI[1]), ZERO),
    (ONE, (-PHI[0], -PHI[1]), ZERO),
    (ZERO, (-ONE[0], ZERO[1]), PHI),
    (ZERO, ONE, PHI),
    (ZERO, (-ONE[0], ZERO[1]), (-PHI[0], -PHI[1])),
    (ZERO, ONE, (-PHI[0], -PHI[1])),
    (PHI, ZERO, (-ONE[0], ZERO[1])),
    (PHI, ZERO, ONE),
    ((-PHI[0], -PHI[1]), ZERO, (-ONE[0], ZERO[1])),
    ((-PHI[0], -PHI[1]), ZERO, ONE),
)
REPAIR_FRAME_COORDINATES: tuple[tuple[Q5, Q5, Q5], ...] = (
    (ZERO, ONE, PHI),
    (ZERO, ONE, (-PHI[0], -PHI[1])),
    (ZERO, (-ONE[0], ZERO[1]), PHI),
    (ZERO, (-ONE[0], ZERO[1]), (-PHI[0], -PHI[1])),
    (ONE, PHI, ZERO),
    (ONE, (-PHI[0], -PHI[1]), ZERO),
    ((-ONE[0], ZERO[1]), PHI, ZERO),
    ((-ONE[0], ZERO[1]), (-PHI[0], -PHI[1]), ZERO),
    (PHI, ZERO, ONE),
    (PHI, ZERO, (-ONE[0], ZERO[1])),
    ((-PHI[0], -PHI[1]), ZERO, ONE),
    ((-PHI[0], -PHI[1]), ZERO, (-ONE[0], ZERO[1])),
)

TOP_LEVEL_KEYS = {
    "schema",
    "status",
    "issues",
    "comparison_data_read",
    "target_data_read",
    "parent_pins",
    "exact_repair_selected_gram",
    "exact_signed_module_completion",
    "support_hop_isometry_implication",
    "weakest_clause_strengthening",
    "countermodel_controls",
    "attainment",
    "claim_boundary",
    "implementation_pins",
    "receipt_sha256",
}


class IndependentVerificationError(RuntimeError):
    """Raised when an independent completion check fails."""


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentVerificationError(message)


def _exact_keys(value: Any, expected: set[str], label: str) -> None:
    _fail(isinstance(value, Mapping) and set(value) == expected, f"{label} schema")


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
            raise IndependentVerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise IndependentVerificationError(f"non-finite JSON constant is forbidden: {value}")


def _load(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentVerificationError(f"cannot load {path}: {error}") from error
    _fail(isinstance(result, dict), f"{path} is not an object")
    return result


def _qadd(left: Q5, right: Q5) -> Q5:
    return left[0] + right[0], left[1] + right[1]


def _qsub(left: Q5, right: Q5) -> Q5:
    return left[0] - right[0], left[1] - right[1]


def _qmul(left: Q5, right: Q5) -> Q5:
    return (
        left[0] * right[0] + 5 * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _qscale(value: Q5, scalar: Fraction) -> Q5:
    return value[0] * scalar, value[1] * scalar


def _qinv(value: Q5) -> Q5:
    denominator = value[0] ** 2 - 5 * value[1] ** 2
    _fail(denominator != 0, "zero Q(sqrt5) inverse")
    return value[0] / denominator, -value[1] / denominator


def _qconj(value: Q5) -> Q5:
    return value[0], -value[1]


def _qtext(value: Q5) -> str:
    def render(item: Fraction) -> str:
        return str(item.numerator) if item.denominator == 1 else f"{item.numerator}/{item.denominator}"

    return f"{render(value[0])}+{render(value[1])}*sqrt5"


def _qsign(value: Q5) -> int:
    a, b = value
    if b == 0:
        return (a > 0) - (a < 0)
    if a == 0:
        return (b > 0) - (b < 0)
    if a > 0 and b > 0:
        return 1
    if a < 0 and b < 0:
        return -1
    if a > 0:
        return 1 if a * a > 5 * b * b else -1
    return 1 if 5 * b * b > a * a else -1


def _matmul(left: Sequence[Sequence[Q5]], right: Sequence[Sequence[Q5]]) -> list[list[Q5]]:
    _fail(bool(left and right and len(left[0]) == len(right)), "matrix dimensions")
    result: list[list[Q5]] = []
    for i in range(len(left)):
        row: list[Q5] = []
        for j in range(len(right[0])):
            value = ZERO
            for k in range(len(right)):
                value = _qadd(value, _qmul(left[i][k], right[k][j]))
            row.append(value)
        result.append(row)
    return result


def _matscale(matrix: Sequence[Sequence[Q5]], scalar: Q5) -> list[list[Q5]]:
    return [[_qmul(value, scalar) for value in row] for row in matrix]


def _transpose(matrix: Sequence[Sequence[Q5]]) -> list[list[Q5]]:
    _fail(bool(matrix) and all(len(row) == len(matrix[0]) for row in matrix), "transpose shape")
    return [list(column) for column in zip(*matrix, strict=True)]


def _matadd(
    left: Sequence[Sequence[Q5]], right: Sequence[Sequence[Q5]]
) -> list[list[Q5]]:
    _fail(
        len(left) == len(right)
        and all(
            len(left_row) == len(right_row)
            for left_row, right_row in zip(left, right, strict=True)
        ),
        "matrix sum dimensions",
    )
    return [
        [_qadd(lvalue, rvalue) for lvalue, rvalue in zip(lrow, rrow, strict=True)]
        for lrow, rrow in zip(left, right, strict=True)
    ]


def _matsub(
    left: Sequence[Sequence[Q5]], right: Sequence[Sequence[Q5]]
) -> list[list[Q5]]:
    _fail(
        len(left) == len(right)
        and all(
            len(left_row) == len(right_row)
            for left_row, right_row in zip(left, right, strict=True)
        ),
        "matrix difference dimensions",
    )
    return [
        [_qsub(lvalue, rvalue) for lvalue, rvalue in zip(lrow, rrow, strict=True)]
        for lrow, rrow in zip(left, right, strict=True)
    ]


def _dot(left: Sequence[Q5], right: Sequence[Q5]) -> Q5:
    result = ZERO
    for lvalue, rvalue in zip(left, right, strict=True):
        result = _qadd(result, _qmul(lvalue, rvalue))
    return result


def _qsum(values: Sequence[Q5]) -> Q5:
    result = ZERO
    for value in values:
        result = _qadd(result, value)
    return result


def _qdet(matrix: Sequence[Sequence[Q5]]) -> Q5:
    work = [list(row) for row in matrix]
    _fail(bool(work) and all(len(row) == len(work) for row in work), "Q5 determinant shape")
    determinant = ONE
    for column in range(len(work)):
        pivot = next((row for row in range(column, len(work)) if work[row][column] != ZERO), None)
        if pivot is None:
            return ZERO
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = _qscale(determinant, Fraction(-1))
        pivot_value = work[column][column]
        determinant = _qmul(determinant, pivot_value)
        inverse = _qinv(pivot_value)
        for row in range(column + 1, len(work)):
            factor = _qmul(work[row][column], inverse)
            for index in range(column, len(work)):
                work[row][index] = _qsub(work[row][index], _qmul(factor, work[column][index]))
    return determinant


def _fraction_det(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    work = [list(row) for row in matrix]
    _fail(bool(work) and all(len(row) == len(work) for row in work), "Q determinant shape")
    determinant = Fraction(1)
    for column in range(len(work)):
        pivot = next((row for row in range(column, len(work)) if work[row][column]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = -determinant
        value = work[column][column]
        determinant *= value
        for row in range(column + 1, len(work)):
            factor = work[row][column] / value
            for index in range(column, len(work)):
                work[row][index] -= factor * work[column][index]
    return determinant


def _parent(path: Path, schema: str, status: str, issue: int) -> dict[str, Any]:
    receipt = _load(path)
    payload = copy.deepcopy(receipt)
    digest = payload.pop("receipt_sha256", None)
    _fail(digest == _sha(payload), f"{path.name} self digest")
    _fail(receipt.get("schema") == schema, f"{path.name} schema")
    _fail(receipt.get("status") == status, f"{path.name} status")
    _fail(receipt.get("issue") == issue, f"{path.name} issue")
    return receipt


def _bounded_parent(path: Path) -> dict[str, Any]:
    receipt = _load(path)
    payload = copy.deepcopy(receipt)
    digest = payload.pop("certificate_payload_sha256", None)
    _fail(digest == _sha(payload), "bounded repair payload digest")
    _fail(receipt.get("schema") == BOUNDED_REPAIR_SCHEMA, "bounded repair schema")
    _fail(receipt.get("status") == BOUNDED_REPAIR_STATUS, "bounded repair status")
    _fail(
        receipt.get("BOUNDED_EXPECTATION_LEVEL_ATOMIC_SELF_READBACK_FIXED_POINT_RECEIPT")
        is True,
        "bounded one-step repair receipt",
    )
    _fail(receipt.get("PHYSICAL_REPAIR_LAW_RECEIPT") is False, "physical repair boundary")
    _fail(receipt.get("FULL_SELF_READING_UNIVERSE_CLOSURE_RECEIPT") is False, "universe boundary")
    exact = receipt["exact_conditional_mean_bridge"]
    _fail(
        exact["identity"] == "E[X_next | X=x] = (I - L_icosahedron/60) x",
        "bounded mean operator",
    )
    _fail(exact["all_probed_states_exact_identity_verified"] is True, "bounded mean checks")
    word = receipt["conditional_free_event_word_law"]
    _fail(word["canonical_a3_alone_implies_markovity"] is False, "Markov boundary")
    _fail(word["proposed_a1r_a2r_temporal_clauses_required"] is True, "temporal premises")
    clauses = receipt["axiom_clause_specialization"]
    _fail(clauses["canonical_three_axiom_derivation"] is False, "three-axiom boundary")
    _fail(clauses["full_a1_repair_grammar_certified"] is False, "grammar boundary")
    return receipt


def _gram_from_fz(fz11: Mapping[str, Any]) -> list[list[Q5]]:
    frame = fz11["exact_port_frame_and_relabel"]
    _fail(
        frame["raw_coordinates_qsqrt5"]
        == [[_qtext(value) for value in row] for row in RAW_COORDINATES],
        "labeled raw coordinates",
    )
    _fail(frame["source_antipodes"] == list(ANTIPODES), "source antipodes")
    _fail(frame["common_raw_norm_squared"] == "5/2+1/2*sqrt5", "raw norm")
    scaled = frame[
        "source_scaled_gram_5G_qsqrt5_integer_pairs"
    ]
    _fail(isinstance(scaled, list) and len(scaled) == PORT_COUNT, "Gram rows")
    output: list[list[Q5]] = []
    for row in scaled:
        _fail(isinstance(row, list) and len(row) == PORT_COUNT, "Gram columns")
        parsed = []
        for pair in row:
            _fail(
                isinstance(pair, list)
                and len(pair) == 2
                and all(type(value) is int for value in pair),
                "Gram pair",
            )
            parsed.append((Fraction(pair[0], 5), Fraction(pair[1], 5)))
        output.append(parsed)
    return output


def _signed_source_projection(source: Mapping[str, Any]) -> dict[str, Any]:
    law = source["constructive_source_law"]
    alphabet = law["a1_complete_event_alphabet"]
    rows = alphabet["event_rows"]
    raw_rows = law["raw_step_rows"]
    capture = law["source_capture"]
    _fail(alphabet["complete_signed_port_orbit"] is True, "complete signed orbit")
    _fail(alphabet["event_count"] == PORT_COUNT and len(rows) == PORT_COUNT, "event census")
    _fail(alphabet["every_event_accepted_before_capture_hash"] is True, "event capture order")
    _fail(len(raw_rows) == PORT_COUNT, "raw-row census")
    _fail(capture["capture_hash_binds_every_event_payload"] is True, "capture binding")
    _fail(capture["event_count"] == PORT_COUNT, "capture event census")
    _fail(
        capture["source_capture_root_sha256"] == _sha(capture["payload"]),
        "source capture root",
    )
    positive_axis = {port: axis for axis, port in enumerate(POSITIVE_PORTS)}
    projection_rows = []
    for port in range(PORT_COUNT):
        row = rows[port]
        raw_row = raw_rows[port]
        positive_port = port if port in positive_axis else ANTIPODES[port]
        direction = [0] * len(POSITIVE_PORTS)
        direction[positive_axis[positive_port]] = 1 if port in positive_axis else -1
        _fail(row["port"] == port and row["antipodal_port"] == ANTIPODES[port], "event labels")
        _fail(row["raw_direction_in_Z_power_6"] == direction, "event direction")
        _fail(row["event_kind"] == "accepted_signed_axis_unit_record", "event kind")
        _fail(row["comparison_or_target_input_used"] is False, "event firewall")
        _fail(raw_row["port"] == port and raw_row["inverse_port"] == ANTIPODES[port], "raw labels")
        _fail(raw_row["direction"] == direction, "raw direction")
        _fail(raw_row["bijective_on_Z_power_6"] is True, "raw translation")
        _fail(raw_row["event_id"] == row["event_id"], "raw/event identity")
        projection_rows.append(
            {
                "port": port,
                "antipodal_port": ANTIPODES[port],
                "raw_direction_in_Z_power_6": direction,
                "event_id": row["event_id"],
            }
        )
    _fail(capture["payload"]["event_rows"] == rows, "capture event payload")
    return {
        "signed_source_control_id": capture["payload"]["source_law_id"],
        "source_capture_root_sha256": capture["source_capture_root_sha256"],
        "event_rows": projection_rows,
    }


def _repair_qtext(value: Q5) -> str:
    def render(item: Fraction) -> str:
        return str(item.numerator) if item.denominator == 1 else f"{item.numerator}/{item.denominator}"

    if value[1] == 0:
        return render(value[0])
    return f"{render(value[0])} + {render(value[1])}*sqrt(5)"


def _distances(adjacency: Sequence[Sequence[Q5]], start: int) -> list[int]:
    result = [-1] * len(adjacency)
    result[start] = 0
    queue = [start]
    while queue:
        vertex = queue.pop(0)
        for other, edge in enumerate(adjacency[vertex]):
            if edge == ONE and result[other] < 0:
                result[other] = result[vertex] + 1
                queue.append(other)
    return result


def _repair_adjacency(
    port_repair: Mapping[str, Any],
) -> tuple[list[list[Q5]], dict[str, Any]]:
    pin = port_repair["source_packet"]["carrier_manifest_pin"]
    _fail(
        pin["repository_relative_path"] == CARRIER_MANIFEST.relative_to(ROOT).as_posix(),
        "repair carrier path",
    )
    _fail(pin["bytes"] == len(CARRIER_MANIFEST.read_bytes()), "repair carrier bytes")
    _fail(pin["sha256"] == _raw_sha(CARRIER_MANIFEST), "repair carrier hash")
    directions = port_repair["exact_orbit_ray_table"]["rows"]["vertex12"]["directions"]
    _fail(
        directions
        == [[_repair_qtext(value) for value in row] for row in REPAIR_FRAME_COORDINATES],
        "repair frame serialization",
    )
    manifest = _load(CARRIER_MANIFEST)
    _fail(manifest["schema"] == "oph.echosahedral_selector_manifest.v1", "carrier schema")
    carrier = manifest["carrier"]
    ports = carrier["ports"]
    _fail(ports == [f"p{port:02d}" for port in range(PORT_COUNT)], "carrier ports")
    index = {label: position for position, label in enumerate(ports)}
    fixture_edges: set[tuple[int, int]] = set()
    fixture_adjacency = [[ZERO for _ in range(PORT_COUNT)] for _ in range(PORT_COUNT)]
    _fail(len(carrier["edges"]) == 30, "carrier edge census")
    for row in carrier["edges"]:
        _fail(isinstance(row, list) and len(row) == 2, "carrier edge shape")
        _fail(row[0] in index and row[1] in index, "carrier edge labels")
        left, right = sorted((index[row[0]], index[row[1]]))
        _fail(left != right and (left, right) not in fixture_edges, "carrier simple edges")
        fixture_edges.add((left, right))
        fixture_adjacency[left][right] = fixture_adjacency[right][left] = ONE
    _fail(
        all(_qsum(row) == (Fraction(5), Fraction(0)) for row in fixture_adjacency),
        "carrier regularity",
    )
    fixture_to_source = []
    for vector in REPAIR_FRAME_COORDINATES:
        matches = [index for index, candidate in enumerate(RAW_COORDINATES) if candidate == vector]
        _fail(len(matches) == 1, "frame matching uniqueness")
        fixture_to_source.append(matches[0])
    _fail(sorted(fixture_to_source) == list(range(PORT_COUNT)), "frame map bijection")
    adjacency = [[ZERO for _ in range(PORT_COUNT)] for _ in range(PORT_COUNT)]
    source_edges = []
    for left, right in sorted(fixture_edges):
        source_left, source_right = sorted((fixture_to_source[left], fixture_to_source[right]))
        adjacency[source_left][source_right] = adjacency[source_right][source_left] = ONE
        source_edges.append([source_left, source_right])
    for port in range(PORT_COUNT):
        _fail(
            [other for other, distance in enumerate(_distances(adjacency, port)) if distance == 3]
            == [ANTIPODES[port]],
            "repair adjacency antipode",
        )
    packet = {
        "origin": "pinned repair carrier incidence, relabeled by exact frame equality",
        "carrier_manifest_path": CARRIER_MANIFEST.relative_to(ROOT).as_posix(),
        "carrier_manifest_raw_sha256": _raw_sha(CARRIER_MANIFEST),
        "upstream_carrier_manifest_raw_sha256": pin["sha256"],
        "fixture_to_source_port_map": fixture_to_source,
        "source_edge_list": source_edges,
        "source_adjacency_sha256": _sha(
            [[1 if value == ONE else 0 for value in row] for row in adjacency]
        ),
    }
    return adjacency, packet


def _verify_exact_math(
    report: Mapping[str, Any],
    gram: Sequence[Sequence[Q5]],
    source_projection: Mapping[str, Any],
    repair_adjacency: Sequence[Sequence[Q5]],
    incidence_packet: Mapping[str, Any],
) -> None:
    spectral = report["exact_repair_selected_gram"]
    _exact_keys(
        spectral,
        {
            "canonical_centered_response_kernel_derivation",
            "independent_repair_incidence",
            "Gram_class_adjacency_matches_independent_repair_incidence",
            "projector_constructed_from_independent_adjacency_polynomial",
            "projector_polynomial",
            "current_A1R_A2R_adopted",
            "current_A1_selects_between_galois_frames",
            "dynamical_selection_scope",
            "full_gram_qsqrt5",
            "full_spectral_resolution",
            "galois_control_gram_qsqrt5",
            "galois_partner_distinct",
            "galois_partner_eigen_identity",
            "intrinsic_local_carrier",
            "gram_branch_selected_by_declared_repair_cost_if_A1R_A2R_adopted",
            "gram_diagonal",
            "gram_squared_identity",
            "laplacian_eigen_identity",
            "positive_clock_rescaling_changes_selected_eigenspace",
            "repair_generator",
            "selected_band",
            "selected_gram_normalization",
            "selected_projector_trace",
            "slowest_band_selection_is_extra_economy_selector",
            "source_backed_discrete_repair",
            "strict_cost_order",
            "unscaled_laplacian_band_costs",
        },
        "repair-selected Gram",
    )
    _exact_keys(
        spectral["full_spectral_resolution"],
        {"costs", "ranks", "projectors_pairwise_orthogonal", "projectors_resolve_identity"},
        "spectral resolution",
    )
    _exact_keys(
        spectral["source_backed_discrete_repair"],
        {
            "IID_or_temporal_independence_proved",
            "continuous_exponential_semigroup_used",
            "exact_power_formula",
            "formal_operator_powers_equal_physical_n_tick_history",
            "full_temporal_grammar_completeness_proved",
            "one_step_eigenvalues_descending",
            "one_step_expectation_operator",
            "one_step_operator_source_backed_by_pinned_ancestry",
            "physical_repair_law_promoted",
            "strict_subunit_order",
        },
        "discrete repair",
    )
    _exact_keys(
        spectral["canonical_centered_response_kernel_derivation"],
        {
            "probe_family",
            "probe_count",
            "probe_weights",
            "stochastic_initial_ensemble_required",
            "response_vectors",
            "kernel_definition",
            "exact_spectral_formula",
            "exact_for_every_nonnegative_integer_n",
            "unique_largest_nonconstant_factor",
            "common_diagonal_formula",
            "trace_formula",
            "projective_limit",
            "trace_one_limit",
            "trace_twelve_limit",
            "unit_diagonal_limit",
            "limit_before_quotient_and_completion_required",
            "finite_n_centered_rank",
            "finite_n_antipodally_odd_rank",
            "finite_n_signed_module_is_discrete_and_complete",
            "strictly_positive_unequal_probe_weights_preserve_limit_rank_three",
            "unequal_weight_limit_form",
            "unequal_weights_preserve_exact_icosahedral_Gram_angles",
            "named_operational_readback_premise",
            "named_Gram_topology_premise",
            "current_A2_contains_completed_asymptotic_kernel_readback",
            "formal_response_powers_are_physical_time_evolution",
            "target_or_comparison_data_used",
            "port_gram_derived_rather_than_supplied_by_A1_RG",
        },
        "centered response kernel",
    )
    _exact_keys(
        spectral["intrinsic_local_carrier"],
        {
            "ambient",
            "definition",
            "real_dimension",
            "labeled_generator",
            "generator_gram_identity",
            "generator_gram_identity_exact",
            "cartesian_coordinates_used_to_define_carrier",
            "cartesian_chart_role",
            "preferred_cartesian_frame_selected",
            "global_or_physical_space_promoted",
        },
        "intrinsic carrier",
    )
    _exact_keys(
        spectral["independent_repair_incidence"],
        {
            "origin",
            "carrier_manifest_path",
            "carrier_manifest_raw_sha256",
            "upstream_carrier_manifest_raw_sha256",
            "fixture_to_source_port_map",
            "source_edge_list",
            "source_adjacency_sha256",
        },
        "independent repair incidence",
    )
    _fail(
        spectral["independent_repair_incidence"] == incidence_packet,
        "repair incidence serialization",
    )
    _fail(
        spectral["Gram_class_adjacency_matches_independent_repair_incidence"] is True,
        "Gram/incidence comparison field",
    )
    gram_class_adjacency: list[list[Q5]] = []
    for i, row in enumerate(gram):
        arow = []
        for j, value in enumerate(row):
            if i == j:
                _fail(value == ONE, "Gram diagonal")
                arow.append(ZERO)
            elif j == ANTIPODES[i]:
                _fail(value == (-ONE[0], ZERO[1]), "Gram antipode")
                arow.append(ZERO)
            elif value == (Fraction(0), Fraction(1, 5)):
                arow.append(ONE)
            else:
                _fail(value == (Fraction(0), Fraction(-1, 5)), "Gram distance class")
                arow.append(ZERO)
        gram_class_adjacency.append(arow)
    _fail(
        gram_class_adjacency == list(map(list, repair_adjacency)),
        "Gram classes versus independent repair incidence",
    )
    adjacency = list(map(list, repair_adjacency))
    for port in range(PORT_COUNT):
        for other in range(PORT_COUNT):
            _fail(
                gram[ANTIPODES[port]][other]
                == _qscale(gram[port][other], Fraction(-1)),
                "Gram antipodal descent",
            )
    identity = [[ONE if i == j else ZERO for j in range(PORT_COUNT)] for i in range(PORT_COUNT)]
    laplacian = [
        [_qsub(_qscale(identity[i][j], Fraction(5)), adjacency[i][j]) for j in range(PORT_COUNT)]
        for i in range(PORT_COUNT)
    ]
    _fail(_matmul(gram, gram) == _matscale(gram, (Fraction(4), Fraction(0))), "G^2=4G")
    _fail(_qsum([gram[i][i] for i in range(PORT_COUNT)]) == (Fraction(12), Fraction(0)), "trace G")
    low = (Fraction(5), Fraction(-1))
    middle = (Fraction(6), Fraction(0))
    high = (Fraction(5), Fraction(1))
    conjugate = [[_qconj(value) for value in row] for row in gram]
    _fail(_matmul(laplacian, gram) == _matscale(gram, low), "low band")
    _fail(_matmul(laplacian, conjugate) == _matscale(conjugate, high), "Galois band")
    _fail(_qsign(_qsub(middle, low)) > 0 and _qsign(_qsub(high, middle)) > 0, "cost order")
    # Construct the low projector solely from independently pinned incidence.
    # The Gram comparison happens after this polynomial construction, so a
    # circular Gram-to-adjacency reconstruction cannot satisfy the verifier.
    adjacency_minus_5i = _matsub(adjacency, _matscale(identity, (Fraction(5), Fraction(0))))
    adjacency_plus_i = _matadd(adjacency, identity)
    adjacency_plus_sqrt5_i = _matadd(
        adjacency, _matscale(identity, (Fraction(0), Fraction(1)))
    )
    denominator = _qmul(
        _qmul((Fraction(-5), Fraction(1)), (Fraction(1), Fraction(1))),
        (Fraction(0), Fraction(2)),
    )
    projector = _matscale(
        _matmul(
            _matmul(adjacency_minus_5i, adjacency_plus_i),
            adjacency_plus_sqrt5_i,
        ),
        _qinv(denominator),
    )
    _fail(
        _matscale(projector, (Fraction(4), Fraction(0))) == list(map(list, gram)),
        "independent adjacency projector versus parent Gram",
    )
    _fail(_matmul(projector, projector) == projector, "projector idempotence")
    _fail(_qsum([projector[i][i] for i in range(PORT_COUNT)]) == (Fraction(3), Fraction(0)), "projector rank")
    identity = [[ONE if i == j else ZERO for j in range(PORT_COUNT)] for i in range(PORT_COUNT)]
    constant_projector = [
        [(Fraction(1, PORT_COUNT), Fraction(0)) for _ in range(PORT_COUNT)]
        for _ in range(PORT_COUNT)
    ]
    high_projector = _matscale(conjugate, (Fraction(1, 4), Fraction(0)))
    middle_projector = _matsub(
        _matsub(_matsub(identity, constant_projector), projector), high_projector
    )
    projectors = [constant_projector, projector, middle_projector, high_projector]
    for candidate in projectors:
        _fail(_matmul(candidate, candidate) == candidate, "full projector idempotence")
    for left_index, left_projector in enumerate(projectors):
        for right_index, right_projector in enumerate(projectors):
            if left_index != right_index:
                _fail(
                    all(
                        value == ZERO
                        for row in _matmul(left_projector, right_projector)
                        for value in row
                    ),
                    "projector orthogonality",
                )
    _fail(
        _matadd(
            _matadd(constant_projector, projector),
            _matadd(middle_projector, high_projector),
        )
        == identity,
        "projector resolution",
    )
    _fail(
        [_qsum([candidate[i][i] for i in range(PORT_COUNT)]) for candidate in projectors]
        == [
            (Fraction(1), Fraction(0)),
            (Fraction(3), Fraction(0)),
            (Fraction(5), Fraction(0)),
            (Fraction(3), Fraction(0)),
        ],
        "full spectral ranks",
    )
    _fail(_matmul(laplacian, constant_projector) == _matscale(constant_projector, ZERO), "constant band")
    _fail(_matmul(laplacian, middle_projector) == _matscale(middle_projector, middle), "middle band")
    antipode_operator = [
        [ONE if j == ANTIPODES[i] else ZERO for j in range(PORT_COUNT)]
        for i in range(PORT_COUNT)
    ]
    odd_projector = _matscale(
        _matsub(identity, antipode_operator), (Fraction(1, 2), Fraction(0))
    )
    even_projector = _matscale(
        _matadd(identity, antipode_operator), (Fraction(1, 2), Fraction(0))
    )
    _fail(odd_projector == _matadd(projector, high_projector), "odd bands")
    _fail(even_projector == _matadd(constant_projector, middle_projector), "even bands")
    low_tick = (Fraction(11, 12), Fraction(1, 60))
    middle_tick = (Fraction(9, 10), Fraction(0))
    high_tick = (Fraction(11, 12), Fraction(-1, 60))
    tick = _matsub(identity, _matscale(laplacian, (Fraction(1, 60), Fraction(0))))
    _fail(
        tick
        == _matadd(
            _matadd(constant_projector, _matscale(projector, low_tick)),
            _matadd(
                _matscale(middle_projector, middle_tick),
                _matscale(high_projector, high_tick),
            ),
        ),
        "discrete tick decomposition",
    )
    _fail(
        _qsign(high_tick) > 0
        and _qsign(_qsub(middle_tick, high_tick)) > 0
        and _qsign(_qsub(low_tick, middle_tick)) > 0
        and _qsign(_qsub(ONE, low_tick)) > 0,
        "discrete tick eigenvalue order",
    )
    _fail(
        all(projector[i][i] == (Fraction(1, 4), Fraction(0)) for i in range(PORT_COUNT)),
        "low uniform diagonal",
    )
    _fail(
        all(high_projector[i][i] == (Fraction(1, 4), Fraction(0)) for i in range(PORT_COUNT)),
        "high uniform diagonal",
    )
    _fail(
        all(middle_projector[i][i] == (Fraction(5, 12), Fraction(0)) for i in range(PORT_COUNT)),
        "rank-five uniform diagonal",
    )
    centered_projector = _matsub(identity, constant_projector)
    _fail(
        centered_projector
        == _matadd(_matadd(projector, middle_projector), high_projector),
        "centered spectral resolution",
    )
    _fail(_transpose(tick) == tick, "symmetric one-step operator")
    _fail(
        _matmul(tick, centered_projector)
        == _matmul(centered_projector, tick),
        "centering commutes with repair",
    )
    intrinsic_generators = [
        [_qscale(projector[row][port], Fraction(2)) for row in range(PORT_COUNT)]
        for port in range(PORT_COUNT)
    ]
    intrinsic_gram = [
        [_dot(intrinsic_generators[left], intrinsic_generators[right])
         for right in range(PORT_COUNT)]
        for left in range(PORT_COUNT)
    ]
    _fail(intrinsic_gram == list(map(list, gram)), "intrinsic carrier Gram")

    _fail(spectral["selected_gram_normalization"] == "G=4*P_slowest_nonconstant", "selected Gram label")
    _fail(
        spectral["projector_constructed_from_independent_adjacency_polynomial"] is True,
        "independent projector field",
    )
    _fail(
        spectral["projector_polynomial"]
        == (
            "P_low=(A-5I)(A+I)(A+sqrt(5)I)/"
            "((sqrt(5)-5)(sqrt(5)+1)(2sqrt(5)))"
        ),
        "projector polynomial field",
    )
    _fail(spectral["unscaled_laplacian_band_costs"] == [_qtext(low), _qtext(middle), _qtext(high)], "cost serialization")
    _fail(spectral["selected_band"] == "adjacency_+sqrt5__laplacian_5-sqrt5__rank_3", "selected band")
    _fail(spectral["selected_projector_trace"] == "3", "projector trace field")
    _fail(spectral["gram_diagonal"] == "1", "Gram diagonal field")
    _fail(spectral["gram_squared_identity"] == "G^2=4G", "Gram square field")
    _fail(
        spectral["laplacian_eigen_identity"] == "L_ico*G=(5-sqrt5)*G",
        "low eigen-identity field",
    )
    _fail(
        spectral["galois_partner_eigen_identity"]
        == "L_ico*conj(G)=(5+sqrt5)*conj(G)",
        "high eigen-identity field",
    )
    _fail(spectral["strict_cost_order"] == "5-sqrt5 < 6 < 5+sqrt5", "cost-order field")
    _fail(spectral["galois_partner_distinct"] is True, "Galois control")
    _fail(
        spectral["full_spectral_resolution"]
        == {
            "costs": ["0", _qtext(low), _qtext(middle), _qtext(high)],
            "ranks": [1, 3, 5, 3],
            "projectors_pairwise_orthogonal": True,
            "projectors_resolve_identity": True,
        },
        "full spectral resolution fields",
    )
    _fail(
        spectral["slowest_band_selection_is_extra_economy_selector"] is False,
        "dynamical rather than economy selection",
    )
    discrete = spectral["source_backed_discrete_repair"]
    _fail(discrete["one_step_expectation_operator"] == "T=I-L_ico/60", "one-step operator")
    _fail(discrete["one_step_operator_source_backed_by_pinned_ancestry"] is True, "one-step ancestry")
    _fail(
        discrete["exact_power_formula"]
        == (
            "T^n=P_0+((55+sqrt5)/60)^n*P_low+(9/10)^n*P_5+"
            "((55-sqrt5)/60)^n*P_high"
        ),
        "discrete power formula",
    )
    _fail(
        discrete["one_step_eigenvalues_descending"]
        == ["1", _qtext(low_tick), _qtext(middle_tick), _qtext(high_tick)],
        "tick eigenvalue serialization",
    )
    _fail(
        discrete["strict_subunit_order"]
        == "0 < (55-sqrt5)/60 < 9/10 < (55+sqrt5)/60 < 1",
        "tick order field",
    )
    for key in (
        "continuous_exponential_semigroup_used",
        "formal_operator_powers_equal_physical_n_tick_history",
        "IID_or_temporal_independence_proved",
        "full_temporal_grammar_completeness_proved",
        "physical_repair_law_promoted",
    ):
        _fail(discrete[key] is False, f"discrete boundary {key}")
    centered = spectral["canonical_centered_response_kernel_derivation"]
    _fail(centered["probe_family"] == "q_p=Q*e_p for all twelve ports with Q=I-P_0", "centered probes")
    _fail(centered["probe_count"] == PORT_COUNT, "centered probe count")
    _fail(centered["probe_weights"] == "equal source-counting weight", "probe weights")
    _fail(centered["stochastic_initial_ensemble_required"] is False, "ensemble boundary")
    _fail(centered["response_vectors"] == "y_p^(n)=T^n*q_p", "response vectors")
    _fail(
        centered["kernel_definition"]
        == "C_n=(T^n*Q)^T*(T^n*Q)=Q*T^(2n)*Q",
        "centered kernel definition",
    )
    _fail(
        centered["exact_spectral_formula"]
        == (
            "C_n=((55+sqrt5)/60)^(2n)*P_low+(9/10)^(2n)*P_5+"
            "((55-sqrt5)/60)^(2n)*P_high"
        ),
        "centered kernel spectrum",
    )
    _fail(centered["exact_for_every_nonnegative_integer_n"] is True, "centered all-n theorem")
    _fail(centered["unique_largest_nonconstant_factor"] == "(55+sqrt5)/60", "dominant factor")
    _fail(
        centered["common_diagonal_formula"]
        == (
            "diag(C_n)=((55+sqrt5)/60)^(2n)/4+"
            "5*(9/10)^(2n)/12+((55-sqrt5)/60)^(2n)/4"
        ),
        "centered diagonal",
    )
    _fail(
        centered["trace_formula"]
        == (
            "trace(C_n)=3*((55+sqrt5)/60)^(2n)+5*(9/10)^(2n)+"
            "3*((55-sqrt5)/60)^(2n)"
        ),
        "centered trace",
    )
    _fail(centered["trace_one_limit"] == "C_n/trace(C_n) -> P_low/3", "centered trace-one limit")
    _fail(centered["projective_limit"] == "[C_n] -> [P_low]", "centered projective limit")
    _fail(centered["trace_twelve_limit"] == "12*C_n/trace(C_n) -> 4*P_low=G", "centered trace-twelve limit")
    _fail(centered["unit_diagonal_limit"] == "C_n/common_diagonal(C_n) -> 4*P_low=G", "centered diagonal limit")
    _fail(centered["limit_before_quotient_and_completion_required"] is True, "limit order")
    _fail(centered["finite_n_centered_rank"] == 11, "finite centered rank")
    _fail(centered["finite_n_antipodally_odd_rank"] == 6, "finite odd rank")
    _fail(centered["finite_n_signed_module_is_discrete_and_complete"] is True, "finite module completion")
    _fail(
        centered["strictly_positive_unequal_probe_weights_preserve_limit_rank_three"] is True,
        "unequal-weight rank robustness",
    )
    _fail(centered["unequal_weight_limit_form"] == "P_low*W*P_low on range(P_low)", "unequal-weight form")
    _fail(centered["unequal_weights_preserve_exact_icosahedral_Gram_angles"] is False, "unequal-weight angle boundary")
    _fail(centered["current_A2_contains_completed_asymptotic_kernel_readback"] is False, "centered A2 boundary")
    _fail(centered["formal_response_powers_are_physical_time_evolution"] is False, "centered time boundary")
    _fail(centered["target_or_comparison_data_used"] is False, "centered target firewall")
    _fail(centered["port_gram_derived_rather_than_supplied_by_A1_RG"] is True, "centered derived Gram")
    _fail(
        centered["named_operational_readback_premise"]
        == (
            "completed future-repair distinguishability is read through the centered "
            "equal-port response kernel"
        ),
        "operational readback premise",
    )
    _fail(
        centered["named_Gram_topology_premise"]
        == "the scale-normalized asymptotic response kernel defines the port metric",
        "Gram topology premise",
    )
    _fail(
        spectral["positive_clock_rescaling_changes_selected_eigenspace"] is False,
        "clock-rescaling invariance",
    )
    _fail(
        "not promoted" in spectral["dynamical_selection_scope"],
        "conditional discrete-dynamics scope",
    )
    _fail(spectral["gram_branch_selected_by_declared_repair_cost_if_A1R_A2R_adopted"] is True, "conditional branch theorem")
    _fail(spectral["current_A1_selects_between_galois_frames"] is False, "A1 boundary")
    _fail(spectral["current_A1R_A2R_adopted"] is False, "repair amendment boundary")
    _fail(spectral["full_gram_qsqrt5"] == [[_qtext(value) for value in row] for row in gram], "Gram serialization")
    _fail(spectral["galois_control_gram_qsqrt5"] == [[_qtext(value) for value in row] for row in conjugate], "Galois serialization")
    intrinsic = spectral["intrinsic_local_carrier"]
    _fail(intrinsic["ambient"] == "twelve-port source-counting space", "intrinsic ambient")
    _fail(intrinsic["definition"] == "H=range(P_low)", "intrinsic definition")
    _fail(intrinsic["real_dimension"] == 3, "intrinsic dimension")
    _fail(intrinsic["labeled_generator"] == "v_p=2*P_low*e_p", "intrinsic generators")
    _fail(intrinsic["generator_gram_identity"] == "<v_p,v_q>=4*(P_low)_pq=G_pq", "intrinsic Gram field")
    _fail(intrinsic["generator_gram_identity_exact"] is True, "intrinsic Gram exactness")
    _fail(intrinsic["cartesian_coordinates_used_to_define_carrier"] is False, "intrinsic coordinate boundary")
    _fail(intrinsic["preferred_cartesian_frame_selected"] is False, "preferred frame boundary")
    _fail(intrinsic["global_or_physical_space_promoted"] is False, "intrinsic physical boundary")

    raw_norm = _dot(RAW_COORDINATES[0], RAW_COORDINATES[0])
    reconstructed = [
        [_qmul(_dot(left, right), _qinv(raw_norm)) for right in RAW_COORDINATES]
        for left in RAW_COORDINATES
    ]
    _fail(reconstructed == list(map(list, gram)), "coordinate Gram factorization")
    rank_ports = (0, 1, 4)
    rank_matrix = [
        [RAW_COORDINATES[port][coordinate] for port in rank_ports]
        for coordinate in range(3)
    ]
    rank_determinant = _qdet(rank_matrix)
    _fail(rank_determinant != ZERO, "rank witness")
    split_rows: list[list[Fraction]] = []
    for coordinate in range(3):
        rational = []
        irrational = []
        for port in POSITIVE_PORTS:
            a, b = RAW_COORDINATES[port][coordinate]
            rational.append(a - b)
            irrational.append(2 * b)
        split_rows.extend((rational, irrational))
    _fail(_fraction_det(split_rows) == -8, "integer injection determinant")

    completion = report["exact_signed_module_completion"]
    _exact_keys(
        completion,
        {
            "antipodal_relations",
            "full_Gram_antipodal_descent_identity",
            "full_Gram_descends_to_signed_record_quotient",
            "gram6_is_the_descended_positive_port_basis_form",
            "atomic_generator_is_not_a_metric_minimum",
            "completion_theorem",
            "completion_translation_action_is_same_raw_action",
            "constructive_source_control_event_count",
            "constructive_source_control_is_canonical_source_law",
            "constructive_source_control_projection_sha256",
            "continuous_carrier_constructed_as_metric_completion",
            "continuous_carrier_is_primitive_input",
            "continuous_field_assumed",
            "density_argument",
            "gram6_qsqrt5",
            "gram_factorization",
            "group_and_action_extension_formalized_in_Lean",
            "group_and_action_extension_scope",
            "hausdorff_metric",
            "hausdorff_on_integer_records",
            "image_contains",
            "image_dense_in_real_three_space",
            "image_module",
            "integer_kernel_is_zero",
            "integer_kernel_witness",
            "nonzero_integer_records_have_a_shortest_positive_gram_length",
            "overall_positive_metric_scale_selected",
            "physical_continuous_field_selected",
            "positive_port_basis",
            "positive_semidefinite",
            "rank_witness_positive_ports",
            "raw_addition_isometric",
            "raw_generator_coordinates_qsqrt5",
            "raw_rank_witness_determinant",
            "real_kernel_dimension",
            "real_quotient",
            "real_quotient_dimension",
            "real_rank",
            "real_scalar_extension",
            "scalar_field_space_selected",
            "signed_cumulative_port_record_module",
            "single_event_generators_have_unit_gram_norm",
            "translation_extension",
            "ordered_history_to_position_quotient_proved",
            "record_order_and_cost_retained_separately",
            "carrier_position_readback_only",
            "limit_before_quotient_and_completion_required",
            "finite_n_centered_rank",
            "finite_n_antipodally_odd_rank",
            "finite_n_signed_module_is_discrete_and_complete",
            "preferred_cartesian_frame_selected",
            "local_carrier_only",
            "faithful_A5_completion_action_formalized",
            "overlap_refinement_gluing_proved",
        },
        "signed completion",
    )
    _exact_keys(
        completion["integer_kernel_witness"],
        {"coefficient_basis", "six_by_six_rational_matrix", "determinant"},
        "integer kernel witness",
    )
    _fail(
        completion["constructive_source_control_projection_sha256"]
        == _sha(source_projection),
        "signed source projection",
    )
    _fail(
        completion["constructive_source_control_event_count"] == PORT_COUNT,
        "signed source event count",
    )
    _fail(
        completion["constructive_source_control_is_canonical_source_law"] is False,
        "signed source control boundary",
    )
    _fail(completion["positive_port_basis"] == list(POSITIVE_PORTS), "positive basis")
    _fail(
        completion["signed_cumulative_port_record_module"]
        == "M_Z=Z[ports]/<e_antipode(p)+e_p> ~= Z^6",
        "signed cumulative record module",
    )
    _fail(completion["real_scalar_extension"] == "M_R=R tensor_Z M_Z ~= R^6", "real extension")
    _fail(
        completion["antipodal_relations"]
        == [[port, ANTIPODES[port]] for port in POSITIVE_PORTS],
        "antipodal relation serialization",
    )
    _fail(
        completion["full_Gram_antipodal_descent_identity"]
        == "G[antipode(p),q]=-G[p,q]=G[p,antipode(q)]",
        "Gram descent identity field",
    )
    _fail(
        completion["full_Gram_descends_to_signed_record_quotient"] is True,
        "Gram quotient descent",
    )
    _fail(
        completion["gram6_is_the_descended_positive_port_basis_form"] is True,
        "descended Gram6 field",
    )
    gram6 = [[gram[left][right] for right in POSITIVE_PORTS] for left in POSITIVE_PORTS]
    _fail(
        completion["gram6_qsqrt5"]
        == [[_qtext(value) for value in row] for row in gram6],
        "Gram6 serialization",
    )
    positive_coordinates = [RAW_COORDINATES[port] for port in POSITIVE_PORTS]
    _fail(
        completion["raw_generator_coordinates_qsqrt5"]
        == [[_qtext(value) for value in row] for row in positive_coordinates],
        "raw generator serialization",
    )
    _fail(completion["gram_factorization"] == "G6=U^T U/(5/2+sqrt5/2)", "Gram factorization field")
    _fail(completion["positive_semidefinite"] is True, "Gram PSD consequence")
    _fail(completion["real_rank"] == 3 and completion["real_kernel_dimension"] == 3, "real quotient dimensions")
    _fail(completion["raw_rank_witness_determinant"] == _qtext(rank_determinant), "rank determinant serialization")
    _fail(completion["integer_kernel_is_zero"] is True, "integer kernel")
    _fail(completion["integer_kernel_witness"]["determinant"] == "-8", "integer determinant field")
    _fail(completion["image_module"] == "finite-index-8 submodule of Z[phi]^3", "image index")
    _fail(completion["image_contains"] == "8*Z[phi]^3", "image containment")
    _fail(completion["image_dense_in_real_three_space"] is True, "density")
    _fail(
        completion["single_event_generators_have_unit_gram_norm"] is True,
        "atomic generator norm",
    )
    _fail(
        completion["nonzero_integer_records_have_a_shortest_positive_gram_length"]
        is False,
        "dense image has no shortest length",
    )
    _fail(
        completion["atomic_generator_is_not_a_metric_minimum"] is True,
        "atomic/minimum distinction",
    )
    _fail(completion["real_quotient_dimension"] == 3, "completion dimension")
    _fail(completion["real_quotient"] == "H_0=(R tensor M_Z)/ker(G6)", "real quotient")
    _fail(
        completion["hausdorff_metric"] == "d_G(m,n)^2=(m-n)^T G6 (m-n)",
        "source-Gram metric formula",
    )
    _fail(completion["hausdorff_on_integer_records"] is True, "integer Hausdorff metric")
    _fail(completion["continuous_field_assumed"] is False, "continuous input boundary")
    _fail(
        completion["continuous_carrier_constructed_as_metric_completion"] is True,
        "continuous carrier completion",
    )
    _fail(
        completion["continuous_carrier_is_primitive_input"] is False,
        "continuous carrier primitive boundary",
    )
    _fail(completion["scalar_field_space_selected"] is False, "scalar field boundary")
    _fail(completion["physical_continuous_field_selected"] is False, "physical field boundary")
    _fail(completion["raw_addition_isometric"] is True, "raw action")
    _fail(
        completion["completion_theorem"]
        == "the metric completion of (M_Z,d_G) is uniquely isometric to H_0",
        "completion theorem field",
    )
    _fail(
        completion["translation_extension"]
        == "every addition map n->n+m extends uniquely by continuity to H_0",
        "translation extension field",
    )
    _fail(completion["completion_translation_action_is_same_raw_action"] is True, "same action extension")
    _fail(
        completion["group_and_action_extension_scope"]
        == (
            "standard completion theorem for a translation-invariant metric group; "
            "the finite Gram, density, and raw isometry premises are machine checked"
        ),
        "completion theorem scope",
    )
    _fail(
        completion["group_and_action_extension_formalized_in_Lean"] is False,
        "completion formalization boundary",
    )
    _fail(completion["ordered_history_to_position_quotient_proved"] is False, "ordered-history boundary")
    _fail(completion["record_order_and_cost_retained_separately"] is True, "record fiber boundary")
    _fail(completion["carrier_position_readback_only"] is True, "position readback scope")
    _fail(completion["limit_before_quotient_and_completion_required"] is True, "completion limit order")
    _fail(completion["finite_n_centered_rank"] == 11, "completion finite centered rank")
    _fail(completion["finite_n_antipodally_odd_rank"] == 6, "completion finite odd rank")
    _fail(completion["finite_n_signed_module_is_discrete_and_complete"] is True, "finite signed module")
    _fail(completion["preferred_cartesian_frame_selected"] is False, "preferred frame boundary")
    _fail(completion["local_carrier_only"] is True, "local carrier scope")
    _fail(completion["faithful_A5_completion_action_formalized"] is False, "A5 action boundary")
    _fail(completion["overlap_refinement_gluing_proved"] is False, "gluing boundary")
    _fail(completion["overall_positive_metric_scale_selected"] is False, "scale boundary")


def _verify_boundaries(
    report: Mapping[str, Any],
    fz11: Mapping[str, Any],
    dual: Mapping[str, Any],
    source: Mapping[str, Any],
    bounded: Mapping[str, Any],
    port_repair: Mapping[str, Any],
) -> None:
    pins = report["parent_pins"]
    _exact_keys(
        pins,
        {
            "fz11_conditional_adapter",
            "primitive_port_dual_measure",
            "constructive_signed_port_record_control",
            "bounded_one_step_expectation_repair",
            "port_repair_propagation_boundary",
        },
        "parent pins",
    )
    expected_pins = {
        "fz11_conditional_adapter": (FZ11_RECEIPT, FZ11_SCHEMA, FZ11_STATUS, fz11),
        "primitive_port_dual_measure": (PORT_DUAL_RECEIPT, PORT_DUAL_SCHEMA, PORT_DUAL_STATUS, dual),
        "constructive_signed_port_record_control": (SOURCE_LAW_RECEIPT, SOURCE_LAW_SCHEMA, SOURCE_LAW_STATUS, source),
    }
    _fail(
        set(pins)
        == set(expected_pins)
        | {"bounded_one_step_expectation_repair", "port_repair_propagation_boundary"},
        "parent pin names",
    )
    for name, (path, schema, status, parent) in expected_pins.items():
        pin = pins[name]
        expected_keys = {"path", "raw_sha256", "receipt_sha256", "schema", "status"}
        if name == "constructive_signed_port_record_control":
            expected_keys.add("canonical_source_selection")
        _exact_keys(pin, expected_keys, f"{name} pin")
        _fail(pin["path"] == path.relative_to(ROOT).as_posix(), f"{name} path")
        _fail(pin["raw_sha256"] == _raw_sha(path), f"{name} raw hash")
        _fail(pin["receipt_sha256"] == parent["receipt_sha256"], f"{name} receipt hash")
        _fail(pin["schema"] == schema and pin["status"] == status, f"{name} contract")
    _fail(pins["constructive_signed_port_record_control"]["canonical_source_selection"] is False, "source control boundary")
    bounded_pin = pins["bounded_one_step_expectation_repair"]
    _exact_keys(
        bounded_pin,
        {
            "path",
            "raw_sha256",
            "certificate_payload_sha256",
            "schema",
            "status",
            "physical_repair_law_receipt",
            "canonical_A3_alone_implies_markovity",
        },
        "bounded repair pin",
    )
    _fail(
        bounded_pin["path"] == BOUNDED_REPAIR_RECEIPT.relative_to(ROOT).as_posix(),
        "bounded repair path",
    )
    _fail(bounded_pin["raw_sha256"] == _raw_sha(BOUNDED_REPAIR_RECEIPT), "bounded raw pin")
    _fail(
        bounded_pin["certificate_payload_sha256"]
        == bounded["certificate_payload_sha256"],
        "bounded payload pin",
    )
    _fail(
        bounded_pin["schema"] == BOUNDED_REPAIR_SCHEMA
        and bounded_pin["status"] == BOUNDED_REPAIR_STATUS,
        "bounded contract",
    )
    _fail(bounded_pin["physical_repair_law_receipt"] is False, "bounded physical boundary")
    _fail(bounded_pin["canonical_A3_alone_implies_markovity"] is False, "bounded Markov boundary")
    repair_pin = pins["port_repair_propagation_boundary"]
    _exact_keys(
        repair_pin,
        {
            "path",
            "raw_sha256",
            "receipt_sha256",
            "schema",
            "status",
            "one_step_operator",
            "spatial_port_hop_source_receipt",
            "same_operator_physical_readout_receipt",
        },
        "port repair pin",
    )
    _fail(
        repair_pin["path"] == PORT_REPAIR_BRIDGE_RECEIPT.relative_to(ROOT).as_posix(),
        "port repair path",
    )
    _fail(repair_pin["raw_sha256"] == _raw_sha(PORT_REPAIR_BRIDGE_RECEIPT), "port repair raw pin")
    _fail(repair_pin["receipt_sha256"] == port_repair["receipt_sha256"], "port repair digest pin")
    _fail(
        repair_pin["schema"] == PORT_REPAIR_BRIDGE_SCHEMA
        and repair_pin["status"] == PORT_REPAIR_BRIDGE_STATUS,
        "port repair contract",
    )
    _fail(repair_pin["one_step_operator"] == "T = I - L_icosahedron/60", "port repair operator")
    _fail(repair_pin["spatial_port_hop_source_receipt"] is False, "spatial hop boundary")
    _fail(repair_pin["same_operator_physical_readout_receipt"] is False, "physical readout boundary")
    internal = port_repair["source_packet"]["internal_seam_repair"]
    _fail(internal["operator"] == "T = I - L_icosahedron/60", "ancestry operator")
    _fail(
        internal["bounded_atomic_receipt_payload_sha256"]
        == bounded["certificate_payload_sha256"],
        "bounded-to-port ancestry",
    )
    _fail(internal["physical_repair_law_receipt"] is False, "ancestry physical boundary")
    _fail(port_repair["SPATIAL_PORT_HOP_SOURCE_RECEIPT"] is False, "port spatial boundary")
    _fail(port_repair["SAME_OPERATOR_PHYSICAL_READOUT_RECEIPT"] is False, "port readout boundary")
    _fail(port_repair["epistemic_boundary"]["comparison_data_read"] is False, "port comparison firewall")

    support = report["support_hop_isometry_implication"]
    _exact_keys(
        support,
        {
            "auxiliary_adapter_normalized_support_and_hop_directions_equal_by_definition",
            "auxiliary_coordinate_equality_is_source_semantic_identity",
            "common_finite_geometry_hash",
            "conditional_common_completion_object",
            "conditional_hop_symbol_scope",
            "conditional_identity_requires_A2_cauchy_readback_clause",
            "conditional_primitive_hop",
            "conditional_support_embedding",
            "normalized_labeled_frame_isometry",
            "dimensionful_support_radius_over_hop_selected",
            "source_semantic_identity_required",
            "dimensionful_support_and_hop_vectors_equal_by_definition",
            "labeled_spanning_equal_gram_isometry_theorem",
            "physical_areal_radius_selected",
            "physical_pixel_identified",
            "port_to_support_vertex_map",
            "support_and_hop_share_semantic_object_in_current_source",
            "support_frame_gram_equals_selected_repair_gram",
        },
        "support-hop implication",
    )
    fz_hash = fz11["exact_port_frame_and_relabel"]["source_geometry_hash"]
    _fail(fz_hash == dual["source_scope"]["source_geometry_hash"], "common geometry")
    _fail(dual["source_scope"]["port_to_defect_vertex_bijection"] == list(range(PORT_COUNT)), "port-support map")
    _fail(support["common_finite_geometry_hash"] == fz_hash, "support geometry field")
    _fail(support["port_to_support_vertex_map"] == list(range(PORT_COUNT)), "support map field")
    _fail(support["support_frame_gram_equals_selected_repair_gram"] is True, "equal Gram theorem")
    _fail(support["normalized_labeled_frame_isometry"] is True, "normalized frame isometry")
    _fail(support["dimensionful_support_radius_over_hop_selected"] is False, "dimensionful ratio boundary")
    _fail(support["source_semantic_identity_required"] is True, "semantic identity gate")
    _fail(
        support["conditional_hop_symbol_scope"]
        == (
            "a is the common norm of one atomic signed-port event, not a shortest "
            "nonzero translation length"
        ),
        "conditional hop scope",
    )
    _fail(
        support[
            "auxiliary_adapter_normalized_support_and_hop_directions_equal_by_definition"
        ]
        is True,
        "auxiliary normalized-direction equality",
    )
    _fail(
        support["dimensionful_support_and_hop_vectors_equal_by_definition"]
        is False,
        "dimensionful rescaling boundary",
    )
    _fail(
        support["auxiliary_coordinate_equality_is_source_semantic_identity"]
        is False,
        "coordinate/semantic identity boundary",
    )
    _fail(support["support_and_hop_share_semantic_object_in_current_source"] is False, "same-object boundary")
    _fail(support["conditional_identity_requires_A2_cauchy_readback_clause"] is True, "A2 premise marker")
    _fail(support["physical_pixel_identified"] is False and support["physical_areal_radius_selected"] is False, "physical support boundary")

    clauses = report["weakest_clause_strengthening"]
    _exact_keys(
        clauses,
        {
            "A1_RG",
            "A2_RC",
            "overall_clock_or_length_unit_left_free",
            "proposed_label",
            "why_no_fourth_axiom_is_needed_if_adopted",
        },
        "clause strengthening",
    )
    _fail(
        clauses["proposed_label"]
        == "A1-RG/A2-RC cumulative port-record completion clause",
        "clause label",
    )
    _fail(len(clauses["A1_RG"]) == 3 and len(clauses["A2_RC"]) == 3, "clause census")
    _fail(clauses["overall_clock_or_length_unit_left_free"] is True, "common scale boundary")

    controls = report["countermodel_controls"]
    _exact_keys(
        controls,
        {
            "completion_without_source_control",
            "finite_n_completion_control",
            "response_kernel_controls",
            "dense_hop_control",
            "finite_quotient_control",
            "galois_branch_control",
            "independent_rescaling_control",
        },
        "countermodel controls",
    )
    _exact_keys(
        controls["galois_branch_control"],
        {"weaker_clause", "surviving_models", "separated_only_by_repair_cost_projector"},
        "Galois control",
    )
    _exact_keys(
        controls["independent_rescaling_control"],
        {"weaker_clause", "family", "all_current_finite_invariants_preserved", "R_A_over_a_remains_free"},
        "rescaling control",
    )
    _exact_keys(
        controls["finite_quotient_control"],
        {"meaning", "fails_metric_factorization_reason", "finite_endpoint_quotient_is_physical_translation_completion"},
        "finite quotient control",
    )
    _exact_keys(
        controls["dense_hop_control"],
        {
            "single_atomic_event_has_unit_normalized_gram_norm",
            "arbitrarily_small_nonzero_composite_translations_exist",
            "atomic_event_length_is_a_minimum_lattice_spacing",
        },
        "dense hop control",
    )
    _exact_keys(
        controls["completion_without_source_control"],
        {"mathematical_completion_exists", "source_native_physical_action_follows_without_A1_RG_A2_RC"},
        "completion source control",
    )
    _exact_keys(
        controls["finite_n_completion_control"],
        {
            "finite_n_centered_response_rank",
            "finite_n_antipodally_odd_response_rank",
            "three_dimensional_completion_before_normalized_limit",
            "normalized_infinite_response_limit_is_load_bearing",
        },
        "finite-n completion control",
    )
    _exact_keys(
        controls["response_kernel_controls"],
        {
            "without_scale_normalization_raw_kernel_limit",
            "asymptotic_readback_normalization_is_load_bearing",
            "unequal_probe_weight_control",
            "equal_probe_counting_is_load_bearing_for_exact_icosahedral_angles",
            "equal_probe_counting_is_load_bearing_for_dimension_three",
        },
        "response-kernel controls",
    )
    _fail(set(controls["galois_branch_control"]["surviving_models"]) == {"G", "conj(G)"}, "Galois models")
    _fail(controls["independent_rescaling_control"]["R_A_over_a_remains_free"] is True, "rescaling control")
    _fail(controls["finite_quotient_control"]["finite_endpoint_quotient_is_physical_translation_completion"] is False, "finite quotient control")
    _fail(
        controls["dense_hop_control"]
        == {
            "single_atomic_event_has_unit_normalized_gram_norm": True,
            "arbitrarily_small_nonzero_composite_translations_exist": True,
            "atomic_event_length_is_a_minimum_lattice_spacing": False,
        },
        "dense hop control",
    )
    _fail(controls["completion_without_source_control"]["source_native_physical_action_follows_without_A1_RG_A2_RC"] is False, "omitted clause control")
    _fail(
        controls["response_kernel_controls"]["without_scale_normalization_raw_kernel_limit"]
        == "0 on the centered subspace",
        "normalization control",
    )
    _fail(
        controls["response_kernel_controls"]["asymptotic_readback_normalization_is_load_bearing"]
        is True,
        "readback control",
    )
    _fail(
        controls["response_kernel_controls"]["equal_probe_counting_is_load_bearing_for_exact_icosahedral_angles"]
        is True,
        "equal-probe control",
    )
    _fail(
        controls["response_kernel_controls"]["equal_probe_counting_is_load_bearing_for_dimension_three"]
        is False,
        "unequal-probe dimension robustness",
    )
    _fail(
        controls["finite_n_completion_control"]
        == {
            "finite_n_centered_response_rank": 11,
            "finite_n_antipodally_odd_response_rank": 6,
            "three_dimensional_completion_before_normalized_limit": False,
            "normalized_infinite_response_limit_is_load_bearing": True,
        },
        "finite-n limit-order control",
    )

    attainment = report["attainment"]
    _exact_keys(
        attainment,
        {
            "A1R_A2R_repair_amendment_adopted",
            "A2_cauchy_operational_completion_clause_present",
            "operational_Gram_Cauchy_readback_and_topology_selected",
            "canonical_signed_port_record_source_selected",
            "comparison_permitted",
            "completion_is_three_dimensional_euclidean_vector_group",
            "continuous_carrier_constructed_as_metric_completion",
            "exact_lowest_repair_band_selects_port_gram",
            "source_conditional_mean_T_exact",
            "galois_branch_separated_by_exact_cost",
            "issue_662_armed",
            "overall_physical_scale_selected",
            "physical_P_pixel_is_primitive_port_sector",
            "physical_prediction_promoted",
            "physical_three_space_promoted",
            "raw_translation_extends_uniquely_to_completion",
            "same_semantic_support_translation_object_emitted",
            "scalar_field_space_selected",
            "signed_record_image_dense",
            "signed_integer_record_metric_hausdorff",
            "signed_module_gram_rank_three",
            "source_native_physical_translation_promoted",
            "support_areal_radius_is_primitive_hop_promoted",
            "support_hop_equal_gram_isometry_implication",
            "faithful_A5_completion_action_formalized",
            "overlap_refinement_gluing_proved",
            "global_carrier_promoted",
        },
        "attainment",
    )
    true_keys = {
        "exact_lowest_repair_band_selects_port_gram",
        "source_conditional_mean_T_exact",
        "galois_branch_separated_by_exact_cost",
        "signed_module_gram_rank_three",
        "signed_integer_record_metric_hausdorff",
        "signed_record_image_dense",
        "completion_is_three_dimensional_euclidean_vector_group",
        "continuous_carrier_constructed_as_metric_completion",
        "raw_translation_extends_uniquely_to_completion",
        "support_hop_equal_gram_isometry_implication",
    }
    false_keys = set(attainment) - true_keys
    _fail(all(attainment.get(key) is True for key in true_keys), "mathematical attainment")
    _fail(all(attainment.get(key) is False for key in false_keys), "promotion boundary")
    _fail(
        {
            "canonical_signed_port_record_source_selected",
            "A1R_A2R_repair_amendment_adopted",
            "A2_cauchy_operational_completion_clause_present",
            "operational_Gram_Cauchy_readback_and_topology_selected",
            "same_semantic_support_translation_object_emitted",
            "source_native_physical_translation_promoted",
            "physical_three_space_promoted",
            "faithful_A5_completion_action_formalized",
            "overlap_refinement_gluing_proved",
            "global_carrier_promoted",
            "scalar_field_space_selected",
            "physical_P_pixel_is_primitive_port_sector",
            "support_areal_radius_is_primitive_hop_promoted",
            "overall_physical_scale_selected",
            "physical_prediction_promoted",
            "comparison_permitted",
            "issue_662_armed",
        }
        <= false_keys,
        "required false-boundary keys",
    )


def verify_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    report = _load(path)
    _fail(set(report) == TOP_LEVEL_KEYS, "top-level schema")
    payload = copy.deepcopy(report)
    digest = payload.pop("receipt_sha256")
    _fail(digest == _sha(payload), "receipt self digest")
    _fail(report["schema"] == SCHEMA and report["status"] == STATUS, "receipt contract")
    _fail(report["issues"] == [655, 663, 664], "issue list")
    _fail(report["comparison_data_read"] is False and report["target_data_read"] is False, "data firewall")

    fz11 = _parent(FZ11_RECEIPT, FZ11_SCHEMA, FZ11_STATUS, 655)
    dual = _parent(PORT_DUAL_RECEIPT, PORT_DUAL_SCHEMA, PORT_DUAL_STATUS, 664)
    source = _parent(SOURCE_LAW_RECEIPT, SOURCE_LAW_SCHEMA, SOURCE_LAW_STATUS, 655)
    bounded = _bounded_parent(BOUNDED_REPAIR_RECEIPT)
    port_repair = _parent(
        PORT_REPAIR_BRIDGE_RECEIPT,
        PORT_REPAIR_BRIDGE_SCHEMA,
        PORT_REPAIR_BRIDGE_STATUS,
        655,
    )
    _fail(fz11["comparison_data_read"] is False, "FZ-11 data firewall")
    _fail(
        dual["comparison_data_read"] is False and dual["target_data_read"] is False,
        "port-dual data firewall",
    )
    _fail(source["comparison_data_read"] is False, "source-control data firewall")
    repair_adjacency, incidence_packet = _repair_adjacency(port_repair)
    _verify_exact_math(
        report,
        _gram_from_fz(fz11),
        _signed_source_projection(source),
        repair_adjacency,
        incidence_packet,
    )
    _verify_boundaries(report, fz11, dual, source, bounded, port_repair)

    expected_pins = {
        path.relative_to(ROOT).as_posix(): (len(path.read_bytes()), _raw_sha(path))
        for path in (PRODUCER_PATH, VERIFIER_PATH, TEST_PATH)
    }
    pins = report["implementation_pins"]
    _fail(isinstance(pins, list) and len(pins) == 3, "implementation pin count")
    observed = {
        row["path"]: (row["bytes"], row["sha256"])
        for row in pins
        if isinstance(row, Mapping) and set(row) == {"path", "bytes", "sha256"}
    }
    _fail(observed == expected_pins, "implementation pins")
    _fail(isinstance(report["claim_boundary"], str) and "does not" in report["claim_boundary"], "claim boundary")
    return {
        "receipt": True,
        "producer_imported": False,
        "exact_Qsqrt5_math_reimplemented": True,
        "repair_selected_gram": True,
        "dense_completion_implication": True,
        "raw_record_addition_extension": True,
        "same_semantic_object_emitted": False,
        "physical_translation_promoted": False,
        "issue_662_armed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="?", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    verify_receipt(args.receipt)
    print("PORT_GRAM_COMPLETION_BRIDGE_INDEPENDENT_PASS")


if __name__ == "__main__":
    main()
