"""Certify the conditional port-Gram completion route for issues #655/#663/#664.

The exact twelve-port Gram form is four times the projector onto the lowest
positive band of the icosahedral seam Laplacian.  Its Galois conjugate is the
other rank-three band and has the largest positive seam cost.  The pinned
bounded repair ancestry emits the one-step expectation operator
``T = I - L/60``.  The deterministic Gram of the complete centered one-hot
port-probe responses is ``C_n=Q*T^(2n)*Q``.  Its scale-normalized limit selects
the Gram ray without a stochastic initial ensemble.  This is a discrete
conditional derivation, not a continuous-time law or an economy selector.

Intrinsically, the selected carrier is ``range(P_low)`` in the twelve-port
counting space, with labeled generators ``v_p=2*P_low*e_p`` and Gram ``G``.
On the signed cumulative antipodal record module ``M_Z = Z^6`` this form has
no integer null vector.  Its image is dense in the real rank-three quotient,
so addition extends to translations of the carrier-position metric
completion.  Cartesian icosahedral coordinates are used only as an isometric
chart and arithmetic density witness.  No scalar field space is selected.

This module deliberately does not promote the implication.  The current
``Z^6`` packet is a constructive control rather than the selected canonical
source law, A1-R/A2-R have not been adopted, and A2 has no asymptotic
carrier-position response topology.  The normalized infinite-response limit
must be taken before quotient and completion.  Ordered histories, exact
records, and repair costs remain a separate fiber.  Temporal grammar
completeness, action/refinement gluing, the physical pixel, overall length
scale, field sector, time law, and comparison remain open.  No public or
target data are read.
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
PRODUCER_PATH = Path(__file__).resolve()
VERIFIER_PATH = ROOT / "oph_fpe/dynamics/verify_port_gram_completion_bridge_independent.py"
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

PORT_COUNT = 12
POSITIVE_PORTS = (0, 1, 4, 5, 8, 9)
ANTIPODES = (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8)

Q5 = tuple[Fraction, Fraction]
ZERO: Q5 = (Fraction(0), Fraction(0))
ONE: Q5 = (Fraction(1), Fraction(0))
SQRT5: Q5 = (Fraction(0), Fraction(1))
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

# Exact frame serialized by the independently pinned repair/carrier ancestor,
# in carrier-manifest port order.  This is kept separate from
# ``RAW_COORDINATES``, whose order comes from the FZ/source-frame parent.
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

CLAIM_BOUNDARY = (
    "The exact finite implication identifies the normalized icosahedral port "
    "Gram as four times the lowest-positive spectral projector of the declared "
    "seam Laplacian. The pinned source supplies one expectation-level discrete "
    "repair step; deriving the Gram as its normalized centered future-response "
    "kernel requires the named A2 carrier-position readback/topology premise. "
    "The intrinsic range of the selected projector is a rank-three local "
    "carrier with the port Gram, before any Cartesian chart is introduced. On "
    "the signed cumulative antipodal record module its position-metric "
    "completion is a three-dimensional Euclidean vector group, and raw record "
    "addition extends to translations of that completion. The normalized "
    "infinite-response limit must be taken before quotient and completion. "
    "Ordered history, exact records, and record cost remain a separate fiber. "
    "The support port frame has the same normalized Gram and is therefore "
    "related by a labeled isometry, while its dimensionful radius remains "
    "independent of the primitive hop until a source semantic identity is "
    "proved. The current source does not select the signed Z^6 module, the "
    "A1-R/A2-R repair amendment is not adopted, and A2 does not contain those "
    "completion/readback clauses. Ordered-history descent, temporal "
    "completeness, faithful action and overlap/refinement gluing, and the "
    "physical repair law remain open. No global or physical space, pixel, "
    "scale, field, time law, prediction, or comparison is promoted, and issue "
    "#662 remains unarmed."
)


class PortGramCompletionError(RuntimeError):
    """Raised when the conditional completion packet fails closed."""


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


def _raw_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PortGramCompletionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise PortGramCompletionError(f"non-finite JSON constant is forbidden: {value}")


def _load_json_strict(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PortGramCompletionError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise PortGramCompletionError(f"{path} is not a JSON object")
    return value


def _validated_self_digest(
    path: Path, *, schema: str, status: str, issue: int
) -> dict[str, Any]:
    receipt = _load_json_strict(path)
    payload = copy.deepcopy(receipt)
    digest = payload.pop("receipt_sha256", None)
    if (
        digest != _sha(payload)
        or receipt.get("schema") != schema
        or receipt.get("status") != status
        or receipt.get("issue") != issue
    ):
        raise PortGramCompletionError(f"parent contract drifted: {path}")
    return receipt


def _validated_bounded_repair_receipt(path: Path) -> dict[str, Any]:
    receipt = _load_json_strict(path)
    payload = copy.deepcopy(receipt)
    digest = payload.pop("certificate_payload_sha256", None)
    try:
        exact_mean = receipt["exact_conditional_mean_bridge"]
        word_law = receipt["conditional_free_event_word_law"]
        clauses = receipt["axiom_clause_specialization"]
    except (KeyError, TypeError) as error:
        raise PortGramCompletionError("bounded repair contract fields are absent") from error
    if (
        digest != _sha(payload)
        or receipt.get("schema") != BOUNDED_REPAIR_SCHEMA
        or receipt.get("status") != BOUNDED_REPAIR_STATUS
        or receipt.get("BOUNDED_EXPECTATION_LEVEL_ATOMIC_SELF_READBACK_FIXED_POINT_RECEIPT")
        is not True
        or receipt.get("PHYSICAL_REPAIR_LAW_RECEIPT") is not False
        or receipt.get("FULL_SELF_READING_UNIVERSE_CLOSURE_RECEIPT") is not False
        or exact_mean.get("identity")
        != "E[X_next | X=x] = (I - L_icosahedron/60) x"
        or exact_mean.get("all_probed_states_exact_identity_verified") is not True
        or word_law.get("canonical_a3_alone_implies_markovity") is not False
        or word_law.get("proposed_a1r_a2r_temporal_clauses_required") is not True
        or clauses.get("canonical_three_axiom_derivation") is not False
        or clauses.get("full_a1_repair_grammar_certified") is not False
    ):
        raise PortGramCompletionError("bounded repair contract drifted")
    return receipt


def _validated_port_repair_bridge(path: Path, bounded: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _validated_self_digest(
        path,
        schema=PORT_REPAIR_BRIDGE_SCHEMA,
        status=PORT_REPAIR_BRIDGE_STATUS,
        issue=655,
    )
    try:
        source = receipt["source_packet"]["internal_seam_repair"]
        classification = receipt["classification"]
        boundary = receipt["epistemic_boundary"]
    except (KeyError, TypeError) as error:
        raise PortGramCompletionError("port-repair ancestry fields are absent") from error
    if (
        receipt.get("INTERNAL_SEAM_REPAIR_CERTIFIED") is not True
        or receipt.get("SPATIAL_PORT_HOP_SOURCE_RECEIPT") is not False
        or receipt.get("SAME_OPERATOR_PHYSICAL_READOUT_RECEIPT") is not False
        or source.get("operator") != "T = I - L_icosahedron/60"
        or source.get("bounded_atomic_receipt_payload_sha256")
        != bounded.get("certificate_payload_sha256")
        or source.get("physical_repair_law_receipt") is not False
        or source.get("spatial_translation_identification") is not False
        or source.get("same_operator_physical_readout") is not False
        or classification.get("internal_seam_repair_certified") is not True
        or classification.get("spatial_hop_source_certified") is not False
        or classification.get("same_operator_physical_readout_certified") is not False
        or boundary.get("comparison_data_read") is not False
        or boundary.get("physical_prediction_unsealed") is not False
    ):
        raise PortGramCompletionError("port-repair ancestry contract drifted")
    return receipt


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
    if denominator == 0:
        raise PortGramCompletionError("attempted to invert zero in Q(sqrt(5))")
    return value[0] / denominator, -value[1] / denominator


def _qconj(value: Q5) -> Q5:
    return value[0], -value[1]


def _qtext(value: Q5) -> str:
    def fraction_text(item: Fraction) -> str:
        return str(item.numerator) if item.denominator == 1 else f"{item.numerator}/{item.denominator}"

    return f"{fraction_text(value[0])}+{fraction_text(value[1])}*sqrt5"


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


def _qlt(left: Q5, right: Q5) -> bool:
    return _qsign(_qsub(right, left)) > 0


def _dot(left: Sequence[Q5], right: Sequence[Q5]) -> Q5:
    result = ZERO
    for lvalue, rvalue in zip(left, right, strict=True):
        result = _qadd(result, _qmul(lvalue, rvalue))
    return result


def _matmul(left: Sequence[Sequence[Q5]], right: Sequence[Sequence[Q5]]) -> list[list[Q5]]:
    if not left or not right or len(left[0]) != len(right):
        raise PortGramCompletionError("incompatible exact matrix product")
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


def _qsum(values: Sequence[Q5]) -> Q5:
    result = ZERO
    for value in values:
        result = _qadd(result, value)
    return result


def _matscale(matrix: Sequence[Sequence[Q5]], scalar: Q5) -> list[list[Q5]]:
    return [[_qmul(value, scalar) for value in row] for row in matrix]


def _transpose(matrix: Sequence[Sequence[Q5]]) -> list[list[Q5]]:
    if not matrix or any(len(row) != len(matrix[0]) for row in matrix):
        raise PortGramCompletionError("exact transpose requires a rectangular matrix")
    return [list(column) for column in zip(*matrix, strict=True)]


def _matadd(
    left: Sequence[Sequence[Q5]], right: Sequence[Sequence[Q5]]
) -> list[list[Q5]]:
    if len(left) != len(right) or any(
        len(left_row) != len(right_row)
        for left_row, right_row in zip(left, right, strict=True)
    ):
        raise PortGramCompletionError("incompatible exact matrix sum")
    return [
        [_qadd(lvalue, rvalue) for lvalue, rvalue in zip(lrow, rrow, strict=True)]
        for lrow, rrow in zip(left, right, strict=True)
    ]


def _matsub(
    left: Sequence[Sequence[Q5]], right: Sequence[Sequence[Q5]]
) -> list[list[Q5]]:
    if len(left) != len(right) or any(
        len(left_row) != len(right_row)
        for left_row, right_row in zip(left, right, strict=True)
    ):
        raise PortGramCompletionError("incompatible exact matrix difference")
    return [
        [_qsub(lvalue, rvalue) for lvalue, rvalue in zip(lrow, rrow, strict=True)]
        for lrow, rrow in zip(left, right, strict=True)
    ]


def _matrix_equal(left: Sequence[Sequence[Q5]], right: Sequence[Sequence[Q5]]) -> bool:
    return list(map(list, left)) == list(map(list, right))


def _qdet(matrix: Sequence[Sequence[Q5]]) -> Q5:
    work = [list(row) for row in matrix]
    if not work or any(len(row) != len(work) for row in work):
        raise PortGramCompletionError("exact determinant requires a square matrix")
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
                work[row][index] = _qsub(
                    work[row][index], _qmul(factor, work[column][index])
                )
    return determinant


def _fraction_determinant(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    work = [list(row) for row in matrix]
    if not work or any(len(row) != len(work) for row in work):
        raise PortGramCompletionError("rational determinant requires a square matrix")
    determinant = Fraction(1)
    for column in range(len(work)):
        pivot = next((row for row in range(column, len(work)) if work[row][column]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = -determinant
        pivot_value = work[column][column]
        determinant *= pivot_value
        for row in range(column + 1, len(work)):
            factor = work[row][column] / pivot_value
            for index in range(column, len(work)):
                work[row][index] -= factor * work[column][index]
    return determinant


def _parse_parent_gram(fz11: Mapping[str, Any]) -> list[list[Q5]]:
    try:
        frame = fz11["exact_port_frame_and_relabel"]
        scaled = frame[
            "source_scaled_gram_5G_qsqrt5_integer_pairs"
        ]
    except (KeyError, TypeError) as error:
        raise PortGramCompletionError("FZ-11 exact Gram is absent") from error
    expected_coordinates = [
        [_qtext(value) for value in row] for row in RAW_COORDINATES
    ]
    if (
        frame.get("raw_coordinates_qsqrt5") != expected_coordinates
        or frame.get("source_antipodes") != list(ANTIPODES)
        or frame.get("common_raw_norm_squared") != "5/2+1/2*sqrt5"
    ):
        raise PortGramCompletionError("FZ-11 labeled source frame drifted")
    if (
        not isinstance(scaled, list)
        or len(scaled) != PORT_COUNT
        or any(not isinstance(row, list) or len(row) != PORT_COUNT for row in scaled)
    ):
        raise PortGramCompletionError("FZ-11 exact Gram has the wrong shape")
    result: list[list[Q5]] = []
    for row in scaled:
        parsed: list[Q5] = []
        for value in row:
            if (
                not isinstance(value, list)
                or len(value) != 2
                or any(type(item) is not int for item in value)
            ):
                raise PortGramCompletionError("FZ-11 Gram entry is not an integer pair")
            parsed.append((Fraction(value[0], 5), Fraction(value[1], 5)))
        result.append(parsed)
    return result


def _signed_source_control_projection(source_law: Mapping[str, Any]) -> dict[str, Any]:
    try:
        alphabet = source_law["constructive_source_law"]["a1_complete_event_alphabet"]
        rows = alphabet["event_rows"]
        raw_rows = source_law["constructive_source_law"]["raw_step_rows"]
        capture = source_law["constructive_source_law"]["source_capture"]
    except (KeyError, TypeError) as error:
        raise PortGramCompletionError("signed source control fields are absent") from error
    if (
        alphabet.get("complete_signed_port_orbit") is not True
        or alphabet.get("event_count") != PORT_COUNT
        or alphabet.get("every_event_accepted_before_capture_hash") is not True
        or not isinstance(rows, list)
        or len(rows) != PORT_COUNT
        or not isinstance(raw_rows, list)
        or len(raw_rows) != PORT_COUNT
        or capture.get("capture_hash_binds_every_event_payload") is not True
        or capture.get("event_count") != PORT_COUNT
        or capture.get("source_capture_root_sha256") != _sha(capture.get("payload"))
    ):
        raise PortGramCompletionError("signed source control census drifted")
    positive_axis = {port: axis for axis, port in enumerate(POSITIVE_PORTS)}
    projection_rows: list[dict[str, Any]] = []
    for port in range(PORT_COUNT):
        row = rows[port]
        raw_row = raw_rows[port]
        positive_port = port if port in positive_axis else ANTIPODES[port]
        sign = 1 if port in positive_axis else -1
        direction = [0] * len(POSITIVE_PORTS)
        direction[positive_axis[positive_port]] = sign
        if (
            row.get("port") != port
            or row.get("antipodal_port") != ANTIPODES[port]
            or row.get("raw_direction_in_Z_power_6") != direction
            or row.get("event_kind") != "accepted_signed_axis_unit_record"
            or row.get("comparison_or_target_input_used") is not False
            or raw_row.get("port") != port
            or raw_row.get("inverse_port") != ANTIPODES[port]
            or raw_row.get("direction") != direction
            or raw_row.get("bijective_on_Z_power_6") is not True
            or raw_row.get("event_id") != row.get("event_id")
        ):
            raise PortGramCompletionError(f"signed source event {port} drifted")
        projection_rows.append(
            {
                "port": port,
                "antipodal_port": ANTIPODES[port],
                "raw_direction_in_Z_power_6": direction,
                "event_id": row["event_id"],
            }
        )
    if capture.get("payload", {}).get("event_rows") != rows:
        raise PortGramCompletionError("source capture does not bind the event rows")
    return {
        "signed_source_control_id": capture["payload"].get("source_law_id"),
        "source_capture_root_sha256": capture.get("source_capture_root_sha256"),
        "event_rows": projection_rows,
    }


def _repair_qtext(value: Q5) -> str:
    def render(item: Fraction) -> str:
        return str(item.numerator) if item.denominator == 1 else f"{item.numerator}/{item.denominator}"

    if value[1] == 0:
        return render(value[0])
    return f"{render(value[0])} + {render(value[1])}*sqrt(5)"


def _graph_distances(adjacency: Sequence[Sequence[Q5]], start: int) -> list[int]:
    distances = [-1] * len(adjacency)
    distances[start] = 0
    queue = [start]
    while queue:
        vertex = queue.pop(0)
        for other, edge in enumerate(adjacency[vertex]):
            if edge == ONE and distances[other] < 0:
                distances[other] = distances[vertex] + 1
                queue.append(other)
    return distances


def _repair_adjacency_packet(
    port_repair_bridge: Mapping[str, Any],
) -> tuple[list[list[Q5]], dict[str, Any]]:
    try:
        upstream_pin = port_repair_bridge["source_packet"]["carrier_manifest_pin"]
        serialized_frame = port_repair_bridge["exact_orbit_ray_table"]["rows"][
            "vertex12"
        ]["directions"]
    except (KeyError, TypeError) as error:
        raise PortGramCompletionError("repair carrier incidence ancestry is absent") from error
    if (
        upstream_pin.get("repository_relative_path")
        != CARRIER_MANIFEST.relative_to(ROOT).as_posix()
        or upstream_pin.get("bytes") != len(CARRIER_MANIFEST.read_bytes())
        or upstream_pin.get("sha256") != _raw_sha(CARRIER_MANIFEST)
        or serialized_frame
        != [[_repair_qtext(value) for value in row] for row in REPAIR_FRAME_COORDINATES]
    ):
        raise PortGramCompletionError("repair carrier/frame ancestry drifted")
    manifest = _load_json_strict(CARRIER_MANIFEST)
    try:
        carrier = manifest["carrier"]
        ports = carrier["ports"]
        edges = carrier["edges"]
    except (KeyError, TypeError) as error:
        raise PortGramCompletionError("repair carrier incidence fields are absent") from error
    expected_ports = [f"p{port:02d}" for port in range(PORT_COUNT)]
    if (
        manifest.get("schema") != "oph.echosahedral_selector_manifest.v1"
        or ports != expected_ports
        or not isinstance(edges, list)
        or len(edges) != 30
    ):
        raise PortGramCompletionError("repair carrier manifest contract drifted")
    port_index = {label: index for index, label in enumerate(ports)}
    fixture_adjacency = [[ZERO for _ in range(PORT_COUNT)] for _ in range(PORT_COUNT)]
    fixture_edges: set[tuple[int, int]] = set()
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or edge[0] not in port_index
            or edge[1] not in port_index
        ):
            raise PortGramCompletionError("repair carrier edge is malformed")
        left, right = sorted((port_index[edge[0]], port_index[edge[1]]))
        if left == right or (left, right) in fixture_edges:
            raise PortGramCompletionError("repair carrier edge is not simple")
        fixture_edges.add((left, right))
        fixture_adjacency[left][right] = fixture_adjacency[right][left] = ONE
    if any(_qsum(row) != (Fraction(5), Fraction(0)) for row in fixture_adjacency):
        raise PortGramCompletionError("repair carrier adjacency is not five-regular")
    fixture_to_source = []
    for vector in REPAIR_FRAME_COORDINATES:
        matches = [index for index, source in enumerate(RAW_COORDINATES) if source == vector]
        if len(matches) != 1:
            raise PortGramCompletionError("repair/source exact frame matching is not unique")
        fixture_to_source.append(matches[0])
    if sorted(fixture_to_source) != list(range(PORT_COUNT)):
        raise PortGramCompletionError("repair/source frame map is not bijective")
    adjacency = [[ZERO for _ in range(PORT_COUNT)] for _ in range(PORT_COUNT)]
    source_edges = []
    for left, right in sorted(fixture_edges):
        source_left, source_right = sorted(
            (fixture_to_source[left], fixture_to_source[right])
        )
        adjacency[source_left][source_right] = adjacency[source_right][source_left] = ONE
        source_edges.append([source_left, source_right])
    for port in range(PORT_COUNT):
        distance_three = [
            other
            for other, distance in enumerate(_graph_distances(adjacency, port))
            if distance == 3
        ]
        if distance_three != [ANTIPODES[port]]:
            raise PortGramCompletionError("repair adjacency antipodes drifted")
    return adjacency, {
        "origin": "pinned repair carrier incidence, relabeled by exact frame equality",
        "carrier_manifest_path": CARRIER_MANIFEST.relative_to(ROOT).as_posix(),
        "carrier_manifest_raw_sha256": _raw_sha(CARRIER_MANIFEST),
        "upstream_carrier_manifest_raw_sha256": upstream_pin["sha256"],
        "fixture_to_source_port_map": fixture_to_source,
        "source_edge_list": source_edges,
        "source_adjacency_sha256": _sha(
            [[1 if value == ONE else 0 for value in row] for row in adjacency]
        ),
    }


def _gram_class_adjacency(gram: Sequence[Sequence[Q5]]) -> list[list[Q5]]:
    adjacency: list[list[Q5]] = []
    classes = {"diagonal": 0, "adjacent": 0, "distance_two": 0, "antipodal": 0}
    for i, row in enumerate(gram):
        output: list[Q5] = []
        for j, value in enumerate(row):
            if i == j:
                expected, label = ONE, "diagonal"
                edge = False
            elif j == ANTIPODES[i]:
                expected, label = (-ONE[0], ZERO[1]), "antipodal"
                edge = False
            elif value == (Fraction(0), Fraction(1, 5)):
                expected, label = value, "adjacent"
                edge = True
            else:
                expected, label = (Fraction(0), Fraction(-1, 5)), "distance_two"
                edge = False
            if value != expected:
                raise PortGramCompletionError("exact Gram class drifted")
            classes[label] += 1
            output.append(ONE if edge else ZERO)
        adjacency.append(output)
    if classes != {"diagonal": 12, "adjacent": 60, "distance_two": 60, "antipodal": 12}:
        raise PortGramCompletionError("ordered Gram class census drifted")
    return adjacency


def _spectral_gram_packet(
    gram: Sequence[Sequence[Q5]],
    adjacency: Sequence[Sequence[Q5]],
    incidence_packet: Mapping[str, Any],
) -> dict[str, Any]:
    gram_class_adjacency = _gram_class_adjacency(gram)
    if not _matrix_equal(gram_class_adjacency, adjacency):
        raise PortGramCompletionError(
            "Gram adjacency classes do not match independent repair incidence"
        )
    identity = [[ONE if i == j else ZERO for j in range(PORT_COUNT)] for i in range(PORT_COUNT)]
    laplacian = [
        [_qsub(_qscale(identity[i][j], Fraction(5)), adjacency[i][j]) for j in range(PORT_COUNT)]
        for i in range(PORT_COUNT)
    ]
    adjacency_minus_five = [
        [_qsub(adjacency[i][j], _qscale(identity[i][j], Fraction(5)))
         for j in range(PORT_COUNT)]
        for i in range(PORT_COUNT)
    ]
    adjacency_plus_one = [
        [_qadd(adjacency[i][j], identity[i][j]) for j in range(PORT_COUNT)]
        for i in range(PORT_COUNT)
    ]
    adjacency_plus_sqrt_five = [
        [_qadd(adjacency[i][j], _qmul(SQRT5, identity[i][j]))
         for j in range(PORT_COUNT)]
        for i in range(PORT_COUNT)
    ]
    spectral_denominator = _qmul(
        _qmul(_qsub(SQRT5, (Fraction(5), Fraction(0))),
              _qadd(SQRT5, ONE)),
        _qscale(SQRT5, Fraction(2)),
    )
    projector_from_adjacency = _matscale(
        _matmul(
            _matmul(adjacency_minus_five, adjacency_plus_one),
            adjacency_plus_sqrt_five,
        ),
        _qinv(spectral_denominator),
    )
    gram_from_adjacency = _matscale(
        projector_from_adjacency, (Fraction(4), Fraction(0))
    )
    if not _matrix_equal(gram_from_adjacency, gram):
        raise PortGramCompletionError(
            "independent repair adjacency polynomial does not reconstruct Gram"
        )
    if not _matrix_equal(_matmul(gram, gram), _matscale(gram, (Fraction(4), Fraction(0)))):
        raise PortGramCompletionError("G squared is not four G")
    if _qsum([gram[i][i] for i in range(PORT_COUNT)]) != (Fraction(12), Fraction(0)):
        raise PortGramCompletionError("Gram trace is not twelve")
    low_cost = (Fraction(5), Fraction(-1))
    middle_cost = (Fraction(6), Fraction(0))
    high_cost = (Fraction(5), Fraction(1))
    if not _matrix_equal(_matmul(laplacian, gram), _matscale(gram, low_cost)):
        raise PortGramCompletionError("Gram is not the lowest positive Laplacian band")
    conjugate = [[_qconj(value) for value in row] for row in gram]
    if not _matrix_equal(_matmul(laplacian, conjugate), _matscale(conjugate, high_cost)):
        raise PortGramCompletionError("Galois Gram is not the highest rank-three band")
    if conjugate == list(map(list, gram)) or not (_qlt(low_cost, middle_cost) and _qlt(middle_cost, high_cost)):
        raise PortGramCompletionError("strict Galois-band separation failed")
    projector = projector_from_adjacency
    if not _matrix_equal(_matmul(projector, projector), projector):
        raise PortGramCompletionError("selected Gram projector is not idempotent")
    projector_trace = _qsum([projector[i][i] for i in range(PORT_COUNT)])
    if projector_trace != (Fraction(3), Fraction(0)):
        raise PortGramCompletionError("selected Gram projector does not have rank three")
    constant_projector = [
        [(Fraction(1, PORT_COUNT), Fraction(0)) for _ in range(PORT_COUNT)]
        for _ in range(PORT_COUNT)
    ]
    conjugate_projector = _matscale(conjugate, (Fraction(1, 4), Fraction(0)))
    middle_projector = _matsub(
        _matsub(_matsub(identity, constant_projector), projector),
        conjugate_projector,
    )
    projectors = {
        "constant": constant_projector,
        "slowest_nonconstant": projector,
        "middle": middle_projector,
        "galois_rank_three": conjugate_projector,
    }
    for name, candidate in projectors.items():
        if not _matrix_equal(_matmul(candidate, candidate), candidate):
            raise PortGramCompletionError(f"{name} spectral projector is not idempotent")
    for left_name, left_projector in projectors.items():
        for right_name, right_projector in projectors.items():
            if left_name == right_name:
                continue
            if any(
                value != ZERO
                for row in _matmul(left_projector, right_projector)
                for value in row
            ):
                raise PortGramCompletionError(
                    f"{left_name}/{right_name} spectral projectors are not orthogonal"
                )
    if not _matrix_equal(
        _matadd(
            _matadd(constant_projector, projector),
            _matadd(middle_projector, conjugate_projector),
        ),
        identity,
    ):
        raise PortGramCompletionError("spectral projectors do not resolve identity")
    projector_traces = {
        name: _qsum([candidate[i][i] for i in range(PORT_COUNT)])
        for name, candidate in projectors.items()
    }
    if projector_traces != {
        "constant": (Fraction(1), Fraction(0)),
        "slowest_nonconstant": (Fraction(3), Fraction(0)),
        "middle": (Fraction(5), Fraction(0)),
        "galois_rank_three": (Fraction(3), Fraction(0)),
    }:
        raise PortGramCompletionError("spectral projector ranks drifted")
    if not _matrix_equal(_matmul(laplacian, constant_projector), _matscale(constant_projector, ZERO)):
        raise PortGramCompletionError("constant projector is not the Laplacian kernel")
    if not _matrix_equal(_matmul(laplacian, middle_projector), _matscale(middle_projector, middle_cost)):
        raise PortGramCompletionError("middle projector is not the cost-six band")
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
    if not _matrix_equal(odd_projector, _matadd(projector, conjugate_projector)):
        raise PortGramCompletionError("odd antipodal subspace does not equal the two rank-three bands")
    if not _matrix_equal(even_projector, _matadd(constant_projector, middle_projector)):
        raise PortGramCompletionError("even antipodal subspace does not equal the rank-one/rank-five bands")
    low_tick = (Fraction(11, 12), Fraction(1, 60))
    middle_tick = (Fraction(9, 10), Fraction(0))
    high_tick = (Fraction(11, 12), Fraction(-1, 60))
    tick = _matsub(identity, _matscale(laplacian, (Fraction(1, 60), Fraction(0))))
    tick_spectral = _matadd(
        _matadd(constant_projector, _matscale(projector, low_tick)),
        _matadd(
            _matscale(middle_projector, middle_tick),
            _matscale(conjugate_projector, high_tick),
        ),
    )
    if not _matrix_equal(tick, tick_spectral):
        raise PortGramCompletionError("one-step repair operator spectral resolution drifted")
    if not (
        _qlt(ZERO, high_tick)
        and _qlt(high_tick, middle_tick)
        and _qlt(middle_tick, low_tick)
        and _qlt(low_tick, ONE)
    ):
        raise PortGramCompletionError("one-step repair eigenvalue order drifted")
    if any(projector[i][i] != (Fraction(1, 4), Fraction(0)) for i in range(PORT_COUNT)):
        raise PortGramCompletionError("low projector diagonal is not uniform")
    if any(conjugate_projector[i][i] != (Fraction(1, 4), Fraction(0)) for i in range(PORT_COUNT)):
        raise PortGramCompletionError("high projector diagonal is not uniform")
    if any(middle_projector[i][i] != (Fraction(5, 12), Fraction(0)) for i in range(PORT_COUNT)):
        raise PortGramCompletionError("rank-five projector diagonal is not uniform")
    centered_projector = _matsub(identity, constant_projector)
    if not _matrix_equal(
        centered_projector,
        _matadd(_matadd(projector, middle_projector), conjugate_projector),
    ):
        raise PortGramCompletionError("centered projector spectral resolution drifted")
    if not _matrix_equal(_transpose(tick), tick):
        raise PortGramCompletionError("one-step expectation operator is not symmetric")
    if not _matrix_equal(
        _matmul(tick, centered_projector), _matmul(centered_projector, tick)
    ):
        raise PortGramCompletionError("centering does not commute with one-step repair")
    intrinsic_generators = [
        [_qscale(projector[row][port], Fraction(2)) for row in range(PORT_COUNT)]
        for port in range(PORT_COUNT)
    ]
    intrinsic_generator_gram = [
        [_dot(intrinsic_generators[left], intrinsic_generators[right])
         for right in range(PORT_COUNT)]
        for left in range(PORT_COUNT)
    ]
    if not _matrix_equal(intrinsic_generator_gram, gram):
        raise PortGramCompletionError("intrinsic projector generators do not have Gram G")
    return {
        "independent_repair_incidence": copy.deepcopy(dict(incidence_packet)),
        "Gram_class_adjacency_matches_independent_repair_incidence": True,
        "projector_constructed_from_independent_adjacency_polynomial": True,
        "projector_polynomial": (
            "P_low=(A-5I)(A+I)(A+sqrt(5)I)/"
            "((sqrt(5)-5)(sqrt(5)+1)(2sqrt(5)))"
        ),
        "selected_gram_normalization": "G=4*P_slowest_nonconstant",
        "repair_generator": "K=L_ico/60",
        "unscaled_laplacian_band_costs": [
            _qtext(low_cost),
            _qtext(middle_cost),
            _qtext(high_cost),
        ],
        "selected_band": "adjacency_+sqrt5__laplacian_5-sqrt5__rank_3",
        "selected_projector_trace": "3",
        "gram_diagonal": "1",
        "gram_squared_identity": "G^2=4G",
        "laplacian_eigen_identity": "L_ico*G=(5-sqrt5)*G",
        "galois_partner_eigen_identity": "L_ico*conj(G)=(5+sqrt5)*conj(G)",
        "galois_partner_distinct": True,
        "strict_cost_order": "5-sqrt5 < 6 < 5+sqrt5",
        "full_spectral_resolution": {
            "costs": ["0", _qtext(low_cost), _qtext(middle_cost), _qtext(high_cost)],
            "ranks": [1, 3, 5, 3],
            "projectors_pairwise_orthogonal": True,
            "projectors_resolve_identity": True,
        },
        "intrinsic_local_carrier": {
            "ambient": "twelve-port source-counting space",
            "definition": "H=range(P_low)",
            "real_dimension": 3,
            "labeled_generator": "v_p=2*P_low*e_p",
            "generator_gram_identity": "<v_p,v_q>=4*(P_low)_pq=G_pq",
            "generator_gram_identity_exact": True,
            "cartesian_coordinates_used_to_define_carrier": False,
            "cartesian_chart_role": "isometric chart and arithmetic density witness only",
            "preferred_cartesian_frame_selected": False,
            "global_or_physical_space_promoted": False,
        },
        "source_backed_discrete_repair": {
            "one_step_expectation_operator": "T=I-L_ico/60",
            "one_step_operator_source_backed_by_pinned_ancestry": True,
            "exact_power_formula": (
                "T^n=P_0+((55+sqrt5)/60)^n*P_low+(9/10)^n*P_5+"
                "((55-sqrt5)/60)^n*P_high"
            ),
            "one_step_eigenvalues_descending": [
                "1",
                _qtext(low_tick),
                _qtext(middle_tick),
                _qtext(high_tick),
            ],
            "strict_subunit_order": (
                "0 < (55-sqrt5)/60 < 9/10 < (55+sqrt5)/60 < 1"
            ),
            "continuous_exponential_semigroup_used": False,
            "formal_operator_powers_equal_physical_n_tick_history": False,
            "IID_or_temporal_independence_proved": False,
            "full_temporal_grammar_completeness_proved": False,
            "physical_repair_law_promoted": False,
        },
        "canonical_centered_response_kernel_derivation": {
            "probe_family": "q_p=Q*e_p for all twelve ports with Q=I-P_0",
            "probe_count": 12,
            "probe_weights": "equal source-counting weight",
            "stochastic_initial_ensemble_required": False,
            "response_vectors": "y_p^(n)=T^n*q_p",
            "kernel_definition": "C_n=(T^n*Q)^T*(T^n*Q)=Q*T^(2n)*Q",
            "exact_spectral_formula": (
                "C_n=((55+sqrt5)/60)^(2n)*P_low+(9/10)^(2n)*P_5+"
                "((55-sqrt5)/60)^(2n)*P_high"
            ),
            "exact_for_every_nonnegative_integer_n": True,
            "unique_largest_nonconstant_factor": "(55+sqrt5)/60",
            "common_diagonal_formula": (
                "diag(C_n)=((55+sqrt5)/60)^(2n)/4+"
                "5*(9/10)^(2n)/12+((55-sqrt5)/60)^(2n)/4"
            ),
            "trace_formula": (
                "trace(C_n)=3*((55+sqrt5)/60)^(2n)+5*(9/10)^(2n)+"
                "3*((55-sqrt5)/60)^(2n)"
            ),
            "projective_limit": "[C_n] -> [P_low]",
            "trace_one_limit": "C_n/trace(C_n) -> P_low/3",
            "trace_twelve_limit": "12*C_n/trace(C_n) -> 4*P_low=G",
            "unit_diagonal_limit": "C_n/common_diagonal(C_n) -> 4*P_low=G",
            "limit_before_quotient_and_completion_required": True,
            "finite_n_centered_rank": 11,
            "finite_n_antipodally_odd_rank": 6,
            "finite_n_signed_module_is_discrete_and_complete": True,
            "strictly_positive_unequal_probe_weights_preserve_limit_rank_three": True,
            "unequal_weight_limit_form": "P_low*W*P_low on range(P_low)",
            "unequal_weights_preserve_exact_icosahedral_Gram_angles": False,
            "named_operational_readback_premise": (
                "completed future-repair distinguishability is read through the centered "
                "equal-port response kernel"
            ),
            "named_Gram_topology_premise": (
                "the scale-normalized asymptotic response kernel defines the port metric"
            ),
            "current_A2_contains_completed_asymptotic_kernel_readback": False,
            "formal_response_powers_are_physical_time_evolution": False,
            "target_or_comparison_data_used": False,
            "port_gram_derived_rather_than_supplied_by_A1_RG": True,
        },
        "slowest_band_selection_is_extra_economy_selector": False,
        "dynamical_selection_scope": (
            "the finite implication uses the pinned one-step expectation operator and "
            "its formal powers; an IID history law, temporal completeness, asymptotic "
            "operational readback, and physical repair law are not promoted"
        ),
        "positive_clock_rescaling_changes_selected_eigenspace": False,
        "gram_branch_selected_by_declared_repair_cost_if_A1R_A2R_adopted": True,
        "current_A1_selects_between_galois_frames": False,
        "current_A1R_A2R_adopted": False,
        "full_gram_qsqrt5": [[_qtext(value) for value in row] for row in gram],
        "galois_control_gram_qsqrt5": [
            [_qtext(value) for value in row] for row in conjugate
        ],
    }


def _module_completion_packet(
    gram: Sequence[Sequence[Q5]], source_projection: Mapping[str, Any]
) -> dict[str, Any]:
    raw_norm = _dot(RAW_COORDINATES[0], RAW_COORDINATES[0])
    if raw_norm != (Fraction(5, 2), Fraction(1, 2)):
        raise PortGramCompletionError("raw port norm drifted")
    reconstructed: list[list[Q5]] = []
    for left in RAW_COORDINATES:
        row = []
        for right in RAW_COORDINATES:
            row.append(_qmul(_dot(left, right), _qinv(raw_norm)))
        reconstructed.append(row)
    if not _matrix_equal(reconstructed, gram):
        raise PortGramCompletionError("coordinate factorization does not reproduce Gram")
    for port in range(PORT_COUNT):
        antipode = ANTIPODES[port]
        for other in range(PORT_COUNT):
            if gram[antipode][other] != _qscale(gram[port][other], Fraction(-1)):
                raise PortGramCompletionError("Gram row does not descend through antipodes")
            if gram[other][antipode] != _qscale(gram[other][port], Fraction(-1)):
                raise PortGramCompletionError("Gram column does not descend through antipodes")

    positive_coordinates = [RAW_COORDINATES[port] for port in POSITIVE_PORTS]
    rank_witness_ports = (0, 1, 4)
    rank_matrix = [
        [RAW_COORDINATES[port][coordinate] for port in rank_witness_ports]
        for coordinate in range(3)
    ]
    rank_determinant = _qdet(rank_matrix)
    if rank_determinant == ZERO:
        raise PortGramCompletionError("three-dimensional rank witness vanished")

    split_rows: list[list[Fraction]] = []
    for coordinate in range(3):
        rational_row: list[Fraction] = []
        phi_row: list[Fraction] = []
        for row in positive_coordinates:
            a, b = row[coordinate]
            rational_row.append(a - b)
            phi_row.append(2 * b)
        split_rows.extend((rational_row, phi_row))
    split_determinant = _fraction_determinant(split_rows)
    if split_determinant != -8:
        raise PortGramCompletionError("Z[phi] coefficient determinant drifted")

    gram6 = [[gram[left][right] for right in POSITIVE_PORTS] for left in POSITIVE_PORTS]
    return {
        "signed_cumulative_port_record_module": (
            "M_Z=Z[ports]/<e_antipode(p)+e_p> ~= Z^6"
        ),
        "constructive_source_control_projection_sha256": _sha(source_projection),
        "constructive_source_control_event_count": PORT_COUNT,
        "constructive_source_control_is_canonical_source_law": False,
        "positive_port_basis": list(POSITIVE_PORTS),
        "antipodal_relations": [
            [port, ANTIPODES[port]] for port in POSITIVE_PORTS
        ],
        "full_Gram_antipodal_descent_identity": (
            "G[antipode(p),q]=-G[p,q]=G[p,antipode(q)]"
        ),
        "full_Gram_descends_to_signed_record_quotient": True,
        "gram6_is_the_descended_positive_port_basis_form": True,
        "real_scalar_extension": "M_R=R tensor_Z M_Z ~= R^6",
        "gram6_qsqrt5": [[_qtext(value) for value in row] for row in gram6],
        "gram_factorization": "G6=U^T U/(5/2+sqrt5/2)",
        "raw_generator_coordinates_qsqrt5": [
            [_qtext(value) for value in row] for row in positive_coordinates
        ],
        "positive_semidefinite": True,
        "real_rank": 3,
        "real_kernel_dimension": 3,
        "rank_witness_positive_ports": list(rank_witness_ports),
        "raw_rank_witness_determinant": _qtext(rank_determinant),
        "integer_kernel_is_zero": True,
        "integer_kernel_witness": {
            "coefficient_basis": "split every coordinate in the Q-basis {1,phi}",
            "six_by_six_rational_matrix": [
                [str(value) for value in row] for row in split_rows
            ],
            "determinant": str(split_determinant),
        },
        "image_module": "finite-index-8 submodule of Z[phi]^3",
        "image_contains": "8*Z[phi]^3",
        "image_dense_in_real_three_space": True,
        "density_argument": (
            "sqrt(5) is irrational, so Z[phi] is dense in R; the determinant-eight "
            "image contains 8*Z[phi]^3 and is dense in R^3"
        ),
        "hausdorff_metric": "d_G(m,n)^2=(m-n)^T G6 (m-n)",
        "hausdorff_on_integer_records": True,
        "single_event_generators_have_unit_gram_norm": True,
        "nonzero_integer_records_have_a_shortest_positive_gram_length": False,
        "atomic_generator_is_not_a_metric_minimum": True,
        "real_quotient": "H_0=(R tensor M_Z)/ker(G6)",
        "real_quotient_dimension": 3,
        "completion_theorem": (
            "the metric completion of (M_Z,d_G) is uniquely isometric to H_0"
        ),
        "continuous_field_assumed": False,
        "continuous_carrier_constructed_as_metric_completion": True,
        "continuous_carrier_is_primitive_input": False,
        "scalar_field_space_selected": False,
        "physical_continuous_field_selected": False,
        "raw_addition_isometric": True,
        "translation_extension": (
            "every addition map n->n+m extends uniquely by continuity to H_0"
        ),
        "completion_translation_action_is_same_raw_action": True,
        "group_and_action_extension_scope": (
            "standard completion theorem for a translation-invariant metric group; "
            "the finite Gram, density, and raw isometry premises are machine checked"
        ),
        "group_and_action_extension_formalized_in_Lean": False,
        "ordered_history_to_position_quotient_proved": False,
        "record_order_and_cost_retained_separately": True,
        "carrier_position_readback_only": True,
        "limit_before_quotient_and_completion_required": True,
        "finite_n_centered_rank": 11,
        "finite_n_antipodally_odd_rank": 6,
        "finite_n_signed_module_is_discrete_and_complete": True,
        "preferred_cartesian_frame_selected": False,
        "local_carrier_only": True,
        "faithful_A5_completion_action_formalized": False,
        "overlap_refinement_gluing_proved": False,
        "overall_positive_metric_scale_selected": False,
    }


def _support_and_clause_packet(
    fz11: Mapping[str, Any], port_dual: Mapping[str, Any], source_law: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        fz_geometry = fz11["exact_port_frame_and_relabel"]["source_geometry_hash"]
        support_scope = port_dual["source_scope"]
        support_geometry = support_scope["source_geometry_hash"]
        port_map = support_scope["port_to_defect_vertex_bijection"]
        source_attainment = source_law["attainment"]
    except (KeyError, TypeError) as error:
        raise PortGramCompletionError("support/source premise audit fields are absent") from error
    if (
        fz_geometry != support_geometry
        or port_map != list(range(PORT_COUNT))
        or source_attainment.get("canonical_source_selection") is not False
        or source_attainment.get("spatial_translation") is not False
        or source_attainment.get("universal_z_power_6_factorization") is not True
    ):
        raise PortGramCompletionError("support/source premise boundary drifted")

    support = {
        "common_finite_geometry_hash": fz_geometry,
        "port_to_support_vertex_map": list(range(PORT_COUNT)),
        "support_frame_gram_equals_selected_repair_gram": True,
        "labeled_spanning_equal_gram_isometry_theorem": (
            "two spanning labeled families with identical Gram matrices admit a "
            "unique linear isometry sending every first-family vector to its labeled mate"
        ),
        "conditional_common_completion_object": "H_0",
        "conditional_support_embedding": "X(p)=class(e_p) in H_0",
        "conditional_primitive_hop": "h_p=class(e_p) in H_0",
        "normalized_labeled_frame_isometry": True,
        "dimensionful_support_radius_over_hop_selected": False,
        "source_semantic_identity_required": True,
        "conditional_hop_symbol_scope": (
            "a is the common norm of one atomic signed-port event, not a shortest "
            "nonzero translation length"
        ),
        "auxiliary_adapter_normalized_support_and_hop_directions_equal_by_definition": True,
        "dimensionful_support_and_hop_vectors_equal_by_definition": False,
        "auxiliary_coordinate_equality_is_source_semantic_identity": False,
        "support_and_hop_share_semantic_object_in_current_source": False,
        "conditional_identity_requires_A2_cauchy_readback_clause": True,
        "physical_pixel_identified": False,
        "physical_areal_radius_selected": False,
    }
    weakest = {
        "proposed_label": "A1-RG/A2-RC cumulative port-record completion clause",
        "A1_RG": [
            (
                "The cumulative signed primitive port-record position is the antipodal "
                "module M_Z. Ordered event history, record identity, and repair cost "
                "remain separate data and are not quotiented into this position object."
            ),
            (
                "On the same port labels, the adopted completed seam-repair source emits "
                "the one-step expectation operator T=I-L_ico/60 on the complete signed "
                "cumulative record module; it does not separately supply a Gram form."
            ),
            (
                "The support seed map, signed cumulative record labels, and complete equal-port "
                "response-probe census are emitted in one source projection and "
                "co-transform under presentation and refinement."
            ),
        ],
        "A2_RC": [
            (
                "Operational meanings of cumulative port positions are Hausdorff and "
                "Cauchy-complete for the response-derived Gram seminorm; zero-distance positions "
                "are identified."
            ),
            (
                "The primitive translation readback and support seed readback factor "
                "through that same universal completion, so their labeled generator maps "
                "are the unique equal-Gram isometry rather than two independently scaled copies."
            ),
            (
                "Completed future-repair distinguishability is the centered equal-port "
                "response kernel, and its scale-normalized asymptotic ray defines the "
                "Gram topology on that same cumulative port-record position. The normalized "
                "infinite-response limit is taken before quotient and completion."
            ),
        ],
        "why_no_fourth_axiom_is_needed_if_adopted": (
            "A1 types the cumulative record position, event history, one-step response, "
            "support labels, and complete "
            "probe census; A2 fixes the response-derived topology, common operational "
            "completion, and same-readback factorization; A3 supplies the source-counting "
            "weights without a new geometric selector."
        ),
        "overall_clock_or_length_unit_left_free": True,
    }
    controls = {
        "galois_branch_control": {
            "weaker_clause": "A5 covariance, antipodes, rank three, and unit diagonal only",
            "surviving_models": ["G", "conj(G)"],
            "separated_only_by_repair_cost_projector": True,
        },
        "independent_rescaling_control": {
            "weaker_clause": (
                "same port labels, same A5 action, same normalized 1/12 measure, "
                "and isomorphic three-dimensional spaces"
            ),
            "family": "X_R(p)=R*u_p and h_a(p)=a*u_p for arbitrary R,a>0",
            "all_current_finite_invariants_preserved": True,
            "R_A_over_a_remains_free": True,
        },
        "finite_quotient_control": {
            "meaning": "(Z/3Z)^6",
            "fails_metric_factorization_reason": (
                "3*e_alpha is zero in the finite endpoint quotient and has nonzero Gram length"
            ),
            "finite_endpoint_quotient_is_physical_translation_completion": False,
        },
        "dense_hop_control": {
            "single_atomic_event_has_unit_normalized_gram_norm": True,
            "arbitrarily_small_nonzero_composite_translations_exist": True,
            "atomic_event_length_is_a_minimum_lattice_spacing": False,
        },
        "completion_without_source_control": {
            "mathematical_completion_exists": True,
            "source_native_physical_action_follows_without_A1_RG_A2_RC": False,
        },
        "finite_n_completion_control": {
            "finite_n_centered_response_rank": 11,
            "finite_n_antipodally_odd_response_rank": 6,
            "three_dimensional_completion_before_normalized_limit": False,
            "normalized_infinite_response_limit_is_load_bearing": True,
        },
        "response_kernel_controls": {
            "without_scale_normalization_raw_kernel_limit": "0 on the centered subspace",
            "asymptotic_readback_normalization_is_load_bearing": True,
            "unequal_probe_weight_control": (
                "strictly positive unequal centered port-probe weights leave a rank-three "
                "P_low*W*P_low form rather than a scalar multiple of P_low"
            ),
            "equal_probe_counting_is_load_bearing_for_exact_icosahedral_angles": True,
            "equal_probe_counting_is_load_bearing_for_dimension_three": False,
        },
    }
    return support, weakest, controls


def produce_receipt() -> dict[str, Any]:
    fz11 = _validated_self_digest(
        FZ11_RECEIPT, schema=FZ11_SCHEMA, status=FZ11_STATUS, issue=655
    )
    port_dual = _validated_self_digest(
        PORT_DUAL_RECEIPT,
        schema=PORT_DUAL_SCHEMA,
        status=PORT_DUAL_STATUS,
        issue=664,
    )
    source_law = _validated_self_digest(
        SOURCE_LAW_RECEIPT,
        schema=SOURCE_LAW_SCHEMA,
        status=SOURCE_LAW_STATUS,
        issue=655,
    )
    bounded_repair = _validated_bounded_repair_receipt(BOUNDED_REPAIR_RECEIPT)
    port_repair_bridge = _validated_port_repair_bridge(
        PORT_REPAIR_BRIDGE_RECEIPT, bounded_repair
    )
    if (
        fz11.get("comparison_data_read") is not False
        or port_dual.get("comparison_data_read") is not False
        or port_dual.get("target_data_read") is not False
        or source_law.get("comparison_data_read") is not False
    ):
        raise PortGramCompletionError("a parent crossed the target-data firewall")

    gram = _parse_parent_gram(fz11)
    repair_adjacency, incidence_packet = _repair_adjacency_packet(
        port_repair_bridge
    )
    spectral = _spectral_gram_packet(gram, repair_adjacency, incidence_packet)
    source_projection = _signed_source_control_projection(source_law)
    completion = _module_completion_packet(gram, source_projection)
    support, weakest, controls = _support_and_clause_packet(fz11, port_dual, source_law)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "issues": [655, 663, 664],
        "comparison_data_read": False,
        "target_data_read": False,
        "parent_pins": {
            "fz11_conditional_adapter": {
                "path": FZ11_RECEIPT.relative_to(ROOT).as_posix(),
                "raw_sha256": _raw_sha(FZ11_RECEIPT),
                "receipt_sha256": fz11["receipt_sha256"],
                "schema": FZ11_SCHEMA,
                "status": FZ11_STATUS,
            },
            "primitive_port_dual_measure": {
                "path": PORT_DUAL_RECEIPT.relative_to(ROOT).as_posix(),
                "raw_sha256": _raw_sha(PORT_DUAL_RECEIPT),
                "receipt_sha256": port_dual["receipt_sha256"],
                "schema": PORT_DUAL_SCHEMA,
                "status": PORT_DUAL_STATUS,
            },
            "constructive_signed_port_record_control": {
                "path": SOURCE_LAW_RECEIPT.relative_to(ROOT).as_posix(),
                "raw_sha256": _raw_sha(SOURCE_LAW_RECEIPT),
                "receipt_sha256": source_law["receipt_sha256"],
                "schema": SOURCE_LAW_SCHEMA,
                "status": SOURCE_LAW_STATUS,
                "canonical_source_selection": False,
            },
            "bounded_one_step_expectation_repair": {
                "path": BOUNDED_REPAIR_RECEIPT.relative_to(ROOT).as_posix(),
                "raw_sha256": _raw_sha(BOUNDED_REPAIR_RECEIPT),
                "certificate_payload_sha256": bounded_repair[
                    "certificate_payload_sha256"
                ],
                "schema": BOUNDED_REPAIR_SCHEMA,
                "status": BOUNDED_REPAIR_STATUS,
                "physical_repair_law_receipt": False,
                "canonical_A3_alone_implies_markovity": False,
            },
            "port_repair_propagation_boundary": {
                "path": PORT_REPAIR_BRIDGE_RECEIPT.relative_to(ROOT).as_posix(),
                "raw_sha256": _raw_sha(PORT_REPAIR_BRIDGE_RECEIPT),
                "receipt_sha256": port_repair_bridge["receipt_sha256"],
                "schema": PORT_REPAIR_BRIDGE_SCHEMA,
                "status": PORT_REPAIR_BRIDGE_STATUS,
                "one_step_operator": "T = I - L_icosahedron/60",
                "spatial_port_hop_source_receipt": False,
                "same_operator_physical_readout_receipt": False,
            },
        },
        "exact_repair_selected_gram": spectral,
        "exact_signed_module_completion": completion,
        "support_hop_isometry_implication": support,
        "weakest_clause_strengthening": weakest,
        "countermodel_controls": controls,
        "attainment": {
            "exact_lowest_repair_band_selects_port_gram": True,
            "source_conditional_mean_T_exact": True,
            "galois_branch_separated_by_exact_cost": True,
            "signed_module_gram_rank_three": True,
            "signed_integer_record_metric_hausdorff": True,
            "signed_record_image_dense": True,
            "completion_is_three_dimensional_euclidean_vector_group": True,
            "continuous_carrier_constructed_as_metric_completion": True,
            "raw_translation_extends_uniquely_to_completion": True,
            "support_hop_equal_gram_isometry_implication": True,
            "canonical_signed_port_record_source_selected": False,
            "A1R_A2R_repair_amendment_adopted": False,
            "A2_cauchy_operational_completion_clause_present": False,
            "operational_Gram_Cauchy_readback_and_topology_selected": False,
            "same_semantic_support_translation_object_emitted": False,
            "source_native_physical_translation_promoted": False,
            "physical_three_space_promoted": False,
            "faithful_A5_completion_action_formalized": False,
            "overlap_refinement_gluing_proved": False,
            "global_carrier_promoted": False,
            "scalar_field_space_selected": False,
            "physical_P_pixel_is_primitive_port_sector": False,
            "support_areal_radius_is_primitive_hop_promoted": False,
            "overall_physical_scale_selected": False,
            "physical_prediction_promoted": False,
            "comparison_permitted": False,
            "issue_662_armed": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "implementation_pins": [
            _raw_pin(PRODUCER_PATH),
            _raw_pin(VERIFIER_PATH),
            _raw_pin(TEST_PATH),
        ],
    }
    payload["receipt_sha256"] = _sha(payload)
    return payload


def verify_receipt(report: Mapping[str, Any]) -> dict[str, Any]:
    expected = produce_receipt()
    if _canonical_bytes(report) != _canonical_bytes(expected):
        raise PortGramCompletionError("receipt differs from exact producer replay")
    return {
        "receipt": True,
        "status": STATUS,
        "exact_completion_implication": True,
        "source_promotion": False,
        "physical_promotion": False,
        "comparison_permitted": False,
    }


def load_receipt_strict(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    report = _load_json_strict(path)
    verify_receipt(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    report = produce_receipt()
    if args.validate_only:
        existing = _load_json_strict(args.output)
        verify_receipt(existing)
        print("PORT_GRAM_COMPLETION_BRIDGE_VALID")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
    print(json.dumps({"output": str(args.output), "status": STATUS}, sort_keys=True))


if __name__ == "__main__":
    main()
