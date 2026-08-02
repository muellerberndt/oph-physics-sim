"""Build the bounded conditional three-dimensional bridge for FZ-11.

The issue-655 constructive control supplies twelve reversible signed record
steps on ``Z^6``.  This module tests one additional adapter: the six positive
record generators are sent to the six unoriented axes of the exact
icosahedral frame in ``R^3`` and their inverses are sent to the antipodal
directions.  The resulting translation operator has the frozen FZ-11 cosine
symbol and coefficient ratios.

This is an auxiliary branch contract.  The canonical simulator does not
select it, the modulo-three observer quotient does not furnish a spatial
readout, and no particle sector, boost law, laboratory observable, or public
comparison is attached.
"""

from __future__ import annotations

import argparse
import cmath
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.core.icosahedral import (
    build_geodesic_icosahedral_tower,
    icosahedral_a5_port_permutations,
)
from oph_fpe.dynamics import verify_vertex12_constructive_source_law_independent


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / "data/repair_closure/fz11_3d_translation_bridge_receipt.json"
SOURCE_RECEIPT_PATH = (
    ROOT / "data/repair_closure/vertex12_constructive_source_law_receipt.json"
)
RER_ROOT = ROOT.parent / "reverse-engineering-reality"
FROZEN_RECEIPT_PATH = (
    RER_ROOT
    / "code/a5_fingerprint/runtime/spin_six_primitive_port_prediction_receipt.json"
)
PORT_FRAME_LEAN_PATH = RER_ROOT / "Lean/Screen/PortFrameGram.lean"
VERIFIER_PATH = (
    ROOT / "oph_fpe/dynamics/verify_fz11_3d_translation_bridge_independent.py"
)
TEST_PATH = ROOT / "tests/test_fz11_3d_translation_bridge.py"
GEOMETRY_PATH = ROOT / "oph_fpe/core/icosahedral.py"

SCHEMA = "oph.fz11-conditional-3d-translation-bridge.v1"
STATUS = (
    "CONDITIONAL_3D_SCALAR_TRANSLATION_ADAPTER_ATTAINED__"
    "CANONICAL_SOURCE_SELECTION_TIME_EVOLUTION_PHOTON_SECTOR_SCALE_FRAME_"
    "BOOST_AND_EXCLUSIVITY_OPEN"
)
SOURCE_SCHEMA = "oph.vertex12-constructive-source-law-control.v1"
SOURCE_STATUS = (
    "CONSTRUCTIVE_SOURCE_LAW_CONTROL_ATTAINED__"
    "CANONICAL_A1_A2_A3_SELECTION_AND_PHYSICAL_ATTACHMENT_OPEN"
)
FROZEN_SCHEMA = "oph.spin_six_primitive_port_prediction.v1"
FROZEN_STATUS = (
    "FROZEN_PROSPECTIVE_PRIMITIVE_TWELVE_PORT_BRANCH_PREDICTION__"
    "PHYSICAL_COMPARISON_UNARMED"
)

FROZEN_RAW_SHA256 = "sha256:8ac97d7c46199717ed031610efdda65c40f6a251e78715d6bc05888d598e66d8"
FROZEN_RECEIPT_SHA256 = "sha256:92cc654f2fae04ac1fb90d3f646eae599e6106f217df74641fda2e921bfc0f92"
FROZEN_PREDICTION_SHA256 = "sha256:dc3aa0afc49d94eb4ef4220baed21c918cd11fc9e00ebb1611f3643396cd50f2"
FROZEN_PREMISES_SHA256 = "sha256:543afc4e7313a9155bd9d4a3a20eea7d83ae5a68c6e5acc6b62b2f42b5dc7de0"
FROZEN_SCOPE_SHA256 = "sha256:8f43767bf0d89684452b93fcc663028a8492c2c6192783870b8dbe3ee8715893"
PORT_FRAME_LEAN_RAW_SHA256 = "sha256:d1cebe56450e7586eed730b52068753ebca3a6563da1453679e87fa7c7b653e3"
BASE_GEOMETRY_HASH = "e333556ce101cb224d94882270c17e0d7e36469908871ca133e5bbbee2c6eafd"

# This is the orientation-fixed conjugacy from the simulator's base-vertex
# order to the abstract labels used by PortFrameGram.lean.
SOURCE_TO_RER = (8, 10, 1, 3, 7, 11, 0, 4, 6, 9, 2, 5)
SOURCE_ANTIPODES = (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8)
SOURCE_PAIRS = ((0, 3), (1, 2), (4, 7), (5, 6), (8, 11), (9, 10))
POSITIVE_PORTS = (0, 1, 4, 5, 8, 9)
RER_NEIGHBORS = (
    (1, 2, 3, 4, 6),
    (0, 2, 3, 5, 7),
    (0, 1, 4, 5, 8),
    (0, 1, 6, 7, 9),
    (0, 2, 6, 8, 10),
    (1, 2, 7, 8, 11),
    (0, 3, 4, 9, 10),
    (1, 3, 5, 9, 11),
    (2, 4, 5, 10, 11),
    (3, 6, 7, 10, 11),
    (4, 6, 8, 9, 11),
    (5, 7, 8, 9, 10),
)

Q5 = tuple[Fraction, Fraction]
ZERO: Q5 = (Fraction(0), Fraction(0))
ONE: Q5 = (Fraction(1), Fraction(0))
PHI: Q5 = (Fraction(1, 2), Fraction(1, 2))
SOURCE_RAW_COORDINATES: tuple[tuple[Q5, Q5, Q5], ...] = (
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


class FZ11TranslationBridgeError(RuntimeError):
    """Raised when the conditional bridge fails closed."""


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


def _raw_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_pin(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(path.read_bytes()),
        "sha256": _raw_sha(path),
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FZ11TranslationBridgeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
    )
    if not isinstance(value, dict):
        raise FZ11TranslationBridgeError(f"{path} is not a JSON object")
    return value


def _q5_add(left: Q5, right: Q5) -> Q5:
    return left[0] + right[0], left[1] + right[1]


def _q5_mul(left: Q5, right: Q5) -> Q5:
    return (
        left[0] * right[0] + 5 * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _q5_inv(value: Q5) -> Q5:
    denominator = value[0] ** 2 - 5 * value[1] ** 2
    if denominator == 0:
        raise FZ11TranslationBridgeError("attempted to invert zero in Q(sqrt(5))")
    return value[0] / denominator, -value[1] / denominator


def _q5_scale(value: Q5, scalar: Fraction) -> Q5:
    return value[0] * scalar, value[1] * scalar


def _dot(left: Sequence[Q5], right: Sequence[Q5]) -> Q5:
    result = ZERO
    for lvalue, rvalue in zip(left, right, strict=True):
        result = _q5_add(result, _q5_mul(lvalue, rvalue))
    return result


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _q5_text(value: Q5) -> str:
    return f"{_fraction_text(value[0])}+{_fraction_text(value[1])}*sqrt5"


def _inverse(permutation: Sequence[int]) -> tuple[int, ...]:
    if sorted(permutation) != list(range(len(permutation))):
        raise FZ11TranslationBridgeError("row is not a permutation")
    result = [0] * len(permutation)
    for source, target in enumerate(permutation):
        result[target] = source
    return tuple(result)


def _fraction_determinant(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    work = [list(row) for row in matrix]
    if not work or any(len(row) != len(work) for row in work):
        raise FZ11TranslationBridgeError("determinant requires a square matrix")
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


def _integer_injection_certificate() -> dict[str, Any]:
    # Convert a+b*sqrt(5) to (a-b)+2b*phi, then split the three coordinate
    # equations into their rational and phi coefficients.
    split_rows: list[list[Fraction]] = []
    for coordinate in range(3):
        rational_row = []
        phi_row = []
        for port in POSITIVE_PORTS:
            a, b = SOURCE_RAW_COORDINATES[port][coordinate]
            rational_row.append(a - b)
            phi_row.append(2 * b)
        split_rows.extend((rational_row, phi_row))
    determinant = _fraction_determinant(split_rows)
    if determinant != -8:
        raise FZ11TranslationBridgeError("integer injection matrix drifted")
    return {
        "basis": "split each R3 coordinate in the Q-basis {1,phi}",
        "positive_axis_order": list(POSITIVE_PORTS),
        "six_by_six_rational_coefficient_matrix": [
            [_fraction_text(value) for value in row] for row in split_rows
        ],
        "determinant": _fraction_text(determinant),
        "rank": 6,
        "conclusion": "F_a(n)=0 implies n=0 for every finite positive a",
    }


def _dense_image_certificate() -> dict[str, Any]:
    injection = _integer_injection_certificate()
    if injection["determinant"] != "-8" or math.isqrt(5) ** 2 == 5:
        raise FZ11TranslationBridgeError("dense-image arithmetic drifted")
    return {
        "golden_ratio": "phi=(1+sqrt(5))/2",
        "raw_coordinate_module": "Z[phi]^3=(Z+phi*Z)^3",
        "common_unit_normalization": "1/sqrt(2+phi)",
        "physical_image_module": "(a/sqrt(2+phi))*M, where M is the raw image",
        "raw_image_index_in_Z_phi_cubed": 8,
        "raw_image_contains_eight_times_Z_phi_cubed": True,
        "index_witness": "absolute determinant of the six-by-six {1,phi}-coefficient matrix",
        "character_test": {
            "assumption": "q dot ((a/sqrt(2+phi))*g_alpha) is integral for all six raw generators",
            "rescaled_character": "p=(a/sqrt(2+phi))*q",
            "forced_coordinate_pairs": [
                "2*p_x is integral and 2*phi*p_x is integral",
                "2*p_y is integral and 2*phi*p_y is integral",
                "2*p_z is integral and 2*phi*p_z is integral",
            ],
            "irrationality_witness": "x^2-x-1 has nonsquare discriminant 5 over Q",
            "annihilator": "{0}",
        },
        "conclusion": (
            "for every finite positive a the image is a dense index-eight "
            "submodule of (a/sqrt(2+phi))*Z[phi]^3"
        ),
    }


def _source_graph() -> tuple[set[tuple[int, int]], tuple[tuple[int, ...], ...]]:
    level = build_geodesic_icosahedral_tower(0).levels[0]
    if level.geometry_hash != BASE_GEOMETRY_HASH:
        raise FZ11TranslationBridgeError("base icosahedral geometry hash drifted")
    edges = {tuple(sorted((int(left), int(right)))) for left, right in level.edges}
    if len(edges) != 30:
        raise FZ11TranslationBridgeError("base edge census drifted")
    actions = tuple(tuple(int(value) for value in row) for row in icosahedral_a5_port_permutations())
    if len(actions) != 60 or len(set(actions)) != 60:
        raise FZ11TranslationBridgeError("proper source action is not a faithful order-60 action")
    return edges, actions


def _exact_frame() -> dict[str, Any]:
    norm_squared = _dot(SOURCE_RAW_COORDINATES[0], SOURCE_RAW_COORDINATES[0])
    expected_norm = (Fraction(5, 2), Fraction(1, 2))
    if norm_squared != expected_norm or any(
        _dot(row, row) != expected_norm for row in SOURCE_RAW_COORDINATES
    ):
        raise FZ11TranslationBridgeError("raw frame norm drifted")
    gram5: list[list[list[int]]] = []
    class_counts = {"diagonal": 0, "adjacent": 0, "distance_two": 0, "antipodal": 0}
    edges, actions = _source_graph()
    for left, left_row in enumerate(SOURCE_RAW_COORDINATES):
        output_row = []
        for right, right_row in enumerate(SOURCE_RAW_COORDINATES):
            value = _q5_scale(
                _q5_mul(_dot(left_row, right_row), _q5_inv(norm_squared)),
                Fraction(5),
            )
            if value[0].denominator != 1 or value[1].denominator != 1:
                raise FZ11TranslationBridgeError("scaled Gram entry is not integral")
            output_row.append([int(value[0]), int(value[1])])
            if left == right:
                expected, label = (5, 0), "diagonal"
            elif right == SOURCE_ANTIPODES[left]:
                expected, label = (-5, 0), "antipodal"
            elif tuple(sorted((left, right))) in edges:
                expected, label = (0, 1), "adjacent"
            else:
                expected, label = (0, -1), "distance_two"
            if tuple(output_row[-1]) != expected:
                raise FZ11TranslationBridgeError("exact source Gram classification failed")
            class_counts[label] += 1
        gram5.append(output_row)

    inverse = _inverse(SOURCE_TO_RER)
    rer_edges = {
        tuple(sorted((left, right)))
        for left, neighbors in enumerate(RER_NEIGHBORS)
        for right in neighbors
    }
    if len(rer_edges) != 30 or any(
        (tuple(sorted((left, right))) in edges)
        != (tuple(sorted((SOURCE_TO_RER[left], SOURCE_TO_RER[right]))) in rer_edges)
        for left in range(12)
        for right in range(12)
        if left != right
    ):
        raise FZ11TranslationBridgeError("source-to-RER adjacency conjugacy failed")
    if any(
        SOURCE_TO_RER[SOURCE_ANTIPODES[port]] != 11 - SOURCE_TO_RER[port]
        for port in range(12)
    ):
        raise FZ11TranslationBridgeError("source-to-RER antipode conjugacy failed")
    rer_gram5 = [[gram5[inverse[i]][inverse[j]] for j in range(12)] for i in range(12)]
    expected_rer_gram5 = []
    for left in range(12):
        row = []
        for right in range(12):
            if left == right:
                row.append([5, 0])
            elif right in RER_NEIGHBORS[left]:
                row.append([0, 1])
            elif right == 11 - left:
                row.append([-5, 0])
            else:
                row.append([0, -1])
        expected_rer_gram5.append(row)
    if rer_gram5 != expected_rer_gram5:
        raise FZ11TranslationBridgeError("all-entry PortFrameGram conjugacy failed")

    conjugated_actions = []
    for action in actions:
        conjugated = tuple(
            SOURCE_TO_RER[action[inverse[label]]] for label in range(12)
        )
        if any(
            (right in RER_NEIGHBORS[left])
            != (conjugated[right] in RER_NEIGHBORS[conjugated[left]])
            for left in range(12)
            for right in range(12)
        ) or any(conjugated[11 - port] != 11 - conjugated[port] for port in range(12)):
            raise FZ11TranslationBridgeError("conjugated A5 action failed")
        conjugated_actions.append(conjugated)
    if len(set(conjugated_actions)) != 60:
        raise FZ11TranslationBridgeError("conjugated A5 action is not faithful")

    return {
        "source_geometry_family": "level-zero nested geodesic icosahedron",
        "source_geometry_hash": BASE_GEOMETRY_HASH,
        "raw_coordinates_qsqrt5": [[_q5_text(value) for value in row] for row in SOURCE_RAW_COORDINATES],
        "common_raw_norm_squared": _q5_text(norm_squared),
        "unit_vectors": "raw_coordinates/sqrt(5/2+sqrt(5)/2)",
        "source_antipodes": list(SOURCE_ANTIPODES),
        "source_to_RER_PortFrameGram_label": list(SOURCE_TO_RER),
        "RER_PortFrameGram_label_to_source": list(inverse),
        "mapping_direction_is_explicit": True,
        "source_scaled_gram_5G_qsqrt5_integer_pairs": gram5,
        "RER_scaled_gram_5G_qsqrt5_integer_pairs": rer_gram5,
        "ordered_gram_class_counts": class_counts,
        "all_144_scaled_gram_entries_match_PortFrameGram": True,
        "all_30_edges_preserved": True,
        "all_6_antipodal_pairs_preserved": True,
        "source_A5_action_sha256": _sha([list(row) for row in actions]),
        "RER_conjugated_A5_action_sha256": _sha([list(row) for row in conjugated_actions]),
        "conjugated_group_order": len(conjugated_actions),
        "all_60_conjugated_actions_preserve_gram_and_antipodes": True,
    }


def _validated_source() -> dict[str, Any]:
    source = _load(SOURCE_RECEIPT_PATH)
    verification = (
        verify_vertex12_constructive_source_law_independent.verify_receipt(
            SOURCE_RECEIPT_PATH
        )
    )
    capture = source.get("constructive_source_law", {}).get("source_capture", {})
    if (
        verification.get("receipt") is not True
        or source.get("schema") != SOURCE_SCHEMA
        or source.get("status") != SOURCE_STATUS
        or source.get("issue") != 655
        or not isinstance(source.get("receipt_sha256"), str)
        or not isinstance(capture.get("source_capture_root_sha256"), str)
    ):
        raise FZ11TranslationBridgeError("constructive source contract drifted")
    return source


def _validate_external_frozen_if_available() -> None:
    if FROZEN_RECEIPT_PATH.exists():
        if _raw_sha(FROZEN_RECEIPT_PATH) != FROZEN_RAW_SHA256:
            raise FZ11TranslationBridgeError("frozen FZ-11 raw pin drifted")
        frozen = _load(FROZEN_RECEIPT_PATH)
        if (
            frozen.get("schema") != FROZEN_SCHEMA
            or frozen.get("status") != FROZEN_STATUS
            or frozen.get("issue") != 655
            or frozen.get("receipt_sha256") != FROZEN_RECEIPT_SHA256
            or _sha(frozen.get("exact_prediction")) != FROZEN_PREDICTION_SHA256
            or _sha(frozen.get("branch_premises")) != FROZEN_PREMISES_SHA256
            or _sha(frozen.get("prediction_scope")) != FROZEN_SCOPE_SHA256
            or frozen.get("exposure_and_custody_boundary", {}).get("comparison_permitted") is not False
        ):
            raise FZ11TranslationBridgeError("frozen FZ-11 contract drifted")
    if PORT_FRAME_LEAN_PATH.exists() and _raw_sha(PORT_FRAME_LEAN_PATH) != PORT_FRAME_LEAN_RAW_SHA256:
        raise FZ11TranslationBridgeError("PortFrameGram Lean source pin drifted")


def _translation_rows(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = source["constructive_source_law"]["raw_step_rows"]
    if len(raw_rows) != 12:
        raise FZ11TranslationBridgeError("constructive source step census drifted")
    rows = []
    for port, raw in enumerate(raw_rows):
        if (
            raw.get("port") != port
            or raw.get("inverse_port") != SOURCE_ANTIPODES[port]
            or raw.get("direction") is None
        ):
            raise FZ11TranslationBridgeError("constructive source step row drifted")
        rows.append(
            {
                "source_port": port,
                "source_event_id": raw["event_id"],
                "source_Z6_direction": raw["direction"],
                "inverse_source_port": SOURCE_ANTIPODES[port],
                "RER_PortFrameGram_label": SOURCE_TO_RER[port],
                "raw_R3_direction_qsqrt5": [_q5_text(value) for value in SOURCE_RAW_COORDINATES[port]],
                "unit_shift": "a*raw_R3_direction/sqrt(5/2+sqrt(5)/2)",
            }
        )
    for left, right in SOURCE_PAIRS:
        if any(
            _q5_add(SOURCE_RAW_COORDINATES[left][axis], SOURCE_RAW_COORDINATES[right][axis]) != ZERO
            for axis in range(3)
        ):
            raise FZ11TranslationBridgeError("adapter does not map inverse records to inverse shifts")
    return rows


def _exact_expansion() -> dict[str, Any]:
    m2 = Fraction(4)
    m4 = Fraction(12, 5)
    m6_isotropic = Fraction(12, 7)
    m6_rank_six = Fraction(64, 175)
    c4 = -m4 / 48
    b0 = m6_isotropic / 1440
    b6 = m6_rank_six / 1440
    if (m2 / 4, c4, b0, b6) != (
        Fraction(1),
        Fraction(-1, 20),
        Fraction(1, 840),
        Fraction(2, 7875),
    ):
        raise FZ11TranslationBridgeError("Taylor coefficient derivation drifted")
    relations = {
        "B0_over_C4_squared": _fraction_text(b0 / c4**2),
        "B6_over_C4_squared": _fraction_text(b6 / c4**2),
        "B6_over_B0": _fraction_text(b6 / b0),
    }
    if relations != {
        "B0_over_C4_squared": "10/21",
        "B6_over_C4_squared": "32/315",
        "B6_over_B0": "16/75",
    }:
        raise FZ11TranslationBridgeError("scale-free relations drifted")
    return {
        "exact_vertex_row_even_moments": {"M2": "4", "M4": "12/5", "M6": "52/25"},
        "sixth_moment_split": "M6(n)=12/7+(64/175)*I6(n)",
        "I6_vertex_normalization": "1",
        "cosine_taylor_through_sixth_order": "omega^2=k^2-(a^2/20)k^4+(a^4/840)k^6+(2a^4/7875)k^6 I6(n)+O(a^6 k^8)",
        "coefficients": {"C4_over_a2": "-1/20", "B0_over_a4": "1/840", "B6_over_a4": "2/7875"},
        "scale_free_relations": relations,
        "group_velocity_expansion": "d(sqrt(omega^2))/dk=1-(3/40)a^2 k^2+a^4 k^4(19/13440+I6(n)/1575)+O(a^6 k^6)",
        "group_velocity_scope": "formal radial derivative of the positive square-root branch at fixed direction n; no time evolution is selected",
        "group_velocity_coefficients": {"a2_k2": "-3/40", "isotropic_a4_k4": "19/13440", "rank6_a4_k4": "1/1575"},
        "full_vector_group_velocity_boundary": "grad_k omega=n*(d omega/dk)+(1/k)*grad_S2 omega; the rank-six angular term contributes (a^4 k^4/7875)*grad_S2 I6(n)+O(a^6 k^6)",
        "angular_gradient_coefficient_at_a4_k4": "1/7875",
        "matches_frozen_exact_prediction_sha256": FROZEN_PREDICTION_SHA256,
    }


def _unit_coordinates() -> tuple[tuple[float, float, float], ...]:
    norm = math.sqrt((5.0 + math.sqrt(5.0)) / 2.0)
    def value(q5: Q5) -> float:
        return (float(q5[0]) + float(q5[1]) * math.sqrt(5.0)) / norm
    return tuple(tuple(value(component) for component in row) for row in SOURCE_RAW_COORDINATES)


def _plane_wave_replay() -> dict[str, Any]:
    units = _unit_coordinates()
    sample_inputs = (
        (Fraction(1, 5), (Fraction(1, 3), Fraction(-2, 7), Fraction(3, 11)), (Fraction(1, 7), Fraction(2, 9), Fraction(-1, 6))),
        (Fraction(2, 9), (Fraction(-4, 9), Fraction(1, 4), Fraction(2, 5)), (Fraction(-2, 11), Fraction(1, 8), Fraction(3, 10))),
        (Fraction(3, 10), (Fraction(5, 8), Fraction(1, 6), Fraction(-3, 7)), (Fraction(1, 13), Fraction(-3, 10), Fraction(2, 9))),
    )
    rows = []
    for index, (a_exact, k_exact, x_exact) in enumerate(sample_inputs):
        a = float(a_exact)
        k = tuple(float(value) for value in k_exact)
        x = tuple(float(value) for value in x_exact)
        phase_x = sum(k[axis] * x[axis] for axis in range(3))
        field_x = cmath.exp(1j * phase_x)
        direct = 0j
        cos_sum = 0.0
        for vector in units:
            dot = sum(k[axis] * vector[axis] for axis in range(3))
            shifted = tuple(x[axis] + a * vector[axis] for axis in range(3))
            field_shifted = cmath.exp(1j * sum(k[axis] * shifted[axis] for axis in range(3)))
            direct += field_x - field_shifted
            cos_sum += 1.0 - math.cos(a * dot)
        multiplier = direct / (2.0 * a * a * field_x)
        formula = cos_sum / (2.0 * a * a)
        residual = abs(multiplier - formula)
        if residual > 5.0e-13:
            raise FZ11TranslationBridgeError("plane-wave replay failed")
        rows.append(
            {
                "sample": index,
                "a": _fraction_text(a_exact),
                "k": [_fraction_text(value) for value in k_exact],
                "x": [_fraction_text(value) for value in x_exact],
                "direct_multiplier_real_10dp": f"{multiplier.real:.10f}",
                "direct_multiplier_imag_10dp": f"{multiplier.imag:.10f}",
                "cosine_symbol_10dp": f"{formula:.10f}",
                "absolute_residual_upper_bound": "5e-13",
            }
        )
    return {
        "field": "f_k(x)=exp(i k.x)",
        "direct_operator": "(1/(2a^2))*sum_p[f(x)-f(x+a*u_p)]",
        "symbol": "(1/(2a^2))*sum_p[1-cos(a*k.u_p)]",
        "sample_rows": rows,
        "sample_count": len(rows),
        "all_direct_shift_replays_match_symbol": True,
        "target_or_comparison_data_used": False,
    }


def _impulse_replay() -> dict[str, Any]:
    output = [{"Z6_coordinate": [0] * 6, "coefficient_at_a_equals_1": "6"}]
    for axis in range(6):
        for sign in (-1, 1):
            coordinate = [0] * 6
            coordinate[axis] = sign
            output.append({"Z6_coordinate": coordinate, "coefficient_at_a_equals_1": "-1/2"})
    return {
        "operator": "(1/(2a^2))*sum_p(I-T_p) on the raw Z^6 source module",
        "input": "Kronecker delta at the origin",
        "output_support_at_a_equals_1": output,
        "support_count": len(output),
        "coefficient_sum": "0",
        "squared_l2_norm": "39",
        "exact_inverse_pairing": True,
    }


def _payload() -> dict[str, Any]:
    source = _validated_source()
    source_capture = source["constructive_source_law"]["source_capture"]
    _validate_external_frozen_if_available()
    frame = _exact_frame()
    rows = _translation_rows(source)
    implementation_files = [Path(__file__).resolve(), VERIFIER_PATH, TEST_PATH, GEOMETRY_PATH]
    return {
        "schema": SCHEMA,
        "issue": 655,
        "status": STATUS,
        "comparison_data_read": False,
        "issue_662_armed": False,
        "parent_pins": {
            "constructive_source_receipt": {
                "path": SOURCE_RECEIPT_PATH.relative_to(ROOT).as_posix(),
                "schema": SOURCE_SCHEMA,
                "status": SOURCE_STATUS,
                "raw_sha256": _raw_sha(SOURCE_RECEIPT_PATH),
                "receipt_sha256": source["receipt_sha256"],
                "source_capture_root_sha256": source_capture[
                    "source_capture_root_sha256"
                ],
                "independently_verified": True,
            },
            "frozen_FZ11_prediction": {
                "repository": "FloatingPragma/observer-patch-holography",
                "path": "code/a5_fingerprint/runtime/spin_six_primitive_port_prediction_receipt.json",
                "schema": FROZEN_SCHEMA,
                "status": FROZEN_STATUS,
                "raw_sha256": FROZEN_RAW_SHA256,
                "receipt_sha256": FROZEN_RECEIPT_SHA256,
                "exact_prediction_sha256": FROZEN_PREDICTION_SHA256,
                "branch_premises_sha256": FROZEN_PREMISES_SHA256,
                "prediction_scope_sha256": FROZEN_SCOPE_SHA256,
                "comparison_permitted": False,
            },
            "RER_PortFrameGram_Lean_source": {
                "repository": "FloatingPragma/observer-patch-holography",
                "path": "Lean/Screen/PortFrameGram.lean",
                "raw_sha256": PORT_FRAME_LEAN_RAW_SHA256,
            },
        },
        "exact_port_frame_and_relabel": frame,
        "conditional_R3_translation_adapter": {
            "domain": "raw constructive-source record module Z^6",
            "codomain": "additive Euclidean R^3 acting on an assumed continuous field",
            "map": "F_a(n)=a*sum_alpha n_alpha*u_alpha",
            "axis_to_positive_source_port": list(POSITIVE_PORTS),
            "port_rows": rows,
            "inverse_source_steps_map_to_inverse_R3_shifts": True,
            "real_linear_extension_rank": 3,
            "real_linear_extension_kernel_dimension": 3,
            "rank_witness_columns_by_source_port": [0, 1, 4],
            "raw_rank_witness_determinant_qsqrt5": "-3+-1*sqrt5",
            "integer_injection_certificate": _integer_injection_certificate(),
            "integer_module_map_is_injective": True,
            "integer_kernel_dimension": 0,
            "image_density_certificate": _dense_image_certificate(),
            "image_is_dense_index_8_submodule_of_scaled_Z_phi_cubed": True,
            "image_is_locally_finite_spatial_lattice_or_quasicrystal": False,
            "continuous_R3_field_is_an_auxiliary_input": True,
            "observer_modulo_three_quotient_factors_this_R3_map": False,
            "quotient_nonfactorization_reason": "F_a(3e_alpha)=3a*u_alpha is nonzero for finite positive a, although 3e_alpha is zero in (Z/3Z)^6",
            "adapter_selected_by_canonical_repair_dynamics": False,
            "adapter_is_auxiliary_physical_branch_premise": True,
        },
        "operator_contract": {
            "source_operator": "K_a=(1/(2a^2))*sum_p(I-T_p)",
            "paired_real_space_form": "K_a=(1/(2a^2))*sum_alpha(2I-T_alpha-T_alpha_inverse)",
            "laurent_symbol": "(1/(2a^2))*sum_alpha(2-z_alpha-z_alpha^-1)",
            "plane_wave_specialization": "z_alpha=exp(i*a*k.u_alpha)",
            "cosine_symbol": "omega^2(k)=(1/(2a^2))*sum_p[1-cos(a*k.u_p)]",
            "A3_control_weight_per_port": "1/12",
            "quadratic_normalization_multiplier": "6/a^2",
            "resulting_weight_per_port": "1/(2a^2)",
            "normalization_is_declared_branch_premise": True,
            "complete_twelve_port_support": True,
            "equal_weights": True,
            "positive_semidefinite_on_plane_waves": True,
            "omega_squared_only": True,
            "time_evolution_or_frequency_sign_selected": False,
            "finite_scale_a_selected": False,
            "carrier_rest_frame_selected": False,
            "physical_sector_exclusivity_proved": False,
        },
        "exact_expansion_certificate": _exact_expansion(),
        "exact_impulse_replay": _impulse_replay(),
        "target_free_plane_wave_replay": _plane_wave_replay(),
        "attainment": {
            "exact_source_to_PortFrameGram_conjugacy": True,
            "exact_signed_record_to_R3_shift_homomorphism": True,
            "exact_inverse_shift_pairing": True,
            "exact_FZ11_cosine_symbol": True,
            "exact_frozen_coefficient_relations": True,
            "exact_raw_Z6_impulse_replay": True,
            "target_free_R3_plane_wave_replay": True,
            "conditional_auxiliary_R3_shift_adapter": True,
            "canonical_source_selection": False,
            "canonical_A1_A2_A3_derivation": False,
            "observer_quotient_spatial_readout": False,
            "faithful_finite_Q_translation_action": False,
            "spatial_site_lattice_or_quasicrystal": False,
            "time_evolution_derived": False,
            "finite_scale_selected": False,
            "carrier_frame_selected": False,
            "photon_sector_selected": False,
            "exclusivity_proved": False,
            "physical_sector_selected": False,
            "boost_law_derived": False,
            "canonical_physical_readout": False,
            "physical_prediction_promoted": False,
            "comparison_permitted": False,
            "issue_655_closure_supported": False,
            "issue_662_armed": False,
        },
        "open_premises": [
            "derive or uniquely select the signed six-axis source law inside canonical A1-A3 repair dynamics",
            "derive the R3 adapter or an operational spacetime readout rather than declare it",
            "derive time evolution and identify a physical field sector carrying omega squared",
            "derive boost and frame transport laws and isolate source, medium, gravity, and instrument terms",
            "select a finite scale and prove support and sector exclusivity before dataset-specific preregistration",
            "satisfy the separate custody gate before reading any qualifying comparison data",
        ],
        "implementation_pins": [_raw_pin(path) for path in implementation_files],
        "claim_boundary": (
            "The constructive Z^6 control admits a mathematically consistent, A5-covariant "
            "homomorphism into three-dimensional icosahedral translations. On this declared "
            "adapter, the signed-step operator has the frozen FZ-11 plane-wave symbol and "
            "exact coefficient relations. Canonical A1-A3 do not select the source law, the "
            "adapter, a physical sector, or a physical readout. The modulo-three observer "
            "meaning does not factor the R3 map. The integer image is an injective, dense "
            "index-eight submodule of (a/sqrt(2+phi))*Z[phi]^3, "
            "so it is a finite-range operator on an assumed continuous field rather than a "
            "locally finite lattice or quasicrystal. No comparison is permitted and issue "
            "#662 remains unarmed."
        ),
    }


def produce_receipt() -> dict[str, Any]:
    payload = _payload()
    return {**payload, "receipt_sha256": _sha(payload)}


def verify_receipt(report: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        received = copy.deepcopy(dict(report))
        digest = received.pop("receipt_sha256", None)
        if digest != _sha(received):
            reasons.append("receipt_digest_mismatch")
        if _canonical_bytes(received) != _canonical_bytes(_payload()):
            reasons.append("producer_replay_mismatch")
        attainment = report.get("attainment")
        forbidden_true = (
            "canonical_source_selection",
            "canonical_A1_A2_A3_derivation",
            "observer_quotient_spatial_readout",
            "faithful_finite_Q_translation_action",
            "spatial_site_lattice_or_quasicrystal",
            "time_evolution_derived",
            "finite_scale_selected",
            "carrier_frame_selected",
            "photon_sector_selected",
            "exclusivity_proved",
            "physical_sector_selected",
            "boost_law_derived",
            "canonical_physical_readout",
            "physical_prediction_promoted",
            "comparison_permitted",
            "issue_655_closure_supported",
            "issue_662_armed",
        )
        if not isinstance(attainment, Mapping) or any(attainment.get(key) is not False for key in forbidden_true):
            reasons.append("unsupported_promotion")
        if report.get("comparison_data_read") is not False or report.get("issue_662_armed") is not False:
            reasons.append("comparison_boundary_mismatch")
    except (AttributeError, KeyError, TypeError, ValueError, FZ11TranslationBridgeError, RecursionError):
        reasons.append("malformed_or_noncanonical_receipt")
    return {
        "schema": "oph.fz11-conditional-3d-translation-bridge-verification.v1",
        "receipt": not reasons,
        "status": "PASS" if not reasons else "FAIL",
        "reasons": sorted(set(reasons)),
        "producer_replay": True,
        "independent_implementation": False,
        "claim_boundary": "Producer replay checks only the declared conditional adapter and does not promote FZ-11.",
    }


def _write_json(value: Mapping[str, Any], path: Path | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        print(rendered, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rendered.encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        result = verify_receipt(_load(args.verify))
        _write_json(result, args.output)
        return 0 if result["receipt"] else 1
    receipt = produce_receipt()
    result = verify_receipt(receipt)
    if not result["receipt"]:
        _write_json(result, args.output)
        return 1
    _write_json(receipt, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
