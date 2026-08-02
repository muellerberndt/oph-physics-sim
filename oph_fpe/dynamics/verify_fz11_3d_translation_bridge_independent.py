"""Independent verifier for the bounded conditional FZ-11 R3 bridge.

The verifier does not import the producer.  It reconstructs the exact
Q(sqrt(5)) frame, the source-to-PortFrameGram conjugacy, all sixty conjugated
proper actions, the Taylor coefficients, the raw impulse, and a paired-shift
plane-wave replay.  Physical promotion flags fail closed.
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
PRODUCER_PATH = ROOT / "oph_fpe/dynamics/fz11_3d_translation_bridge.py"
INDEPENDENT_VERIFIER_PATH = Path(__file__).resolve()
TEST_PATH = ROOT / "tests/test_fz11_3d_translation_bridge.py"
GEOMETRY_PATH = ROOT / "oph_fpe/core/icosahedral.py"
SOURCE_PATH = ROOT / "data/repair_closure/vertex12_constructive_source_law_receipt.json"
RER_ROOT = ROOT.parent / "reverse-engineering-reality"
FROZEN_PATH = RER_ROOT / "code/a5_fingerprint/runtime/spin_six_primitive_port_prediction_receipt.json"
PORT_FRAME_PATH = RER_ROOT / "Lean/Screen/PortFrameGram.lean"

SCHEMA = "oph.fz11-conditional-3d-translation-bridge.v1"
STATUS = (
    "CONDITIONAL_3D_SCALAR_TRANSLATION_ADAPTER_ATTAINED__"
    "CANONICAL_SOURCE_SELECTION_TIME_EVOLUTION_PHOTON_SECTOR_SCALE_FRAME_"
    "BOOST_AND_EXCLUSIVITY_OPEN"
)
FROZEN_RAW_SHA = "sha256:8ac97d7c46199717ed031610efdda65c40f6a251e78715d6bc05888d598e66d8"
FROZEN_SELF_SHA = "sha256:92cc654f2fae04ac1fb90d3f646eae599e6106f217df74641fda2e921bfc0f92"
FROZEN_EXACT_SHA = "sha256:dc3aa0afc49d94eb4ef4220baed21c918cd11fc9e00ebb1611f3643396cd50f2"
FROZEN_PREMISES_SHA = "sha256:543afc4e7313a9155bd9d4a3a20eea7d83ae5a68c6e5acc6b62b2f42b5dc7de0"
FROZEN_SCOPE_SHA = "sha256:8f43767bf0d89684452b93fcc663028a8492c2c6192783870b8dbe3ee8715893"
PORT_FRAME_RAW_SHA = "sha256:d1cebe56450e7586eed730b52068753ebca3a6563da1453679e87fa7c7b653e3"
GEOMETRY_HASH = "e333556ce101cb224d94882270c17e0d7e36469908871ca133e5bbbee2c6eafd"
SOURCE_TO_RER = (8, 10, 1, 3, 7, 11, 0, 4, 6, 9, 2, 5)
ANTIPODES = (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8)
PAIRS = ((0, 3), (1, 2), (4, 7), (5, 6), (8, 11), (9, 10))
POSITIVE = (0, 1, 4, 5, 8, 9)
NEIGHBORS = (
    (1, 2, 3, 4, 6), (0, 2, 3, 5, 7), (0, 1, 4, 5, 8),
    (0, 1, 6, 7, 9), (0, 2, 6, 8, 10), (1, 2, 7, 8, 11),
    (0, 3, 4, 9, 10), (1, 3, 5, 9, 11), (2, 4, 5, 10, 11),
    (3, 6, 7, 10, 11), (4, 6, 8, 9, 11), (5, 7, 8, 9, 10),
)

Q5 = tuple[Fraction, Fraction]
Z: Q5 = (Fraction(0), Fraction(0))
UNIT: Q5 = (Fraction(1), Fraction(0))
P: Q5 = (Fraction(1, 2), Fraction(1, 2))
M: Q5 = (Fraction(-1), Fraction(0))
MP: Q5 = (Fraction(-1, 2), Fraction(-1, 2))
COORDINATES: tuple[tuple[Q5, Q5, Q5], ...] = (
    (M, P, Z), (UNIT, P, Z), (M, MP, Z), (UNIT, MP, Z),
    (Z, M, P), (Z, UNIT, P), (Z, M, MP), (Z, UNIT, MP),
    (P, Z, M), (P, Z, UNIT), (MP, Z, M), (MP, Z, UNIT),
)

OPEN_PREMISES = [
    "derive or uniquely select the signed six-axis source law inside canonical A1-A3 repair dynamics",
    "derive the R3 adapter or an operational spacetime readout rather than declare it",
    "derive time evolution and identify a physical field sector carrying omega squared",
    "derive boost and frame transport laws and isolate source, medium, gravity, and instrument terms",
    "select a finite scale and prove support and sector exclusivity before dataset-specific preregistration",
    "satisfy the separate custody gate before reading any qualifying comparison data",
]
CLAIM_BOUNDARY = (
    "The constructive Z^6 control admits a mathematically consistent, A5-covariant "
    "homomorphism into three-dimensional icosahedral translations. On this declared "
    "adapter, the signed-step operator has the frozen FZ-11 plane-wave symbol and "
    "exact coefficient relations. Canonical A1-A3 do not select the source law, the "
    "adapter, a physical sector, or a physical readout. The modulo-three observer "
    "meaning does not factor the R3 map. The integer image is an injective, dense "
    "index-eight submodule of (a/sqrt(2+phi))*Z[phi]^3, so it is a finite-range "
    "operator on an assumed continuous field rather than a locally finite lattice "
    "or quasicrystal. No comparison is permitted and issue #662 remains unarmed."
)


class IndependentVerificationError(RuntimeError):
    """Raised when any independent bridge check fails."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


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


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
    )
    if not isinstance(value, dict):
        raise IndependentVerificationError(f"{path} is not an object")
    return value


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentVerificationError(message)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    _fail(set(value) == expected, f"{label} key set")


def _qadd(left: Q5, right: Q5) -> Q5:
    return left[0] + right[0], left[1] + right[1]


def _qmul(left: Q5, right: Q5) -> Q5:
    return left[0] * right[0] + 5 * left[1] * right[1], left[0] * right[1] + left[1] * right[0]


def _qinv(value: Q5) -> Q5:
    denominator = value[0] ** 2 - 5 * value[1] ** 2
    _fail(denominator != 0, "Qsqrt5 inverse")
    return value[0] / denominator, -value[1] / denominator


def _qscale(value: Q5, scalar: Fraction) -> Q5:
    return value[0] * scalar, value[1] * scalar


def _dot(left: Sequence[Q5], right: Sequence[Q5]) -> Q5:
    total = Z
    for left_value, right_value in zip(left, right, strict=True):
        total = _qadd(total, _qmul(left_value, right_value))
    return total


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _qtext(value: Q5) -> str:
    return f"{_fraction_text(value[0])}+{_fraction_text(value[1])}*sqrt5"


def _inverse(row: Sequence[int]) -> tuple[int, ...]:
    _fail(sorted(int(value) for value in row) == list(range(len(row))), "permutation")
    output = [0] * len(row)
    for source, target in enumerate(row):
        output[int(target)] = source
    return tuple(output)


def _expected_gram() -> tuple[list[list[list[int]]], list[list[list[int]]]]:
    norm = _dot(COORDINATES[0], COORDINATES[0])
    _fail(norm == (Fraction(5, 2), Fraction(1, 2)), "frame norm")
    source = []
    for left in COORDINATES:
        row = []
        for right in COORDINATES:
            value = _qscale(_qmul(_dot(left, right), _qinv(norm)), Fraction(5))
            _fail(value[0].denominator == value[1].denominator == 1, "integral 5G")
            row.append([int(value[0]), int(value[1])])
        source.append(row)
    inverse = _inverse(SOURCE_TO_RER)
    rer = [[source[inverse[i]][inverse[j]] for j in range(12)] for i in range(12)]
    for i in range(12):
        for j in range(12):
            if i == j:
                expected = [5, 0]
            elif j in NEIGHBORS[i]:
                expected = [0, 1]
            elif j == 11 - i:
                expected = [-5, 0]
            else:
                expected = [0, -1]
            _fail(rer[i][j] == expected, "PortFrameGram all-entry conjugacy")
    return source, rer


def _expected_action_hashes() -> tuple[str, str]:
    source = tuple(tuple(int(value) for value in row) for row in icosahedral_a5_port_permutations())
    _fail(len(source) == len(set(source)) == 60, "proper action census")
    inverse = _inverse(SOURCE_TO_RER)
    conjugated = tuple(
        tuple(SOURCE_TO_RER[action[inverse[label]]] for label in range(12))
        for action in source
    )
    _fail(len(set(conjugated)) == 60, "conjugated action faithfulness")
    for action in conjugated:
        _fail(all(action[11 - i] == 11 - action[i] for i in range(12)), "A5 antipodes")
        _fail(
            all((j in NEIGHBORS[i]) == (action[j] in NEIGHBORS[action[i]]) for i in range(12) for j in range(12)),
            "A5 adjacency",
        )
    return _sha([list(row) for row in source]), _sha([list(row) for row in conjugated])


def _check_parent_pins(report: Mapping[str, Any]) -> dict[str, Any]:
    parents = report["parent_pins"]
    _exact_keys(parents, {"constructive_source_receipt", "frozen_FZ11_prediction", "RER_PortFrameGram_Lean_source"}, "parents")
    source_pin = parents["constructive_source_receipt"]
    source = _load(SOURCE_PATH)
    source_verification = (
        verify_vertex12_constructive_source_law_independent.verify_receipt(SOURCE_PATH)
    )
    _fail(source_verification.get("receipt") is True, "source independent verification")
    _fail(
        source.get("schema") == "oph.vertex12-constructive-source-law-control.v1"
        and source.get("status")
        == "CONSTRUCTIVE_SOURCE_LAW_CONTROL_ATTAINED__CANONICAL_A1_A2_A3_SELECTION_AND_PHYSICAL_ATTACHMENT_OPEN"
        and source.get("issue") == 655,
        "source contract",
    )
    source_capture_sha = source["constructive_source_law"]["source_capture"][
        "source_capture_root_sha256"
    ]
    _fail(source_pin == {
        "path": "data/repair_closure/vertex12_constructive_source_law_receipt.json",
        "schema": "oph.vertex12-constructive-source-law-control.v1",
        "status": "CONSTRUCTIVE_SOURCE_LAW_CONTROL_ATTAINED__CANONICAL_A1_A2_A3_SELECTION_AND_PHYSICAL_ATTACHMENT_OPEN",
        "raw_sha256": _raw_sha(SOURCE_PATH),
        "receipt_sha256": source["receipt_sha256"],
        "source_capture_root_sha256": source_capture_sha,
        "independently_verified": True,
    }, "source parent pin")
    frozen_pin = parents["frozen_FZ11_prediction"]
    _fail(frozen_pin == {
        "repository": "FloatingPragma/observer-patch-holography",
        "path": "code/a5_fingerprint/runtime/spin_six_primitive_port_prediction_receipt.json",
        "schema": "oph.spin_six_primitive_port_prediction.v1",
        "status": "FROZEN_PROSPECTIVE_PRIMITIVE_TWELVE_PORT_BRANCH_PREDICTION__PHYSICAL_COMPARISON_UNARMED",
        "raw_sha256": FROZEN_RAW_SHA,
        "receipt_sha256": FROZEN_SELF_SHA,
        "exact_prediction_sha256": FROZEN_EXACT_SHA,
        "branch_premises_sha256": FROZEN_PREMISES_SHA,
        "prediction_scope_sha256": FROZEN_SCOPE_SHA,
        "comparison_permitted": False,
    }, "frozen parent pin")
    if FROZEN_PATH.exists():
        _fail(_raw_sha(FROZEN_PATH) == FROZEN_RAW_SHA, "frozen bytes")
        frozen = _load(FROZEN_PATH)
        _fail(frozen.get("receipt_sha256") == FROZEN_SELF_SHA, "frozen self digest")
        _fail(_sha(frozen.get("exact_prediction")) == FROZEN_EXACT_SHA, "frozen exact payload")
        _fail(_sha(frozen.get("branch_premises")) == FROZEN_PREMISES_SHA, "frozen premises")
        _fail(_sha(frozen.get("prediction_scope")) == FROZEN_SCOPE_SHA, "frozen scope")
        _fail(frozen["exposure_and_custody_boundary"]["comparison_permitted"] is False, "frozen custody")
    lean_pin = parents["RER_PortFrameGram_Lean_source"]
    _fail(lean_pin == {
        "repository": "FloatingPragma/observer-patch-holography",
        "path": "Lean/Screen/PortFrameGram.lean",
        "raw_sha256": PORT_FRAME_RAW_SHA,
    }, "Lean frame pin")
    if PORT_FRAME_PATH.exists():
        _fail(_raw_sha(PORT_FRAME_PATH) == PORT_FRAME_RAW_SHA, "Lean frame bytes")
    return source


def _check_frame(frame: Mapping[str, Any]) -> None:
    _exact_keys(frame, {
        "source_geometry_family", "source_geometry_hash", "raw_coordinates_qsqrt5",
        "common_raw_norm_squared", "unit_vectors", "source_antipodes",
        "source_to_RER_PortFrameGram_label", "RER_PortFrameGram_label_to_source",
        "mapping_direction_is_explicit", "source_scaled_gram_5G_qsqrt5_integer_pairs",
        "RER_scaled_gram_5G_qsqrt5_integer_pairs", "ordered_gram_class_counts",
        "all_144_scaled_gram_entries_match_PortFrameGram", "all_30_edges_preserved",
        "all_6_antipodal_pairs_preserved", "source_A5_action_sha256",
        "RER_conjugated_A5_action_sha256", "conjugated_group_order",
        "all_60_conjugated_actions_preserve_gram_and_antipodes",
    }, "frame")
    level = build_geodesic_icosahedral_tower(0).levels[0]
    _fail(frame["source_geometry_family"] == "level-zero nested geodesic icosahedron", "geometry family")
    _fail(level.geometry_hash == frame["source_geometry_hash"] == GEOMETRY_HASH, "geometry hash")
    _fail(frame["raw_coordinates_qsqrt5"] == [[_qtext(value) for value in row] for row in COORDINATES], "coordinates")
    _fail(frame["common_raw_norm_squared"] == "5/2+1/2*sqrt5", "norm string")
    _fail(frame["unit_vectors"] == "raw_coordinates/sqrt(5/2+sqrt(5)/2)", "unit normalization")
    _fail(frame["source_antipodes"] == list(ANTIPODES), "source antipodes")
    _fail(frame["source_to_RER_PortFrameGram_label"] == list(SOURCE_TO_RER), "source-to-RER")
    _fail(frame["RER_PortFrameGram_label_to_source"] == list(_inverse(SOURCE_TO_RER)), "RER-to-source")
    source_gram, rer_gram = _expected_gram()
    _fail(frame["source_scaled_gram_5G_qsqrt5_integer_pairs"] == source_gram, "source Gram")
    _fail(frame["RER_scaled_gram_5G_qsqrt5_integer_pairs"] == rer_gram, "RER Gram")
    _fail(frame["ordered_gram_class_counts"] == {"diagonal": 12, "adjacent": 60, "distance_two": 60, "antipodal": 12}, "Gram census")
    source_hash, conjugated_hash = _expected_action_hashes()
    _fail(frame["source_A5_action_sha256"] == source_hash, "source A5 hash")
    _fail(frame["RER_conjugated_A5_action_sha256"] == conjugated_hash, "conjugated A5 hash")
    for key in ("mapping_direction_is_explicit", "all_144_scaled_gram_entries_match_PortFrameGram", "all_30_edges_preserved", "all_6_antipodal_pairs_preserved", "all_60_conjugated_actions_preserve_gram_and_antipodes"):
        _fail(frame[key] is True, key)
    _fail(frame["conjugated_group_order"] == 60, "group order")


def _check_adapter(adapter: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    expected_keys = {
        "domain", "codomain", "map", "axis_to_positive_source_port", "port_rows",
        "inverse_source_steps_map_to_inverse_R3_shifts", "real_linear_extension_rank",
        "real_linear_extension_kernel_dimension", "rank_witness_columns_by_source_port",
        "raw_rank_witness_determinant_qsqrt5", "integer_module_map_is_injective",
        "integer_injection_certificate",
        "integer_kernel_dimension", "image_density_certificate",
        "image_is_dense_index_8_submodule_of_scaled_Z_phi_cubed",
        "image_is_locally_finite_spatial_lattice_or_quasicrystal",
        "continuous_R3_field_is_an_auxiliary_input", "observer_modulo_three_quotient_factors_this_R3_map",
        "quotient_nonfactorization_reason", "adapter_selected_by_canonical_repair_dynamics",
        "adapter_is_auxiliary_physical_branch_premise",
    }
    _exact_keys(adapter, expected_keys, "adapter")
    _fail(adapter["domain"] == "raw constructive-source record module Z^6", "adapter domain")
    _fail(
        adapter["codomain"]
        == "additive Euclidean R^3 acting on an assumed continuous field",
        "adapter codomain",
    )
    _fail(adapter["map"] == "F_a(n)=a*sum_alpha n_alpha*u_alpha", "adapter map")
    _fail(adapter["axis_to_positive_source_port"] == list(POSITIVE), "positive ports")
    _fail(adapter["real_linear_extension_rank"] == 3 and adapter["real_linear_extension_kernel_dimension"] == 3, "real rank")
    _fail(adapter["rank_witness_columns_by_source_port"] == [0, 1, 4], "rank witness")
    determinant = -2 * _qmul(P, P)[0], -2 * _qmul(P, P)[1]
    _fail(determinant == (Fraction(-3), Fraction(-1)), "independent determinant")
    _fail(adapter["raw_rank_witness_determinant_qsqrt5"] == "-3+-1*sqrt5", "determinant row")
    split_rows: list[list[Fraction]] = []
    for coordinate in range(3):
        rational_row = []
        phi_row = []
        for port in POSITIVE:
            a, b = COORDINATES[port][coordinate]
            rational_row.append(a - b)
            phi_row.append(2 * b)
        split_rows.extend((rational_row, phi_row))
    work = [list(row) for row in split_rows]
    injection_determinant = Fraction(1)
    for column in range(6):
        pivot = next((row for row in range(column, 6) if work[row][column]), None)
        _fail(pivot is not None, "injection rank")
        assert pivot is not None
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            injection_determinant = -injection_determinant
        pivot_value = work[column][column]
        injection_determinant *= pivot_value
        for row in range(column + 1, 6):
            factor = work[row][column] / pivot_value
            for index in range(column, 6):
                work[row][index] -= factor * work[column][index]
    _fail(injection_determinant == -8, "injection determinant")
    _fail(adapter["integer_injection_certificate"] == {
        "basis": "split each R3 coordinate in the Q-basis {1,phi}",
        "positive_axis_order": list(POSITIVE),
        "six_by_six_rational_coefficient_matrix": [[_fraction_text(value) for value in row] for row in split_rows],
        "determinant": "-8",
        "rank": 6,
        "conclusion": "F_a(n)=0 implies n=0 for every finite positive a",
    }, "injection certificate")
    # The six independent rational/phi coefficient equations kill the six
    # integer coefficients; hence the Z-module map is injective.
    _fail(adapter["integer_module_map_is_injective"] is True and adapter["integer_kernel_dimension"] == 0, "integer injection")
    # The coefficient matrix puts the raw image at exact index eight in
    # Z[phi]^3.  Equivalently, an integral character on all six generators
    # makes both 2*p_i and 2*phi*p_i integral for each coordinate.  Since
    # phi is irrational, the annihilator is trivial.  Common nonzero scaling
    # by a/sqrt(2+phi) preserves density.
    _fail(math.isqrt(5) ** 2 != 5, "sqrt(5) irrationality witness")
    _fail(adapter["image_density_certificate"] == {
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
    }, "dense-image certificate")
    _fail(
        adapter["image_is_dense_index_8_submodule_of_scaled_Z_phi_cubed"] is True,
        "dense image",
    )
    _fail(adapter["image_is_locally_finite_spatial_lattice_or_quasicrystal"] is False, "non-lattice boundary")
    _fail(adapter["continuous_R3_field_is_an_auxiliary_input"] is True, "continuous input")
    _fail(adapter["observer_modulo_three_quotient_factors_this_R3_map"] is False, "finite quotient boundary")
    _fail(
        adapter["quotient_nonfactorization_reason"]
        == "F_a(3e_alpha)=3a*u_alpha is nonzero for finite positive a, although 3e_alpha is zero in (Z/3Z)^6",
        "quotient nonfactorization reason",
    )
    _fail(adapter["adapter_selected_by_canonical_repair_dynamics"] is False and adapter["adapter_is_auxiliary_physical_branch_premise"] is True, "adapter status")
    raw_rows = source["constructive_source_law"]["raw_step_rows"]
    rows = adapter["port_rows"]
    _fail(len(rows) == 12, "port row count")
    for port, row in enumerate(rows):
        _exact_keys(row, {"source_port", "source_event_id", "source_Z6_direction", "inverse_source_port", "RER_PortFrameGram_label", "raw_R3_direction_qsqrt5", "unit_shift"}, f"port row {port}")
        _fail(row["source_port"] == port, "port index")
        _fail(row["source_event_id"] == raw_rows[port]["event_id"], "event id")
        _fail(row["source_Z6_direction"] == raw_rows[port]["direction"], "source direction")
        _fail(row["inverse_source_port"] == ANTIPODES[port], "inverse port")
        _fail(row["RER_PortFrameGram_label"] == SOURCE_TO_RER[port], "RER label")
        _fail(row["raw_R3_direction_qsqrt5"] == [_qtext(value) for value in COORDINATES[port]], "R3 direction")
    _fail(adapter["inverse_source_steps_map_to_inverse_R3_shifts"] is True, "inverse shifts")


def _check_expansion(expansion: Mapping[str, Any]) -> None:
    _exact_keys(expansion, {
        "exact_vertex_row_even_moments", "sixth_moment_split", "I6_vertex_normalization",
        "cosine_taylor_through_sixth_order", "coefficients", "scale_free_relations",
        "group_velocity_expansion", "group_velocity_scope", "group_velocity_coefficients",
        "full_vector_group_velocity_boundary", "angular_gradient_coefficient_at_a4_k4",
        "matches_frozen_exact_prediction_sha256",
    }, "expansion")
    c4 = -Fraction(12, 5) / 48
    b0 = Fraction(12, 7) / 1440
    b6 = Fraction(64, 175) / 1440
    _fail((c4, b0, b6) == (Fraction(-1, 20), Fraction(1, 840), Fraction(2, 7875)), "coefficients")
    _fail(expansion["exact_vertex_row_even_moments"] == {"M2": "4", "M4": "12/5", "M6": "52/25"}, "moments")
    _fail(expansion["coefficients"] == {"C4_over_a2": "-1/20", "B0_over_a4": "1/840", "B6_over_a4": "2/7875"}, "stored coefficients")
    _fail(expansion["scale_free_relations"] == {"B0_over_C4_squared": "10/21", "B6_over_C4_squared": "32/315", "B6_over_B0": "16/75"}, "ratios")
    _fail(expansion["group_velocity_coefficients"] == {"a2_k2": "-3/40", "isotropic_a4_k4": "19/13440", "rank6_a4_k4": "1/1575"}, "group velocity")
    _fail(
        expansion["group_velocity_scope"]
        == "formal radial derivative of the positive square-root branch at fixed direction n; no time evolution is selected",
        "group velocity scope",
    )
    _fail(expansion["angular_gradient_coefficient_at_a4_k4"] == "1/7875", "angular velocity coefficient")
    _fail(expansion["matches_frozen_exact_prediction_sha256"] == FROZEN_EXACT_SHA, "frozen match")
    _fail(expansion == {
        "exact_vertex_row_even_moments": {"M2": "4", "M4": "12/5", "M6": "52/25"},
        "sixth_moment_split": "M6(n)=12/7+(64/175)*I6(n)",
        "I6_vertex_normalization": "1",
        "cosine_taylor_through_sixth_order": (
            "omega^2=k^2-(a^2/20)k^4+(a^4/840)k^6+"
            "(2a^4/7875)k^6 I6(n)+O(a^6 k^8)"
        ),
        "coefficients": {
            "C4_over_a2": "-1/20",
            "B0_over_a4": "1/840",
            "B6_over_a4": "2/7875",
        },
        "scale_free_relations": {
            "B0_over_C4_squared": "10/21",
            "B6_over_C4_squared": "32/315",
            "B6_over_B0": "16/75",
        },
        "group_velocity_expansion": (
            "d(sqrt(omega^2))/dk=1-(3/40)a^2 k^2+"
            "a^4 k^4(19/13440+I6(n)/1575)+O(a^6 k^6)"
        ),
        "group_velocity_scope": (
            "formal radial derivative of the positive square-root branch at fixed "
            "direction n; no time evolution is selected"
        ),
        "group_velocity_coefficients": {
            "a2_k2": "-3/40",
            "isotropic_a4_k4": "19/13440",
            "rank6_a4_k4": "1/1575",
        },
        "full_vector_group_velocity_boundary": (
            "grad_k omega=n*(d omega/dk)+(1/k)*grad_S2 omega; the rank-six "
            "angular term contributes (a^4 k^4/7875)*grad_S2 I6(n)+O(a^6 k^6)"
        ),
        "angular_gradient_coefficient_at_a4_k4": "1/7875",
        "matches_frozen_exact_prediction_sha256": FROZEN_EXACT_SHA,
    }, "complete expansion packet")


def _check_impulse(impulse: Mapping[str, Any]) -> None:
    _exact_keys(impulse, {
        "operator", "input", "output_support_at_a_equals_1", "support_count",
        "coefficient_sum", "squared_l2_norm", "exact_inverse_pairing",
    }, "impulse")
    rows = impulse["output_support_at_a_equals_1"]
    _fail(
        impulse["operator"]
        == "(1/(2a^2))*sum_p(I-T_p) on the raw Z^6 source module",
        "impulse operator",
    )
    _fail(impulse["input"] == "Kronecker delta at the origin", "impulse input")
    _fail(len(rows) == impulse["support_count"] == 13, "impulse support")
    expected = {tuple([0] * 6): Fraction(6)}
    for axis in range(6):
        for sign in (-1, 1):
            coordinate = [0] * 6
            coordinate[axis] = sign
            expected[tuple(coordinate)] = Fraction(-1, 2)
    received = {tuple(row["Z6_coordinate"]): Fraction(row["coefficient_at_a_equals_1"]) for row in rows}
    _fail(received == expected, "impulse coefficients")
    _fail(sum(received.values()) == 0, "impulse mass")
    _fail(sum(value * value for value in received.values()) == 39, "impulse norm")
    _fail(impulse["coefficient_sum"] == "0" and impulse["squared_l2_norm"] == "39", "impulse summaries")
    _fail(impulse["exact_inverse_pairing"] is True, "impulse inverse pairing")


def _unit_coordinates() -> tuple[tuple[float, float, float], ...]:
    norm = math.sqrt((5.0 + math.sqrt(5.0)) / 2.0)
    return tuple(tuple((float(a) + float(b) * math.sqrt(5.0)) / norm for a, b in row) for row in COORDINATES)


def _check_plane_waves(replay: Mapping[str, Any]) -> None:
    _exact_keys(replay, {
        "field", "direct_operator", "symbol", "sample_rows", "sample_count",
        "all_direct_shift_replays_match_symbol", "target_or_comparison_data_used",
    }, "plane-wave replay")
    units = _unit_coordinates()
    _fail(replay["field"] == "f_k(x)=exp(i k.x)", "plane-wave field")
    _fail(
        replay["direct_operator"]
        == "(1/(2a^2))*sum_p[f(x)-f(x+a*u_p)]",
        "plane-wave direct operator",
    )
    _fail(
        replay["symbol"] == "(1/(2a^2))*sum_p[1-cos(a*k.u_p)]",
        "plane-wave symbol",
    )
    rows = replay["sample_rows"]
    _fail(replay["sample_count"] == len(rows) == 3, "plane-wave count")
    for index, row in enumerate(rows):
        _exact_keys(row, {
            "sample", "a", "k", "x", "direct_multiplier_real_10dp",
            "direct_multiplier_imag_10dp", "cosine_symbol_10dp",
            "absolute_residual_upper_bound",
        }, f"plane-wave row {index}")
        _fail(row["sample"] == index, "sample index")
        a = float(Fraction(row["a"]))
        k = tuple(float(Fraction(value)) for value in row["k"])
        x = tuple(float(Fraction(value)) for value in row["x"])
        field = cmath.exp(1j * sum(k[i] * x[i] for i in range(3)))
        direct = 0j
        for positive in POSITIVE:
            vector = units[positive]
            plus = cmath.exp(1j * sum(k[i] * (x[i] + a * vector[i]) for i in range(3)))
            minus = cmath.exp(1j * sum(k[i] * (x[i] - a * vector[i]) for i in range(3)))
            direct += 2 * field - plus - minus
        multiplier = direct / (2 * a * a * field)
        formula = sum(1 - math.cos(a * sum(k[i] * vector[i] for i in range(3))) for vector in units) / (2 * a * a)
        _fail(abs(multiplier - formula) <= 5e-13, "independent plane wave")
        _fail(row["direct_multiplier_real_10dp"] == f"{multiplier.real:.10f}", "plane-wave real")
        _fail(row["direct_multiplier_imag_10dp"] == f"{multiplier.imag:.10f}", "plane-wave imaginary")
        _fail(row["cosine_symbol_10dp"] == f"{formula:.10f}", "plane-wave symbol")
        _fail(row["absolute_residual_upper_bound"] == "5e-13", "plane-wave residual bound")
    _fail(replay["all_direct_shift_replays_match_symbol"] is True, "plane-wave verdict")
    _fail(replay["target_or_comparison_data_used"] is False, "target boundary")


def _check_implementation_pins(report: Mapping[str, Any]) -> None:
    expected_paths = (PRODUCER_PATH, INDEPENDENT_VERIFIER_PATH, TEST_PATH, GEOMETRY_PATH)
    expected = [
        {"path": path.relative_to(ROOT).as_posix(), "bytes": len(path.read_bytes()), "sha256": _raw_sha(path)}
        for path in expected_paths
    ]
    _fail(report["implementation_pins"] == expected, "implementation pins")


def verify_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    report = _load(path)
    _exact_keys(report, {
        "schema", "issue", "status", "comparison_data_read", "issue_662_armed",
        "parent_pins", "exact_port_frame_and_relabel", "conditional_R3_translation_adapter",
        "operator_contract", "exact_expansion_certificate", "exact_impulse_replay",
        "target_free_plane_wave_replay", "attainment", "open_premises",
        "implementation_pins", "claim_boundary", "receipt_sha256",
    }, "top level")
    received = copy.deepcopy(report)
    digest = received.pop("receipt_sha256")
    _fail(digest == _sha(received), "receipt digest")
    _fail(report["schema"] == SCHEMA and report["status"] == STATUS and report["issue"] == 655, "identity")
    _fail(report["comparison_data_read"] is False and report["issue_662_armed"] is False, "custody")
    source = _check_parent_pins(report)
    _check_frame(report["exact_port_frame_and_relabel"])
    _check_adapter(report["conditional_R3_translation_adapter"], source)
    operator = report["operator_contract"]
    _exact_keys(operator, {
        "source_operator", "paired_real_space_form", "laurent_symbol",
        "plane_wave_specialization", "cosine_symbol", "A3_control_weight_per_port",
        "quadratic_normalization_multiplier", "resulting_weight_per_port",
        "normalization_is_declared_branch_premise", "complete_twelve_port_support",
        "equal_weights", "positive_semidefinite_on_plane_waves", "omega_squared_only",
        "time_evolution_or_frequency_sign_selected", "finite_scale_a_selected",
        "carrier_rest_frame_selected", "physical_sector_exclusivity_proved",
    }, "operator")
    _fail(operator == {
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
    }, "complete operator contract")
    _check_expansion(report["exact_expansion_certificate"])
    _check_impulse(report["exact_impulse_replay"])
    _check_plane_waves(report["target_free_plane_wave_replay"])
    attainment = report["attainment"]
    positive = {
        "exact_source_to_PortFrameGram_conjugacy", "exact_signed_record_to_R3_shift_homomorphism",
        "exact_inverse_shift_pairing", "exact_FZ11_cosine_symbol", "exact_frozen_coefficient_relations",
        "exact_raw_Z6_impulse_replay", "target_free_R3_plane_wave_replay", "conditional_auxiliary_R3_shift_adapter",
    }
    negative = {
        "canonical_source_selection", "canonical_A1_A2_A3_derivation", "observer_quotient_spatial_readout",
        "faithful_finite_Q_translation_action", "spatial_site_lattice_or_quasicrystal", "physical_sector_selected",
        "time_evolution_derived", "finite_scale_selected", "carrier_frame_selected", "photon_sector_selected", "exclusivity_proved",
        "boost_law_derived", "canonical_physical_readout", "physical_prediction_promoted", "comparison_permitted",
        "issue_655_closure_supported", "issue_662_armed",
    }
    _exact_keys(attainment, positive | negative, "attainment")
    _fail(all(attainment[key] is True for key in positive), "attained checks")
    _fail(all(attainment[key] is False for key in negative), "promotion boundary")
    _fail(report["open_premises"] == OPEN_PREMISES, "open premises")
    _fail(report["claim_boundary"] == CLAIM_BOUNDARY, "claim boundary")
    _check_implementation_pins(report)
    return {
        "schema": "oph.fz11-conditional-3d-translation-bridge-independent-verification.v1",
        "receipt": True,
        "status": "PASS",
        "producer_imported": False,
        "exact_Qsqrt5_frame_independently_reimplemented": True,
        "checked_scaled_gram_entries": 144,
        "checked_A5_actions": 60,
        "checked_source_events": 12,
        "checked_impulse_sites": 13,
        "checked_plane_wave_samples": 3,
        "comparison_data_read": False,
        "issue_662_armed": False,
        "claim_boundary": "Independent verification certifies only the bounded conditional adapter; it does not select a physical branch.",
    }


def _write(value: Mapping[str, Any], path: Path | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        print(rendered, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rendered.encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="?", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_receipt(args.receipt)
    except (IndependentVerificationError, AttributeError, KeyError, TypeError, ValueError, RecursionError) as error:
        result = {
            "schema": "oph.fz11-conditional-3d-translation-bridge-independent-verification.v1",
            "receipt": False,
            "status": "FAIL",
            "reasons": [str(error)],
            "producer_imported": False,
        }
    _write(result, args.output)
    return 0 if result["receipt"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
