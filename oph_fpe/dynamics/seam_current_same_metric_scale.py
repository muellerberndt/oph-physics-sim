"""Certify the source-native dimensionless scale of the FZ-12 seam action.

The source seam action is addition by the exact ``D6`` boundary current.  The
repair-selected response Gram is the metric on that same cumulative-record
action.  Their composition therefore fixes an exact *dimensionless* length:

    ||delta_edge||_G^2 = 2 - 2/sqrt(5).

This is the norm of the full unit-current event.  The raw Cartesian witness
used by the frozen FZ-12 moment packet has squared norm four; division by the
common raw port radius ``2 + phi`` gives the response-Gram value above.  The
FZ-12 unit direction is half the raw seam difference, so the source event is
``a_edge`` times that direction with ``a_edge^2 = 2 - 2/sqrt(5)`` in the
unit-diagonal response metric.

No dimensionful ruler follows.  A common conversion ``lambda > 0`` from one
response-metric unit to physical length preserves every pinned finite source
relation while sending the physical seam length to zero as ``lambda`` tends
to zero.  The exact dimensionless scale and the absent physical lower bound
are recorded separately.  No target or comparison data are read.
"""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
RER_ROOT = ROOT.parent / "reverse-engineering-reality"
DEFAULT_RECEIPT = (
    ROOT / "data/repair_closure/seam_current_same_metric_scale_receipt.json"
)
LOAD_QUOTIENT_RECEIPT = (
    ROOT / "data/repair_closure/port_load_metric_quotient_receipt.json"
)
COMPLETION_RECEIPT = (
    ROOT / "data/repair_closure/port_gram_completion_bridge_receipt.json"
)
ACTION_RECEIPT = (
    ROOT / "data/repair_closure/port_gram_equivariant_action_receipt.json"
)
PORT_MEASURE_RECEIPT = (
    ROOT / "data/repair_closure/primitive_port_dual_measure_receipt.json"
)
FZ11_RECEIPT = ROOT / "data/repair_closure/fz11_3d_translation_bridge_receipt.json"
CARRIER_MANIFEST = ROOT / "tests/fixtures/echosahedral_federation_reference.json"
FZ12_RECEIPT = (
    RER_ROOT
    / "code/a5_fingerprint/runtime/seam_current_edge_prediction_receipt.json"
)
HOMOGENEOUS_ACTION_LEAN = (
    RER_ROOT / "Lean/Screen/SeamCurrentHomogeneousAction.lean"
)
PRODUCER_PATH = Path(__file__).resolve()
VERIFIER_PATH = (
    ROOT / "oph_fpe/dynamics/verify_seam_current_same_metric_scale_independent.py"
)
TEST_PATH = ROOT / "tests/test_seam_current_same_metric_scale.py"

SCHEMA = "oph.seam-current-same-metric-scale.v1"
STATUS = (
    "SOURCE_NATIVE_DIMENSIONLESS_SEAM_ACTION_SCALE_ATTAINED__"
    "PHYSICAL_UNIT_CELL_ATTACHMENT_AND_LOWER_BOUND_OPEN"
)
LOAD_QUOTIENT_SCHEMA = "oph.port-load-repair-gram-metric-quotient.v1"
LOAD_QUOTIENT_STATUS = (
    "EXACT_INTEGER_LOAD_METRIC_QUOTIENT_AND_MEAN_INTERTWINER_ATTAINED__"
    "PATHWISE_DESCENT_POSITION_SEMANTICS_AND_PHYSICAL_ACTION_OPEN"
)
COMPLETION_SCHEMA = "oph.port-gram-hausdorff-completion-bridge.v1"
COMPLETION_STATUS = (
    "EXACT_REPAIR_RESPONSE_GRAM_QUOTIENT_AND_3D_COMPLETION_ATTAINED__"
    "A1R_SIGNED_RECORD_MODULE_AND_A2R_POSITION_READBACK_PREMISES_OPEN"
)
ACTION_SCHEMA = "oph.port-gram-equivariant-completion-action.v1"
ACTION_STATUS = (
    "EXACT_FAITHFUL_PROPER_CARRIER_ACTION_AND_FINITE_RECHARTING_COCYCLE_"
    "ATTAINED__SOURCE_SELECTION_COFINAL_GLUING_AND_PHYSICAL_ACTION_OPEN"
)
PORT_MEASURE_SCHEMA = "oph.primitive-port-dual-normalized-measure.v1"
PORT_MEASURE_STATUS = (
    "QUOTIENT_VISIBLE_NORMALIZED_PORT_DUAL_MEASURE_ATTAINED__"
    "PHYSICAL_PIXEL_AND_HOP_IDENTITIES_OPEN"
)
FZ11_SCHEMA = "oph.fz11-conditional-3d-translation-bridge.v1"
FZ11_STATUS = (
    "CONDITIONAL_3D_SCALAR_TRANSLATION_ADAPTER_ATTAINED__"
    "CANONICAL_SOURCE_SELECTION_TIME_EVOLUTION_PHOTON_SECTOR_SCALE_FRAME_"
    "BOOST_AND_EXCLUSIVITY_OPEN"
)
FZ12_SCHEMA = "oph.seam_current_edge_prediction_candidate.v1"
FZ12_STATUS = (
    "EXACT_SOURCE_NATIVE_EDGE_RAY__PROSPECTIVE_PHYSICAL_BRANCH_UNARMED__"
    "PHYSICAL_PRODUCER_OPEN"
)

FZ12_RAW_SHA256 = (
    "sha256:0b8f0f7573f556ef0f47158fe07eca002c5b35790231d1ef2b75518057d12915"
)
FZ12_RECEIPT_SHA256 = (
    "sha256:3b1344dbce713545deba54b48e027a88faedac9b3a8776795b9abe33cafde7c3"
)
FZ12_SOURCE_RESULT_SHA256 = (
    "sha256:135527821a72cf2bffba2a10ced6c17bc7452150e60d6998b7f4b84b19c2ffc1"
)
FZ12_CANDIDATE_SHA256 = (
    "sha256:e3a20b7fade1731bd58ab6ae7bd24d61d92e99801b2210c791fec7f590ff49ef"
)
HOMOGENEOUS_ACTION_LEAN_SHA256 = (
    "sha256:1fa16cbd466e85c5533b2a373b02b5d804efa703db62e8fd0e1cf8b34fcbc265"
)

Q5 = tuple[Fraction, Fraction]
ZERO: Q5 = (Fraction(), Fraction())
ONE: Q5 = (Fraction(1), Fraction())
SEAM_NORM_SQUARED: Q5 = (Fraction(2), Fraction(-2, 5))
RAW_PORT_RADIUS_SQUARED: Q5 = (Fraction(5, 2), Fraction(1, 2))
RAW_SEAM_NORM_SQUARED: Q5 = (Fraction(4), Fraction())

CLAIM_BOUNDARY = (
    "The exact D6 unit-current seam action and the exact response-selected "
    "Gram metric are the same internal object within the pinned finite "
    "ancestry. The parent completion remains conditional on source selection "
    "of the signed record as position and of the A2 response topology. Their "
    "composition "
    "fixes the full seam event's squared dimensionless length to "
    "2-2/sqrt(5), strictly between one and six fifths in normalized port "
    "units. The raw moment-chart value four is related by the common raw port "
    "radius 2+phi and is not itself the normalized response-Gram norm. This "
    "closes a source-native dimensionless scale for the internal FZ-12 action. "
    "It does not make the internal record completion physical position. A "
    "common positive conversion from response units to physical length "
    "preserves every pinned finite relation and has no source-selected lower "
    "bound, so no dimensionful carrier length, physical kappa, experimental "
    "amplitude, prediction promotion, or comparison permission follows."
)


class SameMetricScaleError(RuntimeError):
    """Raised when the same-action scale packet fails closed."""


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


def _external_fz12_sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value) + b"\n").hexdigest()


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
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise SameMetricScaleError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> None:
    raise SameMetricScaleError(f"non-finite JSON constant is forbidden: {value}")


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SameMetricScaleError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise SameMetricScaleError(f"{path} is not a JSON object")
    return value


def _validated_parent(path: Path, schema: str, status: str) -> dict[str, Any]:
    receipt = load_json_strict(path)
    payload = copy.deepcopy(receipt)
    digest = payload.pop("receipt_sha256", None)
    if (
        digest != _sha(payload)
        or receipt.get("schema") != schema
        or receipt.get("status") != status
        or receipt.get("comparison_data_read") is not False
    ):
        raise SameMetricScaleError(f"parent contract drifted: {path}")
    return receipt


def _parse_q5(text: str) -> Q5:
    suffix = "*sqrt5"
    if not isinstance(text, str) or not text.endswith(suffix):
        raise SameMetricScaleError("invalid Q(sqrt(5)) encoding")
    body = text[: -len(suffix)]
    split = body.find("+", 1)
    if split < 1:
        raise SameMetricScaleError("invalid Q(sqrt(5)) split")
    return Fraction(body[:split]), Fraction(body[split + 1 :])


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _qtext(value: Q5) -> str:
    return f"{_fraction_text(value[0])}+{_fraction_text(value[1])}*sqrt5"


def _qadd(left: Q5, right: Q5) -> Q5:
    return left[0] + right[0], left[1] + right[1]


def _qmul(left: Q5, right: Q5) -> Q5:
    return (
        left[0] * right[0] + 5 * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _qscale(value: Q5, scalar: Fraction) -> Q5:
    return value[0] * scalar, value[1] * scalar


def _quadratic_form(
    vector: Sequence[int], gram: Sequence[Sequence[Q5]]
) -> Q5:
    result = ZERO
    for left in range(len(vector)):
        for right in range(len(vector)):
            result = _qadd(
                result,
                _qscale(gram[left][right], Fraction(vector[left] * vector[right])),
            )
    return result


def _validate_fz12_if_available() -> dict[str, Any]:
    pin = {
        "repository": "FloatingPragma/observer-patch-holography",
        "path": (
            "code/a5_fingerprint/runtime/"
            "seam_current_edge_prediction_receipt.json"
        ),
        "schema": FZ12_SCHEMA,
        "status": FZ12_STATUS,
        "raw_sha256": FZ12_RAW_SHA256,
        "receipt_sha256": FZ12_RECEIPT_SHA256,
        "exact_source_result_sha256": FZ12_SOURCE_RESULT_SHA256,
        "conditional_physical_candidate_sha256": FZ12_CANDIDATE_SHA256,
        "comparison_permitted": False,
    }
    if FZ12_RECEIPT.exists():
        if _raw_sha(FZ12_RECEIPT) != FZ12_RAW_SHA256:
            raise SameMetricScaleError("frozen FZ-12 raw pin drifted")
        receipt = load_json_strict(FZ12_RECEIPT)
        if (
            receipt.get("schema") != FZ12_SCHEMA
            or receipt.get("status") != FZ12_STATUS
            or receipt.get("issue") != 666
            or receipt.get("receipt_sha256") != FZ12_RECEIPT_SHA256
            or _external_fz12_sha(receipt.get("exact_source_result"))
            != FZ12_SOURCE_RESULT_SHA256
            or _external_fz12_sha(receipt.get("conditional_physical_candidate"))
            != FZ12_CANDIDATE_SHA256
            or receipt.get("exact_source_result", {})
            .get("finite_geometry_replay", {})
            .get("seam_difference_norm_squared")
            != "4+0*sqrt5"
            or receipt.get("exposure_and_custody_boundary", {}).get(
                "comparison_permitted"
            )
            is not False
        ):
            raise SameMetricScaleError("frozen FZ-12 contract drifted")
    return pin


def _validate_homogeneous_action_if_available() -> dict[str, Any]:
    pin = {
        "repository": "FloatingPragma/observer-patch-holography",
        "path": "Lean/Screen/SeamCurrentHomogeneousAction.lean",
        "raw_sha256": HOMOGENEOUS_ACTION_LEAN_SHA256,
        "required_theorems": [
            "d6Translate_isometry",
            "directedSeamStep_canonical_chart",
            "unitCarrierSeamDirection_norm_sq",
            "normalized_average_generator_eq_edgeCurrentGenerator",
        ],
    }
    if HOMOGENEOUS_ACTION_LEAN.exists():
        raw = HOMOGENEOUS_ACTION_LEAN.read_bytes()
        if "sha256:" + hashlib.sha256(raw).hexdigest() != HOMOGENEOUS_ACTION_LEAN_SHA256:
            raise SameMetricScaleError("homogeneous-action Lean pin drifted")
        text = raw.decode("utf-8")
        if any(name not in text for name in pin["required_theorems"]):
            raise SameMetricScaleError("homogeneous-action theorem list drifted")
        if re.search(r"(?m)^\s*(?:sorry|admit)(?:\s|$)", text):
            raise SameMetricScaleError("homogeneous-action Lean source has a placeholder")
    return pin


def produce_receipt() -> dict[str, Any]:
    load_quotient = _validated_parent(
        LOAD_QUOTIENT_RECEIPT, LOAD_QUOTIENT_SCHEMA, LOAD_QUOTIENT_STATUS
    )
    completion = _validated_parent(
        COMPLETION_RECEIPT, COMPLETION_SCHEMA, COMPLETION_STATUS
    )
    action = _validated_parent(ACTION_RECEIPT, ACTION_SCHEMA, ACTION_STATUS)
    measure = _validated_parent(
        PORT_MEASURE_RECEIPT, PORT_MEASURE_SCHEMA, PORT_MEASURE_STATUS
    )
    fz11 = _validated_parent(FZ11_RECEIPT, FZ11_SCHEMA, FZ11_STATUS)
    load_json_strict(CARRIER_MANIFEST)
    fz12_pin = _validate_fz12_if_available()
    homogeneous_action_pin = _validate_homogeneous_action_if_available()

    module = completion["exact_signed_module_completion"]
    gram = [[_parse_q5(value) for value in row] for row in module["gram6_qsqrt5"]]
    boundary = load_quotient["exact_seam_current_boundary"]
    seam_matrix = boundary["signed_seam_current_matrix_D_after_boundary"]
    if len(seam_matrix) != 6 or any(len(row) != 30 for row in seam_matrix):
        raise SameMetricScaleError("seam-current matrix shape drifted")
    seam_vectors = [
        [int(seam_matrix[row][column]) for row in range(6)]
        for column in range(30)
    ]
    norm_rows = [_quadratic_form(vector, gram) for vector in seam_vectors]
    if set(norm_rows) != {SEAM_NORM_SQUARED}:
        raise SameMetricScaleError("response-Gram seam norm is not uniform")
    if _qmul(SEAM_NORM_SQUARED, RAW_PORT_RADIUS_SQUARED) != RAW_SEAM_NORM_SQUARED:
        raise SameMetricScaleError("raw/response metric normalization drifted")
    unit_direction_response_norm = _qscale(SEAM_NORM_SQUARED, Fraction(1, 4))
    if unit_direction_response_norm != (Fraction(1, 2), Fraction(-1, 10)):
        raise SameMetricScaleError("FZ-12 unit-direction metric norm drifted")

    measure_exact = measure["exact_base_port_dual_measure"]
    measure_attainment = measure["attainment"]
    if (
        measure_exact.get("exact_normalized_measure_per_port") != "1/12"
        or measure_attainment.get("physical_P_pixel_is_primitive_port_sector")
        is not False
        or measure_attainment.get("support_areal_radius_is_issue_655_translation_hop")
        is not False
        or measure_attainment.get("terminal_physical_refinement_stage_selected")
        is not False
    ):
        raise SameMetricScaleError("port measure attachment boundary drifted")

    fz11_frozen = fz11["parent_pins"]["frozen_FZ11_prediction"]
    if (
        fz11_frozen.get("comparison_permitted") is not False
        or fz11["exact_port_frame_and_relabel"].get("common_raw_norm_squared")
        != _qtext(RAW_PORT_RADIUS_SQUARED)
    ):
        raise SameMetricScaleError("FZ-11 normalization or custody drifted")

    same_action_digest = _sha(
        {
            "D6_boundary": boundary,
            "response_Gram": module["gram6_qsqrt5"],
            "completion_action": action["exact_completion_action"],
            "homogeneous_action_Lean_sha256": HOMOGENEOUS_ACTION_LEAN_SHA256,
            "frozen_FZ12_candidate_sha256": FZ12_CANDIDATE_SHA256,
        }
    )
    q_times_three = _qscale(SEAM_NORM_SQUARED, Fraction(3))
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "issue": 664,
        "target_data_read": False,
        "comparison_data_read": False,
        "parent_pins": {
            "FZ11_immutable_prediction_via_bridge": {
                **_raw_pin(FZ11_RECEIPT),
                "schema": fz11["schema"],
                "status": fz11["status"],
                "receipt_sha256": fz11["receipt_sha256"],
                "frozen_prediction_pin": copy.deepcopy(fz11_frozen),
            },
            "FZ12_immutable_prediction": fz12_pin,
            "A1_carrier_definition": _raw_pin(CARRIER_MANIFEST),
            "D6_source_action": {
                **_raw_pin(LOAD_QUOTIENT_RECEIPT),
                "schema": load_quotient["schema"],
                "status": load_quotient["status"],
                "receipt_sha256": load_quotient["receipt_sha256"],
            },
            "response_Gram_completion": {
                **_raw_pin(COMPLETION_RECEIPT),
                "schema": completion["schema"],
                "status": completion["status"],
                "receipt_sha256": completion["receipt_sha256"],
            },
            "completion_isometry_action": {
                **_raw_pin(ACTION_RECEIPT),
                "schema": action["schema"],
                "status": action["status"],
                "receipt_sha256": action["receipt_sha256"],
            },
            "normalized_port_dual_measure": {
                **_raw_pin(PORT_MEASURE_RECEIPT),
                "schema": measure["schema"],
                "status": measure["status"],
                "receipt_sha256": measure["receipt_sha256"],
            },
            "homogeneous_action_Lean_source": homogeneous_action_pin,
        },
        "same_internal_action_binding": {
            "record_carrier": "D6={z in Z^6: sum(z_i) is even}",
            "action": "translation by one directed unit-current seam boundary",
            "metric": "unit-diagonal response-selected Gram pulled back to D6",
            "same_action_metric_digest": same_action_digest,
            "seam_count": 30,
            "directed_seam_count": 60,
            "all_seam_columns_lie_in_D6": True,
            "all_seam_columns_have_the_same_response_Gram_norm": True,
            "D6_translation_isometry_theorem_pinned": True,
            "source_action_and_metric_are_same_internal_object": True,
            "internal_record_completion_is_physical_position": False,
            "seam_event_is_physical_field_propagation": False,
        },
        "typed_objects": {
            "a_edge_internal": (
                "dimensionless length of one full unit-current seam event in "
                "the normalized response-Gram completion"
            ),
            "a_edge_physical": "dimension-one physical length; unselected",
            "a_cell": "dimension-two physical screen-cell area; unselected",
            "ell_star_squared": "dimension-two reference area; unselected",
            "kappa_edge": "dimensionless physical metric conversion; unselected",
            "internal_and_physical_a_edge_identified": False,
            "length_area_and_reference_area_kept_distinct": True,
        },
        "exact_dimensionless_scale": {
            "normalized_port_generator_norm_squared": "1",
            "full_unit_current_seam_norm_squared_qsqrt5": _qtext(
                SEAM_NORM_SQUARED
            ),
            "equivalent_expression": "2-2/sqrt(5)",
            "raw_port_radius_squared_qsqrt5": _qtext(RAW_PORT_RADIUS_SQUARED),
            "raw_seam_difference_norm_squared_qsqrt5": _qtext(
                RAW_SEAM_NORM_SQUARED
            ),
            "raw_to_response_identity": (
                "raw_seam_norm_squared = raw_port_radius_squared * "
                "response_Gram_seam_norm_squared"
            ),
            "raw_to_response_identity_exact": True,
            "half_seam_control_factor_relative_to_full_current": "1/2",
            "half_seam_control_pullback_Gram_norm_squared_qsqrt5": _qtext(
                unit_direction_response_norm
            ),
            "half_seam_control_is_unit_3d_direction": False,
            "FZ12_unit_3d_direction_norm_squared": "1",
            "source_native_a_edge_in_response_units": (
                "2/sqrt(5/2+sqrt(5)/2)"
            ),
            "same_action_derivation": (
                "d6Position(delta)=rawRadius^-1*rawDifference="
                "(2/rawRadius)*FZ12_unit_3d_direction"
            ),
            "source_event_equals_a_edge_times_FZ12_unit_direction": True,
            "source_native_a_edge_squared_qsqrt5": _qtext(SEAM_NORM_SQUARED),
            "strict_rational_bounds": {
                "lower": "1",
                "upper": "6/5",
                "statement": "1 < a_edge^2 < 6/5",
                "positive_root_statement": "1 < a_edge < sqrt(6/5)",
                "lower_proof": "sqrt(5)<5/2 because 5<25/4",
                "upper_proof": "2<sqrt(5) because 4<5",
            },
            "strictly_positive": True,
            "same_metric_vertex_a_squared": "1",
            "same_metric_edge_over_vertex_squared_ratio_qsqrt5": _qtext(
                SEAM_NORM_SQUARED
            ),
            "typed_outcome": "SOURCE_NATIVE_DIMENSIONLESS_CARRIER_SCALE_ATTAINED",
            "frozen_FZ12_physical_a_identified_with_internal_a_edge": False,
            "dimensionful_length_selected": False,
        },
        "conditional_physical_cell_attachment": {
            "normalized_port_dual_area": "1/12",
            "port_sector_is_P_defined_physical_cell": False,
            "support_areal_radius_is_vertex_action_length": False,
            "terminal_physical_refinement_stage_selected": False,
            "conditional_vertex_kappa_if_both_identities_hold": "3/pi",
            "conditional_edge_kappa_if_both_identities_hold": (
                f"({_qtext(q_times_three)})/pi"
            ),
            "conditional_edge_kappa_equivalent": "6*(1-1/sqrt(5))/pi",
            "conditional_relation": (
                "kappa_edge=(2-2/sqrt(5))*kappa_vertex"
            ),
            "conditional_C4": "-kappa_edge*P*ell_star^2/20",
            "conditional_B0": "kappa_edge^2*P^2*ell_star^4/840",
            "conditional_B6": "-kappa_edge^2*P^2*ell_star^4/12600",
            "conditional_a_edge_over_ell_star": "sqrt(kappa_edge*P)",
            "physical_kappa_edge_source_selected": False,
        },
        "dimensionful_rescaling_counterfamily": {
            "parameter": "lambda>0 physical-length units per response-metric unit",
            "vertex_physical_length": "lambda",
            "seam_physical_length": "lambda*sqrt(2-2/sqrt(5))",
            "rescaling": "lambda -> s*lambda for arbitrary s>0",
            "finite_carrier_records_preserved": True,
            "D6_action_and_proper_carrier_covariance_preserved": True,
            "normalized_response_Gram_and_dimensionless_ratio_preserved": True,
            "normalized_port_dual_measure_preserved": True,
            "FZ11_and_FZ12_scale_free_coefficient_rays_preserved": True,
            "physical_coefficients_scale": {
                "C4": "s^2*C4",
                "B0": "s^4*B0",
                "B6": "s^4*B6",
            },
            "arbitrarily_small_positive_physical_seam_lengths_survive": True,
            "positive_dimensionful_lower_bound_from_pinned_ancestry": False,
            "bounded_exhaustiveness": (
                "all source constraints in the pinned finite metric, action, "
                "measure, and frozen-ray ancestry are scale homogeneous"
            ),
            "future_source_attachment_can_break_counterfamily": True,
            "bounded_physical_result": (
                "COMMON_METRIC_RESCALING_COUNTERFAMILY_SURVIVES__"
                "FUTURE_ATTACHMENT_CAN_BREAK"
            ),
            "does_not_replace_primary_dimensionless_typed_outcome": True,
        },
        "attainment": {
            "source_native_dimensionless_seam_action_scale": True,
            "source_native_strict_dimensionless_seam_event_lower_bound": True,
            "raw_norm_four_is_response_Gram_norm_four": False,
            "same_internal_action_metric_binding": True,
            "source_native_SI_carrier_scale": False,
            "independently_calibrated_SI_carrier_scale": False,
            "physical_pixel_attachment": False,
            "physical_position_action_attachment": False,
            "physical_positive_lower_bound": False,
            "physical_prediction_promoted": False,
            "comparison_permitted": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "implementation_pins": [
            _raw_pin(PRODUCER_PATH),
            _raw_pin(VERIFIER_PATH),
            _raw_pin(TEST_PATH),
        ],
    }
    receipt["receipt_sha256"] = _sha(receipt)
    return receipt


def verify_receipt(receipt: Mapping[str, Any]) -> dict[str, bool]:
    expected = produce_receipt()
    if _canonical_bytes(receipt) != _canonical_bytes(expected):
        raise SameMetricScaleError("receipt differs from exact producer replay")
    return {
        "receipt": True,
        "same_internal_action_metric": True,
        "dimensionless_scale": True,
        "dimensionful_scale": False,
        "physical_lower_bound": False,
        "comparison_permitted": False,
    }


def write_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    receipt = produce_receipt()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        verify_receipt(load_json_strict(args.verify))
        print("SEAM_CURRENT_SAME_METRIC_SCALE_VALID")
        return
    write_receipt(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
