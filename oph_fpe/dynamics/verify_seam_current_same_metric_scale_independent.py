"""Independent verifier for the source-native seam-action scale packet.

This verifier does not import the producer.  It reloads every local parent,
reconstructs the thirty response-Gram quadratic forms, checks the raw-chart
normalization independently, and enforces the physical-scale boundary.
"""

from __future__ import annotations

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
LOCAL_PARENT_PATHS = {
    "FZ11_immutable_prediction_via_bridge": (
        ROOT / "data/repair_closure/fz11_3d_translation_bridge_receipt.json"
    ),
    "A1_carrier_definition": (
        ROOT / "tests/fixtures/echosahedral_federation_reference.json"
    ),
    "D6_source_action": (
        ROOT / "data/repair_closure/port_load_metric_quotient_receipt.json"
    ),
    "response_Gram_completion": (
        ROOT / "data/repair_closure/port_gram_completion_bridge_receipt.json"
    ),
    "completion_isometry_action": (
        ROOT / "data/repair_closure/port_gram_equivariant_action_receipt.json"
    ),
    "normalized_port_dual_measure": (
        ROOT / "data/repair_closure/primitive_port_dual_measure_receipt.json"
    ),
}
FZ12_PATH = (
    RER_ROOT
    / "code/a5_fingerprint/runtime/seam_current_edge_prediction_receipt.json"
)
HOMOGENEOUS_ACTION_PATH = (
    RER_ROOT / "Lean/Screen/SeamCurrentHomogeneousAction.lean"
)

SCHEMA = "oph.seam-current-same-metric-scale.v1"
STATUS = (
    "SOURCE_NATIVE_DIMENSIONLESS_SEAM_ACTION_SCALE_ATTAINED__"
    "PHYSICAL_UNIT_CELL_ATTACHMENT_AND_LOWER_BOUND_OPEN"
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
HOMOGENEOUS_ACTION_SHA256 = (
    "sha256:1fa16cbd466e85c5533b2a373b02b5d804efa703db62e8fd0e1cf8b34fcbc265"
)
REQUIRED_LEAN_THEOREMS = [
    "d6Translate_isometry",
    "directedSeamStep_canonical_chart",
    "unitCarrierSeamDirection_norm_sq",
    "normalized_average_generator_eq_edgeCurrentGenerator",
]

EXPECTED_CLAIM_BOUNDARY = (
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

Q5 = tuple[Fraction, Fraction]
ZERO: Q5 = (Fraction(), Fraction())
SEAM_NORM: Q5 = (Fraction(2), Fraction(-2, 5))
RAW_RADIUS: Q5 = (Fraction(5, 2), Fraction(1, 2))
RAW_SEAM_NORM: Q5 = (Fraction(4), Fraction())


class IndependentScaleVerificationError(RuntimeError):
    """Raised when the independent scale audit fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentScaleVerificationError(message)


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


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise IndependentScaleVerificationError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> None:
    raise IndependentScaleVerificationError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentScaleVerificationError(
            f"cannot load {path}: {error}"
        ) from error
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _parse_q5(text: str) -> Q5:
    suffix = "*sqrt5"
    _require(isinstance(text, str) and text.endswith(suffix), "invalid Q5 value")
    body = text[: -len(suffix)]
    split = body.find("+", 1)
    _require(split >= 1, "invalid Q5 split")
    return Fraction(body[:split]), Fraction(body[split + 1 :])


def _qadd(left: Q5, right: Q5) -> Q5:
    return left[0] + right[0], left[1] + right[1]


def _qmul(left: Q5, right: Q5) -> Q5:
    return (
        left[0] * right[0] + 5 * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _qscale(value: Q5, scalar: Fraction) -> Q5:
    return value[0] * scalar, value[1] * scalar


def _quadratic(vector: Sequence[int], gram: Sequence[Sequence[Q5]]) -> Q5:
    result = ZERO
    for left in range(6):
        for right in range(6):
            result = _qadd(
                result,
                _qscale(gram[left][right], Fraction(vector[left] * vector[right])),
            )
    return result


def _validate_self_digest(value: Mapping[str, Any], label: str) -> None:
    payload = copy.deepcopy(dict(value))
    digest = payload.pop("receipt_sha256", None)
    _require(digest == _sha(payload), f"{label} self-digest drifted")


def _verify_local_pins(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    pins = receipt.get("parent_pins")
    _require(isinstance(pins, Mapping), "parent pins absent")
    loaded: dict[str, dict[str, Any]] = {}
    for key, path in LOCAL_PARENT_PATHS.items():
        pin = pins.get(key)
        _require(isinstance(pin, Mapping), f"parent pin absent: {key}")
        _require(pin.get("sha256") == _raw_sha(path), f"raw parent drifted: {key}")
        _require(pin.get("bytes") == len(path.read_bytes()), f"parent size drifted: {key}")
        if key == "A1_carrier_definition":
            _load(path)
            continue
        parent = _load(path)
        _validate_self_digest(parent, key)
        _require(pin.get("schema") == parent.get("schema"), f"schema pin drifted: {key}")
        _require(pin.get("status") == parent.get("status"), f"status pin drifted: {key}")
        _require(
            pin.get("receipt_sha256") == parent.get("receipt_sha256"),
            f"receipt pin drifted: {key}",
        )
        _require(
            parent.get("comparison_data_read") is False,
            f"comparison-exposed parent: {key}",
        )
        loaded[key] = parent
    return loaded


def _verify_external_pins(receipt: Mapping[str, Any]) -> None:
    pins = receipt["parent_pins"]
    fz12 = pins.get("FZ12_immutable_prediction")
    expected_fz12 = {
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
    _require(fz12 == expected_fz12, "FZ-12 immutable pin drifted")
    if FZ12_PATH.exists():
        _require(_raw_sha(FZ12_PATH) == FZ12_RAW_SHA256, "FZ-12 raw bytes drifted")
        external = _load(FZ12_PATH)
        _require(external.get("schema") == FZ12_SCHEMA, "FZ-12 schema drifted")
        _require(external.get("status") == FZ12_STATUS, "FZ-12 status drifted")
        _require(
            external.get("receipt_sha256") == FZ12_RECEIPT_SHA256,
            "FZ-12 receipt digest drifted",
        )
        _require(
            _external_fz12_sha(external.get("exact_source_result"))
            == FZ12_SOURCE_RESULT_SHA256,
            "FZ-12 exact source result drifted",
        )
        _require(
            _external_fz12_sha(external.get("conditional_physical_candidate"))
            == FZ12_CANDIDATE_SHA256,
            "FZ-12 conditional candidate drifted",
        )
        _require(
            external.get("exposure_and_custody_boundary", {}).get(
                "comparison_permitted"
            )
            is False,
            "FZ-12 comparison boundary opened",
        )

    lean = pins.get("homogeneous_action_Lean_source")
    expected_lean = {
        "repository": "FloatingPragma/observer-patch-holography",
        "path": "Lean/Screen/SeamCurrentHomogeneousAction.lean",
        "raw_sha256": HOMOGENEOUS_ACTION_SHA256,
        "required_theorems": REQUIRED_LEAN_THEOREMS,
    }
    _require(lean == expected_lean, "homogeneous-action Lean pin drifted")
    if HOMOGENEOUS_ACTION_PATH.exists():
        _require(
            _raw_sha(HOMOGENEOUS_ACTION_PATH) == HOMOGENEOUS_ACTION_SHA256,
            "homogeneous-action source bytes drifted",
        )
        text = HOMOGENEOUS_ACTION_PATH.read_text(encoding="utf-8")
        _require(
            all(name in text for name in REQUIRED_LEAN_THEOREMS),
            "homogeneous-action theorem declaration absent",
        )
        _require(
            re.search(r"(?m)^\s*(?:sorry|admit)(?:\s|$)", text) is None,
            "homogeneous-action source has a placeholder",
        )


def _reconstruct_scale(
    parents: Mapping[str, Mapping[str, Any]]
) -> tuple[Q5, str]:
    load = parents["D6_source_action"]
    completion = parents["response_Gram_completion"]
    action = parents["completion_isometry_action"]
    load_quotient = load["exact_integer_load_metric_quotient"]
    boundary = load["exact_seam_current_boundary"]
    completion_module = completion["exact_signed_module_completion"]
    completion_action = action["exact_completion_action"]
    required_load_quotient = {
        "signed_module_derived_from_source_load_domain": True,
        "signed_module_independent_input_required": False,
        "difference_map_surjective": True,
        "Gram_factorization_exact": True,
        "quotient_completion_equals_parent_conditional_completion": True,
    }
    for key, expected in required_load_quotient.items():
        _require(
            load_quotient.get(key) is expected,
            f"source-load quotient semantic drifted: {key}",
        )
    required_boundary = {
        "D_boundary_chart_equals_port_coordinate_difference_exact": True,
        "image_equals_pairwise_displacement_generated_D6": True,
        "seam_boundary_columns_are_algebraic_load_currents": True,
        "seam_boundary_columns_are_nonlinear_repair_updates": False,
        "edge30_control_axis_multiset_binding_exact": True,
        "spatial_hop_source_certified": False,
    }
    for key, expected in required_boundary.items():
        _require(
            boundary.get(key) is expected,
            f"seam-boundary semantic drifted: {key}",
        )
    required_completion = {
        "signed_cumulative_port_record_module": (
            "M_Z=Z[ports]/<e_antipode(p)+e_p> ~= Z^6"
        ),
        "full_Gram_descends_to_signed_record_quotient": True,
        "gram6_is_the_descended_positive_port_basis_form": True,
        "completion_translation_action_is_same_raw_action": True,
        "raw_addition_isometric": True,
        "continuous_carrier_is_primitive_input": False,
        "physical_continuous_field_selected": False,
        "overall_positive_metric_scale_selected": False,
        "carrier_position_readback_only": True,
    }
    for key, expected in required_completion.items():
        _require(
            completion_module.get(key) == expected,
            f"completion semantic drifted: {key}",
        )
    required_action = {
        "all_proper_maps_preserve_selected_Gram": True,
        "signed_action_composition_exact": True,
        "extension_is_conditional_on_parent_completion_premises": True,
        "source_native_physical_action_promoted": False,
    }
    for key, expected in required_action.items():
        _require(
            completion_action.get(key) is expected,
            f"completion-action semantic drifted: {key}",
        )
    matrix = boundary["signed_seam_current_matrix_D_after_boundary"]
    gram = [
        [_parse_q5(value) for value in row]
        for row in completion_module["gram6_qsqrt5"]
    ]
    _require(len(matrix) == 6, "D6 matrix row count drifted")
    vectors = [
        [int(matrix[row][column]) for row in range(6)] for column in range(30)
    ]
    norms = {_quadratic(vector, gram) for vector in vectors}
    _require(norms == {SEAM_NORM}, "independent seam norm reconstruction failed")
    _require(
        _qmul(SEAM_NORM, RAW_RADIUS) == RAW_SEAM_NORM,
        "independent raw/response identity failed",
    )
    _require(Fraction(5) < Fraction(25, 4), "lower-bound square proof failed")
    _require(Fraction(4) < Fraction(5), "upper-bound square proof failed")
    digest = _sha(
        {
            "D6_boundary": boundary,
            "response_Gram": completion_module["gram6_qsqrt5"],
            "completion_action": completion_action,
            "homogeneous_action_Lean_sha256": HOMOGENEOUS_ACTION_SHA256,
            "frozen_FZ12_candidate_sha256": FZ12_CANDIDATE_SHA256,
        }
    )
    return next(iter(norms)), digest


def _verify_payload(
    receipt: Mapping[str, Any], parents: Mapping[str, Mapping[str, Any]]
) -> None:
    norm, action_digest = _reconstruct_scale(parents)
    _require(norm == SEAM_NORM, "seam norm exact value drifted")
    same = receipt.get("same_internal_action_binding")
    _require(isinstance(same, Mapping), "same-action binding absent")
    _require(
        same.get("same_action_metric_digest") == action_digest,
        "same-action digest drifted",
    )
    expected_same_flags = {
        "all_seam_columns_lie_in_D6": True,
        "all_seam_columns_have_the_same_response_Gram_norm": True,
        "D6_translation_isometry_theorem_pinned": True,
        "source_action_and_metric_are_same_internal_object": True,
        "internal_record_completion_is_physical_position": False,
        "seam_event_is_physical_field_propagation": False,
    }
    for key, expected in expected_same_flags.items():
        _require(same.get(key) is expected, f"same-action flag drifted: {key}")
    expected_same_values = {
        "record_carrier": "D6={z in Z^6: sum(z_i) is even}",
        "action": "translation by one directed unit-current seam boundary",
        "metric": "unit-diagonal response-selected Gram pulled back to D6",
        "seam_count": 30,
        "directed_seam_count": 60,
    }
    for key, expected in expected_same_values.items():
        _require(same.get(key) == expected, f"same-action value drifted: {key}")

    typed = receipt.get("typed_objects")
    _require(isinstance(typed, Mapping), "typed object boundary absent")
    _require(
        typed.get("internal_and_physical_a_edge_identified") is False,
        "internal and physical seam scales were conflated",
    )
    _require(
        typed.get("length_area_and_reference_area_kept_distinct") is True,
        "length and area types were conflated",
    )
    expected_typed = {
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
    }
    _require(typed == expected_typed, "typed object boundary drifted")

    scale = receipt.get("exact_dimensionless_scale")
    _require(isinstance(scale, Mapping), "dimensionless scale absent")
    expected_scale = {
        "normalized_port_generator_norm_squared": "1",
        "full_unit_current_seam_norm_squared_qsqrt5": "2+-2/5*sqrt5",
        "equivalent_expression": "2-2/sqrt(5)",
        "raw_port_radius_squared_qsqrt5": "5/2+1/2*sqrt5",
        "raw_seam_difference_norm_squared_qsqrt5": "4+0*sqrt5",
        "raw_to_response_identity": (
            "raw_seam_norm_squared = raw_port_radius_squared * "
            "response_Gram_seam_norm_squared"
        ),
        "raw_to_response_identity_exact": True,
        "half_seam_control_factor_relative_to_full_current": "1/2",
        "half_seam_control_pullback_Gram_norm_squared_qsqrt5": (
            "1/2+-1/10*sqrt5"
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
        "source_native_a_edge_squared_qsqrt5": "2+-2/5*sqrt5",
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
        "same_metric_edge_over_vertex_squared_ratio_qsqrt5": (
            "2+-2/5*sqrt5"
        ),
        "typed_outcome": "SOURCE_NATIVE_DIMENSIONLESS_CARRIER_SCALE_ATTAINED",
        "frozen_FZ12_physical_a_identified_with_internal_a_edge": False,
        "dimensionful_length_selected": False,
    }
    _require(scale == expected_scale, "exact dimensionless scale packet drifted")

    conditional = receipt.get("conditional_physical_cell_attachment")
    _require(isinstance(conditional, Mapping), "conditional attachment absent")
    _require(
        conditional.get("conditional_edge_kappa_if_both_identities_hold")
        == "(6+-6/5*sqrt5)/pi",
        "conditional edge kappa drifted",
    )
    _require(
        conditional.get("conditional_relation")
        == "kappa_edge=(2-2/sqrt(5))*kappa_vertex",
        "branch scale relation drifted",
    )
    for key in (
        "port_sector_is_P_defined_physical_cell",
        "support_areal_radius_is_vertex_action_length",
        "terminal_physical_refinement_stage_selected",
        "physical_kappa_edge_source_selected",
    ):
        _require(conditional.get(key) is False, f"physical attachment promoted: {key}")
    expected_conditional = {
        "normalized_port_dual_area": "1/12",
        "port_sector_is_P_defined_physical_cell": False,
        "support_areal_radius_is_vertex_action_length": False,
        "terminal_physical_refinement_stage_selected": False,
        "conditional_vertex_kappa_if_both_identities_hold": "3/pi",
        "conditional_edge_kappa_if_both_identities_hold": (
            "(6+-6/5*sqrt5)/pi"
        ),
        "conditional_edge_kappa_equivalent": "6*(1-1/sqrt(5))/pi",
        "conditional_relation": "kappa_edge=(2-2/sqrt(5))*kappa_vertex",
        "conditional_C4": "-kappa_edge*P*ell_star^2/20",
        "conditional_B0": "kappa_edge^2*P^2*ell_star^4/840",
        "conditional_B6": "-kappa_edge^2*P^2*ell_star^4/12600",
        "conditional_a_edge_over_ell_star": "sqrt(kappa_edge*P)",
        "physical_kappa_edge_source_selected": False,
    }
    _require(conditional == expected_conditional, "conditional attachment drifted")

    counter = receipt.get("dimensionful_rescaling_counterfamily")
    _require(isinstance(counter, Mapping), "rescaling counterfamily absent")
    for key in (
        "finite_carrier_records_preserved",
        "D6_action_and_proper_carrier_covariance_preserved",
        "normalized_response_Gram_and_dimensionless_ratio_preserved",
        "normalized_port_dual_measure_preserved",
        "FZ11_and_FZ12_scale_free_coefficient_rays_preserved",
        "arbitrarily_small_positive_physical_seam_lengths_survive",
        "future_source_attachment_can_break_counterfamily",
    ):
        _require(counter.get(key) is True, f"counterfamily flag drifted: {key}")
    _require(
        counter.get("positive_dimensionful_lower_bound_from_pinned_ancestry")
        is False,
        "dimensionful lower bound was promoted",
    )
    _require(
        counter.get("bounded_physical_result")
        == (
            "COMMON_METRIC_RESCALING_COUNTERFAMILY_SURVIVES__"
            "FUTURE_ATTACHMENT_CAN_BREAK"
        ),
        "bounded physical counterfamily result drifted",
    )
    _require(
        counter.get("does_not_replace_primary_dimensionless_typed_outcome")
        is True,
        "counterfamily was promoted to a second typed outcome",
    )
    expected_counter_values = {
        "parameter": "lambda>0 physical-length units per response-metric unit",
        "vertex_physical_length": "lambda",
        "seam_physical_length": "lambda*sqrt(2-2/sqrt(5))",
        "rescaling": "lambda -> s*lambda for arbitrary s>0",
        "bounded_exhaustiveness": (
            "all source constraints in the pinned finite metric, action, "
            "measure, and frozen-ray ancestry are scale homogeneous"
        ),
        "bounded_physical_result": (
            "COMMON_METRIC_RESCALING_COUNTERFAMILY_SURVIVES__"
            "FUTURE_ATTACHMENT_CAN_BREAK"
        ),
    }
    for key, expected in expected_counter_values.items():
        _require(counter.get(key) == expected, f"counterfamily value drifted: {key}")
    _require(
        counter.get("physical_coefficients_scale")
        == {"C4": "s^2*C4", "B0": "s^4*B0", "B6": "s^4*B6"},
        "coefficient rescaling law drifted",
    )

    attainment = receipt.get("attainment")
    expected_attainment = {
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
    }
    _require(attainment == expected_attainment, "attainment packet drifted")
    _require(
        receipt.get("claim_boundary") == EXPECTED_CLAIM_BOUNDARY,
        "claim boundary drifted",
    )


def _verify_implementation_pins(receipt: Mapping[str, Any]) -> None:
    pins = receipt.get("implementation_pins")
    _require(isinstance(pins, list) and len(pins) == 3, "implementation pins drifted")
    expected_paths = [
        ROOT / "oph_fpe/dynamics/seam_current_same_metric_scale.py",
        Path(__file__).resolve(),
        ROOT / "tests/test_seam_current_same_metric_scale.py",
    ]
    for pin, path in zip(pins, expected_paths, strict=True):
        _require(pin.get("sha256") == _raw_sha(path), f"implementation drifted: {path}")
        _require(pin.get("bytes") == len(path.read_bytes()), f"implementation size drifted: {path}")


def verify_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    receipt = _load(path)
    _require(receipt.get("schema") == SCHEMA, "schema drifted")
    _require(receipt.get("status") == STATUS, "status drifted")
    _require(receipt.get("issue") == 664, "issue drifted")
    _require(receipt.get("target_data_read") is False, "target data boundary opened")
    _require(
        receipt.get("comparison_data_read") is False,
        "comparison data boundary opened",
    )
    _validate_self_digest(receipt, "scale receipt")
    parents = _verify_local_pins(receipt)
    _verify_external_pins(receipt)
    _verify_payload(receipt, parents)
    _verify_implementation_pins(receipt)
    return {
        "receipt": True,
        "producer_imported": False,
        "checked_seam_columns": 30,
        "exact_response_Gram_norm_reconstructed": True,
        "raw_to_response_normalization_reconstructed": True,
        "same_action_digest_reconstructed": True,
        "dimensionless_scale_attained": True,
        "dimensionful_scale_attained": False,
        "physical_lower_bound_attained": False,
        "comparison_permitted": False,
    }


def main() -> None:
    result = verify_receipt()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
