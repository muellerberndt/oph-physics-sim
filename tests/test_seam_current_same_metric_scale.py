from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from oph_fpe.dynamics import seam_current_same_metric_scale as producer
from oph_fpe.dynamics import (
    verify_seam_current_same_metric_scale_independent as independent,
)
from oph_fpe.dynamics.verify_seam_current_same_metric_scale_independent import (
    IndependentScaleVerificationError,
    verify_receipt as verify_independent,
)


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT / "data/repair_closure/seam_current_same_metric_scale_receipt.json"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _rehash(receipt: dict) -> None:
    payload = copy.deepcopy(receipt)
    payload.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(payload)
    ).hexdigest()


def _write_mutation(tmp_path: Path, receipt: dict) -> Path:
    _rehash(receipt)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def canonical() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_receipt_replays_byte_exactly(canonical: dict) -> None:
    assert _canonical_bytes(producer.produce_receipt()) == _canonical_bytes(canonical)


def test_producer_verifier_passes(canonical: dict) -> None:
    result = producer.verify_receipt(canonical)
    assert result == {
        "receipt": True,
        "same_internal_action_metric": True,
        "dimensionless_scale": True,
        "dimensionful_scale": False,
        "physical_lower_bound": False,
        "comparison_permitted": False,
    }
    assert canonical["attainment"]["cross_repository_source_bytes_embedded"] is False
    assert canonical["attainment"]["cross_repository_source_custody_attested"] is False
    assert "absence does not fail" in canonical["claim_boundary"]


def test_independent_verifier_reconstructs_exact_scale() -> None:
    result = verify_independent(RECEIPT)
    assert result["receipt"] is True
    assert result["producer_imported"] is False
    assert result["checked_seam_columns"] == 30
    assert result["exact_response_Gram_norm_reconstructed"] is True
    assert result["raw_to_response_normalization_reconstructed"] is True
    assert result["same_action_digest_reconstructed"] is True
    assert result["dimensionless_scale_attained"] is True
    assert result["dimensionful_scale_attained"] is False
    assert result["physical_lower_bound_attained"] is False
    assert result["comparison_permitted"] is False


def test_exact_response_metric_result_is_not_raw_chart_norm(canonical: dict) -> None:
    scale = canonical["exact_dimensionless_scale"]
    assert scale["full_unit_current_seam_norm_squared_qsqrt5"] == (
        "2+-2/5*sqrt5"
    )
    assert scale["raw_seam_difference_norm_squared_qsqrt5"] == "4+0*sqrt5"
    assert scale["raw_to_response_identity_exact"] is True
    assert scale["half_seam_control_factor_relative_to_full_current"] == "1/2"
    assert scale["half_seam_control_pullback_Gram_norm_squared_qsqrt5"] == (
        "1/2+-1/10*sqrt5"
    )
    assert scale["half_seam_control_is_unit_3d_direction"] is False
    assert scale["FZ12_unit_3d_direction_norm_squared"] == "1"
    assert scale["source_native_a_edge_squared_qsqrt5"] == "2+-2/5*sqrt5"
    assert scale["strict_rational_bounds"]["statement"] == "1 < a_edge^2 < 6/5"
    assert scale["strict_rational_bounds"]["positive_root_statement"] == (
        "1 < a_edge < sqrt(6/5)"
    )
    assert scale["strict_rational_bounds"]["lower_proof"] == (
        "sqrt(5)<5/2 because 5<25/4"
    )
    assert scale["strict_rational_bounds"]["upper_proof"] == (
        "2<sqrt(5) because 4<5"
    )
    assert scale["typed_outcome"] == (
        "SOURCE_NATIVE_DIMENSIONLESS_CARRIER_SCALE_ATTAINED"
    )
    assert scale["frozen_FZ12_physical_a_identified_with_internal_a_edge"] is False
    assert scale["dimensionful_length_selected"] is False


def test_same_metric_branch_relation_is_exact(canonical: dict) -> None:
    typed = canonical["typed_objects"]
    scale = canonical["exact_dimensionless_scale"]
    conditional = canonical["conditional_physical_cell_attachment"]
    assert typed["internal_and_physical_a_edge_identified"] is False
    assert typed["length_area_and_reference_area_kept_distinct"] is True
    assert scale["same_metric_vertex_a_squared"] == "1"
    assert scale["same_metric_edge_over_vertex_squared_ratio_qsqrt5"] == (
        "2+-2/5*sqrt5"
    )
    assert conditional["conditional_relation"] == (
        "kappa_edge=(2-2/sqrt(5))*kappa_vertex"
    )
    assert conditional["conditional_edge_kappa_if_both_identities_hold"] == (
        "(6+-6/5*sqrt5)/pi"
    )
    assert conditional["physical_kappa_edge_source_selected"] is False


def test_dimensionful_rescaling_boundary_is_fail_closed(canonical: dict) -> None:
    counter = canonical["dimensionful_rescaling_counterfamily"]
    assert counter["arbitrarily_small_positive_physical_seam_lengths_survive"] is True
    assert counter["positive_dimensionful_lower_bound_from_pinned_ancestry"] is False
    assert counter["future_source_attachment_can_break_counterfamily"] is True
    assert counter["bounded_physical_result"] == (
        "COMMON_METRIC_RESCALING_COUNTERFAMILY_SURVIVES__"
        "FUTURE_ATTACHMENT_CAN_BREAK"
    )
    assert counter["does_not_replace_primary_dimensionless_typed_outcome"] is True
    attainment = canonical["attainment"]
    assert attainment["source_native_dimensionless_seam_action_scale"] is True
    assert attainment["physical_positive_lower_bound"] is False
    assert attainment["comparison_permitted"] is False


@pytest.mark.parametrize(
    ("parent", "field"),
    [
        ("D6_source_action", "sha256"),
        ("response_Gram_completion", "receipt_sha256"),
        ("completion_isometry_action", "sha256"),
        ("normalized_port_dual_measure", "receipt_sha256"),
    ],
)
def test_local_parent_pin_mutations_are_rejected(
    tmp_path: Path, canonical: dict, parent: str, field: str
) -> None:
    changed = copy.deepcopy(canonical)
    changed["parent_pins"][parent][field] = "sha256:" + "0" * 64
    with pytest.raises(IndependentScaleVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_fz12_pin_mutation_is_rejected(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["parent_pins"]["FZ12_immutable_prediction"][
        "conditional_physical_candidate_sha256"
    ] = "sha256:" + "1" * 64
    with pytest.raises(IndependentScaleVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_homogeneous_action_pin_mutation_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    changed = copy.deepcopy(canonical)
    changed["parent_pins"]["homogeneous_action_Lean_source"][
        "raw_sha256"
    ] = "sha256:" + "2" * 64
    with pytest.raises(IndependentScaleVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


@pytest.mark.parametrize(
    ("parent", "section", "field", "value"),
    [
        (
            "D6_source_action",
            "exact_integer_load_metric_quotient",
            "signed_module_derived_from_source_load_domain",
            False,
        ),
        (
            "D6_source_action",
            "exact_seam_current_boundary",
            "image_equals_pairwise_displacement_generated_D6",
            False,
        ),
        (
            "response_Gram_completion",
            "exact_signed_module_completion",
            "completion_translation_action_is_same_raw_action",
            False,
        ),
        (
            "completion_isometry_action",
            "exact_completion_action",
            "source_native_physical_action_promoted",
            True,
        ),
    ],
)
def test_parent_semantic_mutations_are_rejected(
    canonical: dict,
    parent: str,
    section: str,
    field: str,
    value: object,
) -> None:
    parents = independent._verify_local_pins(canonical)
    changed = copy.deepcopy(parents)
    changed[parent][section][field] = value
    with pytest.raises(IndependentScaleVerificationError):
        independent._reconstruct_scale(changed)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (
            "exact_dimensionless_scale",
            "full_unit_current_seam_norm_squared_qsqrt5",
            "4+0*sqrt5",
        ),
        (
            "exact_dimensionless_scale",
            "source_native_a_edge_squared_qsqrt5",
            "1+0*sqrt5",
        ),
        (
            "exact_dimensionless_scale",
            "dimensionful_length_selected",
            True,
        ),
        (
            "conditional_physical_cell_attachment",
            "physical_kappa_edge_source_selected",
            True,
        ),
        (
            "conditional_physical_cell_attachment",
            "conditional_C4",
            "-kappa_edge*P*ell_star^2/21",
        ),
        (
            "dimensionful_rescaling_counterfamily",
            "positive_dimensionful_lower_bound_from_pinned_ancestry",
            True,
        ),
        (
            "dimensionful_rescaling_counterfamily",
            "seam_physical_length",
            "lambda*sqrt(4)",
        ),
        (
            "same_internal_action_binding",
            "record_carrier",
            "Z^6",
        ),
        (
            "typed_objects",
            "a_edge_internal",
            "physical length",
        ),
        ("typed_objects", "internal_and_physical_a_edge_identified", True),
        ("typed_objects", "length_area_and_reference_area_kept_distinct", False),
        ("attainment", "physical_positive_lower_bound", True),
        ("attainment", "comparison_permitted", True),
    ],
)
def test_semantic_and_scale_mutations_are_rejected(
    tmp_path: Path,
    canonical: dict,
    section: str,
    field: str,
    value: object,
) -> None:
    changed = copy.deepcopy(canonical)
    changed[section][field] = value
    with pytest.raises(IndependentScaleVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_same_action_digest_mutation_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    changed = copy.deepcopy(canonical)
    changed["same_internal_action_binding"]["same_action_metric_digest"] = (
        "sha256:" + "3" * 64
    )
    with pytest.raises(IndependentScaleVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_claim_boundary_mutation_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    changed = copy.deepcopy(canonical)
    changed["claim_boundary"] = "dimensionful scale attained"
    with pytest.raises(IndependentScaleVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_top_level_data_firewall_mutations_are_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    for key in ("target_data_read", "comparison_data_read"):
        changed = copy.deepcopy(canonical)
        changed[key] = True
        with pytest.raises(IndependentScaleVerificationError):
            verify_independent(_write_mutation(tmp_path, changed))


def test_duplicate_json_key_is_rejected(tmp_path: Path, canonical: dict) -> None:
    rendered = json.dumps(canonical, sort_keys=True)
    duplicate = rendered[:-1] + ', "issue": 665}'
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(IndependentScaleVerificationError, match="duplicate JSON key"):
        verify_independent(path)


def test_nonfinite_json_is_rejected(tmp_path: Path, canonical: dict) -> None:
    rendered = json.dumps(canonical, sort_keys=True)
    changed = rendered.replace('"issue": 664', '"issue": NaN', 1)
    path = tmp_path / "nonfinite.json"
    path.write_text(changed, encoding="utf-8")
    with pytest.raises(IndependentScaleVerificationError, match="non-finite"):
        verify_independent(path)


def test_implementation_pin_omission_is_rejected(
    tmp_path: Path, canonical: dict
) -> None:
    changed = copy.deepcopy(canonical)
    changed["implementation_pins"].pop()
    with pytest.raises(IndependentScaleVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))
