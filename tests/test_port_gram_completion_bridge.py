from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oph_fpe.dynamics import port_gram_completion_bridge as producer
from oph_fpe.dynamics.verify_port_gram_completion_bridge_independent import (
    IndependentVerificationError,
    verify_receipt as verify_independent,
)


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data/repair_closure/port_gram_completion_bridge_receipt.json"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _rehash(report: dict) -> None:
    payload = copy.deepcopy(report)
    payload.pop("receipt_sha256", None)
    report["receipt_sha256"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(payload)
    ).hexdigest()


def _write_mutation(tmp_path: Path, report: dict, name: str = "mutated.json") -> Path:
    _rehash(report)
    path = tmp_path / name
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path


def _set_path(report: dict, path: tuple[object, ...], value: object) -> None:
    cursor: object = report
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]


@pytest.fixture(scope="module")
def canonical() -> dict:
    return producer.load_receipt_strict(RECEIPT)


def test_canonical_receipt_replays_exactly(canonical: dict) -> None:
    assert _canonical_bytes(producer.produce_receipt()) == _canonical_bytes(canonical)
    replay = producer.verify_receipt(canonical)
    assert replay["receipt"] is True
    assert replay["exact_completion_implication"] is True
    assert replay["source_promotion"] is False
    assert replay["physical_promotion"] is False


def test_exact_repair_spectrum_selects_gram_dynamically(canonical: dict) -> None:
    packet = canonical["exact_repair_selected_gram"]
    incidence = packet["independent_repair_incidence"]
    assert incidence["origin"].startswith("pinned repair carrier incidence")
    assert len(incidence["source_edge_list"]) == 30
    assert sorted(incidence["fixture_to_source_port_map"]) == list(range(12))
    assert packet[
        "Gram_class_adjacency_matches_independent_repair_incidence"
    ] is True
    assert packet[
        "projector_constructed_from_independent_adjacency_polynomial"
    ] is True
    assert packet["selected_gram_normalization"] == "G=4*P_slowest_nonconstant"
    assert packet["unscaled_laplacian_band_costs"] == [
        "5+-1*sqrt5",
        "6+0*sqrt5",
        "5+1*sqrt5",
    ]
    assert packet["full_spectral_resolution"] == {
        "costs": ["0", "5+-1*sqrt5", "6+0*sqrt5", "5+1*sqrt5"],
        "ranks": [1, 3, 5, 3],
        "projectors_pairwise_orthogonal": True,
        "projectors_resolve_identity": True,
    }
    assert packet["slowest_band_selection_is_extra_economy_selector"] is False
    assert packet["positive_clock_rescaling_changes_selected_eigenspace"] is False
    assert packet["gram_branch_selected_by_declared_repair_cost_if_A1R_A2R_adopted"] is True
    assert packet["current_A1_selects_between_galois_frames"] is False
    assert packet["current_A1R_A2R_adopted"] is False


def test_discrete_repair_and_covariance_asymptotic_are_conditional(canonical: dict) -> None:
    packet = canonical["exact_repair_selected_gram"]
    discrete = packet["source_backed_discrete_repair"]
    assert discrete["one_step_expectation_operator"] == "T=I-L_ico/60"
    assert discrete["one_step_operator_source_backed_by_pinned_ancestry"] is True
    assert discrete["continuous_exponential_semigroup_used"] is False
    assert discrete["formal_operator_powers_equal_physical_n_tick_history"] is False
    assert discrete["IID_or_temporal_independence_proved"] is False
    assert discrete["full_temporal_grammar_completeness_proved"] is False
    assert discrete["physical_repair_law_promoted"] is False
    centered = packet["canonical_centered_response_kernel_derivation"]
    assert centered["probe_count"] == 12
    assert centered["stochastic_initial_ensemble_required"] is False
    assert centered["kernel_definition"] == "C_n=(T^n*Q)^T*(T^n*Q)=Q*T^(2n)*Q"
    assert "P_5" in centered["exact_spectral_formula"]
    assert centered["current_A2_contains_completed_asymptotic_kernel_readback"] is False
    assert centered["formal_response_powers_are_physical_time_evolution"] is False
    assert centered["trace_twelve_limit"].endswith("4*P_low=G")
    assert centered["limit_before_quotient_and_completion_required"] is True
    assert centered["finite_n_centered_rank"] == 11
    assert centered["finite_n_antipodally_odd_rank"] == 6
    assert centered[
        "strictly_positive_unequal_probe_weights_preserve_limit_rank_three"
    ] is True
    assert centered["unequal_weights_preserve_exact_icosahedral_Gram_angles"] is False
    assert centered["port_gram_derived_rather_than_supplied_by_A1_RG"] is True
    assert "not promoted" in packet["dynamical_selection_scope"]
    assert canonical["attainment"]["canonical_signed_port_record_source_selected"] is False

    carrier = packet["intrinsic_local_carrier"]
    assert carrier["definition"] == "H=range(P_low)"
    assert carrier["real_dimension"] == 3
    assert carrier["generator_gram_identity_exact"] is True
    assert carrier["cartesian_coordinates_used_to_define_carrier"] is False
    assert carrier["global_or_physical_space_promoted"] is False


def test_signed_module_completion_is_exactly_three_dimensional(canonical: dict) -> None:
    packet = canonical["exact_signed_module_completion"]
    assert packet["signed_cumulative_port_record_module"].endswith("~= Z^6")
    assert packet["positive_port_basis"] == [0, 1, 4, 5, 8, 9]
    assert packet["full_Gram_descends_to_signed_record_quotient"] is True
    assert packet["gram6_is_the_descended_positive_port_basis_form"] is True
    assert packet["positive_semidefinite"] is True
    assert packet["real_rank"] == 3
    assert packet["real_kernel_dimension"] == 3
    assert packet["integer_kernel_is_zero"] is True
    assert packet["integer_kernel_witness"]["determinant"] == "-8"
    assert packet["image_module"] == "finite-index-8 submodule of Z[phi]^3"
    assert packet["image_contains"] == "8*Z[phi]^3"
    assert packet["image_dense_in_real_three_space"] is True
    assert packet["single_event_generators_have_unit_gram_norm"] is True
    assert packet[
        "nonzero_integer_records_have_a_shortest_positive_gram_length"
    ] is False
    assert packet["atomic_generator_is_not_a_metric_minimum"] is True
    assert packet["real_quotient_dimension"] == 3
    assert packet["continuous_field_assumed"] is False
    assert packet["continuous_carrier_constructed_as_metric_completion"] is True
    assert packet["continuous_carrier_is_primitive_input"] is False
    assert packet["scalar_field_space_selected"] is False
    assert packet["physical_continuous_field_selected"] is False
    assert packet["raw_addition_isometric"] is True
    assert packet["completion_translation_action_is_same_raw_action"] is True
    assert packet["group_and_action_extension_formalized_in_Lean"] is False
    assert packet["ordered_history_to_position_quotient_proved"] is False
    assert packet["record_order_and_cost_retained_separately"] is True
    assert packet["carrier_position_readback_only"] is True
    assert packet["limit_before_quotient_and_completion_required"] is True
    assert packet["preferred_cartesian_frame_selected"] is False
    assert packet["local_carrier_only"] is True
    assert packet["faithful_A5_completion_action_formalized"] is False
    assert packet["overlap_refinement_gluing_proved"] is False
    assert packet["overall_positive_metric_scale_selected"] is False


def test_support_hop_identity_is_only_a_same_completion_implication(
    canonical: dict,
) -> None:
    packet = canonical["support_hop_isometry_implication"]
    assert packet["support_frame_gram_equals_selected_repair_gram"] is True
    assert packet["normalized_labeled_frame_isometry"] is True
    assert packet["dimensionful_support_radius_over_hop_selected"] is False
    assert packet["source_semantic_identity_required"] is True
    assert "not a shortest" in packet["conditional_hop_symbol_scope"]
    assert packet[
        "auxiliary_adapter_normalized_support_and_hop_directions_equal_by_definition"
    ] is True
    assert packet["dimensionful_support_and_hop_vectors_equal_by_definition"] is False
    assert packet["auxiliary_coordinate_equality_is_source_semantic_identity"] is False
    assert packet["support_and_hop_share_semantic_object_in_current_source"] is False
    assert packet["conditional_identity_requires_A2_cauchy_readback_clause"] is True
    assert packet["physical_pixel_identified"] is False
    assert packet["physical_areal_radius_selected"] is False


def test_minimal_clauses_and_three_countermodels_are_retained(canonical: dict) -> None:
    clauses = canonical["weakest_clause_strengthening"]
    assert clauses["proposed_label"] == (
        "A1-RG/A2-RC cumulative port-record completion clause"
    )
    assert len(clauses["A1_RG"]) == 3
    assert len(clauses["A2_RC"]) == 3
    assert clauses["overall_clock_or_length_unit_left_free"] is True
    controls = canonical["countermodel_controls"]
    assert set(controls["galois_branch_control"]["surviving_models"]) == {
        "G",
        "conj(G)",
    }
    assert controls["independent_rescaling_control"]["R_A_over_a_remains_free"] is True
    assert controls["finite_quotient_control"][
        "finite_endpoint_quotient_is_physical_translation_completion"
    ] is False
    assert controls["dense_hop_control"][
        "arbitrarily_small_nonzero_composite_translations_exist"
    ] is True
    assert controls["dense_hop_control"][
        "atomic_event_length_is_a_minimum_lattice_spacing"
    ] is False
    assert controls["completion_without_source_control"][
        "source_native_physical_action_follows_without_A1_RG_A2_RC"
    ] is False
    assert controls["response_kernel_controls"][
        "without_scale_normalization_raw_kernel_limit"
    ] == "0 on the centered subspace"
    assert controls["response_kernel_controls"][
        "equal_probe_counting_is_load_bearing_for_exact_icosahedral_angles"
    ] is True
    assert controls["response_kernel_controls"][
        "equal_probe_counting_is_load_bearing_for_dimension_three"
    ] is False
    assert controls["finite_n_completion_control"] == {
        "finite_n_centered_response_rank": 11,
        "finite_n_antipodally_odd_response_rank": 6,
        "three_dimensional_completion_before_normalized_limit": False,
        "normalized_infinite_response_limit_is_load_bearing": True,
    }


def test_four_certificate_layers_are_separated(canonical: dict) -> None:
    attainment = canonical["attainment"]
    assert attainment["exact_lowest_repair_band_selects_port_gram"] is True
    assert attainment["completion_is_three_dimensional_euclidean_vector_group"] is True
    assert attainment["support_hop_equal_gram_isometry_implication"] is True
    for key in (
        "canonical_signed_port_record_source_selected",
        "A1R_A2R_repair_amendment_adopted",
        "A2_cauchy_operational_completion_clause_present",
        "same_semantic_support_translation_object_emitted",
        "source_native_physical_translation_promoted",
        "physical_three_space_promoted",
        "faithful_A5_completion_action_formalized",
        "overlap_refinement_gluing_proved",
        "global_carrier_promoted",
        "physical_P_pixel_is_primitive_port_sector",
        "support_areal_radius_is_primitive_hop_promoted",
        "overall_physical_scale_selected",
        "physical_prediction_promoted",
        "comparison_permitted",
        "issue_662_armed",
    ):
        assert attainment[key] is False


def test_discrete_repair_ancestry_is_hash_pinned_and_scope_typed(
    canonical: dict,
) -> None:
    bounded = canonical["parent_pins"]["bounded_one_step_expectation_repair"]
    assert bounded["path"].endswith(
        "bounded_atomic_self_readback_closure_receipt.json"
    )
    assert bounded["certificate_payload_sha256"].startswith("sha256:")
    assert bounded["physical_repair_law_receipt"] is False
    assert bounded["canonical_A3_alone_implies_markovity"] is False
    bridge = canonical["parent_pins"]["port_repair_propagation_boundary"]
    assert bridge["path"].endswith("port_repair_propagation_bridge_receipt.json")
    assert bridge["receipt_sha256"].startswith("sha256:")
    assert bridge["one_step_operator"] == "T = I - L_icosahedron/60"
    assert bridge["spatial_port_hop_source_receipt"] is False
    assert bridge["same_operator_physical_readout_receipt"] is False


def test_independent_verifier_reconstructs_math_and_boundaries() -> None:
    result = verify_independent(RECEIPT)
    assert result == {
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


def test_independent_verifier_does_not_import_producer() -> None:
    path = ROOT / "oph_fpe/dynamics/verify_port_gram_completion_bridge_independent.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "oph_fpe.dynamics.port_gram_completion_bridge" not in imported


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("exact_repair_selected_gram", "selected_band"), "galois_high_band"),
        (("exact_repair_selected_gram", "full_gram_qsqrt5", 0, 1), "0+-1/5*sqrt5"),
        (("exact_repair_selected_gram", "strict_cost_order"), "5+sqrt5 < 6"),
        (("exact_repair_selected_gram", "independent_repair_incidence", "fixture_to_source_port_map", 0), 0),
        (("exact_repair_selected_gram", "independent_repair_incidence", "source_adjacency_sha256"), "sha256:" + "0" * 64),
        (("exact_repair_selected_gram", "Gram_class_adjacency_matches_independent_repair_incidence"), False),
        (("exact_repair_selected_gram", "projector_constructed_from_independent_adjacency_polynomial"), False),
        (("exact_repair_selected_gram", "intrinsic_local_carrier", "real_dimension"), 4),
        (("exact_repair_selected_gram", "slowest_band_selection_is_extra_economy_selector"), True),
        (("exact_repair_selected_gram", "positive_clock_rescaling_changes_selected_eigenspace"), True),
        (("exact_repair_selected_gram", "source_backed_discrete_repair", "formal_operator_powers_equal_physical_n_tick_history"), True),
        (("exact_repair_selected_gram", "source_backed_discrete_repair", "IID_or_temporal_independence_proved"), True),
        (("exact_repair_selected_gram", "canonical_centered_response_kernel_derivation", "current_A2_contains_completed_asymptotic_kernel_readback"), True),
        (("exact_repair_selected_gram", "canonical_centered_response_kernel_derivation", "formal_response_powers_are_physical_time_evolution"), True),
        (("exact_repair_selected_gram", "canonical_centered_response_kernel_derivation", "trace_twelve_limit"), "G"),
        (("exact_repair_selected_gram", "canonical_centered_response_kernel_derivation", "limit_before_quotient_and_completion_required"), False),
        (("exact_signed_module_completion", "gram6_qsqrt5", 0, 1), "0+0*sqrt5"),
        (("exact_signed_module_completion", "raw_generator_coordinates_qsqrt5", 0, 0), "0+0*sqrt5"),
        (("exact_signed_module_completion", "antipodal_relations", 0, 1), 2),
        (("exact_signed_module_completion", "full_Gram_descends_to_signed_record_quotient"), False),
        (("exact_signed_module_completion", "gram6_is_the_descended_positive_port_basis_form"), False),
        (("exact_signed_module_completion", "positive_semidefinite"), False),
        (("exact_signed_module_completion", "hausdorff_metric"), "target metric"),
        (("exact_signed_module_completion", "hausdorff_on_integer_records"), False),
        (("exact_signed_module_completion", "integer_kernel_witness", "determinant"), "-4"),
        (("exact_signed_module_completion", "image_dense_in_real_three_space"), False),
        (("exact_signed_module_completion", "completion_translation_action_is_same_raw_action"), False),
        (("exact_signed_module_completion", "continuous_carrier_is_primitive_input"), True),
        (("exact_signed_module_completion", "scalar_field_space_selected"), True),
        (("exact_signed_module_completion", "group_and_action_extension_formalized_in_Lean"), True),
        (("exact_signed_module_completion", "ordered_history_to_position_quotient_proved"), True),
        (("exact_signed_module_completion", "faithful_A5_completion_action_formalized"), True),
        (("exact_signed_module_completion", "overlap_refinement_gluing_proved"), True),
        (("support_hop_isometry_implication", "auxiliary_adapter_normalized_support_and_hop_directions_equal_by_definition"), False),
        (("support_hop_isometry_implication", "dimensionful_support_and_hop_vectors_equal_by_definition"), True),
        (("support_hop_isometry_implication", "auxiliary_coordinate_equality_is_source_semantic_identity"), True),
        (("support_hop_isometry_implication", "support_and_hop_share_semantic_object_in_current_source"), True),
        (("support_hop_isometry_implication", "dimensionful_support_radius_over_hop_selected"), True),
        (("support_hop_isometry_implication", "physical_areal_radius_selected"), True),
        (("countermodel_controls", "independent_rescaling_control", "R_A_over_a_remains_free"), False),
        (("attainment", "canonical_signed_port_record_source_selected"), True),
        (("attainment", "A2_cauchy_operational_completion_clause_present"), True),
        (("attainment", "source_native_physical_translation_promoted"), True),
        (("attainment", "issue_662_armed"), True),
        (("parent_pins", "bounded_one_step_expectation_repair", "canonical_A3_alone_implies_markovity"), True),
        (("parent_pins", "port_repair_propagation_boundary", "same_operator_physical_readout_receipt"), True),
    ],
)
def test_semantic_and_numeric_mutations_fail_closed(
    tmp_path: Path,
    canonical: dict,
    path: tuple[object, ...],
    value: object,
) -> None:
    changed = copy.deepcopy(canonical)
    _set_path(changed, path, value)
    mutated = _write_mutation(tmp_path, changed)
    with pytest.raises(IndependentVerificationError):
        verify_independent(mutated)
    with pytest.raises(producer.PortGramCompletionError):
        producer.load_receipt_strict(mutated)


def test_unregistered_target_field_fails_closed(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["observed_target"] = 1.0
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


@pytest.mark.parametrize(
    "section",
    [
        "exact_repair_selected_gram",
        "exact_signed_module_completion",
        "support_hop_isometry_implication",
        "weakest_clause_strengthening",
        "countermodel_controls",
        "attainment",
    ],
)
def test_unregistered_nested_field_fails_independent_schema(
    tmp_path: Path, canonical: dict, section: str
) -> None:
    changed = copy.deepcopy(canonical)
    changed[section]["hidden_target_result"] = True
    with pytest.raises(IndependentVerificationError, match="schema"):
        verify_independent(_write_mutation(tmp_path, changed))


def test_implementation_pin_removal_fails_closed(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["implementation_pins"].pop()
    with pytest.raises(IndependentVerificationError):
        verify_independent(_write_mutation(tmp_path, changed))


def test_duplicate_json_key_fails_both_strict_loaders(
    tmp_path: Path, canonical: dict
) -> None:
    rendered = json.dumps(canonical, sort_keys=True)
    path = tmp_path / "duplicate.json"
    path.write_text(rendered[:-1] + ', "schema": "duplicate"}', encoding="utf-8")
    with pytest.raises(IndependentVerificationError, match="duplicate JSON key"):
        verify_independent(path)
    with pytest.raises(producer.PortGramCompletionError, match="duplicate JSON key"):
        producer.load_receipt_strict(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_constants_fail_both_strict_loaders(
    tmp_path: Path,
    canonical: dict,
    constant: str,
) -> None:
    rendered = json.dumps(canonical, sort_keys=True)
    path = tmp_path / f"nonfinite-{constant.replace('-', 'minus')}.json"
    path.write_text(rendered[:-1] + f', "control": {constant}}}', encoding="utf-8")
    with pytest.raises(IndependentVerificationError, match="non-finite JSON constant"):
        verify_independent(path)
    with pytest.raises(producer.PortGramCompletionError, match="non-finite JSON constant"):
        producer.load_receipt_strict(path)


def test_both_cli_verifiers_pass() -> None:
    producer_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "oph_fpe.dynamics.port_gram_completion_bridge",
            "--validate-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PORT_GRAM_COMPLETION_BRIDGE_VALID" in producer_run.stdout
    independent_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "oph_fpe.dynamics.verify_port_gram_completion_bridge_independent",
            str(RECEIPT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PORT_GRAM_COMPLETION_BRIDGE_INDEPENDENT_PASS" in independent_run.stdout
