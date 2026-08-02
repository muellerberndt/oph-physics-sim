from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oph_fpe.dynamics import port_load_metric_quotient as producer
from oph_fpe.dynamics.verify_port_load_metric_quotient_independent import (
    IndependentPortLoadQuotientError,
    verify_receipt as verify_independent,
)


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data/repair_closure/port_load_metric_quotient_receipt.json"


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


def _write_mutation(tmp_path: Path, report: dict, name: str) -> Path:
    _rehash(report)
    path = tmp_path / name
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def canonical() -> dict:
    return producer.load_json_strict(RECEIPT)


def test_canonical_receipt_replays_exactly(canonical: dict) -> None:
    assert _canonical_bytes(producer.produce_receipt()) == _canonical_bytes(canonical)
    assert producer.verify_receipt(canonical) == {
        "receipt": True,
        "integer_metric_quotient": True,
        "dense_displacement_completion": True,
        "mean_intertwiner": True,
        "pathwise_descent": False,
        "physical_position": False,
        "comparison_permitted": False,
    }


def test_integer_load_metric_quotient_derives_signed_module(canonical: dict) -> None:
    packet = canonical["exact_integer_load_metric_quotient"]
    assert packet["source_load_group_completion"] == "Grothendieck(N^12)=Z^12"
    assert packet["positive_port_basis"] == [0, 1, 4, 5, 8, 9]
    assert packet["difference_map_surjective"] is True
    assert packet["difference_after_section_is_identity"] is True
    assert packet["Gram_factorization"] == "G12=D^T*G6*D"
    assert packet["Gram_factorization_exact"] is True
    assert packet["integer_metric_kernel_exact"] is True
    assert packet["quotient_isomorphic_to_Z6"] is True
    assert packet["signed_module_independent_input_required"] is False
    assert packet["signed_module_derived_from_source_load_domain"] is True
    assert packet["real_Gram_radical_equals_integer_kernel_scalar_extension"] is False
    assert packet["real_Gram_radical_dimension"] == 9
    assert packet["integer_kernel_rank"] == 6
    assert packet["extra_real_radical_directions_have_nonzero_integer_points"] is False
    assert packet["quotient_completion_equals_parent_conditional_completion"] is True


def test_fixed_total_finite_sets_and_D6_displacements_are_distinguished(
    canonical: dict,
) -> None:
    packet = canonical["fixed_total_source_geometry"]
    rows = packet["exhaustive_source_rows"]
    assert [row["protected_total"] for row in rows] == list(range(5))
    assert [row["source_state_count"] for row in rows] == [1, 12, 78, 364, 1365]
    assert all(row["exact_nonnegative_image_formula_verified"] for row in rows)
    assert packet["one_fixed_total_nonnegative_image_is_finite"] is True
    assert packet["one_fixed_total_nonnegative_image_is_dense"] is False
    assert packet["positive_total_pairwise_displacement_generated_subgroup"] == (
        "D6={d in Z^6: sum d_i even}"
    )
    assert packet["raw_pairwise_displacement_set_at_fixed_total_is_finite"] is True
    assert packet["displacement_lattice_index_in_Z6"] == 2
    assert packet["displacement_lattice_Smith_invariants"] == [1, 1, 1, 1, 1, 2]
    assert packet["leading_unit_minor_and_full_determinant_witness"] == [
        1,
        1,
        1,
        1,
        1,
        2,
    ]
    assert packet["total_one_basis_extends_to_every_positive_total_by_common_filler"] is True
    assert packet["D6_contains_2Z6"] is True
    assert packet["D6_repair_Gram_image_dense_in_parent_completion"] is True
    assert packet["density_uses_displacement_group_completion_not_finite_state_set"] is True


def test_mean_descends_but_nonlinear_repair_kernel_does_not(canonical: dict) -> None:
    mean = canonical["exact_mean_quotient_intertwiner"]
    assert mean["source_operator"] == "T12=I-L_icosahedron/60"
    assert mean["identity"] == "D*T12=T6*D"
    assert mean["identity_exact"] is True
    assert mean["same_operator_as_parent_response_step"] is True
    assert mean["proper_carrier_equivariant"] is True
    assert mean["mean_readback_only"] is True
    assert mean["unitary_or_physical_propagation_identified"] is False

    counter = canonical["pathwise_descent_counterexample"]
    assert counter["protected_total"] == 2
    assert counter["same_input_quotient"] == [0] * 6
    assert counter["one_step_quotient_distributions_equal"] is False
    assert counter["quotient_expectations_equal"] is True
    assert counter["full_nonlinear_repair_kernel_descends"] is False
    assert counter["individual_seam_events_are_translation_generators"] is False
    assert counter["thirty_seam_attempts_identified_with_six_axis_steps"] is False
    assert counter["sixty_completion_labels_identified_with_twelve_translations"] is False


def test_thirty_seam_boundary_generates_D6_and_binds_edge30_control(
    canonical: dict,
) -> None:
    packet = canonical["exact_seam_current_boundary"]
    assert packet["undirected_seam_count"] == 30
    assert len(packet["source_edges"]) == 30
    assert len(packet["boundary_matrix_12_by_30"]) == 12
    assert len(packet["signed_seam_current_matrix_D_after_boundary"]) == 6
    assert packet["every_seam_current_has_even_coordinate_sum"] is True
    assert packet["rank"] == 6
    assert len(packet["index_two_minor_columns"]) == 6
    assert abs(packet["index_two_minor_determinant"]) == 2
    assert packet["image_equals_pairwise_displacement_generated_D6"] is True
    assert packet["Smith_invariants"] == [1, 1, 1, 1, 1, 2]
    assert packet["seam_boundary_columns_are_algebraic_load_currents"] is True
    assert packet["seam_boundary_columns_are_nonlinear_repair_updates"] is False
    assert packet["D_boundary_chart_equals_port_coordinate_difference_exact"] is True
    assert packet["thirty_seams_are_sixty_or_twelve_translation_events"] is False
    assert packet["edge30_control_axis_multiset_binding_exact"] is True
    assert packet["edge30_coordinate_chart_scale"] == "phi=(1+sqrt5)/2"
    assert packet["edge30_orientation_signs_selected"] is False
    assert packet["edge30_control_ray_physicalized"] is False
    assert packet["spatial_hop_source_certified"] is False


def test_source_physical_and_comparison_boundaries_remain_closed(
    canonical: dict,
) -> None:
    assert canonical["target_data_read"] is False
    assert canonical["comparison_data_read"] is False
    attainment = canonical["attainment"]
    assert attainment["source_integer_load_metric_quotient_to_signed_module_derived"] is True
    assert attainment["conditional_mean_descends_and_is_equivariant"] is True
    assert attainment["conditional_completion_action_available"] is True
    assert attainment["signed_module_remains_arbitrary_algebraic_control"] is False
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
        assert attainment[key] is False
    assert "A fixed-total state set itself is finite and is not dense." in canonical[
        "claim_boundary"
    ]
    assert "The full nonlinear repair kernel does not descend" in canonical[
        "claim_boundary"
    ]


def test_independent_verifier_reconstructs_all_layers() -> None:
    assert verify_independent(RECEIPT) == {
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


def test_independent_verifier_does_not_import_producer() -> None:
    path = (
        ROOT / "oph_fpe/dynamics/verify_port_load_metric_quotient_independent.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "oph_fpe.dynamics.port_load_metric_quotient" not in imported


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("exact_integer_load_metric_quotient", "integer_metric_kernel_exact", False),
        ("fixed_total_source_geometry", "one_fixed_total_nonnegative_image_is_dense", True),
        ("fixed_total_source_geometry", "displacement_lattice_index_in_Z6", 1),
        ("exact_seam_current_boundary", "edge30_control_ray_physicalized", True),
        ("exact_mean_quotient_intertwiner", "identity_exact", False),
        ("pathwise_descent_counterexample", "full_nonlinear_repair_kernel_descends", True),
        ("attainment", "operational_position_readback_selected", True),
        ("attainment", "comparison_permitted", True),
    ],
)
def test_independent_verifier_rejects_rehashed_semantic_mutations(
    tmp_path: Path,
    canonical: dict,
    section: str,
    key: str,
    value: object,
) -> None:
    report = copy.deepcopy(canonical)
    report[section][key] = value
    mutated = _write_mutation(tmp_path, report, f"{key}.json")
    with pytest.raises(IndependentPortLoadQuotientError):
        verify_independent(mutated)


def test_independent_verifier_rejects_changed_counterexample_state(
    tmp_path: Path, canonical: dict
) -> None:
    report = copy.deepcopy(canonical)
    report["pathwise_descent_counterexample"]["state_left"][0] = 1
    mutated = _write_mutation(tmp_path, report, "changed-state.json")
    with pytest.raises(IndependentPortLoadQuotientError):
        verify_independent(mutated)


def test_cli_verifiers_pass() -> None:
    environment = {"PYTHONDONTWRITEBYTECODE": "1"}
    replay = subprocess.run(
        [
            sys.executable,
            "-m",
            "oph_fpe.dynamics.port_load_metric_quotient",
            "--verify",
            str(RECEIPT),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PORT_LOAD_METRIC_QUOTIENT_VALID" in replay.stdout
    independent = subprocess.run(
        [
            sys.executable,
            "-m",
            "oph_fpe.dynamics.verify_port_load_metric_quotient_independent",
            str(RECEIPT),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PORT_LOAD_METRIC_QUOTIENT_INDEPENDENT_VALID" in independent.stdout
