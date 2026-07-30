from __future__ import annotations

import copy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oph_fpe.cosmology.angular_transfer_decision import (
    angular_transfer_decision_report,
    exact_equal_port_comb_moment,
    verify_angular_transfer_decision,
    write_angular_transfer_decision,
)


@pytest.fixture(scope="module")
def report() -> dict:
    return angular_transfer_decision_report(probe_count=192)


def test_equal_port_comb_moments_are_exact(report: dict) -> None:
    assert exact_equal_port_comb_moment(6) == Fraction(11, 25)
    assert exact_equal_port_comb_moment(10) == Fraction(247, 1875)
    assert exact_equal_port_comb_moment(12) == Fraction(1071, 3125)
    assert exact_equal_port_comb_moment(2) == 0
    assert exact_equal_port_comb_moment(4) == 0
    assert exact_equal_port_comb_moment(8) == 0
    assert exact_equal_port_comb_moment(7) == 0
    assert exact_equal_port_comb_moment(14) == 0

    geometry = report["geometry_checks"]
    comb = report["equal_port_comb"]
    assert geometry["ordered_pair_class_counts"] == {
        "-1": 12,
        "-1/sqrt(5)": 60,
        "1/sqrt(5)": 60,
        "1": 12,
    }
    assert geometry["regular_icosahedral_port_geometry"] is True
    assert comb["exact_moments"]["6"]["text"] == "11/25"
    assert comb["exact_moments"]["10"]["text"] == "247/1875"
    assert comb["exact_moments"]["12"]["text"] == "1071/3125"
    assert comb["exact_moments"]["14"]["text"] == "0"
    assert comb["normalized_degree_6_10_12_ray_relative_to_degree_6"] == {
        "6": {"denominator": 1, "numerator": 1, "text": "1"},
        "10": {
            "denominator": 825,
            "numerator": 247,
            "text": "247/825",
        },
        "12": {
            "denominator": 1375,
            "numerator": 1071,
            "text": "1071/1375",
        },
    }
    assert comb["maximum_exact_numeric_residual"] < 1.0e-12


def test_two_static_extensions_have_different_degree_six_content(
    report: dict,
) -> None:
    bandlimited = report["minimum_norm_bandlimited_interpolation"]
    comb = report["equal_port_comb"]

    assert bandlimited["evaluation_rank"] == 12
    assert bandlimited["constant_port_interpolation_residual"] < 1.0e-12
    assert bandlimited["constant_extension_nonconstant_coefficient_norm"] < 1.0e-12
    assert bandlimited["degree_6_moment"]["text"] == "0"
    assert comb["exact_moments"]["6"]["text"] == "11/25"


def test_smooth_normalized_counterfamily_is_a_same_codomain_witness(
    report: dict,
) -> None:
    counterfamily = report["smooth_same_codomain_counterfamily"]

    assert counterfamily["exact_h6_value_at_every_port"]["text"] == "11/25"
    assert counterfamily["exact_h10_value_at_every_port"]["text"] == "247/1875"
    assert counterfamily["maximum_counterfunction_port_residual"] < 1.0e-12
    assert counterfamily["sphere_mean_preserved_exactly"] is True
    assert counterfamily["A5_invariant_by_equal_orbit_sum"] is True
    assert (
        counterfamily["sufficient_positivity_bound_on_abs_epsilon"]["text"]
        == "2717/26800"
    )
    assert counterfamily["strictly_positive_subfamily_exists"] is True
    assert counterfamily["continuous_nontrivial_same_sample_family"] is True
    assert all(
        row["maximum_port_sample_residual_from_one"] < 1.0e-12
        for row in counterfamily["epsilon_rows"]
    )
    nonzero_rows = [
        row for row in counterfamily["epsilon_rows"] if row["epsilon"] != 0.0
    ]
    assert all(
        row["maximum_probe_residual_from_constant_extension"] > 1.0e-4
        for row in nonzero_rows
    )
    operators = counterfamily["linear_transfer_operator_family"]
    assert operators["linearity"] is True
    assert operators["A5_equivariance"] is True
    assert operators["same_sphere_mean_as_T0"] is True
    assert operators["checks_pass"] is True
    assert all(
        row["maximum_right_inverse_residual"] < 1.0e-12
        for row in operators["right_inverse_rows"]
    )
    assert (
        max(
            row["maximum_probe_difference_from_epsilon_zero"]
            for row in operators["right_inverse_rows"]
        )
        > 1.0e-4
    )


def test_exact_one_degree_control_exposes_variable_l6_statistic(
    report: dict,
) -> None:
    control = report["exact_one_degree_control_family"]

    assert control["sphere_measure_convention"] == (
        "dmu=dOmega/(4*pi), so integral_S2 dmu=1"
    )
    assert control["normalized_degree_6_statistic_definition"].startswith(
        "S6(f)=(1/13)*integral_S2"
    )
    assert control["exact_H6_value_at_every_port"]["text"] == "132/25"
    assert control["exact_normalized_sphere_integral_H6_squared"]["text"] == "1584/325"
    assert control["exact_degree_6_statistic_prefactor"]["text"] == "1584/4225"
    assert control["sufficient_positivity_bound_on_abs_epsilon"]["text"] == "25/432"
    assert control["checks_pass"] is True
    zero = next(row for row in control["epsilon_rows"] if row["epsilon"]["text"] == "0")
    positive = next(
        row for row in control["epsilon_rows"] if row["epsilon"]["text"] == "1/100"
    )
    assert zero["normalized_degree_6_statistic"]["text"] == "0"
    assert positive["normalized_degree_6_statistic"]["numerator"] > 0
    assert all(
        row["strictly_inside_sufficient_positivity_bound"] is True
        for row in control["epsilon_rows"]
    )


def test_a5_and_arbitrary_rotation_checks_pass(report: dict) -> None:
    rotations = report["equivariance_and_rotation_checks"]

    assert rotations["A5_rotation_count"] == 60
    assert rotations["operator_equivariance_input_basis_count"] == 12
    assert rotations["operator_equivariance_input_basis"] == (
        "all_12_canonical_port_basis_vectors"
    )
    assert rotations["checks_pass"] is True
    assert rotations["minimum_A5_rotation_determinant"] > 0.0
    assert rotations["maximum_A5_rotation_determinant_residual_from_one"] < 1.0e-12
    for key, value in rotations.items():
        if key.endswith("_residual"):
            assert value < 1.0e-10


def test_verdict_fails_closed_at_the_physical_transfer_boundary(
    report: dict,
) -> None:
    gates = report["decision_gates"]

    assert report["verdict"] == ("NONIDENTIFIABLE_WITHOUT_DYNAMICAL_TRANSFER_SELECTOR")
    assert report["strongest_allowed_claim"] == ("STATIC_TRANSFER_NONIDENTIFIABILITY")
    assert gates["instrument_valid"] is True
    assert (
        gates["static_port_geometry_and_equivariance_select_unique_transfer"] is False
    )
    assert gates["smooth_same_sample_counterfamily_constructed"] is True
    assert gates["dynamical_transfer_selector_supplied"] is False
    assert gates["screen_to_sky_identification_supplied"] is False
    assert gates["physical_angular_prediction_receipt"] is False
    assert report["source_inputs"]["external_observational_data_used"] is False
    assert report["source_inputs"]["target_values_used"] is False
    assert verify_angular_transfer_decision(report)["receipt"] is True


def test_verifier_rejects_mutation_and_physical_promotion(report: dict) -> None:
    mutated = copy.deepcopy(report)
    mutated["equal_port_comb"]["exact_moments"]["6"]["text"] = "12/25"
    verification = verify_angular_transfer_decision(mutated)
    assert verification["receipt"] is False
    assert "payload_hash_mismatch" in verification["reasons"]
    assert "independent_recomputation_mismatch" in verification["reasons"]

    promoted = copy.deepcopy(report)
    promoted["decision_gates"]["physical_angular_prediction_receipt"] = True
    verification = verify_angular_transfer_decision(promoted)
    assert verification["receipt"] is False
    assert "forbidden_uniqueness_or_physical_promotion" in verification["reasons"]


def test_verifier_fails_closed_on_nonfinite_malformed_or_unbounded_input(
    report: dict,
) -> None:
    nonfinite = copy.deepcopy(report)
    nonfinite["geometry_checks"]["maximum_dot_class_residual"] = float("nan")
    verification = verify_angular_transfer_decision(nonfinite)
    assert verification["receipt"] is False
    assert "payload_is_not_finite_canonical_json" in verification["reasons"]
    assert "independent_recomputation_mismatch" in verification["reasons"]

    malformed = {"schema": report["schema"], "source_inputs": []}
    verification = verify_angular_transfer_decision(malformed)
    assert verification["receipt"] is False
    assert "source_inputs_missing_or_not_mapping" in verification["reasons"]

    unbounded = copy.deepcopy(report)
    unbounded["source_inputs"]["probe_seed"] = 2**32
    unbounded["source_inputs"]["probe_count"] = 4097
    verification = verify_angular_transfer_decision(unbounded)
    assert verification["receipt"] is False
    assert "probe_seed_missing_or_out_of_bounds" in verification["reasons"]
    assert "probe_count_missing_or_out_of_bounds" in verification["reasons"]


def test_alternate_seed_is_bounded_reproducible_and_verified(report: dict) -> None:
    alternate = angular_transfer_decision_report(seed=644, probe_count=192)

    assert alternate["decision_gates"]["instrument_valid"] is True
    assert alternate["strongest_allowed_claim"] == (
        "STATIC_TRANSFER_NONIDENTIFIABILITY"
    )
    assert alternate["certificate_payload_sha256"] != (
        report["certificate_payload_sha256"]
    )
    assert verify_angular_transfer_decision(alternate)["receipt"] is True

    with pytest.raises(ValueError, match="seed must be an integer"):
        angular_transfer_decision_report(seed=-1, probe_count=192)
    with pytest.raises(ValueError, match="probe_count must be an integer"):
        angular_transfer_decision_report(seed=644, probe_count=4097)


def test_writer_and_module_entrypoint_emit_valid_json(
    tmp_path: Path,
) -> None:
    direct_path = tmp_path / "direct.json"
    direct_report = write_angular_transfer_decision(
        direct_path,
        probe_count=96,
    )
    assert json.loads(direct_path.read_text(encoding="utf-8")) == direct_report
    assert direct_path.read_bytes().endswith(b"\n")
    assert b"\r\n" not in direct_path.read_bytes()

    module_path = tmp_path / "module.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "oph_fpe.cosmology.angular_transfer_decision",
            "--output",
            str(module_path),
            "--probe-count",
            "96",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    module_report = json.loads(module_path.read_text(encoding="utf-8"))
    assert module_report["decision_gates"]["instrument_valid"] is True
    assert (
        module_report["decision_gates"]["physical_angular_prediction_receipt"] is False
    )
