from __future__ import annotations

import copy
import json

import pytest

from oph_fpe.gauge.kinetic_selector_sweep import (
    FINITE_SOURCE_SWEEP_RECEIPT,
    PHYSICAL_GAUGE_KINETIC_ACTION_RECEIPT,
    UNIQUE_SOURCE_RAY_RECEIPT,
    gauge_kinetic_selector_sweep,
    main,
    verify_gauge_kinetic_selector_sweep,
)


def test_frozen_source_grammar_supplies_a_constructive_nonuniqueness_result():
    report = gauge_kinetic_selector_sweep()

    assert report[FINITE_SOURCE_SWEEP_RECEIPT] is True
    assert report[UNIQUE_SOURCE_RAY_RECEIPT] is False
    assert report[PHYSICAL_GAUGE_KINETIC_ACTION_RECEIPT] is False
    assert report["status"] == (
        "FINITE_SOURCE_FILTERS_DO_NOT_SELECT_A_UNIQUE_PORT_RESPONSE_RAY"
    )
    assert report["result"]["constructive_counterexample_found"] is True
    assert report["result"]["physical_gauge_kinetic_ray_selected"] is False
    assert report["source_inputs"]["laboratory_data_used"] is False
    assert report["source_inputs"][
        "downstream_fit_or_target_used_to_construct_laws"
    ] is False
    assert verify_gauge_kinetic_selector_sweep(report)["receipt"] is True


def test_every_frozen_law_is_positive_covariant_and_carrier_local():
    report = gauge_kinetic_selector_sweep()
    carrier = report["carrier_audit"]
    assert carrier["vertex_count"] == 12
    assert carrier["edge_count"] == 30
    assert carrier["degree_sequence"] == [5] * 12
    assert carrier["diameter"] == 3
    assert carrier["spectral_band_ranks_constant_low_five_high"] == [1, 3, 5, 3]
    assert carrier["a5_base_rotation_count"] == 60
    assert carrier["a5_integer_permutation_group_closed"] is True
    assert carrier["a5_integer_permutation_inverses_present"] is True
    assert carrier["a5_faithful_base_vertex_action"] is True
    assert carrier["a5_rotation_group_order_60_receipt"] is True
    assert carrier["reference_a5_geometry_receipt"] is True
    assert report["spectral_bands"]["ranks"] == {
        "constant_singlet": 1,
        "lowest_positive_triplet": 3,
        "five_band": 5,
        "highest_triplet": 3,
    }
    assert report["spectral_bands"]["projector_audit"][
        "complete_and_orthogonal"
    ]

    for row in report["response_laws"]:
        assert row["positive_definite"]
        assert row["a5_covariance_passes"]
        assert row["carrier_locality"]["within_one_carrier"]
        assert row["sector_injection_audit"]["all_band_responses_recovered"]


def test_two_target_free_ward_admissible_laws_give_distinct_rays():
    report = gauge_kinetic_selector_sweep()
    by_name = {row["name"]: row for row in report["response_laws"]}
    for assignment in (
        "weak_is_lowest_positive_triplet",
        "weak_is_highest_triplet",
    ):
        consensus = by_name["consensus_mode_penalty"]["gauge_block_audits"][
            assignment
        ]
        disagreement = by_name["disagreement_mode_penalty"]["gauge_block_audits"][
            assignment
        ]
        assert consensus["finite_sector_ward_proxy_passes"]
        assert disagreement["finite_sector_ward_proxy_passes"]
        assert consensus["common_scale_quotient_u1_su2_su3"] == [2.0, 1.0, 1.0]
        assert disagreement["common_scale_quotient_u1_su2_su3"] == [
            0.5,
            1.0,
            1.0,
        ]

    for row in report["opposite_triplet_assignment_audits"]:
        assert row["finite_ward_admissible_ray_count"] >= 3
        assert row["unique_after_common_scale_quotient"] is False
        assert row["finite_two_step_proxy_ray_count"] >= 2
        assert row["unique_two_step_proxy_ray"] is False


def test_nearest_edge_laws_expose_the_su3_block_isotropy_requirement():
    report = gauge_kinetic_selector_sweep()
    by_name = {row["name"]: row for row in report["response_laws"]}

    assert by_name["edge_repair"]["carrier_locality"]["nearest_edge_only"]
    for name in (
        "edge_repair",
        "double_edge_repair",
        "squared_edge_repair",
        "edge_and_squared_repair",
    ):
        assert not any(
            audit["finite_sector_ward_proxy_passes"]
            for audit in by_name[name]["gauge_block_audits"].values()
        )
        assert all(
            audit["continuum_ward_identity_receipt"] is False
            for audit in by_name[name]["gauge_block_audits"].values()
        )

    strict = report["port_response_nonuniqueness"]["strict_nearest_edge"]
    assert [row["law"] for row in strict["laws"]] == [
        "onsite_unit",
        "edge_repair",
        "double_edge_repair",
    ]
    assert strict["distinct_ray_count"] == 3
    assert strict["unique_after_common_scale_quotient"] is False
    nontrivial_edge = report["port_response_nonuniqueness"][
        "strict_nearest_edge_nontrivial"
    ]
    assert [row["law"] for row in nontrivial_edge["laws"]] == [
        "edge_repair",
        "double_edge_repair",
    ]
    assert nontrivial_edge["distinct_ray_count"] == 2
    assert nontrivial_edge["unique_after_common_scale_quotient"] is False

    low_equalizer = by_name["lowest_triplet_five_equalizer"]
    double_low_equalizer = by_name["double_lowest_triplet_five_equalizer"]
    high_equalizer = by_name["highest_triplet_five_equalizer"]
    double_high_equalizer = by_name["double_highest_triplet_five_equalizer"]
    for equalizer in (
        low_equalizer,
        double_low_equalizer,
        high_equalizer,
        double_high_equalizer,
    ):
        assert equalizer["carrier_locality"]["graph_support_radius"] == 2
        assert equalizer["input_grammar"].startswith("empirical_target_free_")
    assert low_equalizer["gauge_block_audits"][
        "weak_is_highest_triplet"
    ]["finite_sector_ward_proxy_passes"]
    assert double_low_equalizer["gauge_block_audits"][
        "weak_is_highest_triplet"
    ]["finite_sector_ward_proxy_passes"]
    assert high_equalizer["gauge_block_audits"][
        "weak_is_lowest_positive_triplet"
    ]["finite_sector_ward_proxy_passes"]
    assert double_high_equalizer["gauge_block_audits"][
        "weak_is_lowest_positive_triplet"
    ]["finite_sector_ward_proxy_passes"]

    for row in report["opposite_triplet_assignment_audits"]:
        witnesses = row["nontrivial_two_step_proxy_laws"]
        assert len(witnesses) == 2
        assert all(witness["graph_support_radius"] == 2 for witness in witnesses)
        assert all(witness["law"] != "onsite_unit" for witness in witnesses)
        assert row["nontrivial_two_step_proxy_ray_count"] == 2
        assert row["unique_nontrivial_two_step_proxy_ray"] is False


def test_static_trace_ray_is_exact_and_does_not_promote_the_selector():
    report = gauge_kinetic_selector_sweep()
    static = report["static_hilbert_schmidt_audits"]
    trace = static["conditional_one_generation_representation_trace"]

    assert trace["trace_indices_u1_su2_su3"] == ["10/3", "2", "2"]
    assert trace["su2_normalized_ray_u1_su2_su3"] == ["5/3", "1", "1"]
    assert trace["expected_exact_ray_recovered"] is True
    assert trace["used_to_construct_or_select_response_laws"] is False
    assert trace["physical_kinetic_selector"] is False


def test_verifier_rejects_payload_and_promotion_mutations():
    report = gauge_kinetic_selector_sweep()

    mutated = copy.deepcopy(report)
    mutated["result"]["unique_finite_port_response_ray_selected"] = True
    verification = verify_gauge_kinetic_selector_sweep(mutated)
    assert verification["receipt"] is False
    assert "payload_hash_mismatch" in verification["reasons"]
    assert "independent_recomputation_mismatch" in verification["reasons"]

    promoted = copy.deepcopy(report)
    promoted[PHYSICAL_GAUGE_KINETIC_ACTION_RECEIPT] = True
    verification = verify_gauge_kinetic_selector_sweep(promoted)
    assert verification["receipt"] is False
    assert "forbidden_physical_or_uniqueness_promotion" in verification["reasons"]


def test_verifier_fails_closed_on_nonfinite_malformed_or_loose_tolerance():
    report = gauge_kinetic_selector_sweep()

    nonfinite = copy.deepcopy(report)
    nonfinite["response_laws"][0]["minimum_eigenvalue"] = float("nan")
    verification = verify_gauge_kinetic_selector_sweep(nonfinite)
    assert verification["receipt"] is False
    assert "payload_is_not_finite_canonical_json" in verification["reasons"]
    assert "independent_recomputation_mismatch" in verification["reasons"]

    malformed = {"schema": report["schema"], "tolerance": []}
    verification = verify_gauge_kinetic_selector_sweep(malformed)
    assert verification["receipt"] is False
    assert "tolerance_missing_or_not_numeric" in verification["reasons"]

    loose = copy.deepcopy(report)
    loose["tolerance"] = 1.0e-7
    verification = verify_gauge_kinetic_selector_sweep(loose)
    assert verification["receipt"] is False
    assert "tolerance_not_finite_or_out_of_bounds" in verification["reasons"]

    with pytest.raises(ValueError, match="tolerance must be finite"):
        gauge_kinetic_selector_sweep(tolerance=1.0e-7)
    with pytest.raises(ValueError, match="tolerance must be finite"):
        gauge_kinetic_selector_sweep(tolerance=float("nan"))


def test_equalizer_and_hash_boundaries_are_explicit():
    report = gauge_kinetic_selector_sweep()

    assert report["source_inputs"][
        "empirical_targets_used_to_engineer_equalizer_laws"
    ] is False
    assert "engineered algebraically" in report["claim_boundary"]
    assert "not_cross_platform_identity" in report["response_laws"][0][
        "matrix_sha256_scope"
    ]
    assert "Neither hash asserts cross-platform byte identity" in (
        report["hash_boundary"]
    )


def test_module_entrypoint_writes_a_verified_report(tmp_path):
    output = tmp_path / "gauge_kinetic_selector_sweep.json"
    assert main(["--output", str(output)]) == 0
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded[FINITE_SOURCE_SWEEP_RECEIPT] is True
    assert verify_gauge_kinetic_selector_sweep(loaded)["receipt"] is True
