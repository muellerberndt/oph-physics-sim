"""Fail-closed tests for the constructive rank-3/FCC causet control."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oph_fpe.bulk.rank3_fcc_causet_compatibility import (
    DEFAULT_OUTPUT,
    _canonical_bytes,
    _sha,
    produce_rank3_fcc_causet_compatibility_receipt,
)
from oph_fpe.bulk.verify_rank3_fcc_causet_compatibility_independent import (
    IndependentRank3FCCVerificationError,
    verify_receipt,
)


@pytest.fixture(scope="module")
def receipt() -> dict:
    return produce_rank3_fcc_causet_compatibility_receipt()


def test_constructive_control_has_positive_bounded_metrics(receipt: dict) -> None:
    assert receipt["status"] == (
        "BOUNDED_RANK3_PLUS_TIME_ARCHITECTURE_COMPATIBILITY_FIXTURE_PASSED"
    )
    assert receipt["epistemic_status"] == ("POST_HOC_EXPLORATORY_CONSTRUCTIVE_CONTROL")
    assert receipt["CONSTRUCTIVE_ARCHITECTURE_COMPATIBILITY_RECEIPT"] is True
    assert all(receipt["compatibility_checks"].values())

    growth = receipt["carrier_ball_growth_control"]
    assert growth["all_rows_match_exact_polynomial"] is True
    assert growth["exact_polynomial_degree"] == 3
    assert [row["radius"] for row in growth["rows"]] == list(range(21))
    assert growth["rows"][-1]["carrier_ball_count"] == 28_741

    fine_growth = receipt["fine_diamond_growth_control"]
    assert fine_growth["all_oriented_seam_counts_exact"] is True
    assert fine_growth["all_fine_diamond_counts_exact"] is True
    assert fine_growth["frozen_family_counts_match_constructed_fine_dags"] is True
    assert fine_growth["exact_polynomial_degree_in_imposed_half_depth"] == 4
    assert fine_growth["fine_diamond_rows"][-1]["fine_event_count"] == 132_771
    assert fine_growth["poset_height_used_as_independent_variable"] is False
    assert fine_growth["physical_four_volume_claimed"] is False
    assert fine_growth["general_all_n_machine_proof_received"] is False

    family = receipt["nested_thinned_family"]
    assert family["fine_diamonds_are_nested"] is True
    assert family["retained_carriers_are_nested"] is True
    assert family["induced_orders_restrict_exactly"] is True
    final = family["level_summaries"][-1]
    assert final["depth"] == 20
    assert final["fine_event_count"] == 132_771
    assert 2_500 <= final["retained_event_count_minimum"]
    assert final["retained_event_count_maximum"] <= 2_800
    assert 0.08 <= final["ordering_fraction_minimum"]
    assert final["ordering_fraction_maximum"] <= 0.12
    assert 3.5 <= final["myrheim_meyer_candidate_minimum"]
    assert final["myrheim_meyer_candidate_maximum"] <= 4.5
    scaling = receipt["height_count_scaling_control"]
    assert 3.3 < scaling["log_count_vs_log_height_exponent"] < 3.5
    assert scaling["fourth_power_trend_demonstrated"] is False
    assert scaling["normalization_match_claimed"] is False

    profile = receipt["profile_comparison_interpretation"]
    assert profile["d8_to_d20_monotone_segment_used_as_convergence_evidence"] is False
    assert profile["finite_profile_similarity_received"] is False
    assert profile["depth20_constructive_to_control_spread_ratio"] > 30.0
    diagnostics = receipt["exploratory_diagnostics_not_pass_gates"]
    assert diagnostics["all_depth24_seed_profiles_worse_than_depth20"] is True
    extension = receipt["additional_out_of_family_extrapolation_control"]
    assert extension["independent_seed_holdout"] is False
    assert extension["depth_held_out_from_positive_family"] is True
    assert extension["all_seed_profile_rms_worse_than_depth20"] is True
    assert extension["convergence_evidence_received"] is False
    assert extension["runs"][0]["fine_event_count"] == 269_517
    assert [
        row["depth24_profile_rms_to_flat_4d_asymptotic"]
        for row in extension["runs"]
    ] == pytest.approx(
        [0.07264232589486724, 0.07122496502224235, 0.07471763767040634]
    )
    controls = receipt["matched_minkowski_3_plus_1_controls"]
    assert controls["poisson_cardinality_fluctuations_present"] is False
    assert "binomial" in controls["ensemble"]
    sensitivity = receipt["post_hoc_thinning_denominator_sensitivity"]
    assert sensitivity["denominators"] == [25, 40, 50, 60, 75, 100]
    assert sensitivity["myrheim_meyer_is_algebraic_reexpression_of_ordering_fraction"]
    assert sensitivity["profile_rms_not_constant_on_tested_grid"] is True
    assert sensitivity["robustness_beyond_tested_grid_claimed"] is False
    assert sensitivity["pass_gate"] is False


def test_order_is_versioned_and_raw_layering_is_a_negative(receipt: dict) -> None:
    formula = receipt["provenance_formula_control"]
    resources = receipt["versioned_resource_provenance_control"]
    assert formula["exact"] is True
    assert formula["mismatch_count"] == 0
    assert resources["exact"] is True
    assert resources["duplicate_writer_count"] == 0
    assert resources["parent_set_mismatch_count"] == 0
    assert resources["distinguished_boundary_root_read_count"] > 0

    raw = receipt["raw_layered_negative"]
    assert raw["classification"] == (
        "RAW_LAYERED_FINE_DAG_FAILS_4D_INTERVAL_ABUNDANCE_PROFILE"
    )
    assert raw["profile_rms_to_flat_4d_asymptotic"] > 1.0
    final_rms = receipt["nested_thinned_family"]["level_summaries"][-1][
        "profile_rms_to_flat_4d_asymptotic"
    ]
    assert final_rms < raw["profile_rms_to_flat_4d_asymptotic"]


def test_fcc_and_thinning_nonclaims_fail_closed(receipt: dict) -> None:
    assert receipt["gluing_definition"]["symmetry_group"] == "O_h_not_A5"
    assert receipt["gluing_definition"]["global_gluing_imposed"] is True
    calibration = receipt["count_density_calibration"]
    assert calibration["true_random_process_claimed"] is False
    assert calibration["bernoulli_physical_process_claimed"] is False
    assert calibration["poisson_physical_process_claimed"] is False
    assert receipt["held_out_confirmation_status"].startswith("NOT_RUN")
    assert receipt["statistical_significance_claimed"] is False
    for key in (
        "CURRENT_RANDOM_FEDERATION_SELECTS_THIS_GLUING_RECEIPT",
        "EXACT_A5_S2_DIRECTION_COMPATIBILITY_RECEIPT",
        "PHYSICAL_CAUSAL_SET_RECEIPT",
        "FINITE_CAUSAL_SET_LIKENESS_SIMILARITY_RECEIPT",
        "FAITHFUL_EMBEDDING_RECEIPT",
        "MANIFOLDLIKENESS_RECEIPT",
        "PHYSICAL_DIMENSION_3_PLUS_1_DERIVATION_RECEIPT",
        "FOURTH_POWER_HEIGHT_SCALING_RECEIPT",
        "MATCHED_FINITE_PROFILE_CONVERGENCE_RECEIPT",
        "PHYSICAL_VOLUME_CALIBRATION_RECEIPT",
        "LORENTZIAN_MANIFOLD_RECEIPT",
        "CONTINUUM_LIMIT_RECEIPT",
        "ARITHMETIC_MISMATCH_DESCENT_RECEIPT",
        "physical_promotion_allowed",
    ):
        assert receipt[key] is False, key


def test_receipt_is_deterministic_and_frozen(receipt: dict) -> None:
    produce_rank3_fcc_causet_compatibility_receipt.cache_clear()
    recomputed = produce_rank3_fcc_causet_compatibility_receipt()
    assert _canonical_bytes(recomputed) == _canonical_bytes(receipt)
    assert DEFAULT_OUTPUT.read_bytes() == _canonical_bytes(receipt)


def test_independent_verifier_replays_and_rejects_promotion(
    tmp_path: Path, receipt: dict
) -> None:
    verification = verify_receipt(DEFAULT_OUTPUT)
    assert verification["verified"] is True
    assert verification["independent_order_algorithm"].startswith(
        "closed_form_fcc_reachability"
    )

    mutations = (
        (("PHYSICAL_CAUSAL_SET_RECEIPT",), True),
        (("gluing_definition", "symmetry_group"), "A5"),
        (("gluing_definition", "global_gluing_imposed"), False),
        (("gluing_definition", "exact_oph_icosahedral_axes_used"), True),
        (("nested_thinned_family", "retained_carriers_are_nested"), False),
        (("count_density_calibration", "poisson_physical_process_claimed"), True),
        (("count_density_calibration", "physical_volume_calibration_claimed"), True),
        (("frozen_config", "profile_m_max"), 14),
        (("versioned_provenance_semantics", "causal_order"), "declared parents"),
        (("required_next_controls",), []),
    )
    for index, (key_path, value) in enumerate(mutations):
        candidate = json.loads(_canonical_bytes(receipt).decode("ascii"))
        target = candidate
        for key in key_path[:-1]:
            target = target[key]
        target[key_path[-1]] = value
        body = {
            key: field for key, field in candidate.items() if key != "payload_sha256"
        }
        candidate["payload_sha256"] = _sha(body)
        path = tmp_path / f"tampered-{index}.json"
        path.write_bytes(_canonical_bytes(candidate))
        with pytest.raises(IndependentRank3FCCVerificationError):
            verify_receipt(path)
