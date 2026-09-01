"""Fail-closed tests for the exploratory causal-set diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oph_fpe.bulk.causet_likeness import (
    DEFAULT_OUTPUT,
    _canonical_bytes,
    _sha,
    myrheim_meyer_fraction,
    produce_causet_likeness_report,
)
from oph_fpe.bulk.verify_causet_likeness_independent import (
    IndependentCausetVerificationError,
    _summary_line,
    verify_receipt,
)


@pytest.fixture(scope="module")
def report() -> dict:
    return produce_causet_likeness_report()


def test_source_control_is_small_and_local_cutoff_is_not_similar(report: dict) -> None:
    assert report["status"] == (
        "NOT_SIMILAR_AT_CURRENT_CUTOFF__INTERVAL_ORDERING_FRACTIONS_"
        "OUTSIDE_EXPLORATORY_4D_BAND"
    )
    assert report["source_control_status"] == (
        "INCONCLUSIVE__INSUFFICIENT_CERTIFIED_INTERVAL_SIZE"
    )
    source = report["source_statistics"]
    # The bounded source control has only 24 events and no interval at the
    # frozen exploratory minimum size of 32.
    assert source["event_count"] == 24
    assert source["height"] == 12
    assert source["width"] == 2
    assert source["maximum_interval_size"] == 21
    assert source["adequate_dimension_interval_count"] == 0
    assert source["myrheim_meyer_dimension_estimate"] is None
    local = report["existing_local_domain_diagnostic"]
    assert local["status"] == (
        "EXPLORATORY_INTERVAL_DIAGNOSTIC_READY_NOT_OPH_SIMILARITY"
    )
    assert local["causal_pair_sample_is_full_closure"] is True
    assert local["statistics"]["event_count"] == 2304
    assert local["statistics"]["comparable_pair_count"] == 70661
    assert local["statistics"]["height"] == 18
    assert local["statistics"]["width"] == 128
    assert local["statistics"]["maximum_interval_size"] == 72
    assert local["statistics"]["cover_graph_cycle_rank"] == 415
    assert local["statistics"]["cover_graph_is_forest"] is False
    assert local["statistics"]["global_ordering_fraction"] == pytest.approx(
        0.026633813986587544
    )
    comparison = report["single_cutoff_matched_interval_comparison"]
    assert comparison["adequate_interval_count"] == 736
    assert comparison["total_order_interval_count"] == 0
    assert comparison["interval_count_in_exploratory_4d_band"] == 0
    assert comparison["ordering_fraction_quantiles"]["minimum"] == pytest.approx(
        0.33365384615384613
    )
    assert comparison["ordering_fraction_quantiles"]["median"] == pytest.approx(
        0.5719512195121951
    )
    assert comparison["nearest_reference_counts"][
        "total_chain_negative"
    ] == 715
    assert sum(comparison["nearest_reference_counts"].values()) == 736
    carrier_controls = report["event_carrier_selection_controls"]
    assert carrier_controls["repair_only_classification"] == (
        "REPAIR_ONLY_EVENT_CARRIER_IS_ANTICHAIN"
    )
    assert carrier_controls["repair_only_versioned_provenance_edge_count"] == 0


def test_myrheim_meyer_formula_and_sprinkling_calibration(report: dict) -> None:
    assert myrheim_meyer_fraction(2.0) == pytest.approx(0.5)
    assert myrheim_meyer_fraction(4.0) == pytest.approx(0.1)
    for dimension, row in report["reference_controls"][
        "minkowski_alexandrov_sprinklings"
    ].items():
        assert row["calibration_within_one_dimension"], dimension
        assert row["runs"]
    families = report["reference_controls"][
        "nested_poisson_density_controls"
    ]
    assert families["oph_comparison_status"].startswith("NOT_EVALUATED")
    assert families["cross_geometry_similarity_claimed"] is False
    for geometry in families["geometries"].values():
        assert geometry["all_nested_inclusion_couplings_certified"]
        for run in geometry["runs"]:
            assert run["nested_carrier_inclusions_hold"]
            assert run["induced_causal_orders_hold"]
            assert [
                level["target_poisson_mean_count"] for level in run["levels"]
            ] == [64, 128, 256]
    de_sitter = families["geometries"][
        "de_sitter_flat_patch_3_plus_1"
    ]
    for run in de_sitter["runs"]:
        for level in run["levels"]:
            assert level["flat_myrheim_meyer_dimension_estimate"] is None
            assert level["flat_myrheim_meyer_status"] == (
                "NOT_APPLIED_CURVED_FLRW_REFERENCE_CONTROL"
            )


def test_nonmanifold_and_invariance_controls_fail_closed(report: dict) -> None:
    assert report["controls_fail_closed"] is True
    assert all(report["invariance_controls"].values())
    controls = report["reference_controls"]["nonmanifold_controls"]
    assert controls["total_chain"]["width"] == 1
    assert controls["eight_disconnected_record_chains"][
        "weak_component_count"
    ] == 8


def test_no_physical_or_manifoldlike_promotion(report: dict) -> None:
    assert report["CAUSET_DIAGNOSTIC_PIPELINE_REPRODUCTION_RECEIPT"] is True
    assert report["OPH_CAUSAL_SET_SIMILARITY_RECEIPT"] is False
    assert report["CAUSET_MANIFOLDLIKE_RECEIPT"] is False
    assert report["physical_promotion_allowed"] is False
    assert report["held_out_confirmation_status"] == "NOT_RUN_EXPLORATORY_ONLY"
    assert report["refinement_invariance_status"].startswith("NOT_EVALUATED")
    assert report["oph_refinement_family_comparison"]["status"].startswith(
        "NOT_EVALUATED"
    )


def test_report_is_deterministic_and_frozen(report: dict) -> None:
    assert _canonical_bytes(produce_causet_likeness_report()) == _canonical_bytes(
        report
    )
    assert DEFAULT_OUTPUT.read_bytes() == _canonical_bytes(report)


def test_independent_verifier_replays_and_refuses_mutation(
    tmp_path: Path, report: dict
) -> None:
    result = verify_receipt(DEFAULT_OUTPUT)
    assert result["verified"] is True
    assert result["source_event_count"] == 24
    assert result["source_control_status"].startswith("INCONCLUSIVE")
    assert result["local_event_count"] == 2304
    assert result["local_adequate_interval_count"] == 736
    summary = _summary_line(result)
    assert "source_n=24" in summary
    assert "local_n=2304" in summary
    assert "local_adequate_intervals=736" in summary
    candidate = json.loads(_canonical_bytes(report).decode("ascii"))
    candidate["source_statistics"]["height"] += 1
    body = {key: value for key, value in candidate.items() if key != "report_sha256"}
    candidate["report_sha256"] = _sha(body)
    path = tmp_path / "mutated.json"
    path.write_bytes(_canonical_bytes(candidate))
    with pytest.raises(IndependentCausetVerificationError):
        verify_receipt(path)

    binding_mutation = json.loads(_canonical_bytes(report).decode("ascii"))
    binding_mutation["source_binding"]["semantic_poset_sha256"] = (
        "sha256:" + "0" * 64
    )
    body = {
        key: value
        for key, value in binding_mutation.items()
        if key != "report_sha256"
    }
    binding_mutation["report_sha256"] = _sha(body)
    path2 = tmp_path / "mutated_binding.json"
    path2.write_bytes(_canonical_bytes(binding_mutation))
    with pytest.raises(IndependentCausetVerificationError):
        verify_receipt(path2)

    result_mutation = json.loads(_canonical_bytes(report).decode("ascii"))
    result_mutation["single_cutoff_matched_interval_comparison"]["result"] = (
        "SIMILAR"
    )
    body = {
        key: value
        for key, value in result_mutation.items()
        if key != "report_sha256"
    }
    result_mutation["report_sha256"] = _sha(body)
    path3 = tmp_path / "mutated_result.json"
    path3.write_bytes(_canonical_bytes(result_mutation))
    with pytest.raises(IndependentCausetVerificationError):
        verify_receipt(path3)
