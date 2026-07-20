from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from oph_fpe.bulk import physical_h3_kms_aggregate as aggregate_module
from oph_fpe.bulk.physical_h3_kms_campaign import run_frozen_campaign_cell
from oph_fpe.bulk.physical_h3_kms_runtime import THREAD_ENVIRONMENT_KEYS


SEEDS = (20_260_751, 20_260_761, 20_260_771)
RUNGS = (4_096, 16_384, 65_536, 262_144)
REPLICATES = ("primary",)
EXPECTED = tuple(
    (seed, rung, replicate_id)
    for seed in SEEDS
    for rung in RUNGS
    for replicate_id in REPLICATES
)
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
HASH_D = "sha256:" + "d" * 64
HASH_E = "sha256:" + "e" * 64
FAMILY_CONTRACT = {
    "frozen": True,
    "retune_after_freeze": False,
    "retirement_rule": {
        "decisive_rungs": [16_384, 65_536, 262_144],
        "same_predeclared_failure_mode_required": True,
        "all_cells_powered_and_complete_required": True,
        "frozen_before_first_run": True,
        "failure_mode_derivation": "verified_predicate_set_sha256_v1",
    },
}


def _verified_cell(
    key: tuple[int, int, str],
    status: str,
    *,
    failure_predicate: str = "independent_clock_selection_failed",
) -> dict:
    if status == "VALID_PASS":
        stage_statuses = {
            stage_id: "VALID_PASS"
            for stage_id in aggregate_module.REQUIRED_CELL_STAGE_IDS
        }
        not_evaluated_reasons: list[str] = []
        scientific_failures: list[str] = []
    elif status == "VALID_FAIL":
        stage_statuses = {
            stage_id: "VALID_PASS"
            for stage_id in aggregate_module.REQUIRED_CELL_STAGE_IDS
        }
        stage_statuses["P5_frozen_candidate_interventions"] = "VALID_FAIL"
        not_evaluated_reasons = []
        scientific_failures = [
            f"P5_frozen_candidate_interventions:{failure_predicate}"
        ]
    else:
        stage_statuses = {
            stage_id: "NOT_EVALUATED"
            for stage_id in aggregate_module.REQUIRED_CELL_STAGE_IDS
        }
        not_evaluated_reasons = ["independent_producer_missing"]
        scientific_failures = []
    failure_mode, producer_evidence, predicates = (
        aggregate_module._verified_failure_mode(scientific_failures)
    )
    cell = {
        "cell": key,
        "expected_cells": EXPECTED,
        "seeds": SEEDS,
        "rungs": RUNGS,
        "replicate_ids": REPLICATES,
        "campaign_id": "physical-h3-kms-frozen-family-003",
        "instrument_version": "physical-h3-kms-v2",
        "instrument_commit": HASH_A,
        "campaign_family_hash": HASH_B,
        "frozen_campaign_family_sha256": HASH_C,
        "family_contract": FAMILY_CONTRACT,
        "protocol_hash": HASH_D,
        "runtime_sha256": HASH_E,
        "cell_status": status,
        "stage_statuses": stage_statuses,
        "p8_instrument_contract_valid": True,
        "scientific_failures": scientific_failures,
        "not_evaluated_reasons": not_evaluated_reasons,
        "failure_mode": failure_mode,
        "producer_failure_evidence_hash": producer_evidence,
        "failure_evidence_hash": None,
        "result_artifact_hash": HASH_B,
        "source_capture_hash": HASH_C,
        "manifest_byte_sha256": HASH_D,
        "preflight_sha256": HASH_E,
        "run_directory": f"/runs/{key[0]}-{key[1]}-{key[2]}",
    }
    if status == "VALID_FAIL":
        cell["failure_evidence_hash"] = aggregate_module.canonical_sha256(
            {
                "derivation": "aggregate_bound_failure_evidence_v1",
                "cell": list(key),
                "failure_mode": failure_mode,
                "failure_predicates": predicates,
                "stage_statuses": stage_statuses,
                "source_capture_hash": HASH_C,
                "result_artifact_hash": HASH_B,
                "manifest_byte_sha256": HASH_D,
                "preflight_sha256": HASH_E,
                "protocol_hash": HASH_D,
            }
        )
    return cell


def test_single_incomplete_cell_cannot_promote_retire_or_authorize_64k() -> None:
    report = aggregate_module._aggregate_verified_cells(
        [_verified_cell(EXPECTED[0], "INCOMPLETE")]
    )

    assert report["aggregation_instrument_status"] == "VALID_PASS"
    assert report["campaign_status"] == "INCOMPLETE"
    assert report["observed_cell_count"] == 1
    assert report["missing_cell_count"] == 11
    assert report["P8_measurement_status"] == "NOT_EVALUATED"
    assert report["physical_promotion_allowed"] is False
    assert report["stable_failure_rule_satisfied"] is False
    assert report["branch_retirement_authorized"] is False
    assert report["ready_for_64k"] is False
    assert report["observed_physical_stage_status_counts"]["NOT_EVALUATED"] == 8


def test_all_replayed_cells_must_pass_before_promotion() -> None:
    report = aggregate_module._aggregate_verified_cells(
        [_verified_cell(key, "VALID_PASS") for key in EXPECTED]
    )

    assert report["campaign_status"] == "VALID_PASS"
    assert report["P8_measurement_status"] == "VALID_PASS"
    assert report["physical_promotion_allowed"] is True
    assert report["branch_retirement_authorized"] is False
    assert report["missing_cell_count"] == 0
    assert all(report["ready_for_rung"].values())


def test_stable_failure_requires_every_decisive_cell_and_one_mode() -> None:
    cells = [
        _verified_cell(
            key,
            "VALID_PASS" if key[1] == 4_096 else "VALID_FAIL",
            failure_predicate="frozen_clock_separation_failed",
        )
        for key in EXPECTED
    ]
    report = aggregate_module._aggregate_verified_cells(cells)

    assert report["campaign_status"] == "VALID_FAIL"
    assert report["physical_promotion_allowed"] is False
    assert report["stable_failure_rule_satisfied"] is True
    assert report["branch_retirement_authorized"] is True
    expected_mode = aggregate_module._verified_failure_mode(
        ["frozen_clock_separation_failed"]
    )[0]
    assert report["stable_failure_modes"] == [expected_mode]


def test_mixed_failure_modes_do_not_retire_branch() -> None:
    cells = [
        _verified_cell(
            key,
            "VALID_PASS" if key[1] == 4_096 else "VALID_FAIL",
            failure_predicate=(
                "frozen_clock_separation_failed"
                if key[0] != SEEDS[-1]
                else "h3_control_margin_failed"
            ),
        )
        for key in EXPECTED
    ]
    report = aggregate_module._aggregate_verified_cells(cells)

    assert report["campaign_status"] == "VALID_FAIL"
    assert report["stable_failure_rule_satisfied"] is False
    assert report["branch_retirement_authorized"] is False


def test_64k_readiness_requires_all_frozen_4k_and_16k_cells() -> None:
    prerequisites = [key for key in EXPECTED if key[1] in {4_096, 16_384}]
    report = aggregate_module._aggregate_verified_cells(
        [_verified_cell(key, "VALID_PASS") for key in prerequisites]
    )

    assert report["campaign_status"] == "INCOMPLETE"
    assert report["ready_for_rung"]["4096"] is True
    assert report["ready_for_rung"]["16384"] is True
    assert report["ready_for_rung"]["65536"] is True
    assert report["ready_for_64k"] is True
    assert report["ready_for_rung"]["262144"] is False


def test_duplicate_or_mixed_family_cells_are_rejected() -> None:
    cell = _verified_cell(EXPECTED[0], "VALID_PASS")
    with pytest.raises(aggregate_module.CampaignAggregationError, match="duplicate"):
        aggregate_module._aggregate_verified_cells([cell, deepcopy(cell)])

    other = _verified_cell(EXPECTED[1], "VALID_PASS")
    other["protocol_hash"] = HASH_A
    with pytest.raises(aggregate_module.CampaignAggregationError, match="protocol_hash"):
        aggregate_module._aggregate_verified_cells([cell, other])


def test_producer_authored_failure_mode_or_evidence_cannot_authorize_retirement() -> None:
    cell = _verified_cell(EXPECTED[0], "VALID_FAIL")
    forged_mode = deepcopy(cell)
    forged_mode["failure_mode"] = "arbitrary-retirement-label"
    with pytest.raises(
        aggregate_module.CampaignAggregationError,
        match="failure mode is not verifier-derived",
    ):
        aggregate_module._aggregate_verified_cells([forged_mode])

    forged_evidence = deepcopy(cell)
    forged_evidence["failure_evidence_hash"] = HASH_A
    with pytest.raises(
        aggregate_module.CampaignAggregationError,
        match="failure evidence is not aggregate-bound",
    ):
        aggregate_module._aggregate_verified_cells([forged_evidence])


def test_p8_instrument_contract_invalidity_cannot_be_ignored() -> None:
    cell = _verified_cell(EXPECTED[0], "VALID_PASS")
    cell["p8_instrument_contract_valid"] = False
    with pytest.raises(
        aggregate_module.CampaignAggregationError,
        match="P8 instrument contract is invalid",
    ):
        aggregate_module._aggregate_verified_cells([cell])


def test_public_aggregator_fails_closed_without_replay_bundles() -> None:
    report = aggregate_module.aggregate_physical_h3_kms_family([])

    assert report["aggregation_instrument_status"] == "INSTRUMENT_INVALID"
    assert report["campaign_status"] == "INSTRUMENT_INVALID"
    assert report["physical_promotion_allowed"] is False
    assert report["branch_retirement_authorized"] is False
    assert report["ready_for_64k"] is False
    assert report["blockers"]


def test_public_aggregator_replays_and_reduces_one_real_campaign_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise campaign -> disk replay -> preflight -> family reduction."""

    for key in THREAD_ENVIRONMENT_KEYS:
        monkeypatch.setenv(key, "1")
    run_directory = tmp_path / "physical-cell"
    receipt = run_frozen_campaign_cell(run_directory, current_rung=4_096)

    assert receipt["instrument_status"] == "VALID_PASS"
    assert receipt["cell_scientific_status"] == "INCOMPLETE"

    report = aggregate_module.aggregate_physical_h3_kms_family([run_directory])

    assert report["aggregation_instrument_status"] == "VALID_PASS"
    assert report["campaign_status"] == "INCOMPLETE"
    assert report["observed_cell_count"] == 1
    assert report["missing_cell_count"] == 11
    assert report["cell_status_counts"]["INCOMPLETE"] == 1
    assert report["observed_physical_stage_status_counts"] == {
        "NOT_EVALUATED": 8,
        "INCOMPLETE": 0,
        "VALID_PASS": 0,
        "VALID_FAIL": 0,
        "INSTRUMENT_INVALID": 0,
    }
    assert report["physical_promotion_allowed"] is False
    assert report["branch_retirement_authorized"] is False
    assert report["ready_for_64k"] is False
