from __future__ import annotations

import csv
import json
from pathlib import Path

from oph_fpe.bulk.observer_consensus_bulk import (
    _sample_h3_object,
    observer_consensus_bulk_readout_report,
    write_observer_consensus_bulk_readout_report,
)
from oph_fpe.bulk.self_reading_contract import (
    CAUSAL_POLICY_SCHEMA,
    COMMITTED_RECORD_SNAPSHOT_SCHEMA,
    COUNTERFACTUAL_RECORD_SNAPSHOT_SCHEMA,
    RUN_BINDING_SCHEMA,
    observer_population_binding_hash,
    record_signature_from_snapshot,
    validate_self_reading_report_structure,
    write_causal_event_artifact,
    write_record_commit_artifact,
)
from oph_fpe.evidence.hashes import stable_json_hash


def test_h3_object_csv_split_coordinates_round_trip() -> None:
    sampled = _sample_h3_object(
        {
            "object_id": "object-1",
            "h3_x": "-0.25",
            "h3_y": "0.5",
            "h3_z": "1.75",
        }
    )

    assert sampled["h3_spatial_point"] == [-0.25, 0.5, 1.75]


def test_observer_consensus_bulk_readout_is_theorem_assisted_not_strict(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "claims.json",
        {
            "observer_modular_time_receipt": True,
            "observer_facing_3p1d_h3_experience_receipt": True,
            "theorem_assisted_h3_bulk": True,
            "strict_neutral_bulk": False,
            "physical_cmb_output_usable_data_receipt": True,
            "physical_cmb_output_prediction_receipt": False,
        },
    )
    _write_json(
        run / "bulk_proof_certificate_report.json",
        {
            "observer_modular_time_receipt": True,
            "observer_facing_3p1d_h3_experience_receipt": True,
            "THEOREM_ASSISTED_H3_OBJECT_POPULATION_RECEIPT": True,
            "STRICT_NEUTRAL_BULK_RECEIPT": False,
            "selected_object_chart_report": "observer_chart_object_h3_lineage_report.json",
            "selected_object_chart_incidence_mode": "record_sector_checkpoint_lineage",
        },
    )
    _write_json(
        run / "observer_modular_experience_report.json",
        {
            "observer_modular_time_receipt": True,
            "observer_facing_3p1d_h3_experience_receipt": False,
        },
    )
    _write_json(
        run / "strict_neutral_bulk_frontier_report.json",
        {
            "strict_neutral_bulk": False,
            "blockers": ["independent_svd_rank3_selector_not_stable_or_false"],
        },
    )
    _write_json(
        run / "physical_cmb_output_comparison_report.json",
        {
            "USABLE_PHYSICAL_CMB_DATA_RECEIPT": True,
            "PHYSICAL_CMB_PREDICTION_RECEIPT": False,
        },
    )
    (run / "observer_views.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "view_type": "patch_observer",
                        "observer_id": 7,
                        "axis": [0.0, 0.0, 1.0],
                        "support_patch_count": 3,
                        "support_entropy_capacity": 12.0,
                        "observer_relative_times": [0.0, 6.283185307179586],
                        "dominant_record_signature": 4,
                        "modular_depth_mean": 2.0,
                        "repair_load_mean": 0.1,
                        "mismatch_density_mean": 0.0,
                        "visible_signature_entropy": 0.5,
                        "visible_readout_hash": "abcdef0123456789",
                    }
                ),
                json.dumps(
                    {
                        "view_type": "cap_observer",
                        "cap_index": 0,
                        "axis": [1.0, 0.0, 0.0],
                        "theta0": 0.3,
                        "collar_width": 0.1,
                        "observer_relative_times": [0.0, 6.283185307179586],
                        "cap_area_planck": 8.0,
                        "cap_entropy_capacity": 5.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        run / "source_dynamics_repair_record_observer_report.json",
        _valid_self_reading_contract(run, observer_count=1),
    )
    with (run / "h3_objects.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "object_id",
                "record_family_id",
                "family_mode",
                "observer_count",
                "support_size",
                "h3_compactness",
                "h3_compactness_normalized",
                "h3_spatial_point",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "object_id": "obj_001",
                "record_family_id": "family_1",
                "family_mode": "record_family_modular_response_mixture",
                "observer_count": "5",
                "support_size": "9",
                "h3_compactness": "0.2",
                "h3_compactness_normalized": "0.1",
                "h3_spatial_point": "[1.0, 2.0, 3.0]",
            }
        )
    (run / "neutral_objects.jsonl").write_text(
        json.dumps(
            {
                "object_id": "neutral_1",
                "observer_ids": [1, 2, 3],
                "visible_signature_key": "1:2:3",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = observer_consensus_bulk_readout_report([run])
    written = write_observer_consensus_bulk_readout_report([run], tmp_path / "out")

    assert report["OBSERVER_LIKE_SELF_READING_SYSTEM_RECEIPT"] is False
    validation = report["source_dynamics_repair_record_observer"]["validation"]
    assert validate_self_reading_report_structure(
        _valid_self_reading_contract(run, observer_count=1),
        materialized_observer_count=1,
    )["passed"] is True
    assert "source_bundle_architecture_or_repair_evidence_invalid" in validation[
        "blockers"
    ]
    assert report["claim_boundary"].startswith(
        "Observer rows are materialized, but the dedicated causal self-reading "
        "contract does not pass"
    )
    assert report["observer_modular_time_receipt"] is True
    assert report["observer_facing_3p1d_h3_experience_receipt"] is True
    assert report["observer_facing_consensus_3d_bulk_readout_receipt"] is False
    assert report["THEOREM_ASSISTED_CONSENSUS_3D_BULK_READOUT_RECEIPT"] is False
    assert report["STRICT_NEUTRAL_BULK_RECEIPT"] is False
    assert report["chart_blind_strict_neutral_quotient_bulk_receipt"] is False
    assert report["physical_cmb_output_comparison_receipt"] is True
    assert report["physical_cmb_prediction_receipt"] is False
    assert report["bulk_status"] == "not_established"
    assert report["observer_readout"]["observer_view_count"] == 2
    assert report["h3_object_readout"]["spatial_dimension"] == 3
    assert report["h3_object_readout"]["object_count"] == 1
    assert report["neutral_object_readout"]["median_observers_per_neutral_object"] == 3.0
    assert written["strict_neutral_blockers"] == ["independent_svd_rank3_selector_not_stable_or_false"]
    assert (tmp_path / "out" / "observer_consensus_bulk_readout_report.json").exists()
    assert (tmp_path / "out" / "observer_consensus_bulk_readout_report.md").exists()
    assert (tmp_path / "out" / "observer_perspective_rows.csv").exists()
    assert (tmp_path / "out" / "consensus_h3_object_rows.csv").exists()


def test_observer_rows_and_contradictory_summary_do_not_promote_without_feedback(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "observer_views.jsonl").write_text(
        json.dumps(
            {
                "view_type": "patch_observer",
                "observer_id": 1,
                "visible_readout_hash": "abcdef0123456789",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    contract = _valid_self_reading_contract(run, observer_count=1)
    contract["RECORD_READ_AFTER_WRITE_RECEIPT"] = False
    contract["OBSERVER_READBACK_FEEDBACK_CAUSAL_LOOP_RECEIPT"] = False
    contract["OBSERVER_SELF_READING_RECORD_LOOP_RECEIPT"] = False
    contract["record_observer"]["readback_changes_future_local_actions"] = False
    contract["record_observer"]["records_causally_bound_to_writes"] = False
    _write_json(run / "source_dynamics_repair_record_observer_report.json", contract)

    report = observer_consensus_bulk_readout_report([run])

    assert report["observer_readout"]["observer_view_count"] == 1
    assert report["OBSERVER_LIKE_SELF_READING_SYSTEM_RECEIPT"] is False
    assert report["observer_facing_consensus_3d_bulk_readout_receipt"] is False
    assert "row presence alone does not instantiate" in report["claim_boundary"]
    source = report["source_dynamics_repair_record_observer"]
    assert source["report_present"] is True
    assert source["report_path"] == str(
        run / "source_dynamics_repair_record_observer_report.json"
    )
    assert source["record_read_after_write_receipt"] is False
    assert source["observer_readback_feedback_causal_loop_receipt"] is False
    assert source["validation"]["passed"] is False


def test_self_reading_contract_cannot_be_borrowed_from_another_run(
    tmp_path: Path,
) -> None:
    observer_run = tmp_path / "observer_run"
    contract_run = tmp_path / "contract_run"
    observer_run.mkdir()
    contract_run.mkdir()
    (observer_run / "observer_views.jsonl").write_text(
        json.dumps(
            {
                "view_type": "patch_observer",
                "observer_id": 1,
                "visible_readout_hash": "abcdef0123456789",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (contract_run / "observer_views.jsonl").write_text(
        json.dumps(
            {
                "view_type": "patch_observer",
                "observer_id": 2,
                "visible_readout_hash": "fedcba9876543210",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        contract_run / "source_dynamics_repair_record_observer_report.json",
        _valid_self_reading_contract(contract_run, observer_count=1),
    )

    report = observer_consensus_bulk_readout_report([observer_run, contract_run])

    assert report["observer_readout"]["observer_view_count"] == 1
    assert report["OBSERVER_LIKE_SELF_READING_SYSTEM_RECEIPT"] is False
    assert "row presence alone does not instantiate" in report["claim_boundary"]
    assert report["source_dynamics_repair_record_observer"]["report_present"] is False
    assert report["source_dynamics_repair_record_observer"]["report_path"] is None


def test_same_size_self_reading_contract_cannot_be_swapped_between_runs(
    tmp_path: Path,
) -> None:
    contracts: list[dict] = []
    for index, visible_hash in enumerate(("run-a-hash", "run-b-hash")):
        run = tmp_path / f"run_{index}"
        run.mkdir()
        (run / "observer_views.jsonl").write_text(
            json.dumps(
                {
                    "view_type": "patch_observer",
                    "observer_id": 1,
                    "visible_readout_hash": visible_hash,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        contract = _valid_self_reading_contract(run, observer_count=1)
        _write_json(
            run / "source_dynamics_repair_record_observer_report.json", contract
        )
        contracts.append(contract)

    swapped_run = tmp_path / "run_1"
    _write_json(
        swapped_run / "source_dynamics_repair_record_observer_report.json",
        contracts[0],
    )

    report = observer_consensus_bulk_readout_report([swapped_run])

    assert report["OBSERVER_LIKE_SELF_READING_SYSTEM_RECEIPT"] is False
    validation = report["source_dynamics_repair_record_observer"]["validation"]
    assert validation["run_binding_validation"]["passed"] is False
    assert "same_run_observer_population_binding_hash_mismatch" in validation[
        "run_binding_validation"
    ]["blockers"]


def test_generic_causal_self_reading_does_not_require_source_atomic_tier(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "observer_views.jsonl").write_text(
        json.dumps(
            {
                "view_type": "patch_observer",
                "observer_id": 1,
                "visible_readout_hash": "generic-observer-hash",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    contract = _valid_self_reading_contract(run, observer_count=1)
    for key in (
        "SOURCE_PATCH_ARCHITECTURE_RECEIPT",
        "ECHOSAHEDRAL_CARRIER_CONFORMANCE",
        "FEDERATION_SEWING_RECEIPT",
        "CARRIER_QUOTIENT_INVARIANCE_RECEIPT",
        "CARRIER_REFINEMENT_NATURALITY_RECEIPT",
        "TRANSACTION_VALIDATION_COMPLETE_READ_CONFLICT_SET_RECEIPT",
        "UNION_PAYLOAD_ATOMIC_REVALIDATION_RECEIPT",
        "OPH_SOURCE_QUALIFIED_ATOMIC_SELF_READING_SYSTEM_RECEIPT",
    ):
        contract[key] = False
    contract["source_architecture"]["carrier_is_not_support_chart_cell"] = False
    contract["source_architecture"]["carrier_is_not_primitive_observer"] = False
    contract["source_architecture"]["carrier_support_conflation_present"] = True
    contract["repair_dynamics"]["target_free_rule"] = False
    contract["source_generator_target_free"] = False
    contract["source_forbidden_target_hits"] = ["declared_drive"]
    _write_json(
        run / "source_dynamics_repair_record_observer_report.json", contract
    )

    report = observer_consensus_bulk_readout_report([run])

    assert report["OBSERVER_LIKE_SELF_READING_SYSTEM_RECEIPT"] is False
    assert report[
        "OPH_SOURCE_QUALIFIED_ATOMIC_SELF_READING_SYSTEM_RECEIPT"
    ] is False
    validation = report["source_dynamics_repair_record_observer"]["validation"]
    assert validate_self_reading_report_structure(
        contract, materialized_observer_count=1
    )["generic_causal_self_reading_passed"] is True
    assert validation["generic_causal_self_reading_passed"] is False
    assert "source_bundle_architecture_or_repair_evidence_invalid" in validation[
        "generic_causal_self_reading_blockers"
    ]
    assert validation["source_qualified_atomic_passed"] is False


def test_causal_self_reading_rejects_tampered_raw_event_artifact(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "observer_views.jsonl").write_text(
        json.dumps(
            {
                "view_type": "patch_observer",
                "observer_id": 1,
                "visible_readout_hash": "observer-hash",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    contract = _valid_self_reading_contract(run, observer_count=1)
    _write_json(
        run / "source_dynamics_repair_record_observer_report.json", contract
    )
    (run / "source_dynamics_repair_record_observer_events.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )

    report = observer_consensus_bulk_readout_report([run])

    validation = report["source_dynamics_repair_record_observer"]["validation"]
    assert report["OBSERVER_LIKE_SELF_READING_SYSTEM_RECEIPT"] is False
    assert validation["causal_event_artifact_validation"]["passed"] is False
    assert "causal_event_artifact_hash_mismatch" in validation[
        "causal_event_artifact_validation"
    ]["blockers"]


def test_self_reading_contract_rejects_wrong_schema_missing_evidence_and_row_mismatch(
    tmp_path: Path,
) -> None:
    for name, mutate in (
        ("wrong_schema", lambda row: row.update(schema_version="forged")),
        ("missing_evidence", lambda row: row.pop("record_observer")),
        (
            "row_mismatch",
            lambda row: row["record_observer"].update(observer_count=2),
        ),
        (
            "read_before_commit",
            lambda row: row["record_observer"]["record_feedback_audit_rows"][0].update(
                read_cycle=0
            ),
        ),
        (
            "unchanged_action",
            lambda row: row["record_observer"]["record_feedback_audit_rows"][0].update(
                port_state_after=0
            ),
        ),
        (
            "mismatched_signature",
            lambda row: row["record_observer"]["record_feedback_audit_rows"][0].update(
                readback_record_signature=8
            ),
        ),
        (
            "unbounded_port",
            lambda row: row["record_observer"]["record_feedback_audit_rows"][0].update(
                bounded_port_index=12
            ),
        ),
        (
            "mismatched_patch",
            lambda row: row["record_observer"]["record_feedback_audit_rows"][0].update(
                patch_id=999
            ),
        ),
        (
            "independent_action",
            lambda row: row["record_observer"]["record_feedback_audit_rows"][0].update(
                port_state_after=3
            ),
        ),
    ):
        run = tmp_path / name
        run.mkdir()
        (run / "observer_views.jsonl").write_text(
            json.dumps(
                {
                    "view_type": "patch_observer",
                    "observer_id": 1,
                    "visible_readout_hash": "abcdef0123456789",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        contract = _valid_self_reading_contract(run, observer_count=1)
        mutate(contract)
        _write_json(
            run / "source_dynamics_repair_record_observer_report.json", contract
        )

        report = observer_consensus_bulk_readout_report([run])

        assert report["OBSERVER_LIKE_SELF_READING_SYSTEM_RECEIPT"] is False
        assert report["source_dynamics_repair_record_observer"]["validation"][
            "passed"
        ] is False
        if name in {
            "read_before_commit",
            "unchanged_action",
            "mismatched_signature",
            "unbounded_port",
            "mismatched_patch",
            "independent_action",
        }:
            assert "record_feedback_audit_rows_invalid" in report[
                "source_dynamics_repair_record_observer"
            ]["validation"]["blockers"]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_self_reading_contract(run: Path, *, observer_count: int) -> dict:
    patch_rows = [
        json.loads(line)
        for line in (run / "observer_views.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("view_type") == "patch_observer"
    ]
    cap_count = sum(
        1
        for line in (run / "observer_views.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("view_type") == "cap_observer"
    )
    assert len(patch_rows) == observer_count
    config = {"fixture": run.name, "seed": 2026, "dynamics": {"cycles": 4}}
    config_hash = stable_json_hash(config)
    seed = 2026
    _write_json(
        run / "observer_population_report.json",
        {
            "mode": "bounded_materialized_observer_population_v1",
            "materialized_observer_count": observer_count,
            "observer_wide_analyzed_count": observer_count,
            "verbose_jsonl_patch_observer_count": observer_count,
            "verbose_jsonl_cap_observer_count": cap_count,
            "materialized_rows_preserved": True,
            "verbose_jsonl_population": "all_materialized_observers",
        },
    )
    _write_json(run / "config.yml", config)
    _write_json(run / "seed_material.json", {"config_hash": config_hash, "seed": seed})
    _write_json(
        run / "manifest.json",
        {
            "patch_count": 16,
            "edge_count": 32,
            "gauge_coupled_dynamics": {"group_order": 6},
        },
    )
    observer_id = int(patch_rows[0]["observer_id"])
    committed_state = [1, *([0] * 11)]
    counterfactual_state = [2, *([0] * 11)]
    routed_signature = 3
    prior_signature = record_signature_from_snapshot(
        routed_signature, committed_state
    )
    counterfactual_signature = record_signature_from_snapshot(
        routed_signature, counterfactual_state
    )
    commit_id = "commit-1"
    record_id = "record-1"
    committed_snapshot = {
        "schema": COMMITTED_RECORD_SNAPSHOT_SCHEMA,
        "commit_id": commit_id,
        "observer_id": observer_id,
        "patch_id": observer_id,
        "record_id": record_id,
        "commit_cycle": 1,
        "commit_event_index": 1,
        "routed_node_signature_input": routed_signature,
        "canonical_patch_port_state": committed_state,
        "record_signature": prior_signature,
        "stable_count": 1,
        "commit_threshold": 1,
        "incident_mismatch": 0,
        "previous_committed": False,
        "current_committed": True,
        "newly_committed_transition": True,
    }
    committed_snapshot["snapshot_hash"] = stable_json_hash(committed_snapshot)
    counterfactual_snapshot = {
        **{
            key: value
            for key, value in committed_snapshot.items()
            if key != "snapshot_hash"
        },
        "schema": COUNTERFACTUAL_RECORD_SNAPSHOT_SCHEMA,
        "event_id": "event-1",
        "canonical_patch_port_state": counterfactual_state,
        "record_signature": counterfactual_signature,
    }
    rows = [
        {
            "schema": "oph_record_feedback_causal_event_v2",
            "policy_schema": CAUSAL_POLICY_SCHEMA,
            "event_id": "event-1",
            "observer_id": observer_id,
            "patch_id": observer_id,
            "commit_id": commit_id,
            "record_id": record_id,
            "committed_record_snapshot": committed_snapshot,
            "committed_record_snapshot_hash": committed_snapshot["snapshot_hash"],
            "counterfactual_record_snapshot": counterfactual_snapshot,
            "prior_record_signature": prior_signature,
            "readback_record_signature": prior_signature,
            "counterfactual_record_signature": counterfactual_signature,
            "record_coordinate_source": "committed_patch_port_state",
            "record_coordinate_port_index": 0,
            "prior_record_port_value": 1,
            "counterfactual_record_port_value": 2,
            "commit_cycle": 1,
            "read_cycle": 1,
            "write_cycle": 1,
            "commit_event_index": 1,
            "read_event_index": 2,
            "write_event_index": 3,
            "bounded_port_index": 0,
            "port_state_before": 0,
            "port_state_after": 1,
            "counterfactual_port_state_after": 2,
            "group_order": 6,
            "counterfactual_holds_nonrecord_inputs_fixed": True,
            "write_is_bounded_local_port": True,
            "read_count": 1,
            "write_count": 1,
            "records_causally_bound_to_prior_commits": True,
            "readback_changes_future_local_actions": True,
        }
    ]
    contract = {
        "schema_version": "oph_source_repair_record_observer_contract_v3",
        "mode": "source_dynamics_repair_record_observer_contract",
        "SOURCE_PATCH_ARCHITECTURE_RECEIPT": True,
        "PATCH_LOCAL_STATE_RECEIPT": True,
        "PATCH_PORT_BOUNDARY_RECEIPT": True,
        "PATCH_READBACK_RECEIPT": True,
        "PATCH_ALL_PORT_READBACK_RECEIPT": True,
        "RECORD_SIGNATURE_BINDS_ALL_LOCAL_PORT_STATE_RECEIPT": True,
        "ECHOSAHEDRAL_LOCAL_PATCH_ARCHITECTURE_RECEIPT": True,
        "ECHOSAHEDRAL_CARRIER_CONFORMANCE": True,
        "FEDERATION_SEWING_RECEIPT": True,
        "CARRIER_QUOTIENT_INVARIANCE_RECEIPT": True,
        "CARRIER_REFINEMENT_NATURALITY_RECEIPT": True,
        "TRANSACTION_VALIDATION_COMPLETE_READ_CONFLICT_SET_RECEIPT": True,
        "UNION_PAYLOAD_ATOMIC_REVALIDATION_RECEIPT": True,
        "LOCAL_REPAIR_DYNAMICS_RECEIPT": True,
        "RECORD_COMMIT_RECEIPT": True,
        "OBSERVER_SELF_READING_RECORD_LOOP_RECEIPT": True,
        "OBSERVER_LIKE_SELF_READING_SYSTEM_RECEIPT": True,
        "OPH_SOURCE_QUALIFIED_ATOMIC_SELF_READING_SYSTEM_RECEIPT": True,
        "RECORD_READ_AFTER_WRITE_RECEIPT": True,
        "OBSERVER_READBACK_FEEDBACK_CAUSAL_LOOP_RECEIPT": True,
        "record_observer": {
            "observer_count": observer_count,
            "committed_record_count": 1,
            "historical_committed_record_count": 1,
            "current_committed_record_count": 1,
            "causally_verified_observer_count": observer_count,
            "readback_count": 1,
            "feedback_event_count": 1,
            "readback_changes_future_local_actions": True,
            "records_causally_bound_to_writes": True,
            "orphan_read_count": 0,
            "record_readback_feedback_log_hash": stable_json_hash(rows),
            "record_feedback_audit_rows": rows,
            "external_cap_refresh_is_observer_feedback": False,
        },
        "source_architecture": {
            "bounded_patch_system": True,
            "simulation_native_source": True,
            "local_state_factor_count": 12,
            "boundary_port_count": 12,
            "all_local_port_readout_maps_materialized": True,
            "all_local_port_states_bound_into_records": True,
            "carrier_is_not_support_chart_cell": True,
            "carrier_is_not_primitive_observer": True,
            "carrier_support_conflation_present": False,
        },
        "repair_dynamics": {
            "local_update_rule": True,
            "uses_only_local_state_and_ports": True,
            "target_free_rule": True,
            "repair_event_count": 1,
            "nonlocal_write_count": 0,
            "repair_rule_hash": "sha256:test-repair-rule",
            "repair_event_log_hash": "sha256:test-repair-log",
        },
        "source_generator_target_free": True,
        "source_forbidden_target_hits": [],
        "run_binding": {
            "schema": RUN_BINDING_SCHEMA,
            "config_hash": config_hash,
            "seed": seed,
            "patch_count": 16,
            "edge_count": 32,
            "observer_population_binding_hash": observer_population_binding_hash(
                [int(row["observer_id"]) for row in patch_rows],
                [str(row["visible_readout_hash"]) for row in patch_rows],
            ),
        },
    }
    contract["causal_event_artifact"] = write_causal_event_artifact(
        run / "source_dynamics_repair_record_observer_events.jsonl",
        rows,
        run_binding=contract["run_binding"],
    )
    contract["record_commit_artifact"] = write_record_commit_artifact(
        run / "source_dynamics_repair_record_commits.jsonl",
        [committed_snapshot],
        run_binding=contract["run_binding"],
    )
    return contract
