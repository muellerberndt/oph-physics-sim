from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from oph_fpe.bulk.self_reading_contract import (
    CAUSAL_POLICY_SCHEMA,
    validate_observer_population,
    validate_self_reading_contract,
)
from oph_fpe.evidence.hashes import stable_json_hash


def test_compact_observer_population_validates_full_materialized_count(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"patch_count": 16}), encoding="utf-8"
    )
    (run / "observer_views.jsonl").write_text(
        json.dumps(
            {
                "view_type": "patch_observer",
                "observer_id": 11,
                "visible_readout_hash": "hash-11",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    compact_path = run / "observer_population_compact.npz"
    np.savez_compressed(
        compact_path,
        observer_ids=np.asarray([11, 12], dtype=np.int64),
        visible_readout_hashes=np.asarray([b"hash-11", b"hash-12"], dtype="S64"),
        support_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        observer_wide_analysis_included=np.asarray([True, False], dtype=bool),
    )
    payload = compact_path.read_bytes()
    report = {
        "mode": "bounded_materialized_observer_population_v1",
        "materialized_observer_count": 2,
        "observer_wide_analyzed_count": 1,
        "verbose_jsonl_patch_observer_count": 1,
        "verbose_jsonl_cap_observer_count": 0,
        "materialized_rows_preserved": True,
        "verbose_jsonl_population": "deterministic_analysis_subset_plus_cap_observers",
        "compact_population_artifact": {
            "schema": "compact_materialized_observer_population_npz_v1",
            "path": compact_path.name,
            "materialized_observer_count": 2,
            "analysis_enriched_observer_count": 1,
            "byte_count": len(payload),
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        },
    }
    (run / "observer_population_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    validation = validate_observer_population(run)

    assert validation["passed"] is True
    assert validation["materialized_observer_count"] == 2
    assert validation["retained_patch_observer_count"] == 1
    assert validation["observer_population_binding_hash"] is not None

    report["compact_population_artifact"]["sha256"] = "sha256:forged"
    (run / "observer_population_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    assert validate_observer_population(run)["passed"] is False

    report["compact_population_artifact"]["sha256"] = (
        "sha256:" + hashlib.sha256(payload).hexdigest()
    )
    (run / "observer_population_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    (run / "observer_views.jsonl").write_text(
        json.dumps(
            {
                "view_type": "patch_observer",
                "observer_id": 99,
                "visible_readout_hash": "foreign-hash",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    swapped = validate_observer_population(run)
    assert swapped["passed"] is False
    assert "compact_retained_jsonl_population_mismatch" in swapped["blockers"]


def test_feedback_booleans_do_not_replace_structural_observer_evidence() -> None:
    audit_rows = [
        {
            "schema": "oph_record_feedback_causal_event_v1",
            "policy_schema": CAUSAL_POLICY_SCHEMA,
            "event_id": "event-1",
            "observer_id": 1,
            "patch_id": 1,
            "record_id": "record-1",
            "prior_record_signature": -7,
            "readback_record_signature": -7,
            "counterfactual_record_signature": -8,
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
    report = {
        "schema_version": "oph_source_repair_record_observer_contract_v3",
        "mode": "source_dynamics_repair_record_observer_contract",
        "OBSERVER_LIKE_SELF_READING_SYSTEM_RECEIPT": True,
        "RECORD_READ_AFTER_WRITE_RECEIPT": True,
        "OBSERVER_READBACK_FEEDBACK_CAUSAL_LOOP_RECEIPT": True,
        "record_observer": {
            "observer_count": 1,
            "committed_record_count": 1,
            "causally_verified_observer_count": 1,
            "readback_count": 1,
            "feedback_event_count": 1,
            "readback_changes_future_local_actions": True,
            "records_causally_bound_to_writes": True,
            "orphan_read_count": 0,
            "external_cap_refresh_is_observer_feedback": False,
            "record_readback_feedback_log_hash": stable_json_hash(audit_rows),
            "record_feedback_audit_rows": audit_rows,
        },
    }

    validation = validate_self_reading_contract(
        report, materialized_observer_count=1
    )

    assert validation["passed"] is False
    assert "patch_local_state_receipt_missing_or_false" in validation[
        "blockers"
    ]
