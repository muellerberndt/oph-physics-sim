from __future__ import annotations

from oph_fpe.observers.semantic_clock import (
    affine_clock_residual_report,
    distributed_observer_uid,
    normalize_observer_frame,
    observer_registry_audit,
    semantic_history_invariance_report,
)


def test_semantic_history_ignores_scheduler_metadata():
    base = [
        {
            "semantic_event_id": "record:a",
            "observer_record_order": 0,
            "label": "seen",
            "scheduler_event_index": 10,
            "worker_id": "w0",
            "packet_latency": 0.01,
        }
    ]
    rescheduled = [
        {
            "semantic_event_id": "record:a",
            "observer_record_order": 0,
            "label": "seen",
            "scheduler_event_index": 99,
            "worker_id": "w7",
            "packet_latency": 0.20,
        }
    ]

    report = semantic_history_invariance_report([base, rescheduled])

    assert report["SEMANTIC_HISTORY_SCHEDULER_INVARIANCE_RECEIPT"] is True
    assert len(set(report["semantic_history_digests"])) == 1


def test_normalize_observer_frame_separates_execution_and_clock_fields():
    frame = normalize_observer_frame(
        {
            "cycle": 12,
            "eventIndex": 4,
            "modularTime": 1.5,
            "clockUncertainty": 0.125,
        },
        record_order=3,
    )

    assert frame["execution_epoch"] == 12
    assert frame["scheduler_event_index"] == 4
    assert frame["observer_record_order"] == 3
    assert frame["observer_modular_parameter"] == 1.5
    assert frame["observer_clock_uncertainty"] == 0.125
    assert frame["execution_provenance"]["scheduler_event_index"] == 4


def test_registry_audit_rejects_duplicate_or_mixed_namespace_entries():
    uid = distributed_observer_uid(run_id="u", observer_kind="patch", global_observer_index=0)
    report = observer_registry_audit(
        [
            {
                "distributed_observer_uid": uid,
                "observer_kind": "patch",
                "local_observer_index": 0,
                "local_anchor_patch_id": "patch:1",
            },
            {
                "distributed_observer_uid": uid,
                "observer_kind": "cap",
                "local_observer_index": 0,
                "local_anchor_patch_id": "0",
            },
        ]
    )

    assert report["GLOBAL_OBSERVER_REGISTRY_NAMESPACE_RECEIPT"] is False
    assert report["duplicate_uid_count"] == 1
    assert report["anchor_reuse_violation_count"] == 1


def test_affine_clock_residual_report_is_a_finite_certificate_only():
    report = affine_clock_residual_report([0.0, 1.0, 2.0], [1.0, 3.0, 5.0], scale=2.0, shift=1.0)

    assert report["OBSERVER_CLOCK_AFFINE_RESIDUAL_RECEIPT"] is True
    assert report["max_residual"] == 0.0
