from __future__ import annotations

import numpy as np

from oph_fpe.core.screen_receipts import (
    central_record_born_report,
    edge_sector_heat_kernel_report,
    observer_checkpoint_restoration_report,
)
from oph_fpe.defects.array_s3_holonomy import S3_CLASS


def test_edge_sector_heat_kernel_report_uses_s3_classes():
    gauge = np.array([0, 1, 2, 3, 4, 5], dtype=np.int16)

    report = edge_sector_heat_kernel_report(gauge, group_name="S3", beta=1.0, s3_class=S3_CLASS)

    assert report["mode"] == "edge_sector_heat_kernel_casimir_surrogate"
    assert report["group"] == "S3"
    assert report["edge_count"] == 6
    assert report["edge_sector_diagnostic_receipt"] is True
    assert report["heat_kernel_validation_receipt"] is False
    assert report["receipt"] is False
    assert len(report["sector_rows"]) == 3
    assert report["total_variation_to_declared_stationary_law"] >= 0.0


def test_edge_sector_diagnostic_rejects_invalid_labels_classes_and_beta():
    negative_label = edge_sector_heat_kernel_report(
        np.array([-1]),
        group_name="S3",
        s3_class=np.array([0, 1, 2]),
    )
    invalid_classes = edge_sector_heat_kernel_report(
        np.array([0, 1, 2]),
        group_name="S3",
        s3_class=np.array([-7, 1, 99]),
    )
    invalid_beta = edge_sector_heat_kernel_report(
        np.array([0, 1, 2]),
        group_name="S3",
        beta=np.inf,
        s3_class=np.array([0, 1, 2]),
    )

    for report in (negative_label, invalid_classes, invalid_beta):
        assert report["edge_sector_diagnostic_receipt"] is False
        assert report["heat_kernel_validation_receipt"] is False
        assert report["receipt"] is False


def test_central_record_born_report_emits_classical_partition_surface():
    report = central_record_born_report(
        record_signature=np.array([10, 10, 20, 30]),
        committed=np.array([True, True, True, False]),
        stable_count=np.array([4, 4, 5, 1]),
        commit_cycles=4,
    )

    assert report["mode"] == "central_record_born_surface"
    assert report["event_count"] == 2
    assert report["record_partition_filters_commute"] is True
    assert report["partition_filter_idempotent"] is True
    assert report["record_projector_commutation_validation_receipt"] is False
    assert report["luders_conditioning_validation_receipt"] is False
    assert report["classical_record_partition_receipt"] is True
    assert report["centrality_validation_receipt"] is False
    assert report["central_record_algebra_receipt"] is False
    assert report["born_law_validation_receipt"] is False
    assert report["physical_claim"] is False
    assert report["sample_events"] == [
        {
            "event_id": 10,
            "count": 2,
            "empirical_record_frequency": 2 / 3,
            "partition_filter_idempotent": True,
            "luders_conditioning_validation_receipt": False,
        },
        {
            "event_id": 20,
            "count": 1,
            "empirical_record_frequency": 1 / 3,
            "partition_filter_idempotent": True,
            "luders_conditioning_validation_receipt": False,
        },
    ]
    assert report["receipt"] is False


def test_central_record_born_legacy_name_cannot_green_a_born_claim():
    report = central_record_born_report(
        record_signature=np.array([10, 10, 10, 20]),
        committed=np.array([True, True, True, True]),
        stable_count=np.array([4, 4, 4, 4]),
        commit_cycles=4,
    )

    assert report["classical_record_partition_receipt"] is True
    assert report["centrality_validation_receipt"] is False
    assert report["central_record_algebra_receipt"] is False
    assert report["born_law_validation_receipt"] is False
    assert report["receipt"] is False
    assert "not a Born-law" in report["claim_boundary"]


def test_central_record_algebra_rejects_broadcast_shapes_and_bad_cycles():
    broadcast = central_record_born_report(
        record_signature=np.array([10, 20]),
        committed=np.array([True]),
        stable_count=np.array([4, 4]),
        commit_cycles=4,
    )
    negative_cycles = central_record_born_report(
        record_signature=np.array([10, 20]),
        committed=np.array([True, True]),
        stable_count=np.array([4, 4]),
        commit_cycles=-1,
    )
    fractional_ids = central_record_born_report(
        record_signature=np.array([10.2, 10.8]),
        committed=np.array([True, True]),
        stable_count=np.array([4, 4]),
        commit_cycles=4,
    )
    string_mask = central_record_born_report(
        record_signature=np.array([10, 20]),
        committed=np.array(["", "False"]),
        stable_count=np.array([4, 4]),
        commit_cycles=4,
    )
    fractional_stability = central_record_born_report(
        record_signature=np.array([10, 20]),
        committed=np.array([True, True]),
        stable_count=np.array([3.9, 4.1]),
        commit_cycles=4,
    )

    for report in (
        broadcast,
        negative_cycles,
        fractional_ids,
        string_mask,
        fractional_stability,
    ):
        assert report["classical_record_partition_receipt"] is False
        assert report["centrality_validation_receipt"] is False
        assert report["central_record_algebra_receipt"] is False
        assert report["born_law_validation_receipt"] is False
        assert report["receipt"] is False


def test_observer_checkpoint_restoration_exact_copy_has_zero_bound():
    raw_fields = {
        "record_signature": np.array([1, 1, 2, 3]),
        "stable_count": np.array([4, 5, 1, 1]),
        "committed_mask": np.array([1, 1, 0, 0]),
        "repair_load": np.array([0.1, 0.2, 0.3, 0.4]),
        "s3_class_density": np.array([0.0, 0.5, 1.0, 0.5]),
    }
    views = [
        {
            "observer_id": "obs0",
            "view_type": "patch_observer",
            "support_nodes": [0, 1],
        }
    ]

    report = observer_checkpoint_restoration_report(raw_fields, views)

    assert report["mode"] == "observer_checkpoint_restoration"
    assert report["observer_count"] == 1
    assert report["max_future_law_total_variation_bound"] == 0.0
    assert report["receipt"] is True
