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
    assert report["receipt"] is True
    assert len(report["sector_rows"]) == 3
    assert report["total_variation_to_declared_stationary_law"] >= 0.0


def test_central_record_born_report_emits_commuting_event_surface():
    report = central_record_born_report(
        record_signature=np.array([10, 10, 20, 30]),
        committed=np.array([True, True, True, False]),
        stable_count=np.array([4, 4, 5, 1]),
        commit_cycles=4,
    )

    assert report["mode"] == "central_record_born_surface"
    assert report["event_count"] == 2
    assert report["record_projectors_commute"] is True
    assert report["luders_conditioning_idempotent"] is True
    assert report["receipt"] is True


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
