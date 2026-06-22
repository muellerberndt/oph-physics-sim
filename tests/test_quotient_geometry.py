from __future__ import annotations

import numpy as np

from oph_fpe.bulk.quotient_geometry import (
    ChannelMetricSpec,
    ProvenanceRecord,
    ancestry_split_report,
    euclidean_distance_certificate,
    metric_validity_report,
    quotient_geometry_certificate,
)


def _square_distance() -> np.ndarray:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    delta = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def test_quotient_geometry_contract_accepts_exact_metric_certificate() -> None:
    report = quotient_geometry_certificate(
        _square_distance(),
        quotient_ids=["a", "b", "c", "d"],
        channel_manifest=[ChannelMetricSpec(name="record_hist")],
        metric_mode="complete_case",
        jointly_separating=True,
        atlas_receipt={
            "identity_defect": 0.0,
            "inverse_defect": 0.0,
            "cocycle_defect": 0.0,
            "cycle_holonomy_defect": 0.0,
        },
        feature_receipt={
            "max_transport_defect": 0.0,
            "quotient_visible_missingness": True,
        },
        invariance_receipt={
            "gauge_distortion": 0.0,
            "port_distortion": 0.0,
            "order_distortion": 0.0,
            "schedule_distortion": 0.0,
            "partition_distortion": 0.0,
        },
        refinement_receipt={"convergent": True, "tail_modulus": 0.0},
        statistics_receipt={
            "ancestry_leakage_count": 0,
            "test_used_once": True,
            "positive_controls_passed": True,
            "negative_controls_passed": True,
        },
        require_euclidean=True,
    )

    assert report["status"] == "EXACT"
    assert report["QUOTIENT_GEOMETRY_CONTRACT_RECEIPT"] is True
    assert report["metric"]["valid_metric"] is True
    assert report["euclidean"]["exact_rank"] == 2


def test_metric_contract_forbids_pairwise_available_channels() -> None:
    report = metric_validity_report(
        _square_distance(),
        quotient_ids=["a", "b", "c", "d"],
        metric_mode="pairwise_available_channels",
    )

    assert report["valid_pseudometric"] is False
    assert "pairwise_available_channels_forbidden" in report["blockers"]


def test_metric_collision_report_counts_all_pairs_but_caps_witnesses() -> None:
    distance = np.zeros((20, 20), dtype=float)
    report = metric_validity_report(distance, quotient_ids=[f"q{index}" for index in range(20)])

    assert report["valid_pseudometric"] is True
    assert report["valid_metric"] is False
    assert report["zero_distance_collision_count"] == 190
    assert len(report["zero_distance_collision_pairs"]) == 64
    assert "zero_distance_feature_collisions" in report["metric_blockers"]


def test_euclidean_certificate_detects_non_euclidean_distance() -> None:
    distance = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 3.0],
            [1.0, 3.0, 0.0],
        ]
    )

    report = euclidean_distance_certificate(distance)

    assert report["euclidean_realizable"] is False
    assert "gram_matrix_has_negative_spectrum" in report["blockers"]


def test_ancestry_split_report_blocks_batch_and_parent_leakage() -> None:
    records = [
        ProvenanceRecord(
            record_id="train_child",
            split="train",
            batch_id="batch_a",
            seed_id="seed_a",
            boundary_condition_id="bc_a",
            trajectory_family_id="traj_a",
            parent_record_ids=("test_parent",),
        ),
        ProvenanceRecord(
            record_id="test_parent",
            split="test",
            batch_id="batch_a",
            seed_id="seed_a",
            boundary_condition_id="bc_a",
            trajectory_family_id="traj_a",
        ),
    ]

    report = ancestry_split_report(records)

    assert report["ancestry_leakage_count"] > 0
    assert "generative_group_crosses_splits" in report["blockers"]
    assert "ancestry_component_crosses_splits" in report["blockers"]
