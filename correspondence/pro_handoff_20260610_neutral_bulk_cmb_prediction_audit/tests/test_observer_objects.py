from __future__ import annotations

import numpy as np

from oph_fpe.observers.objects import (
    RecordFamily,
    assign_counterfactual_stability_from_records,
    counterfactual_stability,
    extract_record_families,
    object_consensus_score,
    observer_object_report,
)


def test_observer_object_persistence():
    records = {
        "record_signature": np.array([7, 7, 7, 2, 2]),
        "stable_count": np.array([8, 8, 8, 8, 1]),
        "repair_load": np.zeros(5),
    }
    left = np.array([0, 1, 2, 3])
    right = np.array([1, 2, 3, 4])
    families = extract_record_families(records, (left, right), persistence_horizon=8)
    assert families[0].record_signature == 7
    assert families[0].support_nodes == [0, 1, 2]
    assert families[0].persistence == 8


def test_observer_object_coarse_visible_packet_groups_records():
    records = {
        "record_signature": np.array([10, 11, 12, 90, 91, 92]),
        "stable_count": np.array([8, 8, 8, 8, 8, 8]),
        "repair_load": np.zeros(6),
        "s3_sector_class": np.zeros(6, dtype=int),
    }
    left = np.array([0, 1, 2, 3, 4])
    right = np.array([1, 2, 3, 4, 5])
    families = extract_record_families(
        records,
        (left, right),
        projections={
            "packet_mode": "coarse_visible_packet",
            "signature_bins": 2,
            "include_s3_sector": True,
            "min_support_size": 2,
        },
        persistence_horizon=8,
    )

    assert len(families) == 2
    assert families[0].support_nodes == [0, 1, 2]
    assert families[1].support_nodes == [3, 4, 5]


def test_connected_record_families_skip_background_components():
    records = {
        "record_signature": np.ones(12, dtype=int),
        "stable_count": np.ones(12, dtype=int) * 8,
        "repair_load": np.zeros(12),
    }
    left = np.arange(11)
    right = np.arange(1, 12)

    families = extract_record_families(
        records,
        (left, right),
        projections={"min_support_size": 2, "max_support_size": 5},
        persistence_horizon=8,
    )

    assert families == []


def test_transition_affinity_families_do_not_require_connected_support():
    records = {
        "record_signature": np.array([10, 90, 10, 90, 10, 90]),
        "stable_count": np.array([8, 8, 8, 8, 8, 8]),
        "committed_mask": np.ones(6),
        "repair_load": np.array([0.1, 0.8, 0.2, 0.9, 0.15, 0.85]),
        "cumulative_repair_load": np.array([0.2, 0.9, 0.1, 0.8, 0.15, 0.85]),
        "s3_sector_class": np.array([1, 2, 1, 2, 1, 2]),
    }
    left = np.array([0, 1, 2, 3, 4])
    right = np.array([1, 2, 3, 4, 5])

    families = extract_record_families(
        records,
        (left, right),
        projections={
            "family_mode": "transition_affinity",
            "packet_mode": "coarse_visible_packet",
            "signature_bins": 2,
            "include_s3_sector": True,
            "transition_affinity_fields": ["record_family", "s3_sector_class"],
            "record_family_modulus": 8,
            "min_support_size": 2,
        },
        persistence_horizon=8,
    )

    supports = [set(family.support_nodes) for family in families]
    assert {0, 2, 4} in supports
    assert {1, 3, 5} in supports
    assert all(family.transition_affinity for family in families)
    assert all(family.construction_mode == "transition_affinity" for family in families)
    assert all(family.evidence_keys for family in families)


def test_transition_affinity_object_consensus_prefers_visible_histograms_without_support_overlap():
    family = RecordFamily(
        "obj",
        [100, 101],
        3,
        8,
        1.0,
        "hash",
        0.0,
        transition_affinity={"object_packet": 3, "checkpoint_class": 2},
        construction_mode="transition_affinity",
        evidence_keys=["object_packet", "checkpoint_class"],
    )
    views = [
        {
            "view_type": "patch_observer",
            "support_nodes": [0, 1],
            "object_packet_histogram": {"3": 0.8},
            "transition_affinity_histograms": {"checkpoint_class": {"2": 0.75}},
        },
        {
            "view_type": "patch_observer",
            "support_nodes": [2, 3],
            "object_packet_histogram": {"3": 0.6},
            "transition_affinity_histograms": {"checkpoint_class": {"2": 0.5}},
        },
    ]

    score = object_consensus_score(family, views)
    report = observer_object_report([family], views)

    assert score > 0.5
    assert report["transition_object_count"] == 1
    assert report["support_free_transition_object_count"] == 1
    assert report["median_overlap_agreement"] > 0.5


def test_counterfactual_stability_and_report():
    family = RecordFamily("obj", [0, 1], 4, 8, 1.0, "hash", 0.0)
    score = counterfactual_stability(family, [0, 1, 2], lambda _family, perturb: 4 if perturb < 2 else 9)
    family.counterfactual_stability = score
    report = observer_object_report([family], [{"support_nodes": [0, 1, 2]}])
    assert score == 2 / 3
    assert report["object_count"] == 1
    assert report["persistent_object_count"] == 1


def test_histogram_object_consensus_does_not_penalize_large_observer_view():
    family = RecordFamily("obj", [0, 1], 4, 8, 1.0, "hash", 0.0)
    views = [
        {
            "view_type": "patch_observer",
            "support_nodes": list(range(100)),
            "record_signature_histogram": {"4": 0.2, "9": 0.8},
        }
    ]

    assert object_consensus_score(family, views) > 0.5


def test_assign_counterfactual_stability_from_records_visible_subsupport():
    records = {
        "record_signature": np.array([4, 4, 4, 9]),
        "stable_count": np.array([8, 8, 8, 8]),
        "repair_load": np.zeros(4),
    }
    family = RecordFamily("obj", [0, 1, 2], 4, 8, 1.0, "hash", 0.0)

    assign_counterfactual_stability_from_records([family], records, perturbations=8, seed=4)

    assert family.counterfactual_stability == 1.0


def test_fake_record_rewrite_detected():
    families = [
        RecordFamily("a", [0], 1, 8, 1.0, "same", 1.0),
        RecordFamily("b", [1], 2, 8, 1.0, "same", 1.0),
    ]
    assert observer_object_report(families, [])["bad_record_rewrite_detected"] is True
