import numpy as np

from oph_fpe.core.graph import fibonacci_sphere_points
from oph_fpe.observers.subjective import (
    _observer_overlap_counts,
    observer_consensus_report,
    observer_view_rows,
)


def test_observer_overlap_counts_match_bruteforce_pairs():
    neighbor_indices = np.array(
        [
            [1, 2, 3],
            [3, 4, 5],
            [2, 3, 6],
            [7, 8, 9],
        ],
        dtype=np.int64,
    )

    counts = _observer_overlap_counts(neighbor_indices)

    assert counts == {
        (0, 1): 1,
        (0, 2): 2,
        (1, 2): 1,
    }


def test_observer_consensus_report_limits_materialized_sample_pairs():
    points = fibonacci_sphere_points(64)
    raw_fields = {
        "record_signature": np.arange(64, dtype=np.int64) % 5,
        "committed_mask": np.ones(64, dtype=bool),
        "repair_load": np.linspace(0.0, 1.0, 64),
    }

    report = observer_consensus_report(
        points,
        raw_fields=raw_fields,
        cell_entropy=np.ones(64, dtype=float),
        sample_count=24,
        neighborhood_size=8,
        seed=17,
        sample_pair_limit=3,
    )

    assert report["pair_count"] >= len(report["sample_pairs"])
    assert len(report["sample_pairs"]) <= 3
    assert report["sample_pair_limit"] == 3


def test_observer_population_caps_leave_below_cap_behavior_unchanged():
    points = fibonacci_sphere_points(48)
    raw_fields = {
        "record_signature": np.arange(48, dtype=np.int64) % 7,
        "committed_mask": np.ones(48, dtype=bool),
        "stable_count": np.arange(48, dtype=float),
        "repair_load": np.linspace(0.0, 1.0, 48),
        "cumulative_repair_load": np.linspace(1.0, 2.0, 48),
        "local_mismatch_density": np.linspace(0.5, 0.0, 48),
        "modular_depth": np.linspace(-1.0, 1.0, 48),
    }
    kwargs = {
        "raw_fields": raw_fields,
        "visible_fields": {},
        "caps": [],
        "times": [0.0, 1.0],
        "cell_area_planck": np.ones(48, dtype=float),
        "cell_entropy": np.ones(48, dtype=float),
        "sample_count": 16,
        "neighborhood_size": 8,
        "seed": 19,
    }

    uncapped = observer_view_rows(points, **kwargs)
    capped = observer_view_rows(
        points,
        **kwargs,
        overlap_correspondence_max_observers=16,
    )

    assert [row["measured_overlap_correspondences"] for row in uncapped] == [
        row["measured_overlap_correspondences"] for row in capped
    ]
    assert all(
        row["overlap_correspondence_analysis"]["included"] for row in capped
    )
    assert all(
        row["overlap_correspondence_analysis"]["sampling_policy"]
        == "all_materialized_observers"
        for row in capped
    )

    consensus_kwargs = {
        "raw_fields": raw_fields,
        "cell_entropy": np.ones(48, dtype=float),
        "sample_count": 16,
        "neighborhood_size": 8,
        "seed": 19,
    }
    uncapped_report = observer_consensus_report(points, **consensus_kwargs)
    capped_report = observer_consensus_report(
        points,
        **consensus_kwargs,
        analysis_max_observers=16,
    )
    for key in (
        "observer_count",
        "pair_count",
        "median_overlap_jaccard",
        "median_signature_histogram_similarity",
        "p10_signature_histogram_similarity",
        "sample_pairs",
    ):
        assert uncapped_report[key] == capped_report[key]
    assert capped_report["analysis_sampling_policy"] == "all_materialized_observers"


def test_observer_population_caps_are_deterministic_nested_and_fail_transparent():
    points = fibonacci_sphere_points(96)
    raw_fields = {
        "record_signature": np.arange(96, dtype=np.int64) % 11,
        "committed_mask": np.ones(96, dtype=bool),
        "stable_count": np.arange(96, dtype=float),
        "repair_load": np.linspace(0.0, 1.0, 96),
        "cumulative_repair_load": np.linspace(1.0, 3.0, 96),
        "local_mismatch_density": np.linspace(0.75, 0.0, 96),
        "modular_depth": np.linspace(-2.0, 2.0, 96),
    }
    view_kwargs = {
        "raw_fields": raw_fields,
        "visible_fields": {},
        "caps": [],
        "times": [0.0],
        "cell_area_planck": np.ones(96, dtype=float),
        "cell_entropy": np.ones(96, dtype=float),
        "sample_count": 48,
        "neighborhood_size": 10,
        "seed": 23,
        "overlap_correspondence_max_observers": 12,
    }
    first = observer_view_rows(points, **view_kwargs)
    second = observer_view_rows(points, **view_kwargs)
    assert first == second
    included_ids = {
        int(row["observer_id"])
        for row in first
        if row["overlap_correspondence_analysis"]["included"]
    }
    assert len(first) == 48
    assert len(included_ids) == 12
    for row in first:
        analysis = row["overlap_correspondence_analysis"]
        assert analysis["materialized_observer_count"] == 48
        assert analysis["analyzed_observer_count"] == 12
        assert analysis["sampling_policy"] == "deterministic_observer_id_hash_rank_v1"
        if not analysis["included"]:
            assert row["measured_overlap_correspondences"] == []
            assert row["measured_overlap_correspondence_receipt"] is False
        assert all(
            int(peer["peer_observer_id"]) in included_ids
            for peer in row["measured_overlap_correspondences"]
        )

    consensus_kwargs = {
        "raw_fields": raw_fields,
        "cell_entropy": np.ones(96, dtype=float),
        "sample_count": 48,
        "neighborhood_size": 10,
        "seed": 23,
    }
    small = observer_consensus_report(
        points, **consensus_kwargs, analysis_max_observers=12
    )
    repeated = observer_consensus_report(
        points, **consensus_kwargs, analysis_max_observers=12
    )
    larger = observer_consensus_report(
        points, **consensus_kwargs, analysis_max_observers=24
    )
    assert small == repeated
    assert small["materialized_observer_count"] == 48
    assert small["analyzed_observer_count"] == 12
    assert small["observer_count"] == 12
    assert small["analysis_sampling_policy"] == "deterministic_observer_id_hash_rank_v1"
    assert set(small["analysis_observer_ids"]) < set(larger["analysis_observer_ids"])
