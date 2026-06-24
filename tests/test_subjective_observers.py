import numpy as np

from oph_fpe.core.graph import fibonacci_sphere_points
from oph_fpe.observers.subjective import _observer_overlap_counts, observer_consensus_report


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
