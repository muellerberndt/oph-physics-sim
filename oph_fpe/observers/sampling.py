"""Import-cycle-free deterministic observer-population sampling."""

from __future__ import annotations

import numpy as np


def deterministic_observer_analysis_indices(
    observer_ids: np.ndarray | list[int],
    *,
    max_observers: int | None,
) -> np.ndarray:
    """Return a deterministic, nested observer-population subsample.

    The rank is a fixed SplitMix64 permutation of the materialized observer ID.
    Selecting the lowest ranks makes smaller analysis caps strict subsets of
    larger caps while avoiding the screen-latitude bias of first-row slicing.
    """

    ids = np.asarray(observer_ids, dtype=np.int64).reshape(-1)
    count = int(ids.size)
    if max_observers is None or int(max_observers) >= count:
        return np.arange(count, dtype=np.int64)
    limit = max(0, int(max_observers))
    if limit == 0:
        return np.zeros(0, dtype=np.int64)
    with np.errstate(over="ignore"):
        rank = ids.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        rank = (rank ^ (rank >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        rank = (rank ^ (rank >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        rank = rank ^ (rank >> np.uint64(31))
    order = np.lexsort((np.arange(count, dtype=np.int64), rank))
    return np.sort(order[:limit].astype(np.int64, copy=False))


__all__ = ["deterministic_observer_analysis_indices"]
