"""Level graphs and committed refinement incidence for the dimension probe.

Exploratory, non-evidential.  Geometry and lineage are imported from the
committed tower module ``oph_fpe.core.icosahedral``; nothing is recomputed
here.  Per level the cell adjacency is the committed dual seam graph
(3-regular at every level); per adjacent level pair the incidence is the
committed ``CellRefinementMap`` lineage with its spherical-area
conditional-expectation weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oph_fpe.core.icosahedral import (
    GeodesicIcosahedralTower,
    build_geodesic_icosahedral_tower,
    geodesic_icosahedral_patch_arrays,
)


@dataclass(frozen=True)
class LevelGraph:
    """One tower level's cell-adjacency graph (committed dual seam graph)."""

    level: int
    cell_count: int
    edge_left: np.ndarray
    edge_right: np.ndarray

    @property
    def edge_count(self) -> int:
        return int(self.edge_left.size)


@dataclass(frozen=True)
class RefinementIncidence:
    """Committed lineage between adjacent levels with expectation weights.

    ``child_to_parent`` and ``expectation_weights`` are taken verbatim from
    the committed ``CellRefinementMap``; ``expectation_weights`` sums to one
    over each parent's four children (spherical-area weights, the sim
    instance of the Lean fiber normalization; declared convention C7 of
    DESIGN.md records the deviation from the exact uniform 1/4).
    """

    coarse_level: int
    fine_level: int
    child_to_parent: np.ndarray
    expectation_weights: np.ndarray
    max_uniform_deviation: float
    max_parent_weight_sum_residual: float


def build_level_graph(level: int) -> LevelGraph:
    """Load one committed level's cell adjacency arrays."""

    points, edge_left, edge_right = geodesic_icosahedral_patch_arrays(
        level, patch_basis="cells"
    )
    return LevelGraph(
        level=int(level),
        cell_count=int(points.shape[0]),
        edge_left=np.asarray(edge_left, dtype=np.int64),
        edge_right=np.asarray(edge_right, dtype=np.int64),
    )


def build_refinement_incidence(
    tower: GeodesicIcosahedralTower, coarse_level: int
) -> RefinementIncidence:
    """Read the committed refinement map between ``coarse_level`` and its child level."""

    refinement = tower.cell_refinements[coarse_level]
    child_to_parent = np.asarray(refinement.child_to_parent, dtype=np.int64)
    weights = np.asarray(refinement.conditional_expectation_weights, dtype=np.float64)
    parent_sums = np.zeros(len(refinement.children_by_parent), dtype=np.float64)
    np.add.at(parent_sums, child_to_parent, weights)
    return RefinementIncidence(
        coarse_level=int(refinement.coarse_level),
        fine_level=int(refinement.fine_level),
        child_to_parent=child_to_parent,
        expectation_weights=weights,
        max_uniform_deviation=float(np.max(np.abs(weights - 0.25))),
        max_parent_weight_sum_residual=float(np.max(np.abs(parent_sums - 1.0))),
    )


def build_tower_window(
    max_level: int,
) -> tuple[GeodesicIcosahedralTower, list[LevelGraph], list[RefinementIncidence]]:
    """Build levels ``0..max_level`` and their committed adjacent incidences."""

    if int(max_level) != max_level or not 0 <= max_level <= 5:
        raise ValueError("max_level must be an integer in [0, 5] (laptop ceiling)")
    tower = build_geodesic_icosahedral_tower(int(max_level))
    level_graphs = [build_level_graph(level) for level in range(int(max_level) + 1)]
    incidences = [
        build_refinement_incidence(tower, coarse) for coarse in range(int(max_level))
    ]
    return tower, level_graphs, incidences
