"""Probe operator assembly: symmetric weighted Laplacians on level unions.

Exploratory, non-evidential.  The probe operator is ``L = D - W`` on the
union node set of a level window:

* intra-level weights: 1.0 per committed seam adjacency (declared
  convention C2 of DESIGN.md, the level-0 unit-seam normalization of
  ``oph_fpe/em/base_carrier.laplacian_matrix`` extended level-uniformly);
* inter-level weights: ``kappa * conditional_expectation_weights[child]``
  per committed lineage pair (declared convention C3; the single-field
  stationarity of this quadratic form reproduces the committed ``embed``
  and ``conditional_expectation`` of ``oph_fpe/core/icosahedral.py``, the
  sim instance of the Lean section up to its fiber normalization and of the
  normalized marginalization pair of
  ``Lean/QFT/CarrierJoinTransport.lean``);
* the matched-scale control uses ``kappa / 4`` per lineage edge to change only
  the fiber-weight shape, while the all-ones control is explicitly labeled a
  ``coupling_scale_control``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from oph_fpe.dimension.geometry import LevelGraph, RefinementIncidence


def union_offsets(level_graphs: list[LevelGraph]) -> tuple[dict[int, int], int]:
    """Node offsets per level (ascending level order; declared convention C1)."""

    offsets: dict[int, int] = {}
    total = 0
    for graph in level_graphs:
        offsets[graph.level] = total
        total += graph.cell_count
    return offsets, total


def intra_level_edges(
    level_graphs: list[LevelGraph], offsets: dict[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-weight seam edges of every level, shifted into union indexing."""

    rows = [graph.edge_left + offsets[graph.level] for graph in level_graphs]
    cols = [graph.edge_right + offsets[graph.level] for graph in level_graphs]
    row = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    weight = np.ones(row.size, dtype=np.float64)
    return row, col, weight


def inter_level_edges(
    incidences: list[RefinementIncidence],
    offsets: dict[int, int],
    kappa: float,
    *,
    uniform_fiber_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Committed parent-child lineage edges with expectation weights times kappa."""

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for incidence in incidences:
        if incidence.coarse_level not in offsets or incidence.fine_level not in offsets:
            continue
        fine_ids = np.arange(incidence.child_to_parent.size, dtype=np.int64)
        rows.append(incidence.child_to_parent + offsets[incidence.coarse_level])
        cols.append(fine_ids + offsets[incidence.fine_level])
        fiber_weights = (
            np.full(incidence.expectation_weights.shape, 0.25, dtype=np.float64)
            if uniform_fiber_weights
            else incidence.expectation_weights
        )
        weights.append(float(kappa) * fiber_weights)
    if not rows:
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.zeros(0, dtype=np.float64)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(weights)


def laplacian_from_edges(
    row: np.ndarray, col: np.ndarray, weight: np.ndarray, node_count: int
) -> sp.csr_matrix:
    """Symmetric weighted graph Laplacian ``D - W`` from an undirected edge list."""

    if np.any(row == col):
        raise ValueError("self edges are not part of the probe operator")
    sym_row = np.concatenate([row, col])
    sym_col = np.concatenate([col, row])
    sym_weight = np.concatenate([weight, weight])
    adjacency = sp.coo_matrix(
        (sym_weight, (sym_row, sym_col)), shape=(node_count, node_count)
    ).tocsr()
    adjacency.sum_duplicates()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = sp.diags(degree, format="csr") - adjacency
    laplacian.sum_duplicates()
    laplacian.sort_indices()
    return laplacian.tocsr()


def union_laplacian(
    level_graphs: list[LevelGraph],
    incidences: list[RefinementIncidence],
    kappa: float,
    *,
    static_control: bool = False,
    uniform_fiber_weights: bool = False,
) -> tuple[sp.csr_matrix, dict]:
    """Assemble the probe operator on a level window.

    ``kappa = 0.0`` is the decoupled control (block-diagonal union).
    ``uniform_fiber_weights = True`` uses ``kappa / 4`` on every lineage edge,
    a matched-scale weight-shape control. ``static_control = True`` replaces
    every weight by 1.0 and is the coupling-scale control; ``kappa`` is ignored
    on the inter-level block in that mode.
    """

    if static_control and uniform_fiber_weights:
        raise ValueError("static and matched-scale controls are mutually exclusive")
    offsets, node_count = union_offsets(level_graphs)
    intra_row, intra_col, intra_weight = intra_level_edges(level_graphs, offsets)
    inter_row, inter_col, inter_weight = inter_level_edges(
        incidences,
        offsets,
        kappa=1.0 if static_control else kappa,
        uniform_fiber_weights=uniform_fiber_weights,
    )
    if static_control:
        inter_weight = np.ones_like(inter_weight)
        keep = np.ones(inter_weight.size, dtype=bool)
    else:
        keep = inter_weight != 0.0
    row = np.concatenate([intra_row, inter_row[keep]])
    col = np.concatenate([intra_col, inter_col[keep]])
    weight = np.concatenate([intra_weight, inter_weight[keep]])
    laplacian = laplacian_from_edges(row, col, weight, node_count)
    meta = {
        "node_count": int(node_count),
        "levels": [graph.level for graph in level_graphs],
        "kappa": float(kappa),
        "static_control": bool(static_control),
        "uniform_fiber_weights": bool(uniform_fiber_weights),
        "inter_level_weight_model": (
            "all_ones_coupling_scale_control"
            if static_control
            else (
                "matched_scale_uniform_one_quarter"
                if uniform_fiber_weights
                else "spherical_area_conditional_expectation"
            )
        ),
        "intra_edge_count": int(intra_row.size),
        "inter_edge_count": int(np.count_nonzero(keep)),
        "offsets": {str(level): int(offset) for level, offset in offsets.items()},
    }
    return laplacian, meta


def single_level_laplacian(level_graph: LevelGraph) -> tuple[sp.csr_matrix, dict]:
    """Intra-level operator of one level alone (decoupled single-level control)."""

    return union_laplacian([level_graph], [], kappa=0.0)


def symmetry_max_abs_asymmetry(matrix: sp.spmatrix) -> float:
    """Exact symmetry receipt: the maximum absolute entry of ``L - L^T``."""

    difference = (matrix - matrix.T).tocoo()
    if difference.nnz == 0:
        return 0.0
    return float(np.max(np.abs(difference.data)))


def require_symmetric(matrix: sp.spmatrix) -> float:
    """Fail closed on any asymmetry; return the recorded receipt value 0.0."""

    residual = symmetry_max_abs_asymmetry(matrix)
    if residual != 0.0:
        raise ValueError(f"probe operator asymmetry receipt violated: {residual}")
    return residual


def component_structure(matrix: sp.csr_matrix) -> tuple[int, np.ndarray]:
    """Connected components of the operator's weight graph."""

    count, labels = connected_components(matrix, directed=False)
    return int(count), labels


def kernel_basis_from_components(labels: np.ndarray) -> np.ndarray:
    """Orthonormal exact kernel basis: normalized per-component indicators."""

    node_count = labels.size
    component_ids = np.unique(labels)
    basis = np.zeros((node_count, component_ids.size), dtype=np.float64)
    for column, component in enumerate(component_ids):
        members = labels == component
        basis[members, column] = 1.0 / np.sqrt(float(np.count_nonzero(members)))
    return basis
