"""Calibration suite: the same estimators on graphs of known dimension.

Exploratory, non-evidential.  Cases, expectations, and tolerances are
pinned in DESIGN.md section 5.  A calibration failure is repaired in the
estimator, never in the tolerance.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from oph_fpe.dimension.operators import laplacian_from_edges

DIMENSION_TOLERANCE = 0.15
INTEGER_TARGETS = (1, 2, 3)


def cycle_laplacian(node_count: int) -> sp.csr_matrix:
    """Unit-weight cycle graph ``C_n`` (dimension 1)."""

    nodes = np.arange(node_count, dtype=np.int64)
    row = nodes
    col = (nodes + 1) % node_count
    weight = np.ones(node_count, dtype=np.float64)
    return laplacian_from_edges(row, col, weight, node_count)


def torus_laplacian(shape: tuple[int, ...]) -> sp.csr_matrix:
    """Unit-weight periodic lattice on ``prod(shape)`` nodes."""

    node_count = int(np.prod(shape))
    index = np.arange(node_count, dtype=np.int64).reshape(shape)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for axis in range(len(shape)):
        rows.append(index.ravel())
        cols.append(np.roll(index, -1, axis=axis).ravel())
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    weight = np.ones(row.size, dtype=np.float64)
    return laplacian_from_edges(row, col, weight, node_count)


def tree4_laplacian(depth: int) -> tuple[sp.csr_matrix, int]:
    """Rooted tree with root degree 4 and interior degree 4, truncated at ``depth``.

    Interior nodes have three children and one parent; node count is
    ``1 + 2 * (3**depth - 1)``.  The control expectation is anomalous:
    exponential volume growth, no finite ``d_s`` plateau.
    """

    if depth < 1:
        raise ValueError("depth must be at least 1")
    parents: list[int] = []
    children: list[int] = []
    next_id = 1
    frontier = []
    for _ in range(4):
        parents.append(0)
        children.append(next_id)
        frontier.append(next_id)
        next_id += 1
    for _ in range(depth - 1):
        upcoming: list[int] = []
        for node in frontier:
            for _ in range(3):
                parents.append(node)
                children.append(next_id)
                upcoming.append(next_id)
                next_id += 1
        frontier = upcoming
    node_count = next_id
    row = np.asarray(parents, dtype=np.int64)
    col = np.asarray(children, dtype=np.int64)
    weight = np.ones(row.size, dtype=np.float64)
    return laplacian_from_edges(row, col, weight, node_count), node_count


CALIBRATION_CASES = (
    {"name": "cycle_4096", "kind": "integer", "expected": 1, "builder": ("cycle", 4096)},
    {"name": "torus2d_64", "kind": "integer", "expected": 2, "builder": ("torus", (64, 64))},
    {"name": "torus3d_24", "kind": "integer", "expected": 3, "builder": ("torus", (24, 24, 24))},
    {"name": "tree4_depth8", "kind": "anomalous_tree", "expected": None, "builder": ("tree", 8)},
)

CROSS_CHECK_CASES = (
    {"name": "cycle_512", "builder": ("cycle", 512)},
    {"name": "torus2d_40", "builder": ("torus", (40, 40))},
    {"name": "torus3d_12", "builder": ("torus", (12, 12, 12))},
    {"name": "tree4_depth6", "builder": ("tree", 6)},
)


def build_calibration_graph(builder: tuple) -> sp.csr_matrix:
    kind, argument = builder
    if kind == "cycle":
        return cycle_laplacian(int(argument))
    if kind == "torus":
        return torus_laplacian(tuple(argument))
    if kind == "tree":
        matrix, _ = tree4_laplacian(int(argument))
        return matrix
    raise ValueError(f"unknown calibration builder: {kind!r}")


def integer_band_check(value: float | None, expected: int) -> bool:
    """Pinned band ``[expected - 0.15, expected + 0.15]``."""

    if value is None:
        return False
    return abs(float(value) - float(expected)) <= DIMENSION_TOLERANCE


def anomalous_band_check(value: float | None) -> bool:
    """Pinned anomalous control: outside every integer band."""

    if value is None:
        return False
    return all(
        abs(float(value) - float(target)) > DIMENSION_TOLERANCE
        for target in INTEGER_TARGETS
    )


def calibration_verdict(row: dict, case: dict) -> dict:
    """Per-case pass fields for the receipt (calibration only; tower rows
    carry no pass field)."""

    if case["kind"] == "integer":
        expected = int(case["expected"])
        return {
            "expected_dimension": expected,
            "d_s_within_band": integer_band_check(row.get("d_s_median"), expected),
            "d_weyl_within_band": integer_band_check(row.get("d_weyl"), expected),
        }
    return {
        "expected_dimension": None,
        "d_weyl_outside_all_integer_bands": anomalous_band_check(row.get("d_weyl")),
        "d_s_median_outside_all_integer_bands": anomalous_band_check(
            row.get("d_s_median")
        ),
    }
