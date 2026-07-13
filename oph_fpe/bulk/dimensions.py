"""Continuous dimension estimators: internal diagnostics only.

OPH claim-boundary policy: a fractional "bulk dimension" is a category
error. Observers are primary and experience integer 3+1 charts; a bulk is
their agreement (see ``oph_fpe.bulk.observer_agreement``). The estimators
in this module exist to diagnose feature-locality and estimator health,
never to state a physical dimensionality. Every report they emit carries
``claim_level: internal_diagnostic_only`` and ``physical_claim: False``,
and downstream surfaces must not promote their ``estimate`` fields.
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np

DIMENSION_DIAGNOSTIC_POLICY = {
    "claim_level": "internal_diagnostic_only",
    "physical_claim": False,
    "policy": (
        "Continuous dimension estimates are estimator diagnostics. Physical "
        "dimensionality statements are integer observer-chart verdicts plus "
        "the observer mutual-agreement certificate."
    ),
}


def dimension_report(graph: nx.Graph, distances: np.ndarray) -> dict[str, Any]:
    return {
        "distance_source": "graph_shortest_path_mvp",
        "volume_growth_dimension": volume_growth_dimension(distances),
        "correlation_dimension": correlation_dimension(distances),
        "spectral_dimension": spectral_dimension(graph),
        **DIMENSION_DIAGNOSTIC_POLICY,
    }


def volume_growth_dimension(distances: np.ndarray) -> dict[str, Any]:
    rows = _central_rows(distances)
    finite = rows[np.isfinite(rows)]
    positive = finite[finite > 0]
    if positive.size < 4:
        return _empty_fit("not_enough_distances")
    radii = _integer_radii(positive)
    values = []
    for radius in radii:
        counts = np.sum((rows > 0) & (rows <= radius), axis=1)
        values.append(float(np.mean(counts)))
    slope, intercept, used = _loglog_fit(radii, values)
    return {
        "estimate": _finite_or_none(slope),
        "intercept": _finite_or_none(intercept),
        "points_used": used,
        "central_rows": int(rows.shape[0]),
    }


def correlation_dimension(distances: np.ndarray) -> dict[str, Any]:
    rows = _central_rows(distances)
    finite = rows[np.isfinite(rows)]
    positive = finite[finite > 0]
    if positive.size < 4:
        return _empty_fit("not_enough_distances")
    radii = _integer_radii(positive)
    n = rows.shape[0]
    target_n = distances.shape[0]
    values = [
        float(np.sum((rows > 0) & (rows <= radius)) / max(1, n * target_n))
        for radius in radii
    ]
    slope, intercept, used = _loglog_fit(radii, values)
    return {
        "estimate": _finite_or_none(slope),
        "intercept": _finite_or_none(intercept),
        "points_used": used,
        "central_rows": int(rows.shape[0]),
    }


def spectral_dimension(graph: nx.Graph, max_tau: int = 12) -> dict[str, Any]:
    n = graph.number_of_nodes()
    if n < 4:
        return _empty_fit("not_enough_nodes")
    nodes = list(graph.nodes)
    index = {node: pos for pos, node in enumerate(nodes)}
    transition = np.zeros((n, n), dtype=float)
    for node in nodes:
        row = index[node]
        degree = graph.degree[node]
        if degree == 0:
            transition[row, row] = 1.0
            continue
        for neighbor in graph.neighbors(node):
            transition[row, index[neighbor]] = 1.0 / degree
    matrix = np.eye(n)
    taus: list[float] = []
    returns: list[float] = []
    for tau in range(1, max_tau + 1):
        matrix = matrix @ transition
        value = float(np.trace(matrix) / n)
        if value > 0:
            taus.append(float(tau))
            returns.append(value)
    slope, intercept, used = _loglog_fit(taus, returns)
    estimate = -2.0 * slope if math.isfinite(slope) else None
    return {"estimate": estimate, "intercept": _finite_or_none(intercept), "points_used": used}


def _integer_radii(positive_distances: np.ndarray) -> list[float]:
    min_r = max(1, int(math.floor(float(np.min(positive_distances)))))
    max_r = max(min_r + 1, int(math.ceil(float(np.percentile(positive_distances, 60)))))
    return [float(radius) for radius in range(min_r, max_r + 1)]


def _central_rows(distances: np.ndarray) -> np.ndarray:
    finite = np.where(np.isfinite(distances), distances, np.nan)
    eccentricity = np.nanmax(finite, axis=1)
    if eccentricity.size == 0:
        return distances
    cutoff = np.nanpercentile(eccentricity, 25)
    mask = eccentricity <= cutoff
    if not np.any(mask):
        return distances
    return distances[mask]


def _loglog_fit(xs, ys) -> tuple[float, float, int]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pairs) < 2:
        return float("nan"), float("nan"), len(pairs)
    if len(pairs) > 4:
        pairs = pairs[1:-1]
    xlog = np.log([pair[0] for pair in pairs])
    ylog = np.log([pair[1] for pair in pairs])
    slope, intercept = np.polyfit(xlog, ylog, 1)
    return float(slope), float(intercept), len(pairs)


def _empty_fit(reason: str) -> dict[str, Any]:
    return {"estimate": None, "intercept": None, "points_used": 0, "reason": reason}


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None
