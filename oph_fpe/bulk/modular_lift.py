from __future__ import annotations

import math
from typing import Any

import numpy as np

from oph_fpe.core.patchnet import PatchNet


def modular_lift_point_cloud(
    net: PatchNet,
    depth_samples: list[dict[int, float]],
    max_points: int = 100_000,
    seed: int = 1,
) -> np.ndarray:
    nodes = list(net.graph.nodes)
    if not nodes or not depth_samples:
        return np.zeros((0, 3), dtype=float)
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    all_depths = np.array(
        [float(sample.get(node, net.states[node].modular_depth)) for sample in depth_samples for node in nodes],
        dtype=float,
    )
    q_low, q_high = _depth_quantiles(all_depths)
    for sample_index, sample in enumerate(depth_samples):
        for node in nodes:
            xyz = np.asarray(net.graph.nodes[node].get("screen_xyz", (0.0, 0.0, 1.0)), dtype=float)
            depth = float(sample.get(node, net.states[node].modular_depth))
            radius = _depth_to_radius(depth, sample_index, len(depth_samples), q_low, q_high)
            points.append(radius * xyz)
    cloud = np.vstack(points)
    if cloud.shape[0] <= max_points:
        return cloud
    chosen = rng.choice(cloud.shape[0], size=max_points, replace=False)
    return cloud[np.sort(chosen)]


def final_modular_embedding(net: PatchNet) -> dict[int, list[float]]:
    result: dict[int, list[float]] = {}
    for node in net.graph.nodes:
        xyz = np.asarray(net.graph.nodes[node].get("screen_xyz", (0.0, 0.0, 1.0)), dtype=float)
        radius = _depth_to_radius(net.states[node].modular_depth, 0, 1, None, None)
        result[int(node)] = [float(value) for value in radius * xyz]
    return result


def modular_lift_dimension_report(
    net: PatchNet,
    depth_samples: list[dict[int, float]],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    max_points = int(config.get("max_points", 100_000))
    center_samples = int(config.get("center_samples", 1024))
    cloud = modular_lift_point_cloud(net, depth_samples, max_points=max_points, seed=seed)
    report = point_cloud_dimension_report(cloud, center_samples=center_samples, seed=seed)
    report["distance_source"] = "modular_lift_record_history"
    report["point_count"] = int(cloud.shape[0])
    report["screen_patch_count"] = int(net.graph.number_of_nodes())
    report["modular_samples"] = int(len(depth_samples))
    return report


def point_cloud_dimension_report(cloud: np.ndarray, center_samples: int = 1024, seed: int = 1) -> dict[str, Any]:
    if cloud.shape[0] < 16:
        return {
            "distance_source": "point_cloud",
            "volume_growth_dimension": _empty("not_enough_points"),
            "correlation_dimension": _empty("not_enough_points"),
        }
    rng = np.random.default_rng(seed)
    pair_count = max(20_000, min(250_000, center_samples * 128))
    left = rng.integers(0, cloud.shape[0], size=pair_count)
    right = rng.integers(0, cloud.shape[0], size=pair_count)
    pair_dist = np.linalg.norm(cloud[left] - cloud[right], axis=1)
    positive = pair_dist[pair_dist > 1e-12]
    if positive.size < 8:
        return {
            "distance_source": "point_cloud",
            "volume_growth_dimension": _empty("degenerate_cloud"),
            "correlation_dimension": _empty("degenerate_cloud"),
        }
    r_min = float(np.percentile(positive, 0.5))
    r_max = float(np.percentile(positive, 10.0))
    radii = np.geomspace(max(r_min, 1e-6), max(r_max, r_min * 1.5), num=10)
    corr = np.array([float(np.mean(positive <= radius)) for radius in radii])
    slope, intercept, used = _loglog_fit(radii, corr)
    return {
        "distance_source": "point_cloud",
        "volume_growth_dimension": {"estimate": slope, "intercept": intercept, "points_used": used},
        "correlation_dimension": {"estimate": slope, "intercept": intercept, "points_used": used},
    }


def _depth_to_radius(
    depth: float,
    sample_index: int,
    sample_count: int,
    q_low: float | None,
    q_high: float | None,
) -> float:
    # Modular flow supplies a radial/depth coordinate. Quantile normalization
    # chooses the reconstruction scale from the record-visible depth history,
    # rather than imposing a microscopic bulk radius at initialization.
    if q_low is not None and q_high is not None and q_high > q_low:
        normalized = min(1.0, max(0.0, (depth - q_low) / (q_high - q_low)))
        logistic = normalized ** (1.0 / 3.0)
    else:
        logistic = 1.0 / (1.0 + math.exp(-depth))
    if sample_count <= 1:
        return 0.05 + 0.9 * logistic
    return 0.05 + 0.86 * logistic + 0.04 * (sample_index / (sample_count - 1))


def _depth_quantiles(depths: np.ndarray) -> tuple[float | None, float | None]:
    if depths.size < 2 or not np.all(np.isfinite(depths)):
        return None, None
    return float(np.percentile(depths, 1)), float(np.percentile(depths, 99))


def _loglog_fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, int]:
    mask = (xs > 0) & (ys > 0)
    if int(np.sum(mask)) < 2:
        return float("nan"), float("nan"), int(np.sum(mask))
    x = np.log(xs[mask])
    y = np.log(ys[mask])
    if x.size > 4:
        x = x[1:-1]
        y = y[1:-1]
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), int(x.size)


def _empty(reason: str) -> dict[str, Any]:
    return {"estimate": float("nan"), "intercept": float("nan"), "points_used": 0, "reason": reason}
