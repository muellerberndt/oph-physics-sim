from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform

from oph_fpe.claims import STRICT_NEUTRAL_LATENT_H3_SELECTION_RECEIPT


DEFAULT_CANDIDATES = ("E2", "E3", "E4", "H2", "H3", "H4", "H5", "S2")
ONE_SE_MIN = 0.01


@dataclass(frozen=True)
class LatentGeometryFit:
    model: str
    family: str
    dimension: int
    train_rmse: float
    heldout_rmse: float
    heldout_se: float
    pair_count_train: int
    pair_count_heldout: int
    radius: float | None = None
    selected_by_one_se: bool = False
    raw_best: bool = False


def select_latent_geometry(
    distance: np.ndarray,
    *,
    seed: int = 1,
    max_points: int = 192,
    heldout_fraction: float = 0.25,
    candidates: tuple[str, ...] = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    """Select a latent geometry by held-out pair-distance prediction.

    This is the strict-neutral replacement for raw SVD/rank selectors. It only
    sees a distance matrix derived upstream from observer-visible records or
    neutral objects. It does not read screen axes, H3 coordinates, support
    nodes, cap normals, or the theorem-side chart.
    """

    rng = np.random.default_rng(int(seed))
    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1] or distance.shape[0] < 8:
        return _empty_selection("invalid_or_too_small_distance_matrix")
    distance = np.where(np.isfinite(distance), np.maximum(distance, 0.0), 0.0)
    np.fill_diagonal(distance, 0.0)
    n = int(distance.shape[0])
    if n > int(max_points):
        indices = np.sort(rng.choice(n, size=int(max_points), replace=False))
        work = distance[np.ix_(indices, indices)]
    else:
        indices = np.arange(n)
        work = distance

    pair_index = np.transpose(np.triu_indices(work.shape[0], k=1))
    if pair_index.shape[0] < 16:
        return _empty_selection("too_few_pair_distances")
    heldout_count = max(8, int(round(float(heldout_fraction) * pair_index.shape[0])))
    heldout_ids = set(
        int(value)
        for value in rng.choice(
            pair_index.shape[0],
            size=min(heldout_count, pair_index.shape[0] - 1),
            replace=False,
        )
    )
    heldout_pairs = np.asarray([pair_index[index] for index in sorted(heldout_ids)], dtype=int)
    train_pairs = np.asarray(
        [pair_index[index] for index in range(pair_index.shape[0]) if index not in heldout_ids],
        dtype=int,
    )
    train_distance = _impute_heldout(work, heldout_pairs)

    fits: list[LatentGeometryFit] = []
    for label in candidates:
        fit = _fit_candidate(label, work, train_distance, train_pairs, heldout_pairs)
        if fit is not None:
            fits.append(fit)
    if not fits:
        return _empty_selection("no_finite_candidate_fit")

    finite = [fit for fit in fits if math.isfinite(fit.heldout_rmse)]
    if not finite:
        return _empty_selection("no_finite_heldout_score")
    raw_best = min(finite, key=lambda fit: fit.heldout_rmse)
    one_se_threshold = raw_best.heldout_rmse + max(ONE_SE_MIN, raw_best.heldout_se)
    compatible = [fit for fit in finite if fit.heldout_rmse <= one_se_threshold]
    selected = min(compatible, key=_model_complexity_key)
    fits = [
        LatentGeometryFit(
            **{
                **asdict(fit),
                "selected_by_one_se": fit.model == selected.model,
                "raw_best": fit.model == raw_best.model,
            }
        )
        for fit in fits
    ]
    h3 = next((fit for fit in fits if fit.model == "H3"), None)
    s2 = next((fit for fit in fits if fit.model == "S2"), None)
    h2 = next((fit for fit in fits if fit.model == "H2"), None)
    h4 = next((fit for fit in fits if fit.model == "H4"), None)
    h3_selected = selected.model == "H3"
    h3_margin_vs_s2 = _margin(h3, s2)
    h3_margin_vs_h2 = _margin(h3, h2)
    h3_margin_vs_h4 = _margin(h3, h4)
    return {
        "mode": "strict_neutral_heldout_latent_geometry_selection_v0",
        "observer_or_object_count": int(n),
        "fit_count": int(work.shape[0]),
        "subsample_indices": [int(value) for value in indices[: min(indices.size, 4096)]],
        "heldout_pair_count": int(heldout_pairs.shape[0]),
        "train_pair_count": int(train_pairs.shape[0]),
        "raw_best_model": raw_best.model,
        "selected_model": selected.model,
        "selected_family": selected.family,
        "selected_dimension": int(selected.dimension),
        "one_se_threshold": float(one_se_threshold),
        "selection_rule": (
            "lowest-complexity geometry whose held-out RMSE is within max(0.01, one standard error) "
            "of the raw best held-out RMSE"
        ),
        "fits": [asdict(fit) for fit in fits],
        "h3_selected": bool(h3_selected),
        STRICT_NEUTRAL_LATENT_H3_SELECTION_RECEIPT: bool(h3_selected),
        "h3_margin_vs_s2": h3_margin_vs_s2,
        "h3_margin_vs_h2": h3_margin_vs_h2,
        "h3_margin_vs_h4": h3_margin_vs_h4,
        "physical_claim": False,
        "claim_boundary": (
            "Held-out latent-geometry selector over a neutral distance matrix. It replaces raw rank/SVD "
            "as a strict-neutral diagnostic, but it is only one gate; it does not establish bulk emergence "
            "without object extraction, dimension, leakage, control, and refinement receipts."
        ),
    }


def strict_neutral_latent_geometry_gate(
    selection_reports: list[dict[str, Any]],
    *,
    min_fraction: float = 0.75,
) -> dict[str, Any]:
    usable = [report for report in selection_reports if report]
    h3_count = sum(1 for report in usable if bool(report.get("h3_selected", False)))
    fraction = float(h3_count / len(usable)) if usable else 0.0
    passed = bool(usable and fraction >= float(min_fraction))
    return {
        "mode": "strict_neutral_latent_geometry_refinement_gate_v0",
        "report_count": int(len(usable)),
        "h3_selected_count": int(h3_count),
        "h3_selected_fraction": fraction,
        "min_fraction": float(min_fraction),
        STRICT_NEUTRAL_LATENT_H3_SELECTION_RECEIPT: passed,
        "claim_boundary": (
            "Refinement/seed aggregate for the strict-neutral held-out geometry selector. Passing this gate "
            "does not by itself prove a bulk; it feeds the strict neutral object-bulk receipt."
        ),
    }


def write_latent_geometry_selection_report(
    distance: np.ndarray,
    out: Path,
    *,
    seed: int = 1,
    max_points: int = 192,
    heldout_fraction: float = 0.25,
) -> dict[str, Any]:
    report = select_latent_geometry(
        distance,
        seed=seed,
        max_points=max_points,
        heldout_fraction=heldout_fraction,
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _fit_candidate(
    label: str,
    original: np.ndarray,
    train_distance: np.ndarray,
    train_pairs: np.ndarray,
    heldout_pairs: np.ndarray,
) -> LatentGeometryFit | None:
    family = label[0].upper()
    try:
        dim = int(label[1:])
    except (ValueError, IndexError):
        return None
    best_pred: np.ndarray | None = None
    best_radius: float | None = None
    best_score = float("inf")
    if family == "E":
        coords = _classical_mds(train_distance, dim)
        if coords is None:
            return None
        best_pred = squareform(pdist(coords, metric="euclidean"))
    elif family == "S" and dim == 2:
        for radius in _radius_grid(_positive_median(train_distance)):
            coords = _spherical_coords(train_distance, dim=dim, radius=radius)
            if coords is None:
                continue
            predicted = _spherical_distance(coords, radius)
            score, _ = _rmse(original, predicted, heldout_pairs)
            if score < best_score:
                best_score = score
                best_pred = predicted
                best_radius = radius
    elif family == "H":
        for radius in _radius_grid(_positive_median(train_distance)):
            predicted = _hyperbolic_prediction(train_distance, dim=dim, radius=radius)
            if predicted is None:
                continue
            score, _ = _rmse(original, predicted, heldout_pairs)
            if score < best_score:
                best_score = score
                best_pred = predicted
                best_radius = radius
    else:
        return None
    if best_pred is None:
        return None
    train_rmse, _ = _rmse(original, best_pred, train_pairs)
    heldout_rmse, heldout_se = _rmse(original, best_pred, heldout_pairs)
    return LatentGeometryFit(
        model=label,
        family=family,
        dimension=dim,
        train_rmse=float(train_rmse),
        heldout_rmse=float(heldout_rmse),
        heldout_se=float(heldout_se),
        pair_count_train=int(train_pairs.shape[0]),
        pair_count_heldout=int(heldout_pairs.shape[0]),
        radius=best_radius,
    )


def _impute_heldout(distance: np.ndarray, heldout_pairs: np.ndarray) -> np.ndarray:
    work = np.asarray(distance, dtype=float).copy()
    positive = work[np.isfinite(work) & (work > 1e-12)]
    global_median = float(np.median(positive)) if positive.size else 1.0
    for i, j in np.asarray(heldout_pairs, dtype=int):
        row_i = np.delete(work[int(i)], int(j))
        row_j = np.delete(work[int(j)], int(i))
        candidates = np.concatenate(
            [
                row_i[np.isfinite(row_i) & (row_i > 1e-12)],
                row_j[np.isfinite(row_j) & (row_j > 1e-12)],
            ]
        )
        value = float(np.median(candidates)) if candidates.size else global_median
        work[int(i), int(j)] = work[int(j), int(i)] = value
    np.fill_diagonal(work, 0.0)
    return work


def _classical_mds(distance: np.ndarray, dim: int) -> np.ndarray | None:
    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] <= int(dim):
        return None
    centered = np.eye(distance.shape[0]) - np.ones(distance.shape, dtype=float) / distance.shape[0]
    gram = -0.5 * centered @ (distance**2) @ centered
    values, vectors = np.linalg.eigh((gram + gram.T) * 0.5)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    if values.size < int(dim):
        return None
    values = np.maximum(values[: int(dim)], 1e-12)
    return vectors[:, : int(dim)] * np.sqrt(values)[None, :]


def _spherical_coords(distance: np.ndarray, *, dim: int, radius: float) -> np.ndarray | None:
    gram = np.cos(np.clip(distance / max(float(radius), 1e-12), 0.0, math.pi))
    values, vectors = np.linalg.eigh((gram + gram.T) * 0.5)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    width = int(dim) + 1
    if values.size < width:
        return None
    coords = vectors[:, :width] * np.sqrt(np.maximum(values[:width], 1e-12))[None, :]
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    return coords / np.maximum(norms, 1e-12)


def _spherical_distance(coords: np.ndarray, radius: float) -> np.ndarray:
    cosine = np.clip(coords @ coords.T, -1.0, 1.0)
    distance = float(radius) * np.arccos(cosine)
    np.fill_diagonal(distance, 0.0)
    return distance


def _hyperbolic_prediction(distance: np.ndarray, *, dim: int, radius: float) -> np.ndarray | None:
    lorentz_gram = -np.cosh(np.minimum(distance / max(float(radius), 1e-12), 20.0))
    values, vectors = np.linalg.eigh((lorentz_gram + lorentz_gram.T) * 0.5)
    neg_idx = int(np.argmin(values))
    pos_order = [idx for idx in np.argsort(values)[::-1] if values[idx] > 1e-12 and idx != neg_idx]
    if len(pos_order) < int(dim) or values[neg_idx] >= -1e-12:
        return None
    time = np.abs(np.sqrt(-values[neg_idx]) * vectors[:, neg_idx]) + 1e-9
    space = np.column_stack([np.sqrt(values[idx]) * vectors[:, idx] for idx in pos_order[: int(dim)]])
    inner = -np.outer(time, time) + space @ space.T
    predicted = float(radius) * np.arccosh(np.maximum(-inner, 1.0))
    np.fill_diagonal(predicted, 0.0)
    return predicted


def _rmse(actual: np.ndarray, predicted: np.ndarray, pairs: np.ndarray) -> tuple[float, float]:
    if pairs.size == 0:
        return float("inf"), float("inf")
    i = pairs[:, 0]
    j = pairs[:, 1]
    actual_values = np.asarray(actual[i, j], dtype=float)
    predicted_values = np.asarray(predicted[i, j], dtype=float)
    mask = np.isfinite(actual_values) & np.isfinite(predicted_values)
    if not np.any(mask):
        return float("inf"), float("inf")
    residual = predicted_values[mask] - actual_values[mask]
    denom = float(np.sqrt(np.mean(actual_values[mask] ** 2))) + 1e-12
    squared = (residual / denom) ** 2
    rmse = float(np.sqrt(np.mean(squared)))
    se = float(np.std(np.sqrt(squared)) / math.sqrt(max(1, squared.size)))
    return rmse, se


def _positive_median(distance: np.ndarray) -> float:
    values = distance[np.triu_indices(distance.shape[0], k=1)]
    values = values[np.isfinite(values) & (values > 1e-12)]
    return float(np.median(values)) if values.size else 1.0


def _radius_grid(median: float) -> list[float]:
    base = max(float(median), 1e-6)
    return [0.35 * base, 0.5 * base, base, 2.0 * base, 4.0 * base, 8.0 * base]


def _model_complexity_key(fit: LatentGeometryFit) -> tuple[int, int, float, str]:
    curvature_order = {"S": 0, "E": 1, "H": 2}
    return (int(fit.dimension), curvature_order.get(fit.family, 9), float(fit.heldout_rmse), fit.model)


def _margin(a: LatentGeometryFit | None, b: LatentGeometryFit | None) -> float | None:
    if a is None or b is None:
        return None
    if not (math.isfinite(a.heldout_rmse) and math.isfinite(b.heldout_rmse)):
        return None
    return float(b.heldout_rmse - a.heldout_rmse)


def _empty_selection(reason: str) -> dict[str, Any]:
    return {
        "mode": "strict_neutral_heldout_latent_geometry_selection_v0",
        "observer_or_object_count": 0,
        "fit_count": 0,
        "raw_best_model": None,
        "selected_model": None,
        "h3_selected": False,
        STRICT_NEUTRAL_LATENT_H3_SELECTION_RECEIPT: False,
        "fits": [],
        "blocker": str(reason),
        "physical_claim": False,
        "claim_boundary": "No finite held-out latent-geometry selection was possible.",
    }

