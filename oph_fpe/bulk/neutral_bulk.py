from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform


DEFAULT_NEUTRAL_WEIGHTS = {
    "record": 1.0,
    "counterfactual": 1.0,
    "checkpoint": 0.75,
    "sector": 0.75,
    "repair": 0.75,
    "persistence": 0.25,
}


@dataclass(frozen=True)
class NeutralObserverView:
    observer_id: int
    record_transition_hist: np.ndarray
    counterfactual_hist: np.ndarray
    checkpoint_transition_hist: np.ndarray
    sector_transition_hist: np.ndarray
    repair_response_hist: np.ndarray
    persistence_features: np.ndarray


def build_neutral_observer_views(observer_views: list[dict[str, Any]]) -> list[NeutralObserverView]:
    """Extract support-free observer-visible histories for neutral reconstruction.

    This primary extraction deliberately ignores support nodes, S2 axes, cap
    normals, H3 fitted coordinates, and modular-depth/radial-depth coordinates.
    Those may be used only for post-hoc leakage audits outside this feature
    construction.
    """

    views: list[NeutralObserverView] = []
    for index, view in enumerate(observer_views):
        if view.get("view_type") != "patch_observer":
            continue
        descriptor = view.get("transition_history_descriptor") if isinstance(view.get("transition_history_descriptor"), dict) else {}
        steps = descriptor.get("steps") if isinstance(descriptor.get("steps"), list) else []
        views.append(
            NeutralObserverView(
                observer_id=int(view.get("observer_id", index)),
                record_transition_hist=_hist_or_steps(view, steps, "record_family", 32),
                counterfactual_hist=_normalize_or_zero(view.get("counterfactual_continuation_hist", []), width=16),
                checkpoint_transition_hist=_hist_or_steps(view, steps, "checkpoint_class", 32),
                sector_transition_hist=_hist_or_steps(view, steps, "s3_sector_class", 6),
                repair_response_hist=_hist_or_steps(view, steps, "repair_load_bucket", 16),
                persistence_features=np.asarray(
                    [
                        float(view.get("record_persistence", view.get("transition_history_persistence", 0.0)) or 0.0),
                        float(view.get("sector_persistence", 0.0) or 0.0),
                        float(view.get("stable_fraction", view.get("transition_history_mean_modal_mass", 0.0)) or 0.0),
                    ],
                    dtype=float,
                ),
            )
        )
    return views


def js_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _normalize_or_zero(p)
    q = _normalize_or_zero(q, width=p.size if p.size else None)
    if p.size != q.size:
        width = max(p.size, q.size)
        p = _pad(p, width)
        q = _pad(q, width)
    if p.size == 0:
        return 0.0
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > eps
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], eps))))

    return float(math.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))))


def cosine_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        width = max(x.size, y.size)
        x = _pad(x, width)
        y = _pad(y, width)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= eps:
        return 0.0
    return float(1.0 - np.dot(x, y) / denom)


def neutral_distance(
    a: NeutralObserverView,
    b: NeutralObserverView,
    weights: dict[str, float] | None = None,
) -> float:
    weights = weights or DEFAULT_NEUTRAL_WEIGHTS
    terms = {
        "record": js_distance(a.record_transition_hist, b.record_transition_hist),
        "counterfactual": js_distance(a.counterfactual_hist, b.counterfactual_hist),
        "checkpoint": js_distance(a.checkpoint_transition_hist, b.checkpoint_transition_hist),
        "sector": js_distance(a.sector_transition_hist, b.sector_transition_hist),
        "repair": js_distance(a.repair_response_hist, b.repair_response_hist),
        "persistence": cosine_distance(a.persistence_features, b.persistence_features),
    }
    total = max(float(sum(float(weights.get(key, 0.0)) for key in terms)), 1e-12)
    return float(sum(float(weights.get(key, 0.0)) * terms[key] for key in terms) / total)


def neutral_distance_matrix(
    views: list[NeutralObserverView],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    n = len(views)
    distance = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = neutral_distance(views[i], views[j], weights)
            distance[i, j] = distance[j, i] = value
    return distance


def strict_neutral_dimension_report(distance: np.ndarray) -> dict[str, Any]:
    distance = np.asarray(distance, dtype=float)
    corr = _correlation_dimension(distance)
    mle = _local_mle_dimension(distance)
    spectral = _spectral_dimension_proxy(distance)
    elbow = _diffusion_elbow_dimension(distance)
    estimates = [
        corr.get("estimate"),
        mle.get("median_estimate"),
        spectral.get("estimate"),
    ]
    finite = [float(value) for value in estimates if value is not None and np.isfinite(float(value))]
    agree_gap = max(finite) - min(finite) if len(finite) >= 2 else float("inf")
    estimators_agree_3d = bool(
        len(finite) >= 3
        and all(2.7 <= value <= 3.3 for value in finite)
        and agree_gap <= 0.40
    )
    return {
        "correlation_dimension": corr,
        "local_mle_dimension": mle,
        "spectral_dimension": spectral,
        "diffusion_elbow_dimension": elbow,
        "estimator_pairwise_gap": agree_gap if np.isfinite(agree_gap) else None,
        "estimators_agree_3d": estimators_agree_3d,
        "claim_boundary": (
            "Strict neutral dimension diagnostic from observer-visible records only. It is not sufficient "
            "for strict neutral bulk without leakage, controls, and refinement gates."
        ),
    }


def neutral_leakage_audit(distance: np.ndarray, observer_views: list[dict[str, Any]]) -> dict[str, Any]:
    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    axes: list[np.ndarray] = []
    for view in patch_views:
        axis = np.asarray(view.get("axis", []), dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            axes = []
            break
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            axes = []
            break
        axes.append(axis / norm)
    s2_corr = None
    if axes and len(axes) == np.asarray(distance).shape[0]:
        s2_distance = squareform(pdist(np.vstack(axes), metric="euclidean"))
        s2_corr = _upper_triangle_corr(distance, s2_distance)
    return {
        "s2_distance_correlation": s2_corr,
        "s2_leakage_pass": bool(s2_corr is None or abs(float(s2_corr)) < 0.05),
        "h3_coordinates_used": False,
        "cap_normals_used": False,
        "screen_axes_used_in_primary_distance": False,
        "claim_boundary": "Leakage audit compares primary neutral distance to S2 axes post hoc; axes are not used to build the primary distance.",
    }


def strict_neutral_bulk_receipt(
    dimension: dict[str, Any],
    model_selection: dict[str, Any],
    leakage: dict[str, Any],
    controls: dict[str, Any],
    refinement: dict[str, Any],
) -> dict[str, Any]:
    passed = bool(
        dimension.get("estimators_agree_3d", False)
        and model_selection.get("best_model") == "H3"
        and model_selection.get("h3_beats_s2", False)
        and model_selection.get("h3_beats_h2_h4", False)
        and leakage.get("s2_leakage_pass", False)
        and controls.get("shuffled_records_fail", False)
        and controls.get("shuffled_transition_labels_fail", False)
        and controls.get("planted_2d_returns_2d", False)
        and controls.get("planted_3d_returns_3d", False)
        and controls.get("planted_h3_returns_h3", False)
        and refinement.get("stable_across_64k_256k_1m", False)
    )
    return {
        "receipt": "STRICT_NEUTRAL_BULK_RECEIPT",
        "strict_neutral_bulk": passed,
        "physical_claim": passed,
        "claim_boundary": (
            "Neutral third-person bulk reconstructed from observer-visible records without H3/cap-normal "
            "target features. This receipt is false unless dimension, model-selection, leakage, controls, "
            "and refinement gates all pass."
        ),
    }


def neutral_model_selection(
    distance: np.ndarray,
    *,
    seed: int = 1,
    max_points: int = 512,
    heldout_fraction: float = 0.25,
) -> dict[str, Any]:
    """Compare metric families using neutral distances only.

    This deliberately operates only on the already-built neutral distance
    matrix. It does not read H3 coordinates, S2 axes, cap normals, or screen
    support. For large observer sets it uses a deterministic subsample so the
    diagnostic remains cheap enough to run routinely.
    """

    rng = np.random.default_rng(int(seed))
    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1] or distance.shape[0] < 8:
        return _empty_model_selection("invalid_or_too_small_distance_matrix")
    distance = np.where(np.isfinite(distance), np.maximum(distance, 0.0), 0.0)
    n = distance.shape[0]
    if n > int(max_points):
        indices = np.sort(rng.choice(n, size=int(max_points), replace=False))
        work = distance[np.ix_(indices, indices)]
    else:
        indices = np.arange(n)
        work = distance
    pairs = np.transpose(np.triu_indices(work.shape[0], k=1))
    if pairs.shape[0] < 8:
        return _empty_model_selection("too_few_pair_distances")
    heldout_count = max(8, int(round(float(heldout_fraction) * pairs.shape[0])))
    heldout = pairs[rng.choice(pairs.shape[0], size=min(heldout_count, pairs.shape[0]), replace=False)]
    models = {
        "S2": _spherical_model_stress(work, dim=2, heldout=heldout),
        "E2": _euclidean_model_stress(work, dim=2, heldout=heldout),
        "E3": _euclidean_model_stress(work, dim=3, heldout=heldout),
        "H2": _hyperbolic_model_stress(work, dim=2, heldout=heldout),
        "H3": _hyperbolic_model_stress(work, dim=3, heldout=heldout),
        "H4": _hyperbolic_model_stress(work, dim=4, heldout=heldout),
    }
    finite_models = {
        name: value for name, value in models.items() if np.isfinite(float(value.get("heldout_stress", np.inf)))
    }
    best_model = min(finite_models, key=lambda name: finite_models[name]["heldout_stress"]) if finite_models else None
    h3 = models.get("H3", {})
    s2 = models.get("S2", {})
    e3 = models.get("E3", {})
    h2 = models.get("H2", {})
    h4 = models.get("H4", {})
    h3_stress = float(h3.get("heldout_stress", np.inf))
    return {
        "mode": "strict_neutral_distance_model_selection_v0",
        "observer_count": int(n),
        "fit_observer_count": int(work.shape[0]),
        "subsample_indices": [int(value) for value in indices[: min(indices.size, 2048)]],
        "heldout_pair_count": int(heldout.shape[0]),
        "best_model": best_model,
        "models": models,
        "h3_beats_s2": bool(h3_stress + 0.02 < float(s2.get("heldout_stress", np.inf))),
        "h3_beats_e3": bool(h3_stress + 0.01 < float(e3.get("heldout_stress", np.inf))),
        "h3_beats_h2_h4": bool(
            h3_stress + 0.01 < float(h2.get("heldout_stress", np.inf))
            and h3_stress + 0.01 < float(h4.get("heldout_stress", np.inf))
        ),
        "claim_boundary": (
            "Distance-only model-selection diagnostic. It compares neutral observer-record distances to "
            "low-dimensional metric families, but strict neutral bulk still requires controls and refinement."
        ),
    }


def planted_neutral_control_report(
    *,
    point_count: int = 160,
    seed: int = 1,
    max_points: int = 256,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    planted = {
        "planted_2d": _planted_euclidean(rng, int(point_count), 2),
        "planted_3d": _planted_euclidean(rng, int(point_count), 3),
        "planted_4d": _planted_euclidean(rng, int(point_count), 4),
        "planted_h3": _planted_hyperbolic(rng, int(point_count), 3),
    }
    rows: dict[str, Any] = {}
    for name, distance in planted.items():
        rows[name] = {
            "dimension": strict_neutral_dimension_report(distance),
            "model_selection": neutral_model_selection(distance, seed=seed + 17, max_points=max_points),
        }
    controls = {
        "planted_2d_returns_2d": _dimension_in_range(rows["planted_2d"]["dimension"], 1.7, 2.3),
        "planted_3d_returns_3d": _dimension_in_range(rows["planted_3d"]["dimension"], 2.7, 3.3),
        "planted_4d_returns_4d": _dimension_in_range(rows["planted_4d"]["dimension"], 3.7, 4.3),
        "planted_h3_returns_h3": rows["planted_h3"]["model_selection"].get("best_model") == "H3",
    }
    return {
        "mode": "strict_neutral_planted_controls_v0",
        "point_count": int(point_count),
        "seed": int(seed),
        "rows": rows,
        "controls": controls,
        "claim_boundary": "Synthetic controls for the strict neutral distance/model-selection machinery.",
    }


def strict_neutral_bulk_report(
    observer_views: list[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
    model_selection: dict[str, Any] | None = None,
    controls: dict[str, Any] | None = None,
    refinement: dict[str, Any] | None = None,
    seed: int = 1,
    max_model_points: int = 512,
) -> dict[str, Any]:
    neutral_views = build_neutral_observer_views(observer_views)
    distance = neutral_distance_matrix(neutral_views, weights)
    dimension = strict_neutral_dimension_report(distance)
    leakage = neutral_leakage_audit(distance, observer_views)
    model_selection = model_selection or neutral_model_selection(
        distance,
        seed=seed,
        max_points=max_model_points,
    )
    receipt = strict_neutral_bulk_receipt(
        dimension,
        model_selection,
        leakage,
        controls or {},
        refinement or {},
    )
    return {
        "mode": "strict_neutral_bulk_record_transition_audit",
        "observer_count": len(neutral_views),
        "distance_matrix_shape": list(distance.shape),
        "dimension": dimension,
        "model_selection": model_selection,
        "leakage": leakage,
        "receipt": receipt,
        "strict_neutral_bulk": bool(receipt["strict_neutral_bulk"]),
        "primary_features": [
            "record_transition_hist",
            "counterfactual_hist",
            "checkpoint_transition_hist",
            "sector_transition_hist",
            "repair_response_hist",
            "persistence_features",
        ],
        "forbidden_primary_features": [
            "H3 fitted points",
            "cap normals",
            "S2 axes",
            "screen pixel coordinates",
            "lambda_C target coordinates",
            "radial_depth",
            "modular_depth",
        ],
        "claim_boundary": "Strict neutral audit scaffold; receipt remains false until Pro-defined gates pass.",
    }


def write_strict_neutral_bulk_report(
    run_dir: Path,
    out: Path | None = None,
    *,
    seed: int = 1,
    max_model_points: int = 512,
    planted_control_points: int = 160,
) -> dict[str, Any]:
    run = Path(run_dir)
    observer_path = run / "observer_views.jsonl"
    if not observer_path.exists():
        raise FileNotFoundError(observer_path)
    observer_views = _read_jsonl(observer_path)
    planted = planted_neutral_control_report(
        point_count=int(planted_control_points),
        seed=int(seed) + 101,
        max_points=min(int(max_model_points), int(planted_control_points)),
    )
    control_flags = dict(planted["controls"])
    control_flags.update(
        {
            # These require run-specific shuffled observer-record reruns. Keep
            # false until the run emits those explicit controls.
            "shuffled_records_fail": False,
            "shuffled_transition_labels_fail": False,
        }
    )
    report = strict_neutral_bulk_report(
        observer_views,
        controls=control_flags,
        refinement={"stable_across_64k_256k_1m": False},
        seed=seed,
        max_model_points=max_model_points,
    )
    report["planted_controls"] = planted
    report["source_run_dir"] = str(run)
    report["observer_views_path"] = str(observer_path)
    report["blockers"] = _strict_neutral_blockers(report)
    destination = Path(out) if out is not None else run / "strict_neutral_bulk_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _strict_neutral_blockers(report: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not report["dimension"].get("estimators_agree_3d", False):
        blockers.append("neutral_dimension_estimators_do_not_agree_3d")
    model = report.get("model_selection", {})
    if model.get("best_model") != "H3":
        blockers.append("h3_not_best_neutral_model")
    if not model.get("h3_beats_s2", False):
        blockers.append("h3_does_not_clear_s2_stress_margin")
    if not model.get("h3_beats_h2_h4", False):
        blockers.append("h3_does_not_clear_h2_h4_stress_margin")
    if not report.get("leakage", {}).get("s2_leakage_pass", False):
        blockers.append("s2_leakage_audit_failed")
    receipt = report.get("receipt", {})
    if not receipt.get("strict_neutral_bulk", False):
        blockers.append("strict_neutral_receipt_false_pending_controls_or_refinement")
    return blockers


def _empty_model_selection(reason: str) -> dict[str, Any]:
    return {
        "mode": "strict_neutral_distance_model_selection_v0",
        "best_model": None,
        "models": {},
        "h3_beats_s2": False,
        "h3_beats_e3": False,
        "h3_beats_h2_h4": False,
        "blocker": reason,
    }


def _euclidean_model_stress(distance: np.ndarray, *, dim: int, heldout: np.ndarray) -> dict[str, Any]:
    coords = _classical_mds(distance, dim)
    if coords is None:
        return {"heldout_stress": float("inf"), "fit_dimension": int(dim)}
    predicted = squareform(pdist(coords, metric="euclidean"))
    return _stress_report(distance, predicted, heldout, fit_dimension=dim)


def _spherical_model_stress(distance: np.ndarray, *, dim: int, heldout: np.ndarray) -> dict[str, Any]:
    median = _positive_median(distance)
    best: dict[str, Any] | None = None
    for radius in _radius_grid(median):
        gram = np.cos(np.clip(distance / max(radius, 1e-12), 0.0, math.pi))
        coords = _positive_spectral_coords(gram, dim + 1)
        if coords is None:
            continue
        norms = np.linalg.norm(coords, axis=1)
        coords = coords / np.maximum(norms[:, None], 1e-12)
        cosine = np.clip(coords @ coords.T, -1.0, 1.0)
        predicted = radius * np.arccos(cosine)
        current = _stress_report(distance, predicted, heldout, fit_dimension=dim, radius=radius)
        if best is None or current["heldout_stress"] < best["heldout_stress"]:
            best = current
    return best or {"heldout_stress": float("inf"), "fit_dimension": int(dim)}


def _hyperbolic_model_stress(distance: np.ndarray, *, dim: int, heldout: np.ndarray) -> dict[str, Any]:
    median = _positive_median(distance)
    best: dict[str, Any] | None = None
    for radius in _radius_grid(median):
        predicted = _hyperbolic_indefinite_reconstruction(distance, dim=dim, radius=radius)
        if predicted is None:
            continue
        current = _stress_report(distance, predicted, heldout, fit_dimension=dim, radius=radius)
        if best is None or current["heldout_stress"] < best["heldout_stress"]:
            best = current
    return best or {"heldout_stress": float("inf"), "fit_dimension": int(dim)}


def _classical_mds(distance: np.ndarray, dim: int) -> np.ndarray | None:
    distance = np.asarray(distance, dtype=float)
    n = distance.shape[0]
    if n <= dim:
        return None
    squared = distance**2
    centered = np.eye(n) - np.ones((n, n), dtype=float) / n
    gram = -0.5 * centered @ squared @ centered
    values, vectors = np.linalg.eigh((gram + gram.T) * 0.5)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    positive = values[:dim]
    if positive.size < dim or np.any(positive <= 1e-12):
        positive = np.maximum(positive, 1e-12)
    return vectors[:, :dim] * np.sqrt(positive)[None, :]


def _positive_spectral_coords(gram: np.ndarray, dim: int) -> np.ndarray | None:
    values, vectors = np.linalg.eigh((gram + gram.T) * 0.5)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    if values.size < dim:
        return None
    values = np.maximum(values[:dim], 1e-12)
    return vectors[:, :dim] * np.sqrt(values)[None, :]


def _hyperbolic_indefinite_reconstruction(distance: np.ndarray, *, dim: int, radius: float) -> np.ndarray | None:
    lorentz_gram = -np.cosh(np.minimum(distance / max(radius, 1e-12), 20.0))
    values, vectors = np.linalg.eigh((lorentz_gram + lorentz_gram.T) * 0.5)
    neg_idx = int(np.argmin(values))
    pos_order = [idx for idx in np.argsort(values)[::-1] if values[idx] > 1e-12 and idx != neg_idx]
    if len(pos_order) < dim or values[neg_idx] >= -1e-12:
        return None
    time = np.sqrt(-values[neg_idx]) * vectors[:, neg_idx]
    space = np.column_stack([np.sqrt(values[idx]) * vectors[:, idx] for idx in pos_order[:dim]])
    time = np.abs(time) + 1e-9
    inner = -np.outer(time, time) + space @ space.T
    cosh_arg = np.maximum(-inner, 1.0)
    predicted = radius * np.arccosh(cosh_arg)
    np.fill_diagonal(predicted, 0.0)
    return predicted


def _stress_report(
    actual: np.ndarray,
    predicted: np.ndarray,
    heldout: np.ndarray,
    *,
    fit_dimension: int,
    radius: float | None = None,
) -> dict[str, Any]:
    i = heldout[:, 0]
    j = heldout[:, 1]
    a = actual[i, j]
    p = predicted[i, j]
    mask = np.isfinite(a) & np.isfinite(p)
    if not np.any(mask):
        stress = float("inf")
    else:
        stress = float(np.sqrt(np.mean((p[mask] - a[mask]) ** 2)) / (np.sqrt(np.mean(a[mask] ** 2)) + 1e-12))
    return {
        "fit_dimension": int(fit_dimension),
        "radius": None if radius is None else float(radius),
        "heldout_stress": stress,
    }


def _radius_grid(median: float) -> list[float]:
    base = max(float(median), 1e-6)
    return [0.5 * base, base, 2.0 * base, 4.0 * base, 8.0 * base]


def _positive_median(distance: np.ndarray) -> float:
    pairs = _upper_triangle(distance)
    pairs = pairs[np.isfinite(pairs) & (pairs > 1e-12)]
    return float(np.median(pairs)) if pairs.size else 1.0


def _planted_euclidean(rng: np.random.Generator, point_count: int, dim: int) -> np.ndarray:
    coords = rng.normal(size=(int(point_count), int(dim)))
    coords /= max(float(np.sqrt(dim)), 1.0)
    return squareform(pdist(coords, metric="euclidean"))


def _planted_hyperbolic(rng: np.random.Generator, point_count: int, dim: int) -> np.ndarray:
    directions = rng.normal(size=(int(point_count), int(dim)))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    radii = rng.gamma(shape=float(dim), scale=0.4, size=int(point_count))
    space = np.sinh(radii)[:, None] * directions
    time = np.cosh(radii)
    inner = -np.outer(time, time) + space @ space.T
    distance = np.arccosh(np.maximum(-inner, 1.0))
    np.fill_diagonal(distance, 0.0)
    return distance


def _dimension_in_range(report: dict[str, Any], low: float, high: float) -> bool:
    candidates = [
        ((report.get("correlation_dimension") or {}).get("estimate")),
        ((report.get("local_mle_dimension") or {}).get("median_estimate")),
        ((report.get("spectral_dimension") or {}).get("estimate")),
    ]
    finite = [float(value) for value in candidates if value is not None and np.isfinite(float(value))]
    return bool(finite and low <= float(np.median(finite)) <= high)


def _hist_or_steps(view: dict[str, Any], steps: list[Any], field: str, modulus: int) -> np.ndarray:
    histograms = view.get("transition_history_histograms")
    if isinstance(histograms, dict):
        for key in (field, f"{field}_path"):
            histogram = histograms.get(key)
            if isinstance(histogram, dict) and histogram:
                return _histogram_dict_to_vector(histogram, modulus)
    values = [
        int(step[field])
        for step in steps
        if isinstance(step, dict) and field in step and _safe_int(step[field]) is not None
    ]
    return transition_histogram(values, modulus=modulus)


def transition_histogram(values: list[int], *, modulus: int) -> np.ndarray:
    width = max(1, int(modulus))
    hist = np.zeros(width, dtype=float)
    for value in values:
        hist[int(value) % width] += 1.0
    return _normalize_or_zero(hist, width=width)


def _histogram_dict_to_vector(histogram: dict[str, Any], width: int) -> np.ndarray:
    vector = np.zeros(max(1, int(width)), dtype=float)
    for key, value in histogram.items():
        parsed = _safe_int(key)
        if parsed is None:
            continue
        try:
            mass = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(mass):
            vector[int(parsed) % vector.size] += mass
    return _normalize_or_zero(vector, width=vector.size)


def _normalize_or_zero(values: Any, width: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1) if values is not None else np.zeros(0, dtype=float)
    if width is not None:
        array = _pad(array, int(width))[: int(width)]
    if array.size == 0:
        return np.zeros(int(width or 0), dtype=float)
    array = np.where(np.isfinite(array), np.maximum(array, 0.0), 0.0)
    total = float(np.sum(array))
    return array / total if total > 1e-12 else np.zeros_like(array, dtype=float)


def _pad(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size >= int(width):
        return values[: int(width)]
    out = np.zeros(int(width), dtype=float)
    out[: values.size] = values
    return out


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _upper_triangle(distance: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] < 2:
        return np.zeros(0, dtype=float)
    return distance[np.triu_indices(distance.shape[0], k=1)]


def _upper_triangle_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    av = _upper_triangle(a)
    bv = _upper_triangle(b)
    if av.size != bv.size or av.size < 2 or float(np.std(av)) < 1e-12 or float(np.std(bv)) < 1e-12:
        return None
    corr = float(np.corrcoef(av, bv)[0, 1])
    return corr if np.isfinite(corr) else None


def _correlation_dimension(distance: np.ndarray) -> dict[str, Any]:
    pairs = _upper_triangle(distance)
    pairs = pairs[np.isfinite(pairs) & (pairs > 1e-12)]
    if pairs.size < 16:
        return {"estimate": None, "points_used": 0}
    radii = np.quantile(pairs, np.linspace(0.02, 0.20, 10))
    counts = np.asarray([np.mean(pairs <= radius) for radius in radii], dtype=float)
    mask = (radii > 1e-12) & (counts > 1e-12) & np.isfinite(radii) & np.isfinite(counts)
    if int(np.sum(mask)) < 3:
        return {"estimate": None, "points_used": int(np.sum(mask))}
    estimate = float(np.polyfit(np.log(radii[mask]), np.log(counts[mask]), 1)[0])
    return {"estimate": estimate if np.isfinite(estimate) else None, "points_used": int(np.sum(mask))}


def _local_mle_dimension(distance: np.ndarray, k_values: tuple[int, ...] = (8, 12, 16, 24, 32)) -> dict[str, Any]:
    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] < 8:
        return {"median_estimate": None, "by_k": {}}
    sorted_dist = np.sort(np.where(distance > 1e-12, distance, np.inf), axis=1)
    rows: dict[str, float | None] = {}
    for k in k_values:
        if distance.shape[0] <= k + 1:
            continue
        neighbors = sorted_dist[:, :k]
        rk = neighbors[:, -1]
        valid = np.isfinite(rk) & (rk > 1e-12) & np.all(np.isfinite(neighbors[:, :-1]), axis=1)
        if not np.any(valid):
            rows[str(k)] = None
            continue
        logs = np.log(rk[valid, None] / np.maximum(neighbors[valid, :-1], 1e-12))
        denom = np.mean(np.sum(logs, axis=1) / max(k - 1, 1))
        estimate = float(1.0 / denom) if denom > 1e-12 else float("nan")
        rows[str(k)] = estimate if np.isfinite(estimate) else None
    finite = [float(value) for value in rows.values() if value is not None and np.isfinite(float(value))]
    return {"median_estimate": float(np.median(finite)) if finite else None, "by_k": rows}


def _spectral_dimension_proxy(distance: np.ndarray) -> dict[str, Any]:
    distance = np.asarray(distance, dtype=float)
    n = distance.shape[0] if distance.ndim == 2 else 0
    if n < 8:
        return {"estimate": None, "tau_count": 0}
    pairs = _upper_triangle(distance)
    pairs = pairs[np.isfinite(pairs) & (pairs > 1e-12)]
    if pairs.size < 16:
        return {"estimate": None, "tau_count": 0}
    sigma = float(np.median(pairs))
    kernel = np.exp(-(distance**2) / max(sigma**2, 1e-12))
    np.fill_diagonal(kernel, 0.0)
    degrees = np.sum(kernel, axis=1)
    transition = kernel / np.maximum(degrees[:, None], 1e-12)
    current = np.eye(n)
    taus: list[float] = []
    returns: list[float] = []
    for tau in range(1, 7):
        current = current @ transition
        ret = float(np.trace(current) / n)
        if ret > 1e-15 and np.isfinite(ret):
            taus.append(float(tau))
            returns.append(ret)
    if len(taus) < 3:
        return {"estimate": None, "tau_count": len(taus)}
    slope = float(np.polyfit(np.log(taus), np.log(returns), 1)[0])
    estimate = -2.0 * slope
    return {"estimate": estimate if np.isfinite(estimate) else None, "tau_count": len(taus)}


def _diffusion_elbow_dimension(distance: np.ndarray) -> dict[str, Any]:
    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] < 8:
        return {"estimate": None}
    pairs = _upper_triangle(distance)
    pairs = pairs[np.isfinite(pairs) & (pairs > 1e-12)]
    if pairs.size < 16:
        return {"estimate": None}
    sigma = float(np.median(pairs))
    kernel = np.exp(-(distance**2) / max(sigma**2, 1e-12))
    values = np.linalg.eigvalsh(kernel)
    values = np.sort(np.maximum(values, 0.0))[::-1]
    if values.size < 4 or values[0] <= 1e-12:
        return {"estimate": None}
    normalized = values / values[0]
    estimate = int(np.sum(normalized > 0.1))
    return {"estimate": estimate, "eigenvalues": [float(value) for value in normalized[:8]]}
