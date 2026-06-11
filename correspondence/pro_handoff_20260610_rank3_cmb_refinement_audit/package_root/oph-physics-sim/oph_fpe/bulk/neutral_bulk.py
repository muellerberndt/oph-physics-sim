from __future__ import annotations

import json
import math
import hashlib
import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform


DEFAULT_NEUTRAL_WEIGHTS = {
    "record": 1.0,
    "record_signature": 0.9,
    "object_packet": 0.9,
    "counterfactual": 1.0,
    "checkpoint": 0.75,
    "sector": 0.75,
    "repair": 0.75,
    "repair_spectrum": 0.65,
    "modular_response": 0.75,
    "prime_geometric_modular": 0.9,
    "prime_geometric_control_quotient": 0.9,
    "prime_geometric_rank3": 0.9,
    "prime_geometric_rank4": 0.9,
    "prime_geometric_rank8": 0.9,
    "prime_geometric_rank16": 0.9,
    "prime_geometric_rank32": 0.9,
    "prime_geometric_control_quotient_rank3": 0.9,
    "prime_geometric_control_quotient_rank4": 0.9,
    "prime_geometric_control_quotient_rank8": 0.9,
    "prime_geometric_control_quotient_rank16": 0.9,
    "prime_geometric_control_quotient_rank32": 0.9,
    "support_visible_modular": 0.8,
    "repair_modular": 0.35,
    "transition_token": 0.9,
    "transition_token_persistent": 0.65,
    "transition_affinity": 0.75,
    "persistence": 0.25,
    "scalar_readout": 0.35,
}

MODEL_SELECTION_ABS_TOLERANCE = 0.01
MODEL_SELECTION_REL_TOLERANCE = 0.08

NEUTRAL_PROFILE_WEIGHTS: dict[str, dict[str, float] | None] = {
    "all_observer_visible": None,
    "scalar_only": {"scalar_readout": 1.0},
    "transition_core": {
        "record": 1.0,
        "checkpoint": 0.75,
        "sector": 0.75,
        "repair": 0.75,
        "transition_token": 1.0,
        "transition_token_persistent": 0.5,
    },
    "scalar_response": {
        "scalar_readout": 1.0,
        "repair_spectrum": 0.75,
        "modular_response": 0.75,
        "counterfactual": 0.5,
        "persistence": 0.5,
    },
    "prime_geometric_modular": {"prime_geometric_modular": 1.0},
    "prime_geometric_control_quotient": {"prime_geometric_control_quotient": 1.0},
    "prime_geometric_rank3": {"prime_geometric_rank3": 1.0},
    "prime_geometric_rank4": {"prime_geometric_rank4": 1.0},
    "prime_geometric_rank8": {"prime_geometric_rank8": 1.0},
    "prime_geometric_rank16": {"prime_geometric_rank16": 1.0},
    "prime_geometric_rank32": {"prime_geometric_rank32": 1.0},
    "prime_geometric_control_quotient_rank3": {"prime_geometric_control_quotient_rank3": 1.0},
    "prime_geometric_control_quotient_rank4": {"prime_geometric_control_quotient_rank4": 1.0},
    "prime_geometric_control_quotient_rank8": {"prime_geometric_control_quotient_rank8": 1.0},
    "prime_geometric_control_quotient_rank16": {"prime_geometric_control_quotient_rank16": 1.0},
    "prime_geometric_control_quotient_rank32": {"prime_geometric_control_quotient_rank32": 1.0},
    "prime_geometric_modular_counterfactual": {
        "prime_geometric_modular": 1.0,
        "counterfactual": 0.35,
        "persistence": 0.2,
    },
    "prime_geometric_control_quotient_counterfactual": {
        "prime_geometric_control_quotient": 1.0,
        "counterfactual": 0.35,
        "persistence": 0.2,
    },
    "support_visible_modular": {"support_visible_modular": 1.0},
    "support_visible_modular_scalar": {
        "support_visible_modular": 1.0,
        "scalar_readout": 0.35,
    },
    "repair_modular_only": {"repair_modular": 1.0},
}


@dataclass(frozen=True)
class NeutralObserverView:
    observer_id: int
    record_transition_hist: np.ndarray
    record_signature_hist: np.ndarray
    object_packet_hist: np.ndarray
    counterfactual_hist: np.ndarray
    checkpoint_transition_hist: np.ndarray
    sector_transition_hist: np.ndarray
    repair_response_hist: np.ndarray
    repair_response_spectrum: np.ndarray
    modular_response_hist: np.ndarray
    prime_geometric_modular_spectrum: np.ndarray
    prime_geometric_control_quotient_spectrum: np.ndarray
    support_visible_modular_spectrum: np.ndarray
    repair_modular_spectrum: np.ndarray
    transition_token_hist: np.ndarray
    transition_token_persistent_hist: np.ndarray
    transition_affinity_hist: np.ndarray
    persistence_features: np.ndarray
    scalar_readout_features: np.ndarray


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
                record_signature_hist=_histogram_dict_to_vector(view.get("record_signature_histogram", {}), 64),
                object_packet_hist=_histogram_dict_to_vector(view.get("object_packet_histogram", {}), 64),
                counterfactual_hist=_normalize_or_zero(view.get("counterfactual_continuation_hist", []), width=16),
                checkpoint_transition_hist=_hist_or_steps(view, steps, "checkpoint_class", 32),
                sector_transition_hist=_hist_or_steps(view, steps, "s3_sector_class", 6),
                repair_response_hist=_hist_or_steps(view, steps, "repair_load_bucket", 16),
                repair_response_spectrum=_signed_vector_or_zero(view.get("repair_response_spectrum", []), width=32),
                modular_response_hist=_nested_histogram_to_vector(view.get("modular_response_histograms", {}), 64),
                prime_geometric_modular_spectrum=_signed_vector_or_zero(
                    view.get("prime_geometric_modular_spectrum", []),
                    width=64,
                ),
                prime_geometric_control_quotient_spectrum=_signed_vector_or_zero(
                    view.get("prime_geometric_control_quotient_spectrum", []),
                    width=64,
                ),
                support_visible_modular_spectrum=_signed_vector_or_zero(
                    view.get("support_visible_modular_spectrum", []),
                    width=64,
                ),
                repair_modular_spectrum=_signed_vector_or_zero(
                    view.get("repair_modular_spectrum", []),
                    width=32,
                ),
                transition_token_hist=_transition_history_hist(view, "local_transition_token", 128),
                transition_token_persistent_hist=_transition_history_hist(
                    view,
                    "local_transition_token_persistent",
                    128,
                ),
                transition_affinity_hist=_nested_histogram_to_vector(view.get("transition_affinity_histograms", {}), 96),
                persistence_features=np.asarray(
                    [
                        float(view.get("record_persistence", view.get("transition_history_persistence", 0.0)) or 0.0),
                        float(view.get("sector_persistence", 0.0) or 0.0),
                        float(view.get("stable_fraction", view.get("transition_history_mean_modal_mass", 0.0)) or 0.0),
                    ],
                    dtype=float,
                ),
                scalar_readout_features=np.asarray(
                    [
                        float(view.get("committed_fraction", 0.0) or 0.0),
                        float(view.get("record_stability_mean", 0.0) or 0.0),
                        float(view.get("repair_load_mean", 0.0) or 0.0),
                        float(view.get("mismatch_density_mean", 0.0) or 0.0),
                        float(view.get("visible_signature_entropy", 0.0) or 0.0),
                        float(view.get("counterfactual_stability", 0.0) or 0.0),
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


def scaled_l2_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        width = max(x.size, y.size)
        x = _pad(x, width)
        y = _pad(y, width)
    x = np.where(np.isfinite(x), x, 0.0)
    y = np.where(np.isfinite(y), y, 0.0)
    denom = float(np.linalg.norm(x) + np.linalg.norm(y))
    if denom <= eps:
        return 0.0
    return float(np.linalg.norm(x - y) / (denom + eps))


def neutral_distance(
    a: NeutralObserverView,
    b: NeutralObserverView,
    weights: dict[str, float] | None = None,
) -> float:
    weights = weights or DEFAULT_NEUTRAL_WEIGHTS
    terms = {
        "record": js_distance(a.record_transition_hist, b.record_transition_hist),
        "record_signature": js_distance(a.record_signature_hist, b.record_signature_hist),
        "object_packet": js_distance(a.object_packet_hist, b.object_packet_hist),
        "counterfactual": js_distance(a.counterfactual_hist, b.counterfactual_hist),
        "checkpoint": js_distance(a.checkpoint_transition_hist, b.checkpoint_transition_hist),
        "sector": js_distance(a.sector_transition_hist, b.sector_transition_hist),
        "repair": js_distance(a.repair_response_hist, b.repair_response_hist),
        "repair_spectrum": cosine_distance(a.repair_response_spectrum, b.repair_response_spectrum),
        "modular_response": js_distance(a.modular_response_hist, b.modular_response_hist),
        "prime_geometric_modular": cosine_distance(
            a.prime_geometric_modular_spectrum,
            b.prime_geometric_modular_spectrum,
        ),
        "prime_geometric_control_quotient": cosine_distance(
            a.prime_geometric_control_quotient_spectrum,
            b.prime_geometric_control_quotient_spectrum,
        ),
        "prime_geometric_rank3": cosine_distance(
            a.prime_geometric_modular_spectrum[:3],
            b.prime_geometric_modular_spectrum[:3],
        ),
        "prime_geometric_rank4": cosine_distance(
            a.prime_geometric_modular_spectrum[:4],
            b.prime_geometric_modular_spectrum[:4],
        ),
        "prime_geometric_rank8": cosine_distance(
            a.prime_geometric_modular_spectrum[:8],
            b.prime_geometric_modular_spectrum[:8],
        ),
        "prime_geometric_rank16": cosine_distance(
            a.prime_geometric_modular_spectrum[:16],
            b.prime_geometric_modular_spectrum[:16],
        ),
        "prime_geometric_rank32": cosine_distance(
            a.prime_geometric_modular_spectrum[:32],
            b.prime_geometric_modular_spectrum[:32],
        ),
        "prime_geometric_control_quotient_rank3": cosine_distance(
            a.prime_geometric_control_quotient_spectrum[:3],
            b.prime_geometric_control_quotient_spectrum[:3],
        ),
        "prime_geometric_control_quotient_rank4": cosine_distance(
            a.prime_geometric_control_quotient_spectrum[:4],
            b.prime_geometric_control_quotient_spectrum[:4],
        ),
        "prime_geometric_control_quotient_rank8": cosine_distance(
            a.prime_geometric_control_quotient_spectrum[:8],
            b.prime_geometric_control_quotient_spectrum[:8],
        ),
        "prime_geometric_control_quotient_rank16": cosine_distance(
            a.prime_geometric_control_quotient_spectrum[:16],
            b.prime_geometric_control_quotient_spectrum[:16],
        ),
        "prime_geometric_control_quotient_rank32": cosine_distance(
            a.prime_geometric_control_quotient_spectrum[:32],
            b.prime_geometric_control_quotient_spectrum[:32],
        ),
        "support_visible_modular": cosine_distance(
            a.support_visible_modular_spectrum,
            b.support_visible_modular_spectrum,
        ),
        "repair_modular": cosine_distance(a.repair_modular_spectrum, b.repair_modular_spectrum),
        "transition_token": js_distance(a.transition_token_hist, b.transition_token_hist),
        "transition_token_persistent": js_distance(
            a.transition_token_persistent_hist,
            b.transition_token_persistent_hist,
        ),
        "transition_affinity": js_distance(a.transition_affinity_hist, b.transition_affinity_hist),
        "persistence": scaled_l2_distance(a.persistence_features, b.persistence_features),
        "scalar_readout": scaled_l2_distance(a.scalar_readout_features, b.scalar_readout_features),
    }
    total = max(float(sum(float(weights.get(key, 0.0)) for key in terms)), 1e-12)
    value = float(sum(float(weights.get(key, 0.0)) * terms[key] for key in terms) / total)
    return float(max(0.0, value))


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
    ]
    finite = [float(value) for value in estimates if value is not None and np.isfinite(float(value))]
    agree_gap = max(finite) - min(finite) if len(finite) >= 2 else float("inf")
    median_dimension = float(np.median(finite)) if len(finite) >= 2 else None
    all_estimators_individually_3d = bool(
        len(finite) >= 2
        and all(2.7 <= value <= 3.3 for value in finite)
        and agree_gap <= 0.40
    )
    estimators_agree_3d = bool(
        len(finite) >= 2
        and median_dimension is not None
        and 2.7 <= median_dimension <= 3.3
        and agree_gap <= 0.50
    )
    return {
        "correlation_dimension": corr,
        "local_mle_dimension": mle,
        "spectral_dimension": spectral,
        "diffusion_elbow_dimension": elbow,
        "dimension_gate_estimators": ["correlation_dimension", "local_mle_dimension"],
        "estimator_pairwise_gap": agree_gap if np.isfinite(agree_gap) else None,
        "median_dimension_estimate": median_dimension,
        "all_estimators_individually_3d": all_estimators_individually_3d,
        "estimators_agree_3d": estimators_agree_3d,
        "claim_boundary": (
            "Strict neutral dimension diagnostic from observer-visible records only. The finite-regulator "
            "gate uses the median of correlation and local-MLE estimators plus a pairwise-gap bound, because "
            "planted 3D controls show a small finite-sample low bias in the correlation estimator. The "
            "spectral proxy is reported but not gated because it is not calibrated on the finite planted "
            "controls. This is not sufficient for strict neutral bulk without leakage, controls, and "
            "refinement gates."
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
    raw_best_model = min(finite_models, key=lambda name: finite_models[name]["heldout_stress"]) if finite_models else None
    selected = _parsimonious_model_selection(finite_models)
    h3 = models.get("H3", {})
    s2 = models.get("S2", {})
    e3 = models.get("E3", {})
    h2 = models.get("H2", {})
    h4 = models.get("H4", {})
    h3_stress = float(h3.get("heldout_stress", np.inf))
    h4_stress = float(h4.get("heldout_stress", np.inf))
    compatibility = _model_compatibility_tolerance(h4_stress)
    h3_h4_compatible = bool(np.isfinite(h3_stress) and np.isfinite(h4_stress) and h3_stress <= h4_stress + compatibility)
    return {
        "mode": "strict_neutral_distance_model_selection_v0",
        "observer_count": int(n),
        "fit_observer_count": int(work.shape[0]),
        "subsample_indices": [int(value) for value in indices[: min(indices.size, 2048)]],
        "heldout_pair_count": int(heldout.shape[0]),
        "raw_best_model": raw_best_model,
        "selected_model": selected["selected_model"],
        "best_model": selected["selected_model"],
        "selection_rule": selected["selection_rule"],
        "selection_abs_tolerance": MODEL_SELECTION_ABS_TOLERANCE,
        "selection_rel_tolerance": MODEL_SELECTION_REL_TOLERANCE,
        "models": models,
        "h3_beats_s2": bool(h3_stress + 0.02 < float(s2.get("heldout_stress", np.inf))),
        "h3_beats_e3": bool(h3_stress + 0.01 < float(e3.get("heldout_stress", np.inf))),
        "h3_beats_h2_h4": bool(
            selected["selected_model"] == "H3"
            and h3_stress < float(h2.get("heldout_stress", np.inf))
            and h3_h4_compatible
        ),
        "h3_h4_compatible": h3_h4_compatible,
        "h3_selected_by_parsimony": selected["selected_model"] == "H3",
        "raw_lowest_stress_claim_boundary": (
            "raw_best_model is reported for audit only. Since higher-dimensional metric families can "
            "strictly contain lower-dimensional fits, selected_model uses the declared parsimony rule."
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
        "planted_4d_returns_4d": bool(
            _dimension_in_range(rows["planted_4d"]["dimension"], 3.3, 4.4)
            and rows["planted_4d"]["model_selection"].get("selected_model") == "H4"
        ),
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
        "controls": controls or {},
        "refinement": refinement or {},
        "strict_neutral_bulk": bool(receipt["strict_neutral_bulk"]),
        "primary_features": [
            "record_transition_hist",
            "record_signature_hist",
            "object_packet_hist",
            "counterfactual_hist",
            "checkpoint_transition_hist",
            "sector_transition_hist",
            "repair_response_hist",
            "repair_response_spectrum",
            "modular_response_hist",
            "prime_geometric_modular_spectrum",
            "prime_geometric_control_quotient_spectrum",
            "support_visible_modular_spectrum",
            "repair_modular_spectrum",
            "transition_token_hist",
            "transition_token_persistent_hist",
            "transition_affinity_hist",
            "persistence_features",
            "scalar_readout_features",
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
    run_controls = shuffled_neutral_control_report(
        observer_views,
        seed=int(seed) + 303,
        max_model_points=min(int(max_model_points), 96),
    )
    control_flags = dict(planted["controls"])
    control_flags.update(run_controls["controls"])
    report = strict_neutral_bulk_report(
        observer_views,
        controls=control_flags,
        refinement={"stable_across_64k_256k_1m": False},
        seed=seed,
        max_model_points=max_model_points,
    )
    report["planted_controls"] = planted
    report["shuffled_controls"] = run_controls
    report["source_run_dir"] = str(run)
    report["observer_views_path"] = str(observer_path)
    report["blockers"] = _strict_neutral_blockers(report)
    destination = Path(out) if out is not None else run / "strict_neutral_bulk_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def neutral_profile_audit_report(
    observer_views: list[dict[str, Any]],
    *,
    seed: int = 1,
    sample_count: int = 256,
    max_model_points: int = 128,
    profiles: dict[str, dict[str, float] | None] | None = None,
) -> dict[str, Any]:
    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    if not patch_views:
        return {
            "mode": "neutral_distance_profile_audit_v0",
            "observer_count": 0,
            "sampled_observer_count": 0,
            "profile_rows": [],
            "claim_boundary": "No patch_observer rows were available.",
        }
    rng = np.random.default_rng(int(seed))
    sample_count = min(len(patch_views), max(8, int(sample_count)))
    if len(patch_views) > sample_count:
        sample_indices = np.sort(rng.choice(len(patch_views), size=sample_count, replace=False))
        sampled = [patch_views[int(index)] for index in sample_indices]
    else:
        sample_indices = np.arange(len(patch_views))
        sampled = patch_views
    neutral_views = build_neutral_observer_views(sampled)
    profile_map = profiles or NEUTRAL_PROFILE_WEIGHTS
    rows = []
    for profile_name, weights in profile_map.items():
        distance = neutral_distance_matrix(neutral_views, weights)
        dimension = strict_neutral_dimension_report(distance)
        model = neutral_model_selection(
            distance,
            seed=int(seed),
            max_points=min(int(max_model_points), len(neutral_views)),
        )
        leakage = neutral_leakage_audit(distance, sampled)
        rows.append(
            {
                "profile": profile_name,
                "weights": weights or DEFAULT_NEUTRAL_WEIGHTS,
                "dimension": dimension,
                "model_selection": model,
                "leakage": leakage,
                "strict_3d_ready": bool(
                    dimension.get("estimators_agree_3d", False)
                    and model.get("best_model") == "H3"
                    and model.get("h3_beats_s2", False)
                    and model.get("h3_beats_h2_h4", False)
                    and leakage.get("s2_leakage_pass", False)
                ),
                "blockers": _neutral_profile_blockers(dimension, model, leakage),
            }
        )
    return {
        "mode": "neutral_distance_profile_audit_v0",
        "observer_count": len(patch_views),
        "sampled_observer_count": len(sampled),
        "sample_indices": [int(value) for value in sample_indices[: min(2048, len(sample_indices))]],
        "seed": int(seed),
        "max_model_points": int(max_model_points),
        "profile_rows": rows,
        "strict_neutral_bulk": False,
        "physical_claim": False,
        "claim_boundary": (
            "Profile audit for neutral observer-record distances. It diagnoses which support-visible "
            "feature quotient is overcomplete, undercomplete, or geometry-leaky. It is not a bulk proof; "
            "strict neutral bulk still requires the full dimension, leakage, controls, and refinement gates."
        ),
    }


def write_neutral_profile_audit_report(
    run_dir: Path,
    out: Path | None = None,
    *,
    seed: int = 1,
    sample_count: int = 256,
    max_model_points: int = 128,
    profiles: dict[str, dict[str, float] | None] | None = None,
) -> dict[str, Any]:
    run = Path(run_dir)
    observer_path = run / "observer_views.jsonl"
    if not observer_path.exists():
        raise FileNotFoundError(observer_path)
    report = neutral_profile_audit_report(
        _read_jsonl(observer_path),
        seed=seed,
        sample_count=sample_count,
        max_model_points=max_model_points,
        profiles=profiles,
    )
    report["source_run_dir"] = str(run)
    report["observer_views_path"] = str(observer_path)
    destination = Path(out) if out is not None else run / "neutral_profile_audit_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def prime_geometric_rank_sweep_report(
    observer_views: list[dict[str, Any]],
    *,
    ranks: list[int] | tuple[int, ...] = tuple(range(2, 17)),
    seed: int = 1,
    sample_count: int = 256,
    max_model_points: int = 128,
) -> dict[str, Any]:
    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    if not patch_views:
        return {
            "mode": "prime_geometric_rank_sweep_v0",
            "observer_count": 0,
            "sampled_observer_count": 0,
            "rows": [],
            "claim_boundary": "No patch_observer rows were available.",
        }
    rng = np.random.default_rng(int(seed))
    sample_count = min(len(patch_views), max(8, int(sample_count)))
    if len(patch_views) > sample_count:
        sample_indices = np.sort(rng.choice(len(patch_views), size=sample_count, replace=False))
        sampled = [patch_views[int(index)] for index in sample_indices]
    else:
        sample_indices = np.arange(len(patch_views))
        sampled = patch_views
    neutral_views = build_neutral_observer_views(sampled)
    rows: list[dict[str, Any]] = []
    quotient_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    quotient_coordinate_rows: list[dict[str, Any]] = []
    for rank in sorted({int(value) for value in ranks if int(value) > 0}):
        distance = _prime_geometric_prefix_distance_matrix(
            neutral_views,
            rank,
            spectrum_name="prime_geometric_modular_spectrum",
        )
        dimension = strict_neutral_dimension_report(distance)
        model = neutral_model_selection(
            distance,
            seed=int(seed),
            max_points=min(int(max_model_points), len(neutral_views)),
        )
        leakage = neutral_leakage_audit(distance, sampled)
        rows.append(
            {
                "rank": int(rank),
                "dimension": dimension,
                "model_selection": model,
                "leakage": leakage,
                "dimension_3d_window": bool(dimension.get("estimators_agree_3d", False)),
                "strict_3d_ready": bool(
                    dimension.get("estimators_agree_3d", False)
                    and model.get("best_model") == "H3"
                    and model.get("h3_beats_s2", False)
                    and model.get("h3_beats_h2_h4", False)
                    and leakage.get("s2_leakage_pass", False)
                ),
                "blockers": _neutral_profile_blockers(dimension, model, leakage),
            }
        )
        coordinate_distance = _prime_geometric_prefix_coordinate_distance_matrix(
            neutral_views,
            rank,
            spectrum_name="prime_geometric_modular_spectrum",
        )
        coordinate_dimension = strict_neutral_dimension_report(coordinate_distance)
        coordinate_model = neutral_model_selection(
            coordinate_distance,
            seed=int(seed),
            max_points=min(int(max_model_points), len(neutral_views)),
        )
        coordinate_leakage = neutral_leakage_audit(coordinate_distance, sampled)
        coordinate_rows.append(
            {
                "rank": int(rank),
                "distance_metric": "median_normalized_euclidean_on_response_coordinates",
                "dimension": coordinate_dimension,
                "model_selection": coordinate_model,
                "leakage": coordinate_leakage,
                "dimension_3d_window": bool(coordinate_dimension.get("estimators_agree_3d", False)),
                "spatial_3d_ready": _spatial_3d_ready(
                    coordinate_dimension,
                    coordinate_model,
                    coordinate_leakage,
                ),
                "strict_3d_ready": False,
                "blockers": _neutral_spatial_3d_blockers(
                    coordinate_dimension,
                    coordinate_model,
                    coordinate_leakage,
                ),
            }
        )
        quotient_distance = _prime_geometric_prefix_distance_matrix(
            neutral_views,
            rank,
            spectrum_name="prime_geometric_control_quotient_spectrum",
        )
        quotient_dimension = strict_neutral_dimension_report(quotient_distance)
        quotient_model = neutral_model_selection(
            quotient_distance,
            seed=int(seed),
            max_points=min(int(max_model_points), len(neutral_views)),
        )
        quotient_leakage = neutral_leakage_audit(quotient_distance, sampled)
        quotient_rows.append(
            {
                "rank": int(rank),
                "dimension": quotient_dimension,
                "model_selection": quotient_model,
                "leakage": quotient_leakage,
                "dimension_3d_window": bool(quotient_dimension.get("estimators_agree_3d", False)),
                "strict_3d_ready": bool(
                    quotient_dimension.get("estimators_agree_3d", False)
                    and quotient_model.get("best_model") == "H3"
                    and quotient_model.get("h3_beats_s2", False)
                    and quotient_model.get("h3_beats_h2_h4", False)
                    and quotient_leakage.get("s2_leakage_pass", False)
                ),
                "blockers": _neutral_profile_blockers(quotient_dimension, quotient_model, quotient_leakage),
            }
        )
        quotient_coordinate_distance = _prime_geometric_prefix_coordinate_distance_matrix(
            neutral_views,
            rank,
            spectrum_name="prime_geometric_control_quotient_spectrum",
        )
        quotient_coordinate_dimension = strict_neutral_dimension_report(quotient_coordinate_distance)
        quotient_coordinate_model = neutral_model_selection(
            quotient_coordinate_distance,
            seed=int(seed),
            max_points=min(int(max_model_points), len(neutral_views)),
        )
        quotient_coordinate_leakage = neutral_leakage_audit(quotient_coordinate_distance, sampled)
        quotient_coordinate_rows.append(
            {
                "rank": int(rank),
                "distance_metric": "median_normalized_euclidean_on_response_coordinates",
                "dimension": quotient_coordinate_dimension,
                "model_selection": quotient_coordinate_model,
                "leakage": quotient_coordinate_leakage,
                "dimension_3d_window": bool(quotient_coordinate_dimension.get("estimators_agree_3d", False)),
                "spatial_3d_ready": _spatial_3d_ready(
                    quotient_coordinate_dimension,
                    quotient_coordinate_model,
                    quotient_coordinate_leakage,
                ),
                "strict_3d_ready": False,
                "blockers": _neutral_spatial_3d_blockers(
                    quotient_coordinate_dimension,
                    quotient_coordinate_model,
                    quotient_coordinate_leakage,
                ),
            }
        )
    target_best_3d_row = _best_rank_sweep_row(
        [row for row in rows if bool(row.get("dimension_3d_window", False))]
    )
    target_coordinate_best_3d_row = _best_rank_sweep_row(
        [row for row in coordinate_rows if bool(row.get("dimension_3d_window", False))]
    )
    strict_ready_count = sum(1 for row in rows if bool(row.get("strict_3d_ready", False)))
    coordinate_spatial_ready_count = sum(
        1 for row in coordinate_rows if bool(row.get("spatial_3d_ready", False))
    )
    target_dimension_window_count = sum(1 for row in rows if bool(row.get("dimension_3d_window", False)))
    coordinate_dimension_window_count = sum(
        1 for row in coordinate_rows if bool(row.get("dimension_3d_window", False))
    )
    quotient_coordinate_spatial_ready_count = sum(
        1 for row in quotient_coordinate_rows if bool(row.get("spatial_3d_ready", False))
    )
    quotient_coordinate_3d_rows = [
        row
        for row in quotient_coordinate_rows
        if bool(row.get("dimension_3d_window", False))
    ]
    diagnostic_receipt = bool(target_dimension_window_count > 0 or coordinate_dimension_window_count > 0)
    strict_neutral_candidate = bool(strict_ready_count > 0)
    spatial_3d_candidate = bool(coordinate_spatial_ready_count > 0)
    control_quotient_spatial_3d_candidate = bool(quotient_coordinate_spatial_ready_count > 0)
    selected_rank_controls = _prime_geometric_selected_rank_controls(
        neutral_views,
        sampled,
        target_rank=int(target_best_3d_row["rank"]) if target_best_3d_row else None,
        coordinate_rank=int(target_coordinate_best_3d_row["rank"]) if target_coordinate_best_3d_row else None,
        seed=int(seed) + 707,
        max_model_points=max_model_points,
    )
    proof_blockers = _prime_geometric_rank_sweep_proof_blockers(
        target_best_3d_row,
        target_coordinate_best_3d_row,
        control_quotient_coordinate_3d_row=_best_rank_sweep_row(quotient_coordinate_3d_rows),
        strict_ready_count=strict_ready_count,
        coordinate_spatial_ready_count=coordinate_spatial_ready_count,
        control_quotient_coordinate_spatial_ready_count=quotient_coordinate_spatial_ready_count,
        selected_rank_controls=selected_rank_controls,
    )
    return {
        "mode": "prime_geometric_rank_sweep_v0",
        "observer_count": len(patch_views),
        "sampled_observer_count": len(sampled),
        "sample_indices": [int(value) for value in sample_indices[: min(2048, len(sample_indices))]],
        "seed": int(seed),
        "max_model_points": int(max_model_points),
        "rows": rows,
        "strict_3d_ready_count": int(strict_ready_count),
        "dimension_3d_window_count": int(target_dimension_window_count),
        "best_dimension_row": _best_rank_sweep_row(rows),
        "best_3d_dimension_row": target_best_3d_row,
        "coordinate_rows": coordinate_rows,
        "coordinate_spatial_3d_ready_count": int(coordinate_spatial_ready_count),
        "coordinate_dimension_3d_window_count": int(coordinate_dimension_window_count),
        "coordinate_best_dimension_row": _best_rank_sweep_row(coordinate_rows),
        "coordinate_best_3d_dimension_row": target_coordinate_best_3d_row,
        "control_quotient_rows": quotient_rows,
        "control_quotient_strict_3d_ready_count": sum(
            1 for row in quotient_rows if bool(row.get("strict_3d_ready", False))
        ),
        "control_quotient_dimension_3d_window_count": sum(
            1 for row in quotient_rows if bool(row.get("dimension_3d_window", False))
        ),
        "control_quotient_best_dimension_row": _best_rank_sweep_row(quotient_rows),
        "control_quotient_coordinate_rows": quotient_coordinate_rows,
        "control_quotient_coordinate_spatial_3d_ready_count": sum(
            1 for row in quotient_coordinate_rows if bool(row.get("spatial_3d_ready", False))
        ),
        "control_quotient_coordinate_dimension_3d_window_count": sum(
            1 for row in quotient_coordinate_rows if bool(row.get("dimension_3d_window", False))
        ),
        "control_quotient_coordinate_best_dimension_row": _best_rank_sweep_row(quotient_coordinate_rows),
        "control_quotient_coordinate_best_3d_dimension_row": _best_rank_sweep_row(
            quotient_coordinate_3d_rows
        ),
        "regulator_control_quotient_lane": {
            "lane_kind": "target_response_with_finite_regulator_control_directions_removed",
            "is_negative_control": False,
            "interpretation": (
                "This quotient removes observer-level directions spanned by finite-regulator controls "
                "before compression. It is not a shuffled/null negative control, so matching dimension "
                "windows here cannot by itself validate a 3D bulk."
            ),
        },
        "PRIME_GEOMETRIC_QUOTIENT_3D_DIAGNOSTIC_RECEIPT": diagnostic_receipt,
        "prime_geometric_quotient_3d_diagnostic_receipt": diagnostic_receipt,
        "prime_geometric_spatial_3d_candidate_receipt": spatial_3d_candidate,
        "prime_geometric_control_quotient_spatial_3d_candidate_receipt": control_quotient_spatial_3d_candidate,
        "prime_geometric_strict_neutral_candidate_receipt": strict_neutral_candidate,
        "selected_rank_controls": selected_rank_controls,
        "proof_blockers": proof_blockers,
        "physical_claim": False,
        "claim_boundary": (
            "Diagnostic sweep over low-rank quotients of the observer-visible prime-geometric modular "
            "response spectrum and its finite-regulator control quotient. Directional rows use cosine "
            "distance; coordinate rows use median-normalized Euclidean distance on the response coordinates. "
            "The control-quotient lane is not a negative control; it is a target-response quotient with "
            "finite-regulator control directions removed. If its coordinate row passes, it is reported as "
            "a stronger finite-regulator spatial-3D candidate than the raw row, but this report still does "
            "not choose a physical rank and is not a neutral bulk proof. A strict rank still requires H3 "
            "model selection, independent rank selection, null controls, and refinement."
        ),
    }


def _prime_geometric_independent_rank_selection(attachment_report: dict[str, Any]) -> dict[str, Any]:
    if not attachment_report:
        return {
            "mode": "prime_geometric_independent_svd_rank_selection_v0",
            "written": False,
            "control_quotient_rank3_selector_receipt": False,
            "reason": "missing_prime_geometric_response_attachment_report",
            "claim_boundary": (
                "Independent rank selection uses singular-value metadata emitted by the "
                "prime-geometric response attachment. Missing metadata means rank 3 remains a "
                "dimension-window candidate only."
            ),
        }

    prime = _nested_dict(attachment_report, "prime_geometric", "embedding", "rank_selection")
    quotient = _nested_dict(
        attachment_report,
        "prime_geometric_control_quotient",
        "embedding",
        "rank_selection",
    )
    return {
        "mode": "prime_geometric_independent_svd_rank_selection_v0",
        "written": True,
        "prime_geometric": _rank_selection_summary(prime),
        "control_quotient": _rank_selection_summary(quotient),
        "prime_rank3_selector_receipt": bool(prime.get("independent_rank3_selector_receipt", False)),
        "control_quotient_rank3_selector_receipt": bool(
            quotient.get("independent_rank3_selector_receipt", False)
        ),
        "claim_boundary": (
            "Independent rank selector from observer-visible modular-response singular values. It is "
            "separate from downstream dimension estimation and does not read screen axes, H3 coordinates, "
            "or target rank. A false result is a real blocker for promoting coordinate rank-3 windows."
        ),
    }


def _nested_dict(value: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def _rank_selection_summary(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(selection),
        "independent_rank3_selector_receipt": bool(
            selection.get("independent_rank3_selector_receipt", False)
        ),
        "largest_gap_rank": selection.get("largest_gap_rank"),
        "chord_elbow_rank": selection.get("chord_elbow_rank"),
        "effective_rank": selection.get("effective_rank"),
        "participation_rank": selection.get("participation_rank"),
        "rank3_cumulative_explained_variance": selection.get("rank3_cumulative_explained_variance"),
        "rank90": selection.get("rank90"),
        "rank95": selection.get("rank95"),
    }


def write_prime_geometric_rank_sweep_report(
    run_dir: Path,
    out: Path | None = None,
    *,
    ranks: list[int] | tuple[int, ...] = tuple(range(2, 17)),
    seed: int = 1,
    sample_count: int = 256,
    max_model_points: int = 128,
) -> dict[str, Any]:
    run = Path(run_dir)
    observer_path = run / "observer_views.jsonl"
    if not observer_path.exists():
        raise FileNotFoundError(observer_path)
    independent_rank_selection = _prime_geometric_independent_rank_selection(
        _read_json(run / "prime_geometric_response_attachment_report.json")
    )
    report = prime_geometric_rank_sweep_report(
        _read_jsonl(observer_path),
        ranks=tuple(int(value) for value in ranks),
        seed=int(seed),
        sample_count=int(sample_count),
        max_model_points=int(max_model_points),
    )
    report["independent_rank_selection"] = independent_rank_selection
    report["independent_rank3_selector_receipt"] = bool(
        independent_rank_selection.get("control_quotient_rank3_selector_receipt", False)
    )
    if report["independent_rank3_selector_receipt"]:
        report["proof_blockers"] = [
            blocker
            for blocker in report.get("proof_blockers", [])
            if blocker != "requires_independent_rank_selection_rule_before_physical_interpretation"
        ]
    report["source_run_dir"] = str(run)
    report["observer_views_path"] = str(observer_path)
    destination = Path(out) if out is not None else run / "prime_geometric_rank_sweep_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def prime_geometric_rank_refinement_report(report_paths: list[Path]) -> dict[str, Any]:
    reports = [_read_json(path) for path in _find_prime_rank_sweep_reports(report_paths)]
    reports = [report for report in reports if report.get("mode") == "prime_geometric_rank_sweep_v0"]
    rows = [_prime_rank_refinement_row(report) for report in reports]
    rows = [row for row in rows if row]
    by_patch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_patch.setdefault(int(row.get("patch_count") or 0), []).append(row)
    sizes: list[dict[str, Any]] = []
    for patch_count, group in sorted(by_patch.items()):
        medians = [row["candidate_median_dimension"] for row in group if row.get("candidate_median_dimension") is not None]
        corr_dims = [row["candidate_corr_dimension"] for row in group if row.get("candidate_corr_dimension") is not None]
        mle_dims = [row["candidate_mle_dimension"] for row in group if row.get("candidate_mle_dimension") is not None]
        leakages = [row["candidate_s2_leakage_corr"] for row in group if row.get("candidate_s2_leakage_corr") is not None]
        sizes.append(
            {
                "patch_count": int(patch_count),
                "run_count": len(group),
                "candidate_count": int(sum(1 for row in group if row.get("control_quotient_spatial_3d_candidate"))),
                "independent_rank3_count": int(sum(1 for row in group if row.get("independent_rank3_selector"))),
                "candidate_rank_counts": _counts(row.get("candidate_rank") for row in group),
                "candidate_model_counts": _counts(row.get("candidate_model") for row in group),
                "median_candidate_dimension": _median_or_none(medians),
                "median_candidate_corr_dimension": _median_or_none(corr_dims),
                "median_candidate_mle_dimension": _median_or_none(mle_dims),
                "median_candidate_s2_leakage_corr": _median_or_none(leakages),
                "s2_leakage_pass_count": int(sum(1 for row in group if row.get("candidate_s2_leakage_pass"))),
            }
        )
    size_medians = [
        float(row["median_candidate_dimension"])
        for row in sizes
        if row.get("median_candidate_dimension") is not None
    ]
    dimension_drift = float(max(size_medians) - min(size_medians)) if len(size_medians) >= 2 else None
    all_candidates = bool(rows and all(row.get("control_quotient_spatial_3d_candidate") for row in rows))
    all_leakage = bool(rows and all(row.get("candidate_s2_leakage_pass") for row in rows))
    all_rank3_e3 = bool(
        rows
        and all(row.get("candidate_rank") == 3 for row in rows)
        and all(row.get("candidate_model") == "E3" for row in rows)
    )
    multi_scale = len([size for size in sizes if int(size.get("patch_count") or 0) > 0]) >= 2
    stable_dimension = bool(dimension_drift is not None and dimension_drift <= 0.20)
    refinement_candidate = bool(
        multi_scale
        and all_candidates
        and all_leakage
        and all_rank3_e3
        and stable_dimension
    )
    independent_rank3_all = bool(rows and all(row.get("independent_rank3_selector") for row in rows))
    blockers: list[str] = []
    if not refinement_candidate:
        blockers.append("control_quotient_rank3_candidate_not_stable_across_refinement")
    if not independent_rank3_all:
        blockers.append("independent_svd_rank3_selector_not_stable_or_false")
    blockers.extend(
        [
            "control_quotient_lane_is_not_a_negative_control",
            "directional_h3_strict_rank_gate_not_passed",
        ]
    )
    return {
        "mode": "prime_geometric_rank_refinement_v0",
        "run_count": len(rows),
        "rows": rows,
        "sizes": sizes,
        "multi_scale": multi_scale,
        "all_control_quotient_spatial_3d_candidates": all_candidates,
        "all_candidate_s2_leakage_pass": all_leakage,
        "all_candidate_rank3_e3": all_rank3_e3,
        "candidate_dimension_drift": dimension_drift,
        "candidate_dimension_stable": stable_dimension,
        "control_quotient_rank3_refinement_candidate_receipt": refinement_candidate,
        "independent_rank3_selector_all": independent_rank3_all,
        "strict_neutral_bulk_refinement_receipt": False,
        "proof_blockers": blockers,
        "physical_claim": False,
        "claim_boundary": (
            "Refinement diagnostic for the control-quotient coordinate rank-3 spatial window. Passing "
            "this report means the finite-regulator candidate is stable across the supplied patch counts. "
            "It is not strict neutral bulk proof unless an independent rank selector passes, the quotient "
            "lane is replaced by a proper null/negative-control gate, and the directional H3 strict gate passes."
        ),
    }


def write_prime_geometric_rank_refinement_report(report_paths: list[Path], out: Path) -> dict[str, Any]:
    report = prime_geometric_rank_refinement_report(report_paths)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prime_geometric_rank_refinement_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    return report


def shuffled_neutral_control_report(
    observer_views: list[dict[str, Any]],
    *,
    seed: int = 1,
    max_model_points: int = 256,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    if len(patch_views) < 8:
        return {
            "mode": "strict_neutral_run_specific_shuffled_controls_v0",
            "observer_count": len(patch_views),
            "controls": {
                "shuffled_records_fail": False,
                "shuffled_transition_labels_fail": False,
            },
            "reason": "too_few_patch_observer_views",
        }
    sample_count = min(len(patch_views), max(8, int(max_model_points)))
    if len(patch_views) > sample_count:
        sample_indices = set(int(value) for value in rng.choice(len(patch_views), size=sample_count, replace=False))
        control_observer_views = [view for index, view in enumerate(patch_views) if index in sample_indices]
    else:
        control_observer_views = patch_views
    original_views = build_neutral_observer_views(control_observer_views)
    original_distance = neutral_distance_matrix(original_views)
    controls: dict[str, Any] = {}
    rows: dict[str, Any] = {}
    for name, shuffled in {
        "shuffled_records": _shuffle_record_payloads(control_observer_views, rng),
        "shuffled_transition_labels": _shuffle_transition_labels(control_observer_views, rng),
    }.items():
        neutral_views = build_neutral_observer_views(shuffled)
        distance = neutral_distance_matrix(neutral_views)
        corr = _upper_triangle_corr(original_distance, distance)
        delta = _mean_abs_upper_delta(original_distance, distance)
        degraded = _neutral_distance_control_degraded(corr, delta)
        rows[name] = {
            "distance_shape_correlation_to_original": corr,
            "mean_abs_distance_delta": delta,
            "expected_failure_observed": degraded,
            "model_selection_recomputed": False,
            "claim_boundary": (
                "Run-specific shuffled observer-control for the strict neutral quotient. It must degrade "
                "the observer-visible neutral distance; full H3/H4 refits are skipped for runtime."
            ),
        }
        controls[f"{name}_fail"] = degraded
    return {
        "mode": "strict_neutral_run_specific_shuffled_controls_v0",
        "observer_count": len(patch_views),
        "sampled_observer_count": int(sample_count),
        "max_model_points": int(max_model_points),
        "original": {"distance_matrix_shape": list(original_distance.shape)},
        "rows": rows,
        "controls": {
            "shuffled_records_fail": bool(controls.get("shuffled_records_fail", False)),
            "shuffled_transition_labels_fail": bool(controls.get("shuffled_transition_labels_fail", False)),
        },
        "claim_boundary": "Run-specific neutral controls. Passing controls do not prove bulk without dimension/refinement gates.",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _find_prime_rank_sweep_reports(paths: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in paths:
        path = Path(path)
        if path.is_file():
            found.add(path)
            continue
        direct = path / "prime_geometric_rank_sweep_report.json"
        if direct.exists():
            found.add(direct)
        if path.exists():
            found.update(path.glob("**/prime_geometric_rank_sweep_report.json"))
    return sorted(found, key=lambda value: str(value))


def _prime_rank_refinement_row(report: dict[str, Any]) -> dict[str, Any]:
    candidate = report.get("control_quotient_coordinate_best_3d_dimension_row") or {}
    dimension = candidate.get("dimension") if isinstance(candidate.get("dimension"), dict) else {}
    model = candidate.get("model_selection") if isinstance(candidate.get("model_selection"), dict) else {}
    leakage = candidate.get("leakage") if isinstance(candidate.get("leakage"), dict) else {}
    source_run_dir = str(report.get("source_run_dir") or "")
    return {
        "source_run_dir": source_run_dir,
        "patch_count": _patch_count_from_source(source_run_dir),
        "observer_count": report.get("observer_count"),
        "sampled_observer_count": report.get("sampled_observer_count"),
        "control_quotient_spatial_3d_candidate": bool(
            report.get("prime_geometric_control_quotient_spatial_3d_candidate_receipt", False)
        ),
        "independent_rank3_selector": bool(report.get("independent_rank3_selector_receipt", False)),
        "candidate_rank": candidate.get("rank"),
        "candidate_model": model.get("best_model"),
        "candidate_median_dimension": _to_float_or_none(dimension.get("median_dimension_estimate")),
        "candidate_corr_dimension": _to_float_or_none(
            ((dimension.get("correlation_dimension") or {}).get("estimate"))
        ),
        "candidate_mle_dimension": _to_float_or_none(
            ((dimension.get("local_mle_dimension") or {}).get("median_estimate"))
        ),
        "candidate_s2_leakage_corr": _to_float_or_none(leakage.get("s2_distance_correlation")),
        "candidate_s2_leakage_pass": bool(leakage.get("s2_leakage_pass", False)),
        "proof_blockers": report.get("proof_blockers", []),
    }


def _patch_count_from_source(source_run_dir: str) -> int:
    if not source_run_dir:
        return 0
    manifest = _read_json(Path(source_run_dir) / "manifest.json")
    try:
        return int(manifest.get("patch_count") or 0)
    except (TypeError, ValueError):
        return 0


def _to_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _median_or_none(values: list[float]) -> float | None:
    return float(np.median(np.asarray(values, dtype=float))) if values else None


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


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


def _neutral_control_degraded(
    original_model: dict[str, Any],
    control_model: dict[str, Any],
    distance_corr: float | None,
    mean_abs_delta: float,
) -> bool:
    original_h3 = _model_h3_stress(original_model)
    control_h3 = _model_h3_stress(control_model)
    selected = control_model.get("selected_model", control_model.get("best_model"))
    h3_structure_lost = bool(
        selected != "H3"
        or not control_model.get("h3_beats_s2", False)
        or not control_model.get("h3_beats_e3", False)
        or not control_model.get("h3_beats_h2_h4", False)
    )
    stress_degraded = bool(
        np.isfinite(original_h3)
        and np.isfinite(control_h3)
        and control_h3 > original_h3 + max(0.01, 0.15 * max(original_h3, 0.0))
    )
    distance_degraded = bool(
        distance_corr is None
        or not np.isfinite(float(distance_corr))
        or float(distance_corr) < 0.97
        or float(mean_abs_delta) > 1e-6
    )
    return bool(distance_degraded and (h3_structure_lost or stress_degraded))


def _neutral_distance_control_degraded(distance_corr: float | None, mean_abs_delta: float) -> bool:
    return bool(
        distance_corr is None
        or not np.isfinite(float(distance_corr))
        or float(distance_corr) < 0.99
        or float(mean_abs_delta) > 1e-6
    )


def _neutral_profile_blockers(
    dimension: dict[str, Any],
    model: dict[str, Any],
    leakage: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not dimension.get("estimators_agree_3d", False):
        blockers.append("dimension_estimators_do_not_agree_3d")
    if model.get("best_model") != "H3":
        blockers.append("h3_not_selected")
    if not model.get("h3_beats_s2", False):
        blockers.append("h3_does_not_beat_s2")
    if not model.get("h3_beats_h2_h4", False):
        blockers.append("h3_does_not_clear_h2_h4")
    if not leakage.get("s2_leakage_pass", False):
        blockers.append("s2_leakage_failed")
    return blockers


def _model_h3_stress(model: dict[str, Any]) -> float:
    try:
        return float(((model.get("models") or {}).get("H3") or {}).get("heldout_stress", float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _mean_abs_upper_delta(a: np.ndarray, b: np.ndarray) -> float:
    av = _upper_triangle(a)
    bv = _upper_triangle(b)
    count = min(av.size, bv.size)
    if count == 0:
        return 0.0
    return float(np.mean(np.abs(av[:count] - bv[:count])))


def _shuffle_record_payloads(observer_views: list[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    shuffled = copy.deepcopy(observer_views)
    patch_indices = [index for index, view in enumerate(shuffled) if view.get("view_type") == "patch_observer"]
    keys = (
        "record_signature_histogram",
        "object_packet_histogram",
        "counterfactual_continuation_hist",
        "counterfactual_stability",
        "committed_fraction",
        "record_stability_mean",
        "repair_load_mean",
        "mismatch_density_mean",
        "visible_signature_entropy",
    )
    for key in keys:
        values = [copy.deepcopy(shuffled[index].get(key)) for index in patch_indices]
        order = rng.permutation(len(values))
        for local_index, source_index in enumerate(order):
            value = values[int(source_index)]
            if value is None:
                shuffled[patch_indices[local_index]].pop(key, None)
            else:
                shuffled[patch_indices[local_index]][key] = copy.deepcopy(value)
    return shuffled


def _shuffle_transition_labels(observer_views: list[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    shuffled = copy.deepcopy(observer_views)
    for view in shuffled:
        if view.get("view_type") != "patch_observer":
            continue
        for key in (
            "transition_history_histograms",
            "transition_affinity_histograms",
            "modular_response_histograms",
        ):
            if isinstance(view.get(key), dict):
                view[key] = _randomize_histogram_keys(view[key], rng)
    patch_indices = [index for index, view in enumerate(shuffled) if view.get("view_type") == "patch_observer"]
    for key in (
        "repair_response_spectrum",
        "prime_geometric_modular_spectrum",
        "prime_geometric_control_quotient_spectrum",
        "support_visible_modular_spectrum",
        "repair_modular_spectrum",
    ):
        values = [copy.deepcopy(shuffled[index].get(key)) for index in patch_indices]
        order = rng.permutation(len(values))
        for local_index, source_index in enumerate(order):
            value = values[int(source_index)]
            if value is None:
                shuffled[patch_indices[local_index]].pop(key, None)
            else:
                shuffled[patch_indices[local_index]][key] = copy.deepcopy(value)
    return shuffled


def _randomize_histogram_keys(value: Any, rng: np.random.Generator) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, dict):
                out[str(key)] = _randomize_histogram_keys(item, rng)
            else:
                out[str(int(rng.integers(0, 2**50)))] = item
        return out
    return value


def _prime_geometric_prefix_distance_matrix(
    views: list[NeutralObserverView],
    rank: int,
    *,
    spectrum_name: str = "prime_geometric_modular_spectrum",
) -> np.ndarray:
    n = len(views)
    distance = np.zeros((n, n), dtype=float)
    rank = max(1, int(rank))
    for i in range(n):
        xi = getattr(views[i], spectrum_name)[:rank]
        for j in range(i + 1, n):
            xj = getattr(views[j], spectrum_name)[:rank]
            value = cosine_distance(xi, xj)
            distance[i, j] = distance[j, i] = float(max(0.0, value))
    return distance


def _prime_geometric_prefix_coordinate_distance_matrix(
    views: list[NeutralObserverView],
    rank: int,
    *,
    spectrum_name: str = "prime_geometric_modular_spectrum",
) -> np.ndarray:
    rank = max(1, int(rank))
    if not views:
        return np.zeros((0, 0), dtype=float)
    coords = np.vstack([np.asarray(getattr(view, spectrum_name)[:rank], dtype=float) for view in views])
    coords = np.where(np.isfinite(coords), coords, 0.0)
    distance = squareform(pdist(coords, metric="euclidean")) if coords.shape[0] > 1 else np.zeros((coords.shape[0], coords.shape[0]))
    positive = distance[np.isfinite(distance) & (distance > 1e-12)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if scale <= 1e-12 or not np.isfinite(scale):
        scale = 1.0
    return distance / scale


def _spatial_3d_ready(
    dimension: dict[str, Any],
    model: dict[str, Any],
    leakage: dict[str, Any],
) -> bool:
    best = model.get("best_model")
    h3_or_e3 = best in {"E3", "H3"}
    models = model.get("models") if isinstance(model.get("models"), dict) else {}
    selected = models.get(str(best), {}) if isinstance(models.get(str(best), {}), dict) else {}
    h4 = models.get("H4", {}) if isinstance(models.get("H4", {}), dict) else {}
    selected_stress = float(selected.get("heldout_stress", np.inf))
    h4_stress = float(h4.get("heldout_stress", np.inf))
    h4_compatible = bool(
        np.isfinite(selected_stress)
        and np.isfinite(h4_stress)
        and selected_stress <= h4_stress + _model_compatibility_tolerance(h4_stress)
    )
    return bool(
        dimension.get("estimators_agree_3d", False)
        and h3_or_e3
        and leakage.get("s2_leakage_pass", False)
        and h4_compatible
    )


def _neutral_spatial_3d_blockers(
    dimension: dict[str, Any],
    model: dict[str, Any],
    leakage: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not dimension.get("estimators_agree_3d", False):
        blockers.append("dimension_estimators_do_not_agree_3d")
    if model.get("best_model") not in {"E3", "H3"}:
        blockers.append("selected_model_is_not_3d_spatial")
    models = model.get("models") if isinstance(model.get("models"), dict) else {}
    best = model.get("best_model")
    selected = models.get(str(best), {}) if isinstance(models.get(str(best), {}), dict) else {}
    h4 = models.get("H4", {}) if isinstance(models.get("H4", {}), dict) else {}
    try:
        selected_stress = float(selected.get("heldout_stress", np.inf))
        h4_stress = float(h4.get("heldout_stress", np.inf))
    except (TypeError, ValueError):
        selected_stress = float("inf")
        h4_stress = float("inf")
    if not (
        np.isfinite(selected_stress)
        and np.isfinite(h4_stress)
        and selected_stress <= h4_stress + _model_compatibility_tolerance(h4_stress)
    ):
        blockers.append("h4_improvement_exceeds_parsimony_tolerance")
    if not leakage.get("s2_leakage_pass", False):
        blockers.append("s2_leakage_failed")
    return blockers


def _prime_geometric_rank_sweep_proof_blockers(
    best_directional_3d_row: dict[str, Any] | None,
    best_coordinate_3d_row: dict[str, Any] | None,
    *,
    control_quotient_coordinate_3d_row: dict[str, Any] | None = None,
    strict_ready_count: int,
    coordinate_spatial_ready_count: int,
    control_quotient_coordinate_spatial_ready_count: int = 0,
    selected_rank_controls: dict[str, Any] | None = None,
) -> list[str]:
    blockers: list[str] = []
    if best_directional_3d_row is None and best_coordinate_3d_row is None:
        blockers.append("no_target_prime_geometric_3d_dimension_window")
    if strict_ready_count <= 0:
        blockers.append("no_directional_rank_passes_strict_h3_model_and_leakage_gates")
    if coordinate_spatial_ready_count <= 0 and control_quotient_coordinate_spatial_ready_count <= 0:
        blockers.append("no_coordinate_rank_passes_spatial_3d_model_and_leakage_gates")

    rows = [
        row
        for row in (
            best_directional_3d_row,
            best_coordinate_3d_row,
            control_quotient_coordinate_3d_row,
        )
        if row
    ]
    if rows and not any(((row.get("leakage") or {}).get("s2_leakage_pass", False)) for row in rows):
        blockers.append("best_3d_windows_still_fail_s2_leakage_gate")
    if rows and not any(((row.get("model_selection") or {}).get("best_model") in {"H3", "E3"}) for row in rows):
        blockers.append("best_3d_windows_do_not_select_a_3d_model_family")

    controls = selected_rank_controls or {}
    if not controls.get("all_expected_failures_observed", False):
        blockers.append("selected_rank_null_controls_do_not_all_fail")
    blockers.extend(
        [
            "control_quotient_lane_is_not_a_negative_control",
            "requires_refinement_stability_across_regulator_sizes",
            "requires_independent_rank_selection_rule_before_physical_interpretation",
        ]
    )
    return blockers


def _prime_geometric_selected_rank_controls(
    neutral_views: list[NeutralObserverView],
    sampled_observer_rows: list[dict[str, Any]],
    *,
    target_rank: int | None,
    coordinate_rank: int | None,
    seed: int,
    max_model_points: int,
) -> dict[str, Any]:
    if not neutral_views:
        return {
            "mode": "prime_geometric_selected_rank_null_controls_v0",
            "control_rows": [],
            "all_expected_failures_observed": False,
            "reason": "no_neutral_views",
        }
    if target_rank is None and coordinate_rank is None:
        return {
            "mode": "prime_geometric_selected_rank_null_controls_v0",
            "control_rows": [],
            "all_expected_failures_observed": False,
            "reason": "no_selected_3d_rank",
        }
    rng = np.random.default_rng(int(seed))
    spectra = np.vstack([view.prime_geometric_modular_spectrum for view in neutral_views])
    control_specs = _prime_control_spectra(spectra, rng)
    rows: list[dict[str, Any]] = []
    for control_name, control_spectra in control_specs.items():
        control_views = [
            replace(view, prime_geometric_modular_spectrum=np.asarray(control_spectra[index], dtype=float))
            for index, view in enumerate(neutral_views)
        ]
        if target_rank is not None:
            rows.append(
                _prime_selected_rank_control_row(
                    control_views,
                    sampled_observer_rows,
                    int(target_rank),
                    control_name=control_name,
                    metric="directional_cosine",
                    seed=int(seed),
                    max_model_points=max_model_points,
                )
            )
        if coordinate_rank is not None:
            rows.append(
                _prime_selected_rank_control_row(
                    control_views,
                    sampled_observer_rows,
                    int(coordinate_rank),
                    control_name=control_name,
                    metric="coordinate_euclidean",
                    seed=int(seed),
                    max_model_points=max_model_points,
                )
            )
    non_tautological_rows = [
        row
        for row in rows
        if not (row.get("metric") == "coordinate_euclidean" and row.get("rank") == 3)
    ]
    all_failed = bool(
        non_tautological_rows
        and all(row.get("expected_failure_observed", False) for row in non_tautological_rows)
    )
    coordinate_tautology = any(
        row.get("metric") == "coordinate_euclidean"
        and row.get("rank") == 3
        and not row.get("expected_failure_observed", False)
        for row in rows
    )
    return {
        "mode": "prime_geometric_selected_rank_null_controls_v0",
        "target_rank": target_rank,
        "coordinate_rank": coordinate_rank,
        "control_rows": rows,
        "all_expected_failures_observed": all_failed,
        "non_tautological_control_count": int(len(non_tautological_rows)),
        "non_tautological_expected_failure_count": int(
            sum(1 for row in non_tautological_rows if row.get("expected_failure_observed", False))
        ),
        "coordinate_rank3_tautology_warning": bool(coordinate_tautology),
        "claim_boundary": (
            "Selected-rank null controls for the prime-geometric quotient. Directional controls "
            "should lose the selected 3D window. Coordinate rank-3 controls can remain 3D by "
            "construction, so coordinate rank-3 rows are excluded from all_expected_failures_observed "
            "and are never accepted as strict bulk proof."
        ),
    }


def _prime_control_spectra(spectra: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim != 2:
        spectra = np.zeros((0, 0), dtype=float)
    if spectra.size == 0:
        return {"empty": spectra.copy()}
    row_order = rng.permutation(spectra.shape[0])
    shuffled = spectra[row_order, :].copy()
    component_permuted = spectra.copy()
    for row in component_permuted:
        rng.shuffle(row)
    mean = np.mean(spectra, axis=0, keepdims=True)
    std = np.std(spectra, axis=0, keepdims=True)
    std[std < 1e-9] = 1.0
    gaussian = rng.normal(loc=mean, scale=std, size=spectra.shape)
    return {
        "shuffled_observer_prime_spectrum": shuffled,
        "component_permutation_per_observer": component_permuted,
        "gaussian_column_null": gaussian,
    }


def _prime_selected_rank_control_row(
    neutral_views: list[NeutralObserverView],
    sampled_observer_rows: list[dict[str, Any]],
    rank: int,
    *,
    control_name: str,
    metric: str,
    seed: int,
    max_model_points: int,
) -> dict[str, Any]:
    if metric == "coordinate_euclidean":
        distance = _prime_geometric_prefix_coordinate_distance_matrix(
            neutral_views,
            rank,
            spectrum_name="prime_geometric_modular_spectrum",
        )
    else:
        distance = _prime_geometric_prefix_distance_matrix(
            neutral_views,
            rank,
            spectrum_name="prime_geometric_modular_spectrum",
        )
    dimension = strict_neutral_dimension_report(distance)
    model = neutral_model_selection(
        distance,
        seed=int(seed),
        max_points=min(int(max_model_points), len(neutral_views)),
    )
    leakage = neutral_leakage_audit(distance, sampled_observer_rows)
    if metric == "coordinate_euclidean":
        candidate = _spatial_3d_ready(dimension, model, leakage)
    else:
        candidate = bool(
            dimension.get("estimators_agree_3d", False)
            and model.get("best_model") == "H3"
            and model.get("h3_beats_s2", False)
            and model.get("h3_beats_h2_h4", False)
            and leakage.get("s2_leakage_pass", False)
        )
    return {
        "control": str(control_name),
        "metric": str(metric),
        "rank": int(rank),
        "dimension": dimension,
        "model_selection": model,
        "leakage": leakage,
        "candidate_survives_control": bool(candidate),
        "expected_failure_observed": not bool(candidate),
        "rank3_coordinate_tautology_warning": bool(metric == "coordinate_euclidean" and rank == 3),
    }


def _best_rank_sweep_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None

    def score(row: dict[str, Any]) -> float:
        corr = _nested_float(row, "dimension", "correlation_dimension", "estimate")
        mle = _nested_float(row, "dimension", "local_mle_dimension", "median_estimate")
        values = [value for value in (corr, mle) if value is not None and np.isfinite(value)]
        if not values:
            return float("inf")
        return float(np.mean([abs(value - 3.0) for value in values]))

    return min(rows, key=score)


def _nested_float(row: dict[str, Any], *path: str) -> float | None:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _empty_model_selection(reason: str) -> dict[str, Any]:
    return {
        "mode": "strict_neutral_distance_model_selection_v0",
        "best_model": None,
        "raw_best_model": None,
        "selected_model": None,
        "models": {},
        "h3_beats_s2": False,
        "h3_beats_e3": False,
        "h3_beats_h2_h4": False,
        "blocker": reason,
    }


def _parsimonious_model_selection(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    finite = {
        name: value
        for name, value in models.items()
        if np.isfinite(float(value.get("heldout_stress", np.inf)))
    }
    if not finite:
        return {
            "selected_model": None,
            "selection_rule": "no_finite_model",
        }
    raw_best = min(finite, key=lambda name: float(finite[name].get("heldout_stress", np.inf)))
    best_stress = float(finite[raw_best].get("heldout_stress", np.inf))
    tolerance = _model_compatibility_tolerance(best_stress)
    compatible = [
        name
        for name, value in finite.items()
        if float(value.get("heldout_stress", np.inf)) <= best_stress + tolerance
    ]
    complexity_order = {
        "S2": (2, 0),
        "E2": (2, 1),
        "H2": (2, 2),
        "E3": (3, 1),
        "H3": (3, 2),
        "H4": (4, 2),
    }
    selected = min(
        compatible,
        key=lambda name: (
            complexity_order.get(name, (999, 999))[0],
            float(finite[name].get("heldout_stress", np.inf)),
            complexity_order.get(name, (999, 999))[1],
            name,
        ),
    )
    return {
        "selected_model": selected,
        "selection_rule": (
            "lowest_dimension_model_with_heldout_stress_within_"
            f"max({MODEL_SELECTION_ABS_TOLERANCE}, {MODEL_SELECTION_REL_TOLERANCE}*raw_best_stress)"
        ),
    }


def _model_compatibility_tolerance(best_stress: float) -> float:
    if not np.isfinite(best_stress):
        return MODEL_SELECTION_ABS_TOLERANCE
    return max(MODEL_SELECTION_ABS_TOLERANCE, MODEL_SELECTION_REL_TOLERANCE * max(float(best_stress), 0.0))


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
    if not isinstance(histogram, dict):
        return _normalize_or_zero(vector, width=vector.size)
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


def _nested_histogram_to_vector(histograms: Any, width: int) -> np.ndarray:
    vector = np.zeros(max(1, int(width)), dtype=float)
    if not isinstance(histograms, dict):
        return vector
    for field_name, histogram in histograms.items():
        if isinstance(histogram, dict):
            for key, value in histogram.items():
                bucket = _stable_bucket(f"{field_name}:{key}", vector.size)
                try:
                    mass = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(mass):
                    vector[bucket] += max(mass, 0.0)
        else:
            bucket = _stable_bucket(str(field_name), vector.size)
            try:
                mass = float(histogram)
            except (TypeError, ValueError):
                continue
            if np.isfinite(mass):
                vector[bucket] += max(mass, 0.0)
    return _normalize_or_zero(vector, width=vector.size)


def _transition_history_hist(view: dict[str, Any], key: str, width: int) -> np.ndarray:
    histograms = view.get("transition_history_histograms")
    if not isinstance(histograms, dict):
        return np.zeros(max(1, int(width)), dtype=float)
    histogram = histograms.get(key)
    if isinstance(histogram, dict):
        return _histogram_dict_to_vector(histogram, width)
    return np.zeros(max(1, int(width)), dtype=float)


def _signed_vector_or_zero(values: Any, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1) if values is not None else np.zeros(0, dtype=float)
    array = _pad(array, int(width))[: int(width)]
    return np.where(np.isfinite(array), array, 0.0)


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


def _stable_bucket(value: str, width: int) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % max(1, int(width))


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
