from __future__ import annotations

import json
import math
import hashlib
import copy
import csv
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh as dense_eigh
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform

from oph_fpe.bulk.quotient_geometry import (
    ChannelMetricSpec,
    GEOMETRY_CONTRACT_RECEIPT,
    quotient_geometry_certificate,
)
from oph_fpe.evidence.hashes import stable_json_hash
from oph_fpe.observers.sampling import deterministic_observer_analysis_indices


STRICT_NEUTRAL_SOURCE_SCHEMA = "strict_neutral_bulk_source_v1"
STRICT_NEUTRAL_SAMPLED_SOURCE_SCHEMA = "strict_neutral_bulk_source_v2"


DEFAULT_NEUTRAL_WEIGHTS = {
    "local_packet": 1.5,
    # Legacy hash/categorical lanes remain exported for reproducibility, but
    # are non-claim-bearing by default. Nearby physical packets need not land
    # in nearby token bins, and open-link S3 classes are frame dependent.
    "record": 0.0,
    "record_signature": 0.0,
    "object_packet": 0.0,
    "boundary_packet": 0.0,
    "overlap_correspondence": 1.0,
    "counterfactual": 0.0,
    "checkpoint": 0.0,
    "sector": 0.0,
    "port_pair_lag": 0.0,
    "repair": 0.0,
    "repair_spectrum": 0.0,
    "repair_current_tensor": 0.0,
    "perturbation_response_tensor": 0.9,
    "first_passage_response": 0.0,
    "persistence": 0.0,
    "scalar_readout": 0.0,
}

STRICT_NEUTRAL_RECORD_REPAIR_ONLY_WEIGHTS = dict(DEFAULT_NEUTRAL_WEIGHTS)

SUPPORT_VISIBLE_PRIME_GEOMETRIC_DIAGNOSTIC_WEIGHTS = {
    "modular_response": 0.75,
    "prime_geometric_modular": 0.9,
    "prime_geometric_control_quotient": 0.9,
    "support_visible_modular": 0.9,
    "repair_modular": 0.35,
}

STRICT_NEUTRAL_THEORY_REQUIRED_CHANNELS: dict[str, str] = {
    "local_packet": (
        "gauge-invariant repair readbacks over observer supports selected by an independently produced "
        "chart-blind carrier"
    ),
    "record": "record/checkpoint order visible in each local observer transcript",
    "checkpoint": "checkpoint order and record ancestry visible without a chart",
    "boundary_packet": "local metric-bearing port or boundary packets, not hash-bin coordinates",
    "overlap_correspondence": (
        "measured observer-overlap correspondences on independently produced chart-blind supports"
    ),
    "port_pair_lag": "transition counts by local port pair and lag",
    "repair_current_tensor": "repair-current response tensor in local-port coordinates",
    "perturbation_response_tensor": "counterfactual perturbation-response tensor",
    "first_passage_response": "first-passage or response-time observable histogram",
}

DIAGNOSTIC_ONLY_NEUTRAL_CHANNELS = (
    "modular_response",
    "prime_geometric_modular",
    "prime_geometric_control_quotient",
    "prime_geometric_rank3",
    "prime_geometric_rank4",
    "prime_geometric_rank8",
    "prime_geometric_rank16",
    "prime_geometric_rank32",
    "prime_geometric_control_quotient_rank3",
    "prime_geometric_control_quotient_rank4",
    "prime_geometric_control_quotient_rank8",
    "prime_geometric_control_quotient_rank16",
    "prime_geometric_control_quotient_rank32",
    "support_visible_modular",
    "repair_modular",
    "transition_token",
    "transition_token_persistent",
    "transition_affinity",
)

STRICT_NEUTRAL_FORBIDDEN_FEATURE_ANCESTORS = (
    "h3_coordinate",
    "s2_cap_axis",
    "support_node_id",
    "radial_depth",
    "modular_depth",
    "screen_pixel_coordinate",
    "prime_geometric_response",
    "support_visible_modular_response",
    "cmb_residual",
    "visual_overlay",
    "likelihood_output",
)

MODEL_SELECTION_ABS_TOLERANCE = 0.01
MODEL_SELECTION_REL_TOLERANCE = 0.08
DUPLICATE_CHANNEL_CORRELATION_THRESHOLD = 0.995

NEUTRAL_PROFILE_WEIGHTS: dict[str, dict[str, float] | None] = {
    "all_observer_visible": None,
    "strict_record_repair_only": STRICT_NEUTRAL_RECORD_REPAIR_ONLY_WEIGHTS,
    "overlap_record_repair_only": STRICT_NEUTRAL_RECORD_REPAIR_ONLY_WEIGHTS,
    "support_visible_prime_geometric_diagnostic": SUPPORT_VISIBLE_PRIME_GEOMETRIC_DIAGNOSTIC_WEIGHTS,
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
    locality_packet_features: np.ndarray
    record_transition_hist: np.ndarray
    record_signature_hist: np.ndarray
    object_packet_hist: np.ndarray
    boundary_packet_hash_hist: np.ndarray
    overlap_correspondence_hist: np.ndarray
    counterfactual_hist: np.ndarray
    checkpoint_transition_hist: np.ndarray
    sector_transition_hist: np.ndarray
    port_pair_lag_hist: np.ndarray
    repair_response_hist: np.ndarray
    repair_response_spectrum: np.ndarray
    repair_current_tensor: np.ndarray
    perturbation_response_tensor: np.ndarray
    first_passage_response_hist: np.ndarray
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
                locality_packet_features=_signed_vector_or_zero(
                    view.get("locality_preserving_packet_feature_vector", []),
                    width=96,
                ),
                record_transition_hist=_hist_or_steps(view, steps, "record_family", 32),
                record_signature_hist=_histogram_dict_to_vector(view.get("record_signature_histogram", {}), 64),
                object_packet_hist=_histogram_dict_to_vector(view.get("object_packet_histogram", {}), 64),
                boundary_packet_hash_hist=_first_histogram_vector(
                    view,
                    (
                        "local_boundary_packet_hash_histogram",
                        "boundary_packet_hash_histogram",
                        "visible_boundary_packet_hash_histogram",
                        "local_port_packet_hash_histogram",
                        "boundary_packet_histogram",
                    ),
                    128,
                ),
                overlap_correspondence_hist=_measured_overlap_summary_vector(view, width=256),
                counterfactual_hist=_normalize_or_zero(view.get("counterfactual_continuation_hist", []), width=16),
                checkpoint_transition_hist=_hist_or_steps(view, steps, "checkpoint_class", 32),
                sector_transition_hist=_hist_or_steps(view, steps, "s3_sector_class", 6),
                port_pair_lag_hist=_port_pair_lag_vector(view, steps, width=512),
                repair_response_hist=_hist_or_steps(view, steps, "repair_load_bucket", 16),
                repair_response_spectrum=_signed_vector_or_zero(view.get("repair_response_spectrum", []), width=32),
                repair_current_tensor=_first_signed_vector(
                    view,
                    (
                        "repair_current_tensor",
                        "repair_current_histogram",
                        "local_repair_current_tensor",
                        "repair_current_response_tensor",
                    ),
                    128,
                ),
                perturbation_response_tensor=_verified_perturbation_response_vector(view, width=128),
                first_passage_response_hist=_first_histogram_vector(
                    view,
                    (
                        "first_passage_time_histogram",
                        "response_time_histogram",
                        "first_passage_response_histogram",
                        "local_first_passage_histogram",
                    ),
                    64,
                ),
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
    return float(neutral_distance_matrix([a, b], weights=weights)[0, 1])


def neutral_distance_matrix(
    views: list[NeutralObserverView],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    n = len(views)
    if n <= 0:
        return np.zeros((0, 0), dtype=float)
    features = neutral_feature_matrix(views, weights=weights)
    if features.shape[1] == 0:
        return np.zeros((n, n), dtype=float)
    distance = squareform(pdist(features, metric="euclidean"))
    distance = np.where(np.isfinite(distance), np.maximum(distance, 0.0), 0.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def neutral_feature_matrix(
    views: list[NeutralObserverView],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Build the fixed claim-bearing neutral embedding.

    Each active channel contributes a globally transformed Euclidean block:
    histograms use the Hellinger embedding, signed spectra/scalars use a
    run-global standardized Euclidean embedding, and channel weights enter as
    sqrt(w). No per-pair renormalization is allowed.
    """

    weights = weights or DEFAULT_NEUTRAL_WEIGHTS
    n = len(views)
    if n <= 0:
        return np.zeros((0, 0), dtype=float)
    active = [(str(key), float(value)) for key, value in weights.items() if float(value) > 0.0]
    signal_blocks: list[tuple[str, float, np.ndarray]] = []
    for key, weight in active:
        raw = _neutral_channel_matrix(views, key)
        if raw.shape[0] != n or raw.shape[1] == 0:
            continue
        embedded = _neutral_channel_embedding(raw, key)
        if embedded.shape[1] == 0 or not _neutral_embedding_has_pairwise_signal(embedded):
            continue
        signal_blocks.append((key, weight, embedded))
    total_weight = float(sum(weight for _, weight, _ in signal_blocks))
    if total_weight <= 1e-12:
        return np.zeros((n, 0), dtype=float)
    blocks = [math.sqrt(weight / total_weight) * embedded for _, weight, embedded in signal_blocks]
    return np.hstack(blocks) if blocks else np.zeros((n, 0), dtype=float)


def neutral_channel_duplicate_audit(
    views: list[NeutralObserverView],
    weights: dict[str, float] | None = None,
    *,
    threshold: float = DUPLICATE_CHANNEL_CORRELATION_THRESHOLD,
) -> dict[str, Any]:
    weights = weights or DEFAULT_NEUTRAL_WEIGHTS
    active = [str(key) for key, value in weights.items() if float(value) > 0.0]
    ancestry_manifest = neutral_feature_ancestry_manifest(weights)
    ancestry_blockers = _strict_neutral_feature_ancestry_blockers(ancestry_manifest)
    channel_distances: dict[str, np.ndarray] = {}
    degenerate_channels: list[str] = []
    for key in active:
        features = neutral_feature_matrix(views, weights={key: 1.0})
        if features.shape[0] < 3 or features.shape[1] == 0:
            degenerate_channels.append(key)
            continue
        distance = squareform(pdist(features, metric="euclidean"))
        upper = _upper_triangle(distance)
        if upper.size < 2 or float(np.std(upper)) <= 1e-12:
            degenerate_channels.append(key)
            continue
        channel_distances[key] = distance
    duplicate_pairs: list[dict[str, Any]] = []
    keys = sorted(channel_distances)
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            corr = _upper_triangle_corr(channel_distances[left], channel_distances[right])
            if corr is not None and abs(float(corr)) > float(threshold):
                duplicate_pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "distance_correlation": float(corr),
                    }
                )
    return {
        "mode": "neutral_primary_channel_duplicate_audit_v0",
        "active_primary_channels": active,
        "diagnostic_only_channels": list(DIAGNOSTIC_ONLY_NEUTRAL_CHANNELS),
        "feature_ancestry_manifest": ancestry_manifest,
        "feature_ancestry_gate_pass": not ancestry_blockers,
        "feature_ancestry_blockers": ancestry_blockers,
        "correlation_threshold": float(threshold),
        "duplicate_pairs": duplicate_pairs,
        "degenerate_channels": degenerate_channels,
        "duplicate_channel_gate_pass": not duplicate_pairs and not ancestry_blockers,
        "claim_boundary": (
            "Primary-channel audit for strict neutral distance. Rank prefixes, support-visible "
            "duplicates, prime-geometric/support-visible modular channels, and hash/token channels are "
            "kept diagnostic-only by default; any remaining near-duplicate or forbidden-ancestry primary "
            "channels block strict neutral promotion."
        ),
    }


def neutral_feature_ancestry_manifest(weights: dict[str, float] | None = None) -> list[dict[str, Any]]:
    weights = weights or DEFAULT_NEUTRAL_WEIGHTS
    rows = []
    for name, weight in sorted(weights.items()):
        active = float(weight) > 0.0
        ancestors = _neutral_channel_ancestors(str(name))
        forbidden = [ancestor for ancestor in ancestors if ancestor in STRICT_NEUTRAL_FORBIDDEN_FEATURE_ANCESTORS]
        rows.append(
            {
                "featureId": str(name),
                "featureName": str(name),
                "weight": float(weight),
                "active": active,
                "parentFeatures": [],
                "forbiddenAncestors": forbidden,
                "allowedForStrictNeutral": active and not forbidden,
                "quotientVisible": active and not forbidden,
                "presentationInvariant": active and not forbidden,
                "claimBoundary": (
                    "Strict-neutral feature ancestry row. A nonempty forbiddenAncestors list blocks "
                    "strict neutral promotion even if the channel is later renamed."
                ),
            }
        )
    return rows


def _neutral_channel_ancestors(channel: str) -> list[str]:
    if channel in {"local_packet", "overlap_correspondence"}:
        # The current production patch adjacency is generated from S2 screen
        # points. The values are locality-preserving and gauge invariant, but
        # that support-selection ancestry must remain visible to strict gates.
        return ["screen_pixel_coordinate"]
    if channel.startswith("prime_geometric"):
        return ["prime_geometric_response"]
    if channel == "support_visible_modular":
        return ["support_visible_modular_response"]
    if channel == "repair_modular":
        return ["modular_depth"]
    if channel == "modular_response":
        return ["modular_depth"]
    if channel == "perturbation_response_tensor":
        return ["s2_cap_axis", "screen_pixel_coordinate"]
    return []


def _strict_neutral_feature_ancestry_blockers(manifest: list[dict[str, Any]]) -> list[str]:
    blockers = []
    for row in manifest:
        if not row.get("active"):
            continue
        forbidden = row.get("forbiddenAncestors") if isinstance(row.get("forbiddenAncestors"), list) else []
        if forbidden:
            blockers.append(f"forbidden_strict_neutral_feature_ancestry:{row.get('featureId')}:{','.join(map(str, forbidden))}")
    return blockers


def _neutral_channel_matrix(views: list[NeutralObserverView], key: str) -> np.ndarray:
    if not views:
        return np.zeros((0, 0), dtype=float)
    if key == "local_packet":
        return _stack_channel(views, "locality_packet_features")
    if key == "record":
        return _stack_channel(views, "record_transition_hist")
    if key == "record_signature":
        return _stack_channel(views, "record_signature_hist")
    if key == "object_packet":
        return _stack_channel(views, "object_packet_hist")
    if key == "boundary_packet":
        return _stack_channel(views, "boundary_packet_hash_hist")
    if key == "overlap_correspondence":
        return _stack_channel(views, "overlap_correspondence_hist")
    if key == "counterfactual":
        return _stack_channel(views, "counterfactual_hist")
    if key == "checkpoint":
        return _stack_channel(views, "checkpoint_transition_hist")
    if key == "sector":
        return _stack_channel(views, "sector_transition_hist")
    if key == "port_pair_lag":
        return _stack_channel(views, "port_pair_lag_hist")
    if key == "repair":
        return _stack_channel(views, "repair_response_hist")
    if key == "repair_spectrum":
        return _stack_channel(views, "repair_response_spectrum")
    if key == "repair_current_tensor":
        return _stack_channel(views, "repair_current_tensor")
    if key == "perturbation_response_tensor":
        return _stack_channel(views, "perturbation_response_tensor")
    if key == "first_passage_response":
        return _stack_channel(views, "first_passage_response_hist")
    if key == "modular_response":
        return _stack_channel(views, "modular_response_hist")
    if key == "prime_geometric_modular":
        return _stack_channel(views, "prime_geometric_modular_spectrum")
    if key == "prime_geometric_control_quotient":
        return _stack_channel(views, "prime_geometric_control_quotient_spectrum")
    if key == "prime_geometric_rank3":
        return _stack_channel(views, "prime_geometric_modular_spectrum")[:, :3]
    if key == "prime_geometric_rank4":
        return _stack_channel(views, "prime_geometric_modular_spectrum")[:, :4]
    if key == "prime_geometric_rank8":
        return _stack_channel(views, "prime_geometric_modular_spectrum")[:, :8]
    if key == "prime_geometric_rank16":
        return _stack_channel(views, "prime_geometric_modular_spectrum")[:, :16]
    if key == "prime_geometric_rank32":
        return _stack_channel(views, "prime_geometric_modular_spectrum")[:, :32]
    if key == "prime_geometric_control_quotient_rank3":
        return _stack_channel(views, "prime_geometric_control_quotient_spectrum")[:, :3]
    if key == "prime_geometric_control_quotient_rank4":
        return _stack_channel(views, "prime_geometric_control_quotient_spectrum")[:, :4]
    if key == "prime_geometric_control_quotient_rank8":
        return _stack_channel(views, "prime_geometric_control_quotient_spectrum")[:, :8]
    if key == "prime_geometric_control_quotient_rank16":
        return _stack_channel(views, "prime_geometric_control_quotient_spectrum")[:, :16]
    if key == "prime_geometric_control_quotient_rank32":
        return _stack_channel(views, "prime_geometric_control_quotient_spectrum")[:, :32]
    if key == "support_visible_modular":
        return _stack_channel(views, "support_visible_modular_spectrum")
    if key == "repair_modular":
        return _stack_channel(views, "repair_modular_spectrum")
    if key == "transition_token":
        return _stack_channel(views, "transition_token_hist")
    if key == "transition_token_persistent":
        return _stack_channel(views, "transition_token_persistent_hist")
    if key == "transition_affinity":
        return _stack_channel(views, "transition_affinity_hist")
    if key == "persistence":
        return _stack_channel(views, "persistence_features")
    if key == "scalar_readout":
        return _stack_channel(views, "scalar_readout_features")
    return np.zeros((len(views), 0), dtype=float)


def _stack_channel(views: list[NeutralObserverView], attr: str) -> np.ndarray:
    rows = [np.asarray(getattr(view, attr), dtype=float).reshape(-1) for view in views]
    width = max((row.size for row in rows), default=0)
    if width <= 0:
        return np.zeros((len(views), 0), dtype=float)
    return np.vstack([_pad(row, width) for row in rows])


def _neutral_channel_embedding(matrix: np.ndarray, key: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0] if matrix.ndim == 2 else 0, 0), dtype=float)
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    if _neutral_channel_kind(key) == "histogram":
        nonnegative = np.maximum(matrix, 0.0)
        totals = np.sum(nonnegative, axis=1, keepdims=True)
        probs = np.divide(nonnegative, totals, out=np.zeros_like(nonnegative), where=totals > 1e-12)
        return np.sqrt(np.maximum(probs, 0.0))
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, keepdims=True)
    standardized = np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 1e-12)
    row_observed = (np.linalg.norm(matrix, axis=1) > 1e-12).astype(float)[:, None]
    return np.hstack([standardized, row_observed])


def _neutral_embedding_has_pairwise_signal(embedded: np.ndarray) -> bool:
    embedded = np.asarray(embedded, dtype=float)
    if embedded.ndim != 2 or embedded.shape[0] == 0 or embedded.shape[1] == 0:
        return False
    embedded = np.where(np.isfinite(embedded), embedded, 0.0)
    if not np.any(np.linalg.norm(embedded, axis=1) > 1e-12):
        return False
    if embedded.shape[0] <= 1:
        return True
    return bool(np.any(np.std(embedded, axis=0) > 1e-12))


def _neutral_channel_kind(key: str) -> str:
    if key in {
        "repair_spectrum",
        "local_packet",
        "repair_current_tensor",
        "perturbation_response_tensor",
        "prime_geometric_modular",
        "prime_geometric_control_quotient",
        "prime_geometric_rank3",
        "prime_geometric_rank4",
        "prime_geometric_rank8",
        "prime_geometric_rank16",
        "prime_geometric_rank32",
        "prime_geometric_control_quotient_rank3",
        "prime_geometric_control_quotient_rank4",
        "prime_geometric_control_quotient_rank8",
        "prime_geometric_control_quotient_rank16",
        "prime_geometric_control_quotient_rank32",
        "support_visible_modular",
        "repair_modular",
        "persistence",
        "scalar_readout",
    }:
        return "signed"
    return "histogram"


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
        "diagnostic_target": "neutral_record_feature_quotient_dimension",
        "not_the_support_visible_chart_dimension": True,
        "does_not_measure": [
            "support_visible_s2_screen_dimension",
            "canonical_h3_spatial_chart_dimension",
            "forced_simulator_chart_dimension",
            "paper_D3_support_visible_lorentz_chart_receipt",
        ],
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
            "Strict neutral dimension diagnostic for the record-feature quotient after H3 fitted points, "
            "cap normals, S2 axes, screen coordinates, radial depth, and modular depth have been excluded. "
            "It is not a measurement of the support-visible S2/H3 chart dimension, which is the separate "
            "D3 Lorentz/H3 chart branch. The finite-regulator gate uses the median of correlation and "
            "local-MLE estimators plus a pairwise-gap bound, because planted 3D controls show a small "
            "finite-sample low bias in the correlation estimator. The spectral proxy is reported but not "
            "gated because it is not calibrated on the finite planted controls. This is not sufficient for "
            "strict neutral bulk without leakage, controls, and refinement gates."
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
        "s2_leakage_pass": bool(s2_corr is not None and abs(float(s2_corr)) < 0.05),
        "s2_leakage_audit_available": bool(s2_corr is not None),
        "h3_coordinates_used": False,
        "cap_normals_used": False,
        "screen_axes_used_in_primary_distance": False,
        "claim_boundary": (
            "Leakage audit compares primary neutral distance to S2 axes post hoc; axes are not used to "
            "build the primary distance. Missing or malformed axes fail the audit instead of being treated "
            "as evidence of no leakage."
        ),
    }


def strict_neutral_bulk_receipt(
    dimension: dict[str, Any],
    model_selection: dict[str, Any],
    leakage: dict[str, Any],
    controls: dict[str, Any],
    refinement: dict[str, Any],
    quotient_geometry: dict[str, Any] | None = None,
    channel_audit: dict[str, Any] | None = None,
    theory_alignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quotient_contract = quotient_geometry or {}
    quotient_contract_passed = quotient_contract.get(GEOMETRY_CONTRACT_RECEIPT) is True
    channel_audit_passed = (channel_audit or {}).get("duplicate_channel_gate_pass") is True
    feature_ancestry_passed = (channel_audit or {}).get("feature_ancestry_gate_pass") is True
    theory_alignment_passed = (
        (theory_alignment or {}).get("theory_required_channels_present") is True
    )
    refinement_passed = _canonical_strict_neutral_refinement_passed(refinement)
    passed = (
        dimension.get("estimators_agree_3d") is True
        and model_selection.get("best_model") == "H3"
        and model_selection.get("h3_beats_s2") is True
        and model_selection.get("h3_beats_h2_h4") is True
        and leakage.get("s2_leakage_pass") is True
        and channel_audit_passed
        and feature_ancestry_passed
        and theory_alignment_passed
        and controls.get("shuffled_records_fail") is True
        and controls.get("shuffled_transition_labels_fail") is True
        and controls.get("planted_2d_returns_2d") is True
        and controls.get("planted_3d_returns_3d") is True
        and controls.get("planted_h3_returns_h3") is True
        and refinement_passed
        and quotient_contract_passed
    )
    return {
        "receipt": "STRICT_NEUTRAL_BULK_RECEIPT",
        "strict_neutral_bulk": passed,
        "physical_claim": passed,
        GEOMETRY_CONTRACT_RECEIPT: quotient_contract_passed,
        "quotient_geometry_blockers": list(quotient_contract.get("blockers", [])),
        "duplicate_channel_gate_pass": channel_audit_passed,
        "duplicate_channel_blockers": list((channel_audit or {}).get("duplicate_pairs", [])),
        "feature_ancestry_gate_pass": feature_ancestry_passed,
        "feature_ancestry_blockers": list((channel_audit or {}).get("feature_ancestry_blockers", [])),
        "theory_required_channels_present": theory_alignment_passed,
        "theory_evidence_blockers": list((theory_alignment or {}).get("evidence_gaps", [])),
        "canonical_4k_16k_64k_256k_refinement_gate_pass": refinement_passed,
        "claim_boundary": (
            "Neutral third-person bulk reconstructed from observer-visible records without H3/cap-normal "
            "target features. It is intentionally stricter than, and does not negate, the support-visible "
            "S2/Lorentz/H3 chart receipt. This receipt is false unless quotient chart transport, metric "
            "validity, feature missingness, presentation invariance, split leakage, duplicate-channel, "
            "record-feature quotient dimension, model-selection, controls, and refinement gates all pass."
        ),
    }


def _canonical_strict_neutral_refinement_passed(refinement: dict[str, Any]) -> bool:
    required = [4_096, 16_384, 65_536, 262_144]
    sizes = refinement.get("sizes")
    if not isinstance(sizes, list) or len(sizes) != len(required):
        return False
    if any(
        not isinstance(row, dict)
        or type(row.get("patch_count")) is not int
        for row in sizes
    ):
        return False
    observed = sorted(row["patch_count"] for row in sizes)
    exact_true_fields = (
        "required_ladder_complete",
        "multi_scale",
        "all_control_quotient_spatial_3d_candidates",
        "all_candidate_s2_leakage_pass",
        "all_candidate_rank3_e3",
        "candidate_dimension_stable",
        "independent_rank3_selector_all",
        "proper_negative_control_all",
        "directional_h3_strict_all",
        "measured_overlap_geometry_all",
        "strict_neutral_bulk_refinement_receipt",
    )
    return bool(
        refinement.get("mode") == "prime_geometric_rank_refinement_v0"
        and refinement.get("required_patch_count_ladder") == required
        and observed == required
        and refinement.get("missing_required_patch_counts") == []
        and all(refinement.get(field) is True for field in exact_true_fields)
        and refinement.get("proof_blockers") == []
    )


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
        "E4": _euclidean_model_stress(work, dim=4, heldout=heldout),
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
    e4 = models.get("E4", {})
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
            and h3_stress < float(e4.get("heldout_stress", np.inf))
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
            and rows["planted_4d"]["model_selection"].get("selected_model") == "E4"
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
    channel_audit = neutral_channel_duplicate_audit(neutral_views, weights)
    theory_alignment = _strict_neutral_theory_alignment_report(
        neutral_views,
        weights,
        observer_views=observer_views,
    )
    quotient_contract = _neutral_quotient_geometry_contract(
        distance,
        neutral_views,
        observer_views=observer_views,
        controls=controls or {},
        refinement=refinement or {},
        weights=weights,
    )
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
        quotient_contract,
        channel_audit,
        theory_alignment,
    )
    return {
        "mode": "strict_neutral_bulk_record_transition_audit",
        "observer_count": len(neutral_views),
        "distance_matrix_shape": list(distance.shape),
        "neutral_metric_construction": "fixed_weighted_euclidean_feature_embedding_v1",
        "channel_audit": channel_audit,
        "strict_neutral_theory_alignment": theory_alignment,
        "strict_neutral_theory_evidence_gaps": list(theory_alignment.get("evidence_gaps", [])),
        "quotient_geometry_contract": quotient_contract,
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
            "boundary_packet_hash_hist",
            "overlap_correspondence_hist",
            "counterfactual_hist",
            "checkpoint_transition_hist",
            "sector_transition_hist",
            "port_pair_lag_hist",
            "repair_response_hist",
            "repair_response_spectrum",
            "repair_current_tensor",
            "perturbation_response_tensor",
            "first_passage_response_hist",
            "persistence_features",
            "scalar_readout_features",
        ],
        "diagnostic_only_features": [
            "modular_response_hist",
            "prime_geometric_modular_spectrum",
            "prime_geometric_control_quotient_spectrum",
            "prime_geometric_rank_prefixes",
            "prime_geometric_control_quotient_rank_prefixes",
            "support_visible_modular_spectrum",
            "repair_modular_spectrum",
            "transition_token_hist",
            "transition_token_persistent_hist",
            "transition_affinity_hist",
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
        "chart_boundary": (
            "This report audits an observer-record-only quotient after excluding the simulator's "
            "support-visible S2/H3 chart features. A high neutral record-feature dimension is not a "
            "measurement of, or contradiction to, the forced support-visible 3+1D Lorentz/H3 chart."
        ),
    }


def _neutral_quotient_geometry_contract(
    distance: np.ndarray,
    neutral_views: list[NeutralObserverView],
    *,
    observer_views: list[dict[str, Any]],
    controls: dict[str, Any],
    refinement: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    active_weights = weights or DEFAULT_NEUTRAL_WEIGHTS
    channel_manifest = [
        ChannelMetricSpec(name=name, weight=weight, missingness="fixed_missing_symbol")
        for name, weight in active_weights.items()
        if float(weight) > 0.0
    ]
    quotient_inputs = _neutral_quotient_geometry_inputs(
        observer_views,
        active_weights=active_weights,
        controls=controls,
        refinement=refinement,
    )
    return quotient_geometry_certificate(
        distance,
        quotient_ids=[str(view.observer_id) for view in neutral_views],
        channel_manifest=channel_manifest,
        metric_mode="complete_case",
        jointly_separating=False,
        missingness_quotient_visible=True,
        atlas_receipt=quotient_inputs["atlas_receipt"],
        feature_receipt=quotient_inputs["feature_receipt"],
        invariance_receipt=quotient_inputs["invariance_receipt"],
        refinement_receipt=refinement,
        statistics_receipt=quotient_inputs["statistics_receipt"],
        require_metric=True,
        require_euclidean=False,
    )


def _neutral_quotient_geometry_inputs(
    observer_views: list[dict[str, Any]],
    *,
    active_weights: dict[str, float],
    controls: dict[str, Any],
    refinement: dict[str, Any],
) -> dict[str, Any]:
    """Produce the contract inputs that are actually supported by a run.

    Unknown schedule/partition and boundary-port transports are omitted rather
    than filled with zeros. The quotient verifier therefore stays fail closed,
    while the report now distinguishes produced evidence from missing evidence.
    """

    patch_rows = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    locality_schemas = [
        row.get("locality_preserving_packet_feature_schema")
        for row in patch_rows
        if isinstance(row.get("locality_preserving_packet_feature_schema"), dict)
    ]
    shared_locality_frame = bool(
        patch_rows
        and len(locality_schemas) == len(patch_rows)
        and all(schema == locality_schemas[0] for schema in locality_schemas[1:])
    )
    active_channels = {
        str(name) for name, weight in active_weights.items() if float(weight) > 0.0
    }
    frame_proven_channels = {"local_packet"}
    atlas_complete = bool(shared_locality_frame and active_channels <= frame_proven_channels)
    atlas_receipt: dict[str, Any] = {
        "producer": "neutral_global_shared_feature_frame_v1",
        "shared_locality_frame": shared_locality_frame,
        "active_channels": sorted(active_channels),
        "unproven_transport_channels": sorted(active_channels - frame_proven_channels),
    }
    if atlas_complete:
        atlas_receipt.update(
            {
                "identity_defect": 0.0,
                "inverse_defect": 0.0,
                "cocycle_defect": 0.0,
                "cycle_holonomy_defect": 0.0,
            }
        )

    overlap_evidence = _overlap_evidence_report(patch_rows)
    feature_receipt: dict[str, Any] = {
        "producer": "neutral_feature_descent_v1",
        "quotient_visible_missingness": True,
        "measured_overlap_evidence": overlap_evidence,
    }
    feature_transport_proven = bool(
        active_channels <= frame_proven_channels and shared_locality_frame
    )
    if feature_transport_proven:
        feature_receipt["max_transport_defect"] = 0.0

    invariance_receipt: dict[str, Any] = {
        "producer": "neutral_presentation_invariance_partial_v1",
        "order_distortion": 0.0,
        "known_exact_invariances": ["within_observer_packet_order"],
        "unproven_invariances": [
            "gauge",
            "port_relabeling",
            "repair_schedule",
            "partition_or_shard_presentation",
        ],
    }
    presentation = _measured_overlap_presentation_invariance_report(
        patch_rows,
        np.random.default_rng(0),
    )
    if presentation.get("receipt", False):
        invariance_receipt["observer_relabel_distortion"] = float(
            presentation.get("global_observer_relabel_distortion", 0.0)
        )

    positive_controls = all(
        controls.get(name, False)
        for name in (
            "planted_2d_returns_2d",
            "planted_3d_returns_3d",
            "planted_h3_returns_h3",
        )
    )
    negative_controls = all(
        controls.get(name, False)
        for name in (
            "shuffled_records_fail",
            "shuffled_transition_labels_fail",
        )
    )
    statistics_receipt = {
        "producer": "strict_neutral_control_suite_v1",
        "ancestry_leakage_count": 0,
        "test_used_once": False,
        "positive_controls_passed": bool(positive_controls),
        "negative_controls_passed": bool(negative_controls),
        "split_assignment_producer_available": False,
    }
    return {
        "atlas_receipt": atlas_receipt,
        "feature_receipt": feature_receipt,
        "invariance_receipt": invariance_receipt,
        "statistics_receipt": statistics_receipt,
        "refinement_receipt": refinement,
    }


def write_strict_neutral_bulk_report(
    run_dir: Path,
    out: Path | None = None,
    *,
    seed: int = 1,
    max_model_points: int = 512,
    planted_control_points: int = 160,
    max_observers: int | None = None,
) -> dict[str, Any]:
    run = Path(run_dir)
    observer_path = run / "observer_views.jsonl"
    if not observer_path.exists():
        raise FileNotFoundError(observer_path)
    source_observer_views = _read_jsonl(observer_path)
    observer_views, analysis_population = bounded_strict_neutral_observer_views(
        source_observer_views,
        max_observers=max_observers,
    )
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
    refinement_path = run / "prime_geometric_rank_refinement_report.json"
    refinement_report = _read_json(refinement_path) if refinement_path.exists() else {}
    analysis_parameters = {
        "seed": int(seed),
        "max_model_points": int(max_model_points),
        "planted_control_points": int(planted_control_points),
    }
    if max_observers is not None:
        analysis_parameters["max_observers"] = int(max_observers)
    source_manifest = {
        "schema": (
            STRICT_NEUTRAL_SAMPLED_SOURCE_SCHEMA
            if max_observers is not None
            else STRICT_NEUTRAL_SOURCE_SCHEMA
        ),
        "observer_views_path": observer_path.name,
        "observer_views_sha256": _neutral_source_file_sha256(observer_path),
        "observer_view_row_count": len(source_observer_views),
        "analysis_population": analysis_population,
        "analysis_kernel_file_sha256": _neutral_source_file_sha256(Path(__file__)),
        "analysis_parameters": analysis_parameters,
        "refinement_input": {
            "path": refinement_path.name,
            "sha256": (
                _neutral_source_file_sha256(refinement_path)
                if refinement_path.is_file()
                else None
            ),
            "provenance": "derived_report_hash_only",
            "primitive_replay_available": False,
        },
        "claim_boundary": (
            "The observer JSONL and deterministic neutral-analysis parameters are hash-bound and "
            "can be replayed independently. The current rank-refinement report is only hash-bound "
            "derived data; it cannot promote strict neutral bulk until its own primitive replay "
            "chain is implemented."
        ),
    }
    source_manifest_path = run / "strict_neutral_source_manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = strict_neutral_bulk_report(
        observer_views,
        controls=control_flags,
        refinement=refinement_report,
        seed=seed,
        max_model_points=max_model_points,
    )
    report["planted_controls"] = planted
    report["shuffled_controls"] = run_controls
    report["source_run_dir"] = str(run)
    report["observer_views_path"] = str(observer_path)
    report["refinement_report_path"] = str(refinement_path)
    report["source_artifact"] = source_manifest
    report["source_manifest_path"] = str(source_manifest_path)
    report["blockers"] = _strict_neutral_blockers(report)
    destination = Path(out) if out is not None else run / "strict_neutral_bulk_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def bounded_strict_neutral_observer_views(
    observer_views: list[dict[str, Any]],
    *,
    max_observers: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a deterministic, hash-bound patch-observer analysis cohort.

    Strict-neutral feature and overlap distances are dense.  The pipeline's
    declared analysis cohort must therefore bound the matrix before any
    quadratic construction, rather than merely subsampling the later model
    fit.  Ranking by observer ID matches the nested population policy used by
    the production observer writer.
    """

    patch_views = [
        view for view in observer_views if view.get("view_type") == "patch_observer"
    ]
    observer_ids = [
        int(view.get("observer_id", index)) for index, view in enumerate(patch_views)
    ]
    indices = deterministic_observer_analysis_indices(
        observer_ids,
        max_observers=max_observers,
    )
    selected = [patch_views[int(index)] for index in indices]
    selected_ids = [
        int(view.get("observer_id", index)) for index, view in enumerate(selected)
    ]
    metadata = {
        "schema": "strict_neutral_deterministic_observer_cohort_v1",
        "source_row_count": int(len(observer_views)),
        "source_patch_observer_count": int(len(patch_views)),
        "requested_max_observers": (
            int(max_observers) if max_observers is not None else None
        ),
        "analyzed_observer_count": int(len(selected)),
        "sampling_policy": (
            "all_materialized_patch_observers"
            if len(selected) == len(patch_views)
            else "deterministic_observer_id_hash_rank_v1"
        ),
        "observer_ids": selected_ids,
        "observer_id_subset_hash": stable_json_hash(
            {
                "schema": "strict_neutral_deterministic_observer_cohort_v1",
                "source_patch_observer_count": len(patch_views),
                "requested_max_observers": max_observers,
                "observer_ids": selected_ids,
            }
        ),
    }
    return selected, metadata


def _neutral_source_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


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
    control_residualized_rank3_candidate = bool(
        (
            (_best_rank_sweep_row(quotient_coordinate_3d_rows) or {}).get("rank") == 3
        )
        and control_quotient_spatial_3d_candidate
    )
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
            "physical_claim": False,
            "strict_neutral_bulk_participation": "diagnostic_only",
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
        "CONTROL_RESIDUALIZED_RANK3_CANDIDATE_RECEIPT": control_residualized_rank3_candidate,
        "CONTROL_RESIDUALIZED_RANK3_DIAGNOSTIC_RECEIPT": control_residualized_rank3_candidate,
        "control_residualized_rank3_candidate_receipt": control_residualized_rank3_candidate,
        "control_residualized_rank3_diagnostic_receipt": control_residualized_rank3_candidate,
        "control_residualized_rank3_claim_boundary": (
            "Residualized signal construction: target response with finite-regulator control directions "
            "removed. This is not a negative control and is diagnostic-only."
        ),
        "prime_geometric_strict_neutral_candidate_receipt": strict_neutral_candidate,
        "selected_rank_controls": selected_rank_controls,
        "proof_blockers": proof_blockers,
        "physical_claim": False,
        "strict_neutral_bulk_participation": "diagnostic_only",
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
    required_ladder = (4_096, 16_384, 65_536, 262_144)
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
        if int(row.get("patch_count") or 0) in required_ladder
        and row.get("median_candidate_dimension") is not None
    ]
    dimension_drift = float(max(size_medians) - min(size_medians)) if len(size_medians) >= 2 else None
    all_candidates = bool(rows and all(row.get("control_quotient_spatial_3d_candidate") for row in rows))
    all_leakage = bool(rows and all(row.get("candidate_s2_leakage_pass") for row in rows))
    all_rank3_e3 = bool(
        rows
        and all(row.get("candidate_rank") == 3 for row in rows)
        and all(row.get("candidate_model") == "E3" for row in rows)
    )
    available_patch_counts = {int(size.get("patch_count") or 0) for size in sizes}
    missing_ladder_patch_counts = sorted(set(required_ladder) - available_patch_counts)
    ladder_complete = not missing_ladder_patch_counts
    multi_scale = ladder_complete
    stable_dimension = bool(
        ladder_complete
        and len(size_medians) == len(required_ladder)
        and dimension_drift is not None
        and dimension_drift <= 0.10
    )
    refinement_candidate = bool(
        multi_scale
        and all_candidates
        and all_leakage
        and all_rank3_e3
        and stable_dimension
    )
    independent_rank3_all = bool(rows and all(row.get("independent_rank3_selector") for row in rows))
    proper_negative_control_all = bool(
        rows and all(row.get("control_quotient_lane_is_negative_control") for row in rows)
    )
    directional_h3_strict_all = bool(
        rows and all(int(row.get("directional_strict_3d_ready_count") or 0) > 0 for row in rows)
    )
    measured_overlap_all = bool(
        rows and all(row.get("measured_overlap_geometry_receipt") for row in rows)
    )
    strict_refinement_receipt = bool(
        refinement_candidate
        and independent_rank3_all
        and proper_negative_control_all
        and directional_h3_strict_all
        and measured_overlap_all
    )
    blockers: list[str] = []
    if not ladder_complete:
        blockers.append(
            "required_4k_16k_64k_256k_refinement_ladder_incomplete:"
            + ",".join(str(value) for value in missing_ladder_patch_counts)
        )
    if not refinement_candidate:
        blockers.append("control_quotient_rank3_candidate_not_stable_across_refinement")
    if not independent_rank3_all:
        blockers.append("independent_svd_rank3_selector_not_stable_or_false")
    if not proper_negative_control_all:
        blockers.append("control_quotient_lane_is_not_a_negative_control")
    if not directional_h3_strict_all:
        blockers.append("directional_h3_strict_rank_gate_not_passed")
    if not measured_overlap_all:
        blockers.append("measured_cross_observer_overlap_refinement_gate_not_passed")
    return {
        "mode": "prime_geometric_rank_refinement_v0",
        "run_count": len(rows),
        "rows": rows,
        "sizes": sizes,
        "required_patch_count_ladder": list(required_ladder),
        "missing_required_patch_counts": missing_ladder_patch_counts,
        "required_ladder_complete": ladder_complete,
        "multi_scale": multi_scale,
        "all_control_quotient_spatial_3d_candidates": all_candidates,
        "all_candidate_s2_leakage_pass": all_leakage,
        "all_candidate_rank3_e3": all_rank3_e3,
        "candidate_dimension_drift": dimension_drift,
        "candidate_dimension_stable": stable_dimension,
        "control_quotient_rank3_refinement_candidate_receipt": refinement_candidate,
        "CONTROL_RESIDUALIZED_RANK3_CANDIDATE_RECEIPT": refinement_candidate,
        "CONTROL_RESIDUALIZED_RANK3_DIAGNOSTIC_RECEIPT": refinement_candidate,
        "control_residualized_rank3_candidate_receipt": refinement_candidate,
        "control_residualized_rank3_diagnostic_receipt": refinement_candidate,
        "independent_rank3_selector_all": independent_rank3_all,
        "proper_negative_control_all": proper_negative_control_all,
        "directional_h3_strict_all": directional_h3_strict_all,
        "measured_overlap_geometry_all": measured_overlap_all,
        "strict_neutral_bulk_refinement_receipt": strict_refinement_receipt,
        "proof_blockers": blockers,
        "physical_claim": False,
        "strict_neutral_bulk_participation": "diagnostic_only",
        "claim_boundary": (
            "Refinement diagnostic for the control-quotient coordinate rank-3 spatial window. Passing "
            "this report means the finite-regulator candidate is stable across the explicit "
            "4k/16k/64k/256k ladder with dimension drift at most 0.10. "
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


def neutral_3d_bulk_audit_report(report_paths: list[Path]) -> dict[str, Any]:
    """Compact audit for promotion from neutral rank diagnostics to strict 3D bulk."""

    paths = [Path(path) for path in report_paths]
    sweep_paths = _find_prime_rank_sweep_reports(paths)
    sweeps = [
        (path, _read_json(path))
        for path in sweep_paths
    ]
    sweeps = [(path, report) for path, report in sweeps if report.get("mode") == "prime_geometric_rank_sweep_v0"]
    refinement_reports = [
        _read_json(path)
        for path in _find_prime_rank_refinement_reports(paths)
    ]
    refinement_reports = [
        report for report in refinement_reports if report.get("mode") == "prime_geometric_rank_refinement_v0"
    ]
    overlap_control_reports = [
        _read_json(path)
        for path in _find_overlap_native_control_reports(paths)
    ]
    overlap_control_reports = [
        report for report in overlap_control_reports if report.get("mode") == "overlap_native_neutral_control_v0"
    ]
    if refinement_reports:
        refinement = _select_neutral_refinement_report(refinement_reports)
        refinement_source = "existing_prime_geometric_rank_refinement_report"
    elif sweep_paths:
        refinement = prime_geometric_rank_refinement_report(sweep_paths)
        refinement_source = "computed_from_supplied_sweep_reports"
    else:
        refinement = {}
        refinement_source = "missing"

    sweep_summaries = [_neutral_3d_sweep_summary(path, report) for path, report in sweeps]
    blockers = _neutral_3d_bulk_audit_blockers(
        refinement,
        [report for _, report in sweeps],
        overlap_control_reports=overlap_control_reports,
    )
    overlap_summary = _overlap_native_control_audit_summary(overlap_control_reports)
    report = {
        "mode": "neutral_3d_bulk_audit_v0",
        "report_paths": [str(path) for path in paths],
        "sweep_report_count": len(sweeps),
        "refinement_source": refinement_source,
        "strict_neutral_bulk_ready": bool(
            refinement.get("strict_neutral_bulk_refinement_receipt", False) and not blockers
        ),
        "strict_neutral_bulk": False,
        "physical_claim": False,
        "control_residualized_rank3_refinement_candidate": bool(
            refinement.get("control_quotient_rank3_refinement_candidate_receipt", False)
        ),
        "control_residualized_rank3_independent_selector_all": bool(
            refinement.get("independent_rank3_selector_all", False)
        ),
        "control_residualized_rank3_dimension_drift": refinement.get("candidate_dimension_drift"),
        "directional_strict_ready_total": int(
            sum(int(report.get("strict_3d_ready_count") or 0) for _, report in sweeps)
        ),
        "control_quotient_candidate_count": int(
            sum(
                1
                for _, report in sweeps
                if bool(report.get("prime_geometric_control_quotient_spatial_3d_candidate_receipt", False))
            )
        ),
        "overlap_native_negative_control_report_count": overlap_summary["report_count"],
        "overlap_native_negative_control_receipt_count": overlap_summary["receipt_count"],
        "overlap_native_negative_control_receipt_all": overlap_summary["receipt_all"],
        "overlap_native_spatial_3d_candidate_count": overlap_summary["spatial_3d_candidate_count"],
        "overlap_native_strict_h3_candidate_count": overlap_summary["strict_h3_candidate_count"],
        "overlap_native_control_summary": overlap_summary,
        "blockers": blockers,
        "refinement_summary": _neutral_3d_refinement_summary(refinement),
        "sweeps": sweep_summaries,
        "claim_boundary": (
            "Audit for strict neutral 3D bulk promotion. Control-residualized rank-3 rows are treated "
            "as diagnostic candidates only. Strict neutral bulk remains false until independent rank "
            "selection, proper negative/null controls, directional H3 gates, and refinement all pass."
        ),
    }
    return report


def write_neutral_3d_bulk_audit_report(report_paths: list[Path], out: Path) -> dict[str, Any]:
    report = neutral_3d_bulk_audit_report(report_paths)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "neutral_3d_bulk_audit_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "neutral_3d_bulk_audit_report.md").write_text(
        _neutral_3d_bulk_audit_markdown(report),
        encoding="utf-8",
    )
    return report


def neutral_independent_rank_selector_audit_report(report_paths: list[Path]) -> dict[str, Any]:
    """Audit whether neutral rank-3 candidates survive the independent SVD selector."""

    paths = _find_prime_rank_sweep_reports([Path(path) for path in report_paths])
    rows = [
        _neutral_rank_selector_row(path, report)
        for path, report in ((path, _read_json(path)) for path in paths)
        if report.get("mode") == "prime_geometric_rank_sweep_v0"
    ]
    run_count = len(rows)
    prime_rank3_count = int(sum(1 for row in rows if row.get("prime_rank3_selector_receipt")))
    control_rank3_count = int(sum(1 for row in rows if row.get("control_quotient_rank3_selector_receipt")))
    control_candidate_count = int(sum(1 for row in rows if row.get("control_quotient_rank3_candidate")))
    prime_rank3_all = bool(run_count and prime_rank3_count == run_count)
    control_rank3_all = bool(run_count and control_rank3_count == run_count)
    control_rank3_ev = [
        float(row["control_quotient_rank3_cumulative_explained_variance"])
        for row in rows
        if row.get("control_quotient_rank3_cumulative_explained_variance") is not None
    ]
    control_effective = [
        float(row["control_quotient_effective_rank"])
        for row in rows
        if row.get("control_quotient_effective_rank") is not None
    ]
    control_participation = [
        float(row["control_quotient_participation_rank"])
        for row in rows
        if row.get("control_quotient_participation_rank") is not None
    ]
    blockers: list[str] = []
    if not rows:
        blockers.append("missing_prime_geometric_rank_sweep_reports")
    if rows and not prime_rank3_all:
        blockers.append("prime_geometric_independent_rank3_selector_not_all_true")
    if rows and not control_rank3_all:
        blockers.append("control_quotient_independent_rank3_selector_not_all_true")
    max_control_ev = max(control_rank3_ev) if control_rank3_ev else None
    min_control_effective = min(control_effective) if control_effective else None
    if max_control_ev is not None and max_control_ev < 0.20:
        blockers.append("control_quotient_rank3_cumulative_explained_variance_low")
    if min_control_effective is not None and min_control_effective > 10.0:
        blockers.append("control_quotient_effective_rank_not_low_dimensional")
    if rows and any(not bool(row.get("control_quotient_lane_is_negative_control")) for row in rows):
        blockers.append("control_quotient_lane_is_not_a_negative_control")
    receipt = bool(run_count and prime_rank3_all and control_rank3_all and not blockers)
    return {
        "mode": "neutral_independent_rank_selector_audit_v0",
        "report_paths": [str(path) for path in paths],
        "run_count": run_count,
        "NEUTRAL_INDEPENDENT_RANK3_SELECTOR_RECEIPT": receipt,
        "neutral_independent_rank3_selector_receipt": receipt,
        "physical_claim": False,
        "strict_neutral_bulk": False,
        "prime_geometric_rank3_selector_count": prime_rank3_count,
        "control_quotient_rank3_selector_count": control_rank3_count,
        "control_quotient_rank3_candidate_count": control_candidate_count,
        "prime_geometric_rank3_selector_all": prime_rank3_all,
        "control_quotient_rank3_selector_all": control_rank3_all,
        "prime_geometric_largest_gap_rank_counts": _counts(
            row.get("prime_geometric_largest_gap_rank") for row in rows
        ),
        "prime_geometric_chord_elbow_rank_counts": _counts(
            row.get("prime_geometric_chord_elbow_rank") for row in rows
        ),
        "control_quotient_largest_gap_rank_counts": _counts(
            row.get("control_quotient_largest_gap_rank") for row in rows
        ),
        "control_quotient_chord_elbow_rank_counts": _counts(
            row.get("control_quotient_chord_elbow_rank") for row in rows
        ),
        "control_quotient_max_rank3_cumulative_explained_variance": max_control_ev,
        "control_quotient_median_rank3_cumulative_explained_variance": _median_or_none(control_rank3_ev),
        "control_quotient_min_effective_rank": min_control_effective,
        "control_quotient_median_effective_rank": _median_or_none(control_effective),
        "control_quotient_median_participation_rank": _median_or_none(control_participation),
        "rows": rows,
        "blockers": _unique_preserve_order(blockers),
        "claim_boundary": (
            "Independent neutral rank-selector audit. Coordinate rank-3 windows and control-residualized "
            "rank-3 candidates are diagnostic only unless the independent singular-value selector also "
            "selects rank 3 across regulator sizes, with proper negative controls and strict neutral gates."
        ),
    }


def write_neutral_independent_rank_selector_audit_report(
    report_paths: list[Path],
    out: Path,
) -> dict[str, Any]:
    report = neutral_independent_rank_selector_audit_report(report_paths)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "neutral_independent_rank_selector_audit_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "neutral_independent_rank_selector_audit_report.md").write_text(
        _neutral_independent_rank_selector_markdown(report),
        encoding="utf-8",
    )
    return report


def strict_neutral_bulk_frontier_report(report_paths: list[Path]) -> dict[str, Any]:
    """Summarize the current strict-neutral-bulk proof frontier.

    The neutral audit is intentionally conservative and can be hard to read at
    a glance. This report keeps the same claim boundary, but turns the evidence
    into named gates so a run pack can show exactly which receipts are present
    and which are still blocking promotion.
    """

    paths = [Path(path) for path in report_paths]
    audits = [
        _read_json(path)
        for path in _find_neutral_3d_bulk_audit_reports(paths)
    ]
    audits = [report for report in audits if report.get("mode") == "neutral_3d_bulk_audit_v0"]
    selectors = [
        _read_json(path)
        for path in _find_neutral_independent_rank_selector_reports(paths)
    ]
    selectors = [
        report for report in selectors if report.get("mode") == "neutral_independent_rank_selector_audit_v0"
    ]
    graph_reports = [
        _read_json(path)
        for path in _find_overlap_native_graph_geometry_reports(paths)
    ]
    graph_reports = [
        report for report in graph_reports if report.get("mode") == "overlap_native_graph_geometry_v0"
    ]
    residual_graph_reports = [
        _read_json(path)
        for path in _find_overlap_residualized_graph_geometry_reports(paths)
    ]
    residual_graph_reports = [
        report for report in residual_graph_reports if report.get("mode") == "overlap_residualized_graph_geometry_v0"
    ]
    audit = _select_neutral_3d_audit_report(audits) if audits else neutral_3d_bulk_audit_report(paths)
    selector = (
        _select_neutral_selector_report(selectors)
        if selectors
        else neutral_independent_rank_selector_audit_report(paths)
    )
    refinement = audit.get("refinement_summary") if isinstance(audit.get("refinement_summary"), dict) else {}
    overlap_receipt = bool(audit.get("overlap_native_negative_control_receipt_all", False))
    overlap_spatial = int(audit.get("overlap_native_spatial_3d_candidate_count") or 0)
    graph_summary = _overlap_graph_geometry_summary(graph_reports)
    residual_graph_summary = _overlap_residual_graph_geometry_summary(residual_graph_reports)
    gate_rows = [
        {
            "gate": "control_residualized_rank3_refinement_candidate",
            "passed": bool(audit.get("control_residualized_rank3_refinement_candidate", False)),
            "detail": "stable rank-3 diagnostic candidate across supplied finite regulators",
        },
        {
            "gate": "candidate_dimension_stable",
            "passed": bool(refinement.get("candidate_dimension_stable", False)),
            "detail": f"dimension drift={audit.get('control_residualized_rank3_dimension_drift')}",
        },
        {
            "gate": "overlap_native_negative_controls",
            "passed": overlap_receipt,
            "detail": (
                f"{audit.get('overlap_native_negative_control_receipt_count', 0)}/"
                f"{audit.get('overlap_native_negative_control_report_count', 0)} overlap-control receipts"
            ),
        },
        {
            "gate": "overlap_native_raw_spatial_3d",
            "passed": bool(overlap_spatial > 0),
            "detail": f"{overlap_spatial} raw-overlap spatial-3D candidates",
        },
        {
            "gate": "overlap_native_graph_geometry",
            "passed": bool(graph_summary["receipt_all"]),
            "detail": (
                f"{graph_summary['receipt_count']}/{graph_summary['report_count']} graph receipts; "
                f"{graph_summary['spatial_3d_candidate_count']} spatial-3D candidates; "
                f"{graph_summary['model_order_rank3_selector_count']} model-order rank-3 selectors"
            ),
        },
        {
            "gate": "overlap_native_graph_strict_h3",
            "passed": bool(graph_summary["strict_h3_candidate_count"] > 0),
            "detail": f"{graph_summary['strict_h3_candidate_count']} strict-H3 graph candidates",
        },
        {
            "gate": "overlap_residualized_graph_geometry",
            "passed": bool(residual_graph_summary["receipt_all"]),
            "detail": (
                f"{residual_graph_summary['receipt_count']}/{residual_graph_summary['report_count']} "
                f"residualized graph receipts; {residual_graph_summary['spatial_3d_candidate_count']} "
                "spatial-3D candidates; "
                f"{residual_graph_summary['model_order_rank3_selector_count']} model-order rank-3 selectors"
            ),
        },
        {
            "gate": "overlap_residualized_graph_strict_h3",
            "passed": bool(residual_graph_summary["strict_h3_candidate_count"] > 0),
            "detail": f"{residual_graph_summary['strict_h3_candidate_count']} strict-H3 residual graph candidates",
        },
        {
            "gate": "independent_rank3_selector",
            "passed": bool(selector.get("NEUTRAL_INDEPENDENT_RANK3_SELECTOR_RECEIPT", False)),
            "detail": (
                f"control selector count={selector.get('control_quotient_rank3_selector_count', 0)}; "
                f"candidate count={selector.get('control_quotient_rank3_candidate_count', 0)}"
            ),
        },
        {
            "gate": "directional_h3_strict_gate",
            "passed": int(audit.get("directional_strict_ready_total") or 0) > 0,
            "detail": f"strict-ready directional rows={audit.get('directional_strict_ready_total', 0)}",
        },
        {
            "gate": "strict_neutral_bulk_ready",
            "passed": bool(audit.get("strict_neutral_bulk_ready", False)),
            "detail": "all hard promotion gates passed",
        },
    ]
    blockers = _unique_preserve_order(
        [str(blocker) for blocker in audit.get("blockers", [])]
        + [str(blocker) for blocker in selector.get("blockers", [])]
    )
    if graph_reports and graph_summary["strict_h3_candidate_count"] <= 0:
        blockers = _unique_preserve_order(blockers + ["overlap_graph_strict_h3_candidate_false"])
    if residual_graph_reports and residual_graph_summary["strict_h3_candidate_count"] <= 0:
        blockers = _unique_preserve_order(blockers + ["overlap_residual_graph_strict_h3_candidate_false"])
    gate_gap_rows = _strict_neutral_frontier_gap_rows(
        gate_rows=gate_rows,
        blockers=blockers,
        graph_summary=graph_summary,
        residual_graph_summary=residual_graph_summary,
        selector=selector,
        audit=audit,
    )
    return {
        "mode": "strict_neutral_bulk_frontier_v0",
        "report_paths": [str(path) for path in paths],
        "STRICT_NEUTRAL_BULK_FRONTIER_REPORT": True,
        "strict_neutral_bulk": bool(audit.get("strict_neutral_bulk", False)),
        "strict_neutral_bulk_ready": bool(audit.get("strict_neutral_bulk_ready", False)),
        "control_residualized_rank3_refinement_candidate": bool(
            audit.get("control_residualized_rank3_refinement_candidate", False)
        ),
        "control_residualized_rank3_dimension_drift": audit.get("control_residualized_rank3_dimension_drift"),
        "overlap_native_negative_control_receipt_all": overlap_receipt,
        "overlap_native_spatial_3d_candidate_count": overlap_spatial,
        "overlap_native_graph_geometry_report_count": graph_summary["report_count"],
        "overlap_native_graph_geometry_receipt_count": graph_summary["receipt_count"],
        "overlap_native_graph_geometry_receipt_all": graph_summary["receipt_all"],
        "overlap_native_graph_spatial_3d_candidate_count": graph_summary["spatial_3d_candidate_count"],
        "overlap_native_graph_strict_h3_candidate_count": graph_summary["strict_h3_candidate_count"],
        "overlap_native_graph_rank3_selector_count": graph_summary["rank3_selector_count"],
        "overlap_native_graph_model_order_rank3_selector_count": graph_summary[
            "model_order_rank3_selector_count"
        ],
        "overlap_native_graph_nontrivial_model_order_rank3_selector_count": graph_summary[
            "nontrivial_model_order_rank3_selector_count"
        ],
        "overlap_native_graph_summary": graph_summary,
        "overlap_residualized_graph_geometry_report_count": residual_graph_summary["report_count"],
        "overlap_residualized_graph_geometry_receipt_count": residual_graph_summary["receipt_count"],
        "overlap_residualized_graph_geometry_receipt_all": residual_graph_summary["receipt_all"],
        "overlap_residualized_graph_spatial_3d_candidate_count": residual_graph_summary[
            "spatial_3d_candidate_count"
        ],
        "overlap_residualized_graph_strict_h3_candidate_count": residual_graph_summary[
            "strict_h3_candidate_count"
        ],
        "overlap_residualized_graph_rank3_selector_count": residual_graph_summary["rank3_selector_count"],
        "overlap_residualized_graph_model_order_rank3_selector_count": residual_graph_summary[
            "model_order_rank3_selector_count"
        ],
        "overlap_residualized_graph_nontrivial_model_order_rank3_selector_count": residual_graph_summary[
            "nontrivial_model_order_rank3_selector_count"
        ],
        "overlap_residualized_graph_summary": residual_graph_summary,
        "neutral_independent_rank3_selector_receipt": bool(
            selector.get("NEUTRAL_INDEPENDENT_RANK3_SELECTOR_RECEIPT", False)
        ),
        "control_quotient_rank3_candidate_count": int(
            selector.get("control_quotient_rank3_candidate_count")
            or audit.get("control_quotient_candidate_count")
            or 0
        ),
        "control_quotient_rank3_selector_count": int(
            selector.get("control_quotient_rank3_selector_count") or 0
        ),
        "directional_strict_ready_total": int(audit.get("directional_strict_ready_total") or 0),
        "gate_rows": gate_rows,
        "gate_gap_rows": gate_gap_rows,
        "blockers": blockers,
        "next_missing_receipts": _strict_neutral_frontier_next_steps(blockers),
        "claim_boundary": (
            "Strict-neutral-bulk frontier report. It distinguishes overlap-native negative-control "
            "receipts and stable rank-3 diagnostics from the missing hard proof gates. It does not "
            "promote the diagnostic quotient or theorem-assisted H3 viewer to strict neutral bulk."
        ),
    }


def write_strict_neutral_bulk_frontier_report(report_paths: list[Path], out: Path) -> dict[str, Any]:
    report = strict_neutral_bulk_frontier_report(report_paths)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "strict_neutral_bulk_frontier_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "strict_neutral_bulk_frontier_report.md").write_text(
        _strict_neutral_bulk_frontier_markdown(report),
        encoding="utf-8",
    )
    return report


def _strict_neutral_frontier_gap_rows(
    *,
    gate_rows: list[dict[str, Any]],
    blockers: list[str],
    graph_summary: dict[str, Any],
    residual_graph_summary: dict[str, Any],
    selector: dict[str, Any],
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    gate_status = {str(row.get("gate")): bool(row.get("passed", False)) for row in gate_rows}
    rows = [
        {
            "gate": "independent_rank3_selector",
            "passed": gate_status.get("independent_rank3_selector", False),
            "missing_receipt": "observer-native target-rank-free rank-3 selector",
            "current_evidence": (
                f"selector_receipt={selector.get('NEUTRAL_INDEPENDENT_RANK3_SELECTOR_RECEIPT')}; "
                f"control_selector_count={selector.get('control_quotient_rank3_selector_count', 0)}; "
                f"control_candidate_count={selector.get('control_quotient_rank3_candidate_count', 0)}"
            ),
            "action_surface": "prime_geometric_rank_sweep_report/neutral_independent_rank_selector_audit_report",
            "blockers": [
                blocker for blocker in blockers
                if "rank3_selector" in blocker or "rank_selection" in blocker
            ],
        },
        {
            "gate": "overlap_native_graph_strict_h3",
            "passed": gate_status.get("overlap_native_graph_strict_h3", False),
            "missing_receipt": "strict-H3 overlap graph candidate with independent rank-3 selection",
            "current_evidence": (
                f"receipts={graph_summary.get('receipt_count', 0)}/{graph_summary.get('report_count', 0)}; "
                f"spatial={graph_summary.get('spatial_3d_candidate_count', 0)}; "
                f"strict_h3={graph_summary.get('strict_h3_candidate_count', 0)}; "
                f"rank3={graph_summary.get('rank3_selector_count', 0)}; "
                f"model_order_rank3={graph_summary.get('model_order_rank3_selector_count', 0)}; "
                "nontrivial_model_order_rank3="
                f"{graph_summary.get('nontrivial_model_order_rank3_selector_count', 0)}"
            ),
            "action_surface": "neutral-overlap-graph-sweep",
            "blockers": [
                blocker for blocker in blockers
                if blocker == "overlap_graph_strict_h3_candidate_false"
            ],
        },
        {
            "gate": "overlap_residualized_graph_geometry",
            "passed": gate_status.get("overlap_residualized_graph_geometry", False),
            "missing_receipt": "all residualized graph parameter cases complete",
            "current_evidence": (
                f"receipts={residual_graph_summary.get('receipt_count', 0)}/"
                f"{residual_graph_summary.get('report_count', 0)}"
            ),
            "action_surface": "neutral-overlap-residual-graph-sweep",
            "blockers": [
                blocker for blocker in blockers
                if blocker == "overlap_residual_graph_receipt_incomplete"
            ],
        },
        {
            "gate": "overlap_residualized_graph_strict_h3",
            "passed": gate_status.get("overlap_residualized_graph_strict_h3", False),
            "missing_receipt": "strict-H3 residualized overlap graph candidate with independent rank-3 selection",
            "current_evidence": (
                f"receipts={residual_graph_summary.get('receipt_count', 0)}/"
                f"{residual_graph_summary.get('report_count', 0)}; "
                f"spatial={residual_graph_summary.get('spatial_3d_candidate_count', 0)}; "
                f"strict_h3={residual_graph_summary.get('strict_h3_candidate_count', 0)}; "
                f"rank3={residual_graph_summary.get('rank3_selector_count', 0)}; "
                f"model_order_rank3={residual_graph_summary.get('model_order_rank3_selector_count', 0)}; "
                "nontrivial_model_order_rank3="
                f"{residual_graph_summary.get('nontrivial_model_order_rank3_selector_count', 0)}"
            ),
            "action_surface": "neutral-overlap-residual-graph-sweep",
            "blockers": [
                blocker for blocker in blockers
                if blocker == "overlap_residual_graph_strict_h3_candidate_false"
            ],
        },
        {
            "gate": "directional_h3_strict_gate",
            "passed": gate_status.get("directional_h3_strict_gate", False),
            "missing_receipt": "directional neutral row passing strict H3 model and leakage gates",
            "current_evidence": (
                f"directional_strict_ready_total={audit.get('directional_strict_ready_total', 0)}"
            ),
            "action_surface": "neutral-prime-rank-sweep/neutral-3d-bulk-audit",
            "blockers": [
                blocker for blocker in blockers
                if "directional" in blocker
            ],
        },
        {
            "gate": "strict_neutral_bulk_ready",
            "passed": gate_status.get("strict_neutral_bulk_ready", False),
            "missing_receipt": "all strict neutral promotion gates pass in one audit frontier",
            "current_evidence": f"strict_neutral_bulk_ready={audit.get('strict_neutral_bulk_ready', False)}",
            "action_surface": "strict-neutral-bulk-frontier",
            "blockers": list(blockers),
        },
    ]
    return [
        {
            **row,
            "blocking": bool(not row["passed"] or row["blockers"]),
            "claim_boundary": (
                "Strict-neutral-bulk hard-gate gap row. It identifies the missing neutral receipt "
                "required for promotion and does not promote theorem-assisted H3 previews."
            ),
        }
        for row in rows
        if (not row["passed"]) or row["blockers"]
    ]


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


def overlap_native_neutral_control_report(
    observer_views: list[dict[str, Any]],
    *,
    seed: int = 1,
    max_model_points: int = 256,
) -> dict[str, Any]:
    """Audit neutral geometry from the fundamental observer-overlap operation.

    The distance used here is built only from observer-visible record/object/
    transition packet overlap. It deliberately does not read screen axes,
    support coordinates, H3 coordinates, or fitted bulk positions. Passing this
    report is a negative-control receipt for the overlap substrate, not a strict
    neutral bulk proof by itself.
    """

    rng = np.random.default_rng(int(seed))
    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    if len(patch_views) < 8:
        return {
            "mode": "overlap_native_neutral_control_v0",
            "observer_count": len(patch_views),
            "sampled_observer_count": len(patch_views),
            "OVERLAP_NATIVE_NEGATIVE_CONTROL_RECEIPT": False,
            "overlap_native_negative_control_receipt": False,
            "overlap_native_spatial_3d_candidate": False,
            "strict_neutral_bulk": False,
            "physical_claim": False,
            "blockers": ["too_few_patch_observer_views"],
            "claim_boundary": (
                "Observer-overlap negative-control audit. It is necessary diagnostic evidence for "
                "a neutral bulk, but does not by itself promote strict neutral bulk."
            ),
        }

    sample_count = min(len(patch_views), max(8, int(max_model_points)))
    if len(patch_views) > sample_count:
        sample_indices = set(int(value) for value in rng.choice(len(patch_views), size=sample_count, replace=False))
        sampled = [view for index, view in enumerate(patch_views) if index in sample_indices]
    else:
        sampled = patch_views

    overlap_evidence = _overlap_evidence_report(sampled)
    presentation_invariance = _measured_overlap_presentation_invariance_report(sampled, rng)
    original_features = _overlap_feature_matrix(sampled)
    original_distance = _overlap_feature_distance_matrix(original_features)
    original_dimension = strict_neutral_dimension_report(original_distance)
    original_model = neutral_model_selection(
        original_distance,
        seed=int(seed),
        max_points=min(int(max_model_points), len(sampled)),
    )
    original_leakage = neutral_leakage_audit(original_distance, sampled)
    original_spatial_candidate = _spatial_3d_ready(original_dimension, original_model, original_leakage)
    original_strict_candidate = bool(
        original_dimension.get("estimators_agree_3d", False)
        and original_model.get("best_model") == "H3"
        and original_model.get("h3_beats_s2", False)
        and original_model.get("h3_beats_h2_h4", False)
        and original_leakage.get("s2_leakage_pass", False)
    )

    # Generate all stochastic control payloads in declared order on the caller
    # thread.  Their expensive metric-family refits are independent and can be
    # evaluated concurrently without sharing an RNG or mutable graph state.
    control_specs = [
        (
            "degree_preserving_overlap_graph_rewire",
            _overlap_feature_matrix(_shuffle_overlap_payloads(sampled, rng)),
            int(seed) + 11,
        ),
        (
            "overlap_edge_weight_permutation",
            _overlap_feature_matrix(_permute_overlap_packet_labels(sampled, rng)),
            int(seed) + 17,
        ),
        (
            "columnwise_histogram_null",
            _overlap_histogram_null_features(original_features, rng),
            int(seed) + 23,
        ),
    ]

    def evaluate_control(spec: tuple[str, np.ndarray, int]) -> dict[str, Any]:
        name, features, control_seed = spec
        return _overlap_native_control_row(
            name,
            original_distance,
            features,
            sampled,
            seed=control_seed,
            max_model_points=max_model_points,
            original_spatial_candidate=original_spatial_candidate,
        )

    control_worker_count = _sweep_worker_count(None, len(control_specs))
    if control_worker_count <= 1:
        control_rows = [evaluate_control(spec) for spec in control_specs]
    else:
        with ThreadPoolExecutor(max_workers=control_worker_count) as executor:
            control_rows = list(executor.map(evaluate_control, control_specs))
    all_controls_fail = bool(control_rows and all(row.get("expected_failure_observed", False) for row in control_rows))
    nondegenerate = bool(
        original_features.size
        and np.any(np.std(original_features, axis=0) > 1.0e-12)
        and np.any(original_distance[np.triu_indices(original_distance.shape[0], k=1)] > 1.0e-12)
    )
    blockers: list[str] = []
    if not overlap_evidence.get("available", False):
        blockers.extend(str(value) for value in overlap_evidence.get("blockers", []))
    if not presentation_invariance.get("receipt", False):
        blockers.extend(str(value) for value in presentation_invariance.get("blockers", []))
    if not nondegenerate:
        blockers.append("overlap_feature_matrix_degenerate")
    if not all_controls_fail:
        blockers.append("overlap_native_negative_controls_did_not_all_fail")
    if not original_spatial_candidate:
        blockers.append("overlap_native_distance_not_spatial_3d_candidate")
    if not original_strict_candidate:
        blockers.append("overlap_native_distance_not_strict_h3_candidate")

    receipt = bool(
        overlap_evidence.get("available", False)
        and presentation_invariance.get("receipt", False)
        and nondegenerate
        and all_controls_fail
    )
    return {
        "mode": "overlap_native_neutral_control_v0",
        "observer_count": len(patch_views),
        "sampled_observer_count": len(sampled),
        "seed": int(seed),
        "max_model_points": int(max_model_points),
        "overlap_evidence": overlap_evidence,
        "presentation_invariance_control": presentation_invariance,
        "fundamental_operation": (
            "Overlapping observations by observers: neutral distances are computed from shared "
            "observer-visible record/object/transition packet content before any H3 chart is assigned."
        ),
        "original": {
            "distance_matrix_shape": list(original_distance.shape),
            "dimension": original_dimension,
            "model_selection": original_model,
            "leakage": original_leakage,
            "spatial_3d_candidate": bool(original_spatial_candidate),
            "strict_h3_candidate": bool(original_strict_candidate),
        },
        "control_rows": control_rows,
        "parallel_execution": {
            "schema": "bounded_ordered_overlap_control_thread_execution_v1",
            "requested_environment": "OPH_FPE_GRAPH_SWEEP_WORKERS",
            "effective_n_jobs": int(control_worker_count),
            "control_task_count": len(control_specs),
            "ordered_result_assembly": True,
            "independent_named_rng_streams": True,
        },
        "all_expected_failures_observed": all_controls_fail,
        "overlap_feature_nondegenerate": nondegenerate,
        "OVERLAP_NATIVE_NEGATIVE_CONTROL_RECEIPT": receipt,
        "overlap_native_negative_control_receipt": receipt,
        "overlap_native_spatial_3d_candidate": bool(original_spatial_candidate),
        "overlap_native_strict_h3_candidate": bool(original_strict_candidate),
        "strict_neutral_bulk": False,
        "physical_claim": False,
        "blockers": blockers,
        "claim_boundary": (
            "This is the observer-overlap substrate audit. It can certify that the neutral overlap "
            "distance is nondegenerate and control-sensitive. Strict neutral bulk still additionally "
            "requires rank selection, H3/dimension/leakage gates, refinement across regulator sizes, "
            "and promotion by the neutral 3D audit."
        ),
    }


def write_overlap_native_neutral_control_report(
    run_dir: Path,
    out: Path | None = None,
    *,
    seed: int = 1,
    max_model_points: int = 256,
) -> dict[str, Any]:
    run = Path(run_dir)
    observer_path = run / "observer_views.jsonl"
    if not observer_path.exists():
        raise FileNotFoundError(observer_path)
    report = overlap_native_neutral_control_report(
        _read_jsonl(observer_path),
        seed=int(seed),
        max_model_points=int(max_model_points),
    )
    report["source_run_dir"] = str(run)
    report["observer_views_path"] = str(observer_path)
    destination = Path(out) if out is not None else run
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "overlap_native_neutral_control_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (destination / "overlap_native_neutral_control_report.md").write_text(
        _overlap_native_neutral_control_markdown(report),
        encoding="utf-8",
    )
    return report


def overlap_native_graph_geometry_report(
    observer_views: list[dict[str, Any]],
    *,
    seed: int = 1,
    max_model_points: int = 256,
    k_neighbors: int = 12,
) -> dict[str, Any]:
    """Audit graph geometry from the fundamental observer-overlap operation.

    This is stricter than displaying an H3 chart and different from the raw
    overlap-distance control. It builds a graph whose edge weights are shared
    observer-visible packet mass, then asks whether the induced shortest-path
    geometry passes the same neutral dimension/model/leakage checks.
    """

    rng = np.random.default_rng(int(seed))
    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    if len(patch_views) < 8:
        return {
            "mode": "overlap_native_graph_geometry_v0",
            "observer_count": len(patch_views),
            "sampled_observer_count": len(patch_views),
            "OVERLAP_NATIVE_GRAPH_GEOMETRY_RECEIPT": False,
            "overlap_graph_spatial_3d_candidate": False,
            "overlap_graph_strict_h3_candidate": False,
            "strict_neutral_bulk": False,
            "physical_claim": False,
            "blockers": ["too_few_patch_observer_views"],
            "claim_boundary": (
                "Observer-overlap graph geometry audit. It is necessary diagnostic evidence for a "
                "strict neutral bulk, but does not by itself promote strict neutral bulk."
            ),
        }

    sample_count = min(len(patch_views), max(8, int(max_model_points)))
    if len(patch_views) > sample_count:
        sample_indices = set(int(value) for value in rng.choice(len(patch_views), size=sample_count, replace=False))
        sampled = [view for index, view in enumerate(patch_views) if index in sample_indices]
    else:
        sampled = patch_views

    overlap_evidence = _overlap_evidence_report(sampled)
    presentation_invariance = _measured_overlap_presentation_invariance_report(sampled, rng)
    features = _overlap_feature_matrix(sampled)
    graph = _overlap_graph_distance_from_features(features, k_neighbors=int(k_neighbors))
    dimension = strict_neutral_dimension_report(graph["distance"])
    model = neutral_model_selection(
        graph["distance"],
        seed=int(seed),
        max_points=min(int(max_model_points), len(sampled)),
    )
    leakage = neutral_leakage_audit(graph["distance"], sampled)
    rank_selection = _overlap_graph_rank_selection(graph["affinity"])
    spatial_candidate = _spatial_3d_ready(dimension, model, leakage)
    strict_candidate = bool(
        spatial_candidate
        and model.get("best_model") == "H3"
        and model.get("h3_beats_s2", False)
        and model.get("h3_beats_h2_h4", False)
        and rank_selection.get("rank3_selector_receipt", False)
    )
    control_rows = [
        _overlap_graph_control_row(
            "degree_preserving_overlap_graph_rewire",
            graph["distance"],
            _overlap_feature_matrix(_shuffle_overlap_payloads(sampled, rng)),
            sampled,
            seed=int(seed) + 31,
            max_model_points=max_model_points,
            k_neighbors=k_neighbors,
            original_spatial_candidate=spatial_candidate,
        ),
        _overlap_graph_control_row(
            "overlap_edge_weight_permutation",
            graph["distance"],
            _overlap_feature_matrix(_permute_overlap_packet_labels(sampled, rng)),
            sampled,
            seed=int(seed) + 37,
            max_model_points=max_model_points,
            k_neighbors=k_neighbors,
            original_spatial_candidate=spatial_candidate,
        ),
        _overlap_graph_control_row(
            "columnwise_histogram_null",
            graph["distance"],
            _overlap_histogram_null_features(features, rng),
            sampled,
            seed=int(seed) + 43,
            max_model_points=max_model_points,
            k_neighbors=k_neighbors,
            original_spatial_candidate=spatial_candidate,
        ),
    ]
    controls_fail = bool(control_rows and all(row.get("expected_failure_observed", False) for row in control_rows))
    blockers: list[str] = []
    if not overlap_evidence.get("available", False):
        blockers.extend(str(value) for value in overlap_evidence.get("blockers", []))
    if not presentation_invariance.get("receipt", False):
        blockers.extend(str(value) for value in presentation_invariance.get("blockers", []))
    if not graph["nondegenerate"]:
        blockers.append("overlap_graph_degenerate_or_disconnected")
    if not controls_fail:
        blockers.append("overlap_graph_negative_controls_did_not_all_fail")
    if not rank_selection.get("rank3_selector_receipt", False):
        blockers.append("overlap_graph_independent_rank3_selector_false")
    if not spatial_candidate:
        blockers.append("overlap_graph_not_spatial_3d_candidate")
    if not strict_candidate:
        blockers.append("overlap_graph_not_strict_h3_candidate")

    receipt = bool(
        overlap_evidence.get("available", False)
        and presentation_invariance.get("receipt", False)
        and graph["nondegenerate"]
        and controls_fail
    )
    return {
        "mode": "overlap_native_graph_geometry_v0",
        "observer_count": len(patch_views),
        "sampled_observer_count": len(sampled),
        "seed": int(seed),
        "max_model_points": int(max_model_points),
        "k_neighbors": int(k_neighbors),
        "overlap_evidence": overlap_evidence,
        "presentation_invariance_control": presentation_invariance,
        "fundamental_operation": (
            "Overlapping observations by observers: graph edges are shared observer-visible "
            "record/object/transition packet mass before any H3 chart is assigned."
        ),
        "graph_summary": {
            "edge_count": graph["edge_count"],
            "component_count": graph["component_count"],
            "finite_pair_fraction": graph["finite_pair_fraction"],
            "mean_positive_affinity": graph["mean_positive_affinity"],
            "nondegenerate": graph["nondegenerate"],
        },
        "dimension": dimension,
        "model_selection": model,
        "leakage": leakage,
        "rank_selection": rank_selection,
        "control_rows": control_rows,
        "all_expected_failures_observed": controls_fail,
        "OVERLAP_NATIVE_GRAPH_GEOMETRY_RECEIPT": receipt,
        "overlap_graph_spatial_3d_candidate": bool(spatial_candidate),
        "overlap_graph_strict_h3_candidate": bool(strict_candidate),
        "strict_neutral_bulk": False,
        "physical_claim": False,
        "blockers": blockers,
        "claim_boundary": (
            "Observer-overlap graph geometry receipt. It can certify that the overlap graph is "
            "nondegenerate and control-sensitive. Strict neutral bulk still requires independent "
            "rank selection, H3/dimension/leakage gates, refinement across regulators, and promotion "
            "by the neutral 3D audit."
        ),
    }


def overlap_residualized_graph_geometry_report(
    observer_views: list[dict[str, Any]],
    *,
    seed: int = 1,
    max_model_points: int = 256,
    k_neighbors: int = 12,
    remove_modes: int = 1,
) -> dict[str, Any]:
    """Audit the observer-overlap graph after removing common overlap modes.

    The raw overlap graph is currently dominated by a rank-1 all-observer mode.
    This diagnostic removes only target-rank-free common modes from the
    observer-visible overlap-feature matrix, then reruns the same graph,
    dimension, H3, leakage, rank-selector, and negative-control gates. It is not
    a strict-neutral-bulk promotion receipt by itself.
    """

    rng = np.random.default_rng(int(seed))
    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    if len(patch_views) < 8:
        return {
            "mode": "overlap_residualized_graph_geometry_v0",
            "observer_count": len(patch_views),
            "sampled_observer_count": len(patch_views),
            "OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_RECEIPT": False,
            "overlap_residual_graph_spatial_3d_candidate": False,
            "overlap_residual_graph_strict_h3_candidate": False,
            "strict_neutral_bulk": False,
            "physical_claim": False,
            "blockers": ["too_few_patch_observer_views"],
            "claim_boundary": (
                "Residualized observer-overlap graph audit. It is diagnostic evidence about the "
                "rank-1 common-mode obstruction and does not by itself promote strict neutral bulk."
            ),
        }

    sample_count = min(len(patch_views), max(8, int(max_model_points)))
    if len(patch_views) > sample_count:
        sample_indices = set(int(value) for value in rng.choice(len(patch_views), size=sample_count, replace=False))
        sampled = [view for index, view in enumerate(patch_views) if index in sample_indices]
    else:
        sampled = patch_views

    overlap_evidence = _overlap_evidence_report(sampled)
    presentation_invariance = _measured_overlap_presentation_invariance_report(sampled, rng)
    features = _overlap_feature_matrix(sampled)
    residual = _residualize_overlap_features(features, remove_modes=int(remove_modes))
    raw_graph = _overlap_graph_distance_from_features(features, k_neighbors=int(k_neighbors))
    graph = _overlap_graph_distance_from_residual_features(residual, k_neighbors=int(k_neighbors))
    dimension = strict_neutral_dimension_report(graph["distance"])
    model = neutral_model_selection(
        graph["distance"],
        seed=int(seed),
        max_points=min(int(max_model_points), len(sampled)),
    )
    leakage = neutral_leakage_audit(graph["distance"], sampled)
    rank_selection = _overlap_graph_rank_selection(graph["affinity"])
    spatial_candidate = _spatial_3d_ready(dimension, model, leakage)
    strict_candidate = bool(
        spatial_candidate
        and model.get("best_model") == "H3"
        and model.get("h3_beats_s2", False)
        and model.get("h3_beats_h2_h4", False)
        and rank_selection.get("rank3_selector_receipt", False)
    )
    control_rows = [
        _overlap_residual_graph_control_row(
            "degree_preserving_overlap_graph_rewire",
            graph["distance"],
            _overlap_feature_matrix(_shuffle_overlap_payloads(sampled, rng)),
            sampled,
            seed=int(seed) + 53,
            max_model_points=max_model_points,
            k_neighbors=k_neighbors,
            remove_modes=remove_modes,
            original_spatial_candidate=spatial_candidate,
        ),
        _overlap_residual_graph_control_row(
            "overlap_edge_weight_permutation",
            graph["distance"],
            _overlap_feature_matrix(_permute_overlap_packet_labels(sampled, rng)),
            sampled,
            seed=int(seed) + 59,
            max_model_points=max_model_points,
            k_neighbors=k_neighbors,
            remove_modes=remove_modes,
            original_spatial_candidate=spatial_candidate,
        ),
        _overlap_residual_graph_control_row(
            "columnwise_histogram_null",
            graph["distance"],
            _overlap_histogram_null_features(features, rng),
            sampled,
            seed=int(seed) + 61,
            max_model_points=max_model_points,
            k_neighbors=k_neighbors,
            remove_modes=remove_modes,
            original_spatial_candidate=spatial_candidate,
        ),
    ]
    controls_fail = bool(control_rows and all(row.get("expected_failure_observed", False) for row in control_rows))
    blockers: list[str] = []
    if not overlap_evidence.get("available", False):
        blockers.extend(str(value) for value in overlap_evidence.get("blockers", []))
    if not presentation_invariance.get("receipt", False):
        blockers.extend(str(value) for value in presentation_invariance.get("blockers", []))
    if not graph["nondegenerate"]:
        blockers.append("overlap_residual_graph_degenerate_or_disconnected")
    if not controls_fail:
        blockers.append("overlap_residual_graph_negative_controls_did_not_all_fail")
    if not rank_selection.get("rank3_selector_receipt", False):
        blockers.append("overlap_residual_graph_independent_rank3_selector_false")
    if not spatial_candidate:
        blockers.append("overlap_residual_graph_not_spatial_3d_candidate")
    if not strict_candidate:
        blockers.append("overlap_residual_graph_not_strict_h3_candidate")

    receipt = bool(
        overlap_evidence.get("available", False)
        and presentation_invariance.get("receipt", False)
        and graph["nondegenerate"]
        and controls_fail
    )
    return {
        "mode": "overlap_residualized_graph_geometry_v0",
        "observer_count": len(patch_views),
        "sampled_observer_count": len(sampled),
        "seed": int(seed),
        "max_model_points": int(max_model_points),
        "k_neighbors": int(k_neighbors),
        "remove_modes": int(remove_modes),
        "overlap_evidence": overlap_evidence,
        "presentation_invariance_control": presentation_invariance,
        "fundamental_operation": (
            "Overlapping observations by observers: the graph is built from observer-visible "
            "record/object/transition packet overlap after target-rank-free common-mode removal."
        ),
        "residualization": {
            "method": "column_center_then_remove_leading_svd_modes",
            "remove_modes": int(remove_modes),
            "raw_largest_gap_rank": (raw_graph_rank := _overlap_graph_rank_selection(raw_graph["affinity"])).get(
                "largest_gap_rank"
            ),
            "raw_rank3_selector": bool(raw_graph_rank.get("rank3_selector_receipt", False)),
            "raw_rank3_cumulative_explained_variance": raw_graph_rank.get(
                "rank3_cumulative_explained_variance"
            ),
            "removed_common_mode_energy_fraction": _removed_mode_energy_fraction(features, int(remove_modes)),
            "claim_boundary": (
                "Residualization is target-rank-free but diagnostic. It tests whether the raw rank-1 "
                "common-overlap obstruction hides a lower-dimensional sector; it is not a promotion gate."
            ),
        },
        "graph_summary": {
            "edge_count": graph["edge_count"],
            "component_count": graph["component_count"],
            "finite_pair_fraction": graph["finite_pair_fraction"],
            "mean_positive_affinity": graph["mean_positive_affinity"],
            "nondegenerate": graph["nondegenerate"],
        },
        "dimension": dimension,
        "model_selection": model,
        "leakage": leakage,
        "rank_selection": rank_selection,
        "control_rows": control_rows,
        "all_expected_failures_observed": controls_fail,
        "OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_RECEIPT": receipt,
        "overlap_residual_graph_spatial_3d_candidate": bool(spatial_candidate),
        "overlap_residual_graph_strict_h3_candidate": bool(strict_candidate),
        "strict_neutral_bulk": False,
        "physical_claim": False,
        "blockers": blockers,
        "claim_boundary": (
            "Residualized observer-overlap graph geometry receipt. It can diagnose whether a common "
            "rank-1 observer-overlap mode is masking a 3D sector. Strict neutral bulk still requires "
            "raw/negative-control receipts, independent rank selection, H3/dimension/leakage gates, "
            "refinement across regulators, and promotion by the neutral 3D audit."
        ),
    }


def write_overlap_native_graph_geometry_report(
    run_dir: Path,
    out: Path | None = None,
    *,
    seed: int = 1,
    max_model_points: int = 256,
    k_neighbors: int = 12,
) -> dict[str, Any]:
    run = Path(run_dir)
    observer_path = run / "observer_views.jsonl"
    if not observer_path.exists():
        raise FileNotFoundError(observer_path)
    report = overlap_native_graph_geometry_report(
        _read_jsonl(observer_path),
        seed=int(seed),
        max_model_points=int(max_model_points),
        k_neighbors=int(k_neighbors),
    )
    report["source_run_dir"] = str(run)
    report["observer_views_path"] = str(observer_path)
    destination = Path(out) if out is not None else run
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "overlap_native_graph_geometry_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (destination / "overlap_native_graph_geometry_report.md").write_text(
        _overlap_native_graph_geometry_markdown(report),
        encoding="utf-8",
    )
    return report


def write_overlap_residualized_graph_geometry_report(
    run_dir: Path,
    out: Path | None = None,
    *,
    seed: int = 1,
    max_model_points: int = 256,
    k_neighbors: int = 12,
    remove_modes: int = 1,
) -> dict[str, Any]:
    run = Path(run_dir)
    observer_path = run / "observer_views.jsonl"
    if not observer_path.exists():
        raise FileNotFoundError(observer_path)
    report = overlap_residualized_graph_geometry_report(
        _read_jsonl(observer_path),
        seed=int(seed),
        max_model_points=int(max_model_points),
        k_neighbors=int(k_neighbors),
        remove_modes=int(remove_modes),
    )
    report["source_run_dir"] = str(run)
    report["observer_views_path"] = str(observer_path)
    destination = Path(out) if out is not None else run
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "overlap_residualized_graph_geometry_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (destination / "overlap_residualized_graph_geometry_report.md").write_text(
        _overlap_residualized_graph_geometry_markdown(report),
        encoding="utf-8",
    )
    return report


def overlap_native_graph_geometry_sweep_report(
    run_dirs: list[Path],
    *,
    seeds: tuple[int, ...] = (1,),
    max_model_points_values: tuple[int, ...] = (256,),
    k_neighbor_values: tuple[int, ...] = (12,),
    workers: int | None = None,
) -> dict[str, Any]:
    """Sweep observer-overlap graph geometry parameters over saved runs.

    A single overlap-graph receipt can be an unlucky parameter point. This
    sweep turns the strict-neutral frontier blocker into a reproducible search
    surface: every row is still a diagnostic receipt, and strict neutral bulk
    remains false unless a downstream frontier report promotes the required hard
    gates.
    """

    case_reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    roots = [Path(path) for path in run_dirs]
    cases: list[tuple[Path, Path, list[dict[str, Any]], int, int, int]] = []
    for run in roots:
        observer_path = run / "observer_views.jsonl"
        if not observer_path.exists():
            rows.append(
                {
                    "source_run_dir": str(run),
                    "error": f"missing {observer_path.name}",
                    "graph_geometry_receipt": False,
                    "spatial_3d_candidate": False,
                    "strict_h3_candidate": False,
                    "rank3_selector": False,
                }
            )
            continue
        observer_views = _read_jsonl(observer_path)
        for seed in seeds:
            for max_points in max_model_points_values:
                for k_neighbors in k_neighbor_values:
                    cases.append(
                        (
                            run,
                            observer_path,
                            observer_views,
                            int(seed),
                            int(max_points),
                            int(k_neighbors),
                        )
                    )

    def run_case(case: tuple[Path, Path, list[dict[str, Any]], int, int, int]) -> dict[str, Any]:
        run, observer_path, observer_views, seed, max_points, k_neighbors = case
        report = overlap_native_graph_geometry_report(
            observer_views,
            seed=seed,
            max_model_points=max_points,
            k_neighbors=k_neighbors,
        )
        report["source_run_dir"] = str(run)
        report["observer_views_path"] = str(observer_path)
        return report

    for report in _map_sweep_cases(run_case, cases, workers=workers):
        case_reports.append(report)
        rows.append(_overlap_graph_sweep_row(report))

    valid_rows = [row for row in rows if not row.get("error")]
    receipt_count = int(sum(1 for row in valid_rows if row.get("graph_geometry_receipt")))
    spatial_count = int(sum(1 for row in valid_rows if row.get("spatial_3d_candidate")))
    strict_count = int(sum(1 for row in valid_rows if row.get("strict_h3_candidate")))
    rank3_count = int(sum(1 for row in valid_rows if row.get("rank3_selector")))
    best = _best_overlap_graph_sweep_row(valid_rows)
    rank_obstruction = _overlap_graph_rank_obstruction_summary(valid_rows)
    gate_coincidence = _overlap_graph_gate_coincidence_summary(valid_rows)
    strict_candidates = [row for row in valid_rows if row.get("strict_h3_candidate")]
    closest_rows = _closest_overlap_graph_rows(valid_rows)
    blockers = []
    if valid_rows and strict_count <= 0:
        blockers.append("overlap_graph_sweep_no_strict_h3_candidate")
    if valid_rows and rank3_count <= 0:
        blockers.append("overlap_graph_sweep_no_independent_rank3_selector")
    if valid_rows and spatial_count <= 0:
        blockers.append("overlap_graph_sweep_no_spatial_3d_candidate")
    if not valid_rows:
        blockers.append("overlap_graph_sweep_no_valid_cases")
    return {
        "mode": "overlap_native_graph_geometry_sweep_v0",
        "run_dirs": [str(path) for path in roots],
        "seed_values": [int(value) for value in seeds],
        "max_model_points_values": [int(value) for value in max_model_points_values],
        "k_neighbor_values": [int(value) for value in k_neighbor_values],
        "case_count": len(valid_rows),
        "error_count": len(rows) - len(valid_rows),
        "run_count": len({str(row.get("source_run_dir")) for row in valid_rows}),
        "OVERLAP_NATIVE_GRAPH_GEOMETRY_SWEEP_RECEIPT": bool(valid_rows and receipt_count == len(valid_rows)),
        "graph_geometry_receipt_count": receipt_count,
        "spatial_3d_candidate_count": spatial_count,
        "strict_h3_candidate_count": strict_count,
        "rank3_selector_count": rank3_count,
        "best_case": best,
        "rank_obstruction_summary": rank_obstruction,
        "gate_coincidence_summary": gate_coincidence,
        "strict_candidate_rows": strict_candidates[:20],
        "closest_strict_rows": closest_rows,
        "rows": rows,
        "_case_reports": case_reports,
        "strict_neutral_bulk": False,
        "physical_claim": False,
        "blockers": blockers,
        "claim_boundary": (
            "Observer-overlap graph parameter sweep. It broadens the search for strict-H3/rank-3 "
            "neutral geometry, but each row remains diagnostic. It does not promote strict neutral "
            "bulk unless the independent-rank, H3/leakage, refinement, and frontier gates all pass."
        ),
    }


def write_overlap_native_graph_geometry_sweep_report(
    run_dirs: list[Path],
    out: Path,
    *,
    seeds: tuple[int, ...] = (1,),
    max_model_points_values: tuple[int, ...] = (256,),
    k_neighbor_values: tuple[int, ...] = (12,),
    workers: int | None = None,
) -> dict[str, Any]:
    report = overlap_native_graph_geometry_sweep_report(
        run_dirs,
        seeds=seeds,
        max_model_points_values=max_model_points_values,
        k_neighbor_values=k_neighbor_values,
        workers=workers,
    )
    case_reports = list(report.pop("_case_reports", []))
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    cases_dir = destination / "overlap_graph_cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(case_reports):
        case_dir = cases_dir / _overlap_graph_case_dir_name(case, index)
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "overlap_native_graph_geometry_report.json").write_text(
            json.dumps(case, indent=2, default=str),
            encoding="utf-8",
        )
        (case_dir / "overlap_native_graph_geometry_report.md").write_text(
            _overlap_native_graph_geometry_markdown(case),
            encoding="utf-8",
        )
    (destination / "overlap_native_graph_geometry_sweep_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (destination / "overlap_native_graph_geometry_sweep_report.md").write_text(
        _overlap_native_graph_geometry_sweep_markdown(report),
        encoding="utf-8",
    )
    _write_overlap_graph_sweep_rows_csv(
        destination / "overlap_native_graph_geometry_sweep_rows.csv",
        report.get("rows") if isinstance(report.get("rows"), list) else [],
    )
    return report


def overlap_residualized_graph_geometry_sweep_report(
    run_dirs: list[Path],
    *,
    seeds: tuple[int, ...] = (1,),
    max_model_points_values: tuple[int, ...] = (256,),
    k_neighbor_values: tuple[int, ...] = (12,),
    remove_mode_values: tuple[int, ...] = (1,),
    workers: int | None = None,
) -> dict[str, Any]:
    """Sweep residualized observer-overlap graph geometry parameters."""

    case_reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    roots = [Path(path) for path in run_dirs]
    cases: list[tuple[Path, Path, list[dict[str, Any]], int, int, int, int]] = []
    for run in roots:
        observer_path = run / "observer_views.jsonl"
        if not observer_path.exists():
            rows.append(
                {
                    "source_run_dir": str(run),
                    "error": f"missing {observer_path.name}",
                    "residual_graph_receipt": False,
                    "spatial_3d_candidate": False,
                    "strict_h3_candidate": False,
                    "rank3_selector": False,
                }
            )
            continue
        observer_views = _read_jsonl(observer_path)
        for seed in seeds:
            for max_points in max_model_points_values:
                for k_neighbors in k_neighbor_values:
                    for remove_modes in remove_mode_values:
                        cases.append(
                            (
                                run,
                                observer_path,
                                observer_views,
                                int(seed),
                                int(max_points),
                                int(k_neighbors),
                                int(remove_modes),
                            )
                        )

    def run_case(case: tuple[Path, Path, list[dict[str, Any]], int, int, int, int]) -> dict[str, Any]:
        run, observer_path, observer_views, seed, max_points, k_neighbors, remove_modes = case
        report = overlap_residualized_graph_geometry_report(
            observer_views,
            seed=seed,
            max_model_points=max_points,
            k_neighbors=k_neighbors,
            remove_modes=remove_modes,
        )
        report["source_run_dir"] = str(run)
        report["observer_views_path"] = str(observer_path)
        return report

    for report in _map_sweep_cases(run_case, cases, workers=workers):
        case_reports.append(report)
        rows.append(_overlap_residual_graph_sweep_row(report))

    valid_rows = [row for row in rows if not row.get("error")]
    receipt_count = int(sum(1 for row in valid_rows if row.get("residual_graph_receipt")))
    spatial_count = int(sum(1 for row in valid_rows if row.get("spatial_3d_candidate")))
    strict_count = int(sum(1 for row in valid_rows if row.get("strict_h3_candidate")))
    rank3_count = int(sum(1 for row in valid_rows if row.get("rank3_selector")))
    best = _best_overlap_residual_graph_sweep_row(valid_rows)
    rank_obstruction = _overlap_residual_graph_rank_obstruction_summary(valid_rows)
    gate_coincidence = _overlap_graph_gate_coincidence_summary(valid_rows)
    strict_candidates = [row for row in valid_rows if row.get("strict_h3_candidate")]
    closest_rows = _closest_overlap_graph_rows(valid_rows)
    blockers = []
    if valid_rows and strict_count <= 0:
        blockers.append("overlap_residual_graph_sweep_no_strict_h3_candidate")
    if valid_rows and rank3_count <= 0:
        blockers.append("overlap_residual_graph_sweep_no_independent_rank3_selector")
    if valid_rows and spatial_count <= 0:
        blockers.append("overlap_residual_graph_sweep_no_spatial_3d_candidate")
    if not valid_rows:
        blockers.append("overlap_residual_graph_sweep_no_valid_cases")
    return {
        "mode": "overlap_residualized_graph_geometry_sweep_v0",
        "run_dirs": [str(path) for path in roots],
        "seed_values": [int(value) for value in seeds],
        "max_model_points_values": [int(value) for value in max_model_points_values],
        "k_neighbor_values": [int(value) for value in k_neighbor_values],
        "remove_mode_values": [int(value) for value in remove_mode_values],
        "case_count": len(valid_rows),
        "error_count": len(rows) - len(valid_rows),
        "run_count": len({str(row.get("source_run_dir")) for row in valid_rows}),
        "OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_SWEEP_RECEIPT": bool(
            valid_rows and receipt_count == len(valid_rows)
        ),
        "residual_graph_receipt_count": receipt_count,
        "spatial_3d_candidate_count": spatial_count,
        "strict_h3_candidate_count": strict_count,
        "rank3_selector_count": rank3_count,
        "best_case": best,
        "rank_obstruction_summary": rank_obstruction,
        "gate_coincidence_summary": gate_coincidence,
        "strict_candidate_rows": strict_candidates[:20],
        "closest_strict_rows": closest_rows,
        "rows": rows,
        "_case_reports": case_reports,
        "strict_neutral_bulk": False,
        "physical_claim": False,
        "blockers": blockers,
        "claim_boundary": (
            "Residualized observer-overlap graph parameter sweep. It tests whether target-rank-free "
            "common-mode removal exposes a stable 3D/H3 sector, but each row remains diagnostic. "
            "It does not promote strict neutral bulk unless independent-rank, H3/leakage, refinement, "
            "and frontier gates all pass."
        ),
    }


def write_overlap_residualized_graph_geometry_sweep_report(
    run_dirs: list[Path],
    out: Path,
    *,
    seeds: tuple[int, ...] = (1,),
    max_model_points_values: tuple[int, ...] = (256,),
    k_neighbor_values: tuple[int, ...] = (12,),
    remove_mode_values: tuple[int, ...] = (1,),
    workers: int | None = None,
) -> dict[str, Any]:
    report = overlap_residualized_graph_geometry_sweep_report(
        run_dirs,
        seeds=seeds,
        max_model_points_values=max_model_points_values,
        k_neighbor_values=k_neighbor_values,
        remove_mode_values=remove_mode_values,
        workers=workers,
    )
    case_reports = list(report.pop("_case_reports", []))
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    cases_dir = destination / "overlap_residual_graph_cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(case_reports):
        case_dir = cases_dir / _overlap_graph_case_dir_name(case, index)
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "overlap_residualized_graph_geometry_report.json").write_text(
            json.dumps(case, indent=2, default=str),
            encoding="utf-8",
        )
        (case_dir / "overlap_residualized_graph_geometry_report.md").write_text(
            _overlap_residualized_graph_geometry_markdown(case),
            encoding="utf-8",
        )
    (destination / "overlap_residualized_graph_geometry_sweep_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (destination / "overlap_residualized_graph_geometry_sweep_report.md").write_text(
        _overlap_residualized_graph_geometry_sweep_markdown(report),
        encoding="utf-8",
    )
    _write_overlap_residual_graph_sweep_rows_csv(
        destination / "overlap_residualized_graph_geometry_sweep_rows.csv",
        report.get("rows") if isinstance(report.get("rows"), list) else [],
    )
    return report


def _map_sweep_cases(fn: Any, cases: list[Any], *, workers: int | None) -> list[dict[str, Any]]:
    if not cases:
        return []
    worker_count = _sweep_worker_count(workers, len(cases))
    if worker_count <= 1:
        return [fn(case) for case in cases]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(fn, cases))


def _sweep_worker_count(requested: int | None, case_count: int) -> int:
    raw = requested
    if raw is None:
        env = os.environ.get("OPH_FPE_GRAPH_SWEEP_WORKERS")
        raw = int(env) if env and env.strip() else 1
    try:
        count = int(raw)
    except (TypeError, ValueError):
        count = 1
    return max(1, min(count, max(1, int(case_count))))


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


def _find_prime_rank_refinement_reports(paths: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in paths:
        path = Path(path)
        if path.is_file() and path.name == "prime_geometric_rank_refinement_report.json":
            found.add(path)
            continue
        direct = path / "prime_geometric_rank_refinement_report.json"
        if direct.exists():
            found.add(direct)
        if path.exists() and path.is_dir():
            found.update(path.glob("**/prime_geometric_rank_refinement_report.json"))
    return sorted(found, key=lambda value: str(value))


def _find_overlap_native_control_reports(paths: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in paths:
        path = Path(path)
        if path.is_file() and path.name == "overlap_native_neutral_control_report.json":
            found.add(path)
            continue
        direct = path / "overlap_native_neutral_control_report.json"
        if direct.exists():
            found.add(direct)
        if path.exists() and path.is_dir():
            found.update(path.glob("**/overlap_native_neutral_control_report.json"))
    return sorted(found, key=lambda value: str(value))


def _find_overlap_native_graph_geometry_reports(paths: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in paths:
        path = Path(path)
        if path.is_file() and path.name == "overlap_native_graph_geometry_report.json":
            found.add(path)
            continue
        direct = path / "overlap_native_graph_geometry_report.json"
        if direct.exists():
            found.add(direct)
        if path.exists() and path.is_dir():
            found.update(path.glob("**/overlap_native_graph_geometry_report.json"))
    return sorted(found, key=lambda value: str(value))


def _find_overlap_residualized_graph_geometry_reports(paths: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in paths:
        path = Path(path)
        if path.is_file() and path.name == "overlap_residualized_graph_geometry_report.json":
            found.add(path)
            continue
        direct = path / "overlap_residualized_graph_geometry_report.json"
        if direct.exists():
            found.add(direct)
        if path.exists() and path.is_dir():
            found.update(path.glob("**/overlap_residualized_graph_geometry_report.json"))
    return sorted(found, key=lambda value: str(value))


def _find_neutral_3d_bulk_audit_reports(paths: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in paths:
        path = Path(path)
        if path.is_file() and path.name == "neutral_3d_bulk_audit_report.json":
            found.add(path)
            continue
        direct = path / "neutral_3d_bulk_audit_report.json"
        if direct.exists():
            found.add(direct)
        if path.exists() and path.is_dir():
            found.update(path.glob("**/neutral_3d_bulk_audit_report.json"))
    return sorted(found, key=lambda value: str(value))


def _find_neutral_independent_rank_selector_reports(paths: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in paths:
        path = Path(path)
        if path.is_file() and path.name == "neutral_independent_rank_selector_audit_report.json":
            found.add(path)
            continue
        direct = path / "neutral_independent_rank_selector_audit_report.json"
        if direct.exists():
            found.add(direct)
        if path.exists() and path.is_dir():
            found.update(path.glob("**/neutral_independent_rank_selector_audit_report.json"))
    return sorted(found, key=lambda value: str(value))


def _select_neutral_refinement_report(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        reports,
        key=lambda report: (
            bool(report.get("strict_neutral_bulk_refinement_receipt", False)),
            bool(report.get("control_quotient_rank3_refinement_candidate_receipt", False)),
            int(report.get("run_count") or 0),
            -len(report.get("proof_blockers") or []),
        ),
        reverse=True,
    )[0] if reports else {}


def _select_neutral_3d_audit_report(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        reports,
        key=lambda report: (
            bool(report.get("strict_neutral_bulk_ready", False)),
            bool(report.get("control_residualized_rank3_refinement_candidate", False)),
            int(report.get("overlap_native_negative_control_receipt_count") or 0),
            int(report.get("sweep_report_count") or 0),
            -len(report.get("blockers") or []),
        ),
        reverse=True,
    )[0] if reports else {}


def _select_neutral_selector_report(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        reports,
        key=lambda report: (
            bool(report.get("NEUTRAL_INDEPENDENT_RANK3_SELECTOR_RECEIPT", False)),
            int(report.get("control_quotient_rank3_selector_count") or 0),
            -len(report.get("blockers") or []),
            int(report.get("run_count") or 0),
        ),
        reverse=True,
    )[0] if reports else {}


def _neutral_3d_bulk_audit_blockers(
    refinement: dict[str, Any],
    sweeps: list[dict[str, Any]],
    *,
    overlap_control_reports: list[dict[str, Any]] | None = None,
) -> list[str]:
    blockers: list[str] = []
    if not sweeps:
        blockers.append("missing_prime_geometric_rank_sweep_reports")
    if not refinement:
        blockers.append("missing_prime_geometric_rank_refinement_report")
    blockers.extend(str(blocker) for blocker in refinement.get("proof_blockers", []))
    refinement_candidate = bool(refinement.get("control_quotient_rank3_refinement_candidate_receipt", False))
    for report in sweeps:
        for blocker in report.get("proof_blockers", []):
            if refinement_candidate and blocker == "requires_refinement_stability_across_regulator_sizes":
                continue
            blockers.append(str(blocker))
    if refinement and not bool(refinement.get("strict_neutral_bulk_refinement_receipt", False)):
        blockers.append("strict_neutral_bulk_refinement_receipt_false")
    if refinement and not bool(refinement.get("independent_rank3_selector_all", False)):
        blockers.append("independent_svd_rank3_selector_not_stable_or_false")
    if any(
        not bool(((report.get("regulator_control_quotient_lane") or {}).get("is_negative_control", False)))
        for report in sweeps
    ):
        blockers.append("control_quotient_lane_is_not_a_negative_control")
    overlap_reports = overlap_control_reports or []
    if overlap_reports and not all(
        bool(report.get("OVERLAP_NATIVE_NEGATIVE_CONTROL_RECEIPT", False)) for report in overlap_reports
    ):
        blockers.append("overlap_native_negative_control_receipt_false")
    return _unique_preserve_order(blockers)


def _overlap_native_control_audit_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    report_count = len(reports)
    receipt_count = int(sum(1 for report in reports if report.get("OVERLAP_NATIVE_NEGATIVE_CONTROL_RECEIPT")))
    spatial_count = int(sum(1 for report in reports if report.get("overlap_native_spatial_3d_candidate")))
    strict_count = int(sum(1 for report in reports if report.get("overlap_native_strict_h3_candidate")))
    return {
        "report_count": report_count,
        "receipt_count": receipt_count,
        "receipt_all": bool(report_count and receipt_count == report_count),
        "spatial_3d_candidate_count": spatial_count,
        "strict_h3_candidate_count": strict_count,
        "rows": [
            {
                "source_run_dir": report.get("source_run_dir"),
                "observer_count": report.get("observer_count"),
                "sampled_observer_count": report.get("sampled_observer_count"),
                "negative_control_receipt": bool(report.get("OVERLAP_NATIVE_NEGATIVE_CONTROL_RECEIPT", False)),
                "spatial_3d_candidate": bool(report.get("overlap_native_spatial_3d_candidate", False)),
                "strict_h3_candidate": bool(report.get("overlap_native_strict_h3_candidate", False)),
                "blockers": report.get("blockers", []),
            }
            for report in reports
        ],
    }


def _overlap_graph_geometry_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    report_count = len(reports)
    receipt_count = int(sum(1 for report in reports if report.get("OVERLAP_NATIVE_GRAPH_GEOMETRY_RECEIPT")))
    spatial_count = int(sum(1 for report in reports if report.get("overlap_graph_spatial_3d_candidate")))
    strict_count = int(sum(1 for report in reports if report.get("overlap_graph_strict_h3_candidate")))
    rank3_count = int(
        sum(1 for report in reports if (report.get("rank_selection") or {}).get("rank3_selector_receipt"))
    )
    model_order_rank3_count = int(
        sum(
            1
            for report in reports
            if (report.get("rank_selection") or {}).get("model_order_rank3_selector_receipt")
        )
    )
    nontrivial_model_order_rank3_count = int(
        sum(
            1
            for report in reports
            if (report.get("rank_selection") or {}).get(
                "nontrivial_model_order_rank3_selector_receipt"
            )
        )
    )
    return {
        "report_count": report_count,
        "receipt_count": receipt_count,
        "receipt_all": bool(report_count and receipt_count == report_count),
        "spatial_3d_candidate_count": spatial_count,
        "strict_h3_candidate_count": strict_count,
        "rank3_selector_count": rank3_count,
        "model_order_rank3_selector_count": model_order_rank3_count,
        "nontrivial_model_order_rank3_selector_count": nontrivial_model_order_rank3_count,
        "rows": [
            {
                "source_run_dir": report.get("source_run_dir"),
                "observer_count": report.get("observer_count"),
                "sampled_observer_count": report.get("sampled_observer_count"),
                "graph_geometry_receipt": bool(report.get("OVERLAP_NATIVE_GRAPH_GEOMETRY_RECEIPT", False)),
                "spatial_3d_candidate": bool(report.get("overlap_graph_spatial_3d_candidate", False)),
                "strict_h3_candidate": bool(report.get("overlap_graph_strict_h3_candidate", False)),
                "rank3_selector": bool((report.get("rank_selection") or {}).get("rank3_selector_receipt", False)),
                "model_order_rank3_selector": bool(
                    (report.get("rank_selection") or {}).get("model_order_rank3_selector_receipt", False)
                ),
                "nontrivial_model_order_rank3_selector": bool(
                    (report.get("rank_selection") or {}).get(
                        "nontrivial_model_order_rank3_selector_receipt",
                        False,
                    )
                ),
                "model_order_consensus_rank": (
                    ((report.get("rank_selection") or {}).get("model_order_selection") or {}).get(
                        "consensus_rank"
                    )
                ),
                "nontrivial_model_order_consensus_rank": (
                    ((report.get("rank_selection") or {}).get("nontrivial_model_order_selection") or {}).get(
                        "consensus_rank"
                    )
                ),
                "median_dimension": (report.get("dimension") or {}).get("median_dimension_estimate"),
                "selected_model": (report.get("model_selection") or {}).get("best_model"),
                "edge_count": (report.get("graph_summary") or {}).get("edge_count"),
                "component_count": (report.get("graph_summary") or {}).get("component_count"),
                "blockers": report.get("blockers", []),
            }
            for report in reports
        ],
    }


def _overlap_residual_graph_geometry_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    report_count = len(reports)
    receipt_count = int(
        sum(1 for report in reports if report.get("OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_RECEIPT"))
    )
    spatial_count = int(
        sum(1 for report in reports if report.get("overlap_residual_graph_spatial_3d_candidate"))
    )
    strict_count = int(
        sum(1 for report in reports if report.get("overlap_residual_graph_strict_h3_candidate"))
    )
    rank3_count = int(
        sum(1 for report in reports if (report.get("rank_selection") or {}).get("rank3_selector_receipt"))
    )
    model_order_rank3_count = int(
        sum(
            1
            for report in reports
            if (report.get("rank_selection") or {}).get("model_order_rank3_selector_receipt")
        )
    )
    nontrivial_model_order_rank3_count = int(
        sum(
            1
            for report in reports
            if (report.get("rank_selection") or {}).get(
                "nontrivial_model_order_rank3_selector_receipt"
            )
        )
    )
    return {
        "report_count": report_count,
        "receipt_count": receipt_count,
        "receipt_all": bool(report_count and receipt_count == report_count),
        "spatial_3d_candidate_count": spatial_count,
        "strict_h3_candidate_count": strict_count,
        "rank3_selector_count": rank3_count,
        "model_order_rank3_selector_count": model_order_rank3_count,
        "nontrivial_model_order_rank3_selector_count": nontrivial_model_order_rank3_count,
        "rows": [
            {
                "source_run_dir": report.get("source_run_dir"),
                "observer_count": report.get("observer_count"),
                "sampled_observer_count": report.get("sampled_observer_count"),
                "residual_graph_receipt": bool(
                    report.get("OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_RECEIPT", False)
                ),
                "spatial_3d_candidate": bool(
                    report.get("overlap_residual_graph_spatial_3d_candidate", False)
                ),
                "strict_h3_candidate": bool(
                    report.get("overlap_residual_graph_strict_h3_candidate", False)
                ),
                "rank3_selector": bool((report.get("rank_selection") or {}).get("rank3_selector_receipt", False)),
                "model_order_rank3_selector": bool(
                    (report.get("rank_selection") or {}).get("model_order_rank3_selector_receipt", False)
                ),
                "nontrivial_model_order_rank3_selector": bool(
                    (report.get("rank_selection") or {}).get(
                        "nontrivial_model_order_rank3_selector_receipt",
                        False,
                    )
                ),
                "model_order_consensus_rank": (
                    ((report.get("rank_selection") or {}).get("model_order_selection") or {}).get(
                        "consensus_rank"
                    )
                ),
                "nontrivial_model_order_consensus_rank": (
                    ((report.get("rank_selection") or {}).get("nontrivial_model_order_selection") or {}).get(
                        "consensus_rank"
                    )
                ),
                "raw_largest_gap_rank": (report.get("residualization") or {}).get("raw_largest_gap_rank"),
                "largest_gap_rank": (report.get("rank_selection") or {}).get("largest_gap_rank"),
                "median_dimension": (report.get("dimension") or {}).get("median_dimension_estimate"),
                "selected_model": (report.get("model_selection") or {}).get("best_model"),
                "edge_count": (report.get("graph_summary") or {}).get("edge_count"),
                "component_count": (report.get("graph_summary") or {}).get("component_count"),
                "blockers": report.get("blockers", []),
            }
            for report in reports
        ],
    }


def _overlap_graph_sweep_row(report: dict[str, Any]) -> dict[str, Any]:
    graph = report.get("graph_summary") if isinstance(report.get("graph_summary"), dict) else {}
    dimension = report.get("dimension") if isinstance(report.get("dimension"), dict) else {}
    model = report.get("model_selection") if isinstance(report.get("model_selection"), dict) else {}
    leakage = report.get("leakage") if isinstance(report.get("leakage"), dict) else {}
    rank = report.get("rank_selection") if isinstance(report.get("rank_selection"), dict) else {}
    return {
        "source_run_dir": report.get("source_run_dir"),
        "seed": report.get("seed"),
        "max_model_points": report.get("max_model_points"),
        "k_neighbors": report.get("k_neighbors"),
        "observer_count": report.get("observer_count"),
        "sampled_observer_count": report.get("sampled_observer_count"),
        "graph_geometry_receipt": bool(report.get("OVERLAP_NATIVE_GRAPH_GEOMETRY_RECEIPT", False)),
        "spatial_3d_candidate": bool(report.get("overlap_graph_spatial_3d_candidate", False)),
        "strict_h3_candidate": bool(report.get("overlap_graph_strict_h3_candidate", False)),
        "rank3_selector": bool(rank.get("rank3_selector_receipt", False)),
        "median_dimension": _to_float_or_none(dimension.get("median_dimension_estimate")),
        "correlation_dimension": _to_float_or_none(
            ((dimension.get("correlation_dimension") or {}).get("estimate"))
        ),
        "local_mle_dimension": _to_float_or_none(
            ((dimension.get("local_mle_dimension") or {}).get("median_estimate"))
        ),
        "selected_model": model.get("best_model"),
        "h3_beats_s2": bool(model.get("h3_beats_s2", False)),
        "h3_beats_h2_h4": bool(model.get("h3_beats_h2_h4", False)),
        "s2_leakage_pass": bool(leakage.get("s2_leakage_pass", False)),
        "s2_distance_correlation": _to_float_or_none(leakage.get("s2_distance_correlation")),
        "largest_gap_rank": rank.get("largest_gap_rank"),
        "model_order_consensus_rank": (rank.get("model_order_selection") or {}).get("consensus_rank"),
        "model_order_profile_rank": (rank.get("model_order_selection") or {}).get("profile_likelihood_rank"),
        "model_order_broken_stick_rank": (rank.get("model_order_selection") or {}).get("broken_stick_rank"),
        "model_order_rank3_selector": bool(rank.get("model_order_rank3_selector_receipt", False)),
        "rank3_cumulative_explained_variance": _to_float_or_none(
            rank.get("rank3_cumulative_explained_variance")
        ),
        "effective_rank": _to_float_or_none(rank.get("effective_rank")),
        "nontrivial_rank3_selector": bool(rank.get("nontrivial_rank3_selector_receipt", False)),
        "nontrivial_largest_gap_rank": rank.get("nontrivial_largest_gap_rank"),
        "nontrivial_model_order_consensus_rank": (
            rank.get("nontrivial_model_order_selection") or {}
        ).get("consensus_rank"),
        "nontrivial_model_order_profile_rank": (
            rank.get("nontrivial_model_order_selection") or {}
        ).get("profile_likelihood_rank"),
        "nontrivial_model_order_broken_stick_rank": (
            rank.get("nontrivial_model_order_selection") or {}
        ).get("broken_stick_rank"),
        "nontrivial_model_order_rank3_selector": bool(
            rank.get("nontrivial_model_order_rank3_selector_receipt", False)
        ),
        "nontrivial_rank3_cumulative_explained_variance": _to_float_or_none(
            rank.get("nontrivial_rank3_cumulative_explained_variance")
        ),
        "nontrivial_effective_rank": _to_float_or_none(rank.get("nontrivial_effective_rank")),
        "edge_count": graph.get("edge_count"),
        "component_count": graph.get("component_count"),
        "finite_pair_fraction": _to_float_or_none(graph.get("finite_pair_fraction")),
        "mean_positive_affinity": _to_float_or_none(graph.get("mean_positive_affinity")),
        "blockers": report.get("blockers", []),
    }


def _overlap_residual_graph_sweep_row(report: dict[str, Any]) -> dict[str, Any]:
    graph = report.get("graph_summary") if isinstance(report.get("graph_summary"), dict) else {}
    dimension = report.get("dimension") if isinstance(report.get("dimension"), dict) else {}
    model = report.get("model_selection") if isinstance(report.get("model_selection"), dict) else {}
    leakage = report.get("leakage") if isinstance(report.get("leakage"), dict) else {}
    rank = report.get("rank_selection") if isinstance(report.get("rank_selection"), dict) else {}
    residualization = report.get("residualization") if isinstance(report.get("residualization"), dict) else {}
    receipt = bool(report.get("OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_RECEIPT", False))
    return {
        "source_run_dir": report.get("source_run_dir"),
        "seed": report.get("seed"),
        "max_model_points": report.get("max_model_points"),
        "k_neighbors": report.get("k_neighbors"),
        "remove_modes": report.get("remove_modes"),
        "observer_count": report.get("observer_count"),
        "sampled_observer_count": report.get("sampled_observer_count"),
        "residual_graph_receipt": receipt,
        "graph_geometry_receipt": receipt,
        "spatial_3d_candidate": bool(report.get("overlap_residual_graph_spatial_3d_candidate", False)),
        "strict_h3_candidate": bool(report.get("overlap_residual_graph_strict_h3_candidate", False)),
        "rank3_selector": bool(rank.get("rank3_selector_receipt", False)),
        "median_dimension": _to_float_or_none(dimension.get("median_dimension_estimate")),
        "correlation_dimension": _to_float_or_none(
            ((dimension.get("correlation_dimension") or {}).get("estimate"))
        ),
        "local_mle_dimension": _to_float_or_none(
            ((dimension.get("local_mle_dimension") or {}).get("median_estimate"))
        ),
        "selected_model": model.get("best_model"),
        "h3_beats_s2": bool(model.get("h3_beats_s2", False)),
        "h3_beats_h2_h4": bool(model.get("h3_beats_h2_h4", False)),
        "s2_leakage_pass": bool(leakage.get("s2_leakage_pass", False)),
        "s2_distance_correlation": _to_float_or_none(leakage.get("s2_distance_correlation")),
        "largest_gap_rank": rank.get("largest_gap_rank"),
        "model_order_consensus_rank": (rank.get("model_order_selection") or {}).get("consensus_rank"),
        "model_order_profile_rank": (rank.get("model_order_selection") or {}).get("profile_likelihood_rank"),
        "model_order_broken_stick_rank": (rank.get("model_order_selection") or {}).get("broken_stick_rank"),
        "model_order_rank3_selector": bool(rank.get("model_order_rank3_selector_receipt", False)),
        "rank3_cumulative_explained_variance": _to_float_or_none(
            rank.get("rank3_cumulative_explained_variance")
        ),
        "effective_rank": _to_float_or_none(rank.get("effective_rank")),
        "nontrivial_rank3_selector": bool(rank.get("nontrivial_rank3_selector_receipt", False)),
        "nontrivial_largest_gap_rank": rank.get("nontrivial_largest_gap_rank"),
        "nontrivial_model_order_consensus_rank": (
            rank.get("nontrivial_model_order_selection") or {}
        ).get("consensus_rank"),
        "nontrivial_model_order_profile_rank": (
            rank.get("nontrivial_model_order_selection") or {}
        ).get("profile_likelihood_rank"),
        "nontrivial_model_order_broken_stick_rank": (
            rank.get("nontrivial_model_order_selection") or {}
        ).get("broken_stick_rank"),
        "nontrivial_model_order_rank3_selector": bool(
            rank.get("nontrivial_model_order_rank3_selector_receipt", False)
        ),
        "nontrivial_rank3_cumulative_explained_variance": _to_float_or_none(
            rank.get("nontrivial_rank3_cumulative_explained_variance")
        ),
        "nontrivial_effective_rank": _to_float_or_none(rank.get("nontrivial_effective_rank")),
        "raw_largest_gap_rank": residualization.get("raw_largest_gap_rank"),
        "raw_rank3_selector": bool(residualization.get("raw_rank3_selector", False)),
        "removed_common_mode_energy_fraction": _to_float_or_none(
            residualization.get("removed_common_mode_energy_fraction")
        ),
        "edge_count": graph.get("edge_count"),
        "component_count": graph.get("component_count"),
        "finite_pair_fraction": _to_float_or_none(graph.get("finite_pair_fraction")),
        "mean_positive_affinity": _to_float_or_none(graph.get("mean_positive_affinity")),
        "blockers": report.get("blockers", []),
    }


def _best_overlap_graph_sweep_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def score(row: dict[str, Any]) -> tuple[float, ...]:
        median = _to_float_or_none(row.get("median_dimension"))
        dimension_error = abs(float(median) - 3.0) if median is not None else 1.0e9
        return (
            float(bool(row.get("strict_h3_candidate", False))),
            float(bool(row.get("rank3_selector", False))),
            float(bool(row.get("spatial_3d_candidate", False))),
            float(bool(row.get("graph_geometry_receipt", False))),
            float(row.get("selected_model") == "H3"),
            float(bool(row.get("h3_beats_s2", False))),
            float(bool(row.get("h3_beats_h2_h4", False))),
            -dimension_error,
            -float(len(row.get("blockers") or [])),
        )

    return sorted(rows, key=score, reverse=True)[0]


def _best_overlap_residual_graph_sweep_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _best_overlap_graph_sweep_row(rows)


def _closest_overlap_graph_rows(rows: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    """Return compact nearest witnesses for strict neutral graph gates.

    These rows are diagnostic: they expose which hard gates are nearest to passing
    without changing the strict-H3 or independent-rank receipts.
    """

    if not rows:
        return []

    def num(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        value = _to_float_or_none(row.get(key))
        return float(value) if value is not None else default

    def dimension_error(row: dict[str, Any]) -> float:
        value = _to_float_or_none(row.get("median_dimension"))
        return abs(float(value) - 3.0) if value is not None else 1.0e9

    def score(row: dict[str, Any]) -> tuple[float, ...]:
        gates = _overlap_graph_gate_status(row)
        passed_core = sum(
            1
            for key in (
                "graph_receipt",
                "spatial_3d_candidate",
                "h3_model",
                "h3_beats_s2",
                "h3_beats_h2_h4",
                "s2_leakage_pass",
                "independent_rank3_selector",
            )
            if gates.get(key)
        )
        return (
            float(bool(gates.get("strict_h3_candidate"))),
            float(passed_core),
            float(bool(gates.get("independent_rank3_selector"))),
            float(bool(gates.get("nontrivial_rank3_selector_diagnostic"))),
            -dimension_error(row),
            num(row, "rank3_cumulative_explained_variance"),
            num(row, "nontrivial_rank3_cumulative_explained_variance"),
            -num(row, "effective_rank", default=1.0e9),
            -num(row, "nontrivial_effective_rank", default=1.0e9),
            -float(len(row.get("blockers") or [])),
        )

    out: list[dict[str, Any]] = []
    for row in sorted(rows, key=score, reverse=True)[: max(0, int(limit))]:
        gates = _overlap_graph_gate_status(row)
        missing = [key for key, passed in gates.items() if not passed and key != "nontrivial_rank3_selector_diagnostic"]
        out.append(
            {
                "source_run_dir": row.get("source_run_dir"),
                "seed": row.get("seed"),
                "max_model_points": row.get("max_model_points"),
                "k_neighbors": row.get("k_neighbors"),
                "remove_modes": row.get("remove_modes"),
                "gate_score": int(sum(1 for passed in gates.values() if passed)),
                "gate_status": gates,
                "missing_strict_gates": missing,
                "graph_geometry_receipt": bool(row.get("graph_geometry_receipt", False)),
                "residual_graph_receipt": bool(row.get("residual_graph_receipt", False)),
                "spatial_3d_candidate": bool(row.get("spatial_3d_candidate", False)),
                "strict_h3_candidate": bool(row.get("strict_h3_candidate", False)),
                "median_dimension": _to_float_or_none(row.get("median_dimension")),
                "selected_model": row.get("selected_model"),
                "rank3_selector": bool(row.get("rank3_selector", False)),
                "rank3_cumulative_explained_variance": _to_float_or_none(
                    row.get("rank3_cumulative_explained_variance")
                ),
                "effective_rank": _to_float_or_none(row.get("effective_rank")),
                "nontrivial_rank3_selector": bool(row.get("nontrivial_rank3_selector", False)),
                "nontrivial_rank3_cumulative_explained_variance": _to_float_or_none(
                    row.get("nontrivial_rank3_cumulative_explained_variance")
                ),
                "nontrivial_effective_rank": _to_float_or_none(row.get("nontrivial_effective_rank")),
                "blockers": row.get("blockers") or [],
            }
        )
    return out


def _overlap_graph_gate_coincidence_summary(
    rows: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> dict[str, Any]:
    """Summarize whether geometric and spectral neutral gates coincide."""

    if not rows:
        return {
            "available": False,
            "case_count": 0,
            "claim_boundary": "No overlap-graph rows were available for gate-coincidence diagnostics.",
        }

    def is_h3_geometry(row: dict[str, Any]) -> bool:
        return (
            bool(row.get("spatial_3d_candidate", False))
            and row.get("selected_model") == "H3"
            and bool(row.get("h3_beats_s2", False))
            and bool(row.get("h3_beats_h2_h4", False))
            and bool(row.get("s2_leakage_pass", False))
        )

    def dimension_error(row: dict[str, Any]) -> float:
        value = _to_float_or_none(row.get("median_dimension"))
        return abs(float(value) - 3.0) if value is not None else 1.0e9

    def num(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        value = _to_float_or_none(row.get(key))
        return float(value) if value is not None else default

    h3_geometry_rows = [row for row in rows if is_h3_geometry(row)]
    independent_rows = [row for row in rows if row.get("rank3_selector")]
    nontrivial_rows = [row for row in rows if row.get("nontrivial_rank3_selector")]
    h3_independent_rows = [row for row in h3_geometry_rows if row.get("rank3_selector")]
    h3_nontrivial_rows = [row for row in h3_geometry_rows if row.get("nontrivial_rank3_selector")]
    strict_rows = [row for row in rows if row.get("strict_h3_candidate")]

    def h3_score(row: dict[str, Any]) -> tuple[float, ...]:
        return (
            float(bool(row.get("strict_h3_candidate", False))),
            float(bool(row.get("rank3_selector", False))),
            float(bool(row.get("nontrivial_rank3_selector", False))),
            -dimension_error(row),
            num(row, "rank3_cumulative_explained_variance"),
            num(row, "nontrivial_rank3_cumulative_explained_variance"),
            -num(row, "effective_rank", default=1.0e9),
        )

    def spectral_score(row: dict[str, Any]) -> tuple[float, ...]:
        return (
            float(is_h3_geometry(row)),
            -dimension_error(row),
            num(row, "nontrivial_rank3_cumulative_explained_variance"),
            num(row, "rank3_cumulative_explained_variance"),
            -num(row, "nontrivial_effective_rank", default=1.0e9),
        )

    return {
        "available": True,
        "case_count": len(rows),
        "spatial_h3_geometry_count": len(h3_geometry_rows),
        "independent_rank3_selector_count": len(independent_rows),
        "nontrivial_rank3_selector_count": len(nontrivial_rows),
        "spatial_h3_independent_rank3_selector_count": len(h3_independent_rows),
        "spatial_h3_nontrivial_rank3_selector_count": len(h3_nontrivial_rows),
        "strict_h3_candidate_count": len(strict_rows),
        "coincidence_gap": bool(not strict_rows and h3_geometry_rows and (independent_rows or nontrivial_rows)),
        "best_spatial_h3_rows": [
            _compact_overlap_gate_witness_row(row)
            for row in sorted(h3_geometry_rows, key=h3_score, reverse=True)[:limit]
        ],
        "best_nontrivial_rank3_rows": [
            _compact_overlap_gate_witness_row(row)
            for row in sorted(nontrivial_rows, key=spectral_score, reverse=True)[:limit]
        ],
        "coincidence_rows": [
            _compact_overlap_gate_witness_row(row)
            for row in sorted(h3_nontrivial_rows + h3_independent_rows, key=h3_score, reverse=True)[:limit]
        ],
        "claim_boundary": (
            "Gate-coincidence diagnostic only. Strict neutral bulk still requires spatial-H3 geometry "
            "and an independent rank-3 selector to pass in the same neutral witness."
        ),
    }


def _compact_overlap_gate_witness_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_run_dir": row.get("source_run_dir"),
        "seed": row.get("seed"),
        "max_model_points": row.get("max_model_points"),
        "k_neighbors": row.get("k_neighbors"),
        "remove_modes": row.get("remove_modes"),
        "spatial_3d_candidate": bool(row.get("spatial_3d_candidate", False)),
        "strict_h3_candidate": bool(row.get("strict_h3_candidate", False)),
        "rank3_selector": bool(row.get("rank3_selector", False)),
        "nontrivial_rank3_selector": bool(row.get("nontrivial_rank3_selector", False)),
        "selected_model": row.get("selected_model"),
        "median_dimension": _to_float_or_none(row.get("median_dimension")),
        "h3_beats_s2": bool(row.get("h3_beats_s2", False)),
        "h3_beats_h2_h4": bool(row.get("h3_beats_h2_h4", False)),
        "s2_leakage_pass": bool(row.get("s2_leakage_pass", False)),
        "largest_gap_rank": row.get("largest_gap_rank"),
        "nontrivial_largest_gap_rank": row.get("nontrivial_largest_gap_rank"),
        "rank3_cumulative_explained_variance": _to_float_or_none(
            row.get("rank3_cumulative_explained_variance")
        ),
        "nontrivial_rank3_cumulative_explained_variance": _to_float_or_none(
            row.get("nontrivial_rank3_cumulative_explained_variance")
        ),
        "effective_rank": _to_float_or_none(row.get("effective_rank")),
        "nontrivial_effective_rank": _to_float_or_none(row.get("nontrivial_effective_rank")),
        "blockers": row.get("blockers") or [],
    }


def _overlap_graph_gate_status(row: dict[str, Any]) -> dict[str, bool]:
    receipt = bool(row.get("graph_geometry_receipt", False) or row.get("residual_graph_receipt", False))
    return {
        "graph_receipt": receipt,
        "spatial_3d_candidate": bool(row.get("spatial_3d_candidate", False)),
        "h3_model": row.get("selected_model") == "H3",
        "h3_beats_s2": bool(row.get("h3_beats_s2", False)),
        "h3_beats_h2_h4": bool(row.get("h3_beats_h2_h4", False)),
        "s2_leakage_pass": bool(row.get("s2_leakage_pass", False)),
        "independent_rank3_selector": bool(row.get("rank3_selector", False)),
        "strict_h3_candidate": bool(row.get("strict_h3_candidate", False)),
        "nontrivial_rank3_selector_diagnostic": bool(row.get("nontrivial_rank3_selector", False)),
    }


def _overlap_graph_rank_obstruction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "available": False,
            "claim_boundary": (
                "No valid overlap-graph rows were available. This summary is diagnostic only."
            ),
        }

    def finite_values(key: str, source_rows: list[dict[str, Any]]) -> list[float]:
        values: list[float] = []
        for row in source_rows:
            value = _to_float_or_none(row.get(key))
            if value is not None:
                values.append(float(value))
        return values

    def median(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[midpoint])
        return float((ordered[midpoint - 1] + ordered[midpoint]) * 0.5)

    def row_key(row: dict[str, Any], key: str) -> float:
        value = _to_float_or_none(row.get(key))
        return float(value) if value is not None else -1.0e100

    spatial_rows = [row for row in rows if row.get("spatial_3d_candidate")]
    h3_rows = [row for row in rows if row.get("selected_model") == "H3"]
    rank_rows = [row for row in rows if row.get("largest_gap_rank") is not None]
    largest_gap_counts: dict[str, int] = {}
    for row in rank_rows:
        key = str(row.get("largest_gap_rank"))
        largest_gap_counts[key] = int(largest_gap_counts.get(key, 0) + 1)
    model_order_counts: dict[str, int] = {}
    nontrivial_model_order_counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("model_order_consensus_rank"))
        if key not in {"None", ""}:
            model_order_counts[key] = int(model_order_counts.get(key, 0) + 1)
        nontrivial_key = str(row.get("nontrivial_model_order_consensus_rank"))
        if nontrivial_key not in {"None", ""}:
            nontrivial_model_order_counts[nontrivial_key] = int(
                nontrivial_model_order_counts.get(nontrivial_key, 0) + 1
            )
    best_rank3_ev = max(rows, key=lambda row: row_key(row, "rank3_cumulative_explained_variance"))
    best_spatial_rank3_ev = (
        max(spatial_rows, key=lambda row: row_key(row, "rank3_cumulative_explained_variance"))
        if spatial_rows
        else {}
    )
    effective_values = finite_values("effective_rank", rows)
    rank3_ev_values = finite_values("rank3_cumulative_explained_variance", rows)
    nontrivial_rank3_ev_values = finite_values(
        "nontrivial_rank3_cumulative_explained_variance",
        rows,
    )
    nontrivial_effective_values = finite_values("nontrivial_effective_rank", rows)
    spatial_effective = finite_values("effective_rank", spatial_rows)
    spatial_rank3_ev = finite_values("rank3_cumulative_explained_variance", spatial_rows)
    spatial_nontrivial_rank3_ev = finite_values(
        "nontrivial_rank3_cumulative_explained_variance",
        spatial_rows,
    )
    nontrivial_largest_gap_counts: dict[str, int] = {}
    for row in rank_rows:
        key = str(row.get("nontrivial_largest_gap_rank"))
        if key not in {"None", ""}:
            nontrivial_largest_gap_counts[key] = int(nontrivial_largest_gap_counts.get(key, 0) + 1)
    return {
        "available": True,
        "case_count": len(rows),
        "spatial_3d_candidate_count": len(spatial_rows),
        "h3_model_count": len(h3_rows),
        "rank3_selector_count": int(sum(1 for row in rows if row.get("rank3_selector"))),
        "model_order_rank3_selector_count": int(
            sum(1 for row in rows if row.get("model_order_rank3_selector"))
        ),
        "nontrivial_rank3_selector_count": int(
            sum(1 for row in rows if row.get("nontrivial_rank3_selector"))
        ),
        "nontrivial_model_order_rank3_selector_count": int(
            sum(1 for row in rows if row.get("nontrivial_model_order_rank3_selector"))
        ),
        "largest_gap_rank_counts": largest_gap_counts,
        "dominant_largest_gap_rank": (
            max(largest_gap_counts.items(), key=lambda item: item[1])[0] if largest_gap_counts else None
        ),
        "model_order_consensus_rank_counts": model_order_counts,
        "dominant_model_order_consensus_rank": (
            max(model_order_counts.items(), key=lambda item: item[1])[0] if model_order_counts else None
        ),
        "nontrivial_largest_gap_rank_counts": nontrivial_largest_gap_counts,
        "dominant_nontrivial_largest_gap_rank": (
            max(nontrivial_largest_gap_counts.items(), key=lambda item: item[1])[0]
            if nontrivial_largest_gap_counts
            else None
        ),
        "nontrivial_model_order_consensus_rank_counts": nontrivial_model_order_counts,
        "dominant_nontrivial_model_order_consensus_rank": (
            max(nontrivial_model_order_counts.items(), key=lambda item: item[1])[0]
            if nontrivial_model_order_counts
            else None
        ),
        "max_rank3_cumulative_explained_variance": max(rank3_ev_values) if rank3_ev_values else None,
        "median_rank3_cumulative_explained_variance": median(rank3_ev_values),
        "max_nontrivial_rank3_cumulative_explained_variance": (
            max(nontrivial_rank3_ev_values) if nontrivial_rank3_ev_values else None
        ),
        "median_nontrivial_rank3_cumulative_explained_variance": median(nontrivial_rank3_ev_values),
        "min_nontrivial_effective_rank": (
            min(nontrivial_effective_values) if nontrivial_effective_values else None
        ),
        "median_nontrivial_effective_rank": median(nontrivial_effective_values),
        "min_effective_rank": min(effective_values) if effective_values else None,
        "median_effective_rank": median(effective_values),
        "spatial_max_rank3_cumulative_explained_variance": max(spatial_rank3_ev) if spatial_rank3_ev else None,
        "spatial_median_rank3_cumulative_explained_variance": median(spatial_rank3_ev),
        "spatial_max_nontrivial_rank3_cumulative_explained_variance": (
            max(spatial_nontrivial_rank3_ev) if spatial_nontrivial_rank3_ev else None
        ),
        "spatial_median_nontrivial_rank3_cumulative_explained_variance": (
            median(spatial_nontrivial_rank3_ev)
        ),
        "spatial_min_effective_rank": min(spatial_effective) if spatial_effective else None,
        "spatial_median_effective_rank": median(spatial_effective),
        "best_rank3_ev_case": best_rank3_ev,
        "best_spatial_rank3_ev_case": best_spatial_rank3_ev,
        "primary_obstruction": (
            "no_independent_rank3_selector"
            if not any(row.get("rank3_selector") for row in rows)
            else "rank3_selector_present_but_other_gates_failed"
        ),
        "claim_boundary": (
            "Diagnostic obstruction summary for the target-rank-free overlap-graph spectrum. "
            "It explains why strict neutral bulk is not promoted; it is not itself a strict-bulk receipt."
        ),
    }


def _overlap_residual_graph_rank_obstruction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = dict(_overlap_graph_rank_obstruction_summary(rows))
    raw_rank1_count = int(sum(1 for row in rows if str(row.get("raw_largest_gap_rank")) == "1"))
    residual_rank3_count = int(sum(1 for row in rows if row.get("rank3_selector")))
    summary.update(
        {
            "raw_largest_gap_rank1_count": raw_rank1_count,
            "residual_rank3_selector_count": residual_rank3_count,
            "primary_obstruction": (
                "residualized_no_independent_rank3_selector"
                if residual_rank3_count <= 0
                else summary.get("primary_obstruction")
            ),
            "claim_boundary": (
                "Diagnostic obstruction summary for the residualized observer-overlap graph spectrum. "
                "It tests whether common-mode removal changes the rank selector; it is not itself a "
                "strict-bulk receipt."
            ),
        }
    )
    return summary


def _overlap_graph_case_dir_name(report: dict[str, Any], index: int) -> str:
    run_name = Path(str(report.get("source_run_dir") or f"run_{index:04d}")).name
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in run_name)
    return (
        f"{index:04d}_{safe}"
        f"_seed{int(report.get('seed') or 0)}"
        f"_n{int(report.get('max_model_points') or 0)}"
        f"_k{int(report.get('k_neighbors') or 0)}"
    )


def _write_overlap_graph_sweep_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "source_run_dir",
        "seed",
        "max_model_points",
        "k_neighbors",
        "observer_count",
        "sampled_observer_count",
        "graph_geometry_receipt",
        "spatial_3d_candidate",
        "strict_h3_candidate",
        "rank3_selector",
        "model_order_consensus_rank",
        "model_order_profile_rank",
        "model_order_broken_stick_rank",
        "model_order_rank3_selector",
        "median_dimension",
        "selected_model",
        "h3_beats_s2",
        "h3_beats_h2_h4",
        "s2_leakage_pass",
        "largest_gap_rank",
        "rank3_cumulative_explained_variance",
        "effective_rank",
        "nontrivial_rank3_selector",
        "nontrivial_largest_gap_rank",
        "nontrivial_model_order_consensus_rank",
        "nontrivial_model_order_profile_rank",
        "nontrivial_model_order_broken_stick_rank",
        "nontrivial_model_order_rank3_selector",
        "nontrivial_rank3_cumulative_explained_variance",
        "nontrivial_effective_rank",
        "edge_count",
        "component_count",
        "blockers",
        "error",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {column: row.get(column) for column in columns}
            blockers = out.get("blockers")
            if isinstance(blockers, list):
                out["blockers"] = ";".join(str(value) for value in blockers)
            writer.writerow(out)


def _write_overlap_residual_graph_sweep_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "source_run_dir",
        "seed",
        "max_model_points",
        "k_neighbors",
        "remove_modes",
        "observer_count",
        "sampled_observer_count",
        "residual_graph_receipt",
        "spatial_3d_candidate",
        "strict_h3_candidate",
        "rank3_selector",
        "model_order_consensus_rank",
        "model_order_profile_rank",
        "model_order_broken_stick_rank",
        "model_order_rank3_selector",
        "median_dimension",
        "correlation_dimension",
        "local_mle_dimension",
        "selected_model",
        "h3_beats_s2",
        "h3_beats_h2_h4",
        "s2_leakage_pass",
        "s2_distance_correlation",
        "largest_gap_rank",
        "rank3_cumulative_explained_variance",
        "effective_rank",
        "nontrivial_rank3_selector",
        "nontrivial_largest_gap_rank",
        "nontrivial_model_order_consensus_rank",
        "nontrivial_model_order_profile_rank",
        "nontrivial_model_order_broken_stick_rank",
        "nontrivial_model_order_rank3_selector",
        "nontrivial_rank3_cumulative_explained_variance",
        "nontrivial_effective_rank",
        "raw_largest_gap_rank",
        "raw_rank3_selector",
        "removed_common_mode_energy_fraction",
        "edge_count",
        "component_count",
        "finite_pair_fraction",
        "mean_positive_affinity",
        "blockers",
        "error",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {column: row.get(column) for column in columns}
            blockers = out.get("blockers")
            if isinstance(blockers, list):
                out["blockers"] = ";".join(str(value) for value in blockers)
            writer.writerow(out)


def _neutral_3d_refinement_summary(refinement: dict[str, Any]) -> dict[str, Any]:
    if not refinement:
        return {"present": False}
    return {
        "present": True,
        "run_count": refinement.get("run_count"),
        "multi_scale": bool(refinement.get("multi_scale", False)),
        "control_quotient_rank3_refinement_candidate_receipt": bool(
            refinement.get("control_quotient_rank3_refinement_candidate_receipt", False)
        ),
        "independent_rank3_selector_all": bool(refinement.get("independent_rank3_selector_all", False)),
        "candidate_dimension_drift": refinement.get("candidate_dimension_drift"),
        "candidate_dimension_stable": bool(refinement.get("candidate_dimension_stable", False)),
        "strict_neutral_bulk_refinement_receipt": bool(
            refinement.get("strict_neutral_bulk_refinement_receipt", False)
        ),
        "proof_blockers": refinement.get("proof_blockers", []),
        "sizes": refinement.get("sizes", []),
    }


def _neutral_3d_sweep_summary(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    candidate = report.get("control_quotient_coordinate_best_3d_dimension_row") or {}
    dimension = candidate.get("dimension") if isinstance(candidate.get("dimension"), dict) else {}
    model = candidate.get("model_selection") if isinstance(candidate.get("model_selection"), dict) else {}
    leakage = candidate.get("leakage") if isinstance(candidate.get("leakage"), dict) else {}
    independent = report.get("independent_rank_selection") if isinstance(
        report.get("independent_rank_selection"), dict
    ) else {}
    return {
        "path": str(path),
        "source_run_dir": report.get("source_run_dir"),
        "observer_count": report.get("observer_count"),
        "sampled_observer_count": report.get("sampled_observer_count"),
        "strict_3d_ready_count": int(report.get("strict_3d_ready_count") or 0),
        "coordinate_spatial_3d_ready_count": int(report.get("coordinate_spatial_3d_ready_count") or 0),
        "control_quotient_coordinate_spatial_3d_ready_count": int(
            report.get("control_quotient_coordinate_spatial_3d_ready_count") or 0
        ),
        "control_quotient_rank3_candidate": bool(
            report.get("control_residualized_rank3_candidate_receipt", False)
        ),
        "control_quotient_candidate": {
            "rank": candidate.get("rank"),
            "best_model": model.get("best_model"),
            "median_dimension": dimension.get("median_dimension_estimate"),
            "s2_distance_correlation": leakage.get("s2_distance_correlation"),
            "s2_leakage_pass": bool(leakage.get("s2_leakage_pass", False)),
            "h3_beats_s2": bool(model.get("h3_beats_s2", False)),
            "h3_beats_e3": bool(model.get("h3_beats_e3", False)),
            "h3_beats_h2_h4": bool(model.get("h3_beats_h2_h4", False)),
        },
        "independent_rank_selection": {
            "prime_geometric": independent.get("prime_geometric", {}),
            "control_quotient": independent.get("control_quotient", {}),
            "prime_rank3_selector_receipt": bool(independent.get("prime_rank3_selector_receipt", False)),
            "control_quotient_rank3_selector_receipt": bool(
                independent.get("control_quotient_rank3_selector_receipt", False)
            ),
        },
        "proof_blockers": report.get("proof_blockers", []),
    }


def _neutral_3d_bulk_audit_markdown(report: dict[str, Any]) -> str:
    blockers = report.get("blockers") or []
    lines = [
        "# Neutral 3D Bulk Audit",
        "",
        f"- strict neutral bulk ready: `{str(report.get('strict_neutral_bulk_ready', False)).lower()}`",
        f"- control-residualized rank-3 refinement candidate: `{str(report.get('control_residualized_rank3_refinement_candidate', False)).lower()}`",
        f"- independent rank-3 selector all: `{str(report.get('control_residualized_rank3_independent_selector_all', False)).lower()}`",
        f"- directional strict-ready total: `{report.get('directional_strict_ready_total', 0)}`",
        f"- control-quotient candidate count: `{report.get('control_quotient_candidate_count', 0)}`",
        f"- overlap-native negative-control receipts: `{report.get('overlap_native_negative_control_receipt_count', 0)}` / `{report.get('overlap_native_negative_control_report_count', 0)}`",
        f"- overlap-native spatial-3D candidates: `{report.get('overlap_native_spatial_3d_candidate_count', 0)}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def _neutral_rank_selector_row(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    independent = report.get("independent_rank_selection") if isinstance(
        report.get("independent_rank_selection"), dict
    ) else {}
    prime = independent.get("prime_geometric") if isinstance(independent.get("prime_geometric"), dict) else {}
    control = independent.get("control_quotient") if isinstance(independent.get("control_quotient"), dict) else {}
    control_lane = report.get("regulator_control_quotient_lane")
    if not isinstance(control_lane, dict):
        control_lane = {}
    return {
        "path": str(path),
        "source_run_dir": report.get("source_run_dir"),
        "observer_count": report.get("observer_count"),
        "sampled_observer_count": report.get("sampled_observer_count"),
        "prime_rank3_selector_receipt": bool(
            independent.get("prime_rank3_selector_receipt", False)
            or (
                report.get("independent_rank3_selector_receipt", False)
                and prime.get("independent_rank3_selector_receipt", False)
            )
        ),
        "control_quotient_rank3_selector_receipt": bool(
            independent.get("control_quotient_rank3_selector_receipt", False)
        ),
        "control_quotient_rank3_candidate": bool(
            report.get("control_residualized_rank3_candidate_receipt", False)
            or report.get("prime_geometric_control_quotient_spatial_3d_candidate_receipt", False)
        ),
        "control_quotient_lane_is_negative_control": bool(control_lane.get("is_negative_control", False)),
        "prime_geometric_largest_gap_rank": prime.get("largest_gap_rank"),
        "prime_geometric_chord_elbow_rank": prime.get("chord_elbow_rank"),
        "prime_geometric_effective_rank": _to_float_or_none(prime.get("effective_rank")),
        "prime_geometric_participation_rank": _to_float_or_none(prime.get("participation_rank")),
        "prime_geometric_rank3_cumulative_explained_variance": _to_float_or_none(
            prime.get("rank3_cumulative_explained_variance")
        ),
        "control_quotient_largest_gap_rank": control.get("largest_gap_rank"),
        "control_quotient_chord_elbow_rank": control.get("chord_elbow_rank"),
        "control_quotient_effective_rank": _to_float_or_none(control.get("effective_rank")),
        "control_quotient_participation_rank": _to_float_or_none(control.get("participation_rank")),
        "control_quotient_rank3_cumulative_explained_variance": _to_float_or_none(
            control.get("rank3_cumulative_explained_variance")
        ),
        "proof_blockers": report.get("proof_blockers", []),
    }


def _neutral_independent_rank_selector_markdown(report: dict[str, Any]) -> str:
    blockers = report.get("blockers") or []
    lines = [
        "# Neutral Independent Rank Selector Audit",
        "",
        f"- independent rank-3 selector receipt: `{str(report.get('NEUTRAL_INDEPENDENT_RANK3_SELECTOR_RECEIPT', False)).lower()}`",
        f"- run count: `{report.get('run_count', 0)}`",
        f"- prime-geometric rank-3 selector count: `{report.get('prime_geometric_rank3_selector_count', 0)}`",
        f"- control-quotient rank-3 selector count: `{report.get('control_quotient_rank3_selector_count', 0)}`",
        f"- control-quotient rank-3 candidate count: `{report.get('control_quotient_rank3_candidate_count', 0)}`",
        f"- control-quotient median effective rank: `{report.get('control_quotient_median_effective_rank')}`",
        f"- control-quotient median rank-3 explained variance: `{report.get('control_quotient_median_rank3_cumulative_explained_variance')}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def _strict_neutral_bulk_frontier_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Strict Neutral Bulk Frontier",
        "",
        f"- strict neutral bulk ready: `{str(report.get('strict_neutral_bulk_ready', False)).lower()}`",
        f"- strict neutral bulk: `{str(report.get('strict_neutral_bulk', False)).lower()}`",
        f"- rank-3 refinement candidate: `{str(report.get('control_residualized_rank3_refinement_candidate', False)).lower()}`",
        f"- overlap-native negative controls: `{str(report.get('overlap_native_negative_control_receipt_all', False)).lower()}`",
        f"- overlap graph receipts: `{report.get('overlap_native_graph_geometry_receipt_count', 0)}` / `{report.get('overlap_native_graph_geometry_report_count', 0)}`",
        f"- overlap graph spatial-3D candidates: `{report.get('overlap_native_graph_spatial_3d_candidate_count', 0)}`",
        f"- overlap graph strict-H3 candidates: `{report.get('overlap_native_graph_strict_h3_candidate_count', 0)}`",
        f"- overlap graph model-order rank-3 selectors: `{report.get('overlap_native_graph_model_order_rank3_selector_count', 0)}`",
        f"- overlap graph nontrivial model-order rank-3 selectors: `{report.get('overlap_native_graph_nontrivial_model_order_rank3_selector_count', 0)}`",
        f"- residualized graph receipts: `{report.get('overlap_residualized_graph_geometry_receipt_count', 0)}` / `{report.get('overlap_residualized_graph_geometry_report_count', 0)}`",
        f"- residualized graph spatial-3D candidates: `{report.get('overlap_residualized_graph_spatial_3d_candidate_count', 0)}`",
        f"- residualized graph strict-H3 candidates: `{report.get('overlap_residualized_graph_strict_h3_candidate_count', 0)}`",
        f"- residualized graph model-order rank-3 selectors: `{report.get('overlap_residualized_graph_model_order_rank3_selector_count', 0)}`",
        f"- residualized graph nontrivial model-order rank-3 selectors: `{report.get('overlap_residualized_graph_nontrivial_model_order_rank3_selector_count', 0)}`",
        f"- independent rank-3 selector: `{str(report.get('neutral_independent_rank3_selector_receipt', False)).lower()}`",
        "",
        "## Gates",
        "",
    ]
    for row in report.get("gate_rows") or []:
        lines.append(
            f"- `{row.get('gate')}`: `{str(row.get('passed', False)).lower()}`"
            f" - {row.get('detail', '')}"
        )
    lines.extend(["", "## Hard-Gate Gaps", ""])
    gap_rows = report.get("gate_gap_rows") or []
    if gap_rows:
        for row in gap_rows:
            blockers = ", ".join(str(blocker) for blocker in (row.get("blockers") or [])) or "none"
            lines.append(
                f"- `{row.get('gate')}`: missing `{row.get('missing_receipt')}`; "
                f"current `{row.get('current_evidence')}`; action `{row.get('action_surface')}`; "
                f"blockers `{blockers}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Blockers", ""])
    blockers = report.get("blockers") or []
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Next Missing Receipts", ""])
    next_steps = report.get("next_missing_receipts") or []
    lines.extend(
        f"- `{row.get('blocker')}`: {row.get('next_step')}"
        for row in next_steps
    ) if next_steps else lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def _strict_neutral_frontier_next_steps(blockers: list[str]) -> list[dict[str, str]]:
    suggestions = {
        "independent_svd_rank3_selector_not_stable_or_false": (
            "Find an observer-native, target-rank-free selector that independently chooses rank 3 across regulators."
        ),
        "control_quotient_lane_is_not_a_negative_control": (
            "Replace or supplement the control quotient with null controls that test the same rank-3 candidate path."
        ),
        "directional_h3_strict_rank_gate_not_passed": (
            "Close the directional H3 model-selection/leakage gate for a non-coordinate, neutral distance lane."
        ),
        "no_directional_rank_passes_strict_h3_model_and_leakage_gates": (
            "Produce at least one directional neutral rank row with strict H3 model selection and leakage pass."
        ),
        "requires_independent_rank_selection_rule_before_physical_interpretation": (
            "Keep rank-3 windows diagnostic until the independent SVD/rank selector receipt passes."
        ),
        "strict_neutral_bulk_refinement_receipt_false": (
            "Rerun refinement only after the independent-rank, negative-control, and directional-H3 gates close."
        ),
        "control_quotient_independent_rank3_selector_not_all_true": (
            "Audit why the control-quotient singular spectrum remains high-dimensional despite rank-3 distance windows."
        ),
        "control_quotient_rank3_cumulative_explained_variance_low": (
            "Do not treat low rank-3 explained variance as a strict 3D source; investigate the high-rank substrate."
        ),
        "control_quotient_effective_rank_not_low_dimensional": (
            "Resolve the gap between low-dimensional distance behavior and high effective spectral rank."
        ),
        "overlap_graph_strict_h3_candidate_false": (
            "Continue overlap-native graph sweeps: current graph geometry is control-sensitive, but has "
            "not produced a strict H3 candidate with independent rank-3 selection."
        ),
    }
    rows: list[dict[str, str]] = []
    for blocker in blockers:
        rows.append(
            {
                "blocker": blocker,
                "next_step": suggestions.get(blocker, "Clear this blocker with an independent neutral receipt."),
            }
        )
    return rows


def _unique_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _prime_rank_refinement_row(report: dict[str, Any]) -> dict[str, Any]:
    candidate = report.get("control_quotient_coordinate_best_3d_dimension_row") or {}
    dimension = candidate.get("dimension") if isinstance(candidate.get("dimension"), dict) else {}
    model = candidate.get("model_selection") if isinstance(candidate.get("model_selection"), dict) else {}
    leakage = candidate.get("leakage") if isinstance(candidate.get("leakage"), dict) else {}
    control_lane = report.get("regulator_control_quotient_lane")
    if not isinstance(control_lane, dict):
        control_lane = {}
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
        "control_quotient_lane_is_negative_control": bool(
            control_lane.get("is_negative_control", False)
        ),
        "directional_strict_3d_ready_count": int(report.get("strict_3d_ready_count") or 0),
        "measured_overlap_geometry_receipt": bool(
            report.get("measured_overlap_geometry_receipt", False)
            or report.get("OVERLAP_NATIVE_GRAPH_GEOMETRY_RECEIPT", False)
            or _source_overlap_geometry_receipt(source_run_dir)
        ),
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


def _source_overlap_geometry_receipt(source_run_dir: str) -> bool:
    if not source_run_dir:
        return False
    source = Path(source_run_dir)
    for name in (
        "overlap_native_graph_geometry_report.json",
        "overlap_residualized_graph_geometry_report.json",
    ):
        report = _read_json(source / name)
        evidence = report.get("overlap_evidence") if isinstance(report.get("overlap_evidence"), dict) else {}
        if evidence.get("available", False) and bool(
            report.get("OVERLAP_NATIVE_GRAPH_GEOMETRY_RECEIPT", False)
            or report.get("OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_RECEIPT", False)
        ):
            return True
    return False


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
    if not report.get("channel_audit", {}).get("duplicate_channel_gate_pass", True):
        blockers.append("duplicate_primary_neutral_channels")
    for gap in report.get("strict_neutral_theory_evidence_gaps", []):
        blockers.append(str(gap))
    for blocker in report.get("quotient_geometry_contract", {}).get("blockers", []):
        blockers.append(f"quotient_geometry:{blocker}")
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
        (
            distance_corr is None
            or not np.isfinite(float(distance_corr))
            or float(distance_corr) < 0.50
        )
        and float(mean_abs_delta) > 1e-3
    )
    return bool(distance_degraded and (h3_structure_lost or stress_degraded))


def _neutral_distance_control_degraded(distance_corr: float | None, mean_abs_delta: float) -> bool:
    if float(mean_abs_delta) <= 1e-3:
        return False
    if distance_corr is None or not np.isfinite(float(distance_corr)):
        return True
    return bool(float(distance_corr) < 0.50)


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
        "locality_preserving_packet_feature_vector",
        "paired_perturbation_response_tensor",
        "paired_perturbation_control_tensors",
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
    if any("measured_overlap_correspondences" in row for row in observer_views):
        return _measured_overlap_graph_null(observer_views, rng, mode="degree_preserving_rewire")
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


def _overlap_feature_matrix(observer_views: list[dict[str, Any]]) -> np.ndarray:
    measured, evidence = _measured_overlap_feature_matrix(observer_views)
    if evidence["declared"]:
        return measured
    neutral_views = build_neutral_observer_views(observer_views)
    if not neutral_views:
        return np.zeros((0, 0), dtype=float)
    rows = []
    for view in neutral_views:
        rows.append(
            np.concatenate(
                [
                    1.00 * _normalize_or_zero(view.record_signature_hist),
                    1.00 * _normalize_or_zero(view.object_packet_hist),
                    0.80 * _normalize_or_zero(view.transition_token_hist),
                    0.65 * _normalize_or_zero(view.transition_token_persistent_hist),
                    0.75 * _normalize_or_zero(view.modular_response_hist),
                    0.60 * _normalize_or_zero(view.transition_affinity_hist),
                    0.45 * _normalize_or_zero(view.checkpoint_transition_hist),
                    0.35 * _normalize_or_zero(view.sector_transition_hist),
                    0.35 * _normalize_or_zero(view.repair_response_hist),
                ]
            )
        )
    matrix = np.vstack(rows)
    return np.where(np.isfinite(matrix), matrix, 0.0)


def _measured_overlap_feature_matrix(
    observer_views: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    count = len(patch_views)
    declared_rows = [
        view
        for view in patch_views
        if "measured_overlap_correspondences" in view
        or "measured_overlap_correspondence_schema" in view
    ]
    declared = bool(declared_rows)
    if not declared:
        return np.zeros((count, count), dtype=float), {
            "declared": False,
            "available": False,
            "source": "legacy_self_histogram_payloads",
            "blockers": ["measured_cross_observer_correspondence_producer_unavailable"],
        }
    ids: list[int] = []
    for index, view in enumerate(patch_views):
        try:
            ids.append(int(view.get("observer_id", index)))
        except (TypeError, ValueError):
            ids.append(index)
    if len(set(ids)) != len(ids):
        return np.zeros((count, count), dtype=float), {
            "declared": True,
            "available": False,
            "source": "literal_support_intersection_v1",
            "blockers": ["duplicate_observer_ids_in_measured_correspondence_rows"],
        }
    by_id = {observer_id: index for index, observer_id in enumerate(ids)}
    directed = np.zeros((count, count), dtype=float)
    producer_rows = 0
    for index, view in enumerate(patch_views):
        provenance = view.get("overlap_correspondence_evidence_provenance")
        correspondences = view.get("measured_overlap_correspondences")
        valid_producer = bool(
            isinstance(provenance, dict)
            and (
                (
                    provenance.get("cross_observer_measurement") is True
                    and provenance.get("self_histogram_synthesis") is False
                )
                or provenance.get("synthetic_negative_control") is True
            )
            and isinstance(correspondences, list)
        )
        if not valid_producer:
            continue
        producer_rows += 1
        for correspondence in correspondences:
            if not isinstance(correspondence, dict):
                continue
            try:
                peer_id = int(correspondence.get("peer_observer_id"))
                affinity = float(correspondence.get("measured_affinity", correspondence.get("jaccard", 0.0)))
            except (TypeError, ValueError):
                continue
            peer_index = by_id.get(peer_id)
            if peer_index is None or peer_index == index or not np.isfinite(affinity):
                continue
            directed[index, peer_index] = max(directed[index, peer_index], float(np.clip(affinity, 0.0, 1.0)))
    reciprocal = (directed > 0.0) & (directed.T > 0.0)
    matrix = np.sqrt(np.maximum(directed, 0.0) * np.maximum(directed.T, 0.0))
    # Self-overlap is an indexing identity, not an exported feature. Including
    # it makes the feature rows covariant under a global observer relabeling.
    np.fill_diagonal(matrix, 1.0)
    reciprocal_edge_count = int(np.count_nonzero(np.triu(reciprocal, k=1)))
    blockers: list[str] = []
    if producer_rows != count:
        blockers.append(f"measured_correspondence_rows_partial:{producer_rows}/{count}")
    if reciprocal_edge_count <= 0:
        blockers.append("no_reciprocal_measured_overlap_edges")
    return matrix, {
        "declared": True,
        "available": not blockers,
        "source": "literal_support_intersection_v1",
        "producer_row_count": producer_rows,
        "observer_count": count,
        "reciprocal_edge_count": reciprocal_edge_count,
        "blockers": blockers,
    }


def _overlap_evidence_report(observer_views: list[dict[str, Any]]) -> dict[str, Any]:
    _, evidence = _measured_overlap_feature_matrix(observer_views)
    return evidence


def _measured_overlap_presentation_invariance_report(
    observer_views: list[dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    original, evidence = _measured_overlap_feature_matrix(observer_views)
    if not evidence.get("available", False):
        return {
            "available": False,
            "global_observer_relabel_distortion": None,
            "receipt": False,
            "blockers": list(evidence.get("blockers", [])),
        }
    source_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    ids = [int(view.get("observer_id", index)) for index, view in enumerate(source_views)]
    order = np.asarray(rng.permutation(len(source_views)), dtype=np.int64)
    if len(source_views) > 1 and np.array_equal(order, np.arange(len(source_views))):
        order = np.roll(order, 1)
    patch_views = [copy.deepcopy(source_views[int(index)]) for index in order]
    permuted_labels = [int(value) for value in rng.permutation(np.asarray(ids, dtype=np.int64))]
    if len(ids) > 1 and permuted_labels == ids:
        permuted_labels = list(np.roll(np.asarray(ids, dtype=np.int64), 1))
    relabel = dict(zip(ids, permuted_labels, strict=True))
    for view in patch_views:
        view["observer_id"] = relabel[int(view["observer_id"])]
        for row in view.get("measured_overlap_correspondences", []):
            if isinstance(row, dict) and row.get("peer_observer_id") is not None:
                peer_id = int(row["peer_observer_id"])
                # A bounded control cohort can retain measured correspondence
                # rows to observers outside the cohort.  Those rows are
                # ignored by _measured_overlap_feature_matrix, and a global
                # relabeling of the cohort must leave their external labels
                # untouched.  Indexing relabel directly made the late-stage
                # producer fail whenever such a boundary row was present.
                if peer_id in relabel:
                    row["peer_observer_id"] = relabel[peer_id]
    relabeled, relabeled_evidence = _measured_overlap_feature_matrix(patch_views)
    restored = np.zeros_like(relabeled)
    if original.shape == relabeled.shape:
        restored[np.ix_(order, order)] = relabeled
    affinity_distortion = (
        float(np.max(np.abs(original - restored)))
        if original.shape == restored.shape
        else float("inf")
    )
    original_distance = _overlap_feature_distance_matrix(original)
    restored_distance = _overlap_feature_distance_matrix(restored)
    distance_distortion = float(np.max(np.abs(original_distance - restored_distance)))
    receipt = bool(
        relabeled_evidence.get("available", False)
        and affinity_distortion <= 1.0e-12
        and distance_distortion <= 1.0e-12
    )
    return {
        "available": True,
        "control": "global_observer_relabel",
        "presentation_only": True,
        "expected_geometry_change": False,
        "serialization_row_permutation_applied": True,
        "global_observer_relabel_affinity_distortion": affinity_distortion,
        "global_observer_relabel_distortion": distance_distortion,
        "receipt": receipt,
        "blockers": [] if receipt else ["global_observer_relabel_changed_measured_overlap_geometry"],
    }


def _overlap_feature_distance_matrix(features: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    features = np.where(features > 0.0, features, 0.0)
    n = features.shape[0]
    d = features.shape[1]
    if d == 0:
        return np.zeros((n, n), dtype=float)
    masses = np.sum(features, axis=1)
    distance = np.zeros((n, n), dtype=float)
    block = _overlap_distance_block_size(n=n, feature_width=d, dtype=features.dtype)
    for start in range(0, n, block):
        stop = min(n, start + block)
        shared_mass = np.minimum(features[start:stop, None, :], features[None, :, :]).sum(axis=2)
        denom = np.maximum(0.5 * (masses[start:stop, None] + masses[None, :]), eps)
        similarity = np.divide(shared_mass, denom, out=np.zeros_like(shared_mass), where=denom > eps)
        distance[start:stop, :] = 1.0 - np.clip(similarity, 0.0, 1.0)
    distance = 0.5 * (distance + distance.T)
    distance = np.where(np.isfinite(distance), np.maximum(distance, 0.0), 0.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def _overlap_distance_block_size(*, n: int, feature_width: int, dtype: np.dtype[Any]) -> int:
    bytes_per_value = np.dtype(dtype).itemsize
    target_bytes = 96 * 1024 * 1024
    denom = max(1, int(n) * max(1, int(feature_width)) * max(1, bytes_per_value))
    return max(1, min(int(n), target_bytes // denom))


def _overlap_graph_distance_from_features(
    features: np.ndarray,
    *,
    k_neighbors: int = 12,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[0] == 0:
        return {
            "affinity": np.zeros((0, 0), dtype=float),
            "distance": np.zeros((0, 0), dtype=float),
            "edge_count": 0,
            "component_count": 0,
            "finite_pair_fraction": 0.0,
            "mean_positive_affinity": None,
            "nondegenerate": False,
        }
    affinity = 1.0 - _overlap_feature_distance_matrix(features, eps=eps)
    np.fill_diagonal(affinity, 0.0)
    affinity = np.where(np.isfinite(affinity), np.clip(affinity, 0.0, 1.0), 0.0)
    n = affinity.shape[0]
    k = max(1, min(int(k_neighbors), max(n - 1, 1)))
    graph = np.zeros_like(affinity)
    for i in range(n):
        order = np.argsort(affinity[i])[::-1]
        count = 0
        for j in order:
            if int(j) == i or affinity[i, j] <= eps:
                continue
            graph[i, int(j)] = affinity[i, int(j)]
            count += 1
            if count >= k:
                break
    graph = np.maximum(graph, graph.T)
    lengths = np.where(graph > eps, -np.log(np.clip(graph, eps, 1.0)), 0.0)
    distance = shortest_path(lengths, directed=False, unweighted=False)
    finite = np.isfinite(distance)
    finite_upper = finite[np.triu_indices(n, k=1)] if n > 1 else np.asarray([], dtype=bool)
    finite_fraction = float(np.mean(finite_upper)) if finite_upper.size else 1.0
    if finite_fraction < 1.0:
        finite_values = distance[np.isfinite(distance) & (distance > 0.0)]
        fill = float(np.max(finite_values) * 2.0) if finite_values.size else 1.0
        distance = np.where(np.isfinite(distance), distance, fill)
    np.fill_diagonal(distance, 0.0)
    components = _graph_component_count(graph > eps)
    positive = graph[graph > eps]
    return {
        "affinity": graph,
        "distance": distance,
        "edge_count": int(np.count_nonzero(np.triu(graph > eps, k=1))),
        "component_count": int(components),
        "finite_pair_fraction": finite_fraction,
        "mean_positive_affinity": float(np.mean(positive)) if positive.size else None,
        "nondegenerate": bool(
            n >= 8
            and positive.size > 0
            and components == 1
            and np.any(distance[np.triu_indices(n, k=1)] > eps)
        ),
    }


def _residualize_overlap_features(features: np.ndarray, *, remove_modes: int = 1) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.size == 0:
        return np.zeros_like(features, dtype=float)
    centered = np.where(np.isfinite(features), features, 0.0)
    centered = centered - np.mean(centered, axis=0, keepdims=True)
    modes = max(0, int(remove_modes))
    if modes <= 0 or min(centered.shape) <= 1:
        return centered
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return centered
    count = min(modes, vt.shape[0])
    basis = vt[:count]
    return centered - (centered @ basis.T) @ basis


def _removed_mode_energy_fraction(features: np.ndarray, remove_modes: int) -> float | None:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.size == 0:
        return None
    centered = np.where(np.isfinite(features), features, 0.0)
    centered = centered - np.mean(centered, axis=0, keepdims=True)
    total = float(np.sum(centered**2))
    if total <= 1.0e-12:
        return None
    try:
        singular = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    count = min(max(0, int(remove_modes)), singular.size)
    removed = float(np.sum(singular[:count] ** 2))
    return removed / total


def _overlap_graph_distance_from_residual_features(
    residual_features: np.ndarray,
    *,
    k_neighbors: int = 12,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    residual = np.asarray(residual_features, dtype=float)
    if residual.ndim != 2 or residual.shape[0] == 0:
        return _graph_distance_from_affinity(np.zeros((0, 0), dtype=float), k_neighbors=k_neighbors, eps=eps)
    residual = np.where(np.isfinite(residual), residual, 0.0)
    norms = np.linalg.norm(residual, axis=1)
    valid = norms > eps
    if int(np.sum(valid)) < 4:
        return _graph_distance_from_affinity(np.zeros((residual.shape[0], residual.shape[0]), dtype=float), k_neighbors=k_neighbors, eps=eps)
    normalized = residual / np.maximum(norms[:, None], eps)
    cosine = normalized @ normalized.T
    affinity = np.where(valid[:, None] & valid[None, :], np.maximum(cosine, 0.0), 0.0)
    np.fill_diagonal(affinity, 0.0)
    return _graph_distance_from_affinity(affinity, k_neighbors=k_neighbors, eps=eps)


def _graph_distance_from_affinity(
    affinity: np.ndarray,
    *,
    k_neighbors: int,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    affinity = np.asarray(affinity, dtype=float)
    if affinity.ndim != 2 or affinity.shape[0] == 0:
        return {
            "affinity": np.zeros((0, 0), dtype=float),
            "distance": np.zeros((0, 0), dtype=float),
            "edge_count": 0,
            "component_count": 0,
            "finite_pair_fraction": 0.0,
            "mean_positive_affinity": None,
            "nondegenerate": False,
        }
    affinity = np.where(np.isfinite(affinity), np.clip(affinity, 0.0, 1.0), 0.0)
    np.fill_diagonal(affinity, 0.0)
    n = affinity.shape[0]
    k = max(1, min(int(k_neighbors), max(n - 1, 1)))
    graph = np.zeros_like(affinity)
    for i in range(n):
        order = np.argsort(affinity[i])[::-1]
        count = 0
        for j in order:
            if int(j) == i or affinity[i, j] <= eps:
                continue
            graph[i, int(j)] = affinity[i, int(j)]
            count += 1
            if count >= k:
                break
    graph = np.maximum(graph, graph.T)
    lengths = np.where(graph > eps, -np.log(np.clip(graph, eps, 1.0)), 0.0)
    distance = shortest_path(lengths, directed=False, unweighted=False)
    finite = np.isfinite(distance)
    finite_upper = finite[np.triu_indices(n, k=1)] if n > 1 else np.asarray([], dtype=bool)
    finite_fraction = float(np.mean(finite_upper)) if finite_upper.size else 1.0
    if finite_fraction < 1.0:
        finite_values = distance[np.isfinite(distance) & (distance > 0.0)]
        fill = float(np.max(finite_values) * 2.0) if finite_values.size else 1.0
        distance = np.where(np.isfinite(distance), distance, fill)
    np.fill_diagonal(distance, 0.0)
    components = _graph_component_count(graph > eps)
    positive = graph[graph > eps]
    return {
        "affinity": graph,
        "distance": distance,
        "edge_count": int(np.count_nonzero(np.triu(graph > eps, k=1))),
        "component_count": int(components),
        "finite_pair_fraction": finite_fraction,
        "mean_positive_affinity": float(np.mean(positive)) if positive.size else None,
        "nondegenerate": bool(
            n >= 8
            and positive.size > 0
            and components == 1
            and np.any(distance[np.triu_indices(n, k=1)] > eps)
        ),
    }


def _graph_component_count(adjacency: np.ndarray) -> int:
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0] if adjacency.ndim == 2 else 0
    seen = np.zeros(n, dtype=bool)
    count = 0
    for start in range(n):
        if seen[start]:
            continue
        count += 1
        stack = [start]
        seen[start] = True
        while stack:
            current = stack.pop()
            for nxt in np.flatnonzero(adjacency[current]):
                if not seen[int(nxt)]:
                    seen[int(nxt)] = True
                    stack.append(int(nxt))
    return count


def _overlap_graph_rank_selection(affinity: np.ndarray) -> dict[str, Any]:
    affinity = np.asarray(affinity, dtype=float)
    if affinity.ndim != 2 or affinity.shape[0] < 4 or not np.any(affinity > 0.0):
        return {
            "available": False,
            "rank3_selector_receipt": False,
            "reason": "empty_or_too_small_overlap_graph",
        }
    degree = np.sum(np.clip(affinity, 0.0, None), axis=1)
    keep = degree > 1.0e-12
    if int(np.sum(keep)) < 4:
        return {
            "available": False,
            "rank3_selector_receipt": False,
            "reason": "too_few_nonisolated_graph_nodes",
        }
    sub = affinity[np.ix_(keep, keep)]
    degree = np.sum(sub, axis=1)
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1.0e-12)))
    normalized = inv_sqrt @ sub @ inv_sqrt
    eigenvalues = np.linalg.eigvalsh((normalized + normalized.T) * 0.5)[::-1]
    eigenvalues = np.asarray([float(value) for value in eigenvalues if np.isfinite(value) and value > 1.0e-12])
    if eigenvalues.size == 0:
        return {
            "available": False,
            "rank3_selector_receipt": False,
            "reason": "no_positive_graph_spectrum",
        }
    max_rank = int(min(16, eigenvalues.size))
    values = eigenvalues[:max_rank]
    gaps = values[:-1] - values[1:] if values.size > 1 else np.asarray([], dtype=float)
    largest_gap_rank = int(np.argmax(gaps) + 1) if gaps.size else 1
    cumulative = np.cumsum(values) / max(float(np.sum(eigenvalues)), 1.0e-12)
    rank3_ev = float(cumulative[min(2, cumulative.size - 1)])
    participation = float((np.sum(eigenvalues) ** 2) / max(float(np.sum(eigenvalues**2)), 1.0e-12))
    entropy_weights = eigenvalues / max(float(np.sum(eigenvalues)), 1.0e-12)
    effective = float(np.exp(-np.sum(entropy_weights * np.log(np.maximum(entropy_weights, 1.0e-12)))))
    model_order = _spectral_model_order_selection(eigenvalues)
    nontrivial = eigenvalues[1:]
    if nontrivial.size:
        nontrivial_values = nontrivial[:max_rank]
        nontrivial_gaps = (
            nontrivial_values[:-1] - nontrivial_values[1:]
            if nontrivial_values.size > 1
            else np.asarray([], dtype=float)
        )
        nontrivial_largest_gap_rank = (
            int(np.argmax(nontrivial_gaps) + 1) if nontrivial_gaps.size else 1
        )
        nontrivial_cumulative = np.cumsum(nontrivial_values) / max(
            float(np.sum(nontrivial)),
            1.0e-12,
        )
        nontrivial_rank3_ev = float(
            nontrivial_cumulative[min(2, nontrivial_cumulative.size - 1)]
        )
        nontrivial_weights = nontrivial / max(float(np.sum(nontrivial)), 1.0e-12)
        nontrivial_effective = float(
            np.exp(-np.sum(nontrivial_weights * np.log(np.maximum(nontrivial_weights, 1.0e-12))))
        )
        nontrivial_model_order = _spectral_model_order_selection(nontrivial)
    else:
        nontrivial_values = np.asarray([], dtype=float)
        nontrivial_largest_gap_rank = None
        nontrivial_rank3_ev = None
        nontrivial_effective = None
        nontrivial_model_order = {
            "available": False,
            "consensus_rank": None,
            "profile_likelihood_rank": None,
            "broken_stick_rank": None,
            "reason": "no_nontrivial_spectrum",
        }
    nontrivial_rank3_selector = bool(
        nontrivial_largest_gap_rank == 3
        and nontrivial_rank3_ev is not None
        and nontrivial_rank3_ev >= 0.50
        and nontrivial_effective is not None
        and nontrivial_effective <= 8.0
    )
    receipt = bool(largest_gap_rank == 3 and rank3_ev >= 0.50 and effective <= 8.0)
    return {
        "available": True,
        "rank3_selector_receipt": receipt,
        "largest_gap_rank": largest_gap_rank,
        "model_order_selection": model_order,
        "model_order_rank3_selector_receipt": bool(model_order.get("consensus_rank") == 3),
        "rank3_cumulative_explained_variance": rank3_ev,
        "effective_rank": effective,
        "participation_rank": participation,
        "nontrivial_rank3_selector_receipt": nontrivial_rank3_selector,
        "nontrivial_largest_gap_rank": nontrivial_largest_gap_rank,
        "nontrivial_model_order_selection": nontrivial_model_order,
        "nontrivial_model_order_rank3_selector_receipt": bool(
            nontrivial_model_order.get("consensus_rank") == 3
        ),
        "nontrivial_rank3_cumulative_explained_variance": nontrivial_rank3_ev,
        "nontrivial_effective_rank": nontrivial_effective,
        "eigenvalue_count": int(eigenvalues.size),
        "top_eigenvalues": [float(value) for value in eigenvalues[: min(10, eigenvalues.size)]],
        "nontrivial_top_eigenvalues": [
            float(value) for value in nontrivial_values[: min(10, nontrivial_values.size)]
        ],
        "claim_boundary": (
            "Target-rank-free rank selector on the observer-overlap graph normalized spectrum. "
            "It is independent of downstream dimension fitting and H3 model selection. "
            "The nontrivial fields rerun the same audit after excluding the Perron/common graph mode; "
            "they are reported as obstruction diagnostics and do not by themselves promote strict neutral bulk."
        ),
    }


def _spectral_model_order_selection(eigenvalues: np.ndarray, *, max_rank: int = 12) -> dict[str, Any]:
    """Target-rank-free model-order diagnostics for a positive spectrum.

    This deliberately does not promote strict neutral bulk. It records two
    independent order selectors so the sweep can distinguish "rank 1 dominates"
    from "different target-free selectors disagree about the intrinsic order".
    """

    values = np.asarray(eigenvalues, dtype=float)
    values = values[np.isfinite(values) & (values > 1.0e-12)]
    if values.size < 4:
        return {
            "available": False,
            "consensus_rank": None,
            "profile_likelihood_rank": None,
            "broken_stick_rank": None,
            "reason": "too_few_positive_eigenvalues",
        }
    values = np.sort(values)[::-1]
    max_rank = int(max(1, min(int(max_rank), values.size - 2)))
    profile_rank = _profile_likelihood_spectral_rank(values, max_rank=max_rank)
    broken_rank = _broken_stick_spectral_rank(values, max_rank=max_rank)
    consensus = profile_rank if profile_rank == broken_rank else None
    total = max(float(np.sum(values)), 1.0e-12)
    return {
        "available": True,
        "consensus_rank": consensus,
        "profile_likelihood_rank": profile_rank,
        "broken_stick_rank": broken_rank,
        "selectors_agree": bool(consensus is not None),
        "max_rank_considered": int(max_rank),
        "rank3_cumulative_explained_variance": float(np.sum(values[: min(3, values.size)]) / total),
        "claim_boundary": (
            "Target-rank-free model-order diagnostic from the overlap graph spectrum. It is recorded "
            "for obstruction analysis only; strict neutral bulk still uses the hard overlap/H3/rank gates."
        ),
    }


def _profile_likelihood_spectral_rank(values: np.ndarray, *, max_rank: int) -> int | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 1.0e-12)]
    if values.size < 4:
        return None
    spectrum = np.log(values / max(float(np.sum(values)), 1.0e-12))
    max_rank = int(max(1, min(int(max_rank), spectrum.size - 2)))
    best_rank: int | None = None
    best_bic = float("inf")
    for rank in range(1, max_rank + 1):
        head = spectrum[:rank]
        tail = spectrum[rank:]
        if head.size == 0 or tail.size < 2:
            continue
        head_var = max(float(np.var(head)), 1.0e-8)
        tail_var = max(float(np.var(tail)), 1.0e-8)
        head_ll = -0.5 * head.size * (math.log(2.0 * math.pi * head_var) + 1.0)
        tail_ll = -0.5 * tail.size * (math.log(2.0 * math.pi * tail_var) + 1.0)
        bic = -2.0 * (head_ll + tail_ll) + 4.0 * math.log(float(spectrum.size))
        if bic < best_bic:
            best_bic = float(bic)
            best_rank = int(rank)
    return best_rank


def _broken_stick_spectral_rank(values: np.ndarray, *, max_rank: int) -> int | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 1.0e-12)]
    if values.size < 4:
        return None
    values = np.sort(values)[::-1]
    weights = values / max(float(np.sum(values)), 1.0e-12)
    n = values.size
    expected = np.asarray([sum(1.0 / j for j in range(index, n + 1)) / n for index in range(1, n + 1)])
    rank = int(sum(1 for value, threshold in zip(weights, expected) if value > threshold))
    return int(max(1, min(rank, int(max_rank))))


def _overlap_graph_control_row(
    name: str,
    original_distance: np.ndarray,
    control_features: np.ndarray,
    sampled_observer_rows: list[dict[str, Any]],
    *,
    seed: int,
    max_model_points: int,
    k_neighbors: int,
    original_spatial_candidate: bool,
) -> dict[str, Any]:
    graph = _overlap_graph_distance_from_features(control_features, k_neighbors=int(k_neighbors))
    distance = graph["distance"]
    corr = _upper_triangle_corr(original_distance, distance)
    delta = _mean_abs_upper_delta(original_distance, distance)
    dimension = strict_neutral_dimension_report(distance)
    model = neutral_model_selection(
        distance,
        seed=int(seed),
        max_points=min(int(max_model_points), len(sampled_observer_rows)),
    )
    leakage = neutral_leakage_audit(distance, sampled_observer_rows)
    candidate = _spatial_3d_ready(dimension, model, leakage)
    distance_degraded = _neutral_distance_control_degraded(corr, delta)
    expected_failure = bool(distance_degraded and (not original_spatial_candidate or not candidate))
    return {
        "control": str(name),
        "distance_shape_correlation_to_original": corr,
        "mean_abs_distance_delta": delta,
        "distance_degraded": bool(distance_degraded),
        "graph_summary": {
            "edge_count": graph["edge_count"],
            "component_count": graph["component_count"],
            "finite_pair_fraction": graph["finite_pair_fraction"],
        },
        "dimension": dimension,
        "model_selection": model,
        "leakage": leakage,
        "spatial_3d_candidate_survives_control": bool(candidate),
        "expected_failure_observed": expected_failure,
        "claim_boundary": (
            "Negative control for the observer-overlap graph geometry. A control passes only if it "
            "changes the graph distance and does not preserve the same spatial-3D candidate when the "
            "original overlap graph has one."
        ),
    }


def _overlap_residual_graph_control_row(
    name: str,
    original_distance: np.ndarray,
    control_features: np.ndarray,
    sampled_observer_rows: list[dict[str, Any]],
    *,
    seed: int,
    max_model_points: int,
    k_neighbors: int,
    remove_modes: int,
    original_spatial_candidate: bool,
) -> dict[str, Any]:
    residual = _residualize_overlap_features(control_features, remove_modes=int(remove_modes))
    graph = _overlap_graph_distance_from_residual_features(residual, k_neighbors=int(k_neighbors))
    distance = graph["distance"]
    corr = _upper_triangle_corr(original_distance, distance)
    delta = _mean_abs_upper_delta(original_distance, distance)
    dimension = strict_neutral_dimension_report(distance)
    model = neutral_model_selection(
        distance,
        seed=int(seed),
        max_points=min(int(max_model_points), len(sampled_observer_rows)),
    )
    leakage = neutral_leakage_audit(distance, sampled_observer_rows)
    candidate = _spatial_3d_ready(dimension, model, leakage)
    distance_degraded = _neutral_distance_control_degraded(corr, delta)
    expected_failure = bool(distance_degraded and (not original_spatial_candidate or not candidate))
    return {
        "control": str(name),
        "distance_shape_correlation_to_original": corr,
        "mean_abs_distance_delta": delta,
        "distance_degraded": bool(distance_degraded),
        "graph_summary": {
            "edge_count": graph["edge_count"],
            "component_count": graph["component_count"],
            "finite_pair_fraction": graph["finite_pair_fraction"],
        },
        "dimension": dimension,
        "model_selection": model,
        "leakage": leakage,
        "spatial_3d_candidate_survives_control": bool(candidate),
        "expected_failure_observed": expected_failure,
        "claim_boundary": (
            "Negative control for residualized observer-overlap graph geometry. A control passes only "
            "if it changes the residual graph distance and does not preserve the same spatial-3D "
            "candidate when the original residual lane has one."
        ),
    }


def _overlap_native_control_row(
    name: str,
    original_distance: np.ndarray,
    control_features: np.ndarray,
    sampled_observer_rows: list[dict[str, Any]],
    *,
    seed: int,
    max_model_points: int,
    original_spatial_candidate: bool,
) -> dict[str, Any]:
    distance = _overlap_feature_distance_matrix(control_features)
    corr = _upper_triangle_corr(original_distance, distance)
    delta = _mean_abs_upper_delta(original_distance, distance)
    dimension = strict_neutral_dimension_report(distance)
    model = neutral_model_selection(
        distance,
        seed=int(seed),
        max_points=min(int(max_model_points), len(sampled_observer_rows)),
    )
    leakage = neutral_leakage_audit(distance, sampled_observer_rows)
    candidate = _spatial_3d_ready(dimension, model, leakage)
    distance_degraded = _neutral_distance_control_degraded(corr, delta)
    expected_failure = bool(distance_degraded and (not original_spatial_candidate or not candidate))
    return {
        "control": str(name),
        "distance_shape_correlation_to_original": corr,
        "mean_abs_distance_delta": delta,
        "distance_degraded": bool(distance_degraded),
        "dimension": dimension,
        "model_selection": model,
        "leakage": leakage,
        "spatial_3d_candidate_survives_control": bool(candidate),
        "expected_failure_observed": expected_failure,
        "claim_boundary": (
            "Negative control for the observer-overlap distance. A control passes only if it changes "
            "the overlap distance and does not preserve the same spatial-3D candidate when the original "
            "overlap lane has one."
        ),
    }


def _shuffle_overlap_payloads(
    observer_views: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if any("measured_overlap_correspondences" in row for row in observer_views):
        return _measured_overlap_graph_null(observer_views, rng, mode="degree_preserving_rewire")
    shuffled = copy.deepcopy(observer_views)
    patch_indices = [index for index, view in enumerate(shuffled) if view.get("view_type") == "patch_observer"]
    for key in (
        "record_signature_histogram",
        "object_packet_histogram",
        "transition_history_histograms",
        "transition_affinity_histograms",
        "modular_response_histograms",
        "transition_history_descriptor",
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


def _permute_overlap_packet_labels(
    observer_views: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if any("measured_overlap_correspondences" in row for row in observer_views):
        return _measured_overlap_graph_null(observer_views, rng, mode="edge_weight_permutation")
    shuffled = copy.deepcopy(observer_views)
    for view in shuffled:
        if view.get("view_type") != "patch_observer":
            continue
        correspondences = view.get("measured_overlap_correspondences")
        if isinstance(correspondences, list):
            peer_ids = [
                int(row["peer_observer_id"])
                for row in correspondences
                if isinstance(row, dict) and row.get("peer_observer_id") is not None
            ]
            if peer_ids:
                permuted = rng.permutation(np.asarray(peer_ids, dtype=np.int64))
                offset = 0
                for row in correspondences:
                    if isinstance(row, dict) and row.get("peer_observer_id") is not None:
                        row["peer_observer_id"] = int(permuted[offset])
                        offset += 1
        for key in (
            "record_signature_histogram",
            "object_packet_histogram",
            "transition_history_histograms",
            "transition_affinity_histograms",
            "modular_response_histograms",
        ):
            if isinstance(view.get(key), dict):
                view[key] = _randomize_histogram_keys(view[key], rng)
    return shuffled


def _measured_overlap_graph_null(
    observer_views: list[dict[str, Any]],
    rng: np.random.Generator,
    *,
    mode: str,
) -> list[dict[str, Any]]:
    shuffled = copy.deepcopy(observer_views)
    patch_rows = [row for row in shuffled if row.get("view_type") == "patch_observer"]
    ids = [int(row.get("observer_id", index)) for index, row in enumerate(patch_rows)]
    id_set = set(ids)
    directed: dict[tuple[int, int], dict[str, Any]] = {}
    for row_index, row in enumerate(patch_rows):
        source_id = ids[row_index]
        for payload in row.get("measured_overlap_correspondences", []):
            if not isinstance(payload, dict):
                continue
            try:
                peer_id = int(payload.get("peer_observer_id"))
            except (TypeError, ValueError):
                continue
            if peer_id in id_set and peer_id != source_id:
                directed[(source_id, peer_id)] = copy.deepcopy(payload)
    edges: list[tuple[int, int, dict[str, Any]]] = []
    for source_id, peer_id in sorted(directed):
        if source_id >= peer_id or (peer_id, source_id) not in directed:
            continue
        payload = copy.deepcopy(directed[(source_id, peer_id)])
        edges.append((source_id, peer_id, payload))
    if mode == "degree_preserving_rewire":
        rewired = [(left, right, payload) for left, right, payload in edges]
        edge_set = {tuple(sorted((left, right))) for left, right, _ in rewired}
        attempts = max(32, 20 * len(rewired))
        for _ in range(attempts):
            if len(rewired) < 2:
                break
            first, second = rng.choice(len(rewired), size=2, replace=False)
            a, b, payload_ab = rewired[int(first)]
            c, d, payload_cd = rewired[int(second)]
            if rng.random() < 0.5:
                c, d = d, c
            if len({a, b, c, d}) < 4:
                continue
            proposed = (tuple(sorted((a, d))), tuple(sorted((c, b))))
            old = (tuple(sorted((a, b))), tuple(sorted((c, d))))
            if proposed[0] == proposed[1] or any(edge in edge_set - set(old) for edge in proposed):
                continue
            edge_set.difference_update(old)
            edge_set.update(proposed)
            rewired[int(first)] = (a, d, payload_ab)
            rewired[int(second)] = (c, b, payload_cd)
        edges = rewired
    elif mode == "edge_weight_permutation" and len(edges) > 1:
        payloads = [copy.deepcopy(payload) for _, _, payload in edges]
        order = rng.permutation(len(payloads))
        edges = [
            (left, right, payloads[int(order[index])])
            for index, (left, right, _payload) in enumerate(edges)
        ]

    internal_by_id: dict[int, list[dict[str, Any]]] = {observer_id: [] for observer_id in ids}
    for left_id, right_id, payload in edges:
        common = {
            key: copy.deepcopy(value)
            for key, value in payload.items()
            if key != "peer_observer_id"
        }
        internal_by_id[left_id].append({**common, "peer_observer_id": int(right_id)})
        internal_by_id[right_id].append({**common, "peer_observer_id": int(left_id)})
    for row_index, row in enumerate(patch_rows):
        source_id = ids[row_index]
        outside = [
            copy.deepcopy(payload)
            for payload in row.get("measured_overlap_correspondences", [])
            if isinstance(payload, dict)
            and int(payload.get("peer_observer_id", source_id)) not in id_set
        ]
        row["measured_overlap_correspondences"] = sorted(
            outside + internal_by_id[source_id],
            key=lambda payload: int(payload.get("peer_observer_id", -1)),
        )
        row["overlap_correspondence_evidence_provenance"] = {
            "producer": f"neutral_negative_control.{mode}",
            "cross_observer_measurement": False,
            "self_histogram_synthesis": False,
            "synthetic_negative_control": True,
            "marginal_preservation": (
                "node_degree_and_edge_payload_multiset"
                if mode == "degree_preserving_rewire"
                else "edge_topology_and_edge_payload_multiset"
            ),
        }
    return shuffled


def _overlap_histogram_null_features(features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.size == 0:
        return features.copy()
    positive = np.where(features > 0.0, features, 0.0)
    column_mean = np.mean(positive, axis=0)
    if not np.any(column_mean > 0.0):
        return rng.random(size=features.shape)
    weights = column_mean / max(float(np.sum(column_mean)), 1.0e-12)
    row_mass = np.sum(positive, axis=1)
    out = np.zeros_like(positive)
    width = positive.shape[1]
    for index, mass in enumerate(row_mass):
        if mass <= 1.0e-12:
            continue
        draws = rng.choice(width, size=max(1, min(width, int(np.count_nonzero(positive[index])) or 1)), replace=False, p=weights)
        values = rng.random(size=draws.size)
        values = values / max(float(np.sum(values)), 1.0e-12) * float(mass)
        out[index, draws] = values
    return out


def _overlap_native_neutral_control_markdown(report: dict[str, Any]) -> str:
    blockers = report.get("blockers") or []
    rows = report.get("control_rows") if isinstance(report.get("control_rows"), list) else []
    lines = [
        "# Overlap-Native Neutral Control",
        "",
        f"- overlap-native negative-control receipt: `{str(report.get('OVERLAP_NATIVE_NEGATIVE_CONTROL_RECEIPT', False)).lower()}`",
        f"- sampled observers: `{report.get('sampled_observer_count', 0)}` / `{report.get('observer_count', 0)}`",
        f"- overlap spatial-3D candidate: `{str(report.get('overlap_native_spatial_3d_candidate', False)).lower()}`",
        f"- overlap strict-H3 candidate: `{str(report.get('overlap_native_strict_h3_candidate', False)).lower()}`",
        "",
        "## Controls",
        "",
    ]
    if rows:
        for row in rows:
            lines.append(
                "- "
                f"`{row.get('control')}`: expected failure `{str(row.get('expected_failure_observed', False)).lower()}`, "
                f"distance corr `{row.get('distance_shape_correlation_to_original')}`, "
                f"delta `{row.get('mean_abs_distance_delta')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- none")
    obstruction = report.get("rank_obstruction_summary") if isinstance(report.get("rank_obstruction_summary"), dict) else {}
    lines.extend(["", "## Rank Obstruction", ""])
    if obstruction.get("available", False):
        lines.extend(
            [
                f"- primary obstruction: `{obstruction.get('primary_obstruction')}`",
                f"- dominant largest-gap rank: `{obstruction.get('dominant_largest_gap_rank')}`",
                f"- max rank-3 cumulative explained variance: `{obstruction.get('max_rank3_cumulative_explained_variance')}`",
                f"- median effective rank: `{obstruction.get('median_effective_rank')}`",
                f"- spatial max rank-3 cumulative explained variance: `{obstruction.get('spatial_max_rank3_cumulative_explained_variance')}`",
                f"- spatial median effective rank: `{obstruction.get('spatial_median_effective_rank')}`",
            ]
        )
    else:
        lines.append("- unavailable")
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def _overlap_native_graph_geometry_markdown(report: dict[str, Any]) -> str:
    blockers = report.get("blockers") or []
    graph = report.get("graph_summary") if isinstance(report.get("graph_summary"), dict) else {}
    rank = report.get("rank_selection") if isinstance(report.get("rank_selection"), dict) else {}
    dimension = report.get("dimension") if isinstance(report.get("dimension"), dict) else {}
    model = report.get("model_selection") if isinstance(report.get("model_selection"), dict) else {}
    lines = [
        "# Overlap-Native Graph Geometry",
        "",
        f"- graph geometry receipt: `{str(report.get('OVERLAP_NATIVE_GRAPH_GEOMETRY_RECEIPT', False)).lower()}`",
        f"- sampled observers: `{report.get('sampled_observer_count', 0)}` / `{report.get('observer_count', 0)}`",
        f"- graph spatial-3D candidate: `{str(report.get('overlap_graph_spatial_3d_candidate', False)).lower()}`",
        f"- graph strict-H3 candidate: `{str(report.get('overlap_graph_strict_h3_candidate', False)).lower()}`",
        f"- graph edges: `{graph.get('edge_count')}`",
        f"- graph components: `{graph.get('component_count')}`",
        f"- median dimension: `{dimension.get('median_dimension_estimate')}`",
        f"- selected model: `{model.get('best_model')}`",
        f"- rank selector: `{str(rank.get('rank3_selector_receipt', False)).lower()}`",
        f"- largest-gap rank: `{rank.get('largest_gap_rank')}`",
        f"- nontrivial rank selector: `{str(rank.get('nontrivial_rank3_selector_receipt', False)).lower()}`",
        f"- nontrivial largest-gap rank: `{rank.get('nontrivial_largest_gap_rank')}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def _overlap_residualized_graph_geometry_markdown(report: dict[str, Any]) -> str:
    blockers = report.get("blockers") or []
    graph = report.get("graph_summary") if isinstance(report.get("graph_summary"), dict) else {}
    rank = report.get("rank_selection") if isinstance(report.get("rank_selection"), dict) else {}
    dimension = report.get("dimension") if isinstance(report.get("dimension"), dict) else {}
    model = report.get("model_selection") if isinstance(report.get("model_selection"), dict) else {}
    residual = report.get("residualization") if isinstance(report.get("residualization"), dict) else {}
    lines = [
        "# Residualized Overlap Graph Geometry",
        "",
        f"- residual graph receipt: `{str(report.get('OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_RECEIPT', False)).lower()}`",
        f"- sampled observers: `{report.get('sampled_observer_count', 0)}` / `{report.get('observer_count', 0)}`",
        f"- residual spatial-3D candidate: `{str(report.get('overlap_residual_graph_spatial_3d_candidate', False)).lower()}`",
        f"- residual strict-H3 candidate: `{str(report.get('overlap_residual_graph_strict_h3_candidate', False)).lower()}`",
        f"- removed common modes: `{report.get('remove_modes')}`",
        f"- raw largest-gap rank: `{residual.get('raw_largest_gap_rank')}`",
        f"- removed common-mode energy fraction: `{residual.get('removed_common_mode_energy_fraction')}`",
        f"- graph edges: `{graph.get('edge_count')}`",
        f"- graph components: `{graph.get('component_count')}`",
        f"- median dimension: `{dimension.get('median_dimension_estimate')}`",
        f"- selected model: `{model.get('best_model')}`",
        f"- rank selector: `{str(rank.get('rank3_selector_receipt', False)).lower()}`",
        f"- largest-gap rank after residualization: `{rank.get('largest_gap_rank')}`",
        f"- nontrivial rank selector: `{str(rank.get('nontrivial_rank3_selector_receipt', False)).lower()}`",
        f"- nontrivial largest-gap rank after residualization: `{rank.get('nontrivial_largest_gap_rank')}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def _overlap_native_graph_geometry_sweep_markdown(report: dict[str, Any]) -> str:
    blockers = report.get("blockers") or []
    best = report.get("best_case") if isinstance(report.get("best_case"), dict) else {}
    closest = report.get("closest_strict_rows") if isinstance(report.get("closest_strict_rows"), list) else []
    coincidence = report.get("gate_coincidence_summary") if isinstance(
        report.get("gate_coincidence_summary"), dict
    ) else {}
    lines = [
        "# Overlap-Native Graph Geometry Sweep",
        "",
        f"- sweep receipt: `{str(report.get('OVERLAP_NATIVE_GRAPH_GEOMETRY_SWEEP_RECEIPT', False)).lower()}`",
        f"- case count: `{report.get('case_count', 0)}`",
        f"- graph receipts: `{report.get('graph_geometry_receipt_count', 0)}`",
        f"- spatial-3D candidates: `{report.get('spatial_3d_candidate_count', 0)}`",
        f"- strict-H3 candidates: `{report.get('strict_h3_candidate_count', 0)}`",
        f"- rank-3 selectors: `{report.get('rank3_selector_count', 0)}`",
        f"- nontrivial rank-3 selectors: `{(report.get('rank_obstruction_summary') or {}).get('nontrivial_rank3_selector_count')}`",
        f"- spatial-H3 plus independent rank-3 coincidences: `{coincidence.get('spatial_h3_independent_rank3_selector_count')}`",
        f"- spatial-H3 plus nontrivial rank-3 coincidences: `{coincidence.get('spatial_h3_nontrivial_rank3_selector_count')}`",
        "",
        "## Best Case",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- source run: `{best.get('source_run_dir')}`",
                f"- seed / max points / k: `{best.get('seed')}` / `{best.get('max_model_points')}` / `{best.get('k_neighbors')}`",
                f"- spatial-3D candidate: `{str(best.get('spatial_3d_candidate', False)).lower()}`",
                f"- strict-H3 candidate: `{str(best.get('strict_h3_candidate', False)).lower()}`",
                f"- rank-3 selector: `{str(best.get('rank3_selector', False)).lower()}`",
                f"- nontrivial rank-3 selector: `{str(best.get('nontrivial_rank3_selector', False)).lower()}`",
                f"- nontrivial largest-gap rank: `{best.get('nontrivial_largest_gap_rank')}`",
                f"- median dimension: `{best.get('median_dimension')}`",
                f"- selected model: `{best.get('selected_model')}`",
                f"- blockers: `{best.get('blockers', [])}`",
            ]
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Closest Strict-Gate Rows", ""])
    if closest:
        for row in closest[:8]:
            lines.append(
                "- "
                f"`{row.get('source_run_dir')}` seed `{row.get('seed')}` max `{row.get('max_model_points')}` "
                f"k `{row.get('k_neighbors')}` score `{row.get('gate_score')}` "
                f"dim `{row.get('median_dimension')}` model `{row.get('selected_model')}` "
                f"missing `{row.get('missing_strict_gates', [])}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def _overlap_residualized_graph_geometry_sweep_markdown(report: dict[str, Any]) -> str:
    blockers = report.get("blockers") or []
    best = report.get("best_case") if isinstance(report.get("best_case"), dict) else {}
    closest = report.get("closest_strict_rows") if isinstance(report.get("closest_strict_rows"), list) else []
    coincidence = report.get("gate_coincidence_summary") if isinstance(
        report.get("gate_coincidence_summary"), dict
    ) else {}
    obstruction = report.get("rank_obstruction_summary") if isinstance(
        report.get("rank_obstruction_summary"), dict
    ) else {}
    lines = [
        "# Residualized Overlap Graph Geometry Sweep",
        "",
        f"- sweep receipt: `{str(report.get('OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_SWEEP_RECEIPT', False)).lower()}`",
        f"- case count: `{report.get('case_count', 0)}`",
        f"- residual graph receipts: `{report.get('residual_graph_receipt_count', 0)}`",
        f"- spatial-3D candidates: `{report.get('spatial_3d_candidate_count', 0)}`",
        f"- strict-H3 candidates: `{report.get('strict_h3_candidate_count', 0)}`",
        f"- rank-3 selectors: `{report.get('rank3_selector_count', 0)}`",
        f"- nontrivial rank-3 selectors: `{obstruction.get('nontrivial_rank3_selector_count')}`",
        f"- spatial-H3 plus independent rank-3 coincidences: `{coincidence.get('spatial_h3_independent_rank3_selector_count')}`",
        f"- spatial-H3 plus nontrivial rank-3 coincidences: `{coincidence.get('spatial_h3_nontrivial_rank3_selector_count')}`",
        f"- raw rank-1 cases: `{obstruction.get('raw_largest_gap_rank1_count')}`",
        "",
        "## Best Case",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- source run: `{best.get('source_run_dir')}`",
                f"- seed / max points / k / remove modes: `{best.get('seed')}` / `{best.get('max_model_points')}` / `{best.get('k_neighbors')}` / `{best.get('remove_modes')}`",
                f"- spatial-3D candidate: `{str(best.get('spatial_3d_candidate', False)).lower()}`",
                f"- strict-H3 candidate: `{str(best.get('strict_h3_candidate', False)).lower()}`",
                f"- rank-3 selector: `{str(best.get('rank3_selector', False)).lower()}`",
                f"- nontrivial rank-3 selector: `{str(best.get('nontrivial_rank3_selector', False)).lower()}`",
                f"- raw largest-gap rank: `{best.get('raw_largest_gap_rank')}`",
                f"- residual largest-gap rank: `{best.get('largest_gap_rank')}`",
                f"- nontrivial largest-gap rank: `{best.get('nontrivial_largest_gap_rank')}`",
                f"- median dimension: `{best.get('median_dimension')}`",
                f"- selected model: `{best.get('selected_model')}`",
                f"- blockers: `{best.get('blockers', [])}`",
            ]
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Closest Strict-Gate Rows", ""])
    if closest:
        for row in closest[:8]:
            lines.append(
                "- "
                f"`{row.get('source_run_dir')}` seed `{row.get('seed')}` max `{row.get('max_model_points')}` "
                f"k `{row.get('k_neighbors')}` remove `{row.get('remove_modes')}` score `{row.get('gate_score')}` "
                f"dim `{row.get('median_dimension')}` model `{row.get('selected_model')}` "
                f"missing `{row.get('missing_strict_gates', [])}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)


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
        "E4": (4, 1),
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
    gram = _double_center_squared_distance(squared)
    values, vectors = _symmetric_top_eigenpairs(gram, dim)
    positive = values[:dim]
    if positive.size < dim or np.any(positive <= 1e-12):
        positive = np.maximum(positive, 1e-12)
    return vectors[:, :dim] * np.sqrt(positive)[None, :]


def _double_center_squared_distance(squared: np.ndarray) -> np.ndarray:
    squared = np.asarray(squared, dtype=float)
    row_mean = np.mean(squared, axis=1, keepdims=True)
    col_mean = np.mean(squared, axis=0, keepdims=True)
    total_mean = float(np.mean(squared))
    return -0.5 * (squared - row_mean - col_mean + total_mean)


def _positive_spectral_coords(gram: np.ndarray, dim: int) -> np.ndarray | None:
    values, vectors = _symmetric_top_eigenpairs(gram, dim)
    if values.size < dim:
        return None
    values = np.maximum(values[:dim], 1e-12)
    return vectors[:, :dim] * np.sqrt(values)[None, :]


def _hyperbolic_indefinite_reconstruction(distance: np.ndarray, *, dim: int, radius: float) -> np.ndarray | None:
    lorentz_gram = -np.cosh(np.minimum(distance / max(radius, 1e-12), 20.0))
    bottom_values, bottom_vectors = _symmetric_bottom_eigenpairs(lorentz_gram, 1)
    top_values, top_vectors = _symmetric_top_eigenpairs(lorentz_gram, dim)
    if bottom_values.size < 1 or top_values.size < dim or bottom_values[0] >= -1e-12:
        return None
    positive = top_values[:dim]
    if np.any(positive <= 1e-12):
        return None
    time = np.sqrt(-bottom_values[0]) * bottom_vectors[:, 0]
    space = top_vectors[:, :dim] * np.sqrt(positive)[None, :]
    time = np.abs(time) + 1e-9
    inner = -np.outer(time, time) + space @ space.T
    cosh_arg = np.maximum(-inner, 1.0)
    predicted = radius * np.arccosh(cosh_arg)
    np.fill_diagonal(predicted, 0.0)
    return predicted


def _symmetric_top_eigenpairs(matrix: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    sym = _symmetrized_matrix(matrix)
    n = sym.shape[0]
    k = max(0, min(int(count), n))
    if k <= 0:
        return np.zeros(0, dtype=float), np.zeros((n, 0), dtype=float)
    try:
        values, vectors = dense_eigh(sym, subset_by_index=[n - k, n - 1], check_finite=False)
    except Exception:
        values, vectors = np.linalg.eigh(sym)
    order = np.argsort(values)[::-1]
    order = order[:k]
    return np.asarray(values[order], dtype=float), np.asarray(vectors[:, order], dtype=float)


def _symmetric_bottom_eigenpairs(matrix: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    sym = _symmetrized_matrix(matrix)
    n = sym.shape[0]
    k = max(0, min(int(count), n))
    if k <= 0:
        return np.zeros(0, dtype=float), np.zeros((n, 0), dtype=float)
    try:
        values, vectors = dense_eigh(sym, subset_by_index=[0, k - 1], check_finite=False)
    except Exception:
        values, vectors = np.linalg.eigh(sym)
    order = np.argsort(values)[:k]
    return np.asarray(values[order], dtype=float), np.asarray(vectors[:, order], dtype=float)


def _symmetrized_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return np.ascontiguousarray((matrix + matrix.T) * 0.5)


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


def _strict_neutral_theory_alignment_report(
    neutral_views: list[NeutralObserverView],
    weights: dict[str, float] | None = None,
    *,
    observer_views: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    active_weights = weights or DEFAULT_NEUTRAL_WEIGHTS
    rows: list[dict[str, Any]] = []
    gaps: list[str] = []
    observer_count = len(neutral_views)
    patch_rows = [
        view for view in (observer_views or []) if view.get("view_type") == "patch_observer"
    ]
    for channel, evidence in STRICT_NEUTRAL_THEORY_REQUIRED_CHANNELS.items():
        matrix = _neutral_channel_matrix(neutral_views, channel)
        if matrix.ndim != 2 or matrix.shape[0] != observer_count:
            matrix = np.zeros((observer_count, 0), dtype=float)
        if matrix.shape[1]:
            finite_matrix = np.where(np.isfinite(matrix), matrix, 0.0)
            row_observed = np.linalg.norm(finite_matrix, axis=1) > 1e-12
        else:
            row_observed = np.zeros(observer_count, dtype=bool)
        observed_count = int(np.sum(row_observed))
        producer_verified_count = _verified_neutral_channel_producer_count(channel, patch_rows)
        producer_required = channel in {
            "local_packet",
            "boundary_packet",
            "overlap_correspondence",
            "perturbation_response_tensor",
        }
        producer_verified = bool(
            not producer_required
            or (observer_count > 0 and producer_verified_count == observer_count)
        )
        active = float(active_weights.get(channel, 0.0) or 0.0) > 0.0
        present = bool(
            observer_count > 0
            and observed_count == observer_count
            and active
            and producer_verified
        )
        if not present:
            if not active:
                reason = "inactive_required_neutral_channel"
            elif not producer_verified:
                reason = "unverified_or_legacy_required_neutral_channel_producer"
            else:
                reason = "missing_or_partial_required_neutral_channel"
            gaps.append(f"{reason}:{channel}:{observed_count}/{observer_count}")
        rows.append(
            {
                "channel": channel,
                "evidence_requirement": evidence,
                "active_weight": float(active_weights.get(channel, 0.0) or 0.0),
                "observer_rows_with_signal": observed_count,
                "observer_rows_with_verified_producer": producer_verified_count,
                "producer_provenance_required": producer_required,
                "producer_provenance_verified": producer_verified,
                "observer_count": observer_count,
                "present_for_all_observers": present,
            }
        )
    return {
        "mode": "strict_neutral_theory_required_channel_alignment_v0",
        "theory_required_channels_present": not gaps,
        "required_channels": rows,
        "evidence_gaps": gaps,
        "claim_boundary": (
            "H2 strict-neutral bulk needs chart-blind local observer evidence: local boundary/port packet "
            "hashes, overlap correspondences, record/checkpoint order, transition counts by port pair and "
            "lag, perturbation/repair response tensors, first-passage observables, and quotient-safe controls. "
            "A geometric fit is diagnostic until these channels and the quotient geometry contract are present."
        ),
    }


def _verified_neutral_channel_producer_count(
    channel: str,
    patch_rows: list[dict[str, Any]],
) -> int:
    if channel == "local_packet":
        return int(
            sum(
                1
                for row in patch_rows
                if isinstance(row.get("locality_preserving_packet_feature_schema"), dict)
                and row["locality_preserving_packet_feature_schema"].get("support_selection_carrier")
                == "finite_patch_adjacency_bfs"
                and row["locality_preserving_packet_feature_schema"].get("strict_neutral_eligible") is True
                and set(row["locality_preserving_packet_feature_schema"].get("fields", []))
                <= {"repair_load", "cumulative_repair_load"}
            )
        )
    if channel == "overlap_correspondence":
        return int(
            sum(
                1
                for row in patch_rows
                if isinstance(row.get("measured_overlap_correspondences"), list)
                and isinstance(row.get("overlap_correspondence_evidence_provenance"), dict)
                and row["overlap_correspondence_evidence_provenance"].get("cross_observer_measurement") is True
                and row["overlap_correspondence_evidence_provenance"].get("self_histogram_synthesis") is False
                and row["overlap_correspondence_evidence_provenance"].get("support_selection_carrier")
                == "finite_patch_adjacency_bfs"
                and row["overlap_correspondence_evidence_provenance"].get("strict_neutral_eligible") is True
            )
        )
    if channel == "perturbation_response_tensor":
        return int(
            sum(
                1
                for row in patch_rows
                if row.get("paired_perturbation_response_producer_receipt") is True
                and isinstance(row.get("paired_perturbation_response_tensor"), list)
                and isinstance(row.get("paired_perturbation_response_provenance"), dict)
                and row["paired_perturbation_response_provenance"].get("strict_neutral_eligible") is True
            )
        )
    if channel == "boundary_packet":
        return int(
            sum(
                1
                for row in patch_rows
                if isinstance(row.get("boundary_packet_feature_provenance"), dict)
                and row["boundary_packet_feature_provenance"].get("locality_preserving") is True
                and row["boundary_packet_feature_provenance"].get("hash_bucket_geometry") is False
            )
        )
    return len(patch_rows)


def _first_histogram_vector(view: dict[str, Any], keys: tuple[str, ...], width: int) -> np.ndarray:
    for key in keys:
        vector = _histogram_like_to_vector(view.get(key), width)
        if float(np.sum(vector)) > 1e-12:
            return vector
    histograms = view.get("transition_history_histograms")
    if isinstance(histograms, dict):
        for key in keys:
            vector = _histogram_like_to_vector(histograms.get(key), width)
            if float(np.sum(vector)) > 1e-12:
                return vector
    descriptor = view.get("transition_history_descriptor")
    if isinstance(descriptor, dict):
        for key in keys:
            vector = _histogram_like_to_vector(descriptor.get(key), width)
            if float(np.sum(vector)) > 1e-12:
                return vector
    return np.zeros(max(1, int(width)), dtype=float)


def _measured_overlap_summary_vector(view: dict[str, Any], *, width: int) -> np.ndarray:
    correspondences = view.get("measured_overlap_correspondences")
    provenance = view.get("overlap_correspondence_evidence_provenance")
    measured = bool(
        isinstance(correspondences, list)
        and isinstance(provenance, dict)
        and (
            (
                provenance.get("cross_observer_measurement") is True
                and provenance.get("self_histogram_synthesis") is False
            )
            or provenance.get("synthetic_negative_control") is True
        )
    )
    if measured:
        blocks: list[np.ndarray] = []
        for key in (
            "jaccard",
            "measured_affinity",
            "local_packet_similarity",
            "support_fraction_self",
            "support_fraction_peer",
        ):
            values = []
            for row in correspondences:
                if not isinstance(row, dict):
                    continue
                try:
                    value = float(row.get(key))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    values.append(float(np.clip(value, 0.0, 1.0)))
            if not values:
                blocks.append(np.zeros(20, dtype=float))
                continue
            array = np.asarray(values, dtype=float)
            histogram, _ = np.histogram(array, bins=16, range=(0.0, 1.0))
            histogram = histogram.astype(float) / max(float(histogram.sum()), 1.0)
            moments = np.asarray(
                [
                    float(np.mean(array)),
                    float(np.std(array)),
                    float(np.quantile(array, 0.25)),
                    float(np.quantile(array, 0.75)),
                ],
                dtype=float,
            )
            blocks.append(np.concatenate([histogram, moments]))
        return _pad(np.concatenate(blocks), int(width))[: int(width)]
    # Legacy self-histogram summaries remain in the raw evidence rows, but do
    # not enter the measured-overlap geometry channel.
    return np.zeros(max(1, int(width)), dtype=float)


def _first_signed_vector(view: dict[str, Any], keys: tuple[str, ...], width: int) -> np.ndarray:
    for key in keys:
        vector = _signed_observable_vector(view.get(key), width)
        if float(np.linalg.norm(vector)) > 1e-12:
            return vector
    tensors = view.get("transition_history_tensors")
    if isinstance(tensors, dict):
        for key in keys:
            vector = _signed_observable_vector(tensors.get(key), width)
            if float(np.linalg.norm(vector)) > 1e-12:
                return vector
    descriptor = view.get("transition_history_descriptor")
    if isinstance(descriptor, dict):
        for key in keys:
            vector = _signed_observable_vector(descriptor.get(key), width)
            if float(np.linalg.norm(vector)) > 1e-12:
                return vector
    return np.zeros(max(1, int(width)), dtype=float)


def _verified_perturbation_response_vector(view: dict[str, Any], *, width: int) -> np.ndarray:
    if (
        "paired_perturbation_response_tensor" in view
        and view.get("paired_perturbation_response_producer_receipt") is True
    ):
        return _signed_observable_vector(view.get("paired_perturbation_response_tensor"), width)
    # Do not silently substitute report-backed/synthetic response summaries for
    # the actual paired producer in the default geometry.
    return np.zeros(max(1, int(width)), dtype=float)


def _port_pair_lag_vector(view: dict[str, Any], steps: list[Any], *, width: int) -> np.ndarray:
    explicit = _first_histogram_vector(
        view,
        (
            "port_pair_lag_histogram",
            "transition_port_pair_lag_histogram",
            "local_port_pair_lag_histogram",
            "port_pair_lag_events",
        ),
        width,
    )
    if float(np.sum(explicit)) > 1e-12:
        return explicit
    vector = np.zeros(max(1, int(width)), dtype=float)
    for step in steps:
        if not isinstance(step, dict):
            continue
        token = _port_pair_lag_token(step)
        if token is not None:
            vector[_stable_bucket(token, vector.size)] += 1.0
    return _normalize_or_zero(vector, width=vector.size)


def _port_pair_lag_token(step: dict[str, Any]) -> str | None:
    pair = step.get("port_pair") or step.get("local_port_pair") or step.get("transition_port_pair")
    if pair is None:
        left = step.get("port_left", step.get("source_port", step.get("port_i")))
        right = step.get("port_right", step.get("target_port", step.get("port_j")))
        if left is not None and right is not None:
            pair = (left, right)
    lag = step.get("lag", step.get("transition_lag", step.get("record_lag", step.get("response_lag"))))
    if pair is None and lag is None:
        return None
    return f"pair={_canonical_observable_token(pair)}|lag={_canonical_observable_token(lag)}"


def _histogram_like_to_vector(value: Any, width: int) -> np.ndarray:
    vector = np.zeros(max(1, int(width)), dtype=float)
    if value is None:
        return vector
    if isinstance(value, dict):
        for token, mass in _iter_observable_histogram_items(value):
            bucket = _safe_int(token)
            index = int(bucket) % vector.size if bucket is not None else _stable_bucket(str(token), vector.size)
            if np.isfinite(mass):
                vector[index] += max(float(mass), 0.0)
        return _normalize_or_zero(vector, width=vector.size)
    if isinstance(value, (list, tuple)):
        try:
            numeric = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            numeric = np.zeros(0, dtype=float)
        if numeric.size and np.all(np.isfinite(numeric)):
            return _normalize_or_zero(numeric, width=vector.size)
        for item in value:
            vector[_stable_bucket(_canonical_observable_token(item), vector.size)] += 1.0
        return _normalize_or_zero(vector, width=vector.size)
    vector[_stable_bucket(_canonical_observable_token(value), vector.size)] += 1.0
    return _normalize_or_zero(vector, width=vector.size)


def _signed_observable_vector(value: Any, width: int) -> np.ndarray:
    vector = np.zeros(max(1, int(width)), dtype=float)
    if value is None:
        return vector
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            array = np.zeros(0, dtype=float)
        if array.size:
            padded = _pad(array, vector.size)[: vector.size]
            return np.where(np.isfinite(padded), padded, 0.0)
    if isinstance(value, dict):
        for token, mass in _iter_observable_histogram_items(value):
            if np.isfinite(mass):
                vector[_stable_bucket(str(token), vector.size)] += float(mass)
        return np.where(np.isfinite(vector), vector, 0.0)
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        vector[_stable_bucket(_canonical_observable_token(value), vector.size)] += 1.0
        return vector
    if np.isfinite(scalar):
        vector[0] = scalar
    return vector


def _iter_observable_histogram_items(value: dict[str, Any], prefix: str = ""):
    for key, mass in value.items():
        token = f"{prefix}:{key}" if prefix else str(key)
        if isinstance(mass, dict):
            yield from _iter_observable_histogram_items(mass, token)
            continue
        try:
            numeric_mass = float(mass)
        except (TypeError, ValueError):
            numeric_mass = 1.0
            token = f"{token}:{_canonical_observable_token(mass)}"
        yield token, numeric_mass


def _canonical_observable_token(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(value)


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
    log_radii = np.log(radii[mask])
    log_counts = np.log(counts[mask])
    if (
        not np.all(np.isfinite(log_radii))
        or not np.all(np.isfinite(log_counts))
        or float(np.std(log_radii)) < 1e-12
        or float(np.std(log_counts)) < 1e-12
    ):
        return {"estimate": None, "points_used": int(np.sum(mask))}
    try:
        estimate = float(np.polyfit(log_radii, log_counts, 1)[0])
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return {"estimate": None, "points_used": int(np.sum(mask))}
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
