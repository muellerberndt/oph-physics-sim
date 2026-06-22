from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


ALLOWED_METRIC_MODES = {"complete_case", "fixed_missing_symbol", "train_only_imputation"}
FORBIDDEN_METRIC_MODES = {"pairwise_available_channels"}
GEOMETRY_CONTRACT_RECEIPT = "QUOTIENT_GEOMETRY_CONTRACT_RECEIPT"


@dataclass(frozen=True)
class ChannelMetricSpec:
    name: str
    version: str = "v1"
    metric: str = "euclidean"
    weight: float = 1.0
    missingness: str = "complete_case"
    units: str = "dimensionless"
    physical_status: str = "quotient_visible"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("channel name is required")
        if not self.version:
            raise ValueError("channel version is required")
        if not math.isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise ValueError("channel weight must be positive")
        if self.missingness in FORBIDDEN_METRIC_MODES:
            raise ValueError(f"forbidden missingness policy: {self.missingness}")
        if self.missingness not in ALLOWED_METRIC_MODES:
            raise ValueError(f"unknown missingness policy: {self.missingness}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "metric": self.metric,
            "weight": float(self.weight),
            "missingness": self.missingness,
            "units": self.units,
            "physical_status": self.physical_status,
        }


@dataclass(frozen=True)
class ProvenanceRecord:
    record_id: str
    split: str
    batch_id: str
    seed_id: str
    boundary_condition_id: str
    trajectory_family_id: str
    parent_record_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("record_id is required")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split: {self.split}")
        for name in ("batch_id", "seed_id", "boundary_condition_id", "trajectory_family_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        object.__setattr__(self, "parent_record_ids", tuple(str(value) for value in self.parent_record_ids))

    @property
    def group_ids(self) -> tuple[str, ...]:
        return (
            f"batch:{self.batch_id}",
            f"seed:{self.seed_id}",
            f"boundary:{self.boundary_condition_id}",
            f"trajectory:{self.trajectory_family_id}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "split": self.split,
            "batch_id": self.batch_id,
            "seed_id": self.seed_id,
            "boundary_condition_id": self.boundary_condition_id,
            "trajectory_family_id": self.trajectory_family_id,
            "parent_record_ids": list(self.parent_record_ids),
        }


def metric_validity_report(
    distance: np.ndarray,
    *,
    quotient_ids: list[str] | None = None,
    metric_mode: str = "complete_case",
    jointly_separating: bool = False,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    if metric_mode in FORBIDDEN_METRIC_MODES:
        return {
            "mode": metric_mode,
            "valid_pseudometric": False,
            "valid_metric": False,
            "forbidden_metric_mode": True,
            "blockers": ["pairwise_available_channels_forbidden"],
        }
    if metric_mode not in ALLOWED_METRIC_MODES:
        return {
            "mode": metric_mode,
            "valid_pseudometric": False,
            "valid_metric": False,
            "forbidden_metric_mode": False,
            "blockers": ["unknown_metric_mode"],
        }
    d = np.asarray(distance, dtype=float)
    blockers: list[str] = []
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        return {
            "mode": metric_mode,
            "valid_pseudometric": False,
            "valid_metric": False,
            "blockers": ["distance_matrix_not_square"],
        }
    n = int(d.shape[0])
    if quotient_ids is not None and len(quotient_ids) != n:
        blockers.append("quotient_id_count_mismatch")
    if not np.all(np.isfinite(d)):
        blockers.append("distance_matrix_nonfinite")
    min_distance = float(np.min(d)) if d.size else 0.0
    if min_distance < -float(tolerance):
        blockers.append("negative_distance")
    symmetry_defect = float(np.max(np.abs(d - d.T))) if d.size else 0.0
    diagonal_defect = float(np.max(np.abs(np.diag(d)))) if d.size else 0.0
    if symmetry_defect > float(tolerance):
        blockers.append("distance_matrix_not_symmetric")
    if diagonal_defect > float(tolerance):
        blockers.append("distance_diagonal_not_zero")
    triangle = _triangle_violation(d)
    if triangle["max_violation"] > float(tolerance):
        blockers.append("triangle_violation")
    collision_summary = _zero_distance_pair_summary(d, quotient_ids=quotient_ids, tolerance=tolerance)
    pseudometric = not any(
        blocker
        in {
            "distance_matrix_not_square",
            "distance_matrix_nonfinite",
            "negative_distance",
            "distance_matrix_not_symmetric",
            "distance_diagonal_not_zero",
            "triangle_violation",
        }
        for blocker in blockers
    )
    metric = bool(pseudometric and (jointly_separating or collision_summary["count"] == 0))
    metric_blockers = []
    if pseudometric and not metric:
        metric_blockers.append("zero_distance_feature_collisions")
    return {
        "mode": metric_mode,
        "observer_count": n,
        "valid_pseudometric": bool(pseudometric),
        "valid_metric": bool(metric),
        "jointly_separating": bool(jointly_separating),
        "symmetry_defect": symmetry_defect,
        "diagonal_defect": diagonal_defect,
        "triangle_max_violation": triangle["max_violation"],
        "triangle_checked_exact": triangle["checked_exact"],
        "zero_distance_collision_count": collision_summary["count"],
        "zero_distance_collision_pairs": collision_summary["pairs"],
        "blockers": blockers,
        "metric_blockers": metric_blockers,
    }


def euclidean_distance_certificate(distance: np.ndarray, *, tolerance: float = 1.0e-9) -> dict[str, Any]:
    d = np.asarray(distance, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        return {
            "euclidean_realizable": False,
            "blockers": ["distance_matrix_not_square"],
        }
    if d.shape[0] == 0:
        return {
            "euclidean_realizable": True,
            "exact_rank": 0,
            "negative_eigenvalue_mass": 0.0,
            "eigenvalues": [],
            "blockers": [],
        }
    squared = np.square(np.maximum(d, 0.0))
    n = d.shape[0]
    h = np.eye(n) - np.ones((n, n), dtype=float) / float(n)
    gram = -0.5 * h @ squared @ h
    gram = 0.5 * (gram + gram.T)
    eigenvalues = np.linalg.eigvalsh(gram)
    total_mass = float(np.sum(np.abs(eigenvalues)))
    negative_mass = float(np.sum(np.abs(eigenvalues[eigenvalues < -float(tolerance)])))
    negative_ratio = negative_mass / total_mass if total_mass > 0.0 else 0.0
    rank = int(np.sum(eigenvalues > float(tolerance)))
    realizable = bool(negative_mass <= float(tolerance) * max(1, n))
    blockers = [] if realizable else ["gram_matrix_has_negative_spectrum"]
    return {
        "euclidean_realizable": realizable,
        "exact_rank": rank if realizable else None,
        "positive_rank": rank,
        "negative_eigenvalue_mass": negative_ratio,
        "negative_eigenvalue_absolute_mass": negative_mass,
        "min_eigenvalue": float(eigenvalues[0]),
        "max_eigenvalue": float(eigenvalues[-1]),
        "eigenvalues": [float(value) for value in eigenvalues],
        "blockers": blockers,
    }


def ancestry_split_report(records: list[ProvenanceRecord]) -> dict[str, Any]:
    by_group: dict[str, set[str]] = defaultdict(set)
    by_record = {record.record_id: record for record in records}
    parent_edges: list[tuple[str, str]] = []
    missing_parents: list[str] = []
    for record in records:
        for group_id in record.group_ids:
            by_group[group_id].add(record.split)
        for parent_id in record.parent_record_ids:
            if parent_id in by_record:
                parent_edges.append((record.record_id, parent_id))
            else:
                missing_parents.append(parent_id)
    split_leaking_groups = {
        group_id: sorted(splits)
        for group_id, splits in by_group.items()
        if len(splits) > 1
    }
    component_leaks = _parent_component_split_leaks(records, parent_edges)
    leakage_count = len(split_leaking_groups) + len(component_leaks)
    blockers = []
    if split_leaking_groups:
        blockers.append("generative_group_crosses_splits")
    if component_leaks:
        blockers.append("ancestry_component_crosses_splits")
    if missing_parents:
        blockers.append("missing_parent_records")
    return {
        "split_unit": "shard_batch",
        "record_count": len(records),
        "ancestry_leakage_count": leakage_count,
        "split_leaking_groups": split_leaking_groups,
        "component_split_leaks": component_leaks,
        "missing_parent_record_ids": sorted(set(missing_parents)),
        "blockers": blockers,
    }


def quotient_geometry_certificate(
    distance: np.ndarray,
    *,
    quotient_ids: list[str] | None = None,
    channel_manifest: list[ChannelMetricSpec] | list[dict[str, Any]] | None = None,
    metric_mode: str = "complete_case",
    jointly_separating: bool = False,
    missingness_quotient_visible: bool = True,
    atlas_receipt: dict[str, Any] | None = None,
    feature_receipt: dict[str, Any] | None = None,
    invariance_receipt: dict[str, Any] | None = None,
    refinement_receipt: dict[str, Any] | None = None,
    statistics_receipt: dict[str, Any] | None = None,
    provenance_records: list[ProvenanceRecord] | None = None,
    require_metric: bool = False,
    require_euclidean: bool = False,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    metric = metric_validity_report(
        distance,
        quotient_ids=quotient_ids,
        metric_mode=metric_mode,
        jointly_separating=jointly_separating,
        tolerance=tolerance,
    )
    euclidean = euclidean_distance_certificate(distance, tolerance=tolerance)
    ancestry = ancestry_split_report(provenance_records or [])
    atlas = atlas_receipt or {}
    features = feature_receipt or {}
    invariance = invariance_receipt or {}
    refinement = refinement_receipt or {}
    statistics = statistics_receipt or {}

    blockers = list(metric.get("blockers", []))
    if not metric.get("valid_pseudometric", False):
        blockers.append("neutral_distance_not_pseudometric")
    if require_metric and not metric.get("valid_metric", False):
        blockers.extend(str(blocker) for blocker in metric.get("metric_blockers", []))
    if metric_mode == "fixed_missing_symbol" and not missingness_quotient_visible:
        blockers.append("missingness_mask_not_quotient_visible")
    if channel_manifest is None:
        blockers.append("channel_manifest_missing")
    if not _defects_clear(
        atlas,
        ("identity_defect", "inverse_defect", "cocycle_defect", "cycle_holonomy_defect"),
        tolerance,
    ):
        blockers.append("atlas_transport_defects_unproven")
    if not bool(features.get("quotient_visible_missingness", missingness_quotient_visible)):
        blockers.append("feature_missingness_not_quotient_visible")
    if float(features.get("max_transport_defect", 0.0) or 0.0) > float(tolerance):
        blockers.append("feature_transport_defect_positive")
    if not _invariance_clear(invariance, tolerance):
        blockers.append("presentation_invariance_not_certified")
    if refinement and not bool(refinement.get("convergent", refinement.get("stable_across_64k_256k_1m", False))):
        blockers.append("refinement_tail_modulus_not_certified")
    if require_euclidean and not euclidean.get("euclidean_realizable", False):
        blockers.extend(str(blocker) for blocker in euclidean.get("blockers", []))
    if provenance_records and ancestry["ancestry_leakage_count"] > 0:
        blockers.extend(ancestry["blockers"])
    if statistics:
        if int(statistics.get("ancestry_leakage_count", 0) or 0) > 0:
            blockers.append("statistical_ancestry_leakage")
        if statistics.get("test_used_once") is False:
            blockers.append("test_set_reused")
        if statistics.get("positive_controls_passed") is False:
            blockers.append("positive_controls_failed")
        if statistics.get("negative_controls_passed") is False:
            blockers.append("negative_controls_failed")

    blockers = _unique(blockers)
    certified = not blockers
    exact = certified and _all_exact_zero(atlas, features, invariance, tolerance)
    status = "EXACT" if exact else "APPROX_CERTIFIED" if certified else "DIAGNOSTIC_ONLY"
    return {
        "mode": "quotient_geometry_contract_v0",
        "status": status,
        GEOMETRY_CONTRACT_RECEIPT: certified,
        "bulk_promotion_allowed": certified,
        "physical_claim": False,
        "quotient_id_count": len(quotient_ids or []),
        "channel_manifest": _channel_manifest_to_dict(channel_manifest),
        "metric": metric,
        "euclidean": euclidean,
        "atlas": atlas,
        "features": features,
        "invariance": invariance,
        "refinement": refinement,
        "statistics": {**statistics, "ancestry": ancestry},
        "blockers": blockers,
        "claim_boundary": (
            "Quotient-visible geometry contract. It certifies a finite quotient pseudometric or metric "
            "only after atlas transport, feature descent, missingness, presentation invariance, refinement, "
            "and split-leakage gates are explicit. It does not by itself identify the metric with physical "
            "Riemannian or Lorentzian spacetime."
        ),
    }


def _triangle_violation(distance: np.ndarray) -> dict[str, Any]:
    n = int(distance.shape[0])
    if n <= 512:
        max_violation = 0.0
        for k in range(n):
            candidate = distance - (distance[:, [k]] + distance[[k], :])
            max_violation = max(max_violation, float(np.max(candidate)))
        return {"max_violation": max(0.0, max_violation), "checked_exact": True}
    sample = np.linspace(0, n - 1, 512, dtype=int)
    work = distance[np.ix_(sample, sample)]
    return {**_triangle_violation(work), "checked_exact": False}


def _zero_distance_pair_summary(
    distance: np.ndarray,
    *,
    quotient_ids: list[str] | None,
    tolerance: float,
) -> dict[str, Any]:
    pairs: list[tuple[str, str]] = []
    ids = quotient_ids or [str(index) for index in range(distance.shape[0])]
    count = 0
    limit = 64
    threshold = float(tolerance)
    for i in range(max(0, distance.shape[0] - 1)):
        candidates = np.flatnonzero(np.abs(distance[i, i + 1 :]) <= threshold) + i + 1
        if candidates.size == 0:
            continue
        for j in candidates:
            if ids[i] == ids[int(j)]:
                continue
            count += 1
            if len(pairs) < limit:
                pairs.append((ids[i], ids[int(j)]))
    return {"count": count, "pairs": pairs}


def _zero_distance_pairs(
    distance: np.ndarray,
    *,
    quotient_ids: list[str] | None,
    tolerance: float,
) -> list[tuple[str, str]]:
    return list(_zero_distance_pair_summary(distance, quotient_ids=quotient_ids, tolerance=tolerance)["pairs"])


def _parent_component_split_leaks(
    records: list[ProvenanceRecord],
    edges: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    parent = {record.record_id: record.record_id for record in records}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    components: dict[str, list[ProvenanceRecord]] = defaultdict(list)
    for record in records:
        components[find(record.record_id)].append(record)
    leaks = []
    for component in components.values():
        splits = sorted({record.split for record in component})
        if len(splits) > 1:
            leaks.append(
                {
                    "splits": splits,
                    "record_ids": sorted(record.record_id for record in component)[:128],
                }
            )
    return leaks


def _defects_clear(receipt: dict[str, Any], keys: tuple[str, ...], tolerance: float) -> bool:
    return bool(receipt) and all(abs(float(receipt.get(key, float("inf")))) <= float(tolerance) for key in keys)


def _invariance_clear(receipt: dict[str, Any], tolerance: float) -> bool:
    if not receipt:
        return False
    keys = (
        "gauge_distortion",
        "port_distortion",
        "order_distortion",
        "schedule_distortion",
        "partition_distortion",
    )
    return all(abs(float(receipt.get(key, float("inf")))) <= float(tolerance) for key in keys)


def _all_exact_zero(
    atlas: dict[str, Any],
    features: dict[str, Any],
    invariance: dict[str, Any],
    tolerance: float,
) -> bool:
    atlas_exact = _defects_clear(
        atlas,
        ("identity_defect", "inverse_defect", "cocycle_defect", "cycle_holonomy_defect"),
        tolerance,
    )
    feature_exact = abs(float(features.get("max_transport_defect", float("inf")))) <= float(tolerance)
    invariance_exact = _invariance_clear(invariance, tolerance)
    return bool(atlas_exact and feature_exact and invariance_exact)


def _channel_manifest_to_dict(
    channel_manifest: list[ChannelMetricSpec] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if channel_manifest is None:
        return []
    rows = []
    for row in channel_manifest:
        if isinstance(row, ChannelMetricSpec):
            rows.append(row.to_dict())
        else:
            rows.append(dict(row))
    return rows


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
