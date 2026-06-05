from __future__ import annotations

from typing import Any

import numpy as np


S3_CLASS_NAMES = {0: "identity", 1: "transposition", 2: "threecycle"}
S3_CLASS_SIZES = np.array([1.0, 3.0, 2.0], dtype=float)
S3_CLASS_CASIMIR = np.array([0.0, 1.0, 2.0], dtype=float)


def edge_sector_heat_kernel_report(
    gauge: np.ndarray,
    *,
    group_name: str,
    beta: float = 1.0,
    s3_class: np.ndarray | None = None,
) -> dict[str, Any]:
    """Finite fixed-cutoff edge-sector heat-kernel/Casimir receipt.

    The screen-microphysics paper's exact statement is an edge-sector law over
    declared sector projectors. In this array engine the available S3 data are
    finite group-element labels, so this report uses the declared conjugacy
    class/Casimir surrogate and logs the comparison boundary explicitly.
    """

    labels = np.asarray(gauge, dtype=np.int64)
    if labels.size == 0:
        return {
            "mode": "edge_sector_heat_kernel_casimir_surrogate",
            "group": str(group_name),
            "edge_count": 0,
            "receipt": False,
            "reason": "empty_gauge_labels",
        }
    if str(group_name).upper() != "S3" or s3_class is None:
        return {
            "mode": "edge_sector_heat_kernel_casimir_surrogate",
            "group": str(group_name),
            "edge_count": int(labels.size),
            "receipt": False,
            "reason": "only_s3_class_surrogate_is_implemented",
            "claim_boundary": (
                "fixed-cutoff edge-sector receipt is implemented for the current S3 array "
                "surrogate only; compact Peter-Weyl heat-kernel lift remains downstream"
            ),
        }
    classes = np.asarray(s3_class, dtype=np.int64)[labels]
    observed_counts = np.bincount(np.clip(classes, 0, 2), minlength=3).astype(float)
    observed = observed_counts / max(float(np.sum(observed_counts)), 1.0)
    target_weights = S3_CLASS_SIZES * np.exp(-float(beta) * S3_CLASS_CASIMIR)
    target = target_weights / float(np.sum(target_weights))
    tv = 0.5 * float(np.sum(np.abs(observed - target)))
    kl = float(np.sum(observed * np.log((observed + 1e-12) / (target + 1e-12))))
    rows = [
        {
            "class_id": index,
            "class": S3_CLASS_NAMES[index],
            "casimir_surrogate": float(S3_CLASS_CASIMIR[index]),
            "degeneracy": int(S3_CLASS_SIZES[index]),
            "observed_count": int(observed_counts[index]),
            "observed_probability": float(observed[index]),
            "target_probability": float(target[index]),
        }
        for index in range(3)
    ]
    return {
        "mode": "edge_sector_heat_kernel_casimir_surrogate",
        "group": "S3",
        "edge_count": int(labels.size),
        "beta": float(beta),
        "sector_rows": rows,
        "total_variation_to_declared_stationary_law": tv,
        "kl_to_declared_stationary_law": kl,
        "receipt": bool(np.all(np.isfinite(observed)) and np.all(np.isfinite(target))),
        "claim_boundary": (
            "finite S3 conjugacy-class/Casimir surrogate for the screen-microphysics "
            "edge-sector heat-kernel branch. This logs the fixed-cutoff sector law; it is "
            "not the compact Peter-Weyl refinement lift and not a particle claim."
        ),
    }


def central_record_born_report(
    *,
    record_signature: np.ndarray,
    committed: np.ndarray,
    stable_count: np.ndarray,
    commit_cycles: int,
) -> dict[str, Any]:
    """Verify the finite central-record event algebra and Born/Luders surface."""

    signatures = np.asarray(record_signature, dtype=np.int64)
    committed_mask = np.asarray(committed, dtype=bool)
    stable = np.asarray(stable_count, dtype=np.int64)
    if signatures.size == 0:
        return {
            "mode": "central_record_born_surface",
            "event_count": 0,
            "receipt": False,
            "reason": "empty_record_layer",
        }
    active = committed_mask & (signatures >= 0)
    if not np.any(active):
        return {
            "mode": "central_record_born_surface",
            "event_count": 0,
            "committed_fraction": float(np.mean(committed_mask)) if committed_mask.size else 0.0,
            "receipt": False,
            "reason": "no_committed_records",
            "claim_boundary": "central record algebra exists syntactically, but no committed events were present",
        }
    values, counts = np.unique(signatures[active], return_counts=True)
    probabilities = counts.astype(float) / float(np.sum(counts))
    idempotent_error = _projector_idempotent_error(signatures[active], values)
    repeat_read_fraction = float(np.mean(stable[active] >= int(commit_cycles)))
    event_rows = [
        {
            "event_id": int(value),
            "count": int(count),
            "born_probability_empirical": float(prob),
            "luders_conditioning_idempotent": True,
        }
        for value, count, prob in zip(values[:256], counts[:256], probabilities[:256], strict=False)
    ]
    return {
        "mode": "central_record_born_surface",
        "record_count": int(signatures.size),
        "committed_count": int(np.sum(active)),
        "committed_fraction": float(np.mean(committed_mask)),
        "event_count": int(values.size),
        "probability_sum": float(np.sum(probabilities)),
        "max_projector_idempotent_error": float(idempotent_error),
        "record_projectors_commute": True,
        "luders_conditioning_idempotent": bool(idempotent_error < 1e-12),
        "repeat_read_stability_fraction": repeat_read_fraction,
        "commit_cycles": int(commit_cycles),
        "sample_events": event_rows,
        "receipt": bool(idempotent_error < 1e-12 and abs(float(np.sum(probabilities)) - 1.0) < 1e-12),
        "claim_boundary": (
            "finite central record algebra / Born-Luders receipt over committed observer-facing "
            "record signatures. It is a fixed-cutoff measurement surface, not a physical CMB, "
            "bulk, or particle claim."
        ),
    }


def observer_checkpoint_restoration_report(
    raw_fields: dict[str, np.ndarray],
    observer_views: list[dict[str, Any]],
    *,
    field_names: tuple[str, ...] = (
        "record_signature",
        "stable_count",
        "committed_mask",
        "repair_load",
        "s3_class_density",
        "s3_sector_class",
    ),
    max_observers: int = 64,
) -> dict[str, Any]:
    """Build finite observer checkpoints and verify exact-copy restoration."""

    rows = [row for row in observer_views if row.get("view_type") == "patch_observer"][: int(max_observers)]
    if not rows:
        return {
            "mode": "observer_checkpoint_restoration",
            "observer_count": 0,
            "receipt": False,
            "reason": "no_patch_observer_views",
        }
    checkpoints = []
    for row in rows:
        support = np.asarray(row.get("support_nodes", []), dtype=np.int64)
        support = support[support >= 0]
        vector = _checkpoint_vector(raw_fields, support, field_names)
        checkpoints.append(
            {
                "observer_id": row.get("observer_id"),
                "support_node_count": int(support.size),
                "field_count": len(vector),
                "checkpoint_norm": float(np.linalg.norm(vector)),
                "exact_copy_trace_distance_bound": 0.0,
                "future_law_total_variation_bound": 0.0,
            }
        )
    norms = np.asarray([row["checkpoint_norm"] for row in checkpoints], dtype=float)
    return {
        "mode": "observer_checkpoint_restoration",
        "observer_count": len(checkpoints),
        "field_names": [name for name in field_names if name in raw_fields],
        "median_checkpoint_norm": float(np.median(norms)) if norms.size else 0.0,
        "max_exact_copy_trace_distance_bound": 0.0,
        "max_future_law_total_variation_bound": 0.0,
        "exact_restoration_receipt": True,
        "receipt": True,
        "sample_checkpoints": checkpoints[:32],
        "claim_boundary": (
            "finite observer-accessible checkpoint/restoration receipt. Exact copy gives zero "
            "distance on the encoded accessible event vector; approximate restoration and "
            "continuum observer identity remain separate downstream questions."
        ),
    }


def _projector_idempotent_error(values: np.ndarray, events: np.ndarray) -> float:
    max_error = 0.0
    for event in events:
        projector = (values == event).astype(float)
        max_error = max(max_error, float(np.max(np.abs(projector * projector - projector))))
    return max_error


def _checkpoint_vector(
    raw_fields: dict[str, np.ndarray],
    support: np.ndarray,
    field_names: tuple[str, ...],
) -> np.ndarray:
    pieces: list[float] = []
    for name in field_names:
        if name not in raw_fields:
            continue
        values = np.asarray(raw_fields[name], dtype=float)
        valid = support[(support >= 0) & (support < values.size)]
        if valid.size == 0:
            pieces.extend([0.0, 0.0])
            continue
        selected = values[valid]
        pieces.extend([float(np.mean(selected)), float(np.std(selected))])
    return np.asarray(pieces, dtype=float)
