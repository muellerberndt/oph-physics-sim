from __future__ import annotations

from typing import Any

import numpy as np


S3_CLASS_NAMES = {0: "identity", 1: "transposition", 2: "threecycle"}
S3_CLASS_SIZES = np.array([1.0, 3.0, 2.0], dtype=float)
S3_CLASS_CASIMIR = np.array([0.0, 1.0, 2.0], dtype=float)


def _edge_sector_failure(
    *, group_name: str, edge_count: int, reason: str
) -> dict[str, Any]:
    return {
        "mode": "edge_sector_heat_kernel_casimir_surrogate",
        "group": str(group_name),
        "edge_count": int(edge_count),
        "edge_sector_diagnostic_receipt": False,
        "heat_kernel_validation_receipt": False,
        "receipt": False,
        "reason": reason,
        "claim_boundary": (
            "Malformed or unsupported finite S3 inputs cannot earn even the "
            "distribution diagnostic. No heat-kernel or physical claim follows."
        ),
    }


def edge_sector_heat_kernel_report(
    gauge: np.ndarray,
    *,
    group_name: str,
    beta: float = 1.0,
    s3_class: np.ndarray | None = None,
) -> dict[str, Any]:
    """Finite fixed-cutoff edge-sector heat-kernel/Casimir diagnostic.

    The screen-microphysics paper's exact statement is an edge-sector law over
    declared sector projectors. In this array engine the available S3 data are
    finite group-element labels, so this report uses the declared conjugacy
    class/Casimir surrogate and logs the comparison boundary explicitly.
    """

    raw_labels = np.asarray(gauge)
    if raw_labels.ndim != 1 or not np.issubdtype(raw_labels.dtype, np.integer):
        return _edge_sector_failure(
            group_name=group_name,
            edge_count=int(raw_labels.size),
            reason="gauge_labels_must_be_one_dimensional_integers",
        )
    labels = raw_labels.astype(np.int64, copy=False)
    if labels.size == 0:
        return _edge_sector_failure(
            group_name=group_name, edge_count=0, reason="empty_gauge_labels"
        )
    if str(group_name).upper() != "S3" or s3_class is None:
        return _edge_sector_failure(
            group_name=group_name,
            edge_count=int(labels.size),
            reason="only_s3_class_surrogate_is_implemented",
        )
    raw_class_map = np.asarray(s3_class)
    if raw_class_map.ndim != 1 or not np.issubdtype(
        raw_class_map.dtype, np.integer
    ):
        return _edge_sector_failure(
            group_name=group_name,
            edge_count=int(labels.size),
            reason="s3_class_map_must_be_one_dimensional_integers",
        )
    class_map = raw_class_map.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= class_map.size):
        return _edge_sector_failure(
            group_name=group_name,
            edge_count=int(labels.size),
            reason="gauge_label_out_of_s3_class_map_range",
        )
    if np.any(class_map < 0) or np.any(class_map > 2):
        return _edge_sector_failure(
            group_name=group_name,
            edge_count=int(labels.size),
            reason="s3_class_id_out_of_range",
        )
    beta_value = float(beta)
    if not np.isfinite(beta_value) or beta_value < 0.0:
        return _edge_sector_failure(
            group_name=group_name,
            edge_count=int(labels.size),
            reason="beta_must_be_finite_and_nonnegative",
        )
    classes = class_map[labels]
    observed_counts = np.bincount(classes, minlength=3).astype(float)
    observed = observed_counts / max(float(np.sum(observed_counts)), 1.0)
    target_weights = S3_CLASS_SIZES * np.exp(-beta_value * S3_CLASS_CASIMIR)
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
    diagnostic_receipt = bool(
        np.all(np.isfinite(observed)) and np.all(np.isfinite(target))
    )
    return {
        "mode": "edge_sector_heat_kernel_casimir_surrogate",
        "group": "S3",
        "edge_count": int(labels.size),
        "beta": beta_value,
        "sector_rows": rows,
        "total_variation_to_declared_stationary_law": tv,
        "kl_to_declared_stationary_law": kl,
        "edge_sector_diagnostic_receipt": diagnostic_receipt,
        "heat_kernel_validation_receipt": False,
        # The declared target and the observed sample are merely compared.
        # Finiteness of that calculation cannot validate a heat-kernel law.
        "receipt": False,
        "receipt_semantics": "finite_s3_class_distribution_diagnostic_only",
        "physical_claim": False,
        "claim_boundary": (
            "finite S3 conjugacy-class/Casimir comparison against a declared stationary "
            "target. The diagnostic computes class frequencies, total variation, and KL, "
            "but has no predeclared acceptance threshold or generative goodness-of-fit "
            "test, so it does not validate a heat-kernel law. It is not the compact "
            "Peter-Weyl refinement lift, a particle claim, or a physical measurement."
        ),
    }


def central_record_born_report(
    *,
    record_signature: np.ndarray,
    committed: np.ndarray,
    stable_count: np.ndarray,
    commit_cycles: int,
) -> dict[str, Any]:
    """Verify a finite classical partition of categorical record IDs.

    The legacy function/file name contains ``born``.  Nothing supplied to
    this function is an independently predicted quantum probability: the
    only numbers available are frequencies obtained by normalizing the same
    record counts.  Consequently the report can certify the partition
    algebra, its repeat-read stability, and normalization of the empirical
    frequencies. No ambient algebra or action is supplied, so this function
    cannot certify centrality, quantum projectors, a Lüders instrument, or a
    Born-law comparison.
    """

    raw_signatures = np.asarray(record_signature)
    raw_committed = np.asarray(committed)
    raw_stable = np.asarray(stable_count)
    if (
        raw_signatures.ndim != 1
        or raw_committed.ndim != 1
        or raw_stable.ndim != 1
        or raw_signatures.shape != raw_committed.shape
        or raw_signatures.shape != raw_stable.shape
    ):
        return {
            "mode": "central_record_born_surface",
            "event_count": 0,
            "classical_record_partition_receipt": False,
            "centrality_validation_receipt": False,
            "central_record_algebra_receipt": False,
            "born_law_validation_receipt": False,
            "receipt": False,
            "reason": "record_inputs_must_be_equal_length_one_dimensional_arrays",
        }
    committed_is_bool = np.issubdtype(raw_committed.dtype, np.bool_)
    committed_is_binary_integer = bool(
        np.issubdtype(raw_committed.dtype, np.integer)
        and np.all((raw_committed == 0) | (raw_committed == 1))
    )
    if (
        not np.issubdtype(raw_signatures.dtype, np.integer)
        or not np.issubdtype(raw_stable.dtype, np.integer)
        or not (committed_is_bool or committed_is_binary_integer)
    ):
        return {
            "mode": "central_record_born_surface",
            "event_count": 0,
            "classical_record_partition_receipt": False,
            "centrality_validation_receipt": False,
            "central_record_algebra_receipt": False,
            "born_law_validation_receipt": False,
            "receipt": False,
            "reason": (
                "record_signatures_and_stability_must_be_integers_and_"
                "committed_mask_must_be_boolean_or_binary_integer"
            ),
        }
    if (
        not isinstance(commit_cycles, (int, np.integer))
        or isinstance(commit_cycles, (bool, np.bool_))
        or int(commit_cycles) < 0
    ):
        return {
            "mode": "central_record_born_surface",
            "event_count": 0,
            "classical_record_partition_receipt": False,
            "centrality_validation_receipt": False,
            "central_record_algebra_receipt": False,
            "born_law_validation_receipt": False,
            "receipt": False,
            "reason": "commit_cycles_must_be_a_nonnegative_integer",
        }
    signatures = raw_signatures
    committed_mask = raw_committed.astype(bool, copy=False)
    stable = raw_stable
    if signatures.size == 0:
        return {
            "mode": "central_record_born_surface",
            "event_count": 0,
            "classical_record_partition_receipt": False,
            "centrality_validation_receipt": False,
            "central_record_algebra_receipt": False,
            "born_law_validation_receipt": False,
            "receipt": False,
            "reason": "empty_record_layer",
        }
    active = committed_mask & (signatures >= 0)
    if not np.any(active):
        return {
            "mode": "central_record_born_surface",
            "event_count": 0,
            "committed_fraction": float(np.mean(committed_mask)) if committed_mask.size else 0.0,
            "classical_record_partition_receipt": False,
            "centrality_validation_receipt": False,
            "central_record_algebra_receipt": False,
            "born_law_validation_receipt": False,
            "receipt": False,
            "reason": "no_committed_records",
            "claim_boundary": (
                "No committed categorical records were present, so no classical partition "
                "diagnostic is available. No centrality, Lüders, or Born-law claim follows."
            ),
        }
    values, counts = np.unique(signatures[active], return_counts=True)
    probabilities = counts.astype(float) / float(np.sum(counts))
    idempotent_error = _projector_idempotent_error(signatures[active], values)
    repeat_read_fraction = float(np.mean(stable[active] >= int(commit_cycles)))
    partition_receipt = bool(
        idempotent_error < 1e-12 and abs(float(np.sum(probabilities)) - 1.0) < 1e-12
    )
    event_rows = [
        {
            "event_id": int(value),
            "count": int(count),
            "empirical_record_frequency": float(prob),
            "partition_filter_idempotent": True,
            "luders_conditioning_validation_receipt": False,
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
        "record_partition_filters_commute": True,
        "partition_filter_idempotent": bool(idempotent_error < 1e-12),
        # These are validation-status fields, not mathematical negations.  No
        # ambient quantum projectors or instrument are inputs to this report.
        "record_projector_commutation_validation_receipt": False,
        "luders_conditioning_validation_receipt": False,
        "repeat_read_stability_fraction": repeat_read_fraction,
        "commit_cycles": int(commit_cycles),
        "sample_events": event_rows,
        "classical_record_partition_receipt": partition_receipt,
        "centrality_validation_receipt": False,
        "central_record_algebra_receipt": False,
        "born_law_validation_receipt": False,
        # The legacy generic gate once advertised the Born-named surface.
        # Keep it false so old consumers cannot mistake the narrower algebra
        # receipt above for a Born-law result.
        "receipt": False,
        "receipt_semantics": "finite_classical_categorical_partition_only",
        "probability_semantics": (
            "empirical frequencies normalized from these same record counts; no independent "
            "Born probability is supplied or tested"
        ),
        "physical_claim": False,
        "claim_boundary": (
            "finite classical partition of committed categorical record signatures, with "
            "commuting pointwise indicator filters, idempotent refiltering, and repeat-read "
            "diagnostics. No ambient algebra or action is supplied, so centrality and a "
            "Lüders instrument are not tested. The displayed frequencies are normalized from "
            "the same counts, so this is not a Born-law derivation, prediction, validation, "
            "quantum instrument, or physical measurement claim."
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
