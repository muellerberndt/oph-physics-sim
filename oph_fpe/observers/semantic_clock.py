from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from oph_fpe.claims import CONTINUATION, RECOVERED_CORE, with_claim_metadata
from oph_fpe.evidence.hashes import stable_json_hash

OBSERVER_KINDS = ("patch", "cap", "future")
EXECUTION_METADATA_KEYS = {
    "execution_epoch",
    "scheduler_event_index",
    "worker_id",
    "workerId",
    "retry_count",
    "retryCount",
    "timestamp",
    "queue_position",
    "queuePosition",
    "packet_latency",
    "packetLatency",
}


def distributed_observer_uid(*, run_id: str, observer_kind: str, global_observer_index: int) -> str:
    kind = _checked_kind(observer_kind)
    return f"{run_id}:{kind}:{int(global_observer_index)}"


def normalize_observer_frame(frame: Mapping[str, Any], *, record_order: int) -> dict[str, Any]:
    row = dict(frame)
    row["execution_epoch"] = row.get("execution_epoch", row.get("cycle", row.get("frameIndex")))
    row["scheduler_event_index"] = row.get("scheduler_event_index", row.get("eventIndex", row.get("frameIndex")))
    row["observer_record_order"] = int(row.get("observer_record_order", row.get("recordOrder", record_order)))
    row["observer_modular_parameter"] = row.get(
        "observer_modular_parameter",
        row.get("modularTime", row.get("modular_time")),
    )
    row["observer_clock_uncertainty"] = float(
        row.get("observer_clock_uncertainty", row.get("clockUncertainty", row.get("clock_uncertainty", 0.0)))
    )
    row["execution_provenance"] = {
        "execution_epoch": row.get("execution_epoch"),
        "scheduler_event_index": row.get("scheduler_event_index"),
        "worker_id": row.get("worker_id", row.get("workerId")),
        "retry_count": row.get("retry_count", row.get("retryCount", 0)),
        "packet_latency": row.get("packet_latency", row.get("packetLatency")),
    }
    return row


def semantic_event_key(event: Mapping[str, Any]) -> str:
    semantic = {
        key: value
        for key, value in event.items()
        if key not in EXECUTION_METADATA_KEYS and key != "execution_provenance"
    }
    return _stable_hash(semantic)


def semantic_history_digest(events: Iterable[Mapping[str, Any]]) -> str:
    keys = sorted(semantic_event_key(event) for event in events)
    return _stable_hash(keys)


def semantic_history_invariance_report(histories: Sequence[Iterable[Mapping[str, Any]]]) -> dict[str, Any]:
    digests = [semantic_history_digest(history) for history in histories]
    receipt = bool(digests and len(set(digests)) == 1)
    report = {
        "mode": "semantic_history_invariance_v1",
        "SEMANTIC_HISTORY_SCHEDULER_INVARIANCE_RECEIPT": receipt,
        "receipt": receipt,
        "history_count": len(digests),
        "semantic_history_digests": digests,
        "claim_boundary": (
            "Finite semantic-history digest check. Scheduler counters, retries, packet latency, timestamps, "
            "and queue positions are ignored only as execution metadata; timeout-sensitive latency must be "
            "modeled as semantic input instead."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=RECOVERED_CORE if receipt else CONTINUATION,
        receipt="SEMANTIC_HISTORY_SCHEDULER_INVARIANCE_RECEIPT",
    )


def observer_registry_audit(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(entry) for entry in entries]
    uids: set[str] = set()
    duplicate_uids: list[str] = []
    namespace_collisions: list[dict[str, Any]] = []
    anchor_reuse: list[dict[str, Any]] = []
    for row in rows:
        uid = str(row.get("distributed_observer_uid", ""))
        if uid in uids:
            duplicate_uids.append(uid)
        uids.add(uid)
        kind = row.get("observer_kind")
        if kind not in OBSERVER_KINDS:
            namespace_collisions.append({"uid": uid, "observer_kind": kind, "reason": "unknown_kind"})
        local_anchor = str(row.get("local_anchor_patch_id", ""))
        local_observer = str(row.get("local_observer_index", ""))
        if local_anchor == local_observer or local_anchor == f"observer:{local_observer}":
            anchor_reuse.append({"uid": uid, "local_anchor_patch_id": local_anchor, "local_observer_index": local_observer})
    receipt = bool(rows and not duplicate_uids and not namespace_collisions and not anchor_reuse)
    report = {
        "mode": "observer_registry_namespace_audit_v1",
        "GLOBAL_OBSERVER_REGISTRY_NAMESPACE_RECEIPT": receipt,
        "receipt": receipt,
        "observer_count": len(rows),
        "observer_kinds": sorted(set(str(row.get("observer_kind")) for row in rows)),
        "duplicate_uid_count": len(duplicate_uids),
        "namespace_collision_count": len(namespace_collisions),
        "anchor_reuse_violation_count": len(anchor_reuse),
        "duplicate_uids": duplicate_uids[:8],
        "namespace_collisions": namespace_collisions[:8],
        "anchor_reuse_violations": anchor_reuse[:8],
        "claim_boundary": (
            "Registry namespace audit for finite exported observers. It checks disjoint patch/cap/future "
            "observer namespaces and separation of observer indices from anchor patch IDs; it is not by "
            "itself a global observer-identity theorem."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=RECOVERED_CORE if receipt else CONTINUATION,
        receipt="GLOBAL_OBSERVER_REGISTRY_NAMESPACE_RECEIPT",
    )


def affine_clock_residual_report(
    left: Sequence[float],
    right: Sequence[float],
    *,
    scale: float = 1.0,
    shift: float = 0.0,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    count = min(len(left), len(right))
    residuals = [abs((scale * float(left[index]) + shift) - float(right[index])) for index in range(count)]
    max_residual = max(residuals) if residuals else 0.0
    receipt = bool(count and max_residual <= float(tolerance))
    report = {
        "mode": "observer_clock_affine_residual_v1",
        "OBSERVER_CLOCK_AFFINE_RESIDUAL_RECEIPT": receipt,
        "receipt": receipt,
        "sample_count": count,
        "scale": float(scale),
        "shift": float(shift),
        "tolerance": float(tolerance),
        "max_residual": float(max_residual),
        "claim_boundary": (
            "Finite affine clock-residual comparison for declared clock instruments. It measures the "
            "certificate residual and does not define which instrument is the physical clock."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=RECOVERED_CORE if receipt else CONTINUATION,
        receipt="OBSERVER_CLOCK_AFFINE_RESIDUAL_RECEIPT",
    )


def _checked_kind(observer_kind: str) -> str:
    kind = str(observer_kind)
    if kind not in OBSERVER_KINDS:
        raise ValueError(f"unknown observer kind: {observer_kind!r}")
    return kind


def _stable_hash(value: Any) -> str:
    return stable_json_hash(value).removeprefix("sha256:")
