from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import numpy as np

from oph_fpe.evidence.hashes import stable_json_hash


@dataclass
class RecordFamily:
    object_id: str
    support_nodes: list[int]
    record_signature: int
    persistence: int
    overlap_agreement: float
    repair_history_hash: str
    counterfactual_stability: float

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def extract_record_families(
    records: dict[str, np.ndarray] | np.ndarray,
    edges: tuple[np.ndarray, np.ndarray],
    projections: Any = None,
    persistence_horizon: int = 8,
    *,
    max_families: int | None = None,
) -> list[RecordFamily]:
    """Extract persistent overlap-stable connected record families.

    This is an observer-facing object surrogate: it groups nodes whose visible
    record signature agrees across graph overlaps and has persisted for the
    declared stability horizon. Hidden state is not used.
    """

    projection_cfg = projections if isinstance(projections, dict) else {}
    if isinstance(records, dict):
        signatures = np.asarray(records.get("record_signature"), dtype=np.int64)
        stable_count = np.asarray(records.get("stable_count", np.ones_like(signatures)), dtype=np.int64)
        repair_load = np.asarray(records.get("repair_load", np.zeros_like(signatures, dtype=float)), dtype=float)
        object_packets = _visible_object_packets(records, projection_cfg)
    else:
        signatures = np.asarray(records, dtype=np.int64)
        stable_count = np.ones_like(signatures, dtype=np.int64) * int(persistence_horizon)
        repair_load = np.zeros_like(signatures, dtype=float)
        object_packets = signatures
    left, right = (np.asarray(edges[0], dtype=np.int64), np.asarray(edges[1], dtype=np.int64))
    valid = stable_count >= int(persistence_horizon)
    parent = np.arange(signatures.size, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    same = valid[left] & valid[right] & (object_packets[left] == object_packets[right])
    for a, b in zip(left[same], right[same], strict=False):
        union(int(a), int(b))

    valid_nodes = np.flatnonzero(valid).astype(np.int64)
    roots = np.array([find(int(node)) for node in valid_nodes], dtype=np.int64)
    unique_roots, counts = np.unique(roots, return_counts=True)
    order = np.lexsort((unique_roots, -counts))
    families: list[RecordFamily] = []
    min_support_size = int(projection_cfg.get("min_support_size", 1))
    for root in unique_roots[order]:
        nodes = [int(node) for node in valid_nodes[roots == root]]
        if len(nodes) < min_support_size:
            continue
        node_array = np.asarray(nodes, dtype=np.int64)
        signature_values, signature_counts = np.unique(object_packets[node_array], return_counts=True)
        signature = int(signature_values[int(np.argmax(signature_counts))])
        overlap = 1.0
        persistence = int(np.min(stable_count[node_array]))
        history_hash = stable_json_hash(
            {
                "signature": signature,
                "support_size": int(node_array.size),
                "repair_load_mean": float(np.mean(repair_load[node_array])) if node_array.size else 0.0,
            }
        )
        families.append(
            RecordFamily(
                object_id=f"obj_{len(families):06d}",
                support_nodes=nodes,
                record_signature=signature,
                persistence=persistence,
                overlap_agreement=overlap,
                repair_history_hash=history_hash,
                counterfactual_stability=0.0,
            )
        )
        if max_families is not None and len(families) >= int(max_families):
            break
    return families


def object_consensus_score(record_family: RecordFamily, observer_views: Iterable[dict[str, Any]]) -> float:
    support = set(record_family.support_nodes)
    scores: list[float] = []
    for view in observer_views:
        if view.get("view_type") != "patch_observer":
            continue
        view_support = set(int(node) for node in view.get("support_nodes", []))
        if not view_support:
            continue
        overlap = len(support & view_support)
        if overlap:
            histogram = view.get("object_packet_histogram") or view.get("record_signature_histogram", {})
            signature_mass = _histogram_value(histogram, record_family.record_signature)
            coverage = overlap / max(1, len(support))
            if histogram and signature_mass > 0.0:
                scores.append(float(min(1.0, coverage) * signature_mass + (1.0 if signature_mass > 0.0 else 0.0)) / 2.0)
            else:
                scores.append(float(min(1.0, coverage)))
    return float(np.mean(scores)) if scores else float(record_family.overlap_agreement)


def counterfactual_stability(
    record_family: RecordFamily,
    perturbations: Iterable[Any],
    remeasure_fn: Callable[[RecordFamily, Any], int],
) -> float:
    trials = 0
    stable = 0
    for perturbation in perturbations:
        trials += 1
        stable += int(remeasure_fn(record_family, perturbation) == record_family.record_signature)
    return float(stable / trials) if trials else 0.0


def observer_object_report(record_families: list[RecordFamily], observer_views: Iterable[dict[str, Any]]) -> dict[str, Any]:
    views = list(observer_views)
    agreements = np.array([object_consensus_score(family, views) for family in record_families], dtype=float)
    persistent = [family for family in record_families if family.persistence > 0 and family.overlap_agreement > 0.0]
    return {
        "object_count": len(record_families),
        "persistent_object_count": len(persistent),
        "median_overlap_agreement": float(np.median(agreements)) if agreements.size else 0.0,
        "p10_overlap_agreement": float(np.percentile(agreements, 10)) if agreements.size else 0.0,
        "median_counterfactual_stability": float(np.median([family.counterfactual_stability for family in record_families]))
        if record_families
        else 0.0,
        "bad_record_rewrite_detected": _detect_bad_rewrite(record_families),
        "claim_boundary": "observer-facing record-family object construction; not yet bulk reconstruction",
    }


def assign_counterfactual_stability_from_records(
    record_families: list[RecordFamily],
    records: dict[str, np.ndarray],
    projection_cfg: dict[str, Any] | None = None,
    *,
    perturbations: int = 16,
    seed: int = 1,
) -> None:
    """Set each family stability from support-visible bounded subsampling.

    This is not a hidden-state repair rerun. It asks whether the object's visible
    packet label survives bounded observer-facing remeasurement when a local
    observer sees only a random sub-support of the same record family.
    """

    packets = _visible_object_packets(records, projection_cfg or {})
    rng = np.random.default_rng(seed)
    trials = max(0, int(perturbations))
    for family_index, family in enumerate(record_families):
        nodes = np.asarray(family.support_nodes, dtype=np.int64)
        if nodes.size == 0 or trials == 0:
            family.counterfactual_stability = 0.0
            continue
        stable = 0
        local_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + family_index)
        for _ in range(trials):
            keep_fraction = float(local_rng.uniform(0.55, 1.0))
            keep_count = max(1, int(round(nodes.size * keep_fraction)))
            keep = local_rng.choice(nodes, size=keep_count, replace=False) if keep_count < nodes.size else nodes
            stable += int(_modal_int(packets[keep]) == int(family.record_signature))
        family.counterfactual_stability = float(stable / trials)


def visible_object_packets(records: dict[str, np.ndarray], projection_cfg: dict[str, Any] | None = None) -> np.ndarray:
    return _visible_object_packets(records, projection_cfg or {})


def _detect_bad_rewrite(record_families: list[RecordFamily]) -> bool:
    seen: dict[str, int] = {}
    for family in record_families:
        previous = seen.get(family.repair_history_hash)
        if previous is not None and previous != family.record_signature:
            return True
        seen[family.repair_history_hash] = family.record_signature
    return False


def _histogram_value(histogram: Any, key: int) -> float:
    if not isinstance(histogram, dict):
        return 0.0
    if key in histogram:
        return float(histogram[key])
    return float(histogram.get(str(int(key)), 0.0))


def _modal_int(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        return -1
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[int(np.argmax(counts))])


def _visible_object_packets(records: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    signatures = np.asarray(records.get("record_signature"), dtype=np.int64)
    if signatures.size == 0:
        return signatures
    mode = str(config.get("packet_mode", "exact_signature"))
    if mode in {"exact", "exact_signature"}:
        packets = signatures.copy()
    else:
        bins = int(config.get("signature_bins", 16))
        packets = _quantile_bins(signatures.astype(float), bins)
        if bool(config.get("include_s3_sector", True)) and "s3_sector_class" in records:
            sectors = np.asarray(records["s3_sector_class"], dtype=np.int64)
            packets = packets * 8 + np.clip(sectors, 0, 7)
        if bool(config.get("include_repair_load_bin", False)) and "repair_load" in records:
            packets = packets * 16 + _quantile_bins(np.asarray(records["repair_load"], dtype=float), 16)
        if bool(config.get("include_cumulative_repair_bin", False)) and "cumulative_repair_load" in records:
            packets = packets * 16 + _quantile_bins(np.asarray(records["cumulative_repair_load"], dtype=float), 16)
        if bool(config.get("include_mismatch_bin", False)) and "local_mismatch_density" in records:
            packets = packets * 16 + _quantile_bins(np.asarray(records["local_mismatch_density"], dtype=float), 16)
    return packets.astype(np.int64)


def _quantile_bins(values: np.ndarray, bin_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    bins = max(1, int(bin_count))
    if bins == 1 or float(np.std(values)) < 1e-12:
        return np.zeros(values.size, dtype=np.int64)
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.digitize(values, edges, right=False).astype(np.int64)
