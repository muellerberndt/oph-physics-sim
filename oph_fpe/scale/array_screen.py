from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial import cKDTree

from oph_fpe.bulk.modular_lift import point_cloud_dimension_report
from oph_fpe.core.graph import fibonacci_sphere_points
from oph_fpe.core.pixel_scale import pixel_scale_from_config
from oph_fpe.core.screen_microphysics import ports_per_patch_from_config, screen_microphysics_from_config
from oph_fpe.dynamics import dispatch_configured_kernels, kernel_dispatch_manifest_summary
from oph_fpe.evidence import RunBundle
from oph_fpe.evidence.hashes import CANONICAL_HASH_SCHEMA, stable_json_hash
from oph_fpe.gauge.covariant_overlap import (
    GAUGE_COVARIANT_OVERLAP_SCHEMA,
    covariant_mismatch_mask,
    overlap_contract_metadata,
    repair_covariant_port_pairs,
)


def run_array_screen_config(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    seed = int(config.get("seed", 1))
    rng = np.random.default_rng(seed)
    pixel_scale = pixel_scale_from_config(config)
    graph_cfg = config.get("graph", {})
    patch_count = int(graph_cfg.get("patch_count", 65536))
    neighbors = int(graph_cfg.get("neighbors", ports_per_patch_from_config(config)))
    group_name = str(config.get("group", {}).get("name", "S3")).upper()
    group_order = _group_order(group_name)
    points = fibonacci_sphere_points(patch_count)
    left, right = _knn_edges(points, neighbors)
    edge_count = int(left.size)
    screen_microphysics = screen_microphysics_from_config(config, patch_count, edge_count)

    port_left = rng.integers(0, group_order, size=edge_count, dtype=np.int16)
    port_right = rng.integers(0, group_order, size=edge_count, dtype=np.int16)
    gauge = rng.integers(0, group_order, size=edge_count, dtype=np.int16)
    modular_depth = rng.random(patch_count, dtype=np.float64)
    modular_time = np.zeros(patch_count, dtype=np.float64)
    # Record stability can legitimately outlive an int16 counter in long
    # campaigns.  Keep it wide and update it through the saturating helper
    # below so a settled record can never wrap back to a small/negative age.
    stable_count = np.zeros(patch_count, dtype=np.uint32)
    committed = np.zeros(patch_count, dtype=bool)
    prev_signature = np.full(patch_count, -1, dtype=np.int64)
    degree = np.bincount(np.concatenate([left, right]), minlength=patch_count).astype(np.float64)
    degree = np.maximum(degree, 1.0)

    run_id = config.get("run_id") or _run_id(config.get("name", "array_screen"))
    bundle = RunBundle(out_dir, run_id)
    bundle.write_config(config)
    bundle.write_json("pixel_scale.json", pixel_scale.as_jsonable())
    bundle.write_json("pixel_report.json", pixel_scale.as_jsonable())
    bundle.write_json("screen_microphysics.json", screen_microphysics.as_jsonable())

    dyn = config.get("dynamics", {})
    cycles = int(dyn.get("cycles", 64))
    repairs_per_cycle = int(dyn.get("repairs_per_cycle", edge_count // 4))
    commit_cycles = int(dyn.get("record_commit_cycles", 8))
    beta_schedule = dyn.get("beta_schedule", {})
    mod_cfg = config.get("modular_flow", {})
    modular_cap_drive = _modular_cap_drive(points, mod_cfg)
    trace_interval = int(mod_cfg.get("trace_interval", 4))
    depth_samples: list[np.ndarray] = []
    trace: list[dict[str, Any]] = []
    repair_receipts: list[dict[str, Any]] = []

    for cycle in range(cycles):
        beta = _beta_at(beta_schedule, cycle, cycles)
        mismatches = covariant_mismatch_mask(
            port_left,
            port_right,
            gauge,
            group_name=group_name,
            group_order=group_order,
        )
        phi_before = int(np.sum(mismatches))
        active = np.flatnonzero(mismatches)
        if active.size:
            chosen_count = min(repairs_per_cycle, active.size)
            chosen = rng.choice(active, size=chosen_count, replace=False)
            direction = rng.random(chosen_count) < 0.5
            repair_covariant_port_pairs(
                port_left,
                port_right,
                gauge,
                chosen,
                direction,
                group_name=group_name,
                group_order=group_order,
            )
        mismatches_after = covariant_mismatch_mask(
            port_left,
            port_right,
            gauge,
            group_name=group_name,
            group_order=group_order,
        )
        phi_after = int(np.sum(mismatches_after))
        repair_receipts.append(
            {
                "verifier": "array_parallel_repair_nonincrease",
                "cycle": cycle,
                "ok": phi_after <= phi_before,
                "phi_before": phi_before,
                "phi_after": phi_after,
                "beta": beta,
                "mismatch_definition": GAUGE_COVARIANT_OVERLAP_SCHEMA,
            }
        )

        incident_mismatch = (
            np.bincount(left, weights=mismatches_after.astype(float), minlength=patch_count)
            + np.bincount(right, weights=mismatches_after.astype(float), minlength=patch_count)
        )
        repair_load = incident_mismatch / degree
        modular_depth, modular_time = _modular_update(
            points,
            left,
            right,
            degree,
            modular_depth,
            modular_time,
            repair_load,
            mod_cfg,
            cap_drive=modular_cap_drive,
        )
        signature = _node_signature(port_left, port_right, left, right, patch_count)
        stable_count, committed = _advance_record_commit_state(
            signature,
            prev_signature,
            stable_count,
            incident_mismatch,
            commit_cycles=commit_cycles,
        )
        prev_signature = signature
        if cycle % trace_interval == 0 or cycle == cycles - 1:
            depth_samples.append(modular_depth.copy())
        trace.append(
            {
                "cycle": cycle,
                "beta": beta,
                "phi": phi_after,
                "mismatch_edges": phi_after,
                "committed_records": int(np.sum(committed)),
                "record_entropy": _entropy(signature[committed]) if np.any(committed) else 0.0,
                "defect_proxy_count": int(
                    _defect_proxy(
                        gauge,
                        port_left,
                        port_right,
                        group_name=group_name,
                        group_order=group_order,
                    )
                ),
                "modular_depth_mean": float(np.mean(modular_depth)),
                "modular_depth_std": float(np.std(modular_depth)),
            }
        )

    cloud = _modular_cloud(points, depth_samples, config.get("observables", {}).get("modular_lift", {}), seed)
    dimensions = point_cloud_dimension_report(
        cloud,
        center_samples=int(config.get("observables", {}).get("modular_lift", {}).get("center_samples", 4096)),
        seed=seed + 313,
    )
    dimensions["distance_source"] = "array_modular_lift_record_history"
    dimensions["point_count"] = int(cloud.shape[0])
    dimensions["screen_patch_count"] = patch_count
    dimensions["modular_samples"] = len(depth_samples)

    _write_csv(bundle.path / "mismatch_trace.csv", trace)
    bundle.write_jsonl("verifier_receipts.jsonl", repair_receipts)
    bundle.write_json("dimension_report.json", dimensions)
    bundle.write_json(
        "cosmology_observables.json",
        _cosmology_proxy(trace, dimensions, pixel_scale.as_jsonable(), screen_microphysics.as_jsonable()),
    )
    bundle.write_json("state_final_summary.json", _summary(patch_count, edge_count, trace[-1], modular_depth))
    bundle.write_json(
        "seed_material.json",
        {
            "config_hash": stable_json_hash(config),
            "hash_schema": CANONICAL_HASH_SCHEMA,
            "seed": seed,
        },
    )
    bundle.write_json("gauge_covariant_overlap_contract.json", overlap_contract_metadata())
    kernel_dispatch = dispatch_configured_kernels(config, bundle.path, engine="array_screen")
    manifest = {
        "run_id": run_id,
        "name": config.get("name"),
        "engine": "array_screen",
        "claim_boundary": config.get("claim_boundary"),
        "patch_count": patch_count,
        "edge_count": edge_count,
        "group": group_name,
        "gauge_covariant_overlap": overlap_contract_metadata(),
        "pixel_scale": pixel_scale.as_jsonable(),
        "oph_constants": pixel_scale.constants.as_jsonable(),
        "screen_microphysics": screen_microphysics.as_jsonable(),
        "screen_units": screen_microphysics.as_jsonable()["screen_units"],
        "cycles": cycles,
        "final_phi": int(trace[-1]["phi"]),
        "dimension_report": dimensions,
    }
    result = {"run_id": run_id, "path": str(bundle.path), "final_phi": int(trace[-1]["phi"]), "dimensions": dimensions}
    if kernel_dispatch:
        summary = kernel_dispatch_manifest_summary(kernel_dispatch)
        manifest["kernel_dispatch"] = summary
        result["kernel_dispatch"] = summary
    bundle.write_manifest(manifest)
    return result


def _knn_edges(points: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= 1:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    tree = cKDTree(points)
    _, indices = tree.query(points, k=min(points.shape[0], neighbors + 1))
    if indices.ndim == 1:
        indices = indices[:, None]
    neighbor_indices = indices[:, 1:]
    if neighbor_indices.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    node_count = points.shape[0]
    src = np.repeat(np.arange(node_count, dtype=np.int64), neighbor_indices.shape[1])
    dst = neighbor_indices.reshape(-1).astype(np.int64)
    keys = src * node_count + dst
    reverse_keys = dst * node_count + src
    mutual = np.isin(keys, reverse_keys)
    left = np.minimum(src[mutual], dst[mutual])
    right = np.maximum(src[mutual], dst[mutual])
    edge_keys = np.unique(left * node_count + right)
    return edge_keys // node_count, edge_keys % node_count


def _group_order(group_name: str) -> int:
    if group_name == "Z2":
        return 2
    if group_name == "S3":
        return 6
    if group_name.startswith("C") and group_name[1:].isdigit():
        return int(group_name[1:])
    raise ValueError(f"array screen engine supports Z2, S3, and clock groups, not {group_name}")


def _beta_at(schedule: dict[str, Any], cycle: int, total_cycles: int) -> float:
    beta_start = float(schedule.get("beta_start", 0.1))
    beta_end = float(schedule.get("beta_end", 10.0))
    if total_cycles <= 1:
        return beta_end
    frac = cycle / (total_cycles - 1)
    return beta_start * ((beta_end / beta_start) ** frac)


def _modular_update(
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    degree: np.ndarray,
    depth: np.ndarray,
    modular_time: np.ndarray,
    repair_load: np.ndarray,
    config: dict[str, Any],
    *,
    cap_drive: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not config.get("enabled", True):
        return depth, modular_time
    dt = float(config.get("dt", 0.03))
    damping = float(config.get("damping", 0.012))
    load_coupling = float(config.get("repair_load_coupling", 0.35))
    cap_coupling = float(config.get("cap_coupling", 0.08))
    diffusion = float(config.get("diffusion", 0.06))
    if cap_drive is None:
        cap_drive = _modular_cap_drive(points, config)
    neighbor_sum = (
        np.bincount(left, weights=depth[right], minlength=depth.size)
        + np.bincount(right, weights=depth[left], minlength=depth.size)
    )
    neighbor_mean = neighbor_sum / degree
    centered_load = repair_load - float(np.mean(repair_load))
    new_time = modular_time + dt * (1.0 + cap_coupling * cap_drive)
    new_depth = depth + dt * (
        load_coupling * centered_load
        + cap_coupling * cap_drive
        + diffusion * (neighbor_mean - depth)
        - damping * depth
    )
    return new_depth, new_time


def _modular_cap_drive(points: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if not config.get("enabled", True):
        return np.zeros(points.shape[0], dtype=np.float64)
    axes = fibonacci_sphere_points(int(config.get("cap_axes", 12)))
    return np.mean(np.tanh(3.0 * (points @ axes.T)), axis=1)


def _node_signature(
    port_left: np.ndarray,
    port_right: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    patch_count: int,
) -> np.ndarray:
    """Return a deterministic hash of every oriented incident port packet.

    The former signature was just a sum, so records such as ``[0, 2]`` and
    ``[1, 1]`` were indistinguishable.  This vectorized multiset hash keys each
    packet by its edge slot, endpoint orientation, and exact label.  It keeps
    the O(E) memory/time profile needed by the large array runs while reducing
    accidental record collisions to a 64-bit hash collision.
    """

    port_left = np.asarray(port_left, dtype=np.uint64)
    port_right = np.asarray(port_right, dtype=np.uint64)
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if port_left.shape != port_right.shape or port_left.shape != left.shape or left.shape != right.shape:
        raise ValueError("port packets and edge endpoint arrays must have matching shapes")

    edge_slot = np.arange(port_left.size, dtype=np.uint64)
    # The constants are independent odd 64-bit salts.  SplitMix64 then
    # avalanches every input bit before endpoint contributions are combined.
    left_tokens = edge_slot ^ ((port_left + np.uint64(1)) * np.uint64(0xD6E8FEB86659FD93))
    right_tokens = edge_slot ^ ((port_right + np.uint64(1)) * np.uint64(0xA5A3564E27F8862D))
    signatures = np.zeros(int(patch_count), dtype=np.uint64)
    np.bitwise_xor.at(signatures, left, _splitmix64(left_tokens))
    np.bitwise_xor.at(signatures, right, _splitmix64(right_tokens))
    return signatures.view(np.int64)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 finalizer with intentional uint64 wraparound."""

    with np.errstate(over="ignore"):
        mixed = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return mixed ^ (mixed >> np.uint64(31))


def _advance_record_commit_state(
    signature: np.ndarray,
    previous_signature: np.ndarray,
    previous_stable_count: np.ndarray,
    incident_mismatch: np.ndarray,
    *,
    commit_cycles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance record age and derive *current*, revocable commit status.

    A commit denotes a presently stable and overlap-consistent readback.  It is
    therefore revoked as soon as the local record changes or an incident
    mismatch appears; commits are not sticky historical flags.
    """

    signature = np.asarray(signature, dtype=np.int64)
    previous_signature = np.asarray(previous_signature, dtype=np.int64)
    previous_stable_count = np.asarray(previous_stable_count, dtype=np.uint32)
    incident_mismatch = np.asarray(incident_mismatch)
    if not (
        signature.shape
        == previous_signature.shape
        == previous_stable_count.shape
        == incident_mismatch.shape
    ):
        raise ValueError("record commit arrays must have matching shapes")

    maximum = np.uint64(np.iinfo(np.uint32).max)
    incremented = np.minimum(previous_stable_count.astype(np.uint64) + np.uint64(1), maximum).astype(np.uint32)
    stable_count = np.where(signature == previous_signature, incremented, np.uint32(1)).astype(np.uint32)
    threshold = np.uint32(max(1, int(commit_cycles)))
    committed = (stable_count >= threshold) & (incident_mismatch <= 0)
    return stable_count, committed


def _entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def _defect_proxy(
    gauge: np.ndarray,
    port_left: np.ndarray,
    port_right: np.ndarray,
    *,
    group_name: str,
    group_order: int,
) -> int:
    return int(
        np.sum(
            covariant_mismatch_mask(
                port_left,
                port_right,
                gauge,
                group_name=group_name,
                group_order=group_order,
            )
        )
    )


def _modular_cloud(points: np.ndarray, depth_samples: list[np.ndarray], config: dict[str, Any], seed: int) -> np.ndarray:
    max_points = int(config.get("max_points", 200000))
    if not depth_samples:
        return points
    chunks = []
    all_depths = np.concatenate(depth_samples)
    q_low = float(np.percentile(all_depths, 1))
    q_high = float(np.percentile(all_depths, 99))
    for index, depth in enumerate(depth_samples):
        if q_high > q_low:
            normalized = np.clip((depth - q_low) / (q_high - q_low), 0.0, 1.0)
        else:
            normalized = 1.0 / (1.0 + np.exp(-depth))
        radius = 0.05 + 0.86 * np.cbrt(normalized)
        if len(depth_samples) > 1:
            radius = radius + 0.04 * (index / (len(depth_samples) - 1))
        chunks.append(points * radius[:, None])
    cloud = np.vstack(chunks)
    if cloud.shape[0] <= max_points:
        return cloud
    rng = np.random.default_rng(seed)
    chosen = rng.choice(cloud.shape[0], size=max_points, replace=False)
    return cloud[np.sort(chosen)]


def _cosmology_proxy(
    trace: list[dict[str, Any]],
    dimensions: dict[str, Any],
    pixel_scale: dict[str, Any],
    screen_microphysics: dict[str, Any],
) -> dict[str, Any]:
    phi0 = max(float(trace[0]["phi"]), 1e-9)
    final_phi = float(trace[-1]["phi"])
    return {
        "Phi_initial": phi0,
        "Phi_final": final_phi,
        "Phi_drop_fraction": (phi0 - final_phi) / phi0,
        "dimension_estimates": dimensions,
        "pixel_scale": pixel_scale,
        "screen_microphysics": screen_microphysics,
        "modular_flow_status": "array_regulator_side_surrogate",
        "boltzmann_adapter_status": "not_implemented_mvp",
    }


def _summary(patch_count: int, edge_count: int, final_trace: dict[str, Any], depth: np.ndarray) -> dict[str, Any]:
    return {
        "nodes": patch_count,
        "edges": edge_count,
        "final_phi": final_trace["phi"],
        "committed_records": final_trace["committed_records"],
        "modular_depth": {
            "min": float(np.min(depth)),
            "mean": float(np.mean(depth)),
            "max": float(np.max(depth)),
            "std": float(np.std(depth)),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_id(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name).lower()).strip("_")
    return f"{slug}_{int(time.time())}"
