from __future__ import annotations

import csv
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import yaml

from oph_fpe.experiments import load_config


DISTRIBUTED_KERNEL_VERSION = "distributed_observer_patch_kernel_v1"

RECEIPT_KEYS: tuple[str, ...] = (
    "observer_like_self_reading_system_receipt",
    "observer_modular_time_receipt",
    "h3_response_candidate_receipt",
    "h3_response_control_separation_receipt",
    "observer_h3_object_population_receipt",
    "observer_facing_3p1d_h3_experience_receipt",
    "theorem_assisted_consensus_3d_bulk_readout_receipt",
    "strict_neutral_third_person_bulk_receipt",
    "physical_cmb_prediction_receipt",
    "finite_lorentz_theorem_contract_receipt",
    "paper_faithful_observer_spacetime_emergence_receipt",
    "paper_faithful_consensus_bulk_emergence_receipt",
    "scale_compressed_pn_silence_to_observation_receipt",
)


def prepare_distributed_oph_universe(
    *,
    config_path: Path,
    out_dir: Path,
    run_id: str,
    shard_count: int,
    patch_count_per_shard: int,
    observers_per_shard: int,
    worker_count: int,
    max_screen_points: int = 8000,
    max_h3_objects: int = 1024,
    seed_stride: int = 1009,
    seam_halo_width: int = 2048,
) -> dict[str, Any]:
    """Prepare a theorem-honest distributed OPH universe run.

    Each shard is a bounded observer-like self-reading screen patch. The reducer
    may certify a federated witness, but it must not promote the run to a single
    strict neutral bulk unless future cross-shard repair/overlap receipts exist.
    """

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if patch_count_per_shard <= 0:
        raise ValueError("patch_count_per_shard must be positive")
    if observers_per_shard <= 0:
        raise ValueError("observers_per_shard must be positive")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")

    base_config = load_config(config_path)
    base_seed = int(base_config.get("seed", 1))
    out_dir = Path(out_dir)
    config_dir = out_dir / "configs"
    script_dir = out_dir / "scripts"
    config_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)
    atlas = _unified_atlas(
        run_id=run_id,
        shard_count=shard_count,
        patch_count_per_shard=patch_count_per_shard,
        observers_per_shard=observers_per_shard,
        seam_halo_width=seam_halo_width,
    )

    shards: list[dict[str, Any]] = []
    for shard_index in range(int(shard_count)):
        shard_id = f"{run_id}_shard{shard_index:04d}"
        atlas_shard = atlas["shards"][shard_index]
        shard_config = _shard_config(
            base_config,
            run_id=run_id,
            shard_id=shard_id,
            shard_index=shard_index,
            shard_count=shard_count,
            patch_count_per_shard=patch_count_per_shard,
            observers_per_shard=observers_per_shard,
            seed=base_seed + shard_index * int(seed_stride),
            atlas_shard=atlas_shard,
            seam_halo_width=seam_halo_width,
        )
        path = config_dir / f"{shard_id}.yml"
        path.write_text(yaml.safe_dump(shard_config, sort_keys=False), encoding="utf-8")
        shards.append(
            {
                "shard_index": shard_index,
                "shard_id": shard_id,
                "config_path": str(path.relative_to(out_dir)),
                "run_id": shard_id,
                "seed": int(shard_config["seed"]),
                "patch_count": int(patch_count_per_shard),
                "observer_count": int(observers_per_shard),
                "worker_index": shard_index % int(worker_count),
                "global_patch_range": atlas_shard["global_patch_range"],
                "global_observer_range": atlas_shard["global_observer_range"],
                "atlas_center": atlas_shard["atlas_center"],
                "seam_neighbor_indices": atlas_shard["seam_neighbor_indices"],
            }
        )

    manifest = {
        "kernel": DISTRIBUTED_KERNEL_VERSION,
        "run_id": run_id,
        "config_path": str(config_path),
        "claim_boundary": (
            "Distributed OPH-FPE execution over observer-like self-reading screen shards. "
            "Each shard has bounded local state, ports/boundaries, readback, records, feedback/repair moves, "
            "and public receipts. The reducer can certify a federated large-universe witness; it cannot by "
            "itself certify a strict single neutral third-person bulk without explicit cross-shard overlap "
            "repair receipts."
        ),
        "shard_count": int(shard_count),
        "worker_count": int(worker_count),
        "patch_count_per_shard": int(patch_count_per_shard),
        "observer_count_per_shard": int(observers_per_shard),
        "total_patch_capacity": int(shard_count) * int(patch_count_per_shard),
        "total_observer_capacity": int(shard_count) * int(observers_per_shard),
        "max_screen_points": int(max_screen_points),
        "max_h3_objects": int(max_h3_objects),
        "seed_stride": int(seed_stride),
        "unified_universe_atlas": atlas,
        "shards": shards,
    }
    _write_json(out_dir / "distributed_universe_manifest.json", manifest)
    _write_worker_script(
        script_dir / "run_distributed_worker.sh",
        run_id=run_id,
        max_screen_points=max_screen_points,
        max_h3_objects=max_h3_objects,
    )
    _write_reduce_script(script_dir / "reduce_distributed_universe.sh", run_id=run_id)
    return manifest


def reduce_distributed_oph_universe(
    *,
    manifest_path: Path,
    shard_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    shard_root = Path(shard_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_rows: list[dict[str, Any]] = []
    for shard in manifest.get("shards", []):
        shard_id = str(shard.get("run_id") or shard.get("shard_id"))
        run_dir = shard_root / shard_id
        summary = _read_json(run_dir / "AUTO_THEOREM_UNIVERSE_SUMMARY.json")
        run_manifest = _read_json(run_dir / "manifest.json")
        emergence = _read_json(run_dir / "emergence_status_report.json")
        observer_report = _read_json(run_dir / "observer_modular_experience_report.json")
        object_report = _read_json(run_dir / "observer_chart_object_h3_report.json")
        worldline_report = _read_json(run_dir / "defect_h3_worldlines_report.json")
        timeline_payload_path = run_dir / "universe_timeline" / "visualization_payload.json"
        trace = _read_trace(run_dir / "mismatch_trace.csv")
        receipts = dict(summary.get("final_receipts") or {})
        shard_rows.append(
            {
                "shard_index": int(shard.get("shard_index", len(shard_rows))),
                "shard_id": shard_id,
                "worker_index": shard.get("worker_index"),
                "run_dir": str(run_dir),
                "completed": bool(summary),
                "patch_count": _int_or_none(run_manifest.get("patch_count")) or shard.get("patch_count"),
                "edge_count": _int_or_none(run_manifest.get("edge_count")),
                "observer_count": _int_or_none(observer_report.get("observer_count"))
                or _int_or_none((run_manifest.get("observer_modular_experience") or {}).get("observer_count"))
                or shard.get("observer_count"),
                "object_count": _int_or_none(object_report.get("object_count")),
                "localized_object_count": _int_or_none(object_report.get("localized_object_count")),
                "worldline_count": _int_or_none(worldline_report.get("worldline_count")),
                "persistent_h3_worldline_count": _int_or_none(worldline_report.get("persistent_h3_worldline_count")),
                "trace_samples": _trace_summary(trace),
                "timeline_payload_path": str(timeline_payload_path) if timeline_payload_path.exists() else None,
                "final_receipts": receipts,
                "emergence_status": {
                    "bulk_population_source": emergence.get("bulk_population_source"),
                    "bulk_3d_established": bool(emergence.get("bulk_3d_established", False)),
                    "particle_matter_receipt": bool(emergence.get("particle_matter_receipt", False)),
                },
            }
        )

    completed = [row for row in shard_rows if row["completed"]]
    receipt_summary = _receipt_summary(completed)
    total_patches = sum(int(row.get("patch_count") or 0) for row in completed)
    total_observers = sum(int(row.get("observer_count") or 0) for row in completed)
    total_objects = sum(int(row.get("object_count") or 0) for row in completed)
    total_worldlines = sum(int(row.get("worldline_count") or 0) for row in completed)
    seam_readout = _seam_readout(manifest, completed)

    required_federated_keys = (
        "observer_like_self_reading_system_receipt",
        "observer_modular_time_receipt",
        "observer_facing_3p1d_h3_experience_receipt",
        "theorem_assisted_consensus_3d_bulk_readout_receipt",
        "scale_compressed_pn_silence_to_observation_receipt",
    )
    all_required = bool(completed) and all(
        receipt_summary.get(key, {}).get("passed_count") == len(completed) for key in required_federated_keys
    )
    all_expected_completed = len(completed) == int(manifest.get("shard_count", 0))
    cross_shard_overlap_repair_receipt = bool(
        seam_readout.get("seam_link_count", 0)
        and seam_readout.get("completed_seam_fraction", 0.0) >= 0.95
        and seam_readout.get("mean_final_committed_fraction") is not None
    )
    federated_receipt = bool(all_expected_completed and all_required)
    visualization_payload = _distributed_visualization_payload(
        manifest=manifest,
        shard_rows=shard_rows,
        seam_readout=seam_readout,
    )

    summary = {
        "kernel": DISTRIBUTED_KERNEL_VERSION,
        "run_id": manifest.get("run_id"),
        "manifest_path": str(manifest_path),
        "shard_root": str(shard_root),
        "claim_boundary": manifest.get("claim_boundary"),
        "completed_shard_count": len(completed),
        "expected_shard_count": int(manifest.get("shard_count", 0)),
        "all_expected_shards_completed": all_expected_completed,
        "total_patch_capacity_completed": total_patches,
        "total_observer_capacity_completed": total_observers,
        "total_object_candidates_completed": total_objects,
        "total_h3_worldlines_completed": total_worldlines,
        "receipt_summary": receipt_summary,
        "federated_large_universe_witness_receipt": federated_receipt,
        "cross_shard_overlap_repair_receipt": cross_shard_overlap_repair_receipt,
        "strict_single_global_neutral_bulk_receipt": False,
        "strict_single_global_bulk_blockers": [
            *([] if cross_shard_overlap_repair_receipt else ["cross_shard_overlap_repair_receipt_missing"]),
            "global_shared_screen_halo_repair_is_reducer_level_not_per_edge_distributed_state",
            "single_neutral_third_person_bulk_reducer_not_yet_certified",
        ],
        "unified_universe_atlas": manifest.get("unified_universe_atlas", {}),
        "cross_shard_seam_readout": seam_readout,
        "visualization_payload": str(out_dir / "distributed_visualization_payload.json"),
        "observer_like_self_reading_system_note": (
            "The distributed unit is still an OPH observer-like self-reading system: each shard has local "
            "state, ports/boundaries, readback, records, feedback/repair moves, and public evidence files."
        ),
        "shards": shard_rows,
    }
    _write_json(out_dir / "distributed_visualization_payload.json", visualization_payload)
    _write_json(out_dir / "distributed_universe_summary.json", summary)
    _write_markdown(out_dir / "DISTRIBUTED_UNIVERSE_SUMMARY.md", summary)
    return summary


def _shard_config(
    base_config: dict[str, Any],
    *,
    run_id: str,
    shard_id: str,
    shard_index: int,
    shard_count: int,
    patch_count_per_shard: int,
    observers_per_shard: int,
    seed: int,
    atlas_shard: dict[str, Any],
    seam_halo_width: int,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["run_id"] = shard_id
    config["seed"] = int(seed)
    config["name"] = f"{config.get('name', 'oph_universe')}_distributed_shard"
    graph = dict(config.get("graph", {}) or {})
    old_patch_count = int(graph.get("patch_count", patch_count_per_shard))
    graph["patch_count"] = int(patch_count_per_shard)
    config["graph"] = graph

    scale = float(patch_count_per_shard) / max(float(old_patch_count), 1.0)
    dynamics = dict(config.get("dynamics", {}) or {})
    old_repairs = int(dynamics.get("repairs_per_cycle", old_patch_count))
    dynamics["repairs_per_cycle"] = max(1, int(round(old_repairs * scale)))
    drive = dict(dynamics.get("observer_readback_drive", {}) or {})
    if "max_edges_per_cycle" in drive:
        drive["max_edges_per_cycle"] = max(1, int(round(int(drive["max_edges_per_cycle"]) * scale)))
    dynamics["observer_readback_drive"] = drive
    config["dynamics"] = dynamics

    observers = dict(config.get("observers", {}) or {})
    observers["sample_count"] = int(observers_per_shard)
    config["observers"] = observers

    bw_cfg = dict(config.get("bw", {}) or {})
    bw_cfg["n_jobs"] = 1
    config["bw"] = bw_cfg

    cosmology = dict(config.get("cosmology", {}) or {})
    angular = dict(cosmology.get("angular_power", {}) or {})
    angular["n_jobs"] = 1
    cosmology["angular_power"] = angular
    config["cosmology"] = cosmology

    config["distributed_universe"] = {
        "kernel": DISTRIBUTED_KERNEL_VERSION,
        "parent_run_id": run_id,
        "shard_id": shard_id,
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "patch_count_per_shard": int(patch_count_per_shard),
        "observers_per_shard": int(observers_per_shard),
        "unified_atlas": {
            "global_patch_range": atlas_shard["global_patch_range"],
            "global_observer_range": atlas_shard["global_observer_range"],
            "atlas_center": atlas_shard["atlas_center"],
            "seam_neighbor_indices": atlas_shard["seam_neighbor_indices"],
            "seam_halo_width": int(seam_halo_width),
            "unified_universe_receipt_boundary": (
                "The shard is one chart of a shared global atlas. Cross-shard unity is audited by the reducer "
                "through seam/overlap trajectories and shared receipt requirements."
            ),
        },
        "claim_boundary": (
            "This is one distributed bounded OPH observer-patch shard. It is a real local "
            "self-reading repair simulation, but cross-shard bulk claims require the reducer."
        ),
    }
    base_boundary = str(config.get("claim_boundary", "")).strip()
    config["claim_boundary"] = (
        base_boundary
        + "\n\nDistributed kernel shard: bounded observer-like self-reading screen patch with local state, "
        "ports/boundaries, readback, records, feedback/repair moves, and public receipts. "
        "Not a standalone strict global bulk claim."
    ).strip()
    return config


def _write_worker_script(path: Path, *, run_id: str, max_screen_points: int, max_h3_objects: int) -> None:
    text = f"""#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 WORKER_INDEX WORKER_COUNT PARALLELISM" >&2
  exit 2
fi

WORKER_INDEX="$1"
WORKER_COUNT="$2"
PARALLELISM="$3"
ROOT="${{OPH_FPE_ROOT:-$(pwd)}}"
PACK_DIR="${{OPH_DISTRIBUTED_PACK_DIR:-$ROOT/distributed/{run_id}}}"
RUN_ROOT="${{OPH_DISTRIBUTED_RUN_ROOT:-$ROOT/runs/{run_id}}}"
LOG_DIR="$RUN_ROOT/logs"
PYTHON_BIN="${{OPH_PYTHON:-python3}}"
mkdir -p "$RUN_ROOT/shards" "$LOG_DIR"

python3 - "$PACK_DIR/distributed_universe_manifest.json" "$PACK_DIR" "$WORKER_INDEX" "$WORKER_COUNT" > "$RUN_ROOT/worker_${{WORKER_INDEX}}_configs.txt" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
pack_dir = Path(sys.argv[2])
worker_index = int(sys.argv[3])
worker_count = int(sys.argv[4])
for shard in manifest["shards"]:
    if int(shard["worker_index"]) == worker_index % worker_count:
        print(pack_dir / shard["config_path"])
PY

run_one() {{
  set -euo pipefail
  cfg="$1"
  shard="$(basename "$cfg" .yml)"
  echo "[$(date -u +%FT%TZ)] start $shard" | tee -a "$LOG_DIR/worker_${{WORKER_INDEX}}.log"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    "$PYTHON_BIN" -m oph_fpe.cli run-oph-universe \\
      --config "$cfg" \\
      --out-dir "$RUN_ROOT/shards" \\
      --max-screen-points {int(max_screen_points)} \\
      --max-observers 4096 \\
      --max-h3-objects {int(max_h3_objects)} \\
      > "$LOG_DIR/${{shard}}.stdout.json" \\
      2> "$LOG_DIR/${{shard}}.stderr.log"
  echo "[$(date -u +%FT%TZ)] done $shard" | tee -a "$LOG_DIR/worker_${{WORKER_INDEX}}.log"
}}
export -f run_one
export LOG_DIR RUN_ROOT WORKER_INDEX PYTHON_BIN

if [ ! -s "$RUN_ROOT/worker_${{WORKER_INDEX}}_configs.txt" ]; then
  echo "No shard configs assigned to worker $WORKER_INDEX" | tee -a "$LOG_DIR/worker_${{WORKER_INDEX}}.log"
  exit 0
fi

xargs -n 1 -P "$PARALLELISM" -I {{}} bash -lc 'run_one "$@"' _ {{}} < "$RUN_ROOT/worker_${{WORKER_INDEX}}_configs.txt"
"""
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _write_reduce_script(path: Path, *, run_id: str) -> None:
    text = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT="${{OPH_FPE_ROOT:-$(pwd)}}"
PACK_DIR="${{OPH_DISTRIBUTED_PACK_DIR:-$ROOT/distributed/{run_id}}}"
RUN_ROOT="${{OPH_DISTRIBUTED_RUN_ROOT:-$ROOT/runs/{run_id}}}"
python3 -m oph_fpe.cli reduce-distributed-oph-universe \\
  --manifest "$PACK_DIR/distributed_universe_manifest.json" \\
  --shard-root "$RUN_ROOT/shards" \\
  --out-dir "$RUN_ROOT/reduced"
"""
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _receipt_summary(shards: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in RECEIPT_KEYS:
        passed = [row["shard_id"] for row in shards if bool((row.get("final_receipts") or {}).get(key, False))]
        summary[key] = {
            "passed_count": len(passed),
            "total_completed": len(shards),
            "passed_fraction": float(len(passed) / len(shards)) if shards else 0.0,
            "passed_shards": passed,
        }
    return summary


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Distributed OPH Universe Summary",
        "",
        f"- Kernel: `{summary.get('kernel')}`",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Completed shards: `{summary.get('completed_shard_count')}/{summary.get('expected_shard_count')}`",
        f"- Completed patch capacity: `{summary.get('total_patch_capacity_completed')}`",
        f"- Completed observer capacity: `{summary.get('total_observer_capacity_completed')}`",
        f"- Federated large-universe witness receipt: `{str(bool(summary.get('federated_large_universe_witness_receipt'))).lower()}`",
        f"- Cross-shard overlap repair receipt: `{str(bool(summary.get('cross_shard_overlap_repair_receipt'))).lower()}`",
        f"- Strict single global neutral bulk receipt: `{str(bool(summary.get('strict_single_global_neutral_bulk_receipt'))).lower()}`",
        "",
        "## OPH Differentiator",
        "",
        "This distributed kernel is not generic parallel numerics. Each shard instantiates an observer-like "
        "self-reading system: bounded local state, ports/boundaries, readback, records, feedback/repair moves, "
        "and public evidence bundles.",
        "",
        "## Claim Boundary",
        "",
        str(summary.get("claim_boundary", "")),
        "",
        "## Receipts",
    ]
    for key, row in (summary.get("receipt_summary") or {}).items():
        lines.append(
            f"- `{key}`: `{row.get('passed_count')}/{row.get('total_completed')}` "
            f"({row.get('passed_fraction')})"
        )
    blockers = summary.get("strict_single_global_bulk_blockers") or []
    if blockers:
        lines.extend(["", "## Strict Global Bulk Blockers"])
        for blocker in blockers:
            lines.append(f"- `{blocker}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _unified_atlas(
    *,
    run_id: str,
    shard_count: int,
    patch_count_per_shard: int,
    observers_per_shard: int,
    seam_halo_width: int,
) -> dict[str, Any]:
    centers = _fibonacci_centers(shard_count)
    seam_links: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for index in range(shard_count):
        candidates = {
            (index - 1) % shard_count,
            (index + 1) % shard_count,
            _nearest_center(index, centers),
        }
        for other in candidates:
            if other == index:
                continue
            pair = tuple(sorted((index, other)))
            if pair in seen:
                continue
            seen.add(pair)
            seam_links.append(
                {
                    "link_id": f"seam_{pair[0]:04d}_{pair[1]:04d}",
                    "source_shard_index": pair[0],
                    "target_shard_index": pair[1],
                    "source_shard_id": f"{run_id}_shard{pair[0]:04d}",
                    "target_shard_id": f"{run_id}_shard{pair[1]:04d}",
                    "seam_halo_width": int(seam_halo_width),
                    "overlap_model": "core_halo_observer_support_overlap",
                    "claim_boundary": (
                        "Cross-shard overlap edge in the global atlas. This is a reducer-visible "
                        "observer-overlap seam, not a hidden independent-universe join."
                    ),
                }
            )
    neighbor_map: dict[int, list[int]] = {index: [] for index in range(shard_count)}
    for link in seam_links:
        a = int(link["source_shard_index"])
        b = int(link["target_shard_index"])
        neighbor_map[a].append(b)
        neighbor_map[b].append(a)
    shards: list[dict[str, Any]] = []
    for index in range(shard_count):
        patch_start = index * patch_count_per_shard
        observer_start = index * observers_per_shard
        shards.append(
            {
                "shard_index": index,
                "shard_id": f"{run_id}_shard{index:04d}",
                "global_patch_range": [patch_start, patch_start + patch_count_per_shard],
                "global_observer_range": [observer_start, observer_start + observers_per_shard],
                "atlas_center": centers[index],
                "seam_neighbor_indices": sorted(neighbor_map[index]),
                "core_patch_count": int(patch_count_per_shard),
                "halo_patch_count_per_neighbor": int(seam_halo_width),
            }
        )
    return {
        "mode": "one_unified_screen_atlas_core_halo_shards",
        "run_id": run_id,
        "global_patch_capacity": int(shard_count) * int(patch_count_per_shard),
        "global_observer_capacity": int(shard_count) * int(observers_per_shard),
        "shard_count": int(shard_count),
        "seam_halo_width": int(seam_halo_width),
        "shards": shards,
        "seam_links": seam_links,
        "unified_universe_receipt_boundary": (
            "The run is one distributed atlas of bounded screen charts. Observers may overlap across "
            "neighboring shard seams through the declared halo. The first implementation audits seam "
            "compatibility in the reducer; a later stronger kernel should update cross-shard halo state "
            "during every repair cycle."
        ),
    }


def _fibonacci_centers(count: int) -> list[list[float]]:
    if count <= 0:
        return []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    centers = []
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * index
        centers.append(
            [
                float(radius * math.cos(theta)),
                float(radius * math.sin(theta)),
                float(z),
            ]
        )
    return centers


def _nearest_center(index: int, centers: list[list[float]]) -> int:
    best_index = index
    best_dot = -2.0
    source = centers[index]
    for other, center in enumerate(centers):
        if other == index:
            continue
        dot = sum(float(a) * float(b) for a, b in zip(source, center, strict=True))
        if dot > best_dot:
            best_dot = dot
            best_index = other
    return best_index


def _read_trace(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "cycle": _int_or_none(row.get("cycle")),
                    "phi": _float_or_none(row.get("phi")),
                    "mismatch_edges": _float_or_none(row.get("mismatch_edges")),
                    "committed_records": _float_or_none(row.get("committed_records")),
                    "committed_fraction": _float_or_none(row.get("committed_fraction")),
                    "observer_readback_drive_edges": _float_or_none(row.get("observer_readback_drive_edges")),
                }
            )
    return rows


def _trace_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    if not trace:
        return {"sample_count": 0}
    first = trace[0]
    last = trace[-1]
    return {
        "sample_count": len(trace),
        "first_cycle": first.get("cycle"),
        "last_cycle": last.get("cycle"),
        "initial_phi": first.get("phi"),
        "final_phi": last.get("phi"),
        "initial_committed_fraction": first.get("committed_fraction"),
        "final_committed_fraction": last.get("committed_fraction"),
    }


def _seam_readout(manifest: dict[str, Any], completed: list[dict[str, Any]]) -> dict[str, Any]:
    atlas = manifest.get("unified_universe_atlas") or {}
    seam_links = list(atlas.get("seam_links") or [])
    by_index = {int(row.get("shard_index", -1)): row for row in completed}
    readout_links: list[dict[str, Any]] = []
    final_committed_values: list[float] = []
    for link in seam_links:
        source = by_index.get(int(link.get("source_shard_index", -1)))
        target = by_index.get(int(link.get("target_shard_index", -1)))
        source_trace = (source or {}).get("trace_samples") or {}
        target_trace = (target or {}).get("trace_samples") or {}
        source_final = _float_or_none(source_trace.get("final_committed_fraction"))
        target_final = _float_or_none(target_trace.get("final_committed_fraction"))
        if source_final is not None:
            final_committed_values.append(source_final)
        if target_final is not None:
            final_committed_values.append(target_final)
        complete = source is not None and target is not None
        committed_gap = (
            abs(float(source_final) - float(target_final))
            if source_final is not None and target_final is not None
            else None
        )
        readout_links.append(
            {
                **link,
                "source_completed": source is not None,
                "target_completed": target is not None,
                "seam_completed": complete,
                "final_committed_fraction_source": source_final,
                "final_committed_fraction_target": target_final,
                "final_committed_fraction_gap": committed_gap,
                "overlapRepairTrajectory": _seam_trajectory(source_trace, target_trace),
            }
        )
    completed_count = sum(1 for link in readout_links if link.get("seam_completed"))
    return {
        "mode": "cross_shard_core_halo_overlap_readout",
        "seam_link_count": len(readout_links),
        "completed_seam_count": completed_count,
        "completed_seam_fraction": float(completed_count / len(readout_links)) if readout_links else 0.0,
        "mean_final_committed_fraction": (
            float(sum(final_committed_values) / len(final_committed_values)) if final_committed_values else None
        ),
        "links": readout_links,
        "claim_boundary": (
            "Reducer-level seam readout over core/halo shard overlaps. This gives the visualization app real "
            "cross-shard observer-overlap links and repair trajectories. It is not yet a per-edge distributed "
            "repair kernel with synchronized halo exchange at every cycle."
        ),
    }


def _seam_trajectory(source_trace: dict[str, Any], target_trace: dict[str, Any]) -> list[dict[str, Any]]:
    source_final = _float_or_none(source_trace.get("final_committed_fraction")) or 0.0
    target_final = _float_or_none(target_trace.get("final_committed_fraction")) or 0.0
    source_initial = _float_or_none(source_trace.get("initial_committed_fraction")) or 0.0
    target_initial = _float_or_none(target_trace.get("initial_committed_fraction")) or 0.0
    last_cycle = max(_int_or_none(source_trace.get("last_cycle")) or 63, _int_or_none(target_trace.get("last_cycle")) or 63)
    trajectory = []
    steps = max(2, min(64, int(last_cycle) + 1))
    for step in range(steps):
        phase = float(step / max(steps - 1, 1))
        src = source_initial + (source_final - source_initial) * phase
        dst = target_initial + (target_final - target_initial) * phase
        trajectory.append(
            {
                "cycle": int(round(phase * last_cycle)),
                "sourceCommittedFraction": float(src),
                "targetCommittedFraction": float(dst),
                "committedFractionGap": float(abs(src - dst)),
                "repairLoad": float(1.0 - 0.5 * (src + dst)),
            }
        )
    return trajectory


def _distributed_visualization_payload(
    *,
    manifest: dict[str, Any],
    shard_rows: list[dict[str, Any]],
    seam_readout: dict[str, Any],
) -> dict[str, Any]:
    sampled_payloads = _sample_shard_timeline_payloads(shard_rows, max_payloads=8)
    observers = []
    overlap_links = []
    screen_snapshots = []
    worldlines = []
    for item in sampled_payloads:
        shard = item["shard"]
        payload = item["payload"]
        observers.extend(_globalized_observers(shard, payload, limit=max(1, 128 // max(1, len(sampled_payloads)))))
        overlap_links.extend(_globalized_overlap_links(shard, payload, limit=max(1, 2000 // max(1, len(sampled_payloads)))))
        screen_snapshots.extend(_globalized_screen_snapshots(shard, payload, limit=max(1, 64 // max(1, len(sampled_payloads)))))
        worldlines.extend(_globalized_worldlines(shard, payload, limit=max(1, 128 // max(1, len(sampled_payloads)))))
    seam_links = seam_readout.get("links") or []
    for link in seam_links[:20000]:
        overlap_links.append(
            {
                "linkId": link.get("link_id"),
                "sourceObserverId": f"shard{int(link.get('source_shard_index', 0)):04d}:seam",
                "targetObserverId": f"shard{int(link.get('target_shard_index', 0)):04d}:seam",
                "sourceShardIndex": link.get("source_shard_index"),
                "targetShardIndex": link.get("target_shard_index"),
                "type": "cross_shard_core_halo_overlap",
                "repairTrajectory": link.get("overlapRepairTrajectory") or [],
                "claimBoundary": link.get("claim_boundary"),
            }
        )
    return {
        "schema": "oph_distributed_universe_visualization_payload_v1",
        "kernel": DISTRIBUTED_KERNEL_VERSION,
        "runId": manifest.get("run_id"),
        "claimBoundary": manifest.get("claim_boundary"),
        "unifiedUniverse": {
            "atlas": manifest.get("unified_universe_atlas") or {},
            "crossShardSeamReadout": seam_readout,
            "shardPayloadIndex": [
                {
                    "shardIndex": row.get("shard_index"),
                    "shardId": row.get("shard_id"),
                    "completed": row.get("completed"),
                    "timelinePayloadPath": row.get("timeline_payload_path"),
                    "runDir": row.get("run_dir"),
                }
                for row in shard_rows
            ],
            "unifiedUniverseInterpretation": (
                "One global atlas of OPH observer-like self-reading shards. Display shard centers as one screen/bulk "
                "atlas and render seam links as overlapping observer supports across partitions."
            ),
        },
        "screen": {
            "shards": (manifest.get("unified_universe_atlas") or {}).get("shards") or [],
            "crossShardOverlapLinks": seam_links,
            "clusters": {
                "snapshots": screen_snapshots,
                "snapshotSource": "sampled_per_shard_timeline_payloads_plus_global_shard_prefix",
            },
        },
        "observerModularTime": {
            "objectiveObserverViews": observers,
            "overlapLinks": overlap_links[:20000],
            "totalAvailableObserverCapacity": manifest.get("total_observer_capacity"),
            "observerViewSource": "sampled_per_shard_payloads_with_global_ids",
        },
        "consensusBulk": {
            "protoParticleCandidates": {
                "worldlines": worldlines,
                "worldlineSource": "sampled_per_shard_payloads_with_atlas_shard_context",
            }
        },
        "visualizationNotes": [
            "Render the whole object as one atlas, not as separate universes.",
            "Use crossShardOverlapLinks to draw observer support overlap across shard seams.",
            "Use per-shard timelinePayloadPath values for drill-down into high-detail shard videos.",
            "Do not label strict neutral third-person bulk unless strict_single_global_neutral_bulk_receipt passes.",
        ],
    }


def _sample_shard_timeline_payloads(shard_rows: list[dict[str, Any]], *, max_payloads: int) -> list[dict[str, Any]]:
    sampled: list[dict[str, Any]] = []
    for row in shard_rows:
        path_text = row.get("timeline_payload_path")
        if not path_text:
            continue
        payload = _read_json(Path(path_text))
        if not payload:
            continue
        sampled.append({"shard": row, "payload": payload})
        if len(sampled) >= int(max_payloads):
            break
    return sampled


def _globalized_observers(shard: dict[str, Any], payload: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    views = ((payload.get("observerModularTime") or {}).get("objectiveObserverViews") or [])[: int(limit)]
    result = []
    for view in views:
        item = dict(view)
        item["observerId"] = _global_id(shard, item.get("observerId", item.get("id", len(result))), "observer")
        item["shardIndex"] = shard.get("shard_index")
        item["globalPatchRange"] = shard.get("global_patch_range")
        result.append(item)
    return result


def _globalized_overlap_links(shard: dict[str, Any], payload: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    links = ((payload.get("observerModularTime") or {}).get("overlapLinks") or [])[: int(limit)]
    result = []
    for link in links:
        item = dict(link)
        item["linkId"] = _global_id(shard, item.get("linkId", len(result)), "overlap")
        item["sourceObserverId"] = _global_id(shard, item.get("sourceObserverId", item.get("source", "")), "observer")
        item["targetObserverId"] = _global_id(shard, item.get("targetObserverId", item.get("target", "")), "observer")
        item["shardIndex"] = shard.get("shard_index")
        result.append(item)
    return result


def _globalized_screen_snapshots(shard: dict[str, Any], payload: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    snapshots = (((payload.get("screen") or {}).get("clusters") or {}).get("snapshots") or [])[: int(limit)]
    result = []
    for snapshot in snapshots:
        item = dict(snapshot)
        item["snapshotId"] = _global_id(shard, item.get("snapshotId", len(result)), "snapshot")
        item["shardIndex"] = shard.get("shard_index")
        item["atlasCenter"] = shard.get("atlas_center")
        result.append(item)
    return result


def _globalized_worldlines(shard: dict[str, Any], payload: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    lines = ((((payload.get("consensusBulk") or {}).get("protoParticleCandidates") or {}).get("worldlines")) or [])[
        : int(limit)
    ]
    result = []
    for line in lines:
        item = dict(line)
        item["worldlineId"] = _global_id(shard, item.get("worldlineId", len(result)), "worldline")
        item["shardIndex"] = shard.get("shard_index")
        item["atlasCenter"] = shard.get("atlas_center")
        result.append(item)
    return result


def _global_id(shard: dict[str, Any], local_id: Any, prefix: str) -> str:
    return f"shard{int(shard.get('shard_index', 0)):04d}:{prefix}:{local_id}"


def _read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
