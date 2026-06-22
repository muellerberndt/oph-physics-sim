from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from oph_fpe.bulk.h3_worldline_stitch import h3_distance
from oph_fpe.experiments import load_config
from oph_fpe.observers.semantic_clock import (
    OBSERVER_KINDS,
    distributed_observer_uid,
    normalize_observer_frame,
    observer_registry_audit,
    semantic_history_digest,
)


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
    global_carrier = _write_global_carrier_artifacts(
        out_dir=out_dir,
        config_path=config_path,
        run_id=run_id,
        shard_count=shard_count,
        patch_count_per_shard=patch_count_per_shard,
        observers_per_shard=observers_per_shard,
        base_seed=base_seed,
    )
    shard_carriers = {
        int(row["shard_index"]): row
        for row in ((global_carrier.get("partition_map") or {}).get("shards") or [])
    }

    shards: list[dict[str, Any]] = []
    for shard_index in range(int(shard_count)):
        shard_id = f"{run_id}_shard{shard_index:04d}"
        atlas_shard = atlas["shards"][shard_index]
        carrier_shard = shard_carriers.get(shard_index, {})
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
            carrier_shard=carrier_shard,
            global_carrier_artifacts=global_carrier.get("artifacts", {}),
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
                "owned_nodes": carrier_shard.get("owned_nodes", []),
                "ghost_nodes": carrier_shard.get("ghost_nodes", []),
                "cut_edge_ids": carrier_shard.get("cut_edge_ids", []),
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
        "global_carrier": {
            "schema": "oph_distributed_global_carrier_contract_v1",
            "one_global_carrier_before_partition": True,
            "authoritative_owner_projection": "node_owner in partition_map.json is the authoritative copy rule",
            "physical_projection": (
                "Authoritative node states project to the monolithic finite quotient; worker queues, retries, "
                "checkpoints, and logs are implementation metadata."
            ),
            "artifacts": global_carrier.get("artifacts", {}),
            "expected_shard_ids": [row["shard_id"] for row in shards],
            "config_sha256": global_carrier.get("config_sha256"),
            "code_sha256": global_carrier.get("code_sha256"),
            "certificate_fields": global_carrier.get("certificate_fields", {}),
        },
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
    global_carrier = _global_carrier_contract(
        manifest_path=manifest_path,
        manifest=manifest,
        out_dir=out_dir / "global_carrier_contract",
    )

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
                "global_patch_range": shard.get("global_patch_range"),
                "global_observer_range": shard.get("global_observer_range"),
                "atlas_center": shard.get("atlas_center"),
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
    seam_metadata_replay_receipt = bool(
        seam_readout.get("seam_link_count", 0)
        and seam_readout.get("completed_seam_fraction", 0.0) >= 0.95
        and seam_readout.get("mean_final_committed_fraction") is not None
    )

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
    halo_exchange = _global_halo_exchange_reduction(
        manifest=manifest,
        completed=completed,
        seam_readout=seam_readout,
        out_dir=out_dir / "halo_exchange_global",
    )
    cross_shard_overlap_repair_receipt = bool(
        halo_exchange.get("online_cross_shard_overlap_repair_receipt", False)
    )
    neutral_global = _global_neutral_bulk_reduction(
        manifest=manifest,
        completed=completed,
        cross_shard_overlap_repair_receipt=cross_shard_overlap_repair_receipt,
        halo_exchange=halo_exchange,
        out_dir=out_dir / "strict_neutral_global",
    )
    observer_time_global = _global_observer_modular_time_export(
        manifest=manifest,
        shard_rows=shard_rows,
        out_dir=out_dir / "observer_modular_time_global",
    )
    proto_particle_global = _global_proto_particle_worldline_stitch(
        manifest=manifest,
        completed=completed,
        out_dir=out_dir / "proto_particles_global",
    )
    pn_global = _global_pn_resonance_reduction(
        manifest=manifest,
        completed=completed,
        receipt_summary=receipt_summary,
        out_dir=out_dir / "pn_resonance_global",
    )
    physical_cmb_global = _global_physical_cmb_reduction(
        manifest=manifest,
        completed=completed,
        out_dir=out_dir / "physical_cmb_global",
    )
    federated_receipt = bool(all_expected_completed and all_required)
    visualization_payload = _distributed_visualization_payload(
        manifest=manifest,
        shard_rows=shard_rows,
        seam_readout=seam_readout,
        physical_cmb_global=physical_cmb_global,
        global_carrier=global_carrier,
        halo_exchange=halo_exchange,
        neutral_global=neutral_global,
        observer_time_global=observer_time_global,
        proto_particle_global=proto_particle_global,
        pn_global=pn_global,
    )
    run_pack_contract = _distributed_run_pack_contract(
        manifest=manifest,
        all_expected_completed=all_expected_completed,
        federated_receipt=federated_receipt,
        seam_metadata_replay_receipt=seam_metadata_replay_receipt,
        global_carrier=global_carrier,
        halo_exchange=halo_exchange,
        neutral_global=neutral_global,
        observer_time_global=observer_time_global,
        proto_particle_global=proto_particle_global,
        pn_global=pn_global,
        physical_cmb_global=physical_cmb_global,
        visualization_payload_path=out_dir / "distributed_visualization_payload.json",
        out_dir=out_dir,
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
        "global_carrier_contract_receipt": bool(
            global_carrier.get("global_carrier_contract_receipt", False)
        ),
        "one_global_carrier_before_partition_receipt": bool(
            global_carrier.get("one_global_carrier_before_partition_receipt", False)
        ),
        "stable_global_identity_initial_state_receipt": bool(
            global_carrier.get("stable_global_identity_initial_state_receipt", False)
        ),
        "authoritative_owner_projection_receipt": bool(
            global_carrier.get("authoritative_owner_projection_receipt", False)
        ),
        "distributed_realization_event_certificate_receipt": bool(
            global_carrier.get("distributed_realization_event_certificate_receipt", False)
        ),
        "federated_large_universe_witness_receipt": federated_receipt,
        "seam_metadata_replay_receipt": seam_metadata_replay_receipt,
        "cross_shard_overlap_repair_receipt": cross_shard_overlap_repair_receipt,
        "online_cross_shard_overlap_repair_receipt": bool(
            halo_exchange.get("online_cross_shard_overlap_repair_receipt", False)
        ),
        "per_cycle_cross_shard_halo_exchange_receipt": bool(
            halo_exchange.get("per_cycle_cross_shard_halo_exchange_receipt", False)
        ),
        "reducer_halo_exchange_replay_receipt": bool(halo_exchange.get("reducer_halo_exchange_replay_receipt", False)),
        "global_observer_modular_time_export_receipt": bool(
            observer_time_global.get("global_observer_modular_time_export_receipt", False)
        ),
        "global_proto_particle_worldline_export_receipt": bool(
            proto_particle_global.get("global_proto_particle_worldline_export_receipt", False)
        ),
        "global_pn_resonance_receipt": bool(pn_global.get("global_pn_resonance_receipt", False)),
        "all_shards_local_scale_compressed_pn_witness_receipt": bool(
            pn_global.get("all_shards_local_scale_compressed_pn_witness_receipt", False)
        ),
        "global_physical_cmb_input_contract_receipt": bool(
            physical_cmb_global.get("physical_cmb_input_contract_receipt", False)
        ),
        "global_physical_cmb_output_comparison_receipt": bool(
            physical_cmb_global.get("physical_cmb_output_comparison_receipt", False)
        ),
        "global_physical_cmb_prediction_receipt": bool(
            physical_cmb_global.get("physical_cmb_prediction_receipt", False)
        ),
        "physical_cmb_global_reduction": physical_cmb_global,
        "strict_single_global_neutral_bulk_receipt": bool(
            neutral_global.get("strict_single_global_neutral_bulk_receipt", False)
        ),
        "strict_single_global_bulk_blockers": neutral_global.get("blockers") or [],
        "global_halo_exchange_reduction": halo_exchange,
        "global_neutral_bulk_reduction": neutral_global,
        "global_observer_modular_time_export": _compact_observer_time_report(observer_time_global),
        "global_proto_particle_worldlines": _compact_proto_particle_report(proto_particle_global),
        "global_pn_resonance_reduction": pn_global,
        "global_carrier_contract": global_carrier,
        "distributed_run_pack_contract": run_pack_contract,
        "unified_universe_atlas": manifest.get("unified_universe_atlas", {}),
        "cross_shard_seam_readout": seam_readout,
        "visualization_payload": str(out_dir / "distributed_visualization_payload.json"),
        "run_pack_contract_path": str(out_dir / "DISTRIBUTED_RUN_PACK_CONTRACT.json"),
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


def _global_halo_exchange_reduction(
    *,
    manifest: dict[str, Any],
    completed: list[dict[str, Any]],
    seam_readout: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    shard_roots = [Path(row["run_dir"]) for row in completed if row.get("completed")]
    reports = _collect_named_reports(shard_roots, "distributed_halo_exchange_report.json")
    expected = len(shard_roots)
    per_cycle = bool(
        expected
        and len(reports) == expected
        and all(
            bool(report.get("PER_CYCLE_HALO_EXCHANGE_RECEIPT", False))
            or bool(report.get("per_cycle_halo_exchange_receipt", False))
            for report in reports
        )
    )
    seam_links = list(seam_readout.get("links") or [])
    frame_rows = _global_halo_replay_frames(seam_links)
    replay_receipt = bool(seam_links and frame_rows)
    seam_metadata_replay_receipt = bool(seam_readout.get("seam_metadata_replay_receipt", replay_receipt))
    online_receipt_keys = (
        "SEAM_PACKET_RECIPROCITY_RECEIPT",
        "SEAM_VISIBLE_RESTRICTION_RECEIPT",
        "SEAM_REPAIR_DESCENT_RECEIPT",
        "SEAM_ATOMIC_COMMIT_RECEIPT",
        "DISTRIBUTED_LOCAL_DIAMOND_RECEIPT",
        "DISTRIBUTED_REPAIR_COMPLETENESS_RECEIPT",
        "CYCLE_HOLONOMY_ZERO_OR_CLASSIFIED_RECEIPT",
        "SELECTED_FIBER_NONTRIVIAL_ELIMINATION_RECEIPT",
        "SAME_BOUNDARY_MULTISTART_CONFLUENCE_RECEIPT",
        "QUOTIENT_NORMAL_FORM_CANONICAL_HASH_RECEIPT",
        "FAIR_BLOCK_CONTRACTION_RECEIPT",
        "SCHEDULE_INDEPENDENT_NORMAL_FORM_RECEIPT",
        "PARTITION_NATURALITY_RECEIPT",
    )
    online_receipts = {
        key: _all_reports_truthy(reports, key, expected)
        for key in online_receipt_keys
    }
    online_cross_shard_repair = bool(
        per_cycle
        and online_receipts["SEAM_PACKET_RECIPROCITY_RECEIPT"]
        and online_receipts["SEAM_VISIBLE_RESTRICTION_RECEIPT"]
        and online_receipts["SEAM_REPAIR_DESCENT_RECEIPT"]
        and online_receipts["SEAM_ATOMIC_COMMIT_RECEIPT"]
    )
    kernel_scaling_ready = bool(per_cycle and all(online_receipts.values()))
    blockers = []
    if not per_cycle:
        blockers.append("per_cycle_cross_shard_halo_exchange_receipt_missing")
    if not replay_receipt:
        blockers.append("reducer_halo_exchange_replay_empty")
    for key, passed in online_receipts.items():
        if not passed:
            blockers.append(f"{key.lower()}_missing")
    report = {
        "mode": "distributed_global_halo_exchange_reduction_v0",
        "run_id": manifest.get("run_id"),
        "expected_shard_count": int(manifest.get("shard_count", 0)),
        "completed_shard_count": expected,
        "source_report_count": len(reports),
        "per_cycle_cross_shard_halo_exchange_receipt": per_cycle,
        "online_cross_shard_overlap_repair_receipt": online_cross_shard_repair,
        "DISTRIBUTED_KERNEL_SCALING_READY_RECEIPT": kernel_scaling_ready,
        "required_online_seam_receipts": online_receipts,
        "seam_metadata_replay_receipt": seam_metadata_replay_receipt,
        "reducer_halo_exchange_replay_receipt": replay_receipt,
        "seam_link_count": len(seam_links),
        "replay_frame_count": len(frame_rows),
        "replay_frames": frame_rows[:256],
        "source_report_paths": [str(path / "distributed_halo_exchange_report.json") for path in shard_roots],
        "blockers": blockers,
        "claim_boundary": (
            "Reducer halo replay is synthetic audit/visualization metadata over completed shard traces. It is "
            "not live per-cycle halo exchange and cannot certify cross-shard OPH repair. Online seam receipts "
            "pass only when every shard emits reciprocal seam packets, visible restrictions, descent, atomic "
            "commit, diamond/completeness, holonomy, fair-block, schedule, and partition evidence."
        ),
    }
    _write_json(Path(out_dir) / "global_halo_exchange_report.json", report)
    return report


def _global_halo_replay_frames(seam_links: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cycle: dict[int, list[dict[str, Any]]] = {}
    for link in seam_links:
        for row in link.get("overlapRepairTrajectory") or []:
            cycle = _int_or_none(row.get("cycle"))
            if cycle is None:
                continue
            by_cycle.setdefault(cycle, []).append(
                {
                    "linkId": link.get("link_id"),
                    "sourceShardIndex": link.get("source_shard_index"),
                    "targetShardIndex": link.get("target_shard_index"),
                    "repairLoad": _float_or_none(row.get("repairLoad")),
                    "committedFractionGap": _float_or_none(row.get("committedFractionGap")),
                    "sourceCommittedFraction": _float_or_none(row.get("sourceCommittedFraction")),
                    "targetCommittedFraction": _float_or_none(row.get("targetCommittedFraction")),
                    "synthetic": True,
                    "source": "endpoint_interpolation",
                    "physics_receipt_eligible": False,
                }
            )
    frames = []
    for cycle in sorted(by_cycle):
        rows = by_cycle[cycle]
        loads = [float(row["repairLoad"]) for row in rows if row.get("repairLoad") is not None]
        gaps = [float(row["committedFractionGap"]) for row in rows if row.get("committedFractionGap") is not None]
        frames.append(
            {
                "cycle": cycle,
                "seamEdgeCount": len(rows),
                "meanRepairLoad": _mean_or_none(loads),
                "maxCommittedFractionGap": float(max(gaps)) if gaps else None,
                "links": rows[:512],
            }
        )
    return frames


def _all_reports_truthy(reports: list[dict[str, Any]], receipt_key: str, expected: int) -> bool:
    if not expected or len(reports) != expected:
        return False
    return all(_report_truthy(report, receipt_key) for report in reports)


def _report_truthy(report: dict[str, Any], receipt_key: str) -> bool:
    candidates = {
        receipt_key,
        receipt_key.lower(),
        receipt_key.lower().replace("_receipt", ""),
        receipt_key.replace("_RECEIPT", "_receipt"),
        receipt_key.replace("_RECEIPT", "_receipt").lower(),
    }
    return any(bool(report.get(key, False)) for key in candidates)


def _global_neutral_bulk_reduction(
    *,
    manifest: dict[str, Any],
    completed: list[dict[str, Any]],
    cross_shard_overlap_repair_receipt: bool,
    halo_exchange: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    shard_roots = [Path(row["run_dir"]) for row in completed if row.get("completed")]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    observer_views = _combined_observer_views(completed, max_total=4096, per_shard=512)
    combined_path = out_dir / "observer_views.jsonl"
    if observer_views:
        combined_path.write_text(
            "\n".join(json.dumps(row, default=str) for row in observer_views) + "\n",
            encoding="utf-8",
        )
    neutral_report: dict[str, Any] = {}
    frontier: dict[str, Any] = {}
    try:
        if observer_views:
            from oph_fpe.bulk.neutral_bulk import (
                write_strict_neutral_bulk_frontier_report,
                write_strict_neutral_bulk_report,
            )

            neutral_report = write_strict_neutral_bulk_report(
                out_dir,
                out=out_dir / "strict_neutral_bulk_report.json",
                seed=int(manifest.get("seed", 1) or 1),
                max_model_points=512,
                planted_control_points=160,
            )
            frontier = write_strict_neutral_bulk_frontier_report([out_dir, *shard_roots], out_dir)
        elif shard_roots:
            from oph_fpe.bulk.neutral_bulk import write_strict_neutral_bulk_frontier_report

            frontier = write_strict_neutral_bulk_frontier_report(shard_roots, out_dir)
    except Exception as exc:  # pragma: no cover - defensive fail-closed report path.
        neutral_report = {
            "mode": "distributed_global_neutral_bulk_reduction_error",
            "strict_neutral_bulk": False,
            "blockers": [f"global_neutral_bulk_reduction_failed:{type(exc).__name__}:{exc}"],
        }
    online_halo = bool(halo_exchange.get("per_cycle_cross_shard_halo_exchange_receipt", False))
    neutral_ready = bool(
        neutral_report.get("strict_neutral_bulk", False) or frontier.get("strict_neutral_bulk_ready", False)
    )
    strict_receipt = bool(neutral_ready and cross_shard_overlap_repair_receipt and online_halo)
    blockers = _unique_texts(
        list(neutral_report.get("blockers") or [])
        + list(frontier.get("blockers") or [])
        + ([] if observer_views else ["global_observer_views_missing"])
        + ([] if cross_shard_overlap_repair_receipt else ["online_cross_shard_overlap_repair_receipt_missing"])
        + ([] if online_halo else ["per_cycle_cross_shard_halo_exchange_receipt_missing"])
        + ([] if neutral_ready else ["global_strict_neutral_bulk_ready_false"])
    )
    report = {
        "mode": "distributed_global_neutral_bulk_reduction_v0",
        "run_id": manifest.get("run_id"),
        "expected_shard_count": int(manifest.get("shard_count", 0)),
        "completed_shard_count": len(shard_roots),
        "combined_observer_view_count": len(observer_views),
        "combined_observer_views_path": str(combined_path) if observer_views else None,
        "cross_shard_overlap_repair_receipt": bool(cross_shard_overlap_repair_receipt),
        "online_cross_shard_overlap_repair_receipt": bool(cross_shard_overlap_repair_receipt),
        "seam_metadata_replay_receipt": bool(halo_exchange.get("seam_metadata_replay_receipt", False)),
        "per_cycle_cross_shard_halo_exchange_receipt": online_halo,
        "global_strict_neutral_bulk_ready": neutral_ready,
        "strict_single_global_neutral_bulk_receipt": strict_receipt,
        "strict_neutral_bulk_report_path": str(out_dir / "strict_neutral_bulk_report.json"),
        "strict_neutral_bulk_frontier_path": str(out_dir / "strict_neutral_bulk_frontier_report.json"),
        "frontier": {
            "strict_neutral_bulk": bool(frontier.get("strict_neutral_bulk", False)),
            "strict_neutral_bulk_ready": bool(frontier.get("strict_neutral_bulk_ready", False)),
            "gate_rows": frontier.get("gate_rows") or [],
        },
        "blockers": blockers,
        "claim_boundary": (
            "Global neutral-bulk reduction over the distributed atlas. The strict single-bulk receipt requires "
            "a global neutral audit plus cross-shard overlap repair and live per-cycle halo exchange. "
            "A reducer-only replay cannot certify strict neutral third-person bulk."
        ),
    }
    _write_json(out_dir / "global_neutral_bulk_reduction_report.json", report)
    return report


def _global_observer_modular_time_export(
    *,
    manifest: dict[str, Any],
    shard_rows: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = _sample_shard_timeline_payloads(shard_rows, max_payloads=max(1, len(shard_rows)))
    observers: list[dict[str, Any]] = []
    overlap_links: list[dict[str, Any]] = []
    time_frame_counts: list[int] = []
    semantic_history_digests: list[str] = []
    clock_field_separation_count = 0
    visible_object_packet_count = 0
    visible_record_packet_count = 0
    for item in payloads:
        shard = item["shard"]
        payload = item["payload"]
        for view in _globalized_observers(shard, payload, limit=2048):
            frames = view.get("timeFrames") if isinstance(view.get("timeFrames"), list) else []
            time_frame_counts.append(len(frames))
            if bool(view.get("execution_clock_fields_separated_receipt", False)):
                clock_field_separation_count += 1
            semantic_history_digests.append(str(view.get("semantic_history_digest", "")))
            for frame in frames:
                if isinstance(frame, dict):
                    visible_object_packet_count += len(frame.get("visibleObjectPackets") or [])
                    visible_record_packet_count += len(frame.get("visibleRecordPackets") or [])
            observers.append(view)
        overlap_links.extend(_globalized_overlap_links(shard, payload, limit=20000))
    large_contract = bool(
        len(observers) >= 64
        and (min(time_frame_counts) if time_frame_counts else 0) >= 32
        and visible_object_packet_count > 0
        and visible_record_packet_count > 0
    )
    report = {
        "mode": "distributed_global_observer_modular_time_export_v0",
        "run_id": manifest.get("run_id"),
        "reportPath": str(out_dir / "observer_modular_time_global_payload.json"),
        "global_observer_modular_time_export_receipt": bool(observers and time_frame_counts),
        "execution_clock_fields_separated_receipt": bool(observers and clock_field_separation_count == len(observers)),
        "observer_clock_naturality_receipt": False,
        "semantic_history_digest_count": len([digest for digest in semantic_history_digests if digest]),
        "large_visualization_observer_contract_receipt": large_contract,
        "objectiveObserverViewCount": len(observers),
        "overlapLinkCount": len(overlap_links),
        "minTimeFrameCount": min(time_frame_counts) if time_frame_counts else 0,
        "maxTimeFrameCount": max(time_frame_counts) if time_frame_counts else 0,
        "visibleObjectPacketCount": visible_object_packet_count,
        "visibleRecordPacketCount": visible_record_packet_count,
        "objectiveObserverViews": observers[:4096],
        "overlapLinks": overlap_links[:100000],
        "blockers": _unique_texts(
            ([] if observers else ["objective_observer_views_missing"])
            + (
                []
                if bool(observers and clock_field_separation_count == len(observers))
                else ["execution_clock_fields_not_separated"]
            )
            + ([] if (min(time_frame_counts) if time_frame_counts else 0) >= 32 else ["observer_time_frames_below_32"])
            + ([] if visible_object_packet_count > 0 else ["visible_object_packets_missing"])
            + ([] if visible_record_packet_count > 0 else ["visible_record_packets_missing"])
            + ["observer_clock_naturality_certificate_missing"]
        ),
        "claim_boundary": (
            "Global observer modular-time export for visualization and finite evidence. It separates "
            "execution_epoch and scheduler_event_index from observer_record_order, observer_modular_parameter, "
            "and observer_clock_uncertainty. It does not by itself certify scheduler-independent observer "
            "clock naturality or strict neutral bulk."
        ),
    }
    _write_json(out_dir / "observer_modular_time_global_payload.json", report)
    return report


def _global_proto_particle_worldline_stitch(
    *,
    manifest: dict[str, Any],
    completed: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    worldlines: list[dict[str, Any]] = []
    stitched_groups: dict[str, list[str]] = {}
    for shard in completed:
        report = _read_json(Path(str(shard.get("run_dir"))) / "defect_h3_worldlines_report.json")
        for index, row in enumerate(report.get("worldlines") or []):
            if not isinstance(row, dict):
                continue
            converted = _globalized_worldline_row(shard, row, fallback_index=index)
            worldlines.append(converted)
            global_key = row.get("global_worldline_id") or row.get("stitch_key")
            if global_key is not None:
                stitched_groups.setdefault(str(global_key), []).append(converted["worldlineId"])
    moving = [row for row in worldlines if float(row.get("h3PathLength") or 0.0) > 0.0 or float(row.get("meanH3Step") or 0.0) > 0.0]
    localized = [row for row in worldlines if bool(row.get("bulkLocalizationPass", False))]
    stitched = {key: ids for key, ids in stitched_groups.items() if len(set(ids)) > 1}
    physical_stitched: dict[str, list[str]] = {}
    report = {
        "mode": "distributed_global_proto_particle_worldline_stitch_v0",
        "run_id": manifest.get("run_id"),
        "reportPath": str(out_dir / "global_proto_particle_worldlines_report.json"),
        "global_proto_particle_worldline_export_receipt": bool(worldlines),
        "moving_proto_particle_candidate_receipt": bool(moving),
        "cross_shard_worldline_id_collision_receipt": bool(stitched),
        "cross_shard_worldline_stitching_receipt": bool(physical_stitched),
        "particle_matter_receipt": False,
        "worldlineCount": len(worldlines),
        "movingWorldlineCount": len(moving),
        "bulkLocalizationPassCount": len(localized),
        "stitchedCrossShardWorldlineCount": len(physical_stitched),
        "idCollisionGroupCount": len(stitched),
        "stitchedGroups": physical_stitched,
        "idCollisionGroups": stitched,
        "worldlines": worldlines[:4096],
        "blockers": _unique_texts(
            ([] if worldlines else ["proto_particle_worldlines_missing"])
            + ([] if moving else ["moving_h3_worldlines_missing"])
            + ([] if localized else ["bulk_localization_pass_missing"])
            + ["cross_shard_worldline_stitching_transport_evidence_missing"]
            + ["particle_matter_gate_not_promoted"]
        ),
        "claim_boundary": (
            "Global proto-particle worldline export. These are H3-fitted holonomy/defect candidates. "
            "A repeated global ID or stitch key is only an ID-collision hint. Candidates become particles only "
            "after localization, support-visible chart transport, temporal/sector/holonomy continuity, "
            "fusion/scattering classification, repeated-seed, neutral-bulk, and repartition receipts pass."
        ),
    }
    _write_json(out_dir / "global_proto_particle_worldlines_report.json", report)
    return report


def _global_pn_resonance_reduction(
    *,
    manifest: dict[str, Any],
    completed: list[dict[str, Any]],
    receipt_summary: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_roots = [Path(row["run_dir"]) for row in completed if row.get("completed")]
    silence_reports = _collect_named_reports(shard_roots, "silence_to_observation_report.json")
    pn_reports = _collect_named_reports(shard_roots, "pn_resonance_report.json")
    receipt_row = receipt_summary.get("scale_compressed_pn_silence_to_observation_receipt") or {}
    local_pass_all = bool(completed and receipt_row.get("passed_count") == len(completed))
    report_receipts = [
        bool(report.get("scale_compressed_pn_silence_to_observation_receipt", False))
        or bool(report.get("PN_RESONANCE_RECEIPT", False))
        for report in silence_reports + pn_reports
    ]
    report_pass = bool(report_receipts and all(report_receipts))
    local_witness_receipt = bool(local_pass_all and report_pass)
    global_capacity_readback_map_receipt = False
    finite_capacity_fixed_point_receipt = False
    global_receipt = bool(global_capacity_readback_map_receipt and finite_capacity_fixed_point_receipt)
    blockers = _unique_texts(
        ([] if local_pass_all else ["shard_scale_compressed_pn_receipts_not_all_true"])
        + ([] if report_receipts else ["pn_or_silence_reports_missing"])
        + ([] if report_pass else ["pn_or_silence_report_receipts_not_all_true"])
        + ([] if global_capacity_readback_map_receipt else ["global_capacity_readback_map_missing"])
        + ([] if finite_capacity_fixed_point_receipt else ["finite_capacity_fixed_point_solve_missing"])
    )
    report = {
        "mode": "distributed_global_pn_resonance_reduction_v0",
        "run_id": manifest.get("run_id"),
        "global_pn_resonance_receipt": global_receipt,
        "all_shards_local_scale_compressed_pn_witness_receipt": local_witness_receipt,
        "global_capacity_readback_map_receipt": global_capacity_readback_map_receipt,
        "finite_capacity_fixed_point_receipt": finite_capacity_fixed_point_receipt,
        "local_scale_compressed_pn_receipt_all": local_pass_all,
        "silenceReportCount": len(silence_reports),
        "pnResonanceReportCount": len(pn_reports),
        "passedShardCount": int(receipt_row.get("passed_count") or 0),
        "completedShardCount": len(completed),
        "blockers": blockers,
        "claim_boundary": (
            "P/N reduction over distributed shards. Local scale-compressed silence-to-observation witnesses "
            "are not a global P/N resonance proof. The global receipt requires a finite normal-form/readback "
            "map F_r(N), a capacity fixed-point solve, uncertainty, wrong-P controls, and partition naturality."
        ),
    }
    _write_json(out_dir / "global_pn_resonance_report.json", report)
    return report


def _write_global_carrier_artifacts(
    *,
    out_dir: Path,
    config_path: Path,
    run_id: str,
    shard_count: int,
    patch_count_per_shard: int,
    observers_per_shard: int,
    base_seed: int,
) -> dict[str, Any]:
    carrier_dir = Path(out_dir) / "global_carrier"
    carrier_dir.mkdir(parents=True, exist_ok=True)
    shard_count = int(shard_count)
    patch_count_per_shard = int(patch_count_per_shard)
    observers_per_shard = int(observers_per_shard)
    total_nodes = shard_count * patch_count_per_shard
    total_observers = shard_count * observers_per_shard

    nodes = np.arange(total_nodes, dtype=np.int64)
    owners = np.array([_owner_for_node(int(node), patch_count_per_shard) for node in nodes], dtype=np.int64)
    edges = _global_graph_edges(total_nodes)
    for edge in edges:
        source_owner = int(owners[int(edge["source_node"])])
        target_owner = int(owners[int(edge["target_node"])])
        edge["source_owner"] = source_owner
        edge["target_owner"] = target_owner
        edge["is_cut_edge"] = source_owner != target_owner
    edge_array = np.array([[edge["source_node"], edge["target_node"]] for edge in edges], dtype=np.int64)
    cut_edge_mask = np.array([bool(edge["is_cut_edge"]) for edge in edges], dtype=np.bool_)
    state_words = np.array(
        [_stable_uint63(run_id, "patch", int(node), base_seed) for node in nodes],
        dtype=np.int64,
    )
    boundary_sector = np.array([int(value % 17) for value in state_words], dtype=np.int64)

    graph_path = carrier_dir / "global_graph.npz"
    np.savez_compressed(
        graph_path,
        nodes=nodes,
        edges=edge_array,
        node_owner=owners,
        cut_edge_mask=cut_edge_mask,
    )
    state_path = carrier_dir / "global_initial_state.npz"
    np.savez_compressed(
        state_path,
        nodes=nodes,
        canonical_state_word=state_words,
        boundary_sector=boundary_sector,
    )

    cut_edges = [edge for edge in edges if edge["is_cut_edge"]]
    partition_map = _partition_map_payload(
        run_id=run_id,
        shard_count=shard_count,
        patch_count_per_shard=patch_count_per_shard,
        observers_per_shard=observers_per_shard,
        owners=owners,
        cut_edges=cut_edges,
    )
    cut_interfaces = _cut_interfaces_payload(run_id=run_id, cut_edges=cut_edges)
    observer_registry = _observer_registry_payload(
        run_id=run_id,
        shard_count=shard_count,
        patch_count_per_shard=patch_count_per_shard,
        observers_per_shard=observers_per_shard,
        base_seed=base_seed,
    )

    partition_path = carrier_dir / "partition_map.json"
    cut_path = carrier_dir / "cut_interfaces.json"
    registry_path = carrier_dir / "global_observer_registry.json"
    _write_json(partition_path, partition_map)
    _write_json(cut_path, cut_interfaces)
    _write_json(registry_path, observer_registry)

    artifacts = {
        "global_graph": _artifact_row(graph_path, out_dir),
        "global_initial_state": _artifact_row(state_path, out_dir),
        "partition_map": _artifact_row(partition_path, out_dir),
        "cut_interfaces": _artifact_row(cut_path, out_dir),
        "global_observer_registry": _artifact_row(registry_path, out_dir),
    }
    return {
        "artifacts": artifacts,
        "partition_map": partition_map,
        "cut_interfaces": cut_interfaces,
        "global_observer_registry": observer_registry,
        "config_sha256": _file_sha256(Path(config_path)),
        "code_sha256": _file_sha256(Path(__file__)),
        "certificate_fields": {
            "global_graph": "global_graph.npz",
            "global_initial_state": "global_initial_state.npz",
            "partition_map": "partition_map.json",
            "cut_interfaces": "cut_interfaces.json",
            "global_observer_registry": "global_observer_registry.json",
            "linearized_committed_event_log": "required for theorem-grade distributed realization",
            "rollback_checkpoint_roots": "required for restart/replay invariance",
            "monolithic_normal_form_certificate": "required for final equality to n_r(q_0)",
            "final_readout_recomputation": "required for observer-readout equality",
        },
    }


def _global_carrier_contract(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    carrier = manifest.get("global_carrier") or {}
    artifacts = carrier.get("artifacts") if isinstance(carrier.get("artifacts"), dict) else {}
    required = (
        "global_graph",
        "global_initial_state",
        "partition_map",
        "cut_interfaces",
        "global_observer_registry",
    )
    blockers: list[str] = []
    artifact_status: dict[str, Any] = {}
    resolved_paths: dict[str, Path] = {}
    manifest_dir = Path(manifest_path).parent

    for key in required:
        row = artifacts.get(key) if isinstance(artifacts, dict) else None
        if not isinstance(row, dict):
            artifact_status[key] = {"present": False, "blocker": "manifest_artifact_entry_missing"}
            blockers.append(f"{key}_manifest_artifact_entry_missing")
            continue
        path_text = str(row.get("path") or "")
        artifact_path = Path(path_text)
        if not artifact_path.is_absolute():
            artifact_path = manifest_dir / artifact_path
        resolved_paths[key] = artifact_path
        exists = artifact_path.exists()
        sha = _file_sha256(artifact_path) if exists else None
        expected_sha = row.get("sha256")
        hash_ok = bool(exists and expected_sha and sha == expected_sha)
        artifact_status[key] = {
            "path": str(artifact_path),
            "declared_path": path_text,
            "exists": exists,
            "sha256": sha,
            "expected_sha256": expected_sha,
            "hash_receipt": hash_ok,
        }
        if not exists:
            blockers.append(f"{key}_missing")
        elif not hash_ok:
            blockers.append(f"{key}_hash_mismatch")

    graph_info = _npz_graph_info(resolved_paths.get("global_graph"))
    state_info = _npz_state_info(resolved_paths.get("global_initial_state"))
    partition = _read_json(resolved_paths.get("partition_map", Path()))
    cut = _read_json(resolved_paths.get("cut_interfaces", Path()))
    registry = _read_json(resolved_paths.get("global_observer_registry", Path()))
    config_path = Path(str(manifest.get("config_path") or ""))
    if config_path and not config_path.is_absolute():
        config_path = manifest_dir / config_path
    config_sha = _file_sha256(config_path) if config_path else None
    config_hash_receipt = bool(config_sha and config_sha == carrier.get("config_sha256"))
    if not config_hash_receipt:
        blockers.append("config_hash_mismatch_or_missing")
    code_sha = _file_sha256(Path(__file__))
    code_hash_receipt = bool(code_sha and code_sha == carrier.get("code_sha256"))
    if not code_hash_receipt:
        blockers.append("code_hash_mismatch_or_missing")
    run_id_receipt = bool(manifest.get("run_id") and partition.get("run_id") == manifest.get("run_id"))
    if not run_id_receipt:
        blockers.append("run_id_mismatch")

    expected_shard_ids = [str(row.get("shard_id") or row.get("run_id")) for row in manifest.get("shards", [])]
    partition_shard_ids = [str(row.get("shard_id")) for row in partition.get("shards", [])]
    shard_ids_match = bool(expected_shard_ids and partition_shard_ids == expected_shard_ids)
    if not shard_ids_match:
        blockers.append("partition_shard_ids_do_not_match_manifest")

    total_nodes = int(manifest.get("total_patch_capacity", 0) or 0)
    total_observers = int(manifest.get("total_observer_capacity", 0) or 0)
    graph_receipt = bool(
        graph_info.get("load_receipt")
        and graph_info.get("node_count") == total_nodes
        and graph_info.get("edge_count", 0) >= max(total_nodes - 1, 0)
    )
    if not graph_receipt:
        blockers.append("global_graph_shape_invalid")
    state_receipt = bool(
        state_info.get("load_receipt")
        and state_info.get("node_count") == total_nodes
        and state_info.get("stable_identity_rule") == "blake2b_run_patch_base_seed_uint63"
    )
    if not state_receipt:
        blockers.append("global_initial_state_invalid")
    partition_receipt = bool(
        partition.get("schema") == "oph_distributed_partition_map_v1"
        and partition.get("node_count") == total_nodes
        and partition.get("synthetic_partition") is False
        and shard_ids_match
    )
    if not partition_receipt:
        blockers.append("partition_map_invalid")
    cut_edges = cut.get("cut_edges") if isinstance(cut.get("cut_edges"), list) else []
    cut_required = int(manifest.get("shard_count", 0) or 0) > 1
    cut_receipt = bool(
        cut.get("schema") == "oph_distributed_cut_interfaces_v1"
        and cut.get("synthetic_physics_receipt") is False
        and (bool(cut_edges) if cut_required else True)
    )
    if not cut_receipt:
        blockers.append("cut_interfaces_invalid")
    observer_registry_receipt = bool(
        registry.get("schema") == "oph_global_observer_registry_v2"
        and registry.get("observer_count") == total_observers
        and registry.get("stable_identity_rule") == "blake2b_run_observer_kind_global_index_base_seed_uint63"
        and registry.get("observer_kinds") == list(OBSERVER_KINDS)
        and registry.get("registered_identity_count") == total_observers * len(OBSERVER_KINDS)
        and bool(registry.get("global_observer_registry_namespace_receipt", False))
    )
    if not observer_registry_receipt:
        blockers.append("global_observer_registry_invalid")

    artifact_receipt = all(bool((artifact_status.get(key) or {}).get("hash_receipt")) for key in required)
    owner_receipt = bool(partition_receipt and graph_receipt and cut_receipt)
    event_certificate = carrier.get("event_certificate") if isinstance(carrier.get("event_certificate"), dict) else {}
    event_receipt = bool(
        event_certificate.get("linearized_committed_event_log_receipt", False)
        and event_certificate.get("monolithic_normal_form_certificate_receipt", False)
        and event_certificate.get("final_readout_recomputed_receipt", False)
    )
    if not event_receipt:
        blockers.extend(
            [
                "linearized_committed_event_log_missing",
                "monolithic_normal_form_certificate_missing",
                "final_readout_recomputed_receipt_missing",
            ]
        )

    contract_receipt = bool(
        carrier.get("one_global_carrier_before_partition", False)
        and artifact_receipt
        and graph_receipt
        and state_receipt
        and partition_receipt
        and cut_receipt
        and observer_registry_receipt
        and owner_receipt
        and config_hash_receipt
        and code_hash_receipt
        and run_id_receipt
    )
    report = {
        "mode": "distributed_global_carrier_contract_v1",
        "run_id": manifest.get("run_id"),
        "global_carrier_contract_receipt": contract_receipt,
        "one_global_carrier_before_partition_receipt": bool(
            carrier.get("one_global_carrier_before_partition", False) and artifact_receipt
        ),
        "manifest_declared_global_artifacts_receipt": artifact_receipt,
        "global_graph_receipt": graph_receipt,
        "partition_map_receipt": partition_receipt,
        "cut_interface_receipt": cut_receipt,
        "stable_global_identity_initial_state_receipt": state_receipt,
        "global_observer_registry_receipt": observer_registry_receipt,
        "authoritative_owner_projection_receipt": owner_receipt,
        "config_hash_receipt": config_hash_receipt,
        "code_hash_receipt": code_hash_receipt,
        "run_id_receipt": run_id_receipt,
        "distributed_realization_event_certificate_receipt": event_receipt,
        "monolithic_normal_form_certificate_receipt": bool(
            event_certificate.get("monolithic_normal_form_certificate_receipt", False)
        ),
        "final_readout_recomputed_receipt": bool(
            event_certificate.get("final_readout_recomputed_receipt", False)
        ),
        "artifact_status": artifact_status,
        "graph_info": graph_info,
        "state_info": state_info,
        "expected_shard_ids": expected_shard_ids,
        "partition_shard_ids": partition_shard_ids,
        "cut_edge_count": len(cut_edges),
        "observer_count": registry.get("observer_count"),
        "blockers": _unique_texts(blockers),
        "claim_boundary": (
            "This verifies that the run pack declares one finite global OPH carrier before worker "
            "partitioning and that shard ownership, cut interfaces, stable initial identities, and "
            "observer registry artifacts match their manifest hashes. It does not certify distributed "
            "realization of the monolithic normal-form map unless committed events are linearized, "
            "rollbacks return to prior committed roots, and final readouts are recomputed from the "
            "projected monolithic state."
        ),
    }
    _write_json(out_dir / "GLOBAL_CARRIER_CONTRACT.json", report)
    return report


def _global_graph_edges(node_count: int) -> list[dict[str, Any]]:
    if node_count <= 1:
        return []
    pairs = [(node, node + 1) for node in range(node_count - 1)]
    if node_count > 2:
        pairs.append((node_count - 1, 0))
    edges = []
    for index, (source, target) in enumerate(pairs):
        edges.append(
            {
                "edge_id": f"e{index:06d}",
                "source_node": int(source),
                "target_node": int(target),
                "is_cut_edge": False,
            }
        )
    return edges


def _partition_map_payload(
    *,
    run_id: str,
    shard_count: int,
    patch_count_per_shard: int,
    observers_per_shard: int,
    owners: np.ndarray,
    cut_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    edge_ids_by_shard: dict[int, list[str]] = {index: [] for index in range(shard_count)}
    ghost_nodes_by_shard: dict[int, set[int]] = {index: set() for index in range(shard_count)}
    for edge in cut_edges:
        source = int(edge["source_node"])
        target = int(edge["target_node"])
        source_owner = int(edge["source_owner"])
        target_owner = int(edge["target_owner"])
        edge_ids_by_shard[source_owner].append(str(edge["edge_id"]))
        edge_ids_by_shard[target_owner].append(str(edge["edge_id"]))
        ghost_nodes_by_shard[source_owner].add(target)
        ghost_nodes_by_shard[target_owner].add(source)

    shards = []
    for index in range(shard_count):
        patch_start = index * patch_count_per_shard
        observer_start = index * observers_per_shard
        owned = list(range(patch_start, patch_start + patch_count_per_shard))
        shards.append(
            {
                "shard_index": index,
                "shard_id": f"{run_id}_shard{index:04d}",
                "owned_nodes": owned,
                "ghost_nodes": sorted(ghost_nodes_by_shard[index]),
                "cut_edge_ids": sorted(edge_ids_by_shard[index]),
                "global_patch_range": [patch_start, patch_start + patch_count_per_shard],
                "global_observer_range": [observer_start, observer_start + observers_per_shard],
            }
        )
    return {
        "schema": "oph_distributed_partition_map_v1",
        "run_id": run_id,
        "node_count": int(len(owners)),
        "shard_count": int(shard_count),
        "node_owner": [int(value) for value in owners.tolist()],
        "synthetic_partition": False,
        "owner_rule": "contiguous_global_patch_ranges_declared_before_shard_configs",
        "shards": shards,
    }


def _cut_interfaces_payload(*, run_id: str, cut_edges: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for edge in cut_edges:
        source = int(edge["source_node"])
        target = int(edge["target_node"])
        source_owner = int(edge["source_owner"])
        target_owner = int(edge["target_owner"])
        rows.append(
            {
                "edge_id": edge["edge_id"],
                "source_node": source,
                "target_node": target,
                "source_owner": source_owner,
                "target_owner": target_owner,
                "owner_pair": [source_owner, target_owner],
                "restriction_maps": {
                    "source_to_target_visible_boundary": [source, target],
                    "target_to_source_visible_boundary": [target, source],
                },
                "authoritative_commit_rule": (
                    "A cross-cut physical update is admissible only with a linearization witness whose "
                    "projection is a legal monolithic repair path or a physical stutter."
                ),
            }
        )
    return {
        "schema": "oph_distributed_cut_interfaces_v1",
        "run_id": run_id,
        "cut_edge_count": len(rows),
        "synthetic_physics_receipt": False,
        "cut_edges": rows,
        "claim_boundary": (
            "Cut interfaces are derived from declared global graph edges whose endpoint owners differ. "
            "They replace visualization-only atlas seams as the physics-facing boundary contract."
        ),
    }


def _observer_registry_payload(
    *,
    run_id: str,
    shard_count: int,
    patch_count_per_shard: int,
    observers_per_shard: int,
    base_seed: int,
) -> dict[str, Any]:
    shards = []
    samples = []
    registry_entries: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        observer_start = shard_index * observers_per_shard
        patch_start = shard_index * patch_count_per_shard
        observer_stop = observer_start + observers_per_shard
        shards.append(
            {
                "shard_index": shard_index,
                "shard_id": f"{run_id}_shard{shard_index:04d}",
                "global_observer_range": [observer_start, observer_stop],
                "home_patch_rule": "namespaced local_anchor_patch_id = patch:((2*local_observer_index+1) modulo patch_count_per_shard)",
                "home_patch_range": [patch_start, patch_start + patch_count_per_shard],
            }
        )
        for local_observer_index in range(observers_per_shard):
            global_observer_index = observer_start + local_observer_index
            local_anchor_index = (2 * local_observer_index + 1) % max(int(patch_count_per_shard), 1)
            global_anchor_index = patch_start + local_anchor_index
            for observer_kind in OBSERVER_KINDS:
                uid = distributed_observer_uid(
                    run_id=run_id,
                    observer_kind=observer_kind,
                    global_observer_index=global_observer_index,
                )
                entry = {
                    "distributed_observer_uid": uid,
                    "global_observer_index": global_observer_index,
                    "observer_kind": observer_kind,
                    "local_observer_index": local_observer_index,
                    "local_anchor_patch_id": f"patch:{local_anchor_index}",
                    "global_anchor_patch_id": f"patch:{global_anchor_index}",
                    "owner_shard": shard_index,
                    "lineage_parent_uid": None if observer_kind == "patch" else distributed_observer_uid(
                        run_id=run_id,
                        observer_kind="patch",
                        global_observer_index=global_observer_index,
                    ),
                    "stable_identity": _stable_uint63(run_id, observer_kind, global_observer_index, base_seed),
                }
                registry_entries.append(entry)
                if len(samples) < max(1, min(12, shard_count * max(observers_per_shard, 1))):
                    samples.append(entry)
    audit = observer_registry_audit(registry_entries)
    return {
        "schema": "oph_global_observer_registry_v2",
        "run_id": run_id,
        "observer_count": shard_count * observers_per_shard,
        "registered_identity_count": len(registry_entries),
        "observer_kinds": list(OBSERVER_KINDS),
        "namespace_rule": "distributed_observer_uid = run_id:observer_kind:global_observer_index",
        "stable_identity_rule": "blake2b_run_observer_kind_global_index_base_seed_uint63",
        "registry_namespace_audit": audit,
        "global_observer_registry_namespace_receipt": bool(audit.get("GLOBAL_OBSERVER_REGISTRY_NAMESPACE_RECEIPT", False)),
        "shards": shards,
        "sample_observers": samples,
        "claim_boundary": (
            "Global observer registry with disjoint patch/cap/future namespaces and explicit anchor-patch "
            "IDs. It is an evidence contract for finite distributed runs, not a proof of dynamic observer "
            "identity across all OPH branches."
        ),
    }


def _owner_for_node(node: int, patch_count_per_shard: int) -> int:
    return int(node // max(int(patch_count_per_shard), 1))


def _global_cut_pairs(shard_count: int) -> list[tuple[int, int]]:
    if shard_count <= 1:
        return []
    pairs = []
    seen: set[tuple[int, int]] = set()
    for index in range(shard_count):
        pair = tuple(sorted((index, (index + 1) % shard_count)))
        if pair in seen or pair[0] == pair[1]:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _artifact_row(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(Path(path).relative_to(root)),
        "sha256": _file_sha256(path),
        "required": True,
    }


def _stable_uint63(*parts: Any) -> int:
    digest = hashlib.blake2b(
        "|".join(str(part) for part in parts).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _npz_graph_info(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {"load_receipt": False}
    try:
        with np.load(path) as data:
            nodes = data["nodes"]
            edges = data["edges"]
            owners = data["node_owner"]
            cut_mask = data["cut_edge_mask"]
            return {
                "load_receipt": True,
                "node_count": int(len(nodes)),
                "edge_count": int(len(edges)),
                "owner_count": int(len(set(int(value) for value in owners.tolist()))),
                "cut_edge_count": int(np.count_nonzero(cut_mask)),
            }
    except Exception as exc:  # pragma: no cover - defensive artifact report path.
        return {"load_receipt": False, "error": f"{type(exc).__name__}:{exc}"}


def _npz_state_info(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {"load_receipt": False}
    try:
        with np.load(path) as data:
            nodes = data["nodes"]
            state = data["canonical_state_word"]
            sectors = data["boundary_sector"]
            return {
                "load_receipt": bool(len(nodes) == len(state) == len(sectors)),
                "node_count": int(len(nodes)),
                "stable_identity_rule": "blake2b_run_patch_base_seed_uint63",
                "state_word_dtype": str(state.dtype),
            }
    except Exception as exc:  # pragma: no cover - defensive artifact report path.
        return {"load_receipt": False, "error": f"{type(exc).__name__}:{exc}"}


def _distributed_run_pack_contract(
    *,
    manifest: dict[str, Any],
    all_expected_completed: bool,
    federated_receipt: bool,
    seam_metadata_replay_receipt: bool,
    global_carrier: dict[str, Any],
    halo_exchange: dict[str, Any],
    neutral_global: dict[str, Any],
    observer_time_global: dict[str, Any],
    proto_particle_global: dict[str, Any],
    pn_global: dict[str, Any],
    physical_cmb_global: dict[str, Any],
    visualization_payload_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    online_receipts = halo_exchange.get("required_online_seam_receipts") or {}
    gates = {
        "all_expected_shards_completed": bool(all_expected_completed),
        "federated_large_universe_witness_receipt": bool(federated_receipt),
        "global_carrier_contract_receipt": bool(
            global_carrier.get("global_carrier_contract_receipt", False)
        ),
        "one_global_carrier_before_partition_receipt": bool(
            global_carrier.get("one_global_carrier_before_partition_receipt", False)
        ),
        "manifest_declared_global_artifacts_receipt": bool(
            global_carrier.get("manifest_declared_global_artifacts_receipt", False)
        ),
        "partition_map_receipt": bool(global_carrier.get("partition_map_receipt", False)),
        "cut_interface_receipt": bool(global_carrier.get("cut_interface_receipt", False)),
        "stable_global_identity_initial_state_receipt": bool(
            global_carrier.get("stable_global_identity_initial_state_receipt", False)
        ),
        "global_observer_registry_receipt": bool(
            global_carrier.get("global_observer_registry_receipt", False)
        ),
        "authoritative_owner_projection_receipt": bool(
            global_carrier.get("authoritative_owner_projection_receipt", False)
        ),
        "distributed_realization_event_certificate_receipt": bool(
            global_carrier.get("distributed_realization_event_certificate_receipt", False)
        ),
        "seam_metadata_replay_receipt": bool(seam_metadata_replay_receipt),
        "online_cross_shard_overlap_repair_receipt": bool(
            halo_exchange.get("online_cross_shard_overlap_repair_receipt", False)
        ),
        "per_cycle_cross_shard_halo_exchange_receipt": bool(
            halo_exchange.get("per_cycle_cross_shard_halo_exchange_receipt", False)
        ),
        "distributed_local_diamond_receipt": bool(online_receipts.get("DISTRIBUTED_LOCAL_DIAMOND_RECEIPT", False)),
        "distributed_repair_completeness_receipt": bool(
            online_receipts.get("DISTRIBUTED_REPAIR_COMPLETENESS_RECEIPT", False)
        ),
        "cycle_holonomy_zero_or_classified_receipt": bool(
            online_receipts.get("CYCLE_HOLONOMY_ZERO_OR_CLASSIFIED_RECEIPT", False)
        ),
        "selected_fiber_nontrivial_elimination_receipt": bool(
            online_receipts.get("SELECTED_FIBER_NONTRIVIAL_ELIMINATION_RECEIPT", False)
        ),
        "same_boundary_multistart_confluence_receipt": bool(
            online_receipts.get("SAME_BOUNDARY_MULTISTART_CONFLUENCE_RECEIPT", False)
        ),
        "quotient_normal_form_canonical_hash_receipt": bool(
            online_receipts.get("QUOTIENT_NORMAL_FORM_CANONICAL_HASH_RECEIPT", False)
        ),
        "fair_block_contraction_receipt": bool(online_receipts.get("FAIR_BLOCK_CONTRACTION_RECEIPT", False)),
        "schedule_independent_normal_form_receipt": bool(
            online_receipts.get("SCHEDULE_INDEPENDENT_NORMAL_FORM_RECEIPT", False)
        ),
        "partition_naturality_receipt": bool(online_receipts.get("PARTITION_NATURALITY_RECEIPT", False)),
        "global_observer_modular_time_export_receipt": bool(
            observer_time_global.get("global_observer_modular_time_export_receipt", False)
        ),
        "large_visualization_observer_contract_receipt": bool(
            observer_time_global.get("large_visualization_observer_contract_receipt", False)
        ),
        "global_proto_particle_worldline_export_receipt": bool(
            proto_particle_global.get("global_proto_particle_worldline_export_receipt", False)
        ),
        "moving_proto_particle_candidate_receipt": bool(
            proto_particle_global.get("moving_proto_particle_candidate_receipt", False)
        ),
        "cross_shard_worldline_stitching_receipt": bool(
            proto_particle_global.get("cross_shard_worldline_stitching_receipt", False)
        ),
        "all_shards_local_scale_compressed_pn_witness_receipt": bool(
            pn_global.get("all_shards_local_scale_compressed_pn_witness_receipt", False)
        ),
        "global_capacity_readback_map_receipt": bool(pn_global.get("global_capacity_readback_map_receipt", False)),
        "finite_capacity_fixed_point_receipt": bool(pn_global.get("finite_capacity_fixed_point_receipt", False)),
        "global_pn_resonance_receipt": bool(pn_global.get("global_pn_resonance_receipt", False)),
        "strict_single_global_neutral_bulk_receipt": bool(
            neutral_global.get("strict_single_global_neutral_bulk_receipt", False)
        ),
        "physical_cmb_input_contract_receipt": bool(
            physical_cmb_global.get("physical_cmb_input_contract_receipt", False)
        ),
        "physical_cmb_prediction_receipt": bool(physical_cmb_global.get("physical_cmb_prediction_receipt", False)),
    }
    artifact_smoke = bool(
        gates["all_expected_shards_completed"]
        and gates["global_carrier_contract_receipt"]
        and gates["seam_metadata_replay_receipt"]
        and bool(halo_exchange.get("reducer_halo_exchange_replay_receipt", False))
    )
    distributed_kernel_scaling_ready = bool(
        gates["all_expected_shards_completed"]
        and gates["global_carrier_contract_receipt"]
        and gates["distributed_realization_event_certificate_receipt"]
        and gates["online_cross_shard_overlap_repair_receipt"]
        and gates["per_cycle_cross_shard_halo_exchange_receipt"]
        and gates["distributed_local_diamond_receipt"]
        and gates["distributed_repair_completeness_receipt"]
        and gates["cycle_holonomy_zero_or_classified_receipt"]
        and gates["selected_fiber_nontrivial_elimination_receipt"]
        and gates["same_boundary_multistart_confluence_receipt"]
        and gates["quotient_normal_form_canonical_hash_receipt"]
        and gates["fair_block_contraction_receipt"]
        and gates["schedule_independent_normal_form_receipt"]
        and gates["partition_naturality_receipt"]
    )
    observer_visualization_ready = bool(
        artifact_smoke
        and gates["large_visualization_observer_contract_receipt"]
    )
    observer_export_experiment_ready = bool(
        distributed_kernel_scaling_ready
        and gates["global_observer_modular_time_export_receipt"]
    )
    bulk_emergence_experiment_ready = bool(
        distributed_kernel_scaling_ready
        and gates["global_observer_modular_time_export_receipt"]
        and gates["global_proto_particle_worldline_export_receipt"]
    )
    cmb_generation_experiment_ready = bool(
        distributed_kernel_scaling_ready
        and gates["physical_cmb_input_contract_receipt"]
    )
    post_run_science_promotion = bool(
        distributed_kernel_scaling_ready
        and gates["strict_single_global_neutral_bulk_receipt"]
        and gates["cross_shard_worldline_stitching_receipt"]
        and gates["moving_proto_particle_candidate_receipt"]
        and gates["global_pn_resonance_receipt"]
        and gates["physical_cmb_prediction_receipt"]
    )
    large_ready = distributed_kernel_scaling_ready
    profile_requirements = {
        "distributed_artifact_packaging_smoke": (
            "all_expected_shards_completed",
            "global_carrier_contract_receipt",
            "seam_metadata_replay_receipt",
        ),
        "distributed_kernel_scaling": (
            "all_expected_shards_completed",
            "global_carrier_contract_receipt",
            "distributed_realization_event_certificate_receipt",
            "online_cross_shard_overlap_repair_receipt",
            "per_cycle_cross_shard_halo_exchange_receipt",
            "distributed_local_diamond_receipt",
            "distributed_repair_completeness_receipt",
            "cycle_holonomy_zero_or_classified_receipt",
            "selected_fiber_nontrivial_elimination_receipt",
            "same_boundary_multistart_confluence_receipt",
            "quotient_normal_form_canonical_hash_receipt",
            "fair_block_contraction_receipt",
            "schedule_independent_normal_form_receipt",
            "partition_naturality_receipt",
        ),
        "observer_visualization_payload": (
            "all_expected_shards_completed",
            "seam_metadata_replay_receipt",
            "large_visualization_observer_contract_receipt",
        ),
        "observer_export_experiment": (
            "online_cross_shard_overlap_repair_receipt",
            "per_cycle_cross_shard_halo_exchange_receipt",
            "global_observer_modular_time_export_receipt",
        ),
        "bulk_emergence_experiment": (
            "online_cross_shard_overlap_repair_receipt",
            "per_cycle_cross_shard_halo_exchange_receipt",
            "global_observer_modular_time_export_receipt",
            "global_proto_particle_worldline_export_receipt",
        ),
        "cmb_generation_experiment": (
            "online_cross_shard_overlap_repair_receipt",
            "per_cycle_cross_shard_halo_exchange_receipt",
            "physical_cmb_input_contract_receipt",
        ),
        "post_run_science_promotion": (
            "strict_single_global_neutral_bulk_receipt",
            "cross_shard_worldline_stitching_receipt",
            "global_pn_resonance_receipt",
            "physical_cmb_prediction_receipt",
        ),
    }
    profile_blockers = {
        name: [key for key in keys if not gates.get(key, False)]
        for name, keys in profile_requirements.items()
    }
    report = {
        "mode": "distributed_run_pack_contract_v0",
        "run_id": manifest.get("run_id"),
        "distributed_artifact_packaging_smoke_receipt": artifact_smoke,
        "small_scale_smoke_contract_receipt": artifact_smoke,
        "observer_visualization_payload_ready_receipt": observer_visualization_ready,
        "distributed_kernel_scaling_readiness_receipt": distributed_kernel_scaling_ready,
        "observer_export_experiment_readiness_receipt": observer_export_experiment_ready,
        "bulk_emergence_experiment_readiness_receipt": bulk_emergence_experiment_ready,
        "cmb_generation_experiment_readiness_receipt": cmb_generation_experiment_ready,
        "post_run_science_promotion_receipt": post_run_science_promotion,
        "large_scale_cloud_run_ready_receipt": large_ready,
        "gates": gates,
        "profile_blockers": profile_blockers,
        "required_artifacts": {
            "distributed_summary": "distributed_universe_summary.json",
            "visualization_payload": str(visualization_payload_path),
            "global_halo_exchange": "halo_exchange_global/global_halo_exchange_report.json",
            "global_neutral_bulk": "strict_neutral_global/global_neutral_bulk_reduction_report.json",
            "global_observer_modular_time": "observer_modular_time_global/observer_modular_time_global_payload.json",
            "global_proto_particles": "proto_particles_global/global_proto_particle_worldlines_report.json",
            "global_pn_resonance": "pn_resonance_global/global_pn_resonance_report.json",
            "global_physical_cmb": "physical_cmb_global/physical_cmb_global_reduction_report.json",
            "global_carrier_contract": "global_carrier_contract/GLOBAL_CARRIER_CONTRACT.json",
        },
        "global_carrier_contract": global_carrier,
        "blockers": profile_blockers["distributed_kernel_scaling"],
        "claim_boundary": (
            "Run-pack contract with separate profiles. Artifact smoke and observer visualization readiness "
            "only certify packaging/export health. Science-scale distributed-kernel readiness requires online "
            "seam repair, atomic/confluent commits, selected-fiber elimination, normal-form hash, holonomy, "
            "fair-block, schedule, and partition receipts. "
            "Post-run science promotion is separate from launch readiness."
        ),
    }
    _write_json(Path(out_dir) / "DISTRIBUTED_RUN_PACK_CONTRACT.json", report)
    return report


def _global_physical_cmb_reduction(
    *,
    manifest: dict[str, Any],
    completed: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    """Reduce shard-local finite CMB/theorem inputs before running hard gates."""

    shard_roots = [Path(row["run_dir"]) for row in completed if row.get("completed")]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not shard_roots:
        report = {
            "mode": "distributed_physical_cmb_global_reduction_v0",
            "report_dir": str(out_dir),
            "completed_shard_count": 0,
            "expected_shard_count": int(manifest.get("shard_count", 0)),
            "finite_source_global_reduction_receipt": False,
            "physical_cmb_input_contract_receipt": False,
            "physical_cmb_output_comparison_receipt": False,
            "physical_cmb_prediction_receipt": False,
            "blockers": ["no_completed_shards"],
            "claim_boundary": (
                "No physical CMB claim: the distributed reducer had no completed observer-like "
                "self-reading shards to reduce."
            ),
        }
        _write_json(out_dir / "physical_cmb_global_reduction_report.json", report)
        return report

    try:
        from oph_fpe.cosmology.physical_cmb_output import write_physical_cmb_output_comparison_report
        from oph_fpe.cosmology.physical_cmb_prediction import (
            write_physical_cmb_frontier_report,
            write_physical_cmb_input_no_data_use_receipt,
            write_physical_cmb_input_report,
            write_physical_cmb_promotion_audit_report,
        )

        finite_reduction = _write_global_finite_cmb_source_reports(
            manifest=manifest,
            shard_roots=shard_roots,
            out_dir=out_dir,
        )
        write_physical_cmb_input_no_data_use_receipt(shard_roots + [out_dir], out_dir)
        no_data = _write_global_no_data_use_receipt(shard_roots + [out_dir], out_dir)
        input_report = write_physical_cmb_input_report([out_dir], out_dir)
        promotion = write_physical_cmb_promotion_audit_report([out_dir], out_dir)
        output = write_physical_cmb_output_comparison_report([out_dir, *shard_roots], out_dir)
        frontier = write_physical_cmb_frontier_report([out_dir, *shard_roots], out_dir)
        blockers = _unique_texts(
            list(finite_reduction.get("blockers") or [])
            + list(input_report.get("blockers") or [])
            + list(promotion.get("promotion_blockers") or [])
            + list(output.get("contract_blockers") or [])
            + list(output.get("promotion_blockers") or [])
            + list(frontier.get("blockers") or [])
        )
        report = {
            "mode": "distributed_physical_cmb_global_reduction_v0",
            "report_dir": str(out_dir),
            "completed_shard_count": len(shard_roots),
            "expected_shard_count": int(manifest.get("shard_count", 0)),
            "source_run_dirs": [str(path) for path in shard_roots],
            "finite_source_global_reduction_receipt": bool(
                finite_reduction.get("FINITE_CMB_GLOBAL_REDUCTION_RECEIPT", False)
            ),
            "no_data_use_receipt": bool(no_data.get("NO_DATA_USE_RECEIPT", False)),
            "physical_cmb_input_contract_receipt": bool(
                input_report.get("PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT", False)
            ),
            "physical_cmb_prediction_eligible": bool(input_report.get("physical_cmb_prediction_eligible", False)),
            "physical_cmb_promotion_ready": bool(promotion.get("physical_cmb_promotion_ready", False)),
            "physical_cmb_output_comparison_receipt": bool(
                output.get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT", False)
            ),
            "usable_physical_cmb_data_receipt": bool(output.get("USABLE_PHYSICAL_CMB_DATA_RECEIPT", False)),
            "physical_cmb_prediction_receipt": bool(
                output.get("PHYSICAL_CMB_PREDICTION_RECEIPT", False)
                and frontier.get("physical_cmb_prediction_receipt", False)
            ),
            "official_likelihood_ready": bool(frontier.get("official_likelihood_ready", False)),
            "cdm_limit_regression_passed": bool(frontier.get("cdm_limit_regression_passed", False)),
            "measurement_comparable_model_count": int(output.get("measurement_comparable_model_count") or 0),
            "oph_diagnostic_model_count": int(output.get("oph_diagnostic_model_count") or 0),
            "best_oph_diagnostic_model": output.get("best_oph_diagnostic_model") or {},
            "finite_reduction_report_path": str(out_dir / "finite_cmb_global_reduction_report.json"),
            "input_report_path": str(out_dir / "physical_cmb_input_report.json"),
            "promotion_audit_report_path": str(out_dir / "physical_cmb_promotion_audit_report.json"),
            "output_comparison_report_path": str(out_dir / "physical_cmb_output_comparison_report.json"),
            "frontier_report_path": str(out_dir / "physical_cmb_frontier_report.json"),
            "reduced_source_files": finite_reduction.get("reduced_source_files") or [],
            "component_receipts": finite_reduction.get("component_receipts") or {},
            "blockers": blockers,
            "claim_boundary": (
                "Global distributed physical-CMB reduction over completed OPH shards. The reducer first "
                "combines finite-derived theorem/CMB source reports across observer-like self-reading shards, "
                "then runs the hard physical-CMB contract. Physical-unit TT outputs remain diagnostics unless "
                "the global input contract, promotion gates, output comparison, and prediction receipts all pass."
            ),
        }
    except Exception as exc:  # pragma: no cover - defensive fail-closed report path.
        report = {
            "mode": "distributed_physical_cmb_global_reduction_v0",
            "report_dir": str(out_dir),
            "completed_shard_count": len(shard_roots),
            "expected_shard_count": int(manifest.get("shard_count", 0)),
            "finite_source_global_reduction_receipt": False,
            "physical_cmb_input_contract_receipt": False,
            "physical_cmb_output_comparison_receipt": False,
            "physical_cmb_prediction_receipt": False,
            "blockers": [f"global_physical_cmb_reduction_failed:{type(exc).__name__}:{exc}"],
            "claim_boundary": (
                "Physical CMB reduction failed closed. No OPH physical CMB prediction receipt was emitted."
            ),
        }
    _write_json(out_dir / "physical_cmb_global_reduction_report.json", report)
    return report


def _write_global_finite_cmb_source_reports(
    *,
    manifest: dict[str, Any],
    shard_roots: list[Path],
    out_dir: Path,
) -> dict[str, Any]:
    expected = len(shard_roots)
    transition_reports = _collect_named_reports(shard_roots, "finite_repair_transition_matrix_report.json")
    finite_cert_reports = _collect_named_reports(shard_roots, "finite_certificate_report.json")
    ba_reports = _collect_named_reports(shard_roots, "B_A_kernel_report.json")
    ba_parent_reports = _collect_named_reports(shard_roots, "b_a_parent_report.json")
    scale_reports = _collect_named_reports(shard_roots, "scale_compressed_repair_report.json")
    screen_capacity_reports = _collect_named_reports(shard_roots, "screen_capacity_closure_report.json")
    strict_neutral_reports = _collect_named_reports(shard_roots, "strict_neutral_bulk_report.json")
    compressed_likelihood_reports = _collect_named_reports(shard_roots, "oph_compressed_likelihood_report.json")
    official_likelihood_reports = _collect_named_reports(shard_roots, "official_planck_likelihood_readiness_report.json")
    camb_baseline_reports = _collect_named_reports(shard_roots, "camb_lcdm_baseline_report.json")

    eta_values = _finite_values(transition_reports, _transition_eta_value)
    gamma_values = _finite_values(transition_reports, lambda report: _float_or_none((report.get("primary") or {}).get("gamma_continuous")))
    a_zeta_values = _finite_values(finite_cert_reports, _finite_cert_a_zeta)
    q_values = _finite_values(scale_reports, lambda report: _float_or_none((report.get("cmb_parameter_readouts") or {}).get("q_IR")))
    ell_values = _finite_values(scale_reports, lambda report: _float_or_none((report.get("cmb_parameter_readouts") or {}).get("ell_IR")))
    n_values = _finite_values(
        screen_capacity_reports,
        lambda report: _float_or_none((report.get("observed_branch_normalization") or {}).get("N_CRC")),
    )
    b_a_rows = _numeric_rows_from_reports(ba_reports, ("B_A_k_a",))
    rho_a_rows = _numeric_rows_from_reports(finite_cert_reports, ("derived_outputs", "rho_A_a"))
    ba_parent_rows = _collect_row_dicts(ba_parent_reports, ("rows", "observer_view_rows"))

    transition_ready = (
        len(transition_reports) == expected
        and len(eta_values) == expected
        and len(gamma_values) == expected
        and all(
            bool(report.get("finite_transition_matrix_ready", False))
            and (
                bool(report.get("eta_R_finite_lattice_derived", False))
                or bool(report.get("eta_R_empirical_finite_lattice_derived", False))
                or bool(((report.get("clock_modes") or {}).get("empirical") or {}).get("eta_R_finite_lattice_derived", False))
            )
            for report in transition_reports
        )
    )
    finite_certificate_ready = (
        len(finite_cert_reports) == expected
        and len(a_zeta_values) == expected
        and all(value > 0.0 for value in a_zeta_values)
        and bool(rho_a_rows)
        and all(
            bool(report.get("theorem_grade_finite_inputs", False))
            and bool(
                ((report.get("derived_outputs") or {}).get("screen_to_primordial_lift_receipt", False))
                or report.get("SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT", False)
                or report.get("screen_to_primordial_lift_receipt", False)
            )
            for report in finite_cert_reports
        )
    )
    ba_ready = len(ba_reports) == expected and bool(b_a_rows) and all(
        bool(report.get("B_A_KERNEL_RECEIPT", False)) for report in ba_reports
    )
    scale_ready = (
        len(scale_reports) == expected
        and len(q_values) == expected
        and len(ell_values) == expected
        and all(value > 0.0 for value in ell_values)
        and all(bool(report.get("scale_compressed_operator_receipt", False)) for report in scale_reports)
    )
    strict_neutral_ready = len(strict_neutral_reports) == expected and all(
        bool(report.get("strict_neutral_bulk", False)) for report in strict_neutral_reports
    )
    official_likelihood_local_any = any(
        bool(report.get("official_likelihood_ready", False))
        or bool(report.get("official_likelihood_execution_ready", False))
        for report in compressed_likelihood_reports + official_likelihood_reports
    )
    cdm_limit_regression_local_any = any(
        bool(report.get("cdm_limit_regression_passed", False))
        or bool(report.get("CDM_LIMIT_BOLTZMANN_RECEIPT", False))
        for report in compressed_likelihood_reports + camb_baseline_reports
    )
    observed_screen_capacity_ready = bool(n_values)
    local_component_rollups = {
        "finite_repair_transition_clock_local_rollup": transition_ready,
        "finite_certificate_local_rollup": finite_certificate_ready,
        "B_A_kernel_local_rollup": ba_ready,
        "scale_compressed_scalar_local_rollup": scale_ready,
        "strict_neutral_local_rollup": strict_neutral_ready,
        "official_likelihood_local_any": official_likelihood_local_any,
        "cdm_limit_regression_local_any": cdm_limit_regression_local_any,
    }
    global_pooled_sufficient_statistics_ready = False
    transition_ready = False
    finite_certificate_ready = False
    ba_ready = False
    scale_ready = False
    strict_neutral_ready = False
    official_likelihood_ready = False
    cdm_limit_regression_passed = False

    component_receipts = {
        "finite_repair_transition_clock_global_reduction": transition_ready,
        "finite_certificate_global_reduction": finite_certificate_ready,
        "B_A_kernel_global_reduction": ba_ready,
        "scale_compressed_scalar_global_reduction": scale_ready,
        "neutral_or_scale_freezeout_global_reduction": bool(strict_neutral_ready or scale_ready),
        "screen_capacity_global_readout": observed_screen_capacity_ready,
        "global_pooled_sufficient_statistics_receipt": global_pooled_sufficient_statistics_ready,
        "official_likelihood_ready": official_likelihood_ready,
        "cdm_limit_regression_passed": cdm_limit_regression_passed,
    }
    blockers = [
        key
        for key, passed in component_receipts.items()
        if key
        in {
            "finite_repair_transition_clock_global_reduction",
            "finite_certificate_global_reduction",
            "B_A_kernel_global_reduction",
            "scale_compressed_scalar_global_reduction",
            "neutral_or_scale_freezeout_global_reduction",
        }
        and not passed
    ]
    if not global_pooled_sufficient_statistics_ready:
        blockers.append("global_pooled_sufficient_statistics_reducer_missing")
    finite_global_receipt = not blockers

    _write_json(
        out_dir / "finite_repair_transition_matrix_report.json",
        {
            "mode": "distributed_finite_repair_transition_matrix_reduction_v0",
            "finite_transition_matrix_ready": transition_ready,
            "eta_R_finite_lattice_derived": transition_ready,
            "FINITE_REPAIR_TRANSITION_CLOCK_GLOBAL_REDUCTION_RECEIPT": transition_ready,
            "primary": {
                "eta_R_estimate": None,
                "gamma_continuous": None,
                "diagnostic_eta_R_shard_mean": _mean_or_none(eta_values),
                "diagnostic_eta_R_shard_std": _std_or_none(eta_values),
                "diagnostic_gamma_continuous_shard_mean": _mean_or_none(gamma_values),
                "diagnostic_gamma_continuous_shard_std": _std_or_none(gamma_values),
            },
            "local_component_rollup_receipt": local_component_rollups["finite_repair_transition_clock_local_rollup"],
            "shard_count": expected,
            "source_report_count": len(transition_reports),
            "claim_boundary": (
                "Diagnostic reducer output from shard-local finite repair transition clocks. The mean of "
                "shard-local nonlinear estimates is not a physical global estimate; a pooled sufficient-statistic "
                "reducer is required before this can satisfy the physical CMB contract."
            ),
        },
    )
    _write_json(
        out_dir / "finite_certificate_report.json",
        {
            "mode": "distributed_finite_certificate_reduction_v0",
            "theorem_grade_finite_inputs": finite_certificate_ready,
            "FINITE_CERTIFICATE_GLOBAL_REDUCTION_RECEIPT": finite_certificate_ready,
            "SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT": finite_certificate_ready,
            "derived_outputs": {
                "A_zeta": None,
                "diagnostic_A_zeta_shard_mean": _mean_or_none(a_zeta_values),
                "diagnostic_A_zeta_shard_std": _std_or_none(a_zeta_values),
                "rho_A_a": None,
                "diagnostic_rho_A_a_rows": rho_a_rows,
                "screen_to_primordial_lift_receipt": finite_certificate_ready,
            },
            "local_component_rollup_receipt": local_component_rollups["finite_certificate_local_rollup"],
            "shard_count": expected,
            "source_report_count": len(finite_cert_reports),
            "claim_boundary": (
                "Diagnostic reducer output from shard-local finite theorem certificate reports. A_zeta and "
                "rho_A(a) are not promoted from shard means/concatenated rows; the screen-to-primordial lift "
                "requires a global pooled source reducer."
            ),
        },
    )
    _write_json(
        out_dir / "B_A_kernel_report.json",
        {
            "mode": "distributed_B_A_kernel_reduction_v0",
            "B_A_KERNEL_RECEIPT": ba_ready,
            "B_A_GLOBAL_REDUCTION_RECEIPT": ba_ready,
            "B_A_k_a": None,
            "diagnostic_B_A_k_a_rows": b_a_rows,
            "local_component_rollup_receipt": local_component_rollups["B_A_kernel_local_rollup"],
            "shard_count": expected,
            "source_report_count": len(ba_reports),
            "claim_boundary": (
                "Diagnostic B_A(k,a) reducer output. Concatenated shard rows are not a unit-, grid-, "
                "coverage-, covariance-, and provenance-aware global kernel."
            ),
        },
    )
    _write_json(
        out_dir / "b_a_parent_report.json",
        {
            "mode": "distributed_b_a_parent_reduction_v0",
            "readiness": {"checks": {"no_cmb_data_used": True, "global_shard_reduction": bool(ba_parent_rows)}},
            "rows": ba_parent_rows,
            "shard_count": expected,
            "source_report_count": len(ba_parent_reports),
        },
    )
    _write_json(
        out_dir / "scale_compressed_repair_report.json",
        {
            "mode": "distributed_scale_compressed_repair_reduction_v0",
            "scale_compressed_operator_receipt": scale_ready,
            "SCALE_COMPRESSED_GLOBAL_REDUCTION_RECEIPT": scale_ready,
            "logical_repair_rounds": _median_int_or_none(
                [_int_or_none(report.get("logical_repair_rounds")) for report in scale_reports]
            ),
            "cmb_parameter_readouts": {
                "q_IR": None,
                "ell_IR": None,
                "diagnostic_q_IR_shard_mean": _mean_or_none(q_values),
                "diagnostic_q_IR_shard_std": _std_or_none(q_values),
                "diagnostic_ell_IR_shard_mean": _mean_or_none(ell_values),
                "diagnostic_ell_IR_shard_std": _std_or_none(ell_values),
            },
            "SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT": False,
            "local_component_rollup_receipt": local_component_rollups["scale_compressed_scalar_local_rollup"],
            "shard_count": expected,
            "source_report_count": len(scale_reports),
        },
    )
    _write_json(
        out_dir / "screen_capacity_closure_report.json",
        {
            "mode": "distributed_screen_capacity_closure_reduction_v0",
            "observed_branch_normalization": {
                "N_CRC": _consensus_or_none(n_values),
                "N_CRC_shard_mean": _mean_or_none(n_values),
                "N_CRC_shard_std": _std_or_none(n_values),
                "N_CRC_additive_sum_diagnostic": float(sum(n_values)) if n_values else None,
                "N_CRC_shard_values": n_values,
            },
            "readiness_gates": {
                "observed_branch_N_scr_readout_available": bool(_consensus_or_none(n_values) is not None),
                "N_CRC_consensus_invariant": bool(_consensus_or_none(n_values) is not None),
                "additive_capacity_schema_declared": False,
            },
            "shard_count": expected,
            "source_report_count": len(screen_capacity_reports),
            "claim_boundary": (
                "Screen capacity closure reduction treats N_CRC as a consensus invariant by default. The "
                "additive sum is diagnostic only unless shard reports declare non-overlapping coverage and an "
                "additive capacity schema."
            ),
        },
    )
    _write_json(
        out_dir / "strict_neutral_bulk_report.json",
        {
            "mode": "distributed_strict_neutral_bulk_reduction_v0",
            "strict_neutral_bulk": strict_neutral_ready,
            "STRICT_NEUTRAL_BULK_GLOBAL_REDUCTION_RECEIPT": strict_neutral_ready,
            "shard_count": expected,
            "source_report_count": len(strict_neutral_reports),
        },
    )
    _write_json(
        out_dir / "oph_compressed_likelihood_report.json",
        {
            "mode": "distributed_likelihood_gate_reduction_v0",
            "official_likelihood_ready": official_likelihood_ready,
            "cdm_limit_regression_passed": cdm_limit_regression_passed,
            "local_rollups": {
                "official_likelihood_local_any": official_likelihood_local_any,
                "cdm_limit_regression_local_any": cdm_limit_regression_local_any,
            },
            "source_report_count": len(compressed_likelihood_reports),
            "claim_boundary": (
                "Reducer-level likelihood gate report. Shard-local any() readiness is diagnostic only; official "
                "likelihood and CDM-limit readiness must be executed as global reducer tests."
            ),
        },
    )

    report = {
        "mode": "distributed_finite_cmb_source_reduction_v0",
        "FINITE_CMB_GLOBAL_REDUCTION_RECEIPT": finite_global_receipt,
        "expected_shard_count": expected,
        "completed_shard_count": len(shard_roots),
        "component_receipts": component_receipts,
        "local_component_rollups": local_component_rollups,
        "blockers": blockers,
        "input_statistics": {
            "eta_R": _stats(eta_values),
            "gamma_continuous": _stats(gamma_values),
            "A_zeta": _stats(a_zeta_values),
            "q_IR": _stats(q_values),
            "ell_IR": _stats(ell_values),
            "N_CRC": _stats(n_values),
            "B_A_row_count": len(b_a_rows),
            "rho_A_row_count": len(rho_a_rows),
            "b_a_parent_row_count": len(ba_parent_rows),
        },
        "source_report_counts": {
            "finite_repair_transition_matrix_report": len(transition_reports),
            "finite_certificate_report": len(finite_cert_reports),
            "B_A_kernel_report": len(ba_reports),
            "b_a_parent_report": len(ba_parent_reports),
            "scale_compressed_repair_report": len(scale_reports),
            "screen_capacity_closure_report": len(screen_capacity_reports),
            "strict_neutral_bulk_report": len(strict_neutral_reports),
            "oph_compressed_likelihood_report": len(compressed_likelihood_reports),
            "official_planck_likelihood_readiness_report": len(official_likelihood_reports),
            "camb_lcdm_baseline_report": len(camb_baseline_reports),
        },
        "reduced_source_files": [
            "finite_repair_transition_matrix_report.json",
            "finite_certificate_report.json",
            "B_A_kernel_report.json",
            "b_a_parent_report.json",
            "scale_compressed_repair_report.json",
            "screen_capacity_closure_report.json",
            "strict_neutral_bulk_report.json",
            "oph_compressed_likelihood_report.json",
        ],
        "claim_boundary": (
            "Finite CMB source reduction over completed distributed OPH shards. Current outputs preserve "
            "diagnostic shard statistics but do not promote nonlinear shard means or row concatenations into "
            "physical finite inputs. The physical CMB gate requires a future pooled sufficient-statistic reducer."
        ),
    }
    _write_json(out_dir / "finite_cmb_global_reduction_report.json", report)
    return report


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
    carrier_shard: dict[str, Any],
    global_carrier_artifacts: dict[str, Any],
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
        "global_carrier": {
            "one_global_carrier_before_partition": True,
            "artifact_paths": {
                key: (value or {}).get("path")
                for key, value in global_carrier_artifacts.items()
                if isinstance(value, dict)
            },
            "owned_nodes": list(carrier_shard.get("owned_nodes") or []),
            "ghost_nodes": list(carrier_shard.get("ghost_nodes") or []),
            "cut_edge_ids": list(carrier_shard.get("cut_edge_ids") or []),
            "authoritative_owner": int(carrier_shard.get("shard_index", shard_index)),
            "projection_rule": (
                "Only owned nodes are authoritative physical state. Ghost nodes, queues, retry logs, "
                "and checkpoints are implementation metadata unless a committed event certificate "
                "projects them to a monolithic repair."
            ),
        },
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
    cmb = summary.get("physical_cmb_global_reduction") or {}
    halo = summary.get("global_halo_exchange_reduction") or {}
    neutral = summary.get("global_neutral_bulk_reduction") or {}
    observer_time = summary.get("global_observer_modular_time_export") or {}
    proto = summary.get("global_proto_particle_worldlines") or {}
    pn = summary.get("global_pn_resonance_reduction") or {}
    contract = summary.get("distributed_run_pack_contract") or {}
    carrier = summary.get("global_carrier_contract") or {}
    lines = [
        "# Distributed OPH Universe Summary",
        "",
        f"- Kernel: `{summary.get('kernel')}`",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Completed shards: `{summary.get('completed_shard_count')}/{summary.get('expected_shard_count')}`",
        f"- Completed patch capacity: `{summary.get('total_patch_capacity_completed')}`",
        f"- Completed observer capacity: `{summary.get('total_observer_capacity_completed')}`",
        f"- Global carrier contract receipt: `{str(bool(summary.get('global_carrier_contract_receipt'))).lower()}`",
        f"- One carrier before partition receipt: `{str(bool(summary.get('one_global_carrier_before_partition_receipt'))).lower()}`",
        f"- Stable global identity state receipt: `{str(bool(summary.get('stable_global_identity_initial_state_receipt'))).lower()}`",
        f"- Distributed realization event certificate receipt: `{str(bool(summary.get('distributed_realization_event_certificate_receipt'))).lower()}`",
        f"- Federated large-universe witness receipt: `{str(bool(summary.get('federated_large_universe_witness_receipt'))).lower()}`",
        f"- Seam metadata replay receipt: `{str(bool(summary.get('seam_metadata_replay_receipt'))).lower()}`",
        f"- Online cross-shard overlap repair receipt: `{str(bool(summary.get('online_cross_shard_overlap_repair_receipt'))).lower()}`",
        f"- Per-cycle cross-shard halo exchange receipt: `{str(bool(summary.get('per_cycle_cross_shard_halo_exchange_receipt'))).lower()}`",
        f"- Global observer modular-time export receipt: `{str(bool(summary.get('global_observer_modular_time_export_receipt'))).lower()}`",
        f"- Global proto-particle worldline export receipt: `{str(bool(summary.get('global_proto_particle_worldline_export_receipt'))).lower()}`",
        f"- Local scale-compressed P/N witness receipt: `{str(bool(summary.get('all_shards_local_scale_compressed_pn_witness_receipt'))).lower()}`",
        f"- Global P/N resonance receipt: `{str(bool(summary.get('global_pn_resonance_receipt'))).lower()}`",
        f"- Global physical CMB input contract receipt: `{str(bool(summary.get('global_physical_cmb_input_contract_receipt'))).lower()}`",
        f"- Global physical CMB output comparison receipt: `{str(bool(summary.get('global_physical_cmb_output_comparison_receipt'))).lower()}`",
        f"- Global physical CMB prediction receipt: `{str(bool(summary.get('global_physical_cmb_prediction_receipt'))).lower()}`",
        f"- Strict single global neutral bulk receipt: `{str(bool(summary.get('strict_single_global_neutral_bulk_receipt'))).lower()}`",
        f"- Artifact packaging smoke receipt: `{str(bool(contract.get('distributed_artifact_packaging_smoke_receipt'))).lower()}`",
        f"- Distributed-kernel scaling readiness receipt: `{str(bool(contract.get('distributed_kernel_scaling_readiness_receipt'))).lower()}`",
        f"- Observer visualization payload ready receipt: `{str(bool(contract.get('observer_visualization_payload_ready_receipt'))).lower()}`",
        f"- Large-scale cloud-run ready receipt: `{str(bool(contract.get('large_scale_cloud_run_ready_receipt'))).lower()}`",
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
    lines.extend(
        [
            "",
            "## Global Physical CMB Reduction",
            "",
            f"- Finite source global reduction receipt: `{str(bool(cmb.get('finite_source_global_reduction_receipt'))).lower()}`",
            f"- Input contract receipt: `{str(bool(cmb.get('physical_cmb_input_contract_receipt'))).lower()}`",
            f"- Output comparison receipt: `{str(bool(cmb.get('physical_cmb_output_comparison_receipt'))).lower()}`",
            f"- Physical prediction receipt: `{str(bool(cmb.get('physical_cmb_prediction_receipt'))).lower()}`",
            f"- Report directory: `{cmb.get('report_dir')}`",
            "",
            str(cmb.get("claim_boundary", "")),
        ]
    )
    lines.extend(
        [
            "",
            "## Global Distributed Reductions",
            "",
            f"- Global carrier graph nodes: `{(carrier.get('graph_info') or {}).get('node_count')}`",
            f"- Global carrier graph cut edges: `{carrier.get('cut_edge_count')}`",
            f"- Halo online receipt: `{str(bool(halo.get('per_cycle_cross_shard_halo_exchange_receipt'))).lower()}`",
            f"- Halo reducer replay receipt: `{str(bool(halo.get('reducer_halo_exchange_replay_receipt'))).lower()}`",
            f"- Global neutral bulk receipt: `{str(bool(neutral.get('strict_single_global_neutral_bulk_receipt'))).lower()}`",
            f"- Observer views exported: `{observer_time.get('objectiveObserverViewCount')}`",
            f"- Observer min time frames: `{observer_time.get('minTimeFrameCount')}`",
            f"- Proto-particle worldlines: `{proto.get('worldlineCount')}`",
            f"- Moving proto-particle worldlines: `{proto.get('movingWorldlineCount')}`",
            f"- P/N report count: `{pn.get('pnResonanceReportCount')}`",
            "",
            "Sidecar reports:",
            "- `halo_exchange_global/global_halo_exchange_report.json`",
            "- `strict_neutral_global/global_neutral_bulk_reduction_report.json`",
            "- `observer_modular_time_global/observer_modular_time_global_payload.json`",
            "- `proto_particles_global/global_proto_particle_worldlines_report.json`",
            "- `pn_resonance_global/global_pn_resonance_report.json`",
            "- `DISTRIBUTED_RUN_PACK_CONTRACT.json`",
        ]
    )
    cmb_blockers = cmb.get("blockers") or []
    if cmb_blockers:
        lines.extend(["", "### CMB Blockers"])
        for blocker in cmb_blockers:
            lines.append(f"- `{blocker}`")
    contract_blockers = contract.get("blockers") or []
    if contract_blockers:
        lines.extend(["", "### Run-Pack Contract Blockers"])
        for blocker in contract_blockers:
            lines.append(f"- `{blocker}`")
    carrier_blockers = carrier.get("blockers") or []
    if carrier_blockers:
        lines.extend(["", "### Global Carrier Contract Blockers"])
        for blocker in carrier_blockers:
            lines.append(f"- `{blocker}`")
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
    for pair in _global_cut_pairs(shard_count):
        seam_links.append(
            {
                "link_id": f"cut_{pair[0]:04d}_{pair[1]:04d}",
                "source_shard_index": pair[0],
                "target_shard_index": pair[1],
                "source_shard_id": f"{run_id}_shard{pair[0]:04d}",
                "target_shard_id": f"{run_id}_shard{pair[1]:04d}",
                "seam_halo_width": int(seam_halo_width),
                "overlap_model": "global_carrier_cut_edge_visible_collar",
                "source": "partition_map_cut_interfaces",
                "physics_receipt_eligible": False,
                "claim_boundary": (
                    "Cross-shard overlap link derived from a declared global-carrier cut edge. The reducer may "
                    "display its collar metadata, but physics receipts still require online linearized commits."
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
        "mode": "one_unified_screen_atlas_with_global_carrier_cut_links",
        "run_id": run_id,
        "global_patch_capacity": int(shard_count) * int(patch_count_per_shard),
        "global_observer_capacity": int(shard_count) * int(observers_per_shard),
        "shard_count": int(shard_count),
        "seam_halo_width": int(seam_halo_width),
        "shards": shards,
        "seam_links": seam_links,
        "unified_universe_receipt_boundary": (
            "The run is one distributed atlas of bounded screen charts. Display centers are visualization "
            "coordinates only. Cross-shard links used by the reducer are the declared cut edges of the "
            "global carrier, and online repair receipts are required before any physics promotion."
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
                "synthetic": True,
                "source": "endpoint_interpolation",
                "physics_receipt_eligible": False,
                "overlapRepairTrajectory": _seam_trajectory(source_trace, target_trace),
            }
        )
    completed_count = sum(1 for link in readout_links if link.get("seam_completed"))
    replay_receipt = bool(
        readout_links
        and completed_count / max(len(readout_links), 1) >= 0.95
        and final_committed_values
    )
    return {
        "mode": "cross_shard_global_carrier_cut_metadata_replay",
        "seam_link_count": len(readout_links),
        "completed_seam_count": completed_count,
        "completed_seam_fraction": float(completed_count / len(readout_links)) if readout_links else 0.0,
        "mean_final_committed_fraction": (
            float(sum(final_committed_values) / len(final_committed_values)) if final_committed_values else None
        ),
        "seam_metadata_replay_receipt": replay_receipt,
        "cross_shard_overlap_repair_receipt": False,
        "synthetic": True,
        "source": "endpoint_interpolation",
        "physics_receipt_eligible": False,
        "links": readout_links,
        "claim_boundary": (
            "Reducer-level metadata replay over declared global-carrier cut links. It gives the visualization "
            "app cross-shard observer-overlap links and synthetic endpoint-interpolated trajectories. It is "
            "not a per-edge distributed repair kernel and is not eligible for OPH confluence, bulk, or CMB "
            "receipts."
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
                "synthetic": True,
                "source": "endpoint_interpolation",
                "physics_receipt_eligible": False,
            }
        )
    return trajectory


def _distributed_visualization_payload(
    *,
    manifest: dict[str, Any],
    shard_rows: list[dict[str, Any]],
    seam_readout: dict[str, Any],
    physical_cmb_global: dict[str, Any],
    global_carrier: dict[str, Any],
    halo_exchange: dict[str, Any],
    neutral_global: dict[str, Any],
    observer_time_global: dict[str, Any],
    proto_particle_global: dict[str, Any],
    pn_global: dict[str, Any],
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
                "type": "cross_shard_global_carrier_cut_overlap",
                "repairTrajectory": link.get("overlapRepairTrajectory") or [],
                "claimBoundary": link.get("claim_boundary"),
            }
        )
    if observer_time_global.get("objectiveObserverViews"):
        observers = list(observer_time_global.get("objectiveObserverViews") or [])
    if observer_time_global.get("overlapLinks"):
        overlap_links = list(observer_time_global.get("overlapLinks") or [])[:100000] + overlap_links
    if proto_particle_global.get("worldlines"):
        worldlines = list(proto_particle_global.get("worldlines") or [])
    return {
        "schema": "oph_distributed_universe_visualization_payload_v1",
        "kernel": DISTRIBUTED_KERNEL_VERSION,
        "runId": manifest.get("run_id"),
        "claimBoundary": manifest.get("claim_boundary"),
        "unifiedUniverse": {
            "atlas": manifest.get("unified_universe_atlas") or {},
            "globalCarrier": global_carrier,
            "crossShardSeamReadout": seam_readout,
            "haloExchange": halo_exchange,
            "neutralBulk": neutral_global,
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
                "atlas and render declared carrier-cut links as overlapping observer supports across partitions."
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
            "overlapLinks": overlap_links[:100000],
            "totalAvailableObserverCapacity": manifest.get("total_observer_capacity"),
            "globalExport": {
                "reportPath": "observer_modular_time_global/observer_modular_time_global_payload.json",
                "objectiveObserverViewCount": observer_time_global.get("objectiveObserverViewCount"),
                "minTimeFrameCount": observer_time_global.get("minTimeFrameCount"),
                "largeVisualizationObserverContractReceipt": observer_time_global.get(
                    "large_visualization_observer_contract_receipt"
                ),
            },
            "observerViewSource": (
                "global_observer_modular_time_export"
                if observer_time_global.get("objectiveObserverViews")
                else "sampled_per_shard_payloads_with_global_ids"
            ),
        },
        "consensusBulk": {
            "strictNeutralGlobalReduction": neutral_global,
            "protoParticleCandidates": {
                "worldlines": worldlines,
                "globalStitchReport": {
                    "reportPath": "proto_particles_global/global_proto_particle_worldlines_report.json",
                    "worldlineCount": proto_particle_global.get("worldlineCount"),
                    "movingWorldlineCount": proto_particle_global.get("movingWorldlineCount"),
                    "crossShardWorldlineStitchingReceipt": proto_particle_global.get(
                        "cross_shard_worldline_stitching_receipt"
                    ),
                },
                "worldlineSource": (
                    "global_proto_particle_worldline_stitch"
                    if proto_particle_global.get("worldlines")
                    else "sampled_per_shard_payloads_with_atlas_shard_context"
                ),
            }
        },
        "pnSilenceToObservation": {
            "globalReduction": pn_global,
            "claimBoundary": (
                "Shows the scale-compressed P/N silence-to-observation lane across shards. It is not a full "
                "finite-capacity N proof unless the global P/N receipt and capacity gates pass."
            ),
        },
        "physicalCMB": {
            "globalReduction": physical_cmb_global,
            "claimBoundary": (
                "Visualization can show globally reduced CMB readiness and physical-unit diagnostic curves. "
                "It must not label a physical OPH CMB prediction unless physical_cmb_prediction_receipt is true."
            ),
        },
        "visualizationNotes": [
            "Render the whole object as one atlas, not as separate universes.",
            "Use crossShardOverlapLinks to draw observer support overlap across declared carrier cuts.",
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
        local_observer = item.get("observerId", item.get("id", len(result)))
        item["observerId"] = _global_id(shard, local_observer, "observer")
        frames = item.get("timeFrames") if isinstance(item.get("timeFrames"), list) else []
        normalized_frames = [
            normalize_observer_frame(frame if isinstance(frame, dict) else {"value": frame}, record_order=index)
            for index, frame in enumerate(frames)
        ]
        item["timeFrames"] = normalized_frames
        item["semantic_history_digest"] = semantic_history_digest(normalized_frames)
        item["execution_clock_fields_separated_receipt"] = all(
            all(key in frame for key in (
                "execution_epoch",
                "scheduler_event_index",
                "observer_record_order",
                "observer_modular_parameter",
                "observer_clock_uncertainty",
            ))
            for frame in normalized_frames
        )
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


def _compact_observer_time_report(report: dict[str, Any]) -> dict[str, Any]:
    compact = {key: value for key, value in report.items() if key not in {"objectiveObserverViews", "overlapLinks"}}
    compact["objectiveObserverViewsOmittedFromSummary"] = len(report.get("objectiveObserverViews") or [])
    compact["overlapLinksOmittedFromSummary"] = len(report.get("overlapLinks") or [])
    return compact


def _compact_proto_particle_report(report: dict[str, Any]) -> dict[str, Any]:
    compact = {key: value for key, value in report.items() if key not in {"worldlines"}}
    compact["worldlinesOmittedFromSummary"] = len(report.get("worldlines") or [])
    return compact


def _combined_observer_views(
    completed: list[dict[str, Any]],
    *,
    max_total: int,
    per_shard: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in completed:
        run_dir = Path(str(shard.get("run_dir")))
        for row in _read_jsonl_rows(run_dir / "observer_views.jsonl", limit=per_shard):
            item = dict(row)
            local_id = item.get("observer_id", item.get("observerId", len(rows)))
            local_int = _int_or_none(local_id)
            observer_range = shard.get("global_observer_range") if isinstance(shard.get("global_observer_range"), list) else []
            base = _int_or_none(observer_range[0]) if observer_range else 0
            numeric_global_id = int((base or 0) + (local_int if local_int is not None else len(rows)))
            item["observer_id"] = numeric_global_id
            item["observerId"] = _global_id(shard, local_id, "observer")
            item["distributed_observer_id"] = item["observerId"]
            item["distributed_shard_index"] = shard.get("shard_index")
            item["distributed_shard_id"] = shard.get("shard_id")
            rows.append(item)
            if len(rows) >= int(max_total):
                return rows
    return rows


def _read_jsonl_rows(path: Path, *, limit: int) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= int(limit):
                break
            text = line.strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                rows.append(data)
    return rows


def _globalized_worldline_row(shard: dict[str, Any], row: dict[str, Any], *, fallback_index: int) -> dict[str, Any]:
    local_id = row.get("worldline_id", row.get("worldlineId", fallback_index))
    events = []
    for event in row.get("events") or row.get("sample_events") or []:
        if not isinstance(event, dict):
            continue
        point = event.get("h3_spatial_point") or event.get("h3SpatialPoint")
        if not isinstance(point, list) or len(point) < 3:
            continue
        normalized_point = [float(value) for value in point[:4]] if len(point) >= 4 else [float(value) for value in point[:3]]
        events.append(
            {
                "cycle": event.get("cycle"),
                "h3SpatialPoint": normalized_point,
                "fitResidual": event.get("fit_residual", event.get("fitResidual")),
                "supportNodeCount": event.get("support_node_count", event.get("supportNodeCount")),
                "shardIndex": shard.get("shard_index"),
            }
        )
    path_length, mean_step = _worldline_path_metrics(events)
    return {
        "worldlineId": _global_id(shard, local_id, "worldline"),
        "localWorldlineId": local_id,
        "globalWorldlineId": row.get("global_worldline_id") or row.get("globalWorldlineId"),
        "stitchKey": row.get("stitch_key") or row.get("stitchKey"),
        "shardIndex": shard.get("shard_index"),
        "shardId": shard.get("shard_id"),
        "observationCount": row.get("observation_count", row.get("observationCount", len(events))),
        "birthCycle": row.get("birth_cycle", row.get("birthCycle")),
        "deathCycle": row.get("death_cycle", row.get("deathCycle")),
        "h3PathLength": row.get("h3_path_length", row.get("h3PathLength", path_length)),
        "meanH3Step": row.get("mean_h3_step", row.get("meanH3Step", mean_step)),
        "classMode": row.get("class_mode", row.get("classMode")),
        "events": events,
        "bulkLocalizationPass": bool(
            row.get("bulk_localization_pass", False) or row.get("bulkLocalizationPass", False)
        ),
        "localizationPass": bool(row.get("localization_pass", False) or row.get("localizationPass", False)),
        "persistencePass": bool(row.get("persistence_pass", False) or row.get("persistencePass", False)),
        "sectorStabilityPass": bool(
            row.get("sector_stability_pass", False) or row.get("sectorStabilityPass", False)
        ),
        "transportabilityPass": bool(
            row.get("transportability_pass", False) or row.get("transportabilityPass", False)
        ),
        "claimBoundary": (
            "Globalized H3-fitted holonomy/defect worldline candidate. Not a matter particle unless "
            "the global particle receipt passes."
        ),
    }


def _worldline_path_metrics(events: list[dict[str, Any]]) -> tuple[float, float]:
    points = [event.get("h3SpatialPoint") for event in events if isinstance(event.get("h3SpatialPoint"), list)]
    if len(points) < 2:
        return 0.0, 0.0
    length = 0.0
    for left, right in zip(points, points[1:], strict=False):
        if len(left) < 3 or len(right) < 3:
            continue
        length += h3_distance(left[:4] if len(left) >= 4 else left[:3], right[:4] if len(right) >= 4 else right[:3])
    return float(length), float(length / max(len(points) - 1, 1))


def _write_global_no_data_use_receipt(roots: list[Path], out_dir: Path) -> dict[str, Any]:
    source_names = (
        "finite_repair_transition_matrix_report.json",
        "finite_certificate_report.json",
        "B_A_kernel_report.json",
        "B_A_kernel_refinement_report.json",
        "b_a_parent_report.json",
        "scale_compressed_repair_report.json",
        "strict_neutral_bulk_report.json",
        "scalar_quotient_report.json",
        "screen_capacity_closure_report.json",
    )
    source_status: dict[str, Any] = {}
    measurement_used = False
    for name in source_names:
        reports = _collect_named_reports(roots, name)
        flags = [_measurement_data_used_for_input(report) for report in reports]
        if any(flags):
            measurement_used = True
        source_status[name] = {
            "report_count": len(reports),
            "measurement_data_used": any(flags),
        }
    report = {
        "mode": "distributed_physical_cmb_global_no_data_use_receipt_v0",
        "no_data_use_receipt": not measurement_used,
        "NO_DATA_USE_RECEIPT": not measurement_used,
        "measurement_data_used_for_input_functions": measurement_used,
        "source_status": source_status,
        "run_dirs": [str(path) for path in roots],
        "claim_boundary": (
            "Global no-data-use firewall across shard-local and reduced OPH CMB input sources. "
            "Measurement comparison reports may exist, but they must not set OPH input functions."
        ),
    }
    _write_json(Path(out_dir) / "no_data_use_receipt.json", report)
    return report


def _measurement_data_used_for_input(report: dict[str, Any]) -> bool:
    if not report:
        return False
    explicit_false = (
        report.get("no_cmb_data_used") is True
        or (((report.get("readiness") or {}).get("checks") or {}).get("no_cmb_data_used") is True)
    )
    data_use_keys = (
        "measurement_data_used",
        "cmb_data_used",
        "cmb_data_used_for_input",
        "planck_data_used_for_input",
        "fit_to_measurement",
        "fit_to_planck",
        "uses_measurements_to_set_inputs",
    )
    measurement_flag = any(bool(report.get(key, False)) for key in data_use_keys)
    if explicit_false and measurement_flag:
        return True
    if explicit_false:
        return False
    return measurement_flag


def _collect_named_reports(roots: list[Path], name: str) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for root in roots:
        root = Path(root)
        candidates = [root / name]
        if root.exists() and root.is_dir():
            candidates.extend(sorted(root.glob(f"**/{name}")))
        seen: set[Path] = set()
        for path in candidates:
            path = Path(path)
            if path in seen:
                continue
            seen.add(path)
            data = _read_json(path)
            if data:
                reports.append(data)
                break
    return reports


def _transition_eta_value(report: dict[str, Any]) -> float | None:
    primary = report.get("primary") or {}
    value = _float_or_none(primary.get("eta_R_estimate"))
    if value is not None:
        return value
    empirical = (report.get("clock_modes") or {}).get("empirical") or {}
    return _float_or_none(empirical.get("eta_R_value"))


def _finite_cert_a_zeta(report: dict[str, Any]) -> float | None:
    derived = report.get("derived_outputs") or {}
    return _float_or_none(derived.get("A_zeta", report.get("A_zeta")))


def _finite_values(reports: list[dict[str, Any]], getter: Any) -> list[float]:
    values: list[float] = []
    for report in reports:
        value = getter(report)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _numeric_rows_from_reports(reports: list[dict[str, Any]], key_path: tuple[str, ...]) -> list[list[float]]:
    rows: list[list[float]] = []
    for report in reports:
        value: Any = report
        for key in key_path:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                value = None
                break
        rows.extend(_numeric_rows(value))
    return rows


def _numeric_rows(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return [[float(value)]]
    if not isinstance(value, list):
        return []
    rows: list[list[float]] = []
    if value and all(isinstance(item, (int, float)) for item in value):
        row = [float(item) for item in value if math.isfinite(float(item))]
        return [row] if row else []
    for item in value:
        if isinstance(item, list):
            row = []
            for cell in item:
                if isinstance(cell, (int, float)) and math.isfinite(float(cell)):
                    row.append(float(cell))
            if row:
                rows.append(row)
    return rows


def _collect_row_dicts(reports: list[dict[str, Any]], row_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        for key in row_keys:
            value = report.get(key)
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        rows.append(dict(row))
                if value:
                    break
    return rows


def _mean_or_none(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


def _consensus_or_none(values: list[float], *, rtol: float = 1.0e-9, atol: float = 1.0e-12) -> float | None:
    if not values:
        return None
    first = float(values[0])
    scale = max(abs(first), 1.0)
    if all(abs(float(value) - first) <= max(atol, rtol * scale) for value in values):
        return first
    return None


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": _mean_or_none(values),
        "std": _std_or_none(values),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _median_int_or_none(values: list[int | None]) -> int | None:
    finite = sorted(int(value) for value in values if value is not None)
    if not finite:
        return None
    return int(finite[len(finite) // 2])


def _unique_texts(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _file_sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
