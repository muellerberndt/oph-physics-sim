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

from oph_fpe.experiments import load_config
from oph_fpe.cosmology.finite_repair_transition_clock import (
    validate_transition_clock_eligibility,
)
from oph_fpe.observers.semantic_clock import (
    OBSERVER_KINDS,
    distributed_observer_uid,
    normalize_observer_frame,
    observer_registry_audit,
    semantic_history_digest,
)
from oph_fpe.viz.universe_timeline_viewer import (
    H3_COORDINATE_SYSTEM,
    _assumed_ds4_visualization_payload,
    _geometry_and_symmetry_payload,
    _h3_distance,
    _h3_coordinate_contract,
    _subjective_observer_cameras,
    _simulation_assumption_payload,
    _visualization_render_data_payload,
    _visualization_views_payload,
    _cmb_payload,
    _effective_string_theory_payload,
    _emergent_curved_spacetime_payload,
    _hilbert_space_observer_algebra_payload,
    _observer_anatomy_payload,
    _observer_cinema_payload,
)
from oph_fpe.simulation_assumptions import (
    manifest_assumptions_pass,
    revalidate_simulation_assumption_manifest,
    simulation_assumption_manifest,
)
from oph_fpe.viz.visualizer_pack import build_visualizer_pack


DISTRIBUTED_KERNEL_VERSION = "distributed_observer_patch_kernel_v1"


def _literal_true(value: Any) -> bool:
    """Return true only for a JSON boolean true, never a truthy string/number."""

    return value is True

RECEIPT_KEYS: tuple[str, ...] = (
    "observer_like_self_reading_system_receipt",
    "observer_modular_time_receipt",
    "h3_response_candidate_receipt",
    "h3_response_control_separation_receipt",
    "observer_h3_object_population_receipt",
    "observer_facing_3p1d_h3_experience_receipt",
    "theorem_assisted_consensus_3d_bulk_readout_receipt",
    "observer_facing_consensus_3d_bulk_readout_receipt",
    "chart_blind_strict_neutral_quotient_bulk_receipt",
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
    """Prepare a theorem-aligned distributed OPH universe run.

    Each shard is a bounded observer-like self-reading screen patch. The reducer
    may certify a federated witness, but it must not promote the run to a single
    chart-blind strict neutral quotient bulk unless future cross-shard repair/overlap receipts exist.
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
                "owned_node_ranges": carrier_shard.get("owned_node_ranges", []),
                "owned_node_count": carrier_shard.get("owned_node_count", patch_count_per_shard),
                "ghost_nodes": carrier_shard.get("ghost_nodes", []),
                "cut_edge_ids": carrier_shard.get("cut_edge_ids", []),
            }
        )

    observer_view_reduction_cfg = dict(
        ((base_config.get("distributed_universe") or {}).get("observer_view_reduction") or {})
    )
    observer_view_reduction = {
        "max_total": int(observer_view_reduction_cfg.get("max_total", 4096) or 4096),
        "per_shard": int(observer_view_reduction_cfg.get("per_shard", 512) or 512),
    }

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
        "observer_view_reduction": observer_view_reduction,
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
                    "bulk_3d_established": _literal_true(emergence.get("bulk_3d_established")),
                    "particle_matter_receipt": _literal_true(emergence.get("particle_matter_receipt")),
                },
            }
        )

    distributed_assumptions = _distributed_simulation_assumption_manifest(
        manifest_path=Path(manifest_path),
        manifest=manifest,
        shard_rows=shard_rows,
    )
    _write_json(out_dir / "simulation_assumption_manifest.json", distributed_assumptions)

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
    cross_shard_overlap_repair_receipt = _literal_true(
        halo_exchange.get("online_cross_shard_overlap_repair_receipt")
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
        assumption_manifest=distributed_assumptions,
        assumption_manifest_source="simulation_assumption_manifest.json",
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

    visualization_payload_path = out_dir / "distributed_visualization_payload.json"
    _write_json(visualization_payload_path, visualization_payload)
    visualizer_pack = build_visualizer_pack(
        bundle_dir=out_dir,
        out_path=out_dir / "oph_visualizer_pack_v2.tar.zst",
        payload=visualization_payload,
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
        "global_carrier_contract_receipt": _literal_true(
            global_carrier.get("global_carrier_contract_receipt")
        ),
        "one_global_carrier_before_partition_receipt": _literal_true(
            global_carrier.get("one_global_carrier_before_partition_receipt")
        ),
        "stable_global_identity_initial_state_receipt": _literal_true(
            global_carrier.get("stable_global_identity_initial_state_receipt")
        ),
        "authoritative_owner_projection_receipt": _literal_true(
            global_carrier.get("authoritative_owner_projection_receipt")
        ),
        "distributed_realization_event_certificate_receipt": _literal_true(
            global_carrier.get("distributed_realization_event_certificate_receipt")
        ),
        "federated_large_universe_witness_receipt": federated_receipt,
        "seam_metadata_replay_receipt": seam_metadata_replay_receipt,
        "cross_shard_overlap_repair_receipt": cross_shard_overlap_repair_receipt,
        "online_cross_shard_overlap_repair_receipt": _literal_true(
            halo_exchange.get("online_cross_shard_overlap_repair_receipt")
        ),
        "per_cycle_cross_shard_halo_exchange_receipt": _literal_true(
            halo_exchange.get("per_cycle_cross_shard_halo_exchange_receipt")
        ),
        "reducer_halo_exchange_replay_receipt": _literal_true(
            halo_exchange.get("reducer_halo_exchange_replay_receipt")
        ),
        "global_observer_modular_time_export_receipt": _literal_true(
            observer_time_global.get("global_observer_modular_time_export_receipt")
        ),
        "global_proto_particle_worldline_export_receipt": _literal_true(
            proto_particle_global.get("global_proto_particle_worldline_export_receipt")
        ),
        "global_pn_resonance_receipt": _literal_true(pn_global.get("global_pn_resonance_receipt")),
        "all_shards_local_scale_compressed_pn_witness_receipt": _literal_true(
            pn_global.get("all_shards_local_scale_compressed_pn_witness_receipt")
        ),
        "global_physical_cmb_input_contract_receipt": _literal_true(
            physical_cmb_global.get("physical_cmb_input_contract_receipt")
        ),
        "global_physical_cmb_output_comparison_receipt": _literal_true(
            physical_cmb_global.get("physical_cmb_output_comparison_receipt")
        ),
        "global_physical_cmb_prediction_receipt": _literal_true(
            physical_cmb_global.get("physical_cmb_prediction_receipt")
        ),
        "physical_cmb_global_reduction": physical_cmb_global,
        "strict_single_global_neutral_bulk_receipt": _literal_true(
            neutral_global.get("strict_single_global_neutral_bulk_receipt")
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
        "simulation_assumptions": distributed_assumptions,
        "visualization_payload": str(visualization_payload_path),
        "visualizer_pack": visualizer_pack,
        "run_pack_contract_path": str(out_dir / "DISTRIBUTED_RUN_PACK_CONTRACT.json"),
        "observer_like_self_reading_system_note": (
            "The distributed unit is still an OPH observer-like self-reading system: each shard has local "
            "state, ports/boundaries, readback, records, feedback/repair moves, and public evidence files."
        ),
        "shards": shard_rows,
    }
    _write_json(out_dir / "distributed_universe_summary.json", summary)
    _write_markdown(out_dir / "DISTRIBUTED_UNIVERSE_SUMMARY.md", summary)
    return summary


def _distributed_simulation_assumption_manifest(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    shard_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recover one explicit visual-assumption ledger for the reduced universe."""

    shard_manifests: list[dict[str, Any]] = []
    for row in shard_rows:
        run_dir = row.get("run_dir")
        if not run_dir:
            continue
        candidate = Path(str(run_dir)) / "simulation_assumption_manifest.json"
        parsed = _read_json(candidate)
        if parsed:
            shard_manifests.append(revalidate_simulation_assumption_manifest(parsed))

    expected_manifest_count = len([row for row in shard_rows if row.get("run_dir")])
    if shard_manifests and len(shard_manifests) == expected_manifest_count:
        comparison_keys = (
            "schema",
            "profile",
            "scope",
            "policy_id",
            "assumptions",
            "ds4_visualization_parameters",
            "observer_camera_visualization_parameters",
            "cmb_visualization_parameters",
        )
        fingerprints = {
            json.dumps(
                {key: manifest.get(key) for key in comparison_keys},
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            for manifest in shard_manifests
        }
        if len(fingerprints) == 1 and all(
            manifest.get("manifest_integrity_valid") is True for manifest in shard_manifests
        ):
            return shard_manifests[0]

    raw_config_path = manifest.get("config_path")
    if raw_config_path:
        configured = Path(str(raw_config_path))
        candidates = [configured] if configured.is_absolute() else [
            Path.cwd() / configured,
            Path(manifest_path).parent / configured,
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                return simulation_assumption_manifest(load_config(candidate))
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                continue
    fallback = simulation_assumption_manifest({})
    if shard_manifests:
        fallback["distributed_assumption_reduction_blockers"] = [
            "shard_assumption_manifests_missing_or_not_identical"
        ]
    return fallback


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
    reported_per_cycle = bool(
        expected
        and len(reports) == expected
        and all(
            _literal_true(report.get("PER_CYCLE_HALO_EXCHANGE_RECEIPT"))
            or _literal_true(report.get("per_cycle_halo_exchange_receipt"))
            for report in reports
        )
    )
    # No worker currently drives a live transactional seam kernel or emits
    # replayable reciprocal packet/commit primitives.  Self-authored receipt
    # booleans must therefore remain diagnostic and cannot promote D1.
    live_seam_kernel_implemented = False
    distributed_online_evidence_recomputed_receipt = False
    per_cycle = bool(
        reported_per_cycle
        and live_seam_kernel_implemented
        and distributed_online_evidence_recomputed_receipt
    )
    seam_links = list(seam_readout.get("links") or [])
    frame_rows = _global_halo_replay_frames(seam_links)
    replay_receipt = bool(seam_links and frame_rows)
    seam_metadata_replay_receipt = bool(
        _literal_true(seam_readout.get("seam_metadata_replay_receipt")) or replay_receipt
    )
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
    reported_online_receipts = {
        key: _all_reports_truthy(reports, key, expected)
        for key in online_receipt_keys
    }
    online_receipts = {
        key: bool(
            value
            and live_seam_kernel_implemented
            and distributed_online_evidence_recomputed_receipt
        )
        for key, value in reported_online_receipts.items()
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
    if not live_seam_kernel_implemented:
        blockers.append("live_transactional_seam_kernel_not_implemented")
    if not distributed_online_evidence_recomputed_receipt:
        blockers.append("distributed_online_evidence_not_independently_recomputed")
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
        "live_seam_kernel_implemented": live_seam_kernel_implemented,
        "distributed_online_evidence_recomputed_receipt": (
            distributed_online_evidence_recomputed_receipt
        ),
        "reported_per_cycle_halo_exchange_claim": reported_per_cycle,
        "per_cycle_cross_shard_halo_exchange_receipt": per_cycle,
        "online_cross_shard_overlap_repair_receipt": online_cross_shard_repair,
        "DISTRIBUTED_KERNEL_SCALING_READY_RECEIPT": kernel_scaling_ready,
        "required_online_seam_receipts": online_receipts,
        "reported_online_seam_receipt_claims": reported_online_receipts,
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
            "remain disabled until a runtime transactional seam producer emits reciprocal seam packets, visible restrictions, descent, atomic "
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
    return any(report.get(key) is True for key in candidates)


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
    reduction_limits = _observer_view_reduction_limits(manifest)
    observer_views = _combined_observer_views(
        completed,
        max_total=reduction_limits["max_total"],
        per_shard=reduction_limits["per_shard"],
    )
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
    online_evidence_recomputed = _literal_true(
        halo_exchange.get("distributed_online_evidence_recomputed_receipt")
    )
    online_halo = _literal_true(halo_exchange.get("per_cycle_cross_shard_halo_exchange_receipt"))
    neutral_ready = bool(
        _literal_true(neutral_report.get("strict_neutral_bulk"))
        or _literal_true(frontier.get("strict_neutral_bulk_ready"))
    )
    strict_receipt = bool(
        neutral_ready
        and cross_shard_overlap_repair_receipt
        and online_halo
        and online_evidence_recomputed
    )
    blockers = _unique_texts(
        list(neutral_report.get("blockers") or [])
        + list(frontier.get("blockers") or [])
        + ([] if observer_views else ["global_observer_views_missing"])
        + ([] if cross_shard_overlap_repair_receipt else ["online_cross_shard_overlap_repair_receipt_missing"])
        + ([] if online_halo else ["per_cycle_cross_shard_halo_exchange_receipt_missing"])
        + ([] if online_evidence_recomputed else ["distributed_online_evidence_not_independently_recomputed"])
        + ([] if neutral_ready else ["global_strict_neutral_bulk_ready_false"])
    )
    report = {
        "mode": "distributed_global_neutral_bulk_reduction_v0",
        "run_id": manifest.get("run_id"),
        "expected_shard_count": int(manifest.get("shard_count", 0)),
        "completed_shard_count": len(shard_roots),
        "combined_observer_view_count": len(observer_views),
        "observer_view_reduction_limits": reduction_limits,
        "combined_observer_views_path": str(combined_path) if observer_views else None,
        "cross_shard_overlap_repair_receipt": bool(cross_shard_overlap_repair_receipt),
        "online_cross_shard_overlap_repair_receipt": bool(cross_shard_overlap_repair_receipt),
        "seam_metadata_replay_receipt": _literal_true(
            halo_exchange.get("seam_metadata_replay_receipt")
        ),
        "per_cycle_cross_shard_halo_exchange_receipt": online_halo,
        "distributed_online_evidence_recomputed_receipt": online_evidence_recomputed,
        "global_strict_neutral_bulk_ready": neutral_ready,
        "strict_single_global_neutral_bulk_receipt": strict_receipt,
        "strict_neutral_bulk_report_path": str(out_dir / "strict_neutral_bulk_report.json"),
        "strict_neutral_bulk_frontier_path": str(out_dir / "strict_neutral_bulk_frontier_report.json"),
        "frontier": {
            "strict_neutral_bulk": _literal_true(frontier.get("strict_neutral_bulk")),
            "strict_neutral_bulk_ready": _literal_true(frontier.get("strict_neutral_bulk_ready")),
            "gate_rows": frontier.get("gate_rows") or [],
        },
        "blockers": blockers,
        "claim_boundary": (
            "Global neutral-bulk reduction over the distributed atlas. The strict single-bulk receipt requires "
            "a global neutral audit plus cross-shard overlap repair and live per-cycle halo exchange. "
            "A reducer-only replay cannot certify chart-blind strict neutral quotient bulk."
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
            if _literal_true(view.get("execution_clock_fields_separated_receipt")):
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
            "clock naturality or chart-blind strict neutral quotient bulk."
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
        _literal_true(report.get("scale_compressed_pn_silence_to_observation_receipt"))
        or _literal_true(report.get("PN_RESONANCE_RECEIPT"))
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

    nodes = np.arange(total_nodes, dtype=np.int64)
    owners = nodes // max(patch_count_per_shard, 1)
    edge_array = _global_graph_edge_array(total_nodes)
    cut_edge_mask = (
        owners[edge_array[:, 0]] != owners[edge_array[:, 1]]
        if edge_array.size
        else np.zeros(0, dtype=np.bool_)
    )
    state_words = _vectorized_stable_state_words(nodes, run_id=run_id, base_seed=base_seed)
    boundary_sector = np.remainder(state_words, 17).astype(np.int16, copy=False)

    graph_path = carrier_dir / "global_graph.npz"
    np.savez(
        graph_path,
        nodes=nodes,
        edges=edge_array,
        node_owner=owners,
        cut_edge_mask=cut_edge_mask,
    )
    state_path = carrier_dir / "global_initial_state.npz"
    np.savez(
        state_path,
        nodes=nodes,
        canonical_state_word=state_words,
        boundary_sector=boundary_sector,
    )

    cut_edges = _cut_edge_rows(edge_array, owners, cut_edge_mask)
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
        _literal_true(graph_info.get("load_receipt"))
        and graph_info.get("node_count") == total_nodes
        and graph_info.get("edge_count", 0) >= max(total_nodes - 1, 0)
    )
    if not graph_receipt:
        blockers.append("global_graph_shape_invalid")
    state_receipt = bool(
        _literal_true(state_info.get("load_receipt"))
        and state_info.get("node_count") == total_nodes
        and state_info.get("stable_identity_rule") == "blake2b_keyed_splitmix64_patch_state_v1_uint63"
    )
    if not state_receipt:
        blockers.append("global_initial_state_invalid")
    owner_encoding = partition.get("node_owner_encoding") if isinstance(partition.get("node_owner_encoding"), dict) else {}
    encoded_ranges = owner_encoding.get("ranges") if isinstance(owner_encoding.get("ranges"), list) else []
    compact_owner_encoding_receipt = bool(
        owner_encoding.get("mode") == "contiguous_half_open_ranges"
        and len(encoded_ranges) == int(manifest.get("shard_count", 0) or 0)
        and _owner_ranges_cover_carrier(encoded_ranges, total_nodes)
        and graph_info.get("owner_ranges") == encoded_ranges
    )
    partition_receipt = bool(
        partition.get("schema") == "oph_distributed_partition_map_v2"
        and partition.get("node_count") == total_nodes
        and partition.get("synthetic_partition") is False
        and shard_ids_match
        and compact_owner_encoding_receipt
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
        registry.get("schema") == "oph_global_observer_registry_v3"
        and registry.get("observer_count") == total_observers
        and registry.get("stable_identity_rule") == "blake2b_run_observer_kind_global_index_base_seed_uint63"
        and registry.get("registry_encoding") == "range_derived_no_per_observer_json_rows"
        and registry.get("observer_kinds") == list(OBSERVER_KINDS)
        and registry.get("registered_identity_count") == total_observers * len(OBSERVER_KINDS)
        and _literal_true(registry.get("global_observer_registry_namespace_receipt"))
    )
    if not observer_registry_receipt:
        blockers.append("global_observer_registry_invalid")

    artifact_receipt = all(
        _literal_true((artifact_status.get(key) or {}).get("hash_receipt")) for key in required
    )
    owner_receipt = bool(partition_receipt and graph_receipt and cut_receipt)
    event_certificate = carrier.get("event_certificate") if isinstance(carrier.get("event_certificate"), dict) else {}
    event_receipt = bool(
        _literal_true(event_certificate.get("linearized_committed_event_log_receipt"))
        and _literal_true(event_certificate.get("monolithic_normal_form_certificate_receipt"))
        and _literal_true(event_certificate.get("final_readout_recomputed_receipt"))
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
        _literal_true(carrier.get("one_global_carrier_before_partition"))
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
            _literal_true(carrier.get("one_global_carrier_before_partition")) and artifact_receipt
        ),
        "manifest_declared_global_artifacts_receipt": artifact_receipt,
        "global_graph_receipt": graph_receipt,
        "partition_map_receipt": partition_receipt,
        "compact_owner_range_encoding_receipt": compact_owner_encoding_receipt,
        "cut_interface_receipt": cut_receipt,
        "stable_global_identity_initial_state_receipt": state_receipt,
        "global_observer_registry_receipt": observer_registry_receipt,
        "authoritative_owner_projection_receipt": owner_receipt,
        "config_hash_receipt": config_hash_receipt,
        "code_hash_receipt": code_hash_receipt,
        "run_id_receipt": run_id_receipt,
        "distributed_realization_event_certificate_receipt": event_receipt,
        "monolithic_normal_form_certificate_receipt": _literal_true(
            event_certificate.get("monolithic_normal_form_certificate_receipt")
        ),
        "final_readout_recomputed_receipt": _literal_true(
            event_certificate.get("final_readout_recomputed_receipt")
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


def _global_graph_edge_array(node_count: int) -> np.ndarray:
    """Build the declared carrier ring without Python dicts per edge."""

    count = int(node_count)
    if count <= 1:
        return np.zeros((0, 2), dtype=np.int64)
    sources = np.arange(count - 1, dtype=np.int64)
    targets = sources + 1
    edges = np.column_stack((sources, targets))
    if count > 2:
        edges = np.vstack((edges, np.asarray([[count - 1, 0]], dtype=np.int64)))
    return np.ascontiguousarray(edges, dtype=np.int64)


def _cut_edge_rows(
    edges: np.ndarray,
    owners: np.ndarray,
    cut_mask: np.ndarray,
) -> list[dict[str, Any]]:
    """Materialize only O(shard_count) cut-edge metadata for JSON receipts."""

    rows: list[dict[str, Any]] = []
    for edge_index in np.flatnonzero(np.asarray(cut_mask, dtype=bool)):
        source, target = (int(value) for value in edges[int(edge_index)])
        rows.append(
            {
                "edge_id": f"e{int(edge_index):06d}",
                "source_node": source,
                "target_node": target,
                "source_owner": int(owners[source]),
                "target_owner": int(owners[target]),
                "is_cut_edge": True,
            }
        )
    return rows


def _vectorized_stable_state_words(
    nodes: np.ndarray,
    *,
    run_id: str,
    base_seed: int,
) -> np.ndarray:
    """Deterministic SplitMix64 state words keyed once by run and seed."""

    key = np.uint64(_stable_uint63(run_id, "patch_state_vector_v1", int(base_seed)))
    values = np.asarray(nodes, dtype=np.uint64) + key + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    values = values ^ (values >> np.uint64(31))
    values &= np.uint64((1 << 63) - 1)
    return values.astype(np.int64, copy=False)


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
        patch_stop = patch_start + patch_count_per_shard
        observer_start = index * observers_per_shard
        shards.append(
            {
                "shard_index": index,
                "shard_id": f"{run_id}_shard{index:04d}",
                "owned_node_ranges": [[patch_start, patch_stop]],
                "owned_node_count": int(patch_count_per_shard),
                "ghost_nodes": sorted(ghost_nodes_by_shard[index]),
                "cut_edge_ids": sorted(edge_ids_by_shard[index]),
                "global_patch_range": [patch_start, patch_stop],
                "global_observer_range": [observer_start, observer_start + observers_per_shard],
            }
        )
    return {
        "schema": "oph_distributed_partition_map_v2",
        "run_id": run_id,
        "node_count": int(len(owners)),
        "shard_count": int(shard_count),
        "node_owner_encoding": {
            "mode": "contiguous_half_open_ranges",
            "ranges": [
                [
                    int(index * patch_count_per_shard),
                    int((index + 1) * patch_count_per_shard),
                    int(index),
                ]
                for index in range(shard_count)
            ],
        },
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
    samples: list[dict[str, Any]] = []
    sample_limit = max(1, min(12, shard_count * max(observers_per_shard, 1)))
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
        remaining_sample_observers = max(0, math.ceil((sample_limit - len(samples)) / max(len(OBSERVER_KINDS), 1)))
        for local_observer_index in range(min(observers_per_shard, remaining_sample_observers)):
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
                if len(samples) < sample_limit:
                    samples.append(entry)
    sample_audit = observer_registry_audit(samples)
    total_observers = int(shard_count) * int(observers_per_shard)
    registered_identity_count = total_observers * len(OBSERVER_KINDS)
    analytic_namespace_receipt = bool(
        run_id
        and shard_count > 0
        and observers_per_shard > 0
        and len(set(OBSERVER_KINDS)) == len(OBSERVER_KINDS)
        and sample_audit.get("GLOBAL_OBSERVER_REGISTRY_NAMESPACE_RECEIPT") is True
    )
    audit = {
        "mode": "observer_registry_range_namespace_audit_v2",
        "GLOBAL_OBSERVER_REGISTRY_NAMESPACE_RECEIPT": analytic_namespace_receipt,
        "receipt": analytic_namespace_receipt,
        "encoding": "cartesian_product_of_global_observer_range_and_kind_enum",
        "registered_identity_count": registered_identity_count,
        "sample_audit": sample_audit,
        "injectivity_argument": (
            "Within one run_id, (observer_kind, global_observer_index) is serialized injectively; "
            "global observer half-open ranges are disjoint by shard construction."
        ),
    }
    return {
        "schema": "oph_global_observer_registry_v3",
        "run_id": run_id,
        "observer_count": total_observers,
        "registered_identity_count": registered_identity_count,
        "registry_encoding": "range_derived_no_per_observer_json_rows",
        "observer_kinds": list(OBSERVER_KINDS),
        "namespace_rule": "distributed_observer_uid = run_id:observer_kind:global_observer_index",
        "stable_identity_rule": "blake2b_run_observer_kind_global_index_base_seed_uint63",
        "registry_namespace_audit": audit,
        "global_observer_registry_namespace_receipt": analytic_namespace_receipt,
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


def _owner_ranges_cover_carrier(ranges: list[Any], node_count: int) -> bool:
    expected_start = 0
    for expected_owner, row in enumerate(ranges):
        if not isinstance(row, list) or len(row) != 3:
            return False
        try:
            start, stop, owner = (int(value) for value in row)
        except (TypeError, ValueError):
            return False
        if start != expected_start or stop <= start or owner != expected_owner:
            return False
        expected_start = stop
    return bool(expected_start == int(node_count))


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
            nodes = np.asarray(data["nodes"])
            edges = np.asarray(data["edges"])
            owners = np.asarray(data["node_owner"])
            cut_mask = np.asarray(data["cut_edge_mask"])
            node_count = int(nodes.size) if nodes.ndim == 1 else -1
            nodes_sequential = bool(
                node_count >= 0
                and np.issubdtype(nodes.dtype, np.integer)
                and np.array_equal(nodes, np.arange(node_count, dtype=nodes.dtype))
            )
            edge_shape_valid = bool(edges.ndim == 2 and edges.shape[1:] == (2,))
            edge_endpoints_valid = bool(
                edge_shape_valid
                and np.issubdtype(edges.dtype, np.integer)
                and (
                    edges.size == 0
                    or (np.all(edges >= 0) and np.all(edges < max(node_count, 0)))
                )
            )
            owners_valid = bool(
                owners.ndim == 1
                and owners.shape == (max(node_count, 0),)
                and np.issubdtype(owners.dtype, np.integer)
                and (owners.size == 0 or np.all(owners >= 0))
            )
            expected_ring = _global_graph_edge_array(max(node_count, 0))
            ring_topology_receipt = bool(
                edge_endpoints_valid
                and edges.shape == expected_ring.shape
                and np.array_equal(edges, expected_ring)
            )
            computed_cut_mask = (
                owners[edges[:, 0]] != owners[edges[:, 1]]
                if edge_endpoints_valid and owners_valid and edges.size
                else np.zeros(edges.shape[0] if edge_shape_valid else 0, dtype=np.bool_)
            )
            cut_mask_receipt = bool(
                cut_mask.ndim == 1
                and edge_shape_valid
                and cut_mask.shape == (edges.shape[0],)
                and np.issubdtype(cut_mask.dtype, np.bool_)
                and np.array_equal(cut_mask, computed_cut_mask)
            )
            owner_ranges = _owner_ranges_from_array(owners) if owners_valid else []
            owner_range_receipt = _owner_ranges_cover_carrier(owner_ranges, max(node_count, 0))
            load_receipt = bool(
                nodes_sequential
                and edge_endpoints_valid
                and owners_valid
                and ring_topology_receipt
                and cut_mask_receipt
                and owner_range_receipt
            )
            return {
                "load_receipt": load_receipt,
                "node_count": node_count,
                "edge_count": int(edges.shape[0]) if edge_shape_valid else -1,
                "owner_count": len(owner_ranges),
                "owner_ranges": owner_ranges,
                "cut_edge_count": int(np.count_nonzero(cut_mask)),
                "nodes_sequential_receipt": nodes_sequential,
                "edge_endpoints_receipt": edge_endpoints_valid,
                "ring_topology_receipt": ring_topology_receipt,
                "cut_edge_mask_recomputed_receipt": cut_mask_receipt,
                "owner_range_receipt": owner_range_receipt,
            }
    except Exception as exc:  # pragma: no cover - defensive artifact report path.
        return {"load_receipt": False, "error": f"{type(exc).__name__}:{exc}"}


def _npz_state_info(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {"load_receipt": False}
    try:
        with np.load(path) as data:
            nodes = np.asarray(data["nodes"])
            state = np.asarray(data["canonical_state_word"])
            sectors = np.asarray(data["boundary_sector"])
            node_count = int(nodes.size) if nodes.ndim == 1 else -1
            nodes_sequential = bool(
                node_count >= 0
                and np.issubdtype(nodes.dtype, np.integer)
                and np.array_equal(nodes, np.arange(node_count, dtype=nodes.dtype))
            )
            state_shape_valid = bool(
                state.ndim == 1
                and state.shape == (max(node_count, 0),)
                and np.issubdtype(state.dtype, np.integer)
                and (state.size == 0 or np.all(state >= 0))
            )
            sector_recomputed = bool(
                sectors.ndim == 1
                and sectors.shape == (max(node_count, 0),)
                and np.issubdtype(sectors.dtype, np.integer)
                and state_shape_valid
                and np.array_equal(sectors, np.remainder(state, 17))
            )
            return {
                "load_receipt": bool(nodes_sequential and state_shape_valid and sector_recomputed),
                "node_count": node_count,
                "stable_identity_rule": "blake2b_keyed_splitmix64_patch_state_v1_uint63",
                "state_word_dtype": str(state.dtype),
                "nodes_sequential_receipt": nodes_sequential,
                "state_shape_receipt": state_shape_valid,
                "boundary_sector_recomputed_receipt": sector_recomputed,
            }
    except Exception as exc:  # pragma: no cover - defensive artifact report path.
        return {"load_receipt": False, "error": f"{type(exc).__name__}:{exc}"}


def _owner_ranges_from_array(owners: np.ndarray) -> list[list[int]]:
    """Return the compact owner runs encoded by a dense NPZ owner vector."""

    values = np.asarray(owners)
    if values.ndim != 1 or values.size == 0:
        return []
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate((np.asarray([0], dtype=np.int64), changes))
    stops = np.concatenate((changes, np.asarray([values.size], dtype=np.int64)))
    return [
        [int(start), int(stop), int(values[int(start)])]
        for start, stop in zip(starts, stops, strict=True)
    ]


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
        "global_carrier_contract_receipt": _literal_true(
            global_carrier.get("global_carrier_contract_receipt")
        ),
        "one_global_carrier_before_partition_receipt": _literal_true(
            global_carrier.get("one_global_carrier_before_partition_receipt")
        ),
        "manifest_declared_global_artifacts_receipt": _literal_true(
            global_carrier.get("manifest_declared_global_artifacts_receipt")
        ),
        "partition_map_receipt": _literal_true(global_carrier.get("partition_map_receipt")),
        "cut_interface_receipt": _literal_true(global_carrier.get("cut_interface_receipt")),
        "stable_global_identity_initial_state_receipt": _literal_true(
            global_carrier.get("stable_global_identity_initial_state_receipt")
        ),
        "global_observer_registry_receipt": _literal_true(
            global_carrier.get("global_observer_registry_receipt")
        ),
        "authoritative_owner_projection_receipt": _literal_true(
            global_carrier.get("authoritative_owner_projection_receipt")
        ),
        "distributed_realization_event_certificate_receipt": _literal_true(
            global_carrier.get("distributed_realization_event_certificate_receipt")
        ),
        "distributed_online_evidence_recomputed_receipt": _literal_true(
            halo_exchange.get("distributed_online_evidence_recomputed_receipt")
        ),
        "seam_metadata_replay_receipt": bool(seam_metadata_replay_receipt),
        "online_cross_shard_overlap_repair_receipt": _literal_true(
            halo_exchange.get("online_cross_shard_overlap_repair_receipt")
        ),
        "per_cycle_cross_shard_halo_exchange_receipt": _literal_true(
            halo_exchange.get("per_cycle_cross_shard_halo_exchange_receipt")
        ),
        "distributed_local_diamond_receipt": _literal_true(
            online_receipts.get("DISTRIBUTED_LOCAL_DIAMOND_RECEIPT")
        ),
        "distributed_repair_completeness_receipt": _literal_true(
            online_receipts.get("DISTRIBUTED_REPAIR_COMPLETENESS_RECEIPT")
        ),
        "cycle_holonomy_zero_or_classified_receipt": _literal_true(
            online_receipts.get("CYCLE_HOLONOMY_ZERO_OR_CLASSIFIED_RECEIPT")
        ),
        "selected_fiber_nontrivial_elimination_receipt": _literal_true(
            online_receipts.get("SELECTED_FIBER_NONTRIVIAL_ELIMINATION_RECEIPT")
        ),
        "same_boundary_multistart_confluence_receipt": _literal_true(
            online_receipts.get("SAME_BOUNDARY_MULTISTART_CONFLUENCE_RECEIPT")
        ),
        "quotient_normal_form_canonical_hash_receipt": _literal_true(
            online_receipts.get("QUOTIENT_NORMAL_FORM_CANONICAL_HASH_RECEIPT")
        ),
        "fair_block_contraction_receipt": _literal_true(
            online_receipts.get("FAIR_BLOCK_CONTRACTION_RECEIPT")
        ),
        "schedule_independent_normal_form_receipt": _literal_true(
            online_receipts.get("SCHEDULE_INDEPENDENT_NORMAL_FORM_RECEIPT")
        ),
        "partition_naturality_receipt": _literal_true(
            online_receipts.get("PARTITION_NATURALITY_RECEIPT")
        ),
        "global_observer_modular_time_export_receipt": _literal_true(
            observer_time_global.get("global_observer_modular_time_export_receipt")
        ),
        "large_visualization_observer_contract_receipt": _literal_true(
            observer_time_global.get("large_visualization_observer_contract_receipt")
        ),
        "global_proto_particle_worldline_export_receipt": _literal_true(
            proto_particle_global.get("global_proto_particle_worldline_export_receipt")
        ),
        "moving_proto_particle_candidate_receipt": _literal_true(
            proto_particle_global.get("moving_proto_particle_candidate_receipt")
        ),
        "cross_shard_worldline_stitching_receipt": _literal_true(
            proto_particle_global.get("cross_shard_worldline_stitching_receipt")
        ),
        "all_shards_local_scale_compressed_pn_witness_receipt": _literal_true(
            pn_global.get("all_shards_local_scale_compressed_pn_witness_receipt")
        ),
        "global_capacity_readback_map_receipt": _literal_true(
            pn_global.get("global_capacity_readback_map_receipt")
        ),
        "finite_capacity_fixed_point_receipt": _literal_true(
            pn_global.get("finite_capacity_fixed_point_receipt")
        ),
        "global_pn_resonance_receipt": _literal_true(pn_global.get("global_pn_resonance_receipt")),
        "strict_single_global_neutral_bulk_receipt": _literal_true(
            neutral_global.get("strict_single_global_neutral_bulk_receipt")
        ),
        "physical_cmb_input_contract_receipt": _literal_true(
            physical_cmb_global.get("physical_cmb_input_contract_receipt")
        ),
        "physical_cmb_prediction_receipt": _literal_true(
            physical_cmb_global.get("physical_cmb_prediction_receipt")
        ),
    }
    artifact_smoke = bool(
        gates["all_expected_shards_completed"]
        and gates["global_carrier_contract_receipt"]
        and gates["seam_metadata_replay_receipt"]
        and _literal_true(halo_exchange.get("reducer_halo_exchange_replay_receipt"))
    )
    distributed_kernel_scaling_ready = bool(
        gates["all_expected_shards_completed"]
        and gates["global_carrier_contract_receipt"]
        and gates["distributed_realization_event_certificate_receipt"]
        and gates["distributed_online_evidence_recomputed_receipt"]
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
            "distributed_online_evidence_recomputed_receipt",
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
        from oph_fpe.cosmology.physical_cmb_sources import write_physical_cmb_source_readiness_report

        finite_reduction = _write_global_finite_cmb_source_reports(
            manifest=manifest,
            shard_roots=shard_roots,
            out_dir=out_dir,
        )
        write_physical_cmb_input_no_data_use_receipt(shard_roots + [out_dir], out_dir)
        no_data = _write_global_no_data_use_receipt(shard_roots + [out_dir], out_dir)
        cmb_sources = write_physical_cmb_source_readiness_report([out_dir], out_dir)
        input_report = write_physical_cmb_input_report([out_dir], out_dir)
        promotion = write_physical_cmb_promotion_audit_report([out_dir], out_dir)
        output = write_physical_cmb_output_comparison_report([out_dir, *shard_roots], out_dir)
        frontier = write_physical_cmb_frontier_report([out_dir, *shard_roots], out_dir)
        blockers = _unique_texts(
            list(finite_reduction.get("blockers") or [])
            + list(cmb_sources.get("blockers") or [])
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
            "finite_source_global_reduction_receipt": _literal_true(
                finite_reduction.get("FINITE_CMB_GLOBAL_REDUCTION_RECEIPT")
            ),
            "no_data_use_receipt": _literal_true(no_data.get("NO_DATA_USE_RECEIPT")),
            "physical_cmb_input_contract_receipt": _literal_true(
                input_report.get("PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT")
            ),
            "physical_cmb_prediction_eligible": _literal_true(
                input_report.get("physical_cmb_prediction_eligible")
            ),
            "physical_cmb_promotion_ready": _literal_true(
                promotion.get("physical_cmb_promotion_ready")
            ),
            "physical_cmb_source_readiness_written": True,
            "finite_covariant_parent_receipt": _literal_true(
                (cmb_sources.get("finite_covariant_parent") or {}).get("parent_receipt")
            ),
            "finite_collar_boltzmann_bundle_receipt": _literal_true(
                (cmb_sources.get("finite_collar_boltzmann_bundle") or {}).get("source_bundle_receipt")
            ),
            "finite_collar_boltzmann_physical_certificate": _literal_true(
                (cmb_sources.get("finite_collar_boltzmann_bundle") or {}).get("physical_certificate")
            ),
            "physical_cmb_output_comparison_receipt": _literal_true(
                output.get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT")
            ),
            "usable_physical_cmb_data_receipt": _literal_true(
                output.get("USABLE_PHYSICAL_CMB_DATA_RECEIPT")
            ),
            "physical_cmb_prediction_receipt": bool(
                _literal_true(output.get("PHYSICAL_CMB_PREDICTION_RECEIPT"))
                and _literal_true(frontier.get("physical_cmb_prediction_receipt"))
            ),
            "official_likelihood_ready": _literal_true(frontier.get("official_likelihood_ready")),
            "cdm_limit_regression_passed": _literal_true(frontier.get("cdm_limit_regression_passed")),
            "measurement_comparable_model_count": int(output.get("measurement_comparable_model_count") or 0),
            "oph_diagnostic_model_count": int(output.get("oph_diagnostic_model_count") or 0),
            "best_oph_diagnostic_model": output.get("best_oph_diagnostic_model") or {},
            "finite_reduction_report_path": str(out_dir / "finite_cmb_global_reduction_report.json"),
            "input_report_path": str(out_dir / "physical_cmb_input_report.json"),
            "promotion_audit_report_path": str(out_dir / "physical_cmb_promotion_audit_report.json"),
            "source_readiness_report_path": str(out_dir / "physical_cmb_source_readiness_report.json"),
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
            bool(validate_transition_clock_eligibility(report)["eligible"])
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
    observed_screen_capacity_comparison_available = bool(n_values)
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
        # Compatibility key is deliberately false: shard-local/observed N
        # values are comparison readouts, not public-record-capacity producers.
        "screen_capacity_global_readout": False,
        "screen_capacity_global_comparison_available": (
            observed_screen_capacity_comparison_available
        ),
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
                "source": "distributed_observed_horizon_comparison",
                "producer_eligible": False,
                "N_CRC": _consensus_or_none(n_values),
                "N_CRC_shard_mean": _mean_or_none(n_values),
                "N_CRC_shard_std": _std_or_none(n_values),
                "N_CRC_additive_sum_diagnostic": float(sum(n_values)) if n_values else None,
                "N_CRC_shard_values": n_values,
            },
            "readiness_gates": {
                "observed_branch_N_scr_readout_available": bool(_consensus_or_none(n_values) is not None),
                "observed_branch_is_comparison_only": True,
                "N_CRC_consensus_invariant": False,
                "N_CRC_consensus_invariant_receipt": False,
                "finite_correctable_public_record_evaluator_implemented": False,
                "additive_capacity_schema_declared": False,
            },
            "PHYSICAL_N_CLOSURE_RECEIPT": False,
            "complete_terminal_fiber_receipt": False,
            "whole_fiber_scalarization_receipt": False,
            "target_free_capacity_producer_receipt": False,
            "robust_closure_receipt": False,
            "unique_regulator_stable_slack_zero_receipt": False,
            "horizon_record_saturation_receipt": False,
            "shard_count": expected,
            "source_report_count": len(screen_capacity_reports),
            "claim_boundary": (
                "Distributed observed/shard N values are comparison-only. Consensus equality or an additive "
                "sum cannot produce physical public-record capacity. That requires an exact target-free "
                "complete terminal fiber, whole-fiber scalarization, robust closure, a unique regulator-stable "
                "slack zero, and an independent horizon-record saturation receipt."
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
            "owned_node_ranges": list(carrier_shard.get("owned_node_ranges") or []),
            "owned_node_count": int(carrier_shard.get("owned_node_count", patch_count_per_shard)),
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
        passed = [
            row["shard_id"]
            for row in shards
            if _literal_true((row.get("final_receipts") or {}).get(key, False))
        ]
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
    assumption_manifest: dict[str, Any],
    assumption_manifest_source: str,
) -> dict[str, Any]:
    sampled_payloads = _sample_shard_timeline_payloads(shard_rows, max_payloads=8)
    objective_views = []
    overlap_links = []
    screen_snapshots = []
    worldlines = []
    h3_objects = []
    screen_points = []
    screen_values = []
    for item in sampled_payloads:
        shard = item["shard"]
        payload = item["payload"]
        objective_views.extend(_globalized_observers(shard, payload, limit=max(1, 128 // max(1, len(sampled_payloads)))))
        overlap_links.extend(_globalized_overlap_links(shard, payload, limit=max(1, 2000 // max(1, len(sampled_payloads)))))
        screen_snapshots.extend(_globalized_screen_snapshots(shard, payload, limit=max(1, 64 // max(1, len(sampled_payloads)))))
        worldlines.extend(_globalized_worldlines(shard, payload, limit=max(1, 128 // max(1, len(sampled_payloads)))))
        h3_objects.extend(_globalized_h3_objects(shard, payload, limit=max(1, 256 // max(1, len(sampled_payloads)))))
        points, values = _globalized_screen_points(shard, payload, limit=max(1, 512 // max(1, len(sampled_payloads))))
        screen_points.extend(points)
        screen_values.extend(values)
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
        objective_views = list(observer_time_global.get("objectiveObserverViews") or [])
    if observer_time_global.get("overlapLinks"):
        overlap_links = list(observer_time_global.get("overlapLinks") or [])[:100000] + overlap_links
    if proto_particle_global.get("worldlines"):
        worldlines = list(proto_particle_global.get("worldlines") or [])
    objective_views = _objective_views_with_axes(objective_views, manifest)
    observer_rows = _observer_rows_from_objective_views(objective_views, manifest)
    time_frames = _observer_time_frames_from_views(objective_views, halo_exchange, seam_readout)
    observer_payload = {
        "description": (
            "Distributed observer-local modular-time export. Objective views are globalized from shard "
            "observer readouts and seam metadata; execution fields remain provenance, not observer time."
        ),
        "source": "distributed_global_observer_modular_time_export",
        "observers": observer_rows,
        "objectiveObserverViews": objective_views,
        "overlapLinks": overlap_links[:100000],
        "timeFrames": time_frames,
        "totalAvailableObserverCapacity": manifest.get("total_observer_capacity"),
        "receipts": {
            "observer_modular_time_receipt": bool(
                _literal_true(observer_time_global.get("global_observer_modular_time_export_receipt"))
                and _literal_true(observer_time_global.get("observer_clock_naturality_receipt"))
            ),
            "global_observer_modular_time_export_receipt": _literal_true(
                observer_time_global.get("global_observer_modular_time_export_receipt")
            ),
            "execution_clock_fields_separated_receipt": _literal_true(
                observer_time_global.get("execution_clock_fields_separated_receipt")
            ),
            "observer_clock_naturality_receipt": _literal_true(
                observer_time_global.get("observer_clock_naturality_receipt")
            ),
            "observer_facing_3p1d_h3_experience_receipt": False,
            "observer_facing_populated_h3_experience_receipt": False,
            "observer_h3_object_population_receipt": False,
            "support_visible_lorentz_3p1_kinematics_receipt": False,
            "conformal_h3_spatial_chart_receipt": False,
            "bulk_3d_established": False,
        },
        "dataAvailability": {
            "objectiveObserverViewDataAvailable": bool(objective_views),
            "observerTimeFrameDataAvailable": bool(time_frames),
            "h3ObjectDataAvailable": bool(h3_objects),
            "computedReceiptsUnaffectedByAvailability": True,
        },
        "overlapSummary": {
            "observerCount": len(observer_rows),
            "pairCount": len(overlap_links),
            "exportedPairCount": len(overlap_links[:100000]),
            "exportedObserverCount": len(observer_rows),
            "exportedObjectiveObserverViewCount": len(objective_views),
            "overlapLinkSource": (
                "global_observer_modular_time_export_plus_cross_shard_carrier_cut_metadata"
            ),
            "claimBoundary": (
                "Reducer-level observer overlaps are visualization metadata unless live cross-shard "
                "overlap repair receipts pass."
            ),
        },
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
        "blockers": list(observer_time_global.get("blockers") or []),
        "claimBoundary": observer_time_global.get(
            "claim_boundary",
            "Distributed observer modular-time visualization export; not chart-blind neutral bulk.",
        ),
    }
    if not screen_points:
        screen_points, screen_values = _atlas_screen_points(manifest)
    screen_payload = {
        "description": "Distributed global carrier atlas with sampled shard screen and cluster readouts.",
        "source": "distributed_visualization_payload",
        "fieldName": "distributed_screen_support",
        "points": screen_points[:4096],
        "values": screen_values[:4096],
        "repairTrace": _distributed_repair_trace(halo_exchange, seam_readout),
        "shards": (manifest.get("unified_universe_atlas") or {}).get("shards") or [],
        "crossShardOverlapLinks": seam_links,
        "clusters": {
            "snapshots": screen_snapshots,
            "snapshotSource": "sampled_per_shard_timeline_payloads_plus_global_shard_prefix",
        },
        "claimBoundary": (
            "Screen data are sampled/globalized visualization coordinates for the distributed atlas. "
            "They are not a hidden monolithic bulk state."
        ),
    }
    simulation_assumption_payload = _simulation_assumption_payload(
        assumption_manifest,
        source=assumption_manifest_source,
    )
    cmb_payload = _distributed_cmb_comparison_payload(
        physical_cmb_global,
        assumption_manifest=assumption_manifest,
        assumption_manifest_source=assumption_manifest_source,
    )
    pn_payload = _distributed_pn_payload(pn_global)
    bulk_payload = {
        "description": (
            "Distributed theorem-assisted H3 object/worldline readout plus strict-neutral global reduction status."
        ),
        "source": "distributed_visualization_payload",
        "objects": h3_objects,
        "dataAvailability": {
            "h3ObjectDataAvailable": bool(h3_objects),
            "h3ObjectCount": len(h3_objects),
            "protoWorldlineDataAvailable": bool(worldlines),
            "protoWorldlineCount": len(worldlines),
            "computedReceiptsUnaffectedByAvailability": True,
        },
        "protoParticleCandidates": {
            "description": "Globalized H3 defect/worldline candidates. Diagnostic unless particle receipts pass.",
            "worldlines": worldlines,
            "worldlineSource": (
                "global_proto_particle_worldline_stitch"
                if proto_particle_global.get("worldlines")
                else "sampled_per_shard_payloads_with_atlas_shard_context"
            ),
            "globalStitchReport": {
                "reportPath": "proto_particles_global/global_proto_particle_worldlines_report.json",
                "worldlineCount": proto_particle_global.get("worldlineCount"),
                "movingWorldlineCount": proto_particle_global.get("movingWorldlineCount"),
                "crossShardWorldlineStitchingReceipt": proto_particle_global.get(
                    "cross_shard_worldline_stitching_receipt"
                ),
            },
            "receipts": {
                "bulk_worldline_precursor_receipt": bool(
                    _literal_true(proto_particle_global.get("global_proto_particle_worldline_export_receipt"))
                    or _literal_true(proto_particle_global.get("bulk_worldline_precursor_receipt"))
                ),
                "particle_matter_receipt": _literal_true(
                    proto_particle_global.get("particle_matter_receipt")
                ),
                "diagnostic_edge_worldline_count": len(worldlines),
            },
            "claimBoundary": (
                "Globalized proto-worldlines remain diagnostics unless particle-matter and strict-neutral "
                "receipts pass."
            ),
        },
        "strictNeutralGlobalReduction": neutral_global,
        "receipts": _distributed_bulk_receipts(sampled_payloads, neutral_global, h3_objects),
        "strictNeutralBlockers": list(neutral_global.get("blockers") or []),
        "claimBoundary": (
            "Distributed H3 objects are observer-facing chart data. Strict neutral third-person bulk "
            "requires the strict_single_global_neutral_bulk_receipt."
        ),
    }
    subjective_cameras = _subjective_observer_cameras(
        observer_payload,
        bulk_payload=bulk_payload,
        camera_parameters=assumption_manifest.get("observer_camera_visualization_parameters"),
        camera_assumption_receipt=manifest_assumptions_pass(
            assumption_manifest,
            "screen_observer_to_h3_camera_embedding",
        ),
    )
    observer_payload["subjectiveObserverCameraRefs"] = [
        {
            "cameraId": camera.get("cameraId"),
            "observerId": camera.get("observerId"),
            "sourcePath": f"subjectiveObserverCameras[{index}]",
        }
        for index, camera in enumerate(subjective_cameras)
        if isinstance(camera, dict)
    ]
    assumed_ds4_payload = _assumed_ds4_visualization_payload(
        observer_payload=observer_payload,
        subjective_cameras=subjective_cameras,
        bulk_payload=bulk_payload,
        assumption_manifest=assumption_manifest,
        assumption_manifest_source=assumption_manifest_source,
    )
    small_payload = _distributed_small_universe_payload(manifest, seam_readout)
    comparable_payload = _distributed_comparable_observations_payload(cmb_payload, physical_cmb_global, neutral_global)
    geometry_payload = _geometry_and_symmetry_payload(
        small_payload,
        observer_payload,
        bulk_payload,
        pn_payload,
    )
    visualization_views = _visualization_views_payload(
        small_payload=small_payload,
        screen_payload=screen_payload,
        observer_payload=observer_payload,
        bulk_payload=bulk_payload,
        cmb_payload=cmb_payload,
        pn_silence_payload=pn_payload,
        diagnostic_run_dir=_distributed_diagnostic_dir(physical_cmb_global, sampled_payloads),
    )
    emergent_curved_spacetime_payload = _emergent_curved_spacetime_payload(
        visualization_views=visualization_views,
        bulk_payload=bulk_payload,
        screen_payload=screen_payload,
    )
    visualization_render_data = _visualization_render_data_payload(
        small_payload=small_payload,
        screen_payload=screen_payload,
        observer_payload=observer_payload,
        bulk_payload=bulk_payload,
        cmb_payload=cmb_payload,
        pn_silence_payload=pn_payload,
        subjective_cameras=subjective_cameras,
        visualization_views=visualization_views,
        curved_spacetime_payload=emergent_curved_spacetime_payload,
        simulation_assumption_payload=simulation_assumption_payload,
        assumed_ds4_payload=assumed_ds4_payload,
    )
    visualization_render_data["sceneGraph"]["assumedDs4Spacetime"] = {
        "sourcePath": "assumedDs4Spacetime",
        "geometrySource": "assumedDs4Spacetime.geometry",
        "scaleFactorSamplesSource": "assumedDs4Spacetime.scaleFactorSamples",
        "observerReferenceFramesSource": "assumedDs4Spacetime.observerReferenceFrames",
        "defectMatterRenderingSource": "assumedDs4Spacetime.defectMatterRendering",
        "scaleFactorSampleCount": len(assumed_ds4_payload.get("scaleFactorSamples", [])),
        "observerReferenceFrameCount": len(assumed_ds4_payload.get("observerReferenceFrames", [])),
        "receipts": assumed_ds4_payload.get("receipts", {}),
        "claimBoundary": assumed_ds4_payload.get("claimBoundary"),
    }
    hilbert_algebra_payload = _hilbert_space_observer_algebra_payload(observer_payload)
    return {
        "schemaVersion": "oph_universe_timeline_visualization_payload_v1",
        "schema": "oph_universe_timeline_visualization_payload_v1",
        "distributedSchema": "oph_distributed_universe_visualization_payload_v1",
        "title": "Distributed OPH Universe Timeline Visualization",
        "kernel": DISTRIBUTED_KERNEL_VERSION,
        "runId": manifest.get("run_id"),
        "claimBoundary": manifest.get("claim_boundary"),
        "ophDifferentiator": (
            "OPH technology instantiates observer-like self-reading systems: bounded patches with local "
            "state, ports or boundaries, readback, records, feedback/repair moves, and public receipts."
        ),
        "sourcePaths": {
            "distributedManifest": str(manifest.get("manifest_path") or ""),
            "physicalCmbGlobalDir": physical_cmb_global.get("report_dir"),
        },
        "coordinateSystems": {
            H3_COORDINATE_SYSTEM: _h3_coordinate_contract(),
            "ds4_open_h3_slicing_v1": assumed_ds4_payload.get("geometry", {}),
        },
        "simulationAssumptions": simulation_assumption_payload,
        "smallUniverse": small_payload,
        "screen": screen_payload,
        "subjectiveObserverCameras": subjective_cameras,
        "observerModularTime": observer_payload,
        "consensusBulk": bulk_payload,
        "pnSilenceToObservation": pn_payload,
        "cmbComparison": cmb_payload,
        "comparableObservations": comparable_payload,
        "geometriesAndSymmetries": geometry_payload,
        "visualizationViews": visualization_views,
        "visualizationRenderData": visualization_render_data,
        "effectiveStringTheory": _effective_string_theory_payload(
            visualization_views=visualization_views,
            small_payload=small_payload,
            bulk_payload=bulk_payload,
        ),
        "emergentCurvedSpacetime": emergent_curved_spacetime_payload,
        "assumedDs4Spacetime": assumed_ds4_payload,
        "observerCinema": _observer_cinema_payload(
            observer_payload=observer_payload,
            subjective_cameras=subjective_cameras,
        ),
        "hilbertSpaceObserverAlgebra": hilbert_algebra_payload,
        "observerAnatomy": _observer_anatomy_payload(
            observer_payload=observer_payload,
            subjective_cameras=subjective_cameras,
            hilbert_algebra_payload=hilbert_algebra_payload,
        ),
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
            "Do not label chart-blind strict neutral quotient bulk unless strict_single_global_neutral_bulk_receipt passes.",
        ],
    }


def _objective_views_with_axes(views: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    total = max(len(views), int(manifest.get("total_observer_capacity", 0) or 0), 1)
    for index, view in enumerate(views):
        if not isinstance(view, dict):
            continue
        item = dict(view)
        axis = item.get("axis")
        if not isinstance(axis, list) or len(axis) < 3:
            item["axis"] = _observer_axis(index, total)
        result.append(item)
    return result


def _observer_rows_from_objective_views(
    views: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    total = max(len(views), int(manifest.get("total_observer_capacity", 0) or 0), 1)
    for index, view in enumerate(views):
        if not isinstance(view, dict):
            continue
        axis = view.get("axis") if isinstance(view.get("axis"), list) and len(view.get("axis")) >= 3 else _observer_axis(index, total)
        frames = view.get("timeFrames") if isinstance(view.get("timeFrames"), list) else []
        rows.append(
            {
                "observerId": view.get("observerId", index),
                "axis": [float(axis[0]), float(axis[1]), float(axis[2])],
                "supportPatchCount": view.get("supportPatchCount"),
                "visibleSignatureEntropy": _last_frame_value(frames, "visibleSignatureEntropy"),
                "modularDepthMean": _last_frame_value(frames, "modularDepthMean"),
                "dominantRecordSignature": _last_frame_value(frames, "dominantRecordSignature"),
                "dominantObjectPacket": _last_frame_value(frames, "dominantObjectPacket"),
                "visibleReadoutHash": _last_frame_value(frames, "visibleReadoutHash"),
                "shardIndex": view.get("shardIndex"),
                "claimBoundary": view.get("claimBoundary"),
            }
        )
    return rows


def _observer_axis(index: int, total: int) -> list[float]:
    count = max(1, int(total))
    z = 1.0 - 2.0 * ((float(index) + 0.5) / float(count))
    radius = math.sqrt(max(0.0, 1.0 - z * z))
    theta = math.pi * (3.0 - math.sqrt(5.0)) * float(index)
    return [float(radius * math.cos(theta)), float(radius * math.sin(theta)), float(z)]


def _last_frame_value(frames: list[Any], key: str) -> Any:
    for frame in reversed(frames):
        if isinstance(frame, dict) and frame.get(key) is not None:
            return frame.get(key)
    return None


def _observer_time_frames_from_views(
    views: list[dict[str, Any]],
    halo_exchange: dict[str, Any],
    seam_readout: dict[str, Any],
) -> list[dict[str, Any]]:
    for view in views:
        frames = view.get("timeFrames") if isinstance(view, dict) and isinstance(view.get("timeFrames"), list) else []
        if frames:
            return [frame for frame in frames[:128] if isinstance(frame, dict)]
    repair_frames = _distributed_repair_trace(halo_exchange, seam_readout)
    if repair_frames:
        return [
            {
                "relativeTime": float(index / max(len(repair_frames) - 1, 1)),
                "cycle": row.get("cycle"),
                "committedFraction": row.get("meanRepairLoad"),
                "mismatchEdges": row.get("seamEdgeCount"),
            }
            for index, row in enumerate(repair_frames[:128])
        ]
    return [{"relativeTime": 0.0, "cycle": None}]


def _globalized_h3_objects(shard: dict[str, Any], payload: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    rows = ((payload.get("consensusBulk") or {}).get("objects") or [])[: int(limit)]
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        item = dict(row)
        local_id = item.get("objectId", item.get("object_id", index))
        item["objectId"] = _global_id(shard, local_id, "h3object")
        item["localObjectId"] = local_id
        item["shardIndex"] = shard.get("shard_index")
        item["shardId"] = shard.get("shard_id")
        item["atlasCenter"] = shard.get("atlas_center")
        result.append(item)
    return result


def _globalized_screen_points(
    shard: dict[str, Any],
    payload: dict[str, Any],
    *,
    limit: int,
) -> tuple[list[list[float]], list[float]]:
    screen = payload.get("screen") if isinstance(payload.get("screen"), dict) else {}
    points = screen.get("points") if isinstance(screen.get("points"), list) else []
    values = screen.get("values") if isinstance(screen.get("values"), list) else []
    out_points: list[list[float]] = []
    out_values: list[float] = []
    for index, point in enumerate(points[: int(limit)]):
        if not isinstance(point, list) or len(point) < 3:
            continue
        try:
            out_points.append([float(point[0]), float(point[1]), float(point[2])])
        except (TypeError, ValueError):
            continue
        value = values[index] if index < len(values) else shard.get("shard_index", 0)
        number = _float_or_none(value)
        out_values.append(float(number if number is not None else 0.0))
    return out_points, out_values


def _atlas_screen_points(manifest: dict[str, Any]) -> tuple[list[list[float]], list[float]]:
    points = []
    values = []
    shards = (manifest.get("unified_universe_atlas") or {}).get("shards") or []
    for index, shard in enumerate(shards):
        center = shard.get("atlas_center") if isinstance(shard, dict) else None
        if isinstance(center, list) and len(center) >= 3:
            try:
                points.append([float(center[0]), float(center[1]), float(center[2])])
                values.append(float(index / max(len(shards) - 1, 1)))
            except (TypeError, ValueError):
                continue
    if points:
        return points, values
    return [[0.0, 0.0, 1.0]], [0.0]


def _distributed_repair_trace(halo_exchange: dict[str, Any], seam_readout: dict[str, Any]) -> list[dict[str, Any]]:
    frames = halo_exchange.get("replay_frames") if isinstance(halo_exchange.get("replay_frames"), list) else []
    if frames:
        return list(frames[:128])
    rows = []
    for link in list(seam_readout.get("links") or [])[:256]:
        for frame in list(link.get("overlapRepairTrajectory") or [])[:64]:
            if isinstance(frame, dict):
                rows.append(frame)
                if len(rows) >= 128:
                    return rows
    return rows


def _distributed_cmb_comparison_payload(
    physical_cmb_global: dict[str, Any],
    *,
    assumption_manifest: dict[str, Any] | None = None,
    assumption_manifest_source: str | None = None,
) -> dict[str, Any]:
    report_dir = physical_cmb_global.get("report_dir")
    payload = (
        _cmb_payload(
            Path(report_dir),
            assumption_manifest=assumption_manifest,
            assumption_manifest_source=assumption_manifest_source,
        )
        if report_dir and Path(str(report_dir)).exists()
        else _cmb_payload(
            None,
            assumption_manifest=assumption_manifest,
            assumption_manifest_source=assumption_manifest_source,
        )
    )
    if not payload:
        payload = {
            "description": "No distributed CMB comparison rows were available.",
            "source": report_dir,
            "receipts": {},
            "residualRows": [],
            "observedRows": [],
            "modelRows": [],
            "screenDiagnosticSpectrumRows": [],
        }
    receipts = dict(payload.get("receipts") or {})
    receipts.update(
        {
            "PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT": _literal_true(
                physical_cmb_global.get("physical_cmb_output_comparison_receipt")
            ),
            "PHYSICAL_CMB_PREDICTION_RECEIPT": _literal_true(
                physical_cmb_global.get("physical_cmb_prediction_receipt")
            ),
            "physical_cmb_prediction": _literal_true(
                physical_cmb_global.get("physical_cmb_prediction_receipt")
            ),
            "finite_source_global_reduction_receipt": _literal_true(
                physical_cmb_global.get("finite_source_global_reduction_receipt")
            ),
            "physical_cmb_input_contract_receipt": _literal_true(
                physical_cmb_global.get("physical_cmb_input_contract_receipt")
            ),
        }
    )
    payload["receipts"] = receipts
    payload["globalReduction"] = physical_cmb_global
    payload["claimBoundary"] = physical_cmb_global.get(
        "claim_boundary",
        payload.get("claimBoundary", "Distributed CMB diagnostic; not a physical prediction unless receipts pass."),
    )
    return payload


def _distributed_pn_payload(pn_global: dict[str, Any]) -> dict[str, Any]:
    return {
        "description": "Distributed P/N silence-to-observation global reduction.",
        "source": "pn_resonance_global/global_pn_resonance_report.json",
        "globalReduction": pn_global,
        "closureCoordinates": {},
        "finiteRegulatorDepth": {},
        "silenceInitialState": {},
        "observationEmergence": {},
        "detuningControls": {},
        "receipts": {
            "scale_compressed_pn_silence_to_observation_receipt": _literal_true(
                pn_global.get("all_shards_local_scale_compressed_pn_witness_receipt")
            ),
            "literal_global_N_capacity_simulated_receipt": _literal_true(
                pn_global.get("finite_capacity_fixed_point_receipt")
            ),
            "global_pn_resonance_receipt": _literal_true(
                pn_global.get("global_pn_resonance_receipt")
            ),
        },
        "blockers": list(pn_global.get("blockers") or []),
        "claimBoundary": pn_global.get(
            "claim_boundary",
            "Distributed P/N lane; not a full finite-capacity proof unless global gates pass.",
        ),
    }


def _distributed_bulk_receipts(
    sampled_payloads: list[dict[str, Any]],
    neutral_global: dict[str, Any],
    h3_objects: list[dict[str, Any]],
) -> dict[str, Any]:
    source_receipts = []
    for item in sampled_payloads:
        payload = item.get("payload") if isinstance(item, dict) else {}
        receipts = ((payload.get("consensusBulk") or {}).get("receipts") or {}) if isinstance(payload, dict) else {}
        if isinstance(receipts, dict):
            source_receipts.append(receipts)
    def all_receipt(key: str) -> bool:
        return bool(source_receipts) and all(row.get(key) is True for row in source_receipts)

    strict = neutral_global.get("strict_single_global_neutral_bulk_receipt") is True
    return {
        "observer_like_self_reading_system_receipt": all_receipt(
            "observer_like_self_reading_system_receipt"
        ),
        "observer_modular_time_receipt": False,
        "observer_facing_3p1d_h3_experience_receipt": False,
        "observer_facing_populated_h3_experience_receipt": False,
        "observer_h3_object_population_receipt": False,
        "theorem_assisted_consensus_3d_bulk_readout_receipt": False,
        "observer_facing_consensus_3d_bulk_readout_receipt": False,
        "all_sampled_shards_observer_modular_time_receipt": all_receipt(
            "observer_modular_time_receipt"
        ),
        "all_sampled_shards_observer_h3_object_population_receipt": all_receipt(
            "observer_h3_object_population_receipt"
        ),
        "all_sampled_shards_theorem_assisted_consensus_readout_receipt": all_receipt(
            "theorem_assisted_consensus_3d_bulk_readout_receipt"
        ),
        "global_semantic_receipts_require_naturality_and_cross_shard_theorem_contract": True,
        "chart_blind_strict_neutral_quotient_bulk_receipt": strict,
        "strict_neutral_third_person_bulk_receipt": strict,
        "strict_single_global_neutral_bulk_receipt": strict,
        "physical_cmb_output_comparison_receipt": False,
        "physical_cmb_prediction_receipt": False,
    }


def _distributed_small_universe_payload(manifest: dict[str, Any], seam_readout: dict[str, Any]) -> dict[str, Any]:
    seam_links = seam_readout.get("links") or []
    edges = [
        {
            "source": link.get("source_shard_index"),
            "target": link.get("target_shard_index"),
            "kind": "carrier_cut",
        }
        for link in seam_links[:512]
        if isinstance(link, dict)
    ]
    return {
        "description": "Distributed carrier-cut repair graph summary for visualization.",
        "source": "distributed_global_carrier_cut_metadata_replay",
        "nodeCount": int(manifest.get("shard_count", len((manifest.get("shards") or []))) or 0),
        "nodes": [],
        "edges": edges,
        "repairFrames": _seam_repair_frames(seam_links),
        "cycles": {"exactConsensus": [], "frustratedControl": []},
        "receipts": {
            "FINITE_CONSENSUS_THEOREM_RECEIPT": False,
            "seam_metadata_replay_receipt": _literal_true(
                seam_readout.get("seam_metadata_replay_receipt")
            ),
            "cross_shard_overlap_repair_receipt": _literal_true(
                seam_readout.get("cross_shard_overlap_repair_receipt")
            ),
        },
        "claimBoundary": (
            "Distributed seam metadata replay for visualization. This is not an exact finite consensus "
            "mini-universe proof."
        ),
    }


def _seam_repair_frames(seam_links: list[Any]) -> list[dict[str, Any]]:
    frames = []
    for index, link in enumerate(seam_links[:64]):
        if not isinstance(link, dict):
            continue
        frames.append(
            {
                "step": index,
                "stateId": link.get("link_id"),
                "phi": link.get("final_committed_fraction_gap"),
                "action": "distributed_seam_metadata_replay",
                "node": link.get("source_shard_index"),
                "parent": link.get("target_shard_index"),
                "strictDescent": False,
            }
        )
    return frames


def _distributed_comparable_observations_payload(
    cmb_payload: dict[str, Any],
    physical_cmb_global: dict[str, Any],
    neutral_global: dict[str, Any],
) -> dict[str, Any]:
    datasets = []
    residual_rows = cmb_payload.get("residualRows") if isinstance(cmb_payload.get("residualRows"), list) else []
    if residual_rows:
        datasets.append(
            {
                "id": "distributed_cmb_tt_residual_rows",
                "datasetId": "distributed_cmb_tt_residual_rows",
                "kind": "cmb_tt_comparison",
                "rowCount": len(residual_rows),
                "rows": residual_rows[:240],
                "receipts": cmb_payload.get("receipts") or {},
            }
        )
    return {
        "description": "Distributed measurement-comparable diagnostics, fail-closed by source receipts.",
        "source": physical_cmb_global.get("report_dir"),
        "measurementLanes": [
            {
                "id": "distributed_physical_cmb_global",
                "lane": "distributed_physical_cmb_global",
                "runCount": int(physical_cmb_global.get("completed_shard_count", 0) or 0),
                "metrics": {
                    "measurement_comparable_model_count": physical_cmb_global.get("measurement_comparable_model_count"),
                    "oph_diagnostic_model_count": physical_cmb_global.get("oph_diagnostic_model_count"),
                },
            },
            {
                "id": "distributed_strict_neutral_global",
                "lane": "distributed_strict_neutral_global",
                "runCount": int(neutral_global.get("completed_shard_count", 0) or 0),
                "metrics": {
                    "strict_single_global_neutral_bulk_receipt": neutral_global.get(
                        "strict_single_global_neutral_bulk_receipt"
                    )
                },
            },
        ],
        "datasets": datasets,
        "receipts": {
            "physical_cmb_prediction": _literal_true(
                physical_cmb_global.get("physical_cmb_prediction_receipt")
            ),
            "bulk_3d_established_any": _literal_true(
                neutral_global.get("strict_single_global_neutral_bulk_receipt")
            ),
            "strict_neutral_3d_bulk_any": _literal_true(
                neutral_global.get("strict_single_global_neutral_bulk_receipt")
            ),
        },
        "claimBoundary": (
            "Distributed comparable diagnostics only; prediction and strict-neutral receipts remain "
            "controlled by source reductions."
        ),
    }


def _distributed_diagnostic_dir(
    physical_cmb_global: dict[str, Any],
    sampled_payloads: list[dict[str, Any]],
) -> Path | None:
    report_dir = physical_cmb_global.get("report_dir")
    if report_dir and Path(str(report_dir)).exists():
        return Path(str(report_dir))
    for item in sampled_payloads:
        payload = item.get("payload") if isinstance(item, dict) else {}
        source_paths = payload.get("sourcePaths") if isinstance(payload, dict) else {}
        for key in ("consensusPackDir", "observerRunDir"):
            value = source_paths.get(key) if isinstance(source_paths, dict) else None
            if value and Path(str(value)).exists():
                return Path(str(value))
    return None


def _hilbert_observer_algebra_summary(observer_payload: dict[str, Any]) -> dict[str, Any]:
    views = observer_payload.get("objectiveObserverViews") if isinstance(observer_payload.get("objectiveObserverViews"), list) else []
    links = observer_payload.get("overlapLinks") if isinstance(observer_payload.get("overlapLinks"), list) else []
    visible_object_packets = 0
    visible_record_packets = 0
    time_frame_count = 0
    for view in views:
        frames = view.get("timeFrames") if isinstance(view, dict) and isinstance(view.get("timeFrames"), list) else []
        time_frame_count += len(frames)
        for frame in frames:
            if isinstance(frame, dict):
                visible_object_packets += len(frame.get("visibleObjectPackets") or [])
                visible_record_packets += len(frame.get("visibleRecordPackets") or [])
    support_counts = [
        int(row.get("supportPatchCount"))
        for row in observer_payload.get("observers", [])
        if isinstance(row, dict) and _int_or_none(row.get("supportPatchCount")) is not None
    ]
    return {
        "schema": "oph_hilbert_observer_algebra_summary_v1",
        "observerCount": len(observer_payload.get("observers") or []),
        "objectiveObserverViewCount": len(views),
        "overlapLinkCount": len(links),
        "timeFrameCount": time_frame_count,
        "visibleObjectPacketCount": visible_object_packets,
        "visibleRecordPacketCount": visible_record_packets,
        "meanSupportPatchCount": _mean_or_none([float(value) for value in support_counts]),
        "finiteSupportAlgebraPopulated": bool(views and (visible_object_packets or visible_record_packets)),
        "claimBoundary": (
            "Visualization summary of finite observer-accessible records and overlap links. It is not a "
            "derivation of continuum Hilbert space or observer algebra without the dedicated receipts."
        ),
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
        normalized_events = []
        for event in item.get("events", []) if isinstance(item.get("events"), list) else []:
            if not isinstance(event, dict):
                continue
            point = _canonical_visual_h3_spatial_point(
                event.get("h3SpatialPoint", event.get("h3_spatial_point"))
            )
            if point is None:
                continue
            normalized_events.append({**event, "h3SpatialPoint": point})
        item["events"] = normalized_events
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


def _observer_view_reduction_limits(manifest: dict[str, Any]) -> dict[str, int]:
    cfg = manifest.get("observer_view_reduction") if isinstance(manifest.get("observer_view_reduction"), dict) else {}
    def positive_int(key: str, default: int) -> int:
        try:
            value = int(cfg.get(key, default))
        except (TypeError, ValueError):
            value = int(default)
        return max(1, value)

    return {
        "max_total": positive_int("max_total", 4096),
        "per_shard": positive_int("per_shard", 512),
    }


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
        normalized_point = _canonical_visual_h3_spatial_point(point)
        if normalized_point is None:
            continue
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
        length += _h3_distance(left[:3], right[:3])
    return float(length), float(length / max(len(points) - 1, 1))


def _canonical_visual_h3_spatial_point(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) not in (3, 4):
        return None
    try:
        parsed = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in parsed):
        return None
    if len(parsed) == 3:
        return parsed
    time_component = parsed[0]
    spatial = parsed[1:]
    shell_residual = abs(-time_component * time_component + sum(item * item for item in spatial) + 1.0)
    if time_component <= 0.0 or shell_residual > 1.0e-7:
        return None
    return spatial


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
