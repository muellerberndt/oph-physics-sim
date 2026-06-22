from __future__ import annotations

import csv
import html
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.core.graph import fibonacci_sphere_points


def write_universe_timeline_bundle(
    *,
    small_universe_dir: Path,
    observer_run_dir: Path,
    out_dir: Path,
    consensus_pack_dir: Path | None = None,
    consensus_readout_dir: Path | None = None,
    max_screen_points: int = 3500,
    max_observers: int = 96,
    max_h3_objects: int = 512,
) -> dict[str, Any]:
    """Write a compact OPH universe visualization bundle.

    The bundle is deliberately a visualization/readout artifact. It may combine
    a small exact repair universe, a larger observer-flow run, and a pack-level
    consensus/CMB comparison receipt without upgrading any gate.
    """

    small_path = Path(small_universe_dir)
    observer_path = Path(observer_run_dir)
    output_path = Path(out_dir)
    consensus_path = Path(consensus_pack_dir) if consensus_pack_dir is not None else None
    readout_path = Path(consensus_readout_dir) if consensus_readout_dir is not None else None
    if not small_path.exists():
        raise FileNotFoundError(small_path)
    if not observer_path.exists():
        raise FileNotFoundError(observer_path)
    if consensus_path is not None and not consensus_path.exists():
        raise FileNotFoundError(consensus_path)
    if readout_path is not None and not readout_path.exists():
        raise FileNotFoundError(readout_path)

    output_path.mkdir(parents=True, exist_ok=True)
    payload = build_universe_timeline_payload(
        small_universe_dir=small_path,
        observer_run_dir=observer_path,
        consensus_pack_dir=consensus_path,
        consensus_readout_dir=readout_path,
        max_screen_points=max_screen_points,
        max_observers=max_observers,
        max_h3_objects=max_h3_objects,
    )
    payload_path = output_path / "visualization_payload.json"
    viewer_path = output_path / "oph_universe_timeline_viewer.html"
    instructions_path = output_path / "VISUALIZATION_INSTRUCTIONS.md"
    web_agent_path = output_path / "WEB_CODING_AGENT_VISUALIZATION_BRIEF.md"
    payload_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    viewer_path.write_text(_render_html(payload), encoding="utf-8")
    instructions_path.write_text(_visualization_instructions(viewer_path, payload_path, payload), encoding="utf-8")
    web_agent_path.write_text(_web_agent_brief(payload_path, payload), encoding="utf-8")
    summary = {
        "mode": "oph_universe_timeline_visualization_bundle",
        "bundle_dir": str(output_path),
        "viewer_path": str(viewer_path),
        "payload_path": str(payload_path),
        "instructions_path": str(instructions_path),
        "web_coding_agent_brief_path": str(web_agent_path),
        "small_universe_dir": str(small_path),
        "observer_run_dir": str(observer_path),
        "consensus_pack_dir": str(consensus_path) if consensus_path is not None else None,
        "consensus_readout_dir": str(readout_path) if readout_path is not None else None,
        "small_repair_frame_count": len(payload["smallUniverse"]["repairFrames"]),
        "screen_point_count": len(payload["screen"]["points"]),
        "observer_count": len(payload["observerModularTime"]["observers"]),
        "observer_relative_time_count": len(payload["observerModularTime"]["timeFrames"]),
        "objective_observer_view_count": len(payload["observerModularTime"].get("objectiveObserverViews", [])),
        "h3_object_count": len(payload["consensusBulk"]["objects"]),
        "proto_particle_candidate_worldline_count": len(
            payload["consensusBulk"].get("protoParticleCandidates", {}).get("worldlines", [])
        ),
        "scale_compressed_pn_silence_to_observation_receipt": bool(
            payload["pnSilenceToObservation"]["receipts"].get(
                "scale_compressed_pn_silence_to_observation_receipt", False
            )
        ),
        "literal_global_N_capacity_simulated_receipt": bool(
            payload["pnSilenceToObservation"]["receipts"].get(
                "literal_global_N_capacity_simulated_receipt", False
            )
        ),
        "finite_consensus_theorem_receipt": bool(
            payload["smallUniverse"]["receipts"].get("FINITE_CONSENSUS_THEOREM_RECEIPT", False)
        ),
        "observer_modular_time_receipt": bool(
            payload["observerModularTime"]["receipts"].get("observer_modular_time_receipt", False)
        ),
        "theorem_assisted_consensus_3d_bulk_readout_receipt": bool(
            payload["consensusBulk"]["receipts"].get(
                "theorem_assisted_consensus_3d_bulk_readout_receipt", False
            )
        ),
        "chart_blind_strict_neutral_quotient_bulk_receipt": bool(
            payload["consensusBulk"]["receipts"].get(
                "chart_blind_strict_neutral_quotient_bulk_receipt",
                payload["consensusBulk"]["receipts"].get("strict_neutral_third_person_bulk_receipt", False),
            )
        ),
        "physical_cmb_output_comparison_receipt": bool(
            payload["cmbComparison"]["receipts"].get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT", False)
        ),
        "physical_cmb_prediction_receipt": bool(
            payload["cmbComparison"]["receipts"].get("PHYSICAL_CMB_PREDICTION_RECEIPT", False)
        ),
        "claim_boundary": payload["claimBoundary"],
    }
    (output_path / "universe_timeline_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def build_universe_timeline_payload(
    *,
    small_universe_dir: Path,
    observer_run_dir: Path,
    consensus_pack_dir: Path | None,
    consensus_readout_dir: Path | None,
    max_screen_points: int,
    max_observers: int,
    max_h3_objects: int,
) -> dict[str, Any]:
    small_payload = _small_universe_payload(Path(small_universe_dir))
    observer_payload = _observer_modular_time_payload(
        Path(observer_run_dir),
        readout_dir=(
            Path(consensus_readout_dir)
            if consensus_readout_dir is not None
            else Path(consensus_pack_dir)
            if consensus_pack_dir is not None
            else None
        ),
        max_observers=max_observers,
    )
    screen_payload = _screen_payload(Path(observer_run_dir), max_points=max_screen_points)
    bulk_payload = _consensus_bulk_payload(
        Path(consensus_pack_dir) if consensus_pack_dir is not None else None,
        Path(consensus_readout_dir) if consensus_readout_dir is not None else None,
        max_objects=max_h3_objects,
    )
    cmb_payload = _cmb_payload(Path(consensus_pack_dir) if consensus_pack_dir is not None else None)
    subjective_cameras = _subjective_observer_cameras(observer_payload)
    observer_payload["subjectiveObserverCameras"] = subjective_cameras
    comparable_payload = _comparable_observations_payload(
        Path(consensus_pack_dir) if consensus_pack_dir is not None else None,
        cmb_payload,
    )
    pn_silence_payload = _pn_silence_to_observation_payload(
        Path(observer_run_dir),
        Path(consensus_pack_dir) if consensus_pack_dir is not None else None,
    )
    geometry_payload = _geometry_and_symmetry_payload(
        small_payload,
        observer_payload,
        bulk_payload,
        pn_silence_payload,
    )
    visualization_views = _visualization_views_payload(
        small_payload=small_payload,
        screen_payload=screen_payload,
        observer_payload=observer_payload,
        bulk_payload=bulk_payload,
        cmb_payload=cmb_payload,
        pn_silence_payload=pn_silence_payload,
    )
    return {
        "schemaVersion": "oph_universe_timeline_visualization_payload_v1",
        "schema": "oph_universe_timeline_visualization_payload_v1",
        "title": "OPH Universe Timeline Visualization",
        "sourcePaths": {
            "smallUniverseDir": str(Path(small_universe_dir)),
            "observerRunDir": str(Path(observer_run_dir)),
            "consensusPackDir": str(consensus_pack_dir) if consensus_pack_dir is not None else None,
            "consensusReadoutDir": str(consensus_readout_dir) if consensus_readout_dir is not None else None,
        },
        "claimBoundary": (
            "Visualization/readout bundle for OPH observer-like self-reading systems. It shows a finite "
            "observer screen, overlap repair, observer-local modular-time readouts, theorem-assisted H3 "
            "consensus object charts, and measurement-comparable CMB diagnostics when present. It does "
            "not promote chart-blind strict neutral quotient bulk or physical CMB prediction unless those receipts "
            "are true in the payload."
        ),
        "ophDifferentiator": (
            "OPH technology instantiates observer-like self-reading systems: bounded patches with local "
            "state, ports or boundaries, readback, records, feedback/repair moves, and public receipts."
        ),
        "smallUniverse": small_payload,
        "screen": screen_payload,
        "subjectiveObserverCameras": subjective_cameras,
        "observerModularTime": observer_payload,
        "consensusBulk": bulk_payload,
        "pnSilenceToObservation": pn_silence_payload,
        "cmbComparison": cmb_payload,
        "comparableObservations": comparable_payload,
        "geometriesAndSymmetries": geometry_payload,
        "visualizationViews": visualization_views,
    }


def _small_universe_payload(run_dir: Path) -> dict[str, Any]:
    exact = _read_json(run_dir / "exact_consensus_receipt.json")
    frustrated = _read_json(run_dir / "frustrated_control_receipt.json")
    evidence = _read_json(run_dir / "small_oph_universe_evidence.json")
    cycle_holonomy = _read_json(run_dir / "cycle_holonomy.json")
    edges = _edges_from_cycles(cycle_holonomy.get("exact_consensus", []))
    node_count = _infer_node_count(edges, exact.get("terminal_normal_form"))
    positions = _icosahedron_positions(node_count)
    repair_frames = _repair_frames(run_dir, node_count=node_count)
    exact_nonzero = int(exact.get("nonzero_holonomy_cycle_count", 0) or 0)
    frustrated_nonzero = int(frustrated.get("nonzero_holonomy_cycle_count", 0) or 0)
    return {
        "description": (
            "Exact finite overlap-repair mini-universe. Nodes are finite observer-interface patches; "
            "bits are local boundary/readback states; repair moves copy/adjust local records to reduce "
            "the mismatch Lyapunov phi."
        ),
        "source": str(run_dir),
        "nodeCount": node_count,
        "nodes": [
            {"id": index, "position": [float(value) for value in positions[index]]}
            for index in range(node_count)
        ],
        "edges": [{"source": int(left), "target": int(right)} for left, right in edges],
        "repairFrames": repair_frames,
        "cycles": {
            "exactConsensus": _cycle_rows(cycle_holonomy.get("exact_consensus", []), limit=32),
            "frustratedControl": _cycle_rows(cycle_holonomy.get("frustrated_control", []), limit=32),
        },
        "receipts": {
            "FINITE_CONSENSUS_THEOREM_RECEIPT": bool(
                exact.get("FINITE_CONSENSUS_THEOREM_RECEIPT", False)
            ),
            "exact_nonzero_holonomy_cycle_count": exact_nonzero,
            "frustrated_control_holonomy_obstruction_receipt": bool(
                frustrated.get("HOLONOMY_OBSTRUCTION_RECEIPT", False)
            ),
            "frustrated_nonzero_holonomy_cycle_count": frustrated_nonzero,
            "strict_descent_violation_count": int(exact.get("strict_descent_violation_count", 0) or 0),
            "schedule_confluence_violation_count": int(
                exact.get("schedule_confluence_violation_count", 0) or 0
            ),
            "unique_terminal_normal_form_count": int(
                exact.get("unique_terminal_normal_form_count", 0) or 0
            ),
            "terminal_phi": exact.get("terminal_phi"),
            "terminal_normal_form": exact.get("terminal_normal_form"),
            "bundle_receipt": bool(evidence.get("bundle_receipt", False)),
        },
        "claimBoundary": exact.get(
            "claim_boundary",
            "Exact small-universe repair receipt only; not a Lorentz, H3, particle, or cosmology claim.",
        ),
    }


def _repair_frames(run_dir: Path, *, node_count: int, max_steps: int = 32) -> list[dict[str, Any]]:
    start = _state_row(run_dir / "all_states.jsonl", "0" * node_count)
    if not start:
        start = _first_jsonl_row(run_dir / "all_states.jsonl")
    if not start:
        return []
    transitions = _transition_index(run_dir / "repair_transition_table.jsonl")
    frames = [
        {
            "step": 0,
            "stateId": start.get("state_id"),
            "state": start.get("state", []),
            "phi": start.get("phi"),
            "enabledRepairCount": start.get("enabled_repair_count"),
            "action": "initial_state",
            "node": None,
            "parent": None,
            "deltaPhi": None,
        }
    ]
    state_id = str(start.get("state_id", ""))
    seen = {state_id}
    for step in range(1, max_steps + 1):
        rows = transitions.get(state_id, [])
        if not rows:
            break
        candidates = [row for row in rows if row.get("strict_descent", False)]
        if not candidates:
            break
        row = min(candidates, key=lambda item: (float(item.get("phi_after", math.inf)), int(item.get("node", 9999))))
        next_id = str(row.get("next_state_id", ""))
        frames.append(
            {
                "step": step,
                "stateId": next_id,
                "state": row.get("next_state", []),
                "phi": row.get("phi_after"),
                "phiBefore": row.get("phi_before"),
                "action": "repair_move",
                "node": row.get("node"),
                "parent": row.get("parent"),
                "deltaPhi": row.get("delta_phi"),
                "strictDescent": bool(row.get("strict_descent", False)),
            }
        )
        if row.get("phi_after") == 0 or next_id in seen:
            break
        seen.add(next_id)
        state_id = next_id
    return frames


def _observer_modular_time_payload(
    run_dir: Path,
    *,
    readout_dir: Path | None = None,
    max_observers: int,
) -> dict[str, Any]:
    observer_report = _read_json(run_dir / "observer_modular_experience_report.json")
    status = _read_json(run_dir / "emergence_status_report.json")
    consensus = _read_json(run_dir / "observer_consensus_report.json")
    readout = _read_json((readout_dir or run_dir) / "observer_consensus_bulk_readout_report.json")
    views = _read_jsonl(run_dir / "observer_views.jsonl", limit=max_observers)
    trace = _read_trace(run_dir / "mismatch_trace.csv")
    time_grid = observer_report.get("observer_relative_time_grid")
    if not isinstance(time_grid, list) or not time_grid:
        time_grid = next((row.get("observer_relative_times") for row in views if row.get("observer_relative_times")), [])
    if not isinstance(time_grid, list) or not time_grid:
        time_grid = [0.0]
    time_grid = _expanded_time_grid(time_grid, trace, min_count=32)
    trace_frames = _relative_time_frames(time_grid, trace)
    observers = []
    for row in views:
        axis = row.get("axis")
        if not isinstance(axis, list) or len(axis) < 3:
            continue
        observers.append(
            {
                "observerId": row.get("observer_id"),
                "axis": [float(axis[0]), float(axis[1]), float(axis[2])],
                "supportPatchCount": row.get("support_patch_count"),
                "visibleSignatureEntropy": row.get("visible_signature_entropy"),
                "modularDepthMean": row.get("modular_depth_mean"),
                "dominantRecordSignature": row.get("dominant_record_signature"),
                "dominantObjectPacket": row.get("dominant_object_packet"),
                "visibleReadoutHash": row.get("visible_readout_hash"),
                "claimBoundary": row.get("claim_boundary"),
            }
        )
    overlap_links = _observer_overlap_links(views, consensus, trace_frames, max_links=20_000)
    objective_limit = int(max_observers) if int(max_observers) < 64 else min(int(max_observers), 128)
    return {
        "description": (
            "Observer-local modular time readout. The slider is the observer relative-time grid emitted "
            "by the run; each observer row reads only local support, record, modular-depth, and visible "
            "object data."
        ),
        "source": str(run_dir),
        "observers": observers,
        "objectiveObserverViews": _observer_perspective_payloads(views, trace_frames, limit=objective_limit),
        "overlapLinks": overlap_links,
        "overlapSummary": {
            "observerCount": consensus.get("observer_count"),
            "pairCount": consensus.get("pair_count"),
            "exportedPairCount": len(overlap_links),
            "overlapLinkSource": (
                "recomputed_from_exported_observer_supports"
                if overlap_links
                else "observer_consensus_sample_pairs"
            ),
            "overlapTrajectorySource": (
                "global_trace_scaled_by_final_overlap_readout; future runs should emit true per-link traces"
            ),
            "medianOverlapJaccard": consensus.get("median_overlap_jaccard"),
            "medianSignatureHistogramSimilarity": consensus.get("median_signature_histogram_similarity"),
            "p10SignatureHistogramSimilarity": consensus.get("p10_signature_histogram_similarity"),
            "claimBoundary": consensus.get("claim_boundary"),
        },
        "timeFrames": trace_frames,
        "receipts": {
            "observer_modular_time_receipt": bool(
                readout.get("observer_modular_time_receipt", False)
                or observer_report.get("observer_modular_time_receipt", False)
            ),
            "observer_facing_3p1d_h3_experience_receipt": bool(
                readout.get("observer_facing_3p1d_h3_experience_receipt", False)
                or observer_report.get("observer_facing_3p1d_h3_experience_receipt", False)
            ),
            "observer_facing_populated_h3_experience_receipt": bool(
                readout.get("observer_facing_populated_h3_experience_receipt", False)
                or observer_report.get("observer_facing_populated_h3_experience_receipt", False)
            ),
            "observer_h3_object_population_receipt": bool(
                readout.get("observer_h3_object_population_receipt", False)
                or observer_report.get("observer_h3_object_population_receipt", False)
            ),
            "support_visible_lorentz_3p1_kinematics_receipt": bool(
                status.get("support_visible_lorentz_3p1_kinematics_receipt", False)
            ),
            "conformal_h3_spatial_chart_receipt": bool(status.get("conformal_h3_spatial_chart_receipt", False)),
            "bulk_3d_established": bool(status.get("bulk_3d_established", False)),
        },
        "blockers": observer_report.get("blockers", []),
        "claimBoundary": observer_report.get(
            "claim_boundary",
            "Observer-local modular-time readout; not chart-blind strict neutral quotient bulk.",
        ),
    }


def _screen_payload(run_dir: Path, *, max_points: int) -> dict[str, Any]:
    points, field_name, values = _screen_points_and_field(run_dir, max_points=max_points)
    clusters = _screen_cluster_payload(run_dir)
    trace = _read_trace(run_dir / "mismatch_trace.csv")
    return {
        "description": (
            "Finite S2 observer screen / boundary regulator. Screen colors are readback fields from "
            "freezeout records. Defect markers are screen-local repair/holonomy residues, not particles "
            "unless a particle receipt separately passes."
        ),
        "source": str(run_dir),
        "fieldName": field_name,
        "points": [[float(value) for value in row] for row in points],
        "values": [float(value) for value in values],
        "clusters": clusters,
        "repairTrace": trace[:128],
        "claimBoundary": (
            "The screen is the finite observer boundary/readback surface. It is not an initialized "
            "third-person 3D bulk lattice."
        ),
    }


def _subjective_observer_cameras(observer_payload: dict[str, Any]) -> list[dict[str, Any]]:
    cameras: list[dict[str, Any]] = []
    perspectives = observer_payload.get("objectiveObserverViews", [])
    if not isinstance(perspectives, list):
        return cameras
    for index, row in enumerate(perspectives):
        if not isinstance(row, dict):
            continue
        axis = row.get("axis")
        if not isinstance(axis, list) or len(axis) < 3:
            continue
        forward = _unit_vec([float(axis[0]), float(axis[1]), float(axis[2])])
        if _vec_norm(forward) <= 0.0:
            continue
        eye = [float(1.18 * value) for value in forward]
        look_at = [0.0, 0.0, 0.0]
        ref_up = [0.0, 0.0, 1.0] if abs(forward[2]) < 0.92 else [0.0, 1.0, 0.0]
        right = _unit_vec(_cross(ref_up, forward))
        up = _unit_vec(_cross(forward, right))
        frames = []
        for frame in list(row.get("timeFrames", []))[:96] if isinstance(row.get("timeFrames"), list) else []:
            if not isinstance(frame, dict):
                continue
            frames.append(
                {
                    "relativeTime": frame.get("relativeTime"),
                    "cycle": frame.get("cycle"),
                    "visibleReadoutHash": frame.get("visibleReadoutHash"),
                    "dominantRecordSignature": frame.get("dominantRecordSignature"),
                    "dominantObjectPacket": frame.get("dominantObjectPacket"),
                    "visibleObjectPackets": frame.get("visibleObjectPackets", []),
                    "visibleRecordPackets": frame.get("visibleRecordPackets", []),
                    "polarFieldReadout": frame.get("polarFieldReadout", []),
                    "localTransitionStep": frame.get("localTransitionStep"),
                    "framePacketSource": frame.get("framePacketSource"),
                }
            )
        cameras.append(
            {
                "cameraId": f"subjective_observer_{row.get('observerId', index)}",
                "observerId": row.get("observerId"),
                "kind": "observer_local_subjective_camera",
                "eye": eye,
                "lookAt": look_at,
                "up": up,
                "right": right,
                "forward": [-float(value) for value in forward],
                "fovDegrees": 72.0,
                "screenProjection": "local_tangent_readout_camera_on_s2_boundary",
                "supportPatchCount": row.get("supportPatchCount"),
                "supportNodeSample": row.get("supportNodeSample", []),
                "supportEntropyCapacity": row.get("supportEntropyCapacity"),
                "visibleObjectPackets": row.get("visibleObjectPackets", []),
                "visibleRecordPackets": row.get("visibleRecordPackets", []),
                "timeFrames": frames,
                "claimBoundary": (
                    "Subjective observer camera derived from one observer-local visible readout. "
                    "It is a rendering camera for the observer's finite support, not a hidden global view."
                ),
            }
        )
    return cameras


def _comparable_observations_payload(consensus_pack_dir: Path | None, cmb_payload: dict[str, Any]) -> dict[str, Any]:
    if consensus_pack_dir is None or not consensus_pack_dir.exists():
        return {
            "description": "No comparable observation pack supplied.",
            "source": None,
            "measurementLanes": [],
            "datasets": [],
            "receipts": {
                "physical_cmb_prediction": False,
                "physical_matter_power_prediction": False,
            },
            "claimBoundary": "No measurement-comparable rows were supplied.",
        }
    try:
        from oph_fpe.cosmology.comparable_data import comparable_data_report

        report = comparable_data_report([consensus_pack_dir])
    except Exception as exc:  # pragma: no cover - retained for long-run viewer bundles
        return {
            "description": "Comparable-data report failed during visualization payload export.",
            "source": str(consensus_pack_dir),
            "error": repr(exc),
            "measurementLanes": [],
            "datasets": [],
            "receipts": {
                "physical_cmb_prediction": False,
                "physical_matter_power_prediction": False,
            },
            "claimBoundary": "Comparable-data export failed; no measurement claim is promoted.",
        }

    lanes = report.get("measurement_lanes", {}) if isinstance(report.get("measurement_lanes"), dict) else {}
    lane_rows = []
    preferred_lanes = (
        "planck_tt_shape_lite",
        "finite_repair_clock_cmb_camb_transfer",
        "oph_exact_cmb_camb_transfer",
        "finite_screen_cmb_anomaly_readouts",
        "static_galaxy_measurement_fit",
        "static_galaxy_proxy",
        "oph_cnb_neutrino_background",
        "h0_s8_branch_diagnostic",
        "oph_repair_clock_kappa",
        "oph_screen_power_effective_theory",
        "oph_maxent_green_screen_source",
        "observer_chart_object_population",
        "defect_worldline_particle_precursors",
        "neutral_distance_profile_audit",
        "prime_geometric_rank_sweep",
    )
    for lane_name in preferred_lanes:
        lane = lanes.get(lane_name)
        if not isinstance(lane, dict) or int(lane.get("run_count", 0) or 0) <= 0:
            continue
        lane_rows.append(
            {
                "id": lane_name,
                "lane": lane_name,
                "runCount": int(lane.get("run_count", 0) or 0),
                "metrics": _compact_metric_map(lane, limit=18),
            }
        )
    datasets = []
    cmb_rows = cmb_payload.get("residualRows", []) if isinstance(cmb_payload, dict) else []
    if isinstance(cmb_rows, list) and cmb_rows:
        datasets.append(
            {
                "id": "cmb_tt_residual_rows",
                "datasetId": "cmb_tt_residual_rows",
                "kind": "cmb_tt_comparison",
                "rowCount": len(cmb_rows),
                "rows": cmb_rows[:240],
                "receipts": (cmb_payload.get("receipts") or {}) if isinstance(cmb_payload, dict) else {},
                "claimBoundary": (cmb_payload.get("claimBoundary") if isinstance(cmb_payload, dict) else None),
            }
        )
    run_rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    compact_runs = [_compact_comparable_run_row(row) for row in run_rows[:64] if isinstance(row, dict)]
    if compact_runs:
        datasets.append(
            {
                "id": "comparable_run_receipt_rows",
                "datasetId": "comparable_run_receipt_rows",
                "kind": "multi_lane_receipt_summary",
                "rowCount": len(run_rows),
                "rows": compact_runs,
            }
        )
    return {
        "description": (
            "Measurement-comparable diagnostics exported for plotting against public data. These include "
            "screen-CMB/TT residual rows when present plus compact summaries for galaxy, CNB, H0/S8, "
            "repair-clock, object-population, and neutral-frontier lanes."
        ),
        "source": str(consensus_pack_dir),
        "measurementLanes": lane_rows,
        "datasets": datasets,
        "receipts": {
            "physical_cmb_prediction": bool(report.get("physical_cmb_prediction", False)),
            "physical_matter_power_prediction": bool(report.get("physical_matter_power_prediction", False)),
            "bulk_3d_established_any": bool(report.get("bulk_3d_established_any", False)),
            "theorem_assisted_h3_bulk_any": bool(report.get("theorem_assisted_h3_bulk_any", False)),
            "strict_neutral_3d_bulk_any": bool(report.get("strict_neutral_3d_bulk_any", False)),
            "chart_level_3p1_any": bool(report.get("chart_level_3p1_any", False)),
        },
        "claimBoundary": report.get(
            "claim_boundary",
            "Comparable observation diagnostics only; prediction gates stay controlled by source receipts.",
        ),
    }


def _screen_points_and_field(run_dir: Path, *, max_points: int) -> tuple[np.ndarray, str, np.ndarray]:
    npz_path = run_dir / "freezeout_fields.npz"
    if npz_path.exists():
        with np.load(npz_path) as data:
            points = np.asarray(data["points"], dtype=float)
            field_name = "record_signature" if "record_signature" in data.files else next(
                (name for name in data.files if name not in {"points", "cell_area_planck", "cell_entropy"}),
                "uniform",
            )
            values = np.asarray(data[field_name], dtype=float) if field_name in data.files else np.zeros(points.shape[0])
    else:
        count = 512
        points = fibonacci_sphere_points(count)
        field_name = "uniform"
        values = np.zeros(count, dtype=float)
    if points.shape[0] > max_points:
        rng = np.random.default_rng(17)
        indices = np.sort(rng.choice(points.shape[0], size=max_points, replace=False))
        points = points[indices]
        values = values[indices]
    return points, field_name, _normalize(values)


def _screen_cluster_payload(run_dir: Path) -> dict[str, Any]:
    holonomy = _read_json(run_dir / "array_holonomy_report.json")
    timeline = _read_json(run_dir / "defect_timeline_report.json")
    trace = _read_trace(run_dir / "mismatch_trace.csv")
    clusters = []
    for index, row in enumerate(list(holonomy.get("clusters", []))[:256]):
        centroid = row.get("centroid")
        if not isinstance(centroid, list) or len(centroid) < 3:
            continue
        clusters.append(
            {
                "clusterId": row.get("cluster_id", index),
                "worldlineId": row.get("worldline_id"),
                "point": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
                "class": row.get("class"),
                "supportNodeCount": row.get("support_node_count"),
            }
        )
    snapshots = []
    for row in list(timeline.get("snapshots", []))[:64] if timeline else []:
        points = []
        for cluster in list(row.get("clusters", []))[:256]:
            centroid = cluster.get("centroid")
            if isinstance(centroid, list) and len(centroid) >= 3:
                points.append(
                    {
                        "clusterId": cluster.get("cluster_id"),
                        "worldlineId": cluster.get("worldline_id"),
                        "point": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
                        "class": cluster.get("class"),
                        "supportNodeCount": cluster.get("support_node_count"),
                    }
                )
        snapshots.append(
            {
                "cycle": row.get("cycle"),
                "clusterCount": row.get("cluster_count", len(points)),
                "clusters": points,
                "snapshotSource": "defect_timeline_report",
            }
        )
    expanded_snapshots, snapshot_source = _expanded_cluster_snapshots(snapshots, trace)
    return {
        "clusters": clusters,
        "snapshots": expanded_snapshots,
        "rawSnapshotCount": len(snapshots),
        "snapshotSource": snapshot_source,
        "persistentWorldlineCount": int(timeline.get("persistent_worldline_count", 0)) if timeline else 0,
    }


def _expanded_cluster_snapshots(
    snapshots: list[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if not snapshots or not trace or len(snapshots) >= min(64, len(trace)):
        return snapshots, "defect_timeline_report"
    cycles = [int(row.get("cycle") or 0) for row in snapshots]
    by_cycle = {int(row.get("cycle") or 0): row for row in snapshots}
    expanded = []
    for trace_row in trace[:64]:
        cycle = int(trace_row.get("cycle") or 0)
        if cycle in by_cycle:
            expanded.append({**by_cycle[cycle], "interpolated": False})
            continue
        before_cycle = max((value for value in cycles if value <= cycle), default=cycles[0])
        after_cycle = min((value for value in cycles if value >= cycle), default=cycles[-1])
        before = by_cycle[before_cycle]
        after = by_cycle[after_cycle]
        span = max(1, after_cycle - before_cycle)
        alpha = max(0.0, min(1.0, (cycle - before_cycle) / span))
        clusters = _interpolated_clusters(before.get("clusters", []), after.get("clusters", []), alpha)
        expanded.append(
            {
                "cycle": cycle,
                "clusterCount": len(clusters),
                "clusters": clusters,
                "interpolated": True,
                "snapshotSource": "linear_interpolation_between_defect_timeline_samples",
            }
        )
    return expanded, "defect_timeline_report_interpolated_to_trace_cycles"


def _interpolated_clusters(left: Any, right: Any, alpha: float) -> list[dict[str, Any]]:
    left_rows = _cluster_index(left)
    right_rows = _cluster_index(right)
    keys = sorted(set(left_rows) | set(right_rows))
    clusters = []
    for key in keys[:256]:
        a = left_rows.get(key) or right_rows.get(key)
        b = right_rows.get(key) or left_rows.get(key)
        if not a or not b:
            continue
        pa = a.get("point")
        pb = b.get("point")
        if not (isinstance(pa, list) and isinstance(pb, list) and len(pa) >= 3 and len(pb) >= 3):
            continue
        point = [float((1.0 - alpha) * float(pa[index]) + alpha * float(pb[index])) for index in range(3)]
        clusters.append(
            {
                "clusterId": a.get("clusterId") or b.get("clusterId"),
                "worldlineId": a.get("worldlineId") or b.get("worldlineId"),
                "point": point,
                "class": a.get("class") or b.get("class"),
                "supportNodeCount": a.get("supportNodeCount") or b.get("supportNodeCount"),
            }
        )
    return clusters


def _cluster_index(rows: Any) -> dict[str, dict[str, Any]]:
    result = {}
    if not isinstance(rows, list):
        return result
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        key = str(row.get("worldlineId") or row.get("clusterId") or index)
        result[key] = row
    return result


def _consensus_bulk_payload(
    consensus_pack_dir: Path | None,
    readout_dir: Path | None,
    *,
    max_objects: int,
) -> dict[str, Any]:
    if consensus_pack_dir is None:
        return {
            "description": "No consensus pack supplied.",
            "objects": [],
            "receipts": {},
            "claimBoundary": "No H3 consensus bulk receipt supplied.",
        }
    readout = _read_json((readout_dir or consensus_pack_dir) / "observer_consensus_bulk_readout_report.json")
    object_summary = _read_json(consensus_pack_dir / "object_h3_bulk_viewer_summary.json")
    objects = _read_h3_objects(consensus_pack_dir, readout_dir, max_objects=max_objects)
    proto_particles = _read_proto_particle_candidates(consensus_pack_dir, max_worldlines=max(32, max_objects // 4))
    return {
        "description": (
            "Theorem-assisted observer-consensus H3 chart. Dots are consensus object/readback packets "
            "seen by overlapping observers, represented in a derived H3 spatial chart. Holonomy/defect "
            "worldlines are rendered separately as proto-particle candidates when the run emits them."
        ),
        "source": str(consensus_pack_dir),
        "objects": objects,
        "protoParticleCandidates": proto_particles,
        "objectViewerSummary": {
            "objectCount": object_summary.get("object_count"),
            "theoremAssistedH3Bulk": object_summary.get("theorem_assisted_h3_bulk"),
            "observerChartBulkPopulationReceipt": object_summary.get("observer_chart_bulk_population_receipt"),
            "observerOverlapLinkCount": object_summary.get("observer_overlap_link_count"),
            "strictNeutralBulk": object_summary.get("strict_neutral_bulk"),
        },
        "receipts": {
            "observer_like_self_reading_system_receipt": bool(
                readout.get("observer_like_self_reading_system_receipt", False)
            ),
            "observer_modular_time_receipt": bool(readout.get("observer_modular_time_receipt", False)),
            "observer_facing_3p1d_h3_experience_receipt": bool(
                readout.get("observer_facing_3p1d_h3_experience_receipt", False)
            ),
            "observer_facing_populated_h3_experience_receipt": bool(
                readout.get("observer_facing_populated_h3_experience_receipt", False)
            ),
            "observer_h3_object_population_receipt": bool(
                readout.get("observer_h3_object_population_receipt", False)
            ),
            "theorem_assisted_consensus_3d_bulk_readout_receipt": bool(
                readout.get("theorem_assisted_consensus_3d_bulk_readout_receipt", False)
            ),
            "observer_facing_consensus_3d_bulk_readout_receipt": bool(
                readout.get(
                    "observer_facing_consensus_3d_bulk_readout_receipt",
                    readout.get("theorem_assisted_consensus_3d_bulk_readout_receipt", False),
                )
            ),
            "chart_blind_strict_neutral_quotient_bulk_receipt": bool(
                readout.get("chart_blind_strict_neutral_quotient_bulk_receipt", False)
            ),
            "strict_neutral_third_person_bulk_receipt": bool(
                readout.get("strict_neutral_third_person_bulk_receipt", False)
            ),
            "physical_cmb_output_comparison_receipt": bool(
                readout.get("physical_cmb_output_comparison_receipt", False)
            ),
            "physical_cmb_prediction_receipt": bool(readout.get("physical_cmb_prediction_receipt", False)),
        },
        "strictNeutralBlockers": readout.get("strict_neutral_blockers", []),
        "claimBoundary": readout.get(
            "claim_boundary",
            "Theorem-assisted H3 chart visualization; not chart-blind strict neutral quotient bulk, matter particles, "
            "or physical CMB prediction.",
        ),
    }


def _read_h3_objects(
    consensus_pack_dir: Path,
    readout_dir: Path | None,
    *,
    max_objects: int,
) -> list[dict[str, Any]]:
    candidates = []
    candidates.extend([consensus_pack_dir / "h3_objects.csv", consensus_pack_dir / "consensus_h3_object_rows.csv"])
    for path in candidates:
        if not path.exists():
            continue
        rows = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                point = _point_from_h3_row(row)
                if point is None:
                    continue
                rows.append(
                    {
                        "objectId": row.get("object_id") or f"obj_{len(rows):04d}",
                        "recordFamilyId": row.get("record_family_id"),
                        "x": point[0],
                        "y": point[1],
                        "z": point[2],
                        "observerCount": _optional_float(row.get("observer_count")),
                        "supportSize": _optional_float(row.get("support_size")),
                        "h3Compactness": _optional_float(row.get("h3_compactness")),
                        "h3CompactnessNormalized": _optional_float(row.get("h3_compactness_normalized")),
                    }
                )
                if len(rows) >= max_objects:
                    break
        if rows:
            return rows
    for name in ("observer_chart_object_h3_lineage_report.json", "observer_chart_object_h3_report.json"):
        report = _read_json(consensus_pack_dir / name)
        rows = []
        for row in list(report.get("sample_objects", []))[:max_objects] if report else []:
            if not isinstance(row, dict):
                continue
            point = _point_from_h3_sample(row)
            if point is None:
                continue
            rows.append(
                {
                    "objectId": row.get("object_id") or f"obj_{len(rows):04d}",
                    "recordFamilyId": row.get("record_family_id"),
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                    "observerCount": _optional_float(row.get("observer_count")),
                    "supportSize": _optional_float(row.get("support_size")),
                    "h3Compactness": _optional_float(row.get("h3_compactness")),
                    "h3CompactnessNormalized": _optional_float(row.get("h3_compactness_normalized")),
                }
            )
        if rows:
            return rows
    if readout_dir is not None:
        readout_csv = readout_dir / "consensus_h3_object_rows.csv"
        if readout_csv.exists():
            rows = []
            with readout_csv.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    point = _point_from_h3_row(row)
                    if point is None:
                        continue
                    rows.append(
                        {
                            "objectId": row.get("object_id") or f"obj_{len(rows):04d}",
                            "recordFamilyId": row.get("record_family_id"),
                            "x": point[0],
                            "y": point[1],
                            "z": point[2],
                            "observerCount": _optional_float(row.get("observer_count")),
                            "supportSize": _optional_float(row.get("support_size")),
                            "h3Compactness": _optional_float(row.get("h3_compactness")),
                            "h3CompactnessNormalized": _optional_float(row.get("h3_compactness_normalized")),
                        }
                    )
                    if len(rows) >= max_objects:
                        break
            if rows:
                return rows
    return []


def _read_proto_particle_candidates(consensus_pack_dir: Path, *, max_worldlines: int) -> dict[str, Any]:
    worldline_report = _read_json(consensus_pack_dir / "defect_h3_worldlines_report.json")
    particle_report = _read_json(consensus_pack_dir / "particle_likeness_report.json")
    particle_rows = {
        str(row.get("worldline_id")): row
        for row in particle_report.get("worldlines", [])
        if isinstance(row, dict) and row.get("worldline_id") is not None
    }
    worldlines = []
    for row in list(worldline_report.get("worldlines", []))[:max_worldlines] if worldline_report else []:
        if not isinstance(row, dict):
            continue
        events = []
        for event in list(row.get("events", []))[:64] if isinstance(row.get("events"), list) else []:
            point = event.get("h3_spatial_point") if isinstance(event, dict) else None
            if not isinstance(point, list) or len(point) < 3:
                continue
            events.append(
                {
                    "cycle": event.get("cycle"),
                    "h3SpatialPoint": [float(point[0]), float(point[1]), float(point[2])],
                    "fitResidual": event.get("fit_residual"),
                    "supportNodeCount": event.get("support_node_count"),
                }
            )
        if not events:
            continue
        worldline_id = str(row.get("worldline_id") or f"worldline_{len(worldlines):06d}")
        particle = particle_rows.get(worldline_id, {})
        worldlines.append(
            {
                "worldlineId": worldline_id,
                "observationCount": row.get("observation_count"),
                "birthCycle": row.get("birth_cycle"),
                "deathCycle": row.get("death_cycle"),
                "h3PathLength": row.get("h3_path_length"),
                "meanH3Step": row.get("mean_h3_step"),
                "classMode": row.get("class_mode"),
                "events": events,
                "particleLike": bool(particle.get("particle_like", False)),
                "localizationPass": bool(particle.get("localization_pass", False)),
                "persistencePass": bool(particle.get("persistence_pass", False)),
                "sectorStabilityPass": bool(particle.get("sector_stability_pass", False)),
                "transportabilityPass": bool(particle.get("transportability_pass", False)),
                "bulkLocalizationPass": bool(particle.get("bulk_localization_pass", False)),
                "claimBoundary": (
                    "H3-fitted screen/collar holonomy defect worldline. This is a proto-particle "
                    "candidate only unless particle_matter_receipt is true."
                ),
            }
        )
    return {
        "description": (
            "Holonomy/defect worldlines fitted into the same derived H3 chart. These are the right "
            "visual layer for proto-particles in the current simulator, but they are not matter particles "
            "until localization, transport, fusion/scattering, repeated-seed, and neutral-bulk gates pass."
        ),
        "worldlines": worldlines,
        "receipts": {
            "bulk_worldline_precursor_receipt": bool(
                worldline_report.get("bulk_worldline_precursor_receipt", False)
            ),
            "particle_matter_receipt": bool(
                worldline_report.get("particle_matter_receipt", False)
                or particle_report.get("particle_matter_receipt", False)
            ),
            "particle_like_worldline_count": int(
                sum(1 for row in particle_rows.values() if bool(row.get("particle_like", False)))
            ),
            "persistent_h3_worldline_count": int(
                worldline_report.get("persistent_h3_worldline_count", 0) or 0
            ),
        },
        "claimBoundary": worldline_report.get(
            "claim_boundary",
            "No H3 holonomy worldline report was supplied.",
        ),
    }


def _cmb_payload(consensus_pack_dir: Path | None) -> dict[str, Any]:
    if consensus_pack_dir is None:
        return {"description": "No CMB comparison pack supplied.", "receipts": {}, "residualRows": []}
    report = _read_json(consensus_pack_dir / "physical_cmb_output_comparison_report.json")
    if not report:
        cmb_lite = _read_json(consensus_pack_dir / "cmb_lite_comparison_report.json")
        cl_report = _read_json(consensus_pack_dir / "cl_comparison_report.json")
        best_field = cmb_lite.get("best_shape_field")
        fields = cl_report.get("fields", {}) if cl_report else {}
        field_report = fields.get(best_field) if isinstance(best_field, str) else None
        if field_report is None and fields:
            best_field, field_report = next(iter(fields.items()))
        rows = []
        for row in list((field_report or {}).get("spectrum", []))[:160]:
            rows.append(
                {
                    "ell": _optional_float(row.get("ell")),
                    "observed": None,
                    "model": _optional_float(row.get("D_ell")),
                    "residualSigma": None,
                }
            )
        return {
            "description": (
                "Fresh-run CMB-lite/screen angular-spectrum diagnostic. This is useful for visualization "
                "of screen geometry and freezeout fields, but is not usable physical CMB output."
            ),
            "source": str(consensus_pack_dir),
            "receipts": {
                "PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT": False,
                "USABLE_PHYSICAL_CMB_DATA_RECEIPT": False,
                "PHYSICAL_CMB_PREDICTION_RECEIPT": False,
                "physical_cmb_prediction": False,
                "SCREEN_CMB_LITE_DIAGNOSTIC_RECEIPT": bool(cmb_lite or cl_report),
            },
            "bestOphDiagnosticModel": {"source": "cmb_lite_screen_proxy", "field": best_field},
            "bestOphResidualSummary": {},
            "residualRows": rows,
            "claimBoundary": cmb_lite.get(
                "claim_boundary",
                "Screen C_l diagnostic only; not a physical CMB prediction.",
            ),
        }
    residual_summary = report.get("best_oph_residual_summary", {}) if report else {}
    best_model = report.get("best_oph_diagnostic_model", {}) if report else {}
    rows = []
    for row in list(report.get("best_oph_residual_rows", []))[:160] if report else []:
        rows.append(
            {
                "ell": _optional_float(row.get("ell")),
                "observed": _optional_float(row.get("observed")),
                "model": _optional_float(row.get("model")),
                "residualSigma": _optional_float(row.get("residual_sigma")),
            }
        )
    return {
        "description": (
            "Physical-unit, measurement-comparable CMB TT diagnostic rows. These are usable comparison "
            "data, not a physical CMB prediction unless the hard input/promotion gates pass."
        ),
        "source": str(consensus_pack_dir),
        "receipts": {
            "PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT": bool(
                report.get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT", False)
            ),
            "USABLE_PHYSICAL_CMB_DATA_RECEIPT": bool(report.get("USABLE_PHYSICAL_CMB_DATA_RECEIPT", False)),
            "PHYSICAL_CMB_PREDICTION_RECEIPT": bool(report.get("PHYSICAL_CMB_PREDICTION_RECEIPT", False)),
            "physical_cmb_prediction": bool(report.get("physical_cmb_prediction", False)),
        },
        "bestOphDiagnosticModel": best_model,
        "bestOphResidualSummary": residual_summary,
        "residualRows": rows,
        "claimBoundary": report.get(
            "claim_boundary",
            "CMB output comparison diagnostic; not a physical CMB prediction without hard gates.",
        ),
    }


def _pn_silence_to_observation_payload(run_dir: Path, alternate_dir: Path | None) -> dict[str, Any]:
    report = _read_json(run_dir / "silence_to_observation_report.json")
    if not report and alternate_dir is not None and alternate_dir != run_dir:
        report = _read_json(alternate_dir / "silence_to_observation_report.json")
    closure = report.get("closure_coordinates") or {}
    depth = report.get("finite_regulator_depth") or {}
    silence = report.get("silence_initial_state") or {}
    emergence = report.get("observation_emergence") or {}
    controls = report.get("detuning_controls") or {}
    wrong_controls = controls.get("wrong_detuning_multipliers") or []
    return {
        "description": (
            "Scale-compressed OPH P/N silence-to-observation witness. It shows the finite simulator's "
            "core thesis lane: a record-silent observer screen is run on a P-detuned local/global "
            "closure branch, then overlap repair/readback emits observer records, modular time, and "
            "H3 object readouts."
        ),
        "source": report.get("run_dir") or str(run_dir),
        "closureCoordinates": {
            "P": closure.get("P"),
            "phiSilentEquilibrium": closure.get("phi_silent_equilibrium"),
            "PDetuningDelta": closure.get("P_detuning_delta"),
            "alphaFromP": closure.get("alpha_from_P"),
            "alphaInverseFromP": closure.get("alpha_inverse_from_P"),
            "loopDetuningPhase": closure.get("loop_detuning_phase"),
            "NStar": closure.get("N_star"),
            "NStarSource": closure.get("N_star_source"),
            "pnNumericReplay": bool(closure.get("PN_RESONANCE_NUMERIC_REPLAY", False)),
        },
        "finiteRegulatorDepth": {
            "patchCount": depth.get("patch_count"),
            "N_eff": depth.get("regulator_entropy_capacity_N_eff"),
            "localRepairContraction": depth.get("local_repair_contraction_abs_gprime"),
            "effectiveRepairRoundDepth": depth.get("effective_repair_round_depth"),
            "declaredCosmicRepairRoundDepth": depth.get("declared_cosmic_repair_round_depth"),
        },
        "silenceInitialState": {
            "cycle": silence.get("cycle"),
            "committedRecords": silence.get("committed_records"),
            "committedFraction": silence.get("committed_fraction"),
            "recordEntropy": silence.get("record_entropy"),
            "initialRecordSilenceReceipt": bool(silence.get("initial_record_silence_receipt", False)),
        },
        "observationEmergence": {
            "finalCycle": emergence.get("final_cycle"),
            "finalCommittedRecords": emergence.get("final_committed_records"),
            "finalCommittedFraction": emergence.get("final_committed_fraction"),
            "observerCount": emergence.get("observer_count"),
            "h3ObjectCount": emergence.get("h3_object_count"),
            "persistentH3WorldlineCount": emergence.get("persistent_h3_worldline_count"),
            "observationEmergenceReceipt": bool(emergence.get("observation_emergence_receipt", False)),
        },
        "detuningControls": {
            "noDetuningBlocksPNBridge": bool(
                (controls.get("no_detuning_phi_equilibrium") or {}).get("blocks_pn_bridge", False)
            ),
            "wrongDetuningMultipliers": [
                {
                    "multiplier": row.get("detuning_multiplier"),
                    "candidateP": row.get("candidate_P"),
                    "logResidualVsSelectedN": row.get("log_residual_vs_selected_N"),
                    "blocksSelectedBridge": bool(row.get("blocks_selected_bridge", False)),
                }
                for row in wrong_controls
                if isinstance(row, dict)
            ],
        },
        "receipts": {
            "scale_compressed_pn_silence_to_observation_receipt": bool(
                report.get("scale_compressed_pn_silence_to_observation_receipt", False)
            ),
            "literal_global_N_capacity_simulated_receipt": bool(
                report.get("literal_global_N_capacity_simulated_receipt", False)
            ),
            "dynamic_p_detuning_control_receipt": bool(
                report.get("dynamic_p_detuning_control_receipt", False)
            ),
        },
        "readinessGates": report.get("readiness_gates") or {},
        "claimBoundary": report.get(
            "claim_boundary",
            "No P/N silence-to-observation report was supplied.",
        ),
    }


def _geometry_and_symmetry_payload(
    small_payload: dict[str, Any],
    observer_payload: dict[str, Any],
    bulk_payload: dict[str, Any],
    pn_silence_payload: dict[str, Any],
) -> dict[str, Any]:
    small_receipts = small_payload.get("receipts", {})
    observer_receipts = observer_payload.get("receipts", {})
    bulk_receipts = bulk_payload.get("receipts", {})
    pn_receipts = pn_silence_payload.get("receipts", {})
    return {
        "pnSilenceToObservation": {
            "name": "P/N silence-to-observation lane",
            "meaning": (
                "finite record silence on an observer screen followed by P-detuned overlap repair, "
                "observer readback, modular time, and H3 object readouts; scale-compressed, not literal N_CRC"
            ),
            "scaleCompressedReceipt": pn_receipts.get(
                "scale_compressed_pn_silence_to_observation_receipt", False
            ),
            "literalGlobalNSimulated": pn_receipts.get(
                "literal_global_N_capacity_simulated_receipt", False
            ),
        },
        "screenGeometry": {
            "name": "finite S2 observer screen",
            "meaning": "support-visible boundary/readback surface for local observers",
        },
        "repairGeometry": {
            "name": "finite overlap-repair graph",
            "meaning": "local repair moves descend a mismatch Lyapunov phi over overlapping records",
            "finiteConsensusReceipt": small_receipts.get("FINITE_CONSENSUS_THEOREM_RECEIPT", False),
        },
        "cycleHolonomy": {
            "name": "Z2 cycle holonomy",
            "meaning": "cycle consistency/obstruction check on the finite interface graph",
            "exactNonzeroCycleCount": small_receipts.get("exact_nonzero_holonomy_cycle_count"),
            "frustratedControlNonzeroCycleCount": small_receipts.get("frustrated_nonzero_holonomy_cycle_count"),
        },
        "lorentzH3Chart": {
            "name": "SO+(3,1)/H3 chart route",
            "meaning": "paper-side observer/chart geometry; not neutral bulk by itself",
            "supportVisibleLorentz3p1Receipt": observer_receipts.get(
                "support_visible_lorentz_3p1_kinematics_receipt", False
            ),
            "conformalH3SpatialChartReceipt": observer_receipts.get("conformal_h3_spatial_chart_receipt", False),
        },
        "consensusBulk": {
            "name": "observer-consensus H3 object cloud",
            "meaning": "shared object/readback packets represented in a derived H3 chart",
            "theoremAssistedConsensus3dBulkReceipt": bulk_receipts.get(
                "theorem_assisted_consensus_3d_bulk_readout_receipt", False
            ),
            "chartBlindStrictNeutralQuotientBulkReceipt": bulk_receipts.get(
                "chart_blind_strict_neutral_quotient_bulk_receipt",
                bulk_receipts.get("strict_neutral_third_person_bulk_receipt", False),
            ),
        },
        "protoParticleCandidates": {
            "name": "H3 holonomy/defect worldlines",
            "meaning": (
                "persistent screen/collar holonomy residues fitted into the observer-derived H3 chart; "
                "these are proto-particle candidates until particle and neutral-bulk receipts pass"
            ),
            "bulkWorldlinePrecursorReceipt": bulk_payload.get("protoParticleCandidates", {})
            .get("receipts", {})
            .get("bulk_worldline_precursor_receipt", False),
            "particleMatterReceipt": bulk_payload.get("protoParticleCandidates", {})
            .get("receipts", {})
            .get("particle_matter_receipt", False),
        },
    }


def _visualization_views_payload(
    *,
    small_payload: dict[str, Any],
    screen_payload: dict[str, Any],
    observer_payload: dict[str, Any],
    bulk_payload: dict[str, Any],
    cmb_payload: dict[str, Any],
    pn_silence_payload: dict[str, Any],
) -> dict[str, Any]:
    small_receipts = small_payload.get("receipts", {})
    observer_receipts = observer_payload.get("receipts", {})
    bulk_receipts = bulk_payload.get("receipts", {})
    proto_receipts = bulk_payload.get("protoParticleCandidates", {}).get("receipts", {})
    cmb_receipts = cmb_payload.get("receipts", {})
    pn_receipts = pn_silence_payload.get("receipts", {})
    return {
        "fluctuatingQuantumVacuum": {
            "viewId": "fluctuatingQuantumVacuum",
            "label": "Fluctuating quantum vacuum / finite readback field",
            "description": (
                "Render the finite S2 screen as the observer-facing vacuum/readback regulator. "
                "Animate screen.clusters.snapshots and screen.repairTrace as repair/holonomy "
                "fluctuations. This is a diagnostic OPH readback field, not a literal QFT vacuum "
                "unless a separate quantum-field receipt is added."
            ),
            "dataSources": [
                "screen.points",
                "screen.values",
                "screen.fieldName",
                "screen.clusters.snapshots",
                "screen.repairTrace",
                "smallUniverse.repairFrames",
                "cmbComparison.residualRows",
            ],
            "primaryFields": ["screen.values", "screen.clusters.snapshots", "screen.repairTrace"],
            "renderLayers": [
                {"layer": "screen_s2_field", "source": "screen.points + screen.values"},
                {"layer": "repair_fluctuation_markers", "source": "screen.clusters.snapshots[*].clusters"},
                {"layer": "mismatch_commit_trace", "source": "screen.repairTrace"},
                {"layer": "cmb_diagnostic_overlay", "source": "cmbComparison.residualRows"},
            ],
            "receipts": {
                "finite_consensus_theorem_receipt": bool(
                    small_receipts.get("FINITE_CONSENSUS_THEOREM_RECEIPT", False)
                ),
                "screen_cmb_diagnostic_receipt": bool(
                    cmb_receipts.get("SCREEN_CMB_LITE_DIAGNOSTIC_RECEIPT", False)
                    or cmb_receipts.get("USABLE_PHYSICAL_CMB_DATA_RECEIPT", False)
                ),
                "physical_cmb_prediction_receipt": bool(
                    cmb_receipts.get("PHYSICAL_CMB_PREDICTION_RECEIPT", False)
                ),
            },
            "exportSufficiency": "sufficient_for_diagnostic_visualization_not_physical_qft_vacuum",
            "claimBoundary": (
                "Shows finite OPH screen/readback fluctuations and repair residues. Do not label as a "
                "literal fluctuating quantum vacuum without a dedicated QFT-vacuum receipt."
            ),
        },
        "observerCamera": {
            "viewId": "observerCamera",
            "label": "Observer camera / local modular-time readout",
            "description": (
                "Render one observer-local camera at a time from subjectiveObserverCameras. Each camera "
                "is derived from visible support, readback records, packet histograms, and modular-time "
                "frames; it is not a hidden global camera."
            ),
            "dataSources": [
                "subjectiveObserverCameras",
                "observerModularTime.objectiveObserverViews",
                "observerModularTime.overlapLinks",
                "observerModularTime.timeFrames",
            ],
            "primaryFields": [
                "subjectiveObserverCameras[*].eye",
                "subjectiveObserverCameras[*].forward",
                "subjectiveObserverCameras[*].timeFrames",
            ],
            "renderLayers": [
                {"layer": "observer_axis_camera", "source": "subjectiveObserverCameras[*]"},
                {"layer": "visible_record_packets", "source": "subjectiveObserverCameras[*].timeFrames"},
                {"layer": "overlap_support_graph", "source": "observerModularTime.overlapLinks"},
            ],
            "receipts": {
                "observer_modular_time_receipt": bool(
                    observer_receipts.get("observer_modular_time_receipt", False)
                ),
                "observer_facing_3p1d_h3_experience_receipt": bool(
                    observer_receipts.get("observer_facing_3p1d_h3_experience_receipt", False)
                ),
                "observer_facing_populated_h3_experience_receipt": bool(
                    observer_receipts.get("observer_facing_populated_h3_experience_receipt", False)
                ),
            },
            "exportSufficiency": "sufficient_for_observer_local_camera_visualization",
            "claimBoundary": (
                "Subjective observer cameras are visible-readout cameras. They do not expose hidden "
                "state or objective global time."
            ),
        },
        "effectiveStringTheory": {
            "viewId": "effectiveStringTheory",
            "label": "Effective string-theory edge/worldsheet view",
            "description": (
                "Render cyclic repair paths, collar-like screen defect worldlines, and H3 proto-particle "
                "tracks as the current effective string/worldsheet diagnostic layer. The export is enough "
                "for a schematic edge-string view, but not enough to claim a critical string CFT; that "
                "requires EDGE_CYLINDER current, Virasoro, Sugawara, supercurrent, and spin-structure receipts."
            ),
            "dataSources": [
                "smallUniverse.cycles",
                "smallUniverse.edges",
                "smallUniverse.repairFrames",
                "screen.clusters.snapshots",
                "consensusBulk.protoParticleCandidates.worldlines",
                "consensusBulk.objects",
            ],
            "primaryFields": [
                "smallUniverse.cycles.exactConsensus",
                "smallUniverse.repairFrames",
                "consensusBulk.protoParticleCandidates.worldlines[*].events",
            ],
            "renderLayers": [
                {"layer": "cyclic_edge_normal_forms", "source": "smallUniverse.cycles"},
                {"layer": "repair_history_worldsheet_ribbons", "source": "smallUniverse.repairFrames"},
                {"layer": "collar_defect_tracks", "source": "screen.clusters.snapshots"},
                {"layer": "h3_worldline_overlay", "source": "consensusBulk.protoParticleCandidates.worldlines"},
            ],
            "receipts": {
                "finite_consensus_theorem_receipt": bool(
                    small_receipts.get("FINITE_CONSENSUS_THEOREM_RECEIPT", False)
                ),
                "bulk_worldline_precursor_receipt": bool(
                    proto_receipts.get("bulk_worldline_precursor_receipt", False)
                ),
                "particle_matter_receipt": bool(proto_receipts.get("particle_matter_receipt", False)),
                "observer_facing_consensus_3d_bulk_readout_receipt": bool(
                    bulk_receipts.get(
                        "observer_facing_consensus_3d_bulk_readout_receipt",
                        bulk_receipts.get("theorem_assisted_consensus_3d_bulk_readout_receipt", False),
                    )
                ),
                "critical_edge_cft_receipt": False,
            },
            "exportSufficiency": "sufficient_for_schematic_edge_string_view_not_critical_worldsheet_claim",
            "claimBoundary": (
                "This is an effective edge-string diagnostic view. It must not be labeled a proven critical "
                "heterotic worldsheet until the finite-carrier critical-edge receipt suite passes."
            ),
        },
        "silenceToObservation": {
            "viewId": "silenceToObservation",
            "label": "Silence-to-observation bridge",
            "description": (
                "Render the scale-compressed sequence from record silence through P detuning, finite "
                "repair depth, and observer/H3 readout emergence."
            ),
            "dataSources": ["pnSilenceToObservation"],
            "primaryFields": [
                "pnSilenceToObservation.silenceInitialState",
                "pnSilenceToObservation.closureCoordinates",
                "pnSilenceToObservation.finiteRegulatorDepth",
                "pnSilenceToObservation.observationEmergence",
            ],
            "renderLayers": [
                {"layer": "silent_initial_state", "source": "pnSilenceToObservation.silenceInitialState"},
                {"layer": "p_detuning", "source": "pnSilenceToObservation.closureCoordinates"},
                {"layer": "finite_repair_depth", "source": "pnSilenceToObservation.finiteRegulatorDepth"},
                {"layer": "readout_emergence", "source": "pnSilenceToObservation.observationEmergence"},
            ],
            "receipts": {
                "scale_compressed_pn_silence_to_observation_receipt": bool(
                    pn_receipts.get("scale_compressed_pn_silence_to_observation_receipt", False)
                ),
                "literal_global_N_capacity_simulated_receipt": bool(
                    pn_receipts.get("literal_global_N_capacity_simulated_receipt", False)
                ),
            },
            "exportSufficiency": "sufficient_for_scale_compressed_bridge_visualization",
            "claimBoundary": pn_silence_payload.get("claimBoundary"),
        },
    }


def _render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"), default=str)
    title = html.escape(payload.get("title", "OPH Universe Timeline Visualization"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: dark; --bg:#0f1115; --panel:#171b21; --ink:#eef2f6; --muted:#aab4be; --line:#303844; --pass:#1d5f3a; --fail:#683033; --accent:#66d9ef; --gold:#f5c66b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }}
header {{ padding:16px 20px; border-bottom:1px solid var(--line); background:#141820; }}
h1 {{ margin:0 0 8px; font-size:22px; font-weight:650; }}
h2 {{ margin:0; padding:10px 12px; font-size:14px; border-bottom:1px solid var(--line); color:#dce6ef; }}
p {{ margin:0; }}
.sub {{ color:var(--muted); font-size:13px; line-height:1.45; max-width:1120px; }}
.gates {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }}
.gate {{ padding:5px 8px; border-radius:6px; background:#2a3038; color:#cbd5df; font-size:12px; }}
.pass {{ background:var(--pass); color:#c9f3d5; }}
.fail {{ background:var(--fail); color:#ffd1d1; }}
main {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; padding:12px; }}
section {{ min-height:330px; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
.wide {{ grid-column:1 / -1; }}
svg {{ width:100%; height:310px; display:block; background:#11151b; }}
.note {{ padding:8px 12px 12px; color:var(--muted); font-size:12px; line-height:1.45; }}
.controls {{ padding:8px 12px 0; display:flex; gap:10px; align-items:center; color:var(--muted); font-size:12px; }}
input[type=range] {{ width:100%; }}
select {{ background:#10151b; color:#eef2f6; border:1px solid var(--line); border-radius:6px; padding:4px 6px; max-width:210px; }}
.readout {{ color:#e7edf2; font-variant-numeric:tabular-nums; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; padding:8px 12px 12px; }}
.explain {{ padding:10px 12px 14px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }}
.explain div {{ border:1px solid var(--line); border-radius:6px; padding:8px; background:#12161d; }}
.explain strong {{ display:block; font-size:12px; margin-bottom:4px; color:#f0f5f8; }}
.explain span {{ color:var(--muted); font-size:12px; line-height:1.45; }}
@media (max-width: 960px) {{ main {{ grid-template-columns:1fr; }} .wide {{ grid-column:auto; }} .explain {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
<h1>{title}</h1>
<p class="sub" id="claim"></p>
<p class="sub" id="diff"></p>
<div class="gates" id="gates"></div>
</header>
<main>
<section>
<h2>1. Fluctuating Quantum Vacuum / Boundary Readback</h2>
<svg id="screenSvg"></svg>
<div class="controls"><span>repair cycle</span><input id="screenCycle" type="range" min="0" max="0" value="0"><span class="readout" id="screenCycleText"></span></div>
<div class="note" id="screenNote"></div>
</section>
<section>
<h2>2. Exact Overlap Repair Mini-Universe</h2>
<svg id="smallSvg"></svg>
<div class="controls"><span>repair step</span><input id="repairStep" type="range" min="0" max="0" value="0"><span class="readout" id="repairText"></span></div>
<div class="note" id="smallNote"></div>
</section>
<section>
<h2>3. Observer Camera / Modular-Time Perspective</h2>
<svg id="observerSvg"></svg>
<div class="controls"><span>observer</span><select id="observerSelect"></select><span>time</span><input id="observerTime" type="range" min="0" max="0" value="0"><span class="readout" id="observerTimeText"></span></div>
<div class="note" id="observerNote"></div>
</section>
<section>
<h2>4. Effective String / H3 Edge-Worldline View</h2>
<svg id="h3Svg"></svg>
<div class="note" id="h3Note"></div>
</section>
<section>
<h2>5. Repair Trace</h2>
<svg id="traceSvg"></svg>
<div class="note">The trace is the large observer-flow run: mismatch phi falls first, then records commit and modular-depth readouts continue.</div>
</section>
<section>
<h2>6. P/N Silence To Observation</h2>
<svg id="pnSvg"></svg>
<div class="note" id="pnNote"></div>
</section>
<section>
<h2>7. CMB Diagnostic Comparison</h2>
<svg id="cmbSvg"></svg>
<div class="note" id="cmbNote"></div>
</section>
<section class="wide">
<h2>OPH Geometry And Symmetry Explanation</h2>
<div class="explain" id="explain"></div>
</section>
</main>
<script>
const DATA = {data};
const NS = "http://www.w3.org/2000/svg";
function el(name, attrs={{}}) {{ const e=document.createElementNS(NS,name); for (const [k,v] of Object.entries(attrs)) e.setAttribute(k,v); return e; }}
function clear(svg) {{ while(svg.firstChild) svg.removeChild(svg.firstChild); }}
function dims(svg) {{ const r=svg.getBoundingClientRect(); return [Math.max(340,r.width), Math.max(260,r.height || 310)]; }}
function gate(label, ok) {{ return `<span class="gate ${{ok?'pass':'fail'}}">${{label}}: ${{ok?'pass':'open'}}</span>`; }}
function color(v) {{ const x=Math.max(0,Math.min(1,Number(v)||0)); const r=Math.round(55+210*x); const g=Math.round(155-55*x); const b=Math.round(220-170*x); return `rgb(${{r}},${{g}},${{b}})`; }}
function s2(p,w,h) {{ const lon=Math.atan2(p[1],p[0]); const lat=Math.asin(Math.max(-1,Math.min(1,p[2]))); return [w*(0.5+lon/(2*Math.PI)), h*(0.5-lat/Math.PI)]; }}
function proj3(p,w,h,scale=0.33) {{ const x=p[0], y=p[1], z=p[2]||0; return [w*(0.5+scale*(x+0.35*z)), h*(0.52-scale*(y-0.25*z))]; }}
function valueRange(rows,key) {{ const vals=rows.map(r=>Number(r[key])).filter(Number.isFinite); return [Math.min(...vals), Math.max(...vals)]; }}
function drawScreen(snapshotIndex=0) {{
  const svg=document.getElementById("screenSvg"); clear(svg); const [w,h]=dims(svg); svg.setAttribute("viewBox",`0 0 ${{w}} ${{h}}`);
  svg.appendChild(el("rect",{{x:0,y:0,width:w,height:h,fill:"#11151b"}}));
  svg.appendChild(el("ellipse",{{cx:w/2,cy:h/2,rx:w*0.46,ry:h*0.42,fill:"none",stroke:"#39424e","stroke-width":1.2}}));
  DATA.screen.points.forEach((p,i)=>{{ const q=s2(p,w,h); svg.appendChild(el("circle",{{cx:q[0],cy:q[1],r:DATA.screen.points.length>2500?1.1:1.7,fill:color(DATA.screen.values[i]),opacity:0.72}})); }});
  const snaps=DATA.screen.clusters.snapshots||[]; const clusters=snaps.length ? (snaps[snapshotIndex]||snaps[0]).clusters : (DATA.screen.clusters.clusters||[]).map(c=>({{point:c.point,class:c.class}}));
  clusters.forEach(c=>{{ const q=s2(c.point,w,h); svg.appendChild(el("circle",{{cx:q[0],cy:q[1],r:c.class==="threecycle"?4.5:3.2,fill:"none",stroke:c.class==="threecycle"?"#f5c66b":"#66d9ef","stroke-width":1.8,opacity:0.9}})); }});
  const cycle=snaps.length ? (snaps[snapshotIndex]||snaps[0]).cycle : "final";
  document.getElementById("screenCycleText").textContent = `cycle ${{cycle}}`;
  document.getElementById("screenNote").textContent = `${{DATA.screen.points.length}} finite S2 screen samples, field=${{DATA.screen.fieldName}}, defect markers=${{clusters.length}}. ${{DATA.screen.claimBoundary}}`;
}}
function drawSmall(frameIndex=0) {{
  const svg=document.getElementById("smallSvg"); clear(svg); const [w,h]=dims(svg); svg.setAttribute("viewBox",`0 0 ${{w}} ${{h}}`);
  svg.appendChild(el("rect",{{x:0,y:0,width:w,height:h,fill:"#11151b"}}));
  const nodes=DATA.smallUniverse.nodes; const frame=DATA.smallUniverse.repairFrames[frameIndex]||DATA.smallUniverse.repairFrames[0]||{{state:[]}};
  const pos=Object.fromEntries(nodes.map(n=>[n.id,proj3(n.position,w,h,0.32)]));
  DATA.smallUniverse.edges.forEach(edge=>{{ const a=pos[edge.source], b=pos[edge.target]; if(!a||!b) return; svg.appendChild(el("line",{{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:"#3a4451","stroke-width":1.4}})); }});
  if(frame.parent!==null && frame.node!==null && pos[frame.parent] && pos[frame.node]) {{
    const a=pos[frame.parent], b=pos[frame.node]; svg.appendChild(el("line",{{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:"#f5c66b","stroke-width":4,opacity:0.8}}));
  }}
  nodes.forEach(n=>{{ const p=pos[n.id]; const bit=Number((frame.state||[])[n.id]||0); const active=(frame.node===n.id); svg.appendChild(el("circle",{{cx:p[0],cy:p[1],r:active?10:8,fill:bit?"#66d9ef":"#242b35",stroke:active?"#f5c66b":"#93a0ac","stroke-width":active?2.6:1.2}})); const t=el("text",{{x:p[0],y:p[1]+3,"text-anchor":"middle","font-size":9,fill:bit?"#061017":"#e5ebf0"}}); t.textContent=String(n.id); svg.appendChild(t); }});
  document.getElementById("repairText").textContent = `step ${{frame.step}} phi=${{frame.phi}}`;
  document.getElementById("smallNote").textContent = `One visual repair path through the exact exhaustive certificate. Active edge is parent -> repaired node; deltaPhi=${{frame.deltaPhi ?? "n/a"}}. Exact nonzero cycle holonomies=${{DATA.smallUniverse.receipts.exact_nonzero_holonomy_cycle_count}}, frustrated-control nonzero holonomies=${{DATA.smallUniverse.receipts.frustrated_nonzero_holonomy_cycle_count}}.`;
}}
function drawObservers(timeIndex=0) {{
  const svg=document.getElementById("observerSvg"); clear(svg); const [w,h]=dims(svg); svg.setAttribute("viewBox",`0 0 ${{w}} ${{h}}`);
  svg.appendChild(el("rect",{{x:0,y:0,width:w,height:h,fill:"#11151b"}}));
  svg.appendChild(el("ellipse",{{cx:w/2,cy:h/2,rx:w*0.46,ry:h*0.42,fill:"none",stroke:"#39424e","stroke-width":1.2}}));
  const perspectives=DATA.observerModularTime.objectiveObserverViews||[];
  const selectedIndex=Math.max(0,Math.min(perspectives.length-1,Number(document.getElementById("observerSelect").value)||0));
  const selected=perspectives[selectedIndex]||null;
  const pos={{}};
  DATA.observerModularTime.observers.forEach(o=>{{ pos[o.observerId]=s2(o.axis,w,h); }});
  (DATA.observerModularTime.overlapLinks||[]).forEach(link=>{{ const a=pos[link.source], b=pos[link.target]; if(!a||!b) return; const j=Math.max(0.02,Math.min(1,Number(link.jaccard)||0)); svg.appendChild(el("line",{{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:"#5e7288","stroke-width":0.6+4*j,opacity:0.12+0.45*j}})); }});
  DATA.observerModularTime.observers.forEach(o=>{{ const p=pos[o.observerId]; svg.appendChild(el("circle",{{cx:p[0],cy:p[1],r:3.2,fill:color(o.modularDepthMean),opacity:0.86,stroke:"#10151b","stroke-width":0.7}})); }});
  if(selected && pos[selected.observerId]) {{
    const p=pos[selected.observerId];
    svg.appendChild(el("circle",{{cx:p[0],cy:p[1],r:9,fill:"none",stroke:"#f5c66b","stroke-width":2.4,opacity:0.95}}));
  }}
  const frame=(selected?.timeFrames||[])[timeIndex]||DATA.observerModularTime.timeFrames[timeIndex]||DATA.observerModularTime.timeFrames[0]||{{}};
  document.getElementById("observerTimeText").textContent = `tau=${{frame.relativeTime ?? "n/a"}}`;
  const os=DATA.observerModularTime.overlapSummary||{{}};
  const step=frame.localTransitionStep||{{}};
  const packets=(selected?.visibleObjectPackets||[]).slice(0,4).map(p=>`${{p.packet}}:${{Number(p.weight).toFixed(2)}}`).join(", ");
  document.getElementById("observerNote").textContent = `${{DATA.observerModularTime.observers.length}} observer-local readouts, ${{(DATA.observerModularTime.overlapLinks||[]).length}} displayed overlap links from ${{os.pairCount ?? "n/a"}} sampled overlapping observer pairs. Selected observer=${{selected?.observerId ?? "n/a"}}, cycle=${{frame.cycle ?? "n/a"}}, phi=${{frame.globalPhi ?? frame.phi ?? "n/a"}}, record=${{frame.dominantRecordSignature ?? "n/a"}}, objectPacket=${{frame.dominantObjectPacket ?? "n/a"}}, transition=${{JSON.stringify(step)}}, visible object packets=[${{packets}}]. ${{DATA.observerModularTime.claimBoundary}}`;
}}
function drawH3() {{
  const svg=document.getElementById("h3Svg"); clear(svg); const [w,h]=dims(svg); svg.setAttribute("viewBox",`0 0 ${{w}} ${{h}}`);
  svg.appendChild(el("rect",{{x:0,y:0,width:w,height:h,fill:"#11151b"}}));
  svg.appendChild(el("line",{{x1:30,y1:h/2,x2:w-24,y2:h/2,stroke:"#2f3945","stroke-width":1}}));
  svg.appendChild(el("line",{{x1:w/2,y1:20,x2:w/2,y2:h-24,stroke:"#2f3945","stroke-width":1}}));
  const objs=DATA.consensusBulk.objects||[];
  const lines=(DATA.consensusBulk.protoParticleCandidates||{{}}).worldlines||[];
  const pts=objs.map(o=>[o.x,o.y,o.z]);
  lines.forEach(line=>(line.events||[]).forEach(ev=>pts.push(ev.h3SpatialPoint)));
  const maxR=Math.max(...pts.map(p=>Math.hypot(p[0],p[1],p[2])),1e-6);
  objs.forEach(o=>{{ const p=[o.x/maxR,o.y/maxR,o.z/maxR]; const q=proj3(p,w,h,0.42); const r=2.5+Math.min(8,Math.sqrt(Number(o.observerCount)||1)*0.35); svg.appendChild(el("circle",{{cx:q[0],cy:q[1],r,fill:color(o.h3CompactnessNormalized ?? 0.5),opacity:0.72,stroke:"#10151b","stroke-width":0.8}})); }});
  lines.forEach((line,i)=>{{
    const evs=(line.events||[]).filter(ev=>Array.isArray(ev.h3SpatialPoint));
    if(!evs.length) return;
    const path=evs.map((ev,j)=>{{ const p=ev.h3SpatialPoint.map(v=>v/maxR); const q=proj3(p,w,h,0.42); return `${{j?'L':'M'}}${{q[0].toFixed(1)}} ${{q[1].toFixed(1)}}`; }}).join(" ");
    svg.appendChild(el("path",{{d:path,fill:"none",stroke:line.particleLike?"#ff6b6b":"#d66bff","stroke-width":line.particleLike?2.4:1.6,opacity:0.82}}));
    const last=evs[evs.length-1].h3SpatialPoint.map(v=>v/maxR); const q=proj3(last,w,h,0.42);
    svg.appendChild(el("circle",{{cx:q[0],cy:q[1],r:4+Math.min(5,Math.sqrt(Number(line.observationCount)||1)*0.45),fill:line.particleLike?"#ff6b6b":"#d66bff",opacity:0.86,stroke:"#10151b","stroke-width":0.9}}));
  }});
  const pr=(DATA.consensusBulk.protoParticleCandidates||{{}}).receipts||{{}};
  document.getElementById("h3Note").textContent = `${{objs.length}} consensus object packets and ${{lines.length}} holonomy/proto-particle candidate worldlines in the derived H3 chart. Observer-facing consensus bulk=${{DATA.consensusBulk.receipts.observer_facing_consensus_3d_bulk_readout_receipt || DATA.consensusBulk.receipts.theorem_assisted_consensus_3d_bulk_readout_receipt}}, chart-blind neutral quotient=${{DATA.consensusBulk.receipts.chart_blind_strict_neutral_quotient_bulk_receipt || DATA.consensusBulk.receipts.strict_neutral_third_person_bulk_receipt}}, bulk worldline precursor=${{pr.bulk_worldline_precursor_receipt}}, particle matter=${{pr.particle_matter_receipt}}. ${{DATA.consensusBulk.claimBoundary}}`;
}}
function drawLine(svg, rows, key, stroke, yLabel) {{
  const vals=rows.map(r=>Number(r[key])).filter(Number.isFinite); if(!vals.length) return;
  const xs=rows.map((r,i)=>Number(r.cycle ?? r.ell ?? i)); const ymin=Math.min(...vals), ymax=Math.max(...vals); const xmin=Math.min(...xs), xmax=Math.max(...xs); const [w,h]=dims(svg);
  const d=rows.map((r,i)=>{{ const yv=Number(r[key]); if(!Number.isFinite(yv)) return ""; const x=34+(w-58)*(xs[i]-xmin)/Math.max(xmax-xmin,1e-9); const y=22+(h-50)*(1-(yv-ymin)/Math.max(ymax-ymin,1e-9)); return `${{i?'L':'M'}}${{x.toFixed(1)}} ${{y.toFixed(1)}}`; }}).join(" ");
  svg.appendChild(el("path",{{d,fill:"none",stroke,"stroke-width":2}}));
}}
function drawTrace() {{
  const svg=document.getElementById("traceSvg"); clear(svg); const [w,h]=dims(svg); svg.setAttribute("viewBox",`0 0 ${{w}} ${{h}}`); svg.appendChild(el("rect",{{x:0,y:0,width:w,height:h,fill:"#11151b"}}));
  const rows=DATA.screen.repairTrace||[]; drawLine(svg,rows,"phi","#66d9ef"); drawLine(svg,rows,"committed_fraction","#b9f26d"); drawLine(svg,rows,"modular_depth_mean","#f5c66b");
}}
function drawPN() {{
  const svg=document.getElementById("pnSvg"); clear(svg); const [w,h]=dims(svg); svg.setAttribute("viewBox",`0 0 ${{w}} ${{h}}`); svg.appendChild(el("rect",{{x:0,y:0,width:w,height:h,fill:"#11151b"}}));
  const pn=DATA.pnSilenceToObservation||{{}}; const c=pn.closureCoordinates||{{}}; const d=pn.finiteRegulatorDepth||{{}}; const s=pn.silenceInitialState||{{}}; const e=pn.observationEmergence||{{}};
  const y=h*0.5; const xs=[w*0.16,w*0.39,w*0.63,w*0.84];
  const labels=[
    ["silent screen",`records=${{s.committedRecords ?? "n/a"}}`],
    ["P detuning",`P-phi=${{Number(c.PDetuningDelta ?? 0).toExponential(3)}}`],
    ["finite repair",`N_eff=${{Number(d.N_eff ?? 0).toExponential(3)}}`],
    ["observer readout",`objects=${{e.h3ObjectCount ?? "n/a"}}`],
  ];
  for(let i=0;i<xs.length-1;i++) svg.appendChild(el("line",{{x1:xs[i]+34,y1:y,x2:xs[i+1]-34,y2:y,stroke:"#39424e","stroke-width":3}}));
  labels.forEach((row,i)=>{{ const pass=i===0?!!s.initialRecordSilenceReceipt:i===3?!!e.observationEmergenceReceipt:true; svg.appendChild(el("circle",{{cx:xs[i],cy:y,r:28,fill:pass?"#1d5f3a":"#683033",stroke:"#e8eef5","stroke-width":1.2,opacity:0.9}})); const t=el("text",{{x:xs[i],y:y-42,"text-anchor":"middle","font-size":12,fill:"#eef2f6"}}); t.textContent=row[0]; svg.appendChild(t); const u=el("text",{{x:xs[i],y:y+50,"text-anchor":"middle","font-size":11,fill:"#aab4be"}}); u.textContent=row[1]; svg.appendChild(u); }});
  const rc=pn.receipts||{{}};
  document.getElementById("pnNote").textContent = `Scale-compressed P/N silence-to-observation=${{rc.scale_compressed_pn_silence_to_observation_receipt}}, literal global N simulated=${{rc.literal_global_N_capacity_simulated_receipt}}, dynamic detuning controls=${{rc.dynamic_p_detuning_control_receipt}}. P=${{c.P ?? "n/a"}}, N*=${{c.NStar ?? "n/a"}}, effective finite repair-round depth=${{d.effectiveRepairRoundDepth ?? "n/a"}}. ${{pn.claimBoundary}}`;
}}
function drawCmb() {{
  const svg=document.getElementById("cmbSvg"); clear(svg); const [w,h]=dims(svg); svg.setAttribute("viewBox",`0 0 ${{w}} ${{h}}`); svg.appendChild(el("rect",{{x:0,y:0,width:w,height:h,fill:"#11151b"}}));
  const rows=DATA.cmbComparison.residualRows||[]; drawLine(svg,rows,"residualSigma","#f5c66b"); drawLine(svg,rows,"model","#66d9ef");
  const rs=DATA.cmbComparison.bestOphResidualSummary||{{}}; document.getElementById("cmbNote").textContent = `Usable physical CMB comparison data=${{DATA.cmbComparison.receipts.USABLE_PHYSICAL_CMB_DATA_RECEIPT}}, physical CMB prediction=${{DATA.cmbComparison.receipts.PHYSICAL_CMB_PREDICTION_RECEIPT}}. Best diagnostic RMS sigma residual=${{rs.rms_sigma_residual ?? "n/a"}}.`;
}}
function explain() {{
  const rows=DATA.geometriesAndSymmetries; const host=document.getElementById("explain"); host.innerHTML="";
  for (const key of Object.keys(rows)) {{ const item=rows[key]; const div=document.createElement("div"); div.innerHTML=`<strong>${{item.name}}</strong><span>${{item.meaning}}</span>`; host.appendChild(div); }}
}}
function init() {{
  document.getElementById("claim").textContent=DATA.claimBoundary;
  document.getElementById("diff").textContent=DATA.ophDifferentiator;
  const su=DATA.smallUniverse.receipts, om=DATA.observerModularTime.receipts, cb=DATA.consensusBulk.receipts, pn=DATA.pnSilenceToObservation.receipts, cmb=DATA.cmbComparison.receipts;
  document.getElementById("gates").innerHTML =
    gate("finite consensus", su.FINITE_CONSENSUS_THEOREM_RECEIPT)+
    gate("P/N silence witness", pn.scale_compressed_pn_silence_to_observation_receipt)+
    gate("observer modular time", om.observer_modular_time_receipt)+
    gate("observer 3+1D/H3", om.observer_facing_3p1d_h3_experience_receipt)+
    gate("observer H3 consensus bulk", cb.observer_facing_consensus_3d_bulk_readout_receipt || cb.theorem_assisted_consensus_3d_bulk_readout_receipt)+
    gate("chart-blind neutral quotient", cb.chart_blind_strict_neutral_quotient_bulk_receipt || cb.strict_neutral_third_person_bulk_receipt)+
    gate("usable CMB comparison", cmb.USABLE_PHYSICAL_CMB_DATA_RECEIPT)+
    gate("physical CMB prediction", cmb.PHYSICAL_CMB_PREDICTION_RECEIPT);
  const observerSelect=document.getElementById("observerSelect");
  (DATA.observerModularTime.objectiveObserverViews||[]).forEach((row,index)=>{{ const option=document.createElement("option"); option.value=String(index); option.textContent=`observer ${{row.observerId}}`; observerSelect.appendChild(option); }});
  observerSelect.onchange=()=>drawObservers(Number(document.getElementById("observerTime").value));
  const repair=document.getElementById("repairStep"); repair.max=Math.max(0,DATA.smallUniverse.repairFrames.length-1); repair.oninput=()=>drawSmall(Number(repair.value));
  const screen=document.getElementById("screenCycle"); const snaps=DATA.screen.clusters.snapshots||[]; screen.max=Math.max(0,snaps.length-1); screen.oninput=()=>drawScreen(Number(screen.value)); screen.style.display=snaps.length>1?"block":"none";
  const obs=document.getElementById("observerTime"); const perspectiveFrames=(DATA.observerModularTime.objectiveObserverViews||[])[0]?.timeFrames||[]; obs.max=Math.max(0,Math.max(DATA.observerModularTime.timeFrames.length,perspectiveFrames.length)-1); obs.oninput=()=>drawObservers(Number(obs.value));
  drawScreen(0); drawSmall(0); drawObservers(0); drawH3(); drawTrace(); drawPN(); drawCmb(); explain();
}}
init();
</script>
</body>
</html>
"""


def _visualization_instructions(viewer_path: Path, payload_path: Path, payload: dict[str, Any]) -> str:
    return f"""# OPH Universe Visualization Instructions

Open the standalone viewer:

```bash
open {viewer_path}
```

Data payload for custom viewers:

```bash
{payload_path}
```

What to inspect:

- Panel 1 shows the fluctuating-vacuum diagnostic view: the finite S2 observer screen/boundary readback from the larger observer-flow run. Colors are screen readback fields; rings are screen-local defect/holonomy residues. This is a diagnostic OPH readback field, not a literal QFT vacuum unless a future receipt says so.
- Panel 2 shows one deterministic repair path through the exact 12-patch mini-universe certificate. The full certificate checks all finite states/schedules; the slider is a readable path through that certified graph.
- Panel 3 shows the observer-camera view and observer-local modular time. Each dot is an observer-like self-reading row with local support, records, readback hash, and modular-depth readout. Use the observer selector to inspect one observer's objective readout across its modular-time frames: record packet, object packet, transition step, local packet histograms, and the global trace cycle used only for synchronization.
- The payload also exports `subjectiveObserverCameras`: first-person rendering cameras derived from visible observer-local readouts. These are the right inputs for a subjective observer camera map.
- Panel 4 shows the effective string-theory diagnostic view. Consensus object packets are shared readback/object packets from overlapping observers. Magenta/red tracks are holonomy/defect worldlines fitted into the same H3 chart: proto-particle candidates and edge-worldline/collar diagnostics, not matter particles or a critical worldsheet unless the corresponding receipts pass.
- Panel 6 shows the scale-compressed P/N silence-to-observation witness: initial record silence, P detuning, finite regulator depth, and observer/H3 readout emergence. This is not a literal brute-force simulation of astronomical N_CRC.
- Panel 7 shows usable CMB comparison diagnostics when present. `comparableObservations` also carries compact measurement-lane summaries for other public-data-facing diagnostics. None of this is a physical prediction unless the relevant prediction receipt passes.
- `visualizationViews.fluctuatingQuantumVacuum`, `visualizationViews.observerCamera`, and `visualizationViews.effectiveStringTheory` are the canonical view contracts for a custom visualizer.

Claim boundary:

{payload["claimBoundary"]}
"""


def _web_agent_brief(payload_path: Path, payload: dict[str, Any]) -> str:
    return f"""# Web Coding Agent Visualization Brief

Build from `visualization_payload.json` only:

```text
{payload_path}
```

Core product goal:

Create an interactive OPH visualization of observer-like self-reading systems. The differentiator must remain explicit: OPH is not generic particles in a box. It is bounded patches with local state, ports/boundaries, readback, records, feedback/repair moves, and public receipts.

Required views:

1. **Fluctuating quantum vacuum / finite screen view**
   - Render `screen.points` as an S2/equirectangular or sphere view.
   - Color by `screen.values` and label the field with `screen.fieldName`.
   - Overlay `screen.clusters.snapshots[*].clusters` as repair/holonomy residues.
   - Use `visualizationViews.fluctuatingQuantumVacuum` for the canonical layer list and claim boundary.
   - Explain that this is the observer boundary/readback surface, not a literal QFT vacuum or a pre-existing 3D bulk.

2. **Overlap repair view**
   - Render `smallUniverse.nodes`, `smallUniverse.edges`, and `smallUniverse.repairFrames`.
   - Animate repair frames with a slider.
   - Highlight `frame.parent -> frame.node`, show `phi`, `deltaPhi`, and strict descent.
   - Show exact zero-holonomy branch beside the frustrated control nonzero-holonomy count.

3. **P/N silence-to-observation view**
   - Render `pnSilenceToObservation.closureCoordinates`.
   - Show the sequence: `silenceInitialState` -> P detuning -> `finiteRegulatorDepth` -> `observationEmergence`.
   - Gate `scale_compressed_pn_silence_to_observation_receipt`, `literal_global_N_capacity_simulated_receipt`, and `dynamic_p_detuning_control_receipt` separately.
   - Explain that this is a finite scale-compressed witness of the OPH thesis, not literal global `N_CRC` cells.

4. **Observer camera / modular-time view**
   - Render `observerModularTime.observers` on the screen by `axis`.
   - Render `observerModularTime.overlapLinks` as the observer-overlap substrate. These are not decorative graph edges; they are the local shared-support relations from which objectivity is read.
   - Animate `observerModularTime.timeFrames`.
   - Add an observer selector backed by `observerModularTime.objectiveObserverViews`.
   - For the selected observer, animate `objectiveObserverViews[*].timeFrames`: show `relativeTime`, `localTransitionStep`, `dominantRecordSignature`, `dominantObjectPacket`, `visibleReadoutHash`, support size, and packet histograms.
   - Use `subjectiveObserverCameras` for subjective/first-person camera rendering. Each camera is derived from one observer-local visible readout and includes `eye`, `lookAt`, `up`, `right`, `forward`, `fovDegrees`, support samples, and time frames.
   - Use `visualizationViews.observerCamera` for the canonical layer list and claim boundary.
   - Do not present this as external global time. It is observer-local modular readout.

5. **Effective string-theory edge/worldsheet view**
   - Use `visualizationViews.effectiveStringTheory` for the canonical layer list and claim boundary.
   - Render `smallUniverse.cycles` and `smallUniverse.repairFrames` as cyclic edge normal forms and swept repair histories.
   - Render `screen.clusters.snapshots[*].clusters` as collar/defect fluctuation markers.
   - Render `consensusBulk.objects` as a 3D scatter/cloud.
   - Size by `observerCount`; color by `h3CompactnessNormalized`.
   - Render `consensusBulk.protoParticleCandidates.worldlines[*].events` as H3 tracks.
   - Use neutral wording: "edge-worldline diagnostic", "consensus object packet", and "holonomy/proto-particle candidate" unless the stronger receipts pass.
   - Do not label it a critical string CFT unless a future critical-edge receipt is true.
   - Gate labels must show observer-facing H3 consensus bulk and chart-blind neutral quotient bulk separately.

6. **CMB diagnostics view**
   - Plot `cmbComparison.residualRows`.
   - Use `comparableObservations.measurementLanes` and `comparableObservations.datasets` for additional public-data-facing diagnostics such as galaxy, CNB, H0/S8, anomaly, repair-clock, object-population, and neutral-frontier lanes when present.
   - Display `USABLE_PHYSICAL_CMB_DATA_RECEIPT` and `PHYSICAL_CMB_PREDICTION_RECEIPT` separately.
   - Never label diagnostic TT comparison as a physical prediction unless the receipt is true.

Visual language:

- Favor direct geometry: finite S2 screen, repair graph, H3 scatter, receipt gates.
- Avoid decorative sci-fi metaphors. Use the OPH explanation text in the payload.
- The first visible text should state that OPH tech instantiates observer-like self-reading systems.
- Every gate badge must be data-driven from `receipts`; no hard-coded success labels.

Receipt boundary to preserve verbatim:

{payload["claimBoundary"]}
"""


def _edges_from_cycles(cycles: Any) -> list[tuple[int, int]]:
    edge_set: set[tuple[int, int]] = set()
    if not isinstance(cycles, list):
        return []
    for cycle in cycles:
        if not isinstance(cycle, dict):
            continue
        for edge in cycle.get("edges", []):
            if not isinstance(edge, list) or len(edge) < 2:
                continue
            left, right = int(edge[0]), int(edge[1])
            edge_set.add((min(left, right), max(left, right)))
    return sorted(edge_set)


def _infer_node_count(edges: list[tuple[int, int]], terminal: Any) -> int:
    values = [value for edge in edges for value in edge]
    if isinstance(terminal, list):
        values.extend(range(len(terminal)))
    return max(values) + 1 if values else 0


def _icosahedron_positions(count: int) -> list[list[float]]:
    if count == 12:
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        raw = [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ]
        return [_unit(row) for row in raw]
    if count <= 0:
        return []
    return [[float(x), float(y), float(z)] for x, y, z in fibonacci_sphere_points(count)]


def _cycle_rows(cycles: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(cycles, list):
        return []
    rows = []
    for row in cycles[:limit]:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "cycle": row.get("cycle", []),
                "edges": row.get("edges", []),
                "holonomyZ2": row.get("holonomy_z2"),
            }
        )
    return rows


def _state_row(path: Path, state_id: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("state_id") == state_id:
                return row
    return {}


def _first_jsonl_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        line = handle.readline()
    return json.loads(line) if line.strip() else {}


def _transition_index(path: Path) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return index
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("branch") != "exact_consensus":
                continue
            index.setdefault(str(row.get("state_id", "")), []).append(row)
    return index


def _expanded_time_grid(time_grid: list[Any], trace: list[dict[str, Any]], *, min_count: int) -> list[float]:
    values = [_optional_float(value) for value in time_grid]
    values = [float(value) for value in values if value is not None]
    if len(values) >= int(min_count):
        return values
    count = max(int(min_count), len(trace), len(values), 1)
    if count <= 1:
        return [values[0] if values else 0.0]
    return [float(index / float(count - 1)) for index in range(count)]


def _observer_overlap_links(
    views: list[dict[str, Any]],
    consensus: dict[str, Any],
    trace_frames: list[dict[str, Any]],
    *,
    max_links: int,
) -> list[dict[str, Any]]:
    patch_views = [
        row
        for row in views
        if row.get("view_type") in (None, "patch_observer")
        and row.get("observer_id") is not None
        and isinstance(row.get("support_nodes"), list)
    ]
    if not patch_views:
        return _sample_pair_overlap_links(consensus, trace_frames, max_links=max_links)

    support_sizes: list[int] = []
    inverted: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(patch_views):
        support = sorted({int(value) for value in row.get("support_nodes", []) if _optional_float(value) is not None})
        support_sizes.append(len(support))
        for node in support:
            inverted[node].append(index)

    overlap_counts: dict[tuple[int, int], int] = defaultdict(int)
    for observer_indices in inverted.values():
        if len(observer_indices) < 2:
            continue
        for left, right in combinations(observer_indices, 2):
            overlap_counts[(left, right)] += 1

    rows = []
    for (left_index, right_index), overlap_count in overlap_counts.items():
        if overlap_count <= 0:
            continue
        left = patch_views[left_index]
        right = patch_views[right_index]
        source = int(left["observer_id"])
        target = int(right["observer_id"])
        union_size = max(1, support_sizes[left_index] + support_sizes[right_index] - int(overlap_count))
        committed = _mean_optional(left.get("committed_fraction"), right.get("committed_fraction"))
        repair = _mean_optional(left.get("repair_load_mean"), right.get("repair_load_mean"))
        rows.append(
            {
                "source": source,
                "target": target,
                "overlapPatchCount": int(overlap_count),
                "jaccard": float(overlap_count / union_size),
                "signatureSimilarity": _packet_histogram_similarity(
                    left.get("record_signature_histogram"),
                    right.get("record_signature_histogram"),
                ),
                "overlapCommittedFraction": committed,
                "overlapRepairLoadMean": repair,
                "repairTrajectory": _overlap_link_trajectory(trace_frames, committed, repair),
                "trajectorySource": (
                    "global_trace_scaled_by_final_overlap_readout; not a direct per-link repair rerun"
                ),
            }
        )
    if not rows:
        return _sample_pair_overlap_links(consensus, trace_frames, max_links=max_links)
    rows.sort(
        key=lambda row: (
            -int(row["overlapPatchCount"]),
            -float(row["jaccard"]),
            int(row["source"]),
            int(row["target"]),
        )
    )
    return rows[: max(0, int(max_links))]


def _sample_pair_overlap_links(
    consensus: dict[str, Any],
    trace_frames: list[dict[str, Any]],
    *,
    max_links: int,
) -> list[dict[str, Any]]:
    rows = []
    for pair in list(consensus.get("sample_pairs", []))[: max(0, int(max_links))] if consensus else []:
        if not isinstance(pair, dict):
            continue
        left = pair.get("observer_a")
        right = pair.get("observer_b")
        if left is None or right is None:
            continue
        committed = _optional_float(pair.get("overlap_committed_fraction"))
        repair = _optional_float(pair.get("overlap_repair_load_mean"))
        rows.append(
            {
                "source": int(left),
                "target": int(right),
                "overlapPatchCount": pair.get("overlap_patch_count"),
                "jaccard": pair.get("jaccard"),
                "signatureSimilarity": pair.get("signature_histogram_similarity"),
                "overlapCommittedFraction": committed,
                "overlapRepairLoadMean": repair,
                "repairTrajectory": _overlap_link_trajectory(trace_frames, committed, repair),
                "trajectorySource": "stored_consensus_sample_pair_scaled_by_global_trace",
            }
        )
    return rows


def _overlap_link_trajectory(
    trace_frames: list[dict[str, Any]],
    final_committed: float | None,
    final_repair: float | None,
) -> list[dict[str, Any]]:
    committed_target = 0.0 if final_committed is None else float(final_committed)
    repair_target = 0.0 if final_repair is None else float(final_repair)
    rows = []
    for frame in trace_frames[:64]:
        global_committed = _optional_float(frame.get("committedFraction"))
        scale = 0.0 if global_committed is None else max(0.0, min(1.0, global_committed))
        rows.append(
            {
                "relativeTime": frame.get("relativeTime"),
                "cycle": frame.get("cycle"),
                "committedFraction": float(committed_target * scale),
                "repairLoadMean": float(repair_target * scale),
            }
        )
    return rows


def _packet_histogram_similarity(left: Any, right: Any) -> float:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return 0.0
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    distance = 0.0
    for key in keys:
        distance += abs(float(left.get(key, 0.0) or 0.0) - float(right.get(key, 0.0) or 0.0))
    return float(max(0.0, 1.0 - 0.5 * distance))


def _mean_optional(*values: Any) -> float | None:
    finite = [_optional_float(value) for value in values]
    finite = [float(value) for value in finite if value is not None]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _vec_norm(values: list[float]) -> float:
    return float(math.sqrt(sum(float(value) * float(value) for value in values)))


def _unit_vec(values: list[float]) -> list[float]:
    norm = _vec_norm(values)
    if norm <= 0.0:
        return [0.0, 0.0, 0.0]
    return [float(value) / norm for value in values[:3]]


def _cross(left: list[float], right: list[float]) -> list[float]:
    return [
        float(left[1] * right[2] - left[2] * right[1]),
        float(left[2] * right[0] - left[0] * right[2]),
        float(left[0] * right[1] - left[1] * right[0]),
    ]


def _compact_metric_map(row: dict[str, Any], *, limit: int) -> dict[str, Any]:
    skip = {"mode", "claim_boundary", "rows", "sample_rows", "run_paths", "source_paths"}
    result: dict[str, Any] = {}
    for key in sorted(row):
        if key in skip or key.endswith("_rows"):
            continue
        value = row.get(key)
        if isinstance(value, (bool, int, float, str)) or value is None:
            result[key] = value
        if len(result) >= limit:
            break
    return result


def _compact_comparable_run_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "run_id",
        "name",
        "patch_count",
        "bulk_3d_established",
        "bulk_proof_chart_level_3p1",
        "bulk_proof_theorem_assisted_h3_populated_chart",
        "bulk_proof_theorem_assisted_h3_nonboundary_population",
        "bulk_proof_strict_neutral_3d_bulk",
        "bulk_proof_screen_cmb_proxy",
        "bulk_proof_physical_cmb_prediction",
        "object_bulk_population_receipt",
        "object_h3_nonboundary_population_receipt",
        "observer_chart_object_count",
        "observer_chart_localized_object_count",
        "observer_chart_localized_not_boundary_object_count",
        "cmb_lite_best_shape_field",
        "cmb_lite_best_shape_correlation",
        "cmb_lite_best_normalized_rmse",
        "physical_cmb_prediction",
        "strict_neutral_bulk_receipt",
        "strict_neutral_median_dimension_estimate",
        "strict_neutral_selected_model",
    )
    return {key: row.get(key) for key in keys if key in row}


def _relative_time_frames(time_grid: list[Any], trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not trace:
        return [{"relativeTime": _optional_float(value), "cycle": None} for value in time_grid]
    max_index = max(0, len(trace) - 1)
    frames = []
    for value in time_grid:
        tau = _optional_float(value)
        normalized = 0.0 if tau is None else max(0.0, min(1.0, tau))
        row = trace[min(max_index, int(round(normalized * max_index)))]
        frames.append(
            {
                "relativeTime": tau,
                "cycle": row.get("cycle"),
                "phase": row.get("phase"),
                "phi": row.get("phi"),
                "committedFraction": row.get("committed_fraction"),
                "modularDepthMean": row.get("modular_depth_mean"),
                "mismatchEdges": row.get("mismatch_edges"),
            }
        )
    return frames


def _observer_perspective_payloads(
    views: list[dict[str, Any]],
    trace_frames: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Build compact objective readouts for individual observer perspectives.

    The source rows are still observer-local: support, records, packets,
    transition descriptors, and modular-time labels. The frames attach the
    global repair trace only as a synchronization aid for visualization.
    """

    rows: list[dict[str, Any]] = []
    for row in views:
        if row.get("view_type") not in (None, "patch_observer"):
            continue
        observer_id = row.get("observer_id")
        axis = row.get("axis")
        if observer_id is None or not isinstance(axis, list) or len(axis) < 3:
            continue
        times = row.get("observer_relative_times")
        if not isinstance(times, list) or not times:
            times = [frame.get("relativeTime") for frame in trace_frames] or [0.0]
        if len(times) < len(trace_frames):
            times = [frame.get("relativeTime") for frame in trace_frames]
        transition_descriptor = row.get("transition_history_descriptor")
        if not isinstance(transition_descriptor, dict):
            transition_descriptor = {}
        static_object_packets = _histogram_rows(row.get("object_packet_histogram"), limit=8)
        static_record_packets = _histogram_rows(row.get("record_signature_histogram"), limit=8)
        frames = []
        for index, value in enumerate(times):
            trace = trace_frames[min(index, len(trace_frames) - 1)] if trace_frames else {}
            step = _transition_step_for_time(transition_descriptor, index=index, count=len(times))
            frame_object_packets = _frame_object_packets(row, step, static_object_packets)
            frame_record_packets = _frame_record_packets(row, step, static_record_packets)
            frames.append(
                {
                    "relativeTime": _optional_float(value),
                    "cycle": trace.get("cycle"),
                    "phase": trace.get("phase"),
                    "globalPhi": trace.get("phi"),
                    "globalCommittedFraction": trace.get("committedFraction"),
                    "localTransitionStep": step,
                    "dominantRecordSignature": row.get("dominant_record_signature"),
                    "dominantObjectPacket": row.get("dominant_object_packet"),
                    "modularDepthMean": row.get("modular_depth_mean"),
                    "visibleSignatureEntropy": row.get("visible_signature_entropy"),
                    "visibleReadoutHash": str(row.get("visible_readout_hash") or "")[:16],
                    "visibleObjectPackets": frame_object_packets,
                    "visibleRecordPackets": frame_record_packets,
                    "polarFieldReadout": _polar_field_readout(frame_object_packets, frame_record_packets),
                    "framePacketSource": (
                        "transition_history_descriptor_plus_final_visible_histograms"
                        if step
                        else "final_visible_histograms_repeated"
                    ),
                }
            )
        rows.append(
            {
                "observerId": int(observer_id),
                "axis": [float(axis[0]), float(axis[1]), float(axis[2])],
                "supportPatchCount": row.get("support_patch_count"),
                "supportNodeSample": list(row.get("support_nodes", [])[:32]) if isinstance(row.get("support_nodes"), list) else [],
                "supportEntropyCapacity": row.get("support_entropy_capacity"),
                "visibleObjectPackets": static_object_packets,
                "visibleRecordPackets": static_record_packets,
                "transitionHistory": {
                    "hash": row.get("transition_history_hash"),
                    "persistence": row.get("transition_history_persistence"),
                    "meanModalMass": row.get("transition_history_mean_modal_mass"),
                    "descriptor": transition_descriptor,
                },
                "timeFrames": frames,
                "claimBoundary": row.get(
                    "claim_boundary",
                    "observer-local objective readout; hidden representatives are not included",
                ),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _transition_step_for_time(descriptor: dict[str, Any], *, index: int, count: int) -> dict[str, Any]:
    steps = descriptor.get("steps")
    if not isinstance(steps, list) or not steps:
        return {}
    if count <= 1:
        step_index = len(steps) - 1
    else:
        step_index = int(round(index * (len(steps) - 1) / max(1, count - 1)))
    step = steps[max(0, min(len(steps) - 1, step_index))]
    return dict(step) if isinstance(step, dict) else {}


def _frame_object_packets(
    row: dict[str, Any],
    step: dict[str, Any],
    static_packets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    if step:
        token_parts = [
            str(step.get(name))
            for name in ("record_family", "checkpoint_class", "s3_sector_class")
            if step.get(name) is not None
        ]
        if token_parts:
            packets.append(
                {
                    "packet": "transition:" + ":".join(token_parts),
                    "weight": 1.0,
                    "source": "transition_history_descriptor",
                }
            )
    dominant = row.get("dominant_object_packet")
    if dominant is not None:
        packets.append({"packet": str(dominant), "weight": 0.75, "source": "dominant_object_packet"})
    for packet in static_packets[:4]:
        packets.append({**packet, "source": packet.get("source", "final_object_packet_histogram")})
    return _dedup_packet_rows(packets, limit=8)


def _frame_record_packets(
    row: dict[str, Any],
    step: dict[str, Any],
    static_packets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    if step.get("record_family") is not None:
        packets.append(
            {
                "packet": f"record_family:{step.get('record_family')}",
                "weight": 1.0,
                "source": "transition_history_descriptor",
            }
        )
    dominant = row.get("dominant_record_signature")
    if dominant is not None:
        packets.append({"packet": str(dominant), "weight": 0.75, "source": "dominant_record_signature"})
    for packet in static_packets[:4]:
        packets.append({**packet, "source": packet.get("source", "final_record_signature_histogram")})
    return _dedup_packet_rows(packets, limit=8)


def _polar_field_readout(
    object_packets: list[dict[str, Any]],
    record_packets: list[dict[str, Any]],
    *,
    bins: int = 16,
) -> list[dict[str, Any]]:
    values = [0.0 for _ in range(max(1, int(bins)))]
    for scale, packets in ((1.0, object_packets), (0.65, record_packets)):
        for packet in packets:
            token = str(packet.get("packet", ""))
            if not token:
                continue
            index = _stable_token_index(token, len(values))
            values[index] += scale * float(packet.get("weight") or 0.0)
    total = max(sum(abs(value) for value in values), 1.0e-12)
    return [
        {"theta": float(2.0 * math.pi * index / len(values)), "value": float(value / total)}
        for index, value in enumerate(values)
    ]


def _dedup_packet_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        packet = str(row.get("packet", ""))
        if not packet:
            continue
        weight = float(row.get("weight") or 0.0)
        if packet not in merged or weight > float(merged[packet].get("weight") or 0.0):
            merged[packet] = {"packet": packet, "weight": weight, "source": row.get("source")}
    result = list(merged.values())
    result.sort(key=lambda item: (-float(item["weight"]), item["packet"]))
    return result[:limit]


def _stable_token_index(token: str, bins: int) -> int:
    value = 0
    for char in token:
        value = (value * 131 + ord(char)) % 2_147_483_647
    return int(value % max(1, int(bins)))


def _histogram_rows(value: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    rows = []
    for key, weight in value.items():
        number = _optional_float(weight)
        if number is None:
            continue
        rows.append({"packet": str(key), "weight": float(number)})
    rows.sort(key=lambda item: (-float(item["weight"]), item["packet"]))
    return rows[:limit]


def _point_from_h3_row(row: dict[str, str]) -> list[float] | None:
    if {"h3_x", "h3_y", "h3_z"}.issubset(row):
        try:
            return [float(row["h3_x"]), float(row["h3_y"]), float(row["h3_z"])]
        except (TypeError, ValueError):
            return None
    raw = row.get("h3_spatial_point")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or len(parsed) < 3:
        return None
    try:
        return [float(parsed[0]), float(parsed[1]), float(parsed[2])]
    except (TypeError, ValueError):
        return None


def _point_from_h3_sample(row: dict[str, Any]) -> list[float] | None:
    value = row.get("h3_spatial_point")
    if isinstance(value, list) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
    value = row.get("h3_point")
    if isinstance(value, list) and len(value) >= 4:
        try:
            return [float(value[1]), float(value[2]), float(value[3])]
        except (TypeError, ValueError):
            return None
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value == "":
                    parsed[key] = value
                    continue
                number = _optional_float(value)
                parsed[key] = number if number is not None else value
            rows.append(parsed)
    return rows


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < 1e-12:
        return np.full(values.shape, 0.5, dtype=float)
    return (values - min_value) / (max_value - min_value)


def _unit(row: tuple[float, float, float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in row))
    return [float(value / norm) for value in row]
