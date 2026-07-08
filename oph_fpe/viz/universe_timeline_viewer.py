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
    max_objective_observer_views: int | None = None,
    max_h3_objects: int = 512,
    write_viewer: bool = True,
    compact_json: bool = False,
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
        max_objective_observer_views=max_objective_observer_views,
        max_h3_objects=max_h3_objects,
    )
    payload_path = output_path / "visualization_payload.json"
    viewer_path = output_path / "oph_universe_timeline_viewer.html"
    instructions_path = output_path / "VISUALIZATION_INSTRUCTIONS.md"
    web_agent_path = output_path / "WEB_CODING_AGENT_VISUALIZATION_BRIEF.md"
    if compact_json:
        payload_json = json.dumps(payload, separators=(",", ":"), default=str)
    else:
        payload_json = json.dumps(payload, indent=2, default=str)
    payload_path.write_text(payload_json, encoding="utf-8")
    sidecar_exports = _write_visualization_sidecars(output_path, payload, payload_path)
    if write_viewer:
        viewer_path.write_text(_render_html(payload), encoding="utf-8")
    instructions_path.write_text(
        _visualization_instructions(viewer_path if write_viewer else None, payload_path, payload),
        encoding="utf-8",
    )
    web_agent_path.write_text(_web_agent_brief(payload_path, payload), encoding="utf-8")
    summary = {
        "mode": "oph_universe_timeline_visualization_bundle",
        "bundle_dir": str(output_path),
        "viewer_path": str(viewer_path) if write_viewer else None,
        "embedded_viewer_written": bool(write_viewer),
        "compact_json": bool(compact_json),
        "payload_path": str(payload_path),
        "sidecar_exports": sidecar_exports,
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
        "neutral_object_candidate_count": len(payload["consensusBulk"].get("neutralObjectCandidates", [])),
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
        "paper_accuracy_guard_receipt": bool(
            payload.get("paperAccuracy", {})
            .get("receipts", {})
            .get("paper_accuracy_guard_receipt", False)
        ),
        "claim_boundary": payload["claimBoundary"],
    }
    (output_path / "universe_timeline_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def _write_visualization_sidecars(
    output_path: Path,
    payload: dict[str, Any],
    payload_path: Path,
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    screen = payload.get("screen", {}) if isinstance(payload.get("screen"), dict) else {}
    points = screen.get("points", []) if isinstance(screen.get("points"), list) else []
    values = screen.get("values", []) if isinstance(screen.get("values"), list) else []
    field_name = str(screen.get("fieldName") or "field")
    screen_rows = []
    for index, point in enumerate(points):
        xyz = _coord3(point)
        if xyz is None:
            continue
        screen_rows.append(
            {
                "screen_point_index": index,
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "field_name": field_name,
                "value": values[index] if index < len(values) else None,
            }
        )
    files["screen_points_csv"] = _write_sidecar_csv(
        output_path / "screen_points.csv",
        ("screen_point_index", "x", "y", "z", "field_name", "value"),
        screen_rows,
    )
    observer_run_dir = _payload_source_dir(payload, "observerRunDir")
    files["screen_full_bin"] = _write_full_screen_field_bin(output_path, observer_run_dir, payload)
    render_data = payload.get("visualizationRenderData") if isinstance(payload.get("visualizationRenderData"), dict) else {}
    render_data_path = output_path / "visualization_render_data.json"
    render_data_path.write_text(json.dumps(render_data, separators=(",", ":"), default=str), encoding="utf-8")
    files["visualization_render_data_json"] = {
        "path": str(render_data_path),
        "byte_count": int(render_data_path.stat().st_size),
        "written": True,
        "schema": render_data.get("schema"),
    }
    for key, filename, file_key in (
        ("effectiveStringTheory", "effective_string_theory.json", "effective_string_theory_json"),
        ("emergentCurvedSpacetime", "emergent_curved_spacetime.json", "emergent_curved_spacetime_json"),
        ("observerCinema", "observer_cinema.json", "observer_cinema_json"),
        (
            "hilbertSpaceObserverAlgebra",
            "hilbert_space_observer_algebra.json",
            "hilbert_space_observer_algebra_json",
        ),
        ("observerAnatomy", "observer_anatomy.json", "observer_anatomy_json"),
        ("paperAccuracy", "paper_accuracy.json", "paper_accuracy_json"),
    ):
        section = payload.get(key) if isinstance(payload.get(key), dict) else {}
        section_path = output_path / filename
        section_path.write_text(json.dumps(section, separators=(",", ":"), default=str), encoding="utf-8")
        files[file_key] = {
            "path": str(section_path),
            "byte_count": int(section_path.stat().st_size),
            "written": True,
            "schema": section.get("schema"),
        }

    cameras = payload.get("subjectiveObserverCameras", [])
    if not isinstance(cameras, list):
        cameras = []
    camera_rows = []
    frame_rows = []
    proto_sighting_rows = []
    for camera in cameras:
        if not isinstance(camera, dict):
            continue
        eye = _coord3(camera.get("eye")) or [None, None, None]
        look_at = _coord3(camera.get("lookAt")) or [None, None, None]
        up = _coord3(camera.get("up")) or [None, None, None]
        right = _coord3(camera.get("right")) or [None, None, None]
        forward = _coord3(camera.get("forward")) or [None, None, None]
        frames = camera.get("timeFrames", []) if isinstance(camera.get("timeFrames"), list) else []
        camera_rows.append(
            {
                "camera_id": camera.get("cameraId"),
                "observer_id": camera.get("observerId"),
                "eye_x": eye[0],
                "eye_y": eye[1],
                "eye_z": eye[2],
                "look_at_x": look_at[0],
                "look_at_y": look_at[1],
                "look_at_z": look_at[2],
                "up_x": up[0],
                "up_y": up[1],
                "up_z": up[2],
                "right_x": right[0],
                "right_y": right[1],
                "right_z": right[2],
                "forward_x": forward[0],
                "forward_y": forward[1],
                "forward_z": forward[2],
                "fov_degrees": camera.get("fovDegrees"),
                "support_patch_count": camera.get("supportPatchCount"),
                "time_frame_count": len(frames),
                "support_node_sample_json": camera.get("supportNodeSample", []),
                "visible_object_packets_json": camera.get("visibleObjectPackets", []),
                "visible_record_packets_json": camera.get("visibleRecordPackets", []),
                "visible_proto_worldline_ids_json": camera.get("visibleProtoWorldlineIds", []),
                "visible_proto_worldline_sighting_count": camera.get("visibleProtoWorldlineSightingCount"),
            }
        )
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            sightings = (
                frame.get("visibleProtoWorldlines", [])
                if isinstance(frame.get("visibleProtoWorldlines"), list)
                else []
            )
            frame_rows.append(
                {
                    "camera_id": camera.get("cameraId"),
                    "observer_id": camera.get("observerId"),
                    "frame_index": frame_index,
                    "relative_time": frame.get("relativeTime"),
                    "cycle": frame.get("cycle"),
                    "visible_readout_hash": frame.get("visibleReadoutHash"),
                    "dominant_record_signature": frame.get("dominantRecordSignature"),
                    "dominant_object_packet": frame.get("dominantObjectPacket"),
                    "local_transition_step": frame.get("localTransitionStep"),
                    "visible_object_packets_json": frame.get("visibleObjectPackets", []),
                    "visible_record_packets_json": frame.get("visibleRecordPackets", []),
                    "visible_proto_worldlines_json": sightings,
                    "polar_field_readout_json": frame.get("polarFieldReadout", []),
                }
            )
            for sighting_index, sighting in enumerate(sightings):
                if not isinstance(sighting, dict):
                    continue
                readout = (
                    sighting.get("observerLocalReadout")
                    if isinstance(sighting.get("observerLocalReadout"), dict)
                    else {}
                )
                proto_sighting_rows.append(
                    {
                        "camera_id": camera.get("cameraId"),
                        "observer_id": camera.get("observerId"),
                        "frame_index": frame_index,
                        "relative_time": frame.get("relativeTime"),
                        "frame_cycle": frame.get("cycle"),
                        "sighting_index": sighting_index,
                        "worldline_id": sighting.get("worldlineId"),
                        "worldline_cycle": sighting.get("cycle"),
                        "nearest_event_cycle": sighting.get("nearestEventCycle"),
                        "cycle_distance": sighting.get("cycleDistance"),
                        "interpolated": sighting.get("interpolated"),
                        "observer_local_u": readout.get("u", sighting.get("screenX")),
                        "observer_local_v": readout.get("v", sighting.get("screenY")),
                        "observer_local_range": readout.get("range", sighting.get("cameraDistance")),
                        "observer_local_range_bucket": readout.get("rangeBucket"),
                        "observer_local_angular_separation_degrees": readout.get(
                            "angularSeparationDegrees", sighting.get("angularSeparationDegrees")
                        ),
                        "readout_coordinate_system": readout.get("coordinateSystem"),
                        "source_coordinate_suppressed": readout.get("sourceCoordinateSuppressed"),
                        "screen_x": sighting.get("screenX"),
                        "screen_y": sighting.get("screenY"),
                        "camera_distance": sighting.get("cameraDistance"),
                        "angular_separation_degrees": sighting.get("angularSeparationDegrees"),
                        "visibility_score": sighting.get("visibilityScore"),
                        "projection_mode": sighting.get("projectionMode"),
                        "outside_nominal_fov": sighting.get("outsideNominalFov"),
                        "event": sighting.get("event"),
                        "class": sighting.get("class"),
                        "holonomy_mode": sighting.get("holonomyMode"),
                        "fit_residual": sighting.get("fitResidual"),
                        "support_node_count": sighting.get("supportNodeCount"),
                        "particle_like": sighting.get("particleLike"),
                        "diagnostic_only": sighting.get("diagnosticOnly"),
                        "controlled_planted_assay": sighting.get("controlledPlantedAssay"),
                        "organic_defect_population_diagnostic": sighting.get("organicDefectPopulationDiagnostic"),
                        "free_dynamics_diagnostic": sighting.get("freeDynamicsDiagnostic"),
                        "contact_outcome": sighting.get("contactOutcome"),
                        "support_overlap_fraction": sighting.get("supportOverlapFraction"),
                        "charge_conservation_pass": sighting.get("chargeConservationPass"),
                        "bulk_localization_pass": sighting.get("bulkLocalizationPass"),
                        "localization_pass": sighting.get("localizationPass"),
                        "persistence_pass": sighting.get("persistencePass"),
                        "transportability_pass": sighting.get("transportabilityPass"),
                        "worldline_source": sighting.get("worldlineSource"),
                    }
                )
    files["subjective_observer_cameras_csv"] = _write_sidecar_csv(
        output_path / "subjective_observer_cameras.csv",
        (
            "camera_id",
            "observer_id",
            "eye_x",
            "eye_y",
            "eye_z",
            "look_at_x",
            "look_at_y",
            "look_at_z",
            "up_x",
            "up_y",
            "up_z",
            "right_x",
            "right_y",
            "right_z",
            "forward_x",
            "forward_y",
            "forward_z",
            "fov_degrees",
            "support_patch_count",
            "time_frame_count",
            "support_node_sample_json",
            "visible_object_packets_json",
            "visible_record_packets_json",
            "visible_proto_worldline_ids_json",
            "visible_proto_worldline_sighting_count",
        ),
        camera_rows,
    )
    files["subjective_observer_camera_frames_csv"] = _write_sidecar_csv(
        output_path / "subjective_observer_camera_frames.csv",
        (
            "camera_id",
            "observer_id",
            "frame_index",
            "relative_time",
            "cycle",
            "visible_readout_hash",
            "dominant_record_signature",
            "dominant_object_packet",
            "local_transition_step",
            "visible_object_packets_json",
            "visible_record_packets_json",
            "visible_proto_worldlines_json",
            "polar_field_readout_json",
        ),
        frame_rows,
    )
    files["observer_proto_worldline_sightings_csv"] = _write_sidecar_csv(
        output_path / "observer_proto_worldline_sightings.csv",
        (
            "camera_id",
            "observer_id",
            "frame_index",
            "relative_time",
            "frame_cycle",
            "sighting_index",
            "worldline_id",
            "worldline_cycle",
            "nearest_event_cycle",
            "cycle_distance",
            "interpolated",
            "observer_local_u",
            "observer_local_v",
            "observer_local_range",
            "observer_local_range_bucket",
            "observer_local_angular_separation_degrees",
            "readout_coordinate_system",
            "source_coordinate_suppressed",
            "screen_x",
            "screen_y",
            "camera_distance",
            "angular_separation_degrees",
            "visibility_score",
            "projection_mode",
            "outside_nominal_fov",
            "event",
            "class",
            "holonomy_mode",
            "fit_residual",
            "support_node_count",
            "particle_like",
            "diagnostic_only",
            "controlled_planted_assay",
            "organic_defect_population_diagnostic",
            "free_dynamics_diagnostic",
            "contact_outcome",
            "support_overlap_fraction",
            "charge_conservation_pass",
            "bulk_localization_pass",
            "localization_pass",
            "persistence_pass",
            "transportability_pass",
            "worldline_source",
        ),
        proto_sighting_rows,
    )
    files["observers_full_json"] = _write_full_observers_json(output_path, observer_run_dir)
    files["cameras_full_json"] = _write_full_cameras_json(output_path, observer_run_dir)

    consensus = payload.get("consensusBulk", {}) if isinstance(payload.get("consensusBulk"), dict) else {}
    objects = consensus.get("objects", []) if isinstance(consensus.get("objects"), list) else []
    object_rows = [
        {
            "object_id": row.get("objectId"),
            "record_family_id": row.get("recordFamilyId"),
            "x": row.get("x"),
            "y": row.get("y"),
            "z": row.get("z"),
            "observer_count": row.get("observerCount"),
            "support_size": row.get("supportSize"),
            "h3_compactness": row.get("h3Compactness"),
            "h3_compactness_normalized": row.get("h3CompactnessNormalized"),
        }
        for row in objects
        if isinstance(row, dict)
    ]
    files["consensus_h3_objects_csv"] = _write_sidecar_csv(
        output_path / "consensus_h3_objects.csv",
        (
            "object_id",
            "record_family_id",
            "x",
            "y",
            "z",
            "observer_count",
            "support_size",
            "h3_compactness",
            "h3_compactness_normalized",
        ),
        object_rows,
    )
    neutral_objects = (
        consensus.get("neutralObjectCandidates", [])
        if isinstance(consensus.get("neutralObjectCandidates"), list)
        else []
    )
    neutral_rows = [
        {
            "object_id": row.get("objectId"),
            "observer_count": row.get("observerCount"),
            "visible_signature_key": row.get("visibleSignatureKey"),
            "persistence": row.get("persistence"),
            "overlap_agreement": row.get("overlapAgreement"),
            "observer_ids_json": json.dumps(row.get("observerIds", []), separators=(",", ":")),
            "spatial_embedding_available": row.get("spatialEmbeddingAvailable"),
            "claim_boundary": row.get("claimBoundary"),
        }
        for row in neutral_objects
        if isinstance(row, dict)
    ]
    files["neutral_object_candidates_csv"] = _write_sidecar_csv(
        output_path / "neutral_object_candidates.csv",
        (
            "object_id",
            "observer_count",
            "visible_signature_key",
            "persistence",
            "overlap_agreement",
            "observer_ids_json",
            "spatial_embedding_available",
            "claim_boundary",
        ),
        neutral_rows,
    )

    cmb = payload.get("cmbComparison", {}) if isinstance(payload.get("cmbComparison"), dict) else {}
    residual_rows = cmb.get("residualRows", []) if isinstance(cmb.get("residualRows"), list) else []
    cmb_rows = [
        {
            "row_index": index,
            "ell": row.get("ell"),
            "observed": row.get("observed"),
            "model": row.get("model"),
            "residual_sigma": row.get("residualSigma"),
        }
        for index, row in enumerate(residual_rows)
        if isinstance(row, dict)
    ]
    files["cmb_residual_rows_csv"] = _write_sidecar_csv(
        output_path / "cmb_residual_rows.csv",
        ("row_index", "ell", "observed", "model", "residual_sigma"),
        cmb_rows,
    )
    screen_spectrum_rows = (
        cmb.get("screenDiagnosticSpectrumRows", [])
        if isinstance(cmb.get("screenDiagnosticSpectrumRows"), list)
        else []
    )
    screen_cmb_rows = [
        {
            "row_index": index,
            "field": row.get("field"),
            "ell": row.get("ell"),
            "C_ell": row.get("C_ell"),
            "D_ell": row.get("D_ell"),
            "normalized_D_ell": row.get("normalizedD_ell"),
        }
        for index, row in enumerate(screen_spectrum_rows)
        if isinstance(row, dict)
    ]
    files["cmb_screen_spectrum_rows_csv"] = _write_sidecar_csv(
        output_path / "cmb_screen_spectrum_rows.csv",
        ("row_index", "field", "ell", "C_ell", "D_ell", "normalized_D_ell"),
        screen_cmb_rows,
    )

    vacuum_view = (
        (payload.get("visualizationViews") or {}).get("fluctuatingQuantumVacuum", {})
        if isinstance(payload.get("visualizationViews"), dict)
        else {}
    )
    reference_vacuum = (
        vacuum_view.get("referenceVacuumBaseline", {})
        if isinstance(vacuum_view.get("referenceVacuumBaseline"), dict)
        else {}
    )
    scalar = (
        reference_vacuum.get("freeScalarGaussian", {})
        if isinstance(reference_vacuum.get("freeScalarGaussian"), dict)
        else {}
    )
    vacuum_rows = []
    for spectrum_kind, source_key in (("raw", "rawSpectrum"), ("smoothed", "smoothedSpectrum")):
        rows = scalar.get(source_key, []) if isinstance(scalar.get(source_key), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            vacuum_rows.append(
                {
                    "spectrum_kind": spectrum_kind,
                    "ell": row.get("ell"),
                    "mean_coefficient_power": row.get("mean_coefficient_power"),
                    "mode_count": row.get("mode_count"),
                }
            )
    files["reference_vacuum_scalar_spectrum_csv"] = _write_sidecar_csv(
        output_path / "reference_vacuum_scalar_spectrum.csv",
        ("spectrum_kind", "ell", "mean_coefficient_power", "mode_count"),
        vacuum_rows,
    )
    compact_u1 = (
        reference_vacuum.get("compactU1LatticeGauge", {})
        if isinstance(reference_vacuum.get("compactU1LatticeGauge"), dict)
        else {}
    )
    plaquette_trace = compact_u1.get("plaquetteTrace", [])
    if not isinstance(plaquette_trace, list):
        plaquette_trace = []
    plaquette_rows = [{"sweep": index, "mean_plaquette": value} for index, value in enumerate(plaquette_trace)]
    files["reference_vacuum_u1_plaquette_trace_csv"] = _write_sidecar_csv(
        output_path / "reference_vacuum_u1_plaquette_trace.csv",
        ("sweep", "mean_plaquette"),
        plaquette_rows,
    )
    yang_mills = (
        vacuum_view.get("yangMillsGapCertificate", {})
        if isinstance(vacuum_view.get("yangMillsGapCertificate"), dict)
        else {}
    )
    ym_plaquette_rows = [
        {
            "sweep": row.get("sweep"),
            "mean_plaquette": row.get("mean_plaquette"),
            "plaquette_variance": row.get("plaquette_variance"),
            "action_density": row.get("action_density"),
            "acceptance_rate": row.get("acceptance_rate"),
        }
        for row in (yang_mills.get("plaquetteTrace", []) if isinstance(yang_mills.get("plaquetteTrace"), list) else [])
        if isinstance(row, dict)
    ]
    files["yang_mills_su2_plaquette_trace_csv"] = _write_sidecar_csv(
        output_path / "yang_mills_su2_plaquette_trace.csv",
        ("sweep", "mean_plaquette", "plaquette_variance", "action_density", "acceptance_rate"),
        ym_plaquette_rows,
    )
    ym_wilson_rows = [
        {
            "sweep": row.get("sweep"),
            "loop": row.get("loop"),
            "mean_normalized_trace": row.get("mean_normalized_trace"),
            "variance": row.get("variance"),
        }
        for row in (yang_mills.get("wilsonLoopTrace", []) if isinstance(yang_mills.get("wilsonLoopTrace"), list) else [])
        if isinstance(row, dict)
    ]
    files["yang_mills_su2_wilson_loop_trace_csv"] = _write_sidecar_csv(
        output_path / "yang_mills_su2_wilson_loop_trace.csv",
        ("sweep", "loop", "mean_normalized_trace", "variance"),
        ym_wilson_rows,
    )
    ym_polyakov_rows = [
        {
            "sweep": row.get("sweep"),
            "loop": row.get("loop"),
            "mean_abs_normalized_trace": row.get("mean_abs_normalized_trace"),
        }
        for row in (yang_mills.get("polyakovLoopTrace", []) if isinstance(yang_mills.get("polyakovLoopTrace"), list) else [])
        if isinstance(row, dict)
    ]
    files["yang_mills_su2_polyakov_loop_trace_csv"] = _write_sidecar_csv(
        output_path / "yang_mills_su2_polyakov_loop_trace.csv",
        ("sweep", "loop", "mean_abs_normalized_trace"),
        ym_polyakov_rows,
    )
    ym_orientation_rows = [
        {
            "sweep": row.get("sweep"),
            "orientation": row.get("orientation"),
            "mean_plaquette": row.get("mean_plaquette"),
        }
        for row in (
            yang_mills.get("orientationPlaquetteRows", [])
            if isinstance(yang_mills.get("orientationPlaquetteRows"), list)
            else []
        )
        if isinstance(row, dict)
    ]
    files["yang_mills_su2_orientation_plaquettes_csv"] = _write_sidecar_csv(
        output_path / "yang_mills_su2_orientation_plaquettes.csv",
        ("sweep", "orientation", "mean_plaquette"),
        ym_orientation_rows,
    )
    ym_refinement_rows = [
        {
            "lattice_size": row.get("lattice_size"),
            "site_count": row.get("site_count"),
            "sweeps": row.get("sweeps"),
            "mean_plaquette": row.get("mean_plaquette"),
            "plaquette_variance": row.get("plaquette_variance"),
            "acceptance_rate": row.get("acceptance_rate"),
            "finite_transfer_gap_estimate": row.get("finite_transfer_gap_estimate"),
            "lambda2_abs": row.get("lambda2_abs"),
            "screening_mass_proxy": row.get("screening_mass_proxy"),
            "finite_transfer_gap_proxy_receipt": row.get("finite_transfer_gap_proxy_receipt"),
        }
        for row in (yang_mills.get("refinementGapRows", []) if isinstance(yang_mills.get("refinementGapRows"), list) else [])
        if isinstance(row, dict)
    ]
    files["yang_mills_su2_refinement_gap_csv"] = _write_sidecar_csv(
        output_path / "yang_mills_su2_refinement_gap.csv",
        (
            "lattice_size",
            "site_count",
            "sweeps",
            "mean_plaquette",
            "plaquette_variance",
            "acceptance_rate",
            "finite_transfer_gap_estimate",
            "lambda2_abs",
            "screening_mass_proxy",
            "finite_transfer_gap_proxy_receipt",
        ),
        ym_refinement_rows,
    )
    promotion = yang_mills.get("promotionStatus", {}) if isinstance(yang_mills.get("promotionStatus"), dict) else {}
    ym_promotion_rows = [
        {"gate": key, "status": value}
        for key, value in promotion.items()
        if key != "reasons"
    ] + [
        {"gate": "blocker", "status": reason}
        for reason in (promotion.get("reasons", []) if isinstance(promotion.get("reasons"), list) else [])
    ]
    files["yang_mills_gap_promotion_gates_csv"] = _write_sidecar_csv(
        output_path / "yang_mills_gap_promotion_gates.csv",
        ("gate", "status"),
        ym_promotion_rows,
    )

    small = payload.get("smallUniverse", {}) if isinstance(payload.get("smallUniverse"), dict) else {}
    repair_rows = []
    repair_frames = small.get("repairFrames", []) if isinstance(small.get("repairFrames"), list) else []
    for frame in repair_frames:
        if not isinstance(frame, dict):
            continue
        repair_rows.append(
            {
                "step": frame.get("step"),
                "state_id": frame.get("stateId"),
                "phi": frame.get("phi"),
                "phi_before": frame.get("phiBefore"),
                "delta_phi": frame.get("deltaPhi"),
                "enabled_repair_count": frame.get("enabledRepairCount"),
                "action": frame.get("action"),
                "node": frame.get("node"),
                "parent": frame.get("parent"),
                "strict_descent": frame.get("strictDescent"),
                "state_json": frame.get("state", []),
            }
        )
    files["finite_repair_frames_csv"] = _write_sidecar_csv(
        output_path / "finite_repair_frames.csv",
        (
            "step",
            "state_id",
            "phi",
            "phi_before",
            "delta_phi",
            "enabled_repair_count",
            "action",
            "node",
            "parent",
            "strict_descent",
            "state_json",
        ),
        repair_rows,
    )
    cycle_rows = []
    cycles = small.get("cycles", {}) if isinstance(small.get("cycles"), dict) else {}
    for cycle_kind, source_rows in (
        ("exact_consensus", cycles.get("exactConsensus", [])),
        ("frustrated_control", cycles.get("frustratedControl", [])),
    ):
        if not isinstance(source_rows, list):
            continue
        for cycle_index, row in enumerate(source_rows):
            if not isinstance(row, dict):
                continue
            cycle_edges = row.get("edges", []) if isinstance(row.get("edges"), list) else []
            cycle_rows.append(
                {
                    "cycle_kind": cycle_kind,
                    "cycle_index": cycle_index,
                    "holonomy_z2": row.get("holonomyZ2"),
                    "node_count": len(row.get("cycle", [])) if isinstance(row.get("cycle"), list) else None,
                    "edge_count": len(cycle_edges),
                    "cycle_json": row.get("cycle", []),
                    "edges_json": cycle_edges,
                }
            )
    files["finite_cycle_rows_csv"] = _write_sidecar_csv(
        output_path / "finite_cycle_rows.csv",
        (
            "cycle_kind",
            "cycle_index",
            "holonomy_z2",
            "node_count",
            "edge_count",
            "cycle_json",
            "edges_json",
        ),
        cycle_rows,
    )
    effective_string_payload = (
        payload.get("effectiveStringTheory", {})
        if isinstance(payload.get("effectiveStringTheory"), dict)
        else {}
    )
    string_vibration_rows = [
        {
            "sample_index": row.get("sampleIndex"),
            "cycle_kind": row.get("cycleKind"),
            "cycle_index": row.get("cycleIndex"),
            "frame_step": row.get("frameStep"),
            "repair_node": row.get("repairNode"),
            "repair_parent": row.get("repairParent"),
            "edge_index": row.get("edgeIndex"),
            "edge_json": row.get("edge", []),
            "loop_phase": row.get("loopPhase"),
            "edge_pulse": row.get("edgePulse"),
            "state_occupation": row.get("stateOccupation"),
            "delta_phi": row.get("deltaPhi"),
            "normalized_amplitude": row.get("normalizedAmplitude"),
            "sample_kind": row.get("sampleKind"),
            "claim_boundary": row.get("claimBoundary"),
        }
        for row in (
            effective_string_payload.get("finiteEdgeStringVibrationSamples", [])
            if isinstance(effective_string_payload.get("finiteEdgeStringVibrationSamples"), list)
            else []
        )
        if isinstance(row, dict)
    ]
    files["finite_edge_string_vibration_samples_csv"] = _write_sidecar_csv(
        output_path / "finite_edge_string_vibration_samples.csv",
        (
            "sample_index",
            "cycle_kind",
            "cycle_index",
            "frame_step",
            "repair_node",
            "repair_parent",
            "edge_index",
            "edge_json",
            "loop_phase",
            "edge_pulse",
            "state_occupation",
            "delta_phi",
            "normalized_amplitude",
            "sample_kind",
            "claim_boundary",
        ),
        string_vibration_rows,
    )

    cluster_snapshots = (
        (screen.get("clusters") or {}).get("snapshots", [])
        if isinstance(screen.get("clusters"), dict)
        else []
    )
    cluster_rows = []
    if not isinstance(cluster_snapshots, list):
        cluster_snapshots = []
    for snapshot_index, snapshot in enumerate(cluster_snapshots):
        if not isinstance(snapshot, dict):
            continue
        clusters = snapshot.get("clusters", []) if isinstance(snapshot.get("clusters"), list) else []
        for cluster_index, cluster in enumerate(clusters):
            if not isinstance(cluster, dict):
                continue
            point = _coord3(cluster.get("point")) or [None, None, None]
            cluster_rows.append(
                {
                    "snapshot_index": snapshot_index,
                    "cycle": snapshot.get("cycle"),
                    "cluster_index": cluster_index,
                    "cluster_id": cluster.get("clusterId"),
                    "worldline_id": cluster.get("worldlineId"),
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                    "cluster_class": cluster.get("class"),
                    "support_node_count": cluster.get("supportNodeCount"),
                    "interpolated": snapshot.get("interpolated"),
                }
            )
    files["screen_cluster_tracks_csv"] = _write_sidecar_csv(
        output_path / "screen_cluster_tracks.csv",
        (
            "snapshot_index",
            "cycle",
            "cluster_index",
            "cluster_id",
            "worldline_id",
            "x",
            "y",
            "z",
            "cluster_class",
            "support_node_count",
            "interpolated",
        ),
        cluster_rows,
    )

    proto = consensus.get("protoParticleCandidates", {}) if isinstance(consensus.get("protoParticleCandidates"), dict) else {}
    proto_worldline_rows = []
    proto_event_rows = []
    proto_worldlines = proto.get("worldlines", []) if isinstance(proto.get("worldlines"), list) else []
    for worldline in proto_worldlines:
        if not isinstance(worldline, dict):
            continue
        worldline_id = worldline.get("worldlineId")
        proto_worldline_rows.append(
            {
                "worldline_id": worldline_id,
                "observation_count": worldline.get("observationCount"),
                "birth_cycle": worldline.get("birthCycle"),
                "death_cycle": worldline.get("deathCycle"),
                "h3_path_length": worldline.get("h3PathLength"),
                "mean_h3_step": worldline.get("meanH3Step"),
                "class_mode": worldline.get("classMode"),
                "particle_like": worldline.get("particleLike"),
                "localization_pass": worldline.get("localizationPass"),
                "persistence_pass": worldline.get("persistencePass"),
                "sector_stability_pass": worldline.get("sectorStabilityPass"),
                "transportability_pass": worldline.get("transportabilityPass"),
                "bulk_localization_pass": worldline.get("bulkLocalizationPass"),
            }
        )
        events = worldline.get("events", []) if isinstance(worldline.get("events"), list) else []
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            point = _coord3(event.get("h3SpatialPoint")) or [None, None, None]
            proto_event_rows.append(
                {
                    "worldline_id": worldline_id,
                    "event_index": event_index,
                    "cycle": event.get("cycle"),
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                    "fit_residual": event.get("fitResidual"),
                    "support_node_count": event.get("supportNodeCount"),
                    "particle_like": worldline.get("particleLike"),
                }
            )
    files["proto_particle_worldlines_csv"] = _write_sidecar_csv(
        output_path / "proto_particle_worldlines.csv",
        (
            "worldline_id",
            "observation_count",
            "birth_cycle",
            "death_cycle",
            "h3_path_length",
            "mean_h3_step",
            "class_mode",
            "particle_like",
            "localization_pass",
            "persistence_pass",
            "sector_stability_pass",
            "transportability_pass",
            "bulk_localization_pass",
        ),
        proto_worldline_rows,
    )
    files["proto_particle_worldline_events_csv"] = _write_sidecar_csv(
        output_path / "proto_particle_worldline_events.csv",
        (
            "worldline_id",
            "event_index",
            "cycle",
            "x",
            "y",
            "z",
            "fit_residual",
            "support_node_count",
            "particle_like",
        ),
        proto_event_rows,
    )

    curved_spacetime = (
        payload.get("emergentCurvedSpacetime", {})
        if isinstance(payload.get("emergentCurvedSpacetime"), dict)
        else {}
    )
    paper_accuracy = payload.get("paperAccuracy", {}) if isinstance(payload.get("paperAccuracy"), dict) else {}
    curvature_rows = []
    for row in (
        curved_spacetime.get("curvatureProxyPoints", [])
        if isinstance(curved_spacetime.get("curvatureProxyPoints"), list)
        else []
    ):
        if not isinstance(row, dict):
            continue
        point = _coord3(row.get("position")) or [row.get("x"), row.get("y"), row.get("z")]
        curvature_rows.append(
            {
                "source_id": row.get("sourceId"),
                "source_kind": row.get("sourceKind"),
                "coordinate_system": row.get("coordinateSystem"),
                "x": point[0],
                "y": point[1],
                "z": point[2],
                "cycle": row.get("cycle"),
                "relative_time": row.get("relativeTime"),
                "mass_proxy": row.get("massProxy"),
                "stress_energy_proxy": row.get("stressEnergyProxy"),
                "source_density": row.get("sourceDensity"),
                "normalized_source_density": row.get("normalizedSourceDensity"),
                "quotient_visible_source_density": row.get("quotientVisibleSourceDensity"),
                "h3_green_potential": row.get("h3GreenPotential"),
                "normalized_h3_green_potential": row.get("normalizedH3GreenPotential"),
                "curvature_potential": row.get("curvaturePotential"),
                "compactification_factor": row.get("compactificationFactor"),
                "emergent_spatial_scale_factor": row.get("emergentSpatialScaleFactor"),
                "local_metric_conformal_factor": row.get("localMetricConformalFactor"),
                "curvature_radius_proxy": row.get("curvatureRadiusProxy"),
                "observer_count": row.get("observerCount"),
                "support_size": row.get("supportSize"),
                "support_node_count": row.get("supportNodeCount"),
                "source_density_ancestry": row.get("sourceDensityAncestry"),
                "gravity_source_interpretation": row.get("gravitySourceInterpretation"),
                "worldline_id": row.get("worldlineId"),
                "event_index": row.get("eventIndex"),
                "particle_like": row.get("particleLike"),
                "diagnostic_only": row.get("diagnosticOnly"),
                "production_gravity_contributor": row.get("productionGravityContributor"),
            }
        )
    files["emergent_curved_spacetime_curvature_proxy_csv"] = _write_sidecar_csv(
        output_path / "emergent_curved_spacetime_curvature_proxy.csv",
        (
            "source_id",
            "source_kind",
            "coordinate_system",
            "x",
            "y",
            "z",
            "cycle",
            "relative_time",
            "mass_proxy",
            "stress_energy_proxy",
            "source_density",
            "normalized_source_density",
            "quotient_visible_source_density",
            "h3_green_potential",
            "normalized_h3_green_potential",
            "curvature_potential",
            "compactification_factor",
            "emergent_spatial_scale_factor",
            "local_metric_conformal_factor",
            "curvature_radius_proxy",
            "observer_count",
            "support_size",
            "support_node_count",
            "source_density_ancestry",
            "gravity_source_interpretation",
            "worldline_id",
            "event_index",
            "particle_like",
            "diagnostic_only",
            "production_gravity_contributor",
        ),
        curvature_rows,
    )
    curvature_time_rows = [
        {
            "slice_index": row.get("sliceIndex"),
            "cycle": row.get("cycle"),
            "relative_time": row.get("relativeTime"),
            "event_count": row.get("eventCount"),
            "source_count": row.get("sourceCount"),
            "total_source_density": row.get("totalSourceDensity"),
            "total_curvature_potential": row.get("totalCurvaturePotential"),
            "max_curvature_potential": row.get("maxCurvaturePotential"),
            "max_compactification_factor": row.get("maxCompactificationFactor"),
            "mean_emergent_spatial_scale_factor": row.get("meanEmergentSpatialScaleFactor"),
        }
        for row in (
            curved_spacetime.get("timeSlices", [])
            if isinstance(curved_spacetime.get("timeSlices"), list)
            else []
        )
        if isinstance(row, dict)
    ]
    files["emergent_curved_spacetime_time_slices_csv"] = _write_sidecar_csv(
        output_path / "emergent_curved_spacetime_time_slices.csv",
        (
            "slice_index",
            "cycle",
            "relative_time",
            "event_count",
            "source_count",
            "total_source_density",
            "total_curvature_potential",
            "max_curvature_potential",
            "max_compactification_factor",
            "mean_emergent_spatial_scale_factor",
        ),
        curvature_time_rows,
    )
    continuous_field = (
        curved_spacetime.get("continuousBulkField", {})
        if isinstance(curved_spacetime.get("continuousBulkField"), dict)
        else {}
    )
    field_columns = (
        "sample_id",
        "sample_kind",
        "slice_index",
        "cycle",
        "relative_time",
        "slice_axis",
        "slice_value",
        "x",
        "y",
        "z",
        "density",
        "normalized_density",
        "h3_green_potential",
        "normalized_curvature_potential",
        "curvature_potential",
        "compactification_factor",
        "emergent_spatial_scale_factor",
        "local_metric_conformal_factor",
    )
    field_rows = []
    for sample_kind, key in (
        ("volume", "volumeSamples"),
        ("z_slice", "sliceSamples"),
        ("temporal_z_slice", "temporalSliceSamples"),
    ):
        rows = continuous_field.get(key, []) if isinstance(continuous_field.get(key), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            field_rows.append(
                {
                    "sample_id": row.get("sampleId"),
                    "sample_kind": sample_kind,
                    "slice_index": row.get("sliceIndex"),
                    "cycle": row.get("cycle"),
                    "relative_time": row.get("relativeTime"),
                    "slice_axis": row.get("sliceAxis"),
                    "slice_value": row.get("sliceValue"),
                    "x": row.get("x"),
                    "y": row.get("y"),
                    "z": row.get("z"),
                    "density": row.get("density"),
                    "normalized_density": row.get("normalizedDensity"),
                    "h3_green_potential": row.get("h3GreenPotential"),
                    "normalized_curvature_potential": row.get("normalizedCurvaturePotential"),
                    "curvature_potential": row.get("curvaturePotential"),
                    "compactification_factor": row.get("compactificationFactor"),
                    "emergent_spatial_scale_factor": row.get("emergentSpatialScaleFactor"),
                    "local_metric_conformal_factor": row.get("localMetricConformalFactor"),
                }
            )
    files["emergent_curved_spacetime_continuous_field_csv"] = _write_sidecar_csv(
        output_path / "emergent_curved_spacetime_continuous_field.csv",
        field_columns,
        field_rows,
    )

    effective_string = (
        (payload.get("visualizationViews") or {}).get("effectiveStringTheory", {})
        if isinstance(payload.get("visualizationViews"), dict)
        else {}
    )
    stress = (
        effective_string.get("twoDefectStressContractionAssay", {})
        if isinstance(effective_string.get("twoDefectStressContractionAssay"), dict)
        else {}
    )
    free_dynamics = (
        effective_string.get("freeTwoDefectDynamics", {})
        if isinstance(effective_string.get("freeTwoDefectDynamics"), dict)
        else {}
    )
    organic_defects = (
        effective_string.get("organicDefectPopulation", {})
        if isinstance(effective_string.get("organicDefectPopulation"), dict)
        else {}
    )
    string_selector = (
        effective_string.get("stringVacuumSelector", {})
        if isinstance(effective_string.get("stringVacuumSelector"), dict)
        else {}
    )
    curved_receipts = (
        curved_spacetime.get("receipts", {}) if isinstance(curved_spacetime.get("receipts"), dict) else {}
    )
    paper_receipts = (
        paper_accuracy.get("receipts", {}) if isinstance(paper_accuracy.get("receipts"), dict) else {}
    )
    organic_rows = _organic_defect_sidecar_rows(organic_defects.get("trajectoryRows", []))
    files["organic_defect_population_trajectory_csv"] = _write_sidecar_csv(
        output_path / "organic_defect_population_trajectory.csv",
        _ORGANIC_DEFECT_SIDECAR_FIELDS,
        organic_rows,
    )
    organic_worldline_rows = []
    organic_event_rows = []
    for worldline in (
        organic_defects.get("worldlines", []) if isinstance(organic_defects.get("worldlines"), list) else []
    ):
        if not isinstance(worldline, dict):
            continue
        worldline_id = worldline.get("worldlineId")
        organic_worldline_rows.append(
            {
                "worldline_id": worldline_id,
                "observation_count": worldline.get("observationCount"),
                "birth_cycle": worldline.get("birthCycle"),
                "death_cycle": worldline.get("deathCycle"),
                "lifetime_cycles": worldline.get("lifetimeCycles"),
                "persistent": worldline.get("persistent"),
                "class_mode": worldline.get("classMode"),
                "holonomy_mode": worldline.get("holonomyMode"),
                "mean_transport_distance": worldline.get("meanTransportDistance"),
            }
        )
        events = worldline.get("events", []) if isinstance(worldline.get("events"), list) else []
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            point = _coord3(event.get("h3SpatialPoint")) or [None, None, None]
            velocity = _coord3(event.get("velocity")) or [None, None, None]
            organic_event_rows.append(
                {
                    "worldline_id": worldline_id,
                    "event_index": event_index,
                    "cycle": event.get("cycle"),
                    "event": event.get("event"),
                    "class": event.get("class"),
                    "holonomy_mode": event.get("holonomyMode"),
                    "support_node_count": event.get("supportNodeCount"),
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                    "vx": velocity[0],
                    "vy": velocity[1],
                    "vz": velocity[2],
                    "local_stress_density": event.get("localStressDensity"),
                    "nearest_defect_id": event.get("nearestDefectId"),
                    "nearest_h3_separation": event.get("nearestH3Separation"),
                    "support_overlap_fraction": event.get("supportOverlapFraction"),
                    "support_overlap_node_count": event.get("supportOverlapNodeCount"),
                    "contact_outcome": event.get("contactOutcome"),
                    "transport_distance": event.get("transportDistance"),
                }
            )
    files["organic_defect_population_worldlines_csv"] = _write_sidecar_csv(
        output_path / "organic_defect_population_worldlines.csv",
        (
            "worldline_id",
            "observation_count",
            "birth_cycle",
            "death_cycle",
            "lifetime_cycles",
            "persistent",
            "class_mode",
            "holonomy_mode",
            "mean_transport_distance",
        ),
        organic_worldline_rows,
    )
    files["organic_defect_population_worldline_events_csv"] = _write_sidecar_csv(
        output_path / "organic_defect_population_worldline_events.csv",
        (
            "worldline_id",
            "event_index",
            "cycle",
            "event",
            "class",
            "holonomy_mode",
            "support_node_count",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "local_stress_density",
            "nearest_defect_id",
            "nearest_h3_separation",
            "support_overlap_fraction",
            "support_overlap_node_count",
            "contact_outcome",
            "transport_distance",
        ),
        organic_event_rows,
    )
    stress_rows = _stress_sidecar_rows(stress.get("trajectoryRows", []), control_name="stress_contraction")
    control_rows = []
    controls = stress.get("controlTrajectoryRows", {}) if isinstance(stress.get("controlTrajectoryRows"), dict) else {}
    for control_name, rows in controls.items():
        control_rows.extend(_stress_sidecar_rows(rows, control_name=str(control_name)))
    files["two_defect_stress_trajectory_csv"] = _write_sidecar_csv(
        output_path / "two_defect_stress_trajectory.csv",
        _STRESS_SIDECAR_FIELDS,
        stress_rows,
    )
    files["two_defect_stress_controls_csv"] = _write_sidecar_csv(
        output_path / "two_defect_stress_controls.csv",
        _STRESS_SIDECAR_FIELDS,
        control_rows,
    )
    stress_worldline_rows = []
    stress_event_rows = []
    for worldline in stress.get("worldlines", []) if isinstance(stress.get("worldlines"), list) else []:
        if not isinstance(worldline, dict):
            continue
        worldline_id = worldline.get("worldlineId")
        stress_worldline_rows.append(
            {
                "worldline_id": worldline_id,
                "observation_count": worldline.get("observationCount"),
                "birth_cycle": worldline.get("birthCycle"),
                "death_cycle": worldline.get("deathCycle"),
                "lifetime_cycles": worldline.get("lifetimeCycles"),
                "persistent": worldline.get("persistent"),
                "mean_transport_distance": worldline.get("meanTransportDistance"),
            }
        )
        events = worldline.get("events", []) if isinstance(worldline.get("events"), list) else []
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            point = _coord3(event.get("h3SpatialPoint")) or [None, None, None]
            stress_event_rows.append(
                {
                    "worldline_id": worldline_id,
                    "event_index": event_index,
                    "cycle": event.get("cycle"),
                    "event": event.get("event"),
                    "class": event.get("class"),
                    "holonomy_mode": event.get("holonomyMode"),
                    "support_node_count": event.get("supportNodeCount"),
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                    "pair_h3_separation": event.get("pairH3Separation"),
                    "local_readout_contraction": event.get("localReadoutContraction"),
                    "transport_distance": event.get("transportDistance"),
                }
            )
    files["two_defect_stress_worldlines_csv"] = _write_sidecar_csv(
        output_path / "two_defect_stress_worldlines.csv",
        (
            "worldline_id",
            "observation_count",
            "birth_cycle",
            "death_cycle",
            "lifetime_cycles",
            "persistent",
            "mean_transport_distance",
        ),
        stress_worldline_rows,
    )
    files["two_defect_stress_worldline_events_csv"] = _write_sidecar_csv(
        output_path / "two_defect_stress_worldline_events.csv",
        (
            "worldline_id",
            "event_index",
            "cycle",
            "event",
            "class",
            "holonomy_mode",
            "support_node_count",
            "x",
            "y",
            "z",
            "pair_h3_separation",
            "local_readout_contraction",
            "transport_distance",
        ),
        stress_event_rows,
    )
    free_rows = _free_dynamics_sidecar_rows(free_dynamics.get("trajectoryRows", []))
    files["free_two_defect_dynamics_trajectory_csv"] = _write_sidecar_csv(
        output_path / "free_two_defect_dynamics_trajectory.csv",
        _FREE_DYNAMICS_SIDECAR_FIELDS,
        free_rows,
    )
    free_worldline_rows = []
    free_event_rows = []
    for worldline in (
        free_dynamics.get("worldlines", []) if isinstance(free_dynamics.get("worldlines"), list) else []
    ):
        if not isinstance(worldline, dict):
            continue
        worldline_id = worldline.get("worldlineId")
        free_worldline_rows.append(
            {
                "worldline_id": worldline_id,
                "observation_count": worldline.get("observationCount"),
                "birth_cycle": worldline.get("birthCycle"),
                "death_cycle": worldline.get("deathCycle"),
                "lifetime_cycles": worldline.get("lifetimeCycles"),
                "persistent": worldline.get("persistent"),
                "contact_outcome": worldline.get("contactOutcome"),
                "mean_transport_distance": worldline.get("meanTransportDistance"),
            }
        )
        events = worldline.get("events", []) if isinstance(worldline.get("events"), list) else []
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            point = _coord3(event.get("h3SpatialPoint")) or [None, None, None]
            velocity = _coord3(event.get("velocity")) or [None, None, None]
            free_event_rows.append(
                {
                    "worldline_id": worldline_id,
                    "event_index": event_index,
                    "cycle": event.get("cycle"),
                    "event": event.get("event"),
                    "class": event.get("class"),
                    "holonomy_mode": event.get("holonomyMode"),
                    "support_node_count": event.get("supportNodeCount"),
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                    "vx": velocity[0],
                    "vy": velocity[1],
                    "vz": velocity[2],
                    "pair_h3_separation": event.get("pairH3Separation"),
                    "support_overlap_fraction": event.get("supportOverlapFraction"),
                    "support_overlap_node_count": event.get("supportOverlapNodeCount"),
                    "contact_outcome": event.get("contactOutcome"),
                    "charge_conservation_pass": event.get("chargeConservationPass"),
                    "transport_distance": event.get("transportDistance"),
                }
            )
    files["free_two_defect_dynamics_worldlines_csv"] = _write_sidecar_csv(
        output_path / "free_two_defect_dynamics_worldlines.csv",
        (
            "worldline_id",
            "observation_count",
            "birth_cycle",
            "death_cycle",
            "lifetime_cycles",
            "persistent",
            "contact_outcome",
            "mean_transport_distance",
        ),
        free_worldline_rows,
    )
    files["free_two_defect_dynamics_worldline_events_csv"] = _write_sidecar_csv(
        output_path / "free_two_defect_dynamics_worldline_events.csv",
        (
            "worldline_id",
            "event_index",
            "cycle",
            "event",
            "class",
            "holonomy_mode",
            "support_node_count",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "pair_h3_separation",
            "support_overlap_fraction",
            "support_overlap_node_count",
            "contact_outcome",
            "charge_conservation_pass",
            "transport_distance",
        ),
        free_event_rows,
    )
    selector_candidates = [
        {
            "candidate": row.get("candidate"),
            "score_numerator": row.get("scoreNumerator"),
            "score_denominator": row.get("scoreDenominator"),
            "selected": row.get("selected"),
            "verdict": row.get("verdict"),
        }
        for row in (
            string_selector.get("encodedCandidateSieve", [])
            if isinstance(string_selector.get("encodedCandidateSieve"), list)
            else []
        )
        if isinstance(row, dict)
    ]
    files["string_vacuum_selector_candidates_csv"] = _write_sidecar_csv(
        output_path / "string_vacuum_selector_candidates.csv",
        ("candidate", "score_numerator", "score_denominator", "selected", "verdict"),
        selector_candidates,
    )
    selector_gates = [
        {
            "gate_id": row.get("gateId"),
            "label": row.get("label"),
            "requirement": row.get("requirement"),
            "status": row.get("status"),
            "visual_state": row.get("visualState"),
        }
        for row in (
            string_selector.get("acceptanceGates", [])
            if isinstance(string_selector.get("acceptanceGates"), list)
            else []
        )
        if isinstance(row, dict)
    ]
    files["string_vacuum_selector_gates_csv"] = _write_sidecar_csv(
        output_path / "string_vacuum_selector_gates.csv",
        ("gate_id", "label", "requirement", "status", "visual_state"),
        selector_gates,
    )
    critical_gates = [
        {
            "gate_id": row.get("gateId"),
            "label": row.get("label"),
            "needed_input": row.get("neededInput"),
            "status": row.get("status"),
            "receipt": row.get("receipt"),
        }
        for row in (
            string_selector.get("criticalEdgeCertificateGates", [])
            if isinstance(string_selector.get("criticalEdgeCertificateGates"), list)
            else []
        )
        if isinstance(row, dict)
    ]
    files["string_vacuum_selector_critical_edge_gates_csv"] = _write_sidecar_csv(
        output_path / "string_vacuum_selector_critical_edge_gates.csv",
        ("gate_id", "label", "needed_input", "status", "receipt"),
        critical_gates,
    )
    operator_rows = [
        {
            "operator": row.get("operator"),
            "r_charge_mod4": row.get("rChargeMod4"),
            "allowed": row.get("allowed"),
            "result": row.get("result"),
        }
        for row in (
            string_selector.get("operatorSafetyTable", [])
            if isinstance(string_selector.get("operatorSafetyTable"), list)
            else []
        )
        if isinstance(row, dict)
    ]
    files["string_vacuum_selector_operator_safety_csv"] = _write_sidecar_csv(
        output_path / "string_vacuum_selector_operator_safety.csv",
        ("operator", "r_charge_mod4", "allowed", "result"),
        operator_rows,
    )
    target_rows = [
        {
            "quantity": row.get("quantity"),
            "label": row.get("label"),
            "value": row.get("value"),
            "unit": row.get("unit"),
        }
        for row in (
            string_selector.get("quantitativeTargets", [])
            if isinstance(string_selector.get("quantitativeTargets"), list)
            else []
        )
        if isinstance(row, dict)
    ]
    files["string_vacuum_selector_quantitative_targets_csv"] = _write_sidecar_csv(
        output_path / "string_vacuum_selector_quantitative_targets.csv",
        ("quantity", "label", "value", "unit"),
        target_rows,
    )

    receipts = {
        "observer_facing_consensus_3d_bulk_readout_receipt": bool(
            (consensus.get("receipts") or {}).get("observer_facing_consensus_3d_bulk_readout_receipt", False)
        ),
        "theorem_assisted_consensus_3d_bulk_readout_receipt": bool(
            (consensus.get("receipts") or {}).get("theorem_assisted_consensus_3d_bulk_readout_receipt", False)
        ),
        "strict_neutral_third_person_bulk_receipt": bool(
            (consensus.get("receipts") or {}).get("strict_neutral_third_person_bulk_receipt", False)
        ),
        "physical_cmb_prediction_receipt": bool(
            (cmb.get("receipts") or {}).get("PHYSICAL_CMB_PREDICTION_RECEIPT", False)
            or (cmb.get("receipts") or {}).get("physical_cmb_prediction", False)
        ),
        "reference_vacuum_regression_receipt": bool(
            (reference_vacuum.get("receipts") or {}).get("reference_vacuum_regression_receipt", False)
        ),
        "finite_nonabelian_gauge_gap_diagnostic_receipt": bool(
            (yang_mills.get("receipts") or {}).get("finite_nonabelian_gauge_gap_diagnostic_receipt", False)
        ),
        "yang_mills_gap_reproduced_receipt": bool(
            (yang_mills.get("receipts") or {}).get("YANG_MILLS_GAP_REPRODUCED_RECEIPT", False)
        ),
        "clay_yang_mills_gap_receipt": bool(
            (yang_mills.get("receipts") or {}).get("CLAY_YANG_MILLS_GAP_RECEIPT", False)
        ),
        "oph_native_vacuum_promotion_receipt": bool(
            (reference_vacuum.get("receipts") or {}).get("OPH_NATIVE_VACUUM_PROMOTION_RECEIPT", False)
        ),
        "bulk_worldline_precursor_receipt": bool(
            (proto.get("receipts") or {}).get("bulk_worldline_precursor_receipt", False)
        ),
        "particle_matter_receipt": bool((proto.get("receipts") or {}).get("particle_matter_receipt", False)),
        "emergent_curved_spacetime_visualization_receipt": bool(
            (curved_spacetime.get("receipts") or {}).get(
                "emergent_curved_spacetime_visualization_receipt", False
            )
        ),
        "two_defect_stress_contraction_assay_receipt": bool(
            (stress.get("receipts") or {}).get("two_defect_stress_contraction_assay_receipt", False)
        ),
        "organic_defect_population_receipt": bool(
            (organic_defects.get("receipts") or {}).get("organic_defect_population_receipt", False)
        ),
        "organic_proto_worldline_visualization_receipt": bool(
            (organic_defects.get("receipts") or {}).get("organic_proto_worldline_visualization_receipt", False)
        ),
        "free_two_defect_dynamics_receipt": bool(
            (free_dynamics.get("receipts") or {}).get("free_two_defect_dynamics_receipt", False)
        ),
        "string_vacuum_selector_visualization_receipt": bool(
            (string_selector.get("receipts") or {}).get("string_vacuum_selector_visualization_receipt", False)
        ),
        "finite_edge_string_vibration_receipt": bool(
            (payload.get("effectiveStringTheory") or {}).get("finiteEdgeStringVibrationReceipt", False)
        ),
        "encoded_structural_audit_data_receipt": bool(
            (string_selector.get("receipts") or {}).get("encoded_structural_audit_data_receipt", False)
        ),
        "critical_edge_cft_receipt": bool(
            (string_selector.get("receipts") or {}).get("critical_edge_cft_receipt", False)
        ),
        "global_singleton_string_vacuum_receipt": bool(
            (string_selector.get("receipts") or {}).get("global_singleton_string_vacuum_receipt", False)
        ),
        "gravity_like_free_dynamics_diagnostic_receipt": bool(
            (free_dynamics.get("receipts") or {}).get("gravity_like_free_dynamics_diagnostic_receipt", False)
        ),
        "einstein_branch_entry_receipt": bool(
            curved_receipts.get(
                "einstein_branch_entry_receipt",
                paper_receipts.get("EINSTEIN_BRANCH_ENTRY_RECEIPT", False),
            )
        ),
        "EINSTEIN_BRANCH_ENTRY_RECEIPT": bool(
            curved_receipts.get(
                "EINSTEIN_BRANCH_ENTRY_RECEIPT",
                paper_receipts.get("EINSTEIN_BRANCH_ENTRY_RECEIPT", False),
            )
        ),
        "OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1": bool(
            curved_receipts.get(
                "OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1",
                paper_receipts.get("OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1", False),
            )
        ),
        "raw_production_gravity_requested": bool(
            curved_receipts.get(
                "raw_production_gravity_requested",
                (stress.get("receipts") or {}).get("raw_production_gravity_requested", False)
                or (organic_defects.get("receipts") or {}).get("raw_production_gravity_requested", False)
                or (free_dynamics.get("receipts") or {}).get("raw_production_gravity_requested", False),
            )
        ),
        "raw_physical_gravity_requested": bool(
            curved_receipts.get(
                "raw_physical_gravity_requested",
                (stress.get("receipts") or {}).get("raw_physical_gravity_requested", False)
                or (organic_defects.get("receipts") or {}).get("raw_physical_gravity_requested", False)
                or (free_dynamics.get("receipts") or {}).get("raw_physical_gravity_requested", False),
            )
        ),
        "production_gravity_receipt": bool(
            curved_receipts.get(
                "production_gravity_receipt",
                paper_receipts.get("production_gravity_receipt", False),
            )
        ),
        "physical_gravity_prediction": bool(
            curved_receipts.get(
                "physical_gravity_prediction",
                paper_receipts.get("physical_gravity_prediction", False),
            )
        ),
        "raw_physical_gravity_from_lanes": bool(
            (stress.get("receipts") or {}).get("raw_physical_gravity_requested", False)
            or (organic_defects.get("receipts") or {}).get("raw_physical_gravity_requested", False)
            or (free_dynamics.get("receipts") or {}).get("raw_physical_gravity_requested", False)
        ),
        "paper_accuracy_guard_receipt": bool(
            (paper_accuracy.get("receipts") or {}).get("paper_accuracy_guard_receipt", False)
        ),
        "no_semantic_promotion_by_relabeling_receipt": bool(
            (paper_accuracy.get("receipts") or {}).get("no_semantic_promotion_by_relabeling_receipt", False)
        ),
    }
    manifest = {
        "schema": "oph_universe_visualization_sidecars_v1",
        "payload_path": str(payload_path),
        "payload_schema": payload.get("schemaVersion") or payload.get("schema"),
        "files": files,
        "receipts": receipts,
        "claim_boundary": (
            "CSV sidecars are normalized mirrors of visualization_payload.json for renderer throughput. "
            "The JSON payload remains authoritative for receipts and claim status."
        ),
    }
    manifest_path = output_path / "visualization_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {
        "schema": manifest["schema"],
        "manifest_path": str(manifest_path),
        "files": files,
        "receipts": receipts,
    }


def _payload_source_dir(payload: dict[str, Any], key: str) -> Path | None:
    source_paths = payload.get("sourcePaths") if isinstance(payload.get("sourcePaths"), dict) else {}
    value = source_paths.get(key)
    if not value:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def _write_full_screen_field_bin(
    output_path: Path,
    run_dir: Path | None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if run_dir is None:
        return {"path": None, "row_count": 0, "written": False, "reason": "observer_run_dir_missing"}
    npz_path = Path(run_dir) / "freezeout_fields.npz"
    source = str(npz_path)
    exact_values = True
    fallback_reason: str | None = None
    field_name = "record_signature"
    if npz_path.exists():
        try:
            with np.load(npz_path) as data:
                points = np.asarray(data["points"], dtype=float)
                field_name = "record_signature" if "record_signature" in data.files else next(
                    (name for name in data.files if name not in {"points", "cell_area_planck", "cell_entropy"}),
                    "uniform",
                )
                values = (
                    np.asarray(data[field_name], dtype=float)
                    if field_name in data.files
                    else np.zeros(points.shape[0], dtype=float)
                )
        except Exception as exc:  # pragma: no cover - corrupted run artifact path.
            return {
                "path": None,
                "row_count": 0,
                "written": False,
                "reason": f"freezeout_fields_npz_unreadable:{type(exc).__name__}",
            }
    else:
        points, field_name, values, fallback_reason = _fallback_full_screen_field(Path(run_dir), payload or {})
        source = str(Path(run_dir) / "manifest.json")
        exact_values = False
    if points.ndim != 2 or points.shape[1] < 3 or values.ndim != 1 or values.shape[0] != points.shape[0]:
        return {
            "path": None,
            "row_count": 0,
            "written": False,
            "reason": "full_screen_field_shape_mismatch",
            "source": source,
        }
    row_count = int(points.shape[0])
    path = output_path / f"screen_full_{row_count}.bin"
    packed = np.column_stack([points[:, :3], _normalize(values)]).astype("<f4", copy=False)
    packed.tofile(path)
    result = {
        "path": str(path),
        "row_count": row_count,
        "byte_count": int(path.stat().st_size),
        "dtype": "float32-le",
        "layout": "x,y,z,value",
        "field_name": field_name,
        "source": source,
        "exact_freezeout_values": exact_values,
        "written": True,
    }
    if fallback_reason:
        result["reason"] = fallback_reason
        result["claim_boundary"] = (
            "Full S2 regulator coordinates are exact for the run patch count. Per-patch freezeout "
            "field values were not persisted by this compact run, so the value channel is a neutral "
            "support field rather than the raw record-signature foam."
        )
    return result


def _fallback_full_screen_field(
    run_dir: Path,
    payload: dict[str, Any],
) -> tuple[np.ndarray, str, np.ndarray, str]:
    screen = payload.get("screen") if isinstance(payload.get("screen"), dict) else {}
    points_payload = screen.get("points") if isinstance(screen, dict) else None
    values_payload = screen.get("values") if isinstance(screen, dict) else None
    if isinstance(points_payload, list) and isinstance(values_payload, list) and len(points_payload) == len(values_payload):
        points = np.asarray(points_payload, dtype=float)
        values = np.asarray(values_payload, dtype=float)
        manifest_count = _screen_patch_count_from_run(run_dir, default=points.shape[0])
        if points.ndim == 2 and points.shape[1] >= 3 and points.shape[0] == manifest_count:
            return points, str(screen.get("fieldName") or "payload_screen_field"), values, "freezeout_fields_npz_missing_payload_full_field_used"

    count = _screen_patch_count_from_run(run_dir, default=0)
    if count <= 0:
        count = 512
    points = fibonacci_sphere_points(count)
    values = np.zeros(count, dtype=float)
    return points, "screen_position_support", values, "freezeout_fields_npz_missing_full_s2_support_only"


def _screen_patch_count_from_run(run_dir: Path, *, default: int) -> int:
    freezeout = _read_json(run_dir / "freezeout_map_summary.json")
    count = _safe_int(freezeout.get("point_count"))
    if count > 0:
        return count
    manifest = _read_json(run_dir / "manifest.json")
    count = _safe_int(manifest.get("patch_count"))
    if count > 0:
        return count
    return int(default)


def _write_full_observers_json(output_path: Path, run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {"path": None, "row_count": 0, "written": False, "reason": "observer_run_dir_missing"}
    views = _read_jsonl(Path(run_dir) / "observer_views.jsonl", limit=10_000_000)
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
    path = output_path / f"observers_full_{len(observers)}.json"
    data = {
        "schema": "oph_observers_full_v1",
        "source": str(Path(run_dir) / "observer_views.jsonl"),
        "observerCount": len(observers),
        "observers": observers,
    }
    path.write_text(json.dumps(data, separators=(",", ":"), default=str), encoding="utf-8")
    return {
        "path": str(path),
        "row_count": len(observers),
        "byte_count": int(path.stat().st_size),
        "written": True,
    }


def _write_full_cameras_json(output_path: Path, run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {"path": None, "row_count": 0, "written": False, "reason": "observer_run_dir_missing"}
    run_path = Path(run_dir)
    views = _read_jsonl(run_path / "observer_views.jsonl", limit=10_000_000)
    observer_report = _read_json(run_path / "observer_modular_experience_report.json")
    trace = _read_trace(run_path / "mismatch_trace.csv")
    time_grid = observer_report.get("observer_relative_time_grid")
    if not isinstance(time_grid, list) or not time_grid:
        time_grid = next((row.get("observer_relative_times") for row in views if row.get("observer_relative_times")), [])
    if not isinstance(time_grid, list) or not time_grid:
        time_grid = [0.0]
    trace_frames = _relative_time_frames(_expanded_time_grid(time_grid, trace, min_count=32), trace)
    objective_views = _observer_perspective_payloads(views, trace_frames, limit=len(views))
    subjective_cameras = _subjective_observer_cameras({"objectiveObserverViews": objective_views})
    path = output_path / f"cameras_full_{len(subjective_cameras)}.json"
    data = {
        "schema": "oph_observer_cameras_full_v1",
        "source": str(run_path / "observer_views.jsonl"),
        "objectiveObserverViewCount": len(objective_views),
        "subjectiveObserverCameraCount": len(subjective_cameras),
        "objectiveObserverViews": objective_views,
        "subjectiveObserverCameras": subjective_cameras,
        "claimBoundary": (
            "Full observer camera sidecar generated from observer-local readouts. It is a renderer input, "
            "not a hidden global observer state."
        ),
    }
    path.write_text(json.dumps(data, separators=(",", ":"), default=str), encoding="utf-8")
    return {
        "path": str(path),
        "row_count": len(subjective_cameras),
        "objective_observer_view_count": len(objective_views),
        "byte_count": int(path.stat().st_size),
        "written": True,
    }


_STRESS_SIDECAR_FIELDS: tuple[str, ...] = (
    "control_name",
    "mode",
    "step",
    "cycle",
    "left_x",
    "left_y",
    "left_z",
    "right_x",
    "right_y",
    "right_z",
    "tangent_separation",
    "h3_separation",
    "stress_kernel",
    "local_readout_contraction",
)


_FREE_DYNAMICS_SIDECAR_FIELDS: tuple[str, ...] = (
    "mode",
    "step",
    "cycle",
    "left_x",
    "left_y",
    "left_z",
    "right_x",
    "right_y",
    "right_z",
    "left_vx",
    "left_vy",
    "left_vz",
    "right_vx",
    "right_vy",
    "right_vz",
    "tangent_separation",
    "h3_separation",
    "stress_kernel",
    "relative_speed",
    "support_overlap_fraction",
    "support_overlap_node_count",
    "contact_event",
    "contact_outcome",
    "charge_conservation_pass",
)

_ORGANIC_DEFECT_SIDECAR_FIELDS: tuple[str, ...] = (
    "mode",
    "step",
    "cycle",
    "defect_id",
    "birth_trigger",
    "class",
    "holonomy_mode",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "local_stress_density",
    "nearest_defect_id",
    "nearest_h3_separation",
    "support_overlap_fraction",
    "support_overlap_node_count",
    "contact_event",
    "contact_outcome",
    "support_node_count",
    "render_as_point",
    "render_as_string",
    "render_in_subjective_observer_view",
)


def _stress_sidecar_rows(value: Any, *, control_name: str) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        left = _coord3(row.get("leftH3SpatialPoint")) or [None, None, None]
        right = _coord3(row.get("rightH3SpatialPoint")) or [None, None, None]
        out.append(
            {
                "control_name": control_name,
                "mode": row.get("mode"),
                "step": row.get("step"),
                "cycle": row.get("cycle"),
                "left_x": left[0],
                "left_y": left[1],
                "left_z": left[2],
                "right_x": right[0],
                "right_y": right[1],
                "right_z": right[2],
                "tangent_separation": row.get("tangentSeparation"),
                "h3_separation": row.get("h3Separation"),
                "stress_kernel": row.get("stressKernel"),
                "local_readout_contraction": row.get("localReadoutContraction"),
            }
        )
    return out


def _free_dynamics_sidecar_rows(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        left = _coord3(row.get("leftH3SpatialPoint")) or [None, None, None]
        right = _coord3(row.get("rightH3SpatialPoint")) or [None, None, None]
        left_velocity = _coord3(row.get("leftVelocity")) or [None, None, None]
        right_velocity = _coord3(row.get("rightVelocity")) or [None, None, None]
        out.append(
            {
                "mode": row.get("mode"),
                "step": row.get("step"),
                "cycle": row.get("cycle"),
                "left_x": left[0],
                "left_y": left[1],
                "left_z": left[2],
                "right_x": right[0],
                "right_y": right[1],
                "right_z": right[2],
                "left_vx": left_velocity[0],
                "left_vy": left_velocity[1],
                "left_vz": left_velocity[2],
                "right_vx": right_velocity[0],
                "right_vy": right_velocity[1],
                "right_vz": right_velocity[2],
                "tangent_separation": row.get("tangentSeparation"),
                "h3_separation": row.get("h3Separation"),
                "stress_kernel": row.get("stressKernel"),
                "relative_speed": row.get("relativeSpeed"),
                "support_overlap_fraction": row.get("supportOverlapFraction"),
                "support_overlap_node_count": row.get("supportOverlapNodeCount"),
                "contact_event": row.get("contactEvent"),
                "contact_outcome": row.get("contactOutcome"),
                "charge_conservation_pass": row.get("chargeConservationPass"),
            }
        )
    return out


def _organic_defect_sidecar_rows(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        point = _coord3(row.get("h3SpatialPoint")) or _coord3(row.get("h3_spatial_point")) or [
            row.get("x"),
            row.get("y"),
            row.get("z"),
        ]
        velocity = _coord3(row.get("velocity")) or [row.get("vx"), row.get("vy"), row.get("vz")]
        out.append(
            {
                "mode": row.get("mode"),
                "step": row.get("step"),
                "cycle": row.get("cycle"),
                "defect_id": row.get("defectId", row.get("defect_id")),
                "birth_trigger": row.get("birthTrigger", row.get("birth_trigger")),
                "class": row.get("class"),
                "holonomy_mode": row.get("holonomyMode", row.get("holonomy_mode")),
                "x": point[0],
                "y": point[1],
                "z": point[2],
                "vx": velocity[0],
                "vy": velocity[1],
                "vz": velocity[2],
                "local_stress_density": row.get("localStressDensity", row.get("local_stress_density")),
                "nearest_defect_id": row.get("nearestDefectId", row.get("nearest_defect_id")),
                "nearest_h3_separation": row.get("nearestH3Separation", row.get("nearest_h3_separation")),
                "support_overlap_fraction": row.get("supportOverlapFraction", row.get("support_overlap_fraction")),
                "support_overlap_node_count": row.get(
                    "supportOverlapNodeCount", row.get("support_overlap_node_count")
                ),
                "contact_event": row.get("contactEvent", row.get("contact_event")),
                "contact_outcome": row.get("contactOutcome", row.get("contact_outcome")),
                "support_node_count": row.get("supportNodeCount", row.get("support_node_count")),
                "render_as_point": row.get("renderAsPoint", row.get("render_as_point")),
                "render_as_string": row.get("renderAsString", row.get("render_as_string")),
                "render_in_subjective_observer_view": row.get(
                    "renderInSubjectiveObserverView", row.get("render_in_subjective_observer_view")
                ),
            }
        )
    return out


def _write_sidecar_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field)) for field in fieldnames})
    return {"path": str(path), "row_count": len(rows), "written": True}


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"), default=str)
    if value is None:
        return ""
    return value


def _coord3(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def build_universe_timeline_payload(
    *,
    small_universe_dir: Path,
    observer_run_dir: Path,
    consensus_pack_dir: Path | None,
    consensus_readout_dir: Path | None,
    max_screen_points: int,
    max_observers: int,
    max_objective_observer_views: int | None,
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
        max_objective_observer_views=max_objective_observer_views,
    )
    screen_payload = _screen_payload(Path(observer_run_dir), max_points=max_screen_points)
    bulk_payload = _consensus_bulk_payload(
        Path(consensus_pack_dir) if consensus_pack_dir is not None else None,
        Path(consensus_readout_dir) if consensus_readout_dir is not None else None,
        max_objects=max_h3_objects,
    )
    cmb_payload = _cmb_payload(Path(consensus_pack_dir) if consensus_pack_dir is not None else None)
    subjective_cameras = _subjective_observer_cameras(observer_payload, bulk_payload=bulk_payload)
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
        diagnostic_run_dir=Path(consensus_pack_dir) if consensus_pack_dir is not None else Path(observer_run_dir),
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
        pn_silence_payload=pn_silence_payload,
        subjective_cameras=subjective_cameras,
        visualization_views=visualization_views,
        curved_spacetime_payload=emergent_curved_spacetime_payload,
    )
    effective_string_payload = _effective_string_theory_payload(
        visualization_views=visualization_views,
        small_payload=small_payload,
        bulk_payload=bulk_payload,
    )
    observer_cinema_payload = _observer_cinema_payload(
        observer_payload=observer_payload,
        subjective_cameras=subjective_cameras,
    )
    hilbert_algebra_payload = _hilbert_space_observer_algebra_payload(observer_payload)
    observer_anatomy_payload = _observer_anatomy_payload(
        observer_payload=observer_payload,
        subjective_cameras=subjective_cameras,
        hilbert_algebra_payload=hilbert_algebra_payload,
    )
    paper_accuracy_payload = _paper_accuracy_payload(
        small_payload=small_payload,
        observer_payload=observer_payload,
        bulk_payload=bulk_payload,
        cmb_payload=cmb_payload,
        curved_spacetime_payload=emergent_curved_spacetime_payload,
        visualization_views=visualization_views,
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
        "visualizationRenderData": visualization_render_data,
        "effectiveStringTheory": effective_string_payload,
        "emergentCurvedSpacetime": emergent_curved_spacetime_payload,
        "observerCinema": observer_cinema_payload,
        "hilbertSpaceObserverAlgebra": hilbert_algebra_payload,
        "observerAnatomy": observer_anatomy_payload,
        "paperAccuracy": paper_accuracy_payload,
    }


def _paper_accuracy_payload(
    *,
    small_payload: dict[str, Any],
    observer_payload: dict[str, Any],
    bulk_payload: dict[str, Any],
    cmb_payload: dict[str, Any],
    curved_spacetime_payload: dict[str, Any],
    visualization_views: dict[str, Any],
) -> dict[str, Any]:
    small_receipts = small_payload.get("receipts", {}) if isinstance(small_payload.get("receipts"), dict) else {}
    observer_receipts = (
        observer_payload.get("receipts", {}) if isinstance(observer_payload.get("receipts"), dict) else {}
    )
    bulk_receipts = bulk_payload.get("receipts", {}) if isinstance(bulk_payload.get("receipts"), dict) else {}
    proto = (
        bulk_payload.get("protoParticleCandidates", {})
        if isinstance(bulk_payload.get("protoParticleCandidates"), dict)
        else {}
    )
    proto_receipts = proto.get("receipts", {}) if isinstance(proto.get("receipts"), dict) else {}
    cmb_receipts = cmb_payload.get("receipts", {}) if isinstance(cmb_payload.get("receipts"), dict) else {}
    curved_receipts = (
        curved_spacetime_payload.get("receipts", {})
        if isinstance(curved_spacetime_payload.get("receipts"), dict)
        else {}
    )
    vacuum_view = (
        visualization_views.get("fluctuatingQuantumVacuum", {})
        if isinstance(visualization_views.get("fluctuatingQuantumVacuum"), dict)
        else {}
    )
    yang_mills = (
        vacuum_view.get("yangMillsGapCertificate", {})
        if isinstance(vacuum_view.get("yangMillsGapCertificate"), dict)
        else {}
    )
    ym_receipts = yang_mills.get("receipts", {}) if isinstance(yang_mills.get("receipts"), dict) else {}
    checks = [
        {
            "id": "finite_overlap_repair",
            "payloadPath": "smallUniverse",
            "paperStatus": "finite overlap-repair certificate/theorem-core diagnostic",
            "receipt": "FINITE_CONSENSUS_THEOREM_RECEIPT",
            "passed": bool(small_receipts.get("FINITE_CONSENSUS_THEOREM_RECEIPT", False)),
            "allowedClaim": "finite observer-patch repair/readback consistency",
            "notAllowedClaim": "continuum spacetime or physical particle dynamics by itself",
        },
        {
            "id": "observer_modular_time",
            "payloadPath": "observerModularTime",
            "paperStatus": "observer-local modular readout",
            "receipt": "observer_modular_time_receipt",
            "passed": bool(observer_receipts.get("observer_modular_time_receipt", False)),
            "allowedClaim": "observer-local time/readout frames from visible support records",
            "notAllowedClaim": "external global time or hidden omniscient camera",
        },
        {
            "id": "observer_facing_consensus_3d_bulk",
            "payloadPath": "consensusBulk.objects",
            "paperStatus": "observer-facing H3 consensus chart",
            "receipt": "observer_facing_consensus_3d_bulk_readout_receipt",
            "passed": bool(
                bulk_receipts.get(
                    "observer_facing_consensus_3d_bulk_readout_receipt",
                    bulk_receipts.get("theorem_assisted_consensus_3d_bulk_readout_receipt", False),
                )
            ),
            "allowedClaim": "consensus object packets in the derived observer-facing H3 chart",
            "notAllowedClaim": "chart-blind strict neutral third-person bulk unless its receipt also passes",
        },
        {
            "id": "strict_neutral_bulk",
            "payloadPath": "consensusBulk.receipts",
            "paperStatus": "promotion gate",
            "receipt": "strict_neutral_third_person_bulk_receipt",
            "passed": bool(bulk_receipts.get("strict_neutral_third_person_bulk_receipt", False)),
            "allowedClaim": "strict neutral quotient bulk only if true",
            "notAllowedClaim": "neutral 3D bulk from a merely observer-facing H3 chart",
        },
        {
            "id": "proto_worldlines",
            "payloadPath": "consensusBulk.protoParticleCandidates.worldlines",
            "paperStatus": "H3 record-worldline continuation diagnostic",
            "receipt": "bulk_worldline_precursor_receipt",
            "passed": bool(proto_receipts.get("bulk_worldline_precursor_receipt", False)),
            "allowedClaim": "holonomy/proto-worldline candidate motion in the derived chart",
            "notAllowedClaim": "matter particles unless particle_matter_receipt passes",
        },
        {
            "id": "particle_matter",
            "payloadPath": "consensusBulk.protoParticleCandidates.receipts",
            "paperStatus": "promotion gate",
            "receipt": "particle_matter_receipt",
            "passed": bool(proto_receipts.get("particle_matter_receipt", False)),
            "allowedClaim": "particle matter only if true",
            "notAllowedClaim": "defect glyphs as physical particles without the matter bridge",
        },
        {
            "id": "curved_spacetime_compaction",
            "payloadPath": "emergentCurvedSpacetime",
            "paperStatus": "Einstein-branch visualization diagnostic",
            "receipt": "emergent_curved_spacetime_visualization_receipt",
            "passed": bool(curved_receipts.get("emergent_curved_spacetime_visualization_receipt", False)),
            "allowedClaim": (
                "quotient-visible source/readout rows drive an H3 Green-potential compactification "
                "display over the observer-facing chart"
            ),
            "notAllowedClaim": "solved Einstein metric, production gravity, or measured gravitational attraction",
        },
        {
            "id": "einstein_branch_entry",
            "payloadPath": "emergentCurvedSpacetime.einsteinBranchEntry",
            "paperStatus": "E0 OPH5 recovered-core bridge manifest and theorem-tagged sidecar receipts",
            "receipt": "EINSTEIN_BRANCH_ENTRY_RECEIPT",
            "passed": bool(
                curved_receipts.get(
                    "einstein_branch_entry_receipt",
                    curved_receipts.get("EINSTEIN_BRANCH_ENTRY_RECEIPT", False),
                )
            ),
            "allowedClaim": (
                "E0 branch-entry sidecar receipts have supplied the required OPH5 bridge data only if true"
            ),
            "notAllowedClaim": (
                "Einstein equations or production gravity as an unconditional consequence of finite consensus"
            ),
            "issue": 503,
            "issueUrl": "https://github.com/FloatingPragma/observer-patch-holography/issues/503",
            "blockers": (
                curved_spacetime_payload.get("einsteinBranchEntry", {}).get("blockers", [])
                if isinstance(curved_spacetime_payload.get("einsteinBranchEntry"), dict)
                else []
            ),
        },
        {
            "id": "production_gravity",
            "payloadPath": "emergentCurvedSpacetime.receipts",
            "paperStatus": "promotion gate",
            "receipt": "production_gravity_receipt",
            "passed": bool(curved_receipts.get("production_gravity_receipt", False)),
            "allowedClaim": "production gravity only if true",
            "notAllowedClaim": "gravity automatically promoted from diagnostic compaction visuals",
        },
        {
            "id": "cmb_comparison",
            "payloadPath": "cmbComparison",
            "paperStatus": "measurement-comparable screen-spectrum diagnostic",
            "receipt": "USABLE_PHYSICAL_CMB_DATA_RECEIPT",
            "passed": bool(cmb_receipts.get("USABLE_PHYSICAL_CMB_DATA_RECEIPT", False)),
            "allowedClaim": "diagnostic comparison data are usable for visualization/stats if true",
            "notAllowedClaim": "physical TT/TE/EE prediction without frozen source/transfer/likelihood gates",
        },
        {
            "id": "physical_cmb_prediction",
            "payloadPath": "cmbComparison.receipts",
            "paperStatus": "promotion gate",
            "receipt": "PHYSICAL_CMB_PREDICTION_RECEIPT",
            "passed": bool(cmb_receipts.get("PHYSICAL_CMB_PREDICTION_RECEIPT", False)),
            "allowedClaim": "physical CMB prediction only if true",
            "notAllowedClaim": "shape match as proof of physical prediction when the gate is closed",
        },
        {
            "id": "finite_yang_mills_gap_diagnostic",
            "payloadPath": "visualizationViews.fluctuatingQuantumVacuum.yangMillsGapCertificate",
            "paperStatus": "finite compact-gauge diagnostic",
            "receipt": "finite_nonabelian_gauge_gap_diagnostic_receipt",
            "passed": bool(ym_receipts.get("finite_nonabelian_gauge_gap_diagnostic_receipt", False)),
            "allowedClaim": "finite SU(2) Wilson-lattice diagnostic and transfer-gap proxy",
            "notAllowedClaim": "Clay Yang-Mills mass gap reproduction",
        },
        {
            "id": "clay_yang_mills_gap",
            "payloadPath": "visualizationViews.fluctuatingQuantumVacuum.yangMillsGapCertificate.receipts",
            "paperStatus": "promotion gate",
            "receipt": "CLAY_YANG_MILLS_GAP_RECEIPT",
            "passed": bool(ym_receipts.get("CLAY_YANG_MILLS_GAP_RECEIPT", False)),
            "allowedClaim": "Clay Yang-Mills gap only if true",
            "notAllowedClaim": "continuum theorem from finite lattice diagnostics",
        },
    ]
    check_by_id = {str(check["id"]): check for check in checks}
    return {
        "schema": "oph_paper_accuracy_guard_v1",
        "mode": "finite_OPH_diagnostic_fail_closed",
        "sourceDocuments": [
            "markdown/recovering_relativity_and_standard_model_structure_from_observer_overlap_consistency_compact.md",
            "markdown/screen_microphysics_and_observer_synchronization.md",
            "markdown/reality_as_consensus_protocol.md",
        ],
        "checks": checks,
        "receipts": {
            "paper_accuracy_guard_receipt": True,
            "no_semantic_promotion_by_relabeling_receipt": True,
            "observer_facing_consensus_3d_bulk_readout_receipt": bool(
                check_by_id["observer_facing_consensus_3d_bulk"]["passed"]
            ),
            "strict_neutral_third_person_bulk_receipt": bool(check_by_id["strict_neutral_bulk"]["passed"]),
            "particle_matter_receipt": bool(check_by_id["particle_matter"]["passed"]),
            "einstein_branch_entry_receipt": bool(check_by_id["einstein_branch_entry"]["passed"]),
            "EINSTEIN_BRANCH_ENTRY_RECEIPT": bool(check_by_id["einstein_branch_entry"]["passed"]),
            "OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1": bool(
                check_by_id["einstein_branch_entry"]["passed"]
            ),
            "production_gravity_receipt": bool(check_by_id["production_gravity"]["passed"]),
            "physical_gravity_prediction": bool(
                curved_receipts.get("physical_gravity_prediction", False)
            ),
            "PHYSICAL_CMB_PREDICTION_RECEIPT": bool(check_by_id["physical_cmb_prediction"]["passed"]),
            "CLAY_YANG_MILLS_GAP_RECEIPT": bool(check_by_id["clay_yang_mills_gap"]["passed"]),
        },
        "failClosedPromotionReceipts": [
            "strict_neutral_third_person_bulk_receipt",
            "particle_matter_receipt",
            "einstein_branch_entry_receipt",
            "EINSTEIN_BRANCH_ENTRY_RECEIPT",
            "OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1",
            "production_gravity_receipt",
            "einstein_equation_solution_receipt",
            "PHYSICAL_CMB_PREDICTION_RECEIPT",
            "CLAY_YANG_MILLS_GAP_RECEIPT",
        ],
        "paperAccuracyRule": (
            "Renderer-visible fields may show finite OPH diagnostics only at the status their receipts "
            "support. Aesthetic geometry, apparent attraction, CMB-shape similarity, or finite lattice "
            "gaps must not be relabeled as physical predictions without the named promotion receipts."
        ),
        "claimBoundary": (
            "Paper-accuracy guard for the visualization bundle. It keeps the finite simulator aligned "
            "with theorem, continuation, and promotion-gate status in the OPH papers."
        ),
    }


def _small_universe_payload(run_dir: Path) -> dict[str, Any]:
    exact = _read_json(run_dir / "exact_consensus_receipt.json")
    frustrated = _read_json(run_dir / "frustrated_control_receipt.json")
    evidence = _read_json(run_dir / "small_oph_universe_evidence.json")
    theorem_core = _read_json(run_dir / "theorem_core_receipts.json")
    finite_replay = _read_json(run_dir / "finite_consensus_replay_report.json")
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
                or theorem_core.get("FINITE_CONSENSUS_THEOREM_RECEIPT", False)
                or theorem_core.get("finite_consensus_theorem_receipt", False)
                or finite_replay.get("receipt", False)
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
            "bundle_receipt": bool(evidence.get("bundle_receipt", False) or theorem_core.get("receipt", False)),
        },
        "claimBoundary": exact.get(
            "claim_boundary",
            theorem_core.get(
                "claim_boundary",
                "Finite overlap-repair receipt/readout. Large-run fallbacks use theorem-core receipts "
                "when exact mini-universe files are absent; this is not by itself a Lorentz, H3, particle, "
                "or cosmology claim.",
            ),
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
    max_objective_observer_views: int | None,
) -> dict[str, Any]:
    observer_report = _read_json(run_dir / "observer_modular_experience_report.json")
    status = _read_json(run_dir / "emergence_status_report.json")
    consensus = _read_json(run_dir / "observer_consensus_report.json")
    readout = _read_json((readout_dir or run_dir) / "observer_consensus_bulk_readout_report.json")
    views = _read_jsonl(run_dir / "observer_views.jsonl", limit=max_observers)
    trace = _read_trace(run_dir / "mismatch_trace.csv")
    if not views:
        views = _fallback_observer_views(run_dir, observer_report, trace, max_observers=max_observers)
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
    objective_limit = int(max_observers if max_objective_observer_views is None else max_objective_observer_views)
    objective_limit = max(0, min(int(max_observers), objective_limit))
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
            "observerCount": consensus.get("observer_count", observer_report.get("observer_count", len(observers))),
            "pairCount": consensus.get("pair_count"),
            "exportedPairCount": len(overlap_links),
            "exportedObserverCount": len(observers),
            "exportedObjectiveObserverViewCount": objective_limit,
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


def _fallback_observer_views(
    run_dir: Path,
    observer_report: dict[str, Any],
    trace: list[dict[str, Any]],
    *,
    max_observers: int,
) -> list[dict[str, Any]]:
    """Synthesize renderer-safe observer-local rows when compact runs omit JSONL views."""

    reported_count = _safe_int(
        observer_report.get("observer_count", observer_report.get("objective_observer_view_count", 0))
    )
    count = min(max(0, reported_count), max(0, int(max_observers)))
    if count <= 0:
        return []
    patch_count = _screen_patch_count_from_run(run_dir, default=max(count * 4, 1))
    support_count = max(1, min(patch_count, int(round(math.sqrt(max(patch_count, 1)))) * 2))
    stride = max(1, support_count // 2)
    axes = fibonacci_sphere_points(count)
    time_grid = observer_report.get("observer_relative_time_grid")
    if not isinstance(time_grid, list) or not time_grid:
        frame_count = max(32, len(trace), 1)
        time_grid = [float(index / max(frame_count - 1, 1)) for index in range(frame_count)]
    rows = []
    for index, axis in enumerate(axes):
        support_nodes = [
            int((index * stride + offset) % max(patch_count, 1))
            for offset in range(support_count)
        ]
        record_packet = f"record:{(index * 2654435761) % 1024}"
        object_packet = f"support:{index % max(1, min(count, 64))}"
        rows.append(
            {
                "view_type": "patch_observer",
                "observer_id": index,
                "axis": [float(axis[0]), float(axis[1]), float(axis[2])],
                "support_nodes": support_nodes,
                "support_patch_count": len(support_nodes),
                "support_entropy_capacity": float(math.log2(max(len(support_nodes), 1) + 1.0)),
                "visible_signature_entropy": float(math.log2((index % 7) + 2)),
                "modular_depth_mean": float((index + 1) / max(count, 1)),
                "dominant_record_signature": record_packet,
                "dominant_object_packet": object_packet,
                "object_packet_histogram": {
                    object_packet: 0.7,
                    f"neighbor:{(index + 1) % max(count, 1)}": 0.3,
                },
                "record_signature_histogram": {
                    record_packet: 0.75,
                    f"record:{(index + 1) % 1024}": 0.25,
                },
                "observer_relative_times": list(time_grid),
                "transition_history_descriptor": {
                    "steps": [
                        {
                            "record_family": index % 17,
                            "checkpoint_class": step % 4,
                            "s3_sector_class": (index + step) % 8,
                        }
                        for step in range(min(8, max(1, len(time_grid))))
                    ]
                },
                "visible_readout_hash": f"fallback_observer_{index:06d}",
                "claim_boundary": (
                    "Compact-run fallback observer-local readout synthesized from observer_count, "
                    "screen patch count, and the global repair trace because observer_views.jsonl was "
                    "not persisted. It is sufficient for visualization and algebra occupancy only; it "
                    "does not add hidden representatives or upgrade any receipt."
                ),
            }
        )
    return rows


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


def _subjective_observer_cameras(
    observer_payload: dict[str, Any],
    *,
    bulk_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cameras: list[dict[str, Any]] = []
    perspectives = observer_payload.get("objectiveObserverViews", [])
    if not isinstance(perspectives, list):
        return cameras
    proto_worldlines = _proto_worldline_visibility_index(bulk_payload or {})
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
        camera_forward = [-float(value) for value in forward]
        fov_degrees = 72.0
        frames = []
        for frame in list(row.get("timeFrames", []))[:96] if isinstance(row.get("timeFrames"), list) else []:
            if not isinstance(frame, dict):
                continue
            visible_proto_worldlines = _visible_proto_worldline_sightings(
                proto_worldlines,
                cycle=frame.get("cycle"),
                relative_time=frame.get("relativeTime"),
                eye=eye,
                forward=camera_forward,
                right=right,
                up=up,
                fov_degrees=fov_degrees,
            )
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
                    "visibleProtoWorldlines": visible_proto_worldlines,
                    "visibleProtoWorldlineCount": len(visible_proto_worldlines),
                    "protoWorldlineSightingSource": (
                        "observer_camera_h3_cone_projection_from_consensusBulk.protoParticleCandidates"
                    ),
                }
            )
        visible_proto_ids = sorted(
            {
                str(sighting.get("worldlineId"))
                for frame in frames
                for sighting in frame.get("visibleProtoWorldlines", [])
                if isinstance(sighting, dict) and sighting.get("worldlineId") is not None
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
                "forward": camera_forward,
                "fovDegrees": fov_degrees,
                "screenProjection": "local_tangent_readout_camera_on_s2_boundary",
                "supportPatchCount": row.get("supportPatchCount"),
                "supportNodeSample": row.get("supportNodeSample", []),
                "supportEntropyCapacity": row.get("supportEntropyCapacity"),
                "visibleObjectPackets": row.get("visibleObjectPackets", []),
                "visibleRecordPackets": row.get("visibleRecordPackets", []),
                "visibleProtoWorldlineIds": visible_proto_ids,
                "visibleProtoWorldlineSightingCount": sum(
                    len(frame.get("visibleProtoWorldlines", [])) for frame in frames if isinstance(frame, dict)
                ),
                "timeFrames": frames,
                "claimBoundary": (
                    "Subjective observer camera derived from one observer-local visible readout. "
                    "It is a rendering camera for the observer's finite support, not a hidden global view. "
                    "visibleProtoWorldlines are diagnostic H3-cone projections of exported proto-worldline "
                    "events into this camera; they do not promote particle-matter receipts."
                ),
            }
        )
    return cameras


def _proto_worldline_visibility_index(bulk_payload: dict[str, Any]) -> list[dict[str, Any]]:
    proto = (
        bulk_payload.get("protoParticleCandidates", {})
        if isinstance(bulk_payload.get("protoParticleCandidates"), dict)
        else {}
    )
    worldlines = proto.get("worldlines") if isinstance(proto.get("worldlines"), list) else []
    indexed: list[dict[str, Any]] = []
    for row in worldlines:
        if not isinstance(row, dict):
            continue
        events = []
        for event in row.get("events", []) if isinstance(row.get("events"), list) else []:
            if not isinstance(event, dict):
                continue
            point = _coord3(event.get("h3SpatialPoint") or event.get("h3_spatial_point"))
            cycle = _optional_float(event.get("cycle"))
            if point is None or cycle is None:
                continue
            events.append(
                {
                    "cycle": cycle,
                    "point": point,
                    "event": event.get("event"),
                    "class": event.get("class"),
                    "holonomyMode": event.get("holonomyMode", event.get("holonomy_mode")),
                    "fitResidual": event.get("fitResidual", event.get("fit_residual")),
                    "supportNodeCount": event.get("supportNodeCount", event.get("support_node_count")),
                    "velocity": event.get("velocity"),
                    "pairH3Separation": event.get("pairH3Separation", event.get("pair_h3_separation")),
                    "supportOverlapFraction": event.get(
                        "supportOverlapFraction", event.get("support_overlap_fraction")
                    ),
                    "supportOverlapNodeCount": event.get(
                        "supportOverlapNodeCount", event.get("support_overlap_node_count")
                    ),
                    "contactOutcome": event.get("contactOutcome", event.get("contact_outcome")),
                    "chargeConservationPass": event.get(
                        "chargeConservationPass", event.get("charge_conservation_pass")
                    ),
                }
            )
        events.sort(key=lambda item: float(item["cycle"]))
        if not events:
            continue
        indexed.append(
            {
                "worldlineId": row.get("worldlineId") or row.get("worldline_id"),
                "events": events,
                "birthCycle": _optional_float(row.get("birthCycle", row.get("birth_cycle"))),
                "deathCycle": _optional_float(row.get("deathCycle", row.get("death_cycle"))),
                "particleLike": bool(row.get("particleLike", row.get("particle_like", False))),
                "diagnosticOnly": bool(row.get("diagnosticOnly", row.get("diagnostic_only", True))),
                "controlledPlantedAssay": bool(row.get("controlledPlantedAssay", False)),
                "freeDynamicsDiagnostic": bool(row.get("freeDynamicsDiagnostic", False)),
                "contactOutcome": row.get("contactOutcome", row.get("contact_outcome")),
                "bulkLocalizationPass": bool(row.get("bulkLocalizationPass", row.get("bulk_localization_pass", False))),
                "localizationPass": bool(row.get("localizationPass", row.get("localization_pass", False))),
                "persistencePass": bool(row.get("persistencePass", row.get("persistence_pass", False))),
                "transportabilityPass": bool(row.get("transportabilityPass", row.get("transportability_pass", False))),
                "classMode": row.get("classMode", row.get("class_mode")),
                "worldlineSource": row.get("worldlineSource", proto.get("worldlineSource")),
                "claimBoundary": row.get(
                    "claimBoundary",
                    proto.get(
                        "claimBoundary",
                        "Observer-visible proto-worldline projection is diagnostic unless particle receipts pass.",
                    ),
                ),
            }
        )
    return indexed


def _visible_proto_worldline_sightings(
    proto_worldlines: list[dict[str, Any]],
    *,
    cycle: Any,
    relative_time: Any,
    eye: list[float],
    forward: list[float],
    right: list[float],
    up: list[float],
    fov_degrees: float,
    max_sightings: int = 16,
) -> list[dict[str, Any]]:
    if not proto_worldlines:
        return []
    frame_cycle = _optional_float(cycle)
    if frame_cycle is None:
        tau = _optional_float(relative_time)
        if tau is None:
            return []
        min_cycle = min(float(row["events"][0]["cycle"]) for row in proto_worldlines if row.get("events"))
        max_cycle = max(float(row["events"][-1]["cycle"]) for row in proto_worldlines if row.get("events"))
        frame_cycle = min_cycle + max(0.0, min(1.0, tau)) * (max_cycle - min_cycle)
    sightings = []
    for worldline in proto_worldlines:
        sample = _interpolate_proto_worldline_event(worldline, frame_cycle)
        if sample is None:
            continue
        projection = _project_h3_point_to_observer_camera(
            sample["point"],
            eye=eye,
            forward=forward,
            right=right,
            up=up,
            fov_degrees=fov_degrees,
        )
        projection_mode = "nominal_fov_cone"
        outside_nominal_fov = False
        if projection is None:
            projection = _project_h3_point_to_observer_directional_readout(
                sample["point"],
                eye=eye,
                forward=forward,
                right=right,
                up=up,
                fov_degrees=fov_degrees,
            )
            projection_mode = "wide_angle_directional_fallback"
            outside_nominal_fov = True
        if projection is None:
            continue
        readout = {
            "u": projection["screenX"],
            "v": projection["screenY"],
            "range": projection["distance"],
            "rangeBucket": _observer_local_range_bucket(projection["distance"]),
            "angularSeparationDegrees": projection["angularSeparationDegrees"],
            "visibilityScore": projection["visibilityScore"],
            "projectionMode": projection_mode,
            "outsideNominalFov": outside_nominal_fov,
            "nominalFovDegrees": fov_degrees,
            "coordinateSystem": "observer_local_tangent_screen_readout",
            "sourceCoordinateSuppressed": "global_h3_spatial_point",
            "hiddenGlobalH3Suppressed": True,
        }
        sightings.append(
            {
                "worldlineId": worldline.get("worldlineId"),
                "cycle": frame_cycle,
                "nearestEventCycle": sample.get("nearestEventCycle"),
                "cycleDistance": sample.get("cycleDistance"),
                "interpolated": bool(sample.get("interpolated", False)),
                "observerLocalReadout": readout,
                "screenX": projection["screenX"],
                "screenY": projection["screenY"],
                "cameraDistance": projection["distance"],
                "angularSeparationDegrees": projection["angularSeparationDegrees"],
                "visibilityScore": projection["visibilityScore"],
                "projectionMode": projection_mode,
                "outsideNominalFov": outside_nominal_fov,
                "event": sample.get("event"),
                "class": sample.get("class"),
                "holonomyMode": sample.get("holonomyMode"),
                "fitResidual": sample.get("fitResidual"),
                "supportNodeCount": sample.get("supportNodeCount"),
                "pairH3Separation": sample.get("pairH3Separation"),
                "supportOverlapFraction": sample.get("supportOverlapFraction"),
                "supportOverlapNodeCount": sample.get("supportOverlapNodeCount"),
                "contactOutcome": sample.get("contactOutcome", worldline.get("contactOutcome")),
                "chargeConservationPass": sample.get("chargeConservationPass"),
                "particleLike": bool(worldline.get("particleLike", False)),
                "diagnosticOnly": bool(worldline.get("diagnosticOnly", True)),
                "controlledPlantedAssay": bool(worldline.get("controlledPlantedAssay", False)),
                "organicDefectPopulationDiagnostic": bool(
                    worldline.get("organicDefectPopulationDiagnostic", False)
                ),
                "freeDynamicsDiagnostic": bool(worldline.get("freeDynamicsDiagnostic", False)),
                "bulkLocalizationPass": bool(worldline.get("bulkLocalizationPass", False)),
                "localizationPass": bool(worldline.get("localizationPass", False)),
                "persistencePass": bool(worldline.get("persistencePass", False)),
                "transportabilityPass": bool(worldline.get("transportabilityPass", False)),
                "worldlineSource": worldline.get("worldlineSource"),
                "claimBoundary": worldline.get("claimBoundary"),
            }
        )
    sightings.sort(
        key=lambda item: (
            bool(item.get("outsideNominalFov", False)),
            -float(item.get("visibilityScore") or 0.0),
            float(item.get("cameraDistance") or 0.0),
            str(item.get("worldlineId")),
        )
    )
    return sightings[:max_sightings]


def _project_h3_point_to_observer_directional_readout(
    point: list[float],
    *,
    eye: list[float],
    forward: list[float],
    right: list[float],
    up: list[float],
    fov_degrees: float,
) -> dict[str, float] | None:
    ray = [float(point[index]) - float(eye[index]) for index in range(3)]
    distance = _vec_norm(ray)
    if distance <= 1e-12:
        return None
    direction = [value / distance for value in ray]
    forward_unit = _unit_vec(forward)
    right_unit = _unit_vec(right)
    up_unit = _unit_vec(up)
    cos_angle = max(-1.0, min(1.0, _dot(direction, forward_unit)))
    angular = math.acos(cos_angle)
    raw_x = _dot(direction, right_unit)
    raw_y = _dot(direction, up_unit)
    scale = max(1.0, abs(raw_x), abs(raw_y))
    half_fov = math.radians(max(1.0, min(170.0, float(fov_degrees))) / 2.0)
    return {
        "screenX": float(max(-1.0, min(1.0, raw_x / scale))),
        "screenY": float(max(-1.0, min(1.0, raw_y / scale))),
        "distance": float(distance),
        "angularSeparationDegrees": float(math.degrees(angular)),
        "visibilityScore": float(max(0.05, 1.0 - angular / math.pi)),
        "outsideNominalFov": bool(angular > half_fov),
    }


def _interpolate_proto_worldline_event(worldline: dict[str, Any], cycle: float) -> dict[str, Any] | None:
    events = worldline.get("events") if isinstance(worldline.get("events"), list) else []
    if not events:
        return None
    first_cycle = float(events[0]["cycle"])
    last_cycle = float(events[-1]["cycle"])
    tolerance = max(1e-9, 0.05 * max(last_cycle - first_cycle, 1.0))
    if cycle < first_cycle - tolerance or cycle > last_cycle + tolerance:
        return None
    if cycle <= first_cycle:
        event = events[0]
        return _proto_event_sample(event, event, alpha=0.0, cycle=cycle)
    if cycle >= last_cycle:
        event = events[-1]
        return _proto_event_sample(event, event, alpha=0.0, cycle=cycle)
    for left, right in zip(events, events[1:], strict=False):
        left_cycle = float(left["cycle"])
        right_cycle = float(right["cycle"])
        if left_cycle <= cycle <= right_cycle:
            span = max(right_cycle - left_cycle, 1e-12)
            alpha = max(0.0, min(1.0, (cycle - left_cycle) / span))
            return _proto_event_sample(left, right, alpha=alpha, cycle=cycle)
    event = min(events, key=lambda row: abs(float(row["cycle"]) - cycle))
    return _proto_event_sample(event, event, alpha=0.0, cycle=cycle)


def _proto_event_sample(left: dict[str, Any], right: dict[str, Any], *, alpha: float, cycle: float) -> dict[str, Any]:
    left_point = _coord3(left.get("point")) or [0.0, 0.0, 0.0]
    right_point = _coord3(right.get("point")) or left_point
    point = [
        float((1.0 - alpha) * left_point[index] + alpha * right_point[index])
        for index in range(3)
    ]
    nearest = left if abs(float(left["cycle"]) - cycle) <= abs(float(right["cycle"]) - cycle) else right
    return {
        "point": point,
        "nearestEventCycle": nearest.get("cycle"),
        "cycleDistance": abs(float(nearest.get("cycle", cycle)) - cycle),
        "interpolated": bool(alpha > 1e-9 and alpha < 1.0 - 1e-9),
        "event": nearest.get("event"),
        "class": nearest.get("class"),
        "holonomyMode": nearest.get("holonomyMode"),
        "fitResidual": nearest.get("fitResidual"),
        "supportNodeCount": nearest.get("supportNodeCount"),
        "velocity": nearest.get("velocity"),
        "pairH3Separation": nearest.get("pairH3Separation"),
        "supportOverlapFraction": nearest.get("supportOverlapFraction"),
        "supportOverlapNodeCount": nearest.get("supportOverlapNodeCount"),
        "contactOutcome": nearest.get("contactOutcome"),
        "chargeConservationPass": nearest.get("chargeConservationPass"),
    }


def _project_h3_point_to_observer_camera(
    point: list[float],
    *,
    eye: list[float],
    forward: list[float],
    right: list[float],
    up: list[float],
    fov_degrees: float,
) -> dict[str, float] | None:
    ray = [float(point[index]) - float(eye[index]) for index in range(3)]
    distance = _vec_norm(ray)
    if distance <= 1e-12:
        return None
    direction = [value / distance for value in ray]
    forward_unit = _unit_vec(forward)
    right_unit = _unit_vec(right)
    up_unit = _unit_vec(up)
    cos_angle = max(-1.0, min(1.0, _dot(direction, forward_unit)))
    if cos_angle <= 0.0:
        return None
    half_fov = math.radians(max(1.0, min(170.0, float(fov_degrees))) / 2.0)
    tan_half = math.tan(half_fov)
    screen_x = _dot(direction, right_unit) / max(cos_angle * tan_half, 1e-12)
    screen_y = _dot(direction, up_unit) / max(cos_angle * tan_half, 1e-12)
    angular = math.acos(cos_angle)
    if angular > half_fov or abs(screen_x) > 1.0 or abs(screen_y) > 1.0:
        return None
    visibility = max(0.0, min(1.0, 1.0 - angular / max(half_fov, 1e-12)))
    return {
        "screenX": float(screen_x),
        "screenY": float(screen_y),
        "distance": float(distance),
        "angularSeparationDegrees": float(math.degrees(angular)),
        "visibilityScore": float(visibility),
    }


def _observer_local_range_bucket(distance: float) -> str:
    value = max(0.0, float(distance))
    if value < 0.75:
        return "near"
    if value < 1.5:
        return "mid"
    return "far"


def _effective_string_theory_payload(
    *,
    visualization_views: dict[str, Any],
    small_payload: dict[str, Any],
    bulk_payload: dict[str, Any],
) -> dict[str, Any]:
    view = (
        visualization_views.get("effectiveStringTheory", {})
        if isinstance(visualization_views.get("effectiveStringTheory"), dict)
        else {}
    )
    proto = (
        bulk_payload.get("protoParticleCandidates", {})
        if isinstance(bulk_payload.get("protoParticleCandidates"), dict)
        else {}
    )
    worldlines = proto.get("worldlines") if isinstance(proto.get("worldlines"), list) else []
    repair_frames = small_payload.get("repairFrames") if isinstance(small_payload.get("repairFrames"), list) else []
    cycles = small_payload.get("cycles") if isinstance(small_payload.get("cycles"), dict) else {}
    stress = (
        view.get("twoDefectStressContractionAssay", {})
        if isinstance(view.get("twoDefectStressContractionAssay"), dict)
        else {}
    )
    free_dynamics = (
        view.get("freeTwoDefectDynamics", {})
        if isinstance(view.get("freeTwoDefectDynamics"), dict)
        else {}
    )
    organic_defects = (
        view.get("organicDefectPopulation", {})
        if isinstance(view.get("organicDefectPopulation"), dict)
        else {}
    )
    string_selector = (
        view.get("stringVacuumSelector", {})
        if isinstance(view.get("stringVacuumSelector"), dict)
        else {}
    )
    vibration_samples = _finite_edge_string_vibration_samples(cycles, repair_frames)
    return {
        "schema": "oph_effective_string_theory_visualization_v1",
        "viewId": "effectiveStringTheory",
        "label": view.get("label", "Effective string-theory edge/worldsheet view"),
        "sectionKind": view.get("sectionKind", "effective_string_theory_edge_worldsheet"),
        "description": view.get("description"),
        "contentAvailable": bool(
            cycles or repair_frames or worldlines or stress or organic_defects or free_dynamics or string_selector
        ),
        "viewContract": view,
        "finiteEdgeCycles": cycles,
        "finiteRepairWorldsheetFrames": repair_frames[:256],
        "finiteEdgeStringVibrationSamples": vibration_samples,
        "finiteEdgeStringVibrationReceipt": bool(vibration_samples),
        "finiteEdgeStringVibrationClaimBoundary": (
            "Exact finite repair/cycle edge-pulse samples over the OPH carrier. These are the most direct "
            "available string-vibration inputs for the visualizer, but they are not critical-string normal "
            "modes or physical string oscillations without the critical-edge receipt suite."
        ),
        "h3ProtoParticleWorldlines": worldlines,
        "twoDefectStressContractionAssay": stress,
        "organicDefectPopulation": organic_defects,
        "freeTwoDefectDynamics": free_dynamics,
        "stringVacuumSelector": string_selector,
        "renderLayers": view.get("renderLayers", []),
        "animationChannels": view.get("animationChannels", []),
        "receipts": view.get("receipts", {}),
        "nonClaims": view.get("nonClaims", []),
        "claimBoundary": view.get(
            "claimBoundary",
            "Effective edge-string diagnostic data only; not a critical string CFT receipt.",
        ),
    }


def _finite_edge_string_vibration_samples(
    cycles: dict[str, Any],
    repair_frames: list[dict[str, Any]],
    *,
    max_cycles: int = 64,
    max_frames: int = 256,
    max_samples: int = 8192,
) -> list[dict[str, Any]]:
    if not isinstance(cycles, dict) or not repair_frames:
        return []
    rows: list[dict[str, Any]] = []
    cycle_sources = (
        ("exact_consensus", cycles.get("exactConsensus", [])),
        ("frustrated_control", cycles.get("frustratedControl", [])),
    )
    sample_index = 0
    for cycle_kind, source_rows in cycle_sources:
        if not isinstance(source_rows, list):
            continue
        for cycle_index, cycle_row in enumerate(source_rows[:max_cycles]):
            if not isinstance(cycle_row, dict):
                continue
            edges = cycle_row.get("edges", [])
            if not isinstance(edges, list) or not edges:
                continue
            cycle_nodes = cycle_row.get("cycle", []) if isinstance(cycle_row.get("cycle"), list) else []
            normalized_edges = [_edge_pair(edge) for edge in edges]
            edge_lookup = {
                tuple(sorted(edge)): index
                for index, edge in enumerate(normalized_edges)
                if edge is not None
            }
            for frame in repair_frames[:max_frames]:
                if not isinstance(frame, dict):
                    continue
                active_edge = _repair_active_edge(frame)
                active_edge_key = tuple(sorted(active_edge)) if active_edge is not None else None
                active_index = edge_lookup.get(active_edge_key) if active_edge_key is not None else None
                state = frame.get("state", []) if isinstance(frame.get("state"), list) else []
                occupation = _cycle_state_occupation(cycle_nodes, state)
                delta_phi = _optional_float(frame.get("deltaPhi"))
                raw_amplitude = abs(delta_phi) if delta_phi is not None else (1.0 if active_index is not None else 0.0)
                rows.append(
                    {
                        "sampleIndex": sample_index,
                        "cycleKind": cycle_kind,
                        "cycleIndex": cycle_index,
                        "frameStep": frame.get("step"),
                        "repairNode": frame.get("node"),
                        "repairParent": frame.get("parent"),
                        "edgeIndex": active_index,
                        "edge": list(active_edge) if active_edge is not None else None,
                        "loopPhase": (
                            float(active_index / max(1, len(normalized_edges)))
                            if active_index is not None
                            else None
                        ),
                        "edgePulse": bool(active_index is not None),
                        "stateOccupation": occupation,
                        "deltaPhi": delta_phi,
                        "rawAmplitude": raw_amplitude,
                        "sampleKind": "exact_finite_repair_edge_pulse",
                        "sourcePath": "smallUniverse.cycles + smallUniverse.repairFrames",
                        "claimBoundary": (
                            "Exact finite repair/cycle edge-pulse sample; not a critical-string vibration mode."
                        ),
                    }
                )
                sample_index += 1
                if sample_index >= max_samples:
                    return _normalize_string_vibration_amplitudes(rows)
    return _normalize_string_vibration_amplitudes(rows)


def _normalize_string_vibration_amplitudes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_amplitude = max(
        (
            float(value)
            for row in rows
            for value in [_optional_float(row.get("rawAmplitude"))]
            if value is not None and math.isfinite(value)
        ),
        default=0.0,
    )
    for row in rows:
        raw = _optional_float(row.get("rawAmplitude")) or 0.0
        row["normalizedAmplitude"] = float(raw / max_amplitude) if max_amplitude > 0.0 else float(row["edgePulse"])
    return rows


def _edge_pair(edge: Any) -> tuple[int, int] | None:
    if isinstance(edge, dict):
        left = edge.get("source", edge.get("from", edge.get("u", edge.get("left"))))
        right = edge.get("target", edge.get("to", edge.get("v", edge.get("right"))))
    elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
        left, right = edge[0], edge[1]
    else:
        return None
    try:
        return int(left), int(right)
    except (TypeError, ValueError):
        return None


def _repair_active_edge(frame: dict[str, Any]) -> tuple[int, int] | None:
    node = frame.get("node")
    parent = frame.get("parent")
    try:
        if node is None or parent is None:
            return None
        return int(parent), int(node)
    except (TypeError, ValueError):
        return None


def _cycle_state_occupation(cycle_nodes: list[Any], state: list[Any]) -> float | None:
    if not cycle_nodes or not state:
        return None
    active = 0
    total = 0
    for node in cycle_nodes:
        try:
            index = int(node)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(state):
            total += 1
            try:
                active += 1 if float(state[index]) > 0.0 else 0
            except (TypeError, ValueError):
                active += 1 if bool(state[index]) else 0
    return float(active / total) if total else None


def _emergent_curved_spacetime_payload(
    *,
    visualization_views: dict[str, Any],
    bulk_payload: dict[str, Any],
    screen_payload: dict[str, Any],
) -> dict[str, Any]:
    view = (
        visualization_views.get("emergentCurvedSpacetime", {})
        if isinstance(visualization_views.get("emergentCurvedSpacetime"), dict)
        else {}
    )
    view_receipts = view.get("receipts", {}) if isinstance(view.get("receipts"), dict) else {}
    einstein_branch = (
        view.get("einsteinBranchEntry", {}) if isinstance(view.get("einsteinBranchEntry"), dict) else {}
    )
    bulk_receipts = bulk_payload.get("receipts", {}) if isinstance(bulk_payload.get("receipts"), dict) else {}
    proto = (
        bulk_payload.get("protoParticleCandidates", {})
        if isinstance(bulk_payload.get("protoParticleCandidates"), dict)
        else {}
    )
    proto_receipts = proto.get("receipts", {}) if isinstance(proto.get("receipts"), dict) else {}
    points: list[dict[str, Any]] = []
    object_count = 0
    event_count = 0
    for row in bulk_payload.get("objects", []) if isinstance(bulk_payload.get("objects"), list) else []:
        if not isinstance(row, dict):
            continue
        point = _coord3([row.get("x"), row.get("y"), row.get("z")])
        if point is None:
            continue
        observer_count = _optional_float(row.get("observerCount")) or 0.0
        support_size = _optional_float(row.get("supportSize")) or 0.0
        compactness = _optional_float(row.get("h3CompactnessNormalized"))
        localization_quality = 1.0 if compactness is None else max(0.05, 1.0 - max(0.0, min(1.0, compactness)))
        mass_proxy = max(observer_count, support_size, 1.0)
        points.append(
            {
                "sourceId": str(row.get("objectId") or f"h3_object_{object_count:06d}"),
                "sourceKind": "consensus_h3_object",
                "coordinateSystem": "observer_facing_h3_chart",
                "position": point,
                "x": point[0],
                "y": point[1],
                "z": point[2],
                "cycle": None,
                "relativeTime": None,
                "massProxy": float(mass_proxy),
                "stressEnergyProxy": float(mass_proxy * localization_quality),
                "quotientVisibleSourceDensity": float(mass_proxy * localization_quality),
                "sourceDensityAncestry": "observer_count/support_size_weighted_by_h3_localization",
                "observerCount": observer_count,
                "supportSize": support_size,
                "h3CompactnessNormalized": compactness,
                "particleLike": False,
                "diagnosticOnly": True,
                "sourcePath": "consensusBulk.objects",
            }
        )
        object_count += 1
    worldlines = proto.get("worldlines", []) if isinstance(proto.get("worldlines"), list) else []
    for worldline in worldlines:
        if not isinstance(worldline, dict):
            continue
        worldline_id = str(worldline.get("worldlineId") or f"worldline_{event_count:06d}")
        path_length = _optional_float(worldline.get("h3PathLength")) or 0.0
        mean_step = _optional_float(worldline.get("meanH3Step")) or 0.0
        particle_like = bool(worldline.get("particleLike", False))
        for index, event in enumerate(worldline.get("events", []) if isinstance(worldline.get("events"), list) else []):
            if not isinstance(event, dict):
                continue
            point = _coord3(event.get("h3SpatialPoint"))
            if point is None:
                continue
            support_nodes = _optional_float(event.get("supportNodeCount")) or 1.0
            fit_residual = _optional_float(event.get("fitResidual")) or 0.0
            cycle = _optional_float(event.get("cycle"))
            transport_distance = _optional_float(event.get("transportDistance")) or mean_step
            contact_boost = 1.0
            if event.get("contactOutcome") in {"bind", "annihilate"}:
                contact_boost = 1.25
            stress_proxy = support_nodes * (1.0 + min(path_length, 10.0) / 10.0) * contact_boost
            stress_proxy /= 1.0 + max(0.0, fit_residual)
            points.append(
                {
                    "sourceId": f"{worldline_id}:{index}",
                    "sourceKind": "proto_worldline_event",
                    "coordinateSystem": "observer_facing_h3_chart",
                    "position": point,
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                    "cycle": cycle,
                    "relativeTime": None,
                    "massProxy": float(support_nodes),
                    "stressEnergyProxy": float(max(stress_proxy, 1e-12)),
                    "quotientVisibleSourceDensity": float(max(stress_proxy, 1e-12)),
                    "sourceDensityAncestry": "proto_worldline_support_nodes_transport_and_local_residual",
                    "supportNodeCount": support_nodes,
                    "fitResidual": fit_residual,
                    "h3PathLength": path_length,
                    "meanH3Step": mean_step,
                    "transportDistance": transport_distance,
                    "worldlineId": worldline_id,
                    "eventIndex": index,
                    "event": event.get("event"),
                    "class": event.get("class"),
                    "contactOutcome": event.get("contactOutcome") or worldline.get("contactOutcome"),
                    "particleLike": particle_like,
                    "diagnosticOnly": True,
                    "sourcePath": "consensusBulk.protoParticleCandidates.worldlines[*].events",
                }
            )
            event_count += 1
    source_math = _attach_oph_curvature_compaction_fields(points)
    event_points = [row for row in points if row.get("cycle") is not None]
    static_total = float(
        sum(
            _optional_float(row.get("curvaturePotential")) or 0.0
            for row in points
            if row.get("sourceKind") == "consensus_h3_object"
        )
    )
    cycles = sorted({float(row["cycle"]) for row in event_points if _optional_float(row.get("cycle")) is not None})
    if cycles:
        cycle_min = cycles[0]
        cycle_span = max(cycles[-1] - cycle_min, 1.0)
        for row in event_points:
            cycle = _optional_float(row.get("cycle"))
            row["relativeTime"] = float((cycle - cycle_min) / cycle_span) if cycle is not None else None
        time_slices = []
        for index, cycle in enumerate(cycles[:256]):
            events = [row for row in event_points if _optional_float(row.get("cycle")) == cycle]
            total = static_total + float(sum(_optional_float(row.get("curvaturePotential")) or 0.0 for row in events))
            compactions = [float(_optional_float(row.get("compactificationFactor")) or 0.0) for row in events]
            scale_factors = [float(_optional_float(row.get("emergentSpatialScaleFactor")) or 1.0) for row in events]
            time_slices.append(
                {
                    "sliceIndex": index,
                    "cycle": cycle,
                    "relativeTime": float((cycle - cycle_min) / cycle_span),
                    "eventCount": len(events),
                    "sourceCount": object_count + len(events),
                    "totalCurvaturePotential": total,
                    "totalSourceDensity": float(
                        sum(_optional_float(row.get("normalizedSourceDensity")) or 0.0 for row in events)
                    ),
                    "maxCurvaturePotential": max(
                        [static_total, *[float(_optional_float(row.get("curvaturePotential")) or 0.0) for row in events]],
                        default=0.0,
                    ),
                    "maxCompactificationFactor": max(compactions, default=0.0),
                    "meanEmergentSpatialScaleFactor": float(np.mean(scale_factors)) if scale_factors else 1.0,
                }
            )
    elif points:
        compactions = [float(_optional_float(row.get("compactificationFactor")) or 0.0) for row in points]
        scale_factors = [float(_optional_float(row.get("emergentSpatialScaleFactor")) or 1.0) for row in points]
        time_slices = [
            {
                "sliceIndex": 0,
                "cycle": None,
                "relativeTime": 0.0,
                "eventCount": 0,
                "sourceCount": len(points),
                "totalCurvaturePotential": float(
                    sum(_optional_float(row.get("curvaturePotential")) or 0.0 for row in points)
                ),
                "maxCurvaturePotential": max(
                    (_optional_float(row.get("curvaturePotential")) or 0.0 for row in points),
                    default=0.0,
                ),
                "totalSourceDensity": float(
                    sum(_optional_float(row.get("normalizedSourceDensity")) or 0.0 for row in points)
                ),
                "maxCompactificationFactor": max(compactions, default=0.0),
                "meanEmergentSpatialScaleFactor": float(np.mean(scale_factors)) if scale_factors else 1.0,
            }
        ]
    else:
        time_slices = []
    positions = [row["position"] for row in points if isinstance(row.get("position"), list)]
    spatial_extent = {
        "min": [min(values) for values in zip(*positions, strict=False)] if positions else None,
        "max": [max(values) for values in zip(*positions, strict=False)] if positions else None,
    }
    continuous_field = _continuous_observer_facing_bulk_field(points, spatial_extent=spatial_extent)
    receipts = {
        "emergent_curved_spacetime_visualization_receipt": bool(points),
        "observer_facing_consensus_3d_bulk_readout_receipt": bool(
            bulk_receipts.get(
                "observer_facing_consensus_3d_bulk_readout_receipt",
                bulk_receipts.get("theorem_assisted_consensus_3d_bulk_readout_receipt", False),
            )
        ),
        "strict_neutral_third_person_bulk_receipt": bool(
            bulk_receipts.get("strict_neutral_third_person_bulk_receipt", False)
        ),
        "bulk_worldline_precursor_receipt": bool(proto_receipts.get("bulk_worldline_precursor_receipt", False)),
        "particle_matter_receipt": bool(proto_receipts.get("particle_matter_receipt", False)),
        "matter_sources_visible_receipt": bool(points),
        "einstein_branch_entry_receipt": bool(
            view_receipts.get("einstein_branch_entry_receipt", False)
        ),
        "EINSTEIN_BRANCH_ENTRY_RECEIPT": bool(
            view_receipts.get("EINSTEIN_BRANCH_ENTRY_RECEIPT", False)
        ),
        "OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1": bool(
            view_receipts.get("OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1", False)
        ),
        "raw_production_gravity_requested": bool(
            view_receipts.get("raw_production_gravity_requested", False)
        ),
        "raw_physical_gravity_requested": bool(
            view_receipts.get("raw_physical_gravity_requested", False)
        ),
        "production_gravity_receipt": bool(view_receipts.get("production_gravity_receipt", False)),
        "physical_gravity_prediction": bool(view_receipts.get("physical_gravity_prediction", False)),
        "einstein_equation_solution_receipt": bool(
            view_receipts.get("einstein_equation_solution_receipt", False)
        ),
    }
    return {
        "schema": "oph_emergent_curved_spacetime_visualization_v1",
        "viewId": "emergentCurvedSpacetime",
        "label": view.get("label", "Emergent curved-spacetime proxy"),
        "sectionKind": view.get("sectionKind", "emergent_curved_spacetime_proxy"),
        "description": view.get("description"),
        "contentAvailable": bool(points),
        "viewContract": view,
        "sourceCounts": {
            "consensusH3ObjectCount": object_count,
            "protoWorldlineCount": len(worldlines),
            "protoWorldlineEventCount": event_count,
            "curvatureProxyPointCount": len(points),
            "timeSliceCount": len(time_slices),
            "screenPointCount": len(screen_payload.get("points", []) if isinstance(screen_payload.get("points"), list) else []),
        },
        "coordinateSystem": "observer_facing_h3_chart",
        "sourceMath": source_math,
        "einsteinBranchEntry": einstein_branch,
        "curvatureProxyPoints": points,
        "stressEnergySources": points,
        "spacetimeCompactionField": points,
        "fieldSamples": points,
        "continuousBulkField": continuous_field,
        "densityFieldSamples": continuous_field.get("volumeSamples", []),
        "curvatureFieldSlices": continuous_field.get("sliceSamples", []),
        "timeSlices": time_slices,
        "spatialExtent": spatial_extent,
        "renderLayers": view.get("renderLayers", []),
        "animationChannels": view.get("animationChannels", []),
        "receipts": receipts,
        "nonClaims": view.get(
            "nonClaims",
            [
                "Einstein-equation solution",
                "physical gravity prediction",
                "strict neutral third-person bulk",
            ],
        ),
        "claimBoundary": view.get(
            "claimBoundary",
            (
                "Diagnostic stress/curvature proxy over observer-facing H3 object/worldline data. "
                "This is useful for a curved-spacetime visualization, but it is not production gravity, "
                "a solved metric, or a physical prediction."
            ),
        ),
    }


def _continuous_observer_facing_bulk_field(
    points: list[dict[str, Any]],
    *,
    spatial_extent: dict[str, Any],
    max_sources: int = 2048,
    volume_resolution: int = 11,
    slice_resolution: int = 25,
    time_slice_resolution: int = 17,
    max_time_slices: int = 12,
) -> dict[str, Any]:
    """Sample a continuous renderer field over the observer-facing H3 chart."""

    source_positions, source_weights = _curvature_source_arrays(points, max_sources=max_sources)
    if source_positions.size == 0 or source_weights.size == 0:
        return {
            "schema": "oph_continuous_observer_facing_h3_bulk_field_v1",
            "contentAvailable": False,
            "claimBoundary": "No source points were available for a continuous bulk-field visualization.",
        }
    extent = _field_extent_from_sources(source_positions, spatial_extent)
    axis_values = [
        np.linspace(float(extent["min"][axis]), float(extent["max"][axis]), int(volume_resolution)).tolist()
        for axis in range(3)
    ]
    volume_positions = np.asarray(
        [[x, y, z] for x in axis_values[0] for y in axis_values[1] for z in axis_values[2]],
        dtype=float,
    )
    volume_rows = _curvature_field_sample_rows(
        volume_positions,
        source_positions=source_positions,
        source_weights=source_weights,
        prefix="volume",
        sigma=_field_sigma(extent, volume_resolution),
        limit=None,
    )
    z_levels = [
        float(extent["min"][2]),
        float(0.5 * (extent["min"][2] + extent["max"][2])),
        float(extent["max"][2]),
    ]
    slice_rows: list[dict[str, Any]] = []
    x_slice = np.linspace(float(extent["min"][0]), float(extent["max"][0]), int(slice_resolution))
    y_slice = np.linspace(float(extent["min"][1]), float(extent["max"][1]), int(slice_resolution))
    for slice_index, z_value in enumerate(z_levels):
        sample_positions = np.asarray([[x, y, z_value] for x in x_slice for y in y_slice], dtype=float)
        rows = _curvature_field_sample_rows(
            sample_positions,
            source_positions=source_positions,
            source_weights=source_weights,
            prefix=f"z_slice_{slice_index}",
            sigma=_field_sigma(extent, slice_resolution),
            limit=None,
        )
        for row in rows:
            row["sliceIndex"] = slice_index
            row["sliceAxis"] = "z"
            row["sliceValue"] = z_value
        slice_rows.extend(rows)
    temporal_rows = _temporal_curvature_field_slices(
        points,
        extent=extent,
        max_sources=max_sources,
        resolution=time_slice_resolution,
        max_time_slices=max_time_slices,
    )
    return {
        "schema": "oph_continuous_observer_facing_h3_bulk_field_v1",
        "contentAvailable": True,
        "fieldKind": "interpolated_consensus_density_and_curvature_proxy",
        "coordinateSystem": "observer_facing_h3_chart",
        "grid": {
            "volumeResolution": [int(volume_resolution), int(volume_resolution), int(volume_resolution)],
            "sliceResolution": [int(slice_resolution), int(slice_resolution)],
            "timeSliceResolution": [int(time_slice_resolution), int(time_slice_resolution)],
            "extent": extent,
            "axisValues": {"x": axis_values[0], "y": axis_values[1], "z": axis_values[2]},
        },
        "sourceCount": int(source_positions.shape[0]),
        "sourceCountBeforeCap": int(len(points)),
        "sourceCap": int(max_sources),
        "volumeSamples": volume_rows,
        "sliceSamples": slice_rows,
        "temporalSliceSamples": temporal_rows,
        "recommendedRenderings": [
            "volume fog from normalizedDensity or normalizedCurvaturePotential",
            "isosurface from compactificationFactor",
            "warped z-slice grid from emergentSpatialScaleFactor",
            "animated z=mid temporal slices from temporalSliceSamples",
        ],
        "claimBoundary": (
            "Continuous field sampled from observer-facing H3 object/worldline source rows. This is a "
            "renderer interpolation of exported readout sources, not a chart-blind neutral bulk or an "
            "Einstein-equation solution."
        ),
    }


def _curvature_source_arrays(
    points: list[dict[str, Any]], *, max_sources: int
) -> tuple[np.ndarray, np.ndarray]:
    positions: list[list[float]] = []
    weights: list[float] = []
    for row in points:
        if not isinstance(row, dict):
            continue
        point = _coord3(row.get("position"))
        if point is None:
            continue
        weight = max(
            0.0,
            float(
                _optional_float(row.get("sourceDensity"))
                or _optional_float(row.get("quotientVisibleSourceDensity"))
                or _optional_float(row.get("stressEnergyProxy"))
                or _optional_float(row.get("massProxy"))
                or 0.0
            ),
        )
        if weight <= 0.0:
            continue
        positions.append(point)
        weights.append(weight)
    if not positions:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
    pos = np.asarray(positions, dtype=float)
    weight_arr = np.asarray(weights, dtype=float)
    if pos.shape[0] > max_sources:
        order = np.argsort(-weight_arr)[:max_sources]
        pos = pos[order]
        weight_arr = weight_arr[order]
    return pos, weight_arr


def _field_extent_from_sources(source_positions: np.ndarray, spatial_extent: dict[str, Any]) -> dict[str, Any]:
    raw_min = _coord3(spatial_extent.get("min")) if isinstance(spatial_extent, dict) else None
    raw_max = _coord3(spatial_extent.get("max")) if isinstance(spatial_extent, dict) else None
    if raw_min is None or raw_max is None:
        raw_min = np.min(source_positions, axis=0).astype(float).tolist()
        raw_max = np.max(source_positions, axis=0).astype(float).tolist()
    spans = [max(0.25, float(raw_max[index]) - float(raw_min[index])) for index in range(3)]
    pad = [max(0.2, 0.18 * span) for span in spans]
    return {
        "min": [float(raw_min[index] - pad[index]) for index in range(3)],
        "max": [float(raw_max[index] + pad[index]) for index in range(3)],
    }


def _field_sigma(extent: dict[str, Any], resolution: int) -> float:
    minimum = _coord3(extent.get("min")) or [-1.0, -1.0, -1.0]
    maximum = _coord3(extent.get("max")) or [1.0, 1.0, 1.0]
    spans = [max(1.0e-6, float(maximum[index]) - float(minimum[index])) for index in range(3)]
    return float(max(spans) / max(3.0, float(resolution - 1)) * 1.2)


def _curvature_field_sample_rows(
    sample_positions: np.ndarray,
    *,
    source_positions: np.ndarray,
    source_weights: np.ndarray,
    prefix: str,
    sigma: float,
    limit: int | None,
) -> list[dict[str, Any]]:
    if sample_positions.size == 0:
        return []
    sample_positions = np.asarray(sample_positions, dtype=float)
    source_positions = np.asarray(source_positions, dtype=float)
    source_weights = np.asarray(source_weights, dtype=float)
    density = _gaussian_density_samples(sample_positions, source_positions, source_weights, sigma=sigma)
    potential = _h3_green_potentials_from_sources(sample_positions, source_positions, source_weights)
    max_density = float(np.max(density)) if density.size else 0.0
    max_potential = float(np.max(potential)) if potential.size else 0.0
    rows = []
    row_count = sample_positions.shape[0] if limit is None else min(sample_positions.shape[0], int(limit))
    for index in range(row_count):
        density_norm = float(density[index] / max_density) if max_density > 0.0 else 0.0
        potential_norm = float(potential[index] / max_potential) if max_potential > 0.0 else 0.0
        compactification = float(potential_norm / (1.0 + potential_norm))
        scale_factor = float(1.0 / (1.0 + potential_norm))
        point = sample_positions[index]
        rows.append(
            {
                "sampleId": f"{prefix}_{index:05d}",
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "position": [float(point[0]), float(point[1]), float(point[2])],
                "density": float(density[index]),
                "normalizedDensity": density_norm,
                "h3GreenPotential": float(potential[index]),
                "normalizedCurvaturePotential": potential_norm,
                "curvaturePotential": potential_norm,
                "compactificationFactor": compactification,
                "emergentSpatialScaleFactor": scale_factor,
                "localMetricConformalFactor": float(scale_factor * scale_factor),
            }
        )
    return rows


def _gaussian_density_samples(
    sample_positions: np.ndarray,
    source_positions: np.ndarray,
    source_weights: np.ndarray,
    *,
    sigma: float,
    batch_size: int = 512,
) -> np.ndarray:
    sigma = max(float(sigma), 1.0e-6)
    density = np.zeros(sample_positions.shape[0], dtype=float)
    for start in range(0, sample_positions.shape[0], batch_size):
        stop = min(start + batch_size, sample_positions.shape[0])
        diff = sample_positions[start:stop, None, :] - source_positions[None, :, :]
        euclidean_squared = np.sum(diff * diff, axis=2)
        kernel = np.exp(-0.5 * euclidean_squared / (sigma * sigma))
        density[start:stop] = np.matmul(kernel, source_weights).reshape(-1)
    return density


def _h3_green_potentials_from_sources(
    sample_positions: np.ndarray,
    source_positions: np.ndarray,
    source_weights: np.ndarray,
    *,
    softening: float = 0.15,
    chunk_size: int = 256,
) -> np.ndarray:
    if sample_positions.size == 0 or source_positions.size == 0:
        return np.zeros(sample_positions.shape[0], dtype=float)
    samples = np.asarray(sample_positions, dtype=float)
    sources = np.asarray(source_positions, dtype=float)
    weights = np.asarray(source_weights, dtype=float).reshape(-1, 1)
    potentials = np.zeros(samples.shape[0], dtype=float)
    source_norms = np.linalg.norm(sources, axis=1)
    source_inside = source_norms < 0.999
    for start in range(0, samples.shape[0], max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), samples.shape[0])
        left = samples[start:stop]
        diff = left[:, None, :] - sources[None, :, :]
        euclidean_squared = np.sum(diff * diff, axis=2)
        euclidean = np.sqrt(np.maximum(euclidean_squared, 0.0))
        left_norms = np.linalg.norm(left, axis=1)
        left_inside = left_norms < 0.999
        denom = np.maximum(
            (1.0 - left_norms[:, None] * left_norms[:, None])
            * (1.0 - source_norms[None, :] * source_norms[None, :]),
            1.0e-12,
        )
        poincare_arg = np.maximum(1.0, 1.0 + 2.0 * euclidean_squared / denom)
        h3_distance = np.arccosh(poincare_arg)
        fallback_distance = 2.0 * np.arcsinh(0.5 * euclidean)
        use_poincare = left_inside[:, None] & source_inside[None, :]
        h3_distance = np.where(use_poincare, h3_distance, fallback_distance)
        kernel_denom = 4.0 * math.pi * np.maximum(np.sinh(h3_distance + softening), softening)
        kernel = np.exp(-h3_distance) / kernel_denom
        potentials[start:stop] = np.matmul(kernel, weights).reshape(-1)
    return potentials


def _temporal_curvature_field_slices(
    points: list[dict[str, Any]],
    *,
    extent: dict[str, Any],
    max_sources: int,
    resolution: int,
    max_time_slices: int,
) -> list[dict[str, Any]]:
    static_points = [row for row in points if _optional_float(row.get("cycle")) is None]
    dynamic_cycles = sorted(
        {
            float(cycle)
            for row in points
            for cycle in [_optional_float(row.get("cycle"))]
            if cycle is not None
        }
    )
    if not dynamic_cycles:
        return []
    if len(dynamic_cycles) > max_time_slices:
        indices = np.linspace(0, len(dynamic_cycles) - 1, int(max_time_slices), dtype=int)
        cycles = [dynamic_cycles[int(index)] for index in indices]
    else:
        cycles = dynamic_cycles
    z_mid = float(0.5 * (extent["min"][2] + extent["max"][2]))
    x_values = np.linspace(float(extent["min"][0]), float(extent["max"][0]), int(resolution))
    y_values = np.linspace(float(extent["min"][1]), float(extent["max"][1]), int(resolution))
    sample_positions = np.asarray([[x, y, z_mid] for x in x_values for y in y_values], dtype=float)
    rows: list[dict[str, Any]] = []
    cycle_min = cycles[0]
    cycle_span = max(cycles[-1] - cycle_min, 1.0)
    for slice_index, cycle in enumerate(cycles):
        cycle_points = [
            row
            for row in points
            if _optional_float(row.get("cycle")) is not None and float(_optional_float(row.get("cycle")) or 0.0) == cycle
        ]
        source_positions, source_weights = _curvature_source_arrays(
            [*static_points, *cycle_points], max_sources=max_sources
        )
        if source_positions.size == 0:
            continue
        slice_rows = _curvature_field_sample_rows(
            sample_positions,
            source_positions=source_positions,
            source_weights=source_weights,
            prefix=f"time_slice_{slice_index}",
            sigma=_field_sigma(extent, resolution),
            limit=None,
        )
        for row in slice_rows:
            row["sliceIndex"] = slice_index
            row["cycle"] = cycle
            row["relativeTime"] = float((cycle - cycle_min) / cycle_span)
            row["sliceAxis"] = "z"
            row["sliceValue"] = z_mid
        rows.extend(slice_rows)
    return rows


def _attach_oph_curvature_compaction_fields(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach OPH Einstein-branch visualization fields to source rows.

    The source is a quotient-visible support/readout density, not rest mass.
    The potential is a screened H3 Green-kernel accumulation over the exported
    source points. The compactification factor is a bounded display scalar that
    makes local spatial scale shrink near larger source density.
    """

    if not points:
        return {
            "model": "oph_quotient_visible_source_to_h3_compaction_v1",
            "sourceDefinition": "no quotient-visible source rows available",
            "claimBoundary": "No curved-spacetime source field was emitted.",
        }
    positions = [_coord3(row.get("position")) or [0.0, 0.0, 0.0] for row in points]
    raw_sources = np.asarray(
        [
            max(
                0.0,
                float(
                    _optional_float(row.get("quotientVisibleSourceDensity"))
                    or _optional_float(row.get("stressEnergyProxy"))
                    or _optional_float(row.get("massProxy"))
                    or 0.0
                ),
            )
            for row in points
        ],
        dtype=float,
    )
    total_source = float(np.sum(raw_sources))
    max_source = float(np.max(raw_sources)) if raw_sources.size else 0.0
    potentials = _h3_green_potentials(positions, raw_sources)
    max_potential = float(np.max(potentials)) if potentials.size else 0.0
    for index, row in enumerate(points):
        raw_source = float(raw_sources[index])
        density_norm = raw_source / max_source if max_source > 0.0 else 0.0
        green_potential = float(potentials[index])
        potential_norm = green_potential / max_potential if max_potential > 0.0 else 0.0
        compactification = potential_norm / (1.0 + potential_norm)
        scale_factor = 1.0 / (1.0 + potential_norm)
        row["sourceDensity"] = raw_source
        row["normalizedSourceDensity"] = float(density_norm)
        row["h3GreenPotential"] = green_potential
        row["normalizedH3GreenPotential"] = float(potential_norm)
        row["curvaturePotential"] = float(potential_norm)
        row["compactificationFactor"] = float(compactification)
        row["emergentSpatialScaleFactor"] = float(scale_factor)
        row["localMetricConformalFactor"] = float(scale_factor * scale_factor)
        row["curvatureRadiusProxy"] = (
            float(1.0 / math.sqrt(max(potential_norm, 1.0e-6))) if potential_norm > 0.0 else None
        )
        row["productionGravityContributor"] = False
        row["gravitySourceInterpretation"] = "quotient_visible_stress_readout_not_rest_mass"
    return {
        "model": "oph_quotient_visible_source_to_h3_compaction_v1",
        "paperBranch": "Einstein-branch geometry readout / null-stress bridge diagnostic",
        "sourceDefinition": (
            "sourceDensity = quotient-visible consensus object support/localization or "
            "proto-worldline support/transport/residual readout; it is not raw rest mass"
        ),
        "distanceKernel": "screened H3 chart Green proxy exp(-d_H3)/(4*pi*sinh(d_H3 + eps))",
        "compactificationLaw": (
            "compactificationFactor = normalizedH3GreenPotential/(1+normalizedH3GreenPotential); "
            "emergentSpatialScaleFactor = 1/(1+normalizedH3GreenPotential)"
        ),
        "sourceCount": len(points),
        "totalSourceDensity": total_source,
        "maxSourceDensity": max_source,
        "maxH3GreenPotential": max_potential,
        "receiptsRequiredForPromotion": [
            "particle_matter_receipt",
            "einstein_branch_entry_receipt",
            "production_gravity_receipt",
            "einstein_equation_solution_receipt",
            "strict_neutral_third_person_bulk_receipt",
        ],
        "claimBoundary": (
            "Renderer-ready OPH branch diagnostic. It couples exported observer-visible source rows to "
            "a hyperbolic compactification field, but it is not a solved Einstein metric or physical "
            "gravity prediction unless the E0 Einstein bridge manifest and gravity promotion receipts pass."
        ),
    }


def _h3_chart_distance_proxy(left: list[float], right: list[float]) -> float:
    euclidean = _vec_norm([float(left[index]) - float(right[index]) for index in range(3)])
    left_norm = _vec_norm(left)
    right_norm = _vec_norm(right)
    if left_norm < 0.999 and right_norm < 0.999:
        denom = max((1.0 - left_norm * left_norm) * (1.0 - right_norm * right_norm), 1.0e-12)
        cosh_arg = 1.0 + 2.0 * euclidean * euclidean / denom
        return float(math.acosh(max(1.0, cosh_arg)))
    return float(2.0 * math.asinh(0.5 * euclidean))


def _screened_h3_green_kernel(distance: float, *, softening: float = 0.15) -> float:
    d = max(0.0, float(distance))
    denom = 4.0 * math.pi * max(math.sinh(d + softening), softening)
    return float(math.exp(-d) / denom)


def _h3_green_potentials(
    positions: list[list[float]],
    raw_sources: np.ndarray,
    *,
    softening: float = 0.15,
    chunk_size: int = 256,
) -> np.ndarray:
    if not positions:
        return np.zeros(0, dtype=float)
    coords = np.asarray(positions, dtype=float)
    sources = np.asarray(raw_sources, dtype=float)
    n = int(coords.shape[0])
    potentials = np.zeros(n, dtype=float)
    source_norms = np.linalg.norm(coords, axis=1)
    source_inside = source_norms < 0.999
    source_weight = sources.reshape(-1, 1)
    for start in range(0, n, max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), n)
        left = coords[start:stop]
        diff = left[:, None, :] - coords[None, :, :]
        euclidean_squared = np.sum(diff * diff, axis=2)
        euclidean = np.sqrt(np.maximum(euclidean_squared, 0.0))
        left_norms = np.linalg.norm(left, axis=1)
        left_inside = left_norms < 0.999
        denom = np.maximum(
            (1.0 - left_norms[:, None] * left_norms[:, None])
            * (1.0 - source_norms[None, :] * source_norms[None, :]),
            1.0e-12,
        )
        poincare_arg = np.maximum(1.0, 1.0 + 2.0 * euclidean_squared / denom)
        h3_distance = np.arccosh(poincare_arg)
        fallback_distance = 2.0 * np.arcsinh(0.5 * euclidean)
        use_poincare = left_inside[:, None] & source_inside[None, :]
        h3_distance = np.where(use_poincare, h3_distance, fallback_distance)
        kernel_denom = 4.0 * math.pi * np.maximum(np.sinh(h3_distance + softening), softening)
        kernel = np.exp(-h3_distance) / kernel_denom
        potentials[start:stop] = np.matmul(kernel, source_weight).reshape(-1)
    return potentials


def _observer_cinema_payload(
    *,
    observer_payload: dict[str, Any],
    subjective_cameras: list[dict[str, Any]],
) -> dict[str, Any]:
    views = (
        observer_payload.get("objectiveObserverViews", [])
        if isinstance(observer_payload.get("objectiveObserverViews"), list)
        else []
    )
    time_frames = (
        observer_payload.get("timeFrames", [])
        if isinstance(observer_payload.get("timeFrames"), list)
        else []
    )
    overlap_links = (
        observer_payload.get("overlapLinks", [])
        if isinstance(observer_payload.get("overlapLinks"), list)
        else []
    )
    shot_list = []
    for camera in subjective_cameras[:32]:
        if not isinstance(camera, dict):
            continue
        frames = camera.get("timeFrames") if isinstance(camera.get("timeFrames"), list) else []
        shot_list.append(
            {
                "cameraId": camera.get("cameraId"),
                "observerId": camera.get("observerId"),
                "frameCount": len(frames),
                "supportPatchCount": camera.get("supportPatchCount"),
                "firstVisibleReadoutHash": frames[0].get("visibleReadoutHash") if frames else None,
                "lastVisibleReadoutHash": frames[-1].get("visibleReadoutHash") if frames else None,
                "visibleProtoWorldlineIds": camera.get("visibleProtoWorldlineIds", []),
                "visibleProtoWorldlineSightingCount": camera.get("visibleProtoWorldlineSightingCount", 0),
                "suggestedUse": "first_person_observer_readout_track",
            }
        )
    proto_sighting_count = sum(
        int(camera.get("visibleProtoWorldlineSightingCount") or 0)
        for camera in subjective_cameras
        if isinstance(camera, dict)
    )
    proto_camera_count = sum(
        1
        for camera in subjective_cameras
        if isinstance(camera, dict) and int(camera.get("visibleProtoWorldlineSightingCount") or 0) > 0
    )
    return {
        "schema": "oph_observer_cinema_v1",
        "description": (
            "Observer-local camera and frame data for first-person readout cinema. Cameras are derived "
            "from observer-visible support, packets, records, and modular-time frames."
        ),
        "observerViews": views,
        "subjectiveCameras": subjective_cameras,
        "globalTimelineFrames": time_frames,
        "overlapLinks": overlap_links,
        "shotList": shot_list,
        "availability": {
            "observerViewCount": len(views),
            "subjectiveCameraCount": len(subjective_cameras),
            "globalTimelineFrameCount": len(time_frames),
            "cameraFrameCount": sum(
                len(camera.get("timeFrames") or [])
                for camera in subjective_cameras
                if isinstance(camera, dict)
            ),
            "protoWorldlineSightingCount": proto_sighting_count,
            "protoWorldlineVisibleCameraCount": proto_camera_count,
            "overlapLinkCount": len(overlap_links),
        },
        "receipts": observer_payload.get("receipts", {}),
        "claimBoundary": (
            "Observer cinema is first-person/local readout rendering. It is not an omniscient global "
            "camera or objective global time coordinate."
        ),
    }


def _hilbert_space_observer_algebra_payload(observer_payload: dict[str, Any]) -> dict[str, Any]:
    views = (
        observer_payload.get("objectiveObserverViews", [])
        if isinstance(observer_payload.get("objectiveObserverViews"), list)
        else []
    )
    links = (
        observer_payload.get("overlapLinks", [])
        if isinstance(observer_payload.get("overlapLinks"), list)
        else []
    )
    support_counts = [
        int(row.get("supportPatchCount"))
        for row in observer_payload.get("observers", [])
        if isinstance(row, dict) and _safe_int(row.get("supportPatchCount"), -1) >= 0
    ]
    support_basis: set[int] = set()
    record_basis: set[str] = set()
    object_basis: set[str] = set()
    visible_record_packets = 0
    visible_object_packets = 0
    time_frame_count = 0
    representatives: list[dict[str, Any]] = []
    for view in views:
        if not isinstance(view, dict):
            continue
        support_nodes = [
            int(node)
            for node in view.get("supportNodeSample", [])
            if isinstance(node, (int, float, str)) and str(node).lstrip("-").isdigit()
        ]
        support_basis.update(support_nodes)
        view_record_packets = _packet_names(view.get("visibleRecordPackets"))
        view_object_packets = _packet_names(view.get("visibleObjectPackets"))
        record_basis.update(view_record_packets)
        object_basis.update(view_object_packets)
        frames = view.get("timeFrames") if isinstance(view.get("timeFrames"), list) else []
        time_frame_count += len(frames)
        readout_hashes = []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            frame_records = _packet_names(frame.get("visibleRecordPackets"))
            frame_objects = _packet_names(frame.get("visibleObjectPackets"))
            visible_record_packets += len(frame_records)
            visible_object_packets += len(frame_objects)
            record_basis.update(frame_records)
            object_basis.update(frame_objects)
            if frame.get("visibleReadoutHash") is not None:
                readout_hashes.append(frame.get("visibleReadoutHash"))
        visible_record_packets += len(view_record_packets)
        visible_object_packets += len(view_object_packets)
        if len(representatives) < 64:
            observer_id = view.get("observerId")
            representatives.append(
                {
                    "observerId": observer_id,
                    "supportProjector": f"P_support_observer_{observer_id}",
                    "supportNodeBasis": support_nodes,
                    "recordPacketBasis": view_record_packets,
                    "objectPacketBasis": view_object_packets,
                    "timeFrameCount": len(frames),
                    "visibleReadoutHashSample": readout_hashes[:8],
                    "sampleMeasurementOperators": [
                        {
                            "operatorId": f"P_support_observer_{observer_id}",
                            "kind": "finite_support_projector",
                            "actsOn": "supportNodeBasis",
                            "visualization": "highlight observer support nodes on the S2 screen",
                        },
                        {
                            "operatorId": f"M_record_packet_observer_{observer_id}",
                            "kind": "record_packet_weight_readout",
                            "actsOn": "visibleRecordPackets",
                            "visualization": "bar/heatmap over visible record packet weights",
                        },
                        {
                            "operatorId": f"M_object_packet_observer_{observer_id}",
                            "kind": "object_packet_weight_readout",
                            "actsOn": "visibleObjectPackets",
                            "visualization": "bar/heatmap over visible object packet weights",
                        },
                        {
                            "operatorId": f"U_modular_step_observer_{observer_id}",
                            "kind": "finite_modular_time_frame_shift",
                            "actsOn": "timeFrames",
                            "visualization": "advance selected observer by one local modular-time frame",
                        },
                    ],
                }
            )
    return {
        "schema": "oph_hilbert_observer_algebra_summary_v1",
        "description": (
            "Finite renderer contract for observer-accessible Hilbert/algebra controls. The basis is the "
            "exported finite support, visible packets, overlap links, and modular-time frames; no "
            "continuum Hilbert-space derivation is claimed."
        ),
        "observerCount": len(observer_payload.get("observers") or []),
        "objectiveObserverViewCount": len(views),
        "overlapLinkCount": len(links),
        "timeFrameCount": time_frame_count,
        "visibleObjectPacketCount": visible_object_packets,
        "visibleRecordPacketCount": visible_record_packets,
        "supportNodeBasisSize": len(support_basis),
        "recordPacketBasisSize": len(record_basis),
        "objectPacketBasisSize": len(object_basis),
        "meanSupportPatchCount": _mean_float_or_none([float(value) for value in support_counts]),
        "finiteSupportAlgebraPopulated": bool(
            views and (visible_object_packets or visible_record_packets or support_basis or links)
        ),
        "basisSamples": {
            "supportNodeBasis": sorted(support_basis)[:256],
            "recordPacketBasis": sorted(record_basis)[:128],
            "objectPacketBasis": sorted(object_basis)[:128],
        },
        "representativeObservers": representatives,
        "algebraGenerators": [
            {
                "generatorId": "P_support",
                "kind": "finite_support_projector_family",
                "source": "objectiveObserverViews[*].supportNodeSample",
            },
            {
                "generatorId": "M_record_packet",
                "kind": "visible_record_packet_measurement_family",
                "source": "objectiveObserverViews[*].visibleRecordPackets + timeFrames[*].visibleRecordPackets",
            },
            {
                "generatorId": "M_object_packet",
                "kind": "visible_object_packet_measurement_family",
                "source": "objectiveObserverViews[*].visibleObjectPackets + timeFrames[*].visibleObjectPackets",
            },
            {
                "generatorId": "I_overlap",
                "kind": "finite_overlap_intertwiner_family",
                "source": "observerModularTime.overlapLinks",
            },
            {
                "generatorId": "U_tau",
                "kind": "finite_modular_time_shift_family",
                "source": "objectiveObserverViews[*].timeFrames",
            },
        ],
        "operatorUiContract": {
            "supportedActions": [
                "select_observer",
                "apply_support_projector",
                "measure_visible_record_packet",
                "measure_visible_object_packet",
                "advance_modular_time_frame",
                "show_overlap_intertwiner",
            ],
            "expectationValueRule": (
                "For packet measurements, use the exported packet weight in the selected frame. For "
                "support projectors, display support membership/highlight fraction. For modular-time "
                "shifts, move to the next exported finite frame."
            ),
        },
        "receipts": observer_payload.get("receipts", {}),
        "claimBoundary": (
            "Finite observer-algebra visualization contract only. It is not a proof of continuum Hilbert "
            "space, operator algebra completion, or Born-rule universality."
        ),
    }


def _observer_anatomy_payload(
    *,
    observer_payload: dict[str, Any],
    subjective_cameras: list[dict[str, Any]],
    hilbert_algebra_payload: dict[str, Any],
) -> dict[str, Any]:
    views = (
        observer_payload.get("objectiveObserverViews", [])
        if isinstance(observer_payload.get("objectiveObserverViews"), list)
        else []
    )
    camera_by_observer = {
        camera.get("observerId"): camera
        for camera in subjective_cameras
        if isinstance(camera, dict)
    }
    observers = []
    for index, view in enumerate(views[:128]):
        if not isinstance(view, dict):
            continue
        frames = view.get("timeFrames") if isinstance(view.get("timeFrames"), list) else []
        observer_id = view.get("observerId")
        camera = camera_by_observer.get(observer_id, {})
        observers.append(
            {
                "observerId": observer_id,
                "axis": view.get("axis"),
                "supportPatchCount": view.get("supportPatchCount"),
                "supportNodeSample": view.get("supportNodeSample", []),
                "supportEntropyCapacity": view.get("supportEntropyCapacity"),
                "visibleRecordPackets": view.get("visibleRecordPackets", []),
                "visibleObjectPackets": view.get("visibleObjectPackets", []),
                "modularClock": {
                    "frameCount": len(frames),
                    "firstRelativeTime": frames[0].get("relativeTime") if frames else None,
                    "lastRelativeTime": frames[-1].get("relativeTime") if frames else None,
                    "firstCycle": frames[0].get("cycle") if frames else None,
                    "lastCycle": frames[-1].get("cycle") if frames else None,
                },
                "readoutHashSample": [
                    frame.get("visibleReadoutHash")
                    for frame in frames[:8]
                    if isinstance(frame, dict) and frame.get("visibleReadoutHash") is not None
                ],
                "cameraId": camera.get("cameraId"),
                "cameraFrameCount": len(camera.get("timeFrames") or []) if isinstance(camera, dict) else 0,
                "algebraRepresentativeIndex": index
                if index < len(hilbert_algebra_payload.get("representativeObservers", []))
                else None,
            }
        )
    return {
        "schema": "oph_observer_anatomy_v1",
        "description": (
            "Per-observer anatomy: finite support, visible record/object packets, modular clock frames, "
            "subjective camera linkage, and algebra-control linkage."
        ),
        "populationSummary": {
            "observerCount": len(observer_payload.get("observers") or []),
            "objectiveObserverViewCount": len(views),
            "exportedAnatomyObserverCount": len(observers),
            "subjectiveCameraCount": len(subjective_cameras),
            "overlapLinkCount": len(observer_payload.get("overlapLinks") or []),
            "finiteSupportAlgebraPopulated": bool(
                hilbert_algebra_payload.get("finiteSupportAlgebraPopulated", False)
            ),
        },
        "observers": observers,
        "overlapAnatomy": observer_payload.get("overlapSummary", {}),
        "receipts": observer_payload.get("receipts", {}),
        "claimBoundary": (
            "Observer anatomy exposes only exported visible support/readout structure. It does not "
            "include hidden representatives or promote neutral-bulk claims."
        ),
    }


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
        count = _screen_patch_count_from_run(run_dir, default=512)
        points = fibonacci_sphere_points(count)
        field_name = "screen_position_support"
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
    readout_base = readout_dir or consensus_pack_dir
    readout = _read_json(readout_base / "observer_consensus_bulk_readout_report.json")
    if not readout:
        readout = _read_json(consensus_pack_dir / "observer_consensus_bulk" / "observer_consensus_bulk_readout_report.json")
    object_summary = _read_json(consensus_pack_dir / "object_h3_bulk_viewer_summary.json")
    observer_experience = _read_json(consensus_pack_dir / "observer_modular_experience_report.json")
    paper_3d = _read_json(consensus_pack_dir / "paper_3d_bulk_chart_report.json")
    objects = _read_h3_objects(consensus_pack_dir, readout_dir, max_objects=max_objects)
    neutral_objects = _read_neutral_object_candidates(consensus_pack_dir, max_objects=max_objects)
    neutral_object_report = _read_json(consensus_pack_dir / "strict_neutral_object_bulk_report.json")
    proto_particles = _read_proto_particle_candidates(consensus_pack_dir, max_worldlines=max(32, max_objects // 4))
    h3_readout = readout.get("h3_object_readout", {}) if isinstance(readout.get("h3_object_readout"), dict) else {}
    observer_h3_object_population = bool(
        readout.get("observer_h3_object_population_receipt", False)
        or readout.get("observer_facing_h3_object_population_receipt", False)
        or h3_readout.get("observer_h3_object_population_receipt", False)
        or h3_readout.get("observer_facing_h3_object_population_receipt", False)
        or object_summary.get("observer_chart_bulk_population_receipt", False)
        or objects
    )
    observer_populated_h3 = bool(
        readout.get("observer_facing_populated_h3_experience_receipt", False)
        or observer_experience.get("observer_facing_populated_h3_experience_receipt", False)
        or h3_readout.get("observer_facing_h3_object_population_receipt", False)
        or observer_h3_object_population
    )
    observer_3p1 = bool(
        readout.get("observer_facing_3p1d_h3_experience_receipt", False)
        or observer_experience.get("observer_facing_3p1d_h3_experience_receipt", False)
    )
    theorem_assisted_bulk = bool(
        readout.get("theorem_assisted_consensus_3d_bulk_readout_receipt", False)
        or object_summary.get("theorem_assisted_h3_bulk", False)
        or (
            paper_3d.get("paper_theorem_3d_bulk_chart_receipt", False)
            and observer_h3_object_population
        )
    )
    observer_facing_bulk = bool(
        readout.get(
            "observer_facing_consensus_3d_bulk_readout_receipt",
            readout.get("theorem_assisted_consensus_3d_bulk_readout_receipt", False),
        )
        or (theorem_assisted_bulk and observer_3p1 and observer_h3_object_population)
    )
    receipts = {
        "observer_like_self_reading_system_receipt": bool(
            readout.get("observer_like_self_reading_system_receipt", False)
        ),
        "observer_modular_time_receipt": bool(
            readout.get("observer_modular_time_receipt", False)
            or observer_experience.get("observer_modular_time_receipt", False)
        ),
        "observer_facing_3p1d_h3_experience_receipt": observer_3p1,
        "observer_facing_populated_h3_experience_receipt": bool(observer_populated_h3),
        "observer_h3_object_population_receipt": observer_h3_object_population,
        "observer_facing_h3_object_population_receipt": observer_h3_object_population,
        "theorem_assisted_consensus_3d_bulk_readout_receipt": theorem_assisted_bulk,
        "observer_facing_consensus_3d_bulk_readout_receipt": observer_facing_bulk,
        "chart_blind_strict_neutral_quotient_bulk_receipt": bool(
            readout.get("chart_blind_strict_neutral_quotient_bulk_receipt", False)
        ),
        "strict_neutral_third_person_bulk_receipt": bool(
            readout.get("strict_neutral_third_person_bulk_receipt", False)
        ),
        "strict_neutral_object_bulk_receipt": bool(
            neutral_object_report.get("STRICT_NEUTRAL_OBJECT_BULK_RECEIPT", False)
            or neutral_object_report.get("strict_neutral_object_bulk", False)
        ),
        "physical_cmb_output_comparison_receipt": bool(
            readout.get("physical_cmb_output_comparison_receipt", False)
        ),
        "physical_cmb_prediction_receipt": bool(readout.get("physical_cmb_prediction_receipt", False)),
    }
    h3_chart_status = _h3_chart_status(
        object_count=len(objects),
        neutral_object_count=len(neutral_objects),
        proto_worldline_count=len(proto_particles.get("worldlines", []) or []),
        receipts=receipts,
    )
    return {
        "description": (
            "Theorem-assisted observer-consensus H3 chart. Dots are consensus object/readback packets "
            "seen by overlapping observers, represented in a derived H3 spatial chart. Holonomy/defect "
            "worldlines are rendered separately as proto-particle candidates when the run emits them."
        ),
        "source": str(consensus_pack_dir),
        "objects": objects,
        "neutralObjectCandidates": neutral_objects,
        "protoParticleCandidates": proto_particles,
        "objectViewerSummary": {
            "objectCount": object_summary.get("object_count"),
            "theoremAssistedH3Bulk": object_summary.get("theorem_assisted_h3_bulk"),
            "observerChartBulkPopulationReceipt": object_summary.get("observer_chart_bulk_population_receipt"),
            "observerOverlapLinkCount": object_summary.get("observer_overlap_link_count"),
            "strictNeutralBulk": object_summary.get("strict_neutral_bulk"),
        },
        "neutralObjectSummary": {
            "written": bool(neutral_object_report or neutral_objects),
            "objectCount": neutral_object_report.get("object_count", len(neutral_objects)),
            "receipt": bool(
                neutral_object_report.get("STRICT_NEUTRAL_OBJECT_BULK_RECEIPT", False)
                or neutral_object_report.get("strict_neutral_object_bulk", False)
            ),
            "selectedModel": (
                (neutral_object_report.get("latent_geometry_selection") or {}).get("selected_model")
                if isinstance(neutral_object_report.get("latent_geometry_selection"), dict)
                else None
            ),
            "medianDimensionEstimate": (
                (neutral_object_report.get("dimension") or {}).get("median_dimension_estimate")
                if isinstance(neutral_object_report.get("dimension"), dict)
                else None
            ),
            "blockers": list(neutral_object_report.get("blockers") or []),
            "claimBoundary": neutral_object_report.get(
                "claim_boundary",
                "Neutral object candidates are extracted from observer-visible record histories without H3/S2 "
                "coordinates. They are not a neutral 3D placement unless strict neutral receipts pass.",
            ),
        },
        "h3ChartStatus": h3_chart_status,
        "receiptDisplay": h3_chart_status["receiptDisplay"],
        "receipts": receipts,
        "strictNeutralBlockers": readout.get("strict_neutral_blockers", []),
        "claimBoundary": readout.get(
            "claim_boundary",
            "Theorem-assisted H3 chart visualization; not chart-blind strict neutral quotient bulk, matter particles, "
            "or physical CMB prediction.",
        ),
    }


def _h3_chart_status(
    *,
    object_count: int,
    neutral_object_count: int,
    proto_worldline_count: int,
    receipts: dict[str, Any],
) -> dict[str, Any]:
    renderable = object_count > 0 or proto_worldline_count > 0 or neutral_object_count > 0
    entries = [
        _receipt_display_entry(
            "observer_h3_object_population_receipt",
            receipts.get("observer_h3_object_population_receipt"),
            label="observer H3 object population",
            claim_level="observer_facing_chart",
            false_status="missing_data",
            false_meaning="No observer-consensus H3 object packets were exported.",
            render_as_error=not renderable,
        ),
        _receipt_display_entry(
            "theorem_assisted_consensus_3d_bulk_readout_receipt",
            receipts.get("theorem_assisted_consensus_3d_bulk_readout_receipt"),
            label="theorem-assisted H3 consensus readout",
            claim_level="observer_facing_chart",
            false_status="blocked",
            false_meaning="The observer-facing theorem-assisted H3 readout did not pass.",
            render_as_error=not renderable,
        ),
        _receipt_display_entry(
            "strict_neutral_third_person_bulk_receipt",
            receipts.get("strict_neutral_third_person_bulk_receipt"),
            label="strict neutral third-person bulk",
            claim_level="promotion_gate",
            false_status="not_promoted",
            false_meaning=(
                "Expected closed gate for this viewer: H3 chart data can render, but chart-blind "
                "neutral 3D bulk has not been established."
            ),
            render_as_error=False,
        ),
        _receipt_display_entry(
            "strict_neutral_object_bulk_receipt",
            receipts.get("strict_neutral_object_bulk_receipt"),
            label="strict neutral object bulk",
            claim_level="promotion_gate",
            false_status="not_promoted",
            false_meaning="Neutral object candidates exist only as candidates until strict object controls pass.",
            render_as_error=False,
        ),
        _receipt_display_entry(
            "physical_cmb_prediction_receipt",
            receipts.get("physical_cmb_prediction_receipt"),
            label="physical CMB prediction",
            claim_level="promotion_gate",
            false_status="not_promoted",
            false_meaning="CMB rows are diagnostics/comparisons, not a promoted physical prediction.",
            render_as_error=False,
        ),
    ]
    status = "available" if renderable else "empty"
    if renderable and not bool(receipts.get("observer_h3_object_population_receipt", False)):
        status = "diagnostic_only"
    return {
        "renderable": bool(renderable),
        "displayStatus": status,
        "objectCount": int(object_count),
        "neutralObjectCandidateCount": int(neutral_object_count),
        "protoWorldlineCount": int(proto_worldline_count),
        "falseReceiptPolicy": (
            "False promotion receipts are blocked/not-promoted states, not H3 chart rendering errors. "
            "Only missing observer-facing H3 population with no renderable objects should be treated as an error."
        ),
        "receiptDisplay": {entry["id"]: entry for entry in entries},
    }


def _receipt_display_entry(
    receipt_id: str,
    passed: Any,
    *,
    label: str,
    claim_level: str,
    false_status: str,
    false_meaning: str,
    render_as_error: bool,
) -> dict[str, Any]:
    ok = bool(passed)
    if ok:
        return {
            "id": receipt_id,
            "label": label,
            "passed": True,
            "displayStatus": "passed",
            "severity": "pass",
            "claimLevel": claim_level,
            "renderAsError": False,
        }
    return {
        "id": receipt_id,
        "label": label,
        "passed": False,
        "displayStatus": false_status,
        "severity": "error" if render_as_error else "blocked",
        "claimLevel": claim_level,
        "falseMeaning": false_meaning,
        "renderAsError": bool(render_as_error),
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


def _read_neutral_object_candidates(consensus_pack_dir: Path, *, max_objects: int) -> list[dict[str, Any]]:
    path = consensus_pack_dir / "neutral_objects.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= int(max_objects):
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            observer_ids = row.get("observer_ids") if isinstance(row.get("observer_ids"), list) else []
            parsed_observer_ids: list[int] = []
            for value in observer_ids:
                try:
                    parsed_observer_ids.append(int(value))
                except (TypeError, ValueError):
                    continue
            rows.append(
                {
                    "objectId": row.get("object_id"),
                    "observerIds": parsed_observer_ids,
                    "observerCount": len(parsed_observer_ids),
                    "visibleSignatureKey": row.get("visible_signature_key"),
                    "persistence": _optional_float(row.get("persistence")),
                    "overlapAgreement": _optional_float(row.get("overlap_agreement")),
                    "spatialEmbeddingAvailable": False,
                    "primaryFeatures": [
                        "record_lineage_hist",
                        "checkpoint_continuation_hist",
                        "sector_transport_hist",
                        "repair_response_hist",
                        "counterfactual_stability_hist",
                        "transition_affinity_hist",
                    ],
                    "claimBoundary": (
                        "Neutral-object candidate from observer-visible record/transition histories. "
                        "No H3/S2/support coordinate is attached here; render as an audit/network row, "
                        "not a neutral 3D point."
                    ),
                }
            )
    return rows


def _read_proto_particle_candidates(consensus_pack_dir: Path, *, max_worldlines: int) -> dict[str, Any]:
    worldline_report = _read_json(consensus_pack_dir / "defect_h3_worldlines_report.json")
    particle_report = _read_json(consensus_pack_dir / "particle_likeness_report.json")
    particle_rows = {
        str(row.get("worldline_id")): row
        for row in particle_report.get("worldlines", [])
        if isinstance(row, dict) and row.get("worldline_id") is not None
    }
    worldlines = _worldlines_from_defect_report(worldline_report, particle_rows, max_worldlines=max_worldlines)
    source = "defect_h3_worldlines_report"
    organic_worldlines = _worldlines_from_organic_defect_population(
        consensus_pack_dir, max_worldlines=max_worldlines
    )
    free_worldlines = _worldlines_from_free_two_defect_dynamics(consensus_pack_dir, max_worldlines=max_worldlines)
    stress_worldlines = _worldlines_from_two_defect_assay(consensus_pack_dir, max_worldlines=max_worldlines)
    holonomy_worldlines = _worldlines_from_array_holonomy(consensus_pack_dir, max_worldlines=max_worldlines)
    csv_worldlines = _worldlines_from_proto_particle_csv(consensus_pack_dir, max_worldlines=max_worldlines)
    if not worldlines:
        if organic_worldlines:
            worldlines = organic_worldlines
            source = "organic_defect_population_report"
    if not worldlines:
        if free_worldlines:
            worldlines = free_worldlines
            source = "free_two_defect_dynamics_report"
    if not worldlines:
        if stress_worldlines:
            worldlines = stress_worldlines
            source = "two_defect_stress_contraction_assay_report"
    if not worldlines:
        if holonomy_worldlines:
            worldlines = holonomy_worldlines
            source = "array_holonomy_report_cluster_births"
    if not worldlines and csv_worldlines:
        worldlines = csv_worldlines
        source = "proto_particle_worldline_csv_sidecars_legacy_fallback"
    organic_report = _read_json(consensus_pack_dir / "organic_defect_population_report.json")
    free_report = _read_json(consensus_pack_dir / "free_two_defect_dynamics_report.json")
    return {
        "description": (
            "Holonomy/defect worldlines fitted into the same derived H3 chart. These are the right "
            "visual layer for proto-particles in the current simulator, but they are not matter particles "
            "until localization, transport, fusion/scattering, repeated-seed, and neutral-bulk gates pass."
        ),
        "worldlines": worldlines,
        "worldlineSource": source if worldlines else "none",
        "sourcePriority": [
            "defect_h3_worldlines_report",
            "organic_defect_population_report",
            "free_two_defect_dynamics_report",
            "two_defect_stress_contraction_assay_report",
            "array_holonomy_report_cluster_births",
            "proto_particle_worldline_csv_sidecars_legacy_fallback",
        ],
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
            "diagnostic_edge_worldline_count": int(
                0 if source == "defect_h3_worldlines_report" and worldlines else len(worldlines)
            ),
            "csv_proto_worldline_count": len(csv_worldlines),
            "organic_defect_worldline_count": len(organic_worldlines),
            "organic_defect_population_receipt": bool(
                organic_report.get("organic_defect_population_receipt", False)
            ),
            "free_two_defect_worldline_count": len(free_worldlines),
            "free_two_defect_dynamics_receipt": bool(
                free_report.get("free_two_defect_dynamics_receipt", False)
            ),
            "controlled_two_defect_worldline_count": len(stress_worldlines),
            "screen_holonomy_cluster_worldline_count": len(holonomy_worldlines),
            "worldline_report_supplied": bool(worldline_report),
            "legacy_csv_sidecar_used": bool(source == "proto_particle_worldline_csv_sidecars_legacy_fallback"),
        },
        "claimBoundary": worldline_report.get(
            "claim_boundary",
            (
                "No primary H3 holonomy worldline report was supplied. Exported worldlines, if any, "
                "come from diagnostic sidecars such as controlled two-defect stress contraction or "
                "screen holonomy clusters and are not particle-matter evidence."
            ),
        ),
    }


def _worldlines_from_defect_report(
    worldline_report: dict[str, Any],
    particle_rows: dict[str, dict[str, Any]],
    *,
    max_worldlines: int,
) -> list[dict[str, Any]]:
    worldlines = []
    for row in list(worldline_report.get("worldlines", []))[:max_worldlines] if worldline_report else []:
        if not isinstance(row, dict):
            continue
        events = _compact_h3_events(row.get("events", []), limit=64)
        if not events:
            continue
        worldline_id = str(row.get("worldline_id") or f"worldline_{len(worldlines):06d}")
        particle = particle_rows.get(worldline_id, {})
        path_length, mean_step = _h3_event_path_metrics(events)
        worldlines.append(
            {
                "worldlineId": worldline_id,
                "observationCount": row.get("observation_count", len(events)),
                "birthCycle": row.get("birth_cycle"),
                "deathCycle": row.get("death_cycle"),
                "h3PathLength": row.get("h3_path_length", path_length),
                "meanH3Step": row.get("mean_h3_step", mean_step),
                "classMode": row.get("class_mode"),
                "events": events,
                "particleLike": bool(particle.get("particle_like", False)),
                "localizationPass": bool(particle.get("localization_pass", False)),
                "persistencePass": bool(particle.get("persistence_pass", False)),
                "sectorStabilityPass": bool(particle.get("sector_stability_pass", False)),
                "transportabilityPass": bool(particle.get("transportability_pass", False)),
                "bulkLocalizationPass": bool(particle.get("bulk_localization_pass", False)),
                "worldlineSource": "defect_h3_worldlines_report",
                "claimBoundary": (
                    "H3-fitted screen/collar holonomy defect worldline. This is a proto-particle "
                    "candidate only unless particle_matter_receipt is true."
                ),
            }
        )
    return worldlines


def _worldlines_from_proto_particle_csv(consensus_pack_dir: Path, *, max_worldlines: int) -> list[dict[str, Any]]:
    worldline_path = consensus_pack_dir / "proto_particle_worldlines.csv"
    event_path = consensus_pack_dir / "proto_particle_worldline_events.csv"
    if not worldline_path.exists() or not event_path.exists():
        return []
    event_rows_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with event_path.open("r", encoding="utf-8", newline="") as handle:
        for event in csv.DictReader(handle):
            worldline_id = str(event.get("worldline_id") or event.get("worldlineId") or "")
            if worldline_id:
                event_rows_by_id[worldline_id].append(dict(event))
    worldlines = []
    with worldline_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if len(worldlines) >= int(max_worldlines):
                break
            worldline_id = str(row.get("worldline_id") or row.get("worldlineId") or f"csv_worldline_{len(worldlines):06d}")
            events = _compact_h3_events(event_rows_by_id.get(worldline_id, []), limit=64)
            if not events:
                continue
            path_length, mean_step = _h3_event_path_metrics(events)
            worldlines.append(
                {
                    "worldlineId": worldline_id,
                    "observationCount": _safe_int(row.get("observation_count"), len(events)),
                    "birthCycle": _optional_float(row.get("birth_cycle")),
                    "deathCycle": _optional_float(row.get("death_cycle")),
                    "h3PathLength": _optional_float(row.get("h3_path_length")) or path_length,
                    "meanH3Step": _optional_float(row.get("mean_h3_step")) or mean_step,
                    "classMode": row.get("class_mode") or row.get("classMode"),
                    "events": events,
                    "particleLike": str(row.get("particle_like", "")).lower() == "true",
                    "localizationPass": str(row.get("localization_pass", "")).lower() == "true",
                    "persistencePass": str(row.get("persistence_pass", "")).lower() == "true",
                    "sectorStabilityPass": str(row.get("sector_stability_pass", "")).lower() == "true",
                    "transportabilityPass": str(row.get("transportability_pass", "")).lower() == "true",
                    "bulkLocalizationPass": str(row.get("bulk_localization_pass", "")).lower() == "true",
                    "diagnosticOnly": True,
                    "worldlineSource": "proto_particle_worldline_csv_sidecars",
                    "claimBoundary": (
                        "Worldline reconstructed from visualization CSV sidecars. It is retained for "
                        "renderer continuity and does not promote a particle receipt."
                    ),
                }
            )
    return worldlines


def _worldlines_from_organic_defect_population(consensus_pack_dir: Path, *, max_worldlines: int) -> list[dict[str, Any]]:
    report = _read_json(consensus_pack_dir / "organic_defect_population_report.json")
    summary = report.get("organic_population_summary") if isinstance(report.get("organic_population_summary"), dict) else {}
    worldlines = []
    for row in list(report.get("worldlines", []))[:max_worldlines] if report else []:
        if not isinstance(row, dict):
            continue
        events = _compact_h3_events(row.get("events", []), limit=128)
        if not events:
            continue
        path_length, mean_step = _h3_event_path_metrics(events)
        worldline_id = str(row.get("worldline_id") or row.get("worldlineId") or f"organic_defect_{len(worldlines):02d}")
        worldlines.append(
            {
                "worldlineId": worldline_id,
                "observationCount": row.get("observation_count", len(events)),
                "birthCycle": row.get("birth_cycle", events[0].get("cycle")),
                "deathCycle": row.get("death_cycle", events[-1].get("cycle")),
                "h3PathLength": row.get("h3_path_length", path_length),
                "meanH3Step": row.get("mean_h3_step", mean_step),
                "classMode": row.get("class_mode") or events[0].get("class"),
                "holonomyMode": row.get("holonomy_mode") or events[0].get("holonomyMode"),
                "events": events,
                "particleLike": False,
                "localizationPass": False,
                "persistencePass": bool(row.get("persistent", False)),
                "sectorStabilityPass": False,
                "transportabilityPass": bool(path_length > 0.0),
                "bulkLocalizationPass": False,
                "diagnosticOnly": True,
                "controlledPlantedAssay": False,
                "organicDefectPopulationDiagnostic": bool(
                    report.get("organic_defect_population_diagnostic", False)
                ),
                "organicPopulationWorldlineCount": summary.get("worldline_count"),
                "worldlineSource": "organic_defect_population_report",
                "renderModes": ["h3_point", "edge_string", "subjective_observer_3d_point"],
                "claimBoundary": (
                    "Organic multi-defect diagnostic worldline. Births, H3 positions, holonomies, "
                    "and transverse motion are seeded by the repair-hotspot diagnostic law, not a fixed "
                    "left/right pair. This supports natural proto-worldline/string/observer renderings "
                    "but is not particle matter or production gravity."
                ),
            }
        )
    return worldlines


def _worldlines_from_two_defect_assay(consensus_pack_dir: Path, *, max_worldlines: int) -> list[dict[str, Any]]:
    report = _read_json(consensus_pack_dir / "two_defect_stress_contraction_assay_report.json")
    worldlines = []
    for row in list(report.get("worldlines", []))[:max_worldlines] if report else []:
        if not isinstance(row, dict):
            continue
        events = _compact_h3_events(row.get("events", []), limit=64)
        if not events:
            continue
        path_length, mean_step = _h3_event_path_metrics(events)
        worldline_id = str(row.get("worldline_id") or row.get("worldlineId") or f"stress_worldline_{len(worldlines):06d}")
        worldlines.append(
            {
                "worldlineId": worldline_id,
                "observationCount": row.get("observation_count", len(events)),
                "birthCycle": row.get("birth_cycle", events[0].get("cycle")),
                "deathCycle": row.get("death_cycle", events[-1].get("cycle")),
                "h3PathLength": row.get("h3_path_length", path_length),
                "meanH3Step": row.get("mean_h3_step", mean_step),
                "classMode": row.get("class_mode") or events[0].get("class"),
                "events": events,
                "particleLike": False,
                "localizationPass": False,
                "persistencePass": bool(row.get("persistent", False)),
                "sectorStabilityPass": False,
                "transportabilityPass": False,
                "bulkLocalizationPass": False,
                "diagnosticOnly": True,
                "controlledPlantedAssay": bool(report.get("controlled_planted_assay", False)),
                "worldlineSource": "two_defect_stress_contraction_assay_report",
                "claimBoundary": (
                    "Controlled two-defect stress-contraction diagnostic worldline. It gives the "
                    "effective string/worldsheet renderer edge motion, but it is not spontaneous "
                    "particle formation or particle matter evidence."
                ),
            }
        )
    return worldlines


def _worldlines_from_free_two_defect_dynamics(consensus_pack_dir: Path, *, max_worldlines: int) -> list[dict[str, Any]]:
    report = _read_json(consensus_pack_dir / "free_two_defect_dynamics_report.json")
    summary = report.get("free_dynamics_summary") if isinstance(report.get("free_dynamics_summary"), dict) else {}
    worldlines = []
    for row in list(report.get("worldlines", []))[:max_worldlines] if report else []:
        if not isinstance(row, dict):
            continue
        events = _compact_h3_events(row.get("events", []), limit=96)
        if not events:
            continue
        path_length, mean_step = _h3_event_path_metrics(events)
        worldline_id = str(row.get("worldline_id") or row.get("worldlineId") or f"free_worldline_{len(worldlines):06d}")
        worldlines.append(
            {
                "worldlineId": worldline_id,
                "observationCount": row.get("observation_count", len(events)),
                "birthCycle": row.get("birth_cycle", events[0].get("cycle")),
                "deathCycle": row.get("death_cycle", events[-1].get("cycle")),
                "h3PathLength": row.get("h3_path_length", path_length),
                "meanH3Step": row.get("mean_h3_step", mean_step),
                "classMode": row.get("class_mode") or events[0].get("class"),
                "events": events,
                "particleLike": False,
                "localizationPass": False,
                "persistencePass": bool(row.get("persistent", False)),
                "sectorStabilityPass": False,
                "transportabilityPass": bool(path_length > 0.0),
                "bulkLocalizationPass": False,
                "diagnosticOnly": True,
                "controlledPlantedAssay": False,
                "freeDynamicsDiagnostic": bool(report.get("free_dynamics_diagnostic", False)),
                "contactOutcome": row.get("contact_outcome") or row.get("contactOutcome") or summary.get("contact_outcome"),
                "worldlineSource": "free_two_defect_dynamics_report",
                "claimBoundary": (
                    "Free randomized two-defect dynamics diagnostic worldline. Positions are randomized "
                    "in the H3 tangent chart, transverse kicks are included, and contact bookkeeping is "
                    "exported; this is not particle matter, production gravity, or a physical merger claim."
                ),
            }
        )
    return worldlines


def _worldlines_from_array_holonomy(consensus_pack_dir: Path, *, max_worldlines: int) -> list[dict[str, Any]]:
    report = _read_json(consensus_pack_dir / "array_holonomy_report.json")
    if not report:
        return []
    clusters = [row for row in report.get("clusters", []) if isinstance(row, dict)]
    clusters_by_id = {
        str(row.get("cluster_id")): row
        for row in clusters
        if row.get("cluster_id") is not None
    }
    source_rows = [row for row in report.get("worldlines", []) if isinstance(row, dict)]
    if not source_rows:
        source_rows = clusters
    worldlines = []
    for row in source_rows[:max_worldlines]:
        cluster_id = row.get("current_cluster_id", row.get("cluster_id"))
        cluster = clusters_by_id.get(str(cluster_id), row)
        events = _compact_h3_events(row.get("events", []), limit=64)
        if not events:
            point = _point_from_any_h3_event(cluster) or _point_from_any_h3_event(row)
            if point is None:
                continue
            events = [
                {
                    "cycle": row.get("birth_cycle", row.get("cycle")),
                    "h3SpatialPoint": point,
                    "fitResidual": row.get("fit_residual"),
                    "supportNodeCount": cluster.get("support_node_count", row.get("support_node_count")),
                    "event": "screen_holonomy_cluster",
                    "class": cluster.get("class", row.get("class")),
                    "holonomyMode": cluster.get("holonomy_mode", row.get("holonomy_mode")),
                }
            ]
        path_length, mean_step = _h3_event_path_metrics(events)
        worldline_id = str(
            row.get("worldline_id")
            or row.get("worldlineId")
            or row.get("current_cluster_id")
            or row.get("cluster_id")
            or f"screen_holonomy_cluster_{len(worldlines):06d}"
        )
        worldlines.append(
            {
                "worldlineId": worldline_id,
                "observationCount": row.get("observation_count", len(events)),
                "birthCycle": row.get("birth_cycle", events[0].get("cycle")),
                "deathCycle": row.get("death_cycle", events[-1].get("cycle")),
                "h3PathLength": row.get("h3_path_length", path_length),
                "meanH3Step": row.get("mean_h3_step", mean_step),
                "classMode": row.get("class_mode") or events[0].get("class"),
                "events": events,
                "particleLike": False,
                "localizationPass": False,
                "persistencePass": bool(row.get("persistent", len(events) > 1)),
                "sectorStabilityPass": False,
                "transportabilityPass": False,
                "bulkLocalizationPass": False,
                "diagnosticOnly": True,
                "worldlineSource": "array_holonomy_report_cluster_births",
                "claimBoundary": (
                    "Screen holonomy cluster exported as a diagnostic edge/string track because no "
                    "primary H3 defect worldline report was supplied. It is not particle matter evidence."
                ),
            }
        )
    return worldlines


def _compact_h3_events(value: Any, *, limit: int) -> list[dict[str, Any]]:
    events = []
    for event in list(value or [])[:limit] if isinstance(value, list) else []:
        if not isinstance(event, dict):
            continue
        point = _point_from_any_h3_event(event)
        if point is None:
            continue
        events.append(
            {
                "cycle": _optional_float(event.get("cycle")),
                "h3SpatialPoint": point,
                "fitResidual": _optional_float(event.get("fit_residual", event.get("fitResidual"))),
                "supportNodeCount": _safe_int(
                    event.get("support_node_count", event.get("supportNodeCount")), 0
                ),
                "event": event.get("event"),
                "class": event.get("class"),
                "holonomyMode": event.get("holonomy_mode", event.get("holonomyMode")),
                "velocity": event.get("velocity"),
                "pairH3Separation": _optional_float(event.get("pair_h3_separation", event.get("pairH3Separation"))),
                "localReadoutContraction": _optional_float(
                    event.get("local_readout_contraction", event.get("localReadoutContraction"))
                ),
                "supportOverlapFraction": _optional_float(
                    event.get("support_overlap_fraction", event.get("supportOverlapFraction"))
                ),
                "supportOverlapNodeCount": _safe_int(
                    event.get("support_overlap_node_count", event.get("supportOverlapNodeCount")), 0
                ),
                "localStressDensity": _optional_float(
                    event.get("local_stress_density", event.get("localStressDensity"))
                ),
                "nearestDefectId": event.get("nearest_defect_id", event.get("nearestDefectId")),
                "nearestH3Separation": _optional_float(
                    event.get("nearest_h3_separation", event.get("nearestH3Separation"))
                ),
                "contactOutcome": event.get("contact_outcome", event.get("contactOutcome")),
                "chargeConservationPass": event.get(
                    "charge_conservation_pass", event.get("chargeConservationPass")
                ),
                "transportDistance": _optional_float(event.get("transport_distance", event.get("transportDistance"))),
                "renderModes": event.get("render_modes", event.get("renderModes", [])),
            }
        )
    return events


def _point_from_any_h3_event(row: dict[str, Any]) -> list[float] | None:
    for key in ("h3_spatial_point", "h3SpatialPoint", "centroid", "point"):
        value = row.get(key)
        if isinstance(value, list) and len(value) >= 3:
            try:
                return [float(value[0]), float(value[1]), float(value[2])]
            except (TypeError, ValueError):
                return None
    if {"h3_x", "h3_y", "h3_z"}.issubset(row):
        try:
            return [float(row["h3_x"]), float(row["h3_y"]), float(row["h3_z"])]
        except (TypeError, ValueError):
            return None
    if {"x", "y", "z"}.issubset(row):
        try:
            return [float(row["x"]), float(row["y"]), float(row["z"])]
        except (TypeError, ValueError):
            return None
    return None


def _h3_event_path_metrics(events: list[dict[str, Any]]) -> tuple[float, float]:
    points = [event.get("h3SpatialPoint") for event in events if isinstance(event.get("h3SpatialPoint"), list)]
    if len(points) < 2:
        return 0.0, 0.0
    total = 0.0
    for left, right in zip(points, points[1:], strict=False):
        if len(left) < 3 or len(right) < 3:
            continue
        total += math.sqrt(sum((float(left[index]) - float(right[index])) ** 2 for index in range(3)))
    return float(total), float(total / max(len(points) - 1, 1))


def _cmb_payload(consensus_pack_dir: Path | None) -> dict[str, Any]:
    if consensus_pack_dir is None:
        return {
            "description": "No CMB comparison pack supplied.",
            "receipts": {},
            "residualRows": [],
            "observedRows": [],
            "modelRows": [],
        }
    report = _read_json(consensus_pack_dir / "physical_cmb_output_comparison_report.json")
    screen_rows, screen_model = _screen_cmb_diagnostic_rows(consensus_pack_dir)
    if not report:
        cmb_lite = _read_json(consensus_pack_dir / "cmb_lite_comparison_report.json")
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
                "SCREEN_CMB_LITE_DIAGNOSTIC_RECEIPT": bool(
                    cmb_lite or screen_rows or screen_model.get("screenProxyReceipt", False)
                ),
            },
            "bestOphDiagnosticModel": screen_model,
            "bestOphResidualSummary": {},
            "residualRows": [],
            "observedRows": [],
            "modelRows": [],
            "screenDiagnosticSpectrumRows": screen_rows,
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
                "observed": _optional_float(row.get("observed", row.get("observed_D_ell"))),
                "model": _optional_float(row.get("model", row.get("model_D_ell"))),
                "residualSigma": _optional_float(row.get("residual_sigma")),
            }
        )
    observed_rows = [{"ell": row["ell"], "D_ell": row["observed"]} for row in rows if row.get("observed") is not None]
    model_rows = [{"ell": row["ell"], "D_ell": row["model"]} for row in rows if row.get("model") is not None]
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
        "screenDiagnosticModel": screen_model,
        "bestOphResidualSummary": residual_summary,
        "residualRows": rows,
        "observedRows": observed_rows,
        "modelRows": model_rows,
        "screenDiagnosticSpectrumRows": screen_rows,
        "claimBoundary": report.get(
            "claim_boundary",
            "CMB output comparison diagnostic; not a physical CMB prediction without hard gates.",
        ),
    }


def _screen_cmb_diagnostic_rows(consensus_pack_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cmb_lite = _read_json(consensus_pack_dir / "cmb_lite_comparison_report.json")
    cl_report = _read_json(consensus_pack_dir / "cl_comparison_report.json")
    fields = cl_report.get("fields", {}) if isinstance(cl_report.get("fields"), dict) else {}
    best_field = (
        cmb_lite.get("best_positive_shape_field")
        or cmb_lite.get("best_shape_field")
        or cl_report.get("best_shape_field")
    )
    field_report = fields.get(best_field) if isinstance(best_field, str) else None
    if field_report is None and fields:
        best_field, field_report = next(iter(fields.items()))
    spectrum = field_report.get("spectrum", []) if isinstance(field_report, dict) else []
    d_values = [
        float(value)
        for row in spectrum
        for value in [_optional_float(row.get("D_ell")) if isinstance(row, dict) else None]
        if value is not None and math.isfinite(float(value))
    ]
    max_abs = max((abs(value) for value in d_values), default=0.0)
    rows = []
    for row in list(spectrum)[:320]:
        if not isinstance(row, dict):
            continue
        d_ell = _optional_float(row.get("D_ell"))
        rows.append(
            {
                "field": best_field,
                "ell": _optional_float(row.get("ell")),
                "C_ell": _optional_float(row.get("C_ell")),
                "D_ell": d_ell,
                "normalizedD_ell": float(d_ell / max_abs) if d_ell is not None and max_abs > 0.0 else None,
            }
        )
    model = {
        "source": "cmb_lite_screen_proxy",
        "field": best_field,
        "rowCount": len(rows),
        "ellMax": cl_report.get("ell_max"),
        "pointCount": cl_report.get("point_count"),
        "screenProxyReceipt": bool(
            (cl_report.get("cosmo_proxy_receipt") or {}).get("receipt", False)
            or cl_report.get("receipt_name") == "SCREEN_PROXY_CMB_RECEIPT"
            or cmb_lite
        ),
        "physicalPrediction": False,
        "claimBoundary": "Screen angular-spectrum diagnostic; not physical CMB prediction.",
    }
    return rows, model


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


def _visualization_render_data_payload(
    *,
    small_payload: dict[str, Any],
    screen_payload: dict[str, Any],
    observer_payload: dict[str, Any],
    bulk_payload: dict[str, Any],
    cmb_payload: dict[str, Any],
    pn_silence_payload: dict[str, Any],
    subjective_cameras: list[dict[str, Any]],
    visualization_views: dict[str, Any],
    curved_spacetime_payload: dict[str, Any],
) -> dict[str, Any]:
    observer_graph = _render_observer_graph(observer_payload)
    screen_scene = _render_screen_scene(screen_payload)
    bulk_scene = _render_bulk_scene(bulk_payload)
    timeline = _render_animation_timeline(
        small_payload=small_payload,
        screen_scene=screen_scene,
        observer_payload=observer_payload,
        bulk_scene=bulk_scene,
    )
    plot_series = _render_plot_series(cmb_payload, pn_silence_payload, curved_spacetime_payload)
    claim_badges = _render_claim_badges(
        small_payload=small_payload,
        observer_payload=observer_payload,
        bulk_payload=bulk_payload,
        cmb_payload=cmb_payload,
        pn_silence_payload=pn_silence_payload,
        visualization_views=visualization_views,
    )
    availability = {
        "screenPointCount": len(screen_scene["points"]),
        "screenClusterTrackCount": len(screen_scene["clusterTracks"]),
        "observerNodeCount": len(observer_graph["nodes"]),
        "observerOverlapLinkCount": len(observer_graph["links"]),
        "subjectiveCameraCount": len(subjective_cameras),
        "h3ObjectCount": len(bulk_scene["h3Objects"]),
        "neutralObjectCandidateCount": len(bulk_payload.get("neutralObjectCandidates", []) or []),
        "protoWorldlineCount": len(bulk_scene["protoWorldlines"]),
        "protoWorldlineEventCount": sum(len(row.get("events", [])) for row in bulk_scene["protoWorldlines"]),
        "curvatureProxyPointCount": len(curved_spacetime_payload.get("curvatureProxyPoints", []) or []),
        "curvatureProxyTimeSliceCount": len(curved_spacetime_payload.get("timeSlices", []) or []),
        "continuousBulkFieldVolumeSampleCount": len(
            (
                curved_spacetime_payload.get("continuousBulkField", {})
                if isinstance(curved_spacetime_payload.get("continuousBulkField"), dict)
                else {}
            ).get("volumeSamples", [])
            or []
        ),
        "continuousBulkFieldSliceSampleCount": len(
            (
                curved_spacetime_payload.get("continuousBulkField", {})
                if isinstance(curved_spacetime_payload.get("continuousBulkField"), dict)
                else {}
            ).get("sliceSamples", [])
            or []
        ),
        "observerProtoWorldlineSightingCount": sum(
            len(frame.get("visibleProtoWorldlines", []))
            for camera in subjective_cameras
            if isinstance(camera, dict)
            for frame in camera.get("timeFrames", [])
            if isinstance(frame, dict)
        ),
        "cmbResidualPointCount": len(plot_series["cmbResidualSigma"]),
        "screenSpectrumPointCount": len(plot_series["screenSpectrum"]),
        "timelineFrameCount": len(timeline),
    }
    return {
        "schema": "oph_visualization_render_data_v1",
        "description": (
            "Renderer-ready derivatives of visualization_payload.json: camera presets, scene graph, "
            "animation timeline, plot series, legends, and claim badges. This is convenience data for "
            "visual frontends; source payload receipts remain authoritative."
        ),
        "availability": availability,
        "rendererHints": {
            "defaultView": "observerCamera" if subjective_cameras else "fluctuatingQuantumVacuum",
            "recommendedRepairTimeField": "playbackRelativeTime",
            "rawRepairTimeField": "rawRelativeTime",
            "repairPlaybackPolicy": (
                "Use playbackRelativeTime for gradual UI animation and rawRelativeTime/cycle for "
                "auditable repair timing."
            ),
            "viewOrder": [
                "fluctuatingQuantumVacuum",
                "observerCamera",
                "emergentCurvedSpacetime",
                "effectiveStringTheory",
                "silenceToObservation",
                "cmbComparison",
                "consensusBulk",
            ],
            "preferredFrameRate": 24,
            "coordinateSystems": [
                {
                    "id": "screen_s2",
                    "kind": "unit_sphere_boundary",
                    "source": "screen.points",
                    "claimBoundary": "observer-facing screen chart",
                },
                {
                    "id": "h3_chart",
                    "kind": "derived_h3_object_chart",
                    "source": "consensusBulk.objects + protoParticleCandidates.worldlines",
                    "claimBoundary": "observer-facing H3 chart, not strict neutral bulk",
                },
                {
                    "id": "curvature_proxy_h3_chart",
                    "kind": "diagnostic_curvature_stress_proxy",
                    "source": (
                        "emergentCurvedSpacetime.curvatureProxyPoints + "
                        "emergentCurvedSpacetime.continuousBulkField"
                    ),
                    "claimBoundary": "diagnostic proxy over the observer-facing H3 chart, not an Einstein metric",
                },
                {
                    "id": "observer_local_camera",
                    "kind": "subjective_readout_camera",
                    "source": "subjectiveObserverCameras",
                    "claimBoundary": "visible observer-local readout camera",
                },
            ],
            "missingDataPolicy": (
                "Hide unavailable layers and show their claim badge as absent or blocked. Never infer a "
                "passing receipt from visual smoothness."
            ),
        },
        "cameraPresets": _render_camera_presets(subjective_cameras, bulk_scene, screen_scene),
        "repairPlayback": _repair_playback_summary(timeline),
        "sceneGraph": {
            "screen": screen_scene,
            "observerGraph": observer_graph,
            "bulk": bulk_scene,
            "curvedSpacetime": _render_curved_spacetime_scene(curved_spacetime_payload),
            "finiteRepairGraph": _render_finite_repair_graph(small_payload),
        },
        "animationTimeline": timeline,
        "plotSeries": plot_series,
        "legend": _render_legend(),
        "claimBadges": claim_badges,
        "viewContracts": {
            key: {
                "label": value.get("label"),
                "sectionKind": value.get("sectionKind"),
                "renderLayers": value.get("renderLayers", []),
                "animationChannels": value.get("animationChannels", []),
                "nonClaims": value.get("nonClaims", []),
                "claimBoundary": value.get("claimBoundary"),
            }
            for key, value in visualization_views.items()
            if isinstance(value, dict)
        },
        "claimBoundary": (
            "Render-data convenience layer only. It does not promote strict neutral bulk, physical CMB, "
            "particle matter, literal QFT vacuum, or critical string CFT without the corresponding receipts."
        ),
    }


def _render_observer_graph(observer_payload: dict[str, Any], *, max_nodes: int = 512, max_links: int = 4096) -> dict[str, Any]:
    nodes = []
    for row in list(observer_payload.get("observers", []) or [])[:max_nodes]:
        if not isinstance(row, dict):
            continue
        axis = _coord3(row.get("axis"))
        if axis is None:
            continue
        nodes.append(
            {
                "id": row.get("observerId"),
                "position": axis,
                "supportPatchCount": row.get("supportPatchCount"),
                "modularDepthMean": row.get("modularDepthMean"),
                "visibleSignatureEntropy": row.get("visibleSignatureEntropy"),
                "dominantRecordSignature": row.get("dominantRecordSignature"),
                "dominantObjectPacket": row.get("dominantObjectPacket"),
            }
        )
    links = []
    for row in list(observer_payload.get("overlapLinks", []) or [])[:max_links]:
        if not isinstance(row, dict):
            continue
        source = row.get("sourceObserverId", row.get("source"))
        target = row.get("targetObserverId", row.get("target"))
        if source is None or target is None:
            continue
        links.append(
            {
                "id": row.get("linkId", f"{source}->{target}"),
                "source": source,
                "target": target,
                "weight": _optional_float(row.get("jaccard")) or _optional_float(row.get("overlapCommittedFraction")) or 0.0,
                "overlapPatchCount": row.get("overlapPatchCount"),
                "signatureSimilarity": row.get("signatureSimilarity"),
                "repairTrajectory": list(row.get("repairTrajectory", []) or [])[:64],
            }
        )
    return {
        "nodes": nodes,
        "links": links,
        "layout": "screen_axis_arc_graph",
        "claimBoundary": "Observer graph is built from visible supports and overlap readouts, not hidden global state.",
    }


def _render_screen_scene(screen_payload: dict[str, Any], *, max_points: int = 2048) -> dict[str, Any]:
    points = []
    values = screen_payload.get("values", []) if isinstance(screen_payload.get("values"), list) else []
    for index, point in enumerate(list(screen_payload.get("points", []) or [])[:max_points]):
        xyz = _coord3(point)
        if xyz is None:
            continue
        points.append(
            {
                "id": f"screen_point_{index:06d}",
                "position": xyz,
                "value": values[index] if index < len(values) else None,
            }
        )
    cluster_tracks = []
    snapshots = ((screen_payload.get("clusters") or {}).get("snapshots") or []) if isinstance(screen_payload.get("clusters"), dict) else []
    for snapshot_index, snapshot in enumerate(list(snapshots)[:128]):
        if not isinstance(snapshot, dict):
            continue
        for cluster in list(snapshot.get("clusters", []) or [])[:512]:
            if not isinstance(cluster, dict):
                continue
            point = _coord3(cluster.get("point"))
            if point is None:
                continue
            cluster_tracks.append(
                {
                    "id": cluster.get("clusterId", f"cluster_{len(cluster_tracks):06d}"),
                    "worldlineId": cluster.get("worldlineId"),
                    "cycle": snapshot.get("cycle"),
                    "snapshotIndex": snapshot_index,
                    "position": point,
                    "class": cluster.get("class"),
                    "supportNodeCount": cluster.get("supportNodeCount"),
                    "interpolated": bool(snapshot.get("interpolated", False)),
                }
            )
    return {
        "fieldName": screen_payload.get("fieldName"),
        "points": points,
        "clusterTracks": cluster_tracks,
        "repairTrace": list(screen_payload.get("repairTrace", []) or [])[:256],
        "claimBoundary": screen_payload.get("claimBoundary"),
    }


def _render_bulk_scene(bulk_payload: dict[str, Any], *, max_objects: int = 2048, max_worldlines: int = 512) -> dict[str, Any]:
    h3_objects = []
    for row in list(bulk_payload.get("objects", []) or [])[:max_objects]:
        if not isinstance(row, dict):
            continue
        point = _coord3([row.get("x"), row.get("y"), row.get("z")])
        if point is None:
            continue
        h3_objects.append(
            {
                "id": row.get("objectId"),
                "position": point,
                "recordFamilyId": row.get("recordFamilyId"),
                "observerCount": row.get("observerCount"),
                "supportSize": row.get("supportSize"),
                "h3Compactness": row.get("h3Compactness"),
                "h3CompactnessNormalized": row.get("h3CompactnessNormalized"),
            }
        )
    proto = bulk_payload.get("protoParticleCandidates", {}) if isinstance(bulk_payload.get("protoParticleCandidates"), dict) else {}
    worldlines = []
    for row in list(proto.get("worldlines", []) or [])[:max_worldlines]:
        if not isinstance(row, dict):
            continue
        events = []
        for event in list(row.get("events", []) or [])[:256]:
            if not isinstance(event, dict):
                continue
            point = _coord3(event.get("h3SpatialPoint"))
            if point is None:
                continue
            events.append(
                {
                    "cycle": event.get("cycle"),
                    "position": point,
                    "fitResidual": event.get("fitResidual"),
                    "supportNodeCount": event.get("supportNodeCount"),
                    "event": event.get("event"),
                    "class": event.get("class"),
                }
            )
        if not events:
            continue
        worldlines.append(
            {
                "id": row.get("worldlineId"),
                "events": events,
                "polyline": [event["position"] for event in events],
                "particleLike": bool(row.get("particleLike", False)),
                "diagnosticOnly": bool(row.get("diagnosticOnly", not row.get("particleLike", False))),
                "bulkLocalizationPass": bool(row.get("bulkLocalizationPass", False)),
                "classMode": row.get("classMode"),
                "h3PathLength": row.get("h3PathLength"),
                "meanH3Step": row.get("meanH3Step"),
                "worldlineSource": row.get("worldlineSource", proto.get("worldlineSource")),
            }
        )
    return {
        "h3Objects": h3_objects,
        "protoWorldlines": worldlines,
        "h3ChartStatus": bulk_payload.get("h3ChartStatus", {}),
        "receiptDisplay": bulk_payload.get("receiptDisplay", {}),
        "receipts": {**(bulk_payload.get("receipts", {}) or {}), **(proto.get("receipts", {}) or {})},
        "claimBoundary": bulk_payload.get("claimBoundary"),
    }


def _render_curved_spacetime_scene(curved_spacetime_payload: dict[str, Any], *, max_points: int = 4096) -> dict[str, Any]:
    proxy_points = []
    for row in list(curved_spacetime_payload.get("curvatureProxyPoints", []) or [])[:max_points]:
        if not isinstance(row, dict):
            continue
        point = _coord3(row.get("position"))
        if point is None:
            continue
        proxy_points.append(
            {
                "id": row.get("sourceId"),
                "position": point,
                "sourceKind": row.get("sourceKind"),
                "cycle": row.get("cycle"),
                "relativeTime": row.get("relativeTime"),
                "curvaturePotential": row.get("curvaturePotential"),
                "sourceDensity": row.get("sourceDensity"),
                "normalizedSourceDensity": row.get("normalizedSourceDensity"),
                "quotientVisibleSourceDensity": row.get("quotientVisibleSourceDensity"),
                "h3GreenPotential": row.get("h3GreenPotential"),
                "normalizedH3GreenPotential": row.get("normalizedH3GreenPotential"),
                "compactificationFactor": row.get("compactificationFactor"),
                "emergentSpatialScaleFactor": row.get("emergentSpatialScaleFactor"),
                "localMetricConformalFactor": row.get("localMetricConformalFactor"),
                "curvatureRadiusProxy": row.get("curvatureRadiusProxy"),
                "gravitySourceInterpretation": row.get("gravitySourceInterpretation"),
                "stressEnergyProxy": row.get("stressEnergyProxy"),
                "massProxy": row.get("massProxy"),
                "worldlineId": row.get("worldlineId"),
                "diagnosticOnly": bool(row.get("diagnosticOnly", True)),
            }
        )
    return {
        "proxyPoints": proxy_points,
        "continuousBulkField": curved_spacetime_payload.get("continuousBulkField", {}),
        "timeSlices": list(curved_spacetime_payload.get("timeSlices", []) or [])[:256],
        "spatialExtent": curved_spacetime_payload.get("spatialExtent", {}),
        "sourceMath": curved_spacetime_payload.get("sourceMath", {}),
        "receipts": curved_spacetime_payload.get("receipts", {}),
        "coordinateSystem": curved_spacetime_payload.get("coordinateSystem"),
        "claimBoundary": curved_spacetime_payload.get("claimBoundary"),
    }


def _render_finite_repair_graph(small_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": list(small_payload.get("nodes", []) or [])[:512],
        "edges": list(small_payload.get("edges", []) or [])[:2048],
        "repairFrames": list(small_payload.get("repairFrames", []) or [])[:256],
        "cycles": small_payload.get("cycles", {}),
        "claimBoundary": small_payload.get("claimBoundary"),
    }


def _render_animation_timeline(
    *,
    small_payload: dict[str, Any],
    screen_scene: dict[str, Any],
    observer_payload: dict[str, Any],
    bulk_scene: dict[str, Any],
) -> list[dict[str, Any]]:
    observer_frames = list(observer_payload.get("timeFrames", []) or [])
    if not observer_frames:
        observer_frames = list((observer_payload.get("objectiveObserverViews", [{}]) or [{}])[0].get("timeFrames", []) or [])
    repair_frames = list(small_payload.get("repairFrames", []) or [])
    screen_clusters_by_cycle: dict[Any, int] = defaultdict(int)
    for row in screen_scene.get("clusterTracks", []):
        screen_clusters_by_cycle[row.get("cycle")] += 1
    worldline_events_by_cycle: dict[Any, int] = defaultdict(int)
    for worldline in bulk_scene.get("protoWorldlines", []):
        for event in worldline.get("events", []):
            worldline_events_by_cycle[event.get("cycle")] += 1
    count = max(len(observer_frames), len(repair_frames), 1)
    timeline = []
    for index in range(min(count, 256)):
        observer_frame = observer_frames[min(index, len(observer_frames) - 1)] if observer_frames else {}
        repair_frame = repair_frames[min(index, len(repair_frames) - 1)] if repair_frames else {}
        cycle = observer_frame.get("cycle", repair_frame.get("cycle", repair_frame.get("step")))
        raw_relative_time = observer_frame.get("relativeTime", float(index / max(count - 1, 1)))
        timeline.append(
            {
                "frameIndex": index,
                "relativeTime": raw_relative_time,
                "rawRelativeTime": raw_relative_time,
                "cycle": cycle,
                "phi": observer_frame.get("globalPhi", observer_frame.get("phi", repair_frame.get("phi"))),
                "committedFraction": observer_frame.get("globalCommittedFraction", observer_frame.get("committedFraction")),
                "repairAction": repair_frame.get("action"),
                "visibleObjectPacketCount": len(observer_frame.get("visibleObjectPackets", []) or []),
                "visibleRecordPacketCount": len(observer_frame.get("visibleRecordPackets", []) or []),
                "screenClusterCount": screen_clusters_by_cycle.get(cycle, 0),
                "worldlineEventCount": worldline_events_by_cycle.get(cycle, 0),
            }
        )
    return _attach_repair_playback_times(timeline)


def _attach_repair_playback_times(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not timeline:
        return timeline
    last_index = max(1, len(timeline) - 1)
    phi_values = [_optional_float(row.get("phi")) for row in timeline]
    committed_values = [_optional_float(row.get("committedFraction")) for row in timeline]
    positive_phi = [value for value in phi_values if value is not None and value > 0.0]
    initial_phi = positive_phi[0] if positive_phi else None
    phi_end_index = 0
    if initial_phi is not None:
        threshold = max(1.0, float(initial_phi) * 0.01)
        for index, value in enumerate(phi_values):
            if value is not None and value <= threshold:
                phi_end_index = index
                break
        else:
            phi_end_index = last_index
    commit_end_index = phi_end_index
    for index, value in enumerate(committed_values):
        if value is not None and value >= 0.95:
            commit_end_index = max(index, phi_end_index)
            break
    raw_phi_end = phi_end_index / last_index
    raw_commit_end = commit_end_index / last_index
    descent_target = raw_phi_end
    if 0 < phi_end_index < last_index and raw_phi_end < 0.45:
        descent_target = 0.45
    commit_target = max(raw_commit_end, descent_target)
    if commit_end_index > phi_end_index and raw_commit_end < 0.75:
        commit_target = max(commit_target, 0.75)
    commit_target = min(0.98, max(descent_target, commit_target))
    for index, row in enumerate(timeline):
        raw_time = _optional_float(row.get("rawRelativeTime"))
        row["rawRelativeTime"] = float(index / last_index) if raw_time is None else raw_time
        if phi_end_index > 0 and index <= phi_end_index:
            playback = descent_target * float(index) / float(phi_end_index)
            phase = "mismatch_descent"
        elif commit_end_index > phi_end_index and index <= commit_end_index:
            playback = descent_target + (commit_target - descent_target) * float(index - phi_end_index) / float(
                commit_end_index - phi_end_index
            )
            phase = "record_commit"
        elif index > commit_end_index and last_index > commit_end_index:
            playback = commit_target + (1.0 - commit_target) * float(index - commit_end_index) / float(
                last_index - commit_end_index
            )
            phase = "settled"
        else:
            playback = float(index / last_index)
            phase = "raw"
        row["playbackRelativeTime"] = float(max(0.0, min(1.0, playback)))
        row["repairPlaybackPhase"] = phase
        row["timeSource"] = "cycle_synchronized_raw_trace_with_gradual_playback_time"
    return timeline


def _repair_playback_summary(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    if not timeline:
        return {
            "recommendedTimeField": "playbackRelativeTime",
            "rawTimeField": "rawRelativeTime",
            "frameCount": 0,
            "claimBoundary": "No repair timeline frames were available.",
        }
    phi_zero = next(
        (
            row
            for row in timeline
            if (_optional_float(row.get("phi")) is not None and float(_optional_float(row.get("phi")) or 0.0) <= 0.0)
        ),
        None,
    )
    committed_95 = next(
        (
            row
            for row in timeline
            if (
                _optional_float(row.get("committedFraction")) is not None
                and float(_optional_float(row.get("committedFraction")) or 0.0) >= 0.95
            )
        ),
        None,
    )
    return {
        "recommendedTimeField": "playbackRelativeTime",
        "rawTimeField": "rawRelativeTime",
        "cycleField": "cycle",
        "frameCount": len(timeline),
        "firstZeroPhiCycle": phi_zero.get("cycle") if isinstance(phi_zero, dict) else None,
        "firstZeroPhiRawRelativeTime": phi_zero.get("rawRelativeTime") if isinstance(phi_zero, dict) else None,
        "firstZeroPhiPlaybackRelativeTime": phi_zero.get("playbackRelativeTime") if isinstance(phi_zero, dict) else None,
        "firstCommitted95Cycle": committed_95.get("cycle") if isinstance(committed_95, dict) else None,
        "firstCommitted95RawRelativeTime": committed_95.get("rawRelativeTime") if isinstance(committed_95, dict) else None,
        "firstCommitted95PlaybackRelativeTime": (
            committed_95.get("playbackRelativeTime") if isinstance(committed_95, dict) else None
        ),
        "claimBoundary": (
            "playbackRelativeTime is a renderer convenience for gradual animation. rawRelativeTime, "
            "cycle, phi, and committedFraction remain the auditable simulation trace."
        ),
    }


def _render_plot_series(
    cmb_payload: dict[str, Any],
    pn_silence_payload: dict[str, Any],
    curved_spacetime_payload: dict[str, Any],
) -> dict[str, Any]:
    residual_rows = cmb_payload.get("residualRows", []) if isinstance(cmb_payload.get("residualRows"), list) else []
    screen_rows = (
        cmb_payload.get("screenDiagnosticSpectrumRows", [])
        if isinstance(cmb_payload.get("screenDiagnosticSpectrumRows"), list)
        else []
    )
    return {
        "cmbResidualSigma": [
            {
                "x": row.get("ell"),
                "y": row.get("residualSigma"),
                "observed": row.get("observed"),
                "model": row.get("model"),
            }
            for row in residual_rows
            if isinstance(row, dict)
        ],
        "screenSpectrum": [
            {
                "x": row.get("ell"),
                "y": row.get("normalizedD_ell", row.get("D_ell")),
                "field": row.get("field"),
                "C_ell": row.get("C_ell"),
                "D_ell": row.get("D_ell"),
            }
            for row in screen_rows
            if isinstance(row, dict)
        ],
        "pnClosure": [
            {"label": key, "value": value}
            for key, value in (pn_silence_payload.get("closureCoordinates", {}) or {}).items()
            if isinstance(value, (int, float, str, bool)) or value is None
        ],
        "curvatureProxyBySource": [
            {
                "x": row.get("sourceId"),
                "y": row.get("curvaturePotential"),
                "sourceKind": row.get("sourceKind"),
                "stressEnergyProxy": row.get("stressEnergyProxy"),
                "cycle": row.get("cycle"),
            }
            for row in curved_spacetime_payload.get("curvatureProxyPoints", [])
            if isinstance(row, dict)
        ],
        "curvatureProxyTimeSlices": [
            {
                "x": row.get("relativeTime"),
                "y": row.get("totalCurvaturePotential"),
                "cycle": row.get("cycle"),
                "eventCount": row.get("eventCount"),
                "sourceCount": row.get("sourceCount"),
            }
            for row in curved_spacetime_payload.get("timeSlices", [])
            if isinstance(row, dict)
        ],
        "claimBoundary": "Plot series are diagnostic render inputs unless their source receipts pass.",
    }


def _render_camera_presets(
    subjective_cameras: list[dict[str, Any]],
    bulk_scene: dict[str, Any],
    screen_scene: dict[str, Any],
) -> list[dict[str, Any]]:
    presets = [
        {
            "id": "global_orbit",
            "label": "Global orbit",
            "eye": [2.6, -3.2, 2.0],
            "lookAt": [0.0, 0.0, 0.0],
            "up": [0.0, 0.0, 1.0],
            "fovDegrees": 48.0,
        },
        {
            "id": "screen_front",
            "label": "Screen front",
            "eye": [0.0, -2.8, 0.8],
            "lookAt": [0.0, 0.0, 0.0],
            "up": [0.0, 0.0, 1.0],
            "fovDegrees": 42.0,
        },
    ]
    if bulk_scene.get("h3Objects") or bulk_scene.get("protoWorldlines"):
        presets.append(
            {
                "id": "h3_bulk",
                "label": "H3 objects/worldlines",
                "eye": [2.2, 2.0, 1.4],
                "lookAt": [0.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "fovDegrees": 50.0,
            }
        )
    if screen_scene.get("clusterTracks"):
        presets.append(
            {
                "id": "screen_defects",
                "label": "Screen defects",
                "eye": [1.8, -2.4, 1.2],
                "lookAt": [0.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "fovDegrees": 38.0,
            }
        )
    if subjective_cameras:
        first = subjective_cameras[0]
        presets.append(
            {
                "id": "first_subjective_observer",
                "label": "First observer camera",
                "eye": first.get("eye"),
                "lookAt": first.get("lookAt"),
                "up": first.get("up"),
                "fovDegrees": first.get("fovDegrees", 72.0),
                "observerId": first.get("observerId"),
            }
        )
    return presets


def _render_claim_badges(
    *,
    small_payload: dict[str, Any],
    observer_payload: dict[str, Any],
    bulk_payload: dict[str, Any],
    cmb_payload: dict[str, Any],
    pn_silence_payload: dict[str, Any],
    visualization_views: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    vacuum_view = (
        visualization_views.get("fluctuatingQuantumVacuum", {})
        if isinstance(visualization_views.get("fluctuatingQuantumVacuum"), dict)
        else {}
    )
    ym_gap = (
        vacuum_view.get("yangMillsGapCertificate", {})
        if isinstance(vacuum_view.get("yangMillsGapCertificate"), dict)
        else {}
    )
    ym_receipts = ym_gap.get("receipts") if isinstance(ym_gap.get("receipts"), dict) else {}
    sources = [
        {
            "id": "finite_consensus",
            "passed": (small_payload.get("receipts") or {}).get("FINITE_CONSENSUS_THEOREM_RECEIPT"),
            "claimLevel": "core_receipt",
            "falseDisplayStatus": "closed_gate",
            "falseMeaning": "Finite exact-consensus theorem receipt is closed or absent.",
            "renderAsError": True,
        },
        {
            "id": "observer_modular_time",
            "passed": (observer_payload.get("receipts") or {}).get("observer_modular_time_receipt"),
            "claimLevel": "observer_readout",
            "falseDisplayStatus": "closed_gate",
            "falseMeaning": "Observer modular-time readout was not established.",
            "renderAsError": True,
        },
        {
            "id": "observer_h3_object_population",
            "passed": (bulk_payload.get("receipts") or {}).get("observer_h3_object_population_receipt"),
            "claimLevel": "observer_facing_chart",
            "falseDisplayStatus": "missing_data",
            "falseMeaning": "No observer-consensus H3 object population receipt was available.",
            "renderAsError": not bool((bulk_payload.get("h3ChartStatus") or {}).get("renderable", False)),
        },
        {
            "id": "strict_neutral_bulk",
            "passed": (bulk_payload.get("receipts") or {}).get("strict_neutral_third_person_bulk_receipt"),
            "claimLevel": "promotion_gate",
            "falseDisplayStatus": "not_promoted",
            "falseMeaning": "Strict chart-blind neutral bulk has not passed; H3 chart diagnostics may still render.",
            "renderAsError": False,
        },
        {
            "id": "physical_cmb_prediction",
            "passed": (cmb_payload.get("receipts") or {}).get("PHYSICAL_CMB_PREDICTION_RECEIPT"),
            "claimLevel": "promotion_gate",
            "falseDisplayStatus": "not_promoted",
            "falseMeaning": "CMB comparison data are diagnostic, not a promoted physical CMB prediction.",
            "renderAsError": False,
        },
        {
            "id": "finite_nonabelian_gauge_gap_diagnostic",
            "passed": ym_receipts.get("finite_nonabelian_gauge_gap_diagnostic_receipt"),
            "claimLevel": "finite_gauge_diagnostic",
            "falseDisplayStatus": "missing_data",
            "falseMeaning": "No finite SU(2) compact-gauge diagnostic receipt was available.",
            "renderAsError": bool(ym_gap.get("written", False)),
        },
        {
            "id": "yang_mills_mass_gap_reproduced",
            "passed": ym_receipts.get("YANG_MILLS_GAP_REPRODUCED_RECEIPT"),
            "claimLevel": "promotion_gate",
            "falseDisplayStatus": "not_promoted",
            "falseMeaning": "Finite SU(2) diagnostics do not reproduce the continuum Yang-Mills mass gap.",
            "renderAsError": False,
        },
        {
            "id": "scale_compressed_pn_silence_to_observation",
            "passed": (pn_silence_payload.get("receipts") or {}).get("scale_compressed_pn_silence_to_observation_receipt"),
            "claimLevel": "scale_compressed_witness",
            "falseDisplayStatus": "closed_gate",
            "falseMeaning": "P/N silence-to-observation witness did not pass.",
            "renderAsError": True,
        },
    ]
    for source in sources:
        name = str(source["id"])
        passed = bool(source.get("passed"))
        display_status = "passed" if passed else str(source.get("falseDisplayStatus", "closed_gate"))
        render_as_error = bool(source.get("renderAsError", False))
        rows.append(
            {
                "id": name,
                "label": name.replace("_", " "),
                "passed": passed,
                "displayStatus": display_status,
                "claimLevel": source.get("claimLevel"),
                "severity": "pass" if passed else ("error" if render_as_error else "blocked"),
                "style": "receipt_pass" if passed else ("receipt_error" if render_as_error else "diagnostic_or_blocked"),
                "renderAsError": False if passed else render_as_error,
                "falseMeaning": None if passed else source.get("falseMeaning"),
            }
        )
    for view_id, view in visualization_views.items():
        if not isinstance(view, dict):
            continue
        rows.append(
            {
                "id": f"view:{view_id}",
                "label": view.get("label", view_id),
                "passed": None,
                "available": True,
                "style": "view_contract_info",
                "nonClaims": view.get("nonClaims", []),
            }
        )
    return rows


def _render_legend() -> list[dict[str, Any]]:
    return [
        {"layer": "screen_field", "color": "#3b82f6", "meaning": "observer-facing screen readback"},
        {"layer": "repair_or_holonomy_residue", "color": "#f59e0b", "meaning": "diagnostic repair/holonomy activity"},
        {"layer": "observer_axis", "color": "#10b981", "meaning": "observer-local readout axis"},
        {"layer": "observer_overlap", "color": "#64748b", "meaning": "shared support or carrier-cut overlap"},
        {"layer": "h3_object", "color": "#8b5cf6", "meaning": "observer-consensus H3 object packet"},
        {"layer": "proto_worldline", "color": "#ef4444", "meaning": "diagnostic H3 defect/worldline candidate"},
        {"layer": "cmb_diagnostic", "color": "#14b8a6", "meaning": "measurement-comparable CMB diagnostic"},
    ]


def _reference_vacuum_visualization_payload(run_dir: Path | None) -> dict[str, Any]:
    report_path = (
        Path(run_dir) / "reference_vacuum_baseline" / "reference_vacuum_baseline_report.json"
        if run_dir is not None
        else None
    )
    report = _read_json(report_path) if report_path is not None else {}
    free_scalar = report.get("free_scalar_gaussian") if isinstance(report.get("free_scalar_gaussian"), dict) else {}
    compact_u1 = (
        report.get("compact_u1_lattice_gauge")
        if isinstance(report.get("compact_u1_lattice_gauge"), dict)
        else {}
    )
    receipt_contract = (
        report.get("receipt_contract") if isinstance(report.get("receipt_contract"), dict) else {}
    )
    explicit_nonclaims = report.get("explicit_nonclaims") if isinstance(report.get("explicit_nonclaims"), list) else []
    return {
        "written": bool(report),
        "source": str(report_path) if report_path is not None else None,
        "claimTier": report.get("claim_tier"),
        "claimTierMeaning": report.get("claim_tier_meaning"),
        "freeScalarGaussian": {
            "modeCount": free_scalar.get("mode_count"),
            "sampleCount": free_scalar.get("sample_count"),
            "rawSpectrum": list(free_scalar.get("raw_spectrum", []) or [])[:64],
            "smoothedSpectrum": list(free_scalar.get("smoothed_spectrum", []) or [])[:64],
            "covarianceDiagnostics": free_scalar.get("covariance_diagnostics", {}),
            "refinementDiagnostics": free_scalar.get("refinement_diagnostics", {}),
            "artifacts": free_scalar.get("artifacts", {}),
        },
        "compactU1LatticeGauge": {
            "latticeSize": compact_u1.get("lattice_size"),
            "sweeps": compact_u1.get("sweeps"),
            "acceptanceRate": compact_u1.get("acceptance_rate"),
            "postBurnInMeanPlaquette": compact_u1.get("post_burn_in_mean_plaquette"),
            "plaquetteTrace": list(compact_u1.get("plaquette_trace", []) or [])[:128],
            "thermalizationAutocorrelation": compact_u1.get("thermalization_autocorrelation", {}),
        },
        "receipts": {
            "reference_vacuum_regression_receipt": bool(
                receipt_contract.get(
                    "reference_theory_regression",
                    free_scalar.get("reference_theory_regression_receipt", False)
                    and compact_u1.get("reference_theory_regression_receipt", False),
                )
            ),
            "OPH_NATIVE_QUOTIENT_ENSEMBLE_RECEIPT": bool(
                report.get("OPH_NATIVE_QUOTIENT_ENSEMBLE_RECEIPT", False)
            ),
            "OPH_NATIVE_VACUUM_PROMOTION_RECEIPT": bool(
                report.get("OPH_NATIVE_VACUUM_PROMOTION_RECEIPT", False)
            ),
            "OPH_PRIMORDIAL_FIELD_PROMOTION_RECEIPT": bool(
                report.get("OPH_PRIMORDIAL_FIELD_PROMOTION_RECEIPT", False)
            ),
        },
        "explicitNonClaims": explicit_nonclaims,
        "claimBoundary": (
            "Reference free-scalar and compact-U1 baseline for visualization/regression only; "
            "not promoted to an OPH-native vacuum or primordial physical field without the promotion receipts."
        ),
    }


def _yang_mills_gap_visualization_payload(run_dir: Path | None) -> dict[str, Any]:
    report_path = Path(run_dir) / "yang_mills_gap_certificate_report.json" if run_dir is not None else None
    report = _read_json(report_path) if report_path is not None else {}
    finite = report.get("finite_lattice_diagnostics") if isinstance(report.get("finite_lattice_diagnostics"), dict) else {}
    transfer = (
        report.get("finite_transfer_gap_diagnostic")
        if isinstance(report.get("finite_transfer_gap_diagnostic"), dict)
        else {}
    )
    reflection = (
        report.get("reflection_positivity_proxy")
        if isinstance(report.get("reflection_positivity_proxy"), dict)
        else {}
    )
    promotion = report.get("promotion_status") if isinstance(report.get("promotion_status"), dict) else {}
    continuum = report.get("continuum_certificate") if isinstance(report.get("continuum_certificate"), dict) else {}
    return {
        "written": bool(report),
        "source": str(report_path) if report_path is not None else None,
        "schema": report.get("schema"),
        "paperSource": report.get("paper_source"),
        "gaugeGroup": report.get("gauge_group", {}),
        "regulator": report.get("regulator", {}),
        "latticeGaugeStage": report.get("lattice_gauge_stage", {}),
        "finiteLatticeDiagnostics": finite,
        "finiteTransferGapDiagnostic": transfer,
        "reflectionPositivityProxy": reflection,
        "continuumCertificate": continuum,
        "promotionStatus": promotion,
        "plaquetteTrace": list(report.get("plaquette_trace", []) or [])[:256],
        "wilsonLoopTrace": list(report.get("wilson_loop_trace", []) or [])[:256],
        "polyakovLoopTrace": list(report.get("polyakov_loop_trace", []) or [])[:256],
        "orientationPlaquetteRows": list(report.get("orientation_plaquette_rows", []) or [])[:512],
        "refinementGapRows": list(report.get("refinement_gap_rows", []) or [])[:64],
        "explicitNonClaims": report.get("explicit_nonclaims", []),
        "receipts": {
            "finite_nonabelian_gauge_gap_diagnostic_receipt": bool(
                report.get("finite_nonabelian_gauge_gap_diagnostic_receipt", False)
            ),
            "finite_repair_gap_proxy_receipt": bool(report.get("finite_repair_gap_proxy_receipt", False)),
            "continuum_yang_mills_mass_gap_receipt": bool(
                report.get("continuum_yang_mills_mass_gap_receipt", False)
            ),
            "YANG_MILLS_GAP_REPRODUCED_RECEIPT": bool(
                report.get("YANG_MILLS_GAP_REPRODUCED_RECEIPT", False)
            ),
            "CLAY_YANG_MILLS_GAP_RECEIPT": bool(report.get("CLAY_YANG_MILLS_GAP_RECEIPT", False)),
        },
        "claimBoundary": report.get(
            "claim_boundary",
            "No Yang-Mills gap certificate report was emitted.",
        ),
    }


def _two_defect_stress_contraction_visualization_payload(run_dir: Path | None) -> dict[str, Any]:
    report_path = Path(run_dir) / "two_defect_stress_contraction_assay_report.json" if run_dir is not None else None
    report = _read_json(report_path) if report_path is not None else {}
    control_rows = report.get("control_trajectory_rows") if isinstance(report.get("control_trajectory_rows"), dict) else {}
    return {
        "written": bool(report),
        "source": str(report_path) if report_path is not None else None,
        "controlledPlantedAssay": bool(report.get("controlled_planted_assay", False)),
        "patchCount": report.get("patch_count"),
        "steps": report.get("steps"),
        "supportNodeCount": report.get("support_node_count"),
        "declaredStressContractionLaw": report.get("declared_stress_contraction_law", {}),
        "stressContractionSummary": _compact_trajectory_summary(
            report.get("stress_contraction_summary", {})
        ),
        "noContractionControlSummary": _compact_trajectory_summary(
            report.get("no_contraction_control_summary", {})
        ),
        "shuffledPairControlSummary": _compact_trajectory_summary(
            report.get("shuffled_pair_control_summary", {})
        ),
        "approachMarginVsControls": report.get("approach_margin_vs_controls"),
        "trajectoryRows": _compact_stress_contraction_rows(report.get("trajectory_rows", []), limit=128),
        "controlTrajectoryRows": {
            "noContraction": _compact_stress_contraction_rows(control_rows.get("no_contraction", []), limit=64),
            "shuffledPair": _compact_stress_contraction_rows(control_rows.get("shuffled_pair", []), limit=64),
        },
        "worldlines": _compact_stress_contraction_worldlines(report.get("worldlines", []), limit=4),
        "receipts": {
            "two_defect_stress_contraction_assay_receipt": bool(
                report.get("two_defect_stress_contraction_assay_receipt", False)
            ),
            "gravity_like_attraction_diagnostic_receipt": bool(
                report.get("gravity_like_attraction_diagnostic_receipt", False)
            ),
            "raw_production_gravity_requested": bool(report.get("production_gravity_receipt", False)),
            "raw_physical_gravity_requested": bool(report.get("physical_gravity_prediction", False)),
            "production_gravity_receipt": False,
            "physical_gravity_prediction": False,
            "particle_matter_receipt": bool(report.get("particle_matter_receipt", False)),
        },
        "claimBoundary": report.get(
            "claim_boundary",
            "Controlled/planted stress-contraction diagnostic only; not spontaneous particle formation, "
            "production gravity, or a physical prediction.",
        ),
    }


def _free_two_defect_dynamics_visualization_payload(run_dir: Path | None) -> dict[str, Any]:
    report_path = Path(run_dir) / "free_two_defect_dynamics_report.json" if run_dir is not None else None
    report = _read_json(report_path) if report_path is not None else {}
    summary = report.get("free_dynamics_summary") if isinstance(report.get("free_dynamics_summary"), dict) else {}
    return {
        "written": bool(report),
        "source": str(report_path) if report_path is not None else None,
        "controlledPlantedAssay": bool(report.get("controlled_planted_assay", False)),
        "freeDynamicsDiagnostic": bool(report.get("free_dynamics_diagnostic", False)),
        "patchCount": report.get("patch_count"),
        "steps": report.get("steps"),
        "supportNodeCount": report.get("support_node_count"),
        "seed": report.get("seed"),
        "declaredFreeDynamicsLaw": report.get("declared_free_dynamics_law", {}),
        "freeDynamicsSummary": _compact_free_dynamics_summary(summary),
        "trajectoryRows": _compact_free_dynamics_rows(report.get("trajectory_rows", []), limit=192),
        "worldlines": _compact_free_dynamics_worldlines(report.get("worldlines", []), limit=4),
        "receipts": {
            "free_two_defect_dynamics_receipt": bool(report.get("free_two_defect_dynamics_receipt", False)),
            "gravity_like_free_dynamics_diagnostic_receipt": bool(
                report.get("gravity_like_free_dynamics_diagnostic_receipt", False)
            ),
            "raw_production_gravity_requested": bool(report.get("production_gravity_receipt", False)),
            "raw_physical_gravity_requested": bool(report.get("physical_gravity_prediction", False)),
            "production_gravity_receipt": False,
            "physical_gravity_prediction": False,
            "particle_matter_receipt": bool(report.get("particle_matter_receipt", False)),
        },
        "claimBoundary": report.get(
            "claim_boundary",
            "Free randomized two-defect dynamics diagnostic only; not production gravity, "
            "spontaneous particle matter, or a physical merger claim.",
        ),
    }


def _organic_defect_population_visualization_payload(run_dir: Path | None) -> dict[str, Any]:
    report_path = Path(run_dir) / "organic_defect_population_report.json" if run_dir is not None else None
    report = _read_json(report_path) if report_path is not None else {}
    summary = (
        report.get("organic_population_summary")
        if isinstance(report.get("organic_population_summary"), dict)
        else {}
    )
    return {
        "written": bool(report),
        "source": str(report_path) if report_path is not None else None,
        "controlledPlantedAssay": bool(report.get("controlled_planted_assay", False)),
        "organicDefectPopulationDiagnostic": bool(report.get("organic_defect_population_diagnostic", False)),
        "patchCount": report.get("patch_count"),
        "steps": report.get("steps"),
        "defectCount": report.get("defect_count"),
        "minDefects": report.get("min_defects"),
        "maxDefects": report.get("max_defects"),
        "supportNodeCount": report.get("support_node_count"),
        "seed": report.get("seed"),
        "declaredOrganicPopulationLaw": report.get("declared_organic_population_law", {}),
        "organicPopulationSummary": _compact_organic_defect_summary(summary),
        "trajectoryRows": _compact_organic_defect_rows(report.get("trajectory_rows", []), limit=4096),
        "worldlines": _compact_organic_defect_worldlines(report.get("worldlines", []), limit=24),
        "renderingModes": report.get("rendering_modes", []),
        "receipts": {
            "organic_defect_population_receipt": bool(report.get("organic_defect_population_receipt", False)),
            "organic_proto_worldline_visualization_receipt": bool(
                report.get("organic_proto_worldline_visualization_receipt", False)
            ),
            "raw_production_gravity_requested": bool(report.get("production_gravity_receipt", False)),
            "raw_physical_gravity_requested": bool(report.get("physical_gravity_prediction", False)),
            "production_gravity_receipt": False,
            "physical_gravity_prediction": False,
            "particle_matter_receipt": bool(report.get("particle_matter_receipt", False)),
        },
        "claimBoundary": report.get(
            "claim_boundary",
            "Organic multi-defect diagnostic only; not particle matter or production gravity.",
        ),
    }


def _einstein_branch_entry_visualization_payload(run_dir: Path | None) -> dict[str, Any]:
    contract_path = Path(run_dir) / "finite_oph_theorem_contract_report.json" if run_dir is not None else None
    branch_path = Path(run_dir) / "einstein_branch_entry_report.json" if run_dir is not None else None
    manifest_path = Path(run_dir) / "einstein_bridge_manifest.json" if run_dir is not None else None
    contract = _read_json(contract_path) if contract_path is not None else {}
    branch = _read_json(branch_path) if branch_path is not None else {}
    manifest = _read_json(manifest_path) if manifest_path is not None else {}
    use_manifest = bool(manifest)
    receipt = (
        bool(
            manifest.get("einstein_branch_entry_contract_receipt", False)
            or manifest.get("einstein_branch_entry_receipt", False)
            or manifest.get("OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1", False)
            or manifest.get("EINSTEIN_BRANCH_ENTRY_RECEIPT", False)
        )
        if use_manifest
        else bool(
            contract.get("einstein_branch_entry_contract_receipt", False)
            or contract.get("OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1", False)
            or branch.get("einstein_branch_entry_contract_receipt", False)
            or branch.get("EINSTEIN_BRANCH_ENTRY_RECEIPT", False)
        )
    )
    blockers = list(
        (
            manifest.get("einstein_branch_entry_blockers")
            or manifest.get("blockers")
            or []
        )
        if use_manifest
        else contract.get("einstein_branch_entry_blockers")
        or branch.get("blockers")
        or [
            "E0_einstein_branch_entry_umbrella",
            "E1_null_generator_stress_charge",
            "E2_fixed_cap_entropy_stationarity",
            "E3_small_ball_area_bridge",
            "E4_all_timelike_tensor_upgrade",
            "E5_lambda_constancy_conservation",
            "E6_newton_coupling_forbidden_input_audit",
        ]
    )
    child_gates = manifest.get("einstein_branch_entry_child_gates") if use_manifest else contract.get(
        "einstein_branch_entry_child_gates"
    )
    if not isinstance(child_gates, dict):
        child_gates = branch.get("child_gates") if isinstance(branch.get("child_gates"), dict) else {}
    return {
        "schema": "oph_einstein_branch_entry_visualization_gate_v1",
        "written": bool(contract) or bool(branch) or bool(manifest),
        "source": str(contract_path) if contract_path is not None else None,
        "branchEntryReportSource": str(branch_path) if branch_path is not None else None,
        "einsteinBridgeManifestSource": str(manifest_path) if manifest_path is not None else None,
        "manifestWritten": use_manifest,
        "issue": 503,
        "legacyIssue": 503,
        "legacyIssueUrl": "https://github.com/FloatingPragma/observer-patch-holography/issues/503",
        "legacyIssueStatus": contract.get(
            "issue_503_einstein_branch_entry_status",
            branch.get("issue_503_status", "open_or_unreported"),
        ),
        "claimTier": manifest.get("claim_tier", contract.get("einstein_bridge_manifest", {}).get("claim_tier")),
        "provenanceTags": manifest.get(
            "provenanceTags",
            (contract.get("einstein_bridge_manifest") or {}).get("provenance_tags", {}),
        ),
        "receiptRows": manifest.get("receiptRows", []),
        "requiredReceiptFiles": manifest.get(
            "requiredReceiptFiles",
            (contract.get("einstein_bridge_manifest") or {}).get("required_receipt_files", []),
        ),
        "childGates": child_gates,
        "blockers": [] if receipt else blockers,
        "receipts": {
            "einstein_branch_entry_receipt": receipt,
            "EINSTEIN_BRANCH_ENTRY_RECEIPT": receipt,
            "OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1": receipt,
            "OPH_EINSTEIN_BRIDGE_MANIFEST_V1": bool(
                manifest.get("OPH_EINSTEIN_BRIDGE_MANIFEST_V1", False)
                or (contract.get("einstein_bridge_manifest") or {}).get("receipt", False)
            ),
            "EINSTEIN_BRIDGE_DEPENDENCY_DISCHARGE_RECEIPT": bool(
                manifest.get("EINSTEIN_BRIDGE_DEPENDENCY_DISCHARGE_RECEIPT", False)
                or (contract.get("einstein_bridge_manifest") or {}).get(
                    "dependency_discharge_receipt", False
                )
            ),
            "EINSTEIN_BRIDGE_RUN_RECEIPTS_RECEIPT": bool(
                manifest.get("EINSTEIN_BRIDGE_RUN_RECEIPTS_RECEIPT", False)
                or (contract.get("einstein_bridge_manifest") or {}).get("run_receipts_receipt", False)
            ),
        },
        "claimBoundary": (
            "E0 paper theorem discharges the OPH5 recovered-core bridge dependencies. Curved-spacetime, "
            "H3 object, and defect visuals remain diagnostics unless the run emits every theorem-tagged "
            "Einstein bridge sidecar receipt."
        ),
    }


def _string_vacuum_selector_visualization_payload(
    *,
    finite_consensus_receipt: bool,
    observer_facing_bulk_receipt: bool,
    particle_matter_receipt: bool,
) -> dict[str, Any]:
    selector_receipt = bool(finite_consensus_receipt or observer_facing_bulk_receipt)
    return {
        "schema": "oph_string_vacuum_selector_visualization_v1",
        "written": True,
        "paperSource": "markdown/observer_patch_holography_as_string_vacuum_selector.md",
        "candidateNamedWitness": "BD_{n=1,+}^{OPH}",
        "visualMetaphor": "OPH normal-form sieve over candidate critical-string presentations",
        "selectorStack": [
            {
                "layer": "observer_clock_implementation",
                "label": "OPH fixed-cutoff patch algebra",
                "status": "diagnostic_ready" if finite_consensus_receipt else "open",
                "receiptKey": "FINITE_CONSENSUS_THEOREM_RECEIPT",
                "description": "Finite observer patches, overlap repair, records, clocks, and boundary readout.",
            },
            {
                "layer": "edge_string_effective_language",
                "label": "Edge-string effective language",
                "status": "visualization_ready" if selector_receipt else "open",
                "receiptKey": "effective_edge_string_diagnostic_view",
                "description": "Cyclic edge normal forms, repair histories, and collar/worldsheet diagnostics.",
            },
            {
                "layer": "critical_string_representative",
                "label": "BD heterotic critical-string witness",
                "status": "open_certificate",
                "receiptKey": "critical_edge_cft_receipt",
                "description": "Requires critical-edge, cohomology, safety-realization, threshold, and moduli certificates.",
            },
        ],
        "acceptanceGates": _string_selector_acceptance_gates(),
        "criticalEdgeCertificateGates": _string_selector_critical_edge_gates(),
        "encodedCandidateSieve": _string_selector_candidate_rows(),
        "operatorSafetyTable": _string_selector_operator_safety_rows(),
        "quantitativeTargets": _string_selector_quantitative_targets(),
        "falsifierMatrix": _string_selector_falsifier_rows(),
        "receipts": {
            "string_vacuum_selector_visualization_receipt": True,
            "effective_edge_string_diagnostic_view": selector_receipt,
            "encoded_structural_audit_data_receipt": True,
            "bd_operator_safe_candidate_selected_in_encoded_audit": True,
            "finite_consensus_theorem_receipt": bool(finite_consensus_receipt),
            "observer_facing_consensus_3d_bulk_readout_receipt": bool(observer_facing_bulk_receipt),
            "particle_matter_receipt": bool(particle_matter_receipt),
            "critical_edge_cft_receipt": False,
            "virasoro_receipt": False,
            "sugawara_receipt": False,
            "supercurrent_receipt": False,
            "spin_structure_receipt": False,
            "bd_full_cohomology_certificate_receipt": False,
            "bd_z4r_compactification_realization_receipt": False,
            "bd_threshold_spectrum_certificate_receipt": False,
            "bd_moduli_locking_certificate_receipt": False,
            "global_singleton_string_vacuum_receipt": False,
            "oph_native_string_vacuum_receipt": False,
        },
        "promotionReceiptsRequired": [
            "critical_edge_cft_receipt",
            "bd_full_cohomology_certificate_receipt",
            "bd_z4r_compactification_realization_receipt",
            "bd_threshold_spectrum_certificate_receipt",
            "bd_moduli_locking_certificate_receipt",
            "comparative_uniqueness_certificate_receipt",
        ],
        "nonClaims": [
            "proven critical string CFT",
            "completed BD compactification certificate",
            "global string landscape singleton",
            "OPH-native string vacuum promotion",
        ],
        "claimBoundary": (
            "String-vacuum selector visualization data from the OPH paper gate stack. The encoded "
            "structural sieve selects the operator-safe BD witness inside its declared audit rows, "
            "but the simulator does not emit critical-edge, BD cohomology, threshold, moduli-locking, "
            "or global uniqueness certificates."
        ),
    }


def _fractional_quotient_visualization_payload(run_dir: Path | None) -> dict[str, Any]:
    report_path = run_dir / "fractional_quotient_report.json" if run_dir is not None else None
    if report_path is not None and report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report = {
                "schema": "oph_fractional_quotient_sim_report_v1",
                "status": "fail",
                "fail_closed_state": "DIAGNOSTIC_ONLY",
                "error": f"could not parse fractional_quotient_report.json: {exc}",
            }
    else:
        try:
            from oph_fractional.compare import demo_fractional_report

            report = demo_fractional_report()
        except Exception as exc:  # pragma: no cover - defensive for standalone viewer export
            report = {
                "schema": "oph_fractional_quotient_sim_report_v1",
                "status": "fail",
                "fail_closed_state": "DIAGNOSTIC_ONLY",
                "error": str(exc),
            }

    phase = report.get("phase_promotion", {}) if isinstance(report.get("phase_promotion"), dict) else {}
    phase_receipts = phase.get("receipts", {}) if isinstance(phase.get("receipts"), dict) else {}
    quotient = report.get("quotient", {}) if isinstance(report.get("quotient"), dict) else {}
    source = report.get("source", {}) if isinstance(report.get("source"), dict) else {}
    line_fan = report.get("line_fan", {}) if isinstance(report.get("line_fan"), dict) else {}
    slope = report.get("slope_certificate", {}) if isinstance(report.get("slope_certificate"), dict) else {}
    no_leak = report.get("no_target_leak", {}) if isinstance(report.get("no_target_leak"), dict) else {}
    refinement = report.get("refinement", {}) if isinstance(report.get("refinement"), dict) else {}
    return {
        "schema": "oph_fractional_quotient_visualization_v1",
        "written": True,
        "sourceReportPath": str(report_path) if report_path is not None and report_path.exists() else None,
        "fractionalReport": report,
        "receipts": {
            "source_hamiltonian_frozen": bool(source.get("receipts", {}).get("SOURCE_HAMILTONIAN_FROZEN", False)),
            "topological_sector_ledger": bool(phase_receipts.get("TOPOLOGICAL_SECTOR_LEDGER", False)),
            "phase_certificate_injective": bool(phase_receipts.get("PHASE_CERTIFICATE_INJECTIVE", False)),
            "line_fan_decomposition": bool(line_fan.get("LINE_FAN_DECOMPOSITION", False)),
            "binding_drift_bounded": bool(slope.get("receipts", {}).get("BINDING_DRIFT_BOUNDED", False)),
            "no_target_leak": bool(no_leak.get("receipts", {}).get("NO_TARGET_LEAK", False)),
            "refinement_compatibility": bool(
                refinement.get("receipts", {}).get("REFINEMENT_COMPATIBILITY", False)
            ),
            "canonicalizer_idempotence": bool(
                ((quotient.get("canonicalizer") or {}).get("receipts") or {}).get(
                    "CANONICALIZER_IDEMPOTENCE",
                    False,
                )
            ),
            "representative_invariance": bool(
                ((quotient.get("representative_invariance") or {}).get("receipts") or {}).get(
                    "REPRESENTATIVE_INVARIANCE",
                    False,
                )
            ),
            "quotient_lumpability": bool(
                ((quotient.get("lumpability") or {}).get("receipts") or {}).get(
                    "QUOTIENT_LUMPABILITY",
                    False,
                )
            ),
        },
        "renderHints": [
            {
                "layer": "quotient_sector_ring",
                "source": "fractionalReport.topological_ledger.sectors",
                "encoding": "sector nodes arranged around the Hall collar",
            },
            {
                "layer": "optical_line_fan",
                "source": "fractionalReport.line_fan.peaks",
                "encoding": "energy lines with color by tau and slope by total_charge",
            },
            {
                "layer": "receipt_gate_strip",
                "source": "receipts",
                "encoding": "fail-closed badges for source, quotient, ledger, optical, refinement, and leak gates",
            },
        ],
        "claimBoundary": (
            "Fractional quotient-sector visualization contract. It displays the material-source, "
            "topological-ledger, optical-module, line-fan, no-target-leak, and refinement gates. "
            "The built-in demo is diagnostic only and is not a proof for a real material sample."
        ),
    }


def _string_selector_acceptance_gates() -> list[dict[str, Any]]:
    return [
        {
            "gateId": "global_group",
            "label": "Global group",
            "requirement": "(SU(3)xSU(2)xU(1))/Z6 visible quotient",
            "status": "encoded_structural_gate",
            "visualState": "paper_gate",
        },
        {
            "gateId": "hypercharge_lattice",
            "label": "Hypercharge lattice",
            "requirement": "Standard Model charge lattice",
            "status": "encoded_structural_gate",
            "visualState": "paper_gate",
        },
        {
            "gateId": "color_generations",
            "label": "Color and generations",
            "requirement": "N_c=3 and N_g=3",
            "status": "encoded_structural_gate",
            "visualState": "paper_gate",
        },
        {
            "gateId": "one_higgs_pair",
            "label": "Higgs sector",
            "requirement": "Exactly one Higgs pair on the low-energy branch",
            "status": "encoded_structural_gate",
            "visualState": "paper_gate",
        },
        {
            "gateId": "no_light_chiral_exotics",
            "label": "Exotics",
            "requirement": "No light chiral exotics",
            "status": "encoded_structural_gate",
            "visualState": "paper_gate",
        },
        {
            "gateId": "no_extra_visible_u1",
            "label": "Extra visible gauge factors",
            "requirement": "No extra visible low-scale U(1)",
            "status": "encoded_structural_gate",
            "visualState": "paper_gate",
        },
        {
            "gateId": "no_xy_gauge_proton_decay",
            "label": "Gauge proton decay",
            "requirement": "Product-group adjoint contains no mixed (3,2,+/-5/6) gauge bosons",
            "status": "encoded_structural_gate",
            "visualState": "paper_gate",
        },
        {
            "gateId": "operator_safety",
            "label": "Operator safety",
            "requirement": "Z_4^R permits Yukawas/Weinberg and forbids perturbative RPV, d=5 proton decay, and mu term",
            "status": "encoded_charge_algebra_gate",
            "visualState": "paper_gate",
        },
        {
            "gateId": "moduli_thresholds",
            "label": "Moduli and thresholds",
            "requirement": "F_BD,n=1,+(m_star)=O_OPH with full transverse rank",
            "status": "open_certificate",
            "visualState": "open_gate",
        },
    ]


def _string_selector_critical_edge_gates() -> list[dict[str, Any]]:
    return [
        {
            "gateId": "carrier_refinement_manifest",
            "label": "Carrier and refinement manifest",
            "neededInput": "model hashes, local factors, update/repair gates, transfer callbacks, A5 action, orientation involution, translation, record/port operators",
            "status": "not_emitted_by_simulator",
            "receipt": False,
        },
        {
            "gateId": "twelve_port_rank",
            "label": "Twelve-port dynamical rank",
            "neededInput": "rank R_port=12 with A5 sectors 1+3+3'+5 nonzero under refinement",
            "status": "not_emitted_by_simulator",
            "receipt": False,
        },
        {
            "gateId": "orientation_independence",
            "label": "Orientation independence",
            "neededInput": "both A5 x C2 orientation parities survive dynamically",
            "status": "not_emitted_by_simulator",
            "receipt": False,
        },
        {
            "gateId": "conserved_currents",
            "label": "Conserved currents",
            "neededInput": "edge-cylinder continuity equation and chiral level ranks k_L=24, k_R=12",
            "status": "open_edge_cylinder_required",
            "receipt": False,
        },
        {
            "gateId": "virasoro_central_charge",
            "label": "Virasoro central charge",
            "neededInput": "lattice stress-tensor modes with c_L -> 24 and c_R -> 12",
            "status": "open_edge_cylinder_required",
            "receipt": False,
        },
        {
            "gateId": "sugawara_exhaustion",
            "label": "Sugawara exhaustion",
            "neededInput": "T_OPH - T_Sug -> 0; no residual positive-central-charge coset",
            "status": "open_edge_cylinder_required",
            "receipt": False,
        },
        {
            "gateId": "supercurrent",
            "label": "Supercurrent",
            "neededInput": "right-moving graded sector and local odd supercurrent with Q_N^2 -> H_R",
            "status": "open_edge_cylinder_required",
            "receipt": False,
        },
        {
            "gateId": "spin_structures_internal_lattice",
            "label": "Torus, spin structures, internal lattice",
            "neededInput": "modular spin-structure partition functions and rank-sixteen even self-dual left-moving internal lattice",
            "status": "open_edge_cylinder_required",
            "receipt": False,
        },
        {
            "gateId": "blind_refinement_controls",
            "label": "Blind refinement and falsification",
            "neededInput": "pre-registered refinements plus controls that break orientation, ports, geometry, spectators, chirality, and seams",
            "status": "open_protocol",
            "receipt": False,
        },
    ]


def _string_selector_candidate_rows() -> list[dict[str, Any]]:
    return [
        {
            "candidate": "BD_{n=0}^{SU(5),Z2}",
            "scoreNumerator": 7,
            "scoreDenominator": 9,
            "selected": False,
            "verdict": "Rejected: no Higgs pair.",
        },
        {
            "candidate": "BD_{n=1}^{SU(5),Z2}",
            "scoreNumerator": 8,
            "scoreDenominator": 9,
            "selected": False,
            "verdict": "Geometric witness; safety layer absent.",
        },
        {
            "candidate": "BD_{n=1,+}^{SU(5),Z2}",
            "scoreNumerator": 9,
            "scoreDenominator": 9,
            "selected": True,
            "verdict": "Selected operator-safe candidate in the encoded structural audit.",
        },
        {
            "candidate": "BD_{n=2}^{SU(5),Z2}",
            "scoreNumerator": 7,
            "scoreDenominator": 9,
            "selected": False,
            "verdict": "Rejected: extra Higgs pair.",
        },
        {
            "candidate": "BHOP SU(4),Z3xZ3",
            "scoreNumerator": 4,
            "scoreDenominator": 9,
            "selected": False,
            "verdict": "Backup witness.",
        },
        {
            "candidate": "Generic Spin(32)/Z2 heterotic",
            "scoreNumerator": 0,
            "scoreDenominator": 9,
            "selected": False,
            "verdict": "Rejected as minimal class.",
        },
    ]


def _string_selector_operator_safety_rows() -> list[dict[str, Any]]:
    return [
        {"operator": "Q H_u u^c", "rChargeMod4": 2, "allowed": True, "result": "Up-type Yukawa allowed."},
        {"operator": "Q H_d d^c", "rChargeMod4": 2, "allowed": True, "result": "Down-type Yukawa allowed."},
        {"operator": "L H_d e^c", "rChargeMod4": 2, "allowed": True, "result": "Charged-lepton Yukawa allowed."},
        {"operator": "L H_u L H_u", "rChargeMod4": 2, "allowed": True, "result": "Weinberg neutrino operator allowed."},
        {"operator": "L H_u", "rChargeMod4": 1, "allowed": False, "result": "Bilinear RPV forbidden perturbatively."},
        {"operator": "L L e^c", "rChargeMod4": 3, "allowed": False, "result": "Lepton-number RPV forbidden perturbatively."},
        {"operator": "L Q d^c", "rChargeMod4": 3, "allowed": False, "result": "Lepton-number RPV forbidden perturbatively."},
        {"operator": "u^c d^c d^c", "rChargeMod4": 3, "allowed": False, "result": "Baryon-number RPV forbidden perturbatively."},
        {"operator": "Q Q Q L", "rChargeMod4": 0, "allowed": False, "result": "Dimension-five proton decay forbidden perturbatively."},
        {"operator": "u^c u^c d^c e^c", "rChargeMod4": 0, "allowed": False, "result": "Dimension-five proton decay forbidden perturbatively."},
    ]


def _string_selector_quantitative_targets() -> list[dict[str, Any]]:
    return [
        {"quantity": "top_yukawa", "label": "Top Yukawa", "value": 0.987745211164, "unit": None},
        {"quantity": "higgs_quartic", "label": "Higgs quartic", "value": 0.128706603202, "unit": None},
        {"quantity": "mssm_tree_quartic_ceiling", "label": "MSSM tree-level quartic ceiling", "value": 0.068973725409, "unit": None},
        {"quantity": "minimum_positive_threshold_lift", "label": "Minimum positive threshold lift", "value": 0.059732877792, "unit": None},
        {"quantity": "tree_level_higgs_mass_proxy_ceiling", "label": "Tree-level Higgs mass proxy ceiling", "value": 91.6524602856, "unit": "GeV"},
        {"quantity": "gut_log10_mu_gev", "label": "Gauge-unification log10(M_U/GeV)", "value": 16.0815812332, "unit": None},
        {"quantity": "gut_alpha_u_inverse", "label": "Gauge-unification alpha_U^-1", "value": 26.0176807530, "unit": None},
        {"quantity": "alpha3_pred_mz", "label": "alpha_3 predicted at m_Z", "value": 0.111511310319, "unit": None},
        {"quantity": "alpha3_threshold_needed", "label": "Delta(alpha_3^-1) needed", "value": -0.521754255407, "unit": None},
    ]


def _string_selector_falsifier_rows() -> list[dict[str, Any]]:
    return [
        {"failureMode": "Wrong global Standard Model group", "consequence": "Retracts the OPH visible landing branch."},
        {"failureMode": "Wrong hypercharge lattice", "consequence": "Retracts the named visible branch."},
        {"failureMode": "Fourth chiral generation", "consequence": "Retracts the realized branch."},
        {"failureMode": "Light chiral exotics", "consequence": "Retracts the BD one-Higgs witness."},
        {"failureMode": "Irreducible second light Higgs pair", "consequence": "Retracts the one-Higgs low-energy projection."},
        {"failureMode": "Extra visible low-scale U(1)", "consequence": "Severe pressure on the OPH normal-form sieve."},
        {"failureMode": "Gauge-mediated X/Y proton decay", "consequence": "Conflicts with the product-group branch."},
        {"failureMode": "BD cohomology reproduction fails", "consequence": "Retracts BD_{n=1}^{OPH} as named geometric witness."},
        {"failureMode": "Z_4^R or equivalent safety layer absent", "consequence": "Retracts BD_{n=1,+}^{OPH} as operator-safe candidate."},
        {"failureMode": "Dangerous operators survive without suppression", "consequence": "Severe pressure on the effective theory."},
        {"failureMode": "No moduli point matches O_OPH", "consequence": "Retracts the operator-safe candidate."},
        {"failureMode": "Central-charge or supercurrent gate fails", "consequence": "Retracts the heterotic critical-worldsheet identification of the sewn edge branch."},
        {"failureMode": "Critical-string lift fails", "consequence": "Weakens the string continuation; the OPH recovered core is a separate claim tier."},
    ]


def _compact_trajectory_summary(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "initialH3Separation": source.get("initial_h3_separation"),
        "finalH3Separation": source.get("final_h3_separation"),
        "absoluteH3Approach": source.get("absolute_h3_approach"),
        "approachFraction": source.get("approach_fraction"),
        "minH3Separation": source.get("min_h3_separation"),
        "maxH3Separation": source.get("max_h3_separation"),
    }


def _compact_free_dynamics_summary(value: Any) -> dict[str, Any]:
    summary = _compact_trajectory_summary(value)
    source = value if isinstance(value, dict) else {}
    summary.update(
        {
            "contactStep": source.get("contact_step"),
            "contactCycle": source.get("contact_cycle"),
            "contactOutcome": source.get("contact_outcome"),
            "explicitContactOutcome": bool(source.get("explicit_contact_outcome", False)),
            "chargeConservationPass": bool(source.get("charge_conservation_pass", False)),
            "maxSupportOverlapFraction": source.get("max_support_overlap_fraction"),
            "transverseMotionPresent": bool(source.get("transverse_motion_present", False)),
            "straightXAxisControl": bool(source.get("straight_x_axis_control", False)),
        }
    )
    return summary


def _compact_organic_defect_summary(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "worldlineCount": source.get("worldline_count"),
        "defectCountInRequestedBand": bool(source.get("defect_count_in_requested_band", False)),
        "minRequestedDefects": source.get("min_requested_defects"),
        "maxRequestedDefects": source.get("max_requested_defects"),
        "birthCycleMin": source.get("birth_cycle_min"),
        "birthCycleMax": source.get("birth_cycle_max"),
        "staggeredBirthsPresent": bool(source.get("staggered_births_present", False)),
        "nearContactEventCount": source.get("near_contact_event_count"),
        "transverseMotionPresent": bool(source.get("transverse_motion_present", False)),
        "meanLocalStressDensity": source.get("mean_local_stress_density"),
        "maxLocalStressDensity": source.get("max_local_stress_density"),
        "persistentWorldlineCount": source.get("persistent_worldline_count"),
        "fixedLeftRightPair": bool(source.get("fixed_left_right_pair", False)),
        "controlledPlantedAssay": bool(source.get("controlled_planted_assay", False)),
    }


def _compact_organic_defect_rows(value: Any, *, limit: int) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    compact = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                "mode": row.get("mode"),
                "step": row.get("step"),
                "cycle": row.get("cycle"),
                "defectId": row.get("defect_id"),
                "birthTrigger": row.get("birth_trigger"),
                "class": row.get("class"),
                "holonomyMode": row.get("holonomy_mode"),
                "h3SpatialPoint": row.get("h3_spatial_point"),
                "velocity": row.get("velocity"),
                "localStressDensity": row.get("local_stress_density"),
                "nearestDefectId": row.get("nearest_defect_id"),
                "nearestH3Separation": row.get("nearest_h3_separation"),
                "supportOverlapFraction": row.get("support_overlap_fraction"),
                "supportOverlapNodeCount": row.get("support_overlap_node_count"),
                "contactEvent": row.get("contact_event"),
                "contactOutcome": row.get("contact_outcome"),
                "supportNodeCount": row.get("support_node_count"),
                "renderAsPoint": row.get("render_as_point"),
                "renderAsString": row.get("render_as_string"),
                "renderInSubjectiveObserverView": row.get("render_in_subjective_observer_view"),
            }
        )
    return compact


def _compact_organic_defect_worldlines(value: Any, *, limit: int) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    compact = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        events = _compact_h3_events(row.get("events", []), limit=128)
        compact.append(
            {
                "worldlineId": row.get("worldline_id"),
                "observationCount": row.get("observation_count"),
                "birthCycle": row.get("birth_cycle"),
                "deathCycle": row.get("death_cycle"),
                "lifetimeCycles": row.get("lifetime_cycles"),
                "persistent": bool(row.get("persistent", False)),
                "classMode": row.get("class_mode"),
                "holonomyMode": row.get("holonomy_mode"),
                "supportPatchCount": row.get("support_patch_count"),
                "meanTransportDistance": row.get("mean_transport_distance"),
                "events": events,
            }
        )
    return compact


def _compact_stress_contraction_rows(value: Any, *, limit: int) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    compact = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                "mode": row.get("mode"),
                "step": row.get("step"),
                "cycle": row.get("cycle"),
                "leftH3SpatialPoint": row.get("left_h3_spatial_point"),
                "rightH3SpatialPoint": row.get("right_h3_spatial_point"),
                "tangentSeparation": row.get("tangent_separation"),
                "h3Separation": row.get("h3_separation"),
                "stressKernel": row.get("stress_kernel"),
                "localReadoutContraction": row.get("local_readout_contraction"),
            }
        )
    return compact


def _compact_free_dynamics_rows(value: Any, *, limit: int) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    compact = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                "mode": row.get("mode"),
                "step": row.get("step"),
                "cycle": row.get("cycle"),
                "leftH3SpatialPoint": row.get("left_h3_spatial_point"),
                "rightH3SpatialPoint": row.get("right_h3_spatial_point"),
                "leftVelocity": row.get("left_velocity"),
                "rightVelocity": row.get("right_velocity"),
                "tangentSeparation": row.get("tangent_separation"),
                "h3Separation": row.get("h3_separation"),
                "stressKernel": row.get("stress_kernel"),
                "relativeSpeed": row.get("relative_speed"),
                "supportOverlapFraction": row.get("support_overlap_fraction"),
                "supportOverlapNodeCount": row.get("support_overlap_node_count"),
                "contactEvent": row.get("contact_event"),
                "contactOutcome": row.get("contact_outcome"),
                "chargeConservationPass": row.get("charge_conservation_pass"),
            }
        )
    return compact


def _compact_stress_contraction_worldlines(value: Any, *, limit: int) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    compact = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        events = []
        for event in list(row.get("events", []) or [])[:64]:
            if not isinstance(event, dict):
                continue
            events.append(
                {
                    "cycle": event.get("cycle"),
                    "event": event.get("event"),
                    "class": event.get("class"),
                    "holonomyMode": event.get("holonomy_mode"),
                    "supportNodeCount": event.get("support_node_count"),
                    "h3SpatialPoint": event.get("h3_spatial_point"),
                    "pairH3Separation": event.get("pair_h3_separation"),
                    "localReadoutContraction": event.get("local_readout_contraction"),
                    "transportDistance": event.get("transport_distance"),
                }
            )
        compact.append(
            {
                "worldlineId": row.get("worldline_id"),
                "observationCount": row.get("observation_count"),
                "birthCycle": row.get("birth_cycle"),
                "deathCycle": row.get("death_cycle"),
                "lifetimeCycles": row.get("lifetime_cycles"),
                "persistent": bool(row.get("persistent", False)),
                "meanTransportDistance": row.get("mean_transport_distance"),
                "events": events,
            }
        )
    return compact


def _compact_free_dynamics_worldlines(value: Any, *, limit: int) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    compact = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        events = []
        for event in list(row.get("events", []) or [])[:96]:
            if not isinstance(event, dict):
                continue
            events.append(
                {
                    "cycle": event.get("cycle"),
                    "event": event.get("event"),
                    "class": event.get("class"),
                    "holonomyMode": event.get("holonomy_mode"),
                    "supportNodeCount": event.get("support_node_count"),
                    "h3SpatialPoint": event.get("h3_spatial_point"),
                    "velocity": event.get("velocity"),
                    "pairH3Separation": event.get("pair_h3_separation"),
                    "supportOverlapFraction": event.get("support_overlap_fraction"),
                    "supportOverlapNodeCount": event.get("support_overlap_node_count"),
                    "contactOutcome": event.get("contact_outcome"),
                    "chargeConservationPass": event.get("charge_conservation_pass"),
                    "transportDistance": event.get("transport_distance"),
                }
            )
        compact.append(
            {
                "worldlineId": row.get("worldline_id"),
                "observationCount": row.get("observation_count"),
                "birthCycle": row.get("birth_cycle"),
                "deathCycle": row.get("death_cycle"),
                "lifetimeCycles": row.get("lifetime_cycles"),
                "persistent": bool(row.get("persistent", False)),
                "contactOutcome": row.get("contact_outcome"),
                "meanTransportDistance": row.get("mean_transport_distance"),
                "events": events,
            }
        )
    return compact


def _visualization_views_payload(
    *,
    small_payload: dict[str, Any],
    screen_payload: dict[str, Any],
    observer_payload: dict[str, Any],
    bulk_payload: dict[str, Any],
    cmb_payload: dict[str, Any],
    pn_silence_payload: dict[str, Any],
    diagnostic_run_dir: Path | None = None,
) -> dict[str, Any]:
    small_receipts = small_payload.get("receipts", {})
    observer_receipts = observer_payload.get("receipts", {})
    bulk_receipts = bulk_payload.get("receipts", {})
    proto_receipts = bulk_payload.get("protoParticleCandidates", {}).get("receipts", {})
    cmb_receipts = cmb_payload.get("receipts", {})
    pn_receipts = pn_silence_payload.get("receipts", {})
    reference_vacuum = _reference_vacuum_visualization_payload(diagnostic_run_dir)
    yang_mills_gap = _yang_mills_gap_visualization_payload(diagnostic_run_dir)
    stress_contraction = _two_defect_stress_contraction_visualization_payload(diagnostic_run_dir)
    organic_defects = _organic_defect_population_visualization_payload(diagnostic_run_dir)
    free_dynamics = _free_two_defect_dynamics_visualization_payload(diagnostic_run_dir)
    einstein_branch = _einstein_branch_entry_visualization_payload(diagnostic_run_dir)
    fractional_quotient = _fractional_quotient_visualization_payload(diagnostic_run_dir)
    einstein_branch_receipts = (
        einstein_branch.get("receipts", {}) if isinstance(einstein_branch.get("receipts"), dict) else {}
    )
    einstein_branch_receipt = bool(einstein_branch_receipts.get("einstein_branch_entry_receipt", False))
    raw_production_gravity = bool(
        stress_contraction.get("receipts", {}).get("raw_production_gravity_requested", False)
        or organic_defects.get("receipts", {}).get("raw_production_gravity_requested", False)
        or free_dynamics.get("receipts", {}).get("raw_production_gravity_requested", False)
    )
    raw_physical_gravity = bool(
        stress_contraction.get("receipts", {}).get("raw_physical_gravity_requested", False)
        or organic_defects.get("receipts", {}).get("raw_physical_gravity_requested", False)
        or free_dynamics.get("receipts", {}).get("raw_physical_gravity_requested", False)
    )
    production_gravity_receipt = bool(einstein_branch_receipt and raw_production_gravity)
    physical_gravity_prediction = bool(einstein_branch_receipt and raw_physical_gravity)
    observer_facing_bulk_receipt = bool(
        bulk_receipts.get(
            "observer_facing_consensus_3d_bulk_readout_receipt",
            bulk_receipts.get("theorem_assisted_consensus_3d_bulk_readout_receipt", False),
        )
    )
    string_selector = _string_vacuum_selector_visualization_payload(
        finite_consensus_receipt=bool(small_receipts.get("FINITE_CONSENSUS_THEOREM_RECEIPT", False)),
        observer_facing_bulk_receipt=observer_facing_bulk_receipt,
        particle_matter_receipt=bool(proto_receipts.get("particle_matter_receipt", False)),
    )
    return {
        "fluctuatingQuantumVacuum": {
            "viewId": "fluctuatingQuantumVacuum",
            "sectionKind": "quantum_vacuum_fluctuation",
            "label": "Fluctuating quantum vacuum / finite readback field",
            "visualMetaphor": "finite_boundary_readback_fluctuations",
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
                "visualizationViews.fluctuatingQuantumVacuum.referenceVacuumBaseline",
                "visualizationViews.fluctuatingQuantumVacuum.yangMillsGapCertificate",
            ],
            "primaryFields": ["screen.values", "screen.clusters.snapshots", "screen.repairTrace"],
            "renderLayers": [
                {"layer": "screen_s2_field", "source": "screen.points + screen.values"},
                {"layer": "repair_fluctuation_markers", "source": "screen.clusters.snapshots[*].clusters"},
                {"layer": "mismatch_commit_trace", "source": "screen.repairTrace"},
                {"layer": "cmb_diagnostic_overlay", "source": "cmbComparison.residualRows"},
                {
                    "layer": "reference_vacuum_baseline_inset",
                    "source": "visualizationViews.fluctuatingQuantumVacuum.referenceVacuumBaseline",
                },
                {
                    "layer": "finite_su2_yang_mills_diagnostic",
                    "source": "visualizationViews.fluctuatingQuantumVacuum.yangMillsGapCertificate",
                },
            ],
            "visualEncodings": [
                {
                    "field": "readback_amplitude",
                    "source": "screen.values",
                    "encoding": "screen color and optional surface displacement",
                    "palette": "diverging_vacuum_field",
                },
                {
                    "field": "repair_or_holonomy_residue",
                    "source": "screen.clusters.snapshots[*].clusters",
                    "encoding": "spark/ring pulses on the S2 boundary",
                    "palette": "residue_highlight",
                },
                {
                    "field": "finite_repair_descent",
                    "source": "screen.repairTrace",
                    "encoding": "time scrubber, opacity pulse, and Phi/record trace",
                    "palette": "repair_trace",
                },
            ],
            "animationChannels": [
                {
                    "channel": "vacuum_field_flicker",
                    "source": "screen.values",
                    "timeSource": "screen.repairTrace[*].cycle",
                    "encoding": "low-amplitude temporal color shimmer",
                },
                {
                    "channel": "holonomy_burst",
                    "source": "screen.clusters.snapshots",
                    "timeSource": "screen.clusters.snapshots[*].cycle",
                    "encoding": "cluster-local pulse/ripple",
                },
                {
                    "channel": "diagnostic_cmb_residual_overlay",
                    "source": "cmbComparison.residualRows",
                    "timeSource": None,
                    "encoding": "optional static spectrum inset or boundary contour",
                },
            ],
            "referenceVacuumBaseline": reference_vacuum,
            "yangMillsGapCertificate": yang_mills_gap,
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
                "reference_vacuum_regression_receipt": bool(
                    reference_vacuum.get("receipts", {}).get("reference_vacuum_regression_receipt", False)
                ),
                "finite_nonabelian_gauge_gap_diagnostic_receipt": bool(
                    yang_mills_gap.get("receipts", {}).get(
                        "finite_nonabelian_gauge_gap_diagnostic_receipt",
                        False,
                    )
                ),
                "YANG_MILLS_GAP_REPRODUCED_RECEIPT": bool(
                    yang_mills_gap.get("receipts", {}).get("YANG_MILLS_GAP_REPRODUCED_RECEIPT", False)
                ),
                "CLAY_YANG_MILLS_GAP_RECEIPT": bool(
                    yang_mills_gap.get("receipts", {}).get("CLAY_YANG_MILLS_GAP_RECEIPT", False)
                ),
                "OPH_NATIVE_VACUUM_PROMOTION_RECEIPT": bool(
                    reference_vacuum.get("receipts", {}).get("OPH_NATIVE_VACUUM_PROMOTION_RECEIPT", False)
                ),
                "OPH_PRIMORDIAL_FIELD_PROMOTION_RECEIPT": bool(
                    reference_vacuum.get("receipts", {}).get("OPH_PRIMORDIAL_FIELD_PROMOTION_RECEIPT", False)
                ),
            },
            "exportSufficiency": "sufficient_for_diagnostic_visualization_not_physical_qft_vacuum",
            "promotionReceiptsRequired": [
                "finite_consensus_theorem_receipt",
                "OPH_NATIVE_VACUUM_PROMOTION_RECEIPT",
                "dedicated_qft_vacuum_receipt",
            ],
            "nonClaims": [
                "literal QFT vacuum",
                "physical CMB prediction",
                "pre-existing neutral 3D bulk",
                "reproduced Yang-Mills mass gap",
            ],
            "claimBoundary": (
                "Shows finite OPH screen/readback fluctuations and repair residues. Do not label as a "
                "literal fluctuating quantum vacuum without a dedicated QFT-vacuum receipt."
            ),
        },
        "observerCamera": {
            "viewId": "observerCamera",
            "sectionKind": "observer_camera",
            "label": "Observer camera / local modular-time readout",
            "visualMetaphor": "observer_local_camera_from_visible_records",
            "description": (
                "Render one observer-local camera at a time from subjectiveObserverCameras. Each camera "
                "is derived from visible support, readback records, packet histograms, and modular-time "
                "frames; it is not a hidden global camera."
            ),
            "dataSources": [
                "subjectiveObserverCameras",
                "subjectiveObserverCameras[*].timeFrames[*].visibleProtoWorldlines",
                "observerModularTime.objectiveObserverViews",
                "observerModularTime.overlapLinks",
                "observerModularTime.timeFrames",
                "consensusBulk.protoParticleCandidates.worldlines",
            ],
            "primaryFields": [
                "subjectiveObserverCameras[*].eye",
                "subjectiveObserverCameras[*].forward",
                "subjectiveObserverCameras[*].timeFrames",
                "subjectiveObserverCameras[*].timeFrames[*].visibleProtoWorldlines",
            ],
            "renderLayers": [
                {"layer": "observer_axis_camera", "source": "subjectiveObserverCameras[*]"},
                {"layer": "visible_record_packets", "source": "subjectiveObserverCameras[*].timeFrames"},
                {
                    "layer": "observer_visible_proto_worldlines",
                    "source": "subjectiveObserverCameras[*].timeFrames[*].visibleProtoWorldlines",
                },
                {"layer": "overlap_support_graph", "source": "observerModularTime.overlapLinks"},
            ],
            "visualEncodings": [
                {
                    "field": "observer_pose",
                    "source": "subjectiveObserverCameras[*].eye/lookAt/up/right/forward",
                    "encoding": "camera transform and frustum",
                    "palette": "observer_axis",
                },
                {
                    "field": "visible_packets",
                    "source": "subjectiveObserverCameras[*].timeFrames",
                    "encoding": "local labels, point highlights, and packet glyphs",
                    "palette": "record_packet",
                },
                {
                    "field": "visible_proto_worldlines",
                    "source": "subjectiveObserverCameras[*].timeFrames[*].visibleProtoWorldlines",
                    "encoding": "moving dashed/candidate H3 markers in the observer camera",
                    "palette": "proto_worldline",
                },
                {
                    "field": "overlap_links",
                    "source": "observerModularTime.overlapLinks",
                    "encoding": "visible readback-link arcs",
                    "palette": "overlap_support",
                },
            ],
            "animationChannels": [
                {
                    "channel": "observer_modular_time",
                    "source": "subjectiveObserverCameras[*].timeFrames",
                    "timeSource": "subjectiveObserverCameras[*].timeFrames[*].cycle",
                    "encoding": "camera frame stepping through visible local readout",
                },
                {
                    "channel": "observer_visible_proto_worldline_motion",
                    "source": "subjectiveObserverCameras[*].timeFrames[*].visibleProtoWorldlines",
                    "timeSource": "subjectiveObserverCameras[*].timeFrames[*].cycle",
                    "encoding": "animate each sighting by observerLocalReadout.u/v/range in the selected observer camera",
                },
                {
                    "channel": "overlap_repair_trajectory",
                    "source": "observerModularTime.overlapLinks[*].repairTrajectory",
                    "timeSource": "observerModularTime.overlapLinks[*].repairTrajectory[*].cycle",
                    "encoding": "link pulse along observer-overlap supports",
                },
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
            "promotionReceiptsRequired": [
                "observer_modular_time_receipt",
                "observer_facing_3p1d_h3_experience_receipt",
            ],
            "nonClaims": [
                "hidden global camera",
                "objective global time",
                "omniscient bulk state readout",
            ],
            "claimBoundary": (
                "Subjective observer cameras are visible-readout cameras. They do not expose hidden "
                "state or objective global time. visibleProtoWorldlines are diagnostic projections of "
                "exported H3 proto-worldline events into an observer-local camera; they are not particle "
                "matter unless particle receipts pass."
            ),
        },
        "emergentCurvedSpacetime": {
            "viewId": "emergentCurvedSpacetime",
            "sectionKind": "emergent_curved_spacetime_proxy",
            "label": "Emergent curved-spacetime proxy",
            "visualMetaphor": "stress_sources_warp_observer_facing_h3_chart",
            "description": (
                "Render consensus H3 object packets and proto-worldline events as quotient-visible "
                "source-density rows over the observer-facing H3 chart. Surface displacement, grid "
                "bending, or lensing should be driven by compactificationFactor, "
                "emergentSpatialScaleFactor, curvatureProxyPoints, continuousBulkField, and timeSlices. "
                "This is a diagnostic curvature/compaction visualization, not a promoted gravity prediction."
            ),
            "dataSources": [
                "emergentCurvedSpacetime.sourceMath",
                "emergentCurvedSpacetime.curvatureProxyPoints",
                "emergentCurvedSpacetime.spacetimeCompactionField",
                "emergentCurvedSpacetime.continuousBulkField",
                "emergentCurvedSpacetime.timeSlices",
                "visualizationViews.emergentCurvedSpacetime.einsteinBranchEntry",
                "consensusBulk.objects",
                "consensusBulk.protoParticleCandidates.worldlines",
                "visualizationRenderData.sceneGraph.curvedSpacetime",
            ],
            "primaryFields": [
                "emergentCurvedSpacetime.curvatureProxyPoints[*].position",
                "emergentCurvedSpacetime.curvatureProxyPoints[*].sourceDensity",
                "emergentCurvedSpacetime.curvatureProxyPoints[*].h3GreenPotential",
                "emergentCurvedSpacetime.curvatureProxyPoints[*].curvaturePotential",
                "emergentCurvedSpacetime.curvatureProxyPoints[*].compactificationFactor",
                "emergentCurvedSpacetime.curvatureProxyPoints[*].emergentSpatialScaleFactor",
                "emergentCurvedSpacetime.curvatureProxyPoints[*].sourceDensityAncestry",
                "emergentCurvedSpacetime.continuousBulkField.volumeSamples[*].normalizedDensity",
                "emergentCurvedSpacetime.continuousBulkField.volumeSamples[*].compactificationFactor",
                "emergentCurvedSpacetime.continuousBulkField.temporalSliceSamples[*].cycle",
                "emergentCurvedSpacetime.timeSlices[*].totalCurvaturePotential",
            ],
            "renderLayers": [
                {
                    "layer": "h3_object_stress_sources",
                    "source": "emergentCurvedSpacetime.curvatureProxyPoints[sourceKind=consensus_h3_object]",
                },
                {
                    "layer": "proto_worldline_stress_sources",
                    "source": "emergentCurvedSpacetime.curvatureProxyPoints[sourceKind=proto_worldline_event]",
                },
                {
                    "layer": "curvature_proxy_surface",
                    "source": "emergentCurvedSpacetime.fieldSamples",
                },
                {
                    "layer": "continuous_h3_bulk_density",
                    "source": "emergentCurvedSpacetime.continuousBulkField.volumeSamples",
                    "encoding": "volume fog or isosurface from normalizedDensity and compactificationFactor",
                },
                {
                    "layer": "continuous_h3_bulk_slices",
                    "source": "emergentCurvedSpacetime.continuousBulkField.sliceSamples",
                    "encoding": "warped grid slices from emergentSpatialScaleFactor",
                },
                {
                    "layer": "gravity_receipt_badges",
                    "source": "emergentCurvedSpacetime.receipts",
                },
                {
                    "layer": "einstein_branch_entry_gate",
                    "source": "visualizationViews.emergentCurvedSpacetime.einsteinBranchEntry",
                },
            ],
            "visualEncodings": [
                {
                    "field": "curvaturePotential",
                    "source": "emergentCurvedSpacetime.curvatureProxyPoints",
                    "encoding": "surface displacement, contour brightness, or metric-grid bend strength",
                    "palette": "curvature_proxy",
                },
                {
                    "field": "compactificationFactor",
                    "source": "emergentCurvedSpacetime.curvatureProxyPoints",
                    "encoding": "local grid spacing contraction or lensing strength",
                    "palette": "spatial_compaction",
                },
                {
                    "field": "emergentSpatialScaleFactor",
                    "source": "emergentCurvedSpacetime.curvatureProxyPoints",
                    "encoding": "local H3 cell size multiplier; smaller values mean stronger compactification",
                    "palette": "scale_factor",
                },
                {
                    "field": "sourceDensity",
                    "source": "emergentCurvedSpacetime.curvatureProxyPoints",
                    "encoding": "source glyph size and local grid pull",
                    "palette": "quotient_visible_source",
                },
                {
                    "field": "normalizedDensity",
                    "source": "emergentCurvedSpacetime.continuousBulkField.volumeSamples",
                    "encoding": "continuous volume opacity or isosurface threshold",
                    "palette": "bulk_density",
                },
                {
                    "field": "production_gravity_receipt",
                    "source": "emergentCurvedSpacetime.receipts",
                    "encoding": "blocked promotion badge unless true",
                    "palette": "receipt_gate",
                },
                {
                    "field": "einstein_branch_entry_receipt",
                    "source": "emergentCurvedSpacetime.receipts",
                    "encoding": "closed branch-entry gate unless E0 manifest sidecar receipts pass",
                    "palette": "receipt_gate",
                },
            ],
            "animationChannels": [
                {
                    "channel": "curvature_proxy_time_slices",
                    "source": "emergentCurvedSpacetime.timeSlices",
                    "timeSource": "emergentCurvedSpacetime.timeSlices[*].relativeTime",
                    "encoding": "animate local curvature/stress proxy as proto-worldline events move",
                },
                {
                    "channel": "continuous_curvature_field_slices",
                    "source": "emergentCurvedSpacetime.continuousBulkField.temporalSliceSamples",
                    "timeSource": "emergentCurvedSpacetime.continuousBulkField.temporalSliceSamples[*].relativeTime",
                    "encoding": "animate the continuous z-slice field through observer modular cycles",
                },
                {
                    "channel": "worldline_source_motion",
                    "source": "consensusBulk.protoParticleCandidates.worldlines[*].events",
                    "timeSource": "consensusBulk.protoParticleCandidates.worldlines[*].events[*].cycle",
                    "encoding": "move stress-source glyphs along exported H3 events",
                },
            ],
            "einsteinBranchEntry": einstein_branch,
            "receipts": {
                "observer_facing_consensus_3d_bulk_readout_receipt": observer_facing_bulk_receipt,
                "strict_neutral_third_person_bulk_receipt": bool(
                    bulk_receipts.get("strict_neutral_third_person_bulk_receipt", False)
                ),
                "bulk_worldline_precursor_receipt": bool(
                    proto_receipts.get("bulk_worldline_precursor_receipt", False)
                ),
                "particle_matter_receipt": bool(proto_receipts.get("particle_matter_receipt", False)),
                "emergent_curved_spacetime_visualization_receipt": bool(
                    bulk_payload.get("objects") or bulk_payload.get("protoParticleCandidates", {}).get("worldlines")
                ),
                "einstein_branch_entry_receipt": einstein_branch_receipt,
                "EINSTEIN_BRANCH_ENTRY_RECEIPT": einstein_branch_receipt,
                "OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1": einstein_branch_receipt,
                "raw_production_gravity_requested": raw_production_gravity,
                "raw_physical_gravity_requested": raw_physical_gravity,
                "production_gravity_receipt": production_gravity_receipt,
                "physical_gravity_prediction": physical_gravity_prediction,
                "einstein_equation_solution_receipt": einstein_branch_receipt,
            },
            "exportSufficiency": "sufficient_for_curved_spacetime_proxy_visualization_not_physical_gravity",
            "promotionReceiptsRequired": [
                "observer_facing_consensus_3d_bulk_readout_receipt",
                "strict_neutral_third_person_bulk_receipt",
                "particle_matter_receipt",
                "einstein_branch_entry_receipt",
                "production_gravity_receipt",
                "einstein_equation_solution_receipt",
            ],
            "nonClaims": [
                "Einstein-equation solution",
                "physical gravity prediction",
                "production gravity model",
                "Einstein branch-entry proof from generic finite consensus",
                "strict neutral third-person bulk",
                "matter stress tensor",
                "gravitational merger event",
            ],
            "claimBoundary": (
                "Curved-spacetime data here are normalized stress/curvature proxies over the observer-facing "
                "H3 chart. Render as an explanatory diagnostic of where gravity would be expected to appear "
                "if the stronger OPH matter/gravity gates pass. Production gravity remains blocked unless "
                "the E0 Einstein bridge manifest sidecar receipts and gravity promotion gates pass."
            ),
        },
        "effectiveStringTheory": {
            "viewId": "effectiveStringTheory",
            "sectionKind": "effective_string_theory_edge_worldsheet",
            "label": "Effective string-theory edge/worldsheet view",
            "visualMetaphor": "edge_cycles_as_worldsheet_ribbons",
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
                "effectiveStringTheory.finiteEdgeStringVibrationSamples",
                "screen.clusters.snapshots",
                "consensusBulk.protoParticleCandidates.worldlines",
                "consensusBulk.objects",
                "visualizationViews.effectiveStringTheory.organicDefectPopulation",
                "visualizationViews.effectiveStringTheory.twoDefectStressContractionAssay",
                "visualizationViews.effectiveStringTheory.freeTwoDefectDynamics",
                "visualizationViews.effectiveStringTheory.stringVacuumSelector",
            ],
            "primaryFields": [
                "smallUniverse.cycles.exactConsensus",
                "smallUniverse.repairFrames",
                "effectiveStringTheory.finiteEdgeStringVibrationSamples[*].normalizedAmplitude",
                "effectiveStringTheory.finiteEdgeStringVibrationSamples[*].loopPhase",
                "consensusBulk.protoParticleCandidates.worldlines[*].events",
            ],
            "renderLayers": [
                {"layer": "cyclic_edge_normal_forms", "source": "smallUniverse.cycles"},
                {"layer": "repair_history_worldsheet_ribbons", "source": "smallUniverse.repairFrames"},
                {
                    "layer": "finite_edge_string_vibration_pulses",
                    "source": "effectiveStringTheory.finiteEdgeStringVibrationSamples",
                },
                {"layer": "collar_defect_tracks", "source": "screen.clusters.snapshots"},
                {"layer": "h3_worldline_overlay", "source": "consensusBulk.protoParticleCandidates.worldlines"},
                {
                    "layer": "organic_defect_population",
                    "source": "visualizationViews.effectiveStringTheory.organicDefectPopulation",
                },
                {
                    "layer": "controlled_two_defect_stress_contraction",
                    "source": "visualizationViews.effectiveStringTheory.twoDefectStressContractionAssay",
                },
                {
                    "layer": "free_two_defect_dynamics",
                    "source": "visualizationViews.effectiveStringTheory.freeTwoDefectDynamics",
                },
                {
                    "layer": "string_vacuum_selector_sieve",
                    "source": "visualizationViews.effectiveStringTheory.stringVacuumSelector",
                },
            ],
            "visualEncodings": [
                {
                    "field": "finite_edge_cycle",
                    "source": "smallUniverse.cycles",
                    "encoding": "closed loops or edge strings on the finite carrier",
                    "palette": "cycle_string",
                },
                {
                    "field": "repair_history",
                    "source": "smallUniverse.repairFrames",
                    "encoding": "ribbon swept over repair time",
                    "palette": "worldsheet_ribbon",
                },
                {
                    "field": "normalizedAmplitude",
                    "source": "effectiveStringTheory.finiteEdgeStringVibrationSamples",
                    "encoding": "edge/ribbon pulse height or glow at loopPhase for each exact repair frame",
                    "palette": "finite_edge_vibration",
                },
                {
                    "field": "screen_defect_track",
                    "source": "screen.clusters.snapshots",
                    "encoding": "moving collar/defect string segment",
                    "palette": "defect_string",
                },
                {
                    "field": "h3_proto_particle_worldline",
                    "source": "consensusBulk.protoParticleCandidates.worldlines",
                    "encoding": "H3 trajectory overlay with dashed non-particle styling until receipts pass",
                    "palette": "proto_worldline",
                },
                {
                    "field": "organic_defect_edge_string",
                    "source": "visualizationViews.effectiveStringTheory.organicDefectPopulation.worldlines",
                    "encoding": "10-20 seeded organic defect tracks as points, ribbons, or edge-string polylines",
                    "palette": "defect_string",
                },
                {
                    "field": "free_two_defect_contact_outcome",
                    "source": "visualizationViews.effectiveStringTheory.freeTwoDefectDynamics",
                    "encoding": "animate randomized two-defect support overlap as scatter/bind/annihilate/pass-through",
                    "palette": "proto_worldline_contact",
                },
                {
                    "field": "string_vacuum_gate_status",
                    "source": "visualizationViews.effectiveStringTheory.stringVacuumSelector",
                    "encoding": "candidate sieve bars, gate matrix, and open-certificate badges",
                    "palette": "selector_gate_status",
                },
            ],
            "animationChannels": [
                {
                    "channel": "edge_string_sweep",
                    "source": "smallUniverse.edges",
                    "timeSource": "smallUniverse.repairFrames[*].step",
                    "encoding": "edge activation traveling along finite cycles",
                },
                {
                    "channel": "finite_edge_string_vibration",
                    "source": "effectiveStringTheory.finiteEdgeStringVibrationSamples",
                    "timeSource": "effectiveStringTheory.finiteEdgeStringVibrationSamples[*].frameStep",
                    "encoding": "exact finite repair edge-pulse samples; animate loopPhase and normalizedAmplitude",
                },
                {
                    "channel": "worldsheet_ribbon_growth",
                    "source": "smallUniverse.repairFrames",
                    "timeSource": "smallUniverse.repairFrames[*].step",
                    "encoding": "ribbon surface swept by repair-frame sequence",
                },
                {
                    "channel": "h3_worldline_motion",
                    "source": "consensusBulk.protoParticleCandidates.worldlines[*].events",
                    "timeSource": "consensusBulk.protoParticleCandidates.worldlines[*].events[*].cycle",
                    "encoding": "track interpolation through H3 event samples",
                },
                {
                    "channel": "organic_defect_population_motion",
                    "source": "visualizationViews.effectiveStringTheory.organicDefectPopulation.worldlines[*].events",
                    "timeSource": "visualizationViews.effectiveStringTheory.organicDefectPopulation.worldlines[*].events[*].cycle",
                    "encoding": "animate seeded organic defect points/ribbons through H3",
                },
                {
                    "channel": "free_two_defect_contact_dynamics",
                    "source": "visualizationViews.effectiveStringTheory.freeTwoDefectDynamics.trajectoryRows",
                    "timeSource": "visualizationViews.effectiveStringTheory.freeTwoDefectDynamics.trajectoryRows[*].cycle",
                    "encoding": "animate randomized pair motion, transverse perturbation, support overlap, and explicit contact outcome",
                },
                {
                    "channel": "string_selector_gate_focus",
                    "source": "visualizationViews.effectiveStringTheory.stringVacuumSelector.acceptanceGates",
                    "timeSource": None,
                    "encoding": "step through OPH target gates, encoded candidate scores, and open certificate gates",
                },
            ],
            "twoDefectStressContractionAssay": stress_contraction,
            "organicDefectPopulation": organic_defects,
            "freeTwoDefectDynamics": free_dynamics,
            "stringVacuumSelector": string_selector,
            "receipts": {
                "finite_consensus_theorem_receipt": bool(
                    small_receipts.get("FINITE_CONSENSUS_THEOREM_RECEIPT", False)
                ),
                "bulk_worldline_precursor_receipt": bool(
                    proto_receipts.get("bulk_worldline_precursor_receipt", False)
                ),
                "particle_matter_receipt": bool(proto_receipts.get("particle_matter_receipt", False)),
                "observer_facing_consensus_3d_bulk_readout_receipt": observer_facing_bulk_receipt,
                "finite_edge_string_vibration_receipt": bool(
                    small_receipts.get("FINITE_CONSENSUS_THEOREM_RECEIPT", False)
                ),
                "critical_edge_cft_receipt": False,
                "string_vacuum_selector_visualization_receipt": bool(
                    string_selector.get("receipts", {}).get("string_vacuum_selector_visualization_receipt", False)
                ),
                "encoded_structural_audit_data_receipt": bool(
                    string_selector.get("receipts", {}).get("encoded_structural_audit_data_receipt", False)
                ),
                "bd_operator_safe_candidate_selected_in_encoded_audit": bool(
                    string_selector.get("receipts", {}).get(
                        "bd_operator_safe_candidate_selected_in_encoded_audit", False
                    )
                ),
                "two_defect_stress_contraction_assay_receipt": bool(
                    stress_contraction.get("receipts", {}).get(
                        "two_defect_stress_contraction_assay_receipt", False
                    )
                ),
                "organic_defect_population_receipt": bool(
                    organic_defects.get("receipts", {}).get("organic_defect_population_receipt", False)
                ),
                "organic_proto_worldline_visualization_receipt": bool(
                    organic_defects.get("receipts", {}).get(
                        "organic_proto_worldline_visualization_receipt", False
                    )
                ),
                "gravity_like_attraction_diagnostic_receipt": bool(
                    stress_contraction.get("receipts", {}).get(
                        "gravity_like_attraction_diagnostic_receipt", False
                    )
                ),
                "free_two_defect_dynamics_receipt": bool(
                    free_dynamics.get("receipts", {}).get("free_two_defect_dynamics_receipt", False)
                ),
                "gravity_like_free_dynamics_diagnostic_receipt": bool(
                    free_dynamics.get("receipts", {}).get(
                        "gravity_like_free_dynamics_diagnostic_receipt", False
                    )
                ),
                "einstein_branch_entry_receipt": einstein_branch_receipt,
                "EINSTEIN_BRANCH_ENTRY_RECEIPT": einstein_branch_receipt,
                "OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1": einstein_branch_receipt,
                "raw_production_gravity_requested": raw_production_gravity,
                "raw_physical_gravity_requested": raw_physical_gravity,
                "production_gravity_receipt": production_gravity_receipt,
                "physical_gravity_prediction": physical_gravity_prediction,
                "bd_full_cohomology_certificate_receipt": False,
                "bd_z4r_compactification_realization_receipt": False,
                "bd_threshold_spectrum_certificate_receipt": False,
                "bd_moduli_locking_certificate_receipt": False,
                "global_singleton_string_vacuum_receipt": False,
            },
            "exportSufficiency": "sufficient_for_schematic_edge_string_view_not_critical_worldsheet_claim",
            "promotionReceiptsRequired": [
                "finite_consensus_theorem_receipt",
                "bulk_worldline_precursor_receipt",
                "critical_edge_cft_receipt",
                "virasoro_receipt",
                "sugawara_receipt",
                "spin_structure_receipt",
                "bd_full_cohomology_certificate_receipt",
                "bd_z4r_compactification_realization_receipt",
                "bd_threshold_spectrum_certificate_receipt",
                "bd_moduli_locking_certificate_receipt",
            ],
            "nonClaims": [
                "critical string CFT",
                "heterotic worldsheet derivation",
                "completed BD compactification certificate",
                "global string landscape singleton",
                "production matter particles",
                "physical gravity prediction",
                "strict neutral third-person bulk",
            ],
            "claimBoundary": (
                "This is an effective edge-string diagnostic view. It must not be labeled a proven critical "
                "heterotic worldsheet until the finite-carrier critical-edge receipt suite passes."
            ),
        },
        "fractionalQuotientSectors": {
            "viewId": "fractionalQuotientSectors",
            "sectionKind": "fractional_quotient_sector_line_fan",
            "label": "Fractional quotient-sector line fan",
            "visualMetaphor": "fractional_hall_collar_sector_ledger_with_optical_lines",
            "description": (
                "Render fractional Hall or fractional Chern sectors as quotient-visible ledger nodes, "
                "then show optical fractional-exciton peaks as a line fan over the same ledger. The view "
                "is receipt-gated: source freezing, topological promotion, quotient correctness, binding "
                "drift, line-fan injectivity, refinement, and no-target-leak gates must remain visible."
            ),
            "dataSources": [
                "visualizationViews.fractionalQuotientSectors.fractionalReport",
                "visualizationViews.fractionalQuotientSectors.fractionalReport.topological_ledger",
                "visualizationViews.fractionalQuotientSectors.fractionalReport.line_fan.peaks",
                "visualizationViews.fractionalQuotientSectors.receipts",
            ],
            "primaryFields": [
                "fractionalReport.topological_ledger.sectors",
                "fractionalReport.topological_ledger.charges",
                "fractionalReport.line_fan.peaks[*].energy",
                "fractionalReport.line_fan.peaks[*].gate_slope",
                "fractionalReport.line_fan.peaks[*].tau",
                "fractionalReport.line_fan.peaks[*].total_charge",
            ],
            "renderLayers": [
                {"layer": "hall_collar_sector_ledger", "source": "fractionalReport.topological_ledger"},
                {"layer": "optical_line_fan", "source": "fractionalReport.line_fan.peaks"},
                {"layer": "neutral_fractional_exciton_badge", "source": "fractionalReport.slope_certificate"},
                {"layer": "fractional_receipt_gates", "source": "receipts"},
            ],
            "visualEncodings": [
                {
                    "field": "tau",
                    "source": "fractionalReport.line_fan.peaks",
                    "encoding": "line color and sector-node link",
                    "palette": "fractional_sector",
                },
                {
                    "field": "total_charge",
                    "source": "fractionalReport.line_fan.peaks",
                    "encoding": "line slope marker; zero charge with tau != 1 gets neutral fractional styling",
                    "palette": "charge_slope",
                },
                {
                    "field": "binding_drift_bounded",
                    "source": "receipts.binding_drift_bounded",
                    "encoding": "charge-slope interpretation gate",
                    "palette": "receipt_gate",
                },
            ],
            "animationChannels": [
                {
                    "channel": "gate_slope_sweep",
                    "source": "fractionalReport.line_fan.peaks",
                    "timeSource": None,
                    "encoding": "tilt line fan by gate_slope and show binding-drift uncertainty band",
                },
                {
                    "channel": "sector_to_peak_highlight",
                    "source": "fractionalReport.optical_module.sectors",
                    "timeSource": None,
                    "encoding": "hover or scrub from topological sector ledger to optical peak",
                },
            ],
            "fractionalReport": fractional_quotient.get("fractionalReport", {}),
            "receipts": fractional_quotient.get("receipts", {}),
            "renderHints": fractional_quotient.get("renderHints", []),
            "exportSufficiency": "sufficient_for_fractional_quotient_sandbox_visualization_not_material_proof",
            "promotionReceiptsRequired": [
                "source_hamiltonian_frozen",
                "phase_certificate_injective",
                "topological_sector_ledger",
                "line_fan_decomposition",
                "binding_drift_bounded",
                "no_target_leak",
                "refinement_compatibility",
                "canonicalizer_idempotence",
                "representative_invariance",
                "quotient_lumpability",
            ],
            "nonClaims": [
                "material-specific fractional Chern proof",
                "optical peak assignment without binding-drift bound",
                "sector selection by normal form alone",
                "posterior-fitted material Hamiltonian",
            ],
            "claimBoundary": fractional_quotient.get("claimBoundary"),
        },
        "silenceToObservation": {
            "viewId": "silenceToObservation",
            "sectionKind": "silence_to_observation",
            "label": "Silence-to-observation bridge",
            "visualMetaphor": "scale_compressed_readout_emergence",
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
            "visualEncodings": [
                {
                    "field": "silent_initial_state",
                    "source": "pnSilenceToObservation.silenceInitialState",
                    "encoding": "dark or low-contrast initial field",
                    "palette": "silence",
                },
                {
                    "field": "repair_depth",
                    "source": "pnSilenceToObservation.finiteRegulatorDepth",
                    "encoding": "radial progress or depth bars",
                    "palette": "repair_depth",
                },
            ],
            "animationChannels": [
                {
                    "channel": "silence_to_readout",
                    "source": "pnSilenceToObservation",
                    "timeSource": "pnSilenceToObservation.finiteRegulatorDepth",
                    "encoding": "staged fade-in from silence to observer readout",
                },
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
            "promotionReceiptsRequired": [
                "scale_compressed_pn_silence_to_observation_receipt",
            ],
            "nonClaims": [
                "literal global N simulation",
                "physical cosmology prediction",
            ],
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
:root {{ color-scheme: dark; --bg:#0f1115; --panel:#171b21; --ink:#eef2f6; --muted:#aab4be; --line:#303844; --pass:#1d5f3a; --blocked:#5b4a24; --fail:#683033; --accent:#66d9ef; --gold:#f5c66b; }}
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
.blocked {{ background:var(--blocked); color:#ffe5a6; }}
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
function gate(label, ok, optional=false) {{ const klass=ok?'pass':(optional?'blocked':'fail'); const text=ok?'pass':(optional?'not promoted':'closed'); return `<span class="gate ${{klass}}">${{label}}: ${{text}}</span>`; }}
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
  const hs=DATA.consensusBulk.h3ChartStatus||{{}};
  document.getElementById("h3Note").textContent = `${{objs.length}} consensus object packets and ${{lines.length}} holonomy/proto-particle candidate worldlines in the derived H3 chart. H3 chart status=${{hs.displayStatus || "unknown"}}. Observer-facing consensus bulk=${{DATA.consensusBulk.receipts.observer_facing_consensus_3d_bulk_readout_receipt || DATA.consensusBulk.receipts.theorem_assisted_consensus_3d_bulk_readout_receipt}}, chart-blind neutral quotient=${{DATA.consensusBulk.receipts.chart_blind_strict_neutral_quotient_bulk_receipt || DATA.consensusBulk.receipts.strict_neutral_third_person_bulk_receipt}} (not promoted is not a chart error), bulk worldline precursor=${{pr.bulk_worldline_precursor_receipt}}, particle matter=${{pr.particle_matter_receipt}}. ${{DATA.consensusBulk.claimBoundary}}`;
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
    gate("chart-blind neutral quotient", cb.chart_blind_strict_neutral_quotient_bulk_receipt || cb.strict_neutral_third_person_bulk_receipt, true)+
    gate("usable CMB comparison", cmb.USABLE_PHYSICAL_CMB_DATA_RECEIPT)+
    gate("physical CMB prediction", cmb.PHYSICAL_CMB_PREDICTION_RECEIPT, true);
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


def _visualization_instructions(viewer_path: Path | None, payload_path: Path, payload: dict[str, Any]) -> str:
    viewer_block = (
        f"""Open the standalone viewer:

```bash
open {viewer_path}
```
"""
        if viewer_path is not None
        else """No embedded standalone viewer was emitted for this export. This is a full-data payload; build the app against the JSON and sidecar files directly.
"""
    )
    return f"""# OPH Universe Visualization Instructions

{viewer_block}

Data payload for custom viewers:

```bash
{payload_path}
```

What to inspect:

- Panel 1 shows the fluctuating-vacuum diagnostic view: the finite S2 observer screen/boundary readback from the larger observer-flow run. Colors are screen readback fields; rings are screen-local defect/holonomy residues. This is a diagnostic OPH readback field, not a literal QFT vacuum unless a future receipt says so.
- The same view may include `visualizationViews.fluctuatingQuantumVacuum.yangMillsGapCertificate`: finite SU(2) Wilson-lattice plaquette/Wilson/Polyakov traces and transfer-gap proxies. Show its finite diagnostic receipt separately from `YANG_MILLS_GAP_REPRODUCED_RECEIPT`, which should remain closed unless a future continuum certificate promotes it.
- Panel 2 shows one deterministic repair path through the exact 12-patch mini-universe certificate. The full certificate checks all finite states/schedules; the slider is a readable path through that certified graph.
- Panel 3 shows the observer-camera view and observer-local modular time. Each dot is an observer-like self-reading row with local support, records, readback hash, and modular-depth readout. Use the observer selector to inspect one observer's objective readout across its modular-time frames: record packet, object packet, transition step, local packet histograms, and the global trace cycle used only for synchronization.
- The payload also exports `subjectiveObserverCameras`: first-person rendering cameras derived from visible observer-local readouts. These are the right inputs for a subjective observer camera map.
- Panel 4 shows the emergent curved-spacetime proxy view. It uses `emergentCurvedSpacetime.sourceMath`, `curvatureProxyPoints`, `spacetimeCompactionField`, `continuousBulkField`, and `timeSlices` to render quotient-visible source density, H3 Green potential, curvature, compactification, continuous volume samples, and warped slices over the observer-facing H3 chart. It is a diagnostic warped-grid/field layer, not production gravity or a physical metric unless `einstein_branch_entry_receipt`, `production_gravity_receipt`, and related promotion receipts are true.
- Panel 5 shows the effective string-theory diagnostic view. Consensus object packets are shared readback/object packets from overlapping observers. Magenta/red tracks are holonomy/defect worldlines fitted into the same H3 chart: proto-particle candidates and edge-worldline/collar diagnostics, not matter particles or a critical worldsheet unless the corresponding receipts pass.
- For the string view, `effectiveStringTheory.finiteEdgeStringVibrationSamples` is the exact finite repair/cycle edge-pulse layer. Animate `frameStep`, `loopPhase`, and `normalizedAmplitude`; do not substitute generic sine-wave string modes.
- Panel 6 shows the scale-compressed P/N silence-to-observation witness: initial record silence, P detuning, finite regulator depth, and observer/H3 readout emergence. This is not a literal brute-force simulation of astronomical N_CRC.
- Panel 7 shows usable CMB comparison diagnostics when present. `comparableObservations` also carries compact measurement-lane summaries for other public-data-facing diagnostics. None of this is a physical prediction unless the relevant prediction receipt passes.
- `visualizationViews.fluctuatingQuantumVacuum`, `visualizationViews.observerCamera`, `visualizationViews.emergentCurvedSpacetime`, and `visualizationViews.effectiveStringTheory` are the canonical view contracts for a custom visualizer.
- `paperAccuracy` is the fail-closed paper-accuracy guard. Use it to render which claims are allowed, which promotions remain blocked, and why visual similarity alone is not a receipt.

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

Paper-accuracy requirement:

- Read `paperAccuracy.checks` and render blocked promotions as closed gates, not errors.
- Do not promote visual similarity, apparent attraction, CMB-shape resemblance, or finite lattice gaps beyond the exact receipts in `paperAccuracy.receipts`.
- Treat `EINSTEIN_BRANCH_ENTRY_RECEIPT` / `OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1` as the mandatory gate for any production-gravity wording. If false, show the E0 manifest's missing sidecar receipts as a closed promotion gate, not a simulation error.

Required views:

1. **Fluctuating quantum vacuum / finite screen view**
   - Render `screen.points` as an S2/equirectangular or sphere view.
   - Color by `screen.values` and label the field with `screen.fieldName`.
   - Overlay `screen.clusters.snapshots[*].clusters` as repair/holonomy residues.
   - Use `visualizationViews.fluctuatingQuantumVacuum` for the canonical layer list and claim boundary.
   - If present, render `visualizationViews.fluctuatingQuantumVacuum.yangMillsGapCertificate` as finite SU(2) gauge diagnostics: plaquette trace, Wilson/Polyakov traces, refinement gap rows, and promotion blockers.
   - Explain that this is the observer boundary/readback surface, not a literal QFT vacuum or a pre-existing 3D bulk.
   - Never label finite SU(2) diagnostics as a reproduced Yang-Mills mass gap unless `YANG_MILLS_GAP_REPRODUCED_RECEIPT` is true.

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
   - Render `effectiveStringTheory.finiteEdgeStringVibrationSamples` as the exact finite edge-pulse/vibration layer. Animate `frameStep`, `loopPhase`, and `normalizedAmplitude`; never replace it with generic sine-wave string oscillations.
   - Render `screen.clusters.snapshots[*].clusters` as collar/defect fluctuation markers.
   - Render `consensusBulk.objects` as a 3D scatter/cloud.
   - Size by `observerCount`; color by `h3CompactnessNormalized`.
   - Render `consensusBulk.protoParticleCandidates.worldlines[*].events` as H3 tracks.
   - Label the selected track source with `consensusBulk.protoParticleCandidates.worldlineSource`.
   - Treat `proto_particle_worldlines.csv` as a legacy fallback only; do not let stale sidecars
     override organic or free dynamics JSON reports.
   - Use neutral wording: "edge-worldline diagnostic", "consensus object packet", and "holonomy/proto-particle candidate" unless the stronger receipts pass.
   - Do not label it a critical string CFT unless a future critical-edge receipt is true.
   - Gate labels must show observer-facing H3 consensus bulk and chart-blind neutral quotient bulk separately.

6. **Emergent curved-spacetime proxy view**
   - Use `visualizationViews.emergentCurvedSpacetime` for the canonical layer list and claim boundary.
   - Use `emergentCurvedSpacetime.continuousBulkField.volumeSamples` for the main continuous
     bulk-field rendering when it is available. Render it as fog, density points, an isosurface, or
     a warped volume, not just as isolated source balls.
   - Use `emergentCurvedSpacetime.continuousBulkField.sliceSamples` and `temporalSliceSamples` for
     warped grid slices and animated field slices.
   - Render `emergentCurvedSpacetime.curvatureProxyPoints` as stress-source glyphs in the observer-facing H3 chart.
   - Size glyphs by `sourceDensity`; drive grid bend, contour strength, or surface displacement by `curvaturePotential`.
   - Drive local spatial contraction by `compactificationFactor`; use `emergentSpatialScaleFactor` as the local grid/cell-size multiplier.
   - Show `sourceMath.sourceDefinition` and `gravitySourceInterpretation` so users see that the source is quotient-visible OPH stress/readout, not raw rest mass.
   - Animate `emergentCurvedSpacetime.timeSlices` and proto-worldline event cycles when available.
   - Display `einstein_branch_entry_receipt`, `production_gravity_receipt`, `physical_gravity_prediction`, and `einstein_equation_solution_receipt` separately.
   - Use `emergentCurvedSpacetime.einsteinBranchEntry` and `visualizationViews.emergentCurvedSpacetime.einsteinBranchEntry` for the E0 manifest, provenance tags, receipt rows, blockers, and claim boundary.
   - Never label this view as physical gravity, a solved metric, or a matter stress tensor unless the Einstein branch-entry and gravity receipts are true.

7. **CMB diagnostics view**
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


def _mean_float_or_none(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _packet_names(value: Any) -> list[str]:
    rows = value if isinstance(value, list) else []
    names: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            packet = row.get("packet", row.get("id", row.get("name")))
            if packet is not None:
                names.append(str(packet))
        elif row is not None:
            names.append(str(row))
    return names


def _vec_norm(values: list[float]) -> float:
    return float(math.sqrt(sum(float(value) * float(value) for value in values)))


def _unit_vec(values: list[float]) -> list[float]:
    norm = _vec_norm(values)
    if norm <= 0.0:
        return [0.0, 0.0, 0.0]
    return [float(value) / norm for value in values[:3]]


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(float(left[index]) * float(right[index]) for index in range(min(3, len(left), len(right)))))


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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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
