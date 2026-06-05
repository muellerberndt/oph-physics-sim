from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.claims import DEMO, RECOVERED_CORE, with_claim_metadata
from oph_fpe.bulk.bw_verifier import bw_residual_report
from oph_fpe.bulk.cap_geometry import cap_geometry_report, sample_caps
from oph_fpe.bulk.conformal_spatial_chart import conformal_h3_spatial_chart_report
from oph_fpe.bulk.h3_response_fit import modular_response_h3_report
from oph_fpe.bulk.markov_collar import collar_markov_report
from oph_fpe.bulk.modular_probe import state_derived_bw_report
from oph_fpe.bulk.modular_response_kernel import kernel_json_summary, modular_response_kernel
from oph_fpe.bulk.observer_reconstruction import (
    bulk_reconstruction_report,
    observer_distance_matrix,
    observer_similarity_components,
)
from oph_fpe.bulk.record_to_h3 import (
    defect_timeline_to_h3_report,
    observer_chart_object_population_report,
    record_populated_h3_report,
    support_profiles_to_h3_report,
)
from oph_fpe.bulk.transition_selection import transition_scale_selection_report
from oph_fpe.constants.oph_pixel import equal_cell_area_planck, equal_cell_entropy
from oph_fpe.consensus.lyapunov import lyapunov_descent_receipt
from oph_fpe.cosmology import write_freezeout_products
from oph_fpe.core.graph import fibonacci_sphere_points
from oph_fpe.core.pixel_scale import pixel_scale_from_config
from oph_fpe.core.screen_receipts import (
    central_record_born_report,
    edge_sector_heat_kernel_report,
    observer_checkpoint_restoration_report,
)
from oph_fpe.core.screen_microphysics import ports_per_patch_from_config, screen_microphysics_from_config
from oph_fpe.core.screen_ports import assign_echosahedral_ports
from oph_fpe.defects.array_s3_holonomy import (
    S3_CLASS,
    S3_MUL,
    array_holonomy_report,
    defect_interaction_report,
    defect_timeline_report,
    particle_likeness_report,
    s3_class_counts,
    s3_edge_class_density,
)
from oph_fpe.evidence import RunBundle
from oph_fpe.evidence.controls import mandatory_control_report
from oph_fpe.evidence.hashes import stable_json_hash
from oph_fpe.gauge.mar_sieve import standard_model_candidate_sieve
from oph_fpe.gauge.repair_projection import exact_repair_projection_receipt
from oph_fpe.observers import (
    assign_counterfactual_stability_from_records,
    extract_record_families,
    observer_consensus_report,
    observer_object_report,
    observer_view_rows,
    visible_object_packets,
)
from oph_fpe.scale.array_screen import _beta_at, _entropy, _group_order, _knn_edges, _modular_update, _node_signature, _write_csv
from oph_fpe.scale.parallel import jobs_from_config


def run_bw_array_config(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    seed = int(config.get("seed", 1))
    rng = np.random.default_rng(seed)
    outputs_cfg = config.get("outputs", {}) or {}
    output_profile = str(outputs_cfg.get("profile", "evidence"))
    write_jsonl_payloads = bool(outputs_cfg.get("write_jsonl", output_profile in {"evidence", "debug", "viewer"}))
    pixel_scale = pixel_scale_from_config(config)
    graph_cfg = config.get("graph", {})
    patch_count = int(graph_cfg.get("patch_count", 65_536))
    cell_area_planck = equal_cell_area_planck(patch_count, pixel_scale.cell_area_planck)
    cell_entropy = equal_cell_entropy(patch_count, pixel_scale.cell_area_planck)
    neighbors = int(graph_cfg.get("neighbors", ports_per_patch_from_config(config)))
    group_name = str(config.get("group", {}).get("name", "S3")).upper()
    group_order = _group_order(group_name)
    points = fibonacci_sphere_points(patch_count)
    left, right = _knn_edges(points, neighbors)
    edge_count = int(left.size)
    screen_microphysics = screen_microphysics_from_config(config, patch_count, edge_count)
    screen_ports = assign_echosahedral_ports(
        left,
        right,
        patch_count,
        ports_per_patch=ports_per_patch_from_config(config),
    )

    port_left, port_right, boundary_program_report = _initialize_port_packets(
        points,
        left,
        right,
        group_order=group_order,
        rng=rng,
        config=config.get("boundary_program", {}),
    )
    initial_port_left = port_left.copy()
    initial_port_right = port_right.copy()
    gauge = rng.integers(0, group_order, size=edge_count, dtype=np.int16)
    modular_depth = rng.random(patch_count, dtype=np.float64)
    modular_time = np.zeros(patch_count, dtype=np.float64)
    stable_count = np.zeros(patch_count, dtype=np.int16)
    committed = np.zeros(patch_count, dtype=bool)
    prev_signature = np.full(patch_count, -1, dtype=np.int64)
    degree = np.bincount(np.concatenate([left, right]), minlength=patch_count).astype(np.float64)
    degree = np.maximum(degree, 1.0)

    run_id = config.get("run_id") or _run_id(config.get("name", "bw_array"))
    bundle = RunBundle(out_dir, run_id)
    bundle.write_config(config)
    bundle.write_json("pixel_scale.json", pixel_scale.as_jsonable())
    bundle.write_json("pixel_report.json", pixel_scale.as_jsonable())
    bundle.write_json("screen_microphysics.json", screen_microphysics.as_jsonable())
    bundle.write_json("screen_ports.json", screen_ports.as_jsonable())
    bundle.write_json("boundary_program_report.json", boundary_program_report)

    dyn = config.get("dynamics", {})
    cycles = int(dyn.get("cycles", 64))
    repairs_per_cycle = int(dyn.get("repairs_per_cycle", edge_count // 4))
    commit_cycles = int(dyn.get("record_commit_cycles", 8))
    beta_schedule = dyn.get("beta_schedule", {})
    mod_cfg = config.get("modular_flow", {})
    trace: list[dict[str, Any]] = []
    final_repair_load = np.zeros(patch_count, dtype=float)
    final_mismatch_density = np.zeros(patch_count, dtype=float)
    cumulative_repair_load = np.zeros(patch_count, dtype=float)
    freezeout_cfg = config.get("cosmology", {}).get("freezeout", {})
    freezeout_commit_fraction = float(freezeout_cfg.get("commit_fraction", 0.95))
    freezeout_state: dict[str, Any] | None = None
    repair_peak_state: dict[str, Any] | None = None
    repair_peak_score = -1.0
    defects_cfg = config.get("defects", {})
    sector_repair_cfg = defects_cfg.get("sector_repair", {})
    timeline_cfg = defects_cfg.get("timeline", {})
    defect_timeline_enabled = bool(group_name == "S3" and timeline_cfg.get("enabled", False))
    defect_timeline_cycles = _timeline_cycles(cycles, int(timeline_cfg.get("sample_count", 8)))
    defect_gauge_snapshots: list[tuple[int, np.ndarray]] = []

    for cycle in range(cycles):
        beta = _beta_at(beta_schedule, cycle, cycles)
        mismatches = port_left != port_right
        phi_before = int(np.sum(mismatches))
        active = np.flatnonzero(mismatches)
        chosen = np.zeros(0, dtype=np.int64)
        chosen_delta = np.zeros(0, dtype=np.int16)
        if active.size:
            chosen_count = min(repairs_per_cycle, active.size)
            chosen = rng.choice(active, size=chosen_count, replace=False)
            chosen_delta = ((port_left[chosen].astype(np.int64) - port_right[chosen].astype(np.int64)) % group_order).astype(
                np.int16
            )
            direction = rng.random(chosen_count) < 0.5
            port_left[chosen[direction]] = port_right[chosen[direction]]
            port_right[chosen[~direction]] = port_left[chosen[~direction]]
        if chosen.size:
            _repair_sector_labels(gauge, chosen, chosen_delta, group_order=group_order, rng=rng, config=sector_repair_cfg)
        if chosen.size:
            cumulative_repair_load += (
                np.bincount(left[chosen], minlength=patch_count)
                + np.bincount(right[chosen], minlength=patch_count)
            ) / degree
        mismatches_after = port_left != port_right
        phi_after = int(np.sum(mismatches_after))
        incident_mismatch = (
            np.bincount(left, weights=mismatches_after.astype(float), minlength=patch_count)
            + np.bincount(right, weights=mismatches_after.astype(float), minlength=patch_count)
        )
        final_repair_load = incident_mismatch / degree
        final_mismatch_density = final_repair_load.copy()
        modular_depth, modular_time = _modular_update(
            points,
            left,
            right,
            degree,
            modular_depth,
            modular_time,
            final_repair_load,
            mod_cfg,
        )
        signature = _node_signature(port_left, port_right, left, right, patch_count)
        stable_count = np.where(signature == prev_signature, stable_count + 1, 1)
        prev_signature = signature
        committed |= stable_count >= commit_cycles
        committed_fraction = float(np.mean(committed))
        repair_peak_candidate = _state_snapshot(
            cycle=cycle,
            committed_fraction=committed_fraction,
            signature=signature,
            stable_count=stable_count,
            committed=committed,
            repair_load=final_repair_load,
            mismatch_density=final_mismatch_density,
            modular_depth=modular_depth,
            cumulative_repair_load=cumulative_repair_load,
        )
        repair_peak_candidate["mean_mismatch_density"] = float(np.mean(final_mismatch_density))
        repair_peak_candidate["std_mismatch_density"] = float(np.std(final_mismatch_density))
        repair_peak_candidate["mean_cumulative_repair_load"] = float(np.mean(cumulative_repair_load))
        repair_peak_candidate["std_cumulative_repair_load"] = float(np.std(cumulative_repair_load))
        repair_peak_candidate_score = float(
            repair_peak_candidate["mean_mismatch_density"] + repair_peak_candidate["std_mismatch_density"]
        )
        if repair_peak_candidate_score > repair_peak_score:
            repair_peak_score = repair_peak_candidate_score
            repair_peak_state = repair_peak_candidate
        if freezeout_state is None and committed_fraction >= freezeout_commit_fraction:
            freezeout_state = _state_snapshot(
                cycle=cycle,
                committed_fraction=committed_fraction,
                signature=signature,
                stable_count=stable_count,
                committed=committed,
                repair_load=final_repair_load,
                mismatch_density=final_mismatch_density,
                modular_depth=modular_depth,
                cumulative_repair_load=cumulative_repair_load,
            )
        trace.append(
            {
                "cycle": cycle,
                "beta": beta,
                "phi_before": phi_before,
                "phi": phi_after,
                "mismatch_edges": phi_after,
                "committed_records": int(np.sum(committed)),
                "committed_fraction": committed_fraction,
                "record_entropy": _entropy(signature[committed]) if np.any(committed) else 0.0,
                "modular_depth_mean": float(np.mean(modular_depth)),
                "modular_depth_std": float(np.std(modular_depth)),
            }
        )
        if defect_timeline_enabled and cycle in defect_timeline_cycles:
            defect_gauge_snapshots.append((cycle, gauge.copy()))

    if defect_timeline_enabled and (not defect_gauge_snapshots or defect_gauge_snapshots[-1][0] != cycles - 1):
        defect_gauge_snapshots.append((cycles - 1, gauge.copy()))

    fields_all = _observable_fields(
        port_left=port_left,
        port_right=port_right,
        left=left,
        right=right,
        gauge=gauge,
        patch_count=patch_count,
        signature=prev_signature,
        stable_count=stable_count,
        committed=committed,
        repair_load=final_repair_load,
        mismatch_density=final_mismatch_density,
        modular_depth=modular_depth,
        cumulative_repair_load=cumulative_repair_load,
    )
    bw_cfg = config.get("bw", {})
    observables = [str(name) for name in bw_cfg.get("observables", ["record_signature", "repair_load", "s3_class_density", "stable_count"])]
    fields = {}
    for name in observables:
        field_name = _projector_field_name(name)
        if field_name in fields_all:
            fields[field_name] = fields_all[field_name]
        elif name in fields_all:
            fields[name] = fields_all[name]
    fields = _regularize_support_visible_fields(fields, left, right, patch_count, bw_cfg)
    theta_values = [float(value) for value in bw_cfg.get("theta0", [0.35, 0.55, 0.75, 1.0, 1.25])]
    caps = sample_caps(
        points,
        count=int(bw_cfg.get("cap_count", 32)),
        theta_values=theta_values,
        seed=seed + 701,
        collar_width=_collar_width_from_config(bw_cfg, patch_count),
    )
    h3_support_cfg = config.get("h3_support_profiles", {})
    h3_caps, h3_cap_net_report = _h3_reconstruction_caps(
        points,
        caps,
        h3_support_cfg,
        seed=seed + 1707,
        patch_count=patch_count,
        fallback_theta_values=theta_values,
    )
    times = [float(value) for value in bw_cfg.get("times", [0.025, 0.05, 0.1, 0.2])]
    controls = _implemented_controls(config.get("controls", []))
    started = time.time()
    bw_report = bw_residual_report(
        points,
        fields,
        caps,
        times,
        k_interp=int(bw_cfg.get("k_interp", 8)),
        sim_k_interp=int(bw_cfg.get("sim_k_interp", 1)),
        n_jobs=jobs_from_config(bw_cfg.get("n_jobs", 1), default=1),
        controls=controls,
        seed=seed + 911,
        cell_entropy=cell_entropy,
        cell_area_planck=cell_area_planck,
    ).as_jsonable()
    bw_report["mode"] = "kinematic_geometric_bw_sanity"
    bw_report["claim_level"] = DEMO
    bw_report["receipt_name"] = "KINEMATIC_GEOMETRIC_BW_SANITY"
    bw_report["physical_claim"] = False
    bw_report["claim_boundary"] = (
        "kinematic geometry/interpolation sanity check for lambda_C(2*pi*t); "
        "not a state-derived modular-transport receipt"
    )
    bw_report["elapsed_seconds"] = time.time() - started
    bw_report["implemented_controls"] = controls
    bw_report["unimplemented_controls"] = [control for control in config.get("controls", []) if control not in controls]
    bw_report["regulator_collar"] = _collar_report(bw_cfg, patch_count, caps[0].collar_width if caps else 0.0)
    bw_report["support_visible_regularization"] = _regularization_report(bw_cfg)
    cap_report = cap_geometry_report(
        points,
        caps,
        cell_area_planck=cell_area_planck,
        cell_entropy=cell_entropy,
    )
    cap_report["claim_level"] = DEMO
    cap_report["times"] = times
    cap_report["regulator_collar"] = bw_report["regulator_collar"]
    conformal_chart_report = conformal_h3_spatial_chart_report(h3_caps)
    conformal_chart_report["claim_level"] = DEMO
    conformal_chart_report["receipt_name"] = "CONFORMAL_H3_CHART_RECEIPT"
    conformal_chart_report["h3_reconstruction_cap_net"] = h3_cap_net_report
    conformal_chart_report["bw_verifier_cap_count"] = len(caps)
    raw_observer_fields = _observer_raw_fields(
        left=left,
        right=right,
        gauge=gauge,
        patch_count=patch_count,
        signature=prev_signature,
        stable_count=stable_count,
        committed=committed,
        repair_load=final_repair_load,
        mismatch_density=final_mismatch_density,
        modular_depth=modular_depth,
        cumulative_repair_load=cumulative_repair_load,
    )
    if freezeout_state is None:
        freezeout_state = _state_snapshot(
            cycle=cycles - 1,
            committed_fraction=float(np.mean(committed)),
            signature=prev_signature,
            stable_count=stable_count,
            committed=committed,
            repair_load=final_repair_load,
            mismatch_density=final_mismatch_density,
            modular_depth=modular_depth,
            cumulative_repair_load=cumulative_repair_load,
        )
    if repair_peak_state is None:
        repair_peak_state = freezeout_state
    freezeout_fields = _observable_fields_from_snapshot(
        freezeout_state,
        left=left,
        right=right,
        gauge=gauge,
        patch_count=patch_count,
    )
    freezeout_raw_observer_fields = _observer_raw_fields_from_snapshot(
        freezeout_state,
        left=left,
        right=right,
        gauge=gauge,
        patch_count=patch_count,
    )
    repair_peak_raw_observer_fields = _observer_raw_fields_from_snapshot(
        repair_peak_state,
        left=left,
        right=right,
        gauge=gauge,
        patch_count=patch_count,
    )
    collar_report: dict[str, Any] = {}
    state_bw_report: dict[str, Any] = {}
    transition_selection_report: dict[str, Any] = {}
    if str(bw_cfg.get("mode", "kinematic_geometric_bw_sanity")) == "state_derived_modular_probe":
        collar_cfg = config.get("collar_markov", {})
        collar_report = collar_markov_report(
            points,
            caps,
            raw_observer_fields,
            packet_bins=collar_cfg.get("packet_bins", {}),
            max_triplets=int(collar_cfg.get("max_triplets", 4096)),
            seed=seed + 1301,
        )
        state_bw_report = state_derived_bw_report(
            points,
            caps,
            raw_observer_fields,
            collar_report,
            times=times,
            observables=[_projector_field_name(name) for name in bw_cfg.get("observables", ["record_signature"])],
            regularizers=[float(value) for value in bw_cfg.get("regularizer_a", [0.001])],
            controls=controls,
            state_mode=str(bw_cfg.get("state_mode", "cooccurrence_kernel")),
            target_operator_mode=str(bw_cfg.get("target_operator_mode", "nearest")),
            transition_response_time=float(bw_cfg.get("transition_response_time", min(times) if times else 0.025)),
            transition_response_scale=float(bw_cfg.get("transition_response_scale", 2.0 * math.pi)),
            max_basis=int(bw_cfg.get("max_basis", 96)),
            seed=seed + 1401,
        )
        selection_cfg = bw_cfg.get("selection", {})
        if selection_cfg.get("enabled", False):
            selection_sources = [str(source) for source in selection_cfg.get("sources", ["repair_affinity_response"])]
            if selection_cfg.get("include_declared_sanity", False) and "declared_geometric_sanity" not in selection_sources:
                selection_sources.append("declared_geometric_sanity")
            transition_selection_report = transition_scale_selection_report(
                points,
                caps,
                raw_observer_fields,
                times=[float(value) for value in selection_cfg.get("times", times)],
                observables=[_projector_field_name(name) for name in selection_cfg.get("observables", bw_cfg.get("observables", ["record_signature"]))],
                candidate_scales=[float(value) for value in selection_cfg.get("candidate_scales", [1.0, math.pi, 2.0 * math.pi, 4.0 * math.pi])],
                sources=selection_sources,
                declared_response_scale=float(selection_cfg.get("declared_response_scale", bw_cfg.get("transition_response_scale", 2.0 * math.pi))),
                max_basis=int(selection_cfg.get("max_basis", bw_cfg.get("max_basis", 64))),
                seed=seed + 1451,
                graph_response={
                    "left": left,
                    "right": right,
                    "port_left": port_left,
                    "port_right": port_right,
                    "group_order": group_order,
                    "patch_count": patch_count,
                },
                probe_steps=int(selection_cfg.get("probe_steps", 4)),
                probe_repairs_per_source=int(selection_cfg.get("probe_repairs_per_source", max(16, neighbors * 4))),
                probe_max_incident_edges=int(selection_cfg.get("probe_max_incident_edges", max(4, neighbors))),
                kms_response_scale=float(selection_cfg.get("kms_response_scale", 2.0 * math.pi)),
                kms_transport_steps=int(selection_cfg.get("kms_transport_steps", 8)),
            )
    observer_cfg = config.get("observers", {})
    observer_sample_count = int(observer_cfg.get("sample_count", min(64, patch_count)))
    observer_neighborhood = int(observer_cfg.get("neighborhood_size", max(8, min(64, neighbors * 4))))
    observer_rows = observer_view_rows(
        points,
        raw_fields=raw_observer_fields,
        visible_fields=fields,
        caps=caps,
        times=times,
        cell_area_planck=cell_area_planck,
        cell_entropy=cell_entropy,
        sample_count=observer_sample_count,
        neighborhood_size=observer_neighborhood,
        seed=seed + 1201,
    )
    consensus_report = observer_consensus_report(
        points,
        raw_fields=raw_observer_fields,
        cell_entropy=cell_entropy,
        sample_count=observer_sample_count,
        neighborhood_size=observer_neighborhood,
        seed=seed + 1201,
    )
    consensus_report["observer_relative_time_grid"] = times
    consensus_report["cap_count"] = len(caps)
    consensus_report["claim_boundary"] = (
        "observer-facing consensus readout; tracks what finite patch/cap observers can see "
        "before any third-person bulk embedding is inferred"
    )
    edge_sector_report = edge_sector_heat_kernel_report(
        gauge,
        group_name=group_name,
        beta=float((config.get("screen_microphysics_receipts", {}) or {}).get("edge_beta", 1.0)),
        s3_class=S3_CLASS if group_name == "S3" else None,
    )
    edge_sector_report = with_claim_metadata(
        {
            **edge_sector_report,
            "EDGE_HEAT_KERNEL_RECEIPT": bool(edge_sector_report.get("receipt", False)),
        },
        claim_level=RECOVERED_CORE,
        receipt="EDGE_HEAT_KERNEL_RECEIPT",
        physical_claim=False,
    )
    central_record_report = central_record_born_report(
        record_signature=prev_signature,
        committed=committed,
        stable_count=stable_count,
        commit_cycles=commit_cycles,
    )
    checkpoint_report = observer_checkpoint_restoration_report(
        raw_observer_fields,
        observer_rows,
        max_observers=int((config.get("screen_microphysics_receipts", {}) or {}).get("checkpoint_observers", 64)),
    )
    h3_population_cfg = config.get("h3_population", {})
    h3_source_state = str(h3_population_cfg.get("source_state", "freezeout"))
    h3_raw_observer_fields, h3_source_report = _select_h3_source_fields(
        h3_source_state,
        final_raw_fields=raw_observer_fields,
        freezeout_raw_fields=freezeout_raw_observer_fields,
        repair_peak_raw_fields=repair_peak_raw_observer_fields,
        freezeout_state=freezeout_state,
        repair_peak_state=repair_peak_state,
        cycles=cycles,
    )
    if h3_population_cfg.get("enabled", True):
        h3_population_report = record_populated_h3_report(
            points,
            h3_caps,
            h3_raw_observer_fields,
            observer_rows,
            cell_entropy=cell_entropy,
            seed=seed + 1801,
            field_names=tuple(h3_population_cfg.get("field_names", [
                "record_signature",
                "stable_count",
                "repair_load",
                "cumulative_repair_load",
                "s3_class_density",
                "s3_sector_class",
            ])),
            candidate_count=int(h3_population_cfg.get("candidate_count", 2048)),
            candidate_radius=float(h3_population_cfg.get("candidate_radius", 2.0)),
            softness=float(h3_population_cfg.get("softness", 0.25)),
            pass_ratio=float(h3_population_cfg.get("pass_ratio", 0.85)),
            geometry_blend=float(h3_population_cfg.get("geometry_blend", 0.0)),
            response_mode=str(h3_population_cfg.get("response_mode", "field_summary_similarity")),
            transport_time=float(h3_population_cfg.get("transport_time", min(times) if times else 0.1)),
            transport_scale=float(h3_population_cfg.get("transport_scale", 2.0 * math.pi)),
        )
    else:
        h3_population_report = {
            "mode": "record_populated_h3_fit",
            "enabled": False,
            "record_populated_h3_receipt": False,
            "receipt": False,
            "claim_boundary": (
                "disabled for compact proxy output profile; current record-to-H3 observable is a "
                "weak demo path, not the modular response-kernel bulk receipt"
            ),
        }
    h3_population_report = with_claim_metadata(
        h3_population_report,
        claim_level=DEMO,
        receipt="RECORD_TO_H3_DEMO_RECEIPT",
        physical_claim=False,
    )
    h3_population_report["source_state"] = h3_source_state
    h3_population_report["source_report"] = h3_source_report
    conformal_chart_report["record_population_report"] = h3_population_report
    conformal_chart_report["record_populated_h3_receipt"] = bool(
        h3_population_report.get("record_populated_h3_receipt", False)
    )
    conformal_chart_report["h3_chart_report"]["record_population_receipt"] = bool(
        h3_population_report.get("record_populated_h3_receipt", False)
    )
    h3_modular_response_cfg = config.get("h3_modular_response", {}) or {}
    h3_modular_kernel_report: dict[str, Any] = {}
    h3_modular_fit_report: dict[str, Any] = {}
    if h3_modular_response_cfg.get("enabled", False):
        h3_modular_kernel = modular_response_kernel(
            points,
            h3_caps,
            h3_raw_observer_fields,
            observer_rows,
            times=[float(value) for value in h3_modular_response_cfg.get("times", times)],
            field_names=tuple(
                h3_modular_response_cfg.get(
                    "field_names",
                    [
                        "record_signature",
                        "stable_count",
                        "cumulative_repair_load",
                        "s3_class_density",
                        "s3_sector_class",
                    ],
                )
            ),
            cell_entropy=cell_entropy,
            transport_scale=float(h3_modular_response_cfg.get("transport_scale", 2.0 * math.pi)),
            k_transport=int(h3_modular_response_cfg.get("k_transport", 1)),
            observable_mode=str(h3_modular_response_cfg.get("observable_mode", "field_transport")),
            transition_observables=tuple(
                h3_modular_response_cfg.get(
                    "transition_observables",
                    [
                        "checkpoint_class",
                        "stable_flag",
                        "record_family",
                        "s3_sector_class",
                        "repair_load_bucket",
                    ],
                )
            ),
            transition_bins=int(h3_modular_response_cfg.get("transition_bins", 8)),
            record_family_modulus=int(h3_modular_response_cfg.get("record_family_modulus", 16)),
            transform=str(h3_modular_response_cfg.get("transform", "sigmoid")),
            wrong_scales=tuple(float(value) for value in h3_modular_response_cfg.get("wrong_scales", [1.0])),
            graph_state={
                "left": left,
                "right": right,
                "port_left": port_left,
                "port_right": port_right,
                "group_order": group_order,
                "patch_count": patch_count,
            },
            perturb_strength=float(h3_modular_response_cfg.get("perturb_strength", 1.0)),
            repair_steps=int(h3_modular_response_cfg.get("repair_steps", 4)),
            repairs_per_step=int(h3_modular_response_cfg.get("repairs_per_step", max(16, neighbors * 8))),
            perturb_seed=seed + 1889,
        )
        h3_modular_kernel_report = kernel_json_summary(h3_modular_kernel)
        h3_modular_fit_report = modular_response_h3_report(
            h3_modular_kernel,
            h3_caps,
            candidate_count=int(h3_modular_response_cfg.get("candidate_count", 2048)),
            candidate_radius=float(h3_modular_response_cfg.get("candidate_radius", 2.0)),
            softness=float(h3_modular_response_cfg.get("softness", 0.25)),
            seed=seed + 1841,
            pass_ratio=float(h3_modular_response_cfg.get("pass_ratio", 0.85)),
            min_observers=int(h3_modular_response_cfg.get("min_observers", 8)),
            min_features=int(h3_modular_response_cfg.get("min_features", 12)),
            fit_mode=str(h3_modular_response_cfg.get("fit_mode", "row_independent")),
            heldout_fraction=float(h3_modular_response_cfg.get("heldout_fraction", 0.25)),
            anchor_weight=float(h3_modular_response_cfg.get("anchor_weight", 0.05)),
            max_iterations=int(h3_modular_response_cfg.get("max_iterations", 4)),
        )
    object_rows: list[dict[str, Any]] = []
    object_report: dict[str, Any] = {}
    object_cfg = config.get("observer_objects", {})
    if object_cfg.get("enabled", False):
        record_families = extract_record_families(
            raw_observer_fields,
            (left, right),
            projections=object_cfg,
            persistence_horizon=int(object_cfg.get("persistence_horizon", commit_cycles)),
            max_families=int(object_cfg.get("max_families", 2048)),
        )
        object_packets = visible_object_packets(raw_observer_fields, object_cfg)
        _attach_object_packet_histograms(observer_rows, object_packets)
        assign_counterfactual_stability_from_records(
            record_families,
            raw_observer_fields,
            object_cfg,
            perturbations=int(object_cfg.get("counterfactual_perturbations", 16)),
            seed=seed + 1217,
        )
        object_rows = [family.as_jsonable() for family in record_families]
        object_report = observer_object_report(record_families, observer_rows)
    observer_chart_cfg = config.get("observer_chart_population", {}) or {}
    observer_chart_object_report: dict[str, Any] = {}
    if observer_chart_cfg.get("enabled", bool(object_rows and h3_modular_fit_report)):
        observer_chart_object_report = observer_chart_object_population_report(
            observer_rows,
            object_rows,
            h3_modular_fit_report,
            seed=seed + 1853,
            min_objects=int(observer_chart_cfg.get("min_objects", 8)),
            min_observers_per_object=int(observer_chart_cfg.get("min_observers_per_object", 2)),
            pass_ratio=float(observer_chart_cfg.get("pass_ratio", 0.85)),
            max_objects=int(observer_chart_cfg.get("max_objects", 2048)),
        )
    s3_holonomy_report = (
        array_holonomy_report(
            points,
            left,
            right,
            gauge,
            max_triangles=int(defects_cfg.get("max_triangles", 10_000)),
        )
        if group_name == "S3"
        else {}
    )
    s3_defect_timeline_report = (
        defect_timeline_report(
            points,
            left,
            right,
            defect_gauge_snapshots,
            max_triangles=int(timeline_cfg.get("max_triangles", max(1_000, min(int(defects_cfg.get("max_triangles", 10_000)), 5_000)))),
            persistence_cycles=int(timeline_cfg.get("persistence_cycles", 3)),
        )
        if defect_timeline_enabled and defect_gauge_snapshots
        else {}
    )
    defect_interaction = defect_interaction_report(
        s3_defect_timeline_report,
        min_observations=int(timeline_cfg.get("particle_min_observations", timeline_cfg.get("persistence_cycles", 3))),
        min_class_stability=float(timeline_cfg.get("particle_min_class_stability", 0.8)),
        min_transport_distance=float(timeline_cfg.get("interaction_min_transport_distance", 1e-9)),
        min_scattering_transitions=int(timeline_cfg.get("interaction_min_scattering_transitions", 2)),
    ) if s3_defect_timeline_report else {}
    h3_support_enabled = bool(h3_support_cfg.get("enabled", True))
    if h3_support_enabled:
        object_h3_report = support_profiles_to_h3_report(
            points,
            h3_caps,
            object_rows,
            cell_entropy=cell_entropy,
            seed=seed + 1811,
            support_key="support_nodes",
            id_key="object_id",
            label="record_family",
            candidate_count=int(h3_support_cfg.get("candidate_count", h3_population_cfg.get("candidate_count", 2048))),
            candidate_radius=float(h3_support_cfg.get("candidate_radius", h3_population_cfg.get("candidate_radius", 2.0))),
            softness=float(h3_support_cfg.get("softness", h3_population_cfg.get("softness", 0.25))),
            pass_ratio=float(h3_support_cfg.get("pass_ratio", h3_population_cfg.get("pass_ratio", 0.85))),
            min_support_count=int(h3_support_cfg.get("min_support_count", 8)),
            min_cap_count=int(h3_support_cfg.get("min_cap_count", 6)),
        )
        defect_h3_report = support_profiles_to_h3_report(
            points,
            h3_caps,
            list(s3_holonomy_report.get("clusters", [])) if s3_holonomy_report else [],
            cell_entropy=cell_entropy,
            seed=seed + 1821,
            support_key="support_nodes",
            id_key="cluster_id",
            label="s3_holonomy_defect_cluster",
            candidate_count=int(h3_support_cfg.get("candidate_count", h3_population_cfg.get("candidate_count", 2048))),
            candidate_radius=float(h3_support_cfg.get("candidate_radius", h3_population_cfg.get("candidate_radius", 2.0))),
            softness=float(h3_support_cfg.get("softness", h3_population_cfg.get("softness", 0.25))),
            pass_ratio=float(h3_support_cfg.get("pass_ratio", h3_population_cfg.get("pass_ratio", 0.85))),
            min_support_count=int(h3_support_cfg.get("min_support_count", 8)),
            min_cap_count=int(h3_support_cfg.get("min_cap_count", 6)),
        )
    else:
        object_h3_report = {
            "mode": "support_profile_h3_fit",
            "enabled": False,
            "record_populated_h3_receipt": False,
            "claim_boundary": "disabled for compact proxy output profile",
        }
        defect_h3_report = {
            "mode": "support_profile_h3_fit",
            "enabled": False,
            "record_populated_h3_receipt": False,
            "claim_boundary": "disabled for compact proxy output profile",
        }
    object_h3_report = with_claim_metadata(object_h3_report, claim_level=DEMO, receipt="SUPPORT_PROFILE_H3_DEMO_RECEIPT", physical_claim=False)
    defect_h3_report = with_claim_metadata(defect_h3_report, claim_level=DEMO, receipt="SUPPORT_PROFILE_H3_DEMO_RECEIPT", physical_claim=False)
    defect_h3_worldlines_report = (
        defect_timeline_to_h3_report(
            points,
            h3_caps,
            s3_defect_timeline_report,
            raw_fields=h3_raw_observer_fields,
            field_names=tuple(h3_support_cfg.get("timeline_field_names", h3_population_cfg.get("field_names", [
                "record_signature",
                "stable_count",
                "repair_load",
                "cumulative_repair_load",
                "local_mismatch_density",
                "modular_depth",
                "s3_class_density",
                "s3_sector_class",
            ]))),
            cell_entropy=cell_entropy,
            seed=seed + 1831,
            candidate_count=int(h3_support_cfg.get("timeline_candidate_count", h3_support_cfg.get("candidate_count", 2048))),
            candidate_radius=float(h3_support_cfg.get("candidate_radius", h3_population_cfg.get("candidate_radius", 2.0))),
            softness=float(h3_support_cfg.get("softness", h3_population_cfg.get("softness", 0.25))),
            pass_ratio=float(h3_support_cfg.get("pass_ratio", h3_population_cfg.get("pass_ratio", 0.85))),
            max_events=int(h3_support_cfg.get("timeline_max_events", 1024)),
            response_mode=str(h3_support_cfg.get("timeline_response_mode", "support_transport_similarity")),
            transport_time=float(h3_support_cfg.get("timeline_transport_time", h3_population_cfg.get("transport_time", min(times) if times else 0.1))),
            transport_scale=float(h3_support_cfg.get("timeline_transport_scale", h3_population_cfg.get("transport_scale", 2.0 * math.pi))),
        )
        if h3_support_enabled and s3_defect_timeline_report
        else {}
    )
    particle_report = particle_likeness_report(
        s3_defect_timeline_report,
        defect_interaction,
        bulk_localization_pass=bool(defect_h3_worldlines_report.get("bulk_worldline_precursor_receipt", False)),
        max_support_fraction=float(timeline_cfg.get("particle_max_support_fraction", 0.05)),
        min_observations=int(timeline_cfg.get("particle_min_observations", timeline_cfg.get("persistence_cycles", 3))),
        min_class_stability=float(timeline_cfg.get("particle_min_class_stability", 0.8)),
    ) if s3_defect_timeline_report else {}
    mandatory_controls = mandatory_control_report(
        requested_controls=[str(control) for control in config.get("controls", [])],
        points=points,
        left=left,
        right=right,
        initial_port_left=initial_port_left,
        initial_port_right=initial_port_right,
        final_port_left=port_left,
        final_port_right=port_right,
        object_rows=object_rows,
        seed=seed + 1601,
    )
    non_bw_controls = set(mandatory_controls.get("implemented_controls", []))
    bw_report["implemented_non_bw_controls"] = sorted(non_bw_controls)
    bw_report["unimplemented_controls"] = [
        control
        for control in config.get("controls", [])
        if control not in controls and control not in non_bw_controls
    ]
    neutral_report: dict[str, Any] = {}
    neutral_cfg = config.get("neutral_reconstruction", {})
    if neutral_cfg.get("enabled", False):
        object_dicts = object_rows
        neutral_report = bulk_reconstruction_report(observer_rows, object_dicts, state_bw_report, seed=seed + 1701)
    emergence_status = _emergence_status_report(bw_report, consensus_report)
    theorem_core_report = _theorem_core_receipts(trace, committed, config)
    emergence_status["state_derived_modular_transport"] = bool(state_bw_report)
    emergence_status["transition_scale_selection"] = bool(transition_selection_report)
    emergence_status["observer_object_construction"] = bool(object_report)
    emergence_status["mandatory_controls_pass"] = bool(mandatory_controls.get("all_expected_failures_observed"))
    emergence_status["neutral_reconstruction_written"] = bool(neutral_report)
    emergence_status["edge_sector_heat_kernel_receipt"] = bool(edge_sector_report.get("receipt", False))
    emergence_status["central_record_born_receipt"] = bool(central_record_report.get("receipt", False))
    emergence_status["observer_checkpoint_restoration_receipt"] = bool(checkpoint_report.get("receipt", False))
    if state_bw_report:
        emergence_status["state_derived_bw_median"] = state_bw_report.get("median")
        emergence_status["state_derived_correct_beats_controls"] = bool(
            state_bw_report.get("correct_beats_controls", False)
        )
        emergence_status["state_derived_best_control"] = state_bw_report.get("best_control")
        emergence_status["collar_markov_median_epsilon_cmi"] = collar_report.get("median_epsilon_cmi")
        if not state_bw_report.get("correct_beats_controls", False):
            emergence_status["status"] = "diagnostic_only_state_derived_controls_failed"
    if transition_selection_report:
        emergence_status["transition_primary_source"] = transition_selection_report.get("primary_source")
        emergence_status["transition_selected_label"] = transition_selection_report.get("selected_label")
        emergence_status["transition_two_pi_selected_by_primary"] = bool(
            transition_selection_report.get("two_pi_selected", False)
        )
        emergence_status["transition_two_pi_over_best"] = transition_selection_report.get("two_pi_over_best")
        emergence_status["transition_response_degenerate"] = bool(
            transition_selection_report.get("response_degenerate", False)
        )
        if transition_selection_report.get("response_degenerate", False):
            emergence_status["status"] = "diagnostic_only_transition_response_degenerate"
        elif not transition_selection_report.get("two_pi_selected", False):
            emergence_status["status"] = "diagnostic_only_transition_scale_not_selected"
    emergence_status["support_visible_lorentz_3p1_kinematics_receipt"] = bool(
        state_bw_report.get("correct_beats_controls", False)
        and transition_selection_report.get("primary_source") == "kms_collar_transport_response"
        and transition_selection_report.get("two_pi_selected", False)
        and not transition_selection_report.get("response_degenerate", False)
    )
    emergence_status["conformal_h3_spatial_chart_receipt"] = bool(
        conformal_chart_report.get("conformal_h3_spatial_chart_receipt", False)
    )
    emergence_status["record_populated_h3_spatial_receipt"] = bool(
        conformal_chart_report.get("record_populated_h3_receipt", False)
    )
    emergence_status["modular_response_h3_candidate_receipt"] = bool(
        h3_modular_fit_report.get("MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", False)
    )
    emergence_status["modular_response_h3_written"] = bool(h3_modular_fit_report)
    emergence_status["record_family_h3_support_receipt"] = bool(
        object_h3_report.get("record_populated_h3_receipt", False)
    )
    emergence_status["observer_chart_object_h3_receipt"] = bool(
        observer_chart_object_report.get("observer_chart_object_h3_receipt", False)
    )
    emergence_status["observer_chart_bulk_population_receipt"] = bool(
        observer_chart_object_report.get("observer_chart_bulk_population_receipt", False)
    )
    emergence_status["defect_cluster_h3_support_receipt"] = bool(
        defect_h3_report.get("record_populated_h3_receipt", False)
    )
    emergence_status["matter_defect_h3_support_receipt"] = bool(
        defect_h3_report.get("record_populated_h3_receipt", False)
    )
    emergence_status["defect_worldline_precursor_receipt"] = bool(
        s3_defect_timeline_report.get("persistent_worldline_precursor_receipt", False)
    )
    emergence_status["defect_h3_worldline_precursor_receipt"] = bool(
        defect_h3_worldlines_report.get("bulk_worldline_precursor_receipt", False)
    )
    emergence_status["defect_interaction_diagnostic_written"] = bool(defect_interaction)
    emergence_status["defect_transport_proxy_receipt"] = bool(
        defect_interaction.get("interaction_proxy_receipt", False)
    )
    emergence_status["particle_matter_receipt"] = bool(
        particle_report.get("particle_matter_receipt", False)
    )
    emergence_status["particle_likeness_diagnostic_written"] = bool(particle_report)
    emergence_status["spatial_bulk_3d_reconstruction_receipt"] = bool(
        conformal_chart_report.get("record_populated_h3_receipt", False)
        or observer_chart_object_report.get("observer_chart_bulk_population_receipt", False)
        or object_h3_report.get("record_populated_h3_receipt", False)
        or neutral_report.get("bulk_3d_established", False)
    )
    emergence_status["bulk_3d_established"] = bool(emergence_status["spatial_bulk_3d_reconstruction_receipt"])
    emergence_status["lorentz_vs_bulk_claim_boundary"] = (
        "support-visible BW/KMS cap flow plus cap-normal/H3 construction is the finite "
        "Lorentz/conformal chart diagnostic; spatial 3D bulk emergence additionally requires "
        "observer records or object families to populate that H3 chart under controls. "
        "Defect-cluster H3 support is a matter/particle precursor receipt, not by itself "
        "a full bulk reconstruction receipt."
    )
    freezeout_report: dict[str, Any] = {}
    cosmology_gate_report = _cosmology_gate_report(
        config.get("cosmology", {}),
        emergence_status,
        state_bw_report,
        transition_selection_report,
        neutral_report,
    )
    if config.get("cosmology", {}).get("freezeout", {}).get("enabled", False) and cosmology_gate_report["allowed"]:
        cosmology_runtime_config = dict(config.get("cosmology", {}))
        cosmology_runtime_config.setdefault("output_profile", output_profile)
        freezeout_report = write_freezeout_products(
            bundle.path,
            points=points,
            fields=freezeout_fields,
            cell_area_planck=cell_area_planck,
            cell_entropy=cell_entropy,
            freezeout_cycle=int(freezeout_state["cycle"]),
            committed_fraction=float(freezeout_state["committed_fraction"]),
            config=cosmology_runtime_config,
            seed=seed + 1501,
            gate_report=cosmology_gate_report,
        )
    elif config.get("cosmology", {}).get("freezeout", {}).get("enabled", False):
        cosmology_gate_report["freezeout_skipped"] = True

    _write_csv(bundle.path / "mismatch_trace.csv", trace)
    bundle.write_json("bw_report.json", bw_report)
    bundle.write_json("bw_controls.json", bw_report["controls"])
    if collar_report:
        bundle.write_json("collar_markov_report.json", collar_report)
    if state_bw_report:
        bundle.write_json("bw_state_derived_report.json", state_bw_report)
    if transition_selection_report:
        bundle.write_json("transition_scale_selection_report.json", transition_selection_report)
    bundle.write_json("cap_geometry_report.json", cap_report)
    bundle.write_json("conformal_h3_spatial_chart_report.json", conformal_chart_report)
    bundle.write_json("record_populated_h3_report.json", h3_population_report)
    if h3_modular_kernel_report:
        bundle.write_json("modular_response_kernel_report.json", h3_modular_kernel_report)
    if h3_modular_fit_report:
        bundle.write_json("modular_response_h3_report.json", h3_modular_fit_report)
    if observer_chart_object_report:
        bundle.write_json("observer_chart_object_h3_report.json", observer_chart_object_report)
    bundle.write_json("record_family_h3_report.json", object_h3_report)
    bundle.write_json("defect_cluster_h3_report.json", defect_h3_report)
    bundle.write_json("theorem_core_receipts.json", theorem_core_report)
    bundle.write_json("edge_sector_heat_kernel_report.json", edge_sector_report)
    bundle.write_json("central_record_born_report.json", central_record_report)
    bundle.write_json("observer_checkpoint_restoration_report.json", checkpoint_report)
    if defect_h3_worldlines_report:
        bundle.write_json("defect_h3_worldlines_report.json", defect_h3_worldlines_report)
    if write_jsonl_payloads:
        bundle.write_jsonl("observer_views.jsonl", observer_rows)
    bundle.write_json("observer_consensus_report.json", consensus_report)
    if object_rows and write_jsonl_payloads:
        bundle.write_jsonl("observer_objects.jsonl", object_rows)
    if object_report:
        bundle.write_json("object_consensus_report.json", object_report)
    bundle.write_json("mandatory_controls_report.json", mandatory_controls)
    if neutral_report:
        component_similarities, observer_ids = observer_similarity_components(observer_rows, object_rows, state_bw_report)
        similarity = component_similarities.get("composite", np.zeros((0, 0), dtype=float))
        distance = observer_distance_matrix(similarity)
        np.savez_compressed(
            bundle.path / "observer_distance_matrix.npz",
            observer_ids=np.asarray(observer_ids, dtype=np.int64),
            distance=distance,
            similarity=similarity,
        )
        for component_name, component_similarity in component_similarities.items():
            component_distance = observer_distance_matrix(component_similarity)
            np.savez_compressed(
                bundle.path / f"distance_{component_name}.npz",
                observer_ids=np.asarray(observer_ids, dtype=np.int64),
                distance=component_distance,
                similarity=component_similarity,
            )
        bundle.write_json("bulk_reconstruction_report.json", neutral_report)
    bundle.write_json("emergence_status_report.json", emergence_status)
    bundle.write_json("observable_summary.json", _field_summary(fields_all))
    if freezeout_report:
        bundle.write_json("cosmology_observables.json", {"freezeout_cl_proxy": freezeout_report})
    if config.get("cosmology", {}).get("freezeout", {}).get("enabled", False):
        bundle.write_json("cosmology_gate_report.json", cosmology_gate_report)
    bundle.write_json("s3_class_counts.json", s3_class_counts(gauge) if group_name == "S3" else {})
    if s3_holonomy_report:
        bundle.write_json("array_holonomy_report.json", s3_holonomy_report)
    if s3_defect_timeline_report:
        bundle.write_json("defect_timeline_report.json", s3_defect_timeline_report)
    if defect_interaction:
        bundle.write_json("defect_interaction_report.json", defect_interaction)
    if particle_report:
        bundle.write_json("particle_likeness_report.json", particle_report)
    bundle.write_json("seed_material.json", {"config_hash": stable_json_hash(config), "seed": seed})
    bundle.write_json("dimension_report.json", {"status": "not_computed_for_bw_primary_path", "reason": "BW residual is primary"})
    bundle.write_manifest(
        {
            "run_id": run_id,
            "name": config.get("name"),
            "engine": "bw_array",
            "claim_boundary": config.get("claim_boundary"),
            "run_mode": config.get("run_mode", config.get("mode")),
            "output_profile": output_profile,
            "patch_count": patch_count,
            "edge_count": edge_count,
            "group": group_name,
            "pixel_scale": pixel_scale.as_jsonable(),
            "oph_constants": pixel_scale.constants.as_jsonable(),
            "screen_microphysics": screen_microphysics.as_jsonable(),
            "screen_ports": screen_ports.as_jsonable(sample_edges=16),
            "boundary_program": boundary_program_report,
            "screen_units": screen_microphysics.as_jsonable()["screen_units"],
            "cycles": cycles,
            "final_phi": int(trace[-1]["phi"]),
            "bw_median": bw_report["median"],
            "bw_p90": bw_report["p90"],
            "bw_primary_mode": state_bw_report.get("mode", bw_report["mode"]) if state_bw_report else bw_report["mode"],
            "bw_primary_median": state_bw_report.get("median", bw_report["median"]) if state_bw_report else bw_report["median"],
            "collar_markov": {
                "median_epsilon_cmi": collar_report.get("median_epsilon_cmi"),
                "p90_epsilon_cmi": collar_report.get("p90_epsilon_cmi"),
            }
            if collar_report
            else {},
            "observer_consensus": {
                "observer_count": consensus_report.get("observer_count"),
                "global_committed_fraction": consensus_report.get("global_committed_fraction"),
                "median_signature_histogram_similarity": consensus_report.get("median_signature_histogram_similarity"),
            },
            "fixed_cutoff_microphysics_receipts": {
                "edge_sector_heat_kernel": edge_sector_report,
                "central_record_born": central_record_report,
                "observer_checkpoint_restoration": checkpoint_report,
            },
            "theorem_core_receipts": theorem_core_report,
            "observer_objects": object_report,
            "mandatory_controls": mandatory_controls,
            "transition_scale_selection": transition_selection_report,
            "conformal_h3_spatial_chart": conformal_chart_report,
            "record_populated_h3": h3_population_report,
            "modular_response_kernel": h3_modular_kernel_report,
            "modular_response_h3": h3_modular_fit_report,
            "observer_chart_object_h3": observer_chart_object_report,
            "record_family_h3": object_h3_report,
            "defect_cluster_h3": defect_h3_report,
            "defect_h3_worldlines": defect_h3_worldlines_report,
            "neutral_reconstruction": neutral_report,
            "emergence_status": emergence_status,
            "cosmology_observables": {"freezeout_cl_proxy": freezeout_report} if freezeout_report else {},
            "cosmology_gate": cosmology_gate_report if config.get("cosmology", {}).get("freezeout", {}).get("enabled", False) else {},
            "screen_holonomy": {
                "mode": s3_holonomy_report.get("mode"),
                "triangle_count": s3_holonomy_report.get("triangle_count"),
                "defect_triangle_count": s3_holonomy_report.get("defect_triangle_count"),
                "cluster_count": s3_holonomy_report.get("cluster_count"),
                "claim_boundary": s3_holonomy_report.get("claim_boundary"),
            }
            if s3_holonomy_report
            else {},
            "defect_timeline": {
                "mode": s3_defect_timeline_report.get("mode"),
                "snapshot_count": s3_defect_timeline_report.get("snapshot_count"),
                "worldline_count": s3_defect_timeline_report.get("worldline_count"),
                "persistent_worldline_count": s3_defect_timeline_report.get("persistent_worldline_count"),
                "particle_matter_receipt": s3_defect_timeline_report.get("particle_matter_receipt"),
                "claim_boundary": s3_defect_timeline_report.get("claim_boundary"),
            }
            if s3_defect_timeline_report
            else {},
            "defect_interaction": defect_interaction,
            "particle_likeness": particle_report,
            "defect_h3_worldlines": {
                "mode": defect_h3_worldlines_report.get("mode"),
                "event_count": defect_h3_worldlines_report.get("event_count"),
                "worldline_count": defect_h3_worldlines_report.get("worldline_count"),
                "persistent_h3_worldline_count": defect_h3_worldlines_report.get("persistent_h3_worldline_count"),
                "bulk_worldline_precursor_receipt": defect_h3_worldlines_report.get("bulk_worldline_precursor_receipt"),
                "particle_matter_receipt": defect_h3_worldlines_report.get("particle_matter_receipt"),
                "claim_boundary": defect_h3_worldlines_report.get("claim_boundary"),
            }
            if defect_h3_worldlines_report
            else {},
        }
    )
    return {
        "run_id": run_id,
        "path": str(bundle.path),
        "final_phi": int(trace[-1]["phi"]),
        "bw_median": bw_report["median"],
        "bw_p90": bw_report["p90"],
        "bw_primary_mode": state_bw_report.get("mode", bw_report["mode"]) if state_bw_report else bw_report["mode"],
        "bw_primary_median": state_bw_report.get("median", bw_report["median"]) if state_bw_report else bw_report["median"],
        "controls": bw_report["controls"],
        "theorem_core_receipts": {
            "lyapunov": theorem_core_report.get("lyapunov", {}).get("receipt"),
            "exact_repair_projection": theorem_core_report.get("exact_repair_projection", {}).get("receipt"),
            "sm_quotient_gate": theorem_core_report.get("sm_quotient_gate", {}).get("receipt"),
        },
        "transition_scale_selection": {
            "selected_label": transition_selection_report.get("selected_label"),
            "two_pi_selected": transition_selection_report.get("two_pi_selected"),
            "primary_source": transition_selection_report.get("primary_source"),
        }
        if transition_selection_report
        else {},
        "cosmology_gate": cosmology_gate_report if config.get("cosmology", {}).get("freezeout", {}).get("enabled", False) else {},
        "screen_holonomy": {
            "triangle_count": s3_holonomy_report.get("triangle_count"),
            "defect_triangle_count": s3_holonomy_report.get("defect_triangle_count"),
            "cluster_count": s3_holonomy_report.get("cluster_count"),
        }
        if s3_holonomy_report
        else {},
    }


def _observable_fields(
    *,
    port_left: np.ndarray,
    port_right: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    gauge: np.ndarray,
    patch_count: int,
    signature: np.ndarray,
    stable_count: np.ndarray,
    committed: np.ndarray,
    repair_load: np.ndarray,
    mismatch_density: np.ndarray,
    modular_depth: np.ndarray,
    cumulative_repair_load: np.ndarray,
) -> dict[str, np.ndarray]:
    fields = {
        "record_signature": _standardize(signature.astype(float)),
        "stable_count": _standardize(stable_count.astype(float)),
        "committed_mask": committed.astype(float),
        "repair_load": _standardize(repair_load.astype(float)),
        "cumulative_repair_load": _standardize(cumulative_repair_load.astype(float)),
        "local_mismatch_density": _standardize(mismatch_density.astype(float)),
        "modular_depth": _standardize(modular_depth.astype(float)),
    }
    if gauge.size:
        fields["s3_class_density"] = _standardize(s3_edge_class_density(left, right, gauge, patch_count))
        fields["s3_sector_class"] = _node_sector_class(left, right, gauge, patch_count)
    else:
        fields["s3_class_density"] = np.zeros(patch_count, dtype=float)
        fields["s3_sector_class"] = np.zeros(patch_count, dtype=np.int64)
    return fields


def _initialize_port_packets(
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    group_order: int,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mode = str(config.get("mode", "iid_hot")).lower().replace("-", "_")
    edge_count = int(left.size)
    if mode in {"support_visible_cap_net_hot", "kms_cap_net_hot", "cap_net_hot"}:
        node_labels, program_meta = _support_visible_cap_net_labels(
            points,
            group_order=group_order,
            cap_count=int(config.get("cap_count", 24)),
            sharpness=float(config.get("sharpness", 8.0)),
            tangent_weight=float(config.get("tangent_weight", 0.35)),
        )
        noise_probability = float(config.get("endpoint_noise_probability", 0.35))
        port_left = node_labels[left].astype(np.int16, copy=True)
        port_right = node_labels[right].astype(np.int16, copy=True)
        left_noise = rng.random(edge_count) < noise_probability
        right_noise = rng.random(edge_count) < noise_probability
        if np.any(left_noise):
            port_left[left_noise] = rng.integers(0, group_order, size=int(np.sum(left_noise)), dtype=np.int16)
        if np.any(right_noise):
            port_right[right_noise] = rng.integers(0, group_order, size=int(np.sum(right_noise)), dtype=np.int16)
        initial_mismatch_fraction = float(np.mean(port_left != port_right)) if edge_count else 0.0
        return port_left, port_right, {
            "mode": "support_visible_cap_net_hot",
            "cap_count": int(config.get("cap_count", 24)),
            "sharpness": float(config.get("sharpness", 8.0)),
            "tangent_weight": float(config.get("tangent_weight", 0.35)),
            "endpoint_noise_probability": noise_probability,
            "initial_mismatch_fraction": initial_mismatch_fraction,
            "node_label_histogram": _int_histogram(node_labels),
            **program_meta,
            "claim_boundary": (
                "boundary-driven support-visible S2 cap-net initial condition. It is a declared "
                "OPH screen boundary program for diagnostics, not spontaneous 3D bulk emergence."
            ),
        }
    port_left = rng.integers(0, group_order, size=edge_count, dtype=np.int16)
    port_right = rng.integers(0, group_order, size=edge_count, dtype=np.int16)
    return port_left, port_right, {
        "mode": "iid_hot",
        "initial_mismatch_fraction": float(np.mean(port_left != port_right)) if edge_count else 0.0,
        "claim_boundary": "iid hot endpoint packets; useful random control, not a structured OPH cap-net boundary program",
    }


def _support_visible_cap_net_labels(
    points: np.ndarray,
    *,
    group_order: int,
    cap_count: int,
    sharpness: float,
    tangent_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    axes = fibonacci_sphere_points(max(4, int(cap_count)))
    tangents = np.cross(axes, np.array([0.0, 0.0, 1.0]))
    bad = np.linalg.norm(tangents, axis=1) < 1e-9
    if np.any(bad):
        tangents[bad] = np.cross(axes[bad], np.array([0.0, 1.0, 0.0]))
    tangents = tangents / np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    cap_drive = np.tanh(float(sharpness) * (points @ axes.T))
    tangent_drive = np.sin(float(sharpness) * (points @ tangents.T))
    phases = np.mean(cap_drive + float(tangent_weight) * tangent_drive, axis=1)
    phase_min = float(np.min(phases)) if phases.size else 0.0
    phase_max = float(np.max(phases)) if phases.size else 1.0
    normalized = (phases - phase_min) / max(phase_max - phase_min, 1e-12)
    labels = np.floor(normalized * int(group_order)).astype(np.int16)
    labels = np.clip(labels, 0, int(group_order) - 1).astype(np.int16)
    return labels, {
        "phase_min": phase_min,
        "phase_max": phase_max,
        "phase_std": float(np.std(phases)) if phases.size else 0.0,
        "phase_source": "mean_round_cap_drive_plus_tangent_cut_pair_drive",
    }


def _int_histogram(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def _state_snapshot(
    *,
    cycle: int,
    committed_fraction: float,
    signature: np.ndarray,
    stable_count: np.ndarray,
    committed: np.ndarray,
    repair_load: np.ndarray,
    mismatch_density: np.ndarray,
    modular_depth: np.ndarray,
    cumulative_repair_load: np.ndarray,
) -> dict[str, Any]:
    return {
        "cycle": int(cycle),
        "committed_fraction": float(committed_fraction),
        "signature": np.asarray(signature).copy(),
        "stable_count": np.asarray(stable_count).copy(),
        "committed": np.asarray(committed).copy(),
        "repair_load": np.asarray(repair_load, dtype=float).copy(),
        "mismatch_density": np.asarray(mismatch_density, dtype=float).copy(),
        "modular_depth": np.asarray(modular_depth, dtype=float).copy(),
        "cumulative_repair_load": np.asarray(cumulative_repair_load, dtype=float).copy(),
    }


def _observable_fields_from_snapshot(
    snapshot: dict[str, Any],
    *,
    left: np.ndarray,
    right: np.ndarray,
    gauge: np.ndarray,
    patch_count: int,
) -> dict[str, np.ndarray]:
    return _observable_fields(
        port_left=np.zeros(0, dtype=np.int16),
        port_right=np.zeros(0, dtype=np.int16),
        left=left,
        right=right,
        gauge=gauge,
        patch_count=patch_count,
        signature=np.asarray(snapshot["signature"]),
        stable_count=np.asarray(snapshot["stable_count"]),
        committed=np.asarray(snapshot["committed"]),
        repair_load=np.asarray(snapshot["repair_load"], dtype=float),
        mismatch_density=np.asarray(snapshot["mismatch_density"], dtype=float),
        modular_depth=np.asarray(snapshot["modular_depth"], dtype=float),
        cumulative_repair_load=np.asarray(snapshot["cumulative_repair_load"], dtype=float),
    )


def _observer_raw_fields(
    *,
    left: np.ndarray,
    right: np.ndarray,
    gauge: np.ndarray,
    patch_count: int,
    signature: np.ndarray,
    stable_count: np.ndarray,
    committed: np.ndarray,
    repair_load: np.ndarray,
    mismatch_density: np.ndarray,
    modular_depth: np.ndarray,
    cumulative_repair_load: np.ndarray,
) -> dict[str, np.ndarray]:
    fields = {
        "record_signature": signature.astype(float),
        "stable_count": stable_count.astype(float),
        "committed_mask": committed.astype(float),
        "repair_load": repair_load.astype(float),
        "cumulative_repair_load": cumulative_repair_load.astype(float),
        "local_mismatch_density": mismatch_density.astype(float),
        "modular_depth": modular_depth.astype(float),
    }
    if gauge.size:
        fields["s3_class_density"] = s3_edge_class_density(left, right, gauge, patch_count)
        fields["s3_sector_class"] = _node_sector_class(left, right, gauge, patch_count)
    else:
        fields["s3_class_density"] = np.zeros(patch_count, dtype=float)
        fields["s3_sector_class"] = np.zeros(patch_count, dtype=np.int64)
    return fields


def _observer_raw_fields_from_snapshot(
    snapshot: dict[str, Any],
    *,
    left: np.ndarray,
    right: np.ndarray,
    gauge: np.ndarray,
    patch_count: int,
) -> dict[str, np.ndarray]:
    return _observer_raw_fields(
        left=left,
        right=right,
        gauge=gauge,
        patch_count=patch_count,
        signature=np.asarray(snapshot["signature"]),
        stable_count=np.asarray(snapshot["stable_count"]),
        committed=np.asarray(snapshot["committed"]),
        repair_load=np.asarray(snapshot["repair_load"], dtype=float),
        mismatch_density=np.asarray(snapshot["mismatch_density"], dtype=float),
        modular_depth=np.asarray(snapshot["modular_depth"], dtype=float),
        cumulative_repair_load=np.asarray(snapshot["cumulative_repair_load"], dtype=float),
    )


def _select_h3_source_fields(
    source_state: str,
    *,
    final_raw_fields: dict[str, np.ndarray],
    freezeout_raw_fields: dict[str, np.ndarray],
    repair_peak_raw_fields: dict[str, np.ndarray],
    freezeout_state: dict[str, Any],
    repair_peak_state: dict[str, Any],
    cycles: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = source_state.lower().replace("-", "_")
    if source in {"final", "settled", "final_state"}:
        return final_raw_fields, {
            "source_state": "final",
            "cycle": int(cycles - 1),
            "description": "fully settled final observer-visible records",
        }
    if source in {"repair_peak", "peak_repair", "mismatch_peak"}:
        return repair_peak_raw_fields, {
            "source_state": "repair_peak",
            "cycle": int(repair_peak_state.get("cycle", -1)),
            "committed_fraction": float(repair_peak_state.get("committed_fraction", 0.0)),
            "mean_mismatch_density": float(repair_peak_state.get("mean_mismatch_density", 0.0)),
            "std_mismatch_density": float(repair_peak_state.get("std_mismatch_density", 0.0)),
            "mean_cumulative_repair_load": float(repair_peak_state.get("mean_cumulative_repair_load", 0.0)),
            "std_cumulative_repair_load": float(repair_peak_state.get("std_cumulative_repair_load", 0.0)),
            "description": "observer-visible state at maximal local repair/mismatch activity",
        }
    return freezeout_raw_fields, {
        "source_state": "freezeout",
        "cycle": int(freezeout_state.get("cycle", -1)),
        "committed_fraction": float(freezeout_state.get("committed_fraction", 0.0)),
        "description": "first observer-record commit threshold snapshot",
    }


def _timeline_cycles(cycles: int, sample_count: int) -> set[int]:
    if cycles <= 0 or sample_count <= 0:
        return set()
    count = max(1, min(int(sample_count), int(cycles)))
    return {int(value) for value in np.linspace(0, int(cycles) - 1, count, dtype=int)}


def _repair_sector_labels(
    gauge: np.ndarray,
    chosen_edges: np.ndarray,
    chosen_delta: np.ndarray,
    *,
    group_order: int,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> None:
    """Declared finite S3 sector repair surrogate.

    The update is group-correct but intentionally reported only as a screen/collar
    defect-dynamics diagnostic. It composes chosen S3 edge labels with the
    pre-repair interface delta that the overlap repair just removed.
    """

    if group_order != 6 or not config.get("enabled", False) or chosen_edges.size == 0:
        return
    probability = float(config.get("probability", 0.0))
    if probability <= 0.0:
        return
    mask = rng.random(chosen_edges.size) < min(max(probability, 0.0), 1.0)
    if not np.any(mask):
        return
    edges = np.asarray(chosen_edges[mask], dtype=np.int64)
    proposals = np.asarray(chosen_delta[mask], dtype=np.int64) % 6
    nontrivial = proposals != 0
    if not np.any(nontrivial):
        return
    edges = edges[nontrivial]
    proposals = proposals[nontrivial]
    mode = str(config.get("mode", "repair_coupled_group_compose"))
    if mode == "cool_to_identity":
        gauge[edges] = 0
    else:
        gauge[edges] = S3_MUL[gauge[edges].astype(np.int64), proposals].astype(gauge.dtype)


def _node_sector_class(left: np.ndarray, right: np.ndarray, gauge: np.ndarray, patch_count: int) -> np.ndarray:
    from oph_fpe.defects.array_s3_holonomy import S3_CLASS

    classes = S3_CLASS[gauge.astype(np.int64)].astype(float)
    sums = np.bincount(left, weights=classes, minlength=patch_count) + np.bincount(
        right, weights=classes, minlength=patch_count
    )
    degree = np.bincount(np.concatenate([left, right]), minlength=patch_count)
    return np.rint(sums / np.maximum(degree, 1)).astype(np.int64)


def _projector_field_name(name: Any) -> str:
    text = str(name)
    for suffix in ("_projectors", "_projector", "-projectors", "-projector"):
        if text.endswith(suffix):
            return _projector_field_name(text[: -len(suffix)])
    if text == "s3_sector":
        return "s3_sector_class"
    return text


def _standardize(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    std = float(np.std(values))
    if std < 1e-12:
        return values - float(np.mean(values))
    return (values - float(np.mean(values))) / std


def _field_summary(fields: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
        }
        for name, values in fields.items()
    }


def _emergence_status_report(bw_report: dict[str, Any], consensus_report: dict[str, Any]) -> dict[str, Any]:
    control_medians = {
        name: float(report["median"])
        for name, report in bw_report.get("controls", {}).items()
        if isinstance(report, dict) and "median" in report
    }
    bw_median = float(bw_report.get("median", float("nan")))
    correct_beats_controls = bool(control_medians) and all(bw_median < value for value in control_medians.values())
    observer_records_settled = float(consensus_report.get("global_committed_fraction", 0.0)) >= 0.99
    observer_views_overlap = int(consensus_report.get("pair_count", 0)) > 0
    return {
        "status": "bw_observer_consensus_receipt" if correct_beats_controls and observer_records_settled else "diagnostic_only",
        "bulk_3d_established": False,
        "bw_cap_flow_single_run_receipt": bool(correct_beats_controls and observer_records_settled and observer_views_overlap),
        "lorentz_branch_single_run_receipt": False,
        "requires_refinement_scaling": True,
        "requires_neutral_bulk_reconstruction": True,
        "bw_median": bw_median,
        "control_medians": control_medians,
        "correct_2pi_beats_all_controls": correct_beats_controls,
        "observer_records_settled": observer_records_settled,
        "observer_views_overlap": observer_views_overlap,
        "claim_boundary": (
            "This run can support a finite BW/cap-flow and observer-consensus receipt. "
            "It does not by itself establish 3D bulk emergence; that requires refinement scaling "
            "and a neutral observer-record reconstruction path."
        ),
    }


def _attach_object_packet_histograms(observer_rows: list[dict[str, Any]], object_packets: np.ndarray) -> None:
    packets = np.asarray(object_packets, dtype=np.int64)
    for row in observer_rows:
        if row.get("view_type") != "patch_observer":
            continue
        support = np.asarray(row.get("support_nodes", []), dtype=np.int64)
        if support.size == 0 or packets.size == 0:
            row["object_packet_histogram"] = {}
            row["dominant_object_packet"] = None
            continue
        support = support[(support >= 0) & (support < packets.size)]
        if support.size == 0:
            row["object_packet_histogram"] = {}
            row["dominant_object_packet"] = None
            continue
        unique, counts = np.unique(packets[support], return_counts=True)
        total = float(counts.sum())
        histogram = {str(int(key)): float(count / total) for key, count in zip(unique, counts, strict=True)}
        row["object_packet_histogram"] = histogram
        row["dominant_object_packet"] = int(unique[int(np.argmax(counts))])


def _cosmology_gate_report(
    cosmology_cfg: dict[str, Any],
    emergence_status: dict[str, Any],
    state_bw_report: dict[str, Any],
    transition_selection_report: dict[str, Any],
    neutral_report: dict[str, Any],
) -> dict[str, Any]:
    freezeout_cfg = cosmology_cfg.get("freezeout", {})
    if not freezeout_cfg.get("enabled", False):
        return {"enabled": False, "allowed": False, "reason": "freezeout_disabled"}
    require_kms = bool(freezeout_cfg.get("require_kms_bw_pass", True))
    require_state = bool(freezeout_cfg.get("require_state_bw_controls", True))
    require_neutral = bool(freezeout_cfg.get("require_neutral_reconstruction", False))
    allow_screen_proxy = bool(freezeout_cfg.get("allow_screen_proxy_without_bulk", True))
    checks = {
        "kms_bw_pass": (
            bool(transition_selection_report)
            and transition_selection_report.get("primary_source") == "kms_collar_transport_response"
            and bool(transition_selection_report.get("two_pi_selected", False))
            and not bool(transition_selection_report.get("response_degenerate", False))
        ),
        "state_bw_controls_pass": bool(state_bw_report.get("correct_beats_controls", False)) if state_bw_report else False,
        "neutral_reconstruction_written": bool(neutral_report),
        "bulk_3d_established": bool(emergence_status.get("bulk_3d_established", False)),
        "screen_proxy_allowed_without_bulk": allow_screen_proxy,
    }
    required = {
        "kms_bw_pass": require_kms,
        "state_bw_controls_pass": require_state,
        "neutral_reconstruction_written": require_neutral,
    }
    missing = [name for name, is_required in required.items() if is_required and not checks[name]]
    if not allow_screen_proxy and not checks["bulk_3d_established"]:
        missing.append("bulk_3d_established")
    return {
        "enabled": True,
        "allowed": not missing,
        "missing_requirements": missing,
        "checks": checks,
        "required": required,
        "mode": "freezeout_screen_cl_proxy_gate",
        "claim_boundary": (
            "Allows only a measurement-facing screen C_l proxy. This gate does not certify a physical CMB "
            "prediction, a 3D bulk, a P(k), or a Boltzmann adapter."
        ),
    }


def _theorem_core_receipts(trace: list[dict[str, Any]], committed: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    theorem_cfg = config.get("theorem_core", {}) or {}
    lyapunov = lyapunov_descent_receipt(trace)
    sample_count = min(int(theorem_cfg.get("projection_sample", 256)), int(committed.shape[0]))
    if sample_count <= 0:
        projection = np.zeros((1, 1), dtype=float)
    else:
        projection = np.diag(np.asarray(committed[:sample_count], dtype=float))
    exact_repair = exact_repair_projection_receipt(
        projection,
        projection,
        tolerance=float(theorem_cfg.get("projection_tolerance", 1e-10)),
    )
    sm_candidate = theorem_cfg.get("sm_candidate", (config.get("gauge", {}) or {}).get("sm_candidate"))
    if sm_candidate is None and theorem_cfg.get("include_default_sm_candidate", False):
        sm_candidate = {
            "G_phys": "(SU(3)xSU(2)xU(1))/Z6",
            "hypercharge_lattice": "exact",
            "Nc": 3,
            "Ng": 3,
            "higgs_doublets": 1,
            "light_chiral_exotics": 0,
            "extra_low_scale_u1": 0,
            "xy_gauge_bosons": 0,
        }
    sm_gate = (
        standard_model_candidate_sieve(dict(sm_candidate))
        if sm_candidate is not None
        else {
            "mode": "finite_mar_standard_model_candidate_sieve",
            "enabled": False,
            "receipt": False,
            "SM_QUOTIENT_GATE_RECEIPT": False,
            "claim_level": "continuation",
            "claim_boundary": "no declared candidate supplied; SM quotient gate was not evaluated",
        }
    )
    report = {
        "mode": "theorem_core_receipt_bundle",
        "receipt": bool(lyapunov.get("receipt", False) and exact_repair.get("receipt", False)),
        "claim_level": RECOVERED_CORE,
        "receipt_name": "THEOREM_CORE_RECEIPT_BUNDLE",
        "physical_claim": False,
        "lyapunov": lyapunov,
        "exact_repair_projection": exact_repair,
        "sm_quotient_gate": sm_gate,
        "claim_boundary": (
            "fast finite theorem-instantiation bundle. The Lyapunov and projection receipts are "
            "fixed-cutoff recovered-core checks; the SM quotient gate is a declared continuation "
            "candidate sieve when supplied"
        ),
    }
    return report


def _regularize_support_visible_fields(
    fields: dict[str, np.ndarray],
    left: np.ndarray,
    right: np.ndarray,
    patch_count: int,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    steps = int(config.get("support_visible_smoothing_steps", 0))
    if steps <= 0:
        return fields
    alpha = float(config.get("support_visible_smoothing_alpha", 0.35))
    alpha = min(1.0, max(0.0, alpha))
    degree = np.bincount(np.concatenate([left, right]), minlength=patch_count).astype(float)
    degree = np.maximum(degree, 1.0)
    result: dict[str, np.ndarray] = {}
    for name, values in fields.items():
        smoothed = values.astype(float).copy()
        for _ in range(steps):
            neighbor_sum = (
                np.bincount(left, weights=smoothed[right], minlength=patch_count)
                + np.bincount(right, weights=smoothed[left], minlength=patch_count)
            )
            neighbor_mean = neighbor_sum / degree
            smoothed = (1.0 - alpha) * smoothed + alpha * neighbor_mean
        result[name] = _standardize(smoothed)
    return result


def _regularization_report(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "graph_neighbor_diffusion",
        "steps": int(config.get("support_visible_smoothing_steps", 0)),
        "alpha": float(config.get("support_visible_smoothing_alpha", 0.35)),
        "claim_boundary": (
            "finite support-visible extraction proxy; removes raw regulator-scale noise "
            "before BW cap-flow scoring"
        ),
    }


def _implemented_controls(controls: list[str]) -> list[str]:
    supported = {
        "wrong_1x_normalization",
        "wrong_pi_normalization",
        "wrong_4pi_normalization",
        "shuffled_caps",
        "randomized_cap_axes",
        "shuffled_observables",
        "no_modular_flow",
    }
    return [str(control) for control in controls if str(control) in supported]


def _h3_reconstruction_caps(
    points: np.ndarray,
    verifier_caps: list[Any],
    config: dict[str, Any],
    *,
    seed: int,
    patch_count: int,
    fallback_theta_values: list[float],
) -> tuple[list[Any], dict[str, Any]]:
    count = int(config.get("cap_count", len(verifier_caps)))
    theta_values = [float(value) for value in config.get("theta0", config.get("theta_values", fallback_theta_values))]
    use_dedicated = bool(config.get("dedicated_cap_net", count > len(verifier_caps) or bool(config.get("theta0"))))
    if not use_dedicated:
        return verifier_caps, {
            "mode": "reuse_bw_verifier_caps",
            "cap_count": len(verifier_caps),
            "claim_boundary": "H3 population diagnostics reuse the BW verifier cap set",
        }
    cap_cfg = {
        "collar_width_mode": config.get("collar_width_mode", "cell_scaled"),
        "collar_k": config.get("collar_k", 1.0),
        "collar_width": config.get("collar_width", 0.03),
    }
    caps = sample_caps(
        points,
        count=max(1, count),
        theta_values=theta_values,
        seed=int(seed),
        collar_width=_collar_width_from_config(cap_cfg, patch_count),
    )
    return caps, {
        "mode": "dedicated_h3_reconstruction_cap_net",
        "cap_count": len(caps),
        "theta0": theta_values,
        "collar_width": caps[0].collar_width if caps else None,
        "bw_verifier_cap_count": len(verifier_caps),
        "claim_boundary": (
            "denser support-visible cap net used for observer/H3 population and defect mapping; "
            "BW/KMS residuals remain scored on their declared verifier cap family"
        ),
    }


def _collar_width_from_config(config: dict[str, Any], patch_count: int) -> float:
    mode = str(config.get("collar_width_mode", "fixed"))
    if mode in {"cell_scaled", "k_neighborhood"}:
        cell_angular_scale = math.sqrt(4.0 * math.pi / max(1, int(patch_count)))
        return float(config.get("collar_k", 1.0)) * cell_angular_scale
    return float(config.get("collar_width", 0.03))


def _collar_report(config: dict[str, Any], patch_count: int, collar_width: float) -> dict[str, Any]:
    mode = str(config.get("collar_width_mode", "fixed"))
    cell_angular_scale = math.sqrt(4.0 * math.pi / max(1, int(patch_count)))
    return {
        "mode": mode,
        "collar_width": float(collar_width),
        "collar_k": float(config.get("collar_k", 1.0)),
        "cell_angular_scale": cell_angular_scale,
        "claim_boundary": "finite regulator collar; cell_scaled mode shrinks with refinement",
    }


def _run_id(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name).lower()).strip("_")
    return f"{slug}_{int(time.time())}"
