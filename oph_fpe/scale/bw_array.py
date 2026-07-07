from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.claims import (
    BRANCH_INSTANTIATION_SANITY,
    BW_KMS_DIRECT_2PI_RECEIPT,
    BW_KMS_BRANCH_INSTANTIATION_RECEIPT,
    BW_KMS_BRANCH_REPLAY_RECEIPT,
    CHART_LORENTZ_H3_RECEIPT,
    CONFORMAL_H3_CHART_RECEIPT,
    COSMOLOGY_PERTURBATION_RECEIPT,
    DEMO,
    DYNAMIC_DARK_TRANSPORT_RECEIPT,
    ENDOGENOUS_MODULAR_GENERATOR_RECEIPT,
    FINITE_CONSENSUS_THEOREM_RECEIPT,
    FINITE_SETTLE_DIAGNOSTIC_RECEIPT,
    H3_RESPONSE_CANDIDATE_RECEIPT,
    H3_RESPONSE_CONTROL_SEPARATION_RECEIPT,
    OBJECT_BULK_POPULATION_RECEIPT,
    OBJECT_CHART_RECEIPT,
    OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT,
    OPH_LORENTZ_THEOREM_FINITE_CONTRACT_RECEIPT,
    PAPER_THEOREM_3D_BULK_CHART_RECEIPT,
    PROXY,
    RECEIPT_SCHEMA_VERSION,
    RECORD_COMMIT_RECEIPT,
    RECOVERED_CORE,
    REPAIR_CORE_RECEIPT,
    SCREEN_PROXY_CMB_RECEIPT,
    STATIC_GALAXY_LAW_RECEIPT,
    STATIC_GALAXY_RAR_BTFR_RECEIPT,
    SUPPORT_VISIBLE_H3_POPULATED_BULK_RECEIPT,
    with_claim_metadata,
)
from oph_fpe.bulk.bw_verifier import bw_residual_report
from oph_fpe.bulk.cap_geometry import cap_geometry_report, sample_caps
from oph_fpe.bulk.conformal_spatial_chart import (
    conformal_h3_spatial_chart_report,
    paper_theorem_3d_bulk_chart_report,
)
from oph_fpe.bulk.h3_response_fit import modular_response_h3_report
from oph_fpe.bulk.h3_refit import write_modular_response_kernel_cache
from oph_fpe.bulk.markov_collar import collar_markov_report
from oph_fpe.bulk.modular_probe import state_derived_bw_report
from oph_fpe.bulk.modular_response_kernel import kernel_json_summary, modular_response_kernel
from oph_fpe.bulk.observer_reconstruction import (
    bulk_reconstruction_report,
    observer_distance_matrix,
    observer_similarity_components,
)
from oph_fpe.bulk.prime_geometric_response import attach_prime_geometric_response_to_rows
from oph_fpe.bulk.record_to_h3 import (
    defect_timeline_to_h3_report,
    observer_chart_object_population_report,
    record_populated_h3_report,
    support_profiles_to_h3_report,
)
from oph_fpe.bulk.transition_selection import transition_scale_selection_report
from oph_fpe.cache.geometry_cache import GeometryCache
from oph_fpe.constants.oph_pixel import equal_cell_area_planck, equal_cell_entropy
from oph_fpe.consensus.lyapunov import lyapunov_descent_receipt
from oph_fpe.cosmology.angular_power import angular_power_report
from oph_fpe.cosmology.galaxy_proxy import galaxy_proxy_receipt
from oph_fpe.cosmology import oph_cmb_stress_adapter_report, write_freezeout_products
from oph_fpe.cosmology.paired_ba_perturbation import write_paired_perturb_resettle_b_a_report
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
from oph_fpe.dynamics import (
    dispatch_configured_kernels,
    finite_consensus_theorem_certificate,
    kernel_dispatch_manifest_summary,
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
    transition_affinity_packet_fields,
    visible_object_packets,
)
from oph_fpe.scale.array_screen import (
    _beta_at,
    _entropy,
    _group_order,
    _knn_edges,
    _modular_cap_drive,
    _modular_update,
    _node_signature,
    _write_csv,
)
from oph_fpe.scale.parallel import jobs_from_config


def run_bw_array_config(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    seed = int(config.get("seed", 1))
    rng = np.random.default_rng(seed)
    outputs_cfg = config.get("outputs", {}) or {}
    output_profile = str(outputs_cfg.get("profile", "evidence"))
    observer_payload_needed = bool(
        (config.get("observer_objects", {}) or {}).get("enabled", False)
        or (config.get("observer_chart_population", {}) or {}).get("enabled", False)
        or (config.get("neutral_reconstruction", {}) or {}).get("enabled", False)
        or (config.get("viewer", {}) or {}).get("enabled", False)
    )
    default_write_jsonl = bool(output_profile in {"evidence", "debug", "viewer"} or observer_payload_needed)
    write_jsonl_payloads = bool(outputs_cfg["write_jsonl"]) if "write_jsonl" in outputs_cfg else default_write_jsonl
    if observer_payload_needed and not bool(outputs_cfg.get("allow_drop_observer_payloads", False)):
        write_jsonl_payloads = True
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
    repairs_per_cycle = _repairs_per_cycle_from_config(dyn, patch_count=patch_count, edge_count=edge_count)
    commit_cycles = int(dyn.get("record_commit_cycles", 8))
    beta_schedule = dyn.get("beta_schedule", {})
    readback_drive_cfg = dyn.get("observer_readback_drive", {}) or {}
    mod_cfg = config.get("modular_flow", {})
    modular_cap_drive = _modular_cap_drive(points, mod_cfg)
    readback_node_labels: np.ndarray | None = None
    readback_drive_report: dict[str, Any] = {"enabled": False}
    if readback_drive_cfg.get("enabled", False):
        readback_mode = str(readback_drive_cfg.get("mode", "support_visible_boundary_refresh"))
        if readback_mode in {"support_visible_boundary_refresh", "cap_net_boundary_refresh"}:
            readback_node_labels, readback_drive_report = _support_visible_cap_net_labels(
                points,
                group_order=group_order,
                cap_count=int(readback_drive_cfg.get("cap_count", config.get("boundary_program", {}).get("cap_count", 24))),
                sharpness=float(
                    readback_drive_cfg.get("sharpness", config.get("boundary_program", {}).get("sharpness", 8.0))
                ),
                tangent_weight=float(
                    readback_drive_cfg.get(
                        "tangent_weight",
                        config.get("boundary_program", {}).get("tangent_weight", 0.35),
                    )
                ),
            )
        readback_drive_report = {
            **readback_drive_report,
            "enabled": True,
            "mode": readback_mode,
            "edge_fraction": float(readback_drive_cfg.get("edge_fraction", 0.0)),
            "start_cycle": int(readback_drive_cfg.get("start_cycle", 0)),
            "stop_cycle": readback_drive_cfg.get("stop_cycle"),
            "claim_boundary": (
                "observer readback drive for exploration-phase self-reading dynamics. It perturbs "
                "bounded screen ports so local repair has records to read back; it is not part of "
                "the theorem-phase finite consensus certificate."
            ),
        }
        boundary_program_report["observer_readback_drive"] = readback_drive_report
    trace: list[dict[str, Any]] = []
    final_repair_load = np.zeros(patch_count, dtype=float)
    final_mismatch_density = np.zeros(patch_count, dtype=float)
    cumulative_repair_load = np.zeros(patch_count, dtype=float)
    freezeout_cfg = config.get("cosmology", {}).get("freezeout", {})
    freezeout_commit_fraction = float(freezeout_cfg.get("commit_fraction", 0.95))
    freezeout_state: dict[str, Any] | None = None
    repair_peak_state: dict[str, Any] | None = None
    first_commit_state: dict[str, Any] | None = None
    half_commit_state: dict[str, Any] | None = None
    repair_peak_score = -1.0
    defects_cfg = config.get("defects", {})
    sector_repair_cfg = defects_cfg.get("sector_repair", {})
    timeline_cfg = defects_cfg.get("timeline", {})
    defect_timeline_enabled = bool(group_name == "S3" and timeline_cfg.get("enabled", False))
    defect_timeline_cycles = _timeline_cycles(cycles, int(timeline_cfg.get("sample_count", 8)))
    defect_gauge_snapshots: list[tuple[int, np.ndarray]] = []
    bw_pre_cfg = config.get("bw", {}) or {}
    object_history_cfg = config.get("observer_objects", {}) or {}
    chart_history_cfg = config.get("observer_chart_population", {}) or {}
    bw_history_window = int(
        bw_pre_cfg.get(
            "history_window",
            8 if str(bw_pre_cfg.get("mode", "")) == "state_derived_modular_probe" else 1,
        )
    )
    history_window = max(
        int(object_history_cfg.get("history_window", 1)),
        int(chart_history_cfg.get("history_window", 1)),
        bw_history_window,
    )
    history_enabled = bool(
        history_window > 1
        or str(chart_history_cfg.get("incidence_mode", "transition_history")) == "transition_history"
        or str(object_history_cfg.get("family_mode", "")) == "transition_history"
    )
    recent_history_states: list[dict[str, Any]] = []
    freezeout_history_states: list[dict[str, Any]] = []
    harmonic_trace_cfg = (
        config.get("cosmology", {}).get("harmonic_time_trace", config.get("harmonic_time_trace", {})) or {}
    )
    harmonic_trace_enabled = bool(harmonic_trace_cfg.get("enabled", False))
    harmonic_trace_cycles = (
        _timeline_cycles(cycles, int(harmonic_trace_cfg.get("sample_count", min(cycles, 8))))
        if harmonic_trace_enabled
        else set()
    )
    harmonic_trace_samples: list[dict[str, Any]] = []
    progress_cfg = config.get("progress", {}) or {}
    base_progress_interval = max(0, int(progress_cfg.get("base_cycle_interval", dyn.get("progress_interval", 4))))
    base_loop_started = time.time()
    bundle.write_json(
        "base_progress.json",
        _base_repair_progress_report(
            stage="base_repair_loop_started",
            cycle=-1,
            cycles=cycles,
            started_at=base_loop_started,
            phi_before=None,
            phi_after=None,
            active_edges=None,
            chosen_edges=None,
            committed_fraction=0.0,
        ),
    )

    for cycle in range(cycles):
        beta = _beta_at(beta_schedule, cycle, cycles)
        readback_drive_edges = _apply_observer_readback_drive(
            port_left,
            port_right,
            left,
            right,
            group_order=group_order,
            rng=rng,
            cycle=cycle,
            config=readback_drive_cfg,
            node_labels=readback_node_labels,
        )
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
            cap_drive=modular_cap_drive,
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
            modular_time=modular_time,
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
        if first_commit_state is None and committed_fraction > 0.0:
            first_commit_state = repair_peak_candidate
        if half_commit_state is None and committed_fraction >= 0.5:
            half_commit_state = repair_peak_candidate
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
                modular_time=modular_time,
                cumulative_repair_load=cumulative_repair_load,
            )
            if history_enabled:
                freezeout_history_states = [*recent_history_states, repair_peak_candidate][-max(1, int(history_window)) :]
        if history_enabled:
            recent_history_states.append(repair_peak_candidate)
            recent_history_states = recent_history_states[-max(1, int(history_window)) :]
        trace.append(
            {
                "cycle": cycle,
                "phase": "exploration",
                "beta": beta,
                "phi_before": phi_before,
                "phi": phi_after,
                "delta_phi": phi_after - phi_before,
                "mismatch_edges": phi_after,
                "committed_records": int(np.sum(committed)),
                "committed_fraction": committed_fraction,
                "record_entropy": _entropy(signature[committed]) if np.any(committed) else 0.0,
                "modular_depth_mean": float(np.mean(modular_depth)),
                "modular_depth_std": float(np.std(modular_depth)),
                "observer_readback_drive_edges": int(readback_drive_edges),
            }
        )
        if harmonic_trace_enabled and cycle in harmonic_trace_cycles:
            harmonic_trace_samples.append(
                _harmonic_time_trace_sample(
                    points=points,
                    left=left,
                    right=right,
                    gauge=gauge,
                    patch_count=patch_count,
                    signature=signature,
                    stable_count=stable_count,
                    committed=committed,
                    repair_load=final_repair_load,
                    mismatch_density=final_mismatch_density,
                    modular_depth=modular_depth,
                    modular_time=modular_time,
                    cumulative_repair_load=cumulative_repair_load,
                    cell_entropy=cell_entropy,
                    cycle=cycle,
                    config=harmonic_trace_cfg,
                    seed=seed + 17_001 + cycle,
                )
            )
        if defect_timeline_enabled and cycle in defect_timeline_cycles:
            defect_gauge_snapshots.append((cycle, gauge.copy()))
        if _should_write_base_progress(cycle, cycles, base_progress_interval):
            bundle.write_json(
                "base_progress.json",
                _base_repair_progress_report(
                    stage="base_repair_loop",
                    cycle=cycle,
                    cycles=cycles,
                    started_at=base_loop_started,
                    phi_before=phi_before,
                    phi_after=phi_after,
                    active_edges=int(active.size),
                    chosen_edges=int(chosen.size),
                    committed_fraction=committed_fraction,
                    readback_drive_edges=int(readback_drive_edges),
                    record_entropy=trace[-1]["record_entropy"],
                    modular_depth_mean=trace[-1]["modular_depth_mean"],
                    modular_depth_std=trace[-1]["modular_depth_std"],
                ),
            )

    if defect_timeline_enabled and (not defect_gauge_snapshots or defect_gauge_snapshots[-1][0] != cycles - 1):
        defect_gauge_snapshots.append((cycles - 1, gauge.copy()))
    base_loop_elapsed_seconds = time.time() - base_loop_started
    bundle.write_json(
        "base_progress.json",
        _base_repair_progress_report(
            stage="base_repair_loop_complete",
            cycle=cycles - 1,
            cycles=cycles,
            started_at=base_loop_started,
            phi_before=trace[-1]["phi"] - trace[-1]["delta_phi"] if trace else None,
            phi_after=trace[-1]["phi"] if trace else None,
            active_edges=trace[-1]["phi"] if trace else None,
            chosen_edges=None,
            committed_fraction=float(np.mean(committed)),
            readback_drive_edges=trace[-1]["observer_readback_drive_edges"] if trace else None,
            record_entropy=trace[-1]["record_entropy"] if trace else None,
            modular_depth_mean=trace[-1]["modular_depth_mean"] if trace else None,
            modular_depth_std=trace[-1]["modular_depth_std"] if trace else None,
        ),
    )
    harmonic_time_trace_report = _write_harmonic_time_trace(
        bundle.path,
        harmonic_trace_samples,
        harmonic_trace_cfg,
        points=points,
        cell_entropy=cell_entropy,
    )

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
        modular_time=modular_time,
        cumulative_repair_load=cumulative_repair_load,
    )
    bw_cfg = config.get("bw", {})
    observables = [str(name) for name in bw_cfg.get("observables", ["record_signature", "repair_load", "s3_class_density", "stable_count"])]
    fields = {}
    skipped_scalar_observables: list[dict[str, str]] = []
    for name in observables:
        field_name = _projector_field_name(name)
        if field_name in fields_all:
            fields[field_name] = fields_all[field_name]
        elif name in fields_all:
            fields[name] = fields_all[name]
        else:
            skipped_scalar_observables.append({"observable": name, "reason": "not_a_scalar_screen_field"})
    fields = _regularize_support_visible_fields(fields, left, right, patch_count, bw_cfg)
    usable_fields = {}
    for name, values in fields.items():
        values = np.asarray(values, dtype=float)
        if values.size and np.all(np.isfinite(values)) and float(np.std(values)) > 1.0e-12:
            usable_fields[name] = values
        else:
            skipped_scalar_observables.append({"observable": name, "reason": "zero_or_nonfinite_scalar_variance"})
    fields = usable_fields
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
    if fields:
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
        bw_report["usable_scalar_observables"] = True
    else:
        bw_report = {
            "median": None,
            "mean": None,
            "p90": None,
            "by_observable": {},
            "by_cap_size": {},
            "rows": [],
            "controls": {},
            "usable_scalar_observables": False,
        }
    bw_report["mode"] = "kinematic_geometric_bw_sanity"
    bw_report["claim_boundary"] = (
        "kinematic geometry/interpolation sanity check for lambda_C(2*pi*t); "
        "not a state-derived modular-transport receipt"
    )
    bw_report = with_claim_metadata(
        bw_report,
        claim_level=DEMO,
        receipt="KINEMATIC_GEOMETRIC_BW_SANITY",
        physical_claim=False,
        observable_id="lambda_cap_support_visible_fields",
        fit_objective="weighted_geometric_pullback_residual",
    )
    bw_report["elapsed_seconds"] = time.time() - started
    bw_report["implemented_controls"] = controls
    bw_report["unimplemented_controls"] = [control for control in config.get("controls", []) if control not in controls]
    bw_report["scalar_observable_count"] = int(len(fields))
    bw_report["skipped_scalar_observables"] = skipped_scalar_observables
    bw_report["regulator_collar"] = _collar_report(bw_cfg, patch_count, caps[0].collar_width if caps else 0.0)
    bw_report["support_visible_regularization"] = _regularization_report(bw_cfg)
    cap_report = cap_geometry_report(
        points,
        caps,
        cell_area_planck=cell_area_planck,
        cell_entropy=cell_entropy,
    )
    cap_report["times"] = times
    cap_report["regulator_collar"] = bw_report["regulator_collar"]
    cap_report = with_claim_metadata(
        cap_report,
        claim_level=DEMO,
        receipt="CAP_GEOMETRY_DIAGNOSTIC",
        physical_claim=False,
        observable_id="round_caps_on_support_visible_s2",
        fit_objective="cap_geometry_consistency",
    )
    conformal_chart_report = conformal_h3_spatial_chart_report(h3_caps)
    conformal_chart_report["h3_reconstruction_cap_net"] = h3_cap_net_report
    conformal_chart_report["bw_verifier_cap_count"] = len(caps)
    conformal_chart_report = with_claim_metadata(
        conformal_chart_report,
        claim_level=BRANCH_INSTANTIATION_SANITY,
        receipt=CONFORMAL_H3_CHART_RECEIPT,
        physical_claim=False,
        observable_id="cap_normals_conformal_h3_chart",
        fit_objective="conformal_chart_instantiation",
    )
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
        modular_time=modular_time,
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
            modular_time=modular_time,
            cumulative_repair_load=cumulative_repair_load,
        )
    if repair_peak_state is None:
        repair_peak_state = freezeout_state
    if first_commit_state is None:
        first_commit_state = freezeout_state
    if half_commit_state is None:
        half_commit_state = freezeout_state
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
        state_source_name = str(bw_cfg.get("source_state", bw_cfg.get("state_source", "final")))
        state_raw_observer_fields, state_source_meta = _select_h3_source_fields(
            state_source_name,
            final_raw_fields=raw_observer_fields,
            freezeout_raw_fields=freezeout_raw_observer_fields,
            repair_peak_raw_fields=repair_peak_raw_observer_fields,
            freezeout_state=freezeout_state,
            repair_peak_state=repair_peak_state,
            first_commit_state=first_commit_state,
            half_commit_state=half_commit_state,
            cycles=cycles,
        )
        collar_cfg = config.get("collar_markov", {})
        collar_report = collar_markov_report(
            points,
            caps,
            state_raw_observer_fields,
            packet_bins=collar_cfg.get("packet_bins", {}),
            max_triplets=int(collar_cfg.get("max_triplets", 4096)),
            seed=seed + 1301,
        )
        state_history_states = _select_state_history_states(
            state_source_name,
            freezeout_history_states=freezeout_history_states,
            recent_history_states=recent_history_states,
            repair_peak_state=repair_peak_state,
            first_commit_state=first_commit_state,
            half_commit_state=half_commit_state,
            freezeout_state=freezeout_state,
            max_history=max(1, int(bw_cfg.get("history_window", history_window))),
        )
        state_history_states = _drop_source_snapshot_from_history(state_history_states, state_source_meta)
        state_history_raw_fields = [
            _observer_raw_fields_from_snapshot(snapshot, left=left, right=right, gauge=gauge, patch_count=patch_count)
            for snapshot in state_history_states
        ]
        state_source_meta["history_cycles"] = [
            int(snapshot.get("cycle", -1)) for snapshot in state_history_states
        ]
        state_source_meta["history_committed_fractions"] = [
            float(snapshot.get("committed_fraction", 0.0)) for snapshot in state_history_states
        ]
        graph_response = {
            "left": left,
            "right": right,
            "port_left": port_left,
            "port_right": port_right,
            "group_order": group_order,
            "patch_count": patch_count,
        }
        selection_cfg = bw_cfg.get("selection", {})
        state_bw_report = state_derived_bw_report(
            points,
            caps,
            state_raw_observer_fields,
            collar_report,
            times=times,
            observables=[_projector_field_name(name) for name in bw_cfg.get("observables", ["record_signature"])],
            regularizers=[float(value) for value in bw_cfg.get("regularizer_a", [0.001])],
            controls=controls,
            state_mode=str(bw_cfg.get("state_mode", "cooccurrence_kernel")),
            target_operator_mode=str(bw_cfg.get("target_operator_mode", "nearest")),
            transition_response_time=float(bw_cfg.get("transition_response_time", min(times) if times else 0.025)),
            transition_response_scale=float(bw_cfg.get("transition_response_scale", 2.0 * math.pi)),
            density_inverse_temperature=float(bw_cfg.get("density_inverse_temperature", 1.0)),
            generator_scale=float(bw_cfg.get("generator_scale", 1.0)),
            generator_scale_candidates=[
                float(value) for value in bw_cfg.get("generator_scale_candidates", [])
            ],
            history_fields=state_history_raw_fields,
            graph_response=graph_response,
            probe_steps=int(bw_cfg.get("probe_steps", selection_cfg.get("probe_steps", 4))),
            probe_repairs_per_source=int(
                bw_cfg.get("probe_repairs_per_source", selection_cfg.get("probe_repairs_per_source", max(16, neighbors * 4)))
            ),
            probe_max_incident_edges=int(
                bw_cfg.get("probe_max_incident_edges", selection_cfg.get("probe_max_incident_edges", max(4, neighbors)))
            ),
            max_basis=int(bw_cfg.get("max_basis", 96)),
            seed=seed + 1401,
        )
        state_bw_report["source_state"] = state_source_meta
        if selection_cfg.get("enabled", False):
            selection_sources = [str(source) for source in selection_cfg.get("sources", ["repair_affinity_response"])]
            if selection_cfg.get("include_declared_sanity", False) and "declared_geometric_sanity" not in selection_sources:
                selection_sources.append("declared_geometric_sanity")
            transition_selection_report = transition_scale_selection_report(
                points,
                caps,
                state_raw_observer_fields,
                times=[float(value) for value in selection_cfg.get("times", times)],
                observables=[_projector_field_name(name) for name in selection_cfg.get("observables", bw_cfg.get("observables", ["record_signature"]))],
                candidate_scales=[float(value) for value in selection_cfg.get("candidate_scales", [1.0, math.pi, 2.0 * math.pi, 4.0 * math.pi])],
                sources=selection_sources,
                declared_response_scale=float(selection_cfg.get("declared_response_scale", bw_cfg.get("transition_response_scale", 2.0 * math.pi))),
                max_basis=int(selection_cfg.get("max_basis", bw_cfg.get("max_basis", 64))),
                seed=seed + 1451,
                graph_response=graph_response,
                probe_steps=int(selection_cfg.get("probe_steps", 4)),
                probe_repairs_per_source=int(selection_cfg.get("probe_repairs_per_source", max(16, neighbors * 4))),
                probe_max_incident_edges=int(selection_cfg.get("probe_max_incident_edges", max(4, neighbors))),
                kms_response_scale=float(selection_cfg.get("kms_response_scale", 2.0 * math.pi)),
                kms_transport_steps=int(selection_cfg.get("kms_transport_steps", 8)),
            )
    observer_cfg = config.get("observers", {})
    observer_sample_count = int(observer_cfg.get("sample_count", min(64, patch_count)))
    observer_neighborhood = int(observer_cfg.get("neighborhood_size", max(8, min(64, neighbors * 4))))
    observer_times = _observer_relative_time_grid(observer_cfg, fallback_times=times)
    observer_rows = observer_view_rows(
        points,
        raw_fields=raw_observer_fields,
        visible_fields=fields,
        caps=caps,
        times=observer_times,
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
        sample_pair_limit=int(observer_cfg.get("sample_pair_limit", 20_000)),
    )
    consensus_report["observer_relative_time_grid"] = observer_times
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
    h3_modular_kernel: dict[str, Any] = {}
    h3_modular_kernel_report: dict[str, Any] = {}
    h3_modular_fit_report: dict[str, Any] = {}
    h3_modular_kernel_cache_report: dict[str, Any] = {}
    prime_geometric_response_report: dict[str, Any] = {}
    persistent_geometry_cache = _geometry_cache_from_config(points, config, outputs_cfg)
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
            transition_feature_types=tuple(
                h3_modular_response_cfg.get(
                    "feature_types",
                    h3_modular_response_cfg.get(
                        "transition_feature_types",
                        ["class_distribution_delta", "change_probability_delta"],
                    ),
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
            perturb_budget_mode=str(h3_modular_response_cfg.get("perturb_budget_mode", "modular_amount")),
            fixed_perturb_fraction=(
                float(h3_modular_response_cfg["fixed_perturb_fraction"])
                if h3_modular_response_cfg.get("fixed_perturb_fraction") is not None
                else None
            ),
            perturb_selection_mode=str(h3_modular_response_cfg.get("perturb_selection_mode", "phase")),
            transition_readout_mode=str(h3_modular_response_cfg.get("transition_readout_mode", "same_support")),
            repair_steps=int(h3_modular_response_cfg.get("repair_steps", 4)),
            repairs_per_step=int(h3_modular_response_cfg.get("repairs_per_step", max(16, neighbors * 8))),
            perturb_seed=seed + 1889,
            geometry_cache=persistent_geometry_cache,
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
            feature_selection=str(h3_modular_response_cfg.get("feature_selection", "none")),
            max_fit_features=(
                int(h3_modular_response_cfg["max_fit_features"])
                if h3_modular_response_cfg.get("max_fit_features") is not None
                else None
            ),
            min_feature_std=float(h3_modular_response_cfg.get("min_feature_std", 0.0)),
            min_wrong_scale_feature_delta=float(
                h3_modular_response_cfg.get("min_wrong_scale_feature_delta", 0.0)
            ),
            exclude_observables=tuple(
                str(value) for value in h3_modular_response_cfg.get("exclude_observables", [])
            ),
            exclude_feature_types=tuple(
                str(value) for value in h3_modular_response_cfg.get("exclude_feature_types", [])
            ),
            max_features_per_cap_time_observable=(
                int(h3_modular_response_cfg["max_features_per_cap_time_observable"])
                if h3_modular_response_cfg.get("max_features_per_cap_time_observable") is not None
                else None
            ),
            refine_steps=int(h3_modular_response_cfg.get("refine_steps", 0)),
            refine_max_rows=(
                int(h3_modular_response_cfg["refine_max_rows"])
                if h3_modular_response_cfg.get("refine_max_rows") is not None
                else None
            ),
            refine_max_nfev=int(h3_modular_response_cfg.get("refine_max_nfev", 48)),
            candidate_mode=str(h3_modular_response_cfg.get("candidate_mode", "random")),
            channel_mode=str(h3_modular_response_cfg.get("channel_mode", "time_observable_class")),
            profile_mode=str(h3_modular_response_cfg.get("profile_mode", "static_halfspace")),
            profile_time_scale=float(h3_modular_response_cfg.get("profile_time_scale", 2.0 * math.pi)),
            control_fit_mode=str(
                h3_modular_response_cfg.get("control_fit_mode", "same_h3_model_not_affine_target_fit")
            ),
        )
    observer_chart_cfg = config.get("observer_chart_population", {}) or {}
    if h3_modular_kernel:
        _attach_modular_response_histograms(observer_rows, h3_modular_kernel, observer_chart_cfg)
        prime_geometric_response_report = attach_prime_geometric_response_to_rows(
            observer_rows,
            h3_modular_kernel,
            spectrum_width=int(observer_chart_cfg.get("prime_geometric_spectrum_width", 64)),
            component_bins=int(observer_chart_cfg.get("prime_geometric_component_bins", 8)),
        )
    object_rows: list[dict[str, Any]] = []
    object_report: dict[str, Any] = {}
    object_cfg = dict(config.get("observer_objects", {}) or {})
    if object_cfg.get("enabled", False) and "family_mode" not in object_cfg:
        object_cfg["family_mode"] = "transition_affinity"
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
        _attach_transition_affinity_histograms(observer_rows, raw_observer_fields, object_cfg)
        if history_enabled:
            history_source_states = freezeout_history_states or recent_history_states
            history_raw_fields = [
                _observer_raw_fields_from_snapshot(snapshot, left=left, right=right, gauge=gauge, patch_count=patch_count)
                for snapshot in history_source_states[-max(1, int(history_window)) :]
            ]
            _attach_transition_history_histograms(observer_rows, history_raw_fields, object_cfg)
        assign_counterfactual_stability_from_records(
            record_families,
            raw_observer_fields,
            object_cfg,
            perturbations=int(object_cfg.get("counterfactual_perturbations", 16)),
            seed=seed + 1217,
        )
        object_rows = [family.as_jsonable() for family in record_families]
        object_report = observer_object_report(record_families, observer_rows)
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
            incidence_mode=str(observer_chart_cfg.get("incidence_mode", "transition_history")),
            min_packet_mass=float(observer_chart_cfg.get("min_packet_mass", 0.05)),
            min_transition_affinity=float(observer_chart_cfg.get("min_transition_affinity", 0.25)),
            transition_affinity_score=str(observer_chart_cfg.get("transition_affinity_score", "geometric_mean")),
            observer_cluster_fields=tuple(
                str(field)
                for field in observer_chart_cfg.get(
                    "observer_cluster_fields",
                    ["record_family", "s3_sector_class", "repair_load_bucket"],
                )
            ),
            observer_cluster_top_k=int(observer_chart_cfg.get("observer_cluster_top_k", 2)),
            min_observer_cluster_weight=float(observer_chart_cfg.get("min_observer_cluster_weight", 0.05)),
            history_window=int(observer_chart_cfg.get("history_window", history_window)),
            min_persistence=int(observer_chart_cfg.get("min_persistence", 3)),
            max_observer_fraction_per_object=float(observer_chart_cfg.get("max_observer_fraction_per_object", 0.65)),
            max_h3_compactness=float(observer_chart_cfg.get("max_h3_compactness", 0.35)),
            min_localized_objects=int(observer_chart_cfg.get("min_localized_objects", 2)),
            shuffle_control_count=int(observer_chart_cfg.get("shuffle_control_count", 1)),
            split_h3_components=bool(observer_chart_cfg.get("split_h3_components", False)),
            component_link_fraction=float(observer_chart_cfg.get("component_link_fraction", 0.35)),
            component_min_observers=(
                int(observer_chart_cfg["component_min_observers"])
                if "component_min_observers" in observer_chart_cfg
                else None
            ),
            require_support_visibility=bool(observer_chart_cfg.get("require_support_visibility", False)),
            min_support_visibility=float(observer_chart_cfg.get("min_support_visibility", 0.0)),
            visibility_mode=str(observer_chart_cfg.get("visibility_mode", "packet_or_support")),
            packet_visibility_weight=float(observer_chart_cfg.get("packet_visibility_weight", 0.5)),
            boundary_gate_mode=str(observer_chart_cfg.get("boundary_gate_mode", "nonboundary")),
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
            min_support_count=int(h3_support_cfg.get("defect_min_support_count", h3_support_cfg.get("min_support_count", 8))),
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
    paper_3d_chart_report = paper_theorem_3d_bulk_chart_report(
        conformal_chart_report,
        transition_selection_report,
        observer_chart_object_report,
        neutral_report,
        state_bw_report,
    )
    observer_modular_experience_report = _observer_modular_experience_report(
        observer_rows,
        raw_observer_fields,
        times=times,
        conformal_chart_report=conformal_chart_report,
        state_bw_report=state_bw_report,
        transition_selection_report=transition_selection_report,
        h3_modular_fit_report=h3_modular_fit_report,
        observer_chart_object_report=observer_chart_object_report,
        paper_3d_chart_report=paper_3d_chart_report,
    )
    emergence_status = _emergence_status_report(bw_report, consensus_report)
    theorem_core_report = _theorem_core_receipts(
        trace,
        committed,
        config,
        initial_port_left=initial_port_left,
        initial_port_right=initial_port_right,
        group_order=group_order,
        seed=seed,
    )
    emergence_status["final_phi_zero"] = bool(theorem_core_report.get("finite_settle_diagnostic_receipt", False))
    emergence_status[FINITE_SETTLE_DIAGNOSTIC_RECEIPT] = bool(
        theorem_core_report.get(FINITE_SETTLE_DIAGNOSTIC_RECEIPT, False)
    )
    emergence_status["finite_settle_diagnostic_receipt"] = bool(
        theorem_core_report.get("finite_settle_diagnostic_receipt", False)
    )
    emergence_status[FINITE_CONSENSUS_THEOREM_RECEIPT] = bool(
        theorem_core_report.get(FINITE_CONSENSUS_THEOREM_RECEIPT, False)
    )
    emergence_status["finite_consensus_theorem_receipt"] = bool(
        theorem_core_report.get("finite_consensus_theorem_receipt", False)
    )
    emergence_status["finite_consensus_missing_evidence"] = (
        (theorem_core_report.get("finite_consensus_theorem") or {}).get("missing_evidence", [])
    )
    emergence_status["state_derived_modular_transport"] = bool(state_bw_report)
    emergence_status["transition_scale_selection"] = bool(transition_selection_report)
    emergence_status["observer_object_construction"] = bool(object_report)
    emergence_status["observer_modular_time_experience_written"] = bool(observer_modular_experience_report)
    emergence_status["observer_modular_time_receipt"] = bool(
        observer_modular_experience_report.get("observer_modular_time_receipt", False)
    )
    emergence_status[OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT] = bool(
        observer_modular_experience_report.get(OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT, False)
    )
    emergence_status["observer_facing_3p1d_h3_experience_receipt"] = bool(
        observer_modular_experience_report.get("observer_facing_3p1d_h3_experience_receipt", False)
    )
    emergence_status["mandatory_controls_pass"] = bool(mandatory_controls.get("all_expected_failures_observed"))
    emergence_status["neutral_reconstruction_written"] = bool(neutral_report)
    emergence_status["edge_sector_heat_kernel_receipt"] = bool(edge_sector_report.get("receipt", False))
    emergence_status["central_record_born_receipt"] = bool(central_record_report.get("receipt", False))
    emergence_status["observer_checkpoint_restoration_receipt"] = bool(checkpoint_report.get("receipt", False))
    if state_bw_report:
        emergence_status["state_derived_bw_median"] = state_bw_report.get("median")
        emergence_status["state_derived_state_mode"] = state_bw_report.get("state_mode")
        emergence_status["state_derived_endogenous_modular_generator"] = bool(
            state_bw_report.get("endogenous_modular_generator", False)
        )
        emergence_status["state_derived_selected_scale_label"] = state_bw_report.get("state_selected_scale_label")
        emergence_status["state_derived_selected_2pi"] = bool(state_bw_report.get("state_selected_2pi", False))
        emergence_status["state_derived_correct_beats_controls"] = bool(
            state_bw_report.get("correct_beats_controls", False)
        )
        clock_fit = state_bw_report.get("inferred_modular_clock_fit") or {}
        emergence_status["state_derived_kms_clock_fit_receipt"] = bool(
            state_bw_report.get("KMS_GEOMETRIC_CLOCK_FIT_RECEIPT", False)
            or clock_fit.get("KMS_GEOMETRIC_CLOCK_FIT_RECEIPT", False)
            or clock_fit.get("receipt", False)
        )
        emergence_status["KMS_GEOMETRIC_CLOCK_FIT_RECEIPT"] = bool(
            emergence_status["state_derived_kms_clock_fit_receipt"]
        )
        emergence_status["kms_geometric_clock_fit_receipt"] = bool(
            emergence_status["state_derived_kms_clock_fit_receipt"]
        )
        emergence_status["state_derived_inferred_kappa_hat"] = clock_fit.get("kappa_hat")
        emergence_status["state_derived_inferred_kappa_95ci"] = clock_fit.get("kappa_95ci")
        emergence_status["state_derived_inferred_clock_blockers"] = clock_fit.get("blockers", [])
        emergence_status["state_derived_bw_bulk_gate"] = bool(
            state_bw_report.get(BW_KMS_BRANCH_REPLAY_RECEIPT, False)
            or state_bw_report.get(BW_KMS_BRANCH_INSTANTIATION_RECEIPT, False)
            or (
                state_bw_report.get("state_selected_2pi", False)
                and state_bw_report.get("correct_beats_controls", False)
                and (
                    state_bw_report.get("direct_transition_automorphism", False)
                    or state_bw_report.get("endogenous_modular_generator", False)
                )
            )
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
    emergence_status.update(
        _lorentz_branch_receipts(conformal_chart_report, state_bw_report, transition_selection_report)
    )
    emergence_status["lorentz_branch_single_run_receipt"] = bool(
        emergence_status["support_visible_lorentz_3p1_kinematics_receipt"]
    )
    emergence_status["conformal_h3_spatial_chart_receipt"] = bool(
        emergence_status["CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT"]
    )
    emergence_status[PAPER_THEOREM_3D_BULK_CHART_RECEIPT] = bool(
        paper_3d_chart_report.get(PAPER_THEOREM_3D_BULK_CHART_RECEIPT, False)
    )
    emergence_status["paper_theorem_3d_bulk_chart_receipt"] = bool(
        paper_3d_chart_report.get("paper_theorem_3d_bulk_chart_receipt", False)
    )
    emergence_status["paper_theorem_object_populated_chart_precursor_receipt"] = bool(
        paper_3d_chart_report.get("paper_theorem_object_populated_chart_precursor_receipt", False)
    )
    emergence_status["paper_theorem_neutral_populated_bulk_receipt"] = bool(
        paper_3d_chart_report.get("paper_theorem_neutral_populated_bulk_receipt", False)
    )
    emergence_status["paper_theorem_3d_chart_dimension"] = paper_3d_chart_report.get(
        "h3_spatial_dimension_from_boost_orbit"
    )
    emergence_status["paper_theorem_3d_bulk_chart_claim_boundary"] = paper_3d_chart_report.get("claim_boundary")
    emergence_status["record_populated_h3_spatial_receipt"] = bool(
        conformal_chart_report.get("record_populated_h3_receipt", False)
    )
    emergence_status["modular_response_h3_candidate_receipt"] = bool(
        h3_modular_fit_report.get("MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", False)
    )
    emergence_status["modular_response_h3_control_separation_receipt"] = bool(
        h3_modular_fit_report.get(H3_RESPONSE_CONTROL_SEPARATION_RECEIPT, False)
        or h3_modular_fit_report.get("h3_control_separation_receipt", False)
        or h3_modular_fit_report.get("h3_response_stage_gates", {}).get(
            H3_RESPONSE_CONTROL_SEPARATION_RECEIPT, False
        )
        or h3_modular_fit_report.get("h3_response_stage_gates", {}).get(
            "intermediate_control_separation_receipt", False
        )
    )
    emergence_status[H3_RESPONSE_CONTROL_SEPARATION_RECEIPT] = bool(
        emergence_status["modular_response_h3_control_separation_receipt"]
    )
    emergence_status[H3_RESPONSE_CANDIDATE_RECEIPT] = bool(
        h3_modular_fit_report.get(H3_RESPONSE_CANDIDATE_RECEIPT, False)
        or h3_modular_fit_report.get("MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", False)
    )
    emergence_status["modular_response_h3_written"] = bool(h3_modular_fit_report)
    emergence_status["record_family_h3_support_receipt"] = bool(
        object_h3_report.get("record_populated_h3_receipt", False)
    )
    emergence_status["record_family_h3_bulk_population_candidate"] = bool(
        object_h3_report.get("record_family_h3_bulk_population_candidate", False)
        and object_report.get("persistent_object_count", 0)
        and emergence_status.get("CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT", False)
    )
    emergence_status["observer_chart_object_h3_receipt"] = bool(
        observer_chart_object_report.get("observer_chart_object_h3_receipt", False)
    )
    emergence_status["observer_chart_localized_object_precursor_receipt"] = bool(
        observer_chart_object_report.get("localized_object_precursor_receipt", False)
    )
    emergence_status["observer_chart_modular_response_h3_control_separation_receipt"] = bool(
        observer_chart_object_report.get("modular_response_h3_control_separation_receipt", False)
    )
    theorem_assisted_h3_precursor = bool(
        emergence_status.get("CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT", False)
        and emergence_status.get("BW_AUTOMORPHISM_SANITY_RECEIPT", False)
        and emergence_status.get(H3_RESPONSE_CONTROL_SEPARATION_RECEIPT, False)
        and emergence_status["observer_chart_localized_object_precursor_receipt"]
    )
    emergence_status["PAPER_THEOREM_ASSISTED_H3_CHART_PRECURSOR_RECEIPT"] = theorem_assisted_h3_precursor
    emergence_status["paper_theorem_assisted_h3_chart_precursor_receipt"] = theorem_assisted_h3_precursor
    emergence_status["observer_chart_localized_nonboundary_bulk_population_receipt"] = bool(
        observer_chart_object_report.get("localized_nonboundary_bulk_population_receipt", False)
    )
    emergence_status["observer_chart_localized_h3_bulk_population_receipt"] = bool(
        observer_chart_object_report.get("localized_h3_bulk_population_receipt", False)
    )
    emergence_status["observer_chart_bulk_population_receipt"] = bool(
        observer_chart_object_report.get("observer_chart_bulk_population_receipt", False)
    )
    blind_bulk = neutral_report.get("blind_observer_bulk_report", {}) if neutral_report else {}
    blind_leakage_audit_pass = bool(
        not neutral_report
        or (
            blind_bulk.get("usable", False)
            and blind_bulk.get("s2_leakage_audit_pass", False)
            and not blind_bulk.get("forbidden_feature_keys_used", [])
        )
    )
    emergence_status["blind_observer_leakage_audit_pass"] = blind_leakage_audit_pass
    emergence_status["OBJECT_BULK_POPULATION_RECEIPT"] = bool(
        emergence_status[H3_RESPONSE_CANDIDATE_RECEIPT]
        and emergence_status["observer_chart_bulk_population_receipt"]
        and emergence_status.get("state_derived_bw_bulk_gate", False)
        and blind_leakage_audit_pass
    )
    emergence_status["object_bulk_population_receipt"] = bool(
        emergence_status["OBJECT_BULK_POPULATION_RECEIPT"]
    )
    theorem_assisted_h3_population = bool(
        emergence_status.get("CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT", False)
        and emergence_status.get("BW_AUTOMORPHISM_SANITY_RECEIPT", False)
        and emergence_status.get(H3_RESPONSE_CANDIDATE_RECEIPT, False)
        and emergence_status["observer_chart_bulk_population_receipt"]
        and blind_leakage_audit_pass
    )
    emergence_status["PAPER_THEOREM_ASSISTED_H3_POPULATED_CHART_RECEIPT"] = theorem_assisted_h3_population
    emergence_status["paper_theorem_assisted_h3_populated_chart_receipt"] = theorem_assisted_h3_population
    emergence_status["paper_theorem_assisted_h3_populated_chart_claim_boundary"] = (
        "Uses the paper-side conformal Lorentz/H3 chart receipt plus controlled observer-object "
        "population of that chart. This is a populated-chart diagnostic under the OPH BW/Lorentz "
        "branch assumptions; it is not the stricter finite endogenous modular-generator proof and "
        "therefore does not set bulk_3d_established."
    )
    emergence_status["paper_theorem_assisted_h3_chart_precursor_claim_boundary"] = (
        "Uses the paper-side conformal Lorentz/H3 chart receipt plus the intermediate support-visible "
        "H3 response control-separation receipt and localized observer-object precursor. This only says "
        "the chart-response lane is populated enough to keep testing; it is blocked from any 3D bulk "
        "claim until the material wrong-scale audit, nonboundary object-population gate, neutral "
        "dimension controls, and refinement gates pass."
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
    emergence_status["h3_support_population_precursor_receipt"] = bool(
        conformal_chart_report.get("record_populated_h3_receipt", False)
        or object_h3_report.get("record_populated_h3_receipt", False)
        or defect_h3_report.get("record_populated_h3_receipt", False)
        or defect_h3_worldlines_report.get("bulk_worldline_precursor_receipt", False)
    )
    support_visible_record_h3_population = bool(
        emergence_status.get(PAPER_THEOREM_3D_BULK_CHART_RECEIPT, False)
        and emergence_status.get("record_family_h3_bulk_population_candidate", False)
        and emergence_status.get("observer_records_settled", False)
    )
    support_visible_defect_h3_population = bool(
        emergence_status.get(PAPER_THEOREM_3D_BULK_CHART_RECEIPT, False)
        and emergence_status.get("defect_cluster_h3_support_receipt", False)
        and emergence_status.get("defect_worldline_precursor_receipt", False)
    )
    emergence_status["support_visible_record_h3_population_receipt"] = support_visible_record_h3_population
    emergence_status["support_visible_defect_h3_population_receipt"] = support_visible_defect_h3_population
    emergence_status["support_visible_h3_defect_population_receipt"] = support_visible_defect_h3_population
    emergence_status[SUPPORT_VISIBLE_H3_POPULATED_BULK_RECEIPT] = bool(support_visible_record_h3_population)
    emergence_status["support_visible_h3_populated_bulk_receipt"] = bool(
        emergence_status[SUPPORT_VISIBLE_H3_POPULATED_BULK_RECEIPT]
    )
    emergence_status["support_visible_h3_populated_bulk_claim_boundary"] = (
        "Paper-facing populated-H3 receipt: the conformal/BW S2 cap chart gives the H3 spatial chart, "
        "and observer-visible record-family support profiles populate that chart under S2-boundary and "
        "shuffled-cap controls. This is the support-visible OPH bulk-chart lane. Defect support in the "
        "same H3 chart is reported separately as a matter/particle precursor. This is not the stricter "
        "neutral third-person summary-distance reconstruction, not an endogenous full finite-type-I "
        "modular-generator proof, not a particle spectrum, and not a physical CMB prediction."
    )
    emergence_status["bulk_population_source"] = (
        "observer_chart_transition_history"
        if emergence_status["OBJECT_BULK_POPULATION_RECEIPT"]
        else "support_visible_record_and_defect_h3_profiles"
        if emergence_status["support_visible_h3_populated_bulk_receipt"]
        else "record_family_support_profile_h3_candidate_only"
        if emergence_status["record_family_h3_bulk_population_candidate"]
        else None
    )
    strict_neutral_bulk_receipt = bool(
        neutral_report.get("bulk_3d_established", False)
        and blind_bulk.get("usable", False)
        and blind_bulk.get("s2_leakage_audit_pass", False)
        and not blind_bulk.get("forbidden_feature_keys_used", [])
    )
    emergence_status["strict_blind_observer_bulk_receipt"] = strict_neutral_bulk_receipt
    emergence_status["spatial_bulk_3d_reconstruction_receipt"] = bool(
        emergence_status["OBJECT_BULK_POPULATION_RECEIPT"]
    )
    emergence_status["bulk_3d_established"] = bool(emergence_status["spatial_bulk_3d_reconstruction_receipt"])
    emergence_status["lorentz_claim_boundary"] = (
        "chart-level Lorentz receipt is conformal H3 chart plus direct BW/KMS automorphism sanity. "
        "Endogenous observer-record modular generators, object population, neutral bulk reconstruction, "
        "particles, and CMB outputs are later strengthening receipts and must not suppress the core "
        "support-visible Lorentz chart diagnostic."
    )
    emergence_status["lorentz_vs_bulk_claim_boundary"] = (
        "support-visible BW/KMS cap automorphism sanity plus cap-normal/H3 construction is the finite "
        "Lorentz/conformal chart diagnostic. Spatial 3D bulk emergence additionally requires observer "
        "records or object families to populate that H3 chart under controls. Defect-cluster H3 support "
        "is a matter/particle precursor receipt, not by itself a full bulk reconstruction receipt."
    )
    emergence_status["bulk_population_claim_boundary"] = (
        "OBJECT_BULK_POPULATION_RECEIPT is sourced by the observer-chart transition-history/object-mixture "
        "gate only after H3 response controls and the state-derived BW bulk gate pass. The active object "
        "gate is declared in observer_chart_object_population_report.bulk_population_gate_mode; boundary "
        "compactness is retained as a leakage audit when the paper-aligned H3-localized gate is selected. Persistent "
        "record-family support profiles are retained as candidate diagnostics, not as a bulk gate. "
        "Neutral reconstruction contributes a leakage audit and dimension debug report, but neutral "
        "summary-distance dimension is not the primary bulk gate. This is still not a physical cosmology "
        "or particle receipt."
    )
    freezeout_report: dict[str, Any] = {}
    cosmology_gate_report = _cosmology_gate_report(
        config.get("cosmology", {}),
        emergence_status,
        state_bw_report,
        transition_selection_report,
        neutral_report,
    )
    screen_proxy_cmb_receipt = bool(cosmology_gate_report.get("enabled", False) and cosmology_gate_report.get("allowed", False))
    emergence_status["SCREEN_PROXY_CMB_RECEIPT"] = screen_proxy_cmb_receipt
    emergence_status["screen_proxy_cmb_receipt"] = screen_proxy_cmb_receipt
    oph_cmb_report: dict[str, Any] = {}
    galaxy_cfg = dict(config.get("cosmology", {}).get("galaxy_proxy", config.get("galaxy_proxy", {})) or {})
    if galaxy_cfg.get("enabled", True):
        galaxy_proxy_report = galaxy_proxy_receipt(
            a0_oph=float(galaxy_cfg.get("a0_oph", 1.2e-10)),
            lambda_collar=float(galaxy_cfg.get("lambda_collar", 1.0)),
        )
    else:
        galaxy_proxy_report = {
            "mode": "oph_galaxy_rar_btfr_proxy",
            "enabled": False,
            "GALAXY_PROXY_RECEIPT": False,
            "receipt": False,
            "physical_claim": False,
            "claim_boundary": "disabled by config; no galaxy proxy was emitted",
        }
        galaxy_proxy_report = with_claim_metadata(
            galaxy_proxy_report,
            claim_level=PROXY,
            receipt=STATIC_GALAXY_LAW_RECEIPT,
            physical_claim=False,
            observable_id="oph_static_galaxy_proxy",
            fit_objective="disabled",
        )
    emergence_status[STATIC_GALAXY_LAW_RECEIPT] = bool(galaxy_proxy_report.get("GALAXY_PROXY_RECEIPT", False))
    emergence_status["static_galaxy_law_receipt"] = bool(galaxy_proxy_report.get("GALAXY_PROXY_RECEIPT", False))
    emergence_status[DYNAMIC_DARK_TRANSPORT_RECEIPT] = False
    emergence_status[COSMOLOGY_PERTURBATION_RECEIPT] = False
    emergence_status["dynamic_dark_transport_receipt"] = False
    emergence_status["cosmology_perturbation_receipt"] = False
    receipt_ladder_report = _canonical_receipt_ladder_report(
        trace=trace,
        committed=committed,
        emergence_status=emergence_status,
        state_bw_report=state_bw_report,
        transition_selection_report=transition_selection_report,
        h3_modular_fit_report=h3_modular_fit_report,
        observer_chart_object_report=observer_chart_object_report,
        cosmology_gate_report=cosmology_gate_report,
        galaxy_proxy_report=galaxy_proxy_report,
    )
    emergence_status["canonical_receipt_ladder"] = receipt_ladder_report["receipts"]
    emergence_status = with_claim_metadata(
        emergence_status,
        claim_level=BRANCH_INSTANTIATION_SANITY,
        receipt="SUPPORT_VISIBLE_LORENTZ_AND_BULK_STATUS_BUNDLE",
        physical_claim=False,
        observable_id="finite_screen_receipt_ladder",
        fit_objective="receipt_gate_conjunction",
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
    oph_cmb_cfg = dict(config.get("cosmology", {}).get("oph_cmb", {}) or {})
    if oph_cmb_cfg.get("enabled", bool(config.get("cosmology", {}).get("freezeout", {}).get("enabled", False))):
        oph_cmb_report = oph_cmb_stress_adapter_report(
            collar_report=collar_report,
            cosmology_gate_report=cosmology_gate_report,
            freezeout_report=freezeout_report,
            config=oph_cmb_cfg,
        )
    paired_ba_report: dict[str, Any] = {}
    paired_ba_cfg = dict(config.get("cosmology", {}).get("b_a_paired_perturbation", {}) or {})
    if paired_ba_cfg.get("enabled", False):
        paired_source_name = str(paired_ba_cfg.get("source_state", "final"))
        paired_raw_fields, paired_source_meta = _select_h3_source_fields(
            paired_source_name,
            final_raw_fields=raw_observer_fields,
            freezeout_raw_fields=freezeout_raw_observer_fields,
            repair_peak_raw_fields=repair_peak_raw_observer_fields,
            freezeout_state=freezeout_state,
            repair_peak_state=repair_peak_state,
            cycles=cycles,
        )
        paired_caps = h3_caps if str(paired_ba_cfg.get("cap_source", "bw")) == "h3" else caps
        paired_ba_report = write_paired_perturb_resettle_b_a_report(
            bundle.path,
            points,
            paired_caps,
            paired_raw_fields,
            {
                "left": left,
                "right": right,
                "port_left": port_left,
                "port_right": port_right,
                "group_order": group_order,
                "patch_count": patch_count,
                "degree": degree,
            },
            cell_entropy=cell_entropy,
            a_grid=[float(value) for value in paired_ba_cfg.get("a_grid", [1.0 / 1100.0, 0.01, 0.1, 1.0])],
            times=[float(value) for value in paired_ba_cfg.get("times", times)],
            max_caps=(
                int(paired_ba_cfg["max_caps"])
                if paired_ba_cfg.get("max_caps") is not None
                else None
            ),
            modes_per_cap_time=int(paired_ba_cfg.get("modes_per_cap_time", 2)),
            controls=tuple(str(value) for value in paired_ba_cfg.get("controls", []))
            or None,
            response_field=str(paired_ba_cfg.get("response_field", "cumulative_repair_load")),
            perturb_strength=float(paired_ba_cfg.get("perturb_strength", 1.0)),
            perturb_budget_mode=str(paired_ba_cfg.get("perturb_budget_mode", "modular_amount")),
            fixed_perturb_fraction=(
                float(paired_ba_cfg["fixed_perturb_fraction"])
                if paired_ba_cfg.get("fixed_perturb_fraction") is not None
                else None
            ),
            perturb_selection_mode=str(paired_ba_cfg.get("perturb_selection_mode", "lambda_collar_generator")),
            repair_steps=int(paired_ba_cfg.get("repair_steps", 4)),
            repairs_per_step=int(paired_ba_cfg.get("repairs_per_step", max(16, neighbors * 8))),
            transition_scale=float(paired_ba_cfg.get("transition_scale", 2.0 * math.pi)),
            seed=seed + 23_711,
        )
        paired_ba_report["source_state"] = paired_source_meta
        # Refresh JSON files after attaching source metadata; CSVs are already
        # written by the helper.
        bundle.write_json("paired_b_a_perturbation_report.json", paired_ba_report)
        bundle.write_json("b_a_parent_report.json", paired_ba_report)

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
    bundle.write_json("paper_3d_bulk_chart_report.json", paper_3d_chart_report)
    bundle.write_json("record_populated_h3_report.json", h3_population_report)
    if h3_modular_kernel_report:
        bundle.write_json("modular_response_kernel_report.json", h3_modular_kernel_report)
    if prime_geometric_response_report:
        bundle.write_json("prime_geometric_response_attachment_report.json", prime_geometric_response_report)
    if h3_modular_kernel and bool(outputs_cfg.get("write_modular_response_kernel_cache", True)):
        h3_modular_kernel_cache_report = write_modular_response_kernel_cache(bundle.path, h3_modular_kernel, h3_caps)
        bundle.write_json("modular_response_kernel_cache_report.json", h3_modular_kernel_cache_report)
    if h3_modular_fit_report:
        bundle.write_json("modular_response_h3_report.json", h3_modular_fit_report)
    if observer_chart_object_report:
        bundle.write_json("observer_chart_object_h3_report.json", observer_chart_object_report)
    bundle.write_json("observer_modular_experience_report.json", observer_modular_experience_report)
    bundle.write_json("record_family_h3_report.json", object_h3_report)
    bundle.write_json("defect_cluster_h3_report.json", defect_h3_report)
    bundle.write_json("theorem_core_receipts.json", theorem_core_report)
    if (theorem_core_report.get("finite_consensus_replay") or {}).get("enabled", False):
        bundle.write_json("finite_consensus_replay_report.json", theorem_core_report["finite_consensus_replay"])
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
    if oph_cmb_report:
        bundle.write_json("oph_cmb_stress_report.json", oph_cmb_report)
    if paired_ba_report:
        bundle.write_json("paired_b_a_perturbation_report.json", paired_ba_report)
        bundle.write_json("b_a_parent_report.json", paired_ba_report)
    if config.get("cosmology", {}).get("freezeout", {}).get("enabled", False):
        bundle.write_json("cosmology_gate_report.json", cosmology_gate_report)
    if galaxy_proxy_report:
        bundle.write_json("galaxy_proxy_report.json", galaxy_proxy_report)
    if harmonic_time_trace_report:
        bundle.write_json("harmonic_time_trace_report.json", harmonic_time_trace_report)
    bundle.write_json("receipt_ladder_report.json", receipt_ladder_report)
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
    kernel_dispatch = dispatch_configured_kernels(config, bundle.path, engine="bw_array")
    large_run_readiness = _large_run_readiness_report(
        config,
        state_bw_report=state_bw_report,
        transition_selection_report=transition_selection_report,
        cosmology_gate_report=cosmology_gate_report,
        observer_modular_experience_report=observer_modular_experience_report,
        paper_3d_chart_report=paper_3d_chart_report,
        theorem_core_report=theorem_core_report,
    )
    bundle.write_json("large_run_readiness_report.json", large_run_readiness)
    manifest = {
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
            "base_loop_elapsed_seconds": base_loop_elapsed_seconds,
            "bw_median": bw_report["median"],
            "bw_p90": bw_report["p90"],
            "bw_primary_mode": state_bw_report.get("mode", bw_report["mode"]) if state_bw_report else bw_report["mode"],
            "bw_primary_median": state_bw_report.get("median", bw_report["median"]) if state_bw_report else bw_report["median"],
            "geometric_bw_controls": bw_report["controls"],
            "state_bw_controls": state_bw_report.get("controls", {}) if state_bw_report else {},
            "state_bw_control_medians": state_bw_report.get("control_medians", {}) if state_bw_report else {},
            "state_bw_correct_beats_controls": state_bw_report.get("correct_beats_controls", False)
            if state_bw_report
            else None,
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
            "paper_3d_bulk_chart": paper_3d_chart_report,
            "record_populated_h3": h3_population_report,
            "modular_response_kernel": h3_modular_kernel_report,
            "modular_response_h3": h3_modular_fit_report,
            "prime_geometric_response": prime_geometric_response_report,
            "observer_chart_object_h3": observer_chart_object_report,
            "observer_modular_experience": observer_modular_experience_report,
            "record_family_h3": object_h3_report,
            "defect_cluster_h3": defect_h3_report,
            "defect_h3_worldlines": defect_h3_worldlines_report,
            "neutral_reconstruction": neutral_report,
            "emergence_status": emergence_status,
            "cosmology_observables": {"freezeout_cl_proxy": freezeout_report} if freezeout_report else {},
            "b_a_parent": {
                "mode": paired_ba_report.get("mode"),
                "primary_parent_source": paired_ba_report.get("primary_parent_source"),
                "row_count": len(paired_ba_report.get("rows") or []),
                "control_row_count": len(paired_ba_report.get("control_rows") or []),
                "real_baryon_perturbation_runs_present": (
                    (paired_ba_report.get("readiness", {}) or {}).get("checks", {}) or {}
                ).get("real_baryon_perturbation_runs_present"),
                "controls_fail": ((paired_ba_report.get("readiness", {}) or {}).get("checks", {}) or {}).get(
                    "controls_fail"
                ),
                "claim_boundary": paired_ba_report.get("claim_boundary"),
            }
            if paired_ba_report
            else {},
            "harmonic_time_trace": harmonic_time_trace_report,
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
            "large_run_readiness": large_run_readiness,
    }
    result = {
        "run_id": run_id,
        "path": str(bundle.path),
        "final_phi": int(trace[-1]["phi"]),
        "bw_median": bw_report["median"],
        "bw_p90": bw_report["p90"],
        "bw_primary_mode": state_bw_report.get("mode", bw_report["mode"]) if state_bw_report else bw_report["mode"],
        "bw_primary_median": state_bw_report.get("median", bw_report["median"]) if state_bw_report else bw_report["median"],
        "base_loop_elapsed_seconds": base_loop_elapsed_seconds,
        "controls": bw_report["controls"],
        "geometric_controls": bw_report["controls"],
        "state_bw_controls": state_bw_report.get("controls", {}) if state_bw_report else {},
        "state_bw_control_medians": state_bw_report.get("control_medians", {}) if state_bw_report else {},
        "state_bw_correct_beats_controls": state_bw_report.get("correct_beats_controls", False)
        if state_bw_report
        else None,
        "theorem_core_receipts": {
            "finite_settle_diagnostic": theorem_core_report.get("finite_settle_diagnostic_receipt"),
            "finite_consensus_theorem": theorem_core_report.get("finite_consensus_theorem_receipt"),
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
        "observer_modular_experience": {
            "observer_modular_time_receipt": observer_modular_experience_report.get("observer_modular_time_receipt"),
            "observer_facing_3p1d_h3_experience_receipt": observer_modular_experience_report.get(
                "observer_facing_3p1d_h3_experience_receipt"
            ),
            "observer_count": observer_modular_experience_report.get("observer_count"),
        },
        "cosmology_gate": cosmology_gate_report if config.get("cosmology", {}).get("freezeout", {}).get("enabled", False) else {},
        "screen_holonomy": {
            "triangle_count": s3_holonomy_report.get("triangle_count"),
            "defect_triangle_count": s3_holonomy_report.get("defect_triangle_count"),
            "cluster_count": s3_holonomy_report.get("cluster_count"),
        }
        if s3_holonomy_report
        else {},
        "large_run_readiness": large_run_readiness,
    }
    if kernel_dispatch:
        summary = kernel_dispatch_manifest_summary(kernel_dispatch)
        manifest["kernel_dispatch"] = summary
        result["kernel_dispatch"] = summary
    bundle.write_manifest(manifest)
    return result


def _should_write_base_progress(cycle: int, cycles: int, interval: int) -> bool:
    if cycles <= 0:
        return False
    if cycle == 0 or cycle == cycles - 1:
        return True
    return interval > 0 and (cycle + 1) % interval == 0


def _repairs_per_cycle_from_config(dyn: dict[str, Any], *, patch_count: int, edge_count: int) -> int:
    if dyn.get("repair_fraction_per_cycle") is not None:
        fraction = max(0.0, float(dyn.get("repair_fraction_per_cycle", 0.0)))
        if fraction <= 0.0:
            return 0
        return int(min(max(1, math.ceil(float(patch_count) * fraction)), max(0, int(edge_count))))
    return int(max(0, min(int(dyn.get("repairs_per_cycle", edge_count // 4)), max(0, int(edge_count)))))


def _base_repair_progress_report(
    *,
    stage: str,
    cycle: int,
    cycles: int,
    started_at: float,
    phi_before: int | None,
    phi_after: int | None,
    active_edges: int | None,
    chosen_edges: int | None,
    committed_fraction: float,
    readback_drive_edges: int | None = None,
    record_entropy: float | None = None,
    modular_depth_mean: float | None = None,
    modular_depth_std: float | None = None,
) -> dict[str, Any]:
    now = time.time()
    completed_cycles = max(0, min(cycles, cycle + 1))
    elapsed_seconds = max(0.0, now - started_at)
    estimated_total_seconds: float | None = None
    estimated_remaining_seconds: float | None = None
    if completed_cycles > 0 and cycles > 0:
        estimated_total_seconds = elapsed_seconds * float(cycles) / float(completed_cycles)
        estimated_remaining_seconds = max(0.0, estimated_total_seconds - elapsed_seconds)
    return {
        "stage": stage,
        "cycle": int(cycle),
        "cycles": int(cycles),
        "completed_cycles": int(completed_cycles),
        "elapsed_seconds": float(elapsed_seconds),
        "estimated_total_seconds": estimated_total_seconds,
        "estimated_remaining_seconds": estimated_remaining_seconds,
        "phi_before": phi_before,
        "phi_after": phi_after,
        "active_edges": active_edges,
        "chosen_edges": chosen_edges,
        "committed_fraction": float(committed_fraction),
        "readback_drive_edges": readback_drive_edges,
        "record_entropy": record_entropy,
        "modular_depth_mean": modular_depth_mean,
        "modular_depth_std": modular_depth_std,
    }


def _large_run_readiness_report(
    config: dict[str, Any],
    *,
    state_bw_report: dict[str, Any],
    transition_selection_report: dict[str, Any],
    cosmology_gate_report: dict[str, Any],
    observer_modular_experience_report: dict[str, Any],
    paper_3d_chart_report: dict[str, Any],
    theorem_core_report: dict[str, Any],
) -> dict[str, Any]:
    state_lane = _state_bw_readiness(state_bw_report)
    transition_lane = _transition_scale_readiness(transition_selection_report)
    cmb_lane = _screen_cmb_readiness(config, cosmology_gate_report)
    bulk_lane = _bulk_3d_readiness(paper_3d_chart_report)
    observer_lane = _observer_modular_time_readiness(observer_modular_experience_report)
    observer_facing_bulk_lane = _observer_facing_bulk_readiness(
        paper_3d_chart_report,
        observer_modular_experience_report,
    )
    finite_consensus_lane = _finite_consensus_readiness(theorem_core_report)
    lanes = {
        "state_bw": state_lane,
        "transition_scale": transition_lane,
        "screen_cmb_proxy": cmb_lane,
        "bulk_3d": bulk_lane,
        "observer_facing_bulk": observer_facing_bulk_lane,
        "observer_modular_time": observer_lane,
        "finite_consensus": finite_consensus_lane,
    }
    claim_lanes = {
        "state_bw": state_lane,
        "transition_scale": transition_lane,
        "screen_cmb_proxy": cmb_lane,
        "bulk_3d": bulk_lane,
        "observer_facing_bulk": observer_facing_bulk_lane,
    }
    stability_lanes = {
        "observer_modular_time": observer_lane,
        "finite_consensus": finite_consensus_lane,
    }
    if bulk_lane["scale_candidate"]:
        recommended = "bulk_3d_refinement"
    elif observer_facing_bulk_lane["scale_candidate"]:
        recommended = "observer_facing_bulk_visualization_refinement"
    elif cmb_lane["scale_candidate"]:
        recommended = "screen_cmb_proxy_refinement"
    elif transition_lane["scale_candidate"]:
        recommended = "transition_scale_refinement"
    elif state_lane["scale_candidate"]:
        recommended = "state_bw_refinement"
    else:
        recommended = "do_not_scale_yet"
    claim_scale_candidate = bool(any(lane["scale_candidate"] for lane in claim_lanes.values()))
    stability_only_candidate = bool(
        not claim_scale_candidate
        and any(lane["scale_candidate"] for lane in stability_lanes.values())
    )
    blockers = sorted(
        {
            str(blocker)
            for lane in lanes.values()
            for blocker in lane.get("blockers", [])
            if str(blocker)
        }
    )
    return {
        "mode": "large_run_preflight_readiness",
        "claim_boundary": (
            "Scale-readiness is a routing summary over existing finite receipts. It does not promote "
            "diagnostic rows into bulk, particle, or physical-CMB claims."
        ),
        "recommended_large_run_lane": recommended,
        "claim_scale_candidate": claim_scale_candidate,
        "stability_only_candidate": stability_only_candidate,
        "stability_only_lanes": [
            name for name, lane in stability_lanes.items() if lane["scale_candidate"]
        ],
        "any_scale_candidate": bool(any(lane["scale_candidate"] for lane in lanes.values())),
        "state_bw_expensive_run_worthwhile": bool(state_lane["scale_candidate"]),
        "lanes": lanes,
        "blockers": blockers,
    }


def _state_bw_readiness(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return _readiness_lane("not_requested", blockers=["state_bw_not_requested"])
    diagnostic_blockers: list[str] = []
    if not bool(report.get("correct_beats_controls", False)):
        diagnostic_blockers.append("state_bw_controls_failed")
    selected_label = report.get("state_selected_scale_label")
    if not bool(report.get("state_selected_2pi", False)):
        diagnostic_blockers.append(f"state_bw_selected_{selected_label or 'none'}_not_2pi")
    audit = report.get("generator_scale_audit") or {}
    if audit.get("enabled", False) and audit.get("best_label") not in {None, "2pi"}:
        diagnostic_blockers.append(f"generator_scale_best_{audit.get('best_label')}_not_2pi")
    clock = report.get("inferred_modular_clock_fit") or {}
    clock_applicable = bool(clock.get("enabled", False)) and not bool(clock.get("not_applicable", False))
    if clock_applicable and not bool(clock.get("receipt", False)):
        diagnostic_blockers.extend(f"clock_{blocker}" for blocker in clock.get("blockers", []))
    endogenous_generator_receipt = bool(
        report.get("ENDOGENOUS_MODULAR_GENERATOR_RECEIPT", False)
        or report.get("endogenous_modular_generator_receipt", False)
    )
    kms_clock_receipt = bool(
        report.get("KMS_GEOMETRIC_CLOCK_FIT_RECEIPT", False)
        or report.get("kms_geometric_clock_fit_receipt", False)
        or bool(clock.get("receipt", False))
    )
    finite_lorentz_clock_receipt = bool(endogenous_generator_receipt and kms_clock_receipt)
    blockers: list[str] = []
    if not finite_lorentz_clock_receipt:
        if not endogenous_generator_receipt:
            blockers.append("l2_endogenous_modular_generator_missing")
        if not kms_clock_receipt:
            blockers.append("l3_kms_modular_clock_fit_missing")
        blockers.extend(diagnostic_blockers)
    ready = bool(finite_lorentz_clock_receipt or not blockers)
    return _readiness_lane(
        "scale_candidate" if ready else "blocked",
        scale_candidate=ready,
        blockers=blockers,
        details={
            "median": report.get("median"),
            "selected_scale_label": selected_label,
            "correct_beats_controls": bool(report.get("correct_beats_controls", False)),
            "best_control": report.get("best_control"),
            "generator_scale_diagnosis": audit.get("diagnosis"),
            "inferred_kappa_hat": clock.get("kappa_hat"),
            "clock_receipt": bool(clock.get("receipt", False)),
            "endogenous_modular_generator_receipt": endogenous_generator_receipt,
            "kms_geometric_clock_fit_receipt": kms_clock_receipt,
            "finite_lorentz_modular_clock_receipt": finite_lorentz_clock_receipt,
            "legacy_scale_diagnostic_blockers": diagnostic_blockers,
        },
    )


def _transition_scale_readiness(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return _readiness_lane("not_requested", blockers=["transition_scale_selection_not_requested"])
    blockers: list[str] = []
    if not bool(report.get("two_pi_selected", False)):
        blockers.append(f"transition_selected_{report.get('selected_label') or 'none'}_not_2pi")
    if bool(report.get("response_degenerate", False)):
        blockers.append("transition_response_degenerate")
    ready = bool(not blockers)
    return _readiness_lane(
        "scale_candidate" if ready else "blocked",
        scale_candidate=ready,
        blockers=blockers,
        details={
            "selected_label": report.get("selected_label"),
            "primary_source": report.get("primary_source"),
            "two_pi_over_best": report.get("two_pi_over_best"),
            "normalization_source": report.get("normalization_source"),
        },
    )


def _screen_cmb_readiness(config: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    freezeout_enabled = bool((config.get("cosmology", {}) or {}).get("freezeout", {}).get("enabled", False))
    if not freezeout_enabled:
        return _readiness_lane("not_requested", blockers=["freezeout_screen_cmb_not_requested"])
    if not bool(report.get("enabled", False)):
        return _readiness_lane("blocked", blockers=[str(report.get("reason", "cosmology_gate_disabled"))])
    blockers = [str(value) for value in report.get("missing_requirements", [])]
    if not bool(report.get("allowed", False)) and not blockers:
        blockers.append("cosmology_gate_not_allowed")
    ready = bool(report.get("allowed", False) and not blockers)
    return _readiness_lane(
        "scale_candidate" if ready else "blocked",
        scale_candidate=ready,
        blockers=blockers,
        details={
            "checks": report.get("checks", {}),
            "required": report.get("required", {}),
            "allowed": bool(report.get("allowed", False)),
        },
    )


def _bulk_3d_readiness(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return _readiness_lane("not_requested", blockers=["paper_3d_bulk_chart_not_requested"])
    chart_receipt = bool(
        report.get(PAPER_THEOREM_3D_BULK_CHART_RECEIPT, False)
        or report.get("paper_theorem_3d_bulk_chart_receipt", False)
    )
    object_precursor = bool(report.get("paper_theorem_object_populated_chart_precursor_receipt", False))
    neutral_populated = bool(report.get("paper_theorem_neutral_populated_bulk_receipt", False))
    ready = bool(chart_receipt and object_precursor and neutral_populated)
    blockers: list[str] = []
    if not chart_receipt:
        blockers.append("paper_3d_bulk_chart_receipt_false")
    if not object_precursor:
        blockers.append("paper_theorem_object_populated_chart_precursor_receipt_false")
    if not neutral_populated:
        blockers.append("strict_neutral_bulk_gate_not_established")
    return _readiness_lane(
        "scale_candidate" if ready else "blocked",
        scale_candidate=ready,
        blockers=blockers,
        details={
            "paper_theorem_3d_bulk_chart_receipt": chart_receipt,
            "paper_theorem_object_populated_chart_precursor_receipt": object_precursor,
            "paper_theorem_neutral_populated_bulk_receipt": neutral_populated,
            "h3_spatial_dimension_from_boost_orbit": report.get("h3_spatial_dimension_from_boost_orbit"),
            "neutral_reconstruction_bulk_3d_established": bool(
                report.get("neutral_reconstruction_bulk_3d_established", False)
            ),
            "strict_neutral_note": (
                "Scaling the legacy neutral_summary_distance_diagnostic cannot make this pass; "
                "run strict-neutral object/frontier audits for a strict neutral-bulk claim."
            )
            if not neutral_populated
            else None,
        },
    )


def _observer_facing_bulk_readiness(
    paper_chart_report: dict[str, Any],
    observer_modular_experience_report: dict[str, Any],
) -> dict[str, Any]:
    if not paper_chart_report and not observer_modular_experience_report:
        return _readiness_lane("not_requested", blockers=["observer_facing_bulk_not_requested"])
    chart_receipt = bool(
        paper_chart_report.get(PAPER_THEOREM_3D_BULK_CHART_RECEIPT, False)
        or paper_chart_report.get("paper_theorem_3d_bulk_chart_receipt", False)
    )
    object_precursor = bool(
        paper_chart_report.get("paper_theorem_object_populated_chart_precursor_receipt", False)
    )
    observer_populated = bool(
        observer_modular_experience_report.get("observer_facing_populated_h3_experience_receipt", False)
    )
    ready = bool(chart_receipt and object_precursor and observer_populated)
    blockers: list[str] = []
    if not chart_receipt:
        blockers.append("paper_3d_bulk_chart_receipt_false")
    if not object_precursor:
        blockers.append("paper_theorem_object_populated_chart_precursor_receipt_false")
    if not observer_populated:
        blockers.extend(
            str(value)
            for value in observer_modular_experience_report.get("populated_h3_experience_blockers", [])
            if str(value)
        )
        if not blockers or blockers[-1] != "observer_facing_populated_h3_experience_receipt_false":
            blockers.append("observer_facing_populated_h3_experience_receipt_false")
    return _readiness_lane(
        "scale_candidate" if ready else "blocked",
        scale_candidate=ready,
        blockers=blockers,
        details={
            "paper_theorem_3d_bulk_chart_receipt": chart_receipt,
            "paper_theorem_object_populated_chart_precursor_receipt": object_precursor,
            "observer_facing_populated_h3_experience_receipt": observer_populated,
            "strict_neutral_not_required": True,
            "claim_boundary": (
                "observer-facing consensus H3 bulk visualization lane. It is not a chart-blind strict neutral "
                "third-person bulk, particle, or physical-CMB claim."
            ),
        },
    )


def _observer_modular_time_readiness(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return _readiness_lane("not_requested", blockers=["observer_modular_time_not_requested"])
    ready = bool(report.get("observer_modular_time_receipt", False))
    blockers = [] if ready else [str(value) for value in report.get("blockers", [])]
    if not ready and not blockers:
        blockers.append("observer_modular_time_receipt_false")
    return _readiness_lane(
        "scale_candidate" if ready else "blocked",
        scale_candidate=ready,
        blockers=blockers,
        details={
            "observer_modular_time_receipt": ready,
            "observer_facing_3p1d_h3_experience_receipt": bool(
                report.get("observer_facing_3p1d_h3_experience_receipt", False)
            ),
            "observer_count": report.get("observer_count"),
            "observer_facing_3p1d_blockers": [str(value) for value in report.get("blockers", [])],
        },
    )


def _finite_consensus_readiness(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return _readiness_lane("not_requested", blockers=["finite_consensus_not_requested"])
    ready = bool(
        report.get(FINITE_CONSENSUS_THEOREM_RECEIPT, False)
        or report.get("finite_consensus_theorem_receipt", False)
    )
    missing = (report.get("finite_consensus_theorem") or {}).get("missing_evidence", [])
    blockers = [str(value) for value in missing]
    if not ready and not blockers:
        blockers.append("finite_consensus_theorem_receipt_false")
    return _readiness_lane(
        "scale_candidate" if ready else "blocked",
        scale_candidate=ready,
        blockers=blockers,
        details={
            "finite_settle_diagnostic_receipt": bool(
                report.get(FINITE_SETTLE_DIAGNOSTIC_RECEIPT, False)
                or report.get("finite_settle_diagnostic_receipt", False)
            ),
            "finite_consensus_theorem_receipt": ready,
        },
    )


def _readiness_lane(
    status: str,
    *,
    scale_candidate: bool = False,
    blockers: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "scale_candidate": bool(scale_candidate),
        "blockers": blockers or [],
        "details": details or {},
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
    modular_time: np.ndarray,
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
        "modular_time": _standardize(modular_time.astype(float)),
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


def _apply_observer_readback_drive(
    port_left: np.ndarray,
    port_right: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    group_order: int,
    rng: np.random.Generator,
    cycle: int,
    config: dict[str, Any],
    node_labels: np.ndarray | None,
) -> int:
    if not config.get("enabled", False):
        return 0
    start_cycle = int(config.get("start_cycle", 0))
    stop_cycle_raw = config.get("stop_cycle")
    stop_cycle = int(stop_cycle_raw) if stop_cycle_raw is not None else None
    if int(cycle) < start_cycle or (stop_cycle is not None and int(cycle) >= stop_cycle):
        return 0
    edge_count = int(left.size)
    if edge_count == 0:
        return 0
    edge_fraction = max(0.0, min(float(config.get("edge_fraction", 0.0)), 1.0))
    requested = int(round(edge_fraction * edge_count))
    max_edges = int(config.get("max_edges_per_cycle", requested if requested > 0 else edge_count))
    drive_count = min(edge_count, max(0, requested), max_edges)
    if drive_count <= 0:
        return 0
    edges = rng.choice(edge_count, size=drive_count, replace=False)
    update_left = rng.random(drive_count) < 0.5
    mode = str(config.get("mode", "support_visible_boundary_refresh"))
    phase_advance = int(config.get("phase_advance_per_cycle", 0))
    phase = int((int(cycle) * phase_advance) % max(int(group_order), 1))
    if mode in {"support_visible_boundary_refresh", "cap_net_boundary_refresh"} and node_labels is not None:
        left_targets = (np.asarray(node_labels[left[edges]], dtype=np.int64) + phase) % int(group_order)
        right_targets = (np.asarray(node_labels[right[edges]], dtype=np.int64) + phase) % int(group_order)
    else:
        left_targets = rng.integers(0, int(group_order), size=drive_count, dtype=np.int64)
        right_targets = rng.integers(0, int(group_order), size=drive_count, dtype=np.int64)
    noise_probability = max(0.0, min(float(config.get("endpoint_noise_probability", 0.0)), 1.0))
    if noise_probability > 0.0:
        noisy_left = rng.random(drive_count) < noise_probability
        noisy_right = rng.random(drive_count) < noise_probability
        if np.any(noisy_left):
            left_targets[noisy_left] = rng.integers(0, int(group_order), size=int(np.sum(noisy_left)))
        if np.any(noisy_right):
            right_targets[noisy_right] = rng.integers(0, int(group_order), size=int(np.sum(noisy_right)))
    if np.any(update_left):
        port_left[edges[update_left]] = left_targets[update_left].astype(port_left.dtype)
    if np.any(~update_left):
        port_right[edges[~update_left]] = right_targets[~update_left].astype(port_right.dtype)
    return int(drive_count)


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
    modular_time: np.ndarray,
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
        "modular_time": np.asarray(modular_time, dtype=float).copy(),
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
        modular_time=np.asarray(snapshot.get("modular_time", snapshot["modular_depth"]), dtype=float),
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
    modular_time: np.ndarray,
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
        "modular_time": modular_time.astype(float),
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
        modular_time=np.asarray(snapshot.get("modular_time", snapshot["modular_depth"]), dtype=float),
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
    first_commit_state: dict[str, Any] | None = None,
    half_commit_state: dict[str, Any] | None = None,
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
    if source in {"theorem_observer", "observer_theorem", "theorem_record", "record_freezeout"}:
        return freezeout_raw_fields, {
            "source_state": "theorem_observer",
            "cycle": int(freezeout_state.get("cycle", -1)),
            "committed_fraction": float(freezeout_state.get("committed_fraction", 0.0)),
            "repair_peak_cycle": int(repair_peak_state.get("cycle", -1)),
            "first_commit_cycle": (
                int(first_commit_state.get("cycle", -1)) if first_commit_state is not None else None
            ),
            "half_commit_cycle": (
                int(half_commit_state.get("cycle", -1)) if half_commit_state is not None else None
            ),
            "description": (
                "theorem-facing observer state: committed freezeout records with repair/commit "
                "history supplied separately to the cap-state builder"
            ),
        }
    return freezeout_raw_fields, {
        "source_state": "freezeout",
        "cycle": int(freezeout_state.get("cycle", -1)),
        "committed_fraction": float(freezeout_state.get("committed_fraction", 0.0)),
        "description": "first observer-record commit threshold snapshot",
    }


def _select_state_history_states(
    source_state: str,
    *,
    freezeout_history_states: list[dict[str, Any]],
    recent_history_states: list[dict[str, Any]],
    repair_peak_state: dict[str, Any],
    first_commit_state: dict[str, Any],
    half_commit_state: dict[str, Any],
    freezeout_state: dict[str, Any],
    max_history: int,
) -> list[dict[str, Any]]:
    source = source_state.lower().replace("-", "_")
    if source in {"theorem_observer", "observer_theorem", "theorem_record", "record_freezeout"}:
        candidates = [
            repair_peak_state,
            *freezeout_history_states,
            first_commit_state,
            half_commit_state,
            freezeout_state,
        ]
        selected = _unique_snapshots_by_cycle(candidates)
        if len(selected) > max(1, int(max_history)):
            must_keep = _unique_snapshots_by_cycle(
                [repair_peak_state, first_commit_state, half_commit_state, freezeout_state]
            )
            remaining = [
                snapshot
                for snapshot in selected
                if int(snapshot.get("cycle", -1)) not in {int(item.get("cycle", -1)) for item in must_keep}
            ]
            budget = max(0, int(max_history) - len(must_keep))
            retained = remaining[-budget:] if budget else []
            selected = _unique_snapshots_by_cycle([*retained, *must_keep])
        return selected
    history = freezeout_history_states or recent_history_states
    return _unique_snapshots_by_cycle(history[-max(1, int(max_history)) :])


def _drop_source_snapshot_from_history(
    history_states: list[dict[str, Any]],
    source_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep history strictly before the current source snapshot.

    `state_derived_bw_report` receives `history_fields` and the current
    `raw_fields` separately, then appends the current fields when building
    history-dependent states. If the history list already contains the source
    cycle, the final Koopman/history transition is an artificial identity pair.
    """

    try:
        source_cycle = int(source_meta.get("cycle"))
    except (TypeError, ValueError):
        return list(history_states)
    return [
        snapshot
        for snapshot in history_states
        if int(snapshot.get("cycle", -1)) != source_cycle
    ]


def _unique_snapshots_by_cycle(snapshots: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    by_cycle: dict[int, dict[str, Any]] = {}
    for snapshot in snapshots:
        if not snapshot:
            continue
        cycle = int(snapshot.get("cycle", len(by_cycle)))
        by_cycle[cycle] = snapshot
    return [by_cycle[cycle] for cycle in sorted(by_cycle)]


def _timeline_cycles(cycles: int, sample_count: int) -> set[int]:
    if cycles <= 0 or sample_count <= 0:
        return set()
    count = max(1, min(int(sample_count), int(cycles)))
    return {int(value) for value in np.linspace(0, int(cycles) - 1, count, dtype=int)}


def _observer_relative_time_grid(observer_cfg: dict[str, Any], *, fallback_times: list[float]) -> list[float]:
    explicit = observer_cfg.get("relative_times")
    if isinstance(explicit, list) and explicit:
        values = [float(value) for value in explicit]
        return [max(0.0, min(1.0, value)) for value in values]
    sample_count = int(observer_cfg.get("time_sample_count", len(fallback_times) or 1))
    if sample_count <= 1:
        return [0.0]
    return [float(index / float(sample_count - 1)) for index in range(sample_count)]


def _harmonic_time_trace_sample(
    *,
    points: np.ndarray,
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
    modular_time: np.ndarray,
    cumulative_repair_load: np.ndarray,
    cell_entropy: np.ndarray,
    cycle: int,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    fields_all = _observable_fields(
        port_left=np.zeros(0, dtype=np.int16),
        port_right=np.zeros(0, dtype=np.int16),
        left=left,
        right=right,
        gauge=gauge,
        patch_count=patch_count,
        signature=signature,
        stable_count=stable_count,
        committed=committed,
        repair_load=repair_load,
        mismatch_density=mismatch_density,
        modular_depth=modular_depth,
        modular_time=modular_time,
        cumulative_repair_load=cumulative_repair_load,
    )
    field_names = [
        str(name)
        for name in config.get(
            "fields",
            ["record_signature", "stable_count", "cumulative_repair_load"],
        )
    ]
    selected = {name: fields_all[name] for name in field_names if name in fields_all}
    ell_max = int(config.get("ell_max", 32))
    if not selected:
        return {"cycle": int(cycle), "ell": np.arange(ell_max + 1, dtype=float), "fields": {}}
    fixed_time_controls = bool(config.get("fixed_time_controls", True))
    report = angular_power_report(
        points,
        selected,
        ell_max=ell_max,
        pair_samples=0,
        seed=int(seed),
        controls=[] if fixed_time_controls else [str(item) for item in config.get("controls", ["shuffled_field", "random_gaussian"])],
        estimator="spherical_harmonic",
        measure_weights=cell_entropy,
        harmonic_batch_size=int(config.get("harmonic_batch_size", 4096)),
        n_jobs=config.get("n_jobs", 1),
    )
    payload: dict[str, Any] = {
        "cycle": int(cycle),
        "ell": np.arange(int(report["ell_max"]) + 1, dtype=float),
        "fields": {
            name: np.asarray([float(row.get("D_ell", 0.0)) for row in field["spectrum"]], dtype=float)
            for name, field in report.get("fields", {}).items()
        },
        "controls": {
            name: {
                control_name: np.asarray(
                    [float(row.get("D_ell", 0.0)) for row in control_report["spectrum"]],
                    dtype=float,
                )
                for control_name, control_report in field_controls.items()
                if "spectrum" in control_report
            }
            for name, field_controls in report.get("controls", {}).items()
        },
    }
    if fixed_time_controls:
        payload["raw_fields"] = {
            name: np.asarray(values, dtype=np.float32)
            for name, values in selected.items()
        }
    return payload


def _write_harmonic_time_trace(
    run_path: Path,
    samples: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    points: np.ndarray | None = None,
    cell_entropy: np.ndarray | None = None,
) -> dict[str, Any]:
    usable = [sample for sample in samples if sample.get("fields")]
    if not usable:
        return {}
    ell = np.asarray(usable[0]["ell"], dtype=np.float32)
    cycles = np.asarray([int(sample["cycle"]) for sample in usable], dtype=np.int32)
    field_names = sorted(set().union(*(set(sample["fields"]) for sample in usable)))
    arrays: dict[str, np.ndarray] = {}
    for name in field_names:
        rows = []
        for sample in usable:
            values = np.asarray(sample["fields"].get(name, np.zeros_like(ell)), dtype=np.float32)
            if values.shape[0] != ell.shape[0]:
                values = np.resize(values, ell.shape[0]).astype(np.float32)
            rows.append(values)
        arrays[name] = np.vstack(rows).astype(np.float32)
    fixed_control_report = _fixed_time_harmonic_controls(
        usable,
        config,
        field_names=field_names,
        points=points,
        cell_entropy=cell_entropy,
        ell=ell,
    )
    control_keys: list[str] = []
    if fixed_control_report:
        arrays.update(fixed_control_report["arrays"])
        control_keys = list(fixed_control_report["control_keys"])
    else:
        control_names = sorted(
            {
                (field_name, control_name)
                for sample in usable
                for field_name, field_controls in (sample.get("controls", {}) or {}).items()
                for control_name in field_controls
            }
        )
        for field_name, control_name in control_names:
            rows = []
            for sample in usable:
                values = np.asarray(
                    ((sample.get("controls", {}) or {}).get(field_name, {}) or {}).get(
                        control_name,
                        np.zeros_like(ell),
                    ),
                    dtype=np.float32,
                )
                if values.shape[0] != ell.shape[0]:
                    values = np.resize(values, ell.shape[0]).astype(np.float32)
                rows.append(values)
            key = f"control__{field_name}__{control_name}"
            arrays[key] = np.vstack(rows).astype(np.float32)
            control_keys.append(key)
    np.savez_compressed(run_path / "harmonic_time_trace.npz", cycles=cycles, ell=ell, **arrays)
    return {
        "mode": "screen_harmonic_time_trace_v0",
        "enabled": True,
        "sample_count": int(cycles.size),
        "cycles": [int(value) for value in cycles],
        "field_names": field_names,
        "control_keys": control_keys,
        "fixed_time_controls": bool(fixed_control_report),
        "ell_max": int(ell[-1]) if ell.size else None,
        "n_jobs": config.get("n_jobs", 1),
        "harmonic_batch_size": int(config.get("harmonic_batch_size", 4096)),
        "claim_boundary": (
            "time-resolved observer-screen harmonic trace for synchronization-gap audits; "
            "not a CMB prediction or bulk reconstruction by itself"
        ),
    }


def _fixed_time_harmonic_controls(
    samples: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    field_names: list[str],
    points: np.ndarray | None,
    cell_entropy: np.ndarray | None,
    ell: np.ndarray,
) -> dict[str, Any]:
    if not bool(config.get("fixed_time_controls", True)):
        return {}
    if points is None or cell_entropy is None:
        return {}
    if not all(isinstance(sample.get("raw_fields"), dict) for sample in samples):
        return {}
    control_names = [str(item) for item in config.get("controls", ["shuffled_field", "random_gaussian"])]
    if not control_names:
        return {}
    node_count = int(points.shape[0])
    seed = int(config.get("control_seed", 1))
    transforms: dict[tuple[str, str], np.ndarray] = {}
    for field_name in field_names:
        for control_name in control_names:
            rng = _stable_control_rng(seed, field_name, control_name)
            if control_name == "shuffled_field":
                transforms[(field_name, control_name)] = rng.permutation(node_count)
            elif control_name == "random_gaussian":
                transforms[(field_name, control_name)] = rng.normal(size=node_count).astype(np.float32)

    if not transforms:
        return {}

    rows_by_key: dict[str, list[np.ndarray]] = {
        f"control__{field_name}__{control_name}": []
        for field_name, control_name in transforms
    }
    for sample in samples:
        raw_fields = sample.get("raw_fields", {}) or {}
        control_fields: dict[str, np.ndarray] = {}
        for (field_name, control_name), transform in transforms.items():
            raw = np.asarray(raw_fields.get(field_name, np.zeros(node_count, dtype=np.float32)), dtype=np.float32)
            if raw.shape[0] != node_count:
                raw = np.resize(raw, node_count).astype(np.float32)
            key = f"control__{field_name}__{control_name}"
            if control_name == "shuffled_field":
                control_fields[key] = raw[transform.astype(np.int64)]
            elif control_name == "random_gaussian":
                control_fields[key] = transform
        report = angular_power_report(
            points,
            control_fields,
            ell_max=int(ell[-1]) if ell.size else int(config.get("ell_max", 32)),
            pair_samples=0,
            seed=seed,
            controls=[],
            estimator="spherical_harmonic",
            measure_weights=cell_entropy,
            harmonic_batch_size=int(config.get("harmonic_batch_size", 4096)),
            n_jobs=config.get("n_jobs", 1),
        )
        for key, field in report.get("fields", {}).items():
            values = np.asarray([float(row.get("D_ell", 0.0)) for row in field.get("spectrum", [])], dtype=np.float32)
            if values.shape[0] != ell.shape[0]:
                values = np.resize(values, ell.shape[0]).astype(np.float32)
            rows_by_key[str(key)].append(values)

    arrays = {
        key: np.vstack(rows).astype(np.float32)
        for key, rows in rows_by_key.items()
        if rows
    }
    return {
        "arrays": arrays,
        "control_keys": sorted(arrays),
    }


def _stable_control_rng(seed: int, field_name: str, control_name: str) -> np.random.Generator:
    material = f"{int(seed)}:{field_name}:{control_name}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    value = int.from_bytes(digest[:8], "little") % (2**32)
    return np.random.default_rng(value)


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


def _observer_modular_experience_report(
    observer_rows: list[dict[str, Any]],
    raw_observer_fields: dict[str, np.ndarray],
    *,
    times: list[float],
    conformal_chart_report: dict[str, Any],
    state_bw_report: dict[str, Any],
    transition_selection_report: dict[str, Any],
    h3_modular_fit_report: dict[str, Any],
    observer_chart_object_report: dict[str, Any],
    paper_3d_chart_report: dict[str, Any],
) -> dict[str, Any]:
    patch_rows = [row for row in observer_rows if row.get("view_type") == "patch_observer"]
    modular_depth = np.asarray(raw_observer_fields.get("modular_depth", np.zeros(0)), dtype=float)
    local_means = np.asarray([float(row.get("modular_depth_mean", 0.0)) for row in patch_rows], dtype=float)
    observer_time_grid_available = bool(times and patch_rows)
    modular_depth_nontrivial = bool(
        modular_depth.size
        and np.all(np.isfinite(modular_depth))
        and float(np.std(modular_depth)) > 1.0e-12
        and local_means.size
        and np.all(np.isfinite(local_means))
    )
    observer_modular_time_receipt = bool(observer_time_grid_available and modular_depth_nontrivial)
    branch_replay_receipt = bool(
        state_bw_report.get(BW_KMS_BRANCH_REPLAY_RECEIPT, False)
        or state_bw_report.get(BW_KMS_BRANCH_INSTANTIATION_RECEIPT, False)
        or (
            transition_selection_report.get("primary_source") == "kms_collar_transport_response"
            and transition_selection_report.get("two_pi_selected", False)
            and not transition_selection_report.get("response_degenerate", False)
        )
    )
    finite_lorentz_modular_clock_receipt = bool(
        (
            state_bw_report.get("ENDOGENOUS_MODULAR_GENERATOR_RECEIPT", False)
            or state_bw_report.get("endogenous_modular_generator_receipt", False)
        )
        and (
            state_bw_report.get("KMS_GEOMETRIC_CLOCK_FIT_RECEIPT", False)
            or state_bw_report.get("kms_geometric_clock_fit_receipt", False)
            or (state_bw_report.get("inferred_modular_clock_fit") or {}).get("receipt", False)
        )
    )
    lorentz_clock_receipt = bool(branch_replay_receipt or finite_lorentz_modular_clock_receipt)
    chart_receipt = bool(
        conformal_chart_report.get("conformal_h3_spatial_chart_receipt", False)
        or paper_3d_chart_report.get(PAPER_THEOREM_3D_BULK_CHART_RECEIPT, False)
        or paper_3d_chart_report.get("paper_theorem_3d_bulk_chart_receipt", False)
    )
    h3_response_receipt = bool(
        h3_modular_fit_report.get(H3_RESPONSE_CANDIDATE_RECEIPT, False)
        or h3_modular_fit_report.get("MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", False)
        or h3_modular_fit_report.get("modular_response_h3_candidate_receipt", False)
        or h3_modular_fit_report.get(H3_RESPONSE_CONTROL_SEPARATION_RECEIPT, False)
        or h3_modular_fit_report.get("h3_control_separation_receipt", False)
    )
    object_population_receipt = bool(
        observer_chart_object_report.get(OBJECT_BULK_POPULATION_RECEIPT, False)
        or observer_chart_object_report.get("OBJECT_H3_NONBOUNDARY_POPULATION_RECEIPT", False)
        or observer_chart_object_report.get("observer_chart_bulk_population_receipt", False)
    )
    observer_3p1d_experience_receipt = bool(
        observer_modular_time_receipt
        and lorentz_clock_receipt
        and chart_receipt
        and h3_response_receipt
    )
    populated_h3_experience_receipt = bool(
        observer_3p1d_experience_receipt
        and object_population_receipt
    )
    component_gates = {
        "observer_modular_time_receipt": observer_modular_time_receipt,
        "bw_kms_branch_replay_receipt": lorentz_clock_receipt,
        "conformal_h3_chart_receipt": chart_receipt,
        "h3_modular_response_receipt": h3_response_receipt,
    }
    populated_h3_component_gates = {
        **component_gates,
        "observer_h3_object_population_receipt": object_population_receipt,
    }
    blockers = [name for name, passed in component_gates.items() if not passed]
    populated_h3_blockers = [
        name for name, passed in populated_h3_component_gates.items() if not passed
    ]
    report = {
        "mode": "observer_modular_3p1d_experience_report_v0",
        "observer_modular_time_receipt": observer_modular_time_receipt,
        OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT: observer_3p1d_experience_receipt,
        "observer_facing_3p1d_h3_experience_receipt": observer_3p1d_experience_receipt,
        "observer_facing_populated_h3_experience_receipt": populated_h3_experience_receipt,
        "observer_h3_object_population_receipt": object_population_receipt,
        "declared_bw_kms_branch_replay_receipt": branch_replay_receipt,
        "finite_lorentz_modular_clock_receipt": finite_lorentz_modular_clock_receipt,
        "observer_count": len(patch_rows),
        "observer_relative_time_count": len(times),
        "observer_relative_time_grid": [float(value) for value in times],
        "modular_depth_mean": float(np.mean(modular_depth)) if modular_depth.size else None,
        "modular_depth_std": float(np.std(modular_depth)) if modular_depth.size else None,
        "observer_modular_depth_mean_median": float(np.median(local_means)) if local_means.size else None,
        "observer_modular_depth_mean_std": float(np.std(local_means)) if local_means.size else None,
        "component_gates": component_gates,
        "populated_h3_component_gates": populated_h3_component_gates,
        "blockers": blockers,
        "populated_h3_experience_blockers": populated_h3_blockers,
        "sample_observer_rows": [
            {
                "observer_id": row.get("observer_id"),
                "support_patch_count": row.get("support_patch_count"),
                "committed_fraction": row.get("committed_fraction"),
                "modular_depth_mean": row.get("modular_depth_mean"),
                "modular_depth_std": row.get("modular_depth_std"),
                "observer_relative_times": row.get("observer_relative_times"),
            }
            for row in patch_rows[: min(16, len(patch_rows))]
        ],
        "claim_boundary": (
            "Observer-facing modular-time and H3 experience receipt. The modular-time subreceipt is "
            "observer-local. The 3+1D/H3 experience receipt additionally requires either declared BW/KMS "
            "branch replay or the finite endogenous L2/L3 modular-clock receipt, the conformal H3 chart, "
            "and H3 modular-response evidence. Non-boundary "
            "observer object population is reported separately as observer_facing_populated_h3_experience_receipt; "
            "it is not part of the paper-side D3 observer-facing chart claim and is not chart-blind strict neutral bulk."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=DEMO,
        receipt=OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT,
        physical_claim=False,
        observable_id="observer_local_modular_time_and_h3_chart",
        fit_objective="observer_modular_time_plus_h3_experience_gates",
    )


def _emergence_status_report(bw_report: dict[str, Any], consensus_report: dict[str, Any]) -> dict[str, Any]:
    control_medians = {
        name: float(report["median"])
        for name, report in bw_report.get("controls", {}).items()
        if isinstance(report, dict)
        and "median" in report
        and _optional_finite_float(report.get("median")) is not None
    }
    bw_median = _optional_finite_float(bw_report.get("median"))
    kinematic_bw_usable = bool(bw_report.get("usable_scalar_observables", True)) and bw_median is not None
    correct_beats_controls = bool(kinematic_bw_usable and control_medians) and all(
        float(bw_median) < value for value in control_medians.values()
    )
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
        "kinematic_bw_usable": kinematic_bw_usable,
        "scalar_observable_count": int(bw_report.get("scalar_observable_count", 0)),
        "skipped_scalar_observables": bw_report.get("skipped_scalar_observables", []),
        "control_medians": control_medians,
        "correct_2pi_beats_all_controls": correct_beats_controls,
        "observer_records_settled": observer_records_settled,
        "observer_views_overlap": observer_views_overlap,
        "claim_boundary": (
            "This run can support finite repair, records, and BW/cap-flow diagnostics. "
            "It does not by itself establish 3D bulk emergence. Bulk emergence is gated by "
            "observer/object population of the conformal H3 chart, not by neutral summary-distance "
            "dimension estimates."
        ),
    }


def _optional_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _lorentz_branch_receipts(
    conformal_chart_report: dict[str, Any],
    state_bw_report: dict[str, Any],
    transition_selection_report: dict[str, Any],
) -> dict[str, Any]:
    chart_lorentz_receipt = bool(conformal_chart_report.get("conformal_h3_spatial_chart_receipt", False))
    direct_kms_selection_receipt = bool(
        transition_selection_report.get("primary_source") == "kms_collar_transport_response"
        and transition_selection_report.get("two_pi_selected", False)
        and not transition_selection_report.get("response_degenerate", False)
    )
    bw_automorphism_sanity_receipt = bool(
        direct_kms_selection_receipt
        or (
            state_bw_report.get("direct_transition_automorphism", False)
            and state_bw_report.get("state_selected_2pi", False)
            and state_bw_report.get("correct_beats_controls", False)
        )
    )
    if (
        "ENDOGENOUS_MODULAR_GENERATOR_RECEIPT" in state_bw_report
        or "endogenous_modular_generator_receipt" in state_bw_report
    ):
        endogenous_modular_generator_receipt = bool(
            state_bw_report.get("ENDOGENOUS_MODULAR_GENERATOR_RECEIPT", False)
            or state_bw_report.get("endogenous_modular_generator_receipt", False)
        )
    else:
        endogenous_modular_generator_receipt = bool(
            state_bw_report.get("endogenous_modular_generator", False)
            and state_bw_report.get("state_selected_2pi", False)
            and state_bw_report.get("correct_beats_controls", False)
        )
    declared_cap_flow_generator_diagnostic = bool(state_bw_report.get("declared_cap_flow_generator", False))
    support_visible_lorentz_receipt = bool(chart_lorentz_receipt and bw_automorphism_sanity_receipt)
    bw_branch_replay_receipt = bool(bw_automorphism_sanity_receipt)
    finite_lorentz_contract_receipt = False
    return {
        "CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT": chart_lorentz_receipt,
        "BW_AUTOMORPHISM_SANITY_RECEIPT": bw_automorphism_sanity_receipt,
        BW_KMS_BRANCH_REPLAY_RECEIPT: bw_branch_replay_receipt,
        BW_KMS_BRANCH_INSTANTIATION_RECEIPT: bw_branch_replay_receipt,
        "ENDOGENOUS_MODULAR_GENERATOR_RECEIPT": endogenous_modular_generator_receipt,
        CHART_LORENTZ_H3_RECEIPT: bool(chart_lorentz_receipt and bw_automorphism_sanity_receipt),
        BW_KMS_DIRECT_2PI_RECEIPT: direct_kms_selection_receipt,
        ENDOGENOUS_MODULAR_GENERATOR_RECEIPT: endogenous_modular_generator_receipt,
        OPH_LORENTZ_THEOREM_FINITE_CONTRACT_RECEIPT: finite_lorentz_contract_receipt,
        "DECLARED_CAP_FLOW_GENERATOR_DIAGNOSTIC": declared_cap_flow_generator_diagnostic,
        "bw_kms_branch_replay_receipt": bw_branch_replay_receipt,
        "direct_kms_collar_2pi_receipt": direct_kms_selection_receipt,
        "chart_level_conformal_lorentz_receipt": chart_lorentz_receipt,
        "bw_automorphism_sanity_receipt": bw_automorphism_sanity_receipt,
        "endogenous_modular_generator_receipt": endogenous_modular_generator_receipt,
        "finite_lorentz_theorem_contract_receipt": finite_lorentz_contract_receipt,
        "paper_route_lorentz_h3_chart_receipt": support_visible_lorentz_receipt,
        "declared_cap_flow_generator_diagnostic": declared_cap_flow_generator_diagnostic,
        "support_visible_lorentz_3p1_kinematics_receipt": support_visible_lorentz_receipt,
        "lorentz_receipt_taxonomy": "L0_branch_replay_not_L1_L7_finite_contract",
    }


def _canonical_receipt_ladder_report(
    *,
    trace: list[dict[str, Any]],
    committed: np.ndarray,
    emergence_status: dict[str, Any],
    state_bw_report: dict[str, Any],
    transition_selection_report: dict[str, Any],
    h3_modular_fit_report: dict[str, Any],
    observer_chart_object_report: dict[str, Any],
    cosmology_gate_report: dict[str, Any],
    galaxy_proxy_report: dict[str, Any],
) -> dict[str, Any]:
    final_phi = int(trace[-1].get("phi", -1)) if trace else -1
    committed_fraction = float(np.mean(committed)) if np.asarray(committed).size else 0.0
    bw_kms_pass = bool(
        emergence_status.get("BW_AUTOMORPHISM_SANITY_RECEIPT", False)
        or emergence_status.get(BW_KMS_BRANCH_REPLAY_RECEIPT, False)
        or state_bw_report.get(BW_KMS_BRANCH_REPLAY_RECEIPT, False)
        or state_bw_report.get(BW_KMS_BRANCH_INSTANTIATION_RECEIPT, False)
        or (
            transition_selection_report.get("primary_source") == "kms_collar_transport_response"
            and transition_selection_report.get("two_pi_selected", False)
            and not transition_selection_report.get("response_degenerate", False)
        )
    )
    receipts = {
        "R0": {
            "receipt_name": REPAIR_CORE_RECEIPT,
            "passed": bool(final_phi == 0),
            "claim_level": RECOVERED_CORE,
            "observable_id": "overlap_mismatch_phi",
            "fit_objective": "final_phi_zero",
            "canonical_tier": "C0a",
            "diagnostic_receipt_name": FINITE_SETTLE_DIAGNOSTIC_RECEIPT,
            "not_finite_consensus_theorem": True,
            "final_phi": final_phi,
        },
        "R1": {
            "receipt_name": RECORD_COMMIT_RECEIPT,
            "passed": bool(committed_fraction >= 0.95),
            "claim_level": RECOVERED_CORE,
            "observable_id": "observer_record_commit_mask",
            "fit_objective": "committed_fraction_at_least_0p95",
            "committed_fraction": committed_fraction,
        },
        "R2": {
            "receipt_name": BW_KMS_BRANCH_REPLAY_RECEIPT,
            "passed": bw_kms_pass,
            "claim_level": BRANCH_INSTANTIATION_SANITY,
            "observable_id": "kms_collar_transport_response",
            "fit_objective": "declared_two_pi_branch_replay_selected_and_non_degenerate",
            "canonical_tier": "L0",
            "legacy_receipt_name": BW_KMS_BRANCH_INSTANTIATION_RECEIPT,
            "not_finite_lorentz_theorem_contract": True,
            "state_selected_2pi": bool(state_bw_report.get("state_selected_2pi", False)),
            "transition_two_pi_selected": bool(transition_selection_report.get("two_pi_selected", False)),
        },
        "R3": {
            "receipt_name": CHART_LORENTZ_H3_RECEIPT,
            "passed": bool(emergence_status.get("CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT", False)),
            "claim_level": BRANCH_INSTANTIATION_SANITY,
            "observable_id": "cap_normals_conformal_h3_chart",
            "fit_objective": "chart_instantiated",
            "legacy_receipt_name": CONFORMAL_H3_CHART_RECEIPT,
        },
        "R4": {
            "receipt_name": H3_RESPONSE_CANDIDATE_RECEIPT,
            "passed": bool(
                h3_modular_fit_report.get(H3_RESPONSE_CANDIDATE_RECEIPT, False)
                or h3_modular_fit_report.get("MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", False)
            ),
            "claim_level": DEMO,
            "observable_id": "support_visible_modular_response_kernel",
            "fit_objective": "heldout_h3_response_beats_controls",
            "intermediate_receipt_name": H3_RESPONSE_CONTROL_SEPARATION_RECEIPT,
            "intermediate_passed": bool(
                h3_modular_fit_report.get(H3_RESPONSE_CONTROL_SEPARATION_RECEIPT, False)
                or h3_modular_fit_report.get("h3_control_separation_receipt", False)
                or h3_modular_fit_report.get("h3_response_stage_gates", {}).get(
                    H3_RESPONSE_CONTROL_SEPARATION_RECEIPT, False
                )
                or h3_modular_fit_report.get("h3_response_stage_gates", {}).get(
                    "intermediate_control_separation_receipt", False
                )
            ),
            "claim_boundary": (
                "R4 is strict. The intermediate receipt is a support-visible H3 control-separation "
                "precursor only and cannot establish a 3D bulk."
            ),
        },
        "R5": {
            "receipt_name": OBJECT_CHART_RECEIPT,
            "passed": bool(observer_chart_object_report.get("observer_chart_object_h3_receipt", False)),
            "claim_level": DEMO,
            "observable_id": "observer_transition_history_object_families",
            "fit_objective": "object_chart_construction",
        },
        "R6": {
            "receipt_name": OBJECT_BULK_POPULATION_RECEIPT,
            "passed": bool(emergence_status.get(OBJECT_BULK_POPULATION_RECEIPT, False)),
            "claim_level": DEMO,
            "observable_id": "h3_populated_object_families",
            "fit_objective": "active_observer_chart_bulk_population_gate",
            "population_source": emergence_status.get("bulk_population_source"),
        },
        "R7": {
            "receipt_name": SCREEN_PROXY_CMB_RECEIPT,
            "passed": bool(cosmology_gate_report.get("enabled", False) and cosmology_gate_report.get("allowed", False)),
            "claim_level": PROXY,
            "observable_id": "freezeout_screen_cl_proxy",
            "fit_objective": "screen_proxy_gate_allowed",
        },
        "R8": {
            "receipt_name": STATIC_GALAXY_RAR_BTFR_RECEIPT,
            "passed": bool(galaxy_proxy_report.get("GALAXY_PROXY_RECEIPT", False)),
            "claim_level": PROXY,
            "observable_id": "oph_static_galaxy_proxy",
            "fit_objective": "rar_btfr_formula_bookkeeping",
            "legacy_receipt_name": STATIC_GALAXY_LAW_RECEIPT,
        },
        "R9": {
            "receipt_name": DYNAMIC_DARK_TRANSPORT_RECEIPT,
            "passed": False,
            "claim_level": PROXY,
            "observable_id": "dynamic_dark_transport",
            "fit_objective": "not_implemented",
        },
        "R10": {
            "receipt_name": COSMOLOGY_PERTURBATION_RECEIPT,
            "passed": False,
            "claim_level": PROXY,
            "observable_id": "cosmology_perturbation_adapter",
            "fit_objective": "not_implemented",
        },
    }
    for row in receipts.values():
        row["receipt_schema_version"] = RECEIPT_SCHEMA_VERSION
        row["physical_claim"] = False
    report = {
        "mode": "canonical_receipt_ladder",
        "receipts": receipts,
        "passed_receipt_names": [row["receipt_name"] for row in receipts.values() if row["passed"]],
        "bulk_3d_established": bool(emergence_status.get("bulk_3d_established", False)),
        FINITE_CONSENSUS_THEOREM_RECEIPT: bool(
            emergence_status.get(FINITE_CONSENSUS_THEOREM_RECEIPT, False)
        ),
        "finite_consensus_theorem_receipt": bool(
            emergence_status.get("finite_consensus_theorem_receipt", False)
        ),
        OPH_LORENTZ_THEOREM_FINITE_CONTRACT_RECEIPT: False,
        "finite_lorentz_theorem_contract_receipt": False,
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "claim_boundary": (
            "canonical lane ladder. R2 is now L0 branch replay: it confirms the declared BW/KMS "
            "2pi route executes and beats implemented controls, but it is not the L1-L7 finite "
            "Lorentz theorem contract. R3 is the conformal H3 chart route; R6 is an object-population "
            "gate but does not establish chart-blind 3D bulk. R7 is a screen proxy only and R8 is a "
            "static-galaxy proxy, not a full cosmology prediction."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=BRANCH_INSTANTIATION_SANITY,
        receipt="CANONICAL_RECEIPT_LADDER",
        physical_claim=False,
        observable_id="finite_screen_receipt_ladder",
        fit_objective="receipt_gate_summary",
    )


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


def _attach_transition_affinity_histograms(
    observer_rows: list[dict[str, Any]],
    raw_fields: dict[str, np.ndarray],
    object_cfg: dict[str, Any],
) -> None:
    fields = transition_affinity_packet_fields(raw_fields, object_cfg)
    if not fields:
        return
    for row in observer_rows:
        if row.get("view_type") != "patch_observer":
            continue
        support = np.asarray(row.get("support_nodes", []), dtype=np.int64)
        histograms: dict[str, dict[str, float]] = {}
        dominants: dict[str, int | None] = {}
        for name, packets in fields.items():
            packets = np.asarray(packets, dtype=np.int64)
            valid = support[(support >= 0) & (support < packets.size)]
            if valid.size == 0:
                histograms[str(name)] = {}
                dominants[str(name)] = None
                continue
            unique, counts = np.unique(packets[valid], return_counts=True)
            total = float(counts.sum())
            histograms[str(name)] = {str(int(key)): float(count / total) for key, count in zip(unique, counts, strict=True)}
            dominants[str(name)] = int(unique[int(np.argmax(counts))])
        row["transition_affinity_histograms"] = histograms
        row["transition_affinity_dominants"] = dominants


def _attach_modular_response_histograms(
    observer_rows: list[dict[str, Any]],
    response_kernel: dict[str, Any],
    chart_cfg: dict[str, Any],
) -> None:
    matrix = np.asarray(response_kernel.get("matrix", np.zeros((0, 0))), dtype=float)
    observer_ids = [int(value) for value in response_kernel.get("observer_ids", [])]
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0 or len(observer_ids) != matrix.shape[0]:
        return
    components = max(1, int(chart_cfg.get("modular_response_cluster_components", 2)))
    bins = max(2, int(chart_cfg.get("modular_response_cluster_bins", 4)))
    component_bins, cluster_tokens = _modular_response_cluster_tokens(
        matrix,
        components=components,
        bins=bins,
    )
    row_by_id = {
        int(row.get("observer_id", -1)): row
        for row in observer_rows
        if row.get("view_type") == "patch_observer"
    }
    for row_index, observer_id in enumerate(observer_ids):
        row = row_by_id.get(int(observer_id))
        if row is None:
            continue
        histograms = dict(row.get("modular_response_histograms", {}) or {})
        token = int(cluster_tokens[row_index])
        histograms["modular_response_cluster"] = {str(token): 1.0}
        for component_index in range(component_bins.shape[1]):
            histograms[f"modular_response_component_{component_index}"] = {
                str(int(component_bins[row_index, component_index])): 1.0
            }
        row["modular_response_histograms"] = histograms
        row["modular_response_cluster"] = int(token)
        response_row = np.asarray(matrix[row_index], dtype=float)
        if response_row.size:
            response_row = np.where(np.isfinite(response_row), response_row, 0.0)
            centered = response_row - float(np.mean(response_row))
            scale = float(np.std(centered))
            if scale > 1e-12:
                centered = centered / scale
            row["repair_response_spectrum"] = [float(value) for value in centered[:32]]
        row["perturb_resettle_signature"] = [
            float(value)
            for value in component_bins[row_index, : min(component_bins.shape[1], 8)]
        ]


def _modular_response_cluster_tokens(
    matrix: np.ndarray,
    *,
    components: int,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, keepdims=True)
    scale[scale < 1e-9] = 1.0
    standardized = centered / scale
    try:
        u, singular_values, _vh = np.linalg.svd(standardized, full_matrices=False)
        embedding = u[:, :components] * singular_values[:components][None, :]
    except np.linalg.LinAlgError:
        embedding = standardized[:, :components]
    if embedding.shape[1] < int(components):
        padding = np.zeros((embedding.shape[0], int(components) - embedding.shape[1]), dtype=float)
        embedding = np.hstack([embedding, padding])
    component_bins = np.zeros((embedding.shape[0], int(components)), dtype=np.int64)
    for component_index in range(int(components)):
        values = embedding[:, component_index]
        edges = np.quantile(values, np.linspace(0.0, 1.0, int(bins) + 1)[1:-1])
        if edges.size and np.allclose(edges, edges[0]):
            component_bins[:, component_index] = 0
        else:
            component_bins[:, component_index] = np.digitize(values, edges, right=False)
    cluster_tokens = np.asarray(
        [
            _stable_hash_to_int(stable_json_hash([int(value) for value in row]))
            for row in component_bins
        ],
        dtype=np.int64,
    )
    return component_bins, cluster_tokens


def _attach_transition_history_histograms(
    observer_rows: list[dict[str, Any]],
    history_raw_fields: list[dict[str, np.ndarray]],
    object_cfg: dict[str, Any],
) -> None:
    field_names = [
        str(field)
        for field in object_cfg.get(
            "transition_history_fields",
            object_cfg.get(
                "transition_affinity_fields",
                [
                    "record_family",
                    "checkpoint_class",
                    "stable_flag",
                    "s3_sector_class",
                    "repair_load_bucket",
                ],
            ),
        )
    ]
    key_field_names = [
        str(field)
        for field in object_cfg.get(
            "transition_history_key_fields",
            field_names,
        )
    ]
    persistence_field_names = [
        str(field)
        for field in object_cfg.get(
            "transition_persistence_fields",
            ["record_family", "s3_sector_class"],
        )
    ]
    include_readout_hash = bool(object_cfg.get("transition_history_include_readout_hash", False))
    readout_hash_prefix_chars = max(1, int(object_cfg.get("transition_history_readout_hash_prefix_chars", 8)))
    if not field_names or not history_raw_fields:
        return
    history_fields = [transition_affinity_packet_fields(raw_fields, object_cfg) for raw_fields in history_raw_fields]
    history_fields = [fields for fields in history_fields if fields]
    if not history_fields:
        return
    for row in observer_rows:
        if row.get("view_type") != "patch_observer":
            continue
        support = np.asarray(row.get("support_nodes", []), dtype=np.int64)
        steps: list[dict[str, int]] = []
        step_masses: list[float] = []
        field_paths: dict[str, list[int]] = {name: [] for name in field_names}
        for fields in history_fields:
            descriptor: dict[str, int] = {}
            mass_terms: list[float] = []
            for name in field_names:
                packets = fields.get(name)
                if packets is None:
                    continue
                packets = np.asarray(packets, dtype=np.int64)
                valid = support[(support >= 0) & (support < packets.size)]
                if valid.size == 0:
                    continue
                values, counts = np.unique(packets[valid], return_counts=True)
                dominant = int(values[int(np.argmax(counts))])
                descriptor[str(name)] = dominant
                field_paths[str(name)].append(dominant)
                mass_terms.append(float(np.max(counts) / max(1, counts.sum())))
            if descriptor:
                steps.append(descriptor)
                step_masses.append(float(np.mean(mass_terms)) if mass_terms else 0.0)
        if not steps:
            row["transition_history_key"] = None
            row["transition_history_persistence"] = 0
            row["transition_history_histogram"] = {}
            continue
        persistence = _transition_history_persistence(steps, fields=persistence_field_names)
        descriptor_payload = {
            "fields": field_names,
            "steps": steps,
            "persistence": int(persistence),
            "persistence_fields": persistence_field_names,
            "sector_change_count": _path_change_count(field_paths.get("s3_sector_class", [])),
            "record_family_change_count": _path_change_count(field_paths.get("record_family", [])),
        }
        key_payload = {
            "key_fields": key_field_names,
            "field_counts": {
                name: _path_value_counts(field_paths.get(name, []))
                for name in key_field_names
                if field_paths.get(name)
            },
            "final": {name: steps[-1][name] for name in key_field_names if name in steps[-1]},
            "persistence": int(persistence),
            "sector_change_count": _path_change_count(field_paths.get("s3_sector_class", [])),
            "record_family_change_count": _path_change_count(field_paths.get("record_family", [])),
        }
        if include_readout_hash:
            readout_hash = str(row.get("visible_readout_hash", ""))
            if ":" in readout_hash:
                readout_hash = readout_hash.split(":", 1)[1]
            key_payload["observer_visible_readout_hash_prefix"] = readout_hash[:readout_hash_prefix_chars]
        signature_hash = stable_json_hash(key_payload)
        signature = _stable_hash_to_int(signature_hash)
        row["transition_history_key"] = signature
        row["transition_history_hash"] = signature_hash
        row["transition_history_descriptor"] = descriptor_payload
        row["transition_history_key_descriptor"] = key_payload
        row["transition_history_persistence"] = int(persistence)
        row["transition_history_mean_modal_mass"] = float(np.mean(step_masses)) if step_masses else 0.0
        row["transition_history_histogram"] = {str(signature): 1.0}
        local_token_histogram, local_token_persistent_histogram = _local_transition_token_histograms(
            support,
            history_fields,
            key_field_names=key_field_names,
            persistence_field_names=persistence_field_names,
            min_persistence=int(object_cfg.get("transition_history_local_min_persistence", 1)),
        )
        row["transition_history_histograms"] = {
            "transition_history_key": {str(signature): 1.0},
            "local_transition_token": local_token_histogram,
            "local_transition_token_persistent": local_token_persistent_histogram,
            **{
                f"{name}_path": {str(_stable_hash_to_int(stable_json_hash(values))): 1.0}
                for name, values in field_paths.items()
                if values
            },
        }
        row["record_transition_histogram"] = _histogram_summary_vector(
            local_token_persistent_histogram or local_token_histogram or row["transition_history_histogram"],
            max_items=16,
        )
        row["checkpoint_class_transition"] = _histogram_summary_vector(
            row["transition_history_histograms"].get("checkpoint_class_path", {}),
            max_items=8,
        )
        row["sector_change_signature"] = [
            float(descriptor_payload["sector_change_count"]),
            float(descriptor_payload["record_family_change_count"]),
            float(persistence),
            float(row["transition_history_mean_modal_mass"]),
        ]
        row["counterfactual_stability"] = float(row["transition_history_mean_modal_mass"])


def _transition_history_persistence(steps: list[dict[str, int]], *, fields: list[str]) -> int:
    if not steps:
        return 0
    selected_fields = [str(field) for field in fields if str(field) in steps[-1]]
    final = {field: steps[-1][field] for field in selected_fields} if selected_fields else steps[-1]
    count = 0
    for step in reversed(steps):
        current = {field: step[field] for field in selected_fields if field in step} if selected_fields else step
        if current != final:
            break
        count += 1
    return int(count)


def _local_transition_token_histograms(
    support: np.ndarray,
    history_fields: list[dict[str, np.ndarray]],
    *,
    key_field_names: list[str],
    persistence_field_names: list[str],
    min_persistence: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Observer-visible per-node transition history tokens for local objects.

    The support-modal transition history intentionally summarizes what an
    observer sees as one packet. This helper keeps the bounded local packet
    mixture inside the same observer support, so localized sub-histories are not
    erased by a dominant background packet. It uses only transition packet
    fields already exposed to observer rows.
    """

    support = np.asarray(support, dtype=np.int64)
    if support.size == 0 or not history_fields:
        return {}, {}
    names = [str(name) for name in key_field_names if str(name)]
    if not names:
        names = sorted({str(name) for fields in history_fields for name in fields.keys()})
    counts: dict[int, int] = {}
    persistent_counts: dict[int, int] = {}
    for node in support:
        node_index = int(node)
        steps: list[dict[str, int]] = []
        paths: dict[str, list[int]] = {name: [] for name in names}
        for fields in history_fields:
            descriptor: dict[str, int] = {}
            for name in names:
                values = fields.get(name)
                if values is None:
                    continue
                values = np.asarray(values, dtype=np.int64)
                if node_index < 0 or node_index >= values.size:
                    continue
                value = int(values[node_index])
                descriptor[str(name)] = value
                paths[str(name)].append(value)
            if descriptor:
                steps.append(descriptor)
        if not steps:
            continue
        persistence = _transition_history_persistence(steps, fields=persistence_field_names)
        token_payload = {
            "key_fields": names,
            "steps": steps,
            "persistence": int(persistence),
            "sector_change_count": _path_change_count(paths.get("s3_sector_class", [])),
            "record_family_change_count": _path_change_count(paths.get("record_family", [])),
        }
        token = _stable_hash_to_int(stable_json_hash(token_payload))
        counts[token] = counts.get(token, 0) + 1
        if persistence >= int(min_persistence):
            persistent_counts[token] = persistent_counts.get(token, 0) + 1
    return _normalize_int_counts(counts), _normalize_int_counts(persistent_counts)


def _path_change_count(values: list[int]) -> int:
    if len(values) < 2:
        return 0
    return int(sum(1 for before, after in zip(values, values[1:], strict=False) if int(before) != int(after)))


def _path_value_counts(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def _normalize_int_counts(counts: dict[int, int]) -> dict[str, float]:
    total = float(sum(int(value) for value in counts.values()))
    if total <= 0.0:
        return {}
    return {str(int(key)): float(value / total) for key, value in sorted(counts.items())}


def _histogram_summary_vector(histogram: dict[str, float], *, max_items: int) -> list[float]:
    """Distributional observer-visible packet summary for blind reconstruction.

    The blind bulk audit cannot consume support nodes, axes, or cap-membership
    data. Exact transition-token identities are intentionally not included
    here: categorical packet hashes can manufacture high-rank distances. Object
    extraction still receives the full token histograms separately, while the
    blind dimension audit gets only distributional continuation statistics.
    """

    if not histogram:
        return []
    values: list[float] = []
    for value in histogram.values():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value) and numeric_value > 0.0:
            values.append(float(numeric_value))
    if not values:
        return []
    array = np.asarray(sorted(values, reverse=True), dtype=float)
    total = float(np.sum(array))
    if total <= 0.0:
        return []
    probs = array / total
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-15))))
    effective_count = float(1.0 / np.sum(probs * probs))
    tail_mass = float(np.sum(probs[max(1, int(max_items)) :])) if probs.size > int(max_items) else 0.0
    top = probs[: min(3, probs.size)]
    padded = np.zeros(3, dtype=float)
    padded[: top.size] = top
    return [
        entropy,
        effective_count,
        float(probs[0]),
        tail_mass,
        float(probs.size),
        *[float(value) for value in padded],
    ]


def _stable_hash_to_int(hash_value: str, *, hex_digits: int = 12) -> int:
    digest = str(hash_value).split(":", 1)[-1]
    return int(digest[: int(hex_digits)], 16)


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


def _theorem_core_receipts(
    trace: list[dict[str, Any]],
    committed: np.ndarray,
    config: dict[str, Any],
    *,
    initial_port_left: np.ndarray | None = None,
    initial_port_right: np.ndarray | None = None,
    group_order: int = 1,
    seed: int = 1,
) -> dict[str, Any]:
    theorem_cfg = config.get("theorem_core", {}) or {}
    lyapunov = lyapunov_descent_receipt(trace)
    final_phi = int(trace[-1].get("phi", -1)) if trace else -1
    finite_settle = with_claim_metadata(
        {
            "mode": "finite_settle_diagnostic",
            FINITE_SETTLE_DIAGNOSTIC_RECEIPT: bool(trace and final_phi == 0),
            "finite_settle_diagnostic_receipt": bool(trace and final_phi == 0),
            "receipt": bool(trace and final_phi == 0),
            "final_phi": final_phi,
            "final_phi_zero": bool(trace and final_phi == 0),
            "canonical_tier": "C0a",
            "not_finite_consensus_theorem": True,
            "claim_boundary": (
                "C0a settling diagnostic only. final_phi == 0 does not certify strict descent, "
                "local diamonds, repair completeness, or schedule-independent normal form."
            ),
        },
        claim_level=RECOVERED_CORE,
        receipt=FINITE_SETTLE_DIAGNOSTIC_RECEIPT,
        physical_claim=False,
        observable_id="overlap_mismatch_phi",
        fit_objective="final_phi_zero_diagnostic",
    )
    replay_report = _array_port_pair_consensus_replay_report(
        initial_port_left,
        initial_port_right,
        group_order=group_order,
        config=theorem_cfg.get("consensus_replay", {}),
        seed=seed + 31_337,
    )
    consensus_evidence = (
        replay_report.get("evidence", {})
        if replay_report.get("enabled", False)
        else theorem_cfg.get("finite_consensus_evidence") or theorem_cfg.get("consensus_evidence") or {}
    )
    consensus_trace = replay_report.get("sample_events", []) if replay_report.get("enabled", False) else trace
    finite_consensus = finite_consensus_theorem_certificate(
        consensus_trace,
        evidence=consensus_evidence,
        strict_tol=float(theorem_cfg.get("strict_descent_tolerance", 1.0e-12)),
    )
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
        "finite_settle_diagnostic": finite_settle,
        "finite_consensus_theorem": finite_consensus,
        "finite_consensus_replay": replay_report,
        FINITE_SETTLE_DIAGNOSTIC_RECEIPT: bool(finite_settle.get("receipt", False)),
        FINITE_CONSENSUS_THEOREM_RECEIPT: bool(finite_consensus.get("receipt", False)),
        "finite_settle_diagnostic_receipt": bool(finite_settle.get("receipt", False)),
        "finite_consensus_theorem_receipt": bool(finite_consensus.get("receipt", False)),
        "lyapunov": lyapunov,
        "exact_repair_projection": exact_repair,
        "sm_quotient_gate": sm_gate,
        "claim_boundary": (
            "fast finite theorem-instantiation bundle. C0a/final settling is diagnostic only. "
            "C0b finite consensus requires theorem-phase replay evidence and fails closed when "
            "that evidence is absent. The Lyapunov and projection receipts are fixed-cutoff "
            "recovered-core checks; the SM quotient gate is a declared continuation candidate sieve when supplied"
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=RECOVERED_CORE,
        receipt="THEOREM_CORE_RECEIPT_BUNDLE",
        physical_claim=False,
        observable_id="finite_theorem_core_receipts",
        fit_objective="lyapunov_and_projection_receipts",
    )


def _array_port_pair_consensus_replay_report(
    initial_port_left: np.ndarray | None,
    initial_port_right: np.ndarray | None,
    *,
    group_order: int,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return {
            "mode": "array_port_pair_strict_consensus_replay",
            "enabled": False,
            "evidence": {},
            "sample_events": [],
            "claim_boundary": "disabled; C0b finite consensus receipt remains fail-closed",
        }
    if initial_port_left is None or initial_port_right is None:
        return {
            "mode": "array_port_pair_strict_consensus_replay",
            "enabled": True,
            "receipt": False,
            "evidence": {},
            "sample_events": [],
            "claim_boundary": "missing initial port-pair state; C0b finite consensus receipt remains fail-closed",
        }
    left0 = np.asarray(initial_port_left, dtype=np.int16)
    right0 = np.asarray(initial_port_right, dtype=np.int16)
    if left0.shape != right0.shape:
        raise ValueError("initial theorem replay port arrays must have matching shape")

    rng = np.random.default_rng(seed)
    active = np.flatnonzero(left0 != right0).astype(np.int64)
    initial_phi = int(active.size)
    schedule_replays = int(cfg.get("schedule_replays", 16))
    requested_schedule_replays = int(cfg.get("requested_schedule_replays", schedule_replays))
    max_event_rows = int(cfg.get("max_event_rows", 256))
    disjoint_checks = int(cfg.get("disjoint_checks", 256))
    terminal_hashes: list[str] = []
    first_sample_events: list[dict[str, Any]] = []
    strict_descent_violations = 0
    phi_increase_violations = 0
    terminal_phi_violations = 0
    for replay_index in range(max(0, schedule_replays)):
        order = active.copy()
        rng.shuffle(order)
        left = left0.copy()
        right = right0.copy()
        phi = initial_phi
        replay_events: list[dict[str, Any]] = []
        for step, edge in enumerate(order):
            before = int(left[edge] != right[edge])
            if before == 0:
                continue
            capture_event = replay_index == 0 and len(replay_events) < max_event_rows
            quotient_hash_before = _array_port_pair_hash(left, right, group_order) if capture_event else None
            canonical = min(int(left[edge]), int(right[edge]))
            left[edge] = canonical
            right[edge] = canonical
            after = int(left[edge] != right[edge])
            delta_touched = after - before
            phi_after = phi + delta_touched
            delta_global = phi_after - phi
            if delta_touched >= 0:
                strict_descent_violations += 1
            if delta_global > 0:
                phi_increase_violations += 1
            if capture_event:
                event = {
                    "cycle": step,
                    "phase": "theorem",
                    "node": int(edge),
                    "move_id": "canonical_port_pair_equalization",
                    "touched_edges": [int(edge)],
                    "touched_phi_before": before,
                    "touched_phi_after": after,
                    "global_phi_before": phi,
                    "global_phi_after": phi_after,
                    "delta_touched_phi": delta_touched,
                    "delta_global_phi": delta_global,
                    "quotient_hash_before": quotient_hash_before,
                    "accepted": True,
                    "theorem_eligible": True,
                    "reason": "strict_canonical_edge_normalization",
                }
            phi = phi_after
            if capture_event:
                event["quotient_hash_after"] = _array_port_pair_hash(left, right, group_order)
                replay_events.append(event)
        if phi != 0:
            terminal_phi_violations += 1
        terminal_hashes.append(_array_port_pair_hash(left, right, group_order))
        if replay_index == 0:
            first_sample_events = replay_events

    disjoint_violation_count = _array_port_pair_disjoint_commutation_violations(
        left0,
        right0,
        active,
        group_order=group_order,
        checks=disjoint_checks,
        rng=rng,
    )
    unique_terminal_hashes = sorted(set(terminal_hashes))
    evidence = {
        "theorem_phase_event_count": initial_phi,
        "accepted_theorem_move_count": initial_phi,
        "strict_descent_violation_count": strict_descent_violations,
        "accepted_phi_increase_violation_count": phi_increase_violations,
        "disjoint_commutation_violation_count": disjoint_violation_count,
        "local_diamond_violation_count": 0,
        "repair_completeness_violation_count": terminal_phi_violations,
        "unique_terminal_quotient_hash_count": len(unique_terminal_hashes),
        "schedule_replay_count": schedule_replays,
        "requested_schedule_replays": requested_schedule_replays,
    }
    receipt = bool(
        initial_phi > 0
        and strict_descent_violations == 0
        and phi_increase_violations == 0
        and disjoint_violation_count == 0
        and terminal_phi_violations == 0
        and len(unique_terminal_hashes) == 1
        and schedule_replays >= requested_schedule_replays
    )
    return {
        "mode": "array_port_pair_strict_consensus_replay",
        "enabled": True,
        "receipt": receipt,
        "evidence": evidence,
        "sample_events": first_sample_events,
        "initial_phi": initial_phi,
        "initial_edge_count": int(left0.size),
        "terminal_hash": unique_terminal_hashes[0] if unique_terminal_hashes else None,
        "unique_terminal_hash_count": len(unique_terminal_hashes),
        "sample_event_count": len(first_sample_events),
        "local_diamond_checked_pair_count": 0,
        "local_diamond_status": "vacuous_for_single_edge_canonical_rewrite",
        "claim_boundary": (
            "C0b replay evidence for the finite array port-pair quotient only. The theorem-phase "
            "normalizer is strict and deterministic; it does not certify OPH record algebra C1 or "
            "Lorentz L1-L7 receipts."
        ),
    }


def _array_port_pair_disjoint_commutation_violations(
    left0: np.ndarray,
    right0: np.ndarray,
    active: np.ndarray,
    *,
    group_order: int,
    checks: int,
    rng: np.random.Generator,
) -> int:
    if active.size < 2 or checks <= 0:
        return 0
    violation_count = 0
    for _ in range(min(checks, int(active.size * (active.size - 1) // 2))):
        first, second = rng.choice(active, size=2, replace=False)
        first = int(first)
        second = int(second)
        first_value_ab = min(int(left0[first]), int(right0[first]))
        second_value_ab = min(int(left0[second]), int(right0[second]))
        first_value_ba = min(int(left0[first]), int(right0[first]))
        second_value_ba = min(int(left0[second]), int(right0[second]))
        if (first_value_ab, second_value_ab) != (first_value_ba, second_value_ba):
            violation_count += 1
    return violation_count


def _array_port_pair_hash(left: np.ndarray, right: np.ndarray, group_order: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(int(group_order)).encode("ascii"))
    hasher.update(np.asarray(left, dtype=np.int16).tobytes())
    hasher.update(np.asarray(right, dtype=np.int16).tobytes())
    return "sha256:" + hasher.hexdigest()


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
        "angular_prefactor": config.get(
            "angular_prefactor",
            config.get("collar_angular_prefactor", config.get("collar_c", 0.8)),
        ),
        "angular_exponent": config.get(
            "angular_exponent",
            config.get("collar_angular_exponent", 0.25),
        ),
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
    patch_count = max(1, int(patch_count))
    if mode in {"double_scaling", "paper_double_scaling", "n_quarter"}:
        prefactor = float(config.get("angular_prefactor", config.get("collar_c", 0.8)))
        exponent = float(config.get("angular_exponent", 0.25))
        return prefactor * float(patch_count) ** (-exponent)
    if mode in {"cell_scaled", "k_neighborhood"}:
        cell_angular_scale = math.sqrt(4.0 * math.pi / patch_count)
        return float(config.get("collar_k", 1.0)) * cell_angular_scale
    return float(config.get("collar_width", 0.03))


def _collar_report(config: dict[str, Any], patch_count: int, collar_width: float) -> dict[str, Any]:
    mode = str(config.get("collar_width_mode", "fixed"))
    patch_count = max(1, int(patch_count))
    cell_angular_scale = math.sqrt(4.0 * math.pi / patch_count)
    collar_to_cell_ratio = float(collar_width) / max(cell_angular_scale, 1.0e-12)
    angular_prefactor = float(config.get("angular_prefactor", config.get("collar_c", 0.8)))
    angular_exponent = float(config.get("angular_exponent", 0.25))
    double_scaling = mode in {"double_scaling", "paper_double_scaling", "n_quarter"}
    return {
        "mode": mode,
        "collar_width": float(collar_width),
        "collar_k": float(config.get("collar_k", 1.0)),
        "cell_angular_scale": cell_angular_scale,
        "collar_to_cell_ratio": collar_to_cell_ratio,
        "angular_prefactor": angular_prefactor,
        "angular_exponent": angular_exponent,
        "double_scaling_delta_to_zero": bool(double_scaling and angular_exponent > 0.0),
        "double_scaling_collar_to_cell_diverges": bool(double_scaling and angular_exponent < 0.5),
        "double_scaling_formula": "delta_N = c * N^(-alpha); paper default c=0.8, alpha=0.25",
        "claim_boundary": (
            "finite regulator collar; double_scaling mode instantiates delta_N -> 0 and "
            "delta_N / ell_UV -> infinity for 0 < alpha < 1/2"
        ),
    }


def _geometry_cache_from_config(
    points: np.ndarray,
    config: dict[str, Any],
    outputs_cfg: dict[str, Any],
) -> GeometryCache | None:
    cache_cfg = dict(config.get("cache", {}).get("geometry", {}) or {})
    output_cache_cfg = outputs_cfg.get("geometry_cache")
    if isinstance(output_cache_cfg, dict):
        cache_cfg = {**cache_cfg, **output_cache_cfg}
    elif isinstance(output_cache_cfg, str):
        cache_cfg["cache_dir"] = output_cache_cfg
        cache_cfg["enabled"] = True
    enabled = bool(cache_cfg.get("enabled", False))
    cache_dir = cache_cfg.get("cache_dir", cache_cfg.get("dir"))
    if not enabled and cache_dir is None:
        return None
    if cache_dir is None:
        cache_dir = ".oph_fpe_cache/geometry"
    return GeometryCache(points, cache_dir=Path(cache_dir))


def _run_id(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name).lower()).strip("_")
    return f"{slug}_{int(time.time())}"
