"""Run the target-blind coupled internal-observables campaign.

This pipeline owns a declared diagnostic model variant.  It does not replace
the protected finite-consensus kernel and it does not identify graph hops,
update cycles, field values, defects, or excitation candidates with physical
cosmology.  Its purpose is to create a fully replayable run bundle in which
genuinely graph-local coupled dynamics can be interrogated from observer
records and exact paired counterfactuals.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from oph_fpe.core.icosahedral import geodesic_icosahedral_patch_arrays
from oph_fpe.cosmology.observer_excitation_observables import (
    write_observer_excitation_observables,
)
from oph_fpe.cosmology.structural_observables import structural_observables_report
from oph_fpe.dynamics.coupled_patch import (
    CoupledPatchConfig,
    CoupledPatchResult,
    LocalizedIntervention,
    run_collision_counterfactual,
    run_paired_counterfactual,
    simulate_coupled_patch,
    write_coupled_patch_run,
)


SCHEMA = "oph_coupled_internal_observables_campaign_v1"
CLASSIFICATION = "INTERNAL_DIAGNOSTIC_ONLY"
DEFAULT_CONFIG = Path("configs/coupled_internal_observables_v1.yml")
SOURCE_FILES = (
    "oph_fpe/dynamics/coupled_patch.py",
    "oph_fpe/cosmology/structural_observables.py",
    "oph_fpe/cosmology/observer_excitation_observables.py",
    "oph_fpe/cosmology/coupled_campaign.py",
    "tests/test_coupled_patch.py",
    "tests/test_structural_observables.py",
    "tests/test_observer_excitation_observables.py",
    "tests/test_coupled_campaign.py",
)
DIAGNOSTIC_KEYS = {
    "stored_frame_limit",
    "graph_shell_max_hops",
    "correlation_anchor_count",
    "latent_neighborhood_anchor_count",
    "observer_graph_radius",
    "excitation_threshold",
    "excitation_min_lifetime_frames",
    "persistence_threshold_count",
    "random_walk_probe_count",
    "random_walk_max_steps",
    "graph_spectrum_modes",
    "control_draws",
}
TOP_LEVEL_KEYS = {
    "name",
    "campaign_id",
    "schema",
    "classification",
    "claim_boundary",
    "graph",
    "model",
    "intervention",
    "campaign",
    "diagnostics",
    "outputs",
}
MODEL_KEYS = {
    "cycles",
    "dt",
    "frame_stride",
    "coupling",
    "damping",
    "quartic",
    "mass_squared_start",
    "mass_squared_end",
    "quench_start_cycle",
    "quench_cycles",
    "initial_state_scale",
    "initial_velocity_scale",
    "noise_amplitude",
    "state_bound",
    "velocity_bound",
    "state_levels",
    "velocity_levels",
    "record_threshold",
    "record_persistence",
    "record_amplitude",
    "feedback_strength",
    "defect_threshold",
    "stability_limit",
}
OUTPUT_KEYS = {
    "root",
    "save_full_state_history",
    "save_full_velocity_history",
    "save_full_record_history",
    "save_commit_history",
    "save_defect_history",
    "save_paired_delta_history",
    "write_csv_tables",
    "write_visualization_pack",
    "hash_every_artifact",
}


def run_coupled_campaign(
    config_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    stage: str = "full",
) -> dict[str, Any]:
    """Execute a smoke or full frozen campaign into a new run directory."""

    source_path = Path(config_path).resolve()
    config = _load_config(source_path)
    root = Path(out_dir or config["outputs"]["root"]).resolve()
    if root.exists():
        raise FileExistsError(
            f"campaign output already exists: {root}; runs are append-never"
        )
    if stage not in {"smoke", "full"}:
        raise ValueError("stage must be 'smoke' or 'full'")
    root.mkdir(parents=True)
    (root / "cells").mkdir()
    (root / "aggregate").mkdir()
    (root / "source_snapshot").mkdir()

    _write_yaml(root / "campaign_config.yml", config)
    _snapshot_sources(root, source_path)
    started = _runtime_receipt(config, source_path, stage)
    _write_json(root / "campaign_start_receipt.json", started)

    run_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []

    if stage == "smoke":
        smoke = config["campaign"]["smoke"]
        level = int(smoke["refinement_level"])
        seed = int(smoke["seeds"][0])
        quench_cycles = int(smoke["quench_cycles"][0])
        cycles = int(smoke["cycles"])
        pair_summary = _run_pair_cell(
            root,
            config,
            lane="smoke_paired_intervention",
            level=level,
            seed=seed,
            quench_cycles=quench_cycles,
            cycles=cycles,
        )
        pair_rows.append(pair_summary)
        run_rows.extend(pair_summary["run_index_rows"])
    else:
        matrix = config["campaign"]["seed_and_quench_matrix"]
        matrix_level = int(matrix["refinement_level"])
        for seed in matrix["seeds"]:
            for quench_cycles in matrix["quench_cycles"]:
                summary = _run_baseline_cell(
                    root,
                    config,
                    lane="seed_and_quench_matrix",
                    level=matrix_level,
                    seed=int(seed),
                    quench_cycles=int(quench_cycles),
                )
                baseline_rows.append(summary)
                run_rows.append(summary["run_index_row"])

        refinement = config["campaign"]["refinement_ladder"]
        for level in refinement["levels"]:
            summary = _run_baseline_cell(
                root,
                config,
                lane="refinement_ladder",
                level=int(level),
                seed=int(refinement["seed"]),
                quench_cycles=int(refinement["quench_cycles"]),
            )
            baseline_rows.append(summary)
            run_rows.append(summary["run_index_row"])

        paired = config["campaign"]["paired_intervention"]
        for seed in paired["seeds"]:
            summary = _run_pair_cell(
                root,
                config,
                lane="paired_intervention",
                level=int(paired["refinement_level"]),
                seed=int(seed),
                quench_cycles=int(paired["quench_cycles"]),
            )
            pair_rows.append(summary)
            run_rows.extend(summary["run_index_rows"])

        collision = config["campaign"]["collision_probe"]
        collision_summary = _run_collision_cell(
            root,
            config,
            lane="collision_probe",
            level=int(collision["refinement_level"]),
            seed=int(collision["seed"]),
            quench_cycles=int(collision["quench_cycles"]),
        )
        collision_rows.append(collision_summary)
        run_rows.extend(collision_summary["run_index_rows"])

        for ablation in config["campaign"].get("ablations", []):
            name = str(ablation["name"])
            overrides = {
                key: value for key, value in ablation.items() if key != "name"
            }
            if name == "quantized_no_quartic_no_feedback":
                summary = _run_collision_cell(
                    root,
                    config,
                    lane=f"ablation_{name}",
                    level=int(collision["refinement_level"]),
                    seed=int(collision["seed"]),
                    quench_cycles=int(collision["quench_cycles"]),
                    overrides=overrides,
                )
                collision_rows.append(summary)
                run_rows.extend(summary["run_index_rows"])
            elif name == "no_graph_coupling":
                summary = _run_pair_cell(
                    root,
                    config,
                    lane=f"ablation_{name}",
                    level=int(collision["refinement_level"]),
                    seed=int(collision["seed"]),
                    quench_cycles=int(collision["quench_cycles"]),
                    overrides=overrides,
                )
                pair_rows.append(summary)
                run_rows.extend(summary["run_index_rows"])
            elif name == "no_record_feedback":
                summary = _run_parameter_ablation_cell(
                    root,
                    config,
                    lane=f"ablation_{name}",
                    level=int(collision["refinement_level"]),
                    seed=int(collision["seed"]),
                    quench_cycles=int(collision["quench_cycles"]),
                    overrides=overrides,
                )
                baseline_rows.append(summary["ablation"])
                run_rows.extend(summary["run_index_rows"])
            else:
                summary = _run_baseline_cell(
                    root,
                    config,
                    lane=f"ablation_{name}",
                    level=int(collision["refinement_level"]),
                    seed=int(collision["seed"]),
                    quench_cycles=int(collision["quench_cycles"]),
                    overrides=overrides,
                )
                baseline_rows.append(summary)
                run_rows.append(summary["run_index_row"])

    aggregate = _write_campaign_aggregates(
        root,
        config,
        stage=stage,
        baseline_rows=baseline_rows,
        pair_rows=pair_rows,
        collision_rows=collision_rows,
    )
    _write_json(root / "run_index.json", {"schema": SCHEMA, "runs": run_rows})
    _write_json(root / "observable_catalog.json", _observable_catalog(run_rows))
    _write_campaign_readme(root, config, stage, aggregate)
    # The summary is the final non-index artifact.  Count it before writing so
    # the subsequently generated index can hash the completed summary once.
    pre_summary_artifact_count = len(_artifact_index(root)["artifacts"])
    final = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "stage": stage,
        "campaign_id": config["campaign_id"],
        "root": str(root),
        "run_count": len(run_rows),
        "aggregate": aggregate,
        "artifact_count_excluding_index": pre_summary_artifact_count + 1,
        "target_data_read": False,
        "physical_identification": False,
        "all_requested_observable_families_have_status_rows": True,
        "all_run_arms_zero_state_and_velocity_clipping": bool(
            run_rows
            and all(row["zero_state_and_velocity_clipping"] for row in run_rows)
        ),
    }
    _write_json(root / "campaign_summary.json", final)
    artifact_index = _artifact_index(root)
    _write_json(root / "artifact_index.json", artifact_index)
    return final


def _run_baseline_cell(
    root: Path,
    campaign: Mapping[str, Any],
    *,
    lane: str,
    level: int,
    seed: int,
    quench_cycles: int,
    cycles: int | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cell_id = _cell_id(lane, level, seed, quench_cycles, "baseline")
    cell_root = root / "cells" / cell_id
    points, left, right = _campaign_graph_arrays(campaign, level)
    kernel_config = _kernel_config(
        campaign,
        seed=seed,
        quench_cycles=quench_cycles,
        cycles=cycles,
        overrides=overrides,
    )
    result = simulate_coupled_patch(points, left, right, kernel_config)
    summary = _write_analyzed_run(
        cell_root,
        result,
        campaign,
        lane=lane,
        level=level,
        seed=seed,
        quench_cycles=quench_cycles,
        branch="baseline",
    )
    summary["run_index_row"] = _run_index_row(
        root, cell_root, lane, level, seed, quench_cycles, "baseline", summary
    )
    del result
    gc.collect()
    return summary


def _run_pair_cell(
    root: Path,
    campaign: Mapping[str, Any],
    *,
    lane: str,
    level: int,
    seed: int,
    quench_cycles: int,
    cycles: int | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cell_id = _cell_id(lane, level, seed, quench_cycles, "pair")
    pair_root = root / "cells" / cell_id
    points, left, right = _campaign_graph_arrays(campaign, level)
    kernel_config = _kernel_config(
        campaign,
        seed=seed,
        quench_cycles=quench_cycles,
        cycles=cycles,
        overrides=overrides,
    )
    event = _primary_intervention(campaign, points, kernel_config, quench_cycles)
    pair = run_paired_counterfactual(
        points, left, right, kernel_config, event
    )
    control_root = pair_root / "control"
    treated_root = pair_root / "intervention"
    control_summary = _write_analyzed_run(
        control_root,
        pair.control,
        campaign,
        lane=lane,
        level=level,
        seed=seed,
        quench_cycles=quench_cycles,
        branch="control",
    )
    treated_summary = _write_analyzed_run(
        treated_root,
        pair.intervened,
        campaign,
        lane=lane,
        level=level,
        seed=seed,
        quench_cycles=quench_cycles,
        branch="intervention",
        paired_delta=pair.state_delta_frames,
    )
    receipt = _causal_front_receipt(pair, event)
    _write_json(pair_root / "paired_counterfactual_receipt.json", dict(pair.receipt))
    _write_json(pair_root / "causal_front_receipt.json", receipt)
    _write_rows(pair_root / "causal_front.csv", receipt["rows"])
    np.savez_compressed(
        pair_root / "paired_delta_frames.npz",
        cycles=pair.control.cycles,
        state_delta_frames=pair.state_delta_frames.astype(np.float32),
        velocity_delta_frames=pair.velocity_delta_frames.astype(np.float32),
        record_delta_frames=pair.record_delta_frames.astype(np.float32),
        defect_xor_frames=pair.defect_xor_frames,
    )
    rows = [
        _run_index_row(
            root,
            control_root,
            lane,
            level,
            seed,
            quench_cycles,
            "control",
            control_summary,
        ),
        _run_index_row(
            root,
            treated_root,
            lane,
            level,
            seed,
            quench_cycles,
            "intervention",
            treated_summary,
        ),
    ]
    summary = {
        "lane": lane,
        "level": level,
        "seed": seed,
        "quench_cycles": quench_cycles,
        "event": asdict(event),
        "causal_receipt": receipt,
        "control": control_summary,
        "intervention": treated_summary,
        "run_index_rows": rows,
    }
    del pair
    gc.collect()
    return summary


def _run_parameter_ablation_cell(
    root: Path,
    campaign: Mapping[str, Any],
    *,
    lane: str,
    level: int,
    seed: int,
    quench_cycles: int,
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain a common-random-number main/parameter-ablation packet."""

    cell_id = _cell_id(lane, level, seed, quench_cycles, "parameter_pair")
    pair_root = root / "cells" / cell_id
    points, left, right = _campaign_graph_arrays(campaign, level)
    main_config = _kernel_config(
        campaign, seed=seed, quench_cycles=quench_cycles
    )
    ablation_config = _kernel_config(
        campaign,
        seed=seed,
        quench_cycles=quench_cycles,
        overrides=overrides,
    )
    main_result = simulate_coupled_patch(points, left, right, main_config)
    ablation_result = simulate_coupled_patch(
        points, left, right, ablation_config
    )
    same_initial = bool(
        main_result.provenance["initial_state_sha256"]
        == ablation_result.provenance["initial_state_sha256"]
        and main_result.provenance["initial_velocity_sha256"]
        == ablation_result.provenance["initial_velocity_sha256"]
    )
    same_noise = bool(
        main_result.provenance["noise_stream_sha256"]
        == ablation_result.provenance["noise_stream_sha256"]
    )
    aligned = bool(np.array_equal(main_result.cycles, ablation_result.cycles))
    if not (same_initial and same_noise and aligned):
        raise RuntimeError("parameter ablation failed common-random-number contract")
    state_delta = ablation_result.state_frames - main_result.state_frames
    velocity_delta = ablation_result.velocity_frames - main_result.velocity_frames
    main_root = pair_root / "main"
    ablation_root = pair_root / "ablation"
    main_summary = _write_analyzed_run(
        main_root,
        main_result,
        campaign,
        lane=lane,
        level=level,
        seed=seed,
        quench_cycles=quench_cycles,
        branch="main",
    )
    ablation_summary = _write_analyzed_run(
        ablation_root,
        ablation_result,
        campaign,
        lane=lane,
        level=level,
        seed=seed,
        quench_cycles=quench_cycles,
        branch="ablation",
    )
    rows = [
        {
            "frame_index": index,
            "cycle": int(cycle),
            "state_delta_l1": float(np.sum(np.abs(state_delta[index]))),
            "state_delta_l2": float(np.linalg.norm(state_delta[index])),
            "state_delta_nonzero_count": int(np.count_nonzero(state_delta[index])),
            "velocity_delta_l2": float(np.linalg.norm(velocity_delta[index])),
        }
        for index, cycle in enumerate(main_result.cycles)
    ]
    receipt = {
        "schema": "oph_coupled_parameter_ablation_counterfactual_v1",
        "classification": CLASSIFICATION,
        "same_initial_state": same_initial,
        "same_process_noise_draws": same_noise,
        "aligned_snapshot_cycles": aligned,
        "only_declared_parameter_overrides": dict(overrides),
        "maximum_state_delta_l2": float(
            max((row["state_delta_l2"] for row in rows), default=0.0)
        ),
        "rows": rows,
        "physical_interpretation_allowed": False,
    }
    _write_json(pair_root / "parameter_ablation_receipt.json", receipt)
    _write_rows(pair_root / "parameter_ablation_timeseries.csv", rows)
    np.savez_compressed(
        pair_root / "parameter_ablation_delta_frames.npz",
        cycles=main_result.cycles,
        state_delta_frames=state_delta.astype(np.float32),
        velocity_delta_frames=velocity_delta.astype(np.float32),
    )
    ablation_summary["parameter_ablation_receipt"] = receipt
    summary = {
        "lane": lane,
        "level": level,
        "seed": seed,
        "quench_cycles": quench_cycles,
        "main": main_summary,
        "ablation": ablation_summary,
        "parameter_ablation_receipt": receipt,
        "run_index_rows": [
            _run_index_row(
                root,
                main_root,
                lane,
                level,
                seed,
                quench_cycles,
                "main",
                main_summary,
            ),
            _run_index_row(
                root,
                ablation_root,
                lane,
                level,
                seed,
                quench_cycles,
                "ablation",
                ablation_summary,
            ),
        ],
    }
    del main_result, ablation_result, state_delta, velocity_delta
    gc.collect()
    return summary


def _run_collision_cell(
    root: Path,
    campaign: Mapping[str, Any],
    *,
    lane: str,
    level: int,
    seed: int,
    quench_cycles: int,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cell_id = _cell_id(lane, level, seed, quench_cycles, "collision")
    collision_root = root / "cells" / cell_id
    points, left, right = _campaign_graph_arrays(campaign, level)
    kernel_config = _kernel_config(
        campaign,
        seed=seed,
        quench_cycles=quench_cycles,
        overrides=overrides,
    )
    event_a, event_b = _collision_interventions(
        campaign, points, left, right, kernel_config
    )
    collision = run_collision_counterfactual(
        points, left, right, kernel_config, event_a, event_b
    )
    all_arms = {
        "baseline": collision.baseline,
        "impulse_a": collision.a,
        "impulse_b": collision.b,
        "impulses_ab": collision.ab,
    }
    declared_branches = campaign["campaign"]["collision_probe"]["branches"]
    arms = {str(branch): all_arms[str(branch)] for branch in declared_branches}
    summaries: dict[str, dict[str, Any]] = {}
    index_rows: list[dict[str, Any]] = []
    for branch, result in arms.items():
        branch_root = collision_root / branch
        delta = (
            None
            if branch == "baseline"
            else result.state_frames - collision.baseline.state_frames
        )
        summary = _write_analyzed_run(
            branch_root,
            result,
            campaign,
            lane=lane,
            level=level,
            seed=seed,
            quench_cycles=quench_cycles,
            branch=branch,
            paired_delta=delta,
        )
        summaries[branch] = summary
        index_rows.append(
            _run_index_row(
                root,
                branch_root,
                lane,
                level,
                seed,
                quench_cycles,
                branch,
                summary,
            )
        )
    residual_receipt = _collision_residual_receipt(
        collision, event_a, event_b
    )
    _write_json(collision_root / "collision_counterfactual_receipt.json", dict(collision.receipt))
    _write_json(collision_root / "collision_residual_report.json", residual_receipt)
    _write_rows(
        collision_root / "collision_residual_timeseries.csv",
        residual_receipt["rows"],
    )
    np.savez_compressed(
        collision_root / "collision_nonlinear_residual_frames.npz",
        cycles=collision.baseline.cycles,
        state_nonlinear_residual_frames=(
            collision.state_nonlinear_residual_frames.astype(np.float32)
        ),
        velocity_nonlinear_residual_frames=(
            collision.velocity_nonlinear_residual_frames.astype(np.float32)
        ),
        record_nonlinear_residual_frames=(
            collision.record_nonlinear_residual_frames.astype(np.float32)
        ),
        defect_nonlinear_residual_frames=(
            collision.defect_nonlinear_residual_frames.astype(np.int8)
        ),
    )
    summary = {
        "lane": lane,
        "level": level,
        "seed": seed,
        "quench_cycles": quench_cycles,
        "finite_state_domain": {
            "state_bound": float(kernel_config.state_bound),
            "state_levels": int(kernel_config.state_levels),
            "state_grid_spacing": float(
                2.0 * kernel_config.state_bound / (kernel_config.state_levels - 1)
            ),
        },
        "counterfactual_receipt": dict(collision.receipt),
        "event_a": asdict(event_a),
        "event_b": asdict(event_b),
        "residual": residual_receipt,
        "arms": summaries,
        "run_index_rows": index_rows,
    }
    del collision
    gc.collect()
    return summary


def _write_analyzed_run(
    out: Path,
    result: CoupledPatchResult,
    campaign: Mapping[str, Any],
    *,
    lane: str,
    level: int,
    seed: int,
    quench_cycles: int,
    branch: str,
    paired_delta: np.ndarray | None = None,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    write_coupled_patch_run(result, out)
    diagnostics = campaign["diagnostics"]
    stored_frame_limit = int(diagnostics["stored_frame_limit"])
    if result.cycles.size > stored_frame_limit:
        raise ValueError(
            f"run retained {result.cycles.size} frames, exceeding the declared "
            f"stored_frame_limit={stored_frame_limit}"
        )
    if result.config["snapshot_stride"] == 1 and result.cycles.size != int(
        result.config["cycles"]
    ) + 1:
        raise RuntimeError("full-cycle history contract was not satisfied")
    spectral_modes = int(diagnostics["graph_spectrum_modes"])
    threshold_count = max(3, int(diagnostics["persistence_threshold_count"]))
    thresholds = tuple(
        float(value)
        for value in np.linspace(0.05, 0.95, threshold_count)
    )
    structural = structural_observables_report(
        result.points,
        result.left,
        result.right,
        result.cycles,
        result.state_frames,
        velocity_frames=result.velocity_frames,
        commit_frames=result.commit_frames,
        defect_frames=result.defect_frames,
        paired_delta_frames=paired_delta,
        quench_values=result.mass2_frames,
        thresholds=thresholds,
        threshold_mode="quantile",
        max_graph_hops=int(diagnostics["graph_shell_max_hops"]),
        anchor_count=min(
            int(diagnostics["correlation_anchor_count"]),
            int(result.points.shape[0]),
        ),
        spectral_modes=spectral_modes,
        diffusion_steps=_diffusion_steps(
            int(diagnostics["random_walk_max_steps"])
        ),
        random_walk_probe_count=int(diagnostics["random_walk_probe_count"]),
        correlation_null_draws=min(16, int(diagnostics["control_draws"])),
        seed=seed + 101,
    )
    structural_dir = out / "diagnostics" / "structural"
    structural_dir.mkdir(parents=True)
    _write_json(
        structural_dir / "structural_observables_report.json", structural
    )
    structural_tables = _export_report_tables(structural_dir / "tables", structural)

    observer_dir = out / "diagnostics" / "observer_excitation"
    intervention_contract: dict[str, Any] = {}
    if paired_delta is not None and result.intervention_cycles.size:
        unique_cycles = np.unique(result.intervention_cycles)
        if unique_cycles.size == 1:
            intervention_contract = {
                "intervention_origin_cycle": float(unique_cycles[0]),
                "intervention_origin_mask": np.any(
                    result.intervention_masks, axis=0
                ),
                "locality_hops_per_cycle": 1.0,
            }
    observer = write_observer_excitation_observables(
        observer_dir,
        points=result.points,
        left=result.left,
        right=result.right,
        cycles=result.cycles,
        state_frames=result.state_frames,
        velocity_frames=result.velocity_frames,
        record_frames=result.record_frames,
        commit_frames=result.commit_frames,
        defect_frames=result.defect_frames,
        intervention_delta=paired_delta,
        **intervention_contract,
        entropy_bins=8,
        excitation_threshold=float(diagnostics["excitation_threshold"]),
        excitation_min_lifetime_frames=int(
            diagnostics["excitation_min_lifetime_frames"]
        ),
        latent_neighborhood_anchor_count=int(
            diagnostics["latent_neighborhood_anchor_count"]
        ),
        max_graph_hops=int(diagnostics["observer_graph_radius"]),
        seed=seed + 211,
    )

    frame_rows = _frame_summary_rows(result, paired_delta)
    _write_rows(out / "frame_summary.csv", frame_rows)
    _write_visualization_pack(out / "visualization", result, paired_delta)
    summary = _run_summary(
        result,
        structural,
        observer,
        lane=lane,
        level=level,
        seed=seed,
        quench_cycles=quench_cycles,
        branch=branch,
        structural_table_count=structural_tables,
    )
    _write_json(out / "run_analysis_summary.json", summary)
    return summary


def _kernel_config(
    campaign: Mapping[str, Any],
    *,
    seed: int,
    quench_cycles: int,
    cycles: int | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> CoupledPatchConfig:
    model = dict(campaign["model"])
    cycle_count = int(cycles if cycles is not None else model.pop("cycles"))
    model.pop("cycles", None)
    snapshot_stride = int(model.pop("frame_stride"))
    default_quench_cycles = int(model.pop("quench_cycles"))
    selected_quench_cycles = int(quench_cycles or default_quench_cycles)
    quench_start_cycle = int(model.pop("quench_start_cycle"))
    mass2_start = float(model.pop("mass_squared_start"))
    mass2_end = float(model.pop("mass_squared_end"))
    values: dict[str, Any] = {
        **model,
        "cycles": cycle_count,
        "snapshot_stride": snapshot_stride,
        "seed": int(seed),
        "mass2_start": mass2_start,
        "mass2_end": mass2_end,
        "quench_start_fraction": float(quench_start_cycle / cycle_count),
        "quench_end_fraction": float(
            min(cycle_count, quench_start_cycle + selected_quench_cycles)
            / cycle_count
        ),
        "quench_kind": "smoothstep",
    }
    if overrides:
        values.update(dict(overrides))
    allowed = {field.name for field in fields(CoupledPatchConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unconsumed coupled model configuration keys: {unknown}")
    return CoupledPatchConfig(**values)


def _campaign_graph_arrays(
    campaign: Mapping[str, Any], level: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    graph = campaign["graph"]
    if graph["family"] != "icosahedral_tower":
        raise ValueError("only the icosahedral_tower graph family is supported")
    return geodesic_icosahedral_patch_arrays(
        int(level), patch_basis=str(graph["patch_basis"])
    )


def _select_center(
    points: np.ndarray,
    policy: str,
    *,
    candidates: np.ndarray | None = None,
) -> int:
    choices = (
        np.arange(points.shape[0], dtype=np.int64)
        if candidates is None
        else np.asarray(candidates, dtype=np.int64)
    )
    if not choices.size:
        raise ValueError("center policy received an empty candidate set")
    if policy == "deterministic_max_positive_z":
        values = points[choices, 2]
        target = float(np.max(values))
        return int(np.min(choices[np.isclose(values, target)]))
    if policy == "deterministic_min_z_on_exact_distance_shell":
        values = points[choices, 2]
        target = float(np.min(values))
        return int(np.min(choices[np.isclose(values, target)]))
    raise ValueError(f"unsupported deterministic center policy: {policy!r}")


def _primary_intervention(
    campaign: Mapping[str, Any],
    points: np.ndarray,
    config: CoupledPatchConfig,
    quench_cycles: int,
) -> LocalizedIntervention:
    spec = campaign["intervention"]
    quench_end = int(campaign["model"]["quench_start_cycle"]) + int(
        quench_cycles
    )
    cycle = min(
        int(spec["cycle"]),
        max(quench_end + 4, int(config.cycles * 0.60)),
        config.cycles - 2,
    )
    return LocalizedIntervention(
        center_node=_select_center(points, str(spec["center_policy"])),
        cycle=int(cycle),
        radius_hops=int(spec["graph_radius"]),
        velocity_delta=float(spec["velocity_impulse"]),
    )


def _collision_interventions(
    campaign: Mapping[str, Any],
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    config: CoupledPatchConfig,
) -> tuple[LocalizedIntervention, LocalizedIntervention]:
    spec = campaign["intervention"]
    pair_spec = spec["collision_pair"]
    amplitude = float(pair_spec["velocity_impulse"])
    center_a = _select_center(
        points, str(pair_spec["first_center_policy"])
    )
    distance = _graph_distances(points.shape[0], left, right, [center_a])
    target_hops = int(pair_spec["separation_hops"])
    candidates = np.flatnonzero(distance == target_hops)
    if not candidates.size:
        raise ValueError(
            f"collision separation {target_hops} hops is absent from this graph"
        )
    center_b = _select_center(
        points,
        str(pair_spec["second_center_policy"]),
        candidates=candidates,
    )
    cycle = min(int(pair_spec["cycle"]), config.cycles - 2)
    return (
        LocalizedIntervention(
            center_node=center_a,
            cycle=cycle,
            radius_hops=int(pair_spec["graph_radius"]),
            velocity_delta=amplitude,
        ),
        LocalizedIntervention(
            center_node=center_b,
            cycle=cycle,
            radius_hops=int(pair_spec["graph_radius"]),
            velocity_delta=-amplitude,
        ),
    )


def _causal_front_receipt(pair: Any, event: LocalizedIntervention) -> dict[str, Any]:
    distance = _graph_distances(
        pair.control.points.shape[0],
        pair.control.left,
        pair.control.right,
        np.flatnonzero(pair.intervened.intervention_mask),
    )
    rows: list[dict[str, Any]] = []
    leak_count = 0
    pre_equal = True
    for index, cycle_value in enumerate(pair.control.cycles):
        cycle = int(cycle_value)
        delta = np.abs(pair.state_delta_frames[index])
        active = delta > 0.0
        if cycle <= event.cycle and np.any(active):
            pre_equal = False
        allowed_hops = max(-1, cycle - event.cycle)
        leaks = active & (distance > allowed_hops)
        leak_count += int(np.count_nonzero(leaks))
        reached = distance[active]
        rows.append(
            {
                "frame_index": index,
                "cycle": cycle,
                "allowed_radius_hops": allowed_hops,
                "affected_node_count": int(np.count_nonzero(active)),
                "maximum_affected_radius_hops": (
                    int(np.max(reached)) if reached.size else None
                ),
                "outside_exact_graph_cone_count": int(np.count_nonzero(leaks)),
                "state_delta_l1": float(np.sum(delta)),
                "state_delta_l2": float(np.linalg.norm(delta)),
            }
        )
    same_noise = bool(pair.receipt["same_process_noise_draws"])
    same_initial = bool(pair.receipt["same_initial_state"])
    effective = bool(
        pair.receipt.get("all_interventions_effective_after_quantization", False)
    )
    return {
        "schema": "oph_coupled_patch_internal_causal_front_v1",
        "classification": CLASSIFICATION,
        "same_seed_noise": same_noise,
        "same_initial_state": same_initial,
        "all_interventions_effective_after_quantization": effective,
        "pre_intervention_histories_equal": pre_equal,
        "localized_intervention_only_declared_difference": True,
        "outside_exact_one_hop_per_update_cone_count": leak_count,
        "finite_graph_causal_contract_pass": bool(
            same_noise and same_initial and effective and pre_equal and leak_count == 0
        ),
        "rows": rows,
        "physical_light_cone_or_speed_claim": False,
        "units": "graph_hops_per_internal_update_cycle",
    }


def _collision_residual_receipt(
    collision: Any,
    event_a: LocalizedIntervention,
    event_b: LocalizedIntervention,
) -> dict[str, Any]:
    distance_a = _graph_distances(
        collision.baseline.points.shape[0],
        collision.baseline.left,
        collision.baseline.right,
        np.flatnonzero(collision.a.intervention_mask),
    )
    support_b = np.flatnonzero(collision.b.intervention_mask)
    separation = int(np.min(distance_a[support_b]))
    earliest_meeting_cycle = int(
        event_a.cycle + math.ceil(max(0, separation) / 2.0)
    )
    rows: list[dict[str, Any]] = []
    first_nonzero_cycle: int | None = None
    premature_count = 0
    post_overlap_nonzero_count = 0
    for index, cycle_value in enumerate(collision.baseline.cycles):
        cycle = int(cycle_value)
        residual = collision.state_nonlinear_residual_frames[index]
        nonzero = np.abs(residual) > 0.0
        count = int(np.count_nonzero(nonzero))
        if count and first_nonzero_cycle is None:
            first_nonzero_cycle = cycle
        if count and cycle < earliest_meeting_cycle:
            premature_count += count
        if count and cycle >= earliest_meeting_cycle:
            post_overlap_nonzero_count += count
        rows.append(
            {
                "frame_index": index,
                "cycle": cycle,
                "nonzero_state_residual_node_count": count,
                "state_residual_l1": float(np.sum(np.abs(residual))),
                "state_residual_l2": float(np.linalg.norm(residual)),
                "state_residual_linf": float(np.max(np.abs(residual))),
                "velocity_residual_l2": float(
                    np.linalg.norm(
                        collision.velocity_nonlinear_residual_frames[index]
                    )
                ),
                "record_residual_nonzero_count": int(
                    np.count_nonzero(
                        collision.record_nonlinear_residual_frames[index]
                    )
                ),
                "defect_residual_nonzero_count": int(
                    np.count_nonzero(
                        collision.defect_nonlinear_residual_frames[index]
                    )
                ),
            }
        )
    actuator = dict(collision.receipt.get("actuator_diagnostics", {}))
    actuator_pass = bool(
        actuator.get("clean_dynamical_residual_interpretation", False)
    )
    disjoint = not bool(
        actuator.get("same_cycle_requested_support_overlap", True)
    )
    return {
        "schema": "oph_coupled_patch_internal_collision_residual_v1",
        "classification": CLASSIFICATION,
        "residual_formula": "AB-A-B+baseline",
        "support_separation_hops": separation,
        "earliest_graph_cone_meeting_cycle": earliest_meeting_cycle,
        "first_nonzero_residual_cycle": first_nonzero_cycle,
        "premature_residual_node_count": premature_count,
        "locality_before_cone_intersection_pass": premature_count == 0,
        "post_overlap_nonzero_residual_count": post_overlap_nonzero_count,
        "post_overlap_nonzero_power_pass": post_overlap_nonzero_count > 0,
        "actuator_additivity": actuator,
        "actuator_additivity_pass": actuator_pass,
        "disjoint_injection_supports": disjoint,
        "controlled_internal_interaction_receipt_pass": bool(
            actuator_pass
            and disjoint
            and premature_count == 0
            and post_overlap_nonzero_count > 0
        ),
        "maximum_state_residual_l2": float(
            max((row["state_residual_l2"] for row in rows), default=0.0)
        ),
        "rows": rows,
        "physical_scattering_amplitude_or_cross_section": False,
        "interpretation": (
            "finite-map nonlinear interaction residual with a separately run "
            "linear/quantization ablation; no asymptotic particle states"
        ),
    }


def _frame_summary_rows(
    result: CoupledPatchResult, paired_delta: np.ndarray | None
) -> list[dict[str, Any]]:
    config = result.config
    node_count = result.points.shape[0]
    rows: list[dict[str, Any]] = []
    for index, cycle in enumerate(result.cycles):
        state = result.state_frames[index]
        velocity = result.velocity_frames[index]
        record = result.record_frames[index]
        committed = result.commit_frames[index]
        mass2 = float(result.mass2_frames[index])
        edge_difference = state[result.left] - state[result.right]
        kinetic = 0.5 * float(np.mean(velocity**2))
        local_potential = float(
            np.mean(
                0.5 * mass2 * state**2
                + 0.25 * float(config["quartic"]) * state**4
            )
        )
        gradient = (
            0.5
            * float(config["coupling"])
            * float(np.sum(edge_difference**2))
            / max(1, node_count)
        )
        feedback = (
            0.5
            * float(config["feedback_strength"])
            * float(np.mean(committed * (record - state) ** 2))
        )
        delta = (
            np.zeros(node_count, dtype=float)
            if paired_delta is None
            else np.asarray(paired_delta[index], dtype=float)
        )
        rows.append(
            {
                "frame_index": index,
                "cycle": int(cycle),
                "mass2": mass2,
                "state_mean": float(np.mean(state)),
                "state_abs_mean": float(np.mean(np.abs(state))),
                "state_std": float(np.std(state)),
                "velocity_rms": float(np.sqrt(np.mean(velocity**2))),
                "committed_fraction": float(np.mean(committed)),
                "signed_record_mean": float(np.mean(record)),
                "defect_edge_fraction": float(
                    np.mean(result.defect_frames[index])
                ),
                "feedback_force_rms": float(
                    np.sqrt(np.mean(result.feedback_force_frames[index] ** 2))
                ),
                "kinetic_density_proxy": kinetic,
                "local_potential_density_proxy": local_potential,
                "gradient_density_proxy": gradient,
                "record_feedback_density_proxy": feedback,
                "total_declared_energy_density_proxy": (
                    kinetic + local_potential + gradient + feedback
                ),
                "paired_delta_active_count": int(np.count_nonzero(delta)),
                "paired_delta_l2": float(np.linalg.norm(delta)),
            }
        )
    return rows


def _write_visualization_pack(
    out: Path,
    result: CoupledPatchResult,
    paired_delta: np.ndarray | None,
) -> None:
    out.mkdir(parents=True)
    max_nodes = 4096
    max_frames = 49
    node_indices = _even_indices(result.points.shape[0], max_nodes)
    frame_indices = _even_indices(result.cycles.size, max_frames)
    incident_defect = np.zeros(
        (result.cycles.size, result.points.shape[0]), dtype=np.float32
    )
    degree = np.bincount(
        np.concatenate((result.left, result.right)),
        minlength=result.points.shape[0],
    ).astype(float)
    for frame in range(result.cycles.size):
        weights = result.defect_frames[frame].astype(float)
        incident = np.bincount(
            result.left, weights=weights, minlength=result.points.shape[0]
        )
        incident += np.bincount(
            result.right, weights=weights, minlength=result.points.shape[0]
        )
        incident_defect[frame] = np.divide(
            incident,
            np.maximum(degree, 1.0),
        ).astype(np.float32)
    selected_delta = (
        np.zeros((frame_indices.size, node_indices.size), dtype=np.float32)
        if paired_delta is None
        else np.asarray(paired_delta)[np.ix_(frame_indices, node_indices)].astype(
            np.float32
        )
    )
    np.savez_compressed(
        out / "visualization_arrays.npz",
        source_node_indices=node_indices.astype(np.int32),
        source_frame_indices=frame_indices.astype(np.int32),
        points=result.points[node_indices].astype(np.float32),
        cycles=result.cycles[frame_indices].astype(np.int32),
        state_frames=result.state_frames[np.ix_(frame_indices, node_indices)].astype(
            np.float32
        ),
        velocity_frames=result.velocity_frames[
            np.ix_(frame_indices, node_indices)
        ].astype(np.float32),
        record_frames=result.record_frames[
            np.ix_(frame_indices, node_indices)
        ].astype(np.float32),
        commit_frames=result.commit_frames[
            np.ix_(frame_indices, node_indices)
        ],
        incident_defect_fraction_frames=incident_defect[
            np.ix_(frame_indices, node_indices)
        ],
        feedback_force_frames=result.feedback_force_frames[
            np.ix_(frame_indices, node_indices)
        ].astype(np.float32),
        paired_state_delta_frames=selected_delta,
    )
    manifest = {
        "schema": "oph_coupled_internal_visualization_pack_v1",
        "classification": CLASSIFICATION,
        "artifact": "visualization_arrays.npz",
        "source_node_count": int(result.points.shape[0]),
        "visualized_node_count": int(node_indices.size),
        "source_frame_count": int(result.cycles.size),
        "visualized_frame_count": int(frame_indices.size),
        "node_sampling": "deterministic_even_index_with_endpoints",
        "frame_sampling": "deterministic_even_index_with_endpoints",
        "full_lossless_history": "../coupled_patch_frames.npz",
        "fields": {
            "state_frames": "signed bounded patch order parameter",
            "velocity_frames": "bounded conjugate update coordinate",
            "record_frames": "signed committed observer record",
            "commit_frames": "observer-record completion bit",
            "incident_defect_fraction_frames": (
                "fraction of incident seams above the declared state-difference threshold"
            ),
            "feedback_force_frames": "later local write force read from committed records",
            "paired_state_delta_frames": "same-seed intervention minus control",
        },
        "units": "dimensionless internal cycles, graph coordinates, and field values",
        "physical_visualization_claim": False,
    }
    _write_json(out / "visualization_manifest.json", manifest)
    (out / "README.md").write_text(
        "# Visualization pack\n\n"
        "Load `visualization_arrays.npz` with NumPy. Rows in `points` match "
        "the second dimension of every field frame; `cycles` matches the first. "
        "The complete unsampled history remains one directory above. Values use "
        "internal graph/update units and carry no physical cosmology mapping.\n",
        encoding="utf-8",
    )


def _run_summary(
    result: CoupledPatchResult,
    structural: Mapping[str, Any],
    observer: Mapping[str, Any],
    *,
    lane: str,
    level: int,
    seed: int,
    quench_cycles: int,
    branch: str,
    structural_table_count: int,
) -> dict[str, Any]:
    final_state = result.state_frames[-1]
    phase = structural["phase_transition_proxies"]
    carrier = structural["carrier_geometry"]
    correlation = structural["field_correlation_horizon"]
    phase_rows = [dict(row) for row in phase.get("rows", [])]
    if len(phase_rows) != int(result.cycles.size):
        raise RuntimeError(
            "phase-transition rows do not cover every retained simulator frame"
        )
    if result.mass2_frames.shape != result.cycles.shape:
        raise RuntimeError("mass2 history does not align with retained cycles")
    for index, row in enumerate(phase_rows):
        if not math.isclose(
            float(row["cycle"]),
            float(result.cycles[index]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError("phase-transition row cycle is misaligned")
        row["mass2"] = float(result.mass2_frames[index])
    return {
        "schema": "oph_coupled_internal_run_analysis_summary_v1",
        "classification": CLASSIFICATION,
        "lane": lane,
        "branch": branch,
        "refinement_level": int(level),
        "node_count": int(result.points.shape[0]),
        "edge_count": int(result.left.size),
        "seed": int(seed),
        "quench_cycles": int(quench_cycles),
        "frame_count": int(result.cycles.size),
        "final_cycle": int(result.cycles[-1]),
        "final_state_mean": float(np.mean(final_state)),
        "final_abs_order_parameter": float(np.mean(np.abs(final_state))),
        "final_state_variance": float(np.var(final_state)),
        "final_defect_edge_fraction": float(np.mean(result.defect_frames[-1])),
        "defect_fraction_time_rows": [
            {
                "cycle": int(cycle),
                "defect_edge_fraction": float(np.mean(result.defect_frames[index])),
            }
            for index, cycle in enumerate(result.cycles)
        ],
        "final_committed_fraction": float(np.mean(result.commit_frames[-1])),
        "final_null_excess_correlation_extent_hops": _latest_horizon(correlation),
        "carrier_spectral_dimension_curve": carrier.get(
            "spectral_dimension_curve", []
        ),
        "phase_time_rows": phase_rows,
        "structural_table_count": int(structural_table_count),
        "observer_information_available": bool(
            observer.get("information_dynamics", {}).get("available", True)
        ),
        "excitation_track_count": int(
            observer.get("localized_excitations", {}).get("track_count", 0)
        ),
        "scattering_candidate_count": int(
            observer.get("candidate_scattering_channels", {}).get(
                "encounter_count", 0
            )
        ),
        "candidate_family_status": observer.get(
            "candidate_family_clustering", {}
        ).get("available", False),
        "zero_state_and_velocity_clipping": bool(
            result.provenance["numerical_checks"].get("state_clip_count", 0) == 0
            and result.provenance["numerical_checks"].get("velocity_clip_count", 0) == 0
        ),
        "record_irreversibility_is_imposed": True,
        "defects_are_gradient_threshold_edges": True,
        "target_data_read": False,
        "physical_identification": False,
    }


def _latest_horizon(report: Mapping[str, Any]) -> int | None:
    rows = report.get("frame_rows") or report.get("rows") or []
    if not rows:
        return None
    last_cycle = max(float(row.get("cycle", 0.0)) for row in rows)
    horizons = [
        row.get("largest_null_excess_threshold_crossing_hop")
        for row in rows
        if float(row.get("cycle", 0.0)) == last_cycle
        and row.get("largest_null_excess_threshold_crossing_hop") is not None
    ]
    return int(horizons[-1]) if horizons else None


def _nearest_defect_fraction(
    summary: Mapping[str, Any], cycle: float
) -> float:
    rows = list(summary.get("defect_fraction_time_rows", []))
    if not rows:
        raise ValueError("defect time rows are required for cycle-matched aggregation")
    selected = min(rows, key=lambda row: abs(float(row["cycle"]) - cycle))
    return float(selected["defect_edge_fraction"])


def _bootstrap_mean_ci(
    values: np.ndarray, *, seed: int, draws: int = 4096
) -> list[float] | None:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, array.size), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return [float(lower), float(upper)]


def _write_campaign_aggregates(
    root: Path,
    campaign: Mapping[str, Any],
    *,
    stage: str,
    baseline_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    collision_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    aggregate_root = root / "aggregate"
    flat_baselines = [_compact_summary(row) for row in baseline_rows]
    _write_rows(aggregate_root / "baseline_run_matrix.csv", flat_baselines)

    quench = _quench_scaling_report(campaign, baseline_rows)
    refinement = _refinement_report(baseline_rows)
    causal = _paired_aggregate_report(pair_rows)
    collisions = _collision_aggregate_report(collision_rows)
    feedback = _record_feedback_ablation_report(baseline_rows)
    _write_json(aggregate_root / "quench_scaling_report.json", quench)
    _write_json(aggregate_root / "refinement_report.json", refinement)
    _write_json(aggregate_root / "causal_intervention_report.json", causal)
    _write_json(aggregate_root / "collision_control_report.json", collisions)
    _write_json(aggregate_root / "record_feedback_ablation_report.json", feedback)
    _write_rows(
        aggregate_root / "quench_scaling_rows.csv", quench.get("rows", [])
    )
    _write_rows(
        aggregate_root / "refinement_rows.csv", refinement.get("rows", [])
    )
    return {
        "stage": stage,
        "baseline_run_count": len(baseline_rows),
        "paired_experiment_count": len(pair_rows),
        "collision_experiment_count": len(collision_rows),
        "quench_scaling": quench,
        "refinement": refinement,
        "causal_interventions": causal,
        "collision_controls": collisions,
        "record_feedback_ablation": feedback,
    }


def _quench_scaling_report(
    campaign: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    lane_rows = [
        row for row in rows if row.get("lane") == "seed_and_quench_matrix"
    ]
    if not lane_rows:
        return {
            "available": False,
            "reason": "seed_and_quench_matrix_not_run",
            "physical_critical_exponent_claim": False,
            "rows": [],
        }
    start = int(campaign["model"]["quench_start_cycle"])
    offset = 24
    declared_mass2_end = campaign["model"].get("mass_squared_end")
    measurement_rows: list[dict[str, Any]] = []
    for row in lane_rows:
        phase_rows = list(row.get("phase_time_rows", []))
        realized_end_cycle: float | None = None
        if declared_mass2_end is not None:
            endpoint_rows = [
                value
                for value in phase_rows
                if value.get("mass2") is not None
                and math.isclose(
                    float(value["mass2"]),
                    float(declared_mass2_end),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ]
            if endpoint_rows:
                realized_end_cycle = float(
                    min(float(value["cycle"]) for value in endpoint_rows)
                )
        nominal_end_cycle = start + int(row["quench_cycles"])
        target_cycle = (
            realized_end_cycle + offset
            if realized_end_cycle is not None
            else nominal_end_cycle + offset
        )
        selected = min(
            phase_rows,
            key=lambda value: abs(float(value["cycle"]) - target_cycle),
        )
        measurement_rows.append(
            {
                "seed": int(row["seed"]),
                "quench_cycles": int(row["quench_cycles"]),
                "target_offset_after_quench_cycles": offset,
                "nominal_quench_end_cycle": int(nominal_end_cycle),
                "realized_quench_end_cycle": realized_end_cycle,
                "realized_quench_endpoint_resolved": realized_end_cycle is not None,
                "sampled_cycle": float(selected["cycle"]),
                "node_count": int(row["node_count"]),
                "global_spatial_mean_order_parameter": float(
                    selected.get(
                        "spatial_state_mean",
                        selected.get("spatial_mean_order_parameter", 0.0),
                    )
                ),
                "spatial_site_variance": float(
                    selected.get(
                        "spatial_site_variance",
                        selected.get("spatial_variance", 0.0),
                    )
                ),
                "spatial_fluctuation_shape_u4": selected.get(
                    "spatial_fluctuation_shape_u4",
                    selected.get(
                        "spatial_centered_fourth_shape",
                        selected.get("spatial_binder_cumulant_proxy"),
                    ),
                ),
                "defect_edge_fraction_at_sampled_cycle": _nearest_defect_fraction(
                    row, float(selected["cycle"])
                ),
                "null_excess_correlation_extent_hops_final": row.get(
                    "final_null_excess_correlation_extent_hops"
                ),
            }
        )
    grouped: list[dict[str, Any]] = []
    for duration in sorted({row["quench_cycles"] for row in measurement_rows}):
        group = [row for row in measurement_rows if row["quench_cycles"] == duration]
        magnetization = np.asarray(
            [row["global_spatial_mean_order_parameter"] for row in group],
            dtype=float,
        )
        defect = np.asarray(
            [row["defect_edge_fraction_at_sampled_cycle"] for row in group],
            dtype=float,
        )
        m2 = float(np.mean(magnetization**2))
        binder = (
            None
            if m2 <= 0.0
            else float(1.0 - np.mean(magnetization**4) / (3.0 * m2 * m2))
        )
        grouped.append(
            {
                "quench_cycles": int(duration),
                "seed_count": len(group),
                "mean_defect_edge_fraction": float(np.mean(defect)),
                "std_defect_edge_fraction": float(np.std(defect, ddof=1))
                if defect.size > 1
                else 0.0,
                "mean_abs_global_order_parameter": float(
                    np.mean(np.abs(magnetization))
                ),
                "ensemble_global_order_variance_times_n": float(
                    group[0]["node_count"] * np.var(magnetization)
                ),
                "ensemble_global_order_binder_cumulant": binder,
                "mean_defect_fraction_95pct_bootstrap_ci": _bootstrap_mean_ci(
                    defect,
                    seed=20260827 + int(duration),
                ),
                "ensemble_estimate_is_exploratory": True,
            }
        )
    positive = [
        row
        for row in grouped
        if row["mean_defect_edge_fraction"] > 0.0
        and row["quench_cycles"] > 0
    ]
    fit = _loglog_fit(
        [float(row["quench_cycles"]) for row in positive],
        [float(row["mean_defect_edge_fraction"]) for row in positive],
    )
    return {
        "available": len(grouped) >= 3,
        "classification": "FINITE_INTERNAL_QUENCH_RATE_DIAGNOSTIC",
        "evaluation_rule": (
            "nearest retained cycle 24 updates after the first saved frame at "
            "the declared final mass-squared value, falling back to the nominal "
            "end only when the retained rows cannot resolve that endpoint; state "
            "and defect quantities use the same cycle"
        ),
        "rows": measurement_rows,
        "grouped_rows": grouped,
        "defect_density_loglog_fit": fit,
        "kibble_zurek_exponent_identified": False,
        "reason": (
            "the slope is a declared-model finite-size diagnostic; physical Kibble-Zurek "
            "identification requires a physical clock, length, control parameter, and "
            "independently validated universality class"
        ),
    }


def _refinement_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("lane") == "refinement_ladder"]
    compact = [_compact_summary(row) for row in selected]
    compact.sort(key=lambda row: int(row["refinement_level"]))
    return {
        "available": len(compact) >= 2,
        "classification": "TRUE_ICOSAHEDRAL_REGULATOR_REFINEMENT_DIAGNOSTIC",
        "rows": compact,
        "carrier_geometry_evolves": False,
        "continuum_or_physical_limit_established": False,
        "interpretation": (
            "compares the same declared internal model across exact nested cell counts; "
            "it does not derive the carrier or a physical continuum"
        ),
    }


def _paired_aggregate_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    receipts = [row["causal_receipt"] for row in rows]
    return {
        "available": bool(receipts),
        "pair_count": len(receipts),
        "all_same_seed_noise": bool(
            receipts and all(row["same_seed_noise"] for row in receipts)
        ),
        "all_pre_intervention_histories_equal": bool(
            receipts
            and all(row["pre_intervention_histories_equal"] for row in receipts)
        ),
        "total_outside_graph_cone_count": int(
            sum(row["outside_exact_one_hop_per_update_cone_count"] for row in receipts)
        ),
        "all_finite_graph_causal_contracts_pass": bool(
            receipts
            and all(row["finite_graph_causal_contract_pass"] for row in receipts)
        ),
        "physical_causal_speed_claim": False,
    }


def _collision_numerical_gate(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    run_rows = list(row.get("run_index_rows", []))
    return bool(
        run_rows
        and all(
            item.get("zero_state_and_velocity_clipping") is True
            for item in run_rows
        )
    )


def _collision_packet_gate(row: Mapping[str, Any] | None) -> bool:
    if row is None or not _collision_numerical_gate(row):
        return False
    receipt = row.get("counterfactual_receipt", {})
    residual = row.get("residual", {})
    actuator = receipt.get("actuator_diagnostics", {})
    return bool(
        receipt.get("same_initial_state") is True
        and receipt.get("same_process_noise_draws") is True
        and receipt.get("aligned_snapshot_cycles") is True
        and actuator.get("clean_dynamical_residual_interpretation") is True
        and residual.get("actuator_additivity_pass") is True
        and residual.get("disjoint_injection_supports") is True
        and residual.get("locality_before_cone_intersection_pass") is True
    )


def _collision_aggregate_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {"available": False, "reason": "collision_probe_not_run"}
    main = next((row for row in rows if row["lane"] == "collision_probe"), None)
    linear = next(
        (
            row
            for row in rows
            if row["lane"] == "ablation_quantized_no_quartic_no_feedback"
        ),
        None,
    )
    main_peak = (
        float(main["residual"]["maximum_state_residual_l2"])
        if main is not None
        else None
    )
    linear_peak = (
        float(linear["residual"]["maximum_state_residual_l2"])
        if linear is not None
        else None
    )
    ratio = (
        main_peak / linear_peak
        if main_peak is not None and linear_peak is not None and linear_peak > 0.0
        else None
    )
    main_numerical = _collision_numerical_gate(main)
    linear_numerical = _collision_numerical_gate(linear)
    main_packet = _collision_packet_gate(main)
    linear_packet = _collision_packet_gate(linear)
    same_finite_state_domain = bool(
        main is not None
        and linear is not None
        and main.get("finite_state_domain") == linear.get("finite_state_domain")
    )
    quantized_linear_null_available = bool(
        linear_packet and same_finite_state_domain
    )
    nonzero_linear_residual = bool(
        quantized_linear_null_available
        and linear_peak is not None
        and linear_peak > 0.0
    )
    return {
        "available": main is not None,
        "main_nonlinear_residual_l2_peak": main_peak,
        "quantized_no_quartic_no_feedback_residual_l2_peak": linear_peak,
        "main_to_declared_ablation_peak_ratio": ratio,
        "main_all_arms_zero_state_and_velocity_clipping": main_numerical,
        "declared_ablation_all_arms_zero_state_and_velocity_clipping": linear_numerical,
        "main_collision_packet_gate_pass": main_packet,
        "declared_ablation_collision_packet_gate_pass": linear_packet,
        "same_finite_state_domain": same_finite_state_domain,
        "quantized_linear_null_available": quantized_linear_null_available,
        "quantized_linear_control_residual_nonzero": nonzero_linear_residual,
        "nonzero_residual_survives_quantized_linear_ablation": nonzero_linear_residual,
        "quartic_or_record_feedback_attribution_available": False,
        "attribution_rule": (
            "Term attribution is unavailable without a same-grid factorial "
            "design. A nonzero residual in the accepted quantized linear "
            "control establishes that finite-grid quantization alone can "
            "produce AB-A-B+baseline residual power."
        ),
        "main_locality_before_cone_intersection_pass": (
            main["residual"]["locality_before_cone_intersection_pass"]
            if main is not None
            else None
        ),
        "linear_locality_before_cone_intersection_pass": (
            linear["residual"]["locality_before_cone_intersection_pass"]
            if linear is not None
            else None
        ),
        "physical_scattering_claim": False,
        "cross_section_or_s_matrix_available": False,
    }


def _record_feedback_ablation_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ablation = next(
        (row for row in rows if row.get("lane") == "ablation_no_record_feedback"),
        None,
    )
    reference = next(
        (
            row
            for row in rows
            if row.get("lane") == "seed_and_quench_matrix"
            and row.get("seed") == (ablation or {}).get("seed")
            and row.get("quench_cycles") == (ablation or {}).get("quench_cycles")
        ),
        None,
    )
    receipt = (ablation or {}).get("parameter_ablation_receipt")
    if ablation is None or reference is None or not receipt:
        return {
            "available": False,
            "reason": "matched main/no-feedback summaries unavailable",
            "literal_frame_difference_retained": False,
        }
    return {
        "available": True,
        "matched_seed": int(reference["seed"]),
        "matched_quench_cycles": int(reference["quench_cycles"]),
        "main_final_state_mean": float(reference["final_state_mean"]),
        "no_feedback_final_state_mean": float(ablation["final_state_mean"]),
        "main_final_defect_fraction": float(reference["final_defect_edge_fraction"]),
        "no_feedback_final_defect_fraction": float(
            ablation["final_defect_edge_fraction"]
        ),
        "same_seed_summary_comparison_only": False,
        "literal_framewise_feedback_ablation_receipt": True,
        "same_initial_state": bool(receipt["same_initial_state"]),
        "same_process_noise_draws": bool(receipt["same_process_noise_draws"]),
        "maximum_state_delta_l2": float(receipt["maximum_state_delta_l2"]),
        "parameter_ablation_timeseries": receipt["rows"],
        "reason": (
            "the only declared parameter change is feedback_strength=0; the "
            "complete common-random-number frame difference is retained"
        ),
    }


def _compact_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "lane",
        "branch",
        "refinement_level",
        "node_count",
        "edge_count",
        "seed",
        "quench_cycles",
        "frame_count",
        "final_cycle",
        "final_state_mean",
        "final_abs_order_parameter",
        "final_state_variance",
        "final_defect_edge_fraction",
        "final_committed_fraction",
        "final_null_excess_correlation_extent_hops",
        "excitation_track_count",
        "scattering_candidate_count",
    )
    return {key: row.get(key) for key in keys}


def _observable_catalog(run_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families = [
        (
            "correlation_horizons_and_intervention_fronts",
            "permutation-null correlation extent heuristics plus exact same-seed graph-cone receipts",
        ),
        (
            "spectral_and_causal_dimension_volume_growth_curvature",
            "fixed-carrier return probability/spectral dimension and graph-ball growth; dynamic geometry remains closed",
        ),
        (
            "phase_transition_and_quench_scaling",
            "spatial field moments plus exploratory multi-seed global-order Binder/variance and multi-rate defect scaling diagnostics",
        ),
        (
            "dynamic_defects",
            "edge-defect density, component persistence, motion, merge, split, creation, and annihilation diagnostics",
        ),
        (
            "clusters_voids_topology_persistence",
            "threshold components, percolation, graph Euler/cycle rank, and zero-dimensional persistence",
        ),
        (
            "spectra_higher_moments_parity_chirality_axes",
            "graph spectrum, higher moments, invariant-status gates, parity and embedding-axis diagnostics",
        ),
        (
            "information_mixing_record_arrow",
            "one-step information, autocorrelation, and consequences of the imposed irreversible record latch",
        ),
        (
            "observer_local_skies_and_causal_diamonds",
            "local variance, overlap agreement, homogeneity scale, and operational graph-cycle cone dependence",
        ),
        (
            "localized_excitations_dispersion_scattering_families",
            "controlled feature tracks, mode/IPR proxies, encounter candidates, four-arm collision residuals, and fail-closed families",
        ),
    ]
    return {
        "schema": "oph_coupled_internal_observable_catalog_v1",
        "classification": CLASSIFICATION,
        "run_count": len(run_rows),
        "families": [
            {
                "observable_family": name,
                "status": "COMPUTED_OR_EXPLICITLY_FAIL_CLOSED_IN_EACH_RUN",
                "products": description,
                "target_data_read": False,
                "physical_identification": False,
            }
            for name, description in families
        ],
    }


def _run_index_row(
    campaign_root: Path,
    run_root: Path,
    lane: str,
    level: int,
    seed: int,
    quench_cycles: int,
    branch: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_root.parent.name
        if run_root.name in {"control", "intervention", "baseline", "impulse_a", "impulse_b", "impulses_ab"}
        else run_root.name,
        "lane": lane,
        "branch": branch,
        "refinement_level": int(level),
        "node_count": int(summary["node_count"]),
        "seed": int(seed),
        "quench_cycles": int(quench_cycles),
        "relative_path": run_root.relative_to(campaign_root).as_posix(),
        "history_artifact": (
            run_root / "coupled_patch_frames.npz"
        ).relative_to(campaign_root).as_posix(),
        "analysis_summary": (
            run_root / "run_analysis_summary.json"
        ).relative_to(campaign_root).as_posix(),
        "visualization_manifest": (
            run_root / "visualization" / "visualization_manifest.json"
        ).relative_to(campaign_root).as_posix(),
        "classification": CLASSIFICATION,
        "zero_state_and_velocity_clipping": bool(
            summary["zero_state_and_velocity_clipping"]
        ),
    }


def _cell_id(
    lane: str,
    level: int,
    seed: int,
    quench_cycles: int,
    kind: str,
) -> str:
    safe_lane = "".join(character if character.isalnum() else "_" for character in lane)
    return f"{safe_lane}__L{level}__seed{seed}__q{quench_cycles}__{kind}"


def _graph_distances(
    node_count: int,
    left: np.ndarray,
    right: np.ndarray,
    sources: Iterable[int],
) -> np.ndarray:
    rows: list[list[int]] = [[] for _ in range(int(node_count))]
    for first, second in zip(left, right, strict=True):
        a = int(first)
        b = int(second)
        rows[a].append(b)
        rows[b].append(a)
    distance = np.full(node_count, -1, dtype=np.int32)
    frontier = sorted({int(source) for source in sources})
    for source in frontier:
        distance[source] = 0
    while frontier:
        following: list[int] = []
        for node in frontier:
            for neighbor in rows[node]:
                if distance[neighbor] < 0:
                    distance[neighbor] = distance[node] + 1
                    following.append(neighbor)
        frontier = following
    return distance


def _even_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.unique(
        np.rint(np.linspace(0, count - 1, limit)).astype(np.int64)
    )


def _diffusion_steps(max_step: int) -> tuple[int, ...]:
    if max_step < 1:
        raise ValueError("random_walk_max_steps must be positive")
    preferred = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    selected = [value for value in preferred if value <= max_step]
    if max_step not in selected:
        selected.append(max_step)
    return tuple(sorted(set(selected)))


def _loglog_fit(x_values: Sequence[float], y_values: Sequence[float]) -> dict[str, Any]:
    if len(x_values) < 3 or len(set(x_values)) < 3:
        return {
            "available": False,
            "reason": "at_least_three_positive_unique_scales_required",
        }
    x = np.log(np.asarray(x_values, dtype=float))
    y = np.log(np.asarray(y_values, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    total = float(np.sum((y - np.mean(y)) ** 2))
    residual = float(np.sum((y - predicted) ** 2))
    r_squared = None if total <= 0.0 else float(1.0 - residual / total)
    return {
        "available": True,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "point_count": len(x_values),
        "uncertainty_available": False,
        "reason": "three rate means permit a diagnostic slope but not a reliable exponent uncertainty",
    }


def _export_report_tables(root: Path, report: Mapping[str, Any]) -> int:
    root.mkdir(parents=True, exist_ok=True)
    count = 0

    def visit(value: Any, path: tuple[str, ...]) -> None:
        nonlocal count
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, path + (str(key),))
            return
        if (
            isinstance(value, list)
            and value
            and all(isinstance(row, Mapping) for row in value)
        ):
            name = "__".join(path[-4:]) or "rows"
            safe = "".join(
                character if character.isalnum() or character in {"_", "-"} else "_"
                for character in name
            )
            _write_rows(root / f"{safe}.csv", [dict(row) for row in value])
            count += 1

    visit(report, ())
    return count


def _artifact_index(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_index.json":
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "byte_count": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        )
    return {
        "schema": "oph_coupled_internal_campaign_artifact_index_v1",
        "root": str(root),
        "artifacts": rows,
        "total_byte_count": int(sum(row["byte_count"] for row in rows)),
    }


def _snapshot_sources(root: Path, config_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    destination = root / "source_snapshot"
    for relative in SOURCE_FILES:
        source = repository / relative
        if not source.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(config_path, destination / config_path.name)


def _runtime_receipt(
    config: Mapping[str, Any], config_path: Path, stage: str
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {
        "schema": "oph_coupled_internal_campaign_start_receipt_v1",
        "classification": CLASSIFICATION,
        "campaign_id": config["campaign_id"],
        "stage": stage,
        "config_path": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "target_data_read": False,
        "source_snapshot_written": True,
    }


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("coupled campaign config must be a mapping")
    _require_exact_keys("top level", value, TOP_LEVEL_KEYS)
    if value["classification"] != CLASSIFICATION:
        raise ValueError("coupled campaign classification must fail closed")
    if value["name"] != "coupled_internal_observables_v1":
        raise ValueError("unexpected coupled campaign name")
    if value["schema"] != "oph_coupled_internal_observables_campaign_config_v1":
        raise ValueError("unexpected coupled campaign schema")
    _validate_whole_config_contract(value)
    return value


def _require_exact_keys(
    label: str, value: Mapping[str, Any], expected: set[str]
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} key mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _validate_whole_config_contract(config: Mapping[str, Any]) -> None:
    graph = config["graph"]
    _require_exact_keys("graph", graph, {"family", "patch_basis"})
    if graph != {"family": "icosahedral_tower", "patch_basis": "cells"}:
        raise ValueError("campaign requires the icosahedral cell tower")

    _require_exact_keys("model", config["model"], MODEL_KEYS)
    if int(config["model"]["frame_stride"]) != 1:
        raise ValueError("canonical campaign requires frame_stride=1")

    intervention = config["intervention"]
    _require_exact_keys(
        "intervention",
        intervention,
        {
            "enabled",
            "cycle",
            "center_policy",
            "graph_radius",
            "velocity_impulse",
            "collision_pair",
        },
    )
    if intervention["enabled"] is not True:
        raise ValueError("localized intervention must be enabled")
    if intervention["center_policy"] != "deterministic_max_positive_z":
        raise ValueError("unsupported primary intervention center policy")
    collision_pair = intervention["collision_pair"]
    _require_exact_keys(
        "collision_pair",
        collision_pair,
        {
            "first_center_policy",
            "second_center_policy",
            "graph_radius",
            "cycle",
            "separation_hops",
            "velocity_impulse",
        },
    )
    if collision_pair["first_center_policy"] != "deterministic_max_positive_z":
        raise ValueError("unsupported first collision center policy")
    if collision_pair["second_center_policy"] != (
        "deterministic_min_z_on_exact_distance_shell"
    ):
        raise ValueError("unsupported second collision center policy")

    campaign = config["campaign"]
    _require_exact_keys(
        "campaign",
        campaign,
        {
            "smoke",
            "seed_and_quench_matrix",
            "refinement_ladder",
            "paired_intervention",
            "collision_probe",
            "ablations",
        },
    )
    section_keys = {
        "smoke": {"refinement_level", "seeds", "cycles", "quench_cycles"},
        "seed_and_quench_matrix": {
            "refinement_level",
            "seeds",
            "quench_cycles",
        },
        "refinement_ladder": {"levels", "seed", "quench_cycles"},
        "paired_intervention": {"refinement_level", "seeds", "quench_cycles"},
        "collision_probe": {
            "refinement_level",
            "seed",
            "quench_cycles",
            "branches",
        },
    }
    for name, expected in section_keys.items():
        _require_exact_keys(name, campaign[name], expected)
    if len(campaign["smoke"]["seeds"]) != 1 or len(
        campaign["smoke"]["quench_cycles"]
    ) != 1:
        raise ValueError("smoke campaign must declare exactly one seed and quench")
    if campaign["collision_probe"]["branches"] != [
        "baseline",
        "impulse_a",
        "impulse_b",
        "impulses_ab",
    ]:
        raise ValueError("collision branch contract must retain all four arms")
    expected_ablation_keys = {
        "no_graph_coupling": {"name", "coupling"},
        "no_record_feedback": {"name", "feedback_strength"},
        "quantized_no_quartic_no_feedback": {
            "name",
            "quartic",
            "feedback_strength",
        },
    }
    ablations = campaign["ablations"]
    if {str(row.get("name")) for row in ablations} != set(expected_ablation_keys):
        raise ValueError("campaign must declare the exact three control ablations")
    for row in ablations:
        name = str(row["name"])
        _require_exact_keys(f"ablation {name}", row, expected_ablation_keys[name])

    _require_exact_keys("diagnostics", config["diagnostics"], DIAGNOSTIC_KEYS)
    if int(config["diagnostics"]["stored_frame_limit"]) < int(
        config["model"]["cycles"]
    ) + 1:
        raise ValueError("stored_frame_limit cannot retain the complete history")

    outputs = config["outputs"]
    _require_exact_keys("outputs", outputs, OUTPUT_KEYS)
    for key in OUTPUT_KEYS - {"root"}:
        if outputs[key] is not True:
            raise ValueError(f"canonical campaign requires outputs.{key}=true")


def _write_campaign_readme(
    root: Path,
    config: Mapping[str, Any],
    stage: str,
    aggregate: Mapping[str, Any],
) -> None:
    text = f"""# Coupled internal-observables campaign

Classification: **{CLASSIFICATION}**

Campaign: `{config['campaign_id']}`  
Stage: `{stage}`

{config['claim_boundary']}

## What is here

- `campaign_config.yml`: frozen campaign declaration.
- `campaign_start_receipt.json`: source/environment custody.
- `source_snapshot/`: exact source and tests used for this run.
- `cells/`: one directory per baseline, intervention, collision, or ablation arm.
- `run_index.json`: machine index for every full history and visualization pack.
- `observable_catalog.json`: status row for every requested observable family.
- `aggregate/`: seed/quench/refinement, causal, collision, and feedback summaries.
- `artifact_index.json`: SHA-256 and byte count for every generated artifact.

Each arm retains `coupled_patch_frames.npz`, a documented manifest, a frame
summary, structural JSON/CSV tables, observer/information/excitation JSON/CSV
tables, derived label arrays, and a bounded visualization pack. Paired runs
also retain exact state/velocity/record/defect differences. Collision runs
retain all four arms and the literal `AB-A-B+baseline` residual.

## Coordinate and claim boundary

All distances are graph hops or unit-sphere embedding coordinates. All times
are internal update cycles. Energies, masses, temperatures, densities,
curvatures, particles, scattering channels, and causal fronts are diagnostic
proxies unless their individual report explicitly says otherwise. No report
in this campaign supplies a physical source-to-observable bridge, and no
public measurement data was read.

The fixed carrier geometry is an input. Its spectral dimension and graph-ball
growth can be measured, while evolving spacetime dimension remains unavailable.
Localized tracks and numerical clusters remain unpromoted features unless
matched null controls establish an excess. The record-production direction is
partly imposed by irreversible latching, and the reported defects are
gradient-threshold edges rather than certified topological defects. The
four-arm residual is a nonlinear finite-map diagnostic and is not an S-matrix
element or cross section.

## Minimal Python loading example

```python
import json
import numpy as np

index = json.load(open("run_index.json"))
first = index["runs"][0]
history = np.load(first["history_artifact"])
cycles = history["cycles"]
state = history["state_frames"]
```

Aggregate counts at completion: {json.dumps({key: aggregate[key] for key in ('stage', 'baseline_run_count', 'paired_experiment_count', 'collision_experiment_count')}, sort_keys=True)}
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(value), sort_keys=False), encoding="utf-8")


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _csv_value(row.get(key))
                    for key in keys
                }
            )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="run target-blind coupled internal-observables campaign"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    args = parser.parse_args(argv)
    result = run_coupled_campaign(
        args.config,
        args.out,
        stage=args.stage,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
