from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oph_fpe.bulk.h3_refit import write_h3_refit_report
from oph_fpe.bulk.neutral_bulk import (
    write_neutral_3d_bulk_audit_report,
    write_neutral_independent_rank_selector_audit_report,
    write_overlap_native_graph_geometry_sweep_report,
    write_overlap_native_neutral_control_report,
    write_overlap_residualized_graph_geometry_sweep_report,
    write_prime_geometric_rank_refinement_report,
    write_prime_geometric_rank_sweep_report,
    write_strict_neutral_bulk_frontier_report,
)
from oph_fpe.bulk.observer_consensus_bulk import write_observer_consensus_bulk_readout_report
from oph_fpe.bulk.proof_certificate import write_bulk_proof_certificate
from oph_fpe.bulk.record_to_h3 import recompute_object_chart_from_saved_run
from oph_fpe.bulk.theorem_contract import write_finite_oph_theorem_contract_report
from oph_fpe.claims import (
    H3_RESPONSE_CANDIDATE_RECEIPT,
    H3_RESPONSE_CONTROL_SEPARATION_RECEIPT,
    OBJECT_BULK_POPULATION_RECEIPT,
    OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT,
)
from oph_fpe.cosmology.physical_cmb_output import write_physical_cmb_output_comparison_report
from oph_fpe.cosmology.physical_cmb_prediction import (
    write_physical_cmb_frontier_report,
    write_physical_cmb_input_no_data_use_receipt,
    write_physical_cmb_input_report,
    write_physical_cmb_promotion_audit_report,
)
from oph_fpe.cosmology.silence_to_observation import write_silence_to_observation_report
from oph_fpe.experiments import load_config
from oph_fpe.scale.bw_array import run_bw_array_config
from oph_fpe.viz import (
    write_cmb_neutral_frontier_viewer,
    write_object_h3_bulk_viewer,
    write_run_viewer,
    write_universe_timeline_bundle,
)


@dataclass(frozen=True)
class H3RefitRecipe:
    label: str
    kwargs: dict[str, Any]


REFIT_RECIPES: tuple[H3RefitRecipe, ...] = (
    H3RefitRecipe(
        "scale_rank_512",
        {
            "candidate_count": 4096,
            "candidate_mode": "fibonacci_ball",
            "fit_mode": "joint_global",
            "heldout_fraction": 0.25,
            "anchor_weight": 0.0,
            "max_iterations": 6,
            "feature_selection": "class_distribution_and_change_scale_rank",
            "max_fit_features": 512,
            "min_feature_std": 0.01,
            "min_wrong_scale_feature_delta": 1.0e-4,
            "exclude_observables": ("record_family",),
            "max_features_per_cap_time_observable": 4,
            "refine_steps": 1,
            "refine_max_rows": 24,
            "refine_max_nfev": 24,
            "profile_mode": "static_halfspace",
            "profile_time_scale": 2.0 * math.pi,
            "control_fit_mode": "same_h3_model_not_affine_target_fit",
        },
    ),
    H3RefitRecipe(
        "scale_weighted_512",
        {
            "candidate_count": 4096,
            "candidate_mode": "fibonacci_ball",
            "fit_mode": "joint_global",
            "heldout_fraction": 0.25,
            "anchor_weight": 0.0,
            "max_iterations": 6,
            "feature_selection": "class_distribution_and_change_scale_weighted_rank",
            "max_fit_features": 512,
            "min_feature_std": 0.01,
            "min_wrong_scale_feature_delta": 1.0e-4,
            "exclude_observables": ("record_family",),
            "max_features_per_cap_time_observable": 4,
            "refine_steps": 1,
            "refine_max_rows": 24,
            "refine_max_nfev": 24,
            "profile_mode": "static_halfspace",
            "profile_time_scale": 2.0 * math.pi,
            "control_fit_mode": "same_h3_model_not_affine_target_fit",
        },
    ),
    H3RefitRecipe(
        "scale_rank_768",
        {
            "candidate_count": 4096,
            "candidate_mode": "fibonacci_ball",
            "fit_mode": "joint_global",
            "heldout_fraction": 0.25,
            "anchor_weight": 0.0,
            "max_iterations": 6,
            "feature_selection": "class_distribution_and_change_scale_rank",
            "max_fit_features": 768,
            "min_feature_std": 0.01,
            "min_wrong_scale_feature_delta": 1.0e-4,
            "exclude_observables": ("record_family",),
            "max_features_per_cap_time_observable": 6,
            "refine_steps": 1,
            "refine_max_rows": 24,
            "refine_max_nfev": 24,
            "profile_mode": "static_halfspace",
            "profile_time_scale": 2.0 * math.pi,
            "control_fit_mode": "same_h3_model_not_affine_target_fit",
        },
    ),
    H3RefitRecipe(
        "grouped_scale_weighted",
        {
            "candidate_count": 4096,
            "candidate_mode": "fibonacci_ball",
            "fit_mode": "joint_global",
            "heldout_fraction": 0.25,
            "anchor_weight": 0.0,
            "max_iterations": 6,
            "feature_selection": "grouped_class_distribution_and_change_scale_weighted_rank",
            "max_fit_features": 1024,
            "min_feature_std": 0.01,
            "min_wrong_scale_feature_delta": 1.0e-4,
            "exclude_observables": ("record_family",),
            "max_features_per_cap_time_observable": 8,
            "refine_steps": 1,
            "refine_max_rows": 24,
            "refine_max_nfev": 24,
            "profile_mode": "static_halfspace",
            "profile_time_scale": 2.0 * math.pi,
            "control_fit_mode": "same_h3_model_not_affine_target_fit",
        },
    ),
    H3RefitRecipe(
        "grouped_low_order_transition_contract",
        {
            "candidate_count": 4096,
            "candidate_mode": "fibonacci_ball",
            "fit_mode": "joint_global",
            "heldout_fraction": 0.25,
            "anchor_weight": 0.0,
            "max_iterations": 6,
            "feature_selection": "grouped_signed_transition_no_matrix_scale_weighted_rank",
            "max_fit_features": 512,
            "min_feature_std": 0.01,
            "min_wrong_scale_feature_delta": 1.0e-4,
            "exclude_observables": ("record_family",),
            "max_features_per_cap_time_observable": 8,
            "refine_steps": 1,
            "refine_max_rows": 24,
            "refine_max_nfev": 24,
            "profile_mode": "static_halfspace",
            "profile_time_scale": 2.0 * math.pi,
            "control_fit_mode": "same_h3_model_not_affine_target_fit",
        },
    ),
)

OBJECT_INCIDENCE_MODES: tuple[str, ...] = (
    "transition_history_mixture_cluster",
    "observer_transition_mixture_cluster",
)


def run_oph_universe_pipeline(
    *,
    config_path: Path,
    out_dir: Path,
    run_id: str | None = None,
    source_run_dir: Path | None = None,
    skip_base_run: bool = False,
    max_screen_points: int = 5000,
    max_observers: int = 128,
    max_h3_objects: int = 512,
    emit_visualizations: bool = True,
) -> dict[str, Any]:
    """Run the canonical theorem-following OPH universe pipeline.

    The pipeline keeps the claim boundary strict: it may refine the cached
    modular-response fit and feed the selected audited report forward, but it
    does not relax the H3 candidate, object-population, neutral-bulk, or
    physical-CMB gates.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if skip_base_run:
        if source_run_dir is None:
            raise ValueError("skip_base_run requires source_run_dir")
        run_dir = Path(source_run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"missing source run dir: {run_dir}")
        base_result: dict[str, Any] = {
            "path": str(run_dir),
            "base_run_skipped": True,
            "config": str(config_path),
        }
    else:
        config = dict(load_config(config_path))
        if run_id:
            config["run_id"] = str(run_id)
        base_result = run_bw_array_config(config, out_dir)
        run_dir = Path(base_result["path"])
    refinement_dir = run_dir / "auto_theorem_refinement"
    refinement_dir.mkdir(parents=True, exist_ok=True)

    original_h3 = _read_json(run_dir / "modular_response_h3_report.json")
    original_object = _read_json(run_dir / "observer_chart_object_h3_report.json")
    original_observer = _read_json(run_dir / "observer_modular_experience_report.json")
    _backup_original(run_dir / "modular_response_h3_report.json", refinement_dir)
    _backup_original(run_dir / "observer_chart_object_h3_report.json", refinement_dir)
    _backup_original(run_dir / "observer_modular_experience_report.json", refinement_dir)

    h3_candidates = _run_h3_refinement_sweep(run_dir, refinement_dir, original_h3)
    selected_h3 = max(h3_candidates, key=lambda row: row["score"]) if h3_candidates else None
    if selected_h3 is not None:
        selected_h3_report = _read_json(Path(selected_h3["path"]))
        selected_h3_report["selected_by_oph_universe_pipeline"] = True
        selected_h3_report["selected_h3_refit_label"] = selected_h3["label"]
        _write_json(run_dir / "modular_response_h3_report.json", selected_h3_report)
    else:
        selected_h3_report = original_h3

    object_candidates = _run_object_chart_sweep(run_dir, refinement_dir, h3_candidates)
    if object_candidates:
        selected_object = max(object_candidates, key=lambda row: row["score"])
        selected_object_report = _read_json(Path(selected_object["path"]))
        selected_object_report["selected_by_oph_universe_pipeline"] = True
        selected_object_report["selected_object_chart_label"] = selected_object["label"]
        _write_json(run_dir / "observer_chart_object_h3_report.json", selected_object_report)
    else:
        selected_object = None
        selected_object_report = original_object

    observer_report = _postprocess_observer_experience(
        original_observer,
        selected_h3_report,
        selected_object_report,
    )
    _write_json(run_dir / "observer_modular_experience_report.json", observer_report)
    _patch_emergence_status_from_selected(
        run_dir,
        selected_h3_report,
        selected_object_report,
        observer_report,
    )
    frontier_artifacts = _write_frontier_artifacts(run_dir)

    theorem_contract = write_finite_oph_theorem_contract_report(
        run_dir,
        run_dir / "finite_oph_theorem_contract_report.json",
    )
    proof = write_bulk_proof_certificate(run_dir, run_dir / "bulk_proof_certificate_report.json")
    readout_dir = run_dir / "observer_consensus_bulk"
    readout = write_observer_consensus_bulk_readout_report(
        [run_dir],
        readout_dir,
        observer_sample_count=max(12, min(int(max_observers), 512)),
        object_sample_count=max(24, min(int(max_h3_objects), 1024)),
    )
    silence_to_observation = write_silence_to_observation_report(run_dir, run_dir)
    object_viewer: dict[str, Any] = {}
    run_viewer: dict[str, Any] = {}
    timeline: dict[str, Any] = {}
    cmb_neutral_viewer: dict[str, Any] = {}
    if emit_visualizations:
        object_viewer = write_object_h3_bulk_viewer(
            run_dir,
            run_dir / "object_h3_bulk_viewer.html",
            max_objects=max_h3_objects,
        )
        run_viewer = write_run_viewer(run_dir, run_dir / "oph_realtime_viewer.html", max_screen_points=max_screen_points)
        timeline = write_universe_timeline_bundle(
            small_universe_dir=run_dir,
            observer_run_dir=run_dir,
            consensus_pack_dir=run_dir,
            consensus_readout_dir=readout_dir,
            out_dir=run_dir / "universe_timeline",
            max_screen_points=max_screen_points,
            max_observers=max_observers,
            max_h3_objects=max_h3_objects,
        )
        cmb_neutral_viewer = write_cmb_neutral_frontier_viewer(
            run_dir,
            run_dir / "cmb_neutral_frontier_viewer.html",
        )

    summary = {
        "mode": "oph_universe_theorem_following_pipeline_v1",
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "base_result": base_result,
        "selected_h3_refit": selected_h3,
        "selected_object_chart": selected_object,
        "h3_refit_candidates": _strip_scores(h3_candidates),
        "object_chart_candidates": _strip_scores(object_candidates),
        "final_receipts": {
            "observer_like_self_reading_system_receipt": bool(
                readout.get("observer_like_self_reading_system_receipt", False)
            ),
            "observer_modular_time_receipt": bool(readout.get("observer_modular_time_receipt", False)),
            "h3_response_candidate_receipt": bool(
                selected_h3_report.get(H3_RESPONSE_CANDIDATE_RECEIPT, False)
            ),
            "h3_response_control_separation_receipt": bool(
                selected_h3_report.get(H3_RESPONSE_CONTROL_SEPARATION_RECEIPT, False)
                or selected_h3_report.get("h3_control_separation_receipt", False)
            ),
            "observer_h3_object_population_receipt": bool(
                selected_object_report.get(OBJECT_BULK_POPULATION_RECEIPT, False)
                or selected_object_report.get("observer_chart_bulk_population_receipt", False)
            ),
            "observer_facing_3p1d_h3_experience_receipt": bool(
                readout.get("observer_facing_3p1d_h3_experience_receipt", False)
            ),
            "theorem_assisted_consensus_3d_bulk_readout_receipt": bool(
                readout.get("theorem_assisted_consensus_3d_bulk_readout_receipt", False)
            ),
            "strict_neutral_third_person_bulk_receipt": bool(
                readout.get("strict_neutral_third_person_bulk_receipt", False)
            ),
            "physical_cmb_output_comparison_receipt": bool(
                readout.get("physical_cmb_output_comparison_receipt", False)
            ),
            "physical_cmb_prediction_receipt": bool(readout.get("physical_cmb_prediction_receipt", False)),
            "finite_lorentz_theorem_contract_receipt": bool(
                theorem_contract.get("finite_lorentz_theorem_contract_receipt", False)
            ),
            "paper_faithful_observer_spacetime_emergence_receipt": bool(
                theorem_contract.get("paper_faithful_observer_spacetime_emergence_receipt", False)
            ),
            "paper_faithful_consensus_bulk_emergence_receipt": bool(
                theorem_contract.get("paper_faithful_consensus_bulk_emergence_receipt", False)
            ),
            "scale_compressed_pn_silence_to_observation_receipt": bool(
                silence_to_observation.get("scale_compressed_pn_silence_to_observation_receipt", False)
            ),
            "literal_global_N_capacity_simulated_receipt": bool(
                silence_to_observation.get("literal_global_N_capacity_simulated_receipt", False)
            ),
            "dynamic_p_detuning_control_receipt": bool(
                silence_to_observation.get("dynamic_p_detuning_control_receipt", False)
            ),
        },
        "finite_theorem_contract_summary": {
            "finite_lorentz_theorem_contract_receipt": theorem_contract.get(
                "finite_lorentz_theorem_contract_receipt"
            ),
            "paper_faithful_observer_spacetime_emergence_receipt": theorem_contract.get(
                "paper_faithful_observer_spacetime_emergence_receipt"
            ),
            "paper_faithful_consensus_bulk_emergence_receipt": theorem_contract.get(
                "paper_faithful_consensus_bulk_emergence_receipt"
            ),
            "primary_blockers": theorem_contract.get("primary_blockers", []),
        },
        "proof_summary": {
            "bulk_3d_established_theorem_assisted": proof.get("bulk_3d_established_theorem_assisted"),
            "bulk_3d_established_strict": proof.get("bulk_3d_established_strict"),
            "physical_cmb_prediction": proof.get("physical_cmb_prediction"),
        },
        "silence_to_observation_summary": {
            "scale_compressed_pn_silence_to_observation_receipt": silence_to_observation.get(
                "scale_compressed_pn_silence_to_observation_receipt"
            ),
            "literal_global_N_capacity_simulated_receipt": silence_to_observation.get(
                "literal_global_N_capacity_simulated_receipt"
            ),
            "dynamic_p_detuning_control_receipt": silence_to_observation.get(
                "dynamic_p_detuning_control_receipt"
            ),
            "P_detuning_delta": silence_to_observation.get("closure_coordinates", {}).get("P_detuning_delta"),
            "N_eff": silence_to_observation.get("finite_regulator_depth", {}).get(
                "regulator_entropy_capacity_N_eff"
            ),
            "effective_repair_round_depth": silence_to_observation.get("finite_regulator_depth", {}).get(
                "effective_repair_round_depth"
            ),
        },
        "frontier_artifacts": frontier_artifacts,
        "viewer_outputs": {
            "visualizations_emitted": bool(emit_visualizations),
            "run_viewer": run_viewer.get("viewer_path"),
            "object_h3_viewer": object_viewer.get("viewer_path"),
            "timeline_viewer": timeline.get("viewer_path"),
            "timeline_payload": timeline.get("payload_path"),
            "timeline_instructions": timeline.get("instructions_path"),
            "web_agent_brief": timeline.get("web_coding_agent_brief_path"),
            "cmb_neutral_frontier_viewer": cmb_neutral_viewer.get("viewer_path"),
        },
        "claim_boundary": (
            "Canonical theorem-following OPH universe run. The pipeline instantiates observer-like "
            "self-reading systems, runs a finite overlap-repair simulation, refits the cached modular "
            "response through controlled H3 candidates, recomputes observer-object chart population, "
            "and emits proof/readout/viewer artifacts. It does not lower receipt thresholds. Failed "
            "receipts are blockers, not bugs hidden by configuration choice."
        ),
    }
    _write_json(run_dir / "AUTO_THEOREM_UNIVERSE_SUMMARY.json", summary)
    _write_readme(run_dir / "README_OPH_UNIVERSE_PACK.md", summary)
    return summary


def _write_frontier_artifacts(run_dir: Path) -> dict[str, Any]:
    """Emit hard-gate frontier reports before proof/readout aggregation."""

    neutral_paths: list[Path] = []
    observer_path = run_dir / "observer_views.jsonl"
    neutral_summary: dict[str, Any] = {"neutral_frontier_written": False}
    if observer_path.exists():
        overlap_dir = run_dir / "neutral_overlap_control"
        sweep_dir = run_dir / "neutral_prime_rank_sweep"
        refinement_dir = run_dir / "neutral_prime_rank_refinement"
        selector_dir = run_dir / "neutral_rank_selector_audit"
        graph_dir = run_dir / "neutral_overlap_graph_sweep"
        residual_graph_dir = run_dir / "neutral_overlap_residual_graph_sweep"
        write_overlap_native_neutral_control_report(
            run_dir,
            overlap_dir,
            seed=11,
            max_model_points=384,
        )
        write_prime_geometric_rank_sweep_report(
            run_dir,
            sweep_dir / "prime_geometric_rank_sweep_report.json",
            ranks=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
            seed=11,
            sample_count=384,
            max_model_points=256,
        )
        write_prime_geometric_rank_refinement_report([sweep_dir], refinement_dir)
        write_neutral_independent_rank_selector_audit_report(
            [sweep_dir, refinement_dir],
            selector_dir,
        )
        write_overlap_native_graph_geometry_sweep_report(
            [run_dir],
            graph_dir,
            seeds=(7, 11),
            max_model_points_values=(256, 384),
            k_neighbor_values=(8, 12, 16),
        )
        write_overlap_residualized_graph_geometry_sweep_report(
            [run_dir],
            residual_graph_dir,
            seeds=(7, 11),
            max_model_points_values=(256, 384),
            k_neighbor_values=(8, 12, 16),
            remove_mode_values=(1, 2, 3),
        )
        neutral_paths = [
            overlap_dir,
            sweep_dir,
            refinement_dir,
            selector_dir,
            graph_dir,
            residual_graph_dir,
        ]
        neutral_audit = write_neutral_3d_bulk_audit_report(neutral_paths, run_dir)
        neutral_frontier = write_strict_neutral_bulk_frontier_report(
            [*neutral_paths, run_dir],
            run_dir,
        )
        neutral_summary = {
            "neutral_frontier_written": True,
            "strict_neutral_bulk": bool(neutral_frontier.get("strict_neutral_bulk", False)),
            "strict_neutral_bulk_ready": bool(neutral_frontier.get("strict_neutral_bulk_ready", False)),
            "neutral_blockers": list(neutral_frontier.get("blockers") or [])[:16],
            "neutral_audit_blockers": list(neutral_audit.get("blockers") or [])[:16],
            "neutral_report_dirs": [str(path) for path in neutral_paths],
        }

    no_data = write_physical_cmb_input_no_data_use_receipt([run_dir], run_dir)
    cmb_input = write_physical_cmb_input_report([run_dir], run_dir)
    cmb_promotion = write_physical_cmb_promotion_audit_report([run_dir], run_dir)
    cmb_output = write_physical_cmb_output_comparison_report([run_dir], run_dir)
    cmb_frontier = write_physical_cmb_frontier_report([run_dir], run_dir)
    return {
        **neutral_summary,
        "physical_cmb_frontier_written": True,
        "physical_cmb_no_data_use_receipt": bool(no_data.get("NO_DATA_USE_RECEIPT", False)),
        "physical_cmb_input_contract_receipt": bool(
            cmb_input.get("PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT", False)
        ),
        "physical_cmb_promotion_ready": bool(cmb_promotion.get("physical_cmb_promotion_ready", False)),
        "physical_cmb_output_comparison_receipt": bool(
            cmb_output.get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT", False)
        ),
        "usable_physical_cmb_data_receipt": bool(
            cmb_output.get("USABLE_PHYSICAL_CMB_DATA_RECEIPT", False)
        ),
        "physical_cmb_prediction_receipt": bool(
            cmb_frontier.get("physical_cmb_prediction_receipt", False)
        ),
        "physical_cmb_blockers": list(cmb_frontier.get("blockers") or [])[:16],
    }


def _run_h3_refinement_sweep(
    run_dir: Path,
    refinement_dir: Path,
    original_h3: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if original_h3:
        original_path = refinement_dir / "h3_baseline_original.json"
        _write_json(original_path, original_h3)
        candidates.append(_h3_candidate_row("baseline_original", original_path, original_h3))
    cache = run_dir / "modular_response_kernel_cache.json"
    payload = run_dir / "modular_response_kernel_payload.npz"
    if not cache.exists() or not payload.exists():
        return candidates
    for index, recipe in enumerate(REFIT_RECIPES):
        out_path = refinement_dir / f"h3_refit_{recipe.label}.json"
        try:
            report = write_h3_refit_report(
                run_dir,
                out_path,
                seed=20260752 + index,
                **recipe.kwargs,
            )
        except Exception as exc:  # pragma: no cover - retained in output for long runs
            error_path = refinement_dir / f"h3_refit_{recipe.label}_error.json"
            _write_json(error_path, {"label": recipe.label, "error": str(exc)})
            continue
        candidates.append(_h3_candidate_row(recipe.label, out_path, report))
    return candidates


def _run_object_chart_sweep(
    run_dir: Path,
    refinement_dir: Path,
    h3_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    object_candidates: list[dict[str, Any]] = []
    selected_h3_candidates = sorted(h3_candidates, key=lambda row: row["score"], reverse=True)[:1]
    for h3_row in selected_h3_candidates:
        h3_path = Path(h3_row["path"])
        for mode in OBJECT_INCIDENCE_MODES:
            label = f"{h3_row['label']}__{mode}"
            out_path = refinement_dir / f"object_chart_{_safe_label(label)}.json"
            try:
                report = recompute_object_chart_from_saved_run(
                    run_dir=run_dir,
                    h3_report_path=h3_path,
                    out_path=out_path,
                    shuffle_control_count=16,
                    incidence_mode=mode,
                )
            except Exception as exc:  # pragma: no cover - retained in output for long runs
                _write_json(refinement_dir / f"object_chart_{_safe_label(label)}_error.json", {"error": str(exc)})
                continue
            object_candidates.append(_object_candidate_row(label, out_path, report))
    return object_candidates


def _postprocess_observer_experience(
    original: dict[str, Any],
    h3_report: dict[str, Any],
    object_report: dict[str, Any],
) -> dict[str, Any]:
    component_gates = dict(original.get("component_gates", {}))
    observer_modular_time = bool(
        original.get("observer_modular_time_receipt", False)
        or component_gates.get("observer_modular_time_receipt", False)
    )
    bw_kms = bool(component_gates.get("bw_kms_branch_replay_receipt", False))
    chart = bool(component_gates.get("conformal_h3_chart_receipt", False))
    h3_response = bool(
        h3_report.get(H3_RESPONSE_CANDIDATE_RECEIPT, False)
        or h3_report.get("MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", False)
        or h3_report.get(H3_RESPONSE_CONTROL_SEPARATION_RECEIPT, False)
        or h3_report.get("h3_control_separation_receipt", False)
    )
    object_population = bool(
        object_report.get(OBJECT_BULK_POPULATION_RECEIPT, False)
        or object_report.get("OBJECT_H3_NONBOUNDARY_POPULATION_RECEIPT", False)
        or object_report.get("observer_chart_bulk_population_receipt", False)
    )
    component_gates = {
        "observer_modular_time_receipt": observer_modular_time,
        "bw_kms_branch_replay_receipt": bw_kms,
        "conformal_h3_chart_receipt": chart,
        "h3_modular_response_receipt": h3_response,
        "observer_h3_object_population_receipt": object_population,
    }
    full = bool(all(component_gates.values()))
    report = dict(original)
    report.update(
        {
            "postprocessed_by_oph_universe_pipeline": True,
            "component_gates": component_gates,
            "blockers": [name for name, passed in component_gates.items() if not passed],
            OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT: full,
            "observer_facing_3p1d_h3_experience_receipt": full,
            "claim_boundary": (
                str(original.get("claim_boundary", "")).strip()
                + " Postprocessed after canonical H3 refit/object-chart selection; this only updates "
                "downstream receipt wiring from selected audited reports and does not relax thresholds."
            ).strip(),
        }
    )
    return report


def _patch_emergence_status_from_selected(
    run_dir: Path,
    h3_report: dict[str, Any],
    object_report: dict[str, Any],
    observer_report: dict[str, Any],
) -> dict[str, Any]:
    path = run_dir / "emergence_status_report.json"
    emergence = _read_json(path)
    if not emergence:
        return {}
    h3_gates = h3_report.get("h3_response_stage_gates") or {}
    h3_candidate = bool(
        h3_report.get(H3_RESPONSE_CANDIDATE_RECEIPT, False)
        or h3_report.get("MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", False)
    )
    h3_control = bool(
        h3_report.get(H3_RESPONSE_CONTROL_SEPARATION_RECEIPT, False)
        or h3_report.get("h3_control_separation_receipt", False)
        or h3_gates.get(H3_RESPONSE_CONTROL_SEPARATION_RECEIPT, False)
        or h3_gates.get("intermediate_control_separation_receipt", False)
    )
    object_population = bool(
        object_report.get(OBJECT_BULK_POPULATION_RECEIPT, False)
        or object_report.get("observer_chart_bulk_population_receipt", False)
    )
    localized_nonboundary = bool(
        object_report.get("localized_nonboundary_bulk_population_receipt", False)
        or object_report.get("OBJECT_H3_NONBOUNDARY_POPULATION_RECEIPT", False)
    )
    observer_3p1d = bool(
        observer_report.get(OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT, False)
        or observer_report.get("observer_facing_3p1d_h3_experience_receipt", False)
    )
    emergence.update(
        {
            "postprocessed_by_oph_universe_pipeline": True,
            "selected_h3_refit_label": h3_report.get("selected_h3_refit_label"),
            "selected_object_chart_label": object_report.get("selected_object_chart_label"),
            "modular_response_h3_candidate_receipt": h3_candidate,
            H3_RESPONSE_CANDIDATE_RECEIPT: h3_candidate,
            "modular_response_h3_control_separation_receipt": h3_control,
            H3_RESPONSE_CONTROL_SEPARATION_RECEIPT: h3_control,
            "observer_chart_object_h3_receipt": bool(
                object_report.get("observer_chart_object_h3_receipt", False)
            ),
            "observer_chart_localized_object_precursor_receipt": bool(
                object_report.get("localized_object_precursor_receipt", False)
                or object_report.get("localized_nonboundary_object_precursor_receipt", False)
                or object_report.get("localized_h3_object_precursor_receipt", False)
            ),
            "observer_chart_modular_response_h3_control_separation_receipt": bool(
                object_report.get("modular_response_h3_control_separation_receipt", False)
                or h3_control
            ),
            "observer_chart_localized_nonboundary_bulk_population_receipt": localized_nonboundary,
            "observer_chart_localized_h3_bulk_population_receipt": bool(
                object_report.get("localized_h3_bulk_population_receipt", False) or localized_nonboundary
            ),
            "observer_chart_bulk_population_receipt": object_population,
            OBJECT_BULK_POPULATION_RECEIPT: object_population,
            "object_bulk_population_receipt": object_population,
            OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT: observer_3p1d,
            "observer_facing_3p1d_h3_experience_receipt": observer_3p1d,
            "bulk_3d_established": False,
            "claim_boundary": (
                str(emergence.get("claim_boundary", "")).strip()
                + " Postprocessed by the OPH universe pipeline to point at the selected audited "
                "H3/object reports. This synchronization does not relax theorem or bulk gates."
            ).strip(),
        }
    )
    _write_json(path, emergence)
    return emergence


def _h3_candidate_row(label: str, path: Path, report: dict[str, Any]) -> dict[str, Any]:
    gates = report.get("h3_response_stage_gates", {}) if isinstance(report, dict) else {}
    h3_fit = report.get("h3_fit", {}) if isinstance(report, dict) else {}
    row = {
        "label": label,
        "path": str(path),
        "candidate_receipt": bool(report.get(H3_RESPONSE_CANDIDATE_RECEIPT, False)),
        "control_separation_receipt": bool(
            report.get(H3_RESPONSE_CONTROL_SEPARATION_RECEIPT, False)
            or report.get("h3_control_separation_receipt", False)
        ),
        "signal_gate": bool(gates.get("signal_gate", False)),
        "geometry_gate": bool(gates.get("geometry_gate", False)),
        "aggregate_wrong_scale_gate": bool(gates.get("aggregate_wrong_scale_gate", False)),
        "material_feature_gate": bool(gates.get("material_feature_gate", False)),
        "heldout_explained_variance": _maybe_float(
            gates.get("h3_heldout_explained_variance", h3_fit.get("heldout_explained_variance"))
        ),
        "heldout_normalized_rmse": _maybe_float(
            gates.get("h3_heldout_normalized_rmse", h3_fit.get("heldout_normalized_rmse"))
        ),
        "material_wrong_scale_win_fraction": _maybe_float(gates.get("material_wrong_scale_win_fraction")),
        "feature_count": int(report.get("feature_count", 0) or 0),
    }
    row["score"] = _h3_score(row)
    return row


def _object_candidate_row(label: str, path: Path, report: dict[str, Any]) -> dict[str, Any]:
    row = {
        "label": label,
        "path": str(path),
        "object_bulk_population_receipt": bool(
            report.get(OBJECT_BULK_POPULATION_RECEIPT, False)
            or report.get("observer_chart_bulk_population_receipt", False)
        ),
        "object_preview_receipt": bool(report.get("THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT", False)),
        "object_median_receipt": bool(report.get("observer_chart_object_h3_median_receipt", False)),
        "localized_precursor_receipt": bool(
            report.get("localized_object_precursor_receipt", False)
            or report.get("localized_nonboundary_object_precursor_receipt", False)
            or report.get("localized_h3_object_precursor_receipt", False)
        ),
        "h3_beats_shuffled_robust": bool(report.get("h3_beats_shuffled_incidence_robust", False)),
        "boundary_leakage_audit_pass": bool(report.get("boundary_leakage_audit_pass", False)),
        "localized_object_count": int(report.get("localized_object_count", 0) or 0),
        "localized_not_boundary_object_count": int(report.get("localized_not_boundary_object_count", 0) or 0),
        "object_count": int(report.get("object_count", 0) or 0),
        "median_h3_compactness_normalized": _maybe_float(report.get("median_h3_compactness_normalized")),
        "median_shuffled_h3_compactness_normalized": _maybe_float(
            report.get("median_shuffled_h3_compactness_normalized")
        ),
    }
    row["score"] = _object_score(row)
    return row


def _h3_score(row: dict[str, Any]) -> tuple[Any, ...]:
    ev = row.get("heldout_explained_variance")
    rmse = row.get("heldout_normalized_rmse")
    material = row.get("material_wrong_scale_win_fraction")
    return (
        int(row["candidate_receipt"]),
        int(row["control_separation_receipt"]),
        int(row["signal_gate"]),
        int(row["geometry_gate"]),
        int(row["aggregate_wrong_scale_gate"]),
        int(row["material_feature_gate"]),
        -(float(material) if material is not None else 1.0e9),
        float(ev) if ev is not None else -1.0e9,
        -(float(rmse) if rmse is not None else 1.0e9),
        int(row["feature_count"]),
    )


def _object_score(row: dict[str, Any]) -> tuple[Any, ...]:
    h3 = row.get("median_h3_compactness_normalized")
    shuffled = row.get("median_shuffled_h3_compactness_normalized")
    margin = (float(shuffled) - float(h3)) if h3 is not None and shuffled is not None else -1.0e9
    return (
        int(row["object_bulk_population_receipt"]),
        int(row["object_preview_receipt"]),
        int(row["localized_precursor_receipt"]),
        int(row["h3_beats_shuffled_robust"]),
        int(row["object_median_receipt"]),
        int(row["boundary_leakage_audit_pass"]),
        int(row["localized_not_boundary_object_count"]),
        int(row["localized_object_count"]),
        margin,
        int(row["object_count"]),
    )


def _strip_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "score"}
        stripped.append(item)
    return stripped


def _write_readme(path: Path, summary: dict[str, Any]) -> None:
    receipts = summary.get("final_receipts", {})
    viewers = summary.get("viewer_outputs", {})
    lines = [
        "# OPH Universe Simulation Pack",
        "",
        "This pack was generated by the canonical theorem-following OPH universe pipeline.",
        "It instantiates observer-like self-reading systems and keeps failed theorem gates explicit.",
        "",
        "## Key Receipts",
    ]
    for key, value in receipts.items():
        lines.append(f"- `{key}`: `{str(bool(value)).lower()}`")
    lines.extend(
        [
            "",
            "## Viewers",
            f"- Realtime screen/repair viewer: `{viewers.get('run_viewer')}`",
            f"- Object H3 viewer: `{viewers.get('object_h3_viewer')}`",
            f"- Universe timeline viewer: `{viewers.get('timeline_viewer')}`",
            f"- Visualization payload: `{viewers.get('timeline_payload')}`",
            f"- Visualization instructions: `{viewers.get('timeline_instructions')}`",
            f"- Web coding agent brief: `{viewers.get('web_agent_brief')}`",
            "",
            "## Claim Boundary",
            str(summary.get("claim_boundary", "")),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _backup_original(path: Path, backup_dir: Path) -> None:
    if not path.exists():
        return
    destination = backup_dir / f"original_{path.name}"
    if not destination.exists():
        shutil.copy2(path, destination)


def _safe_label(value: str) -> str:
    safe = []
    for char in str(value):
        if char.isalnum() or char in {"_", "-", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe)


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
