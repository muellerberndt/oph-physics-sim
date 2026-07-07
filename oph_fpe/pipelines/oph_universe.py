from __future__ import annotations

import json
import math
import os
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
    write_strict_neutral_bulk_report,
    write_strict_neutral_bulk_frontier_report,
)
from oph_fpe.bulk.neutral_object_bulk import write_strict_neutral_object_bulk_report
from oph_fpe.bulk.observer_consensus_bulk import write_observer_consensus_bulk_readout_report
from oph_fpe.bulk.proof_certificate import write_bulk_proof_certificate
from oph_fpe.bulk.record_to_h3 import recompute_object_chart_from_saved_run
from oph_fpe.bulk.einstein_bridge import write_einstein_bridge_manifest
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
from oph_fpe.cosmology.physical_cmb_sources import write_physical_cmb_source_readiness_report
from oph_fpe.cosmology.ba_kernel import ba_kernel_report_from_parent_report
from oph_fpe.cosmology.camb_adapter import write_official_planck_readiness_report
from oph_fpe.cosmology.compressed_likelihood import write_compressed_likelihood_reference_report
from oph_fpe.cosmology.finite_certificates import write_run_proxy_finite_certificate_bundle
from oph_fpe.cosmology.finite_repair_transition_clock import write_finite_repair_transition_clock_report
from oph_fpe.cosmology.camb_adapter import (
    write_camb_lcdm_baseline_report,
    write_finite_repair_clock_cmb_camb_report,
    write_scale_compressed_cmb_camb_report,
)
from oph_fpe.cosmology.silence_to_observation import write_silence_to_observation_report
from oph_fpe.defects.gravity_assay import (
    write_free_two_defect_dynamics_report,
    write_organic_defect_population_report,
    write_two_defect_stress_contraction_assay_report,
)
from oph_fpe.ensembles.reference_vacuum import write_reference_vacuum_baseline_report
from oph_fpe.experiments import load_config
from oph_fpe.gauge.yang_mills_gap import write_yang_mills_gap_certificate_report
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


H3_RESPONSE_EXCLUDED_AUX_OBSERVABLES: tuple[str, ...] = ("record_family", "repair_load_bucket")


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
            "exclude_observables": H3_RESPONSE_EXCLUDED_AUX_OBSERVABLES,
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
            "exclude_observables": H3_RESPONSE_EXCLUDED_AUX_OBSERVABLES,
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
            "exclude_observables": H3_RESPONSE_EXCLUDED_AUX_OBSERVABLES,
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
            "exclude_observables": H3_RESPONSE_EXCLUDED_AUX_OBSERVABLES,
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
            "exclude_observables": H3_RESPONSE_EXCLUDED_AUX_OBSERVABLES,
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
    seed: int | None = None,
    inner_jobs: int | None = None,
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
    base_config = dict(load_config(config_path))
    export_settings = _visualization_export_settings(
        base_config,
        max_screen_points=max_screen_points,
        max_observers=max_observers,
        max_h3_objects=max_h3_objects,
    )
    max_screen_points = export_settings["max_screen_points"]
    max_observers = export_settings["max_observers"]
    max_h3_objects = export_settings["max_h3_objects"]
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
        config = dict(base_config)
        if run_id:
            config["run_id"] = str(run_id)
        if seed is not None:
            config["seed"] = int(seed)
        if inner_jobs is not None:
            bw_cfg = dict(config.get("bw", {}) or {})
            bw_cfg["n_jobs"] = int(inner_jobs)
            config["bw"] = bw_cfg
            cosmology_cfg = dict(config.get("cosmology", {}) or {})
            angular_cfg = dict(cosmology_cfg.get("angular_power", {}) or {})
            angular_cfg["n_jobs"] = int(inner_jobs)
            cosmology_cfg["angular_power"] = angular_cfg
            config["cosmology"] = cosmology_cfg
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
    frontier_artifacts = _write_frontier_artifacts(run_dir, base_config)

    write_einstein_bridge_manifest(
        run_dir,
        run_dir / "einstein_bridge_manifest.json",
    )
    theorem_contract = write_finite_oph_theorem_contract_report(
        run_dir,
        run_dir / "finite_oph_theorem_contract_report.json",
    )
    proof = write_bulk_proof_certificate(run_dir, run_dir / "bulk_proof_certificate_report.json")
    readout_dir = run_dir / "observer_consensus_bulk"
    readout = write_observer_consensus_bulk_readout_report(
        [run_dir],
        readout_dir,
        observer_sample_count=export_settings["readout_observer_sample_count"],
        object_sample_count=export_settings["readout_object_sample_count"],
    )
    visualizer_csv_aliases = _write_visualizer_csv_aliases(run_dir, readout_dir)
    silence_to_observation = write_silence_to_observation_report(run_dir, run_dir)
    cmb_diagnostics = _cmb_diagnostic_summary(run_dir)
    post_theorem_readiness = _post_theorem_large_run_readiness(
        theorem_contract=theorem_contract,
        proof=proof,
        readout=readout,
        frontier_artifacts=frontier_artifacts,
        cmb_diagnostics=cmb_diagnostics,
    )
    _write_json(run_dir / "large_run_readiness_report.json", post_theorem_readiness)
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
        "visualization_export_settings": export_settings,
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
            "screen_proxy_cmb_receipt": bool(cmb_diagnostics["screen_proxy_cmb_receipt"]),
            "cmb_lite_shape_comparison_receipt": bool(cmb_diagnostics["cmb_lite_shape_comparison_receipt"]),
            "cmb_lite_real_ell_physical_comparison_receipt": bool(
                cmb_diagnostics["cmb_lite_real_ell_physical_comparison_receipt"]
            ),
            "finite_lorentz_theorem_contract_receipt": bool(
                theorem_contract.get("finite_lorentz_theorem_contract_receipt", False)
            ),
            "paper_faithful_observer_spacetime_emergence_receipt": bool(
                theorem_contract.get("paper_faithful_observer_spacetime_emergence_receipt", False)
            ),
            "paper_faithful_consensus_bulk_emergence_receipt": bool(
                theorem_contract.get("paper_faithful_consensus_bulk_emergence_receipt", False)
            ),
            "paper_geometric_branch_lorentz_contract_receipt": bool(
                theorem_contract.get("paper_geometric_branch_lorentz_contract_receipt", False)
            ),
            "paper_geometric_branch_observer_spacetime_emergence_receipt": bool(
                theorem_contract.get("paper_geometric_branch_observer_spacetime_emergence_receipt", False)
            ),
            "paper_geometric_branch_consensus_bulk_emergence_receipt": bool(
                theorem_contract.get("paper_geometric_branch_consensus_bulk_emergence_receipt", False)
            ),
            "simulation_matches_observer_facing_oph_spacetime_bulk_prediction_receipt": bool(
                theorem_contract.get(
                    "simulation_matches_observer_facing_oph_spacetime_bulk_prediction_receipt",
                    theorem_contract.get("paper_geometric_branch_consensus_bulk_emergence_receipt", False),
                )
            ),
            "strict_neutral_bulk_contract_receipt": bool(
                theorem_contract.get("strict_neutral_bulk_contract_receipt", False)
            ),
            "einstein_branch_entry_contract_receipt": bool(
                theorem_contract.get("einstein_branch_entry_contract_receipt", False)
                or theorem_contract.get("OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1", False)
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
            "paper_geometric_branch_lorentz_contract_receipt": theorem_contract.get(
                "paper_geometric_branch_lorentz_contract_receipt"
            ),
            "paper_geometric_branch_observer_spacetime_emergence_receipt": theorem_contract.get(
                "paper_geometric_branch_observer_spacetime_emergence_receipt"
            ),
            "paper_geometric_branch_consensus_bulk_emergence_receipt": theorem_contract.get(
                "paper_geometric_branch_consensus_bulk_emergence_receipt"
            ),
            "strict_neutral_bulk_contract_receipt": theorem_contract.get("strict_neutral_bulk_contract_receipt"),
            "einstein_branch_entry_contract_receipt": theorem_contract.get(
                "einstein_branch_entry_contract_receipt"
            ),
            "einstein_branch_entry_primary_blockers": theorem_contract.get(
                "einstein_branch_entry_primary_blockers", []
            ),
            "einstein_branch_entry_blockers": theorem_contract.get("einstein_branch_entry_blockers", []),
            "chart_blind_strict_neutral_blockers": theorem_contract.get(
                "chart_blind_strict_neutral_blockers", []
            ),
            "paper_geometric_branch_primary_blockers": theorem_contract.get(
                "paper_geometric_branch_primary_blockers", []
            ),
            "primary_blockers": theorem_contract.get("primary_blockers", []),
        },
        "proof_summary": {
            "bulk_3d_established_theorem_assisted": proof.get("bulk_3d_established_theorem_assisted"),
            "bulk_3d_established_observer_facing_consensus": proof.get(
                "bulk_3d_established_observer_facing_consensus"
            ),
            "bulk_3d_established_paper_geometric_branch_observer_facing_consensus": proof.get(
                "bulk_3d_established_paper_geometric_branch_observer_facing_consensus"
            ),
            "bulk_3d_established_strict": proof.get("bulk_3d_established_strict"),
            "bulk_3d_established_chart_blind_strict_neutral": proof.get(
                "bulk_3d_established_chart_blind_strict_neutral"
            ),
            "physical_cmb_prediction": proof.get("physical_cmb_prediction"),
        },
        "post_theorem_large_run_readiness": post_theorem_readiness,
        "cmb_diagnostic_summary": cmb_diagnostics,
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
            "timeline_sidecar_exports": timeline.get("sidecar_exports"),
            "timeline_instructions": timeline.get("instructions_path"),
            "web_agent_brief": timeline.get("web_coding_agent_brief_path"),
            "cmb_neutral_frontier_viewer": cmb_neutral_viewer.get("viewer_path"),
            "visualizer_csv_aliases": visualizer_csv_aliases,
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


def _write_visualizer_csv_aliases(run_dir: Path, readout_dir: Path) -> dict[str, Any]:
    """Expose stable root-level CSV names consumed by standalone viewers."""

    aliases: dict[str, Any] = {}
    for source_name, alias_name in (
        ("consensus_h3_object_rows.csv", "h3_objects.csv"),
        ("observer_perspective_rows.csv", "observer_perspective_rows.csv"),
    ):
        source = Path(readout_dir) / source_name
        destination = Path(run_dir) / alias_name
        if not source.exists():
            aliases[alias_name] = {
                "written": False,
                "source": str(source),
                "path": str(destination),
                "reason": "source_missing",
            }
            continue
        shutil.copyfile(source, destination)
        aliases[alias_name] = {
            "written": True,
            "source": str(source),
            "path": str(destination),
        }
    return aliases


def _post_theorem_large_run_readiness(
    *,
    theorem_contract: dict[str, Any],
    proof: dict[str, Any],
    readout: dict[str, Any],
    frontier_artifacts: dict[str, Any],
    cmb_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    observer_facing_bulk = bool(
        readout.get("observer_facing_consensus_3d_bulk_readout_receipt", False)
        and (
            theorem_contract.get("paper_geometric_branch_consensus_bulk_emergence_receipt", False)
            or theorem_contract.get("paper_faithful_consensus_bulk_emergence_receipt", False)
        )
    )
    strict_neutral = bool(
        theorem_contract.get("strict_neutral_bulk_contract_receipt", False)
        or proof.get("bulk_3d_established_chart_blind_strict_neutral", False)
    )
    physical_cmb = bool(
        frontier_artifacts.get("physical_cmb_prediction_receipt", False)
        or proof.get("physical_cmb_prediction", False)
    )
    screen_cmb = bool(cmb_diagnostics.get("screen_proxy_cmb_receipt", False))
    vacuum_reference = bool(frontier_artifacts.get("reference_vacuum_regression_receipt", False))
    vacuum_native = bool(frontier_artifacts.get("oph_native_vacuum_promotion_receipt", False))
    controlled_gravity_diagnostic = bool(frontier_artifacts.get("two_defect_stress_contraction_assay_receipt", False))
    free_gravity_diagnostic = bool(frontier_artifacts.get("free_two_defect_dynamics_receipt", False))
    gravity_diagnostic = bool(controlled_gravity_diagnostic or free_gravity_diagnostic)
    einstein_branch_entry = bool(
        theorem_contract.get("einstein_branch_entry_contract_receipt", False)
        or theorem_contract.get("OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1", False)
        or proof.get("einstein_branch_entry_contract_receipt", False)
        or proof.get("OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1", False)
    )
    raw_production_gravity = bool(
        frontier_artifacts.get("raw_production_gravity_receipt", False)
        or frontier_artifacts.get("production_gravity_receipt", False)
    )
    production_gravity = bool(raw_production_gravity and einstein_branch_entry)

    if physical_cmb and strict_neutral and observer_facing_bulk:
        recommended = "physical_cmb_and_strict_bulk_large_run"
    elif observer_facing_bulk:
        recommended = "observer_facing_visualization_large_run_with_diagnostic_cmb_vacuum"
    elif screen_cmb:
        recommended = "screen_cmb_proxy_refinement"
    else:
        recommended = "do_not_scale_yet"

    blockers = []
    if not observer_facing_bulk:
        blockers.extend(
            theorem_contract.get("paper_geometric_branch_primary_blockers")
            or readout.get("paper_geometric_branch_primary_blockers")
            or readout.get("finite_theorem_contract_primary_blockers")
            or ["observer_facing_consensus_bulk_receipt_false"]
        )
    if not strict_neutral:
        blockers.extend(theorem_contract.get("strict_neutral_blockers") or [])
    if not physical_cmb:
        blockers.extend(frontier_artifacts.get("physical_cmb_blockers") or [])
    if not vacuum_native:
        blockers.append("oph_native_vacuum_promotion_receipt_false")
    if not production_gravity:
        blockers.append("production_gravity_receipt_false")
    if raw_production_gravity and not einstein_branch_entry:
        blockers.append("einstein_branch_entry_contract_receipt_false")
        blockers.extend(theorem_contract.get("einstein_branch_entry_primary_blockers") or [])

    return {
        "mode": "post_theorem_large_run_readiness_v0",
        "recommended_large_run_lane": recommended,
        "cloud_run_safe_for_visualization_data": observer_facing_bulk,
        "cloud_run_safe_for_physical_cmb_prediction": physical_cmb,
        "cloud_run_safe_for_strict_neutral_bulk_claim": strict_neutral,
        "lanes": {
            "observer_facing_3p1d_h3_bulk_visualization": {
                "status": "scale_candidate" if observer_facing_bulk else "blocked",
                "scale_candidate": observer_facing_bulk,
                "details": {
                    "readout_receipt": readout.get("observer_facing_consensus_3d_bulk_readout_receipt"),
                    "paper_geometric_branch_receipt": theorem_contract.get(
                        "paper_geometric_branch_consensus_bulk_emergence_receipt"
                    ),
                    "paper_faithful_receipt": theorem_contract.get(
                        "paper_faithful_consensus_bulk_emergence_receipt"
                    ),
                },
            },
            "strict_neutral_third_person_bulk": {
                "status": "scale_candidate" if strict_neutral else "blocked",
                "scale_candidate": strict_neutral,
                "blockers": list(theorem_contract.get("strict_neutral_blockers") or []),
            },
            "screen_cmb_proxy": {
                "status": "scale_candidate" if screen_cmb else "blocked",
                "scale_candidate": screen_cmb,
                "physical_prediction": False,
            },
            "physical_cmb_prediction": {
                "status": "scale_candidate" if physical_cmb else "blocked",
                "scale_candidate": physical_cmb,
                "blockers": list(frontier_artifacts.get("physical_cmb_blockers") or []),
            },
            "reference_vacuum": {
                "status": "diagnostic_ready" if vacuum_reference else "blocked",
                "scale_candidate": vacuum_reference,
                "oph_native_vacuum": vacuum_native,
            },
            "two_defect_gravity": {
                "status": "diagnostic_ready" if gravity_diagnostic else "blocked",
                "scale_candidate": gravity_diagnostic,
                "controlled_stress_contraction": controlled_gravity_diagnostic,
                "free_two_defect_dynamics": free_gravity_diagnostic,
                "free_contact_outcome": frontier_artifacts.get("free_two_defect_contact_outcome"),
                "einstein_branch_entry_contract": einstein_branch_entry,
                "raw_production_gravity_requested": raw_production_gravity,
                "production_gravity": production_gravity,
                "production_gravity_blockers": (
                    []
                    if production_gravity
                    else (
                        theorem_contract.get("einstein_branch_entry_primary_blockers")
                        or ["einstein_branch_entry_contract_receipt_false"]
                    )
                ),
            },
        },
        "blockers": sorted({str(blocker) for blocker in blockers if str(blocker)}),
        "claim_boundary": (
            "Post-theorem routing after H3/object refinement, finite theorem contract audit, proof "
            "certificate, and observer readout. A visualization-safe lane can be true while physical "
            "CMB, strict-neutral bulk, OPH-native vacuum, and production-gravity lanes remain closed."
        ),
    }


def refresh_post_theorem_large_run_readiness_report(run_dir: Path) -> dict[str, Any]:
    """Rewrite the public readiness report from the final OPH-universe summary."""

    run_path = Path(run_dir)
    summary = _read_json(run_path / "AUTO_THEOREM_UNIVERSE_SUMMARY.json")
    report = summary.get("post_theorem_large_run_readiness") if isinstance(summary, dict) else None
    if not isinstance(report, dict) or not report:
        theorem_contract = _read_json(run_path / "finite_oph_theorem_contract_report.json")
        proof = _read_json(run_path / "bulk_proof_certificate_report.json")
        readout = _read_json(run_path / "observer_consensus_bulk" / "observer_consensus_bulk_readout_report.json")
        frontier_artifacts = summary.get("frontier_artifacts", {}) if isinstance(summary, dict) else {}
        cmb_diagnostics = summary.get("cmb_diagnostic_summary", {}) if isinstance(summary, dict) else {}
        report = _post_theorem_large_run_readiness(
            theorem_contract=theorem_contract,
            proof=proof,
            readout=readout,
            frontier_artifacts=frontier_artifacts if isinstance(frontier_artifacts, dict) else {},
            cmb_diagnostics=cmb_diagnostics if isinstance(cmb_diagnostics, dict) else {},
        )
    _write_json(run_path / "large_run_readiness_report.json", report)
    return report


def _write_frontier_artifacts(run_dir: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Emit hard-gate frontier reports before proof/readout aggregation."""

    neutral_cfg = _neutral_frontier_settings(config or {})
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
        strict_neutral_record = write_strict_neutral_bulk_report(
            run_dir,
            out=run_dir / "strict_neutral_bulk_report.json",
            seed=neutral_cfg["seed"],
            max_model_points=neutral_cfg["strict_bulk_max_model_points"],
            planted_control_points=neutral_cfg["planted_control_points"],
        )
        strict_neutral_object = write_strict_neutral_object_bulk_report(
            run_dir,
            out=run_dir / "strict_neutral_object_bulk_report.json",
            seed=neutral_cfg["seed"],
            min_objects=neutral_cfg["object_min_objects"],
            min_observers_per_object=neutral_cfg["object_min_observers_per_object"],
            max_observer_fraction_per_object=neutral_cfg["object_max_observer_fraction_per_object"],
            max_model_points=neutral_cfg["object_max_model_points"],
            heldout_fraction=neutral_cfg["object_heldout_fraction"],
        )
        write_overlap_native_neutral_control_report(
            run_dir,
            overlap_dir,
            seed=neutral_cfg["seed"],
            max_model_points=neutral_cfg["overlap_control_max_model_points"],
        )
        write_prime_geometric_rank_sweep_report(
            run_dir,
            sweep_dir / "prime_geometric_rank_sweep_report.json",
            ranks=neutral_cfg["ranks"],
            seed=neutral_cfg["seed"],
            sample_count=neutral_cfg["sample_count"],
            max_model_points=neutral_cfg["max_model_points"],
        )
        write_prime_geometric_rank_refinement_report([sweep_dir], refinement_dir)
        write_neutral_independent_rank_selector_audit_report(
            [sweep_dir, refinement_dir],
            selector_dir,
        )
        write_overlap_native_graph_geometry_sweep_report(
            [run_dir],
            graph_dir,
            seeds=neutral_cfg["graph_seeds"],
            max_model_points_values=neutral_cfg["graph_max_model_points_values"],
            k_neighbor_values=neutral_cfg["graph_k_neighbor_values"],
            workers=neutral_cfg["graph_workers"],
        )
        write_overlap_residualized_graph_geometry_sweep_report(
            [run_dir],
            residual_graph_dir,
            seeds=neutral_cfg["graph_seeds"],
            max_model_points_values=neutral_cfg["graph_max_model_points_values"],
            k_neighbor_values=neutral_cfg["graph_k_neighbor_values"],
            remove_mode_values=neutral_cfg["residual_remove_mode_values"],
            workers=neutral_cfg["graph_workers"],
        )
        neutral_paths = [
            run_dir / "strict_neutral_bulk_report.json",
            run_dir / "strict_neutral_object_bulk_report.json",
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
            "strict_neutral_record_report_written": True,
            "strict_neutral_record_bulk": bool(strict_neutral_record.get("strict_neutral_bulk", False)),
            "strict_neutral_record_blockers": list(strict_neutral_record.get("blockers") or [])[:16],
            "strict_neutral_object_report_written": True,
            "strict_neutral_object_bulk": bool(strict_neutral_object.get("strict_neutral_object_bulk", False)),
            "strict_neutral_object_count": int(strict_neutral_object.get("object_count") or 0),
            "strict_neutral_object_blockers": list(strict_neutral_object.get("blockers") or [])[:16],
            "neutral_blockers": list(neutral_frontier.get("blockers") or [])[:16],
            "neutral_audit_blockers": list(neutral_audit.get("blockers") or [])[:16],
            "neutral_report_dirs": [str(path) for path in neutral_paths],
            "neutral_frontier_settings": neutral_cfg,
        }

    visualization_diagnostics = _write_visualization_diagnostic_artifacts(run_dir, config or {})
    source_artifacts = _write_physical_cmb_source_artifacts(run_dir)
    transfer_artifacts = _write_physical_cmb_transfer_artifacts(run_dir, config or {})
    no_data = write_physical_cmb_input_no_data_use_receipt([run_dir], run_dir)
    cmb_sources = write_physical_cmb_source_readiness_report([run_dir], run_dir)
    cmb_input = write_physical_cmb_input_report([run_dir], run_dir)
    cmb_promotion = write_physical_cmb_promotion_audit_report([run_dir], run_dir)
    cmb_output = write_physical_cmb_output_comparison_report([run_dir], run_dir)
    cmb_frontier = write_physical_cmb_frontier_report([run_dir], run_dir)
    return {
        **neutral_summary,
        **visualization_diagnostics,
        **source_artifacts,
        **transfer_artifacts,
        "physical_cmb_frontier_written": True,
        "physical_cmb_source_readiness_written": True,
        "finite_covariant_parent_receipt": bool(
            (cmb_sources.get("finite_covariant_parent") or {}).get("parent_receipt", False)
        ),
        "finite_covariant_parent_blockers": list(
            ((cmb_sources.get("finite_covariant_parent") or {}).get("blockers") or [])
        )[:16],
        "oph_boltzmann_input_written": bool((cmb_sources.get("oph_boltzmann_input") or {}).get("written", False)),
        "oph_boltzmann_physical_prediction_ready": bool(
            (cmb_sources.get("oph_boltzmann_input") or {}).get("physical_prediction_ready", False)
        ),
        "finite_collar_boltzmann_bundle_receipt": bool(
            (cmb_sources.get("finite_collar_boltzmann_bundle") or {}).get("source_bundle_receipt", False)
        ),
        "finite_collar_boltzmann_physical_certificate": bool(
            (cmb_sources.get("finite_collar_boltzmann_bundle") or {}).get("physical_certificate", False)
        ),
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


def _write_physical_cmb_source_artifacts(run_dir: Path) -> dict[str, Any]:
    """Emit fail-closed source reports consumed by the physical-CMB contract.

    These builders expose missing source evidence explicitly. They do not turn
    diagnostic/proxy artifacts into physical CMB predictions.
    """

    run_dir = Path(run_dir)
    summary: dict[str, Any] = {
        "finite_repair_transition_clock_written": False,
        "finite_certificate_report_written": False,
        "B_A_kernel_report_written": False,
        "compressed_likelihood_reference_written": False,
        "official_planck_likelihood_readiness_written": False,
        "physical_cmb_source_artifact_errors": [],
    }

    def record_error(name: str, exc: Exception) -> None:
        summary["physical_cmb_source_artifact_errors"].append(f"{name}: {type(exc).__name__}: {exc}")

    if not (run_dir / "finite_repair_transition_matrix_report.json").exists():
        try:
            write_finite_repair_transition_clock_report(run_dir, run_dir)
            summary["finite_repair_transition_clock_written"] = True
        except Exception as exc:  # pragma: no cover - defensive pipeline audit path
            record_error("finite_repair_transition_clock", exc)
    else:
        summary["finite_repair_transition_clock_written"] = True

    if not (run_dir / "finite_certificate_report.json").exists():
        try:
            write_run_proxy_finite_certificate_bundle(run_dir, run_dir)
            summary["finite_certificate_report_written"] = True
        except Exception as exc:  # pragma: no cover - defensive pipeline audit path
            record_error("finite_certificate", exc)
    else:
        summary["finite_certificate_report_written"] = True

    if not (run_dir / "B_A_kernel_report.json").exists() and (run_dir / "b_a_parent_report.json").exists():
        try:
            ba_kernel_report_from_parent_report(run_dir / "b_a_parent_report.json", run_dir)
            summary["B_A_kernel_report_written"] = True
        except Exception as exc:  # pragma: no cover - defensive pipeline audit path
            record_error("B_A_kernel", exc)
    elif (run_dir / "B_A_kernel_report.json").exists():
        summary["B_A_kernel_report_written"] = True

    if not (run_dir / "oph_compressed_likelihood_report.json").exists():
        try:
            write_compressed_likelihood_reference_report(run_dir)
            summary["compressed_likelihood_reference_written"] = True
        except Exception as exc:  # pragma: no cover - defensive pipeline audit path
            record_error("compressed_likelihood_reference", exc)
    else:
        summary["compressed_likelihood_reference_written"] = True

    if not (run_dir / "official_planck_likelihood_readiness_report.json").exists():
        try:
            write_official_planck_readiness_report(run_dir)
            summary["official_planck_likelihood_readiness_written"] = True
        except Exception as exc:  # pragma: no cover - defensive pipeline audit path
            record_error("official_planck_likelihood_readiness", exc)
    else:
        summary["official_planck_likelihood_readiness_written"] = True

    return summary


def _write_physical_cmb_transfer_artifacts(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Emit physical-unit CAMB comparison rows when benchmark inputs exist.

    These reports provide data for plots and residual tables. They do not
    certify a physical OPH CMB prediction; promotion remains controlled by the
    physical input/promotion gates.
    """

    run_dir = Path(run_dir)
    cosmology_cfg = dict(config.get("cosmology", {}) or {})
    lite_cfg = dict(cosmology_cfg.get("cmb_lite", {}) or {})
    transfer_cfg = dict(cosmology_cfg.get("physical_cmb_transfer", {}) or {})
    benchmark = Path(
        transfer_cfg.get("benchmark_path")
        or lite_cfg.get("benchmark_path")
        or "data/measurements/planck2018/COM_PowerSpect_CMB-TT-binned_R3.01.txt"
    )
    label = str(transfer_cfg.get("benchmark_label") or lite_cfg.get("benchmark_label") or "Planck2018_TT_binned")
    lmax = int(transfer_cfg.get("lmax", 2600) or 2600)
    summary: dict[str, Any] = {
        "physical_cmb_transfer_benchmark_path": str(benchmark),
        "physical_cmb_transfer_benchmark_found": benchmark.exists(),
        "camb_lcdm_baseline_written": False,
        "scale_compressed_cmb_camb_written": False,
        "finite_repair_clock_cmb_camb_written": False,
        "physical_cmb_transfer_errors": [],
    }
    if not benchmark.exists():
        summary["physical_cmb_transfer_errors"].append(f"benchmark_missing:{benchmark}")
        return summary

    def record_error(name: str, exc: Exception) -> None:
        summary["physical_cmb_transfer_errors"].append(f"{name}: {type(exc).__name__}: {exc}")

    if not (run_dir / "camb_lcdm_baseline_report.json").exists():
        try:
            write_camb_lcdm_baseline_report(
                benchmark,
                run_dir,
                lmax=lmax,
                benchmark_label=label,
            )
            summary["camb_lcdm_baseline_written"] = True
        except Exception as exc:  # pragma: no cover - depends on optional CAMB runtime
            record_error("camb_lcdm_baseline", exc)
    else:
        summary["camb_lcdm_baseline_written"] = True

    scale_report = run_dir / "scale_compressed_repair_report.json"
    if scale_report.exists():
        if not (run_dir / "scale_compressed_cmb_camb_report.json").exists():
            try:
                write_scale_compressed_cmb_camb_report(
                    scale_report,
                    benchmark,
                    run_dir,
                    lmax=lmax,
                    benchmark_label=label,
                )
                summary["scale_compressed_cmb_camb_written"] = True
            except Exception as exc:  # pragma: no cover - depends on optional CAMB runtime
                record_error("scale_compressed_cmb_camb", exc)
        else:
            summary["scale_compressed_cmb_camb_written"] = True

    finite_clock_report = run_dir / "finite_repair_transition_matrix_report.json"
    if finite_clock_report.exists():
        if not (run_dir / "finite_repair_clock_cmb_camb_report.json").exists():
            try:
                write_finite_repair_clock_cmb_camb_report(
                    finite_clock_report,
                    benchmark,
                    run_dir,
                    source_dir=run_dir,
                    lmax=lmax,
                    benchmark_label=label,
                )
                summary["finite_repair_clock_cmb_camb_written"] = True
            except Exception as exc:  # pragma: no cover - depends on optional CAMB runtime
                record_error("finite_repair_clock_cmb_camb", exc)
        else:
            summary["finite_repair_clock_cmb_camb_written"] = True

    return summary


def _write_visualization_diagnostic_artifacts(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config.get("visualization_diagnostics", {}) or {})
    vacuum_cfg = dict(cfg.get("reference_vacuum", {}) or {})
    gravity_cfg = dict(cfg.get("two_defect_gravity_assay", {}) or {})
    organic_cfg = dict(cfg.get("organic_defect_population", {}) or {})
    ym_cfg = dict(cfg.get("yang_mills_gap_certificate", {}) or {})

    write_vacuum = bool(vacuum_cfg.get("enabled", True))
    write_gravity = bool(gravity_cfg.get("enabled", True))
    write_organic = bool(organic_cfg.get("enabled", True))
    write_ym = bool(ym_cfg.get("enabled", True))
    summary: dict[str, Any] = {
        "reference_vacuum_baseline_written": False,
        "yang_mills_gap_certificate_written": False,
        "organic_defect_population_written": False,
        "two_defect_stress_contraction_assay_written": False,
        "free_two_defect_dynamics_written": False,
        "raw_production_gravity_receipt": False,
        "production_gravity_receipt": False,
        "physical_gravity_prediction": False,
        "einstein_branch_entry_contract_receipt": False,
        "einstein_branch_entry_issue": 503,
    }
    if write_vacuum:
        vacuum_report = write_reference_vacuum_baseline_report(
            run_dir / "reference_vacuum_baseline",
            ell_max=_positive_int(vacuum_cfg.get("ell_max"), 16),
            sample_count=_positive_int(vacuum_cfg.get("sample_count"), 256),
            amplitude=_positive_float(vacuum_cfg.get("amplitude"), 1.0),
            theta=float(vacuum_cfg.get("theta", 0.0) or 0.0),
            seed_key=str(vacuum_cfg.get("seed_key", "oph-universe-reference-vacuum")),
            smoothing_sigma=_optional_float(vacuum_cfg.get("smoothing_sigma")),
            coarse_ell_max=_optional_int(vacuum_cfg.get("coarse_ell_max")),
            u1_lattice_size=_positive_int(vacuum_cfg.get("u1_lattice_size"), 4),
            u1_sweeps=_positive_int(vacuum_cfg.get("u1_sweeps"), 32),
            u1_beta=_positive_float(vacuum_cfg.get("u1_beta"), 0.5),
            u1_step_size=_positive_float(vacuum_cfg.get("u1_step_size"), math.pi / 2.0),
        )
        summary.update(
            {
                "reference_vacuum_baseline_written": True,
                "reference_vacuum_baseline_path": vacuum_report.get("report_path"),
                "reference_vacuum_regression_receipt": bool(
                    (vacuum_report.get("receipt_contract") or {}).get("reference_theory_regression", False)
                ),
                "oph_native_vacuum_promotion_receipt": bool(
                    vacuum_report.get("OPH_NATIVE_VACUUM_PROMOTION_RECEIPT", False)
                ),
                "oph_primordial_field_promotion_receipt": bool(
                    vacuum_report.get("OPH_PRIMORDIAL_FIELD_PROMOTION_RECEIPT", False)
                ),
            }
        )
    if write_ym:
        ym_report = write_yang_mills_gap_certificate_report(
            run_dir / "yang_mills_gap_certificate_report.json",
            lattice_size=_positive_int(ym_cfg.get("lattice_size"), 2),
            sweeps=_positive_int(ym_cfg.get("sweeps"), 16),
            beta=_positive_float(ym_cfg.get("beta"), 2.2),
            proposal_width=_positive_float(ym_cfg.get("proposal_width"), 0.35),
            seed=_positive_int(ym_cfg.get("seed"), 20260706),
            transition_bins=_positive_int(ym_cfg.get("transition_bins"), 8),
            refinement_lattice_sizes=_positive_int_tuple(
                ym_cfg.get("refinement_lattice_sizes"),
                (2, 3),
            ),
            refinement_sweeps=_optional_int(ym_cfg.get("refinement_sweeps")),
            continuum_certificate=(
                ym_cfg.get("continuum_certificate")
                if isinstance(ym_cfg.get("continuum_certificate"), dict)
                else None
            ),
        )
        summary.update(
            {
                "yang_mills_gap_certificate_written": True,
                "finite_nonabelian_gauge_gap_diagnostic_receipt": bool(
                    ym_report.get("finite_nonabelian_gauge_gap_diagnostic_receipt", False)
                ),
                "finite_repair_gap_proxy_receipt": bool(
                    ym_report.get("finite_repair_gap_proxy_receipt", False)
                ),
                "yang_mills_gap_reproduced_receipt": bool(
                    ym_report.get("YANG_MILLS_GAP_REPRODUCED_RECEIPT", False)
                ),
                "clay_yang_mills_gap_receipt": bool(
                    ym_report.get("CLAY_YANG_MILLS_GAP_RECEIPT", False)
                ),
                "yang_mills_identification": ym_report.get("yang_mills_identification"),
            }
        )
    if write_organic:
        organic_report = write_organic_defect_population_report(
            run_dir / "organic_defect_population_report.json",
            patch_count=_positive_int(
                organic_cfg.get("patch_count"),
                _positive_int(gravity_cfg.get("patch_count"), 65_536),
            ),
            steps=_positive_int(
                organic_cfg.get("steps"),
                _positive_int(gravity_cfg.get("free_steps", gravity_cfg.get("steps")), 128),
            ),
            defect_count=_positive_int(organic_cfg.get("defect_count"), 16),
            min_defects=_positive_int(organic_cfg.get("min_defects"), 10),
            max_defects=_positive_int(organic_cfg.get("max_defects"), 20),
            support_node_count=_positive_int(
                organic_cfg.get("support_node_count"),
                _positive_int(gravity_cfg.get("support_node_count"), 8),
            ),
            seed=_positive_int(
                organic_cfg.get("seed"),
                _positive_int(gravity_cfg.get("free_seed", gravity_cfg.get("seed")), 2039),
            ),
            initial_speed=_positive_float(
                organic_cfg.get("initial_speed"),
                _positive_float(gravity_cfg.get("initial_speed"), 0.028),
            ),
            stress_coupling=_positive_float(
                organic_cfg.get("stress_coupling"),
                _positive_float(gravity_cfg.get("free_stress_coupling"), 0.018),
            ),
            transverse_kick=_positive_float(
                organic_cfg.get("transverse_kick"),
                _positive_float(gravity_cfg.get("transverse_kick"), 0.010),
            ),
            stress_radius=_positive_float(
                organic_cfg.get("stress_radius"),
                _positive_float(gravity_cfg.get("stress_radius"), 0.9),
            ),
            curvature_radius=_positive_float(
                organic_cfg.get("curvature_radius"),
                _positive_float(gravity_cfg.get("curvature_radius"), 1.0),
            ),
            cycle_stride=_positive_int(
                organic_cfg.get("cycle_stride"),
                _positive_int(gravity_cfg.get("cycle_stride"), 1),
            ),
            contact_radius=_positive_float(
                organic_cfg.get("contact_radius"),
                _positive_float(gravity_cfg.get("contact_radius"), 0.12),
            ),
            overlap_radius=_positive_float(
                organic_cfg.get("overlap_radius"),
                _positive_float(gravity_cfg.get("overlap_radius"), 0.28),
            ),
            spawn_radius=_positive_float(organic_cfg.get("spawn_radius"), 1.25),
        )
        organic_summary = (
            organic_report.get("organic_population_summary")
            if isinstance(organic_report.get("organic_population_summary"), dict)
            else {}
        )
        summary.update(
            {
                "organic_defect_population_written": True,
                "organic_defect_population_receipt": bool(
                    organic_report.get("organic_defect_population_receipt", False)
                ),
                "organic_proto_worldline_visualization_receipt": bool(
                    organic_report.get("organic_proto_worldline_visualization_receipt", False)
                ),
                "organic_defect_worldline_count": organic_summary.get("worldline_count"),
                "organic_defect_near_contact_event_count": organic_summary.get("near_contact_event_count"),
                "raw_production_gravity_receipt": bool(
                    summary.get("raw_production_gravity_receipt", False)
                    or organic_report.get("production_gravity_receipt", False)
                ),
                "raw_physical_gravity_prediction": bool(
                    summary.get("raw_physical_gravity_prediction", False)
                    or organic_report.get("physical_gravity_prediction", False)
                ),
                "production_gravity_receipt": False,
                "physical_gravity_prediction": False,
            }
        )
    if write_gravity:
        gravity_report = write_two_defect_stress_contraction_assay_report(
            run_dir / "two_defect_stress_contraction_assay_report.json",
            patch_count=_positive_int(gravity_cfg.get("patch_count"), 65_536),
            steps=_positive_int(gravity_cfg.get("steps"), 64),
            support_node_count=_positive_int(gravity_cfg.get("support_node_count"), 8),
            holonomy=_positive_int(gravity_cfg.get("holonomy"), 1),
            initial_separation=_positive_float(gravity_cfg.get("initial_separation"), 1.2),
            stress_coupling=_positive_float(gravity_cfg.get("stress_coupling"), 0.04),
            stress_radius=_positive_float(gravity_cfg.get("stress_radius"), 1.0),
            curvature_radius=_positive_float(gravity_cfg.get("curvature_radius"), 1.0),
            cycle_stride=_positive_int(gravity_cfg.get("cycle_stride"), 1),
            min_approach_fraction=_positive_float(gravity_cfg.get("min_approach_fraction"), 0.25),
            min_control_margin=_positive_float(gravity_cfg.get("min_control_margin"), 0.15),
        )
        summary.update(
            {
                "two_defect_stress_contraction_assay_written": True,
                "two_defect_stress_contraction_assay_receipt": bool(
                    gravity_report.get("two_defect_stress_contraction_assay_receipt", False)
                ),
                "gravity_like_attraction_diagnostic_receipt": bool(
                    gravity_report.get("gravity_like_attraction_diagnostic_receipt", False)
                ),
                "raw_production_gravity_receipt": bool(
                    summary.get("raw_production_gravity_receipt", False)
                    or gravity_report.get("production_gravity_receipt", False)
                ),
                "raw_physical_gravity_prediction": bool(
                    summary.get("raw_physical_gravity_prediction", False)
                    or gravity_report.get("physical_gravity_prediction", False)
                ),
                "production_gravity_receipt": False,
                "physical_gravity_prediction": False,
            }
        )
        if bool(gravity_cfg.get("free_dynamics_enabled", True)):
            free_report = write_free_two_defect_dynamics_report(
                run_dir / "free_two_defect_dynamics_report.json",
                patch_count=_positive_int(gravity_cfg.get("patch_count"), 65_536),
                steps=_positive_int(gravity_cfg.get("free_steps", gravity_cfg.get("steps")), 96),
                support_node_count=_positive_int(gravity_cfg.get("support_node_count"), 8),
                holonomy=_positive_int(gravity_cfg.get("holonomy"), 1),
                seed=_positive_int(gravity_cfg.get("free_seed", gravity_cfg.get("seed")), 1729),
                initial_separation=_positive_float(gravity_cfg.get("initial_separation"), 1.2),
                initial_speed=_positive_float(gravity_cfg.get("initial_speed"), 0.035),
                stress_coupling=_positive_float(gravity_cfg.get("free_stress_coupling"), 0.03),
                transverse_kick=_positive_float(gravity_cfg.get("transverse_kick"), 0.008),
                stress_radius=_positive_float(gravity_cfg.get("stress_radius"), 1.0),
                curvature_radius=_positive_float(gravity_cfg.get("curvature_radius"), 1.0),
                cycle_stride=_positive_int(gravity_cfg.get("cycle_stride"), 1),
                contact_radius=_positive_float(gravity_cfg.get("contact_radius"), 0.10),
                overlap_radius=_positive_float(gravity_cfg.get("overlap_radius"), 0.22),
                bind_speed_threshold=_positive_float(gravity_cfg.get("bind_speed_threshold"), 0.055),
                annihilation_overlap_threshold=_positive_float(
                    gravity_cfg.get("annihilation_overlap_threshold"), 0.85
                ),
            )
            free_summary = (
                free_report.get("free_dynamics_summary")
                if isinstance(free_report.get("free_dynamics_summary"), dict)
                else {}
            )
            summary.update(
                {
                    "free_two_defect_dynamics_written": True,
                    "free_two_defect_dynamics_receipt": bool(
                        free_report.get("free_two_defect_dynamics_receipt", False)
                    ),
                    "gravity_like_free_dynamics_diagnostic_receipt": bool(
                        free_report.get("gravity_like_free_dynamics_diagnostic_receipt", False)
                    ),
                    "free_two_defect_contact_outcome": free_summary.get("contact_outcome"),
                    "raw_production_gravity_receipt": bool(
                        summary.get("raw_production_gravity_receipt", False)
                        or free_report.get("production_gravity_receipt", False)
                    ),
                    "raw_physical_gravity_prediction": bool(
                        summary.get("raw_physical_gravity_prediction", False)
                        or free_report.get("physical_gravity_prediction", False)
                    ),
                    "production_gravity_receipt": False,
                    "physical_gravity_prediction": False,
                }
            )
    return summary


def _neutral_frontier_settings(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config.get("neutral_frontier", {}) or {})
    observer_cfg = dict(config.get("observers", {}) or {})
    observer_sample_count = int(observer_cfg.get("sample_count", 384) or 384)
    sample_count = int(cfg.get("sample_count", observer_sample_count) or observer_sample_count)
    max_model_points = int(cfg.get("max_model_points", min(sample_count, 512)) or min(sample_count, 512))
    graph_max = tuple(
        int(value)
        for value in cfg.get(
            "graph_max_model_points_values",
            sorted({min(max_model_points, 256), min(max_model_points, 384), max_model_points}),
        )
    )
    graph_max = tuple(value for value in graph_max if value > 0)
    requested_workers = cfg.get("graph_workers")
    if requested_workers is None:
        requested_workers = os.environ.get("OPH_FPE_GRAPH_SWEEP_WORKERS") or os.environ.get("OPH_FPE_CPUS")
    try:
        graph_workers = int(requested_workers) if requested_workers is not None else 1
    except (TypeError, ValueError):
        graph_workers = 1
    return {
        "seed": int(cfg.get("seed", 11) or 11),
        "sample_count": max(8, sample_count),
        "max_model_points": max(8, max_model_points),
        "strict_bulk_max_model_points": max(
            8,
            int(cfg.get("strict_bulk_max_model_points", max_model_points) or max_model_points),
        ),
        "planted_control_points": max(16, int(cfg.get("planted_control_points", 160) or 160)),
        "object_min_objects": max(1, int(cfg.get("object_min_objects", 16) or 16)),
        "object_min_observers_per_object": max(
            1,
            int(cfg.get("object_min_observers_per_object", 3) or 3),
        ),
        "object_max_observer_fraction_per_object": max(
            0.01,
            min(1.0, float(cfg.get("object_max_observer_fraction_per_object", 0.65) or 0.65)),
        ),
        "object_max_model_points": max(
            8,
            int(cfg.get("object_max_model_points", min(max_model_points, 192)) or min(max_model_points, 192)),
        ),
        "object_heldout_fraction": max(
            0.01,
            min(0.95, float(cfg.get("object_heldout_fraction", 0.25) or 0.25)),
        ),
        "graph_workers": max(1, graph_workers),
        "overlap_control_max_model_points": max(
            8,
            int(cfg.get("overlap_control_max_model_points", max(max_model_points, 384)) or max(max_model_points, 384)),
        ),
        "ranks": tuple(
            int(value)
            for value in cfg.get("ranks", (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
            if int(value) > 0
        ),
        "graph_seeds": tuple(int(value) for value in cfg.get("graph_seeds", (7, 11))),
        "graph_max_model_points_values": graph_max or (max(8, max_model_points),),
        "graph_k_neighbor_values": tuple(
            int(value) for value in cfg.get("graph_k_neighbor_values", (8, 12, 16)) if int(value) > 0
        ),
        "residual_remove_mode_values": tuple(
            int(value) for value in cfg.get("residual_remove_mode_values", (1, 2, 3)) if int(value) > 0
        ),
    }


def _visualization_export_settings(
    config: dict[str, Any],
    *,
    max_screen_points: int,
    max_observers: int,
    max_h3_objects: int,
) -> dict[str, int]:
    export_cfg = dict(config.get("visualization_export", {}) or {})
    readout_cfg = dict(config.get("observer_consensus_readout", {}) or {})
    observer_cfg = dict(config.get("observers", {}) or {})

    observer_sample_count = _positive_int(observer_cfg.get("sample_count"), max_observers)
    effective_max_screen_points = _positive_int(export_cfg.get("max_screen_points"), max_screen_points)
    effective_max_observers = min(
        observer_sample_count,
        _positive_int(export_cfg.get("max_observers"), max_observers),
    )
    effective_max_h3_objects = _positive_int(export_cfg.get("max_h3_objects"), max_h3_objects)

    readout_observers_default = min(observer_sample_count, max(12, effective_max_observers))
    readout_objects_default = max(24, effective_max_h3_objects)
    readout_observers = min(
        observer_sample_count,
        _positive_int(readout_cfg.get("observer_sample_count"), readout_observers_default),
    )
    readout_objects = _positive_int(readout_cfg.get("object_sample_count"), readout_objects_default)
    return {
        "max_screen_points": effective_max_screen_points,
        "max_observers": effective_max_observers,
        "max_h3_objects": effective_max_h3_objects,
        "readout_observer_sample_count": readout_observers,
        "readout_object_sample_count": readout_objects,
    }


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(1, parsed)


def _positive_int_tuple(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return tuple(max(1, int(item)) for item in default)
    if isinstance(value, str):
        items: list[Any] = [item.strip() for item in value.split(",") if item.strip()]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    parsed = tuple(_positive_int(item, default[0] if default else 1) for item in items)
    return parsed or tuple(max(1, int(item)) for item in default)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return parsed if math.isfinite(parsed) and parsed > 0.0 else float(default)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _cmb_diagnostic_summary(run_dir: Path) -> dict[str, Any]:
    emergence = _read_json(Path(run_dir) / "emergence_status_report.json")
    cl_report = _read_json(Path(run_dir) / "cl_comparison_report.json")
    lite_report = _read_json(Path(run_dir) / "cmb_lite_comparison_report.json")
    fields = lite_report.get("field_comparisons") if isinstance(lite_report.get("field_comparisons"), dict) else {}
    positive_shape_fields = [
        str(name)
        for name, row in fields.items()
        if isinstance(row, dict) and bool(row.get("usable_positive_shape", False))
    ]
    real_ell_fields = [
        str(name)
        for name, row in fields.items()
        if isinstance(row, dict)
        and bool((row.get("real_ell_physical_comparison") or {}).get("usable", False))
    ]
    overlap_ell_fields = [
        str(name)
        for name, row in fields.items()
        if isinstance(row, dict)
        and bool((row.get("overlap_ell_physical_comparison") or {}).get("usable", False))
    ]
    return {
        "mode": "cmb_diagnostic_summary_v0",
        "screen_proxy_cmb_receipt": bool(
            cl_report.get("SCREEN_PROXY_CMB_RECEIPT", False)
            or cl_report.get("screen_proxy_cmb_receipt", False)
            or (
                cl_report.get("receipt_name") == "SCREEN_PROXY_CMB_RECEIPT"
                and (cl_report.get("cosmo_proxy_receipt") or {}).get("receipt", False)
            )
            or emergence.get("SCREEN_PROXY_CMB_RECEIPT", False)
            or emergence.get("screen_proxy_cmb_receipt", False)
        ),
        "cmb_lite_shape_comparison_receipt": bool(lite_report.get("best_shape_field")),
        "cmb_lite_best_shape_field": lite_report.get("best_shape_field"),
        "cmb_lite_best_positive_shape_field": lite_report.get("best_positive_shape_field"),
        "cmb_lite_positive_shape_field_count": len(positive_shape_fields),
        "cmb_lite_positive_shape_fields_sample": positive_shape_fields[:12],
        "cmb_lite_real_ell_physical_comparison_receipt": bool(real_ell_fields),
        "cmb_lite_real_ell_fields_sample": real_ell_fields[:12],
        "cmb_lite_overlap_ell_physical_comparison_receipt": bool(overlap_ell_fields),
        "cmb_lite_overlap_ell_fields_sample": overlap_ell_fields[:12],
        "claim_boundary": (
            "Screen/CMB-lite diagnostics only. Shape comparisons and finite-screen angular spectra are "
            "measurement-facing debug data, not a physical CMB prediction and not a substitute for the "
            "finite source, Boltzmann-transfer, and frozen-likelihood gates."
        ),
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
    }
    populated_h3_component_gates = {
        **component_gates,
        "observer_h3_object_population_receipt": object_population,
    }
    observer_3p1d = bool(all(component_gates.values()))
    populated_h3 = bool(all(populated_h3_component_gates.values()))
    report = dict(original)
    report.update(
        {
            "postprocessed_by_oph_universe_pipeline": True,
            "component_gates": component_gates,
            "blockers": [name for name, passed in component_gates.items() if not passed],
            "populated_h3_component_gates": populated_h3_component_gates,
            "populated_h3_experience_blockers": [
                name for name, passed in populated_h3_component_gates.items() if not passed
            ],
            OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT: observer_3p1d,
            "observer_facing_3p1d_h3_experience_receipt": observer_3p1d,
            "observer_facing_populated_h3_experience_receipt": populated_h3,
            "observer_h3_object_population_receipt": object_population,
            "claim_boundary": (
                str(original.get("claim_boundary", "")).strip()
                + " Postprocessed after canonical H3 refit/object-chart selection; this only updates "
                "downstream receipt wiring from selected audited reports and does not relax thresholds. "
                "Observer-facing 3+1D/H3 experience is split from populated-H3 object emergence."
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
    excluded_observables = _h3_excluded_observables(report)
    theorem_clean_feature_policy = set(H3_RESPONSE_EXCLUDED_AUX_OBSERVABLES) <= set(excluded_observables)
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
        "material_wrong_scale_gate_metric": gates.get("material_wrong_scale_gate_metric"),
        "material_wrong_scale_gate_value": _maybe_float(gates.get("material_wrong_scale_gate_value")),
        "feature_count": int(report.get("feature_count", 0) or 0),
        "excluded_observables": excluded_observables,
        "theorem_clean_feature_policy": theorem_clean_feature_policy,
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
    material = row.get("material_wrong_scale_gate_value")
    if material is None:
        material = row.get("material_wrong_scale_win_fraction")
    return (
        int(row["candidate_receipt"]),
        int(row.get("theorem_clean_feature_policy", False)),
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


def _h3_excluded_observables(report: dict[str, Any]) -> list[str]:
    feature_selection = report.get("feature_selection") if isinstance(report, dict) else {}
    if isinstance(feature_selection, dict):
        values = feature_selection.get("exclude_observables", [])
    else:
        values = []
    return sorted({str(value) for value in values if str(value)})


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
