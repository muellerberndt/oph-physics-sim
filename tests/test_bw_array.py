from pathlib import Path
from math import isclose, pi

import json
import numpy as np

from oph_fpe.experiments import load_config
from oph_fpe.scale.bw_array import (
    _attach_modular_response_histograms,
    _attach_transition_history_histograms,
    _array_port_pair_consensus_replay_report,
    _collar_report,
    _collar_width_from_config,
    _drop_source_snapshot_from_history,
    _interface_quotient,
    _lorentz_branch_receipts,
    _repairs_per_cycle_from_config,
    run_bw_array_config,
)
from oph_fpe.viz.universe_timeline_viewer import _read_proto_particle_candidates


def test_repair_fraction_per_cycle_scales_with_patch_count():
    assert _repairs_per_cycle_from_config(
        {"repair_fraction_per_cycle": 0.0625, "repairs_per_cycle": 65_536},
        patch_count=4096,
        edge_count=49_152,
    ) == 256
    assert _repairs_per_cycle_from_config(
        {"repair_fraction_per_cycle": 0.0625, "repairs_per_cycle": 65_536},
        patch_count=65_536,
        edge_count=786_432,
    ) == 4096
    assert _repairs_per_cycle_from_config({"repairs_per_cycle": 512}, patch_count=4096, edge_count=49_152) == 512


def test_bw_array_writes_bw_report(tmp_path: Path):
    config = load_config(Path("configs/e1_s3_bw_screen_64k.yml"))
    config = dict(config)
    config["run_id"] = "bw_array_smoke"
    config["graph"] = dict(config["graph"], patch_count=512, neighbors=8)
    config["dynamics"] = dict(config["dynamics"], cycles=8, repairs_per_cycle=512)
    config["bw"] = dict(config["bw"], cap_count=4, times=[0.025, 0.05], n_jobs=2)
    config["cosmology"] = {
        "freezeout": {
            "enabled": True,
            "commit_fraction": 0.75,
            "fields": ["record_signature", "cumulative_repair_load", "s3_class_density"],
            "require_kms_bw_pass": False,
            "require_state_bw_controls": False,
        },
        "angular_power": {"ell_max": 12, "pair_samples": 4096, "controls": ["shuffled_field"]},
        "harmonic_time_trace": {
            "enabled": True,
            "sample_count": 3,
            "ell_max": 4,
            "fields": ["record_signature", "stable_count"],
            "harmonic_batch_size": 128,
        },
    }

    result = run_bw_array_config(config, tmp_path)
    run_path = Path(result["path"])

    assert result["final_phi"] >= 0
    assert result["bw_median"] >= 0
    assert (run_path / "bw_report.json").exists()
    assert (run_path / "cap_geometry_report.json").exists()
    assert (run_path / "conformal_h3_spatial_chart_report.json").exists()
    assert (run_path / "record_populated_h3_report.json").exists()
    assert (run_path / "record_family_h3_report.json").exists()
    assert (run_path / "defect_cluster_h3_report.json").exists()
    assert (run_path / "observer_views.jsonl").exists()
    assert (run_path / "observer_consensus_report.json").exists()
    assert (run_path / "mandatory_controls_report.json").exists()
    assert (run_path / "screen_ports.json").exists()
    assert (run_path / "boundary_program_report.json").exists()
    assert (run_path / "emergence_status_report.json").exists()
    assert (run_path / "screen_microphysics.json").exists()
    assert (run_path / "base_progress.json").exists()
    assert (run_path / "edge_sector_heat_kernel_report.json").exists()
    assert (run_path / "central_record_born_report.json").exists()
    assert (run_path / "observer_checkpoint_restoration_report.json").exists()
    assert (run_path / "observer_modular_experience_report.json").exists()
    assert (run_path / "array_holonomy_report.json").exists()
    assert (run_path / "freezeout_fields.npz").exists()
    assert (run_path / "freezeout_map_summary.json").exists()
    assert (run_path / "cl_proxy.csv").exists()
    assert (run_path / "cl_controls.csv").exists()
    assert (run_path / "cl_comparison_report.json").exists()
    assert (run_path / "harmonic_time_trace.npz").exists()
    assert (run_path / "harmonic_time_trace_report.json").exists()

    bw_report = json.loads((run_path / "bw_report.json").read_text(encoding="utf-8"))
    cap_report = json.loads((run_path / "cap_geometry_report.json").read_text(encoding="utf-8"))
    chart_report = json.loads((run_path / "conformal_h3_spatial_chart_report.json").read_text(encoding="utf-8"))
    h3_population_report = json.loads((run_path / "record_populated_h3_report.json").read_text(encoding="utf-8"))
    record_family_h3 = json.loads((run_path / "record_family_h3_report.json").read_text(encoding="utf-8"))
    defect_cluster_h3 = json.loads((run_path / "defect_cluster_h3_report.json").read_text(encoding="utf-8"))
    consensus_report = json.loads((run_path / "observer_consensus_report.json").read_text(encoding="utf-8"))
    emergence_status = json.loads((run_path / "emergence_status_report.json").read_text(encoding="utf-8"))
    cl_report = json.loads((run_path / "cl_comparison_report.json").read_text(encoding="utf-8"))
    harmonic_trace = np.load(run_path / "harmonic_time_trace.npz")
    observer_rows = [
        json.loads(line)
        for line in (run_path / "observer_views.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    base_progress = json.loads((run_path / "base_progress.json").read_text(encoding="utf-8"))
    ports_report = json.loads((run_path / "screen_ports.json").read_text(encoding="utf-8"))
    boundary_report = json.loads((run_path / "boundary_program_report.json").read_text(encoding="utf-8"))
    holonomy_report = json.loads((run_path / "array_holonomy_report.json").read_text(encoding="utf-8"))
    edge_sector = json.loads((run_path / "edge_sector_heat_kernel_report.json").read_text(encoding="utf-8"))
    born = json.loads((run_path / "central_record_born_report.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((run_path / "observer_checkpoint_restoration_report.json").read_text(encoding="utf-8"))
    theorem = json.loads((run_path / "theorem_core_receipts.json").read_text(encoding="utf-8"))
    experience = json.loads((run_path / "observer_modular_experience_report.json").read_text(encoding="utf-8"))

    assert bw_report["rows"][0]["weight_measure"] == "cell_entropy_capacity"
    assert isclose(bw_report["rows"][0]["target_scale"], 2.0 * pi)
    assert isclose(bw_report["rows"][0]["sim_scale"], 2.0 * pi)

    assert bw_report["rows"][0]["cap_area_planck"] > 0.0
    assert bw_report["rows"][0]["cap_entropy_capacity"] > 0.0
    assert cap_report["weight_measure"] == "cell_entropy_capacity"
    assert chart_report["conformal_h3_spatial_chart_receipt"] is True
    assert "record_populated_h3_receipt" in chart_report
    assert h3_population_report["mode"] == "record_populated_h3_fit"
    assert h3_population_report["source_state"] in {"freezeout", "final", "repair_peak"}
    assert "source_report" in h3_population_report
    assert h3_population_report["record_populated_h3_receipt"] is chart_report["record_populated_h3_receipt"]
    assert record_family_h3["mode"] == "support_profile_h3_fit"
    assert defect_cluster_h3["mode"] == "support_profile_h3_fit"
    assert cap_report["regulator_collar"]["mode"] == "cell_scaled"
    assert cap_report["regulator_collar"]["collar_width"] > 0.0
    assert cap_report["regulator_collar"]["collar_to_cell_ratio"] > 0.0
    assert cap_report["regulator_collar"]["double_scaling_delta_to_zero"] is False
    assert cap_report["caps"][0]["soft_cap_area_planck"] > 0.0
    assert cap_report["caps"][0]["soft_cap_entropy_capacity"] > 0.0
    assert observer_rows
    assert {row["view_type"] for row in observer_rows} == {"patch_observer", "cap_observer"}
    patch_rows = [row for row in observer_rows if row["view_type"] == "patch_observer"]
    assert "modular_depth_mean" in patch_rows[0]
    assert "observer_relative_times" in patch_rows[0]
    assert experience["observer_modular_time_receipt"] is True
    assert experience["OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT"] is False
    assert "h3_modular_response_receipt" in experience["blockers"]
    assert consensus_report["observer_count"] > 0
    assert consensus_report["global_committed_fraction"] >= 0.0
    assert emergence_status["bulk_3d_established"] is False
    assert emergence_status["final_phi_zero"] is (result["final_phi"] == 0)
    assert emergence_status["FINITE_SETTLE_DIAGNOSTIC_RECEIPT"] is (result["final_phi"] == 0)
    assert emergence_status["FINITE_CONSENSUS_THEOREM_RECEIPT"] is False
    assert "theorem_phase_repair_events" in emergence_status["finite_consensus_missing_evidence"]
    assert emergence_status["requires_refinement_scaling"] is True
    assert theorem["finite_settle_diagnostic"]["canonical_tier"] == "C0a"
    assert theorem["finite_consensus_theorem"]["FINITE_CONSENSUS_THEOREM_RECEIPT"] is False
    assert theorem["finite_consensus_theorem_receipt"] is False
    assert "observer_consensus" in manifest
    assert "mandatory_controls" in manifest
    assert "emergence_status" in manifest
    assert manifest["base_loop_elapsed_seconds"] >= 0.0
    assert base_progress["stage"] == "base_repair_loop_complete"
    assert base_progress["completed_cycles"] == 8
    assert "cosmology_observables" in manifest
    assert manifest["theorem_core_receipts"]["finite_consensus_theorem_receipt"] is False
    assert manifest["observer_modular_experience"]["observer_modular_time_receipt"] is True
    assert "harmonic_time_trace" in manifest
    assert harmonic_trace["cycles"].shape[0] == 3
    assert harmonic_trace["ell"].shape[0] == 5
    assert harmonic_trace["record_signature"].shape == (3, 5)
    assert harmonic_trace["stable_count"].shape == (3, 5)
    assert manifest["screen_units"]["mode"] == "numerical_regulator"
    assert bw_report["support_visible_regularization"]["steps"] == 4
    assert ports_report["port_names"][0] == "P0"
    assert boundary_report["mode"] in {"iid_hot", "support_visible_cap_net_hot"}
    assert "boundary_program" in manifest
    assert holonomy_report["mode"] == "array_s3_screen_holonomy"
    assert holonomy_report["claim_boundary"].startswith("screen/collar S3 holonomy")
    assert "screen_holonomy" in manifest
    assert "fixed_cutoff_microphysics_receipts" in manifest
    assert edge_sector["mode"] == "edge_sector_heat_kernel_casimir_surrogate"
    assert born["mode"] == "central_record_born_surface"
    assert checkpoint["mode"] == "observer_checkpoint_restoration"
    assert cl_report["claim_boundary"].startswith("screen-only")
    assert cl_report["ell_max"] == 12
    assert "record_signature" in cl_report["fields"]


def test_bw_array_writes_organic_defect_population_before_two_defect_fallback(tmp_path: Path):
    config = load_config(Path("configs/e1_s3_bw_screen_64k.yml"))
    config = dict(config)
    config["run_id"] = "bw_array_organic_defect_smoke"
    config["graph"] = dict(config["graph"], patch_count=256, neighbors=6)
    config["dynamics"] = dict(config["dynamics"], cycles=6, repairs_per_cycle=256)
    config["bw"] = dict(config["bw"], cap_count=2, times=[0.025], n_jobs=1)
    config["observers"] = dict(config.get("observers", {}), sample_count=8, neighborhood_size=16)
    config["cosmology"] = {"freezeout": {"enabled": False}, "oph_cmb": {"enabled": False}}
    config["visualization_diagnostics"] = {
        "organic_defect_population": {
            "enabled": True,
            "patch_count": 256,
            "steps": 18,
            "defect_count": 12,
            "min_defects": 10,
            "max_defects": 14,
            "support_node_count": 4,
            "seed": 20260708,
        },
        "two_defect_gravity_assay": {
            "enabled": True,
            "free_dynamics_enabled": True,
            "patch_count": 256,
            "steps": 12,
            "free_steps": 16,
            "support_node_count": 4,
            "free_seed": 1729,
        },
    }

    result = run_bw_array_config(config, tmp_path)
    run_path = Path(result["path"])

    summary = json.loads((run_path / "visualization_defect_diagnostics_summary.json").read_text(encoding="utf-8"))
    organic = json.loads((run_path / "organic_defect_population_report.json").read_text(encoding="utf-8"))
    proto = _read_proto_particle_candidates(run_path, max_worldlines=24)

    assert summary["organic_defect_population_written"] is True
    assert summary["free_two_defect_dynamics_written"] is True
    assert summary["two_defect_stress_contraction_assay_written"] is True
    assert summary["selected_proto_worldline_preference"] == "organic_defect_population_report"
    assert result["visualization_defect_diagnostics"]["organic_defect_population_written"] is True
    assert organic["organic_defect_population_receipt"] is True
    assert organic["organic_population_summary"]["fixed_left_right_pair"] is False
    assert proto["worldlineSource"] == "organic_defect_population_report"
    assert proto["receipts"]["controlled_two_defect_worldline_count"] == 2
    assert proto["receipts"]["organic_defect_worldline_count"] >= 10
    assert all(
        not str(row["worldlineId"]).startswith(("stress_pair", "free_pair"))
        for row in proto["worldlines"]
    )


def test_transition_history_key_can_include_observer_visible_readout_hash():
    rows = [
        {
            "view_type": "patch_observer",
            "support_nodes": [0, 1],
            "visible_readout_hash": "sha256:aaaa1111",
        },
        {
            "view_type": "patch_observer",
            "support_nodes": [0, 1],
            "visible_readout_hash": "sha256:bbbb2222",
        },
    ]
    history = [
        {
            "record_signature": np.asarray([3, 3, 9, 9]),
            "stable_count": np.asarray([8, 8, 8, 8]),
            "committed_mask": np.asarray([1, 1, 1, 1]),
            "s3_sector_class": np.asarray([0, 0, 1, 1]),
            "repair_load": np.zeros(4),
            "cumulative_repair_load": np.zeros(4),
        }
    ] * 4

    base_cfg = {
        "transition_history_fields": ["record_family", "checkpoint_class", "stable_flag", "s3_sector_class"],
        "transition_history_key_fields": ["record_family", "checkpoint_class", "stable_flag", "s3_sector_class"],
        "transition_persistence_fields": ["record_family", "s3_sector_class"],
        "record_family_modulus": 16,
        "transition_bins": 8,
    }
    same_rows = [dict(row) for row in rows]
    _attach_transition_history_histograms(same_rows, history, base_cfg)
    assert same_rows[0]["transition_history_key"] == same_rows[1]["transition_history_key"]
    assert same_rows[0]["record_transition_histogram"]
    assert same_rows[0]["checkpoint_class_transition"]
    assert same_rows[0]["sector_change_signature"]
    assert "counterfactual_stability" in same_rows[0]

    split_rows = [dict(row) for row in rows]
    split_cfg = dict(
        base_cfg,
        transition_history_include_readout_hash=True,
        transition_history_readout_hash_prefix_chars=8,
    )
    _attach_transition_history_histograms(split_rows, history, split_cfg)
    assert split_rows[0]["transition_history_key"] != split_rows[1]["transition_history_key"]


def test_modular_response_histograms_attach_by_observer_id():
    rows = [
        {"view_type": "patch_observer", "observer_id": 10},
        {"view_type": "patch_observer", "observer_id": 20},
        {"view_type": "cap_observer", "observer_id": 10},
    ]
    kernel = {
        "matrix": np.asarray(
            [
                [1.0, 0.1, -0.2],
                [0.2, 1.1, 0.4],
            ],
            dtype=float,
        ),
        "observer_ids": [20, 10],
    }

    _attach_modular_response_histograms(
        rows,
        kernel,
        {"modular_response_cluster_components": 2, "modular_response_cluster_bins": 2},
    )

    for row in rows[:2]:
        histograms = row["modular_response_histograms"]
        assert "modular_response_cluster" in histograms
        assert "modular_response_component_0" in histograms
        assert sum(histograms["modular_response_cluster"].values()) == 1.0
        assert row["repair_response_spectrum"]
        assert row["perturb_resettle_signature"]
    assert "modular_response_histograms" not in rows[2]


def test_lorentz_branch_receipts_split_chart_automorphism_and_endogenous():
    chart = {"conformal_h3_spatial_chart_receipt": True}
    direct_state = {
        "direct_transition_automorphism": True,
        "state_selected_2pi": True,
        "correct_beats_controls": True,
        "endogenous_modular_generator": False,
    }
    transition = {
        "primary_source": "kms_collar_transport_response",
        "two_pi_selected": True,
        "response_degenerate": False,
    }

    direct_receipts = _lorentz_branch_receipts(chart, direct_state, transition)

    assert direct_receipts["CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT"] is True
    assert direct_receipts["BW_AUTOMORPHISM_SANITY_RECEIPT"] is True
    assert direct_receipts["BW_KMS_BRANCH_REPLAY_RECEIPT"] is True
    assert direct_receipts["OPH_LORENTZ_THEOREM_FINITE_CONTRACT_V1"] is False
    assert direct_receipts["finite_lorentz_theorem_contract_receipt"] is False
    assert direct_receipts["lorentz_receipt_taxonomy"] == "L0_branch_replay_not_L1_L7_finite_contract"
    assert direct_receipts["ENDOGENOUS_MODULAR_GENERATOR_RECEIPT"] is False
    assert direct_receipts["support_visible_lorentz_3p1_kinematics_receipt"] is True

    cap_flow_state = {
        "declared_cap_flow_generator": True,
        "state_selected_2pi": False,
        "correct_beats_controls": False,
    }
    cap_flow_receipts = _lorentz_branch_receipts(chart, cap_flow_state, {})

    assert cap_flow_receipts["CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT"] is True
    assert cap_flow_receipts["DECLARED_CAP_FLOW_GENERATOR_DIAGNOSTIC"] is True
    assert cap_flow_receipts["BW_AUTOMORPHISM_SANITY_RECEIPT"] is False
    assert cap_flow_receipts["BW_KMS_BRANCH_REPLAY_RECEIPT"] is False
    assert cap_flow_receipts["OPH_LORENTZ_THEOREM_FINITE_CONTRACT_V1"] is False
    assert cap_flow_receipts["support_visible_lorentz_3p1_kinematics_receipt"] is False


def test_double_scaling_collar_shrinks_but_expands_relative_to_uv():
    cfg = {
        "collar_width_mode": "double_scaling",
        "angular_prefactor": 0.8,
        "angular_exponent": 0.25,
    }

    width_4k = _collar_width_from_config(cfg, 4096)
    width_256k = _collar_width_from_config(cfg, 262144)
    report_4k = _collar_report(cfg, 4096, width_4k)
    report_256k = _collar_report(cfg, 262144, width_256k)

    assert isclose(width_4k, 0.8 * 4096.0 ** -0.25)
    assert width_256k < width_4k
    assert report_4k["double_scaling_delta_to_zero"] is True
    assert report_4k["double_scaling_collar_to_cell_diverges"] is True
    assert report_256k["collar_to_cell_ratio"] > report_4k["collar_to_cell_ratio"]


def test_state_history_excludes_current_source_snapshot():
    history = [
        {"cycle": 0, "committed_fraction": 0.0},
        {"cycle": 7, "committed_fraction": 0.1},
        {"cycle": 18, "committed_fraction": 0.95},
    ]

    trimmed = _drop_source_snapshot_from_history(history, {"cycle": 18})

    assert [row["cycle"] for row in trimmed] == [0, 7]


def test_array_port_pair_consensus_replay_emits_c0b_evidence():
    left = np.asarray([2, 1, 4, 3, 0], dtype=np.int16)
    right = np.asarray([1, 1, 0, 5, 0], dtype=np.int16)

    report = _array_port_pair_consensus_replay_report(
        left,
        right,
        group_order=6,
        config={"enabled": True, "schedule_replays": 4, "requested_schedule_replays": 4, "max_event_rows": 2},
        seed=17,
    )

    assert report["receipt"] is True
    assert report["evidence"]["theorem_phase_event_count"] == 3
    assert report["evidence"]["strict_descent_violation_count"] == 0
    assert report["evidence"]["unique_terminal_quotient_hash_count"] == 1
    assert len(report["sample_events"]) == 2
    assert all(event["phase"] == "theorem" for event in report["sample_events"])


def test_s3_interface_quotient_uses_group_product_not_index_subtraction():
    quotient = _interface_quotient(
        np.asarray([1], dtype=np.int16),
        np.asarray([4], dtype=np.int16),
        group_name="S3",
        group_order=6,
    )

    assert int(quotient[0]) == 5
    assert int(quotient[0]) != (1 - 4) % 6


def test_consensus_replay_executes_shared_node_ab_ba_diamonds():
    port_left = np.asarray([5, 4, 3, 2], dtype=np.int16)
    port_right = np.asarray([0, 1, 0, 1], dtype=np.int16)
    edge_left = np.asarray([0, 0, 1, 2], dtype=np.int64)
    edge_right = np.asarray([1, 2, 2, 3], dtype=np.int64)

    report = _array_port_pair_consensus_replay_report(
        port_left,
        port_right,
        edge_left=edge_left,
        edge_right=edge_right,
        group_order=6,
        config={
            "enabled": True,
            "schedule_replays": 3,
            "requested_schedule_replays": 3,
            "disjoint_checks": 4,
            "local_diamond_checks": 4,
        },
        seed=91,
    )

    assert report["receipt"] is True
    assert report["local_diamond_status"] == "computed_ab_ba_edge_slot_diamonds"
    assert report["local_diamond_checked_pair_count"] > 0
    assert report["shared_node_diamond_checked_pair_count"] > 0
    assert report["evidence"]["local_diamond_violation_count"] == 0
    assert "edge-slot quotient" in report["claim_boundary"]


def test_consensus_replay_cannot_disable_required_ab_ba_coverage():
    report = _array_port_pair_consensus_replay_report(
        np.asarray([2, 1, 4], dtype=np.int16),
        np.asarray([1, 0, 3], dtype=np.int16),
        group_order=6,
        config={
            "enabled": True,
            "schedule_replays": 2,
            "requested_schedule_replays": 2,
            "disjoint_checks": 0,
            "local_diamond_checks": 0,
        },
        seed=19,
    )

    assert report["receipt"] is False
    assert report["local_diamond_status"] == "required_ab_ba_checks_not_requested"


def test_state_modular_array_writes_gated_receipts(tmp_path: Path):
    config = load_config(Path("configs/e1_s3_state_modular_screen_4k.yml"))
    config = dict(config)
    config["run_id"] = "state_modular_smoke"
    config["graph"] = dict(config["graph"], patch_count=384, neighbors=8)
    config["dynamics"] = dict(config["dynamics"], cycles=10, repairs_per_cycle=384)
    config["bw"] = dict(config["bw"], cap_count=2, times=[0.025], n_jobs=1, max_basis=16)
    config["observers"] = dict(config["observers"], sample_count=16, neighborhood_size=16)
    config["observer_objects"] = dict(config["observer_objects"], max_families=64)
    config["neutral_reconstruction"] = {"enabled": True, "require_bw_refinement_pass": False}

    result = run_bw_array_config(config, tmp_path)
    run_path = Path(result["path"])

    state_report = json.loads((run_path / "bw_state_derived_report.json").read_text(encoding="utf-8"))
    status = json.loads((run_path / "emergence_status_report.json").read_text(encoding="utf-8"))
    mandatory = json.loads((run_path / "mandatory_controls_report.json").read_text(encoding="utf-8"))
    ports = json.loads((run_path / "screen_ports.json").read_text(encoding="utf-8"))

    assert result["bw_primary_mode"] == "state_derived_modular_probe"
    assert "geometric_controls" in result
    assert "state_bw_controls" in result
    assert result["state_bw_controls"] == state_report["controls"]
    assert result["state_bw_control_medians"] == state_report["control_medians"]
    assert result["state_bw_correct_beats_controls"] == state_report["correct_beats_controls"]
    assert (run_path / "collar_markov_report.json").exists()
    assert (run_path / "object_consensus_report.json").exists()
    assert (run_path / "bulk_reconstruction_report.json").exists()
    assert (run_path / "observer_distance_matrix.npz").exists()
    assert (run_path / "distance_composite.npz").exists()
    assert (run_path / "distance_overlap_projection.npz").exists()
    assert state_report["controls"]
    assert "correct_beats_controls" in state_report
    assert status["bulk_3d_established"] is False
    assert status["conformal_h3_spatial_chart_receipt"] is True
    assert "record_populated_h3_spatial_receipt" in status
    assert "record_family_h3_support_receipt" in status
    assert "defect_cluster_h3_support_receipt" in status
    assert "matter_defect_h3_support_receipt" in status
    assert mandatory["all_expected_failures_observed"] is True
    assert ports["port_names"] == [f"P{index}" for index in range(12)]


def test_transition_response_array_writes_scale_selection_receipt(tmp_path: Path):
    config = load_config(Path("configs/e1_s3_transition_response_screen_4k.yml"))
    config = dict(config)
    config["run_id"] = "transition_selection_smoke"
    config["graph"] = dict(config["graph"], patch_count=384, neighbors=8)
    config["dynamics"] = dict(config["dynamics"], cycles=10, repairs_per_cycle=384)
    config["bw"] = dict(config["bw"], cap_count=2, times=[0.1], n_jobs=1, max_basis=16)
    config["bw"]["selection"] = dict(config["bw"]["selection"], max_basis=16)
    config["observers"] = dict(config["observers"], sample_count=16, neighborhood_size=16)
    config["observer_objects"] = dict(config["observer_objects"], max_families=64)
    config["theorem_core"] = {
        "consensus_replay": {
            "enabled": True,
            "schedule_replays": 4,
            "requested_schedule_replays": 4,
            "max_event_rows": 4,
            "disjoint_checks": 16,
        }
    }

    result = run_bw_array_config(config, tmp_path)
    run_path = Path(result["path"])

    selection = json.loads((run_path / "transition_scale_selection_report.json").read_text(encoding="utf-8"))
    state = json.loads((run_path / "bw_state_derived_report.json").read_text(encoding="utf-8"))
    status = json.loads((run_path / "emergence_status_report.json").read_text(encoding="utf-8"))
    theorem = json.loads((run_path / "theorem_core_receipts.json").read_text(encoding="utf-8"))
    replay = json.loads((run_path / "finite_consensus_replay_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))

    assert state["direct_transition_automorphism"] is True
    assert state["endogenous_modular_generator"] is False
    assert selection["mode"] == "transition_scale_selection"
    assert selection["primary_source"] == "kms_collar_transport_response"
    assert "perturb_remeasure_response" in selection["source_reports"]
    assert "repair_affinity_response" in selection["source_reports"]
    assert selection["source_reports"]["declared_geometric_sanity"]["two_pi_selected"] is True
    assert selection["source_reports"]["kms_collar_transport_response"]["two_pi_selected"] is True
    assert status["transition_scale_selection"] is True
    assert "transition_two_pi_selected_by_primary" in status
    assert "CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT" in status
    assert "BW_AUTOMORPHISM_SANITY_RECEIPT" in status
    assert "BW_KMS_BRANCH_REPLAY_RECEIPT" in status
    assert "OPH_LORENTZ_THEOREM_FINITE_CONTRACT_V1" in status
    assert "ENDOGENOUS_MODULAR_GENERATOR_RECEIPT" in status
    assert "support_visible_lorentz_3p1_kinematics_receipt" in status
    assert status["CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT"] is True
    assert status["BW_KMS_BRANCH_REPLAY_RECEIPT"] is True
    assert status["FINITE_CONSENSUS_THEOREM_RECEIPT"] is True
    assert status["finite_consensus_theorem_receipt"] is True
    assert status["OPH_LORENTZ_THEOREM_FINITE_CONTRACT_V1"] is False
    assert status["finite_lorentz_theorem_contract_receipt"] is False
    assert status["ENDOGENOUS_MODULAR_GENERATOR_RECEIPT"] is False
    assert status["bulk_3d_established"] is False
    assert "Endogenous observer-record modular generators" in status["lorentz_claim_boundary"]
    assert manifest["transition_scale_selection"]["primary_source"] == "kms_collar_transport_response"
    assert "support_visible_lorentz_3p1_kinematics_receipt" in manifest["emergence_status"]
    assert manifest["theorem_core_receipts"]["finite_consensus_theorem_receipt"] is True
    assert manifest["emergence_status"]["bulk_3d_established"] is False
    assert result["transition_scale_selection"]["primary_source"] == "kms_collar_transport_response"
    assert result["theorem_core_receipts"]["finite_consensus_theorem"] is True
    assert theorem["finite_consensus_theorem"]["FINITE_CONSENSUS_THEOREM_RECEIPT"] is True
    assert replay["receipt"] is True
    assert replay["evidence"]["schedule_replay_count"] == 4


def test_lorentz_receipt_uses_direct_kms_selection_when_endogenous_state_fails():
    chart = {"conformal_h3_spatial_chart_receipt": True}
    failed_endogenous_state = {
        "endogenous_modular_generator": True,
        "state_selected_2pi": False,
        "correct_beats_controls": False,
    }
    transition = {
        "primary_source": "kms_collar_transport_response",
        "two_pi_selected": True,
        "response_degenerate": False,
    }

    receipts = _lorentz_branch_receipts(chart, failed_endogenous_state, transition)

    assert receipts["CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT"] is True
    assert receipts["BW_KMS_DIRECT_2PI_RECEIPT"] is True
    assert receipts["BW_KMS_BRANCH_REPLAY_RECEIPT"] is True
    assert receipts["CHART_LORENTZ_H3_RECEIPT"] is True
    assert receipts["support_visible_lorentz_3p1_kinematics_receipt"] is True
    assert receipts["ENDOGENOUS_MODULAR_GENERATOR_RECEIPT"] is False
    assert receipts["OPH_LORENTZ_THEOREM_FINITE_CONTRACT_V1"] is False
    assert receipts["finite_lorentz_theorem_contract_receipt"] is False


def test_kms_freezeout_cl_proxy_is_gate_checked(tmp_path: Path):
    config = load_config(Path("configs/e2_kms_freezeout_cl_screen_64k.yml"))
    config = dict(config)
    config["run_id"] = "kms_freezeout_smoke"
    config["graph"] = dict(config["graph"], patch_count=512, neighbors=8)
    config["dynamics"] = dict(config["dynamics"], cycles=10, repairs_per_cycle=512)
    config["bw"] = dict(config["bw"], cap_count=2, times=[0.1], n_jobs=1, max_basis=16)
    config["bw"]["selection"] = dict(config["bw"]["selection"], max_basis=16)
    config["observers"] = dict(config["observers"], sample_count=16, neighborhood_size=16)
    config["observer_objects"] = dict(config["observer_objects"], max_families=64)
    config["cosmology"] = dict(config["cosmology"])
    config["cosmology"]["freezeout"] = dict(
        config["cosmology"]["freezeout"],
        require_state_bw_controls=False,
    )
    config["cosmology"]["angular_power"] = dict(config["cosmology"]["angular_power"], ell_max=10, pair_samples=4096)

    result = run_bw_array_config(config, tmp_path)
    run_path = Path(result["path"])

    gate = json.loads((run_path / "cosmology_gate_report.json").read_text(encoding="utf-8"))
    cl_report = json.loads((run_path / "cl_comparison_report.json").read_text(encoding="utf-8"))
    cosmology = json.loads((run_path / "cosmology_observables.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    interaction = json.loads((run_path / "defect_interaction_report.json").read_text(encoding="utf-8"))
    particle = json.loads((run_path / "particle_likeness_report.json").read_text(encoding="utf-8"))

    assert gate["allowed"] is True
    assert gate["checks"]["kms_bw_pass"] is True
    assert gate["required"]["state_bw_controls_pass"] is False
    assert cl_report["gate_report"]["allowed"] is True
    assert "control_comparison" in cl_report["fields"]["record_signature"]
    assert cosmology["freezeout_cl_proxy"]["claim_boundary"].startswith("first measurement-facing")
    assert manifest["cosmology_gate"]["allowed"] is True
    assert interaction["mode"] == "screen_s3_defect_interaction_diagnostic"
    assert "defect_interaction" in manifest
    assert particle["particle_matter_receipt"] is False


def test_freezeout_cl_proxy_skips_when_gate_fails(tmp_path: Path):
    config = load_config(Path("configs/e2_kms_freezeout_cl_screen_64k.yml"))
    config = dict(config)
    config["run_id"] = "kms_freezeout_gate_fail_smoke"
    config["graph"] = dict(config["graph"], patch_count=384, neighbors=8)
    config["dynamics"] = dict(config["dynamics"], cycles=10, repairs_per_cycle=384)
    config["bw"] = dict(config["bw"], cap_count=2, times=[0.1], n_jobs=1, max_basis=16)
    config["bw"]["selection"] = dict(config["bw"]["selection"], max_basis=16)
    config["observers"] = dict(config["observers"], sample_count=16, neighborhood_size=16)
    config["observer_objects"] = dict(config["observer_objects"], max_families=64)
    config["cosmology"] = dict(config["cosmology"])
    config["cosmology"]["freezeout"] = dict(config["cosmology"]["freezeout"], require_neutral_reconstruction=True)

    result = run_bw_array_config(config, tmp_path)
    run_path = Path(result["path"])

    gate = json.loads((run_path / "cosmology_gate_report.json").read_text(encoding="utf-8"))

    assert gate["allowed"] is False
    assert "neutral_reconstruction_written" in gate["missing_requirements"]
    assert gate["freezeout_skipped"] is True
    assert not (run_path / "cl_comparison_report.json").exists()
    assert "cosmology_observables" not in result or not result.get("cosmology_observables")


def test_e3_cosmo_proxy_compact_profile_suppresses_debug_payloads(tmp_path: Path):
    config = load_config(Path("configs/e3_cosmo_proxy_screen_64k.yml"))
    config = dict(config)
    config["run_id"] = "e3_cosmo_proxy_compact_smoke"
    config["graph"] = dict(config["graph"], patch_count=384, neighbors=8)
    config["dynamics"] = dict(config["dynamics"], cycles=10, repairs_per_cycle=384)
    config["bw"] = dict(config["bw"], cap_count=2, times=[0.1], n_jobs=1, max_basis=16)
    config["bw"]["selection"] = dict(config["bw"]["selection"], max_basis=16)
    config["observers"] = dict(config["observers"], sample_count=16, neighborhood_size=16)
    config["h3_modular_response"] = {
        "enabled": True,
        "times": [0.1],
        "field_names": ["record_signature", "stable_count"],
        "candidate_count": 512,
        "candidate_radius": 1.2,
        "softness": 0.25,
        "min_observers": 8,
        "min_features": 4,
    }
    config["cosmology"] = dict(config["cosmology"])
    config["cosmology"]["freezeout"] = dict(
        config["cosmology"]["freezeout"],
        require_kms_bw_pass=False,
        commit_fraction=0.75,
    )
    config["cosmology"]["angular_power"] = dict(config["cosmology"]["angular_power"], ell_max=8, pair_samples=2048)

    result = run_bw_array_config(config, tmp_path)
    run_path = Path(result["path"])

    h3 = json.loads((run_path / "record_populated_h3_report.json").read_text(encoding="utf-8"))
    modular_kernel = json.loads((run_path / "modular_response_kernel_report.json").read_text(encoding="utf-8"))
    modular_h3 = json.loads((run_path / "modular_response_h3_report.json").read_text(encoding="utf-8"))
    theorem = json.loads((run_path / "theorem_core_receipts.json").read_text(encoding="utf-8"))
    cl_report = json.loads((run_path / "cl_comparison_report.json").read_text(encoding="utf-8"))
    proxy = json.loads((run_path / "oph_cosmo_proxy_v0_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))

    assert h3["enabled"] is False
    assert h3["claim_level"] == "demo"
    assert modular_kernel["mode"] == "observer_modular_response_kernel"
    assert modular_h3["mode"] == "modular_response_kernel_to_h3_fit"
    assert modular_h3["claim_level"] == "demo"
    assert modular_h3["physical_claim"] is False
    assert theorem["lyapunov"]["LYAPUNOV_DESCENT_RECEIPT"] is True
    assert theorem["exact_repair_projection"]["EXACT_REPAIR_EQUALS_PROJECTION_RECEIPT"] is True
    assert theorem["finite_settle_diagnostic"]["FINITE_SETTLE_DIAGNOSTIC_RECEIPT"] is True
    assert theorem["finite_consensus_theorem"]["FINITE_CONSENSUS_THEOREM_RECEIPT"] is False
    assert theorem["sm_quotient_gate"]["SM_QUOTIENT_GATE_RECEIPT"] is True
    assert cl_report["output_profile"] == "compact"
    assert cl_report["claim_level"] == "proxy"
    assert proxy["mode"] == "OPH_COSMO_PROXY_V0"
    assert proxy["physical_claim"] is False
    assert manifest["output_profile"] == "compact"
    assert "modular_response_h3" in manifest
    assert manifest["emergence_status"]["modular_response_h3_written"] is True
    assert manifest["emergence_status"]["bulk_3d_established"] is False
    assert manifest["pixel_scale"]["pixel_mode"] == "source_candidate"
    assert not (run_path / "observer_views.jsonl").exists()
    assert not (run_path / "freezeout_fields.npz").exists()
    assert not (run_path / "cl_proxy.csv").exists()
