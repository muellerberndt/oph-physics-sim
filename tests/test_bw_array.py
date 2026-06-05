from pathlib import Path
from math import isclose, pi

import json

from oph_fpe.experiments import load_config
from oph_fpe.scale.bw_array import run_bw_array_config


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
    assert (run_path / "edge_sector_heat_kernel_report.json").exists()
    assert (run_path / "central_record_born_report.json").exists()
    assert (run_path / "observer_checkpoint_restoration_report.json").exists()
    assert (run_path / "array_holonomy_report.json").exists()
    assert (run_path / "freezeout_fields.npz").exists()
    assert (run_path / "freezeout_map_summary.json").exists()
    assert (run_path / "cl_proxy.csv").exists()
    assert (run_path / "cl_controls.csv").exists()
    assert (run_path / "cl_comparison_report.json").exists()

    bw_report = json.loads((run_path / "bw_report.json").read_text(encoding="utf-8"))
    cap_report = json.loads((run_path / "cap_geometry_report.json").read_text(encoding="utf-8"))
    chart_report = json.loads((run_path / "conformal_h3_spatial_chart_report.json").read_text(encoding="utf-8"))
    h3_population_report = json.loads((run_path / "record_populated_h3_report.json").read_text(encoding="utf-8"))
    record_family_h3 = json.loads((run_path / "record_family_h3_report.json").read_text(encoding="utf-8"))
    defect_cluster_h3 = json.loads((run_path / "defect_cluster_h3_report.json").read_text(encoding="utf-8"))
    consensus_report = json.loads((run_path / "observer_consensus_report.json").read_text(encoding="utf-8"))
    emergence_status = json.loads((run_path / "emergence_status_report.json").read_text(encoding="utf-8"))
    cl_report = json.loads((run_path / "cl_comparison_report.json").read_text(encoding="utf-8"))
    observer_rows = [
        json.loads(line)
        for line in (run_path / "observer_views.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    ports_report = json.loads((run_path / "screen_ports.json").read_text(encoding="utf-8"))
    boundary_report = json.loads((run_path / "boundary_program_report.json").read_text(encoding="utf-8"))
    holonomy_report = json.loads((run_path / "array_holonomy_report.json").read_text(encoding="utf-8"))
    edge_sector = json.loads((run_path / "edge_sector_heat_kernel_report.json").read_text(encoding="utf-8"))
    born = json.loads((run_path / "central_record_born_report.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((run_path / "observer_checkpoint_restoration_report.json").read_text(encoding="utf-8"))

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
    assert cap_report["caps"][0]["soft_cap_area_planck"] > 0.0
    assert cap_report["caps"][0]["soft_cap_entropy_capacity"] > 0.0
    assert observer_rows
    assert {row["view_type"] for row in observer_rows} == {"patch_observer", "cap_observer"}
    assert consensus_report["observer_count"] > 0
    assert consensus_report["global_committed_fraction"] >= 0.0
    assert emergence_status["bulk_3d_established"] is False
    assert emergence_status["requires_refinement_scaling"] is True
    assert "observer_consensus" in manifest
    assert "mandatory_controls" in manifest
    assert "emergence_status" in manifest
    assert "cosmology_observables" in manifest
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

    result = run_bw_array_config(config, tmp_path)
    run_path = Path(result["path"])

    selection = json.loads((run_path / "transition_scale_selection_report.json").read_text(encoding="utf-8"))
    status = json.loads((run_path / "emergence_status_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))

    assert selection["mode"] == "transition_scale_selection"
    assert selection["primary_source"] == "kms_collar_transport_response"
    assert "perturb_remeasure_response" in selection["source_reports"]
    assert "repair_affinity_response" in selection["source_reports"]
    assert selection["source_reports"]["declared_geometric_sanity"]["two_pi_selected"] is True
    assert selection["source_reports"]["kms_collar_transport_response"]["two_pi_selected"] is True
    assert status["transition_scale_selection"] is True
    assert "transition_two_pi_selected_by_primary" in status
    assert manifest["transition_scale_selection"]["primary_source"] == "kms_collar_transport_response"
    assert result["transition_scale_selection"]["primary_source"] == "kms_collar_transport_response"


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
