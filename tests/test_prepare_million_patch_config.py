from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from oph_fpe.experiments import load_config


def test_million_patch_config_is_bounded_and_loadable(tmp_path: Path) -> None:
    destination = tmp_path / "million.yml"
    subprocess.run(
        [
            sys.executable,
            "tools/prepare_million_patch_config.py",
            "--out",
            str(destination),
            "--seed",
            "20260753",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    config = load_config(destination)

    assert config["graph"]["patch_count"] == 1_048_576
    assert config["observers"]["sample_count"] == 64_000
    assert config["observers"]["overlap_correspondence_max_observers"] == 8_192
    assert config["observers"]["consensus_analysis_max_observers"] == 8_192
    assert config["observers"]["observer_wide_analysis_max_observers"] == 8_192
    assert config["observers"]["compact_unenriched_rows"] is True
    assert config["million_patch_preparation"]["materialized_observer_count"] == 64_000
    assert config["million_patch_preparation"]["observer_wide_analysis_count"] == 8_192
    assert config["neutral_reconstruction"]["distance_matrix_max_observers"] == 2_048
    assert config["visualization_export"] == {
        "max_screen_points": 3_500,
        "max_observers": 96,
        "max_h3_objects": 512,
    }
    assert config["cosmology"]["harmonic_time_trace"]["save_raw_frames"] is False
    assert config["cosmology"]["harmonic_time_trace"]["sample_count"] == 4
    assert config["cosmology"]["harmonic_time_trace"]["ell_max"] == 12
    assert config["cosmology"]["angular_power"]["ell_max"] == 24
    assert len(config["cosmology"]["freezeout"]["fields"]) == 4
    assert config["cosmology"]["freezeout"]["require_kms_bw_pass"] is False
    assert config["cosmology"]["freezeout"]["million_patch_diagnostic_export_policy"] == (
        "export_final_screen_fields_for_visualization_without_promoting_bw_bulk_or_physical_cmb"
    )
    assert config["h3_support_profiles"]["cap_count"] == 8
    assert config["h3_modular_response"]["observable_mode"] == "perturb_resettle_transition"
    assert config["h3_modular_response"]["max_full_graph_simulations"] == 32
    assert config["h3_modular_response"]["full_graph_budget_policy"] == "skip_if_exceeded"
    assert config["h3_modular_response"]["full_graph_n_jobs"] == 8
    assert config["cosmology"]["b_a_paired_perturbation"]["max_caps"] == 4
    assert config["cosmology"]["b_a_paired_perturbation"]["a_grid"] == [0.01, 0.1]
    assert config["cosmology"]["b_a_paired_perturbation"]["max_full_graph_simulations"] == 24
    assert config["cosmology"]["b_a_paired_perturbation"]["full_graph_n_jobs"] == 8
    assert config["cosmology"]["b_a_paired_perturbation"]["reuse_dynamics_across_a_grid"] is True
    assert config["million_patch_preparation"]["full_graph_probe_workers"] == {
        "h3_modular_response": 8,
        "paired_b_a": 8,
        "execution": "bounded_ordered_thread_pool",
    }
    assert config["million_patch_preparation"]["full_graph_simulation_caps"] == {
        "h3_modular_response": 32,
        "paired_b_a": 24,
        "paired_without_a_grid_reuse": 48,
    }
    assert config["defects"]["timeline"]["max_analysis_clusters_per_snapshot"] == 512
    assert config["defects"]["timeline"]["max_serialized_snapshots"] == 32
    assert config["defects"]["timeline"]["max_serialized_json_bytes"] == 64_000_000
    assert config["theorem_core"]["include_default_sm_candidate"] is False
    assert config["theorem_core"]["consensus_replay"]["schedule_replays"] == 0
    assert config["theorem_core"]["consensus_replay"]["requested_schedule_replays"] == 0
    assert config["theorem_core"]["consensus_replay"][
        "large_run_exact_failure_short_circuit"
    ] is True


def test_acceptance_config_is_the_production_config_with_only_scale_changes(
    tmp_path: Path,
) -> None:
    production_path = tmp_path / "production.yml"
    acceptance_path = tmp_path / "acceptance.yml"
    common = [
        sys.executable,
        "tools/prepare_million_patch_config.py",
        "--seed",
        "20260753",
    ]
    subprocess.run(
        [*common, "--out", str(production_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [*common, "--acceptance-4k", "--out", str(acceptance_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    production = load_config(production_path)
    acceptance = load_config(acceptance_path)
    assert acceptance["graph"]["patch_count"] == 4_096
    assert acceptance["dynamics"]["repairs_per_cycle"] == 256
    assert acceptance["observers"] == production["observers"]
    assert acceptance["bw"] == production["bw"]
    assert acceptance["h3_modular_response"] == production["h3_modular_response"]
    assert acceptance["cosmology"] == production["cosmology"]
    assert acceptance["defects"] == production["defects"]
    assert acceptance["visualization_export"] == production["visualization_export"]
    gate = acceptance["million_patch_preparation"]["acceptance_gate"]
    assert gate["production_target_patch_count"] == 1_048_576
    assert gate["acceptance_patch_count"] == 4_096


def test_pilot_config_retains_production_gates_with_64k_16k_scale(
    tmp_path: Path,
) -> None:
    production_path = tmp_path / "production.yml"
    pilot_path = tmp_path / "pilot.yml"
    common = [
        sys.executable,
        "tools/prepare_million_patch_config.py",
        "--seed",
        "20260753",
    ]
    subprocess.run(
        [*common, "--out", str(production_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [*common, "--pilot-64k-16k", "--out", str(pilot_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    production = load_config(production_path)
    pilot = load_config(pilot_path)
    assert pilot["graph"]["patch_count"] == 65_536
    assert pilot["dynamics"]["repairs_per_cycle"] == 4_096
    assert pilot["observers"]["sample_count"] == 16_000
    assert pilot["observers"]["observer_wide_analysis_max_observers"] == 8_192
    assert pilot["observers"]["consensus_analysis_max_observers"] == 8_192
    assert pilot["neutral_reconstruction"] == production["neutral_reconstruction"]
    assert pilot["bw"] == production["bw"]
    assert pilot["h3_modular_response"] == production["h3_modular_response"]
    assert pilot["cosmology"] == production["cosmology"]
    assert pilot["defects"] == production["defects"]
    assert pilot["visualization_export"] == production["visualization_export"]
    gate = pilot["million_patch_preparation"]["pilot_gate"]
    assert gate["pilot_patch_count"] == 65_536
    assert gate["pilot_materialized_observer_count"] == 16_000


def test_large_pilot_config_retains_production_gates_with_128k_32k_scale(
    tmp_path: Path,
) -> None:
    production_path = tmp_path / "production.yml"
    pilot_path = tmp_path / "large_pilot.yml"
    common = [
        sys.executable,
        "tools/prepare_million_patch_config.py",
        "--seed",
        "20260753",
    ]
    subprocess.run(
        [*common, "--out", str(production_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [*common, "--pilot-128k-32k", "--out", str(pilot_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    production = load_config(production_path)
    pilot = load_config(pilot_path)
    assert pilot["graph"]["patch_count"] == 131_072
    assert pilot["dynamics"]["repairs_per_cycle"] == 8_192
    assert pilot["observers"]["sample_count"] == 32_000
    assert pilot["observers"]["observer_wide_analysis_max_observers"] == 8_192
    assert pilot["observers"]["consensus_analysis_max_observers"] == 8_192
    assert pilot["neutral_reconstruction"] == production["neutral_reconstruction"]
    assert pilot["bw"] == production["bw"]
    assert pilot["h3_modular_response"] == production["h3_modular_response"]
    assert pilot["cosmology"] == production["cosmology"]
    assert pilot["defects"] == production["defects"]
    assert pilot["visualization_export"] == production["visualization_export"]
    gate = pilot["million_patch_preparation"]["pilot_gate"]
    assert gate["pilot_patch_count"] == 131_072
    assert gate["pilot_materialized_observer_count"] == 32_000
