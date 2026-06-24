from pathlib import Path

import yaml

from oph_fpe.experiments import load_config
import oph_fpe.pipelines.oph_universe_sweep as universe_sweep
from oph_fpe.scale.bw_array import _large_run_readiness_report
from oph_fpe.scale.bw_sweep import run_bw_sweep
from oph_fpe.scale.parallel import jobs_from_config, sweep_parallel_plan


def test_bw_sweep_runs_configs_in_parallel(tmp_path: Path):
    config = load_config(Path("configs/e1_s3_bw_screen_4k.yml"))
    config = dict(config)
    config["name"] = "bw_sweep_smoke"
    config["graph"] = dict(config["graph"], patch_count=256, neighbors=8)
    config["dynamics"] = dict(config["dynamics"], cycles=6, repairs_per_cycle=256)
    config["bw"] = dict(config["bw"], cap_count=3, times=[0.025], n_jobs=1)
    config_path = tmp_path / "bw_sweep_smoke.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = run_bw_sweep([config_path], tmp_path / "runs", seeds=[1, 2], workers=2, inner_jobs=1)

    assert summary["job_count"] == 2
    assert summary["workers"] == 2
    assert Path(summary["summary_path"]).exists()
    assert all(row["ok"] for row in summary["results"])
    assert all(row["final_phi"] >= 0 for row in summary["results"])
    assert all("large_run_readiness" in row for row in summary["results"])
    readiness = summary["large_run_readiness_summary"]
    assert readiness["ok_job_count"] == 2
    assert readiness["recommended_large_run_lanes"]["do_not_scale_yet"] == 2
    assert readiness["claim_scale_candidate_count"] == 0
    assert readiness["stability_only_candidate_count"] == 2
    assert readiness["state_bw_expensive_run_worthwhile_count"] == 0


def test_parallel_plan_fills_available_cpus_without_explicit_limits(monkeypatch):
    monkeypatch.setenv("OPH_FPE_CPUS", "10")

    workers, inner_jobs = sweep_parallel_plan(job_count=3, workers=None, inner_jobs=None, reserve=1)

    assert workers == 3
    assert inner_jobs == 3


def test_parallel_plan_respects_explicit_limits(monkeypatch):
    monkeypatch.setenv("OPH_FPE_CPUS", "32")

    workers, inner_jobs = sweep_parallel_plan(job_count=8, workers=4, inner_jobs=2, reserve=1)

    assert workers == 4
    assert inner_jobs == 2
    assert jobs_from_config("auto", reserve=1) == 32


def test_oph_universe_sweep_aggregates_stubbed_pipeline(tmp_path: Path, monkeypatch):
    config = load_config(Path("configs/e1_s3_bw_screen_4k.yml"))
    config = dict(config)
    config["name"] = "universe_sweep_stub"
    config_path = tmp_path / "universe_sweep_stub.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def fake_pipeline(**kwargs):
        return {
            "run_dir": str(tmp_path / str(kwargs["run_id"])),
            "final_receipts": {
                "observer_modular_time_receipt": True,
                "physical_cmb_prediction_receipt": False,
            },
            "viewer_outputs": {"timeline_payload": "payload.json"},
            "proof_summary": {"physical_cmb_prediction": False},
            "frontier_artifacts": {},
        }

    monkeypatch.setattr(universe_sweep, "run_oph_universe_pipeline", fake_pipeline)

    summary = universe_sweep.run_oph_universe_sweep(
        [config_path],
        tmp_path / "runs",
        seeds=[11, 12],
        workers=1,
        inner_jobs=3,
        max_observers=4,
    )

    assert summary["job_count"] == 2
    assert summary["workers"] == 1
    assert summary["inner_jobs"] == 3
    assert Path(summary["summary_path"]).exists()
    assert all(row["ok"] for row in summary["results"])
    assert summary["receipt_summary"]["passed_receipt_counts"]["observer_modular_time_receipt"] == 2
    assert "physical_cmb_prediction_receipt" not in summary["receipt_summary"]["passed_receipt_counts"]


def test_large_run_readiness_blocks_wrong_scale_state_bw():
    report = _large_run_readiness_report(
        {"cosmology": {"freezeout": {"enabled": False}}},
        state_bw_report={
            "median": 1.2,
            "correct_beats_controls": False,
            "state_selected_scale_label": "1x",
            "state_selected_2pi": False,
            "best_control": "wrong_1x_normalization",
            "generator_scale_audit": {
                "enabled": True,
                "best_label": "1x",
                "diagnosis": "different_scale_best",
            },
            "inferred_modular_clock_fit": {
                "enabled": True,
                "receipt": False,
                "blockers": ["nearest_known_scale_is_1x"],
            },
        },
        transition_selection_report={},
        cosmology_gate_report={},
        observer_modular_experience_report={},
        paper_3d_chart_report={},
        theorem_core_report={},
    )

    state_lane = report["lanes"]["state_bw"]
    assert state_lane["scale_candidate"] is False
    assert "state_bw_controls_failed" in state_lane["blockers"]
    assert "state_bw_selected_1x_not_2pi" in state_lane["blockers"]
    assert report["state_bw_expensive_run_worthwhile"] is False
    assert report["recommended_large_run_lane"] == "do_not_scale_yet"
    assert report["claim_scale_candidate"] is False


def test_large_run_readiness_accepts_finite_lorentz_clock_over_legacy_scale_diagnostic():
    report = _large_run_readiness_report(
        {"cosmology": {"freezeout": {"enabled": False}}},
        state_bw_report={
            "ENDOGENOUS_MODULAR_GENERATOR_RECEIPT": True,
            "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT": True,
            "correct_beats_controls": False,
            "state_selected_scale_label": "1x",
            "state_selected_2pi": False,
            "generator_scale_audit": {
                "enabled": True,
                "best_label": "0",
                "diagnosis": "no_flow_best",
            },
            "inferred_modular_clock_fit": {
                "enabled": True,
                "receipt": True,
                "nearest_known_scale": "2pi",
                "blockers": [],
            },
        },
        transition_selection_report={},
        cosmology_gate_report={},
        observer_modular_experience_report={},
        paper_3d_chart_report={},
        theorem_core_report={},
    )

    state_lane = report["lanes"]["state_bw"]
    assert state_lane["scale_candidate"] is True
    assert state_lane["blockers"] == []
    assert state_lane["details"]["finite_lorentz_modular_clock_receipt"] is True
    assert "state_bw_controls_failed" in state_lane["details"]["legacy_scale_diagnostic_blockers"]
    assert report["state_bw_expensive_run_worthwhile"] is True
    assert report["recommended_large_run_lane"] == "state_bw_refinement"
    assert report["claim_scale_candidate"] is True


def test_large_run_readiness_routes_screen_proxy_without_state_bw_promotion():
    report = _large_run_readiness_report(
        {"cosmology": {"freezeout": {"enabled": True}}},
        state_bw_report={
            "correct_beats_controls": False,
            "state_selected_scale_label": "1x",
            "state_selected_2pi": False,
        },
        transition_selection_report={
            "selected_label": "2pi",
            "two_pi_selected": True,
            "response_degenerate": False,
            "primary_source": "kms_collar_transport_response",
        },
        cosmology_gate_report={
            "enabled": True,
            "allowed": True,
            "missing_requirements": [],
            "checks": {"state_bw_controls_pass": False},
            "required": {"state_bw_controls_pass": False},
        },
        observer_modular_experience_report={"observer_modular_time_receipt": True},
        paper_3d_chart_report={
            "paper_theorem_3d_bulk_chart_receipt": True,
            "paper_theorem_object_populated_chart_precursor_receipt": True,
            "paper_theorem_neutral_populated_bulk_receipt": False,
        },
        theorem_core_report={"finite_consensus_theorem_receipt": False},
    )

    assert report["lanes"]["state_bw"]["scale_candidate"] is False
    assert report["lanes"]["screen_cmb_proxy"]["scale_candidate"] is True
    assert report["lanes"]["bulk_3d"]["scale_candidate"] is False
    assert report["lanes"]["bulk_3d"]["blockers"] == [
        "strict_neutral_bulk_gate_not_established",
    ]
    assert report["recommended_large_run_lane"] == "screen_cmb_proxy_refinement"
    assert report["claim_scale_candidate"] is True
    assert report["state_bw_expensive_run_worthwhile"] is False


def test_large_run_readiness_routes_observer_facing_bulk_without_strict_neutral():
    report = _large_run_readiness_report(
        {"cosmology": {"freezeout": {"enabled": False}}},
        state_bw_report={},
        transition_selection_report={},
        cosmology_gate_report={},
        observer_modular_experience_report={
            "observer_modular_time_receipt": True,
            "observer_facing_3p1d_h3_experience_receipt": True,
            "observer_facing_populated_h3_experience_receipt": True,
        },
        paper_3d_chart_report={
            "paper_theorem_3d_bulk_chart_receipt": True,
            "paper_theorem_object_populated_chart_precursor_receipt": True,
            "paper_theorem_neutral_populated_bulk_receipt": False,
        },
        theorem_core_report={},
    )

    assert report["lanes"]["observer_facing_bulk"]["scale_candidate"] is True
    assert report["lanes"]["bulk_3d"]["scale_candidate"] is False
    assert report["recommended_large_run_lane"] == "observer_facing_bulk_visualization_refinement"
    assert report["claim_scale_candidate"] is True
