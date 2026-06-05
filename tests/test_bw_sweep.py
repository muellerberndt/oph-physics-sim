from pathlib import Path

import yaml

from oph_fpe.experiments import load_config
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
