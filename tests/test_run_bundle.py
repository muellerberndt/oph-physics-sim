from pathlib import Path

from oph_fpe.experiments import load_config, run_config


def test_e0_run_writes_receipts(tmp_path: Path):
    config = load_config(Path("configs/e0_z2_patchnet.yml"))
    config = dict(config)
    config["run_id"] = "test_e0"
    config["graph"] = dict(config["graph"], patch_count=12)
    config["dynamics"] = dict(config["dynamics"], cycles=6, repairs_per_cycle=6, record_commit_cycles=2)

    result = run_config(config, tmp_path)
    run_path = Path(result["path"])

    assert (run_path / "manifest.json").exists()
    assert (run_path / "verifier_receipts.jsonl").exists()
    assert (run_path / "dimension_report.json").exists()
    assert result["final_phi"] >= 0
