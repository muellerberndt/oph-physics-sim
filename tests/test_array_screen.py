from pathlib import Path

from oph_fpe.experiments import load_config
from oph_fpe.scale import run_array_screen_config


def test_array_screen_smoke_writes_dimension_report(tmp_path: Path):
    config = load_config(Path("configs/e1_s3_modular_screen_4k.yml"))
    config = dict(config)
    config["run_id"] = "array_smoke"
    config["graph"] = dict(config["graph"], patch_count=256, neighbors=6)
    config["dynamics"] = dict(config["dynamics"], cycles=8, repairs_per_cycle=512)
    config["observables"] = dict(config["observables"])
    config["observables"]["modular_lift"] = {"max_points": 4096, "center_samples": 128}

    result = run_array_screen_config(config, tmp_path)

    assert result["final_phi"] >= 0
    assert result["dimensions"]["distance_source"] == "array_modular_lift_record_history"
    assert (Path(result["path"]) / "verifier_receipts.jsonl").exists()
