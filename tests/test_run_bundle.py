import json
from pathlib import Path

import pytest

from oph_fpe.evidence.bundle import RunBundle
from oph_fpe.experiments import load_config, run_config


def test_run_bundle_can_write_compact_json(tmp_path: Path):
    bundle = RunBundle(tmp_path, "compact_json")

    receipt = bundle.write_json(
        "timeline.json",
        {"rows": [1, 2], "receipt": False},
        compact=True,
        max_bytes=64,
    )

    assert (bundle.path / "timeline.json").read_text(encoding="utf-8") == (
        '{"rows":[1,2],"receipt":false}'
    )
    assert receipt["byte_count"] == 30
    assert receipt["under_hard_limit"] is True

    with pytest.raises(ValueError, match="hard gate"):
        bundle.write_json("too_large.json", {"payload": "x" * 80}, max_bytes=64)
    assert not (bundle.path / "too_large.json").exists()


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


def test_e0_run_can_attach_positive_geometry_kernel_dispatch(tmp_path: Path):
    config = load_config(Path("configs/e0_z2_patchnet.yml"))
    config = dict(config)
    config["run_id"] = "test_e0_kernel_dispatch"
    config["graph"] = dict(config["graph"], patch_count=12)
    config["dynamics"] = dict(config["dynamics"], cycles=4, repairs_per_cycle=6, record_commit_cycles=2)
    config["kernels"] = {"positive_geometry": {"enabled": True}}

    result = run_config(config, tmp_path)
    run_path = Path(result["path"])
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))

    assert (run_path / "kernel_dispatch_report.json").exists()
    assert (run_path / "positive_geometry_kernel_report.json").exists()
    assert result["kernel_dispatch"]["routing_decision"] == "generic_oph_repair_with_kernel_receipts"
    assert result["kernel_dispatch"]["effective_acceleration_enabled"] is False
    assert manifest["kernel_dispatch"]["positive_geometry"]["dispatch_status"] == "geometry_certified_diagnostic_only"
    assert manifest["kernel_dispatch"]["physical_observables_changed"] is False
    assert manifest["source_provenance"]["simulator"]["commit"] == manifest["git_commit"]
    assert manifest["source_provenance"]["simulator"]["worktree_state_sha256"].startswith("sha256:")
    assert "dirty" in manifest["source_provenance"]["simulator"]
    assert manifest["dependency_provenance"]["pyproject_sha256"].startswith("sha256:")
