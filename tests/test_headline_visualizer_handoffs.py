from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "visualizer_handoffs/oph-headlines-2026-08-20"
MAX_BYTES = 200_000_000


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_suite_has_all_priority_and_headline_packages() -> None:
    manifest = _load(SUITE / "suite_manifest.json")
    ids = {row["package_id"] for row in manifest["packages"]}
    assert manifest["schema"] == "oph.visualizer-handoff-suite.v1"
    assert manifest["display_data_only"] is True
    assert manifest["whole_run_included"] is False
    assert manifest["package_count"] == 14
    assert {
        "observer-quantum-born-collapse",
        "refinement-depth-emergence",
        "defect-emergence",
        "defect-grouping-and-interactions",
        "s2-carrier-network-interactions",
        "individual-carrier-repair-animation",
        "repair-confluence-and-public-records",
        "observer-cameras",
        "observer-modular-time",
        "observer-spacetime-emergence",
        "theorem-paper-simulation-evidence-atlas",
    }.issubset(ids)


def test_every_package_is_hash_complete_and_under_200_mb() -> None:
    suite = _load(SUITE / "suite_manifest.json")
    for package_row in suite["packages"]:
        package = SUITE / package_row["package_id"]
        manifest_path = package / "manifest.json"
        manifest = _load(manifest_path)
        assert manifest["schema"] == "oph.visualizer-handoff.v1"
        assert manifest["display_data_only"] is True
        assert manifest["whole_run_included"] is False
        assert manifest["total_bytes"] < MAX_BYTES
        assert package_row["manifest_sha256"] == _sha256(manifest_path)
        assert (package / "DISPLAY_INSTRUCTIONS.md").is_file()
        for row in manifest["files"]:
            path = package / row["path"]
            assert path.is_file()
            assert not path.is_symlink()
            assert path.stat().st_size == row["bytes"]
            assert _sha256(path) == row["sha256"]


def test_quantum_package_exposes_exact_collapse_scenarios() -> None:
    value = _load(
        SUITE
        / "observer-quantum-born-collapse/data/QM_OBSERVER_VIZ.v1.json"
    )
    kinds = {row["kind"] for row in value["scenarios"]}
    assert value["schema"] == "oph.sim.qm_observer_viz.v1"
    assert {"base_context", "collapse_chain", "interference"}.issubset(kinds)
    assert any(row["collapse_events"] for row in value["scenarios"])


def test_depth_package_keeps_matched_scale_and_scale_controls_separate() -> None:
    value = _load(
        SUITE / "refinement-depth-emergence/data/depth_emergence.json"
    )
    names = {row["name"] for row in value["rows"]}
    assert value["evidential_status"] == "exploratory_non_evidential"
    assert "area_weighted_kappa_1" in names
    assert "matched_uniform_kappa_1" in names
    assert "all_ones_scale_control" in names
    assert all(row["window_saturation_guard_applied"] for row in value["rows"])


def test_carrier_frame_export_is_downsampled_display_data() -> None:
    path = (
        SUITE
        / "s2-carrier-network-interactions/data/carrier_network_frames.npz"
    )
    with np.load(path) as data:
        assert data["points"].shape == (1024, 3)
        assert data["mismatch"].shape[1] == 1024
        assert data["cycles"].shape[0] == data["mismatch"].shape[0]


def test_evidence_atlas_covers_registry_and_lean_surfaces() -> None:
    value = _load(
        SUITE
        / "theorem-paper-simulation-evidence-atlas/data/evidence_atlas.json"
    )
    assert len(value["axioms"]) == 3
    assert len(value["claims"]) >= 100
    assert len(value["lean_modules"]) >= 100
    assert sum(row["theorem_or_lemma_count"] for row in value["lean_modules"]) > 0
