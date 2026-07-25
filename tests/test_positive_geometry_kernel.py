from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from oph_fpe.dynamics.positive_geometry import (
    default_pgk_manifest,
    default_kernel_runtime,
    _load_kernel_runtime,
    positive_geometry_kernel_report,
    write_positive_geometry_kernel_report,
)

EXTERNAL_PGK_AVAILABLE = (
    default_kernel_runtime().is_file() and default_pgk_manifest().is_file()
)


@pytest.mark.skipif(
    not EXTERNAL_PGK_AVAILABLE,
    reason="declared external positive-geometry plugin is not installed",
)
def test_default_a614_pilot_is_geometry_certified_but_not_enabled():
    report = positive_geometry_kernel_report()
    gates = report["readiness_gates"]

    assert report["mode"] == "oph_positive_geometry_kernel_integration_v2"
    assert report["kernel_constitution"]["manifest_status_strings_authoritative"] is False
    assert report["receipt"]["verdict"] == "CERTIFIED_NATIVE_BACKEND_NOT_ENABLED"
    assert report["execution_mode"] == "CERTIFIED_NATIVE"
    assert report["trusted_acceleration_enabled"] is False
    assert gates["PGK_geometry_certified"] is True
    assert gates["PGK_sector_recognition_certified"] is False
    assert gates["PGK_quotient_compiler_certified"] is False
    assert gates["PGK_readout_equivalence_certified"] is False
    assert gates["PGK_event_commit_certified"] is False
    assert gates["generic_oph_repair_fallback_required"] is True


@pytest.mark.skipif(
    not EXTERNAL_PGK_AVAILABLE,
    reason="declared external positive-geometry plugin is not installed",
)
def test_self_reported_pass_manifest_cannot_enable_acceleration(tmp_path: Path):
    manifest = _load_default_manifest()
    manifest["claim_scope"] = "certified_accelerator"
    manifest["sector"]["self_reported_status"] = "PASS"
    manifest["recognition"]["self_reported_status"] = "PASS"
    manifest["compiler"]["self_reported_status"] = "PASS"
    manifest["semantics"]["self_reported_status"] = "PASS"
    manifest["commit"]["self_reported_status"] = "PASS"
    path = tmp_path / "self_reported_pass_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = positive_geometry_kernel_report(path)

    assert report["receipt"]["verdict"] == "CERTIFIED_NATIVE_BACKEND_NOT_ENABLED"
    assert report["execution_mode"] == "CERTIFIED_NATIVE"
    assert report["trusted_acceleration_enabled"] is False
    assert report["readiness_gates"]["trusted_oph_scattering_readout"] is False
    assert report["readiness_gates"]["generic_oph_repair_fallback_required"] is True


@pytest.mark.skipif(
    not EXTERNAL_PGK_AVAILABLE,
    reason="declared external positive-geometry plugin is not installed",
)
def test_host_registered_verifiers_enable_acceleration(tmp_path: Path):
    runtime = _load_kernel_runtime(default_kernel_runtime())
    registry = runtime.VerifierRegistry()

    def pass_verifier(_manifest: dict, _native_checks: dict) -> tuple[bool, str]:
        return True, "unit-test host verifier passed"

    verifier_id = "oph.test.pass"
    for category in ("recognition", "compiler", "semantics", "commit"):
        registry.add_test_verifier(category, verifier_id, pass_verifier)

    manifest = _load_default_manifest()
    manifest["claim_scope"] = "certified_accelerator"
    manifest["recognition"]["verifier_id"] = verifier_id
    manifest["compiler"]["verifier_id"] = verifier_id
    manifest["semantics"]["verifier_id"] = verifier_id
    manifest["commit"]["verifier_id"] = verifier_id
    path = tmp_path / "host_verified_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = positive_geometry_kernel_report(path, registry=registry)

    assert report["receipt"]["verdict"] == "CERTIFIED_ACCELERATED"
    assert report["execution_mode"] == "CERTIFIED_ACCELERATED"
    assert report["trusted_acceleration_enabled"] is True
    assert report["readiness_gates"]["trusted_oph_scattering_readout"] is True
    assert report["readiness_gates"]["generic_oph_repair_fallback_required"] is False


@pytest.mark.skipif(
    not EXTERNAL_PGK_AVAILABLE,
    reason="declared external positive-geometry plugin is not installed",
)
def test_bad_geometry_manifest_fails_closed(tmp_path: Path):
    manifest = _load_default_manifest()
    manifest["native"]["payload"]["external_data"]["Z_rows"][4][5] = 1
    path = tmp_path / "bad_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = positive_geometry_kernel_report(path)

    assert report["receipt"]["verdict"] == "REJECTED"
    assert report["execution_mode"] == "EXACT_GENERIC_FALLBACK"
    assert report["trusted_acceleration_enabled"] is False
    assert report["readiness_gates"]["PGK_failed_gates_present"] is True
    assert report["readiness_gates"]["generic_oph_repair_fallback_required"] is True


def test_missing_declared_plugin_fails_closed(tmp_path: Path):
    missing_root = tmp_path / "missing-plugin"
    missing_manifest = missing_root / "missing-manifest.json"
    report = positive_geometry_kernel_report(
        missing_manifest,
        pgk_root=missing_root,
    )

    assert report["receipt"]["verdict"] == "DECLARED_PLUGIN_UNAVAILABLE"
    assert report["execution_mode"] == "EXACT_GENERIC_FALLBACK"
    assert report["trusted_acceleration_enabled"] is False
    assert report["physical_prediction"] is False
    assert report["readiness_gates"]["PGK_open_gates_present"] is True
    assert report["readiness_gates"]["generic_oph_repair_fallback_required"] is True


def test_positive_geometry_kernel_report_writes_fail_closed_artifacts(
    tmp_path: Path,
):
    out = tmp_path / "out"
    missing_root = tmp_path / "missing-plugin"
    missing_manifest = missing_root / "missing-manifest.json"
    report = write_positive_geometry_kernel_report(
        out,
        missing_manifest,
        pgk_root=missing_root,
    )

    assert (out / "positive_geometry_kernel_report.json").exists()
    assert (out / "positive_geometry_kernel_report.md").exists()
    assert not (out / "positive_geometry_kernel_manifest.json").exists()
    assert (out / "positive_geometry_kernel_receipt.json").exists()
    loaded = json.loads(
        (out / "positive_geometry_kernel_report.json").read_text(encoding="utf-8")
    )
    assert loaded["receipt"]["verdict"] == report["receipt"]["verdict"]


def test_measurement_pack_copies_positive_geometry_kernel_receipt(tmp_path: Path):
    from oph_fpe.measurement_pack import export_measurement_pack

    run = tmp_path / "run"
    out = tmp_path / "pack"
    missing_root = tmp_path / "missing-plugin"
    write_positive_geometry_kernel_report(
        run,
        missing_root / "missing-manifest.json",
        pgk_root=missing_root,
    )

    pack = export_measurement_pack([run], out)

    assert (out / "positive_geometry_kernel_report.json").exists()
    manifest_placeholder = out / "positive_geometry_kernel_manifest.json"
    assert manifest_placeholder.exists()
    assert json.loads(manifest_placeholder.read_text(encoding="utf-8")) == {}
    assert (out / "positive_geometry_kernel_receipt.json").exists()
    assert pack["claims"]["positive_geometry_kernel_written"] is True
    assert (
        pack["claims"]["positive_geometry_kernel_verdict"]
        == "DECLARED_PLUGIN_UNAVAILABLE"
    )
    assert pack["claims"]["positive_geometry_kernel_geometry_certified"] is False
    assert pack["claims"]["positive_geometry_kernel_acceleration_enabled"] is False


def _load_default_manifest() -> dict:
    return copy.deepcopy(json.loads(default_pgk_manifest().read_text(encoding="utf-8")))
