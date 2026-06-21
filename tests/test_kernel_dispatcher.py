from __future__ import annotations

from pathlib import Path

from oph_fpe.dynamics.kernel_dispatcher import (
    dispatch_configured_kernels,
    kernel_dispatch_manifest_summary,
)
from oph_fpe.measurement_pack import export_measurement_pack


def test_positive_geometry_dispatch_writes_fail_closed_receipts(tmp_path: Path):
    report = dispatch_configured_kernels(
        {"kernels": {"positive_geometry": {"enabled": True}}},
        tmp_path,
        engine="unit_test",
    )
    positive_geometry = report["kernels"]["positive_geometry"]

    assert (tmp_path / "kernel_dispatch_report.json").exists()
    assert (tmp_path / "kernel_dispatch_report.md").exists()
    assert (tmp_path / "positive_geometry_kernel_report.json").exists()
    assert (tmp_path / "positive_geometry_kernel_receipt.json").exists()
    assert report["routing_decision"] == "generic_oph_repair_with_kernel_receipts"
    assert report["generic_repair_executed"] is True
    assert report["effective_acceleration_enabled"] is False
    assert report["physical_observables_changed"] is False
    assert positive_geometry["dispatch_status"] == "geometry_certified_diagnostic_only"
    assert positive_geometry["native_verdict"] == "CERTIFIED_NATIVE_BACKEND_NOT_ENABLED"
    assert positive_geometry["fallback_required"] is True

    summary = kernel_dispatch_manifest_summary(report)
    assert summary["positive_geometry"]["native_trusted_acceleration"] is False
    assert summary["positive_geometry"]["effective_acceleration_enabled"] is False


def test_positive_geometry_dispatch_fails_closed_on_bad_manifest(tmp_path: Path):
    report = dispatch_configured_kernels(
        {"kernels": {"positive_geometry": {"enabled": True, "manifest_path": tmp_path / "missing.json"}}},
        tmp_path,
        engine="unit_test",
    )
    positive_geometry = report["kernels"]["positive_geometry"]

    assert positive_geometry["dispatch_status"] == "fail_closed_error"
    assert positive_geometry["fallback_required"] is True
    assert positive_geometry["effective_acceleration_enabled"] is False
    assert (tmp_path / "positive_geometry_kernel_error.json").exists()
    assert (tmp_path / "kernel_dispatch_report.json").exists()


def test_measurement_pack_copies_kernel_dispatch_receipt(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    dispatch_configured_kernels(
        {"kernels": {"positive_geometry": {"enabled": True}}},
        run,
        engine="unit_test",
    )

    out = tmp_path / "pack"
    pack = export_measurement_pack([run], out)

    assert (out / "kernel_dispatch_report.json").exists()
    assert (out / "kernel_dispatch_report.md").exists()
    assert pack["claims"]["kernel_dispatch_written"] is True
    assert pack["claims"]["kernel_dispatch_routing_decision"] == "generic_oph_repair_with_kernel_receipts"
    assert pack["claims"]["kernel_dispatch_effective_acceleration_enabled"] is False
    assert pack["claims"]["kernel_dispatch_physical_observables_changed"] is False
