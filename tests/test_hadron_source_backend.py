from __future__ import annotations

from pathlib import Path

from oph_fpe.cosmology.hadron_source_backend import (
    CLAIM_TIERS,
    FORBIDDEN_SOURCE_INPUTS,
    REQUIRED_FILES,
    HadronSourceBackendInputs,
    hadron_source_backend_report,
    write_hadron_source_backend_bundle,
)


def test_default_hadron_source_backend_is_fail_closed():
    report = hadron_source_backend_report()

    assert report["mode"] == "oph_qcd_hadron_source_backend_v1"
    assert report["milestone"] == "HVP_ALPHA_SOURCE_PROTOTYPE"
    assert report["claim"] == "SOURCE_PROTOTYPE_NOT_PROMOTED"
    assert report["claim_tier"] == "H2"
    assert set(report["claim_tiers"]) == set(CLAIM_TIERS)
    assert report["promotion_allowed"] is False
    assert report["source_open"] is True
    assert report["forbidden_source_inputs"] == list(FORBIDDEN_SOURCE_INPUTS)
    assert report["readiness_gates"]["two_current_hadronic_backend_receipt"] is False
    assert report["readiness_gates"]["full_hadronic_precision_backend_receipt"] is False
    assert report["readiness_gates"]["fine_structure_endpoint_promotion_receipt"] is False
    assert "source_qcd_law_not_promoted" in report["blockers"]


def test_hadron_source_backend_bundle_writes_required_receipts(tmp_path: Path):
    report = write_hadron_source_backend_bundle(tmp_path)

    assert (tmp_path / "hadron_source_backend_report.json").exists()
    assert (tmp_path / "hadron_source_backend_report.md").exists()
    assert (tmp_path / "claim.md").read_text(encoding="utf-8") == "SOURCE_PROTOTYPE_NOT_PROMOTED\n"
    assert report["manifest"]["missing_files"] == []
    for rel_path in REQUIRED_FILES:
        assert (tmp_path / rel_path).exists(), rel_path
    assert "manifest.json" in report["manifest"]["file_hashes"]
    assert report["readiness_gates"]["forbidden_source_inputs_excluded"] is True


def test_source_interval_promoted_sets_backend_receipts_but_not_endpoint_predictions(tmp_path: Path):
    report = write_hadron_source_backend_bundle(
        tmp_path,
        HadronSourceBackendInputs(claim="SOURCE_INTERVAL_PROMOTED", tier="H7"),
    )

    assert report["promotion_allowed"] is True
    assert report["source_open"] is False
    assert report["readiness_gates"]["two_current_hadronic_backend_receipt"] is True
    assert report["readiness_gates"]["full_hadronic_precision_backend_receipt"] is True
    assert report["readiness_gates"]["fine_structure_endpoint_promotion_receipt"] is False


def test_hadron_source_backend_lazy_export_and_measurement_pack(tmp_path: Path):
    from oph_fpe.cosmology import hadron_source_backend_report as exported_report
    from oph_fpe.measurement_pack import export_measurement_pack

    run = tmp_path / "run"
    pack_dir = tmp_path / "pack"
    write_hadron_source_backend_bundle(run)
    pack = export_measurement_pack([run], pack_dir)

    assert exported_report()["mode"] == "oph_qcd_hadron_source_backend_v1"
    assert (pack_dir / "hadron_source_backend_report.json").exists()
    assert (pack_dir / "hadron_source_backend_report.md").exists()
    assert pack["claims"]["hadron_source_backend_written"] is True
    assert pack["claims"]["hadron_source_backend_claim"] == "SOURCE_PROTOTYPE_NOT_PROMOTED"
    assert pack["claims"]["hadron_source_backend_two_current_receipt"] is False
    assert pack["claims"]["hadron_source_backend_full_precision_receipt"] is False
