from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.gamma_morphology import (
    REQUIRED_FILES,
    GammaMorphologyInputs,
    gamma_morphology_report,
    signed_template_amplitude_interval,
    write_gamma_morphology_bundle,
)


def test_default_gamma_morphology_is_fail_closed(tmp_path: Path) -> None:
    report = write_gamma_morphology_bundle(tmp_path)

    assert report["mode"] == "oph_gamma_morphology_v1"
    assert report["milestone"] == "Q8_GAMMA_MORPHOLOGY_AUDIT"
    assert report["strongest_allowed_claim"] == "DIAGNOSTIC_GAMMA_MAP"
    assert report["first_blocked_gate"] == "GAMMA_SOURCE_ARTIFACT_RECEIPT"
    assert report["promotion_allowed"] is False
    assert report["readiness_gates"]["GAMMA_ROUTE_DECLARATION_RECEIPT"] is True
    assert report["readiness_gates"]["GAMMA_NO_DATA_USE_RECEIPT"] is True
    assert report["readiness_gates"]["GAMMA_SOURCE_ARTIFACT_RECEIPT"] is False
    assert (tmp_path / "claim.md").read_text(encoding="utf-8") == "DIAGNOSTIC_GAMMA_MAP\n"
    assert report["manifest"]["missing_files"] == []
    for rel_path in REQUIRED_FILES:
        assert (tmp_path / rel_path).exists(), rel_path


def test_gamma_source_dag_blocks_residual_maps(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"inputs": ["gamma_residual_maps", "likelihood_values"]}), encoding="utf-8")

    report = gamma_morphology_report(GammaMorphologyInputs(config=config))

    assert report["readiness_gates"]["GAMMA_NO_DATA_USE_RECEIPT"] is False
    assert "gamma_residual_maps" in report["target_leak_hits"]
    assert "likelihood_values" in report["target_leak_hits"]
    assert "gamma_source_dag_reads_target_data" in report["blockers"]


def test_direct_anomaly_gamma_requires_em_current_receipt() -> None:
    report = gamma_morphology_report(
        GammaMorphologyInputs(direct_anomaly_gamma=True, anomaly_em_current_receipt=False)
    )

    assert report["readiness_gates"]["ANOMALY_SM_CURRENT_NULL_RECEIPT"] is False
    assert "direct_anomaly_gamma_without_em_current_theorem" in report["blockers"]


def test_signed_template_amplitude_interval() -> None:
    interval = signed_template_amplitude_interval([10.0, 20.0, 30.0], [2.0, -4.0, 0.0])

    assert interval["nonempty"] is True
    assert interval["amplitude_min"] == -5.0
    assert interval["amplitude_max"] == 5.0


def test_measurement_pack_exports_gamma_report(tmp_path: Path) -> None:
    from oph_fpe.measurement_pack import export_measurement_pack

    run = tmp_path / "run"
    pack = tmp_path / "pack"
    write_gamma_morphology_bundle(run)

    report = export_measurement_pack([run], pack)

    assert (pack / "gamma_morphology_report.json").exists()
    assert (pack / "gamma_morphology_report.md").exists()
    assert report["claims"]["gamma_morphology_written"] is True
    assert report["claims"]["gamma_morphology_strongest_allowed_claim"] == "DIAGNOSTIC_GAMMA_MAP"
    assert report["claims"]["gamma_morphology_prediction_receipt"] is False
