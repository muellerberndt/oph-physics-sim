from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.frozen_transfer_likelihood import (
    COMPARISON_OBSERVABLES,
    SOURCE_REPORT_NAMES,
    FrozenTransferConfig,
    frozen_transfer_likelihood_report,
    write_frozen_transfer_likelihood_report,
)


def test_frozen_transfer_likelihood_fails_closed_without_sources_or_pins(tmp_path: Path) -> None:
    report = frozen_transfer_likelihood_report([tmp_path])

    assert report["FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_RECEIPT"] is False
    assert report["physical_cmb_prediction"] is False
    assert "source_freeze_manifest_not_certified" in report["blockers"]
    assert "solver_assumption_pin_not_certified" in report["blockers"]
    assert "custom_parent_cdm_limit_regression_not_passed" in report["blockers"]
    assert "standard_model_off_regression_not_passed" in report["blockers"]
    assert "full_observable_likelihood_not_executed" in report["blockers"]


def test_frozen_transfer_likelihood_closure_passes_with_complete_frozen_lane(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_source_allowlist(run)
    _write_json(
        run / "camb_lcdm_baseline_report.json",
        {
            "CDM_LIMIT_BOLTZMANN_RECEIPT": True,
            "software": {"camb_version": "1.5.0"},
        },
    )
    _write_json(
        run / "cmb1_cdm_limit_regression_report.json",
        {
            "CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT": True,
            "solver_native_cdm_receipt": True,
            "custom_parent_cdm_receipt": True,
            "max_relative_tt_delta": 1.0e-8,
        },
    )
    _write_json(
        run / "standard_model_off_regression_report.json",
        {
            "STANDARD_MODEL_OFF_REGRESSION_RECEIPT": True,
            "standard_model_sector_off": True,
            "oph_anomaly_sector_off": True,
            "no_particle_sector_sources": True,
            "max_relative_tt_delta": 1.0e-8,
        },
    )
    _write_json(
        run / "official_planck_likelihood_readiness_report.json",
        {"official_likelihood_execution_ready": True, "software": {"camb_version": "1.5.0"}},
    )
    _write_json(
        run / "official_likelihood_execution_report.json",
        {
            "official_likelihood_execution_ready": True,
            "OFFICIAL_LIKELIHOOD_EXECUTION_RECEIPT": True,
            "BLINDED_COMPARISON_SETUP_RECEIPT": True,
            "FULL_OBSERVABLE_LIKELIHOOD_RECEIPT": True,
            "observables": {name: True for name in COMPARISON_OBSERVABLES},
            "likelihood_hash": _hash("9"),
            "software": {"camb_version": "1.5.0"},
        },
    )

    report = write_frozen_transfer_likelihood_report(
        [run],
        out,
        config=FrozenTransferConfig(
            solver="CAMB",
            solver_version_pin="1.5.0",
            source_plugin_hash=_hash("8"),
            recombination_assumption="HyRec-compatible CAMB recombination pin",
            neutrino_assumption="sum_mnu_0.06eV_one_massive_two_massless",
            tolerance=1.0e-5,
        ),
    )

    assert report["FROZEN_SOURCE_MANIFEST_RECEIPT"] is True
    assert report["SOLVER_ASSUMPTION_PIN_RECEIPT"] is True
    assert report["CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT"] is True
    assert report["STANDARD_MODEL_OFF_REGRESSION_RECEIPT"] is True
    assert report["BLINDED_COMPARISON_SETUP_RECEIPT"] is True
    assert report["FULL_OBSERVABLE_LIKELIHOOD_RECEIPT"] is True
    assert report["FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT"] is True
    assert report["FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_RECEIPT"] is True
    assert report["blockers"] == []
    assert report["frozen_source_hash"].startswith("sha256:")
    assert (out / "frozen_transfer_likelihood_report.json").exists()
    assert (out / "frozen_source_manifest.csv").exists()


def _write_source_allowlist(run: Path) -> None:
    for name in SOURCE_REPORT_NAMES:
        _write_json(run / name, {"mode": name.removesuffix(".json"), "source_side": True})


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _hash(char: str) -> str:
    return "sha256:" + char * 64
