from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cli import main
from oph_fpe.jwst import (
    REQUIRED_SCHEMA_FILES,
    write_degeneracy_audit_report,
    write_simulation_plan_report,
    write_source_artifact_report,
)


def test_jwst_source_artifact_is_fail_closed_and_detects_target_leakage(tmp_path: Path):
    clean = write_source_artifact_report(tmp_path / "clean")

    assert clean["mode"] == "jwst_object_source_artifact_v1"
    assert clean["readiness_gates"]["NO_TARGET_LEAKAGE_RECEIPT"] is True
    assert clean["readiness_gates"]["OBJECT_SOURCE_LAW_RECEIPT"] is False
    assert "OBJECT_SOURCE_LAW_RECEIPT_missing" in clean["blockers"]

    config = tmp_path / "leaky_source.json"
    config.write_text(json.dumps({"source_inputs": ["jwst_catalog_counts"]}), encoding="utf-8")
    leaky = write_source_artifact_report(tmp_path / "leaky", config=config)

    assert leaky["readiness_gates"]["NO_TARGET_LEAKAGE_RECEIPT"] is False
    assert "jwst_catalog" in leaky["inputs"]["target_leak_hits"]
    assert "catalog_counts" in leaky["inputs"]["target_leak_hits"]
    assert "NO_TARGET_LEAKAGE_RECEIPT_missing" in leaky["blockers"]


def test_jwst_schema_files_are_present():
    repo_root = Path(__file__).resolve().parents[1]

    for rel_path in REQUIRED_SCHEMA_FILES:
        assert (repo_root / rel_path).is_file(), rel_path


def test_jwst_degeneracy_audit_blocks_mass_age_overpromotion(tmp_path: Path):
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        json.dumps(
            {
                "object_id": "synthetic_pair_1",
                "synthetic_degeneracy_pair": True,
                "claim_label": "PHYSICAL_MASS_AGE_TENSION",
                "receipts": {"DUST_RECEIPT": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = write_degeneracy_audit_report(tmp_path / "audit", catalog=catalog)

    assert report["synthetic_degeneracy_pair_count"] == 1
    assert report["overpromoted_object_ids"] == ["synthetic_pair_1"]
    assert report["recommended_label_when_open"] == "DEGENERACY_OPEN"
    assert report["readiness_gates"]["MASS_AGE_TENSION_PROMOTION_RECEIPT"] is False
    assert report["readiness_gates"]["SYNTHETIC_DEGENERACY_PAIR_GUARD_RECEIPT"] is False
    assert "degeneracy_overpromotion_detected" in report["blockers"]


def test_jwst_cli_plan_recomputes_strongest_allowed_claim(tmp_path: Path):
    source_dir = tmp_path / "source"
    plan_dir = tmp_path / "plan"

    assert main(["jwst-object-source-artifact", "--out", str(source_dir)]) == 0
    assert (source_dir / "jwst_object_source_artifact_report.json").exists()

    assert main(["jwst-compact-object-simulation-plan", "--run-dir", str(tmp_path), "--out", str(plan_dir)]) == 0
    plan = json.loads((plan_dir / "jwst_compact_object_simulation_plan.json").read_text(encoding="utf-8"))

    assert plan["strongest_allowed_claim"] == "J0_DIAGNOSTIC_PROXY"
    assert plan["first_blocked_gate"] == "CATALOG_INGESTION_RECEIPT"
    assert plan["next_required_command"] == "jwst-degeneracy-audit"


def test_jwst_measurement_pack_copies_reports_and_claims(tmp_path: Path):
    from oph_fpe.measurement_pack import export_measurement_pack

    run = tmp_path / "run"
    pack_dir = tmp_path / "pack"
    write_source_artifact_report(run / "source")
    write_degeneracy_audit_report(run / "audit", catalog=tmp_path / "missing_catalog.jsonl")
    write_simulation_plan_report(run / "plan", run_dir=run)

    pack = export_measurement_pack([run], pack_dir)

    assert (pack_dir / "jwst_object_source_artifact_report.json").exists()
    assert (pack_dir / "jwst_degeneracy_audit_report.json").exists()
    assert (pack_dir / "jwst_compact_object_simulation_plan.json").exists()
    assert "JWST compact-object strongest claim" in (pack_dir / "README.md").read_text(encoding="utf-8")
    assert pack["claims"]["jwst_compact_object_plan_written"] is True
    assert pack["claims"]["jwst_compact_object_strongest_allowed_claim"] == "J0_DIAGNOSTIC_PROXY"
    assert pack["claims"]["jwst_no_target_leakage_receipt"] is True
    assert pack["claims"]["jwst_degeneracy_audit_receipt"] is False
