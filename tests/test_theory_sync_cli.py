from __future__ import annotations

import json

from oph_fpe.cli import main
from oph_fpe.cosmology.edge_center_clock import EDGE_CENTER_EVIDENCE_RECEIPTS


def test_public_record_capacity_cli(tmp_path) -> None:
    out = tmp_path / "capacity"

    assert main(["public-record-capacity", "--out", str(out), "--capacity-dimension", "3"]) == 0
    report = json.loads((out / "public_record_capacity_report.json").read_text(encoding="utf-8"))

    assert report["evaluation"]["exact_zero_error_capacity_M0"] == 3
    assert report["physical_N_closure_receipt"] is False


def test_a5_structural_certificate_cli(tmp_path) -> None:
    out = tmp_path / "a5"

    assert main(["a5-sm-structural-certificate", "--out", str(out)]) == 0
    report = json.loads((out / "a5_sm_structural_certificate.json").read_text(encoding="utf-8"))

    assert report["A5_TWELVE_PORT_STRUCTURAL_RECEIPT"] is True
    assert report["PHYSICAL_STANDARD_MODEL_PROMOTION_RECEIPT"] is False


def test_scr330_receipt_cli_fails_closed_on_measurement_ancestry(tmp_path) -> None:
    dag = tmp_path / "dag.json"
    dag.write_text(
        json.dumps({"nodes": [{"id": "source"}, {"id": "fit", "likelihood": True}]}),
        encoding="utf-8",
    )
    out = tmp_path / "radial.json"

    assert main(
        [
            "scr330-radial-receipt",
            "--out",
            str(out),
            "--source-dag",
            str(dag),
            "--receipt",
            "SCR330_RADIAL_NULL_REPORT",
            "--claim-tier",
            "E3",
            "--claimed-pass",
        ]
    ) == 0
    report = json.loads(out.read_text(encoding="utf-8"))

    assert report["passed"] is False
    assert "measurement_fit_or_likelihood_ancestor" in report["blockers"]


def test_edge_center_cli_rejects_copied_receipts_and_digest_strings(tmp_path) -> None:
    evidence = tmp_path / "clock_evidence.json"
    evidence.write_text(
        json.dumps(
            {
                **{receipt: True for receipt in EDGE_CENTER_EVIDENCE_RECEIPTS},
                "clock_binding_sha256": "sha256:" + "a" * 64,
                "source_dag_sha256": "sha256:" + "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "clock.json"

    assert main(
        [
            "edge-center-clock-certificate",
            "--evidence",
            str(evidence),
            "--out",
            str(out),
        ]
    ) == 0
    report = json.loads(out.read_text(encoding="utf-8"))

    assert report["EDGE_CENTER_CLOCK_RECEIPT"] is False
    assert report["observed"]["clock_binding_hash_matches"] is False
    assert report["observed"]["source_dag_hash_matches"] is False
