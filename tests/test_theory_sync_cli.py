from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from oph_fpe.cli import main
from oph_fpe.cosmology.edge_center_clock import EDGE_CENTER_EVIDENCE_RECEIPTS


SCHEMA_ROOT = Path(__file__).resolve().parents[1] / "schemas"


def _validate_schema(report: dict[str, object], relative_path: str) -> None:
    schema = json.loads((SCHEMA_ROOT / relative_path).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)


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
        json.dumps(
            {
                "nodes": [
                    {"id": "source", "kind": "source"},
                    {"id": "fit", "kind": "source", "likelihood": True},
                ],
                "edges": [{"source": "source", "target": "fit"}],
            }
        ),
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


def test_boundary_fiber_cli_emits_nonpromoting_replay_artifact(tmp_path) -> None:
    packet = tmp_path / "boundary.json"
    packet.write_text(
        json.dumps(
            {
                "fiber_rows": [
                    {
                        "record_id": "r0",
                        "boundary": "b",
                        "sector": "s",
                        "gauge_class": "g",
                    }
                ],
                "transition_rows": [{"source": "r0", "target": "r0"}],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "boundary_certificate.json"

    assert main(
        [
            "boundary-fiber-certificate",
            "--packet",
            str(packet),
            "--out",
            str(out),
        ]
    ) == 0
    report = json.loads(out.read_text(encoding="utf-8"))

    assert report["BOUNDARY_FIBER_SUPPLIED_TABLE_CONSISTENCY_RECEIPT"] is True
    assert report["BOUNDARY_CONDITIONED_UNIQUENESS_RECEIPT"] is False
    assert report["RUN_ARTIFACT_BINDING_RECEIPT"] is False
    _validate_schema(report, "consensus/boundary_fiber_certificate.schema.json")


def test_fair_block_cli_emits_arithmetic_without_consensus_promotion(tmp_path) -> None:
    packet = tmp_path / "fair.json"
    packet.write_text(
        json.dumps(
            {
                "transition_matrix": [[0.8, 0.2], [0.3, 0.7]],
                "initial_distribution": [1.0, 0.0],
                "fair_states": [1],
                "time_horizon_steps": 12,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "fair_certificate.json"

    assert main(
        ["fair-block-certificate", "--packet", str(packet), "--out", str(out)]
    ) == 0
    report = json.loads(out.read_text(encoding="utf-8"))

    assert report["FAIR_BLOCK_MARKOV_RECOMPUTATION_RECEIPT"] is True
    assert report["FAIR_BLOCK_CONSENSUS_CERTIFICATE"] is False
    assert report["RUN_ARTIFACT_BINDING_RECEIPT"] is False
    _validate_schema(report, "consensus/fair_block_certificate.schema.json")


def test_collar_poisson_cli_emits_arithmetic_without_physical_promotion(
    tmp_path,
) -> None:
    packet = tmp_path / "poisson.json"
    packet.write_text(
        json.dumps(
            {
                "activation_probabilities": [0.02] * 25,
                "limiting_mean": 0.5,
                "cut_sqrt_measure": 2.0,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "poisson_certificate.json"

    assert main(
        [
            "collar-poisson-certificate",
            "--packet",
            str(packet),
            "--out",
            str(out),
        ]
    ) == 0
    report = json.loads(out.read_text(encoding="utf-8"))

    assert report["COLLAR_POISSON_COUNTING_RECOMPUTATION_RECEIPT"] is True
    assert report["PHYSICAL_COLLAR_MODEL_REALIZATION_RECEIPT"] is False
    assert report["RUN_ARTIFACT_BINDING_RECEIPT"] is False
    _validate_schema(report, "cosmology/collar_poisson_certificate.schema.json")


def test_theory_certificate_cli_rejects_oversized_input_before_json_decode(
    tmp_path,
) -> None:
    packet = tmp_path / "oversized.json"
    packet.write_bytes(b" " * 1_000_001)

    with pytest.raises(ValueError, match="command limit"):
        main(
            [
                "collar-poisson-certificate",
                "--packet",
                str(packet),
                "--out",
                str(tmp_path / "out.json"),
            ]
        )
