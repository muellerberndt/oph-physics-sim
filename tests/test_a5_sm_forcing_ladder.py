from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.gauge.a5_sm_forcing_ladder import (
    EXPECTED_DAG_EDGES,
    EXPECTED_DAG_NODES,
    THEOREM_APPLICATION_ARTIFACT_TYPE,
    THEOREM_AUDIT_ARTIFACT_TYPE,
    THEOREM_AUDIT_SCHEMA_VERSION,
    TRANSFORMS,
    apply_a5_sm_theorem_shortcuts,
    validate_theorem_audit,
)


def _synthetic_theorem_audit() -> dict:
    return {
        "schema_version": THEOREM_AUDIT_SCHEMA_VERSION,
        "artifact_type": THEOREM_AUDIT_ARTIFACT_TYPE,
        "dependency_dag": {
            "verified": True,
            "node_ids": list(EXPECTED_DAG_NODES),
            "edges": [list(edge) for edge in EXPECTED_DAG_EDGES],
        },
        "theorems": {
            transform.theorem_id: {
                "theorem_id": transform.theorem_id,
                "evidence_class": "test_conditional_theorem",
                "positive_lean_proof": False,
                "verified": True,
                "hypothesis_receipts": list(transform.hypothesis_receipts),
                "conclusion_receipts": list(transform.conclusion_receipts),
                "statement_bundle_sha256": f"{index + 1:064x}",
            }
            for index, transform in enumerate(TRANSFORMS)
        },
        "master_closure_receipt_consumed": False,
    }


def _write_receipts(path: Path, receipts: dict[str, bool]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "physical_source_receipts.json").write_text(
        json.dumps(receipts),
        encoding="utf-8",
    )


def _primitive_hypotheses() -> set[str]:
    conclusions = {
        receipt for transform in TRANSFORMS for receipt in transform.conclusion_receipts
    }
    return {
        receipt
        for transform in TRANSFORMS
        for receipt in transform.hypothesis_receipts
        if receipt not in conclusions
    }


def test_verified_theorems_without_physical_hypotheses_do_not_promote(tmp_path: Path):
    _write_receipts(tmp_path, {})

    report = apply_a5_sm_theorem_shortcuts(tmp_path, _synthetic_theorem_audit())

    assert report["artifact_type"] == THEOREM_APPLICATION_ARTIFACT_TYPE
    assert all(row["passed"] is False for row in report["transforms"].values())
    assert report["PHYSICAL_FINITE_SM_Q0_CORE_RECEIPT"] is False
    assert report["THEOREM_SHORTCUT_SOURCE_LAW_PROMOTION_COUNT"] == 0
    assert report["LEAN_NO_GO_USED_FOR_POSITIVE_PROMOTION"] is False


def test_caller_booleans_cannot_discharge_theorem_hypotheses(tmp_path: Path):
    _write_receipts(tmp_path, {name: True for name in _primitive_hypotheses()})

    report = apply_a5_sm_theorem_shortcuts(tmp_path, _synthetic_theorem_audit())

    assert all(row["passed"] is False for row in report["transforms"].values())
    assert report["A5_PORT_SELECTOR_THEOREM_APPLICATION_RECEIPT"] is False
    assert report["PHYSICAL_SM_LIE_CURRENT_ALGEBRA_RECEIPT"] is False
    assert report["PHYSICAL_LOCAL_SM_LIE_CURRENT_FIBER_RECEIPT"] is False
    assert report["PHYSICAL_Z6_GLOBAL_FORM_AND_LATTICE_RECEIPT"] is False
    assert report["PHYSICAL_FINITE_SM_Q0_CORE_RECEIPT"] is False
    assert report["source_inventory"]["registered_physical_source_verifier_count"] == 0
    assert report["Q1_CLASSICAL_REGULATOR_RECEIPT"] is False
    assert report["Q4_CONTINUUM_RECEIPT"] is False


def test_one_false_physical_source_law_blocks_all_downstream_transforms(tmp_path: Path):
    hypotheses = {name: True for name in _primitive_hypotheses()}
    hypotheses["RAW_SYMMETRIC_RESPONSE_TOMOGRAPHY_RECEIPT"] = False
    _write_receipts(tmp_path, hypotheses)

    report = apply_a5_sm_theorem_shortcuts(tmp_path, _synthetic_theorem_audit())

    assert report["transforms"]["O_PORT"]["passed"] is False
    assert report["transforms"]["O_CURRENT"]["passed"] is False
    assert report["transforms"]["O_GLOBAL"]["passed"] is False
    assert report["transforms"]["O_SMCORE_Q0"]["passed"] is False


def test_forged_positive_lean_claim_invalidates_theorem_audit(tmp_path: Path):
    audit = _synthetic_theorem_audit()
    audit["theorems"][TRANSFORMS[0].theorem_id]["positive_lean_proof"] = True
    _write_receipts(tmp_path, {name: True for name in _primitive_hypotheses()})

    errors = validate_theorem_audit(audit)
    report = apply_a5_sm_theorem_shortcuts(tmp_path, audit)

    assert any(error.startswith("positive_lean_claim_forbidden") for error in errors)
    assert all(row["passed"] is False for row in report["transforms"].values())
    assert report["PHYSICAL_FINITE_SM_Q0_CORE_RECEIPT"] is False


def test_moving_master_closure_cannot_be_used_as_shortcut(tmp_path: Path):
    audit = _synthetic_theorem_audit()
    audit["master_closure_receipt_consumed"] = True
    _write_receipts(tmp_path, {name: True for name in _primitive_hypotheses()})

    report = apply_a5_sm_theorem_shortcuts(tmp_path, audit)

    assert "moving_master_closure_must_not_be_consumed" in report[
        "theorem_audit_validation_errors"
    ]
    assert report["PHYSICAL_FINITE_SM_Q0_CORE_RECEIPT"] is False
