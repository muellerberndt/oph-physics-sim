"""Frozen tests for the issue-634 stage-4 inhabitation layer."""

import hashlib
import json
import shutil
from pathlib import Path

from oph_fpe.local_domain.stage4_inhabitation import (
    _recompute_verdict,
    verify_local_domain_bundle,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "local_domain"


def test_bundle_verifier_passes_on_frozen_artifacts():
    result = verify_local_domain_bundle()
    assert result["passed"], result["blockers"]
    assert result["stage_verdicts"] == {
        "stage1": "ATTAINED",
        "stage2": "ATTAINED",
        "stage3": "ATTAINED",
    }
    assert result["capture_sha256"].startswith("sha256:")


def test_verdict_recomputation_ignores_receipt_booleans():
    receipt = {
        "clause_verdicts": {"a": True, "b": False},
        "negative_controls": {"c": {"control_failure_detected": True}},
        "blockers": [],
        "verdict": "ATTAINED",
    }
    assert _recompute_verdict(receipt) == "NOT_ATTAINED"
    receipt["clause_verdicts"]["b"] = True
    assert _recompute_verdict(receipt) == "ATTAINED"
    receipt["negative_controls"]["c"]["control_failure_detected"] = False
    assert _recompute_verdict(receipt) == "NOT_ATTAINED"


def test_bundle_verifier_fails_closed_on_tampering(tmp_path):
    for name in (
        "manifest.json",
        "stage1_arrays.npz.gz",
        "stage1_receipt.json",
        "stage2_receipt.json",
        "stage3_receipt.json",
    ):
        shutil.copyfile(DATA_DIR / name, tmp_path / name)
    assert verify_local_domain_bundle(tmp_path)["passed"]

    receipt_path = tmp_path / "stage3_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["verdict"] = "NOT_ATTAINED"
    raw = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    receipt_path.write_bytes(raw)
    result = verify_local_domain_bundle(tmp_path)
    assert not result["passed"]
    assert "stage3_receipt_hash_mismatch" in result["blockers"]

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["stage3_receipt_sha256"] = (
        "sha256:" + hashlib.sha256(raw).hexdigest()
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n"
    )
    result = verify_local_domain_bundle(tmp_path)
    assert not result["passed"]
    assert "stage3_verdict_disagrees_with_recomputation" in result["blockers"]


def test_frozen_stage4_receipt_binding():
    manifest = json.loads(
        (DATA_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    receipt_bytes = (DATA_DIR / "stage4_receipt.json").read_bytes()
    assert manifest["stage4_receipt_sha256"] == "sha256:" + hashlib.sha256(
        receipt_bytes
    ).hexdigest()
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    assert receipt["schema"] == "oph.local-domain-stage4.v1"
    assert receipt["issue"] == 634
    assert receipt["physical_promotion_allowed"] is False
    assert receipt["verdict"] == "ATTAINED"
    assert receipt["blockers"] == []
    assert all(receipt["clause_verdicts"].values())
    assert receipt["independent_replay"]["all_byte_exact"] is True
    assert all(
        row["detected"] for row in receipt["issue_control_matrix"].values()
    )
    assert all(
        row["holds"] for row in receipt["preservation_squares"].values()
    )
    assert receipt["provenance_dag"]["acyclic"] is True
    stage1 = json.loads(
        (DATA_DIR / "stage1_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["capture_sha256"] == stage1["capture_sha256"]
