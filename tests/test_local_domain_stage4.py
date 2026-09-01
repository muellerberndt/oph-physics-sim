"""Frozen tests for the stage-4 inhabitation layer."""

import gzip
import hashlib
import io
import json
import shutil
from pathlib import Path

import pytest
import numpy as np

from oph_fpe.local_domain.stage4_inhabitation import (
    _recompute_verdict,
    _semantic_receipt,
    produce_stage4_receipt,
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
    assert result["source_projection_sha256"].startswith("sha256:")


def test_top_level_verdict_recomputed_from_recorded_summaries():
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


def test_bundle_verifier_fails_closed_on_invalid_json(
    tmp_path, monkeypatch
):
    for name in (
        "manifest.json",
        "stage1_arrays.npz.gz",
        "stage1_receipt.json",
        "stage2_receipt.json",
        "stage3_receipt.json",
    ):
        shutil.copyfile(DATA_DIR / name, tmp_path / name)

    (tmp_path / "manifest.json").write_bytes(b"{invalid")
    result = verify_local_domain_bundle(tmp_path)
    assert not result["passed"]
    assert result["blockers"] == ["manifest_json_invalid"]

    shutil.copyfile(DATA_DIR / "manifest.json", tmp_path / "manifest.json")
    invalid = b"{invalid"
    (tmp_path / "stage2_receipt.json").write_bytes(invalid)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["stage2_receipt_sha256"] = (
        "sha256:" + hashlib.sha256(invalid).hexdigest()
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n"
    )
    result = verify_local_domain_bundle(tmp_path)
    assert not result["passed"]
    assert "stage2_receipt_json_invalid" in result["blockers"]
    monkeypatch.setattr(
        "oph_fpe.local_domain.stage4_inhabitation.DATA_DIR", tmp_path
    )
    generated = produce_stage4_receipt(run_replay=False)
    assert generated["verdict"] == "NOT_ATTAINED"
    assert any(
        "stage2_receipt_json_invalid" in blocker
        for blocker in generated["blockers"]
    )


def test_bundle_verifier_rejects_manifest_schema_and_nested_type(tmp_path):
    for name in (
        "manifest.json",
        "stage1_arrays.npz.gz",
        "stage1_receipt.json",
        "stage2_receipt.json",
        "stage3_receipt.json",
    ):
        shutil.copyfile(DATA_DIR / name, tmp_path / name)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["schema"] = "unexpected"
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n"
    )
    assert "manifest_schema_invalid" in verify_local_domain_bundle(
        tmp_path
    )["blockers"]

    manifest["schema"] = "oph.local-domain-stage1.manifest.v1"
    receipt_path = tmp_path / "stage2_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["stage1_binding"] = []
    raw = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    receipt_path.write_bytes(raw)
    manifest["stage2_receipt_sha256"] = (
        "sha256:" + hashlib.sha256(raw).hexdigest()
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n"
    )
    result = verify_local_domain_bundle(tmp_path)
    assert not result["passed"]
    assert any(
        blocker.startswith("cross_stage_binding_payload_invalid")
        for blocker in result["blockers"]
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("clause_verdicts", [], "stage2_clause_payload_invalid"),
        ("negative_controls", [], "stage2_control_payload_invalid"),
        (
            "negative_controls",
            {"bad": []},
            "stage2_control_payload_invalid",
        ),
    ],
)
def test_bundle_verifier_rejects_malformed_summary_types(
    tmp_path, field, value, expected
):
    for name in (
        "manifest.json",
        "stage1_arrays.npz.gz",
        "stage1_receipt.json",
        "stage2_receipt.json",
        "stage3_receipt.json",
    ):
        shutil.copyfile(DATA_DIR / name, tmp_path / name)
    receipt_path = tmp_path / "stage2_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = value
    raw = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    receipt_path.write_bytes(raw)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["stage2_receipt_sha256"] = (
        "sha256:" + hashlib.sha256(raw).hexdigest()
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n"
    )
    result = verify_local_domain_bundle(tmp_path)
    assert not result["passed"]
    assert any(
        blocker.startswith(expected) for blocker in result["blockers"]
    )


def test_bundle_verifier_rejects_rehashed_source_projection_tamper(tmp_path):
    for name in (
        "manifest.json",
        "stage1_arrays.npz.gz",
        "stage1_receipt.json",
        "stage2_receipt.json",
        "stage3_receipt.json",
    ):
        shutil.copyfile(DATA_DIR / name, tmp_path / name)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest_key = {
        "stage1": "receipt_sha256",
        "stage2": "stage2_receipt_sha256",
        "stage3": "stage3_receipt_sha256",
    }
    for stage in ("stage1", "stage2", "stage3"):
        receipt_path = tmp_path / f"{stage}_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["source_projection_sha256"] = "sha256:" + "0" * 64
        raw = (
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        receipt_path.write_bytes(raw)
        manifest[manifest_key[stage]] = (
            "sha256:" + hashlib.sha256(raw).hexdigest()
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n"
    )
    result = verify_local_domain_bundle(tmp_path)
    assert not result["passed"]
    assert (
        "stage2_nested_source_projection_binding_invalid"
        in result["blockers"]
    )


def test_bundle_verifier_rejects_rehashed_array_value_tamper(tmp_path):
    for name in (
        "manifest.json",
        "stage1_arrays.npz.gz",
        "stage1_receipt.json",
        "stage2_receipt.json",
        "stage3_receipt.json",
    ):
        shutil.copyfile(DATA_DIR / name, tmp_path / name)

    arrays_path = tmp_path / "stage1_arrays.npz.gz"
    with np.load(
        io.BytesIO(gzip.decompress(arrays_path.read_bytes())),
        allow_pickle=False,
    ) as bundle:
        arrays = {name: bundle[name].copy() for name in bundle.files}
    arrays["chart"][0, 0] += 1.0
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    tampered = gzip.compress(buffer.getvalue(), mtime=0)
    arrays_path.write_bytes(tampered)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["arrays_sha256"] = (
        "sha256:" + hashlib.sha256(tampered).hexdigest()
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n"
    )
    result = verify_local_domain_bundle(tmp_path)
    assert not result["passed"]
    assert "stage1_array_value_hash_mismatch:chart" in result["blockers"]
    assert "stage1_chart_freeze_mismatch" in result["blockers"]


@pytest.mark.parametrize(
    ("path", "expected_blocker"),
    [
        (
            (
                "prescribed_chart_rank_certificate",
                "prescribed_four_coordinate_chart_nondegenerate",
            ),
            "stage1_clause_content_mismatch:"
            "prescribed_four_coordinate_chart_nondegenerate",
        ),
        (
            ("causal_certificates", "acyclic"),
            "stage1_clause_content_mismatch:causal_order_acyclic",
        ),
    ],
)
def test_bundle_verifier_rejects_rehashed_supporting_predicate_tamper(
    tmp_path, path, expected_blocker
):
    for name in (
        "manifest.json",
        "stage1_arrays.npz.gz",
        "stage1_receipt.json",
        "stage2_receipt.json",
        "stage3_receipt.json",
    ):
        shutil.copyfile(DATA_DIR / name, tmp_path / name)
    receipt_path = tmp_path / "stage1_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[path[0]][path[1]] = False
    raw = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    receipt_path.write_bytes(raw)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["receipt_sha256"] = (
        "sha256:" + hashlib.sha256(raw).hexdigest()
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n"
    )
    result = verify_local_domain_bundle(tmp_path)
    assert not result["passed"]
    assert expected_blocker in result["blockers"]


def test_bundle_verifier_rejects_rehashed_inertia_gate_and_result_tamper(
    tmp_path,
):
    for name in (
        "manifest.json",
        "stage1_arrays.npz.gz",
        "stage1_receipt.json",
        "stage2_receipt.json",
        "stage3_receipt.json",
    ):
        shutil.copyfile(DATA_DIR / name, tmp_path / name)
    receipt_path = tmp_path / "stage1_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["declared_acceptance_gates"][
        "held_out_feature_form_target_inertia"
    ] = [0, 4]
    receipt["held_out_quadratic_fit"]["inertia"] = [0, 4]
    raw = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    receipt_path.write_bytes(raw)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["receipt_sha256"] = (
        "sha256:" + hashlib.sha256(raw).hexdigest()
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n"
    )
    result = verify_local_domain_bundle(tmp_path)
    assert not result["passed"]
    assert (
        "stage1_clause_content_mismatch:"
        "held_out_feature_form_inertia_1_3"
    ) in result["blockers"]


def test_semantic_replay_ignores_only_full_capture_diagnostics():
    left = {
        "capture_sha256": "sha256:a",
        "capture_sha256_role": "diagnostic role",
        "source_projection_sha256": "sha256:stable",
        "stage1_binding": {
            "stage1_capture_sha256": "sha256:a",
            "stage2_capture_sha256": "sha256:a",
        },
        "scientific_payload": {
            "capture_sha256": "sha256:scientific",
            "value": 7,
        },
    }
    right = {
        "capture_sha256": "sha256:b",
        "capture_sha256_role": "diagnostic role",
        "source_projection_sha256": "sha256:stable",
        "stage1_binding": {
            "stage1_capture_sha256": "sha256:b",
            "stage2_capture_sha256": "sha256:b",
        },
        "scientific_payload": {
            "capture_sha256": "sha256:scientific",
            "value": 7,
        },
    }
    assert _semantic_receipt(left) == _semantic_receipt(right)
    right["scientific_payload"]["capture_sha256"] = "sha256:tampered"
    assert _semantic_receipt(left) != _semantic_receipt(right)
    right["scientific_payload"]["capture_sha256"] = "sha256:scientific"
    right["capture_sha256_role"] = "tampered role"
    assert _semantic_receipt(left) != _semantic_receipt(right)
    right["capture_sha256_role"] = "diagnostic role"
    right["source_projection_sha256"] = "sha256:tampered"
    assert _semantic_receipt(left) != _semantic_receipt(right)


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
    assert receipt["physical_promotion_allowed"] is False
    assert receipt["verdict"] == "ATTAINED"
    assert receipt["blockers"] == []
    assert all(receipt["clause_verdicts"].values())
    replay = receipt["producer_semantic_replay"]
    assert replay["all_semantic_exact"] is True
    assert replay["producer_independence"] is False
    assert all(
        row["detected"] for row in receipt["negative_control_matrix"].values()
    )
    assert all(
        row["holds"] for row in receipt["preservation_rows"].values()
    )
    assert receipt["provenance_dag"]["acyclic"] is True
    dag = receipt["provenance_dag"]
    assert dag["references_resolve"] is True
    assert dag["outputs_unique"] is True
    assert dag["single_source"] is True
    processes = {row["process"]: row for row in dag["processes"]}
    assert processes["capture_physical_source"]["output"] == (
        "full_capture_diagnostic"
    )
    assert processes["project_local_domain_source"]["inputs"] == [
        "full_capture_diagnostic"
    ]
    assert processes["project_local_domain_source"]["output"] == (
        "source_projection"
    )
    assert processes["produce_stage1_receipt"][
        "supporting_evaluators"
    ] == ["reconstruction_module"]
    assert processes["produce_stage1_array_bundle"]["output"] == (
        "stage1_arrays"
    )
    assert dag["artifacts"]["stage1_arrays"]["kind"] == (
        "receipt_bound_array_bundle"
    )
    stage1 = json.loads(
        (DATA_DIR / "stage1_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["source_projection_sha256"] == stage1[
        "source_projection_sha256"
    ]
