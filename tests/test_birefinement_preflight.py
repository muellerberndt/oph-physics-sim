from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from oph_fpe.refinement import birefinement_preflight as producer
from oph_fpe.refinement import verify_birefinement_preflight_independent as independent


RECEIPT = producer.build_birefinement_preflight()


def _rehash(receipt: dict) -> None:
    payload = copy.deepcopy(receipt)
    payload.pop("payload_sha256", None)
    receipt["payload_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_receipt_is_strict_and_independently_verified() -> None:
    schema = json.loads(producer.SCHEMA_PATH.read_text("utf-8"))
    Draft202012Validator(schema).validate(RECEIPT)
    verification = independent.verify_birefinement_preflight(RECEIPT)
    assert verification["receipt"] is True, verification["reasons"]
    assert verification["verdict"] == "VERIFIED_SOURCE_PRODUCER_MISSING"


def test_every_issue_requirement_maps_to_concrete_cr0_rows_and_is_blocked() -> None:
    rows = RECEIPT["acceptance_requirements"]
    assert [row["requirement_id"] for row in rows] == [
        item[0] for item in producer.REQUIREMENTS
    ]
    assert len(rows) == 7
    assert all(row["capability_rows"] for row in rows)
    assert all(row["blocking_capability_ids"] for row in rows)
    assert all(row["requirement_satisfied"] is False for row in rows)
    assert set(RECEIPT["blocking_capability_ids"]) == {
        "physical_quotient",
        "refinement_tower_physical_scale_ratios",
        "scalar_register",
        "source_ensemble_action",
        "support_geometry",
    }


def test_controls_adapters_and_comparison_access_are_refused() -> None:
    policy = RECEIPT["admission_policy"]
    assert policy["admitted_classifications"] == ["AVAILABLE_SIMULATOR_NATIVE"]
    assert policy["conditional_objects_admitted"] is False
    assert policy["control_objects_admitted"] is False
    assert policy["adapter_promotion_admitted"] is False
    assert policy["comparison_access"] == "REFUSED"
    assert policy["producer_input_paths"] == [
        "data/common_reserve/producer_capability_matrix.json"
    ]
    assert all(
        policy[key] is False
        for key in (
            "network_accessed",
            "environment_inputs_accessed",
            "public_data_accessed",
            "measurement_data_accessed",
            "target_values_accessed",
        )
    )


def test_no_eigenvalue_exponent_or_comparison_value_is_emitted() -> None:
    assert RECEIPT["scientific_outputs"] == {
        "produced": False,
        "numeric_value_count": 0,
        "covariance_eigenvalues_emitted": False,
        "scaling_exponents_emitted": False,
        "comparison_statistics_emitted": False,
    }
    serialized = json.dumps(RECEIPT, sort_keys=True)
    for forbidden_key in ("lambda_2", "lambda_3", "theta_2", "theta_3"):
        assert forbidden_key not in serialized


def test_external_theorem_packet_is_not_falsely_byte_verified() -> None:
    prerequisite = RECEIPT["external_mathematical_prerequisite"]
    assert prerequisite["issue"] == 656
    assert prerequisite["local_source_projection_vendored"] is False
    assert prerequisite["sibling_repository_bytes_verified"] is False
    assert prerequisite["physical_source_producer_supplied"] is False


def test_status_and_requirement_promotions_fail_closed() -> None:
    status = copy.deepcopy(RECEIPT)
    status["status"] = "PURE_POWER_RIGIDITY_ATTAINED"
    status["decision"]["verdict"] = "PURE_POWER_RIGIDITY_ATTAINED"
    _rehash(status)
    result = independent.verify_birefinement_preflight(status)
    assert result["receipt"] is False
    assert any(reason.startswith("schema:") for reason in result["reasons"])

    requirement = copy.deepcopy(RECEIPT)
    row = requirement["acceptance_requirements"][0]
    row["requirement_satisfied"] = True
    row["blocking_capability_ids"] = []
    _rehash(requirement)
    result = independent.verify_birefinement_preflight(requirement)
    assert result["receipt"] is False
    assert "acceptance_requirement_mapping_mismatch" in result["reasons"]


def test_control_or_adapter_promotion_fails_closed() -> None:
    control = copy.deepcopy(RECEIPT)
    control["admission_policy"]["control_objects_admitted"] = True
    _rehash(control)
    result = independent.verify_birefinement_preflight(control)
    assert result["receipt"] is False
    assert any(reason.startswith("schema:") for reason in result["reasons"])

    adapter = copy.deepcopy(RECEIPT)
    adapter["admission_policy"]["adapter_promotion_admitted"] = True
    _rehash(adapter)
    result = independent.verify_birefinement_preflight(adapter)
    assert result["receipt"] is False
    assert any(reason.startswith("schema:") for reason in result["reasons"])


def test_comparison_and_output_promotions_fail_closed() -> None:
    comparison = copy.deepcopy(RECEIPT)
    comparison["admission_policy"]["comparison_access"] = "OPEN"
    comparison["admission_policy"]["measurement_data_accessed"] = True
    _rehash(comparison)
    result = independent.verify_birefinement_preflight(comparison)
    assert result["receipt"] is False
    assert any(reason.startswith("schema:") for reason in result["reasons"])

    output = copy.deepcopy(RECEIPT)
    output["scientific_outputs"]["produced"] = True
    output["scientific_outputs"]["numeric_value_count"] = 1
    output["scientific_outputs"]["covariance_eigenvalues_emitted"] = True
    _rehash(output)
    result = independent.verify_birefinement_preflight(output)
    assert result["receipt"] is False
    assert any(reason.startswith("schema:") for reason in result["reasons"])


def test_false_sibling_byte_verification_and_pin_mutation_fail_closed() -> None:
    sibling = copy.deepcopy(RECEIPT)
    sibling["external_mathematical_prerequisite"][
        "sibling_repository_bytes_verified"
    ] = True
    _rehash(sibling)
    result = independent.verify_birefinement_preflight(sibling)
    assert result["receipt"] is False
    assert any(reason.startswith("schema:") for reason in result["reasons"])

    pin = copy.deepcopy(RECEIPT)
    pin["sources"]["cr0_capability_matrix"]["sha256"] = "sha256:" + "0" * 64
    _rehash(pin)
    result = independent.verify_birefinement_preflight(pin)
    assert result["receipt"] is False
    assert "cr0_source_pin_mismatch" in result["reasons"]


def test_writer_report_and_committed_artifacts_are_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    report = tmp_path / "report.md"
    written = producer.write_birefinement_preflight(output, report)
    assert written == RECEIPT
    assert json.loads(output.read_text("utf-8")) == RECEIPT
    assert report.read_text("utf-8") == producer.render_report(RECEIPT)
    assert "SOURCE_PRODUCER_MISSING" in report.read_text("utf-8")

    committed = json.loads(producer.DEFAULT_OUTPUT.read_text("utf-8"))
    assert committed == RECEIPT
    assert producer.DEFAULT_REPORT.read_text("utf-8") == producer.render_report(RECEIPT)
    verification = independent.verify_file(producer.DEFAULT_OUTPUT)
    assert verification["receipt"] is True, verification["reasons"]
