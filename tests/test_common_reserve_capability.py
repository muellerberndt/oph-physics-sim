from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from oph_fpe import cli
from oph_fpe.common_reserve import capability
from oph_fpe.common_reserve import verify_capability_independent as independent
from oph_fpe.core.charged_response import produce_charged_response_artifact


MATRIX = capability.build_capability_matrix()


def _rehash(matrix: dict) -> None:
    payload = copy.deepcopy(matrix)
    payload.pop("payload_sha256", None)
    matrix["payload_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _row(capability_id: str) -> dict:
    return next(row for row in MATRIX["capabilities"] if row["capability_id"] == capability_id)


def test_cr0_matrix_is_strict_and_independently_verified() -> None:
    schema = json.loads(capability.SCHEMA_PATH.read_text("utf-8"))
    Draft202012Validator(schema).validate(MATRIX)
    verification = independent.verify_capability_matrix(MATRIX)
    assert verification["receipt"] is True, verification["reasons"]
    assert MATRIX["scientific_promotion_allowed"] is False
    assert MATRIX["lane_stop_rules"]["cr1_or_later_implemented_here"] is False
    assert MATRIX["lane_stop_rules"]["large_simulation_authorized"] is False


def test_classifications_are_exact_and_lane_local() -> None:
    assert MATRIX["classification_counts"] == {
        "AVAILABLE_SIMULATOR_NATIVE": 2,
        "AVAILABLE_CONDITIONAL": 1,
        "AVAILABLE_CONTROL_ONLY": 1,
        "MISSING": 7,
        "AMBIGUOUS": 0,
    }
    assert _row("primitive_repair_law")["classification"] == (
        capability.AVAILABLE_SIMULATOR_NATIVE
    )
    assert _row("raw_twelve_port_response")["classification"] == (
        capability.AVAILABLE_SIMULATOR_NATIVE
    )
    assert _row("physical_quotient")["classification"] == capability.MISSING
    assert _row("boundary_sector")["classification"] == capability.AVAILABLE_CONDITIONAL
    assert _row("refinement_tower_physical_scale_ratios")["classification"] == (
        capability.MISSING
    )
    assert _row("source_ensemble_action")["classification"] == (
        capability.AVAILABLE_CONTROL_ONLY
    )
    assert {
        row["capability_id"]
        for row in MATRIX["capabilities"]
        if row["classification"] == capability.MISSING
    } == {
        "scalar_register",
        "protected_z6_reserve",
        "scalar_reserve_coregistration",
        "physical_quotient",
        "refinement_tower_physical_scale_ratios",
        "support_geometry",
        "full_half_collars",
    }
    assert MATRIX["lane_stop_rules"] == {
        "reserve_lane_blocked": True,
        "cocycle_lane_blocked": True,
        "screen_lane_blocked": True,
        "raw_response_archived_for_later_bridge": True,
        "independent_internal_repair_audits_may_continue": True,
        "large_simulation_authorized": False,
        "cr1_or_later_implemented_here": False,
    }


def test_exact_response_probe_is_recomputed_from_raw_carrier() -> None:
    probe = MATRIX["raw_twelve_port_response_probe"]
    payload = probe["recurrence_trace_payload"]
    assert probe["source_artifact_schema"] == "oph.charged_response_semantic_artifact.v3"
    assert probe["response_operator"] == "negative_graph_antipode_involution"
    assert probe["runtime_response_source"] == "finite_unitary_carrier_channel"
    assert probe["finite_simulator_response_identified"] is True
    assert probe["current_lift_source_selected"] is False
    assert probe["physical_A_T_identification"] is False
    assert len(payload["adjacency_powers_k0_through_k3"]) == 4
    assert len(payload["distance_rows"]) == 12
    assert len(set(payload["farthest_port_map"])) == 12
    assert probe["antipode_port_map"] == payload["farthest_port_map"]
    assert independent.verify_capability_matrix(MATRIX)["receipt"] is True


def test_pinned_charged_response_artifact_is_exact_producer_output() -> None:
    manifest = json.loads(
        (capability.REPOSITORY_ROOT / capability.SOURCE_PATHS["carrier_manifest"]).read_text(
            "utf-8"
        )
    )
    committed = json.loads(
        (
            capability.REPOSITORY_ROOT
            / capability.SOURCE_PATHS["charged_response_artifact"]
        ).read_text("utf-8")
    )
    assert committed == produce_charged_response_artifact(manifest)
    assert MATRIX["sources"]["charged_response_artifact"]["sha256"].startswith("sha256:")


def test_partial_quotient_and_unjoined_refinements_cannot_be_promoted() -> None:
    for capability_id in (
        "physical_quotient",
        "refinement_tower_physical_scale_ratios",
    ):
        mutated = copy.deepcopy(MATRIX)
        row = next(
            item for item in mutated["capabilities"] if item["capability_id"] == capability_id
        )
        row["classification"] = capability.AVAILABLE_CONDITIONAL
        mutated["classification_counts"][capability.MISSING] -= 1
        mutated["classification_counts"][capability.AVAILABLE_CONDITIONAL] += 1
        _rehash(mutated)
        result = independent.verify_capability_matrix(mutated)
        assert result["receipt"] is False
        assert f"classification_mismatch:{capability_id}" in result["reasons"]


def test_placeholder_cannot_promote_missing_reserve() -> None:
    mutated = copy.deepcopy(MATRIX)
    reserve = next(
        row for row in mutated["capabilities"] if row["capability_id"] == "protected_z6_reserve"
    )
    reserve["classification"] = capability.AVAILABLE_SIMULATOR_NATIVE
    reserve["verified_evidence"].append("placeholder random variable")
    reserve["missing_evidence"] = []
    mutated["classification_counts"][capability.MISSING] -= 1
    mutated["classification_counts"][capability.AVAILABLE_SIMULATOR_NATIVE] += 1
    _rehash(mutated)

    result = independent.verify_capability_matrix(mutated)
    assert result["receipt"] is False
    assert "classification_mismatch:protected_z6_reserve" in result["reasons"]


def test_added_adapter_source_is_rejected_even_with_a_fresh_digest() -> None:
    mutated = copy.deepcopy(MATRIX)
    mutated["sources"]["reserve_adapter"] = copy.deepcopy(
        mutated["sources"]["reference_ensemble_producer"]
    )
    reserve = next(
        row for row in mutated["capabilities"] if row["capability_id"] == "protected_z6_reserve"
    )
    reserve["source_pin_ids"].append("reserve_adapter")
    _rehash(mutated)

    result = independent.verify_capability_matrix(mutated)
    assert result["receipt"] is False
    assert "source_catalog_mismatch" in result["reasons"]


def test_source_pin_trace_and_status_mutations_fail_closed() -> None:
    pin_mutation = copy.deepcopy(MATRIX)
    pin_mutation["sources"]["carrier_manifest"]["sha256"] = "sha256:" + "0" * 64
    _rehash(pin_mutation)
    pin_result = independent.verify_capability_matrix(pin_mutation)
    assert pin_result["receipt"] is False
    assert "source_sha256_mismatch:carrier_manifest" in pin_result["reasons"]

    trace_mutation = copy.deepcopy(MATRIX)
    trace = trace_mutation["raw_twelve_port_response_probe"]["recurrence_trace_payload"]
    trace["adjacency_powers_k0_through_k3"][1][0][0] = 1
    trace_mutation["raw_twelve_port_response_probe"]["recurrence_trace_payload_sha256"] = (
        "sha256:" + hashlib.sha256(
            json.dumps(
                trace,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    _rehash(trace_mutation)
    trace_result = independent.verify_capability_matrix(trace_mutation)
    assert trace_result["receipt"] is False
    assert "raw_twelve_port_response_probe_mismatch" in trace_result["reasons"]

    response_mutation = copy.deepcopy(MATRIX)
    response_mutation["raw_twelve_port_response_probe"]["antipode_port_map"][0] = 0
    _rehash(response_mutation)
    response_result = independent.verify_capability_matrix(response_mutation)
    assert response_result["receipt"] is False
    assert "raw_twelve_port_response_probe_mismatch" in response_result["reasons"]


def test_target_ancestry_and_schema_mutations_fail_closed() -> None:
    ancestry = copy.deepcopy(MATRIX)
    ancestry["target_ancestry"]["root_modules"].append("oph_fpe.constants.oph_pixel")
    _rehash(ancestry)
    result = independent.verify_capability_matrix(ancestry)
    assert result["receipt"] is False
    assert "target_ancestry_mismatch:root_modules" in result["reasons"]

    extra = copy.deepcopy(MATRIX)
    extra["placeholder_availability"] = True
    _rehash(extra)
    result = independent.verify_capability_matrix(extra)
    assert result["receipt"] is False
    assert any(reason.startswith("schema:") for reason in result["reasons"])


def test_writer_report_cli_and_committed_artifacts_are_deterministic(tmp_path: Path) -> None:
    out = tmp_path / "producer_capability_matrix.json"
    report = tmp_path / "producer_capability_matrix.md"
    written = capability.write_capability_matrix(out, report)
    assert written == MATRIX
    assert json.loads(out.read_text("utf-8")) == MATRIX
    assert report.read_text("utf-8") == capability.render_report(MATRIX)
    assert "`MISSING`" in report.read_text("utf-8")

    cli_out = tmp_path / "cli-matrix.json"
    cli_report = tmp_path / "cli-report.md"
    assert cli.main(
        [
            "common-reserve-capability",
            "--out",
            str(cli_out),
            "--report",
            str(cli_report),
        ]
    ) == 0
    assert json.loads(cli_out.read_text("utf-8")) == MATRIX

    committed = json.loads(capability.DEFAULT_OUTPUT.read_text("utf-8"))
    assert committed == MATRIX
    assert capability.DEFAULT_REPORT.read_text("utf-8") == capability.render_report(MATRIX)
    assert independent.verify_file(capability.DEFAULT_OUTPUT)["receipt"] is True


def test_cr0_does_not_create_later_stage_producers() -> None:
    package = capability.Path(capability.__file__).resolve().parent
    forbidden = {
        "reserve_native.py",
        "cocycle_native.py",
        "screen_native.py",
        "controls.py",
        "reduce.py",
    }
    assert not forbidden.intersection(path.name for path in package.iterdir())
