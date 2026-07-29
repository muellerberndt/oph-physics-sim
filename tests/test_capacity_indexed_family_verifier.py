from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from oph_fpe.cosmology.capacity_indexed_family_verifier import (
    SCHEMA_PATH,
    canonical_json_bytes,
    compact_replay_receipt,
    load_projection,
    verify_projection,
    verify_projection_file,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "data"
    / "capacity_readback"
    / "capacity_indexed_source_family_projection.json"
)
RECEIPT = (
    ROOT
    / "data"
    / "capacity_readback"
    / "capacity_indexed_source_family_independent_receipt.json"
)
MODULE = (
    ROOT
    / "oph_fpe"
    / "cosmology"
    / "capacity_indexed_family_verifier.py"
)


def _payload() -> dict:
    return load_projection(FIXTURE)


def test_projection_schema_and_bytes_are_canonical() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    payload = _payload()
    Draft202012Validator(schema).validate(payload)
    assert FIXTURE.read_bytes() == canonical_json_bytes(payload)


def test_independent_replay_recovers_all_formulas_and_zero_sets() -> None:
    report = verify_projection_file(FIXTURE)

    assert report["status"] == "PASS"
    assert report["scientific_verdict_replayed"] == (
        "BOUNDED_COMPLETION_CLASS_NONIDENTIFIABLE"
    )
    assert report["target_clean"] is True
    assert report["complete_declared_branch_grammar"] is True
    assert report["distinct_bounded_zero_sets"] is True
    assert report["scope"] == {
        "finite_sample_replay": True,
        "all_positive_integer_rungs_proved": False,
        "producer_implementation_independent": True,
        "physical_n_closure_promoted": False,
        "full_a1_a3_packet_lift_replayed": False,
    }

    branches = {row["branch_id"]: row for row in report["branch_reports"]}
    assert set(branches) == {
        "reversible_identity",
        "copy_collapse_erasure",
        "capped_two_class",
        "hidden_spectator",
    }
    assert [
        row["k"] for row in branches["reversible_identity"]["bounded_zero_set"]
    ] == [1, 2, 3, 4]
    assert [
        row["k"] for row in branches["copy_collapse_erasure"]["bounded_zero_set"]
    ] == [1]
    assert [
        row["k"] for row in branches["capped_two_class"]["bounded_zero_set"]
    ] == [1, 2]
    assert branches["hidden_spectator"]["bounded_zero_set"] == [
        {"k": k, "spectator_multiplicity": 1} for k in range(1, 5)
    ]


def test_graph_replay_supplies_matching_lower_and_upper_capacity_witnesses() -> None:
    report = verify_projection_file(FIXTURE)
    for branch in report["branch_reports"]:
        for row in branch["sample_rows"]:
            assert row["clique_component_count"] == row["claimed_capacity_M0"]
            assert row["independent_witness_size"] == row["claimed_capacity_M0"]
            assert sum(row["component_size_multiset"]) == row["graph_vertex_count"]
            assert row["graph_edge_count"] == sum(
                size * (size - 1) // 2 for size in row["component_size_multiset"]
            )


def test_hidden_spectator_is_counted_in_raw_dimension_and_quotiented_from_capacity() -> None:
    report = verify_projection_file(FIXTURE)
    spectator = next(
        branch
        for branch in report["branch_reports"]
        if branch["branch_id"] == "hidden_spectator"
    )
    row = next(
        row
        for row in spectator["sample_rows"]
        if row["k"] == 4 and row["spectator_multiplicity"] == 3
    )
    assert row["raw_dimension"] == 288
    assert row["public_dimension"] == 96
    assert row["claimed_capacity_M0"] == 96
    assert row["component_size_multiset"] == [3] * 96
    assert row["claimed_slack_zero"] is False


def test_capacity_and_bounded_zero_set_mutations_fail_closed() -> None:
    payload = _payload()
    payload["branches"][0]["sample_rows"][0]["claimed_capacity_M0"] += 1
    assert verify_projection(payload)["status"] == "FAIL"

    payload = _payload()
    payload["branches"][0]["claimed_bounded_zero_set"] = []
    assert verify_projection(payload)["status"] == "FAIL"


def test_source_signature_and_pin_mutations_fail_closed() -> None:
    payload = _payload()
    payload["branches"][0]["shared_source_signature_sha256"] = "sha256:" + "0" * 64
    assert verify_projection(payload)["status"] == "FAIL"

    payload = _payload()
    payload["branches"][0]["upstream_pins"][
        "fixed_packet_projection_sha256"
    ] = "sha256:" + "0" * 64
    assert verify_projection(payload)["status"] == "FAIL"

    payload = _payload()
    payload["shared_source"]["sample_rungs"].append(5)
    assert verify_projection(payload)["status"] == "FAIL"

    payload = _payload()
    payload["shared_source"]["full_a1_a3_packet_lift_required"] = False
    assert verify_projection(payload)["status"] == "FAIL"


def test_incomplete_branch_grammar_and_channel_drift_fail_closed() -> None:
    payload = _payload()
    payload["branches"].pop()
    assert verify_projection(payload)["status"] == "FAIL"

    payload = _payload()
    payload["branches"][0]["channel_rule"] = "output=port"
    assert verify_projection(payload)["status"] == "FAIL"


def test_target_taint_is_rejected_by_the_strict_file_contract(tmp_path: Path) -> None:
    payload = _payload()
    payload["target_cleanliness"]["measured_cosmological_constant_read"] = True
    candidate = tmp_path / "tainted.json"
    candidate.write_bytes(canonical_json_bytes(payload))

    report = verify_projection_file(candidate)

    assert report["status"] == "FAIL"
    assert "schema violation" in report["error"]


def test_noncanonical_and_duplicate_json_are_rejected(tmp_path: Path) -> None:
    payload = _payload()
    noncanonical = tmp_path / "pretty.json"
    noncanonical.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    assert verify_projection_file(noncanonical)["status"] == "FAIL"

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}\n', encoding="ascii")
    assert verify_projection_file(duplicate)["status"] == "FAIL"


def test_verifier_has_no_producer_import_or_target_data_dependency() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert "reverse_engineering_reality" not in source
    assert "correctable_public_record_capacity" not in source
    assert "observed_cosmological_constant" not in source
    assert "measured_horizon_radius" not in source


def test_compact_receipt_binds_exact_independent_replay() -> None:
    report = verify_projection_file(FIXTURE)
    receipt = compact_replay_receipt(report, projection_path=FIXTURE)

    assert receipt["status"] == "PASS"
    assert receipt["issue"] == 551
    assert receipt["scientific_verdict_replayed"] == (
        "BOUNDED_COMPLETION_CLASS_NONIDENTIFIABLE"
    )
    assert receipt["scope"]["producer_implementation_independent"] is True
    assert receipt["scope"]["all_positive_integer_rungs_proved"] is False
    assert receipt["target_clean"] is True
    assert receipt["branch_ids_replayed"] == [
        "capped_two_class",
        "copy_collapse_erasure",
        "hidden_spectator",
        "reversible_identity",
    ]
    assert RECEIPT.read_bytes() == canonical_json_bytes(receipt)
