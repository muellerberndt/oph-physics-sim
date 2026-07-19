from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from oph_fpe.cosmology.public_record_capacity import (
    MAX_TERMINAL_FIBER_PACKETS,
    build_reference_packet,
    approximate_public_capacity,
    certify_reversible_packet,
    certify_unique_slack_zero,
    evaluate_terminal,
    evaluate_terminal_fiber,
    greatest_fixed_point,
    no_new_confusability,
    section_id,
    write_public_record_capacity_report,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas/cosmology/public_record_capacity_receipt.schema.json"
)
BUNDLE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas/cosmology/public_record_capacity_bundle.schema.json"
)


def test_reversible_twelve_port_packet_recomputes_capacity() -> None:
    packet = build_reference_packet(4)
    certificate = certify_reversible_packet(packet)
    assert certificate["status"] == "PASS"
    assert certificate["port_count"] == 12
    assert certificate["interface_count"] == 30
    assert certificate["exact_zero_error_capacity_M0"] == 4
    assert math.isclose(certificate["readback_nats_log_M0"], math.log(4.0))
    assert certificate["REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT"] is True
    assert certificate["PHYSICAL_N_CLOSURE_RECEIPT"] is False

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(certificate)


def test_target_taint_and_circular_self_read_fail_closed() -> None:
    packet = build_reference_packet()
    packet["lambda_used"] = True
    assert evaluate_terminal(packet)["status"] == "TARGET_TAINTED"
    packet = build_reference_packet()
    packet["self_read_predicate_injected"] = True
    assert evaluate_terminal(packet)["status"] == "CIRCULAR_CAPACITY_DEFINITION"
    packet = build_reference_packet()
    packet["supplied_capacity_metadata_read_by_producer"] = True
    assert evaluate_terminal(packet)["status"] == "CIRCULAR_CAPACITY_DEFINITION"


def test_local_marginals_cannot_replace_global_joint_kernel() -> None:
    packet = build_reference_packet(2)
    reachable = sorted(packet["projection_supports"])
    packet["global_checkpoint_kernels"] = [
        {
            "authorized_observers": list(packet["observers"]),
            "continuation_id": "erasing_joint_channel",
            "rows": {source: {reachable[0]: 1.0} for source in reachable},
        }
    ]
    packet["expected_continuation_ids"] = ["erasing_joint_channel"]
    packet["receiver_known_continuation_ids"] = ["erasing_joint_channel"]
    result = evaluate_terminal(packet)
    assert result["status"] == "PASS"
    assert result["exact_zero_error_capacity_M0"] == 1
    assert result["saturation_passed"] is False


def test_incomplete_and_ambiguous_terminal_fibers_fail_closed() -> None:
    packet = build_reference_packet(2)
    terminal_id = packet["terminal_id"]
    assert (
        evaluate_terminal_fiber(
            [packet],
            expected_terminal_ids=[terminal_id, terminal_id],
        )["status"]
        == "INCOMPLETE_TERMINAL_FIBER"
    )
    erased = copy.deepcopy(packet)
    packet["terminal_id"] = "terminal-preserving"
    erased["terminal_id"] = "terminal-erasing"
    reachable = sorted(erased["projection_supports"])
    erased["global_checkpoint_kernels"] = [
        {
            "authorized_observers": list(erased["observers"]),
            "continuation_id": "erase",
            "rows": {source: {reachable[0]: 1.0} for source in reachable},
        }
    ]
    erased["expected_continuation_ids"] = ["erase"]
    erased["receiver_known_continuation_ids"] = ["erase"]
    result = evaluate_terminal_fiber(
        [packet, erased],
        expected_terminal_ids=["terminal-preserving", "terminal-erasing"],
    )
    assert result["status"] == "AMBIGUOUS_CAPACITY_READBACK"
    assert result["terminal_fiber_capacity_set"] == [1, 2]


def test_order_theory_does_not_claim_uniqueness() -> None:
    identity = greatest_fixed_point({1: 1, 2: 2, 3: 3})
    assert identity["greatest_fixed_point"] == 3
    assert identity["uniqueness_proved"] is False
    erasure = greatest_fixed_point({1: 1, 2: 1, 3: 1})
    assert erasure["fixed_points"] == [1]
    assert erasure["uniqueness_proved"] is True


def test_unique_slack_and_new_confusability_controls() -> None:
    assert certify_unique_slack_zero({1: 1, 2: 1, 3: 2}, 1)["status"] == "PASS"
    assert certify_unique_slack_zero({1: 1, 2: 2}, 2)["status"] == "FINITE_SIZE_SELECTOR_FAILED"
    coarse = {"a": set(), "b": set()}
    fine = {"A": {"B"}, "B": {"A"}}
    assert no_new_confusability(coarse, fine, {"a": "A", "b": "B"}) is False


def test_writer_labels_reference_as_nonphysical(tmp_path) -> None:
    report = write_public_record_capacity_report(tmp_path, capacity_dimension=3)
    assert report["reference_control"] is True
    assert report["evaluation"]["exact_zero_error_capacity_M0"] == 3
    assert report["physical_N_closure_receipt"] is False
    assert (tmp_path / "public_checkpoint_packet.json").exists()
    assert (tmp_path / "public_record_capacity_report.json").exists()
    certificate_path = tmp_path / "public_record_capacity_reference_certificate.json"
    assert certificate_path.exists()
    bundle_schema = json.loads(BUNDLE_SCHEMA_PATH.read_text(encoding="utf-8"))
    receipt_schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(bundle_schema).validate(report)
    Draft202012Validator(receipt_schema).validate(
        json.loads(certificate_path.read_text(encoding="utf-8"))
    )

    truncated = copy.deepcopy(report)
    truncated["packet"] = {}
    truncated["evaluation"] = {"status": "PASS"}
    truncated["reversible_certificate"] = {
        "status": "PASS",
        "REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT": True,
        "PHYSICAL_N_CLOSURE_RECEIPT": False,
    }
    assert list(Draft202012Validator(bundle_schema).iter_errors(truncated))

    substituted = copy.deepcopy(report)
    substituted["packet"].update(
        {
            "capacity_dimension": 2,
            "observers": {"x": ["record_0", "record_1"]},
            "interfaces": [{}],
            "reachability_witnesses": {"fake": ["event"]},
            "publicness_policy": [["x"]],
            "global_checkpoint_kernels": [{}, {}, {}],
            "projection_supports": {"fake": [0]},
        }
    )
    assert list(Draft202012Validator(bundle_schema).iter_errors(substituted))


def test_writer_rejects_nonfinite_packet_json(tmp_path) -> None:
    packet = build_reference_packet(2)
    packet["unexpected"] = float("nan")

    with pytest.raises(ValueError, match="strict finite JSON"):
        write_public_record_capacity_report(tmp_path, packet=packet)


@pytest.mark.parametrize("probability", [10**400, 1.0 + 5.0e-13])
def test_checkpoint_kernel_requires_an_exact_bounded_simplex(probability: int | float) -> None:
    packet = build_reference_packet(2)
    row = next(iter(packet["global_checkpoint_kernels"][0]["rows"].values()))
    output = next(iter(row))
    row[output] = probability

    report = evaluate_terminal(packet)

    assert report["status"] == "LOCAL_MARGINAL_MISMATCH"


def test_canonical_section_ids_are_injective_under_delimiter_adversary() -> None:
    historical_collision_left = {"a": "b|c=d"}
    historical_collision_right = {"a": "b", "c": "d"}

    assert section_id(historical_collision_left) != section_id(
        historical_collision_right
    )
    assert section_id({"c": "d", "a": "b"}) == section_id(
        historical_collision_right
    )


def test_continuation_manifest_and_receiver_knowledge_are_explicit() -> None:
    packet = build_reference_packet(2)
    packet["expected_continuation_ids"] = ["identity", "identity"]
    assert evaluate_terminal(packet)["status"] == "INCOMPLETE_CONTINUATION_MANIFEST"

    packet = build_reference_packet(2)
    packet["global_checkpoint_kernels"].pop()
    assert evaluate_terminal(packet)["status"] == "INCOMPLETE_CONTINUATION_MANIFEST"

    packet = build_reference_packet(2)
    packet["receiver_known_continuation_ids"] = ["identity"]
    assert (
        evaluate_terminal(packet)["status"]
        == "RECEIVER_CONTINUATION_KNOWLEDGE_INCOMPLETE"
    )


def test_local_marginals_are_recomputed_not_trusted_from_boolean() -> None:
    packet = build_reference_packet(2)
    packet["local_marginal_consistency_passed"] = False
    valid = evaluate_terminal(packet)
    assert valid["status"] == "PASS"
    assert valid["local_marginal_consistency_recomputed"] is True
    assert valid["legacy_local_marginal_declaration_promoted"] is False

    packet = build_reference_packet(2)
    first_rows = packet["global_checkpoint_kernels"][0]["rows"]
    for source in first_rows:
        first_rows[source] = {"fabricated-global-output": 1.0}
    forged = evaluate_terminal(packet)
    assert forged["status"] == "LOCAL_MARGINAL_MISMATCH"


def test_terminal_manifest_rejects_missing_unexpected_and_duplicate_ids() -> None:
    packet = build_reference_packet(2)
    assert (
        evaluate_terminal_fiber(
            [packet],
            expected_terminal_ids=[packet["terminal_id"], "missing-terminal"],
        )["status"]
        == "INCOMPLETE_TERMINAL_FIBER"
    )
    duplicate = copy.deepcopy(packet)
    assert (
        evaluate_terminal_fiber(
            [packet, duplicate],
            expected_terminal_ids=[packet["terminal_id"]],
        )["status"]
        == "INCOMPLETE_TERMINAL_FIBER"
    )


def test_malformed_reversible_packets_fail_safely() -> None:
    malformed = certify_reversible_packet({"observers": []})
    assert malformed["status"] == "INVALID_REFERENCE_PACKET"
    assert malformed["REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT"] is False

    packet = build_reference_packet(2)
    packet["global_checkpoint_kernels"] = "not-a-kernel-list"
    result = certify_reversible_packet(packet)
    assert result["status"] == "INVALID_REFERENCE_PACKET"
    assert result["REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT"] is False

    packet = build_reference_packet(2)
    first_row = next(iter(packet["global_checkpoint_kernels"][0]["rows"].values()))
    first_row[next(iter(first_row))] = True
    result = certify_reversible_packet(packet)
    assert result["status"] == "INVALID_REFERENCE_PACKET"
    assert result["REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT"] is False


def test_reference_receipt_rejects_noncanonical_topology_or_metadata() -> None:
    packet = build_reference_packet(2)
    atoms = packet["observers"]["north"]
    identity = {atom: atom for atom in atoms}
    packet["interfaces"] = [
        {
            "interface_id": f"north--{observer}",
            "left_observer": "north",
            "right_observer": observer,
            "left_readout": identity,
            "right_readout": identity,
        }
        for observer in packet["observers"]
        if observer != "north"
    ]

    certificate = certify_reversible_packet(packet)

    assert certificate["status"] == "NONREFERENCE_REVERSIBLE_PACKET"
    assert certificate["reference_topology_validated"] is False
    assert certificate["REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT"] is False


def test_reference_receipt_rejects_wrong_named_generator_action() -> None:
    packet = build_reference_packet(3)
    identity_rows = copy.deepcopy(packet["global_checkpoint_kernels"][0]["rows"])
    packet["global_checkpoint_kernels"][1]["rows"] = identity_rows

    certificate = certify_reversible_packet(packet)

    assert certificate["status"] == "NONCANONICAL_REFERENCE_GENERATOR"
    assert certificate["REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT"] is False


def test_strict_packet_bounds_and_types_fail_closed() -> None:
    packet = build_reference_packet(2)
    packet["observers"]["north"][0] = True
    assert evaluate_terminal(packet)["status"] == "NO_RECORD_ATOM_RESTRICTION"

    packet = build_reference_packet(2)
    packet["unexpected"] = "field"
    assert evaluate_terminal(packet)["status"] == "INVALID_PACKET"

    packet = build_reference_packet(2)
    nested: dict[str, object] = {}
    for _ in range(2000):
        nested = {"nested": nested}
    packet["unexpected"] = nested
    assert evaluate_terminal(packet)["status"] == "INVALID_PACKET"


def test_approximate_decoder_rejects_exponential_output_search() -> None:
    reachable = [f"record-{index}" for index in range(12)]
    channel = {
        "rows": {
            source: {
                f"output-{index}-a": 0.5,
                f"output-{index}-b": 0.5,
            }
            for index, source in enumerate(reachable)
        }
    }

    with pytest.raises(ValueError, match="output alphabet"):
        approximate_public_capacity(reachable, [channel], epsilon=0.1)


def test_approximate_capacity_rejects_caller_expanded_vertex_budget() -> None:
    with pytest.raises(ValueError, match="bounded evaluator range"):
        approximate_public_capacity([], [], epsilon=0.1, max_vertices=13)


def test_approximate_capacity_requires_a_bounded_nonempty_channel_family() -> None:
    with pytest.raises(ValueError, match="nonempty bounded"):
        approximate_public_capacity(["record"], [], epsilon=0.1)


def test_approximate_capacity_requires_a_unique_identifier_alphabet() -> None:
    duplicate_channel = {"rows": {"record": {"record": 1.0}}}

    with pytest.raises(ValueError, match="unique record identifiers"):
        approximate_public_capacity(
            ["record", "record"],
            [duplicate_channel],
            epsilon=0.0,
        )
    with pytest.raises(ValueError, match="bounded sequence"):
        approximate_public_capacity("record", [duplicate_channel], epsilon=0.0)
    with pytest.raises(ValueError, match="at least one"):
        approximate_public_capacity([], [duplicate_channel], epsilon=0.0)


def test_terminal_fiber_rejects_oversized_packet_manifest_before_iteration() -> None:
    report = evaluate_terminal_fiber(
        [{}] * 4097,
        expected_terminal_ids=["terminal"],
    )

    assert report["status"] == "INCOMPLETE_TERMINAL_FIBER"
    assert "bounded evaluator" in report["reason"]

    report = evaluate_terminal_fiber(
        [{}] * (MAX_TERMINAL_FIBER_PACKETS + 1),
        expected_terminal_ids=["terminal"],
    )
    assert report["status"] == "INCOMPLETE_TERMINAL_FIBER"
