from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator

from oph_fpe.cosmology.public_record_capacity import (
    build_reference_packet,
    certify_reversible_packet,
    certify_unique_slack_zero,
    evaluate_terminal,
    evaluate_terminal_fiber,
    greatest_fixed_point,
    no_new_confusability,
    write_public_record_capacity_report,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas/cosmology/public_record_capacity_receipt.schema.json"
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
            "rows": {source: {"same_public_output": 1.0} for source in reachable},
        }
    ]
    result = evaluate_terminal(packet)
    assert result["status"] == "PASS"
    assert result["exact_zero_error_capacity_M0"] == 1
    assert result["saturation_passed"] is False


def test_incomplete_and_ambiguous_terminal_fibers_fail_closed() -> None:
    packet = build_reference_packet(2)
    assert evaluate_terminal_fiber([packet], manifest_complete=False)["status"] == "INCOMPLETE_TERMINAL_FIBER"
    erased = copy.deepcopy(packet)
    reachable = sorted(erased["projection_supports"])
    erased["global_checkpoint_kernels"] = [
        {
            "authorized_observers": list(erased["observers"]),
            "continuation_id": "erase",
            "rows": {source: {"same": 1.0} for source in reachable},
        }
    ]
    result = evaluate_terminal_fiber([packet, erased], manifest_complete=True)
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
