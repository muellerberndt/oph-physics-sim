"""Gates for the source-derived causal-order producer and verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oph_fpe.bulk.source_derived_causal_order import (
    DEFAULT_CONFIG,
    SourceDerivedCausalOrderError,
    _canonical_bytes,
    _sha256,
    declared_ancestry_projection,
    generated_provenance_edges,
    produce_source_derived_causal_order_report,
)
from oph_fpe.bulk.physical_h3_kms_source_capture import (
    _semantic_source_events,
)
from oph_fpe.bulk.verify_source_derived_causal_order_independent import (
    DEFAULT_RECEIPT,
    IndependentVerificationError,
    verify_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict:
    return produce_source_derived_causal_order_report()


def test_report_is_deterministic(report: dict) -> None:
    replay = produce_source_derived_causal_order_report(dict(DEFAULT_CONFIG))
    assert _canonical_bytes(replay) == _canonical_bytes(report)


def test_generation_is_parent_blind_and_sound(report: dict) -> None:
    assert report["SOURCE_DERIVED_CAUSAL_ORDER_RECEIPT"] is True
    assert report["generated_acyclic"] is True
    assert report["sequence_compatible"] is True
    assert report["controls_fail_closed"] is True
    assert report["physical_promotion_allowed"] is False
    assert report["negative_controls"][
        "declared_parent_mutation_leaves_semantic_order_unchanged"
    ] is True
    assert report["negative_controls"][
        "record_order_mutation_leaves_semantic_order_unchanged"
    ] is True
    assert report["negative_controls"][
        "record_full_state_hash_mutation_is_refused"
    ] is True
    assert report["negative_controls"][
        "phase_observer_permutation_leaves_semantic_order_unchanged"
    ] is True
    assert report["capture_ancestry_matches_generated"] is True
    assert report["cross_observer_edge_count"] > 0
    assert report["event_carrier_scope"] == (
        "observer_instrumentation_history_over_source_state_snapshots"
    )
    assert report["underlying_repair_transactions_promoted_as_events"] is False


def test_byte_identity_clause_is_internally_consistent(report: dict) -> None:
    clause = report["byte_identity_clause"]
    assert clause["scope"] == (
        "canonical_projected_provenance_edge_rows_on_bounded_"
        "source_observer_instrumentation_log"
    )
    assert clause["event_count"] == 24
    both_empty = (
        clause["declared_only_pair_count"] == 0
        and clause["generated_only_pair_count"] == 0
    )
    if clause["byte_identical"]:
        assert both_empty
        assert clause["verdict"] == "ATTAINED"
    else:
        assert clause["verdict"] == "NOT_ATTAINED"
        assert not both_empty or (
            report["generated_edges"] != report["declared_edges"]
        )


def test_repair_only_event_carrier_control_is_an_antichain(report: dict) -> None:
    control = report["repair_only_event_carrier_control"]
    assert control["repair_event_count"] == 24
    assert control["versioned_provenance_edges"] == []
    assert control["versioned_provenance_edge_count"] == 0
    assert control["all_reads_are_version_zero_roots"] is True
    assert control["classification"] == (
        "REPAIR_ONLY_EVENT_CARRIER_IS_ANTICHAIN"
    )
    assert control["physical_causet_promotion_allowed"] is False


def test_declared_ancestry_equals_generated_provenance(report: dict) -> None:
    clause = report["byte_identity_clause"]
    assert clause["byte_identical"] is True
    assert clause["verdict"] == "ATTAINED"
    assert clause["declared_only_pair_count"] == 0
    assert clause["generated_only_pair_count"] == 0
    assert report["generated_edges"] == report["declared_edges"]


def test_record_reads_are_explicit_source_or_applied_feedback_reads(
    report: dict,
) -> None:
    for event in report["semantic_events"]:
        if event["canonical_semantic_payload"]["event_kind"] == "RECORD_COMMIT":
            payload = event["canonical_semantic_payload"]
            feedback_id = payload["applied_feedback_event_id"]
            expected_feedback_resource = (
                None if feedback_id is None else f"local-action:{feedback_id}"
            )
            for resource in event["read_resource_ids"]:
                assert resource.startswith("source-state:") or (
                    resource == expected_feedback_resource
                )
            if feedback_id is None:
                assert event["parent_event_ids"] == []
            else:
                assert feedback_id in event["parent_event_ids"]
                assert expected_feedback_resource in event["read_resource_ids"]


def test_cross_reads_bind_other_committed_record_contents(report: dict) -> None:
    cross_reads = [
        event
        for event in report["semantic_events"]
        if event["canonical_semantic_payload"].get(
            "cross_read_record_versions"
        )
    ]
    assert cross_reads
    for event in cross_reads:
        payload = event["canonical_semantic_payload"]
        for version in payload["cross_read_record_versions"]:
            assert version["record_event_id"].startswith("sha256:")
            assert version["committed_full_port_state_sha256"].startswith(
                "sha256:"
            )
            assert version["visibility_witness_kind"] == (
                "shared_declared_support_carrier"
            )
            assert version["record_carrier_id"]
            assert version["record_observer_token"] != event["observer_token"]


def test_record_full_state_commitment_mutation_is_refused(report: dict) -> None:
    observer = json.loads(
        _canonical_bytes(report["observer_log_material"]).decode("ascii")
    )
    record = next(
        row for row in observer["events"] if row["kind"] == "RECORD_COMMIT"
    )
    record["full_port_state"][0] += 1.0
    with pytest.raises(RuntimeError, match="full-state commitment"):
        _semantic_source_events(observer, validate_transport_ids=False)


def test_cross_read_overlap_witness_mutation_is_refused(report: dict) -> None:
    observer = json.loads(
        _canonical_bytes(report["observer_log_material"]).decode("ascii")
    )
    readback = next(
        row
        for row in observer["events"]
        if row.get("cross_read_overlap_witnesses")
    )
    readback["cross_read_overlap_witnesses"][0]["record_carrier_id"] = (
        "carrier-not-in-support"
    )
    with pytest.raises(RuntimeError, match="overlap witness"):
        _semantic_source_events(observer, validate_transport_ids=False)


def test_cross_read_version_hash_mismatch_is_refused(report: dict) -> None:
    observer = json.loads(
        _canonical_bytes(report["observer_log_material"]).decode("ascii")
    )
    readback = next(
        row
        for row in observer["events"]
        if row.get("cross_read_record_state_sha256s")
    )
    readback["cross_read_record_state_sha256s"][0] = "sha256:" + "0" * 64
    with pytest.raises(RuntimeError, match="cross-read state hash"):
        _semantic_source_events(observer, validate_transport_ids=False)


def test_two_writer_material_is_refused() -> None:
    view = [
        {
            "event_key": "a",
            "read_resource_ids": [],
            "write_resource_ids": ["r"],
        },
        {
            "event_key": "b",
            "read_resource_ids": [],
            "write_resource_ids": ["r"],
        },
    ]
    with pytest.raises(SourceDerivedCausalOrderError):
        generated_provenance_edges(view)


def test_missing_nonroot_writer_is_refused() -> None:
    view = [
        {
            "event_key": "child",
            "read_resource_ids": ["record:absent"],
            "write_resource_ids": ["readback:child"],
        }
    ]
    with pytest.raises(SourceDerivedCausalOrderError):
        generated_provenance_edges(view)
    assert generated_provenance_edges(
        view, distinguished_source_resource_ids=["record:absent"]
    ) == []


def test_semantic_ids_ignore_declared_parent_and_record_order_metadata(
    report: dict,
) -> None:
    observer = json.loads(
        _canonical_bytes(report["observer_log_material"]).decode("ascii")
    )
    baseline, generated, _, _ = _semantic_source_events(observer)
    parent_mutation = json.loads(
        _canonical_bytes(observer).decode("ascii")
    )
    next(row for row in parent_mutation["events"] if row["parents"])[
        "parents"
    ] = []
    mutated, mutated_generated, _, _ = _semantic_source_events(
        parent_mutation, validate_transport_ids=False
    )
    assert mutated == baseline
    assert mutated_generated == generated

    order_mutation = json.loads(_canonical_bytes(observer).decode("ascii"))
    next(
        row
        for row in order_mutation["events"]
        if row["kind"] == "RECORD_COMMIT"
    )["record_order_previous_event_ids"] = ["sha256:" + "f" * 64]
    mutated, mutated_generated, _, _ = _semantic_source_events(
        order_mutation, validate_transport_ids=False
    )
    assert mutated == baseline
    assert mutated_generated == generated


def test_declared_projection_sorts_canonically() -> None:
    rows = [
        {
            "parent_event_id": "b",
            "child_event_id": "c",
            "shared_resource_ids": ["y", "x"],
        },
        {
            "parent_event_id": "a",
            "child_event_id": "c",
            "shared_resource_ids": [],
        },
    ]
    projected = declared_ancestry_projection(rows)
    assert projected[0]["parent_event_id"] == "a"
    assert projected[1]["shared_resource_ids"] == ["x", "y"]


def test_frozen_receipt_replays_byte_for_byte(report: dict) -> None:
    frozen = DEFAULT_RECEIPT.read_bytes()
    assert frozen == _canonical_bytes(report)


def test_independent_verifier_accepts_frozen_receipt() -> None:
    result = verify_receipt(DEFAULT_RECEIPT)
    assert result["receipt"] is True
    assert result["byte_identical"] is True


def _restamp(receipt: dict) -> dict:
    body = {k: v for k, v in receipt.items() if k != "report_sha256"}
    receipt["report_sha256"] = _sha256(body)
    return receipt


def test_independent_verifier_fails_closed(tmp_path: Path, report: dict) -> None:
    stale = json.loads(_canonical_bytes(report).decode("ascii"))
    stale["generated_edge_count"] += 1
    target0 = tmp_path / "stale_hash_receipt.json"
    target0.write_bytes(_canonical_bytes(stale))
    with pytest.raises(IndependentVerificationError):
        verify_receipt(target0)

    mutated = _restamp(json.loads(_canonical_bytes(report).decode("ascii")))
    mutated["generated_edges"] = mutated["generated_edges"][:-1]
    mutated = _restamp(mutated)
    target = tmp_path / "mutated_receipt.json"
    target.write_bytes(_canonical_bytes(mutated))
    with pytest.raises(IndependentVerificationError):
        verify_receipt(target)

    flipped = json.loads(_canonical_bytes(report).decode("ascii"))
    flipped["byte_identity_clause"]["byte_identical"] = False
    flipped["byte_identity_clause"]["verdict"] = "NOT_ATTAINED"
    flipped = _restamp(flipped)
    target2 = tmp_path / "flipped_receipt.json"
    target2.write_bytes(_canonical_bytes(flipped))
    with pytest.raises(IndependentVerificationError):
        verify_receipt(target2)


@pytest.mark.parametrize(
    "mutation",
    [
        "rank",
        "negative_control",
        "observer_log_hash",
        "checkpoint",
        "raw_parent",
        "semantic_key",
        "repair_control",
        "phase_permutation",
    ],
)
def test_independent_verifier_recomputes_every_material_clause(
    tmp_path: Path, report: dict, mutation: str
) -> None:
    candidate = json.loads(_canonical_bytes(report).decode("ascii"))
    if mutation == "rank":
        candidate["generated_longest_path_rank_max"] += 1
    elif mutation == "negative_control":
        candidate["negative_controls"][
            "missing_nonroot_writer_is_refused"
        ] = False
        candidate["controls_fail_closed"] = False
    elif mutation == "observer_log_hash":
        candidate["observer_event_log_sha256"] = "sha256:" + "0" * 64
    elif mutation == "checkpoint":
        candidate["observer_log_material"]["checkpoint"][
            "prefix_root"
        ] = "sha256:" + "0" * 64
        candidate["observer_log_material_sha256"] = _sha256(
            candidate["observer_log_material"]
        )
    elif mutation == "raw_parent":
        next(
            row
            for row in candidate["observer_log_material"]["events"]
            if row["parents"]
        )["parents"] = []
        candidate["observer_log_material_sha256"] = _sha256(
            candidate["observer_log_material"]
        )
    elif mutation == "semantic_key":
        candidate["semantic_events"][0]["event_key"] = "sha256:" + "1" * 64
    elif mutation == "repair_control":
        candidate["repair_only_event_carrier_control"][
            "versioned_provenance_edge_count"
        ] = 1
    else:
        candidate["observer_log_material"][
            "phase_observer_permutation_control"
        ]["source_order_invariant"] = False
        candidate["observer_log_material_sha256"] = _sha256(
            candidate["observer_log_material"]
        )
    candidate = _restamp(candidate)
    target = tmp_path / f"{mutation}.json"
    target.write_bytes(_canonical_bytes(candidate))
    with pytest.raises(IndependentVerificationError):
        verify_receipt(target)


def test_receipt_flag_requires_byte_identity(tmp_path: Path, report: dict) -> None:
    candidate = json.loads(_canonical_bytes(report).decode("ascii"))
    candidate["declared_edges"] = candidate["declared_edges"][:-1]
    candidate["declared_edge_count"] -= 1
    candidate["declared_edges_sha256"] = _sha256(candidate["declared_edges"])
    generated_pairs = {
        (row["parent_event_id"], row["child_event_id"])
        for row in candidate["generated_edges"]
    }
    declared_pairs = {
        (row["parent_event_id"], row["child_event_id"])
        for row in candidate["declared_edges"]
    }
    missing = sorted(generated_pairs - declared_pairs)
    candidate["byte_identity_clause"].update(
        {
            "byte_identical": False,
            "verdict": "NOT_ATTAINED",
            "generated_only_pair_count": len(missing),
            "generated_only_pairs": [list(pair) for pair in missing],
        }
    )
    candidate["status"] = (
        "SOURCE_DERIVED_CAUSAL_ORDER_BYTE_IDENTITY_NOT_ATTAINED"
        "__PHYSICAL_ATTACHMENT_OPEN"
    )
    # A report that keeps the positive receipt flag despite failed identity
    # must be refused even when its outer hash is recomputed.
    candidate["SOURCE_DERIVED_CAUSAL_ORDER_RECEIPT"] = True
    candidate = _restamp(candidate)
    target = tmp_path / "false_green_identity.json"
    target.write_bytes(_canonical_bytes(candidate))
    with pytest.raises(IndependentVerificationError):
        verify_receipt(target)
