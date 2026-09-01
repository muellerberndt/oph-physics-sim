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
        "declared_parents_stripped_before_generation"
    ] is True


def test_byte_identity_clause_is_internally_consistent(report: dict) -> None:
    clause = report["byte_identity_clause"]
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


def test_declared_ancestry_is_strict_subset_on_default_log(report: dict) -> None:
    clause = report["byte_identity_clause"]
    assert clause["byte_identical"] is False
    assert clause["declared_only_pair_count"] == 0
    assert clause["generated_only_pair_count"] == 8
    kinds = {
        event["event_key"]: event["canonical_semantic_payload"]["event_kind"]
        for event in report["semantic_events"]
    }
    observers = {
        event["event_key"]: event["observer_token"]
        for event in report["semantic_events"]
    }
    for parent, child in clause["generated_only_pairs"]:
        assert kinds[parent] == "RECORD_COMMIT"
        assert kinds[child] == "LOCAL_FEEDBACK"
        assert observers[parent] == observers[child]


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
    assert result["byte_identical"] is False


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
    flipped["byte_identity_clause"]["byte_identical"] = True
    flipped["byte_identity_clause"]["verdict"] = "ATTAINED"
    flipped = _restamp(flipped)
    target2 = tmp_path / "flipped_receipt.json"
    target2.write_bytes(_canonical_bytes(flipped))
    with pytest.raises(IndependentVerificationError):
        verify_receipt(target2)
