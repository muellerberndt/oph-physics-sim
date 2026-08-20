"""Regression tests for the 2026-08-20 ol_a1_replication catalog review.

Commit 42aa966 added seventeen tracked JSON files under
``data/ol_a1_replication/`` without extending the closed canonical catalog,
which stopped the ancestry-inventory producer at import time.  The repair
admitted the paths through the subsystem's own review procedure: matching
contract rows in the producer and in the independent verifier, then a
producer regeneration of the pinned inventory and of the receipts that pin
it.  These tests hold that review in place: the rows exist with the exact
recorded contracts, the pinned payload replays, and a mutated payload fails.
"""

from __future__ import annotations

import copy
import json

import pytest

from oph_fpe.dynamics import source_operator_inventory as inventory
from oph_fpe.dynamics import (
    verify_source_operator_inventory_independent as independent,
)


REPORT = inventory.build_inventory()

OL_A1_RUN_PATHS = tuple(
    f"data/ol_a1_replication/run_{cell}_ola1.r{replicate}.json"
    for cell in ("A1", "A2", "C1")
    for replicate in range(1, 6)
)
OL_A1_PATHS = (
    "data/ol_a1_replication/campaign_summary.json",
    "data/ol_a1_replication/manifest.json",
    *OL_A1_RUN_PATHS,
)


def _row(report: dict, path: str) -> dict:
    return next(
        row for row in report["canonical_artifact_rows"] if row["path"] == path
    )


def test_ol_a1_replication_paths_are_reviewed_into_both_catalogs() -> None:
    assert len(OL_A1_PATHS) == 17
    for path in OL_A1_PATHS:
        assert path in inventory.CANONICAL_CONTRACTS
        assert path in independent.EXPECTED_CONTRACTS
        assert (
            inventory.CANONICAL_CONTRACTS[path]
            == independent.EXPECTED_CONTRACTS[path]
        )


def test_ol_a1_rows_carry_the_recorded_contracts() -> None:
    summary = _row(REPORT, "data/ol_a1_replication/campaign_summary.json")
    assert summary["schema"] == "oph.ol-a1-signature-replication.summary.v1"
    assert summary["status"] is None
    assert summary["disposition"] == (
        "OL_A1_TIER_A_REPLICATION_VERDICT_RECORD_NOT_VERTEX12_OPERATOR"
    )
    assert summary["semantic_scan_excluded_as_recursive_output"] is False
    assert summary["raw_pin"]["path"] == summary["path"]

    manifest = _row(REPORT, "data/ol_a1_replication/manifest.json")
    assert manifest["schema"] == "oph.ol-a1-signature-replication.manifest.v1"
    assert manifest["disposition"] == (
        "OL_A1_REPLICATION_CAMPAIGN_MANIFEST_NOT_VERTEX12_OPERATOR"
    )

    for path in OL_A1_RUN_PATHS:
        row = _row(REPORT, path)
        assert row["schema"] == "oph.ol-a1-signature-replication.receipt.v1"
        assert row["status"] is None
        assert row["disposition"] == (
            "OL_A1_REPLICATION_CELL_RECEIPT_NOT_VERTEX12_OPERATOR"
        )


def test_ol_a1_records_carry_no_promotion_signals() -> None:
    scan = REPORT["current_canonical_json_contract_scan"]
    assert scan["registered_source_packet_rows_excluding_recursive_outputs"] == []
    assert scan["positive_promotion_signal_rows_excluding_recursive_outputs"] == []
    assert REPORT["status"] == inventory.STATUS


def test_repaired_inventory_replays() -> None:
    replay = inventory.verify_inventory(REPORT)
    assert replay["receipt"] is True, replay["reasons"]
    verification = independent.verify(REPORT)
    assert verification["receipt"] is True, verification["reasons"]


def test_committed_pinned_inventory_payload_replays() -> None:
    committed = json.loads(
        inventory.OUTPUT_PATH.read_text(encoding="utf-8")
    )
    replay = inventory.verify_inventory(committed)
    assert replay["receipt"] is True, replay["reasons"]


@pytest.mark.parametrize(
    "mutation",
    [
        ("disposition", "MUTATED_DISPOSITION"),
        ("schema", "oph.mutated-schema.v1"),
    ],
)
def test_mutated_ol_a1_row_fails_replay(mutation: tuple[str, str]) -> None:
    key, value = mutation
    mutated = copy.deepcopy(REPORT)
    row = _row(mutated, "data/ol_a1_replication/campaign_summary.json")
    row[key] = value
    replay = inventory.verify_inventory(mutated)
    assert replay["receipt"] is False
    verification = independent.verify(mutated)
    assert verification["receipt"] is False
