"""Tests for the issue-577 semantic countermodel matrix."""

from __future__ import annotations

import json

import pytest

from oph_fpe.bulk.einstein_branch_countermodels import (
    PHYSICAL_PROMOTION_ALLOWED,
    produce_countermodel_matrix,
)

_SMALL = {"carrier_count": 32, "cycles": 6, "seed": 20260751}


@pytest.fixture(scope="module")
def report() -> dict:
    return produce_countermodel_matrix(config=_SMALL)


def test_every_countermodel_is_isolated(report: dict) -> None:
    for family, row in report["countermodels"].items():
        assert row["isolated"] is True, family
        assert row["baseline"] != row["countermodel_value"], family
    assert report["all_countermodels_isolated"] is True
    assert report["COUNTERMODEL_ISOLATION_RECEIPT"] is True


def test_baseline_records_raw_snapshot_rank_deficiency(report: dict) -> None:
    # Measured finding about the present source: the raw record-snapshot
    # moment does not span port space, so state faithfulness inside the
    # instruments is carried by the declared regularizer. The state-support
    # countermodel flips this in the attaining direction by enriching the
    # snapshots to full rank.
    assert report["baseline_clause_vector"]["state_support"] is False
    row = report["countermodels"]["state_support"]
    assert row["countermodel"] == "full_rank_snapshot_enrichment"
    assert row["countermodel_value"] is True


def test_countermodels_are_semantic_not_syntactic(report: dict) -> None:
    kinds = {row["kind"] for row in report["countermodels"].values()}
    assert kinds <= {
        "semantic_source_modification",
        "semantic_declaration_modification",
    }
    assert report["physical_promotion_allowed"] is False
    assert PHYSICAL_PROMOTION_ALLOWED is False


def test_report_is_deterministic() -> None:
    first = produce_countermodel_matrix(config=_SMALL)
    second = produce_countermodel_matrix(config=_SMALL)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
