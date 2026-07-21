"""Fail-closed tests for the issue-576 stress/coupling instrument."""

from __future__ import annotations

import json

import numpy as np
import pytest

from oph_fpe.bulk.stress_coupling_producer import (
    PHYSICAL_PROMOTION_ALLOWED,
    UNIVERSALITY_ENVELOPE,
    produce_stress_coupling_report,
)

_SMALL = {"carrier_count": 32, "cycles": 6, "seed": 20260751}


@pytest.fixture(scope="module")
def report() -> dict:
    return produce_stress_coupling_report(config=_SMALL)


def test_instrument_is_valid_all_controls_fail_closed(report: dict) -> None:
    controls = report["negative_controls"]
    assert set(controls) == {
        "permuted_pairing",
        "mixed_source_entropy",
        "shuffled_seam_ledger",
        "degenerate_cap_family",
    }
    for name, row in controls.items():
        assert row["control_failure_detected"] is True, name
    assert report["controls_fail_closed"] is True


def test_same_source_reconstruction_clauses_attained(report: dict) -> None:
    verdicts = report["clause_verdicts"]
    assert verdicts["same_source_stress_reconstructed"] is True
    assert verdicts["same_source_entropy_reconstructed"] is True
    assert all(value > 0.0 for value in report["stress_flux_by_cap"])
    assert all(np.isfinite(value) for value in report["entropy_by_cap"])


def test_current_source_has_no_universal_coupling(report: dict) -> None:
    # Frozen empirical status of the present source at this cutoff: the
    # per-cap entropy-to-stress ratios spread far beyond the preregistered
    # universality envelope, so no single coupling constant describes the
    # family. A future source law that changes this verdict must update this
    # assertion deliberately alongside the issue-576 status.
    assert report["clause_verdicts"]["coupling_universal_across_family"] is False
    assert report["coupling_relative_spread"] > UNIVERSALITY_ENVELOPE
    assert report["verdict"] == "NOT_ATTAINED"
    assert report["blockers"] == [
        "coupling_ratio_spread_exceeds_universality_envelope"
    ]


def test_receipt_typing_is_fail_closed(report: dict) -> None:
    assert report["STRESS_COUPLING_RECEIPT"] is (report["verdict"] == "ATTAINED")
    assert report["physical_promotion_allowed"] is False
    assert PHYSICAL_PROMOTION_ALLOWED is False


def test_report_is_deterministic() -> None:
    first = produce_stress_coupling_report(config=_SMALL)
    second = produce_stress_coupling_report(config=_SMALL)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
