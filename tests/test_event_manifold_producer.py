"""Fail-closed tests for the prescribed event-chart/cone diagnostic."""

from __future__ import annotations

import json

import pytest

from oph_fpe.bulk.event_manifold_producer import (
    PHYSICAL_PROMOTION_ALLOWED,
    TARGET_INERTIA,
    produce_event_manifold_report,
)

_SMALL = {"carrier_count": 32, "cycles": 6, "seed": 20260751}


@pytest.fixture(scope="module")
def report() -> dict:
    return produce_event_manifold_report(config=_SMALL)


def test_instrument_is_valid_all_controls_fail_closed(report: dict) -> None:
    controls = report["negative_controls"]
    assert set(controls) == {
        "shuffled_ancestry",
        "collapsed_chart",
        "mixed_source_events",
    }
    for name, row in controls.items():
        assert row["control_failure_detected"] is True, name
    assert report["controls_fail_closed"] is True


def test_causal_structure_is_nondegenerate(report: dict) -> None:
    verdicts = report["clause_verdicts"]
    assert verdicts["event_classes_and_order_constructed"] is True
    assert verdicts["prescribed_four_coordinate_feature_chart_constructed"] is True
    assert verdicts["nondegenerate_causal_structure"] is True
    assert report["causal_pair_count"] > 0
    assert report["spacelike_pair_count"] > 0


def test_current_source_event_geometry_is_not_lorentzian(report: dict) -> None:
    # Frozen empirical status for the source-round producer: the prescribed
    # ancestry-depth plus three-spectral-coordinate ansatz has Euclidean
    # inertia (4,0) and a negative cone margin. This is not independent
    # dimension evidence. A future source law that changes this verdict must
    # update this assertion deliberately.
    fit = report["held_out_quadratic_fit"]
    assert fit["fitted"] is True
    assert tuple(fit["inertia"]) == (4, 0)
    assert tuple(fit["inertia"]) != TARGET_INERTIA
    assert fit["cone_margin"] < 0.0
    assert report["diagnostic_verdict"] == "NOT_ATTAINED"
    assert set(report["blockers"]) == {
        "cone_margin_not_positive_on_held_out_pairs",
        "held_out_form_inertia_is_not_lorentzian",
    }


def test_receipt_typing_is_fail_closed(report: dict) -> None:
    assert report["EVENT_MANIFOLD_RECEIPT"] is False
    assert report["legacy_event_manifold_receipt_retired"] is True
    assert report["SOURCE_DERIVED_CAUSAL_3P1_MANIFOLD_LIMIT_RECEIPT"] is False
    assert report["FINITE_EVENT_CHART_DIAGNOSTIC_REPRODUCTION_RECEIPT"] is True
    assert report["PRESCRIBED_EVENT_CHART_CONE_COMPATIBILITY_DIAGNOSTIC"] is False
    assert report["physical_promotion_allowed"] is False
    assert PHYSICAL_PROMOTION_ALLOWED is False


def test_report_is_deterministic() -> None:
    first = produce_event_manifold_report(config=_SMALL)
    second = produce_event_manifold_report(config=_SMALL)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
