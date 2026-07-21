"""Fail-closed tests for the issue-574 finite GNS-tower instrument."""

from __future__ import annotations

import json

import pytest

from oph_fpe.bulk.gns_tower_producer import (
    PHYSICAL_PROMOTION_ALLOWED,
    cap_ports,
    produce_gns_tower_report,
)

_SMALL = {"carrier_count": 32, "cycles": 6, "seed": 20260751}
_FINE = {"carrier_count": 64, "cycles": 6, "seed": 20260751}


@pytest.fixture(scope="module")
def report() -> dict:
    return produce_gns_tower_report(config=_SMALL, refinement_config=_FINE)


def test_instrument_is_valid_all_controls_fail_closed(report: dict) -> None:
    controls = report["negative_controls"]
    assert set(controls) == {
        "support_collapse",
        "identity_inclusion",
        "inconsistent_intersection",
        "noncyclic_vector",
        "broken_generator_relation",
    }
    for name, row in controls.items():
        assert row["control_failure_detected"] is True, name
    assert report["controls_fail_closed"] is True


def test_cyclicity_and_separation_clauses_attained(report: dict) -> None:
    verdicts = report["clause_verdicts"]
    assert verdicts["cyclicity_uniform_on_test_frame"] is True
    assert verdicts["state_separating_support_floor"] is True
    assert report["tower_levels"]["main"]["support_floor"] > 0.0


def test_modular_intersection_constructed_on_transverse_pair(report: dict) -> None:
    verdicts = report["clause_verdicts"]
    assert verdicts["modular_intersection_constructed"] is True
    assert verdicts["modular_intersection_converged"] is True
    main = report["tower_levels"]["main"]
    assert main["intersection_nonempty"] is True
    overlap = set(cap_ports(0)) & set(cap_ports(5))
    assert set(main["intersection_ports"]) == overlap
    assert main["modular_intersection_residual"] is not None


def test_current_source_does_not_attain_future_cone_spectrum(
    report: dict,
) -> None:
    # Frozen empirical status of the present source at this cutoff: three of
    # the four candidate null translations built from the modular generator
    # and the two m4 directions are positive; the fourth is not, so the
    # future-cone clause fails. A future source law that changes this verdict
    # must update this assertion deliberately alongside the issue-574 status.
    assembly = report["null_generator_assembly"]
    assert assembly["future_cone_spectrum_attained"] is False
    assert assembly["positive_candidate_count"] == 3
    assert report["verdict"] == "NOT_ATTAINED"
    assert report["blockers"] == ["future_cone_spectrum_not_attained"]


def test_receipt_typing_is_fail_closed(report: dict) -> None:
    assert report["GNS_TOWER_CLAUSES_RECEIPT"] is (
        report["verdict"] == "ATTAINED"
    )
    assert report["physical_promotion_allowed"] is False
    assert PHYSICAL_PROMOTION_ALLOWED is False


def test_report_is_deterministic() -> None:
    first = produce_gns_tower_report(config=_SMALL, refinement_config=_FINE)
    second = produce_gns_tower_report(config=_SMALL, refinement_config=_FINE)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
