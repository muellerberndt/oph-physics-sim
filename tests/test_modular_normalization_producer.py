"""Fail-closed tests for the issue-573 modular-normalization instrument."""

from __future__ import annotations

import json

import numpy as np
import pytest

from oph_fpe.bulk.modular_normalization_producer import (
    PHYSICAL_PROMOTION_ALLOWED,
    TWO_PI,
    cap_interior_state,
    geometric_flow_rate,
    produce_modular_normalization_report,
    _axis_frame,
)

_SMALL = {"carrier_count": 32, "cycles": 6, "seed": 20260751}
_FINE = {"carrier_count": 64, "cycles": 6, "seed": 20260751}


@pytest.fixture(scope="module")
def report() -> dict:
    return produce_modular_normalization_report(
        config=_SMALL, refinement_config=_FINE
    )


def test_instrument_is_valid_all_controls_fail_closed(report: dict) -> None:
    controls = report["negative_controls"]
    assert set(controls) == {
        "wrong_frame",
        "reversed_orientation",
        "nonfaithful_state",
        "truncated_extraction",
        "permuted_cap_state",
    }
    for name, row in controls.items():
        assert row["control_failure_detected"] is True, name
    assert report["controls_fail_closed"] is True


def test_geometry_side_is_frozen_and_fits_cleanly(report: dict) -> None:
    assert report["geometric_rate"] == pytest.approx(1.0, abs=1.0e-9)
    assert report["geometric_fit_residual"] < 1.0e-12
    assert report["geometry_side_freeze_sha256"].startswith("sha256:")
    assert report["state_side_freeze_sha256"].startswith("sha256:")


def test_cap_family_is_nondegenerate_and_faithful(report: dict) -> None:
    assert report["cap_family_axis_count"] == 6
    assert report["faithful_axis_count"] == 6


def test_verdict_is_fail_closed_and_typed(report: dict) -> None:
    assert report["verdict"] in {"ATTAINED", "NOT_ATTAINED"}
    assert report["GEOMETRIC_MODULAR_NORMALIZATION_RECEIPT"] is (
        report["verdict"] == "ATTAINED"
    )
    assert report["physical_promotion_allowed"] is False
    assert PHYSICAL_PROMOTION_ALLOWED is False


def test_current_source_dynamics_does_not_attain_bw_normalization(
    report: dict,
) -> None:
    # Frozen empirical status of the present repair dynamics at this cutoff:
    # the framed cap states do not thermalize at the geometric temperature,
    # so the fitted normalization interval sits outside the preregistered
    # Bisognano-Wichmann acceptance band.  If a future source law changes
    # this verdict, this assertion must be updated deliberately alongside the
    # issue-573 status; it must never be weakened to pass silently.
    assert report["verdict"] == "NOT_ATTAINED"
    assert "normalization_interval_outside_acceptance_band" in report["blockers"]
    low, high = report["normalization_interval"]
    band_low, band_high = report["bw_acceptance_band"]
    assert high < band_low or low > band_high
    assert report["bw_acceptance_band"][0] < TWO_PI < report["bw_acceptance_band"][1]


def test_wrong_bands_are_preregistered_and_disjoint(report: dict) -> None:
    bands = report["preregistered_wrong_bands"]
    assert len(bands) == 2
    acceptance = report["bw_acceptance_band"]
    for band in bands:
        assert band[1] < acceptance[0] or acceptance[1] < band[0]


def test_report_is_deterministic() -> None:
    first = produce_modular_normalization_report(
        config=_SMALL, refinement_config=_FINE
    )
    second = produce_modular_normalization_report(
        config=_SMALL, refinement_config=_FINE
    )
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_frames_are_orthonormal() -> None:
    for axis in range(6):
        frame = _axis_frame(axis)
        assert np.allclose(frame.T @ frame, np.eye(4), atol=1.0e-12)


def test_nonfaithful_input_is_rejected_by_state_builder() -> None:
    rank_one = np.ones((1, 12))
    state = cap_interior_state(rank_one, _axis_frame(0), regularizer=0.0)
    assert state["faithful"] is False
    assert state["modular_hamiltonian"] is None
