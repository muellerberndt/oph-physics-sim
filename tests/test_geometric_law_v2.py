"""Frozen status of the declared v2 geometric law against the instruments."""

from __future__ import annotations

import numpy as np
import pytest

from oph_fpe.dynamics.geometric_law_v2 import (
    PHYSICAL_PROMOTION_ALLOWED,
    capture_geometric_law,
)
from oph_fpe.bulk.gns_tower_producer import _null_generator_report
from oph_fpe.bulk.stress_coupling_producer import cap_stress_flux
from oph_fpe.bulk.modular_normalization_producer import geometric_flow_rate


@pytest.fixture(scope="module")
def capture() -> dict:
    return capture_geometric_law({"carrier_count": 32, "cycles": 12})


def test_target2_future_cone_attained_by_construction(capture: dict) -> None:
    report = _null_generator_report(capture)
    assert report["positive_candidate_count"] == 4
    assert report["future_cone_spectrum_attained"] is True


def test_target4_flux_universal_with_zero_spread(capture: dict) -> None:
    # The law is A5-equivariant by construction, so per-cap flux is equal
    # across the family, matching the Lean theorem A5CouplingSymmetry.
    flux = [cap_stress_flux(capture, axis) for axis in range(6)]
    assert max(flux) - min(flux) < 1.0e-9
    assert min(flux) > 0.0


def test_target5_record_snapshots_span_port_space(capture: dict) -> None:
    samples = np.stack(
        [
            np.asarray(row["full_port_state"])
            for snapshot in capture["source_artifacts"]["dynamics"][
                "record_state_snapshots"
            ]
            for row in snapshot["carrier_rows"]
        ]
    )
    moment = samples.T @ samples / samples.shape[0]
    assert float(np.linalg.eigvalsh(moment).min()) > 1.0e-12


def test_target1_v2_design_defects_are_recorded(capture: dict) -> None:
    # Honest frozen status: the v2 thermalization design does NOT attain the
    # issue-573 target. Two identified defects: the transported geometry rows
    # are not Moebius-correct (fitted rate far from one), and the framed
    # projection of the twelve-port equilibrium is not the framed Gibbs
    # state. Fixing both is open work under #595; this assertion must be
    # flipped deliberately when it lands.
    geometry = geometric_flow_rate(capture)
    assert abs(geometry["rate"] - 1.0) > 0.5


def test_no_promotion_flag(capture: dict) -> None:
    assert capture["physical_promotion_allowed"] is False
    assert PHYSICAL_PROMOTION_ALLOWED is False
    assert capture["law_id"] == "geometric_detailed_balance_v2"
