from __future__ import annotations

import math

from oph_fpe.homochirality import (
    analytic_frank_excess,
    chiral_fixed_points,
    homochirality_demo_report,
)


def test_analytic_frank_excess_matches_closed_form_examples() -> None:
    target = analytic_frank_excess(0.01, 1.0, 6.554)

    assert math.isclose(target, 0.99, rel_tol=2.0e-4)
    assert analytic_frank_excess(0.0, 1.0, 10.0) == 0.0
    assert analytic_frank_excess(-0.01, 1.0, 6.554) < 0.0


def test_chiral_fixed_points_obey_repair_threshold() -> None:
    assert chiral_fixed_points(0.3, 0.15) == [0.0]
    points = chiral_fixed_points(1.0, 0.15)

    assert len(points) == 3
    assert math.isclose(points[-1], math.sqrt(0.7))
    assert points[0] == -points[-1]


def test_homochirality_demo_is_explicitly_nonphysical_and_bounded() -> None:
    report = homochirality_demo_report(e0=0.01, kappa=1.0, mu=0.15)

    assert report["status"] == "model_demonstrator"
    assert report["physicalClaim"] is False
    assert report["branchCriterion"]["macroscopicBranchesInModel"] is True
    assert report["receipts"]["NORMAL_FORM_NUMERICAL_BOUNDEDNESS_CHECK"] is True
    assert report["receipts"]["PREBIOTIC_SOURCE_RATE_RECEIPT"] is False
    assert all(-1.0 <= value <= 1.0 for value in report["trajectory"]["enantiomericExcess"])
