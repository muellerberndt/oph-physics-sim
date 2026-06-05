from __future__ import annotations

from oph_fpe.scale.refinement_report import refinement_scaling_report


def test_refinement_report_negative_slope():
    rows = [
        {"patch_count": 4096, "state_derived_median": 0.8, "epsilon_cmi": 0.2},
        {"patch_count": 65536, "state_derived_median": 0.5, "epsilon_cmi": 0.18},
        {"patch_count": 262144, "state_derived_median": 0.35, "epsilon_cmi": 0.17},
    ]
    report = refinement_scaling_report(rows)
    assert report["run_count"] == 3
    assert report["slope_negative"] is True
    assert report["sizes"][0]["patch_count"] == 4096


def test_refinement_report_flags_numerical_floor():
    rows = [
        {"patch_count": 4096, "state_derived_median": 2e-15, "epsilon_cmi": 0.2},
        {"patch_count": 65536, "state_derived_median": 3e-15, "epsilon_cmi": 0.18},
    ]
    report = refinement_scaling_report(rows)
    assert report["numerical_floor_detected"] is True
    assert report["slope_interpretation"] == "not_meaningful_at_numerical_floor"
