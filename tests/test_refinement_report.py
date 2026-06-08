from __future__ import annotations

from oph_fpe.scale.refinement_report import refinement_scaling_report


def test_refinement_report_negative_slope():
    rows = [
        {
            "patch_count": 4096,
            "state_derived_median": 0.8,
            "epsilon_cmi": 0.2,
            "state_derived_correct_beats_controls": False,
            "state_selected_2pi": False,
            "state_selected_scale_label": "1x",
            "transition_two_pi_selected": True,
            "transition_selected_label": "2pi",
            "neutral_bulk_3d_established": False,
            "physical_cmb_prediction": False,
        },
        {
            "patch_count": 65536,
            "state_derived_median": 0.5,
            "epsilon_cmi": 0.18,
            "state_derived_correct_beats_controls": True,
            "state_selected_2pi": True,
            "state_selected_scale_label": "2pi",
            "transition_two_pi_selected": True,
            "transition_selected_label": "2pi",
            "neutral_bulk_3d_established": False,
            "physical_cmb_prediction": False,
        },
        {
            "patch_count": 262144,
            "state_derived_median": 0.35,
            "epsilon_cmi": 0.17,
            "state_derived_correct_beats_controls": True,
            "state_selected_2pi": True,
            "state_selected_scale_label": "2pi",
            "transition_two_pi_selected": True,
            "transition_selected_label": "2pi",
            "neutral_bulk_3d_established": False,
            "physical_cmb_prediction": False,
        },
    ]
    report = refinement_scaling_report(rows)
    assert report["run_count"] == 3
    assert report["slope_negative"] is True
    assert report["sizes"][0]["patch_count"] == 4096
    assert report["state_control_pass_total"] == 2
    assert report["state_2pi_selected_total"] == 2
    assert report["transition_2pi_selected_total"] == 3
    assert report["physical_cmb_prediction_total"] == 0
    assert report["sizes"][1]["state_selected_scale_counts"] == {"2pi": 1}


def test_refinement_report_flags_numerical_floor():
    rows = [
        {"patch_count": 4096, "state_derived_median": 2e-15, "epsilon_cmi": 0.2},
        {"patch_count": 65536, "state_derived_median": 3e-15, "epsilon_cmi": 0.18},
    ]
    report = refinement_scaling_report(rows)
    assert report["numerical_floor_detected"] is True
    assert report["slope_interpretation"] == "not_meaningful_at_numerical_floor"
