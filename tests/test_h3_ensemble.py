from oph_fpe.bulk.h3_ensemble import h3_ensemble_report


def test_h3_ensemble_aggregates_defect_precursor_without_bulk_claim():
    rows = [
        _row(4096, defect=True, h3=0.1, s2=0.5, shuffled=0.4),
        _row(4096, defect=False, h3=0.3, s2=0.4, shuffled=0.35),
        _row(65536, defect=True, h3=0.01, s2=0.08, shuffled=0.03),
    ]

    report = h3_ensemble_report(rows)

    assert report["run_count"] == 3
    assert report["support_visible_lorentz_all"] is True
    assert report["conformal_h3_chart_all"] is True
    assert report["defect_h3_support_any_scale"] is True
    assert report["defect_h3_worldline_any_scale"] is False
    assert report["bulk_3d_established"] is False
    assert report["physical_particle_prediction"] is False
    small = report["sizes"][0]
    assert small["patch_count"] == 4096
    assert small["defect_cluster_support_fraction"] == 0.5
    defect = small["reports"]["defect_cluster"]
    assert defect["run_count"] == 2
    assert defect["receipt_fraction"] == 0.5
    assert defect["median_h3_over_s2"] < 1.0
    worldlines = small["reports"]["defect_h3_worldlines"]
    assert worldlines["run_count"] == 2
    assert worldlines["receipt_fraction"] == 0.0
    assert worldlines["median_h3_over_s2"] > 1.0


def test_h3_ensemble_handles_empty_rows():
    report = h3_ensemble_report([])

    assert report["run_count"] == 0
    assert report["sizes"] == []
    assert report["bulk_3d_established"] is False


def _row(patch_count: int, *, defect: bool, h3: float, s2: float, shuffled: float):
    report = {
        "mode": "support_profile_h3_fit",
        "support_count": 12,
        "support_size_summary": {"median": 20},
        "h3_fit": {"median_residual": h3},
        "s2_boundary_control": {"median_residual": s2},
        "shuffled_cap_response_control": {"median_residual": shuffled},
        "record_populated_h3_receipt": defect,
    }
    return {
        "patch_count": patch_count,
        "support_visible_lorentz_3p1_kinematics_receipt": True,
        "conformal_h3_spatial_chart_receipt": True,
        "record_populated_h3_spatial_receipt": False,
        "record_family_h3_support_receipt": False,
        "defect_cluster_h3_support_receipt": defect,
        "matter_defect_h3_support_receipt": defect,
        "defect_worldline_precursor_receipt": True,
        "defect_h3_worldline_precursor_receipt": False,
        "spatial_bulk_3d_reconstruction_receipt": False,
        "bulk_3d_established": False,
        "reports": {
            "observer_cap_response": {},
            "record_family": {},
            "defect_cluster": report,
            "defect_h3_worldlines": {
                "mode": "defect_timeline_h3_worldline_fit",
                "event_count": 12,
                "worldline_count": 3,
                "persistent_h3_worldline_count": 3,
                "median_h3_residual": h3,
                "median_s2_boundary_residual": max(s2 * 0.2, 1e-9),
                "median_shuffled_residual": shuffled,
                "bulk_worldline_precursor_receipt": False,
            },
        },
    }
