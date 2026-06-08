from __future__ import annotations

from oph_fpe.cosmology.cl_ensemble import cl_ensemble_report


def test_cl_ensemble_aggregates_gate_allowed_runs():
    rows = [
        _row(4096, True, [0.1, 0.2, 0.3], peak=4, delta=0.5, corr=0.2),
        _row(4096, True, [0.12, 0.19, 0.28], peak=4, delta=0.6, corr=0.25),
        _row(4096, False, [9.0, 9.0, 9.0], peak=2, delta=0.0, corr=1.0),
    ]

    report = cl_ensemble_report(rows)

    assert report["run_count"] == 3
    assert report["gate_allowed_count"] == 2
    assert report["physical_cmb_prediction"] is False
    size = report["sizes"][0]
    assert size["gate_allowed_fraction"] == 2 / 3
    assert size["cmb_lite"]["best_positive_field_counts"] == {"record_signature_smooth_k64": 2}
    assert size["cmb_lite"]["mean_best_positive_shape_correlation"] == 0.35
    field = size["fields"]["record_signature"]
    assert field["run_count"] == 2
    assert field["peak_ell_mode"] == 4
    assert field["median_min_relative_l2_delta_to_controls"] == 0.55
    assert field["mean_spectrum"][0]["ell"] == 2
    assert field["mean_spectrum"][0]["mean_D_ell"] == 0.11
    assert field["mean_pairwise_shape_correlation"] is not None


def test_cl_ensemble_reports_empty_fields_when_gate_fails():
    report = cl_ensemble_report([_row(4096, False, [1.0], peak=2, delta=0.1, corr=0.9)])

    assert report["gate_allowed_count"] == 0
    assert report["sizes"][0]["fields"] == {}


def _row(patch_count: int, allowed: bool, values: list[float], *, peak: int, delta: float, corr: float):
    return {
        "patch_count": patch_count,
        "gate_allowed": allowed,
        "fields": {
            "record_signature": {
                "peak_ell": peak,
                "spectrum": [
                    {"ell": ell, "D_ell": value, "C_ell": value}
                    for ell, value in enumerate(values, start=2)
                ],
                "control_comparison": {
                    "min_relative_l2_delta": delta,
                    "max_shape_correlation": corr,
                },
            }
        },
        "cmb_lite": {
            "best_field": "stable_count",
            "best_shape_correlation": -0.7,
            "best_normalized_rmse": 0.65,
            "best_positive_field": "record_signature_smooth_k64",
            "best_positive_shape_correlation": 0.35,
            "best_positive_normalized_rmse": 0.91,
        },
    }
