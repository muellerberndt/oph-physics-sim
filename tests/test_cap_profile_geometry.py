import numpy as np

from oph_fpe.bulk.cap_normals import cap_normal, minkowski_dot
from oph_fpe.bulk.cap_profile_geometry import (
    cap_profile_for_h3_point,
    caps_to_h3_minimal_receipt,
    fit_h3_point_from_cap_profile,
    h3_point_from_ball,
    sample_round_caps,
    verify_h3_point,
)


def test_cap_profile_h3_point_reconstructs_from_caps():
    caps = sample_round_caps(24, [0.45, 0.75, 1.05], seed=11)
    normals = np.vstack([cap_normal(cap) for cap in caps])
    assert float(np.max(np.abs(minkowski_dot(normals, normals) - 1.0))) < 1.0e-10

    point = h3_point_from_ball(np.array([0.35, -0.2, 0.5]))
    assert verify_h3_point(point)

    profile = cap_profile_for_h3_point(point, normals, softness=0.25)
    fit = fit_h3_point_from_cap_profile(
        profile,
        normals,
        softness=0.25,
        radius=1.5,
        restarts=10,
        seed=12,
        max_nfev=160,
    )

    assert fit.mean_squared_error < 1.0e-4
    assert verify_h3_point(fit.point)


def test_caps_to_h3_minimal_receipt_passes_and_beats_controls():
    report = caps_to_h3_minimal_receipt(
        axis_count=24,
        theta_values=[0.45, 0.75, 1.05],
        object_count=5,
        object_radius=0.85,
        fit_radius=1.5,
        softness=0.25,
        restarts=10,
        seed=13,
        max_median_error=0.02,
    )

    assert report["S2_CAP_PROFILE_TO_H3_RECEIPT"] is True
    assert report["cap_normal_checks_pass"] is True
    assert report["h3_beats_shuffled"] is True
    assert report["h3_beats_s2_boundary"] is True
