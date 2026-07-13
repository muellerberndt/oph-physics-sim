import numpy as np

from oph_fpe.cosmology.screen_parity import (
    _knn,
    pseudoscalar_statistic,
    screen_parity_report,
)


def _cap_points(count: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Spherical cap around +z (theta < 1.0 rad), away from the coordinate
    # singularity so the planted chiral pair is well defined.
    theta = 0.15 + 0.85 * rng.random(count)
    phi = 2.0 * np.pi * rng.random(count)
    return np.column_stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def test_planted_chirality_detected_and_mirror_flips():
    points = _cap_points(1500)
    # Plant: a = x, b = y on a +z polar cap. The tangent cross-gradient
    # pseudo-scalar of the coordinate pair equals the Jacobian z > 0 on the
    # cap, a smooth, single-valued chiral plant.
    field_a = points[:, 0].copy()
    field_b = points[:, 1].copy()
    neighbors = _knn(points, 12)
    chi = pseudoscalar_statistic(points, field_a, field_b, neighbors=neighbors)
    # The pseudo-scalar of the coordinate pair equals the Jacobian z on the
    # cap: mean(z) ~ 0.75 for this cap. The shuffle null is intentionally
    # NOT used here: shuffling a smooth field inflates gradient variance,
    # which makes the report's z-vs-shuffle a conservative detection
    # criterion for smooth fields, and an overpowered one for plants.
    assert 0.5 < chi < 1.05

    mirrored = points.copy()
    mirrored[:, 0] *= -1.0
    chi_mirror = pseudoscalar_statistic(mirrored, field_a, field_b, neighbors=neighbors)
    assert np.isclose(chi_mirror, -chi, rtol=1e-6, atol=1e-9)


def test_parity_symmetric_fields_read_null():
    points = _cap_points(1200, seed=9)
    rng = np.random.default_rng(4)
    field_a = rng.standard_normal(points.shape[0])
    field_b = rng.standard_normal(points.shape[0])
    neighbors = _knn(points, 12)
    chi = pseudoscalar_statistic(points, field_a, field_b, neighbors=neighbors)
    nulls = [
        pseudoscalar_statistic(points, field_a, rng.permutation(field_b), neighbors=neighbors)
        for _ in range(24)
    ]
    null_std = float(np.std(nulls))
    assert abs(chi) < 4.0 * null_std


def test_screen_parity_report_missing_artifacts(tmp_path):
    report = screen_parity_report(tmp_path)
    assert report["status"] == "missing_freezeout_fields"
