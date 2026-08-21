"""Conditioning repair for the event-chart quadratic form.

The chart mixes an integer ancestry-depth coordinate with three spectral
coordinates whose spreads differ by about three orders of magnitude.  Fitting
the form on the raw coordinates lets the depth direction dominate and pushes
the spectral directions into a numerically degenerate eigenvalue, so the
reported inertia turns on a near-zero direction.  These tests pin the repair
and pin that the default path is unchanged.
"""

import numpy as np

from oph_fpe.bulk.event_manifold_producer import (
    _fit_quadratic_form,
    standardize_chart,
)


def _mixed_scale_chart(count: int = 64) -> np.ndarray:
    rng = np.random.default_rng(20260821)
    chart = np.zeros((count, 4))
    chart[:, 0] = rng.integers(0, 120, size=count)
    chart[:, 1:] = rng.normal(scale=0.05, size=(count, 3))
    return chart


def _pairs(count: int = 64) -> dict[str, list[tuple[int, int]]]:
    """Pairs split across both parities so the fit has training and held-out rows."""

    causal = [(i, i + 2) for i in range(0, count - 2)]
    spacelike = [(i, i + 5) for i in range(0, count - 5)]
    return {"causal": causal, "spacelike": spacelike}


def test_standardize_chart_gives_unit_spread_per_coordinate() -> None:
    standardized = standardize_chart(_mixed_scale_chart())
    assert np.allclose(standardized.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(standardized.std(axis=0), 1.0, atol=1e-12)


def test_standardize_chart_leaves_constant_coordinates_finite() -> None:
    chart = _mixed_scale_chart()
    chart[:, 2] = 7.0
    standardized = standardize_chart(chart)
    assert np.all(np.isfinite(standardized))
    assert np.allclose(standardized[:, 2], 0.0)


def test_raw_fit_keeps_a_numerically_degenerate_direction() -> None:
    fit = _fit_quadratic_form(_mixed_scale_chart(), _pairs())
    assert fit["fitted"]
    assert fit["standardized"] is False
    eigenvalues = np.abs(np.asarray(fit["eigenvalues"], dtype=float))
    assert eigenvalues.min() / eigenvalues.max() < 1.0e-3


def test_standardized_fit_resolves_every_direction() -> None:
    fit = _fit_quadratic_form(_mixed_scale_chart(), _pairs(), standardize=True)
    assert fit["fitted"]
    assert fit["standardized"] is True
    eigenvalues = np.abs(np.asarray(fit["eigenvalues"], dtype=float))
    assert eigenvalues.min() / eigenvalues.max() > 1.0e-3


def test_standardized_fit_is_invariant_under_coordinate_rescaling() -> None:
    chart = _mixed_scale_chart()
    rescaled = chart * np.array([1.0, 1000.0, 0.001, 10.0])
    base = _fit_quadratic_form(chart, _pairs(), standardize=True)
    other = _fit_quadratic_form(rescaled, _pairs(), standardize=True)
    assert np.allclose(base["eigenvalues"], other["eigenvalues"], rtol=1e-8)


def test_default_path_is_unchanged_by_the_repair() -> None:
    chart = _mixed_scale_chart()
    assert _fit_quadratic_form(chart, _pairs())["eigenvalues"] == (
        _fit_quadratic_form(chart, _pairs(), standardize=False)["eigenvalues"]
    )
