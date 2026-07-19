from __future__ import annotations

import math
from itertools import repeat

from oph_fpe.cosmology.collar_poisson import collar_poisson_counting_certificate
from oph_fpe.cosmology.collar_poisson import (
    IMPORTED_THEORY_WITNESS_CERTIFICATE_ID,
    IMPORTED_THEORY_WITNESS_SHA256,
)


def test_collar_poisson_recomputes_le_cam_and_exact_tv_bound() -> None:
    report = collar_poisson_counting_certificate(
        activation_probabilities=[0.02] * 25,
        limiting_mean=0.5,
        cut_sqrt_measure=2.0,
    )

    assert report["COLLAR_POISSON_COUNTING_RECOMPUTATION_RECEIPT"] is True
    assert math.isclose(report["mu_r"], 0.5)
    assert math.isclose(report["sum_p_squared"], 0.01)
    assert report["exact_total_variation_computed"] is True
    assert report["exact_total_variation_within_bound"] is True
    assert math.isclose(report["lambda_collar_from_declared_limit"], 0.25)
    assert report["PHYSICAL_COLLAR_MODEL_REALIZATION_RECEIPT"] is False
    assert report["PHYSICAL_GALAXY_POISSON_RECEIPT"] is False
    assert (
        report["imported_theory_witness_certificate_id"]
        == IMPORTED_THEORY_WITNESS_CERTIFICATE_ID
    )
    assert report["imported_theory_witness_sha256"] == IMPORTED_THEORY_WITNESS_SHA256
    assert report["IMPORTED_THEORY_WITNESS_SIMULATION_RECEIPT_ELIGIBLE"] is False


def test_collar_poisson_rejects_boolean_and_keeps_physical_gate_closed() -> None:
    report = collar_poisson_counting_certificate(
        activation_probabilities=[False, 0.1],
        limiting_mean=0.1,
    )

    assert report["COLLAR_POISSON_COUNTING_RECOMPUTATION_RECEIPT"] is False
    assert report["DECLARED_INDEPENDENT_BERNOULLI_MODEL"] is False
    assert "activation_probabilities_invalid" in report["blockers"]
    assert report["PHYSICAL_COLLAR_MODEL_REALIZATION_RECEIPT"] is False


def test_collar_poisson_rejects_numeric_strings() -> None:
    report = collar_poisson_counting_certificate(
        activation_probabilities=["0.1"],  # type: ignore[list-item]
        limiting_mean=0.1,
    )

    assert report["COLLAR_POISSON_COUNTING_RECOMPUTATION_RECEIPT"] is False
    assert report["DECLARED_INDEPENDENT_BERNOULLI_MODEL"] is False


def test_collar_poisson_fails_closed_on_integer_float_overflow() -> None:
    huge = 10**400
    probability = collar_poisson_counting_certificate(
        activation_probabilities=[huge],
        limiting_mean=0.0,
    )
    mean = collar_poisson_counting_certificate(
        activation_probabilities=[0.0],
        limiting_mean=huge,
    )

    assert probability["COLLAR_POISSON_COUNTING_RECOMPUTATION_RECEIPT"] is False
    assert "activation_probabilities_invalid" in probability["blockers"]
    assert mean["COLLAR_POISSON_COUNTING_RECOMPUTATION_RECEIPT"] is False
    assert "limiting_mean_invalid" in mean["blockers"]


def test_collar_poisson_stops_consuming_an_unbounded_iterable_at_the_cell_cap() -> None:
    report = collar_poisson_counting_certificate(
        activation_probabilities=repeat(0.0),
        limiting_mean=0.0,
    )

    assert report["cell_count"] == 4097
    assert report["DECLARED_INDEPENDENT_BERNOULLI_MODEL"] is False
    assert "bernoulli_cell_budget_exceeded" in report["blockers"]


def test_collar_poisson_rejects_nonfinite_derived_intensity() -> None:
    report = collar_poisson_counting_certificate(
        activation_probabilities=[0.0],
        limiting_mean=1.0e308,
        cut_sqrt_measure=5.0e-324,
    )

    assert report["COLLAR_POISSON_COUNTING_RECOMPUTATION_RECEIPT"] is False
    assert report["lambda_collar_from_declared_limit"] is None
    assert "lambda_collar_not_finite" in report["blockers"]
