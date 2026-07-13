from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from scipy.integrate import quad
from scipy.optimize import brentq

from oph_fpe.constants.oph_pixel import P_STAR


G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30

# Frozen continuation values reported by the OPH dark-response paper.  These
# are calibrated/conditional readouts, not theorem outputs from the simulator.
OPH_A0_M_S2 = 1.029186271e-10
OPH_LAMBDA_Z6_TARGET = math.exp(-P_STAR / 24.0)

# Park et al. (2026), arXiv:2602.17884.  The public comparison is a published
# summary statistic; the underlying Cassini radiometric data are not bundled.
PARK_GALACTIC_EXTERNAL_ACCELERATION_M_S2 = 2.32e-10
PARK_GALACTIC_EXTERNAL_ACCELERATION_SIGMA_M_S2 = 0.16e-10
PARK_RAR_BENCHMARK_A0_M_S2 = 1.02e-10
PARK_RAR_SPHERICAL_Q2_S2 = 3.387e-26
PARK_RAR_DISK_Q2_S2 = 3.411e-26
PARK_RAR_DISK_E_NEWTONIAN = 1.659
CASSINI_Q2_CENTRAL_S2 = 1.6e-27
CASSINI_Q2_SIGMA_S2 = 1.8e-27


def rar_nu(y: float) -> float:
    """RAR/QUMOND interpolating function nu(y)=1/[1-exp(-sqrt(y))]."""

    value = max(float(y), 0.0)
    root = math.sqrt(value)
    if root == 0.0:
        return math.inf
    return 1.0 / (-math.expm1(-root))


def rar_nu_minus_one(y: float) -> float:
    """Stable evaluation of ``rar_nu(y) - 1``."""

    root = math.sqrt(max(float(y), 0.0))
    if root == 0.0:
        return math.inf
    if root > 700.0:
        return 0.0
    return 1.0 / math.expm1(root)


def newtonian_external_field_ratio(
    a0_m_s2: float,
    *,
    external_acceleration_m_s2: float = PARK_GALACTIC_EXTERNAL_ACCELERATION_M_S2,
) -> float:
    """Solve e_N nu(e_N)=a_e/a0 for the spherical RAR conversion."""

    a0 = float(a0_m_s2)
    external = float(external_acceleration_m_s2)
    if not math.isfinite(a0) or a0 <= 0.0:
        raise ValueError("a0_m_s2 must be finite and positive")
    if not math.isfinite(external) or external <= 0.0:
        raise ValueError("external_acceleration_m_s2 must be finite and positive")
    ratio = external / a0
    return float(brentq(lambda value: value * rar_nu(value) - ratio, 1.0e-12, ratio))


@lru_cache(maxsize=64)
def qumond_dimensionless_q(
    e_newtonian: float,
    *,
    epsabs: float = 2.0e-10,
    epsrel: float = 2.0e-10,
    limit: int = 500,
) -> float:
    """Evaluate Park et al. (2026) Eq. (9b) for the RAR interpolation.

    The outer integral is split around ``v=sqrt(e_N)``, where the Newtonian
    field can vanish at the angular endpoint.  The isolated zero-measure
    ``0 * infinity`` point is assigned its limiting integration-safe value.
    """

    e_n = float(e_newtonian)
    if not math.isfinite(e_n) or e_n <= 0.0:
        raise ValueError("e_newtonian must be finite and positive")
    if epsabs <= 0.0 or epsrel <= 0.0 or limit < 2:
        raise ValueError("integration tolerances and limit must be positive")

    def angular_integrand(xi: float, v: float) -> float:
        radicand = e_n * e_n + v**4 + 2.0 * e_n * v * v * xi
        field_magnitude = math.sqrt(max(radicand, 0.0))
        bracket = e_n * (3.0 * xi - 5.0 * xi**3) + v * v * (1.0 - 3.0 * xi * xi)
        if field_magnitude <= 1.0e-30:
            return 0.0
        return rar_nu_minus_one(field_magnitude) * bracket

    def angular_integral(v: float) -> float:
        value, _ = quad(
            lambda xi: angular_integrand(xi, v),
            -1.0,
            1.0,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
        return float(value)

    cancellation_scale = math.sqrt(e_n)
    intervals = (
        (0.0, 0.999999 * cancellation_scale),
        (0.999999 * cancellation_scale, 1.000001 * cancellation_scale),
        (1.000001 * cancellation_scale, 10.0),
        (10.0, math.inf),
    )
    integral = 0.0
    for lower, upper in intervals:
        value, _ = quad(
            angular_integral,
            lower,
            upper,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
        integral += float(value)
    return 1.5 * integral


def qumond_external_field_quadrupole(
    a0_m_s2: float,
    *,
    external_acceleration_m_s2: float = PARK_GALACTIC_EXTERNAL_ACCELERATION_M_S2,
    e_newtonian: float | None = None,
) -> dict[str, float]:
    """Return ``e_N``, dimensionless ``q``, and ``Q2`` for one fixed branch."""

    a0 = float(a0_m_s2)
    e_n = (
        newtonian_external_field_ratio(
            a0,
            external_acceleration_m_s2=external_acceleration_m_s2,
        )
        if e_newtonian is None
        else float(e_newtonian)
    )
    q = qumond_dimensionless_q(e_n)
    q2 = -1.5 * a0**1.5 * q / math.sqrt(G_SI * M_SUN_KG)
    return {
        "a0_m_s2": a0,
        "external_acceleration_m_s2": float(external_acceleration_m_s2),
        "e_newtonian": e_n,
        "q_dimensionless": q,
        "Q2_s2": float(q2),
    }


def cassini_external_field_report(
    *,
    cassini_central_s2: float = CASSINI_Q2_CENTRAL_S2,
    cassini_sigma_s2: float = CASSINI_Q2_SIGMA_S2,
    external_acceleration_m_s2: float = PARK_GALACTIC_EXTERNAL_ACCELERATION_M_S2,
    external_acceleration_sigma_m_s2: float = PARK_GALACTIC_EXTERNAL_ACCELERATION_SIGMA_M_S2,
    park_benchmark_a0_m_s2: float = PARK_RAR_BENCHMARK_A0_M_S2,
    park_benchmark_q2_s2: float = PARK_RAR_SPHERICAL_Q2_S2,
    oph_a0_m_s2: float = OPH_A0_M_S2,
    lambda_collar: float = OPH_LAMBDA_Z6_TARGET,
) -> dict[str, Any]:
    """Reproduce the Cassini/RAR benchmark and test fixed OPH static branches.

    The pulls reported here divide by the published Cassini standard
    uncertainty only.  They are transparent fixed-input diagnostics, not the
    posterior-predictive galaxy-versus-Cassini tensions reported by Park et al.
    """

    central = float(cassini_central_s2)
    sigma = float(cassini_sigma_s2)
    external_sigma = float(external_acceleration_sigma_m_s2)
    lam = float(lambda_collar)
    if not math.isfinite(central):
        raise ValueError("cassini_central_s2 must be finite")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("cassini_sigma_s2 must be finite and positive")
    if not math.isfinite(external_sigma) or external_sigma < 0.0:
        raise ValueError("external_acceleration_sigma_m_s2 must be finite and nonnegative")
    if external_sigma >= float(external_acceleration_m_s2):
        raise ValueError("external-field uncertainty must be smaller than its central value")
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError("lambda_collar must be finite and positive")

    benchmark = qumond_external_field_quadrupole(
        park_benchmark_a0_m_s2,
        external_acceleration_m_s2=external_acceleration_m_s2,
    )
    disk_benchmark = qumond_external_field_quadrupole(
        park_benchmark_a0_m_s2,
        external_acceleration_m_s2=external_acceleration_m_s2,
        e_newtonian=PARK_RAR_DISK_E_NEWTONIAN,
    )
    a0_eff = float(oph_a0_m_s2) / (lam * lam)
    z6_branch = qumond_external_field_quadrupole(
        a0_eff,
        external_acceleration_m_s2=external_acceleration_m_s2,
    )
    unit_branch = qumond_external_field_quadrupole(
        oph_a0_m_s2,
        external_acceleration_m_s2=external_acceleration_m_s2,
    )
    for branch in (benchmark, disk_benchmark, z6_branch, unit_branch):
        branch["raw_pull_vs_cassini_sigma"] = (branch["Q2_s2"] - central) / sigma
        branch["Q2_over_cassini_sigma"] = branch["Q2_s2"] / sigma

    benchmark_relative_error = abs(benchmark["Q2_s2"] - park_benchmark_q2_s2) / abs(
        park_benchmark_q2_s2
    )
    disk_benchmark_relative_error = abs(
        disk_benchmark["Q2_s2"] - PARK_RAR_DISK_Q2_S2
    ) / abs(PARK_RAR_DISK_Q2_S2)
    pulls = [unit_branch["raw_pull_vs_cassini_sigma"], z6_branch["raw_pull_vs_cassini_sigma"]]
    q2_values = [unit_branch["Q2_s2"], z6_branch["Q2_s2"]]
    two_sigma_upper = central + 2.0 * sigma
    if external_sigma > 0.0:
        z6_external_low = qumond_external_field_quadrupole(
            a0_eff,
            external_acceleration_m_s2=float(external_acceleration_m_s2) - external_sigma,
        )
        z6_external_high = qumond_external_field_quadrupole(
            a0_eff,
            external_acceleration_m_s2=float(external_acceleration_m_s2) + external_sigma,
        )
        z6_prediction_sigma_external = abs(
            z6_external_high["Q2_s2"] - z6_external_low["Q2_s2"]
        ) / 2.0
    else:
        z6_external_low = dict(z6_branch)
        z6_external_high = dict(z6_branch)
        z6_prediction_sigma_external = 0.0
    gaia_only_combined_sigma = math.sqrt(sigma * sigma + z6_prediction_sigma_external**2)

    return {
        "mode": "conditional_static_dark_response_external_field_test",
        "method": {
            "source_equations": "Park_et_al_2026_Eq_9a_9b",
            "interpolating_function": "nu(y)=1/[1-exp(-sqrt(y))]",
            "oph_equivalence": (
                "nu_lambda(g_N/a0_OPH)=nu_RAR(g_N/(a0_OPH/lambda_collar^2))"
            ),
            "integration": {
                "algorithm": "nested_adaptive_quadrature_split_at_v_equals_sqrt_eN",
                "epsabs": 2.0e-10,
                "epsrel": 2.0e-10,
                "limit": 500,
            },
        },
        "cassini": {
            "Q2_central_s2": central,
            "Q2_standard_uncertainty_s2": sigma,
            "confidence_level": "one_standard_uncertainty",
            "raw_tracking_data_bundled": False,
        },
        "inputs": {
            "G_m3_kg_s2": G_SI,
            "M_sun_kg": M_SUN_KG,
            "galactic_external_acceleration_m_s2": float(external_acceleration_m_s2),
            "galactic_external_acceleration_sigma_m_s2": external_sigma,
            "oph_a0_m_s2": float(oph_a0_m_s2),
            "P_endpoint_calibrated": float(P_STAR),
            "lambda_collar_z6_target": lam,
            "a0_effective_z6_m_s2": a0_eff,
            "lambda_status": "conditional_exact_uniform_product_thickening_target",
            "a0_status": "calibrated_benchmark_not_theorem_output",
        },
        "validation": {
            "park_rar_spherical": benchmark,
            "published_Q2_s2": float(park_benchmark_q2_s2),
            "relative_error": benchmark_relative_error,
            "receipt": benchmark_relative_error < 1.0e-4,
            "park_rar_disk": disk_benchmark,
            "published_disk_Q2_s2": PARK_RAR_DISK_Q2_S2,
            "disk_relative_error": disk_benchmark_relative_error,
            "disk_receipt": disk_benchmark_relative_error < 1.0e-3,
        },
        "oph_branches": {
            "z6_exact_uniform_target": z6_branch,
            "unit_lambda_endpoint": unit_branch,
            "jensen_lambda_band": {
                "lambda_min": min(lam, 1.0),
                "lambda_max": max(lam, 1.0),
                "Q2_min_s2": min(q2_values),
                "Q2_max_s2": max(q2_values),
                "raw_pull_min_sigma": min(pulls),
                "raw_pull_max_sigma": max(pulls),
            },
        },
        "fixed_input_diagnostic": {
            "raw_pull_not_nuisance_marginalized": True,
            "z6_raw_pull_sigma": z6_branch["raw_pull_vs_cassini_sigma"],
            "unit_lambda_raw_pull_sigma": unit_branch["raw_pull_vs_cassini_sigma"],
            "z6_Q2_s2": z6_branch["Q2_s2"],
            "unit_lambda_Q2_s2": unit_branch["Q2_s2"],
            "two_sigma_upper_Q2_s2": two_sigma_upper,
            "maximum_multiplicative_fraction_for_two_sigma_upper_z6": (
                two_sigma_upper / z6_branch["Q2_s2"]
            ),
            "z6_Q2_at_external_field_minus_one_sigma_s2": z6_external_low["Q2_s2"],
            "z6_Q2_at_external_field_plus_one_sigma_s2": z6_external_high["Q2_s2"],
            "z6_prediction_sigma_from_external_field_linearized_s2": (
                z6_prediction_sigma_external
            ),
            "z6_gaia_only_combined_pull_sigma": (
                (z6_branch["Q2_s2"] - central) / gaia_only_combined_sigma
            ),
            "gaia_only_combined_pull_not_full_nuisance_likelihood": True,
        },
        "applicability": {
            "assumption_tested": "universal_full_source_static_QUMOND_extension",
            "assumption_derived_by_current_oph": False,
            "current_paper_scope_match": False,
            "current_scope": "old_settled_low_acceleration_galaxy_equilibrium_continuation",
            "missing_gate": "source_derived_solar_system_applicability_or_screening_reduction",
        },
        "comparison_receipt": bool(
            benchmark_relative_error < 1.0e-4 and disk_benchmark_relative_error < 1.0e-3
        ),
        "physical_prediction_receipt": False,
        "assessment": (
            "strong negative result for the natural universal/full-source static extension; the currently "
            "scoped continuation instead has an unresolved Solar-System applicability gate"
        ),
        "claim_boundary": (
            "The fixed-input raw pull uses only the Cassini standard uncertainty and is not a nuisance-"
            "marginalized or posterior-predictive tension. It conditionally excludes applying the OPH "
            "QUMOND-form static continuation to the full Milky-Way-plus-Sun source. The OPH paper scopes "
            "that continuation to settled galaxies, so this does not falsify recovered OPH core; it requires "
            "a source-derived applicability, coarse-graining, screening, or transported-parent reduction "
            "with a quantitative Solar-System Q2."
        ),
    }
