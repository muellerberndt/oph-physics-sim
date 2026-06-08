from __future__ import annotations

import pytest

from oph_fpe.cosmology.anomaly_fluid import (
    anomaly_background_rhs,
    anomaly_perturbation_rhs,
    exchange_closure_report,
)


def test_anomaly_background_rhs_cdm_limit_is_pressureless_dust():
    rhs = anomaly_background_rhs(0.5, 8.0, H=2.0, Gamma=0.0, rho_A_eq=0.0)

    assert rhs == pytest.approx(-48.0)


def test_anomaly_background_rhs_includes_repair_exchange_term():
    rhs = anomaly_background_rhs(0.5, 8.0, H=2.0, Gamma=0.25, rho_A_eq=2.0)

    assert rhs == pytest.approx(-48.75)


def test_anomaly_perturbation_rhs_cdm_limit_terms():
    delta_prime, theta_prime = anomaly_perturbation_rhs(
        1.0,
        (0.2, 0.3),
        k=2.0,
        a=0.5,
        Hconf=3.0,
        Phi=0.0,
        Phi_prime=0.1,
        Psi=0.4,
        delta_b=0.7,
        rho_A=1.0,
        rho_A_eq=1.0,
        Gamma=0.0,
        B_A=0.0,
    )

    assert delta_prime == pytest.approx(0.0)
    assert theta_prime == pytest.approx(0.7)


def test_anomaly_perturbation_rhs_repair_exchange_term():
    delta_prime, _ = anomaly_perturbation_rhs(
        1.0,
        (0.2, 0.0),
        k=1.0,
        a=lambda eta, k: 0.5,
        Hconf=0.0,
        Phi=0.0,
        Phi_prime=0.0,
        Psi=0.0,
        delta_b=0.1,
        rho_A=2.0,
        rho_A_eq=1.0,
        Gamma=lambda k, a: 0.4,
        B_A=lambda k, a: 3.0,
    )

    assert delta_prime == pytest.approx(0.01)


def test_exchange_closure_report_keeps_nonzero_gamma_unready_without_compensating_sector():
    report = exchange_closure_report(gamma_enabled=True, compensating_sector_declared=False)

    assert report["energy_momentum_exchange_closed"] is False
    assert report["physical_exchange_ready"] is False
    assert exchange_closure_report(gamma_enabled=False, compensating_sector_declared=False)[
        "energy_momentum_exchange_closed"
    ] is True
