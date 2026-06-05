import numpy as np

from oph_fpe.cosmology.galaxy_proxy import btfr_summary, effective_a0, galaxy_proxy_receipt, rar_curve
from oph_fpe.cosmology.proxy_pipeline import cosmo_proxy_receipt


def test_galaxy_proxy_reports_rar_btfr_surfaces():
    g_b = np.logspace(-12, -9, 12)
    g_obs = rar_curve(g_b, lambda_collar=1.4)
    mass = np.array([1e8, 2e9, 5e10, 2e11])
    velocity = (mass / 50.0) ** 0.25

    report = galaxy_proxy_receipt(
        g_baryon=g_b,
        g_observed=g_obs,
        baryonic_mass=mass,
        flat_velocity=velocity,
        lambda_collar=1.4,
    )

    assert report["GALAXY_PROXY_RECEIPT"] is True
    assert report["claim_level"] == "proxy"
    assert report["physical_claim"] is False
    assert report["lambda_collar_estimate"]["usable"] is True
    assert abs(report["a0_eff"] - effective_a0(lambda_collar=1.4)) < 1e-24
    assert report["btfr"]["usable"] is True


def test_btfr_summary_recovers_slope_four_for_v4_mock():
    velocity = np.array([50.0, 100.0, 200.0, 300.0])
    mass = 12.0 * velocity**4

    report = btfr_summary(mass, velocity)

    assert report["usable"] is True
    assert abs(report["slope_logM_vs_logV"] - 4.0) < 1e-10


def test_cosmo_proxy_selects_best_control_separated_field():
    cl_report = {
        "ell_max": 4,
        "estimator": "spherical_harmonic",
        "point_count": 128,
        "gate_report": {"allowed": True},
        "fields": {
            "weak": {
                "spectrum": [{"ell": 2, "D_ell": 1.0}, {"ell": 3, "D_ell": 0.5}],
                "control_comparison": {"min_relative_l2_delta": 0.1},
            },
            "strong": {
                "spectrum": [{"ell": 2, "D_ell": 1.0}, {"ell": 3, "D_ell": 0.2}],
                "control_comparison": {"min_relative_l2_delta": 2.0},
            },
        },
    }

    report = cosmo_proxy_receipt(cl_report)

    assert report["mode"] == "OPH_COSMO_PROXY_V0"
    assert report["claim_level"] == "proxy"
    assert report["physical_claim"] is False
    assert report["best_field"] == "strong"
