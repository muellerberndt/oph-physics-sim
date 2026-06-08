from __future__ import annotations

import numpy as np

from oph_fpe.cmb_fossil import (
    apply_low_l_repair_suppression,
    apply_parity_term,
    cl_oph_screen,
    cmb_fossil_bridge_report,
    primordial_power,
    screen_cl_to_primordial_modulation,
)


def test_screen_covariance_has_positive_decaying_shape():
    ell = np.arange(2, 128)
    cl = cl_oph_screen(ell, A=2.0, eta=0.02, ell_cap=1000.0)

    assert cl.shape == ell.shape
    assert np.all(cl > 0.0)
    assert cl[0] > cl[-1]


def test_low_l_repair_suppression_and_parity_are_finite():
    ell = np.arange(2, 32)
    cl = cl_oph_screen(ell)
    suppressed = apply_low_l_repair_suppression(cl, ell, q_ir=0.2)
    parity = apply_parity_term(cl, ell, eps_p=0.05)

    assert np.all(np.isfinite(suppressed))
    assert np.all(suppressed <= cl)
    assert np.all(np.isfinite(parity))
    assert not np.allclose(parity, cl)


def test_screen_cl_to_primordial_modulation_interpolates_ratio():
    ell = np.arange(2, 64)
    base = np.ones_like(ell, dtype=float)
    oph = np.linspace(0.8, 1.2, ell.size)
    k = np.asarray([2.0 / 10_000.0, 20.0 / 10_000.0, 60.0 / 10_000.0])

    modulation = screen_cl_to_primordial_modulation(
        k,
        D_star_mpc=10_000.0,
        ell_grid=ell,
        cl_oph=oph,
        cl_base=base,
    )
    power = primordial_power(k, A_s=2.1e-9, n_s=0.965, k0=0.05, F_oph=modulation)

    assert modulation[0] == oph[0]
    assert modulation[-1] > modulation[0]
    assert np.all(power > 0.0)


def test_cmb_fossil_bridge_report_is_claim_bounded():
    report = cmb_fossil_bridge_report({"eta": 0.035}, {"shape_correlation": 0.5})

    assert report["receipt"] == "OPH_CMB_FOSSIL_BRIDGE_DIAGNOSTIC"
    assert report["physical_cmb_prediction"] is False
    assert report["bulk_required"] is False
    assert report["parameters"]["eta"] == 0.035
