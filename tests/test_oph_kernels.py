from __future__ import annotations

import math

import numpy as np

from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.oph_constants import OPHConstants
from oph_fpe.cosmology.oph_kernels import (
    B_A_minimal_one_pole,
    B_A_z6_poisson_five_of_seven,
    W_a,
    W_k,
    apply_projected_wl_selector,
    compressed_projection_fraction,
    normalized_projection_average,
    projected_amplitude,
    projected_amplitude_ratio,
)


def test_oph_constants_five_of_seven_values():
    constants = OPHConstants()

    assert constants.N_CRC > 1.0e122
    assert constants.N_patch_bare_ratio == constants.N_CRC / math.pi
    assert constants.Lambda_lP2 == 3.0 * math.pi / constants.N_CRC
    assert constants.P_cell_count_for_N_CRC == 4.0 * constants.N_CRC / constants.P
    assert constants.z6_normalized_trace_mean == P_STAR / 24.0
    assert constants.z6_reciprocal_trace == 24.0 / P_STAR
    assert constants.lambda_collar_exact_gate == "UNIFORM_PRODUCT_THICKENING_EXACT"
    assert constants.lambda_collar == math.exp(-P_STAR / 24.0)
    assert constants.lambda_collar == constants.lambda_collar_exact_uniform_product_thickening
    assert constants.finite_thickness_jensen_band == [constants.lambda_collar, 1.0]
    assert constants.pi_wl == 5.0 / 7.0
    assert constants.epsilon_A_wl == constants.pi_wl * (1.0 - constants.lambda_collar)
    assert constants.R_wl == 1.0 - constants.epsilon_A_wl
    assert apply_projected_wl_selector(constants.S8_oph_compressed, constants) == constants.S8_projected_wl
    assert abs(constants.S8_projected_wl - 0.7900242005) < 1.0e-9


def test_z6_poisson_kernel_windows_and_saturation():
    constants = OPHConstants()

    assert W_k(0.0, 0.1) == 0.0
    assert np.all(W_a(np.array([0.5, 1.0]), 1.0, 1.0, 67.4, 0.315, 9.2e-5, 0.684) > 0.0)
    saturated = B_A_z6_poisson_five_of_seven(
        1.0e9,
        1.0,
        kA_hMpc=1.0e-9,
        tau_rec_Gyr=1.0e-9,
        constants=constants,
    )

    assert abs(float(saturated) - constants.R_wl) < 1.0e-9


def test_minimal_one_pole_and_projection_theorem_helpers():
    constants = OPHConstants()
    pi_wl = compressed_projection_fraction(0.790, 0.828924043, constants)
    reconstructed = projected_amplitude(0.828924043, pi_wl, constants)
    saturated = B_A_minimal_one_pole(
        1.0e9,
        1.0,
        kA_hMpc=1.0e-9,
        tau_rec_Gyr=1.0e-9,
        constants=constants,
    )

    assert abs(pi_wl - 0.7147300876158466) < 1.0e-12
    assert abs(reconstructed - 0.790) < 1.0e-14
    assert abs(projected_amplitude_ratio(pi_wl, constants) - 0.790 / 0.828924043) < 1.0e-14
    assert abs(float(saturated) - constants.lambda_collar) < 1.0e-9
    assert normalized_projection_average(np.array([0.0, 0.5, 1.0]), np.array([1.0, 2.0, 1.0])) == 0.5
