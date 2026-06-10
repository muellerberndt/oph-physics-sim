import numpy as np

from oph_fpe.microphysics.three_way_vertex import (
    scatter_vertex,
    scattering_receipt,
    three_way_scattering_matrix,
    vertex_power,
)


def test_three_way_matrix_basic_properties():
    matrix = three_way_scattering_matrix()

    assert np.allclose(matrix, matrix.T)
    assert np.allclose(matrix @ matrix, np.eye(3))
    assert np.allclose(matrix.T @ matrix, np.eye(3))


def test_single_channel_scattering():
    scattered = scatter_vertex(np.array([1.0, 0.0, 0.0]))

    assert np.allclose(scattered, [-1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])
    assert abs(vertex_power(scattered) - 1.0) < 1.0e-12


def test_receipt_passes_without_physical_claims():
    report = scattering_receipt()

    assert report["passed"] is True
    assert report["neutral_oph_bulk_claim"] is False
    assert report["physical_cmb_prediction"] is False
