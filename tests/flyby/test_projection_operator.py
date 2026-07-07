import numpy as np

from oph_fpe.flyby import projection_artifact, projection_operator


def test_projection_operator_matches_closed_form() -> None:
    Jx = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
    Jn = np.array([[0.2], [0.5], [-0.1]])
    C = np.diag([1.0, 2.0, 1.5])
    expected = -np.linalg.inv(Jx.T @ np.linalg.inv(C) @ Jx) @ Jx.T @ np.linalg.inv(C) @ Jn
    assert np.allclose(projection_operator(Jx, Jn, C), expected)


def test_projection_artifact_reports_readout() -> None:
    Jx = np.array([[1.0], [2.0], [3.0]])
    Jn = np.array([[0.0], [1.0], [0.5]])
    C = np.eye(3)
    delta_n = np.array([0.25])
    L = np.array([10.0])
    artifact = projection_artifact(Jx, Jn, C, delta_n, L)
    assert artifact["P_x_from_n"].shape == (1, 1)
    assert np.isclose(artifact["A_proj_mm_s"], float(L @ artifact["delta_x"]))
