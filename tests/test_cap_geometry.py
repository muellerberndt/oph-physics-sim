import math

import numpy as np

from oph_fpe.bulk.cap_geometry import RoundCap, cap_weights, lambda_cap, pullback_field


def test_lambda_cap_identity_and_inverse():
    points = _sphere_points(200)
    cap = RoundCap(axis=np.array([0.0, 0.0, 1.0]), theta0=0.8, tangent=np.array([1.0, 0.0, 0.0]))

    identity = lambda_cap(points, cap, 0.0)
    moved = lambda_cap(points, cap, 0.35)
    restored = lambda_cap(moved, cap, -0.35)

    assert np.max(np.linalg.norm(identity - points, axis=1)) < 1e-12
    assert np.max(np.linalg.norm(restored - points, axis=1)) < 1e-10


def test_lambda_cap_preserves_cap_membership():
    points = _sphere_points(500)
    cap = RoundCap(axis=np.array([0.0, 0.0, 1.0]), theta0=0.9, tangent=np.array([1.0, 0.0, 0.0]))
    before = cap_weights(points, cap, soft=False)
    after_points = lambda_cap(points, cap, 0.7)
    after = cap_weights(after_points, cap, soft=False)

    assert np.array_equal(before, after)


def test_pullback_field_identity():
    points = _sphere_points(300)
    values = points[:, 2] + 0.25 * points[:, 0]
    cap = RoundCap(axis=np.array([0.0, 0.0, 1.0]), theta0=0.75, tangent=np.array([1.0, 0.0, 0.0]))

    pulled = pullback_field(points, values, cap, 0.0, k=1)

    assert np.max(np.abs(pulled - values)) < 1e-12


def test_boundary_derivative_matches_exp_minus_s():
    s = 0.4
    a = math.tanh(s / 2.0)
    eps = 1e-5
    derivative = abs(((1.0 - eps + a) / (a * (1.0 - eps) + 1.0) - 1.0) / (-eps))

    assert abs(derivative - math.exp(-s)) < 1e-5


def _sphere_points(count: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(count, 3))
    return values / np.linalg.norm(values, axis=1, keepdims=True)
