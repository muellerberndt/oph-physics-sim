import numpy as np

from oph_fpe.bulk.cap_geometry import RoundCap, sample_caps
from oph_fpe.bulk.cap_normals import boundary_null_directions, cap_boundary_residual, cap_normal, cap_normal_report, minkowski_dot
from oph_fpe.bulk.conformal_spatial_chart import conformal_h3_spatial_chart_report
from oph_fpe.bulk.h3_chart import h3_distance_matrix, h3_origin, h3_point_from_tangent, h3_tangent_from_point, random_h3_points
from oph_fpe.core.graph import fibonacci_sphere_points


def test_cap_normal_is_de_sitter_unit_and_boundary_vanishes():
    theta = 0.8
    cap = RoundCap(axis=np.array([0.0, 0.0, 1.0]), theta0=theta, tangent=np.array([1.0, 0.0, 0.0]))
    normal = cap_normal(cap)
    boundary_point = np.array([[np.sin(theta), 0.0, np.cos(theta)]])

    assert abs(float(minkowski_dot(normal, normal)) - 1.0) < 1e-12
    assert abs(float(cap_boundary_residual(boundary_point, cap)[0])) < 1e-12
    assert abs(float(minkowski_dot(boundary_null_directions(boundary_point)[0], normal))) < 1e-12


def test_h3_points_have_negative_unit_norm_and_distance_zero_diagonal():
    origin = h3_origin()
    point = h3_point_from_tangent(np.array([0.2, 0.3, 0.4]))
    points = np.vstack([origin, point, *random_h3_points(4, seed=3, radius=0.7)])
    norms = minkowski_dot(points, points)
    distances = h3_distance_matrix(points)

    assert np.max(np.abs(norms + 1.0)) < 1e-12
    assert np.allclose(np.diag(distances), 0.0)
    assert np.all(distances >= 0.0)


def test_h3_tangent_roundtrip_preserves_point():
    tangent = np.array([0.2, -0.15, 0.35])
    point = h3_point_from_tangent(tangent)
    restored = h3_point_from_tangent(h3_tangent_from_point(point))

    assert np.max(np.abs(restored - point)) < 1e-12


def test_conformal_h3_spatial_chart_receipt_from_caps():
    points = fibonacci_sphere_points(128)
    caps = sample_caps(points, count=8, theta_values=[0.55, 0.75, 1.0], seed=5)
    report = conformal_h3_spatial_chart_report(caps)
    normal_report = cap_normal_report(caps)

    assert report["conformal_h3_spatial_chart_receipt"] is True
    assert report["record_populated_h3_receipt"] is False
    assert report["h3_chart_report"]["spatial_dimension"] == 3
    assert normal_report["unit_normal_receipt"] is True
