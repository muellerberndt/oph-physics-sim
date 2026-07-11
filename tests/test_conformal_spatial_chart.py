import numpy as np

from oph_fpe.bulk.cap_geometry import RoundCap, sample_caps
from oph_fpe.bulk.cap_normals import boundary_null_directions, cap_boundary_residual, cap_normal, cap_normal_report, minkowski_dot
from oph_fpe.bulk.conformal_spatial_chart import (
    conformal_h3_spatial_chart_report,
    paper_theorem_3d_bulk_chart_report,
    write_paper_chart_receipts,
)
from oph_fpe.bulk.h3_chart import h3_distance_matrix, h3_origin, h3_point_from_tangent, h3_tangent_from_point, random_h3_points
from oph_fpe.bulk.h3_chart import h3_chart_report
from oph_fpe.bulk.lorentz_algebra import lorentz_algebra_report
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
    assert report["lorentz_algebra_receipt"] is True
    assert report["record_populated_h3_receipt"] is False
    assert report["h3_chart_report"]["spatial_dimension"] == 3
    assert report["h3_chart_report"]["spatial_dimension_derivation"] == "dim SO+(3,1)-dim SO(3)=6-3=3"
    assert report["spatial_dimension_derivation"] == "dim SO+(3,1)-dim SO(3)=6-3=3"
    assert report["lorentz_algebra_report"]["h3_spatial_dimension_from_boost_orbit"] == 3
    assert normal_report["unit_normal_receipt"] is True


def test_h3_chart_rejects_empty_cap_family() -> None:
    report = h3_chart_report([])

    assert report["conformal_h3_spatial_chart_receipt"] is False
    assert report["blockers"] == ["missing_caps"]


def test_paper_theorem_3d_bulk_chart_receipt_is_separate_from_neutral_bulk():
    points = fibonacci_sphere_points(128)
    caps = sample_caps(points, count=8, theta_values=[0.55, 0.75, 1.0], seed=51)
    chart = conformal_h3_spatial_chart_report(caps)
    transition = {
        "primary_source": "explicit_visualization_assumption",
        "scope": "visualization_only",
        "SIMULATION_ASSUMED_BW_2PI_GEOMETRIC_BRANCH_RECEIPT": True,
        "computed_theorem_receipts_unchanged": True,
    }
    objects = {
        "observer_chart_object_h3_receipt": True,
        "localized_object_precursor_receipt": True,
        "localized_nonboundary_bulk_population_receipt": False,
    }
    neutral = {"bulk_3d_established": False}

    report = paper_theorem_3d_bulk_chart_report(chart, transition, objects, neutral)

    assert report["PAPER_THEOREM_3D_BULK_CHART_RECEIPT"] is False
    assert report["SIMULATION_ASSUMED_BW_2PI_GEOMETRIC_BRANCH_RECEIPT"] is True
    assert report["SIMULATION_ASSUMED_3D_H3_CHART_RECEIPT"] is True
    assert report["paper_theorem_object_populated_chart_precursor_receipt"] is False
    assert report["paper_theorem_neutral_populated_bulk_receipt"] is False
    assert report["h3_spatial_dimension_from_boost_orbit"] == 3
    assert report["spatial_dimension_derivation"] == "dim SO+(3,1)-dim SO(3)=6-3=3"
    assert report["finite_point_cloud_dimension_estimator_used"] is False


def test_paper_theorem_object_precursor_accepts_control_separation_without_strict_bulk():
    points = fibonacci_sphere_points(128)
    caps = sample_caps(points, count=8, theta_values=[0.55, 0.75, 1.0], seed=52)
    chart = conformal_h3_spatial_chart_report(caps)
    transition = {
        "primary_source": "explicit_visualization_assumption",
        "scope": "visualization_only",
        "SIMULATION_ASSUMED_BW_2PI_GEOMETRIC_BRANCH_RECEIPT": True,
        "computed_theorem_receipts_unchanged": True,
    }
    objects = {
        "observer_chart_object_h3_receipt": False,
        "localized_object_precursor_receipt": True,
        "modular_response_h3_control_separation_receipt": True,
        "localized_nonboundary_bulk_population_receipt": False,
    }
    neutral = {"bulk_3d_established": False}

    report = paper_theorem_3d_bulk_chart_report(chart, transition, objects, neutral)

    assert report["paper_theorem_3d_bulk_chart_receipt"] is False
    assert report["SIMULATION_ASSUMED_3D_H3_CHART_RECEIPT"] is True
    assert report["paper_theorem_object_populated_chart_precursor_receipt"] is False
    assert report["paper_theorem_neutral_populated_bulk_receipt"] is False
    assert report["observer_object_precursor_components"]["strict_object_h3_receipt"] is False
    assert report["observer_object_precursor_components"]["h3_control_separation_receipt"] is True


def test_paper_theorem_chart_accepts_finite_lorentz_modular_clock():
    points = fibonacci_sphere_points(128)
    caps = sample_caps(points, count=8, theta_values=[0.55, 0.75, 1.0], seed=53)
    chart = conformal_h3_spatial_chart_report(caps)
    objects = {
        "observer_chart_object_h3_receipt": True,
        "localized_nonboundary_object_precursor_receipt": True,
    }
    state_bw = {
        "ENDOGENOUS_MODULAR_GENERATOR_RECEIPT": True,
        "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT": True,
        "inferred_modular_clock_fit": {"receipt": True, "nearest_known_scale": "2pi"},
    }

    report = paper_theorem_3d_bulk_chart_report(
        chart,
        {},
        objects,
        {"bulk_3d_established": False},
        state_bw,
    )

    assert report["paper_theorem_3d_bulk_chart_receipt"] is True
    assert report["declared_bw_2pi_cap_flow_receipt"] is False
    assert report["finite_lorentz_modular_clock_receipt"] is True
    assert report["bw_2pi_cap_flow_source"] == "finite_endogenous_l2_l3_modular_clock"
    assert report["paper_theorem_object_populated_chart_precursor_receipt"] is True


def test_lorentz_algebra_report_verifies_so_3_1_relations():
    report = lorentz_algebra_report()

    assert report["lorentz_algebra_receipt"] is True
    assert report["group"] == "SO+(3,1)"
    assert report["spatial_homogeneous_space"] == "H3 = SO+(3,1)/SO(3)"
    assert report["h3_spatial_dimension_from_boost_orbit"] == 3
    assert report["stabilizer_dimension"] == 3
    assert report["spatial_dimension_derivation"] == "dim SO+(3,1)-dim SO(3)=6-3=3"
    assert report["max_commutator_error"] < 1e-12
    assert report["max_null_cone_preservation_error"] < 1e-12


def test_write_paper_chart_receipts_writes_run_folder_reports(tmp_path):
    report = write_paper_chart_receipts(tmp_path, point_count=128, cap_count=8, seed=4)

    assert report["paper_theorem_3d_bulk_chart_receipt"] is False
    assert report["simulation_assumed_bw_2pi_geometric_branch_receipt"] is True
    assert report["simulation_assumed_3d_h3_chart_receipt"] is True
    assert (tmp_path / "conformal_h3_spatial_chart_report.json").exists()
    assert (tmp_path / "transition_selection_report.json").exists()
    assert (tmp_path / "paper_3d_bulk_chart_report.json").exists()
    assert (tmp_path / "emergence_status_report.json").exists()


def test_declared_two_pi_string_cannot_raise_assumption_or_computed_receipt() -> None:
    points = fibonacci_sphere_points(128)
    chart = conformal_h3_spatial_chart_report(
        sample_caps(points, count=8, theta_values=[0.55, 0.75], seed=54)
    )

    report = paper_theorem_3d_bulk_chart_report(
        chart,
        {
            "scope": "visualization_only",
            "SIMULATION_ASSUMED_BW_2PI_GEOMETRIC_BRANCH_RECEIPT": "true",
            "computed_theorem_receipts_unchanged": True,
        },
    )

    assert report["PAPER_THEOREM_3D_BULK_CHART_RECEIPT"] is False
    assert report["SIMULATION_ASSUMED_BW_2PI_GEOMETRIC_BRANCH_RECEIPT"] is False
