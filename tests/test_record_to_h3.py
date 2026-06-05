import numpy as np

from oph_fpe.bulk.cap_geometry import sample_caps
from oph_fpe.bulk.cap_normals import cap_normals
from oph_fpe.bulk.h3_chart import h3_halfspace_profile, random_h3_points
from oph_fpe.bulk.record_to_h3 import (
    defect_timeline_to_h3_report,
    fit_response_profiles_to_h3,
    observer_chart_object_population_report,
    record_cap_response_matrix,
    record_populated_h3_report,
    support_profiles_to_h3_report,
)
from oph_fpe.bulk.h3_chart import h3_point_from_tangent
from oph_fpe.core.graph import fibonacci_sphere_points


def test_fit_response_profiles_to_h3_recovers_synthetic_profiles():
    points = fibonacci_sphere_points(256)
    caps = sample_caps(points, count=10, theta_values=[0.55, 0.75, 1.0], seed=7)
    h3_points = random_h3_points(12, seed=8, radius=1.2)
    response = h3_halfspace_profile(h3_points, cap_normals(caps), softness=0.2)

    report = fit_response_profiles_to_h3(response, caps, candidate_count=4096, candidate_radius=1.2, softness=0.2, seed=8)

    assert report["median_residual"] < 0.08
    assert report["candidate_count"] == 4096
    assert report["local_refinement"]["enabled"] is True


def test_record_populated_h3_report_emits_controlled_nonclaim():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=6, theta_values=[0.55, 0.75], seed=9)
    raw_fields = {
        "record_signature": np.arange(points.shape[0]) % 17,
        "stable_count": np.ones(points.shape[0]) * 8,
        "repair_load": np.linspace(0.0, 1.0, points.shape[0]),
        "cumulative_repair_load": np.linspace(1.0, 0.0, points.shape[0]),
        "s3_class_density": np.sin(np.linspace(0.0, 4.0, points.shape[0])),
        "s3_sector_class": np.arange(points.shape[0]) % 3,
    }
    observer_views = [
        {
            "view_type": "patch_observer",
            "observer_id": index,
            "axis": [float(value) for value in points[index]],
            "support_nodes": list(range(index, min(index + 16, points.shape[0]))),
        }
        for index in range(0, 128, 16)
    ]

    report = record_populated_h3_report(
        points,
        caps,
        raw_fields,
        observer_views,
        seed=10,
        candidate_count=512,
    )

    assert report["mode"] == "record_populated_h3_fit"
    assert report["observer_count"] == len(observer_views)
    assert "h3_fit" in report
    assert "s2_boundary_control" in report
    assert "shuffled_cap_response_control" in report
    assert "physical CMB" in report["claim_boundary"]


def test_cap_transport_response_mode_uses_bw_transport():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=4, theta_values=[0.55, 0.75], seed=11)
    raw_fields = {
        "record_signature": np.arange(points.shape[0]) % 19,
        "stable_count": np.maximum(1, np.arange(points.shape[0]) % 11),
        "repair_load": np.sin(points[:, 0] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 1] * 5.0),
        "s3_class_density": np.sin(points[:, 2] * 7.0),
        "s3_sector_class": np.arange(points.shape[0]) % 3,
    }
    observer_views = [
        {
            "view_type": "patch_observer",
            "observer_id": index,
            "axis": [float(value) for value in points[index]],
            "support_nodes": list(range(index, min(index + 12, points.shape[0]))),
        }
        for index in range(0, 96, 12)
    ]

    response, patch_views, meta = record_cap_response_matrix(
        points,
        caps,
        raw_fields,
        observer_views,
        response_mode="cap_transport_similarity",
        transport_time=0.1,
    )

    assert response.shape == (len(observer_views), len(caps))
    assert len(patch_views) == len(observer_views)
    assert meta["response_mode"] == "cap_transport_similarity"
    assert meta["transport_scale"] > 6.0
    assert np.all(response >= 0.0)
    assert np.all(response <= 1.0)


def test_support_profiles_to_h3_report_handles_record_supports():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=6, theta_values=[0.55, 0.75], seed=12)
    supports = [
        {"object_id": "obj_a", "support_nodes": list(range(0, 24))},
        {"object_id": "obj_b", "support_nodes": list(range(120, 152))},
        {"object_id": "obj_c", "support_nodes": list(range(320, 360))},
    ]

    report = support_profiles_to_h3_report(points, caps, supports, candidate_count=512, seed=13)

    assert report["mode"] == "support_profile_h3_fit"
    assert report["label"] == "record_family"
    assert report["support_count"] == 3
    assert report["support_size_summary"]["max"] >= 24
    assert "h3_fit" in report
    assert "shuffled_cap_response_control" in report


def test_observer_chart_object_population_uses_modular_response_chart():
    h3_points = []
    observer_views = []
    object_rows = []
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    for cluster in range(3):
        support_start = cluster * 100
        signature = cluster + 4
        object_rows.append(
            {
                "object_id": f"obj_{cluster}",
                "support_nodes": list(range(support_start, support_start + 12)),
                "record_signature": signature,
            }
        )
        for local in range(4):
            observer_id = cluster * 4 + local
            tangent = np.array([float(cluster) * 1.4 + 0.03 * local, 0.02 * local, 0.0])
            h3_points.append(h3_point_from_tangent(tangent))
            observer_views.append(
                {
                    "view_type": "patch_observer",
                    "observer_id": observer_id,
                    "axis": [float(value) for value in axes[cluster]],
                    "support_nodes": list(range(support_start + local, support_start + local + 8)),
                    "object_packet_histogram": {str(signature): 1.0},
                }
            )
    h3_report = {
        "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": True,
        "observer_ids": list(range(12)),
        "h3_fit": {"fitted_h3_points": [[float(value) for value in row] for row in h3_points]},
    }

    report = observer_chart_object_population_report(
        observer_views,
        object_rows,
        h3_report,
        seed=4,
        min_objects=3,
        min_observers_per_object=2,
        pass_ratio=0.95,
    )

    assert report["mode"] == "observer_chart_object_h3_population"
    assert report["object_count"] == 3
    assert report["observer_chart_object_h3_receipt"] is True
    assert report["median_h3_compactness_normalized"] < report["median_shuffled_h3_compactness_normalized"]
    assert report["sample_objects"][0]["h3_spatial_point"]


def test_defect_timeline_to_h3_report_emits_worldline_precursor_nonclaim():
    points = fibonacci_sphere_points(256)
    caps = sample_caps(points, count=6, theta_values=[0.55, 0.75], seed=14)
    timeline = {
        "worldlines": [
            {
                "worldline_id": "w0",
                "persistent": True,
                "events": [
                    {"cycle": 0, "class": "transposition", "support_nodes": list(range(0, 20)), "support_node_count": 20},
                    {"cycle": 1, "class": "transposition", "support_nodes": list(range(4, 24)), "support_node_count": 20},
                    {"cycle": 2, "class": "transposition", "support_nodes": list(range(8, 28)), "support_node_count": 20},
                ],
            }
        ]
    }

    report = defect_timeline_to_h3_report(points, caps, timeline, candidate_count=512, seed=15)

    assert report["mode"] == "defect_timeline_h3_worldline_fit"
    assert report["event_count"] == 3
    assert report["worldline_count"] == 1
    assert report["persistent_h3_worldline_count"] == 1
    assert report["particle_matter_receipt"] is False
    assert report["worldlines"][0]["events"][0]["h3_spatial_point"]


def test_defect_timeline_to_h3_can_use_transport_response_fields():
    points = fibonacci_sphere_points(256)
    caps = sample_caps(points, count=6, theta_values=[0.55, 0.75], seed=16)
    raw_fields = {
        "record_signature": np.arange(points.shape[0]) % 23,
        "stable_count": np.arange(points.shape[0]) % 11,
        "repair_load": np.sin(points[:, 0] * 5.0),
        "cumulative_repair_load": np.cos(points[:, 1] * 7.0),
        "local_mismatch_density": np.sin(points[:, 2] * 3.0),
        "modular_depth": np.linspace(0.0, 1.0, points.shape[0]),
        "s3_class_density": np.cos(points[:, 0] * 2.0),
        "s3_sector_class": np.arange(points.shape[0]) % 3,
    }
    timeline = {
        "worldlines": [
            {
                "worldline_id": "w0",
                "persistent": True,
                "events": [
                    {"cycle": 0, "class": "transposition", "support_nodes": list(range(0, 20)), "support_node_count": 20},
                    {"cycle": 1, "class": "transposition", "support_nodes": list(range(4, 24)), "support_node_count": 20},
                    {"cycle": 2, "class": "transposition", "support_nodes": list(range(8, 28)), "support_node_count": 20},
                ],
            }
        ]
    }

    report = defect_timeline_to_h3_report(
        points,
        caps,
        timeline,
        raw_fields=raw_fields,
        response_mode="support_transport_similarity",
        candidate_count=512,
        seed=17,
    )

    assert report["response_report"]["response_mode"] == "support_transport_similarity"
    assert report["response_report"]["field_names"]
    assert "lambda_C" in report["response_report"]["response_source"]
    assert report["local_refinement"]["enabled"] is True
    assert report["particle_matter_receipt"] is False
