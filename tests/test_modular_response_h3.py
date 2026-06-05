import numpy as np

from oph_fpe.bulk.cap_geometry import sample_caps
from oph_fpe.bulk.h3_response_fit import modular_response_h3_report, synthetic_h3_modular_kernel
from oph_fpe.bulk.modular_response_kernel import kernel_json_summary, modular_response_kernel
from oph_fpe.cache.geometry_cache import GeometryCache
from oph_fpe.core.graph import fibonacci_sphere_points


def test_geometry_cache_reuses_cap_transport_map():
    points = fibonacci_sphere_points(256)
    caps = sample_caps(points, count=2, theta_values=[0.55], seed=1)
    cache = GeometryCache(points)

    first_indices, first_weights = cache.cap_transport_map(caps[0], 0.25, k=1, cap_id=0)
    second_indices, second_weights = cache.cap_transport_map(caps[0], 0.25, k=1, cap_id=0)

    assert first_indices.shape == (256, 1)
    assert np.array_equal(first_indices, second_indices)
    assert np.array_equal(first_weights, second_weights)
    assert cache.report()["transport_map_count"] == 1


def test_modular_response_kernel_shape_and_summary():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=3, theta_values=[0.55, 0.75], seed=2)
    raw_fields = {
        "record_signature": np.sin(points[:, 0] * 4.0),
        "stable_count": np.cos(points[:, 1] * 5.0),
        "s3_class_density": np.sin(points[:, 2] * 7.0),
    }
    observer_views = [
        {
            "view_type": "patch_observer",
            "observer_id": index,
            "support_nodes": list(range(index, min(index + 16, points.shape[0]))),
        }
        for index in range(0, 128, 16)
    ]

    kernel = modular_response_kernel(
        points,
        caps,
        raw_fields,
        observer_views,
        times=[0.05, 0.1],
        field_names=["record_signature", "stable_count"],
        geometry_cache=GeometryCache(points),
    )
    summary = kernel_json_summary(kernel)

    assert kernel["matrix"].shape == (len(observer_views), 3 * 2 * 2)
    assert kernel["s2_boundary_control"].shape == kernel["matrix"].shape
    assert kernel["shuffled_control"].shape == kernel["matrix"].shape
    assert kernel["no_modular_flow_control"].shape == kernel["matrix"].shape
    assert summary["feature_count"] == 12
    assert summary["geometry_cache"]["transport_map_count"] == 6
    assert 0.0 <= kernel["response_summary"]["min"] <= kernel["response_summary"]["max"] <= 1.0


def test_synthetic_modular_response_h3_receipt_can_pass_controls():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=10, theta_values=[0.55, 0.75, 1.0], seed=3)
    kernel = synthetic_h3_modular_kernel(
        caps,
        observer_count=18,
        times=[0.05, 0.1],
        field_names=["record_signature", "stable_count"],
        seed=4,
        softness=0.2,
    )

    report = modular_response_h3_report(
        kernel,
        caps,
        candidate_count=4096,
        candidate_radius=1.2,
        softness=0.2,
        seed=4,
        pass_ratio=0.95,
        min_observers=8,
        min_features=12,
    )

    assert report["mode"] == "modular_response_kernel_to_h3_fit"
    assert report["MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT"] is True
    assert report["claim_level"] == "demo"
    assert report["physical_claim"] is False


def test_object_transition_kernel_is_signed_and_writes_controls():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=4, theta_values=[0.55, 0.75], seed=5)
    raw_fields = {
        "record_signature": np.arange(points.shape[0]) % 17,
        "stable_count": (np.arange(points.shape[0]) % 9) + 1,
        "committed_mask": (np.arange(points.shape[0]) % 3 == 0).astype(float),
        "repair_load": np.abs(points[:, 0]),
        "s3_sector_class": np.arange(points.shape[0]) % 6,
    }
    observer_views = [
        {
            "view_type": "patch_observer",
            "observer_id": index,
            "axis": [float(value) for value in points[index]],
            "support_nodes": list(range(index, min(index + 24, points.shape[0]))),
        }
        for index in range(0, 192, 24)
    ]

    kernel = modular_response_kernel(
        points,
        caps,
        raw_fields,
        observer_views,
        times=[0.125, 0.25],
        observable_mode="object_transition",
        transition_observables=["checkpoint_class", "record_family", "s3_sector_class"],
        transform="signed_zscore",
        geometry_cache=GeometryCache(points),
        wrong_scales=[1.0, np.pi],
    )

    assert kernel["observable_mode"] == "object_transition"
    assert kernel["matrix"].shape[0] == len(observer_views)
    assert kernel["matrix"].shape == kernel["no_modular_flow_control"].shape
    assert kernel["matrix"].shape == kernel["s2_boundary_control"].shape
    assert "1x" in kernel["wrong_scale_controls"]
    assert "pi" in kernel["wrong_scale_controls"]
    assert abs(float(np.mean(kernel["matrix"]))) < 1e-9
    assert kernel["response_summary"]["std"] > 0.1


def test_perturb_resettle_transition_kernel_uses_graph_state():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=3, theta_values=[0.55, 0.75], seed=8)
    left = np.arange(0, 256, dtype=np.int64)
    right = (left + 7) % points.shape[0]
    port_left = (np.arange(left.size) % 6).astype(np.int16)
    port_right = ((np.arange(left.size) + 1) % 6).astype(np.int16)
    raw_fields = {
        "record_signature": np.arange(points.shape[0]) % 19,
        "stable_count": (np.arange(points.shape[0]) % 11) + 1,
        "committed_mask": (np.arange(points.shape[0]) % 2 == 0).astype(float),
        "repair_load": np.abs(points[:, 1]),
        "cumulative_repair_load": np.abs(points[:, 2]),
        "s3_sector_class": np.arange(points.shape[0]) % 6,
    }
    observer_views = [
        {
            "view_type": "patch_observer",
            "observer_id": index,
            "axis": [float(value) for value in points[index]],
            "support_nodes": list(range(index, min(index + 24, points.shape[0]))),
        }
        for index in range(0, 192, 24)
    ]

    kernel = modular_response_kernel(
        points,
        caps,
        raw_fields,
        observer_views,
        times=[0.125, 0.25],
        observable_mode="perturb_resettle_transition",
        transition_observables=["checkpoint_class", "record_family", "repair_load_bucket"],
        transform="signed_zscore",
        graph_state={
            "left": left,
            "right": right,
            "port_left": port_left,
            "port_right": port_right,
            "group_order": 6,
            "patch_count": points.shape[0],
        },
        perturb_strength=1.0,
        repair_steps=2,
        repairs_per_step=32,
        perturb_seed=9,
        wrong_scales=[1.0, np.pi],
    )

    assert kernel["observable_mode"] == "perturb_resettle_transition"
    assert kernel["matrix"].shape[0] == len(observer_views)
    assert kernel["matrix"].shape == kernel["no_modular_flow_control"].shape
    assert "perturb_resettle_report" in kernel
    assert kernel["raw_response_summary"]["std"] > 0.0
    assert kernel["response_summary"]["std"] > 0.0


def test_synthetic_modular_response_joint_h3_receipt_can_pass_controls():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=12, theta_values=[0.55, 0.75, 1.0], seed=6)
    kernel = synthetic_h3_modular_kernel(
        caps,
        observer_count=24,
        times=[0.05, 0.1],
        field_names=["record_signature", "stable_count"],
        seed=7,
        softness=0.2,
    )

    report = modular_response_h3_report(
        kernel,
        caps,
        candidate_count=4096,
        candidate_radius=1.2,
        softness=0.2,
        seed=7,
        pass_ratio=0.95,
        min_observers=8,
        min_features=12,
        fit_mode="joint_global",
        anchor_weight=0.0,
        max_iterations=3,
    )

    assert report["fit_mode"] == "joint_global"
    assert report["MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT"] is True
    assert report["h3_fit"]["heldout_normalized_rmse"] < report["s2_boundary_control"]["heldout_normalized_rmse"]
