import numpy as np

from oph_fpe.bulk.cap_geometry import sample_caps
from oph_fpe.bulk.h3_refit import (
    _h3_ensemble_summary_from_rows,
    load_modular_response_kernel_cache,
    write_h3_refit_ensemble_report,
    write_h3_refit_report,
    write_modular_response_kernel_cache,
)
from oph_fpe.bulk.h3_response_fit import (
    _assign_candidates,
    _channel_keys,
    _h3_profile_matrix,
    _h3_response_stage_gates,
    _select_fit_features,
    _strict_wrong_scale_feature_audit,
    modular_response_h3_report,
    synthetic_h3_modular_kernel,
)
from oph_fpe.bulk.modular_response_kernel import (
    _modular_selected_edges,
    _perturb_side,
    kernel_json_summary,
    modular_response_kernel,
)
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


def test_assign_candidates_matches_scalar_cost_formula():
    rng = np.random.default_rng(17)
    response = rng.normal(size=(7, 11))
    candidate_profiles = rng.normal(size=(13, 11))
    train_mask = np.array([True, False, True, True, False, True, True, False, True, True, False])
    offsets = rng.normal(scale=0.1, size=11)
    amplitudes = rng.normal(loc=1.0, scale=0.2, size=11)
    observer_axes = rng.normal(size=(7, 3))
    candidates = rng.normal(size=(13, 4))

    actual = _assign_candidates(
        response,
        candidate_profiles,
        train_mask,
        offsets,
        amplitudes,
        observer_axes,
        candidates,
        anchor_weight=0.15,
    )

    train_indices = np.flatnonzero(train_mask)
    predicted = offsets[None, :] + candidate_profiles * amplitudes[None, :]
    anchor_cost = 1.0 - np.clip(
        (observer_axes / np.maximum(np.linalg.norm(observer_axes, axis=1, keepdims=True), 1e-12))
        @ (
            candidates[:, 1:]
            / np.maximum(np.linalg.norm(candidates[:, 1:], axis=1, keepdims=True), 1e-12)
        ).T,
        -1.0,
        1.0,
    )
    expected = []
    for row_index in range(response.shape[0]):
        diff = predicted[:, train_indices] - response[row_index, train_indices][None, :]
        costs = np.mean(diff * diff, axis=1) + 0.15 * anchor_cost[row_index]
        expected.append(int(np.argmin(costs)))

    assert actual.tolist() == expected


def test_geometry_cache_persists_cap_transport_map(tmp_path):
    points = fibonacci_sphere_points(256)
    caps = sample_caps(points, count=2, theta_values=[0.55], seed=11)
    cache_dir = tmp_path / "geometry"
    cold_cache = GeometryCache(points, cache_dir=cache_dir)

    first_indices, first_weights = cold_cache.cap_transport_map(caps[0], 0.25, k=1, cap_id=0)
    cold_report = cold_cache.report()

    warm_cache = GeometryCache(points, cache_dir=cache_dir)
    second_indices, second_weights = warm_cache.cap_transport_map(caps[0], 0.25, k=1, cap_id=0)
    warm_report = warm_cache.report()

    assert np.array_equal(first_indices, second_indices)
    assert np.array_equal(first_weights, second_weights)
    assert cold_report["persistent_cache_disk_writes"] == 1
    assert warm_report["persistent_cache_disk_hits"] == 1
    assert warm_report["persistent_cache_disk_writes"] == 0
    assert list(cache_dir.rglob("transport_*.npz"))


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
    assert report["h3_chart_dimension_debug"]["mode"] == "h3_chart_dimension_debug"


def test_h3_channel_keys_can_tie_modular_time_by_cap():
    rows = [
        {
            "cap_index": 1,
            "time_index": 0,
            "observable": "record_signature",
            "feature_type": "class_distribution_delta",
            "target_class": 2,
        },
        {
            "cap_index": 1,
            "time_index": 1,
            "observable": "record_signature",
            "feature_type": "class_distribution_delta",
            "target_class": 2,
        },
        {
            "cap_index": 2,
            "time_index": 0,
            "observable": "record_signature",
            "feature_type": "class_distribution_delta",
            "target_class": 2,
        },
    ]

    default_keys = _channel_keys(rows)
    tied_by_cap = _channel_keys(rows, mode="cap_observable_class")
    tied_global = _channel_keys(rows, mode="observable_class")

    assert default_keys[0] != default_keys[1]
    assert tied_by_cap[0] == tied_by_cap[1]
    assert tied_by_cap[0] != tied_by_cap[2]
    assert tied_global[0] == tied_global[1] == tied_global[2]


def test_h3_modular_time_profile_is_time_dependent():
    points = fibonacci_sphere_points(256)
    caps = sample_caps(points, count=2, theta_values=[0.75], seed=31)
    kernel = synthetic_h3_modular_kernel(
        caps,
        observer_count=12,
        times=[0.0, 0.2],
        field_names=["record_signature"],
        seed=32,
    )
    h3_points = np.asarray(kernel["h3_source_points"], dtype=float)
    rows = list(kernel["feature_rows"])

    profile = _h3_profile_matrix(
        h3_points,
        [caps[int(row["cap_index"])] for row in rows],
        [{**row, "cap_index": index} for index, row in enumerate(rows)],
        softness=0.25,
        profile_mode="modular_time_delta",
        profile_time_scale=2.0 * np.pi,
    )

    assert profile.shape == kernel["matrix"].shape
    assert np.allclose(profile[:, 0], 0.0, atol=1e-12)
    assert float(np.std(profile[:, 1])) > 1e-4


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
        perturb_budget_mode="fixed_collar_fraction",
        fixed_perturb_fraction=0.25,
        perturb_selection_mode="lambda_displacement",
        repair_steps=2,
        repairs_per_step=32,
        perturb_seed=9,
        wrong_scales=[1.0, np.pi, 2.0 * np.pi],
    )

    assert kernel["observable_mode"] == "perturb_resettle_transition"
    assert kernel["matrix"].shape[0] == len(observer_views)
    assert kernel["matrix"].shape == kernel["no_modular_flow_control"].shape
    assert "perturb_resettle_report" in kernel
    assert kernel["perturb_resettle_report"]["perturb_selection_mode"] == "lambda_displacement"
    assert kernel["perturb_resettle_report"]["perturb_budget_mode"] == "fixed_collar_fraction"
    assert kernel["perturb_resettle_report"]["transition_readout_mode"] == "same_support"
    assert np.allclose(kernel["wrong_scale_controls"]["2pi"], kernel["matrix"])
    assert kernel["raw_response_summary"]["std"] > 0.0
    assert kernel["response_summary"]["std"] > 0.0


def test_collar_generator_selection_targets_cap_collar_and_oriented_side():
    points = fibonacci_sphere_points(1024)
    caps = sample_caps(points, count=1, theta_values=[0.75], seed=81)
    cap = caps[0]
    left = np.arange(0, 768, dtype=np.int64)
    right = (left + 17) % points.shape[0]
    all_edges = np.arange(left.size, dtype=np.int64)
    rng = np.random.default_rng(82)

    chosen = _modular_selected_edges(
        points,
        cap,
        left,
        right,
        all_edges,
        96,
        scale=2.0 * np.pi,
        time_value=0.125,
        rng=rng,
        mode="lambda_collar_generator",
    )

    mids = points[left] + points[right]
    mids = mids / np.maximum(np.linalg.norm(mids, axis=1, keepdims=True), 1e-12)
    threshold = np.cos(cap.theta0)
    all_boundary_distance = np.abs(mids @ cap.axis - threshold)
    selected_boundary_distance = all_boundary_distance[chosen]
    assert float(np.median(selected_boundary_distance)) < float(np.median(all_boundary_distance))

    side = _perturb_side(
        points,
        cap,
        left[chosen],
        right[chosen],
        scale=2.0 * np.pi,
        time_value=0.125,
        mode="lambda_collar_generator",
    )
    assert side.shape == chosen.shape
    assert side.dtype == np.bool_
    assert 0 < int(np.sum(side)) < side.size


def test_collar_operator_transition_preserves_lambda_flow_readout():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=3, theta_values=[0.55, 0.75], seed=18)
    raw_fields = {
        "record_signature": np.arange(points.shape[0]) % 23,
        "stable_count": (np.arange(points.shape[0]) % 11) + 1,
        "committed_mask": (np.arange(points.shape[0]) % 2 == 0).astype(float),
        "repair_load": np.abs(points[:, 0]),
        "cumulative_repair_load": np.abs(points[:, 1]),
        "s3_sector_class": np.arange(points.shape[0]) % 6,
    }
    observer_views = [
        {
            "view_type": "patch_observer",
            "observer_id": index,
            "axis": [float(value) for value in points[index]],
            "support_nodes": list(range(index, min(index + 32, points.shape[0]))),
        }
        for index in range(0, 256, 32)
    ]

    kernel = modular_response_kernel(
        points,
        caps,
        raw_fields,
        observer_views,
        times=[0.125, 0.25],
        observable_mode="collar_operator_transition",
        transition_observables=["checkpoint_class", "record_family", "s3_sector_class"],
        transition_feature_types=[
            "class_distribution_delta",
            "transition_matrix_delta",
            "change_probability_delta",
        ],
        transform="signed_robust_zscore",
        wrong_scales=[1.0, 2.0 * np.pi],
    )

    assert kernel["observable_mode"] == "collar_operator_transition"
    assert kernel["matrix"].shape[0] == len(observer_views)
    assert kernel["matrix"].shape == kernel["no_modular_flow_control"].shape
    assert kernel["matrix"].shape == kernel["wrong_scale_controls"]["2pi"].shape
    assert np.allclose(kernel["wrong_scale_controls"]["2pi"], kernel["matrix"])
    assert not np.allclose(kernel["wrong_scale_controls"]["1x"], kernel["matrix"])
    assert kernel["raw_response_summary"]["std"] > 0.0
    assert kernel["response_summary"]["std"] > 0.0
    assert kernel["collar_operator_report"]["readout"] == "source_support_to_lambda_transported_support_with_collar_flow_weights"


def test_grouped_robust_zscore_preserves_time_groups():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=2, theta_values=[0.75], seed=93)
    raw_fields = {
        "record_signature": np.arange(points.shape[0]) % 23,
        "stable_count": (np.arange(points.shape[0]) % 13) + 1,
        "committed_mask": (np.arange(points.shape[0]) % 2 == 0).astype(float),
        "repair_load": np.abs(points[:, 0]),
        "s3_sector_class": np.arange(points.shape[0]) % 6,
    }
    observer_views = [
        {
            "view_type": "patch_observer",
            "observer_id": index,
            "axis": [float(value) for value in points[index]],
            "support_nodes": list(range(index, min(index + 32, points.shape[0]))),
        }
        for index in range(0, 256, 32)
    ]

    kernel = modular_response_kernel(
        points,
        caps,
        raw_fields,
        observer_views,
        times=[0.125, 0.25, 0.5],
        observable_mode="collar_operator_transition",
        transition_observables=["record_family", "s3_sector_class"],
        transition_feature_types=["class_distribution_delta", "change_probability_delta"],
        transform="signed_group_robust_zscore",
        wrong_scales=[1.0, 2.0 * np.pi],
    )

    assert kernel["transform_report"]["mode"] == "signed_group_robust_zscore"
    assert kernel["transform_report"]["group_count"] < kernel["matrix"].shape[1]
    assert kernel["matrix"].shape == kernel["wrong_scale_controls"]["2pi"].shape
    assert np.allclose(kernel["wrong_scale_controls"]["2pi"], kernel["matrix"])


def test_perturb_resettle_can_read_transported_observer_supports():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=2, theta_values=[0.75], seed=88)
    left = np.arange(0, 384, dtype=np.int64)
    right = (left + 11) % points.shape[0]
    port_left = (np.arange(left.size) % 6).astype(np.int16)
    port_right = ((np.arange(left.size) + 2) % 6).astype(np.int16)
    raw_fields = {
        "record_signature": np.arange(points.shape[0]) % 23,
        "stable_count": (np.arange(points.shape[0]) % 13) + 1,
        "committed_mask": (np.arange(points.shape[0]) % 2 == 0).astype(float),
        "repair_load": np.abs(points[:, 0]),
        "cumulative_repair_load": np.abs(points[:, 1]),
        "s3_sector_class": np.arange(points.shape[0]) % 6,
    }
    observer_views = [
        {
            "view_type": "patch_observer",
            "observer_id": index,
            "axis": [float(value) for value in points[index]],
            "support_nodes": list(range(index, min(index + 32, points.shape[0]))),
        }
        for index in range(0, 256, 32)
    ]

    kernel = modular_response_kernel(
        points,
        caps,
        raw_fields,
        observer_views,
        times=[0.2],
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
        perturb_budget_mode="fixed_collar_fraction",
        fixed_perturb_fraction=0.25,
        perturb_selection_mode="lambda_displacement",
        transition_readout_mode="transported_support",
        repair_steps=2,
        repairs_per_step=32,
        perturb_seed=89,
        wrong_scales=[1.0, 2.0 * np.pi],
    )

    assert kernel["observable_mode"] == "perturb_resettle_transition"
    assert kernel["perturb_resettle_report"]["transition_readout_mode"] == "transported_support"
    assert np.allclose(kernel["wrong_scale_controls"]["2pi"], kernel["matrix"])
    assert not np.allclose(kernel["wrong_scale_controls"]["1x"], kernel["matrix"])
    assert kernel["raw_response_summary"]["std"] > 0.0


def test_transition_response_alias_uses_perturb_resettle_kernel():
    points = fibonacci_sphere_points(192)
    caps = sample_caps(points, count=1, theta_values=[0.55], seed=10)
    left = np.arange(0, 96, dtype=np.int64)
    right = (left + 5) % points.shape[0]
    raw_fields = {
        "record_signature": np.arange(points.shape[0]) % 11,
        "stable_count": (np.arange(points.shape[0]) % 7) + 1,
        "committed_mask": (np.arange(points.shape[0]) % 2 == 0).astype(float),
        "repair_load": np.abs(points[:, 0]),
        "s3_sector_class": np.arange(points.shape[0]) % 6,
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

    kernel = modular_response_kernel(
        points,
        caps,
        raw_fields,
        observer_views,
        times=[0.125],
        observable_mode="transition_response",
        transition_observables=["checkpoint_class", "record_family"],
        transform="signed_zscore",
        graph_state={
            "left": left,
            "right": right,
            "port_left": (np.arange(left.size) % 6).astype(np.int16),
            "port_right": ((np.arange(left.size) + 1) % 6).astype(np.int16),
            "group_order": 6,
            "patch_count": points.shape[0],
        },
        repair_steps=1,
        repairs_per_step=16,
    )

    assert kernel["observable_mode"] == "perturb_resettle_transition"
    assert kernel["matrix"].shape[0] == len(observer_views)
    assert kernel["transform_report"]["mode"] == "signed_zscore"


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
    matrix = np.asarray(kernel["matrix"], dtype=float)
    kernel["wrong_scale_controls"] = {
        "1x": 0.5 * np.ones_like(matrix),
        "pi": np.flip(matrix, axis=1),
    }

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
        control_fit_mode="affine_target_fit",
    )

    assert report["fit_mode"] == "joint_global"
    assert report["MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT"] is True
    assert report["H3_RESPONSE_CONTROL_SEPARATION_RECEIPT"] is True
    assert report["h3_fit"]["heldout_normalized_rmse"] < report["s2_boundary_control"]["heldout_normalized_rmse"]
    assert report["h3_chart_dimension_debug"]["point_count"] == 24
    assert report["wrong_scale_feature_audit"]["eligible"] is True
    assert report["wrong_scale_feature_audit"]["audited_feature_count"] > 0
    assert "2pi_h3_fit" in report["wrong_scale_feature_audit"]["winner_counts"]
    assert "2pi_h3_fit" in report["wrong_scale_feature_audit"]["material_winner_counts"]
    assert report["wrong_scale_feature_audit"]["material_margin"] == 0.02
    assert report["wrong_scale_feature_audit"]["worst_groups"]
    assert "material_wrong_scale_advantage_energy_fraction" in report["wrong_scale_feature_audit"]
    assert report["h3_response_stage_gates"]["material_wrong_scale_gate_metric"] == (
        "material_wrong_scale_advantage_energy_fraction"
    )


def test_material_wrong_scale_gate_uses_residual_energy_not_singleton_count():
    h3_rmse = np.ones(20, dtype=float)
    wrong = np.ones(20, dtype=float) + 0.2
    wrong[:8] = 0.95
    feature_rows = [
        {
            "cap_index": index,
            "time_index": 0,
            "time": 0.1,
            "observable": "checkpoint_class",
            "feature_type": "class_distribution_delta",
        }
        for index in range(20)
    ]

    audit = _strict_wrong_scale_feature_audit(
        {"feature_rmse": h3_rmse},
        {"pi": {"feature_rmse": wrong}},
        feature_rows,
        np.zeros(20, dtype=bool),
    )
    gates = _h3_response_stage_gates(
        {
            "heldout_normalized_rmse": 0.90,
            "heldout_explained_variance": 0.12,
        },
        {"heldout_normalized_rmse": 0.96},
        {
            "no_perturbation": {"heldout_normalized_rmse": 1.0},
            "shuffled_response": {"heldout_normalized_rmse": 1.0},
        },
        {"pi": {"heldout_normalized_rmse": 1.0}},
        audit,
    )

    assert audit["material_wrong_scale_win_fraction"] == 0.4
    assert audit["material_wrong_scale_advantage_energy_fraction"] < 0.05
    assert gates["material_feature_gate"] is True
    assert gates["H3_RESPONSE_CANDIDATE_RECEIPT"] is True


def test_modular_response_h3_feature_selection_filters_dead_columns():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=10, theta_values=[0.55, 0.75, 1.0], seed=16)
    kernel = synthetic_h3_modular_kernel(
        caps,
        observer_count=24,
        times=[0.05, 0.1],
        field_names=["record_signature", "stable_count"],
        seed=17,
        softness=0.2,
    )
    matrix = kernel["matrix"]
    dead = np.zeros((matrix.shape[0], 20), dtype=float)
    kernel["matrix"] = np.hstack([matrix, dead])
    kernel["s2_boundary_control"] = np.hstack([kernel["s2_boundary_control"], dead])
    kernel["shuffled_control"] = np.hstack([kernel["shuffled_control"], dead])
    kernel["no_modular_flow_control"] = np.hstack([kernel["no_modular_flow_control"], dead])
    for index in range(dead.shape[1]):
        kernel["feature_rows"].append(
            {
                "feature_index": len(kernel["feature_rows"]),
                "cap_index": 0,
                "time_index": 0,
                "time": 0.0,
                "field": f"dead_{index}",
            }
        )

    report = modular_response_h3_report(
        kernel,
        caps,
        candidate_count=4096,
        candidate_radius=1.2,
        softness=0.2,
        seed=17,
        pass_ratio=0.95,
        min_observers=8,
        min_features=12,
        fit_mode="joint_global",
        anchor_weight=0.0,
        max_iterations=3,
        feature_selection="variance",
        max_fit_features=24,
        min_feature_std=1.0e-4,
    )

    assert report["feature_selection"]["mode"] == "variance"
    assert report["feature_selection"]["original_feature_count"] == matrix.shape[1] + 20
    assert report["feature_selection"]["selected_feature_count"] == 24
    assert report["feature_count"] == 24
    assert report["MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT"] is True


def test_h3_feature_exclusion_applies_to_none_mode():
    matrix = np.asarray(
        [
            [1.0, 0.2, 0.4],
            [0.9, 0.3, 0.5],
            [1.1, 0.1, 0.6],
        ],
        dtype=float,
    )
    feature_rows = [
        {"observable": "record_family", "feature_type": "class_distribution_delta", "cap_index": 0},
        {"observable": "s3_sector_class", "feature_type": "class_distribution_delta", "cap_index": 0},
        {"observable": "checkpoint_class", "feature_type": "change_probability_delta", "cap_index": 0},
    ]
    kernel = {
        "matrix": matrix,
        "feature_rows": feature_rows,
        "s2_boundary_control": np.zeros_like(matrix),
        "shuffled_control": np.zeros_like(matrix),
        "no_modular_flow_control": np.zeros_like(matrix),
        "wrong_scale_controls": {"1x": np.zeros_like(matrix)},
    }

    selected_kernel, selected_matrix, selected_rows, selection = _select_fit_features(
        kernel,
        matrix,
        feature_rows,
        mode="none",
        max_features=None,
        min_std=0.0,
        min_wrong_scale_delta=0.0,
        exclude_observables=("record_family",),
        exclude_feature_types=(),
        min_features=1,
        max_features_per_cap_time_observable=None,
    )

    assert selected_matrix.shape[1] == 2
    assert selected_kernel["s2_boundary_control"].shape == selected_matrix.shape
    assert selection["excluded_feature_count"] == 1
    assert selection["pre_exclusion_feature_count"] == 3
    assert all(row["observable"] != "record_family" for row in selected_rows)


def test_modular_response_h3_feature_selection_can_filter_wrong_scale_indistinct_columns():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=10, theta_values=[0.55, 0.75, 1.0], seed=166)
    kernel = synthetic_h3_modular_kernel(
        caps,
        observer_count=24,
        times=[0.05, 0.1],
        field_names=["record_signature", "stable_count"],
        seed=167,
        softness=0.2,
    )
    matrix = kernel["matrix"]
    ambiguous = matrix[:, :12].copy()
    kernel["matrix"] = np.hstack([matrix, ambiguous])
    kernel["s2_boundary_control"] = np.hstack([kernel["s2_boundary_control"], ambiguous])
    kernel["shuffled_control"] = np.hstack([kernel["shuffled_control"], ambiguous])
    kernel["no_modular_flow_control"] = np.hstack([kernel["no_modular_flow_control"], ambiguous])
    wrong_original = matrix + 0.05
    kernel["wrong_scale_controls"] = {
        "1x": np.hstack([wrong_original, ambiguous]),
        "pi": np.hstack([wrong_original + 0.02, ambiguous]),
    }
    for index in range(ambiguous.shape[1]):
        kernel["feature_rows"].append(
            {
                "feature_index": len(kernel["feature_rows"]),
                "cap_index": 0,
                "time_index": 0,
                "time": 0.0,
                "field": f"ambiguous_{index}",
                "feature_type": "class_distribution_delta",
            }
        )

    report = modular_response_h3_report(
        kernel,
        caps,
        candidate_count=4096,
        candidate_radius=1.2,
        softness=0.2,
        seed=167,
        pass_ratio=0.95,
        min_observers=8,
        min_features=12,
        fit_mode="joint_global",
        anchor_weight=0.0,
        max_iterations=3,
        feature_selection="class_distribution_and_change",
        max_fit_features=48,
        min_feature_std=1.0e-4,
        min_wrong_scale_feature_delta=1.0e-4,
    )

    selection = report["feature_selection"]
    assert selection["wrong_scale_delta_filter_applied"] is True
    assert selection["min_wrong_scale_feature_delta"] == 1.0e-4
    assert selection["selected_feature_count"] < matrix.shape[1] + ambiguous.shape[1]
    assert selection["scale_delta_min"] >= 1.0e-4


def test_modular_response_h3_feature_selection_can_rank_by_wrong_scale_delta():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=10, theta_values=[0.55, 0.75, 1.0], seed=176)
    kernel = synthetic_h3_modular_kernel(
        caps,
        observer_count=24,
        times=[0.05, 0.1],
        field_names=["record_signature", "stable_count"],
        seed=177,
        softness=0.2,
    )
    matrix = kernel["matrix"]
    for row in kernel["feature_rows"]:
        row["observable"] = str(row.get("field", "record_signature"))
        row["feature_type"] = "class_distribution_delta"
    ambiguous = 10.0 * matrix[:, :8].copy()
    kernel["matrix"] = np.hstack([matrix, ambiguous])
    kernel["s2_boundary_control"] = np.hstack([kernel["s2_boundary_control"], ambiguous])
    kernel["shuffled_control"] = np.hstack([kernel["shuffled_control"], ambiguous])
    kernel["no_modular_flow_control"] = np.hstack([kernel["no_modular_flow_control"], ambiguous])
    kernel["wrong_scale_controls"] = {
        "1x": np.hstack([matrix + 0.2, ambiguous]),
        "pi": np.hstack([matrix + 0.3, ambiguous]),
    }
    for index in range(ambiguous.shape[1]):
        kernel["feature_rows"].append(
            {
                "feature_index": len(kernel["feature_rows"]),
                "cap_index": 0,
                "time_index": 0,
                "time": 0.0,
                "field": f"high_variance_wrong_scale_ambiguous_{index}",
                "observable": "record_signature",
                "feature_type": "class_distribution_delta",
            }
        )

    report = modular_response_h3_report(
        kernel,
        caps,
        candidate_count=1024,
        candidate_radius=1.2,
        softness=0.2,
        seed=177,
        pass_ratio=1.0,
        min_observers=8,
        min_features=12,
        fit_mode="joint_global",
        anchor_weight=0.0,
        max_iterations=2,
        feature_selection="class_distribution_and_change_scale_rank",
        max_fit_features=matrix.shape[1],
        min_feature_std=1.0e-4,
        min_wrong_scale_feature_delta=1.0e-4,
    )

    selection = report["feature_selection"]
    assert selection["rank_strategy"] == "wrong_scale_delta"
    assert selection["selected_feature_count"] == matrix.shape[1]
    assert selection["scale_delta_min"] >= 0.19
    assert all(
        "high_variance_wrong_scale_ambiguous" not in row["field"]
        for row in report["feature_rows_sample"]
    )


def test_modular_response_h3_feature_selection_can_keep_change_probability_only():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=8, theta_values=[0.55, 0.75, 1.0], seed=26)
    kernel = synthetic_h3_modular_kernel(
        caps,
        observer_count=24,
        times=[0.05, 0.1],
        field_names=["record_signature", "stable_count"],
        seed=27,
        softness=0.2,
    )
    for index, row in enumerate(kernel["feature_rows"]):
        row["observable"] = str(row.get("field", "record_signature"))
        row["feature_type"] = "change_probability_delta" if index % 2 == 0 else "target_distribution_delta"
        row["target_class"] = None if index % 2 == 0 else 0

    report = modular_response_h3_report(
        kernel,
        caps,
        candidate_count=1024,
        candidate_radius=1.2,
        softness=0.2,
        seed=27,
        pass_ratio=1.0,
        min_observers=8,
        min_features=4,
        fit_mode="joint_global",
        anchor_weight=0.0,
        max_iterations=2,
        feature_selection="change_probability_only",
    )

    assert report["feature_selection"]["mode"] == "change_probability_only"
    assert report["feature_selection"]["metadata_filter"] == "feature_type=change_probability_delta"
    assert report["feature_selection"]["selected_feature_count"] == len(kernel["feature_rows"]) // 2
    assert all(
        row["feature_type"] == "change_probability_delta"
        for row in report["feature_rows_sample"]
    )


def test_modular_response_h3_feature_selection_accepts_pro_low_order_alias():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=8, theta_values=[0.55, 0.75, 1.0], seed=28)
    kernel = synthetic_h3_modular_kernel(
        caps,
        observer_count=24,
        times=[0.05, 0.1],
        field_names=["record_signature", "stable_count"],
        seed=29,
        softness=0.2,
    )
    feature_types = [
        "class_distribution_delta",
        "class_log_odds_delta",
        "transition_matrix_delta",
        "entropy_delta",
        "sector_preservation_delta",
        "change_probability_delta",
    ]
    for index, row in enumerate(kernel["feature_rows"]):
        row["observable"] = str(row.get("field", "record_signature"))
        row["feature_type"] = feature_types[index % len(feature_types)]
        row["target_class"] = index % 6

    report = modular_response_h3_report(
        kernel,
        caps,
        candidate_count=1024,
        candidate_radius=1.2,
        softness=0.2,
        seed=29,
        pass_ratio=1.0,
        min_observers=8,
        min_features=4,
        fit_mode="joint_global",
        anchor_weight=0.0,
        max_iterations=2,
        feature_selection="signed_class_resolved_transition_features_without_transition_matrix",
    )

    assert report["feature_selection"]["mode"] == "signed_class_resolved_transition_features_without_transition_matrix"
    assert report["feature_selection"]["metadata_filter"] == "signed_class_resolved_transition_features_without_transition_matrix"
    assert report["feature_selection"]["selected_feature_count"] > 0
    assert all(
        row["feature_type"] != "transition_matrix_delta"
        for row in report["feature_rows_sample"]
    )


def test_grouped_h3_feature_selection_aggregates_controls_identically():
    matrix = np.arange(24, dtype=float).reshape(4, 6)
    feature_rows = []
    for cap_index in range(2):
        for target_class in range(3):
            feature_rows.append(
                {
                    "feature_index": len(feature_rows),
                    "cap_index": cap_index,
                    "time_index": 0,
                    "time": 0.1,
                    "observable": "record_family",
                    "feature_type": "class_distribution_delta",
                    "target_class": target_class,
                }
            )
    kernel = {
        "matrix": matrix,
        "feature_rows": feature_rows,
        "s2_boundary_control": matrix + 10.0,
        "shuffled_control": matrix + 20.0,
        "shuffled_response_control": matrix + 30.0,
        "shuffled_observer_labels_control": matrix + 40.0,
        "no_modular_flow_control": matrix + 50.0,
        "wrong_scale_controls": {"1x": matrix + 60.0},
    }

    selected_kernel, selected_matrix, selected_rows, report = _select_fit_features(
        kernel,
        matrix,
        feature_rows,
        mode="grouped_signed_transition_no_matrix",
        max_features=None,
        min_std=0.0,
        min_wrong_scale_delta=0.0,
        exclude_observables=(),
        exclude_feature_types=(),
        min_features=2,
        max_features_per_cap_time_observable=None,
    )

    assert report["mode"] == "grouped_signed_transition_no_matrix"
    assert report["aggregation_mode"] == "cap_time_observable_feature_type_mean"
    assert report["pre_aggregation_selected_feature_count"] == 6
    assert report["selected_feature_count"] == 2
    assert selected_matrix.shape == (4, 2)
    assert np.allclose(selected_matrix[:, 0], np.mean(matrix[:, :3], axis=1))
    assert np.allclose(selected_matrix[:, 1], np.mean(matrix[:, 3:], axis=1))
    assert np.allclose(selected_kernel["s2_boundary_control"][:, 0], np.mean(matrix[:, :3] + 10.0, axis=1))
    assert np.allclose(selected_kernel["wrong_scale_controls"]["1x"][:, 1], np.mean(matrix[:, 3:] + 60.0, axis=1))
    assert selected_rows[0]["feature_type"] == "grouped_class_distribution_delta"
    assert selected_rows[0]["source_feature_count"] == 3


def test_modular_response_kernel_cache_roundtrips_and_refits(tmp_path):
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=8, theta_values=[0.55, 0.75, 1.0], seed=31)
    kernel = synthetic_h3_modular_kernel(
        caps,
        observer_count=24,
        times=[0.05, 0.1],
        field_names=["record_signature", "stable_count"],
        seed=32,
        softness=0.2,
    )
    cache_report = write_modular_response_kernel_cache(tmp_path, kernel, caps)
    loaded_kernel, loaded_caps, metadata = load_modular_response_kernel_cache(tmp_path)

    assert cache_report["observer_count"] == 24
    assert loaded_kernel["matrix"].shape == kernel["matrix"].shape
    assert loaded_kernel["wrong_scale_controls"] == {}
    assert len(loaded_caps) == len(caps)
    assert metadata["feature_rows"][0]["cap_index"] == 0

    out = tmp_path / "refit.json"
    report = write_h3_refit_report(
        tmp_path,
        out,
        candidate_count=2048,
        candidate_radius=1.2,
        softness=0.2,
        seed=32,
        pass_ratio=0.95,
        min_observers=8,
        min_features=12,
        fit_mode="joint_global",
        anchor_weight=0.0,
        max_iterations=3,
    )

    assert out.exists()
    assert report["MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT"] is True
    assert report["kernel_cache"]["observer_count"] == 24


def test_h3_refit_seed_ensemble_reports_robustness(tmp_path):
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=8, theta_values=[0.55, 0.75, 1.0], seed=41)
    kernel = synthetic_h3_modular_kernel(
        caps,
        observer_count=24,
        times=[0.05, 0.1],
        field_names=["record_signature", "stable_count"],
        seed=42,
        softness=0.2,
    )
    write_modular_response_kernel_cache(tmp_path, kernel, caps)

    report = write_h3_refit_ensemble_report(
        tmp_path,
        tmp_path / "ensemble.json",
        seeds=[42, 43, 44],
        required_receipt_fraction=0.5,
        candidate_count=2048,
        candidate_radius=1.2,
        softness=0.2,
        pass_ratio=0.95,
        min_observers=8,
        min_features=12,
        fit_mode="joint_global",
        anchor_weight=0.0,
        max_iterations=3,
    )

    assert report["mode"] == "h3_refit_seed_ensemble"
    assert report["seed_count"] == 3
    assert report["receipt_count"] >= 1
    assert report["h3_response_seed_robust_receipt"] is True
    assert report["physical_claim"] is False


def test_h3_ensemble_gate_uses_aggregate_control_separation_not_candidate_fraction():
    rows = [
        {
            "candidate_receipt": index < 4,
            "control_separation_receipt": True,
            "heldout_explained_variance": 0.13,
            "heldout_normalized_rmse": 0.93,
            "material_wrong_scale_win_fraction": 0.05859375,
            "material_wrong_scale_gate_value": 0.01,
            "assignment_unique_count": 150,
            "candidate_3d_dimension_window": False,
            "receipt": index < 4,
        }
        for index in range(8)
    ]

    report = _h3_ensemble_summary_from_rows(
        rows,
        required_receipt_fraction=0.75,
        required_median_ev=0.08,
        required_p75_material_wrong_fraction=0.075,
        required_dim3_fraction=0.5,
    )

    assert report["candidate_receipt_fraction"] == 0.5
    assert report["candidate_receipt_fraction_diagnostic_only"] == 0.5
    assert report["control_separation_receipt_fraction"] == 1.0
    assert report["p75_material_wrong_scale_win_fraction"] == 0.05859375
    assert report["p75_material_wrong_scale_gate_value"] == 0.01
    assert report["H3_RESPONSE_ENSEMBLE_RECEIPT"] is True
    assert report["h3_chart_3d_seed_robust_receipt"] is False
    assert report["ensemble_gate_uses"] == [
        "control_separation_receipt_fraction",
        "median_heldout_explained_variance",
        "p75_material_wrong_scale_gate_value",
    ]


def test_joint_h3_fit_local_refinement_is_fair_to_controls():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=8, theta_values=[0.55, 0.75, 1.0], seed=51)
    kernel = synthetic_h3_modular_kernel(
        caps,
        observer_count=24,
        times=[0.05, 0.1],
        field_names=["record_signature", "stable_count"],
        seed=52,
        softness=0.2,
    )

    report = modular_response_h3_report(
        kernel,
        caps,
        candidate_count=512,
        candidate_radius=1.2,
        softness=0.2,
        seed=52,
        pass_ratio=0.95,
        min_observers=8,
        min_features=12,
        fit_mode="joint_global",
        anchor_weight=0.0,
        max_iterations=2,
        refine_steps=1,
        refine_max_rows=8,
        refine_max_nfev=8,
    )

    assert report["h3_fit"]["refinement"]["enabled"] is True
    assert "shuffled_response" in report["control_fits"]
    assert report["control_fits"]["shuffled_response"]["refinement"]["enabled"] is True


def test_joint_h3_fit_can_use_deterministic_candidate_ball():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=8, theta_values=[0.55, 0.75, 1.0], seed=61)
    kernel = synthetic_h3_modular_kernel(
        caps,
        observer_count=24,
        times=[0.05, 0.1],
        field_names=["record_signature", "stable_count"],
        seed=62,
        softness=0.2,
    )

    report = modular_response_h3_report(
        kernel,
        caps,
        candidate_count=1024,
        candidate_radius=1.2,
        softness=0.2,
        seed=62,
        pass_ratio=0.95,
        min_observers=8,
        min_features=12,
        fit_mode="joint_global",
        anchor_weight=0.0,
        max_iterations=3,
        candidate_mode="fibonacci_ball",
    )

    assert report["h3_fit"]["candidate_mode"] == "fibonacci_ball"
    assert report["h3_fit"]["assignment_unique_count"] > 0
    assert report["MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT"] is True
