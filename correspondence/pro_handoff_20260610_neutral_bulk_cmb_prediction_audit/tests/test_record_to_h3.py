import json
from pathlib import Path

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
    recompute_object_chart_from_saved_run,
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
        incidence_mode="support_overlap",
    )

    assert report["mode"] == "observer_chart_object_h3_population"
    assert report["object_count"] == 3
    assert report["observer_chart_object_h3_receipt"] is True
    assert report["median_h3_compactness_normalized"] < report["median_shuffled_h3_compactness_normalized"]
    assert report["sample_objects"][0]["h3_spatial_point"]


def test_observer_chart_object_population_can_audit_boundary_compactness_without_veto():
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
        signature = cluster + 60
        object_rows.append(
            {
                "object_id": f"obj_boundary_audit_{cluster}",
                "support_nodes": list(range(support_start, support_start + 12)),
                "record_signature": signature,
            }
        )
        for local in range(4):
            observer_id = cluster * 4 + local
            tangent = np.array([float(cluster) * 1.5 + 0.02 * local, 0.01 * local, 0.0])
            h3_points.append(h3_point_from_tangent(tangent))
            observer_views.append(
                {
                    "view_type": "patch_observer",
                    "observer_id": observer_id,
                    # Deliberately identical axes inside each cluster: in OPH this is
                    # the angular chart being compact, not an automatic bulk veto.
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
        seed=404,
        min_objects=3,
        min_observers_per_object=2,
        pass_ratio=0.95,
        incidence_mode="support_overlap",
        max_h3_compactness=0.35,
        min_localized_objects=3,
        shuffle_control_count=8,
        boundary_gate_mode="boundary_leakage_audit",
    )

    assert report["h3_not_boundary_dominated"] is False
    assert report["boundary_leakage_audit_pass"] is False
    assert report["localized_nonboundary_bulk_population_receipt"] is False
    assert report["localized_h3_bulk_population_receipt"] is True
    assert report["THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT"] is True
    assert report["OBJECT_H3_NONBOUNDARY_POPULATION_RECEIPT"] is False
    assert report["observer_chart_object_h3_receipt"] is True
    assert report["observer_chart_bulk_population_receipt"] is False
    assert report["bulk_population_gate_mode"] == "localized_h3_subpopulation_vs_shuffled_with_boundary_leakage_audit"


def test_observer_chart_object_population_splits_precursor_from_strict_bulk_gate():
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
        signature = cluster + 40
        object_rows.append(
            {
                "object_id": f"obj_precursor_{cluster}",
                "support_nodes": list(range(support_start, support_start + 12)),
                "record_signature": signature,
            }
        )
        for local in range(4):
            observer_id = cluster * 4 + local
            tangent = np.array([float(cluster) * 1.6 + 0.02 * local, 0.02 * local, 0.0])
            h3_points.append(h3_point_from_tangent(tangent))
            angle = float(cluster) * 1.7 + float(local) * 0.45
            axis = np.array([np.cos(angle), np.sin(angle), 0.1 * float(local + 1)])
            axis = axis / np.linalg.norm(axis)
            observer_views.append(
                {
                    "view_type": "patch_observer",
                    "observer_id": observer_id,
                    "axis": [float(value) for value in axis],
                    "support_nodes": list(range(support_start + local, support_start + local + 8)),
                    "object_packet_histogram": {str(signature): 1.0},
                }
            )
    h3_report = {
        "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": False,
        "H3_RESPONSE_CONTROL_SEPARATION_RECEIPT": True,
        "observer_ids": list(range(12)),
        "h3_fit": {"fitted_h3_points": [[float(value) for value in row] for row in h3_points]},
    }

    report = observer_chart_object_population_report(
        observer_views,
        object_rows,
        h3_report,
        seed=44,
        min_objects=3,
        min_observers_per_object=2,
        pass_ratio=0.95,
        incidence_mode="support_overlap",
    )

    assert report["modular_response_h3_control_separation_receipt"] is True
    assert report["modular_response_h3_strict_receipt"] is False
    assert report["localized_object_precursor_receipt"] is True
    assert report["localized_nonboundary_bulk_population_receipt"] is False


def test_observer_chart_object_population_can_use_packet_mass_incidence():
    h3_points = [h3_point_from_tangent(np.array([0.02 * index, 0.0, 0.0])) for index in range(6)]
    observer_views = [
        {
            "view_type": "patch_observer",
            "observer_id": index,
            "axis": [1.0 if index % 2 == 0 else -1.0, 0.0, 0.0],
            "support_nodes": list(range(1000 + 10 * index, 1008 + 10 * index)),
            "object_packet_histogram": {"42": 0.4 if index < 4 else 0.0},
        }
        for index in range(6)
    ]
    object_rows = [
        {
            "object_id": "obj_packet",
            "support_nodes": [0, 1, 2],
            "record_signature": 42,
        }
    ]
    h3_report = {
        "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": True,
        "observer_ids": list(range(6)),
        "h3_fit": {"fitted_h3_points": [[float(value) for value in row] for row in h3_points]},
    }

    report = observer_chart_object_population_report(
        observer_views,
        object_rows,
        h3_report,
        seed=5,
        min_objects=1,
        min_observers_per_object=3,
        incidence_mode="packet_mass",
        min_packet_mass=0.1,
    )

    assert report["incidence_mode"] == "packet_mass"
    assert report["object_count"] == 1
    assert report["sample_objects"][0]["observer_count"] == 4


def test_observer_chart_object_population_can_use_observer_transition_clusters():
    h3_points = []
    observer_views = []
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    for cluster in range(3):
        for local in range(4):
            observer_id = cluster * 4 + local
            tangent = np.array([float(cluster) * 1.5 + 0.02 * local, 0.01 * local, 0.0])
            h3_points.append(h3_point_from_tangent(tangent))
            observer_views.append(
                {
                    "view_type": "patch_observer",
                    "observer_id": observer_id,
                    "axis": [float(value) for value in axes[(cluster + local) % len(axes)]],
                    "support_nodes": list(range(10_000 + observer_id * 10, 10_008 + observer_id * 10)),
                    "transition_affinity_dominants": {
                        "record_family": cluster,
                        "s3_sector_class": cluster % 3,
                        "repair_load_bucket": local % 2,
                    },
                    "transition_affinity_histograms": {
                        "record_family": {str(cluster): 1.0},
                        "s3_sector_class": {str(cluster % 3): 1.0},
                        "repair_load_bucket": {str(local % 2): 1.0},
                    },
                }
            )
    # Deliberately global record family: support overlap would not localize this.
    object_rows = [
        {
            "object_id": "global_background",
            "support_nodes": list(range(0, 1024)),
            "record_signature": 42,
        }
    ]
    h3_report = {
        "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": True,
        "observer_ids": list(range(12)),
        "h3_fit": {"fitted_h3_points": [[float(value) for value in row] for row in h3_points]},
    }

    report = observer_chart_object_population_report(
        observer_views,
        object_rows,
        h3_report,
        seed=6,
        min_objects=3,
        min_observers_per_object=2,
        incidence_mode="observer_transition_cluster",
        observer_cluster_fields=("record_family", "s3_sector_class"),
        max_observer_fraction_per_object=0.5,
        pass_ratio=0.95,
        max_h3_compactness=0.5,
        min_localized_objects=2,
    )

    assert report["incidence_mode"] == "observer_transition_cluster"
    assert report["object_count"] == 3
    assert report["observer_chart_object_h3_receipt"] is True
    assert report["sample_objects"][0]["family_mode"] == "observer_transition_cluster"
    assert report["median_h3_compactness_normalized"] < report["median_shuffled_h3_compactness_normalized"]
    assert report["localized_object_precursor_receipt"] is True


def test_observer_chart_object_population_can_use_record_family_modular_response_mixture():
    h3_points = []
    observer_views = []
    object_rows = []
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    for cluster in range(3):
        object_rows.append(
            {
                "object_id": f"obj_family_{cluster}",
                "support_nodes": list(range(1000 + 100 * cluster, 1024 + 100 * cluster)),
                "record_signature": cluster + 80,
                "transition_affinity": {
                    "record_family": cluster,
                    "s3_sector_class": cluster % 3,
                },
            }
        )
        for local in range(4):
            observer_id = cluster * 4 + local
            tangent = np.array([float(cluster) * 1.5 + 0.02 * local, 0.01 * local, 0.0])
            h3_points.append(h3_point_from_tangent(tangent))
            axis = axes[(cluster + local) % len(axes)]
            observer_views.append(
                {
                    "view_type": "patch_observer",
                    "observer_id": observer_id,
                    "axis": [float(value) for value in axis],
                    "support_nodes": list(range(50_000 + observer_id * 10, 50_008 + observer_id * 10)),
                    "transition_affinity_histograms": {
                        "record_family": {str(cluster): 1.0},
                        "s3_sector_class": {str(cluster % 3): 1.0},
                    },
                    "modular_response_histograms": {
                        "modular_response_cluster": {str(cluster): 1.0},
                        "modular_response_component_0": {str(cluster): 1.0},
                    },
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
        seed=66,
        min_objects=3,
        min_observers_per_object=2,
        incidence_mode="record_family_modular_response_mixture",
        observer_cluster_fields=("modular_response_cluster", "modular_response_component_0"),
        min_transition_affinity=0.1,
        min_observer_cluster_weight=0.1,
        max_observer_fraction_per_object=0.5,
        pass_ratio=0.95,
        max_h3_compactness=0.5,
        min_localized_objects=2,
    )

    assert report["incidence_mode"] == "record_family_modular_response_mixture"
    assert report["object_count"] == 3
    assert report["observer_chart_object_h3_receipt"] is True
    assert report["sample_objects"][0]["family_mode"] == "record_family_modular_response_mixture"
    assert report["localized_object_precursor_receipt"] is True


def test_record_family_modular_response_mixture_can_use_packet_or_support_visibility():
    h3_points = []
    observer_views = []
    object_rows = []
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    for cluster in range(3):
        object_rows.append(
            {
                "object_id": f"obj_packet_visible_{cluster}",
                "support_nodes": list(range(1000 + 100 * cluster, 1024 + 100 * cluster)),
                "record_signature": cluster + 90,
                "transition_affinity": {
                    "record_family": cluster,
                    "s3_sector_class": cluster % 3,
                },
            }
        )
        for local in range(4):
            observer_id = cluster * 4 + local
            tangent = np.array([float(cluster) * 1.5 + 0.02 * local, 0.01 * local, 0.0])
            h3_points.append(h3_point_from_tangent(tangent))
            observer_views.append(
                {
                    "view_type": "patch_observer",
                    "observer_id": observer_id,
                    "axis": [float(value) for value in axes[(cluster + local) % len(axes)]],
                    # No finite support overlap with object_rows; visibility is
                    # through the observer-visible record/modular packet lane.
                    "support_nodes": list(range(50_000 + observer_id * 10, 50_008 + observer_id * 10)),
                    "transition_affinity_histograms": {
                        "record_family": {str(cluster): 1.0},
                        "s3_sector_class": {str(cluster % 3): 1.0},
                    },
                    "modular_response_histograms": {
                        "modular_response_cluster": {str(cluster): 1.0},
                        "modular_response_component_0": {str(cluster): 1.0},
                    },
                }
            )
    h3_report = {
        "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": True,
        "observer_ids": list(range(12)),
        "h3_fit": {"fitted_h3_points": [[float(value) for value in row] for row in h3_points]},
    }

    support_only = observer_chart_object_population_report(
        observer_views,
        object_rows,
        h3_report,
        seed=77,
        min_objects=3,
        min_observers_per_object=2,
        incidence_mode="record_family_modular_response_mixture",
        observer_cluster_fields=("modular_response_cluster", "modular_response_component_0"),
        min_transition_affinity=0.1,
        min_observer_cluster_weight=0.1,
        require_support_visibility=True,
        min_support_visibility=0.01,
    )
    packet_or_support = observer_chart_object_population_report(
        observer_views,
        object_rows,
        h3_report,
        seed=77,
        min_objects=3,
        min_observers_per_object=2,
        incidence_mode="record_family_modular_response_mixture",
        observer_cluster_fields=("modular_response_cluster", "modular_response_component_0"),
        min_transition_affinity=0.1,
        min_observer_cluster_weight=0.1,
        require_support_visibility=True,
        min_support_visibility=0.01,
        visibility_mode="packet_or_support",
        packet_visibility_weight=0.5,
    )

    assert support_only["observer_chart_object_h3_receipt"] is False
    assert packet_or_support["visibility_mode"] == "packet_or_support"
    assert packet_or_support["object_count"] == 3
    assert packet_or_support["observer_chart_object_h3_receipt"] is True


def test_recompute_object_chart_from_saved_run_selects_best_h3_seed(tmp_path: Path):
    h3_points = []
    observer_views = []
    object_rows = []
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    for cluster in range(8):
        object_rows.append(
            {
                "object_id": f"obj_recompute_{cluster}",
                "support_nodes": list(range(1000 + 100 * cluster, 1024 + 100 * cluster)),
                "record_signature": cluster + 150,
                "transition_affinity": {
                    "record_family": cluster,
                    "s3_sector_class": cluster % 3,
                },
            }
        )
        for local in range(4):
            observer_id = cluster * 4 + local
            tangent = np.array([float(cluster) * 1.5 + 0.02 * local, 0.01 * local, 0.0])
            h3_points.append(h3_point_from_tangent(tangent))
            observer_views.append(
                {
                    "view_type": "patch_observer",
                    "observer_id": observer_id,
                    "axis": [float(value) for value in axes[(cluster + local) % len(axes)]],
                    "support_nodes": list(range(50_000 + observer_id * 10, 50_008 + observer_id * 10)),
                    "transition_affinity_histograms": {
                        "record_family": {str(cluster): 1.0},
                        "s3_sector_class": {str(cluster % 3): 1.0},
                    },
                    "modular_response_histograms": {
                        "modular_response_cluster": {str(cluster): 1.0},
                        "modular_response_component_0": {str(cluster): 1.0},
                    },
                }
            )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "observer_views.jsonl", observer_views)
    _write_jsonl(run_dir / "observer_objects.jsonl", object_rows)
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    seed_report = {
        "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": True,
        "observer_ids": list(range(32)),
        "h3_fit": {"fitted_h3_points": [[float(value) for value in row] for row in h3_points]},
    }
    seed_path = seed_dir / "seed_2.json"
    seed_path.write_text(json.dumps(seed_report), encoding="utf-8")
    worse_seed_path = seed_dir / "seed_1.json"
    worse_seed_path.write_text(json.dumps(seed_report), encoding="utf-8")
    ensemble_path = tmp_path / "h3_refit_ensemble_report.json"
    ensemble_path.write_text(
        json.dumps(
            {
                "mode": "h3_refit_seed_ensemble",
                "rows": [
                    {
                        "candidate_receipt": True,
                        "heldout_explained_variance": 0.01,
                        "material_wrong_scale_win_fraction": 0.2,
                        "report_path": str(worse_seed_path),
                    },
                    {
                        "candidate_receipt": True,
                        "heldout_explained_variance": 0.2,
                        "material_wrong_scale_win_fraction": 0.01,
                        "report_path": str(seed_path),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "object_chart_recomputed.json"
    report = recompute_object_chart_from_saved_run(
        run_dir,
        ensemble_path,
        out_path,
        shuffle_control_count=2,
    )

    assert out_path.exists()
    assert report["postprocess_recomputed"] is True
    assert report["selected_h3_report"] == str(seed_path)
    assert report["observer_chart_object_h3_receipt"] is True
    assert report["object_count"] == 8


def test_observer_chart_object_population_can_use_transition_history_incidence():
    h3_points = []
    observer_views = []
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    for cluster in range(3):
        for local in range(4):
            observer_id = cluster * 4 + local
            tangent = np.array([float(cluster) * 1.5 + 0.02 * local, 0.01 * local, 0.0])
            h3_points.append(h3_point_from_tangent(tangent))
            observer_views.append(
                {
                    "view_type": "patch_observer",
                    "observer_id": observer_id,
                    "axis": [float(value) for value in axes[(cluster + local) % len(axes)]],
                    "support_nodes": list(range(30_000 + observer_id * 10, 30_008 + observer_id * 10)),
                    "transition_history_key": 10_000 + cluster,
                    "transition_history_persistence": 4,
                    "transition_history_mean_modal_mass": 0.9,
                    "transition_history_descriptor": {
                        "steps": [
                            {
                                "record_family": cluster,
                                "checkpoint_class": 3,
                                "stable_flag": 1,
                                "s3_sector_class": cluster % 3,
                                "repair_load_bucket": 2,
                            }
                        ]
                        * 4
                    },
                }
            )
    h3_report = {
        "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": True,
        "observer_ids": list(range(12)),
        "h3_fit": {"fitted_h3_points": [[float(value) for value in row] for row in h3_points]},
    }

    report = observer_chart_object_population_report(
        observer_views,
        [],
        h3_report,
        seed=8,
        min_objects=3,
        min_observers_per_object=2,
        incidence_mode="transition_history",
        history_window=4,
        min_persistence=3,
        max_observer_fraction_per_object=0.5,
        pass_ratio=0.95,
        max_h3_compactness=0.5,
        min_localized_objects=2,
    )

    assert report["incidence_mode"] == "transition_history"
    assert report["history_window"] == 4
    assert report["min_persistence"] == 3
    assert report["object_count"] == 3
    assert report["sample_objects"][0]["family_mode"] == "transition_history"
    assert report["observer_chart_object_h3_receipt"] is True
    assert report["localized_object_precursor_receipt"] is True


def test_observer_chart_object_population_can_use_record_sector_checkpoint_lineage():
    h3_points = []
    observer_views = []
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    for cluster in range(3):
        for local in range(4):
            observer_id = cluster * 4 + local
            tangent = np.array([float(cluster) * 1.6 + 0.015 * local, 0.01 * local, 0.0])
            h3_points.append(h3_point_from_tangent(tangent))
            observer_views.append(
                {
                    "view_type": "patch_observer",
                    "observer_id": observer_id,
                    "axis": [float(value) for value in axes[(cluster + local) % len(axes)]],
                    "support_nodes": list(range(60_000 + observer_id * 10, 60_008 + observer_id * 10)),
                    "transition_history": [
                        {
                            "record_family": cluster,
                            "checkpoint_class": 10 + cluster,
                            "s3_sector_class": cluster % 3,
                            "repair_load_bucket": local % 2,
                            "stable_flag": True,
                        }
                        for _step in range(5)
                    ],
                }
            )
    h3_report = {
        "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": True,
        "observer_ids": list(range(12)),
        "h3_fit": {"fitted_h3_points": [[float(value) for value in row] for row in h3_points]},
    }

    report = observer_chart_object_population_report(
        observer_views,
        [],
        h3_report,
        seed=8,
        min_objects=3,
        min_observers_per_object=3,
        incidence_mode="record_sector_checkpoint_lineage",
        max_observer_fraction_per_object=0.5,
        pass_ratio=0.95,
        max_h3_compactness=0.5,
        min_localized_objects=2,
        shuffle_control_count=8,
    )

    assert report["incidence_mode"] == "record_sector_checkpoint_lineage"
    assert report["object_count"] == 3
    assert report["observer_chart_object_h3_receipt"] is True
    assert report["localized_object_precursor_receipt"] is True
    assert report["sample_objects"][0]["family_mode"] == "record_sector_checkpoint_lineage"
    assert "checkpoint_class" in report["sample_objects"][0]["cluster_key"]


def test_observer_chart_object_population_can_use_observer_transition_mixtures():
    h3_points = []
    observer_views = []
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    for cluster in range(3):
        for local in range(4):
            observer_id = cluster * 4 + local
            tangent = np.array([float(cluster) * 1.5 + 0.02 * local, 0.01 * local, 0.0])
            h3_points.append(h3_point_from_tangent(tangent))
            observer_views.append(
                {
                    "view_type": "patch_observer",
                    "observer_id": observer_id,
                    "axis": [float(value) for value in axes[(cluster + local) % len(axes)]],
                    "support_nodes": list(range(20_000 + observer_id * 10, 20_008 + observer_id * 10)),
                    "object_packet_histogram": {"99": 0.55, str(20 + cluster): 0.25},
                    "transition_affinity_histograms": {
                        "cumulative_repair_load_bucket": {"1": 0.6, str(4 + cluster): 0.2},
                    },
                }
            )
    h3_report = {
        "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": True,
        "observer_ids": list(range(12)),
        "h3_fit": {"fitted_h3_points": [[float(value) for value in row] for row in h3_points]},
    }

    report = observer_chart_object_population_report(
        observer_views,
        [],
        h3_report,
        seed=7,
        min_objects=3,
        min_observers_per_object=2,
        incidence_mode="observer_transition_mixture_cluster",
        observer_cluster_fields=("object_packet", "cumulative_repair_load_bucket"),
        observer_cluster_top_k=2,
        min_observer_cluster_weight=0.05,
        max_observer_fraction_per_object=0.5,
        pass_ratio=0.95,
        max_h3_compactness=0.5,
        min_localized_objects=2,
    )

    assert report["incidence_mode"] == "observer_transition_mixture_cluster"
    assert report["object_count"] >= 3
    assert report["sample_objects"][0]["family_mode"] == "observer_transition_mixture_cluster"
    assert report["observer_chart_object_h3_receipt"] is True
    assert report["localized_object_precursor_receipt"] is True


def test_observer_chart_object_population_can_use_local_transition_history_tokens():
    h3_points = []
    observer_views = []
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    for cluster in range(3):
        for local in range(4):
            observer_id = cluster * 4 + local
            tangent = np.array([float(cluster) * 1.5 + 0.02 * local, 0.01 * local, 0.0])
            h3_points.append(h3_point_from_tangent(tangent))
            observer_views.append(
                {
                    "view_type": "patch_observer",
                    "observer_id": observer_id,
                    "axis": [float(value) for value in axes[(cluster + local) % len(axes)]],
                    "support_nodes": list(range(40_000 + observer_id * 10, 40_008 + observer_id * 10)),
                    "transition_history_histograms": {
                        "local_transition_token": {"777": 0.55, str(30 + cluster): 0.30},
                        "local_transition_token_persistent": {str(30 + cluster): 0.30},
                    },
                }
            )
    h3_report = {
        "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": True,
        "observer_ids": list(range(12)),
        "h3_fit": {"fitted_h3_points": [[float(value) for value in row] for row in h3_points]},
    }

    report = observer_chart_object_population_report(
        observer_views,
        [],
        h3_report,
        seed=9,
        min_objects=3,
        min_observers_per_object=2,
        incidence_mode="transition_history_mixture_cluster",
        observer_cluster_fields=("local_transition_token",),
        observer_cluster_top_k=2,
        min_observer_cluster_weight=0.05,
        max_observer_fraction_per_object=0.5,
        pass_ratio=0.95,
        max_h3_compactness=0.5,
        min_localized_objects=2,
        shuffle_control_count=3,
    )

    assert report["incidence_mode"] == "transition_history_mixture_cluster"
    assert report["shuffle_control_count"] == 3
    assert report["p90_shuffled_h3_compactness_normalized"] is not None
    assert report["object_count"] >= 3
    assert report["sample_objects"][0]["family_mode"] == "observer_transition_mixture_cluster"
    assert report["observer_chart_object_h3_receipt"] is True
    assert report["localized_object_precursor_receipt"] is True


def test_observer_chart_object_population_can_split_broad_tokens_into_h3_components():
    h3_points = []
    observer_views = []
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    ]
    for cluster in range(3):
        for local in range(6):
            observer_id = cluster * 6 + local
            tangent = np.array([float(cluster) * 2.2 + 0.02 * local, 0.02 * (local % 2), 0.0])
            h3_points.append(h3_point_from_tangent(tangent))
            observer_views.append(
                {
                    "view_type": "patch_observer",
                    "observer_id": observer_id,
                    "axis": [float(value) for value in axes[(cluster + local) % len(axes)]],
                    "support_nodes": list(range(50_000 + observer_id * 10, 50_008 + observer_id * 10)),
                    "transition_history_histograms": {
                        "local_transition_token": {str(100 + cluster): 0.75, "999": 0.25},
                        "local_transition_token_persistent": {str(100 + cluster): 0.75},
                    },
                }
            )
    h3_report = {
        "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": True,
        "observer_ids": list(range(18)),
        "h3_fit": {"fitted_h3_points": [[float(value) for value in row] for row in h3_points]},
    }

    report = observer_chart_object_population_report(
        observer_views,
        [],
        h3_report,
        seed=11,
        min_objects=3,
        min_observers_per_object=3,
        incidence_mode="transition_history_mixture_cluster",
        observer_cluster_fields=("local_transition_token",),
        observer_cluster_top_k=2,
        min_observer_cluster_weight=0.05,
        max_observer_fraction_per_object=0.8,
        split_h3_components=True,
        component_link_fraction=0.25,
        component_min_observers=3,
        pass_ratio=0.95,
        max_h3_compactness=0.35,
        min_localized_objects=3,
        shuffle_control_count=5,
    )

    assert report["split_h3_components"] is True
    assert report["object_count"] >= 3
    assert report["sample_objects"][0]["component_split"] is True
    assert report["localized_object_count"] >= 3
    assert report["localized_not_boundary_object_count"] >= 3
    assert report["h3_localized"] is True
    assert report["h3_not_boundary_dominated"] is True
    # The robust shuffled-envelope gate can still fail on this tiny sample;
    # splitting is a diagnostic refinement, not a receipt bypass.
    assert report["observer_chart_bulk_population_receipt"] is False


def test_observer_chart_object_population_can_use_transition_affinity_incidence():
    h3_points = [h3_point_from_tangent(np.array([0.02 * index, 0.0, 0.0])) for index in range(8)]
    observer_views = []
    for index in range(8):
        matches = index < 5
        observer_views.append(
            {
                "view_type": "patch_observer",
                "observer_id": index,
                "axis": [1.0 if index % 2 == 0 else -1.0, 0.0, 0.0],
                "support_nodes": list(range(1000 + 10 * index, 1008 + 10 * index)),
                "object_packet_histogram": {"7": 0.2 if matches else 0.0},
                "transition_affinity_histograms": {
                    "checkpoint_class": {"3": 0.8 if matches else 0.0},
                    "s3_sector_class": {"2": 0.7 if matches else 0.1},
                    "repair_load_bucket": {"4": 0.6 if matches else 0.0},
                },
            }
        )
    object_rows = [
        {
            "object_id": "obj_transition",
            "support_nodes": [0, 2, 4, 6],
            "record_signature": 7,
            "transition_affinity": {
                "object_packet": 7,
                "checkpoint_class": 3,
                "s3_sector_class": 2,
                "repair_load_bucket": 4,
            },
        }
    ]
    h3_report = {
        "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": True,
        "observer_ids": list(range(8)),
        "h3_fit": {"fitted_h3_points": [[float(value) for value in row] for row in h3_points]},
    }

    report = observer_chart_object_population_report(
        observer_views,
        object_rows,
        h3_report,
        seed=6,
        min_objects=1,
        min_observers_per_object=4,
        incidence_mode="transition_affinity",
        min_transition_affinity=0.25,
    )

    assert report["incidence_mode"] == "transition_affinity"
    assert report["object_count"] == 1
    assert report["sample_objects"][0]["observer_count"] == 5


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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
