from __future__ import annotations

import numpy as np

from oph_fpe.defects.array_s3_holonomy import (
    S3_INV,
    S3_MUL,
    array_holonomy_report,
    cluster_defects,
    defect_class,
    defect_interaction_report,
    defect_timeline_report,
    particle_likeness_report,
    oriented_triangles,
    s3_triangle_holonomy,
    triangle_holonomies,
    track_defect_worldlines,
)
from oph_fpe.defects.controlled_assay import controlled_s3_particle_assay_report


def test_s3_holonomy_identity_triangle():
    a, b = 1, 4
    ab = S3_MUL[a, b]
    c = S3_INV[ab]
    holonomy = s3_triangle_holonomy(a, b, c)
    assert int(holonomy) == 0
    assert int(defect_class(holonomy)) == 0


def test_s3_holonomy_wrong_orientation_fails():
    a, b = 1, 4
    ab = S3_MUL[a, b]
    c = S3_INV[ab]
    forward = s3_triangle_holonomy(a, b, c)
    wrong = s3_triangle_holonomy(a, c, b)
    assert int(forward) == 0
    assert int(wrong) != 0


def test_oriented_triangles_and_defect_clusters():
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    left = np.array([0, 1, 2, 0, 1, 2])
    right = np.array([1, 2, 0, 3, 3, 3])
    triangles = oriented_triangles(points, left, right)
    classes = np.ones(triangles.shape[0], dtype=np.int64)
    clusters = cluster_defects(triangles, classes, points)
    worldlines = track_defect_worldlines([], clusters)
    assert triangles.shape[1] == 3
    assert clusters
    assert worldlines[0]["event"] == "birth"


def test_defect_clusters_have_disjoint_transitive_triangle_membership():
    angles = np.asarray([0.0, 0.1, 0.2])
    centers = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    points = np.repeat(centers, 3, axis=0)
    triangles = np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
    classes = np.ones(3, dtype=np.int64)

    clusters = cluster_defects(triangles, classes, points)
    memberships = [index for cluster in clusters for index in cluster["triangle_indices"]]

    assert len(clusters) == 1
    assert sorted(memberships) == [0, 1, 2]
    assert len(memberships) == len(set(memberships))


def test_triangle_holonomies_use_reverse_edge_inverse():
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    left = np.array([0, 1, 0])
    right = np.array([1, 2, 2])
    a, b = 1, 4
    c = int(S3_INV[int(S3_MUL[a, b])])
    gauge = np.array([a, b, int(S3_INV[c])])
    triangles = np.array([[0, 1, 2]])

    holonomies = triangle_holonomies(triangles, left, right, gauge)

    assert int(holonomies[0]) == 0


def test_array_holonomy_report_has_claim_boundary():
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    left = np.array([0, 1, 2])
    right = np.array([1, 2, 0])
    gauge = np.array([1, 1, 1])

    report = array_holonomy_report(points, left, right, gauge)

    assert report["triangle_count"] == 1
    assert "matter particles" in report["claim_boundary"]
    assert report["class_counts"]["identity"] + report["defect_triangle_count"] == 1


def test_defect_timeline_reports_persistent_precursor_without_particle_claim():
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    left = np.array([0, 1, 2])
    right = np.array([1, 2, 0])
    gauge = np.array([1, 1, 1])

    report = defect_timeline_report(points, left, right, [(0, gauge), (1, gauge), (2, gauge)])

    assert report["snapshot_count"] == 3
    assert report["persistent_worldline_precursor_receipt"] is True
    assert report["particle_matter_receipt"] is False
    assert report["persistent_worldline_count"] >= 1

    interaction = defect_interaction_report(report)
    particle = particle_likeness_report(report, interaction, max_support_fraction=1.0)
    assert particle["worldline_count"] >= 1
    assert particle["persistent_count"] >= 1
    assert particle["localized_count"] >= 1
    assert interaction["mode"] == "screen_s3_defect_interaction_diagnostic"
    assert "screen_transport_proxy_count" in interaction
    assert particle["particle_matter_receipt"] is False
    assert particle["worldlines"][0]["particle_like"] is False


def test_defect_interaction_reports_screen_transport_and_fusion_proxies():
    timeline = {
        "worldlines": [
            {
                "worldline_id": "w0",
                "observation_count": 3,
                "events": [
                    {"class": "transposition", "transport_distance": None},
                    {"class": "transposition", "transport_distance": 0.2},
                    {"class": "transposition", "transport_distance": 0.3},
                ],
            }
        ],
        "snapshots": [
            {
                "cycle": 0,
                "clusters": [
                    {"cluster_id": "a", "worldline_id": "w0", "holonomy_mode": 1},
                    {"cluster_id": "b", "worldline_id": "w1", "holonomy_mode": int(S3_INV[1])},
                ],
            }
        ],
    }

    report = defect_interaction_report(timeline)
    particle = particle_likeness_report(timeline, report, bulk_localization_pass=False, max_support_fraction=1.0)

    assert report["screen_transport_proxy_count"] == 1
    assert report["fusion_identity_candidate_count"] == 1
    assert report["particle_matter_receipt"] is False
    assert particle["transportable_count"] == 1
    assert particle["particle_matter_receipt"] is False


def test_fusion_conservation_gate_keeps_nonidentity_encounters():
    timeline = {
        "worldlines": [],
        "snapshots": [
            {
                "cycle": 0,
                "clusters": [
                    {"cluster_id": "a", "holonomy_mode": 1},
                    {"cluster_id": "b", "holonomy_mode": 1},
                    {"cluster_id": "c", "holonomy_mode": 2},
                ],
            }
        ],
    }

    report = defect_interaction_report(timeline)

    assert report["fusion_candidate_count"] == 3
    assert report["fusion_identity_candidate_count"] == 1
    assert report["fusion_identity_fraction"] == 1.0 / 3.0
    assert report["fusion_geometrically_verified_candidate_count"] == 0
    assert report["fusion_legacy_unverified_candidate_count"] == 3
    assert report["fusion_conservation_proxy_pass"] is False
    assert any(not row["identity_product"] for row in report["fusion_candidates"])
    assert all(not row["encounter_geometry_verified"] for row in report["fusion_candidates"])


def test_fusion_gate_uses_only_intrinsic_near_encounters_with_provenance():
    def point(angle: float) -> list[float]:
        return [float(np.cos(angle)), float(np.sin(angle)), 0.0]

    timeline = {
        "worldlines": [],
        "snapshots": [
            {
                "cycle": 4,
                "clusters": [
                    {"cluster_id": "near_a", "worldline_id": "wa", "holonomy_mode": 1, "centroid": point(0.0)},
                    {"cluster_id": "near_b", "worldline_id": "wb", "holonomy_mode": 1, "centroid": point(0.1)},
                    {"cluster_id": "far", "worldline_id": "wf", "holonomy_mode": 2, "centroid": point(1.2)},
                ],
            }
        ],
    }

    report = defect_interaction_report(timeline, max_fusion_angular_distance=0.25)

    assert report["fusion_candidate_count"] == 1
    assert report["fusion_geometrically_verified_candidate_count"] == 1
    assert report["fusion_legacy_unverified_candidate_count"] == 0
    assert report["fusion_conservation_proxy_pass"] is True
    assert report["fusion_common_basepoint_transport_receipt"] is False
    assert report["fusion_gauge_covariant_receipt"] is False
    assert report["fusion_theorem_receipt"] is False
    candidate = report["fusion_candidates"][0]
    assert {candidate["left_cluster_id"], candidate["right_cluster_id"]} == {"near_a", "near_b"}
    assert np.isclose(candidate["centroid_angular_distance"], 0.1)
    assert candidate["encounter_geometry_verified"] is True
    assert candidate["candidate_basis"] == "intrinsic_s2_nearest_within_angular_cutoff"


def test_legacy_fusion_pair_without_centroids_is_fail_closed():
    timeline = {
        "worldlines": [],
        "snapshots": [
            {
                "cycle": 0,
                "clusters": [
                    {"cluster_id": "a", "holonomy_mode": 1},
                    {"cluster_id": "b", "holonomy_mode": int(S3_INV[1])},
                ],
            }
        ],
    }

    report = defect_interaction_report(timeline)

    assert report["fusion_candidate_count"] == 1
    assert report["fusion_identity_candidate_count"] == 1
    assert report["fusion_geometrically_verified_candidate_count"] == 0
    assert report["fusion_conservation_proxy_pass"] is False
    assert report["fusion_candidates"][0]["candidate_basis"] == "legacy_missing_centroid_unverified_pair"


def test_worldline_transport_uses_intrinsic_spherical_distance_and_cutoff():
    angle = 0.5
    previous = [
        {
            "cluster_id": "before",
            "class": "transposition",
            "centroid": [1.0, 0.0, 0.0],
        }
    ]
    current = [
        {
            "cluster_id": "after",
            "class": "transposition",
            "centroid": [float(np.cos(angle)), float(np.sin(angle)), 0.0],
        }
    ]

    tracked = track_defect_worldlines(previous, current)

    assert tracked[0]["event"] == "continue"
    assert np.isclose(tracked[0]["transport_distance"], angle, atol=1.0e-12)


def test_public_worldline_matching_is_permutation_stable_and_inherits_ids():
    def point(angle: float) -> list[float]:
        return [float(np.cos(angle)), float(np.sin(angle)), 0.0]

    previous = [
        {
            "cluster_id": "previous_a",
            "worldline_id": "worldline_000010",
            "class": "transposition",
            "centroid": point(0.0),
        },
        {
            "cluster_id": "previous_b",
            "worldline_id": "worldline_000011",
            "class": "transposition",
            "centroid": point(1.0),
        },
    ]
    current = [
        {"cluster_id": "current_a", "class": "transposition", "centroid": point(0.08)},
        {"cluster_id": "new_birth", "class": "threecycle", "centroid": point(2.8)},
        {"cluster_id": "current_b", "class": "transposition", "centroid": point(0.92)},
    ]

    forward = track_defect_worldlines(previous, current)
    permuted = track_defect_worldlines(previous[::-1], [current[2], current[0], current[1]])
    by_cluster_forward = {row["current_cluster_id"]: row for row in forward}
    by_cluster_permuted = {row["current_cluster_id"]: row for row in permuted}

    assert by_cluster_forward == by_cluster_permuted
    assert by_cluster_forward["current_a"]["worldline_id"] == "worldline_000010"
    assert by_cluster_forward["current_b"]["worldline_id"] == "worldline_000011"
    assert by_cluster_forward["new_birth"]["worldline_id"] == "worldline_000012"
    assert by_cluster_forward["new_birth"]["event"] == "birth"


def test_controlled_s3_particle_assay_validates_particle_gate_without_physical_claim():
    report = controlled_s3_particle_assay_report(patch_count=4096, observation_count=5, support_node_count=6)

    assert report["controlled_planted_assay"] is True
    assert report["s3_inverse_identity_pass"] is True
    assert report["interaction_proxy_receipt"] is True
    assert report["fusion_conservation_proxy_pass"] is True
    assert report["particle_detector_positive_receipt"] is True
    assert report["particle_like_count"] >= 1
    assert report["physical_particle_emergence"] is False
    assert "not evidence" in report["claim_boundary"]
