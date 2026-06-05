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
