from __future__ import annotations

import numpy as np

from oph_fpe.evidence.controls import mandatory_control_report


def test_mandatory_controls_report_expected_failures():
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    left = np.array([0, 1, 2, 3, 4, 5])
    right = np.array([1, 2, 3, 4, 5, 0])
    initial_left = np.array([0, 0, 0, 0, 0, 0])
    initial_right = np.array([1, 1, 1, 1, 1, 1])
    final_left = np.array([0, 0, 0, 0, 0, 0])
    final_right = np.array([0, 0, 0, 0, 0, 0])

    report = mandatory_control_report(
        requested_controls=[
            "no_repair",
            "shuffled_interfaces",
            "random_same_degree_graph",
            "wrong_s3_orientation",
            "fake_record_rewrite",
        ],
        points=points,
        left=left,
        right=right,
        initial_port_left=initial_left,
        initial_port_right=initial_right,
        final_port_left=final_left,
        final_port_right=final_right,
        seed=3,
    )

    assert set(report["implemented_controls"]) == {
        "fake_record_rewrite",
        "no_repair",
        "random_same_degree_graph",
        "shuffled_interfaces",
        "wrong_s3_orientation",
    }
    assert report["controls"]["no_repair"]["expected_failure_observed"] is True
    assert report["controls"]["wrong_s3_orientation"]["expected_failure_observed"] is True
    assert report["controls"]["fake_record_rewrite"]["expected_failure_observed"] is True
