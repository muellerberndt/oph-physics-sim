from __future__ import annotations

from typing import Any

import numpy as np

from oph_fpe.defects.array_s3_holonomy import S3_INV, S3_MUL, defect_class, s3_triangle_holonomy
from oph_fpe.observers.objects import RecordFamily, observer_object_report


def mandatory_control_report(
    *,
    requested_controls: list[str],
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    initial_port_left: np.ndarray,
    initial_port_right: np.ndarray,
    final_port_left: np.ndarray,
    final_port_right: np.ndarray,
    object_rows: list[dict[str, Any]] | None = None,
    seed: int = 1,
) -> dict[str, Any]:
    requested = [str(control) for control in requested_controls]
    controls: dict[str, Any] = {}
    if "no_repair" in requested:
        controls["no_repair"] = _no_repair_control(initial_port_left, initial_port_right, final_port_left, final_port_right)
    if "shuffled_interfaces" in requested:
        controls["shuffled_interfaces"] = _shuffled_interfaces_control(final_port_left, final_port_right, seed)
    if "random_same_degree_graph" in requested:
        controls["random_same_degree_graph"] = _random_same_degree_graph_control(points, left, right, seed)
    if "wrong_s3_orientation" in requested:
        controls["wrong_s3_orientation"] = _wrong_s3_orientation_control()
    if "fake_record_rewrite" in requested:
        controls["fake_record_rewrite"] = _fake_record_rewrite_control(object_rows or [])
    return {
        "mode": "mandatory_negative_controls",
        "implemented_controls": sorted(controls),
        "controls": controls,
        "all_expected_failures_observed": bool(controls) and all(bool(row.get("expected_failure_observed")) for row in controls.values()),
        "claim_boundary": "negative-control receipts; required before any 3D-bulk or early-universe claim",
    }


def _no_repair_control(initial_left: np.ndarray, initial_right: np.ndarray, final_left: np.ndarray, final_right: np.ndarray) -> dict[str, Any]:
    initial_phi = int(np.sum(initial_left != initial_right))
    final_phi = int(np.sum(final_left != final_right))
    return {
        "initial_phi_without_repair": initial_phi,
        "final_phi_with_repair": final_phi,
        "expected_failure_observed": bool(initial_phi > final_phi and initial_phi > 0),
        "failure_mode": "Phi does not settle when repair is absent",
    }


def _shuffled_interfaces_control(final_left: np.ndarray, final_right: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    shuffled_right = final_right[rng.permutation(final_right.size)]
    shuffled_phi = int(np.sum(final_left != shuffled_right))
    threshold = 0.25 * int(final_left.size)
    return {
        "shuffled_phi": shuffled_phi,
        "edge_count": int(final_left.size),
        "expected_failure_observed": bool(shuffled_phi > threshold),
        "failure_mode": "interface label shuffle destroys overlap agreement",
    }


def _random_same_degree_graph_control(points: np.ndarray, left: np.ndarray, right: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    degree = np.bincount(np.concatenate([left, right]), minlength=points.shape[0])
    stubs = np.repeat(np.arange(points.shape[0], dtype=np.int64), degree)
    if stubs.size % 2:
        stubs = stubs[:-1]
    shuffled = stubs[rng.permutation(stubs.size)]
    random_left = shuffled[0::2]
    random_right = shuffled[1::2]
    original_chord = _mean_chord(points, left, right)
    random_chord = _mean_chord(points, random_left, random_right)
    random_degree = np.bincount(np.concatenate([random_left, random_right]), minlength=points.shape[0])
    degree_l1_error = int(np.sum(np.abs(random_degree - degree)))
    return {
        "degree_l1_error": degree_l1_error,
        "original_mean_chord": original_chord,
        "random_mean_chord": random_chord,
        "expected_failure_observed": bool(degree_l1_error == 0 and random_chord > 1.25 * max(original_chord, 1e-12)),
        "failure_mode": "same-degree random graph destroys screen-local adjacency geometry",
    }


def _wrong_s3_orientation_control() -> dict[str, Any]:
    a, b = 1, 4
    c = int(S3_INV[int(S3_MUL[a, b])])
    forward = int(s3_triangle_holonomy(a, b, c))
    wrong = int(s3_triangle_holonomy(a, c, b))
    return {
        "forward_holonomy": forward,
        "wrong_orientation_holonomy": wrong,
        "wrong_orientation_class": int(defect_class(wrong)),
        "expected_failure_observed": bool(forward == 0 and wrong != 0),
        "failure_mode": "nonabelian S3 orientation reversal changes holonomy",
    }


def _fake_record_rewrite_control(object_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if object_rows:
        first = object_rows[0]
        families = [
            RecordFamily(
                object_id="rewrite_a",
                support_nodes=[0],
                record_signature=int(first.get("record_signature", 1)),
                persistence=int(first.get("persistence", 1)),
                overlap_agreement=1.0,
                repair_history_hash=str(first.get("repair_history_hash", "same")),
                counterfactual_stability=1.0,
            ),
            RecordFamily(
                object_id="rewrite_b",
                support_nodes=[1],
                record_signature=int(first.get("record_signature", 1)) + 1,
                persistence=int(first.get("persistence", 1)),
                overlap_agreement=1.0,
                repair_history_hash=str(first.get("repair_history_hash", "same")),
                counterfactual_stability=1.0,
            ),
        ]
    else:
        families = [
            RecordFamily("rewrite_a", [0], 1, 1, 1.0, "same", 1.0),
            RecordFamily("rewrite_b", [1], 2, 1, 1.0, "same", 1.0),
        ]
    detected = bool(observer_object_report(families, [])["bad_record_rewrite_detected"])
    return {
        "bad_record_rewrite_detected": detected,
        "expected_failure_observed": detected,
        "failure_mode": "record-family verifier rejects same-history/different-signature rewrite",
    }


def _mean_chord(points: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(points[left] - points[right], axis=1)))
