"""Frozen tests for the issue-634 stage-1 event-complex producer."""

import hashlib
import json
from pathlib import Path

import numpy as np

from oph_fpe.local_domain.stage1_event_complex import (
    MAIN_CONFIG,
    apply_frame,
    causal_certificates,
    chart_frame,
    frame_transition,
    orientation_certificate,
    pair_classes_capped,
    prescribed_chart_rank_certificate,
    produce_stage1_receipt,
    refuse_forbidden_config,
    local_domain_source_sha256,
    time_orientation_certificate,
    verify_transition,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "local_domain"

SMALL_CONFIG = {
    "carrier_count": 2_048,
    "cycles": 16,
    "seed": 20260751,
    "observer_count": 16,
    "observer_support_size": 96,
    "observer_samples": 6,
    "observer_cross_reads": True,
    "snapshot_coverage": "spanning",
    "geometry_transport": "held_out_flow",
}


def _projection_capture(
    *, diagnostic: float = 1.0, event_prefix: str = "runtime-a"
) -> dict:
    first_key = f"{event_prefix}-event-0"
    second_key = f"{event_prefix}-event-1"
    return {
        "capture_sha256": f"sha256:{diagnostic}",
        "unused_float_diagnostic": diagnostic,
        "postrun_capture": {
            "carrier_port_trajectories": [{}, {}],
            "semantic_events": [
                {
                    "event_key": first_key,
                    "source_sequence_index": 0,
                    "observer_token": "observer-0",
                    "visible_footprint": ["carrier-00000:port-0"],
                    "unused_float": diagnostic,
                },
                {
                    "event_key": second_key,
                    "source_sequence_index": 1,
                    "observer_token": "observer-0",
                    "visible_footprint": ["carrier-00001:port-0"],
                    "unused_float": diagnostic,
                },
            ],
            "raw_ancestry_relations": [
                {
                    "parent_event_id": first_key,
                    "child_event_id": second_key,
                }
            ],
            "raw_overlap_relations": [
                {
                    "overlap_id": "seam-0",
                    "left_carrier_id": "carrier-00000",
                    "right_carrier_id": "carrier-00001",
                    "left_ports": [0],
                    "right_ports": [0],
                    "orientation_signs": [-1],
                    "visible_to_observer_tokens": ["observer-0"],
                    "interface_algebra_sha256": "sha256:a",
                    "unused_float": diagnostic,
                }
            ],
        },
    }


def test_source_projection_ignores_unconsumed_float_diagnostics_and_tampers():
    first = _projection_capture(diagnostic=1.0, event_prefix="float-hash-a")
    second = _projection_capture(diagnostic=2.0, event_prefix="float-hash-b")
    assert local_domain_source_sha256(first, SMALL_CONFIG) == (
        local_domain_source_sha256(second, SMALL_CONFIG)
    )

    second["postrun_capture"]["raw_ancestry_relations"][0][
        "child_event_id"
    ] = second["postrun_capture"]["semantic_events"][0]["event_key"]
    assert local_domain_source_sha256(first, SMALL_CONFIG) != (
        local_domain_source_sha256(second, SMALL_CONFIG)
    )

    second = _projection_capture(diagnostic=2.0, event_prefix="float-hash-b")
    second["postrun_capture"]["carrier_port_trajectories"].append({})
    assert local_domain_source_sha256(first, SMALL_CONFIG) != (
        local_domain_source_sha256(second, SMALL_CONFIG)
    )

    second = _projection_capture(diagnostic=2.0, event_prefix="float-hash-b")
    second["postrun_capture"]["semantic_events"][1]["event_key"] = second[
        "postrun_capture"
    ]["semantic_events"][0]["event_key"]
    assert local_domain_source_sha256(first, SMALL_CONFIG) != (
        local_domain_source_sha256(second, SMALL_CONFIG)
    )

    second = _projection_capture(diagnostic=2.0, event_prefix="float-hash-b")
    second["postrun_capture"]["raw_overlap_relations"][0][
        "orientation_signs"
    ] = [1]
    assert local_domain_source_sha256(first, SMALL_CONFIG) != (
        local_domain_source_sha256(second, SMALL_CONFIG)
    )


def _chain_complex() -> dict:
    return {
        "table": {
            "count": 4,
            "depth": {0: 0, 1: 1, 2: 2, 3: 0},
            "reachable": {0: set(), 1: {0}, 2: {0, 1}, 3: set()},
            "footprints": [set(), set(), set(), set()],
        },
        "edges": [(0, 1), (1, 2)],
        "sequence": [0, 1, 2, 3],
        "tokens": ["a", "a", "a", "b"],
    }


def test_causal_certificates_exact_on_chain():
    certificates = causal_certificates(_chain_complex())
    assert certificates["acyclic"]
    assert certificates["antisymmetric"]
    assert certificates["edge_sequence_strict"]
    assert certificates["edge_depth_strict"]
    assert certificates["strict_time_function"]


def test_causal_certificates_detect_cycle_and_depth_violation():
    broken = _chain_complex()
    broken["table"]["reachable"] = {0: {1}, 1: {0}, 2: set(), 3: set()}
    certificates = causal_certificates(broken)
    assert not certificates["antisymmetric"]

    flat = _chain_complex()
    flat["table"]["depth"] = {0: 0, 1: 0, 2: 1, 3: 0}
    certificates = causal_certificates(flat)
    assert not certificates["edge_depth_strict"]
    assert not certificates["strict_time_function"]


def test_causal_certificates_fail_on_unresolved_or_duplicate_identifiers():
    complex_data = _chain_complex()
    complex_data["missing_ancestry_endpoint_count"] = 1
    assert not causal_certificates(complex_data)[
        "ancestry_endpoints_resolved"
    ]
    complex_data["missing_ancestry_endpoint_count"] = 0
    complex_data["duplicate_event_key_count"] = 1
    assert not causal_certificates(complex_data)["event_keys_unique"]


def test_prescribed_chart_rank_gate():
    rng = np.random.default_rng(634)
    chart = rng.normal(size=(200, 4))
    assert prescribed_chart_rank_certificate(chart)[
        "prescribed_four_coordinate_chart_nondegenerate"
    ]
    collapsed = chart.copy()
    collapsed[:, 3] = 0.0
    assert not prescribed_chart_rank_certificate(collapsed)[
        "prescribed_four_coordinate_chart_nondegenerate"
    ]
    dependent = chart.copy()
    dependent[:, 0] = dependent[:, 1]
    certificate = prescribed_chart_rank_certificate(dependent)
    assert certificate["time_spread"] > 0.0
    assert certificate["spatial_rank_ratio"] > 0.0
    assert not certificate[
        "prescribed_four_coordinate_chart_nondegenerate"
    ]
    assert certificate["full_chart_rank_ratio"] < certificate[
        "full_chart_rank_ratio_floor"
    ]


def test_frame_transitions_are_exact_and_composable():
    rng = np.random.default_rng(575)
    chart = rng.normal(size=(120, 4)) * np.array([10.0, 0.02, 0.02, 0.02])
    events_a = list(range(0, 80))
    events_b = list(range(40, 120))
    events_c = list(range(20, 100))
    frames = [chart_frame(chart, e) for e in (events_a, events_b, events_c)]
    coords = [apply_frame(chart, f) for f in frames]
    shared_ab = list(range(40, 80))
    matrix_ab, offset_ab = frame_transition(frames[0], frames[1])
    fit = verify_transition(
        coords[0][shared_ab], coords[1][shared_ab], matrix_ab, offset_ab
    )
    assert fit["residual_rel"] < 1.0e-9
    assert fit["supported"]

    matrix_bc, offset_bc = frame_transition(frames[1], frames[2])
    matrix_ac, offset_ac = frame_transition(frames[0], frames[2])
    composed = coords[0] @ matrix_ab + offset_ab
    composed = composed @ matrix_bc + offset_bc
    direct = coords[0] @ matrix_ac + offset_ac
    assert float(np.abs(composed - direct).max()) < 1.0e-9


def test_flipped_axis_breaks_transition_verification():
    rng = np.random.default_rng(11)
    chart = rng.normal(size=(100, 4)) * np.array([10.0, 0.02, 0.02, 0.02])
    events = list(range(100))
    frame = chart_frame(chart, events)
    coords = apply_frame(chart, frame)
    matrix, offset = frame_transition(frame, frame)
    flipped = coords.copy()
    flipped[:, 1] = -flipped[:, 1]
    fit = verify_transition(flipped, coords, matrix, offset)
    assert fit["residual_rel"] > 0.25


def _atlas_with_signs(signs: dict[tuple[int, int], float]) -> dict:
    overlaps = []
    for (a, b), sign in signs.items():
        matrix = np.diag([1.0, sign, 1.0, 1.0])
        overlaps.append(
            {"pair": (a, b), "events": [], "fit": {"supported": True, "matrix": matrix}}
        )
    return {"seed_count": 3, "overlaps": overlaps}


def test_orientation_certificate_consistent_and_odd_cycle():
    consistent = _atlas_with_signs({(0, 1): 1.0, (1, 2): -1.0, (0, 2): -1.0})
    result = orientation_certificate(consistent)
    assert result["orientable"]
    odd = _atlas_with_signs({(0, 1): -1.0, (1, 2): -1.0, (0, 2): -1.0})
    assert not orientation_certificate(odd)["orientable"]


def test_time_orientation_certificate_floor():
    good = _atlas_with_signs({(0, 1): 1.0})
    assert time_orientation_certificate(good)["time_orientable"]
    reversed_time = _atlas_with_signs({(0, 1): 1.0})
    reversed_time["overlaps"][0]["fit"]["matrix"] = np.diag([-1.0, 1.0, 1.0, 1.0])
    assert not time_orientation_certificate(reversed_time)["time_orientable"]


def test_pair_classes_capped_stride():
    table = {
        "count": 6,
        "reachable": {0: set(), 1: {0}, 2: {0, 1}, 3: set(), 4: set(), 5: set()},
    }
    pairs = pair_classes_capped(table, cap=4)
    assert pairs["causal_total"] == 3
    assert pairs["spacelike_total"] == 12
    assert len(pairs["spacelike"]) <= 4
    assert pairs["spacelike_stride"] == 3


def test_forbidden_config_refusal():
    assert refuse_forbidden_config({"target_signature": "x"}) == [
        "target_signature"
    ]
    receipt = produce_stage1_receipt(
        config={**SMALL_CONFIG, "einstein_conclusion": "yes"}, run_controls=False
    )
    assert receipt["verdict"] == "REFUSED"
    assert receipt["blockers"] == ["forbidden_config_key:einstein_conclusion"]


def test_small_capture_receipt_frozen_shape():
    receipt = produce_stage1_receipt(
        config=SMALL_CONFIG, seed_count=4, run_controls=False
    )
    assert receipt["verdict"] == "NOT_ATTAINED"
    assert receipt["blockers"] == [
        "clause_failed:held_out_feature_form_inertia_1_3"
    ]
    clauses = receipt["clause_verdicts"]
    assert clauses["causal_order_acyclic"]
    assert clauses["strict_time_function"]
    assert clauses["prescribed_four_coordinate_chart_nondegenerate"]
    assert receipt["prescribed_chart_construction"] == {
        "ancestry_depth_coordinate_count": 1,
        "spectral_embedding_coordinate_count": 3,
        "total_coordinate_count": 4,
        "status": (
            "prescribed feature construction; no dimension-selection "
            "claim"
        ),
    }
    assert receipt["declared_acceptance_gates"][
        "held_out_feature_form_target_inertia"
    ] == [1, 3]
    assert clauses["atlas_covers_all_events"]
    assert clauses["atlas_nerve_connected"]
    assert clauses["transitions_supported"]
    assert clauses["cocycles_within_gate"]
    assert clauses["orientation_assignment_exists"]
    assert clauses["time_orientation_consistent"]
    assert all(
        value < 1.0e-9
        for value in receipt["atlas"]["transition_residuals"].values()
    )


def test_frozen_main_receipt_binding():
    receipt_path = DATA_DIR / "stage1_receipt.json"
    manifest_path = DATA_DIR / "manifest.json"
    arrays_path = DATA_DIR / "stage1_arrays.npz.gz"
    assert receipt_path.exists() and manifest_path.exists() and arrays_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_bytes = receipt_path.read_bytes()
    assert manifest["receipt_sha256"] == "sha256:" + hashlib.sha256(
        receipt_bytes
    ).hexdigest()
    assert manifest["arrays_sha256"] == "sha256:" + hashlib.sha256(
        arrays_path.read_bytes()
    ).hexdigest()
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    assert receipt["schema"] == "oph.local-domain-stage1.v1"
    assert receipt["issue"] == 634
    assert receipt["physical_promotion_allowed"] is False
    assert receipt["main_config"] == MAIN_CONFIG
    assert receipt["source_projection_sha256"].startswith("sha256:")
    assert receipt["verdict"] == "ATTAINED"
    assert receipt["blockers"] == []
    assert all(receipt["clause_verdicts"].values())
    assert receipt["controls_fail_closed"] is True
    assert tuple(receipt["held_out_quadratic_fit"]["inertia"]) == (1, 3)
    assert "legacy_ladder_binding" not in receipt
    promotion = receipt["continuum_promotion"]
    assert promotion["status"] == "BLOCKED"
    assert promotion["failed_conditions"]
    assert not any(
        condition["holds"]
        for condition in promotion["failed_conditions"].values()
    )
    assert "cone_margins_nonnegative" in promotion["failed_conditions"]
    assert "cofinal_refinement_family" in promotion["failed_conditions"]
