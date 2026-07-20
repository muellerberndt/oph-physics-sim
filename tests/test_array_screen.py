import json
from pathlib import Path

import numpy as np

from oph_fpe.experiments import load_config
from oph_fpe.bulk.neutral_bulk import build_neutral_observer_views
from oph_fpe.scale import run_array_screen_config
from oph_fpe.scale.array_screen import _advance_record_commit_state, _node_signature
from oph_fpe.scale.bw_array import (
    _attach_h2_neutral_evidence_channels,
    _h2_repair_current_tensor,
    _repair_budget_for_cycle,
)


def test_array_screen_smoke_writes_dimension_report(tmp_path: Path):
    config = load_config(Path("configs/e1_s3_modular_screen_4k.yml"))
    config = dict(config)
    config["run_id"] = "array_smoke"
    config["graph"] = dict(config["graph"], patch_count=256, neighbors=6)
    config["dynamics"] = dict(config["dynamics"], cycles=8, repairs_per_cycle=512)
    config["observables"] = dict(config["observables"])
    config["observables"]["modular_lift"] = {"max_points": 4096, "center_samples": 128}

    result = run_array_screen_config(config, tmp_path)

    assert result["final_phi"] >= 0
    assert result["dimensions"]["distance_source"] == "array_modular_lift_record_history"
    run_path = Path(result["path"])
    assert (run_path / "verifier_receipts.jsonl").exists()
    assert (run_path / "echosahedral_patch_state_report.json").exists()
    assert (run_path / "echosahedral_patch_state.npz").exists()
    state_report = json.loads(
        (run_path / "echosahedral_patch_state_report.json").read_text(encoding="utf-8")
    )
    state_artifact = np.load(run_path / "echosahedral_patch_state.npz")
    assert state_report["ECHOSAHEDRAL_PATCH_STATE_INSTANTIATION_RECEIPT"] is True
    assert state_report["RECORD_SIGNATURE_BINDS_ALL_LOCAL_PORT_STATE_RECEIPT"] is True
    assert state_report["record_signature_binding"]["local_frame_gauge_quotient"] is True
    assert state_report["record_signature_binding"]["final_token_recomputation_match"] is True
    assert state_report["PHYSICAL_STANDARD_MODEL_EMERGENCE_RECEIPT"] is False
    assert state_artifact["patch_port_state"].shape == (256, 12)
    assert state_artifact["record_signature"].shape == (256,)
    assert state_artifact["canonical_record_port_state"].shape == (256, 12)
    assert state_artifact["port_names"].tolist() == [f"P{index}" for index in range(12)]
    assert np.array_equal(
        state_artifact["patch_port_state"][
            state_artifact["edge_left"], state_artifact["left_port"]
        ],
        state_artifact["routed_left_state"],
    )
    assert np.array_equal(
        state_artifact["patch_port_state"][
            state_artifact["edge_right"], state_artifact["right_port"]
        ],
        state_artifact["routed_right_state"],
    )
    assert result["echosahedral_patch_state"]["receipt"] is True


def test_repair_budget_schedule_is_gradual_and_traceable():
    dyn = {
        "repair_budget_schedule": {
            "enabled": True,
            "floor_fraction": 0.2,
            "warmup_fraction": 0.35,
            "peak_fraction": 0.55,
            "peak_width_fraction": 0.2,
            "taper_start_fraction": 0.72,
            "taper_strength": 0.4,
            "jitter_fraction": 0.0,
        }
    }

    budgets = [
        _repair_budget_for_cycle(1000, dyn, cycle=cycle, cycles=20, patch_count=4096, edge_count=8192)
        for cycle in range(20)
    ]

    assert budgets[0] < budgets[8]
    assert budgets[-1] < budgets[8]
    assert min(budgets) > 0
    assert max(budgets) <= 1000


def test_node_signature_distinguishes_equal_sum_port_records():
    left = np.asarray([0, 0], dtype=np.int64)
    right = np.asarray([1, 2], dtype=np.int64)
    remote = np.asarray([0, 0], dtype=np.int16)

    signature_02 = _node_signature(np.asarray([0, 2]), remote, left, right, 3)
    signature_11 = _node_signature(np.asarray([1, 1]), remote, left, right, 3)

    assert int(signature_02[0]) != int(signature_11[0])
    assert np.array_equal(signature_02, _node_signature(np.asarray([0, 2]), remote, left, right, 3))


def test_record_commit_is_revocable_consistent_and_saturating():
    maximum = np.iinfo(np.uint32).max
    previous_signature = np.asarray([10, 20], dtype=np.int64)
    previous_age = np.asarray([maximum, 7], dtype=np.uint32)

    stable, committed = _advance_record_commit_state(
        np.asarray([10, 20]),
        previous_signature,
        previous_age,
        np.asarray([0, 1]),
        commit_cycles=4,
    )

    assert int(stable[0]) == maximum
    assert committed.tolist() == [True, False]

    stable, committed = _advance_record_commit_state(
        np.asarray([11, 20]),
        previous_signature,
        stable,
        np.asarray([0, 0]),
        commit_cycles=4,
    )
    assert int(stable[0]) == 1
    assert committed.tolist() == [False, True]


def test_strict_neutral_repair_tensor_ignores_cap_driven_modular_fields():
    support = np.asarray([0, 1, 2], dtype=np.int64)
    base = {
        "repair_load": np.asarray([0.1, 0.2, 0.3]),
        "cumulative_repair_load": np.asarray([1.0, 2.0, 3.0]),
        "local_mismatch_density": np.asarray([0.3, 0.2, 0.1]),
        "stable_count": np.asarray([3.0, 4.0, 5.0]),
        "committed_mask": np.asarray([1.0, 1.0, 0.0]),
        "s3_class_density": np.asarray([0.0, 0.5, 1.0]),
        "modular_depth": np.asarray([1.0, 2.0, 3.0]),
        "modular_time": np.asarray([0.1, 0.2, 0.3]),
    }
    cap_mutated = {
        **base,
        "modular_depth": np.asarray([1.0e9, -1.0e9, 7.0e8]),
        "modular_time": np.asarray([-8.0e8, 6.0e8, 4.0e8]),
    }

    baseline = _h2_repair_current_tensor(support, raw_history=[], current_raw_fields=base, width=64)
    mutated = _h2_repair_current_tensor(support, raw_history=[], current_raw_fields=cap_mutated, width=64)

    assert np.array_equal(baseline, mutated)


def test_h2_neutral_evidence_channels_are_emitted_without_chart_ids():
    observer_rows = [
        {
            "view_type": "patch_observer",
            "observer_id": 17,
            "support_nodes": [0, 1, 2, 3],
            "object_packet_histogram": {"3": 0.5, "9": 0.5},
            "transition_history_descriptor": {
                "steps": [
                    {"record_family": 1, "checkpoint_class": 0, "s3_sector_class": 2, "repair_load_bucket": 0},
                    {"record_family": 1, "checkpoint_class": 1, "s3_sector_class": 3, "repair_load_bucket": 2},
                    {"record_family": 4, "checkpoint_class": 1, "s3_sector_class": 3, "repair_load_bucket": 3},
                ],
            },
            "transition_history_persistence": 1,
            "transition_history_histograms": {
                "local_transition_token": {"11": 0.75, "19": 0.25},
                "local_transition_token_persistent": {"11": 1.0},
            },
            "repair_response_spectrum": [0.1, -0.2, 0.3],
            "perturb_resettle_signature": [1.0, 2.0],
            "sector_change_signature": [1.0, 1.0, 1.0, 0.75],
        }
    ]
    base_raw = {
        "record_signature": np.asarray([11, 11, 14, 18, 20], dtype=float),
        "stable_count": np.asarray([1, 2, 3, 4, 5], dtype=float),
        "committed_mask": np.asarray([0, 1, 1, 1, 0], dtype=float),
        "repair_load": np.asarray([0.0, 0.2, 0.5, 0.8, 1.0], dtype=float),
        "cumulative_repair_load": np.asarray([1, 2, 4, 8, 16], dtype=float),
        "local_mismatch_density": np.asarray([0.6, 0.4, 0.2, 0.1, 0.0], dtype=float),
        "modular_depth": np.asarray([1, 1, 2, 3, 5], dtype=float),
        "modular_time": np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float),
        "s3_class_density": np.asarray([0, 1, 1, 2, 3], dtype=float),
        "s3_sector_class": np.asarray([0, 1, 2, 3, 4], dtype=float),
    }
    late_raw = {
        **base_raw,
        "record_signature": np.asarray([11, 12, 14, 19, 20], dtype=float),
        "repair_load": np.asarray([0.1, 0.4, 0.5, 0.9, 1.0], dtype=float),
        "cumulative_repair_load": np.asarray([2, 3, 6, 9, 18], dtype=float),
        "local_mismatch_density": np.asarray([0.5, 0.3, 0.2, 0.0, 0.0], dtype=float),
    }
    object_cfg = {
        "transition_affinity_fields": [
            "checkpoint_class",
            "record_family",
            "s3_sector_class",
            "repair_load_bucket",
            "cumulative_repair_load_bucket",
        ],
        "transition_history_key_fields": [
            "record_family",
            "checkpoint_class",
            "s3_sector_class",
            "repair_load_bucket",
        ],
        "transition_bins": 4,
        "record_family_modulus": 16,
    }

    _attach_h2_neutral_evidence_channels(
        observer_rows,
        history_raw_fields=[base_raw, late_raw],
        current_raw_fields=late_raw,
        object_cfg=object_cfg,
    )

    row = observer_rows[0]
    assert row["h2_neutral_evidence_schema"] == "observer_local_h2_neutral_evidence_v1"
    assert row["local_boundary_packet_hash_histogram"]
    assert row["overlap_correspondence_histogram"]
    assert row["port_pair_lag_histogram"]
    assert len(row["repair_current_tensor"]) == 128
    assert len(row["perturbation_response_tensor"]) == 128
    assert row["first_passage_time_histogram"]
    leaked_support_ids = {str(value) for value in row["support_nodes"]}
    assert not (set(row["local_boundary_packet_hash_histogram"]) & leaked_support_ids)
    assert not (set(row["overlap_correspondence_histogram"]) & leaked_support_ids)

    neutral_view = build_neutral_observer_views(observer_rows)[0]
    assert neutral_view.boundary_packet_hash_hist.sum() > 0.0
    assert neutral_view.overlap_correspondence_hist.sum() == 0.0
    assert neutral_view.port_pair_lag_hist.sum() > 0.0
    assert np.linalg.norm(neutral_view.repair_current_tensor) > 0.0
    # The legacy H2 summary is still exported, but it cannot masquerade as the
    # real paired perturb-resettle producer in claim-bearing geometry.
    assert np.linalg.norm(neutral_view.perturbation_response_tensor) == 0.0
    assert neutral_view.first_passage_response_hist.sum() > 0.0
