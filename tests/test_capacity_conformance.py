"""Tests for the capacity accounting and scheduler-class conformance lane.

Covers the private/shared spend split, the clause-by-clause class-membership
receipts against PR-60..PR-63, the fail-closed sampled transactional
verifier, and the kernel integrations.  Mutation cases plant budget,
locality, mismatch-increase, and non-descending defects and require the
receipts to catch each one.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from oph_fpe.experiments import load_config
from oph_fpe.gauge.covariant_overlap import (
    absorb_discrepancy_into_gauge,
    covariant_discrepancy,
    covariant_mismatch_mask,
    repair_covariant_port_pairs,
)
from oph_fpe.scale.array_screen import run_array_screen_config
from oph_fpe.scale.capacity_conformance import (
    CapacityConformanceTracker,
    named_conformance_stream,
    replay_edge_repair_transaction,
    snapshot_chosen_edge_state,
)


def _fixture_state(seed: int = 7, edges: int = 12, patches: int = 6):
    rng = np.random.default_rng(seed)
    left = (np.arange(edges) % patches).astype(np.int64)
    right = ((np.arange(edges) + 1) % patches).astype(np.int64)
    port_left = rng.integers(0, 6, edges).astype(np.int16)
    port_right = rng.integers(0, 6, edges).astype(np.int16)
    gauge = rng.integers(0, 6, edges).astype(np.int16)
    return left, right, port_left, port_right, gauge


def _tracker(left, right, patches: int = 6, sample: int = 8):
    return CapacityConformanceTracker(
        patch_count=patches,
        edge_left=left,
        edge_right=right,
        group_name="S3",
        group_order=6,
        seed=99,
        sample_edges_per_cycle=sample,
        engine="test",
    )


def _phi(port_left, port_right, gauge) -> tuple[int, np.ndarray]:
    mask = covariant_mismatch_mask(
        port_left, port_right, gauge, group_name="S3", group_order=6
    )
    return int(mask.sum()), mask


def test_private_shared_split_sums_to_total_repairs():
    left, right, port_left, port_right, gauge = _fixture_state()
    tracker = _tracker(left, right)
    phi_before, mask = _phi(port_left, port_right, gauge)
    chosen = np.flatnonzero(mask)[:5].astype(np.int64)
    assert chosen.size == 5
    before = snapshot_chosen_edge_state(chosen, port_left, port_right, gauge)
    direction = np.array([True, False, True, True, False])
    repair_covariant_port_pairs(
        port_left, port_right, gauge, chosen, direction, group_name="S3", group_order=6
    )
    phi_after, mask_after = _phi(port_left, port_right, gauge)

    row = tracker.record_cycle(
        cycle=0,
        phi_before=phi_before,
        phi_after=phi_after,
        before=before,
        direction=direction,
        port_left=port_left,
        port_right=port_right,
        gauge=gauge,
        mismatches_after=mask_after,
        repair_budget=8,
        sector_link_writes_reported=0,
        readback_drive_edges=0,
    )

    assert row["private_spend_actions"] == 5
    assert row["shared_spend_actions"] == 0
    assert row["total_repair_actions"] == 5
    assert (
        row["private_spend_actions"] + row["shared_spend_actions"]
        == row["total_repair_actions"]
    )
    assert row["spend_split_sums_to_total"] is True
    assert (
        row["left_endpoint_writes"] + row["right_endpoint_writes"]
        == row["private_spend_actions"]
    )
    assert row["far_endpoint_writes"] == 0
    assert row["observed_schedule_in_declared_class"] is True
    assert row["claim_kind"] == "measured"
    per_patch = tracker.per_patch_spend()
    assert int(per_patch["private"].sum()) == row["private_spend_actions"]
    assert float(per_patch["shared"].sum()) == float(row["shared_spend_actions"])


def test_synthetic_all_gauge_cycle_yields_shared_only():
    left, right, port_left, port_right, gauge = _fixture_state(seed=11)
    tracker = _tracker(left, right)
    phi_before, mask = _phi(port_left, port_right, gauge)
    chosen = np.flatnonzero(mask).astype(np.int64)
    assert chosen.size >= 3
    before = snapshot_chosen_edge_state(chosen, port_left, port_right, gauge)
    delta = covariant_discrepancy(
        port_left[chosen],
        port_right[chosen],
        gauge[chosen],
        group_name="S3",
        group_order=6,
    )
    # The link absorbs the full discrepancy; no endpoint slot is touched.
    absorb_discrepancy_into_gauge(
        gauge, chosen, delta, group_name="S3", group_order=6
    )
    phi_after, mask_after = _phi(port_left, port_right, gauge)
    assert phi_after == 0

    row = tracker.record_cycle(
        cycle=0,
        phi_before=phi_before,
        phi_after=phi_after,
        before=before,
        direction=np.ones(chosen.size, dtype=bool),
        port_left=port_left,
        port_right=port_right,
        gauge=gauge,
        mismatches_after=mask_after,
        repair_budget=int(chosen.size),
        sector_link_writes_reported=int(chosen.size),
        readback_drive_edges=0,
    )

    assert row["private_spend_actions"] == 0
    assert row["shared_spend_actions"] == int(chosen.size)
    assert row["total_repair_actions"] == int(chosen.size)
    assert row["shared_reported_matches_measured"] is True
    assert row["observed_schedule_in_declared_class"] is True
    per_patch = tracker.per_patch_spend()
    assert int(per_patch["private"].sum()) == 0
    assert float(per_patch["shared"].sum()) == float(chosen.size)


def test_budget_violation_is_caught_and_fails_run_receipt():
    left, right, port_left, port_right, gauge = _fixture_state(seed=13)
    tracker = _tracker(left, right)
    phi_before, mask = _phi(port_left, port_right, gauge)
    chosen = np.flatnonzero(mask)[:4].astype(np.int64)
    before = snapshot_chosen_edge_state(chosen, port_left, port_right, gauge)
    direction = np.ones(chosen.size, dtype=bool)
    repair_covariant_port_pairs(
        port_left, port_right, gauge, chosen, direction, group_name="S3", group_order=6
    )
    phi_after, mask_after = _phi(port_left, port_right, gauge)

    row = tracker.record_cycle(
        cycle=0,
        phi_before=phi_before,
        phi_after=phi_after,
        before=before,
        direction=direction,
        port_left=port_left,
        port_right=port_right,
        gauge=gauge,
        mismatches_after=mask_after,
        repair_budget=int(chosen.size) - 1,
        sector_link_writes_reported=0,
        readback_drive_edges=0,
    )

    assert row["clauses"]["within_declared_cycle_budget"] is False
    assert row["observed_schedule_in_declared_class"] is False
    report = tracker.run_report()
    assert report["all_cycles_in_declared_class"] is False
    assert report["violating_cycles"] == [0]
    assert report["SCHEDULER_CLASS_CONFORMANCE_RECEIPT"] is False


def test_locality_violation_is_caught():
    left, right, port_left, port_right, gauge = _fixture_state(seed=17)
    tracker = _tracker(left, right, sample=0)
    phi_before, mask = _phi(port_left, port_right, gauge)
    chosen = np.flatnonzero(mask)[:3].astype(np.int64)
    before = snapshot_chosen_edge_state(chosen, port_left, port_right, gauge)
    direction = np.ones(chosen.size, dtype=bool)
    repair_covariant_port_pairs(
        port_left, port_right, gauge, chosen, direction, group_name="S3", group_order=6
    )
    # Mutant: a second write lands on the far endpoint of the first edge.
    mutant_edge = int(chosen[0])
    port_right[mutant_edge] = np.int16((int(port_right[mutant_edge]) + 1) % 6)
    phi_after, mask_after = _phi(port_left, port_right, gauge)

    row = tracker.record_cycle(
        cycle=0,
        phi_before=phi_before,
        phi_after=phi_after,
        before=before,
        direction=direction,
        port_left=port_left,
        port_right=port_right,
        gauge=gauge,
        mismatches_after=mask_after,
        repair_budget=8,
        sector_link_writes_reported=0,
        readback_drive_edges=0,
    )

    assert row["clauses"]["single_slot_locality"] is False
    assert row["clauses"]["unit_expected_capacity_per_step"] is False
    assert row["observed_schedule_in_declared_class"] is False


def test_shared_plus_private_write_on_one_step_fails_capacity_clause():
    left, right, port_left, port_right, gauge = _fixture_state(seed=18)
    tracker = _tracker(left, right, sample=0)
    phi_before, mask = _phi(port_left, port_right, gauge)
    chosen = np.flatnonzero(mask)[:1].astype(np.int64)
    assert chosen.size == 1
    before = snapshot_chosen_edge_state(chosen, port_left, port_right, gauge)
    direction = np.ones(chosen.size, dtype=bool)
    repair_covariant_port_pairs(
        port_left, port_right, gauge, chosen, direction, group_name="S3", group_order=6
    )
    edge = int(chosen[0])
    gauge[edge] = np.int16((int(gauge[edge]) + 1) % 6)
    phi_after, mask_after = _phi(port_left, port_right, gauge)

    row = tracker.record_cycle(
        cycle=0,
        phi_before=phi_before,
        phi_after=phi_after,
        before=before,
        direction=direction,
        port_left=port_left,
        port_right=port_right,
        gauge=gauge,
        mismatches_after=mask_after,
        repair_budget=1,
        sector_link_writes_reported=1,
        readback_drive_edges=0,
    )

    assert row["two_register_writes"] == 1
    assert row["max_capacity_per_scheduled_step"] == 2
    assert row["clauses"]["single_slot_locality"] is False
    assert row["clauses"]["unit_expected_capacity_per_step"] is False
    assert row["observed_schedule_in_declared_class"] is False


def test_mismatch_increase_is_caught():
    left, right, port_left, port_right, gauge = _fixture_state(seed=19)
    tracker = _tracker(left, right, sample=0)
    # Make the first two edges consistent so the mutant has records to break.
    for edge in (0, 1):
        port_left[edge] = np.int16(_transported(int(port_right[edge]), int(gauge[edge])))
    phi_before, mask = _phi(port_left, port_right, gauge)
    consistent = np.flatnonzero(~mask)
    assert consistent.size >= 2
    chosen = np.flatnonzero(mask)[:1].astype(np.int64)
    before = snapshot_chosen_edge_state(chosen, port_left, port_right, gauge)
    direction = np.ones(chosen.size, dtype=bool)
    repair_covariant_port_pairs(
        port_left, port_right, gauge, chosen, direction, group_name="S3", group_order=6
    )
    # Mutant: out-of-schedule writes break two previously consistent edges,
    # so the cycle raises the mismatch it was supposed to lower.
    for broken in consistent[:2].tolist():
        port_left[broken] = np.int16((int(port_left[broken]) + 1) % 6)
    phi_after, mask_after = _phi(port_left, port_right, gauge)
    assert phi_after > phi_before

    row = tracker.record_cycle(
        cycle=0,
        phi_before=phi_before,
        phi_after=phi_after,
        before=before,
        direction=direction,
        port_left=port_left,
        port_right=port_right,
        gauge=gauge,
        mismatches_after=mask_after,
        repair_budget=8,
        sector_link_writes_reported=0,
        readback_drive_edges=0,
    )

    assert row["decoupled_edge_repair_interference"] != 0
    assert row["edge_slot_decoupling_ok"] is False
    assert row["clauses"]["mismatch_nonincrease"] is False
    assert row["observed_schedule_in_declared_class"] is False


def test_sampled_transactional_verifier_catches_planted_nondescending_step():
    left, right, port_left, port_right, gauge = _fixture_state(seed=23)
    tracker = _tracker(left, right, sample=8)
    phi_before, mask = _phi(port_left, port_right, gauge)
    chosen = np.flatnonzero(mask)[:2].astype(np.int64)
    before = snapshot_chosen_edge_state(chosen, port_left, port_right, gauge)
    direction = np.ones(chosen.size, dtype=bool)
    repair_covariant_port_pairs(
        port_left, port_right, gauge, chosen, direction, group_name="S3", group_order=6
    )
    # Mutant: the first repaired edge is rewritten to a value that leaves the
    # edge mismatched, so the sampled step cannot descend.
    mutant_edge = int(chosen[0])
    port_left[mutant_edge] = np.int16((int(port_left[mutant_edge]) + 1) % 6)
    phi_after, mask_after = _phi(port_left, port_right, gauge)
    assert bool(mask_after[mutant_edge]) is True

    row = tracker.record_cycle(
        cycle=0,
        phi_before=phi_before,
        phi_after=phi_after,
        before=before,
        direction=direction,
        port_left=port_left,
        port_right=port_right,
        gauge=gauge,
        mismatches_after=mask_after,
        repair_budget=8,
        sector_link_writes_reported=0,
        readback_drive_edges=0,
    )

    sampled = row["sampled_transaction"]
    assert sampled["sample_count"] == 2
    assert mutant_edge in sampled["sample_edge_indices"]
    assert sampled["all_samples_committed"] is False
    assert any(
        failure["edge_index"] == mutant_edge for failure in sampled["failures"]
    )
    report = tracker.run_report()
    assert report["transactional_sampling"]["failure_count"] >= 1
    assert report["transactional_sampling"]["all_samples_committed"] is False
    assert report["SCHEDULER_CLASS_CONFORMANCE_RECEIPT"] is False


def test_replay_accepts_genuine_repair_and_rejects_corrupting_write():
    transported = _transported(4, 1)
    mismatched_left = (transported + 1) % 6
    # Genuine repair: the edge is mismatched before and consistent after.
    row = replay_edge_repair_transaction(
        edge_index=4,
        endpoint_left=1,
        endpoint_right=2,
        before=(mismatched_left, 4, 1),
        after=(transported, 4, 1),
        group_name="S3",
        group_order=6,
        proposal_id="genuine",
    )
    assert row["committed"] is True
    assert row["ok"] is True
    assert row["post_state_matches_kernel"] is True

    # A step that only corrupts a slot out of the group range fails closed.
    corrupt = replay_edge_repair_transaction(
        edge_index=4,
        endpoint_left=1,
        endpoint_right=2,
        before=(mismatched_left, 4, 1),
        after=(9, 4, 1),
        group_name="S3",
        group_order=6,
        proposal_id="corrupt",
    )
    assert corrupt["committed"] is False
    assert corrupt["ok"] is False


def _transported(right_value: int, gauge_value: int) -> int:
    from oph_fpe.gauge.covariant_overlap import transport_right_to_left

    return int(
        transport_right_to_left(
            np.asarray([right_value], dtype=np.int16),
            np.asarray([gauge_value], dtype=np.int16),
            group_name="S3",
            group_order=6,
        )[0]
    )


def test_named_conformance_stream_is_deterministic_and_name_isolated():
    stream_a, report_a = named_conformance_stream(20260820)
    stream_b, report_b = named_conformance_stream(20260820)
    stream_c, report_c = named_conformance_stream(20260821)
    draws_a = stream_a.integers(0, 1 << 30, 8)
    draws_b = stream_b.integers(0, 1 << 30, 8)
    draws_c = stream_c.integers(0, 1 << 30, 8)
    assert np.array_equal(draws_a, draws_b)
    assert not np.array_equal(draws_a, draws_c)
    assert report_a == report_b
    assert report_a["stream_name"] == "conformance"
    assert report_a["stream_id"] == report_c["stream_id"]
    assert report_a["entropy_words_u32"] != report_c["entropy_words_u32"]


def test_array_screen_run_emits_class_conformance_receipts(tmp_path: Path):
    config = load_config(Path("configs/e1_s3_modular_screen_4k.yml"))
    config = dict(config)
    config["run_id"] = "conformance_smoke"
    config["graph"] = dict(config["graph"], patch_count=128, neighbors=6)
    config["dynamics"] = dict(config["dynamics"], cycles=6, repairs_per_cycle=128)
    config["observables"] = dict(config["observables"])
    config["observables"]["modular_lift"] = {"max_points": 1024, "center_samples": 64}

    result = run_array_screen_config(config, tmp_path)
    run_path = Path(result["path"])

    report = json.loads(
        (run_path / "scheduler_class_conformance.json").read_text(encoding="utf-8")
    )
    assert report["schema"] == "oph.scheduler_class_conformance.v1"
    assert report["engine"] == "array_screen"
    assert report["cycles_observed"] == 6
    assert report["all_cycles_in_declared_class"] is True
    assert report["SCHEDULER_CLASS_CONFORMANCE_RECEIPT"] is True
    assert report["spend_split_sums_to_total"] is True
    assert report["declared_class"]["register_rows"] == [
        "PR-60",
        "PR-61",
        "PR-62",
        "PR-63",
    ]
    assert report["transactional_sampling"]["all_samples_committed"] is True
    assert report["transactional_sampling"]["fail_closed"] is True
    totals = report["totals"]
    assert (
        totals["private_spend_actions"] + totals["shared_spend_actions"]
        == totals["total_repair_actions"]
    )
    assert totals["shared_spend_actions"] == 0

    rows = [
        json.loads(line)
        for line in (run_path / "scheduler_class_conformance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(rows) == 6
    assert all(row["observed_schedule_in_declared_class"] for row in rows)
    assert all(
        row["source_term"]["excluded_from_repair_step"] is True for row in rows
    )

    with (run_path / "mismatch_trace.csv").open(encoding="utf-8") as handle:
        trace_rows = list(csv.DictReader(handle))
    assert len(trace_rows) == 6
    for trace_row in trace_rows:
        assert int(trace_row["private_spend_actions"]) + int(
            trace_row["shared_spend_actions"]
        ) == int(trace_row["total_repair_actions"])
        assert trace_row["class_conformance_ok"] == "True"
        assert trace_row["sampled_transaction_ok"] == "True"

    receipts = [
        json.loads(line)
        for line in (run_path / "verifier_receipts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert all("private_spend_actions" in receipt for receipt in receipts)
    assert all(
        receipt["observed_schedule_in_declared_class"] is True for receipt in receipts
    )

    spend = np.load(run_path / "capacity_spend_per_patch.npz")
    assert spend["private_spend"].shape == (128,)
    assert spend["shared_spend"].shape == (128,)
    assert int(spend["private_spend"].sum()) == totals["private_spend_actions"]


def test_bw_array_run_emits_class_conformance_with_shared_spend(tmp_path: Path):
    from oph_fpe.scale.bw_array import run_bw_array_config

    config = load_config(Path("configs/e1_s3_bw_screen_64k.yml"))
    config = dict(config)
    config["run_id"] = "bw_conformance_smoke"
    config["graph"] = dict(config["graph"], patch_count=256, neighbors=6)
    config["dynamics"] = dict(config["dynamics"], cycles=6, repairs_per_cycle=256)
    config["dynamics"]["observer_readback_drive"] = {
        "enabled": True,
        "edge_fraction": 0.05,
        "start_cycle": 0,
    }
    config["bw"] = dict(config["bw"], cap_count=4, times=[0.025], n_jobs=1)
    config["defects"] = dict(
        config.get("defects", {}) or {},
        sector_repair={"enabled": True, "probability": 0.5},
    )

    result = run_bw_array_config(config, tmp_path)
    run_path = Path(result["path"])

    report = json.loads(
        (run_path / "scheduler_class_conformance.json").read_text(encoding="utf-8")
    )
    assert report["engine"] == "bw_array"
    assert report["SCHEDULER_CLASS_CONFORMANCE_RECEIPT"] is True
    assert report["all_cycles_in_declared_class"] is True
    totals = report["totals"]
    assert totals["shared_spend_actions"] > 0
    assert totals["private_spend_actions"] > 0
    assert (
        totals["private_spend_actions"] + totals["shared_spend_actions"]
        == totals["total_repair_actions"]
    )
    assert totals["sector_link_writes_reported"] == totals["shared_spend_actions"]
    # The readback drive is receipted as a source term outside the repair step.
    assert report["source_term_separation"]["edges_touched_total"] > 0
    assert report["source_term_separation"]["excluded_from_repair_step"] is True
    assert report["transactional_sampling"]["all_samples_committed"] is True
    assert report["transactional_sampling"]["failure_count"] == 0

    rows = [
        json.loads(line)
        for line in (run_path / "scheduler_class_conformance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(rows) == 6
    for row in rows:
        assert row["spend_split_sums_to_total"] is True
        assert row["shared_reported_matches_measured"] is True
        assert row["observed_schedule_in_declared_class"] is True
        assert row["source_term"]["excluded_from_repair_step"] is True
        assert row["source_term"]["injected_before_phi_before_measurement"] is True

    with (run_path / "mismatch_trace.csv").open(encoding="utf-8") as handle:
        trace_rows = list(csv.DictReader(handle))
    assert len(trace_rows) == 6
    for trace_row in trace_rows:
        assert int(trace_row["private_spend_actions"]) + int(
            trace_row["shared_spend_actions"]
        ) == int(trace_row["total_repair_actions"])
        assert trace_row["class_conformance_ok"] == "True"

    spend = np.load(run_path / "capacity_spend_per_patch.npz")
    assert int(spend["private_spend"].sum()) == totals["private_spend_actions"]
    assert float(spend["shared_spend"].sum()) == float(totals["shared_spend_actions"])
