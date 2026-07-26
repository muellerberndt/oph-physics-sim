from __future__ import annotations

import pytest

from oph_fpe.consensus.phase_repair_bridge import (
    circular_phase_midpoint,
    circular_phase_mismatch,
    declared_pair_mismatch_measure,
    induced_phase_repair_transactions,
    pairwise_phase_dispersion,
    phase_boundary_guard,
    phase_lock_measurement_report,
    phase_register_key,
    phase_repair_bridge_report,
    phase_repair_confluence_report,
    phase_state_registers,
)
from oph_fpe.consensus.transactional_repair import (
    commit_transaction,
    prepare_transaction,
)


SEED = 20260596


def test_circular_phase_primitives_are_exact_on_the_wrap() -> None:
    assert circular_phase_mismatch(0.95, 0.05) == pytest.approx(0.1)
    assert circular_phase_mismatch(0.25, 0.75) == pytest.approx(0.5)
    assert circular_phase_midpoint(0.9, 0.1) == pytest.approx(0.0)
    assert circular_phase_midpoint(0.2, 0.4) == pytest.approx(0.3)
    dispersion = pairwise_phase_dispersion([0.0, 0.0, 0.5])
    assert dispersion["max_pairwise_mismatch"] == pytest.approx(0.5)


def test_phase_lock_measurement_conserves_dispersion_under_shared_step() -> None:
    report = phase_lock_measurement_report(
        carrier_count=6, seed=SEED, steps=9, intrinsic_step=0.07
    )

    assert report["PHASE_LOCK_MEASUREMENT_RECEIPT"] is True
    assert len(report["dispersion_trajectory"]) == 10
    assert len(report["locked_at_threshold"]) == 10
    assert report["dispersion_conservation_residual"] <= 5.0e-12
    # Deterministic replay of the same declared batch.
    replay = phase_lock_measurement_report(
        carrier_count=6, seed=SEED, steps=9, intrinsic_step=0.07
    )
    assert replay["terminal_phases"] == report["terminal_phases"]


def test_locked_batch_reports_lock_and_induces_no_repairs() -> None:
    tight = [0.40, 0.41, 0.42, 0.43]
    report = phase_lock_measurement_report(
        carrier_count=4,
        seed=SEED,
        steps=3,
        lock_threshold=0.1,
        initial_phases=tight,
    )
    assert all(report["locked_at_threshold"])

    induction = induced_phase_repair_transactions(
        tight, mismatch_threshold=0.1
    )
    assert induction["transactions"] == []
    assert induction["matched_pairs"] == []


def test_bridge_report_receipts_are_computed_true_on_reference_batch() -> None:
    report = phase_repair_bridge_report(carrier_count=8, seed=SEED)

    assert report["PHASE_LOCK_MEASUREMENT_RECEIPT"] is True
    assert report["PHASE_INDUCED_REPAIR_ACCEPTANCE_RECEIPT"] is True
    assert report["PHASE_REPAIR_CONFLUENCE_RECEIPT"] is True
    assert report["induced_repairs"]["matched_pairs"]
    assert report["induced_repairs"]["declared_mismatch_after"] < (
        report["induced_repairs"]["declared_mismatch_before"]
    )
    confluence = report["confluence"]
    assert confluence["schedule_independent_terminal_state"] is True
    assert confluence["every_accepted_repair_strictly_descends"] is True
    assert confluence["pairwise_disjoint_transactions"] is True
    assert len(set(confluence["terminal_state_hashes"].values())) == 1
    assert report["BW_KMS_CLOCK_RECEIPT"] is False
    assert report["PHYSICAL_2PI_CLOCK_SELECTION_RECEIPT"] is False
    assert "no KMS" in report["claim_boundary"]
    assert "no physical clock normalization" in report["claim_boundary"]

    replay = phase_repair_bridge_report(carrier_count=8, seed=SEED)
    assert replay["terminal_state_hash"] == report["terminal_state_hash"]


def test_non_descending_repair_proposal_is_rejected() -> None:
    phases = [0.0, 0.5, 0.25, 0.75]
    state = phase_state_registers(phases)
    pairs = ((0, 1), (2, 3))
    measure = declared_pair_mismatch_measure(pairs)
    boundary = phase_boundary_guard()
    # This proposal keeps the pair mismatch identical, so descent fails.
    widening = prepare_transaction(
        state,
        tx_id="adversarial-widen",
        read_set={phase_register_key(0), phase_register_key(1)},
        payload={phase_register_key(0): 0.1, phase_register_key(1): 0.6},
    )
    result = commit_transaction(state, widening, measure=measure, boundary=boundary)

    assert result.committed is False
    assert result.status == "NO_DESCENT"

    report = phase_repair_confluence_report(
        state, [widening], matched_pairs=pairs
    )
    assert report["PHASE_REPAIR_CONFLUENCE_RECEIPT"] is False
    assert any(
        blocker.startswith("repair_transaction_rejected:")
        and blocker.endswith("NO_DESCENT")
        for blocker in report["blockers"]
    )


def test_schedule_dependent_mutant_fails_confluence() -> None:
    phases = [0.0, 0.5, 0.9]
    state = phase_state_registers(phases)
    pairs = ((0, 1), (1, 2))
    measure = declared_pair_mismatch_measure(pairs)
    key_0 = phase_register_key(0)
    key_1 = phase_register_key(1)
    key_2 = phase_register_key(2)
    # Both mutant transactions write register one with different values, so
    # whichever commits second is stale and the terminal states diverge.
    first = prepare_transaction(
        state,
        tx_id="mutant-a",
        read_set={key_0, key_1},
        payload={key_0: 0.25, key_1: 0.25},
    )
    second = prepare_transaction(
        state,
        tx_id="mutant-b",
        read_set={key_1, key_2},
        payload={key_1: 0.7, key_2: 0.7},
    )
    baseline = measure(state)
    assert measure({**state, key_0: 0.25, key_1: 0.25}) < baseline
    assert measure({**state, key_1: 0.7, key_2: 0.7}) < baseline

    report = phase_repair_confluence_report(
        state, [first, second], matched_pairs=pairs
    )

    assert report["PHASE_REPAIR_CONFLUENCE_RECEIPT"] is False
    assert report["schedule_independent_terminal_state"] is False
    assert report["pairwise_disjoint_transactions"] is False
    assert (
        report["terminal_state_hashes"]["declared_order"]
        != report["terminal_state_hashes"]["reversed_order"]
    )
    assert "terminal_states_differ_between_schedules" in report["blockers"]


def test_empty_transaction_set_fails_closed() -> None:
    state = phase_state_registers([0.1, 0.2])
    report = phase_repair_confluence_report(state, [], matched_pairs=())

    assert report["PHASE_REPAIR_CONFLUENCE_RECEIPT"] is False
    assert "no_induced_repair_transactions" in report["blockers"]


def test_induced_repairs_target_only_over_threshold_disjoint_pairs() -> None:
    phases = [0.0, 0.02, 0.5, 0.52]
    induction = induced_phase_repair_transactions(
        phases, mismatch_threshold=0.125
    )

    matched = {tuple(pair) for pair in induction["matched_pairs"]}
    # Within-cluster pairs sit below the threshold; only cross-cluster pairs
    # qualify and the matching keeps them register-disjoint.
    assert matched
    assert all(
        circular_phase_mismatch(phases[i], phases[j]) > 0.125 for i, j in matched
    )
    seen: set[int] = set()
    for i, j in matched:
        assert i not in seen and j not in seen
        seen.update((i, j))
