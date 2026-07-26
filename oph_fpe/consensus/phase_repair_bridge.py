"""Finite bridge from local carrier phase dynamics to confluent repair.

The bridge connects two accepted finite instruments.  On one side the
target-blind twelve-port carrier dynamics propagates a batch of carriers with
one shared coupling and a dimensionless mod-one phase register.  On the other
side the transactional repair machinery commits typed strict-descent
transactions with atomic diamond witnesses.  This module measures circular
phase dispersion over the propagated batch, induces typed repair transactions
for carrier pairs whose phase mismatch exceeds a declared threshold, and
proves on the finite batch that the induced repair set is confluent and
strictly descending.

Every receipt is computed from the executed checks.  The claim boundary is a
finite carrier-level bridge: the intrinsic phase register carries no physical
clock normalization and no KMS or thermal-time claim.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from oph_fpe.consensus.transactional_repair import (
    Transaction,
    atomic_diamond_report,
    canonical_state_hash,
    commit_transaction,
    prepare_transaction,
    transactions_conflict,
)
from oph_fpe.core.echosahedral_dynamics import (
    LocalRecurrentCarrierState,
    initialize_local_recurrent_carriers,
    propagate_local_recurrent_carriers,
)


_PHASE_KEY_PREFIX = "carrier_phase_"
_DEFAULT_CARRIER_COUNT = 8
_DEFAULT_SEED = 20260596
_DEFAULT_STEPS = 12
_DEFAULT_INTRINSIC_STEP = 0.0625
_DEFAULT_COUPLING_STRENGTH = 1.0
_DEFAULT_MISMATCH_THRESHOLD = 0.125
_DISPERSION_INVARIANCE_TOLERANCE = 5.0e-12

CLAIM_BOUNDARY = (
    "Finite carrier-level bridge only: circular phase dispersion is measured "
    "on one propagated batch, induced repairs are typed strict-descent "
    "transactions on the declared mismatch measure, and confluence is a "
    "schedule-independence witness on that finite batch. The intrinsic phase "
    "register is dimensionless, carries no physical clock normalization, and "
    "supports no KMS or thermal-time claim."
)


def phase_register_key(carrier_index: int) -> str:
    """Return the typed state register name for one carrier phase."""

    return f"{_PHASE_KEY_PREFIX}{int(carrier_index):04d}"


def circular_phase_mismatch(left: float, right: float) -> float:
    """Exact circular distance between two mod-one phases, in [0, 1/2]."""

    delta = abs(float(left) - float(right)) % 1.0
    return min(delta, 1.0 - delta)


def circular_phase_midpoint(left: float, right: float) -> float:
    """Return the mod-one midpoint on the shorter arc between two phases."""

    start = float(left) % 1.0
    signed = ((float(right) - float(left) + 0.5) % 1.0) - 0.5
    return (start + signed / 2.0) % 1.0


def pairwise_phase_dispersion(phases: Sequence[float]) -> dict[str, float]:
    """Mean and maximum pairwise circular mismatch over one phase batch."""

    values = [float(value) for value in phases]
    count = len(values)
    if count < 2:
        raise ValueError("phase dispersion requires at least two carriers")
    mismatches = [
        circular_phase_mismatch(values[i], values[j])
        for i in range(count)
        for j in range(i + 1, count)
    ]
    return {
        "mean_pairwise_mismatch": float(sum(mismatches) / len(mismatches)),
        "max_pairwise_mismatch": float(max(mismatches)),
    }


def phase_state_registers(phases: Sequence[float]) -> dict[str, float]:
    """Materialize a phase batch as a typed repair state."""

    return {
        phase_register_key(index): float(value) % 1.0
        for index, value in enumerate(phases)
    }


def declared_pair_mismatch_measure(
    pairs: Sequence[tuple[int, int]],
) -> Any:
    """Build the declared mismatch measure over one frozen pair matching.

    The measure is the sum of circular mismatches over the declared pairs,
    evaluated in fixed pair order.  It is the acceptance predicate for the
    induced repairs: the commit machinery rejects any transaction that fails
    to strictly lower it.
    """

    frozen = tuple((int(i), int(j)) for i, j in pairs)

    def measure(state: Mapping[str, Any]) -> float:
        total = 0.0
        for i, j in frozen:
            total += circular_phase_mismatch(
                float(state[phase_register_key(i)]),
                float(state[phase_register_key(j)]),
            )
        return total

    return measure


def phase_boundary_guard() -> Any:
    """Boundary function protecting the carrier register set itself."""

    def boundary(state: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(sorted(state))

    return boundary


def phase_lock_measurement_report(
    *,
    carrier_count: int = _DEFAULT_CARRIER_COUNT,
    seed: int = _DEFAULT_SEED,
    steps: int = _DEFAULT_STEPS,
    intrinsic_step: float = _DEFAULT_INTRINSIC_STEP,
    coupling_strength: float = _DEFAULT_COUPLING_STRENGTH,
    lock_threshold: float = _DEFAULT_MISMATCH_THRESHOLD,
    initial_phases: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Propagate one batch and measure the circular dispersion trajectory.

    The shared coupling adds the same intrinsic step to every phase register,
    so pairwise mismatches are conserved along the trajectory; the report
    verifies that conservation as a computed check instead of assuming it.
    """

    count = int(carrier_count)
    step_count = int(steps)
    threshold = float(lock_threshold)
    if count < 2:
        raise ValueError("phase lock measurement requires at least two carriers")
    if step_count < 1:
        raise ValueError("phase lock measurement requires at least one step")
    if not (0.0 < threshold < 0.5):
        raise ValueError("lock_threshold must lie in (0, 0.5)")
    state = initialize_local_recurrent_carriers(count, seed=int(seed))
    if initial_phases is not None:
        state = LocalRecurrentCarrierState(
            amplitudes=state.amplitudes,
            intrinsic_phase=np.asarray(
                [float(value) % 1.0 for value in initial_phases], dtype=float
            ),
        )
    trajectory: list[dict[str, float]] = []
    phase_rows: list[list[float]] = []
    current = state
    for _ in range(step_count + 1):
        phases = [float(value) for value in current.intrinsic_phase]
        phase_rows.append(phases)
        trajectory.append(pairwise_phase_dispersion(phases))
        current = propagate_local_recurrent_carriers(
            current,
            intrinsic_step=float(intrinsic_step),
            coupling_strength=float(coupling_strength),
        )
    dispersion_values = [row["mean_pairwise_mismatch"] for row in trajectory]
    conservation_residual = float(
        max(abs(value - dispersion_values[0]) for value in dispersion_values)
    )
    all_finite = all(
        math.isfinite(row["mean_pairwise_mismatch"])
        and math.isfinite(row["max_pairwise_mismatch"])
        and 0.0 <= row["max_pairwise_mismatch"] <= 0.5
        for row in trajectory
    )
    locked_flags = [
        bool(row["max_pairwise_mismatch"] <= threshold) for row in trajectory
    ]
    receipt = bool(
        len(trajectory) == step_count + 1
        and all_finite
        and conservation_residual <= _DISPERSION_INVARIANCE_TOLERANCE
    )
    return {
        "schema": "oph.phase_repair_bridge.phase_lock_measurement.v1",
        "carrier_count": count,
        "seed": int(seed),
        "steps": step_count,
        "intrinsic_step": float(intrinsic_step),
        "coupling_strength": float(coupling_strength),
        "lock_threshold": threshold,
        "dispersion_trajectory": trajectory,
        "locked_at_threshold": locked_flags,
        "terminal_phases": phase_rows[-1],
        "dispersion_conservation_residual": conservation_residual,
        "dispersion_conservation_tolerance": _DISPERSION_INVARIANCE_TOLERANCE,
        "PHASE_LOCK_MEASUREMENT_RECEIPT": receipt,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def induced_phase_repair_transactions(
    phases: Sequence[float],
    *,
    mismatch_threshold: float = _DEFAULT_MISMATCH_THRESHOLD,
    candidate_pairs: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Induce typed repair transactions for over-threshold carrier pairs.

    Pairs are selected as a greedy maximal matching over the above-threshold
    mismatches, largest mismatch first, so the induced transactions touch
    disjoint registers.  Each transaction snaps both members of its pair to
    their circular midpoint, which zeroes that pair's contribution to the
    declared mismatch measure.
    """

    threshold = float(mismatch_threshold)
    if not (0.0 < threshold < 0.5):
        raise ValueError("mismatch_threshold must lie in (0, 0.5)")
    values = [float(value) % 1.0 for value in phases]
    if len(values) < 2:
        raise ValueError("repair induction requires at least two carriers")
    state = phase_state_registers(values)
    if candidate_pairs is None:
        admissible_pairs = tuple(
            (i, j)
            for i in range(len(values))
            for j in range(i + 1, len(values))
        )
        candidate_pair_source = "complete_pair_graph"
    else:
        normalized_pairs: set[tuple[int, int]] = set()
        for raw_pair in candidate_pairs:
            if len(raw_pair) != 2:
                raise ValueError("candidate repair pairs must have length two")
            left, right = (int(raw_pair[0]), int(raw_pair[1]))
            if (
                left == right
                or left < 0
                or right < 0
                or left >= len(values)
                or right >= len(values)
            ):
                raise ValueError("candidate repair pair is outside the carrier batch")
            normalized_pairs.add(tuple(sorted((left, right))))
        if not normalized_pairs:
            raise ValueError("candidate repair pair set must be nonempty")
        admissible_pairs = tuple(sorted(normalized_pairs))
        candidate_pair_source = "declared_federation_seams"
    candidates = sorted(
        (
            (circular_phase_mismatch(values[i], values[j]), i, j)
            for i, j in admissible_pairs
            if circular_phase_mismatch(values[i], values[j]) > threshold
        ),
        key=lambda row: (-row[0], row[1], row[2]),
    )
    matched: list[tuple[int, int]] = []
    used: set[int] = set()
    for _, i, j in candidates:
        if i in used or j in used:
            continue
        used.update((i, j))
        matched.append((i, j))
    transactions: list[Transaction] = []
    proposal_rows: list[dict[str, Any]] = []
    for i, j in matched:
        midpoint = circular_phase_midpoint(values[i], values[j])
        left_key = phase_register_key(i)
        right_key = phase_register_key(j)
        transaction = prepare_transaction(
            state,
            tx_id=f"phase-repair-{i:04d}-{j:04d}",
            read_set={left_key, right_key},
            payload={left_key: midpoint, right_key: midpoint},
        )
        transactions.append(transaction)
        proposal_rows.append(
            {
                "tx_id": transaction.tx_id,
                "pair": [i, j],
                "mismatch_before": circular_phase_mismatch(values[i], values[j]),
                "midpoint": midpoint,
            }
        )
    return {
        "schema": "oph.phase_repair_bridge.induced_repairs.v1",
        "mismatch_threshold": threshold,
        "candidate_pair_source": candidate_pair_source,
        "candidate_pairs": [list(pair) for pair in admissible_pairs],
        "state": state,
        "matched_pairs": [list(pair) for pair in matched],
        "over_threshold_pair_count": len(candidates),
        "transactions": transactions,
        "proposal_rows": proposal_rows,
    }


def phase_repair_confluence_report(
    state: Mapping[str, Any],
    transactions: Sequence[Transaction],
    *,
    matched_pairs: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Prove schedule independence and strict descent on the finite batch.

    The declared measure is the summed circular mismatch over the matched
    pairs.  The transaction set is committed under two schedules, forward and
    reversed; the receipt requires every commit to succeed with strict
    descent in both schedules, exact agreement of the terminal state hashes,
    and an atomic diamond witness for the first disjoint transaction pair.
    """

    blockers: list[str] = []
    pairs = tuple((int(i), int(j)) for i, j in matched_pairs)
    measure = declared_pair_mismatch_measure(pairs)
    boundary = phase_boundary_guard()
    ordered = list(transactions)
    if not ordered:
        blockers.append("no_induced_repair_transactions")
    schedules = {
        "declared_order": ordered,
        "reversed_order": list(reversed(ordered)),
    }
    schedule_rows: dict[str, Any] = {}
    terminal_hashes: dict[str, str | None] = {}
    strict_descent = bool(ordered)
    all_committed = bool(ordered)
    for schedule_name, schedule in schedules.items():
        current: dict[str, Any] = dict(state)
        commit_rows: list[dict[str, Any]] = []
        for transaction in schedule:
            result = commit_transaction(
                current, transaction, measure=measure, boundary=boundary
            )
            descended = bool(
                result.committed and result.after_measure < result.before_measure
            )
            commit_rows.append(
                {
                    "tx_id": transaction.tx_id,
                    "status": result.status,
                    "committed": result.committed,
                    "before_measure": result.before_measure,
                    "after_measure": result.after_measure,
                    "strict_descent": descended,
                    "reason": result.reason,
                }
            )
            if not result.committed:
                all_committed = False
                blockers.append(
                    f"repair_transaction_rejected:{schedule_name}:"
                    f"{transaction.tx_id}:{result.status}"
                )
                continue
            if not descended:
                strict_descent = False
            current = result.state
        schedule_rows[schedule_name] = commit_rows
        terminal_hashes[schedule_name] = canonical_state_hash(current)
    hashes = set(terminal_hashes.values())
    schedule_independent = bool(ordered and all_committed and len(hashes) == 1)
    if ordered and not schedule_independent:
        blockers.append("terminal_states_differ_between_schedules")
    if ordered and not strict_descent:
        blockers.append("committed_repair_without_strict_descent")

    pairwise_disjoint = all(
        not transactions_conflict(ordered[a], ordered[b])
        for a in range(len(ordered))
        for b in range(a + 1, len(ordered))
    )
    diamond: dict[str, Any] | None = None
    if len(ordered) >= 2 and pairwise_disjoint:
        diamond = atomic_diamond_report(
            dict(state),
            ordered[0],
            ordered[1],
            measure=measure,
            boundary=boundary,
        )
        if not diamond.get("DISTRIBUTED_LOCAL_DIAMOND_RECEIPT"):
            blockers.append("atomic_diamond_witness_failed")
    receipt = bool(
        ordered
        and all_committed
        and strict_descent
        and schedule_independent
        and (
            len(ordered) < 2
            or (
                pairwise_disjoint
                and diamond is not None
                and diamond.get("DISTRIBUTED_LOCAL_DIAMOND_RECEIPT") is True
            )
        )
    )
    return {
        "schema": "oph.phase_repair_bridge.confluence.v1",
        "transaction_count": len(ordered),
        "matched_pairs": [list(pair) for pair in pairs],
        "schedules": schedule_rows,
        "terminal_state_hashes": terminal_hashes,
        "schedule_independent_terminal_state": schedule_independent,
        "all_transactions_committed": all_committed,
        "every_accepted_repair_strictly_descends": bool(
            ordered and all_committed and strict_descent
        ),
        "pairwise_disjoint_transactions": pairwise_disjoint,
        "atomic_diamond_witness": diamond,
        "blockers": sorted(set(blockers)),
        "PHASE_REPAIR_CONFLUENCE_RECEIPT": receipt,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def phase_repair_bridge_report(
    *,
    carrier_count: int = _DEFAULT_CARRIER_COUNT,
    seed: int = _DEFAULT_SEED,
    steps: int = _DEFAULT_STEPS,
    intrinsic_step: float = _DEFAULT_INTRINSIC_STEP,
    coupling_strength: float = _DEFAULT_COUPLING_STRENGTH,
    mismatch_threshold: float = _DEFAULT_MISMATCH_THRESHOLD,
    initial_phases: Sequence[float] | None = None,
    candidate_pairs: Sequence[Sequence[int]] | None = None,
    carrier_ids: Sequence[str] | None = None,
    federation_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the full bridge: measurement, induced repair, and confluence."""

    lock = phase_lock_measurement_report(
        carrier_count=carrier_count,
        seed=seed,
        steps=steps,
        intrinsic_step=intrinsic_step,
        coupling_strength=coupling_strength,
        lock_threshold=mismatch_threshold,
        initial_phases=initial_phases,
    )
    induction = induced_phase_repair_transactions(
        lock["terminal_phases"],
        mismatch_threshold=mismatch_threshold,
        candidate_pairs=candidate_pairs,
    )
    confluence = phase_repair_confluence_report(
        induction["state"],
        induction["transactions"],
        matched_pairs=induction["matched_pairs"],
    )
    terminal_hash = confluence["terminal_state_hashes"].get("declared_order")
    measure = declared_pair_mismatch_measure(
        tuple((int(i), int(j)) for i, j in induction["matched_pairs"])
    )
    mismatch_before = measure(induction["state"])
    repaired_state: dict[str, Any] = dict(induction["state"])
    for transaction in induction["transactions"]:
        repaired_state.update(transaction.payload_dict)
    mismatch_after = measure(repaired_state)
    acceptance = bool(
        induction["transactions"]
        and confluence["all_transactions_committed"]
        and confluence["every_accepted_repair_strictly_descends"]
        and mismatch_after < mismatch_before
    )
    carrier_binding_valid = bool(
        carrier_ids is not None
        and len(carrier_ids) == int(carrier_count)
        and len(set(carrier_ids)) == int(carrier_count)
        and all(isinstance(item, str) and item for item in carrier_ids)
        and isinstance(federation_bundle_sha256, str)
        and federation_bundle_sha256.startswith("sha256:")
        and len(federation_bundle_sha256) == 71
        and induction["candidate_pair_source"] == "declared_federation_seams"
    )
    return {
        "schema": "oph.phase_repair_bridge.report.v1",
        "phase_lock_measurement": lock,
        "induced_repairs": {
            "mismatch_threshold": induction["mismatch_threshold"],
            "candidate_pair_source": induction["candidate_pair_source"],
            "candidate_pairs": induction["candidate_pairs"],
            "matched_pairs": induction["matched_pairs"],
            "over_threshold_pair_count": induction["over_threshold_pair_count"],
            "proposal_rows": induction["proposal_rows"],
            "declared_mismatch_before": mismatch_before,
            "declared_mismatch_after": mismatch_after,
        },
        "confluence": confluence,
        "terminal_state_hash": terminal_hash,
        "carrier_ids": [] if carrier_ids is None else list(carrier_ids),
        "federation_bundle_sha256": federation_bundle_sha256,
        "PHASE_LOCK_MEASUREMENT_RECEIPT": bool(
            lock["PHASE_LOCK_MEASUREMENT_RECEIPT"]
        ),
        "PHASE_INDUCED_REPAIR_ACCEPTANCE_RECEIPT": acceptance,
        "PHASE_REPAIR_CONFLUENCE_RECEIPT": bool(
            confluence["PHASE_REPAIR_CONFLUENCE_RECEIPT"]
        ),
        "FEDERATION_PHASE_TO_REPAIR_BINDING_RECEIPT": carrier_binding_valid,
        "BW_KMS_CLOCK_RECEIPT": False,
        "PHYSICAL_2PI_CLOCK_SELECTION_RECEIPT": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


__all__ = [
    "CLAIM_BOUNDARY",
    "circular_phase_midpoint",
    "circular_phase_mismatch",
    "declared_pair_mismatch_measure",
    "induced_phase_repair_transactions",
    "pairwise_phase_dispersion",
    "phase_boundary_guard",
    "phase_lock_measurement_report",
    "phase_register_key",
    "phase_repair_bridge_report",
    "phase_repair_confluence_report",
    "phase_state_registers",
]
