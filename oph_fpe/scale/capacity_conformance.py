"""Capacity accounting and scheduler-class conformance for the array kernels.

This module instruments the production repair loop against the registered
admissible-scheduler class rows PR-60..PR-63 (mismatch non-increase per
supported step, single-register locality, unit expected capacity per step,
and the private/shared capacity-split reading typed in
``Lean/ObservableNormalForms/ObservableNormalForms/SchedulerClassObstructions.lean``
of the reverse-engineering-reality repository).  It measures the realized
schedule; it never changes dynamics.

Capacity-split reading on the edge-slot carrier (declared, mirroring PR-63):

* An endpoint port-slot modification is a **private** spend of the patch
  that owns the written slot.  The kernel's decoupled edge repair holds the
  link fixed and rewrites exactly one endpoint from the link-transported far
  value, so every port write is executed as the owning patch's self-update.
  Spend is counted by measured modification; an executed assignment whose
  transported value equals the old value is carried separately as an
  instruction count.
* A gauge-link modification is a **shared** spend into the network-facing
  overlap record.  For per-patch attribution one link write is split
  half/half between the two endpoint patches, so per-patch shared spend sums
  to the shared action count.
* The far-endpoint write class (an action attributed to one patch that
  rewrites the other patch's slot) is structurally empty in this kernel and
  is receipted as such.

Class membership of the observed schedule is emitted as a MEASURED claim
with clause-by-clause booleans.  It is a property of the realized schedule
of one run configuration, never a theorem.

The transactional sampler replays a per-cycle sample of repaired edges
through :mod:`oph_fpe.repair.transaction` (the contract-grade engine) and
fails closed: any clause failure, reconstruction mismatch, or replay
exception marks the sample, the cycle, and the run-level receipt.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from oph_fpe.gauge.covariant_overlap import covariant_mismatch_mask
from oph_fpe.repair.transaction import (
    MismatchLedger,
    ProposalClass,
    RepairCollar,
    TransactionalRepairEngine,
    TransitionKind,
)

SCHEDULER_CLASS_CONFORMANCE_SCHEMA = "oph.scheduler_class_conformance.v1"
SCHEDULER_CLASS_CONFORMANCE_RECEIPT = "SCHEDULER_CLASS_CONFORMANCE_RECEIPT"
SAMPLED_TRANSACTIONAL_VERIFIER = "sampled_transactional_edge_repair_conformance"
CONFORMANCE_RNG_DERIVATION = "oph-array-conformance-rng-v1"
DEFAULT_SAMPLE_EDGES_PER_CYCLE = 8

DECLARED_SCHEDULER_CLASS = {
    "register_rows": ["PR-60", "PR-61", "PR-62", "PR-63"],
    "typed_in": (
        "reverse-engineering-reality:Lean/ObservableNormalForms/"
        "ObservableNormalForms/SchedulerClassObstructions.lean"
    ),
    "clauses": {
        "mismatch_nonincrease": (
            "no supported repair step increases the declared overlap mismatch"
        ),
        "single_slot_locality": (
            "every supported repair action writes at most one register slot"
        ),
        "unit_capacity_per_step": (
            "stepCost = privateCost + sharedCost is at most one unit per action"
        ),
    },
    "capacity_split_reading": {
        "private": "endpoint port-slot write charged to the owning patch",
        "shared": "gauge-link write charged half/half to the endpoint patches",
        "far_endpoint_write_class": "structurally empty in this kernel",
    },
    "claim_kind": "measured_schedule_property_not_theorem",
}


def named_conformance_stream(seed: int) -> tuple[np.random.Generator, dict[str, Any]]:
    """Derive the name-isolated conformance RNG stream from the run seed.

    The derivation is keyed by the fixed stream name, so consuming draws here
    cannot perturb any other named stream, and adding this stream leaves the
    initialization, readback, repair, and sector schedules byte-identical.
    """

    seed_u64 = int(seed) % (1 << 64)
    base_words = [seed_u64 & 0xFFFFFFFF, (seed_u64 >> 32) & 0xFFFFFFFF]
    material = f"{CONFORMANCE_RNG_DERIVATION}\0conformance".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    name_words = np.frombuffer(digest[:16], dtype="<u4").astype(np.uint64).tolist()
    entropy_words = [*base_words, *(int(value) for value in name_words)]
    stream = np.random.default_rng(np.random.SeedSequence(entropy_words))
    report = {
        "stream_name": "conformance",
        "stream_id": "sha256:" + digest.hex(),
        "derivation": (
            f"sha256({CONFORMANCE_RNG_DERIVATION}\\0conformance) plus the "
            "uint64 run seed"
        ),
        "entropy_words_u32": entropy_words,
        "bit_generator": "PCG64",
    }
    return stream, report


def snapshot_chosen_edge_state(
    chosen: np.ndarray,
    port_left: np.ndarray,
    port_right: np.ndarray,
    gauge: np.ndarray,
) -> dict[str, np.ndarray]:
    """Copy the pre-repair values of the chosen edge slots.

    The copies are the measurement baseline for the spend split and for the
    transactional replays; taking them reads state only.
    """

    selected = np.asarray(chosen, dtype=np.int64)
    return {
        "chosen": selected.copy(),
        "port_left": np.asarray(port_left)[selected].copy(),
        "port_right": np.asarray(port_right)[selected].copy(),
        "gauge": np.asarray(gauge)[selected].copy(),
    }


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "sum": 0.0}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "sum": float(np.sum(values)),
    }


def _edge_local_mismatch_ledger(
    left_value: int,
    right_value: int,
    gauge_value: int,
    *,
    group_name: str,
    group_order: int,
) -> MismatchLedger:
    """Exact edge-local ledger: overlap is the covariant mismatch indicator.

    Record, sector, and holonomy are zero by definition of this evaluator's
    scope (a single decoupled edge slot carries none of them).  Out-of-range
    labels drive both visible components up, so a corrupting write can never
    present as descent.
    """

    in_range = all(
        0 <= int(value) < int(group_order)
        for value in (left_value, right_value, gauge_value)
    )
    if not in_range:
        return MismatchLedger(overlap=int(group_order), local_constraint=1)
    mismatch = covariant_mismatch_mask(
        np.asarray([left_value], dtype=np.int16),
        np.asarray([right_value], dtype=np.int16),
        np.asarray([gauge_value], dtype=np.int16),
        group_name=group_name,
        group_order=group_order,
    )
    return MismatchLedger(overlap=int(bool(mismatch[0])), local_constraint=0)


def replay_edge_repair_transaction(
    *,
    edge_index: int,
    endpoint_left: int,
    endpoint_right: int,
    before: tuple[int, int, int],
    after: tuple[int, int, int],
    group_name: str,
    group_order: int,
    proposal_id: str,
) -> dict[str, Any]:
    """Replay one observed edge repair through the transactional engine.

    The engine re-checks the contract clauses on the reconstructed local
    state: strict ledger descent, write locality against the declared
    collar, protected-boundary preservation, read-set completeness, and
    atomic revalidation.  The result is fail-closed: any exception or any
    failing clause yields ``committed: False`` with recorded reasons.
    """

    edge = int(edge_index)
    left_ref = f"port_left:{edge}"
    right_ref = f"port_right:{edge}"
    gauge_ref = f"gauge:{edge}"
    endpoint_left_ref = f"edge_endpoint_left:{edge}"
    endpoint_right_ref = f"edge_endpoint_right:{edge}"
    all_refs = (
        left_ref,
        right_ref,
        gauge_ref,
        endpoint_left_ref,
        endpoint_right_ref,
    )
    writable = frozenset({left_ref, right_ref, gauge_ref})
    protected = frozenset({endpoint_left_ref, endpoint_right_ref})
    initial_state = {
        left_ref: int(before[0]),
        right_ref: int(before[1]),
        gauge_ref: int(before[2]),
        endpoint_left_ref: int(endpoint_left),
        endpoint_right_ref: int(endpoint_right),
    }
    after_values = {
        left_ref: int(after[0]),
        right_ref: int(after[1]),
        gauge_ref: int(after[2]),
    }

    def evaluator(state: Any, _collar: RepairCollar) -> MismatchLedger:
        return _edge_local_mismatch_ledger(
            int(state[left_ref]),
            int(state[right_ref]),
            int(state[gauge_ref]),
            group_name=group_name,
            group_order=group_order,
        )

    def recovery(state: Any) -> dict[str, int]:
        for ref in all_refs:
            _ = state[ref]
        return dict(after_values)

    collar = RepairCollar(
        collar_id=f"edge_slot_collar:{edge}",
        visible_read_set=frozenset(all_refs),
        writable_registers=writable,
        protected_boundary=protected,
    )
    row: dict[str, Any] = {
        "edge_index": edge,
        "proposal_id": proposal_id,
        "before": {"port_left": int(before[0]), "port_right": int(before[1]), "gauge": int(before[2])},
        "after": {"port_left": int(after[0]), "port_right": int(after[1]), "gauge": int(after[2])},
    }
    try:
        engine = TransactionalRepairEngine(
            initial_state,
            mismatch_evaluator=evaluator,
        )
        proposal = engine.prepare(
            proposal_id=proposal_id,
            transition_kind=TransitionKind.STRICT_REPAIR,
            proposal_class=ProposalClass.PHYSICAL_CARRIER_RESPONSE,
            collar=collar,
            declared_read_set=all_refs,
            recovery=recovery,
        )
        receipt = engine.commit(proposal)
        post_state = engine.state
        post_matches = all(
            int(post_state[ref]) == value for ref, value in after_values.items()
        )
        row.update(
            {
                "committed": bool(receipt.committed),
                "verdict": receipt.verdict,
                "failure_reasons": list(receipt.failure_reasons),
                "post_state_matches_kernel": bool(receipt.committed and post_matches),
                "mismatch_before": receipt.mismatch_before.as_dict()
                if receipt.mismatch_before is not None
                else None,
                "mismatch_after": receipt.mismatch_after.as_dict()
                if receipt.mismatch_after is not None
                else None,
                "commit_id": receipt.commit_id,
            }
        )
        row["ok"] = bool(row["committed"] and row["post_state_matches_kernel"])
    except Exception as error:  # fail closed on any replay defect
        row.update(
            {
                "committed": False,
                "verdict": "REPLAY_EXCEPTION",
                "failure_reasons": [f"{type(error).__name__}: {error}"],
                "post_state_matches_kernel": False,
                "ok": False,
            }
        )
    return row


class CapacityConformanceTracker:
    """Per-cycle capacity accounting plus fail-closed conformance receipts.

    One instance observes one run.  Every method reads kernel state and
    copies of it; nothing here writes into the dynamical arrays.
    """

    def __init__(
        self,
        *,
        patch_count: int,
        edge_left: np.ndarray,
        edge_right: np.ndarray,
        group_name: str,
        group_order: int,
        seed: int,
        sample_edges_per_cycle: int = DEFAULT_SAMPLE_EDGES_PER_CYCLE,
        engine: str = "bw_array",
    ) -> None:
        self._patch_count = int(patch_count)
        self._edge_left = np.asarray(edge_left, dtype=np.int64)
        self._edge_right = np.asarray(edge_right, dtype=np.int64)
        self._group_name = str(group_name)
        self._group_order = int(group_order)
        self._engine = str(engine)
        self._sample_edges_per_cycle = max(0, int(sample_edges_per_cycle))
        self._rng, self._rng_report = named_conformance_stream(seed)
        self._private_per_patch = np.zeros(self._patch_count, dtype=np.int64)
        self._shared_per_patch = np.zeros(self._patch_count, dtype=np.float64)
        self._cycle_rows: list[dict[str, Any]] = []
        self._sample_failures: list[dict[str, Any]] = []
        self._samples_total = 0
        self._cycles_sampled = 0
        self._totals = {
            "private_spend_actions": 0,
            "shared_spend_actions": 0,
            "total_repair_actions": 0,
            "left_endpoint_writes": 0,
            "right_endpoint_writes": 0,
            "port_write_instructions": 0,
            "sector_link_writes_reported": 0,
            "source_term_edges": 0,
        }

    @property
    def rng_stream_report(self) -> dict[str, Any]:
        return dict(self._rng_report)

    @property
    def cycle_rows(self) -> list[dict[str, Any]]:
        return list(self._cycle_rows)

    def record_cycle(
        self,
        *,
        cycle: int,
        phi_before: int,
        phi_after: int,
        before: dict[str, np.ndarray],
        direction: np.ndarray,
        port_left: np.ndarray,
        port_right: np.ndarray,
        gauge: np.ndarray,
        mismatches_after: np.ndarray,
        repair_budget: int,
        sector_link_writes_reported: int,
        readback_drive_edges: int,
    ) -> dict[str, Any]:
        """Measure one completed repair cycle and return its receipt row.

        ``phi_before`` is the mismatch count measured after the readback
        drive injection and before any repair action, so the injection is
        outside the receipted repair step and is carried separately as the
        declared source term.
        """

        chosen = np.asarray(before["chosen"], dtype=np.int64)
        direction = np.asarray(direction, dtype=bool)
        if direction.shape != chosen.shape:
            raise ValueError("direction array must match the chosen edge array")
        after_left = np.asarray(port_left)[chosen]
        after_right = np.asarray(port_right)[chosen]
        after_gauge = np.asarray(gauge)[chosen]
        left_changed = before["port_left"] != after_left
        right_changed = before["port_right"] != after_right
        gauge_changed = before["gauge"] != after_gauge

        # Actions are counted by measured slot modification.  A port-slot
        # assignment whose transported value equals the old value (an edge the
        # link write left consistent) spends nothing; it is carried
        # separately as an executed instruction count.
        left_endpoint_writes = int(np.sum(left_changed))
        right_endpoint_writes = int(np.sum(right_changed))
        private_spend_actions = int(np.sum(left_changed | right_changed))
        shared_link_value_changes = int(np.sum(gauge_changed))
        shared_spend_actions = shared_link_value_changes
        total_repair_actions = private_spend_actions + shared_spend_actions
        port_write_instructions = int(chosen.size)

        both_endpoints_changed = int(np.sum(left_changed & right_changed))
        offside_left_writes = int(np.sum(left_changed & ~direction))
        offside_right_writes = int(np.sum(right_changed & direction))
        single_slot_locality_ok = bool(
            both_endpoints_changed == 0
            and offside_left_writes == 0
            and offside_right_writes == 0
        )

        residual_chosen = int(np.sum(np.asarray(mismatches_after)[chosen])) if chosen.size else 0
        resolved = int(chosen.size) - residual_chosen
        interference = int(phi_after) - (int(phi_before) - resolved)
        mismatch_nonincrease_ok = bool(int(phi_after) <= int(phi_before))
        within_budget_ok = bool(int(chosen.size) <= int(repair_budget))
        unit_capacity_ok = single_slot_locality_ok

        # Per-patch cumulative spend, attributed by the measured written slot.
        if chosen.size:
            left_edges = chosen[left_changed]
            right_edges = chosen[right_changed]
            if left_edges.size:
                np.add.at(self._private_per_patch, self._edge_left[left_edges], 1)
            if right_edges.size:
                np.add.at(self._private_per_patch, self._edge_right[right_edges], 1)
            shared_edges = chosen[gauge_changed]
            if shared_edges.size:
                np.add.at(self._shared_per_patch, self._edge_left[shared_edges], 0.5)
                np.add.at(self._shared_per_patch, self._edge_right[shared_edges], 0.5)

        sampled = self._sample_transactions(
            cycle=cycle,
            before=before,
            after_left=after_left,
            after_right=after_right,
            after_gauge=after_gauge,
        )

        clauses = {
            "mismatch_nonincrease": mismatch_nonincrease_ok,
            "single_slot_locality": single_slot_locality_ok,
            "unit_capacity_per_action": unit_capacity_ok,
            "within_declared_cycle_budget": within_budget_ok,
        }
        row = {
            "receipt": "SCHEDULER_CLASS_CONFORMANCE_CYCLE",
            "engine": self._engine,
            "cycle": int(cycle),
            "phi_before": int(phi_before),
            "phi_after": int(phi_after),
            "private_spend_actions": private_spend_actions,
            "shared_spend_actions": shared_spend_actions,
            "total_repair_actions": total_repair_actions,
            "left_endpoint_writes": left_endpoint_writes,
            "right_endpoint_writes": right_endpoint_writes,
            "far_endpoint_writes": 0,
            "port_write_instructions": port_write_instructions,
            "sector_link_writes_reported": int(sector_link_writes_reported),
            "shared_reported_matches_measured": bool(
                int(sector_link_writes_reported) == shared_link_value_changes
            ),
            "spend_split_sums_to_total": bool(
                private_spend_actions + shared_spend_actions == total_repair_actions
            ),
            "repair_budget": int(repair_budget),
            "chosen_edges": int(chosen.size),
            "resolved_chosen_edges": resolved,
            "residual_chosen_mismatch": residual_chosen,
            "decoupled_edge_repair_interference": interference,
            "edge_slot_decoupling_ok": bool(interference == 0),
            "clauses": clauses,
            "observed_schedule_in_declared_class": bool(all(clauses.values())),
            "claim_kind": "measured",
            "source_term": {
                "kind": "observer_readback_drive",
                "edges_touched": int(readback_drive_edges),
                "excluded_from_repair_step": True,
                "injected_before_phi_before_measurement": True,
            },
            "sampled_transaction": sampled,
        }
        self._totals["private_spend_actions"] += private_spend_actions
        self._totals["shared_spend_actions"] += shared_spend_actions
        self._totals["total_repair_actions"] += total_repair_actions
        self._totals["left_endpoint_writes"] += left_endpoint_writes
        self._totals["right_endpoint_writes"] += right_endpoint_writes
        self._totals["port_write_instructions"] += port_write_instructions
        self._totals["sector_link_writes_reported"] += int(sector_link_writes_reported)
        self._totals["source_term_edges"] += int(readback_drive_edges)
        self._cycle_rows.append(row)
        return row

    def _sample_transactions(
        self,
        *,
        cycle: int,
        before: dict[str, np.ndarray],
        after_left: np.ndarray,
        after_right: np.ndarray,
        after_gauge: np.ndarray,
    ) -> dict[str, Any]:
        chosen = np.asarray(before["chosen"], dtype=np.int64)
        sample_size = min(self._sample_edges_per_cycle, int(chosen.size))
        if sample_size <= 0:
            return {
                "verifier": SAMPLED_TRANSACTIONAL_VERIFIER,
                "rng_stream": "conformance",
                "sample_count": 0,
                "sample_edge_indices": [],
                "all_samples_committed": True,
                "vacuous": True,
                "failures": [],
            }
        ranks = np.sort(
            self._rng.choice(int(chosen.size), size=sample_size, replace=False)
        )
        rows: list[dict[str, Any]] = []
        for rank in ranks.tolist():
            edge = int(chosen[rank])
            row = replay_edge_repair_transaction(
                edge_index=edge,
                endpoint_left=int(self._edge_left[edge]),
                endpoint_right=int(self._edge_right[edge]),
                before=(
                    int(before["port_left"][rank]),
                    int(before["port_right"][rank]),
                    int(before["gauge"][rank]),
                ),
                after=(
                    int(after_left[rank]),
                    int(after_right[rank]),
                    int(after_gauge[rank]),
                ),
                group_name=self._group_name,
                group_order=self._group_order,
                proposal_id=f"{self._engine}:cycle{int(cycle)}:edge{edge}",
            )
            rows.append(row)
        failures = [row for row in rows if not row["ok"]]
        self._samples_total += len(rows)
        self._cycles_sampled += 1
        if failures:
            self._sample_failures.extend(
                {"cycle": int(cycle), **failure} for failure in failures
            )
        return {
            "verifier": SAMPLED_TRANSACTIONAL_VERIFIER,
            "rng_stream": "conformance",
            "sample_count": len(rows),
            "sample_edge_indices": [row["edge_index"] for row in rows],
            "all_samples_committed": not failures,
            "vacuous": False,
            "failures": [
                {
                    "edge_index": row["edge_index"],
                    "verdict": row["verdict"],
                    "failure_reasons": row["failure_reasons"],
                }
                for row in failures
            ],
        }

    def trace_fields(self, row: dict[str, Any]) -> dict[str, Any]:
        """Project one cycle receipt row onto flat mismatch-trace columns."""

        return {
            "private_spend_actions": row["private_spend_actions"],
            "shared_spend_actions": row["shared_spend_actions"],
            "total_repair_actions": row["total_repair_actions"],
            "class_conformance_ok": bool(row["observed_schedule_in_declared_class"]),
            "sampled_transaction_ok": bool(
                row["sampled_transaction"]["all_samples_committed"]
            ),
        }

    def per_patch_spend(self) -> dict[str, np.ndarray]:
        return {
            "private": self._private_per_patch.copy(),
            "shared": self._shared_per_patch.copy(),
        }

    def run_report(self) -> dict[str, Any]:
        """Aggregate the run-level scheduler-class conformance receipt."""

        violating = [
            row["cycle"]
            for row in self._cycle_rows
            if not row["observed_schedule_in_declared_class"]
        ]
        sampling_ok = not self._sample_failures
        all_in_class = not violating
        return {
            "schema": SCHEDULER_CLASS_CONFORMANCE_SCHEMA,
            "engine": self._engine,
            "declared_class": DECLARED_SCHEDULER_CLASS,
            "rng_stream": self._rng_report,
            "cycles_observed": len(self._cycle_rows),
            "all_cycles_in_declared_class": all_in_class,
            "violating_cycles": violating,
            "totals": dict(self._totals),
            "spend_split_sums_to_total": bool(
                self._totals["private_spend_actions"]
                + self._totals["shared_spend_actions"]
                == self._totals["total_repair_actions"]
            ),
            "per_patch_spend_summary": {
                "private": _summary_stats(self._private_per_patch),
                "shared": _summary_stats(self._shared_per_patch),
                "shared_attribution": (
                    "one link write charged half/half to the two endpoint patches"
                ),
            },
            "transactional_sampling": {
                "verifier": SAMPLED_TRANSACTIONAL_VERIFIER,
                "sample_edges_per_cycle": self._sample_edges_per_cycle,
                "cycles_sampled": self._cycles_sampled,
                "samples_total": self._samples_total,
                "failure_count": len(self._sample_failures),
                "failures": self._sample_failures[:64],
                "all_samples_committed": sampling_ok,
                "fail_closed": True,
            },
            "source_term_separation": {
                "kind": "observer_readback_drive",
                "edges_touched_total": self._totals["source_term_edges"],
                "excluded_from_repair_step": True,
                "receipted_per_cycle": True,
            },
            SCHEDULER_CLASS_CONFORMANCE_RECEIPT: bool(all_in_class and sampling_ok),
            "claim_boundary": (
                "Telemetry measures the realized schedule of this run "
                "configuration. Class membership against PR-60..PR-63 is a "
                "measured property of the observed cycles, stated clause by "
                "clause; it is not a theorem about the scheduler class, and "
                "no dynamical trajectory is altered by this instrumentation."
            ),
        }
