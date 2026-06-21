from __future__ import annotations

from typing import Any, Iterable

from oph_fpe.claims import FINITE_CONSENSUS_THEOREM_RECEIPT, RECOVERED_CORE, with_claim_metadata


def finite_consensus_theorem_certificate(
    trace: Iterable[dict[str, Any]],
    evidence: dict[str, Any] | None = None,
    *,
    strict_tol: float = 1.0e-12,
) -> dict[str, Any]:
    """Fail-closed C0b consensus theorem certificate.

    Ordinary simulator traces only show that the current search settled.  C0b
    needs theorem-phase strict descent plus replay/confluence evidence.  Missing
    evidence therefore fails the theorem receipt instead of promoting settling.
    """

    rows = list(trace)
    evidence = dict(evidence or {})
    theorem_rows = [row for row in rows if str(row.get("phase", "")) == "theorem"]
    accepted_theorem_rows = [row for row in theorem_rows if bool(row.get("accepted", False))]
    strict_descent_violations = [
        _event_violation(row, index=index, key="delta_touched_phi")
        for index, row in enumerate(accepted_theorem_rows)
        if _float_or(row.get("delta_touched_phi", row.get("delta_phi")), 0.0) >= -strict_tol
    ]
    phi_increase_violations = [
        _event_violation(row, index=index, key="delta_global_phi")
        for index, row in enumerate(accepted_theorem_rows)
        if _float_or(row.get("delta_global_phi", row.get("delta_phi")), 0.0) > strict_tol
    ]

    required_replay_fields = {
        "disjoint_commutation_violation_count": 0,
        "local_diamond_violation_count": 0,
        "repair_completeness_violation_count": 0,
        "unique_terminal_quotient_hash_count": 1,
    }
    theorem_event_count = int(evidence.get("theorem_phase_event_count", len(theorem_rows)))
    accepted_theorem_move_count = int(evidence.get("accepted_theorem_move_count", len(accepted_theorem_rows)))
    strict_descent_violation_count = int(
        evidence.get("strict_descent_violation_count", len(strict_descent_violations))
    )
    phi_increase_violation_count = int(
        evidence.get("accepted_phi_increase_violation_count", len(phi_increase_violations))
    )
    missing_evidence: list[str] = []
    if theorem_event_count <= 0:
        missing_evidence.append("theorem_phase_repair_events")
    for name in required_replay_fields:
        if name not in evidence:
            missing_evidence.append(name)

    disjoint_violations = int(evidence.get("disjoint_commutation_violation_count", -1))
    diamond_violations = int(evidence.get("local_diamond_violation_count", -1))
    completeness_violations = int(evidence.get("repair_completeness_violation_count", -1))
    terminal_hash_count = int(evidence.get("unique_terminal_quotient_hash_count", 0))
    schedule_replay_count = int(evidence.get("schedule_replay_count", 0))
    requested_schedule_replays = int(evidence.get("requested_schedule_replays", 16))
    if "schedule_replay_count" not in evidence:
        missing_evidence.append("schedule_replay_count")
    if schedule_replay_count < requested_schedule_replays:
        missing_evidence.append("sufficient_schedule_replays")

    passed = bool(
        theorem_event_count > 0
        and not missing_evidence
        and strict_descent_violation_count == 0
        and phi_increase_violation_count == 0
        and disjoint_violations == required_replay_fields["disjoint_commutation_violation_count"]
        and diamond_violations == required_replay_fields["local_diamond_violation_count"]
        and completeness_violations == required_replay_fields["repair_completeness_violation_count"]
        and terminal_hash_count == required_replay_fields["unique_terminal_quotient_hash_count"]
        and schedule_replay_count >= requested_schedule_replays
    )
    report = {
        "mode": "finite_consensus_theorem_certificate_v1",
        FINITE_CONSENSUS_THEOREM_RECEIPT: passed,
        "finite_consensus_theorem_receipt": passed,
        "receipt": passed,
        "theorem_phase_event_count": theorem_event_count,
        "accepted_theorem_move_count": accepted_theorem_move_count,
        "strict_descent_violation_count": strict_descent_violation_count,
        "accepted_phi_increase_violation_count": phi_increase_violation_count,
        "disjoint_commutation_violation_count": disjoint_violations,
        "local_diamond_violation_count": diamond_violations,
        "repair_completeness_violation_count": completeness_violations,
        "unique_terminal_quotient_hash_count": terminal_hash_count,
        "schedule_replay_count": schedule_replay_count,
        "requested_schedule_replays": requested_schedule_replays,
        "missing_evidence": sorted(set(missing_evidence)),
        "strict_descent_violations": strict_descent_violations[:16],
        "accepted_phi_increase_violations": phi_increase_violations[:16],
        "claim_boundary": (
            "C0b finite consensus theorem receipt. This is stricter than final_phi == 0: it requires "
            "theorem-phase strict touched-overlap descent, disjoint/local replay checks, repair "
            "completeness, and schedule-confluent terminal quotient hashes."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=RECOVERED_CORE,
        receipt=FINITE_CONSENSUS_THEOREM_RECEIPT,
        physical_claim=False,
        observable_id="theorem_phase_repair_replay_and_terminal_hashes",
        fit_objective="strict_descent_confluence_and_repair_completeness",
    )


def _event_violation(row: dict[str, Any], *, index: int, key: str) -> dict[str, Any]:
    return {
        "index": index,
        "cycle": row.get("cycle"),
        "node": row.get("node"),
        key: _float_or(row.get(key, row.get("delta_phi")), 0.0),
        "reason": row.get("reason"),
    }


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
