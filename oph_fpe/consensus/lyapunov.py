from __future__ import annotations

import math
from typing import Any, Iterable

from oph_fpe.claims import RECOVERED_CORE, with_claim_metadata


def lyapunov_descent_receipt(trace: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Chained Lyapunov non-increase receipt for a repair trajectory.

    The receipt matches the theory-side contract of the proven repair
    layer: every accepted repair move keeps the broken-edge functional
    non-increasing, so the whole trajectory must be monotone across
    consecutive rows, not merely inside each row's repair window. Two
    checks therefore run: the within-row check ``phi <= phi_before`` on
    every row, and the cross-row chain check
    ``phi_before[n] <= phi[n-1]`` between consecutive rows. A trajectory
    whose mismatch count is raised between cycles by an exogenous drive
    fails the receipt; the injected mass is reported per row in
    ``cross_cycle_injections`` and flagged as ``driven_trajectory``. The
    receipt never silently narrows to the within-cycle sub-move.

    Rows must carry an explicit before value under ``phi_before`` or
    ``phi_prev``; a row carrying only ``phi`` raises, because comparing
    a value with itself can never witness descent.
    """
    rows = list(trace)
    deltas: list[float] = []
    violations: list[dict[str, Any]] = []
    cross_cycle_injections: list[dict[str, Any]] = []
    previous_after: float | None = None
    for index, row in enumerate(rows):
        if "phi_before" not in row and "phi_prev" not in row:
            raise ValueError(
                "Lyapunov trace row lacks an explicit phi_before/phi_prev; "
                "a self-comparison cannot witness descent"
            )
        before = _finite_float(row.get("phi_before", row.get("phi_prev")))
        after = _finite_float(row.get("phi", row.get("phi_after", row.get("mismatch_edges"))))
        delta = after - before
        deltas.append(delta)
        if delta > 1e-12:
            violations.append({"index": index, "phi_before": before, "phi_after": after, "delta": delta})
        if previous_after is not None:
            injection = before - previous_after
            if injection > 1e-12:
                cross_cycle_injections.append(
                    {
                        "index": index,
                        "previous_phi": previous_after,
                        "phi_before": before,
                        "injection_delta": injection,
                    }
                )
        previous_after = after
    driven = bool(cross_cycle_injections)
    final_phi = _finite_float(rows[-1].get("phi", rows[-1].get("phi_after", 0.0))) if rows else 0.0
    max_injection = (
        float(max(entry["injection_delta"] for entry in cross_cycle_injections))
        if cross_cycle_injections
        else 0.0
    )
    passed = bool(rows) and not violations and not driven
    report = {
        "mode": "finite_overlap_repair_lyapunov_descent_chained",
        "LYAPUNOV_DESCENT_RECEIPT": passed,
        "receipt": passed,
        "step_count": len(rows),
        "strict_descent_steps": int(sum(delta < -1e-12 for delta in deltas)),
        "max_delta": float(max(deltas)) if deltas else 0.0,
        "final_phi": final_phi,
        "violations": violations,
        "driven_trajectory": driven,
        "cross_cycle_injection_count": len(cross_cycle_injections),
        "max_cross_cycle_injection": max_injection,
        "cross_cycle_injections": cross_cycle_injections[:64],
        "claim_boundary": (
            "finite fixed-cutoff overlap-repair Lyapunov check chained across "
            "consecutive trace rows for the simulated quotient state; a "
            "trajectory with exogenous cross-cycle mismatch injection fails "
            "and is reported as driven; not a same-boundary uniqueness "
            "theorem and not a continuum convergence proof"
        ),
    }
    return with_claim_metadata(report, claim_level=RECOVERED_CORE, receipt="LYAPUNOV_DESCENT_RECEIPT")


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Lyapunov trace contains a non-finite phi value")
    return result
