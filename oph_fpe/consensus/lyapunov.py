from __future__ import annotations

import math
from typing import Any, Iterable

from oph_fpe.claims import RECOVERED_CORE, with_claim_metadata


def lyapunov_descent_receipt(trace: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(trace)
    deltas: list[float] = []
    violations: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        before = _finite_float(row.get("phi_before", row.get("phi_prev", row.get("phi"))))
        after = _finite_float(row.get("phi", row.get("phi_after", row.get("mismatch_edges"))))
        delta = after - before
        deltas.append(delta)
        if delta > 1e-12:
            violations.append({"index": index, "phi_before": before, "phi_after": after, "delta": delta})
    final_phi = _finite_float(rows[-1].get("phi", rows[-1].get("phi_after", 0.0))) if rows else 0.0
    report = {
        "mode": "finite_overlap_repair_lyapunov_descent",
        "LYAPUNOV_DESCENT_RECEIPT": bool(rows) and not violations,
        "receipt": bool(rows) and not violations,
        "step_count": len(rows),
        "strict_descent_steps": int(sum(delta < -1e-12 for delta in deltas)),
        "max_delta": float(max(deltas)) if deltas else 0.0,
        "final_phi": final_phi,
        "violations": violations,
        "claim_boundary": (
            "finite fixed-cutoff overlap-repair Lyapunov check for the simulated quotient state; "
            "not a same-boundary uniqueness theorem and not a continuum convergence proof"
        ),
    }
    return with_claim_metadata(report, claim_level=RECOVERED_CORE, receipt="LYAPUNOV_DESCENT_RECEIPT")


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Lyapunov trace contains a non-finite phi value")
    return result
