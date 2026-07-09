from __future__ import annotations

from typing import Any


class RefinementAudit:
    def compare_event_laws(self, run_fine: dict[str, float], run_coarse: dict[str, float]) -> dict[str, Any]:
        keys = sorted(set(run_fine) | set(run_coarse))
        distance = 0.5 * sum(abs(float(run_fine.get(key, 0.0)) - float(run_coarse.get(key, 0.0))) for key in keys)
        return {
            "event_law_distance": distance,
            "keys": keys,
            "REFINEMENT_STABILITY_RECEIPT": False,
        }

    def tv_bound(self, error_ledger: dict[str, float]) -> float:
        return (
            float(error_ledger.get("epsilon_mu", 0.0))
            + float(error_ledger.get("expected_path_length", 0.0)) * float(error_ledger.get("epsilon_K", 0.0))
            + float(error_ledger.get("epsilon_E", 0.0))
            + float(error_ledger.get("epsilon_prop", 0.0))
            + float(error_ledger.get("epsilon_detector", 0.0))
            + float(error_ledger.get("epsilon_canon", 0.0))
            + float(error_ledger.get("epsilon_clock", 0.0))
            + float(error_ledger.get("epsilon_mc", 0.0))
        )
