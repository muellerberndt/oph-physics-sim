from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompactHistory:
    source_id: str
    steps: list[dict[str, Any]] = field(default_factory=list)

    def append_step(
        self,
        q_prev: str,
        q_next: str,
        packets: dict[str, Any],
        step_receipt: dict[str, Any],
        delta_tau: float | None,
    ) -> None:
        self.steps.append(
            {
                "q_prev": q_prev,
                "q_next": q_next,
                "packets": dict(packets),
                "step_receipt": dict(step_receipt),
                "delta_tau": delta_tau,
            }
        )

    def conservation_residual(self) -> float:
        total = 0.0
        for step in self.steps:
            receipt = step.get("step_receipt") or {}
            total += abs(float(receipt.get("conservation_residual", 0.0)))
        return total

    def visible_history(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "step_count": len(self.steps),
            "states": [step["q_next"] for step in self.steps],
            "packet_ledgers": [step["packets"] for step in self.steps],
            "conservation_residual_abs_sum": self.conservation_residual(),
        }
