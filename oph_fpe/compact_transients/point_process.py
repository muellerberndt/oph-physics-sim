from __future__ import annotations

import math
from typing import Any, Callable


class MarkedCatalogProcess:
    def intensity(self, observer_record: dict[str, Any], model: dict[str, Any]) -> float:
        base = float(model.get("base_intensity", 0.0))
        weight = float(observer_record.get("weight", 1.0))
        return max(0.0, base * weight)

    def compensator(self, obs_window: dict[str, Any], model: dict[str, Any]) -> float:
        duration = float(obs_window.get("duration", obs_window.get("T_stop", 1.0) - obs_window.get("T_start", 0.0)))
        exposure = float(obs_window.get("exposure_fraction", 1.0))
        return max(0.0, float(model.get("base_intensity", 0.0)) * duration * exposure)

    def loglike(self, catalog: list[dict[str, Any]], obs_window: dict[str, Any], model: dict[str, Any]) -> float:
        event_term = 0.0
        for row in catalog:
            lam = max(self.intensity(row, model), 1.0e-300)
            event_term += math.log(lam)
        return event_term - self.compensator(obs_window, model)


class RepeaterHistoryLikelihood:
    def conditional_intensity(
        self,
        t: float,
        mark: dict[str, Any],
        source_history: list[dict[str, Any]],
        model: dict[str, Any],
    ) -> float:
        base = float(model.get("base_rate", 0.0))
        reload_time = float(model.get("reload_time", 1.0))
        last_t = max((float(row.get("t", 0.0)) for row in source_history), default=0.0)
        fluence = float(source_history[-1].get("fluence", 0.0)) if source_history else 0.0
        recovery = max(0.0, 1.0 - math.exp(-max(0.0, float(t) - last_t) / max(reload_time + fluence, 1.0e-12)))
        mark_weight = float(mark.get("weight", 1.0))
        return max(0.0, base * recovery * mark_weight)

    def compensator(
        self,
        source_history: list[dict[str, Any]],
        exposure: dict[str, Any],
        model: dict[str, Any],
        *,
        grid: int = 64,
    ) -> float:
        start = float(exposure.get("T_start", 0.0))
        stop = float(exposure.get("T_stop", exposure.get("duration", 1.0)))
        if stop <= start:
            return 0.0
        dt = (stop - start) / max(1, int(grid))
        total = 0.0
        history_prefix = [dict(row) for row in source_history]
        for index in range(max(1, int(grid))):
            t = start + (index + 0.5) * dt
            total += self.conditional_intensity(t, {"weight": 1.0}, history_prefix, model) * dt
        return total * float(exposure.get("detection_fraction", 1.0))

    def loglike(self, bursts: list[dict[str, Any]], exposure: dict[str, Any], model: dict[str, Any]) -> float:
        history: list[dict[str, Any]] = []
        event_term = 0.0
        for burst in sorted(bursts, key=lambda row: float(row.get("t", 0.0))):
            lam = max(self.conditional_intensity(float(burst.get("t", 0.0)), burst, history, model), 1.0e-300)
            event_term += math.log(lam)
            history.append(dict(burst))
        return event_term - self.compensator(history, exposure, model)


def heldout_gain(
    oph_loglike: float,
    control_loglikes: list[float],
    delta_min: float,
) -> dict[str, Any]:
    best_control = max(control_loglikes) if control_loglikes else float("-inf")
    gain = float(oph_loglike) - best_control
    return {
        "oph_loglike": float(oph_loglike),
        "best_control_loglike": best_control,
        "delta_min": float(delta_min),
        "heldout_gain": gain,
        "CONTROL_MODEL_RECEIPT": gain > float(delta_min),
    }
