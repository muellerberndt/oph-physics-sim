from __future__ import annotations

from typing import Any


class DetectionThinner:
    def detection_probability(self, packet: dict[str, Any], obs_window: dict[str, Any]) -> float:
        fluence = float(packet.get("fluence", packet.get("signal", 0.0)))
        threshold = float(obs_window.get("fluence_threshold", obs_window.get("threshold", 1.0)))
        duty = min(1.0, max(0.0, float(obs_window.get("duty_cycle", 1.0))))
        if threshold <= 0.0:
            return duty
        return max(0.0, min(1.0, duty * fluence / threshold))

    def thin(self, packet: dict[str, Any], obs_window: dict[str, Any], rng: Any) -> dict[str, Any] | None:
        p_det = self.detection_probability(packet, obs_window)
        draw = float(rng.random()) if hasattr(rng, "random") else float(rng())
        if draw <= p_det:
            return {"detected": True, "p_det": p_det, "packet": dict(packet)}
        return None


class CensoringModel:
    def upper_limit(self, packet: dict[str, Any], obs_window: dict[str, Any]) -> dict[str, Any]:
        threshold = float(obs_window.get("fluence_threshold", obs_window.get("threshold", 1.0)))
        return {
            "record_type": "upper_limit",
            "source_id": packet.get("source_id"),
            "upper_limit": threshold,
            "obs_window_id": obs_window.get("obs_window_id", "obs_window"),
        }

    def nondetection_record(self, exposure: dict[str, Any], source_id: str) -> dict[str, Any]:
        return {
            "record_type": "nondetection",
            "source_id": source_id,
            "exposure": dict(exposure),
        }
