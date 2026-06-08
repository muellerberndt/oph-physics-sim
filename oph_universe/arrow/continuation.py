from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oph_universe.arrow.entropy import landauer_record_bound_ok


@dataclass
class ContinuationStep:
    t0: int
    t1: int
    touched_patches: tuple[int, ...]
    phi_before: float
    phi_after: float
    delta_record_payload_bits: float
    blank_negentropy_consumed_bits: float
    entropy_exported_bits: float
    preexisting_provenance_bits: float
    records_written: tuple[str, ...]
    accepted: bool


@dataclass
class ContinuationLaw:
    law_id: str
    schedule_class_id: str
    allow_record_write: bool = True
    enforce_landauer: bool = True
    enforce_phi_descent: bool = True

    def step(self, universe: Any, rng: Any) -> ContinuationStep:
        phi_before = float(getattr(universe, "phi", 0.0))
        delta_payload = float(getattr(universe, "next_payload_bits", 0.0)) if self.allow_record_write else 0.0
        blank = min(float(getattr(universe, "blank_negentropy_bits", 0.0)), delta_payload)
        export = max(0.0, delta_payload - blank)
        phi_after = max(0.0, phi_before - float(getattr(universe, "repair_delta", 1.0)))
        accepted = True
        if self.enforce_phi_descent and phi_after > phi_before:
            accepted = False
        if self.enforce_landauer and not landauer_record_bound_ok(delta_payload, blank, export):
            accepted = False
        if accepted:
            universe.phi = phi_after
            universe.blank_negentropy_bits = float(getattr(universe, "blank_negentropy_bits", 0.0)) - blank
            universe.t = int(getattr(universe, "t", 0)) + 1
        record_id = f"record:{getattr(universe, 't', 0)}" if accepted and delta_payload > 0 else ""
        return ContinuationStep(
            t0=int(getattr(universe, "t", 0)) - int(accepted),
            t1=int(getattr(universe, "t", 0)),
            touched_patches=tuple(getattr(universe, "touched_patches", ())),
            phi_before=phi_before,
            phi_after=phi_after,
            delta_record_payload_bits=delta_payload,
            blank_negentropy_consumed_bits=blank,
            entropy_exported_bits=export,
            preexisting_provenance_bits=float(getattr(universe, "preexisting_provenance_bits", 0.0)),
            records_written=(record_id,) if record_id else (),
            accepted=accepted,
        )

