from __future__ import annotations

from typing import Any

import numpy as np

from oph_fpe.claims import (
    DECLARED_SHAPE_SUBSTRATE_WITNESS,
    SHAPE_LOOP_PARTICLE_RECEIPT,
)


def loop_mode_energy(amp: np.ndarray, arc_ids: list[int], mode: int) -> float:
    arc_values = np.asarray(amp)[np.asarray(arc_ids, dtype=np.int64)]
    loop_length = len(arc_ids)
    phases = np.exp(-2j * np.pi * int(mode) * np.arange(loop_length) / loop_length)
    coeff = np.sum(phases * arc_values) / max(loop_length, 1)
    return float(abs(coeff) ** 2)


def detect_loop_particles(
    amp: np.ndarray,
    face_arc_ids: list[list[int]],
    *,
    energy_threshold: float,
    persistence: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    particles: list[dict[str, Any]] = []
    persistence = persistence or {}
    for face_id, arc_ids in enumerate(face_arc_ids):
        if len(arc_ids) < 3:
            continue
        energies = {
            mode: loop_mode_energy(amp, arc_ids, mode)
            for mode in range(1, len(arc_ids))
        }
        best_mode = max(energies, key=energies.get)
        best_energy = float(energies[best_mode])
        if best_energy >= float(energy_threshold):
            particles.append(
                {
                    "face_id": int(face_id),
                    "loop_length": int(len(arc_ids)),
                    "mode": int(best_mode),
                    "energy": best_energy,
                    "particle_class": f"pentagon_mode_{best_mode}",
                    "persistence": int(persistence.get(int(face_id), 1)),
                }
            )
    return particles


def shape_loop_particle_receipt(
    particle_tracks: list[dict[str, Any]],
    *,
    min_lifetime: int = 8,
) -> dict[str, Any]:
    persistent = [
        track
        for track in particle_tracks
        if int(track.get("lifetime", track.get("persistence", 0))) >= int(min_lifetime)
        and bool(track.get("class_preserved", True))
    ]
    return {
        "receipt": SHAPE_LOOP_PARTICLE_RECEIPT,
        "receipt_name": SHAPE_LOOP_PARTICLE_RECEIPT,
        "persistent_loop_particle_count": len(persistent),
        "particle_track_count": len(particle_tracks),
        "passed": len(persistent) >= 1,
        "claim_level": DECLARED_SHAPE_SUBSTRATE_WITNESS,
        "neutral_oph_bulk_claim": False,
        "standard_model_particle_claim": False,
        "physical_cmb_prediction": False,
    }
