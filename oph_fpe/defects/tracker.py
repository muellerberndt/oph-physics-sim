from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from oph_fpe.defects.holonomy import HolonomyDefect


@dataclass
class DefectTrack:
    particle_id: str
    cycle_key: tuple[int, ...]
    holonomy_label: str
    birth_cycle: int
    last_cycle: int
    support_size: int
    observations: list[dict[str, Any]] = field(default_factory=list)


class DefectTracker:
    def __init__(self):
        self._tracks: dict[tuple[int, ...], DefectTrack] = {}
        self._next_id = 0

    def update(self, cycle: int, defects: list[HolonomyDefect]) -> None:
        for defect in defects:
            track = self._tracks.get(defect.cycle_key)
            if track is None:
                track = DefectTrack(
                    particle_id=f"defect_{self._next_id:06d}",
                    cycle_key=defect.cycle_key,
                    holonomy_label=defect.holonomy_label,
                    birth_cycle=cycle,
                    last_cycle=cycle,
                    support_size=defect.support_size,
                )
                self._tracks[defect.cycle_key] = track
                self._next_id += 1
            track.last_cycle = cycle
            track.observations.append(
                {
                    "cycle": cycle,
                    "ordered_cycle": list(defect.ordered_cycle),
                    "holonomy_class": defect.holonomy_label,
                    "support_size": defect.support_size,
                }
            )

    def worldlines(self) -> list[dict[str, Any]]:
        return [
            {
                "particle_id": track.particle_id,
                "birth_cycle": track.birth_cycle,
                "death_cycle": track.last_cycle,
                "support_size": track.support_size,
                "holonomy_class": track.holonomy_label,
                "cycle_key": list(track.cycle_key),
                "worldline": track.observations,
            }
            for track in sorted(self._tracks.values(), key=lambda item: item.particle_id)
        ]
