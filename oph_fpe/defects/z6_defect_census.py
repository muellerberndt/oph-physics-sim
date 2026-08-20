"""Defect census pipeline on the committed base carrier (lane C6,
exploratory, non-evidential).

Design record: ``oph_fpe/defects/DESIGN.md`` sections 5, 6, and 8, fixed
before this module. Repair rule ``single_seam_strict_local_descent.v1``;
label readout ``total_chord_holonomy_with_canonical_quotients.v1``. The
census is a census of conserved sector classes (chord-holonomy 19-tuples)
realized as repair fixed points of seeded ensembles, with A5-orbit data.

This module contains no committed matter-table values, no descent
congruence, and no reference to expected occupancy; the comparison stage is
the separate module ``z6_matter_grammar_verifier.py``, run after census
receipts exist (DESIGN.md section 10).

Boundaries: exploratory and non-evidential; a defect class is a finite
combinatorial invariant, not a particle; the label is a declared
convention; no instrument is frozen or armed.
"""

from __future__ import annotations

import json
import random
from typing import Sequence

from oph_fpe.defects.z6_a5_action import (
    rotation_group,
    rotation_group_receipt,
    sector_orbit,
)
from oph_fpe.defects.z6_carrier_defects import (
    MOD,
    CarrierSpec,
    Config,
    SectorClass,
    base_carrier_spec,
    chord_holonomies,
    circle_distance,
    conservation_receipt,
    face_curvature,
    mismatch_energy,
    seam_faces,
    sector_representative,
    structural_receipt,
)

REPAIR_RULE = "single_seam_strict_local_descent.v1"
LABEL_READOUT = "total_chord_holonomy_with_canonical_quotients.v1"
CENSUS_SCHEMA = "oph.sim.defect_census.z6_carrier.v1"

# Declared ensemble streams (DESIGN.md section 8): name, seed, size.
DECLARED_STREAMS: tuple[tuple[str, int, int], ...] = (
    ("uniform_iid", 20260820, 160),
    ("sparse_pair", 20260821, 80),
)


# ---------------------------------------------------------------------------
# Repair rule
# ---------------------------------------------------------------------------

def _best_move(f1_value: int, f2_value: int, sign: int
               ) -> tuple[int, int]:
    """``(delta, dE)`` for the locally best strict-descent increment at one
    seam; ``delta = 0`` when no strict decrease exists. Tie-break: smaller
    ``rho(delta)``, then smaller ``delta``."""
    base = circle_distance(f1_value) + circle_distance(f2_value)
    best_delta, best_de, best_rho = 0, 0, 0
    for delta in range(1, MOD):
        de = (
            circle_distance(f1_value + sign * delta)
            + circle_distance(f2_value - sign * delta)
            - base
        )
        rho = circle_distance(delta)
        if de < best_de or (
            de == best_de and best_delta != 0
            and (rho, delta) < (best_rho, best_delta)
        ):
            best_delta, best_de, best_rho = delta, de, rho
    if best_de < 0:
        return best_delta, best_de
    return 0, 0


def repair(spec: CarrierSpec, config: Sequence[int]
           ) -> tuple[Config, list[tuple[int, int]]]:
    """Evolve to the repair fixed point; returns ``(fixed_config, trace)``
    with the trace the ordered list of applied ``(seam, delta)`` moves.

    Gauge-covariant: every decision reads only the two incident face
    curvatures and the incidence sign (DESIGN.md section 6 item 2)."""
    faces = seam_faces(spec)
    state = [x % MOD for x in config]
    curvature = face_curvature(spec, state)
    trace: list[tuple[int, int]] = []
    changed = True
    while changed:
        changed = False
        for e in range(spec.seams):
            f1, f2, sign = faces[e]
            delta, de = _best_move(curvature[f1], curvature[f2], sign)
            if delta == 0:
                continue
            state[e] = (state[e] + delta) % MOD
            curvature[f1] = (curvature[f1] + sign * delta) % MOD
            curvature[f2] = (curvature[f2] - sign * delta) % MOD
            trace.append((e, delta))
            changed = True
            assert de < 0
    return state, trace


def is_stable(spec: CarrierSpec, curvature: Sequence[int]) -> bool:
    """No single-seam move strictly reduces the mismatch energy: a sector
    property (curvature is a sector invariant)."""
    for f1, f2, sign in seam_faces(spec):
        _, de = _best_move(curvature[f1], curvature[f2], sign)
        if de < 0:
            return False
    return True


def neutral_escapable(spec: CarrierSpec, curvature: Sequence[int]) -> bool:
    """One energy-neutral single-seam move unlocks a strictly improving
    second move (DESIGN.md section 6 item 5); a sector property."""
    faces = seam_faces(spec)
    base = list(curvature)
    for f1, f2, sign in faces:
        cost = circle_distance(base[f1]) + circle_distance(base[f2])
        for delta in range(1, MOD):
            after1 = (base[f1] + sign * delta) % MOD
            after2 = (base[f2] - sign * delta) % MOD
            if circle_distance(after1) + circle_distance(after2) != cost:
                continue
            probe = list(base)
            probe[f1], probe[f2] = after1, after2
            if not is_stable(spec, probe):
                return True
    return False


# ---------------------------------------------------------------------------
# Label readout (declared convention, DESIGN.md section 8)
# ---------------------------------------------------------------------------

def sector_label(sector: Sequence[int]) -> tuple[int, int, int]:
    """``(q, t, d)`` with ``q`` the total chord holonomy mod 6 and ``t, d``
    its canonical Z/3 and Z/2 quotients."""
    q = sum(sector) % MOD
    return (q, q % 3, q % 2)


# ---------------------------------------------------------------------------
# Ensembles (named, seeded streams)
# ---------------------------------------------------------------------------

def sample_stream(spec: CarrierSpec, name: str, seed: int,
                  size: int) -> list[Config]:
    rng = random.Random(seed)
    members: list[Config] = []
    for _ in range(size):
        if name == "uniform_iid":
            members.append([rng.randrange(MOD) for _ in range(spec.seams)])
        elif name == "sparse_pair":
            config = [0] * spec.seams
            first, second = rng.sample(range(spec.seams), 2)
            config[first] = rng.randrange(1, MOD)
            config[second] = rng.randrange(1, MOD)
            members.append(config)
        else:
            raise ValueError(f"undeclared stream: {name}")
    return members


# ---------------------------------------------------------------------------
# Census pipeline
# ---------------------------------------------------------------------------

def run_census(spec: CarrierSpec | None = None,
               streams: Sequence[tuple[str, int, int]] = DECLARED_STREAMS
               ) -> dict:
    """Run the DESIGN.md section 8 census and return the receipt dict.

    Exact arithmetic mod 6 throughout; seeded streams; deterministic."""
    spec = base_carrier_spec() if spec is None else spec
    carrier_pin = structural_receipt(spec)
    conservation = conservation_receipt(spec)
    rotations = rotation_group(spec)
    rotation_receipt = rotation_group_receipt(spec, rotations)

    per_class: dict[SectorClass, dict] = {}
    vacuum = tuple([0] * len(spec.chords))
    vacuum_count = 0
    vacuum_by_stream = {name: 0 for name, _, _ in streams}
    sector_changed = 0
    total_members = 0

    for name, seed, size in streams:
        for config in sample_stream(spec, name, seed, size):
            total_members += 1
            initial = chord_holonomies(spec, config)
            fixed, _ = repair(spec, config)
            final = chord_holonomies(spec, fixed)
            if final != initial:
                sector_changed += 1
            if final == vacuum:
                vacuum_count += 1
                vacuum_by_stream[name] += 1
                continue
            entry = per_class.setdefault(final, {
                "multiplicity": 0,
                "by_stream": {n: 0 for n, _, _ in streams},
                "sector_unchanged": 0,
            })
            entry["multiplicity"] += 1
            entry["by_stream"][name] += 1
            if final == initial:
                entry["sector_unchanged"] += 1

    classes = []
    orbit_members: dict[SectorClass, list[SectorClass]] = {}
    for sector in sorted(per_class):
        rep = sector_representative(spec, sector)
        curvature = face_curvature(spec, rep)
        energy = mismatch_energy(spec, rep)
        stable = is_stable(spec, curvature)
        if not stable:
            raise AssertionError(
                "repair fixed point not stable: rule/receipt drift"
            )
        q, t, d = sector_label(sector)
        orbit_rep, orbit_size = sector_orbit(spec, rotations, sector)
        orbit_members.setdefault(orbit_rep, []).append(sector)
        classes.append({
            "sector": list(sector),
            "label_qtd": [q, t, d],
            "curvature": list(curvature),
            "energy": energy,
            "chord_support": sum(1 for x in sector if x != 0),
            "curvature_support": sum(1 for x in curvature if x != 0),
            "cycle_labels_nonzero": sorted(x for x in sector if x != 0),
            "multiplicity": per_class[sector]["multiplicity"],
            "by_stream": per_class[sector]["by_stream"],
            "sector_unchanged": per_class[sector]["sector_unchanged"],
            "stable": True,
            "neutral_escapable": neutral_escapable(spec, curvature),
            "orbit_representative": list(orbit_rep),
            "orbit_size": orbit_size,
        })
    classes.sort(key=lambda c: (-c["multiplicity"], c["sector"]))

    orbits = []
    for orbit_rep in sorted(orbit_members):
        members = sorted(orbit_members[orbit_rep])
        labels = sorted(
            list(sector_label(m)) for m in members
        )
        orbits.append({
            "orbit_representative": list(orbit_rep),
            "realized_classes": len(members),
            "total_multiplicity": sum(
                per_class[m]["multiplicity"] for m in members
            ),
            "label_multiset": labels,
        })

    return {
        "schema": CENSUS_SCHEMA,
        "exploratory": True,
        "evidential": False,
        "frozen": False,
        "instrument_armed": False,
        "repair_rule": REPAIR_RULE,
        "label_readout": LABEL_READOUT,
        "carrier": carrier_pin,
        "conservation_receipt": conservation,
        "rotation_group_receipt": rotation_receipt,
        "streams": [
            {"name": name, "seed": seed, "size": size}
            for name, seed, size in streams
        ],
        "members_total": total_members,
        "sector_changed_by_repair": sector_changed,
        "vacuum": {
            "sector": list(vacuum),
            "multiplicity": vacuum_count,
            "by_stream": vacuum_by_stream,
        },
        "defect_class_count": len(classes),
        "orbit_count": len(orbits),
        "classes": classes,
        "orbits": orbits,
        "statement": (
            "exploratory, non-evidential defect census on the committed "
            "base carrier; sector classes are finite combinatorial "
            "invariants, labels are the declared DESIGN.md readout, and "
            "no physical particle claim is made"
        ),
    }


def census_json(receipt: dict) -> str:
    return json.dumps(receipt, sort_keys=True, indent=1)


if __name__ == "__main__":  # pragma: no cover - exploratory entry point
    print(census_json(run_census()))
