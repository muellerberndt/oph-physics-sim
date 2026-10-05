"""Bounded diagnostic of the existing scalar seam-mean primitive; no EoS claim.

Run with Python >=3.11 in an installed oph-physics-sim checkout, or supply
--sim-root. Exact fractions are authoritative; decimal fields aid reading.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("native_trace.json"))
    args = parser.parse_args()
    root = args.sim_root.resolve()
    sys.path.insert(0, str(root))
    from oph_fpe.dynamics.canonical_seam_repair import (
        reference_edges, edge_conditional_expectation, apply_fraction_matrix,
    )

    edges = reference_edges()
    matrices = {edge: edge_conditional_expectation(edge) for edge in edges}
    pins = {}
    for rel in ("oph_fpe/dynamics/canonical_seam_repair.py", "oph_fpe/core/icosahedral.py"):
        pins[rel] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    configurations = [
        ("pulse_repair_seed7", [12] + [0]*11, True, 7),
        ("pulse_repair_seed11", [12] + [0]*11, True, 11),
        ("constant_repair_control", [1]*12, True, 7),
        ("pulse_no_repair_control", [12] + [0]*11, False, 7),
    ]
    cases = []
    for name, initial, enabled, seed in configurations:
        rng = random.Random(seed)
        schedule = []
        for _ in range(10):
            sweep = list(edges)
            rng.shuffle(sweep)
            schedule.extend(sweep)
        state = tuple(map(F, initial))
        total = sum(state)
        events = []
        seam_increases = 0
        def squared_norm(values):
            return sum(x*x for x in values)
        def seam_mismatch(values):
            return sum((values[i]-values[j])**2 for i, j in edges)
        for step, edge in enumerate(schedule, 1):
            old = state
            state = apply_fraction_matrix(matrices[edge], old) if enabled else old
            delta_v = squared_norm(state)-squared_norm(old)
            expected_delta = -(old[edge[0]]-old[edge[1]])**2/2 if enabled else F(0)
            assert sum(state) == total and delta_v == expected_delta and delta_v <= 0
            seam_increases += seam_mismatch(state) > seam_mismatch(old)
            events.append({
                "attempt": step, "seam": list(edge),
                "readback_before": [str(old[i]) for i in edge],
                "state_after": list(map(str, state)),
                "sum_load": str(sum(state)), "squared_norm": str(squared_norm(state)),
                "delta_squared_norm": str(delta_v),
                "seam_mismatch": str(seam_mismatch(state)),
            })
        final_v = squared_norm(state)
        residual = final_v-total*total/len(state)
        cases.append({
            "name": name, "repair_enabled": enabled, "schedule_seed": seed,
            "initial_state": initial, "events": events,
            "summary": {
                "attempts": len(events), "sum_load_initial": str(total),
                "sum_load_final": str(sum(state)),
                "squared_norm_initial": str(sum(x*x for x in initial)),
                "squared_norm_final": str(final_v),
                "squared_norm_final_decimal": float(final_v),
                "centered_squared_norm_final": str(residual),
                "centered_squared_norm_final_decimal": float(residual),
                "exact_sum_conservation": True, "exact_local_norm_identity": True,
                "seam_mismatch_increase_count": seam_increases,
            },
        })
    result = {
        "schema": "oph.ipi.native_scalar_diagnostic.v1",
        "scope": "One isolated twelve-port scalar carrier; declared seam-mean repair only. "
                 "External diagnostic records its reads and writes. This is not a full "
                 "self-reading federation, thermodynamic measurement, or Maxwell derivation.",
        "units": "scalar load units and update attempts; no physical energy or time identification",
        "schedule": "ten independently shuffled sweeps, each visiting all thirty seams once; "
                    "a declared diagnostic schedule, not IID seam sampling or physical time",
        "stopping_rule": "exactly 300 attempts for every case; no convergence-based stopping",
        "ports": 12, "seams": [list(e) for e in edges],
        "thermodynamics": {"pressure": None, "energy_density": None, "w": None,
                           "reason": "No thermodynamic stress, volume/work law, or energy-density "
                                     "identification is provided by this primitive. Null is not zero."},
        "electromagnetism": {"displacement_current": None,
                              "reason": "No electric or magnetic field update in this primitive."},
        "provenance": {
            "repository": "https://github.com/muellerberndt/oph-physics-sim",
            "head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": pins,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "networkx")},
            "source_diff": subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", *pins], text=True),
        },
        "cases": cases,
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.out), "cases": [
        {"name": c["name"], **c["summary"]} for c in cases]}, indent=2))


if __name__ == "__main__":
    main()
