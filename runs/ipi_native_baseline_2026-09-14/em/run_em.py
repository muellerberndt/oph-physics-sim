#!/usr/bin/env python3
"""Replay the supplied Maxwell instrument; this is not a native-repair EoS.

Run with a sibling reverse-engineering-reality checkout. Only this bundle's
output directory is written; canonical producer receipts are never changed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction as Q
import hashlib
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, default=str,
                       allow_nan=False) + "\n").encode("utf-8")


def dot(left, right):
    return sum((a*b for a, b in zip(left, right, strict=True)), Q(0))


def git_head(root):
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True).strip()


def main():
    here = Path(__file__).resolve().parent
    simulator = here.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, default=simulator)
    parser.add_argument("--rer-root", type=Path,
                        default=simulator.parent/"reverse-engineering-reality")
    parser.add_argument("--output-dir", type=Path, default=here)
    args = parser.parse_args()
    simulator = args.sim_root.resolve()
    rer = args.rer_root.resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(rer/"code/electromagnetism"))
    producer = importlib.import_module("serial_maxwell_readout")
    verifier = importlib.import_module("verify_serial_maxwell_readout")

    # Call the API rather than the producer CLI, which overwrites its canonical
    # receipt. The verifier independently reconstructs the matrices and action.
    raw = producer.build()
    verified = verifier.verify(raw)
    mutant = deepcopy(raw)
    public_mutant = mutant["executions"][0]["events"][-1]["writes"]
    public_mutant["E/0/0"] = str(Q(public_mutant["E/0/0"])+1)
    try:
        verifier.verify(mutant)
    except ValueError as error:
        mutation = {"tampered_electric_readout_rejected": True, "reason": str(error)}
    else:
        raise ValueError("Independent verifier accepted a changed electric readout")

    public = raw["executions"][0]["events"][-1]["writes"]
    state = {}
    for event in raw["executions"][0]["events"]:
        state.update(event["writes"])
    h = Q(state["h"])
    electric = [[Q(public[f"E/{n}/{e}"]) for e in range(30)] for n in range(2)]
    magnetic = [[Q(public[f"B/{n}/{f}"]) for f in range(20)] for n in (1, 2)]
    current = [Q(public[f"public_J/0/{e}"]) for e in range(30)]
    _, _, c = verifier.carrier()
    curl = [[Q(str(c[f, e])) for e in range(30)] for f in range(20)]
    a0 = [Q(state[f"d/0/{12+e}"]) for e in range(30)]
    b0 = [dot(row, a0) for row in curl]
    delta_e = [(b-a)/h for a, b in zip(electric[0], electric[1], strict=True)]
    curl_b = [sum((curl[f][e]*magnetic[0][f] for f in range(20)), Q(0))
              for e in range(30)]
    residual = [de-cb+j for de, cb, j in zip(delta_e, curl_b, current, strict=True)]
    if any(residual):
        raise ValueError("Nonzero exact Ampere residual")
    form0 = (dot(electric[0], electric[0])+dot(b0, magnetic[0]))/2
    form1 = (dot(electric[1], electric[1])+dot(magnetic[0], magnetic[1]))/2
    work = -h*dot([a+b for a, b in zip(electric[0], electric[1], strict=True)], current)/2
    if form1-form0 != work:
        raise ValueError("Nonzero exact supplied-model field-form work residual")

    pin_head_matches = {}
    for path, digest in raw["pins"].items():
        data = subprocess.check_output(["git", "-C", str(rer), "show", f"HEAD:{path}"])
        pin_head_matches[path] = hashlib.sha256(data).hexdigest() == digest
    source = {"reverse_engineering_reality_commit": git_head(rer),
              "oph_physics_sim_commit": git_head(simulator),
              "provider_files_match_recorded_rer_head": pin_head_matches,
              "python": platform.python_version(),
              "sympy": importlib.import_module("sympy").__version__,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    verification = {"schema": "oph.ipi.supplied_em_verification.v1",
                    "verified": True, "independent_verifier":
                    "reverse-engineering-reality/code/electromagnetism/verify_serial_maxwell_readout.py",
                    "independent_replay": verified, "mutation_control": mutation,
                    "exact_ampere_residual_all_zero": True,
                    "exact_field_form_work_balance": True, "provenance": source}
    summary = {
        "schema": "oph.ipi.supplied_em_summary.v1", "scope": raw["scope"],
        "native_repair_eos_demonstrated": False,
        "native_repair_displacement_current_demonstrated": False,
        "supplied_model_displacement_term_nonzero": any(delta_e),
        "interpretation": "delta_E/h is the displacement term in the separately supplied dimensionless Maxwell law; no physical D, SI units or constitutive response is identified",
        "assumptions": raw["assumptions"], "h": str(h),
        "events_per_execution": verified["events"],
        "probe_feedback_cycles_per_execution": verified["cycles"],
        "gauge_control_executions": len(raw["executions"]),
        "sample_zero_conduction_current_seam": {"seam": 1,
            "E0": str(electric[0][1]), "E1": str(electric[1][1]),
            "J0": str(current[1]), "delta_E_over_h": str(delta_e[1]),
            "curl_B1": str(curl_b[1]), "ampere_residual": str(residual[1])},
        "all_seams_delta_E_over_h": [str(x) for x in delta_e],
        "supplied_model_field_form": {"definition": "H[n]=(||E[n]||^2+<B[n],B[n+1]>)/2",
            "H0": str(form0), "H1": str(form1), "source_work": str(work),
            "balance_residual": str(form1-form0-work),
            "interpretation": "algebraic field quantity of the supplied finite Maxwell action, not a native substrate thermodynamic measurement"},
        "thermodynamic_output": {"pressure": None, "energy_density": None,
            "temperature": None, "equation_of_state_w": None,
            "reason": "This executable specifies no native substrate stress, thermodynamic volume, physical energy or equilibrium ensemble."},
        "provenance": source,
    }
    outputs = {"raw_execution.json": raw, "verification.json": verification,
               "summary.json": summary}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, value in outputs.items():
        data = encoded(value)
        (args.output_dir/name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    (args.output_dir/"sha256.json").write_bytes(encoded(hashes))
    print(json.dumps({"verified": True, "output_directory": str(args.output_dir),
        "events_per_execution": verified["events"],
        "zero_conduction_seam_displacement_term": str(delta_e[1]),
        "thermodynamic_eos_w": None}, indent=2))


if __name__ == "__main__":
    main()
