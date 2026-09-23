"""Capture the registered C12 operation-to-ledger map without success filtering.

The existing source driver is executed unchanged. The compact export contains
its complete repair-log commitment and the inputs needed for independent replay.
Constructed instrument and archive controls are exported separately; none is
substituted into the native driver or identified with selected M1 geometry.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from copy import deepcopy

import numpy as np

from oph_fpe.bulk.physical_h3_kms_source_capture import (
    _build_federation, _normalize_config, _source_dynamics, _topology_seed,
    capture_physical_source,
    _observer_loop, _sha,
)
from oph_fpe.core.echosahedral_dynamics import (
    LocalRecurrentCarrierState, propagate_local_recurrent_carriers,
    reference_icosahedral_coupling,
)
from oph_fpe.core.icosahedral import icosahedral_a5_port_permutations

from oph_fpe.bulk.primitive_source_instruments import instrument_control

ROOT = Path(__file__).resolve().parents[2]
PLAN = ((4, 4, 1), (4, 8, 1), (4, 16, 1), (4, 32, 1),
        (8, 16, 1), (16, 16, 1), (4, 16, 2))
STEP = 2.0**-17


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False)+"\n").encode("ascii")


def source_files():
    """Conservative static closure of local Python imports, including initializers."""
    pending = ["oph_fpe.bulk.primitive_source_reads",
               "oph_fpe.bulk.verify_primitive_source_reads_independent"]
    found = set()
    while pending:
        module = pending.pop()
        path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
        if not path.is_file():
            path = ROOT.joinpath(*module.split("."), "__init__.py")
        if not path.is_file() or path in found:
            continue
        found.add(path)
        parts = path.relative_to(ROOT).parts
        for length in range(1, len(parts)):
            init = ROOT.joinpath(*parts[:length], "__init__.py")
            if init.is_file() and init not in found:
                pending.append(".".join((*parts[:length], "__init__")))
        package = (module if path.name == "__init__.py" and not module.endswith(".__init__")
                   else module.rsplit(".", 1)[0])
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                pending.extend(a.name for a in node.names if a.name.startswith("oph_"))
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                if node.level:
                    base = package.split(".")[:len(package.split("."))-node.level+1]
                    name = ".".join(base+([name] if name else []))
                if name.startswith("oph_"):
                    pending.append(name)
                    pending.extend(name+"."+a.name for a in node.names)
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(found)}


def case(count, cycles, level):
    config = _normalize_config({"carrier_count": count, "seed": 1729,
        "propagation_steps": 1, "intrinsic_step": STEP, "cycles": cycles,
        "record_commit_cycles": min(4, cycles), "support_refinement_level": level})
    federation = _build_federation(config)
    dynamics, recurrent, _, final, _ = _source_dynamics(config, federation)
    if not dynamics["repair_event_examples_complete"]:
        raise ValueError("a complete native repair log is required")
    index = {c.carrier_id: i for i, c in enumerate(federation.carriers)}
    seams = [[s.seam_id, 12*index[s.left_carrier_id]+s.left_ports[0],
              12*index[s.right_carrier_id]+s.right_ports[0]] for s in federation.seams]
    order = np.random.Generator(np.random.PCG64(_topology_seed(config, "repair-order"))).permutation(len(seams))
    initial = np.abs(recurrent)**2
    # Bind the projection to the complete driver too, including its actual
    # observer loop and source-state root. Do not synthesize observer payloads.
    full = capture_physical_source(config)["source_artifacts"]
    if full["dynamics"]["repair_event_log_sha256"] != dynamics["repair_event_log_sha256"]:
        raise ValueError("complete-driver repair binding")
    records = [r for r in full["observer_log"]["events"]
               if r["kind"] == "RECORD_COMMIT" and r["observer_token"] == "observer-0000"]
    if not records or any(r["source_state_root"] != full["source_state_root_sha256"] for r in records):
        raise ValueError("observer global-root binding")
    return {"carriers": count, "cycles": cycles, "support_level": level,
            "seams": seams, "order": order.tolist(),
            "initial_hex": [float(x).hex() for x in initial.flat],
            "final_hex": [float(x).hex() for x in final.flat],
            "commits": dynamics["repair_event_count"], "noops": dynamics["repair_noop_count"],
            "native_log_sha256": dynamics["repair_event_log_sha256"],
            "cycle_counts": [[r["committed_transaction_count"], r["skipped_noop_count"]]
                             for r in dynamics["repair_cycle_ledger"]],
            "observer_records": [[r["sample"], index[r["carrier_id"]], r["record_cycle"], r["port"],
                                  [float(v).hex() for v in r["full_port_state"]]] for r in records]}


def quantum_control():
    laplacian = reference_icosahedral_coupling()
    if not np.array_equal(laplacian, laplacian.astype(int)):
        raise ValueError("nonintegral source generator")
    basis = LocalRecurrentCarrierState(np.eye(12, dtype=complex), np.zeros(12))
    # Row p of this executed result is U|p>; transpose to output-input indexing.
    unitary = propagate_local_recurrent_carriers(basis, intrinsic_step=STEP).amplitudes.T
    neighbor = int(np.flatnonzero(laplacian[0] == -1)[0])
    states = np.zeros((2, 12), dtype=complex)
    states[:, 0] = 1/np.sqrt(2)
    states[:, neighbor] = (1j/np.sqrt(2), -1j/np.sqrt(2))
    result = propagate_local_recurrent_carriers(
        LocalRecurrentCarrierState(states, np.zeros(2)), intrinsic_step=STEP)
    config = _normalize_config({"carrier_count": 4, "seed": 1729,
                                "cycles": 4, "support_refinement_level": 1})
    federation = _build_federation(config)
    dynamics = deepcopy(_source_dynamics(config, federation)[0])
    # A controlled, admissible observer-input snapshot. This is not claimed
    # to be produced by a particular integer initialization seed.
    uniform = [1/12]*12
    for snapshot in dynamics["record_state_snapshots"]:
        for row in snapshot["carrier_rows"]:
            row["full_port_state"] = uniform[:]
            row["full_port_state_sha256"] = _sha(uniform)
    observer = _observer_loop(config, federation, dynamics, "sha256:"+"0"*64)
    feedback = next(e for e in observer["events"] if e["kind"] == "LOCAL_FEEDBACK")
    next_port = feedback["observed_action_material_next_port"]
    permutation = next(p for p in icosahedral_a5_port_permutations()
                       if p[0] == 0 and p[next_port] != next_port
                       and all(p[p[p[p[p[i]]]]] == i for i in range(12)))
    return {"laplacian": laplacian.astype(int).tolist(), "step": "1/131072",
            "unitary_hex": [[[float(z.real).hex(), float(z.imag).hex()] for z in row] for row in unitary],
            "phase_ports": [0, neighbor],
            "phase_output_hex": [float(abs(row[0])**2).hex() for row in result.amplitudes],
            "uniform_snapshot_feedback": {"input_port": 0, "native_next_port": next_port,
                "stabilizer_permutation": list(permutation)}}


def produce():
    packet = {"schema": "oph.primitive-source-reads.v2",
        "source": {"repository": "https://github.com/muellerberndt/oph-physics-sim",
                   "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "files": source_files()},
        "scope": {"driver": "registered_all_port_capture", "complete_A1_A3": False,
                  "M1_derived": False, "record_channel": "local_numeric_ledger_payload",
                  "global_custody_hash_is_local_readout": False},
        "cases": [case(*spec) for spec in PLAN], "quantum": quantum_control()}
    packet["instruments"] = instrument_control(packet["cases"])
    packet["sha256"] = hashlib.sha256(canonical(packet)).hexdigest()
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    packet = produce()
    from oph_fpe.bulk.verify_primitive_source_reads_independent import verify
    receipt = verify(packet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(packet))
    print(canonical(receipt).decode("ascii"), end="")


if __name__ == "__main__":
    main()
