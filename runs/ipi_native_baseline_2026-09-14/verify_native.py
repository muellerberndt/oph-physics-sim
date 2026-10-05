"""Independent stdlib replay of native_trace.json; no simulator imports.

The carrier is reconstructed from exact golden-ratio vertex distances. The
transition verifier replaces two endpoints directly, without calling the
producer's matrix builder or matrix application. Hash checks bind the driver
and, when --sim-root is supplied, the two named simulator source files.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def keys(value, expected, context):
    require(isinstance(value, dict) and set(value) == set(expected), context + ": schema keys differ")


def rational(value, context):
    require(isinstance(value, str), context + ": exact rational must be a string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise VerificationError(context + ": invalid rational") from exc
    require(str(result) == value, context + ": rational is not canonical")
    return result


def rational_equal(value, expected, context):
    require(rational(value, context) == expected, context + ": exact rational differs")


def carrier_edges():
    """Distance squared 4 identifies edges before vertex normalization.

    Each coordinate is (a,b), representing a+b*phi, with phi^2=phi+1.
    This does not read the producer's edge/face list or any simulator code.
    """
    zero, one, negative, phi, negative_phi = (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)
    vertices = (
        (negative, phi, zero), (one, phi, zero),
        (negative, negative_phi, zero), (one, negative_phi, zero),
        (zero, negative, phi), (zero, one, phi),
        (zero, negative, negative_phi), (zero, one, negative_phi),
        (phi, zero, negative), (phi, zero, one),
        (negative_phi, zero, negative), (negative_phi, zero, one),
    )
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            constant = linear = 0
            for (a, b), (c, d) in zip(vertices[i], vertices[j]):
                u, v = a - c, b - d
                constant += u*u + v*v
                linear += 2*u*v + v*v
            if (constant, linear) == (4, 0):
                edges.append((i, j))
    require(len(edges) == 30, "independent carrier does not have thirty edges")
    edge_set = set(edges)
    require(all(sum(i in edge for edge in edges) == 5 for i in range(12)), "independent carrier is not five-regular")
    triangles = sum(
        (i, j) in edge_set and (i, k) in edge_set and (j, k) in edge_set
        for i in range(12) for j in range(i+1, 12) for k in range(j+1, 12)
    )
    require(triangles == 20, "independent carrier does not have twenty triangles")
    return tuple(edges)


def norm(values):
    return sum(value*value for value in values)


def seam_norm(values, edges):
    return sum((values[i]-values[j])**2 for i, j in edges)


EXPECTED_METADATA = {
    "schema": "oph.ipi.native_scalar_diagnostic.v1",
    "scope": "One isolated twelve-port scalar carrier; declared seam-mean repair only. "
             "External diagnostic records its reads and writes. This is not a full "
             "self-reading federation, thermodynamic measurement, or Maxwell derivation.",
    "units": "scalar load units and update attempts; no physical energy or time identification",
    "schedule": "ten independently shuffled sweeps, each visiting all thirty seams once; "
                "a declared diagnostic schedule, not IID seam sampling or physical time",
    "stopping_rule": "exactly 300 attempts for every case; no convergence-based stopping",
    "thermodynamics": {
        "pressure": None, "energy_density": None, "w": None,
        "reason": "No thermodynamic stress, volume/work law, or energy-density "
                  "identification is provided by this primitive. Null is not zero.",
    },
    "electromagnetism": {
        "displacement_current": None,
        "reason": "No electric or magnetic field update in this primitive.",
    },
}
EXPECTED_CASES = (
    ("pulse_repair_seed7", [12]+[0]*11, True, 7),
    ("pulse_repair_seed11", [12]+[0]*11, True, 11),
    ("constant_repair_control", [1]*12, True, 7),
    ("pulse_no_repair_control", [12]+[0]*11, False, 7),
)
SOURCE_NAMES = (
    "oph_fpe/dynamics/canonical_seam_repair.py",
    "oph_fpe/core/icosahedral.py",
)


def verify_trace(report):
    keys(report, [*EXPECTED_METADATA, "ports", "seams", "provenance", "cases"], "root")
    for key, value in EXPECTED_METADATA.items():
        require(report[key] == value, key + ": scientific boundary or metadata differs")
    require(type(report["ports"]) is int and report["ports"] == 12, "port count differs")
    edges = carrier_edges()
    require(report["seams"] == [list(edge) for edge in edges], "reported seams differ from independent carrier")
    provenance = report["provenance"]
    keys(provenance, ["repository", "head", "source_sha256", "driver_sha256", "python", "packages", "source_diff"], "provenance")
    require(provenance["repository"] == "https://github.com/muellerberndt/oph-physics-sim", "repository differs")
    require(isinstance(provenance["head"], str) and re.fullmatch(r"[0-9a-f]{40}", provenance["head"]), "invalid Git commit")
    keys(provenance["source_sha256"], SOURCE_NAMES, "source pins")
    for name, digest in [*provenance["source_sha256"].items(), ("driver", provenance["driver_sha256"])]:
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), name + ": malformed hash")
    require(provenance["source_diff"] == "", "declared primitive differs from recorded Git HEAD")
    require(isinstance(provenance["python"], str) and provenance["python"], "Python version missing")
    keys(provenance["packages"], ["numpy", "scipy", "networkx"], "package versions")
    require(all(isinstance(value, str) and value for value in provenance["packages"].values()), "package version missing")
    cases = report["cases"]
    require(isinstance(cases, list) and len(cases) == 4, "exactly four fixed cases required")
    summaries = []
    for case, (name, initial, enabled, seed) in zip(cases, EXPECTED_CASES):
        keys(case, ["name", "repair_enabled", "schedule_seed", "initial_state", "events", "summary"], name)
        require(case["name"] == name, name + ": case name/order differs")
        require(case["repair_enabled"] is enabled, name + ": repair toggle differs")
        require(type(case["schedule_seed"]) is int and case["schedule_seed"] == seed, name + ": schedule seed differs")
        require(case["initial_state"] == initial and all(type(value) is int for value in case["initial_state"]), name + ": initial state differs")
        rng = random.Random(seed)
        schedule = []
        for _ in range(10):
            sweep = list(edges)
            rng.shuffle(sweep)
            schedule.extend(sweep)
        events = case["events"]
        require(isinstance(events, list) and len(events) == 300, name + ": stopping rule violated")
        state = list(map(Fraction, initial))
        initial_norm = norm(state)
        initial_total = sum(state)
        increases = 0
        changed_steps = 0
        for step, (event, edge) in enumerate(zip(events, schedule), 1):
            context = name + "/attempt=" + str(step)
            keys(event, ["attempt", "seam", "readback_before", "state_after", "sum_load", "squared_norm", "delta_squared_norm", "seam_mismatch"], context)
            require(type(event["attempt"]) is int and event["attempt"] == step, context + ": attempt number differs")
            require(event["seam"] == list(edge), context + ": seeded schedule differs")
            require(isinstance(event["readback_before"], list) and len(event["readback_before"]) == 2, context + ": readback shape differs")
            for raw, index in zip(event["readback_before"], edge):
                rational_equal(raw, state[index], context + ": readback")
            before = state.copy()
            left, right = edge
            if enabled:
                average = (before[left] + before[right])/2
                state[left] = average
                state[right] = average
            require(isinstance(event["state_after"], list) and len(event["state_after"]) == 12, context + ": state shape differs")
            for raw, expected in zip(event["state_after"], state):
                rational_equal(raw, expected, context + ": state")
            delta = norm(state)-norm(before)
            expected_delta = -(before[left]-before[right])**2/2 if enabled else Fraction(0)
            require(delta == expected_delta and delta <= 0, context + ": independent norm-drop identity failed")
            require(sum(state) == initial_total, context + ": independent conserved sum failed")
            rational_equal(event["sum_load"], initial_total, context + ": sum")
            rational_equal(event["squared_norm"], norm(state), context + ": norm")
            rational_equal(event["delta_squared_norm"], delta, context + ": norm increment")
            rational_equal(event["seam_mismatch"], seam_norm(state, edges), context + ": seam mismatch")
            increases += seam_norm(state, edges) > seam_norm(before, edges)
            changed_steps += state != before
        for sweep in range(10):
            observed = [tuple(event["seam"]) for event in events[30*sweep:30*(sweep+1)]]
            require(sorted(observed) == list(edges), name + ": sweep does not visit every seam exactly once")
        centered = norm(state)-initial_total**2/12
        expected_summary = {
            "attempts": 300, "sum_load_initial": str(initial_total),
            "sum_load_final": str(initial_total),
            "squared_norm_initial": str(initial_norm),
            "squared_norm_final": str(norm(state)),
            "squared_norm_final_decimal": float(norm(state)),
            "centered_squared_norm_final": str(centered),
            "centered_squared_norm_final_decimal": float(centered),
            "exact_sum_conservation": True, "exact_local_norm_identity": True,
            "seam_mismatch_increase_count": increases,
        }
        keys(case["summary"], expected_summary, name + ": summary")
        require(case["summary"] == expected_summary, name + ": summary differs from full independent replay")
        require(case["summary"]["exact_sum_conservation"] is True and case["summary"]["exact_local_norm_identity"] is True, name + ": verdict fields must be booleans")
        if name.endswith("control"):
            require(state == list(map(Fraction, initial)) and changed_steps == 0, name + ": fixed control moved")
        summaries.append({"name": name, "verified_events": 300, "changed_state_count": changed_steps,
                          "seam_mismatch_increase_count": increases,
                          "centered_squared_norm_final": str(centered)})
    return summaries


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("native_trace.json"))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("native_verification.json"))
    parser.add_argument("--sim-root", type=Path, help="Also check the named source hashes against this checkout")
    args = parser.parse_args()
    report = json.loads(args.input.read_text(), object_pairs_hook=reject_duplicate_keys,
                        parse_constant=lambda value: (_ for _ in ()).throw(VerificationError("nonfinite JSON constant: " + value)))
    summaries = verify_trace(report)
    driver = args.input.with_name("run_native.py")
    require(driver.is_file() and sha(driver) == report["provenance"]["driver_sha256"], "driver file/hash differs")
    sources = {}
    if args.sim_root:
        for name, digest in report["provenance"]["source_sha256"].items():
            path = args.sim_root/name
            require(path.is_file() and sha(path) == digest, name + ": source file/hash differs")
            sources[name] = digest
    mutations = []
    for label in ("edited_state", "invented_numeric_w", "edited_readback", "edited_seed", "removed_control", "false_summary"):
        candidate = copy.deepcopy(report)
        if label == "edited_state":
            candidate["cases"][0]["events"][0]["state_after"][0] = "999"
        elif label == "invented_numeric_w":
            candidate["thermodynamics"]["w"] = 1/3
        elif label == "edited_readback":
            candidate["cases"][0]["events"][0]["readback_before"][0] = "999"
        elif label == "edited_seed":
            candidate["cases"][0]["schedule_seed"] = 12345
        elif label == "removed_control":
            candidate["cases"].pop()
        else:
            candidate["cases"][0]["summary"]["seam_mismatch_increase_count"] = 0
        try:
            verify_trace(candidate)
        except VerificationError as exc:
            mutations.append({"mutation": label, "rejected": True, "reason": str(exc)})
        else:
            raise VerificationError("mutation falsely accepted: " + label)
    receipt = {
        "schema": "oph.ipi.native_scalar_independent_verification.v1",
        "verified": True,
        "input_sha256": sha(args.input),
        "driver_sha256": sha(driver),
        "verifier_sha256": sha(Path(__file__)),
        "implementation": "Python standard library only; exact Q(phi) carrier reconstruction and Fraction endpoint replacement; no simulator imports",
        "independent_checks": ["carrier incidence", "fixed cases/initial states/toggles/seeds", "complete ten shuffled sweeps",
                               "all 1200 readbacks and twelve-coordinate states", "all sums/norms/local norm increments/seam mismatches",
                               "all summaries and two unchanged controls", "thermodynamic and electromagnetic nonclaims", "driver hash"],
        "source_file_hashes_checked": bool(args.sim_root),
        "checked_source_sha256": sources,
        "source_history_note": "File hashes were checked when a source root was supplied; Git history and external publication were not independently authenticated.",
        "cases": summaries,
        "mutation_rejections": mutations,
        "scientific_boundary": "Verifies finite scalar averaging output only; establishes no physical energy, pressure, EoS, Maxwell law, or full self-reading federation.",
    }
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"verified": True, "events": 1200, "mutations_rejected": len(mutations), "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
