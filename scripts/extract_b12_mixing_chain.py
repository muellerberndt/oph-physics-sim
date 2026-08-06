#!/usr/bin/env python3
"""B12 receipt 3: extract the mixing recurrent chain from the preregistered run.

Reads the pinned transition-clock objects of runs/b12_prereg_16k_20260806
(finite_repair_transition_matrix.npz and its report), recomputes the
transition counts from observer_views.jsonl with exact integer arithmetic,
decomposes the raw chain into strongly connected components, restricts to the
unique closed recurrent class, and certifies with exact rational arithmetic:

* closure of the recurrent class (no counted transition leaves it);
* irreducibility (strong connectivity of the positive-entry digraph);
* aperiodicity (gcd of directed cycle lengths equals one);
* exact stationarity of the exact stationary law (linear solve over Fraction);
* nonconstant protected-record labelling across the recurrent states.

The payload is written to docs/B12_MIXING_CHAIN_PAYLOAD.json.

Count convention. The pinned npz weights every transition of an observer by
that observer's transition_history_mean_modal_mass, a float, so the pinned
count matrix is float valued. The exact realization of this script uses the
unweighted integer multiplicity counts of the same transition windows under
the same quotient. Both count matrices are verified to share one positivity
support, so they define the same digraph, the same component decomposition,
and the same recurrent class; the exact chain is the row normalization of the
integer counts. A Fraction-exact recount of the weighted matrix is checked
against the pinned npz as a cross check.

Claim boundary. Representation-level realization over earned literals. The
physical collar identification and the energy-clock calibration remain with
E5 (#703). A reducible raw chain without a usable closed class, or a constant
record labelling on the realized recurrent class, is reported as a negative
result with the exact obstruction and a nonzero exit code.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from fractions import Fraction
from math import gcd
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "runs" / "b12_prereg_16k_20260806"
OUT_PATH = REPO_ROOT / "docs" / "B12_MIXING_CHAIN_PAYLOAD.json"

PINNED_INPUTS = (
    "finite_repair_transition_matrix.npz",
    "finite_repair_transition_matrix_report.json",
    "observer_views.jsonl",
    "conditional_resampling_realization_receipt.json",
    "manifest.json",
    "seed_material.json",
    "git_commit.txt",
)

WEIGHT_RECOUNT_ABS_TOLERANCE = 1e-6
SPECTRUM_MATCH_ABS_TOLERANCE = 1e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(obstruction: str) -> None:
    print(f"NEGATIVE RESULT: {obstruction}", file=sys.stderr)
    raise SystemExit(1)


def require(condition: bool, obstruction: str) -> None:
    if not condition:
        fail(obstruction)


def strongly_connected_components(n: int, adjacency: list[list[bool]]) -> list[list[int]]:
    """Iterative Tarjan strongly-connected-component decomposition."""
    index_of: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    on_stack: set[int] = set()
    stack: list[int] = []
    components: list[list[int]] = []
    counter = 0
    for root in range(n):
        if root in index_of:
            continue
        work = [(root, 0)]
        while work:
            node, edge_ptr = work.pop()
            if edge_ptr == 0:
                index_of[node] = counter
                lowlink[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            advanced = False
            for target in range(edge_ptr, n):
                if not adjacency[node][target]:
                    continue
                if target not in index_of:
                    work.append((node, target + 1))
                    work.append((target, 0))
                    advanced = True
                    break
                if target in on_stack:
                    lowlink[node] = min(lowlink[node], index_of[target])
            if advanced:
                continue
            if lowlink[node] == index_of[node]:
                component: list[int] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component))
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
    return components


def digraph_period(nodes: list[int], adjacency: list[list[bool]]) -> int:
    """gcd of directed cycle lengths of a strongly connected digraph."""
    start = nodes[0]
    depth = {start: 0}
    frontier = [start]
    period = 0
    while frontier:
        next_frontier: list[int] = []
        for node in frontier:
            for target in nodes:
                if not adjacency[node][target]:
                    continue
                if target in depth:
                    period = gcd(period, depth[node] + 1 - depth[target])
                else:
                    depth[target] = depth[node] + 1
                    next_frontier.append(target)
        frontier = next_frontier
    return abs(period)


def exact_stationary_law(chain: list[list[Fraction]]) -> list[Fraction]:
    """Solve pi P = pi with sum(pi) = 1 by Gaussian elimination over Fraction."""
    n = len(chain)
    # Rows 0..n-1: (P^T - I) pi = 0; final row replaced by the mass constraint.
    rows = [
        [chain[j][i] - (Fraction(1) if i == j else Fraction(0)) for j in range(n)]
        for i in range(n)
    ]
    rhs = [Fraction(0)] * n
    rows[n - 1] = [Fraction(1)] * n
    rhs[n - 1] = Fraction(1)
    for col in range(n):
        pivot = next((r for r in range(col, n) if rows[r][col] != 0), None)
        if pivot is None:
            fail("stationary linear system is singular on the restricted class")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        scale = rows[col][col]
        rows[col] = [value / scale for value in rows[col]]
        rhs[col] = rhs[col] / scale
        for other in range(n):
            if other == col or rows[other][col] == 0:
                continue
            factor = rows[other][col]
            rows[other] = [a - factor * b for a, b in zip(rows[other], rows[col])]
            rhs[other] = rhs[other] - factor * rhs[col]
    return [rhs[i] for i in range(n)]


def main() -> None:
    require(RUN_DIR.is_dir(), f"missing preregistered run directory {RUN_DIR}")
    for name in PINNED_INPUTS:
        require((RUN_DIR / name).is_file(), f"missing pinned input {name}")

    report = json.loads((RUN_DIR / "finite_repair_transition_matrix_report.json").read_text())
    manifest = json.loads((RUN_DIR / "manifest.json").read_text())
    seed_material = json.loads((RUN_DIR / "seed_material.json").read_text())
    resampling = json.loads((RUN_DIR / "conditional_resampling_realization_receipt.json").read_text())
    git_commit = (RUN_DIR / "git_commit.txt").read_text().strip()

    fields = [str(f) for f in report["packet_fields"]]
    labels = [tuple((str(f), int(v)) for f, v in json.loads(s)) for s in report["state_labels"]]
    state_index = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    require(n == int(report["state_count"]), "state_count disagrees with state_labels")

    # Exact recount of the transition windows under the report's quotient.
    int_counts = [[0] * n for _ in range(n)]
    weighted_counts = [[Fraction(0)] * n for _ in range(n)]
    field_value_sets: dict[str, set[int]] = {"record_family": set(), "s3_sector_class": set()}
    observer_count = 0
    skipped = 0
    transition_count = 0
    with (RUN_DIR / "observer_views.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            observer_count += 1
            view = json.loads(line)
            steps = (view.get("transition_history_descriptor") or {}).get("steps") or []
            if len(steps) < 2:
                skipped += 1
                continue
            weight = Fraction(float(view.get("transition_history_mean_modal_mass", 1.0)))
            require(weight > 0, "nonpositive observer weight breaks the shared-support argument")
            encoded = []
            for step in steps:
                for name in field_value_sets:
                    field_value_sets[name].add(int(step.get(name, 0)))
                key = tuple((f, int(step.get(f, 0))) for f in fields)
                require(key in state_index, f"transition step visits a state absent from the report alphabet: {key}")
                encoded.append(state_index[key])
            for left, right in zip(encoded, encoded[1:]):
                int_counts[left][right] += 1
                weighted_counts[left][right] += weight
                transition_count += 1

    require(observer_count == int(report["observer_count"]), "observer count disagrees with the report")
    require(skipped == int(report["skipped_observer_count"]), "skipped observer count disagrees with the report")
    require(transition_count == int(report["transition_count"]), "transition count disagrees with the report")

    # Cross checks against the pinned weighted count matrix.
    pinned = np.load(RUN_DIR / "finite_repair_transition_matrix.npz", allow_pickle=True)
    pinned_counts = pinned["counts"]
    require(pinned_counts.shape == (n, n), "pinned count matrix shape disagrees with the alphabet")
    weight_recount_dev = max(
        abs(float(weighted_counts[i][j]) - float(pinned_counts[i][j]))
        for i in range(n)
        for j in range(n)
    )
    require(
        weight_recount_dev <= WEIGHT_RECOUNT_ABS_TOLERANCE,
        f"weighted recount deviates from the pinned npz by {weight_recount_dev}",
    )
    support_int = {(i, j) for i in range(n) for j in range(n) if int_counts[i][j] > 0}
    support_pinned = {(i, j) for i in range(n) for j in range(n) if pinned_counts[i][j] > 0}
    require(
        support_int == support_pinned,
        "integer counts and pinned weighted counts disagree on the positivity support",
    )

    # Component decomposition of the raw chain digraph.
    adjacency = [[int_counts[i][j] > 0 for j in range(n)] for i in range(n)]
    components = strongly_connected_components(n, adjacency)
    closed_classes = []
    for component in components:
        member = set(component)
        closed = all(not adjacency[i][j] for i in component for j in range(n) if j not in member)
        if closed:
            closed_classes.append(component)
    raw_irreducible = len(components) == 1
    require(not raw_irreducible or n == len(components[0]), "component bookkeeping is inconsistent")

    if raw_irreducible:
        recurrent = list(range(n))
        realization_branch = "irreducible_raw_chain"
    else:
        require(
            len(closed_classes) == 1,
            f"raw chain is reducible with {len(closed_classes)} closed classes; "
            "no unique restricted recurrent chain exists",
        )
        recurrent = closed_classes[0]
        realization_branch = "explicitly_restricted_unique_closed_class"
    require(
        len(recurrent) >= 2,
        "the unique closed class is a singleton, so the recurrent chain is trivial "
        "and cannot carry a nonconstant record",
    )

    # Protected-record labelling on the recurrent class, from the quotient definition.
    def field_value(state: int, name: str) -> int:
        return dict(labels[state])[name]

    record_label = {state: field_value(state, "checkpoint_class") for state in recurrent}
    require(
        len(set(record_label.values())) >= 2,
        "checkpoint_class record labelling is constant on the realized recurrent class",
    )
    constant_fields = {
        name: sorted(values)
        for name, values in field_value_sets.items()
        if len(values) == 1
    }

    # Exact restricted chain.
    m = len(recurrent)
    sub_counts = [[int_counts[i][j] for j in recurrent] for i in recurrent]
    for local, state in enumerate(recurrent):
        leak = sum(int_counts[state][j] for j in range(n)) - sum(sub_counts[local])
        require(leak == 0, f"state {state} emits {leak} counted transitions outside the closed class")
    row_sums = [sum(row) for row in sub_counts]
    require(all(s > 0 for s in row_sums), "a recurrent state has no counted outgoing transition")
    chain = [
        [Fraction(sub_counts[i][j], row_sums[i]) for j in range(m)]
        for i in range(m)
    ]
    require(all(sum(row) == 1 for row in chain), "restricted chain rows fail exact stochasticity")

    sub_adjacency = [[sub_counts[i][j] > 0 for j in range(m)] for i in range(m)]
    sub_components = strongly_connected_components(m, sub_adjacency)
    require(len(sub_components) == 1, "restricted chain digraph is not strongly connected")
    period = digraph_period(list(range(m)), sub_adjacency)
    require(period == 1, f"restricted chain has period {period}")
    positive_diagonal = all(sub_counts[i][i] > 0 for i in range(m))

    stationary = exact_stationary_law(chain)
    require(all(p > 0 for p in stationary), "exact stationary law has a nonpositive entry")
    require(sum(stationary) == 1, "exact stationary law fails normalization")
    for j in range(m):
        require(
            sum(stationary[i] * chain[i][j] for i in range(m)) == stationary[j],
            f"exact stationarity fails at state column {j}",
        )
    detailed_balance = all(
        stationary[i] * chain[i][j] == stationary[j] * chain[j][i]
        for i in range(m)
        for j in range(m)
    )

    # Spectral cross check for the two-state case: eigenvalues are 1 and trace-1.
    spectral: dict[str, object] = {}
    if m == 2:
        second = chain[0][0] + chain[1][1] - 1
        gap = 1 - second
        pinned_second = (
            float(pinned_counts[recurrent[0]][recurrent[0]]) / float(pinned_counts[recurrent[0]].sum())
            + float(pinned_counts[recurrent[1]][recurrent[1]]) / float(pinned_counts[recurrent[1]].sum())
            - 1.0
        )
        report_spectrum = [float(v) for v in report["matrices"]["raw_empirical"]["top_abs_eigenvalues"]]
        pinned_match = min(abs(v - abs(pinned_second)) for v in report_spectrum)
        require(
            pinned_match <= SPECTRUM_MATCH_ABS_TOLERANCE,
            "weighted restricted-class eigenvalue is absent from the pinned spectrum",
        )
        spectral = {
            "second_eigenvalue_exact": str(second),
            "second_eigenvalue_float": float(second),
            "spectral_gap_exact": str(gap),
            "spectral_gap_float": float(gap),
            "weighted_restricted_second_eigenvalue_float": pinned_second,
            "weighted_eigenvalue_matches_pinned_spectrum_within": pinned_match,
        }

    payload = {
        "schema": "oph.b12.mixing_chain_realization.v1",
        "issue": 688,
        "receipt": "collar-matrix realization on a mixing recurrent chain (receipt 3)",
        "claim_boundary": (
            "Representation-level realization over earned literals from one "
            "preregistered bounded source run. The raw 26-state quotient chain is "
            "reducible, so the realized object is the explicitly restricted chain "
            "on its unique closed recurrent class. The physical collar "
            "identification and the energy-clock calibration remain with E5 "
            "(#703). The state-side conditional-resampling kernel of the same run "
            "is an idempotent projection; the mixing content certified here lives "
            "entirely in the transition-clock object."
        ),
        "provenance": {
            "run_id": manifest["run_id"],
            "run_dir": str(RUN_DIR.relative_to(REPO_ROOT)),
            "seed": seed_material["seed"],
            "config_hash": seed_material["config_hash"],
            "git_commit": git_commit,
            "producer": "oph_fpe/cosmology/finite_repair_transition_clock.py",
            "extraction_script": "scripts/extract_b12_mixing_chain.py",
            "pinned_input_sha256": {name: sha256_file(RUN_DIR / name) for name in PINNED_INPUTS},
        },
        "source_object": {
            "report_mode": report["mode"],
            "packet_fields": fields,
            "state_count": n,
            "observer_count": observer_count,
            "skipped_observer_count": skipped,
            "transition_count": transition_count,
            "pinned_weight_field": report["weight_field"],
        },
        "count_convention": {
            "primary": (
                "unweighted integer multiplicity counts of the same transition "
                "windows under the report quotient; the exact chain is the row "
                "normalization of these integers"
            ),
            "pinned_npz_counts_are_float_weighted": True,
            "weighted_fraction_recount_max_abs_dev_vs_pinned_npz": weight_recount_dev,
            "positivity_support_shared_with_pinned_npz": True,
            "shared_support_consequence": (
                "both count matrices define one digraph, one component "
                "decomposition, and one recurrent class"
            ),
        },
        "full_chain": {
            "state_labels": [dict(label) for label in labels],
            "integer_counts_sparse": [
                [i, j, int_counts[i][j]]
                for i in range(n)
                for j in range(n)
                if int_counts[i][j] > 0
            ],
            "row_sums": [sum(int_counts[i]) for i in range(n)],
            "irreducible": raw_irreducible,
            "strongly_connected_components": components,
            "closed_classes": closed_classes,
            "reducibility_note": (
                "the raw chain fails irreducibility; receipt 3 is realized on the "
                "explicitly restricted branch permitted by issue 688"
            ),
        },
        "restricted_recurrent_chain": {
            "realization_branch": realization_branch,
            "global_state_indices": recurrent,
            "state_labels": [dict(labels[state]) for state in recurrent],
            "closure_verified_exact": True,
            "integer_counts": sub_counts,
            "row_sums": row_sums,
            "transition_matrix_exact": [[str(entry) for entry in row] for row in chain],
            "row_stochastic_exact": True,
            "irreducible": True,
            "aperiodic": True,
            "cycle_length_gcd": period,
            "positive_diagonal": positive_diagonal,
            "all_entries_positive": all(sub_counts[i][j] > 0 for i in range(m) for j in range(m)),
            "stationary_law_exact": [str(p) for p in stationary],
            "stationary_law_float": [float(p) for p in stationary],
            "stationarity_verified_exact": True,
            "stationary_law_positive": True,
            "detailed_balance_exact": detailed_balance,
            "detailed_balance_note": (
                "every stationary two-state chain satisfies detailed balance, so "
                "reversibility of the restricted chain carries no evidence weight"
            ),
            "spectral": spectral,
        },
        "protected_record": {
            "transition_side_label_field": "checkpoint_class",
            "labels_by_global_state": {str(state): record_label[state] for state in recurrent},
            "nonconstant": True,
            "label_semantics": (
                "checkpoint_class = 2 * committed_record_flag + stable_flag per "
                "oph_fpe/observers/objects.py; the recurrent states are the two "
                "committed record classes, stable and unstable"
            ),
            "committed_record_flag_by_global_state": {
                str(state): int(record_label[state] >= 2) for state in recurrent
            },
            "state_side_receipt": {
                "label": resampling["protected_record"]["label"],
                "class_count": resampling["protected_record"]["class_count"],
                "nonconstant": resampling["protected_record"]["nonconstant"],
                "unchanged_by_resampling": resampling["protected_record"]["unchanged_by_resampling"],
            },
            "constant_fields_across_all_counted_steps": constant_fields,
            "labelling_boundary": (
                "record_family and s3_sector_class are identically zero across "
                "all counted transition steps of this run, so the record_family "
                "projection of the state-side protected record is constant on the "
                "transition alphabet; the nonconstant protected labelling carried "
                "by the recurrent chain is the committed-record checkpoint class"
            ),
        },
        "shared_reference": {
            "statement": (
                "the state-side conditional-resampling receipt and this "
                "transition-side chain are computed from the same preregistered "
                "source run, pinned by the run id, seed, config hash, git commit, "
                "and the sha256 digests above"
            ),
            "state_side_artifact": "conditional_resampling_realization_receipt.json",
            "transition_side_artifact": "finite_repair_transition_matrix_report.json",
        },
        "verification": {
            "exact_arithmetic": "python fractions.Fraction throughout",
            "alphabet_matches_report": True,
            "observer_and_transition_totals_match_report": True,
            "support_matches_pinned_npz": True,
            "closure_exact": True,
            "irreducibility_exact": True,
            "aperiodicity_exact": True,
            "stationarity_exact": True,
            "record_label_nonconstant": True,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print(
        "restricted recurrent chain on global states "
        f"{recurrent} with labels {[record_label[s] for s in recurrent]}"
    )
    print("integer counts:", sub_counts)
    print("exact chain:", [[str(entry) for entry in row] for row in chain])
    print("exact stationary law:", [str(p) for p in stationary])


if __name__ == "__main__":
    main()
