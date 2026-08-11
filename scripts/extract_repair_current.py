#!/usr/bin/env python3
"""Extract the repair-load irreversibility current from the preregistered run.

Post-hoc extraction from the retained bundle of the preregistered run
``runs/b12_prereg_16k_20260806`` in the house pattern: pinned inputs are
hashed, the ordered transition windows of ``observer_views.jsonl`` are
recounted with exact integer arithmetic under the report's state alphabet
(the same counting convention as ``extract_b12_mixing_chain.py``), and the
counts are projected onto the declared ``repair_load_bucket`` coordinate.

The payload records the exact 8-by-8 ordered count table, its antisymmetric
part, and two declared orientation invariants:

* the designated pair: the lexicographically least ordered bucket pair
  maximizing ``|C(a,b) - C(b,a)|``;
* the designated cycle: the lexicographically least 3-cycle maximizing the
  absolute difference of forward and backward count products.

Falsifiable cross-checks: observer, skip, and transition totals must equal
the pinned report; every step must lie inside the report alphabet; every
step field must be present with an exact integer value; and the recounted
26-state integer matrix must match the pinned npz weighted matrix on
positivity support and within tolerance on weighted recount, before any
projection.  Two further facts are identities of ordered recounting, not
data checks, and are labeled so: reversing every window transposes the
count table, and re-bucketing conserves totals.  Both invariants therefore
flip sign under time reversal of the counted order by construction.  A
count table with ``C = C^T`` would be reported as a negative result: the
designated pair and cycle would then be empty and the payload says so.  The
pair and cycle conditions are logically independent, so the payload reports
both applicability flags; the Lean orientation bit consumes the cycle flag.

Claim boundary.  These are ordered counts of one committed bounded run under
one declared quotient.  No physical arrow of time, thermodynamic-limit
current, or laboratory statement is made; the physical clock is owned by E5
(#703).  The payload is consumed by the B13 (#702) orientation-selection
Lean packet as literal input.

Usage (from the repository root, venv python):

    .venv/bin/python scripts/extract_repair_current.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "runs" / "b12_prereg_16k_20260806"
OUT_PATH = REPO_ROOT / "docs" / "REPAIR_CURRENT_PAYLOAD.json"

PINNED_INPUTS = (
    "finite_repair_transition_matrix_report.json",
    "finite_repair_transition_matrix.npz",
    "observer_views.jsonl",
    "conditional_resampling_realization_receipt.json",
    "git_commit.txt",
)

PROJECTION_FIELD = "repair_load_bucket"
MAX_PAYLOAD_BYTES = 100_000
WEIGHT_RECOUNT_ABS_TOLERANCE = 1e-6


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


def count_transitions(run_dir: Path):
    report = json.loads(
        (run_dir / "finite_repair_transition_matrix_report.json").read_text()
    )
    fields = [str(f) for f in report["packet_fields"]]
    require(PROJECTION_FIELD in fields, "projection field absent from packet fields")
    labels = [
        tuple((str(f), int(v)) for f, v in json.loads(s))
        for s in report["state_labels"]
    ]
    state_index = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    require(n == int(report["state_count"]), "state_count disagrees with labels")

    forward = [[0] * n for _ in range(n)]
    weighted = [[Fraction(0)] * n for _ in range(n)]
    observer_count = 0
    skipped = 0
    transition_count = 0
    with (run_dir / "observer_views.jsonl").open(encoding="utf-8") as handle:
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
            require(weight > 0, "nonpositive observer weight breaks the support check")
            encoded = []
            for step in steps:
                values = []
                for f in fields:
                    require(f in step, f"transition step misses the field {f}")
                    value = step[f]
                    require(
                        isinstance(value, int) and not isinstance(value, bool),
                        f"non-integer value for field {f}: {value!r}",
                    )
                    values.append((f, value))
                key = tuple(values)
                require(
                    key in state_index,
                    f"transition step outside the report alphabet: {key}",
                )
                encoded.append(state_index[key])
            for left, right in zip(encoded, encoded[1:]):
                forward[left][right] += 1
                weighted[left][right] += weight
                transition_count += 1

    require(
        observer_count == int(report["observer_count"]),
        "observer count disagrees with the report",
    )
    require(
        skipped == int(report["skipped_observer_count"]),
        "skipped observer count disagrees with the report",
    )
    require(
        transition_count == int(report["transition_count"]),
        "transition count disagrees with the report",
    )
    # matrix-level cross-check against the pinned weighted count matrix
    pinned = np.load(run_dir / "finite_repair_transition_matrix.npz", allow_pickle=True)
    pinned_counts = pinned["counts"]
    require(
        pinned_counts.shape == (n, n),
        "pinned count matrix shape disagrees with the alphabet",
    )
    weight_recount_dev = max(
        abs(float(weighted[i][j]) - float(pinned_counts[i][j]))
        for i in range(n)
        for j in range(n)
    )
    require(
        weight_recount_dev <= WEIGHT_RECOUNT_ABS_TOLERANCE,
        f"weighted recount deviates from the pinned npz by {weight_recount_dev}",
    )
    support_int = {(i, j) for i in range(n) for j in range(n) if forward[i][j] > 0}
    support_pinned = {(i, j) for i in range(n) for j in range(n) if pinned_counts[i][j] > 0}
    require(
        support_int == support_pinned,
        "integer counts and pinned weighted counts disagree on the positivity support",
    )
    return report, labels, forward, observer_count, transition_count


def main() -> None:
    require(RUN_DIR.is_dir(), f"missing preregistered run directory {RUN_DIR}")
    for name in PINNED_INPUTS:
        require((RUN_DIR / name).is_file(), f"missing pinned input {name}")

    report, labels, full_counts, observer_count, transition_count = (
        count_transitions(RUN_DIR)
    )
    n = len(labels)

    buckets = sorted({dict(label)[PROJECTION_FIELD] for label in labels})
    bucket_index = {b: k for k, b in enumerate(buckets)}
    m = len(buckets)
    counts = [[0] * m for _ in range(m)]
    for i in range(n):
        bi = bucket_index[dict(labels[i])[PROJECTION_FIELD]]
        for j in range(n):
            bj = bucket_index[dict(labels[j])[PROJECTION_FIELD]]
            counts[bi][bj] += full_counts[i][j]
    # identity of re-bucketing, kept as a guard against indexing bugs only
    require(
        sum(map(sum, counts)) == transition_count,
        "projected counts lose transitions",
    )

    current = [[counts[a][b] - counts[b][a] for b in range(m)] for a in range(m)]
    symmetric = all(current[a][b] == 0 for a in range(m) for b in range(m))

    designated_pair = None
    if not symmetric:
        best = None
        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                diff = counts[a][b] - counts[b][a]
                key = (-abs(diff), a, b)
                if best is None or key < best[0]:
                    best = (key, a, b, diff)
        _, a, b, diff = best
        designated_pair = {
            "bucket_from": buckets[a],
            "bucket_to": buckets[b],
            "count_forward": counts[a][b],
            "count_backward": counts[b][a],
            "difference": diff,
        }

    designated_cycle = None
    if not symmetric:
        best = None
        for a in range(m):
            for b in range(m):
                for c in range(m):
                    if len({a, b, c}) < 3:
                        continue
                    fwd = counts[a][b] * counts[b][c] * counts[c][a]
                    bwd = counts[b][a] * counts[c][b] * counts[a][c]
                    if fwd == bwd:
                        continue
                    key = (-abs(fwd - bwd), a, b, c)
                    if best is None or key < best[0]:
                        best = (key, a, b, c, fwd, bwd)
        if best is not None:
            _, a, b, c, fwd, bwd = best
            designated_cycle = {
                "buckets": [buckets[a], buckets[b], buckets[c]],
                "forward_product": fwd,
                "backward_product": bwd,
                "difference": fwd - bwd,
            }

    payload = {
        "schema": "oph.sim.repair_current_payload.v2",
        "provenance": {
            "run_id": RUN_DIR.name,
            "run_git_commit": (RUN_DIR / "git_commit.txt").read_text().strip(),
            "pinned_input_sha256": {
                name: sha256_file(RUN_DIR / name) for name in PINNED_INPUTS
            },
            "counting_convention": (
                "ordered pairs of consecutive transition_history_descriptor "
                "steps per observer, unweighted integer multiplicities, under "
                "the report state alphabet; identical to the mixing-chain "
                "extraction convention"
            ),
            "falsifiable_checks": (
                "observer, skip, and transition totals equal the pinned "
                "report; every step lies inside the report alphabet with "
                "exact integer field values; the recounted 26-state integer "
                "matrix matches the pinned npz weighted matrix on positivity "
                "support and within 1e-6 on weighted recount"
            ),
            "recount_identities": (
                "reversing every transition window transposes the count "
                "table, and re-bucketing conserves totals; both hold for any "
                "ordered recount by construction and are identities, not "
                "data checks"
            ),
            "projection_field": PROJECTION_FIELD,
            "designated_pair_rule": (
                "lexicographically least ordered bucket pair maximizing "
                "|C(a,b)-C(b,a)|"
            ),
            "designated_cycle_rule": (
                "lexicographically least ordered bucket 3-cycle maximizing "
                "|forward product - backward product|"
            ),
        },
        "observer_count": observer_count,
        "transition_count": transition_count,
        "buckets": buckets,
        "ordered_counts": counts,
        "current_antisymmetric_part": current,
        "pair_orientation_nonempty": not symmetric,
        "cycle_orientation_nonempty": designated_cycle is not None,
        "designated_pair": designated_pair,
        "designated_cycle": designated_cycle,
    }
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    require(
        len(text.encode()) <= MAX_PAYLOAD_BYTES,
        f"payload exceeds {MAX_PAYLOAD_BYTES} bytes",
    )
    OUT_PATH.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    print(f"wrote {OUT_PATH} ({len(text.encode())} bytes)")
    print(f"payload_sha256: {digest}")
    print(f"pair_orientation_nonempty: {not symmetric}")
    print(f"cycle_orientation_nonempty: {designated_cycle is not None}")
    if designated_pair:
        print(
            "designated pair: "
            f"{designated_pair['bucket_from']}->{designated_pair['bucket_to']} "
            f"forward {designated_pair['count_forward']} backward "
            f"{designated_pair['count_backward']}"
        )
    if designated_cycle:
        print(
            f"designated cycle: {designated_cycle['buckets']} forward "
            f"{designated_cycle['forward_product']} backward "
            f"{designated_cycle['backward_product']}"
        )


if __name__ == "__main__":
    main()
