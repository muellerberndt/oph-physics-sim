#!/usr/bin/env python3
"""Extract source-local transition operators for the E1 adapter preflight.

Post-hoc extraction from the retained bundle ``runs/e1_prereg2_64k_20260810``
(the gates-passing rich-fibre run).  For the four committed rich observers of
``docs/E1_RICH_FIBRE_PAYLOAD_2.json`` it exports, verbatim and exactly:

* the 32-step ``transition_history_descriptor`` label path of each observer
  over the five packet fields, its realized state alphabet, its counted
  per-observer transition matrix, and the field-value table of every realized
  state (the diagonal field projectors);
* the step-aligned joint label path and counted joint transition table of
  every observer pair;
* the alignment receipt, a three-link conditional chain: the committed
  engine computes every observer's step labels from one shared global
  snapshot list (``history_source_states`` in ``oph_fpe/scale/bw_array.py``,
  built once for all rows), that list is capped at the pinned
  ``history_window`` of 32 entries (checked against ``config.yml``), and the
  engine may drop snapshots per observer, so shared-list membership alone
  does not align steps; every rich observer carrying exactly 32 steps
  (checked) closes the chain, because a 32-step descriptor over a list of at
  most 32 snapshots drops none, and then ``steps[t]`` of two observers refer
  to one global snapshot.  The related 31-entry ledger
  ``source_state.history_cycles`` of ``bw_state_derived_report.json`` is
  recorded verbatim as context without a one-to-one step indexing claim,
  because the bundle does not serialize per-snapshot cycles for the observer
  history list itself;
* the window receipts: the four truncated twenty-node windows of the
  committed payload are pairwise disjoint (a hard check); the full 96-node
  support lists are exported verbatim and the pairwise overlap sizes of the
  full supports are reported as data, because the greedy-disjoint scan
  pinned only the truncated windows.  Two pairs have fully disjoint
  supports, which downstream constructions may consume as a kernel-checked
  source receipt.

Falsifiable checks: every rich observer carries exactly 32 steps over exactly
the five declared fields with strict integer values; every occurrence of
``history_window`` in the pinned ``config.yml`` equals 32; the four
truncated windows are pairwise disjoint; ``history_cycles`` is nonempty and
strictly increasing; and the committed rich-fibre payload names this run.
Identities of the assembly, recorded as such and not as data checks: the
alphabet size is at most 32 because an alphabet of 32 steps cannot exceed
32 labels, every walk visits its whole realized alphabet because the
alphabet is defined from the walk, and marginals of the joint paths equal
the per-observer paths.  The genuine walk-surjectivity content lives in the
Lean packet as a two-literal cross-check between the transcribed alphabet
and path.

Claim boundary.  This is a post-hoc adapter-preflight extraction; neither the
statistic nor its rules were preregistered, and the payload is ineligible as
validation.  The step labels are support-modal packet values of the committed
run; no physical channel, instrument, causality, or prediction claim is made.
The payload is consumed by the E1 (#692) source-operator Lean packet as
literal input.

Usage (from the repository root, venv python):

    .venv/bin/python scripts/extract_e1_source_operators.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "runs" / "e1_prereg2_64k_20260810"
RICH_PAYLOAD_PATH = REPO_ROOT / "docs" / "E1_RICH_FIBRE_PAYLOAD_2.json"
OUT_PATH = REPO_ROOT / "docs" / "E1_SOURCE_OPERATOR_PAYLOAD.json"

PINNED_INPUTS = (
    "observer_views.jsonl",
    "bw_state_derived_report.json",
    "config.yml",
    "git_commit.txt",
)

DESCRIPTOR_FIELDS = (
    "record_family",
    "checkpoint_class",
    "stable_flag",
    "s3_sector_class",
    "repair_load_bucket",
)

STEP_COUNT = 32
MAX_ALPHABET = 32
MAX_PAYLOAD_BYTES = 200_000


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


def strict_int(value: object, context: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"non-integer value in {context}: {value!r}",
    )
    return value


def main() -> None:
    require(RUN_DIR.is_dir(), f"missing run directory {RUN_DIR}")
    for name in PINNED_INPUTS:
        require((RUN_DIR / name).is_file(), f"missing pinned input {name}")
    require(RICH_PAYLOAD_PATH.is_file(), "missing committed rich-fibre payload")

    rich = json.loads(RICH_PAYLOAD_PATH.read_text())
    require(
        rich.get("schema") == "oph.sim.e1_rich_fibre_payload.v2",
        "unexpected rich-fibre payload schema",
    )
    require(
        rich["provenance"]["run_id"] == RUN_DIR.name,
        "rich-fibre payload provenance names a different run",
    )
    rich_ids = [strict_int(x, "G2 ids") for x in rich["gates"]["G2_rich_observer_ids"]]
    require(len(rich_ids) == 4, "expected exactly four rich observers")
    windows = {}
    supports = {}
    for entry in rich["observers"]:
        oid = strict_int(entry["observer_id"], "payload observer id")
        if oid in rich_ids:
            windows[oid] = [strict_int(n, "window node") for n in entry["truncated_support_nodes"]]
    require(set(windows) == set(rich_ids), "payload lacks a rich observer entry")

    views = {}
    with (RUN_DIR / "observer_views.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            view = json.loads(line)
            oid = view.get("observer_id")
            if view.get("view_type") == "patch_observer" and oid in set(rich_ids):
                require(oid not in views, f"duplicate observer row {oid}")
                views[oid] = view
    require(set(views) == set(rich_ids), "run lacks a rich observer row")

    config_text = (RUN_DIR / "config.yml").read_text()
    window_values = [
        int(line.split(":", 1)[1].strip())
        for line in config_text.splitlines()
        if line.strip().startswith("history_window:")
    ]
    require(len(window_values) > 0, "config.yml pins no history_window")
    require(
        all(v == STEP_COUNT for v in window_values),
        f"history_window values {window_values} differ from {STEP_COUNT}",
    )

    derived = json.loads((RUN_DIR / "bw_state_derived_report.json").read_text())
    cycles = derived["source_state"]["history_cycles"]
    require(isinstance(cycles, list) and len(cycles) > 0, "history_cycles absent")
    cycles = [strict_int(c, "history cycle") for c in cycles]
    require(
        all(a < b for a, b in zip(cycles, cycles[1:])),
        "history cycles are not strictly increasing",
    )

    observers = []
    paths = {}
    alphabets = {}
    for oid in sorted(rich_ids):
        view = views[oid]
        supports[oid] = [strict_int(n, "support node") for n in view["support_nodes"]]
        descriptor = view["transition_history_descriptor"]
        fields = [str(f) for f in descriptor["fields"]]
        require(fields == list(DESCRIPTOR_FIELDS), f"observer {oid} field list differs")
        steps = descriptor["steps"]
        require(len(steps) == STEP_COUNT, f"observer {oid} step count differs")
        labels = []
        for pos, step in enumerate(steps):
            require(
                set(step) == set(DESCRIPTOR_FIELDS),
                f"observer {oid} step {pos} carries unexpected fields",
            )
            labels.append(
                tuple(strict_int(step[f], f"observer {oid} step {pos}") for f in fields)
            )
        alphabet = sorted(set(labels))
        require(len(alphabet) <= MAX_ALPHABET, f"observer {oid} alphabet too large")
        index = {label: i for i, label in enumerate(alphabet)}
        path = [index[label] for label in labels]
        transitions = {}
        for a, b in zip(path, path[1:]):
            transitions[(a, b)] = transitions.get((a, b), 0) + 1
        observers.append(
            {
                "observer_id": oid,
                "support_size": len(supports[oid]),
                "full_support_nodes": supports[oid],
                "truncated_window_nodes": windows[oid],
                "state_alphabet": [list(label) for label in alphabet],
                "state_count": len(alphabet),
                "step_path": path,
                "transition_counts": [
                    [a, b, n] for (a, b), n in sorted(transitions.items())
                ],
                "walk_visits_every_state": sorted(set(path)) == list(range(len(alphabet))),
            }
        )
        paths[oid] = path
        alphabets[oid] = alphabet

    for entry in observers:
        require(
            entry["walk_visits_every_state"],
            f"observer {entry['observer_id']} walk misses a realized state",
        )

    support_overlaps = []
    for i, a in enumerate(sorted(rich_ids)):
        for b in sorted(rich_ids)[i + 1 :]:
            require(
                not (set(windows[a]) & set(windows[b])),
                f"truncated windows of {a} and {b} intersect",
            )
            support_overlaps.append(
                [a, b, len(set(supports[a]) & set(supports[b]))]
            )

    pairs = []
    ordered = sorted(rich_ids)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            joint_path = [[paths[a][t], paths[b][t]] for t in range(STEP_COUNT)]
            joint_transitions = {}
            for (u, v), (u2, v2) in zip(joint_path, joint_path[1:]):
                key = (u, v, u2, v2)
                joint_transitions[key] = joint_transitions.get(key, 0) + 1
            both_change = sum(
                n
                for (u, v, u2, v2), n in joint_transitions.items()
                if u != u2 and v != v2
            )
            left_only = sum(
                n
                for (u, v, u2, v2), n in joint_transitions.items()
                if u != u2 and v == v2
            )
            right_only = sum(
                n
                for (u, v, u2, v2), n in joint_transitions.items()
                if u == u2 and v != v2
            )
            pairs.append(
                {
                    "left_observer": a,
                    "right_observer": b,
                    "left_state_count": len(alphabets[a]),
                    "right_state_count": len(alphabets[b]),
                    "joint_step_path": joint_path,
                    "joint_transition_counts": [
                        [u, v, u2, v2, n]
                        for (u, v, u2, v2), n in sorted(joint_transitions.items())
                    ],
                    "steps_left_only_change": left_only,
                    "steps_right_only_change": right_only,
                    "steps_both_change": both_change,
                    "steps_no_change": STEP_COUNT - 1 - left_only - right_only - both_change,
                }
            )

    payload = {
        "schema": "oph.sim.e1_source_operator_payload.v2",
        "provenance": {
            "run_id": RUN_DIR.name,
            "run_git_commit": (RUN_DIR / "git_commit.txt").read_text().strip(),
            "pinned_input_sha256": {
                name: sha256_file(RUN_DIR / name) for name in PINNED_INPUTS
            },
            "rich_fibre_payload_sha256": sha256_file(RICH_PAYLOAD_PATH),
            "descriptor_fields": list(DESCRIPTOR_FIELDS),
            "state_convention": (
                "a state is the exact five-field support-modal label tuple of "
                "one transition_history_descriptor step; the alphabet is the "
                "sorted set of realized tuples per observer"
            ),
            "alignment_receipt": (
                "conditional chain: the committed engine builds one shared "
                "history snapshot list (history_source_states, "
                "oph_fpe/scale/bw_array.py) capped at the pinned "
                "history_window of 32 (checked in config.yml), the engine "
                "may drop snapshots per observer, and every rich observer "
                "carries exactly 32 steps (checked), which forces zero "
                "drops, so steps[t] of two observers refer to one global "
                "snapshot; joint paths pair equal step indices; the "
                "shared-list and drop-branch facts are a declared engine "
                "reading of the pinned run commit"
            ),
            "history_cycles_note": (
                "source_state.history_cycles is a related 31-snapshot "
                "ledger of the derived report, recorded verbatim as "
                "context; the bundle does not serialize per-snapshot "
                "cycles for the observer history list, so no one-to-one "
                "step indexing is claimed"
            ),
            "identity_note": (
                "identities of the assembly, not data checks: joint-path "
                "marginals equal the per-observer paths, alphabet sizes "
                "stay at most the step count, and every walk visits its "
                "realized alphabet by definition of the alphabet"
            ),
            "epistemic_status": (
                "post-hoc adapter-preflight extraction from a retained "
                "bundle; not preregistered; ineligible as validation"
            ),
        },
        "derived_report_history_cycles": cycles,
        "step_count": STEP_COUNT,
        "observers": observers,
        "full_support_overlap_sizes": support_overlaps,
        "pairs": pairs,
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
    for entry in observers:
        print(
            f"observer {entry['observer_id']}: states {entry['state_count']}, "
            f"transitions {len(entry['transition_counts'])}"
        )
    for pair in pairs:
        print(
            f"pair ({pair['left_observer']},{pair['right_observer']}): "
            f"left-only {pair['steps_left_only_change']}, "
            f"right-only {pair['steps_right_only_change']}, "
            f"both {pair['steps_both_change']}, "
            f"none {pair['steps_no_change']}"
        )


if __name__ == "__main__":
    main()
