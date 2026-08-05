"""Extract a small exact Lean witness payload from the B12 preregistered run.

The payload selects a handful of committed patch observers from
``observer_views.jsonl``, truncates each support-node list to its first
eight nodes, and labels every selected node with the integer record class
and companion class produced by the same binning that built the run's
conditional-resampling realization receipt
(``oph_fpe.dynamics.conditional_resampling.realization_inputs_from_freezeout``
on ``freezeout_fields.npz``).  It records the realized record-class counts
over the whole screen and over each selected support, and one exact joint
table of (record class, companion class) counts over the union of the
selected supports.  Every datum in the payload is an integer or a pinned
identifier; the provenance block pins the run id, seed, git commit,
receipt sha256, input sha256s, and extraction parameters.

The script fails hard when the recomputed class structure disagrees with
the receipt's pinned reference (fiber count, state count, total mass).
The payload is realized finite data for a later Lean witness to mirror as
literals; it promotes no physical claim.

Usage (from the repository root, venv python):

    .venv/bin/python scripts/extract_lean_witness_payload.py

Options allow a different run directory, observer count, truncation
length, or output path; defaults reproduce
``docs/SIM_EARNED_WITNESS_PAYLOAD.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oph_fpe.dynamics.conditional_resampling import (  # noqa: E402
    realization_inputs_from_freezeout,
)

PAYLOAD_SCHEMA = "oph.sim.lean_witness_payload.v1"
MAX_PAYLOAD_BYTES = 100_000

DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "b12_prereg_16k_20260806"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "SIM_EARNED_WITNESS_PAYLOAD.json"
RECEIPT_NAME = "conditional_resampling_realization_receipt.json"
FREEZEOUT_NAME = "freezeout_fields.npz"
VIEWS_NAME = "observer_views.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_observers(views_path: Path, count: int) -> list[dict]:
    """Return the first ``count`` patch-observer rows in file order.

    File order is pinned by the sha256 of ``observer_views.jsonl`` recorded
    in the provenance block, so the selection is a declared deterministic
    rule rather than a choice made after inspection.
    """

    selected: list[dict] = []
    with views_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("view_type") != "patch_observer":
                continue
            selected.append(row)
            if len(selected) == count:
                return selected
    raise ValueError(
        f"observer_views has only {len(selected)} patch observers; "
        f"{count} requested"
    )


def _class_counts(labels: list[int]) -> list[list[int]]:
    """Sorted [class, count] pairs; integers only."""

    counts = Counter(labels)
    return [[int(cls), int(counts[cls])] for cls in sorted(counts)]


def build_payload(
    run_dir: Path,
    *,
    observer_count: int,
    support_truncation: int,
) -> dict:
    receipt_path = run_dir / RECEIPT_NAME
    freezeout_path = run_dir / FREEZEOUT_NAME
    views_path = run_dir / VIEWS_NAME
    for path in (receipt_path, freezeout_path, views_path):
        if not path.exists():
            raise FileNotFoundError(path)

    receipt = json.loads(receipt_path.read_text())
    inputs = realization_inputs_from_freezeout(freezeout_path)
    record = inputs.record_classes
    companion = inputs.companion_classes

    joint = Counter(zip(record, companion))
    fiber_count = len(set(record))
    state_count = len(joint)
    total_mass = sum(joint.values())

    pinned = receipt["pinned_reference"]
    checks = {
        "fiber_count": (fiber_count, pinned["fiber_count"]),
        "state_count": (state_count, pinned["state_count"]),
        "total_mass_count": (total_mass, pinned["total_mass_count"]),
        "record_class_count": (
            fiber_count,
            receipt["protected_record"]["class_count"],
        ),
    }
    mismatches = {
        name: values for name, values in checks.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(
            f"recomputed class structure disagrees with receipt: {mismatches}"
        )

    seed = receipt["empirical_realization"]["seed"]
    git_commit_path = run_dir / "git_commit.txt"
    git_commit = (
        git_commit_path.read_text().strip() if git_commit_path.exists() else None
    )

    observers_raw = _select_observers(views_path, observer_count)
    observers = []
    union_nodes: set[int] = set()
    for row in observers_raw:
        support = [int(node) for node in row["support_nodes"]]
        union_nodes.update(support)
        first = support[:support_truncation]
        observers.append(
            {
                "observer_id": int(row["observer_id"]),
                "support_patch_count": int(row["support_patch_count"]),
                "support_nodes_first8": first,
                "support_nodes_first8_record_classes": [
                    int(record[node]) for node in first
                ],
                "support_nodes_first8_companion_classes": [
                    int(companion[node]) for node in first
                ],
                "support_record_class_counts": _class_counts(
                    [record[node] for node in support]
                ),
            }
        )

    union_sorted = sorted(union_nodes)
    union_joint = Counter(
        (record[node], companion[node]) for node in union_sorted
    )
    joint_cells = [
        [int(r), int(c), int(count)]
        for (r, c), count in sorted(union_joint.items())
    ]

    payload = {
        "schema": PAYLOAD_SCHEMA,
        "provenance": {
            "run_id": run_dir.name,
            "run_git_commit": git_commit,
            "seed": int(seed),
            "receipt_path": str(receipt_path.relative_to(REPO_ROOT)),
            "receipt_sha256": _sha256(receipt_path),
            "receipt_schema": receipt["schema"],
            "freezeout_path": str(freezeout_path.relative_to(REPO_ROOT)),
            "freezeout_sha256": _sha256(freezeout_path),
            "observer_views_path": str(views_path.relative_to(REPO_ROOT)),
            "observer_views_sha256": _sha256(views_path),
            "extraction": {
                "script": "scripts/extract_lean_witness_payload.py",
                "binning_producer": (
                    "oph_fpe.dynamics.conditional_resampling."
                    "realization_inputs_from_freezeout"
                ),
                "record_field": inputs.record_label,
                "companion_field": inputs.companion_label,
                "max_record_classes": 32,
                "max_companion_classes": 16,
                "observer_selection_rule": (
                    f"first {observer_count} patch_observer rows of "
                    "observer_views.jsonl in file order"
                ),
                "support_truncation": support_truncation,
            },
        },
        "class_structure": {
            "record_class_count": fiber_count,
            "realized_companion_class_count": len(set(companion)),
            "companion_class_labels_realized": sorted(set(companion)),
            "joint_state_count": state_count,
            "total_patch_count": total_mass,
            "matches_receipt_pinned_reference": True,
        },
        "realized_record_class_counts": _class_counts(list(record)),
        "observers": observers,
        "selected_support_joint_table": {
            "domain": (
                "union of the full supports of the selected observers, "
                "each patch counted once"
            ),
            "support_union_patch_count": len(union_sorted),
            "cell_format": ["record_class", "companion_class", "count"],
            "cells": joint_cells,
        },
        "claim_boundary": (
            "Realized finite data from one committed run under a declared "
            "binning and a declared observer selection rule. The integers "
            "are counts and class labels on this run's freezeout fields; "
            "no physical claim, no probability claim beyond realized "
            "frequencies, and no canonical status for the selection are "
            "asserted."
        ),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=DEFAULT_RUN_DIR,
        help="run directory holding the receipt, freezeout, and views",
    )
    parser.add_argument(
        "--observer-count", type=int, default=4,
        help="number of observers to select (3 to 5)",
    )
    parser.add_argument(
        "--support-truncation", type=int, default=8,
        help="nodes kept from the head of each support list",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="payload output path",
    )
    args = parser.parse_args()
    if not 3 <= args.observer_count <= 5:
        parser.error("--observer-count must be between 3 and 5")

    payload = build_payload(
        args.run_dir.resolve(),
        observer_count=args.observer_count,
        support_truncation=args.support_truncation,
    )
    text = json.dumps(payload, indent=1)
    encoded = text.encode()
    if len(encoded) > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"payload is {len(encoded)} bytes; limit {MAX_PAYLOAD_BYTES}"
        )
    args.output.write_text(text + "\n")
    print(f"wrote {args.output} ({len(encoded)} bytes)")


if __name__ == "__main__":
    main()
