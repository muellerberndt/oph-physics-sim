"""Extract the E1 rich-fibre payload from the preregistered 64k run.

This is the declared extraction step of
``docs/E1_PREREGISTERED_RICH_FIBRE_RUN_2026-08-10.md`` (issue #692 of
``reverse-engineering-reality``).  It generalizes the pinned
``extract_e1_regional_payload.py`` rule from the B12 run to the new
preregistered run: same record field (``record_signature``), same
companion field (``cumulative_repair_load``), same split-fibre rule (a
record class of a truncated support is a split fibre when it carries at
least two distinct companion classes there), same observer selection
(the first four ``patch_observer`` rows in ``observer_views.jsonl`` file
order), with the single declared change ``support_truncation = 12``.

The script recomputes record and companion classes from
``freezeout_fields.npz`` with the pinned producer
(``realization_inputs_from_freezeout``), fails hard when the recomputed
class structure disagrees with the run's own conditional-resampling
receipt, evaluates the preregistered gates G1 through G3 exactly, and
writes a deterministic JSON payload carrying the per-observer fibre
structure, the gate verdicts, and the provenance hashes.  Gate G4
(byte pinning and independent replay) is discharged by re-running this
script and comparing bytes; gate G5 (fail-closed) is a campaign rule,
not a computation.

The designated-block rule of the pinned extractor is generalized
per observer: for every observer with at least one split fibre, the
designated block is its first split fibre in truncated support position
order.  All split fibres are reported; the rule selects without
discarding.

Usage (from the repository root, venv python):

    .venv/bin/python scripts/extract_e1_rich_fibre_payload.py

Options allow a different run directory, truncation, or output path;
defaults reproduce ``docs/E1_RICH_FIBRE_PAYLOAD.json`` from
``runs/e1_prereg_64k_20260810``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oph_fpe.dynamics.conditional_resampling import (  # noqa: E402
    realization_inputs_from_freezeout,
)

PAYLOAD_SCHEMA = "oph.sim.e1_rich_fibre_payload.v1"
MAX_PAYLOAD_BYTES = 200_000

DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "e1_prereg_64k_20260810"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "E1_RICH_FIBRE_PAYLOAD.json"
RECEIPT_NAME = "conditional_resampling_realization_receipt.json"
FREEZEOUT_NAME = "freezeout_fields.npz"
VIEWS_NAME = "observer_views.jsonl"
OBSERVER_COUNT = 4
DEFAULT_TRUNCATION = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_observers(views_path: Path, count: int) -> list[dict]:
    """Return the first ``count`` patch-observer rows in file order."""

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


def _fibres(nodes: list[int], record, companion) -> "OrderedDict[int, list]":
    """Record class -> list of (position, node, companion) in position order."""

    fibres: OrderedDict[int, list] = OrderedDict()
    for position, node in enumerate(nodes):
        fibres.setdefault(int(record[node]), []).append(
            (position, int(node), int(companion[node]))
        )
    return fibres


def _fibre_rows(fibres: "OrderedDict[int, list]") -> list[dict]:
    rows = []
    for record_class, entries in fibres.items():
        companion_counts = Counter(entry[2] for entry in entries)
        rows.append(
            {
                "record_class": record_class,
                "entries": [
                    {
                        "support_position": position,
                        "node": node,
                        "companion_class": companion_class,
                    }
                    for position, node, companion_class in entries
                ],
                "companion_class_counts": [
                    [cls, companion_counts[cls]]
                    for cls in sorted(companion_counts)
                ],
                "distinct_companion_class_count": len(companion_counts),
            }
        )
    return rows


def build_payload(run_dir: Path, *, support_truncation: int) -> dict:
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

    observers_raw = _select_observers(views_path, OBSERVER_COUNT)
    observers = []
    truncated_windows: list[tuple[int, list[int]]] = []
    for row in observers_raw:
        observer_id = int(row["observer_id"])
        support = [int(node) for node in row["support_nodes"]]
        if len(support) < support_truncation:
            raise ValueError(
                f"observer {observer_id} support has only {len(support)} nodes; "
                f"truncation {support_truncation} requested"
            )
        window = support[:support_truncation]
        truncated_windows.append((observer_id, window))

        truncated_rows = _fibre_rows(_fibres(window, record, companion))
        split_rows = [
            r for r in truncated_rows
            if r["distinct_companion_class_count"] >= 2
        ]
        full_rows = _fibre_rows(_fibres(support, record, companion))
        full_split_count = sum(
            1 for r in full_rows if r["distinct_companion_class_count"] >= 2
        )
        designated = (
            split_rows[0]["record_class"] if split_rows else None
        )
        observers.append(
            {
                "observer_id": observer_id,
                "support_size": len(support),
                "truncated_support_nodes": window,
                "truncated_record_classes": [int(record[n]) for n in window],
                "truncated_companion_classes": [int(companion[n]) for n in window],
                "truncated_fibres": truncated_rows,
                "truncated_split_fibres": [
                    r["record_class"] for r in split_rows
                ],
                "truncated_split_fibre_count": len(split_rows),
                "designated_block_record_class": designated,
                "full_support_split_fibre_count": full_split_count,
            }
        )

    # Preregistered gates, evaluated exactly on the truncated windows.
    split_counts = [o["truncated_split_fibre_count"] for o in observers]
    gate_g1 = sum(1 for c in split_counts if c >= 1) >= 3
    rich_observers = [
        o["observer_id"] for o in observers
        if o["truncated_split_fibre_count"] >= 2
    ]
    gate_g2 = len(rich_observers) >= 2
    node_lists = [set(window) for _, window in truncated_windows]
    union_size = len(set().union(*node_lists))
    pairwise_disjoint = union_size == OBSERVER_COUNT * support_truncation
    gate_g3 = pairwise_disjoint

    payload = OrderedDict(
        {
            "schema": PAYLOAD_SCHEMA,
            "provenance": OrderedDict(
                {
                    "run_id": run_dir.name,
                    "seed": seed,
                    "run_git_commit": git_commit,
                    "receipt_sha256": _sha256(receipt_path),
                    "freezeout_sha256": _sha256(freezeout_path),
                    "observer_views_sha256": _sha256(views_path),
                    "extraction": OrderedDict(
                        {
                            "record_field": "record_signature",
                            "companion_field": "cumulative_repair_load",
                            "support_truncation": support_truncation,
                            "observer_rule": (
                                "first four patch_observer rows in "
                                "observer_views.jsonl file order"
                            ),
                            "split_fibre_rule": (
                                "a record class of a truncated support is a "
                                "split fibre when it carries at least two "
                                "distinct companion classes there"
                            ),
                            "designated_block_rule": (
                                "per observer, the first split fibre in "
                                "truncated support position order"
                            ),
                        }
                    ),
                }
            ),
            "class_structure": OrderedDict(
                {
                    "fiber_count": fiber_count,
                    "state_count": state_count,
                    "total_mass_count": total_mass,
                }
            ),
            "observers": observers,
            "gates": OrderedDict(
                {
                    "G1_three_of_four_observers_split": gate_g1,
                    "G2_two_observers_with_two_splits": gate_g2,
                    "G2_rich_observer_ids": rich_observers,
                    "G3_pairwise_disjoint_truncated_supports": gate_g3,
                    "G3_truncated_union_size": union_size,
                    "all_preregistered_gates_pass": bool(
                        gate_g1 and gate_g2 and gate_g3
                    ),
                }
            ),
        }
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--truncation", type=int, default=DEFAULT_TRUNCATION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_payload(args.run_dir, support_truncation=args.truncation)
    text = json.dumps(payload, indent=1) + "\n"
    if len(text.encode()) > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"payload would be {len(text.encode())} bytes; "
            f"cap is {MAX_PAYLOAD_BYTES}"
        )
    args.output.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    gates = payload["gates"]
    print(f"wrote {args.output} ({len(text.encode())} bytes)")
    print(f"payload_sha256: {digest}")
    for name in (
        "G1_three_of_four_observers_split",
        "G2_two_observers_with_two_splits",
        "G3_pairwise_disjoint_truncated_supports",
        "all_preregistered_gates_pass",
    ):
        print(f"{name}: {gates[name]}")
    for observer in payload["observers"]:
        print(
            f"observer {observer['observer_id']}: "
            f"{observer['truncated_split_fibre_count']} truncated split "
            f"fibres, {observer['full_support_split_fibre_count']} on the "
            "full support"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
