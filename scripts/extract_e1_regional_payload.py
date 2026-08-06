"""Extract the E1 regional fibre payload from the B12 preregistered run.

For each of the four payload observers of
``docs/SIM_EARNED_WITNESS_PAYLOAD.json`` this script reports the realized
fibre structure of its support under the pinned conditional-resampling
binning: which record classes carry which companion classes, with exact
counts, on the truncated 8-node support and on the full 96-node support.
A record class whose truncated support carries at least two distinct
companion classes is a split fibre; the two realized companion states of
a split fibre span a two-by-two matrix block, which is the datum the Lean
module ``Lean/QFT/SourceRegionalNet.lean`` of
``reverse-engineering-reality`` consumes to enrich one regional algebra
of the sim-earned net beyond the diagonal.

The script recomputes record and companion classes with the same pinned
producer used by the realization receipt
(``oph_fpe.dynamics.conditional_resampling.realization_inputs_from_freezeout``
on ``freezeout_fields.npz``), fails hard when the recomputed class
structure disagrees with the receipt's pinned reference, and fails hard
when the truncated class labels disagree with the committed witness
payload literals that the Lean side mirrors.

The designated block is selected by a declared rule fixed before
extraction: the first split fibre in observer file order and truncated
support position order.  All split fibres are reported with exact counts;
the rule selects among them without discarding any.

Usage (from the repository root, venv python):

    .venv/bin/python scripts/extract_e1_regional_payload.py

Options allow a different run directory, truncation length, or output
path; defaults reproduce ``docs/E1_REGIONAL_PAYLOAD.json``.
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

PAYLOAD_SCHEMA = "oph.sim.e1_regional_payload.v1"
MAX_PAYLOAD_BYTES = 100_000

DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "b12_prereg_16k_20260806"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "E1_REGIONAL_PAYLOAD.json"
WITNESS_PAYLOAD = REPO_ROOT / "docs" / "SIM_EARNED_WITNESS_PAYLOAD.json"
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


def _truncated_fibre_rows(fibres: "OrderedDict[int, list]") -> list[dict]:
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

    witness = json.loads(WITNESS_PAYLOAD.read_text())
    witness_observers = {
        row["observer_id"]: row for row in witness["observers"]
    }

    seed = receipt["empirical_realization"]["seed"]
    git_commit_path = run_dir / "git_commit.txt"
    git_commit = (
        git_commit_path.read_text().strip() if git_commit_path.exists() else None
    )

    observers_raw = _select_observers(views_path, len(witness_observers))
    observers = []
    split_registry: list[dict] = []
    for row in observers_raw:
        observer_id = int(row["observer_id"])
        support = [int(node) for node in row["support_nodes"]]
        first = support[:support_truncation]

        mirror = witness_observers.get(observer_id)
        if mirror is None:
            raise ValueError(
                f"observer {observer_id} is absent from the witness payload"
            )
        recomputed_records = [int(record[node]) for node in first]
        recomputed_companions = [int(companion[node]) for node in first]
        if (
            first != mirror["support_nodes_first8"]
            or recomputed_records
            != mirror["support_nodes_first8_record_classes"]
            or recomputed_companions
            != mirror["support_nodes_first8_companion_classes"]
        ):
            raise ValueError(
                f"observer {observer_id} truncated classes disagree with "
                "the witness payload literals"
            )

        truncated_fibres = _fibres(first, record, companion)
        truncated_rows = _truncated_fibre_rows(truncated_fibres)
        split_rows = [
            r for r in truncated_rows
            if r["distinct_companion_class_count"] >= 2
        ]
        for r in split_rows:
            split_registry.append(
                {
                    "observer_id": observer_id,
                    "record_class": r["record_class"],
                    "first_support_position": r["entries"][0][
                        "support_position"
                    ],
                    "entries": r["entries"],
                }
            )

        full_fibres = _fibres(support, record, companion)
        full_distinct = {
            cls: len({entry[2] for entry in entries})
            for cls, entries in full_fibres.items()
        }
        observers.append(
            {
                "observer_id": observer_id,
                "support_patch_count": len(support),
                "truncated_support_nodes": first,
                "truncated_fibres": truncated_rows,
                "truncated_split_fibre_record_classes": [
                    r["record_class"] for r in split_rows
                ],
                "truncated_split_fibre_count": len(split_rows),
                "full_support_fibre_statistics": {
                    "record_class_count": len(full_fibres),
                    "multi_companion_record_class_count": sum(
                        1 for n in full_distinct.values() if n >= 2
                    ),
                    "max_distinct_companion_classes": max(
                        full_distinct.values()
                    ),
                    "distinct_companion_class_counts": [
                        [cls, full_distinct[cls]]
                        for cls in sorted(full_distinct)
                    ],
                },
            }
        )

    if not split_registry:
        raise ValueError(
            "no truncated support carries a split fibre; the enrichment "
            "payload cannot be built from this run"
        )
    designated = split_registry[0]

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
            "witness_payload_path": str(
                WITNESS_PAYLOAD.relative_to(REPO_ROOT)
            ),
            "witness_payload_sha256": _sha256(WITNESS_PAYLOAD),
            "extraction": {
                "script": "scripts/extract_e1_regional_payload.py",
                "binning_producer": (
                    "oph_fpe.dynamics.conditional_resampling."
                    "realization_inputs_from_freezeout"
                ),
                "record_field": inputs.record_label,
                "companion_field": inputs.companion_label,
                "observer_selection_rule": (
                    "the observers of the witness payload, in its file "
                    "order"
                ),
                "support_truncation": support_truncation,
                "split_fibre_rule": (
                    "a record class of a truncated support is a split "
                    "fibre when it carries at least two distinct "
                    "companion classes there"
                ),
                "designated_block_rule": (
                    "first split fibre in observer file order and "
                    "truncated support position order"
                ),
            },
        },
        "observers": observers,
        "split_fibres": split_registry,
        "designated_block": designated,
        "enlargement_statistic": {
            "definition": (
                "per observer, the number of record classes on the "
                "truncated support that carry at least two distinct "
                "companion classes; every such class supports one "
                "two-by-two matrix block in the regional algebra"
            ),
            "at_truncation": support_truncation,
            "values": [
                [obs["observer_id"], obs["truncated_split_fibre_count"]]
                for obs in observers
            ],
            "full_support_values": [
                [
                    obs["observer_id"],
                    obs["full_support_fibre_statistics"][
                        "multi_companion_record_class_count"
                    ],
                ]
                for obs in observers
            ],
            "requirement_for_fully_noncommutative_net": (
                "a payload in which every observer has "
                "truncated_split_fibre_count >= 1; the full-support "
                "values show what a deeper truncation of the same run "
                "would provide"
            ),
        },
        "claim_boundary": (
            "Realized finite data from one committed run under a "
            "declared binning and the witness payload's observer "
            "selection. The fibre tables are exact counts on this "
            "run's freezeout fields; the split-fibre and designation "
            "rules are declared extraction parameters with no "
            "canonical status; no physical claim and no probability "
            "claim beyond realized frequencies are asserted."
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
        "--support-truncation", type=int, default=8,
        help="nodes kept from the head of each support list",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="payload output path",
    )
    args = parser.parse_args()

    payload = build_payload(
        args.run_dir.resolve(),
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
