"""Extract the realized Born context web from the B12 preregistered run.

The E1 payload ``docs/E1_REGIONAL_PAYLOAD.json`` records two split
fibres on observer 64; each split fibre carries one realized two-by-two
matrix block in the regional algebra.  The run's final edge gauge state
``s3_gauge_state.npz`` records one S3 group element per edge.  S3 has an
exact two-dimensional irreducible representation with entries in the
field Q(sqrt 3); permutation-order-3 elements map to rotations by 120
degrees, and the conjugates of the diagonal record projector under those
rotations fail to commute with it.  This script joins the two data
sources into a context web on the designated split fibre:

1. the realized S3 elements on edges incident to the payload observers'
   support nodes, with exact permutation labels and per-element counts;
2. the exact image of every realized element under the two-dimensional
   irreducible representation, entries encoded as rational pairs
   ``[a, b]`` meaning ``a + b*sqrt(3)``;
3. for the designated split fibre of observer 64, the realized context
   web: the diagonal record/companion context with its outcome counts
   from the committed joint frequency table, one conjugated binary
   context per realized nonidentity gauge element with exact projector
   entries, exact commutator certificates against the record projector,
   and a machine-readable outcome-frequency boundary naming which
   contexts carry realized statistics and which carry none.

Every group-theoretic assertion is verified in exact fraction
arithmetic before the payload is written: the representation is checked
to be a homomorphism on all 36 ordered pairs against the composition
table of ``oph_fpe.finite_groups``, every image is checked orthogonal,
and the trace of every image is checked against the conjugacy class of
the permutation.  The class census of the gauge field is checked
against the run's committed ``s3_class_counts.json``, the recomputed
joint table structure is checked against the realization receipt's
pinned reference, and the designated-block literals are checked against
the E1 payload.  Any mismatch aborts the extraction.

Claim boundary: the conjugated contexts are effect data carried by the
realized gauge elements; the run supplies realized outcome statistics
for the diagonal context alone.  The rotated contexts carry no realized
frequencies, the payload records that absence in a machine-readable
field, and the missing statistics name a simulator capability gap.  The
gap is a missing producer, and it must never be filled by computing
Born predictions from the diagonal state.

Usage (from the repository root, venv python):

    .venv/bin/python scripts/extract_born_context_web.py

Options allow a different run directory or output path; defaults
reproduce ``docs/BORN_CONTEXT_WEB_PAYLOAD.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oph_fpe.dynamics.conditional_resampling import (  # noqa: E402
    realization_inputs_from_freezeout,
)
from oph_fpe.finite_groups import S3_CLASS, S3_ELEMENTS, S3_MUL  # noqa: E402

PAYLOAD_SCHEMA = "oph.sim.born_context_web_payload.v1"
MAX_PAYLOAD_BYTES = 100_000

DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "b12_prereg_16k_20260806"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "BORN_CONTEXT_WEB_PAYLOAD.json"
E1_PAYLOAD = REPO_ROOT / "docs" / "E1_REGIONAL_PAYLOAD.json"
RECEIPT_NAME = "conditional_resampling_realization_receipt.json"
FREEZEOUT_NAME = "freezeout_fields.npz"
GAUGE_NAME = "s3_gauge_state.npz"
CLASS_COUNTS_NAME = "s3_class_counts.json"

CLASS_LABELS = {0: "identity", 1: "transposition", 2: "threecycle"}
ELEMENT_ORDERS = {0: 1, 1: 2, 2: 3}

# ---------------------------------------------------------------------------
# Exact arithmetic in Q(sqrt 3): a scalar is a pair (a, b) of Fractions
# meaning a + b*sqrt(3); a matrix is a tuple of row tuples of scalars.
# ---------------------------------------------------------------------------


def _scalar(a, b=0) -> tuple[Fraction, Fraction]:
    return (Fraction(a), Fraction(b))


def _add(x, y):
    return (x[0] + y[0], x[1] + y[1])


def _sub(x, y):
    return (x[0] - y[0], x[1] - y[1])


def _mul(x, y):
    return (x[0] * y[0] + 3 * x[1] * y[1], x[0] * y[1] + x[1] * y[0])


def _mat_mul(m, n):
    return tuple(
        tuple(
            _add(_mul(m[i][0], n[0][j]), _mul(m[i][1], n[1][j]))
            for j in range(2)
        )
        for i in range(2)
    )


def _mat_sub(m, n):
    return tuple(
        tuple(_sub(m[i][j], n[i][j]) for j in range(2)) for i in range(2)
    )


def _mat_transpose(m):
    return tuple(tuple(m[j][i] for j in range(2)) for i in range(2))


def _mat_trace(m):
    return _add(m[0][0], m[1][1])


def _mat_is_zero(m) -> bool:
    return all(m[i][j] == _scalar(0) for i in range(2) for j in range(2))


def _encode_scalar(x) -> list[str]:
    return [str(x[0]), str(x[1])]


def _encode_matrix(m) -> list[list[list[str]]]:
    return [[_encode_scalar(m[i][j]) for j in range(2)] for i in range(2)]


IDENTITY = (
    (_scalar(1), _scalar(0)),
    (_scalar(0), _scalar(1)),
)

# The two-dimensional irreducible representation, fixed by the generator
# assignment rho((1,2,0)) = rotation by 120 degrees and
# rho((0,2,1)) = diag(1, -1), extended multiplicatively under the
# composition convention (g*h)[i] = g[h[i]] of oph_fpe.finite_groups.
HALF = Fraction(1, 2)
IRREP = {
    0: IDENTITY,
    1: ((_scalar(1), _scalar(0)), (_scalar(0), _scalar(-1))),
    2: (
        (_scalar(-HALF), _scalar(0, HALF)),
        (_scalar(0, HALF), _scalar(HALF)),
    ),
    3: (
        (_scalar(-HALF), _scalar(0, -HALF)),
        (_scalar(0, HALF), _scalar(-HALF)),
    ),
    4: (
        (_scalar(-HALF), _scalar(0, HALF)),
        (_scalar(0, -HALF), _scalar(-HALF)),
    ),
    5: (
        (_scalar(-HALF), _scalar(0, -HALF)),
        (_scalar(0, -HALF), _scalar(HALF)),
    ),
}

RECORD_PROJECTOR = (
    (_scalar(1), _scalar(0)),
    (_scalar(0), _scalar(0)),
)
COMPANION_PROJECTOR = (
    (_scalar(0), _scalar(0)),
    (_scalar(0), _scalar(1)),
)

# Character of the two-dimensional irreducible representation by class.
CLASS_CHARACTER = {0: _scalar(2), 1: _scalar(0), 2: _scalar(-1)}


def _verify_irrep() -> None:
    """Fail hard unless IRREP is the exact orthogonal irreducible
    representation matching the composition table of the simulator."""

    for g in range(6):
        m = IRREP[g]
        if _mat_mul(m, _mat_transpose(m)) != IDENTITY:
            raise ValueError(f"element {g}: image is not orthogonal")
        if _mat_mul(_mat_transpose(m), m) != IDENTITY:
            raise ValueError(f"element {g}: transpose is not a left inverse")
        expected = CLASS_CHARACTER[int(S3_CLASS[g])]
        if _mat_trace(m) != expected:
            raise ValueError(f"element {g}: trace disagrees with character")
    for g in range(6):
        for h in range(6):
            product = _mat_mul(IRREP[g], IRREP[h])
            if product != IRREP[int(S3_MUL[g][h])]:
                raise ValueError(
                    f"homomorphism failure at pair ({g}, {h})"
                )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _incident_counts(left, right, gauge, nodes) -> tuple[int, list[int]]:
    """Edge rows with at least one endpoint in ``nodes``: total and
    per-element counts.  Every npz row is one undirected edge with
    ``left < right`` and no duplicate pairs, verified by the caller."""

    node_list = sorted(set(int(n) for n in nodes))
    mask = np.isin(left, node_list) | np.isin(right, node_list)
    counts = Counter(int(v) for v in gauge[mask])
    return int(mask.sum()), [counts.get(g, 0) for g in range(6)]


def build_payload(run_dir: Path) -> dict:
    receipt_path = run_dir / RECEIPT_NAME
    freezeout_path = run_dir / FREEZEOUT_NAME
    gauge_path = run_dir / GAUGE_NAME
    class_counts_path = run_dir / CLASS_COUNTS_NAME
    for path in (receipt_path, freezeout_path, gauge_path,
                 class_counts_path, E1_PAYLOAD):
        if not path.exists():
            raise FileNotFoundError(path)

    _verify_irrep()

    receipt = json.loads(receipt_path.read_text())
    e1 = json.loads(E1_PAYLOAD.read_text())

    # --- gauge state -------------------------------------------------------
    bundle = np.load(gauge_path)
    left = np.asarray(bundle["left"], dtype=np.int64)
    right = np.asarray(bundle["right"], dtype=np.int64)
    gauge = np.asarray(bundle["gauge"], dtype=np.int64)
    if gauge.min() < 0 or gauge.max() > 5:
        raise ValueError("gauge indices outside S3 range")
    if not bool((left < right).all()):
        raise ValueError("edge rows are not in left < right form")
    pair_count = len(set(zip(left.tolist(), right.tolist())))
    if pair_count != len(left):
        raise ValueError("duplicate edge rows in the gauge state")

    committed_class_counts = json.loads(class_counts_path.read_text())
    census = Counter(int(S3_CLASS[int(v)]) for v in gauge)
    recomputed = {
        "identity": census.get(0, 0),
        "transposition": census.get(1, 0),
        "threecycle": census.get(2, 0),
    }
    if recomputed != committed_class_counts:
        raise ValueError(
            "gauge class census disagrees with s3_class_counts.json: "
            f"{recomputed} vs {committed_class_counts}"
        )

    # --- joint table under the pinned binning ------------------------------
    inputs = realization_inputs_from_freezeout(freezeout_path)
    record = inputs.record_classes
    companion = inputs.companion_classes
    joint = Counter(zip(record, companion))
    pinned = receipt["pinned_reference"]
    checks = {
        "fiber_count": (len(set(record)), pinned["fiber_count"]),
        "state_count": (len(joint), pinned["state_count"]),
        "total_mass_count": (sum(joint.values()), pinned["total_mass_count"]),
    }
    mismatches = {
        name: values for name, values in checks.items()
        if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(
            f"recomputed class structure disagrees with receipt: {mismatches}"
        )

    # --- designated block from the E1 payload ------------------------------
    designated = e1["designated_block"]
    record_class = int(designated["record_class"])
    entries = designated["entries"]
    if len(entries) != 2:
        raise ValueError("designated block does not have two entries")
    block_nodes = [int(entry["node"]) for entry in entries]
    block_companions = [int(entry["companion_class"]) for entry in entries]
    for node, cls in zip(block_nodes, block_companions):
        if int(record[node]) != record_class:
            raise ValueError(
                f"node {node}: recomputed record class disagrees with the "
                "E1 designated block"
            )
        if int(companion[node]) != cls:
            raise ValueError(
                f"node {node}: recomputed companion class disagrees with "
                "the E1 designated block"
            )
    designated_observer = int(designated["observer_id"])

    block_cells = [
        {
            "record_class": record_class,
            "companion_class": cls,
            "count": joint[(record_class, cls)],
        }
        for cls in block_companions
    ]
    fibre_mass = sum(
        count for (r, _), count in joint.items() if r == record_class
    )
    restricted_mass = sum(cell["count"] for cell in block_cells)
    total_mass = sum(joint.values())

    # --- incident-edge extraction ------------------------------------------
    per_observer = []
    for observer in e1["observers"]:
        support = [int(n) for n in observer["truncated_support_nodes"]]
        edge_count, element_counts = _incident_counts(
            left, right, gauge, support
        )
        per_observer.append(
            {
                "observer_id": int(observer["observer_id"]),
                "truncated_support_nodes": support,
                "incident_edge_count": edge_count,
                "element_counts": [
                    [g, element_counts[g]] for g in range(6)
                ],
            }
        )

    per_split_fibre = []
    for fibre in e1["split_fibres"]:
        nodes = [int(entry["node"]) for entry in fibre["entries"]]
        edge_count, element_counts = _incident_counts(
            left, right, gauge, nodes
        )
        per_split_fibre.append(
            {
                "observer_id": int(fibre["observer_id"]),
                "record_class": int(fibre["record_class"]),
                "nodes": nodes,
                "incident_edge_count": edge_count,
                "element_counts": [
                    [g, element_counts[g]] for g in range(6)
                ],
            }
        )

    designated_support_counts = next(
        row for row in per_observer
        if row["observer_id"] == designated_observer
    )
    designated_fibre_counts = next(
        row for row in per_split_fibre
        if row["observer_id"] == designated_observer
        and row["record_class"] == record_class
    )

    # --- irrep table --------------------------------------------------------
    elements = []
    for g in range(6):
        elements.append(
            {
                "index": g,
                "permutation": list(S3_ELEMENTS[g]),
                "class": CLASS_LABELS[int(S3_CLASS[g])],
                "order": ELEMENT_ORDERS[int(S3_CLASS[g])],
                "matrix": _encode_matrix(IRREP[g]),
            }
        )

    # --- conjugated contexts ------------------------------------------------
    conjugated_contexts = []
    coincidence: dict[tuple, list] = {}
    for g in range(1, 6):
        u = IRREP[g]
        ut = _mat_transpose(u)
        conj_record = _mat_mul(_mat_mul(u, RECORD_PROJECTOR), ut)
        conj_companion = _mat_mul(_mat_mul(u, COMPANION_PROJECTOR), ut)
        commutator = _mat_sub(
            _mat_mul(conj_record, RECORD_PROJECTOR),
            _mat_mul(RECORD_PROJECTOR, conj_record),
        )
        support_count = designated_support_counts["element_counts"][g][1]
        fibre_count = designated_fibre_counts["element_counts"][g][1]
        if support_count == 0:
            raise ValueError(
                f"element {g} is not realized on the designated observer "
                "support; the context is not earned by this run"
            )
        conjugated_contexts.append(
            {
                "context_id": f"conjugated_{g}",
                "gauge_element_index": g,
                "permutation": list(S3_ELEMENTS[g]),
                "class": CLASS_LABELS[int(S3_CLASS[g])],
                "projectors": [
                    _encode_matrix(conj_record),
                    _encode_matrix(conj_companion),
                ],
                "equals_diagonal_context": conj_record == RECORD_PROJECTOR,
                "commutator_with_record_projector": _encode_matrix(
                    commutator
                ),
                "noncommutation_certificate": not _mat_is_zero(commutator),
                "realized_incident_edge_count_on_observer_support": (
                    support_count
                ),
                "realized_incident_edge_count_on_fibre_nodes": fibre_count,
                "realized_outcome_counts": None,
            }
        )
        key = tuple(
            tuple(conj_record[i][j] for j in range(2)) for i in range(2)
        )
        coincidence.setdefault(key, []).append(f"conjugated_{g}")

    order_three = [
        row for row in conjugated_contexts if row["class"] == "threecycle"
    ]
    if not order_three:
        raise ValueError("no realized order-3 element; web is commutative")
    for row in order_three:
        if not row["noncommutation_certificate"]:
            raise ValueError(
                f"order-3 element {row['gauge_element_index']} has a "
                "vanishing commutator; exact computation is inconsistent"
            )

    diagonal_key = tuple(
        tuple(RECORD_PROJECTOR[i][j] for j in range(2)) for i in range(2)
    )
    coincidence.setdefault(diagonal_key, []).insert(0, "diagonal")
    coincidence_classes = sorted(
        coincidence.values(), key=lambda names: names[0]
    )

    payload = {
        "schema": PAYLOAD_SCHEMA,
        "provenance": {
            "run_id": run_dir.name,
            "run_git_commit": e1["provenance"]["run_git_commit"],
            "seed": int(receipt["empirical_realization"]["seed"]),
            "receipt_path": str(receipt_path.relative_to(REPO_ROOT)),
            "receipt_sha256": _sha256(receipt_path),
            "receipt_schema": receipt["schema"],
            "freezeout_path": str(freezeout_path.relative_to(REPO_ROOT)),
            "freezeout_sha256": _sha256(freezeout_path),
            "gauge_state_path": str(gauge_path.relative_to(REPO_ROOT)),
            "gauge_state_sha256": _sha256(gauge_path),
            "e1_payload_path": str(E1_PAYLOAD.relative_to(REPO_ROOT)),
            "e1_payload_sha256": _sha256(E1_PAYLOAD),
            "e1_payload_schema": e1["schema"],
            "extraction": {
                "script": "scripts/extract_born_context_web.py",
                "binning_producer": (
                    "oph_fpe.dynamics.conditional_resampling."
                    "realization_inputs_from_freezeout"
                ),
                "record_field": inputs.record_label,
                "companion_field": inputs.companion_label,
                "gauge_element_tables": "oph_fpe.finite_groups",
                "incidence_rule": (
                    "an npz edge row counts once when at least one "
                    "endpoint lies in the node set; rows are undirected "
                    "with left < right and no duplicates"
                ),
            },
        },
        "irrep": {
            "name": "standard_two_dimensional",
            "scalar_encoding": (
                "a scalar is a pair [a, b] of rationals in lowest terms "
                "meaning a + b*sqrt(3); a matrix is a row-major list of "
                "such pairs"
            ),
            "composition_convention": (
                "(g*h)[i] = g[h[i]], the S3_MUL table of "
                "oph_fpe.finite_groups"
            ),
            "generator_assignment": (
                "rho((1,2,0)) = rotation by 120 degrees, "
                "rho((0,2,1)) = diag(1, -1)"
            ),
            "verification": {
                "homomorphism_pairs_checked": 36,
                "orthogonality_checked": True,
                "class_character_checked": True,
                "arithmetic": "exact fractions in Q(sqrt 3)",
            },
            "elements": elements,
        },
        "edge_gauge_extraction": {
            "edge_row_count": int(len(left)),
            "gauge_class_census": recomputed,
            "class_census_matches_committed": True,
            "per_observer": per_observer,
            "per_split_fibre": per_split_fibre,
        },
        "context_web": {
            "observer_id": designated_observer,
            "record_class": record_class,
            "block_basis": [
                {
                    "block_index": index,
                    "node": node,
                    "companion_class": cls,
                }
                for index, (node, cls) in enumerate(
                    zip(block_nodes, block_companions)
                )
            ],
            "diagonal_context": {
                "context_id": "diagonal",
                "kind": "record_companion_projectors",
                "projectors": [
                    _encode_matrix(RECORD_PROJECTOR),
                    _encode_matrix(COMPANION_PROJECTOR),
                ],
                "realized_outcome_counts": [
                    cell["count"] for cell in block_cells
                ],
                "joint_table_cells": block_cells,
                "fibre_mass": fibre_mass,
                "block_restricted_mass": restricted_mass,
                "total_mass": total_mass,
                "block_conditional_frequencies": [
                    str(Fraction(cell["count"], restricted_mass))
                    for cell in block_cells
                ],
            },
            "conjugated_contexts": conjugated_contexts,
            "context_coincidence_classes": coincidence_classes,
        },
        "outcome_frequency_boundary": {
            "contexts_with_realized_frequencies": ["diagonal"],
            "contexts_without_realized_frequencies": [
                row["context_id"] for row in conjugated_contexts
            ],
            "requirement_for_additivity_receipt": (
                "realized per-context outcome counts with positive mass "
                "on every conjugated context, produced by a run, on top "
                "of the diagonal counts above"
            ),
            "named_capability_gap": (
                "the simulator has no producer that realizes a "
                "gauge-conjugated binary readout on the committed fibre "
                "states; rotated-context statistics require that producer"
            ),
            "prohibited_fill": (
                "computing Born predictions from the diagonal state and "
                "reporting them as rotated-context frequencies"
            ),
        },
        "claim_boundary": (
            "Realized finite data from one committed run: the gauge "
            "elements, their incidence counts, and the diagonal outcome "
            "counts are exact reads of committed artifacts, and the "
            "irrep images, conjugated projectors, and commutators are "
            "exact consequences of the element labels alone. The web "
            "earns noncommuting effect contexts from run data; no "
            "outcome statistics exist for the conjugated contexts, no "
            "noncontextual additivity receipt over the web is claimed, "
            "and no Born-rule closure is claimed."
        ),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=DEFAULT_RUN_DIR,
        help="run directory holding the gauge state, receipt, and freezeout",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="payload output path",
    )
    args = parser.parse_args()

    payload = build_payload(args.run_dir.resolve())
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
