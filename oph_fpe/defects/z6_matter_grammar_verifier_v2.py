"""Verifier stage v2: census labels with independent triality and duality
against the committed matter grammar (lane D2, exploratory, non-evidential).

Design record: ``oph_fpe/defects/DESIGN_V2.md`` sections 2d and 2e, fixed
before this module. The committed matter-table values are imported from the
quarantined v1 verifier module ``z6_matter_grammar_verifier``; the census
pipeline and the v2 readout are consumed read-only, and nothing feeds back.

The v2 readout image covers the full 36-element lattice (non-vacuity
receipt), so the descent congruence, the committed row occupancy against
the lattice complement, and the blocked control label are live checks with
the lattice baseline of one sixth.

Entry point: ``python -m oph_fpe.defects.z6_matter_grammar_verifier_v2``
reruns the exact C6 census streams (uniform_iid seed 20260820 N = 160,
sparse_pair seed 20260821 N = 80), attaches v2 labels, verifies, prints one
canonical receipt JSON to stdout and its SHA-256 to stderr; ``--out PATH``
writes a byte-identical copy. No writes under ``data/``.

Boundaries: exploratory, non-evidential, numbers only; a defect class is a
finite combinatorial invariant, not a particle; no instrument is frozen or
armed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from typing import Sequence

from oph_fpe.defects.z6_a5_action import rotation_group
from oph_fpe.defects.z6_carrier_defects import (
    CarrierSpec,
    base_carrier_spec,
)
from oph_fpe.defects.z6_defect_census import (
    DECLARED_STREAMS,
    census_json,
    run_census,
)
from oph_fpe.defects.z6_family_readout import (
    FAMILY_READOUT,
    FamilyWeights,
    build_readout,
    crt_diagonal,
    family_label,
    orbit_label_multiset,
)
from oph_fpe.defects.z6_matter_grammar_verifier import (
    COMMITTED_BIDEGREE,
    COMMITTED_CHARGE,
    CONTROL_LABEL,
    DESCENDING_COUNT,
    EVEN_SECTOR_ROWS,
    LATTICE_SIZE,
    ODD_SECTOR_ROWS,
    center_char,
    committed_row_weights,
    descends,
    is_diagonal,
)

VERIFIER_V2_SCHEMA = "oph.sim.defect_census.matter_grammar_verifier.v2"


def _fraction(numerator: int, denominator: int) -> str:
    return str(Fraction(numerator, denominator)) if denominator else "0"


def _all_labels() -> list[tuple[int, int, int]]:
    return [
        (q, t, d) for q in range(6) for t in range(3) for d in range(2)
    ]


# ---------------------------------------------------------------------------
# Census label attachment
# ---------------------------------------------------------------------------

def attach_v2_labels(census: dict, spec: CarrierSpec,
                     rotations: Sequence[Sequence[int]],
                     weights: FamilyWeights) -> dict:
    """Per-class v2 labels and per-orbit full-orbit label multisets on a
    finished census receipt; read-only consumption."""
    classes = []
    orbit_multisets: dict[tuple[int, ...], list[list[int]]] = {}
    for entry in census["classes"]:
        sector = tuple(int(x) for x in entry["sector"])
        orbit_rep = tuple(int(x) for x in entry["orbit_representative"])
        if orbit_rep not in orbit_multisets:
            orbit_multisets[orbit_rep] = orbit_label_multiset(
                spec, rotations, weights, orbit_rep
            )
        classes.append({
            "sector": list(sector),
            "label_qtd_v1": list(entry["label_qtd"]),
            "label_qtd_v2": list(family_label(weights, sector)),
            "energy": int(entry["energy"]),
            "multiplicity": int(entry["multiplicity"]),
            "depth2_stable": not bool(entry["neutral_escapable"]),
            "orbit_representative": list(orbit_rep),
            "orbit_size": int(entry["orbit_size"]),
        })
    classes.sort(key=lambda c: tuple(c["sector"]))
    orbits = []
    for orbit_rep in sorted(orbit_multisets):
        realized = [
            c for c in classes
            if tuple(c["orbit_representative"]) == orbit_rep
        ]
        orbits.append({
            "orbit_representative": list(orbit_rep),
            "orbit_size": len(orbit_multisets[orbit_rep]),
            "realized_classes": len(realized),
            "label_multiset_full": orbit_multisets[orbit_rep],
            "label_multiset_realized": sorted(
                c["label_qtd_v2"] for c in realized
            ),
        })
    return {
        "readout": FAMILY_READOUT,
        "classes": classes,
        "orbits": orbits,
    }


# ---------------------------------------------------------------------------
# Verifier tables (design record section 2e)
# ---------------------------------------------------------------------------

def verify_census_v2(labeled: dict) -> dict:
    classes = labeled["classes"]
    orbits = labeled["orbits"]
    labels = [tuple(c["label_qtd_v2"]) for c in classes]
    multiplicities = [c["multiplicity"] for c in classes]
    total_classes = len(classes)
    total_multiplicity = sum(multiplicities)
    realized = set(labels)
    diagonal = crt_diagonal()

    # (4) Descent congruence: live pass fraction plus vacuity detection.
    descending_classes = sum(1 for lab in labels if descends(lab))
    descending_mult = sum(
        m for lab, m in zip(labels, multiplicities, strict=True)
        if descends(lab)
    )
    all_diagonal = all(is_diagonal(lab) for lab in labels)
    descent = {
        "classes_total": total_classes,
        "classes_descending": descending_classes,
        "fraction_by_class": _fraction(descending_classes, total_classes),
        "fraction_by_multiplicity": _fraction(
            descending_mult, total_multiplicity
        ),
        "lattice_baseline": _fraction(DESCENDING_COUNT, LATTICE_SIZE),
        "diagonal_readout_detected": all_diagonal,
        "structural_note": (
            "realized labels all diagonal; under a diagonal readout the"
            " descent fraction is structurally forced and not a finding"
            if all_diagonal else
            "labels are not all diagonal; the descent fraction is live"
        ),
    }

    # (1) Full lattice occupancy over classes and over orbits.
    lattice = []
    for lab in _all_labels():
        class_count = sum(1 for x in labels if x == lab)
        multiplicity = sum(
            m for x, m in zip(labels, multiplicities, strict=True)
            if x == lab
        )
        orbit_full = sum(
            1 for o in orbits
            if list(lab) in o["label_multiset_full"]
        )
        orbit_realized = sum(
            1 for o in orbits
            if list(lab) in o["label_multiset_realized"]
        )
        lattice.append({
            "label_qtd": list(lab),
            "diagonal": lab in diagonal,
            "descends": descends(lab),
            "class_count": class_count,
            "multiplicity": multiplicity,
            "orbit_count_full_multiset": orbit_full,
            "orbit_count_realized": orbit_realized,
        })
    labels_realized = {
        "distinct": len(realized),
        "of_lattice": LATTICE_SIZE,
        "distinct_descending": sum(1 for lab in realized if descends(lab)),
        "of_descending": DESCENDING_COUNT,
        "distinct_off_diagonal": sum(
            1 for lab in realized if lab not in diagonal
        ),
    }

    # (2) Committed row occupancy and the lattice partition.
    weights_committed = committed_row_weights()
    rows = []
    for i in range(10):
        weight = weights_committed[i]
        rows.append({
            "row": i,
            "bidegree": list(COMMITTED_BIDEGREE[i]),
            "charge_6Y": COMMITTED_CHARGE[i],
            "weight_qtd": list(weight),
            "realized": weight in realized,
            "class_count": sum(1 for x in labels if x == weight),
            "multiplicity": sum(
                m for x, m in zip(labels, multiplicities, strict=True)
                if x == weight
            ),
        })
    committed_set = set(weights_committed)
    complement = [lab for lab in _all_labels() if lab not in committed_set]
    in_committed_classes = sum(
        1 for lab in labels if lab in committed_set
    )
    in_committed_mult = sum(
        m for lab, m in zip(labels, multiplicities, strict=True)
        if lab in committed_set
    )
    partition = {
        "committed_label_count": len(committed_set),
        "complement_label_count": len(complement),
        "row_collapse_note": (
            "the ten committed rows carry six distinct labels, all"
            " diagonal, forced by the charge binding q = -2c + 3w; the"
            " lane brief's 26-point complement presupposes ten distinct"
            " row labels, and the 30-point complement is used"
        ),
        "classes_in_committed": in_committed_classes,
        "classes_in_complement": total_classes - in_committed_classes,
        "fraction_classes_in_committed": _fraction(
            in_committed_classes, total_classes
        ),
        "multiplicity_in_committed": in_committed_mult,
        "multiplicity_in_complement": total_multiplicity - in_committed_mult,
        "fraction_multiplicity_in_committed": _fraction(
            in_committed_mult, total_multiplicity
        ),
        "baseline_uniform_lattice": _fraction(
            len(committed_set), LATTICE_SIZE
        ),
        "baseline_uniform_realized": _fraction(
            len(committed_set & realized), len(realized)
        ),
    }
    sector_occupancy = {
        "even_sector_rows_realized": sum(
            1 for i in EVEN_SECTOR_ROWS if rows[i]["realized"]
        ),
        "odd_sector_rows_realized": sum(
            1 for i in ODD_SECTOR_ROWS if rows[i]["realized"]
        ),
        "rows_per_sector": 5,
    }

    # (5) Control label: live under the v2 readout.
    control = {
        "label": list(CONTROL_LABEL),
        "class_count": sum(1 for lab in labels if lab == CONTROL_LABEL),
        "multiplicity": sum(
            m for lab, m in zip(labels, multiplicities, strict=True)
            if lab == CONTROL_LABEL
        ),
        "realized": CONTROL_LABEL in realized,
        "structural_note": (
            "the control label is reachable under the v2 readout"
            " (non-vacuity receipt); occupancy is a measurement, and the"
            " reporting path is receipted by a planted-class test"
        ),
    }

    # (6) Cross-tabulation label x energy x depth-2 stability.
    def crosstab(subset: list[dict]) -> list[dict]:
        cells: dict[tuple[tuple[int, ...], int, bool], list[int]] = {}
        for c in subset:
            key = (
                tuple(c["label_qtd_v2"]), c["energy"], c["depth2_stable"]
            )
            cell = cells.setdefault(key, [0, 0])
            cell[0] += 1
            cell[1] += c["multiplicity"]
        return [
            {
                "label_qtd": list(lab),
                "energy": energy,
                "depth2_stable": stable,
                "class_count": count,
                "multiplicity": mult,
            }
            for (lab, energy, stable), (count, mult) in sorted(
                cells.items(),
                key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]),
            )
        ]

    stable_classes = [c for c in classes if c["depth2_stable"]]
    cross = {
        "depth2_stable_classes": len(stable_classes),
        "all_classes": total_classes,
        "stable_cells": crosstab(stable_classes),
        "all_cells": crosstab(classes),
    }

    # (7) Multiplicity structure among depth-2 stable classes per label.
    shared: list[dict] = []
    stable_by_label: dict[tuple[int, ...], list[dict]] = {}
    for c in stable_classes:
        stable_by_label.setdefault(
            tuple(c["label_qtd_v2"]), []
        ).append(c)
    for lab in sorted(stable_by_label):
        members = stable_by_label[lab]
        shared.append({
            "label_qtd": list(lab),
            "stable_class_count": len(members),
            "energies": sorted(c["energy"] for c in members),
            "orbit_representatives_distinct": len({
                tuple(c["orbit_representative"]) for c in members
            }),
        })

    return {
        "schema": VERIFIER_V2_SCHEMA,
        "exploratory": True,
        "evidential": False,
        "frozen": False,
        "instrument_armed": False,
        "readout": labeled["readout"],
        "committed_reference": {
            "congruence": "2t + 3d + q = 0 mod 6",
            "lattice": [6, 3, 2],
            "lattice_size": LATTICE_SIZE,
            "descending_count": DESCENDING_COUNT,
            "control_label": list(CONTROL_LABEL),
            "sources": [
                "Lean/Screen/GlobalFormCharacterDescent.lean",
                "Lean/Screen/ExteriorSelection.lean",
                "Lean/Screen/ExteriorComponentBridge.lean",
                "Lean/Screen/QuantumMatterIntegration.lean",
            ],
        },
        "descent": descent,
        "lattice_occupancy": lattice,
        "labels_realized": labels_realized,
        "row_occupancy": {
            "rows": rows,
            "rows_realized": sum(1 for r in rows if r["realized"]),
        },
        "committed_vs_complement": partition,
        "sector_occupancy": sector_occupancy,
        "control": control,
        "cross_tabulation": cross,
        "stable_label_multiplicity": shared,
        "no_tuning": (
            "report-only stage; no census or readout parameter is read"
            " back or adjusted from this comparison"
        ),
        "statement": (
            "exploratory, non-evidential comparison of declared-convention"
            " v2 census labels against the committed bookkeeping; numbers"
            " only; no physical particle claim"
        ),
    }


# ---------------------------------------------------------------------------
# Pipeline and CLI
# ---------------------------------------------------------------------------

def run_v2(spec: CarrierSpec | None = None,
           streams: Sequence[tuple[str, int, int]] = DECLARED_STREAMS
           ) -> dict:
    """The full v2 receipt on the exact declared census streams: readout
    receipts, census pin, per-class labels, per-orbit multisets, and the
    verifier tables. Deterministic; fail-closed on any receipt."""
    spec = base_carrier_spec() if spec is None else spec
    rotations = rotation_group(spec)
    readout = build_readout(spec, rotations)
    census = run_census(spec, streams)
    census_bytes = census_json(census).encode("utf-8")
    labeled = attach_v2_labels(
        census, spec, rotations, readout["weights"]
    )
    verifier = verify_census_v2(labeled)
    return {
        "schema": "oph.sim.defect_census.family_readout_run.v2",
        "exploratory": True,
        "evidential": False,
        "frozen": False,
        "instrument_armed": False,
        "design_record": "oph_fpe/defects/DESIGN_V2.md",
        "streams": [
            {"name": name, "seed": seed, "size": size}
            for name, seed, size in streams
        ],
        "census_pin": {
            "schema": census["schema"],
            "sha256": hashlib.sha256(census_bytes).hexdigest(),
            "members_total": census["members_total"],
            "defect_class_count": census["defect_class_count"],
            "orbit_count": census["orbit_count"],
            "vacuum_multiplicity": census["vacuum"]["multiplicity"],
        },
        "readout_receipts": readout["receipts"],
        "class_labels": labeled["classes"],
        "orbit_labels": labeled["orbits"],
        "verifier": verifier,
        "statement": (
            "exploratory, non-evidential v2 label census on the committed"
            " carrier; the readout is the declared convention of"
            " DESIGN_V2.md section 2b; numbers only; no physical particle"
            " claim"
        ),
    }


def receipt_json(receipt: dict) -> str:
    return json.dumps(receipt, sort_keys=True, indent=1)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_path = None
    if argv:
        if len(argv) == 2 and argv[0] == "--out":
            out_path = argv[1]
        else:
            print(
                "usage: python -m"
                " oph_fpe.defects.z6_matter_grammar_verifier_v2"
                " [--out PATH]",
                file=sys.stderr,
            )
            return 2
    receipt = run_v2()
    text = receipt_json(receipt)
    data = text.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    sys.stdout.write(text + "\n")
    print(f"sha256 {digest}", file=sys.stderr)
    if out_path is not None:
        with open(out_path, "wb") as handle:
            handle.write(data)
        print(f"written {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - exploratory entry point
    raise SystemExit(main())
