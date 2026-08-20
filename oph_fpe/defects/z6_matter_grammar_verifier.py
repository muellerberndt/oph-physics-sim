"""Verifier stage: census labels against the committed matter grammar
(lane C6, exploratory, non-evidential).

Design record: ``oph_fpe/defects/DESIGN.md`` section 9. This is the only
census-lane module that carries committed matter-table values. It consumes
a finished census receipt and reports; it feeds nothing back into the
census pipeline, and no census parameter is tuned in response to it.

Committed reference surfaces (RER, read-only):

* ``Lean/Screen/GlobalFormCharacterDescent.lean``: label lattice
  W = Z/6 x Z/3 x Z/2 of (q = 6Y mod 6, triality t, duality d); descent
  congruence 2t + 3d + q = 0 mod 6 (``descent_integer_congruence``);
  exactly 6 of 36 labels descend, every center-character fibre has six
  labels (``descent_count``, ``centerChar_fibres_equal``); blocked control
  label (1, 0, 0) (``control_label_blocked``).
* ``Lean/Screen/ExteriorSelection.lean``: committed charge column and the
  two parity survivor masks (even {2,3,4,8,9}, odd {0,1,5,6,7}).
* ``Lean/Screen/ExteriorComponentBridge.lean``: committed bidegree column
  (``componentDegree``); row weights are (charge mod 6, color degree mod 3,
  weak degree mod 2) per ``QuantumMatterIntegration.componentWeight``.
"""

from __future__ import annotations

from fractions import Fraction

VERIFIER_SCHEMA = "oph.sim.defect_census.matter_grammar_verifier.v1"

# Committed charge column q = 6Y (ExteriorSelection.charge).
COMMITTED_CHARGE: tuple[int, ...] = (-2, 3, -4, 1, 6, 4, -1, -6, 2, -3)

# Committed bidegree column (color degree, weak degree)
# (ExteriorComponentBridge.componentDegree).
COMMITTED_BIDEGREE: tuple[tuple[int, int], ...] = (
    (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
    (1, 2), (2, 1), (3, 0), (2, 2), (3, 1),
)

# Committed parity survivor masks (ExteriorSelection even/odd sectors).
EVEN_SECTOR_ROWS: tuple[int, ...] = (2, 3, 4, 8, 9)
ODD_SECTOR_ROWS: tuple[int, ...] = (0, 1, 5, 6, 7)

# Committed blocked control label (control_label_blocked).
CONTROL_LABEL: tuple[int, int, int] = (1, 0, 0)

LATTICE_SIZE = 36
DESCENDING_COUNT = 6


def committed_row_weights() -> list[tuple[int, int, int]]:
    """Row weights (q, t, d) computed from the committed charge and
    bidegree columns, per ``componentWeight``: q = charge mod 6, t = color
    degree mod 3, d = weak degree mod 2."""
    return [
        (COMMITTED_CHARGE[i] % 6, COMMITTED_BIDEGREE[i][0] % 3,
         COMMITTED_BIDEGREE[i][1] % 2)
        for i in range(10)
    ]


def center_char(label: tuple[int, int, int]) -> int:
    """The committed congruence value ``(2t + 3d + q) mod 6``."""
    q, t, d = label
    return (2 * t + 3 * d + q) % 6


def descends(label: tuple[int, int, int]) -> bool:
    return center_char(label) == 0


def is_diagonal(label: tuple[int, int, int]) -> bool:
    """Whether the label lies on the CRT diagonal (t, d the canonical
    quotients of q): the image of the declared census readout."""
    q, t, d = label
    return t == q % 3 and d == q % 2


def verify_census(census: dict) -> dict:
    """Compare a finished census receipt against the committed matter
    grammar; report only (DESIGN.md section 9)."""
    classes = census["classes"]
    labels = [tuple(c["label_qtd"]) for c in classes]
    multiplicities = [int(c["multiplicity"]) for c in classes]
    total_classes = len(classes)
    total_multiplicity = sum(multiplicities)

    # (a) Descent congruence against the lattice baseline of one sixth.
    descending_classes = sum(1 for lab in labels if descends(lab))
    descending_multiplicity = sum(
        m for lab, m in zip(labels, multiplicities, strict=True)
        if descends(lab)
    )
    diagonal = all(is_diagonal(lab) for lab in labels)
    descent = {
        "classes_total": total_classes,
        "classes_descending": descending_classes,
        "fraction_by_class": (
            str(Fraction(descending_classes, total_classes))
            if total_classes else "0"
        ),
        "fraction_by_multiplicity": (
            str(Fraction(descending_multiplicity, total_multiplicity))
            if total_multiplicity else "0"
        ),
        "lattice_baseline": str(Fraction(DESCENDING_COUNT, LATTICE_SIZE)),
        "diagonal_readout_detected": diagonal,
        "structural_note": (
            "every CRT-diagonal label satisfies 2t + 3d + q = 6q = 0 mod 6;"
            " under the declared diagonal readout the descent fraction is"
            " structurally forced to 1 and is not a finding"
            if diagonal else
            "labels are not all diagonal; the descent fraction is live"
        ),
    }

    # (b) Committed row occupancy.
    weights = committed_row_weights()
    realized_labels = set(labels)
    label_class_count = {
        lab: sum(1 for x in labels if x == lab) for lab in realized_labels
    }
    label_multiplicity = {
        lab: sum(
            m for x, m in zip(labels, multiplicities, strict=True)
            if x == lab
        )
        for lab in realized_labels
    }
    rows = []
    for i in range(10):
        weight = weights[i]
        rows.append({
            "row": i,
            "bidegree": list(COMMITTED_BIDEGREE[i]),
            "charge_6Y": COMMITTED_CHARGE[i],
            "weight_qtd": list(weight),
            "realized": weight in realized_labels,
            "class_count": label_class_count.get(weight, 0),
            "multiplicity": label_multiplicity.get(weight, 0),
        })
    rows_realized = sum(1 for row in rows if row["realized"])
    sector_occupancy = {
        "even_sector_rows_realized": sum(
            1 for i in EVEN_SECTOR_ROWS if rows[i]["realized"]
        ),
        "odd_sector_rows_realized": sum(
            1 for i in ODD_SECTOR_ROWS if rows[i]["realized"]
        ),
        "rows_per_sector": 5,
    }

    # (c) Negative control.
    control = {
        "label": list(CONTROL_LABEL),
        "class_count": label_class_count.get(CONTROL_LABEL, 0),
        "multiplicity": label_multiplicity.get(CONTROL_LABEL, 0),
        "realized": CONTROL_LABEL in realized_labels,
        "structural_note": (
            "the control label is off-diagonal and unreachable under the"
            " declared readout; a nonzero count indicates a pipeline defect"
        ),
    }

    distinct_descending = sum(
        1 for lab in realized_labels if descends(lab)
    )
    return {
        "schema": VERIFIER_SCHEMA,
        "exploratory": True,
        "evidential": False,
        "frozen": False,
        "instrument_armed": False,
        "census_schema": census.get("schema"),
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
            ],
        },
        "descent": descent,
        "row_occupancy": {
            "rows": rows,
            "rows_realized": rows_realized,
        },
        "sector_occupancy": sector_occupancy,
        "control": control,
        "labels_realized": {
            "distinct": len(realized_labels),
            "of_lattice": LATTICE_SIZE,
            "distinct_descending": distinct_descending,
            "of_descending": DESCENDING_COUNT,
        },
        "no_tuning": (
            "report-only stage; no census parameter is read back or"
            " adjusted from this comparison"
        ),
        "statement": (
            "exploratory, non-evidential comparison of declared-convention"
            " census labels against the committed bookkeeping; no physical"
            " particle claim"
        ),
    }
