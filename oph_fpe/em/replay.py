"""Replay harness for the committed #733 discrete Coulomb Green receipt.

Design-only, non-evidential.  This replay does NOT arm an instrument; the
committed decision-rule template stays unarmed; nothing is frozen.  The
harness recomputes every declared observable on the simulator-side base
carrier and compares by exact rational equality against the committed
receipt values, reporting one of the template's three verdict labels:

* ``REPLICATED``: every declared observable matches exactly;
* ``FAILED``: at least one declared observable mismatches under a valid
  replay;
* ``INCONCLUSIVE``: replay could not be completed (pin drift, missing
  file, or tool failure).

Comparison sides.  The receipt side derives values only from numbers in the
committed receipt JSON (its seam table, its Green matrix times 180, its
loads, its committed energy strings).  The package side derives values only
from the ``oph_fpe.em`` tables and the package Green solve.  The two sides
meet only in exact ``Fraction`` equality checks.

Declared observable families (the committed handoff block, schema
``oph.discrete_coulomb_green.handoff.v2``, declares the first two; the
chord family carries the superseded v1 name
``chord_field_strength_component`` and is replayed against committed
receipt data: the committed spanning tree, Green matrix, and Thomson
cycle parts):

1. ``port_potential_difference``: all ordered port pairs for the three
   committed Thomson loads;
2. ``seam_flux``: all thirty seams for the three committed loads;
3. ``chord_field_strength_component``: the nineteen committed chords for
   the Coulomb field (committed value zero: a pure gradient) and for the
   committed tree Gauss solution (committed value: the Thomson cycle
   part) of the three loads.

Boundaries.  Finite exact mathematics on the committed base carrier; no
physical charge, field, space, or readout attachment (PR-53, PR-54 open);
no frozen instrument; no physical claim.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

from oph_fpe.em import base_carrier, green
from oph_fpe.em.base_carrier import (
    CHORD_SEAMS,
    PORTS,
    SEAM_LEFT,
    SEAM_RIGHT,
    SEAMS,
    TREE_SEAMS,
    Matrix,
    Vector,
    ZERO,
    ONE,
)

RECEIPT_SCHEMA = "oph.discrete_coulomb_green.v1"
RECEIPT_STATUS = (
    "EXACT_DISCRETE_COULOMB_GREEN_THOMSON_AND_REPAIR_RECEIPT__"
    "PHYSICAL_ATTACHMENTS_PR53_PR54_OPEN"
)
HANDOFF_SCHEMA = "oph.discrete_coulomb_green.handoff.v2"
VERDICT_LABELS = ("REPLICATED", "FAILED", "INCONCLUSIVE")

REPLAY_SCHEMA = "oph.sim.em_handoff_replay.v1"

_DEFAULT_RECEIPT = (
    Path(__file__).resolve().parents[3]
    / "reverse-engineering-reality"
    / "code"
    / "electromagnetism"
    / "runtime"
    / "discrete_coulomb_green_receipt.json"
)


def default_receipt_path() -> Path:
    return _DEFAULT_RECEIPT


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


class _Inconclusive(Exception):
    """Replay could not be completed: pin drift, missing file, or tool
    failure."""


def _load_committed_receipt(path: Path) -> dict:
    if not path.is_file():
        raise _Inconclusive(f"missing committed receipt: {path}")
    raw = path.read_bytes()
    try:
        committed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise _Inconclusive(f"invalid committed receipt JSON: {error}")
    if not isinstance(committed, dict):
        raise _Inconclusive("committed receipt root is not an object")
    if raw != _canonical_json_bytes(committed):
        raise _Inconclusive("committed receipt is not canonical (pin drift)")
    if committed.get("schema") != RECEIPT_SCHEMA:
        raise _Inconclusive("committed receipt schema drift")
    if committed.get("status") != RECEIPT_STATUS:
        raise _Inconclusive("committed receipt status drift")
    body = {k: v for k, v in committed.items() if k != "receipt_sha256"}
    digest = "sha256:" + hashlib.sha256(
        _canonical_json_bytes(body)
    ).hexdigest()
    if committed.get("receipt_sha256") != digest:
        raise _Inconclusive("committed receipt self-digest drift")
    handoff = committed.get("handoff_interface")
    if not isinstance(handoff, dict):
        raise _Inconclusive("committed handoff block missing")
    if handoff.get("schema") != HANDOFF_SCHEMA:
        raise _Inconclusive("committed handoff schema drift")
    template = handoff.get("decision_rule_template")
    if not isinstance(template, dict):
        raise _Inconclusive("committed decision-rule template missing")
    if tuple(template.get("verdict_labels", ())) != VERDICT_LABELS:
        raise _Inconclusive("committed verdict labels drift")
    if template.get("template_only") is not True:
        raise _Inconclusive("committed template arming drift")
    if handoff.get("design_only") is not True or handoff.get("frozen"):
        raise _Inconclusive("committed handoff design-only flags drift")
    return committed


def _fraction(text: Any) -> Fraction:
    return Fraction(str(text))


def _receipt_load(entry: dict) -> Vector:
    load = [ZERO] * PORTS
    for key, value in entry.items():
        load[int(key)] = _fraction(value)
    return load


def _tree_solution_from_tables(
    left: Sequence[int],
    right: Sequence[int],
    tree_seams: Sequence[int],
    load: Vector,
) -> Vector:
    """Receipt-side tree Gauss solution using only receipt tables."""
    adjacency: dict[int, list[tuple[int, int]]] = {p: [] for p in range(PORTS)}
    for e in tree_seams:
        adjacency[left[e]].append((right[e], e))
        adjacency[right[e]].append((left[e], e))
    parent: dict[int, tuple[int, int] | None] = {0: None}
    order = [0]
    queue = [0]
    while queue:
        node = queue.pop(0)
        for neighbour, seam in adjacency[node]:
            if neighbour not in parent:
                parent[neighbour] = (node, seam)
                order.append(neighbour)
                queue.append(neighbour)
    if len(order) != PORTS:
        raise _Inconclusive("committed tree does not span the ports")
    subtree = {p: Fraction(load[p]) for p in range(PORTS)}
    for node in reversed(order):
        entry = parent[node]
        if entry is not None:
            subtree[entry[0]] += subtree[node]
    current = [ZERO] * len(left)
    for node in order[1:]:
        entry = parent[node]
        assert entry is not None
        _, seam = entry
        if right[seam] == node:
            current[seam] = subtree[node]
        else:
            current[seam] = -subtree[node]
    return current


def _bfs_distances() -> list[list[int]]:
    adjacency: list[set[int]] = [set() for _ in range(PORTS)]
    for e in range(SEAMS):
        adjacency[SEAM_LEFT[e]].add(SEAM_RIGHT[e])
        adjacency[SEAM_RIGHT[e]].add(SEAM_LEFT[e])
    table = []
    for source in range(PORTS):
        dist = {source: 0}
        frontier = [source]
        while frontier:
            nxt = []
            for u in frontier:
                for w in adjacency[u]:
                    if w not in dist:
                        dist[w] = dist[u] + 1
                        nxt.append(w)
            frontier = nxt
        table.append([dist[p] for p in range(PORTS)])
    return table


def _pair_average_matrix() -> Matrix:
    average = [[ZERO] * PORTS for _ in range(PORTS)]
    for e in range(SEAMS):
        pair = (SEAM_LEFT[e], SEAM_RIGHT[e])
        for i in range(PORTS):
            for j in range(PORTS):
                if i in pair:
                    entry = Fraction(1, 2) if j in pair else ZERO
                else:
                    entry = ONE if i == j else ZERO
                average[i][j] += Fraction(1, SEAMS) * entry
    return average


class _Comparator:
    def __init__(self) -> None:
        self.observable_counts: dict[str, dict[str, int]] = {}
        self.supporting_counts: dict[str, int] = {}
        self.mismatches: list[str] = []

    def observable(self, family: str, label: str, got: Any, want: Any) -> None:
        counts = self.observable_counts.setdefault(
            family, {"comparisons": 0, "mismatches": 0}
        )
        counts["comparisons"] += 1
        if got != want:
            counts["mismatches"] += 1
            self.mismatches.append(
                f"declared:{family}:{label}: got {got}, committed {want}"
            )

    def supporting(self, group: str, label: str, got: Any, want: Any) -> None:
        self.supporting_counts[group] = self.supporting_counts.get(group, 0) + 1
        if got != want:
            self.mismatches.append(
                f"supporting:{group}:{label}: got {got}, committed {want}"
            )


def _package_green(green_override: Matrix | None) -> Matrix:
    if green_override is None:
        return green.green_matrix()
    return [list(row) for row in green_override]


def _package_potential(g: Matrix, load: Vector) -> Vector:
    return [
        sum((g[p][q] * load[q] for q in range(PORTS)), start=ZERO)
        for p in range(PORTS)
    ]


def replay_committed_receipt(
    receipt_path: Path | str | None = None,
    green_override: Matrix | None = None,
) -> dict:
    """Recompute every declared observable and compare against the committed
    receipt by exact rational equality.  Returns the replay report with the
    template verdict.  ``green_override`` replaces the package-side Green
    matrix; it exists for mutation-guard tests only."""
    path = Path(receipt_path) if receipt_path is not None else _DEFAULT_RECEIPT
    report: dict[str, Any] = {
        "schema": REPLAY_SCHEMA,
        "design_only": True,
        "frozen": False,
        "instrument_armed": False,
        "template_armed": False,
        "committed_receipt_path": str(path),
        "statement": (
            "design-only replay of the committed #733 handoff observables "
            "on the simulator base carrier; the committed decision-rule "
            "template stays unarmed; no freeze, no physical claim, no "
            "public measurement read"
        ),
    }
    try:
        committed = _load_committed_receipt(path)
        result = _run_comparisons(committed, green_override)
    except _Inconclusive as reason:
        report["verdict"] = "INCONCLUSIVE"
        report["reason"] = str(reason)
        return report
    except Exception as error:  # tool failure is INCONCLUSIVE, not FAILED
        report["verdict"] = "INCONCLUSIVE"
        report["reason"] = f"tool failure: {type(error).__name__}: {error}"
        return report
    report.update(result)
    return report


def _run_comparisons(
    committed: dict, green_override: Matrix | None
) -> dict[str, Any]:
    for key in (
        "seam_table",
        "degree_sequence",
        "laplacian_matrix",
        "green_matrix_times_180",
        "distance_receipt",
        "dipole_receipt",
        "thomson_spot_checks",
        "spanning_tree",
        "green_identities",
        "repair_bridge",
    ):
        if key not in committed:
            raise _Inconclusive(f"committed receipt key missing: {key}")

    comp = _Comparator()

    # Receipt-side data (numbers from the committed receipt only).
    rec_left = [int(x) for x in committed["seam_table"]["seam_left"]]
    rec_right = [int(x) for x in committed["seam_table"]["seam_right"]]
    if len(rec_left) != SEAMS or len(rec_right) != SEAMS:
        raise _Inconclusive("committed seam table arity drift")
    rec_green: Matrix = [
        [Fraction(int(v), 180) for v in row]
        for row in committed["green_matrix_times_180"]
    ]
    if len(rec_green) != PORTS or any(len(r) != PORTS for r in rec_green):
        raise _Inconclusive("committed green matrix arity drift")
    rec_tree_seams = [int(x) for x in committed["spanning_tree"]["tree_seams"]]
    rec_chords = [int(x) for x in committed["spanning_tree"]["chords"]]
    rec_loads = [
        _receipt_load(entry["load"])
        for entry in committed["thomson_spot_checks"]
    ]
    if len(rec_loads) != 3:
        raise _Inconclusive("committed spot-check count drift")

    # Package side.
    g_pkg = _package_green(green_override)
    lap_pkg = base_carrier.laplacian_matrix()
    base_carrier.base_carrier_receipt()

    # --- Supporting equalities: carrier pins -------------------------------
    for e in range(SEAMS):
        comp.supporting("carrier", f"seam_left[{e}]", SEAM_LEFT[e], rec_left[e])
        comp.supporting(
            "carrier", f"seam_right[{e}]", SEAM_RIGHT[e], rec_right[e]
        )
    for p, want in enumerate(committed["degree_sequence"]):
        comp.supporting(
            "carrier", f"degree[{p}]", int(lap_pkg[p][p]), int(want)
        )
    for i in range(PORTS):
        for j in range(PORTS):
            comp.supporting(
                "laplacian",
                f"L[{i}][{j}]",
                lap_pkg[i][j],
                Fraction(int(committed["laplacian_matrix"][i][j])),
            )
    for e, want in enumerate(rec_tree_seams):
        comp.supporting("spanning_tree", f"tree_seams[{e}]", TREE_SEAMS[e], want)
    for e, want in enumerate(rec_chords):
        comp.supporting("spanning_tree", f"chords[{e}]", CHORD_SEAMS[e], want)
    comp.supporting(
        "spanning_tree",
        "chord_count",
        len(CHORD_SEAMS),
        int(committed["spanning_tree"]["chord_count"]),
    )

    # --- Supporting equalities: Green matrix and identities ----------------
    for i in range(PORTS):
        for j in range(PORTS):
            scaled = g_pkg[i][j] * 180
            comp.supporting(
                "green_matrix",
                f"180*G[{i}][{j}]",
                scaled,
                Fraction(int(committed["green_matrix_times_180"][i][j])),
            )
    identities = committed["green_identities"]
    lg_exact = all(
        sum((lap_pkg[i][k] * g_pkg[k][j] for k in range(PORTS)), start=ZERO)
        == (ONE if i == j else ZERO) - Fraction(1, 12)
        for i in range(PORTS)
        for j in range(PORTS)
    )
    comp.supporting(
        "green_identities", "lg_identity_exact",
        lg_exact, bool(identities["lg_identity_exact"]),
    )
    comp.supporting(
        "green_identities", "symmetric",
        all(
            g_pkg[i][j] == g_pkg[j][i]
            for i in range(PORTS) for j in range(PORTS)
        ),
        bool(identities["symmetric"]),
    )
    comp.supporting(
        "green_identities", "row_sums_zero",
        all(sum(g_pkg[i], start=ZERO) == 0 for i in range(PORTS)),
        bool(identities["row_sums_zero"]),
    )
    comp.supporting(
        "green_identities", "laplacian_rank",
        base_carrier.matrix_rank(lap_pkg),
        int(identities["laplacian_rank"]),
    )

    # --- Supporting equalities: distance receipt ---------------------------
    distances = _bfs_distances()
    single_source = {
        int(k): _fraction(v)
        for k, v in committed["distance_receipt"]["single_source_values"].items()
    }
    for p in range(PORTS):
        for q in range(PORTS):
            comp.supporting(
                "distance",
                f"G[{p}][{q}]=value[hop]",
                g_pkg[p][q],
                single_source[distances[p][q]],
            )
    ordered = sorted(single_source)
    comp.supporting(
        "distance", "strictly_decreasing",
        all(
            single_source[a] > single_source[b]
            for a, b in zip(ordered, ordered[1:], strict=False)
        ),
        bool(committed["distance_receipt"]["strictly_decreasing"]),
    )

    # --- Supporting equalities: dipole class values ------------------------
    dipole_load = [ZERO] * PORTS
    poles = [int(x) for x in committed["dipole_receipt"]["poles"]]
    dipole_load[poles[0]], dipole_load[poles[1]] = ONE, -ONE
    class_values = {
        key: _fraction(value)
        for key, value in committed["dipole_receipt"]["class_values"].items()
    }
    phi_dipole_pkg = _package_potential(g_pkg, dipole_load)
    seen_classes = set()
    for p in range(PORTS):
        key = f"{distances[poles[0]][p]},{distances[poles[1]][p]}"
        if key not in class_values:
            raise _Inconclusive(f"committed dipole class missing: {key}")
        comp.supporting(
            "dipole", f"phi[{p}]", phi_dipole_pkg[p], class_values[key]
        )
        seen_classes.add(key)
    comp.supporting(
        "dipole", "class_census_complete",
        seen_classes == set(class_values),
        bool(committed["dipole_receipt"]["class_census_complete"]),
    )

    # --- Supporting equalities: repair bridge ------------------------------
    average = _pair_average_matrix()
    for i in range(PORTS):
        for j in range(PORTS):
            expected = (ONE if i == j else ZERO) - Fraction(1, 60) * lap_pkg[i][j]
            comp.supporting(
                "repair_bridge", f"avg[{i}][{j}]", average[i][j], expected
            )
    comp.supporting(
        "repair_bridge", "pair_average_identity_exact",
        True, bool(committed["repair_bridge"]["pair_average_identity_exact"]),
    )
    comp.supporting(
        "repair_bridge", "kernel_dimension",
        1, int(committed["repair_bridge"]["kernel_dimension"]),
    )

    # --- Declared observables over the three committed loads ---------------
    for index, (entry, load) in enumerate(
        zip(committed["thomson_spot_checks"], rec_loads, strict=True)
    ):
        # Receipt side.
        rec_phi = [
            sum((rec_green[p][q] * load[q] for q in range(PORTS)), start=ZERO)
            for p in range(PORTS)
        ]
        rec_flux = [
            rec_phi[rec_right[e]] - rec_phi[rec_left[e]] for e in range(SEAMS)
        ]
        rec_tree = _tree_solution_from_tables(
            rec_left, rec_right, rec_tree_seams, load
        )
        rec_cycle_part = [
            t - c for t, c in zip(rec_tree, rec_flux, strict=True)
        ]

        # Package side.
        if green_override is None:
            phi_pkg = green.green_potential(load)
            flux_pkg = green.seam_flux(load)
            chords_coulomb_pkg = green.chord_field_strength_components(flux_pkg)
            tree_pkg = base_carrier.tree_solution(load)
            chords_tree_pkg = green.chord_field_strength_components(tree_pkg)
        else:
            phi_pkg = _package_potential(g_pkg, load)
            flux_pkg = base_carrier.coboundary(phi_pkg)
            tree_pkg = base_carrier.tree_solution(load)

            def _strength(field: Vector) -> dict[int, Fraction]:
                pot = _package_potential(
                    g_pkg, base_carrier.boundary(field)
                )
                grad = base_carrier.coboundary(pot)
                return {
                    chord: field[chord] - grad[chord]
                    for chord in CHORD_SEAMS
                }

            chords_coulomb_pkg = _strength(flux_pkg)
            chords_tree_pkg = _strength(tree_pkg)

        # Observable 1: port potential differences, all ordered pairs.
        for p in range(PORTS):
            for q in range(PORTS):
                if p == q:
                    continue
                comp.observable(
                    "port_potential_difference",
                    f"load{index}:({p},{q})",
                    phi_pkg[p] - phi_pkg[q],
                    rec_phi[p] - rec_phi[q],
                )

        # Observable 2: seam fluxes on the thirty committed seams.
        for e in range(SEAMS):
            comp.observable(
                "seam_flux",
                f"load{index}:seam{e}",
                flux_pkg[e],
                rec_flux[e],
            )

        # Observable 3: chord field-strength components.  The Coulomb field
        # is a pure gradient, so its committed components are zero; the tree
        # solution's committed components are the Thomson cycle part.
        for chord in CHORD_SEAMS:
            comp.observable(
                "chord_field_strength_component",
                f"load{index}:coulomb:chord{chord}",
                chords_coulomb_pkg[chord],
                ZERO,
            )
            comp.observable(
                "chord_field_strength_component",
                f"load{index}:tree:chord{chord}",
                chords_tree_pkg[chord],
                rec_cycle_part[chord],
            )

        # Committed Thomson anchors (exact energy strings).
        comp.supporting(
            "thomson", f"load{index}:coulomb_energy",
            base_carrier.seam_energy(flux_pkg),
            _fraction(entry["coulomb_energy"]),
        )
        comp.supporting(
            "thomson", f"load{index}:tree_solution_energy",
            base_carrier.seam_energy(tree_pkg),
            _fraction(entry["tree_solution_energy"]),
        )
        cycle_part_pkg = [
            t - c for t, c in zip(tree_pkg, flux_pkg, strict=True)
        ]
        comp.supporting(
            "thomson", f"load{index}:cycle_part_energy",
            base_carrier.seam_energy(cycle_part_pkg),
            _fraction(entry["cycle_part_energy"]),
        )
        comp.supporting(
            "thomson", f"load{index}:gauss_boundary",
            base_carrier.boundary(flux_pkg) == load,
            bool(entry["gauss_boundary_check_passed"]),
        )
        comp.supporting(
            "thomson", f"load{index}:energy_decomposition",
            base_carrier.seam_energy(tree_pkg)
            == base_carrier.seam_energy(flux_pkg)
            + base_carrier.seam_energy(cycle_part_pkg),
            bool(entry["energy_decomposition_exact"]),
        )

    verdict = "REPLICATED" if not comp.mismatches else "FAILED"
    declared_total = sum(
        counts["comparisons"] for counts in comp.observable_counts.values()
    )
    return {
        "verdict": verdict,
        "observables": comp.observable_counts,
        "declared_comparisons_total": declared_total,
        "supporting_comparisons": comp.supporting_counts,
        "supporting_comparisons_total": sum(comp.supporting_counts.values()),
        "mismatch_count": len(comp.mismatches),
        "mismatches": comp.mismatches[:40],
        "decision_rule_template": {
            "template_only": True,
            "not_a_freeze": True,
            "armed": False,
            "verdict_labels": list(VERDICT_LABELS),
        },
    }


def main() -> int:
    report = replay_committed_receipt()
    print(json.dumps(report, indent=2, default=str))
    print(f"EM_HANDOFF_REPLAY_VERDICT: {report['verdict']}")
    return 0 if report["verdict"] == "REPLICATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
