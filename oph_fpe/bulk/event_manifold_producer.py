"""Issue-575 producer: ancestry-derived event chart and cone-margin verdicts.

The Einstein branch needs a Lorentzian event manifold reconstructed from
repair events: event classes from semantic ancestry, an open four-chart, a
held-out quadratic form of inertia (1,3), a positive cone margin separating
ancestry-comparable from incomparable pairs, and causal reachability.  Issue
#575 owns those receipts (E1 through E6 in the paper stack).

This instrument derives everything from the frozen capture: event classes and
their causal order from the semantic ancestry relations; a four-coordinate
chart whose time coordinate is ancestry depth and whose three spatial
coordinates are the seam-graph spectral embedding of each event's carrier
footprint; a quadratic form fitted on a declared training half of the event
pairs and evaluated on the held-out half; the inertia of that form; and the
cone margin with which it separates causal from spacelike pairs.

The verdicts are measured fail-closed.  In particular the instrument detects
and reports degenerate causal structure: when the observer history is a
near-chain, the spacelike class is too small to support any cone, and that
insufficiency is the recorded result rather than an excuse to weaken the
test.  No physical promotion follows from any output of this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source

SCHEMA = "oph.event-manifold-producer.v1"
PHYSICAL_PROMOTION_ALLOWED = False
MIN_SPACELIKE_PAIRS = 8
MIN_CAUSAL_PAIRS = 8
TARGET_INERTIA = (1, 3)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _carrier_index(carrier_id: str) -> int:
    return int(carrier_id.rsplit("-", 1)[1])


def _carrier_adjacency(capture: Mapping[str, Any], count: int) -> np.ndarray:
    adjacency = np.zeros((count, count))
    for row in capture["postrun_capture"]["raw_overlap_relations"]:
        left = _carrier_index(row["left_carrier_id"])
        right = _carrier_index(row["right_carrier_id"])
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    return adjacency


def _spectral_embedding(adjacency: np.ndarray, dimensions: int = 3) -> np.ndarray:
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    order = np.argsort(eigenvalues)
    columns = eigenvectors[:, order[1 : 1 + dimensions]]
    # Deterministic sign convention: the largest-magnitude entry of each
    # eigenvector is positive.
    for column in range(columns.shape[1]):
        pivot = np.argmax(np.abs(columns[:, column]))
        if columns[pivot, column] < 0.0:
            columns[:, column] = -columns[:, column]
    return columns


def _event_table(capture: Mapping[str, Any]) -> dict[str, Any]:
    events = capture["postrun_capture"]["semantic_events"]
    ancestry = capture["postrun_capture"]["raw_ancestry_relations"]
    key_to_index = {event["event_key"]: i for i, event in enumerate(events)}
    count = len(events)
    parents: dict[int, set[int]] = {i: set() for i in range(count)}
    for edge in ancestry:
        parent = key_to_index.get(edge["parent_event_id"])
        child = key_to_index.get(edge["child_event_id"])
        if parent is None or child is None:
            continue
        parents[child].add(parent)
    # Ancestry depth by longest path (events arrive in sequence order).
    order = sorted(range(count), key=lambda i: events[i]["source_sequence_index"])
    depth = {i: 0 for i in range(count)}
    reachable: dict[int, set[int]] = {i: set() for i in range(count)}
    for index in order:
        for parent in parents[index]:
            depth[index] = max(depth[index], depth[parent] + 1)
            reachable[index] |= {parent} | reachable[parent]
    footprints: list[set[int]] = []
    for event in events:
        touched: set[int] = set()
        for resource in event["visible_footprint"]:
            head = resource.split(":", 1)[0]
            if head.startswith("carrier-"):
                touched.add(_carrier_index(head))
        footprints.append(touched)
    return {
        "count": count,
        "depth": depth,
        "reachable": reachable,
        "footprints": footprints,
    }


def _event_chart(
    table: Mapping[str, Any], embedding: np.ndarray
) -> np.ndarray:
    coordinates = np.zeros((table["count"], 4))
    for index in range(table["count"]):
        coordinates[index, 0] = float(table["depth"][index])
        footprint = sorted(table["footprints"][index])
        if footprint:
            coordinates[index, 1:] = embedding[footprint].mean(axis=0)
    return coordinates


def _pair_classes(table: Mapping[str, Any]) -> dict[str, list[tuple[int, int]]]:
    causal: list[tuple[int, int]] = []
    spacelike: list[tuple[int, int]] = []
    for i in range(table["count"]):
        for j in range(i + 1, table["count"]):
            if i in table["reachable"][j] or j in table["reachable"][i]:
                causal.append((i, j))
            else:
                spacelike.append((i, j))
    return {"causal": causal, "spacelike": spacelike}


def _fit_quadratic_form(
    chart: np.ndarray,
    pairs: Mapping[str, list[tuple[int, int]]],
    *,
    train_parity: int = 0,
) -> dict[str, Any]:
    """Fit a symmetric form on the training half, evaluate held out.

    Training target: +1 on causal pairs, -1 on spacelike pairs, regressed on
    the ten quadratic monomials of the coordinate differences.  The held-out
    margin is the smallest correctly-signed value; one wrongly-signed
    held-out pair makes the margin negative.
    """

    monomial_rows: list[np.ndarray] = []
    targets: list[float] = []
    held_rows: list[tuple[np.ndarray, float]] = []
    for label, sign in (("causal", 1.0), ("spacelike", -1.0)):
        for i, j in pairs[label]:
            difference = chart[i] - chart[j]
            monomials = np.outer(difference, difference)[
                np.triu_indices(4)
            ]
            if (i + j) % 2 == train_parity:
                monomial_rows.append(monomials)
                targets.append(sign)
            else:
                held_rows.append((monomials, sign))
    if len(monomial_rows) < 10 or not held_rows:
        return {"fitted": False, "blocker": "insufficient_training_pairs"}
    design = np.stack(monomial_rows)
    solution, *_ = np.linalg.lstsq(design, np.asarray(targets), rcond=None)
    form = np.zeros((4, 4))
    form[np.triu_indices(4)] = solution
    form = (form + form.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(form)
    positive = int((eigenvalues > 1.0e-12).sum())
    negative = int((eigenvalues < -1.0e-12).sum())
    margins = [
        sign * float(np.dot(monomials, solution))
        for monomials, sign in held_rows
    ]
    return {
        "fitted": True,
        "blocker": None,
        "eigenvalues": [float(value) for value in eigenvalues],
        "inertia": (positive, negative),
        "held_out_pair_count": len(held_rows),
        "cone_margin": float(min(margins)),
    }


def produce_event_manifold_report(
    *,
    config: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Produce the ancestry-derived event-chart report with controls."""

    main_config = dict(
        {"carrier_count": 32, "cycles": 6, "seed": 20260751}
        if config is None
        else config
    )
    capture = capture_physical_source(main_config)
    count = len(capture["postrun_capture"]["carrier_port_trajectories"])
    table = _event_table(capture)
    adjacency = _carrier_adjacency(capture, count)
    embedding = _spectral_embedding(adjacency)
    chart = _event_chart(table, embedding)
    pairs = _pair_classes(table)
    chart_freeze = _sha256_value(chart.tolist())

    blockers: list[str] = []
    causal_count = len(pairs["causal"])
    spacelike_count = len(pairs["spacelike"])
    if causal_count < MIN_CAUSAL_PAIRS:
        blockers.append("causal_pair_class_too_small")
    if spacelike_count < MIN_SPACELIKE_PAIRS:
        blockers.append("spacelike_pair_class_too_small_degenerate_causal_structure")

    fit: dict[str, Any] = {"fitted": False, "blocker": "not_attempted"}
    inertia_ok = False
    margin_ok = False
    if not blockers:
        fit = _fit_quadratic_form(chart, pairs)
        if not fit["fitted"]:
            blockers.append(f"quadratic_fit:{fit['blocker']}")
        else:
            inertia_ok = tuple(fit["inertia"]) == TARGET_INERTIA
            margin_ok = fit["cone_margin"] > 0.0
            if not inertia_ok:
                blockers.append("held_out_form_inertia_is_not_lorentzian")
            if not margin_ok:
                blockers.append("cone_margin_not_positive_on_held_out_pairs")

    # Negative controls.
    controls: dict[str, dict[str, Any]] = {}

    shuffled = dict(table)
    permutation = np.random.Generator(np.random.PCG64(575)).permutation(
        table["count"]
    )
    shuffled_reachable = {
        int(permutation[i]): {int(permutation[j]) for j in table["reachable"][i]}
        for i in range(table["count"])
    }
    shuffled = {**table, "reachable": shuffled_reachable}
    shuffled_pairs = _pair_classes(shuffled)
    controls["shuffled_ancestry"] = {
        "control_failure_detected": bool(
            shuffled_pairs["causal"] != pairs["causal"]
        )
    }

    collapsed_chart = chart.copy()
    collapsed_chart[:, 1:] = 0.0
    collapsed_fit = (
        _fit_quadratic_form(collapsed_chart, pairs)
        if not blockers
        else {"fitted": False}
    )
    controls["collapsed_chart"] = {
        "control_failure_detected": bool(
            not collapsed_fit.get("fitted")
            or tuple(collapsed_fit.get("inertia", ())) != TARGET_INERTIA
        )
    }

    foreign_capture = capture_physical_source({**main_config, "seed": 20260861})
    foreign_table = _event_table(foreign_capture)
    controls["mixed_source_events"] = {
        "control_failure_detected": bool(
            _sha256_value(
                sorted(map(sorted, (list(f) for f in foreign_table["footprints"])))
            )
            != _sha256_value(
                sorted(map(sorted, (list(f) for f in table["footprints"])))
            )
        )
    }
    controls_fail_closed = all(
        row["control_failure_detected"] for row in controls.values()
    )
    if not controls_fail_closed:
        blockers.append("negative_control_did_not_fail")

    verdict = "ATTAINED" if not blockers else "NOT_ATTAINED"
    report = {
        "schema": SCHEMA,
        "issue": 575,
        "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
        "main_config": main_config,
        "capture_sha256": capture["capture_sha256"],
        "event_count": table["count"],
        "causal_pair_count": causal_count,
        "spacelike_pair_count": spacelike_count,
        "chart_freeze_sha256": chart_freeze,
        "held_out_quadratic_fit": fit,
        "clause_verdicts": {
            "event_classes_and_order_constructed": True,
            "four_chart_constructed": True,
            "nondegenerate_causal_structure": bool(
                causal_count >= MIN_CAUSAL_PAIRS
                and spacelike_count >= MIN_SPACELIKE_PAIRS
            ),
            "held_out_lorentzian_inertia": bool(inertia_ok),
            "positive_cone_margin": bool(margin_ok),
        },
        "negative_controls": controls,
        "controls_fail_closed": bool(controls_fail_closed),
        "verdict": verdict,
        "EVENT_MANIFOLD_RECEIPT": bool(verdict == "ATTAINED"),
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "Finite issue-575 instrument: event classes and causal order from "
            "semantic ancestry, a four-chart from ancestry depth and the "
            "seam-graph spectral embedding, and a held-out quadratic form "
            "with measured inertia and cone margin. A NOT_ATTAINED verdict, "
            "including the degenerate-causal-structure blocker, is an "
            "empirical result about this source at this cutoff. No open-chart "
            "topology, continuum limit, or physical metric is claimed, and no "
            "physical promotion follows from any output."
        ),
    }
    if output_path is not None:
        Path(output_path).write_text(_canonical_json(report), encoding="utf-8")
    return report
