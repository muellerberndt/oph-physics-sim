"""Observer mutual-agreement certificate over a shared gauge record.

OPH claim boundary implemented here: observers are primary. Each observer
experiences an integer 3+1 chart (H3 spatial chart plus one modular time
direction). A "bulk spacetime" is never fundamental; what can hold at scale
is AGREEMENT, meaning that overlapping observers admit one mutual
re-gauging of their charts and that these re-gaugings compose consistently
on triple overlaps (the Cech cocycle condition). This module measures that
agreement on run artifacts.

Policy (enforced by construction): no output of this module contains a
fractional bulk dimension. ``bulk_dimension_claim`` is always ``None``.
Dimensionality statements are integer observer-chart verdicts imported from
the observer-experience receipts, plus the agreement fractions below.
Continuous dimension estimators elsewhere in the codebase are internal
diagnostics and never physical claims.

Construction (v1):

- The committed record supplies one S3 edge-gauge field ``gauge`` on edges
  ``(left, right)`` (``s3_gauge_state.npz``).
- Each patch observer ``i`` carries a deterministic private vertex frame
  ``f_i`` (a SplitMix-hashed S3 element per vertex, seeded by observer id).
  Its chart view of an edge ``e = (u, v)`` inside its support is the dressed
  gauge ``view_i(e) = f_i(u) g(e) f_i(v)^{-1}``. The frame stands for the
  observer's private chart-gauge choice; the record is shared, the frames
  are not.
- For an overlapping pair ``(i, j)`` the certificate reconstructs a vertex
  re-gauging ``h`` with ``view_j(e) = h(u) view_i(e) h(v)^{-1}`` by BFS over
  the overlap subgraph and counts violated edges. Perfect agreement means
  defect 0 on every non-tree edge.
- For a triple ``(i, j, k)`` the recovered maps must satisfy
  ``h_ik = h_jk h_ij`` on common vertices (cocycle condition).
- Controls: an edge-shuffled view must fail (defect near 1 - 1/6), and the
  identity-frame sanity pair must recover ``h = id`` exactly.

The v1 certificate establishes the harness and the receipt names on the
shared-record dynamics. The follow-on gate (tracked in the run claim
boundary) is genuinely independent per-observer commit histories, where the
same certificate measures physical consensus quality rather than chart
bookkeeping.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.finite_groups import S3_INV, S3_MUL

OBSERVER_AGREEMENT_SCHEMA = "observer_mutual_agreement_v1"
MUTUAL_GAUGE_CHART_AGREEMENT_RECEIPT = "MUTUAL_GAUGE_CHART_AGREEMENT_RECEIPT"
OBSERVER_SPACETIME_CONSENSUS_RECEIPT = "OBSERVER_SPACETIME_CONSENSUS_RECEIPT"

_SPLITMIX_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    mixed = (int(value) + 0x9E3779B97F4A7C15) & _SPLITMIX_MASK
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _SPLITMIX_MASK
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _SPLITMIX_MASK
    return (mixed ^ (mixed >> 31)) & _SPLITMIX_MASK


def observer_frame(seed: int, observer_id: int, node_ids: np.ndarray) -> np.ndarray:
    """Deterministic private S3 frame for one observer over given vertices."""

    nodes = np.asarray(node_ids, dtype=np.int64)
    base = _splitmix64((int(seed) << 32) ^ (int(observer_id) * 0x9E3779B9))
    values = np.fromiter(
        (_splitmix64(base ^ _splitmix64(int(node))) % 6 for node in nodes),
        dtype=np.int64,
        count=nodes.size,
    )
    return values


def dressed_edge_views(
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    edge_gauge: np.ndarray,
    frame_by_node: dict[int, int],
) -> np.ndarray:
    """Observer chart view of edges: ``f(u) g(e) f(v)^{-1}``."""

    left_frame = np.asarray([frame_by_node[int(u)] for u in edge_left], dtype=np.int64)
    right_frame = np.asarray([frame_by_node[int(v)] for v in edge_right], dtype=np.int64)
    dressed = S3_MUL[left_frame, np.asarray(edge_gauge, dtype=np.int64)]
    return np.asarray(S3_MUL[dressed, S3_INV[right_frame]], dtype=np.int64)


def _overlap_edges(
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    node_set: set[int],
) -> np.ndarray:
    keep = np.fromiter(
        ((int(u) in node_set) and (int(v) in node_set) for u, v in zip(edge_left, edge_right, strict=True)),
        dtype=bool,
        count=edge_left.size,
    )
    return np.nonzero(keep)[0]


def recover_regauging(
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    view_a: np.ndarray,
    view_b: np.ndarray,
) -> dict[str, Any]:
    """Solve ``view_b(e) = h(u) view_a(e) h(v)^{-1}`` for a vertex map ``h``.

    BFS over the overlap subgraph propagates ``h`` along a spanning tree
    (``h(v) = view_b(e)^{-1} h(u) view_a(e)`` when walking ``u -> v``); every
    non-tree edge then either confirms or violates the reconstruction. All
    six root elements are tried per connected component and the best kept.
    Returns the defect fraction over checkable (non-tree) edges, component
    structure, and the recovered map.
    """

    nodes = sorted({int(u) for u in edge_left} | {int(v) for v in edge_right})
    node_index = {node: position for position, node in enumerate(nodes)}
    adjacency: dict[int, list[tuple[int, int, bool]]] = {index: [] for index in range(len(nodes))}
    for edge_id in range(edge_left.size):
        u = node_index[int(edge_left[edge_id])]
        v = node_index[int(edge_right[edge_id])]
        adjacency[u].append((v, edge_id, True))
        adjacency[v].append((u, edge_id, False))

    h_map = np.full(len(nodes), -1, dtype=np.int64)
    visited = np.zeros(len(nodes), dtype=bool)
    violated_total = 0
    checkable_total = 0
    component_count = 0
    ambiguous_components = 0

    for start in range(len(nodes)):
        if visited[start]:
            continue
        component_count += 1
        component_nodes: list[int] = []
        tree_edges: list[tuple[int, int, int, bool]] = []
        component_edges: set[int] = set()
        queue = deque([start])
        visited[start] = True
        while queue:
            u = queue.popleft()
            component_nodes.append(u)
            for v, edge_id, forward in adjacency[u]:
                component_edges.add(edge_id)
                if not visited[v]:
                    visited[v] = True
                    tree_edges.append((u, v, edge_id, forward))
                    queue.append(v)
        best = None
        zero_violation_roots = 0
        tree_ids = {edge_id for _, _, edge_id, _ in tree_edges}
        for root_element in range(6):
            h_try = np.full(len(nodes), -1, dtype=np.int64)
            h_try[start] = root_element
            for u, v, edge_id, forward in tree_edges:
                va = int(view_a[edge_id])
                vb = int(view_b[edge_id])
                if forward:
                    h_try[v] = int(S3_MUL[S3_MUL[S3_INV[vb], h_try[u]], va])
                else:
                    # walking v -> u along stored edge (u_edge, v_edge):
                    # h(u_edge) = view_b h(v_edge) view_a^{-1}
                    h_try[v] = int(S3_MUL[S3_MUL[vb, h_try[u]], S3_INV[va]])
            violated = 0
            checkable = 0
            for edge_id in component_edges:
                if edge_id in tree_ids:
                    continue
                u = node_index[int(edge_left[edge_id])]
                v = node_index[int(edge_right[edge_id])]
                predicted = S3_MUL[S3_MUL[h_try[u], int(view_a[edge_id])], S3_INV[h_try[v]]]
                checkable += 1
                if int(predicted) != int(view_b[edge_id]):
                    violated += 1
            if violated == 0 and checkable > 0:
                zero_violation_roots += 1
            if best is None or violated < best[0]:
                best = (violated, checkable, h_try.copy())
        violated, checkable, h_component = best
        # A component whose section is not pinned by any non-tree edge, or is
        # reproduced by more than one root element, carries an arbitrary
        # section choice: exclude it from composition-sensitive uses.
        if checkable == 0 or zero_violation_roots > 1:
            ambiguous_components += 1
        violated_total += violated
        checkable_total += checkable
        for u in component_nodes:
            h_map[u] = h_component[u]

    defect = float(violated_total) / float(checkable_total) if checkable_total else None
    return {
        "nodes": nodes,
        "h_by_node": {int(node): int(h_map[node_index[node]]) for node in nodes},
        "components": component_count,
        "ambiguous_components": ambiguous_components,
        "section_unique": bool(component_count >= 1 and ambiguous_components == 0),
        "checkable_edges": int(checkable_total),
        "violated_edges": int(violated_total),
        "defect": defect,
    }


def _load_patch_observers(run_dir: Path, max_observers: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (run_dir / "observer_views.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("view_type") != "patch_observer":
                continue
            rows.append(
                {
                    "observer_id": int(row.get("observer_id", len(rows))),
                    "support_nodes": [int(node) for node in row.get("support_nodes", [])],
                    "modular_depth_mean": row.get("modular_depth_mean"),
                    "observer_relative_times": row.get("observer_relative_times", []),
                }
            )
            if max_observers is not None and len(rows) >= int(max_observers):
                break
    return rows


def observer_agreement_report(
    run_dir: str | Path,
    *,
    seed: int = 1,
    max_pairs: int = 512,
    max_triples: int = 128,
    min_overlap_edges: int = 8,
    max_observers: int | None = None,
    full_records: bool = False,
) -> dict[str, Any]:
    run = Path(run_dir)
    gauge_path = run / "s3_gauge_state.npz"
    if not gauge_path.exists():
        return {
            "schema": OBSERVER_AGREEMENT_SCHEMA,
            "status": "missing_s3_gauge_state",
            "bulk_dimension_claim": None,
        }
    payload = np.load(gauge_path)
    edge_left = np.asarray(payload["left"], dtype=np.int64)
    edge_right = np.asarray(payload["right"], dtype=np.int64)
    edge_gauge = np.asarray(payload["gauge"], dtype=np.int64)

    observers = _load_patch_observers(run, max_observers)
    rng = np.random.default_rng(int(seed))

    supports = [set(observer["support_nodes"]) for observer in observers]
    frames: list[dict[int, int]] = []
    for observer in observers:
        nodes = np.asarray(observer["support_nodes"], dtype=np.int64)
        frame_values = observer_frame(seed, observer["observer_id"], nodes)
        frames.append({int(node): int(value) for node, value in zip(nodes, frame_values, strict=True)})

    # Candidate pairs by support overlap, via an inverted patch -> observer
    # index (the direct O(n^2) support-intersection scan is infeasible at
    # tens of thousands of observers).
    patch_to_observers: dict[int, list[int]] = {}
    for index, support in enumerate(supports):
        for node in support:
            patch_to_observers.setdefault(node, []).append(index)
    co_support: dict[tuple[int, int], int] = {}
    for members in patch_to_observers.values():
        if len(members) < 2:
            continue
        for position, a in enumerate(members):
            for b in members[position + 1 :]:
                key = (a, b) if a < b else (b, a)
                co_support[key] = co_support.get(key, 0) + 1
    pair_candidates = [key for key, count in co_support.items() if count >= 3]
    rng.shuffle(pair_candidates)

    # Triangle-first ordering: the cocycle statistic needs closed triangles
    # of evaluated pairs, which random pair sampling rarely produces. Build
    # the overlap graph, enumerate a bounded set of triangles, and push
    # their pairs to the front of the evaluation order.
    candidate_set = set(pair_candidates)
    adjacency: dict[int, set[int]] = {}
    for a, b in pair_candidates:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    triangle_pairs: list[tuple[int, int]] = []
    triangles_found = 0
    for a in sorted(adjacency, key=lambda index: -len(adjacency[index])):
        if triangles_found >= int(max_triples) * 2:
            break
        neighbors = sorted(adjacency[a])
        for i, b in enumerate(neighbors):
            if b <= a:
                continue
            common = adjacency[a] & adjacency[b]
            for c in sorted(common):
                if c <= b:
                    continue
                triangles_found += 1
                for pair in ((a, b), (b, c), (a, c)):
                    key = (min(pair), max(pair))
                    if key in candidate_set:
                        triangle_pairs.append(key)
                if triangles_found >= int(max_triples) * 2:
                    break
            if triangles_found >= int(max_triples) * 2:
                break
    seen_pairs: set[tuple[int, int]] = set()
    ordered_candidates: list[tuple[int, int]] = []
    for key in triangle_pairs + pair_candidates:
        if key not in seen_pairs:
            seen_pairs.add(key)
            ordered_candidates.append(key)
    pair_candidates = ordered_candidates

    pair_records: list[dict[str, Any]] = []
    control_defects: list[float] = []
    pair_regaugings: dict[tuple[int, int], dict[int, int]] = {}
    for a, b in pair_candidates:
        if len(pair_records) >= int(max_pairs):
            break
        shared = supports[a] & supports[b]
        edge_ids = _overlap_edges(edge_left, edge_right, shared)
        if edge_ids.size < int(min_overlap_edges):
            continue
        sub_left = edge_left[edge_ids]
        sub_right = edge_right[edge_ids]
        sub_gauge = edge_gauge[edge_ids]
        view_a = dressed_edge_views(sub_left, sub_right, sub_gauge, frames[a])
        view_b = dressed_edge_views(sub_left, sub_right, sub_gauge, frames[b])
        recovery = recover_regauging(sub_left, sub_right, view_a, view_b)
        if recovery["defect"] is None:
            # No non-tree edge pins the section: the overlap carries no
            # agreement evidence either way.
            continue
        pair_records.append(
            {
                "observer_a": observers[a]["observer_id"],
                "observer_b": observers[b]["observer_id"],
                "overlap_nodes": len(shared),
                "overlap_edges": int(edge_ids.size),
                "components": recovery["components"],
                "section_unique": recovery["section_unique"],
                "checkable_edges": recovery["checkable_edges"],
                "defect": recovery["defect"],
            }
        )
        if recovery["section_unique"]:
            pair_regaugings[(a, b)] = recovery["h_by_node"]
        # Shuffled control: replace observer b's view with random labels.
        shuffled = rng.integers(0, 6, size=view_b.size, dtype=np.int64)
        control = recover_regauging(sub_left, sub_right, view_a, shuffled)
        if control["defect"] is not None:
            control_defects.append(control["defect"])

    # Cocycle condition on triples with pairwise recoveries available.
    evaluated_pairs = {key for key in pair_regaugings}
    triple_records: list[dict[str, Any]] = []
    indices_with_pairs = sorted({index for pair in evaluated_pairs for index in pair})
    for a in indices_with_pairs:
        if len(triple_records) >= int(max_triples):
            break
        for b in indices_with_pairs:
            if b <= a or (a, b) not in evaluated_pairs:
                continue
            for c in indices_with_pairs:
                if c <= b:
                    continue
                if (b, c) not in evaluated_pairs or (a, c) not in evaluated_pairs:
                    continue
                common = supports[a] & supports[b] & supports[c]
                h_ab = pair_regaugings[(a, b)]
                h_bc = pair_regaugings[(b, c)]
                h_ac = pair_regaugings[(a, c)]
                checked = 0
                failed = 0
                for node in common:
                    if node in h_ab and node in h_bc and node in h_ac:
                        composed = int(S3_MUL[h_bc[node], h_ab[node]])
                        checked += 1
                        if composed != int(h_ac[node]):
                            failed += 1
                if checked >= 3:
                    triple_records.append(
                        {
                            "observers": [
                                observers[a]["observer_id"],
                                observers[b]["observer_id"],
                                observers[c]["observer_id"],
                            ],
                            "checked_nodes": checked,
                            "cocycle_failures": failed,
                            "cocycle_defect": failed / checked,
                        }
                    )
                if len(triple_records) >= int(max_triples):
                    break
            if len(triple_records) >= int(max_triples):
                break

    defects = np.asarray([record["defect"] for record in pair_records], dtype=float)
    cocycle_defects = np.asarray(
        [record["cocycle_defect"] for record in triple_records], dtype=float
    )
    median_defect = float(np.median(defects)) if defects.size else None
    control_median = float(np.median(np.asarray(control_defects))) if control_defects else None
    perfect_fraction = float(np.mean(defects == 0.0)) if defects.size else None
    cocycle_perfect = float(np.mean(cocycle_defects == 0.0)) if cocycle_defects.size else None

    # Modular-time agreement: shared relative-time grids across observers.
    grids = {tuple(observer["observer_relative_times"]) for observer in observers if observer["observer_relative_times"]}
    shared_grid = len(grids) <= 1 and bool(grids)
    depth_means = np.asarray(
        [observer["modular_depth_mean"] for observer in observers if observer["modular_depth_mean"] is not None],
        dtype=float,
    )

    experience_path = run / "observer_modular_experience_report.json"
    experienced_spatial = None
    experienced_time = None
    experience_receipt = None
    if experience_path.exists():
        experience = json.loads(experience_path.read_text())
        experience_receipt = bool(
            experience.get("observer_facing_3p1d_h3_experience_receipt", False)
        )
        if experience_receipt:
            experienced_spatial = 3
            experienced_time = 1

    agreement_receipt = bool(
        defects.size >= 16
        and median_defect is not None
        and median_defect <= 0.02
        and (control_median is None or control_median >= 0.30)
        and (cocycle_defects.size == 0 or float(np.median(cocycle_defects)) <= 0.02)
    )
    consensus_receipt = bool(agreement_receipt and experience_receipt)
    blockers = []
    if defects.size < 16:
        blockers.append("insufficient_overlapping_pairs")
    if median_defect is not None and median_defect > 0.02:
        blockers.append("pair_regauging_defect_above_tolerance")
    if control_median is not None and control_median < 0.30:
        blockers.append("shuffled_control_not_separated")
    if cocycle_defects.size and float(np.median(cocycle_defects)) > 0.02:
        blockers.append("cocycle_defect_above_tolerance")
    if not experience_receipt:
        blockers.append("observer_3p1d_experience_receipt_missing_or_false")

    return {
        "schema": OBSERVER_AGREEMENT_SCHEMA,
        "status": "evaluated",
        "parameters": {
            "seed": int(seed),
            "max_pairs": int(max_pairs),
            "max_triples": int(max_triples),
            "min_overlap_edges": int(min_overlap_edges),
            "max_observers": None if max_observers is None else int(max_observers),
        },
        "population": {
            "patch_observers": len(observers),
            "candidate_pairs": len(pair_candidates),
            "evaluated_pairs": int(defects.size),
            "evaluated_triples": int(cocycle_defects.size),
            "mean_overlap_edges": float(
                np.mean([record["overlap_edges"] for record in pair_records])
            )
            if pair_records
            else None,
        },
        "pair_agreement": {
            "median_defect": median_defect,
            "p95_defect": float(np.percentile(defects, 95)) if defects.size else None,
            "perfect_fraction": perfect_fraction,
            "fragmented_pairs": int(
                sum(1 for record in pair_records if record["components"] > 1)
            ),
            "section_unique_pairs": int(
                sum(1 for record in pair_records if record["section_unique"])
            ),
        },
        "control": {
            "median_defect_shuffled": control_median,
            "separation_receipt": bool(
                median_defect is not None
                and control_median is not None
                and control_median - median_defect >= 0.25
            ),
        },
        "cocycle": {
            "median_defect": float(np.median(cocycle_defects)) if cocycle_defects.size else None,
            "perfect_fraction": cocycle_perfect,
        },
        "modular_time": {
            "shared_relative_time_grid": shared_grid,
            "grid_count": len(grids),
            "modular_depth_mean_std": float(np.std(depth_means)) if depth_means.size else None,
        },
        "experienced_chart": {
            "spatial_dimension": experienced_spatial,
            "time_dimension": experienced_time,
            "integer_by_construction": True,
            "source": "observer_modular_experience_report.json",
        },
        MUTUAL_GAUGE_CHART_AGREEMENT_RECEIPT: agreement_receipt,
        OBSERVER_SPACETIME_CONSENSUS_RECEIPT: consensus_receipt,
        "blockers": blockers,
        "bulk_dimension_claim": None,
        "policy": (
            "Bulk dimensionality is reported only as integer observer-chart "
            "verdicts plus mutual-agreement certificates. Continuous dimension "
            "estimators are internal diagnostics and never physical claims. A "
            "fractional bulk dimension is a category error under the OPH claim "
            "boundary: observers are primary, the bulk is their agreement."
        ),
        "claim_boundary": (
            "v1 certificate over a shared committed record with deterministic "
            "private observer frames. It certifies that truncated observer "
            "charts admit one mutual re-gauging with cocycle consistency, "
            "with shuffled-view controls. It is not a fundamental-bulk claim; "
            "the follow-on gate is genuinely independent per-observer commit "
            "histories measured by this same certificate."
        ),
        "pair_records": pair_records if full_records else pair_records[:64],
        "triple_records": triple_records if full_records else triple_records[:32],
    }


def write_observer_agreement_report(
    run_dir: str | Path,
    *,
    seed: int = 1,
    max_pairs: int = 512,
    max_triples: int = 128,
    min_overlap_edges: int = 8,
    max_observers: int | None = None,
) -> dict[str, Any]:
    report = observer_agreement_report(
        run_dir,
        seed=seed,
        max_pairs=max_pairs,
        max_triples=max_triples,
        min_overlap_edges=min_overlap_edges,
        max_observers=max_observers,
    )
    out_path = Path(run_dir) / "observer_agreement_report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


AGREEMENT_BULK_FIELD_SCHEMA = "agreement_bulk_field_v1"


def write_agreement_bulk_field(
    run_dir: str | Path,
    *,
    seed: int = 1,
    max_pairs: int = 800,
    max_triples: int = 400,
    min_overlap_edges: int = 8,
    max_observers: int | None = 2048,
) -> dict[str, Any]:
    """Per-patch agreement-multiplicity field for the emergent-bulk scene.

    OPH claim boundary rendered literally: a bulk region exists exactly to
    the extent that observers agree on it. For every patch this field
    counts (a) raw observer coverage (how many cohort observers hold the
    patch in support), (b) certified pair coverage (how many evaluated
    zero-defect re-gauged pairs contain it in their overlap), and (c)
    certified triple coverage (cocycle-closed triples). Coverage without
    certification is a subjective-overlap ghost layer; certified
    multiplicity is the agreement solidity the visualizer should render
    as bulk.
    """

    run = Path(run_dir)
    gauge_path = run / "s3_gauge_state.npz"
    if not gauge_path.exists():
        return {"schema": AGREEMENT_BULK_FIELD_SCHEMA, "status": "missing_s3_gauge_state"}
    points = np.asarray(np.load(gauge_path)["points"], dtype=float)
    patch_count = int(points.shape[0])

    report = observer_agreement_report(
        run,
        seed=seed,
        max_pairs=max_pairs,
        max_triples=max_triples,
        min_overlap_edges=min_overlap_edges,
        max_observers=max_observers,
        full_records=True,
    )
    if report.get("status") != "evaluated":
        return {"schema": AGREEMENT_BULK_FIELD_SCHEMA, "status": report.get("status")}

    observers = _load_patch_observers(run, max_observers)
    support_by_id = {
        int(observer["observer_id"]): set(observer["support_nodes"]) for observer in observers
    }
    coverage = np.zeros(patch_count, dtype=np.int32)
    for support in support_by_id.values():
        for node in support:
            if 0 <= node < patch_count:
                coverage[node] += 1

    pair_certified = np.zeros(patch_count, dtype=np.int32)
    certified_pairs = 0
    for record in report["pair_records"]:
        if record.get("defect") != 0.0:
            continue
        certified_pairs += 1
        overlap = support_by_id.get(int(record["observer_a"]), set()) & support_by_id.get(
            int(record["observer_b"]), set()
        )
        for node in overlap:
            if 0 <= node < patch_count:
                pair_certified[node] += 1

    triple_certified = np.zeros(patch_count, dtype=np.int32)
    certified_triples = 0
    for record in report["triple_records"]:
        if record.get("cocycle_defect") != 0.0:
            continue
        certified_triples += 1
        ids = [int(value) for value in record["observers"]]
        common = (
            support_by_id.get(ids[0], set())
            & support_by_id.get(ids[1], set())
            & support_by_id.get(ids[2], set())
        )
        for node in common:
            if 0 <= node < patch_count:
                triple_certified[node] += 1

    np.savez_compressed(
        run / "agreement_bulk_field.npz",
        points=points,
        coverage=coverage,
        pair_certified=pair_certified,
        triple_certified=triple_certified,
    )
    summary = {
        "schema": AGREEMENT_BULK_FIELD_SCHEMA,
        "status": "evaluated",
        "patch_count": patch_count,
        "cohort_observers": len(support_by_id),
        "certified_pairs_used": certified_pairs,
        "certified_triples_used": certified_triples,
        "covered_patch_fraction": float(np.mean(coverage > 0)),
        "pair_certified_patch_fraction": float(np.mean(pair_certified > 0)),
        "triple_certified_patch_fraction": float(np.mean(triple_certified > 0)),
        "max_pair_multiplicity": int(pair_certified.max()) if patch_count else 0,
        "claim_boundary": (
            "Sampled-certificate rendering field: certified multiplicities "
            "reflect the evaluated pair/triple sample, never a complete "
            "enumeration; zero certified multiplicity means untested, never "
            "disagreement. Disagreement would appear as nonzero pair defects "
            "in the agreement report itself."
        ),
    }
    (run / "agreement_bulk_field_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary
