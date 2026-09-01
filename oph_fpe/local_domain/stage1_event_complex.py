"""Stage 1: source event complex with a certified causal cover.

The local action domain starts from one target-clean source capture and
derives, without any target import, the object every downstream typing
stage consumes: the event complex with an exact causal-order
certificate, the reconstructed four-coordinate event-manifold candidate, a
cover of closed observer neighborhoods with
wiring transitions, and the prescribed-chart rank, fitted-form inertia,
orientation, and time-orientation certificates on that cover.  The
cover is a family of
closed finite sets with exact bookkeeping maps between local frames;
it is not an atlas of open coordinate charts, and no continuum
structure is asserted anywhere in this module.

Construction:

* Event classes, ancestry order, footprints, and the global prescribed
  four-coordinate feature chart are the finite event-manifold reconstruction: one
  coordinate is ancestry depth and exactly three coordinates are the
  seam-graph spectral embedding averaged over each event's carrier
  footprint.  The rank certificate checks nondegeneracy of this chosen
  chart; it does not select a spacetime dimension.  The held-out
  quadratic form supplies an inertia (1, 3) test on the prescribed
  chart, not a signature theorem about a manifold.
* The causal-order certificate is exact rather than fitted: every
  ancestry edge must advance the source sequence index and the ancestry
  depth, and reachability must be acyclic and antisymmetric.  The depth
  function is then a strict time function, which is the finite form of
  stable causality.
* The cover consists of closed Voronoi observer neighborhoods on the
  seam-visibility graph: two observers are adjacent when one overlap
  relation is visible to both.  Each neighborhood carries an affine
  local frame of the global reconstruction: depth recentred on the
  neighborhood and the spatial block recentred and rescaled by its own
  root-mean-square spread.  Transitions between neighborhoods are
  therefore exact affine maps computed from the frames and verified on
  shared events.  They are induced consistency checks of one prescribed
  global feature chart, not independently reconstructed observer charts.
  Their residual, cocycle, orientation, and
  time-orientation certificates are wiring certificates of the
  restriction cover.  The source-level transition content, the seam
  orientation signs and port transport of the overlap relations,
  enters at stage 2 as exact cocycle data.  Each neighborhood
  additionally records its own held-out quadratic fit, the local
  visibility surface of the fitted inertia.
* The continuum conditions are machine-readable and not attained at this
  finite cutoff: every measured cone margin is negative, one neighborhood
  fit has inertia (0, 4), the neighborhoods are closed finite sets,
  and no cofinal refinement family establishes convergence, so the
  finite domain does not promote to a continuum Lorentzian spacetime.

Fail-closed controls: a collapsed prescribed spatial axis, a
cross-read-free source with a different fitted feature form, a post-fit
coordinate flip (failed cocycle), a foreign-seed capture (mixed source),
and a config
carrying target vocabulary (target injection).  Each control must be
detected for the receipt to pass.  No output of this module carries
physical promotion.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.bulk.event_manifold_producer import (
    _event_table,
    _fit_quadratic_form,
    _spectral_embedding,
    _event_chart,
)
from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source

SCHEMA = "oph.local-domain-stage1.v1"
PHYSICAL_PROMOTION_ALLOWED = False

TARGET_INERTIA = (1, 3)
PAIR_CAP = 300_000
FULL_CHART_RANK_RATIO_FLOOR = 1.0e-8
MIN_CHART_EVENTS = 40
MIN_OVERLAP_EVENTS = 12
MIN_TRIPLE_EVENTS = 4
TRANSITION_RESIDUAL_GATE = 0.25
COCYCLE_DEFECT_GATE = 0.25
TIME_COEFF_FLOOR = 0.5

FORBIDDEN_CONFIG_KEYS = frozenset(
    {
        "expected_normalization",
        "target_signature",
        "target_coupling",
        "target_vacuum",
        "target_scale",
        "einstein_conclusion",
    }
)

MAIN_CONFIG: dict[str, Any] = {
    "carrier_count": 16_384,
    "cycles": 16,
    "seed": 20260751,
    "observer_count": 128,
    "observer_support_size": 96,
    "observer_samples": 6,
    "observer_cross_reads": True,
    "snapshot_coverage": "spanning",
    "geometry_transport": "held_out_flow",
}

CONTROL_SPLIT_CONFIG: dict[str, Any] = {
    "carrier_count": 2_048,
    "cycles": 16,
    "seed": 20260751,
    "observer_count": 16,
    "observer_support_size": 96,
    "observer_samples": 6,
    "observer_cross_reads": False,
    "snapshot_coverage": "spanning",
    "geometry_transport": "held_out_flow",
}

FOREIGN_SEED_OFFSET = 110


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _carrier_index(carrier_id: str) -> int:
    return int(carrier_id.rsplit("-", 1)[1])


_LOCAL_EVENT_KEYS = (
    "source_sequence_index",
    "observer_token",
    "visible_footprint",
)
_LOCAL_OVERLAP_KEYS = (
    "left_carrier_id",
    "right_carrier_id",
    "left_ports",
    "right_ports",
    "orientation_signs",
    "visible_to_observer_tokens",
    "interface_algebra_sha256",
)


def local_domain_source_projection(
    capture: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonical discrete source projection consumed by the local domain.

    The complete source capture also contains floating diagnostics whose
    final bits can vary with the Python/NumPy/SciPy runtime.  Stages 1--3
    consume only the event identities, ancestry, footprints, and seam
    transport rows below.  Their identity gate therefore hashes exactly
    this projection.  The complete capture hash remains a diagnostic and
    is deliberately not used as a cross-stage identity.
    """

    post = capture["postrun_capture"]
    event_count = len(post["semantic_events"])
    event_index = {
        row["event_key"]: index
        for index, row in enumerate(post["semantic_events"])
    }

    def endpoint(key: str) -> int | dict[str, str]:
        index = event_index.get(key)
        return index if index is not None else {"missing_event_key": key}

    return {
        "schema": "oph.local-domain-source-projection.v1",
        "declared_config": dict(config),
        "observed_carrier_count": len(post["carrier_port_trajectories"]),
        "event_key_census": {
            "event_count": event_count,
            "unique_event_key_count": len(event_index),
            "duplicate_event_key_count": event_count - len(event_index),
        },
        "semantic_events": [
            {key: row[key] for key in _LOCAL_EVENT_KEYS}
            for row in post["semantic_events"]
        ],
        "raw_ancestry_relations": [
            {
                "parent_event_index": endpoint(row["parent_event_id"]),
                "child_event_index": endpoint(row["child_event_id"]),
            }
            for row in post["raw_ancestry_relations"]
        ],
        "raw_overlap_relations": [
            {key: row[key] for key in _LOCAL_OVERLAP_KEYS}
            for row in post["raw_overlap_relations"]
        ],
    }


def local_domain_source_sha256(
    capture: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    """Hash the canonical source projection used by the local domain."""

    return _sha256_value(local_domain_source_projection(capture, config))


def refuse_forbidden_config(config: Mapping[str, Any]) -> list[str]:
    """Return the forbidden keys present in a config, sorted."""

    return sorted(FORBIDDEN_CONFIG_KEYS.intersection(config.keys()))


def build_event_complex(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Event table, ancestry edge list, and prescribed global feature chart."""

    post = capture["postrun_capture"]
    table = _event_table(capture)
    events = post["semantic_events"]
    key_to_index = {event["event_key"]: i for i, event in enumerate(events)}
    duplicate_event_key_count = len(events) - len(key_to_index)
    edges: list[tuple[int, int]] = []
    missing_ancestry_endpoint_count = 0
    for edge in post["raw_ancestry_relations"]:
        parent = key_to_index.get(edge["parent_event_id"])
        child = key_to_index.get(edge["child_event_id"])
        if parent is None or child is None:
            missing_ancestry_endpoint_count += 1
            continue
        edges.append((parent, child))
    sequence = [event["source_sequence_index"] for event in events]
    tokens = [event["observer_token"] for event in events]

    carrier_count = len(post["carrier_port_trajectories"])
    touched = sorted({c for footprint in table["footprints"] for c in footprint})
    index_of = {c: i for i, c in enumerate(touched)}
    sub = np.zeros((len(touched), len(touched)))
    for row in post["raw_overlap_relations"]:
        left = index_of.get(_carrier_index(row["left_carrier_id"]))
        right = index_of.get(_carrier_index(row["right_carrier_id"]))
        if left is not None and right is not None:
            sub[left, right] = sub[right, left] = 1.0
    embedding = np.zeros((carrier_count, 3))
    if len(touched) >= 4:
        small = _spectral_embedding(sub)
        for carrier, i in index_of.items():
            embedding[carrier] = small[i]
    chart = _event_chart(table, embedding)
    return {
        "table": table,
        "edges": edges,
        "sequence": sequence,
        "tokens": tokens,
        "chart": chart,
        "embedding": embedding,
        "touched_carriers": touched,
        "duplicate_event_key_count": duplicate_event_key_count,
        "missing_ancestry_endpoint_count": missing_ancestry_endpoint_count,
    }


def pair_classes_capped(table: Mapping[str, Any], cap: int = PAIR_CAP) -> dict:
    """Causal and spacelike pair classes with the declared stride cap."""

    causal: list[tuple[int, int]] = []
    spacelike: list[tuple[int, int]] = []
    count = table["count"]
    reachable = table["reachable"]
    for i in range(count):
        reach_i = reachable[i]
        for j in range(i + 1, count):
            if i in reachable[j] or j in reach_i:
                causal.append((i, j))
            else:
                spacelike.append((i, j))
    out: dict[str, Any] = {}
    for name, pairs in (("causal", causal), ("spacelike", spacelike)):
        stride = max(1, len(pairs) // cap)
        out[name] = pairs[::stride][:cap]
        out[f"{name}_total"] = len(pairs)
        out[f"{name}_stride"] = stride
    return out


def causal_certificates(complex_data: Mapping[str, Any]) -> dict[str, Any]:
    """Exact acyclicity, antisymmetry, and strict-time-function checks."""

    table = complex_data["table"]
    depth = table["depth"]
    reachable = table["reachable"]
    sequence = complex_data["sequence"]
    edges = complex_data["edges"]
    count = table["count"]

    edge_sequence_strict = all(
        sequence[parent] < sequence[child] for parent, child in edges
    )
    edge_depth_strict = all(
        depth[parent] < depth[child] for parent, child in edges
    )
    acyclic = all(i not in reachable[i] for i in range(count))
    antisymmetric = True
    for i in range(count):
        for j in reachable[i]:
            if i in reachable[j]:
                antisymmetric = False
                break
        if not antisymmetric:
            break
    return {
        "edge_count": len(edges),
        "event_keys_unique": bool(
            complex_data.get("duplicate_event_key_count", 0) == 0
        ),
        "ancestry_endpoints_resolved": bool(
            complex_data.get("missing_ancestry_endpoint_count", 0) == 0
        ),
        "edge_sequence_strict": bool(edge_sequence_strict),
        "edge_depth_strict": bool(edge_depth_strict),
        "acyclic": bool(acyclic),
        "antisymmetric": bool(antisymmetric),
        "strict_time_function": bool(edge_depth_strict and acyclic),
    }


def prescribed_chart_rank_certificate(
    chart: np.ndarray,
) -> dict[str, Any]:
    """Nondegeneracy test for the prescribed four-coordinate chart."""

    coordinate_count = int(chart.shape[1]) if chart.ndim == 2 else 0
    if chart.ndim != 2 or coordinate_count != 4 or len(chart) == 0:
        return {
            "coordinate_count": coordinate_count,
            "time_spread": 0.0,
            "full_chart_singulars": [],
            "full_chart_rank_ratio": 0.0,
            "full_chart_rank_ratio_floor": (
                FULL_CHART_RANK_RATIO_FLOOR
            ),
            "spatial_singulars": [],
            "spatial_rank_ratio": 0.0,
            "prescribed_four_coordinate_chart_nondegenerate": False,
        }
    time_spread = float(chart[:, 0].std())
    centered = chart - chart.mean(axis=0)
    full_singulars = np.linalg.svd(
        centered / np.sqrt(len(chart)), compute_uv=False
    )
    full_rank_ratio = (
        float(full_singulars[-1] / full_singulars[0])
        if full_singulars[0] > 0
        else 0.0
    )
    spatial = chart[:, 1:] - chart[:, 1:].mean(axis=0)
    singulars = np.linalg.svd(spatial / np.sqrt(len(chart)), compute_uv=False)
    ratio = float(singulars[2] / singulars[0]) if singulars[0] > 0 else 0.0
    return {
        "coordinate_count": coordinate_count,
        "time_spread": time_spread,
        "full_chart_singulars": [float(s) for s in full_singulars],
        "full_chart_rank_ratio": full_rank_ratio,
        "full_chart_rank_ratio_floor": FULL_CHART_RANK_RATIO_FLOOR,
        "spatial_singulars": [float(s) for s in singulars],
        "spatial_rank_ratio": ratio,
        "prescribed_four_coordinate_chart_nondegenerate": bool(
            full_rank_ratio >= FULL_CHART_RANK_RATIO_FLOOR
        ),
    }


def observer_visibility_graph(
    capture: Mapping[str, Any],
) -> tuple[list[str], np.ndarray]:
    """Observer adjacency from shared seam visibility."""

    post = capture["postrun_capture"]
    tokens = sorted({e["observer_token"] for e in post["semantic_events"]})
    index = {t: i for i, t in enumerate(tokens)}
    adjacency = np.zeros((len(tokens), len(tokens)), dtype=bool)
    for row in post["raw_overlap_relations"]:
        visible = [index[t] for t in row["visible_to_observer_tokens"] if t in index]
        for a in range(len(visible)):
            for b in range(a + 1, len(visible)):
                adjacency[visible[a], visible[b]] = True
                adjacency[visible[b], visible[a]] = True
    return tokens, adjacency


def _bfs_distances(adjacency: np.ndarray, start: int) -> np.ndarray:
    unreached = adjacency.shape[0] + 1
    distance = np.full(adjacency.shape[0], unreached, dtype=int)
    distance[start] = 0
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in np.nonzero(adjacency[u])[0]:
            if distance[v] > distance[u] + 1:
                distance[v] = distance[u] + 1
                queue.append(v)
    return distance


def blockwise_residual(
    predicted: np.ndarray, actual: np.ndarray
) -> dict[str, float]:
    """Time and spatial residuals, each normalized by its own block scale.

    The time coordinate carries the ancestry-depth scale and the spatial
    coordinates carry the spectral-embedding scale; a joint norm would
    let the larger block mask defects in the smaller one."""

    out: dict[str, float] = {}
    for name, columns in (("time", slice(0, 1)), ("spatial", slice(1, 4))):
        block = actual[:, columns]
        centred = block - block.mean(axis=0)
        scale = float(np.sqrt((centred**2).mean()))
        residual = float(
            np.sqrt(((predicted[:, columns] - block) ** 2).mean())
        )
        out[name] = residual / scale if scale > 0 else float("inf")
    out["max"] = max(out["time"], out["spatial"])
    return out


def chart_frame(
    chart: np.ndarray, event_indices: list[int]
) -> dict[str, Any]:
    """Affine local frame: recentred depth, recentred and rescaled space."""

    block = chart[event_indices]
    time_mean = float(block[:, 0].mean()) if event_indices else 0.0
    spatial_mean = (
        block[:, 1:].mean(axis=0) if event_indices else np.zeros(3)
    )
    centred = block[:, 1:] - spatial_mean if event_indices else np.zeros((0, 3))
    spatial_rms = float(np.sqrt((centred**2).mean())) if event_indices else 0.0
    scale = spatial_rms if spatial_rms > 0 else 1.0
    return {
        "time_mean": time_mean,
        "spatial_mean": spatial_mean,
        "spatial_scale": scale,
        "degenerate_spatial": bool(spatial_rms == 0.0),
    }


def apply_frame(chart: np.ndarray, frame: Mapping[str, Any]) -> np.ndarray:
    coords = chart.copy()
    coords[:, 0] -= frame["time_mean"]
    coords[:, 1:] = (coords[:, 1:] - frame["spatial_mean"]) / frame[
        "spatial_scale"
    ]
    return coords


def frame_transition(
    frame_a: Mapping[str, Any], frame_b: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Exact affine map from frame-a to frame-b coordinates."""

    ratio = frame_a["spatial_scale"] / frame_b["spatial_scale"]
    matrix = np.diag([1.0, ratio, ratio, ratio])
    offset = np.zeros(4)
    offset[0] = frame_a["time_mean"] - frame_b["time_mean"]
    offset[1:] = (frame_a["spatial_mean"] - frame_b["spatial_mean"]) / frame_b[
        "spatial_scale"
    ]
    return matrix, offset


def verify_transition(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
) -> dict[str, Any]:
    """Evaluate the frame transition on shared events, blockwise."""

    count = len(coords_a)
    spatial = coords_a[:, 1:] - coords_a[:, 1:].mean(axis=0)
    singulars = np.linalg.svd(
        spatial / np.sqrt(max(count, 1)), compute_uv=False
    )
    cond = float(singulars[2] / singulars[0]) if singulars[0] > 0 else 0.0
    predicted = coords_a @ matrix + offset
    residual = blockwise_residual(predicted, coords_b)
    supported = bool(
        count >= MIN_OVERLAP_EVENTS
        and residual["max"] <= TRANSITION_RESIDUAL_GATE
    )
    return {
        "event_count": count,
        "spatial_cond": cond,
        "matrix": matrix,
        "offset": offset,
        "residual_rel": residual["max"],
        "residual_time": residual["time"],
        "residual_spatial": residual["spatial"],
        "supported": supported,
    }


def build_atlas(
    capture: Mapping[str, Any],
    complex_data: Mapping[str, Any],
    seed_count: int = 6,
) -> dict[str, Any]:
    """Closed Voronoi observer charts framed on the global reconstruction."""

    tokens, adjacency = observer_visibility_graph(capture)
    nobs = len(tokens)
    count = complex_data["table"]["count"]
    event_tokens = complex_data["tokens"]

    seeds = [min(k * nobs // seed_count, nobs - 1) for k in range(seed_count)]
    distances = np.stack([_bfs_distances(adjacency, s) for s in seeds])
    cell = np.argmin(distances, axis=0)
    unreached = int((distances.min(axis=0) > nobs).sum())

    chart_observers: list[set[int]] = []
    for k in range(seed_count):
        members = set(np.nonzero(cell == k)[0].tolist())
        closure = set(members)
        for u in members:
            closure.update(np.nonzero(adjacency[u])[0].tolist())
        chart_observers.append(closure)

    chart_events: list[list[int]] = []
    for k in range(seed_count):
        member_tokens = {tokens[i] for i in chart_observers[k]}
        chart_events.append(
            [i for i in range(count) if event_tokens[i] in member_tokens]
        )

    global_chart = complex_data["chart"]
    charts: list[dict[str, Any]] = []
    for k in range(seed_count):
        events = chart_events[k]
        frame = chart_frame(global_chart, events)
        coords = apply_frame(global_chart, frame)
        local_pairs = {"causal": [], "spacelike": []}
        reachable = complex_data["table"]["reachable"]
        event_set = events
        for a_pos in range(len(event_set)):
            i = event_set[a_pos]
            for b_pos in range(a_pos + 1, len(event_set)):
                j = event_set[b_pos]
                if i in reachable[j] or j in reachable[i]:
                    local_pairs["causal"].append((i, j))
                else:
                    local_pairs["spacelike"].append((i, j))
        local_fit = _fit_quadratic_form(coords, local_pairs)
        charts.append(
            {
                "chart_id": k,
                "seed_observer": tokens[seeds[k]],
                "observer_count": len(chart_observers[k]),
                "event_indices": events,
                "frame": frame,
                "coords": coords,
                "degenerate_spatial": frame["degenerate_spatial"],
                "local_causal_pairs": len(local_pairs["causal"]),
                "local_spacelike_pairs": len(local_pairs["spacelike"]),
                "local_fit": {
                    key: local_fit.get(key)
                    for key in ("fitted", "inertia", "cone_margin")
                },
            }
        )

    overlaps: list[dict[str, Any]] = []
    for a in range(seed_count):
        events_a = set(charts[a]["event_indices"])
        for b in range(a + 1, seed_count):
            shared = sorted(events_a.intersection(charts[b]["event_indices"]))
            if not shared:
                continue
            matrix, offset = frame_transition(
                charts[a]["frame"], charts[b]["frame"]
            )
            fit = verify_transition(
                charts[a]["coords"][shared],
                charts[b]["coords"][shared],
                matrix,
                offset,
            )
            overlaps.append({"pair": (a, b), "events": shared, "fit": fit})

    supported_edges = [o["pair"] for o in overlaps if o["fit"]["supported"]]
    component = list(range(seed_count))

    def find(x: int) -> int:
        while component[x] != x:
            component[x] = component[component[x]]
            x = component[x]
        return x

    for a, b in supported_edges:
        component[find(a)] = find(b)
    nerve_connected = len({find(k) for k in range(seed_count)}) == 1

    transition_of = {
        tuple(o["pair"]): o for o in overlaps if o["fit"]["supported"]
    }
    triples: list[dict[str, Any]] = []
    for a in range(seed_count):
        for b in range(a + 1, seed_count):
            for c in range(b + 1, seed_count):
                if (
                    (a, b) not in transition_of
                    or (b, c) not in transition_of
                    or (a, c) not in transition_of
                ):
                    continue
                common = sorted(
                    set(transition_of[(a, b)]["events"])
                    & set(transition_of[(b, c)]["events"])
                    & set(transition_of[(a, c)]["events"])
                )
                if len(common) < MIN_TRIPLE_EVENTS:
                    continue
                x_a = charts[a]["coords"][common]
                fit_ab = transition_of[(a, b)]["fit"]
                fit_bc = transition_of[(b, c)]["fit"]
                fit_ac = transition_of[(a, c)]["fit"]
                via_b = (x_a @ fit_ab["matrix"] + fit_ab["offset"]) @ fit_bc[
                    "matrix"
                ] + fit_bc["offset"]
                direct = x_a @ fit_ac["matrix"] + fit_ac["offset"]
                defect_rel = blockwise_residual(via_b, direct)["max"]
                triples.append(
                    {
                        "triple": (a, b, c),
                        "event_count": len(common),
                        "defect_rel": defect_rel,
                        "within_gate": bool(defect_rel <= COCYCLE_DEFECT_GATE),
                    }
                )

    return {
        "seed_count": seed_count,
        "observer_tokens": tokens,
        "unreached_observers": unreached,
        "charts": charts,
        "overlaps": overlaps,
        "supported_edges": supported_edges,
        "nerve_connected": bool(nerve_connected),
        "triples": triples,
    }


def orientation_certificate(atlas: Mapping[str, Any]) -> dict[str, Any]:
    """Chart sign assignment from full transition-Jacobian determinants.

    The determinant of the full four-by-four affine part is
    multiplicative under composition, so a consistent cocycle forces a
    consistent sign assignment; an odd negative cycle is the exact
    orientability obstruction on the nerve."""

    seed_count = atlas["seed_count"]
    edge_signs: dict[tuple[int, int], int] = {}
    for overlap in atlas["overlaps"]:
        if not overlap["fit"]["supported"]:
            continue
        det = float(np.linalg.det(overlap["fit"]["matrix"]))
        if det == 0.0:
            return {
                "orientable": False,
                "blocker": "singular_transition_jacobian",
                "edge_signs": {},
            }
        edge_signs[tuple(overlap["pair"])] = 1 if det > 0 else -1

    assignment = [0] * seed_count
    orientable = True
    for start in range(seed_count):
        if assignment[start] != 0:
            continue
        assignment[start] = 1
        queue = deque([start])
        while queue and orientable:
            u = queue.popleft()
            for (a, b), sign in edge_signs.items():
                other = b if a == u else a if b == u else None
                if other is None:
                    continue
                expected = assignment[u] * sign
                if assignment[other] == 0:
                    assignment[other] = expected
                    queue.append(other)
                elif assignment[other] != expected:
                    orientable = False
                    break
    return {
        "orientable": bool(orientable),
        "edge_signs": {f"{a}-{b}": s for (a, b), s in sorted(edge_signs.items())},
        "chart_signs": assignment if orientable else [],
    }


def time_orientation_certificate(atlas: Mapping[str, Any]) -> dict[str, Any]:
    """Positive time-time coefficient on every supported transition."""

    coefficients = {}
    consistent = True
    for overlap in atlas["overlaps"]:
        if not overlap["fit"]["supported"]:
            continue
        coefficient = float(overlap["fit"]["matrix"][0, 0])
        coefficients[f"{overlap['pair'][0]}-{overlap['pair'][1]}"] = coefficient
        if coefficient < TIME_COEFF_FLOOR:
            consistent = False
    return {
        "time_orientable": bool(consistent),
        "time_coefficients": coefficients,
        "floor": TIME_COEFF_FLOOR,
    }


def _footprint_freeze(table: Mapping[str, Any]) -> str:
    return _sha256_value(
        sorted(map(sorted, (list(f) for f in table["footprints"])))
    )


def produce_stage1_receipt(
    *,
    config: Mapping[str, Any] | None = None,
    seed_count: int = 6,
    output_dir: str | Path | None = None,
    run_controls: bool = True,
) -> dict[str, Any]:
    """Produce the stage-1 event-complex receipt with fail-closed controls."""

    main_config = dict(MAIN_CONFIG if config is None else config)
    forbidden = refuse_forbidden_config(main_config)
    if forbidden:
        return {
            "schema": SCHEMA,
            "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
            "verdict": "REFUSED",
            "blockers": [f"forbidden_config_key:{key}" for key in forbidden],
        }

    capture = capture_physical_source(main_config)
    source_projection_sha256 = local_domain_source_sha256(capture, main_config)
    complex_data = build_event_complex(capture)
    table = complex_data["table"]
    chart = complex_data["chart"]

    causal = causal_certificates(complex_data)
    chart_rank = prescribed_chart_rank_certificate(chart)
    pairs = pair_classes_capped(table)
    fit = _fit_quadratic_form(
        chart, {"causal": pairs["causal"], "spacelike": pairs["spacelike"]}
    )
    inertia_ok = bool(
        fit.get("fitted") and tuple(fit.get("inertia", ())) == TARGET_INERTIA
    )

    atlas = build_atlas(capture, complex_data, seed_count=seed_count)
    orientation = orientation_certificate(atlas)
    time_orientation = time_orientation_certificate(atlas)

    chart_sizes = [len(c["event_indices"]) for c in atlas["charts"]]
    covered = set()
    for c in atlas["charts"]:
        covered.update(c["event_indices"])
    atlas_covers = bool(
        len(covered) == table["count"]
        and all(size >= MIN_CHART_EVENTS for size in chart_sizes)
        and atlas["unreached_observers"] == 0
    )
    transitions_supported = bool(
        atlas["supported_edges"]
        and all(
            o["fit"]["supported"]
            for o in atlas["overlaps"]
            if tuple(o["pair"]) in set(map(tuple, atlas["supported_edges"]))
        )
    )
    triples_checked = [t for t in atlas["triples"]]
    cocycles_ok = bool(
        triples_checked and all(t["within_gate"] for t in triples_checked)
    )

    controls: dict[str, dict[str, Any]] = {}
    if run_controls:
        collapsed = chart.copy()
        collapsed[:, 3] = 0.0
        collapsed_cert = prescribed_chart_rank_certificate(collapsed)
        controls["collapsed_prescribed_spatial_axis"] = {
            "control_failure_detected": bool(
                not collapsed_cert[
                    "prescribed_four_coordinate_chart_nondegenerate"
                ]
            ),
            "collapsed_rank_ratio": collapsed_cert["spatial_rank_ratio"],
        }

        split_capture = capture_physical_source(dict(CONTROL_SPLIT_CONFIG))
        split_complex = build_event_complex(split_capture)
        split_pairs = pair_classes_capped(split_complex["table"])
        split_fit = _fit_quadratic_form(
            split_complex["chart"],
            {"causal": split_pairs["causal"], "spacelike": split_pairs["spacelike"]},
        )
        split_cross = sum(
            1
            for parent, child in split_complex["edges"]
            if split_complex["tokens"][parent] != split_complex["tokens"][child]
        )
        controls["split_source_feature_form"] = {
            "control_failure_detected": bool(
                split_cross == 0
                and tuple(split_fit.get("inertia", ())) != TARGET_INERTIA
            ),
            "control_inertia": split_fit.get("inertia"),
            "control_cross_edges": split_cross,
        }

        flip_detected = False
        flipped_offender = None
        for overlap in atlas["overlaps"]:
            if not overlap["fit"]["supported"]:
                continue
            a, b = overlap["pair"]
            shared = overlap["events"]
            coords_a = atlas["charts"][a]["coords"][shared].copy()
            coords_a[:, 1] = -coords_a[:, 1]
            predicted = coords_a @ overlap["fit"]["matrix"] + overlap["fit"]["offset"]
            actual = atlas["charts"][b]["coords"][shared]
            residual_rel = blockwise_residual(predicted, actual)["max"]
            if residual_rel > TRANSITION_RESIDUAL_GATE:
                flip_detected = True
                flipped_offender = f"{a}-{b}"
                break
        controls["failed_cocycle"] = {
            "control_failure_detected": bool(flip_detected),
            "detected_on_overlap": flipped_offender,
        }

        foreign_config = dict(main_config)
        foreign_config["seed"] = int(main_config["seed"]) + FOREIGN_SEED_OFFSET
        foreign_capture = capture_physical_source(foreign_config)
        foreign_table = _event_table(foreign_capture)
        controls["mixed_source"] = {
            "control_failure_detected": bool(
                _footprint_freeze(foreign_table) != _footprint_freeze(table)
            )
        }

        injected = dict(main_config)
        injected["target_signature"] = "lorentzian"
        refusal = produce_stage1_receipt(
            config=injected, seed_count=seed_count, run_controls=False
        )
        controls["target_injection"] = {
            "control_failure_detected": bool(refusal.get("verdict") == "REFUSED")
        }

    controls_fail_closed = bool(controls) and all(
        row["control_failure_detected"] for row in controls.values()
    )

    clause_verdicts = {
        "event_classes_and_order_constructed": True,
        "source_relation_identifiers_well_formed": bool(
            causal["event_keys_unique"]
            and causal["ancestry_endpoints_resolved"]
        ),
        "causal_order_acyclic": causal["acyclic"] and causal["antisymmetric"],
        "strict_time_function": causal["strict_time_function"],
        "prescribed_four_coordinate_chart_nondegenerate": chart_rank[
            "prescribed_four_coordinate_chart_nondegenerate"
        ],
        "held_out_feature_form_inertia_1_3": inertia_ok,
        "atlas_covers_all_events": atlas_covers,
        "atlas_nerve_connected": atlas["nerve_connected"],
        "transitions_supported": transitions_supported,
        "cocycles_within_gate": cocycles_ok,
        "orientation_assignment_exists": orientation["orientable"],
        "time_orientation_consistent": time_orientation["time_orientable"],
    }
    blockers = sorted(
        f"clause_failed:{name}"
        for name, verdict in clause_verdicts.items()
        if not verdict
    )
    if run_controls and not controls_fail_closed:
        blockers.append("negative_control_did_not_fail")
    verdict = "ATTAINED" if not blockers else "NOT_ATTAINED"

    arrays = {
        "chart": chart,
        # The capped pair arrays are fit/sensitivity samples. Preserve the
        # exact generated Hasse-input relation separately so consumers can
        # reconstruct the full finite order without mistaking a stride sample
        # for the poset.
        "direct_ancestry_edges": np.asarray(
            complex_data["edges"], dtype=np.int64
        ),
        "causal_pairs": np.asarray(pairs["causal"], dtype=np.int64),
        "spacelike_pairs": np.asarray(
            pairs["spacelike"], dtype=np.int64
        ),
    }
    for k, chart_row in enumerate(atlas["charts"]):
        arrays[f"chart_coords_{k}"] = chart_row["coords"]
        arrays[f"chart_events_{k}"] = np.asarray(
            chart_row["event_indices"], dtype=np.int64
        )
    array_specs = {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "value_sha256": _sha256_value(array.tolist()),
        }
        for name, array in sorted(arrays.items())
    }

    receipt = {
        "schema": SCHEMA,
        "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
        "main_config": main_config,
        "capture_sha256": capture["capture_sha256"],
        "capture_sha256_role": (
            "environment-sensitive full-capture diagnostic; not an "
            "identity gate for the local-domain stages"
        ),
        "source_projection_sha256": source_projection_sha256,
        "declared_acceptance_gates": {
            "prescribed_chart_full_rank_ratio_floor": (
                FULL_CHART_RANK_RATIO_FLOOR
            ),
            "held_out_feature_form_target_inertia": list(TARGET_INERTIA),
            "status": (
                "declared tests on a prescribed feature chart; not "
                "dimension or signature selection"
            ),
        },
        "event_count": table["count"],
        "ancestry_edge_count": causal["edge_count"],
        "direct_ancestry_array_name": "direct_ancestry_edges",
        "causal_pair_total": pairs["causal_total"],
        "spacelike_pair_total": pairs["spacelike_total"],
        "pair_cap": PAIR_CAP,
        "chart_freeze_sha256": _sha256_value(chart.tolist()),
        "array_bundle_binding": {
            "schema": "oph.local-domain-stage1-arrays.v1",
            "array_names": sorted(arrays),
            "array_specs": array_specs,
        },
        "causal_certificates": causal,
        "prescribed_chart_construction": {
            "ancestry_depth_coordinate_count": 1,
            "spectral_embedding_coordinate_count": 3,
            "total_coordinate_count": 4,
            "status": (
                "prescribed feature construction; no dimension-selection "
                "claim"
            ),
        },
        "prescribed_chart_rank_certificate": chart_rank,
        "cover_transition_scope": {
            "construction": (
                "affine recentering and rescaling of one prescribed "
                "global feature chart"
            ),
            "independent_observer_chart_reconstruction": False,
            "certificate_scope": (
                "induced transition consistency on a closed-neighborhood "
                "cover"
            ),
        },
        "held_out_quadratic_fit": {
            key: fit.get(key)
            for key in ("fitted", "inertia", "cone_margin", "held_out_pair_count")
        },
        "continuum_conditions": {
            "status": "NOT_ATTAINED_AT_CURRENT_FINITE_CUTOFF",
            "conditions": {
                "cone_margins_nonnegative": {
                    "holds": bool(
                        fit.get("fitted")
                        and float(fit.get("cone_margin", -1.0)) >= 0.0
                    ),
                    "measured_global_margin": fit.get("cone_margin"),
                },
                "all_neighborhood_fits_lorentzian": {
                    "holds": all(
                        tuple(c["local_fit"].get("inertia") or ())
                        == TARGET_INERTIA
                        for c in atlas["charts"]
                        if c["local_fit"].get("fitted")
                    ),
                    "measured_inertias": [
                        list(c["local_fit"].get("inertia") or [])
                        for c in atlas["charts"]
                    ],
                },
                "open_chart_topology": {
                    "holds": False,
                    "note": (
                        "the cover is a family of closed finite observer "
                        "neighborhoods with wiring transitions, not open "
                        "coordinate charts"
                    ),
                },
                "cofinal_refinement_family": {
                    "holds": False,
                    "note": (
                        "no certified cofinal refinement family or convergence "
                        "certificate is supplied by this finite receipt"
                    ),
                },
                "invariant_continuum_metric": {
                    "holds": False,
                    "note": (
                        "the held-out inertia is a fitted-form result about "
                        "this finite instrument, not a manifold signature "
                        "theorem; no differentiable structure or physical "
                        "light cone is produced"
                    ),
                },
            },
            "consumer_rule": (
                "downstream consumers must treat this receipt as a finite "
                "causal and local-operator domain and may not assume a "
                "continuum Lorentzian spacetime"
            ),
            "interpretation": (
                "The named continuum conditions are not met by this finite "
                "object at this cutoff. This is not a nonexistence result."
            ),
        },
        "atlas": {
            "seed_count": atlas["seed_count"],
            "covered_event_count": len(covered),
            "chart_event_counts": chart_sizes,
            "chart_observer_counts": [
                c["observer_count"] for c in atlas["charts"]
            ],
            "chart_local_fits": [
                {
                    "chart_id": c["chart_id"],
                    "causal_pairs": c["local_causal_pairs"],
                    "spacelike_pairs": c["local_spacelike_pairs"],
                    **c["local_fit"],
                }
                for c in atlas["charts"]
            ],
            "unreached_observers": atlas["unreached_observers"],
            "nerve_connected": atlas["nerve_connected"],
            "supported_edges": [list(e) for e in atlas["supported_edges"]],
            "transition_residuals": {
                f"{o['pair'][0]}-{o['pair'][1]}": round(
                    o["fit"]["residual_rel"], 6
                )
                for o in atlas["overlaps"]
            },
            "triple_defects": {
                "-".join(map(str, t["triple"])): round(t["defect_rel"], 6)
                for t in atlas["triples"]
            },
            "atlas_freeze_sha256": _sha256_value(
                [c["coords"].tolist() for c in atlas["charts"]]
            ),
        },
        "orientation_certificate": orientation,
        "time_orientation_certificate": time_orientation,
        "clause_verdicts": clause_verdicts,
        "negative_controls": controls,
        "controls_fail_closed": controls_fail_closed,
        "verdict": verdict,
        "STAGE1_EVENT_COMPLEX_RECEIPT": bool(verdict == "ATTAINED"),
        "blockers": blockers,
        "claim_boundary": (
            "The frozen source capture inhabits a finite causal and "
            "local-operator domain. The feature chart is prescribed as "
            "one ancestry-depth coordinate plus three spectral embedding "
            "coordinates; its nondegeneracy and fitted inertia do not "
            "select a spacetime dimension. Its global fitted form has "
            "inertia (1, 3), a held-out feature-form result about this "
            "finite instrument rather than a signature theorem about a "
            "manifold. The "
            "negative cone margins, one neighborhood fit with inertia "
            "(0, 4), the closed finite neighborhoods, and the missing "
            "refinement limit prevent promotion to a continuum Lorentzian "
            "spacetime; the continuum_conditions record the current finite "
            "non-satisfaction machine-readably. The transition, "
            "cocycle, orientation, and time-orientation certificates are "
            "induced wiring checks of recentered and rescaled copies of "
            "that same global chart on a closed-neighborhood cover. They "
            "are not independent observer-chart reconstructions and are "
            "guarded by the coordinate-flip control. No physical promotion "
            "follows from any output."
        ),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        receipt_bytes = _canonical_json(receipt).encode("utf-8")
        (out / "stage1_receipt.json").write_bytes(receipt_bytes)
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        compressed = gzip.compress(buffer.getvalue(), mtime=0)
        (out / "stage1_arrays.npz.gz").write_bytes(compressed)
        manifest_path = out / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"schema": "oph.local-domain-stage1.manifest.v1"}
        )
        manifest["receipt"] = "stage1_receipt.json"
        manifest["receipt_sha256"] = (
            "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()
        )
        manifest["arrays"] = "stage1_arrays.npz.gz"
        manifest["arrays_sha256"] = (
            "sha256:" + hashlib.sha256(compressed).hexdigest()
        )
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=1) + "\n",
            encoding="utf-8",
        )
    return receipt
