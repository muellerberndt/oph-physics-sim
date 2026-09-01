"""Independent verifier for the source-causal history-family receipt.

The history-family producer and capture module are not imported.  Semantic
events and provenance edges are reconstructed separately from each embedded
raw cutoff log by the independent source-order implementation.  The verifier
then recomputes each complete carrier, direct and transitive order, exact-width
certificate, adjacent induced restriction, scaling row, and the prescribed
shared-frame ``R x rank3`` diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from oph_fpe.bulk.verify_source_derived_causal_order_independent import (
    IndependentVerificationError,
    _checkpoint_audit,
    _edge_projection,
    _reconstruct_semantic_projection,
)
from oph_fpe.core.echosahedral_federation import reference_echosahedral_carrier


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = (
    ROOT / "data" / "causal_order" / "source_causal_history_family_receipt.json"
)
DEFAULT_PUBLICATION_PROJECTION = (
    ROOT
    / "data"
    / "causal_order"
    / "source_causal_history_family_publication_projection.json"
)
SOURCE_RECEIPT = (
    ROOT / "data" / "causal_order" / "source_derived_causal_order_receipt.json"
)
EXPECTED_SCHEMA = "oph.source-causal-history-family.v1"
EXPECTED_STATUS = (
    "CERTIFIED_INFORMATIONAL_HISTORY_EXTENSION_FAMILY__"
    "FIXED_WIDTH_NOT_SPACETIME_REFINEMENT"
)
EXPECTED_CUTOFFS = (4, 8, 16, 32, 64)
EXPECTED_PUBLICATION_SCHEMA = (
    "oph.source-causal-history-family-publication-projection.v1"
)
EXPECTED_ARTIFACT_TYPE = (
    "SOURCE_DERIVED_INDEPENDENT_INFORMATIONAL_HISTORY_EXTENSION_FAMILY"
)
EXPECTED_REQUIRED_NEXT_STEP = (
    "Construct a source-selected regulator family that increases local "
    "event density and spatial antichain capacity on comparable physical "
    "regions, eventize signal-capable repair/propagation links, and freeze "
    "multi-statistic Minkowski/FLRW comparisons before evaluating it."
)
EXPECTED_CLAIM_BOUNDARY = (
    "Exact positive custody result for independently executed complete-round "
    "cutoffs of one deterministic two-observer source configuration. Each "
    "cutoff order is reconstructed from its own embedded raw observer log and "
    "authenticated read-after-write provenance; adjacent independently "
    "generated direct and transitive orders restrict exactly. The two observer "
    "chains cover each carrier and two same-round records witness width exactly two. "
    "This family changes elapsed record history, not regulator resolution "
    "or event density in a fixed physical region. It therefore supplies no "
    "physical causal attachment, faithful embedding, manifoldlikeness, "
    "dimension, topology, count-volume density, Lorentzian metric, or "
    "continuum-limit result."
)


class IndependentHistoryFamilyVerificationError(RuntimeError):
    """Raised when any independently replayed clause disagrees."""


def _fail(message: str) -> None:
    raise IndependentHistoryFamilyVerificationError(message)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_sha(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith(
        "sha256:"
    ):
        _fail(f"{label} is not a strict SHA-256 identifier")
    try:
        int(value[7:], 16)
    except ValueError:
        _fail(f"{label} is not a strict SHA-256 identifier")
    return value


def _sample(event: Mapping[str, Any]) -> int:
    payload = event.get("canonical_semantic_payload")
    if not isinstance(payload, Mapping) or type(payload.get("sample")) is not int:
        _fail("semantic event lacks an exact round")
    return int(payload["sample"])


def _transitive_data(
    nodes: Iterable[str], edges: Iterable[tuple[str, str]]
) -> tuple[list[str], list[set[int]], list[set[int]], list[int]]:
    ordered = sorted(set(str(node) for node in nodes))
    if not ordered:
        _fail("empty prefix carrier")
    index = {node: position for position, node in enumerate(ordered)}
    children = [set() for _ in ordered]
    indegree = [0 for _ in ordered]
    for parent_key, child_key in set(edges):
        parent = index.get(parent_key)
        child = index.get(child_key)
        if parent is None or child is None or parent == child:
            _fail("relation is not strict on its declared carrier")
        if child not in children[parent]:
            children[parent].add(child)
            indegree[child] += 1
    frontier = sorted(i for i, degree in enumerate(indegree) if degree == 0)
    order: list[int] = []
    work = list(indegree)
    while frontier:
        node = frontier.pop()
        order.append(node)
        for child in sorted(children[node]):
            work[child] -= 1
            if work[child] == 0:
                frontier.append(child)
    if len(order) != len(ordered):
        _fail("prefix relation is cyclic")
    ancestors = [set() for _ in ordered]
    descendants = [set() for _ in ordered]
    heights = [1 for _ in ordered]
    for node in order:
        for child in children[node]:
            ancestors[child].add(node)
            ancestors[child].update(ancestors[node])
            heights[child] = max(heights[child], heights[node] + 1)
    for node in reversed(order):
        for child in children[node]:
            descendants[node].add(child)
            descendants[node].update(descendants[child])
    return ordered, ancestors, descendants, heights


def _closure_pairs(
    ordered: list[str], ancestors: list[set[int]]
) -> set[tuple[str, str]]:
    return {
        (ordered[parent], ordered[child])
        for child, parents in enumerate(ancestors)
        for parent in parents
    }


def _prefix_summary(
    events: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    cutoff: int,
    *,
    require_exact_cutoff_run: bool = False,
) -> tuple[dict[str, Any], set[str], set[tuple[str, str]], set[tuple[str, str]]]:
    selected_events = [event for event in events if _sample(event) < cutoff]
    carrier = {str(event["event_key"]) for event in selected_events}
    selected_edges = [
        row
        for row in edges
        if str(row["parent_event_id"]) in carrier
        and str(row["child_event_id"]) in carrier
    ]
    if require_exact_cutoff_run and (
        len(selected_events) != len(events) or len(selected_edges) != len(edges)
    ):
        _fail("cutoff level contains events or edges outside its own complete run")
    direct = {
        (str(row["parent_event_id"]), str(row["child_event_id"]))
        for row in selected_edges
    }
    ordered, ancestors, descendants, heights = _transitive_data(carrier, direct)
    closure = _closure_pairs(ordered, ancestors)
    by_observer: dict[str, list[tuple[int, str]]] = {}
    kind_counts: dict[str, int] = {}
    for event in selected_events:
        by_observer.setdefault(str(event["observer_token"]), []).append(
            (int(event["source_sequence_index"]), str(event["event_key"]))
        )
        kind = str(event["canonical_semantic_payload"]["event_kind"])
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    chains = [
        [key for _, key in sorted(by_observer[observer])]
        for observer in sorted(by_observer)
    ]
    chain_cover = all(
        all((left, right) in closure for left, right in zip(chain, chain[1:]))
        for chain in chains
    )
    antichain = sorted(
        str(event["event_key"])
        for event in selected_events
        if _sample(event) == 0
        and event["canonical_semantic_payload"]["event_kind"] == "RECORD_COMMIT"
    )
    antichain_holds = bool(
        len(antichain) == len(chains)
        and all(
            (left, right) not in closure and (right, left) not in closure
            for position, left in enumerate(antichain)
            for right in antichain[position + 1 :]
        )
    )
    event_count = len(selected_events)
    indices = sorted(int(event["source_sequence_index"]) for event in selected_events)
    if indices != list(range(event_count)) or event_count != cutoff * len(chains) * 3:
        _fail("prefix is not a complete-round initial source segment")
    maximum_interval = 0
    adequate_count = 0
    for future, pasts in enumerate(ancestors):
        for past in pasts:
            size = len(descendants[past] & ancestors[future]) + 2
            maximum_interval = max(maximum_interval, size)
            adequate_count += size >= 32
    event_projection = sorted(
        selected_events, key=lambda event: int(event["source_sequence_index"])
    )
    direct_projection = sorted(
        (
            {
                "parent_event_id": str(row["parent_event_id"]),
                "child_event_id": str(row["child_event_id"]),
                "shared_resource_ids": sorted(
                    str(value) for value in row["shared_resource_ids"]
                ),
            }
            for row in selected_edges
        ),
        key=lambda row: (row["parent_event_id"], row["child_event_id"]),
    )
    summary = {
        "complete_round_cutoff": cutoff,
        "event_count": event_count,
        "event_kind_counts": {key: kind_counts[key] for key in sorted(kind_counts)},
        "observer_count": len(chains),
        "direct_edge_count": len(direct),
        "comparable_pair_count": len(closure),
        "ordering_fraction": 2.0 * len(closure) / (event_count * (event_count - 1)),
        "height": max(heights),
        "width": len(chains),
        "maximum_interval_size": maximum_interval,
        "interval_count_at_least_32": adequate_count,
        "semantic_events_sha256": _sha(event_projection),
        "semantic_carrier_sha256": _sha(sorted(carrier)),
        "direct_order_sha256": _sha(direct_projection),
        "transitive_order_sha256": _sha(sorted(closure)),
        "observer_chain_cover": {
            "chain_count": len(chains),
            "chains_sha256": _sha(chains),
            "all_chain_successors_comparable": chain_cover,
        },
        "matching_antichain": {
            "event_keys": antichain,
            "event_keys_sha256": _sha(antichain),
            "pairwise_incomparable": antichain_holds,
        },
        "exact_width_certificate": bool(chain_cover and antichain_holds),
    }
    return summary, carrier, direct, closure


def _prescribed_shared_frame_rank3_diagnostic(
    events: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    observer_log: Mapping[str, Any],
) -> dict[str, Any]:
    selected_events = [event for event in events if _sample(event) < 4]
    selected_events.sort(key=lambda event: int(event["source_sequence_index"]))
    carrier = {str(event["event_key"]) for event in selected_events}
    direct = {
        (str(row["parent_event_id"]), str(row["child_event_id"]))
        for row in edges
        if str(row["parent_event_id"]) in carrier
        and str(row["child_event_id"]) in carrier
    }
    raw_pool = [
        dict(row)
        for row in observer_log["events"]
        if type(row.get("sample")) is int and int(row["sample"]) < 4
    ]
    raw_lookup = {
        (int(row["sample"]), str(row["kind"]), str(row["observer_token"])): row
        for row in raw_pool
    }
    if len(raw_lookup) != len(raw_pool):
        _fail("cone raw source log repeats a round/kind/observer event")
    try:
        raw_events = [
            raw_lookup[
                (
                    _sample(event),
                    str(event["canonical_semantic_payload"]["event_kind"]),
                    str(event["observer_token"]),
                )
            ]
            for event in selected_events
        ]
    except KeyError:
        _fail("cone raw/semantic alignment mismatch")
    if len(raw_events) != len(raw_pool) or any(
        str(raw["kind"])
        != str(event["canonical_semantic_payload"]["event_kind"])
        or str(raw["observer_token"]) != str(event["observer_token"])
        or int(raw["sample"]) != _sample(event)
        or (
            str(raw["kind"]) == "RECORD_COMMIT"
            and (
                int(raw["port"])
                != int(event["canonical_semantic_payload"]["port"])
                or str(raw["carrier_id"])
                != str(event["canonical_semantic_payload"]["carrier_id"])
            )
        )
        for raw, event in zip(raw_events, selected_events, strict=True)
    ):
        _fail("cone raw/semantic population mismatch")
    vertices = np.asarray(
        reference_echosahedral_carrier("independent-rank3-cone").port_coordinates,
        dtype=float,
    )
    raw_by_id = {str(row["event_id"]): row for row in raw_events}
    anchors: dict[str, np.ndarray] = {}
    for row in raw_events:
        if row["kind"] == "RECORD_COMMIT":
            port = row.get("port")
            if type(port) is not int or not (0 <= int(port) < len(vertices)):
                _fail("record rank-three port anchor missing")
            anchors[str(row["event_id"])] = vertices[int(port)]
    record_sets: list[list[str]] = []
    coordinates: list[np.ndarray] = []
    for row in raw_events:
        if row["kind"] == "RECORD_COMMIT":
            ids = [str(row["event_id"])]
        elif row["kind"] == "READBACK":
            ids = [
                str(row["record_event_id"]),
                *(str(value) for value in row["cross_read_record_event_ids"]),
            ]
        else:
            readback = raw_by_id[str(row["readback_event_id"])]
            ids = [
                str(row["action_input_record_event_id"]),
                str(readback["record_event_id"]),
                *(
                    str(value)
                    for value in readback["cross_read_record_event_ids"]
                ),
            ]
        ids = sorted(set(ids))
        record_sets.append(ids)
        if not ids or any(key not in anchors for key in ids):
            _fail("cone event has an unresolved record anchor")
        coordinates.append(np.mean([anchors[key] for key in ids], axis=0))
    spatial = np.asarray(coordinates)
    ordered, ancestors, _, heights = _transitive_data(carrier, direct)
    closure = _closure_pairs(ordered, ancestors)
    order_index = {key: index for index, key in enumerate(ordered)}
    ranks = [
        heights[order_index[str(event["event_key"])]] - 1
        for event in selected_events
    ]
    source_sequence = {
        str(event["event_key"]): int(event["source_sequence_index"])
        for event in selected_events
    }
    sequence_time_orientation_compatible = all(
        source_sequence[past] < source_sequence[future]
        and heights[order_index[past]] < heights[order_index[future]]
        for past, future in closure
    )
    if not sequence_time_orientation_compatible:
        _fail("cone source-sequence/derived-rank orientation mismatch")
    lower = 0.0
    lower_row: dict[str, Any] | None = None
    upper = float("inf")
    upper_row: dict[str, Any] | None = None
    same_rank: list[list[str]] = []
    zero_separation: list[list[str]] = []
    rows: list[dict[str, Any]] = []
    comparable_count = 0
    for right in range(len(selected_events)):
        for left in range(right):
            left_key = str(selected_events[left]["event_key"])
            right_key = str(selected_events[right]["event_key"])
            comparable = (left_key, right_key) in closure or (
                right_key, left_key
            ) in closure
            rank_gap = abs(ranks[right] - ranks[left])
            distance = float(np.linalg.norm(spatial[right] - spatial[left]))
            row = {
                "left_event_key": left_key,
                "right_event_key": right_key,
                "comparable": comparable,
                "rank_gap": rank_gap,
                "spatial_distance": distance,
            }
            rows.append(row)
            if comparable:
                comparable_count += 1
                ratio = distance / rank_gap
                if ratio > lower:
                    lower = ratio
                    lower_row = row
            elif rank_gap == 0 and distance <= 1.0e-12:
                same_rank.append([left_key, right_key])
            elif rank_gap > 0:
                ratio = distance / rank_gap
                if ratio < upper:
                    upper = ratio
                    upper_row = row
                if distance <= 1.0e-12:
                    zero_separation.append([left_key, right_key])
    upper_value = None if upper == float("inf") else upper
    injective = not same_rank
    interval = bool(upper_value is not None and lower < upper_value)
    forward = all(
        not row["comparable"]
        or lower * row["rank_gap"] + 1.0e-12 >= row["spatial_distance"]
        for row in rows
    )
    reverse = all(
        row["comparable"]
        or lower * row["rank_gap"] < row["spatial_distance"] - 1.0e-12
        for row in rows
    )
    faithful = bool(
        injective
        and interval
        and sequence_time_orientation_compatible
        and forward
        and reverse
    )
    return {
        "status": (
            "ATTAINED_FINITE_FAITHFUL_CONE_PLACEMENT"
            if faithful
            else "NOT_ATTAINED_NO_ADMISSIBLE_GLOBAL_TIME_SCALE_OR_INJECTIVITY"
        ),
        "event_count": len(selected_events),
        "comparable_pair_count": comparable_count,
        "incomparable_pair_count": len(rows) - comparable_count,
        "time_coordinate": "global_scale_times_source_derived_longest_path_rank",
        "placement_class": (
            "prescribed_single_shared_reference_frame_port_anchor_and_"
            "consumed_record_barycentre_ansatz"
        ),
        "spatial_coordinate": (
            "prescribed_shared_frame_authenticated_rank3_icosahedral_"
            "record_port_anchor; "
            "readback_and_feedback_use_distinct_consumed-record barycentres"
        ),
        "inter_carrier_frame_gluing_source_derived": False,
        "consumed_record_barycentre_rule_source_derived": False,
        "other_source_selected_placements_excluded": False,
        "physical_no_go_for_other_source_selected_placements": False,
        "adjustable_parameter_count": 1,
        "adjustable_parameter": "single_positive_global_time_scale",
        "rank3_port_coordinates_sha256": _sha(vertices.tolist()),
        "event_source_record_sets_sha256": _sha(record_sets),
        "event_spatial_coordinates_sha256": _sha(spatial.tolist()),
        "pair_constraint_population_sha256": _sha(rows),
        "causal_lower_time_scale_bound": lower,
        "causal_lower_bound_witness": lower_row,
        "spacelike_upper_time_scale_bound": upper_value,
        "spacelike_upper_bound_witness": upper_row,
        "global_time_scale_interval_nonempty": interval,
        "coincident_same_rank_incomparable_pair_count": len(same_rank),
        "coincident_same_rank_incomparable_pairs_sha256": _sha(same_rank),
        "incomparable_zero_spatial_separation_pair_count": len(zero_separation),
        "incomparable_zero_spatial_separation_pairs_sha256": _sha(zero_separation),
        "injective_four_coordinate_map": injective,
        "source_sequence_time_orientation_compatible": (
            sequence_time_orientation_compatible
        ),
        "all_precedence_pairs_future_causal_at_lower_bound": forward,
        "all_incomparable_pairs_spacelike_at_lower_bound": reverse,
        "precedence_iff_future_causal": faithful,
        "FINITE_FAITHFUL_RANK3_CONE_PLACEMENT_RECEIPT": faithful,
        "physical_promotion_allowed": False,
        "interpretation": (
            "This prescribed single-shared-frame port-anchor/consumed-record-"
            "barycentre ansatz fails before any manifold test: some "
            "incomparable readback/feedback events have identical rank-three "
            "barycentres, and "
            "the causal lower scale bound is not below the spacelike upper "
            "bound. No inter-carrier frame gluing or barycentre-selection rule "
            "has been source-derived, so this exact negative is not a no-go "
            "for other source-selected placements. Arbitrary event-specific "
            "coordinates or a multi-parameter fit are deliberately not used "
            "to force a positive result."
        ),
    }


def verify_receipt(path: Path | str = DEFAULT_RECEIPT) -> dict[str, Any]:
    receipt_path = Path(path)
    receipt_bytes = receipt_path.read_bytes()
    report = json.loads(receipt_bytes.decode("ascii"))
    if receipt_bytes != _canonical_bytes(report):
        _fail("full receipt is not in canonical byte form")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != _sha(body):
        _fail("report hash mismatch")
    if report.get("schema") != EXPECTED_SCHEMA or report.get("status") != EXPECTED_STATUS:
        _fail("schema or status mismatch")
    if report.get("artifact_type") != EXPECTED_ARTIFACT_TYPE:
        _fail("artifact type mismatch")
    if tuple(report.get("round_cutoffs", [])) != EXPECTED_CUTOFFS:
        _fail("round cutoff family mismatch")
    source = json.loads(SOURCE_RECEIPT.read_text(encoding="ascii"))
    source_body = {key: value for key, value in source.items() if key != "report_sha256"}
    if source.get("report_sha256") != _sha(source_body):
        _fail("canonical four-round source receipt hash mismatch")
    expected_maximum_config = dict(source["config"])
    expected_maximum_config["observer_samples"] = max(EXPECTED_CUTOFFS)
    if (
        not isinstance(report.get("maximum_config"), Mapping)
        or dict(report["maximum_config"]) != expected_maximum_config
    ):
        _fail("maximum source config mismatch")

    raw_evidence = report.get("cutoff_run_evidence")
    if not isinstance(raw_evidence, list) or len(raw_evidence) != len(
        EXPECTED_CUTOFFS
    ):
        _fail("one raw evidence package per cutoff is required")
    expected_levels: list[dict[str, Any]] = []
    carriers: list[set[str]] = []
    directs: list[set[tuple[str, str]]] = []
    closures: list[set[tuple[str, str]]] = []
    cutoff_events: list[list[dict[str, Any]]] = []
    cutoff_generated: list[list[dict[str, Any]]] = []
    cutoff_logs: list[Mapping[str, Any]] = []
    capture_hashes: list[str] = []
    for cutoff, raw_row in zip(EXPECTED_CUTOFFS, raw_evidence, strict=True):
        if not isinstance(raw_row, Mapping):
            _fail(f"cutoff {cutoff} evidence row is not a mapping")
        evidence = dict(raw_row)
        evidence_hash = evidence.pop("cutoff_run_evidence_sha256", None)
        if evidence_hash != _sha(evidence):
            _fail(f"cutoff {cutoff} evidence self-hash mismatch")
        expected_config = dict(source["config"])
        expected_config["observer_samples"] = cutoff
        if (
            raw_row.get("complete_round_cutoff") != cutoff
            or not isinstance(raw_row.get("config"), Mapping)
            or dict(raw_row["config"]) != expected_config
            or raw_row.get("config_sha256") != _sha(expected_config)
        ):
            _fail(f"cutoff {cutoff} does not bind its own exact source config")
        if raw_row.get("source_order_schema") != source["schema"]:
            _fail(f"cutoff {cutoff} source-order schema mismatch")
        _require_sha(
            raw_row.get("source_order_report_sha256"),
            f"cutoff {cutoff} source-order report hash",
        )
        binding = raw_row.get("source_capture_binding")
        if not isinstance(binding, Mapping) or set(binding) != {
            "capture_sha256",
            "postrun_capture_sha256",
            "source_root_sha256",
        }:
            _fail(f"cutoff {cutoff} source-capture binding is malformed")
        for key in (
            "capture_sha256",
            "postrun_capture_sha256",
            "source_root_sha256",
        ):
            _require_sha(binding.get(key), f"cutoff {cutoff} {key}")
        capture_hashes.append(str(binding["capture_sha256"]))

        observer_log = raw_row.get("observer_log_material")
        if not isinstance(observer_log, Mapping):
            _fail(f"cutoff {cutoff} embedded observer log is missing")
        if raw_row.get("observer_log_material_sha256") != _sha(observer_log):
            _fail(f"cutoff {cutoff} observer-log material hash mismatch")
        if (
            raw_row.get("observer_event_log_sha256")
            != observer_log.get("event_log_sha256")
            or observer_log.get("event_log_sha256")
            != _raw_sha(observer_log.get("events"))
        ):
            _fail(f"cutoff {cutoff} observer event-log hash mismatch")
        try:
            _checkpoint_audit(observer_log, expected_config)
            events, generated, declared, _ = _reconstruct_semantic_projection(
                observer_log
            )
        except IndependentVerificationError as error:
            _fail(f"cutoff {cutoff} raw-log reconstruction failed: {error}")
        generated_projection = _edge_projection(generated)
        declared_projection = _edge_projection(declared)
        if generated_projection != declared_projection:
            _fail(
                f"cutoff {cutoff} declared ancestry differs from regenerated provenance"
            )
        semantic_projection = sorted(
            events, key=lambda event: int(event["source_sequence_index"])
        )
        if (
            raw_row.get("semantic_events_sha256") != _sha(semantic_projection)
            or raw_row.get("generated_edges_sha256")
            != _sha(generated_projection)
            or raw_row.get("declared_edges_sha256") != _sha(declared_projection)
            or raw_row.get("generated_edge_count") != len(generated_projection)
            or raw_row.get("declared_edge_count") != len(declared_projection)
            or raw_row.get("source_derived_order_receipt") is not True
            or raw_row.get("physical_promotion_allowed") is not False
        ):
            _fail(f"cutoff {cutoff} reconstructed source-order custody mismatch")
        level, carrier, direct, closure = _prefix_summary(
            events,
            generated_projection,
            cutoff,
            require_exact_cutoff_run=True,
        )
        level["independent_cutoff_run_evidence_sha256"] = evidence_hash
        level["generated_from_own_cutoff_capture"] = True
        expected_levels.append(level)
        carriers.append(carrier)
        directs.append(direct)
        closures.append(closure)
        cutoff_events.append(events)
        cutoff_generated.append(generated_projection)
        cutoff_logs.append(observer_log)
    if len(set(capture_hashes)) != len(EXPECTED_CUTOFFS):
        _fail("cutoff source-capture bindings are not pairwise distinct")
    if report.get("all_cutoffs_independently_generated") is not True:
        _fail("independent cutoff-generation certificate missing")
    first_evidence = raw_evidence[0]
    if (
        first_evidence.get("source_order_report_sha256")
        != source["report_sha256"]
        or first_evidence.get("source_capture_binding")
        != source["source_capture_binding"]
        or first_evidence.get("observer_event_log_sha256")
        != source["observer_event_log_sha256"]
        or first_evidence.get("observer_log_material_sha256")
        != source["observer_log_material_sha256"]
    ):
        _fail("independently regenerated four-round member is not canonical")
    if report.get("levels") != expected_levels:
        _fail("independent cutoff level summaries do not replay")

    expected_embeddings: list[dict[str, Any]] = []
    for index in range(len(EXPECTED_CUTOFFS) - 1):
        lower = carriers[index]
        direct_restriction = {
            pair for pair in directs[index + 1] if pair[0] in lower and pair[1] in lower
        }
        closure_restriction = {
            pair for pair in closures[index + 1] if pair[0] in lower and pair[1] in lower
        }
        row = {
            "from_complete_round_cutoff": EXPECTED_CUTOFFS[index],
            "to_complete_round_cutoff": EXPECTED_CUTOFFS[index + 1],
            "proper_carrier_inclusion": lower < carriers[index + 1],
            "direct_order_is_induced_restriction": directs[index]
            == direct_restriction,
            "transitive_order_is_induced_restriction": closures[index]
            == closure_restriction,
        }
        row["embedding_certificate_sha256"] = _sha(row)
        expected_embeddings.append(row)
    if report.get("induced_order_embeddings") != expected_embeddings or not report.get(
        "all_induced_order_embeddings_certified"
    ):
        _fail("induced-order embedding certificates do not replay")

    binding = report.get("canonical_four_round_source_binding")
    expected_binding = {
        "path": "data/causal_order/source_derived_causal_order_receipt.json",
        "report_sha256": source["report_sha256"],
        "source_schema": source["schema"],
        "source_receipt_attained": source.get(
            "SOURCE_DERIVED_CAUSAL_ORDER_RECEIPT"
        )
        is True,
        "four_round_semantic_events_byte_identical": _sha(
            sorted(
                source["semantic_events"],
                key=lambda event: int(event["source_sequence_index"]),
            )
        )
        == expected_levels[0]["semantic_events_sha256"],
        "four_round_direct_order_byte_identical": source["generated_edges_sha256"]
        == expected_levels[0]["direct_order_sha256"],
    }
    if binding != expected_binding or not all(
        expected_binding[key]
        for key in (
            "source_receipt_attained",
            "four_round_semantic_events_byte_identical",
            "four_round_direct_order_byte_identical",
        )
    ):
        _fail("canonical four-round source binding mismatch")

    expected_cone = _prescribed_shared_frame_rank3_diagnostic(
        cutoff_events[0], cutoff_generated[0], cutoff_logs[0]
    )
    if report.get("prescribed_single_frame_source_port_placement") != expected_cone:
        _fail("prescribed shared-frame rank-three diagnostic does not replay")
    if expected_cone["FINITE_FAITHFUL_RANK3_CONE_PLACEMENT_RECEIPT"]:
        _fail("frozen cone diagnostic unexpectedly promoted")

    expected_scaling = {
        "event_counts": [level["event_count"] for level in expected_levels],
        "heights": [level["height"] for level in expected_levels],
        "widths": [level["width"] for level in expected_levels],
        "ordering_fractions": [
            level["ordering_fraction"] for level in expected_levels
        ],
        "maximum_interval_sizes": [
            level["maximum_interval_size"] for level in expected_levels
        ],
        "width_constant_at_observer_count": True,
        "height_strictly_increases": True,
        "ordering_fraction_strictly_increases": True,
        "interpretation": (
            "Independently executed complete-round cutoffs lengthen a "
            "two-chain observer history and form a certified directed "
            "family of informational orders. They are not a fixed-region "
            "density refinement and provide no evidence for spatial "
            "dimension, volume, or manifoldlikeness."
        ),
    }
    if report.get("scaling_diagnostic") != expected_scaling:
        _fail("scaling diagnostic mismatch")
    if report.get("negative_controls") != {
        "direct_edge_deletion_changes_maximal_order_hash": True,
        "future_event_injection_breaks_initial_segment": True,
        "four_round_member_binds_canonical_source_receipt": True,
        "all_levels_have_exact_chain_cover_antichain_width_certificates": True,
        "cutoff_capture_bindings_are_pairwise_distinct": True,
        "every_level_binds_its_own_cutoff_raw_log": True,
        "maximum_log_substitution_changes_nonmaximal_evidence": True,
    } or report.get("controls_fail_closed") is not True:
        _fail("negative controls do not replay")
    for field in (
        "INFORMATIONAL_HISTORY_EXTENSION_FAMILY_RECEIPT",
        "INFORMATIONAL_INDUCED_PREFIX_REFINEMENT_RECEIPT",
        "INFORMATIONAL_INDEPENDENT_CUTOFF_GENERATION_RECEIPT",
    ):
        if report.get(field) is not True:
            _fail(f"positive informational refinement receipt missing: {field}")
    for field in (
        "PHYSICAL_CAUSAL_ATTACHMENT_RECEIPT",
        "SOURCE_SELECTED_SPACETIME_REFINEMENT_FAMILY_RECEIPT",
        "CAUSET_FAITHFUL_EMBEDDING_RECEIPT",
        "CAUSET_MANIFOLDLIKE_REFINEMENT_RECEIPT",
        "CAUSET_DIMENSION_3P1_RECEIPT",
        "CAUSET_COUNT_VOLUME_DENSITY_RECEIPT",
        "SOURCE_LORENTZ_CONE_COMPATIBILITY_RECEIPT",
        "SOURCE_CAUSAL_STABLE_TIME_FUNCTION_RECEIPT",
        "PHYSICAL_SOURCE_CAUSAL_REFINEMENT_COMPATIBILITY_RECEIPT",
        "EVENT_TOPOLOGY_ATLAS_LIMIT_RECEIPT",
        "SOURCE_DERIVED_CAUSAL_3P1_MANIFOLD_LIMIT_RECEIPT",
        "physical_promotion_allowed",
    ):
        if report.get(field) is not False:
            _fail(f"forbidden promotion flag raised: {field}")
    if report.get("required_next_step") != EXPECTED_REQUIRED_NEXT_STEP:
        _fail("required-next-step boundary mismatch")
    if report.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY:
        _fail("claim boundary mismatch")
    return {
        "verified": True,
        "level_count": len(expected_levels),
        "event_counts": [level["event_count"] for level in expected_levels],
        "widths": [level["width"] for level in expected_levels],
        "cone_status": expected_cone["status"],
        "independent_cutoff_generation": True,
        "spacetime_refinement_promoted": False,
    }


def _expected_publication_projection(
    report: Mapping[str, Any], full_receipt_bytes: bytes
) -> dict[str, Any]:
    level_keys = (
        "complete_round_cutoff",
        "event_count",
        "event_kind_counts",
        "observer_count",
        "direct_edge_count",
        "comparable_pair_count",
        "ordering_fraction",
        "height",
        "width",
        "maximum_interval_size",
        "interval_count_at_least_32",
        "semantic_events_sha256",
        "semantic_carrier_sha256",
        "direct_order_sha256",
        "transitive_order_sha256",
        "observer_chain_cover",
        "exact_width_certificate",
        "independent_cutoff_run_evidence_sha256",
        "generated_from_own_cutoff_capture",
    )
    levels = []
    for raw_level in report["levels"]:
        level = {key: raw_level[key] for key in level_keys}
        level["matching_antichain"] = {
            "event_keys_sha256": raw_level["matching_antichain"][
                "event_keys_sha256"
            ],
            "pairwise_incomparable": raw_level["matching_antichain"][
                "pairwise_incomparable"
            ],
        }
        levels.append(level)
    promotion_keys = (
        "PHYSICAL_CAUSAL_ATTACHMENT_RECEIPT",
        "SOURCE_SELECTED_SPACETIME_REFINEMENT_FAMILY_RECEIPT",
        "CAUSET_FAITHFUL_EMBEDDING_RECEIPT",
        "CAUSET_MANIFOLDLIKE_REFINEMENT_RECEIPT",
        "CAUSET_DIMENSION_3P1_RECEIPT",
        "CAUSET_COUNT_VOLUME_DENSITY_RECEIPT",
        "SOURCE_LORENTZ_CONE_COMPATIBILITY_RECEIPT",
        "SOURCE_CAUSAL_STABLE_TIME_FUNCTION_RECEIPT",
        "PHYSICAL_SOURCE_CAUSAL_REFINEMENT_COMPATIBILITY_RECEIPT",
        "EVENT_TOPOLOGY_ATLAS_LIMIT_RECEIPT",
        "SOURCE_DERIVED_CAUSAL_3P1_MANIFOLD_LIMIT_RECEIPT",
        "physical_promotion_allowed",
    )
    payload = {
        "schema": EXPECTED_PUBLICATION_SCHEMA,
        "status": report["status"],
        "artifact_type": "SOURCE_CAUSAL_HISTORY_FAMILY_PUBLICATION_PROJECTION",
        "full_receipt_relative_path": (
            "data/causal_order/source_causal_history_family_receipt.json"
        ),
        "full_receipt_schema": report["schema"],
        "full_receipt_report_sha256": report["report_sha256"],
        "full_receipt_file_sha256": (
            "sha256:" + hashlib.sha256(full_receipt_bytes).hexdigest()
        ),
        "canonical_four_round_source_binding": report[
            "canonical_four_round_source_binding"
        ],
        "round_cutoffs": report["round_cutoffs"],
        "cutoff_run_evidence_sha256s": [
            row["cutoff_run_evidence_sha256"]
            for row in report["cutoff_run_evidence"]
        ],
        "all_cutoffs_independently_generated": report[
            "all_cutoffs_independently_generated"
        ],
        "levels": levels,
        "induced_order_embeddings": report["induced_order_embeddings"],
        "all_induced_order_embeddings_certified": report[
            "all_induced_order_embeddings_certified"
        ],
        "scaling_diagnostic": report["scaling_diagnostic"],
        "prescribed_single_frame_source_port_placement": report[
            "prescribed_single_frame_source_port_placement"
        ],
        "negative_controls": report["negative_controls"],
        "controls_fail_closed": report["controls_fail_closed"],
        "INFORMATIONAL_HISTORY_EXTENSION_FAMILY_RECEIPT": report[
            "INFORMATIONAL_HISTORY_EXTENSION_FAMILY_RECEIPT"
        ],
        "INFORMATIONAL_INDUCED_PREFIX_REFINEMENT_RECEIPT": report[
            "INFORMATIONAL_INDUCED_PREFIX_REFINEMENT_RECEIPT"
        ],
        "INFORMATIONAL_INDEPENDENT_CUTOFF_GENERATION_RECEIPT": report[
            "INFORMATIONAL_INDEPENDENT_CUTOFF_GENERATION_RECEIPT"
        ],
        "promotion_and_nonclaim_flags": {
            key: report[key] for key in promotion_keys
        },
        "required_next_step": report["required_next_step"],
        "claim_boundary": report["claim_boundary"],
        "publication_scope": (
            "Compact theorem-level projection only. The full receipt embeds "
            "and independently replays raw evidence from every cutoff and "
            "remains canonical for event custody."
        ),
    }
    expected = dict(payload)
    expected["projection_sha256"] = _sha(payload)
    return expected


def verify_publication_projection(
    full_receipt_path: Path | str = DEFAULT_RECEIPT,
    projection_path: Path | str = DEFAULT_PUBLICATION_PROJECTION,
) -> dict[str, Any]:
    """Verify compact publication bytes against an independently replayed full receipt."""

    verify_receipt(full_receipt_path)
    full_bytes = Path(full_receipt_path).read_bytes()
    full_report = json.loads(full_bytes.decode("ascii"))
    projection_bytes = Path(projection_path).read_bytes()
    projection = json.loads(projection_bytes.decode("ascii"))
    if projection_bytes != _canonical_bytes(projection):
        _fail("publication projection is not in canonical byte form")
    expected = _expected_publication_projection(full_report, full_bytes)
    if projection != expected:
        _fail("publication projection does not match verified full receipt")
    if len(projection_bytes) >= 20_000:
        _fail("publication projection exceeds 20,000 bytes")
    return {
        "verified": True,
        "projection_bytes": len(projection_bytes),
        "projection_sha256": projection["projection_sha256"],
        "full_receipt_file_sha256": projection["full_receipt_file_sha256"],
    }


def _summary_line(result: Mapping[str, Any]) -> str:
    return (
        "source causal history family independent verification: "
        f"verified={result['verified']} levels={result['level_count']} "
        f"events={result['event_counts']} widths={result['widths']} "
        f"cone={result['cone_status']} spacetime_refinement=False"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", default=str(DEFAULT_RECEIPT))
    parser.add_argument("--projection")
    args = parser.parse_args()
    result = verify_receipt(args.receipt)
    if args.projection:
        projection_result = verify_publication_projection(
            args.receipt, args.projection
        )
        result["publication_projection"] = projection_result
    print(_summary_line(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
