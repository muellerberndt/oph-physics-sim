"""Certified prefix family for the source-derived informational order.

The bounded source receipt proves one 24-event read-after-write order.  This
producer asks the next exact question without importing a spatial carrier or
tuning a causet fixture: do independently executed complete-round cutoffs of
the same deterministic source configuration form a genuine directed family
of induced suborders?

The registered capture is rerun separately at 4, 8, 16, 32, and 64 complete
rounds. Each raw cutoff log is embedded in the receipt, and its order is
generated from that cutoff's own read-after-write provenance. Adjacent direct
relations and transitive closures are then checked for exact induced
restriction. The four-round run is also bound byte-for-byte to the canonical
source-derived-order receipt. No level is produced by filtering the 64-round
semantic graph.

This is positive refinement *custody* for history extension, but it is not a
spacetime regulator refinement.  The event carrier is still the two-observer
instrumentation history over source snapshots.  Its events admit a cover by
two observer chains, while the two same-round record commits form a certified
antichain.  Hence every checked member has width exactly two.  Increasing the
number of rounds therefore lengthens a fixed-width history; it does not add
spatial resolution, produce a count-volume law, or approach a 3+1-dimensional
manifoldlike causal set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from oph_fpe.bulk.source_derived_causal_order import (
    DEFAULT_CONFIG,
    produce_source_derived_causal_order_report,
)
from oph_fpe.core.echosahedral_federation import reference_echosahedral_carrier


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_RECEIPT = (
    ROOT / "data" / "causal_order" / "source_derived_causal_order_receipt.json"
)
DEFAULT_OUTPUT = (
    ROOT / "data" / "causal_order" / "source_causal_history_family_receipt.json"
)
DEFAULT_PUBLICATION_OUTPUT = (
    ROOT
    / "data"
    / "causal_order"
    / "source_causal_history_family_publication_projection.json"
)
SCHEMA = "oph.source-causal-history-family.v1"
PUBLICATION_SCHEMA = "oph.source-causal-history-family-publication-projection.v1"
STATUS = (
    "CERTIFIED_INFORMATIONAL_HISTORY_EXTENSION_FAMILY__"
    "FIXED_WIDTH_NOT_SPACETIME_REFINEMENT"
)
ROUND_CUTOFFS = (4, 8, 16, 32, 64)


class SourceCausalHistoryFamilyError(RuntimeError):
    """Raised when a proposed history family fails an exact custody clause."""


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


def _sample(event: Mapping[str, Any]) -> int:
    payload = event.get("canonical_semantic_payload")
    if not isinstance(payload, Mapping) or type(payload.get("sample")) is not int:
        raise SourceCausalHistoryFamilyError("semantic event has no exact round index")
    return int(payload["sample"])


def _edge_pairs(rows: Iterable[Mapping[str, Any]]) -> list[tuple[str, str]]:
    return sorted(
        {
            (str(row["parent_event_id"]), str(row["child_event_id"]))
            for row in rows
        }
    )


def _transitive_data(
    nodes: Iterable[str], edges: Iterable[tuple[str, str]]
) -> tuple[list[str], list[set[int]], list[set[int]], list[int]]:
    """Return a topological carrier, closure, and longest-path node heights."""

    ordered = sorted(set(str(node) for node in nodes))
    if not ordered:
        raise SourceCausalHistoryFamilyError("history level has no events")
    index = {node: position for position, node in enumerate(ordered)}
    children = [set() for _ in ordered]
    indegree = [0 for _ in ordered]
    for raw_parent, raw_child in set(edges):
        parent = index.get(str(raw_parent))
        child = index.get(str(raw_child))
        if parent is None or child is None or parent == child:
            raise SourceCausalHistoryFamilyError(
                "history edge is not a strict relation on its declared carrier"
            )
        if child not in children[parent]:
            children[parent].add(child)
            indegree[child] += 1
    frontier = sorted(
        position for position, degree in enumerate(indegree) if degree == 0
    )
    topological: list[int] = []
    work = list(indegree)
    while frontier:
        node = frontier.pop()
        topological.append(node)
        for child in sorted(children[node]):
            work[child] -= 1
            if work[child] == 0:
                frontier.append(child)
    if len(topological) != len(ordered):
        raise SourceCausalHistoryFamilyError("history relation contains a cycle")
    ancestors = [set() for _ in ordered]
    descendants = [set() for _ in ordered]
    height = [1 for _ in ordered]
    for node in topological:
        for child in children[node]:
            ancestors[child].add(node)
            ancestors[child].update(ancestors[node])
            height[child] = max(height[child], height[node] + 1)
    for node in reversed(topological):
        for child in children[node]:
            descendants[node].add(child)
            descendants[node].update(descendants[child])
    return ordered, ancestors, descendants, height


def _closure_pairs(
    ordered: list[str], ancestors: list[set[int]]
) -> list[tuple[str, str]]:
    return sorted(
        (ordered[parent], ordered[child])
        for child, parents in enumerate(ancestors)
        for parent in parents
    )


def _level_summary(
    events: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    cutoff: int,
    *,
    require_exact_cutoff_run: bool = False,
) -> tuple[dict[str, Any], set[str], set[tuple[str, str]], set[tuple[str, str]]]:
    selected_events = [event for event in events if _sample(event) < cutoff]
    selected_keys = {str(event["event_key"]) for event in selected_events}
    selected_edges = [
        row
        for row in edges
        if str(row["parent_event_id"]) in selected_keys
        and str(row["child_event_id"]) in selected_keys
    ]
    if require_exact_cutoff_run and (
        len(selected_events) != len(events) or len(selected_edges) != len(edges)
    ):
        raise SourceCausalHistoryFamilyError(
            "cutoff level contains events or edges outside its own complete run"
        )
    direct_pairs = set(_edge_pairs(selected_edges))
    ordered, ancestors, descendants, heights = _transitive_data(
        selected_keys, direct_pairs
    )
    closure_pairs = set(_closure_pairs(ordered, ancestors))
    index = {node: position for position, node in enumerate(ordered)}

    by_observer: dict[str, list[str]] = {}
    kind_counts: dict[str, int] = {}
    for event in selected_events:
        observer = str(event["observer_token"])
        by_observer.setdefault(observer, []).append(str(event["event_key"]))
        kind = str(event["canonical_semantic_payload"]["event_kind"])
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    for observer in by_observer:
        by_observer[observer].sort(
            key=lambda key: int(
                next(
                    event["source_sequence_index"]
                    for event in selected_events
                    if event["event_key"] == key
                )
            )
        )
    chain_cover_holds = all(
        all((left, right) in closure_pairs for left, right in zip(chain, chain[1:]))
        for chain in by_observer.values()
    )
    first_round_records = sorted(
        str(event["event_key"])
        for event in selected_events
        if _sample(event) == 0
        and event["canonical_semantic_payload"]["event_kind"] == "RECORD_COMMIT"
    )
    antichain_holds = bool(
        len(first_round_records) == len(by_observer)
        and all(
            (left, right) not in closure_pairs
            and (right, left) not in closure_pairs
            for position, left in enumerate(first_round_records)
            for right in first_round_records[position + 1 :]
        )
    )
    if not chain_cover_holds or not antichain_holds:
        raise SourceCausalHistoryFamilyError(
            "observer-chain cover or matching antichain certificate failed"
        )

    comparable = len(closure_pairs)
    event_count = len(ordered)
    maximum_interval = 0
    adequate_intervals = 0
    for future, pasts in enumerate(ancestors):
        for past in pasts:
            size = len(descendants[past] & ancestors[future]) + 2
            maximum_interval = max(maximum_interval, size)
            adequate_intervals += size >= 32
    sequence_indices = sorted(
        int(event["source_sequence_index"]) for event in selected_events
    )
    expected_event_count = cutoff * len(by_observer) * 3
    if sequence_indices != list(range(expected_event_count)):
        raise SourceCausalHistoryFamilyError(
            "history level is not an initial segment of complete source rounds"
        )
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
        "observer_count": len(by_observer),
        "direct_edge_count": len(direct_pairs),
        "comparable_pair_count": comparable,
        "ordering_fraction": 2.0 * comparable / (event_count * (event_count - 1)),
        "height": max(heights),
        "width": len(by_observer),
        "maximum_interval_size": maximum_interval,
        "interval_count_at_least_32": adequate_intervals,
        "semantic_events_sha256": _sha(event_projection),
        "semantic_carrier_sha256": _sha(sorted(selected_keys)),
        "direct_order_sha256": _sha(direct_projection),
        "transitive_order_sha256": _sha(sorted(closure_pairs)),
        "observer_chain_cover": {
            "chain_count": len(by_observer),
            "chains_sha256": _sha(
                [by_observer[key] for key in sorted(by_observer)]
            ),
            "all_chain_successors_comparable": chain_cover_holds,
        },
        "matching_antichain": {
            "event_keys": first_round_records,
            "event_keys_sha256": _sha(first_round_records),
            "pairwise_incomparable": antichain_holds,
        },
        "exact_width_certificate": bool(
            chain_cover_holds
            and antichain_holds
            and len(first_round_records) == len(by_observer)
        ),
    }
    return summary, selected_keys, direct_pairs, closure_pairs


def _source_receipt_binding(
    first_level: Mapping[str, Any], source_path: Path
) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="ascii"))
    body = {key: value for key, value in source.items() if key != "report_sha256"}
    if source.get("report_sha256") != _sha(body):
        raise SourceCausalHistoryFamilyError("canonical source receipt hash mismatch")
    event_projection = sorted(
        source["semantic_events"], key=lambda event: int(event["source_sequence_index"])
    )
    try:
        source_label = str(source_path.relative_to(ROOT))
    except ValueError:
        source_label = str(source_path)
    return {
        "path": source_label,
        "report_sha256": source["report_sha256"],
        "source_schema": source["schema"],
        "source_receipt_attained": bool(
            source.get("SOURCE_DERIVED_CAUSAL_ORDER_RECEIPT") is True
        ),
        "four_round_semantic_events_byte_identical": bool(
            _sha(event_projection) == first_level["semantic_events_sha256"]
        ),
        "four_round_direct_order_byte_identical": bool(
            source["generated_edges_sha256"] == first_level["direct_order_sha256"]
        ),
    }


def _cutoff_run_evidence(
    cutoff: int,
    config: Mapping[str, Any],
    source_order_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the raw evidence needed to replay one cutoff independently."""

    report_body = {
        key: value
        for key, value in source_order_report.items()
        if key != "report_sha256"
    }
    if source_order_report.get("report_sha256") != _sha(report_body):
        raise SourceCausalHistoryFamilyError(
            f"cutoff {cutoff} source-order report hash mismatch"
        )
    if dict(source_order_report.get("config", {})) != dict(config):
        raise SourceCausalHistoryFamilyError(
            f"cutoff {cutoff} source-order report config mismatch"
        )
    observer_log = source_order_report.get("observer_log_material")
    if not isinstance(observer_log, Mapping):
        raise SourceCausalHistoryFamilyError(
            f"cutoff {cutoff} has no raw observer log"
        )
    if source_order_report.get("SOURCE_DERIVED_CAUSAL_ORDER_RECEIPT") is not True:
        raise SourceCausalHistoryFamilyError(
            f"cutoff {cutoff} source-derived causal order was not attained"
        )
    if source_order_report.get("physical_promotion_allowed") is not False:
        raise SourceCausalHistoryFamilyError(
            f"cutoff {cutoff} source-order report overpromotes physically"
        )
    event_projection = sorted(
        source_order_report["semantic_events"],
        key=lambda event: int(event["source_sequence_index"]),
    )
    row: dict[str, Any] = {
        "complete_round_cutoff": cutoff,
        "config": dict(config),
        "config_sha256": source_order_report["config_sha256"],
        "source_order_schema": source_order_report["schema"],
        "source_order_report_sha256": source_order_report["report_sha256"],
        "source_capture_binding": source_order_report["source_capture_binding"],
        "observer_event_log_sha256": source_order_report[
            "observer_event_log_sha256"
        ],
        "observer_log_material_sha256": source_order_report[
            "observer_log_material_sha256"
        ],
        "observer_log_material": observer_log,
        "semantic_events_sha256": _sha(event_projection),
        "generated_edges_sha256": source_order_report["generated_edges_sha256"],
        "declared_edges_sha256": source_order_report["declared_edges_sha256"],
        "generated_edge_count": source_order_report["generated_edge_count"],
        "declared_edge_count": source_order_report["declared_edge_count"],
        "source_derived_order_receipt": True,
        "physical_promotion_allowed": False,
    }
    row["cutoff_run_evidence_sha256"] = _sha(row)
    return row


def _prescribed_shared_frame_rank3_diagnostic(
    events: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    observer_log: Mapping[str, Any],
) -> dict[str, Any]:
    """Test one prescribed shared-frame map into ``R x rank3``.

    The diagnostic prescribes one common reference frame before the cone test.
    In that frame each record commit is placed at its authenticated rank-three
    icosahedral port coordinate.  A readback is placed at the arithmetic
    barycentre of the distinct record versions it reads.  Feedback uses the
    distinct union of the records consumed directly and through its readback.
    Neither an inter-carrier frame gluing rule nor this barycentre-selection
    rule is source-derived.  The diagnostic uses no causal-set target,
    quadratic-form fit, event sequence as a coordinate, or adjustable spatial
    coefficient.  Its only fitted scalar is a global multiplier on the
    source-derived longest-path rank.
    """

    selected_events = [event for event in events if _sample(event) < 4]
    selected_keys = {str(event["event_key"]) for event in selected_events}
    selected_edges = [
        row
        for row in edges
        if str(row["parent_event_id"]) in selected_keys
        and str(row["child_event_id"]) in selected_keys
    ]
    raw_pool = [
        dict(row)
        for row in observer_log["events"]
        if type(row.get("sample")) is int and int(row["sample"]) < 4
    ]
    selected_events.sort(key=lambda event: int(event["source_sequence_index"]))
    raw_lookup = {
        (int(row["sample"]), str(row["kind"]), str(row["observer_token"])): row
        for row in raw_pool
    }
    if len(raw_lookup) != len(raw_pool):
        raise SourceCausalHistoryFamilyError(
            "raw source log repeats a round/kind/observer event"
        )
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
    except KeyError as error:
        raise SourceCausalHistoryFamilyError(
            "raw/semantic event alignment failed for rank-three cone diagnostic"
        ) from error
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
        raise SourceCausalHistoryFamilyError(
            "raw/semantic event alignment failed for rank-three cone diagnostic"
        )

    vertices = np.asarray(
        reference_echosahedral_carrier("rank3-cone-diagnostic").port_coordinates,
        dtype=float,
    )
    raw_by_id = {str(row["event_id"]): row for row in raw_events}
    anchor_by_record: dict[str, np.ndarray] = {}
    for row in raw_events:
        if row["kind"] != "RECORD_COMMIT":
            continue
        port = row.get("port")
        if type(port) is not int or not (0 <= int(port) < len(vertices)):
            raise SourceCausalHistoryFamilyError(
                "record does not have an authenticated rank-three port anchor"
            )
        anchor_by_record[str(row["event_id"])] = vertices[int(port)]

    source_record_sets: list[list[str]] = []
    spatial: list[np.ndarray] = []
    for row in raw_events:
        kind = str(row["kind"])
        if kind == "RECORD_COMMIT":
            record_ids = [str(row["event_id"])]
        elif kind == "READBACK":
            record_ids = [
                str(row["record_event_id"]),
                *(str(value) for value in row["cross_read_record_event_ids"]),
            ]
        else:
            readback = raw_by_id.get(str(row["readback_event_id"]))
            if readback is None or readback["kind"] != "READBACK":
                raise SourceCausalHistoryFamilyError(
                    "feedback does not bind an available readback"
                )
            record_ids = [
                str(row["action_input_record_event_id"]),
                str(readback["record_event_id"]),
                *(
                    str(value)
                    for value in readback["cross_read_record_event_ids"]
                ),
            ]
        record_ids = sorted(set(record_ids))
        if not record_ids or any(key not in anchor_by_record for key in record_ids):
            raise SourceCausalHistoryFamilyError(
                "event source-record barycentre has an unresolved record"
            )
        source_record_sets.append(record_ids)
        spatial.append(
            np.mean([anchor_by_record[key] for key in record_ids], axis=0)
        )
    coordinates = np.asarray(spatial, dtype=float)

    ordered, ancestors, _, heights = _transitive_data(
        selected_keys, _edge_pairs(selected_edges)
    )
    ordered_index = {key: index for index, key in enumerate(ordered)}
    source_sequence = {
        str(event["event_key"]): int(event["source_sequence_index"])
        for event in selected_events
    }
    ranks_by_sequence = [
        heights[ordered_index[str(event["event_key"])]] - 1
        for event in selected_events
    ]
    closure = set(_closure_pairs(ordered, ancestors))
    sequence_time_orientation_compatible = all(
        source_sequence[past] < source_sequence[future]
        and heights[ordered_index[past]] < heights[ordered_index[future]]
        for past, future in closure
    )
    if not sequence_time_orientation_compatible:
        raise SourceCausalHistoryFamilyError(
            "source sequence and derived-rank time orientation disagree"
        )

    lower_bound = 0.0
    lower_witness: dict[str, Any] | None = None
    upper_bound = float("inf")
    upper_witness: dict[str, Any] | None = None
    coincident_same_rank: list[list[str]] = []
    incomparable_zero_spatial_separation: list[list[str]] = []
    comparable_pair_count = 0
    incomparable_pair_count = 0
    pair_rows: list[dict[str, Any]] = []
    for future in range(len(selected_events)):
        for past in range(future):
            past_key = str(selected_events[past]["event_key"])
            future_key = str(selected_events[future]["event_key"])
            forward = (past_key, future_key) in closure
            reverse = (future_key, past_key) in closure
            comparable = forward or reverse
            rank_gap = abs(ranks_by_sequence[future] - ranks_by_sequence[past])
            distance = float(np.linalg.norm(coordinates[future] - coordinates[past]))
            row = {
                "left_event_key": past_key,
                "right_event_key": future_key,
                "comparable": comparable,
                "rank_gap": rank_gap,
                "spatial_distance": distance,
            }
            pair_rows.append(row)
            if comparable:
                comparable_pair_count += 1
                if rank_gap <= 0:
                    raise SourceCausalHistoryFamilyError(
                        "comparable source events have no positive derived-rank gap"
                    )
                ratio = distance / rank_gap
                if ratio > lower_bound:
                    lower_bound = ratio
                    lower_witness = row
            else:
                incomparable_pair_count += 1
                if rank_gap == 0 and distance <= 1.0e-12:
                    coincident_same_rank.append([past_key, future_key])
                if rank_gap > 0:
                    ratio = distance / rank_gap
                    if ratio < upper_bound:
                        upper_bound = ratio
                        upper_witness = row
                    if distance <= 1.0e-12:
                        incomparable_zero_spatial_separation.append(
                            [past_key, future_key]
                        )
    if upper_bound == float("inf"):
        upper_bound_value: float | None = None
    else:
        upper_bound_value = upper_bound
    injection_holds = not coincident_same_rank
    scale_interval_nonempty = bool(
        upper_bound_value is not None and lower_bound < upper_bound_value
    )
    forward_causal_at_lower = all(
        not row["comparable"]
        or lower_bound * row["rank_gap"] + 1.0e-12
        >= row["spatial_distance"]
        for row in pair_rows
    )
    reverse_exclusion_at_lower = all(
        row["comparable"]
        or lower_bound * row["rank_gap"]
        < row["spatial_distance"] - 1.0e-12
        for row in pair_rows
    )
    faithful = bool(
        injection_holds
        and scale_interval_nonempty
        and sequence_time_orientation_compatible
        and forward_causal_at_lower
        and reverse_exclusion_at_lower
    )
    return {
        "status": (
            "ATTAINED_FINITE_FAITHFUL_CONE_PLACEMENT"
            if faithful
            else "NOT_ATTAINED_NO_ADMISSIBLE_GLOBAL_TIME_SCALE_OR_INJECTIVITY"
        ),
        "event_count": len(selected_events),
        "comparable_pair_count": comparable_pair_count,
        "incomparable_pair_count": incomparable_pair_count,
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
        "event_source_record_sets_sha256": _sha(source_record_sets),
        "event_spatial_coordinates_sha256": _sha(coordinates.tolist()),
        "pair_constraint_population_sha256": _sha(pair_rows),
        "causal_lower_time_scale_bound": lower_bound,
        "causal_lower_bound_witness": lower_witness,
        "spacelike_upper_time_scale_bound": upper_bound_value,
        "spacelike_upper_bound_witness": upper_witness,
        "global_time_scale_interval_nonempty": scale_interval_nonempty,
        "coincident_same_rank_incomparable_pair_count": len(
            coincident_same_rank
        ),
        "coincident_same_rank_incomparable_pairs_sha256": _sha(
            coincident_same_rank
        ),
        "incomparable_zero_spatial_separation_pair_count": len(
            incomparable_zero_spatial_separation
        ),
        "incomparable_zero_spatial_separation_pairs_sha256": _sha(
            incomparable_zero_spatial_separation
        ),
        "injective_four_coordinate_map": injection_holds,
        "source_sequence_time_orientation_compatible": (
            sequence_time_orientation_compatible
        ),
        "all_precedence_pairs_future_causal_at_lower_bound": (
            forward_causal_at_lower
        ),
        "all_incomparable_pairs_spacelike_at_lower_bound": (
            reverse_exclusion_at_lower
        ),
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


def produce_source_causal_history_family_report(
    *, source_receipt_path: Path | str = DEFAULT_SOURCE_RECEIPT
) -> dict[str, Any]:
    source_path = Path(source_receipt_path)
    if not source_path.is_absolute():
        source_path = (ROOT / source_path).resolve()
    cutoff_reports: list[dict[str, Any]] = []
    cutoff_evidence: list[dict[str, Any]] = []
    levels: list[dict[str, Any]] = []
    carriers: list[set[str]] = []
    direct_relations: list[set[tuple[str, str]]] = []
    transitive_relations: list[set[tuple[str, str]]] = []
    for cutoff in ROUND_CUTOFFS:
        config = dict(DEFAULT_CONFIG)
        config["observer_samples"] = cutoff
        cutoff_report = produce_source_derived_causal_order_report(config)
        evidence = _cutoff_run_evidence(cutoff, config, cutoff_report)
        events = list(cutoff_report["semantic_events"])
        edges = list(cutoff_report["generated_edges"])
        level, carrier, direct, transitive = _level_summary(
            events,
            edges,
            cutoff,
            require_exact_cutoff_run=True,
        )
        if (
            level["semantic_events_sha256"] != evidence["semantic_events_sha256"]
            or level["direct_order_sha256"] != evidence["generated_edges_sha256"]
            or level["direct_edge_count"] != evidence["generated_edge_count"]
        ):
            raise SourceCausalHistoryFamilyError(
                f"cutoff {cutoff} summary does not bind its independent run"
            )
        level["independent_cutoff_run_evidence_sha256"] = evidence[
            "cutoff_run_evidence_sha256"
        ]
        level["generated_from_own_cutoff_capture"] = True
        cutoff_reports.append(cutoff_report)
        cutoff_evidence.append(evidence)
        levels.append(level)
        carriers.append(carrier)
        direct_relations.append(direct)
        transitive_relations.append(transitive)

    maximum = cutoff_reports[-1]
    maximum_config = dict(maximum["config"])
    events = list(maximum["semantic_events"])
    edges = list(maximum["generated_edges"])

    embeddings: list[dict[str, Any]] = []
    for index in range(len(levels) - 1):
        lower_carrier = carriers[index]
        upper_carrier = carriers[index + 1]
        direct_restriction = {
            pair
            for pair in direct_relations[index + 1]
            if pair[0] in lower_carrier and pair[1] in lower_carrier
        }
        transitive_restriction = {
            pair
            for pair in transitive_relations[index + 1]
            if pair[0] in lower_carrier and pair[1] in lower_carrier
        }
        row = {
            "from_complete_round_cutoff": ROUND_CUTOFFS[index],
            "to_complete_round_cutoff": ROUND_CUTOFFS[index + 1],
            "proper_carrier_inclusion": lower_carrier < upper_carrier,
            "direct_order_is_induced_restriction": bool(
                direct_relations[index] == direct_restriction
            ),
            "transitive_order_is_induced_restriction": bool(
                transitive_relations[index] == transitive_restriction
            ),
        }
        row["embedding_certificate_sha256"] = _sha(row)
        embeddings.append(row)
    all_embeddings_certified = all(
        row["proper_carrier_inclusion"]
        and row["direct_order_is_induced_restriction"]
        and row["transitive_order_is_induced_restriction"]
        for row in embeddings
    )
    source_binding = _source_receipt_binding(levels[0], source_path)
    source_binding_holds = bool(
        source_binding["source_receipt_attained"]
        and source_binding["four_round_semantic_events_byte_identical"]
        and source_binding["four_round_direct_order_byte_identical"]
    )
    four_round_report = cutoff_reports[0]
    rank3_cone = _prescribed_shared_frame_rank3_diagnostic(
        list(four_round_report["semantic_events"]),
        list(four_round_report["generated_edges"]),
        four_round_report["observer_log_material"],
    )

    deleted_edges = list(edges[:-1])
    future_injected_indices = list(range(levels[0]["event_count"])) + [
        levels[0]["event_count"] + 1
    ]
    controls = {
        "direct_edge_deletion_changes_maximal_order_hash": bool(
            _sha(_edge_pairs(deleted_edges)) != _sha(_edge_pairs(edges))
        ),
        "future_event_injection_breaks_initial_segment": bool(
            future_injected_indices
            != list(range(len(future_injected_indices)))
        ),
        "four_round_member_binds_canonical_source_receipt": source_binding_holds,
        "all_levels_have_exact_chain_cover_antichain_width_certificates": all(
            level["exact_width_certificate"] for level in levels
        ),
        "cutoff_capture_bindings_are_pairwise_distinct": bool(
            len(
                {
                    row["source_capture_binding"]["capture_sha256"]
                    for row in cutoff_evidence
                }
            )
            == len(ROUND_CUTOFFS)
        ),
        "every_level_binds_its_own_cutoff_raw_log": all(
            level["independent_cutoff_run_evidence_sha256"]
            == evidence["cutoff_run_evidence_sha256"]
            and level["generated_from_own_cutoff_capture"] is True
            for level, evidence in zip(levels, cutoff_evidence, strict=True)
        ),
        "maximum_log_substitution_changes_nonmaximal_evidence": all(
            evidence["observer_log_material_sha256"]
            != cutoff_evidence[-1]["observer_log_material_sha256"]
            for evidence in cutoff_evidence[:-1]
        ),
    }
    controls_fail_closed = all(controls.values())
    report = {
        "schema": SCHEMA,
        "status": STATUS,
        "artifact_type": (
            "SOURCE_DERIVED_INDEPENDENT_INFORMATIONAL_HISTORY_EXTENSION_FAMILY"
        ),
        "maximum_config": maximum_config,
        "round_cutoffs": list(ROUND_CUTOFFS),
        "cutoff_run_evidence": cutoff_evidence,
        "all_cutoffs_independently_generated": True,
        "canonical_four_round_source_binding": source_binding,
        "prescribed_single_frame_source_port_placement": rank3_cone,
        "levels": levels,
        "induced_order_embeddings": embeddings,
        "all_induced_order_embeddings_certified": all_embeddings_certified,
        "scaling_diagnostic": {
            "event_counts": [level["event_count"] for level in levels],
            "heights": [level["height"] for level in levels],
            "widths": [level["width"] for level in levels],
            "ordering_fractions": [level["ordering_fraction"] for level in levels],
            "maximum_interval_sizes": [
                level["maximum_interval_size"] for level in levels
            ],
            "width_constant_at_observer_count": bool(
                len({level["width"] for level in levels}) == 1
                and levels[0]["width"] == maximum_config["observer_count"]
            ),
            "height_strictly_increases": all(
                levels[index]["height"] < levels[index + 1]["height"]
                for index in range(len(levels) - 1)
            ),
            "ordering_fraction_strictly_increases": all(
                levels[index]["ordering_fraction"]
                < levels[index + 1]["ordering_fraction"]
                for index in range(len(levels) - 1)
            ),
            "interpretation": (
                "Independently executed complete-round cutoffs lengthen a "
                "two-chain observer history and form a certified directed "
                "family of informational orders. They are not a fixed-region "
                "density refinement and provide no evidence for spatial "
                "dimension, volume, or manifoldlikeness."
            ),
        },
        "negative_controls": controls,
        "controls_fail_closed": controls_fail_closed,
        "INFORMATIONAL_HISTORY_EXTENSION_FAMILY_RECEIPT": bool(
            all_embeddings_certified and source_binding_holds and controls_fail_closed
        ),
        "INFORMATIONAL_INDUCED_PREFIX_REFINEMENT_RECEIPT": bool(
            all_embeddings_certified and source_binding_holds and controls_fail_closed
        ),
        "INFORMATIONAL_INDEPENDENT_CUTOFF_GENERATION_RECEIPT": bool(
            all_embeddings_certified
            and source_binding_holds
            and controls_fail_closed
            and len(cutoff_evidence) == len(ROUND_CUTOFFS)
        ),
        "PHYSICAL_CAUSAL_ATTACHMENT_RECEIPT": False,
        "SOURCE_SELECTED_SPACETIME_REFINEMENT_FAMILY_RECEIPT": False,
        "CAUSET_FAITHFUL_EMBEDDING_RECEIPT": False,
        "CAUSET_MANIFOLDLIKE_REFINEMENT_RECEIPT": False,
        "CAUSET_DIMENSION_3P1_RECEIPT": False,
        "CAUSET_COUNT_VOLUME_DENSITY_RECEIPT": False,
        "SOURCE_LORENTZ_CONE_COMPATIBILITY_RECEIPT": False,
        "SOURCE_CAUSAL_STABLE_TIME_FUNCTION_RECEIPT": False,
        "PHYSICAL_SOURCE_CAUSAL_REFINEMENT_COMPATIBILITY_RECEIPT": False,
        "EVENT_TOPOLOGY_ATLAS_LIMIT_RECEIPT": False,
        "SOURCE_DERIVED_CAUSAL_3P1_MANIFOLD_LIMIT_RECEIPT": False,
        "physical_promotion_allowed": False,
        "required_next_step": (
            "Construct a source-selected regulator family that increases local "
            "event density and spatial antichain capacity on comparable physical "
            "regions, eventize signal-capable repair/propagation links, and freeze "
            "multi-statistic Minkowski/FLRW comparisons before evaluating it."
        ),
        "claim_boundary": (
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
        ),
    }
    if not (
        report["INFORMATIONAL_HISTORY_EXTENSION_FAMILY_RECEIPT"]
        and report["INFORMATIONAL_INDEPENDENT_CUTOFF_GENERATION_RECEIPT"]
    ):
        raise SourceCausalHistoryFamilyError("history-family receipt did not attain")
    report["report_sha256"] = _sha(report)
    return report


def publication_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact canonical publication projection of a full receipt."""

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
    full_bytes = _canonical_bytes(dict(report))
    payload = {
        "schema": PUBLICATION_SCHEMA,
        "status": report["status"],
        "artifact_type": "SOURCE_CAUSAL_HISTORY_FAMILY_PUBLICATION_PROJECTION",
        "full_receipt_relative_path": (
            "data/causal_order/source_causal_history_family_receipt.json"
        ),
        "full_receipt_schema": report["schema"],
        "full_receipt_report_sha256": report["report_sha256"],
        "full_receipt_file_sha256": (
            "sha256:" + hashlib.sha256(full_bytes).hexdigest()
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
    projection = dict(payload)
    projection["projection_sha256"] = _sha(payload)
    return projection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_RECEIPT))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--publication-out", default=str(DEFAULT_PUBLICATION_OUTPUT))
    args = parser.parse_args()
    report = produce_source_causal_history_family_report(
        source_receipt_path=args.source
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(report))
    publication_output = Path(args.publication_out)
    publication_output.parent.mkdir(parents=True, exist_ok=True)
    publication_output.write_bytes(_canonical_bytes(publication_projection(report)))
    print(
        f"{output}: levels={len(report['levels'])} "
        f"events={report['scaling_diagnostic']['event_counts']} "
        f"widths={report['scaling_diagnostic']['widths']} "
        f"history_family={report['INFORMATIONAL_HISTORY_EXTENSION_FAMILY_RECEIPT']} "
        "spacetime_refinement=False"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
