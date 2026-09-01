"""Source-derived causal order for the observer event log (issue #763).

The committed capture pipeline emits semantic observer events whose
ancestry edges are read from declared ``parent_event_ids``. This producer
regenerates the causal edge set from resource provenance alone: an edge
runs from one event to another exactly when the second reads a resource
the first committed. Every committed resource in the observer log has one
writer, so the generated relation is read-after-write provenance with no
appeal to declared parents, sequence positions, worker metadata, or
timestamps.

The report compares the generated edge set with the declared ancestry
byte for byte under one canonical projection, quantifies both differences
(declared edges without a read-after-write witness, and provenance edges
absent from the declaration), certifies acyclicity and the derived
longest-path rank of the generated set, and checks that every generated
edge advances the archival source sequence index, the executed-history
counterpart of the append-only ancestry rank.

Claim boundary: the generated order is informational structure on one
presentation-bound finite log. No physical causality, spacetime, manifold,
Lorentzian, or continuum statement follows, and no promotion of the
capture's other reports is implied. A byte-identity failure is a finding
about the declared ancestry, not a physical verdict.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from oph_fpe.bulk.physical_h3_kms_source_capture import (
    _semantic_source_events,
    capture_physical_source,
)

SCHEMA = "oph.source-derived-causal-order.v1"
ISSUE = 763

DEFAULT_CONFIG: dict[str, Any] = {
    "carrier_count": 4,
    "seed": 1729,
    "propagation_steps": 2,
    "cycles": 16,
    "repair_fraction_per_cycle": 0.0625,
    "record_commit_cycles": 4,
    "observer_count": 2,
    "observer_support_size": 2,
    "observer_samples": 4,
    "observer_cross_reads": True,
    "prediction_control": "semantic_hash_shuffle_v1",
    "feedback_enabled": True,
    "checkpoint_interval": 3,
    "support_refinement_level": 1,
    "geometry_sample_count": 4,
    "rung": 4,
    "replicate_id": "primary",
    "preregistered_plan_sha256": "sha256:" + "a" * 64,
    "intrinsic_step": 0.137,
    "coupling_strength": 1.0,
    "state_space": "normalized_complex_amplitude_in_C12",
    "rng_family": "numpy_generator_pcg64_v1",
    "initialization_distribution": "normalized_complex_gaussian_v1",
    "intrinsic_phase_distribution": "uniform_unit_interval_v1",
    "seam_update_rule": "disjoint_single_port_endpoint_arithmetic_mean_v1",
}


class SourceDerivedCausalOrderError(RuntimeError):
    """Raised when the provenance material violates a structural clause."""


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


def _sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _provenance_view(semantic_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project events onto the fields provenance generation may read.

    The declared ``parent_event_ids`` and every archival or executor field
    are removed, so the generator is structurally unable to consult them.
    """

    view: list[dict[str, Any]] = []
    for event in semantic_events:
        view.append(
            {
                "event_key": str(event["event_key"]),
                "read_resource_ids": sorted(
                    str(item) for item in event["read_resource_ids"]
                ),
                "write_resource_ids": sorted(
                    str(item) for item in event["write_resource_ids"]
                ),
            }
        )
    return view


def generated_provenance_edges(
    provenance_view: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate causal edges from read-after-write resource provenance.

    An edge (parent, child) exists exactly when the child reads a resource
    the parent wrote. The single-writer clause is checked fail-closed: a
    resource with two writers has no unambiguous version provenance in
    this projection and aborts generation.
    """

    writer_of: dict[str, str] = {}
    for event in provenance_view:
        for resource in event["write_resource_ids"]:
            if resource in writer_of:
                raise SourceDerivedCausalOrderError(
                    "resource has two writers in the observer log: "
                    f"{resource!r} by {writer_of[resource]!r} and "
                    f"{event['event_key']!r}"
                )
            writer_of[resource] = event["event_key"]

    shared: dict[tuple[str, str], list[str]] = {}
    for event in provenance_view:
        child = event["event_key"]
        for resource in event["read_resource_ids"]:
            parent = writer_of.get(resource)
            if parent is None or parent == child:
                continue
            shared.setdefault((parent, child), []).append(resource)

    edges = [
        {
            "parent_event_id": parent,
            "child_event_id": child,
            "shared_resource_ids": sorted(resources),
        }
        for (parent, child), resources in shared.items()
    ]
    edges.sort(key=lambda row: (row["parent_event_id"], row["child_event_id"]))
    return edges


def declared_ancestry_projection(
    ancestry: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project the declared ancestry onto the comparable edge shape."""

    edges = [
        {
            "parent_event_id": str(row["parent_event_id"]),
            "child_event_id": str(row["child_event_id"]),
            "shared_resource_ids": sorted(
                str(item) for item in row["shared_resource_ids"]
            ),
        }
        for row in ancestry
    ]
    edges.sort(key=lambda row: (row["parent_event_id"], row["child_event_id"]))
    return edges


def _longest_path_ranks(
    events: list[str], edges: list[dict[str, Any]]
) -> tuple[bool, dict[str, int]]:
    """Kahn layering of the generated set: acyclicity plus derived rank."""

    indegree = {event: 0 for event in events}
    children: dict[str, list[str]] = {event: [] for event in events}
    for row in edges:
        indegree[row["child_event_id"]] += 1
        children[row["parent_event_id"]].append(row["child_event_id"])
    rank = {event: 0 for event in events}
    frontier = sorted(event for event, degree in indegree.items() if degree == 0)
    visited = 0
    while frontier:
        event = frontier.pop()
        visited += 1
        for child in children[event]:
            rank[child] = max(rank[child], rank[event] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                frontier.append(child)
    return visited == len(events), rank


def _writer_permutation_control(
    provenance_view: list[dict[str, Any]],
    generated: list[dict[str, Any]],
) -> bool:
    """Rotating write attributions must change the generated edge set."""

    writers = [event for event in provenance_view if event["write_resource_ids"]]
    if len(writers) < 2:
        return False
    rotated = [dict(event) for event in provenance_view]
    writer_keys = [event["event_key"] for event in writers]
    write_sets = [event["write_resource_ids"] for event in writers]
    rotation = {
        key: write_sets[(index + 1) % len(write_sets)]
        for index, key in enumerate(writer_keys)
    }
    for event in rotated:
        if event["event_key"] in rotation:
            event["write_resource_ids"] = rotation[event["event_key"]]
    try:
        permuted = generated_provenance_edges(rotated)
    except SourceDerivedCausalOrderError:
        return True
    return _canonical_bytes(permuted) != _canonical_bytes(generated)


def produce_source_derived_causal_order_report(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    capture = capture_physical_source(dict(config) if config else DEFAULT_CONFIG)
    observer_log = capture["source_artifacts"]["observer_log"]
    semantic_events, declared_ancestry = _semantic_source_events(observer_log)

    provenance_view = _provenance_view(semantic_events)
    generated = generated_provenance_edges(provenance_view)
    declared = declared_ancestry_projection(declared_ancestry)

    generated_bytes = _canonical_bytes(generated)
    declared_bytes = _canonical_bytes(declared)
    byte_identical = generated_bytes == declared_bytes

    generated_pairs = {
        (row["parent_event_id"], row["child_event_id"]) for row in generated
    }
    declared_pairs = {
        (row["parent_event_id"], row["child_event_id"]) for row in declared
    }
    declared_only = sorted(declared_pairs - generated_pairs)
    generated_only = sorted(generated_pairs - declared_pairs)
    declared_empty_shared = sorted(
        (row["parent_event_id"], row["child_event_id"])
        for row in declared
        if not row["shared_resource_ids"]
    )

    event_keys = [event["event_key"] for event in provenance_view]
    acyclic, ranks = _longest_path_ranks(event_keys, generated)
    sequence_of = {
        str(event["event_key"]): int(event["source_sequence_index"])
        for event in semantic_events
    }
    sequence_compatible = all(
        sequence_of[row["parent_event_id"]] < sequence_of[row["child_event_id"]]
        for row in generated
    )

    controls = {
        "writer_permutation_changes_edges": _writer_permutation_control(
            provenance_view, generated
        ),
        "declared_parents_stripped_before_generation": True,
        "single_writer_clause_checked": True,
    }
    controls_fail_closed = all(controls.values())
    receipt_flag = bool(acyclic and sequence_compatible and controls_fail_closed)

    report = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "config": dict(config) if config else dict(DEFAULT_CONFIG),
        "event_count": len(provenance_view),
        "observer_event_log_sha256": observer_log["event_log_sha256"],
        "semantic_events": semantic_events,
        "generated_edges": generated,
        "declared_edges": declared,
        "generated_edges_sha256": _sha256(generated),
        "declared_edges_sha256": _sha256(declared),
        "generated_edge_count": len(generated),
        "declared_edge_count": len(declared),
        "byte_identity_clause": {
            "byte_identical": byte_identical,
            "verdict": "ATTAINED" if byte_identical else "NOT_ATTAINED",
            "declared_only_pair_count": len(declared_only),
            "generated_only_pair_count": len(generated_only),
            "declared_only_pairs": declared_only,
            "generated_only_pairs": generated_only,
            "declared_edges_without_read_after_write_witness": declared_empty_shared,
        },
        "generated_acyclic": acyclic,
        "generated_longest_path_rank_max": max(ranks.values(), default=0),
        "sequence_compatible": sequence_compatible,
        "negative_controls": controls,
        "controls_fail_closed": controls_fail_closed,
        "SOURCE_DERIVED_CAUSAL_ORDER_RECEIPT": receipt_flag,
        "physical_promotion_allowed": False,
        "claim_boundary": (
            "The generated relation is informational read-after-write "
            "provenance on one presentation-bound finite observer log. "
            "It selects no schedule, supplies no physical causality, "
            "spacetime, manifold, Lorentzian, or continuum statement, and "
            "promotes no other capture report. A byte-identity failure "
            "classifies the declared ancestry against provenance and "
            "carries no physical verdict."
        ),
    }
    report["report_sha256"] = _sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Produce the source-derived causal-order report."
    )
    parser.add_argument(
        "--out",
        default="data/causal_order/source_derived_causal_order_receipt.json",
        help="receipt output path",
    )
    args = parser.parse_args()
    report = produce_source_derived_causal_order_report()
    from pathlib import Path

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_canonical_bytes(report))
    print(
        f"{out}: events={report['event_count']} "
        f"generated={report['generated_edge_count']} "
        f"declared={report['declared_edge_count']} "
        f"byte_identical={report['byte_identity_clause']['byte_identical']} "
        f"receipt={report['SOURCE_DERIVED_CAUSAL_ORDER_RECEIPT']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
