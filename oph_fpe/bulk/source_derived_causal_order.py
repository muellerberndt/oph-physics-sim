"""Source-derived causal order for the observer event log.

The committed capture pipeline emits semantic observer events together with
an ancestry surface generated from resource provenance. This producer
independently regenerates the causal edge set from those resources alone: an
edge runs from one event to another exactly when the second reads a resource
the first committed. Every committed resource has one writer, and every
unwritten input must be an explicit distinguished source root, so the
generated relation makes no appeal to declared parents, sequence positions,
worker metadata, or timestamps.

The report compares the generated edge set with the declared ancestry
byte for byte under one canonical projection, quantifies both differences
(declared edges without a read-after-write witness, and provenance edges
absent from the declaration), certifies acyclicity and the derived
longest-path rank of the generated set, and checks that every generated
edge advances the archival source sequence index, the executed-history
counterpart of the append-only ancestry rank.

Claim boundary: the generated order is informational structure on one finite
log. Its semantic IDs exclude transport metadata but remain carrier/port
presentation-bound. No physical causality, spacetime, manifold,
Lorentzian, or continuum statement follows, and no promotion of the
capture's other reports is implied. A byte-identity failure is a finding
about the declared ancestry, not a physical verdict.
"""

from __future__ import annotations

import hashlib
import json
import copy
from typing import Any, Mapping

from oph_fpe.bulk.physical_h3_kms_source_capture import (
    _semantic_source_events,
    capture_physical_source,
)

SCHEMA = "oph.source-derived-causal-order.v2"

DEFAULT_CONFIG: dict[str, Any] = {
    "carrier_count": 4,
    "seed": 1729,
    "propagation_steps": 2,
    "cycles": 16,
    "repair_fraction_per_cycle": 0.0625,
    "record_commit_cycles": 4,
    "observer_count": 2,
    # Three carriers force an actual shared-support cross-read in this bounded
    # control log; support size two produced only disconnected observer units.
    "observer_support_size": 3,
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


def _raw_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


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
    *,
    distinguished_source_resource_ids: list[str] | tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Generate causal edges from read-after-write resource provenance.

    An edge (parent, child) exists exactly when the child reads a resource
    the parent wrote. The single-writer clause is checked fail-closed: a
    resource with two writers has no unambiguous version provenance in
    this projection and aborts generation.
    """

    roots = {str(value) for value in distinguished_source_resource_ids}
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
            if parent is None:
                if resource not in roots:
                    raise SourceDerivedCausalOrderError(
                        "read resource has neither a writer nor a distinguished "
                        f"source root: {resource!r}"
                    )
                continue
            if parent == child:
                raise SourceDerivedCausalOrderError(
                    f"event {child!r} reads its own committed resource {resource!r}"
                )
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
    distinguished_roots: list[str],
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
        permuted = generated_provenance_edges(
            rotated,
            distinguished_source_resource_ids=distinguished_roots,
        )
    except SourceDerivedCausalOrderError:
        return True
    return _canonical_bytes(permuted) != _canonical_bytes(generated)


def _refusal_control(
    provenance_view: list[dict[str, Any]],
    distinguished_roots: list[str],
    *,
    duplicate_writer: bool,
) -> bool:
    """Exercise the two fail-closed resource-integrity clauses."""

    mutated = copy.deepcopy(provenance_view)
    if duplicate_writer:
        writer = next(row for row in mutated if row["write_resource_ids"])
        other = next(row for row in mutated if row is not writer)
        other["write_resource_ids"] = [writer["write_resource_ids"][0]]
    else:
        mutated[0]["read_resource_ids"] = ["unrooted:missing-writer"]
    try:
        generated_provenance_edges(
            mutated,
            distinguished_source_resource_ids=distinguished_roots,
        )
    except SourceDerivedCausalOrderError:
        return True
    return False


def _transport_metadata_controls(
    observer_log: Mapping[str, Any],
    semantic_events: list[dict[str, Any]],
    generated: list[dict[str, Any]],
) -> dict[str, bool]:
    """Mutate noncausal transport metadata and replay the semantic projection."""

    baseline_events = _canonical_bytes(semantic_events)
    baseline_generated = _canonical_bytes(generated)

    parent_mutation = copy.deepcopy(observer_log)
    child = next(
        row for row in parent_mutation["events"] if row.get("parents")
    )
    child["parents"] = []
    parent_semantic, parent_generated, parent_declared, _ = (
        _semantic_source_events(parent_mutation, validate_transport_ids=False)
    )
    parent_generated_projection = declared_ancestry_projection(
        parent_generated
    )
    parent_declared_projection = declared_ancestry_projection(parent_declared)

    order_mutation = copy.deepcopy(observer_log)
    record = next(
        row
        for row in order_mutation["events"]
        if row.get("kind") == "RECORD_COMMIT"
    )
    record["record_order_previous_event_ids"] = ["sha256:" + "f" * 64]
    order_semantic, order_generated, _, _ = _semantic_source_events(
        order_mutation, validate_transport_ids=False
    )
    order_generated_projection = declared_ancestry_projection(order_generated)

    state_mutation = copy.deepcopy(observer_log)
    mutated_record = next(
        row
        for row in state_mutation["events"]
        if row.get("kind") == "RECORD_COMMIT"
    )
    mutated_record["full_port_state"][0] = (
        float(mutated_record["full_port_state"][0]) + 1.0
    )
    try:
        _semantic_source_events(state_mutation, validate_transport_ids=False)
        state_mutation_refused = False
    except RuntimeError:
        state_mutation_refused = True

    phase_order = ("RECORD_COMMIT", "READBACK", "LOCAL_FEEDBACK")
    permuted_log = copy.deepcopy(observer_log)
    permuted_events: list[dict[str, Any]] = []
    samples = sorted(
        {int(row["sample"]) for row in permuted_log["events"]}
    )
    for sample in samples:
        for kind in phase_order:
            permuted_events.extend(
                sorted(
                    (
                        row
                        for row in permuted_log["events"]
                        if int(row["sample"]) == sample and row["kind"] == kind
                    ),
                    key=lambda row: str(row["observer_token"]),
                    reverse=True,
                )
            )
    permuted_log["events"] = permuted_events
    permuted_semantic, permuted_generated, _, _ = _semantic_source_events(
        permuted_log
    )

    def presentation_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            (
                {
                    key: value
                    for key, value in row.items()
                    if key != "source_sequence_index"
                }
                for row in rows
            ),
            key=lambda row: str(row["event_key"]),
        )

    return {
        "declared_parent_mutation_leaves_semantic_order_unchanged": bool(
            _canonical_bytes(parent_semantic) == baseline_events
            and _canonical_bytes(parent_generated_projection)
            == baseline_generated
            and bool(parent_declared_projection != generated)
        ),
        "record_order_mutation_leaves_semantic_order_unchanged": bool(
            _canonical_bytes(order_semantic) == baseline_events
            and _canonical_bytes(order_generated_projection)
            == baseline_generated
        ),
        "record_full_state_hash_mutation_is_refused": state_mutation_refused,
        "phase_observer_permutation_leaves_semantic_order_unchanged": bool(
            presentation_events(permuted_semantic)
            == presentation_events(semantic_events)
            and declared_ancestry_projection(permuted_generated) == generated
            and observer_log.get("phase_observer_permutation_control", {}).get(
                "transport_event_material_set_invariant"
            )
            is True
            and observer_log.get("phase_observer_permutation_control", {}).get(
                "semantic_event_set_invariant"
            )
            is True
            and observer_log.get("phase_observer_permutation_control", {}).get(
                "source_order_invariant"
            )
            is True
        ),
    }


def _repair_only_event_carrier_control(
    dynamics: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify the current versioned seam-repair log as an event carrier."""

    events = [dict(row) for row in dynamics["repair_event_log"]]
    if not bool(dynamics["repair_event_examples_complete"]):
        raise SourceDerivedCausalOrderError(
            "bounded repair-only control requires the complete repair log"
        )
    writer_of_version: dict[tuple[str, int, int], str] = {}
    edges: set[tuple[str, str, str]] = set()
    roots: set[str] = set()
    for event in events:
        event_id = str(event["event_id"])
        material = {key: value for key, value in event.items() if key != "event_id"}
        if event_id != _raw_sha256(material):
            raise SourceDerivedCausalOrderError(
                "repair event ID does not bind transaction material"
            )
        for read in event["read_set"]:
            carrier = str(read["carrier_id"])
            port = int(read["port"])
            version = int(read["version"])
            resource = f"repair-port:{carrier}:{port:02d}:version-{version}"
            writer = writer_of_version.get((carrier, port, version))
            if writer is None:
                if version != 0:
                    raise SourceDerivedCausalOrderError(
                        "repair event reads an unwritten nonroot version"
                    )
                roots.add(resource)
            else:
                edges.add((writer, event_id, resource))
        for write in event["write_set"]:
            carrier = str(write["carrier_id"])
            port = int(write["port"])
            expected = int(write["expected_version"])
            committed = int(write["committed_version"])
            if committed != expected + 1:
                raise SourceDerivedCausalOrderError(
                    "repair write does not advance its exact version"
                )
            key = (carrier, port, committed)
            if key in writer_of_version:
                raise SourceDerivedCausalOrderError(
                    "repair resource version has multiple writers"
                )
            writer_of_version[key] = event_id
    projected_edges = [
        {
            "parent_event_id": parent,
            "child_event_id": child,
            "shared_resource_id": resource,
        }
        for parent, child, resource in sorted(edges)
    ]
    return {
        "schema": "oph.repair-only-event-carrier-control.v1",
        "repair_event_material": events,
        "repair_event_material_sha256": _sha256(events),
        "repair_event_count": len(events),
        "versioned_provenance_edges": projected_edges,
        "versioned_provenance_edge_count": len(projected_edges),
        "distinguished_version_zero_root_count": len(roots),
        "all_reads_are_version_zero_roots": bool(
            events
            and all(
                int(read["version"]) == 0
                for event in events
                for read in event["read_set"]
            )
        ),
        "classification": (
            "REPAIR_ONLY_EVENT_CARRIER_IS_ANTICHAIN"
            if events and not projected_edges
            else "REPAIR_ONLY_EVENT_CARRIER_HAS_VERSIONED_DEPENDENCIES"
        ),
        "physical_causet_promotion_allowed": False,
        "required_model_change": (
            "eventize_and_interleave_local_recurrent_propagation_with_"
            "versioned_seam_repair_so_state_can_transport_between_seams"
        ),
    }


def produce_source_derived_causal_order_report(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    capture = capture_physical_source(dict(config) if config else DEFAULT_CONFIG)
    observer_log = capture["source_artifacts"]["observer_log"]
    (
        semantic_events,
        capture_generated_ancestry,
        declared_ancestry,
        distinguished_roots,
    ) = _semantic_source_events(observer_log)

    provenance_view = _provenance_view(semantic_events)
    generated = generated_provenance_edges(
        provenance_view,
        distinguished_source_resource_ids=distinguished_roots,
    )
    declared = declared_ancestry_projection(declared_ancestry)
    capture_generated = declared_ancestry_projection(capture_generated_ancestry)
    postrun_generated = declared_ancestry_projection(
        capture["postrun_capture"]["raw_ancestry_relations"]
    )
    capture_ancestry_matches_generated = bool(
        generated == capture_generated == postrun_generated
    )

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
            provenance_view, generated, distinguished_roots
        ),
        "single_writer_mutation_is_refused": _refusal_control(
            provenance_view, distinguished_roots, duplicate_writer=True
        ),
        "missing_nonroot_writer_is_refused": _refusal_control(
            provenance_view, distinguished_roots, duplicate_writer=False
        ),
        **_transport_metadata_controls(
            observer_log, semantic_events, generated
        ),
    }
    controls_fail_closed = all(controls.values())
    repair_only_control = _repair_only_event_carrier_control(
        capture["source_artifacts"]["dynamics"]
    )
    observer_of = {
        str(event["event_key"]): str(event["observer_token"])
        for event in semantic_events
    }
    cross_observer_edge_count = sum(
        observer_of[row["parent_event_id"]]
        != observer_of[row["child_event_id"]]
        for row in generated
    )
    receipt_flag = bool(
        byte_identical
        and acyclic
        and sequence_compatible
        and controls_fail_closed
        and capture_ancestry_matches_generated
        and cross_observer_edge_count > 0
    )

    status = (
        "SOURCE_DERIVED_CAUSAL_ORDER_BYTE_IDENTITY_ATTAINED__PHYSICAL_ATTACHMENT_OPEN"
        if byte_identical
        else "SOURCE_DERIVED_CAUSAL_ORDER_BYTE_IDENTITY_NOT_ATTAINED__PHYSICAL_ATTACHMENT_OPEN"
    )
    report = {
        "schema": SCHEMA,
        "status": status,
        "config": dict(config) if config else dict(DEFAULT_CONFIG),
        "config_sha256": _sha256(
            dict(config) if config else dict(DEFAULT_CONFIG)
        ),
        "event_count": len(provenance_view),
        "observer_event_log_sha256": observer_log["event_log_sha256"],
        "observer_log_material_sha256": _sha256(observer_log),
        "observer_log_material": observer_log,
        "source_capture_binding": {
            "capture_sha256": capture["capture_sha256"],
            "source_root_sha256": capture["source_root_sha256"],
            "postrun_capture_sha256": capture["postrun_capture"][
                "primitive_root_sha256"
            ],
        },
        "event_carrier_scope": observer_log["event_carrier_scope"],
        "underlying_repair_transactions_promoted_as_events": observer_log[
            "underlying_repair_transactions_promoted_as_events"
        ],
        "distinguished_source_resource_ids": distinguished_roots,
        "semantic_events": semantic_events,
        "generated_edges": generated,
        "declared_edges": declared,
        "generated_edges_sha256": _sha256(generated),
        "declared_edges_sha256": _sha256(declared),
        "generated_edge_count": len(generated),
        "declared_edge_count": len(declared),
        "capture_ancestry_matches_generated": (
            capture_ancestry_matches_generated
        ),
        "cross_observer_edge_count": cross_observer_edge_count,
        "byte_identity_clause": {
            "scope": (
                "canonical_projected_provenance_edge_rows_on_bounded_"
                "source_observer_instrumentation_log"
            ),
            "comparison_representation": (
                "sorted_parent_child_shared_resource_rows_without_transport_"
                "sequence_or_edge_id_fields"
            ),
            "event_count": len(provenance_view),
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
        "repair_only_event_carrier_control": repair_only_control,
        "SOURCE_DERIVED_CAUSAL_ORDER_RECEIPT": receipt_flag,
        "physical_promotion_allowed": False,
        "claim_boundary": (
            "The generated relation is informational read-after-write "
            "provenance on one finite observer instrumentation log over "
            "source-state snapshots. The underlying seam-repair transactions "
            "are not events in this order, so it is not a complete physical "
            "repair-event causet. Semantic IDs exclude "
            "declared parents, record-order metadata, checkpoint placement, "
            "and source sequence, but remain carrier/port presentation-bound. "
            "It selects no schedule, supplies no physical causality, "
            "spacetime, manifold, Lorentzian, or continuum statement, and "
            "promotes no other capture report. A byte-identity failure "
            "classifies the canonical projected declared edge rows against "
            "the canonical projected provenance rows; it is not a claim of "
            "raw-log byte identity and "
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
