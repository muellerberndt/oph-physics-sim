"""Independent verifier for the source-derived causal-order receipt.

The producer and capture modules are intentionally not imported. Starting
from the embedded raw observer log, this verifier checks transport event and
checkpoint commitments, reconstructs metadata-independent semantic IDs,
regenerates the read-after-write relation, and recomputes every reported
clause, count, hash, rank, and adversarial control.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / "data" / "causal_order" / (
    "source_derived_causal_order_receipt.json"
)
EXPECTED_SCHEMA = "oph.source-derived-causal-order.v2"


class IndependentVerificationError(RuntimeError):
    """Raised when the receipt fails an independent recomputation."""


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


def _raw_canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_raw_canonical_bytes(value)).hexdigest()


def _strict_sha(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    suffix = value.removeprefix("sha256:")
    return len(suffix) == 64 and all(char in "0123456789abcdef" for char in suffix)


def _fail(message: str) -> None:
    raise IndependentVerificationError(message)


def _reconstruct_semantic_projection(
    observer_log: Mapping[str, Any],
    *,
    validate_transport_ids: bool = True,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    raw_events = [dict(row) for row in observer_log.get("events", [])]
    if not raw_events:
        _fail("observer log contains no events")
    support_visibility: dict[str, dict[str, set[str]]] = {}
    for row in observer_log.get("observer_support_visibility_contract", []):
        token = str(row.get("observer_token") or "")
        if not token or token in support_visibility:
            _fail("observer visibility contracts must have unique tokens")
        support_visibility[token] = {
            "carrier_ids": {str(value) for value in row.get("carrier_ids", [])},
            "visible_seam_ids": {
                str(value) for value in row.get("visible_seam_ids", [])
            },
        }
    seam_endpoints: dict[str, set[str]] = {}
    for row in observer_log.get("overlap_seam_endpoint_contract", []):
        seam_id = str(row.get("seam_id") or "")
        endpoints = {str(value) for value in row.get("carrier_ids", [])}
        if not seam_id or seam_id in seam_endpoints or len(endpoints) != 2:
            _fail("overlap seam endpoint contract is malformed")
        seam_endpoints[seam_id] = endpoints
    if not support_visibility or not seam_endpoints:
        _fail("observer overlap-visibility contract is absent")
    if observer_log.get("cross_read_visibility_rule") != (
        "same_round_shared_declared_support_carrier_v1"
    ):
        _fail("unknown cross-read visibility rule")
    raw_to_semantic: dict[str, str] = {}
    raw_records: dict[str, dict[str, Any]] = {}
    processed_raw: dict[str, dict[str, Any]] = {}
    semantic_events: list[dict[str, Any]] = []
    resources: dict[str, dict[str, set[str]]] = {}
    roots: set[str] = set()

    def reference(raw_id: object, field: str) -> str:
        key = str(raw_id)
        if key not in raw_to_semantic:
            _fail(f"{field} names absent or non-prior event {key!r}")
        return raw_to_semantic[key]

    for index, row in enumerate(raw_events):
        raw_id = str(row.get("event_id") or "")
        if not _strict_sha(raw_id) or raw_id in raw_to_semantic:
            _fail("transport event IDs must be unique sha256 values")
        material = {key: value for key, value in row.items() if key != "event_id"}
        if validate_transport_ids and _raw_sha256(material) != raw_id:
            _fail(f"transport event commitment mismatch at index {index}")
        kind = str(row.get("kind") or "")
        observer = str(row.get("observer_token") or "")
        if not observer:
            _fail(f"observer token missing at event {index}")
        reads: set[str] = set()
        footprint: list[str] = []
        payload: dict[str, Any] = {"event_kind": kind}
        if kind == "RECORD_COMMIT":
            carrier = str(row.get("carrier_id") or "")
            port = int(row.get("port", -1))
            full_state = row.get("full_port_state")
            if not isinstance(full_state, list) or len(full_state) != 12:
                _fail("record does not bind a twelve-port state")
            if _raw_sha256(full_state) != row.get("full_port_state_sha256"):
                _fail("record full-state hash mismatch")
            contract = support_visibility.get(observer)
            if contract is None or carrier not in contract["carrier_ids"]:
                _fail("record carrier is outside observer support")
            footprint = [f"{carrier}:port-{item:02d}" for item in range(12)]
            source_reads = {
                f"source-state:{row['source_state_root']}:{carrier}:port-{item:02d}"
                for item in range(12)
            }
            reads.update(source_reads)
            roots.update(source_reads)
            applied_feedback_raw = row.get("applied_feedback_event_id")
            applied_feedback_id: str | None = None
            if applied_feedback_raw is not None:
                applied_feedback_id = reference(
                    applied_feedback_raw, "applied_feedback_event_id"
                )
                prior_feedback = processed_raw.get(str(applied_feedback_raw))
                if prior_feedback is None or prior_feedback.get("kind") != (
                    "LOCAL_FEEDBACK"
                ):
                    _fail("applied feedback does not name prior local feedback")
                if prior_feedback.get("observer_token") != observer:
                    _fail("record consumes feedback from another observer")
                if int(
                    prior_feedback["observed_action_material_next_port"]
                ) != port:
                    _fail("record port does not replay consumed feedback action")
                reads.add(f"local-action:{applied_feedback_id}")
            payload.update(
                {
                    "carrier_id": carrier,
                    "port": port,
                    "sample": int(row["sample"]),
                    "record_cycle": int(row["record_cycle"]),
                    "port_value": float(row["port_value"]),
                    "full_port_state_sha256": str(row["full_port_state_sha256"]),
                    "source_state_root": str(row["source_state_root"]),
                    "applied_feedback_event_id": applied_feedback_id,
                }
            )
            raw_records[raw_id] = row
        elif kind == "READBACK":
            raw_record = str(row.get("record_event_id") or "")
            record_id = reference(raw_record, "record_event_id")
            if raw_record not in raw_records:
                _fail("readback record_event_id does not name a record")
            committed_hash = str(
                raw_records[raw_record]["full_port_state_sha256"]
            )
            recomputed_hash = str(row["recomputed_full_port_state_sha256"])
            signature_matches = recomputed_hash == committed_hash
            if signature_matches != bool(
                row["record_signature_matches_source_field"]
            ):
                _fail("readback signature verdict is inconsistent")
            if not signature_matches:
                _fail("readback does not authenticate its named record")
            sample = int(row["sample"])
            if int(raw_records[raw_record]["sample"]) != sample:
                _fail("readback does not name its same-round record")
            raw_cross_ids = [
                str(value) for value in row.get("cross_read_record_event_ids", [])
            ]
            cross_hashes = [
                str(value)
                for value in row.get("cross_read_record_state_sha256s", [])
            ]
            if len(raw_cross_ids) != len(cross_hashes):
                _fail("cross-read ID/hash lengths differ")
            witnesses = [
                dict(value)
                for value in row.get("cross_read_overlap_witnesses", [])
            ]
            if len(witnesses) != len(raw_cross_ids):
                _fail("cross-read version/witness lengths differ")
            reader_contract = support_visibility.get(observer)
            if reader_contract is None:
                _fail("readback observer has no visibility contract")
            cross_versions: list[dict[str, Any]] = []
            for raw_cross, state_hash, witness in zip(
                raw_cross_ids, cross_hashes, witnesses, strict=True
            ):
                cross_id = reference(raw_cross, "cross_read_record_event_ids")
                record = raw_records.get(raw_cross)
                if record is None:
                    _fail("cross-read ID does not name a prior record")
                if str(record["full_port_state_sha256"]) != state_hash:
                    _fail("cross-read committed-state hash mismatch")
                if int(record["sample"]) != sample:
                    _fail("cross-read record is not from the same round")
                cross_observer = str(record["observer_token"])
                cross_carrier = str(record["carrier_id"])
                cross_contract = support_visibility.get(cross_observer)
                witness_seams = [
                    str(value) for value in witness.get(
                        "shared_visible_seam_ids", []
                    )
                ]
                if (
                    str(witness.get("record_event_id")) != raw_cross
                    or str(witness.get("record_observer_token"))
                    != cross_observer
                    or str(witness.get("record_carrier_id")) != cross_carrier
                    or witness.get("visibility_witness_kind")
                    != "shared_declared_support_carrier"
                    or cross_contract is None
                    or cross_carrier not in reader_contract["carrier_ids"]
                    or cross_carrier not in cross_contract["carrier_ids"]
                    or witness_seams != sorted(set(witness_seams))
                ):
                    _fail("cross-read overlap witness is malformed")
                for seam_id in witness_seams:
                    if (
                        seam_id not in reader_contract["visible_seam_ids"]
                        or seam_id not in cross_contract["visible_seam_ids"]
                        or cross_carrier not in seam_endpoints.get(seam_id, set())
                    ):
                        _fail("cross-read seam witness is not declared visible")
                cross_versions.append(
                    {
                        "record_event_id": cross_id,
                        "committed_full_port_state_sha256": state_hash,
                        "record_observer_token": cross_observer,
                        "record_carrier_id": cross_carrier,
                        "visibility_witness_kind": (
                            "shared_declared_support_carrier"
                        ),
                        "shared_visible_seam_ids": witness_seams,
                    }
                )
            reads.add(f"record:{record_id}")
            reads.update(
                f"record:{value['record_event_id']}" for value in cross_versions
            )
            payload.update(
                {
                    "sample": sample,
                    "record_event_id": record_id,
                    "cross_read_record_versions": sorted(
                        cross_versions,
                        key=lambda value: (
                            value["record_event_id"],
                            value["committed_full_port_state_sha256"],
                            value["record_observer_token"],
                        ),
                    ),
                    "recomputed_full_port_state_sha256": recomputed_hash,
                    "record_signature_matches_source_field": signature_matches,
                }
            )
        elif kind == "LOCAL_FEEDBACK":
            readback_id = reference(
                row.get("readback_event_id"), "readback_event_id"
            )
            record_id = reference(
                row.get("action_input_record_event_id"),
                "action_input_record_event_id",
            )
            reads.update({f"readback:{readback_id}", f"record:{record_id}"})
            sample = int(row["sample"])
            raw_readback = processed_raw[str(row["readback_event_id"])]
            raw_record = processed_raw[str(row["action_input_record_event_id"])]
            if int(raw_readback["sample"]) != sample or int(
                raw_record["sample"]
            ) != sample:
                _fail("feedback inputs do not belong to its source round")
            payload.update(
                {
                    "sample": sample,
                    "readback_event_id": readback_id,
                    "action_input_record_event_id": record_id,
                    "predicted_action": str(row["predicted_action"]),
                    "observed_action": str(row["observed_action"]),
                    "ablated_action": str(row["ablated_action"]),
                    "predicted_action_material_next_port": int(
                        row["predicted_action_material_next_port"]
                    ),
                    "observed_action_material_next_port": int(
                        row["observed_action_material_next_port"]
                    ),
                    "ablated_action_material_next_port": int(
                        row["ablated_action_material_next_port"]
                    ),
                    "observed_action_recomputed_from_record": bool(
                        row["observed_action_recomputed_from_record"]
                    ),
                    "observed_action_recomputed_from_source_field": bool(
                        row["observed_action_recomputed_from_source_field"]
                    ),
                }
            )
        else:
            _fail(f"unknown observer event kind {kind!r}")
        identity_material = {
            "schema": "oph.semantic-source-event-id.v1",
            "canonical_semantic_payload": payload,
            "observer_token": observer,
            "visible_footprint": sorted(footprint),
            "read_resource_ids": sorted(reads),
        }
        semantic_id = _raw_sha256(identity_material)
        if semantic_id in resources:
            _fail("semantic event identity collision")
        resource_kind = {
            "RECORD_COMMIT": "record",
            "READBACK": "readback",
            "LOCAL_FEEDBACK": "local-action",
        }[kind]
        writes = {f"{resource_kind}:{semantic_id}"}
        raw_to_semantic[raw_id] = semantic_id
        resources[semantic_id] = {"reads": reads, "writes": writes}
        semantic_events.append(
            {
                "event_key": semantic_id,
                "canonical_semantic_payload": payload,
                "observer_token": observer,
                "visible_footprint": footprint,
                "parent_event_ids": [],
                "read_resource_ids": sorted(reads),
                "write_resource_ids": sorted(writes),
                "source_sequence_index": index,
            }
        )
        processed_raw[raw_id] = row

    writer_of: dict[str, str] = {}
    for event in semantic_events:
        for resource in event["write_resource_ids"]:
            if resource in writer_of:
                _fail(f"two semantic writers for {resource!r}")
            writer_of[resource] = str(event["event_key"])
    shared: dict[tuple[str, str], list[str]] = {}
    for event in semantic_events:
        child = str(event["event_key"])
        for resource in event["read_resource_ids"]:
            parent = writer_of.get(resource)
            if parent is None:
                if resource not in roots:
                    _fail(f"unrooted semantic read {resource!r}")
                continue
            if parent == child:
                _fail("semantic self-read detected")
            shared.setdefault((parent, child), []).append(resource)
    by_key = {str(event["event_key"]): event for event in semantic_events}
    sequence = {
        str(event["event_key"]): int(event["source_sequence_index"])
        for event in semantic_events
    }

    def edge(parent: str, child: str, witnesses: list[str]) -> dict[str, Any]:
        material = {
            "parent_event_id": parent,
            "child_event_id": child,
            "observer_token": by_key[child]["observer_token"],
            "parent_sequence_index": sequence[parent],
            "child_sequence_index": sequence[child],
            "shared_resource_ids": sorted(witnesses),
        }
        return {**material, "edge_id": _raw_sha256(material)}

    generated: list[dict[str, Any]] = []
    for (parent, child), witnesses in sorted(shared.items()):
        by_key[child]["parent_event_ids"].append(parent)
        generated.append(edge(parent, child, witnesses))
    for event in semantic_events:
        event["parent_event_ids"].sort()
    declared: list[dict[str, Any]] = []
    for row in raw_events:
        child = raw_to_semantic[str(row["event_id"])]
        for raw_parent in row.get("parents", []):
            parent = raw_to_semantic.get(str(raw_parent))
            if parent is None:
                _fail("declared parent names an absent transport event")
            witnesses = sorted(
                resources[parent]["writes"] & resources[child]["reads"]
            )
            declared.append(edge(parent, child, witnesses))
    generated.sort(key=lambda row: (row["parent_event_id"], row["child_event_id"]))
    declared.sort(key=lambda row: (row["parent_event_id"], row["child_event_id"]))
    return semantic_events, generated, declared, sorted(roots)


def _edge_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [
        {
            "parent_event_id": str(row["parent_event_id"]),
            "child_event_id": str(row["child_event_id"]),
            "shared_resource_ids": sorted(
                str(value) for value in row["shared_resource_ids"]
            ),
        }
        for row in rows
    ]
    result.sort(key=lambda row: (row["parent_event_id"], row["child_event_id"]))
    return result


def _regenerate_edges(
    events: list[dict[str, Any]], roots: list[str]
) -> list[dict[str, Any]]:
    writer_of: dict[str, str] = {}
    for event in events:
        for resource in event["write_resource_ids"]:
            if resource in writer_of:
                _fail(f"two writers for resource {resource!r}")
            writer_of[str(resource)] = str(event["event_key"])
    root_set = set(roots)
    shared: dict[tuple[str, str], list[str]] = {}
    for event in events:
        child = str(event["event_key"])
        for resource in event["read_resource_ids"]:
            resource = str(resource)
            parent = writer_of.get(resource)
            if parent is None:
                if resource not in root_set:
                    _fail(f"missing nonroot writer for resource {resource!r}")
                continue
            if parent == child:
                _fail("event reads its own write")
            shared.setdefault((parent, child), []).append(resource)
    result = [
        {
            "parent_event_id": parent,
            "child_event_id": child,
            "shared_resource_ids": sorted(resources),
        }
        for (parent, child), resources in shared.items()
    ]
    result.sort(key=lambda row: (row["parent_event_id"], row["child_event_id"]))
    return result


def _longest_path(events: list[str], edges: list[dict[str, Any]]) -> tuple[bool, int]:
    indegree = {event: 0 for event in events}
    children: dict[str, list[str]] = {event: [] for event in events}
    rank = {event: 0 for event in events}
    for row in edges:
        parent = row["parent_event_id"]
        child = row["child_event_id"]
        if parent not in indegree or child not in indegree:
            _fail("generated edge names absent event")
        indegree[child] += 1
        children[parent].append(child)
    frontier = sorted(key for key, value in indegree.items() if value == 0)
    visited = 0
    while frontier:
        node = frontier.pop()
        visited += 1
        for child in children[node]:
            rank[child] = max(rank[child], rank[node] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                frontier.append(child)
    return visited == len(events), max(rank.values(), default=0)


def _checkpoint_audit(observer_log: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    events = list(observer_log["events"])
    if _raw_sha256(events) != observer_log.get("event_log_sha256"):
        _fail("observer event_log_sha256 mismatch")
    counts = {
        "record_count": sum(row.get("kind") == "RECORD_COMMIT" for row in events),
        "readback_count": sum(row.get("kind") == "READBACK" for row in events),
        "feedback_count": sum(row.get("kind") == "LOCAL_FEEDBACK" for row in events),
    }
    for field, expected in counts.items():
        if observer_log.get(field) != expected:
            _fail(f"observer {field} mismatch")
    checkpoint = dict(observer_log.get("checkpoint") or {})
    declared_id = checkpoint.pop("checkpoint_id", None)
    if declared_id != _raw_sha256(checkpoint):
        _fail("checkpoint_id mismatch")
    if checkpoint.get("requested_checkpoint_interval") != config.get(
        "checkpoint_interval"
    ):
        _fail("checkpoint interval not bound to config")
    if checkpoint.get("checkpoint_interval_unit") != "complete_source_rounds":
        _fail("checkpoint is not placed on complete source rounds")
    round_count = int(config.get("observer_samples", -1))
    cut_rounds = int(checkpoint.get("cut_round_count", -1))
    if (
        observer_log.get("source_round_count") != round_count
        or cut_rounds != min(int(config["checkpoint_interval"]), round_count - 1)
        or checkpoint.get("next_round_index") != cut_rounds
    ):
        _fail("checkpoint round boundary does not match config")
    phase_rank = {"RECORD_COMMIT": 0, "READBACK": 1, "LOCAL_FEEDBACK": 2}
    ordering = [
        (int(row["sample"]), phase_rank[str(row["kind"])]) for row in events
    ]
    if ordering != sorted(ordering):
        _fail("observer events are not serialized by complete source phases")
    cut_id = checkpoint.get("cut_event_id")
    indices = [
        index for index, row in enumerate(events) if row.get("event_id") == cut_id
    ]
    if len(indices) != 1:
        _fail("checkpoint cut event is absent or duplicated")
    cut = indices[0] + 1
    if (
        int(events[cut - 1]["sample"]) != cut_rounds - 1
        or str(events[cut - 1]["kind"]) != "LOCAL_FEEDBACK"
        or int(events[cut]["sample"]) != cut_rounds
        or str(events[cut]["kind"]) != "RECORD_COMMIT"
    ):
        _fail("checkpoint cut is not an exact round boundary")
    if checkpoint.get("prefix_root") != _raw_sha256(events[:cut]):
        _fail("checkpoint prefix_root mismatch")
    if checkpoint.get("suffix_root") != _raw_sha256(events[cut:]):
        _fail("checkpoint suffix_root mismatch")
    if checkpoint.get("continuation_event_count") != len(events[cut:]):
        _fail("checkpoint continuation count mismatch")
    if observer_log.get("checkpoint_replay_exact") is not True:
        _fail("checkpoint replay is not exact")
    state = checkpoint.get("saved_continuation_state")
    if not isinstance(state, Mapping):
        _fail("saved continuation state is absent")
    prefix = events[:cut]
    for token in sorted(
        str(row["observer_token"])
        for row in observer_log["observer_support_visibility_contract"]
    ):
        records = [
            row
            for row in prefix
            if row["kind"] == "RECORD_COMMIT" and row["observer_token"] == token
        ]
        feedback = [
            row
            for row in prefix
            if row["kind"] == "LOCAL_FEEDBACK" and row["observer_token"] == token
        ]
        if not records or not feedback:
            _fail("checkpoint prefix lacks complete observer round state")
        last_record = records[-1]
        last_feedback = feedback[-1]
        expected = {
            "last_event_by_observer": str(last_feedback["event_id"]),
            "last_record_carrier_by_observer": str(last_record["carrier_id"]),
            "last_record_event_by_observer": str(last_record["event_id"]),
            "last_record_state_sha256_by_observer": str(
                last_record["full_port_state_sha256"]
            ),
            "next_port_by_observer": int(
                last_feedback["observed_action_material_next_port"]
            ),
        }
        for field, value in expected.items():
            if (state.get(field) or {}).get(token) != value:
                _fail(f"saved continuation state mismatch in {field}")


def _phase_permutation_audit(observer_log: Mapping[str, Any]) -> bool:
    events = [dict(row) for row in observer_log["events"]]
    phase_order = ("RECORD_COMMIT", "READBACK", "LOCAL_FEEDBACK")
    permuted: list[dict[str, Any]] = []
    samples = sorted({int(row["sample"]) for row in events})
    for sample in samples:
        for kind in phase_order:
            permuted.extend(
                sorted(
                    (
                        row
                        for row in events
                        if int(row["sample"]) == sample and row["kind"] == kind
                    ),
                    key=lambda row: str(row["observer_token"]),
                    reverse=True,
                )
            )
    if len(permuted) != len(events):
        _fail("phase permutation did not preserve every raw event")
    alternate_log = dict(observer_log)
    alternate_log["events"] = permuted
    canonical_semantic, canonical_generated, _, _ = (
        _reconstruct_semantic_projection(observer_log)
    )
    permuted_semantic, permuted_generated, _, _ = (
        _reconstruct_semantic_projection(alternate_log)
    )

    def semantic_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    canonical_events = semantic_projection(canonical_semantic)
    permuted_events = semantic_projection(permuted_semantic)
    canonical_edges = _edge_projection(canonical_generated)
    permuted_edges = _edge_projection(permuted_generated)
    expected = {
        "algorithm_id": "reverse_observer_order_within_each_source_round_phase_v1",
        "source_round_count": len(samples),
        "transport_event_material_set_invariant": bool(
            sorted(events, key=lambda row: str(row["event_id"]))
            == sorted(permuted, key=lambda row: str(row["event_id"]))
        ),
        "canonical_semantic_event_set_sha256": _raw_sha256(canonical_events),
        "permuted_semantic_event_set_sha256": _raw_sha256(permuted_events),
        "semantic_event_set_invariant": canonical_events == permuted_events,
        "canonical_source_order_sha256": _raw_sha256(canonical_edges),
        "permuted_source_order_sha256": _raw_sha256(permuted_edges),
        "source_order_invariant": canonical_edges == permuted_edges,
    }
    if observer_log.get("phase_observer_permutation_control") != expected:
        _fail("phase-observer permutation control does not independently replay")
    return bool(
        expected["transport_event_material_set_invariant"]
        and expected["semantic_event_set_invariant"]
        and expected["source_order_invariant"]
    )


def _repair_only_event_carrier_audit(
    embedded: Mapping[str, Any],
) -> dict[str, Any]:
    events = [dict(row) for row in embedded.get("repair_event_material", [])]
    if not events:
        _fail("repair-only event-carrier control has no events")
    writer_of_version: dict[tuple[str, int, int], str] = {}
    roots: set[str] = set()
    edges: set[tuple[str, str, str]] = set()
    for event in events:
        event_id = str(event.get("event_id") or "")
        material = {key: value for key, value in event.items() if key != "event_id"}
        if event_id != _raw_sha256(material):
            _fail("repair event commitment mismatch")
        for read in event.get("read_set", []):
            carrier = str(read["carrier_id"])
            port = int(read["port"])
            version = int(read["version"])
            resource = f"repair-port:{carrier}:{port:02d}:version-{version}"
            writer = writer_of_version.get((carrier, port, version))
            if writer is None:
                if version != 0:
                    _fail("repair event reads an unwritten nonroot version")
                roots.add(resource)
            else:
                edges.add((writer, event_id, resource))
        for write in event.get("write_set", []):
            carrier = str(write["carrier_id"])
            port = int(write["port"])
            expected = int(write["expected_version"])
            committed = int(write["committed_version"])
            key = (carrier, port, committed)
            if committed != expected + 1 or key in writer_of_version:
                _fail("repair write version continuity is invalid")
            writer_of_version[key] = event_id
    projected = [
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
        "versioned_provenance_edges": projected,
        "versioned_provenance_edge_count": len(projected),
        "distinguished_version_zero_root_count": len(roots),
        "all_reads_are_version_zero_roots": all(
            int(read["version"]) == 0
            for event in events
            for read in event["read_set"]
        ),
        "classification": (
            "REPAIR_ONLY_EVENT_CARRIER_IS_ANTICHAIN"
            if not projected
            else "REPAIR_ONLY_EVENT_CARRIER_HAS_VERSIONED_DEPENDENCIES"
        ),
        "physical_causet_promotion_allowed": False,
        "required_model_change": (
            "eventize_and_interleave_local_recurrent_propagation_with_"
            "versioned_seam_repair_so_state_can_transport_between_seams"
        ),
    }


def _independent_controls(
    observer_log: Mapping[str, Any],
    semantic: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    roots: list[str],
) -> dict[str, bool]:
    view = [
        {
            "event_key": event["event_key"],
            "read_resource_ids": list(event["read_resource_ids"]),
            "write_resource_ids": list(event["write_resource_ids"]),
        }
        for event in semantic
    ]
    writers = [event for event in view if event["write_resource_ids"]]
    rotated = copy.deepcopy(view)
    write_sets = [event["write_resource_ids"] for event in writers]
    rotation = {
        event["event_key"]: write_sets[(index + 1) % len(write_sets)]
        for index, event in enumerate(writers)
    }
    for event in rotated:
        if event["event_key"] in rotation:
            event["write_resource_ids"] = rotation[event["event_key"]]
    try:
        writer_permutation = _regenerate_edges(rotated, roots) != generated
    except IndependentVerificationError:
        writer_permutation = True
    duplicated = copy.deepcopy(view)
    duplicated[1]["write_resource_ids"] = duplicated[0]["write_resource_ids"]
    try:
        _regenerate_edges(duplicated, roots)
        duplicate_refused = False
    except IndependentVerificationError:
        duplicate_refused = True
    orphaned = copy.deepcopy(view)
    orphaned[0]["read_resource_ids"] = ["unrooted:missing-writer"]
    try:
        _regenerate_edges(orphaned, roots)
        missing_refused = False
    except IndependentVerificationError:
        missing_refused = True
    parent_mutation = copy.deepcopy(observer_log)
    declared_child = next(
        row for row in parent_mutation["events"] if row.get("parents")
    )
    declared_child["parents"] = []
    parent_semantic, parent_generated, parent_declared, _ = (
        _reconstruct_semantic_projection(
            parent_mutation, validate_transport_ids=False
        )
    )
    order_mutation = copy.deepcopy(observer_log)
    record = next(
        row
        for row in order_mutation["events"]
        if row.get("kind") == "RECORD_COMMIT"
    )
    record["record_order_previous_event_ids"] = ["sha256:" + "f" * 64]
    order_semantic, order_generated, _, _ = _reconstruct_semantic_projection(
        order_mutation, validate_transport_ids=False
    )
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
        _reconstruct_semantic_projection(
            state_mutation, validate_transport_ids=False
        )
        state_refused = False
    except IndependentVerificationError:
        state_refused = True
    return {
        "writer_permutation_changes_edges": bool(writer_permutation),
        "single_writer_mutation_is_refused": bool(duplicate_refused),
        "missing_nonroot_writer_is_refused": bool(missing_refused),
        "declared_parent_mutation_leaves_semantic_order_unchanged": bool(
            parent_semantic == semantic
            and _edge_projection(parent_generated) == generated
            and _edge_projection(parent_declared) != generated
        ),
        "record_order_mutation_leaves_semantic_order_unchanged": bool(
            order_semantic == semantic
            and _edge_projection(order_generated) == generated
        ),
        "record_full_state_hash_mutation_is_refused": state_refused,
        "phase_observer_permutation_leaves_semantic_order_unchanged": (
            _phase_permutation_audit(observer_log)
        ),
    }


def verify_receipt(path: Path | str = DEFAULT_RECEIPT) -> dict[str, Any]:
    receipt = json.loads(Path(path).read_text(encoding="ascii"))
    if receipt.get("schema") != EXPECTED_SCHEMA:
        _fail(f"unexpected schema {receipt.get('schema')!r}")
    body = {key: value for key, value in receipt.items() if key != "report_sha256"}
    if _sha256(body) != receipt.get("report_sha256"):
        _fail("report_sha256 does not match the receipt body")
    config = receipt.get("config")
    if not isinstance(config, Mapping) or _sha256(config) != receipt.get(
        "config_sha256"
    ):
        _fail("config_sha256 mismatch")
    observer_log = receipt.get("observer_log_material")
    if not isinstance(observer_log, Mapping):
        _fail("observer_log_material is absent")
    if _sha256(observer_log) != receipt.get("observer_log_material_sha256"):
        _fail("observer_log_material_sha256 mismatch")
    _checkpoint_audit(observer_log, config)
    if observer_log.get("event_log_sha256") != receipt.get(
        "observer_event_log_sha256"
    ):
        _fail("observer event-log binding mismatch")

    semantic, generated_raw, declared_raw, roots = (
        _reconstruct_semantic_projection(observer_log)
    )
    generated = _edge_projection(generated_raw)
    declared = _edge_projection(declared_raw)
    if semantic != receipt.get("semantic_events"):
        _fail("semantic events do not match independent raw-log reconstruction")
    if roots != receipt.get("distinguished_source_resource_ids"):
        _fail("distinguished source roots mismatch")
    if generated != receipt.get("generated_edges"):
        _fail("generated edges do not match independent reconstruction")
    if declared != receipt.get("declared_edges"):
        _fail("declared edges do not match independent reconstruction")
    if _regenerate_edges(semantic, roots) != generated:
        _fail("embedded semantic resources do not replay generated edges")
    if len(semantic) != receipt.get("event_count"):
        _fail("event_count mismatch")
    if len(generated) != receipt.get("generated_edge_count"):
        _fail("generated_edge_count mismatch")
    if len(declared) != receipt.get("declared_edge_count"):
        _fail("declared_edge_count mismatch")
    if _sha256(generated) != receipt.get("generated_edges_sha256"):
        _fail("generated_edges_sha256 mismatch")
    if _sha256(declared) != receipt.get("declared_edges_sha256"):
        _fail("declared_edges_sha256 mismatch")

    generated_pairs = {
        (row["parent_event_id"], row["child_event_id"]) for row in generated
    }
    declared_pairs = {
        (row["parent_event_id"], row["child_event_id"]) for row in declared
    }
    declared_only = sorted(declared_pairs - generated_pairs)
    generated_only = sorted(generated_pairs - declared_pairs)
    empty_witnesses = sorted(
        (row["parent_event_id"], row["child_event_id"])
        for row in declared
        if not row["shared_resource_ids"]
    )
    clause = receipt.get("byte_identity_clause") or {}
    byte_identical = _canonical_bytes(generated) == _canonical_bytes(declared)
    expected_clause = {
        "scope": (
            "canonical_projected_provenance_edge_rows_on_bounded_"
            "source_observer_instrumentation_log"
        ),
        "comparison_representation": (
            "sorted_parent_child_shared_resource_rows_without_transport_"
            "sequence_or_edge_id_fields"
        ),
        "event_count": len(semantic),
        "byte_identical": byte_identical,
        "verdict": "ATTAINED" if byte_identical else "NOT_ATTAINED",
        "declared_only_pair_count": len(declared_only),
        "generated_only_pair_count": len(generated_only),
        "declared_only_pairs": [list(pair) for pair in declared_only],
        "generated_only_pairs": [list(pair) for pair in generated_only],
        "declared_edges_without_read_after_write_witness": [
            list(pair) for pair in empty_witnesses
        ],
    }
    if clause != expected_clause:
        _fail("byte-identity clause mismatch")
    expected_status = (
        "SOURCE_DERIVED_CAUSAL_ORDER_BYTE_IDENTITY_ATTAINED__PHYSICAL_ATTACHMENT_OPEN"
        if byte_identical
        else "SOURCE_DERIVED_CAUSAL_ORDER_BYTE_IDENTITY_NOT_ATTAINED__PHYSICAL_ATTACHMENT_OPEN"
    )
    if receipt.get("status") != expected_status:
        _fail("status does not match byte identity")

    event_keys = [str(event["event_key"]) for event in semantic]
    acyclic, rank_max = _longest_path(event_keys, generated)
    if receipt.get("generated_acyclic") is not acyclic:
        _fail("generated_acyclic mismatch")
    if receipt.get("generated_longest_path_rank_max") != rank_max:
        _fail("longest-path rank mismatch")
    sequence = {
        str(event["event_key"]): int(event["source_sequence_index"])
        for event in semantic
    }
    sequence_compatible = all(
        sequence[row["parent_event_id"]] < sequence[row["child_event_id"]]
        for row in generated
    )
    if receipt.get("sequence_compatible") is not sequence_compatible:
        _fail("sequence compatibility mismatch")
    observer = {
        str(event["event_key"]): str(event["observer_token"])
        for event in semantic
    }
    cross_count = sum(
        observer[row["parent_event_id"]] != observer[row["child_event_id"]]
        for row in generated
    )
    if receipt.get("cross_observer_edge_count") != cross_count:
        _fail("cross-observer edge count mismatch")

    controls = _independent_controls(observer_log, semantic, generated, roots)
    if receipt.get("negative_controls") != controls:
        _fail("negative controls do not match independent mutation replay")
    controls_fail_closed = all(controls.values())
    if receipt.get("controls_fail_closed") is not controls_fail_closed:
        _fail("controls_fail_closed mismatch")
    repair_control = receipt.get("repair_only_event_carrier_control")
    if not isinstance(repair_control, Mapping) or (
        _repair_only_event_carrier_audit(repair_control) != repair_control
    ):
        _fail("repair-only event-carrier control mismatch")
    if receipt.get("capture_ancestry_matches_generated") is not True:
        _fail("capture ancestry was not certified as generated")
    binding = receipt.get("source_capture_binding")
    if (
        not isinstance(binding, Mapping)
        or set(binding)
        != {
            "capture_sha256",
            "source_root_sha256",
            "postrun_capture_sha256",
        }
        or not all(_strict_sha(value) for value in binding.values())
    ):
        _fail("source capture binding is malformed")
    if receipt.get("physical_promotion_allowed") is not False:
        _fail("physical_promotion_allowed must be false")
    if receipt.get("event_carrier_scope") != (
        "observer_instrumentation_history_over_source_state_snapshots"
    ):
        _fail("event carrier scope is missing or overstated")
    if receipt.get("underlying_repair_transactions_promoted_as_events") is not False:
        _fail("underlying repair transactions must remain explicitly unpromoted")
    expected_receipt = bool(
        byte_identical
        and acyclic
        and sequence_compatible
        and controls_fail_closed
        and cross_count > 0
    )
    if receipt.get("SOURCE_DERIVED_CAUSAL_ORDER_RECEIPT") is not expected_receipt:
        _fail("receipt flag does not match all required clauses")
    return {
        "receipt": True,
        "byte_identical": byte_identical,
        "generated_edge_count": len(generated),
        "declared_edge_count": len(declared),
        "cross_observer_edge_count": cross_count,
        "generated_only_pair_count": len(generated_only),
        "declared_only_pair_count": len(declared_only),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Independently verify the source-derived causal-order receipt."
    )
    parser.add_argument("--receipt", default=str(DEFAULT_RECEIPT))
    args = parser.parse_args()
    try:
        result = verify_receipt(args.receipt)
    except IndependentVerificationError as error:
        print(f"REFUSED: {error}")
        return 1
    print(
        "verified: "
        f"byte_identical={result['byte_identical']} "
        f"generated={result['generated_edge_count']} "
        f"declared={result['declared_edge_count']} "
        f"cross_observer={result['cross_observer_edge_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
