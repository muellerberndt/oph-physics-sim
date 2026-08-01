"""Exact source packet for the issue-655 internal port-transfer boundary.

The canonical finite source federation pairs every carrier exactly once at
each of its twelve local port labels.  A full repair cycle therefore realizes
twelve fixed-point-free matching involutions and the associated averaging
projectors.  The source record loop commits and rereads, in process, the
resulting full twelve-port snapshot for every carrier in the bounded diagnostic
below.

These are internal federation seam operations.  They are not spatial
translations.  In particular, the matching at a carrier port is not the
inverse of the matching at its icosahedral antipode, and no nontrivial quotient
of the eight source carriers both preserves all twelve matchings and repairs
that inverse relation.  No frame, boost, sector, or physical-readout binding is
introduced here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from oph_fpe.bulk.physical_h3_kms_source_capture import (
    capture_physical_source,
    verify_physical_source_capture,
)


SCHEMA = "oph.vertex12-atomic-port-transfer-subpacket.v1"
VERIFICATION_SCHEMA = "oph.vertex12-atomic-port-transfer-subpacket-verification.v1"
STATUS = (
    "INTERNAL_VERTEX12_SEAM_MATCHING_PROJECTORS_AND_IN_PROCESS_SNAPSHOT_REREAD_ATTAINED__"
    "SPATIAL_PHYSICAL_BRIDGE_OPEN"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "data/repair_closure/vertex12_atomic_port_transfer_receipt.json"
)
CARRIER_MANIFEST = (
    REPOSITORY_ROOT / "tests/fixtures/echosahedral_federation_reference.json"
)
PRODUCER_PATH = Path(__file__).resolve()
INDEPENDENT_VERIFIER_PATH = (
    REPOSITORY_ROOT
    / "oph_fpe/dynamics/verify_vertex12_atomic_port_transfer_independent.py"
)
TEST_PATH = REPOSITORY_ROOT / "tests/test_vertex12_atomic_port_transfer.py"

AUDITED_SURFACES = {
    "screen_port_reference": REPOSITORY_ROOT / "oph_fpe/core/screen_ports.py",
    "federation": REPOSITORY_ROOT / "oph_fpe/core/echosahedral_federation.py",
    "local_dynamics": REPOSITORY_ROOT / "oph_fpe/core/echosahedral_dynamics.py",
    "source_capture": REPOSITORY_ROOT
    / "oph_fpe/bulk/physical_h3_kms_source_capture.py",
    "repair_to_propagation_classifier": (
        REPOSITORY_ROOT / "oph_fpe/dynamics/port_repair_propagation_bridge.py"
    ),
    "producer_capability_matrix": (
        REPOSITORY_ROOT / "data/common_reserve/producer_capability_matrix.json"
    ),
    "carrier_manifest": CARRIER_MANIFEST,
    "icosahedral_geometry": REPOSITORY_ROOT / "oph_fpe/core/icosahedral.py",
    "covariant_overlap": REPOSITORY_ROOT / "oph_fpe/gauge/covariant_overlap.py",
    "finite_groups": REPOSITORY_ROOT / "oph_fpe/finite_groups.py",
}

SERIALIZED_DECIMAL_PLACES = 15
SERIALIZED_ABSOLUTE_TOLERANCE = 2e-15
REPAIR_EVENT_KEYS = {
    "cycle",
    "transaction_index",
    "seam_id",
    "read_set",
    "write_set",
    "mismatch_before",
    "mismatch_after",
    "strict_descent",
    "update",
    "event_id",
}
READ_ROW_KEYS = {"carrier_id", "port", "version", "value"}
WRITE_ROW_KEYS = {
    "carrier_id",
    "port",
    "expected_version",
    "committed_version",
    "value",
}

# The seed, carrier count, and replica are the canonical defaults.  The one
# full repair cycle and spanning snapshot are diagnostic coverage settings;
# their seam ledger is checked against the default source topology below.
SOURCE_CONFIG: dict[str, Any] = {
    "carrier_count": 8,
    "seed": 20260751,
    "rung": 8,
    "replicate_id": "primary",
    "propagation_steps": 1,
    "cycles": 1,
    "repair_fraction_per_cycle": 1.0,
    "record_commit_cycles": 1,
    "observer_count": 8,
    "observer_support_size": 1,
    "observer_samples": 3,
    "checkpoint_interval": 1,
    "support_refinement_level": 1,
    "geometry_sample_count": 4,
    "snapshot_coverage": "spanning",
}


class PacketError(ValueError):
    """Raised when an upstream source field cannot support the exact packet."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PacketError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_strict_json_object
    )
    if not isinstance(value, dict):
        raise PacketError("receipt JSON root is not an object")
    return value


def _raw_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _port_index(value: str) -> int:
    if len(value) != 3 or value[0] != "p" or not value[1:].isdigit():
        raise PacketError("carrier manifest port label is malformed")
    result = int(value[1:])
    if not 0 <= result < 12:
        raise PacketError("carrier manifest port label is out of range")
    return result


def _antipode_map() -> tuple[int, ...]:
    manifest = json.loads(CARRIER_MANIFEST.read_text(encoding="utf-8"))
    carrier = manifest.get("carrier")
    if not isinstance(carrier, Mapping):
        raise PacketError("carrier manifest block is missing")
    ports = carrier.get("ports")
    edges = carrier.get("edges")
    if not isinstance(ports, list) or [_port_index(item) for item in ports] != list(
        range(12)
    ):
        raise PacketError("carrier manifest port order is not p00 through p11")
    if not isinstance(edges, list) or len(edges) != 30:
        raise PacketError("carrier manifest does not contain thirty edges")
    adjacency = [set() for _ in range(12)]
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise PacketError("carrier edge is malformed")
        left, right = (_port_index(str(item)) for item in edge)
        if left == right or right in adjacency[left]:
            raise PacketError("carrier edge is a loop or duplicate")
        adjacency[left].add(right)
        adjacency[right].add(left)
    if {len(row) for row in adjacency} != {5}:
        raise PacketError("carrier graph is not five-regular")
    antipodes: list[int] = []
    for source in range(12):
        distances = [-1] * 12
        distances[source] = 0
        queue = [source]
        for left in queue:
            for right in sorted(adjacency[left]):
                if distances[right] < 0:
                    distances[right] = distances[left] + 1
                    queue.append(right)
        maximum = max(distances)
        candidates = [
            index for index, value in enumerate(distances) if value == maximum
        ]
        if maximum != 3 or len(candidates) != 1:
            raise PacketError("carrier graph lacks a unique distance-three antipode")
        antipodes.append(candidates[0])
    if any(antipodes[antipodes[index]] != index for index in range(12)):
        raise PacketError("distance-three map is not an involution")
    return tuple(antipodes)


def _matching_permutations(
    bundle: Mapping[str, Any],
) -> tuple[
    tuple[str, ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[str, ...], ...],
    dict[str, tuple[tuple[str, int], tuple[str, int]]],
]:
    carrier_ids_raw = bundle.get("carrier_ids")
    seams = bundle.get("seams")
    if not isinstance(carrier_ids_raw, list) or not isinstance(seams, list):
        raise PacketError("source federation bundle lacks carriers or seams")
    carrier_ids = tuple(str(item) for item in carrier_ids_raw)
    if len(carrier_ids) != 8 or len(set(carrier_ids)) != 8:
        raise PacketError("diagnostic must contain eight distinct carriers")
    index = {carrier_id: position for position, carrier_id in enumerate(carrier_ids)}
    partners: list[list[int | None]] = [[None] * len(carrier_ids) for _ in range(12)]
    seam_ids: list[list[str]] = [[] for _ in range(12)]
    seam_contracts: dict[str, tuple[tuple[str, int], tuple[str, int]]] = {}
    seen_seams: set[str] = set()
    for seam in seams:
        if not isinstance(seam, Mapping):
            raise PacketError("seam row is malformed")
        seam_id = str(seam.get("seam_id"))
        if seam_id in seen_seams:
            raise PacketError("seam identifier is duplicated")
        seen_seams.add(seam_id)
        left_ports = seam.get("left_ports")
        right_ports = seam.get("right_ports")
        if (
            not isinstance(left_ports, list)
            or not isinstance(right_ports, list)
            or len(left_ports) != 1
            or left_ports != right_ports
            or seam.get("left_to_right_ports") != left_ports
            or seam.get("right_to_left_ports") != right_ports
        ):
            raise PacketError("source seam is not a same-label singleton port seam")
        port = left_ports[0]
        if type(port) is not int or not 0 <= port < 12:
            raise PacketError("source seam port is out of range")
        if (
            seam.get("left_to_right_orientation") != [-1]
            or seam.get("right_to_left_orientation") != [-1]
            or seam.get("collar_kind") != "single_port"
        ):
            raise PacketError("source seam orientation or collar type drifted")
        left_id = str(seam.get("left_carrier_id"))
        right_id = str(seam.get("right_carrier_id"))
        if left_id not in index or right_id not in index or left_id == right_id:
            raise PacketError("source seam endpoint is missing or degenerate")
        left = index[left_id]
        right = index[right_id]
        if partners[port][left] is not None or partners[port][right] is not None:
            raise PacketError("a carrier is used twice at one port")
        partners[port][left] = right
        partners[port][right] = left
        seam_ids[port].append(seam_id)
        seam_contracts[seam_id] = ((left_id, port), (right_id, port))
    if len(seams) != 48:
        raise PacketError("eight-carrier all-port source must contain 48 seams")
    normalized: list[tuple[int, ...]] = []
    normalized_seams: list[tuple[str, ...]] = []
    for port, row in enumerate(partners):
        if any(value is None for value in row):
            raise PacketError(f"port {port} does not pair every carrier")
        permutation = tuple(int(value) for value in row)
        if sorted(permutation) != list(range(len(carrier_ids))):
            raise PacketError(f"port {port} partner map is not a permutation")
        if any(
            permutation[permutation[index]] != index
            for index in range(len(carrier_ids))
        ):
            raise PacketError(f"port {port} partner map is not an involution")
        if any(permutation[index] == index for index in range(len(carrier_ids))):
            raise PacketError(f"port {port} partner map has a fixed point")
        if len(seam_ids[port]) != len(carrier_ids) // 2:
            raise PacketError(f"port {port} does not contain one perfect matching")
        normalized.append(permutation)
        normalized_seams.append(tuple(sorted(seam_ids[port])))
    return carrier_ids, tuple(normalized), tuple(normalized_seams), seam_contracts


def _partition_labels(size: int) -> Iterable[tuple[int, ...]]:
    """Enumerate set partitions as canonical restricted-growth strings."""

    if size < 1:
        return

    def visit(prefix: list[int], maximum: int) -> Iterable[tuple[int, ...]]:
        if len(prefix) == size:
            yield tuple(prefix)
            return
        for label in range(maximum + 2):
            yield from visit([*prefix, label], max(maximum, label))

    yield from visit([0], 0)


def _descended_permutations(
    labels: Sequence[int],
    permutations: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...] | None:
    block_count = max(labels) + 1
    result: list[tuple[int, ...]] = []
    for permutation in permutations:
        descended: list[int | None] = [None] * block_count
        for source, target in enumerate(permutation):
            source_block = labels[source]
            target_block = labels[target]
            if descended[source_block] is None:
                descended[source_block] = target_block
            elif descended[source_block] != target_block:
                return None
        row = tuple(int(item) for item in descended)
        if sorted(row) != list(range(block_count)):
            return None
        result.append(row)
    return tuple(result)


def _inverse(permutation: Sequence[int]) -> tuple[int, ...]:
    result = [0] * len(permutation)
    for source, target in enumerate(permutation):
        result[target] = source
    return tuple(result)


def _quotient_audit(
    permutations: Sequence[Sequence[int]], antipodes: Sequence[int]
) -> dict[str, Any]:
    partition_count = 0
    common_congruences: list[dict[str, Any]] = []
    compatible: list[dict[str, Any]] = []
    for labels in _partition_labels(len(permutations[0])):
        partition_count += 1
        descended = _descended_permutations(labels, permutations)
        if descended is None:
            continue
        row = {
            "block_count": max(labels) + 1,
            "canonical_labels": list(labels),
        }
        common_congruences.append(row)
        inverse_compatible = all(
            descended[antipodes[port]] == _inverse(descended[port])
            for port in range(12)
        )
        if inverse_compatible:
            compatible.append(row)
    nontrivial = [
        row
        for row in common_congruences
        if 1 < row["block_count"] < len(permutations[0])
    ]
    physical_candidates = [row for row in compatible if row["block_count"] > 1]
    return {
        "set_partition_count_checked": partition_count,
        "common_congruence_count": len(common_congruences),
        "common_congruences": common_congruences,
        "nontrivial_proper_common_congruence_count": len(nontrivial),
        "antipodal_inverse_compatible_quotient_count": len(compatible),
        "antipodal_inverse_compatible_quotients": compatible,
        "noncollapsed_antipodal_inverse_compatible_quotient_count": len(
            physical_candidates
        ),
        "only_inverse_compatible_quotient_collapses_all_carriers": bool(
            len(compatible) == 1 and compatible[0]["block_count"] == 1
        ),
    }


def _repair_history_audit(
    dynamics: Mapping[str, Any],
    expected_seams: Mapping[str, tuple[tuple[str, int], tuple[str, int]]],
    carrier_ids: Sequence[str],
) -> dict[str, Any]:
    events = dynamics.get("repair_event_log")
    if not isinstance(events, list) or len(events) != len(expected_seams):
        raise PacketError("full repair event ledger is missing or incomplete")
    event_rows: list[dict[str, Any]] = []
    observed_seams: set[str] = set()
    observed_transaction_indices: set[int] = set()
    terminal_values: dict[tuple[str, int], Any] = {}
    for event in events:
        if not isinstance(event, Mapping) or set(event) != REPAIR_EVENT_KEYS:
            raise PacketError("repair event is malformed")
        transaction_index = event.get("transaction_index")
        if (
            type(event.get("cycle")) is not int
            or event.get("cycle") != 0
            or type(transaction_index) is not int
            or transaction_index in observed_transaction_indices
        ):
            raise PacketError("repair event cycle or transaction index is malformed")
        observed_transaction_indices.add(transaction_index)
        material = dict(event)
        event_id = material.pop("event_id", None)
        if event_id != _sha(material):
            raise PacketError("repair event digest does not replay")
        if type(event.get("seam_id")) is not str:
            raise PacketError("repair history seam identifier is not a string")
        seam_id = event["seam_id"]
        if seam_id not in expected_seams or seam_id in observed_seams:
            raise PacketError("repair history has an unknown or duplicate seam")
        observed_seams.add(seam_id)
        reads = event.get("read_set")
        writes = event.get("write_set")
        if (
            not isinstance(reads, list)
            or not isinstance(writes, list)
            or len(reads) != 2
            or len(writes) != 2
            or any(not isinstance(row, Mapping) or set(row) != READ_ROW_KEYS for row in reads)
            or any(not isinstance(row, Mapping) or set(row) != WRITE_ROW_KEYS for row in writes)
        ):
            raise PacketError("repair event is not a two-endpoint transaction")
        read_keys = [(row.get("carrier_id"), row.get("port")) for row in reads]
        write_keys = [(row.get("carrier_id"), row.get("port")) for row in writes]
        expected_keys = list(expected_seams[seam_id])
        if (
            any(
                type(row.get("carrier_id")) is not str
                or type(row.get("port")) is not int
                or type(row.get("version")) is not int
                for row in reads
            )
            or any(
                type(row.get("carrier_id")) is not str
                or type(row.get("port")) is not int
                or type(row.get("expected_version")) is not int
                or type(row.get("committed_version")) is not int
                for row in writes
            )
            or _canonical_bytes(read_keys) != _canonical_bytes(expected_keys)
            or _canonical_bytes(write_keys) != _canonical_bytes(expected_keys)
            or reads[0].get("port") != reads[1].get("port")
        ):
            raise PacketError("repair transaction changed its endpoint or port domain")
        numeric_values = [
            reads[0].get("value"),
            reads[1].get("value"),
            writes[0].get("value"),
            writes[1].get("value"),
            event.get("mismatch_before"),
            event.get("mismatch_after"),
        ]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric_values
        ):
            raise PacketError("repair transaction contains a non-finite numeric field")
        left_value = float(reads[0]["value"])
        right_value = float(reads[1]["value"])
        left_write = float(writes[0]["value"])
        right_write = float(writes[1]["value"])
        serialized_mean = 0.5 * (left_value + right_value)
        serialized_before = abs(left_value - right_value)
        serialized_after = abs(left_write - right_write)
        if (
            abs(left_write - serialized_mean) > SERIALIZED_ABSOLUTE_TOLERANCE
            or abs(right_write - serialized_mean) > SERIALIZED_ABSOLUTE_TOLERANCE
            or abs(float(event["mismatch_before"]) - serialized_before)
            > SERIALIZED_ABSOLUTE_TOLERANCE
            or abs(float(event["mismatch_after"]) - serialized_after)
            > SERIALIZED_ABSOLUTE_TOLERANCE
            or event.get("strict_descent") is not (serialized_after < serialized_before)
        ):
            raise PacketError(
                "repair transaction fails the serialized arithmetic-mean/descent contract"
            )
        if (
            writes[0].get("value") != writes[1].get("value")
            or event.get("mismatch_after") != 0.0
            or event.get("strict_descent") is not True
            or event.get("update") != "endpoint_arithmetic_mean"
            or any(row.get("version") != 0 for row in reads)
            or any(
                row.get("expected_version") != 0 or row.get("committed_version") != 1
                for row in writes
            )
        ):
            raise PacketError(
                "repair transaction does not realize the atomic mean rule"
            )
        for write in writes:
            key = (str(write["carrier_id"]), int(write["port"]))
            if key in terminal_values:
                raise PacketError(
                    "repair history writes one carrier-port coordinate twice"
                )
            terminal_values[key] = write["value"]
        event_rows.append(
            {
                "seam_id": seam_id,
                "port": int(reads[0]["port"]),
                "event_id": str(event_id),
            }
        )
    if observed_seams != set(expected_seams):
        raise PacketError("repair event ledger does not cover every seam")
    if observed_transaction_indices != set(range(len(events))):
        raise PacketError("repair transaction indices are not exactly contiguous")
    expected_coordinates = {
        (carrier_id, port) for carrier_id in carrier_ids for port in range(12)
    }
    if set(terminal_values) != expected_coordinates:
        raise PacketError(
            "repair history does not write the complete carrier-port field"
        )
    terminal_rows = [
        [terminal_values[(carrier_id, port)] for port in range(12)]
        for carrier_id in carrier_ids
    ]
    if dynamics.get("repair_event_examples_complete") is not True:
        raise PacketError("repair event ledger is only an example window")
    required_receipts = (
        "TRANSACTION_VALIDATION_COMPLETE_READ_CONFLICT_SET_RECEIPT",
        "UNION_PAYLOAD_ATOMIC_REVALIDATION_RECEIPT",
        "REPAIR_ORDER_REPLAY_EXACT_RECEIPT",
        "REPAIR_IDEMPOTENCE_REPLAY_EXACT_RECEIPT",
        "REPAIR_TERMINAL_FIXED_POINT_RECEIPT",
    )
    if any(dynamics.get(key) is not True for key in required_receipts):
        raise PacketError("upstream source repair receipt is not attained")
    event_rows.sort(key=lambda row: row["seam_id"])
    return {
        "event_count": len(event_rows),
        "event_rows": event_rows,
        "event_rows_sha256": _sha(event_rows),
        "terminal_write_state_rows_sha256": _sha(terminal_rows),
        "terminal_write_coordinate_count": len(terminal_values),
        "every_seam_replayed_once": True,
        "every_event_matches_named_federation_seam": True,
        "every_event_matches_atomic_two_endpoint_mean_rule_within_serialized_tolerance": True,
        "serialized_mismatch_and_strict_descent_recomputed": True,
        "serialized_numeric_contract": {
            "decimal_places": SERIALIZED_DECIMAL_PLACES,
            "absolute_tolerance": "2e-15",
            "scope": (
                "each source-ledger scalar is rounded independently; the symbolic "
                "matching/projector identities remain exact"
            ),
        },
        "read_conflict_validation_complete": True,
        "union_atomic_revalidation_complete": True,
        "order_replay_exact": True,
        "idempotence_replay_exact": True,
        "terminal_seam_fixed_point": True,
    }


def _readback_audit(
    dynamics: Mapping[str, Any],
    observer: Mapping[str, Any],
    carrier_ids: Sequence[str],
) -> dict[str, Any]:
    snapshots = dynamics.get("record_state_snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != 1:
        raise PacketError("diagnostic must expose one terminal spanning snapshot")
    snapshot = snapshots[0]
    if not isinstance(snapshot, Mapping) or snapshot.get("cycle") != 0:
        raise PacketError("terminal spanning snapshot has the wrong cycle")
    carrier_rows = snapshot.get("carrier_rows")
    if not isinstance(carrier_rows, list):
        raise PacketError("snapshot carrier rows are missing")
    row_by_carrier: dict[str, Mapping[str, Any]] = {}
    for row in carrier_rows:
        if not isinstance(row, Mapping):
            raise PacketError("snapshot carrier row is malformed")
        carrier_id = str(row.get("carrier_id"))
        state = row.get("full_port_state")
        if (
            carrier_id in row_by_carrier
            or not isinstance(state, list)
            or len(state) != 12
        ):
            raise PacketError("snapshot carrier row is duplicated or incomplete")
        if row.get("full_port_state_sha256") != _sha(state):
            raise PacketError("snapshot carrier state digest does not replay")
        row_by_carrier[carrier_id] = row
    if set(row_by_carrier) != set(carrier_ids):
        raise PacketError("spanning snapshot does not cover every carrier")
    if snapshot.get("carrier_rows_sha256") != _sha(carrier_rows):
        raise PacketError("snapshot carrier-row digest does not replay")

    events = observer.get("events")
    if not isinstance(events, list):
        raise PacketError("observer event ledger is missing")
    records: dict[str, Mapping[str, Any]] = {}
    readbacks: list[Mapping[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            raise PacketError("observer event row is malformed")
        material = dict(event)
        event_id = material.pop("event_id", None)
        if event_id != _sha(material):
            raise PacketError("observer event digest does not replay")
        if event.get("kind") == "RECORD_COMMIT":
            records[str(event_id)] = event
        elif event.get("kind") == "READBACK":
            readbacks.append(event)
    if len(records) != 24 or len(readbacks) != 24:
        raise PacketError("diagnostic record/readback count drifted")
    pair_rows: list[dict[str, Any]] = []
    covered: set[str] = set()
    for readback in readbacks:
        record_id = str(readback.get("record_event_id"))
        if record_id not in records:
            raise PacketError("readback has no committed parent record")
        record = records[record_id]
        carrier_id = str(record.get("carrier_id"))
        source_row = row_by_carrier.get(carrier_id)
        if source_row is None:
            raise PacketError("record carrier is outside the spanning snapshot")
        state = record.get("full_port_state")
        state_digest = record.get("full_port_state_sha256")
        if (
            record.get("record_cycle") != 0
            or state != source_row.get("full_port_state")
            or state_digest != source_row.get("full_port_state_sha256")
            or state_digest != _sha(state)
            or readback.get("carrier_id") != carrier_id
            or readback.get("record_cycle") != 0
            or readback.get("recomputed_full_port_state_sha256") != state_digest
            or readback.get("record_signature_matches_source_field") is not True
            or readback.get("parents") != [record_id]
        ):
            raise PacketError(
                "record/readback is not digest-identical to the source snapshot"
            )
        covered.add(carrier_id)
        pair_rows.append(
            {
                "carrier_id": carrier_id,
                "record_event_id": record_id,
                "state_sha256": str(state_digest),
                "readback_event_id": str(readback.get("event_id")),
            }
        )
    if covered != set(carrier_ids):
        raise PacketError("record/readback pairs do not cover every carrier")
    pair_rows.sort(key=lambda row: (row["carrier_id"], row["record_event_id"]))
    return {
        "snapshot_cycle": 0,
        "visible_state_sha256": str(snapshot.get("visible_state_sha256")),
        "carrier_rows_sha256": str(snapshot.get("carrier_rows_sha256")),
        "terminal_state_rows_sha256": _sha(
            [
                row_by_carrier[carrier_id]["full_port_state"]
                for carrier_id in carrier_ids
            ]
        ),
        "record_count": len(records),
        "readback_count": len(readbacks),
        "covered_carrier_count": len(covered),
        "covered_port_coordinate_count": len(covered) * 12,
        "record_readback_pairs": pair_rows,
        "record_readback_pairs_sha256": _sha(pair_rows),
        "every_carrier_full_port_state_committed": True,
        "every_carrier_full_port_state_reread_in_process": True,
        "record_and_readback_state_digests_identical": True,
        "readback_domain": "post_repair_internal_visible_port_snapshot",
        "readback_mechanism": "in_process_snapshot_lookup_digest_reread",
        "independent_persistence_readback": False,
        "independent_second_producer_readback": False,
        "physical_sector_readout": False,
    }


def _payload() -> dict[str, Any]:
    capture = capture_physical_source(SOURCE_CONFIG)
    source_verification = verify_physical_source_capture(capture)
    if source_verification.get("SOURCE_CAPTURE_REPLAY_RECEIPT") is not True:
        raise PacketError("upstream source capture does not replay")
    source = capture["reports"]["source_observer"]
    if (
        source.get("SOURCE_PATCH_ARCHITECTURE_RECEIPT") is not True
        or source.get("PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT")
        is not True
    ):
        raise PacketError("upstream finite source architecture is not attained")
    bundle = capture["source_artifacts"]["federation_bundle"]
    carrier_ids, permutations, seam_ids, seam_contracts = _matching_permutations(bundle)
    expected_seams = set(seam_contracts)
    antipodes = _antipode_map()

    default_capture = capture_physical_source()
    default_bundle = default_capture["source_artifacts"]["federation_bundle"]
    (
        default_carriers,
        default_permutations,
        default_seams,
        default_seam_contracts,
    ) = _matching_permutations(
        default_bundle
    )
    seam_ledger_matches_default = bool(
        carrier_ids == default_carriers
        and permutations == default_permutations
        and seam_ids == default_seams
        and seam_contracts == default_seam_contracts
    )
    if not seam_ledger_matches_default:
        raise PacketError("coverage config changed the canonical default seam topology")

    port_rows = []
    for port, permutation in enumerate(permutations):
        partner = antipodes[port]
        inverse_partner_relation = permutations[partner] == _inverse(permutation)
        port_rows.append(
            {
                "port": port,
                "carrier_partner_permutation": list(permutation),
                "carrier_partner_permutation_sha256": _sha(list(permutation)),
                "seam_ids": list(seam_ids[port]),
                "seam_ids_sha256": _sha(list(seam_ids[port])),
                "matching_size": len(seam_ids[port]),
                "fixed_point_free": True,
                "permutation": True,
                "transfer_involution_Sp_squared_is_identity": True,
                "transfer_is_self_adjoint_permutation": True,
                "repair_projector_formula": "A_p=(I+S_p)/2 over Q",
                "repair_projector_idempotent": True,
                "repair_projector_self_adjoint": True,
                "antipodal_port": partner,
                "antipodal_map_equals_inverse_transfer": inverse_partner_relation,
            }
        )
    if any(row["antipodal_map_equals_inverse_transfer"] for row in port_rows):
        raise PacketError(
            "canonical source unexpectedly satisfied an antipodal inverse pair"
        )

    repair_history = _repair_history_audit(
        capture["source_artifacts"]["dynamics"], seam_contracts, carrier_ids
    )
    readback = _readback_audit(
        capture["source_artifacts"]["dynamics"],
        capture["source_artifacts"]["observer_log"],
        carrier_ids,
    )
    if (
        repair_history["terminal_write_state_rows_sha256"]
        != readback["terminal_state_rows_sha256"]
    ):
        raise PacketError("terminal repair writes do not equal the readback snapshot")
    repair_history["terminal_write_state_matches_readback_snapshot"] = True
    quotient = _quotient_audit(permutations, antipodes)
    if not (
        quotient["set_partition_count_checked"] == 4140
        and quotient["common_congruence_count"] == 2
        and quotient["nontrivial_proper_common_congruence_count"] == 0
        and quotient["only_inverse_compatible_quotient_collapses_all_carriers"]
        and quotient["noncollapsed_antipodal_inverse_compatible_quotient_count"] == 0
    ):
        raise PacketError("canonical source quotient classification drifted")

    source_hashes = capture["source_hashes"]
    payload = {
        "schema": SCHEMA,
        "issue": 655,
        "status": STATUS,
        "source_config": copy.deepcopy(capture["input_config"]),
        "source_capture_binding": {
            "schema": capture["schema"],
            "artifact_type": capture["artifact_type"],
            "source_capture_replay_receipt": True,
            "source_engine_independently_reimplemented": False,
            "finite_source_architecture_receipt": True,
            "finite_federation_realization_receipt": True,
            "canonical_default_seed": True,
            "canonical_default_replica": True,
            "coverage_config_seam_ledger_matches_default_source": True,
            "input_config_sha256": source_hashes["input_config"],
            "federation_bundle_sha256": source_hashes["federation"],
            "source_state_root_sha256": capture["source_artifacts"][
                "source_state_root_sha256"
            ],
            "repair_log_sha256": source_hashes["repair_log"],
            "observer_log_sha256": source_hashes["observer_log"],
            "postrun_capture_sha256": source_hashes["postrun_capture"],
            "target_or_comparison_token_hits": source["source_forbidden_target_hits"],
        },
        "audited_surface_pins": {
            name: _raw_pin(path) for name, path in sorted(AUDITED_SURFACES.items())
        },
        "atomic_transfer_operator": {
            "domain": "internal_federation_visible_port_fiber_Q^(8_times_12)",
            "carrier_ids": list(carrier_ids),
            "carrier_count": len(carrier_ids),
            "port_count": 12,
            "seam_count": len(expected_seams),
            "port_rows": port_rows,
            "port_rows_sha256": _sha(port_rows),
            "complete_matching_on_every_port": True,
            "all_twelve_transfer_involutions_exact": True,
            "all_twelve_rational_repair_projectors_exact": True,
            "exact_symbolic_matching_and_projector_algebra": True,
            "block_diagonal_full_repair_formula": "A=direct_sum_p (I+S_p)/2",
            "source_native_internal_seam_partner_operator_receipt": True,
            "source_native_spatial_translation_receipt": False,
        },
        "source_history_replay": repair_history,
        "post_repair_in_process_snapshot_reread": readback,
        "quotient_and_spatial_boundary": {
            "carrier_antipode_map": list(antipodes),
            "all_six_antipodal_transfer_pairs_fail_inverse_relation": True,
            "quotient_enumeration": quotient,
            "internal_seam_transfer_is_spatial_translation": False,
            "directed_antipode_inverse_transport_receipt": False,
            "noncollapsed_quotient_site_map_receipt": False,
            "a5_covariant_site_action_receipt": False,
            "coherent_frame_transport_receipt": False,
            "orientation_transport_receipt": False,
            "declared_boost_law_receipt": False,
            "scalar_or_polarization_sector_attachment_receipt": False,
            "same_operator_physical_readout_receipt": False,
            "physical_prediction_unsealed": False,
        },
        "candidate_next_typed_source_object": {
            "schema": "oph.vertex12-directed-transport-ledger.v1",
            "object": "source_emitted_directed_transport_on_a_noncollapsed_quotient_site_set",
            "required_fields": [
                "source_capture_root_sha256",
                "quotient_site_ids_and_carrier_to_site_map",
                "twelve_directed_port_transport_permutations",
                "icosahedral_antipode_map",
                "exact_T_antipode_p_equals_inverse_T_p_receipt",
                "source_transition_event_ids_for_every_transport",
                "site_A5_action_and_exact_covariance_receipt",
            ],
            "low_cost_producer_design": [
                "emit the twelve directed site maps from source transition events rather than assigning them after capture",
                "reject any source whose port maps do not descend to a noncollapsed common quotient",
                "reject unless antipodal ports act by inverse maps and the site A5 action conjugates T_p to T_g(p)",
                "bind a later sector readout to the exact transport digest; frame, orientation, and boost transport remain separate gates",
            ],
            "current_fixed_matching_family_has_no_qualifying_carrier_set_quotient": True,
        },
        "comparison_data_read": False,
        "implementation_pins": {
            "producer": _raw_pin(PRODUCER_PATH),
            "independent_packet_verifier": _raw_pin(INDEPENDENT_VERIFIER_PATH),
            "mutation_tests": _raw_pin(TEST_PATH),
        },
        "claim_boundary": (
            "For the fixed canonical-seed eight-carrier source topology, the "
            "same-port seams supply twelve exact internal matching involutions "
            "and their rational averaging projectors. The repair ledger matches "
            "the named seams and arithmetic-mean rule under the declared "
            "15-decimal serialization tolerance. Its 24 record/readback pairs "
            "are an in-process digest reread of the same captured post-repair "
            "snapshot, not independent persistence or a second producer. The "
            "inferred seam-partner matchings fail the antipodal inverse equation, "
            "and an exhaustive 4,140-partition check finds no noncollapsed common "
            "carrier-set quotient preserving all twelve matchings that repairs it. "
            "This does not exclude other seeds, carrier counts, directed maps, "
            "port-dependent or linear quotients, or enlarged site spaces. No "
            "spatial propagation, physical sector, frame, boost, or physical "
            "prediction follows from this receipt."
        ),
    }
    return payload


def produce_receipt() -> dict[str, Any]:
    payload = _payload()
    return {**payload, "receipt_sha256": _sha(payload)}


def verify_receipt(report: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        received = copy.deepcopy(dict(report))
        digest = received.pop("receipt_sha256", None)
        if digest != _sha(received):
            reasons.append("receipt_digest_mismatch")
        if _canonical_bytes(received) != _canonical_bytes(_payload()):
            reasons.append("producer_replay_mismatch")
        if report.get("status") != STATUS:
            reasons.append("status_mismatch")
        boundary = report.get("quotient_and_spatial_boundary")
        if not isinstance(boundary, Mapping):
            reasons.append("spatial_boundary_missing")
        elif any(
            boundary.get(key) is not False
            for key in (
                "internal_seam_transfer_is_spatial_translation",
                "directed_antipode_inverse_transport_receipt",
                "noncollapsed_quotient_site_map_receipt",
                "same_operator_physical_readout_receipt",
                "physical_prediction_unsealed",
            )
        ):
            reasons.append("forbidden_physical_promotion")
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        PacketError,
        RecursionError,
    ):
        reasons.append("malformed_or_noncanonical_receipt")
    passed = not reasons
    return {
        "schema": VERIFICATION_SCHEMA,
        "receipt": passed,
        "status": "PASS" if passed else "FAIL",
        "reasons": sorted(set(reasons)),
        "producer_replay": True,
        "independent_implementation": False,
        "claim_boundary": (
            "Producer replay checks the finite source and packet. It does not "
            "supply a spatial or physical bridge."
        ),
    }


def _write_json(value: Mapping[str, Any], path: Path | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        print(rendered, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        report = _load_json(args.verify)
        verification = verify_receipt(report)
        _write_json(verification, args.output)
        return 0 if verification["receipt"] else 1
    report = produce_receipt()
    verification = verify_receipt(report)
    if not verification["receipt"]:
        _write_json(verification, args.output)
        return 1
    _write_json(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
