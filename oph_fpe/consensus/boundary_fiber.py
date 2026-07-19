from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.claims import CONTINUATION, with_claim_metadata
from oph_fpe.evidence.artifact_paths import companion_input_packet_path
from oph_fpe.evidence.validation import utf8_byte_length


MAX_FIBER_ROWS = 100_000
MAX_TRANSITION_ROWS = 250_000
MAX_CANONICAL_VALUE_DEPTH = 16
MAX_CANONICAL_VALUE_NODES = 2_048
MAX_CANONICAL_VALUE_BYTES = 32_768
MAX_FIBER_CANONICAL_BYTES = 5_000_000
MAX_RECORD_ID_BYTES = 256
MAX_BOUNDARY_PACKET_BYTES = 20_000_000
MAX_BOUNDARY_PACKET_NODES = 1_500_000
MAX_BOUNDARY_PACKET_DEPTH = 24
PINNED_BOUNDARY_FIBER_THEORY_RELEASE = "r1556@bec81e2d"
PINNED_BOUNDARY_FIBER_THEORY_ARTIFACTS = {
    "rer-r1556-boundary-fiber-lean": {
        "path": "Lean/ObserverPatchHolography/Source/ObserverPatchHolography/BoundaryFiber.lean",
        "sha256": "sha256:4b009184b992322f71463a7689c890e74ac0b1fe5901080d5c2cb855646689f3",
    },
    "rer-r1556-primitives-boundary-theorem-lean": {
        "path": "Lean/ObserverPatchHolography/Source/ObserverPatchHolography/Primitives.lean",
        "sha256": "sha256:5739d15f7e012d068f995d85033a7bfaf61f7b024ab334f418f3b6cae06fa144",
    },
    "rer-r1556-rule90-boundary-witness-lean": {
        "path": "Lean/ObserverPatchHolography/Source/ObserverPatchHolography/Rule90.lean",
        "sha256": "sha256:dd217298a7acce9977d20357be19e4b80e030aa4e4f7bc0686f0d5305097e526",
    },
}


def boundary_conditioned_uniqueness_receipt(
    *,
    fiber_rows: Sequence[Mapping[str, Any]] | None = None,
    transition_rows: Sequence[Mapping[str, Any]] | None = None,
    boundary_map_preserved: bool | None = None,
    sector_map_preserved: bool | None = None,
    consistent_extension_count: int | None = None,
    checked_states: int | None = None,
) -> dict[str, Any]:
    """Recompute finite boundary-fiber uniqueness modulo gauge.

    ``fiber_rows`` is the replayable quotient table.  Each row must name a
    unique ``record_id`` and provide JSON values for ``boundary``, ``sector``
    and ``gauge_class``.  ``transition_rows`` names source/target record IDs;
    preservation is recomputed from the table rather than accepted as a
    boolean declaration.

    The three legacy scalar arguments remain readable for old callers, but can
    never earn the receipt.  A count of one does not establish that the whole
    consistent quotient fiber was enumerated.
    """

    legacy_declaration_present = any(
        value is not None
        for value in (
            boundary_map_preserved,
            sector_map_preserved,
            consistent_extension_count,
        )
    )
    blockers: list[str] = []
    rows: list[Mapping[str, Any]] = []
    transitions: list[Mapping[str, Any]] = []
    if fiber_rows is None:
        pass
    elif not isinstance(fiber_rows, Sequence) or isinstance(
        fiber_rows, (str, bytes)
    ):
        blockers.append("fiber_rows_not_a_bounded_sequence")
    elif len(fiber_rows) > MAX_FIBER_ROWS:
        blockers.append("fiber_operation_budget_exceeded")
    else:
        rows = list(fiber_rows)
    if transition_rows is None:
        pass
    elif not isinstance(transition_rows, Sequence) or isinstance(
        transition_rows, (str, bytes)
    ):
        blockers.append("transition_rows_not_a_bounded_sequence")
    elif len(transition_rows) > MAX_TRANSITION_ROWS:
        blockers.append("transition_operation_budget_exceeded")
    else:
        transitions = list(transition_rows)
    if not rows:
        blockers.append("explicit_consistent_quotient_fiber_missing")

    records: dict[str, tuple[str, str, str]] = {}
    fibers: dict[tuple[str, str], set[str]] = defaultdict(set)
    malformed_rows: list[int] = []
    duplicate_ids: list[str] = []
    canonical_byte_count = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or len(row) != 4:
            malformed_rows.append(index)
            continue
        record_id = row.get("record_id")
        if (
            not isinstance(record_id, str)
            or not record_id
            or (record_id_bytes := utf8_byte_length(record_id)) is None
            or record_id_bytes > MAX_RECORD_ID_BYTES
        ):
            malformed_rows.append(index)
            continue
        if record_id in records:
            duplicate_ids.append(record_id)
            continue
        if set(row) != {"record_id", "boundary", "sector", "gauge_class"}:
            malformed_rows.append(index)
            continue
        try:
            boundary = _canonical_json(row["boundary"])
            sector = _canonical_json(row["sector"])
            gauge_class = _canonical_json(row["gauge_class"])
        except (TypeError, ValueError):
            malformed_rows.append(index)
            continue
        canonical_byte_count += sum(
            len(value.encode("utf-8")) for value in (boundary, sector, gauge_class)
        )
        if canonical_byte_count > MAX_FIBER_CANONICAL_BYTES:
            malformed_rows.append(index)
            blockers.append("fiber_canonical_byte_budget_exceeded")
            break
        records[record_id] = (boundary, sector, gauge_class)
        fibers[(boundary, sector)].add(gauge_class)

    if malformed_rows:
        blockers.append("malformed_fiber_rows")
    if duplicate_ids:
        blockers.append("duplicate_fiber_record_ids")

    malformed_transitions: list[int] = []
    boundary_violations: list[int] = []
    sector_violations: list[int] = []
    for index, transition in enumerate(transitions):
        if (
            not isinstance(transition, Mapping)
            or len(transition) != 2
            or set(transition) != {"source", "target"}
        ):
            malformed_transitions.append(index)
            continue
        source = transition.get("source")
        target = transition.get("target")
        if (
            not isinstance(source, str)
            or not isinstance(target, str)
            or (source_bytes := utf8_byte_length(source)) is None
            or (target_bytes := utf8_byte_length(target)) is None
            or source_bytes > MAX_RECORD_ID_BYTES
            or target_bytes > MAX_RECORD_ID_BYTES
        ):
            malformed_transitions.append(index)
            continue
        if source not in records or target not in records:
            malformed_transitions.append(index)
            continue
        source_boundary, source_sector, _ = records[source]
        target_boundary, target_sector, _ = records[target]
        if source_boundary != target_boundary:
            boundary_violations.append(index)
        if source_sector != target_sector:
            sector_violations.append(index)

    if malformed_transitions:
        blockers.append("malformed_or_unresolved_transition_rows")
    if boundary_violations:
        blockers.append("boundary_not_preserved_by_transition")
    if sector_violations:
        blockers.append("sector_not_preserved_by_transition")

    multi_gauge_fibers = [
        {"boundary": boundary, "sector": sector, "gauge_class_count": len(gauge_classes)}
        for (boundary, sector), gauge_classes in sorted(fibers.items())
        if len(gauge_classes) != 1
    ]
    supplied_table_manifest_well_formed = bool(rows) and not malformed_rows and not duplicate_ids
    preservation_recomputed = bool(transitions) and not malformed_transitions
    boundary_preserved = preservation_recomputed and not boundary_violations
    sector_preserved = preservation_recomputed and not sector_violations
    singleton_modulo_gauge = supplied_table_manifest_well_formed and not multi_gauge_fibers
    supplied_table_receipt = bool(
        not blockers
        and supplied_table_manifest_well_formed
        and preservation_recomputed
        and boundary_preserved
        and sector_preserved
        and singleton_modulo_gauge
    )
    if rows and not transitions:
        blockers.append("transition_replay_missing")
    if multi_gauge_fibers:
        blockers.append("boundary_fiber_not_singleton_modulo_gauge")
    supplied_table_receipt = supplied_table_receipt and not blockers

    report = {
        "schema_version": "boundary-fiber-replay-v2",
        "mode": "finite_boundary_fiber_quotient_recomputation",
        "GENERIC_BOUNDARY_FIBER_THEOREM_AVAILABLE": True,
        "TREE_PACKET_NET_BOUNDARY_FIBER_THEOREM_PINNED": True,
        "pinned_theory_registry_release": PINNED_BOUNDARY_FIBER_THEORY_RELEASE,
        "pinned_theory_artifacts": [
            {"artifact_id": artifact_id, **artifact}
            for artifact_id, artifact in PINNED_BOUNDARY_FIBER_THEORY_ARTIFACTS.items()
        ],
        "BOUNDARY_FIBER_SUPPLIED_TABLE_CONSISTENCY_RECEIPT": supplied_table_receipt,
        "BOUNDARY_CONDITIONED_UNIQUENESS_RECEIPT": False,
        "PHYSICAL_EINSTEIN_BOUNDARY_APPLICATION_RECEIPT": False,
        "RUN_ARTIFACT_BINDING_RECEIPT": False,
        "receipt": False,
        "legacy_declaration_present": legacy_declaration_present,
        "legacy_declarations_promoted": False,
        "complete_fiber_manifest": False,
        "supplied_table_manifest_well_formed": supplied_table_manifest_well_formed,
        "external_complete_fiber_manifest_resolved": False,
        "preservation_recomputed": preservation_recomputed,
        "boundary_map_preserved": boundary_preserved,
        "sector_map_preserved": sector_preserved,
        "singleton_modulo_gauge": singleton_modulo_gauge,
        "fiber_count": len(fibers),
        "consistent_record_count": len(records),
        "checked_states": len(records),
        "legacy_declared_checked_states": (
            checked_states
            if isinstance(checked_states, int) and not isinstance(checked_states, bool)
            else None
        ),
        "multi_gauge_fibers": multi_gauge_fibers,
        "malformed_fiber_rows": malformed_rows,
        "duplicate_record_ids": sorted(set(duplicate_ids)),
        "malformed_transition_rows": malformed_transitions,
        "boundary_violation_rows": boundary_violations,
        "sector_violation_rows": sector_violations,
        "blockers": blockers,
        "uniqueness_blockers": ["external_complete_fiber_manifest_unresolved"],
        "claim_boundary": (
            "recomputed preservation and singleton consistency on the supplied quotient table; the generic "
            "Lean theorem and stronger TreePacketNet application are pinned to the locked r1556 source "
            "files, but a caller-selected table cannot authenticate that the whole "
            "consistent fiber of an external run was enumerated, so boundary-conditioned uniqueness and "
            "the physical Einstein-tower application remain false"
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=CONTINUATION,
        receipt="BOUNDARY_CONDITIONED_UNIQUENESS_RECEIPT",
    )


def write_boundary_fiber_certificate(
    path: Path,
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a replayable supplied-table consistency certificate."""

    if not isinstance(packet, Mapping) or set(packet) != {
        "fiber_rows",
        "transition_rows",
    }:
        raise ValueError("boundary-fiber packet has missing or unknown fields")
    encoded = _bounded_packet_json_bytes(packet)
    report = boundary_conditioned_uniqueness_receipt(
        fiber_rows=packet["fiber_rows"],
        transition_rows=packet["transition_rows"],
    )
    report["input_packet_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        destination = destination / "boundary_fiber_certificate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    replay_path = companion_input_packet_path(
        destination,
        canonical_certificate_filename="boundary_fiber_certificate.json",
        canonical_input_filename="boundary_fiber_input_packet.json",
    )
    replay_path.write_text(
        json.dumps(packet, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _canonical_json(value: Any) -> str:
    _validate_bounded_json(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _validate_bounded_json(value: Any) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > MAX_CANONICAL_VALUE_NODES or depth > MAX_CANONICAL_VALUE_DEPTH:
            raise ValueError("canonical JSON value exceeds its structure budget")
        if item is None or isinstance(item, (bool, int, str)):
            continue
        if isinstance(item, float):
            if item != item or item in {float("inf"), float("-inf")}:
                raise ValueError("canonical JSON value must be finite")
            continue
        if isinstance(item, list):
            if len(item) > MAX_CANONICAL_VALUE_NODES:
                raise ValueError("canonical JSON value exceeds its container budget")
            pending.extend((child, depth + 1) for child in item)
            continue
        if (
            isinstance(item, dict)
            and len(item) <= MAX_CANONICAL_VALUE_NODES
            and all(isinstance(key, str) for key in item)
        ):
            pending.extend((child, depth + 1) for child in item.values())
            continue
        raise TypeError("canonical JSON value has an unsupported type")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("canonical JSON value is not strict UTF-8 JSON") from exc
    if len(encoded) > MAX_CANONICAL_VALUE_BYTES:
        raise ValueError("canonical JSON value exceeds its byte budget")


def _bounded_packet_json_bytes(value: Any) -> bytes:
    pending: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > MAX_BOUNDARY_PACKET_NODES:
            raise ValueError("boundary-fiber packet exceeds the JSON node budget")
        if depth > MAX_BOUNDARY_PACKET_DEPTH:
            raise ValueError("boundary-fiber packet exceeds the JSON depth budget")
        if item is None or isinstance(item, (bool, int)):
            continue
        if isinstance(item, float):
            if item != item or item in {float("inf"), float("-inf")}:
                raise ValueError("boundary-fiber packet contains a nonfinite number")
            continue
        if isinstance(item, str):
            byte_length = utf8_byte_length(item)
            if byte_length is None or byte_length > MAX_CANONICAL_VALUE_BYTES:
                raise ValueError("boundary-fiber packet contains an oversized string")
            continue
        if isinstance(item, Mapping):
            if len(item) > MAX_CANONICAL_VALUE_NODES or not all(
                isinstance(key, str) for key in item
            ):
                raise ValueError("boundary-fiber packet contains a non-string mapping key")
            pending.extend((child, depth + 1) for child in item.values())
            continue
        if isinstance(item, list):
            container_limit = (
                max(MAX_FIBER_ROWS, MAX_TRANSITION_ROWS)
                if depth == 1
                else MAX_CANONICAL_VALUE_NODES
            )
            if len(item) > container_limit:
                raise ValueError("boundary-fiber packet contains an oversized array")
            pending.extend((child, depth + 1) for child in item)
            continue
        raise ValueError("boundary-fiber packet contains unsupported JSON data")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("boundary-fiber packet is not canonical JSON data") from exc
    if len(encoded) > MAX_BOUNDARY_PACKET_BYTES:
        raise ValueError("boundary-fiber packet exceeds the byte budget")
    return encoded
