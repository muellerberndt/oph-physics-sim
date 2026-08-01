"""Independent verifier for the issue-655 atomic port-transfer subpacket.

This verifier does not import the packet producer.  It reruns the pinned
source engine and independently reconstructs the twelve matching maps, repair
history, full-port source readback, icosahedral antipodes, and exhaustive
quotient classification.  The upstream source engine is shared and is not an
independent simulator implementation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source


SCHEMA = "oph.vertex12-atomic-port-transfer-subpacket.v1"
VERIFICATION_SCHEMA = (
    "oph.vertex12-atomic-port-transfer-subpacket-independent-verification.v1"
)
STATUS = (
    "INTERNAL_VERTEX12_ATOMIC_TRANSFER_AND_COMPLETE_SOURCE_READBACK_ATTAINED__"
    "SPATIAL_PHYSICAL_BRIDGE_OPEN"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = (
    REPOSITORY_ROOT / "data/repair_closure/vertex12_atomic_port_transfer_receipt.json"
)
CARRIER_MANIFEST = (
    REPOSITORY_ROOT / "tests/fixtures/echosahedral_federation_reference.json"
)
PRODUCER_PATH = REPOSITORY_ROOT / "oph_fpe/dynamics/vertex12_atomic_port_transfer.py"
INDEPENDENT_VERIFIER_PATH = Path(__file__).resolve()
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
}
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


def _raw_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _exact_keys(value: Any, expected: set[str], label: str, reasons: list[str]) -> bool:
    if not isinstance(value, Mapping):
        reasons.append(f"{label}_not_object")
        return False
    if set(value) != expected:
        reasons.append(f"{label}_keyset_mismatch")
        return False
    return True


def _port_index(label: Any) -> int:
    if (
        not isinstance(label, str)
        or len(label) != 3
        or label[0] != "p"
        or not label[1:].isdigit()
    ):
        raise ValueError("malformed carrier port label")
    value = int(label[1:])
    if not 0 <= value < 12:
        raise ValueError("carrier port label out of range")
    return value


def _antipodes() -> tuple[int, ...]:
    manifest = json.loads(CARRIER_MANIFEST.read_text(encoding="utf-8"))
    carrier = manifest["carrier"]
    if [_port_index(item) for item in carrier["ports"]] != list(range(12)):
        raise ValueError("carrier port order drift")
    adjacency = [set() for _ in range(12)]
    for edge in carrier["edges"]:
        left, right = (_port_index(item) for item in edge)
        if left == right or right in adjacency[left]:
            raise ValueError("carrier edge loop or duplicate")
        adjacency[left].add(right)
        adjacency[right].add(left)
    if len(carrier["edges"]) != 30 or {len(row) for row in adjacency} != {5}:
        raise ValueError("carrier incidence drift")
    result: list[int] = []
    for source in range(12):
        distance = [-1] * 12
        distance[source] = 0
        queue = [source]
        for left in queue:
            for right in adjacency[left]:
                if distance[right] == -1:
                    distance[right] = distance[left] + 1
                    queue.append(right)
        farthest = [
            index for index, value in enumerate(distance) if value == max(distance)
        ]
        if max(distance) != 3 or len(farthest) != 1:
            raise ValueError("carrier antipode is not unique")
        result.append(farthest[0])
    if any(result[result[index]] != index for index in range(12)):
        raise ValueError("carrier antipode is not involutive")
    return tuple(result)


def _matchings(
    bundle: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...], tuple[tuple[str, ...], ...]]:
    carrier_ids = tuple(str(item) for item in bundle["carrier_ids"])
    if len(carrier_ids) != 8 or len(set(carrier_ids)) != 8:
        raise ValueError("carrier census drift")
    carrier_index = {item: index for index, item in enumerate(carrier_ids)}
    partner: list[list[int | None]] = [[None] * 8 for _ in range(12)]
    seam_ids: list[list[str]] = [[] for _ in range(12)]
    seams = bundle["seams"]
    if not isinstance(seams, list) or len(seams) != 48:
        raise ValueError("seam census drift")
    for seam in seams:
        left_ports = seam["left_ports"]
        port = left_ports[0]
        if (
            len(left_ports) != 1
            or seam["right_ports"] != left_ports
            or seam["left_to_right_ports"] != left_ports
            or seam["right_to_left_ports"] != left_ports
            or seam["left_to_right_orientation"] != [-1]
            or seam["right_to_left_orientation"] != [-1]
            or seam["collar_kind"] != "single_port"
            or type(port) is not int
            or not 0 <= port < 12
        ):
            raise ValueError("seam type drift")
        left = carrier_index[seam["left_carrier_id"]]
        right = carrier_index[seam["right_carrier_id"]]
        if (
            left == right
            or partner[port][left] is not None
            or partner[port][right] is not None
        ):
            raise ValueError("seam matching collision")
        partner[port][left] = right
        partner[port][right] = left
        seam_ids[port].append(str(seam["seam_id"]))
    permutations: list[tuple[int, ...]] = []
    normalized_seams: list[tuple[str, ...]] = []
    for port in range(12):
        if any(item is None for item in partner[port]):
            raise ValueError("incomplete port matching")
        row = tuple(int(item) for item in partner[port])
        if (
            sorted(row) != list(range(8))
            or any(row[row[index]] != index for index in range(8))
            or any(row[index] == index for index in range(8))
            or len(seam_ids[port]) != 4
        ):
            raise ValueError("port matching is not fixed-point-free involution")
        permutations.append(row)
        normalized_seams.append(tuple(sorted(seam_ids[port])))
    return carrier_ids, tuple(permutations), tuple(normalized_seams)


def _inverse(permutation: Sequence[int]) -> tuple[int, ...]:
    result = [0] * len(permutation)
    for source, target in enumerate(permutation):
        result[target] = source
    return tuple(result)


def _partitions(size: int) -> Iterable[tuple[int, ...]]:
    def visit(prefix: list[int], maximum: int) -> Iterable[tuple[int, ...]]:
        if len(prefix) == size:
            yield tuple(prefix)
            return
        for label in range(maximum + 2):
            yield from visit(prefix + [label], max(maximum, label))

    yield from visit([0], 0)


def _quotient(
    permutations: Sequence[Sequence[int]], antipodes: Sequence[int]
) -> dict[str, Any]:
    count = 0
    congruences: list[dict[str, Any]] = []
    compatible: list[dict[str, Any]] = []
    for labels in _partitions(8):
        count += 1
        block_count = max(labels) + 1
        descended: list[tuple[int, ...]] = []
        valid = True
        for permutation in permutations:
            row: list[int | None] = [None] * block_count
            for source, target in enumerate(permutation):
                source_block = labels[source]
                target_block = labels[target]
                if row[source_block] is None:
                    row[source_block] = target_block
                elif row[source_block] != target_block:
                    valid = False
                    break
            if not valid:
                break
            normalized = tuple(int(item) for item in row)
            if sorted(normalized) != list(range(block_count)):
                valid = False
                break
            descended.append(normalized)
        if not valid:
            continue
        description = {"block_count": block_count, "canonical_labels": list(labels)}
        congruences.append(description)
        if all(
            descended[antipodes[port]] == _inverse(descended[port])
            for port in range(12)
        ):
            compatible.append(description)
    proper = [row for row in congruences if 1 < row["block_count"] < 8]
    noncollapsed = [row for row in compatible if row["block_count"] > 1]
    return {
        "set_partition_count_checked": count,
        "common_congruence_count": len(congruences),
        "common_congruences": congruences,
        "nontrivial_proper_common_congruence_count": len(proper),
        "antipodal_inverse_compatible_quotient_count": len(compatible),
        "antipodal_inverse_compatible_quotients": compatible,
        "noncollapsed_antipodal_inverse_compatible_quotient_count": len(noncollapsed),
        "only_inverse_compatible_quotient_collapses_all_carriers": bool(
            len(compatible) == 1 and compatible[0]["block_count"] == 1
        ),
    }


def _expected_history(
    dynamics: Mapping[str, Any], seam_ids: set[str], carrier_ids: Sequence[str]
) -> dict[str, Any]:
    events = dynamics["repair_event_log"]
    if len(events) != 48 or dynamics["repair_event_examples_complete"] is not True:
        raise ValueError("repair history is incomplete")
    rows: list[dict[str, Any]] = []
    observed: set[str] = set()
    terminal_values: dict[tuple[str, int], Any] = {}
    for event in events:
        material = dict(event)
        event_id = material.pop("event_id")
        if event_id != _sha(material):
            raise ValueError("repair event hash mismatch")
        seam_id = str(event["seam_id"])
        if seam_id not in seam_ids or seam_id in observed:
            raise ValueError("repair seam event mismatch")
        observed.add(seam_id)
        reads = event["read_set"]
        writes = event["write_set"]
        if (
            len(reads) != 2
            or len(writes) != 2
            or [(row["carrier_id"], row["port"]) for row in reads]
            != [(row["carrier_id"], row["port"]) for row in writes]
            or reads[0]["port"] != reads[1]["port"]
            or writes[0]["value"] != writes[1]["value"]
            or event["mismatch_after"] != 0.0
            or event["strict_descent"] is not True
            or event["update"] != "endpoint_arithmetic_mean"
            or any(row["version"] != 0 for row in reads)
            or any(
                row["expected_version"] != 0 or row["committed_version"] != 1
                for row in writes
            )
        ):
            raise ValueError("repair atomicity mismatch")
        for write in writes:
            key = (str(write["carrier_id"]), int(write["port"]))
            if key in terminal_values:
                raise ValueError("duplicate terminal carrier-port write")
            terminal_values[key] = write["value"]
        rows.append(
            {"seam_id": seam_id, "port": reads[0]["port"], "event_id": event_id}
        )
    if observed != seam_ids:
        raise ValueError("repair seam coverage mismatch")
    expected_coordinates = {
        (carrier_id, port) for carrier_id in carrier_ids for port in range(12)
    }
    if set(terminal_values) != expected_coordinates:
        raise ValueError("terminal repair writes do not cover the port field")
    terminal_rows = [
        [terminal_values[(carrier_id, port)] for port in range(12)]
        for carrier_id in carrier_ids
    ]
    keys = (
        "TRANSACTION_VALIDATION_COMPLETE_READ_CONFLICT_SET_RECEIPT",
        "UNION_PAYLOAD_ATOMIC_REVALIDATION_RECEIPT",
        "REPAIR_ORDER_REPLAY_EXACT_RECEIPT",
        "REPAIR_IDEMPOTENCE_REPLAY_EXACT_RECEIPT",
        "REPAIR_TERMINAL_FIXED_POINT_RECEIPT",
    )
    if any(dynamics[key] is not True for key in keys):
        raise ValueError("upstream repair receipt failed")
    rows.sort(key=lambda row: row["seam_id"])
    return {
        "event_count": 48,
        "event_rows": rows,
        "event_rows_sha256": _sha(rows),
        "terminal_write_state_rows_sha256": _sha(terminal_rows),
        "terminal_write_coordinate_count": len(terminal_values),
        "every_seam_replayed_once": True,
        "every_event_is_atomic_two_endpoint_mean": True,
        "read_conflict_validation_complete": True,
        "union_atomic_revalidation_complete": True,
        "order_replay_exact": True,
        "idempotence_replay_exact": True,
        "terminal_seam_fixed_point": True,
    }


def _expected_readback(
    dynamics: Mapping[str, Any], observer: Mapping[str, Any], carrier_ids: Sequence[str]
) -> dict[str, Any]:
    snapshots = dynamics["record_state_snapshots"]
    if len(snapshots) != 1 or snapshots[0]["cycle"] != 0:
        raise ValueError("snapshot schedule mismatch")
    snapshot = snapshots[0]
    carrier_rows = snapshot["carrier_rows"]
    by_carrier = {row["carrier_id"]: row for row in carrier_rows}
    if set(by_carrier) != set(carrier_ids) or len(by_carrier) != len(carrier_rows):
        raise ValueError("snapshot coverage mismatch")
    for row in carrier_rows:
        if len(row["full_port_state"]) != 12 or row["full_port_state_sha256"] != _sha(
            row["full_port_state"]
        ):
            raise ValueError("snapshot row hash mismatch")
    if snapshot["carrier_rows_sha256"] != _sha(carrier_rows):
        raise ValueError("snapshot aggregate hash mismatch")
    records: dict[str, Mapping[str, Any]] = {}
    readbacks: list[Mapping[str, Any]] = []
    for event in observer["events"]:
        material = dict(event)
        event_id = material.pop("event_id")
        if event_id != _sha(material):
            raise ValueError("observer event hash mismatch")
        if event["kind"] == "RECORD_COMMIT":
            records[event_id] = event
        elif event["kind"] == "READBACK":
            readbacks.append(event)
    if len(records) != 24 or len(readbacks) != 24:
        raise ValueError("record/readback count mismatch")
    covered: set[str] = set()
    pairs: list[dict[str, Any]] = []
    for readback in readbacks:
        record_id = readback["record_event_id"]
        record = records[record_id]
        carrier_id = record["carrier_id"]
        row = by_carrier[carrier_id]
        digest = record["full_port_state_sha256"]
        if (
            record["record_cycle"] != 0
            or record["full_port_state"] != row["full_port_state"]
            or digest != row["full_port_state_sha256"]
            or digest != _sha(record["full_port_state"])
            or readback["carrier_id"] != carrier_id
            or readback["record_cycle"] != 0
            or readback["recomputed_full_port_state_sha256"] != digest
            or readback["record_signature_matches_source_field"] is not True
            or readback["parents"] != [record_id]
        ):
            raise ValueError("record/readback semantic mismatch")
        covered.add(carrier_id)
        pairs.append(
            {
                "carrier_id": carrier_id,
                "record_event_id": record_id,
                "state_sha256": digest,
                "readback_event_id": readback["event_id"],
            }
        )
    if covered != set(carrier_ids):
        raise ValueError("readback does not cover every carrier")
    pairs.sort(key=lambda row: (row["carrier_id"], row["record_event_id"]))
    return {
        "snapshot_cycle": 0,
        "visible_state_sha256": snapshot["visible_state_sha256"],
        "carrier_rows_sha256": snapshot["carrier_rows_sha256"],
        "terminal_state_rows_sha256": _sha(
            [by_carrier[carrier_id]["full_port_state"] for carrier_id in carrier_ids]
        ),
        "record_count": 24,
        "readback_count": 24,
        "covered_carrier_count": 8,
        "covered_port_coordinate_count": 96,
        "record_readback_pairs": pairs,
        "record_readback_pairs_sha256": _sha(pairs),
        "every_carrier_full_port_state_committed": True,
        "every_carrier_full_port_state_read_back": True,
        "record_and_readback_state_digests_identical": True,
        "readback_domain": "post_repair_internal_visible_port_state",
        "physical_sector_readout": False,
    }


def verify_report(report: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        top_keys = {
            "schema",
            "issue",
            "status",
            "source_config",
            "source_capture_binding",
            "audited_surface_pins",
            "atomic_transfer_operator",
            "source_history_replay",
            "post_repair_source_readback",
            "quotient_and_spatial_boundary",
            "smallest_missing_typed_source_object",
            "comparison_data_read",
            "implementation_pins",
            "claim_boundary",
            "receipt_sha256",
        }
        if not _exact_keys(report, top_keys, "report", reasons):
            raise ValueError("top-level schema mismatch")
        payload = copy.deepcopy(dict(report))
        digest = payload.pop("receipt_sha256")
        if digest != _sha(payload):
            reasons.append("receipt_digest_mismatch")
        if (
            report["schema"] != SCHEMA
            or report["issue"] != 655
            or report["status"] != STATUS
        ):
            reasons.append("header_mismatch")
        if report["comparison_data_read"] is not False:
            reasons.append("comparison_data_boundary_mismatch")

        capture = capture_physical_source(SOURCE_CONFIG)
        if report["source_config"] != capture["input_config"]:
            reasons.append("source_config_mismatch")
        source = capture["reports"]["source_observer"]
        binding = report["source_capture_binding"]
        expected_binding = {
            "schema": capture["schema"],
            "artifact_type": capture["artifact_type"],
            "source_capture_replay_receipt": True,
            "source_engine_independently_reimplemented": False,
            "finite_source_architecture_receipt": True,
            "finite_federation_realization_receipt": True,
            "canonical_default_seed": True,
            "canonical_default_replica": True,
            "coverage_config_seam_ledger_matches_default_source": True,
            "input_config_sha256": capture["source_hashes"]["input_config"],
            "federation_bundle_sha256": capture["source_hashes"]["federation"],
            "source_state_root_sha256": capture["source_artifacts"][
                "source_state_root_sha256"
            ],
            "repair_log_sha256": capture["source_hashes"]["repair_log"],
            "observer_log_sha256": capture["source_hashes"]["observer_log"],
            "postrun_capture_sha256": capture["source_hashes"]["postrun_capture"],
            "target_or_comparison_token_hits": source["source_forbidden_target_hits"],
        }
        if binding != expected_binding:
            reasons.append("source_capture_binding_mismatch")
        if (
            source["SOURCE_PATCH_ARCHITECTURE_RECEIPT"] is not True
            or source["PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT"]
            is not True
        ):
            reasons.append("upstream_source_receipt_missing")

        carrier_ids, permutations, seam_rows = _matchings(
            capture["source_artifacts"]["federation_bundle"]
        )
        default_ids, default_permutations, default_seams = _matchings(
            capture_physical_source()["source_artifacts"]["federation_bundle"]
        )
        if (carrier_ids, permutations, seam_rows) != (
            default_ids,
            default_permutations,
            default_seams,
        ):
            reasons.append("coverage_config_changed_default_topology")
        antipodes = _antipodes()
        expected_port_rows = []
        for port, permutation in enumerate(permutations):
            expected_port_rows.append(
                {
                    "port": port,
                    "carrier_partner_permutation": list(permutation),
                    "carrier_partner_permutation_sha256": _sha(list(permutation)),
                    "seam_ids": list(seam_rows[port]),
                    "seam_ids_sha256": _sha(list(seam_rows[port])),
                    "matching_size": 4,
                    "fixed_point_free": True,
                    "permutation": True,
                    "transfer_involution_Sp_squared_is_identity": True,
                    "transfer_is_self_adjoint_permutation": True,
                    "repair_projector_formula": "A_p=(I+S_p)/2 over Q",
                    "repair_projector_idempotent": True,
                    "repair_projector_self_adjoint": True,
                    "antipodal_port": antipodes[port],
                    "antipodal_map_equals_inverse_transfer": False,
                }
            )
            if permutations[antipodes[port]] == _inverse(permutation):
                reasons.append("unexpected_antipodal_inverse_relation")
        expected_operator = {
            "domain": "internal_federation_visible_port_fiber_Q^(8_times_12)",
            "carrier_ids": list(carrier_ids),
            "carrier_count": 8,
            "port_count": 12,
            "seam_count": 48,
            "port_rows": expected_port_rows,
            "port_rows_sha256": _sha(expected_port_rows),
            "complete_matching_on_every_port": True,
            "all_twelve_transfer_involutions_exact": True,
            "all_twelve_rational_repair_projectors_exact": True,
            "block_diagonal_full_repair_formula": "A=direct_sum_p (I+S_p)/2",
            "source_native_internal_port_transfer_receipt": True,
            "source_native_spatial_translation_receipt": False,
        }
        if report["atomic_transfer_operator"] != expected_operator:
            reasons.append("atomic_transfer_operator_mismatch")

        seam_ids = {item for row in seam_rows for item in row}
        expected_history = _expected_history(
            capture["source_artifacts"]["dynamics"], seam_ids, carrier_ids
        )
        expected_readback = _expected_readback(
            capture["source_artifacts"]["dynamics"],
            capture["source_artifacts"]["observer_log"],
            carrier_ids,
        )
        if (
            expected_history["terminal_write_state_rows_sha256"]
            != expected_readback["terminal_state_rows_sha256"]
        ):
            reasons.append("terminal_repair_to_snapshot_binding_mismatch")
        expected_history["terminal_write_state_matches_readback_snapshot"] = True
        if report["source_history_replay"] != expected_history:
            reasons.append("source_history_replay_mismatch")
        if report["post_repair_source_readback"] != expected_readback:
            reasons.append("source_readback_mismatch")

        quotient = _quotient(permutations, antipodes)
        expected_boundary = {
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
        }
        if report["quotient_and_spatial_boundary"] != expected_boundary:
            reasons.append("quotient_or_spatial_boundary_mismatch")
        if not (
            quotient["set_partition_count_checked"] == 4140
            and quotient["common_congruence_count"] == 2
            and quotient["nontrivial_proper_common_congruence_count"] == 0
            and quotient["noncollapsed_antipodal_inverse_compatible_quotient_count"]
            == 0
            and quotient["only_inverse_compatible_quotient_collapses_all_carriers"]
        ):
            reasons.append("quotient_enumeration_exit_mismatch")

        expected_missing = {
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
            "current_source_requires_topology_or_quotient_producer_change": True,
        }
        if report["smallest_missing_typed_source_object"] != expected_missing:
            reasons.append("missing_source_object_contract_mismatch")

        expected_surface_pins = {
            name: _raw_pin(path) for name, path in sorted(AUDITED_SURFACES.items())
        }
        if report["audited_surface_pins"] != expected_surface_pins:
            reasons.append("audited_surface_pin_mismatch")
        expected_implementation_pins = {
            "producer": _raw_pin(PRODUCER_PATH),
            "independent_packet_verifier": _raw_pin(INDEPENDENT_VERIFIER_PATH),
            "mutation_tests": _raw_pin(TEST_PATH),
        }
        if report["implementation_pins"] != expected_implementation_pins:
            reasons.append("implementation_pin_mismatch")
        expected_claim = (
            "The canonical eight-carrier source topology supplies twelve exact "
            "port-labelled matching involutions, their rational averaging "
            "projectors, a complete atomic repair history, and digest-identical "
            "post-repair full-port source readback covering all eight carriers. "
            "This is an internal federation result. The antipodal matchings are "
            "not inverse transports, and an exhaustive 4,140-partition check "
            "finds no noncollapsed common quotient that repairs the relation. "
            "No spatial propagation, physical sector, frame, boost, or physical "
            "prediction follows from this receipt."
        )
        if report["claim_boundary"] != expected_claim:
            reasons.append("claim_boundary_mismatch")
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        RecursionError,
    ):
        reasons.append("malformed_or_noncanonical_report")
    passed = not reasons
    return {
        "schema": VERIFICATION_SCHEMA,
        "receipt": passed,
        "status": "PASS" if passed else "FAIL",
        "reasons": sorted(set(reasons)),
        "packet_analysis_independently_reimplemented": True,
        "source_engine_independently_reimplemented": False,
        "comparison_data_read": False,
        "claim_boundary": (
            "This verifier independently reconstructs the packet analysis from "
            "the shared pinned source engine. It does not certify spatial or "
            "physical propagation."
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
    parser.add_argument("report", nargs="?", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = verify_report(report)
    _write_json(result, args.output)
    return 0 if result["receipt"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
