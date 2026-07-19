"""Exact finite public-record capacity receipts for the OPH N-closure lane.

The primary finite quantity is the zero-error code size
``M_0(q) = alpha(G_q)`` of a source-supplied public checkpoint packet.  The
universe coordinate is its logarithm, ``N = log M_0(U_N)``.  This module never
uses a measured cosmological constant, an observed horizon radius, an
electroweak target, or a caller-supplied desired capacity to construct the
packet.

The bundled twelve-port packet is a schema/control for the reversible fast
branch.  It is not a source-derived physical universe packet and can therefore
never, by itself, promote the physical N closure.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations, product
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.evidence.validation import utf8_byte_length


FAIL_STATUSES = {
    "NO_CAPACITY_READBACK",
    "AMBIGUOUS_CAPACITY_READBACK",
    "INCOMPLETE_TERMINAL_FIBER",
    "NO_RECORD_ATOM_RESTRICTION",
    "NO_PUBLIC_RECORD_REACHABILITY",
    "NO_PUBLICNESS_POLICY",
    "NO_GLOBAL_PUBLIC_CHECKPOINT_COUPLING",
    "LOCAL_MARGINAL_MISMATCH",
    "NO_CAPACITY_CARRIER_REPRESENTATION",
    "FINITE_SIZE_SELECTOR_FAILED",
    "CIRCULAR_CAPACITY_DEFINITION",
    "TARGET_TAINTED",
    "INVALID_PACKET",
    "INCOMPLETE_CONTINUATION_MANIFEST",
    "RECEIVER_CONTINUATION_KNOWLEDGE_INCOMPLETE",
}

SECTION_ID_SCHEME = "canonical-json-observer-atom-pairs-v1"
MAX_PACKET_BYTES = 1_000_000
MAX_IDENTIFIER_LENGTH = 256
MAX_SECTION_ID_BYTES = 32_768
MAX_OBSERVERS = 32
MAX_ATOMS_PER_OBSERVER = 64
MAX_INTERFACES = 512
MAX_ENUMERATED_SECTIONS = 4096
MAX_BACKTRACK_VISITS = 1_000_000
MAX_CONTINUATIONS = 128
MAX_CAPACITY_DIMENSION = 4096
MAX_EXACT_MIS_VERTICES = 128
MAX_MIS_SEARCH_VISITS = 2_000_000
MAX_DECODER_OUTPUTS = 16
MAX_DECODER_ASSIGNMENTS = 2_000_000
MAX_APPROXIMATE_CAPACITY_VERTICES = 12
MAX_TERMINAL_FIBER_PACKETS = 32
MAX_TERMINAL_FIBER_BYTES = 8_000_000

_PACKET_FIELDS = {
    "schema",
    "status",
    "regulator_id",
    "carrier_type_id",
    "terminal_id",
    "capacity_dimension",
    "observers",
    "interfaces",
    "reachability_witnesses",
    "publicness_policy",
    "expected_continuation_ids",
    "receiver_known_continuation_ids",
    "continuation_manifest_complete",
    "global_checkpoint_kernels",
    "local_marginal_consistency_passed",
    "projection_supports",
    "self_read_predicate_injected",
    "supplied_capacity_metadata_read_by_producer",
    "lambda_used",
    "ew_bridge_used",
    "rho_used",
}

PORTS = (
    "north",
    "south",
    *(f"upper_{index}" for index in range(5)),
    *(f"lower_{index}" for index in range(5)),
)


def _fail(status: str, **details: Any) -> dict[str, Any]:
    if status not in FAIL_STATUSES:
        raise ValueError(f"unknown public-record status: {status}")
    return {"status": status, **details}


def section_id(section: Mapping[str, str]) -> str:
    """Return an injective canonical identifier for a finite local section.

    JSON string escaping makes delimiter-bearing observer and atom names
    unambiguous, unlike the historical ``observer=atom|...`` encoding.
    """

    if not isinstance(section, Mapping) or not section:
        raise ValueError("section must be a nonempty mapping")
    observer_ids = list(section)
    for observer in observer_ids:
        _validate_identifier(observer, "section observer")
    pairs: list[list[str]] = []
    for observer in sorted(observer_ids):
        atom = section[observer]
        _validate_identifier(atom, f"section atom for {observer}")
        pairs.append([observer, atom])
    return "section:v1:" + json.dumps(
        pairs,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def public_global_sections(
    observers: Mapping[str, Sequence[str]],
    interfaces: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Enumerate compatible global sections of finite local record atoms."""
    if not isinstance(observers, Mapping) or not observers:
        raise ValueError("observers must be a nonempty mapping")
    if len(observers) > MAX_OBSERVERS:
        raise ValueError("observer count exceeds the bounded packet limit")
    if not isinstance(interfaces, Sequence) or isinstance(interfaces, (str, bytes)):
        raise ValueError("interfaces must be a sequence of mappings")
    if len(interfaces) > MAX_INTERFACES:
        raise ValueError("interface count exceeds the bounded packet limit")
    if any(
        not isinstance(atoms, Sequence)
        or isinstance(atoms, (str, bytes))
        or not atoms
        or len(atoms) > MAX_ATOMS_PER_OBSERVER
        for atoms in observers.values()
    ):
        raise ValueError("every observer needs a nonempty record-atom set")
    raw_observer_ids = list(observers)
    for observer in raw_observer_ids:
        _validate_identifier(observer, "observer id")
    observer_ids = sorted(raw_observer_ids)
    for observer in observer_ids:
        for atom in observers[observer]:
            _validate_identifier(atom, f"record atom for {observer}")
    if any(len(set(observers[item])) != len(observers[item]) for item in observer_ids):
        raise ValueError("local record atoms must be unique")
    interface_ids: set[str] = set()
    for interface in interfaces:
        if not isinstance(interface, Mapping):
            raise ValueError("interfaces must contain mappings")
        if set(interface) != {
            "interface_id",
            "left_observer",
            "right_observer",
            "left_readout",
            "right_readout",
        }:
            raise ValueError("interfaces have missing or unknown fields")
        interface_id = interface.get("interface_id")
        _validate_identifier(interface_id, "interface_id")
        if interface_id in interface_ids:
            raise ValueError("interface_id values must be unique")
        interface_ids.add(interface_id)
        left = interface.get("left_observer")
        right = interface.get("right_observer")
        _validate_identifier(left, f"{interface_id}.left_observer")
        _validate_identifier(right, f"{interface_id}.right_observer")
        if left not in observers or right not in observers or left == right:
            raise ValueError("interface references an unknown observer or self-loop")
        left_map = interface.get("left_readout")
        right_map = interface.get("right_readout")
        if not isinstance(left_map, Mapping) or not isinstance(right_map, Mapping):
            raise ValueError("total RECORD-ATOM-RESTRICTION maps are required")
        if set(left_map) != set(observers[left]) or set(right_map) != set(observers[right]):
            raise ValueError("atom readout maps must be total on endpoint atoms")
        for endpoint, readout in (("left", left_map), ("right", right_map)):
            for output in readout.values():
                _validate_identifier(output, f"{interface_id}.{endpoint}_readout output")

    # Backtrack with early interface pruning.  A literal Cartesian product is
    # exact but needlessly enumerates D^12 assignments for the twelve-port
    # reference packet even though the first connected seam forces equality.
    incident: dict[str, list[Mapping[str, Any]]] = {item: [] for item in observer_ids}
    for interface in interfaces:
        incident[str(interface["left_observer"])].append(interface)
        incident[str(interface["right_observer"])].append(interface)
    sections: list[dict[str, str]] = []
    candidate: dict[str, str] = {}
    backtrack_visits = 0

    def extend(index: int) -> None:
        nonlocal backtrack_visits
        backtrack_visits += 1
        if backtrack_visits > MAX_BACKTRACK_VISITS:
            raise ValueError("section enumeration exceeds the bounded search limit")
        if index == len(observer_ids):
            if len(sections) >= MAX_ENUMERATED_SECTIONS:
                raise ValueError("public section count exceeds the bounded packet limit")
            sections.append(dict(candidate))
            return
        observer = observer_ids[index]
        for atom in observers[observer]:
            candidate[observer] = atom
            compatible = True
            for interface in incident[observer]:
                left = str(interface["left_observer"])
                right = str(interface["right_observer"])
                if left in candidate and right in candidate:
                    compatible = (
                        interface["left_readout"][candidate[left]]
                        == interface["right_readout"][candidate[right]]
                    )
                    if not compatible:
                        break
            if compatible:
                extend(index + 1)
        candidate.pop(observer, None)

    extend(0)
    return sections


def reachable_public_sections(
    public_sections: Sequence[Mapping[str, str]],
    reachability_witnesses: Mapping[str, Sequence[str]],
) -> list[str]:
    """Select sections carrying a nonempty endogenous semantic history."""
    if not isinstance(public_sections, Sequence) or isinstance(
        public_sections, (str, bytes)
    ):
        raise ValueError("public_sections must be a sequence")
    if not isinstance(reachability_witnesses, Mapping):
        raise ValueError("reachability_witnesses must be a mapping")
    section_ids = [section_id(section) for section in public_sections]
    if len(section_ids) != len(set(section_ids)):
        raise ValueError("canonical section identifiers must be injective")
    valid = set(section_ids)
    unknown = set(reachability_witnesses) - valid
    if unknown:
        raise ValueError(f"reachability witnesses reference unknown sections: {sorted(unknown)}")
    reachable: list[str] = []
    for sid, history in reachability_witnesses.items():
        if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
            raise ValueError("reachability histories must be sequences")
        if len(history) > MAX_CONTINUATIONS:
            raise ValueError("reachability history exceeds the bounded packet limit")
        for event in history:
            _validate_identifier(event, f"reachability history for {sid}")
        if history:
            reachable.append(sid)
    return sorted(reachable)


def _channel_rows(
    channel: Mapping[str, Any], reachable: Sequence[str]
) -> dict[str, dict[str, float]]:
    rows = channel.get("rows")
    if not isinstance(rows, Mapping) or set(rows) != set(reachable):
        raise ValueError("each joint checkpoint kernel needs one row per reachable record")
    normalized: dict[str, dict[str, float]] = {}
    for source in reachable:
        raw = rows[source]
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("checkpoint rows must be nonempty mappings")
        if len(raw) > len(reachable) + 1:
            raise ValueError("checkpoint output alphabet exceeds the bounded limit")
        values: dict[str, float] = {}
        for output, probability in raw.items():
            _validate_identifier(
                output,
                "checkpoint output",
                max_bytes=MAX_SECTION_ID_BYTES,
            )
            try:
                parsed_probability = float(probability)
            except (OverflowError, TypeError, ValueError):
                parsed_probability = math.nan
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(parsed_probability)
                or not 0.0 <= parsed_probability <= 1.0
            ):
                raise ValueError("checkpoint probabilities must be finite JSON numbers")
            values[output] = parsed_probability
        if math.fsum(values.values()) != 1.0:
            raise ValueError("checkpoint rows must be exactly normalized")
        normalized[source] = values
    return normalized


def _parse_section_id(value: str) -> dict[str, str]:
    prefix = "section:v1:"
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError("checkpoint output is not a canonical public section")
    try:
        pairs = json.loads(value[len(prefix) :])
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("checkpoint output has malformed canonical section JSON") from exc
    if (
        not isinstance(pairs, list)
        or not pairs
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(item, str) for item in pair)
            for pair in pairs
        )
    ):
        raise ValueError("checkpoint output has malformed observer/atom pairs")
    section = {observer: atom for observer, atom in pairs}
    if len(section) != len(pairs) or section_id(section) != value:
        raise ValueError("checkpoint output is not in canonical section form")
    return section


def _recompute_local_marginal_consistency(
    channels: Sequence[Mapping[str, Any]],
    reachable: Sequence[str],
    observers: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Derive every authorized local marginal from each global joint kernel.

    There is no caller boolean in this computation.  Every positive output
    symbol must resolve to a canonical public section on the frozen observer
    atom sets; the local kernels are then literal marginals of the supplied
    global kernel.
    """

    observer_atoms = {observer: set(atoms) for observer, atoms in observers.items()}
    derived_row_count = 0
    positive_output_count = 0
    for channel in channels:
        authorized = tuple(channel["authorized_observers"])
        rows = _channel_rows(channel, reachable)
        for source, row in rows.items():
            _parse_section_id(source)
            derived_row_count += len(authorized)
            for output, probability in row.items():
                if probability <= 0.0:
                    continue
                section = _parse_section_id(output)
                if set(section) != set(observers):
                    raise ValueError("checkpoint output does not cover the frozen observer set")
                if any(section[observer] not in observer_atoms[observer] for observer in observers):
                    raise ValueError("checkpoint output contains an unknown local record atom")
                if any(observer not in section for observer in authorized):
                    raise ValueError("authorized local marginal is undefined on checkpoint output")
                positive_output_count += 1
    return {
        "local_marginal_consistency_recomputed": True,
        "derived_local_marginal_row_count": derived_row_count,
        "positive_joint_output_count": positive_output_count,
    }


def compound_confusability_graph(
    reachable: Sequence[str], channels: Sequence[Mapping[str, Any]]
) -> dict[str, set[str]]:
    """Union the confusability graphs of all declared joint kernels."""
    if not channels:
        raise ValueError("GLOBAL-PUBLIC-CHECKPOINT-COUPLING is required")
    graph = {source: set() for source in reachable}
    for channel in channels:
        rows = _channel_rows(channel, reachable)
        supports = {
            source: {output for output, probability in row.items() if probability > 0.0}
            for source, row in rows.items()
        }
        for index, left in enumerate(reachable):
            for right in reachable[index + 1 :]:
                if supports[left] & supports[right]:
                    graph[left].add(right)
                    graph[right].add(left)
    return graph


def maximum_independent_set(graph: Mapping[str, set[str]]) -> list[str]:
    """Compute an exact finite maximum independent set by branch and bound."""
    vertices = tuple(sorted(graph))
    if len(vertices) > MAX_EXACT_MIS_VERTICES:
        raise ValueError("exact independent-set evaluator exceeds its vertex budget")
    if set(graph) != set(vertices) or any(not set(neighbors).issubset(graph) for neighbors in graph.values()):
        raise ValueError("confusability graph has unresolved vertices")
    best: tuple[str, ...] = ()
    visits = 0

    def search(candidates: tuple[str, ...], chosen: tuple[str, ...]) -> None:
        nonlocal best, visits
        visits += 1
        if visits > MAX_MIS_SEARCH_VISITS:
            raise ValueError("exact independent-set evaluator exceeds its search budget")
        if len(chosen) + len(candidates) <= len(best):
            return
        if not candidates:
            if len(chosen) > len(best):
                best = chosen
            return
        vertex = max(candidates, key=lambda item: len(graph[item].intersection(candidates)))
        remainder = tuple(item for item in candidates if item != vertex)
        compatible = tuple(item for item in remainder if item not in graph[vertex])
        search(compatible, chosen + (vertex,))
        search(remainder, chosen)

    search(vertices, ())
    return list(best)


def _decoder_success(rows: Mapping[str, Mapping[str, float]], code: Sequence[str]) -> float:
    outputs = sorted({output for source in code for output in rows[source]})
    if len(outputs) > MAX_DECODER_OUTPUTS:
        raise ValueError("decoder output alphabet exceeds the bounded exact evaluator")
    assignment_count = (len(code) + 1) ** len(outputs)
    if assignment_count > MAX_DECODER_ASSIGNMENTS:
        raise ValueError("decoder assignment search exceeds the bounded exact evaluator")
    best = 0.0
    targets: tuple[str | None, ...] = (None, *code)
    for assignment in product(targets, repeat=len(outputs)):
        decoded = dict(zip(outputs, assignment, strict=True))
        success = min(
            sum(probability for output, probability in rows[source].items() if decoded[output] == source)
            for source in code
        )
        best = max(best, success)
    return best


def approximate_public_capacity(
    reachable: Sequence[str],
    channels: Sequence[Mapping[str, Any]],
    epsilon: float,
    *,
    max_vertices: int = 12,
) -> dict[str, Any]:
    """Compute receipt-scale compound worst-input ``M_epsilon`` exactly."""
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(epsilon)
        or not 0.0 <= epsilon <= 1.0
    ):
        raise ValueError("epsilon must lie in [0,1]")
    if (
        isinstance(max_vertices, bool)
        or not isinstance(max_vertices, int)
        or max_vertices < 1
        or max_vertices > MAX_APPROXIMATE_CAPACITY_VERTICES
    ):
        raise ValueError("max_vertices is outside the bounded evaluator range")
    if not isinstance(reachable, Sequence) or isinstance(reachable, (str, bytes)):
        raise ValueError("reachable must be a bounded sequence of unique record identifiers")
    if not reachable:
        raise ValueError("reachable must contain at least one record identifier")
    if len(reachable) > max_vertices:
        raise ValueError("approximate evaluator is restricted to receipt-scale alphabets")
    for index, record_id in enumerate(reachable):
        _validate_identifier(record_id, f"reachable[{index}]")
    if len(set(reachable)) != len(reachable):
        raise ValueError("reachable must contain unique record identifiers")
    if (
        not isinstance(channels, Sequence)
        or isinstance(channels, (str, bytes))
        or not channels
        or len(channels) > MAX_CONTINUATIONS
        or any(not isinstance(channel, Mapping) for channel in channels)
    ):
        raise ValueError("channels must be a nonempty bounded checkpoint sequence")
    normalized = [_channel_rows(channel, reachable) for channel in channels]
    for size in range(len(reachable), 0, -1):
        for code in combinations(sorted(reachable), size):
            successes = [_decoder_success(rows, code) for rows in normalized]
            if min(successes) + 1.0e-12 >= 1.0 - epsilon:
                return {
                    "capacity": size,
                    "code_witness": list(code),
                    "worst_input_success_by_channel": successes,
                }
    return {"capacity": 0, "code_witness": [], "worst_input_success_by_channel": []}


def certify_capacity_carrier(
    capacity_dimension: int,
    reachable: Sequence[str],
    projection_supports: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    if (
        isinstance(capacity_dimension, bool)
        or not isinstance(capacity_dimension, int)
        or not 1 <= capacity_dimension <= MAX_CAPACITY_DIMENSION
    ):
        raise ValueError("capacity_dimension must be a positive integer")
    if not isinstance(projection_supports, Mapping):
        return _fail("NO_CAPACITY_CARRIER_REPRESENTATION")
    if set(projection_supports) != set(reachable):
        return _fail("NO_CAPACITY_CARRIER_REPRESENTATION")
    used: set[int] = set()
    ranks: dict[str, int] = {}
    for record in reachable:
        raw_support = projection_supports[record]
        if not isinstance(raw_support, Sequence) or isinstance(
            raw_support, (str, bytes)
        ):
            return _fail("NO_CAPACITY_CARRIER_REPRESENTATION", record=record)
        support = list(raw_support)
        if not support or len(set(support)) != len(support):
            return _fail("NO_CAPACITY_CARRIER_REPRESENTATION", record=record)
        if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < capacity_dimension for index in support):
            return _fail("NO_CAPACITY_CARRIER_REPRESENTATION", record=record)
        if used.intersection(support):
            return _fail("NO_CAPACITY_CARRIER_REPRESENTATION", record=record)
        used.update(support)
        ranks[record] = len(support)
    return {"status": "PASS", "projection_ranks": ranks, "rank_sum": len(used)}


def evaluate_terminal(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one finite ``PUBLIC_CHECKPOINT_PACKET`` without trusting gates."""
    if not isinstance(packet, Mapping):
        return _fail("INVALID_PACKET", reason="packet must be a mapping")
    try:
        packet_size = len(
            json.dumps(packet, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        return _fail("INVALID_PACKET", reason=f"packet is not canonical JSON: {exc}")
    if packet_size > MAX_PACKET_BYTES:
        return _fail("INVALID_PACKET", reason="packet exceeds the bounded byte limit")
    if set(packet) != _PACKET_FIELDS:
        return _fail("INVALID_PACKET", reason="packet has missing or unknown fields")
    for field in (
        "self_read_predicate_injected",
        "supplied_capacity_metadata_read_by_producer",
        "lambda_used",
        "ew_bridge_used",
        "rho_used",
        "continuation_manifest_complete",
        "local_marginal_consistency_passed",
    ):
        if not isinstance(packet.get(field), bool):
            return _fail("INVALID_PACKET", reason=f"{field} must be boolean")
    for field in ("schema", "status", "regulator_id", "carrier_type_id", "terminal_id"):
        try:
            _validate_identifier(packet.get(field), field)
        except ValueError as exc:
            return _fail("INVALID_PACKET", reason=str(exc))
    if packet.get("self_read_predicate_injected") or packet.get(
        "supplied_capacity_metadata_read_by_producer"
    ):
        return _fail("CIRCULAR_CAPACITY_DEFINITION")
    if packet.get("lambda_used") or packet.get("ew_bridge_used") or packet.get("rho_used"):
        return _fail("TARGET_TAINTED")
    try:
        sections = public_global_sections(packet["observers"], packet["interfaces"])
    except (KeyError, TypeError, ValueError) as exc:
        return _fail("NO_RECORD_ATOM_RESTRICTION", reason=str(exc))
    witnesses = packet.get("reachability_witnesses")
    if not isinstance(witnesses, Mapping):
        return _fail("NO_PUBLIC_RECORD_REACHABILITY")
    try:
        reachable = reachable_public_sections(sections, witnesses)
    except ValueError as exc:
        return _fail("NO_PUBLIC_RECORD_REACHABILITY", reason=str(exc))
    if not reachable:
        return _fail("NO_PUBLIC_RECORD_REACHABILITY")
    policy = packet.get("publicness_policy")
    if not isinstance(policy, Sequence) or isinstance(policy, (str, bytes)) or not policy:
        return _fail("NO_PUBLICNESS_POLICY")
    observer_ids = set(packet.get("observers", {}))
    normalized_policy: list[tuple[str, ...]] = []
    for authorized in policy:
        if (
            not isinstance(authorized, Sequence)
            or isinstance(authorized, (str, bytes))
            or not authorized
            or len(authorized) > MAX_OBSERVERS
            or any(not isinstance(item, str) for item in authorized)
            or len(set(authorized)) != len(authorized)
            or not set(authorized).issubset(observer_ids)
        ):
            return _fail("NO_PUBLICNESS_POLICY")
        normalized_policy.append(tuple(sorted(set(authorized))))
    if len(normalized_policy) != len(set(normalized_policy)):
        return _fail("NO_PUBLICNESS_POLICY", reason="duplicate publicness policy entries")
    channels = packet.get("global_checkpoint_kernels")
    if (
        not isinstance(channels, Sequence)
        or isinstance(channels, (str, bytes))
        or not channels
        or len(channels) > MAX_CONTINUATIONS
    ):
        return _fail("NO_GLOBAL_PUBLIC_CHECKPOINT_COUPLING")
    try:
        expected_continuations = _unique_identifier_manifest(
            packet.get("expected_continuation_ids"),
            "expected_continuation_ids",
            max_items=MAX_CONTINUATIONS,
        )
        receiver_known = _unique_identifier_manifest(
            packet.get("receiver_known_continuation_ids"),
            "receiver_known_continuation_ids",
            max_items=MAX_CONTINUATIONS,
        )
    except ValueError as exc:
        return _fail("INCOMPLETE_CONTINUATION_MANIFEST", reason=str(exc))
    if not packet.get("continuation_manifest_complete"):
        return _fail(
            "INCOMPLETE_CONTINUATION_MANIFEST",
            reason="declarative completeness flag is false",
        )
    continuation_ids: list[str] = []
    for channel in channels:
        if not isinstance(channel, Mapping):
            return _fail("NO_GLOBAL_PUBLIC_CHECKPOINT_COUPLING")
        if set(channel) != {"authorized_observers", "continuation_id", "rows"}:
            return _fail(
                "NO_GLOBAL_PUBLIC_CHECKPOINT_COUPLING",
                reason="checkpoint kernel has missing or unknown fields",
            )
        authorized = channel.get("authorized_observers")
        if not isinstance(authorized, Sequence) or isinstance(authorized, (str, bytes)):
            return _fail("NO_GLOBAL_PUBLIC_CHECKPOINT_COUPLING")
        continuation_id = channel.get("continuation_id")
        try:
            _validate_identifier(continuation_id, "continuation_id")
        except ValueError as exc:
            return _fail("NO_GLOBAL_PUBLIC_CHECKPOINT_COUPLING", reason=str(exc))
        if (
            len(authorized) > MAX_OBSERVERS
            or any(not isinstance(item, str) for item in authorized)
            or len(set(authorized)) != len(authorized)
            or tuple(sorted(authorized)) not in normalized_policy
        ):
            return _fail("NO_GLOBAL_PUBLIC_CHECKPOINT_COUPLING")
        continuation_ids.append(continuation_id)
    if len(continuation_ids) != len(set(continuation_ids)):
        return _fail(
            "INCOMPLETE_CONTINUATION_MANIFEST",
            reason="continuation_id values must be unique",
        )
    if set(continuation_ids) != set(expected_continuations):
        return _fail(
            "INCOMPLETE_CONTINUATION_MANIFEST",
            missing=sorted(set(expected_continuations) - set(continuation_ids)),
            unexpected=sorted(set(continuation_ids) - set(expected_continuations)),
        )
    if set(receiver_known) != set(expected_continuations):
        return _fail(
            "RECEIVER_CONTINUATION_KNOWLEDGE_INCOMPLETE",
            missing=sorted(set(expected_continuations) - set(receiver_known)),
            unexpected=sorted(set(receiver_known) - set(expected_continuations)),
        )
    try:
        marginal_audit = _recompute_local_marginal_consistency(
            channels,
            reachable,
            packet["observers"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _fail("LOCAL_MARGINAL_MISMATCH", reason=str(exc))
    try:
        graph = compound_confusability_graph(reachable, channels)
    except ValueError as exc:
        return _fail("NO_GLOBAL_PUBLIC_CHECKPOINT_COUPLING", reason=str(exc))
    try:
        carrier = certify_capacity_carrier(
            packet["capacity_dimension"], reachable, packet.get("projection_supports", {})
        )
    except (KeyError, ValueError) as exc:
        return _fail("NO_CAPACITY_CARRIER_REPRESENTATION", reason=str(exc))
    if carrier["status"] != "PASS":
        return carrier
    try:
        code = maximum_independent_set(graph)
    except ValueError as exc:
        return _fail("NO_CAPACITY_READBACK", reason=str(exc))
    capacity = len(code)
    dimension = int(packet["capacity_dimension"])
    return {
        "status": "PASS",
        "terminal_id": packet.get("terminal_id", "UNNAMED"),
        "public_global_section_count": len(sections),
        "reachable_public_sections": reachable,
        "publicness_policy": [list(authorized) for authorized in normalized_policy],
        "expected_continuation_ids": expected_continuations,
        "receiver_known_continuation_ids": receiver_known,
        "receiver_continuation_knowledge_complete": True,
        "canonical_section_id_scheme": SECTION_ID_SCHEME,
        **marginal_audit,
        "legacy_local_marginal_declaration": packet.get(
            "local_marginal_consistency_passed"
        ),
        "legacy_local_marginal_declaration_promoted": False,
        "confusability_graph": {key: sorted(value) for key, value in graph.items()},
        "independent_set_witness": code,
        "independence_number": capacity,
        "exact_zero_error_capacity_M0": capacity,
        "readback_record_count": capacity,
        "readback_nats": math.log(capacity) if capacity > 0 else None,
        "capacity_dimension_D": dimension,
        "input_coordinate_nats_log_D": math.log(dimension),
        "dimension_bound_passed": capacity <= dimension,
        "saturation_passed": capacity == dimension,
        "saturation_rank_one_complete": (
            capacity == dimension
            and len(reachable) == dimension
            and all(rank == 1 for rank in carrier["projection_ranks"].values())
        ),
        "capacity_slack_nats": math.log(dimension) - math.log(capacity),
        "equation": "M_0(q)=alpha(G_q); N=log M_0(U_N)",
    }


def evaluate_terminal_fiber(
    packets: Sequence[Mapping[str, Any]],
    *,
    expected_terminal_ids: Sequence[str],
) -> dict[str, Any]:
    """Scalarize only an explicitly enumerated, unique terminal fiber."""
    if not isinstance(packets, Sequence) or isinstance(packets, (str, bytes)):
        return _fail("INCOMPLETE_TERMINAL_FIBER", reason="packets must be a sequence")
    if len(packets) > MAX_TERMINAL_FIBER_PACKETS:
        return _fail(
            "INCOMPLETE_TERMINAL_FIBER",
            reason="terminal packet manifest exceeds the bounded evaluator",
        )
    try:
        expected = _unique_identifier_manifest(
            expected_terminal_ids,
            "expected_terminal_ids",
            max_items=MAX_TERMINAL_FIBER_PACKETS,
        )
    except ValueError as exc:
        return _fail("INCOMPLETE_TERMINAL_FIBER", reason=str(exc))
    if not packets:
        return _fail("NO_CAPACITY_READBACK")
    terminal_ids: list[str] = []
    aggregate_packet_bytes = 0
    for packet in packets:
        if not isinstance(packet, Mapping):
            return _fail("INCOMPLETE_TERMINAL_FIBER", reason="terminal packet is not a mapping")
        try:
            packet_bytes = len(
                json.dumps(
                    packet,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            return _fail(
                "INCOMPLETE_TERMINAL_FIBER",
                reason=f"terminal packet is not bounded canonical JSON: {exc}",
            )
        if packet_bytes > MAX_PACKET_BYTES:
            return _fail(
                "INCOMPLETE_TERMINAL_FIBER",
                reason="one terminal packet exceeds the bounded byte limit",
            )
        aggregate_packet_bytes += packet_bytes
        if aggregate_packet_bytes > MAX_TERMINAL_FIBER_BYTES:
            return _fail(
                "INCOMPLETE_TERMINAL_FIBER",
                reason="terminal packet manifest exceeds the aggregate byte budget",
            )
        terminal_id = packet.get("terminal_id")
        try:
            _validate_identifier(terminal_id, "terminal_id")
        except ValueError as exc:
            return _fail("INCOMPLETE_TERMINAL_FIBER", reason=str(exc))
        terminal_ids.append(terminal_id)
    if len(terminal_ids) != len(set(terminal_ids)):
        return _fail(
            "INCOMPLETE_TERMINAL_FIBER",
            reason="terminal_id values must be unique",
        )
    if set(terminal_ids) != set(expected):
        return _fail(
            "INCOMPLETE_TERMINAL_FIBER",
            missing=sorted(set(expected) - set(terminal_ids)),
            unexpected=sorted(set(terminal_ids) - set(expected)),
        )
    results = [evaluate_terminal(packet) for packet in packets]
    if any(result["status"] != "PASS" for result in results):
        return {"status": "NO_CAPACITY_READBACK", "terminal_results": results}
    capacities = [int(result["exact_zero_error_capacity_M0"]) for result in results]
    support = sorted(set(capacities))
    base = {
        "terminal_results": results,
        "expected_terminal_ids": expected,
        "observed_terminal_ids": sorted(terminal_ids),
        "terminal_manifest_complete": True,
        "terminal_fiber_capacity_set": support,
        "unclosed_readback_kernel": dict(Counter(capacities)),
    }
    if len(support) != 1:
        return {"status": "AMBIGUOUS_CAPACITY_READBACK", **base}
    dimension = int(results[0]["capacity_dimension_D"])
    if any(int(result["capacity_dimension_D"]) != dimension for result in results):
        raise ValueError("one terminal fiber must use one frozen carrier dimension")
    return {
        "status": "PASS",
        **base,
        "scalar_readback_dimension": support[0],
        "scalar_readback_nats": math.log(support[0]),
        "robust_closure": support == [dimension],
    }


def greatest_fixed_point(capacity_map: Mapping[int, int]) -> dict[str, Any]:
    """Compute the greatest fixed point; deliberately makes no uniqueness claim."""
    domain = sorted(capacity_map)
    if not domain or domain != list(range(1, domain[-1] + 1)):
        raise ValueError("domain must be the positive chain 1..D_max")
    if any(value not in capacity_map for value in capacity_map.values()):
        raise ValueError("map must be total and closed on the declared chain")
    if any(capacity_map[dimension] > dimension for dimension in domain):
        raise ValueError("active capacity map must be deflationary")
    if any(capacity_map[left] > capacity_map[right] for left, right in zip(domain, domain[1:], strict=False)):
        raise ValueError("capacity extension must be monotone")
    path = [domain[-1]]
    while capacity_map[path[-1]] != path[-1]:
        path.append(capacity_map[path[-1]])
    fixed = [dimension for dimension in domain if capacity_map[dimension] == dimension]
    return {
        "path": path,
        "fixed_points": fixed,
        "greatest_fixed_point": path[-1],
        "uniqueness_proved": len(fixed) == 1,
        "claim_boundary": "monotone deflationary iteration proves the greatest fixed point, not uniqueness",
    }


def no_new_confusability(
    coarse_graph: Mapping[str, set[str]],
    fine_graph: Mapping[str, set[str]],
    embedding: Mapping[str, str],
) -> bool:
    if set(embedding) != set(coarse_graph) or len(set(embedding.values())) != len(embedding):
        return False
    if any(image not in fine_graph for image in embedding.values()):
        return False
    return all(
        right in coarse_graph[left]
        or embedding[right] not in fine_graph[embedding[left]]
        for left in coarse_graph
        for right in coarse_graph
        if left != right
    )


def certify_unique_slack_zero(capacity_map: Mapping[int, int], selected: int) -> dict[str, Any]:
    zeros = sorted(dimension for dimension, value in capacity_map.items() if value == dimension)
    if selected not in capacity_map or zeros != [selected]:
        return _fail("FINITE_SIZE_SELECTOR_FAILED", fixed_points=zeros)
    return {"status": "PASS", "selected_dimension": selected, "fixed_points": zeros}


def tv_robustness_bound(epsilon: float, delta: float) -> float:
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
        for value in (epsilon, delta)
    ):
        raise ValueError("epsilon and delta must lie in [0,1]")
    return min(1.0, epsilon + delta)


def _validate_identifier(
    value: Any,
    field_name: str,
    *,
    max_bytes: int = MAX_IDENTIFIER_LENGTH,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or (byte_length := utf8_byte_length(value)) is None
        or byte_length > max_bytes
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ValueError(
            f"{field_name} must be a nonempty bounded string without control characters"
        )
    return value


def _unique_identifier_manifest(
    value: Any,
    field_name: str,
    *,
    max_items: int,
) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or len(value) > max_items
    ):
        raise ValueError(f"{field_name} must be a nonempty bounded sequence")
    manifest = [
        _validate_identifier(item, f"{field_name} item")
        for item in value
    ]
    if len(manifest) != len(set(manifest)):
        raise ValueError(f"{field_name} must contain unique identifiers")
    return manifest


def icosahedral_edges() -> list[tuple[str, str]]:
    """Return the 30 edges of the canonical twelve-port control screen."""
    edges: set[tuple[str, str]] = set()

    def add(left: str, right: str) -> None:
        edges.add(tuple(sorted((left, right))))

    for index in range(5):
        upper = f"upper_{index}"
        lower = f"lower_{index}"
        add("north", upper)
        add("south", lower)
        add(upper, f"upper_{(index + 1) % 5}")
        add(lower, f"lower_{(index + 1) % 5}")
        add(upper, lower)
        add(upper, f"lower_{(index - 1) % 5}")
    if len(edges) != 30:
        raise AssertionError("icosahedral edge construction must yield 30 edges")
    return sorted(edges)


def _reference_topology_matches(packet: Mapping[str, Any]) -> bool:
    observers = packet.get("observers")
    interfaces = packet.get("interfaces")
    if (
        packet.get("schema") != "PUBLIC_CHECKPOINT_PACKET/v1-reversible"
        or packet.get("status") != "REFERENCE_CONTROL_NOT_PHYSICAL_RECEIPT"
        or packet.get("regulator_id") != "icosahedral-12-port-reference-r0"
        or packet.get("carrier_type_id") != "public-record-carrier"
        or not isinstance(observers, Mapping)
        or set(observers) != set(PORTS)
        or not isinstance(interfaces, Sequence)
        or isinstance(interfaces, (str, bytes))
    ):
        return False
    dimension = packet.get("capacity_dimension")
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or not 2 <= dimension <= MAX_ATOMS_PER_OBSERVER
    ):
        return False
    atoms = [f"record_{index}" for index in range(dimension)]
    if any(list(observers[port]) != atoms for port in PORTS):
        return False
    expected_edges = set(icosahedral_edges())
    observed_edges: list[tuple[str, str]] = []
    for interface in interfaces:
        if not isinstance(interface, Mapping):
            return False
        left = interface.get("left_observer")
        right = interface.get("right_observer")
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        edge = tuple(sorted((left, right)))
        observed_edges.append(edge)
        if interface.get("interface_id") != f"{edge[0]}--{edge[1]}":
            return False
        identity = {atom: atom for atom in atoms}
        if interface.get("left_readout") != identity or interface.get("right_readout") != identity:
            return False
    if len(observed_edges) != len(set(observed_edges)) or set(observed_edges) != expected_edges:
        return False
    expected_continuations = {
        "identity",
        "cyclic_relabel",
        "orientation_reflection",
    }
    return bool(
        set(packet.get("expected_continuation_ids", ())) == expected_continuations
        and set(packet.get("receiver_known_continuation_ids", ()))
        == expected_continuations
        and packet.get("publicness_policy")
        and len(packet["publicness_policy"]) == 1
        and set(packet["publicness_policy"][0]) == set(PORTS)
    )


def _canonical_reference_generator_rows(
    sections: Sequence[Mapping[str, str]],
    dimension: int,
) -> dict[str, dict[str, dict[str, float]]]:
    label_to_section = {
        section["north"]: section_id(section)
        for section in sections
    }

    def rows(permutation: Mapping[int, int]) -> dict[str, dict[str, float]]:
        return {
            label_to_section[f"record_{index}"]: {
                label_to_section[f"record_{permutation[index]}"]: 1.0
            }
            for index in range(dimension)
        }

    identity = {index: index for index in range(dimension)}
    rotation = {index: (index + 1) % dimension for index in range(dimension)}
    reflection = {index: (-index) % dimension for index in range(dimension)}
    return {
        "identity": rows(identity),
        "cyclic_relabel": rows(rotation),
        "orientation_reflection": rows(reflection),
    }


def build_reference_packet(capacity_dimension: int = 4) -> dict[str, Any]:
    """Build the target-free reversible twelve-port schema/control packet."""
    if (
        isinstance(capacity_dimension, bool)
        or not isinstance(capacity_dimension, int)
        or not 2 <= capacity_dimension <= MAX_ATOMS_PER_OBSERVER
    ):
        raise ValueError("reference packet requires an integer dimension of at least two")
    atoms = [f"record_{index}" for index in range(capacity_dimension)]
    observers = {port: atoms for port in PORTS}
    interfaces = [
        {
            "interface_id": f"{left}--{right}",
            "left_observer": left,
            "right_observer": right,
            "left_readout": {atom: atom for atom in atoms},
            "right_readout": {atom: atom for atom in atoms},
        }
        for left, right in icosahedral_edges()
    ]
    sections = public_global_sections(observers, interfaces)
    section_ids = [section_id(section) for section in sections]
    label_to_section = {
        section["north"]: sid for section, sid in zip(sections, section_ids, strict=True)
    }

    def rows(permutation: Mapping[int, int]) -> dict[str, dict[str, float]]:
        return {
            label_to_section[f"record_{index}"]: {
                label_to_section[f"record_{permutation[index]}"]: 1.0
            }
            for index in range(capacity_dimension)
        }

    identity = {index: index for index in range(capacity_dimension)}
    rotate = {index: (index + 1) % capacity_dimension for index in range(capacity_dimension)}
    reflect = {index: (-index) % capacity_dimension for index in range(capacity_dimension)}
    all_ports = list(PORTS)
    expected_continuations = [
        "identity",
        "cyclic_relabel",
        "orientation_reflection",
    ]
    return {
        "schema": "PUBLIC_CHECKPOINT_PACKET/v1-reversible",
        "status": "REFERENCE_CONTROL_NOT_PHYSICAL_RECEIPT",
        "regulator_id": "icosahedral-12-port-reference-r0",
        "carrier_type_id": "public-record-carrier",
        "terminal_id": "icosahedral-reversible-q0",
        "capacity_dimension": capacity_dimension,
        "observers": observers,
        "interfaces": interfaces,
        "reachability_witnesses": {
            sid: ["source_seed", f"endogenous_write_{index}", "public_seam_check"]
            for index, sid in enumerate(section_ids)
        },
        "publicness_policy": [all_ports],
        "expected_continuation_ids": list(expected_continuations),
        "receiver_known_continuation_ids": list(expected_continuations),
        "continuation_manifest_complete": True,
        "global_checkpoint_kernels": [
            {"authorized_observers": all_ports, "continuation_id": "identity", "rows": rows(identity)},
            {"authorized_observers": all_ports, "continuation_id": "cyclic_relabel", "rows": rows(rotate)},
            {"authorized_observers": all_ports, "continuation_id": "orientation_reflection", "rows": rows(reflect)},
        ],
        "local_marginal_consistency_passed": True,
        "projection_supports": {sid: [index] for index, sid in enumerate(section_ids)},
        "self_read_predicate_injected": False,
        "supplied_capacity_metadata_read_by_producer": False,
        "lambda_used": False,
        "ew_bridge_used": False,
        "rho_used": False,
    }


def certify_reversible_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the bounded canonical reference packet and its permutations.

    A reversible packet on another topology may be a useful diagnostic, but it
    cannot receive the named twelve-port reference receipt.
    """

    evaluation = evaluate_terminal(packet)
    if evaluation["status"] != "PASS":
        return {
            "status": "INVALID_REFERENCE_PACKET",
            "reason": evaluation,
            "REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT": False,
        }
    try:
        sections = public_global_sections(packet["observers"], packet["interfaces"])
        reachable = reachable_public_sections(sections, packet["reachability_witnesses"])
        channels = packet["global_checkpoint_kernels"]
        reference_topology = _reference_topology_matches(packet)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {
            "status": "INVALID_REFERENCE_PACKET",
            "reason": str(exc),
            "REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT": False,
        }
    if not reference_topology:
        return {
            "status": "NONREFERENCE_REVERSIBLE_PACKET",
            "reference_topology_validated": False,
            "REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT": False,
            "PHYSICAL_N_CLOSURE_RECEIPT": False,
            "claim_boundary": (
                "Reversibility on an arbitrary observer topology does not earn the canonical "
                "twelve-port reference-control receipt."
            ),
        }
    expected_generator_rows = _canonical_reference_generator_rows(
        sections,
        int(packet["capacity_dimension"]),
    )
    generators: list[dict[str, Any]] = []
    for channel in channels:
        try:
            normalized = _channel_rows(channel, reachable)
        except (TypeError, ValueError, OverflowError) as exc:
            return {
                "status": "INVALID_CHECKPOINT_GENERATOR",
                "reason": str(exc),
                "REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT": False,
            }
        image: list[str] = []
        for source in reachable:
            positive = [(output, p) for output, p in normalized[source].items() if p > 0.0]
            if len(positive) != 1 or positive[0][1] != 1.0:
                return {
                    "status": "NONDETERMINISTIC_CHECKPOINT_GENERATOR",
                    "source": source,
                    "REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT": False,
                }
            image.append(positive[0][0])
        if len(set(image)) != len(reachable):
            return {
                "status": "NONINJECTIVE_CHECKPOINT_GENERATOR",
                "continuation_id": channel.get("continuation_id"),
                "REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT": False,
            }
        if set(image) != set(reachable):
            return {
                "status": "CHECKPOINT_NOT_CLOSED_ON_REACHABLE_RECORDS",
                "continuation_id": channel.get("continuation_id"),
                "REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT": False,
            }
        continuation_id = channel.get("continuation_id")
        if normalized != expected_generator_rows.get(continuation_id):
            return {
                "status": "NONCANONICAL_REFERENCE_GENERATOR",
                "continuation_id": continuation_id,
                "REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT": False,
                "PHYSICAL_N_CLOSURE_RECEIPT": False,
            }
        generators.append(
            {
                "continuation_id": continuation_id,
                "deterministic": True,
                "injective": True,
                "permutation_of_reachable_records": True,
                "receiver_knows_continuation": True,
                "canonical_reference_action_matches": True,
            }
        )
    count = len(reachable)
    if evaluation["exact_zero_error_capacity_M0"] != count:
        return {
            "status": "REVERSIBLE_CAPACITY_IDENTITY_FAILED",
            "reachable_count": count,
            "REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT": False,
        }
    return {
        "schema_version": "oph-public-record-capacity-v1",
        "status": "PASS",
        "packet_status": packet.get("status"),
        "regulator_id": packet.get("regulator_id"),
        "port_count": len(packet["observers"]),
        "interface_count": len(packet["interfaces"]),
        "public_global_section_count": len(sections),
        "reachable_public_record_count": count,
        "canonical_section_id_scheme": SECTION_ID_SCHEME,
        "expected_continuation_ids": evaluation["expected_continuation_ids"],
        "receiver_continuation_knowledge_complete": evaluation[
            "receiver_continuation_knowledge_complete"
        ],
        "local_marginal_consistency_recomputed": evaluation[
            "local_marginal_consistency_recomputed"
        ],
        "legacy_local_marginal_declaration_promoted": evaluation[
            "legacy_local_marginal_declaration_promoted"
        ],
        "reference_topology_validated": True,
        "canonical_reference_generator_actions_verified": True,
        "checkpoint_generator_receipts": generators,
        "exact_zero_error_capacity_M0": evaluation["exact_zero_error_capacity_M0"],
        "readback_nats_log_M0": evaluation["readback_nats"],
        "capacity_dimension_D": evaluation["capacity_dimension_D"],
        "fast_branch_identity": "M_0(q)=|X_reach(q)|",
        "robust_terminal_saturation": evaluation["saturation_passed"],
        "rank_one_complete": evaluation["saturation_rank_one_complete"],
        "REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT": True,
        "PHYSICAL_N_CLOSURE_RECEIPT": False,
        "remaining_physical_obligations": [
            "source-derived complete capacity-indexed trial universe and terminal fiber",
            "unique regulator-stable finite-size slack zero",
            "independent horizon-record carrier identification",
        ],
        "claim_boundary": (
            "Exact finite reversible control. It certifies the evaluator and twelve-port packet schema, "
            "not a source-derived physical universe packet or the cosmological value of N."
        ),
    }


def write_public_record_capacity_report(
    out_dir: Path,
    *,
    packet: Mapping[str, Any] | None = None,
    packet_path: Path | None = None,
    capacity_dimension: int = 4,
) -> dict[str, Any]:
    """Write the evaluated packet and fail-closed certificate bundle."""
    if packet is not None and packet_path is not None:
        raise ValueError("supply packet or packet_path, not both")
    if packet_path is not None:
        with Path(packet_path).open("rb") as handle:
            raw_packet = handle.read(MAX_PACKET_BYTES + 1)
        if len(raw_packet) > MAX_PACKET_BYTES:
            raise ValueError("packet file exceeds the bounded byte limit")
        loaded = json.loads(raw_packet)
        if not isinstance(loaded, Mapping):
            raise ValueError("packet JSON must contain an object")
        selected: Mapping[str, Any] = loaded
        reference_control = False
    elif packet is not None:
        selected = packet
        reference_control = False
    else:
        selected = build_reference_packet(capacity_dimension)
        reference_control = True
    try:
        compact_packet = json.dumps(
            selected,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("packet must be strict finite JSON data") from exc
    if len(compact_packet) > MAX_PACKET_BYTES:
        raise ValueError("packet exceeds the bounded byte limit")
    evaluation = evaluate_terminal(selected)
    reversible = certify_reversible_packet(selected)
    report = {
        "schema_version": "oph-public-record-capacity-bundle-v1",
        "reference_control": reference_control,
        "packet": selected,
        "evaluation": evaluation,
        "reversible_certificate": reversible,
        "physical_N_closure_receipt": False,
    }
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "public_checkpoint_packet.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "public_record_capacity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if reversible.get("REFERENCE_REVERSIBLE_PUBLIC_CHECKPOINT_RECEIPT") is True:
        (output / "public_record_capacity_reference_certificate.json").write_text(
            json.dumps(reversible, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report
