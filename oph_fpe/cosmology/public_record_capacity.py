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
    return "|".join(f"{observer}={section[observer]}" for observer in sorted(section))


def public_global_sections(
    observers: Mapping[str, Sequence[str]],
    interfaces: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Enumerate compatible global sections of finite local record atoms."""
    if not observers or any(not atoms for atoms in observers.values()):
        raise ValueError("every observer needs a nonempty record-atom set")
    observer_ids = sorted(observers)
    if any(len(set(observers[item])) != len(observers[item]) for item in observer_ids):
        raise ValueError("local record atoms must be unique")
    for interface in interfaces:
        left = interface.get("left_observer")
        right = interface.get("right_observer")
        if left not in observers or right not in observers:
            raise ValueError("interface references an unknown observer")
        left_map = interface.get("left_readout")
        right_map = interface.get("right_readout")
        if not isinstance(left_map, Mapping) or not isinstance(right_map, Mapping):
            raise ValueError("total RECORD-ATOM-RESTRICTION maps are required")
        if set(left_map) != set(observers[left]) or set(right_map) != set(observers[right]):
            raise ValueError("atom readout maps must be total on endpoint atoms")

    # Backtrack with early interface pruning.  A literal Cartesian product is
    # exact but needlessly enumerates D^12 assignments for the twelve-port
    # reference packet even though the first connected seam forces equality.
    incident: dict[str, list[Mapping[str, Any]]] = {item: [] for item in observer_ids}
    for interface in interfaces:
        incident[str(interface["left_observer"])].append(interface)
        incident[str(interface["right_observer"])].append(interface)
    sections: list[dict[str, str]] = []
    candidate: dict[str, str] = {}

    def extend(index: int) -> None:
        if index == len(observer_ids):
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
    valid = {section_id(section) for section in public_sections}
    unknown = set(reachability_witnesses) - valid
    if unknown:
        raise ValueError(f"reachability witnesses reference unknown sections: {sorted(unknown)}")
    return sorted(
        sid
        for sid, history in reachability_witnesses.items()
        if isinstance(history, Sequence)
        and not isinstance(history, (str, bytes))
        and len(history) > 0
    )


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
        values = {str(output): float(probability) for output, probability in raw.items()}
        if any(not math.isfinite(p) or p < 0.0 for p in values.values()):
            raise ValueError("checkpoint probabilities must be finite and nonnegative")
        if abs(sum(values.values()) - 1.0) > 1.0e-12:
            raise ValueError("checkpoint rows must be normalized")
        normalized[source] = values
    return normalized


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
    best: tuple[str, ...] = ()

    def search(candidates: tuple[str, ...], chosen: tuple[str, ...]) -> None:
        nonlocal best
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
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0,1]")
    if len(reachable) > max_vertices:
        raise ValueError("approximate evaluator is restricted to receipt-scale alphabets")
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
    if isinstance(capacity_dimension, bool) or not isinstance(capacity_dimension, int) or capacity_dimension < 1:
        raise ValueError("capacity_dimension must be a positive integer")
    if set(projection_supports) != set(reachable):
        return _fail("NO_CAPACITY_CARRIER_REPRESENTATION")
    used: set[int] = set()
    ranks: dict[str, int] = {}
    for record in reachable:
        support = list(projection_supports[record])
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
    if packet.get("self_read_predicate_injected", False) or packet.get(
        "supplied_capacity_metadata_read_by_producer", False
    ):
        return _fail("CIRCULAR_CAPACITY_DEFINITION")
    if packet.get("lambda_used", False) or packet.get("ew_bridge_used", False) or packet.get("rho_used", False):
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
            or not set(authorized).issubset(observer_ids)
        ):
            return _fail("NO_PUBLICNESS_POLICY")
        normalized_policy.append(tuple(sorted(set(authorized))))
    channels = packet.get("global_checkpoint_kernels")
    if not isinstance(channels, Sequence) or isinstance(channels, (str, bytes)) or not channels:
        return _fail("NO_GLOBAL_PUBLIC_CHECKPOINT_COUPLING")
    if not packet.get("continuation_manifest_complete", False):
        return _fail("NO_GLOBAL_PUBLIC_CHECKPOINT_COUPLING", reason="incomplete continuation manifest")
    for channel in channels:
        if not isinstance(channel, Mapping):
            return _fail("NO_GLOBAL_PUBLIC_CHECKPOINT_COUPLING")
        authorized = channel.get("authorized_observers")
        if not isinstance(authorized, Sequence) or isinstance(authorized, (str, bytes)):
            return _fail("NO_GLOBAL_PUBLIC_CHECKPOINT_COUPLING")
        if tuple(sorted(set(authorized))) not in normalized_policy or not channel.get("continuation_id"):
            return _fail("NO_GLOBAL_PUBLIC_CHECKPOINT_COUPLING")
    if not packet.get("local_marginal_consistency_passed", False):
        return _fail("LOCAL_MARGINAL_MISMATCH")
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
    code = maximum_independent_set(graph)
    capacity = len(code)
    dimension = int(packet["capacity_dimension"])
    return {
        "status": "PASS",
        "terminal_id": packet.get("terminal_id", "UNNAMED"),
        "public_global_section_count": len(sections),
        "reachable_public_sections": reachable,
        "publicness_policy": [list(authorized) for authorized in normalized_policy],
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
    packets: Sequence[Mapping[str, Any]], *, manifest_complete: bool
) -> dict[str, Any]:
    """Scalarize only a complete terminal fiber with a common capacity."""
    if not manifest_complete:
        return _fail("INCOMPLETE_TERMINAL_FIBER")
    if not packets:
        return _fail("NO_CAPACITY_READBACK")
    results = [evaluate_terminal(packet) for packet in packets]
    if any(result["status"] != "PASS" for result in results):
        return {"status": "NO_CAPACITY_READBACK", "terminal_results": results}
    capacities = [int(result["exact_zero_error_capacity_M0"]) for result in results]
    support = sorted(set(capacities))
    base = {
        "terminal_results": results,
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
    if not 0.0 <= epsilon <= 1.0 or not 0.0 <= delta <= 1.0:
        raise ValueError("epsilon and delta must lie in [0,1]")
    return min(1.0, epsilon + delta)


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


def build_reference_packet(capacity_dimension: int = 4) -> dict[str, Any]:
    """Build the target-free reversible twelve-port schema/control packet."""
    if isinstance(capacity_dimension, bool) or not isinstance(capacity_dimension, int) or capacity_dimension < 2:
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
    """Verify every checkpoint is a permutation and recompute its capacity."""
    try:
        sections = public_global_sections(packet["observers"], packet["interfaces"])
        reachable = reachable_public_sections(sections, packet["reachability_witnesses"])
        channels = packet["global_checkpoint_kernels"]
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "INVALID_REFERENCE_PACKET", "reason": str(exc)}
    generators: list[dict[str, Any]] = []
    for channel in channels:
        try:
            normalized = _channel_rows(channel, reachable)
        except ValueError as exc:
            return {"status": "INVALID_CHECKPOINT_GENERATOR", "reason": str(exc)}
        image: list[str] = []
        for source in reachable:
            positive = [(output, p) for output, p in normalized[source].items() if p > 0.0]
            if len(positive) != 1 or positive[0][1] != 1.0:
                return {"status": "NONDETERMINISTIC_CHECKPOINT_GENERATOR", "source": source}
            image.append(positive[0][0])
        if len(set(image)) != len(reachable):
            return {"status": "NONINJECTIVE_CHECKPOINT_GENERATOR", "continuation_id": channel.get("continuation_id")}
        if set(image) != set(reachable):
            return {"status": "CHECKPOINT_NOT_CLOSED_ON_REACHABLE_RECORDS", "continuation_id": channel.get("continuation_id")}
        generators.append(
            {
                "continuation_id": channel.get("continuation_id"),
                "deterministic": True,
                "injective": True,
                "permutation_of_reachable_records": True,
            }
        )
    evaluation = evaluate_terminal(packet)
    if evaluation["status"] != "PASS":
        return evaluation
    count = len(reachable)
    if evaluation["exact_zero_error_capacity_M0"] != count:
        return {"status": "REVERSIBLE_CAPACITY_IDENTITY_FAILED", "reachable_count": count}
    return {
        "schema_version": "oph-public-record-capacity-v1",
        "status": "PASS",
        "packet_status": packet.get("status"),
        "regulator_id": packet.get("regulator_id"),
        "port_count": len(packet["observers"]),
        "interface_count": len(packet["interfaces"]),
        "public_global_section_count": len(sections),
        "reachable_public_record_count": count,
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
        loaded = json.loads(Path(packet_path).read_text(encoding="utf-8"))
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
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "public_record_capacity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
