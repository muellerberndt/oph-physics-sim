"""Deterministic, target-blind finite source capture for the H3/KMS preflight.

This module stops before candidate clocks, geometry-model fitting, or campaign
outcomes.  It constructs and replays finite source-side diagnostics used by
preflight stages P0--P3: an exact twelve-port carrier federation, local
A5-equivariant propagation and seam repair, operational record feedback, a
commutative icosahedral refinement scaffold, a constructed M4/Gibbs matrix
diagnostic, and a source-derived ordered-frame cross-ratio sample on disjoint
rows.

The capture is a finite software-instrument receipt, not evidence for H3, KMS,
2pi, an event manifold, gravity, or the Standard Model.

The full frozen 262,144-carrier rung is representable, but this reference
implementation materializes its exact carrier, state, and log arrays.  Large
rungs therefore require remote memory sizing or a future streaming writer; the
caller must not assume that accepting the cardinality makes a local run cheap.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from oph_fpe.core.echosahedral_dynamics import (
    LocalRecurrentCarrierState,
    local_a5_dynamics_report,
    local_port_statistics,
    propagate_local_recurrent_carriers,
)
from oph_fpe.core.echosahedral_federation import (
    EchosahedralFederation,
    InterfaceAlgebraBinding,
    ObserverSupport,
    SeamBundle,
    TripleOverlapBundle,
    carrier_quotient_invariance_report,
    finite_matrix_interface_algebra_schema,
    federation_sewing_report,
    interface_algebra_sha256,
    reference_echosahedral_carrier,
    reference_federation_instrument_bundle,
    relabel_federation_ports,
)
from oph_fpe.core.icosahedral import (
    build_geodesic_icosahedral_tower,
    icosahedral_a5_equivariance_report,
    icosahedral_a5_port_permutations,
)


SCHEMA = "oph.physical-source-capture/1.0.0"
ARTIFACT_TYPE = "OPH_TARGET_BLIND_FINITE_SOURCE_CAPTURE"
VERIFIER_VERSION = "oph-target-blind-source-capture-replay-v1"
POSTRUN_CAPTURE_SCHEMA = "oph.physical_h3_kms.target_blind_capture.v3"
TYPED_CLOCK_PAIR_INPUT_SCHEMA = "oph.physical-source.typed-clock-pair-input.v1"
TYPED_CLOCK_PAIR_CONTRACT_SCHEMA = (
    "oph.physical-source.typed-clock-pair-contract.v1"
)
_INTRINSIC_STEP = 0.137
_COUPLING_STRENGTH = 1.0
_REPAIR_EVENT_EXAMPLE_LIMIT = 256
_CONSTRUCTED_M4_STATE_MODE = "constructed_m4_amplitude_gibbs_diagnostic"
_DEFAULT_PREREGISTERED_PLAN_SHA256 = "sha256:" + hashlib.sha256(
    b"oph.physical-source-capture.default-preregistered-plan.v1"
).hexdigest()
_ALLOWED_CONFIG_KEYS = frozenset(
    {
        "carrier_count",
        "seed",
        "rung",
        "replicate_id",
        "preregistered_plan_sha256",
        "intrinsic_step",
        "coupling_strength",
        "state_space",
        "rng_family",
        "initialization_distribution",
        "intrinsic_phase_distribution",
        "seam_update_rule",
        "cycles",
        "repair_fraction_per_cycle",
        "record_commit_cycles",
        "propagation_steps",
        "observer_count",
        "observer_support_size",
        "observer_samples",
        "prediction_control",
        "feedback_enabled",
        "checkpoint_interval",
        "support_refinement_level",
        "geometry_sample_count",
        "geometry_transport",
        "observer_cross_reads",
        "snapshot_coverage",
    }
)
_POSTRUN_COMPONENT_KEYS = (
    "registration",
    "carrier_port_trajectories",
    "intervention_rows",
    "response_rows",
    "clock_pair_input",
    "geometry_samples",
    "geometry_control_rows",
    "semantic_events",
    "raw_overlap_relations",
    "raw_ancestry_relations",
)
_CLOCK_PAIR_JOIN_KEY_FIELDS = (
    "intervention_id",
    "event_id",
    "observer_or_cap_id",
    "refinement_level",
    "trajectory_group_id",
)
_CLOCK_PAIR_GROUP_KEY_FIELDS = (
    "source_seed",
    "observer_or_cap_id",
    "trajectory_group_id",
)
_CLOCK_PAIR_INPUT_KEYS = frozenset(
    {"schema", "contract", "modular_transport_rows", "geometric_flow_rows"}
)
_CLOCK_PAIR_CONTRACT_KEYS = frozenset(
    {
        "schema",
        "status",
        "join_key_fields",
        "group_key_fields",
        "minimum_refinement_level_count",
        "modular_transport_producer_id",
        "modular_transport_producer_code_sha256",
        "modular_transport_source_field_sha256",
        "geometric_flow_producer_id",
        "geometric_flow_producer_code_sha256",
        "geometric_flow_source_field_sha256",
        "source_fixed_oriented_frame_incidence_required",
        "scoring_constants_absent",
        "unavailable_reason",
    }
)
_MODULAR_TRANSPORT_ROW_KEYS = frozenset(
    {
        "row_id",
        *_CLOCK_PAIR_JOIN_KEY_FIELDS,
        "source_seed",
        "modular_transport_time",
        "producer_source_field_sha256",
        "row_sha256",
    }
)
_GEOMETRIC_FLOW_ROW_KEYS = frozenset(
    {
        "row_id",
        *_CLOCK_PAIR_JOIN_KEY_FIELDS,
        "source_seed",
        "geometric_flow_parameter",
        "producer_source_field_sha256",
        "oriented_frame_incidence_sha256",
        "row_sha256",
    }
)
_POSTRUN_CAPTURE_KEYS = frozenset(
    {
        "schema",
        *_POSTRUN_COMPONENT_KEYS,
        "declared_hashes",
        "primitive_root_sha256",
    }
)
_POSTRUN_ROW_KEYS = {
    "carrier_port_trajectories": frozenset(
        {
            "carrier_id",
            "initial_port_intensities",
            "settled_port_intensities",
            "repaired_port_intensities",
            "initial_intrinsic_phase",
            "settled_intrinsic_phase",
        }
    ),
    "intervention_rows": frozenset(
        {
            "row_id",
            "operation",
            "seam_id",
            "read_set",
            "write_set",
            "mismatch_before",
            "mismatch_after",
        }
    ),
    "response_rows": frozenset(
        {
            "row_id",
            "record_event_id",
            "feedback_event_id",
            "observer_token",
            "carrier_id",
            "port",
            "independent_geometric_parameter",
            "raw_response",
            "initial_port_intensity",
            "settled_port_intensity",
            "repaired_port_intensity",
            "propagation_delta",
            "repair_delta",
            "refinement_level",
            "geometry_source_row_id",
            "row_sha256",
        }
    ),
    "geometry_control_rows": frozenset(
        {
            "row_id",
            "source_geometry_row_id",
            "source_response_row_id",
            "observer_token",
            "trajectory_group_id",
            "ordered_frame",
            "orientation",
            "cross_ratio",
            "geometric_parameter",
            "neutral_feature_vector",
            "observed_source_value",
            "predictor_source_phase",
            "response_source_phase",
            "predictor_response_field_intersection",
            "row_sha256",
        }
    ),
    "semantic_events": frozenset(
        {
            "event_key",
            "canonical_semantic_payload",
            "observer_token",
            "visible_footprint",
            "parent_event_ids",
            "read_resource_ids",
            "write_resource_ids",
            "source_sequence_index",
        }
    ),
    "raw_overlap_relations": frozenset(
        {
            "overlap_id",
            "left_carrier_id",
            "right_carrier_id",
            "left_ports",
            "right_ports",
            "left_to_right_ports",
            "right_to_left_ports",
            "orientation_signs",
            "visible_to_observer_tokens",
            "interface_algebra_sha256",
            "row_sha256",
        }
    ),
    "raw_ancestry_relations": frozenset(
        {
            "parent_event_id",
            "child_event_id",
            "observer_token",
            "parent_sequence_index",
            "child_sequence_index",
            "shared_resource_ids",
            "edge_id",
        }
    ),
}
_POSTRUN_GEOMETRY_SAMPLE_KEYS = frozenset(
    {
        "derivation_method",
        "orientation_fixed_from_source",
        "raw_primitive_rows",
        "support_refinement_levels",
        "conditional_expectation_rows",
    }
)
_POSTRUN_FORBIDDEN_KEYS = frozenset(
    {
        "candidate",
        "candidates",
        "candidate_labels",
        "event_position",
        "geometries",
        "h3_frame",
        "lorentz",
        "model",
        "models",
        "relation",
        "selected_model",
        "selected_scale",
        "target",
        "translation",
    }
)
_POSTRUN_FORBIDDEN_LABELS = frozenset(
    {
        "1x",
        "pi",
        "2pi",
        "4pi",
        "h3",
        "s2",
        "e3",
        "e4",
        "lorentz",
        "causal",
        "spacelike",
        "null",
    }
)
_FORBIDDEN_SOURCE_TOKENS = (
    "target",
    "candidate",
    "h3",
    "kms",
    "2pi",
    "4pi",
    "kappa",
    "temperature",
)


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


def _clean_float(value: float) -> float:
    result = round(float(value), 15)
    return 0.0 if result == 0.0 else result


def _complex_rows(value: np.ndarray) -> list[list[list[float]]]:
    array = np.asarray(value, dtype=np.complex128)
    return [
        [[_clean_float(item.real), _clean_float(item.imag)] for item in row]
        for row in array
    ]


def _real_rows(value: np.ndarray) -> list[list[float]]:
    return [
        [_clean_float(item) for item in row]
        for row in np.asarray(value, dtype=float)
    ]


def _matrix_pairs(value: np.ndarray) -> list[list[list[float]]]:
    return _complex_rows(np.asarray(value, dtype=np.complex128))


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise TypeError("source capture config must be a mapping")
    unknown = set(config) - _ALLOWED_CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown source-capture config fields: {sorted(unknown)}")
    defaults = {
        "carrier_count": 8,
        "seed": 20260751,
        "propagation_steps": 4,
        "cycles": 16,
        "record_commit_cycles": 4,
        "observer_count": 2,
        "observer_support_size": 2,
        "observer_samples": 6,
        "checkpoint_interval": 4,
        "support_refinement_level": 2,
        "geometry_sample_count": 4,
    }
    values: dict[str, Any] = {}
    for key, default in defaults.items():
        raw = config.get(key, default)
        if type(raw) is not int:
            raise TypeError(f"{key} must be an exact integer")
        values[key] = raw
    for key, default in (
        ("intrinsic_step", _INTRINSIC_STEP),
        ("coupling_strength", _COUPLING_STRENGTH),
        ("repair_fraction_per_cycle", 0.0625),
    ):
        raw = config.get(key, default)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"{key} must be a strict finite number")
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{key} must be finite and positive")
        if key == "repair_fraction_per_cycle" and value > 1.0:
            raise ValueError("repair_fraction_per_cycle must not exceed one")
        values[key] = value
    raw_rung = config.get("rung", values["carrier_count"])
    if type(raw_rung) is not int:
        raise TypeError("rung must be an exact integer")
    values["rung"] = raw_rung
    replicate_id = config.get("replicate_id", "primary")
    if (
        not isinstance(replicate_id, str)
        or not replicate_id
        or len(replicate_id) > 128
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", replicate_id)
    ):
        raise ValueError("replicate_id must be a nonempty bounded identifier")
    values["replicate_id"] = replicate_id
    plan_sha256 = config.get(
        "preregistered_plan_sha256", _DEFAULT_PREREGISTERED_PLAN_SHA256
    )
    if not isinstance(plan_sha256, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", plan_sha256
    ):
        raise ValueError("preregistered_plan_sha256 must be a strict SHA-256 digest")
    values["preregistered_plan_sha256"] = plan_sha256
    declarative_inputs = {
        "state_space": "normalized_complex_amplitude_in_C12",
        "rng_family": "numpy_generator_pcg64_v1",
        "initialization_distribution": "normalized_complex_gaussian_v1",
        "intrinsic_phase_distribution": "uniform_unit_interval_v1",
        "seam_update_rule": (
            "disjoint_single_port_endpoint_arithmetic_mean_v1"
        ),
        "prediction_control": "semantic_hash_shuffle_v1",
    }
    for key, expected in declarative_inputs.items():
        raw = config.get(key, expected)
        if raw != expected:
            raise ValueError(f"{key} must equal the registered executable literal")
        values[key] = expected
    feedback_enabled = config.get("feedback_enabled", True)
    if type(feedback_enabled) is not bool:
        raise TypeError("feedback_enabled must be an exact boolean")
    if feedback_enabled is not True:
        raise ValueError("feedback_enabled must be true for the registered source")
    values["feedback_enabled"] = feedback_enabled
    geometry_transport = config.get("geometry_transport", "legacy")
    if geometry_transport not in ("legacy", "held_out_flow"):
        raise ValueError("geometry_transport must be 'legacy' or 'held_out_flow'")
    values["geometry_transport"] = geometry_transport
    observer_cross_reads = config.get("observer_cross_reads", False)
    if type(observer_cross_reads) is not bool:
        raise TypeError("observer_cross_reads must be an exact boolean")
    values["observer_cross_reads"] = observer_cross_reads
    snapshot_coverage = config.get("snapshot_coverage", "support")
    if snapshot_coverage not in ("support", "spanning"):
        raise ValueError("snapshot_coverage must be 'support' or 'spanning'")
    values["snapshot_coverage"] = snapshot_coverage
    if any(
        re.search(
            rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])",
            replicate_id.lower(),
        )
        for label in (_FORBIDDEN_SOURCE_TOKENS + tuple(_POSTRUN_FORBIDDEN_LABELS))
    ):
        raise ValueError("replicate_id must not contain a scoring or model label")
    limits = {
        "carrier_count": (2, 262_144),
        "seed": (0, 2**63 - 1),
        "rung": (2, 262_144),
        "propagation_steps": (1, 64),
        "cycles": (1, 4096),
        "record_commit_cycles": (1, 4096),
        "observer_count": (1, values["carrier_count"]),
        "observer_support_size": (1, values["carrier_count"]),
        "observer_samples": (3, 64),
        "checkpoint_interval": (1, 1_000_000),
        "support_refinement_level": (1, 7),
        "geometry_sample_count": (3, 32),
    }
    for key, (lower, upper) in limits.items():
        if not lower <= values[key] <= upper:
            raise ValueError(f"{key} must lie in [{lower}, {upper}]")
    if values["rung"] != values["carrier_count"]:
        raise ValueError("rung must equal the exact source carrier_count")
    if values["carrier_count"] % 2:
        raise ValueError("carrier_count must be even for the all-port federation")
    if values["record_commit_cycles"] > values["cycles"]:
        raise ValueError("record_commit_cycles must not exceed cycles")
    return values


def _topology_seed(config: Mapping[str, Any], domain: str) -> int:
    """Return a domain-separated seed without consuming the source RNG stream."""

    material = (
        f"oph-source-federation-v2:{config['seed']}:{config['replicate_id']}:{domain}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _build_federation(config: Mapping[str, Any]) -> EchosahedralFederation:
    """Build a connected target-blind federation that sews every local port.

    Ports 0 and 1 form alternating perfect matchings on one seeded carrier
    ordering, so their union is a Hamiltonian cycle.  Each remaining port uses
    an independently seeded perfect matching.  Thus every one of the twelve
    ports of every (even-cardinality) carrier participates in exactly one seam.
    The lexicographically first triangle in the resulting source graph supplies
    an explicit nonempty triple-overlap restriction. Neither the support
    regulator nor any downstream geometry label can affect source topology.
    """

    count = config["carrier_count"]
    if count % 2:
        raise ValueError("all-port source federation requires an even carrier_count")
    carrier_ids = tuple(f"carrier-{index:05d}" for index in range(count))
    carriers = tuple(reference_echosahedral_carrier(item) for item in carrier_ids)
    interface_id = "finite-overlap-visible-algebra-v1"
    algebra_hash = interface_algebra_sha256(
        finite_matrix_interface_algebra_schema(interface_id, 1)
    )
    binding = InterfaceAlgebraBinding(
        interface_algebra_id=interface_id,
        interface_algebra_sha256=algebra_hash,
        left_interface_algebra_sha256=algebra_hash,
        right_interface_algebra_sha256=algebra_hash,
    )
    base_order = np.random.Generator(
        np.random.PCG64(_topology_seed(config, "connected-order"))
    ).permutation(count).tolist()
    port_orders: list[list[int]] = [base_order, base_order[1:] + base_order[:1]]
    for port in range(2, 12):
        port_orders.append(
            np.random.Generator(
                np.random.PCG64(_topology_seed(config, f"port-{port:02d}"))
            ).permutation(count).tolist()
        )

    seams_list: list[SeamBundle] = []
    for port, order in enumerate(port_orders):
        for pair_index in range(0, count, 2):
            left_index = int(order[pair_index])
            right_index = int(order[pair_index + 1])
            seams_list.append(
                SeamBundle(
                    seam_id=f"seam-p{port:02d}-{pair_index // 2:06d}",
                    left_carrier_id=carrier_ids[left_index],
                    right_carrier_id=carrier_ids[right_index],
                    left_ports=(port,),
                    right_ports=(port,),
                    left_to_right_ports=(port,),
                    right_to_left_ports=(port,),
                    left_to_right_orientation=(-1,),
                    right_to_left_orientation=(-1,),
                    collar_kind="single_port",
                    interface_algebra=binding,
                )
            )
    seams = tuple(seams_list)

    adjacency: dict[str, set[str]] = {carrier_id: set() for carrier_id in carrier_ids}
    seam_ids_by_pair: dict[frozenset[str], set[str]] = {}
    for seam in seams:
        adjacency[seam.left_carrier_id].add(seam.right_carrier_id)
        adjacency[seam.right_carrier_id].add(seam.left_carrier_id)
        seam_ids_by_pair.setdefault(
            frozenset((seam.left_carrier_id, seam.right_carrier_id)), set()
        ).add(seam.seam_id)

    triple_overlaps: tuple[TripleOverlapBundle, ...] = ()
    for left in carrier_ids:
        for middle in sorted(item for item in adjacency[left] if item > left):
            candidates = sorted(
                item
                for item in adjacency[left] & adjacency[middle]
                if item > middle
            )
            if not candidates:
                continue
            right = candidates[0]
            pairs = (
                frozenset((left, middle)),
                frozenset((middle, right)),
                frozenset((right, left)),
            )
            triple_overlaps = (
                TripleOverlapBundle(
                    overlap_id="source-graph-triple-0000",
                    oriented_carrier_ids=(left, middle, right),
                    oriented_seam_ids=tuple(
                        sorted(seam_ids_by_pair[pair])[0] for pair in pairs
                    ),
                    restriction_algebra_id=interface_id,
                    restriction_algebra_sha256=algebra_hash,
                ),
            )
            break
        if triple_overlaps:
            break
    if not triple_overlaps:
        raise RuntimeError(
            "source federation has no nonempty triangle for the triple-overlap contract"
        )

    width = min(config["observer_support_size"], count)
    supports: list[ObserverSupport] = []
    for observer_index in range(config["observer_count"]):
        anchor = carrier_ids[(observer_index * count) // config["observer_count"]]
        selected = {anchor}
        queue = deque([anchor])
        while queue and len(selected) < width:
            current = queue.popleft()
            for neighbor in sorted(adjacency[current]):
                if neighbor not in selected:
                    selected.add(neighbor)
                    queue.append(neighbor)
                    if len(selected) == width:
                        break
        ids = frozenset(selected)
        visible = frozenset(
            seam_id
            for pair, seam_ids in seam_ids_by_pair.items()
            if pair <= ids
            for seam_id in seam_ids
        )
        supports.append(
            ObserverSupport(
                observer_token=f"observer-{observer_index:04d}",
                carrier_ids=ids,
                visible_seam_ids=visible,
                record_algebra_sha256=interface_algebra_sha256(
                    {"kind": "finite-central-record", "observer": observer_index}
                ),
                checkpoint_cut_sha256=interface_algebra_sha256(
                    {"kind": "semantic-history-cut", "observer": observer_index}
                ),
            )
        )
    return EchosahedralFederation(
        federation_id=f"source-all-port-federation-{count}-seed-{config['seed']}",
        carriers=carriers,
        seams=seams,
        triple_overlaps=triple_overlaps,
        external_boundaries=(),
        observer_supports=tuple(supports),
    )


def _initialize_source_carriers(
    carrier_count: int,
    seed: int,
) -> LocalRecurrentCarrierState:
    rng = np.random.Generator(np.random.PCG64(seed))
    amplitudes = rng.normal(size=(carrier_count, 12)) + 1j * rng.normal(
        size=(carrier_count, 12)
    )
    amplitudes = np.asarray(
        amplitudes / np.linalg.norm(amplitudes, axis=1, keepdims=True),
        dtype=np.complex128,
    )
    phase = rng.random(carrier_count, dtype=np.float64)
    return LocalRecurrentCarrierState(
        amplitudes=amplitudes,
        intrinsic_phase=phase,
    )


def _array_sha256(value: np.ndarray) -> str:
    """Bind a finite numeric ledger with an endian-stable byte encoding."""

    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    hasher = hashlib.sha256()
    hasher.update(b"oph.numeric-array.v1\0")
    hasher.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(array.tobytes(order="C"))
    return "sha256:" + hasher.hexdigest()


def _record_commit_schedule(cycles: int, count: int) -> tuple[int, ...]:
    schedule = tuple(((index + 1) * cycles) // count - 1 for index in range(count))
    if len(schedule) != len(set(schedule)) or schedule[-1] != cycles - 1:
        raise RuntimeError("internal record-commit schedule is not exact and terminal")
    return schedule


def _source_dynamics(
    config: Mapping[str, Any], federation: EchosahedralFederation
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    """Run local propagation followed by the frozen transactional repair schedule.

    The normalized complex recurrent state and its mutable overlap-visible
    response ledger are kept as distinct typed fields of one joint local state.
    Repair acts on the visible ledger.  A phase-preserving terminal complex lift
    is exported for finite-algebra instrumentation, but no physical lift theorem
    is claimed.
    """

    state = _initialize_source_carriers(config["carrier_count"], config["seed"])
    initial_rows = _complex_rows(state.amplitudes)
    initial_phase = [_clean_float(item) for item in state.intrinsic_phase]
    initial_norm_residual = float(
        np.max(np.abs(np.linalg.norm(state.amplitudes, axis=1) - 1.0))
    )
    for _ in range(config["propagation_steps"]):
        state = propagate_local_recurrent_carriers(
            state,
            intrinsic_step=config["intrinsic_step"],
            coupling_strength=config["coupling_strength"],
        )
    propagated = local_port_statistics(state)
    repaired = np.array(propagated, copy=True)
    versions = np.zeros(repaired.shape, dtype=np.int32)
    first_examples: list[dict[str, Any]] = []
    last_examples: deque[dict[str, Any]] = deque(
        maxlen=_REPAIR_EVENT_EXAMPLE_LIMIT // 2
    )
    event_hasher = hashlib.sha256()
    event_count = 0
    noop_count = 0
    before_total = 0.0
    after_total = 0.0
    seams = federation.seams
    seam_count = len(seams)
    repair_count_per_cycle = max(
        1, int(math.ceil(config["repair_fraction_per_cycle"] * seam_count))
    )
    order = np.random.Generator(
        np.random.PCG64(_topology_seed(config, "repair-order"))
    ).permutation(seam_count)
    support_indices = sorted(
        {
            int(carrier_id.rsplit("-", 1)[1])
            for support in federation.observer_supports
            for carrier_id in support.carrier_ids
        }
    )
    if str(config.get("snapshot_coverage", "support")) == "spanning":
        # Spanning coverage: add a deterministic stride subset so the sampled
        # record vectors span port space and the whole federation, bounded to
        # 256 extra carriers so snapshot payloads stay small at large rungs.
        carrier_total = int(config["carrier_count"])
        stride = max(1, carrier_total // 256)
        support_indices = sorted(
            set(support_indices) | set(range(0, carrier_total, stride))
        )
    record_schedule = _record_commit_schedule(
        config["cycles"], config["record_commit_cycles"]
    )
    record_schedule_set = set(record_schedule)
    record_state_snapshots: list[dict[str, Any]] = []
    cycle_ledger: list[dict[str, Any]] = []
    transaction_validation_complete = True
    union_atomic_revalidation = True
    order_replay_exact = True
    idempotence_replay_exact = True
    for cycle in range(config["cycles"]):
        state_before = np.array(repaired, copy=True)
        version_before = np.array(versions, copy=True)
        start = (cycle * repair_count_per_cycle) % seam_count
        selected_indices = [
            int(order[(start + offset) % seam_count])
            for offset in range(repair_count_per_cycle)
        ]
        if len(selected_indices) != len(set(selected_indices)):
            raise RuntimeError("repair cycle selected a seam more than once")
        selected_endpoint_count = 0
        cycle_commits: list[tuple[int, int, int, int, float, float, Any]] = []
        cycle_noops = 0
        selected_material: list[str] = []
        endpoint_keys: set[tuple[int, int]] = set()
        for transaction_index, seam_index in enumerate(selected_indices):
            seam = seams[seam_index]
            left = int(seam.left_carrier_id.rsplit("-", 1)[1])
            right = int(seam.right_carrier_id.rsplit("-", 1)[1])
            left_port = int(seam.left_ports[0])
            right_port = int(seam.right_ports[0])
            left_key = (left, left_port)
            right_key = (right, right_port)
            if left_key in endpoint_keys or right_key in endpoint_keys:
                transaction_validation_complete = False
            endpoint_keys.update((left_key, right_key))
            selected_endpoint_count += 2
            left_value = float(state_before[left, left_port])
            right_value = float(state_before[right, right_port])
            before = abs(left_value - right_value)
            selected_material.append(seam.seam_id)
            if before <= 1.0e-15:
                cycle_noops += 1
                noop_count += 1
                continue
            average = 0.5 * (left_value + right_value)
            cycle_commits.append(
                (
                    left,
                    left_port,
                    right,
                    right_port,
                    left_value,
                    right_value,
                    seam,
                )
            )

        reverse_replay = np.array(state_before, copy=True)
        for left, left_port, right, right_port, _, _, _ in reversed(cycle_commits):
            average = 0.5 * (
                float(state_before[left, left_port])
                + float(state_before[right, right_port])
            )
            reverse_replay[left, left_port] = average
            reverse_replay[right, right_port] = average

        for transaction_index, (
            left,
            left_port,
            right,
            right_port,
            left_value,
            right_value,
            seam,
        ) in enumerate(cycle_commits):
            if (
                versions[left, left_port] != version_before[left, left_port]
                or versions[right, right_port] != version_before[right, right_port]
            ):
                union_atomic_revalidation = False
                continue
            average = 0.5 * (left_value + right_value)
            repaired[left, left_port] = average
            repaired[right, right_port] = average
            versions[left, left_port] += 1
            versions[right, right_port] += 1
            after = abs(
                float(repaired[left, left_port] - repaired[right, right_port])
            )
            material = {
                "cycle": cycle,
                "transaction_index": transaction_index,
                "seam_id": seam.seam_id,
                "read_set": [
                    {
                        "carrier_id": seam.left_carrier_id,
                        "port": left_port,
                        "version": int(version_before[left, left_port]),
                        "value": _clean_float(left_value),
                    },
                    {
                        "carrier_id": seam.right_carrier_id,
                        "port": right_port,
                        "version": int(version_before[right, right_port]),
                        "value": _clean_float(right_value),
                    },
                ],
                "write_set": [
                    {
                        "carrier_id": seam.left_carrier_id,
                        "port": left_port,
                        "expected_version": int(version_before[left, left_port]),
                        "committed_version": int(versions[left, left_port]),
                        "value": _clean_float(average),
                    },
                    {
                        "carrier_id": seam.right_carrier_id,
                        "port": right_port,
                        "expected_version": int(version_before[right, right_port]),
                        "committed_version": int(versions[right, right_port]),
                        "value": _clean_float(average),
                    },
                ],
                "mismatch_before": _clean_float(abs(left_value - right_value)),
                "mismatch_after": _clean_float(after),
                "strict_descent": bool(after < abs(left_value - right_value)),
                "update": "endpoint_arithmetic_mean",
            }
            event = {**material, "event_id": _sha(material)}
            encoded = _canonical_bytes(event)
            event_hasher.update(len(encoded).to_bytes(8, "big"))
            event_hasher.update(encoded)
            if len(first_examples) < _REPAIR_EVENT_EXAMPLE_LIMIT // 2:
                first_examples.append(event)
            else:
                last_examples.append(event)
            event_count += 1
            before_total += abs(left_value - right_value)
            after_total += after

        order_replay_exact = bool(
            order_replay_exact and np.array_equal(reverse_replay, repaired)
        )
        idempotence_probe = np.array(repaired, copy=True)
        for left, left_port, right, right_port, _, _, _ in cycle_commits:
            average = 0.5 * (
                float(idempotence_probe[left, left_port])
                + float(idempotence_probe[right, right_port])
            )
            idempotence_probe[left, left_port] = average
            idempotence_probe[right, right_port] = average
        idempotence_replay_exact = bool(
            idempotence_replay_exact and np.array_equal(idempotence_probe, repaired)
        )
        cycle_ledger.append(
            {
                "cycle": cycle,
                "snapshot_state_sha256": _array_sha256(state_before),
                "selected_seam_count": len(selected_indices),
                "selected_endpoint_count": selected_endpoint_count,
                "selected_seam_ids_sha256": _sha(selected_material),
                "committed_transaction_count": len(cycle_commits),
                "skipped_noop_count": cycle_noops,
                "committed_state_sha256": _array_sha256(repaired),
            }
        )
        if cycle in record_schedule_set:
            carrier_rows = [
                {
                    "carrier_id": f"carrier-{index:05d}",
                    "full_port_state": _real_rows(repaired[index : index + 1])[0],
                    "full_port_state_sha256": _sha(
                        _real_rows(repaired[index : index + 1])[0]
                    ),
                }
                for index in support_indices
            ]
            record_state_snapshots.append(
                {
                    "cycle": cycle,
                    "visible_state_sha256": _array_sha256(repaired),
                    "carrier_rows": carrier_rows,
                    "carrier_rows_sha256": _sha(carrier_rows),
                }
            )

    events = first_examples + list(last_examples)
    terminal_mismatches = []
    for seam in seams:
        left = int(seam.left_carrier_id.rsplit("-", 1)[1])
        right = int(seam.right_carrier_id.rsplit("-", 1)[1])
        terminal_mismatches.append(
            abs(
                float(
                    repaired[left, seam.left_ports[0]]
                    - repaired[right, seam.right_ports[0]]
                )
            )
        )
    phases = np.angle(state.amplitudes)
    terminal_lift = np.sqrt(np.maximum(repaired, 0.0)) * np.exp(1j * phases)
    audit = local_a5_dynamics_report(
        intrinsic_step=config["intrinsic_step"],
        coupling_strength=config["coupling_strength"],
    )
    primitives = {
        "algorithm_id": "seeded-a5-propagation-transactional-visible-repair-v2",
        "seed": config["seed"],
        "state_space": config["state_space"],
        "rng_family": config["rng_family"],
        "initialization_distribution": config["initialization_distribution"],
        "intrinsic_phase_distribution": config["intrinsic_phase_distribution"],
        "initial_port_amplitudes": initial_rows,
        "initial_intrinsic_phase": initial_phase,
        "initial_state_sha256": _sha(
            {"amplitudes": initial_rows, "intrinsic_phase": initial_phase}
        ),
        "initial_maximum_unit_norm_residual": initial_norm_residual,
        "propagation_steps": config["propagation_steps"],
        "intrinsic_step": config["intrinsic_step"],
        "coupling_strength": config["coupling_strength"],
        "propagated_port_statistics_sha256": _sha(_real_rows(propagated)),
        "repaired_port_statistics_sha256": _sha(_real_rows(repaired)),
        "joint_terminal_state_sha256": _sha(
            {
                "hidden_recurrent_amplitudes": _complex_rows(state.amplitudes),
                "visible_repaired_port_state": _real_rows(repaired),
            }
        ),
        "terminal_complex_lift_sha256": _sha(_complex_rows(terminal_lift)),
        "terminal_complex_lift_semantics": (
            "phase_preserving_instrument_projection_from_joint_hidden_visible_state"
        ),
        "PHYSICAL_COMPLEX_REPAIR_LIFT_RECEIPT": False,
        "local_a5_equivariance": {
            "receipt": audit["LOCAL_A5_EQUIVARIANT_PROPAGATION_RECEIPT"],
            "maximum_commutator_residual": audit[
                "maximum_a5_coupling_commutator_residual"
            ],
            "maximum_propagation_residual": audit[
                "maximum_a5_propagation_equivariance_residual"
            ],
            "dynamics_sha256": audit["dynamics_sha256"],
        },
        "repair_rule": config["seam_update_rule"],
        "cycles": config["cycles"],
        "repair_fraction_per_cycle": config["repair_fraction_per_cycle"],
        "repair_count_per_cycle": repair_count_per_cycle,
        "record_commit_cycles": config["record_commit_cycles"],
        "record_commit_schedule": list(record_schedule),
        "record_state_snapshots": record_state_snapshots,
        "record_state_snapshots_sha256": _sha(record_state_snapshots),
        "repair_mismatch_before": _clean_float(before_total),
        "repair_mismatch_after": _clean_float(after_total),
        "terminal_maximum_seam_mismatch": _clean_float(max(terminal_mismatches)),
        "repair_event_count": event_count,
        "repair_noop_count": noop_count,
        "repair_event_log": events,
        "repair_event_examples_complete": event_count <= _REPAIR_EVENT_EXAMPLE_LIMIT,
        "repair_event_example_limit": _REPAIR_EVENT_EXAMPLE_LIMIT,
        "repair_event_examples_sha256": _sha(events),
        "repair_event_log_sha256": "sha256:" + event_hasher.hexdigest(),
        "repair_cycle_ledger": cycle_ledger,
        "repair_cycle_ledger_sha256": _sha(cycle_ledger),
        "TRANSACTION_VALIDATION_COMPLETE_READ_CONFLICT_SET_RECEIPT": bool(
            transaction_validation_complete
        ),
        "UNION_PAYLOAD_ATOMIC_REVALIDATION_RECEIPT": bool(
            union_atomic_revalidation
        ),
        "REPAIR_ORDER_REPLAY_EXACT_RECEIPT": bool(order_replay_exact),
        "REPAIR_IDEMPOTENCE_REPLAY_EXACT_RECEIPT": bool(
            idempotence_replay_exact
        ),
        "REPAIR_TERMINAL_FIXED_POINT_RECEIPT": bool(
            terminal_mismatches and max(terminal_mismatches) <= 1.0e-15
        ),
    }
    return primitives, state.amplitudes, terminal_lift, repaired, events


def _observer_loop(
    config: Mapping[str, Any],
    federation: EchosahedralFederation,
    dynamics: Mapping[str, Any],
    source_state_root: str,
) -> dict[str, Any]:
    snapshots = [dict(row) for row in dynamics["record_state_snapshots"]]
    snapshot_lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for snapshot in snapshots:
        for carrier_row in snapshot["carrier_rows"]:
            snapshot_lookup[(int(snapshot["cycle"]), str(carrier_row["carrier_id"]))] = dict(
                carrier_row
            )

    def future_action_material(
        record: Mapping[str, Any], *, ablate_full_state: bool = False
    ) -> dict[str, Any]:
        """Derive a concrete next-port action from one committed C12 field.

        The ablation removes the readback values before recomputing the action;
        it does not merely change a hash-domain label.  This lets the observer
        audit compare the actual next port selected with and without readback.
        """

        recorded_state = [float(value) for value in record["full_port_state"]]
        full_state = [0.0] * 12 if ablate_full_state else recorded_state
        weighted_value = sum((index + 1) * value for index, value in enumerate(full_state))
        quantized_value = int(round(weighted_value * 1_000_000.0))
        next_port = (int(record["port"]) + 1 + abs(quantized_value) % 11) % 12
        return {
            "algorithm_id": "full-c12-record-conditioned-local-action-v2",
            "readback_ablation": ablate_full_state,
            "observer_token": str(record["observer_token"]),
            "carrier_id": str(record["carrier_id"]),
            "record_cycle": int(record["record_cycle"]),
            "next_port": next_port,
            "quantized_full_state_functional": quantized_value,
            "full_port_state_sha256": str(record["full_port_state_sha256"]),
            "source_state_root": str(record["source_state_root"]),
        }

    supports = tuple(
        sorted(federation.observer_supports, key=lambda item: item.observer_token)
    )
    stable_observer_index = {
        support.observer_token: index for index, support in enumerate(supports)
    }
    support_by_token = {support.observer_token: support for support in supports}
    seam_endpoints = {
        seam.seam_id: frozenset((seam.left_carrier_id, seam.right_carrier_id))
        for seam in federation.seams
    }
    visibility_contract = [
        {
            "observer_token": support.observer_token,
            "carrier_ids": sorted(support.carrier_ids),
            "visible_seam_ids": sorted(support.visible_seam_ids),
        }
        for support in supports
    ]
    seam_endpoint_contract = [
        {
            "seam_id": seam_id,
            "carrier_ids": sorted(endpoints),
        }
        for seam_id, endpoints in sorted(seam_endpoints.items())
    ]

    def emit_rounds(
        start_round: int,
        stop_round: int,
        continuation_state: Mapping[str, Any],
        *,
        phase_supports: tuple[ObserverSupport, ...] = supports,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Emit complete RECORD/READBACK/FEEDBACK source rounds.

        Observer enumeration within a phase is serialization metadata only.
        Every readback is evaluated against the complete same-round record
        map, so reversing that enumeration cannot alter any event material or
        read-after-write edge.
        """

        rows: list[dict[str, Any]] = []
        last_events = {
            str(key): str(value)
            for key, value in dict(
                continuation_state.get("last_event_by_observer", {})
            ).items()
        }
        last_record_carrier = {
            str(key): str(value)
            for key, value in dict(
                continuation_state.get("last_record_carrier_by_observer", {})
            ).items()
        }
        last_record_event = {
            str(key): str(value)
            for key, value in dict(
                continuation_state.get("last_record_event_by_observer", {})
            ).items()
        }
        last_record_state_hash = {
            str(key): str(value)
            for key, value in dict(
                continuation_state.get(
                    "last_record_state_sha256_by_observer", {}
                )
            ).items()
        }
        next_ports = {
            str(key): int(value)
            for key, value in dict(
                continuation_state.get("next_port_by_observer", {})
            ).items()
        }
        for sample in range(start_round, stop_round):
            snapshot = snapshots[sample % len(snapshots)]
            record_cycle = int(snapshot["cycle"])
            round_records: dict[str, dict[str, Any]] = {}

            # Phase 1: all records commit from the prior round's continuation.
            for support in phase_supports:
                token = support.observer_token
                carrier_ids = sorted(support.carrier_ids)
                carrier_id = carrier_ids[sample % len(carrier_ids)]
                applied_feedback_event_id = (
                    last_events.get(token) if token in next_ports else None
                )
                port = next_ports.get(
                    token,
                    (sample * 5 + stable_observer_index[token] * 3) % 12,
                )
                field_row = snapshot_lookup[(record_cycle, carrier_id)]
                full_state = [
                    float(value) for value in field_row["full_port_state"]
                ]
                full_state_hash = _sha(full_state)
                record_material = {
                    "kind": "RECORD_COMMIT",
                    "observer_token": token,
                    "carrier_id": carrier_id,
                    "port": port,
                    "sample": sample,
                    "record_cycle": record_cycle,
                    "port_value": _clean_float(full_state[port]),
                    "full_port_state": full_state,
                    "full_port_state_sha256": full_state_hash,
                    "source_state_root": source_state_root,
                    "applied_feedback_event_id": applied_feedback_event_id,
                    "parents": (
                        []
                        if applied_feedback_event_id is None
                        else [applied_feedback_event_id]
                    ),
                    # Canonical record presentation is metadata.  When it
                    # names the prior record, the semantic continuation edge
                    # still comes only from applied_feedback_event_id.
                    "record_order_previous_event_ids": (
                        []
                        if token not in last_record_event
                        else [last_record_event[token]]
                    ),
                }
                record = {**record_material, "event_id": _sha(record_material)}
                rows.append(record)
                round_records[token] = record

            round_readbacks: dict[str, dict[str, Any]] = {}
            # Phase 2: each observer sees the complete same-round record map.
            for support in phase_supports:
                token = support.observer_token
                record = round_records[token]
                carrier_id = str(record["carrier_id"])
                independently_loaded = snapshot_lookup[(record_cycle, carrier_id)]
                independently_loaded_state = [
                    float(value)
                    for value in independently_loaded["full_port_state"]
                ]
                recomputed_hash = _sha(independently_loaded_state)
                cross_parents: list[str] = []
                cross_read_state_hashes: list[str] = []
                overlap_witnesses: list[dict[str, Any]] = []
                if bool(config.get("observer_cross_reads", False)):
                    for other_token in sorted(round_records):
                        if other_token == token:
                            continue
                        other_record = round_records[other_token]
                        other_support = support_by_token[other_token]
                        other_carrier = str(other_record["carrier_id"])
                        shared_visible_seams = sorted(
                            seam_id
                            for seam_id in (
                                support.visible_seam_ids
                                & other_support.visible_seam_ids
                            )
                            if other_carrier in seam_endpoints[seam_id]
                        )
                        if (
                            other_carrier not in support.carrier_ids
                        ):
                            continue
                        cross_parents.append(str(other_record["event_id"]))
                        cross_read_state_hashes.append(
                            str(other_record["full_port_state_sha256"])
                        )
                        overlap_witnesses.append(
                            {
                                "record_event_id": str(other_record["event_id"]),
                                "record_observer_token": other_token,
                                "record_carrier_id": other_carrier,
                                "visibility_witness_kind": (
                                    "shared_declared_support_carrier"
                                ),
                                "shared_visible_seam_ids": shared_visible_seams,
                            }
                        )
                read_material = {
                    "kind": "READBACK",
                    "observer_token": token,
                    "sample": sample,
                    "record_event_id": record["event_id"],
                    "carrier_id": carrier_id,
                    "record_cycle": record_cycle,
                    "recomputed_full_port_state_sha256": recomputed_hash,
                    "record_signature_matches_source_field": bool(
                        recomputed_hash == record["full_port_state_sha256"]
                    ),
                    "cross_read_record_event_ids": cross_parents,
                    "cross_read_record_state_sha256s": cross_read_state_hashes,
                    "cross_read_overlap_witnesses": overlap_witnesses,
                    "parents": [record["event_id"], *cross_parents],
                }
                read = {**read_material, "event_id": _sha(read_material)}
                rows.append(read)
                round_readbacks[token] = read

            # Phase 3: local feedback consumes that observer's record/readback.
            for support in phase_supports:
                token = support.observer_token
                record = round_records[token]
                read = round_readbacks[token]
                port = int(record["port"])
                independently_loaded = snapshot_lookup[
                    (record_cycle, str(record["carrier_id"]))
                ]
                independently_loaded_state = [
                    float(value)
                    for value in independently_loaded["full_port_state"]
                ]
                recomputed_hash = _sha(independently_loaded_state)
                predicted_material = future_action_material(record)
                predicted_action = _sha(predicted_material)
                reconstructed_record = {
                    **record,
                    "port_value": _clean_float(independently_loaded_state[port]),
                    "full_port_state": independently_loaded_state,
                    "full_port_state_sha256": recomputed_hash,
                }
                observed_material = future_action_material(reconstructed_record)
                observed_action = _sha(observed_material)
                ablated_material = future_action_material(
                    reconstructed_record, ablate_full_state=True
                )
                ablated_action = _sha(ablated_material)
                feedback_material = {
                    "kind": "LOCAL_FEEDBACK",
                    "observer_token": token,
                    "sample": sample,
                    "readback_event_id": read["event_id"],
                    "parents": [read["event_id"], record["event_id"]],
                    "action_input_record_event_id": record["event_id"],
                    "predicted_action_material_sha256": _sha(predicted_material),
                    "observed_recomputation_material_sha256": _sha(
                        observed_material
                    ),
                    "observed_action_recomputed_from_record": True,
                    "observed_action_recomputed_from_source_field": True,
                    "predicted_action_material_next_port": int(
                        predicted_material["next_port"]
                    ),
                    "observed_action_material_next_port": int(
                        observed_material["next_port"]
                    ),
                    "ablated_action_material_next_port": int(
                        ablated_material["next_port"]
                    ),
                    "predicted_action": predicted_action,
                    "observed_action": observed_action,
                    "ablated_action": ablated_action,
                }
                feedback = {
                    **feedback_material,
                    "event_id": _sha(feedback_material),
                }
                if config["feedback_enabled"]:
                    rows.append(feedback)
                    last_events[token] = feedback["event_id"]
                    next_ports[token] = int(observed_material["next_port"])

            for token, record in round_records.items():
                last_record_carrier[token] = str(record["carrier_id"])
                last_record_event[token] = str(record["event_id"])
                last_record_state_hash[token] = str(
                    record["full_port_state_sha256"]
                )
        return rows, {
            "last_event_by_observer": last_events,
            "last_record_carrier_by_observer": last_record_carrier,
            "last_record_event_by_observer": last_record_event,
            "last_record_state_sha256_by_observer": last_record_state_hash,
            "next_port_by_observer": next_ports,
        }

    round_count = int(config["observer_samples"])
    full_replay, _ = emit_rounds(0, round_count, {})
    cut_rounds = min(int(config["checkpoint_interval"]), round_count - 1)
    prefix, continuation_state = emit_rounds(0, cut_rounds, {})
    continuation, _ = emit_rounds(
        cut_rounds, round_count, continuation_state
    )
    events = prefix + continuation
    permuted_replay, _ = emit_rounds(
        0, round_count, {}, phase_supports=tuple(reversed(supports))
    )

    def presentation_projection(
        event_rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        semantic, generated, _, _ = _semantic_source_events(
            {
                "events": event_rows,
                "observer_support_visibility_contract": visibility_contract,
                "overlap_seam_endpoint_contract": seam_endpoint_contract,
                "cross_read_visibility_rule": (
                    "same_round_shared_declared_support_carrier_v1"
                ),
            }
        )
        semantic_projection = sorted(
            (
                {
                    key: value
                    for key, value in row.items()
                    if key != "source_sequence_index"
                }
                for row in semantic
            ),
            key=lambda row: str(row["event_key"]),
        )
        edge_projection = sorted(
            (
                {
                    "parent_event_id": row["parent_event_id"],
                    "child_event_id": row["child_event_id"],
                    "shared_resource_ids": row["shared_resource_ids"],
                }
                for row in generated
            ),
            key=lambda row: (row["parent_event_id"], row["child_event_id"]),
        )
        return semantic_projection, edge_projection

    canonical_semantic_projection, canonical_edge_projection = (
        presentation_projection(events)
    )
    permuted_semantic_projection, permuted_edge_projection = (
        presentation_projection(permuted_replay)
    )
    phase_permutation_control = {
        "algorithm_id": "reverse_observer_order_within_each_source_round_phase_v1",
        "source_round_count": round_count,
        "transport_event_material_set_invariant": bool(
            sorted(events, key=lambda row: str(row["event_id"]))
            == sorted(permuted_replay, key=lambda row: str(row["event_id"]))
        ),
        "canonical_semantic_event_set_sha256": _sha(
            canonical_semantic_projection
        ),
        "permuted_semantic_event_set_sha256": _sha(
            permuted_semantic_projection
        ),
        "semantic_event_set_invariant": bool(
            canonical_semantic_projection == permuted_semantic_projection
        ),
        "canonical_source_order_sha256": _sha(canonical_edge_projection),
        "permuted_source_order_sha256": _sha(permuted_edge_projection),
        "source_order_invariant": bool(
            canonical_edge_projection == permuted_edge_projection
        ),
    }
    record_rows = [row for row in events if row["kind"] == "RECORD_COMMIT"]
    feedback_rows = [row for row in events if row["kind"] == "LOCAL_FEEDBACK"]
    direct = sum(
        row["predicted_action_material_next_port"]
        == row["observed_action_material_next_port"]
        for row in feedback_rows
    )
    control_order = sorted(
        range(len(record_rows)),
        key=lambda index: _sha(
            {
                "algorithm_id": config["prediction_control"],
                "semantic_record_id": record_rows[index]["event_id"],
            }
        ),
    )
    donor_by_index = {
        control_order[index]: control_order[(index + 1) % len(control_order)]
        for index in range(len(control_order))
    }
    shuffled_predictions = [
        int(future_action_material(record_rows[donor_by_index[index]])["next_port"])
        for index in range(len(record_rows))
    ]
    shuffled = sum(
        predicted == row["observed_action_material_next_port"]
        for predicted, row in zip(shuffled_predictions, feedback_rows, strict=True)
    )
    checkpoint_material = {
        "requested_checkpoint_interval": config["checkpoint_interval"],
        "checkpoint_interval_unit": "complete_source_rounds",
        "cut_round_count": cut_rounds,
        "next_round_index": cut_rounds,
        "cut_event_id": prefix[-1]["event_id"],
        "prefix_root": _sha(prefix),
        "saved_continuation_state": continuation_state,
        "suffix_root": _sha(continuation),
        "continuation_event_count": len(continuation),
    }
    checkpoint = {**checkpoint_material, "checkpoint_id": _sha(checkpoint_material)}
    feedback_changed = sum(
        row["observed_action_material_next_port"]
        != row["ablated_action_material_next_port"]
        for row in feedback_rows
    )
    return {
        "events": events,
        "event_log_sha256": _sha(events),
        "event_carrier_scope": (
            "observer_instrumentation_history_over_source_state_snapshots"
        ),
        "underlying_repair_transactions_promoted_as_events": False,
        "observer_support_visibility_contract": visibility_contract,
        "overlap_seam_endpoint_contract": seam_endpoint_contract,
        "cross_read_visibility_rule": (
            "same_round_shared_declared_support_carrier_v1"
        ),
        "source_round_count": round_count,
        "phase_order": ["RECORD_COMMIT", "READBACK", "LOCAL_FEEDBACK"],
        "phase_observer_permutation_control": phase_permutation_control,
        "checkpoint": checkpoint,
        "checkpoint_replay_exact": events == full_replay,
        "record_count": len(record_rows),
        "readback_count": sum(row["kind"] == "READBACK" for row in events),
        "feedback_count": len(feedback_rows),
        "direct_prediction_match_count": direct,
        "shuffled_prediction_match_count": shuffled,
        "feedback_changed_action_count": feedback_changed,
        "feedback_action_applied_to_next_record_count": sum(
            int(records[index + 1]["port"])
            == int(feedback[index]["observed_action_material_next_port"])
            for records, feedback in (
                (
                    [
                        row
                        for row in record_rows
                        if row["observer_token"] == observer_token
                    ],
                    [
                        row
                        for row in feedback_rows
                        if row["observer_token"] == observer_token
                    ],
                )
                for observer_token in sorted(
                    {row["observer_token"] for row in record_rows}
                )
            )
            for index in range(len(records) - 1)
        ),
        "observed_action_recomputation_count": sum(
            row["observed_action_recomputed_from_record"] for row in feedback_rows
        ),
        "source_field_recomputation_count": sum(
            row["observed_action_recomputed_from_source_field"]
            for row in feedback_rows
        ),
        "full_port_record_signature_count": sum(
            len(row["full_port_state"]) == 12
            and row["full_port_state_sha256"] == _sha(row["full_port_state"])
            for row in record_rows
        ),
        "readback_source_signature_match_count": sum(
            row["record_signature_matches_source_field"]
            for row in events
            if row["kind"] == "READBACK"
        ),
        "prediction_control": config["prediction_control"],
        "feedback_enabled": config["feedback_enabled"],
    }


def _m4_cap_state(terminal_complex_lift: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """Build a deterministic matrix diagnostic without promoting it physically.

    The first sixteen presentation-ordered amplitudes are compressed into an
    Hermitian matrix and exponentiated.  This produces a faithful density
    matrix on an abstract M4 algebra, but it is neither an A5-quotient-invariant
    selection rule nor a derivation of a physical cap or maximum-entropy state.
    """

    flat = np.asarray(terminal_complex_lift, dtype=np.complex128).reshape(-1)
    source_values = np.resize(flat, 16).reshape(4, 4)
    hermitian = 0.5 * (source_values + source_values.conj().T)
    scale = max(float(np.linalg.norm(hermitian, ord=2)), 1.0e-12)
    hamiltonian = hermitian / scale
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    gibbs_values = np.exp(-eigenvalues)
    rho = (eigenvectors * gibbs_values) @ eigenvectors.conj().T
    rho /= np.trace(rho)
    rho_eigenvalues, rho_eigenvectors = np.linalg.eigh(rho)
    generator = (rho_eigenvectors * (-np.log(rho_eigenvalues))) @ rho_eigenvectors.conj().T
    z = np.diag(np.arange(4, dtype=float)).astype(np.complex128)
    x = np.roll(np.eye(4, dtype=np.complex128), 1, axis=1)
    commutator_norm = float(np.linalg.norm(z @ x - x @ z, ord="fro"))
    left = np.kron(z, np.eye(4))
    right = np.kron(np.eye(4), x.T)
    gns_commutator = float(np.linalg.norm(left @ right - right @ left, ord="fro"))
    rho_rows = _matrix_pairs(rho)
    generator_rows = _matrix_pairs(generator)
    raw = {
        "source_matrix": _matrix_pairs(source_values),
        "source_hermitian_constraint": _matrix_pairs(hamiltonian),
        "gibbs_parameter": 1,
        "rho": rho_rows,
        "modular_generator": generator_rows,
        "m4_generator_z": _matrix_pairs(z),
        "m4_generator_x": _matrix_pairs(x),
    }
    report = {
        "state_mode": _CONSTRUCTED_M4_STATE_MODE,
        "algebra_scope": "abstract_M4_matrix_diagnostic",
        "state_construction": (
            "first_16_presentation_ordered_amplitudes_hermitian_gibbs_map"
        ),
        "state_status": "CONSTRUCTED_DIAGNOSTIC_ONLY",
        "PHYSICAL_PRIME_GEOMETRIC_CAP_STATE_RECEIPT": False,
        "MAXIMUM_ENTROPY_STATE_DERIVATION_RECEIPT": False,
        "source_selection_rule": "first_16_flattened_terminal_complex_amplitudes",
        "SOURCE_SELECTION_A5_QUOTIENT_INVARIANCE_RECEIPT": False,
        "noncommutative_algebra": True,
        "noncommutative_algebra_scope": "abstract_M4_matrix_algebra_only",
        "commutator_norm": commutator_norm,
        "source_primitive_fields": [
            "terminal_complex_lift",
            "presentation_ordered_first_16_amplitudes",
        ],
        "surrogate_inputs": [
            "presentation_ordered_first_16_terminal_complex_amplitudes"
        ],
        "rho": {
            "dimension": 4,
            "trace": float(np.trace(rho).real),
            "minimum_eigenvalue": float(np.min(rho_eigenvalues)),
            "hermiticity_residual": float(np.linalg.norm(rho - rho.conj().T)),
            "matrix_hash": _sha(rho_rows),
        },
        "modular_generator": {
            "construction": "negative_log_density_matrix",
            "dimension": 4,
            "functional_calculus_residual": float(
                np.linalg.norm(
                    (rho_eigenvectors * (-np.log(rho_eigenvalues)))
                    @ rho_eigenvectors.conj().T
                    - generator
                )
            ),
            "noncentrality_norm": float(
                np.linalg.norm(generator - np.trace(generator) * np.eye(4) / 4)
            ),
            "matrix_hash": _sha(generator_rows),
        },
        "mixed_gns": {
            "constructed": True,
            "left_right_representation": gns_commutator <= 1.0e-12,
            "cyclic_separating_support": float(np.min(rho_eigenvalues)) > 1.0e-12,
            "left_right_commutator_residual": gns_commutator,
        },
        "raw_primitives": raw,
        "raw_primitives_sha256": _sha(raw),
    }
    return report, rho


def _refinement_report(level: int, rho: np.ndarray) -> dict[str, Any]:
    """Report the geometric tower and quarantine the repeated-rho diagnostic."""

    tower = build_geodesic_icosahedral_tower(level)
    symmetry = icosahedral_a5_equivariance_report(level)
    levels: list[dict[str, Any]] = []
    for index, mesh in enumerate(tower.levels):
        row: dict[str, Any] = {
            "level_id": f"support-r{index}",
            "patch_count": mesh.face_count,
            "geometry_hash": "sha256:" + mesh.geometry_hash,
        }
        if index:
            mapping = tower.cell_refinements[index - 1]
            row["parent_level_id"] = f"support-r{index - 1}"
            row["lineage_hash"] = "sha256:" + mapping.map_hash
        levels.append(row)
    expectations: list[dict[str, Any]] = []
    for index, mapping in enumerate(tower.cell_refinements, start=1):
        embedded = np.repeat(rho[None, :, :], 4, axis=0)
        weights = mapping.conditional_expectation_weights[:4]
        recovered = np.tensordot(weights, embedded, axes=(0, 0))
        positive = bool(np.linalg.eigvalsh(recovered).min() >= -1.0e-12)
        expectations.append(
            {
                "fine_level_id": f"support-r{index}",
                "coarse_level_id": f"support-r{index - 1}",
                "operator_hash": "sha256:" + mapping.map_hash,
                "unital": bool(mapping.normalization_residual <= 5.0e-13),
                "positive": positive,
                # Repeating the same constructed rho on all four child fibers
                # makes recovery tautological.  It does not certify an
                # independently specified fine state or a physical cap tower.
                "state_preserving": False,
                "cap_isotony_compatible": False,
                "noncommutative_prime_cap_expectation": False,
                "fiber_algebra": "constructed_repeated_M4_matrix_diagnostic",
                "left_inverse_residual": float(np.linalg.norm(recovered - rho)),
            }
        )
    structural_receipt = bool(
        tower.receipt()["GEODESIC_ICOSAHEDRAL_TOWER_RECEIPT"]
        and symmetry["A5_EQUIVARIANT_REFINEMENT_RECEIPT"]
    )
    repeated_state_identity_receipt = bool(
        expectations
        and all(
            row["unital"]
            and row["positive"]
            and row["left_inverse_residual"] <= 1.0e-11
            for row in expectations
        )
    )
    certificate_blockers = [
        "fine_level_M4_states_not_independently_instantiated",
        "same_constructed_rho_repeated_on_every_child_fiber",
        "cap_isotony_not_instantiated",
        "physical_carrier_to_support_realization_not_established",
    ]
    certificate_material = {
        "levels": levels,
        "maps": expectations,
        "structural_receipt": structural_receipt,
        "repeated_state_identity_receipt": repeated_state_identity_receipt,
        "paper_certificate": False,
        "certificate_blockers": certificate_blockers,
    }
    return {
        "mesh_family": "nested_geodesic_icosahedral",
        "nested_lineage_receipt": structural_receipt,
        "conditional_expectations_receipt": False,
        "TRUE_ICOSAHEDRAL_REFINEMENT_TOWER_RECEIPT": structural_receipt,
        "A5_EQUIVARIANT_REFINEMENT_RECEIPT": structural_receipt,
        "COMMUTATIVE_CELL_REFINEMENT_DIAGNOSTIC_RECEIPT": structural_receipt,
        "CONSTRUCTED_REPEATED_RHO_IDENTITY_RECEIPT": (
            repeated_state_identity_receipt
        ),
        "PAPER_MULTIRESOLUTION_REGULATOR_CERTIFICATE": False,
        "conditional_expectation_status": (
            "NOT_ESTABLISHED_REPEATED_CONSTRUCTED_RHO_ONLY"
        ),
        "paper_multiresolution_regulator_status": "NOT_EVALUATED",
        "certificate_blockers": certificate_blockers,
        "levels": levels,
        "conditional_expectations": expectations,
        "certificate_scope": (
            "commutative_cell_geometry_plus_repeated_constructed_M4_diagnostic"
        ),
        "certificate_sha256": _sha(certificate_material),
    }


def _independent_geometry(
    config: Mapping[str, Any],
    initial_intensities: np.ndarray, sample_count: int
) -> dict[str, Any]:
    """Build a target-blind, same-source geometry diagnostic.

    The legacy function/report slot says ``independent_geometry`` for consumer
    compatibility.  The producer is not independent: its frames are defined
    deterministically from the pre-intervention initial source array.  Using
    only the initial state prevents the downstream repair response from being
    copied into a candidate feature vector.
    """

    flat = np.asarray(initial_intensities, dtype=float).reshape(-1)
    rows: list[dict[str, Any]] = []
    for index in range(2 * sample_count):
        offsets = np.asarray(
            [
                float(position)
                + 0.125 * float(flat[(index * 7 + position * 11) % flat.size])
                for position in range(4)
            ]
        )
        offsets.sort()
        if str(config.get("geometry_transport", "legacy")) == "held_out_flow":
            # Held-out flow transport: the parameter is declared first, the
            # dilation flow x -> exp(t) * x transports the third frame point,
            # and the cross ratio is measured from the transported frame with
            # the far reference point.  The parameter and ratio are thereby
            # independently produced; source intensities seed the base point.
            value = 0.25 + 0.125 * float(index) + 0.03125 * float(
                flat[index % flat.size]
            )
            base_x = 1.0 + 0.25 * abs(float(flat[(index * 3) % flat.size]))
            a = 0.0
            b = 2.0 * base_x
            c = base_x * math.exp(value)
            d = 1.0e15
            cross_ratio = ((c - a) * (d - b)) / ((b - a) * (d - c))
            offsets = np.asarray([a, b, c, d])
        else:
            a, b, c, d = offsets
            cross_ratio = ((c - a) * (d - b)) / ((b - a) * (d - c))
            value = math.log(cross_ratio)
        partition = "source-frame" if index < sample_count else "heldout-flow"
        material = {
            "row_id": f"{partition}-{index % sample_count:04d}",
            "ordered_frame": [_clean_float(item) for item in offsets],
            "orientation": "ascending",
            "cross_ratio": _clean_float(cross_ratio),
            "geometric_parameter": _clean_float(value),
        }
        rows.append({**material, "row_sha256": _sha(material)})
    source_rows = rows[:sample_count]
    heldout_rows = rows[sample_count:]
    return {
        "derivation_method": "ordered_bw_frame_cross_ratio",
        "source_primitive_fields": [
            "initial_port_intensities",
            "deterministic_index_schedule",
        ],
        "derivation_expression": (
            "log(cross_ratio(sort(index_offsets+0.125*source_intensity_samples)))"
        ),
        "derivation_scope": "same_source_self_derived_diagnostic",
        "source_phase": "pre_intervention_initial_state",
        "forbidden_token_hits": [],
        "target_blind_derivation": True,
        "independent_of_modular_fit": True,
        "independent_of_kms_target": True,
        "orientation_fixed_from_source": False,
        "orientation_status": "CONSTRUCTED_BY_ASCENDING_SORT",
        "INDEPENDENT_GEOMETRY_PRODUCER_RECEIPT": False,
        "independent_geometry_producer_status": (
            "NOT_INDEPENDENT_SAME_SOURCE_CAPTURE"
        ),
        "geometric_parameter_values": [row["geometric_parameter"] for row in source_rows],
        "geometry_source_row_ids": [row["row_id"] for row in source_rows],
        "kms_score_row_ids": [],
        "heldout_control_row_ids": [row["row_id"] for row in heldout_rows],
        "geometry_derivation_hash": _sha(rows),
        "raw_primitive_rows": rows,
    }


def _port_intensities_from_complex_rows(
    rows: list[list[list[float]]],
) -> list[list[float]]:
    return [
        [
            _clean_float(float(real) * float(real) + float(imag) * float(imag))
            for real, imag in carrier
        ]
        for carrier in rows
    ]


def _semantic_source_events(
    observer_log: Mapping[str, Any],
    *,
    validate_transport_ids: bool = True,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    """Project a transport log into semantic events and source-derived edges.

    The transport ``event_id``, archival sequence, declared ``parents``, and
    ``record_order_previous_event_ids`` are deliberately excluded from the
    semantic identity.  Explicit read fields are translated from transport
    IDs to the semantic IDs of the records they name.  Resource provenance
    then generates the consumed ancestry relation; declared transport parents
    are returned separately only for the byte-identity audit.

    The identity remains carrier/port-presentation-bound and is not claimed to
    be an A5 quotient identity.  It is, however, invariant under changes to
    executor ancestry, checkpoint placement, and record-order metadata.
    """

    raw_events = [dict(row) for row in observer_log["events"]]
    support_visibility: dict[str, dict[str, set[str]]] = {}
    for row in observer_log.get("observer_support_visibility_contract", []):
        token = str(row["observer_token"])
        if token in support_visibility:
            raise RuntimeError(f"duplicate observer visibility contract: {token}")
        support_visibility[token] = {
            "carrier_ids": {str(value) for value in row["carrier_ids"]},
            "visible_seam_ids": {
                str(value) for value in row["visible_seam_ids"]
            },
        }
    seam_endpoints: dict[str, set[str]] = {}
    for row in observer_log.get("overlap_seam_endpoint_contract", []):
        seam_id = str(row["seam_id"])
        endpoints = {str(value) for value in row["carrier_ids"]}
        if seam_id in seam_endpoints or len(endpoints) != 2:
            raise RuntimeError(f"malformed overlap seam contract: {seam_id}")
        seam_endpoints[seam_id] = endpoints
    if not support_visibility or not seam_endpoints:
        raise RuntimeError("observer overlap-visibility contract is absent")
    if observer_log.get("cross_read_visibility_rule") != (
        "same_round_shared_declared_support_carrier_v1"
    ):
        raise RuntimeError("unknown cross-read visibility rule")
    raw_to_semantic: dict[str, str] = {}
    processed_raw: dict[str, dict[str, Any]] = {}
    semantic_events: list[dict[str, Any]] = []
    resources: dict[str, dict[str, set[str]]] = {}
    distinguished_roots: set[str] = set()

    def referenced_semantic(raw_id: object, field: str) -> str:
        key = str(raw_id)
        if key not in raw_to_semantic:
            raise RuntimeError(
                f"{field} names an absent or non-prior transport event: {key}"
            )
        return raw_to_semantic[key]

    for index, row in enumerate(raw_events):
        raw_event_id = str(row["event_id"])
        if raw_event_id in raw_to_semantic:
            raise RuntimeError(f"duplicate transport event id: {raw_event_id}")
        if validate_transport_ids:
            material = {key: value for key, value in row.items() if key != "event_id"}
            if raw_event_id != _sha(material):
                raise RuntimeError(
                    f"transport event id does not bind event material: {raw_event_id}"
                )

        reads: set[str] = set()
        footprint: list[str] = []
        payload: dict[str, Any] = {"event_kind": str(row["kind"])}
        if row["kind"] == "RECORD_COMMIT":
            carrier_id = str(row["carrier_id"])
            port = int(row["port"])
            full_state = [float(value) for value in row["full_port_state"]]
            if len(full_state) != 12 or _sha(full_state) != str(
                row["full_port_state_sha256"]
            ):
                raise RuntimeError("record full-state commitment is invalid")
            observer_contract = support_visibility.get(str(row["observer_token"]))
            if (
                observer_contract is None
                or carrier_id not in observer_contract["carrier_ids"]
            ):
                raise RuntimeError("record carrier is outside observer support")
            footprint = [f"{carrier_id}:port-{item:02d}" for item in range(12)]
            source_reads = {
                f"source-state:{row['source_state_root']}:{carrier_id}:port-{item:02d}"
                for item in range(12)
            }
            reads.update(source_reads)
            distinguished_roots.update(source_reads)
            applied_feedback_raw = row.get("applied_feedback_event_id")
            applied_feedback_id: str | None = None
            if applied_feedback_raw is not None:
                applied_feedback_id = referenced_semantic(
                    applied_feedback_raw, "applied_feedback_event_id"
                )
                prior_feedback = processed_raw[str(applied_feedback_raw)]
                if prior_feedback.get("kind") != "LOCAL_FEEDBACK":
                    raise RuntimeError(
                        "applied_feedback_event_id does not name local feedback"
                    )
                if prior_feedback.get("observer_token") != row.get(
                    "observer_token"
                ):
                    raise RuntimeError(
                        "record consumes feedback from a different observer"
                    )
                if int(prior_feedback["observed_action_material_next_port"]) != port:
                    raise RuntimeError(
                        "record port does not equal the consumed feedback action"
                    )
                reads.add(f"local-action:{applied_feedback_id}")
            payload.update(
                {
                    "carrier_id": carrier_id,
                    "port": port,
                    "sample": int(row["sample"]),
                    "record_cycle": int(row["record_cycle"]),
                    "port_value": float(row["port_value"]),
                    "full_port_state_sha256": str(
                        row["full_port_state_sha256"]
                    ),
                    "source_state_root": str(row["source_state_root"]),
                    "applied_feedback_event_id": applied_feedback_id,
                }
            )
        elif row["kind"] == "READBACK":
            raw_record_id = str(row["record_event_id"])
            record_id = referenced_semantic(
                raw_record_id, "record_event_id"
            )
            source_record = processed_raw[raw_record_id]
            if source_record.get("kind") != "RECORD_COMMIT":
                raise RuntimeError("record_event_id does not name a record commit")
            committed_hash = str(source_record["full_port_state_sha256"])
            recomputed_hash = str(row["recomputed_full_port_state_sha256"])
            signature_matches = recomputed_hash == committed_hash
            if signature_matches != bool(
                row["record_signature_matches_source_field"]
            ):
                raise RuntimeError("readback signature verdict is inconsistent")
            if not signature_matches:
                raise RuntimeError("readback does not authenticate its named record")
            sample = int(row["sample"])
            if int(source_record["sample"]) != sample:
                raise RuntimeError("readback does not name its same-round record")
            raw_cross_ids = [
                str(value)
                for value in row.get("cross_read_record_event_ids", [])
            ]
            cross_hashes = [
                str(value)
                for value in row.get("cross_read_record_state_sha256s", [])
            ]
            if len(raw_cross_ids) != len(cross_hashes):
                raise RuntimeError(
                    "cross-read record IDs and committed-state hashes differ in length"
                )
            overlap_witnesses = [
                dict(value)
                for value in row.get("cross_read_overlap_witnesses", [])
            ]
            if len(overlap_witnesses) != len(raw_cross_ids):
                raise RuntimeError(
                    "cross-read version and visibility witness lengths differ"
                )
            reader_token = str(row["observer_token"])
            reader_contract = support_visibility.get(reader_token)
            if reader_contract is None:
                raise RuntimeError("readback observer has no visibility contract")
            cross_versions: list[dict[str, Any]] = []
            for raw_cross_id, state_hash, witness in zip(
                raw_cross_ids, cross_hashes, overlap_witnesses, strict=True
            ):
                cross_id = referenced_semantic(
                    raw_cross_id, "cross_read_record_event_ids"
                )
                cross_record = processed_raw[raw_cross_id]
                if cross_record.get("kind") != "RECORD_COMMIT":
                    raise RuntimeError("cross-read ID does not name a record commit")
                if str(cross_record["full_port_state_sha256"]) != state_hash:
                    raise RuntimeError(
                        "cross-read state hash does not match the named record"
                    )
                if int(cross_record["sample"]) != sample:
                    raise RuntimeError("cross-read record is not from the same round")
                cross_token = str(cross_record["observer_token"])
                cross_carrier = str(cross_record["carrier_id"])
                cross_contract = support_visibility.get(cross_token)
                witness_seams = [
                    str(value) for value in witness["shared_visible_seam_ids"]
                ]
                if (
                    str(witness.get("record_event_id")) != raw_cross_id
                    or str(witness.get("record_observer_token")) != cross_token
                    or str(witness.get("record_carrier_id")) != cross_carrier
                    or cross_contract is None
                    or cross_carrier not in reader_contract["carrier_ids"]
                    or cross_carrier not in cross_contract["carrier_ids"]
                    or witness.get("visibility_witness_kind")
                    != "shared_declared_support_carrier"
                    or witness_seams != sorted(set(witness_seams))
                ):
                    raise RuntimeError("cross-read overlap witness is malformed")
                for seam_id in witness_seams:
                    if (
                        seam_id not in reader_contract["visible_seam_ids"]
                        or seam_id not in cross_contract["visible_seam_ids"]
                        or cross_carrier not in seam_endpoints.get(seam_id, set())
                    ):
                        raise RuntimeError(
                            "cross-read is not witnessed by declared overlap visibility"
                        )
                cross_versions.append(
                    {
                        "record_event_id": cross_id,
                        "committed_full_port_state_sha256": state_hash,
                        "record_observer_token": cross_token,
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
        elif row["kind"] == "LOCAL_FEEDBACK":
            readback_id = referenced_semantic(
                row["readback_event_id"], "readback_event_id"
            )
            input_record_id = referenced_semantic(
                row["action_input_record_event_id"],
                "action_input_record_event_id",
            )
            reads.update(
                {f"readback:{readback_id}", f"record:{input_record_id}"}
            )
            sample = int(row["sample"])
            raw_readback = processed_raw[str(row["readback_event_id"])]
            raw_record = processed_raw[str(row["action_input_record_event_id"])]
            if int(raw_readback["sample"]) != sample or int(
                raw_record["sample"]
            ) != sample:
                raise RuntimeError("feedback inputs do not belong to its source round")
            payload.update(
                {
                    "sample": sample,
                    "readback_event_id": readback_id,
                    "action_input_record_event_id": input_record_id,
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
            raise RuntimeError(f"unknown observer event kind: {row['kind']!r}")

        identity_material = {
            "schema": "oph.semantic-source-event-id.v1",
            "canonical_semantic_payload": payload,
            "observer_token": str(row["observer_token"]),
            "visible_footprint": sorted(footprint),
            "read_resource_ids": sorted(reads),
        }
        semantic_id = _sha(identity_material)
        if semantic_id in resources:
            raise RuntimeError(f"duplicate semantic event id: {semantic_id}")
        kind = str(row["kind"])
        resource_kind = {
            "RECORD_COMMIT": "record",
            "READBACK": "readback",
            "LOCAL_FEEDBACK": "local-action",
        }[kind]
        writes = {f"{resource_kind}:{semantic_id}"}
        raw_to_semantic[raw_event_id] = semantic_id
        resources[semantic_id] = {"reads": reads, "writes": writes}
        semantic_events.append(
            {
                "event_key": semantic_id,
                "canonical_semantic_payload": payload,
                "observer_token": str(row["observer_token"]),
                "visible_footprint": footprint,
                "parent_event_ids": [],
                "read_resource_ids": sorted(reads),
                "write_resource_ids": sorted(writes),
                "source_sequence_index": index,
            }
        )
        processed_raw[raw_event_id] = row

    writer_of: dict[str, str] = {}
    for event in semantic_events:
        for resource in event["write_resource_ids"]:
            if resource in writer_of:
                raise RuntimeError(f"semantic resource has two writers: {resource}")
            writer_of[resource] = str(event["event_key"])

    generated_parent_resources: dict[tuple[str, str], list[str]] = {}
    for event in semantic_events:
        child = str(event["event_key"])
        for resource in event["read_resource_ids"]:
            parent = writer_of.get(resource)
            if parent is None:
                if resource not in distinguished_roots:
                    raise RuntimeError(
                        f"semantic read has no writer or distinguished root: {resource}"
                    )
                continue
            if parent == child:
                raise RuntimeError(f"semantic event reads its own write: {child}")
            generated_parent_resources.setdefault((parent, child), []).append(
                resource
            )

    sequence_of = {
        str(event["event_key"]): int(event["source_sequence_index"])
        for event in semantic_events
    }
    by_semantic = {str(event["event_key"]): event for event in semantic_events}
    generated_ancestry: list[dict[str, Any]] = []
    for (parent, child), shared in sorted(generated_parent_resources.items()):
        by_semantic[child]["parent_event_ids"].append(parent)
        material = {
            "parent_event_id": parent,
            "child_event_id": child,
            "observer_token": by_semantic[child]["observer_token"],
            "parent_sequence_index": sequence_of[parent],
            "child_sequence_index": sequence_of[child],
            "shared_resource_ids": sorted(shared),
        }
        generated_ancestry.append({**material, "edge_id": _sha(material)})
    for event in semantic_events:
        event["parent_event_ids"].sort()

    declared_ancestry: list[dict[str, Any]] = []
    for raw_child in raw_events:
        child = raw_to_semantic[str(raw_child["event_id"])]
        for raw_parent in raw_child.get("parents", []):
            parent = raw_to_semantic.get(str(raw_parent))
            if parent is None:
                raise RuntimeError(
                    f"declared parent names an absent transport event: {raw_parent}"
                )
            shared = sorted(
                resources[parent]["writes"].intersection(
                    resources[child]["reads"]
                )
            )
            material = {
                "parent_event_id": parent,
                "child_event_id": child,
                "observer_token": by_semantic[child]["observer_token"],
                "parent_sequence_index": sequence_of[parent],
                "child_sequence_index": sequence_of[child],
                "shared_resource_ids": shared,
            }
            declared_ancestry.append({**material, "edge_id": _sha(material)})
    generated_ancestry.sort(
        key=lambda row: (row["parent_event_id"], row["child_event_id"])
    )
    declared_ancestry.sort(
        key=lambda row: (row["parent_event_id"], row["child_event_id"])
    )
    return (
        semantic_events,
        generated_ancestry,
        declared_ancestry,
        sorted(distinguished_roots),
    )


def _raw_overlap_relations(
    federation: EchosahedralFederation,
) -> list[dict[str, Any]]:
    visible_to_observers: dict[str, list[str]] = {}
    for support in federation.observer_supports:
        for seam_id in support.visible_seam_ids:
            visible_to_observers.setdefault(seam_id, []).append(
                support.observer_token
            )
    rows: list[dict[str, Any]] = []
    for seam in federation.seams:
        material = {
            "overlap_id": seam.seam_id,
            "left_carrier_id": seam.left_carrier_id,
            "right_carrier_id": seam.right_carrier_id,
            "left_ports": list(seam.left_ports),
            "right_ports": list(seam.right_ports),
            "left_to_right_ports": list(seam.left_to_right_ports),
            "right_to_left_ports": list(seam.right_to_left_ports),
            "orientation_signs": list(seam.left_to_right_orientation),
            "visible_to_observer_tokens": sorted(
                visible_to_observers.get(seam.seam_id, [])
            ),
            "interface_algebra_sha256": (
                seam.interface_algebra.interface_algebra_sha256
            ),
        }
        rows.append({**material, "row_sha256": _sha(material)})
    return rows


def _unavailable_clock_pair_input() -> dict[str, Any]:
    """Declare the independent two-producer interface without inventing data.

    The built-in finite source has no independently derived modular transport
    time or geometric-flow observable.  Keeping the two row families separate
    makes the future producer path schema-reachable while ensuring that the
    legacy intensity response cannot silently stand in for either observable.
    """

    return {
        "schema": TYPED_CLOCK_PAIR_INPUT_SCHEMA,
        "contract": {
            "schema": TYPED_CLOCK_PAIR_CONTRACT_SCHEMA,
            "status": "UNAVAILABLE",
            "join_key_fields": list(_CLOCK_PAIR_JOIN_KEY_FIELDS),
            "group_key_fields": list(_CLOCK_PAIR_GROUP_KEY_FIELDS),
            "minimum_refinement_level_count": 2,
            "modular_transport_producer_id": None,
            "modular_transport_producer_code_sha256": None,
            "modular_transport_source_field_sha256": None,
            "geometric_flow_producer_id": None,
            "geometric_flow_producer_code_sha256": None,
            "geometric_flow_source_field_sha256": None,
            "source_fixed_oriented_frame_incidence_required": True,
            "scoring_constants_absent": True,
            "unavailable_reason": (
                "independent_modular_transport_and_geometric_flow_producers_"
                "not_instantiated"
            ),
        },
        "modular_transport_rows": [],
        "geometric_flow_rows": [],
    }


def _build_postrun_capture(
    config: Mapping[str, Any],
    federation: EchosahedralFederation,
    dynamics: Mapping[str, Any],
    propagated_amplitudes: np.ndarray,
    repaired: np.ndarray,
    observer_log: Mapping[str, Any],
    geometry: Mapping[str, Any],
    refinement: Mapping[str, Any],
) -> dict[str, Any]:
    initial = _port_intensities_from_complex_rows(
        dynamics["initial_port_amplitudes"]
    )
    propagated = _real_rows(np.abs(propagated_amplitudes) ** 2)
    repaired_rows = _real_rows(repaired)
    initial_phases = [float(value) for value in dynamics["initial_intrinsic_phase"]]
    final_phases = [
        _clean_float(
            (value + config["propagation_steps"] * config["intrinsic_step"])
            % 1.0
        )
        for value in initial_phases
    ]
    trajectories = [
        {
            "carrier_id": carrier.carrier_id,
            "initial_port_intensities": initial[index],
            "settled_port_intensities": propagated[index],
            "repaired_port_intensities": repaired_rows[index],
            "initial_intrinsic_phase": _clean_float(initial_phases[index]),
            "settled_intrinsic_phase": final_phases[index],
        }
        for index, carrier in enumerate(federation.carriers)
    ]
    interventions = [
        {
            "row_id": str(event["event_id"]),
            "operation": str(event["update"]),
            "seam_id": str(event["seam_id"]),
            "read_set": [
                [str(item["carrier_id"]), int(item["port"])]
                for item in event["read_set"]
            ],
            "write_set": [
                [str(item["carrier_id"]), int(item["port"])]
                for item in event["write_set"]
            ],
            "mismatch_before": float(event["mismatch_before"]),
            "mismatch_after": float(event["mismatch_after"]),
        }
        for event in dynamics["repair_event_log"]
    ]

    events = [dict(row) for row in observer_log["events"]]
    feedback_by_record = {
        str(row["action_input_record_event_id"]): row
        for row in events
        if row["kind"] == "LOCAL_FEEDBACK"
    }
    geometry_rows = list(geometry["raw_primitive_rows"])
    records = [row for row in events if row["kind"] == "RECORD_COMMIT"]
    response_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        carrier_index = int(str(record["carrier_id"]).rsplit("-", 1)[1])
        port = int(record["port"])
        source_geometry = geometry_rows[index % len(geometry_rows)]
        feedback = feedback_by_record[str(record["event_id"])]
        initial_value = initial[carrier_index][port]
        settled_value = propagated[carrier_index][port]
        repaired_value = float(record["port_value"])
        material = {
            "row_id": f"response-{index:06d}",
            "record_event_id": str(record["event_id"]),
            "feedback_event_id": str(feedback["event_id"]),
            "observer_token": str(record["observer_token"]),
            "carrier_id": str(record["carrier_id"]),
            "port": port,
            "independent_geometric_parameter": float(
                source_geometry["geometric_parameter"]
            ),
            "raw_response": _clean_float(repaired_value - initial_value),
            "initial_port_intensity": initial_value,
            "settled_port_intensity": settled_value,
            "repaired_port_intensity": repaired_value,
            "propagation_delta": _clean_float(settled_value - initial_value),
            "repair_delta": _clean_float(repaired_value - settled_value),
            "refinement_level": int(config["support_refinement_level"]),
            "geometry_source_row_id": str(source_geometry["row_id"]),
        }
        response_rows.append({**material, "row_sha256": _sha(material)})

    neutral_geometry_rows: list[dict[str, Any]] = []
    for index, source_geometry in enumerate(geometry_rows):
        response = response_rows[index % len(response_rows)]
        frame = [float(value) for value in source_geometry["ordered_frame"]]
        gaps = [
            _clean_float(frame[1] - frame[0]),
            _clean_float(frame[2] - frame[1]),
            _clean_float(frame[3] - frame[2]),
        ]
        material = {
            "row_id": f"geometry-{index:06d}",
            "source_geometry_row_id": str(source_geometry["row_id"]),
            "source_response_row_id": str(response["row_id"]),
            "observer_token": str(response["observer_token"]),
            "trajectory_group_id": (
                f"seed-{int(config['seed'])}:"
                f"{response['observer_token']}:{response['carrier_id']}:"
                f"port-{int(response['port']):02d}"
            ),
            "ordered_frame": frame,
            "orientation": str(source_geometry["orientation"]),
            "cross_ratio": float(source_geometry["cross_ratio"]),
            "geometric_parameter": float(source_geometry["geometric_parameter"]),
            "neutral_feature_vector": [
                *gaps,
                float(source_geometry["cross_ratio"]),
                _clean_float(frame[3] - frame[0]),
                _clean_float(sum(frame) / 4.0),
                _clean_float(gaps[0] - gaps[2]),
            ],
            "observed_source_value": float(response["raw_response"]),
            "predictor_source_phase": "pre_intervention_initial_state",
            "response_source_phase": "post_repair_minus_initial_response",
            "predictor_response_field_intersection": [],
        }
        neutral_geometry_rows.append({**material, "row_sha256": _sha(material)})

    semantic_events, ancestry, _, _ = _semantic_source_events(observer_log)
    registered_source_inputs = {
            "carrier_count": int(config["carrier_count"]),
            "seed": int(config["seed"]),
            "rung": int(config["rung"]),
            "replicate_id": str(config["replicate_id"]),
            "preregistered_plan_sha256": str(
                config["preregistered_plan_sha256"]
            ),
            "propagation_steps": int(config["propagation_steps"]),
            "intrinsic_step": float(config["intrinsic_step"]),
            "coupling_strength": float(config["coupling_strength"]),
            "state_space": str(config["state_space"]),
            "rng_family": str(config["rng_family"]),
            "initialization_distribution": str(
                config["initialization_distribution"]
            ),
            "intrinsic_phase_distribution": str(
                config["intrinsic_phase_distribution"]
            ),
            "seam_update_rule": str(config["seam_update_rule"]),
            "cycles": int(config["cycles"]),
            "repair_fraction_per_cycle": float(
                config["repair_fraction_per_cycle"]
            ),
            "record_commit_cycles": int(config["record_commit_cycles"]),
            "observer_count": int(config["observer_count"]),
            "observer_support_size": int(config["observer_support_size"]),
            "observer_samples": int(config["observer_samples"]),
            "prediction_control": str(config["prediction_control"]),
            "feedback_enabled": bool(config["feedback_enabled"]),
            "checkpoint_interval": int(config["checkpoint_interval"]),
            "support_refinement_level": int(
                config["support_refinement_level"]
            ),
            "geometry_sample_count": int(config["geometry_sample_count"]),
            "geometry_transport": str(config["geometry_transport"]),
            "observer_cross_reads": bool(config["observer_cross_reads"]),
            "snapshot_coverage": str(config["snapshot_coverage"]),
    }
    registration = {
        "schema": "oph.physical-source-capture.registration.v1",
        "seed": int(config["seed"]),
        "rung": int(config["rung"]),
        "replicate_id": str(config["replicate_id"]),
        "carrier_count": int(config["carrier_count"]),
        "support_regulator_count": int(refinement["levels"][-1]["patch_count"]),
        "support_refinement_level": int(config["support_refinement_level"]),
        "observer_count": int(config["observer_count"]),
        "observer_support_size": int(config["observer_support_size"]),
        "preregistered_plan_sha256": str(
            config["preregistered_plan_sha256"]
        ),
        "source_inputs": registered_source_inputs,
        "source_inputs_sha256": _sha(registered_source_inputs),
    }
    geometry_samples = {
        "derivation_method": str(geometry["derivation_method"]),
        "orientation_fixed_from_source": bool(
            geometry["orientation_fixed_from_source"]
        ),
        "raw_primitive_rows": copy.deepcopy(geometry_rows),
        "support_refinement_levels": copy.deepcopy(refinement["levels"]),
        "conditional_expectation_rows": copy.deepcopy(
            refinement["conditional_expectations"]
        ),
    }
    components = {
        "registration": registration,
        "carrier_port_trajectories": trajectories,
        "intervention_rows": interventions,
        "response_rows": response_rows,
        "clock_pair_input": _unavailable_clock_pair_input(),
        "geometry_samples": geometry_samples,
        "geometry_control_rows": neutral_geometry_rows,
        "semantic_events": semantic_events,
        "raw_overlap_relations": _raw_overlap_relations(federation),
        "raw_ancestry_relations": ancestry,
    }
    declared_hashes = {name: _sha(components[name]) for name in _POSTRUN_COMPONENT_KEYS}
    capture = {
        "schema": POSTRUN_CAPTURE_SCHEMA,
        **components,
        "declared_hashes": declared_hashes,
        "primitive_root_sha256": _sha(
            {"schema": POSTRUN_CAPTURE_SCHEMA, "components": declared_hashes}
        ),
    }
    blockers = _postrun_capture_schema_blockers(capture)
    if blockers:
        raise RuntimeError("internal postrun capture invalid: " + ",".join(blockers))
    return capture


def _clock_pair_input_schema_blockers(value: Any) -> list[str]:
    blockers: list[str] = []
    if not isinstance(value, Mapping) or set(value) != _CLOCK_PAIR_INPUT_KEYS:
        return ["clock_pair_input_field_set_mismatch"]
    if value.get("schema") != TYPED_CLOCK_PAIR_INPUT_SCHEMA:
        blockers.append("clock_pair_input_schema_mismatch")
    contract = value.get("contract")
    if not isinstance(contract, Mapping) or set(contract) != _CLOCK_PAIR_CONTRACT_KEYS:
        return [*blockers, "clock_pair_contract_field_set_mismatch"]
    if contract.get("schema") != TYPED_CLOCK_PAIR_CONTRACT_SCHEMA:
        blockers.append("clock_pair_contract_schema_mismatch")
    if contract.get("join_key_fields") != list(_CLOCK_PAIR_JOIN_KEY_FIELDS):
        blockers.append("clock_pair_join_key_contract_mismatch")
    if contract.get("group_key_fields") != list(_CLOCK_PAIR_GROUP_KEY_FIELDS):
        blockers.append("clock_pair_group_key_contract_mismatch")
    if contract.get("minimum_refinement_level_count") != 2:
        blockers.append("clock_pair_refinement_level_contract_mismatch")
    if contract.get("source_fixed_oriented_frame_incidence_required") is not True:
        blockers.append("clock_pair_oriented_frame_contract_mismatch")
    if contract.get("scoring_constants_absent") is not True:
        blockers.append("clock_pair_scoring_firewall_contract_mismatch")

    modular_rows = value.get("modular_transport_rows")
    geometric_rows = value.get("geometric_flow_rows")
    if not isinstance(modular_rows, list) or not isinstance(geometric_rows, list):
        return [*blockers, "clock_pair_rows_must_be_lists"]
    producer_fields = (
        "modular_transport_producer_id",
        "modular_transport_producer_code_sha256",
        "modular_transport_source_field_sha256",
        "geometric_flow_producer_id",
        "geometric_flow_producer_code_sha256",
        "geometric_flow_source_field_sha256",
    )
    status = contract.get("status")
    if status == "UNAVAILABLE":
        if modular_rows or geometric_rows:
            blockers.append("unavailable_clock_pair_contract_has_rows")
        if any(contract.get(field) is not None for field in producer_fields):
            blockers.append("unavailable_clock_pair_contract_has_producer_binding")
        reason = contract.get("unavailable_reason")
        if not isinstance(reason, str) or not reason:
            blockers.append("unavailable_clock_pair_reason_missing")
        return blockers
    if status != "AVAILABLE":
        return [*blockers, "clock_pair_contract_status_invalid"]
    if contract.get("unavailable_reason") is not None:
        blockers.append("available_clock_pair_contract_has_unavailable_reason")
    if not modular_rows or not geometric_rows:
        blockers.append("available_clock_pair_contract_rows_missing")

    modular_id = contract.get("modular_transport_producer_id")
    geometric_id = contract.get("geometric_flow_producer_id")
    if (
        not isinstance(modular_id, str)
        or not modular_id
        or not isinstance(geometric_id, str)
        or not geometric_id
        or modular_id == geometric_id
    ):
        blockers.append("clock_pair_producer_ids_not_disjoint")
    hash_pairs = (
        (
            contract.get("modular_transport_producer_code_sha256"),
            contract.get("geometric_flow_producer_code_sha256"),
            "clock_pair_producer_code_hashes_not_disjoint",
        ),
        (
            contract.get("modular_transport_source_field_sha256"),
            contract.get("geometric_flow_source_field_sha256"),
            "clock_pair_source_field_hashes_not_disjoint",
        ),
    )
    for left, right, reason in hash_pairs:
        if (
            not isinstance(left, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", left) is None
            or not isinstance(right, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", right) is None
            or left == right
        ):
            blockers.append(reason)

    def validate_rows(
        rows: list[Any],
        expected_keys: frozenset[str],
        value_field: str,
        expected_source_hash: Any,
        *,
        require_oriented_hash: bool,
    ) -> tuple[set[tuple[Any, ...]], set[int]]:
        join_keys: set[tuple[Any, ...]] = set()
        levels: set[int] = set()
        row_ids: set[str] = set()
        for index, raw_row in enumerate(rows):
            if not isinstance(raw_row, Mapping) or set(raw_row) != expected_keys:
                blockers.append(f"{value_field}_row_{index}_field_set_mismatch")
                continue
            row = dict(raw_row)
            row_id = row.get("row_id")
            string_ids = [row.get(field) for field in _CLOCK_PAIR_JOIN_KEY_FIELDS if field != "refinement_level"]
            if (
                not isinstance(row_id, str)
                or not row_id
                or row_id in row_ids
                or any(not isinstance(item, str) or not item for item in string_ids)
            ):
                blockers.append(f"{value_field}_row_{index}_identity_invalid")
            else:
                row_ids.add(row_id)
            seed = row.get("source_seed")
            level = row.get("refinement_level")
            measured = row.get(value_field)
            if type(seed) is not int or type(level) is not int or level < 0:
                blockers.append(f"{value_field}_row_{index}_group_type_invalid")
            else:
                levels.add(level)
            if (
                isinstance(measured, bool)
                or not isinstance(measured, (int, float))
                or not math.isfinite(float(measured))
            ):
                blockers.append(f"{value_field}_row_{index}_value_invalid")
            if row.get("producer_source_field_sha256") != expected_source_hash:
                blockers.append(f"{value_field}_row_{index}_source_field_mismatch")
            if require_oriented_hash and (
                not isinstance(row.get("oriented_frame_incidence_sha256"), str)
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(row.get("oriented_frame_incidence_sha256")),
                )
                is None
            ):
                blockers.append(f"{value_field}_row_{index}_oriented_frame_hash_invalid")
            declared = row.pop("row_sha256", None)
            if declared != _sha(row):
                blockers.append(f"{value_field}_row_{index}_row_sha256_mismatch")
            join_key = tuple(raw_row.get(field) for field in _CLOCK_PAIR_JOIN_KEY_FIELDS)
            if join_key in join_keys:
                blockers.append(f"{value_field}_join_key_duplicated")
            join_keys.add(join_key)
        return join_keys, levels

    modular_keys, modular_levels = validate_rows(
        modular_rows,
        _MODULAR_TRANSPORT_ROW_KEYS,
        "modular_transport_time",
        contract.get("modular_transport_source_field_sha256"),
        require_oriented_hash=False,
    )
    geometric_keys, geometric_levels = validate_rows(
        geometric_rows,
        _GEOMETRIC_FLOW_ROW_KEYS,
        "geometric_flow_parameter",
        contract.get("geometric_flow_source_field_sha256"),
        require_oriented_hash=True,
    )
    if modular_keys != geometric_keys:
        blockers.append("clock_pair_join_key_sets_differ")
    if modular_levels != geometric_levels or len(modular_levels) < 2:
        blockers.append("clock_pair_actual_refinement_levels_insufficient")
    return blockers


def _postrun_capture_schema_blockers(capture: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if set(capture) != _POSTRUN_CAPTURE_KEYS:
        blockers.append("top_level_field_set_mismatch")
    if capture.get("schema") != POSTRUN_CAPTURE_SCHEMA:
        blockers.append("schema_mismatch")
    declared = capture.get("declared_hashes")
    if not isinstance(declared, Mapping) or set(declared) != set(
        _POSTRUN_COMPONENT_KEYS
    ):
        blockers.append("declared_hash_field_set_mismatch")
        declared = {}
    computed = {
        name: _sha(capture.get(name)) for name in _POSTRUN_COMPONENT_KEYS
    }
    if dict(declared) != computed:
        blockers.append("declared_component_hash_mismatch")
    expected_root = _sha(
        {"schema": POSTRUN_CAPTURE_SCHEMA, "components": computed}
    )
    if capture.get("primitive_root_sha256") != expected_root:
        blockers.append("primitive_root_sha256_mismatch")
    registration = capture.get("registration")
    if not isinstance(registration, Mapping) or set(registration) != {
        "schema",
        "seed",
        "rung",
        "replicate_id",
        "carrier_count",
        "support_regulator_count",
        "support_refinement_level",
        "observer_count",
        "observer_support_size",
        "preregistered_plan_sha256",
        "source_inputs",
        "source_inputs_sha256",
    }:
        blockers.append("registration_field_set_mismatch")
    elif (
        any(
            type(registration[key]) is not int
            for key in (
                "seed",
                "rung",
                "carrier_count",
                "support_regulator_count",
                "support_refinement_level",
                "observer_count",
                "observer_support_size",
            )
        )
        or registration["rung"] != registration["carrier_count"]
        or not isinstance(registration["replicate_id"], str)
        or not registration["replicate_id"]
        or not isinstance(registration["preregistered_plan_sha256"], str)
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            registration["preregistered_plan_sha256"],
        )
        is None
        or registration.get("source_inputs_sha256")
        != _sha(registration.get("source_inputs"))
    ):
        blockers.append("registration_value_contract_mismatch")
    elif not isinstance(registration["source_inputs"], Mapping) or set(
        registration["source_inputs"]
    ) != {
        "carrier_count",
        "seed",
        "rung",
        "replicate_id",
        "preregistered_plan_sha256",
        "propagation_steps",
        "intrinsic_step",
        "coupling_strength",
        "state_space",
        "rng_family",
        "initialization_distribution",
        "intrinsic_phase_distribution",
        "seam_update_rule",
        "cycles",
        "repair_fraction_per_cycle",
        "record_commit_cycles",
        "observer_count",
        "observer_support_size",
        "observer_samples",
        "prediction_control",
        "feedback_enabled",
        "checkpoint_interval",
        "support_refinement_level",
        "geometry_sample_count",
        "geometry_transport",
        "observer_cross_reads",
        "snapshot_coverage",
    }:
        blockers.append("source_inputs_field_set_mismatch")
    else:
        source_inputs = registration["source_inputs"]
        if (
            source_inputs["state_space"]
            != "normalized_complex_amplitude_in_C12"
            or source_inputs["rng_family"] != "numpy_generator_pcg64_v1"
            or source_inputs["initialization_distribution"]
            != "normalized_complex_gaussian_v1"
            or source_inputs["intrinsic_phase_distribution"]
            != "uniform_unit_interval_v1"
            or source_inputs["seam_update_rule"]
            != "disjoint_single_port_endpoint_arithmetic_mean_v1"
            or source_inputs["prediction_control"]
            != "semantic_hash_shuffle_v1"
            or source_inputs["feedback_enabled"] is not True
            or source_inputs["geometry_transport"]
            not in {"legacy", "held_out_flow"}
            or type(source_inputs["observer_cross_reads"]) is not bool
            or source_inputs["snapshot_coverage"] not in {"support", "spanning"}
            or any(
                type(source_inputs[key]) is not int
                for key in (
                    "carrier_count",
                    "seed",
                    "rung",
                    "propagation_steps",
                    "cycles",
                    "record_commit_cycles",
                    "observer_count",
                    "observer_support_size",
                    "observer_samples",
                    "checkpoint_interval",
                    "support_refinement_level",
                    "geometry_sample_count",
                )
            )
            or any(
                isinstance(source_inputs[key], bool)
                or not isinstance(source_inputs[key], (int, float))
                or not math.isfinite(float(source_inputs[key]))
                or float(source_inputs[key]) <= 0.0
                for key in (
                    "intrinsic_step",
                    "coupling_strength",
                    "repair_fraction_per_cycle",
                )
            )
            or float(source_inputs["repair_fraction_per_cycle"]) > 1.0
            or source_inputs["record_commit_cycles"] > source_inputs["cycles"]
            or source_inputs["carrier_count"] != registration["carrier_count"]
            or source_inputs["seed"] != registration["seed"]
            or source_inputs["rung"] != registration["rung"]
            or source_inputs["replicate_id"] != registration["replicate_id"]
            or source_inputs["preregistered_plan_sha256"]
            != registration["preregistered_plan_sha256"]
            or source_inputs["observer_count"] != registration["observer_count"]
            or source_inputs["observer_support_size"]
            != registration["observer_support_size"]
            or source_inputs["support_refinement_level"]
            != registration["support_refinement_level"]
        ):
            blockers.append("source_inputs_value_contract_mismatch")
    geometry_samples = capture.get("geometry_samples")
    if not isinstance(geometry_samples, Mapping) or set(
        geometry_samples
    ) != _POSTRUN_GEOMETRY_SAMPLE_KEYS:
        blockers.append("geometry_samples_field_set_mismatch")
    blockers.extend(_clock_pair_input_schema_blockers(capture.get("clock_pair_input")))
    for component, required_keys in _POSTRUN_ROW_KEYS.items():
        rows = capture.get(component)
        if not isinstance(rows, list) or not rows:
            blockers.append(f"{component}_must_be_nonempty_list")
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != required_keys:
                blockers.append(f"{component}_{index}_field_set_mismatch")
                break
    for component in (
        "response_rows",
        "geometry_control_rows",
        "raw_overlap_relations",
    ):
        for index, raw_row in enumerate(capture.get(component, [])):
            if not isinstance(raw_row, Mapping):
                continue
            row = dict(raw_row)
            declared_row_hash = row.pop("row_sha256", None)
            if declared_row_hash != _sha(row):
                blockers.append(f"{component}_{index}_row_sha256_mismatch")
                break
    for index, raw_row in enumerate(capture.get("raw_ancestry_relations", [])):
        if not isinstance(raw_row, Mapping):
            continue
        row = dict(raw_row)
        declared_edge_id = row.pop("edge_id", None)
        if declared_edge_id != _sha(row):
            blockers.append(f"raw_ancestry_relations_{index}_edge_id_mismatch")
            break
    source_material = {
        name: capture.get(name) for name in _POSTRUN_COMPONENT_KEYS
    }
    forbidden_hits = _postrun_forbidden_hits(source_material)
    if forbidden_hits:
        blockers.append("target_or_interpretive_labels_in_source_material")
    return blockers


def _postrun_forbidden_hits(value: Any, path: str = "postrun") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).lower()
            if key in _POSTRUN_FORBIDDEN_KEYS:
                hits.append(f"key:{key}@{path}")
            hits.extend(_postrun_forbidden_hits(item, f"{path}.{raw_key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            hits.extend(_postrun_forbidden_hits(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        normalized = value.lower()
        for label in _POSTRUN_FORBIDDEN_LABELS:
            if re.search(
                rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])",
                normalized,
            ):
                hits.append(f"label:{label}@{path}")
    return sorted(set(hits))


def _target_hits(value: Any, path: str = "config") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            hits.extend(_target_hits(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            hits.extend(_target_hits(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        normalized = value.lower()
        for token in _FORBIDDEN_SOURCE_TOKENS:
            if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", normalized):
                hits.append(f"{token}@{path}")
    return sorted(set(hits))


def capture_physical_source(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct one deterministic source-side capture and its P0--P3 reports."""

    normalized = _normalize_config(config)
    federation = _build_federation(normalized)
    sewing = federation_sewing_report(federation)
    bundle = reference_federation_instrument_bundle(federation)
    actions = icosahedral_a5_port_permutations()
    permutations = {
        carrier.carrier_id: actions[
            1
            + int(
                hashlib.sha256(
                    f"{normalized['seed']}:{carrier.carrier_id}".encode("ascii")
                ).hexdigest(),
                16,
            )
            % (len(actions) - 1)
        ]
        for carrier in federation.carriers
    }
    quotient = carrier_quotient_invariance_report(
        federation,
        relabel_federation_ports(federation, permutations),
        permutations,
    )
    (
        dynamics,
        propagated_amplitudes,
        terminal_complex_lift,
        repaired,
        repair_events,
    ) = _source_dynamics(normalized, federation)
    state_root = _sha(
        {
            "federation_bundle_sha256": _sha(bundle),
            "initial_state_sha256": dynamics["initial_state_sha256"],
            "propagated_state_sha256": dynamics[
                "propagated_port_statistics_sha256"
            ],
            "repaired_state_sha256": dynamics["repaired_port_statistics_sha256"],
            "joint_terminal_state_sha256": dynamics[
                "joint_terminal_state_sha256"
            ],
            "repair_cycle_ledger_sha256": dynamics[
                "repair_cycle_ledger_sha256"
            ],
            "record_state_snapshots_sha256": dynamics[
                "record_state_snapshots_sha256"
            ],
        }
    )
    observer = _observer_loop(normalized, federation, dynamics, state_root)
    cap_report, rho = _m4_cap_state(terminal_complex_lift)
    refinement = _refinement_report(normalized["support_refinement_level"], rho)
    initial_intensities = np.asarray(
        _port_intensities_from_complex_rows(
            dynamics["initial_port_amplitudes"]
        ),
        dtype=float,
    )
    geometry = _independent_geometry(
        normalized, initial_intensities, normalized["geometry_sample_count"]
    )
    postrun_capture = _build_postrun_capture(
        normalized,
        federation,
        dynamics,
        propagated_amplitudes,
        repaired,
        observer,
        geometry,
        refinement,
    )
    support_rows_pass = bool(
        sewing["observer_support_count"] > 0
        and not sewing["observer_support_failure_examples"]
    )
    transaction_complete = bool(
        dynamics["repair_event_count"] > 0
        and dynamics[
            "TRANSACTION_VALIDATION_COMPLETE_READ_CONFLICT_SET_RECEIPT"
        ]
        and dynamics["UNION_PAYLOAD_ATOMIC_REVALIDATION_RECEIPT"]
        and dynamics["REPAIR_ORDER_REPLAY_EXACT_RECEIPT"]
        and dynamics["REPAIR_IDEMPOTENCE_REPLAY_EXACT_RECEIPT"]
        and dynamics["REPAIR_TERMINAL_FIXED_POINT_RECEIPT"]
    )
    prediction_passed = bool(
        observer["direct_prediction_match_count"]
        > observer["shuffled_prediction_match_count"]
    )
    feedback_passed = bool(
        observer["feedback_changed_action_count"] > 0
        and observer["feedback_action_applied_to_next_record_count"]
        == observer["record_count"] - normalized["observer_count"]
    )
    generator_dependencies = [
        "source_seed",
        "carrier_cardinality",
        "regular_icosahedron_template",
        "typed_seam_ledger",
        "dimensionless_intrinsic_step",
        "support_regulator_separate_from_source_topology",
    ]
    source_forbidden_hits = _target_hits(
        {"input_config": normalized, "generator_dependencies": generator_dependencies}
    )
    local_template_hash = sewing["carrier_conformance_summary"][
        "structural_class_examples"
    ][0]["structural_class_sha256"]
    p0_receipt = bool(
        sewing["FEDERATION_SEWING_RECEIPT"]
        and quotient["CARRIER_QUOTIENT_INVARIANCE_RECEIPT"]
        and dynamics["local_a5_equivariance"]["receipt"]
        and transaction_complete
        and support_rows_pass
        and prediction_passed
        and feedback_passed
        and observer["checkpoint_replay_exact"]
        and observer["full_port_record_signature_count"] == observer["record_count"]
        and observer["readback_source_signature_match_count"]
        == observer["readback_count"]
        and sewing["sewn_local_port_count"]
        == normalized["carrier_count"] * 12
        and sewing["declared_external_local_port_count"] == 0
        and not source_forbidden_hits
    )
    source_observer = {
        "schema_version": "oph_physical_h3_kms_source_capture_contract_v2",
        "SOURCE_PATCH_ARCHITECTURE_RECEIPT": p0_receipt,
        "PATCH_LOCAL_STATE_RECEIPT": p0_receipt,
        "PATCH_PORT_BOUNDARY_RECEIPT": p0_receipt,
        "PATCH_READBACK_RECEIPT": p0_receipt,
        "PATCH_ALL_PORT_READBACK_RECEIPT": p0_receipt,
        "RECORD_SIGNATURE_BINDS_ALL_LOCAL_PORT_STATE_RECEIPT": p0_receipt,
        "ECHOSAHEDRAL_LOCAL_PATCH_ARCHITECTURE_RECEIPT": p0_receipt,
        "ECHOSAHEDRAL_CARRIER_CONFORMANCE": sewing[
            "ECHOSAHEDRAL_CARRIER_CONFORMANCE"
        ],
        "FEDERATION_SEWING_RECEIPT": sewing["FEDERATION_SEWING_RECEIPT"],
        "CARRIER_QUOTIENT_INVARIANCE_RECEIPT": quotient[
            "CARRIER_QUOTIENT_INVARIANCE_RECEIPT"
        ],
        "carrier_quotient_invariance_scope": (
            "federation_topology_only_excludes_semantic_event_identity"
        ),
        "SEMANTIC_EVENT_A5_QUOTIENT_INVARIANCE_RECEIPT": False,
        "semantic_event_identity_status": (
            "TRANSPORT_METADATA_INVARIANT_PRESENTATION_BOUND_SEMANTIC_KEY"
        ),
        "semantic_event_identity_basis": [
            "presentation_carrier_id",
            "presentation_port_index",
            "presentation_bound_source_state_root",
            "explicit_semantic_read_resources",
            "declared_parents_record_order_and_sequence_excluded",
        ],
        "INDEPENDENT_SUPPORT_REGULATOR_RECEIPT": refinement[
            "COMMUTATIVE_CELL_REFINEMENT_DIAGNOSTIC_RECEIPT"
        ],
        "independent_support_regulator_scope": (
            "construction_order_and_source_topology_only"
        ),
        "PHYSICAL_INDEPENDENT_SUPPORT_REGULATOR_RECEIPT": False,
        "SUPPORT_REGULATOR_STRUCTURAL_DIAGNOSTIC_RECEIPT": refinement[
            "COMMUTATIVE_CELL_REFINEMENT_DIAGNOSTIC_RECEIPT"
        ],
        "support_regulator_claim_status": (
            "STRUCTURAL_GEOMETRY_ONLY_NONCOMMUTATIVE_CERTIFICATE_NOT_ESTABLISHED"
        ),
        "CARRIER_REFINEMENT_NATURALITY_RECEIPT": False,
        "carrier_refinement_naturality_status": "NOT_EVALUATED",
        "PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT": bool(
            sewing["PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT"]
        ),
        "CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT": False,
        "physical_federation_status": (
            "NONEMPTY_TRIPLE_OVERLAP_AND_COCYCLE_SEWING_REALIZED"
            if sewing["PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT"]
            else "BLOCKED_BY_INTERFACE_ALGEBRA_SEWING_CHECKS"
        ),
        "carrier_to_support_realization_status": "NOT_EVALUATED",
        "TRANSACTION_VALIDATION_COMPLETE_READ_CONFLICT_SET_RECEIPT": transaction_complete,
        "UNION_PAYLOAD_ATOMIC_REVALIDATION_RECEIPT": transaction_complete,
        "source_generator_target_free": not source_forbidden_hits,
        "source_forbidden_target_hits": source_forbidden_hits,
        "source_architecture": {
            "bounded_patch_system": True,
            "simulation_native_source": True,
            "carrier_count": normalized["carrier_count"],
            "local_state_space": "normalized_complex_amplitude_in_C12",
            "local_state_dimension": 12,
            "local_state_complex_coordinate_count": 12,
            "local_state_real_coordinate_count": 24,
            "local_state_factor_count": 12,
            "local_state_factor_count_semantics": (
                "legacy_preflight_alias_for_complex_port_channel_count"
            ),
            "materialized_local_state_coordinate_count": normalized["carrier_count"] * 12,
            "boundary_port_count": 12,
            "carrier_family": "federated_echosahedral_patch_system",
            "one_local_echosahedron_per_carrier": True,
            "carrier_is_not_support_chart_cell": True,
            "carrier_is_not_primitive_observer": True,
            "all_local_port_readout_maps_materialized": True,
            "all_local_port_states_bound_into_records": True,
            "record_binding_scope": (
                "every_sampled_carrier_record_commits_the_full_twelve_port_field"
            ),
            "all_twelve_ports_sewn_once_per_carrier": bool(
                sewing["sewn_local_port_count"]
                == normalized["carrier_count"] * 12
            ),
            "support_regulator_drives_source_topology": False,
            "local_patch_template_hash": local_template_hash,
            "patch_port_state_sha256": dynamics["joint_terminal_state_sha256"],
            "source_architecture_hash": state_root,
        },
        "repair_dynamics": {
            "local_update_rule": True,
            "uses_only_local_state_and_ports": True,
            "target_free_rule": not source_forbidden_hits,
            "repair_event_count": dynamics["repair_event_count"],
            "repair_event_examples_materialized": len(repair_events),
            "repair_event_examples_complete": dynamics[
                "repair_event_examples_complete"
            ],
            "repair_cycles_executed": dynamics["cycles"],
            "repair_fraction_per_cycle": dynamics[
                "repair_fraction_per_cycle"
            ],
            "terminal_maximum_seam_mismatch": dynamics[
                "terminal_maximum_seam_mismatch"
            ],
            "physical_complex_repair_lift_receipt": dynamics[
                "PHYSICAL_COMPLEX_REPAIR_LIFT_RECEIPT"
            ],
            "nonlocal_write_count": 0,
            "repair_rule_hash": _sha(
                {"rule": dynamics["repair_rule"], "federation": _sha(bundle)}
            ),
            "repair_event_log_hash": dynamics["repair_event_log_sha256"],
        },
        "record_observer": {
            "observer_count": normalized["observer_count"],
            "committed_record_count": observer["record_count"],
            "readback_count": observer["readback_count"],
            "feedback_event_count": observer["feedback_count"],
            "readback_changes_future_local_actions": feedback_passed,
            "records_causally_bound_to_writes": True,
            "bounded_interface_verified": support_rows_pass,
            "self_prediction_beats_shuffled_control": prediction_passed,
            "feedback_ablation_changes_future_actions": feedback_passed,
            "feedback_action_applied_to_next_record_count": observer[
                "feedback_action_applied_to_next_record_count"
            ],
            "checkpoint_continuation_verified": observer[
                "checkpoint_replay_exact"
            ],
            "orphan_read_count": 0,
            "record_readback_feedback_log_hash": observer["event_log_sha256"],
            "every_record_binds_full_c12_state": bool(
                observer["full_port_record_signature_count"]
                == observer["record_count"]
            ),
            "readback_recomputed_from_source_field": bool(
                observer["readback_source_signature_match_count"]
                == observer["readback_count"]
            ),
        },
    }
    preflight_config = {
        "seed": normalized["seed"],
        "rung": normalized["rung"],
        "replicate_id": normalized["replicate_id"],
        "preregistered_plan_sha256": normalized[
            "preregistered_plan_sha256"
        ],
        "source_federation": {
            "family": "federated_echosahedral_carriers",
            "carrier_count": normalized["carrier_count"],
        },
        "support_regulator": {
            "family": "nested_geodesic_icosahedral",
            "patch_basis": "cells",
            "refinement_level": normalized["support_refinement_level"],
        },
        "bw": {"state_mode": _CONSTRUCTED_M4_STATE_MODE},
    }
    source_artifacts = {
        "generator_dependencies": generator_dependencies,
        "federation_bundle": bundle,
        "federation_bundle_sha256": _sha(bundle),
        "federation_sewing_sha256": _sha(sewing),
        "presentation_quotient_sha256": _sha(quotient),
        "dynamics": dynamics,
        "observer_log": observer,
        "support_refinement_sha256": refinement["certificate_sha256"],
        "cap_state_raw_primitives": cap_report["raw_primitives"],
        "geometry_raw_primitives": geometry["raw_primitive_rows"],
        "source_state_root_sha256": state_root,
    }
    source_hashes = {
        "input_config": _sha(normalized),
        "federation": source_artifacts["federation_bundle_sha256"],
        "initial_state": dynamics["initial_state_sha256"],
        "propagated_state": dynamics["propagated_port_statistics_sha256"],
        "repaired_state": dynamics["repaired_port_statistics_sha256"],
        "repair_log": dynamics["repair_event_log_sha256"],
        "observer_log": observer["event_log_sha256"],
        "checkpoint": _sha(observer["checkpoint"]),
        "refinement": refinement["certificate_sha256"],
        "cap_state": cap_report["raw_primitives_sha256"],
        "independent_geometry": geometry["geometry_derivation_hash"],
        "postrun_capture": postrun_capture["primitive_root_sha256"],
    }
    result = {
        "schema": SCHEMA,
        "artifact_type": ARTIFACT_TYPE,
        "verifier_version": VERIFIER_VERSION,
        "input_config": normalized,
        "config": preflight_config,
        "reports": {
            "source_observer": source_observer,
            "refinement": refinement,
            "prime_geometric_state": cap_report,
            "independent_geometry": geometry,
        },
        "source_artifacts": source_artifacts,
        "postrun_capture": postrun_capture,
        "source_hashes": source_hashes,
        "source_root_sha256": _sha(source_hashes),
        "claim_boundary": (
            "Finite target-blind source capture through preflight P3 only. It does "
            "not perform or score clock candidates, geometry models, event-manifold "
            "fits, campaign cells, physical promotion, or branch retirement. The "
            "constructed first-16-amplitude M4/Gibbs state, repeated-rho refinement "
            "check, same-source ordered-frame geometry, and presentation-bound event "
            "keys are diagnostics only; they do not establish a physical/max-entropy "
            "cap state, noncommutative multiresolution certificate, independent "
            "geometry producer, or A5-quotient-invariant semantic identity."
        ),
    }
    result["capture_sha256"] = _sha(result)
    return result


def verify_physical_source_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a capture from its primitive config and compare exact JSON bytes."""

    blockers: list[str] = []
    if not isinstance(capture, Mapping):
        return {
            "schema": "oph.physical-source-capture-verification/1.0.0",
            "SOURCE_CAPTURE_REPLAY_RECEIPT": False,
            "blockers": ["capture_must_be_a_mapping"],
        }
    candidate = copy.deepcopy(dict(capture))
    declared_hash = candidate.pop("capture_sha256", None)
    if declared_hash != _sha(candidate):
        blockers.append("capture_sha256_mismatch")
    try:
        recomputed = capture_physical_source(candidate.get("input_config", {}))
    except (TypeError, ValueError) as exc:
        blockers.append(f"capture_replay_failed:{type(exc).__name__}:{exc}")
        recomputed = None
    if recomputed is not None and _canonical_bytes(dict(capture)) != _canonical_bytes(
        recomputed
    ):
        blockers.append("capture_is_not_exact_replay_output")
    return {
        "schema": "oph.physical-source-capture-verification/1.0.0",
        "verifier_version": VERIFIER_VERSION,
        "SOURCE_CAPTURE_REPLAY_RECEIPT": not blockers,
        "source_root_sha256": (
            recomputed.get("source_root_sha256") if recomputed is not None else None
        ),
        "blockers": blockers,
    }


def write_physical_source_capture(
    output_path: str | Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one canonical JSON capture and return the in-memory value."""

    capture = capture_physical_source(config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(capture, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return capture


__all__ = [
    "ARTIFACT_TYPE",
    "POSTRUN_CAPTURE_SCHEMA",
    "SCHEMA",
    "TYPED_CLOCK_PAIR_CONTRACT_SCHEMA",
    "TYPED_CLOCK_PAIR_INPUT_SCHEMA",
    "VERIFIER_VERSION",
    "capture_physical_source",
    "verify_physical_source_capture",
    "write_physical_source_capture",
]
