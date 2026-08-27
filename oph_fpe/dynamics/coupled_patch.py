"""Finite, conditionally graph-local order-parameter dynamics for diagnostics.

This module is deliberately separate from the protected-authority repair
kernel.  It supplies a small experimental model in which patch state really
is coupled across graph edges and committed observer records can influence
later state transitions.  It is an instrument for studying internal
correlations, defects, response fronts, and related diagnostics.  Its update
parameter, graph distance, order parameter, and energy are abstract units;
the model has no physical CMB, matter, clock, length, or energy
identification.

Every patch has a bounded scalar order parameter, a bounded conjugate
velocity, a three-valued record, a commit bit, and a finite persistence
counter.  The two continuous-looking coordinates are rounded after every
move to declared finite grids, so the local dynamical state space is finite.
The force at a patch reads only that patch, its incident neighbors, and its
own previously committed record, conditional on an exogenous global update
cycle, spatially uniform quench value, and per-node random draw.  A record
committed after transition ``t`` can first contribute a readback force during
transition ``t + 1``.

The implementation is intentionally target-blind and labelled
``INTERNAL_DIAGNOSTIC_ONLY`` throughout.  It does not select parameters by
comparison with observations and it grants no physical-comparison license.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "oph.coupled_patch_internal_diagnostic.v1"
CLASSIFICATION = "INTERNAL_DIAGNOSTIC_ONLY"
FRAME_ARTIFACT = "coupled_patch_frames.npz"
MANIFEST_ARTIFACT = "coupled_patch_manifest.json"
README_ARTIFACT = "README.md"


@dataclass(frozen=True)
class CoupledPatchConfig:
    """Declared parameters for the finite coupled-patch map.

    The defaults are conservative diagnostic values, not fitted or physical
    constants.  ``mass2`` is smoothly quenched from ``mass2_start`` to
    ``mass2_end``.  The local potential is

    ``V(q,t) = mass2(t) q**2 / 2 + quartic q**4 / 4``.

    State and velocity are rounded to odd-cardinality grids containing zero
    and both declared endpoints.  Quantization uses NumPy ``rint``: nearest
    grid index with ties to the even index.
    """

    cycles: int = 256
    dt: float = 0.025
    seed: int = 20260827
    state_bound: float = 2.0
    velocity_bound: float = 4.0
    state_levels: int = 32769
    velocity_levels: int = 32769
    initial_state_scale: float = 0.05
    initial_velocity_scale: float = 0.0
    coupling: float = 0.30
    quartic: float = 1.0
    mass2_start: float = 1.0
    mass2_end: float = -1.0
    quench_start_fraction: float = 0.10
    quench_end_fraction: float = 0.55
    quench_kind: str = "smoothstep"
    damping: float = 0.25
    noise_amplitude: float = 0.02
    record_threshold: float = 0.60
    record_persistence: int = 4
    record_amplitude: float = 1.0
    feedback_strength: float = 0.15
    defect_threshold: float = 1.0
    snapshot_stride: int = 1
    stability_limit: float = 0.50


@dataclass(frozen=True)
class LocalizedIntervention:
    """A bounded state/velocity kick on a graph ball before one transition.

    ``cycle=t`` means the impulse is applied after the saved state at ``t``
    and before the force that produces state ``t+1``.
    """

    center_node: int
    cycle: int
    radius_hops: int = 0
    state_delta: float = 0.0
    velocity_delta: float = 0.0


@dataclass(frozen=True)
class CoupledPatchResult:
    """Lossless saved-frame contract consumed by downstream diagnostics.

    ``cycles[k]`` labels all arrays whose name ends in ``_frames`` at index
    ``k``.  ``feedback_force_frames[k]`` is the record-derived force used to
    reach that snapshot; it is zero at the initial frame.  Edge arrays use
    the common edge order stored in ``left`` and ``right``.

    ``intervention_mask`` is the union over affected nodes.
    ``intervention_delta[:, 0]`` and ``intervention_delta[:, 1]`` are the
    aggregate exact, post-quantization changes to state and velocity.  The
    plural arrays retain every impulse separately; their first dimension has
    length zero for a control run.  When ``snapshot_stride`` exceeds one the
    saved arrays are an explicitly sampled history; omitted transition states
    are not represented by this object.
    """

    points: np.ndarray
    left: np.ndarray
    right: np.ndarray
    cycles: np.ndarray
    state_frames: np.ndarray
    velocity_frames: np.ndarray
    record_frames: np.ndarray
    commit_frames: np.ndarray
    defect_frames: np.ndarray
    feedback_force_frames: np.ndarray
    mass2_frames: np.ndarray
    intervention_mask: np.ndarray
    intervention_delta: np.ndarray
    intervention_cycles: np.ndarray
    intervention_masks: np.ndarray
    intervention_deltas: np.ndarray
    config: Mapping[str, Any]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class PairedCoupledPatchResult:
    """Exact same-seed control and localized counterfactual."""

    control: CoupledPatchResult
    intervened: CoupledPatchResult
    state_delta_frames: np.ndarray
    velocity_delta_frames: np.ndarray
    record_delta_frames: np.ndarray
    defect_xor_frames: np.ndarray
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class CollisionCounterfactualResult:
    """Baseline, A, B, and A+B common-noise nonlinear response packet.

    Each residual is computed literally as ``AB - A - B + baseline``.  This
    is an internal nonlinear-interaction diagnostic.  A nonzero value is not
    a physical scattering amplitude or cross section.
    """

    baseline: CoupledPatchResult
    a: CoupledPatchResult
    b: CoupledPatchResult
    ab: CoupledPatchResult
    state_nonlinear_residual_frames: np.ndarray
    velocity_nonlinear_residual_frames: np.ndarray
    record_nonlinear_residual_frames: np.ndarray
    defect_nonlinear_residual_frames: np.ndarray
    receipt: Mapping[str, Any]


def simulate_coupled_patch(
    points: np.ndarray | Sequence[Sequence[float]],
    left: np.ndarray | Sequence[int],
    right: np.ndarray | Sequence[int],
    config: CoupledPatchConfig | Mapping[str, Any] | None = None,
    *,
    intervention: (
        LocalizedIntervention
        | Mapping[str, Any]
        | Sequence[LocalizedIntervention | Mapping[str, Any]]
        | None
    ) = None,
    initial_state: np.ndarray | Sequence[float] | None = None,
    initial_velocity: np.ndarray | Sequence[float] | None = None,
) -> CoupledPatchResult:
    """Run the target-blind coupled-patch diagnostic.

    The graph is canonicalized to sorted unoriented edges before evolution.
    Random initialization and process noise use independent ``PCG64`` streams
    derived from the declared seed.  Supplying initial arrays does not change
    the noise stream, which is important for paired counterfactuals.
    """

    cfg = _coerce_config(config)
    pts, edge_left, edge_right, degree = _validate_graph(points, left, right)
    _validate_config(cfg, node_count=pts.shape[0], max_degree=int(degree.max()))
    events = _coerce_interventions(intervention)
    for event in events:
        _validate_intervention(event, cfg, pts.shape[0])

    node_count = int(pts.shape[0])
    seed_sequence = np.random.SeedSequence(cfg.seed)
    state_seed, velocity_seed, noise_seed = seed_sequence.spawn(3)
    state_rng = np.random.Generator(np.random.PCG64(state_seed))
    velocity_rng = np.random.Generator(np.random.PCG64(velocity_seed))
    noise_rng = np.random.Generator(np.random.PCG64(noise_seed))

    state, initial_state_clip_count = _initial_coordinate(
        initial_state,
        node_count=node_count,
        bound=cfg.state_bound,
        levels=cfg.state_levels,
        scale=cfg.initial_state_scale,
        rng=state_rng,
        label="initial_state",
    )
    velocity, initial_velocity_clip_count = _initial_coordinate(
        initial_velocity,
        node_count=node_count,
        bound=cfg.velocity_bound,
        levels=cfg.velocity_levels,
        scale=cfg.initial_velocity_scale,
        rng=velocity_rng,
        label="initial_velocity",
    )
    initial_state_hash = _array_sha256(state)
    initial_velocity_hash = _array_sha256(velocity)

    record = np.zeros(node_count, dtype=np.int8)
    committed = np.zeros(node_count, dtype=bool)
    candidate_sign = np.zeros(node_count, dtype=np.int8)
    persistence = np.zeros(node_count, dtype=np.int32)

    intervention_mask = np.zeros(node_count, dtype=bool)
    intervention_delta = np.zeros((node_count, 2), dtype=np.float64)
    intervention_masks = np.zeros((len(events), node_count), dtype=bool)
    intervention_deltas = np.zeros((len(events), node_count, 2), dtype=np.float64)
    intervention_state_clip_counts = np.zeros(len(events), dtype=np.int64)
    intervention_velocity_clip_counts = np.zeros(len(events), dtype=np.int64)
    for event_index, event in enumerate(events):
        intervention_masks[event_index] = _graph_ball_mask(
            node_count,
            edge_left,
            edge_right,
            center=event.center_node,
            radius=event.radius_hops,
        )
    if events:
        intervention_mask = np.any(intervention_masks, axis=0)

    snapshots = _snapshot_cycles(cfg.cycles, cfg.snapshot_stride)
    snapshot_set = set(snapshots)
    frame_cycles: list[int] = []
    state_frames: list[np.ndarray] = []
    velocity_frames: list[np.ndarray] = []
    record_frames: list[np.ndarray] = []
    commit_frames: list[np.ndarray] = []
    defect_frames: list[np.ndarray] = []
    feedback_force_frames: list[np.ndarray] = []
    mass2_frames: list[float] = []

    def append_frame(cycle: int, feedback_force: np.ndarray, mass2: float) -> None:
        frame_cycles.append(int(cycle))
        state_frames.append(state.copy())
        velocity_frames.append(velocity.copy())
        record_frames.append(record.astype(np.float64) * cfg.record_amplitude)
        commit_frames.append(committed.copy())
        defect_frames.append(
            _edge_defects(state, edge_left, edge_right, cfg.defect_threshold)
        )
        feedback_force_frames.append(feedback_force.copy())
        mass2_frames.append(float(mass2))

    append_frame(0, np.zeros(node_count, dtype=np.float64), _mass2_at(cfg, 0))
    noise_hasher = hashlib.sha256()
    state_clip_count = 0
    velocity_clip_count = 0
    hard_overshoot_count = 0
    readback_nonzero_force_count = 0
    readback_velocity_write_count = 0
    readback_state_write_count = 0
    readback_any_write_count = 0
    first_commit_cycle = np.full(node_count, -1, dtype=np.int64)

    for step in range(cfg.cycles):
        for event_index, event in enumerate(events):
            if step != event.cycle:
                continue
            event_mask = intervention_masks[event_index]
            before_state = state.copy()
            before_velocity = velocity.copy()
            requested_state = state[event_mask] + event.state_delta
            requested_velocity = velocity[event_mask] + event.velocity_delta
            intervention_state_clip_counts[event_index] = int(
                np.count_nonzero(np.abs(requested_state) > cfg.state_bound)
            )
            intervention_velocity_clip_counts[event_index] = int(
                np.count_nonzero(np.abs(requested_velocity) > cfg.velocity_bound)
            )
            state[event_mask] = _quantize(
                np.clip(
                    requested_state,
                    -cfg.state_bound,
                    cfg.state_bound,
                ),
                cfg.state_bound,
                cfg.state_levels,
            )
            velocity[event_mask] = _quantize(
                np.clip(
                    requested_velocity,
                    -cfg.velocity_bound,
                    cfg.velocity_bound,
                ),
                cfg.velocity_bound,
                cfg.velocity_levels,
            )
            intervention_deltas[event_index, :, 0] = state - before_state
            intervention_deltas[event_index, :, 1] = velocity - before_velocity
            if not np.any(intervention_deltas[event_index]):
                raise ValueError(
                    f"intervention {event_index} is an exact no-op after "
                    "finite-grid quantization"
                )
            intervention_delta += intervention_deltas[event_index]

        mass2 = _mass2_at(cfg, step)
        laplacian = _nearest_neighbor_laplacian(
            state, edge_left, edge_right, node_count
        )
        feedback_force = (
            cfg.feedback_strength
            * committed.astype(np.float64)
            * (record.astype(np.float64) * cfg.record_amplitude - state)
        )
        readback_nonzero_force_count += int(np.count_nonzero(feedback_force))
        local_force = -mass2 * state - cfg.quartic * state**3
        force_without_readback = local_force + cfg.coupling * laplacian

        noise = noise_rng.standard_normal(node_count, dtype=np.float64)
        noise_hasher.update(np.ascontiguousarray(noise).tobytes())
        proposed_velocity_without_readback = (
            (1.0 - cfg.damping * cfg.dt) * velocity
            + cfg.dt * force_without_readback
            + cfg.noise_amplitude * math.sqrt(cfg.dt) * noise
        )
        proposed_velocity = (
            proposed_velocity_without_readback + cfg.dt * feedback_force
        )
        velocity_clip_count += int(
            np.count_nonzero(np.abs(proposed_velocity) > cfg.velocity_bound)
        )
        if np.any(np.abs(proposed_velocity) > 4.0 * cfg.velocity_bound):
            hard_overshoot_count += int(
                np.count_nonzero(
                    np.abs(proposed_velocity) > 4.0 * cfg.velocity_bound
                )
            )
            raise FloatingPointError(
                "velocity exceeded four times its declared bound; aborting "
                "instead of hiding numerical instability by clipping"
            )
        next_velocity = _quantize(
            np.clip(
                proposed_velocity, -cfg.velocity_bound, cfg.velocity_bound
            ),
            cfg.velocity_bound,
            cfg.velocity_levels,
        )
        proposed_state = state + cfg.dt * next_velocity
        state_clip_count += int(np.count_nonzero(np.abs(proposed_state) > cfg.state_bound))
        if np.any(np.abs(proposed_state) > 4.0 * cfg.state_bound):
            hard_overshoot_count += int(
                np.count_nonzero(np.abs(proposed_state) > 4.0 * cfg.state_bound)
            )
            raise FloatingPointError(
                "state exceeded four times its declared bound; aborting "
                "instead of hiding numerical instability by clipping"
            )
        next_state = _quantize(
            np.clip(proposed_state, -cfg.state_bound, cfg.state_bound),
            cfg.state_bound,
            cfg.state_levels,
        )
        if np.any(feedback_force != 0.0):
            shadow_velocity = _quantize(
                np.clip(
                    proposed_velocity_without_readback,
                    -cfg.velocity_bound,
                    cfg.velocity_bound,
                ),
                cfg.velocity_bound,
                cfg.velocity_levels,
            )
            shadow_state = _quantize(
                np.clip(
                    state + cfg.dt * shadow_velocity,
                    -cfg.state_bound,
                    cfg.state_bound,
                ),
                cfg.state_bound,
                cfg.state_levels,
            )
            velocity_written = next_velocity != shadow_velocity
            state_written = next_state != shadow_state
            readback_velocity_write_count += int(np.count_nonzero(velocity_written))
            readback_state_write_count += int(np.count_nonzero(state_written))
            readback_any_write_count += int(
                np.count_nonzero(velocity_written | state_written)
            )
        velocity = next_velocity
        state = next_state
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(velocity)):
            raise FloatingPointError("non-finite coupled-patch state")

        sign = np.sign(state).astype(np.int8)
        eligible = np.abs(state) >= cfg.record_threshold
        same_candidate = eligible & (sign == candidate_sign) & (sign != 0)
        persistence = np.where(
            same_candidate,
            np.minimum(persistence + 1, cfg.record_persistence),
            np.where(eligible & (sign != 0), 1, 0),
        ).astype(np.int32)
        candidate_sign = np.where(eligible, sign, 0).astype(np.int8)
        newly_committed = (~committed) & (persistence >= cfg.record_persistence)
        record[newly_committed] = sign[newly_committed]
        first_commit_cycle[newly_committed] = step + 1
        committed |= newly_committed

        completed_cycle = step + 1
        if completed_cycle in snapshot_set:
            append_frame(completed_cycle, feedback_force, mass2)

    config_payload = asdict(cfg)
    graph_payload = {
        "point_count": node_count,
        "edge_count": int(edge_left.size),
        "points_sha256": _array_sha256(pts),
        "left_sha256": _array_sha256(edge_left),
        "right_sha256": _array_sha256(edge_right),
    }
    stability_number = _stability_number(cfg, int(degree.max()))
    provenance: dict[str, Any] = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "physical_interpretation_allowed": False,
        "units": "dimensionless_nonphysical_internal_update_units",
        "target_data_consumed": False,
        # Compatibility key.  The qualified fields below are authoritative:
        # the state update is graph-local conditional on exogenous cycle,
        # quench, and per-node noise inputs.
        "graph_local": True,
        "graph_local_conditional_on_exogenous_inputs": True,
        "conditional_read_radius_hops": 1,
        "maximum_intervention_influence_growth_hops_per_transition": 1,
        "exogenous_spatially_uniform_quench": True,
        "autonomous_local_clock": False,
        "global_update_cycle_identified_with_physical_time": False,
        "finite_local_state": True,
        "state_grid_levels": cfg.state_levels,
        "velocity_grid_levels": cfg.velocity_levels,
        "finite_grid": {
            "state_spacing": 2.0 * cfg.state_bound / (cfg.state_levels - 1),
            "velocity_spacing": (
                2.0 * cfg.velocity_bound / (cfg.velocity_levels - 1)
            ),
            "rounding_rule": "nearest_grid_index_numpy_rint_ties_to_even",
            "endpoints_included": True,
            "zero_included": True,
        },
        "saved_history": {
            "snapshot_stride": cfg.snapshot_stride,
            "saved_frame_count": len(frame_cycles),
            "transition_state_count": cfg.cycles + 1,
            "all_transition_frames_saved": cfg.snapshot_stride == 1,
            "semantics": (
                "complete_transition_history"
                if cfg.snapshot_stride == 1
                else "sampled_transition_history_with_initial_and_final"
            ),
            "omitted_transition_count": cfg.cycles + 1 - len(frame_cycles),
        },
        "rng": "numpy.random.PCG64",
        "noise_locality": (
            "distinct per-node draws supplied by one exogenous global PCG64 stream"
        ),
        "numpy_version": np.__version__,
        "seed": cfg.seed,
        "noise_stream_sha256": "sha256:" + noise_hasher.hexdigest(),
        "initial_state_sha256": initial_state_hash,
        "initial_velocity_sha256": initial_velocity_hash,
        "config_sha256": _json_sha256(config_payload),
        "graph": graph_payload,
        "point_coordinates_drive_dynamics": False,
        "update_rule": {
            "laplacian": "sum_over_incident_edges(q_neighbor-q_patch)",
            "potential": "mass2(cycle)*q^2/2 + quartic*q^4/4",
            "integrator": "damped_stochastic_symplectic_euler_then_finite_grid_rounding",
            "quench": cfg.quench_kind,
            "quench_semantics": "exogenous_spatially_uniform_cycle_schedule",
            "record_commit": (
                "threshold_and_same_sign_persistence_then_irreversible_latch"
            ),
            "readback": "feedback_strength*committed*(record_target-q)",
            "defect": (
                "gradient_threshold_edge:abs(q_left-q_right)>=defect_threshold"
            ),
        },
        "causal_order": [
            "optional_intervention",
            "local_and_incident-edge_force",
            "previously_committed_record_readback",
            "velocity_write",
            "state_write",
            "record_commit",
            "snapshot",
        ],
        "record_readback_first_affects_next_transition": True,
        "readback_nonzero_force_count": readback_nonzero_force_count,
        "readback_quantized_velocity_write_count": readback_velocity_write_count,
        "readback_quantized_state_write_count": readback_state_write_count,
        "readback_quantized_any_write_count": readback_any_write_count,
        "readback_caused_later_quantized_write": readback_any_write_count > 0,
        # Backward-compatible name, now carrying the literal write count rather
        # than the former nonzero-force count.
        "readback_nonzero_write_count": readback_any_write_count,
        "readback_write_counter_semantics": (
            "one_step_same_state_same_velocity_same_noise_quantized_shadow_without_readback"
        ),
        "first_commit_cycle": first_commit_cycle.tolist(),
        "intervention": None if not events else asdict(events[0]),
        "intervention_count": len(events),
        "interventions": [
            {
                **asdict(event),
                "event_index": event_index,
                "affected_node_count": int(
                    np.count_nonzero(intervention_masks[event_index])
                ),
                "realized_state_delta_l1": float(
                    np.sum(np.abs(intervention_deltas[event_index, :, 0]))
                ),
                "realized_velocity_delta_l1": float(
                    np.sum(np.abs(intervention_deltas[event_index, :, 1]))
                ),
                "realized_affected_node_count": int(
                    np.count_nonzero(
                        np.any(intervention_deltas[event_index] != 0.0, axis=1)
                    )
                ),
                "state_clip_count": int(
                    intervention_state_clip_counts[event_index]
                ),
                "velocity_clip_count": int(
                    intervention_velocity_clip_counts[event_index]
                ),
                "effective_after_quantization": bool(
                    np.any(intervention_deltas[event_index] != 0.0)
                ),
            }
            for event_index, event in enumerate(events)
        ],
        "same_cycle_intervention_order": "declared_sequence_order",
        "numerical_checks": {
            "deterministic_stiffness_step_number": stability_number,
            "deterministic_stiffness_step_limit": cfg.stability_limit,
            "deterministic_stiffness_step_guard_pass": bool(
                stability_number <= cfg.stability_limit
            ),
            "deterministic_stiffness_step_guard_scope": (
                "smooth bounded force Jacobian only; excludes noise, threshold "
                "switching, interventions, initialization truncation, and clipping"
            ),
            # Compatibility aliases retained for existing consumers.
            "stability_number": stability_number,
            "stability_limit": cfg.stability_limit,
            "stability_bound_pass": bool(stability_number <= cfg.stability_limit),
            "damping_step": cfg.damping * cfg.dt,
            "finite_frames": True,
            "initial_state_clip_count": initial_state_clip_count,
            "initial_velocity_clip_count": initial_velocity_clip_count,
            "state_clip_count": state_clip_count,
            "velocity_clip_count": velocity_clip_count,
            "intervention_state_clip_count": int(
                np.sum(intervention_state_clip_counts)
            ),
            "intervention_velocity_clip_count": int(
                np.sum(intervention_velocity_clip_counts)
            ),
            "hard_overshoot_count": hard_overshoot_count,
            "clip_free_run": bool(
                state_clip_count == 0
                and velocity_clip_count == 0
                and initial_state_clip_count == 0
                and initial_velocity_clip_count == 0
                and not np.any(intervention_state_clip_counts)
                and not np.any(intervention_velocity_clip_counts)
            ),
            "numerical_acceptance_pass": bool(
                stability_number <= cfg.stability_limit
                and state_clip_count == 0
                and velocity_clip_count == 0
                and initial_state_clip_count == 0
                and initial_velocity_clip_count == 0
                and not np.any(intervention_state_clip_counts)
                and not np.any(intervention_velocity_clip_counts)
                and hard_overshoot_count == 0
            ),
        },
        "nonclaims": [
            "no physical clock or duration",
            "no physical distance, curvature, energy, or temperature",
            "no source-to-CMB or source-to-matter identification",
            "no derivation of this update law from the OPH axioms",
            "irreversible record growth is imposed by the latch rule, not emergent",
            "gradient-threshold edges are not proven topological defects",
            "the exogenous global quench is not emergent causal synchronization",
        ],
    }
    return CoupledPatchResult(
        points=pts,
        left=edge_left,
        right=edge_right,
        cycles=np.asarray(frame_cycles, dtype=np.int64),
        state_frames=np.stack(state_frames, axis=0),
        velocity_frames=np.stack(velocity_frames, axis=0),
        record_frames=np.stack(record_frames, axis=0),
        commit_frames=np.stack(commit_frames, axis=0),
        defect_frames=np.stack(defect_frames, axis=0),
        feedback_force_frames=np.stack(feedback_force_frames, axis=0),
        mass2_frames=np.asarray(mass2_frames, dtype=np.float64),
        intervention_mask=intervention_mask,
        intervention_delta=intervention_delta,
        intervention_cycles=np.asarray(
            [event.cycle for event in events], dtype=np.int64
        ),
        intervention_masks=intervention_masks,
        intervention_deltas=intervention_deltas,
        config=config_payload,
        provenance=provenance,
    )


def run_paired_counterfactual(
    points: np.ndarray | Sequence[Sequence[float]],
    left: np.ndarray | Sequence[int],
    right: np.ndarray | Sequence[int],
    config: CoupledPatchConfig | Mapping[str, Any] | None,
    intervention: (
        LocalizedIntervention
        | Mapping[str, Any]
        | Sequence[LocalizedIntervention | Mapping[str, Any]]
    ),
    *,
    initial_state: np.ndarray | Sequence[float] | None = None,
    initial_velocity: np.ndarray | Sequence[float] | None = None,
) -> PairedCoupledPatchResult:
    """Run an intervention and its exact same-seed/noise control."""

    cfg = _coerce_config(config)
    events = _coerce_interventions(intervention)
    if not events:
        raise ValueError("paired counterfactual requires an intervention")
    control = simulate_coupled_patch(
        points,
        left,
        right,
        cfg,
        initial_state=initial_state,
        initial_velocity=initial_velocity,
    )
    treated = simulate_coupled_patch(
        points,
        left,
        right,
        cfg,
        intervention=events,
        initial_state=initial_state,
        initial_velocity=initial_velocity,
    )
    if not np.array_equal(control.cycles, treated.cycles):
        raise RuntimeError("paired runs produced different snapshot cycles")
    same_noise = (
        control.provenance["noise_stream_sha256"]
        == treated.provenance["noise_stream_sha256"]
    )
    same_initial = bool(
        control.provenance["initial_state_sha256"]
        == treated.provenance["initial_state_sha256"]
        and control.provenance["initial_velocity_sha256"]
        == treated.provenance["initial_velocity_sha256"]
    )
    if not same_noise or not same_initial:
        raise RuntimeError("paired counterfactual failed its same-seed contract")
    realized_interventions = treated.provenance["interventions"]
    receipt = {
        "schema": "oph.coupled_patch_paired_counterfactual.v1",
        "classification": CLASSIFICATION,
        "same_seed": True,
        "same_initial_state": same_initial,
        "same_process_noise_draws": same_noise,
        "noise_stream_sha256": control.provenance["noise_stream_sha256"],
        "interventions": [asdict(event) for event in events],
        "requested_intervention_count": len(events),
        "realized_intervention_count": sum(
            bool(item["effective_after_quantization"])
            for item in realized_interventions
        ),
        "all_interventions_effective_after_quantization": all(
            bool(item["effective_after_quantization"])
            for item in realized_interventions
        ),
        "realized_interventions": realized_interventions,
        "physical_interpretation_allowed": False,
    }
    return PairedCoupledPatchResult(
        control=control,
        intervened=treated,
        state_delta_frames=treated.state_frames - control.state_frames,
        velocity_delta_frames=treated.velocity_frames - control.velocity_frames,
        record_delta_frames=treated.record_frames - control.record_frames,
        defect_xor_frames=np.logical_xor(
            treated.defect_frames, control.defect_frames
        ),
        receipt=receipt,
    )


def run_collision_counterfactual(
    points: np.ndarray | Sequence[Sequence[float]],
    left: np.ndarray | Sequence[int],
    right: np.ndarray | Sequence[int],
    config: CoupledPatchConfig | Mapping[str, Any] | None,
    intervention_a: (
        LocalizedIntervention
        | Mapping[str, Any]
        | Sequence[LocalizedIntervention | Mapping[str, Any]]
    ),
    intervention_b: (
        LocalizedIntervention
        | Mapping[str, Any]
        | Sequence[LocalizedIntervention | Mapping[str, Any]]
    ),
    *,
    initial_state: np.ndarray | Sequence[float] | None = None,
    initial_velocity: np.ndarray | Sequence[float] | None = None,
) -> CollisionCounterfactualResult:
    """Compute the exact finite-map nonlinear collision residual.

    The four arms share graph, configuration, seed-derived initial state, and
    every process-noise draw.  When events share a cycle, the combined arm
    applies declared A events before declared B events.
    """

    cfg = _coerce_config(config)
    events_a = _coerce_interventions(intervention_a)
    events_b = _coerce_interventions(intervention_b)
    if not events_a or not events_b:
        raise ValueError("collision counterfactual requires nonempty A and B events")
    common = {
        "initial_state": initial_state,
        "initial_velocity": initial_velocity,
    }
    baseline = simulate_coupled_patch(points, left, right, cfg, **common)
    arm_a = simulate_coupled_patch(
        points, left, right, cfg, intervention=events_a, **common
    )
    arm_b = simulate_coupled_patch(
        points, left, right, cfg, intervention=events_b, **common
    )
    arm_ab = simulate_coupled_patch(
        points, left, right, cfg, intervention=events_a + events_b, **common
    )
    runs = (baseline, arm_a, arm_b, arm_ab)
    aligned_cycles = all(
        np.array_equal(run.cycles, baseline.cycles) for run in runs[1:]
    )
    noise_hashes = {run.provenance["noise_stream_sha256"] for run in runs}
    initial_hashes = {
        (
            run.provenance["initial_state_sha256"],
            run.provenance["initial_velocity_sha256"],
        )
        for run in runs
    }
    if not aligned_cycles or len(noise_hashes) != 1 or len(initial_hashes) != 1:
        raise RuntimeError(
            "collision arms failed their exact common-random-number contract"
        )

    actuator_residual = (
        arm_ab.intervention_delta
        - arm_a.intervention_delta
        - arm_b.intervention_delta
    )
    actuator_residual_nonzero = np.any(actuator_residual != 0.0, axis=1)
    overlap_rows: list[dict[str, int]] = []
    for a_index, a_event in enumerate(events_a):
        for b_index, b_event in enumerate(events_b):
            if a_event.cycle != b_event.cycle:
                continue
            overlap_count = int(
                np.count_nonzero(
                    arm_a.intervention_masks[a_index]
                    & arm_b.intervention_masks[b_index]
                )
            )
            if overlap_count:
                overlap_rows.append(
                    {
                        "cycle": int(a_event.cycle),
                        "a_event_index": a_index,
                        "b_event_index": b_index,
                        "overlap_node_count": overlap_count,
                    }
                )

    def intervention_clip_count(run: CoupledPatchResult) -> int:
        return int(
            sum(
                int(item["state_clip_count"]) + int(item["velocity_clip_count"])
                for item in run.provenance["interventions"]
            )
        )

    actuator_clip_counts = {
        "a": intervention_clip_count(arm_a),
        "b": intervention_clip_count(arm_b),
        "ab": intervention_clip_count(arm_ab),
    }
    aggregate_actuator_additive = bool(not np.any(actuator_residual_nonzero))
    actuator_saturation_detected = any(
        value > 0 for value in actuator_clip_counts.values()
    )

    state_residual = (
        arm_ab.state_frames
        - arm_a.state_frames
        - arm_b.state_frames
        + baseline.state_frames
    )
    velocity_residual = (
        arm_ab.velocity_frames
        - arm_a.velocity_frames
        - arm_b.velocity_frames
        + baseline.velocity_frames
    )
    record_residual = (
        arm_ab.record_frames
        - arm_a.record_frames
        - arm_b.record_frames
        + baseline.record_frames
    )
    defect_residual = (
        arm_ab.defect_frames.astype(np.int8)
        - arm_a.defect_frames.astype(np.int8)
        - arm_b.defect_frames.astype(np.int8)
        + baseline.defect_frames.astype(np.int8)
    )
    receipt = {
        "schema": "oph.coupled_patch_collision_counterfactual.v1",
        "classification": CLASSIFICATION,
        "same_seed": True,
        "same_initial_state": True,
        "same_process_noise_draws": True,
        "aligned_snapshot_cycles": aligned_cycles,
        "residual_formula": "AB-A-B+baseline",
        "combined_same_cycle_order": "A_events_then_B_events",
        "a_interventions": [asdict(event) for event in events_a],
        "b_interventions": [asdict(event) for event in events_b],
        "noise_stream_sha256": baseline.provenance["noise_stream_sha256"],
        "actuator_diagnostics": {
            "aggregate_realized_kick_formula": "AB-A-B",
            "aggregate_realized_kick_additive": aggregate_actuator_additive,
            "aggregate_residual_nonzero_node_count": int(
                np.count_nonzero(actuator_residual_nonzero)
            ),
            "aggregate_residual_state_l1": float(
                np.sum(np.abs(actuator_residual[:, 0]))
            ),
            "aggregate_residual_velocity_l1": float(
                np.sum(np.abs(actuator_residual[:, 1]))
            ),
            "same_cycle_requested_support_overlap": bool(overlap_rows),
            "same_cycle_overlap_rows": overlap_rows,
            "intervention_clip_counts": actuator_clip_counts,
            "saturation_detected": actuator_saturation_detected,
            "clean_dynamical_residual_interpretation": bool(
                aggregate_actuator_additive and not actuator_saturation_detected
            ),
            "interpretation": (
                "A nonadditive or clipped actuator can itself contribute to "
                "AB-A-B+baseline; only a clean actuator receipt isolates later-map "
                "nonlinearity within this internal model."
            ),
        },
        "physical_interpretation_allowed": False,
        "nonclaim": (
            "a nonzero residual is not a physical scattering amplitude or "
            "cross section"
        ),
    }
    return CollisionCounterfactualResult(
        baseline=baseline,
        a=arm_a,
        b=arm_b,
        ab=arm_ab,
        state_nonlinear_residual_frames=state_residual,
        velocity_nonlinear_residual_frames=velocity_residual,
        record_nonlinear_residual_frames=record_residual,
        defect_nonlinear_residual_frames=defect_residual,
        receipt=receipt,
    )


def write_coupled_patch_run(result: CoupledPatchResult, out_dir: Path) -> dict[str, Any]:
    """Write saved frames losslessly plus a fail-closed run manifest."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame_path = out / FRAME_ARTIFACT
    np.savez_compressed(
        frame_path,
        points=result.points,
        left=result.left,
        right=result.right,
        cycles=result.cycles,
        state_frames=result.state_frames,
        velocity_frames=result.velocity_frames,
        record_frames=result.record_frames,
        commit_frames=result.commit_frames,
        defect_frames=result.defect_frames,
        feedback_force_frames=result.feedback_force_frames,
        mass2_frames=result.mass2_frames,
        intervention_mask=result.intervention_mask,
        intervention_delta=result.intervention_delta,
        intervention_cycles=result.intervention_cycles,
        intervention_masks=result.intervention_masks,
        intervention_deltas=result.intervention_deltas,
    )
    arrays = {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in _result_arrays(result).items()
    }
    manifest = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "physical_interpretation_allowed": False,
        "files": {
            FRAME_ARTIFACT: {
                "sha256": _file_sha256(frame_path),
                "bytes": frame_path.stat().st_size,
            },
            README_ARTIFACT: {
                "role": "array contract and interpretation boundary"
            },
        },
        "arrays": arrays,
        "config": dict(result.config),
        "provenance": dict(result.provenance),
    }
    (out / MANIFEST_ARTIFACT).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out / README_ARTIFACT).write_text(_run_readme(result), encoding="utf-8")
    return manifest


def _run_readme(result: CoupledPatchResult) -> str:
    frames, nodes = result.state_frames.shape
    edges = result.left.size
    stride = int(result.config["snapshot_stride"])
    history_sentence = (
        "Every transition state is saved."
        if stride == 1
        else (
            f"Frames are sampled every {stride} transitions, with the initial "
            "and final states retained; omitted transition states are not present."
        )
    )
    return f"""# Coupled patch internal-diagnostic run

Classification: **{CLASSIFICATION}**. Physical interpretation is disabled.

This directory contains a target-blind finite-state experiment whose update is
graph-local conditional on an exogenous global quench schedule and per-node
noise inputs. The global update cycle is not an autonomous local clock.
It is suitable for internal correlation, morphology, defect, response-front,
and observer-record diagnostics. Update cycles, graph distances, state,
velocity, and potential parameters use nonphysical dimensionless units. No
array is identified with CMB temperature, matter density, laboratory energy,
length, or time.

## Files

- `{FRAME_ARTIFACT}`: losslessly compressed values for the saved NumPy frames.
- `{MANIFEST_ARTIFACT}`: configuration, provenance, hashes, shapes, numerical
  checks, causal ordering, and nonclaims.
- `{README_ARTIFACT}`: this interpretation and replay guide.

## Array contract

There are {frames} saved cycles, {nodes} patches, and {edges} unoriented
edges. {history_sentence} `cycles` indexes every `*_frames` array. Node-frame arrays are
`state_frames`, `velocity_frames`, `record_frames`, `commit_frames`, and
`feedback_force_frames`. `defect_frames` uses the edge order in `left` and
`right`; a reported defect is exactly the declared gradient-threshold test on
endpoint state difference and is not thereby a topological defect.
`mass2_frames[k]` records the quench value used by the incoming transition to
saved frame `k` (with the initial entry recording the initial schedule value).
Committed records are irreversibly latched by declaration, so monotone record
growth is not an emergent arrow. `intervention_mask` is
the per-node union and `intervention_delta[:,0:2]` is the aggregate exact
realized state and velocity kick. `intervention_cycles`,
`intervention_masks`, and `intervention_deltas` retain every impulse
separately. The manifest is the authoritative interpretation boundary.

## Replay

Re-run `simulate_coupled_patch` with the manifest configuration, graph arrays,
and any explicitly supplied initial arrays. Exact paired controls should have
identical initial-state and process-noise hashes. Randomness uses NumPy PCG64
with three seed-derived streams for initial state, initial velocity, and
process noise.

Finite coordinates use the spacings recorded in the manifest. Quantization is
to the nearest grid index with NumPy `rint`, including its ties-to-even rule.
"""


def _coerce_config(
    config: CoupledPatchConfig | Mapping[str, Any] | None,
) -> CoupledPatchConfig:
    if config is None:
        return CoupledPatchConfig()
    if isinstance(config, CoupledPatchConfig):
        return config
    if isinstance(config, Mapping):
        return CoupledPatchConfig(**dict(config))
    raise TypeError("config must be CoupledPatchConfig, a mapping, or None")


def _coerce_interventions(
    intervention: (
        LocalizedIntervention
        | Mapping[str, Any]
        | Sequence[LocalizedIntervention | Mapping[str, Any]]
        | None
    ),
) -> tuple[LocalizedIntervention, ...]:
    if intervention is None:
        return ()
    if isinstance(intervention, LocalizedIntervention):
        return (intervention,)
    if isinstance(intervention, Mapping):
        return (LocalizedIntervention(**dict(intervention)),)
    if isinstance(intervention, Sequence) and not isinstance(
        intervention, (str, bytes, bytearray)
    ):
        events: list[LocalizedIntervention] = []
        for item in intervention:
            if isinstance(item, LocalizedIntervention):
                events.append(item)
            elif isinstance(item, Mapping):
                events.append(LocalizedIntervention(**dict(item)))
            else:
                raise TypeError(
                    "each intervention must be LocalizedIntervention or a mapping"
                )
        return tuple(events)
    raise TypeError(
        "intervention must be one intervention, a sequence of interventions, or None"
    )


def _validate_graph(
    points: np.ndarray | Sequence[Sequence[float]],
    left: np.ndarray | Sequence[int],
    right: np.ndarray | Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 1:
        raise ValueError("points must have shape (node_count, coordinate_dimension)")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points contain non-finite coordinates")
    edge_left = _integral_edge_array(left, "left")
    edge_right = _integral_edge_array(right, "right")
    if edge_left.size == 0 or edge_left.shape != edge_right.shape:
        raise ValueError("left/right must describe the same nonempty edge list")
    node_count = int(pts.shape[0])
    if (
        np.any(edge_left < 0)
        or np.any(edge_right < 0)
        or np.any(edge_left >= node_count)
        or np.any(edge_right >= node_count)
    ):
        raise ValueError("edge endpoint outside point array")
    if np.any(edge_left == edge_right):
        raise ValueError("self loops are not supported")
    low = np.minimum(edge_left, edge_right)
    high = np.maximum(edge_left, edge_right)
    order = np.lexsort((high, low))
    low = low[order]
    high = high[order]
    if low.size > 1 and np.any((low[1:] == low[:-1]) & (high[1:] == high[:-1])):
        raise ValueError("duplicate unoriented edge")
    degree = np.bincount(
        np.concatenate((low, high)), minlength=node_count
    ).astype(np.int64)
    if np.any(degree == 0):
        raise ValueError("every patch must have at least one incident edge")
    return pts.copy(), low, high, degree


def _validate_config(
    config: CoupledPatchConfig, *, node_count: int, max_degree: int
) -> None:
    integer_positive = {
        "cycles": config.cycles,
        "state_levels": config.state_levels,
        "velocity_levels": config.velocity_levels,
        "record_persistence": config.record_persistence,
        "snapshot_stride": config.snapshot_stride,
    }
    for name, value in integer_positive.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.state_levels < 3 or config.velocity_levels < 3:
        raise ValueError("finite coordinate grids need at least three levels")
    if config.state_levels % 2 == 0 or config.velocity_levels % 2 == 0:
        raise ValueError("coordinate grid levels must be odd so zero is represented")
    if isinstance(config.seed, bool) or not isinstance(config.seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if int(config.seed) < 0:
        raise ValueError("seed must be nonnegative")
    finite_values = {
        name: float(value)
        for name, value in asdict(config).items()
        if name not in integer_positive and name not in {"seed", "quench_kind"}
    }
    if not all(math.isfinite(value) for value in finite_values.values()):
        raise ValueError("all floating configuration values must be finite")
    if config.dt <= 0 or config.state_bound <= 0 or config.velocity_bound <= 0:
        raise ValueError("dt and coordinate bounds must be positive")
    if config.initial_state_scale < 0 or config.initial_velocity_scale < 0:
        raise ValueError("initialization scales must be nonnegative")
    if config.coupling < 0 or config.quartic < 0:
        raise ValueError("coupling and quartic must be nonnegative")
    if config.damping < 0 or config.noise_amplitude < 0:
        raise ValueError("damping and noise amplitude must be nonnegative")
    if not (0 <= config.quench_start_fraction <= config.quench_end_fraction <= 1):
        raise ValueError("quench fractions must obey 0 <= start <= end <= 1")
    if config.quench_kind not in {"linear", "smoothstep"}:
        raise ValueError("quench_kind must be 'linear' or 'smoothstep'")
    if not (0 < config.record_threshold <= config.state_bound):
        raise ValueError("record threshold must lie inside the state bound")
    if not (0 < config.record_amplitude <= config.state_bound):
        raise ValueError("record amplitude must lie inside the state bound")
    if config.feedback_strength < 0:
        raise ValueError("feedback strength must be nonnegative")
    if not (0 < config.defect_threshold <= 2.0 * config.state_bound):
        raise ValueError("defect threshold must be a possible endpoint difference")
    if not (0 < config.stability_limit <= 1):
        raise ValueError("stability_limit must lie in (0, 1]")
    if config.damping * config.dt > 1.0:
        raise ValueError("damping*dt exceeds the monotone explicit damping bound")
    stability_number = _stability_number(config, max_degree)
    if stability_number > config.stability_limit:
        raise ValueError(
            "declared timestep fails the deterministic stiffness-step guard "
            "(legacy stability bound): "
            f"{stability_number:.6g} > {config.stability_limit:.6g}"
        )
    if node_count < 2:
        raise ValueError("coupled dynamics needs at least two patches")


def _validate_intervention(
    event: LocalizedIntervention, config: CoupledPatchConfig, node_count: int
) -> None:
    if isinstance(event.center_node, (bool, np.bool_)) or not isinstance(
        event.center_node, (int, np.integer)
    ):
        raise ValueError("intervention center_node must be an integer")
    if isinstance(event.cycle, (bool, np.bool_)) or not isinstance(
        event.cycle, (int, np.integer)
    ):
        raise ValueError("intervention cycle must be an integer")
    if not (0 <= event.center_node < node_count):
        raise ValueError("intervention center outside graph")
    if not (0 <= event.cycle < config.cycles):
        raise ValueError("intervention cycle outside update range")
    if isinstance(event.radius_hops, bool) or not isinstance(
        event.radius_hops, (int, np.integer)
    ) or event.radius_hops < 0:
        raise ValueError("intervention radius_hops must be a nonnegative integer")
    for name, value in (
        ("state_delta", event.state_delta),
        ("velocity_delta", event.velocity_delta),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"intervention {name} must be a finite real number")
        if not math.isfinite(float(value)):
            raise ValueError(f"intervention {name} must be a finite real number")
    if abs(event.state_delta) > 2.0 * config.state_bound:
        raise ValueError("state intervention exceeds the full bounded range")
    if abs(event.velocity_delta) > 2.0 * config.velocity_bound:
        raise ValueError("velocity intervention exceeds the full bounded range")
    if event.state_delta == 0 and event.velocity_delta == 0:
        raise ValueError("intervention must change state or velocity")


def _integral_edge_array(
    values: np.ndarray | Sequence[int], name: str
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} edge endpoints must be one-dimensional")
    items = raw.tolist()
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in items
    ):
        raise ValueError(f"{name} edge endpoints must be integer typed")
    lower = int(np.iinfo(np.int64).min)
    upper = int(np.iinfo(np.int64).max)
    if any(not lower <= int(value) <= upper for value in items):
        raise ValueError(f"{name} edge endpoint exceeds int64 range")
    return np.asarray([int(value) for value in items], dtype=np.int64)


def _initial_coordinate(
    supplied: np.ndarray | Sequence[float] | None,
    *,
    node_count: int,
    bound: float,
    levels: int,
    scale: float,
    rng: np.random.Generator,
    label: str,
) -> tuple[np.ndarray, int]:
    if supplied is None:
        raw = rng.normal(0.0, scale, size=node_count)
        clip_count = int(np.count_nonzero(np.abs(raw) > bound))
        return _quantize(np.clip(raw, -bound, bound), bound, levels), clip_count
    supplied_array = np.asarray(supplied)
    if supplied_array.ndim != 1 or supplied_array.shape != (node_count,):
        raise ValueError(f"{label} must have shape ({node_count},)")
    if np.issubdtype(supplied_array.dtype, np.bool_) or (
        supplied_array.dtype == object
        and any(
            isinstance(value, (bool, np.bool_))
            for value in supplied_array.tolist()
        )
    ):
        raise ValueError(f"{label} must contain non-boolean real values")
    if any(
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
        for value in supplied_array.tolist()
    ):
        raise ValueError(f"{label} must contain real values")
    try:
        raw = np.asarray(supplied, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain real values") from exc
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{label} contains non-finite values")
    if np.any(np.abs(raw) > bound):
        raise ValueError(f"{label} exceeds its declared bound")
    return _quantize(raw, bound, levels), 0


def _nearest_neighbor_laplacian(
    state: np.ndarray, left: np.ndarray, right: np.ndarray, node_count: int
) -> np.ndarray:
    laplacian = np.zeros(node_count, dtype=np.float64)
    difference = state[right] - state[left]
    np.add.at(laplacian, left, difference)
    np.add.at(laplacian, right, -difference)
    return laplacian


def _edge_defects(
    state: np.ndarray, left: np.ndarray, right: np.ndarray, threshold: float
) -> np.ndarray:
    return np.abs(state[left] - state[right]) >= threshold


def _mass2_at(config: CoupledPatchConfig, step: int) -> float:
    if config.cycles <= 1:
        fraction = 1.0
    else:
        fraction = min(1.0, max(0.0, step / (config.cycles - 1)))
    start = config.quench_start_fraction
    end = config.quench_end_fraction
    if end == start:
        progress = float(fraction >= end)
    else:
        progress = min(1.0, max(0.0, (fraction - start) / (end - start)))
    if config.quench_kind == "smoothstep":
        progress = progress * progress * (3.0 - 2.0 * progress)
    return config.mass2_start + progress * (config.mass2_end - config.mass2_start)


def _quantize(values: np.ndarray, bound: float, levels: int) -> np.ndarray:
    scale = (levels - 1) / (2.0 * bound)
    indices = np.rint((np.asarray(values, dtype=np.float64) + bound) * scale)
    indices = np.clip(indices, 0, levels - 1)
    return indices / scale - bound


def _graph_ball_mask(
    node_count: int,
    left: np.ndarray,
    right: np.ndarray,
    *,
    center: int,
    radius: int,
) -> np.ndarray:
    neighbors: list[list[int]] = [[] for _ in range(node_count)]
    for a, b in zip(left.tolist(), right.tolist(), strict=True):
        neighbors[a].append(b)
        neighbors[b].append(a)
    mask = np.zeros(node_count, dtype=bool)
    mask[center] = True
    frontier = {center}
    for _ in range(radius):
        frontier = {
            neighbor
            for node in frontier
            for neighbor in neighbors[node]
            if not mask[neighbor]
        }
        if not frontier:
            break
        mask[list(frontier)] = True
    return mask


def _snapshot_cycles(cycles: int, stride: int) -> tuple[int, ...]:
    values = list(range(0, cycles + 1, stride))
    if values[-1] != cycles:
        values.append(cycles)
    return tuple(values)


def _stability_number(config: CoupledPatchConfig, max_degree: int) -> float:
    stiffness_bound = (
        max(abs(config.mass2_start), abs(config.mass2_end))
        + 3.0 * config.quartic * config.state_bound**2
        + 2.0 * config.coupling * max_degree
        + config.feedback_strength
    )
    return config.dt**2 * stiffness_bound


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = f"{contiguous.dtype.str}|{contiguous.shape}|".encode("ascii")
    return "sha256:" + hashlib.sha256(header + contiguous.tobytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return "sha256:" + hasher.hexdigest()


def _result_arrays(result: CoupledPatchResult) -> dict[str, np.ndarray]:
    return {
        "points": result.points,
        "left": result.left,
        "right": result.right,
        "cycles": result.cycles,
        "state_frames": result.state_frames,
        "velocity_frames": result.velocity_frames,
        "record_frames": result.record_frames,
        "commit_frames": result.commit_frames,
        "defect_frames": result.defect_frames,
        "feedback_force_frames": result.feedback_force_frames,
        "mass2_frames": result.mass2_frames,
        "intervention_mask": result.intervention_mask,
        "intervention_delta": result.intervention_delta,
        "intervention_cycles": result.intervention_cycles,
        "intervention_masks": result.intervention_masks,
        "intervention_deltas": result.intervention_deltas,
    }


__all__ = [
    "CLASSIFICATION",
    "CollisionCounterfactualResult",
    "CoupledPatchConfig",
    "CoupledPatchResult",
    "LocalizedIntervention",
    "PairedCoupledPatchResult",
    "run_collision_counterfactual",
    "run_paired_counterfactual",
    "simulate_coupled_patch",
    "write_coupled_patch_run",
]
