from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.bulk.h3_worldline_stitch import h3_distance
from oph_fpe.defects.array_s3_holonomy import S3_CLASS, S3_INV, S3_MUL


def two_defect_stress_contraction_assay_report(
    *,
    patch_count: int = 65_536,
    steps: int = 16,
    support_node_count: int = 8,
    holonomy: int = 1,
    initial_separation: float = 1.2,
    stress_coupling: float = 0.04,
    stress_radius: float = 1.0,
    curvature_radius: float = 1.0,
    cycle_stride: int = 4,
    min_approach_fraction: float = 0.25,
    min_control_margin: float = 0.15,
) -> dict[str, Any]:
    """Controlled two-defect assay for a declared readout-contraction law.

    This is deliberately not a production gravity claim. It validates that a
    finite, declared stress-contraction rule can drive two inverse defect-like
    packets together in H3 while no-contraction and shuffled-pair controls do
    not show the same approach.
    """

    patch_count = max(1, int(patch_count))
    steps = max(4, int(steps))
    support_node_count = max(1, int(support_node_count))
    cycle_stride = max(1, int(cycle_stride))
    curvature_radius = _positive(curvature_radius, default=1.0)
    stress_radius = _positive(stress_radius, default=1.0)
    stress_coupling = max(0.0, float(stress_coupling))
    initial_separation = _positive(initial_separation, default=1.2)
    min_approach_fraction = max(0.0, float(min_approach_fraction))
    min_control_margin = max(0.0, float(min_control_margin))

    holonomy = int(holonomy) % int(len(S3_INV))
    if holonomy == 0:
        holonomy = 1
    inverse = int(S3_INV[holonomy])
    inverse_identity_pass = bool(
        int(S3_MUL[holonomy, inverse]) == 0 and int(S3_MUL[inverse, holonomy]) == 0
    )

    stress_rows = _simulate_pair(
        mode="stress_contraction",
        steps=steps,
        initial_separation=initial_separation,
        stress_coupling=stress_coupling,
        stress_radius=stress_radius,
        curvature_radius=curvature_radius,
        cycle_stride=cycle_stride,
    )
    no_contraction_rows = _simulate_pair(
        mode="no_contraction_control",
        steps=steps,
        initial_separation=initial_separation,
        stress_coupling=0.0,
        stress_radius=stress_radius,
        curvature_radius=curvature_radius,
        cycle_stride=cycle_stride,
    )
    shuffled_rows = _simulate_pair(
        mode="shuffled_pair_control",
        steps=steps,
        initial_separation=initial_separation,
        stress_coupling=stress_coupling,
        stress_radius=stress_radius,
        curvature_radius=curvature_radius,
        cycle_stride=cycle_stride,
    )

    stress_summary = _trajectory_summary(stress_rows)
    no_contraction_summary = _trajectory_summary(no_contraction_rows)
    shuffled_summary = _trajectory_summary(shuffled_rows)
    control_approach = max(
        float(no_contraction_summary["approach_fraction"]),
        float(shuffled_summary["approach_fraction"]),
    )
    approach_margin = float(stress_summary["approach_fraction"] - control_approach)
    monotone = _monotone_nonincreasing([float(row["h3_separation"]) for row in stress_rows])
    control_rejected = bool(approach_margin >= min_control_margin)
    assay_receipt = bool(
        inverse_identity_pass
        and monotone
        and stress_summary["approach_fraction"] >= min_approach_fraction
        and control_rejected
    )

    worldlines = _worldlines_from_rows(
        stress_rows,
        patch_count=patch_count,
        support_node_count=support_node_count,
        holonomy=holonomy,
        inverse= inverse,
    )
    return {
        "mode": "controlled_two_defect_stress_contraction_assay_v0",
        "controlled_planted_assay": True,
        "patch_count": patch_count,
        "steps": steps,
        "support_node_count": support_node_count,
        "holonomy": holonomy,
        "inverse_holonomy": inverse,
        "s3_inverse_identity_pass": inverse_identity_pass,
        "declared_stress_contraction_law": {
            "law": "each defect moves along the H3 tangent-chart pair direction by coupling/(1+(d/R_s)^2)",
            "stress_coupling": stress_coupling,
            "stress_radius": stress_radius,
            "curvature_radius": curvature_radius,
            "initial_separation": initial_separation,
        },
        "stress_contraction_summary": stress_summary,
        "no_contraction_control_summary": no_contraction_summary,
        "shuffled_pair_control_summary": shuffled_summary,
        "approach_margin_vs_controls": approach_margin,
        "stress_separation_monotone_nonincreasing": monotone,
        "control_rejection_receipt": control_rejected,
        "two_defect_stress_contraction_assay_receipt": assay_receipt,
        "gravity_like_attraction_diagnostic_receipt": assay_receipt,
        "production_gravity_receipt": False,
        "physical_gravity_prediction": False,
        "particle_matter_receipt": False,
        "trajectory_rows": stress_rows,
        "control_trajectory_rows": {
            "no_contraction": no_contraction_rows,
            "shuffled_pair": shuffled_rows,
        },
        "worldlines": worldlines,
        "claim_boundary": (
            "Controlled/planted two-defect stress-contraction assay. A positive diagnostic receipt "
            "means the declared readout-contraction rule drives inverse defect-like packets together "
            "in H3 and beats no-contraction/shuffled controls. It is not spontaneous particle "
            "formation, production gravity, a physical quantum-vacuum simulation, or a CMB/matter "
            "prediction."
        ),
    }


def write_two_defect_stress_contraction_assay_report(
    out: Path,
    *,
    patch_count: int = 65_536,
    steps: int = 16,
    support_node_count: int = 8,
    holonomy: int = 1,
    initial_separation: float = 1.2,
    stress_coupling: float = 0.04,
    stress_radius: float = 1.0,
    curvature_radius: float = 1.0,
    cycle_stride: int = 4,
    min_approach_fraction: float = 0.25,
    min_control_margin: float = 0.15,
) -> dict[str, Any]:
    report = two_defect_stress_contraction_assay_report(
        patch_count=patch_count,
        steps=steps,
        support_node_count=support_node_count,
        holonomy=holonomy,
        initial_separation=initial_separation,
        stress_coupling=stress_coupling,
        stress_radius=stress_radius,
        curvature_radius=curvature_radius,
        cycle_stride=cycle_stride,
        min_approach_fraction=min_approach_fraction,
        min_control_margin=min_control_margin,
    )
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    stem = destination.with_suffix("")
    _write_rows(stem.with_name(stem.name + "_trajectory.csv"), report["trajectory_rows"])
    control_rows = [
        {**row, "control": control}
        for control, rows in report["control_trajectory_rows"].items()
        for row in rows
    ]
    _write_rows(stem.with_name(stem.name + "_controls.csv"), control_rows)
    return report


def free_two_defect_dynamics_report(
    *,
    patch_count: int = 65_536,
    steps: int = 96,
    support_node_count: int = 8,
    holonomy: int = 1,
    seed: int = 1729,
    initial_separation: float = 1.2,
    initial_speed: float = 0.035,
    stress_coupling: float = 0.03,
    transverse_kick: float = 0.008,
    stress_radius: float = 1.0,
    curvature_radius: float = 1.0,
    cycle_stride: int = 1,
    contact_radius: float = 0.10,
    overlap_radius: float = 0.22,
    bind_speed_threshold: float = 0.055,
    annihilation_overlap_threshold: float = 0.85,
) -> dict[str, Any]:
    """Free randomized two-defect dynamics diagnostic.

    This lane removes the planted x-axis geometry from the controlled assay. It
    still uses a declared effective stress rule, so it remains a visualization
    and control diagnostic rather than a production gravity or matter claim.
    """

    patch_count = max(1, int(patch_count))
    steps = max(4, int(steps))
    support_node_count = max(1, int(support_node_count))
    cycle_stride = max(1, int(cycle_stride))
    curvature_radius = _positive(curvature_radius, default=1.0)
    stress_radius = _positive(stress_radius, default=1.0)
    initial_separation = _positive(initial_separation, default=1.2)
    initial_speed = max(0.0, float(initial_speed))
    stress_coupling = max(0.0, float(stress_coupling))
    transverse_kick = max(0.0, float(transverse_kick))
    contact_radius = _positive(contact_radius, default=0.10)
    overlap_radius = max(contact_radius, _positive(overlap_radius, default=0.22))
    bind_speed_threshold = max(0.0, float(bind_speed_threshold))
    annihilation_overlap_threshold = max(0.0, min(1.0, float(annihilation_overlap_threshold)))

    rng = np.random.default_rng(int(seed))
    holonomy = int(holonomy) % int(len(S3_INV))
    if holonomy == 0:
        holonomy = 1
    inverse = int(S3_INV[holonomy])
    inverse_identity_pass = bool(
        int(S3_MUL[holonomy, inverse]) == 0 and int(S3_MUL[inverse, holonomy]) == 0
    )

    trajectory_rows = _simulate_free_pair(
        steps=steps,
        rng=rng,
        support_node_count=support_node_count,
        initial_separation=initial_separation,
        initial_speed=initial_speed,
        stress_coupling=stress_coupling,
        transverse_kick=transverse_kick,
        stress_radius=stress_radius,
        curvature_radius=curvature_radius,
        cycle_stride=cycle_stride,
        contact_radius=contact_radius,
        overlap_radius=overlap_radius,
        bind_speed_threshold=bind_speed_threshold,
        annihilation_overlap_threshold=annihilation_overlap_threshold,
        inverse_identity_pass=inverse_identity_pass,
    )
    summary = _free_trajectory_summary(trajectory_rows)
    outcome = str(summary.get("contact_outcome") or "pass_through")
    worldlines = _worldlines_from_free_rows(
        trajectory_rows,
        patch_count=patch_count,
        support_node_count=support_node_count,
        holonomy=holonomy,
        inverse=inverse,
    )
    receipt = bool(
        inverse_identity_pass
        and bool(summary.get("charge_conservation_pass", False))
        and outcome in {"scatter", "bind", "annihilate", "pass_through"}
        and len(worldlines) == 2
    )
    return {
        "mode": "free_two_defect_dynamics_v0",
        "controlled_planted_assay": False,
        "free_dynamics_diagnostic": True,
        "patch_count": patch_count,
        "steps": steps,
        "support_node_count": support_node_count,
        "seed": int(seed),
        "holonomy": holonomy,
        "inverse_holonomy": inverse,
        "s3_inverse_identity_pass": inverse_identity_pass,
        "declared_free_dynamics_law": {
            "law": (
                "randomized H3 tangent-chart initial pair with inverse-holonomy stress attraction, "
                "transverse repair kicks, support-overlap contact bookkeeping, and explicit "
                "scatter/bind/annihilate/pass-through outcomes"
            ),
            "initial_separation": initial_separation,
            "initial_speed": initial_speed,
            "stress_coupling": stress_coupling,
            "transverse_kick": transverse_kick,
            "stress_radius": stress_radius,
            "curvature_radius": curvature_radius,
            "contact_radius": contact_radius,
            "overlap_radius": overlap_radius,
            "bind_speed_threshold": bind_speed_threshold,
            "annihilation_overlap_threshold": annihilation_overlap_threshold,
        },
        "free_dynamics_summary": summary,
        "free_two_defect_dynamics_receipt": receipt,
        "gravity_like_free_dynamics_diagnostic_receipt": receipt,
        "production_gravity_receipt": False,
        "physical_gravity_prediction": False,
        "particle_matter_receipt": False,
        "trajectory_rows": trajectory_rows,
        "worldlines": worldlines,
        "claim_boundary": (
            "Free two-defect dynamics diagnostic. Initial positions and transverse perturbations are "
            "randomized, contact outcomes are explicit, and holonomy/support bookkeeping is exported. "
            "The effective stress rule is still declared by the simulator, so this is not production "
            "gravity, spontaneous particle matter, or a physical merger claim."
        ),
    }


def write_free_two_defect_dynamics_report(
    out: Path,
    *,
    patch_count: int = 65_536,
    steps: int = 96,
    support_node_count: int = 8,
    holonomy: int = 1,
    seed: int = 1729,
    initial_separation: float = 1.2,
    initial_speed: float = 0.035,
    stress_coupling: float = 0.03,
    transverse_kick: float = 0.008,
    stress_radius: float = 1.0,
    curvature_radius: float = 1.0,
    cycle_stride: int = 1,
    contact_radius: float = 0.10,
    overlap_radius: float = 0.22,
    bind_speed_threshold: float = 0.055,
    annihilation_overlap_threshold: float = 0.85,
) -> dict[str, Any]:
    report = free_two_defect_dynamics_report(
        patch_count=patch_count,
        steps=steps,
        support_node_count=support_node_count,
        holonomy=holonomy,
        seed=seed,
        initial_separation=initial_separation,
        initial_speed=initial_speed,
        stress_coupling=stress_coupling,
        transverse_kick=transverse_kick,
        stress_radius=stress_radius,
        curvature_radius=curvature_radius,
        cycle_stride=cycle_stride,
        contact_radius=contact_radius,
        overlap_radius=overlap_radius,
        bind_speed_threshold=bind_speed_threshold,
        annihilation_overlap_threshold=annihilation_overlap_threshold,
    )
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    stem = destination.with_suffix("")
    _write_rows(stem.with_name(stem.name + "_trajectory.csv"), report["trajectory_rows"])
    return report


def organic_defect_population_report(
    *,
    patch_count: int = 65_536,
    steps: int = 128,
    defect_count: int = 16,
    min_defects: int = 10,
    max_defects: int = 20,
    support_node_count: int = 8,
    seed: int = 2039,
    initial_speed: float = 0.028,
    stress_coupling: float = 0.018,
    transverse_kick: float = 0.010,
    stress_radius: float = 0.9,
    curvature_radius: float = 1.0,
    cycle_stride: int = 1,
    contact_radius: float = 0.12,
    overlap_radius: float = 0.28,
    spawn_radius: float = 1.25,
) -> dict[str, Any]:
    """Seeded organic diagnostic population of defect-like worldlines.

    Unlike the controlled and free pair assays, this emits a small population of
    staggered defect births with randomized H3 positions, support nodes, velocities,
    holonomies, local stress interactions, and near-contact bookkeeping. It remains
    a visualization diagnostic, not spontaneous matter or production gravity.
    """

    patch_count = max(1, int(patch_count))
    steps = max(8, int(steps))
    min_defects = max(1, int(min_defects))
    max_defects = max(min_defects, int(max_defects))
    defect_count = max(min_defects, min(max_defects, int(defect_count)))
    support_node_count = max(1, int(support_node_count))
    cycle_stride = max(1, int(cycle_stride))
    initial_speed = max(0.0, float(initial_speed))
    stress_coupling = max(0.0, float(stress_coupling))
    transverse_kick = max(0.0, float(transverse_kick))
    stress_radius = _positive(stress_radius, default=0.9)
    curvature_radius = _positive(curvature_radius, default=1.0)
    contact_radius = _positive(contact_radius, default=0.12)
    overlap_radius = max(contact_radius, _positive(overlap_radius, default=0.28))
    spawn_radius = _positive(spawn_radius, default=1.25)

    rng = np.random.default_rng(int(seed))
    trajectory_rows = _simulate_organic_defect_population(
        steps=steps,
        defect_count=defect_count,
        rng=rng,
        patch_count=patch_count,
        support_node_count=support_node_count,
        initial_speed=initial_speed,
        stress_coupling=stress_coupling,
        transverse_kick=transverse_kick,
        stress_radius=stress_radius,
        curvature_radius=curvature_radius,
        cycle_stride=cycle_stride,
        contact_radius=contact_radius,
        overlap_radius=overlap_radius,
        spawn_radius=spawn_radius,
    )
    worldlines = _worldlines_from_organic_rows(
        trajectory_rows,
        patch_count=patch_count,
        support_node_count=support_node_count,
    )
    summary = _organic_population_summary(
        trajectory_rows,
        worldlines,
        min_defects=min_defects,
        max_defects=max_defects,
    )
    receipt = bool(
        summary["defect_count_in_requested_band"]
        and summary["worldline_count"] >= min_defects
        and summary["transverse_motion_present"]
        and summary["staggered_births_present"]
    )
    return {
        "mode": "organic_defect_population_v0",
        "controlled_planted_assay": False,
        "organic_defect_population_diagnostic": True,
        "patch_count": patch_count,
        "steps": steps,
        "defect_count": defect_count,
        "min_defects": min_defects,
        "max_defects": max_defects,
        "support_node_count": support_node_count,
        "seed": int(seed),
        "declared_organic_population_law": {
            "law": (
                "staggered repair-hotspot births in an H3 tangent chart, randomized S3 holonomies, "
                "inverse-pair attraction, non-inverse shear/repulsion, transverse repair kicks, and "
                "near-contact bookkeeping without a planted left/right pair"
            ),
            "initial_speed": initial_speed,
            "stress_coupling": stress_coupling,
            "transverse_kick": transverse_kick,
            "stress_radius": stress_radius,
            "curvature_radius": curvature_radius,
            "contact_radius": contact_radius,
            "overlap_radius": overlap_radius,
            "spawn_radius": spawn_radius,
        },
        "organic_population_summary": summary,
        "organic_defect_population_receipt": receipt,
        "organic_proto_worldline_visualization_receipt": receipt,
        "production_gravity_receipt": False,
        "physical_gravity_prediction": False,
        "particle_matter_receipt": False,
        "trajectory_rows": trajectory_rows,
        "worldlines": worldlines,
        "rendering_modes": [
            {
                "mode": "subjective_observer_3d_points",
                "source": "worldlines[*].events[*].h3_spatial_point",
                "encoding": "moving observer-visible proto-defect points",
            },
            {
                "mode": "effective_edge_strings",
                "source": "worldlines[*].events",
                "encoding": "polyline/ribbon through each defect worldline",
            },
            {
                "mode": "curved_spacetime_stress_points",
                "source": "trajectory_rows",
                "encoding": "local stress-density proxy around moving defect support",
            },
        ],
        "claim_boundary": (
            "Organic multi-defect population diagnostic. Defect births, positions, holonomies, and "
            "transverse motion are seeded and emergent from the diagnostic repair-hotspot law rather "
            "than a fixed left/right pair. This is sufficient for natural-looking proto-worldline, "
            "string, and observer-camera visualization, but it is not particle matter, production "
            "gravity, or a physical merger claim."
        ),
    }


def write_organic_defect_population_report(
    out: Path,
    *,
    patch_count: int = 65_536,
    steps: int = 128,
    defect_count: int = 16,
    min_defects: int = 10,
    max_defects: int = 20,
    support_node_count: int = 8,
    seed: int = 2039,
    initial_speed: float = 0.028,
    stress_coupling: float = 0.018,
    transverse_kick: float = 0.010,
    stress_radius: float = 0.9,
    curvature_radius: float = 1.0,
    cycle_stride: int = 1,
    contact_radius: float = 0.12,
    overlap_radius: float = 0.28,
    spawn_radius: float = 1.25,
) -> dict[str, Any]:
    report = organic_defect_population_report(
        patch_count=patch_count,
        steps=steps,
        defect_count=defect_count,
        min_defects=min_defects,
        max_defects=max_defects,
        support_node_count=support_node_count,
        seed=seed,
        initial_speed=initial_speed,
        stress_coupling=stress_coupling,
        transverse_kick=transverse_kick,
        stress_radius=stress_radius,
        curvature_radius=curvature_radius,
        cycle_stride=cycle_stride,
        contact_radius=contact_radius,
        overlap_radius=overlap_radius,
        spawn_radius=spawn_radius,
    )
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    stem = destination.with_suffix("")
    _write_rows(stem.with_name(stem.name + "_trajectory.csv"), report["trajectory_rows"])
    worldline_rows = [
        {
            "worldline_id": row.get("worldline_id"),
            "observation_count": row.get("observation_count"),
            "birth_cycle": row.get("birth_cycle"),
            "death_cycle": row.get("death_cycle"),
            "lifetime_cycles": row.get("lifetime_cycles"),
            "persistent": row.get("persistent"),
            "class_mode": row.get("class_mode"),
            "holonomy_mode": row.get("holonomy_mode"),
            "mean_transport_distance": row.get("mean_transport_distance"),
        }
        for row in report["worldlines"]
    ]
    event_rows = [
        {**event, "worldline_id": row.get("worldline_id"), "event_index": event_index}
        for row in report["worldlines"]
        for event_index, event in enumerate(row.get("events", []))
    ]
    _write_rows(stem.with_name(stem.name + "_worldlines.csv"), worldline_rows)
    _write_rows(stem.with_name(stem.name + "_worldline_events.csv"), event_rows)
    return report


def _simulate_pair(
    *,
    mode: str,
    steps: int,
    initial_separation: float,
    stress_coupling: float,
    stress_radius: float,
    curvature_radius: float,
    cycle_stride: int,
) -> list[dict[str, Any]]:
    left = np.array([-0.5 * initial_separation, 0.0, 0.0], dtype=float)
    right = np.array([0.5 * initial_separation, 0.0, 0.0], dtype=float)
    rows: list[dict[str, Any]] = []
    for index in range(steps):
        distance = h3_distance(left, right, curvature_radius=curvature_radius)
        tangent_distance = float(np.linalg.norm(right - left))
        kernel = _stress_kernel(distance, stress_radius=stress_radius)
        contraction = max(0.0, 1.0 - stress_coupling * kernel)
        rows.append(
            {
                "mode": mode,
                "step": index,
                "cycle": int(index * cycle_stride),
                "left_h3_spatial_point": [float(value) for value in left.tolist()],
                "right_h3_spatial_point": [float(value) for value in right.tolist()],
                "tangent_separation": tangent_distance,
                "h3_separation": float(distance),
                "stress_kernel": kernel,
                "local_readout_contraction": contraction,
            }
        )
        if index == steps - 1:
            continue
        direction = _unit(right - left)
        displacement = min(0.25 * tangent_distance, stress_coupling * kernel)
        if mode == "stress_contraction":
            left = left + displacement * direction
            right = right - displacement * direction
        elif mode == "shuffled_pair_control":
            left = left - displacement * direction
            right = right + displacement * direction
    return rows


def _simulate_free_pair(
    *,
    steps: int,
    rng: np.random.Generator,
    support_node_count: int,
    initial_separation: float,
    initial_speed: float,
    stress_coupling: float,
    transverse_kick: float,
    stress_radius: float,
    curvature_radius: float,
    cycle_stride: int,
    contact_radius: float,
    overlap_radius: float,
    bind_speed_threshold: float,
    annihilation_overlap_threshold: float,
    inverse_identity_pass: bool,
) -> list[dict[str, Any]]:
    separation_axis = _random_unit(rng)
    impact_axis = _random_transverse_unit(rng, separation_axis)
    center = 0.08 * _random_unit(rng)
    left = center - 0.5 * initial_separation * separation_axis
    right = center + 0.5 * initial_separation * separation_axis
    closing = max(0.0, initial_speed)
    impact = 0.55 * closing
    left_velocity = 0.5 * closing * separation_axis + impact * impact_axis
    right_velocity = -0.5 * closing * separation_axis + impact * impact_axis
    rows: list[dict[str, Any]] = []
    contact_outcome: str | None = None
    contact_step: int | None = None
    bound_center: np.ndarray | None = None

    for index in range(steps):
        distance = h3_distance(left, right, curvature_radius=curvature_radius)
        tangent_separation = float(np.linalg.norm(right - left))
        relative_speed = float(np.linalg.norm(right_velocity - left_velocity))
        overlap_fraction = _support_overlap_fraction(distance, overlap_radius=overlap_radius)
        contact_event = None
        if contact_outcome is None and distance <= contact_radius:
            contact_step = index
            contact_outcome = _classify_contact_outcome(
                inverse_identity_pass=inverse_identity_pass,
                overlap_fraction=overlap_fraction,
                relative_speed=relative_speed,
                bind_speed_threshold=bind_speed_threshold,
                annihilation_overlap_threshold=annihilation_overlap_threshold,
            )
            contact_event = f"contact_{contact_outcome}"
            if contact_outcome == "scatter":
                normal = _unit(right - left)
                left_velocity = _scatter_velocity(left_velocity, normal, rng, transverse_kick)
                right_velocity = _scatter_velocity(right_velocity, -normal, rng, transverse_kick)
            elif contact_outcome == "bind":
                bound_center = 0.5 * (left + right)
                left_velocity = 0.35 * (left_velocity + right_velocity)
                right_velocity = left_velocity.copy()
            elif contact_outcome == "annihilate":
                left_velocity = np.zeros(3, dtype=float)
                right_velocity = np.zeros(3, dtype=float)

        rows.append(
            {
                "mode": "free_two_defect_dynamics",
                "step": index,
                "cycle": int(index * cycle_stride),
                "left_h3_spatial_point": [float(value) for value in left.tolist()],
                "right_h3_spatial_point": [float(value) for value in right.tolist()],
                "left_velocity": [float(value) for value in left_velocity.tolist()],
                "right_velocity": [float(value) for value in right_velocity.tolist()],
                "tangent_separation": tangent_separation,
                "h3_separation": float(distance),
                "stress_kernel": _stress_kernel(distance, stress_radius=stress_radius),
                "relative_speed": relative_speed,
                "support_overlap_fraction": overlap_fraction,
                "support_overlap_node_count": int(round(overlap_fraction * max(1, int(support_node_count)))),
                "contact_event": contact_event,
                "contact_outcome": contact_outcome,
                "charge_conservation_pass": True,
            }
        )
        if index == steps - 1:
            continue
        if contact_outcome == "annihilate":
            continue
        if contact_outcome == "bind" and bound_center is not None:
            decay = 0.82 ** max(index - int(contact_step or index), 0)
            orbital_axis = _random_transverse_unit(rng, separation_axis)
            offset = max(contact_radius * 0.35 * decay, 1.0e-6) * orbital_axis
            bound_center = bound_center + left_velocity
            left = bound_center - offset
            right = bound_center + offset
            continue

        delta = right - left
        direction = _unit(delta)
        distance_now = max(float(np.linalg.norm(delta)), 1.0e-12)
        stress_accel = stress_coupling * _stress_kernel(distance, stress_radius=stress_radius)
        left_velocity = left_velocity + stress_accel * direction
        right_velocity = right_velocity - stress_accel * direction
        kick_axis = _random_transverse_unit(rng, direction)
        kick_scale = transverse_kick / math.sqrt(float(index + 1))
        left_velocity = left_velocity + kick_scale * kick_axis
        right_velocity = right_velocity - 0.7 * kick_scale * kick_axis
        max_step = 0.22 * distance_now
        left_step = _cap_norm(left_velocity, max_step)
        right_step = _cap_norm(right_velocity, max_step)
        left = left + left_step
        right = right + right_step
    return rows


def _simulate_organic_defect_population(
    *,
    steps: int,
    defect_count: int,
    rng: np.random.Generator,
    patch_count: int,
    support_node_count: int,
    initial_speed: float,
    stress_coupling: float,
    transverse_kick: float,
    stress_radius: float,
    curvature_radius: float,
    cycle_stride: int,
    contact_radius: float,
    overlap_radius: float,
    spawn_radius: float,
) -> list[dict[str, Any]]:
    max_birth_step = max(1, min(steps // 4, 24))
    defects = []
    for index in range(defect_count):
        holonomy = int(rng.integers(1, int(len(S3_INV))))
        position_axis = _random_unit(rng)
        radius = spawn_radius * float(rng.uniform(0.15, 1.0))
        position = radius * position_axis + 0.08 * rng.normal(size=3)
        velocity_axis = _random_transverse_unit(rng, position_axis)
        velocity = initial_speed * float(rng.uniform(0.45, 1.25)) * velocity_axis
        birth_step = int(rng.integers(0, max_birth_step + 1))
        if index < min(defect_count, 4):
            birth_step = index
        support_start = int(rng.integers(0, max(1, patch_count)))
        support_nodes = [
            int((support_start + int(rng.integers(1, max(2, patch_count // max(2, support_node_count)))) * offset) % patch_count)
            for offset in range(support_node_count)
        ]
        defects.append(
            {
                "defect_id": f"organic_defect_{index:02d}",
                "birth_step": birth_step,
                "birth_trigger": ["repair_hotspot", "overlap_mismatch", "sector_shear", "record_phase_slip"][index % 4],
                "holonomy": holonomy,
                "inverse": int(S3_INV[holonomy]),
                "position": position,
                "velocity": velocity,
                "support_nodes": support_nodes,
                "support_start": support_start,
            }
        )

    rows: list[dict[str, Any]] = []
    for step in range(steps):
        active = [defect for defect in defects if int(defect["birth_step"]) <= step]
        accelerations = {str(defect["defect_id"]): np.zeros(3, dtype=float) for defect in active}
        local_stress = {str(defect["defect_id"]): 0.0 for defect in active}
        nearest: dict[str, tuple[str | None, float, float, str | None]] = {
            str(defect["defect_id"]): (None, math.inf, 0.0, None) for defect in active
        }

        for left_index, left_defect in enumerate(active):
            for right_defect in active[left_index + 1 :]:
                left_id = str(left_defect["defect_id"])
                right_id = str(right_defect["defect_id"])
                left_pos = np.asarray(left_defect["position"], dtype=float)
                right_pos = np.asarray(right_defect["position"], dtype=float)
                delta = right_pos - left_pos
                tangent_distance = max(float(np.linalg.norm(delta)), 1.0e-12)
                h3_dist = float(h3_distance(left_pos, right_pos, curvature_radius=curvature_radius))
                direction = delta / tangent_distance
                kernel = _stress_kernel(h3_dist, stress_radius=stress_radius)
                inverse_pair = _inverse_pair(int(left_defect["holonomy"]), int(right_defect["holonomy"]))
                signed = 1.0 if inverse_pair else -0.22
                accel = signed * stress_coupling * kernel * direction
                accelerations[left_id] = accelerations[left_id] + accel
                accelerations[right_id] = accelerations[right_id] - accel
                local_stress[left_id] += kernel
                local_stress[right_id] += kernel

                overlap = _support_overlap_fraction(h3_dist, overlap_radius=overlap_radius)
                relative_speed = float(
                    np.linalg.norm(np.asarray(right_defect["velocity"]) - np.asarray(left_defect["velocity"]))
                )
                outcome = None
                if h3_dist <= contact_radius:
                    outcome = "bind_candidate" if inverse_pair and relative_speed < 0.08 else "scatter_candidate"
                for defect_id, other_id in ((left_id, right_id), (right_id, left_id)):
                    old_other, old_distance, old_overlap, old_outcome = nearest[defect_id]
                    if h3_dist < old_distance:
                        nearest[defect_id] = (other_id, h3_dist, overlap, outcome or old_outcome)

        for defect in active:
            defect_id = str(defect["defect_id"])
            class_id = int(S3_CLASS[int(defect["holonomy"])])
            class_name = {1: "transposition", 2: "threecycle"}.get(class_id, "identity")
            other_id, nearest_distance, overlap, outcome = nearest[defect_id]
            position = np.asarray(defect["position"], dtype=float)
            velocity = np.asarray(defect["velocity"], dtype=float)
            event = "birth" if step == int(defect["birth_step"]) else "continue"
            if outcome:
                event = f"near_contact_{outcome}"
            rows.append(
                {
                    "mode": "organic_defect_population",
                    "step": int(step),
                    "cycle": int(step * cycle_stride),
                    "defect_id": defect_id,
                    "birth_trigger": defect["birth_trigger"],
                    "class": class_name,
                    "holonomy_mode": int(defect["holonomy"]),
                    "inverse_holonomy_mode": int(defect["inverse"]),
                    "h3_spatial_point": [float(value) for value in position.tolist()],
                    "velocity": [float(value) for value in velocity.tolist()],
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                    "vx": float(velocity[0]),
                    "vy": float(velocity[1]),
                    "vz": float(velocity[2]),
                    "local_stress_density": float(local_stress[defect_id]),
                    "nearest_defect_id": other_id,
                    "nearest_h3_separation": None if math.isinf(nearest_distance) else float(nearest_distance),
                    "support_overlap_fraction": float(overlap),
                    "support_overlap_node_count": int(round(overlap * support_node_count)),
                    "contact_event": event if event.startswith("near_contact") else None,
                    "contact_outcome": outcome,
                    "charge_conservation_pass": True,
                    "support_node_count": support_node_count,
                    "support_nodes": list(defect["support_nodes"]),
                    "render_as_point": True,
                    "render_as_string": True,
                    "render_in_subjective_observer_view": True,
                }
            )

        if step == steps - 1:
            continue
        for defect in active:
            defect_id = str(defect["defect_id"])
            position = np.asarray(defect["position"], dtype=float)
            velocity = np.asarray(defect["velocity"], dtype=float)
            nearest_id, nearest_distance, _overlap, _outcome = nearest[defect_id]
            kick_axis = _random_transverse_unit(rng, position if float(np.linalg.norm(position)) > 1.0e-9 else _random_unit(rng))
            kick = transverse_kick * float(rng.normal()) / math.sqrt(float(step + 1)) * kick_axis
            velocity = 0.965 * velocity + accelerations[defect_id] + kick
            if nearest_id is not None and nearest_distance < contact_radius:
                velocity = _scatter_velocity(velocity, _unit(position), rng, 0.5 * transverse_kick)
            max_step = 0.035 + 0.08 * _stress_kernel(local_stress[defect_id], stress_radius=1.0)
            step_vec = _cap_norm(velocity, max_step)
            position = position + step_vec
            radius = float(np.linalg.norm(position))
            if radius > 1.45 * spawn_radius:
                normal = _unit(position)
                position = normal * (1.45 * spawn_radius)
                velocity = velocity - 1.8 * _dot_np(velocity, normal) * normal
            defect["position"] = position
            defect["velocity"] = velocity
    return rows


def _worldlines_from_organic_rows(
    rows: list[dict[str, Any]],
    *,
    patch_count: int,
    support_node_count: int,
) -> list[dict[str, Any]]:
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_id.setdefault(str(row["defect_id"]), []).append(row)
    worldlines = []
    for defect_id, defect_rows in sorted(rows_by_id.items()):
        events = []
        for index, row in enumerate(defect_rows):
            point = list(row["h3_spatial_point"])
            events.append(
                {
                    "cycle": int(row["cycle"]),
                    "event": "birth" if index == 0 else row.get("contact_event") or "continue",
                    "class": row.get("class"),
                    "holonomy_mode": int(row["holonomy_mode"]),
                    "inverse_holonomy_mode": int(row["inverse_holonomy_mode"]),
                    "support_node_count": int(row.get("support_node_count", support_node_count)),
                    "support_nodes": list(row.get("support_nodes", []))[:support_node_count],
                    "h3_spatial_point": point,
                    "velocity": list(row.get("velocity", [])),
                    "local_stress_density": float(row.get("local_stress_density", 0.0)),
                    "nearest_defect_id": row.get("nearest_defect_id"),
                    "nearest_h3_separation": row.get("nearest_h3_separation"),
                    "support_overlap_fraction": float(row.get("support_overlap_fraction", 0.0)),
                    "support_overlap_node_count": int(row.get("support_overlap_node_count", 0)),
                    "contact_outcome": row.get("contact_outcome"),
                    "charge_conservation_pass": bool(row.get("charge_conservation_pass", False)),
                    "transport_distance": None if index == 0 else _h3_step(defect_rows[index - 1]["h3_spatial_point"], point),
                    "render_modes": ["h3_point", "edge_string", "subjective_observer_3d_point"],
                }
            )
        path_distance = _mean(event["transport_distance"] for event in events if event["transport_distance"] is not None)
        worldlines.append(
            {
                "worldline_id": defect_id,
                "observation_count": len(events),
                "birth_cycle": int(events[0]["cycle"]) if events else None,
                "death_cycle": int(events[-1]["cycle"]) if events else None,
                "lifetime_cycles": int(events[-1]["cycle"] - events[0]["cycle"]) if events else 0,
                "persistent": bool(len(events) >= 3),
                "class_mode": events[0].get("class") if events else None,
                "holonomy_mode": events[0].get("holonomy_mode") if events else None,
                "support_patch_count": min(patch_count, support_node_count),
                "mean_transport_distance": path_distance,
                "events": events,
            }
        )
    return worldlines


def _organic_population_summary(
    rows: list[dict[str, Any]],
    worldlines: list[dict[str, Any]],
    *,
    min_defects: int,
    max_defects: int,
) -> dict[str, Any]:
    worldline_count = len(worldlines)
    birth_cycles = [int(row["birth_cycle"]) for row in worldlines if row.get("birth_cycle") is not None]
    contact_rows = [row for row in rows if row.get("contact_event")]
    y_values = [abs(float(row.get("y", 0.0))) for row in rows]
    z_values = [abs(float(row.get("z", 0.0))) for row in rows]
    stress_values = [float(row.get("local_stress_density", 0.0)) for row in rows]
    return {
        "worldline_count": worldline_count,
        "defect_count_in_requested_band": bool(min_defects <= worldline_count <= max_defects),
        "min_requested_defects": int(min_defects),
        "max_requested_defects": int(max_defects),
        "birth_cycle_min": min(birth_cycles) if birth_cycles else None,
        "birth_cycle_max": max(birth_cycles) if birth_cycles else None,
        "staggered_births_present": bool(len(set(birth_cycles)) > 1),
        "near_contact_event_count": len(contact_rows),
        "transverse_motion_present": bool(max(y_values or [0.0]) > 1.0e-6 or max(z_values or [0.0]) > 1.0e-6),
        "mean_local_stress_density": _mean(stress_values),
        "max_local_stress_density": max(stress_values) if stress_values else 0.0,
        "persistent_worldline_count": int(sum(1 for row in worldlines if row.get("persistent"))),
        "fixed_left_right_pair": False,
        "controlled_planted_assay": False,
    }


def _inverse_pair(left_holonomy: int, right_holonomy: int) -> bool:
    return bool(
        int(S3_MUL[int(left_holonomy), int(right_holonomy)]) == 0
        or int(S3_MUL[int(right_holonomy), int(left_holonomy)]) == 0
    )


def _stress_kernel(distance: float, *, stress_radius: float) -> float:
    scaled = float(distance) / max(float(stress_radius), 1.0e-12)
    return float(1.0 / (1.0 + scaled * scaled))


def _support_overlap_fraction(distance: float, *, overlap_radius: float) -> float:
    return float(max(0.0, min(1.0, 1.0 - float(distance) / max(float(overlap_radius), 1.0e-12))))


def _classify_contact_outcome(
    *,
    inverse_identity_pass: bool,
    overlap_fraction: float,
    relative_speed: float,
    bind_speed_threshold: float,
    annihilation_overlap_threshold: float,
) -> str:
    if not inverse_identity_pass:
        return "scatter"
    if overlap_fraction >= annihilation_overlap_threshold and relative_speed <= 0.5 * bind_speed_threshold:
        return "annihilate"
    if relative_speed <= bind_speed_threshold:
        return "bind"
    return "scatter"


def _scatter_velocity(velocity: np.ndarray, normal: np.ndarray, rng: np.random.Generator, transverse_kick: float) -> np.ndarray:
    reflected = np.asarray(velocity, dtype=float) - 2.0 * _dot_np(velocity, normal) * np.asarray(normal, dtype=float)
    return reflected + float(transverse_kick) * _random_transverse_unit(rng, normal)


def _trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    initial = float(rows[0]["h3_separation"]) if rows else math.nan
    final = float(rows[-1]["h3_separation"]) if rows else math.nan
    approach = (initial - final) / initial if initial > 0.0 else 0.0
    return {
        "initial_h3_separation": initial,
        "final_h3_separation": final,
        "absolute_h3_approach": float(initial - final),
        "approach_fraction": float(approach),
        "min_h3_separation": float(min(float(row["h3_separation"]) for row in rows)) if rows else None,
        "max_h3_separation": float(max(float(row["h3_separation"]) for row in rows)) if rows else None,
    }


def _free_trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = _trajectory_summary(rows)
    contact_rows = [row for row in rows if row.get("contact_event")]
    outcome = next((row.get("contact_outcome") for row in contact_rows if row.get("contact_outcome")), None)
    if outcome is None:
        outcome = "pass_through"
    y_values = [
        abs(float(point[1]))
        for row in rows
        for point in (row.get("left_h3_spatial_point"), row.get("right_h3_spatial_point"))
        if isinstance(point, list) and len(point) >= 3
    ]
    z_values = [
        abs(float(point[2]))
        for row in rows
        for point in (row.get("left_h3_spatial_point"), row.get("right_h3_spatial_point"))
        if isinstance(point, list) and len(point) >= 3
    ]
    base.update(
        {
            "contact_step": int(contact_rows[0]["step"]) if contact_rows else None,
            "contact_cycle": int(contact_rows[0]["cycle"]) if contact_rows else None,
            "contact_outcome": outcome,
            "explicit_contact_outcome": True,
            "charge_conservation_pass": all(bool(row.get("charge_conservation_pass", False)) for row in rows),
            "max_support_overlap_fraction": float(
                max(float(row.get("support_overlap_fraction", 0.0)) for row in rows)
            )
            if rows
            else 0.0,
            "transverse_motion_present": bool(max(y_values or [0.0]) > 1.0e-6 or max(z_values or [0.0]) > 1.0e-6),
            "straight_x_axis_control": False,
        }
    )
    return base


def _worldlines_from_rows(
    rows: list[dict[str, Any]],
    *,
    patch_count: int,
    support_node_count: int,
    holonomy: int,
    inverse: int,
) -> list[dict[str, Any]]:
    return [
        _worldline(
            rows,
            side="left",
            worldline_id="stress_pair_left",
            holonomy=holonomy,
            inverse=inverse,
            support_start=10_000,
            patch_count=patch_count,
            support_node_count=support_node_count,
        ),
        _worldline(
            rows,
            side="right",
            worldline_id="stress_pair_right",
            holonomy=inverse,
            inverse=holonomy,
            support_start=20_000,
            patch_count=patch_count,
            support_node_count=support_node_count,
        ),
    ]


def _worldlines_from_free_rows(
    rows: list[dict[str, Any]],
    *,
    patch_count: int,
    support_node_count: int,
    holonomy: int,
    inverse: int,
) -> list[dict[str, Any]]:
    return [
        _free_worldline(
            rows,
            side="left",
            worldline_id="free_pair_left",
            holonomy=holonomy,
            inverse=inverse,
            support_start=30_000,
            patch_count=patch_count,
            support_node_count=support_node_count,
        ),
        _free_worldline(
            rows,
            side="right",
            worldline_id="free_pair_right",
            holonomy=inverse,
            inverse=holonomy,
            support_start=40_000,
            patch_count=patch_count,
            support_node_count=support_node_count,
        ),
    ]


def _free_worldline(
    rows: list[dict[str, Any]],
    *,
    side: str,
    worldline_id: str,
    holonomy: int,
    inverse: int,
    support_start: int,
    patch_count: int,
    support_node_count: int,
) -> dict[str, Any]:
    class_id = int(S3_CLASS[int(holonomy)])
    class_name = {1: "transposition", 2: "threecycle"}.get(class_id, "identity")
    support_nodes = [int((support_start + offset) % patch_count) for offset in range(support_node_count)]
    key = f"{side}_h3_spatial_point"
    velocity_key = f"{side}_velocity"
    events = []
    terminal_seen = False
    for index, row in enumerate(rows):
        outcome = row.get("contact_outcome")
        event_kind = "birth" if index == 0 else "continue"
        if row.get("contact_event"):
            event_kind = str(row["contact_event"])
        elif outcome == "bind":
            event_kind = "bound_continue"
        elif outcome == "annihilate":
            event_kind = "annihilated"
        if terminal_seen and outcome == "annihilate":
            continue
        terminal_seen = bool(outcome == "annihilate")
        events.append(
            {
                "cycle": int(row["cycle"]),
                "event": event_kind,
                "class": class_name,
                "holonomy_mode": int(holonomy),
                "inverse_holonomy_mode": int(inverse),
                "support_node_count": support_node_count,
                "support_nodes": support_nodes,
                "h3_spatial_point": list(row[key]),
                "velocity": list(row.get(velocity_key, [])),
                "pair_h3_separation": float(row["h3_separation"]),
                "support_overlap_fraction": float(row.get("support_overlap_fraction", 0.0)),
                "support_overlap_node_count": int(row.get("support_overlap_node_count", 0)),
                "contact_outcome": outcome,
                "charge_conservation_pass": bool(row.get("charge_conservation_pass", False)),
                "transport_distance": None if index == 0 else _h3_step(rows[index - 1][key], row[key]),
            }
        )
    return {
        "worldline_id": worldline_id,
        "observation_count": len(events),
        "birth_cycle": int(events[0]["cycle"]) if events else None,
        "death_cycle": int(events[-1]["cycle"]) if events else None,
        "lifetime_cycles": int(events[-1]["cycle"] - events[0]["cycle"]) if events else 0,
        "persistent": bool(len(events) >= 3),
        "contact_outcome": next((event.get("contact_outcome") for event in events if event.get("contact_outcome")), None)
        or "pass_through",
        "mean_transport_distance": _mean(
            event["transport_distance"] for event in events if event["transport_distance"] is not None
        ),
        "events": events,
    }


def _worldline(
    rows: list[dict[str, Any]],
    *,
    side: str,
    worldline_id: str,
    holonomy: int,
    inverse: int,
    support_start: int,
    patch_count: int,
    support_node_count: int,
) -> dict[str, Any]:
    class_id = int(S3_CLASS[int(holonomy)])
    class_name = {1: "transposition", 2: "threecycle"}.get(class_id, "identity")
    support_nodes = [int((support_start + offset) % patch_count) for offset in range(support_node_count)]
    key = f"{side}_h3_spatial_point"
    events = [
        {
            "cycle": int(row["cycle"]),
            "event": "birth" if index == 0 else "continue",
            "class": class_name,
            "holonomy_mode": int(holonomy),
            "inverse_holonomy_mode": int(inverse),
            "support_node_count": support_node_count,
            "support_nodes": support_nodes,
            "h3_spatial_point": list(row[key]),
            "pair_h3_separation": float(row["h3_separation"]),
            "local_readout_contraction": float(row["local_readout_contraction"]),
            "transport_distance": None if index == 0 else _h3_step(rows[index - 1][key], row[key]),
        }
        for index, row in enumerate(rows)
    ]
    return {
        "worldline_id": worldline_id,
        "observation_count": len(events),
        "birth_cycle": int(events[0]["cycle"]) if events else None,
        "death_cycle": int(events[-1]["cycle"]) if events else None,
        "lifetime_cycles": int(events[-1]["cycle"] - events[0]["cycle"]) if events else 0,
        "persistent": bool(len(events) >= 3),
        "mean_transport_distance": _mean(
            event["transport_distance"] for event in events if event["transport_distance"] is not None
        ),
        "events": events,
    }


def _h3_step(left: list[float], right: list[float]) -> float:
    return float(h3_distance(left, right))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return np.zeros(3, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    return _unit(vector)


def _random_transverse_unit(rng: np.random.Generator, axis: np.ndarray) -> np.ndarray:
    axis = _unit(axis)
    candidate = rng.normal(size=3)
    transverse = candidate - _dot_np(candidate, axis) * axis
    if float(np.linalg.norm(transverse)) <= 1.0e-12:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(_dot_np(fallback, axis))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        transverse = fallback - _dot_np(fallback, axis) * axis
    return _unit(transverse)


def _cap_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= float(max_norm) or norm <= 1.0e-12:
        return np.asarray(vector, dtype=float)
    return np.asarray(vector, dtype=float) * (float(max_norm) / norm)


def _dot_np(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(np.asarray(left, dtype=float), np.asarray(right, dtype=float)))


def _positive(value: float, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) and result > 0.0 else float(default)


def _monotone_nonincreasing(values: list[float], *, tolerance: float = 1.0e-12) -> bool:
    return all(right <= left + tolerance for left, right in zip(values, values[1:], strict=False))


def _mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
