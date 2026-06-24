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


def _stress_kernel(distance: float, *, stress_radius: float) -> float:
    scaled = float(distance) / max(float(stress_radius), 1.0e-12)
    return float(1.0 / (1.0 + scaled * scaled))


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
