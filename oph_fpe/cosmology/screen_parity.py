"""Parity-odd screen statistics with mirror and shuffle controls.

Motivation (epic-wins tracker DK-06/DK-07): does the freezeout screen carry
an intrinsic chirality that could feed cosmological parity-violating
observables (uniform polarization rotation, parity-odd correlations)?

Group-theoretic scoping result recorded here: S3 is ambivalent (every
element is conjugate to its inverse), so every conjugation-invariant
function of S3 holonomies is parity-even. The S3 gauge sector therefore
cannot source a parity-odd signal through holonomy classes alone; any
screen chirality must live in geometric or transport data. The statistic
below is the leading geometric candidate: the pseudo-scalar cross-gradient

    chi(a, b) = < r_hat . (grad a x grad b) >

over pairs of committed scalar screen fields. chi is gauge-invariant
(fields are invariant), flips sign under a spatial mirror, and vanishes in
the mean for a parity-symmetric ensemble.

Controls:
- mirror covariance: reflecting the screen through a plane must flip the
  sign of chi exactly (pipeline covariance check, catches orientation
  bugs);
- value shuffle: permuting one field's values across points must send chi
  to zero within the null spread;
- hemisphere jackknife: a crude spatial error bar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SCREEN_PARITY_SCHEMA = "screen_parity_pseudoscalar_v1"

DEFAULT_FIELD_PAIRS = (
    ("record_signature", "s3_class_density"),
    ("record_signature", "cell_entropy"),
    ("s3_class_density", "cumulative_repair_load"),
    ("record_signature_smooth_k16", "s3_class_density"),
)


def _tangent_gradients(
    points: np.ndarray,
    values: np.ndarray,
    neighbors: np.ndarray,
) -> np.ndarray:
    """Least-squares tangent-plane gradient of a scalar field per point.

    ``neighbors`` holds indices of the k nearest neighbors per point. The
    gradient is solved in the tangent plane of the unit sphere at each
    point (radial component projected out).
    """

    count, k = neighbors.shape
    gradients = np.zeros((count, 3), dtype=float)
    base = points / np.linalg.norm(points, axis=1, keepdims=True)
    for index in range(count):
        neighbor_ids = neighbors[index]
        deltas = base[neighbor_ids] - base[index]
        radial = base[index]
        deltas = deltas - np.outer(deltas @ radial, radial)
        rhs = values[neighbor_ids] - values[index]
        solution, *_ = np.linalg.lstsq(deltas, rhs, rcond=None)
        solution = solution - (solution @ radial) * radial
        gradients[index] = solution
    return gradients


def _knn(points: np.ndarray, k: int) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    _, neighbor_ids = tree.query(points, k=k + 1)
    return neighbor_ids[:, 1:]


def pseudoscalar_statistic(
    points: np.ndarray,
    field_a: np.ndarray,
    field_b: np.ndarray,
    *,
    neighbors: np.ndarray,
) -> float:
    base = points / np.linalg.norm(points, axis=1, keepdims=True)
    grad_a = _tangent_gradients(points, field_a, neighbors)
    grad_b = _tangent_gradients(points, field_b, neighbors)
    cross = np.cross(grad_a, grad_b)
    return float(np.mean(np.einsum("ij,ij->i", cross, base)))


def screen_parity_report(
    run_dir: str | Path,
    *,
    field_pairs: tuple[tuple[str, str], ...] = DEFAULT_FIELD_PAIRS,
    k_neighbors: int = 12,
    max_points: int = 20000,
    shuffle_draws: int = 32,
    seed: int = 1,
) -> dict[str, Any]:
    run = Path(run_dir)
    fields_path = run / "freezeout_fields.npz"
    if not fields_path.exists():
        return {"schema": SCREEN_PARITY_SCHEMA, "status": "missing_freezeout_fields"}
    payload = np.load(fields_path)
    if "points" not in payload:
        return {"schema": SCREEN_PARITY_SCHEMA, "status": "missing_points"}
    points = np.asarray(payload["points"], dtype=float)
    rng = np.random.default_rng(int(seed))
    if points.shape[0] > int(max_points):
        keep = rng.choice(points.shape[0], size=int(max_points), replace=False)
        keep.sort()
    else:
        keep = np.arange(points.shape[0])
    points_used = points[keep]
    neighbors = _knn(points_used, int(k_neighbors))
    base = points_used / np.linalg.norm(points_used, axis=1, keepdims=True)

    results: list[dict[str, Any]] = []
    for name_a, name_b in field_pairs:
        if name_a not in payload or name_b not in payload:
            results.append(
                {
                    "fields": [name_a, name_b],
                    "status": "field_missing",
                }
            )
            continue
        field_a = np.asarray(payload[name_a], dtype=float)[keep]
        field_b = np.asarray(payload[name_b], dtype=float)[keep]
        if float(np.std(field_a)) == 0.0 or float(np.std(field_b)) == 0.0:
            results.append({"fields": [name_a, name_b], "status": "degenerate_field"})
            continue
        field_a = (field_a - np.mean(field_a)) / np.std(field_a)
        field_b = (field_b - np.mean(field_b)) / np.std(field_b)

        chi = pseudoscalar_statistic(points_used, field_a, field_b, neighbors=neighbors)

        # Mirror covariance: reflect x -> -x. Tangent gradients transform as
        # vectors; the triple product flips sign exactly when recomputed in
        # the mirrored frame.
        mirrored_points = points_used.copy()
        mirrored_points[:, 0] *= -1.0
        mirrored_neighbors = neighbors  # kNN structure is mirror-invariant
        chi_mirror = pseudoscalar_statistic(
            mirrored_points, field_a, field_b, neighbors=mirrored_neighbors
        )
        mirror_covariant = bool(
            np.isfinite(chi)
            and np.isfinite(chi_mirror)
            and abs(chi + chi_mirror) <= 1.0e-8 + 1.0e-3 * abs(chi)
        )

        # Null distribution: shuffle field_b across points.
        null_values = []
        for _ in range(int(shuffle_draws)):
            permuted = rng.permutation(field_b)
            null_values.append(
                pseudoscalar_statistic(points_used, field_a, permuted, neighbors=neighbors)
            )
        null_std = float(np.std(np.asarray(null_values))) if null_values else None
        z_score = float(chi / null_std) if null_std and null_std > 0 else None

        # Hemisphere jackknife for a crude spatial error bar.
        hemisphere_values = []
        for axis in range(3):
            for sign in (1.0, -1.0):
                mask = (base[:, axis] * sign) > 0
                if int(np.sum(mask)) < 100:
                    continue
                sub_points = points_used[mask]
                sub_neighbors = _knn(sub_points, int(k_neighbors))
                hemisphere_values.append(
                    pseudoscalar_statistic(
                        sub_points,
                        field_a[mask],
                        field_b[mask],
                        neighbors=sub_neighbors,
                    )
                )
        hemisphere_std = (
            float(np.std(np.asarray(hemisphere_values))) if hemisphere_values else None
        )

        results.append(
            {
                "fields": [name_a, name_b],
                "status": "evaluated",
                "chi": chi,
                "chi_mirror": chi_mirror,
                "mirror_covariant": mirror_covariant,
                "null_std_shuffle": null_std,
                "z_score_vs_shuffle": z_score,
                "hemisphere_std": hemisphere_std,
                "hemisphere_count": len(hemisphere_values),
            }
        )

    evaluated = [entry for entry in results if entry.get("status") == "evaluated"]
    max_abs_z = (
        float(max(abs(entry["z_score_vs_shuffle"]) for entry in evaluated if entry["z_score_vs_shuffle"] is not None))
        if evaluated
        else None
    )
    return {
        "schema": SCREEN_PARITY_SCHEMA,
        "status": "evaluated" if evaluated else "no_field_pairs",
        "points_used": int(points_used.shape[0]),
        "k_neighbors": int(k_neighbors),
        "shuffle_draws": int(shuffle_draws),
        "pairs": results,
        "max_abs_z": max_abs_z,
        "chirality_detected": bool(max_abs_z is not None and max_abs_z >= 4.0),
        "s3_holonomy_scoping_note": (
            "S3 is ambivalent: conjugation-invariant holonomy statistics are "
            "parity-even by group theory. A parity-odd screen signal must be "
            "geometric or transport-side; this report measures the leading "
            "geometric candidate."
        ),
    }


def write_screen_parity_report(run_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    report = screen_parity_report(run_dir, **kwargs)
    out_path = Path(run_dir) / "screen_parity_report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
