from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from oph_fpe.claims import PROXY, with_claim_metadata
from oph_fpe.cosmology.angular_power import angular_power_report
from oph_fpe.cosmology.proxy_pipeline import cosmo_proxy_receipt


def write_freezeout_products(
    run_path: Path,
    *,
    points: np.ndarray,
    fields: dict[str, np.ndarray],
    cell_area_planck: np.ndarray,
    cell_entropy: np.ndarray,
    freezeout_cycle: int,
    committed_fraction: float,
    config: dict[str, Any],
    seed: int,
    gate_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    freezeout_cfg = config.get("freezeout", {})
    angular_cfg = config.get("angular_power", {})
    output_profile = str(config.get("output_profile", config.get("outputs", {}).get("profile", "evidence")))
    write_npz = bool(config.get("write_freezeout_npz", output_profile != "compact"))
    write_csv = bool(config.get("write_cl_csv", output_profile in {"evidence", "debug", "viewer"}))
    augmented_fields = _augment_freezeout_fields(points, fields, freezeout_cfg.get("derived_fields", []))
    selected_names = [str(name) for name in freezeout_cfg.get("fields", list(augmented_fields.keys()))]
    selected = {name: augmented_fields[name] for name in selected_names if name in augmented_fields}

    if write_npz:
        np.savez_compressed(
            run_path / "freezeout_fields.npz",
            points=points.astype(np.float32),
            cell_area_planck=cell_area_planck.astype(np.float32),
            cell_entropy=cell_entropy.astype(np.float32),
            **{name: values.astype(np.float32) for name, values in selected.items()},
        )

    summary = with_claim_metadata({
        "freezeout_cycle": int(freezeout_cycle),
        "committed_fraction": float(committed_fraction),
        "point_count": int(points.shape[0]),
        "fields": {name: _field_stats(values) for name, values in selected.items()},
        "output_profile": output_profile,
        "freezeout_npz_written": write_npz,
        "claim_boundary": (
            "freezeout screen field bundle; this is an observer-screen angular statistic, "
            "not a reconstructed 3D bulk observable"
        ),
        "gate_report": gate_report or {},
    }, claim_level=PROXY, receipt="FREEZEOUT_SCREEN_FIELD_BUNDLE", physical_claim=False)
    (run_path / "freezeout_map_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    report = angular_power_report(
        points,
        selected,
        ell_max=int(angular_cfg.get("ell_max", 64)),
        pair_samples=int(angular_cfg.get("pair_samples", 200_000)),
        seed=seed,
        controls=[str(item) for item in angular_cfg.get("controls", ["shuffled_field", "random_gaussian"])],
        estimator=str(angular_cfg.get("estimator", "spherical_harmonic")),
        measure_weights=cell_entropy,
        harmonic_batch_size=int(angular_cfg.get("harmonic_batch_size", 4096)),
        n_jobs=angular_cfg.get("n_jobs", 1),
    )
    report = with_claim_metadata(report, claim_level=PROXY, receipt="SCREEN_FREEZEOUT_CL_PROXY", physical_claim=False)
    report["freezeout_cycle"] = int(freezeout_cycle)
    report["committed_fraction"] = float(committed_fraction)
    report["gate_report"] = gate_report or {}
    report["output_profile"] = output_profile
    proxy_report = cosmo_proxy_receipt(report)
    report["cosmo_proxy_receipt"] = {
        "mode": proxy_report["mode"],
        "receipt": proxy_report["receipt"],
        "claim_level": proxy_report["claim_level"],
        "best_field": proxy_report["best_field"],
        "physical_claim": proxy_report["physical_claim"],
    }
    (run_path / "cl_comparison_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (run_path / "oph_cosmo_proxy_v0_report.json").write_text(
        json.dumps(proxy_report, indent=2, default=str), encoding="utf-8"
    )
    if write_csv:
        _write_spectrum_csv(run_path / "cl_proxy.csv", report["fields"])
        _write_control_csv(run_path / "cl_controls.csv", report["controls"])
    return with_claim_metadata({
        "freezeout_cycle": int(freezeout_cycle),
        "committed_fraction": float(committed_fraction),
        "ell_max": int(report["ell_max"]),
        "pair_samples": int(report["pair_samples"]),
        "output_profile": output_profile,
        "freezeout_npz_written": write_npz,
        "cl_csv_written": write_csv,
        "fields": {
            name: {
                "peak_ell": field_report.get("peak_ell"),
                "peak_D_ell": field_report.get("peak_D_ell"),
                "low_ell_power_2_10": field_report.get("low_ell_power_2_10"),
                "control_comparison": field_report.get("control_comparison"),
            }
            for name, field_report in report["fields"].items()
        },
        "gate_report": gate_report or {},
        "cosmo_proxy_receipt": proxy_report,
        "claim_boundary": (
            "first measurement-facing screen proxy gated by finite BW/KMS receipts; "
            "not a physical CMB prediction, not CAMB/CLASS input, and not a 3D-bulk claim"
        ),
    }, claim_level=PROXY, receipt="SCREEN_FREEZEOUT_CL_PROXY", physical_claim=False)


def _field_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _augment_freezeout_fields(
    points: np.ndarray,
    fields: dict[str, np.ndarray],
    derived_specs: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, np.ndarray]:
    augmented = {str(name): np.asarray(values, dtype=float) for name, values in fields.items()}
    if not derived_specs:
        return augmented
    tree: cKDTree | None = None
    neighbor_cache: dict[int, np.ndarray] = {}
    for spec in derived_specs:
        if not isinstance(spec, dict):
            continue
        kind = str(spec.get("kind", "knn_smooth"))
        source = str(spec.get("source", ""))
        name = str(spec.get("name", source))
        if not name:
            continue
        if kind == "knn_smooth":
            if source not in augmented:
                continue
            if tree is None:
                tree = cKDTree(points)
            k = max(1, int(spec.get("k", 16)))
            steps = max(1, int(spec.get("steps", 1)))
            alpha = float(np.clip(spec.get("alpha", 1.0), 0.0, 1.0))
            if k not in neighbor_cache:
                _dist, neighbors = tree.query(points, k=min(k, points.shape[0]))
                if neighbors.ndim == 1:
                    neighbors = neighbors[:, None]
                neighbor_cache[k] = np.asarray(neighbors, dtype=np.int64)
            augmented[name] = _knn_smooth(augmented[source], neighbor_cache[k], steps=steps, alpha=alpha)
        elif kind == "weighted_product":
            modulator = str(spec.get("modulator", ""))
            if source not in augmented or modulator not in augmented:
                continue
            strength = float(spec.get("strength", 1.0))
            source_values = _standardize_field(augmented[source])
            mod_values = _standardize_field(augmented[modulator])
            augmented[name] = _standardize_field(source_values * (1.0 + strength * mod_values))
        elif kind == "linear_combo":
            terms = spec.get("terms", [])
            if not isinstance(terms, list):
                continue
            combo = np.zeros(points.shape[0], dtype=float)
            used = False
            for term in terms:
                if not isinstance(term, dict):
                    continue
                field_name = str(term.get("field", ""))
                if field_name not in augmented:
                    continue
                combo += float(term.get("weight", 1.0)) * _standardize_field(augmented[field_name])
                used = True
            if used:
                augmented[name] = _standardize_field(combo)
    return augmented


def _knn_smooth(values: np.ndarray, neighbors: np.ndarray, *, steps: int, alpha: float) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    for _ in range(max(1, int(steps))):
        neighbor_mean = np.mean(result[neighbors], axis=1)
        result = (1.0 - float(alpha)) * result + float(alpha) * neighbor_mean
    return _standardize_field(result)


def _standardize_field(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - float(np.mean(values)) if values.size else values
    scale = float(np.std(centered)) if centered.size else 0.0
    if scale <= 1e-12:
        return np.zeros_like(centered, dtype=float)
    return centered / scale


def _write_spectrum_csv(path: Path, fields: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["field", "ell", "C_ell", "D_ell"])
        writer.writeheader()
        for name, report in fields.items():
            for row in report["spectrum"]:
                writer.writerow({"field": name, **row})


def _write_control_csv(path: Path, controls: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["field", "control", "ell", "C_ell", "D_ell"])
        writer.writeheader()
        for field_name, control_reports in controls.items():
            for control_name, report in control_reports.items():
                for row in report["spectrum"]:
                    writer.writerow({"field": field_name, "control": control_name, **row})
