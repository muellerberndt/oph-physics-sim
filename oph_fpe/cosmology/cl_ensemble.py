from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def collect_cl_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in run_dirs:
        for cl_path in sorted(Path(root).glob("**/cl_comparison_report.json")):
            run_path = cl_path.parent.resolve()
            if run_path in seen:
                continue
            seen.add(run_path)
            cl_report = json.loads(cl_path.read_text(encoding="utf-8"))
            manifest = _read_json(run_path / "manifest.json")
            gate = _read_json(run_path / "cosmology_gate_report.json")
            cmb = _read_json(run_path / "cmb_lite_comparison_report.json")
            cmb_fields = cmb.get("field_comparisons", {}) if cmb else {}
            best_field = cmb.get("best_shape_field") if cmb else None
            best_positive_field = cmb.get("best_positive_shape_field") if cmb else None
            best = cmb_fields.get(best_field, {}) if best_field else {}
            best_positive = cmb_fields.get(best_positive_field, {}) if best_positive_field else {}
            rows.append(
                {
                    "run_id": manifest.get("run_id", run_path.name),
                    "path": str(run_path),
                    "patch_count": int(manifest.get("patch_count", 0)),
                    "seed": _seed_from_config(run_path / "config.yml"),
                    "estimator": cl_report.get("estimator"),
                    "ell_max": int(cl_report.get("ell_max", 0)),
                    "gate_allowed": bool(gate.get("allowed", False)),
                    "gate_checks": gate.get("checks", {}),
                    "bulk_3d_established": bool(gate.get("checks", {}).get("bulk_3d_established", False)),
                    "fields": cl_report.get("fields", {}),
                    "cmb_lite": {
                        "benchmark": (cmb.get("benchmark", {}) if cmb else {}).get("label"),
                        "best_field": best_field,
                        "best_shape_correlation": best.get("shape_correlation"),
                        "best_normalized_rmse": best.get("normalized_rmse"),
                        "best_positive_field": best_positive_field,
                        "best_positive_shape_correlation": best_positive.get("shape_correlation"),
                        "best_positive_normalized_rmse": best_positive.get("normalized_rmse"),
                    }
                    if cmb
                    else {},
                }
            )
    return rows


def cl_ensemble_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_n: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_n.setdefault(int(row.get("patch_count", 0)), []).append(row)
    sizes: list[dict[str, Any]] = []
    for patch_count, group in sorted(by_n.items()):
        allowed = [row for row in group if row.get("gate_allowed")]
        sizes.append(
            {
                "patch_count": patch_count,
                "run_count": len(group),
                "gate_allowed_count": len(allowed),
                "gate_allowed_fraction": float(len(allowed) / max(len(group), 1)),
                "fields": _field_ensembles(allowed),
                "cmb_lite": _cmb_lite_ensemble(allowed),
            }
        )
    return {
        "mode": "freezeout_screen_cl_ensemble",
        "run_count": len(rows),
        "gate_allowed_count": sum(1 for row in rows if row.get("gate_allowed")),
        "sizes": sizes,
        "physical_cmb_prediction": False,
        "bulk_3d_established": False,
        "claim_boundary": (
            "ensemble of gated observer-screen C_l proxy receipts; useful for seed/control stability, "
            "not a Planck likelihood, not CAMB/CLASS input, and not a 3D-bulk or early-universe claim"
        ),
    }


def write_cl_ensemble_report(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    rows = collect_cl_runs(run_dirs)
    report = cl_ensemble_report(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cl_ensemble_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "cl_ensemble_rows.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return report


def _field_ensembles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    field_names = sorted({name for row in rows for name in row.get("fields", {})})
    return {name: _one_field_ensemble([row["fields"][name] for row in rows if name in row.get("fields", {})]) for name in field_names}


def _cmb_lite_ensemble(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row.get("cmb_lite", {}) for row in rows if row.get("cmb_lite")]
    return {
        "run_count": len(usable),
        "best_field_counts": _counts(row.get("best_field") for row in usable),
        "best_positive_field_counts": _counts(row.get("best_positive_field") for row in usable),
        "mean_best_shape_correlation": _mean(row.get("best_shape_correlation") for row in usable),
        "mean_best_normalized_rmse": _mean(row.get("best_normalized_rmse") for row in usable),
        "mean_best_positive_shape_correlation": _mean(
            row.get("best_positive_shape_correlation") for row in usable
        ),
        "mean_best_positive_normalized_rmse": _mean(row.get("best_positive_normalized_rmse") for row in usable),
        "claim_boundary": (
            "CMB-lite shape aggregation over gated screen C_l proxies; not a Planck likelihood and "
            "not a physical CMB prediction"
        ),
    }


def _one_field_ensemble(field_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not field_reports:
        return {"run_count": 0}
    spectra = [_d_vector(report.get("spectrum", [])) for report in field_reports]
    min_len = min((vector.size for vector in spectra), default=0)
    spectra = [vector[:min_len] for vector in spectra if vector.size >= min_len and min_len > 0]
    matrix = np.vstack(spectra) if spectra else np.zeros((0, 0), dtype=float)
    peak_ells = [int(report.get("peak_ell")) for report in field_reports if report.get("peak_ell") is not None]
    comparisons = [report.get("control_comparison", {}) for report in field_reports]
    min_deltas = [
        float(comp["min_relative_l2_delta"])
        for comp in comparisons
        if comp.get("min_relative_l2_delta") is not None and math.isfinite(float(comp["min_relative_l2_delta"]))
    ]
    max_corrs = [
        float(comp["max_shape_correlation"])
        for comp in comparisons
        if comp.get("max_shape_correlation") is not None and math.isfinite(float(comp["max_shape_correlation"]))
    ]
    mean_spectrum = np.mean(matrix, axis=0) if matrix.size else np.zeros(0, dtype=float)
    std_spectrum = np.std(matrix, axis=0) if matrix.size else np.zeros(0, dtype=float)
    return {
        "run_count": len(field_reports),
        "peak_ell_mode": _mode(peak_ells),
        "peak_ell_values": peak_ells,
        "mean_pairwise_shape_correlation": _mean_pairwise_correlation(matrix),
        "median_min_relative_l2_delta_to_controls": _median(min_deltas),
        "median_max_shape_correlation_to_controls": _median(max_corrs),
        "mean_total_abs_D_ell_2_plus": float(np.mean(np.sum(np.abs(matrix), axis=1))) if matrix.size else 0.0,
        "mean_spectrum": [
            {"ell": ell + 2, "mean_D_ell": float(mean), "std_D_ell": float(std)}
            for ell, (mean, std) in enumerate(zip(mean_spectrum, std_spectrum))
        ],
    }


def _d_vector(spectrum: list[dict[str, Any]]) -> np.ndarray:
    rows = sorted((row for row in spectrum if int(row.get("ell", -1)) >= 2), key=lambda row: int(row["ell"]))
    return np.asarray([float(row.get("D_ell", 0.0)) for row in rows], dtype=float)


def _mean_pairwise_correlation(matrix: np.ndarray) -> float | None:
    if matrix.shape[0] < 2 or matrix.shape[1] == 0:
        return None
    cors: list[float] = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            corr = _shape_correlation(matrix[i], matrix[j])
            if math.isfinite(corr):
                cors.append(corr)
    return float(np.mean(cors)) if cors else None


def _shape_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("nan")
    count = min(left.size, right.size)
    left = left[:count] - float(np.mean(left[:count]))
    right = right[:count] - float(np.mean(right[:count]))
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denom)


def _mode(values: list[int]) -> int | None:
    if not values:
        return None
    return int(Counter(values).most_common(1)[0][0])


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _mean(values: Any) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return float(np.mean(numeric)) if numeric else None


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_from_config(path: Path) -> int | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("seed:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None
