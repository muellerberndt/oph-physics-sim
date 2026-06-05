from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def collect_refinement_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in run_dirs:
        for manifest_path in sorted(Path(root).glob("**/manifest.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            state_path = manifest_path.with_name("bw_state_derived_report.json")
            collar_path = manifest_path.with_name("collar_markov_report.json")
            state_report = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            collar_report = json.loads(collar_path.read_text(encoding="utf-8")) if collar_path.exists() else {}
            rows.append(
                {
                    "run_id": manifest.get("run_id"),
                    "path": str(manifest_path.parent),
                    "patch_count": int(manifest.get("patch_count", 0)),
                    "seed": _seed_from_config(manifest_path.with_name("config.yml")),
                    "bw_primary_mode": manifest.get("bw_primary_mode"),
                    "bw_primary_median": float(manifest.get("bw_primary_median", state_report.get("median", math.nan))),
                    "state_derived_median": float(state_report.get("median", math.nan)),
                    "state_derived_corrected_upper_median": float(state_report.get("corrected_upper_median", math.nan)),
                    "epsilon_cmi": float(collar_report.get("median_epsilon_cmi", math.nan)),
                    "r_fr_proxy": 2.0 * math.sqrt(max(float(collar_report.get("median_epsilon_cmi", 0.0)), 0.0)),
                    "observer_consensus": manifest.get("observer_consensus", {}),
                }
            )
    return rows


def refinement_scaling_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_n: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_n.setdefault(int(row["patch_count"]), []).append(row)
    aggregates = []
    for patch_count, group in sorted(by_n.items()):
        residuals = np.array([row["state_derived_median"] for row in group if math.isfinite(row["state_derived_median"])])
        cmi = np.array([row["epsilon_cmi"] for row in group if math.isfinite(row["epsilon_cmi"])])
        aggregates.append(
            {
                "patch_count": patch_count,
                "run_count": len(group),
                "state_derived_median": float(np.median(residuals)) if residuals.size else float("nan"),
                "epsilon_cmi_median": float(np.median(cmi)) if cmi.size else float("nan"),
            }
        )
    xs = np.array([row["patch_count"] for row in aggregates if row["patch_count"] > 0 and math.isfinite(row["state_derived_median"])])
    ys = np.array([row["state_derived_median"] for row in aggregates if row["patch_count"] > 0 and math.isfinite(row["state_derived_median"])])
    if xs.size >= 2 and np.all(ys > 0.0):
        slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
    else:
        slope, intercept = float("nan"), float("nan")
    numerical_floor = bool(ys.size > 0 and float(np.nanmedian(ys)) < 1e-10)
    return {
        "mode": "state_derived_refinement_scaling",
        "run_count": len(rows),
        "sizes": aggregates,
        "log_residual_vs_log_n_slope": float(slope),
        "log_residual_vs_log_n_intercept": float(intercept),
        "slope_negative": bool(math.isfinite(float(slope)) and float(slope) < 0.0),
        "numerical_floor_detected": numerical_floor,
        "slope_interpretation": "not_meaningful_at_numerical_floor" if numerical_floor else "diagnostic_only",
        "claim_boundary": (
            "refinement aggregation diagnostic; no theorem-grade claim without controls and bootstrap CI. "
            "If numerical_floor_detected is true, residual slope is an implementation sanity check, "
            "not empirical convergence evidence."
        ),
    }


def write_refinement_report(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    rows = collect_refinement_runs(run_dirs)
    report = refinement_scaling_report(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "refinement_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "refinement_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return report


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
