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
            transition_path = manifest_path.with_name("transition_scale_selection_report.json")
            neutral_path = manifest_path.with_name("bulk_reconstruction_report.json")
            object_chart_path = manifest_path.with_name("observer_chart_object_h3_report.json")
            cmb_path = manifest_path.with_name("cmb_lite_comparison_report.json")
            state_report = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            collar_report = json.loads(collar_path.read_text(encoding="utf-8")) if collar_path.exists() else {}
            transition_report = (
                json.loads(transition_path.read_text(encoding="utf-8")) if transition_path.exists() else {}
            )
            neutral_report = json.loads(neutral_path.read_text(encoding="utf-8")) if neutral_path.exists() else {}
            object_chart_report = (
                json.loads(object_chart_path.read_text(encoding="utf-8")) if object_chart_path.exists() else {}
            )
            cmb_report = json.loads(cmb_path.read_text(encoding="utf-8")) if cmb_path.exists() else {}
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
                    "state_derived_correct_beats_controls": bool(state_report.get("correct_beats_controls", False)),
                    "state_selected_scale_label": state_report.get("state_selected_scale_label"),
                    "state_selected_2pi": bool(state_report.get("state_selected_2pi", False)),
                    "state_best_control": state_report.get("best_control"),
                    "state_best_control_median": _optional_float(state_report.get("best_control_median")),
                    "state_control_medians": state_report.get("control_medians", {}),
                    "epsilon_cmi": float(collar_report.get("median_epsilon_cmi", math.nan)),
                    "r_fr_proxy": 2.0 * math.sqrt(max(float(collar_report.get("median_epsilon_cmi", 0.0)), 0.0)),
                    "transition_primary_source": transition_report.get("primary_source"),
                    "transition_selected_label": transition_report.get("selected_label"),
                    "transition_two_pi_selected": bool(transition_report.get("two_pi_selected", False)),
                    "transition_two_pi_over_best": _optional_float(transition_report.get("two_pi_over_best")),
                    "neutral_bulk_3d_established": bool(neutral_report.get("bulk_3d_established", False)),
                    "neutral_candidate_3d_dimension_window": bool(
                        neutral_report.get("candidate_3d_dimension_window", False)
                    ),
                    "neutral_primary_dimension": _optional_float(neutral_report.get("primary_dimension_estimate")),
                    "neutral_control_gate_passed": bool(neutral_report.get("control_gate_passed", False)),
                    "observer_chart_bulk_population_receipt": bool(
                        object_chart_report.get("observer_chart_bulk_population_receipt", False)
                    ),
                    "observer_chart_localized_nonboundary_bulk_population_receipt": bool(
                        object_chart_report.get("localized_nonboundary_bulk_population_receipt", False)
                    ),
                    "observer_chart_object_h3_receipt": bool(
                        object_chart_report.get("observer_chart_object_h3_receipt", False)
                    ),
                    "cmb_best_positive_shape_correlation": _cmb_best_positive_shape_correlation(cmb_report),
                    "physical_cmb_prediction": bool(cmb_report.get("physical_cmb_prediction", False)),
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
                "state_controls_pass_count": int(
                    sum(1 for row in group if row.get("state_derived_correct_beats_controls"))
                ),
                "state_2pi_selected_count": int(sum(1 for row in group if row.get("state_selected_2pi"))),
                "state_selected_scale_counts": _value_counts(row.get("state_selected_scale_label") for row in group),
                "epsilon_cmi_median": float(np.median(cmi)) if cmi.size else float("nan"),
                "transition_2pi_selected_count": int(sum(1 for row in group if row.get("transition_two_pi_selected"))),
                "transition_selected_scale_counts": _value_counts(
                    row.get("transition_selected_label") for row in group
                ),
                "neutral_bulk_3d_established_count": int(
                    sum(1 for row in group if row.get("neutral_bulk_3d_established"))
                ),
                "neutral_primary_dimension_median": _median_optional(
                    row.get("neutral_primary_dimension") for row in group
                ),
                "observer_chart_bulk_population_count": int(
                    sum(1 for row in group if row.get("observer_chart_bulk_population_receipt"))
                ),
                "observer_chart_localized_nonboundary_bulk_population_count": int(
                    sum(1 for row in group if row.get("observer_chart_localized_nonboundary_bulk_population_receipt"))
                ),
                "cmb_best_positive_shape_correlation_median": _median_optional(
                    row.get("cmb_best_positive_shape_correlation") for row in group
                ),
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
        "state_control_pass_total": int(sum(1 for row in rows if row.get("state_derived_correct_beats_controls"))),
        "state_2pi_selected_total": int(sum(1 for row in rows if row.get("state_selected_2pi"))),
        "transition_2pi_selected_total": int(sum(1 for row in rows if row.get("transition_two_pi_selected"))),
        "bulk_3d_established_total": int(sum(1 for row in rows if row.get("neutral_bulk_3d_established"))),
        "physical_cmb_prediction_total": int(sum(1 for row in rows if row.get("physical_cmb_prediction"))),
        "slope_interpretation": "not_meaningful_at_numerical_floor" if numerical_floor else "diagnostic_only",
        "claim_boundary": (
            "refinement aggregation diagnostic; no theorem-grade claim without controls and bootstrap CI. "
            "If numerical_floor_detected is true, residual slope is an implementation sanity check, "
            "not empirical convergence evidence. CMB entries are screen-proxy comparisons only unless "
            "physical_cmb_prediction_total is nonzero."
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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _median_optional(values: Any) -> float | None:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(numeric)) if numeric else None


def _value_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _cmb_best_positive_shape_correlation(report: dict[str, Any]) -> float | None:
    field_name = report.get("best_positive_shape_field")
    comparisons = report.get("field_comparisons", {})
    if not field_name or not isinstance(comparisons, dict):
        return None
    field_report = comparisons.get(str(field_name), {})
    if not isinstance(field_report, dict):
        return None
    return _optional_float(field_report.get("shape_correlation"))
