from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any


RELEVANT_REPORTS = (
    "modular_response_h3_report.json",
    "observer_chart_object_h3_report.json",
    "cmb_lite_comparison_report.json",
    "cl_comparison_report.json",
    "array_holonomy_report.json",
    "emergence_status_report.json",
)


def collect_comparable_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = [_extract_run_row(path) for path in _find_run_dirs(run_dirs)]
    return [row for row in rows if row.get("has_comparable_data")]


def comparable_data_report(run_dirs: list[Path]) -> dict[str, Any]:
    rows = collect_comparable_runs(run_dirs)
    return {
        "mode": "oph_fpe_comparable_data_snapshot",
        "run_count": len(rows),
        "measurement_lanes": {
            "planck_tt_shape_lite": _planck_lite_summary(rows),
            "h3_modular_response_controls": _h3_summary(rows),
            "observer_chart_object_population": _observer_chart_summary(rows),
            "screen_holonomy_defect_proxy": _holonomy_summary(rows),
        },
        "rows": rows,
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "bulk_3d_established": any(bool(row.get("bulk_3d_established")) for row in rows),
        "claim_boundary": (
            "Comparable-data snapshot for current OPH-FPE receipts. Planck comparisons are shape-only "
            "C_l diagnostics with normalized axes and amplitude rescaling. H3 values are internal "
            "modular-response-vs-control receipts. Defect values are screen/collar holonomy proxies. "
            "This is not a physical CMB prediction, not a P(k), not a Boltzmann likelihood, and not a "
            "completed 3D-bulk or particle-emergence result."
        ),
    }


def write_comparable_data_package(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    report = comparable_data_report(run_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparable_data_snapshot.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    _write_rows_csv(out_dir / "comparable_data_rows.csv", report["rows"])
    (out_dir / "comparable_data_snapshot.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _find_run_dirs(roots: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        root = Path(root)
        if any((root / name).exists() for name in RELEVANT_REPORTS):
            paths.add(root)
        if root.exists():
            for name in RELEVANT_REPORTS:
                for report_path in root.glob(f"**/{name}"):
                    paths.add(report_path.parent)
    return sorted(paths, key=lambda path: str(path))


def _extract_run_row(run_path: Path) -> dict[str, Any]:
    manifest = _read_json(run_path / "manifest.json")
    h3 = _read_json(run_path / "modular_response_h3_report.json")
    cmb = _read_json(run_path / "cmb_lite_comparison_report.json")
    cl = _read_json(run_path / "cl_comparison_report.json")
    hol = _read_json(run_path / "array_holonomy_report.json")
    emergence = _read_json(run_path / "emergence_status_report.json")
    object_chart = _read_json(run_path / "observer_chart_object_h3_report.json")

    h3_fit = h3.get("h3_fit", {}) if h3 else {}
    s2 = h3.get("s2_boundary_control", {}) if h3 else {}
    controls = h3.get("control_fits", {}) if h3 else {}
    wrong = h3.get("wrong_scale_control_fits", {}) if h3 else {}
    cmb_fields = cmb.get("field_comparisons", {}) if cmb else {}
    cmb_best_name = cmb.get("best_shape_field") if cmb else None
    cmb_best = cmb_fields.get(cmb_best_name, {}) if cmb_best_name else {}
    record_cmb = cmb_fields.get("record_signature", {})
    cl_record = (cl.get("fields", {}) if cl else {}).get("record_signature", {})
    cl_record_control = cl_record.get("control_comparison", {})

    row = {
        "run_path": str(run_path),
        "run_id": manifest.get("run_id") or run_path.name,
        "name": manifest.get("name"),
        "patch_count": manifest.get("patch_count") or (cl or {}).get("point_count"),
        "has_comparable_data": any((h3, cmb, cl, hol)),
        "bulk_3d_established": bool(emergence.get("bulk_3d_established", False)),
        "physical_cmb_prediction": bool(cmb.get("physical_cmb_prediction", False)),
        "h3_receipt": bool(h3.get("h3_bulk_candidate_receipt", h3.get("MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", False))),
        "h3_rmse": h3_fit.get("heldout_normalized_rmse"),
        "h3_explained_variance": h3_fit.get("heldout_explained_variance"),
        "s2_boundary_rmse": s2.get("heldout_normalized_rmse"),
        "shuffled_response_rmse": _nested(controls, "shuffled_response", "heldout_normalized_rmse"),
        "shuffled_observer_labels_rmse": _nested(controls, "shuffled_observer_labels", "heldout_normalized_rmse"),
        "no_perturbation_rmse": _nested(controls, "no_perturbation", "heldout_normalized_rmse"),
        "wrong_1x_rmse": _nested(wrong, "1x", "heldout_normalized_rmse"),
        "wrong_pi_rmse": _nested(wrong, "pi", "heldout_normalized_rmse"),
        "wrong_4pi_rmse": _nested(wrong, "4pi", "heldout_normalized_rmse"),
        "observer_chart_object_receipt": object_chart.get("observer_chart_object_h3_receipt"),
        "observer_chart_bulk_population_receipt": object_chart.get("observer_chart_bulk_population_receipt"),
        "observer_chart_object_count": object_chart.get("object_count"),
        "observer_chart_h3_compactness": object_chart.get("median_h3_compactness_normalized"),
        "observer_chart_s2_compactness": object_chart.get("median_s2_boundary_compactness_normalized"),
        "observer_chart_shuffled_h3_compactness": object_chart.get("median_shuffled_h3_compactness_normalized"),
        "observer_chart_h3_beats_shuffled": object_chart.get("h3_beats_shuffled_incidence"),
        "observer_chart_h3_not_boundary_dominated": object_chart.get("h3_not_boundary_dominated"),
        "cmb_benchmark": (cmb.get("benchmark", {}) if cmb else {}).get("label"),
        "cmb_best_field": cmb_best_name,
        "cmb_best_shape_correlation": cmb_best.get("shape_correlation"),
        "cmb_best_normalized_rmse": cmb_best.get("normalized_rmse"),
        "cmb_best_peak_fraction_delta": cmb_best.get("peak_fraction_delta"),
        "record_signature_shape_correlation": record_cmb.get("shape_correlation"),
        "record_signature_normalized_rmse": record_cmb.get("normalized_rmse"),
        "record_signature_peak_fraction_delta": record_cmb.get("peak_fraction_delta"),
        "record_signature_peak_ell": cl_record.get("peak_ell"),
        "record_signature_total_abs_D_ell": cl_record.get("total_abs_D_ell_2_plus"),
        "record_signature_control_min_l2_delta": cl_record_control.get("min_relative_l2_delta"),
        "freezeout_cycle": cl.get("freezeout_cycle") if cl else None,
        "committed_fraction": cl.get("committed_fraction") if cl else None,
        "holonomy_triangle_count": hol.get("triangle_count") if hol else None,
        "holonomy_defect_fraction": hol.get("defect_fraction") if hol else None,
        "holonomy_cluster_count": hol.get("cluster_count") if hol else None,
    }
    return row


def _planck_lite_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("cmb_best_normalized_rmse") is not None]
    return {
        "run_count": len(usable),
        "best_field_counts": _counts(row.get("cmb_best_field") for row in usable),
        "mean_best_shape_correlation": _mean(row.get("cmb_best_shape_correlation") for row in usable),
        "mean_best_normalized_rmse": _mean(row.get("cmb_best_normalized_rmse") for row in usable),
        "mean_record_signature_shape_correlation": _mean(
            row.get("record_signature_shape_correlation") for row in usable
        ),
        "mean_record_signature_normalized_rmse": _mean(row.get("record_signature_normalized_rmse") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Currently useful as a negative/diagnostic Planck-lite screen comparison. The best low-RMSE "
            "fields are typically anticorrelated with Planck TT; record_signature is weakly positive."
        ),
    }


def _h3_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("h3_rmse") is not None]
    return {
        "run_count": len(usable),
        "receipt_count": sum(1 for row in usable if row.get("h3_receipt")),
        "mean_h3_rmse": _mean(row.get("h3_rmse") for row in usable),
        "mean_h3_explained_variance": _mean(row.get("h3_explained_variance") for row in usable),
        "mean_s2_boundary_rmse": _mean(row.get("s2_boundary_rmse") for row in usable),
        "mean_shuffled_response_rmse": _mean(row.get("shuffled_response_rmse") for row in usable),
        "mean_wrong_pi_rmse": _mean(row.get("wrong_pi_rmse") for row in usable),
        "interpretation": (
            "This is the most useful current comparable signal: H3 modular-response residuals beat "
            "implemented controls, but explained variance is still weak."
        ),
    }


def _observer_chart_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("observer_chart_h3_compactness") is not None]
    return {
        "run_count": len(usable),
        "object_chart_receipt_count": sum(1 for row in usable if row.get("observer_chart_object_receipt")),
        "bulk_population_receipt_count": sum(1 for row in usable if row.get("observer_chart_bulk_population_receipt")),
        "mean_object_count": _mean(row.get("observer_chart_object_count") for row in usable),
        "mean_h3_compactness": _mean(row.get("observer_chart_h3_compactness") for row in usable),
        "mean_s2_boundary_compactness": _mean(row.get("observer_chart_s2_compactness") for row in usable),
        "mean_shuffled_h3_compactness": _mean(row.get("observer_chart_shuffled_h3_compactness") for row in usable),
        "h3_beats_shuffled_count": sum(1 for row in usable if row.get("observer_chart_h3_beats_shuffled")),
        "h3_not_boundary_dominated_count": sum(
            1 for row in usable if row.get("observer_chart_h3_not_boundary_dominated")
        ),
        "interpretation": (
            "Observer-facing objects currently localize in the sampled H3 observer chart better than "
            "shuffled observer-object incidence, but the population is still boundary-dominated unless "
            "the H3 compactness also stops being worse than the S2 observer-axis compactness."
        ),
    }


def _holonomy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("holonomy_triangle_count") is not None]
    return {
        "run_count": len(usable),
        "mean_defect_fraction": _mean(row.get("holonomy_defect_fraction") for row in usable),
        "mean_cluster_count": _mean(row.get("holonomy_cluster_count") for row in usable),
        "interpretation": (
            "Screen/collar S3 holonomy defect statistics are comparable across runs and controls. They are "
            "not matter-particle observables until persistent worldlines pass in a neutral 3D reconstruction."
        ),
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(report: dict[str, Any]) -> str:
    lanes = report["measurement_lanes"]
    h3 = lanes["h3_modular_response_controls"]
    object_chart = lanes["observer_chart_object_population"]
    planck = lanes["planck_tt_shape_lite"]
    hol = lanes["screen_holonomy_defect_proxy"]
    lines = [
        "# OPH-FPE Comparable Data Snapshot",
        "",
        report["claim_boundary"],
        "",
        "## Current Answer",
        "",
        "Yes: the current simulator can emit diagnostic, measurement-facing values. No: it does not yet emit a physical CMB prediction, physical P(k), populated 3D bulk, or particle spectrum.",
        "",
        "## H3 Modular-Response Receipt",
        "",
        f"- runs with H3 fits: {h3['run_count']}",
        f"- H3 receipts: {h3['receipt_count']}",
        f"- mean H3 RMSE: {_fmt(h3['mean_h3_rmse'])}",
        f"- mean H3 explained variance: {_fmt(h3['mean_h3_explained_variance'])}",
        f"- mean S2-boundary RMSE: {_fmt(h3['mean_s2_boundary_rmse'])}",
        f"- mean shuffled-response RMSE: {_fmt(h3['mean_shuffled_response_rmse'])}",
        "",
        "## Observer-Chart Object Population",
        "",
        f"- runs with object-chart reports: {object_chart['run_count']}",
        f"- object-chart receipts: {object_chart['object_chart_receipt_count']}",
        f"- bulk-population receipts: {object_chart['bulk_population_receipt_count']}",
        f"- mean object count: {_fmt(object_chart['mean_object_count'])}",
        f"- mean H3 compactness: {_fmt(object_chart['mean_h3_compactness'])}",
        f"- mean S2-boundary compactness: {_fmt(object_chart['mean_s2_boundary_compactness'])}",
        f"- mean shuffled-H3 compactness: {_fmt(object_chart['mean_shuffled_h3_compactness'])}",
        "",
        "## Planck-Lite Screen C_l Shape",
        "",
        f"- runs with CMB-lite comparisons: {planck['run_count']}",
        f"- best-field counts: {planck['best_field_counts']}",
        f"- mean best-field correlation: {_fmt(planck['mean_best_shape_correlation'])}",
        f"- mean best-field RMSE: {_fmt(planck['mean_best_normalized_rmse'])}",
        f"- mean record-signature correlation: {_fmt(planck['mean_record_signature_shape_correlation'])}",
        f"- mean record-signature RMSE: {_fmt(planck['mean_record_signature_normalized_rmse'])}",
        "",
        "## Screen Holonomy Defect Proxy",
        "",
        f"- runs with holonomy reports: {hol['run_count']}",
        f"- mean defect fraction: {_fmt(hol['mean_defect_fraction'])}",
        f"- mean cluster count: {_fmt(hol['mean_cluster_count'])}",
        "",
        "## Output Files",
        "",
        "- `comparable_data_snapshot.json`",
        "- `comparable_data_rows.csv`",
        "- `comparable_data_snapshot.md`",
        "",
    ]
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _nested(data: dict[str, Any], first: str, second: str) -> Any:
    value = data.get(first, {})
    if not isinstance(value, dict):
        return None
    return value.get(second)


def _mean(values: Any) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return fmean(numeric) if numeric else None


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.10g}"
    return "n/a"
