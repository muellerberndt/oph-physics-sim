from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


REPORT_FILES = {
    "observer_cap_response": "record_populated_h3_report.json",
    "record_family": "record_family_h3_report.json",
    "defect_cluster": "defect_cluster_h3_report.json",
    "defect_h3_worldlines": "defect_h3_worldlines_report.json",
}


def collect_h3_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in run_dirs:
        for status_path in sorted(Path(root).glob("**/emergence_status_report.json")):
            run_path = status_path.parent.resolve()
            if run_path in seen:
                continue
            if not any((run_path / name).exists() for name in REPORT_FILES.values()):
                continue
            seen.add(run_path)
            manifest = _read_json(run_path / "manifest.json")
            status = _read_json(status_path)
            boundary = _read_json(run_path / "boundary_program_report.json")
            reports = {label: _read_json(run_path / filename) for label, filename in REPORT_FILES.items()}
            rows.append(
                {
                    "run_id": manifest.get("run_id", run_path.name),
                    "path": str(run_path),
                    "patch_count": int(manifest.get("patch_count", 0)),
                    "seed": _seed_from_config(run_path / "config.yml"),
                    "boundary_program": boundary.get("mode"),
                    "final_phi": manifest.get("final_phi"),
                    "support_visible_lorentz_3p1_kinematics_receipt": bool(
                        status.get("support_visible_lorentz_3p1_kinematics_receipt", False)
                    ),
                    "conformal_h3_spatial_chart_receipt": bool(status.get("conformal_h3_spatial_chart_receipt", False)),
                    "record_populated_h3_spatial_receipt": bool(status.get("record_populated_h3_spatial_receipt", False)),
                    "record_family_h3_support_receipt": bool(status.get("record_family_h3_support_receipt", False)),
                    "defect_cluster_h3_support_receipt": bool(status.get("defect_cluster_h3_support_receipt", False)),
                    "matter_defect_h3_support_receipt": bool(status.get("matter_defect_h3_support_receipt", False)),
                    "defect_worldline_precursor_receipt": bool(
                        status.get("defect_worldline_precursor_receipt", False)
                    ),
                    "defect_h3_worldline_precursor_receipt": bool(
                        status.get("defect_h3_worldline_precursor_receipt", False)
                    ),
                    "spatial_bulk_3d_reconstruction_receipt": bool(
                        status.get("spatial_bulk_3d_reconstruction_receipt", False)
                    ),
                    "bulk_3d_established": bool(status.get("bulk_3d_established", False)),
                    "reports": reports,
                }
            )
    return rows


def h3_ensemble_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_n: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_n.setdefault(int(row.get("patch_count", 0)), []).append(row)
    sizes = []
    for patch_count, group in sorted(by_n.items()):
        sizes.append(
            {
                "patch_count": patch_count,
                "run_count": len(group),
                "support_visible_lorentz_fraction": _fraction(
                    row.get("support_visible_lorentz_3p1_kinematics_receipt") for row in group
                ),
                "conformal_h3_chart_fraction": _fraction(row.get("conformal_h3_spatial_chart_receipt") for row in group),
                "record_populated_bulk_fraction": _fraction(
                    row.get("record_populated_h3_spatial_receipt") for row in group
                ),
                "record_family_support_fraction": _fraction(row.get("record_family_h3_support_receipt") for row in group),
                "defect_cluster_support_fraction": _fraction(row.get("defect_cluster_h3_support_receipt") for row in group),
                "defect_worldline_precursor_fraction": _fraction(
                    row.get("defect_worldline_precursor_receipt") for row in group
                ),
                "defect_h3_worldline_precursor_fraction": _fraction(
                    row.get("defect_h3_worldline_precursor_receipt") for row in group
                ),
                "spatial_bulk_reconstruction_fraction": _fraction(
                    row.get("spatial_bulk_3d_reconstruction_receipt") for row in group
                ),
                "bulk_3d_established_fraction": _fraction(row.get("bulk_3d_established") for row in group),
                "reports": {
                    label: _report_ensemble(label, [row["reports"].get(label, {}) for row in group])
                    for label in REPORT_FILES
                },
            }
        )
    defect_fractions = [row["defect_cluster_support_fraction"] for row in sizes if row["run_count"]]
    h3_worldline_fractions = [row["defect_h3_worldline_precursor_fraction"] for row in sizes if row["run_count"]]
    record_bulk_fractions = [row["record_populated_bulk_fraction"] for row in sizes if row["run_count"]]
    return {
        "mode": "h3_support_ensemble",
        "run_count": len(rows),
        "sizes": sizes,
        "support_visible_lorentz_all": bool(rows) and all(
            row.get("support_visible_lorentz_3p1_kinematics_receipt") for row in rows
        ),
        "conformal_h3_chart_all": bool(rows) and all(row.get("conformal_h3_spatial_chart_receipt") for row in rows),
        "defect_h3_support_any_scale": bool(defect_fractions) and max(defect_fractions) > 0.0,
        "defect_h3_support_all_scales": bool(defect_fractions) and all(value >= 1.0 for value in defect_fractions),
        "defect_h3_worldline_any_scale": bool(h3_worldline_fractions) and max(h3_worldline_fractions) > 0.0,
        "defect_h3_worldline_all_scales": bool(h3_worldline_fractions)
        and all(value >= 1.0 for value in h3_worldline_fractions),
        "record_populated_bulk_any_scale": bool(record_bulk_fractions) and max(record_bulk_fractions) > 0.0,
        "bulk_3d_established": False,
        "physical_particle_prediction": False,
        "claim_boundary": (
            "ensemble of support-visible H3 chart population receipts. Defect-cluster and H3-worldline "
            "support are matter/particle precursors only; full 3D bulk and particle claims still require "
            "record/object bulk population, transport/fusion controls, and repeated-seed/refinement stability."
        ),
    }


def write_h3_ensemble_report(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    rows = collect_h3_runs(run_dirs)
    report = h3_ensemble_report(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "h3_ensemble_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "h3_ensemble_rows.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return report


def _report_ensemble(label: str, reports: list[dict[str, Any]]) -> dict[str, Any]:
    if label == "defect_h3_worldlines":
        return _worldline_report_ensemble(reports)
    return _one_report_ensemble(reports)


def _one_report_ensemble(reports: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [report for report in reports if report]
    if not reports:
        return {"run_count": 0}
    h3 = _metric(reports, ("h3_fit", "median_residual"))
    s2 = _metric(reports, ("s2_boundary_control", "median_residual"))
    shuffled = _metric(reports, ("shuffled_cap_response_control", "median_residual"))
    support_counts = [float(report.get("support_count")) for report in reports if report.get("support_count") is not None]
    support_medians = [
        float(report.get("support_size_summary", {}).get("median"))
        for report in reports
        if report.get("support_size_summary", {}).get("median") is not None
    ]
    receipts = [bool(report.get("record_populated_h3_receipt", False)) for report in reports]
    return {
        "run_count": len(reports),
        "receipt_count": int(sum(receipts)),
        "receipt_fraction": float(sum(receipts) / max(len(receipts), 1)),
        "median_h3_residual": _median(h3),
        "median_s2_boundary_residual": _median(s2),
        "median_shuffled_residual": _median(shuffled),
        "median_h3_over_s2": _median(_ratios(h3, s2)),
        "median_h3_over_shuffled": _median(_ratios(h3, shuffled)),
        "median_support_count": _median(support_counts),
        "median_support_size_median": _median(support_medians),
    }


def _worldline_report_ensemble(reports: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [report for report in reports if report]
    if not reports:
        return {"run_count": 0}
    h3 = [float(report["median_h3_residual"]) for report in reports if report.get("median_h3_residual") is not None]
    s2 = [
        float(report["median_s2_boundary_residual"])
        for report in reports
        if report.get("median_s2_boundary_residual") is not None
    ]
    shuffled = [
        float(report["median_shuffled_residual"])
        for report in reports
        if report.get("median_shuffled_residual") is not None
    ]
    receipts = [bool(report.get("bulk_worldline_precursor_receipt", False)) for report in reports]
    return {
        "run_count": len(reports),
        "receipt_count": int(sum(receipts)),
        "receipt_fraction": float(sum(receipts) / max(len(receipts), 1)),
        "median_h3_residual": _median(h3),
        "median_s2_boundary_residual": _median(s2),
        "median_shuffled_residual": _median(shuffled),
        "median_h3_over_s2": _median(_ratios(h3, s2)),
        "median_h3_over_shuffled": _median(_ratios(h3, shuffled)),
        "median_event_count": _median([float(report.get("event_count", 0)) for report in reports]),
        "median_worldline_count": _median([float(report.get("worldline_count", 0)) for report in reports]),
        "median_persistent_h3_worldline_count": _median(
            [float(report.get("persistent_h3_worldline_count", 0)) for report in reports]
        ),
    }


def _metric(reports: list[dict[str, Any]], path: tuple[str, str]) -> list[float]:
    values: list[float] = []
    for report in reports:
        value = report.get(path[0], {}).get(path[1])
        if value is not None:
            values.append(float(value))
    return values


def _ratios(numerators: list[float], denominators: list[float]) -> list[float]:
    return [num / den for num, den in zip(numerators, denominators, strict=False) if den > 0.0]


def _fraction(values: Any) -> float:
    values = [bool(value) for value in values]
    return float(sum(values) / max(len(values), 1))


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


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
