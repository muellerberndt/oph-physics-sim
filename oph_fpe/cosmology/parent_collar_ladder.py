from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPORT_NAME = "parent_collar_ladder_report.json"


def parent_collar_ladder_report(
    run_dirs: list[Path],
    *,
    min_n_values: int = 3,
    max_final_p90_epsilon_cmi: float = 0.05,
    max_final_p90_epsilon_cmi_per_collar_patch: float = 0.002,
    max_final_p90_r_fr: float = 0.5,
    max_slope_log_cmi_vs_log_n: float = -0.2,
    max_slope_log_cmi_density_vs_log_n: float = -0.1,
) -> dict[str, Any]:
    """Aggregate diagonal collar-Markov reports into a finite-regulator ladder.

    This is intentionally conservative. The diagonal collar state is a useful
    finite support-visible receipt, but it is not the noncommutative parent
    collar/recovery theorem object by itself. A pass here only means the cached
    finite reports have the right scaling shape to become theorem inputs.
    """

    reports = _load_collar_reports(run_dirs)
    rows = [_row_from_collar_report(path, report, item) for path, report in reports for item in report.get("rows", [])]
    rows = [row for row in rows if row.get("patch_count") is not None and row.get("epsilon_cmi") is not None]
    cap_family = _cap_family_summary(reports)
    by_n = _aggregate_by_n(rows)
    n_values = sorted(by_n)
    p90_values = [by_n[n]["p90_epsilon_cmi"] for n in n_values if by_n[n].get("p90_epsilon_cmi") is not None]
    p90_density_values = [
        by_n[n]["p90_epsilon_cmi_per_collar_patch"]
        for n in n_values
        if by_n[n].get("p90_epsilon_cmi_per_collar_patch") is not None
    ]
    slope = _log_slope(
        [float(n) for n in n_values if by_n[n].get("p90_epsilon_cmi") is not None],
        p90_values,
    )
    density_slope = _log_slope(
        [float(n) for n in n_values if by_n[n].get("p90_epsilon_cmi_per_collar_patch") is not None],
        p90_density_values,
    )
    final = by_n[n_values[-1]] if n_values else {}
    regulator_ladder_ready = len(n_values) >= min_n_values
    cmi_scaling_improves = slope is not None and slope < max_slope_log_cmi_vs_log_n
    cmi_density_scaling_improves = (
        density_slope is not None and density_slope < max_slope_log_cmi_density_vs_log_n
    )
    final_cmi_ok = (
        final.get("p90_epsilon_cmi") is not None
        and float(final["p90_epsilon_cmi"]) <= max_final_p90_epsilon_cmi
    )
    final_cmi_density_ok = (
        final.get("p90_epsilon_cmi_per_collar_patch") is not None
        and float(final["p90_epsilon_cmi_per_collar_patch"]) <= max_final_p90_epsilon_cmi_per_collar_patch
    )
    final_recovery_ok = (
        final.get("p90_r_fr_bound") is not None
        and float(final["p90_r_fr_bound"]) <= max_final_p90_r_fr
    )
    local_density_pass = bool(regulator_ladder_ready and cmi_density_scaling_improves and final_cmi_density_ok)
    strict_local_density_pass = bool(local_density_pass and cap_family["strict_cap_family_matched"])
    scaling_pass = bool(
        regulator_ladder_ready
        and cmi_scaling_improves
        and cmi_density_scaling_improves
        and final_cmi_ok
        and final_cmi_density_ok
        and final_recovery_ok
        and cap_family["strict_cap_family_matched"]
    )
    diagnosis = _diagnosis(
        regulator_ladder_ready=regulator_ladder_ready,
        cmi_scaling_improves=cmi_scaling_improves,
        cmi_density_scaling_improves=cmi_density_scaling_improves,
        final_cmi_ok=final_cmi_ok,
        final_cmi_density_ok=final_cmi_density_ok,
        final_recovery_ok=final_recovery_ok,
        slope=slope,
        density_slope=density_slope,
        final=final,
    )
    return {
        "mode": "oph_parent_collar_recovery_ladder_v0",
        "report_count": len(reports),
        "row_count": len(rows),
        "patch_counts": n_values,
        "cap_family": cap_family,
        "by_patch_count": {str(n): by_n[n] for n in n_values},
        "scaling": {
            "x": "log(patch_count)",
            "y": "log(p90_epsilon_cmi)",
            "slope": slope,
            "max_allowed_slope": max_slope_log_cmi_vs_log_n,
            "improves_with_refinement": cmi_scaling_improves,
        },
        "local_density_scaling": {
            "x": "log(patch_count)",
            "y": "log(p90_epsilon_cmi_per_collar_patch)",
            "slope": density_slope,
            "max_allowed_slope": max_slope_log_cmi_density_vs_log_n,
            "improves_with_refinement": cmi_density_scaling_improves,
        },
        "thresholds": {
            "min_n_values": min_n_values,
            "max_final_p90_epsilon_cmi": max_final_p90_epsilon_cmi,
            "max_final_p90_epsilon_cmi_per_collar_patch": max_final_p90_epsilon_cmi_per_collar_patch,
            "max_final_p90_r_fr": max_final_p90_r_fr,
        },
        "compiler_ready": bool(rows),
        "regulator_ladder_ready": regulator_ladder_ready,
        "cmi_scaling_improves": cmi_scaling_improves,
        "cmi_density_scaling_improves": cmi_density_scaling_improves,
        "final_p90_epsilon_cmi_below_threshold": final_cmi_ok,
        "final_p90_epsilon_cmi_per_collar_patch_below_threshold": final_cmi_density_ok,
        "final_p90_recovery_bound_below_threshold": final_recovery_ok,
        "local_recovery_density_receipt": local_density_pass,
        "strict_local_recovery_density_receipt": strict_local_density_pass,
        "parent_collar_recovery_ladder_receipt": scaling_pass,
        "theorem_grade_parent_collar_ladder": False,
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "diagnosis": diagnosis,
        "rows": rows,
        "claim_boundary": (
            "Classical diagonal collar-Markov refinement diagnostic from cached finite OPH-FPE runs. "
            "It audits whether raw cap/collar CMI, CMI per collar patch, and Fawzi-Renner proxy errors "
            "improve with regulator size. "
            "It is not the noncommutative parent-collar theorem object, not a physical CMB prediction, "
            "and not a 3D bulk proof. A failing or nonmonotone ladder is valid negative evidence about "
            "the current finite implementation."
        ),
    }


def write_parent_collar_ladder_report(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    report = parent_collar_ladder_report(run_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "parent_collar_ladder_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _load_collar_reports(run_dirs: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    paths: set[Path] = set()
    for root in run_dirs:
        root = Path(root)
        if root.is_file() and root.name == "collar_markov_report.json":
            paths.add(root)
            continue
        if root.is_file() and root.name == REPORT_NAME:
            continue
        if (root / "collar_markov_report.json").exists():
            paths.add(root / "collar_markov_report.json")
        if root.exists() and root.is_dir():
            paths.update(root.glob("**/collar_markov_report.json"))
    reports: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(paths, key=lambda item: str(item)):
        report = _read_json(path)
        if report:
            reports.append((path, report))
    return reports


def _cap_family_summary(reports: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    signatures: list[dict[str, Any]] = []
    theta_multisets: set[tuple[float, ...]] = set()
    unique_theta_sets: set[tuple[float, ...]] = set()
    cap_counts: set[int] = set()
    collar_width_uv_values: list[float] = []
    for path, report in reports:
        rows = report.get("rows", []) if isinstance(report, dict) else []
        theta_values = [
            round(float(row.get("theta0")), 8)
            for row in rows
            if _float_or_none(row.get("theta0")) is not None
        ]
        theta_multiset = tuple(sorted(theta_values))
        unique_theta_set = tuple(sorted(set(theta_multiset)))
        cap_count = int(report.get("cap_count") or len(rows))
        cap_counts.add(cap_count)
        theta_multisets.add(theta_multiset)
        unique_theta_sets.add(unique_theta_set)
        for row in rows:
            patch_count = _float_or_none(row.get("sample_count"))
            collar_width = _float_or_none(row.get("collar_width"))
            if patch_count and collar_width is not None:
                collar_width_uv_values.append(float(collar_width) * math.sqrt(float(patch_count)))
        signatures.append(
            {
                "report_path": str(path),
                "patch_count": rows[0].get("sample_count") if rows else None,
                "cap_count": cap_count,
                "theta_multiset": theta_multiset,
                "unique_theta_set": unique_theta_set,
            }
        )
    return {
        "strict_cap_family_matched": bool(reports and len(theta_multisets) == 1 and len(cap_counts) == 1),
        "unique_theta_family_matched": bool(reports and len(unique_theta_sets) == 1),
        "cap_count_values": sorted(cap_counts),
        "theta_multiset_count": len(theta_multisets),
        "unique_theta_set_count": len(unique_theta_sets),
        "mean_collar_width_over_uv_spacing": _mean(collar_width_uv_values),
        "report_signatures": signatures,
        "claim_boundary": (
            "strict_cap_family_matched requires identical cap count and theta multiset across regulator sizes. "
            "unique_theta_family_matched permits repeated caps with the same unique theta values and is "
            "exploratory, not theorem-grade."
        ),
    }


def _row_from_collar_report(path: Path, report: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    patch_count = item.get("sample_count") or report.get("sample_count") or _manifest_patch_count(path.parent)
    epsilon = _float_or_none(item.get("epsilon_cmi"))
    r_fr = _float_or_none(item.get("r_fr_bound"))
    if r_fr is None and epsilon is not None:
        r_fr = 2.0 * math.sqrt(max(epsilon, 0.0))
    collar_width = _float_or_none(item.get("collar_width"))
    ell_uv = 1.0 / math.sqrt(float(patch_count)) if patch_count else None
    collar_count = _float_or_none(item.get("collar_count"))
    return {
        "run_path": str(path.parent),
        "report_path": str(path),
        "patch_count": int(patch_count) if patch_count else None,
        "cap_id": item.get("cap_id"),
        "theta0": _float_or_none(item.get("theta0")),
        "collar_width": collar_width,
        "collar_width_over_uv_spacing": (collar_width / ell_uv) if collar_width is not None and ell_uv else None,
        "inside_count": item.get("inside_count"),
        "collar_count": item.get("collar_count"),
        "outside_count": item.get("outside_count"),
        "epsilon_cmi": epsilon,
        "epsilon_cmi_per_collar_patch": (epsilon / collar_count) if epsilon is not None and collar_count else None,
        "r_fr_bound": r_fr,
        "packet_alphabet_size": item.get("packet_alphabet_size"),
        "triplet_count": item.get("triplet_count"),
        "source_mode": report.get("mode"),
    }


def _aggregate_by_n(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_n: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        patch_count = row.get("patch_count")
        if patch_count is None:
            continue
        by_n.setdefault(int(patch_count), []).append(row)
    return {n: _aggregate_rows(items) for n, items in sorted(by_n.items())}


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eps = [_float_or_none(row.get("epsilon_cmi")) for row in rows]
    eps = [value for value in eps if value is not None]
    eps_density = [_float_or_none(row.get("epsilon_cmi_per_collar_patch")) for row in rows]
    eps_density = [value for value in eps_density if value is not None]
    r_fr = [_float_or_none(row.get("r_fr_bound")) for row in rows]
    r_fr = [value for value in r_fr if value is not None]
    collar_counts = [_float_or_none(row.get("collar_count")) for row in rows]
    collar_counts = [value for value in collar_counts if value is not None]
    patch_count = int(rows[0]["patch_count"]) if rows else None
    return {
        "patch_count": patch_count,
        "run_count": len({row.get("run_path") for row in rows}),
        "cap_row_count": len(rows),
        "median_epsilon_cmi": _percentile(eps, 50.0),
        "mean_epsilon_cmi": _mean(eps),
        "p90_epsilon_cmi": _percentile(eps, 90.0),
        "median_epsilon_cmi_per_collar_patch": _percentile(eps_density, 50.0),
        "mean_epsilon_cmi_per_collar_patch": _mean(eps_density),
        "p90_epsilon_cmi_per_collar_patch": _percentile(eps_density, 90.0),
        "median_r_fr_bound": _percentile(r_fr, 50.0),
        "mean_r_fr_bound": _mean(r_fr),
        "p90_r_fr_bound": _percentile(r_fr, 90.0),
        "mean_collar_count": _mean(collar_counts),
    }


def _diagnosis(
    *,
    regulator_ladder_ready: bool,
    cmi_scaling_improves: bool,
    cmi_density_scaling_improves: bool,
    final_cmi_ok: bool,
    final_cmi_density_ok: bool,
    final_recovery_ok: bool,
    slope: float | None,
    density_slope: float | None,
    final: dict[str, Any],
) -> str:
    if not regulator_ladder_ready:
        return "insufficient regulator sizes for a parent-collar refinement ladder"
    if not cmi_scaling_improves and cmi_density_scaling_improves and final_cmi_density_ok:
        return (
            "raw collar CMI does not improve, but CMI per collar patch does improve; this is a "
            "positive local-density diagnostic and not yet a theorem-grade parent-collar receipt "
            f"(raw slope={slope}, density slope={density_slope})"
        )
    if not cmi_scaling_improves:
        return (
            "collar CMI does not improve with refinement; current cached runs are not theorem-grade "
            f"parent-collar inputs (log-log slope={slope})"
        )
    if not final_cmi_ok:
        return (
            "collar CMI improves but final p90 epsilon_CMI is still too large for theorem-grade input "
            f"(final p90={final.get('p90_epsilon_cmi')})"
        )
    if not final_recovery_ok:
        return (
            "collar CMI improves but final Fawzi-Renner proxy bound is still too large "
            f"(final p90={final.get('p90_r_fr_bound')})"
        )
    return "classical collar ladder passes its conservative finite-regulator thresholds"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# OPH Parent-Collar Recovery Ladder",
        "",
        report["claim_boundary"],
        "",
        f"- collar reports: {report['report_count']}",
        f"- cap rows: {report['row_count']}",
        f"- patch counts: {report['patch_counts']}",
        f"- strict cap family matched: {report['cap_family']['strict_cap_family_matched']}",
        f"- unique theta family matched: {report['cap_family']['unique_theta_family_matched']}",
        f"- compiler ready: {report['compiler_ready']}",
        f"- regulator ladder ready: {report['regulator_ladder_ready']}",
        f"- CMI scaling improves: {report['cmi_scaling_improves']}",
        f"- CMI density scaling improves: {report['cmi_density_scaling_improves']}",
        f"- local recovery-density receipt: {report['local_recovery_density_receipt']}",
        f"- strict local recovery-density receipt: {report['strict_local_recovery_density_receipt']}",
        f"- parent-collar recovery ladder receipt: {report['parent_collar_recovery_ladder_receipt']}",
        f"- theorem-grade parent-collar ladder: {report['theorem_grade_parent_collar_ladder']}",
        f"- physical CMB prediction: {report['physical_cmb_prediction']}",
        f"- diagnosis: {report['diagnosis']}",
        "",
        "## By Patch Count",
        "",
        "| N | cap rows | median epsilon_CMI | p90 epsilon_CMI | p90 epsilon_CMI/collar | p90 r_FR |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n, item in report["by_patch_count"].items():
        lines.append(
            "| "
            f"{n} | {item['cap_row_count']} | {_fmt(item['median_epsilon_cmi'])} | "
            f"{_fmt(item['p90_epsilon_cmi'])} | {_fmt(item['p90_epsilon_cmi_per_collar_patch'])} | "
            f"{_fmt(item['p90_r_fr_bound'])} |"
        )
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _manifest_patch_count(run_dir: Path) -> int | None:
    manifest = _read_json(run_dir / "manifest.json")
    value = manifest.get("patch_count")
    return int(value) if isinstance(value, (int, float)) else None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _log_slope(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys, strict=False) if x > 0.0 and y > 0.0]
    if len(pairs) < 2:
        return None
    log_x = np.log(np.asarray([x for x, _ in pairs], dtype=np.float64))
    log_y = np.log(np.asarray([y for _, y in pairs], dtype=np.float64))
    return float(np.polyfit(log_x, log_y, deg=1)[0])


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:.6g}"
