from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BAKernelRow:
    k_bin: float
    a_bin: float
    B_A: float
    sem: float
    sample_count: int
    control_B_A: float | None
    control_sem: float | None
    z_score: float | None


def estimate_B_A_from_paired_runs(
    base_rows: list[dict[str, Any]],
    perturbed_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]] | None = None,
    *,
    min_delta_baryon: float = 1.0e-12,
) -> list[BAKernelRow]:
    """Estimate B_A(k,a) from paired finite OPH rows.

    Rows must expose `k_bin`/`k_h_mpc` and `a_bin`/`a`, plus an anomaly-like
    field and a baryon-source field. This module intentionally does not fit CMB
    data. It is a contract surface for genuine plus/minus finite reruns.
    """

    base = _group_rows(base_rows)
    pert = _group_rows(perturbed_rows)
    controls = _group_rows(control_rows or [])
    out: list[BAKernelRow] = []
    for key in sorted(set(base).intersection(pert)):
        b = base[key]
        p = pert[key]
        count = min(len(b), len(p))
        if count <= 0:
            continue
        values: list[float] = []
        for left, right in zip(b[:count], p[:count], strict=False):
            delta_anomaly = _anomaly_value(right) - _anomaly_value(left)
            delta_baryon = _baryon_value(right) - _baryon_value(left)
            if abs(delta_baryon) > float(min_delta_baryon):
                values.append(delta_anomaly / delta_baryon)
        if not values:
            continue
        ctrl_values = [_anomaly_value(row) / max(abs(_baryon_value(row)), min_delta_baryon) for row in controls.get(key, [])]
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        ctrl_mean = float(np.mean(ctrl_values)) if ctrl_values else None
        ctrl_sem = float(np.std(ctrl_values, ddof=1) / math.sqrt(len(ctrl_values))) if len(ctrl_values) > 1 else None
        denom = float(np.std(ctrl_values, ddof=1)) if len(ctrl_values) > 1 else None
        z_score = (mean - ctrl_mean) / max(denom, 1.0e-12) if ctrl_mean is not None and denom is not None else None
        out.append(
            BAKernelRow(
                k_bin=float(key[0]),
                a_bin=float(key[1]),
                B_A=mean,
                sem=sem,
                sample_count=int(len(values)),
                control_B_A=ctrl_mean,
                control_sem=ctrl_sem,
                z_score=None if z_score is None or not np.isfinite(z_score) else float(z_score),
            )
        )
    return out


def B_A_kernel_receipt(rows: list[BAKernelRow], *, min_good_rows: int = 3, min_sample_count: int = 16) -> dict[str, Any]:
    good = [
        row
        for row in rows
        if row.sample_count >= int(min_sample_count)
        and np.isfinite(row.B_A)
        and (row.z_score is None or abs(float(row.z_score)) >= 3.0)
    ]
    required = max(int(min_good_rows), len(rows) // 2)
    diagnostic_candidate = bool(len(good) >= required and rows)
    physical_passed = False
    return {
        "B_A_DIAGNOSTIC_CANDIDATE_RECEIPT": diagnostic_candidate,
        "B_A_KERNEL_CANDIDATE_RECEIPT": diagnostic_candidate,
        "B_A_KERNEL_RECEIPT": physical_passed,
        "B_A_k_a_physical_emitted": physical_passed,
        "row_count": int(len(rows)),
        "good_row_count": int(len(good)),
        "required_good_row_count": int(required),
        "min_sample_count": int(min_sample_count),
        "promotion_blockers": _physical_source_promotion_blockers(diagnostic_candidate),
        "claim_boundary": (
            "Paired finite-difference B_A(k,a) diagnostic candidate. Significance and sample-count checks "
            "do not satisfy the physical kernel receipt until the source-functional, tangent-space, "
            "source-vector, lift-independence, gauge, and refinement receipts pass."
        ),
    }


def ba_kernel_report_from_paired_csv(
    base_csv: Path,
    perturbed_csv: Path,
    out_dir: Path,
    *,
    control_csv: Path | None = None,
    min_good_rows: int = 3,
    min_sample_count: int = 16,
) -> dict[str, Any]:
    base_rows = _read_csv(base_csv)
    perturbed_rows = _read_csv(perturbed_csv)
    control_rows = _read_csv(control_csv) if control_csv is not None else []
    rows = estimate_B_A_from_paired_runs(base_rows, perturbed_rows, control_rows)
    receipt = B_A_kernel_receipt(rows, min_good_rows=min_good_rows, min_sample_count=min_sample_count)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_rows(out / "B_A_k_a.csv", rows)
    report = {
        "mode": "paired_finite_difference_B_A_kernel_v0",
        "B_A_source": "parent_collar_finite_difference",
        "base_csv": str(base_csv),
        "perturbed_csv": str(perturbed_csv),
        "control_csv": None if control_csv is None else str(control_csv),
        "rows": [asdict(row) for row in rows],
        "B_A_k_a": [[row.k_bin, row.a_bin, row.B_A, row.sem, row.sample_count] for row in rows],
        **receipt,
    }
    (out / "B_A_kernel_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def ba_kernel_report_from_parent_report(parent_report_path: Path, out_dir: Path) -> dict[str, Any]:
    """Promote a paired parent-collar report into an auditable B_A kernel candidate.

    The parent report may already contain plus/minus finite perturbation rows and
    control rows. This writer keeps the kernel candidate visible while refusing
    the physical B_A kernel receipt unless the parent report also closes the
    physical calibration, refinement, and closure gates.
    """

    parent_path = Path(parent_report_path)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if not isinstance(parent, dict):
        parent = {}
    rows = _kernel_rows_from_parent(parent.get("rows") or parent.get("paired_perturbation_rows") or [])
    readiness = parent.get("readiness") if isinstance(parent.get("readiness"), dict) else {}
    checks = readiness.get("checks") if isinstance(readiness.get("checks"), dict) else {}
    candidate_checks = {
        "paired_perturb_resettle_rows_emitted": bool(checks.get("paired_perturb_resettle_rows_emitted", False)),
        "finite_difference_rows_emitted": bool(checks.get("finite_difference_rows_emitted", False)),
        "control_rows_emitted": bool(checks.get("control_rows_emitted", False)),
        "no_cmb_data_used": bool(checks.get("no_cmb_data_used", False)),
        "real_baryon_perturbation_runs_present": bool(
            checks.get("real_baryon_perturbation_runs_present", False)
        ),
        "full_perturbation_rerun": bool(checks.get("full_perturbation_rerun", False)),
        "report_backed_surrogate_parent": bool(checks.get("report_backed_surrogate_parent", True)),
        "controls_fail": bool(checks.get("controls_fail", False)),
        "sign_stable": bool(checks.get("sign_stable", False)),
    }
    physical_checks = {
        "B_A_PARENT_RECEIPT": bool(parent.get("B_A_PARENT_RECEIPT", False)),
        "scale_calibrated_k_h_mpc": bool(checks.get("scale_calibrated_k_h_mpc", False)),
        "calibrated_a_evolution": bool(checks.get("calibrated_a_evolution", False)),
        "finite_observer_view_parent_variation": bool(checks.get("finite_observer_view_parent_variation", False)),
        "energy_momentum_exchange_closed": bool(checks.get("energy_momentum_exchange_closed", False)),
        "gauge_consistency_audited": bool(checks.get("gauge_consistency_audited", False)),
        "refinement_convergence_passed": bool(checks.get("refinement_convergence_passed", False)),
    }
    diagnostic_candidate = bool(
        rows
        and all(value for key, value in candidate_checks.items() if key != "report_backed_surrogate_parent")
        and not candidate_checks["report_backed_surrogate_parent"]
    )
    physical_receipt = False
    report = {
        "mode": "parent_collar_B_A_kernel_promotion_v0",
        "parent_report": str(parent_path),
        "B_A_source": "parent_collar_finite_difference",
        "B_A_DIAGNOSTIC_CANDIDATE_RECEIPT": diagnostic_candidate,
        "B_A_KERNEL_CANDIDATE_RECEIPT": diagnostic_candidate,
        "B_A_KERNEL_RECEIPT": physical_receipt,
        "B_A_k_a_physical_emitted": physical_receipt,
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "row_count": len(rows),
        "candidate_checks": candidate_checks,
        "physical_checks": physical_checks,
        "promotion_blockers": _parent_kernel_promotion_blockers(candidate_checks, physical_checks, rows),
        "rows": [asdict(row) for row in rows],
        "B_A_k_a": [[row.k_bin, row.a_bin, row.B_A, row.sem, row.sample_count] for row in rows],
        "control_failures": readiness.get("control_failures", {}),
        "claim_boundary": (
            "Parent-collar B_A kernel promotion audit. Candidate rows come from paired finite perturbation "
            "reruns and controls, but the physical B_A kernel receipt remains false until a source functional, "
            "admissible tangent, source-vector response, lift-independence, physical k/a calibration, "
            "exchange/gauge audits, and derivative-level refinement receipts pass."
        ),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_rows(out / "B_A_kernel_candidate.csv", rows)
    _write_rows(out / "B_A_k_a.csv", rows)
    (out / "B_A_kernel_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "B_A_kernel_report.md").write_text(_markdown_parent_kernel_report(report), encoding="utf-8")
    return report


def ba_kernel_refinement_report(report_paths: list[Path]) -> dict[str, Any]:
    source_reports = _collect_ba_kernel_sources([Path(path) for path in report_paths])
    rows = [_kernel_refinement_source_row(path, report) for path, report in source_reports]
    rows = [row for row in rows if row["row_count"] > 0]
    by_patch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("patch_count") is not None:
            by_patch.setdefault(int(row["patch_count"]), []).append(row)
    patch_rows = []
    patch_vectors: dict[int, dict[tuple[float, float], float]] = {}
    for patch_count, group in sorted(by_patch.items()):
        aggregate = _aggregate_kernel_vectors([row["kernel"] for row in group])
        patch_vectors[int(patch_count)] = aggregate
        values = list(aggregate.values())
        patch_rows.append(
            {
                "patch_count": int(patch_count),
                "source_count": int(len(group)),
                "kernel_row_count": int(len(aggregate)),
                "mean_abs_B_A": _mean_abs(values),
                "candidate_receipt_count": int(sum(1 for row in group if row.get("candidate_receipt"))),
                "physical_receipt_count": int(sum(1 for row in group if row.get("physical_receipt"))),
            }
        )
    pair_rows = _kernel_refinement_pair_rows(patch_vectors)
    key_pair_rows = _kernel_refinement_key_pair_rows(patch_vectors)
    stable_key_rows = [row for row in key_pair_rows if row.get("key_refinement_pass")]
    common_key_count = min((len(set(vector)) for vector in patch_vectors.values()), default=0)
    usable_pair_count = sum(1 for row in pair_rows if row.get("common_key_count", 0) > 0)
    pair_passes = [
        bool(
            row.get("relative_mean_abs_delta") is not None
            and row.get("relative_mean_abs_delta") <= 0.25
            and row.get("sign_agreement_fraction", 0.0) >= 0.75
        )
        for row in pair_rows
    ]
    three_scale = len(patch_vectors) >= 3
    convergence = bool(three_scale and pair_rows and all(pair_passes))
    blockers = []
    if not rows:
        blockers.append("no_B_A_kernel_or_parent_reports")
    if len(patch_vectors) < 3:
        blockers.append("requires_at_least_three_patch_counts_for_refinement_convergence")
    if usable_pair_count <= 0:
        blockers.append("no_common_k_a_grid_across_patch_counts")
    if pair_rows and not all(pair_passes):
        blockers.append("B_A_kernel_pairwise_drift_or_sign_instability")
    report = {
        "mode": "B_A_kernel_refinement_v0",
        "source_report_count": len(source_reports),
        "usable_source_count": len(rows),
        "patch_count_count": len(patch_vectors),
        "patch_counts": sorted(int(value) for value in patch_vectors),
        "common_key_count": int(common_key_count),
        "two_scale_diagnostic_receipt": bool(len(patch_vectors) >= 2 and usable_pair_count > 0),
        "B_A_KERNEL_REFINEMENT_CONVERGENCE_RECEIPT": convergence,
        "refinement_convergence_passed": convergence,
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "patch_rows": patch_rows,
        "pair_rows": pair_rows,
        "key_pair_rows": key_pair_rows,
        "key_pair_row_count": int(len(key_pair_rows)),
        "key_pair_stable_fraction": (
            float(len(stable_key_rows) / len(key_pair_rows)) if key_pair_rows else None
        ),
        "source_rows": [
            {key: value for key, value in row.items() if key != "kernel"}
            for row in rows
        ],
        "blockers": blockers,
        "claim_boundary": (
            "B_A kernel refinement audit across finite regulator sizes. A two-scale diagnostic is useful "
            "for drift discovery, but physical convergence requires at least three patch counts with a "
            "stable common k/a grid, small relative drift, and stable signs. This report does not make a "
            "physical CMB prediction."
        ),
    }
    return report


def write_ba_kernel_refinement_report(report_paths: list[Path], out_dir: Path) -> dict[str, Any]:
    report = ba_kernel_refinement_report(report_paths)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "B_A_kernel_refinement_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    _write_refinement_rows(out / "B_A_kernel_refinement_pairs.csv", report.get("pair_rows") or [])
    _write_refinement_key_rows(
        out / "B_A_kernel_refinement_key_pairs.csv",
        report.get("key_pair_rows") or [],
    )
    (out / "B_A_kernel_refinement_report.md").write_text(
        _markdown_refinement_report(report),
        encoding="utf-8",
    )
    return report


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[float, float], list[dict[str, Any]]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        k = _float(row.get("k_bin", row.get("k_h_mpc", row.get("k_proxy_inverse_theta"))))
        a = _float(row.get("a_bin", row.get("a")))
        if k is None or a is None:
            continue
        grouped.setdefault((float(k), float(a)), []).append(row)
    return grouped


def _kernel_rows_from_parent(rows: list[dict[str, Any]]) -> list[BAKernelRow]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        k = _float(row.get("k_h_mpc", row.get("k_proxy_inverse_theta", row.get("k_bin"))))
        a = _float(row.get("a", row.get("a_bin")))
        if k is None or a is None:
            continue
        grouped.setdefault((float(k), float(a)), []).append(row)
    out: list[BAKernelRow] = []
    for (k, a), group in sorted(grouped.items()):
        values = [_float(row.get("B_A_mean", row.get("B_A"))) for row in group]
        values = [float(value) for value in values if value is not None]
        if not values:
            continue
        row_sem = [_float(row.get("B_A_sem")) for row in group]
        row_sem = [float(value) for value in row_sem if value is not None]
        spread_sem = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        intrinsic_sem = float(math.sqrt(sum(value * value for value in row_sem)) / max(len(row_sem), 1)) if row_sem else 0.0
        sample_count = sum(_int_or_zero(row.get("mode_count")) for row in group)
        if sample_count <= 0:
            sample_count = len(values)
        out.append(
            BAKernelRow(
                k_bin=float(k),
                a_bin=float(a),
                B_A=float(np.mean(values)),
                sem=float(max(spread_sem, intrinsic_sem)),
                sample_count=int(sample_count),
                control_B_A=None,
                control_sem=None,
                z_score=None,
            )
        )
    return out


def _collect_ba_kernel_sources(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    found: dict[Path, dict[str, Any]] = {}
    names = ("B_A_kernel_report.json", "b_a_parent_report.json", "paired_b_a_perturbation_report.json")
    for item in paths:
        path = Path(item)
        candidates: list[Path] = []
        if path.is_file():
            candidates.append(path)
        if path.exists() and path.is_dir():
            for name in names:
                direct = path / name
                if direct.exists():
                    candidates.append(direct)
                candidates.extend(sorted(path.glob(f"**/{name}")))
        for candidate in candidates:
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and _kernel_rows_from_source_report(data):
                found[candidate.resolve()] = data
    return sorted(found.items(), key=lambda item: str(item[0]))


def _kernel_refinement_source_row(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    kernel_rows = _kernel_rows_from_source_report(report)
    kernel = {(round(row.k_bin, 12), round(row.a_bin, 12)): float(row.B_A) for row in kernel_rows}
    patch_count = _patch_count_for_report(Path(path), report)
    values = list(kernel.values())
    return {
        "path": str(path),
        "mode": report.get("mode"),
        "patch_count": patch_count,
        "row_count": len(kernel_rows),
        "mean_abs_B_A": _mean_abs(values),
        "candidate_receipt": bool(
            report.get("B_A_KERNEL_CANDIDATE_RECEIPT", False)
            or report.get("B_A_PAIRED_DIAGNOSTIC_RECEIPT", False)
            or ((report.get("readiness") or {}).get("checks") or {}).get("paired_B_A_diagnostic_receipt", False)
        ),
        "physical_receipt": bool(report.get("B_A_KERNEL_RECEIPT", False) or report.get("B_A_PARENT_RECEIPT", False)),
        "kernel": kernel,
    }


def _kernel_rows_from_source_report(report: dict[str, Any]) -> list[BAKernelRow]:
    if report.get("B_A_k_a"):
        rows = []
        for row in report.get("B_A_k_a") or []:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            k = _float(row[0])
            a = _float(row[1])
            b = _float(row[2])
            if k is None or a is None or b is None:
                continue
            sem = _float(row[3]) if len(row) > 3 else None
            rows.append(
                BAKernelRow(
                    k_bin=k,
                    a_bin=a,
                    B_A=b,
                    sem=sem if sem is not None else 0.0,
                    sample_count=_int_or_zero(row[4]) if len(row) > 4 else 1,
                    control_B_A=None,
                    control_sem=None,
                    z_score=None,
                )
            )
        return rows
    source_rows = report.get("rows") or report.get("paired_perturbation_rows") or []
    return _kernel_rows_from_parent(source_rows if isinstance(source_rows, list) else [])


def _patch_count_for_report(path: Path, report: dict[str, Any]) -> int | None:
    for key in ("patch_count", "point_count"):
        parsed = _int_or_none(report.get(key))
        if parsed:
            return parsed
    current = Path(path).parent
    for _ in range(6):
        manifest = current / "manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            for key in ("patch_count", "point_count"):
                parsed = _int_or_none(data.get(key))
                if parsed:
                    return parsed
        if current.parent == current:
            break
        current = current.parent
    text = str(path)
    import re

    match = re.search(r"(?<!\d)(\d+)([kKmM])(?:_|\b)", text)
    if match:
        value = int(match.group(1))
        suffix = match.group(2).lower()
        return value * (1024 if suffix == "k" else 1024 * 1024)
    return None


def _aggregate_kernel_vectors(vectors: list[dict[tuple[float, float], float]]) -> dict[tuple[float, float], float]:
    grouped: dict[tuple[float, float], list[float]] = {}
    for vector in vectors:
        for key, value in vector.items():
            if np.isfinite(float(value)):
                grouped.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) for key, values in grouped.items() if values}


def _kernel_refinement_pair_rows(
    patch_vectors: dict[int, dict[tuple[float, float], float]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = sorted(patch_vectors)
    for left, right in zip(counts, counts[1:], strict=False):
        lv = patch_vectors[left]
        rv = patch_vectors[right]
        common = sorted(set(lv).intersection(rv))
        if not common:
            rows.append(
                {
                    "left_patch_count": int(left),
                    "right_patch_count": int(right),
                    "common_key_count": 0,
                    "mean_abs_delta": None,
                    "relative_mean_abs_delta": None,
                    "sign_agreement_fraction": None,
                    "pair_refinement_pass": False,
                }
            )
            continue
        deltas = [abs(rv[key] - lv[key]) for key in common]
        scale = max(_mean_abs([lv[key] for key in common]) or 0.0, _mean_abs([rv[key] for key in common]) or 0.0, 1e-300)
        sign_agree = [
            np.sign(rv[key]) == np.sign(lv[key])
            for key in common
            if abs(rv[key]) > 1e-300 or abs(lv[key]) > 1e-300
        ]
        relative = float(np.mean(deltas) / scale)
        sign_fraction = float(sum(sign_agree) / len(sign_agree)) if sign_agree else 0.0
        rows.append(
            {
                "left_patch_count": int(left),
                "right_patch_count": int(right),
                "common_key_count": int(len(common)),
                "mean_abs_delta": float(np.mean(deltas)),
                "relative_mean_abs_delta": relative,
                "sign_agreement_fraction": sign_fraction,
                "pair_refinement_pass": bool(relative <= 0.25 and sign_fraction >= 0.75),
            }
        )
    return rows


def _kernel_refinement_key_pair_rows(
    patch_vectors: dict[int, dict[tuple[float, float], float]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = sorted(patch_vectors)
    for left, right in zip(counts, counts[1:], strict=False):
        lv = patch_vectors[left]
        rv = patch_vectors[right]
        for k_bin, a_bin in sorted(set(lv).intersection(rv)):
            left_value = float(lv[(k_bin, a_bin)])
            right_value = float(rv[(k_bin, a_bin)])
            scale = max(abs(left_value), abs(right_value), 1e-300)
            relative = abs(right_value - left_value) / scale
            sign_agree = bool(
                np.sign(right_value) == np.sign(left_value)
                or (abs(right_value) <= 1e-300 and abs(left_value) <= 1e-300)
            )
            rows.append(
                {
                    "left_patch_count": int(left),
                    "right_patch_count": int(right),
                    "k_bin": float(k_bin),
                    "a_bin": float(a_bin),
                    "left_B_A": left_value,
                    "right_B_A": right_value,
                    "abs_delta": float(abs(right_value - left_value)),
                    "relative_abs_delta": float(relative),
                    "sign_agree": sign_agree,
                    "key_refinement_pass": bool(relative <= 0.25 and sign_agree),
                }
            )
    return rows


def _parent_kernel_promotion_blockers(
    candidate_checks: dict[str, bool],
    physical_checks: dict[str, bool],
    rows: list[BAKernelRow],
) -> list[str]:
    blockers: list[str] = []
    if not rows:
        blockers.append("B_A_kernel_rows_missing")
    for key, value in candidate_checks.items():
        if key == "report_backed_surrogate_parent":
            if value:
                blockers.append("report_backed_surrogate_parent_true")
            continue
        if not value:
            blockers.append(f"candidate_check_failed_{key}")
    for key, value in physical_checks.items():
        if not value:
            blockers.append(f"physical_check_failed_{key}")
    blockers.extend(_physical_source_promotion_blockers(bool(rows)))
    return blockers


def _physical_source_promotion_blockers(candidate_present: bool) -> list[str]:
    blockers = [
        "common_source_functional_receipt_missing",
        "admissible_source_tangent_receipt_missing",
        "constraint_preserving_retraction_receipt_missing",
        "B_A_source_lift_independence_receipt_missing",
        "source_vector_sufficiency_receipt_missing",
        "fixed_geometry_partial_response_receipt_missing",
        "source_design_identifiability_receipt_missing",
        "finite_difference_order_receipt_missing",
        "C1_refinement_receipt_missing",
        "order_of_limits_receipt_missing",
    ]
    if candidate_present:
        blockers.insert(0, "B_A_diagnostic_candidate_not_physical_kernel")
    return blockers


def _mean_abs(values: list[float]) -> float | None:
    finite = [abs(float(value)) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _anomaly_value(row: dict[str, Any]) -> float:
    for key in ("repair_anomaly", "rho_A_eq", "A_repair_density", "B_A_mean", "base_epsilon_cmi"):
        value = _float(row.get(key))
        if value is not None:
            return value
    return 0.0


def _baryon_value(row: dict[str, Any]) -> float:
    for key in ("baryon_source", "delta_baryon", "rho_b", "baryon_delta"):
        value = _float(row.get(key))
        if value is not None:
            return value
    return 0.0


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _int_or_zero(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _int_or_none(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _read_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[BAKernelRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(BAKernelRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_refinement_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "left_patch_count",
        "right_patch_count",
        "common_key_count",
        "mean_abs_delta",
        "relative_mean_abs_delta",
        "sign_agreement_fraction",
        "pair_refinement_pass",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_refinement_key_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "left_patch_count",
        "right_patch_count",
        "k_bin",
        "a_bin",
        "left_B_A",
        "right_B_A",
        "abs_delta",
        "relative_abs_delta",
        "sign_agree",
        "key_refinement_pass",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_parent_kernel_report(report: dict[str, Any]) -> str:
    blockers = report.get("promotion_blockers") or []
    lines = [
        "# B_A Kernel Promotion Audit",
        "",
        f"- candidate receipt: `{str(report.get('B_A_KERNEL_CANDIDATE_RECEIPT', False)).lower()}`",
        f"- physical B_A kernel receipt: `{str(report.get('B_A_KERNEL_RECEIPT', False)).lower()}`",
        f"- row count: `{report.get('row_count', 0)}`",
        "",
        "## Promotion Blockers",
        "",
    ]
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def _markdown_refinement_report(report: dict[str, Any]) -> str:
    blockers = report.get("blockers") or []
    lines = [
        "# B_A Kernel Refinement Audit",
        "",
        f"- two-scale diagnostic: `{str(report.get('two_scale_diagnostic_receipt', False)).lower()}`",
        f"- refinement convergence: `{str(report.get('B_A_KERNEL_REFINEMENT_CONVERGENCE_RECEIPT', False)).lower()}`",
        f"- patch counts: `{report.get('patch_counts', [])}`",
        f"- key-pair stable fraction: `{report.get('key_pair_stable_fraction')}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)
