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
    passed = bool(len(good) >= required and rows)
    return {
        "B_A_KERNEL_RECEIPT": passed,
        "B_A_k_a_physical_emitted": passed,
        "row_count": int(len(rows)),
        "good_row_count": int(len(good)),
        "required_good_row_count": int(required),
        "min_sample_count": int(min_sample_count),
        "claim_boundary": (
            "Paired finite-difference B_A(k,a) receipt. It is physical only for real paired OPH runs with "
            "predeclared baryon perturbations and controls; report-backed surrogate rows must not pass this gate."
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


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[float, float], list[dict[str, Any]]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        k = _float(row.get("k_bin", row.get("k_h_mpc", row.get("k_proxy_inverse_theta"))))
        a = _float(row.get("a_bin", row.get("a")))
        if k is None or a is None:
            continue
        grouped.setdefault((float(k), float(a)), []).append(row)
    return grouped


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
