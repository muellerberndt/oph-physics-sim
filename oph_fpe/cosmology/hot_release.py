from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any

from oph_fpe.claims import CONTINUATION, COSMOLOGY_PERTURBATION_RECEIPT, with_claim_metadata


def hot_release_report(
    run_dirs: list[Path],
    *,
    phi_tolerance: float = 0.0,
    min_committed_fraction: float = 0.99,
    max_collar_cmi: float = 0.05,
) -> dict[str, Any]:
    """Audit the OPH hot-MaxEnt release replacement for reheating.

    The report identifies the first finite run cycle where overlap repair has
    settled and records have committed. It then checks whether collar-Markov
    errors are small enough to treat the release as a theorem-grade MaxEnt
    branch. Current simulations generally pass the mechanical release surface
    but fail the collar theorem-grade threshold.
    """

    rows = [
        _run_row(
            path,
            phi_tolerance=float(phi_tolerance),
            min_committed_fraction=float(min_committed_fraction),
            max_collar_cmi=float(max_collar_cmi),
        )
        for path in _find_run_dirs(run_dirs)
    ]
    rows = [row for row in rows if row.get("has_release_inputs")]
    aggregate = _aggregate(rows)
    report = {
        "mode": "oph_hot_maxent_release_audit_v0",
        "run_count": len(rows),
        "thresholds": {
            "phi_tolerance": float(phi_tolerance),
            "min_committed_fraction": float(min_committed_fraction),
            "max_collar_cmi": float(max_collar_cmi),
        },
        "aggregate": aggregate,
        "rows": rows,
        "hot_release_theorem_ready": bool(rows and aggregate.get("hot_release_theorem_ready_count") == len(rows)),
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Finite-run audit of the OPH hot-release replacement for inflaton reheating. The release cycle "
            "and beta/T proxy are simulator quantities. They are not a physical reheating temperature unless "
            "the screen/collar-to-SM unit map and collar Markov/recovery gates are closed."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=CONTINUATION,
        receipt=COSMOLOGY_PERTURBATION_RECEIPT,
        physical_claim=False,
        observable_id="oph_hot_maxent_release",
        fit_objective="sync_release_surface_and_collar_error_gate",
    )


def write_hot_release_report(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    phi_tolerance: float = 0.0,
    min_committed_fraction: float = 0.99,
    max_collar_cmi: float = 0.05,
) -> dict[str, Any]:
    report = hot_release_report(
        run_dirs,
        phi_tolerance=float(phi_tolerance),
        min_committed_fraction=float(min_committed_fraction),
        max_collar_cmi=float(max_collar_cmi),
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hot_release_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "hot_release_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "hot_release_rows.csv", report["rows"])
    return report


def _run_row(
    run_path: Path,
    *,
    phi_tolerance: float,
    min_committed_fraction: float,
    max_collar_cmi: float,
) -> dict[str, Any]:
    trace = _read_trace(Path(run_path) / "mismatch_trace.csv")
    freezeout = _read_json(Path(run_path) / "freezeout_map_summary.json")
    collar = _read_json(Path(run_path) / "collar_markov_report.json")
    release = _first_release_row(trace, phi_tolerance=phi_tolerance, min_committed_fraction=min_committed_fraction)
    median_cmi = _float_or_none(collar.get("median_epsilon_cmi"))
    p90_cmi = _float_or_none(collar.get("p90_epsilon_cmi"))
    beta = _float_or_none(release.get("beta")) if release else None
    t_proxy = 1.0 / beta if beta and beta > 0.0 else None
    mechanical = bool(release)
    collar_ok = bool(median_cmi is not None and median_cmi <= float(max_collar_cmi))
    return {
        "run_path": str(run_path),
        "has_release_inputs": bool(trace or freezeout or collar),
        "mechanical_release_surface_found": mechanical,
        "release_cycle": _int_or_none(release.get("cycle")) if release else None,
        "release_beta": beta,
        "release_temperature_proxy_inverse_beta": t_proxy,
        "release_phi": _float_or_none(release.get("phi")) if release else None,
        "release_committed_fraction": _float_or_none(release.get("committed_fraction")) if release else None,
        "release_record_entropy": _float_or_none(release.get("record_entropy")) if release else None,
        "freezeout_cycle": _int_or_none(freezeout.get("freezeout_cycle")),
        "freezeout_committed_fraction": _float_or_none(freezeout.get("committed_fraction")),
        "median_epsilon_cmi": median_cmi,
        "p90_epsilon_cmi": p90_cmi,
        "collar_markov_gate_pass": collar_ok,
        "hot_release_theorem_ready": bool(mechanical and collar_ok),
        "missing_gates": [
            name
            for name, passed in {
                "mechanical_release_surface": mechanical,
                "collar_markov_recovery_error_small": collar_ok,
                "physical_SM_unit_map": False,
                "conserved_charge_payload": False,
            }.items()
            if not passed
        ],
    }


def _first_release_row(
    trace: list[dict[str, Any]],
    *,
    phi_tolerance: float,
    min_committed_fraction: float,
) -> dict[str, Any] | None:
    for row in trace:
        phi = _float_or_none(row.get("phi"))
        committed = _float_or_none(row.get("committed_fraction"))
        if phi is not None and committed is not None and phi <= float(phi_tolerance) and committed >= float(min_committed_fraction):
            return row
    return None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    release_cycles = [int(row["release_cycle"]) for row in rows if row.get("release_cycle") is not None]
    cmi = [float(row["median_epsilon_cmi"]) for row in rows if row.get("median_epsilon_cmi") is not None]
    return {
        "run_count": len(rows),
        "mechanical_release_surface_count": sum(1 for row in rows if row.get("mechanical_release_surface_found")),
        "collar_markov_gate_pass_count": sum(1 for row in rows if row.get("collar_markov_gate_pass")),
        "hot_release_theorem_ready_count": sum(1 for row in rows if row.get("hot_release_theorem_ready")),
        "median_release_cycle": median(release_cycles) if release_cycles else None,
        "mean_release_cycle": fmean(release_cycles) if release_cycles else None,
        "mean_median_epsilon_cmi": fmean(cmi) if cmi else None,
    }


def _find_run_dirs(roots: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            paths.add(root.parent)
        if (root / "mismatch_trace.csv").exists() or (root / "freezeout_map_summary.json").exists():
            paths.add(root)
        if root.exists():
            for name in ("mismatch_trace.csv", "freezeout_map_summary.json"):
                for path in root.glob(f"**/{name}"):
                    paths.add(path.parent)
    return sorted(paths, key=lambda path: str(path))


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_or_none(value: Any) -> int | None:
    numeric = _float_or_none(value)
    return int(numeric) if numeric is not None else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat_rows = [{key: (",".join(value) if isinstance(value, list) else value) for key, value in row.items()} for row in rows]
    keys = list(dict.fromkeys(key for row in flat_rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat_rows)


def _markdown_report(report: dict[str, Any]) -> str:
    agg = report.get("aggregate", {})
    return "\n".join(
        [
            "# OPH Hot MaxEnt Release Audit",
            "",
            f"- Runs: {report.get('run_count')}",
            f"- Mechanical release surfaces: {agg.get('mechanical_release_surface_count')}",
            f"- Collar Markov gate passes: {agg.get('collar_markov_gate_pass_count')}",
            f"- Theorem-ready hot release count: {agg.get('hot_release_theorem_ready_count')}",
            f"- Hot release theorem ready: `{str(report.get('hot_release_theorem_ready')).lower()}`",
            "",
            "## Claim Boundary",
            "",
            str(report.get("claim_boundary", "")),
            "",
        ]
    )

