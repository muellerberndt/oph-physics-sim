from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from oph_fpe.claims import CONTINUATION, COSMOLOGY_PERTURBATION_RECEIPT, with_claim_metadata


DEFAULT_CHANNELS = {
    "photon_proxy": {"field": "record_signature", "w": 1.0 / 3.0},
    "baryon_proxy": {"field": "cumulative_repair_load", "w": 0.0},
    "neutrino_proxy": {"field": "stable_count", "w": 1.0 / 3.0},
    "anomaly_proxy": {"field": "s3_class_density", "w": 0.0},
}


def adiabaticity_report(
    run_dirs: list[Path],
    *,
    max_entropy_residual_std: float = 0.25,
    min_common_clock_corr: float = 0.85,
) -> dict[str, Any]:
    """Audit same-boundary adiabaticity/isocurvature suppression proxies.

    This is a finite-screen proxy. The channel labels are observer-visible
    fields standing in for species record channels; they are not physical
    photon/baryon/neutrino fluids unless a later Boltzmann bridge supplies that
    mapping.
    """

    rows = [
        _run_row(
            path,
            max_entropy_residual_std=float(max_entropy_residual_std),
            min_common_clock_corr=float(min_common_clock_corr),
        )
        for path in _find_run_dirs(run_dirs)
    ]
    rows = [row for row in rows if row.get("has_adiabaticity_inputs")]
    aggregate = _aggregate(rows)
    report = {
        "mode": "oph_same_boundary_adiabaticity_audit_v0",
        "run_count": len(rows),
        "thresholds": {
            "max_entropy_residual_std": float(max_entropy_residual_std),
            "min_common_clock_corr": float(min_common_clock_corr),
        },
        "channel_contract": DEFAULT_CHANNELS,
        "aggregate": aggregate,
        "rows": rows,
        "adiabaticity_established": bool(rows and aggregate.get("adiabaticity_proxy_pass_count") == len(rows)),
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Observer-visible proxy for OPH same-boundary adiabaticity. It checks whether declared record "
            "channels share one scalar clock/displacement and have small entropy-mode residuals. It is not "
            "a physical isocurvature likelihood until these channels are mapped to Boltzmann species."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=CONTINUATION,
        receipt=COSMOLOGY_PERTURBATION_RECEIPT,
        physical_claim=False,
        observable_id="oph_same_boundary_adiabaticity",
        fit_objective="species_channel_entropy_residual_proxy",
    )


def write_adiabaticity_report(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    max_entropy_residual_std: float = 0.25,
    min_common_clock_corr: float = 0.85,
) -> dict[str, Any]:
    report = adiabaticity_report(
        run_dirs,
        max_entropy_residual_std=float(max_entropy_residual_std),
        min_common_clock_corr=float(min_common_clock_corr),
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adiabaticity_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "adiabaticity_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "adiabaticity_rows.csv", report["rows"])
    return report


def _run_row(
    run_path: Path,
    *,
    max_entropy_residual_std: float,
    min_common_clock_corr: float,
) -> dict[str, Any]:
    npz_path = Path(run_path) / "freezeout_fields.npz"
    if not npz_path.exists():
        return {"run_path": str(run_path), "has_adiabaticity_inputs": False, "reason": "freezeout_fields_missing"}
    payload = np.load(npz_path)
    channels = _load_channels(payload)
    if len(channels) < 2:
        return {"run_path": str(run_path), "has_adiabaticity_inputs": False, "reason": "not_enough_channels"}
    zeta = _standardize(next(iter(channels.values()))["raw"])
    channel_rows = []
    scaled: dict[str, np.ndarray] = {}
    for name, channel in channels.items():
        value = _standardize(channel["raw"]) / (1.0 + float(channel["w"]))
        scaled[name] = value
        channel_rows.append(
            {
                "channel": name,
                "field": channel["field"],
                "w": float(channel["w"]),
                "std": float(np.std(value)),
                "corr_with_common_clock": _corr(value, zeta),
            }
        )
    pair_rows = []
    names = list(scaled)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            residual = scaled[left] - scaled[right]
            pair_rows.append(
                {
                    "left": left,
                    "right": right,
                    "entropy_residual_std": float(np.std(residual)),
                    "entropy_residual_mean_abs": float(np.mean(np.abs(residual))),
                    "channel_corr": _corr(scaled[left], scaled[right]),
                }
            )
    max_residual = max((float(row["entropy_residual_std"]) for row in pair_rows), default=None)
    min_corr = min((float(row["corr_with_common_clock"]) for row in channel_rows), default=None)
    pass_gate = bool(
        max_residual is not None
        and min_corr is not None
        and max_residual <= float(max_entropy_residual_std)
        and min_corr >= float(min_common_clock_corr)
    )
    return {
        "run_path": str(run_path),
        "has_adiabaticity_inputs": True,
        "channel_count": len(channels),
        "max_entropy_residual_std": max_residual,
        "mean_entropy_residual_std": fmean(float(row["entropy_residual_std"]) for row in pair_rows) if pair_rows else None,
        "min_common_clock_corr": min_corr,
        "mean_common_clock_corr": fmean(float(row["corr_with_common_clock"]) for row in channel_rows) if channel_rows else None,
        "adiabaticity_proxy_pass": pass_gate,
        "channel_rows": channel_rows,
        "pair_rows": pair_rows,
        "missing_gates": [
            name
            for name, passed in {
                "small_entropy_residuals": bool(max_residual is not None and max_residual <= float(max_entropy_residual_std)),
                "shared_common_clock": bool(min_corr is not None and min_corr >= float(min_common_clock_corr)),
                "physical_species_mapping": False,
                "Boltzmann_isocurvature_likelihood": False,
            }.items()
            if not passed
        ],
    }


def _load_channels(payload: Any) -> dict[str, dict[str, Any]]:
    channels = {}
    for name, spec in DEFAULT_CHANNELS.items():
        field = str(spec["field"])
        if field in payload:
            values = np.asarray(payload[field], dtype=float)
            if values.ndim == 1 and float(np.std(values)) > 1.0e-12:
                channels[name] = {"field": field, "w": float(spec["w"]), "raw": values}
    return channels


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    residuals = [float(row["max_entropy_residual_std"]) for row in rows if row.get("max_entropy_residual_std") is not None]
    corrs = [float(row["min_common_clock_corr"]) for row in rows if row.get("min_common_clock_corr") is not None]
    return {
        "run_count": len(rows),
        "adiabaticity_proxy_pass_count": sum(1 for row in rows if row.get("adiabaticity_proxy_pass")),
        "mean_max_entropy_residual_std": fmean(residuals) if residuals else None,
        "mean_min_common_clock_corr": fmean(corrs) if corrs else None,
    }


def _find_run_dirs(roots: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            paths.add(root.parent)
        if (root / "freezeout_fields.npz").exists():
            paths.add(root)
        if root.exists():
            for path in root.glob("**/freezeout_fields.npz"):
                paths.add(path.parent)
    return sorted(paths, key=lambda path: str(path))


def _standardize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    centered = arr - float(np.mean(arr))
    scale = float(np.std(centered))
    return centered / scale if scale > 1.0e-12 else np.zeros_like(centered)


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left = _standardize(left)
    right = _standardize(right)
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(np.mean(left * right))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key not in {"channel_rows", "pair_rows", "missing_gates"}}
        flat["missing_gates"] = ",".join(row.get("missing_gates", []))
        flat_rows.append(flat)
    keys = list(dict.fromkeys(key for row in flat_rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat_rows)


def _markdown_report(report: dict[str, Any]) -> str:
    agg = report.get("aggregate", {})
    return "\n".join(
        [
            "# OPH Adiabaticity / Isocurvature Proxy Audit",
            "",
            f"- Runs: {report.get('run_count')}",
            f"- Proxy passes: {agg.get('adiabaticity_proxy_pass_count')}",
            f"- Mean max entropy residual std: {agg.get('mean_max_entropy_residual_std')}",
            f"- Mean min common-clock correlation: {agg.get('mean_min_common_clock_corr')}",
            f"- Adiabaticity established: `{str(report.get('adiabaticity_established')).lower()}`",
            "",
            "## Claim Boundary",
            "",
            str(report.get("claim_boundary", "")),
            "",
        ]
    )

