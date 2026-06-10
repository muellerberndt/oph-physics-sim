from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.claims import PROXY, SCREEN_PROXY_CMB_RECEIPT, with_claim_metadata
from oph_fpe.cosmology.angular_power import angular_power_report
from oph_fpe.cosmology.cmb_compare import write_cmb_lite_comparison
from oph_fpe.cosmology.freezeout import _write_control_csv, _write_spectrum_csv


RESERVED_NPZ_KEYS = frozenset({"points", "cell_area_planck", "cell_entropy"})


def write_cl_from_freezeout_npz(
    run_dir: Path,
    out_dir: Path | None = None,
    *,
    ell_max: int = 256,
    fields: list[str] | tuple[str, ...] | None = None,
    harmonic_batch_size: int = 512,
    n_jobs: int = 1,
    controls: list[str] | tuple[str, ...] = ("shuffled_field", "random_gaussian"),
    benchmark: Path | None = None,
    benchmark_label: str = "Planck2018_TT_binned",
    source_url: str | None = None,
    seed: int = 1,
) -> dict[str, Any]:
    """Recompute a freezeout-screen C_l report from a saved field bundle.

    This is deliberately a postprocess-only path. It does not rerun repair,
    records, BW/KMS, H3, or object extraction; it only changes angular-spectrum
    resolution for fields already present in a run bundle.
    """

    run_dir = Path(run_dir)
    destination = Path(out_dir) if out_dir is not None else run_dir
    npz_path = run_dir / "freezeout_fields.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing freezeout field bundle: {npz_path}")
    destination.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path) as payload:
        points = np.asarray(payload["points"], dtype=float)
        cell_entropy = np.asarray(payload["cell_entropy"], dtype=float)
        available = [str(key) for key in payload.files if str(key) not in RESERVED_NPZ_KEYS]
        selected_names = [str(name) for name in (fields or available) if str(name) in available]
        selected = {name: np.asarray(payload[name], dtype=float) for name in selected_names}

    if not selected:
        raise ValueError("no selected freezeout fields available for C_l postprocess")

    source_cl = _read_json(run_dir / "cl_comparison_report.json")
    source_freezeout = _read_json(run_dir / "freezeout_map_summary.json")
    source_gate = _read_json(run_dir / "cosmology_gate_report.json")
    freezeout_cycle = _first_present(source_cl.get("freezeout_cycle"), source_freezeout.get("freezeout_cycle"))
    committed_fraction = _first_present(
        source_cl.get("committed_fraction"),
        source_freezeout.get("committed_fraction"),
    )

    report = angular_power_report(
        points,
        selected,
        ell_max=int(ell_max),
        seed=int(seed),
        controls=[str(item) for item in controls],
        estimator="spherical_harmonic",
        measure_weights=cell_entropy,
        harmonic_batch_size=int(harmonic_batch_size),
        n_jobs=int(n_jobs),
    )
    report = with_claim_metadata(
        report,
        claim_level=PROXY,
        receipt=SCREEN_PROXY_CMB_RECEIPT,
        physical_claim=False,
        observable_id="freezeout_screen_cl_proxy_postprocess",
        fit_objective="screen_angular_spectrum_high_ell_postprocess",
    )
    report["source_run_path"] = str(run_dir)
    report["source_freezeout_npz"] = str(npz_path)
    report["freezeout_cycle"] = int(freezeout_cycle) if freezeout_cycle is not None else None
    report["committed_fraction"] = float(committed_fraction) if committed_fraction is not None else None
    report["gate_report"] = source_cl.get("gate_report") or source_freezeout.get("gate_report") or source_gate
    report["postprocess_only"] = True
    report["physical_cmb_prediction"] = False
    report["claim_boundary"] = (
        "high-ell postprocess of a saved finite screen freezeout field bundle. "
        "This changes only angular-spectrum resolution; it does not rerun OPH dynamics, "
        "does not establish a 3D bulk, and is not a physical CMB prediction."
    )
    (destination / "cl_comparison_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    _write_spectrum_csv(destination / "cl_proxy.csv", report["fields"])
    _write_control_csv(destination / "cl_controls.csv", report["controls"])

    cmb_report: dict[str, Any] | None = None
    if benchmark is not None:
        cmb_report = write_cmb_lite_comparison(
            destination,
            Path(benchmark),
            benchmark_label=str(benchmark_label),
            source_url=source_url,
            field_names=selected_names,
        )
    summary = {
        "mode": "freezeout_cl_from_npz_postprocess",
        "source_run_path": str(run_dir),
        "out_dir": str(destination),
        "ell_max": int(ell_max),
        "field_count": len(selected),
        "fields": selected_names,
        "cmb_lite_written": bool(cmb_report),
        "cl_csv_written": True,
        "source_gate_allowed": bool(report["gate_report"].get("allowed", False)),
        "physical_cmb_prediction": False,
        "claim_boundary": report["claim_boundary"],
    }
    (destination / "cl_postprocess_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
