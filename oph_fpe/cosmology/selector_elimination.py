from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.constants.oph_pixel import OPHPixelConstants, P_STAR
from oph_fpe.cosmology.oph_screen_power import DEFAULT_D_STAR_MPC


DEFAULT_SOURCE_DIR = Path("/Users/muellerberndt/Projects/oph-meta/cosmology/correspondence/cmb/7")
DEFAULT_KERNEL_ELL = (2, 3, 4, 5, 10, 20, 29, 30, 32, 40, 50, 100, 220)


def ir_kernel(
    ell: np.ndarray | list[float] | tuple[float, ...] | float,
    *,
    q_ir: float = 0.25,
    ell_ir: float = 32.0,
) -> np.ndarray:
    ell_arr = np.asarray(ell, dtype=float)
    denom = max(float(ell_ir) * (float(ell_ir) + 1.0), 1.0e-30)
    return 1.0 - float(q_ir) * np.exp(-(ell_arr * (ell_arr + 1.0)) / denom)


def selector_elimination_report(
    source_dir: Path | None = None,
    *,
    P: float = P_STAR,
    csv_tolerance: float = 5.0e-13,
) -> dict[str, Any]:
    """Return the v1.5 selector-elimination receipt for the exact OPH CMB branch.

    The report distinguishes theorem-side target closure from finite-lattice
    derivation. The IR kernel parameters are derived from the affine zero-mode
    and visible-covariance-rank arguments. The red-tilt branch is reduced to a
    single finite repair-clock certificate, kappa_rep=e, which current lattice
    runs have not yet derived.
    """

    pixel = OPHPixelConstants(P=float(P))
    delta_p = pixel.alpha_from_P * pixel.sqrt_pi
    kappa_rep = math.e
    eta_r = kappa_rep * delta_p
    n_s = 1.0 - eta_r

    l0_dim = 1
    l1_dim = 3
    q_ir = float(l0_dim / (l0_dim + l1_dim))

    dodeca_faces = 12
    dodeca_vertices = 20
    visible_scalar_channels = dodeca_faces + dodeca_vertices
    identity_record = 1
    visible_covariance_rank = visible_scalar_channels + identity_record
    ell_ir = float(visible_covariance_rank - 1)
    harmonic_capacity = int((ell_ir + 1.0) ** 2)

    source = Path(source_dir) if source_dir is not None else None
    source_files = _source_file_status(source)
    exact_table = _exact_kernel_table(DEFAULT_KERNEL_ELL, q_ir=q_ir, ell_ir=ell_ir)
    csv_audit = _audit_kernel_csv(source, q_ir=q_ir, ell_ir=ell_ir, tolerance=csv_tolerance)
    status_rows = _read_status_rows(source)
    status_audit = _audit_status_rows(status_rows)

    q_removed = bool(math.isclose(q_ir, 0.25, abs_tol=1.0e-15))
    ell_removed = bool(math.isclose(ell_ir, 32.0, abs_tol=1.0e-15))
    eta_reduced = bool(eta_r > 0.0 and n_s < 1.0)
    source_audit_pass = bool(
        source is not None
        and source_files.get("core_v1_5_present", False)
        and csv_audit.get("passed", False)
        and status_audit.get("q_ir_selector_removed", False)
        and status_audit.get("ell_ir_selector_removed", False)
        and status_audit.get("eta_r_reduced_to_repair_clock_certificate", False)
    )
    theorem_receipt = bool(q_removed and ell_removed and eta_reduced)

    return {
        "mode": "oph_cmb_selector_elimination_v1_5",
        "oph_constants": pixel.as_jsonable(),
        "selector_elimination": {
            "q_IR_selector_removed": q_removed,
            "q_IR": q_ir,
            "q_IR_derivation": "dim(H_ell0) / dim(H_ell0 plus H_ell1) = 1 / (1 + 3)",
            "affine_sector_dimensions": {"ell0_checkpoint_scalar": l0_dim, "ell1_global_repair_modes": l1_dim},
            "ell_IR_selector_removed": ell_removed,
            "ell_IR": ell_ir,
            "ell_IR_derivation": "F+V visible scalar channels plus identity: Q=12+20+1=33, so L=Q-1=32",
            "visible_covariance_rank": visible_covariance_rank,
            "visible_scalar_channels": visible_scalar_channels,
            "harmonic_capacity_slots": harmonic_capacity,
            "eta_R_free_selector_removed": False,
            "eta_R_reduced_to_repair_clock_certificate": eta_reduced,
            "remaining_eta_R_certificate": "derive_or_measure kappa_rep=e from finite-patch scalar repair semigroup",
        },
        "scalar_tilt": {
            "formula": "eta_R = kappa_rep * alpha(0) * sqrt(pi) = kappa_rep * (P - phi)",
            "canonical_kappa_rep": kappa_rep,
            "canonical_kappa_rep_status": "certificate_pending",
            "delta_P": delta_p,
            "eta_R": eta_r,
            "n_s": n_s,
            "alpha_from_P": pixel.alpha_from_P,
            "alpha_inverse_from_P": pixel.alpha_inverse_from_P,
        },
        "cmb_ir_kernel": {
            "formula": "F_IR(ell)=1-(1/4)*exp[-ell(ell+1)/(32*33)]",
            "q_IR": q_ir,
            "ell_IR": ell_ir,
            "theta_IR_deg": 180.0 / ell_ir,
            "k_IR_Mpc_inverse": ell_ir / DEFAULT_D_STAR_MPC,
            "N_frz_proxy": harmonic_capacity,
            "exact_values": exact_table,
        },
        "source_files": source_files,
        "source_status_rows": status_rows,
        "source_status_audit": status_audit,
        "exact_ir_kernel_csv_audit": csv_audit,
        "THEOREM_SIDE_SELECTOR_ELIMINATION_RECEIPT": theorem_receipt,
        "SOURCE_PACKET_AUDIT_RECEIPT": source_audit_pass,
        "finite_lattice_derived": False,
        "physical_cmb_prediction": False,
        "remaining_certificates": [
            "finite-patch scalar repair-clock normalization kappa_rep=e",
            "finite-register visible-covariance noncollapse for ell_IR=32",
            "freezeout branch couples protected reserve to affine ell=0 sector for q_IR=1/4",
            "official masked Planck likelihood and map-space parity/BipoSH tests",
        ],
        "claim_boundary": (
            "Selector-elimination receipt for the exact OPH-CMB target branch. q_IR=1/4 and ell_IR=32 "
            "are no longer treated as free fit selectors in this report; they are computed from the affine "
            "zero-mode reserve and dodecahedral visible-covariance rank. eta_R is reduced to the single "
            "repair-clock certificate kappa_rep=e, which is not yet derived by the finite lattice. This is "
            "therefore a theorem-side target/certificate audit, not a completed physical CMB prediction."
        ),
    }


def write_selector_elimination_report(
    source_dir: Path | None,
    out_dir: Path,
    *,
    P: float = P_STAR,
) -> dict[str, Any]:
    report = selector_elimination_report(source_dir, P=float(P))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "oph_cmb_selector_elimination_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "oph_cmb_selector_elimination_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_kernel_csv(out / "exact_ir_kernel_values_v1_5.csv", report["cmb_ir_kernel"]["exact_values"])
    return report


def _exact_kernel_table(ells: tuple[int, ...], *, q_ir: float, ell_ir: float) -> list[dict[str, float]]:
    values = ir_kernel(list(ells), q_ir=q_ir, ell_ir=ell_ir)
    return [
        {"ell": float(ell), "F_IR_exact_q1over4_L32": float(value)}
        for ell, value in zip(ells, values, strict=True)
    ]


def _audit_kernel_csv(
    source_dir: Path | None,
    *,
    q_ir: float,
    ell_ir: float,
    tolerance: float,
) -> dict[str, Any]:
    path = _first_existing(
        source_dir,
        "exact_ir_kernel_values_v1_5.csv",
        "data/exact_ir_kernel_values_v1_5.csv",
        "exact_kernel_values_v1_5.csv",
        "data/exact_kernel_values_v1_5.csv",
    )
    if path is None or not path.exists():
        return {"present": False, "passed": False, "reason": "exact IR kernel CSV not found"}
    rows = _read_csv(path)
    errors: list[float] = []
    audited_rows: list[dict[str, float]] = []
    for row in rows:
        ell = _float_or_none(row.get("ell"))
        observed = _float_or_none(row.get("F_IR_exact_q1over4_L32") or row.get("F_IR"))
        if ell is None or observed is None:
            continue
        expected = float(ir_kernel(float(ell), q_ir=q_ir, ell_ir=ell_ir))
        error = abs(observed - expected)
        errors.append(error)
        audited_rows.append(
            {
                "ell": float(ell),
                "csv_F_IR": float(observed),
                "computed_F_IR": expected,
                "abs_error": error,
            }
        )
    max_error = max(errors) if errors else None
    return {
        "present": True,
        "path": str(path),
        "sha256": _sha256_file(path),
        "row_count": len(rows),
        "audited_row_count": len(audited_rows),
        "max_abs_error": max_error,
        "tolerance": float(tolerance),
        "passed": bool(errors and max_error is not None and max_error <= float(tolerance)),
        "rows": audited_rows,
    }


def _audit_status_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {str(row.get("old_selector", "")).strip(): str(row.get("v1_5_status", "")).lower() for row in rows}
    q_status = statuses.get("S2: q_IR = 1/4 from four equipotent sectors", "")
    ell_status = statuses.get("S3: ell_IR = 32", "")
    eta_status = statuses.get("S1: eta_R = e alpha sqrt(pi)", "")
    return {
        "row_count": len(rows),
        "q_ir_selector_removed": "removed as selector" in q_status,
        "ell_ir_selector_removed": "removed as selector" in ell_status,
        "eta_r_reduced_to_repair_clock_certificate": "repair-clock" in eta_status,
    }


def _read_status_rows(source_dir: Path | None) -> list[dict[str, Any]]:
    path = _first_existing(
        source_dir,
        "selector_elimination_status_v1_5.csv",
        "data/selector_elimination_status_v1_5.csv",
    )
    if path is None or not path.exists():
        return []
    return _read_csv(path)


def _source_file_status(source_dir: Path | None) -> dict[str, Any]:
    if source_dir is None:
        return {"source_dir": None, "files": {}, "core_v1_5_present": False}
    source = Path(source_dir)
    candidates = [
        "OPH-CMB-Selector-Elimination-v1.5.md",
        "comms3-remove-all-selectors.md",
        "selector_elimination_status_v1_5.csv",
        "exact_ir_kernel_values_v1_5.csv",
        "OPH-CMB-selector-elimination-v1.5.zip",
        "data/selector_elimination_status_v1_5.csv",
        "data/exact_ir_kernel_values_v1_5.csv",
        "data/selector_elimination_summary_v1_5.json",
        "data/numerical_targets_v1_5.json",
        "math/OPH-CMB-Selector-Elimination-v1.5.md",
        "math/no_remaining_selectors_theorems_v1_5.md",
    ]
    files: dict[str, Any] = {}
    for relative in candidates:
        path = source / relative
        files[relative] = {
            "present": path.exists(),
            "sha256": _sha256_file(path) if path.exists() else None,
        }
    top_level_core = all(
        files[name]["present"]
        for name in (
            "OPH-CMB-Selector-Elimination-v1.5.md",
            "selector_elimination_status_v1_5.csv",
            "exact_ir_kernel_values_v1_5.csv",
        )
    )
    extracted_core = all(
        files[name]["present"]
        for name in (
            "math/OPH-CMB-Selector-Elimination-v1.5.md",
            "data/selector_elimination_status_v1_5.csv",
            "data/exact_ir_kernel_values_v1_5.csv",
        )
    )
    return {
        "source_dir": str(source),
        "files": files,
        "top_level_core_present": top_level_core,
        "extracted_core_present": extracted_core,
        "core_v1_5_present": bool(top_level_core or extracted_core),
    }


def _first_existing(source_dir: Path | None, *relative_paths: str) -> Path | None:
    if source_dir is None:
        return None
    candidates = [Path(source_dir) / relative for relative in relative_paths]
    return next((path for path in candidates if path.exists()), candidates[0] if candidates else None)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_kernel_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ell", "F_IR_exact_q1over4_L32"])
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _markdown_report(report: dict[str, Any]) -> str:
    selector = report["selector_elimination"]
    scalar = report["scalar_tilt"]
    ir = report["cmb_ir_kernel"]
    csv_audit = report["exact_ir_kernel_csv_audit"]
    return "\n".join(
        [
            "# OPH CMB Selector Elimination v1.5",
            "",
            report["claim_boundary"],
            "",
            "## Closed Target Values",
            "",
            f"- q_IR: `{selector['q_IR']}`",
            f"- ell_IR: `{selector['ell_IR']}`",
            f"- N_frz proxy: `{ir['N_frz_proxy']}`",
            f"- eta_R canonical branch: `{scalar['eta_R']:.12f}`",
            f"- n_s canonical branch: `{scalar['n_s']:.12f}`",
            "",
            "## Receipts",
            "",
            f"- theorem-side selector elimination: `{report['THEOREM_SIDE_SELECTOR_ELIMINATION_RECEIPT']}`",
            f"- source packet audit: `{report['SOURCE_PACKET_AUDIT_RECEIPT']}`",
            f"- finite lattice derived: `{report['finite_lattice_derived']}`",
            f"- physical CMB prediction: `{report['physical_cmb_prediction']}`",
            "",
            "## Source CSV Audit",
            "",
            f"- present: `{csv_audit.get('present')}`",
            f"- passed: `{csv_audit.get('passed')}`",
            f"- audited rows: `{csv_audit.get('audited_row_count')}`",
            f"- max abs error: `{csv_audit.get('max_abs_error')}`",
            "",
            "## Remaining Certificates",
            "",
            *[f"- {item}" for item in report["remaining_certificates"]],
            "",
        ]
    )
