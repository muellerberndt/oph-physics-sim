from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from oph_fpe.claims import QUANTITATIVE_BRANCH, SCREEN_PROXY_CMB_RECEIPT, with_claim_metadata
from oph_fpe.constants.oph_pixel import OPHPixelConstants, P_STAR
from oph_fpe.cosmology.oph_screen_power import (
    DEFAULT_A_S,
    DEFAULT_D_STAR_MPC,
    DEFAULT_K0_MPC,
    OPHScreenPowerParams,
    D_ell_from_C_ell,
    primordial_power_oph,
)
from oph_fpe.cosmology.selector_elimination import selector_elimination_report


def maxent_green_spectrum_report(
    source_dir: Path | None = None,
    *,
    patch_count: int = 262_144,
    ell_max: int = 256,
    P: float = P_STAR,
    kappa_rep: float | None = None,
    amplitude: float = 1.0,
    mu: float = 0.0,
    primordial_k_count: int = 256,
    primordial_k_min: float = 1.0e-4,
    primordial_k_max: float = 1.0,
) -> dict[str, Any]:
    """Write the paper-side MaxEnt Green-spectrum source law as a finite receipt.

    This is intentionally not fitted to a final simulated sky field. It encodes
    the CMB theorem lane from the notes: finite normal-form fields, MaxEnt
    inverse-Laplacian covariance on the screen, the OPH fractional-repair tilt
    target, and the selector-eliminated IR kernel. The report keeps the remaining
    finite repair-clock certificate visible instead of silently promoting the
    target to a physical CMB prediction.
    """

    if int(ell_max) < 2:
        raise ValueError("ell_max must be at least 2")
    if int(patch_count) <= 0:
        raise ValueError("patch_count must be positive")
    selector = selector_elimination_report(source_dir, P=float(P))
    pixel = OPHPixelConstants(P=float(P))
    canonical_kappa = math.e if kappa_rep is None else float(kappa_rep)
    delta_p = float(pixel.P - pixel.phi)
    eta_r = float(canonical_kappa * delta_p)
    n_s = float(1.0 - eta_r)
    q_ir = float(selector["cmb_ir_kernel"]["q_IR"])
    ell_ir = float(selector["cmb_ir_kernel"]["ell_IR"])
    n_frz = int(selector["cmb_ir_kernel"]["N_frz_proxy"])
    rows = _screen_rows(
        ell_max=int(ell_max),
        amplitude=float(amplitude),
        eta_r=eta_r,
        q_ir=q_ir,
        ell_ir=ell_ir,
        mu=float(mu),
    )
    audit = _spectrum_audit(rows, eta_r=eta_r)
    finite_capacity = _finite_regulator_capacity(int(patch_count), int(ell_max), n_frz=n_frz)
    repair_clock_certificate = bool(
        selector.get("scalar_tilt", {}).get("canonical_kappa_rep_status") == "certificate_passed"
    )
    theorem_source_receipt = bool(
        audit["eta0_flat_D_ell_receipt"]
        and selector.get("THEOREM_SIDE_SELECTOR_ELIMINATION_RECEIPT", False)
        and finite_capacity["bandlimit_for_ir_receipt"]
    )
    params = OPHScreenPowerParams(
        A_chi=float(amplitude),
        eta_R=eta_r,
        mu=float(mu),
        ell_cap=None,
        N_cap_eff=float(patch_count),
        q_IR=q_ir,
        ell_IR=ell_ir,
        epsilon_parity=0.0,
        ell_parity=8.0,
    )
    primordial_rows = _primordial_rows(
        params,
        k_count=int(primordial_k_count),
        k_min=float(primordial_k_min),
        k_max=float(primordial_k_max),
    )
    report: dict[str, Any] = {
        "mode": "oph_maxent_green_screen_source_v0",
        "source_dir": str(source_dir) if source_dir is not None else None,
        "oph_constants": pixel.as_jsonable(),
        "finite_regulator": finite_capacity,
        "maxent_inverse_laplacian": {
            "operator": "centered graph/sphere Laplacian Green function",
            "eta0_covariance": "C_ell = A_chi/[ell(ell+1)]",
            "eta0_D_ell": "ell(ell+1) C_ell/(2*pi) = A_chi/(2*pi)",
            "eta0_flat_D_ell_receipt": audit["eta0_flat_D_ell_receipt"],
            "eta0_D_ell_relative_std": audit["eta0_D_ell_relative_std"],
            "eta0_fit_eta_R": audit["eta0_fit_eta_R"],
        },
        "fractional_repair_tilt": {
            "formula": "eta_R = kappa_rep * alpha(0) * sqrt(pi) = kappa_rep * (P - phi)",
            "kappa_rep": canonical_kappa,
            "kappa_rep_source": "canonical_e_target" if kappa_rep is None else "cli_override",
            "repair_clock_certificate": repair_clock_certificate,
            "delta_P": delta_p,
            "eta_R": eta_r,
            "n_s": n_s,
            "fit_eta_R_from_generated_spectrum": audit["tilted_fit_eta_R"],
            "fit_n_s_from_generated_spectrum": audit["tilted_fit_n_s"],
            "fit_eta_R_abs_error": audit["tilted_fit_eta_R_abs_error"],
        },
        "selector_elimination_v1_5": {
            "q_IR": q_ir,
            "ell_IR": ell_ir,
            "N_frz_proxy": n_frz,
            "theorem_side_receipt": bool(selector.get("THEOREM_SIDE_SELECTOR_ELIMINATION_RECEIPT", False)),
            "source_packet_audit_receipt": bool(selector.get("SOURCE_PACKET_AUDIT_RECEIPT", False)),
            "q_IR_selector_removed": bool(
                selector.get("selector_elimination", {}).get("q_IR_selector_removed", False)
            ),
            "ell_IR_selector_removed": bool(
                selector.get("selector_elimination", {}).get("ell_IR_selector_removed", False)
            ),
            "eta_R_reduced_to_repair_clock_certificate": bool(
                selector.get("selector_elimination", {}).get("eta_R_reduced_to_repair_clock_certificate", False)
            ),
            "remaining_eta_R_certificate": selector.get("selector_elimination", {}).get(
                "remaining_eta_R_certificate"
            ),
            "exact_ir_kernel_csv_audit": {
                key: value
                for key, value in (selector.get("exact_ir_kernel_csv_audit", {}) or {}).items()
                if key != "rows"
            },
        },
        "screen_spectrum": {
            "ell_min": 2,
            "ell_max": int(ell_max),
            "row_count": len(rows),
            "A_chi": float(amplitude),
            "mu": float(mu),
            "mean_F_IR_ell2_29": _mean(row["F_IR"] for row in rows if 2 <= row["ell"] <= 29),
            "F_IR_ell2": _value_at_ell(rows, 2, "F_IR"),
            "F_IR_ell32": _value_at_ell(rows, 32, "F_IR"),
            "tilted_D_ell_slope_proxy": audit["tilted_log_D_slope"],
        },
        "primordial_bridge": {
            "status": "maxent_green_screen_to_primordial_table_emitted",
            "A_s": DEFAULT_A_S,
            "k0_mpc": DEFAULT_K0_MPC,
            "D_star_mpc": DEFAULT_D_STAR_MPC,
            "row_count": len(primordial_rows),
            "reference_source": "paper_maxent_green_spectrum_plus_selector_elimination",
            "simulator_eta_R_ready": False,
            "excludes": ["parity_envelope", "BipoSH_off_diagonal_covariance", "late-time_B_A_kernel"],
        },
        "reference_screen_parameters": params.as_jsonable(),
        "MAXENT_GREEN_SOURCE_RECEIPT": theorem_source_receipt,
        "finite_lattice_derived": False,
        "physical_cmb_prediction": False,
        "remaining_certificates": [
            "finite scalar repair-clock certificate kappa_rep=e",
            "finite normal-form scalar X_r emitted without CMB-target tuning",
            "finite freezeout branch emits q_IR=1/4 and ell_IR=32 from observer records",
            "parity/BipoSH angular covariance derived in a_lm space with masks",
            "OPH anomaly kernel B_A(k,a) and Gamma_rec(k,a) emitted for CAMB/CLASS",
            "official Planck likelihood/map-space tests",
        ],
        "claim_boundary": (
            "Paper-side finite-screen CMB source certificate. It verifies the MaxEnt Green-spectrum "
            "covariance and selector-eliminated q_IR/ell_IR target as a reproducible finite operator "
            "readout. The exact eta_R value uses the canonical kappa_rep=e target but the finite "
            "repair-clock certificate is still pending, so this is not a physical CMB prediction and "
            "not a final finite-lattice derivation."
        ),
    }
    report = with_claim_metadata(
        report,
        claim_level=QUANTITATIVE_BRANCH,
        receipt=SCREEN_PROXY_CMB_RECEIPT,
        physical_claim=False,
        observable_id="oph_maxent_green_screen_source",
        fit_objective="paper_maxent_inverse_laplacian_cmb_source_certificate",
    )
    report["_rows"] = rows
    report["_primordial_rows"] = primordial_rows
    report["_screen_power_scaffold"] = _screen_power_scaffold(report, params, primordial_rows)
    return report


def write_maxent_green_spectrum_report(
    out_dir: Path,
    *,
    source_dir: Path | None = None,
    patch_count: int = 262_144,
    ell_max: int = 256,
    P: float = P_STAR,
    kappa_rep: float | None = None,
    amplitude: float = 1.0,
    mu: float = 0.0,
    primordial_k_count: int = 256,
    primordial_k_min: float = 1.0e-4,
    primordial_k_max: float = 1.0,
) -> dict[str, Any]:
    report = maxent_green_spectrum_report(
        source_dir,
        patch_count=patch_count,
        ell_max=ell_max,
        P=P,
        kappa_rep=kappa_rep,
        amplitude=amplitude,
        mu=mu,
        primordial_k_count=primordial_k_count,
        primordial_k_min=primordial_k_min,
        primordial_k_max=primordial_k_max,
    )
    rows = list(report.pop("_rows"))
    primordial_rows = list(report.pop("_primordial_rows"))
    screen_power_scaffold = dict(report.pop("_screen_power_scaffold"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "maxent_green_spectrum_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "maxent_green_spectrum_report.md").write_text(_markdown_report(report), encoding="utf-8")
    (out / "oph_screen_power_report.json").write_text(
        json.dumps(screen_power_scaffold, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "oph_screen_power_report.md").write_text(
        _screen_power_markdown(screen_power_scaffold),
        encoding="utf-8",
    )
    _write_csv(out / "maxent_green_spectrum_rows.csv", rows)
    _write_csv(out / "oph_primordial_power_table.csv", primordial_rows)
    _write_primordial_txt(out / "oph_primordial_power_CLASS_CAMB.txt", primordial_rows)
    return report


def _screen_rows(
    *,
    ell_max: int,
    amplitude: float,
    eta_r: float,
    q_ir: float,
    ell_ir: float,
    mu: float,
) -> list[dict[str, float]]:
    ell = np.arange(2, int(ell_max) + 1, dtype=float)
    base = np.maximum(ell * (ell + 1.0) + float(mu) ** 2, 1.0e-30)
    c_eta0 = float(amplitude) / base
    d_eta0 = D_ell_from_C_ell(ell, c_eta0)
    c_tilt = float(amplitude) / (base ** (1.0 + float(eta_r) / 2.0))
    d_tilt = D_ell_from_C_ell(ell, c_tilt)
    denom = max(float(ell_ir) * (float(ell_ir) + 1.0), 1.0e-30)
    f_ir = 1.0 - float(q_ir) * np.exp(-(ell * (ell + 1.0)) / denom)
    return [
        {
            "ell": float(ell[index]),
            "C_ell_eta0_green": float(c_eta0[index]),
            "D_ell_eta0_green": float(d_eta0[index]),
            "C_ell_fractional_tilt": float(c_tilt[index]),
            "D_ell_fractional_tilt": float(d_tilt[index]),
            "F_IR": float(f_ir[index]),
            "C_ell_fractional_tilt_ir": float(c_tilt[index] * f_ir[index]),
            "D_ell_fractional_tilt_ir": float(d_tilt[index] * f_ir[index]),
        }
        for index in range(ell.size)
    ]


def _spectrum_audit(rows: list[dict[str, float]], *, eta_r: float) -> dict[str, Any]:
    eta0 = np.asarray([row["D_ell_eta0_green"] for row in rows], dtype=float)
    rel_std = float(np.std(eta0) / max(abs(float(np.mean(eta0))), 1.0e-300))
    eta0_fit = _eta_fit(rows, "D_ell_eta0_green")
    tilted_fit = _eta_fit(rows, "D_ell_fractional_tilt")
    return {
        "eta0_D_ell_relative_std": rel_std,
        "eta0_flat_D_ell_receipt": bool(rel_std <= 1.0e-12 and abs(eta0_fit) <= 1.0e-12),
        "eta0_fit_eta_R": eta0_fit,
        "tilted_fit_eta_R": tilted_fit,
        "tilted_fit_n_s": float(1.0 - tilted_fit),
        "tilted_fit_eta_R_abs_error": float(abs(tilted_fit - float(eta_r))),
        "tilted_log_D_slope": float(-tilted_fit),
    }


def _eta_fit(rows: list[dict[str, float]], key: str) -> float:
    ell = np.asarray([row["ell"] for row in rows], dtype=float)
    values = np.asarray([row[key] for row in rows], dtype=float)
    mask = (ell >= 20.0) & (values > 0.0)
    if int(np.sum(mask)) < 4:
        mask = values > 0.0
    x = 0.5 * np.log(np.maximum(ell[mask] * (ell[mask] + 1.0), 1.0e-30))
    y = np.log(np.maximum(values[mask], 1.0e-300))
    slope = float(np.polyfit(x, y, 1)[0])
    return float(-slope)


def _finite_regulator_capacity(patch_count: int, ell_max: int, *, n_frz: int) -> dict[str, Any]:
    max_bandlimited_ell = int(math.floor(math.sqrt(float(patch_count))) - 1)
    requested_slots = int((int(ell_max) + 1) ** 2)
    return {
        "mode": "numerical_regulator",
        "patch_count": int(patch_count),
        "requested_ell_max": int(ell_max),
        "requested_harmonic_slots": requested_slots,
        "sqrt_patch_bandlimit_proxy": max_bandlimited_ell,
        "bandlimit_for_requested_ell_receipt": bool(requested_slots <= int(patch_count)),
        "target_N_frz_proxy": int(n_frz),
        "bandlimit_for_ir_receipt": bool(int(patch_count) >= int(n_frz)),
        "claim_boundary": (
            "Patch count is treated as a numerical screen regulator. The slot test checks harmonic "
            "capacity only; it is not a claim that the regulator has cosmological horizon capacity."
        ),
    }


def _screen_power_scaffold(
    report: dict[str, Any],
    params: OPHScreenPowerParams,
    primordial_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    scaffold = {
        "mode": "oph_screen_power_effective_theory_v0",
        "source_run_count": 0,
        "fit_row_count": 0,
        "fit_rows": [],
        "aggregate": {
            "available_fit_count": 0,
            "field_summary": {},
            "best_planck_eta_diagnostic_field": None,
            "planck_eta_R_target": float(report["fractional_repair_tilt"]["eta_R"]),
            "planck_eta_R_sigma": 0.0,
        },
        "reference_mode": "paper-maxent-green",
        "simulator_primordial_reference_ready": False,
        "primordial_reference_source": "paper_maxent_green_spectrum_plus_selector_elimination_not_finite_lattice",
        "reference_screen_parameters": params.as_jsonable(),
        "primordial_bridge": {
            "status": "paper_maxent_green_to_primordial_table_emitted",
            "A_s": DEFAULT_A_S,
            "k0_mpc": DEFAULT_K0_MPC,
            "D_star_mpc": DEFAULT_D_STAR_MPC,
            "row_count": len(primordial_rows),
            "simulator_eta_R_ready": False,
            "reference_source": "paper_maxent_green_spectrum_plus_selector_elimination_not_finite_lattice",
            "excludes": ["parity_envelope", "BipoSH_off_diagonal_covariance", "late-time_B_A_kernel"],
            "claim_boundary": (
                "CAMB/CLASS scalar-table scaffold from the paper MaxEnt Green-spectrum target. "
                "It is not simulator-derived until the finite repair-clock and freezeout certificates pass."
            ),
        },
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Compatible OPH screen-power scaffold emitted by maxent_green_spectrum_report. "
            "It is a paper-side target source, not a finite simulated screen-field fit."
        ),
    }
    return with_claim_metadata(
        scaffold,
        claim_level=QUANTITATIVE_BRANCH,
        receipt=SCREEN_PROXY_CMB_RECEIPT,
        physical_claim=False,
        observable_id="oph_maxent_green_screen_source",
        fit_objective="paper_maxent_green_to_camb_scaffold",
    )


def _primordial_rows(
    params: OPHScreenPowerParams,
    *,
    k_count: int,
    k_min: float,
    k_max: float,
) -> list[dict[str, Any]]:
    k = np.geomspace(float(k_min), float(k_max), int(k_count))
    power = primordial_power_oph(k, params)
    return [
        {
            "k_mpc": float(power["k_mpc"][index]),
            "ell_proxy": float(power["ell_proxy"][index]),
            "P_R": float(power["P_R"][index]),
            "P_R_base": float(power["P_R_base"][index]),
            "F_IR": float(power["F_IR"][index]),
            "F_cap": float(power["F_cap"][index]),
            "F_OPH": float(power["F_OPH"][index]),
            "eta_R": float(params.eta_R),
            "n_s_proxy": float(params.n_s_proxy),
        }
        for index in range(k.size)
    ]


def _value_at_ell(rows: list[dict[str, float]], ell: int, key: str) -> float | None:
    for row in rows:
        if int(round(row["ell"])) == int(ell):
            return float(row[key])
    return None


def _mean(values: Any) -> float | None:
    items = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not items:
        return None
    return float(fmean(items))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row if not isinstance(row.get(key), (dict, list))})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _write_primordial_txt(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# OPH MaxEnt Green-spectrum primordial scaffold",
        "# columns: k_mpc P_R F_OPH F_IR F_cap ell_proxy",
        "# claim_boundary: paper-side target source; not a finite-lattice physical CMB prediction",
    ]
    for row in rows:
        lines.append(
            f"{row['k_mpc']:.12e} {row['P_R']:.12e} {row['F_OPH']:.12e} "
            f"{row['F_IR']:.12e} {row['F_cap']:.12e} {row['ell_proxy']:.12e}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# OPH MaxEnt Green-Spectrum Source",
            "",
            report["claim_boundary"],
            "",
            "## Readout",
            "",
            f"- patch count: {report['finite_regulator']['patch_count']}",
            f"- ell max: {report['screen_spectrum']['ell_max']}",
            f"- eta_R: {report['fractional_repair_tilt']['eta_R']:.12g}",
            f"- n_s: {report['fractional_repair_tilt']['n_s']:.12g}",
            f"- q_IR: {report['selector_elimination_v1_5']['q_IR']:.12g}",
            f"- ell_IR: {report['selector_elimination_v1_5']['ell_IR']:.12g}",
            f"- MaxEnt Green receipt: {report['MAXENT_GREEN_SOURCE_RECEIPT']}",
            f"- repair-clock certificate: {report['fractional_repair_tilt']['repair_clock_certificate']}",
            f"- physical CMB prediction: {report['physical_cmb_prediction']}",
            "",
            "## Remaining Certificates",
            "",
            *[f"- {item}" for item in report["remaining_certificates"]],
            "",
        ]
    )


def _screen_power_markdown(report: dict[str, Any]) -> str:
    params = report["reference_screen_parameters"]
    return "\n".join(
        [
            "# OPH Screen Power Scaffold",
            "",
            report["claim_boundary"],
            "",
            f"- reference source: {report['primordial_reference_source']}",
            f"- eta_R: {params['eta_R']:.12g}",
            f"- n_s proxy: {params['n_s_proxy']:.12g}",
            f"- q_IR: {params['q_IR']:.12g}",
            f"- ell_IR: {params['ell_IR']:.12g}",
            f"- simulator eta ready: {report['simulator_primordial_reference_ready']}",
            f"- physical CMB prediction: {report['physical_cmb_prediction']}",
            "",
        ]
    )
