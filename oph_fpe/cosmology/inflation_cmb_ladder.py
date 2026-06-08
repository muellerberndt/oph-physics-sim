from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from oph_fpe.claims import CONTINUATION, COSMOLOGY_PERTURBATION_RECEIPT, with_claim_metadata
from oph_fpe.constants.oph_pixel import P_STAR, OPHPixelConstants
from oph_fpe.cosmology.unique_predictions import unique_prediction_gate_report


PLANCK_NS_TARGET = 0.965
PLANCK_NS_SIGMA = 0.004
PLANCK_AS_REFERENCE = 2.1e-9
EPSILON_CMB_STAR_BITS = 3.61e-11


def screen_spectrum_prediction(
    *,
    P: float = P_STAR,
    epsilon_star_bits: float = EPSILON_CMB_STAR_BITS,
    sachs_wolfe_curvature_ratio: float = 5.0,
) -> dict[str, Any]:
    """Return the Pro v2 OPH screen-spectrum theorem-target numbers.

    The note's scalar branch is

        Delta_zeta^2(k) = A_zeta (k/k_*)^(-P/48)

    with A_zeta = R_zeta^2 * 4 ln(2) * epsilon_star and R_zeta ~= 5
    from the Sachs-Wolfe temperature-to-curvature conversion.
    """

    theta_oph = float(P) / 48.0
    lambda_collar = math.exp(-float(P) / 24.0)
    a_temperature = 4.0 * math.log(2.0) * float(epsilon_star_bits)
    a_zeta = float(sachs_wolfe_curvature_ratio) ** 2 * a_temperature
    ns = 1.0 - theta_oph
    return {
        "mode": "oph_screen_green_spectrum_theorem_target",
        "P": float(P),
        "lambda_collar": float(lambda_collar),
        "theta_OPH": float(theta_oph),
        "n_s": float(ns),
        "planck_n_s_target": PLANCK_NS_TARGET,
        "planck_n_s_sigma": PLANCK_NS_SIGMA,
        "n_s_pull_vs_planck": float((ns - PLANCK_NS_TARGET) / PLANCK_NS_SIGMA),
        "epsilon_star_bits": float(epsilon_star_bits),
        "A_T_temperature": float(a_temperature),
        "sachs_wolfe_curvature_ratio": float(sachs_wolfe_curvature_ratio),
        "A_zeta": float(a_zeta),
        "Planck_A_s_reference": PLANCK_AS_REFERENCE,
        "A_zeta_over_Planck_A_s_reference": float(a_zeta / PLANCK_AS_REFERENCE),
        "primordial_power_law": "Delta_zeta^2(k)=A_zeta*(k/k_star)^(-P/48)",
        "claim_boundary": (
            "Theorem-target scalar screen spectrum from the Pro inflation/CMB notes. This is a "
            "quantitative OPH continuation target, not evidence that the current finite lattice "
            "has derived the scalar spectrum from collar microphysics."
        ),
    }


def flat_sector_selection_report(
    *,
    omega_lambda_oph: float = 0.684095,
    omega_b0: float = 0.04931,
    omega_nu0: float = 0.00230,
    omega_r0: float = 9.2e-5,
) -> dict[str, Any]:
    omega_a0 = 1.0 - float(omega_lambda_oph) - float(omega_b0) - float(omega_nu0) - float(omega_r0)
    return {
        "mode": "oph_flat_zero_holonomy_sector_selector",
        "selected_curvature_holonomy": 0.0,
        "selected_K": 0.0,
        "selected_Omega_K": 0.0,
        "inputs": {
            "Omega_Lambda_OPH": float(omega_lambda_oph),
            "Omega_b0": float(omega_b0),
            "Omega_nu0": float(omega_nu0),
            "Omega_r0": float(omega_r0),
        },
        "Omega_A0_residual": float(omega_a0),
        "rho_A_over_rho_b": float(omega_a0 / omega_b0) if omega_b0 else None,
        "selector_statement": (
            "Nonzero FLRW spatial curvature is treated as visible geometric holonomy. In the ordinary "
            "cosmological boundary sector with no independent curvature charge, MAR zero-obstruction "
            "selection chooses h_K=0, hence K=0 and Omega_K=0."
        ),
        "claim_boundary": (
            "Flat-sector theorem target from the Pro notes. This is a branch-selection statement; "
            "a finite simulation still has to emit curvature-holonomy receipts and boundary-sector "
            "bookkeeping before it becomes a lattice-derived result."
        ),
    }


def cmb_success_ladder_report(source_dir: Path) -> dict[str, Any]:
    source = Path(source_dir)
    summary_path = _first_existing(
        source,
        "oph_cmb_success_summary_v0_4.json",
        "2/oph_cmb_success_summary_v0_4.json",
        "3/oph_cmb_success_summary_v0_4.json",
    )
    model_path = _first_existing(
        source,
        "01_model_selection_lowell_and_high_ell_v0_4.csv",
        "2/01_model_selection_lowell_and_high_ell_v0_4.csv",
        "3/01_model_selection_lowell_and_high_ell_v0_4.csv",
    )
    mc_path = _first_existing(
        source,
        "oph_lowell_fullsky_mc_summary_v0_4.csv",
        "2/oph_lowell_fullsky_mc_summary_v0_4.csv",
        "3/oph_lowell_fullsky_mc_summary_v0_4.csv",
    )
    hard_gate_paths = {
        "tt_te_ee_diagonal": _first_existing(
            source,
            "01_TT_TE_EE_diagonal_proxy_chi2_v0_5.csv",
            "3/01_TT_TE_EE_diagonal_proxy_chi2_v0_5.csv",
        ),
        "combined_lowell": _first_existing(
            source,
            "02_combined_lowell_TT_TE_EE_proxy_v0_5.csv",
            "3/02_combined_lowell_TT_TE_EE_proxy_v0_5.csv",
        ),
        "parity_lowpower": _first_existing(
            source,
            "04_parity_lowpower_statistics_TT_TE_EE_v0_5.csv",
            "3/04_parity_lowpower_statistics_TT_TE_EE_v0_5.csv",
        ),
        "scalar_s12": _first_existing(
            source,
            "05_scalar_S12_correlation_proxy_TT_EE_v0_5.csv",
            "3/05_scalar_S12_correlation_proxy_TT_EE_v0_5.csv",
        ),
        "map_likelihood_gates": _first_existing(
            source,
            "06_planck_map_and_likelihood_data_gates_v0_5.csv",
            "3/06_planck_map_and_likelihood_data_gates_v0_5.csv",
        ),
        "status_ledger": _first_existing(
            source,
            "07_success_falsifier_status_ledger_v0_5.csv",
            "3/07_success_falsifier_status_ledger_v0_5.csv",
        ),
    }
    summary = _read_json(summary_path)
    model_rows = _read_csv(model_path)
    mc_rows = _read_csv(mc_path)
    hard_gate_rows = {name: _read_csv(path) for name, path in hard_gate_paths.items()}
    core = summary.get("core_numbers", {}) if isinstance(summary, dict) else {}
    return {
        "mode": "oph_cmb_success_ladder_v0_4_v0_5_import",
        "source_dir": str(source),
        "files": {
            "summary_json": str(summary_path),
            "model_selection_csv": str(model_path),
            "fullsky_mc_csv": str(mc_path),
            **{name: str(path) for name, path in hard_gate_paths.items()},
        },
        "files_present": {
            "summary_json": summary_path.exists(),
            "model_selection_csv": model_path.exists(),
            "fullsky_mc_csv": mc_path.exists(),
            **{name: path.exists() for name, path in hard_gate_paths.items()},
        },
        "source_claim_boundary": summary.get("claim_boundary") if isinstance(summary, dict) else None,
        "success_criteria_met": list(summary.get("success_criteria_met", [])) if isinstance(summary, dict) else [],
        "core_numbers": core,
        "low_ell_model_selection": _model_selection_summary(model_rows),
        "fullsky_monte_carlo": _mc_summary(mc_rows),
        "hard_gates_v0_5": _hard_gate_summary(hard_gate_rows),
        "diagnostic_cmb_data_available": bool(
            (summary_path.exists() and model_path.exists() and mc_path.exists())
            or any(path.exists() for path in hard_gate_paths.values())
        ),
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Imports the Pro OPH-CMB v0.4/v0.5 public-data diagnostic ladder: Planck low-ell TT fit, "
            "CAMB-transferred IR kernel, ideal full-sky anomaly MC, and TT/TE/EE hard-gate proxy tables. "
            "This is measured-data comparison, but not an official Planck likelihood, not a completed "
            "masked-map pipeline, and not a finite-lattice derivation of q_IR, ell_IR, or parity parameters."
        ),
    }


def inflation_cmb_bridge_report(source_dir: Path | None = None) -> dict[str, Any]:
    screen = screen_spectrum_prediction()
    flat = flat_sector_selection_report()
    cmb = cmb_success_ladder_report(Path(source_dir)) if source_dir is not None else None
    unique = unique_prediction_gate_report(Path(source_dir)) if source_dir is not None else unique_prediction_gate_report()
    report = {
        "mode": "oph_inflation_cmb_bridge_v0",
        "oph_constants": OPHPixelConstants().as_jsonable(),
        "flat_sector_selection": flat,
        "screen_spectrum_prediction": screen,
        "unique_prediction_gate_v0_9": unique,
        "cmb_success_ladder": cmb,
        "inflation_replacement_modules": {
            "flatness": "zero visible curvature holonomy sector, Omega_K=0",
            "spectrum": "legacy screen Green target n_s=1-P/48 plus current unique target n_s=1-e*alpha(0)*sqrt(pi)",
            "amplitude": "collar-error amplitude selector A_zeta=100 ln(2) epsilon_star",
            "transfer": "standard photon-baryon Boltzmann transfer with an OPH anomaly stress component",
            "hot_start": "hot MaxEnt release state, not inflaton reheating",
            "adiabaticity": "same-boundary scalar normal form gives S_ij=0 up to synchronization residue",
        },
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Integrated theorem-target and measured-diagnostic bridge from the Pro CMB/inflation notes. "
            "It gives comparable numbers now, but it is not yet a proof that the finite OPH lattice "
            "derives the same numbers from state-derived cap/collar microphysics."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=CONTINUATION,
        receipt=COSMOLOGY_PERTURBATION_RECEIPT,
        physical_claim=False,
        observable_id="oph_inflation_cmb_bridge",
        fit_objective="p48_screen_spectrum_and_v04_cmb_diagnostic_import",
    )


def write_inflation_cmb_bridge_report(source_dir: Path | None, out_dir: Path) -> dict[str, Any]:
    report = inflation_cmb_bridge_report(source_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "oph_inflation_cmb_bridge_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "oph_inflation_cmb_bridge_report.md").write_text(_markdown_report(report), encoding="utf-8")
    cmb = report.get("cmb_success_ladder") or {}
    if cmb:
        _write_csv(out / "oph_cmb_v04_model_selection_summary.csv", cmb["low_ell_model_selection"].get("rows", []))
        _write_csv(out / "oph_cmb_v04_fullsky_mc_summary.csv", cmb["fullsky_monte_carlo"].get("rows", []))
        hard_gates = cmb.get("hard_gates_v0_5", {})
        _write_csv(out / "oph_cmb_v05_status_ledger.csv", hard_gates.get("ledger", []))
    unique = report.get("unique_prediction_gate_v0_9", {})
    if unique:
        _write_csv(out / "oph_unique_prediction_ranking_rows.csv", unique.get("ranking_rows", []))
        _write_csv(out / "oph_unique_prediction_assessment_rows.csv", unique.get("assessment_rows", []))
    return report


def _first_existing(source: Path, *relative_paths: str) -> Path:
    candidates = [source / relative for relative in relative_paths]
    return next((path for path in candidates if path.exists()), candidates[0])


def _model_selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    low = [row for row in rows if _int(row.get("ell_min")) == 2 and _int(row.get("ell_max")) == 29]
    by_model = {str(row.get("model")): row for row in low}
    return {
        "row_count": len(rows),
        "low_ell_row_count": len(low),
        "rows": low,
        "camb_lcdm_chi2": _float((by_model.get("CAMB_LCDM_powerlaw") or {}).get("chi2_diag")),
        "oph_ir_bestfit_chi2": _float((by_model.get("OPH_IR_bestfit_lowell_q0p2446_ell33p615") or {}).get("chi2_diag")),
        "oph_joint_ir_chi2": _float((by_model.get("OPH_IR_joint_IRpart_q0p1670_ell67p979") or {}).get("chi2_diag")),
        "best_delta_aic_vs_lcdm": min(
            (_float(row.get("delta_AIC_vs_LCDM")) for row in low if _float(row.get("delta_AIC_vs_LCDM")) is not None),
            default=None,
        ),
    }


def _mc_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model = {str(row.get("model")): row for row in rows}
    lcdm = by_model.get("LCDM_bestfit_theory") or {}
    joint = by_model.get("OPH_IR_plus_parity_bestfit") or {}
    parity = by_model.get("OPH_parity_bestfit") or {}
    return {
        "row_count": len(rows),
        "rows": rows,
        "LCDM_PTE_R_OE_upper": _float(lcdm.get("PTE_R_OE_upper_tail")),
        "OPH_parity_PTE_R_OE_upper": _float(parity.get("PTE_R_OE_upper_tail")),
        "OPH_joint_PTE_R_OE_upper": _float(joint.get("PTE_R_OE_upper_tail")),
        "LCDM_PTE_S_1_2_lower": _float(lcdm.get("PTE_S_1_2_lower_tail")),
        "OPH_joint_PTE_S_1_2_lower": _float(joint.get("PTE_S_1_2_lower_tail")),
    }


def _hard_gate_summary(rows_by_name: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    diagonal = rows_by_name.get("tt_te_ee_diagonal", [])
    combined = rows_by_name.get("combined_lowell", [])
    ledger = rows_by_name.get("status_ledger", [])
    best_model = "OPH_IR_bestfit_lowell_q0p2446_ell33p615"
    summary = {
        "files_available": {name: bool(rows) for name, rows in rows_by_name.items()},
        "row_counts": {name: len(rows) for name, rows in rows_by_name.items()},
        "TT_lowell_delta_chi2": _delta_chi2(diagonal, "TT", 2, 29, best_model),
        "TE_lowell_delta_chi2": _delta_chi2(diagonal, "TE", 2, 29, best_model),
        "EE_lowell_delta_chi2": _delta_chi2(diagonal, "EE", 2, 29, best_model),
        "TT_high_ell_delta_chi2_30_1200": _delta_chi2(diagonal, "TT", 30, 1200, best_model),
        "combined_TT_TE_EE_lowell_delta_chi2": _delta_chi2(
            combined,
            "TT+TE+EE",
            2,
            29,
            best_model,
            spectrum_key="spectra",
        ),
        "ledger": ledger,
        "pressure_points": [
            row
            for row in ledger
            if "pressure" in str(row.get("v0_5_status", "")).lower()
            or "not yet" in str(row.get("v0_5_status", "")).lower()
        ],
        "claim_boundary": (
            "v0.5 hard-gate proxy import. Positive delta chi2 means improvement over the CAMB LCDM "
            "power-law baseline in the public-spectrum diagonal proxy. It is not an official Planck "
            "likelihood and phase-sensitive map-space gates still require component maps and masks."
        ),
    }
    return summary


def _delta_chi2(
    rows: list[dict[str, Any]],
    spectrum: str,
    ell_min: int,
    ell_max: int,
    model: str,
    *,
    spectrum_key: str = "spectrum",
) -> float | None:
    for row in rows:
        if (
            str(row.get(spectrum_key)) == spectrum
            and _int(row.get("ell_min")) == ell_min
            and _int(row.get("ell_max")) == ell_max
            and str(row.get("model")) == model
        ):
            return _float(row.get("delta_chi2_improvement_vs_LCDM"))
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(report: dict[str, Any]) -> str:
    screen = report["screen_spectrum_prediction"]
    flat = report["flat_sector_selection"]
    cmb = report.get("cmb_success_ladder") or {}
    core = cmb.get("core_numbers", {})
    lines = [
        "# OPH Inflation/CMB Bridge",
        "",
        report["claim_boundary"],
        "",
        "## Screen Spectrum",
        "",
        f"- P: {screen['P']:.16g}",
        f"- theta_OPH = P/48: {screen['theta_OPH']:.12g}",
        f"- n_s = 1 - P/48: {screen['n_s']:.12g}",
        f"- Planck n_s pull: {screen['n_s_pull_vs_planck']:.4g} sigma",
        f"- A_zeta selector: {screen['A_zeta']:.12g}",
        "",
        "## Flat Sector",
        "",
        f"- Omega_K selected: {flat['selected_Omega_K']}",
        f"- Omega_A0 residual: {flat['Omega_A0_residual']:.12g}",
        f"- rho_A/rho_b: {flat['rho_A_over_rho_b']:.12g}",
        "",
        "## CMB v0.4 Diagnostic Import",
        "",
        f"- source present: {bool(cmb.get('diagnostic_cmb_data_available'))}",
        f"- q_IR best: {core.get('v0_2_IR_bestfit_q_IR', 'n/a')}",
        f"- ell_IR best: {core.get('v0_2_IR_bestfit_ell_IR', 'n/a')}",
        f"- CAMB low-ell LCDM chi2: {core.get('v0_3_camb_lowell_LCDM_chi2_ell2_29', 'n/a')}",
        f"- CAMB low-ell OPH IR chi2: {core.get('v0_3_camb_lowell_IR_bestfit_chi2_ell2_29', 'n/a')}",
        f"- LCDM R_OE PTE upper: {core.get('v0_4_LCDM_PTE_R_OE_upper', 'n/a')}",
        f"- OPH parity R_OE PTE upper: {core.get('v0_4_parity_PTE_R_OE_upper', 'n/a')}",
        "",
        "## CMB v0.5 Hard Gates",
        "",
        f"- TT low-ell delta chi2: {cmb.get('hard_gates_v0_5', {}).get('TT_lowell_delta_chi2', 'n/a')}",
        f"- TE low-ell delta chi2: {cmb.get('hard_gates_v0_5', {}).get('TE_lowell_delta_chi2', 'n/a')}",
        f"- EE low-ell delta chi2: {cmb.get('hard_gates_v0_5', {}).get('EE_lowell_delta_chi2', 'n/a')}",
        f"- TT high-ell delta chi2, ell=30..1200: {cmb.get('hard_gates_v0_5', {}).get('TT_high_ell_delta_chi2_30_1200', 'n/a')}",
        f"- combined TT+TE+EE low-ell delta chi2: {cmb.get('hard_gates_v0_5', {}).get('combined_TT_TE_EE_lowell_delta_chi2', 'n/a')}",
        "",
        "## Output Files",
        "",
        "- `oph_inflation_cmb_bridge_report.json`",
        "- `oph_cmb_v04_model_selection_summary.csv`",
        "- `oph_cmb_v04_fullsky_mc_summary.csv`",
        "",
    ]
    return "\n".join(lines)


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
