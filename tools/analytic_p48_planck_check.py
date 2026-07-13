#!/usr/bin/env python3
"""Run the conditional analytic n_s=1-P/48 branch through CAMB and Planck TT bins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.camb_adapter import write_oph_inflation_cmb_camb_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the conditional analytic OPH scalar tilt n_s=1-P/48 with "
            "the local Planck PR3 binned TT table using conventional CAMB transfer inputs."
        )
    )
    parser.add_argument(
        "--planck-tt",
        type=Path,
        default=Path("data/measurements/planck2018/COM_PowerSpect_CMB-TT-binned_R3.01.txt"),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lmax", type=int, default=2600)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    theta = float(P_STAR / 48.0)
    bridge = {
        "mode": "conditional_analytic_p48_planck_check_input",
        "screen_spectrum_prediction": {
            "P": float(P_STAR),
            "theta_OPH": theta,
            "n_s": float(1.0 - theta),
            "A_zeta": None,
            "screen_to_primordial_lift_receipt": False,
            "coefficient_status": "conditional_paper_side_P_over_48_target",
        },
        "cmb_success_ladder": {"core_numbers": {}},
        "physical_prediction_receipt": False,
        "claim_boundary": (
            "This freezes the paper-side P/48 arithmetic candidate only. The scalar-source theorem, "
            "screen-to-primordial radial lift, amplitude, conventional background inputs, and official "
            "Planck likelihood remain open or imported."
        ),
    }
    bridge_path = args.out / "analytic_p48_bridge_input.json"
    bridge_path.write_text(json.dumps(bridge, indent=2) + "\n", encoding="utf-8")
    report = write_oph_inflation_cmb_camb_report(
        bridge_path,
        args.planck_tt,
        args.out,
        lmax=args.lmax,
        benchmark_label="Planck2018_TT_binned_PR3_conditional_P48",
    )
    baseline = report["comparison"]["camb_lcdm_powerlaw"]
    p48 = report["comparison"]["oph_p48_powerlaw"]
    p48_total = float(p48["amplitude_fit_chi2_per_bin"]) * int(p48["bin_count"])
    baseline_total = float(baseline["amplitude_fit_chi2_per_bin"]) * int(
        baseline["bin_count"]
    )
    delta = p48_total - baseline_total
    summary = {
        "P": float(P_STAR),
        "theta_P_over_48": theta,
        "n_s_P_over_48": float(1.0 - theta),
        "planck_reference_n_s": 0.9649,
        "planck_reference_sigma_n_s": 0.0042,
        "arithmetic_pull_sigma": (float(1.0 - theta) - 0.9649) / 0.0042,
        "bin_count": int(p48["bin_count"]),
        "p48_amplitude_fit_chi2_per_bin": float(p48["amplitude_fit_chi2_per_bin"]),
        "baseline_amplitude_fit_chi2_per_bin": float(
            baseline["amplitude_fit_chi2_per_bin"]
        ),
        "p48_total_diagonal_chi2": p48_total,
        "baseline_total_diagonal_chi2": baseline_total,
        "p48_minus_baseline_total_diagonal_chi2": delta,
        "comparison_receipt": True,
        "physical_prediction_receipt": False,
        "claim_boundary": bridge["claim_boundary"],
    }
    (args.out / "analytic_p48_planck_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
