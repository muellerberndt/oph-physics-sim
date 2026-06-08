from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from oph_fpe.claims import PROXY, COSMOLOGY_PERTURBATION_RECEIPT, with_claim_metadata


OPH_COMPRESSED_POINT = {
    "Omega_m": 0.315905207,
    "sigma_8": 0.807787208,
    "H0": 67.4,
    "S_8": 0.8289240425,
    "chi2_diag": 11.46326129,
}

COMPRESSED_ROWS = [
    {"row": "Weak-lensing S8", "pull": 2.433, "chi2": 5.918},
    {"row": "DESI_DR1_BAO_BBN_theta_star_H0", "pull": -1.806, "chi2": 3.263},
    {"row": "DESI_DR1_BAO_Omega_m", "pull": 1.394, "chi2": 1.942},
    {"row": "Planck_sigma8", "pull": -0.535, "chi2": 0.287},
    {"row": "Planck_S8", "pull": -0.190, "chi2": 0.036},
    {"row": "Planck_Omega_m", "pull": 0.129, "chi2": 0.017},
]

COMPRESSED_SCAN_POINTS = [
    {
        "case": "OPH compressed point",
        "Omega_m": 0.315905207,
        "sigma_8": 0.807787208,
        "H0": 67.4,
        "S_8": 0.8289240425,
        "chi2": 11.4633,
    },
    {
        "case": "Best grid point, H0 fixed",
        "Omega_m": 0.309000000,
        "sigma_8": 0.809583333,
        "H0": 67.4,
        "S_8": 0.8216373463,
        "chi2": 9.5504,
    },
    {
        "case": "Free compressed best",
        "Omega_m": 0.309020500,
        "sigma_8": 0.809357206,
        "H0": 68.519973,
        "S_8": 0.8214350999,
        "chi2": 6.2852,
    },
]


def compressed_likelihood_reference_report() -> dict[str, Any]:
    row_sum = float(sum(float(row["chi2"]) for row in COMPRESSED_ROWS))
    pull_sum = float(sum(float(row["pull"]) ** 2 for row in COMPRESSED_ROWS))
    report = {
        "mode": "oph_compressed_cmb_bao_growth_reference_v0",
        "oph_compressed_point": dict(OPH_COMPRESSED_POINT),
        "row_contributions": list(COMPRESSED_ROWS),
        "row_chi2_sum_from_rounded_table": row_sum,
        "pull_chi2_sum_from_rounded_pulls": pull_sum,
        "scan_points": list(COMPRESSED_SCAN_POINTS),
        "acceptance": {
            "reference_chi2_target": 11.46326129,
            "reproduces_reference_chi2": math.isclose(
                float(OPH_COMPRESSED_POINT["chi2_diag"]), 11.46326129, rel_tol=0.0, abs_tol=1.0e-10
            ),
            "weak_lensing_S8_tension_visible": True,
            "full_likelihood_required": True,
        },
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "claim_boundary": (
            "Reference compressed CMB/BAO/growth/S8 diagnostic from the research note. It preserves the "
            "OPH compressed point and weak-lensing S8 tension for regression testing. It is not a full "
            "Boltzmann likelihood, not a Planck likelihood, and not a physical OPH prediction by itself."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=PROXY,
        receipt=COSMOLOGY_PERTURBATION_RECEIPT,
        physical_claim=False,
        observable_id="oph_compressed_cmb_bao_growth_reference",
        fit_objective="compressed_measurement_surface_regression_reference",
    )


def write_compressed_likelihood_reference_report(out_dir: Path) -> dict[str, Any]:
    report = compressed_likelihood_reference_report()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "oph_compressed_likelihood_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (out / "oph_compressed_likelihood_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "oph_compressed_likelihood_rows.csv", report["row_contributions"])
    _write_csv(out / "oph_compressed_likelihood_scan_points.csv", report["scan_points"])
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(report: dict[str, Any]) -> str:
    point = report["oph_compressed_point"]
    lines = [
        "# OPH Compressed CMB/BAO/Growth Reference",
        "",
        report["claim_boundary"],
        "",
        "## OPH Compressed Point",
        "",
        f"- Omega_m: {point['Omega_m']}",
        f"- sigma_8: {point['sigma_8']}",
        f"- H0: {point['H0']}",
        f"- S_8: {point['S_8']}",
        f"- chi2_diag: {point['chi2_diag']}",
        "",
        "## Largest Contribution",
        "",
        f"- {report['row_contributions'][0]['row']}: chi2={report['row_contributions'][0]['chi2']}",
        "",
        "## Output Files",
        "",
        "- `oph_compressed_likelihood_report.json`",
        "- `oph_compressed_likelihood_rows.csv`",
        "- `oph_compressed_likelihood_scan_points.csv`",
        "",
    ]
    return "\n".join(lines)
