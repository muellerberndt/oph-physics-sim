from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from oph_fpe.constants.oph_pixel import P_STAR, total_entropy_capacity

DEFAULT_R_DS_M = 1.66e26
DEFAULT_L_PLANCK_M = 1.616e-35
DEFAULT_REGULATOR_PATCH_COUNTS = (4_096, 65_536, 262_144, 1_048_576)


def bare_horizon_area_ratio(radius_m: float = DEFAULT_R_DS_M, planck_length_m: float = DEFAULT_L_PLANCK_M) -> float:
    return float((float(radius_m) / float(planck_length_m)) ** 2)


def entropy_capacity_from_radius(radius_m: float = DEFAULT_R_DS_M, planck_length_m: float = DEFAULT_L_PLANCK_M) -> float:
    return math.pi * bare_horizon_area_ratio(radius_m, planck_length_m)


def lambda_planck2_from_capacity(n_crc: float) -> float:
    return 3.0 * math.pi / float(n_crc)


def physical_cells_for_entropy_capacity(n_scr: float, p_value: float = P_STAR) -> float:
    """Cells needed when each local cell carries entropy capacity P/4."""

    return 4.0 * float(n_scr) / float(p_value)


def screen_capacity_closure_report(
    *,
    p_value: float = P_STAR,
    radius_m: float = DEFAULT_R_DS_M,
    planck_length_m: float = DEFAULT_L_PLANCK_M,
    regulator_patch_counts: tuple[int, ...] = DEFAULT_REGULATOR_PATCH_COUNTS,
) -> dict[str, Any]:
    n_patch = bare_horizon_area_ratio(radius_m, planck_length_m)
    n_scr = entropy_capacity_from_radius(radius_m, planck_length_m)
    lambda_l_planck2 = lambda_planck2_from_capacity(n_scr)
    physical_cells = physical_cells_for_entropy_capacity(n_scr, p_value)
    return {
        "mode": "oph_screen_capacity_closure_v0",
        "source": "observers_are_all_you_need.tex cosmic record-capacity closure",
        "closure_equations": {
            "cosmic_record_closure": "N_CRC = F(N_CRC)",
            "readback_map": "F(N)=Cap_read(Obs(nf(U_N)))",
            "lambda_readout": "Lambda_CRC * l_P^2 = 3*pi / N_CRC",
            "count_density_selector": "N_star = MAR argmax_N [log|Omega_N^sc| - N]",
        },
        "observed_branch_normalization": {
            "R_dS_m": float(radius_m),
            "planck_length_m": float(planck_length_m),
            "N_patch_bare_radius_squared_ratio": n_patch,
            "N_scr_entropy_capacity": n_scr,
            "Lambda_lP2": lambda_l_planck2,
            "N_cells_if_tiled_by_local_P_cells": physical_cells,
            "cell_entropy_capacity": float(p_value) / 4.0,
            "P": float(p_value),
        },
        "regulator_scale_comparison": [
            {
                "patch_count": int(count),
                "regulator_entropy_capacity": total_entropy_capacity(int(count), p_value),
                "fraction_of_observed_N_scr": total_entropy_capacity(int(count), p_value) / n_scr,
                "claim_boundary": "finite simulation regulator count, not the cosmic record capacity",
            }
            for count in regulator_patch_counts
        ],
        "readiness_gates": {
            "local_P_cell_capacity_available": True,
            "observed_branch_N_scr_readout_available": True,
            "F_N_readback_map_implemented": False,
            "count_density_normal_form_enumerator_implemented": False,
            "N_CRC_fixed_point_solved_from_finite_simulator": False,
            "Lambda_from_finite_simulator_record_closure": False,
        },
        "simulation_relevance": (
            "This closure is globally relevant to cosmology and capacity normalization. It should not be "
            "used to reinterpret ordinary finite run patch counts as physical horizon capacity. Current "
            "64k/256k/1M runs remain numerical regulators unless a dedicated readback map F(N) and "
            "self-closure normal-form enumeration are implemented."
        ),
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "claim_boundary": (
            "Observed-branch screen-capacity closure report. Computes the de Sitter entropy-capacity "
            "normalization and Lambda*l_P^2 readout from the paper equations, but does not solve the "
            "OPH readback fixed point from simulator data."
        ),
    }


def write_screen_capacity_closure_report(out_dir: Path, **kwargs: Any) -> dict[str, Any]:
    report = screen_capacity_closure_report(**kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "screen_capacity_closure_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "screen_capacity_closure_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    observed = report["observed_branch_normalization"]
    gates = report["readiness_gates"]
    return "\n".join(
        [
            "# OPH Screen-Capacity Closure",
            "",
            str(report["claim_boundary"]),
            "",
            "## Observed Branch",
            "",
            f"- N_patch bare ratio: `{observed['N_patch_bare_radius_squared_ratio']:.6e}`",
            f"- N_scr entropy capacity: `{observed['N_scr_entropy_capacity']:.6e}`",
            f"- Lambda l_P^2: `{observed['Lambda_lP2']:.6e}`",
            f"- local P-cell count for N_scr: `{observed['N_cells_if_tiled_by_local_P_cells']:.6e}`",
            "",
            "## Gates",
            "",
            *[f"- {key}: `{str(value).lower()}`" for key, value in gates.items()],
            "",
            "## Simulation Relevance",
            "",
            str(report["simulation_relevance"]),
            "",
        ]
    )
