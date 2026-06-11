from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oph_fpe.constants.oph_pixel import P_STAR, total_entropy_capacity

DEFAULT_R_DS_M = 1.66e26
DEFAULT_L_PLANCK_M = 1.616e-35
DEFAULT_REGULATOR_PATCH_COUNTS = (4_096, 65_536, 262_144, 1_048_576)
DEFAULT_N_CRC = math.pi * (DEFAULT_R_DS_M / DEFAULT_L_PLANCK_M) ** 2


@dataclass(frozen=True)
class OPHScreenCapacityConstants:
    """Global capacity closure value, separate from finite regulator patch count."""

    n_crc: float = DEFAULT_N_CRC
    p_value: float = P_STAR
    source: str = "observed_branch_public"

    @property
    def n_patch_bare_ratio(self) -> float:
        return float(self.n_crc) / math.pi

    @property
    def lambda_l_planck2(self) -> float:
        return lambda_planck2_from_capacity(self.n_crc)

    @property
    def radius_planck(self) -> float:
        return math.sqrt(self.n_patch_bare_ratio)

    @property
    def physical_cell_count(self) -> float:
        return physical_cells_for_entropy_capacity(self.n_crc, self.p_value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "N_CRC": float(self.n_crc),
            "P": float(self.p_value),
            "source": self.source,
            "N_patch_bare_radius_squared_ratio": self.n_patch_bare_ratio,
            "Lambda_lP2": self.lambda_l_planck2,
            "radius_planck": self.radius_planck,
            "N_cells_if_tiled_by_local_P_cells": self.physical_cell_count,
        }


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
    n_crc: float | None = None,
    radius_m: float = DEFAULT_R_DS_M,
    planck_length_m: float = DEFAULT_L_PLANCK_M,
    regulator_patch_counts: tuple[int, ...] = DEFAULT_REGULATOR_PATCH_COUNTS,
) -> dict[str, Any]:
    if n_crc is None:
        input_mode = "observed_de_sitter_radius_readout"
        n_patch = bare_horizon_area_ratio(radius_m, planck_length_m)
        n_scr = entropy_capacity_from_radius(radius_m, planck_length_m)
        r_ds_m = float(radius_m)
    else:
        input_mode = "direct_N_CRC_closure_input"
        n_scr = float(n_crc)
        n_patch = n_scr / math.pi
        r_ds_m = math.sqrt(n_patch) * float(planck_length_m)

    capacity = OPHScreenCapacityConstants(
        n_crc=n_scr,
        p_value=p_value,
        source=input_mode,
    )
    lambda_l_planck2 = lambda_planck2_from_capacity(n_scr)
    physical_cells = physical_cells_for_entropy_capacity(n_scr, p_value)
    return {
        "mode": "oph_screen_capacity_closure_v0",
        "source": "observers_are_all_you_need.tex cosmic record-capacity closure",
        "closure_equations": {
            "cosmic_record_closure": "N_CRC = F(N_CRC)",
            "readback_map": "F(N)=Cap_read(Obs(nf(U_N)))",
            "active_capacity": "N_CRC = log dim Z_boundary^act after predictive quotient",
            "lambda_readout": "Lambda_CRC * l_P^2 = 3*pi / N_CRC",
            "dimensionless_lambda_readout": "Lambda_CRC * ell_star^2 = 3*pi / N_CRC",
            "count_density_selector": "N_star = MAR argmax_N [log|Omega_N^sc| - N]",
            "pressure_certificate": "ell'(N_star)=0 with ell''<0, or Banach contraction for F",
        },
        "observed_branch_normalization": {
            "input_mode": input_mode,
            "R_dS_m": r_ds_m,
            "planck_length_m": float(planck_length_m),
            "N_CRC": n_scr,
            "N_patch_bare_radius_squared_ratio": n_patch,
            "N_scr_entropy_capacity": n_scr,
            "Lambda_lP2": lambda_l_planck2,
            "Lambda_lP2_is_dimensionless": True,
            "dimensionful_Lambda_m2": None,
            "dimensionful_ell_star_squared_m2": None,
            "dimensionful_G_SI": None,
            "N_cells_if_tiled_by_local_P_cells": physical_cells,
            "cell_entropy_capacity": float(p_value) / 4.0,
            "P": float(p_value),
            "constants": capacity.as_dict(),
        },
        "active_capacity_requirements": {
            "capacity_variable": "entropy_capacity_N_not_raw_Hilbert_dimension",
            "active_edge_center_algebra": "Z_boundary^act = Z_boundary^raw / predictive-equivalence",
            "predictive_equivalence": (
                "central record labels are identified when they induce the same future observer-accessible "
                "probability law under same-interface continuations"
            ),
            "observer_sector": "Obs(nf(U_N)) must select stable self-reading observer-supporting terminal normal forms",
            "readback_value": "Cap_read returns the active horizon record capacity reconstructed by observers",
            "finite_regulator_status": "not implemented here; finite patch counts remain numerical regulators",
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
            "N_CRC_closure_value_declared": True,
            "observed_branch_N_scr_readout_available": True,
            "active_edge_center_predictive_quotient_implemented": False,
            "observer_supporting_terminal_sector_implemented": False,
            "capacity_readback_map_from_terminal_records_implemented": False,
            "F_N_readback_map_implemented": False,
            "count_density_normal_form_enumerator_implemented": False,
            "banach_contraction_certificate_implemented": False,
            "pressure_certificate_implemented": False,
            "N_CRC_fixed_point_solved_from_finite_simulator": False,
            "Lambda_from_finite_simulator_record_closure": False,
            "independent_scale_bridge_supplied": False,
            "dimensionful_G_SI_eligible": False,
            "finite_simulator_derived_G_SI": False,
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
            "normalization and dimensionless Lambda*l_P^2 readout from the paper equations, but does not "
            "solve the OPH readback fixed point from simulator data and does not supply an independent "
            "dimensionful scale bridge for G_SI."
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
            f"- input mode: `{observed['input_mode']}`",
            f"- N_CRC: `{observed['N_CRC']:.6e}`",
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
