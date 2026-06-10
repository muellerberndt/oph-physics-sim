from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from oph_fpe.constants.oph_pixel import OPHPixelConstants, P_STAR
from oph_fpe.cosmology.screen_capacity import DEFAULT_N_CRC, lambda_planck2_from_capacity


DEFAULT_REPAIR_ROUNDS = 24
DEFAULT_REGULATOR_PATCH_COUNTS = (4_096, 65_536, 262_144, 1_048_576, 1_000_000_000)


def local_repair_contraction_from_p(p_value: float = P_STAR) -> float:
    """Declared local repair-map contraction from the pixel closure branch.

    This is the Maarten/repair-round hypothesis lane:
    |g'(P)| = alpha(P) / phi^2 = (P - phi) / (sqrt(pi) phi^2).
    It is a scale-closure diagnostic, not a finite proof of populated H3 bulk.
    """

    pixel = OPHPixelConstants(P=float(p_value))
    return float(pixel.alpha_from_P / (pixel.phi * pixel.phi))


def contraction_from_capacity(n_crc: float = DEFAULT_N_CRC, rounds: int = DEFAULT_REPAIR_ROUNDS) -> float:
    exponent = -1.0 / (2.0 * int(rounds))
    return float(float(n_crc) ** exponent)


def capacity_from_contraction(contraction: float, rounds: int = DEFAULT_REPAIR_ROUNDS) -> float:
    g_abs = abs(float(contraction))
    if g_abs <= 0.0:
        raise ValueError("contraction must be positive")
    return float(g_abs ** (-2.0 * int(rounds)))


def effective_repair_round_depth(n_capacity: float, contraction: float) -> float:
    """Effective multiplicative scale rounds represented by a finite capacity."""

    n = float(n_capacity)
    g_abs = abs(float(contraction))
    if n <= 0.0 or not math.isfinite(n):
        raise ValueError("n_capacity must be positive")
    if g_abs <= 0.0 or g_abs >= 1.0 or not math.isfinite(g_abs):
        raise ValueError("contraction must be finite and in (0, 1)")
    return float(math.log(n) / (-2.0 * math.log(g_abs)))


def repair_scale_closure_report(
    *,
    p_value: float = P_STAR,
    n_crc: float = DEFAULT_N_CRC,
    repair_rounds: int = DEFAULT_REPAIR_ROUNDS,
    regulator_patch_counts: tuple[int, ...] = DEFAULT_REGULATOR_PATCH_COUNTS,
) -> dict[str, Any]:
    if int(repair_rounds) <= 0:
        raise ValueError("repair_rounds must be positive")
    pixel = OPHPixelConstants(P=float(p_value))
    exponent = 2 * int(repair_rounds)
    g_p = local_repair_contraction_from_p(p_value)
    g_capacity = contraction_from_capacity(n_crc, rounds=repair_rounds)
    n_pred = capacity_from_contraction(g_p, rounds=repair_rounds)
    q_round = 1.0 / g_p
    length_ratio = q_round ** int(repair_rounds)
    eta_p48 = float(pixel.P / float(exponent))
    n_s_p48 = float(1.0 - eta_p48)
    rel_error = abs(g_p - g_capacity) / max(abs(g_capacity), 1.0e-300)
    rows = [
        _finite_round_row(
            int(count),
            contraction=g_p,
            target_rounds=int(repair_rounds),
            p_value=float(pixel.P),
        )
        for count in regulator_patch_counts
    ]
    report = {
        "mode": "oph_repair_scale_closure_hypothesis_v0",
        "source": "Maarten repair-round scale-closure note; docs/repair_rounds.txt",
        "hypothesis": {
            "local_repair_map": "delta_{n+1} = g'(P) delta_n",
            "contraction_formula": "|g'(P)| = alpha(P) / phi^2",
            "capacity_formula": "N_CRC = |g'(P)|^(-2 m)",
            "declared_repair_rounds_m": int(repair_rounds),
            "capacity_exponent_2m": int(exponent),
            "length_round_factor_q": "q = 1 / |g'(P)|",
            "tilt_candidate": "eta_R = P / (2 m); n_s = 1 - P/(2 m)",
        },
        "local_pixel_inputs": {
            "P": float(pixel.P),
            "phi": float(pixel.phi),
            "sqrt_pi": float(pixel.sqrt_pi),
            "alpha_from_P": float(pixel.alpha_from_P),
            "alpha_over_phi_squared": float(g_p),
        },
        "global_capacity_inputs": {
            "N_CRC": float(n_crc),
            "contraction_from_N_CRC": float(g_capacity),
            "Lambda_lP2_from_N_CRC": float(lambda_planck2_from_capacity(n_crc)),
        },
        "closure_outputs": {
            "local_repair_contraction_abs_gprime": float(g_p),
            "scale_factor_per_round_q": float(q_round),
            "length_ratio_after_m_rounds": float(length_ratio),
            "capacity_predicted_from_local_P": float(n_pred),
            "Lambda_lP2_predicted_from_local_P": float(lambda_planck2_from_capacity(n_pred)),
            "relative_error_gprime_vs_N_CRC_closure": float(rel_error),
            "eta_R_p_over_2m": eta_p48,
            "n_s_p_over_2m": n_s_p48,
        },
        "finite_regulator_round_depth": rows,
        "readiness_gates": {
            "local_P_closure_available": True,
            "global_N_CRC_declared": True,
            "scale_closure_numeric_match_within_1_percent": bool(rel_error <= 0.01),
            "twenty_four_round_hypothesis_declared": True,
            "twenty_four_round_hypothesis_derived_from_finite_selector": False,
            "finite_lattice_24_round_operator_implemented": False,
            "finite_lattice_derived_eta_R": False,
            "populated_3d_bulk_established": False,
            "physical_cmb_prediction": False,
        },
        "physical_cmb_prediction": False,
        "bulk_3d_established": False,
        "claim_boundary": (
            "Scale-closure / repair-depth hypothesis report. It connects the local OPH pixel constant P "
            "to a global capacity scale through a declared 24-round repair-depth ansatz. It explains why "
            "ordinary 64k/256k/1M patch regulators are only about one effective scale round deep, and it "
            "exports the paper-side n_s = 1 - P/48 candidate. It is not a finite proof of 24 repair rounds, "
            "not a strict 3D bulk receipt, and not a physical CMB prediction."
        ),
    }
    return report


def write_repair_scale_closure_report(out_dir: Path, **kwargs: Any) -> dict[str, Any]:
    report = repair_scale_closure_report(**kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "repair_scale_closure_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "repair_scale_closure_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_round_rows(out / "repair_scale_round_depth.csv", report["finite_regulator_round_depth"])
    return report


def _finite_round_row(
    patch_count: int,
    *,
    contraction: float,
    target_rounds: int,
    p_value: float,
) -> dict[str, Any]:
    depth_patch = effective_repair_round_depth(float(patch_count), contraction)
    entropy_capacity = float(patch_count) * float(p_value) / 4.0
    depth_entropy = effective_repair_round_depth(entropy_capacity, contraction)
    return {
        "patch_count": int(patch_count),
        "patch_count_as_capacity_depth_rounds": float(depth_patch),
        "P_entropy_capacity": float(entropy_capacity),
        "P_entropy_capacity_depth_rounds": float(depth_entropy),
        "fraction_of_declared_24_round_depth_patch_count": float(depth_patch / float(target_rounds)),
        "fraction_of_declared_24_round_depth_entropy_capacity": float(depth_entropy / float(target_rounds)),
        "claim_boundary": "finite regulator depth estimate only; not a literal cosmic-capacity simulation",
    }


def _write_round_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "patch_count",
        "patch_count_as_capacity_depth_rounds",
        "P_entropy_capacity",
        "P_entropy_capacity_depth_rounds",
        "fraction_of_declared_24_round_depth_patch_count",
        "fraction_of_declared_24_round_depth_entropy_capacity",
        "claim_boundary",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(report: dict[str, Any]) -> str:
    outputs = report["closure_outputs"]
    gates = report["readiness_gates"]
    return "\n".join(
        [
            "# OPH Repair-Scale Closure",
            "",
            str(report["claim_boundary"]),
            "",
            "## Closure Outputs",
            "",
            f"- |g'(P)|: `{outputs['local_repair_contraction_abs_gprime']:.12g}`",
            f"- q per repair round: `{outputs['scale_factor_per_round_q']:.12g}`",
            f"- N_CRC predicted from P: `{outputs['capacity_predicted_from_local_P']:.12e}`",
            f"- relative |g'| error vs declared N_CRC: `{outputs['relative_error_gprime_vs_N_CRC_closure']:.6g}`",
            f"- eta_R = P/48: `{outputs['eta_R_p_over_2m']:.12g}`",
            f"- n_s = 1 - P/48: `{outputs['n_s_p_over_2m']:.12g}`",
            "",
            "## Gates",
            "",
            *[f"- {key}: `{str(value).lower()}`" for key, value in gates.items()],
            "",
        ]
    )
