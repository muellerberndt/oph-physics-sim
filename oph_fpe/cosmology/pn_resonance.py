from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.repair_scale_closure import (
    effective_repair_round_depth,
    repair_scale_closure_report,
)
from oph_fpe.cosmology.scale_bridge import ScaleBridgeInputs, scale_bridge_report
from oph_fpe.cosmology.screen_capacity import (
    DEFAULT_N_CRC,
    capacity_readback_proxy_report,
    screen_capacity_closure_report,
)


ALPHA_U_P_STAR = 0.041124336195630495
ALPHA_U_INTERVAL = (0.041123336195630494, 0.041125336195630496)
N_SOURCE_EW_BRIDGE = "ew_bridge"
N_SOURCE_SCREEN_CAPACITY_DEFAULT = "screen_capacity_default"
N_SOURCE_DIRECT = "direct"


@dataclass(frozen=True)
class PNResonanceInputs:
    """Inputs for the paper-side P/N resonance replay.

    The default uses the compact-paper electroweak bridge relation
    N = pi exp(6*pi/(P alpha_U(P))). This is a paper-declared source replay,
    not a finite-simulator derivation of P, alpha_U, N, or G_SI.
    """

    P_star: float = P_STAR
    alpha_U: float = ALPHA_U_P_STAR
    N_star: float | None = None
    N_source: str = N_SOURCE_EW_BRIDGE
    B_ell_m2_inverse: float | None = None
    Lambda_star_m2_inverse: float | None = None
    repair_rounds: int = 24
    regulator_patch_counts: tuple[int, ...] = (4_096, 65_536, 262_144, 1_048_576)
    exact_log_tolerance: float = 1.0e-10
    observed_display_log_tolerance: float = 0.1
    source: str = "paper_declared_pn_resonance_replay"


def ew_bridge_capacity_from_p_alpha(
    p_star: float = P_STAR,
    alpha_u: float = ALPHA_U_P_STAR,
) -> float:
    """Return N_EW = pi exp(6*pi/(P alpha_U(P)))."""

    p_value = _positive_finite(p_star, "p_star")
    alpha_value = _positive_finite(alpha_u, "alpha_u")
    return float(math.pi * math.exp(_ew_bridge_log_depth(p_value, alpha_value)))


def pn_resonance_report(
    inputs: PNResonanceInputs | None = None,
    *,
    run_dirs: list[Path] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    resonance_inputs = inputs if inputs is not None else PNResonanceInputs(**kwargs)
    p_value = _positive_finite(resonance_inputs.P_star, "P_star")
    alpha_value = _positive_finite(resonance_inputs.alpha_U, "alpha_U")
    n_source = _normalize_n_source(resonance_inputs.N_source)
    n_bridge = ew_bridge_capacity_from_p_alpha(p_value, alpha_value)
    n_used = _selected_capacity(resonance_inputs.N_star, n_source, n_bridge)
    exact_log_tolerance = _nonnegative_finite(resonance_inputs.exact_log_tolerance, "exact_log_tolerance")
    observed_display_log_tolerance = _nonnegative_finite(
        resonance_inputs.observed_display_log_tolerance,
        "observed_display_log_tolerance",
    )

    screen_capacity = screen_capacity_closure_report(
        p_value=p_value,
        n_crc=n_used,
        regulator_patch_counts=tuple(int(value) for value in resonance_inputs.regulator_patch_counts),
    )
    scale_bridge = scale_bridge_report(
        ScaleBridgeInputs(
            P_star=p_value,
            N_star=n_used,
            Lambda_star_m2_inverse=resonance_inputs.Lambda_star_m2_inverse,
            B_ell_m2_inverse=resonance_inputs.B_ell_m2_inverse,
            source=resonance_inputs.source,
        )
    )
    repair_scale = repair_scale_closure_report(
        p_value=p_value,
        n_crc=n_used,
        repair_rounds=int(resonance_inputs.repair_rounds),
        regulator_patch_counts=tuple(int(value) for value in resonance_inputs.regulator_patch_counts),
    )
    capacity_proxy = (
        capacity_readback_proxy_report(
            [Path(path) for path in run_dirs],
            p_value=p_value,
            n_crc=n_used,
        )
        if run_dirs
        else {}
    )

    ew_depth = _ew_bridge_log_depth(p_value, alpha_value)
    selected_depth = math.log(n_used / math.pi)
    log_residual = selected_depth - ew_depth
    paper_relation_exact = abs(log_residual) <= exact_log_tolerance
    observed_display_log_residual = math.log(n_used / DEFAULT_N_CRC)
    observed_display_compatible = abs(observed_display_log_residual) <= observed_display_log_tolerance

    repair_outputs = repair_scale["closure_outputs"]
    repair_n = float(repair_outputs["capacity_implied_by_declared_repair_depth_ansatz"])
    repair_log_residual = math.log(repair_n / n_used)
    repair_capacity_relative_error = abs(repair_n - n_used) / n_used
    repair_contraction_relative_error = float(repair_outputs["relative_error_gprime_vs_N_CRC_closure"])
    g_abs = float(repair_outputs["local_repair_contraction_abs_gprime"])
    effective_round_depth = effective_repair_round_depth(n_used, g_abs)

    screen_gates = screen_capacity["readiness_gates"]
    scale_gates = scale_bridge["readiness_gates"]
    repair_gates = repair_scale["readiness_gates"]
    proxy_gates = capacity_proxy.get("readiness_gates") or {}

    theorem_grade_gates = {
        "paper_bridge_relation_exact": paper_relation_exact,
        "P_alpha_U_source_certificate_implemented": False,
        "N_CRC_fixed_point_solved_from_finite_simulator": bool(
            screen_gates.get("N_CRC_fixed_point_solved_from_finite_simulator", False)
        ),
        "F_N_readback_map_implemented": bool(screen_gates.get("F_N_readback_map_implemented", False)),
        "capacity_readback_proxy_written": bool(capacity_proxy),
        "capacity_readback_proxy_fixed_point_solved": bool(
            proxy_gates.get("N_CRC_fixed_point_solved_from_finite_simulator", False)
        ),
        "repair_depth_derived_from_finite_selector": bool(
            repair_gates.get("twenty_four_round_hypothesis_derived_from_finite_selector", False)
        ),
        "independent_scale_bridge_supplied": bool(
            scale_gates.get("independent_scale_bridge_supplied", False)
        ),
        "dimensionful_G_SI_eligible": bool(scale_gates.get("dimensionful_G_SI_eligible", False)),
        "finite_simulator_derived_G_SI": bool(scale_gates.get("finite_simulator_derived_G_SI", False)),
    }
    theorem_grade = bool(
        theorem_grade_gates["paper_bridge_relation_exact"]
        and theorem_grade_gates["P_alpha_U_source_certificate_implemented"]
        and theorem_grade_gates["N_CRC_fixed_point_solved_from_finite_simulator"]
        and theorem_grade_gates["F_N_readback_map_implemented"]
        and theorem_grade_gates["repair_depth_derived_from_finite_selector"]
    )
    branch_status = _branch_status(
        paper_relation_exact=paper_relation_exact,
        theorem_grade=theorem_grade,
        independent_scale_bridge=theorem_grade_gates["independent_scale_bridge_supplied"],
        n_source=n_source,
    )

    report = {
        "mode": "oph_pn_resonance_v0",
        "source": resonance_inputs.source,
        "PN_RESONANCE_NUMERIC_REPLAY": bool(paper_relation_exact),
        "PN_RESONANCE_RECEIPT": bool(theorem_grade),
        "branch_status": branch_status,
        "inputs": {
            "P_star": p_value,
            "alpha_U": alpha_value,
            "alpha_U_interval": list(ALPHA_U_INTERVAL),
            "N_source": n_source,
            "N_star": n_used,
            "repair_rounds": int(resonance_inputs.repair_rounds),
            "exact_log_tolerance": exact_log_tolerance,
            "observed_display_log_tolerance": observed_display_log_tolerance,
        },
        "paper_bridge_relation": {
            "formula": "N_EW(P)=pi*exp(6*pi/(P*alpha_U(P)))",
            "N_EW_from_P_alpha_U": n_bridge,
            "selected_N_star": n_used,
            "log_N_over_pi": selected_depth,
            "target_log_N_over_pi": ew_depth,
            "log_residual_selected_minus_target": log_residual,
            "capacity_relative_error_vs_bridge": abs(n_used - n_bridge) / n_bridge,
            "paper_bridge_relation_exact": bool(paper_relation_exact),
            "claim_boundary": (
                "This relation is replayed from the paper-declared P and alpha_U source values. "
                "The finite simulator has not emitted the alpha_U proof record or the global F(N) "
                "capacity fixed-point proof."
            ),
        },
        "observed_branch_sidecar": {
            "DEFAULT_N_CRC": DEFAULT_N_CRC,
            "selected_N_over_default_N_CRC": n_used / DEFAULT_N_CRC,
            "log_residual_selected_vs_default_N_CRC": observed_display_log_residual,
            "observed_display_compatible": bool(observed_display_compatible),
            "claim_boundary": (
                "The public 3.31e122 capacity display is a branch-scale approximation unless an exact "
                "capacity/source certificate is supplied."
            ),
        },
        "repair_depth_ansatz_sidecar": {
            "formula": "N_repair=|g'(P)|^(-2m)",
            "N_implied_by_declared_repair_depth_ansatz": repair_n,
            "selected_N_star": n_used,
            "capacity_log_residual_repair_minus_selected": repair_log_residual,
            "capacity_relative_error_repair_vs_selected": repair_capacity_relative_error,
            "contraction_relative_error_vs_selected_N": repair_contraction_relative_error,
            "effective_round_depth_for_selected_N": effective_round_depth,
            "declared_repair_rounds": int(resonance_inputs.repair_rounds),
            "repair_depth_derived_from_finite_selector": bool(
                repair_gates.get("twenty_four_round_hypothesis_derived_from_finite_selector", False)
            ),
            "claim_boundary": (
                "The 24-round repair-depth lane is a diagnostic ansatz. It is not the exact P/N "
                "bridge relation and it is not promoted as a finite derivation of N from P."
            ),
        },
        "component_reports": {
            "screen_capacity_closure_report": screen_capacity,
            "oph_scale_bridge_report": scale_bridge,
            "repair_scale_closure_report": repair_scale,
            "capacity_readback_proxy_report": capacity_proxy,
        },
        "readiness_gates": {
            **theorem_grade_gates,
            "finite_regulator_patch_count_used_as_cosmic_capacity": False,
            "observed_display_compatible": bool(observed_display_compatible),
            "repair_ansatz_capacity_matches_selected_N": bool(abs(repair_log_residual) <= exact_log_tolerance),
            "theorem_grade_pn_resonance": bool(theorem_grade),
            "scale_compressed_pn_resonance_replay_eligible": bool(paper_relation_exact),
            "physical_cmb_prediction": False,
            "strict_neutral_bulk": False,
        },
        "claim_boundary": (
            "Paper-faithful P/N resonance replay. The simulator can run on the selected P/N branch by "
            "using the exact dimensionless bridge relation and by keeping finite patch counts as "
            "regulators. This is not a brute-force N_CRC-cell simulation, not a finite derivation of "
            "alpha_U(P), not a solved F(N) capacity fixed point, and not a finite-simulator derivation "
            "of G_SI."
        ),
    }
    return report


def write_pn_resonance_report(
    out_dir: Path,
    inputs: PNResonanceInputs | None = None,
    *,
    run_dirs: list[Path] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    report = pn_resonance_report(inputs, run_dirs=run_dirs, **kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pn_resonance_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "pn_resonance_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _selected_capacity(n_star: float | None, n_source: str, n_bridge: float) -> float:
    if n_star is not None:
        return _positive_finite(n_star, "N_star")
    if n_source == N_SOURCE_EW_BRIDGE:
        return n_bridge
    if n_source == N_SOURCE_SCREEN_CAPACITY_DEFAULT:
        return DEFAULT_N_CRC
    raise ValueError("N_source='direct' requires N_star")


def _normalize_n_source(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"ew", "ew_bridge", "paper_ew_bridge", "electroweak_bridge"}:
        return N_SOURCE_EW_BRIDGE
    if normalized in {"screen_capacity_default", "observed", "observed_branch", "default"}:
        return N_SOURCE_SCREEN_CAPACITY_DEFAULT
    if normalized == "direct":
        return N_SOURCE_DIRECT
    raise ValueError(f"unknown N_source: {value}")


def _ew_bridge_log_depth(p_star: float, alpha_u: float) -> float:
    return float(6.0 * math.pi / (float(p_star) * float(alpha_u)))


def _branch_status(
    *,
    paper_relation_exact: bool,
    theorem_grade: bool,
    independent_scale_bridge: bool,
    n_source: str,
) -> str:
    if not paper_relation_exact:
        return "off_pn_bridge_relation_diagnostic"
    if theorem_grade and independent_scale_bridge:
        return "theorem_grade_scale_bridged_pn_resonance"
    if theorem_grade:
        return "theorem_grade_dimensionless_pn_resonance"
    if n_source == N_SOURCE_EW_BRIDGE:
        return "paper_declared_pn_resonance_replay_diagnostic"
    return "direct_pn_resonance_replay_diagnostic"


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{float(value):.12g}"
    return str(value)


def _markdown_report(report: dict[str, Any]) -> str:
    bridge = report["paper_bridge_relation"]
    observed = report["observed_branch_sidecar"]
    repair = report["repair_depth_ansatz_sidecar"]
    gates = report["readiness_gates"]
    return "\n".join(
        [
            "# OPH P/N Resonance",
            "",
            str(report["claim_boundary"]),
            "",
            "## Status",
            "",
            f"- branch status: `{report['branch_status']}`",
            f"- numeric P/N replay: `{str(report['PN_RESONANCE_NUMERIC_REPLAY']).lower()}`",
            f"- theorem-grade P/N receipt: `{str(report['PN_RESONANCE_RECEIPT']).lower()}`",
            "",
            "## Paper Bridge",
            "",
            f"- selected N_star: `{_fmt(bridge['selected_N_star'])}`",
            f"- N_EW(P, alpha_U): `{_fmt(bridge['N_EW_from_P_alpha_U'])}`",
            f"- log residual: `{_fmt(bridge['log_residual_selected_minus_target'])}`",
            f"- relation exact: `{str(bridge['paper_bridge_relation_exact']).lower()}`",
            "",
            "## Sidecars",
            "",
            f"- selected/default N_CRC: `{_fmt(observed['selected_N_over_default_N_CRC'])}`",
            f"- observed-display compatible: `{str(observed['observed_display_compatible']).lower()}`",
            f"- repair ansatz N: `{_fmt(repair['N_implied_by_declared_repair_depth_ansatz'])}`",
            f"- repair capacity relative error: `{_fmt(repair['capacity_relative_error_repair_vs_selected'])}`",
            f"- repair contraction relative error: `{_fmt(repair['contraction_relative_error_vs_selected_N'])}`",
            "",
            "## Gates",
            "",
            *[f"- {key}: `{_fmt(value)}`" for key, value in gates.items()],
            "",
        ]
    )
