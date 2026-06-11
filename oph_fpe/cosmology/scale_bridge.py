from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.screen_capacity import DEFAULT_N_CRC


C_SI = 299_792_458.0
HBAR_SI = 6.626_070_15e-34 / (2.0 * math.pi)


@dataclass(frozen=True)
class ScaleBridgeInputs:
    """Inputs for the OPH P/N/G scale bridge audit.

    P and N determine dimensionless products only. A dimensionful report needs
    either an independent B_ell value or an independent Lambda_star value plus
    N_star, since B_ell = Lambda_star N_star.
    """

    P_star: float = P_STAR
    N_star: float | None = DEFAULT_N_CRC
    Lambda_star_m2_inverse: float | None = None
    B_ell_m2_inverse: float | None = None
    source: str = "simulator_scale_bridge_report"


def dimensionless_pn_invariants(
    *,
    p_star: float = P_STAR,
    n_star: float | None = DEFAULT_N_CRC,
) -> dict[str, Any]:
    """Return the P/N invariants that do not set an SI scale by themselves."""

    p_value = _positive_finite(p_star, "p_star")
    n_value = _optional_positive_finite(n_star, "n_star")
    invariants: dict[str, Any] = {
        "P_star": p_value,
        "N_star": n_value,
        "a_cell_over_ell_star_squared": p_value,
        "B_ell_ell_star_squared": 3.0 * math.pi,
        "B_ell_a_cell": 3.0 * math.pi * p_value,
        "P_N_determine_dimensionless_invariants_only": True,
        "P_N_determine_dimensionful_scale": False,
        "P_cancels_from_local_gravity_readout": True,
        "G_geom_in_OPH_units": "ell_star_squared",
        "relation": "P=a_cell/ell_star^2, N=3*pi/(Lambda_star*ell_star^2), B_ell=Lambda_star*N",
    }
    if n_value is not None:
        invariants.update(
            {
                "Lambda_star_ell_star_squared": 3.0 * math.pi / n_value,
                "Lambda_star_a_cell": 3.0 * math.pi * p_value / n_value,
            }
        )
    else:
        invariants.update(
            {
                "Lambda_star_ell_star_squared": None,
                "Lambda_star_a_cell": None,
            }
        )
    return invariants


def scale_bridge_report(inputs: ScaleBridgeInputs | None = None, **kwargs: Any) -> dict[str, Any]:
    bridge_inputs = inputs if inputs is not None else ScaleBridgeInputs(**kwargs)
    p_value = _positive_finite(bridge_inputs.P_star, "P_star")
    n_value = _optional_positive_finite(bridge_inputs.N_star, "N_star")
    lambda_value = _optional_positive_finite(bridge_inputs.Lambda_star_m2_inverse, "Lambda_star_m2_inverse")
    b_value = _optional_positive_finite(bridge_inputs.B_ell_m2_inverse, "B_ell_m2_inverse")

    if b_value is not None and lambda_value is not None:
        raise ValueError("provide either B_ell_m2_inverse or Lambda_star_m2_inverse, not both")
    if lambda_value is not None and n_value is None:
        raise ValueError("Lambda_star_m2_inverse scale bridge requires N_star")

    mode = "none"
    derived_b = None
    if b_value is not None:
        mode = "direct_B_ell"
        derived_b = b_value
    elif lambda_value is not None and n_value is not None:
        mode = "Lambda_star_times_N_star"
        derived_b = lambda_value * n_value

    supplied = derived_b is not None
    ell_star_squared = (3.0 * math.pi / derived_b) if derived_b is not None else None
    ell_star = math.sqrt(ell_star_squared) if ell_star_squared is not None else None
    g_si = (ell_star_squared * C_SI**3 / HBAR_SI) if ell_star_squared is not None else None

    invariants = dimensionless_pn_invariants(p_star=p_value, n_star=n_value)
    bridge = {
        "source_input_mode": mode,
        "source": bridge_inputs.source,
        "P_star": p_value,
        "N_star": n_value,
        "Lambda_star_m2_inverse": lambda_value,
        "B_ell_m2_inverse": derived_b,
        "B_ell_m^-2": derived_b,
        "ell_star_squared_m2": ell_star_squared,
        "ell_star_m": ell_star,
        "G_geom": ell_star_squared,
        "G_SI": g_si,
        "G_SI_units": "m^3 kg^-1 s^-2" if supplied else None,
        "independent_scale_bridge_supplied": supplied,
        "OPH_independent_scale_bridge_supplied": supplied,
        "dimensionful_G_SI_eligible": supplied,
        "finite_simulator_derived_G_SI": False,
        "P_N_alone_dimensionful_scale_eligible": False,
        "P_cancels_from_G_geom": True,
    }
    gates = {
        "P_branch_available": True,
        "N_capacity_branch_declared": bool(n_value is not None),
        "P_N_dimensionless_invariants_available": bool(n_value is not None),
        "independent_scale_bridge_supplied": supplied,
        "OPH_independent_scale_bridge_supplied": supplied,
        "B_ell_available": supplied,
        "dimensionful_ell_star_available": supplied,
        "dimensionful_G_SI_eligible": supplied,
        "finite_simulator_derived_G_SI": False,
        "physical_cmb_prediction": False,
        "strict_neutral_bulk": False,
    }
    return {
        "mode": "oph_pn_scale_bridge_v1",
        "source": bridge_inputs.source,
        "constants": {
            "c_SI": C_SI,
            "hbar_SI": HBAR_SI,
            "G_SI_relation": "G_SI = ell_star^2 c^3 / hbar once ell_star^2 is independently bridged to m^2",
        },
        "dimensionless_invariants": invariants,
        "scale_bridge": bridge,
        "readiness_gates": gates,
        "physical_cmb_prediction": False,
        "strict_neutral_bulk": False,
        "claim_boundary": (
            "P and N fix dimensionless OPH invariants only. They do not determine a dimensionful "
            "Planck area, B_ell, or G_SI by themselves. A dimensionful G_SI readout is eligible only "
            "after an independent scale bridge supplies B_ell directly, or supplies Lambda_star and N_star "
            "so B_ell=Lambda_star*N_star. This report is a bookkeeping gate, not a finite-simulator "
            "derivation of G."
        ),
    }


def write_scale_bridge_report(out_dir: Path, inputs: ScaleBridgeInputs | None = None, **kwargs: Any) -> dict[str, Any]:
    report = scale_bridge_report(inputs, **kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "oph_scale_bridge_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "oph_scale_bridge_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _positive_finite(value: float, name: str) -> float:
    parsed = float(value)
    if parsed <= 0.0 or not math.isfinite(parsed):
        raise ValueError(f"{name} must be positive and finite")
    return parsed


def _optional_positive_finite(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return _positive_finite(value, name)


def _fmt(value: Any) -> str:
    if value is None:
        return "not supplied"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{float(value):.12e}"
    return str(value)


def _markdown_report(report: dict[str, Any]) -> str:
    invariants = report["dimensionless_invariants"]
    bridge = report["scale_bridge"]
    gates = report["readiness_gates"]
    return "\n".join(
        [
            "# OPH P/N Scale Bridge",
            "",
            str(report["claim_boundary"]),
            "",
            "## Dimensionless P/N Invariants",
            "",
            f"- P_star: `{_fmt(invariants['P_star'])}`",
            f"- N_star: `{_fmt(invariants['N_star'])}`",
            f"- Lambda_star ell_star^2: `{_fmt(invariants['Lambda_star_ell_star_squared'])}`",
            f"- Lambda_star a_cell: `{_fmt(invariants['Lambda_star_a_cell'])}`",
            f"- B_ell ell_star^2: `{_fmt(invariants['B_ell_ell_star_squared'])}`",
            f"- B_ell a_cell: `{_fmt(invariants['B_ell_a_cell'])}`",
            "",
            "## Independent Scale Bridge",
            "",
            f"- input mode: `{bridge['source_input_mode']}`",
            f"- B_ell [m^-2]: `{_fmt(bridge['B_ell_m2_inverse'])}`",
            f"- ell_star^2 [m^2]: `{_fmt(bridge['ell_star_squared_m2'])}`",
            f"- G_SI: `{_fmt(bridge['G_SI'])}`",
            f"- finite-simulator derived G_SI: `{str(bridge['finite_simulator_derived_G_SI']).lower()}`",
            "",
            "## Gates",
            "",
            *[f"- {key}: `{str(value).lower()}`" for key, value in gates.items()],
            "",
        ]
    )
