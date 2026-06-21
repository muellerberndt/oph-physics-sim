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
NU_CS_HZ = 9_192_631_770.0
EPSILON_CS_SELECTED = 3.113_930_513_416_012_8e-33
GAMMA_STAR_SELECTED = EPSILON_CS_SELECTED / (2.0 * math.pi)
R_GAMMA_REQUIRED_COMPONENTS = (
    "R_U",
    "R_alpha",
    "R_e_abs",
    "R_QCD_nuc_133Cs",
    "R_atom_133Cs",
)
R_GAMMA_FORBIDDEN_DEPENDENCIES = (
    "measured_G",
    "G_SI_measured",
    "Planck_area",
    "l_planck_measured",
    "measured_Lambda",
    "Lambda_measured",
    "gravity_calibrated_scale",
    "electroweak_measured_calibration",
    "measured_electroweak_calibration",
)


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


@dataclass(frozen=True)
class NoGClockBridgeInputs:
    """Inputs for the OPH no-G R_gamma clock bridge audit.

    The epsilon_Cs/gamma_star checksum is useful bookkeeping, but it is not a
    source-predictive G receipt unless the public dependency graph and the
    paper-side source/certificate gates are supplied.
    """

    epsilon_cs: float = EPSILON_CS_SELECTED
    nu_cs_hz: float = NU_CS_HZ
    dependency_graph: dict[str, list[str]] | None = None
    source_components: tuple[str, ...] = R_GAMMA_REQUIRED_COMPONENTS
    forbidden_dependencies: tuple[str, ...] = R_GAMMA_FORBIDDEN_DEPENDENCIES
    source: str = "compact_proof_R_gamma_display"
    source_readback_map_emitted: bool = False
    contraction_certificate: bool = False
    residual_certificate: bool = False
    public_dependency_graph: bool = False


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


def no_g_clock_bridge_report(inputs: NoGClockBridgeInputs | None = None, **kwargs: Any) -> dict[str, Any]:
    bridge_inputs = inputs if inputs is not None else NoGClockBridgeInputs(**kwargs)
    epsilon = _positive_finite(bridge_inputs.epsilon_cs, "epsilon_cs")
    nu_cs = _positive_finite(bridge_inputs.nu_cs_hz, "nu_cs_hz")
    gamma_star = epsilon / (2.0 * math.pi)
    ell_star_m = gamma_star * C_SI / nu_cs
    ell_star_squared = ell_star_m**2
    g_si = ell_star_squared * C_SI**3 / HBAR_SI
    b_ell = 3.0 * math.pi / ell_star_squared

    graph = _normalized_dependency_graph(bridge_inputs.dependency_graph)
    required_components = tuple(str(value) for value in R_GAMMA_REQUIRED_COMPONENTS)
    supplied_components = tuple(str(value) for value in bridge_inputs.source_components)
    missing_components = sorted(set(required_components) - set(supplied_components))
    forbidden_paths = _forbidden_dependency_paths(
        graph,
        tuple(str(value) for value in bridge_inputs.forbidden_dependencies),
    )
    no_forbidden_paths = bool(bridge_inputs.public_dependency_graph and graph and not forbidden_paths)
    gates = {
        "R_gamma_components_complete": not missing_components,
        "public_dependency_graph": bool(bridge_inputs.public_dependency_graph),
        "source_readback_map_emitted": bool(bridge_inputs.source_readback_map_emitted),
        "contraction_certificate": bool(bridge_inputs.contraction_certificate),
        "residual_certificate": bool(bridge_inputs.residual_certificate),
        "no_forbidden_dependency_paths": no_forbidden_paths,
        "forbidden_dependency_path_count": len(forbidden_paths),
    }
    blockers: list[str] = []
    if missing_components:
        blockers.append("R_gamma_source_components_incomplete")
    if not bridge_inputs.public_dependency_graph:
        blockers.append("public_dependency_graph_missing")
    if not graph:
        blockers.append("dependency_graph_empty_or_not_supplied")
    if not bridge_inputs.source_readback_map_emitted:
        blockers.append("source_readback_map_missing")
    if not bridge_inputs.contraction_certificate:
        blockers.append("contraction_certificate_missing")
    if not bridge_inputs.residual_certificate:
        blockers.append("residual_certificate_missing")
    if forbidden_paths:
        blockers.append("forbidden_dependency_path_present")
    elif bridge_inputs.public_dependency_graph and graph:
        pass
    else:
        blockers.append("forbidden_dependency_absence_not_certified")

    receipt = bool(
        gates["R_gamma_components_complete"]
        and gates["public_dependency_graph"]
        and gates["source_readback_map_emitted"]
        and gates["contraction_certificate"]
        and gates["residual_certificate"]
        and gates["no_forbidden_dependency_paths"]
    )
    clock_bridge = {
        "source": bridge_inputs.source,
        "epsilon_Cs": epsilon,
        "gamma_star": gamma_star,
        "nu_Cs_Hz": nu_cs,
        "ell_star_m": ell_star_m,
        "ell_star_squared_m2": ell_star_squared,
        "B_ell_m2_inverse": b_ell,
        "B_ell_m^-2": b_ell,
        "G_geom": ell_star_squared,
        "G_SI": g_si,
        "G_SI_units": "m^3 kg^-1 s^-2",
        "R_gamma_relation": "gamma_star = epsilon_Cs/(2*pi), ell_star = gamma_star*c/nu_Cs",
        "forbidden_dependencies": tuple(str(value) for value in bridge_inputs.forbidden_dependencies),
        "dependency_graph_node_count": len(graph),
        "source_components": supplied_components,
        "required_source_components": required_components,
        "missing_source_components": missing_components,
        "forbidden_dependency_paths": forbidden_paths,
        "calibration_checksum_available": True,
        "source_predictive_G_SI": receipt,
        "independent_scale_bridge_supplied": receipt,
        "OPH_independent_scale_bridge_supplied": receipt,
        "dimensionful_G_SI_eligible": receipt,
        "finite_simulator_derived_G_SI": False,
    }
    return {
        "mode": "oph_no_g_clock_bridge_v0",
        "source": bridge_inputs.source,
        "constants": {
            "c_SI": C_SI,
            "hbar_SI": HBAR_SI,
            "nu_Cs_Hz": NU_CS_HZ,
            "G_SI_relation": "G_SI = ell_star^2 c^3 / hbar after no-G R_gamma clock bridge certification",
        },
        "clock_bridge": clock_bridge,
        "readiness_gates": gates,
        "blockers": _unique_strings(blockers),
        "NO_G_CLOCK_BRIDGE_RECEIPT": receipt,
        "source_predictive_G_SI": receipt,
        "independent_scale_bridge_supplied": receipt,
        "OPH_independent_scale_bridge_supplied": receipt,
        "dimensionful_G_SI_eligible": receipt,
        "finite_simulator_derived_G_SI": False,
        "physical_cmb_prediction": False,
        "strict_neutral_bulk": False,
        "claim_boundary": (
            "No-G R_gamma clock bridge audit. The epsilon_Cs checksum gives ell_star^2 and a G_SI "
            "checksum, but G_SI is source-predictive only when the public dependency graph, source "
            "readback map, contraction certificate, residual certificate, and no-forbidden-dependency "
            "audit all pass. Forbidden dependencies include measured G, Planck-area input, measured "
            "Lambda, gravity-calibrated scales, and measured electroweak calibration."
        ),
    }


def write_scale_bridge_report(out_dir: Path, inputs: ScaleBridgeInputs | None = None, **kwargs: Any) -> dict[str, Any]:
    report = scale_bridge_report(inputs, **kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "oph_scale_bridge_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "oph_scale_bridge_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def write_no_g_clock_bridge_report(
    out_dir: Path,
    inputs: NoGClockBridgeInputs | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    report = no_g_clock_bridge_report(inputs, **kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "no_g_clock_bridge_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "no_g_clock_bridge_report.md").write_text(
        _markdown_clock_bridge_report(report),
        encoding="utf-8",
    )
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


def _normalized_dependency_graph(graph: dict[str, list[str]] | None) -> dict[str, list[str]]:
    if not isinstance(graph, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for node, deps in graph.items():
        if not isinstance(deps, list):
            continue
        normalized[str(node)] = [str(dep) for dep in deps]
    return normalized


def _forbidden_dependency_paths(
    graph: dict[str, list[str]],
    forbidden_dependencies: tuple[str, ...],
) -> list[list[str]]:
    if not graph:
        return []
    forbidden_lc = {value.lower() for value in forbidden_dependencies}
    start_nodes = [node for node in ("gamma_star", "epsilon_Cs") if node in graph]
    if not start_nodes and "R_gamma" in graph:
        start_nodes = ["R_gamma"]
    if not start_nodes:
        start_nodes = sorted(graph)
    paths: list[list[str]] = []

    def visit(node: str, path: list[str], active: set[str]) -> None:
        if node in active:
            return
        deps = graph.get(node, [])
        for dep in deps:
            dep_text = str(dep)
            next_path = [*path, dep_text]
            if dep_text.lower() in forbidden_lc:
                paths.append(next_path)
                continue
            if dep_text in graph:
                visit(dep_text, next_path, {*active, node})

    for start in start_nodes:
        visit(start, [start], set())
    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for path in paths:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def _unique_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


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


def _markdown_clock_bridge_report(report: dict[str, Any]) -> str:
    bridge = report["clock_bridge"]
    gates = report["readiness_gates"]
    blockers = report.get("blockers") or []
    lines = [
        "# OPH No-G Clock Bridge",
        "",
        str(report["claim_boundary"]),
        "",
        "## Checksum",
        "",
        f"- epsilon_Cs: `{_fmt(bridge['epsilon_Cs'])}`",
        f"- gamma_star: `{_fmt(bridge['gamma_star'])}`",
        f"- ell_star^2 [m^2]: `{_fmt(bridge['ell_star_squared_m2'])}`",
        f"- B_ell [m^-2]: `{_fmt(bridge['B_ell_m2_inverse'])}`",
        f"- G_SI checksum: `{_fmt(bridge['G_SI'])}`",
        f"- source-predictive G_SI: `{str(bridge['source_predictive_G_SI']).lower()}`",
        "",
        "## Gates",
        "",
        *[f"- {key}: `{str(value).lower()}`" for key, value in gates.items()],
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- none")
    lines.append("")
    return "\n".join(lines)
