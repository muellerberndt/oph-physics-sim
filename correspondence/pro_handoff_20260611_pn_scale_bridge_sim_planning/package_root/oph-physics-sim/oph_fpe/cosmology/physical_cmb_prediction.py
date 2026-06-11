from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.cosmology.physical_cmb_contract import (
    PhysicalCMBInputContract,
    validate_physical_cmb_contract,
)


def build_physical_cmb_input_contract(run_dirs: list[Path]) -> tuple[PhysicalCMBInputContract, dict[str, Any]]:
    roots = [Path(path) for path in run_dirs]
    no_data = _first_json(roots, "no_data_use_receipt.json")
    finite_transition = _first_json(roots, "finite_repair_transition_matrix_report.json")
    scalar = _first_json(roots, "scalar_repair_semigroup_report.json")
    finite_cert = _first_json(roots, "finite_certificate_report.json")
    ba_kernel = _first_json(roots, "B_A_kernel_report.json")
    ba_parent = _first_json(roots, "b_a_parent_report.json")
    scale = _first_json(roots, "scale_compressed_repair_report.json")
    screen_capacity = _first_json(roots, "screen_capacity_closure_report.json")
    strict_neutral = _first_json(roots, "strict_neutral_bulk_report.json")
    scalar_quotient = _first_json(roots, "scalar_quotient_report.json")
    compressed_likelihood = _first_json(roots, "oph_compressed_likelihood_report.json")

    eta_source, eta_value = _eta_R_from_reports(finite_transition, scalar, scale, scalar_quotient)
    gamma_source, gamma_grid = _Gamma_rec_from_reports(finite_transition)
    a_source, a_value = _A_zeta_from_reports(finite_cert, scale)
    q_source, q_value = _scalar_value_from_reports(scale, scalar_quotient, "q_IR")
    ell_source, ell_value = _scalar_value_from_reports(scale, scalar_quotient, "ell_IR")
    b_source, b_grid = _B_A_from_reports(ba_kernel, ba_parent)
    rho_source, rho_grid = _rho_A_from_reports(finite_cert, ba_parent)
    freezeout_source, freezeout_surface = _freezeout_from_reports(strict_neutral, scale)
    official_likelihood_ready = bool(compressed_likelihood.get("official_likelihood_ready", False))
    cdm_limit_regression_passed = bool(
        compressed_likelihood.get("cdm_limit_regression_passed", False)
        or _truthy_any(_first_json(roots, "oph_boltzmann_input_report.json"), "cdm_limit_regression_passed")
    )
    contract = PhysicalCMBInputContract(
        no_data_use_receipt=bool(no_data.get("no_data_use_receipt", False) or no_data.get("NO_DATA_USE_RECEIPT", False)),
        P_source="OPH_pixel_branch_predeclared",
        N_source=_N_source_from_screen_capacity(screen_capacity),
        eta_R_source=eta_source,
        eta_R_value=eta_value,
        A_zeta_source=a_source,
        A_zeta_value=a_value,
        q_IR_source=q_source,
        q_IR_value=q_value,
        ell_IR_source=ell_source,
        ell_IR_value=ell_value,
        B_A_source=b_source,
        B_A_k_a=b_grid,
        Gamma_rec_source=gamma_source,
        Gamma_rec_k_a=gamma_grid,
        rho_A_source=rho_source,
        rho_A_a=rho_grid,
        freezeout_source=freezeout_source,
        freezeout_surface=freezeout_surface,
        official_likelihood_ready=official_likelihood_ready,
        cdm_limit_regression_passed=cdm_limit_regression_passed,
    )
    sources = {
        "no_data_use_receipt": no_data,
        "finite_transition_matrix_report": finite_transition,
        "scalar_repair_semigroup_report": scalar,
        "finite_certificate_report": finite_cert,
        "B_A_kernel_report": ba_kernel,
        "b_a_parent_report": ba_parent,
        "scale_compressed_repair_report": scale,
        "screen_capacity_closure_report": screen_capacity,
        "strict_neutral_bulk_report": strict_neutral,
        "scalar_quotient_report": scalar_quotient,
        "oph_compressed_likelihood_report": compressed_likelihood,
    }
    return contract, sources


def write_physical_cmb_input_report(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    contract, sources = build_physical_cmb_input_contract(run_dirs)
    validation = validate_physical_cmb_contract(contract)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    contract_dict = _contract_to_jsonable(contract)
    (out / "physical_cmb_input_contract.json").write_text(json.dumps(contract_dict, indent=2, default=str), encoding="utf-8")
    (out / "physical_cmb_input_validation.json").write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
    _write_array(out / "B_A_k_a.csv", contract.B_A_k_a, ["k_or_row", "a_or_col", "B_A"])
    _write_array(out / "Gamma_rec_k_a.csv", contract.Gamma_rec_k_a, ["k_or_row", "a_or_col", "Gamma_rec"])
    _write_array(out / "rho_A_a.csv", contract.rho_A_a, ["row", "col", "rho_A"])
    report = {
        "mode": "physical_cmb_input_contract_report_v0",
        "run_dirs": [str(path) for path in run_dirs],
        "contract": contract_dict,
        "validation": validation,
        "PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT": validation["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"],
        "physical_cmb_prediction": False,
        "physical_cmb_prediction_eligible": validation["physical_cmb_prediction_eligible"],
        "blockers": validation["blockers"],
        "source_summary": _source_summary(sources),
        "claim_boundary": (
            "Physical CMB input contract assembly. This report may gather measurement-comparable diagnostics, "
            "but it does not run a physical CMB prediction unless every finite-input and likelihood gate passes."
        ),
    }
    (out / "physical_cmb_input_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "physical_cmb_input_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def write_physical_cmb_input_no_data_use_receipt(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    """Write the no-data-use receipt for physical-CMB input construction.

    This receipt is deliberately narrow: it audits whether the OPH input
    functions assembled for a future physical CMB prediction are sourced from
    finite/theorem-side reports rather than fitted from measurement tables. It
    does not certify that those inputs are theorem-grade or likelihood-ready.
    """

    roots = [Path(path) for path in run_dirs]
    finite_transition = _first_json(roots, "finite_repair_transition_matrix_report.json")
    finite_cert = _first_json(roots, "finite_certificate_report.json")
    ba_kernel = _first_json(roots, "B_A_kernel_report.json")
    ba_parent = _first_json(roots, "b_a_parent_report.json")
    scale = _first_json(roots, "scale_compressed_repair_report.json")
    strict_neutral = _first_json(roots, "strict_neutral_bulk_report.json")
    scalar_quotient = _first_json(roots, "scalar_quotient_report.json")
    screen_capacity = _first_json(roots, "screen_capacity_closure_report.json")
    compressed_likelihood = _first_json(roots, "oph_compressed_likelihood_report.json")

    source_status = {
        "finite_repair_transition_matrix_report": {
            "present": bool(finite_transition),
            "used_for": ["eta_R", "Gamma_rec"],
            "measurement_data_used": _measurement_data_used(finite_transition),
        },
        "finite_certificate_report": {
            "present": bool(finite_cert),
            "used_for": ["A_zeta", "rho_A"],
            "measurement_data_used": _measurement_data_used(finite_cert),
        },
        "B_A_kernel_report": {
            "present": bool(ba_kernel),
            "used_for": ["B_A"],
            "measurement_data_used": _measurement_data_used(ba_kernel),
        },
        "b_a_parent_report": {
            "present": bool(ba_parent),
            "used_for": ["diagnostic_B_A", "diagnostic_rho_A"],
            "measurement_data_used": _measurement_data_used(ba_parent)
            or not bool(((ba_parent.get("readiness") or {}).get("checks") or {}).get("no_cmb_data_used", True)),
        },
        "scale_compressed_repair_report": {
            "present": bool(scale),
            "used_for": ["q_IR", "ell_IR", "freezeout_surface"],
            "measurement_data_used": _measurement_data_used(scale),
        },
        "strict_neutral_bulk_report": {
            "present": bool(strict_neutral),
            "used_for": ["neutral_freezeout_if_strict"],
            "measurement_data_used": _measurement_data_used(strict_neutral),
        },
        "scalar_quotient_report": {
            "present": bool(scalar_quotient),
            "used_for": ["diagnostic_eta_R", "diagnostic_q_IR", "diagnostic_ell_IR", "scalar_freezeout_gate"],
            "measurement_data_used": _measurement_data_used(scalar_quotient),
        },
        "screen_capacity_closure_report": {
            "present": bool(screen_capacity),
            "used_for": ["N_source"],
            "measurement_data_used": _measurement_data_used(screen_capacity),
        },
        "oph_compressed_likelihood_report": {
            "present": bool(compressed_likelihood),
            "used_for": ["validation_gate_only"],
            "measurement_data_used": False,
            "note": "Measurement reference reports may be present, but this receipt does not allow them to set OPH input functions.",
        },
    }
    measurement_used = any(bool(row.get("measurement_data_used", False)) for row in source_status.values())
    report = {
        "mode": "physical_cmb_input_no_data_use_receipt_v0",
        "no_data_use_receipt": not measurement_used,
        "NO_DATA_USE_RECEIPT": not measurement_used,
        "measurement_data_used_for_input_functions": measurement_used,
        "measurement_reference_reports_present": bool(compressed_likelihood),
        "source_status": source_status,
        "run_dirs": [str(path) for path in roots],
        "claim_boundary": (
            "Certifies only the data firewall for assembling OPH CMB input functions. "
            "It does not certify finite-theorem grade B_A/rho_A/A_zeta inputs, CDM-limit regression, "
            "official likelihood readiness, or a physical CMB prediction."
        ),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "no_data_use_receipt.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _eta_R_from_reports(
    finite_transition: dict[str, Any],
    scalar: dict[str, Any],
    scale: dict[str, Any],
    scalar_quotient: dict[str, Any],
) -> tuple[str, float | None]:
    if finite_transition.get("eta_R_finite_lattice_derived", False):
        return "finite_repair_transition_clock", _float((finite_transition.get("primary") or {}).get("eta_R_estimate"))
    empirical = (finite_transition.get("clock_modes") or {}).get("empirical") or {}
    if finite_transition.get("eta_R_empirical_finite_lattice_derived", False) or empirical.get(
        "eta_R_finite_lattice_derived", False
    ):
        return "finite_repair_transition_clock", _float(empirical.get("eta_R_value"))
    if scalar.get("eta_R_finite_lattice_derived", False):
        return "finite_repair_transition_clock", _float(scalar.get("eta_R"))
    params = scale.get("cmb_parameter_readouts") or {}
    if scale.get("scale_compressed_operator_receipt", False) and params.get("eta_R") is not None:
        return "scale_compressed_24_round_finite_ladder", _float(params.get("eta_R"))
    edge = scalar_quotient.get("edge_center_readout") or {}
    if scalar_quotient.get("finite_lattice_cmb_scalar_release_ready", False) and edge.get("theta_OPH_P_over_48") is not None:
        return "finite_lattice", _float(edge.get("theta_OPH_P_over_48"))
    if scalar_quotient and edge.get("theta_OPH_P_over_48") is not None:
        return "diagnostic_proxy", _float(edge.get("theta_OPH_P_over_48"))
    return "diagnostic_proxy", _float((finite_transition.get("primary") or {}).get("eta_R_estimate"))


def _Gamma_rec_from_reports(finite_transition: dict[str, Any]) -> tuple[str, np.ndarray | None]:
    primary = finite_transition.get("primary") or {}
    gamma = _float(primary.get("gamma_continuous"))
    if gamma is None:
        return "unknown", None
    source = "finite_repair_transition_clock" if finite_transition.get("finite_transition_matrix_ready", False) else "diagnostic_proxy"
    return source, np.asarray([[gamma]], dtype=float)


def _A_zeta_from_reports(finite_cert: dict[str, Any], scale: dict[str, Any]) -> tuple[str, float | None]:
    derived = finite_cert.get("derived_outputs") or {}
    value = _float(derived.get("A_zeta", finite_cert.get("A_zeta")))
    if finite_cert.get("theorem_grade_finite_inputs", False) and value is not None:
        return "finite_lattice", value
    params = scale.get("cmb_parameter_readouts") or {}
    if params.get("A_zeta") is not None and scale.get("scale_compressed_operator_receipt", False):
        return "scale_compressed_24_round_finite_ladder", _float(params.get("A_zeta"))
    return "diagnostic_proxy", value


def _scalar_value_from_reports(
    scale: dict[str, Any],
    scalar_quotient: dict[str, Any],
    key: str,
) -> tuple[str, float | None]:
    params = scale.get("cmb_parameter_readouts") or {}
    value = _float(params.get(key))
    if scale.get("scale_compressed_operator_receipt", False) and value is not None:
        return "scale_compressed_24_round_finite_ladder", value
    levels = scalar_quotient.get("active_angular_levels") or {}
    if key == "ell_IR":
        value = _float(levels.get("target_ell_IR"))
    elif key == "q_IR":
        value = 0.25 if scalar_quotient else None
    else:
        value = None
    if scalar_quotient.get("finite_lattice_cmb_scalar_release_ready", False) and value is not None:
        return "finite_lattice", value
    return "diagnostic_proxy", value


def _B_A_from_reports(ba_kernel: dict[str, Any], ba_parent: dict[str, Any]) -> tuple[str, np.ndarray | None]:
    if ba_kernel.get("B_A_KERNEL_RECEIPT", False):
        return "parent_collar_finite_difference", _array(ba_kernel.get("B_A_k_a"))
    rows = ba_parent.get("rows") or ba_parent.get("observer_view_rows") or []
    values = []
    for row in rows:
        k = _float(row.get("k_h_mpc", row.get("k_proxy_inverse_theta")))
        a = _float(row.get("a"))
        b = _float(row.get("B_A_mean"))
        if k is not None and a is not None and b is not None:
            values.append([k, a, b])
    return ("diagnostic_proxy", np.asarray(values, dtype=float) if values else None)


def _rho_A_from_reports(finite_cert: dict[str, Any], ba_parent: dict[str, Any]) -> tuple[str, np.ndarray | None]:
    derived = finite_cert.get("derived_outputs") or {}
    finite_rho = _array(derived.get("rho_A_a", finite_cert.get("rho_A_a")))
    if finite_cert.get("theorem_grade_finite_inputs", False) and finite_rho is not None:
        return "finite_lattice", finite_rho
    rows = ba_parent.get("rows") or ba_parent.get("observer_view_rows") or []
    values = []
    for row in rows:
        a = _float(row.get("a"))
        base = _float(
            row.get(
                "rho_A",
                row.get("rho_A_base", row.get("base_epsilon_cmi", row.get("B_A_mean"))),
            )
        )
        if a is not None and base is not None:
            values.append([a, abs(base)])
    return ("diagnostic_proxy", np.asarray(values, dtype=float) if values else None)


def _freezeout_from_reports(strict_neutral: dict[str, Any], scale: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    if strict_neutral.get("strict_neutral_bulk", False):
        return "neutral_bulk_freezeout", {"source": "strict_neutral_bulk_report"}
    if scale.get("scale_compressed_operator_receipt", False):
        return "scale_compressed_24_round_finite_ladder", {
            "repair_rounds": scale.get("logical_repair_rounds"),
            "source": "scale_compressed_repair_report",
        }
    return "unknown", None


def _source_summary(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "present": bool(report),
            "mode": report.get("mode"),
            "physical_cmb_prediction": report.get("physical_cmb_prediction"),
        }
        for name, report in sources.items()
    }


def _N_source_from_screen_capacity(screen_capacity: dict[str, Any]) -> str:
    if not screen_capacity:
        return "unknown"
    if screen_capacity.get("screen_capacity_closure_receipt", False) or screen_capacity.get(
        "SCREEN_CAPACITY_CLOSURE_RECEIPT", False
    ):
        return "OPH_screen_capacity_branch_predeclared"
    observed = screen_capacity.get("observed_branch_normalization") or {}
    gates = screen_capacity.get("readiness_gates") or {}
    if observed.get("N_CRC") is not None and gates.get("observed_branch_N_scr_readout_available", False):
        return "OPH_screen_capacity_observed_branch_readout"
    return "unknown"


def _contract_to_jsonable(contract: PhysicalCMBInputContract) -> dict[str, Any]:
    data = asdict(contract)
    for key in ("B_A_k_a", "Gamma_rec_k_a", "rho_A_a"):
        value = data.get(key)
        if value is not None:
            data[key] = np.asarray(value, dtype=float).tolist()
    return data


def _write_array(path: Path, array: np.ndarray | None, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        if array is None:
            return
        arr = np.asarray(array, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        for i in range(arr.shape[0]):
            if arr.shape[1] >= 3:
                writer.writerow([arr[i, 0], arr[i, 1], arr[i, 2]])
            elif arr.shape[1] == 2:
                writer.writerow([i, arr[i, 0], arr[i, 1]])
            else:
                writer.writerow([i, 0, arr[i, 0]])


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Physical CMB Input Contract",
        "",
        f"- contract receipt: `{str(report['PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT']).lower()}`",
        f"- physical CMB prediction eligible: `{str(report['physical_cmb_prediction_eligible']).lower()}`",
        f"- physical CMB prediction: `{str(report['physical_cmb_prediction']).lower()}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = report.get("blockers") or []
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", report.get("claim_boundary", ""), ""])
    return "\n".join(lines)


def _first_json(roots: list[Path], name: str) -> dict[str, Any]:
    for root in roots:
        root = Path(root)
        candidates = [root / name]
        if root.exists() and root.is_dir():
            candidates.extend(sorted(root.glob(f"**/{name}")))
        for path in candidates:
            if path.exists() and path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    return data
    return {}


def _truthy_any(data: dict[str, Any], *keys: str) -> bool:
    return any(bool(data.get(key, False)) for key in keys)


def _measurement_data_used(report: dict[str, Any]) -> bool:
    if not report:
        return False
    explicit_false = (
        report.get("no_cmb_data_used") is True
        or (((report.get("readiness") or {}).get("checks") or {}).get("no_cmb_data_used") is True)
    )
    if explicit_false:
        return False
    data_use_keys = (
        "measurement_data_used",
        "cmb_data_used",
        "cmb_data_used_for_input",
        "planck_data_used_for_input",
        "fit_to_measurement",
        "fit_to_planck",
        "uses_measurements_to_set_inputs",
    )
    return any(bool(report.get(key, False)) for key in data_use_keys)


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return array if array.size and np.all(np.isfinite(array)) else None
