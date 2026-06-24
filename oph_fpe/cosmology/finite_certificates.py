from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.constants.oph_pixel import P_STAR

LN2 = math.log(2.0)
HBAR_SI = 1.054571817e-34
C_SI = 299_792_458.0

SOURCE_LOCALIZATION_SATURATION_RECEIPT = "SOURCE_LOCALIZATION_SATURATION_RECEIPT"
MODULAR_SOURCE_CHARGE_RECEIPT = "MODULAR_SOURCE_CHARGE_RECEIPT"
ANOMALY_CURRENT_CONSERVATION_RECEIPT = "ANOMALY_CURRENT_CONSERVATION_RECEIPT"

FORBIDDEN_INPUT_KEYS = (
    "cmb_likelihood",
    "cmb_spectra",
    "planck_likelihood",
    "planck_compressed_rows",
    "bao_likelihood",
    "desi_likelihood",
    "sparc_likelihood",
    "weak_lensing_likelihood",
    "cluster_likelihood",
    "supernova_likelihood",
    "rsd_likelihood",
    "observed_A_s",
    "observed_sigma8",
    "observed_S8",
    "planck_tt",
    "planck_te",
    "planck_ee",
)

ALLOWED_INPUT_CLASSES = (
    "finite OPH regulator state",
    "screen/collar topology",
    "P fixed-point value",
    "declared branch data",
    "simulation random seed",
    "finite repair menu",
    "finite packet/collar transition counts",
)

OUTPUT_FILENAMES = {
    "release_code": "release_code_certificate.json",
    "parent_collar": "parent_collar_certificate.json",
    "repair_matrix": "repair_matrix_certificate.json",
    "boltzmann_export": "boltzmann_export_certificate.json",
    "no_data_use": "no_data_use_receipt.json",
    "manifest": "finite_certificate_manifest.json",
    "report": "finite_certificate_report.json",
}


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def entropy_bits(p: np.ndarray) -> float:
    values = np.asarray(p, dtype=np.float64)
    values = values[values > 0.0]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log2(values)))


def cmi_bits(p_abd: np.ndarray) -> float:
    """Compute classical I(A:D|B) in bits from a finite A/B/D joint table."""

    p = np.asarray(p_abd, dtype=np.float64)
    total = float(np.sum(p))
    if total <= 0.0:
        raise ValueError("joint distribution has zero mass")
    p = p / total
    p_ab = p.sum(axis=2)
    p_bd = p.sum(axis=0)
    p_b = p.sum(axis=(0, 2))
    value = entropy_bits(p_ab) + entropy_bits(p_bd) - entropy_bits(p_b) - entropy_bits(p)
    return max(0.0, float(value))


def release_code_certificate(data: dict[str, Any], input_hash: str) -> dict[str, Any]:
    release = data.get("release_code", {})
    packets = release.get("packets", [])
    if not isinstance(packets, list) or not packets:
        raise ValueError("release_code.packets must be a non-empty list")

    cmi_values: list[tuple[str, float, float]] = []
    for index, packet in enumerate(packets):
        if not bool(packet.get("scalar_visible", False)):
            continue
        packet_id = str(packet.get("id") or packet.get("packet_id") or f"packet_{index}")
        cmi = _packet_cmi_bits(packet)
        weight = float(packet.get("W_rel", packet.get("weight", 1.0)))
        cmi_values.append((packet_id, cmi, weight))

    threshold = float(release.get("positive_cmi_threshold", 1.0e-15))
    positive = [(packet_id, cmi, weight) for packet_id, cmi, weight in cmi_values if cmi > threshold]
    if not positive:
        raise ValueError("no positive scalar-visible CMI values in release code")
    epsilon_star = min(cmi for _, cmi, _ in positive)
    tolerance = max(1.0e-14, epsilon_star * 1.0e-9)
    minimizers = [(packet_id, cmi, weight) for packet_id, cmi, weight in positive if abs(cmi - epsilon_star) <= tolerance]
    kappa_base = float(release.get("kappa_rel", 1.0))
    weight_avg = sum(weight for _, _, weight in minimizers) / len(minimizers)
    kappa_rel = kappa_base * weight_avg
    a_q_cmi_upper_bound = 4.0 * LN2 * kappa_rel * epsilon_star
    energy_budget = _float_or_none(release.get("scalar_energy_budget"))
    physical_mode_count = _float_or_none(release.get("physical_mode_count"))
    a_q_energy = None
    scalar_release_amplitude_certificate = False
    if energy_budget is not None and physical_mode_count is not None and energy_budget >= 0.0 and physical_mode_count > 0.0:
        a_q_energy = float(energy_budget / physical_mode_count)
        scalar_release_amplitude_certificate = True
    no_data = no_data_use_receipt(data)
    return {
        "certificate_type": "release_code_certificate",
        "mode": "finite_scalar_release_code",
        "epsilon_star_bits": float(epsilon_star),
        "kappa_rel": float(kappa_rel),
        "N_rel": len(positive),
        "A_q_cmi_upper_bound": float(a_q_cmi_upper_bound),
        "A_q_energy": a_q_energy,
        "scalar_energy_budget": energy_budget,
        "physical_mode_count": physical_mode_count,
        "SCALAR_RELEASE_AMPLITUDE_CERTIFICATE": scalar_release_amplitude_certificate,
        "A_zeta": None,
        "A_zeta_formula": "pending SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT",
        "SCREEN_TO_RADIAL_LIFT_RECEIPT": False,
        "SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT": False,
        "minimizer_packets": [packet_id for packet_id, _, _ in minimizers],
        "packets_checked": len(packets),
        "scalar_visible_packets_checked": len(cmi_values),
        "all_positive_cmi_bits": [float(cmi) for _, cmi, _ in positive],
        "no_data_use": bool(no_data["no_data_use_receipt"]),
        "input_hash": input_hash,
        "claim_boundary": (
            "Finite scalar-release code certificate from packet/collar CMI data. CMI supplies an upper "
            "bound on scalar release observables, not A_zeta. A_q is derived only when finite scalar "
            "release energy and physical mode count are supplied; A_zeta remains null until the "
            "screen-to-primordial lift receipt passes."
        ),
    }


def parent_collar_certificate(data: dict[str, Any], input_hash: str) -> dict[str, Any]:
    parent = data.get("parent_collar", {})
    samples = parent.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError("parent_collar.samples must be a non-empty list")

    weights = np.asarray([float(sample.get("weight", 1.0)) for sample in samples], dtype=np.float64)
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("parent collar weights have zero mass")
    cmi_nats_values = np.asarray([_sample_cmi_diagnostic_nats(sample) for sample in samples], dtype=np.float64)
    d_cmi_values = np.asarray(
        [float(sample.get("d_cmi_d_cap_angle_proxy", sample.get("dI_d_delta_b", sample.get("d_cmi_d_delta_b", 0.0)))) for sample in samples],
        dtype=np.float64,
    )
    weighted_cmi_nats = float(np.sum(weights * cmi_nats_values) / np.sum(weights))
    weighted_derivative_proxy = float(np.sum(weights * d_cmi_values) / np.sum(weights))

    source_receipt = bool(parent.get(SOURCE_LOCALIZATION_SATURATION_RECEIPT, False))
    conservation_receipt = bool(parent.get(ANOMALY_CURRENT_CONSERVATION_RECEIPT, False))
    modular_charge_values = [
        _float_or_none(sample.get("modular_source_charge_nats", sample.get("source_charge_nats")))
        for sample in samples
    ]
    charge_available = all(value is not None and value >= 0.0 for value in modular_charge_values)
    weighted_modular_charge_nats = (
        float(np.sum(weights * np.asarray(modular_charge_values, dtype=np.float64)) / np.sum(weights))
        if charge_available
        else None
    )
    source_residual_values = [
        _float_or_none(sample.get("source_localization_residual_nats"))
        for sample in samples
        if sample.get("source_localization_residual_nats") is not None
    ]
    source_localization_residual_nats = (
        float(max(source_residual_values)) if source_residual_values else _float_or_none(parent.get("source_localization_residual_nats"))
    )

    ell = float(parent.get("proper_ell_m", parent.get("ell_m", parent.get("ell", parent.get("ell_r", 1.0)))))
    c_light = float(parent.get("c_si", parent.get("c", C_SI)))
    if ell <= 0.0 or c_light <= 0.0:
        raise ValueError("parent_collar.ell and parent_collar.c must be positive")
    density_factor = 15.0 * HBAR_SI / (8.0 * math.pi**2 * c_light * ell**4)
    energy_density_factor = 15.0 * HBAR_SI * c_light / (8.0 * math.pi**2 * ell**4)
    rho_a0 = density_factor * weighted_modular_charge_nats if source_receipt and weighted_modular_charge_nats is not None else None
    epsilon_a0 = energy_density_factor * weighted_modular_charge_nats if source_receipt and weighted_modular_charge_nats is not None else None
    rho_b = float(parent.get("rho_b", parent.get("rho_b_background", 1.0)))
    rho_a_background_default = rho_a0 if rho_a0 is not None and rho_a0 > 0.0 else 1.0
    rho_a_background = float(parent.get("rho_A_background", rho_a_background_default))
    if rho_b == 0.0 or rho_a_background == 0.0:
        raise ValueError("rho_b and rho_A_background must be non-zero for kernel export")
    k_a_rho = None
    b_a = None
    v_com = float(parent.get("V_com", 1.0))
    a_values = parent.get("a_values", [1.0])
    rho_a_by_a = (
        {str(a): float(rho_a0 * (float(a) ** -3)) for a in a_values}
        if rho_a0 is not None and conservation_receipt
        else {}
    )
    no_data = no_data_use_receipt(data)
    return {
        "certificate_type": "parent_collar_certificate",
        "mode": "finite_parent_collar_source_localization",
        "Q_A": float(rho_a0 * v_com) if rho_a0 is not None and conservation_receipt else None,
        "rho_A0_kg_m3": float(rho_a0) if rho_a0 is not None else None,
        "epsilon_A0_J_m3": float(epsilon_a0) if epsilon_a0 is not None else None,
        "rho_A_by_a": rho_a_by_a,
        "kernels": [
            {
                "k_screen_angle_proxy": sample.get("k_screen_angle_proxy", sample.get("k", "aggregate")),
                "a": sample.get("a", 1.0),
                "K_A_rho": k_a_rho,
                "B_A": b_a,
            }
            for sample in samples[:1]
        ],
        "cmi_diagnostic_nats": float(weighted_cmi_nats),
        "weighted_collar_cmi_nats": float(weighted_cmi_nats),
        "weighted_collar_derivative_cap_angle_proxy_nats": float(weighted_derivative_proxy),
        "modular_source_charge_nats": float(weighted_modular_charge_nats) if weighted_modular_charge_nats is not None else None,
        "source_localization_residual_nats": source_localization_residual_nats,
        SOURCE_LOCALIZATION_SATURATION_RECEIPT: bool(source_receipt and weighted_modular_charge_nats is not None),
        MODULAR_SOURCE_CHARGE_RECEIPT: bool(weighted_modular_charge_nats is not None),
        ANOMALY_CURRENT_CONSERVATION_RECEIPT: conservation_receipt,
        "samples_checked": len(samples),
        "small_field_support": parent.get("small_field_support", {"passes": False}),
        "refinement_convergence": parent.get("refinement_convergence", {"provided": False}),
        "parent_formula": "rho_A = 15 hbar modular_source_charge_nats/(8 pi^2 c proper_ell_m^4) after SOURCE_LOCALIZATION_SATURATION_RECEIPT",
        "no_data_use": bool(no_data["no_data_use_receipt"]),
        "input_hash": input_hash,
        "claim_boundary": (
            "Finite parent-collar certificate. Classical collar CMI is diagnostic only; physical anomaly "
            "density is emitted only from modular_source_charge_nats after the source-localization saturation "
            "receipt, and rho_A(a) is emitted only after anomaly-current conservation is certified."
        ),
    }


def build_metropolis(pi: np.ndarray, edges: list[list[int]]) -> np.ndarray:
    weights = np.asarray(pi, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("pi must be a non-empty one-dimensional array")
    if np.any(weights <= 0.0):
        raise ValueError("pi entries must be positive")
    weights = weights / float(np.sum(weights))
    n = int(weights.size)
    q = np.zeros((n, n), dtype=np.float64)
    degree = np.zeros(n, dtype=np.int64)
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        _validate_state_index(i, n)
        _validate_state_index(j, n)
        if i == j:
            continue
        degree[i] += 1
        degree[j] += 1
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        if i == j:
            continue
        q[i, j] = 1.0 / float(degree[i])
        q[j, i] = 1.0 / float(degree[j])

    kernel = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j or q[i, j] == 0.0:
                continue
            ratio = (weights[j] * q[j, i]) / (weights[i] * q[i, j])
            kernel[i, j] = q[i, j] * min(1.0, float(ratio))
        kernel[i, i] = 1.0 - float(np.sum(kernel[i]))
    return kernel


def repair_matrix_certificate(data: dict[str, Any], input_hash: str) -> dict[str, Any]:
    repair = data.get("repair_matrix", {})
    states = list(repair.get("states", []))
    if not states:
        raise ValueError("repair_matrix.states must be non-empty")
    pi = np.asarray(repair.get("pi_eq", repair.get("stationary_distribution", [])), dtype=np.float64)
    if pi.size != len(states):
        raise ValueError("repair_matrix.pi_eq length must match states")
    pi = pi / float(np.sum(pi))
    kernel = build_metropolis(pi, repair.get("proposal_edges", []))
    sym = np.diag(np.sqrt(pi)) @ kernel @ np.diag(1.0 / np.sqrt(pi))
    eig = np.linalg.eigvalsh((sym + sym.T) / 2.0)
    nontrivial = [abs(float(value)) for value in eig if abs(float(value) - 1.0) > 1.0e-10]
    lambda_2 = max(nontrivial) if nontrivial else 0.0
    dt_eta = float(repair.get("dt_eta", repair.get("delta_eta", 1.0)))
    gamma_rec = -math.log(lambda_2) / dt_eta if lambda_2 > 0.0 else float("inf")
    detailed_balance_error = _detailed_balance_max_error(pi, kernel)
    no_data = no_data_use_receipt(data)
    return {
        "certificate_type": "repair_matrix_certificate",
        "mode": "finite_reversible_metropolis_repair_kernel",
        "states": states,
        "stationary_distribution": pi.tolist(),
        "transition_matrix": kernel.tolist(),
        "row_sum_max_error": float(np.max(np.abs(kernel.sum(axis=1) - 1.0))),
        "lambda_2": float(lambda_2),
        "Gamma_rec": float(gamma_rec),
        "dt_eta": float(dt_eta),
        "detailed_balance_max_error": float(detailed_balance_error),
        "no_data_use": bool(no_data["no_data_use_receipt"]),
        "input_hash": input_hash,
        "claim_boundary": (
            "Finite repair-matrix certificate for a reversible packet-state Metropolis kernel. "
            "A physical relaxation kernel needs this construction for the actual simulator packet space."
        ),
    }


def boltzmann_export_certificate(
    release: dict[str, Any],
    parent: dict[str, Any],
    repair: dict[str, Any],
    data: dict[str, Any],
    input_hash: str,
) -> dict[str, Any]:
    p_value = float(data.get("metadata", {}).get("P", P_STAR))
    no_data = no_data_use_receipt(data)
    return {
        "certificate_type": "boltzmann_export_certificate",
        "mode": "finite_certificate_boltzmann_handoff",
        "n_s": float(1.0 - p_value / 48.0),
        "A_zeta": release.get("A_zeta"),
        "A_q_energy": release.get("A_q_energy"),
        "A_q_cmi_upper_bound": release.get("A_q_cmi_upper_bound"),
        "SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT": bool(release.get("SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT", False)),
        "SCREEN_TO_RADIAL_LIFT_RECEIPT": bool(release.get("SCREEN_TO_RADIAL_LIFT_RECEIPT", False)),
        "primordial_lift_ready": bool(release.get("SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT", False)),
        "rho_A_by_a": dict(parent["rho_A_by_a"]),
        "kernels": list(parent["kernels"]),
        "Gamma_rec": float(repair["Gamma_rec"]),
        "variables_required_by_module": [
            "rho_A(a)",
            "rho_A_eq(a)",
            "w_A(a)",
            "c_s_A^2(k,a)",
            "sigma_A(k,a)",
            "Q_A^mu",
            "B_A(k,a)",
            "Gamma_rec(k,a)",
        ],
        "cold_limit_test_required": True,
        "cold_limit_test_passed": bool(data.get("boltzmann_export", {}).get("cold_limit_test_passed", False)),
        "no_data_use": bool(no_data["no_data_use_receipt"]),
        "input_hash": input_hash,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Boltzmann-export certificate assembled from finite OPH certificates. It is a handoff "
            "contract, not a physical CMB prediction. Primordial amplitude remains unavailable until "
            "the screen-to-primordial lift receipt passes, and transfer remains gated by OPH anomaly "
            "kernels and cold-limit tests."
        ),
    }


def no_data_use_receipt(data: dict[str, Any]) -> dict[str, Any]:
    present = sorted(set(_find_forbidden_keys(data)))
    declared = data.get("no_data_use", data.get("metadata", {}).get("no_data_use", True))
    return {
        "no_data_use_receipt": bool(declared) and not present,
        "declared_no_data_use": bool(declared),
        "forbidden_inputs_present": present,
        "forbidden_inputs": [
            "CMB spectra/likelihoods",
            "BAO/DESI likelihoods",
            "SPARC fitting targets",
            "weak-lensing likelihoods",
            "cluster maps",
            "supernova/RSD likelihoods",
            "Planck compressed rows",
        ],
        "allowed_inputs": list(ALLOWED_INPUT_CLASSES),
        "claim_boundary": (
            "The finite certificate compiler may only consume OPH regulator/collar/repair state and declared "
            "branch constants. Measurement data must enter later through clearly separated comparison modules."
        ),
    }


def finite_certificate_bundle(data: dict[str, Any]) -> dict[str, Any]:
    input_hash = sha256_json(data)
    no_data = no_data_use_receipt(data)
    release = release_code_certificate(data, input_hash)
    parent = parent_collar_certificate(data, input_hash)
    repair = repair_matrix_certificate(data, input_hash)
    boltzmann = boltzmann_export_certificate(release, parent, repair, data, input_hash)
    outputs = {
        OUTPUT_FILENAMES["release_code"]: release,
        OUTPUT_FILENAMES["parent_collar"]: parent,
        OUTPUT_FILENAMES["repair_matrix"]: repair,
        OUTPUT_FILENAMES["boltzmann_export"]: boltzmann,
        OUTPUT_FILENAMES["no_data_use"]: no_data,
    }
    output_hashes = {name: sha256_json(obj) for name, obj in outputs.items()}
    readiness = {
        "release_code_certificate": bool(release.get("no_data_use")),
        "scalar_release_amplitude_certificate": bool(release.get("SCALAR_RELEASE_AMPLITUDE_CERTIFICATE")),
        "screen_to_primordial_lift_receipt": bool(release.get("SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT")),
        "screen_to_radial_lift_receipt": bool(release.get("SCREEN_TO_RADIAL_LIFT_RECEIPT")),
        "parent_collar_certificate": bool(parent.get("no_data_use")),
        "repair_matrix_certificate": bool(repair.get("no_data_use"))
        and repair.get("row_sum_max_error", 1.0) <= 1.0e-12
        and repair.get("detailed_balance_max_error", 1.0) <= 1.0e-12,
        "boltzmann_export_certificate": bool(boltzmann.get("no_data_use"))
        and bool(boltzmann.get("primordial_lift_ready", False)),
        "no_data_use_firewall": bool(no_data.get("no_data_use_receipt")),
    }
    compiler_ready = bool(
        readiness["release_code_certificate"]
        and readiness["parent_collar_certificate"]
        and readiness["repair_matrix_certificate"]
        and readiness["no_data_use_firewall"]
    )
    stack_ready = bool(all(readiness.values()))
    theorem_grade_inputs = bool(_theorem_grade_finite_inputs(data))
    real_physics_certificate = bool(
        compiler_ready
        and theorem_grade_inputs
        and not data.get("metadata", {}).get("proxy_certificate", True)
    )
    report = {
        "mode": "oph_finite_cosmology_certificate_bundle_v0",
        "input_hash": input_hash,
        "output_hashes": output_hashes,
        "readiness_gates": readiness,
        "finite_certificate_compiler_ready": compiler_ready,
        "finite_certificate_stack_ready": stack_ready,
        "theorem_grade_finite_inputs": theorem_grade_inputs,
        "proxy_certificate": not theorem_grade_inputs,
        "no_data_use_receipt": no_data,
        "derived_outputs": {
            "epsilon_star_bits": release["epsilon_star_bits"],
            "kappa_rel": release["kappa_rel"],
            "N_rel": release["N_rel"],
            "A_q_cmi_upper_bound": release["A_q_cmi_upper_bound"],
            "A_q_energy": release["A_q_energy"],
            "scalar_release_amplitude_certificate": release["SCALAR_RELEASE_AMPLITUDE_CERTIFICATE"],
            "screen_to_primordial_lift_receipt": release["SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT"],
            "A_zeta": release["A_zeta"],
            "Q_A": parent["Q_A"],
            "rho_A_by_a": parent["rho_A_by_a"],
            "B_A": [row.get("B_A") for row in parent["kernels"]],
            "Gamma_rec": repair["Gamma_rec"],
            "n_s": boltzmann["n_s"],
        },
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "real_physics_certificate": real_physics_certificate,
        "claim_boundary": (
            "Computed finite OPH cosmology certificate bundle. `finite_certificate_compiler_ready` means "
            "the compiler emitted internally consistent artifacts. It does not mean the inputs are "
            "theorem-grade. Physical prediction gates stay closed until the simulator emits real OPH "
            "regulator release packets, scalar-release energy, a screen-to-primordial lift receipt, "
            "collar response ladders, repair packet spaces, and Boltzmann cold-limit receipts without "
            "measurement data."
        ),
    }
    manifest = {
        "manifest_type": "oph_finite_certificate_manifest",
        "version": "0.2.0",
        "input_hash": input_hash,
        "outputs": output_hashes,
        "readiness_gates": readiness,
        "finite_certificate_compiler_ready": report["finite_certificate_compiler_ready"],
        "finite_certificate_stack_ready": report["finite_certificate_stack_ready"],
        "theorem_grade_finite_inputs": report["theorem_grade_finite_inputs"],
        "proxy_certificate": report["proxy_certificate"],
        "no_data_use_receipt": no_data["no_data_use_receipt"],
        "physical_cmb_prediction": False,
        "claim_boundary": report["claim_boundary"],
    }
    return {
        "release_code": release,
        "parent_collar": parent,
        "repair_matrix": repair,
        "boltzmann_export": boltzmann,
        "no_data_use": no_data,
        "manifest": manifest,
        "report": report,
    }


def _theorem_grade_finite_inputs(data: dict[str, Any]) -> bool:
    meta = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    release = data.get("release_code", {}) if isinstance(data.get("release_code"), dict) else {}
    parent = data.get("parent_collar", {}) if isinstance(data.get("parent_collar"), dict) else {}
    repair = data.get("repair_matrix", {}) if isinstance(data.get("repair_matrix"), dict) else {}
    boltzmann = data.get("boltzmann_export", {}) if isinstance(data.get("boltzmann_export"), dict) else {}
    small_field = parent.get("small_field_support", {}) if isinstance(parent.get("small_field_support"), dict) else {}
    refinement = (
        parent.get("refinement_convergence", {})
        if isinstance(parent.get("refinement_convergence"), dict)
        else {}
    )

    return bool(
        meta.get("real_physics_certificate", False)
        and not meta.get("proxy_certificate", True)
        and release.get("theorem_grade_release_code", False)
        and parent.get("theorem_grade_parent_collar_ladder", False)
        and small_field.get("passes", False)
        and refinement.get("passes", False)
        and repair.get("theorem_grade_repair_matrix", False)
        and repair.get("actual_repair_event_trace", False)
        and boltzmann.get("cold_limit_test_passed", False)
    )


def write_finite_certificate_bundle(
    input_path: Path | None,
    out_dir: Path,
    *,
    toy: bool = False,
) -> dict[str, Any]:
    if toy:
        data = toy_certificate_input()
        input_label = "built_in_toy_universe_state"
    else:
        if input_path is None:
            raise ValueError("--input is required unless --toy is set")
        data = json.loads(Path(input_path).read_text(encoding="utf-8"))
        input_label = str(input_path)
    bundle = finite_certificate_bundle(data)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, filename in OUTPUT_FILENAMES.items():
        if key in {"manifest", "report"}:
            continue
        (out / filename).write_text(json.dumps(bundle[key], indent=2, sort_keys=True, default=str), encoding="utf-8")
    (out / OUTPUT_FILENAMES["manifest"]).write_text(
        json.dumps({**bundle["manifest"], "inputs": {input_label: bundle["report"]["input_hash"]}}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / OUTPUT_FILENAMES["report"]).write_text(
        json.dumps({**bundle["report"], "input": input_label}, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    (out / "finite_certificate_report.md").write_text(_markdown_report(bundle["report"], input_label), encoding="utf-8")
    return {**bundle["report"], "input": input_label, "out_dir": str(out)}


def run_proxy_certificate_input(run_dir: Path) -> dict[str, Any]:
    """Build a finite-certificate input object from cached OPH-FPE run receipts.

    This intentionally emits a proxy bundle. It uses only simulator receipts and
    declared OPH constants, but current cached runs do not yet contain the full
    noncommutative release code, theorem-grade parent-collar ladder, or physical
    repair packet matrix required for a physical CMB prediction.
    """

    root = Path(run_dir)
    collar_path = root / "collar_markov_report.json"
    collar = read_json_file(collar_path)
    rows = collar.get("rows") if isinstance(collar.get("rows"), list) else []
    if not rows:
        raise ValueError(f"{collar_path} has no collar rows")

    packets: list[dict[str, Any]] = []
    parent_samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        cmi = _float_from_any(row.get("epsilon_cmi"), 0.0)
        if cmi <= 0.0:
            continue
        sample_count = max(1.0, _float_from_any(row.get("sample_count"), 1.0))
        collar_count = max(1.0, _float_from_any(row.get("collar_count"), 1.0))
        collar_fraction = min(1.0, collar_count / sample_count)
        cap_id = row.get("cap_id", index)
        packets.append(
            {
                "id": f"cap_{cap_id}",
                "scalar_visible": True,
                "cmi_bits": float(cmi / LN2),
                "cmi_diagnostic_nats": float(cmi),
                "W_rel": float(collar_fraction),
                "source": "collar_markov_report",
                "theta0": row.get("theta0"),
                "collar_width": row.get("collar_width"),
                "collar_count": row.get("collar_count"),
                "sample_count": row.get("sample_count"),
                "claim_boundary": "diagonal collar CMI proxy, not a noncommutative scalar-release proof",
            }
        )
        derivative = _finite_collar_derivative_proxy(row, rows, index)
        parent_samples.append(
            {
                "id": f"cap_{cap_id}",
                "weight": float(collar_count),
                "I_bits": float(cmi),
                "cmi_diagnostic_nats": float(cmi * LN2),
                "d_cmi_d_cap_angle_proxy": float(derivative * LN2),
                "theta0": row.get("theta0"),
                "k_screen_angle_proxy": _theta_to_k_proxy(row.get("theta0")),
                "k_proxy_units": "inverse_cap_opening_angle_proxy",
                "a": 1.0,
                "source": "finite_collar_derivative_proxy_from_collar_markov_report",
            }
        )
    if not packets:
        raise ValueError(f"{collar_path} has no positive epsilon_cmi rows")

    manifest = read_json_file(root / "manifest.json")
    h0s8 = read_json_file(root / "h0s8_branch_report.json")
    boltzmann = read_json_file(root / "oph_boltzmann_input_report.json")
    s3_counts = read_json_file(root / "s3_class_counts.json")

    omega_a = _nested_float(h0s8, ("background_values", "Omega_A"), default=0.26447041034523616)
    omega_b = _nested_float(h0s8, ("background_values", "Omega_b"), default=0.049301692328524445)
    a_values = _a_values_from_boltzmann_report(boltzmann)
    repair_states, repair_pi = _repair_state_distribution(s3_counts)

    source_hashes = {}
    for name in (
        "collar_markov_report.json",
        "manifest.json",
        "s3_class_counts.json",
        "h0s8_branch_report.json",
        "oph_boltzmann_input_report.json",
    ):
        path = root / name
        if path.exists():
            source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "metadata": {
            "name": f"finite_certificate_proxy_from_{manifest.get('run_id') or root.name}",
            "version": "0.3.0",
            "source_run_dir": str(root),
            "source_hashes": source_hashes,
            "P": _nested_float(manifest, ("oph_constants", "P"), default=P_STAR),
            "no_data_use": True,
            "proxy_certificate": True,
            "real_physics_certificate": False,
            "theorem_grade_release_code": False,
            "theorem_grade_parent_collar_ladder": False,
            "theorem_grade_repair_matrix": False,
            "claim_boundary": (
                "Proxy finite-certificate input assembled from cached OPH-FPE receipts. "
                "It uses no measurement likelihoods, but it is not a physical CMB certificate "
                "because parent-collar and repair-matrix entries are finite diagnostic proxies."
            ),
        },
        "release_code": {
            "kappa_rel": 1.0,
            "positive_cmi_threshold": 0.0,
            "theorem_grade_release_code": False,
            "packets": packets,
        },
        "parent_collar": {
            "theorem_grade_parent_collar_ladder": False,
            "ell": 1.0,
            "proper_ell_m": 1.0,
            SOURCE_LOCALIZATION_SATURATION_RECEIPT: False,
            MODULAR_SOURCE_CHARGE_RECEIPT: False,
            ANOMALY_CURRENT_CONSERVATION_RECEIPT: False,
            "a_values": a_values,
            "V_com": 1.0,
            "c_si": C_SI,
            "rho_b": omega_b,
            "rho_A_background": omega_a,
            "samples": parent_samples,
            "small_field_support": {
                "passes": False,
                "reason": "not emitted by current finite run; derivative rows are collar proxies",
            },
            "refinement_convergence": {
                "passes": False,
                "provided": False,
                "reason": "single cached run, not a regulator ladder",
            },
        },
        "repair_matrix": {
            "theorem_grade_repair_matrix": False,
            "actual_repair_event_trace": False,
            "states": repair_states,
            "pi_eq": repair_pi,
            "proposal_edges": [[i, j] for i in range(len(repair_states)) for j in range(i + 1, len(repair_states))],
            "dt_eta": 1.0,
            "source": "s3_class_counts_proxy_distribution",
        },
        "boltzmann_export": {
            "cold_limit_test_passed": bool(_nested_get(boltzmann, "readiness", "cdm_limit_solver_ready", default=False)),
            "source": "oph_boltzmann_input_report",
        },
        "no_data_use": True,
    }


def write_run_proxy_finite_certificate_bundle(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    data = run_proxy_certificate_input(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_path = out / "finite_certificate_input_from_run.json"
    input_path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    result = write_finite_certificate_bundle(input_path, out)
    report_path = out / OUTPUT_FILENAMES["report"]
    report = read_json_file(report_path)
    report["source_run_dir"] = str(run_dir)
    report["input_builder"] = "run_proxy_certificate_input"
    report["proxy_certificate"] = True
    report["theorem_grade_finite_inputs"] = False
    report["real_physics_certificate"] = False
    report["physical_cmb_prediction"] = False
    report["physical_matter_power_prediction"] = False
    report["claim_boundary"] = (
        "Run-derived finite-certificate proxy bundle. All finite-certificate algorithms validated on "
        "cached OPH-FPE receipts, but physical prediction gates remain closed because the parent-collar "
        "response and repair matrix are diagnostic proxies, not theorem-grade finite Universe Simulation certificates."
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (out / "finite_certificate_report.md").write_text(_markdown_report(report, str(input_path)), encoding="utf-8")
    return {**report, "out_dir": str(out)}


def toy_certificate_input() -> dict[str, Any]:
    return {
        "metadata": {
            "name": "toy_universe_certificate_test",
            "version": "0.2.0",
            "note": "Toy format test only; not a physical OPH certificate.",
            "P": P_STAR,
            "no_data_use": True,
            "real_physics_certificate": False,
        },
        "release_code": {
            "kappa_rel": 1.0,
            "packets": [
                {
                    "id": "q0_zero",
                    "scalar_visible": True,
                    "W_rel": 1.0,
                    "joint_p_abd": [[[0.25, 0.25], [0.0, 0.0]], [[0.0, 0.0], [0.25, 0.25]]],
                },
                {
                    "id": "q1_defect",
                    "scalar_visible": True,
                    "W_rel": 1.0,
                    "joint_p_abd": [[[0.36, 0.04], [0.04, 0.06]], [[0.04, 0.06], [0.06, 0.34]]],
                },
                {
                    "id": "q2_defect",
                    "scalar_visible": True,
                    "W_rel": 0.7,
                    "joint_p_abd": [[[0.31, 0.09], [0.02, 0.08]], [[0.02, 0.08], [0.10, 0.30]]],
                },
            ],
        },
        "parent_collar": {
            "ell": 1.0,
            "proper_ell_m": 1.0,
            SOURCE_LOCALIZATION_SATURATION_RECEIPT: True,
            MODULAR_SOURCE_CHARGE_RECEIPT: True,
            ANOMALY_CURRENT_CONSERVATION_RECEIPT: True,
            "a_values": [1.0, 0.5],
            "V_com": 1.0,
            "c_si": C_SI,
            "rho_b": 0.049,
            "rho_A_background": 0.264,
            "samples": [
                {
                    "id": "C0",
                    "weight": 1.0,
                    "I_bits": 0.012,
                    "modular_source_charge_nats": 0.012 * LN2,
                    "source_localization_residual_nats": 0.0,
                    "d_cmi_d_cap_angle_proxy": 0.0010 * LN2,
                },
                {
                    "id": "C1",
                    "weight": 2.0,
                    "I_bits": 0.018,
                    "modular_source_charge_nats": 0.018 * LN2,
                    "source_localization_residual_nats": 0.0,
                    "d_cmi_d_cap_angle_proxy": 0.0015 * LN2,
                },
                {
                    "id": "C2",
                    "weight": 1.5,
                    "I_bits": 0.015,
                    "modular_source_charge_nats": 0.015 * LN2,
                    "source_localization_residual_nats": 0.0,
                    "d_cmi_d_cap_angle_proxy": 0.0012 * LN2,
                },
            ],
            "small_field_support": {"passes": True, "min_x_positive": 0.001},
        },
        "repair_matrix": {
            "states": ["s0", "s1", "s2"],
            "pi_eq": [0.5, 0.3, 0.2],
            "proposal_edges": [[0, 1], [1, 2], [0, 2]],
            "dt_eta": 1.0,
        },
    }


def _finite_collar_derivative_proxy(row: dict[str, Any], rows: list[dict[str, Any]], index: int) -> float:
    theta = _float_from_any(row.get("theta0"), float(index + 1))
    cmi = _float_from_any(row.get("epsilon_cmi"), 0.0)
    neighbors = []
    for other_index, other in enumerate(rows):
        if other_index == index:
            continue
        other_theta = _float_from_any(other.get("theta0"), math.nan)
        other_cmi = _float_from_any(other.get("epsilon_cmi"), math.nan)
        if math.isfinite(other_theta) and math.isfinite(other_cmi) and other_theta != theta:
            neighbors.append((abs(other_theta - theta), other_theta, other_cmi))
    if neighbors:
        _, other_theta, other_cmi = min(neighbors, key=lambda item: item[0])
        return (cmi - other_cmi) / (theta - other_theta)
    return cmi


def _theta_to_k_proxy(theta: Any) -> float:
    value = _float_from_any(theta, 1.0)
    return 1.0 / max(value, 1.0e-12)


def _a_values_from_boltzmann_report(report: dict[str, Any]) -> list[float]:
    values = _nested_get(report, "grids", "a_grid", default=None)
    if isinstance(values, list) and values:
        return [float(value) for value in values]
    return [0.000909090909, 0.01, 0.1, 1.0]


def _repair_state_distribution(counts: dict[str, Any]) -> tuple[list[str], list[float]]:
    preferred = ["identity", "transposition", "threecycle"]
    values = [max(0.0, _float_from_any(counts.get(name), 0.0)) for name in preferred]
    if sum(values) <= 0.0:
        return ["identity", "transposition", "threecycle"], [1.0, 1.0, 1.0]
    positive = [(name, value) for name, value in zip(preferred, values) if value > 0.0]
    return [name for name, _ in positive], [float(value) for _, value in positive]


def _nested_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _nested_float(data: dict[str, Any], keys: tuple[str, ...], *, default: float) -> float:
    return _float_from_any(_nested_get(data, *keys, default=default), default)


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _float_from_any(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return parsed


def _packet_cmi_bits(packet: dict[str, Any]) -> float:
    if "joint_p_abd" in packet:
        return cmi_bits(np.asarray(packet["joint_p_abd"], dtype=np.float64))
    for key in ("cmi_bits", "I_bits", "cmi", "I"):
        if key in packet:
            return max(0.0, float(packet[key]))
    entropy = packet.get("entropy")
    if isinstance(entropy, dict):
        return max(
            0.0,
            float(entropy.get("AB", 0.0))
            + float(entropy.get("BD", 0.0))
            - float(entropy.get("B", 0.0))
            - float(entropy.get("ABD", 0.0)),
        )
    raise ValueError(f"packet {packet.get('id') or packet.get('packet_id')} has no CMI source")


def _sample_cmi_diagnostic_nats(sample: dict[str, Any]) -> float:
    for key in ("cmi_diagnostic_nats", "I_nats", "cmi_nats"):
        value = _float_or_none(sample.get(key))
        if value is not None:
            return max(0.0, float(value))
    for key in ("I_bits", "cmi_bits", "cmi", "I"):
        value = _float_or_none(sample.get(key))
        if value is not None:
            return max(0.0, float(value) * LN2)
    return 0.0


def _validate_state_index(index: int, state_count: int) -> None:
    if index < 0 or index >= state_count:
        raise ValueError(f"proposal edge index {index} outside state range 0..{state_count - 1}")


def _detailed_balance_max_error(pi: np.ndarray, kernel: np.ndarray) -> float:
    error = 0.0
    for i in range(len(pi)):
        for j in range(len(pi)):
            error = max(error, abs(float(pi[i] * kernel[i, j] - pi[j] * kernel[j, i])))
    return float(error)


def _find_forbidden_keys(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip()
            if normalized in FORBIDDEN_INPUT_KEYS:
                found.append(normalized)
            found.extend(_find_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_forbidden_keys(child))
    return found


def _markdown_report(report: dict[str, Any], input_label: str) -> str:
    derived = report["derived_outputs"]
    gates = report["readiness_gates"]
    return "\n".join(
        [
            "# OPH Finite Cosmology Certificate Bundle",
            "",
            report["claim_boundary"],
            "",
            f"- input: `{input_label}`",
            f"- compiler ready: {report['finite_certificate_compiler_ready']}",
            f"- legacy stack ready: {report['finite_certificate_stack_ready']}",
            f"- theorem-grade finite inputs: {report['theorem_grade_finite_inputs']}",
            f"- proxy certificate: {report['proxy_certificate']}",
            f"- no-data-use firewall: {gates['no_data_use_firewall']}",
            f"- physical CMB prediction: {report['physical_cmb_prediction']}",
            f"- real physics certificate: {report['real_physics_certificate']}",
            "",
            "## Derived Values",
            "",
            f"- epsilon_star_bits: {derived['epsilon_star_bits']}",
            f"- A_q CMI upper bound: {derived['A_q_cmi_upper_bound']}",
            f"- A_q energy: {derived['A_q_energy']}",
            f"- A_zeta: {derived['A_zeta'] if derived['A_zeta'] is not None else 'pending lift receipt'}",
            f"- n_s: {derived['n_s']}",
            f"- Q_A: {derived['Q_A']}",
            f"- B_A: {derived['B_A']}",
            f"- Gamma_rec: {derived['Gamma_rec']}",
            "",
            "## Gates",
            "",
            *[f"- {key}: {value}" for key, value in gates.items()],
            "",
        ]
    )
