from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.constants.oph_pixel import P_STAR

LN2 = math.log(2.0)

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
    a_zeta = 100.0 * LN2 * kappa_rel * epsilon_star
    no_data = no_data_use_receipt(data)
    return {
        "certificate_type": "release_code_certificate",
        "mode": "finite_scalar_release_code",
        "epsilon_star_bits": float(epsilon_star),
        "kappa_rel": float(kappa_rel),
        "N_rel": len(positive),
        "A_zeta": float(a_zeta),
        "A_zeta_formula": "A_zeta = 100 ln(2) kappa_rel epsilon_star_bits",
        "minimizer_packets": [packet_id for packet_id, _, _ in minimizers],
        "packets_checked": len(packets),
        "scalar_visible_packets_checked": len(cmi_values),
        "all_positive_cmi_bits": [float(cmi) for _, cmi, _ in positive],
        "no_data_use": bool(no_data["no_data_use_receipt"]),
        "input_hash": input_hash,
        "claim_boundary": (
            "Finite scalar-release code certificate from packet/collar CMI data. This certifies the "
            "finite input and algorithm only; it is a physical CMB-amplitude input only when the source "
            "data are real OPH regulator outputs and the full certificate stack passes."
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
    cmi_values = np.asarray([float(sample.get("I_bits", sample.get("cmi_bits", sample.get("cmi", 0.0)))) for sample in samples])
    d_cmi_values = np.asarray([float(sample.get("dI_d_delta_b", sample.get("d_cmi_d_delta_b", 0.0))) for sample in samples])
    weighted_cmi = float(np.sum(weights * cmi_values) / np.sum(weights))
    weighted_derivative = float(np.sum(weights * d_cmi_values) / np.sum(weights))

    ell = float(parent.get("ell", parent.get("ell_r", 1.0)))
    c_light = float(parent.get("c", 1.0))
    if ell <= 0.0 or c_light <= 0.0:
        raise ValueError("parent_collar.ell and parent_collar.c must be positive")
    factor = 15.0 / (8.0 * math.pi**2 * ell**4 * c_light**2)
    rho_a0 = factor * weighted_cmi
    drho_a_ddeltab = factor * weighted_derivative
    rho_b = float(parent.get("rho_b", parent.get("rho_b_background", 1.0)))
    rho_a_background = float(parent.get("rho_A_background", rho_a0 if rho_a0 > 0.0 else 1.0))
    if rho_b == 0.0 or rho_a_background == 0.0:
        raise ValueError("rho_b and rho_A_background must be non-zero for kernel export")
    k_a_rho = drho_a_ddeltab / rho_b
    b_a = (rho_b / rho_a_background) * k_a_rho
    v_com = float(parent.get("V_com", 1.0))
    a_values = parent.get("a_values", [1.0])
    rho_a_by_a = {str(a): float(rho_a0 * (float(a) ** -3)) for a in a_values}
    no_data = no_data_use_receipt(data)
    return {
        "certificate_type": "parent_collar_certificate",
        "mode": "finite_parent_collar_anomaly_load",
        "Q_A": float(rho_a0 * v_com),
        "rho_A_by_a": rho_a_by_a,
        "kernels": [{"k": sample.get("k", "aggregate"), "a": sample.get("a", 1.0), "K_A_rho": float(k_a_rho), "B_A": float(b_a)} for sample in samples[:1]],
        "weighted_collar_cmi_bits": float(weighted_cmi),
        "weighted_collar_derivative_bits": float(weighted_derivative),
        "samples_checked": len(samples),
        "small_field_support": parent.get("small_field_support", {"passes": False}),
        "refinement_convergence": parent.get("refinement_convergence", {"provided": False}),
        "parent_formula": "rho_A_eq c^2 = 15/(8 pi^2 ell^4) weighted_avg I(A:D|B)",
        "no_data_use": bool(no_data["no_data_use_receipt"]),
        "input_hash": input_hash,
        "claim_boundary": (
            "Finite parent-collar certificate for homogeneous anomaly load and first response-kernel row. "
            "It remains a finite-collar diagnostic until a regulator ladder and small-field response checks pass."
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
        "A_zeta": float(release["A_zeta"]),
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
            "contract, not a physical CMB prediction until the OPH anomaly module and cold-limit tests pass."
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
        "parent_collar_certificate": bool(parent.get("no_data_use")),
        "repair_matrix_certificate": bool(repair.get("no_data_use"))
        and repair.get("row_sum_max_error", 1.0) <= 1.0e-12
        and repair.get("detailed_balance_max_error", 1.0) <= 1.0e-12,
        "boltzmann_export_certificate": bool(boltzmann.get("no_data_use")),
        "no_data_use_firewall": bool(no_data.get("no_data_use_receipt")),
    }
    report = {
        "mode": "oph_finite_cosmology_certificate_bundle_v0",
        "input_hash": input_hash,
        "output_hashes": output_hashes,
        "readiness_gates": readiness,
        "finite_certificate_stack_ready": bool(all(readiness.values())),
        "no_data_use_receipt": no_data,
        "derived_outputs": {
            "epsilon_star_bits": release["epsilon_star_bits"],
            "kappa_rel": release["kappa_rel"],
            "N_rel": release["N_rel"],
            "A_zeta": release["A_zeta"],
            "Q_A": parent["Q_A"],
            "rho_A_by_a": parent["rho_A_by_a"],
            "B_A": [row.get("B_A") for row in parent["kernels"]],
            "Gamma_rec": repair["Gamma_rec"],
            "n_s": boltzmann["n_s"],
        },
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "real_physics_certificate": bool(data.get("metadata", {}).get("real_physics_certificate", False)),
        "claim_boundary": (
            "Computed finite OPH cosmology certificate bundle. Toy or proxy inputs validate the compiler only. "
            "Physical prediction gates stay closed until the simulator emits real OPH regulator release packets, "
            "collar response ladders, repair packet spaces, and Boltzmann cold-limit receipts without measurement data."
        ),
    }
    manifest = {
        "manifest_type": "oph_finite_certificate_manifest",
        "version": "0.2.0",
        "input_hash": input_hash,
        "outputs": output_hashes,
        "readiness_gates": readiness,
        "finite_certificate_stack_ready": report["finite_certificate_stack_ready"],
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
            "a_values": [1.0, 0.5],
            "V_com": 1.0,
            "c": 1.0,
            "rho_b": 0.049,
            "rho_A_background": 0.264,
            "samples": [
                {"id": "C0", "weight": 1.0, "I_bits": 0.012, "dI_d_delta_b": 0.0010},
                {"id": "C1", "weight": 2.0, "I_bits": 0.018, "dI_d_delta_b": 0.0015},
                {"id": "C2", "weight": 1.5, "I_bits": 0.015, "dI_d_delta_b": 0.0012},
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
            f"- stack ready: {report['finite_certificate_stack_ready']}",
            f"- no-data-use firewall: {gates['no_data_use_firewall']}",
            f"- physical CMB prediction: {report['physical_cmb_prediction']}",
            f"- real physics certificate: {report['real_physics_certificate']}",
            "",
            "## Derived Values",
            "",
            f"- epsilon_star_bits: {derived['epsilon_star_bits']}",
            f"- A_zeta: {derived['A_zeta']}",
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
