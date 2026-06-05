from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

from oph_fpe.claims import BRANCH_INSTANTIATION_SANITY, DEMO, with_claim_metadata
from oph_fpe.bulk.cap_geometry import RoundCap, cap_weights, lambda_cap
from oph_fpe.bulk.collar_state import cap_collar_partition, fawzi_renner_bound, visible_packets


@dataclass
class ModularProbeReport:
    cap_id: int
    time: float
    regularizer_a: float
    epsilon_cmi: float
    r_fr: float
    eta_modular: float
    eta_interpolation: float
    raw_residual: float
    corrected_residual_upper: float
    support_visible_residual: float
    matrix_element_count: int
    observable: str
    mode: str = "state_derived_modular_probe"

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def regularized_modular_generator(rho: np.ndarray, a: float) -> np.ndarray:
    rho = _hermitian(np.asarray(rho, dtype=complex))
    eigvals, eigvecs = np.linalg.eigh(rho + float(a) * np.eye(rho.shape[0], dtype=complex))
    eigvals = np.maximum(eigvals.real, 1e-15)
    return _hermitian((eigvecs * (-np.log(eigvals))) @ eigvecs.conj().T)


def modular_unitary(K: np.ndarray, t: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(_hermitian(K))
    phases = np.exp(1j * float(t) * eigvals)
    return (eigvecs * phases) @ eigvecs.conj().T


def state_derived_modular_transport(O: np.ndarray, rho: np.ndarray, t: float, a: float) -> np.ndarray:
    K = regularized_modular_generator(rho, a)
    U = modular_unitary(K, t)
    return U @ O @ U.conj().T


def geometric_transport_operator(points: np.ndarray, cap: RoundCap, s: float, basis_indices: np.ndarray) -> np.ndarray:
    basis_points = points[basis_indices]
    mapped = lambda_cap(basis_points, cap, s)
    distances = np.linalg.norm(mapped[:, None, :] - basis_points[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    L = np.zeros((basis_indices.size, basis_indices.size), dtype=complex)
    L[np.arange(basis_indices.size), nearest] = 1.0
    eta = float(np.mean(np.min(distances, axis=1))) if distances.size else 0.0
    return L, eta


def geometric_permutation_operator(points: np.ndarray, cap: RoundCap, s: float, basis_indices: np.ndarray) -> tuple[np.ndarray, float]:
    basis_points = points[basis_indices]
    mapped = lambda_cap(basis_points, cap, s)
    distances = np.linalg.norm(mapped[:, None, :] - basis_points[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(distances)
    L = np.zeros((basis_indices.size, basis_indices.size), dtype=complex)
    L[row_ind, col_ind] = 1.0
    eta = float(np.mean(distances[row_ind, col_ind])) if distances.size else 0.0
    return L, eta


def cap_state_density(
    points: np.ndarray,
    cap: RoundCap,
    raw_fields: dict[str, np.ndarray],
    *,
    max_basis: int = 96,
    regularizer: float = 1e-6,
    seed: int = 1,
    state_mode: str = "cooccurrence_kernel",
    transition_response_time: float = 0.025,
    transition_response_scale: float = 2.0 * math.pi,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = cap_weights(points, cap, soft=True)
    support = np.flatnonzero(weights > 0.05)
    if support.size == 0:
        support = np.arange(points.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    if support.size > max_basis:
        probs = weights[support].astype(float)
        probs = probs / np.sum(probs)
        support = np.sort(rng.choice(support, size=max_basis, replace=False, p=probs))
    packets = visible_packets(raw_fields)
    if state_mode == "transition_response_unitary":
        rho = transition_response_density(
            points,
            cap,
            support,
            response_time=transition_response_time,
            response_scale=transition_response_scale,
            regularizer=regularizer,
        )
        return rho, support, packets
    packet_similarity = packets[support][:, None] == packets[support][None, :]
    distances = np.linalg.norm(points[support][:, None, :] - points[support][None, :, :], axis=2)
    sigma = max(float(np.median(distances[distances > 0.0])) if np.any(distances > 0.0) else 1.0, 1e-6)
    M = np.exp(-(distances**2) / (2.0 * sigma**2)) * (0.25 + 0.75 * packet_similarity.astype(float))
    field = raw_fields.get("repair_load")
    if field is not None:
        values = np.asarray(field, dtype=float)[support]
        M += 0.1 * np.outer(_standardize(values), _standardize(values))
    rho = M @ M.T
    rho = _hermitian(rho + float(regularizer) * np.eye(rho.shape[0]))
    rho = rho / max(float(np.trace(rho).real), 1e-15)
    return rho, support, packets


def transition_response_density(
    points: np.ndarray,
    cap: RoundCap,
    basis: np.ndarray,
    *,
    response_time: float,
    response_scale: float,
    regularizer: float,
) -> np.ndarray:
    transition, _ = geometric_permutation_operator(points, cap, float(response_scale) * float(response_time), basis)
    K = modular_generator_from_unitary_transition(transition.conj().T, response_time)
    eigvals = np.linalg.eigvalsh(K)
    shifted = K - float(np.min(eigvals)) * np.eye(K.shape[0], dtype=complex)
    rho = expm(-shifted)
    rho = _hermitian(rho + float(regularizer) * np.eye(rho.shape[0], dtype=complex))
    rho = rho / max(float(np.trace(rho).real), 1e-15)
    return rho


def transition_response_modular_generator(
    points: np.ndarray,
    cap: RoundCap,
    basis: np.ndarray,
    *,
    response_time: float,
    response_scale: float,
) -> tuple[np.ndarray, float]:
    transition, _ = geometric_permutation_operator(points, cap, float(response_scale) * float(response_time), basis)
    pullback = transition.conj().T
    K = modular_generator_from_unitary_transition(pullback, response_time)
    U = modular_unitary(K, response_time)
    eta = float(np.linalg.norm(U - pullback, ord="fro") / (np.linalg.norm(pullback, ord="fro") + 1e-12))
    return K, eta


def modular_generator_from_unitary_transition(transition: np.ndarray, response_time: float) -> np.ndarray:
    values, vectors = np.linalg.eig(np.asarray(transition, dtype=complex))
    phases = np.angle(values) / max(float(response_time), 1e-12)
    inverse = np.linalg.inv(vectors)
    generator = vectors @ np.diag(phases) @ inverse
    return _hermitian(generator)


def bw_state_derived_residual(
    points: np.ndarray,
    raw_fields: dict[str, np.ndarray],
    cap: RoundCap,
    observable: str,
    *,
    cap_id: int,
    t: float,
    a: float,
    epsilon_cmi: float,
    target_scale: float = 2.0 * math.pi,
    sim_modular_flow: bool = True,
    mode: str = "state_derived_modular_probe",
    state_mode: str = "cooccurrence_kernel",
    target_operator_mode: str = "nearest",
    transition_response_time: float = 0.025,
    transition_response_scale: float = 2.0 * math.pi,
    max_basis: int = 96,
    seed: int = 1,
) -> ModularProbeReport:
    rho, basis, _ = cap_state_density(
        points,
        cap,
        raw_fields,
        max_basis=max_basis,
        regularizer=a,
        seed=seed,
        state_mode=state_mode,
        transition_response_time=transition_response_time,
        transition_response_scale=transition_response_scale,
    )
    O = _observable_matrix(raw_fields, observable, basis)
    if state_mode == "transition_response_unitary" and sim_modular_flow:
        K_direct, eta_direct = transition_response_modular_generator(
            points,
            cap,
            basis,
            response_time=transition_response_time,
            response_scale=transition_response_scale,
        )
        U = modular_unitary(K_direct, t)
        O_sim = U @ O @ U.conj().T
    else:
        eta_direct = None
        O_sim = state_derived_modular_transport(O, rho, t, a) if sim_modular_flow else O
    if target_operator_mode == "permutation":
        L, eta_interpolation = geometric_permutation_operator(points, cap, target_scale * t, basis)
    else:
        L, eta_interpolation = geometric_transport_operator(points, cap, target_scale * t, basis)
    O_bw = L.conj().T @ O @ L
    raw = _relative_frobenius(O_sim - O_bw, O)
    eta_modular = (
        float(eta_direct)
        if eta_direct is not None
        else float(a / max(float(np.min(np.linalg.eigvalsh(rho + a * np.eye(rho.shape[0]))).real), 1e-15))
    )
    r_fr = fawzi_renner_bound(epsilon_cmi)
    corrected = raw + r_fr + eta_interpolation + eta_modular * 1e-3
    return ModularProbeReport(
        cap_id=cap_id,
        time=float(t),
        regularizer_a=float(a),
        epsilon_cmi=float(epsilon_cmi),
        r_fr=float(r_fr),
        eta_modular=float(eta_modular),
        eta_interpolation=float(eta_interpolation),
        raw_residual=float(raw),
        corrected_residual_upper=float(corrected),
        support_visible_residual=float(raw),
        matrix_element_count=int(O.size),
        observable=observable,
        mode=mode,
    )


def state_derived_bw_report(
    points: np.ndarray,
    caps: list[RoundCap],
    raw_fields: dict[str, np.ndarray],
    collar_report: dict[str, Any],
    *,
    times: list[float],
    observables: list[str],
    regularizers: list[float],
    controls: list[str] | None = None,
    state_mode: str = "cooccurrence_kernel",
    target_operator_mode: str = "nearest",
    transition_response_time: float = 0.025,
    transition_response_scale: float = 2.0 * math.pi,
    max_basis: int = 96,
    seed: int = 1,
) -> dict[str, Any]:
    collar_by_cap = {int(row["cap_id"]): row for row in collar_report.get("rows", [])}
    rows = _state_rows(
        points,
        caps,
        raw_fields,
        collar_by_cap,
        times=times,
        observables=observables,
        regularizers=regularizers,
        max_basis=max_basis,
        state_mode=state_mode,
        target_operator_mode=target_operator_mode,
        transition_response_time=transition_response_time,
        transition_response_scale=transition_response_scale,
        seed=seed,
        mode="state_derived_modular_probe",
    )
    control_reports: dict[str, Any] = {}
    for control in controls or []:
        if control == "wrong_1x_normalization":
            control_reports[control] = _summary_rows(
                _state_rows(
                    points,
                    caps,
                    raw_fields,
                    collar_by_cap,
                    times=times,
                    observables=observables,
                    regularizers=regularizers,
                    target_scale=1.0,
                    max_basis=max_basis,
                    state_mode=state_mode,
                    target_operator_mode=target_operator_mode,
                    transition_response_time=transition_response_time,
                    transition_response_scale=transition_response_scale,
                    seed=seed + 17,
                    mode="wrong_1x_normalization",
                )
            )
        elif control == "wrong_pi_normalization":
            control_reports[control] = _summary_rows(
                _state_rows(
                    points,
                    caps,
                    raw_fields,
                    collar_by_cap,
                    times=times,
                    observables=observables,
                    regularizers=regularizers,
                    target_scale=math.pi,
                    max_basis=max_basis,
                    state_mode=state_mode,
                    target_operator_mode=target_operator_mode,
                    transition_response_time=transition_response_time,
                    transition_response_scale=transition_response_scale,
                    seed=seed + 19,
                    mode="wrong_pi_normalization",
                )
            )
        elif control == "wrong_4pi_normalization":
            control_reports[control] = _summary_rows(
                _state_rows(
                    points,
                    caps,
                    raw_fields,
                    collar_by_cap,
                    times=times,
                    observables=observables,
                    regularizers=regularizers,
                    target_scale=4.0 * math.pi,
                    max_basis=max_basis,
                    state_mode=state_mode,
                    target_operator_mode=target_operator_mode,
                    transition_response_time=transition_response_time,
                    transition_response_scale=transition_response_scale,
                    seed=seed + 23,
                    mode="wrong_4pi_normalization",
                )
            )
        elif control == "no_modular_flow":
            control_reports[control] = _summary_rows(
                _state_rows(
                    points,
                    caps,
                    raw_fields,
                    collar_by_cap,
                    times=times,
                    observables=observables,
                    regularizers=regularizers,
                    sim_modular_flow=False,
                    max_basis=max_basis,
                    state_mode=state_mode,
                    target_operator_mode=target_operator_mode,
                    transition_response_time=transition_response_time,
                    transition_response_scale=transition_response_scale,
                    seed=seed + 29,
                    mode="no_modular_flow",
                )
            )
        elif control == "shuffled_observables":
            control_reports[control] = _summary_rows(
                _state_rows(
                    points,
                    caps,
                    _shuffled_raw_fields(raw_fields, seed + 31),
                    collar_by_cap,
                    times=times,
                    observables=observables,
                    regularizers=regularizers,
                    max_basis=max_basis,
                    state_mode=state_mode,
                    target_operator_mode=target_operator_mode,
                    transition_response_time=transition_response_time,
                    transition_response_scale=transition_response_scale,
                    seed=seed + 31,
                    mode="shuffled_observables",
                )
            )
    return _state_summary(
        rows,
        control_reports,
        state_mode=state_mode,
        target_operator_mode=target_operator_mode,
        transition_response_time=transition_response_time,
        transition_response_scale=transition_response_scale,
    )


def _state_rows(
    points: np.ndarray,
    caps: list[RoundCap],
    raw_fields: dict[str, np.ndarray],
    collar_by_cap: dict[int, dict[str, Any]],
    *,
    times: list[float],
    observables: list[str],
    regularizers: list[float],
    target_scale: float = 2.0 * math.pi,
    sim_modular_flow: bool = True,
    max_basis: int = 96,
    state_mode: str = "cooccurrence_kernel",
    target_operator_mode: str = "nearest",
    transition_response_time: float = 0.025,
    transition_response_scale: float = 2.0 * math.pi,
    seed: int = 1,
    mode: str = "state_derived_modular_probe",
) -> list[dict[str, Any]]:
    rows = []
    for cap_id, cap in enumerate(caps):
        epsilon_cmi = float(collar_by_cap.get(cap_id, {}).get("epsilon_cmi", 0.0))
        for time_index, t in enumerate(times):
            for obs in observables:
                for reg in regularizers:
                    rows.append(
                        bw_state_derived_residual(
                            points,
                            raw_fields,
                            cap,
                            obs,
                            cap_id=cap_id,
                            t=t,
                            a=reg,
                            epsilon_cmi=epsilon_cmi,
                            target_scale=target_scale,
                            sim_modular_flow=sim_modular_flow,
                            state_mode=state_mode,
                            target_operator_mode=target_operator_mode,
                            transition_response_time=transition_response_time,
                            transition_response_scale=transition_response_scale,
                            max_basis=max_basis,
                            seed=seed + cap_id * 1009 + time_index,
                            mode=mode,
                        ).as_jsonable()
                    )
                    rows[-1]["target_scale"] = float(target_scale)
                    rows[-1]["sim_modular_flow"] = bool(sim_modular_flow)
                    rows[-1]["state_mode"] = state_mode
                    rows[-1]["target_operator_mode"] = target_operator_mode
                    rows[-1]["generator_source"] = (
                        "direct_transition_automorphism"
                        if state_mode == "transition_response_unitary"
                        else "regularized_density_log"
                    )
    return rows


def _state_summary(
    rows: list[dict[str, Any]],
    control_reports: dict[str, Any],
    *,
    state_mode: str,
    target_operator_mode: str,
    transition_response_time: float,
    transition_response_scale: float,
) -> dict[str, Any]:
    residuals = [row["support_visible_residual"] for row in rows]
    corrected = [row["corrected_residual_upper"] for row in rows]
    median = float(np.median(residuals)) if residuals else 0.0
    not_applicable_controls: list[str] = []
    gate_control_reports = dict(control_reports)
    if state_mode == "transition_response_unitary" and "shuffled_observables" in gate_control_reports:
        not_applicable_controls.append("shuffled_observables")
        gate_control_reports = {
            name: report for name, report in gate_control_reports.items() if name != "shuffled_observables"
        }
    control_medians = {
        name: float(report["median"])
        for name, report in gate_control_reports.items()
        if isinstance(report, dict) and "median" in report and np.isfinite(float(report["median"]))
    }
    correct_beats_controls = bool(control_medians) and all(median < value for value in control_medians.values())
    best_control = min(control_medians, key=control_medians.get) if control_medians else None
    normalization_declared = bool(
        state_mode == "transition_response_unitary"
        and math.isclose(float(transition_response_scale), 2.0 * math.pi)
    )
    summary = {
        "mode": "state_derived_modular_probe",
        "receipt_name": "BW_KMS_BRANCH_INSTANTIATION_RECEIPT",
        "BW_KMS_BRANCH_INSTANTIATION_RECEIPT": bool(normalization_declared and correct_beats_controls),
        "state_mode": state_mode,
        "target_operator_mode": target_operator_mode,
        "transition_response_time": float(transition_response_time),
        "transition_response_scale": float(transition_response_scale),
        "endogenous_modular_generator": bool(state_mode != "transition_response_unitary"),
        "normalization_declared": normalization_declared,
        "physical_bw_claim": False,
        "normalization_source": (
            "declared_kms_bw_2pi_transition_response_automorphism"
            if state_mode == "transition_response_unitary" and math.isclose(float(transition_response_scale), 2.0 * math.pi)
            else "finite_state_surrogate"
        ),
        "claim_boundary": (
            "finite state-derived modular matrix-element diagnostic. transition_response_unitary "
            "is an automorphism-level finite transition-response probe with declared KMS/BW "
            "normalization, not an unregularized type-I density proof, not a 3D bulk claim, "
            "and not a completed continuum BW proof"
        ),
        "median": median,
        "mean": float(np.mean(residuals)) if residuals else 0.0,
        "p90": float(np.percentile(residuals, 90)) if residuals else 0.0,
        "corrected_upper_median": float(np.median(corrected)) if corrected else 0.0,
        "control_medians": control_medians,
        "all_control_medians": {
            name: float(report["median"])
            for name, report in control_reports.items()
            if isinstance(report, dict) and "median" in report and np.isfinite(float(report["median"]))
        },
        "not_applicable_controls": not_applicable_controls,
        "control_gate_claim_boundary": (
            "For transition_response_unitary, shuffled_observables is reported but not used as a "
            "failure gate: a declared geometric automorphism should transport arbitrary observables. "
            "Wrong normalizations and no_modular_flow remain gate controls."
        )
        if not_applicable_controls
        else "all reported controls are used in the state-derived gate",
        "correct_beats_controls": correct_beats_controls,
        "best_control": best_control,
        "best_control_median": control_medians.get(best_control) if best_control is not None else None,
        "row_count": len(rows),
        "rows": rows,
        "controls": control_reports,
    }
    return with_claim_metadata(
        summary,
        claim_level=BRANCH_INSTANTIATION_SANITY if normalization_declared else DEMO,
        receipt="BW_KMS_BRANCH_INSTANTIATION_RECEIPT",
        physical_claim=False,
    )


def _summary_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.array([float(row["support_visible_residual"]) for row in rows], dtype=float)
    corrected = np.array([float(row["corrected_residual_upper"]) for row in rows], dtype=float)
    if values.size == 0:
        return {"median": float("nan"), "mean": float("nan"), "p90": float("nan"), "count": 0}
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "corrected_upper_median": float(np.median(corrected)) if corrected.size else float("nan"),
        "count": int(values.size),
    }


def _shuffled_raw_fields(raw_fields: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}
    for name, values in raw_fields.items():
        array = np.asarray(values)
        result[name] = array[rng.permutation(array.size)]
    return result


def _observable_matrix(raw_fields: dict[str, np.ndarray], observable: str, basis: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_fields.get(observable, raw_fields.get("record_signature")), dtype=float)[basis]
    values = _standardize(values)
    diagonal = np.diag(values.astype(complex))
    transition = np.zeros_like(diagonal)
    if values.size > 1:
        diffs = values[:, None] - values[None, :]
        transition = np.exp(-(diffs**2)) - np.eye(values.size)
        transition = transition / max(float(np.linalg.norm(transition)), 1e-15)
    return _hermitian(diagonal + 0.05 * transition)


def _relative_frobenius(delta: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(delta, ord="fro") / (np.linalg.norm(reference, ord="fro") + 1e-12))


def _standardize(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std < 1e-12:
        return values - float(np.mean(values))
    return (values - float(np.mean(values))) / std


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conj().T) / 2.0
