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


DECLARED_CAP_FLOW_STATE_MODES = frozenset({"cap_flow_graph_generator", "cap_flow_detailed_balance_kernel"})
DIRECT_AUTOMORPHISM_STATE_MODES = frozenset({"transition_response_unitary"})


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


def state_derived_modular_transport(
    O: np.ndarray,
    rho: np.ndarray,
    t: float,
    a: float,
    *,
    generator_scale: float = 1.0,
) -> np.ndarray:
    K = regularized_modular_generator(rho, a)
    K = float(generator_scale) * K
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
    history_fields: list[dict[str, np.ndarray]] | None = None,
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
    if state_mode in {"history_transition_kernel", "transition_count_kernel", "record_history_kernel"}:
        rho = history_transition_density(
            points,
            cap,
            support,
            raw_fields,
            history_fields or [],
            regularizer=regularizer,
        )
        return rho, support, packets
    if state_mode in DECLARED_CAP_FLOW_STATE_MODES:
        rho = cap_flow_graph_density(
            points,
            cap,
            support,
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


def history_transition_density(
    points: np.ndarray,
    cap: RoundCap,
    basis: np.ndarray,
    raw_fields: dict[str, np.ndarray],
    history_fields: list[dict[str, np.ndarray]],
    *,
    regularizer: float,
) -> np.ndarray:
    """Build an endogenous cap state from observer-visible transition histories.

    This finite surrogate keeps the basis on screen nodes but lets the density
    matrix see packet trajectories and local field changes. It deliberately does
    not call `lambda_cap`; the geometric BW map remains only the comparison
    target in `bw_state_derived_residual`.
    """

    basis = np.asarray(basis, dtype=np.int64)
    cap = cap.normalized()
    states = [state for state in [*list(history_fields or []), raw_fields] if state]
    if not states:
        states = [raw_fields]
    features: list[np.ndarray] = []
    packets = [visible_packets(state)[basis] for state in states if _field_size(state) > int(np.max(basis, initial=-1))]
    if packets:
        packet_seq = np.vstack(packets).astype(np.int64)
        features.append(_packet_histogram_features(packet_seq, bins=32))
        if packet_seq.shape[0] > 1:
            transition_tokens = (
                (np.mod(packet_seq[:-1], 32) * 32)
                + np.mod(packet_seq[1:], 32)
            )
            features.append(_packet_histogram_features(transition_tokens, bins=64))
            features.append(_standardize_columns((packet_seq[1:] != packet_seq[:-1]).T.astype(float)))
    for key in (
        "record_signature",
        "stable_count",
        "committed_mask",
        "repair_load",
        "cumulative_repair_load",
        "local_mismatch_density",
        "s3_class_density",
        "s3_sector_class",
    ):
        series = _history_field_series(states, key, basis)
        if series.size == 0:
            continue
        features.append(_standardize_columns(series.T))
        if series.shape[0] > 1:
            features.append(_standardize_columns(np.diff(series, axis=0).T))
    tangent = cap.tangent / max(float(np.linalg.norm(cap.tangent)), 1e-12)
    bitangent = np.cross(cap.axis, tangent)
    bitangent = bitangent / max(float(np.linalg.norm(bitangent)), 1e-12)
    cap_coords = np.column_stack(
        [
            points[basis] @ cap.axis,
            points[basis] @ tangent,
            points[basis] @ bitangent,
            cap_weights(points[basis], cap, soft=True),
        ]
    )
    features.append(0.15 * _standardize_columns(cap_coords))
    feature_matrix = np.hstack([feature for feature in features if feature.size])
    if feature_matrix.size == 0:
        feature_matrix = np.eye(basis.size, dtype=float)
    feature_matrix = _standardize_columns(feature_matrix)
    gram = feature_matrix @ feature_matrix.T / max(float(feature_matrix.shape[1]), 1.0)
    rho = gram @ gram.T
    rho = _hermitian(rho + float(regularizer) * np.eye(rho.shape[0], dtype=complex))
    rho = rho / max(float(np.trace(rho).real), 1e-15)
    return rho


def cap_flow_graph_density(
    points: np.ndarray,
    cap: RoundCap,
    basis: np.ndarray,
    *,
    regularizer: float,
    flow_epsilon: float = 1.0e-4,
    graph_k: int = 6,
) -> np.ndarray:
    """Finite support-visible cap-flow state from a local graph generator.

    This is a declared finite-regulator branch: it uses the cap chart to build
    an infinitesimal support-visible flow on the selected screen cells, then
    stores that Hermitian generator as a positive density matrix. It is not an
    unconstrained record-history discovery of BW flow.
    """

    basis = np.asarray(basis, dtype=np.int64)
    if basis.size == 0:
        return np.zeros((0, 0), dtype=complex)
    x = np.asarray(points[basis], dtype=float)
    if basis.size == 1:
        return np.ones((1, 1), dtype=complex)
    eps = max(float(flow_epsilon), 1.0e-8)
    mapped = lambda_cap(x, cap, eps)
    velocity = (mapped - x) / eps
    velocity = velocity - np.sum(velocity * x, axis=1, keepdims=True) * x
    distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    k = min(max(1, int(graph_k)), max(1, basis.size - 1))
    neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    antisym = np.zeros((basis.size, basis.size), dtype=float)
    for i in range(basis.size):
        for j in neighbors[i]:
            j = int(j)
            if not np.isfinite(distances[i, j]) or distances[i, j] <= 1.0e-12:
                continue
            midpoint = x[i] + x[j]
            midpoint_norm = float(np.linalg.norm(midpoint))
            if midpoint_norm > 1.0e-12:
                midpoint = midpoint / midpoint_norm
            edge = x[j] - x[i]
            edge = edge - float(np.dot(edge, midpoint)) * midpoint
            edge_norm = float(np.linalg.norm(edge))
            if edge_norm <= 1.0e-12:
                continue
            edge = edge / edge_norm
            flow = 0.5 * (velocity[i] + velocity[j])
            score = float(np.dot(flow, edge)) / max(float(distances[i, j]), 1.0e-6)
            antisym[i, j] += score
            antisym[j, i] -= score
    row_scale = float(np.median(np.sum(np.abs(antisym), axis=1)))
    if row_scale > 1.0e-12:
        antisym = antisym / row_scale
    generator = _hermitian(1j * antisym)
    eigvals = np.linalg.eigvalsh(generator)
    shifted = generator - float(np.min(eigvals).real) * np.eye(generator.shape[0], dtype=complex)
    rho = expm(-shifted)
    rho = _hermitian(rho + float(regularizer) * np.eye(rho.shape[0], dtype=complex))
    rho = rho / max(float(np.trace(rho).real), 1.0e-15)
    return rho


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
    generator_scale: float = 1.0,
    history_fields: list[dict[str, np.ndarray]] | None = None,
    max_basis: int = 96,
    seed: int = 1,
) -> ModularProbeReport:
    rho, basis, _ = cap_state_density(
        points,
        cap,
        raw_fields,
        history_fields=history_fields,
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
        O_sim = (
            state_derived_modular_transport(O, rho, t, a, generator_scale=generator_scale)
            if sim_modular_flow
            else O
        )
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
    report = ModularProbeReport(
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
    return report


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
    generator_scale: float = 1.0,
    history_fields: list[dict[str, np.ndarray]] | None = None,
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
        generator_scale=generator_scale,
        history_fields=history_fields,
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
                    generator_scale=generator_scale,
                    history_fields=history_fields,
                    seed=seed,
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
                    generator_scale=generator_scale,
                    history_fields=history_fields,
                    seed=seed,
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
                    generator_scale=generator_scale,
                    history_fields=history_fields,
                    seed=seed,
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
                    generator_scale=generator_scale,
                    history_fields=history_fields,
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
                    generator_scale=generator_scale,
                    history_fields=history_fields,
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
        generator_scale=generator_scale,
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
    generator_scale: float = 1.0,
    history_fields: list[dict[str, np.ndarray]] | None = None,
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
                            generator_scale=generator_scale,
                            history_fields=history_fields,
                            max_basis=max_basis,
                            seed=seed + cap_id * 1009 + time_index,
                            mode=mode,
                        ).as_jsonable()
                    )
                    rows[-1]["target_scale"] = float(target_scale)
                    rows[-1]["sim_modular_flow"] = bool(sim_modular_flow)
                    rows[-1]["state_mode"] = state_mode
                    rows[-1]["target_operator_mode"] = target_operator_mode
                    rows[-1]["generator_scale"] = float(generator_scale)
                    if state_mode in DIRECT_AUTOMORPHISM_STATE_MODES:
                        rows[-1]["generator_source"] = "direct_transition_automorphism"
                    elif state_mode in DECLARED_CAP_FLOW_STATE_MODES:
                        rows[-1]["generator_source"] = "declared_cap_flow_graph_generator"
                    else:
                        rows[-1]["generator_source"] = "regularized_density_log"
    return rows


def _state_summary(
    rows: list[dict[str, Any]],
    control_reports: dict[str, Any],
    *,
    state_mode: str,
    target_operator_mode: str,
    transition_response_time: float,
    transition_response_scale: float,
    generator_scale: float,
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
    candidate_medians = {"2pi": median}
    candidate_medians.update(
        {
            _control_scale_label(name): value
            for name, value in control_medians.items()
            if name.startswith("wrong_") and name.endswith("_normalization")
        }
    )
    selected_scale_label = min(candidate_medians, key=candidate_medians.get) if candidate_medians else None
    correct_beats_controls = bool(control_medians) and all(median < value for value in control_medians.values())
    best_control = min(control_medians, key=control_medians.get) if control_medians else None
    direct_automorphism = state_mode in DIRECT_AUTOMORPHISM_STATE_MODES
    declared_cap_flow = state_mode in DECLARED_CAP_FLOW_STATE_MODES
    endogenous_generator = bool(not direct_automorphism and not declared_cap_flow)
    normalization_declared = bool(direct_automorphism and math.isclose(float(transition_response_scale), 2.0 * math.pi))
    summary = {
        "mode": "state_derived_modular_probe",
        "receipt_name": "BW_KMS_BRANCH_INSTANTIATION_RECEIPT",
        "BW_KMS_BRANCH_INSTANTIATION_RECEIPT": bool(normalization_declared and correct_beats_controls),
        "state_mode": state_mode,
        "target_operator_mode": target_operator_mode,
        "transition_response_time": float(transition_response_time),
        "transition_response_scale": float(transition_response_scale),
        "generator_scale": float(generator_scale),
        "endogenous_modular_generator": endogenous_generator,
        "declared_cap_flow_generator": declared_cap_flow,
        "direct_transition_automorphism": direct_automorphism,
        "normalization_declared": normalization_declared,
        "physical_bw_claim": False,
        "normalization_source": (
            "declared_kms_bw_2pi_transition_response_automorphism"
            if direct_automorphism and math.isclose(float(transition_response_scale), 2.0 * math.pi)
            else "declared_cap_flow_graph_generator_scaled"
            if declared_cap_flow
            else "bw_normalized_finite_state_generator"
            if math.isclose(float(generator_scale), 2.0 * math.pi)
            else "finite_state_surrogate"
        ),
        "claim_boundary": (
            "finite state-derived modular matrix-element diagnostic. transition_response_unitary "
            "is an automorphism-level finite transition-response probe with declared KMS/BW "
            "normalization. cap_flow_graph_generator is a declared finite cap-flow surrogate, "
            "not an endogenous observer-record modular generator. None of these lanes is an "
            "unregularized type-I density proof, a 3D bulk claim, or a completed continuum BW proof"
        ),
        "median": median,
        "mean": float(np.mean(residuals)) if residuals else 0.0,
        "p90": float(np.percentile(residuals, 90)) if residuals else 0.0,
        "corrected_upper_median": float(np.median(corrected)) if corrected else 0.0,
        "control_medians": control_medians,
        "target_scale_candidate_medians": candidate_medians,
        "state_selected_scale_label": selected_scale_label,
        "state_selected_2pi": bool(selected_scale_label == "2pi"),
        "scale_controls_same_basis": True,
        "state_scale_selection_claim_boundary": (
            "diagnostic only: compares the endogenous finite-state modular transport against declared "
            "target normalizations. Declared cap-flow surrogates are reported separately from endogenous "
            "observer-record generators. A non-2pi selection means the finite state surrogate has not "
            "reached the paper BW normalization surface."
        ),
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


def _control_scale_label(control_name: str) -> str:
    text = str(control_name)
    if text.startswith("wrong_") and text.endswith("_normalization"):
        return text[len("wrong_") : -len("_normalization")]
    return text


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


def _field_size(state: dict[str, np.ndarray]) -> int:
    for values in state.values():
        array = np.asarray(values)
        if array.ndim >= 1 and array.size:
            return int(array.shape[0])
    return 0


def _history_field_series(states: list[dict[str, np.ndarray]], key: str, basis: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    max_index = int(np.max(basis, initial=-1))
    for state in states:
        if key not in state:
            continue
        values = np.asarray(state[key], dtype=float)
        if values.ndim != 1 or values.size <= max_index:
            continue
        rows.append(values[basis])
    return np.vstack(rows) if rows else np.zeros((0, basis.size), dtype=float)


def _packet_histogram_features(packet_sequence: np.ndarray, *, bins: int) -> np.ndarray:
    sequence = np.asarray(packet_sequence, dtype=np.int64)
    if sequence.ndim != 2 or sequence.size == 0:
        return np.zeros((0, 0), dtype=float)
    bins = max(1, int(bins))
    features = np.zeros((sequence.shape[1], bins), dtype=float)
    for column in range(sequence.shape[1]):
        counts = np.bincount(np.mod(np.abs(sequence[:, column]), bins), minlength=bins).astype(float)
        features[column] = counts / max(float(np.sum(counts)), 1.0)
    return _standardize_columns(features)


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    std = np.std(values, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (values - np.mean(values, axis=0)) / std


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
