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
DECLARED_RESPONSE_DENSITY_STATE_MODES = frozenset({"transition_response_density_log"})
REPAIR_RESPONSE_DENSITY_STATE_MODES = frozenset({"repair_affinity_response_density_log"})
PERTURB_RESPONSE_DENSITY_STATE_MODES = frozenset({"perturb_remeasure_response_density_log"})
PERTURB_RESPONSE_KERNEL_STATE_MODES = frozenset({"perturb_remeasure_response_kernel_log"})
COLLAR_OPERATOR_STATE_MODES = frozenset({"collar_operator_system", "support_visible_collar_operator_system"})


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
    density_inverse_temperature: float = 1.0,
    graph_response: dict[str, Any] | None = None,
    probe_steps: int = 4,
    probe_repairs_per_source: int = 64,
    probe_max_incident_edges: int = 8,
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
            density_inverse_temperature=density_inverse_temperature,
        )
        return rho, support, packets
    if state_mode in DECLARED_RESPONSE_DENSITY_STATE_MODES:
        rho = transition_response_density(
            points,
            cap,
            support,
            response_time=transition_response_time,
            response_scale=transition_response_scale,
            regularizer=regularizer,
            density_inverse_temperature=density_inverse_temperature,
        )
        return rho, support, packets
    if state_mode in REPAIR_RESPONSE_DENSITY_STATE_MODES:
        rho = repair_affinity_response_density(
            points,
            cap,
            support,
            raw_fields,
            response_time=transition_response_time,
            regularizer=regularizer,
            density_inverse_temperature=density_inverse_temperature,
        )
        return rho, support, packets
    if state_mode in PERTURB_RESPONSE_DENSITY_STATE_MODES:
        if graph_response is None:
            raise ValueError("perturb_remeasure_response_density_log requires graph_response data")
        rho = perturb_remeasure_response_density(
            points,
            cap,
            support,
            raw_fields,
            graph_response=graph_response,
            response_time=transition_response_time,
            regularizer=regularizer,
            density_inverse_temperature=density_inverse_temperature,
            seed=seed,
            probe_steps=probe_steps,
            probe_repairs_per_source=probe_repairs_per_source,
            probe_max_incident_edges=probe_max_incident_edges,
        )
        return rho, support, packets
    if state_mode in PERTURB_RESPONSE_KERNEL_STATE_MODES:
        if graph_response is None:
            raise ValueError("perturb_remeasure_response_kernel_log requires graph_response data")
        rho = perturb_remeasure_response_kernel_density(
            points,
            cap,
            support,
            raw_fields,
            graph_response=graph_response,
            regularizer=regularizer,
            seed=seed,
            probe_steps=probe_steps,
            probe_repairs_per_source=probe_repairs_per_source,
            probe_max_incident_edges=probe_max_incident_edges,
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
    if state_mode in COLLAR_OPERATOR_STATE_MODES:
        rho = collar_operator_system_density(
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
            density_inverse_temperature=density_inverse_temperature,
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


def collar_operator_system_density(
    points: np.ndarray,
    cap: RoundCap,
    basis: np.ndarray,
    raw_fields: dict[str, np.ndarray],
    history_fields: list[dict[str, np.ndarray]],
    *,
    regularizer: float,
    flow_weight: float = 0.35,
    collar_weight: float = 0.25,
) -> np.ndarray:
    """Build a sparse finite cap/collar operator-system density.

    The rows/columns are support-visible screen nodes. The density is generated
    from observer-visible history features plus noncommuting collar-flow links.
    The geometric BW map is still held out as the comparison target in
    `bw_state_derived_residual`; this state only sees the cap/collar split,
    local visible packets, and the infinitesimal collar flow direction.
    """

    basis = np.asarray(basis, dtype=np.int64)
    if basis.size == 0:
        return np.zeros((0, 0), dtype=complex)
    if basis.size == 1:
        return np.ones((1, 1), dtype=complex)
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
            transition_tokens = (np.mod(packet_seq[:-1], 32) * 32) + np.mod(packet_seq[1:], 32)
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
    cap_distance = points[basis] @ cap.axis - float(np.cos(cap.theta0))
    partition = cap_collar_partition(points, cap, cap.collar_width)
    inside = partition.inside_mask[basis].astype(float)
    collar = partition.collar_mask[basis].astype(float)
    outside = partition.outside_mask[basis].astype(float)
    cap_coords = np.column_stack(
        [
            points[basis] @ cap.axis,
            points[basis] @ tangent,
            points[basis] @ bitangent,
            cap_weights(points[basis], cap, soft=True),
            cap_distance,
            inside,
            collar,
            outside,
        ]
    )
    features.append(0.2 * _standardize_columns(cap_coords))
    feature_matrix = np.hstack([feature for feature in features if feature.size])
    if feature_matrix.size == 0:
        feature_matrix = np.eye(basis.size, dtype=float)
    feature_matrix = _standardize_columns(feature_matrix)
    gram = feature_matrix @ feature_matrix.T / max(float(feature_matrix.shape[1]), 1.0)
    antisym = _cap_flow_antisymmetric(points, cap, basis)
    collar_links = _collar_link_matrix(points, basis, partition)
    operator = gram.astype(complex) + 1j * float(flow_weight) * antisym + float(collar_weight) * collar_links
    rho = operator @ operator.conj().T
    rho = _hermitian(rho + float(regularizer) * np.eye(rho.shape[0], dtype=complex))
    trace = max(float(np.trace(rho).real), 1.0e-15)
    return rho / trace


def cap_flow_graph_density(
    points: np.ndarray,
    cap: RoundCap,
    basis: np.ndarray,
    *,
    regularizer: float,
    density_inverse_temperature: float = 1.0,
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
    # Preserve the cap-flow derivative scale here. The BW target fixes
    # s = 2*pi*t, so an arbitrary row-norm rescaling would silently change the
    # finite generator's modular-time normalization. The stabilized helper
    # `_cap_flow_antisymmetric` is still used for endogenous collar diagnostics,
    # but the declared cap-flow graph state keeps the derivative scale visible.
    generator = _hermitian(1j * antisym)
    eigvals = np.linalg.eigvalsh(generator)
    shifted = generator - float(np.min(eigvals).real) * np.eye(generator.shape[0], dtype=complex)
    beta = max(float(density_inverse_temperature), 1.0e-12)
    rho = expm(-beta * shifted)
    rho = _hermitian(rho + float(regularizer) * np.eye(rho.shape[0], dtype=complex))
    rho = rho / max(float(np.trace(rho).real), 1.0e-15)
    return rho


def _cap_flow_antisymmetric(
    points: np.ndarray,
    cap: RoundCap,
    basis: np.ndarray,
    *,
    flow_epsilon: float = 1.0e-4,
    graph_k: int = 6,
) -> np.ndarray:
    basis = np.asarray(basis, dtype=np.int64)
    x = np.asarray(points[basis], dtype=float)
    if basis.size <= 1:
        return np.zeros((basis.size, basis.size), dtype=float)
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
        for j_raw in neighbors[i]:
            j = int(j_raw)
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
    scale = float(np.median(np.sum(np.abs(antisym), axis=1)))
    if scale > 1.0e-12:
        antisym = antisym / scale
    return antisym


def _collar_link_matrix(points: np.ndarray, basis: np.ndarray, partition: Any, *, graph_k: int = 4) -> np.ndarray:
    basis = np.asarray(basis, dtype=np.int64)
    if basis.size <= 1:
        return np.zeros((basis.size, basis.size), dtype=complex)
    x = np.asarray(points[basis], dtype=float)
    inside = partition.inside_mask[basis]
    collar = partition.collar_mask[basis]
    outside = partition.outside_mask[basis]
    distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    k = min(max(1, int(graph_k)), max(1, basis.size - 1))
    neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    links = np.zeros((basis.size, basis.size), dtype=complex)
    for i in range(basis.size):
        for j_raw in neighbors[i]:
            j = int(j_raw)
            if not np.isfinite(distances[i, j]):
                continue
            crosses_collar = bool(collar[i] or collar[j] or (inside[i] and outside[j]) or (outside[i] and inside[j]))
            if not crosses_collar:
                continue
            weight = math.exp(-float(distances[i, j]))
            links[i, j] += weight
            links[j, i] += weight
    norm = float(np.linalg.norm(links, ord="fro"))
    if norm > 1.0e-12:
        links = links / norm
    return _hermitian(links)


def transition_response_density(
    points: np.ndarray,
    cap: RoundCap,
    basis: np.ndarray,
    *,
    response_time: float,
    response_scale: float,
    regularizer: float,
    density_inverse_temperature: float = 1.0,
) -> np.ndarray:
    transition, _ = geometric_permutation_operator(points, cap, float(response_scale) * float(response_time), basis)
    K = modular_generator_from_unitary_transition(transition.conj().T, response_time)
    eigvals = np.linalg.eigvalsh(K)
    shifted = K - float(np.min(eigvals)) * np.eye(K.shape[0], dtype=complex)
    beta = max(float(density_inverse_temperature), 1.0e-12)
    rho = expm(-beta * shifted)
    rho = _hermitian(rho + float(regularizer) * np.eye(rho.shape[0], dtype=complex))
    rho = rho / max(float(np.trace(rho).real), 1e-15)
    return rho


def repair_affinity_response_density(
    points: np.ndarray,
    cap: RoundCap,
    basis: np.ndarray,
    raw_fields: dict[str, np.ndarray],
    *,
    response_time: float,
    regularizer: float,
    density_inverse_temperature: float = 1.0,
) -> np.ndarray:
    """Build a density-log modular surrogate from observer-visible repair affinity.

    This is the non-declared counterpart to `transition_response_density`: it
    estimates a cap-local support-visible transition from record, repair,
    sector, and collar fields, then stores the corresponding finite generator
    as a Gibbs-like density. It deliberately does not call `lambda_cap`; the
    geometric cap flow is reserved for the residual target and controls.
    """

    pullback, _eta, _meta = _repair_affinity_pullback(points, cap, raw_fields, basis)
    K = modular_generator_from_unitary_transition(pullback, response_time)
    eigvals = np.linalg.eigvalsh(K)
    shifted = K - float(np.min(eigvals)) * np.eye(K.shape[0], dtype=complex)
    beta = max(float(density_inverse_temperature), 1.0e-12)
    rho = expm(-beta * shifted)
    rho = _hermitian(rho + float(regularizer) * np.eye(rho.shape[0], dtype=complex))
    rho = rho / max(float(np.trace(rho).real), 1e-15)
    return rho


def perturb_remeasure_response_density(
    points: np.ndarray,
    cap: RoundCap,
    basis: np.ndarray,
    raw_fields: dict[str, np.ndarray],
    *,
    graph_response: dict[str, Any],
    response_time: float,
    regularizer: float,
    density_inverse_temperature: float = 1.0,
    seed: int = 1,
    probe_steps: int = 4,
    probe_repairs_per_source: int = 64,
    probe_max_incident_edges: int = 8,
) -> np.ndarray:
    """Build rho_C from bounded support-visible perturb/repair/remeasure response.

    This is an endogenous finite-regulator surrogate: the transition is inferred
    from local port perturbations, declared repair scoring, and packet
    remeasurement on the actual screen graph. The geometric cap flow remains
    only the held-out BW comparison target.
    """

    pullback, _eta, _meta = _perturb_remeasure_pullback(
        points,
        cap,
        raw_fields,
        basis,
        graph_response=graph_response,
        seed=seed,
        probe_steps=probe_steps,
        probe_repairs_per_source=probe_repairs_per_source,
        probe_max_incident_edges=probe_max_incident_edges,
    )
    K = modular_generator_from_unitary_transition(pullback, response_time)
    eigvals = np.linalg.eigvalsh(K)
    shifted = K - float(np.min(eigvals)) * np.eye(K.shape[0], dtype=complex)
    beta = max(float(density_inverse_temperature), 1.0e-12)
    rho = expm(-beta * shifted)
    rho = _hermitian(rho + float(regularizer) * np.eye(rho.shape[0], dtype=complex))
    rho = rho / max(float(np.trace(rho).real), 1e-15)
    return rho


def perturb_remeasure_response_kernel_density(
    points: np.ndarray,
    cap: RoundCap,
    basis: np.ndarray,
    raw_fields: dict[str, np.ndarray],
    *,
    graph_response: dict[str, Any],
    regularizer: float,
    seed: int = 1,
    probe_steps: int = 4,
    probe_repairs_per_source: int = 64,
    probe_max_incident_edges: int = 8,
) -> np.ndarray:
    """Build rho_C directly from perturb/repair/remeasure response counts.

    This follows the finite sparse-block surrogate idea more closely than the
    permutation lane: measured transition counts form M, then rho_C is
    regularized from M M^T. No geometric cap map is used in the state.
    """

    response_matrix, _meta = _perturb_remeasure_response_matrix(
        points,
        raw_fields,
        basis,
        graph_response=graph_response,
        seed=seed,
        probe_steps=probe_steps,
        probe_repairs_per_source=probe_repairs_per_source,
        probe_max_incident_edges=probe_max_incident_edges,
    )
    kernel = response_matrix / np.maximum(np.sum(response_matrix, axis=1, keepdims=True), 1.0e-12)
    features = [_standardize_rows(response_matrix), _standardize_columns(kernel)]
    basis_points = points[np.asarray(basis, dtype=np.int64)]
    spatial = np.linalg.norm(basis_points[:, None, :] - basis_points[None, :, :], axis=2)
    spatial_scale = _positive_median(spatial)
    proximity = np.exp(-np.square(spatial / spatial_scale))
    features.append(0.1 * _standardize_columns(proximity))
    M = np.hstack([feature for feature in features if feature.size])
    if M.size == 0:
        M = np.eye(np.asarray(basis).size, dtype=float)
    rho = M @ M.T
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
    density_inverse_temperature: float = 1.0,
    generator_scale: float = 1.0,
    history_fields: list[dict[str, np.ndarray]] | None = None,
    graph_response: dict[str, Any] | None = None,
    probe_steps: int = 4,
    probe_repairs_per_source: int = 64,
    probe_max_incident_edges: int = 8,
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
        density_inverse_temperature=density_inverse_temperature,
        graph_response=graph_response,
        probe_steps=probe_steps,
        probe_repairs_per_source=probe_repairs_per_source,
        probe_max_incident_edges=probe_max_incident_edges,
    )
    O = _observable_matrix(raw_fields, observable, basis, points=points, cap=cap)
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
    density_inverse_temperature: float = 1.0,
    generator_scale: float = 1.0,
    generator_scale_candidates: list[float] | None = None,
    history_fields: list[dict[str, np.ndarray]] | None = None,
    graph_response: dict[str, Any] | None = None,
    probe_steps: int = 4,
    probe_repairs_per_source: int = 64,
    probe_max_incident_edges: int = 8,
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
        density_inverse_temperature=density_inverse_temperature,
        generator_scale=generator_scale,
        history_fields=history_fields,
        graph_response=graph_response,
        probe_steps=probe_steps,
        probe_repairs_per_source=probe_repairs_per_source,
        probe_max_incident_edges=probe_max_incident_edges,
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
                    density_inverse_temperature=density_inverse_temperature,
                    generator_scale=generator_scale,
                    history_fields=history_fields,
                    graph_response=graph_response,
                    probe_steps=probe_steps,
                    probe_repairs_per_source=probe_repairs_per_source,
                    probe_max_incident_edges=probe_max_incident_edges,
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
                    density_inverse_temperature=density_inverse_temperature,
                    generator_scale=generator_scale,
                    history_fields=history_fields,
                    graph_response=graph_response,
                    probe_steps=probe_steps,
                    probe_repairs_per_source=probe_repairs_per_source,
                    probe_max_incident_edges=probe_max_incident_edges,
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
                    density_inverse_temperature=density_inverse_temperature,
                    generator_scale=generator_scale,
                    history_fields=history_fields,
                    graph_response=graph_response,
                    probe_steps=probe_steps,
                    probe_repairs_per_source=probe_repairs_per_source,
                    probe_max_incident_edges=probe_max_incident_edges,
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
                    density_inverse_temperature=density_inverse_temperature,
                    generator_scale=generator_scale,
                    history_fields=history_fields,
                    graph_response=graph_response,
                    probe_steps=probe_steps,
                    probe_repairs_per_source=probe_repairs_per_source,
                    probe_max_incident_edges=probe_max_incident_edges,
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
                    density_inverse_temperature=density_inverse_temperature,
                    generator_scale=generator_scale,
                    history_fields=history_fields,
                    graph_response=graph_response,
                    probe_steps=probe_steps,
                    probe_repairs_per_source=probe_repairs_per_source,
                    probe_max_incident_edges=probe_max_incident_edges,
                    seed=seed + 31,
                    mode="shuffled_observables",
                )
            )
    generator_scale_audit = _generator_scale_audit(
        points,
        caps,
        raw_fields,
        collar_by_cap,
        times=times,
        observables=observables,
        regularizers=regularizers,
        configured_generator_scale=generator_scale,
        generator_scale_candidates=generator_scale_candidates,
        max_basis=max_basis,
        state_mode=state_mode,
        target_operator_mode=target_operator_mode,
        transition_response_time=transition_response_time,
        transition_response_scale=transition_response_scale,
        density_inverse_temperature=density_inverse_temperature,
        history_fields=history_fields,
        graph_response=graph_response,
        probe_steps=probe_steps,
        probe_repairs_per_source=probe_repairs_per_source,
        probe_max_incident_edges=probe_max_incident_edges,
        seed=seed + 7919,
    )
    return _state_summary(
        rows,
        control_reports,
        state_mode=state_mode,
        target_operator_mode=target_operator_mode,
        transition_response_time=transition_response_time,
        transition_response_scale=transition_response_scale,
        density_inverse_temperature=density_inverse_temperature,
        generator_scale=generator_scale,
        generator_scale_audit=generator_scale_audit,
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
    density_inverse_temperature: float = 1.0,
    generator_scale: float = 1.0,
    history_fields: list[dict[str, np.ndarray]] | None = None,
    graph_response: dict[str, Any] | None = None,
    probe_steps: int = 4,
    probe_repairs_per_source: int = 64,
    probe_max_incident_edges: int = 8,
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
                            density_inverse_temperature=density_inverse_temperature,
                            generator_scale=generator_scale,
                            history_fields=history_fields,
                            graph_response=graph_response,
                            probe_steps=probe_steps,
                            probe_repairs_per_source=probe_repairs_per_source,
                            probe_max_incident_edges=probe_max_incident_edges,
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
                    rows[-1]["density_inverse_temperature"] = float(density_inverse_temperature)
                    if state_mode in DIRECT_AUTOMORPHISM_STATE_MODES:
                        rows[-1]["generator_source"] = "direct_transition_automorphism"
                    elif state_mode in DECLARED_RESPONSE_DENSITY_STATE_MODES:
                        rows[-1]["generator_source"] = "declared_transition_response_density_log"
                    elif state_mode in REPAIR_RESPONSE_DENSITY_STATE_MODES:
                        rows[-1]["generator_source"] = "repair_affinity_response_density_log"
                    elif state_mode in PERTURB_RESPONSE_DENSITY_STATE_MODES:
                        rows[-1]["generator_source"] = "perturb_remeasure_response_density_log"
                    elif state_mode in PERTURB_RESPONSE_KERNEL_STATE_MODES:
                        rows[-1]["generator_source"] = "perturb_remeasure_response_kernel_log"
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
    density_inverse_temperature: float,
    generator_scale: float,
    generator_scale_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    residuals = [row["support_visible_residual"] for row in rows]
    corrected = [row["corrected_residual_upper"] for row in rows]
    median = float(np.median(residuals)) if residuals else 0.0
    direct_automorphism = state_mode in DIRECT_AUTOMORPHISM_STATE_MODES
    declared_cap_flow = state_mode in DECLARED_CAP_FLOW_STATE_MODES
    declared_response_density = state_mode in DECLARED_RESPONSE_DENSITY_STATE_MODES
    repair_response_density = state_mode in REPAIR_RESPONSE_DENSITY_STATE_MODES
    perturb_response_density = state_mode in PERTURB_RESPONSE_DENSITY_STATE_MODES
    perturb_response_kernel = state_mode in PERTURB_RESPONSE_KERNEL_STATE_MODES
    endogenous_generator = bool(not direct_automorphism and not declared_cap_flow and not declared_response_density)
    not_applicable_controls: list[str] = []
    gate_control_reports = dict(control_reports)
    if (direct_automorphism or declared_response_density) and "shuffled_observables" in gate_control_reports:
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
    degenerate_scale_controls = [
        label
        for label, value in candidate_medians.items()
        if label != "2pi" and np.isfinite(float(value)) and float(value) <= 1.0e-10
    ]
    selected_scale_label = min(candidate_medians, key=candidate_medians.get) if candidate_medians else None
    correct_beats_controls = bool(control_medians) and all(median < value for value in control_medians.values())
    best_control = min(control_medians, key=control_medians.get) if control_medians else None
    normalization_declared = bool(direct_automorphism and math.isclose(float(transition_response_scale), 2.0 * math.pi))
    density_log_calibration_receipt = bool(
        declared_response_density
        and selected_scale_label == "2pi"
        and correct_beats_controls
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
        "density_inverse_temperature": float(density_inverse_temperature),
        "generator_scale": float(generator_scale),
        "endogenous_modular_generator": endogenous_generator,
        "declared_cap_flow_generator": declared_cap_flow,
        "declared_transition_response_density": declared_response_density,
        "repair_affinity_response_density": repair_response_density,
        "perturb_remeasure_response_density": perturb_response_density,
        "perturb_remeasure_response_kernel": perturb_response_kernel,
        "TRANSITION_RESPONSE_DENSITY_LOG_CALIBRATION_RECEIPT": density_log_calibration_receipt,
        "direct_transition_automorphism": direct_automorphism,
        "normalization_declared": normalization_declared,
        "physical_bw_claim": False,
        "normalization_source": (
            "declared_kms_bw_2pi_transition_response_automorphism"
            if direct_automorphism and math.isclose(float(transition_response_scale), 2.0 * math.pi)
            else "declared_transition_response_density_log"
            if declared_response_density
            else "declared_cap_flow_graph_generator_scaled"
            if declared_cap_flow
            else "endogenous_repair_affinity_response_density_log"
            if repair_response_density
            else "endogenous_perturb_remeasure_response_density_log"
            if perturb_response_density
            else "endogenous_perturb_remeasure_response_kernel_log"
            if perturb_response_kernel
            else "bw_normalized_finite_state_generator"
            if math.isclose(float(generator_scale), 2.0 * math.pi)
            else "finite_state_surrogate"
        ),
        "claim_boundary": (
            "finite state-derived modular matrix-element diagnostic. transition_response_unitary "
            "is an automorphism-level finite transition-response probe with declared KMS/BW "
            "normalization. cap_flow_graph_generator is a declared finite cap-flow surrogate, "
            "not an endogenous observer-record modular generator. transition_response_density_log "
            "checks whether the regularized density-log machinery can recover a declared transition "
            "state; it is a calibration lane, not an endogenous proof. repair_affinity_response_density_log "
            "is an observer-visible repair/collar-field transition surrogate, still short of a full "
            "cap/collar rho_C reconstruction. perturb_remeasure_response_density_log builds its transition "
            "from bounded support-visible port perturbations, declared repair, and packet remeasurement on "
            "the actual screen graph. perturb_remeasure_response_kernel_log uses the same measured response "
            "counts directly as M M^T instead of forcing a permutation assignment. These are current "
            "endogenous finite-regulator probes, not completed continuum BW proofs. None of these lanes is "
            "an unregularized type-I density proof, a 3D bulk claim, or a completed continuum BW proof"
        ),
        "median": median,
        "mean": float(np.mean(residuals)) if residuals else 0.0,
        "p90": float(np.percentile(residuals, 90)) if residuals else 0.0,
        "corrected_upper_median": float(np.median(corrected)) if corrected else 0.0,
        "control_medians": control_medians,
        "target_scale_candidate_medians": candidate_medians,
        "state_selected_scale_label": selected_scale_label,
        "state_selected_2pi": bool(selected_scale_label == "2pi"),
        "target_scale_control_degenerate": bool(degenerate_scale_controls),
        "degenerate_target_scale_controls": degenerate_scale_controls,
        "scale_controls_same_basis": True,
        "state_scale_selection_claim_boundary": (
            "diagnostic only: compares the endogenous finite-state modular transport against declared "
            "target normalizations. Declared cap-flow surrogates are reported separately from endogenous "
            "observer-record generators. A non-2pi selection means the finite state surrogate has not "
            "reached the paper BW normalization surface. Degenerate wrong-normalization controls indicate "
            "that the finite basis/time choice made a wrong target effectively identity and should not be "
            "used as positive evidence."
        ),
        "generator_scale_audit": generator_scale_audit or {
            "enabled": False,
            "claim_boundary": (
                "not run. Enable bw.generator_scale_candidates to test whether the finite "
                "K=-log(rho+aI) generator wants a different sign or scale against lambda_C(2*pi*t)."
            ),
        },
        "all_control_medians": {
            name: float(report["median"])
            for name, report in control_reports.items()
            if isinstance(report, dict) and "median" in report and np.isfinite(float(report["median"]))
        },
        "not_applicable_controls": not_applicable_controls,
        "control_gate_claim_boundary": (
            "For declared geometric transport lanes, shuffled_observables is reported but not used as "
            "a failure gate: a declared automorphism/density-log transport should covariantly transport "
            "arbitrary observables. Wrong normalizations and no_modular_flow remain gate controls."
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


def _generator_scale_audit(
    points: np.ndarray,
    caps: list[RoundCap],
    raw_fields: dict[str, np.ndarray],
    collar_by_cap: dict[int, dict[str, Any]],
    *,
    times: list[float],
    observables: list[str],
    regularizers: list[float],
    configured_generator_scale: float,
    generator_scale_candidates: list[float] | None,
    max_basis: int,
    state_mode: str,
    target_operator_mode: str,
    transition_response_time: float,
    transition_response_scale: float,
    density_inverse_temperature: float,
    history_fields: list[dict[str, np.ndarray]] | None,
    graph_response: dict[str, Any] | None,
    probe_steps: int,
    probe_repairs_per_source: int,
    probe_max_incident_edges: int,
    seed: int,
) -> dict[str, Any]:
    if state_mode in DIRECT_AUTOMORPHISM_STATE_MODES:
        return {
            "enabled": False,
            "not_applicable": True,
            "state_mode": state_mode,
            "claim_boundary": (
                "not run: direct transition-response automorphism lanes construct the finite "
                "unitary generator explicitly from the declared cap transition, so "
                "generator_scale is intentionally ignored. Wrong-normalization and "
                "no-modular-flow controls remain meaningful; a density-log generator-scale "
                "audit would be a false diagnostic for this lane."
            ),
        }
    if not generator_scale_candidates:
        return {
            "enabled": False,
            "claim_boundary": (
                "not run. Enable bw.generator_scale_candidates to test whether the finite "
                "K=-log(rho+aI) generator wants a different sign or scale against lambda_C(2*pi*t)."
            ),
        }
    candidates = _unique_floats([configured_generator_scale, *generator_scale_candidates])
    by_scale: dict[str, Any] = {}
    for index, scale in enumerate(candidates):
        label = _scale_label(scale)
        rows = _state_rows(
            points,
            caps,
            raw_fields,
            collar_by_cap,
            times=times,
            observables=observables,
            regularizers=regularizers,
            target_scale=2.0 * math.pi,
            sim_modular_flow=not math.isclose(float(scale), 0.0, abs_tol=1e-15),
            max_basis=max_basis,
            state_mode=state_mode,
            target_operator_mode=target_operator_mode,
            transition_response_time=transition_response_time,
            transition_response_scale=transition_response_scale,
            density_inverse_temperature=density_inverse_temperature,
            generator_scale=float(scale),
            history_fields=history_fields,
            graph_response=graph_response,
            probe_steps=probe_steps,
            probe_repairs_per_source=probe_repairs_per_source,
            probe_max_incident_edges=probe_max_incident_edges,
            seed=seed + index * 101,
            mode="generator_scale_audit",
        )
        summary = _summary_rows(rows)
        by_scale[label] = {
            "scale": float(scale),
            "median": float(summary["median"]),
            "mean": float(summary["mean"]),
            "p90": float(summary["p90"]),
            "corrected_upper_median": float(summary["corrected_upper_median"]),
            "count": int(summary["count"]),
        }
    finite = {
        label: row
        for label, row in by_scale.items()
        if np.isfinite(float(row.get("median", float("nan"))))
    }
    selected_label = min(finite, key=lambda label: float(finite[label]["median"])) if finite else None
    configured_label = _scale_label(configured_generator_scale)
    configured_score = by_scale.get(configured_label, {}).get("median")
    two_pi_label = _scale_label(2.0 * math.pi)
    two_pi_score = by_scale.get(two_pi_label, {}).get("median")
    best_score = by_scale.get(selected_label, {}).get("median") if selected_label is not None else None
    diagnosis = "not_run"
    if selected_label is not None:
        selected_scale = float(by_scale[selected_label]["scale"])
        if math.isclose(selected_scale, float(configured_generator_scale), rel_tol=1e-9, abs_tol=1e-12):
            diagnosis = "configured_scale_best"
        elif selected_scale < 0.0 and float(configured_generator_scale) > 0.0:
            diagnosis = "opposite_sign_best"
        elif math.isclose(selected_scale, 0.0, abs_tol=1e-15):
            diagnosis = "no_flow_best"
        else:
            diagnosis = "different_scale_best"
    return {
        "enabled": True,
        "mode": "generator_scale_audit",
        "target_scale_label": "2pi",
        "target_scale": float(2.0 * math.pi),
        "configured_scale_label": configured_label,
        "configured_scale": float(configured_generator_scale),
        "configured_score": configured_score,
        "two_pi_generator_score": two_pi_score,
        "best_label": selected_label,
        "best_scale": by_scale.get(selected_label, {}).get("scale") if selected_label is not None else None,
        "best_score": best_score,
        "configured_is_best": bool(selected_label == configured_label),
        "two_pi_generator_is_best": bool(selected_label == two_pi_label),
        "diagnosis": diagnosis,
        "by_scale": by_scale,
        "claim_boundary": (
            "diagnostic only: holds the geometric target fixed at lambda_C(2*pi*t) and scans the finite "
            "state-derived generator multiplier. A best non-2pi or opposite-sign multiplier is evidence "
            "that the current finite density-log surrogate is not yet on the paper BW/KMS normalization "
            "surface; it is not a license to tune the proof criterion."
        ),
    }


def _control_scale_label(control_name: str) -> str:
    text = str(control_name)
    if text.startswith("wrong_") and text.endswith("_normalization"):
        return text[len("wrong_") : -len("_normalization")]
    return text


def _scale_label(value: float) -> str:
    value = float(value)
    known = [
        (0.0, "0"),
        (1.0, "1x"),
        (math.pi, "pi"),
        (2.0 * math.pi, "2pi"),
        (4.0 * math.pi, "4pi"),
    ]
    sign = "minus_" if value < 0.0 else ""
    magnitude = abs(value)
    for scale, label in known:
        if math.isclose(magnitude, scale, rel_tol=1e-9, abs_tol=1e-12):
            return f"{sign}{label}" if sign and label != "0" else label
    return f"{value:.6g}"


def _unique_floats(values: list[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        candidate = float(value)
        if not any(math.isclose(candidate, existing, rel_tol=1e-12, abs_tol=1e-15) for existing in result):
            result.append(candidate)
    return result


def _repair_affinity_pullback(
    points: np.ndarray,
    cap: RoundCap,
    raw_fields: dict[str, np.ndarray],
    basis: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    basis = np.asarray(basis, dtype=np.int64)
    if basis.size == 0:
        return np.zeros((0, 0), dtype=complex), 0.0, {"response_source": "repair_affinity_response"}
    if basis.size == 1:
        return np.ones((1, 1), dtype=complex), 0.0, {
            "response_source": "repair_affinity_response",
            "identity_fraction": 1.0,
        }
    cap = cap.normalized()
    basis_points = points[basis]
    spatial = np.linalg.norm(basis_points[:, None, :] - basis_points[None, :, :], axis=2)
    spatial = spatial / _positive_median(spatial)

    features = _repair_affinity_features(points, cap, raw_fields, basis)
    if features.size:
        feature_delta = features[:, None, :] - features[None, :, :]
        feature_dist = np.sqrt(np.mean(feature_delta**2, axis=2))
    else:
        feature_dist = np.zeros_like(spatial)

    response_score = _repair_affinity_score(raw_fields, basis)
    descent_bonus = np.maximum(response_score[:, None] - response_score[None, :], 0.0)
    collar_distance = np.abs(points[basis] @ cap.axis - math.cos(cap.theta0)) / max(cap.collar_width, 1.0e-6)
    collar_preference = np.exp(-0.5 * np.square(collar_distance))
    collar_bonus = 0.5 * (collar_preference[:, None] + collar_preference[None, :])

    cost = 0.65 * spatial + 0.45 * feature_dist - 0.18 * descent_bonus - 0.04 * collar_bonus
    np.fill_diagonal(cost, np.diag(cost) + 0.02)
    row_ind, col_ind = linear_sum_assignment(cost)
    transition = np.zeros((basis.size, basis.size), dtype=complex)
    transition[row_ind, col_ind] = 1.0
    pullback = transition.conj().T
    assigned = np.argmax(np.abs(transition), axis=1)
    return pullback, 0.0, {
        "response_source": "repair_affinity_response",
        "identity_fraction": float(np.mean(assigned == np.arange(basis.size))),
        "mean_assignment_cost": float(np.mean(cost[row_ind, col_ind])) if cost.size else 0.0,
        "response_score_std": float(np.std(response_score)) if response_score.size else 0.0,
        "response_feature_count": int(features.shape[1]) if features.ndim == 2 else 0,
    }


def _perturb_remeasure_pullback(
    points: np.ndarray,
    cap: RoundCap,
    raw_fields: dict[str, np.ndarray],
    basis: np.ndarray,
    *,
    graph_response: dict[str, Any],
    seed: int,
    probe_steps: int,
    probe_repairs_per_source: int,
    probe_max_incident_edges: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    basis = np.asarray(basis, dtype=np.int64)
    if basis.size == 0:
        return np.zeros((0, 0), dtype=complex), 0.0, {"response_source": "perturb_remeasure_response"}
    if basis.size == 1:
        return np.ones((1, 1), dtype=complex), 0.0, {
            "response_source": "perturb_remeasure_response",
        "identity_fraction": 1.0,
    }

    response_matrix, meta = _perturb_remeasure_response_matrix(
        points,
        raw_fields,
        basis,
        graph_response=graph_response,
        seed=seed,
        probe_steps=probe_steps,
        probe_repairs_per_source=probe_repairs_per_source,
        probe_max_incident_edges=probe_max_incident_edges,
    )
    basis_points = points[basis]
    spatial = np.linalg.norm(basis_points[:, None, :] - basis_points[None, :, :], axis=2)
    spatial = spatial / _positive_median(spatial)
    affinity = _standardize_rows(response_matrix) - 0.15 * spatial
    row_ind, col_ind = linear_sum_assignment(-affinity)
    transition = np.zeros((basis.size, basis.size), dtype=complex)
    transition[row_ind, col_ind] = 1.0
    pullback = transition.conj().T
    assigned = np.argmax(np.abs(transition), axis=1)
    return pullback, 0.0, {
        "response_source": "perturb_remeasure_response",
        "identity_fraction": float(np.mean(assigned == np.arange(basis.size))),
        "mean_assignment_affinity": float(np.mean(affinity[row_ind, col_ind])) if affinity.size else 0.0,
        **meta,
    }


def _perturb_remeasure_response_matrix(
    points: np.ndarray,
    raw_fields: dict[str, np.ndarray],
    basis: np.ndarray,
    *,
    graph_response: dict[str, Any],
    seed: int,
    probe_steps: int,
    probe_repairs_per_source: int,
    probe_max_incident_edges: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    basis = np.asarray(basis, dtype=np.int64)
    left = np.asarray(graph_response["left"], dtype=np.int64)
    right = np.asarray(graph_response["right"], dtype=np.int64)
    port_left = np.asarray(graph_response["port_left"], dtype=np.int64)
    port_right = np.asarray(graph_response["port_right"], dtype=np.int64)
    group_order = int(graph_response.get("group_order", 6))
    patch_count = int(graph_response.get("patch_count", points.shape[0]))
    if left.size != right.size or left.size != port_left.size or left.size != port_right.size:
        raise ValueError("graph_response edge arrays must have the same length")

    rng = np.random.default_rng(seed)
    incident_edges = _incident_edges(left, right, patch_count)
    node_score = _repair_affinity_score(raw_fields, np.arange(patch_count, dtype=np.int64))
    before_signature = _node_packet_signature(port_left, port_right, left, right, patch_count)
    response_matrix = np.zeros((basis.size, basis.size), dtype=float)
    total_perturbed_edges = 0
    total_repaired_edges = 0

    for row, node_raw in enumerate(basis):
        node = int(node_raw)
        pl = port_left.copy()
        pr = port_right.copy()
        incident = np.asarray(incident_edges[node], dtype=np.int64)
        if incident.size > int(probe_max_incident_edges):
            incident = rng.choice(incident, size=int(probe_max_incident_edges), replace=False)
        for edge in incident:
            if int(left[edge]) == node:
                pl[edge] = (pl[edge] + 1) % group_order
            else:
                pr[edge] = (pr[edge] + 1) % group_order
        total_perturbed_edges += int(incident.size)

        repair_count = np.zeros(patch_count, dtype=float)
        for _ in range(max(1, int(probe_steps))):
            active = np.flatnonzero(pl != pr)
            if active.size == 0:
                break
            if active.size > int(probe_repairs_per_source):
                active = rng.choice(active, size=int(probe_repairs_per_source), replace=False)
            left_score = node_score[left[active]]
            right_score = node_score[right[active]]
            repair_left = left_score >= right_score
            if np.any(repair_left):
                pl[active[repair_left]] = pr[active[repair_left]]
            if np.any(~repair_left):
                pr[active[~repair_left]] = pl[active[~repair_left]]
            repair_count += np.bincount(left[active], minlength=patch_count)
            repair_count += np.bincount(right[active], minlength=patch_count)
            total_repaired_edges += int(active.size)

        after_signature = _node_packet_signature(pl, pr, left, right, patch_count)
        changed = (after_signature != before_signature).astype(float)
        response = repair_count + 0.5 * changed
        if not np.any(response[basis] > 0.0):
            response[node] = 1.0
        response_matrix[row, :] = response[basis]

    row_mass = np.sum(response_matrix, axis=1)
    return response_matrix, {
        "mean_perturbed_edges_per_source": float(total_perturbed_edges / max(int(basis.size), 1)),
        "mean_repaired_edges_per_source": float(total_repaired_edges / max(int(basis.size), 1)),
        "response_row_mass_mean": float(np.mean(row_mass)) if row_mass.size else 0.0,
        "response_row_mass_std": float(np.std(row_mass)) if row_mass.size else 0.0,
        "probe_steps": int(probe_steps),
        "probe_repairs_per_source": int(probe_repairs_per_source),
        "probe_max_incident_edges": int(probe_max_incident_edges),
        "response_feature_count": 0,
    }


def _repair_affinity_features(
    points: np.ndarray,
    cap: RoundCap,
    raw_fields: dict[str, np.ndarray],
    basis: np.ndarray,
) -> np.ndarray:
    cap = cap.normalized()
    cross = np.cross(cap.axis, cap.tangent)
    columns = [
        _standardize(points[basis] @ cap.axis - math.cos(cap.theta0)),
        _standardize(points[basis] @ cap.tangent),
        _standardize(points[basis] @ cross),
    ]
    for name in (
        "stable_count",
        "committed_mask",
        "repair_load",
        "cumulative_repair_load",
        "local_mismatch_density",
        "s3_class_density",
        "s3_sector_class",
    ):
        if name in raw_fields:
            columns.append(_standardize(np.asarray(raw_fields[name], dtype=float)[basis]))
    if "record_signature" in raw_fields:
        signature = np.asarray(raw_fields["record_signature"], dtype=np.int64)[basis]
        columns.append(_standardize((np.mod(signature, 257)).astype(float) / 257.0))
        columns.append(_standardize((np.mod(signature, 17)).astype(float) / 17.0))
    return np.column_stack(columns) if columns else np.zeros((basis.size, 0), dtype=float)


def _repair_affinity_score(raw_fields: dict[str, np.ndarray], basis: np.ndarray) -> np.ndarray:
    score = np.zeros(basis.size, dtype=float)
    terms = {
        "local_mismatch_density": 0.35,
        "repair_load": 0.25,
        "cumulative_repair_load": 0.25,
        "s3_class_density": 0.10,
        "stable_count": -0.15,
        "committed_mask": -0.05,
    }
    for name, weight in terms.items():
        if name in raw_fields:
            score += float(weight) * _standardize(np.asarray(raw_fields[name], dtype=float)[basis])
    return _standardize(score)


def _positive_median(values: np.ndarray) -> float:
    positive = np.asarray(values, dtype=float)[np.asarray(values, dtype=float) > 0.0]
    if positive.size == 0:
        return 1.0
    return max(float(np.median(positive)), 1.0e-12)


def _incident_edges(left: np.ndarray, right: np.ndarray, patch_count: int) -> list[list[int]]:
    incident: list[list[int]] = [[] for _ in range(int(patch_count))]
    for edge, (i, j) in enumerate(zip(left.tolist(), right.tolist(), strict=True)):
        incident[int(i)].append(edge)
        incident[int(j)].append(edge)
    return incident


def _node_packet_signature(
    port_left: np.ndarray,
    port_right: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    patch_count: int,
) -> np.ndarray:
    left_packet = port_left.astype(float) + 1.0
    right_packet = port_right.astype(float) + 1.0
    return (
        131.0 * np.bincount(left, weights=left_packet, minlength=patch_count)
        + 17.0 * np.bincount(left, weights=right_packet, minlength=patch_count)
        + 29.0 * np.bincount(right, weights=right_packet, minlength=patch_count)
        + 7.0 * np.bincount(right, weights=left_packet, minlength=patch_count)
    ).astype(np.int64)


def _standardize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    row_sum = np.sum(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_sum, 1.0e-12)
    centered = normalized - np.mean(normalized, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    return centered / np.maximum(scale, 1.0e-12)



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


def _observable_matrix(
    raw_fields: dict[str, np.ndarray],
    observable: str,
    basis: np.ndarray,
    *,
    points: np.ndarray | None = None,
    cap: RoundCap | None = None,
) -> np.ndarray:
    if points is not None and cap is not None and observable in {
        "cap_flow_current",
        "collar_crossing_current",
        "packet_flow_current",
    }:
        antisym = _cap_flow_antisymmetric(points, cap, basis)
        if observable in {"collar_crossing_current", "packet_flow_current"}:
            partition = cap_collar_partition(points, cap, cap.collar_width)
            inside = partition.inside_mask[basis].astype(float)
            collar = partition.collar_mask[basis].astype(float)
            outside = partition.outside_mask[basis].astype(float)
            collar_pair = np.maximum(collar[:, None], collar[None, :])
            crossing_pair = (inside[:, None] * outside[None, :]) + (outside[:, None] * inside[None, :])
            antisym = antisym * np.maximum(collar_pair, crossing_pair)
        if observable == "packet_flow_current":
            packets = visible_packets(raw_fields)[basis]
            packet_pair = (packets[:, None] != packets[None, :]).astype(float)
            antisym = antisym * packet_pair
        matrix = _hermitian(1j * antisym)
        norm = float(np.linalg.norm(matrix, ord="fro"))
        if norm > 1.0e-12:
            return matrix / norm
        links = _collar_link_matrix(points, basis, cap_collar_partition(points, cap, cap.collar_width))
        link_norm = float(np.linalg.norm(links, ord="fro"))
        if link_norm > 1.0e-12:
            return links / link_norm
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
