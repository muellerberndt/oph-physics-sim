from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm, logm, polar
from scipy.optimize import linear_sum_assignment

from oph_fpe.claims import (
    BRANCH_INSTANTIATION_SANITY,
    BW_KMS_BRANCH_INSTANTIATION_RECEIPT,
    BW_KMS_BRANCH_REPLAY_RECEIPT,
    DEMO,
    ENDOGENOUS_MODULAR_GENERATOR_RECEIPT,
    OPH_LORENTZ_THEOREM_FINITE_CONTRACT_RECEIPT,
    with_claim_metadata,
)
from oph_fpe.algebra.maxent_cap_state import maxent_record_operator_cap_state
from oph_fpe.bulk.cap_geometry import RoundCap, cap_weights, lambda_cap
from oph_fpe.bulk.collar_state import cap_collar_partition, fawzi_renner_bound, visible_packets
from oph_fpe.gauge.covariant_overlap import (
    covariant_discrepancy,
    covariant_mismatch_mask,
    gauge_invariant_edge_residual,
    group_multiply_indices,
    repair_covariant_port_pairs,
    repair_production_sector_links,
)


DECLARED_CAP_FLOW_STATE_MODES = frozenset({"cap_flow_graph_generator", "cap_flow_detailed_balance_kernel"})
DIRECT_AUTOMORPHISM_STATE_MODES = frozenset({"transition_response_unitary"})
DECLARED_RESPONSE_DENSITY_STATE_MODES = frozenset({"transition_response_density_log"})
REPAIR_RESPONSE_DENSITY_STATE_MODES = frozenset({"repair_affinity_response_density_log"})
PERTURB_RESPONSE_DENSITY_STATE_MODES = frozenset({"perturb_remeasure_response_density_log"})
PERTURB_RESPONSE_KERNEL_STATE_MODES = frozenset({"perturb_remeasure_response_kernel_log"})
COLLAR_OPERATOR_STATE_MODES = frozenset({"collar_operator_system", "support_visible_collar_operator_system"})
MAXENT_RECORD_OPERATOR_STATE_MODES = frozenset({"maxent_record_operator_state"})
KOOPMAN_GENERATOR_STATE_MODES = frozenset({"history_koopman_generator_state"})


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
    """Return the paper-convention modular unitary ``exp(-i t K)``.

    The compact paper defines ``K=-log(rho)`` and transports observables as
    ``exp(-i t K) A exp(+i t K)``.  Keeping that sign here avoids silently
    reversing the oriented cap flow in every state-derived probe.
    """
    eigvals, eigvecs = np.linalg.eigh(_hermitian(K))
    phases = np.exp(-1j * float(t) * eigvals)
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


def geometric_soft_transport_operator(
    points: np.ndarray,
    cap: RoundCap,
    s: float,
    basis_indices: np.ndarray,
    *,
    k: int = 8,
) -> tuple[np.ndarray, float]:
    basis_points = points[basis_indices]
    mapped = lambda_cap(basis_points, cap, s)
    distances = np.linalg.norm(mapped[:, None, :] - basis_points[None, :, :], axis=2)
    if distances.size == 0:
        return np.zeros((basis_indices.size, basis_indices.size), dtype=complex), 0.0
    neighbor_count = min(max(1, int(k)), basis_indices.size)
    nearest = np.argpartition(distances, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]
    nearest_distances = np.take_along_axis(distances, nearest, axis=1)
    positive = nearest_distances[nearest_distances > 0.0]
    sigma = max(float(np.median(positive)) if positive.size else _positive_median(distances), 1.0e-12)
    weights = np.exp(-0.5 * np.square(nearest_distances / sigma))
    weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1.0e-15)
    L = np.zeros((basis_indices.size, basis_indices.size), dtype=complex)
    rows = np.arange(basis_indices.size)[:, None]
    L[rows, nearest] = weights
    eta = float(np.mean(np.sum(weights * nearest_distances, axis=1)))
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
    if state_mode in {
        "history_transition_kernel",
        "transition_count_kernel",
        "record_history_kernel",
        *KOOPMAN_GENERATOR_STATE_MODES,
    }:
        rho = history_transition_density(
            points,
            cap,
            support,
            raw_fields,
            history_fields or [],
            regularizer=regularizer,
        )
        return rho, support, packets
    if state_mode in MAXENT_RECORD_OPERATOR_STATE_MODES:
        result = maxent_record_operator_cap_state(
            raw_fields,
            history_fields or [],
            support,
            regularizer=regularizer,
        )
        return result.rho, support, packets
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
        "modular_time",
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
        "modular_time",
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
    unitary = np.asarray(transition, dtype=complex)
    lag = max(float(response_time), 1.0e-12)
    # U=exp(-i dt K), so on the principal logarithm branch K=(+i/dt)log(U).
    # The previous Koopman path used -i/dt and therefore reconstructed U^*,
    # reversing the inferred modular-time orientation.
    return _hermitian((1j / lag) * logm(unitary))


def history_koopman_modular_generator(
    raw_fields: dict[str, np.ndarray],
    history_fields: list[dict[str, np.ndarray]],
    basis: np.ndarray,
    *,
    response_lag: float,
    regularizer: float = 1.0e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer a finite unitary generator from observer-visible record histories."""

    basis = np.asarray(basis, dtype=np.int64)
    if basis.size == 0:
        return np.zeros((0, 0), dtype=complex), {"usable": False, "reason": "empty_basis"}
    if basis.size == 1:
        return np.zeros((1, 1), dtype=complex), {"usable": True, "rank": 1, "nonunitary_defect": 0.0}
    states = [state for state in [*list(history_fields or []), raw_fields] if state]
    series = _koopman_feature_series(states, basis)
    if len(series) < 2:
        return np.zeros((basis.size, basis.size), dtype=complex), {
            "usable": False,
            "reason": "requires_at_least_two_history_states",
            "rank": 0,
            "nonunitary_defect": 1.0,
        }
    left_columns: list[np.ndarray] = []
    right_columns: list[np.ndarray] = []
    for before, after in zip(series[:-1], series[1:], strict=False):
        width = min(before.shape[1], after.shape[1])
        if width <= 0:
            continue
        left_columns.append(_standardize_columns(before[:, :width]))
        right_columns.append(_standardize_columns(after[:, :width]))
    if not left_columns:
        return np.zeros((basis.size, basis.size), dtype=complex), {
            "usable": False,
            "reason": "empty_history_feature_pairs",
            "rank": 0,
            "nonunitary_defect": 1.0,
        }
    effective_response_lag, response_lag_source = _koopman_effective_response_lag(
        states,
        basis,
        fallback_lag=float(response_lag),
    )
    X = np.hstack(left_columns)
    Y = np.hstack(right_columns)
    sample_count = max(float(X.shape[1]), 1.0)
    eye = np.eye(basis.size, dtype=float)
    C00 = (X @ X.T) / sample_count + float(regularizer) * eye
    C01 = (X @ Y.T) / sample_count
    C11 = (Y @ Y.T) / sample_count + float(regularizer) * eye
    C00_inv, rank_00 = _psd_invsqrt(C00)
    C11_inv, rank_11 = _psd_invsqrt(C11)
    transition = C00_inv @ C01 @ C11_inv
    try:
        unitary, _positive = polar(transition)
        generator = modular_generator_from_unitary_transition(unitary, effective_response_lag)
    except Exception:
        generator = np.zeros((basis.size, basis.size), dtype=complex)
        unitary = np.eye(basis.size, dtype=complex)
    if not np.all(np.isfinite(generator)):
        generator = np.zeros((basis.size, basis.size), dtype=complex)
    nonunitary_defect = _relative_frobenius(transition - unitary, transition)
    generator_reconstruction_error = _relative_frobenius(
        modular_unitary(generator, effective_response_lag) - unitary,
        unitary,
    )
    generator_norm = float(np.linalg.norm(generator, ord="fro"))
    return generator, {
        "usable": True,
        "mode": "history_koopman_modular_generator",
        "history_state_count": int(len(series)),
        "history_pair_count": int(len(left_columns)),
        "feature_column_count": int(X.shape[1]),
        "rank": int(min(rank_00, rank_11)),
        "rank_00": int(rank_00),
        "rank_11": int(rank_11),
        "configured_response_lag": float(response_lag),
        "effective_response_lag": float(effective_response_lag),
        "response_lag_source": response_lag_source,
        "nonunitary_defect": float(nonunitary_defect),
        "generator_reconstruction_error": float(generator_reconstruction_error),
        "generator_frobenius_norm": generator_norm,
        "geometry_dependency_count": 0,
        "claim_boundary": (
            "finite Koopman generator inferred from observer-visible record/history feature vectors. "
            "It does not use lambda_C, cap tangent coordinates, H3 coordinates, or a declared 2*pi target."
        ),
    }


def _koopman_effective_response_lag(
    states: list[dict[str, np.ndarray]],
    basis: np.ndarray,
    *,
    fallback_lag: float,
) -> tuple[float, str]:
    """Infer Koopman time units from observer-visible modular-time records."""

    basis = np.asarray(basis, dtype=np.int64)
    if basis.size == 0:
        return float(fallback_lag), "configured_fallback_empty_basis"
    max_index = int(np.max(basis, initial=-1))
    increments: list[float] = []
    for before, after in zip(states[:-1], states[1:], strict=False):
        before_time = before.get("modular_time")
        after_time = after.get("modular_time")
        if before_time is None or after_time is None:
            continue
        before_array = np.asarray(before_time, dtype=float)
        after_array = np.asarray(after_time, dtype=float)
        if before_array.ndim != 1 or after_array.ndim != 1:
            continue
        if before_array.size <= max_index or after_array.size <= max_index:
            continue
        delta = after_array[basis] - before_array[basis]
        finite = delta[np.isfinite(delta)]
        positive = finite[finite > 1.0e-12]
        if positive.size:
            increments.append(float(np.median(positive)))
    if increments:
        lag = float(np.median(increments))
        if np.isfinite(lag) and lag > 1.0e-12:
            return lag, "observer_visible_modular_time_median_increment"
    return float(fallback_lag), "configured_fallback_no_modular_time_increment"


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
    elif state_mode in KOOPMAN_GENERATOR_STATE_MODES and sim_modular_flow:
        K_koopman, koopman_meta = history_koopman_modular_generator(
            raw_fields,
            history_fields or [],
            basis,
            response_lag=transition_response_time,
        )
        U = modular_unitary(K_koopman, t)
        eta_direct = float(koopman_meta.get("nonunitary_defect", 0.0))
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
    elif target_operator_mode == "soft":
        L, eta_interpolation = geometric_soft_transport_operator(points, cap, target_scale * t, basis)
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
    audit_candidates, audit_candidate_source = _generator_scale_audit_candidates(
        configured_generator_scale=generator_scale,
        explicit_candidates=generator_scale_candidates,
        controls=controls or [],
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
        generator_scale_candidates=audit_candidates,
        candidate_source=audit_candidate_source,
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
    inferred_clock_fit = _inferred_modular_clock_fit(
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
        seed=seed + 15401,
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
        inferred_clock_fit=inferred_clock_fit,
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
                    elif state_mode in MAXENT_RECORD_OPERATOR_STATE_MODES:
                        rows[-1]["generator_source"] = "maxent_record_operator_state"
                    elif state_mode in KOOPMAN_GENERATOR_STATE_MODES:
                        rows[-1]["generator_source"] = "history_koopman_generator_state"
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
    inferred_clock_fit: dict[str, Any] | None = None,
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
    maxent_record_operator = state_mode in MAXENT_RECORD_OPERATOR_STATE_MODES
    koopman_generator = state_mode in KOOPMAN_GENERATOR_STATE_MODES
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
    inferred_clock_fit = inferred_clock_fit or {"enabled": False, "receipt": False}
    finite_generator_rows = bool(rows and np.isfinite(float(median)))
    clock_response_degenerate = bool(inferred_clock_fit.get("response_degenerate", False))
    endogenous_modular_generator_receipt = bool(
        endogenous_generator
        and finite_generator_rows
        and not clock_response_degenerate
    )
    kms_geometric_clock_fit_receipt = bool(inferred_clock_fit.get("receipt", False))
    summary = {
        "mode": "state_derived_modular_probe",
        "receipt_name": BW_KMS_BRANCH_REPLAY_RECEIPT,
        BW_KMS_BRANCH_REPLAY_RECEIPT: bool(normalization_declared and correct_beats_controls),
        BW_KMS_BRANCH_INSTANTIATION_RECEIPT: bool(normalization_declared and correct_beats_controls),
        "legacy_receipt_name": BW_KMS_BRANCH_INSTANTIATION_RECEIPT,
        OPH_LORENTZ_THEOREM_FINITE_CONTRACT_RECEIPT: False,
        "finite_lorentz_theorem_contract_receipt": False,
        "receipt_taxonomy": "L0_branch_replay_not_L1_L7_finite_contract",
        "state_mode": state_mode,
        "target_operator_mode": target_operator_mode,
        "transition_response_time": float(transition_response_time),
        "transition_response_scale": float(transition_response_scale),
        "density_inverse_temperature": float(density_inverse_temperature),
        "generator_scale": float(generator_scale),
        "endogenous_modular_generator": endogenous_generator,
        "endogenous_generator_non_degenerate": bool(endogenous_modular_generator_receipt),
        "endogenous_generator_receipt_boundary": (
            "L2 only: the modular generator is finite and observer-record/cap-state derived. "
            "It does not certify the 2*pi KMS/BW clock; that is the separate L3 "
            "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT gate."
        ),
        "declared_cap_flow_generator": declared_cap_flow,
        "declared_transition_response_density": declared_response_density,
        "repair_affinity_response_density": repair_response_density,
        "perturb_remeasure_response_density": perturb_response_density,
        "perturb_remeasure_response_kernel": perturb_response_kernel,
        "maxent_record_operator_state": maxent_record_operator,
        "history_koopman_generator_state": koopman_generator,
        "TRANSITION_RESPONSE_DENSITY_LOG_CALIBRATION_RECEIPT": density_log_calibration_receipt,
        ENDOGENOUS_MODULAR_GENERATOR_RECEIPT: endogenous_modular_generator_receipt,
        "ENDOGENOUS_MODULAR_GENERATOR_RECEIPT": endogenous_modular_generator_receipt,
        "endogenous_modular_generator_receipt": endogenous_modular_generator_receipt,
        "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT": kms_geometric_clock_fit_receipt,
        "kms_geometric_clock_fit_receipt": kms_geometric_clock_fit_receipt,
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
            else "endogenous_maxent_record_operator_state"
            if maxent_record_operator
            else "endogenous_history_koopman_generator_state"
            if koopman_generator
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
            "counts directly as M M^T instead of forcing a permutation assignment. maxent_record_operator_state "
            "constructs rho_C from observer record/history operators without the geometric flow target. "
            "history_koopman_generator_state infers a unitary transition generator from observer-visible "
            "history features without lambda_C or H3 input. These are current "
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
        "inferred_modular_clock_fit": inferred_clock_fit,
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
        receipt=BW_KMS_BRANCH_REPLAY_RECEIPT,
        physical_claim=False,
    )


def _inferred_modular_clock_fit(
    points: np.ndarray,
    caps: list[RoundCap],
    raw_fields: dict[str, np.ndarray],
    collar_by_cap: dict[int, dict[str, Any]],
    *,
    times: list[float],
    observables: list[str],
    regularizers: list[float],
    max_basis: int,
    state_mode: str,
    target_operator_mode: str,
    transition_response_time: float,
    transition_response_scale: float,
    density_inverse_temperature: float,
    generator_scale: float,
    history_fields: list[dict[str, np.ndarray]] | None,
    graph_response: dict[str, Any] | None,
    probe_steps: int,
    probe_repairs_per_source: int,
    probe_max_incident_edges: int,
    seed: int,
) -> dict[str, Any]:
    if state_mode in (
        DIRECT_AUTOMORPHISM_STATE_MODES
        | DECLARED_CAP_FLOW_STATE_MODES
        | DECLARED_RESPONSE_DENSITY_STATE_MODES
    ):
        return {
            "enabled": False,
            "not_applicable": True,
            "receipt": False,
            "state_mode": state_mode,
            "claim_boundary": (
                "not run for declared/direct BW lanes. L3 requires a clock coefficient inferred "
                "from an endogenous observer-record cap state, not from a supplied geometric flow."
            ),
        }
    finite_times = sorted({float(t) for t in times if np.isfinite(float(t)) and abs(float(t)) > 1.0e-12})
    known_scales = [0.0, 1.0, math.pi, 2.0 * math.pi, 4.0 * math.pi]
    scale_grid = _unique_floats(
        [
            *[float(value) for value in np.linspace(0.0, 4.0 * math.pi, 33)],
            *known_scales,
        ]
    )
    rows: list[dict[str, Any]] = []
    for cap_id, cap in enumerate(caps):
        for time_index, t in enumerate(finite_times):
            for reg_index, reg in enumerate(regularizers):
                rho, basis, _ = cap_state_density(
                    points,
                    cap,
                    raw_fields,
                    history_fields=history_fields,
                    max_basis=max_basis,
                    regularizer=float(reg),
                    seed=seed + cap_id * 1009 + time_index * 101 + reg_index,
                    state_mode=state_mode,
                    transition_response_time=transition_response_time,
                    transition_response_scale=transition_response_scale,
                    density_inverse_temperature=density_inverse_temperature,
                    graph_response=graph_response,
                    probe_steps=probe_steps,
                    probe_repairs_per_source=probe_repairs_per_source,
                    probe_max_incident_edges=probe_max_incident_edges,
                )
                if basis.size == 0:
                    continue
                for observable in observables:
                    O = _observable_matrix(raw_fields, observable, basis, points=points, cap=cap)
                    if state_mode in KOOPMAN_GENERATOR_STATE_MODES:
                        K_koopman, _koopman_meta = history_koopman_modular_generator(
                            raw_fields,
                            history_fields or [],
                            basis,
                            response_lag=transition_response_time,
                        )
                        U = modular_unitary(K_koopman, float(t))
                        O_sim = U @ O @ U.conj().T
                    else:
                        O_sim = state_derived_modular_transport(
                            O,
                            rho,
                            float(t),
                            float(reg),
                            generator_scale=generator_scale,
                        )
                    residual_by_scale: dict[str, float] = {}
                    best_scale = None
                    best_residual = float("inf")
                    for scale in scale_grid:
                        if target_operator_mode == "permutation":
                            L, _ = geometric_permutation_operator(points, cap, float(scale) * float(t), basis)
                        elif target_operator_mode == "soft":
                            L, _ = geometric_soft_transport_operator(points, cap, float(scale) * float(t), basis)
                        else:
                            L, _ = geometric_transport_operator(points, cap, float(scale) * float(t), basis)
                        O_target = L.conj().T @ O @ L
                        residual = _relative_frobenius(O_sim - O_target, O)
                        label = _scale_label(scale)
                        residual_by_scale[label] = float(residual)
                        if residual < best_residual:
                            best_residual = float(residual)
                            best_scale = float(scale)
                    known_residuals = {
                        _scale_label(scale): residual_by_scale.get(_scale_label(scale))
                        for scale in known_scales
                        if residual_by_scale.get(_scale_label(scale)) is not None
                    }
                    best_known_scale_label = (
                        min(known_residuals, key=known_residuals.get) if known_residuals else None
                    )
                    scale_diagnostics = _clock_row_scale_diagnostics(known_residuals)
                    rows.append(
                        {
                            "cap_id": int(cap_id),
                            "time": float(t),
                            "observable": str(observable),
                            "regularizer_a": float(reg),
                            "s_hat": float(best_scale * float(t)) if best_scale is not None else None,
                            "kappa_row_hat": float(best_scale) if best_scale is not None else None,
                            "best_scale_label": _scale_label(float(best_scale)) if best_scale is not None else None,
                            "best_residual": float(best_residual),
                            "known_scale_residuals": known_residuals,
                            "best_known_scale_label": best_known_scale_label,
                            **scale_diagnostics,
                            "clock_carrier_row": bool(best_known_scale_label not in (None, "0")),
                            "informative_clock_carrier_row": bool(
                                best_known_scale_label not in (None, "0")
                                and scale_diagnostics["known_scale_best_advantage"] >= 0.005
                                and scale_diagnostics["known_scale_best_gap"] >= 0.001
                            ),
                            "basis_count": int(basis.size),
                        }
                    )
    valid = [
        row
        for row in rows
        if row.get("s_hat") is not None
        and row.get("kappa_row_hat") is not None
        and np.isfinite(float(row["s_hat"]))
        and np.isfinite(float(row["kappa_row_hat"]))
    ]
    distinct_time_count = len({float(row["time"]) for row in valid})
    if not valid or distinct_time_count < 3:
        return {
            "enabled": True,
            "receipt": False,
            "mode": "inferred_modular_clock_fit",
            "row_count": int(len(rows)),
            "distinct_time_count": int(distinct_time_count),
            "blockers": ["requires_at_least_three_nonzero_modular_times"],
            "rows": rows[:128],
            "claim_boundary": (
                "finite inferred-clock audit. It searches continuously over cap-flow coefficient "
                "kappa through s_hat(t)=kappa*t+b and only then compares against 2*pi controls."
            ),
        }
    known = {
        "1x": 1.0,
        "pi": math.pi,
        "2pi": 2.0 * math.pi,
        "4pi": 4.0 * math.pi,
    }
    all_row_fit = _clock_fit_from_rows(valid, known=known, fit_label="all_rows")
    clock_carrier_rows = [row for row in valid if bool(row.get("clock_carrier_row", False))]
    clock_carrier_fit = _clock_fit_from_rows(
        clock_carrier_rows,
        known=known,
        fit_label="nonstatic_clock_carrier_rows",
    )
    informative_clock_carrier_rows = [
        row for row in clock_carrier_rows if bool(row.get("informative_clock_carrier_row", False))
    ]
    informative_clock_carrier_fit = _clock_fit_from_rows(
        informative_clock_carrier_rows,
        known=known,
        fit_label="informative_nonstatic_clock_carrier_rows",
    )
    # This selection rule is fixed before looking at which fit passes.  The old
    # implementation tried several overlapping subsets and selected the first
    # passing one, which made the receipt vulnerable to subset shopping.
    # Prefer genuinely scale-informative carriers; if there are too few of
    # those, report the non-static carrier fit as a diagnostic.  ``all_rows``
    # remains visible below but can no longer rescue the receipt.
    informative_ready = bool(
        informative_clock_carrier_fit.get("valid_row_count", 0) > 0
        and informative_clock_carrier_fit.get("distinct_time_count", 0) >= 3
    )
    selected_fit = informative_clock_carrier_fit if informative_ready else clock_carrier_fit
    receipt = bool(selected_fit.get("receipt", False))
    blockers: list[str] = []
    if not receipt:
        blockers.extend(
            f"all_rows:{blocker}" for blocker in list(all_row_fit.get("blockers") or [])
        )
        blockers.extend(
            f"clock_carrier_rows:{blocker}"
            for blocker in list(clock_carrier_fit.get("blockers") or [])
        )
        blockers.extend(
            f"informative_clock_carrier_rows:{blocker}"
            for blocker in list(informative_clock_carrier_fit.get("blockers") or [])
        )
    return {
        "enabled": True,
        "receipt": receipt,
        "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT": receipt,
        "kms_geometric_clock_fit_receipt": receipt,
        "mode": "inferred_modular_clock_fit",
        "selected_clock_fit": selected_fit.get("fit_label"),
        "clock_fit_selection_policy": (
            "predeclared_informative_nonstatic_carriers_else_nonstatic_carriers_no_pass_shopping"
        ),
        "row_count": int(len(rows)),
        "valid_row_count": int(len(valid)),
        "clock_carrier_row_count": int(len(clock_carrier_rows)),
        "informative_clock_carrier_row_count": int(len(informative_clock_carrier_rows)),
        "static_or_no_flow_row_count": int(len(valid) - len(clock_carrier_rows)),
        "distinct_time_count": int(distinct_time_count),
        "kappa_hat": selected_fit.get("kappa_hat"),
        "kappa_95ci": selected_fit.get("kappa_95ci"),
        "intercept_hat": selected_fit.get("intercept_hat"),
        "median_fit_abs_residual": selected_fit.get("median_fit_abs_residual"),
        "nearest_known_scale": selected_fit.get("nearest_known_scale"),
        "median_known_scale_residuals": selected_fit.get("median_known_scale_residuals", {}),
        "wrong_scale_residual_ratios_vs_2pi": selected_fit.get(
            "wrong_scale_residual_ratios_vs_2pi",
            {},
        ),
        "no_flow_selected_fraction": selected_fit.get("no_flow_selected_fraction"),
        "response_degenerate": selected_fit.get("response_degenerate", False),
        "all_row_fit": all_row_fit,
        "clock_carrier_fit": clock_carrier_fit,
        "informative_clock_carrier_fit": informative_clock_carrier_fit,
        "clock_row_quality_summary": _clock_row_quality_summary(valid),
        "blockers": blockers,
        "rows": rows[:128],
        "claim_boundary": (
            "finite inferred-clock audit. The fit searches over cap-flow parameter s without naming "
            "2*pi, fits s_hat(t)=kappa*t+b from endogenous modular transport, and only afterward "
            "compares kappa to 1, pi, 2*pi, and 4*pi. Static rows whose own known-scale diagnostic "
            "selects zero flow are retained in all_row_fit but excluded from clock_carrier_fit, so "
            "invariant repair/readout observables cannot dominate the KMS clock estimate. Rows also "
            "report target-free known-scale contrast and gap diagnostics, so nearly flat residual "
            "landscapes are visible rather than treated as strong clock evidence. Passing this is still "
            "finite-regulator evidence, not a continuum theorem proof."
        ),
    }


def _clock_row_scale_diagnostics(known_residuals: dict[str, float]) -> dict[str, Any]:
    finite = {
        str(label): float(value)
        for label, value in (known_residuals or {}).items()
        if value is not None and np.isfinite(float(value))
    }
    if len(finite) < 2:
        return {
            "known_scale_spread": 0.0,
            "known_scale_best_advantage": 0.0,
            "known_scale_best_gap": 0.0,
            "known_scale_residual_flat": True,
        }
    values = sorted(finite.values())
    median = max(float(np.median(values)), 1.0e-15)
    best = values[0]
    second = values[1]
    spread = (values[-1] - best) / median
    advantage = (median - best) / median
    gap = (second - best) / median
    return {
        "known_scale_spread": float(spread),
        "known_scale_best_advantage": float(advantage),
        "known_scale_best_gap": float(gap),
        "known_scale_residual_flat": bool(advantage < 0.005 or gap < 0.001),
    }


def _clock_row_quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("known_scale_best_advantage") is not None
        and np.isfinite(float(row.get("known_scale_best_advantage", 0.0)))
    ]
    if not valid:
        return {
            "row_count": 0,
            "flat_known_scale_residual_count": 0,
            "informative_clock_carrier_count": 0,
        }
    advantages = np.asarray([float(row.get("known_scale_best_advantage", 0.0)) for row in valid], dtype=float)
    gaps = np.asarray([float(row.get("known_scale_best_gap", 0.0)) for row in valid], dtype=float)
    return {
        "row_count": int(len(valid)),
        "flat_known_scale_residual_count": int(sum(bool(row.get("known_scale_residual_flat", False)) for row in valid)),
        "clock_carrier_count": int(sum(bool(row.get("clock_carrier_row", False)) for row in valid)),
        "informative_clock_carrier_count": int(
            sum(bool(row.get("informative_clock_carrier_row", False)) for row in valid)
        ),
        "median_known_scale_best_advantage": float(np.median(advantages)),
        "median_known_scale_best_gap": float(np.median(gaps)),
        "claim_boundary": (
            "Target-free row-quality diagnostics for the inferred KMS/BW clock. A row with a nearly "
            "flat known-scale residual landscape is retained for audit but should not be read as strong "
            "clock evidence."
        ),
    }


def _clock_fit_from_rows(
    rows: list[dict[str, Any]],
    *,
    known: dict[str, float],
    fit_label: str,
) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("s_hat") is not None
        and row.get("kappa_row_hat") is not None
        and np.isfinite(float(row["s_hat"]))
        and np.isfinite(float(row["kappa_row_hat"]))
    ]
    distinct_time_count = len({float(row["time"]) for row in valid})
    if not valid or distinct_time_count < 3:
        return {
            "fit_label": fit_label,
            "receipt": False,
            "valid_row_count": int(len(valid)),
            "distinct_time_count": int(distinct_time_count),
            "blockers": ["requires_at_least_three_nonzero_modular_times"],
        }
    x = np.asarray([float(row["time"]) for row in valid], dtype=float)
    y = np.asarray([float(row["s_hat"]) for row in valid], dtype=float)
    design = np.column_stack([x, np.ones_like(x)])
    kappa_hat, intercept_hat = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ np.asarray([kappa_hat, intercept_hat], dtype=float)
    residual = y - fitted
    row_kappas = np.asarray([float(row["kappa_row_hat"]) for row in valid], dtype=float)
    stderr = (
        float(np.std(row_kappas, ddof=1) / math.sqrt(max(row_kappas.size, 1)))
        if row_kappas.size > 1
        else float("inf")
    )
    ci_half_width = 1.96 * stderr if np.isfinite(stderr) else float("inf")
    ci_low = float(kappa_hat - ci_half_width)
    ci_high = float(kappa_hat + ci_half_width)
    nearest_known = min(known, key=lambda label: abs(float(kappa_hat) - known[label]))
    median_known_residuals: dict[str, float] = {}
    for label in ["0", "1x", "pi", "2pi", "4pi"]:
        values = [
            float(row["known_scale_residuals"][label])
            for row in valid
            if label in row.get("known_scale_residuals", {})
            and np.isfinite(float(row["known_scale_residuals"][label]))
        ]
        if values:
            median_known_residuals[label] = float(np.median(values))
    two_pi_residual = median_known_residuals.get("2pi")
    wrong_scale_ratios = {
        label: float(value) / max(float(two_pi_residual), 1.0e-15)
        for label, value in median_known_residuals.items()
        if label != "2pi" and two_pi_residual is not None
    }
    no_flow_fraction = float(np.mean(np.isclose(row_kappas, 0.0, atol=0.25))) if row_kappas.size else 1.0
    response_degenerate = bool(no_flow_fraction >= 0.5 or nearest_known == "1x")
    blockers: list[str] = []
    ci_tol = 1.0e-9
    if not (ci_low - ci_tol <= 2.0 * math.pi <= ci_high + ci_tol):
        blockers.append("two_pi_not_inside_kappa_confidence_interval")
    for label in ("1x", "pi", "4pi"):
        value = known[label]
        if ci_low + ci_tol <= value <= ci_high - ci_tol:
            blockers.append(f"{label}_not_excluded_by_kappa_confidence_interval")
    if nearest_known != "2pi":
        blockers.append(f"nearest_known_scale_is_{nearest_known}")
    if two_pi_residual is None:
        blockers.append("two_pi_residual_missing")
    else:
        competing = {
            label: value
            for label, value in median_known_residuals.items()
            if label in {"0", "1x", "pi", "4pi"}
        }
        if competing and not all(float(two_pi_residual) < float(value) for value in competing.values()):
            blockers.append("two_pi_does_not_strictly_beat_all_declared_scale_controls")
    if response_degenerate:
        blockers.append("inferred_clock_response_degenerate_or_wrong_scale")
    return {
        "fit_label": fit_label,
        "receipt": bool(not blockers),
        "valid_row_count": int(len(valid)),
        "distinct_time_count": int(distinct_time_count),
        "kappa_hat": float(kappa_hat),
        "kappa_95ci": [ci_low, ci_high],
        "intercept_hat": float(intercept_hat),
        "median_fit_abs_residual": float(np.median(np.abs(residual))) if residual.size else None,
        "nearest_known_scale": nearest_known,
        "median_known_scale_residuals": median_known_residuals,
        "wrong_scale_residual_ratios_vs_2pi": wrong_scale_ratios,
        "no_flow_selected_fraction": no_flow_fraction,
        "response_degenerate": response_degenerate,
        "blockers": blockers,
    }


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
    candidate_source: str,
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
            "candidate_source": candidate_source,
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
            "candidate_source": candidate_source,
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
        "candidate_source": candidate_source,
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


def _generator_scale_audit_candidates(
    *,
    configured_generator_scale: float,
    explicit_candidates: list[float] | None,
    controls: list[str],
) -> tuple[list[float] | None, str]:
    if explicit_candidates:
        return [float(value) for value in explicit_candidates], "explicit_config"
    if not any(name.startswith("wrong_") and name.endswith("_normalization") for name in controls):
        return None, "not_requested"
    return (
        _unique_floats(
            [
                float(configured_generator_scale),
                0.0,
                1.0,
                math.pi,
                2.0 * math.pi,
                4.0 * math.pi,
            ]
        ),
        "auto_from_wrong_normalization_controls",
    )


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
    missing_graph_fields = [
        key
        for key in ("gauge", "group_name", "group_order")
        if key not in graph_response
    ]
    sector_repair_enabled = bool(graph_response.get("production_sector_repair_enabled", False))
    sector_repair_config = dict(graph_response.get("production_sector_repair_config") or {})
    sector_config_missing = sector_repair_enabled and not sector_repair_config
    if missing_graph_fields or sector_config_missing:
        blockers = [f"gauge_coupled_graph_field_missing:{key}" for key in missing_graph_fields]
        if sector_config_missing:
            blockers.append("production_sector_repair_config_missing_for_replay")
        return np.eye(basis.size, dtype=complex), 0.0, {
            "response_source": "perturb_remeasure_response_fail_closed",
            "identity_fraction": 1.0,
            "gauge_covariant_probe_receipt": False,
            "proof_blockers": blockers,
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
    missing_graph_fields = [
        key
        for key in ("gauge", "group_name", "group_order")
        if key not in graph_response
    ]
    if missing_graph_fields:
        raise ValueError(
            "perturb_remeasure_response requires gauge-coupled graph fields: "
            + ", ".join(missing_graph_fields)
        )
    sector_repair_enabled = bool(graph_response.get("production_sector_repair_enabled", False))
    sector_repair_config = dict(graph_response.get("production_sector_repair_config") or {})
    if sector_repair_enabled and not sector_repair_config:
        raise ValueError(
            "perturb_remeasure_response cannot replay production sector repair "
            "without production_sector_repair_config"
        )
    left = np.asarray(graph_response["left"], dtype=np.int64)
    right = np.asarray(graph_response["right"], dtype=np.int64)
    port_left = np.asarray(graph_response["port_left"], dtype=np.int64)
    port_right = np.asarray(graph_response["port_right"], dtype=np.int64)
    gauge = np.asarray(graph_response["gauge"], dtype=np.int16)
    group_name = str(graph_response["group_name"])
    group_order = int(graph_response.get("group_order", 6))
    patch_count = int(graph_response.get("patch_count", points.shape[0]))
    if not (left.size == right.size == port_left.size == port_right.size == gauge.size):
        raise ValueError("graph_response edge arrays must have the same length")

    rng = np.random.default_rng(seed)
    incident_edges = _incident_edges(left, right, patch_count)
    node_score = _repair_affinity_score(raw_fields, np.arange(patch_count, dtype=np.int64))
    before_residual = gauge_invariant_edge_residual(
        port_left,
        port_right,
        gauge,
        group_name=group_name,
        group_order=group_order,
    )
    before_signature = _node_packet_signature(before_residual, before_residual, left, right, patch_count)
    response_matrix = np.zeros((basis.size, basis.size), dtype=float)
    total_perturbed_edges = 0
    total_repaired_edges = 0
    exact_basis_masses: list[float] = []
    projected_basis_masses: list[float] = []
    projection_fallbacks = 0
    source_echo_masses: list[float] = []
    source_echo_suppressed = 0

    sector_links_mutated_total = 0
    for row, node_raw in enumerate(basis):
        node = int(node_raw)
        pl = port_left.copy()
        pr = port_right.copy()
        # Production sector repair mutates the edge links during repair; the
        # replay therefore runs against a per-source local copy of the gauge
        # so every probe source resettles under the identical production law
        # from the identical committed state.
        local_gauge = gauge.copy() if sector_repair_enabled else gauge
        incident = np.asarray(incident_edges[node], dtype=np.int64)
        if incident.size > int(probe_max_incident_edges):
            incident = rng.choice(incident, size=int(probe_max_incident_edges), replace=False)
        perturb_left = left[incident] == node
        left_edges = incident[perturb_left]
        right_edges = incident[~perturb_left]
        if left_edges.size:
            pl[left_edges] = group_multiply_indices(
                pl[left_edges],
                np.ones(left_edges.size, dtype=np.int16),
                group_name=group_name,
                group_order=group_order,
            )
        if right_edges.size:
            pr[right_edges] = group_multiply_indices(
                pr[right_edges],
                np.ones(right_edges.size, dtype=np.int16),
                group_name=group_name,
                group_order=group_order,
            )
        total_perturbed_edges += int(incident.size)

        repair_count = np.zeros(patch_count, dtype=float)
        for _ in range(max(1, int(probe_steps))):
            active = np.flatnonzero(
                covariant_mismatch_mask(
                    pl,
                    pr,
                    local_gauge,
                    group_name=group_name,
                    group_order=group_order,
                )
            )
            if active.size == 0:
                break
            if active.size > int(probe_repairs_per_source):
                active = rng.choice(active, size=int(probe_repairs_per_source), replace=False)
            if sector_repair_enabled:
                # Production law replay: covariant discrepancy feeds the
                # sector-link mutation, then the endpoint repair applies with
                # the production coin-flip side selection.
                chosen_delta = covariant_discrepancy(
                    pl[active],
                    pr[active],
                    local_gauge[active],
                    group_name=group_name,
                    group_order=group_order,
                )
                sector_links_mutated_total += repair_production_sector_links(
                    local_gauge,
                    active,
                    chosen_delta,
                    group_name=group_name,
                    group_order=group_order,
                    rng=rng,
                    config=sector_repair_config,
                )
                repair_left = rng.random(active.size) < 0.5
            else:
                left_score = node_score[left[active]]
                right_score = node_score[right[active]]
                repair_left = left_score >= right_score
            repair_covariant_port_pairs(
                pl,
                pr,
                local_gauge,
                active,
                repair_left,
                group_name=group_name,
                group_order=group_order,
            )
            repair_count += np.bincount(left[active], minlength=patch_count)
            repair_count += np.bincount(right[active], minlength=patch_count)
            total_repaired_edges += int(active.size)

        after_residual = gauge_invariant_edge_residual(
            pl,
            pr,
            local_gauge,
            group_name=group_name,
            group_order=group_order,
        )
        after_signature = _node_packet_signature(after_residual, after_residual, left, right, patch_count)
        changed = (after_signature != before_signature).astype(float)
        response = repair_count + 0.5 * changed
        exact_basis_masses.append(float(np.sum(response[basis])))
        basis_response = _project_patch_response_to_basis(points, basis, response, source_node=node)
        if not np.any(basis_response > 0.0):
            projection_fallbacks += 1
            source_position = int(np.flatnonzero(basis == node)[0]) if np.any(basis == node) else 0
            basis_response[source_position] = 1.0
        source_positions = np.flatnonzero(basis == node)
        if source_positions.size:
            source_position = int(source_positions[0])
            source_echo = float(basis_response[source_position])
            nonself_mass = float(np.sum(basis_response) - source_echo)
            if nonself_mass > 1.0e-12:
                source_echo_masses.append(source_echo)
                basis_response[source_position] = 0.0
                source_echo_suppressed += 1
        projected_basis_masses.append(float(np.sum(basis_response)))
        response_matrix[row, :] = basis_response

    row_mass = np.sum(response_matrix, axis=1)
    return response_matrix, {
        "production_sector_repair_replayed": bool(sector_repair_enabled),
        "sector_links_mutated_per_source": (
            float(sector_links_mutated_total / max(int(basis.size), 1))
            if sector_repair_enabled
            else 0.0
        ),
        "probe_side_selection": "production_coin" if sector_repair_enabled else "affinity_score",
        "gauge_covariant_probe_receipt": True,
        "mean_perturbed_edges_per_source": float(total_perturbed_edges / max(int(basis.size), 1)),
        "mean_repaired_edges_per_source": float(total_repaired_edges / max(int(basis.size), 1)),
        "response_row_mass_mean": float(np.mean(row_mass)) if row_mass.size else 0.0,
        "response_row_mass_std": float(np.std(row_mass)) if row_mass.size else 0.0,
        "exact_basis_response_mass_mean": float(np.mean(exact_basis_masses)) if exact_basis_masses else 0.0,
        "projected_basis_response_mass_mean": (
            float(np.mean(projected_basis_masses)) if projected_basis_masses else 0.0
        ),
        "basis_response_projection_fallback_fraction": float(projection_fallbacks / max(int(basis.size), 1)),
        "source_echo_suppression_fraction": float(source_echo_suppressed / max(int(basis.size), 1)),
        "source_echo_mass_mean_before_suppression": (
            float(np.mean(source_echo_masses)) if source_echo_masses else 0.0
        ),
        "basis_response_projection": "local_full_graph_response_projected_to_sparse_cap_basis",
        "probe_steps": int(probe_steps),
        "probe_repairs_per_source": int(probe_repairs_per_source),
        "probe_max_incident_edges": int(probe_max_incident_edges),
        "response_feature_count": 0,
        "gauge_covariant_probe_receipt": True,
        "perturbation_action": "right_multiplication_in_source_endpoint_frame",
    }


def _project_patch_response_to_basis(
    points: np.ndarray,
    basis: np.ndarray,
    response: np.ndarray,
    *,
    source_node: int,
    projection_k: int = 8,
) -> np.ndarray:
    """Compress a full-graph repair response onto the finite cap basis.

    The perturb/repair probe measures on all graph patches, but the finite
    density is represented on a sparse support basis. Directly reading only
    `response[basis]` turns local responses into identity rows whenever the
    changed neighboring patches are not themselves sampled in the basis.
    """

    basis = np.asarray(basis, dtype=np.int64)
    response = np.asarray(response, dtype=float)
    if basis.size == 0:
        return np.zeros(0, dtype=float)
    projected = np.asarray(response[basis], dtype=float).copy()
    active = np.flatnonzero(response > 0.0)
    if active.size:
        basis_points = np.asarray(points[basis], dtype=float)
        active_points = np.asarray(points[active], dtype=float)
        distances = np.linalg.norm(active_points[:, None, :] - basis_points[None, :, :], axis=2)
        k = min(max(1, int(projection_k)), basis.size)
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        nearest_distances = np.take_along_axis(distances, nearest, axis=1)
        basis_spacing = _positive_median(np.linalg.norm(basis_points[:, None, :] - basis_points[None, :, :], axis=2))
        active_spacing = _positive_median(nearest_distances[:, 0])
        sigma = max(0.35 * basis_spacing, active_spacing, 1.0e-12)
        for active_row, neighbors in enumerate(nearest):
            local_distances = nearest_distances[active_row]
            weights = np.exp(-0.5 * np.square(local_distances / sigma))
            weight_sum = float(np.sum(weights))
            if weight_sum <= 1.0e-15:
                continue
            projected[neighbors] += float(response[active[active_row]]) * weights / weight_sum
    if not np.any(projected > 0.0) and 0 <= int(source_node) < points.shape[0]:
        source_distances = np.linalg.norm(np.asarray(points[basis], dtype=float) - np.asarray(points[int(source_node)], dtype=float), axis=1)
        k = min(max(1, int(projection_k)), basis.size)
        nearest = np.argpartition(source_distances, kth=k - 1)[:k]
        weights = np.exp(-0.5 * np.square(source_distances[nearest] / max(_positive_median(source_distances), 1.0e-12)))
        weight_sum = float(np.sum(weights))
        projected[nearest] = weights / max(weight_sum, 1.0e-15)
    return projected


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


def _koopman_feature_series(states: list[dict[str, np.ndarray]], basis: np.ndarray) -> list[np.ndarray]:
    fields = (
        "record_signature",
        "stable_count",
        "committed_mask",
        "repair_load",
        "cumulative_repair_load",
        "local_mismatch_density",
        "modular_depth",
        "modular_time",
        "s3_class_density",
        "s3_sector_class",
    )
    result: list[np.ndarray] = []
    max_index = int(np.max(basis, initial=-1))
    for state in states:
        columns: list[np.ndarray] = []
        for field in fields:
            values = state.get(field)
            if values is None:
                continue
            array = np.asarray(values)
            if array.ndim != 1 or array.size <= max_index:
                continue
            columns.append(_standardize(array[basis].astype(float)))
        if columns:
            result.append(np.column_stack(columns))
    return result


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


def _psd_invsqrt(matrix: np.ndarray, *, cutoff: float = 1.0e-9) -> tuple[np.ndarray, int]:
    matrix = np.asarray(matrix, dtype=float)
    eigvals, eigvecs = np.linalg.eigh((matrix + matrix.T) / 2.0)
    positive = eigvals > float(cutoff)
    inv = np.zeros_like(eigvals, dtype=float)
    inv[positive] = 1.0 / np.sqrt(eigvals[positive])
    return (eigvecs * inv) @ eigvecs.T, int(np.sum(positive))


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
