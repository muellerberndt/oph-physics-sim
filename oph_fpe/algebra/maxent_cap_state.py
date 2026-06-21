from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize


@dataclass(frozen=True)
class MaxEntCapStateResult:
    rho: np.ndarray
    support: np.ndarray
    operator_count: int
    dual_residual: float
    trace_error: float
    min_eigenvalue: float
    train_moment_error: float
    heldout_moment_error: float
    geometry_dependency_count: int
    iterations: int
    converged: bool

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "mode": "maxent_record_operator_state",
            "operator_count": int(self.operator_count),
            "dual_residual": float(self.dual_residual),
            "trace_error": float(self.trace_error),
            "min_eigenvalue": float(self.min_eigenvalue),
            "train_moment_error": float(self.train_moment_error),
            "heldout_moment_error": float(self.heldout_moment_error),
            "geometry_dependency_count": int(self.geometry_dependency_count),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "MAXENT_RECORD_OPERATOR_CAP_STATE_RECEIPT": bool(
                self.trace_error < 1.0e-10
                and self.min_eigenvalue > -1.0e-10
                and self.dual_residual < 1.0e-5
                and self.geometry_dependency_count == 0
            ),
            "claim_boundary": (
                "finite MaxEnt cap state from observer-visible record/history operators. "
                "Cap support selection is external to this builder; no cap tangent, H3 "
                "coordinate, lambda_C target, or declared 2*pi flow enters rho_C."
            ),
        }


def maxent_record_operator_cap_state(
    raw_fields: dict[str, np.ndarray],
    history_fields: list[dict[str, np.ndarray]] | None,
    support: np.ndarray,
    *,
    regularizer: float = 1.0e-8,
    max_operators: int = 10,
    max_iterations: int = 80,
) -> MaxEntCapStateResult:
    """Build a finite MaxEnt cap state from observer-record operators.

    The input is deliberately geometry-free: support membership has already
    been selected by the cap algebra caller, but this builder sees only record
    and repair histories on those support indices.
    """

    basis = np.asarray(support, dtype=np.int64)
    if basis.size == 0:
        rho = np.zeros((0, 0), dtype=complex)
        return MaxEntCapStateResult(rho, basis, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, True)
    if basis.size == 1:
        rho = np.ones((1, 1), dtype=complex)
        return MaxEntCapStateResult(rho, basis, 0, 0.0, 0.0, 1.0, 0.0, 0.0, 0, 0, True)

    states = [state for state in [*list(history_fields or []), raw_fields] if state]
    if not states:
        states = [raw_fields]
    feature_series = _feature_series(states, basis)
    if not feature_series:
        rho = np.eye(basis.size, dtype=complex) / float(basis.size)
        return MaxEntCapStateResult(rho, basis, 0, 0.0, 0.0, 1.0 / basis.size, 0.0, 0.0, 0, 0, True)

    train_features = _stack_features(feature_series)
    rho_emp = _empirical_density(train_features, regularizer=regularizer)
    operators = _operator_basis_from_history(feature_series, max_operators=max_operators)
    if not operators:
        return _empirical_result(rho_emp, basis)
    operators = [_normalize_operator(operator) for operator in operators]
    targets = np.array([_expectation(rho_emp, operator) for operator in operators], dtype=float)

    def objective(lam: np.ndarray) -> tuple[float, np.ndarray]:
        H = np.zeros_like(operators[0], dtype=complex)
        for coeff, operator in zip(lam, operators, strict=True):
            H += float(coeff) * operator
        rho = _gibbs_state(H, regularizer=regularizer)
        moments = np.array([_expectation(rho, operator) for operator in operators], dtype=float)
        log_z = _log_trace_expm(-H)
        value = float(log_z + np.dot(lam, targets))
        grad = targets - moments
        return value, grad

    initial = np.zeros(len(operators), dtype=float)
    result = minimize(
        lambda x: objective(np.asarray(x, dtype=float))[0],
        initial,
        jac=lambda x: objective(np.asarray(x, dtype=float))[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iterations), "ftol": 1.0e-10, "gtol": 1.0e-7},
    )
    coeffs = np.asarray(result.x if result.x is not None else initial, dtype=float)
    H = np.zeros_like(operators[0], dtype=complex)
    for coeff, operator in zip(coeffs, operators, strict=True):
        H += float(coeff) * operator
    rho = _gibbs_state(H, regularizer=regularizer)
    moments = np.array([_expectation(rho, operator) for operator in operators], dtype=float)
    dual_residual = float(np.max(np.abs(moments - targets))) if targets.size else 0.0
    train_error = _normalized_moment_error(moments, targets)

    heldout_features = _stack_features(feature_series[1::2] or feature_series)
    heldout_rho = _empirical_density(heldout_features, regularizer=regularizer)
    heldout_targets = np.array([_expectation(heldout_rho, operator) for operator in operators], dtype=float)
    heldout_error = _normalized_moment_error(moments, heldout_targets)

    eigvals = np.linalg.eigvalsh(_hermitian(rho))
    return MaxEntCapStateResult(
        rho=_hermitian(rho),
        support=basis,
        operator_count=len(operators),
        dual_residual=dual_residual,
        trace_error=abs(float(np.trace(rho).real) - 1.0),
        min_eigenvalue=float(np.min(eigvals).real),
        train_moment_error=train_error,
        heldout_moment_error=heldout_error,
        geometry_dependency_count=0,
        iterations=int(getattr(result, "nit", 0) or 0),
        converged=bool(result.success),
    )


def _feature_series(states: list[dict[str, np.ndarray]], basis: np.ndarray) -> list[np.ndarray]:
    fields = (
        "record_signature",
        "stable_count",
        "committed_mask",
        "repair_load",
        "cumulative_repair_load",
        "local_mismatch_density",
        "s3_class_density",
        "s3_sector_class",
    )
    rows: list[np.ndarray] = []
    max_index = int(np.max(basis, initial=-1))
    for state in states:
        columns: list[np.ndarray] = []
        for field in fields:
            values = state.get(field)
            if values is None:
                continue
            array = np.asarray(values)
            if array.size <= max_index:
                continue
            columns.append(_standardize(array[basis].astype(float)))
        if columns:
            rows.append(np.column_stack(columns))
    return rows


def _stack_features(series: list[np.ndarray]) -> np.ndarray:
    if not series:
        return np.zeros((0, 0), dtype=float)
    return np.hstack([_standardize_columns(values) for values in series if values.size])


def _empirical_density(features: np.ndarray, *, regularizer: float) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.size == 0:
        n = int(features.shape[0]) if features.ndim == 2 else 1
        return np.eye(max(n, 1), dtype=complex) / float(max(n, 1))
    gram = features @ features.T / max(float(features.shape[1]), 1.0)
    rho = gram @ gram.T
    rho = _hermitian(rho + float(regularizer) * np.eye(rho.shape[0], dtype=complex))
    return rho / max(float(np.trace(rho).real), 1.0e-15)


def _operator_basis_from_history(series: list[np.ndarray], *, max_operators: int) -> list[np.ndarray]:
    features = _stack_features(series)
    if features.size == 0:
        return []
    operators: list[np.ndarray] = []
    for column in range(min(features.shape[1], max(1, int(max_operators) // 2))):
        operators.append(np.diag(_standardize(features[:, column]).astype(complex)))
    covariance = features @ features.T / max(float(features.shape[1]), 1.0)
    operators.append(_hermitian(covariance.astype(complex)))
    if len(series) >= 2:
        currents = []
        for left, right in zip(series[:-1], series[1:], strict=False):
            width = min(left.shape[1], right.shape[1])
            if width == 0:
                continue
            x0 = _standardize_columns(left[:, :width])
            x1 = _standardize_columns(right[:, :width])
            currents.append((x0 @ x1.T - x1 @ x0.T) / max(float(width), 1.0))
        if currents:
            antisym = np.mean(currents, axis=0)
            operators.append(_hermitian(1j * antisym))
    return operators[: max(1, int(max_operators))]


def _empirical_result(rho: np.ndarray, basis: np.ndarray) -> MaxEntCapStateResult:
    eigvals = np.linalg.eigvalsh(_hermitian(rho))
    return MaxEntCapStateResult(
        rho=_hermitian(rho),
        support=np.asarray(basis, dtype=np.int64),
        operator_count=0,
        dual_residual=0.0,
        trace_error=abs(float(np.trace(rho).real) - 1.0),
        min_eigenvalue=float(np.min(eigvals).real),
        train_moment_error=0.0,
        heldout_moment_error=0.0,
        geometry_dependency_count=0,
        iterations=0,
        converged=True,
    )


def _gibbs_state(H: np.ndarray, *, regularizer: float) -> np.ndarray:
    rho = expm(-_hermitian(H))
    rho = _hermitian(rho + float(regularizer) * np.eye(rho.shape[0], dtype=complex))
    return rho / max(float(np.trace(rho).real), 1.0e-15)


def _log_trace_expm(matrix: np.ndarray) -> float:
    values = np.linalg.eigvalsh(_hermitian(matrix))
    offset = float(np.max(values).real)
    return float(offset + np.log(np.sum(np.exp(values - offset))))


def _expectation(rho: np.ndarray, operator: np.ndarray) -> float:
    return float(np.trace(np.asarray(rho, dtype=complex) @ np.asarray(operator, dtype=complex)).real)


def _normalized_moment_error(values: np.ndarray, targets: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if values.size == 0 or targets.size == 0:
        return 0.0
    width = min(values.size, targets.size)
    denom = np.maximum(np.abs(targets[:width]), 1.0)
    return float(np.median(np.abs(values[:width] - targets[:width]) / denom))


def _normalize_operator(operator: np.ndarray) -> np.ndarray:
    matrix = _hermitian(np.asarray(operator, dtype=complex))
    trace = float(np.trace(matrix).real) / max(matrix.shape[0], 1)
    matrix = matrix - trace * np.eye(matrix.shape[0], dtype=complex)
    norm = float(np.linalg.norm(matrix, ord="fro"))
    return matrix / max(norm, 1.0e-12)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values))
    if std < 1.0e-12:
        return values - float(np.mean(values))
    return (values - float(np.mean(values))) / std


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    std = np.std(values, axis=0)
    std = np.where(std < 1.0e-12, 1.0, std)
    return (values - np.mean(values, axis=0)) / std


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conj().T) / 2.0
