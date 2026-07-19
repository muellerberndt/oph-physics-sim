from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from oph_fpe.claims import CONTINUATION, with_claim_metadata
from oph_fpe.evidence.artifact_paths import companion_input_packet_path
from oph_fpe.evidence.validation import utf8_byte_length


MAX_MARKOV_STATES = 512
MAX_HORIZON_STEPS = 1_000_000
MAX_MARKOV_MULTIPLY_WORK = 5_000_000
MAX_MARKOV_CUBIC_WORK = 20_000_000
MAX_FAIR_BLOCK_PACKET_BYTES = 10_000_000
MAX_FAIR_BLOCK_PACKET_NODES = 300_000
MAX_FAIR_BLOCK_PACKET_DEPTH = 8


def fair_block_consensus_certificate(
    *,
    transition_matrix: Sequence[Sequence[float]] | None = None,
    initial_distribution: Sequence[float] | None = None,
    fair_states: Sequence[int] | None = None,
    time_horizon_steps: int | None = None,
    lambda_contraction: float | None = None,
    epsilon_noise: float | None = None,
    beta: float | None = None,
    lipschitz_L: float | None = None,
    block_count: int | None = None,
    active_fraction: float | None = None,
) -> dict[str, Any]:
    """Recompute a finite-horizon fair-block certificate from a Markov kernel.

    Legacy scalar constants are retained as nonpromoting declarations.  A
    positive receipt requires a concrete row-stochastic kernel, an initial law,
    an explicit fair-state set and a finite horizon.  The certificate reports
    stationary, mixing and expected occupation quantities; it never labels the
    result as permanent settling under persistent noise.
    """

    blockers: list[str] = []
    legacy = {
        "lambda_contraction": _finite_scalar(lambda_contraction),
        "epsilon_noise": _finite_scalar(epsilon_noise),
        "beta": _finite_scalar(beta),
        "lipschitz_L": _finite_scalar(lipschitz_L),
        "block_count": _integer_or_none(block_count),
        "active_fraction": _finite_scalar(active_fraction),
    }
    legacy_declaration_present = any(value is not None for value in legacy.values())

    matrix = _matrix_or_none(transition_matrix)
    if matrix is None:
        blockers.append("finite_markov_kernel_missing_or_invalid")
        state_count = 0
    else:
        state_count = int(matrix.shape[0])
        if state_count == 0 or state_count > MAX_MARKOV_STATES:
            blockers.append("markov_state_budget_exceeded")
        if matrix.shape[1] != state_count:
            blockers.append("markov_kernel_not_square")
        elif np.any(matrix < 0.0) or not np.array_equal(
            matrix.sum(axis=1), np.ones(state_count)
        ):
            blockers.append("markov_kernel_not_row_stochastic")

    initial = _probability_vector_or_none(initial_distribution, state_count)
    if initial is None:
        blockers.append("initial_distribution_missing_or_invalid")

    horizon = _positive_integer_or_none(time_horizon_steps)
    if horizon is None or horizon > MAX_HORIZON_STEPS:
        blockers.append("finite_time_horizon_missing_or_out_of_budget")
    elif state_count and state_count * state_count * horizon > MAX_MARKOV_MULTIPLY_WORK:
        blockers.append("finite_markov_operation_budget_exceeded")

    fair = _fair_state_set(fair_states, state_count)
    if fair is None:
        blockers.append("fair_state_set_missing_or_invalid")

    kernel_valid = not any(
        blocker
        in {
            "finite_markov_kernel_missing_or_invalid",
            "markov_state_budget_exceeded",
            "markov_kernel_not_square",
            "markov_kernel_not_row_stochastic",
        }
        for blocker in blockers
    )
    irreducible = bool(kernel_valid and matrix is not None and _is_irreducible(matrix))
    if kernel_valid and not irreducible:
        blockers.append("markov_kernel_not_irreducible")
    cubic_work = state_count * state_count * state_count
    if kernel_valid and cubic_work > MAX_MARKOV_CUBIC_WORK:
        blockers.append("stationary_and_contraction_operation_budget_exceeded")

    stationary: np.ndarray | None = None
    stationary_residual: float | None = None
    contraction: float | None = None
    if (
        kernel_valid
        and matrix is not None
        and cubic_work <= MAX_MARKOV_CUBIC_WORK
    ):
        stationary = _stationary_distribution(matrix)
        stationary_residual = float(np.max(np.abs(stationary @ matrix - stationary)))
        contraction = _dobrushin_coefficient(matrix)
        if stationary_residual > 1.0e-10:
            blockers.append("stationary_distribution_residual_too_large")
        if contraction >= 1.0:
            blockers.append("strict_contraction_not_established")

    final_distribution: np.ndarray | None = None
    occupation_fraction: float | None = None
    final_tv_to_stationary: float | None = None
    stationary_fair_mass: float | None = None
    max_distribution_renormalization = 0.0
    if (
        not blockers
        and matrix is not None
        and initial is not None
        and stationary is not None
        and fair is not None
        and horizon is not None
    ):
        distribution = initial.copy()
        fair_mass_sum = 0.0
        # Repeated vector multiplication is deliberate: it produces the
        # finite-horizon expectation actually certified by this receipt.
        for _ in range(horizon):
            fair_mass_sum += float(distribution[list(fair)].sum())
            next_distribution = distribution @ matrix
            total = float(next_distribution.sum())
            if (
                not np.all(np.isfinite(next_distribution))
                or np.any(next_distribution < -1.0e-15)
                or not math.isfinite(total)
                or total <= 0.0
            ):
                blockers.append("finite_markov_propagation_left_probability_simplex")
                break
            max_distribution_renormalization = max(
                max_distribution_renormalization,
                abs(total - 1.0),
            )
            distribution = np.maximum(next_distribution, 0.0) / total
        if not blockers:
            final_distribution = distribution
            occupation_fraction = min(
                1.0,
                max(0.0, fair_mass_sum / float(horizon)),
            )
            final_tv_to_stationary = min(
                1.0,
                max(
                    0.0,
                    0.5 * float(np.abs(distribution - stationary).sum()),
                ),
            )
            stationary_fair_mass = min(
                1.0,
                max(0.0, float(stationary[list(fair)].sum())),
            )

    recomputation_receipt = not blockers
    consensus_blockers = [
        "run_bound_fair_state_semantics_and_acceptance_threshold_unavailable"
    ]
    report = {
        "schema_version": "finite-markov-fair-block-v2",
        "mode": "finite_markov_fair_block_recomputation",
        "probability_mode": "finite_horizon_expectation",
        "FAIR_BLOCK_MARKOV_RECOMPUTATION_RECEIPT": recomputation_receipt,
        "FAIR_BLOCK_CONSENSUS_CERTIFICATE": False,
        "PERMANENT_SETTLING_RECEIPT": False,
        "ALL_TIME_TUBE_RECEIPT": False,
        "RUN_ARTIFACT_BINDING_RECEIPT": False,
        "receipt": False,
        "time_horizon_steps": horizon,
        "state_count": state_count,
        "estimated_cubic_work": cubic_work,
        "max_distribution_renormalization": max_distribution_renormalization,
        "fair_states": sorted(fair) if fair is not None else [],
        "irreducible": irreducible,
        "dobrushin_contraction": contraction,
        "stationary_distribution": stationary.tolist() if stationary is not None else None,
        "stationary_residual": stationary_residual,
        "stationary_fair_mass": stationary_fair_mass,
        "initial_distribution": initial.tolist() if initial is not None else None,
        "final_distribution": final_distribution.tolist() if final_distribution is not None else None,
        "final_total_variation_to_stationary": final_tv_to_stationary,
        "expected_fair_occupation_fraction": occupation_fraction,
        "legacy_declared_constants": legacy,
        "legacy_declaration_present": legacy_declaration_present,
        "legacy_declarations_promoted": False,
        "blockers": blockers,
        "consensus_blockers": consensus_blockers,
        "claim_boundary": (
            "recomputed finite-horizon expectation, stationary law and mixing diagnostics for the supplied "
            "finite Markov kernel; the caller-selected fair-state labels have no authenticated block/activity "
            "semantics or acceptance threshold, so they do not earn a fair-block consensus certificate; "
            "persistent stochastic noise also forbids an all-time or permanent settling claim"
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=CONTINUATION,
        receipt="FAIR_BLOCK_CONSENSUS_CERTIFICATE",
    )


def write_fair_block_certificate(
    path: Path,
    packet: dict[str, Any],
) -> dict[str, Any]:
    """Write a replayable finite-Markov arithmetic certificate."""

    expected_fields = {
        "transition_matrix",
        "initial_distribution",
        "fair_states",
        "time_horizon_steps",
    }
    if not isinstance(packet, dict) or set(packet) != expected_fields:
        raise ValueError("fair-block packet has missing or unknown fields")
    encoded = _bounded_packet_json_bytes(packet)
    report = fair_block_consensus_certificate(
        transition_matrix=packet["transition_matrix"],
        initial_distribution=packet["initial_distribution"],
        fair_states=packet["fair_states"],
        time_horizon_steps=packet["time_horizon_steps"],
    )
    report["input_packet_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        destination = destination / "fair_block_certificate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    replay_path = companion_input_packet_path(
        destination,
        canonical_certificate_filename="fair_block_certificate.json",
        canonical_input_filename="fair_block_input_packet.json",
    )
    replay_path.write_text(
        json.dumps(packet, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _matrix_or_none(value: Sequence[Sequence[float]] | None) -> np.ndarray | None:
    if value is None:
        return None
    row_count = _bounded_sequence_length(value, MAX_MARKOV_STATES)
    if row_count is None:
        return None
    try:
        for row in value:
            if _bounded_sequence_length(row, MAX_MARKOV_STATES) is None:
                return None
            if any(isinstance(item, (bool, np.bool_)) for item in row):
                return None
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            return None
        matrix = np.asarray(value, dtype=float)
    except (OverflowError, TypeError, ValueError):
        return None
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        return None
    return matrix


def _probability_vector_or_none(value: Sequence[float] | None, size: int) -> np.ndarray | None:
    if value is None:
        return None
    if _bounded_sequence_length(value, MAX_MARKOV_STATES) is None:
        return None
    try:
        if any(isinstance(item, (bool, np.bool_)) for item in value):
            return None
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            return None
        vector = np.asarray(value, dtype=float)
    except (OverflowError, TypeError, ValueError):
        return None
    if (
        vector.shape != (size,)
        or not np.all(np.isfinite(vector))
        or np.any(vector < 0.0)
        or np.any(vector > 1.0)
        or float(vector.sum()) != 1.0
    ):
        return None
    return vector


def _fair_state_set(value: Sequence[int] | None, size: int) -> set[int] | None:
    length = _bounded_sequence_length(value, MAX_MARKOV_STATES)
    if length is None or length == 0:
        return None
    states: set[int] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < size:
            return None
        states.add(item)
    return states or None


def _bounded_sequence_length(value: Any, limit: int) -> int | None:
    if isinstance(value, (str, bytes, bytearray, dict)):
        return None
    try:
        length = len(value)
    except (TypeError, OverflowError):
        return None
    return length if 0 <= length <= limit else None


def _bounded_packet_json_bytes(value: Any) -> bytes:
    pending: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > MAX_FAIR_BLOCK_PACKET_NODES:
            raise ValueError("fair-block packet exceeds the JSON node budget")
        if depth > MAX_FAIR_BLOCK_PACKET_DEPTH:
            raise ValueError("fair-block packet exceeds the JSON depth budget")
        if item is None or isinstance(item, (bool, int)):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("fair-block packet contains a nonfinite number")
            continue
        if isinstance(item, str):
            byte_length = utf8_byte_length(item)
            if byte_length is None or byte_length > 1024:
                raise ValueError("fair-block packet contains an oversized string")
            continue
        if isinstance(item, dict):
            if len(item) > 16 or not all(isinstance(key, str) for key in item):
                raise ValueError("fair-block packet contains a non-string mapping key")
            pending.extend((child, depth + 1) for child in item.values())
            continue
        if isinstance(item, (list, tuple)):
            if len(item) > MAX_MARKOV_STATES:
                raise ValueError("fair-block packet contains an oversized array")
            pending.extend((child, depth + 1) for child in item)
            continue
        raise ValueError("fair-block packet contains unsupported JSON data")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("fair-block packet is not canonical JSON data") from exc
    if len(encoded) > MAX_FAIR_BLOCK_PACKET_BYTES:
        raise ValueError("fair-block packet exceeds the byte budget")
    return encoded


def _positive_integer_or_none(value: int | None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _integer_or_none(value: int | None) -> int | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _finite_scalar(value: float | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _is_irreducible(matrix: np.ndarray) -> bool:
    adjacency = matrix > 0.0
    size = matrix.shape[0]

    def reachable(graph: np.ndarray) -> bool:
        seen = {0}
        stack = [0]
        while stack:
            node = stack.pop()
            for neighbor in np.flatnonzero(graph[node]):
                target = int(neighbor)
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        return len(seen) == size

    return bool(size and reachable(adjacency) and reachable(adjacency.T))


def _stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    size = matrix.shape[0]
    lhs = np.vstack((matrix.T - np.eye(size), np.ones(size)))
    rhs = np.concatenate((np.zeros(size), np.ones(1)))
    stationary, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    stationary = np.maximum(stationary, 0.0)
    return stationary / float(stationary.sum())


def _dobrushin_coefficient(matrix: np.ndarray) -> float:
    maximum = 0.0
    for row in matrix:
        maximum = max(
            maximum,
            float(np.max(np.abs(matrix - row).sum(axis=1))),
        )
    return 0.5 * maximum
