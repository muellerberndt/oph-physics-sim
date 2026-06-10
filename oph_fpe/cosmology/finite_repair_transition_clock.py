from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from oph_fpe.constants.oph_pixel import OPHPixelConstants, P_STAR
from oph_fpe.cosmology.scalar_repair_semigroup import ScalarRepairSemigroupSpec, scalar_repair_semigroup_report


DEFAULT_PACKET_FIELDS = ("checkpoint_class", "stable_flag", "s3_sector_class", "repair_load_bucket")


@dataclass(frozen=True)
class FiniteRepairTransitionClockConfig:
    packet_fields: tuple[str, ...] = DEFAULT_PACKET_FIELDS
    primary_matrix: str = "raw_empirical"
    repair_step_time: float = 1.0
    weight_field: str = "transition_history_mean_modal_mass"
    min_transition_count: int = 1
    p_value: float = P_STAR


def write_finite_repair_transition_clock_report(
    run_dir: Path,
    out_dir: Path,
    *,
    packet_fields: tuple[str, ...] = DEFAULT_PACKET_FIELDS,
    primary_matrix: str = "raw_empirical",
    repair_step_time: float = 1.0,
    weight_field: str = "transition_history_mean_modal_mass",
    min_transition_count: int = 1,
    p_value: float = P_STAR,
) -> dict[str, Any]:
    report, matrices = finite_repair_transition_clock_report(
        run_dir,
        FiniteRepairTransitionClockConfig(
            packet_fields=tuple(packet_fields),
            primary_matrix=primary_matrix,
            repair_step_time=repair_step_time,
            weight_field=weight_field,
            min_transition_count=min_transition_count,
            p_value=p_value,
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "finite_repair_transition_matrix.npz",
        counts=matrices["counts"],
        raw_empirical=matrices["raw_empirical"],
        reversible_empirical=matrices["reversible_empirical"],
        state_labels=np.asarray(report["state_labels"], dtype=object),
    )
    (out_dir / "finite_repair_transition_matrix_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    _write_matrix_rows(out_dir / "finite_repair_transition_rows.csv", report)

    scalar = scalar_repair_semigroup_report(
        ScalarRepairSemigroupSpec(
            dimension=max(int(report["state_count"]), 2),
            kappa_rep=float(report["primary"]["kappa_rep_estimate"]),
            source="finite_state_transition_matrix",
            finite_lattice_derived=bool(report["finite_lattice_derived"]),
            matrix_source=str(out_dir / "finite_repair_transition_matrix.npz"),
            p_value=p_value,
        )
    )
    scalar["transition_matrix_certificate"] = {
        "source_report": str(out_dir / "finite_repair_transition_matrix_report.json"),
        "primary_matrix": report["primary_matrix"],
        "matrix_ready": bool(report["finite_transition_matrix_ready"]),
        "clock_normalization_certified": bool(report["clock_normalization_certified"]),
        "required_repair_step_time_for_kappa_e": report["primary"].get(
            "required_repair_step_time_for_kappa_e"
        ),
        "clock_normalization_candidates": report.get("clock_normalization_candidates", []),
        "primary_lambda_2": report["primary"].get("lambda_2"),
        "primary_gamma": report["primary"].get("gamma_continuous"),
    }
    scalar["eligible_for_repair_clock_certificate"] = bool(
        scalar["eligible_for_repair_clock_certificate"] and report["clock_normalization_certified"]
    )
    scalar["repair_clock_certificate"] = bool(
        scalar["repair_clock_certificate"] and report["clock_normalization_certified"]
    )
    (out_dir / "scalar_repair_semigroup_report.json").write_text(
        json.dumps(scalar, indent=2, default=str),
        encoding="utf-8",
    )
    return report


def finite_repair_transition_clock_report(
    run_dir: Path,
    config: FiniteRepairTransitionClockConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    config = config or FiniteRepairTransitionClockConfig()
    if config.primary_matrix not in {"raw_empirical", "reversible_empirical"}:
        raise ValueError("primary_matrix must be raw_empirical or reversible_empirical")
    if config.repair_step_time <= 0.0:
        raise ValueError("repair_step_time must be positive")
    observer_path = Path(run_dir) / "observer_views.jsonl"
    if not observer_path.exists():
        raise FileNotFoundError(observer_path)

    counts, labels, observer_count, transition_count, skipped = _transition_counts_from_observer_views(
        observer_path,
        fields=config.packet_fields,
        weight_field=config.weight_field,
        min_transition_count=config.min_transition_count,
    )
    raw = _row_stochastic(counts)
    reversible = _reversible_projection(counts)
    pixel = OPHPixelConstants(P=float(config.p_value))
    delta_p = float(pixel.P - pixel.phi)
    target_kappa = math.e
    target_eta = float(target_kappa * delta_p)
    raw_summary = _matrix_summary(raw, delta_p=delta_p, repair_step_time=config.repair_step_time)
    reversible_summary = _matrix_summary(
        reversible,
        delta_p=delta_p,
        repair_step_time=config.repair_step_time,
    )
    matrix_summaries = {
        "raw_empirical": raw_summary,
        "reversible_empirical": reversible_summary,
    }
    primary = matrix_summaries[config.primary_matrix]
    finite_ready = bool(
        transition_count > 0
        and len(labels) >= 2
        and primary.get("finite")
        and primary.get("lambda_2") is not None
    )
    rel_error = abs(float(primary["kappa_rep_estimate"]) - target_kappa) / target_kappa if finite_ready else None
    clock_certified = bool(finite_ready and rel_error is not None and rel_error <= 0.05)
    report = {
        "mode": "oph_finite_repair_transition_clock_v0",
        "source_run_dir": str(Path(run_dir)),
        "observer_views_path": str(observer_path),
        "packet_fields": list(config.packet_fields),
        "primary_matrix": config.primary_matrix,
        "repair_step_time": float(config.repair_step_time),
        "weight_field": config.weight_field,
        "observer_count": int(observer_count),
        "transition_count": int(transition_count),
        "skipped_observer_count": int(skipped),
        "state_count": int(len(labels)),
        "state_labels": [json.dumps(label, sort_keys=True) for label in labels],
        "target": {
            "required_kappa_rep": target_kappa,
            "required_eta_R": target_eta,
            "required_n_s": 1.0 - target_eta,
            "P": float(pixel.P),
            "phi": float(pixel.phi),
            "delta_P": delta_p,
        },
        "matrices": matrix_summaries,
        "primary": primary,
        "clock_normalization_candidates": _clock_normalization_candidates(
            Path(run_dir),
            required_step_time=primary.get("required_repair_step_time_for_kappa_e"),
        ),
        "relative_error_to_kappa_e": rel_error,
        "finite_transition_matrix_ready": finite_ready,
        "finite_lattice_derived": finite_ready,
        "clock_normalization_certified": clock_certified,
        "repair_clock_certificate": clock_certified,
        "eta_R_finite_lattice_derived": clock_certified,
        "physical_cmb_prediction": False,
        "blockers": _blockers(finite_ready, clock_certified, primary, config.primary_matrix),
        "claim_boundary": (
            "Finite observer-visible transition-matrix clock diagnostic. Packet paths are read from "
            "observer_views.jsonl and projected to a declared support-visible quotient alphabet. The report "
            "is finite-lattice-derived as a transition-matrix diagnostic, but it certifies the exact CMB "
            "repair clock only if the primary matrix yields kappa_rep=e under a predeclared repair-step time."
        ),
    }
    return report, {"counts": counts, "raw_empirical": raw, "reversible_empirical": reversible}


def _transition_counts_from_observer_views(
    path: Path,
    *,
    fields: tuple[str, ...],
    weight_field: str,
    min_transition_count: int,
) -> tuple[np.ndarray, list[tuple[tuple[str, int], ...]], int, int, int]:
    state_to_idx: dict[tuple[tuple[str, int], ...], int] = {}
    counts: Counter[tuple[int, int]] = Counter()
    observer_count = 0
    transition_count = 0
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            observer_count += 1
            view = json.loads(line)
            descriptor = view.get("transition_history_descriptor") or {}
            steps = descriptor.get("steps") or []
            if len(steps) < 2:
                skipped += 1
                continue
            weight = _float_or_default(view.get(weight_field), 1.0)
            encoded = []
            for step in steps:
                key = tuple((field, int(step.get(field, 0))) for field in fields)
                encoded.append(state_to_idx.setdefault(key, len(state_to_idx)))
            local_transition_count = 0
            for left, right in zip(encoded, encoded[1:], strict=False):
                counts[(left, right)] += float(weight)
                transition_count += 1
                local_transition_count += 1
            if local_transition_count < min_transition_count:
                skipped += 1
    labels = [None] * len(state_to_idx)
    for key, idx in state_to_idx.items():
        labels[idx] = key
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for (left, right), value in counts.items():
        matrix[left, right] += float(value)
    return matrix, labels, observer_count, transition_count, skipped


def _row_stochastic(counts: np.ndarray) -> np.ndarray:
    if counts.size == 0:
        return counts.copy()
    row_sum = counts.sum(axis=1)
    matrix = np.zeros_like(counts, dtype=np.float64)
    active = row_sum > 0.0
    matrix[active] = counts[active] / row_sum[active, None]
    for idx, is_active in enumerate(active):
        if not is_active:
            matrix[idx, idx] = 1.0
    return matrix


def _reversible_projection(counts: np.ndarray) -> np.ndarray:
    if counts.size == 0:
        return counts.copy()
    symmetric = 0.5 * (counts + counts.T)
    if np.all(symmetric.sum(axis=1) <= 0.0):
        return np.eye(counts.shape[0], dtype=np.float64)
    return _row_stochastic(symmetric)


def _matrix_summary(matrix: np.ndarray, *, delta_p: float, repair_step_time: float) -> dict[str, Any]:
    if matrix.size == 0:
        return {"finite": False, "lambda_2": None, "gamma_continuous": None}
    row_error = float(np.max(np.abs(matrix.sum(axis=1) - 1.0))) if matrix.shape[0] else 0.0
    irreducible = _is_strongly_connected(matrix > 1.0e-12)
    aperiodic = bool(irreducible and np.any(np.diag(matrix) > 1.0e-12))
    vals = np.linalg.eigvals(matrix)
    sorted_abs = sorted((float(abs(value)) for value in vals), reverse=True)
    lambda_2 = sorted_abs[1] if len(sorted_abs) > 1 else 0.0
    gamma = -math.log(max(float(lambda_2), 1.0e-12)) / float(repair_step_time)
    kappa = gamma / max(delta_p, 1.0e-30)
    required_dt = gamma * repair_step_time / max(math.e * delta_p, 1.0e-30)
    stationary = _stationary_distribution(matrix)
    detailed_balance_error = _detailed_balance_error(matrix, stationary)
    return {
        "finite": bool(np.isfinite(gamma) and np.isfinite(lambda_2)),
        "irreducible": irreducible,
        "aperiodic": aperiodic,
        "row_sum_max_abs_error": row_error,
        "lambda_2": float(lambda_2),
        "gamma_continuous": float(gamma),
        "gamma_discrete_one_minus_lambda2": float(1.0 - lambda_2),
        "kappa_rep_estimate": float(kappa),
        "eta_R_estimate": float(kappa * delta_p),
        "n_s_estimate": float(1.0 - kappa * delta_p),
        "required_repair_step_time_for_kappa_e": float(required_dt),
        "stationary_min": float(np.min(stationary)) if stationary.size else None,
        "stationary_max": float(np.max(stationary)) if stationary.size else None,
        "detailed_balance_max_abs_error": detailed_balance_error,
        "top_abs_eigenvalues": [float(value) for value in sorted_abs[:8]],
    }


def _stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float64)
    vals, vecs = np.linalg.eig(matrix.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    vector = np.real(vecs[:, idx])
    vector = np.abs(vector)
    total = float(np.sum(vector))
    if total <= 1.0e-30:
        return np.full(matrix.shape[0], 1.0 / max(matrix.shape[0], 1), dtype=np.float64)
    return vector / total


def _detailed_balance_error(matrix: np.ndarray, stationary: np.ndarray) -> float | None:
    if matrix.size == 0 or stationary.size == 0:
        return None
    flow = stationary[:, None] * matrix
    return float(np.max(np.abs(flow - flow.T)))


def _blockers(finite_ready: bool, clock_certified: bool, primary: dict[str, Any], primary_matrix: str) -> list[str]:
    blockers: list[str] = []
    if not finite_ready:
        blockers.append("finite transition matrix is missing, degenerate, or numerically invalid")
    if finite_ready and not primary.get("irreducible", False):
        blockers.append("primary transition matrix is reducible, so it is not a finite repair-matrix certificate")
    if finite_ready and not primary.get("aperiodic", False):
        blockers.append("primary transition matrix is not certified aperiodic")
    if primary_matrix == "raw_empirical" and primary.get("detailed_balance_max_abs_error", 0.0) not in (None, 0.0):
        blockers.append("raw empirical matrix is not a reversible/GNS self-adjoint repair operator")
    if finite_ready and not clock_certified:
        blockers.append("finite transition matrix does not yield kappa_rep=e under the declared repair-step time")
    return blockers


def _clock_normalization_candidates(run_dir: Path, *, required_step_time: Any) -> list[dict[str, Any]]:
    required = _float_or_none(required_step_time)
    config = _read_yaml(run_dir / "config.yml")
    dynamics = config.get("dynamics", {}) if isinstance(config, dict) else {}
    bw = config.get("bw", {}) if isinstance(config, dict) else {}
    observer_objects = config.get("observer_objects", {}) if isinstance(config, dict) else {}
    cycles = _float_or_none(dynamics.get("cycles"))
    commit = _float_or_none(dynamics.get("record_commit_cycles"))
    history = _float_or_none(observer_objects.get("history_window"))
    times = bw.get("times") if isinstance(bw, dict) else None
    bw_time = _float_or_none(times[0]) if isinstance(times, list) and times else _float_or_none(bw.get("transition_response_time"))
    bw_scale = _float_or_none(bw.get("transition_response_scale")) or (2.0 * math.pi)
    bw_s = bw_time * bw_scale if bw_time is not None and bw_scale is not None else None
    candidates: list[tuple[str, float | None, str]] = [
        ("unit_step", 1.0, "one simulator transition-history step"),
        ("record_commit_cycles", commit, "record commit horizon in cycles"),
        ("history_window", history, "observer transition-history window"),
        ("record_commit_cycles_times_2pi", commit * 2.0 * math.pi if commit is not None else None, "KMS 2pi times record commit horizon"),
        ("history_window_times_2pi", history * 2.0 * math.pi if history is not None else None, "KMS 2pi times observer history window"),
        (
            "cycles_times_bw_modular_time",
            cycles * bw_s if cycles is not None and bw_s is not None else None,
            "simulation cycles times BW modular parameter s=2pi*t",
        ),
        (
            "commit_times_history_times_bw_modular_time",
            commit * history * bw_s if commit is not None and history is not None and bw_s is not None else None,
            "record commit horizon times history window times BW modular parameter",
        ),
    ]
    rows = []
    for name, value, description in candidates:
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            continue
        rel_error = abs(float(value) - required) / max(abs(required), 1.0e-30) if required is not None else None
        rows.append(
            {
                "name": name,
                "value": float(value),
                "relative_error_to_required": rel_error,
                "description": description,
            }
        )
    rows.sort(key=lambda row: float("inf") if row["relative_error_to_required"] is None else row["relative_error_to_required"])
    return rows


def _is_strongly_connected(adjacency: np.ndarray) -> bool:
    if adjacency.size == 0:
        return False
    n = int(adjacency.shape[0])
    if n == 1:
        return True
    return len(_reachable(adjacency, 0)) == n and len(_reachable(adjacency.T, 0)) == n


def _reachable(adjacency: np.ndarray, start: int) -> set[int]:
    seen = {int(start)}
    stack = [int(start)]
    while stack:
        node = stack.pop()
        for nxt in np.flatnonzero(adjacency[node]):
            nxt = int(nxt)
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _write_matrix_rows(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for name, summary in report.get("matrices", {}).items():
        rows.append(
            {
                "matrix": name,
                "lambda_2": summary.get("lambda_2"),
                "gamma_continuous": summary.get("gamma_continuous"),
                "kappa_rep_estimate": summary.get("kappa_rep_estimate"),
                "eta_R_estimate": summary.get("eta_R_estimate"),
                "required_repair_step_time_for_kappa_e": summary.get(
                    "required_repair_step_time_for_kappa_e"
                ),
                "detailed_balance_max_abs_error": summary.get("detailed_balance_max_abs_error"),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["matrix"])
        writer.writeheader()
        writer.writerows(rows)


def _float_or_default(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, IndexError):
        return None
    return result if math.isfinite(result) else None


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}
