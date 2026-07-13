from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from typing import Any

import numpy as np
from scipy.special import sph_harm_y_all


def angular_power_report(
    points: np.ndarray,
    fields: dict[str, np.ndarray],
    *,
    ell_max: int = 64,
    pair_samples: int = 200_000,
    seed: int = 1,
    controls: list[str] | None = None,
    estimator: str = "spherical_harmonic",
    measure_weights: np.ndarray | None = None,
    harmonic_batch_size: int = 4096,
    exact_pair_limit: int = 1_500_000,
    n_jobs: int | str | None = 1,
) -> dict[str, Any]:
    """Estimate screen angular spectra for finite equal-area screen regulators.

    The default direct harmonic estimator computes nonnegative auto-power from
    finite spherical-harmonic coefficients. The sampled Legendre mode is retained
    as a lightweight legacy proxy and diagnostic fallback.
    """

    rng = np.random.default_rng(seed)
    if controls is None:
        controls = ["shuffled_field", "random_gaussian"]
    estimator = str(estimator)
    spectrum_fn = _spectrum_estimator(estimator)
    resolved_jobs = _resolve_n_jobs(n_jobs)
    tasks: list[dict[str, Any]] = []
    task_id = 0
    prepared_controls: dict[str, dict[str, np.ndarray]] = {}
    target_task_ids: dict[str, int] = {}
    control_task_ids: dict[tuple[str, str], int] = {}
    for name, values in fields.items():
        values = np.asarray(values)
        target_task_ids[name] = task_id
        tasks.append(
            {
                "task_id": task_id,
                "values": values,
                "rng_seed": int(rng.integers(0, np.iinfo(np.uint32).max)),
            }
        )
        task_id += 1
        prepared_controls[name] = {}
        if "shuffled_field" in controls:
            shuffled = np.array(values, copy=True)
            rng.shuffle(shuffled)
            prepared_controls[name]["shuffled_field"] = shuffled
            control_task_ids[(name, "shuffled_field")] = task_id
            tasks.append(
                {
                    "task_id": task_id,
                    "values": shuffled,
                    "rng_seed": int(rng.integers(0, np.iinfo(np.uint32).max)),
                }
            )
            task_id += 1
        if "random_gaussian" in controls:
            random_values = rng.normal(size=values.shape[0])
            prepared_controls[name]["random_gaussian"] = random_values
            control_task_ids[(name, "random_gaussian")] = task_id
            tasks.append(
                {
                    "task_id": task_id,
                    "values": random_values,
                    "rng_seed": int(rng.integers(0, np.iinfo(np.uint32).max)),
                }
            )
            task_id += 1

    spectra = _run_spectrum_tasks(
        tasks,
        points=points,
        spectrum_fn=spectrum_fn,
        ell_max=ell_max,
        pair_samples=pair_samples,
        measure_weights=measure_weights,
        harmonic_batch_size=harmonic_batch_size,
        exact_pair_limit=exact_pair_limit,
        n_jobs=resolved_jobs,
    )

    report_fields: dict[str, Any] = {}
    report_controls: dict[str, Any] = {}
    for name, values in fields.items():
        spectrum = spectra[target_task_ids[name]]
        report_fields[name] = _summarize_spectrum(spectrum)
        report_fields[name]["spectrum"] = spectrum
        report_controls[name] = {}
        if "shuffled_field" in controls:
            control_spectrum = spectra[control_task_ids[(name, "shuffled_field")]]
            report_controls[name]["shuffled_field"] = _summarize_spectrum(control_spectrum)
            report_controls[name]["shuffled_field"]["spectrum"] = control_spectrum
        if "random_gaussian" in controls:
            control_spectrum = spectra[control_task_ids[(name, "random_gaussian")]]
            report_controls[name]["random_gaussian"] = _summarize_spectrum(control_spectrum)
            report_controls[name]["random_gaussian"]["spectrum"] = control_spectrum
        report_fields[name]["control_comparison"] = _control_comparison(
            spectrum,
            {
                control_name: control_report["spectrum"]
                for control_name, control_report in report_controls[name].items()
                if "spectrum" in control_report
            },
        )

    return {
        "estimator": estimator,
        "claim_boundary": (
            "screen-only freezeout angular-spectrum proxy; not a Planck likelihood, "
            "not a 3D bulk P(k), and not evidence for bulk_3d_established"
        ),
        "ell_max": int(ell_max),
        "pair_samples": int(pair_samples),
        "harmonic_batch_size": int(harmonic_batch_size),
        "harmonic_basis_cache": "batched_all_fields" if spectrum_fn is spherical_harmonic_cl else "not_used",
        "n_jobs": int(resolved_jobs),
        "point_count": int(points.shape[0]),
        "fields": report_fields,
        "controls": report_controls,
    }


def _run_spectrum_tasks(
    tasks: list[dict[str, Any]],
    *,
    points: np.ndarray,
    spectrum_fn,
    ell_max: int,
    pair_samples: int,
    measure_weights: np.ndarray | None,
    harmonic_batch_size: int,
    exact_pair_limit: int,
    n_jobs: int,
) -> dict[int, list[dict[str, float]]]:
    if spectrum_fn is spherical_harmonic_cl:
        return _run_harmonic_tasks_cached(
            tasks,
            points=points,
            ell_max=ell_max,
            measure_weights=measure_weights,
            harmonic_batch_size=harmonic_batch_size,
            n_jobs=n_jobs,
        )

    def compute(task: dict[str, Any]) -> tuple[int, list[dict[str, float]]]:
        spectrum = spectrum_fn(
            points,
            task["values"],
            ell_max=ell_max,
            pair_samples=pair_samples,
            rng=np.random.default_rng(int(task["rng_seed"])),
            measure_weights=measure_weights,
            harmonic_batch_size=harmonic_batch_size,
            exact_pair_limit=exact_pair_limit,
        )
        return int(task["task_id"]), spectrum

    if n_jobs <= 1 or len(tasks) <= 1:
        return dict(compute(task) for task in tasks)

    results: dict[int, list[dict[str, float]]] = {}
    with ThreadPoolExecutor(max_workers=min(int(n_jobs), len(tasks))) as pool:
        futures = [pool.submit(compute, task) for task in tasks]
        for future in as_completed(futures):
            task_id, spectrum = future.result()
            results[task_id] = spectrum
    return results


def _run_harmonic_tasks_cached(
    tasks: list[dict[str, Any]],
    *,
    points: np.ndarray,
    ell_max: int,
    measure_weights: np.ndarray | None,
    harmonic_batch_size: int,
    n_jobs: int = 1,
) -> dict[int, list[dict[str, float]]]:
    weights = _quadrature_weights(points.shape[0], measure_weights)
    standardized = []
    active = []
    for task in tasks:
        values = _standardize(np.asarray(task["values"], dtype=float), weights)
        norm = float(np.sqrt(np.sum(weights * values * values)))
        standardized.append(weights * values)
        active.append(np.any(np.isfinite(values)) and norm >= 1e-12)
    m_max = int(ell_max)
    coeffs = np.zeros((len(tasks), m_max + 1, 2 * m_max + 1), dtype=np.complex128)
    theta = np.arccos(np.clip(points[:, 2].astype(float), -1.0, 1.0))
    phi = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * math.pi)
    values_by_task = np.vstack(standardized) if standardized else np.zeros((0, points.shape[0]), dtype=float)
    requested_jobs = max(1, int(n_jobs))
    batch_size = max(128, int(harmonic_batch_size))
    worker_count = min(requested_jobs, max(1, math.ceil(points.shape[0] / batch_size)))
    # Keep the total in-flight spherical-harmonic tensor size close to the
    # single-worker batch size. This lets exact C_l runs use CPU parallelism
    # without multiplying peak memory by n_jobs.
    effective_batch_size = max(128, batch_size // worker_count) if worker_count > 1 else batch_size
    ranges = [
        (start, min(points.shape[0], start + effective_batch_size))
        for start in range(0, points.shape[0], effective_batch_size)
    ]

    def compute_range(bounds: tuple[int, int]) -> np.ndarray:
        start, stop = bounds
        y_all = sph_harm_y_all(m_max, m_max, theta[start:stop], phi[start:stop])
        return np.einsum(
            "lmp,tp->tlm",
            np.conj(y_all),
            values_by_task[:, start:stop],
            optimize=True,
        )

    if worker_count <= 1 or len(ranges) <= 1:
        for bounds in ranges:
            coeffs += compute_range(bounds)
    else:
        coeffs, _peak_in_flight = _bounded_ordered_thread_reduce(
            ranges,
            compute_range,
            initial=coeffs,
            max_workers=worker_count,
        )

    return {
        int(task["task_id"]): _coeffs_to_spectrum(coeffs[index], ell_max=m_max) if active[index] else _zero_spectrum(m_max)
        for index, task in enumerate(tasks)
    }


def _bounded_ordered_thread_reduce(
    inputs,
    compute,
    *,
    initial: np.ndarray,
    max_workers: int,
) -> tuple[np.ndarray, int]:
    """Accumulate threaded batch results with bounded retained futures.

    ``ThreadPoolExecutor`` accepts an unbounded submission queue.  Submitting
    every harmonic batch at million-patch scale retains each completed tensor
    on its ``Future`` until the full list is released.  This sliding window
    keeps at most one batch per worker in flight and reduces in input order,
    making both peak memory and floating-point accumulation order explicit.
    """

    worker_count = max(1, int(max_workers))
    iterator = iter(inputs)
    result = np.asarray(initial).copy()
    pending = deque()
    peak_in_flight = 0
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for _ in range(worker_count):
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append(pool.submit(compute, item))
        peak_in_flight = len(pending)
        while pending:
            result += pending.popleft().result()
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.append(pool.submit(compute, item))
            peak_in_flight = max(peak_in_flight, len(pending))
    return result, peak_in_flight


def spherical_harmonic_cl(
    points: np.ndarray,
    values: np.ndarray,
    *,
    ell_max: int,
    pair_samples: int,
    rng: np.random.Generator,
    measure_weights: np.ndarray | None = None,
    harmonic_batch_size: int = 4096,
    exact_pair_limit: int = 1_500_000,
) -> list[dict[str, float]]:
    del pair_samples, rng, exact_pair_limit
    weights = _quadrature_weights(points.shape[0], measure_weights)
    values = _standardize(values, weights)
    if not np.any(np.isfinite(values)) or float(np.sqrt(np.sum(weights * values * values))) < 1e-12:
        return _zero_spectrum(int(ell_max))
    theta = np.arccos(np.clip(points[:, 2].astype(float), -1.0, 1.0))
    phi = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * math.pi)
    weighted_values = weights * values
    m_max = int(ell_max)
    coeffs = np.zeros((m_max + 1, 2 * m_max + 1), dtype=np.complex128)
    batch_size = max(128, int(harmonic_batch_size))
    for start in range(0, points.shape[0], batch_size):
        stop = min(points.shape[0], start + batch_size)
        y_all = sph_harm_y_all(m_max, m_max, theta[start:stop], phi[start:stop])
        coeffs += np.einsum(
            "lmp,p->lm",
            np.conj(y_all),
            weighted_values[start:stop],
            optimize=True,
        )
    return _coeffs_to_spectrum(coeffs, ell_max=int(ell_max))


def _coeffs_to_spectrum(coeffs: np.ndarray, *, ell_max: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    m_max = int(ell_max)
    for ell in range(int(ell_max) + 1):
        valid = [*range(0, ell + 1), *range(2 * m_max - ell + 1, 2 * m_max + 1)]
        power = float(np.sum(np.abs(coeffs[ell, valid]) ** 2))
        c_ell = power / float(2 * ell + 1)
        d_ell = ell * (ell + 1) * c_ell / (2.0 * math.pi) if ell > 0 else 0.0
        rows.append({"ell": ell, "C_ell": float(c_ell), "D_ell": float(d_ell)})
    return rows


def _zero_spectrum(ell_max: int) -> list[dict[str, float]]:
    return [
        {"ell": ell, "C_ell": 0.0, "D_ell": 0.0}
        for ell in range(int(ell_max) + 1)
    ]


def pseudo_cl(
    points: np.ndarray,
    values: np.ndarray,
    *,
    ell_max: int,
    pair_samples: int,
    rng: np.random.Generator,
    measure_weights: np.ndarray | None = None,
    harmonic_batch_size: int = 4096,
    exact_pair_limit: int = 1_500_000,
) -> list[dict[str, float]]:
    del measure_weights, harmonic_batch_size
    values = _standardize(values)
    if not np.any(np.isfinite(values)) or float(np.std(values)) < 1e-12:
        return [
            {"ell": ell, "C_ell": 0.0, "D_ell": 0.0}
            for ell in range(int(ell_max) + 1)
        ]
    i, j = _pair_indices(values.shape[0], pair_samples, rng, exact_pair_limit)
    mu = np.einsum("ij,ij->i", points[i], points[j])
    mu = np.clip(mu.astype(float), -1.0, 1.0)
    product = values[i] * values[j]
    rows: list[dict[str, float]] = []

    p_l_minus_1 = np.ones_like(mu)
    c_0 = 4.0 * math.pi * float(np.mean(product * p_l_minus_1))
    rows.append({"ell": 0, "C_ell": c_0, "D_ell": 0.0})
    if ell_max <= 0:
        return rows

    p_l = mu
    c_1 = 4.0 * math.pi * float(np.mean(product * p_l))
    rows.append({"ell": 1, "C_ell": c_1, "D_ell": c_1 / math.pi})
    for ell in range(2, int(ell_max) + 1):
        p_next = ((2 * ell - 1) * mu * p_l - (ell - 1) * p_l_minus_1) / ell
        c_ell = 4.0 * math.pi * float(np.mean(product * p_next))
        d_ell = ell * (ell + 1) * c_ell / (2.0 * math.pi)
        rows.append({"ell": ell, "C_ell": c_ell, "D_ell": d_ell})
        p_l_minus_1, p_l = p_l, p_next
    return rows


def _pair_indices(
    node_count: int,
    pair_samples: int,
    rng: np.random.Generator,
    exact_pair_limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    exact_pairs = node_count * node_count
    if exact_pairs <= exact_pair_limit:
        base = np.arange(node_count, dtype=np.int64)
        return np.repeat(base, node_count), np.tile(base, node_count)
    samples = int(max(1, pair_samples))
    return (
        rng.integers(0, node_count, size=samples, dtype=np.int64),
        rng.integers(0, node_count, size=samples, dtype=np.int64),
    )


def _spectrum_estimator(estimator: str):
    if estimator in {"spherical_harmonic", "direct_spherical_harmonic_cl", "harmonic"}:
        return spherical_harmonic_cl
    if estimator in {"sampled_legendre_two_point_pseudo_cl", "pseudo_cl", "pair"}:
        return pseudo_cl
    raise ValueError(f"unknown angular-power estimator: {estimator}")


def _resolve_n_jobs(value: int | str | None) -> int:
    if value is None:
        return 1
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none"}:
            return 1
        if text in {"auto", "all", "-1", "0"}:
            try:
                from oph_fpe.scale.parallel import available_cpu_count

                return available_cpu_count(reserve=1)
            except Exception:
                return 1
        return max(1, int(text))
    numeric = int(value)
    if numeric <= 0:
        try:
            from oph_fpe.scale.parallel import available_cpu_count

            return available_cpu_count(reserve=1)
        except Exception:
            return 1
    return numeric


def _quadrature_weights(node_count: int, measure_weights: np.ndarray | None) -> np.ndarray:
    if measure_weights is None:
        return np.full(node_count, 4.0 * math.pi / float(node_count), dtype=float)
    weights = np.asarray(measure_weights, dtype=float)
    if weights.shape[0] != node_count:
        raise ValueError("measure_weights length must match point count")
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(node_count, 4.0 * math.pi / float(node_count), dtype=float)
    return weights * (4.0 * math.pi / total)


def _standardize(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if weights is None:
        values = values - float(np.mean(values))
        std = float(np.std(values))
    else:
        norm = float(np.sum(weights))
        mean = float(np.sum(weights * values) / max(norm, 1e-12))
        values = values - mean
        std = math.sqrt(float(np.sum(weights * values * values) / max(norm, 1e-12)))
    if std < 1e-12:
        return np.zeros_like(values)
    return values / std


def _summarize_spectrum(spectrum: list[dict[str, float]]) -> dict[str, Any]:
    usable = [row for row in spectrum if row["ell"] >= 2]
    if not usable:
        return {"peak_ell": None, "peak_D_ell": 0.0, "low_ell_power_2_10": 0.0}
    peak = max(usable, key=lambda row: row["D_ell"])
    low = [row["D_ell"] for row in usable if 2 <= row["ell"] <= 10]
    return {
        "peak_ell": int(peak["ell"]),
        "peak_D_ell": float(peak["D_ell"]),
        "low_ell_power_2_10": float(np.mean(low)) if low else 0.0,
        "total_abs_D_ell_2_plus": float(np.sum(np.abs([row["D_ell"] for row in usable]))),
    }


def _control_comparison(
    spectrum: list[dict[str, float]],
    controls: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    target = _d_vector(spectrum)
    target_norm = float(np.linalg.norm(target))
    rows: dict[str, Any] = {}
    for name, control_spectrum in controls.items():
        control = _d_vector(control_spectrum)
        control_norm = float(np.linalg.norm(control))
        delta = target - control
        rows[name] = {
            "relative_l2_delta": float(np.linalg.norm(delta) / (target_norm + 1e-12)),
            "target_to_control_abs_power_ratio": float(
                np.sum(np.abs(target)) / (np.sum(np.abs(control)) + 1e-12)
            ),
            "shape_correlation": _shape_correlation(target, control),
        }
    if not rows:
        return {
            "control_count": 0,
            "min_relative_l2_delta": None,
            "max_shape_correlation": None,
            "by_control": {},
        }
    return {
        "control_count": len(rows),
        "min_relative_l2_delta": float(min(row["relative_l2_delta"] for row in rows.values())),
        "max_shape_correlation": float(max(row["shape_correlation"] for row in rows.values())),
        "by_control": rows,
    }


def _d_vector(spectrum: list[dict[str, float]]) -> np.ndarray:
    return np.asarray([float(row["D_ell"]) for row in spectrum if int(row["ell"]) >= 2], dtype=float)


def _shape_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    count = min(left.size, right.size)
    left = left[:count] - float(np.mean(left[:count]))
    right = right[:count] - float(np.mean(right[:count]))
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)
