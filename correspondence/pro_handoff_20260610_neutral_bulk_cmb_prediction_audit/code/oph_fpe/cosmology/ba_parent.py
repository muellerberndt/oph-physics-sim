from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Protocol

import numpy as np
from scipy.spatial import cKDTree

from oph_fpe.bulk.cap_geometry import RoundCap


DEFAULT_B_A_CONTROLS = [
    "phase_shuffled_baryon_mode",
    "wrong_k_label",
    "random_collar_labels",
    "no_repair_load_channel",
    "baryon_delta_applied_after_record_freezeout",
]


@dataclass(frozen=True)
class BaryonPerturbationMode:
    k_h_mpc: float
    seed: int
    mode_index: int
    phase: float
    control: str | None = None


class CollarResponseParent(Protocol):
    """Minimal callback surface for non-fit B_A finite differences."""

    def make_background(self, a: float) -> Any: ...

    def apply_baryon_delta(self, background: Any, amplitude: float, mode: BaryonPerturbationMode) -> Any: ...

    def rho_A(self, state: Any) -> float: ...

    def rho_A_eq(self, state: Any) -> float: ...

    def baryon_density(self, a: float) -> float: ...


def estimate_b_a_grid(
    collar_parent: CollarResponseParent,
    *,
    a_grid: Iterable[float],
    k_grid_h_mpc: Iterable[float],
    eps: float = 1.0e-3,
    modes_per_k: int = 16,
    seeds: Iterable[int] = range(8),
    controls: Iterable[str] = DEFAULT_B_A_CONTROLS,
) -> dict[str, Any]:
    """Estimate a diagnostic B_A(k,a) parent by finite baryon perturbations.

    This uses controlled plus/minus baryon perturbations and never consults CMB
    data. The emitted rows are diagnostic until refinement convergence, control
    failure, time/scale calibration, and energy-momentum closure are proven.
    """

    a_values = [float(value) for value in a_grid]
    k_values = [float(value) for value in k_grid_h_mpc]
    seed_values = [int(value) for value in seeds]
    control_values = [str(value) for value in controls]
    rows = [
        _estimate_row(collar_parent, a, k, eps=float(eps), modes_per_k=int(modes_per_k), seeds=seed_values)
        for a in a_values
        for k in k_values
    ]
    control_rows: list[dict[str, Any]] = []
    for control in control_values:
        for a in a_values:
            for k in k_values:
                control_rows.append(
                    _estimate_row(
                        collar_parent,
                        a,
                        k,
                        eps=float(eps),
                        modes_per_k=int(modes_per_k),
                        seeds=seed_values,
                        control=control,
                    )
                )
    readiness = _readiness(rows, control_rows)
    return {
        "mode": "finite_difference_baryon_response_B_A_parent_v0",
        "a_grid": a_values,
        "k_grid_h_mpc": k_values,
        "eps": float(eps),
        "modes_per_k": int(modes_per_k),
        "seed_count": len(seed_values),
        "rows": rows,
        "control_rows": control_rows,
        "readiness": readiness,
        "B_A_PARENT_RECEIPT": readiness["B_A_PARENT_RECEIPT"],
        "physical_prediction_ready": False,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Finite-difference collar response to controlled baryon perturbations. "
            "Rows are non-fit B_A(k,a) parent diagnostics; they are not physical "
            "Boltzmann inputs until convergence, controls, scale/time calibration, "
            "and energy-momentum exchange closure pass."
        ),
    }


def write_b_a_parent_report(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    a_grid: Iterable[float] | None = None,
    k_grid_h_mpc: Iterable[float] | None = None,
    eps: float = 1.0e-3,
    modes_per_k: int = 8,
    seeds: Iterable[int] = range(4),
    controls: Iterable[str] = DEFAULT_B_A_CONTROLS,
) -> dict[str, Any]:
    """Write a first-class B_A parent diagnostic from finite collar reports.

    The current OPH-FPE runs do not yet contain real plus/minus baryon
    perturbation reruns. When only `oph_cmb_stress_report.json` files are
    available, this writer wraps their finite-collar kernel proxy in the same
    finite-difference contract used by `estimate_b_a_grid`, labels the parent
    as report-backed surrogate, and keeps all physical gates closed.
    """

    seed_values = [int(value) for value in seeds]
    control_values = [str(value) for value in controls]
    source_reports = _load_stress_reports(run_dirs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    source_results: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    stress_control_rows: list[dict[str, Any]] = []
    for index, source in enumerate(source_reports):
        report = source["report"]
        parent = StressReportDiagnosticCollarParent(report)
        source_a_grid = list(a_grid) if a_grid is not None else _report_a_grid(report)
        source_k_grid = list(k_grid_h_mpc) if k_grid_h_mpc is not None else parent.k_proxy_grid()
        if not source_k_grid:
            continue
        result = estimate_b_a_grid(
            parent,
            a_grid=source_a_grid,
            k_grid_h_mpc=source_k_grid,
            eps=eps,
            modes_per_k=modes_per_k,
            seeds=seed_values,
            controls=control_values,
        )
        for row in result["rows"]:
            row.update(
                {
                    "source_report_index": index,
                    "source_report_path": source["path"],
                    "k_proxy_inverse_theta": row.get("k_h_mpc"),
                    "k_units": "inverse_cap_opening_angle_proxy",
                    "parent_source": "oph_cmb_stress_report_diagnostic_kernel_proxy",
                }
            )
        for row in result["control_rows"]:
            row.update(
                {
                    "source_report_index": index,
                    "source_report_path": source["path"],
                    "k_proxy_inverse_theta": row.get("k_h_mpc"),
                    "k_units": "inverse_cap_opening_angle_proxy",
                    "parent_source": "oph_cmb_stress_report_diagnostic_kernel_proxy",
                }
            )
        stress_rows.extend(result["rows"])
        stress_control_rows.extend(result["control_rows"])
        source_results.append(
            {
                "source_report_index": index,
                "source_report_path": source["path"],
                "row_count": len(result["rows"]),
                "control_row_count": len(result["control_rows"]),
                "readiness": result["readiness"],
                "finite_collar_parent_status": (
                    (report.get("finite_collar_parent") or {}).get("status")
                ),
                "kernel_proxy_status": (
                    (report.get("diagnostic_kernel_proxy") or {}).get("status")
                ),
            }
        )

    observer_sources = _load_observer_view_sources(run_dirs)
    observer_rows: list[dict[str, Any]] = []
    observer_control_rows: list[dict[str, Any]] = []
    observer_results: list[dict[str, Any]] = []
    for index, source in enumerate(observer_sources):
        result = _observer_view_parent_variation(
            source,
            a_grid=list(a_grid) if a_grid is not None else _report_a_grid(source.get("stress_report", {})),
            k_grid_h_mpc=list(k_grid_h_mpc) if k_grid_h_mpc is not None else None,
            eps=float(eps),
            modes_per_k=int(modes_per_k),
            seeds=seed_values,
            controls=control_values,
        )
        observer_rows.extend(result["rows"])
        observer_control_rows.extend(result["control_rows"])
        observer_results.append(
            {
                "source_index": index,
                "run_dir": source["run_dir"],
                "observer_count": result["observer_count"],
                "cap_count": result["cap_count"],
                "row_count": len(result["rows"]),
                "control_row_count": len(result["control_rows"]),
                "readiness": result["readiness"],
            }
        )

    rows = observer_rows if observer_rows else stress_rows
    control_rows = observer_control_rows if observer_rows else stress_control_rows
    primary_source = (
        "observer_view_finite_collar_packet_variation"
        if observer_rows
        else "stress_report_diagnostic_kernel_proxy"
    )
    readiness = _aggregate_report_readiness(source_reports, rows, control_rows)
    readiness["checks"]["finite_observer_view_parent_variation"] = bool(observer_rows)
    if observer_rows:
        readiness["missing_gates"] = [
            name for name, passed in readiness["checks"].items() if not passed
        ]
    report = {
        "mode": "report_backed_finite_collar_B_A_parent_diagnostic_v0",
        "source_report_count": len(source_reports),
        "observer_view_source_count": len(observer_sources),
        "primary_parent_source": primary_source,
        "source_results": source_results,
        "observer_view_results": observer_results,
        "a_grid": sorted({float(row["a"]) for row in rows}) if rows else [],
        "k_grid_proxy_inverse_theta": sorted({float(row["k_proxy_inverse_theta"]) for row in rows})
        if rows
        else [],
        "eps": float(eps),
        "modes_per_k": int(modes_per_k),
        "seed_count": len(seed_values),
        "rows": rows,
        "control_rows": control_rows,
        "observer_view_rows": observer_rows,
        "observer_view_control_rows": observer_control_rows,
        "stress_report_surrogate_rows": stress_rows,
        "stress_report_surrogate_control_rows": stress_control_rows,
        "readiness": readiness,
        "B_A_PARENT_RECEIPT": readiness["B_A_PARENT_RECEIPT"],
        "physical_prediction_ready": False,
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "claim_boundary": (
            "This is a first-class diagnostic B_A(k,a) artifact built from finite "
            "collar readouts without CMB data. When observer_views.jsonl and cached "
            "caps are available, rows use observer-view packet finite variations of "
            "the collar CMI parent functional; otherwise rows fall back to the older "
            "stress-report shape surrogate. It is still not a physical Boltzmann "
            "kernel until real perturbation reruns, calibrated k/a units, energy "
            "exchange closure, and refinement convergence pass."
        ),
    }
    (out / "b_a_parent_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "b_a_parent_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "b_a_parent_rows.csv", rows)
    _write_csv(out / "b_a_parent_control_rows.csv", control_rows)
    _write_csv(out / "b_a_parent_observer_view_rows.csv", observer_rows)
    _write_csv(out / "b_a_parent_observer_view_control_rows.csv", observer_control_rows)
    _write_csv(out / "b_a_parent_stress_surrogate_rows.csv", stress_rows)
    return report


class StressReportDiagnosticCollarParent:
    """Report-backed surrogate for the finite collar response contract.

    This adapter exists so downstream code can exercise the B_A finite
    difference path before expensive baryon perturbation reruns are available.
    Its output is intentionally not theorem-grade.
    """

    def __init__(self, report: dict[str, Any]):
        self.report = report
        self.parent = report.get("finite_collar_parent", {}) or {}
        self.kernel = report.get("diagnostic_kernel_proxy", {}) or {}
        self.kernel_rows = [
            row for row in (self.kernel.get("kernel_proxy_rows") or []) if isinstance(row, dict)
        ]
        self.base_rho = max(
            abs(_float_or_none(self.parent.get("weighted_collar_repair_defect_R")) or 0.0),
            abs(_float_or_none(self.parent.get("mean_epsilon_cmi")) or 0.0),
            1.0e-12,
        )

    def make_background(self, a: float) -> dict[str, Any]:
        return {"a": float(a), "delta": 0.0, "mode": None}

    def apply_baryon_delta(
        self,
        background: dict[str, Any],
        amplitude: float,
        mode: BaryonPerturbationMode,
    ) -> dict[str, Any]:
        state = dict(background)
        state["delta"] = float(amplitude)
        state["mode"] = mode
        return state

    def rho_A(self, state: dict[str, Any]) -> float:
        a = max(float(state["a"]), 1.0e-12)
        return float(self.base_rho * a ** -3)

    def rho_A_eq(self, state: dict[str, Any]) -> float:
        background = self.rho_A(state)
        mode = state.get("mode")
        if mode is None:
            return background
        b_value = self._b_a_shape(float(mode.k_h_mpc))
        if mode.control in {
            "random_collar_labels",
            "no_repair_load_channel",
            "baryon_delta_applied_after_record_freezeout",
        }:
            b_value = 0.0
        return float(background + float(state["delta"]) * float(mode.phase) * float(b_value) * background)

    def baryon_density(self, a: float) -> float:
        # A positive normalization is enough because the finite-difference
        # estimator divides the response by rho_A. Keep this explicit instead
        # of importing measured omega_b values into a no-data-use diagnostic.
        return float(max(float(a), 1.0e-12) ** -3)

    def k_proxy_grid(self) -> list[float]:
        values = [
            _float_or_none(row.get("k_proxy_inverse_theta"))
            for row in self.kernel_rows
            if _float_or_none(row.get("k_proxy_inverse_theta")) is not None
        ]
        return sorted({float(value) for value in values})

    def _b_a_shape(self, k_value: float) -> float:
        if not self.kernel_rows:
            return 0.0
        best_row = min(
            self.kernel_rows,
            key=lambda row: abs(float(_float_or_none(row.get("k_proxy_inverse_theta")) or 0.0) - float(k_value)),
        )
        return float(_float_or_none(best_row.get("B_A_shape_proxy")) or 0.0)


def _observer_view_parent_variation(
    source: dict[str, Any],
    *,
    a_grid: Iterable[float],
    k_grid_h_mpc: Iterable[float] | None,
    eps: float,
    modes_per_k: int,
    seeds: list[int],
    controls: list[str],
) -> dict[str, Any]:
    views = [view for view in source["observer_views"] if view.get("view_type") == "patch_observer"]
    points = _observer_axes(views)
    caps = source["caps"]
    packets = _observer_view_packets(views, include_repair=True)
    packet_controls = {
        "no_repair_load_channel": _observer_view_packets(views, include_repair=False),
    }
    a_values = [float(value) for value in a_grid]
    k_values = [float(value) for value in (k_grid_h_mpc or _cap_k_grid(caps))]
    if not a_values:
        a_values = [1.0 / 1100.0, 0.01, 0.1, 1.0]
    rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for a in a_values:
        for k in k_values:
            rows.extend(
                _observer_view_rows_for_k(
                    points,
                    caps,
                    packets,
                    a=a,
                    k_proxy=float(k),
                    eps=float(eps),
                    modes_per_k=int(modes_per_k),
                    seeds=seeds,
                    source=source,
                    control=None,
                    packet_controls=packet_controls,
                )
            )
            for control in controls:
                control_rows.extend(
                    _observer_view_rows_for_k(
                        points,
                        caps,
                        packets,
                        a=a,
                        k_proxy=float(k),
                        eps=float(eps),
                        modes_per_k=int(modes_per_k),
                        seeds=seeds,
                        source=source,
                        control=control,
                        packet_controls=packet_controls,
                    )
                )
    readiness = _observer_view_readiness(rows, control_rows)
    return {
        "observer_count": len(views),
        "cap_count": len(caps),
        "rows": rows,
        "control_rows": control_rows,
        "readiness": readiness,
    }


def _observer_view_rows_for_k(
    points: np.ndarray,
    caps: list[RoundCap],
    packets: np.ndarray,
    *,
    a: float,
    k_proxy: float,
    eps: float,
    modes_per_k: int,
    seeds: list[int],
    source: dict[str, Any],
    control: str | None,
    packet_controls: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng_seed = 9173 + int(abs(float(k_proxy)) * 1_000_000)
    work_packets = np.asarray(packet_controls.get(control, packets), dtype=np.int64).copy()
    if control == "random_collar_labels":
        rng = np.random.default_rng(rng_seed)
        work_packets = work_packets.copy()
        rng.shuffle(work_packets)
    k_for_mode = float(k_proxy) * (1.7 if control == "wrong_k_label" else 1.0)
    for cap_index, cap in enumerate(caps):
        triplets = _observer_collar_triplets(points, cap)
        if triplets["collar"].size == 0:
            continue
        a_packets = work_packets[triplets["inside"]]
        b_packets = work_packets[triplets["collar"]]
        d_packets = work_packets[triplets["outside"]]
        base_weights = np.ones(triplets["collar"].size, dtype=float)
        if control == "baryon_delta_applied_after_record_freezeout":
            plus_values = [0.0]
            minus_values = [0.0]
            b_values = [0.0]
        else:
            b_values = []
            plus_values = []
            minus_values = []
            for seed in seeds:
                for mode_index in range(max(1, int(modes_per_k))):
                    mode = _observer_baryon_mode(
                        points[triplets["collar"]],
                        cap,
                        k_for_mode,
                        seed=int(seed),
                        mode_index=int(mode_index),
                        phase_shuffled=control == "phase_shuffled_baryon_mode",
                    )
                    plus_weights = base_weights * np.clip(1.0 + float(eps) * mode, 1.0e-9, None)
                    minus_weights = base_weights * np.clip(1.0 - float(eps) * mode, 1.0e-9, None)
                    plus = _weighted_classical_cmi(a_packets, b_packets, d_packets, plus_weights)
                    minus = _weighted_classical_cmi(a_packets, b_packets, d_packets, minus_weights)
                    base = _weighted_classical_cmi(a_packets, b_packets, d_packets, base_weights)
                    derivative = (plus - minus) / (2.0 * max(float(eps), 1.0e-12))
                    b_values.append(float(derivative / max(abs(base), 1.0e-12)))
                    plus_values.append(float(plus))
                    minus_values.append(float(minus))
        values = np.asarray(b_values, dtype=float)
        plus_arr = np.asarray(plus_values, dtype=float)
        minus_arr = np.asarray(minus_values, dtype=float)
        base = _weighted_classical_cmi(a_packets, b_packets, d_packets, base_weights)
        rows.append(
            {
                "a": float(a),
                "k_h_mpc": float(k_proxy),
                "k_proxy_inverse_theta": float(k_proxy),
                "k_units": "inverse_cap_opening_angle_proxy",
                "control": control,
                "cap_index": int(cap_index),
                "theta0": float(cap.theta0),
                "collar_width": float(cap.collar_width),
                "observer_count": int(points.shape[0]),
                "collar_observer_count": int(triplets["collar"].size),
                "base_epsilon_cmi": float(base),
                "plus_epsilon_cmi_mean": float(np.mean(plus_arr)) if plus_arr.size else None,
                "minus_epsilon_cmi_mean": float(np.mean(minus_arr)) if minus_arr.size else None,
                "B_A_mean": float(np.mean(values)) if values.size else None,
                "B_A_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "B_A_sem": float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0,
                "mode_count": int(values.size),
                "sign_stable": _sign_stable(values),
                "source": "observer_view_finite_collar_packet_variation",
                "source_run_dir": source["run_dir"],
                "parent_source": "observer_views_jsonl_visible_packet_cmi_variation",
            }
        )
    return rows


def _observer_collar_triplets(points: np.ndarray, cap: RoundCap) -> dict[str, np.ndarray]:
    cap = cap.normalized()
    # Cached caps were calibrated for the full patch screen. On sampled observer
    # views the collar must be widened to keep a finite support-visible band.
    width = max(float(cap.collar_width), 2.5 / max(float(np.sqrt(points.shape[0])), 1.0))
    signed = points @ cap.axis - float(np.cos(cap.theta0))
    collar_count = max(8, min(points.shape[0] // 4, int(round(0.08 * points.shape[0]))))
    collar = np.argsort(np.abs(signed))[:collar_count]
    hard_inside = signed >= 0.0
    inside_candidates = np.flatnonzero(hard_inside)
    outside_candidates = np.flatnonzero(~hard_inside)
    inside_candidates = np.setdiff1d(inside_candidates, collar, assume_unique=False)
    outside_candidates = np.setdiff1d(outside_candidates, collar, assume_unique=False)
    if inside_candidates.size == 0 or outside_candidates.size == 0:
        return {
            "inside": np.zeros(0, dtype=np.int64),
            "collar": np.zeros(0, dtype=np.int64),
            "outside": np.zeros(0, dtype=np.int64),
        }
    inside = _nearest_indices(points, inside_candidates, points[collar])
    outside = _nearest_indices(points, outside_candidates, points[collar])
    return {
        "inside": np.asarray(inside, dtype=np.int64),
        "collar": np.asarray(collar, dtype=np.int64),
        "outside": np.asarray(outside, dtype=np.int64),
        "effective_collar_width": float(width),
    }


def _observer_baryon_mode(
    collar_points: np.ndarray,
    cap: RoundCap,
    k_proxy: float,
    *,
    seed: int,
    mode_index: int,
    phase_shuffled: bool,
) -> np.ndarray:
    cap = cap.normalized()
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(mode_index) * 1543 + 991)
    phase = float(rng.uniform(0.0, 2.0 * np.pi)) if phase_shuffled else float(mode_index) * np.pi / 4.0
    cross = np.cross(cap.axis, cap.tangent)
    x = collar_points @ cap.tangent
    y = collar_points @ cross
    signal = np.cos(float(k_proxy) * x / max(float(np.sin(cap.theta0)), 1.0e-6) + phase)
    signal += 0.5 * np.sin(float(k_proxy) * y / max(float(np.sin(cap.theta0)), 1.0e-6) + 0.5 * phase)
    signal = signal - float(np.mean(signal))
    std = float(np.std(signal))
    if std < 1.0e-12:
        return np.ones_like(signal)
    return signal / std


def _observer_view_packets(views: list[dict[str, Any]], *, include_repair: bool) -> np.ndarray:
    rows: list[tuple[int, ...]] = []
    for view in views:
        row = [
            _bounded_int(view.get("dominant_record_signature"), 257),
            _bounded_int(view.get("dominant_object_packet"), 257),
            _bounded_int(view.get("modular_response_cluster"), 4099),
            _bounded_int(view.get("transition_history_key"), 4099),
            _float_bin(view.get("visible_signature_entropy"), bins=16, scale=8.0),
            _float_bin(view.get("record_stability_mean"), bins=16, scale=128.0),
            _float_bin(view.get("counterfactual_stability"), bins=8, scale=1.0),
            _float_bin(view.get("transition_history_mean_modal_mass"), bins=8, scale=1.0),
        ]
        if include_repair:
            sig = view.get("perturb_resettle_signature") or []
            row.extend(
                [
                    _float_bin(view.get("repair_load_mean"), bins=8, scale=8.0),
                    _float_bin(view.get("mismatch_density_mean"), bins=8, scale=1.0),
                    _float_bin(sig[0] if len(sig) > 0 else 0.0, bins=16, scale=16.0),
                    _float_bin(sig[1] if len(sig) > 1 else 0.0, bins=16, scale=16.0),
                ]
            )
        rows.append(tuple(row))
    mapping: dict[tuple[int, ...], int] = {}
    packets = np.zeros(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        if row not in mapping:
            mapping[row] = len(mapping)
        packets[index] = mapping[row]
    return packets


def _weighted_classical_cmi(
    a_packets: np.ndarray,
    b_packets: np.ndarray,
    d_packets: np.ndarray,
    weights: np.ndarray,
) -> float:
    h_ab = _weighted_entropy(zip(a_packets, b_packets, strict=True), weights)
    h_bd = _weighted_entropy(zip(b_packets, d_packets, strict=True), weights)
    h_b = _weighted_entropy(((int(value),) for value in b_packets), weights)
    h_abd = _weighted_entropy(zip(a_packets, b_packets, d_packets, strict=True), weights)
    return max(0.0, float(h_ab + h_bd - h_b - h_abd))


def _weighted_entropy(keys: Iterable[Iterable[int]], weights: np.ndarray) -> float:
    totals: dict[tuple[int, ...], float] = {}
    total = 0.0
    for key, weight in zip(keys, np.asarray(weights, dtype=float), strict=True):
        w = max(float(weight), 0.0)
        if w <= 0.0:
            continue
        packed = tuple(int(item) for item in key)
        totals[packed] = totals.get(packed, 0.0) + w
        total += w
    if total <= 0.0:
        return 0.0
    probs = np.asarray([value / total for value in totals.values() if value > 0.0], dtype=float)
    return float(-np.sum(probs * np.log(probs))) if probs.size else 0.0


def _observer_view_readiness(rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]) -> dict[str, Any]:
    main = [abs(float(row["B_A_mean"])) for row in rows if isinstance(row.get("B_A_mean"), (int, float))]
    main_scale = fmean(main) if main else 0.0
    grouped: dict[str, list[float]] = {}
    for row in control_rows:
        if isinstance(row.get("B_A_mean"), (int, float)):
            grouped.setdefault(str(row.get("control")), []).append(abs(float(row["B_A_mean"])))
    control_failures = {
        name: bool(values and fmean(values) < 0.5 * max(main_scale, 1.0e-300))
        for name, values in grouped.items()
    }
    checks = {
        "observer_view_rows_emitted": bool(rows),
        "observer_view_control_rows_emitted": bool(control_rows),
        "no_cmb_data_used": True,
        "controls_fail": bool(control_failures and all(control_failures.values())),
        "calibrated_k_h_mpc": False,
        "calibrated_a_evolution": False,
        "full_perturbation_rerun": False,
        "refinement_convergence_passed": False,
    }
    return {
        "checks": checks,
        "control_failures": control_failures,
        "missing_gates": [name for name, passed in checks.items() if not passed],
    }


def _estimate_row(
    parent: CollarResponseParent,
    a: float,
    k_h_mpc: float,
    *,
    eps: float,
    modes_per_k: int,
    seeds: list[int],
    control: str | None = None,
) -> dict[str, Any]:
    background = parent.make_background(float(a))
    rho_a = max(float(parent.rho_A(background)), 1.0e-300)
    rho_b = max(float(parent.baryon_density(float(a))), 1.0e-300)
    estimates: list[float] = []
    for seed in seeds:
        for mode_index in range(int(modes_per_k)):
            mode = _sample_mode(float(k_h_mpc), int(seed), int(mode_index), control=control)
            plus = _apply_delta(parent, background, +float(eps), mode)
            minus = _apply_delta(parent, background, -float(eps), mode)
            derivative = (float(parent.rho_A_eq(plus)) - float(parent.rho_A_eq(minus))) / (2.0 * float(eps))
            k_parent = derivative / rho_b
            estimates.append(float((rho_b / rho_a) * k_parent))
    values = np.asarray(estimates, dtype=float)
    mean = float(np.mean(values)) if values.size else None
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    sign_stable = _sign_stable(values)
    return {
        "a": float(a),
        "k_h_mpc": float(k_h_mpc),
        "control": control,
        "B_A_mean": mean,
        "B_A_std": std,
        "B_A_sem": float(std / np.sqrt(values.size)) if values.size else None,
        "mode_count": int(values.size),
        "sign_stable": bool(sign_stable),
        "source": "finite_collar_baryon_perturbation_derivative",
    }


def _sample_mode(k_h_mpc: float, seed: int, mode_index: int, *, control: str | None) -> BaryonPerturbationMode:
    rng = np.random.default_rng(seed * 1_000_003 + mode_index * 97 + 11)
    phase = 1.0
    k_for_parent = float(k_h_mpc)
    if control == "phase_shuffled_baryon_mode":
        phase = float(rng.choice([-1.0, 1.0]))
    elif control == "wrong_k_label":
        k_for_parent = float(k_h_mpc) * 1.7
    return BaryonPerturbationMode(
        k_h_mpc=k_for_parent,
        seed=int(seed),
        mode_index=int(mode_index),
        phase=phase,
        control=control,
    )


def _apply_delta(
    parent: CollarResponseParent,
    background: Any,
    amplitude: float,
    mode: BaryonPerturbationMode,
) -> Any:
    try:
        return parent.apply_baryon_delta(background, float(amplitude), mode)
    except TypeError:
        return parent.apply_baryon_delta(background, float(amplitude), mode.k_h_mpc)


def _sign_stable(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    mean = float(np.mean(values))
    if abs(mean) < 1.0e-12:
        return False
    signs = np.sign(values[np.abs(values) > 1.0e-12])
    if signs.size == 0:
        return False
    return bool(float(np.mean(signs == np.sign(mean))) >= 0.8)


def _readiness(rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sign_stable = bool(rows and all(bool(row.get("sign_stable")) for row in rows))
    controls_by_name: dict[str, list[dict[str, Any]]] = {}
    for row in control_rows:
        controls_by_name.setdefault(str(row.get("control")), []).append(row)
    main_abs = [abs(float(row["B_A_mean"])) for row in rows if isinstance(row.get("B_A_mean"), (int, float))]
    main_scale = fmean(main_abs) if main_abs else 0.0
    control_failures: dict[str, bool] = {}
    for name, group in controls_by_name.items():
        values = [abs(float(row["B_A_mean"])) for row in group if isinstance(row.get("B_A_mean"), (int, float))]
        control_failures[name] = bool(values and fmean(values) < 0.5 * max(main_scale, 1.0e-300))
    controls_fail = bool(control_failures and all(control_failures.values()))
    checks = {
        "finite_difference_rows_emitted": bool(rows),
        "no_cmb_data_used": True,
        "sign_stable": sign_stable,
        "controls_fail": controls_fail,
        "convergence_64k_256k": False,
        "rho_A_of_a_emitted": False,
        "rho_A_eq_of_a_emitted": False,
        "Gamma_rec_of_k_a_emitted": False,
        "energy_momentum_exchange_closed": False,
        "gauge_consistency_audited": False,
    }
    return {
        "checks": checks,
        "control_failures": control_failures,
        "missing_gates": [name for name, passed in checks.items() if not passed],
        "B_A_PARENT_RECEIPT": False,
        "physical_prediction_ready": False,
    }


def _load_stress_reports(run_dirs: list[Path]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in [Path(path) for path in run_dirs]:
        candidates: list[Path] = []
        if root.is_file() and root.name == "oph_cmb_stress_report.json":
            candidates.append(root)
        if root.is_dir():
            direct = root / "oph_cmb_stress_report.json"
            if direct.exists():
                candidates.append(direct)
            candidates.extend(sorted(root.glob("**/oph_cmb_stress_report.json")))
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if report:
                sources.append({"path": str(path), "report": report})
    return sources


def _load_observer_view_sources(run_dirs: list[Path]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in [Path(path) for path in run_dirs]:
        candidates: list[Path] = []
        if root.is_dir():
            if (root / "observer_views.jsonl").exists():
                candidates.append(root)
            candidates.extend(path.parent for path in sorted(root.glob("**/observer_views.jsonl")))
        for run_dir in candidates:
            resolved = run_dir.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            views_path = run_dir / "observer_views.jsonl"
            cache_path = run_dir / "modular_response_kernel_cache.json"
            if not views_path.exists() or not cache_path.exists():
                continue
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
                caps = _caps_from_cache(cache)
                views = _read_jsonl(views_path)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if views and caps:
                stress_report = {}
                stress_path = run_dir / "oph_cmb_stress_report.json"
                if stress_path.exists():
                    try:
                        stress_report = json.loads(stress_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        stress_report = {}
                sources.append(
                    {
                        "run_dir": str(run_dir),
                        "observer_views_path": str(views_path),
                        "cap_cache_path": str(cache_path),
                        "observer_views": views,
                        "caps": caps,
                        "stress_report": stress_report,
                    }
                )
    return sources


def _caps_from_cache(cache: dict[str, Any]) -> list[RoundCap]:
    caps: list[RoundCap] = []
    for row in cache.get("caps") or []:
        if not isinstance(row, dict):
            continue
        axis = np.asarray(row.get("axis"), dtype=float)
        tangent = np.asarray(row.get("tangent"), dtype=float)
        if axis.shape != (3,) or tangent.shape != (3,):
            continue
        caps.append(
            RoundCap(
                axis=axis,
                theta0=float(row.get("theta0", 0.75)),
                tangent=tangent,
                collar_width=float(row.get("collar_width", 0.03)),
            ).normalized()
        )
    return caps


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _observer_axes(views: list[dict[str, Any]]) -> np.ndarray:
    axes = np.asarray([view.get("axis", [0.0, 0.0, 1.0]) for view in views], dtype=float)
    if axes.ndim != 2 or axes.shape[1] != 3:
        raise ValueError("observer views need 3D axes")
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    return axes / np.maximum(norms, 1.0e-15)


def _cap_k_grid(caps: list[RoundCap]) -> list[float]:
    return sorted({round(1.0 / max(float(cap.theta0), 1.0e-9), 12) for cap in caps})


def _nearest_indices(points: np.ndarray, candidates: np.ndarray, queries: np.ndarray) -> np.ndarray:
    tree = cKDTree(points[np.asarray(candidates, dtype=np.int64)])
    _, indices = tree.query(queries, k=1)
    return np.asarray(candidates, dtype=np.int64)[np.asarray(indices, dtype=np.int64)]


def _bounded_int(value: Any, modulus: int) -> int:
    try:
        return int(value) % max(1, int(modulus))
    except (TypeError, ValueError, OverflowError):
        return 0


def _float_bin(value: Any, *, bins: int, scale: float) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if not np.isfinite(numeric):
        numeric = 0.0
    clipped = min(max(numeric / max(float(scale), 1.0e-12), 0.0), 0.999999)
    return int(clipped * max(1, int(bins)))


def _report_a_grid(report: dict[str, Any]) -> list[float]:
    values = [
        float(value)
        for value in ((report.get("diagnostic_kernel_proxy") or {}).get("a_grid") or [])
        if isinstance(value, (int, float))
    ]
    if not values:
        values = [1.0 / 1100.0, 0.01, 0.1, 1.0]
    return sorted({round(float(value), 12) for value in values})


def _aggregate_report_readiness(
    source_reports: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "source_stress_reports_available": bool(source_reports),
        "finite_difference_rows_emitted": bool(rows),
        "control_rows_emitted": bool(control_rows),
        "no_cmb_data_used": True,
        "real_baryon_perturbation_runs_present": False,
        "report_backed_surrogate_parent": True,
        "finite_collar_parent_theorem_grade": False,
        "scale_calibrated_k_h_mpc": False,
        "rho_A_of_a_physical_emitted": False,
        "rho_A_eq_of_a_physical_emitted": False,
        "Gamma_rec_of_k_a_emitted": False,
        "energy_momentum_exchange_closed": False,
        "gauge_consistency_audited": False,
        "refinement_convergence_passed": False,
    }
    control_groups: dict[str, list[float]] = {}
    for row in control_rows:
        if isinstance(row.get("B_A_mean"), (int, float)):
            control_groups.setdefault(str(row.get("control")), []).append(abs(float(row["B_A_mean"])))
    main_values = [abs(float(row["B_A_mean"])) for row in rows if isinstance(row.get("B_A_mean"), (int, float))]
    main_scale = fmean(main_values) if main_values else 0.0
    control_failures = {
        name: bool(values and fmean(values) < 0.5 * max(main_scale, 1.0e-300))
        for name, values in control_groups.items()
    }
    checks["controls_fail"] = bool(control_failures and all(control_failures.values()))
    return {
        "checks": checks,
        "control_failures": control_failures,
        "B_A_PARENT_RECEIPT": False,
        "physical_prediction_ready": False,
        "missing_gates": [name for name, passed in checks.items() if not passed],
        "claim_boundary": (
            "The finite-difference contract is exercised, but theorem-grade receipt "
            "requires real baryon perturbation reruns, calibrated k/a units, closed "
            "repair/energy exchange, and refinement convergence."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(value) for key, value in row.items()})


def _markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    row_values = [
        float(row["B_A_mean"])
        for row in report["rows"]
        if isinstance(row.get("B_A_mean"), (int, float))
    ]
    control_values = [
        float(row["B_A_mean"])
        for row in report["control_rows"]
        if isinstance(row.get("B_A_mean"), (int, float))
    ]
    lines = [
        "# OPH B_A Parent Diagnostic",
        "",
        report["claim_boundary"],
        "",
        "## Readiness",
        "",
        f"- primary parent source: {report['primary_parent_source']}",
        f"- source stress reports: {report['source_report_count']}",
        f"- observer-view sources: {report['observer_view_source_count']}",
        f"- finite-difference rows emitted: {readiness['checks']['finite_difference_rows_emitted']}",
        f"- observer-view finite variation: {readiness['checks'].get('finite_observer_view_parent_variation', False)}",
        f"- no CMB data used: {readiness['checks']['no_cmb_data_used']}",
        f"- real baryon perturbation runs present: {readiness['checks']['real_baryon_perturbation_runs_present']}",
        f"- controls fail as required: {readiness['checks'].get('controls_fail', False)}",
        f"- theorem-grade B_A parent receipt: {report['B_A_PARENT_RECEIPT']}",
        f"- physical prediction ready: {report['physical_prediction_ready']}",
        f"- missing gates: {', '.join(readiness['missing_gates'])}",
        "",
        "## Diagnostic Values",
        "",
        f"- B_A row count: {len(report['rows'])}",
        f"- control row count: {len(report['control_rows'])}",
        f"- mean B_A diagnostic: {_fmt(fmean(row_values) if row_values else None)}",
        f"- mean control B_A diagnostic: {_fmt(fmean(control_values) if control_values else None)}",
        "",
        "## Output Files",
        "",
        "- `b_a_parent_report.json`",
        "- `b_a_parent_rows.csv`",
        "- `b_a_parent_control_rows.csv`",
        "",
    ]
    return "\n".join(lines)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.10g}"
    return "n/a"
