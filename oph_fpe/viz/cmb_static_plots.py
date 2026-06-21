from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_cmb_static_plots(run_dir: Path, out_dir: Path | None = None) -> dict[str, Any]:
    """Write shareable static CMB and neutral-frontier plots from a measurement pack."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run_dir)
    out = Path(out_dir) if out_dir is not None else run
    out.mkdir(parents=True, exist_ok=True)

    claims = _read_json(run / "claims.json")
    cmb_output = _read_json(run / "physical_cmb_output_comparison_report.json")
    neutral_sweep = _read_json(run / "overlap_residualized_graph_geometry_sweep_report.json")
    lcdm_rows = _read_rows(run / "camb_lcdm_tt_bins.csv")
    scale_rows = _read_rows(run / "scale_compressed_cmb_tt_bins.csv")
    finite_rows = _read_rows(run / "finite_repair_clock_cmb_tt_bins.csv")
    residual_rows = _read_rows(run / "physical_cmb_best_oph_residuals.csv")
    comparison_rows = _read_rows(run / "physical_cmb_output_comparison_rows.csv")
    peak_rows = _read_rows(run / "physical_cmb_peak_features.csv")

    comparison_path = out / "physical_cmb_tt_comparison.png"
    residual_path = out / "physical_cmb_best_oph_residuals.png"
    peak_path = out / "physical_cmb_peak_features.png"
    frontier_path = out / "strict_neutral_gate_coincidence.png"
    rank_selector_path = out / "strict_neutral_rank_selector_diagnostics.png"
    near_miss_path = out / "strict_neutral_near_miss_frontier.png"
    near_miss_csv_path = out / "strict_neutral_near_miss_frontier.csv"

    best_oph = cmb_output.get("best_oph_diagnostic_model") or {}
    peak_summary = cmb_output.get("best_oph_peak_feature_summary") or {}
    _plot_tt_comparison(plt, comparison_path, lcdm_rows, scale_rows, finite_rows)
    _plot_best_residuals(plt, residual_path, residual_rows)
    _plot_peak_features(plt, peak_path, peak_rows, str(best_oph.get("model_id") or ""))
    _plot_neutral_coincidence(plt, frontier_path, neutral_sweep)
    _plot_rank_selector_diagnostics(plt, rank_selector_path, neutral_sweep)
    near_miss_rows = _write_neutral_near_miss_rows(near_miss_csv_path, neutral_sweep)
    _plot_neutral_near_miss(plt, near_miss_path, near_miss_rows)

    residual_summary = cmb_output.get("best_oph_residual_summary") or {}
    coincidence = neutral_sweep.get("gate_coincidence_summary") or {}
    best_near_miss = near_miss_rows[0] if near_miss_rows else {}
    summary = {
        "mode": "cmb_static_plots_v0",
        "run_dir": str(run),
        "out_dir": str(out),
        "physical_cmb_prediction": bool(claims.get("physical_cmb_prediction", False)),
        "physical_cmb_output_comparison_receipt": bool(
            cmb_output.get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT", False)
        ),
        "best_oph_model": best_oph.get("model_id"),
        "best_oph_chi2_per_bin": best_oph.get("amplitude_fit_chi2_per_bin"),
        "best_oph_residual_bin_count": int(residual_summary.get("bin_count") or 0),
        "best_oph_rms_sigma_residual": residual_summary.get("rms_sigma_residual"),
        "best_oph_peak_count": int(peak_summary.get("peak_count") or 0),
        "best_oph_mean_abs_peak_ell_delta": peak_summary.get("mean_abs_peak_ell_delta"),
        "best_oph_mean_abs_peak_height_fractional_delta": peak_summary.get(
            "mean_abs_peak_height_fractional_delta"
        ),
        "strict_neutral_bulk": bool(claims.get("strict_neutral_bulk", False)),
        "residual_sweep_case_count": int(neutral_sweep.get("case_count") or 0),
        "residual_sweep_spatial_h3_count": int(coincidence.get("spatial_h3_geometry_count") or 0),
        "residual_sweep_nontrivial_rank3_count": int(
            coincidence.get("nontrivial_rank3_selector_count") or 0
        ),
        "residual_sweep_model_order_rank3_count": int(
            (neutral_sweep.get("rank_obstruction_summary") or {}).get("model_order_rank3_selector_count") or 0
        ),
        "residual_sweep_nontrivial_model_order_rank3_count": int(
            (neutral_sweep.get("rank_obstruction_summary") or {}).get(
                "nontrivial_model_order_rank3_selector_count"
            )
            or 0
        ),
        "residual_sweep_spatial_h3_nontrivial_rank3_coincidence_count": int(
            coincidence.get("spatial_h3_nontrivial_rank3_selector_count") or 0
        ),
        "strict_neutral_near_miss_count": len(near_miss_rows),
        "strict_neutral_best_near_miss_gate_score": _int_or_none(
            best_near_miss.get("strict_gate_score")
        ),
        "strict_neutral_best_near_miss_dimension_error": _float_or_none(
            best_near_miss.get("dimension_error")
        ),
        "strict_neutral_best_near_miss_nontrivial_rank3_ev": _float_or_none(
            best_near_miss.get("nontrivial_rank3_cumulative_explained_variance")
        ),
        "strict_neutral_best_near_miss_missing_gates": best_near_miss.get("missing_strict_gates", ""),
        "model_comparison_count": len(comparison_rows),
        "files": [
            comparison_path.name,
            residual_path.name,
            peak_path.name,
            frontier_path.name,
            rank_selector_path.name,
            near_miss_path.name,
            near_miss_csv_path.name,
        ],
        "claim_boundary": (
            "Static measurement-facing plots. They visualize physical-unit CMB output comparisons "
            "and neutral-gate diagnostics, but do not promote physical_cmb_prediction or "
            "strict_neutral_bulk."
        ),
    }
    (out / "cmb_static_plots_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def _plot_tt_comparison(
    plt: Any,
    out: Path,
    lcdm_rows: list[dict[str, str]],
    scale_rows: list[dict[str, str]],
    finite_rows: list[dict[str, str]],
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=150)
    ells = _series(lcdm_rows, "ell")
    observed = _series(lcdm_rows, "observed_D_ell")
    sigma = _series(lcdm_rows, "sigma_D_ell")
    if ells and observed:
        ax.errorbar(
            ells,
            observed,
            yerr=sigma if len(sigma) == len(observed) else None,
            fmt=".",
            markersize=3.5,
            linewidth=0.5,
            color="#2f3a44",
            ecolor="#aeb8c2",
            elinewidth=0.45,
            alpha=0.85,
            label="Planck binned TT",
        )
    _plot_line(ax, lcdm_rows, "amplitude_fit_camb_D_ell", "LCDM baseline", "#4b6cb7")
    _plot_line(ax, scale_rows, "scale_compressed_scalar_tilt_D_ell", "OPH scale-compressed tilt", "#c26a2c")
    _plot_line(ax, scale_rows, "scale_compressed_ir_kernel_D_ell", "OPH scale-compressed IR", "#9367a8")
    _plot_line(ax, finite_rows, "finite_repair_clock_scalar_tilt_D_ell", "OPH finite-repair tilt", "#2a9d8f")
    ax.set_title("Measurement-comparable CMB TT outputs")
    ax.set_xlabel("ell")
    ax.set_ylabel("D_ell [microK^2]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_best_residuals(plt: Any, out: Path, residual_rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=150)
    ells = _series(residual_rows, "ell")
    residuals = _series(residual_rows, "residual_sigma")
    if ells and residuals:
        ax.axhline(0.0, color="#2f3a44", linewidth=1.0)
        ax.axhline(2.0, color="#c26a2c", linewidth=0.8, linestyle="--")
        ax.axhline(-2.0, color="#c26a2c", linewidth=0.8, linestyle="--")
        ax.plot(ells, residuals, color="#4b6cb7", linewidth=1.4)
        ax.scatter(ells, residuals, color="#4b6cb7", s=10)
    ax.set_title("Best OPH diagnostic CMB residuals")
    ax.set_xlabel("ell")
    ax.set_ylabel("residual / sigma")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_peak_features(
    plt: Any,
    out: Path,
    peak_rows: list[dict[str, str]],
    best_model_id: str,
) -> None:
    selected = [
        row
        for row in peak_rows
        if not best_model_id or row.get("model_id") == best_model_id
    ]
    if not selected:
        selected = [row for row in peak_rows if row.get("model_role") == "oph_diagnostic"] or peak_rows
    selected = sorted(
        selected,
        key=lambda row: (
            str(row.get("model_id") or ""),
            _float_value(row.get("peak_index"), default=0.0),
        ),
    )
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), dpi=150, sharex=True)
    fig.suptitle("Best OPH diagnostic CMB peak features")
    labels: list[str] = []
    ell_deltas: list[float] = []
    height_deltas: list[float] = []
    for row in selected:
        peak_index = _int_or_none(row.get("peak_index"))
        ell_delta = _float_or_none(row.get("ell_delta"))
        height_delta = _float_or_none(row.get("fractional_D_ell_delta"))
        if peak_index is None or ell_delta is None or height_delta is None:
            continue
        labels.append(str(peak_index))
        ell_deltas.append(ell_delta)
        height_deltas.append(height_delta)
    xs = list(range(len(labels)))
    if xs:
        colors = ["#4b6cb7" if value >= 0 else "#c26a2c" for value in ell_deltas]
        axes[0].bar(xs, ell_deltas, color=colors)
        axes[1].bar(xs, height_deltas, color=["#2a9d8f" if value >= 0 else "#7a3b3f" for value in height_deltas])
        for ax in axes:
            ax.axhline(0.0, color="#2f3a44", linewidth=0.8)
            ax.grid(True, axis="y", alpha=0.25)
        axes[1].set_xticks(xs, labels)
    else:
        axes[0].text(0.5, 0.5, "No peak feature rows", ha="center", va="center", transform=axes[0].transAxes)
        axes[1].axis("off")
    axes[0].set_ylabel("model ell - observed ell")
    axes[1].set_ylabel("fractional height delta")
    axes[1].set_xlabel("peak index")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_neutral_coincidence(plt: Any, out: Path, sweep: dict[str, Any]) -> None:
    coincidence = sweep.get("gate_coincidence_summary") or {}
    labels = [
        "spatial H3",
        "nontriv rank3",
        "H3 + nontriv",
        "strict H3",
    ]
    values = [
        int(coincidence.get("spatial_h3_geometry_count") or 0),
        int(coincidence.get("nontrivial_rank3_selector_count") or 0),
        int(coincidence.get("spatial_h3_nontrivial_rank3_selector_count") or 0),
        int(coincidence.get("strict_h3_candidate_count") or 0),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=150)
    colors = ["#4b6cb7", "#c26a2c", "#2a9d8f", "#7a3b3f"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Strict-neutral gate coincidence")
    ax.set_ylabel("case count")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(values + [1]) * 0.015,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_rank_selector_diagnostics(plt: Any, out: Path, sweep: dict[str, Any]) -> None:
    rank = sweep.get("rank_obstruction_summary") or {}
    labels = [
        "largest-gap rank3",
        "model-order rank3",
        "nontriv gap rank3",
        "nontriv model-order rank3",
        "strict H3",
    ]
    values = [
        int(rank.get("rank3_selector_count") or 0),
        int(rank.get("model_order_rank3_selector_count") or 0),
        int(rank.get("nontrivial_rank3_selector_count") or 0),
        int(rank.get("nontrivial_model_order_rank3_selector_count") or 0),
        int(sweep.get("strict_h3_candidate_count") or 0),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 5.4), dpi=150)
    colors = ["#4b6cb7", "#9367a8", "#2a9d8f", "#c26a2c", "#7a3b3f"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Neutral rank-selector diagnostics")
    ax.set_ylabel("case count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=18)
    dominant_gap = rank.get("dominant_largest_gap_rank")
    dominant_order = rank.get("dominant_model_order_consensus_rank")
    subtitle = f"dominant largest-gap rank={dominant_gap}; model-order consensus rank={dominant_order}"
    ax.text(0.01, 0.98, subtitle, transform=ax.transAxes, va="top", fontsize=8, color="#3f4852")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(values + [1]) * 0.015,
            str(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _write_neutral_near_miss_rows(out: Path, sweep: dict[str, Any], *, limit: int = 48) -> list[dict[str, Any]]:
    rows = sweep.get("rows") if isinstance(sweep.get("rows"), list) else []
    ranked = sorted((_neutral_near_miss_row(row) for row in rows), key=_near_miss_sort_key, reverse=True)
    output_rows = ranked[: max(0, int(limit))]
    fieldnames = [
        "strict_gate_score",
        "dimension_error",
        "source_run_dir",
        "seed",
        "max_model_points",
        "k_neighbors",
        "remove_modes",
        "spatial_3d_candidate",
        "strict_h3_candidate",
        "h3_geometry",
        "rank3_selector",
        "nontrivial_rank3_selector",
        "selected_model",
        "median_dimension",
        "rank3_cumulative_explained_variance",
        "nontrivial_rank3_cumulative_explained_variance",
        "largest_gap_rank",
        "nontrivial_largest_gap_rank",
        "effective_rank",
        "nontrivial_effective_rank",
        "missing_strict_gates",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    return output_rows


def _neutral_near_miss_row(row: Any) -> dict[str, Any]:
    row = row if isinstance(row, dict) else {}
    median_dimension = _float_or_none(row.get("median_dimension"))
    dimension_error = abs(median_dimension - 3.0) if median_dimension is not None else None
    h3_geometry = bool(
        row.get("spatial_3d_candidate", False)
        and row.get("selected_model") == "H3"
        and row.get("h3_beats_s2", False)
        and row.get("h3_beats_h2_h4", False)
        and row.get("s2_leakage_pass", False)
    )
    gates = {
        "graph_receipt": bool(row.get("graph_geometry_receipt", False) or row.get("residual_graph_receipt", False)),
        "spatial_3d_candidate": bool(row.get("spatial_3d_candidate", False)),
        "h3_model": row.get("selected_model") == "H3",
        "h3_beats_s2": bool(row.get("h3_beats_s2", False)),
        "h3_beats_h2_h4": bool(row.get("h3_beats_h2_h4", False)),
        "s2_leakage_pass": bool(row.get("s2_leakage_pass", False)),
        "independent_rank3_selector": bool(row.get("rank3_selector", False)),
        "strict_h3_candidate": bool(row.get("strict_h3_candidate", False)),
    }
    missing = [key for key, passed in gates.items() if not passed]
    return {
        "strict_gate_score": int(sum(1 for passed in gates.values() if passed)),
        "dimension_error": dimension_error,
        "source_run_dir": row.get("source_run_dir"),
        "seed": row.get("seed"),
        "max_model_points": row.get("max_model_points"),
        "k_neighbors": row.get("k_neighbors"),
        "remove_modes": row.get("remove_modes"),
        "spatial_3d_candidate": bool(row.get("spatial_3d_candidate", False)),
        "strict_h3_candidate": bool(row.get("strict_h3_candidate", False)),
        "h3_geometry": h3_geometry,
        "rank3_selector": bool(row.get("rank3_selector", False)),
        "nontrivial_rank3_selector": bool(row.get("nontrivial_rank3_selector", False)),
        "selected_model": row.get("selected_model"),
        "median_dimension": median_dimension,
        "rank3_cumulative_explained_variance": _float_or_none(
            row.get("rank3_cumulative_explained_variance")
        ),
        "nontrivial_rank3_cumulative_explained_variance": _float_or_none(
            row.get("nontrivial_rank3_cumulative_explained_variance")
        ),
        "largest_gap_rank": row.get("largest_gap_rank"),
        "nontrivial_largest_gap_rank": row.get("nontrivial_largest_gap_rank"),
        "effective_rank": _float_or_none(row.get("effective_rank")),
        "nontrivial_effective_rank": _float_or_none(row.get("nontrivial_effective_rank")),
        "missing_strict_gates": ";".join(missing),
    }


def _near_miss_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(bool(row.get("strict_h3_candidate", False))),
        float(bool(row.get("rank3_selector", False))),
        float(bool(row.get("h3_geometry", False))),
        float(bool(row.get("nontrivial_rank3_selector", False))),
        float(row.get("strict_gate_score") or 0),
        -_float_value(row.get("dimension_error"), default=1.0e9),
        _float_value(row.get("nontrivial_rank3_cumulative_explained_variance")),
        _float_value(row.get("rank3_cumulative_explained_variance")),
        -_float_value(row.get("nontrivial_effective_rank"), default=1.0e9),
    )


def _plot_neutral_near_miss(plt: Any, out: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=150)
    xs = []
    ys = []
    colors = []
    sizes = []
    for row in rows:
        dimension_error = _float_or_none(row.get("dimension_error"))
        nontrivial_ev = _float_or_none(row.get("nontrivial_rank3_cumulative_explained_variance"))
        if dimension_error is None or nontrivial_ev is None:
            continue
        xs.append(dimension_error)
        ys.append(nontrivial_ev)
        colors.append(float(row.get("strict_gate_score") or 0))
        sizes.append(72.0 if row.get("h3_geometry") else 36.0)
    if xs and ys:
        scatter = ax.scatter(xs, ys, c=colors, s=sizes, cmap="viridis", alpha=0.82, edgecolors="#2f3a44")
        fig.colorbar(scatter, ax=ax, label="strict gate score")
    ax.axhline(0.50, color="#c26a2c", linestyle="--", linewidth=0.9, label="nontrivial rank-3 EV gate")
    ax.axvline(0.0, color="#2f3a44", linewidth=0.8)
    ax.set_title("Strict-neutral near-miss frontier")
    ax.set_xlabel("|median dimension - 3|")
    ax.set_ylabel("nontrivial rank-3 cumulative EV")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_line(ax: Any, rows: list[dict[str, str]], column: str, label: str, color: str) -> None:
    xs = _series(rows, "ell")
    ys = _series(rows, column)
    if xs and ys and len(xs) == len(ys):
        ax.plot(xs, ys, linewidth=1.45, color=color, label=label)


def _series(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _float_or_none(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_value(value: Any, *, default: float = 0.0) -> float:
    parsed = _float_or_none(value)
    return float(parsed) if parsed is not None else float(default)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
