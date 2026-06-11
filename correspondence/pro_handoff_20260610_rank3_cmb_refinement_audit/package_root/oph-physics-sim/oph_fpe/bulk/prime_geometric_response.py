from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np


PRIME_GEOMETRIC_OBSERVABLES = ("checkpoint_class", "stable_flag", "s3_sector_class")
SUPPORT_VISIBLE_OBSERVABLES = (
    "checkpoint_class",
    "stable_flag",
    "s3_sector_class",
    "repair_load_bucket",
)
REPAIR_OBSERVABLES = ("repair_load_bucket",)
MODULAR_FEATURE_TYPES = (
    "class_distribution_delta",
    "target_distribution_delta",
    "class_log_odds_delta",
    "transition_matrix_delta",
    "entropy_delta",
    "sector_preservation_delta",
    "change_probability_delta",
)
CONTROL_QUOTIENT_CONTROLS = (
    "s2_boundary_control",
    "no_modular_flow_control",
    "wrong_scale_control_1x",
    "wrong_scale_control_pi",
    "wrong_scale_control_4pi",
)


def attach_prime_geometric_response_to_rows(
    observer_rows: list[dict[str, Any]],
    response_kernel: dict[str, Any],
    *,
    spectrum_width: int = 64,
    component_bins: int = 8,
) -> dict[str, Any]:
    """Attach paper-aligned modular response spectra to observer rows.

    The OPH Lorentz/BW target lives on the overlap-generated geometric subnet,
    not on record/pointer auxiliaries. This helper therefore derives observer
    features only from the cached support-visible modular-response kernel. It
    never reads S2 axes, H3 fitted coordinates, support nodes, radial depth, or
    modular depth.
    """

    matrix = np.asarray(response_kernel.get("matrix", np.zeros((0, 0))), dtype=float)
    observer_ids = [int(value) for value in response_kernel.get("observer_ids", [])]
    feature_rows = list(response_kernel.get("feature_rows", []))
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0 or len(observer_ids) != matrix.shape[0]:
        return _empty_report("empty_or_malformed_response_kernel")
    if len(feature_rows) != matrix.shape[1]:
        return _empty_report("missing_or_mismatched_feature_rows")

    prime_matrix, prime_report = grouped_modular_response_matrix(
        matrix,
        feature_rows,
        observables=PRIME_GEOMETRIC_OBSERVABLES,
    )
    prime_quotient_matrix, prime_quotient_report = control_quotient_response_matrix(
        matrix,
        response_kernel,
        feature_rows,
        observables=PRIME_GEOMETRIC_OBSERVABLES,
    )
    support_matrix, support_report = grouped_modular_response_matrix(
        matrix,
        feature_rows,
        observables=SUPPORT_VISIBLE_OBSERVABLES,
    )
    repair_matrix, repair_report = grouped_modular_response_matrix(
        matrix,
        feature_rows,
        observables=REPAIR_OBSERVABLES,
    )
    prime_spectrum, prime_embedding_report = response_component_spectrum(
        prime_matrix,
        width=spectrum_width,
    )
    prime_quotient_spectrum, prime_quotient_embedding_report = response_component_spectrum(
        prime_quotient_matrix,
        width=spectrum_width,
    )
    support_spectrum, support_embedding_report = response_component_spectrum(
        support_matrix,
        width=spectrum_width,
    )
    repair_spectrum, repair_embedding_report = response_component_spectrum(
        repair_matrix,
        width=min(32, int(spectrum_width)),
    )

    row_by_id = {
        int(row.get("observer_id", -1)): row
        for row in observer_rows
        if row.get("view_type") == "patch_observer"
    }
    attached = 0
    for row_index, observer_id in enumerate(observer_ids):
        row = row_by_id.get(int(observer_id))
        if row is None:
            continue
        row["prime_geometric_modular_spectrum"] = _json_float_list(prime_spectrum[row_index])
        row["prime_geometric_control_quotient_spectrum"] = _json_float_list(
            prime_quotient_spectrum[row_index]
        )
        row["support_visible_modular_spectrum"] = _json_float_list(support_spectrum[row_index])
        row["repair_modular_spectrum"] = _json_float_list(repair_spectrum[row_index])
        histograms = dict(row.get("modular_response_histograms", {}) or {})
        _attach_component_bins(
            histograms,
            "prime_geometric_modular",
            prime_spectrum[row_index],
            bins=int(component_bins),
            max_components=min(8, int(spectrum_width)),
        )
        _attach_component_bins(
            histograms,
            "prime_geometric_control_quotient",
            prime_quotient_spectrum[row_index],
            bins=int(component_bins),
            max_components=min(8, int(spectrum_width)),
        )
        _attach_component_bins(
            histograms,
            "support_visible_modular",
            support_spectrum[row_index],
            bins=int(component_bins),
            max_components=min(8, int(spectrum_width)),
        )
        row["modular_response_histograms"] = histograms
        attached += 1
    return {
        "mode": "prime_geometric_modular_response_attachment_v0",
        "observer_count": int(matrix.shape[0]),
        "attached_observer_count": int(attached),
        "input_feature_count": int(matrix.shape[1]),
        "spectrum_width": int(spectrum_width),
        "component_bins": int(component_bins),
        "prime_geometric": prime_report | {"embedding": prime_embedding_report},
        "prime_geometric_control_quotient": prime_quotient_report
        | {"embedding": prime_quotient_embedding_report},
        "support_visible": support_report | {"embedding": support_embedding_report},
        "repair_only": repair_report | {"embedding": repair_embedding_report},
        "primary_forbidden_features": [
            "S2 axes",
            "support_nodes",
            "H3 fitted points",
            "cap target coordinates",
            "radial_depth",
            "modular_depth",
        ],
        "claim_boundary": (
            "Observer-visible modular-response attachment for neutral bulk diagnostics. The prime "
            "geometric spectrum excludes record-family and repair-load auxiliaries; support-visible "
            "and repair-only spectra are emitted as diagnostics. The control-quotient spectrum removes "
            "observer-level directions spanned by finite-regulator controls before compression. This "
            "does not prove 3D bulk by itself."
        ),
    }


def grouped_modular_response_matrix(
    matrix: np.ndarray,
    feature_rows: list[dict[str, Any]],
    *,
    observables: tuple[str, ...],
    feature_types: tuple[str, ...] = MODULAR_FEATURE_TYPES,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(matrix, dtype=float)
    observable_set = {str(value) for value in observables}
    feature_type_set = {str(value) for value in feature_types}
    groups: dict[tuple[int, int, str, str], list[int]] = {}
    for index, row in enumerate(feature_rows):
        observable = str(row.get("observable", row.get("field", "")))
        feature_type = str(row.get("feature_type", ""))
        if observable not in observable_set or feature_type not in feature_type_set:
            continue
        key = (
            int(row.get("cap_index", -1)),
            int(row.get("time_index", -1)),
            observable,
            feature_type,
        )
        groups.setdefault(key, []).append(int(index))
    ordered = sorted(groups.items(), key=lambda item: item[0])
    if not ordered:
        return np.zeros((matrix.shape[0], 0), dtype=float), {
            "observables": sorted(observable_set),
            "feature_types": sorted(feature_type_set),
            "selected_feature_count": 0,
            "grouped_feature_count": 0,
            "reason": "no_matching_features",
        }
    columns = [
        np.mean(matrix[:, np.asarray(indices, dtype=np.int64)], axis=1)
        for _key, indices in ordered
    ]
    grouped = np.vstack(columns).T
    return grouped, {
        "observables": sorted(observable_set),
        "feature_types": sorted(feature_type_set),
        "selected_feature_count": int(sum(len(indices) for _key, indices in ordered)),
        "grouped_feature_count": int(grouped.shape[1]),
        "group_keys_sample": [
            {
                "cap_index": int(key[0]),
                "time_index": int(key[1]),
                "observable": str(key[2]),
                "feature_type": str(key[3]),
                "source_feature_count": int(len(indices)),
            }
            for key, indices in ordered[:64]
        ],
    }


def response_component_spectrum(matrix: np.ndarray, *, width: int) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(matrix, dtype=float)
    width = max(1, int(width))
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0] if matrix.ndim == 2 else 0, width), dtype=float), {
            "mode": "svd_component_spectrum",
            "component_count": 0,
            "reason": "empty_matrix",
        }
    work = np.where(np.isfinite(matrix), matrix, 0.0)
    work = work - np.mean(work, axis=0, keepdims=True)
    scale = np.std(work, axis=0, keepdims=True)
    scale[scale < 1e-9] = 1.0
    work = work / scale
    try:
        u, singular_values, _vh = np.linalg.svd(work, full_matrices=False)
        component_count = min(width, int(u.shape[1]), int(singular_values.size))
        spectrum = u[:, :component_count] * singular_values[:component_count][None, :]
    except np.linalg.LinAlgError:
        component_count = min(width, int(work.shape[1]))
        spectrum = work[:, :component_count]
        singular_values = np.zeros(0, dtype=float)
    if component_count > 0:
        spectrum = spectrum - np.mean(spectrum, axis=0, keepdims=True)
        component_scale = np.std(spectrum, axis=0, keepdims=True)
        component_scale[component_scale < 1e-9] = 1.0
        spectrum = spectrum / component_scale
    out = np.zeros((matrix.shape[0], width), dtype=float)
    out[:, :component_count] = spectrum[:, :component_count]
    return out, {
        "mode": "svd_component_spectrum",
        "input_feature_count": int(matrix.shape[1]),
        "component_count": int(component_count),
        "output_width": int(width),
        "rank_selection": _svd_rank_selection_report(singular_values),
        "claim_boundary": (
            "Data-derived compression of observer-visible modular-response rows. It uses no target geometry; "
            "distances computed from it must still pass leakage and shuffled controls."
        ),
    }


def _svd_rank_selection_report(singular_values: np.ndarray, *, max_rank: int = 64) -> dict[str, Any]:
    singular_values = np.asarray(singular_values, dtype=float).reshape(-1)
    singular_values = singular_values[np.isfinite(singular_values) & (singular_values > 0.0)]
    if singular_values.size == 0:
        return {
            "mode": "svd_rank_selection_v0",
            "component_count": 0,
            "independent_rank3_selector_receipt": False,
            "reason": "no_positive_singular_values",
        }
    energy = singular_values * singular_values
    total = float(np.sum(energy))
    if total <= 0.0 or not np.isfinite(total):
        fractions = np.zeros_like(energy)
    else:
        fractions = energy / total
    cumulative = np.cumsum(fractions)
    probabilities = fractions[fractions > 0.0]
    spectral_entropy = float(-np.sum(probabilities * np.log(probabilities))) if probabilities.size else 0.0
    effective_rank = float(np.exp(spectral_entropy)) if probabilities.size else 0.0
    denom = float(np.sum(fractions * fractions))
    participation_rank = float(1.0 / denom) if denom > 0.0 else 0.0
    gap_rank = _largest_singular_gap_rank(singular_values, max_rank=max_rank)
    chord_rank = _singular_chord_elbow_rank(singular_values, max_rank=max_rank)
    selector_votes = [rank for rank in (gap_rank, chord_rank) if rank is not None]
    independent_rank3 = bool(selector_votes and selector_votes.count(3) >= 2)
    return {
        "mode": "svd_rank_selection_v0",
        "component_count": int(singular_values.size),
        "singular_values_top": _json_float_list(singular_values[: min(max_rank, singular_values.size)]),
        "explained_variance_fraction_top": _json_float_list(fractions[: min(max_rank, fractions.size)]),
        "cumulative_explained_variance_top": _json_float_list(cumulative[: min(max_rank, cumulative.size)]),
        "rank3_cumulative_explained_variance": float(cumulative[2]) if cumulative.size >= 3 else None,
        "rank90": _rank_for_cumulative(cumulative, 0.90),
        "rank95": _rank_for_cumulative(cumulative, 0.95),
        "spectral_entropy": spectral_entropy,
        "effective_rank": effective_rank,
        "participation_rank": participation_rank,
        "largest_gap_rank": gap_rank,
        "chord_elbow_rank": chord_rank,
        "independent_rank3_selector_receipt": independent_rank3,
        "claim_boundary": (
            "Rank selector from observer-visible modular-response singular values only. It is independent "
            "of downstream dimension estimation, S2 axes, H3 coordinates, and target rank. It is diagnostic "
            "unless stable across refinement and controls."
        ),
    }


def _rank_for_cumulative(cumulative: np.ndarray, threshold: float) -> int | None:
    hits = np.where(cumulative >= float(threshold))[0]
    return int(hits[0] + 1) if hits.size else None


def _largest_singular_gap_rank(singular_values: np.ndarray, *, max_rank: int) -> int | None:
    count = min(int(max_rank), int(singular_values.size) - 1)
    if count <= 0:
        return None
    ratios = singular_values[:count] / np.maximum(singular_values[1 : count + 1], 1e-12)
    if ratios.size == 0 or not np.any(np.isfinite(ratios)):
        return None
    return int(np.nanargmax(ratios) + 1)


def _singular_chord_elbow_rank(singular_values: np.ndarray, *, max_rank: int) -> int | None:
    count = min(int(max_rank), int(singular_values.size))
    if count < 3:
        return None
    x = np.arange(1, count + 1, dtype=float)
    y = np.log(np.maximum(singular_values[:count], 1e-12))
    x1, y1 = float(x[0]), float(y[0])
    x2, y2 = float(x[-1]), float(y[-1])
    denom = math.hypot(y2 - y1, x2 - x1)
    if denom <= 1e-12:
        return None
    distances = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
    return int(np.nanargmax(distances) + 1)


def control_quotient_response_matrix(
    matrix: np.ndarray,
    response_kernel: dict[str, Any],
    feature_rows: list[dict[str, Any]],
    *,
    observables: tuple[str, ...],
    feature_types: tuple[str, ...] = MODULAR_FEATURE_TYPES,
    max_control_rank: int = 24,
    singular_value_floor: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project raw observer response away from cached regulator-control directions.

    The compact paper separates the geometric subnet from auxiliary and
    regulator-visible kernels. This is still only a finite diagnostic, but it
    is stricter than the raw response SVD: the neutral-bulk feature is the part
    of the observer-visible modular response not explained by S2-boundary,
    no-flow, and wrong-normalization control responses.
    """

    raw_grouped, raw_report = grouped_modular_response_matrix(
        matrix,
        feature_rows,
        observables=observables,
        feature_types=feature_types,
    )
    if raw_grouped.size == 0:
        return raw_grouped, {
            "mode": "control_quotient_response_matrix_v0",
            "raw": raw_report,
            "reason": "empty_raw_grouped_matrix",
            "control_count": 0,
            "control_basis_rank": 0,
        }
    control_grouped: list[np.ndarray] = []
    control_rows: list[dict[str, Any]] = []
    for name, control in _iter_control_matrices(response_kernel):
        grouped, report = grouped_modular_response_matrix(
            control,
            feature_rows,
            observables=observables,
            feature_types=feature_types,
        )
        if grouped.shape != raw_grouped.shape or grouped.size == 0:
            control_rows.append(
                {
                    "name": name,
                    "used": False,
                    "reason": "shape_mismatch_or_empty",
                    "shape": list(grouped.shape),
                }
            )
            continue
        control_grouped.append(grouped)
        control_rows.append(
            {
                "name": name,
                "used": True,
                "shape": list(grouped.shape),
                "grouped_feature_count": report.get("grouped_feature_count"),
            }
        )
    if not control_grouped:
        return raw_grouped, {
            "mode": "control_quotient_response_matrix_v0",
            "raw": raw_report,
            "control_count": 0,
            "control_basis_rank": 0,
            "reason": "no_usable_controls",
            "claim_boundary": "No control quotient was applied because no compatible cached controls were available.",
        }
    control_design = np.hstack([_standardized_columns(value) for value in control_grouped])
    control_design = np.where(np.isfinite(control_design), control_design, 0.0)
    control_design = control_design - np.mean(control_design, axis=0, keepdims=True)
    try:
        u, singular_values, _vh = np.linalg.svd(control_design, full_matrices=False)
    except np.linalg.LinAlgError:
        u = np.zeros((raw_grouped.shape[0], 0), dtype=float)
        singular_values = np.zeros(0, dtype=float)
    positive = np.where(singular_values > float(singular_value_floor))[0]
    max_rank = max(0, min(int(max_control_rank), int(raw_grouped.shape[0]) - 1, int(positive.size)))
    if max_rank <= 0:
        residual = raw_grouped - np.mean(raw_grouped, axis=0, keepdims=True)
        basis_rank = 0
    else:
        q = u[:, :max_rank]
        centered = raw_grouped - np.mean(raw_grouped, axis=0, keepdims=True)
        residual = centered - q @ (q.T @ centered)
        basis_rank = int(max_rank)
    residual_energy = float(np.linalg.norm(residual))
    centered_energy = float(np.linalg.norm(raw_grouped - np.mean(raw_grouped, axis=0, keepdims=True)))
    return residual, {
        "mode": "control_quotient_response_matrix_v0",
        "raw": raw_report,
        "controls": control_rows,
        "control_count": int(sum(1 for row in control_rows if row.get("used"))),
        "control_basis_rank": int(basis_rank),
        "max_control_rank": int(max_control_rank),
        "singular_value_floor": float(singular_value_floor),
        "raw_centered_frobenius_norm": centered_energy,
        "residual_frobenius_norm": residual_energy,
        "residual_energy_fraction": float(residual_energy / centered_energy) if centered_energy > 1e-12 else 0.0,
        "claim_boundary": (
            "Finite diagnostic quotient: raw observer-visible prime-geometric modular responses are "
            "projected away from observer-level directions spanned by cached regulator controls. This "
            "approximates, but does not prove, the paper-side repair-invariant overlap-trivial quotient."
        ),
    }


def load_cached_response_kernel(run_dir: Path) -> dict[str, Any]:
    run = Path(run_dir)
    cache_path = run / "modular_response_kernel_cache.json"
    payload_path = run / "modular_response_kernel_payload.npz"
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    if not payload_path.exists():
        raise FileNotFoundError(payload_path)
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    with np.load(payload_path, allow_pickle=True) as payload:
        wrong_scale_controls: dict[str, np.ndarray] = {}
        for index, row in enumerate(cache.get("wrong_scale_controls", []) or []):
            label = str(row.get("label", index)) if isinstance(row, dict) else str(index)
            key = str(row.get("array_key", f"wrong_scale_control_{index}")) if isinstance(row, dict) else f"wrong_scale_control_{index}"
            if key in payload:
                wrong_scale_controls[label] = np.asarray(payload[key], dtype=float)
        out = {
            "matrix": np.asarray(payload["matrix"], dtype=float),
            "s2_boundary_control": _payload_array(payload, "s2_boundary_control"),
            "shuffled_control": _payload_array(payload, "shuffled_control"),
            "shuffled_response_control": _payload_array(payload, "shuffled_response_control"),
            "shuffled_observer_labels_control": _payload_array(payload, "shuffled_observer_labels_control"),
            "no_modular_flow_control": _payload_array(payload, "no_modular_flow_control"),
            "wrong_scale_controls": wrong_scale_controls,
        }
    out.update({
        "observer_ids": [int(value) for value in cache.get("observer_ids", [])],
        "feature_rows": list(cache.get("feature_rows", [])),
        "cap_count": int(cache.get("cap_count", 0) or 0),
        "time_count": int(cache.get("time_count", 0) or 0),
        "field_names": list(cache.get("field_names", [])),
    })
    return out


def write_prime_geometric_response_attachment(
    run_dir: Path,
    out: Path | None = None,
    *,
    spectrum_width: int = 64,
    component_bins: int = 8,
    backup: bool = True,
) -> dict[str, Any]:
    run = Path(run_dir)
    source = run / "observer_views.jsonl"
    if not source.exists():
        raise FileNotFoundError(source)
    rows = _read_jsonl(source)
    report = attach_prime_geometric_response_to_rows(
        rows,
        load_cached_response_kernel(run),
        spectrum_width=int(spectrum_width),
        component_bins=int(component_bins),
    )
    destination = Path(out) if out is not None else source
    if destination == source and backup:
        backup_path = run / "observer_views.before_prime_geometric_response.jsonl"
        if not backup_path.exists():
            shutil.copy2(source, backup_path)
        report["backup_observer_views_path"] = str(backup_path)
    _write_jsonl(destination, rows)
    report["observer_views_path"] = str(destination)
    report["source_run_dir"] = str(run)
    report_path = run / "prime_geometric_response_attachment_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _iter_control_matrices(response_kernel: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    matrix = np.asarray(response_kernel.get("matrix", np.zeros((0, 0))), dtype=float)
    rows: list[tuple[str, np.ndarray]] = []
    for key in ("s2_boundary_control", "no_modular_flow_control"):
        value = response_kernel.get(key)
        if value is None:
            continue
        control = np.asarray(value, dtype=float)
        if control.shape == matrix.shape:
            rows.append((key, control))
    wrong_scale_controls = response_kernel.get("wrong_scale_controls", {})
    if isinstance(wrong_scale_controls, dict):
        for label, value in sorted(wrong_scale_controls.items(), key=lambda item: str(item[0])):
            control = np.asarray(value, dtype=float)
            if control.shape != matrix.shape:
                continue
            normalized = _normalized_wrong_scale_label(str(label))
            rows.append((f"wrong_scale_control_{normalized}", control))
    return rows


def _normalized_wrong_scale_label(label: str) -> str:
    value = str(label).strip().lower().replace(" ", "_")
    value = value.replace("wrong_", "").replace("normalization", "")
    value = value.strip("_")
    aliases = {
        "1": "1x",
        "x1": "1x",
        "one": "1x",
        "pi": "pi",
        "π": "pi",
        "4pi": "4pi",
        "4_pi": "4pi",
        "four_pi": "4pi",
    }
    return aliases.get(value, value or "unknown")


def _standardized_columns(matrix: np.ndarray) -> np.ndarray:
    work = np.asarray(matrix, dtype=float)
    work = np.where(np.isfinite(work), work, 0.0)
    work = work - np.mean(work, axis=0, keepdims=True)
    scale = np.std(work, axis=0, keepdims=True)
    scale[scale < 1e-9] = 1.0
    return work / scale


def _payload_array(payload: Any, key: str) -> np.ndarray | None:
    if key not in payload:
        return None
    return np.asarray(payload[key], dtype=float)


def _attach_component_bins(
    histograms: dict[str, Any],
    prefix: str,
    values: np.ndarray,
    *,
    bins: int,
    max_components: int,
) -> None:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return
    bins = max(2, int(bins))
    max_components = max(1, min(int(max_components), int(values.size)))
    clipped = np.clip(values[:max_components], -3.0, 3.0)
    bucket_values = np.floor((clipped + 3.0) / 6.0 * bins).astype(int)
    bucket_values = np.clip(bucket_values, 0, bins - 1)
    for index, bucket in enumerate(bucket_values):
        histograms[f"{prefix}_component_{index}"] = {str(int(bucket)): 1.0}


def _json_float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _empty_report(reason: str) -> dict[str, Any]:
    return {
        "mode": "prime_geometric_modular_response_attachment_v0",
        "attached_observer_count": 0,
        "reason": str(reason),
        "claim_boundary": "No observer-visible modular response spectrum was attached.",
    }
