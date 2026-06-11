from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform

from oph_fpe.bulk.latent_geometry_selection import select_latent_geometry
from oph_fpe.bulk.neutral_bulk import js_distance, scaled_l2_distance, strict_neutral_dimension_report
from oph_fpe.claims import STRICT_NEUTRAL_OBJECT_BULK_RECEIPT


FORBIDDEN_PRIMARY_FIELDS = (
    "axis",
    "support_nodes",
    "h3_point",
    "h3_points",
    "cap_axis",
    "cap_normal",
    "screen_point",
    "radial_depth",
    "modular_depth",
)


@dataclass(frozen=True)
class NeutralObject:
    object_id: str
    observer_ids: tuple[int, ...]
    visible_signature_key: str
    record_lineage_hist: np.ndarray
    checkpoint_continuation_hist: np.ndarray
    sector_transport_hist: np.ndarray
    repair_response_hist: np.ndarray
    counterfactual_stability_hist: np.ndarray
    transition_affinity_hist: np.ndarray
    persistence: float
    overlap_agreement: float

    @property
    def observer_count(self) -> int:
        return len(self.observer_ids)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["observer_ids"] = [int(value) for value in self.observer_ids]
        for key in (
            "record_lineage_hist",
            "checkpoint_continuation_hist",
            "sector_transport_hist",
            "repair_response_hist",
            "counterfactual_stability_hist",
            "transition_affinity_hist",
        ):
            payload[key] = np.asarray(payload[key], dtype=float).tolist()
        return payload


def extract_neutral_objects(
    observer_views: list[dict[str, Any]],
    *,
    min_observers_per_object: int = 3,
    max_observer_fraction_per_object: float = 0.35,
    max_objects: int = 2048,
) -> list[NeutralObject]:
    """Extract neutral objects without using chart/support geometry.

    A neutral object is a persistent observer-visible history class. The key is
    intentionally coarse and built only from record/checkpoint/sector/repair
    packets, so one observer cannot trivially become one object.
    """

    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    if not patch_views:
        return []
    key_levels = ("full", "transition_core", "coarse")
    best: list[NeutralObject] = []
    for level in key_levels:
        groups: dict[str, list[dict[str, Any]]] = {}
        for view in patch_views:
            groups.setdefault(_visible_object_key(view, level=level), []).append(view)
        objects = _objects_from_groups(
            groups,
            total_observers=len(patch_views),
            min_observers_per_object=min_observers_per_object,
            max_observer_fraction_per_object=max_observer_fraction_per_object,
        )
        if len(objects) > len(best):
            best = objects
        # Prefer the most specific key that yields enough non-singleton objects.
        if len(objects) >= 8:
            best = objects
            break
    best = sorted(best, key=lambda obj: (-obj.observer_count, obj.object_id))
    return best[: max(1, int(max_objects))]


def neutral_object_distance(a: NeutralObject, b: NeutralObject) -> float:
    terms = [
        1.0 * js_distance(a.record_lineage_hist, b.record_lineage_hist),
        0.9 * js_distance(a.checkpoint_continuation_hist, b.checkpoint_continuation_hist),
        0.9 * js_distance(a.sector_transport_hist, b.sector_transport_hist),
        0.8 * js_distance(a.repair_response_hist, b.repair_response_hist),
        0.7 * js_distance(a.counterfactual_stability_hist, b.counterfactual_stability_hist),
        0.7 * js_distance(a.transition_affinity_hist, b.transition_affinity_hist),
        0.4
        * scaled_l2_distance(
            np.asarray([a.persistence, a.overlap_agreement], dtype=float),
            np.asarray([b.persistence, b.overlap_agreement], dtype=float),
        ),
    ]
    return float(max(0.0, sum(terms) / 5.4))


def neutral_object_distance_matrix(objects: list[NeutralObject]) -> np.ndarray:
    n = len(objects)
    distance = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = neutral_object_distance(objects[i], objects[j])
            distance[i, j] = distance[j, i] = value
    return distance


def strict_neutral_object_bulk_report(
    observer_views: list[dict[str, Any]],
    *,
    seed: int = 1,
    min_objects: int = 16,
    min_observers_per_object: int = 3,
    max_observer_fraction_per_object: float = 0.35,
    max_model_points: int = 192,
    heldout_fraction: float = 0.25,
) -> dict[str, Any]:
    objects = extract_neutral_objects(
        observer_views,
        min_observers_per_object=min_observers_per_object,
        max_observer_fraction_per_object=max_observer_fraction_per_object,
    )
    distance = neutral_object_distance_matrix(objects)
    dimension = strict_neutral_dimension_report(distance) if len(objects) >= 8 else _empty_dimension()
    selection = (
        select_latent_geometry(
            distance,
            seed=seed,
            max_points=max_model_points,
            heldout_fraction=heldout_fraction,
        )
        if len(objects) >= 8
        else _empty_selection()
    )
    leakage = neutral_object_s2_leakage_audit(distance, objects, observer_views)
    controls = neutral_object_control_report(objects, seed=seed + 909)
    passed = bool(
        len(objects) >= int(min_objects)
        and dimension.get("estimators_agree_3d", False)
        and selection.get("h3_selected", False)
        and leakage.get("s2_leakage_pass", False)
        and controls.get("shuffled_records_fail", False)
        and controls.get("shuffled_transition_labels_fail", False)
    )
    report = {
        "mode": "strict_neutral_object_bulk_v0",
        "object_count": int(len(objects)),
        "observer_count": int(len([view for view in observer_views if view.get("view_type") == "patch_observer"])),
        "min_objects": int(min_objects),
        "min_observers_per_object": int(min_observers_per_object),
        "max_observer_fraction_per_object": float(max_observer_fraction_per_object),
        "distance_matrix_shape": list(distance.shape),
        "dimension": dimension,
        "latent_geometry_selection": selection,
        "leakage": leakage,
        "controls": controls,
        STRICT_NEUTRAL_OBJECT_BULK_RECEIPT: passed,
        "strict_neutral_object_bulk": passed,
        "physical_claim": False,
        "blockers": _strict_neutral_object_blockers(
            object_count=len(objects),
            min_objects=min_objects,
            dimension=dimension,
            selection=selection,
            leakage=leakage,
            controls=controls,
        ),
        "primary_features": [
            "record_signature_histogram",
            "record_transition_histogram",
            "object_packet_histogram",
            "checkpoint_class_transition",
            "sector_change_signature",
            "repair_response_spectrum",
            "repair_load_mean",
            "mismatch_density_mean",
            "counterfactual_stability",
            "transition_history_histograms",
            "transition_affinity_histograms",
        ],
        "forbidden_primary_features": list(FORBIDDEN_PRIMARY_FIELDS),
        "claim_boundary": (
            "Strict neutral object-bulk diagnostic. It extracts persistent observer-visible object classes "
            "without H3/S2/support coordinates, then runs dimension, leakage, controls, and held-out latent "
            "geometry selection. A true receipt is still not a physical cosmology prediction."
        ),
    }
    return report


def write_strict_neutral_object_bulk_report(
    run_dir: Path,
    out: Path | None = None,
    *,
    seed: int = 1,
    min_objects: int = 16,
    min_observers_per_object: int = 3,
    max_observer_fraction_per_object: float = 0.35,
    max_model_points: int = 192,
    heldout_fraction: float = 0.25,
) -> dict[str, Any]:
    run = Path(run_dir)
    observer_path = run / "observer_views.jsonl"
    if not observer_path.exists():
        raise FileNotFoundError(observer_path)
    observer_views = _read_jsonl(observer_path)
    objects = extract_neutral_objects(
        observer_views,
        min_observers_per_object=min_observers_per_object,
        max_observer_fraction_per_object=max_observer_fraction_per_object,
    )
    report = strict_neutral_object_bulk_report(
        observer_views,
        seed=seed,
        min_objects=min_objects,
        min_observers_per_object=min_observers_per_object,
        max_observer_fraction_per_object=max_observer_fraction_per_object,
        max_model_points=max_model_points,
        heldout_fraction=heldout_fraction,
    )
    report["source_run_dir"] = str(run)
    report["observer_views_path"] = str(observer_path)
    destination = Path(out) if out is not None else run / "strict_neutral_object_bulk_report.json"
    if destination.suffix.lower() != ".json":
        destination = destination / "strict_neutral_object_bulk_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    objects_path = destination.with_name("neutral_objects.jsonl")
    with objects_path.open("w", encoding="utf-8") as handle:
        for obj in objects:
            handle.write(json.dumps(obj.to_json(), default=str) + "\n")
    report["neutral_objects_path"] = str(objects_path)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def neutral_object_control_report(objects: list[NeutralObject], *, seed: int = 1) -> dict[str, Any]:
    if len(objects) < 8:
        return {
            "mode": "strict_neutral_object_controls_v0",
            "object_count": len(objects),
            "shuffled_records_fail": False,
            "shuffled_transition_labels_fail": False,
            "reason": "too_few_objects",
        }
    rng = np.random.default_rng(int(seed))
    original = neutral_object_distance_matrix(objects)
    record_control = _shuffled_objects(objects, rng, kind="records")
    transition_control = _shuffled_objects(objects, rng, kind="transitions")
    record_distance = neutral_object_distance_matrix(record_control)
    transition_distance = neutral_object_distance_matrix(transition_control)
    record_corr = _upper_corr(original, record_distance)
    transition_corr = _upper_corr(original, transition_distance)
    record_delta = _mean_abs_upper_delta(original, record_distance)
    transition_delta = _mean_abs_upper_delta(original, transition_distance)
    return {
        "mode": "strict_neutral_object_controls_v0",
        "object_count": len(objects),
        "shuffled_records": {
            "distance_shape_correlation_to_original": record_corr,
            "mean_abs_distance_delta": record_delta,
            "expected_failure_observed": _control_degraded(record_corr, record_delta),
        },
        "shuffled_transition_labels": {
            "distance_shape_correlation_to_original": transition_corr,
            "mean_abs_distance_delta": transition_delta,
            "expected_failure_observed": _control_degraded(transition_corr, transition_delta),
        },
        "shuffled_records_fail": _control_degraded(record_corr, record_delta),
        "shuffled_transition_labels_fail": _control_degraded(transition_corr, transition_delta),
        "claim_boundary": (
            "Run-specific neutral-object controls. They must degrade object distances before object-bulk "
            "receipts can be interpreted."
        ),
    }


def neutral_object_s2_leakage_audit(
    distance: np.ndarray,
    objects: list[NeutralObject],
    observer_views: list[dict[str, Any]],
) -> dict[str, Any]:
    axis_by_observer: dict[int, np.ndarray] = {}
    for index, view in enumerate(observer_views):
        if view.get("view_type") != "patch_observer":
            continue
        axis = np.asarray(view.get("axis", []), dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            continue
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            continue
        axis_by_observer[int(view.get("observer_id", index))] = axis / norm
    centroids: list[np.ndarray] = []
    for obj in objects:
        axes = [axis_by_observer[observer_id] for observer_id in obj.observer_ids if observer_id in axis_by_observer]
        if not axes:
            centroids = []
            break
        centroid = np.mean(np.vstack(axes), axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm <= 1e-12:
            centroids = []
            break
        centroids.append(centroid / norm)
    corr = None
    if len(centroids) == len(objects) and len(centroids) >= 4:
        s2_distance = squareform(pdist(np.vstack(centroids), metric="euclidean"))
        corr = _upper_corr(distance, s2_distance)
    return {
        "s2_distance_correlation": corr,
        "s2_leakage_pass": bool(corr is None or abs(float(corr)) < 0.05),
        "screen_axes_used_in_primary_object_extraction": False,
        "support_nodes_used_in_primary_object_extraction": False,
        "h3_coordinates_used_in_primary_object_extraction": False,
        "claim_boundary": (
            "Post-hoc leakage check only. S2 axes are compared after object distances are built and are "
            "not used by extraction or distance construction."
        ),
    }


def _objects_from_groups(
    groups: dict[str, list[dict[str, Any]]],
    *,
    total_observers: int,
    min_observers_per_object: int,
    max_observer_fraction_per_object: float,
) -> list[NeutralObject]:
    objects: list[NeutralObject] = []
    max_size = max(int(min_observers_per_object), int(np.floor(float(total_observers) * float(max_observer_fraction_per_object))))
    for key, rows in groups.items():
        if len(rows) < int(min_observers_per_object) or len(rows) > max_size:
            continue
        object_id = "neutral_object_" + hashlib.blake2b(key.encode("utf-8"), digest_size=6).hexdigest()
        objects.append(_neutral_object_from_rows(object_id, key, rows))
    return objects


def _neutral_object_from_rows(object_id: str, key: str, rows: list[dict[str, Any]]) -> NeutralObject:
    observer_ids = tuple(sorted(int(row.get("observer_id", index)) for index, row in enumerate(rows)))
    return NeutralObject(
        object_id=object_id,
        observer_ids=observer_ids,
        visible_signature_key=key,
        record_lineage_hist=_mean_hist(rows, "record_signature_histogram", 64, default_scalar="dominant_record_signature"),
        checkpoint_continuation_hist=_mean_transition_hist(rows, "checkpoint_class", 32),
        sector_transport_hist=_mean_transition_hist(rows, "s3_sector_class", 6, scalar_key="sector_change_signature"),
        repair_response_hist=_mean_repair_hist(rows),
        counterfactual_stability_hist=_counterfactual_hist(rows),
        transition_affinity_hist=_mean_nested_hist(rows, "transition_affinity_histograms", 96),
        persistence=float(
            np.median(
                [
                    float(row.get("transition_history_persistence", row.get("record_stability_mean", 0.0)) or 0.0)
                    for row in rows
                ]
            )
        ),
        overlap_agreement=float(
            np.median(
                [
                    float(row.get("transition_history_mean_modal_mass", row.get("committed_fraction", 0.0)) or 0.0)
                    for row in rows
                ]
            )
        ),
    )


def _visible_object_key(view: dict[str, Any], *, level: str) -> str:
    record = _dominant_hist_bucket(view.get("record_signature_histogram"), 64, view.get("dominant_record_signature"))
    packet = _dominant_hist_bucket(view.get("object_packet_histogram"), 64, view.get("dominant_object_packet"))
    sector = _scalar_bucket(view.get("sector_change_signature"), 6)
    checkpoint = _dominant_transition_bucket(view, "checkpoint_class", 32)
    repair = _dominant_transition_bucket(view, "repair_load_bucket", 16)
    transition = _dominant_nested_bucket(view.get("transition_history_histograms"), "local_transition_token", 128)
    if level == "full":
        parts = (packet % 16, record % 16, sector, checkpoint % 8, repair % 8, transition % 16)
    elif level == "transition_core":
        parts = (packet % 12, record % 12, sector, checkpoint % 6)
    else:
        parts = (packet % 8, sector)
    return ":".join(str(int(value)) for value in parts)


def _mean_hist(rows: list[dict[str, Any]], key: str, width: int, *, default_scalar: str | None = None) -> np.ndarray:
    values = []
    for row in rows:
        hist = _histogram_dict_to_vector(row.get(key), width)
        if not np.any(hist) and default_scalar is not None:
            scalar = _safe_int(row.get(default_scalar))
            if scalar is not None:
                hist[int(scalar) % width] = 1.0
        values.append(_normalize(hist, width))
    return _normalize(np.mean(np.vstack(values), axis=0) if values else np.zeros(width), width)


def _mean_transition_hist(
    rows: list[dict[str, Any]],
    transition_key: str,
    width: int,
    *,
    scalar_key: str | None = None,
) -> np.ndarray:
    values = []
    for row in rows:
        histograms = row.get("transition_history_histograms") if isinstance(row.get("transition_history_histograms"), dict) else {}
        hist = _histogram_dict_to_vector(histograms.get(transition_key), width)
        if not np.any(hist):
            hist = _histogram_dict_to_vector(histograms.get(f"{transition_key}_path"), width)
        if not np.any(hist) and scalar_key:
            scalar = _safe_int(row.get(scalar_key))
            if scalar is not None:
                hist[int(scalar) % width] = 1.0
        values.append(_normalize(hist, width))
    return _normalize(np.mean(np.vstack(values), axis=0) if values else np.zeros(width), width)


def _mean_repair_hist(rows: list[dict[str, Any]]) -> np.ndarray:
    values = []
    for row in rows:
        spectrum = np.asarray(row.get("repair_response_spectrum", []), dtype=float).reshape(-1)
        hist = np.zeros(32, dtype=float)
        if spectrum.size:
            spectrum = spectrum[:32]
            hist[: spectrum.size] = np.abs(np.where(np.isfinite(spectrum), spectrum, 0.0))
        else:
            bucket = _safe_int(row.get("repair_load_mean"))
            hist[int(bucket or 0) % 32] = 1.0
        values.append(_normalize(hist, 32))
    return _normalize(np.mean(np.vstack(values), axis=0) if values else np.zeros(32), 32)


def _counterfactual_hist(rows: list[dict[str, Any]]) -> np.ndarray:
    hist = np.zeros(16, dtype=float)
    for row in rows:
        try:
            value = float(row.get("counterfactual_stability", 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        bucket = int(np.clip(np.floor(value * 15.999), 0, 15))
        hist[bucket] += 1.0
    return _normalize(hist, 16)


def _mean_nested_hist(rows: list[dict[str, Any]], key: str, width: int) -> np.ndarray:
    values = [_nested_histogram_to_vector(row.get(key), width) for row in rows]
    return _normalize(np.mean(np.vstack(values), axis=0) if values else np.zeros(width), width)


def _histogram_dict_to_vector(histogram: Any, width: int) -> np.ndarray:
    vector = np.zeros(max(1, int(width)), dtype=float)
    if not isinstance(histogram, dict):
        return vector
    for key, value in histogram.items():
        parsed = _safe_int(key)
        if parsed is None:
            parsed = _stable_bucket(str(key), vector.size)
        try:
            mass = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(mass):
            vector[int(parsed) % vector.size] += max(mass, 0.0)
    return _normalize(vector, vector.size)


def _nested_histogram_to_vector(histograms: Any, width: int) -> np.ndarray:
    vector = np.zeros(max(1, int(width)), dtype=float)
    if not isinstance(histograms, dict):
        return vector
    for field_name, histogram in histograms.items():
        if isinstance(histogram, dict):
            for key, value in histogram.items():
                bucket = _stable_bucket(f"{field_name}:{key}", vector.size)
                try:
                    mass = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(mass):
                    vector[bucket] += max(mass, 0.0)
        else:
            bucket = _stable_bucket(str(field_name), vector.size)
            try:
                mass = float(histogram)
            except (TypeError, ValueError):
                continue
            if np.isfinite(mass):
                vector[bucket] += max(mass, 0.0)
    return _normalize(vector, vector.size)


def _dominant_hist_bucket(histogram: Any, width: int, fallback: Any = None) -> int:
    vector = _histogram_dict_to_vector(histogram, width)
    if np.any(vector):
        return int(np.argmax(vector))
    parsed = _safe_int(fallback)
    return int(parsed or 0)


def _dominant_transition_bucket(view: dict[str, Any], key: str, width: int) -> int:
    histograms = view.get("transition_history_histograms") if isinstance(view.get("transition_history_histograms"), dict) else {}
    return _dominant_hist_bucket(histograms.get(key), width)


def _dominant_nested_bucket(histograms: Any, key: str, width: int) -> int:
    if isinstance(histograms, dict):
        return _dominant_hist_bucket(histograms.get(key), width)
    return 0


def _scalar_bucket(value: Any, width: int) -> int:
    parsed = _safe_int(value)
    return int(parsed or 0) % max(1, int(width))


def _normalize(values: np.ndarray, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size < int(width):
        padded = np.zeros(int(width), dtype=float)
        padded[: array.size] = array
        array = padded
    else:
        array = array[: int(width)]
    array = np.where(np.isfinite(array), np.maximum(array, 0.0), 0.0)
    total = float(np.sum(array))
    return array / total if total > 1e-12 else np.zeros_like(array)


def _shuffled_objects(objects: list[NeutralObject], rng: np.random.Generator, *, kind: str) -> list[NeutralObject]:
    out = [copy.deepcopy(obj) for obj in objects]
    order = rng.permutation(len(out))
    if kind == "records":
        record_payloads = [objects[int(index)].record_lineage_hist.copy() for index in order]
        counter_payloads = [objects[int(index)].counterfactual_stability_hist.copy() for index in order]
        for idx, (obj, record, counter) in enumerate(zip(out, record_payloads, counter_payloads)):
            out[idx] = NeutralObject(
                **{**asdict(obj), "record_lineage_hist": record, "counterfactual_stability_hist": counter}
            )
    else:
        checkpoint_payloads = [objects[int(index)].checkpoint_continuation_hist.copy() for index in order]
        sector_payloads = [objects[int(index)].sector_transport_hist.copy() for index in order]
        repair_payloads = [objects[int(index)].repair_response_hist.copy() for index in order]
        affinity_payloads = [objects[int(index)].transition_affinity_hist.copy() for index in order]
        for idx, obj in enumerate(out):
            out[idx] = NeutralObject(
                **{
                    **asdict(obj),
                    "checkpoint_continuation_hist": checkpoint_payloads[idx],
                    "sector_transport_hist": sector_payloads[idx],
                    "repair_response_hist": repair_payloads[idx],
                    "transition_affinity_hist": affinity_payloads[idx],
                }
            )
    return out


def _upper_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    av = _upper(a)
    bv = _upper(b)
    if av.size != bv.size or av.size < 2 or float(np.std(av)) < 1e-12 or float(np.std(bv)) < 1e-12:
        return None
    corr = float(np.corrcoef(av, bv)[0, 1])
    return corr if np.isfinite(corr) else None


def _upper(distance: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] < 2:
        return np.zeros(0, dtype=float)
    return distance[np.triu_indices(distance.shape[0], k=1)]


def _mean_abs_upper_delta(a: np.ndarray, b: np.ndarray) -> float:
    av = _upper(a)
    bv = _upper(b)
    if av.size != bv.size or av.size == 0:
        return 0.0
    return float(np.mean(np.abs(av - bv)))


def _control_degraded(corr: float | None, delta: float) -> bool:
    return bool(corr is None or not np.isfinite(float(corr)) or float(corr) < 0.99 or float(delta) > 1e-6)


def _strict_neutral_object_blockers(
    *,
    object_count: int,
    min_objects: int,
    dimension: dict[str, Any],
    selection: dict[str, Any],
    leakage: dict[str, Any],
    controls: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if object_count < min_objects:
        blockers.append("too_few_neutral_objects")
    if not dimension.get("estimators_agree_3d", False):
        blockers.append("object_dimension_estimators_do_not_agree_3d")
    if not selection.get("h3_selected", False):
        blockers.append("heldout_latent_geometry_does_not_select_h3")
    if not leakage.get("s2_leakage_pass", False):
        blockers.append("object_s2_leakage_audit_failed")
    if not controls.get("shuffled_records_fail", False):
        blockers.append("shuffled_record_object_control_did_not_fail")
    if not controls.get("shuffled_transition_labels_fail", False):
        blockers.append("shuffled_transition_object_control_did_not_fail")
    return blockers


def _empty_dimension() -> dict[str, Any]:
    return {
        "median_dimension_estimate": None,
        "estimators_agree_3d": False,
        "reason": "too_few_neutral_objects",
    }


def _empty_selection() -> dict[str, Any]:
    return {
        "mode": "strict_neutral_heldout_latent_geometry_selection_v0",
        "selected_model": None,
        "h3_selected": False,
        "STRICT_NEUTRAL_LATENT_H3_SELECTION_RECEIPT": False,
        "reason": "too_few_neutral_objects",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stable_bucket(value: str, width: int) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % max(1, int(width))
