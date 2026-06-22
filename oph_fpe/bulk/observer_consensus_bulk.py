from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def observer_consensus_bulk_readout_report(
    run_dirs: list[Path],
    *,
    observer_sample_count: int = 12,
    object_sample_count: int = 24,
) -> dict[str, Any]:
    """Assemble an observer-facing consensus-bulk readout from run receipts.

    This is intentionally a readout layer. It does not change the strict neutral
    bulk gate; it makes explicit the OPH-specific object of study: instantiated
    observer-like systems reading local records and agreeing on shared objects.
    """

    roots = [Path(path) for path in run_dirs]
    claims = _first_json(roots, "claims.json")
    proof = _first_json(roots, "bulk_proof_certificate_report.json")
    finite_contract = _first_json(roots, "finite_oph_theorem_contract_report.json")
    observer_experience = _first_json(roots, "observer_modular_experience_report.json")
    frontier = _first_json(roots, "strict_neutral_bulk_frontier_report.json")
    cmb_output = _first_json(roots, "physical_cmb_output_comparison_report.json")
    object_viewer = _first_json(roots, "object_h3_bulk_viewer_summary.json")
    object_population = _first_json(roots, "observer_chart_object_h3_report.json")
    if not object_population:
        object_population = _first_json(roots, "observer_chart_object_h3_lineage_report.json")

    observer_path = _first_path(roots, "observer_views.jsonl")
    h3_object_path = _first_path(roots, "h3_objects.csv")
    neutral_object_path = _first_path(roots, "neutral_objects.jsonl")

    observer_readout = _observer_view_summary(observer_path, observer_sample_count)
    object_readout = _h3_object_summary(h3_object_path, object_sample_count, object_population)
    neutral_readout = _neutral_object_summary(neutral_object_path, object_sample_count)

    observer_modular_time = bool(
        observer_experience.get("observer_modular_time_receipt", False)
        or proof.get("observer_modular_time_receipt", False)
        or claims.get("observer_modular_time_receipt", False)
    )
    observer_3p1d_experience = bool(
        observer_experience.get("OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT", False)
        or observer_experience.get("observer_facing_3p1d_h3_experience_receipt", False)
        or proof.get("observer_facing_3p1d_h3_experience_receipt", False)
        or claims.get("observer_facing_3p1d_h3_experience_receipt", False)
    )
    theorem_assisted_h3 = bool(
        proof.get("THEOREM_ASSISTED_H3_OBJECT_POPULATION_RECEIPT", False)
        or proof.get("theorem_assisted_h3_nonboundary_population_established", False)
        or claims.get("theorem_assisted_h3_bulk", False)
    )
    chart_blind_strict_neutral = bool(
        proof.get("STRICT_NEUTRAL_BULK_RECEIPT", False)
        or proof.get("bulk_3d_established_strict", False)
        or proof.get("bulk_3d_established_chart_blind_strict_neutral", False)
        or proof.get("chart_blind_strict_neutral_quotient_bulk_receipt", False)
        or frontier.get("strict_neutral_bulk", False)
        or frontier.get("strict_neutral_bulk_frontier_ready", False)
        or claims.get("strict_neutral_bulk", False)
    )
    consensus_bulk_readout = bool(
        observer_modular_time
        and observer_3p1d_experience
        and theorem_assisted_h3
        and object_readout["object_count"] > 0
        and object_readout["spatial_dimension"] == 3
    )
    finite_theorem_contract = bool(
        finite_contract.get("finite_lorentz_theorem_contract_receipt", False)
        or proof.get("finite_lorentz_theorem_contract_receipt", False)
    )
    paper_faithful_observer_spacetime = bool(
        finite_contract.get("paper_faithful_observer_spacetime_emergence_receipt", False)
        or proof.get("paper_faithful_observer_spacetime_emergence_receipt", False)
    )
    paper_faithful_consensus_bulk = bool(
        finite_contract.get("paper_faithful_consensus_bulk_emergence_receipt", False)
        or proof.get("paper_faithful_consensus_bulk_emergence_receipt", False)
    )
    physical_cmb_output = bool(
        cmb_output.get("USABLE_PHYSICAL_CMB_DATA_RECEIPT", False)
        or claims.get("physical_cmb_output_usable_data_receipt", False)
    )
    physical_cmb_prediction = bool(
        cmb_output.get("PHYSICAL_CMB_PREDICTION_RECEIPT", False)
        or claims.get("physical_cmb_output_prediction_receipt", False)
        or claims.get("physical_cmb_prediction", False)
    )

    return {
        "mode": "observer_consensus_bulk_readout_v0",
        "run_dirs": [str(path) for path in roots],
        "OBSERVER_LIKE_SELF_READING_SYSTEM_RECEIPT": bool(observer_readout["observer_view_count"] > 0),
        "observer_like_self_reading_system_receipt": bool(observer_readout["observer_view_count"] > 0),
        "observer_modular_time_receipt": observer_modular_time,
        "observer_facing_3p1d_h3_experience_receipt": observer_3p1d_experience,
        "observer_facing_consensus_3d_bulk_readout_receipt": consensus_bulk_readout,
        "theorem_assisted_consensus_3d_bulk_readout_receipt": consensus_bulk_readout,
        "THEOREM_ASSISTED_CONSENSUS_3D_BULK_READOUT_RECEIPT": consensus_bulk_readout,
        "finite_lorentz_theorem_contract_receipt": finite_theorem_contract,
        "paper_faithful_observer_spacetime_emergence_receipt": paper_faithful_observer_spacetime,
        "paper_faithful_consensus_bulk_emergence_receipt": paper_faithful_consensus_bulk,
        "simulation_matches_observer_facing_oph_spacetime_bulk_prediction_receipt": paper_faithful_consensus_bulk,
        "simulation_matches_full_oph_spacetime_bulk_prediction_receipt": paper_faithful_consensus_bulk,
        "chart_blind_strict_neutral_quotient_bulk_receipt": chart_blind_strict_neutral,
        "strict_neutral_third_person_bulk_receipt": chart_blind_strict_neutral,
        "STRICT_NEUTRAL_BULK_RECEIPT": chart_blind_strict_neutral,
        "physical_cmb_output_comparison_receipt": physical_cmb_output,
        "physical_cmb_prediction_receipt": physical_cmb_prediction,
        "bulk_status": (
            "theorem_assisted_observer_facing_consensus_3d_bulk"
            if consensus_bulk_readout and not chart_blind_strict_neutral
            else "observer_facing_consensus_3d_bulk_plus_chart_blind_strict_neutral_quotient"
            if consensus_bulk_readout and chart_blind_strict_neutral
            else "chart_blind_strict_neutral_quotient_bulk_only"
            if chart_blind_strict_neutral
            else "not_established"
        ),
        "observer_readout": observer_readout,
        "h3_object_readout": object_readout,
        "neutral_object_readout": neutral_readout,
        "selected_object_chart_report": proof.get("selected_object_chart_report"),
        "selected_object_chart_incidence_mode": proof.get("selected_object_chart_incidence_mode"),
        "object_viewer_summary": {
            "object_count": object_viewer.get("object_count"),
            "reported_object_count": object_viewer.get("reported_object_count"),
            "observer_overlap_link_count": object_viewer.get("observer_overlap_link_count"),
            "viewer_path": object_viewer.get("viewer_path"),
        },
        "strict_neutral_blockers": list(
            frontier.get("blockers")
            or frontier.get("promotion_blockers")
            or claims.get("strict_neutral_bulk_frontier_blockers")
            or []
        ),
        "finite_theorem_contract_blockers": list(
            finite_contract.get("blockers") or proof.get("finite_theorem_contract_summary", {}).get("blockers") or []
        ),
        "finite_theorem_contract_primary_blockers": list(
            finite_contract.get("primary_blockers")
            or proof.get("finite_theorem_contract_summary", {}).get("primary_blockers")
            or []
        ),
        "claim_boundary": (
            "Instantiation of observer-like self-reading systems: local observer rows read support, "
            "record, modular-time, and visible-object data from run artifacts. A true theorem-assisted "
            "consensus 3D bulk readout means observer-local records share objects in a 3D H3 chart under "
            "the current theorem-assisted route. The stricter paper-faithful observer-spacetime and "
            "observer-facing consensus-bulk receipts additionally require the finite OPH theorem contract. "
            "The chart-blind strict neutral quotient audit is separate: it deliberately discards the S2/H3 "
            "chart prior and can fail without falsifying the observer-facing 3+1D/H3 theorem receipt. "
            "This is not a physical CMB prediction unless physical_cmb_prediction_receipt passes."
        ),
    }


def write_observer_consensus_bulk_readout_report(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    observer_sample_count: int = 12,
    object_sample_count: int = 24,
) -> dict[str, Any]:
    report = observer_consensus_bulk_readout_report(
        run_dirs,
        observer_sample_count=observer_sample_count,
        object_sample_count=object_sample_count,
    )
    requested_out = Path(out_dir)
    out = requested_out.parent if requested_out.suffix.lower() == ".json" else requested_out
    out.mkdir(parents=True, exist_ok=True)
    json_path = (
        requested_out
        if requested_out.suffix.lower() == ".json"
        else out / "observer_consensus_bulk_readout_report.json"
    )
    json_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "observer_consensus_bulk_readout_report.md").write_text(
        _markdown_report(report),
        encoding="utf-8",
    )
    _write_observer_rows(out / "observer_perspective_rows.csv", report["observer_readout"]["sample_views"])
    _write_object_rows(out / "consensus_h3_object_rows.csv", report["h3_object_readout"]["sample_objects"])
    return report


def _observer_view_summary(path: Path | None, sample_count: int) -> dict[str, Any]:
    patch_views: list[dict[str, Any]] = []
    cap_views: list[dict[str, Any]] = []
    total = 0
    if path is not None and path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                row = json.loads(line)
                view_type = str(row.get("view_type", ""))
                if view_type == "patch_observer" and len(patch_views) < sample_count:
                    patch_views.append(_sample_observer_view(row))
                elif view_type == "cap_observer" and len(cap_views) < max(1, sample_count // 2):
                    cap_views.append(_sample_cap_view(row))
    sample_views = [*patch_views, *cap_views]
    relative_times = sorted(
        {
            float(value)
            for row in sample_views
            for value in row.get("observer_relative_times", [])
            if _finite(value)
        }
    )
    return {
        "source_path": str(path) if path is not None else None,
        "observer_view_count": total,
        "sample_patch_observer_count": len(patch_views),
        "sample_cap_observer_count": len(cap_views),
        "relative_time_count": len(relative_times),
        "relative_times_sample": relative_times[:12],
        "sample_views": sample_views,
        "claim_boundary": (
            "Observer-local self-readout rows: visible support, modular time, record signature, "
            "and local repair/mismatch summaries. Hidden representatives are not used."
        ),
    }


def _h3_object_summary(
    path: Path | None,
    sample_count: int,
    object_population_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    object_count = 0
    points: list[list[float]] = []
    if path is not None and path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                object_count += 1
                sample = _sample_h3_object(row)
                point = sample.get("h3_spatial_point")
                if isinstance(point, list) and len(point) == 3 and all(_finite(value) for value in point):
                    points.append([float(value) for value in point])
                if len(rows) < sample_count:
                    rows.append(sample)
    elif isinstance(object_population_report, dict) and object_population_report:
        object_count = int(object_population_report.get("object_count") or 0)
        for row in object_population_report.get("sample_objects") or []:
            if not isinstance(row, dict):
                continue
            sample = _sample_h3_object_from_report(row)
            point = sample.get("h3_spatial_point")
            if isinstance(point, list) and len(point) == 3 and all(_finite(value) for value in point):
                points.append([float(value) for value in point])
            if len(rows) < sample_count:
                rows.append(sample)
    centroid = _centroid(points)
    return {
        "source_path": str(path) if path is not None else None,
        "source_report": object_population_report.get("mode") if isinstance(object_population_report, dict) else None,
        "object_count": object_count,
        "sample_object_count": len(rows),
        "spatial_dimension": 3 if any(len(point) == 3 for point in points) else 0,
        "centroid": centroid,
        "rms_radius_about_centroid": _rms_radius(points, centroid),
        "observer_chart_object_h3_receipt": bool(
            object_population_report.get("observer_chart_object_h3_receipt", False)
        ) if isinstance(object_population_report, dict) else None,
        "observer_facing_h3_object_population_receipt": bool(
            object_population_report.get("observer_facing_h3_object_population_receipt", False)
        ) if isinstance(object_population_report, dict) else None,
        "object_population_blockers": list(
            object_population_report.get("blockers") or []
        ) if isinstance(object_population_report, dict) else [],
        "sample_objects": rows,
        "claim_boundary": (
            "Observer-facing shared objects placed in the theorem-side H3 spatial chart. "
            "These are consensus-bulk readout objects, not strict neutral third-person objects."
        ),
    }


def _neutral_object_summary(path: Path | None, sample_count: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    count = 0
    observer_counts: list[int] = []
    if path is not None and path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                row = json.loads(line)
                observers = row.get("observer_ids") if isinstance(row.get("observer_ids"), list) else []
                observer_counts.append(len(observers))
                if len(rows) < sample_count:
                    rows.append(
                        {
                            "object_id": row.get("object_id"),
                            "observer_count": len(observers),
                            "visible_signature_key": row.get("visible_signature_key"),
                        }
                    )
    return {
        "source_path": str(path) if path is not None else None,
        "neutral_object_count": count,
        "sample_neutral_object_count": len(rows),
        "median_observers_per_neutral_object": _median(observer_counts),
        "sample_neutral_objects": rows,
        "claim_boundary": (
            "Neutral-object rows are included for audit context. Their existence does not by itself "
            "clear chart-blind strict neutral quotient bulk gates."
        ),
    }


def _sample_observer_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "view_type": row.get("view_type"),
        "observer_id": row.get("observer_id"),
        "axis": row.get("axis"),
        "support_patch_count": row.get("support_patch_count"),
        "support_entropy_capacity": row.get("support_entropy_capacity"),
        "observer_relative_times": row.get("observer_relative_times") or [],
        "dominant_record_signature": row.get("dominant_record_signature"),
        "modular_depth_mean": row.get("modular_depth_mean"),
        "modular_depth_std": row.get("modular_depth_std"),
        "repair_load_mean": row.get("repair_load_mean"),
        "mismatch_density_mean": row.get("mismatch_density_mean"),
        "visible_signature_entropy": row.get("visible_signature_entropy"),
        "visible_readout_hash": str(row.get("visible_readout_hash") or "")[:16],
    }


def _sample_cap_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "view_type": row.get("view_type"),
        "cap_index": row.get("cap_index"),
        "axis": row.get("axis"),
        "theta0": row.get("theta0"),
        "collar_width": row.get("collar_width"),
        "observer_relative_times": row.get("observer_relative_times") or [],
        "cap_area_planck": row.get("cap_area_planck"),
        "cap_entropy_capacity": row.get("cap_entropy_capacity"),
        "repair_load_mean": row.get("repair_load_mean"),
        "mismatch_density_mean": row.get("mismatch_density_mean"),
    }


def _sample_h3_object(row: dict[str, str]) -> dict[str, Any]:
    return {
        "object_id": row.get("object_id"),
        "record_family_id": row.get("record_family_id"),
        "family_mode": row.get("family_mode"),
        "observer_count": _int_or_none(row.get("observer_count")),
        "parent_observer_count": _int_or_none(row.get("parent_observer_count")),
        "support_size": _int_or_none(row.get("support_size")),
        "h3_compactness": _float_or_none(row.get("h3_compactness")),
        "h3_compactness_normalized": _float_or_none(row.get("h3_compactness_normalized")),
        "h3_spatial_point": _json_list(row.get("h3_spatial_point")),
        "mean_observer_key_weight": _float_or_none(row.get("mean_observer_key_weight")),
    }


def _sample_h3_object_from_report(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_id": row.get("object_id"),
        "record_family_id": row.get("record_family_id"),
        "family_mode": row.get("family_mode"),
        "observer_count": _int_or_none(row.get("observer_count")),
        "parent_observer_count": _int_or_none(row.get("parent_observer_count")),
        "support_size": _int_or_none(row.get("support_size")),
        "h3_compactness": _float_or_none(row.get("h3_compactness")),
        "h3_compactness_normalized": _float_or_none(row.get("h3_compactness_normalized")),
        "h3_spatial_point": _coerce_float_list(row.get("h3_spatial_point")),
        "mean_observer_key_weight": _float_or_none(row.get("mean_observer_key_weight")),
    }


def _first_json(roots: list[Path], name: str) -> dict[str, Any]:
    path = _first_path(roots, name)
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _first_path(roots: list[Path], name: str) -> Path | None:
    for root in roots:
        root = Path(root)
        if root.is_file() and root.name == name:
            return root
        candidate = root / name
        if candidate.exists():
            return candidate
        if root.exists() and root.is_dir():
            matches = sorted(root.glob(f"**/{name}"))
            if matches:
                return matches[0]
    return None


def _json_list(value: str | None) -> list[float] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    out: list[float] = []
    for item in parsed:
        number = _float_or_none(item)
        if number is None:
            return None
        out.append(number)
    return out


def _coerce_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _json_list(value)
    if not isinstance(value, list):
        return None
    out: list[float] = []
    for item in value:
        number = _float_or_none(item)
        if number is None:
            return None
        out.append(number)
    return out


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _centroid(points: list[list[float]]) -> list[float] | None:
    if not points:
        return None
    dim = len(points[0])
    return [sum(point[index] for point in points) / len(points) for index in range(dim)]


def _rms_radius(points: list[list[float]], centroid: list[float] | None) -> float | None:
    if not points or centroid is None:
        return None
    squares = [
        sum((point[index] - centroid[index]) ** 2 for index in range(len(centroid)))
        for point in points
    ]
    return math.sqrt(sum(squares) / len(squares))


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2)


def _write_observer_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "view_type",
        "observer_id",
        "cap_index",
        "support_patch_count",
        "support_entropy_capacity",
        "relative_time_count",
        "dominant_record_signature",
        "modular_depth_mean",
        "repair_load_mean",
        "mismatch_density_mean",
        "visible_signature_entropy",
        "visible_readout_hash",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "view_type": row.get("view_type"),
                    "observer_id": row.get("observer_id"),
                    "cap_index": row.get("cap_index"),
                    "support_patch_count": row.get("support_patch_count"),
                    "support_entropy_capacity": row.get("support_entropy_capacity"),
                    "relative_time_count": len(row.get("observer_relative_times") or []),
                    "dominant_record_signature": row.get("dominant_record_signature"),
                    "modular_depth_mean": row.get("modular_depth_mean"),
                    "repair_load_mean": row.get("repair_load_mean"),
                    "mismatch_density_mean": row.get("mismatch_density_mean"),
                    "visible_signature_entropy": row.get("visible_signature_entropy"),
                    "visible_readout_hash": row.get("visible_readout_hash"),
                }
            )


def _write_object_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "object_id",
        "record_family_id",
        "observer_count",
        "support_size",
        "h3_compactness",
        "h3_compactness_normalized",
        "h3_x",
        "h3_y",
        "h3_z",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            point = row.get("h3_spatial_point") or []
            writer.writerow(
                {
                    "object_id": row.get("object_id"),
                    "record_family_id": row.get("record_family_id"),
                    "observer_count": row.get("observer_count"),
                    "support_size": row.get("support_size"),
                    "h3_compactness": row.get("h3_compactness"),
                    "h3_compactness_normalized": row.get("h3_compactness_normalized"),
                    "h3_x": point[0] if len(point) > 0 else None,
                    "h3_y": point[1] if len(point) > 1 else None,
                    "h3_z": point[2] if len(point) > 2 else None,
                }
            )


def _markdown_report(report: dict[str, Any]) -> str:
    observer = report["observer_readout"]
    objects = report["h3_object_readout"]
    neutral = report["neutral_object_readout"]
    blockers = report.get("strict_neutral_blockers") or []
    blocker_text = "\n".join(f"- {item}" for item in blockers[:12]) or "- none"
    contract_blockers = report.get("finite_theorem_contract_primary_blockers") or []
    contract_blocker_text = "\n".join(f"- {item}" for item in contract_blockers[:12]) or "- none"
    return (
        "# Observer Consensus Bulk Readout\n\n"
        f"- observer-like self-reading receipt: {report['observer_like_self_reading_system_receipt']}\n"
        f"- observer modular time receipt: {report['observer_modular_time_receipt']}\n"
        f"- observer-facing 3+1D/H3 experience receipt: {report['observer_facing_3p1d_h3_experience_receipt']}\n"
        f"- theorem-assisted consensus 3D bulk readout: "
        f"{report['theorem_assisted_consensus_3d_bulk_readout_receipt']}\n"
        f"- finite theorem contract receipt: {report['finite_lorentz_theorem_contract_receipt']}\n"
        f"- paper-faithful observer spacetime emergence: "
        f"{report['paper_faithful_observer_spacetime_emergence_receipt']}\n"
        f"- paper-faithful consensus bulk emergence: "
        f"{report['paper_faithful_consensus_bulk_emergence_receipt']}\n"
        "- chart-blind strict neutral quotient bulk receipt: "
        f"{report['chart_blind_strict_neutral_quotient_bulk_receipt']}\n"
        f"- physical CMB output comparison receipt: {report['physical_cmb_output_comparison_receipt']}\n"
        f"- physical CMB prediction receipt: {report['physical_cmb_prediction_receipt']}\n"
        f"- bulk status: `{report['bulk_status']}`\n\n"
        "## Observer Readout\n\n"
        f"- observer view rows: {observer['observer_view_count']}\n"
        f"- sampled patch observers: {observer['sample_patch_observer_count']}\n"
        f"- sampled cap observers: {observer['sample_cap_observer_count']}\n"
        f"- relative time samples: {observer['relative_times_sample']}\n\n"
        "## Consensus Objects\n\n"
        f"- H3 object rows: {objects['object_count']}\n"
        f"- spatial dimension: {objects['spatial_dimension']}\n"
        f"- RMS radius about centroid: {objects['rms_radius_about_centroid']}\n"
        f"- neutral object rows: {neutral['neutral_object_count']}\n"
        f"- median observers per neutral object: {neutral['median_observers_per_neutral_object']}\n\n"
        "## Finite Theorem Contract Blockers\n\n"
        f"{contract_blocker_text}\n\n"
        "## Strict-Neutral Blockers\n\n"
        f"{blocker_text}\n\n"
        "## Claim Boundary\n\n"
        f"{report['claim_boundary']}\n"
    )
