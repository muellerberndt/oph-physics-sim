from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oph_fpe.bulk.h3_worldline_stitch import (
    H3_PRIMITIVE_SCHEMA,
    H3_PRIMITIVE_SOURCE_KIND,
)


H3_STITCH_PRODUCER_MODE = "oph_h3_worldline_stitch_producer_v0"
MISSING_INPUTS_REPORT_NAME = "h3_worldline_stitch_producer_missing_inputs_report.json"
PRIMITIVES_ARTIFACT_NAME = "oph_h3_worldline_stitch_primitives_v1.json"

# The measurement lane that would turn complete cross-shard inputs into
# hash-bound primitives does not exist yet.  The producer therefore refuses to
# emit the primitives artifact under every input condition; the refusal reason
# is machine-visible in the report below.
MEASUREMENT_LANE_BLOCKER = "cross_shard_primitive_measurement_lane_not_implemented"

_WORLDLINE_REPORT_NAME = "defect_h3_worldlines_report.json"
_GAUGE_STATE_NAME = "s3_gauge_state.npz"
_SEAM_CROSSING_NAME = "seam_interface_crossings.jsonl"
_MANIFEST_NAME = "distributed_universe_manifest.json"


def h3_worldline_stitch_producer_report(
    left_run_dir: Path,
    right_run_dir: Path,
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Inventory the measured inputs required to produce stitch primitives.

    The ``oph_h3_worldline_stitch_primitives_v1`` verifier consumes measured
    cross-shard interface primitives.  This producer checks, input by input,
    whether the supplied run directories carry the data those primitives must
    be computed from.  It emits the primitives artifact only when every input
    is measured; with the current simulator output surface that condition is
    never met, so the producer emits nothing except this report.  No input is
    ever synthesized.
    """

    left = Path(left_run_dir)
    right = Path(right_run_dir)
    inputs: dict[str, dict[str, Any]] = {}

    inputs["distinct_run_dirs"] = _input_row(
        present=left.is_dir() and right.is_dir() and left.resolve() != right.resolve(),
        detail=(
            "two existing, distinct shard run directories are required for a "
            "cross-shard interface"
        ),
        path=f"{left} | {right}",
    )

    for label, run_dir in (("left", left), ("right", right)):
        catalog = _read_json(run_dir / _WORLDLINE_REPORT_NAME)
        worldlines = catalog.get("worldlines") if isinstance(catalog, dict) else None
        catalog_present = bool(isinstance(worldlines, list) and worldlines)
        inputs[f"{label}_worldline_catalog"] = _input_row(
            present=catalog_present,
            detail="per-shard H3 worldline catalog with fitted worldline events",
            path=str(run_dir / _WORLDLINE_REPORT_NAME),
        )
        inputs[f"{label}_worldline_event_support_node_ids"] = _input_row(
            present=catalog_present and _events_carry_support_node_ids(worldlines),
            detail=(
                "worldline events must carry per-event support node ids; the "
                "current emitter records support_node_count only, so interface "
                "contact and overlap-descent margins cannot be measured"
            ),
            path=str(run_dir / _WORLDLINE_REPORT_NAME),
        )
        inputs[f"{label}_gauge_state"] = _input_row(
            present=(run_dir / _GAUGE_STATE_NAME).is_file(),
            detail="per-shard S3 gauge state for sector/gauge transport continuity",
            path=str(run_dir / _GAUGE_STATE_NAME),
        )
        inputs[f"{label}_seam_interface_crossings"] = _input_row(
            present=(run_dir / _SEAM_CROSSING_NAME).is_file(),
            detail=(
                "measured seam interface contact events with signed-distance and "
                "normal-velocity margins; no runtime component emits these"
            ),
            path=str(run_dir / _SEAM_CROSSING_NAME),
        )

    manifest, manifest_source = _load_manifest(manifest_path, left, right)
    shard_rows = manifest.get("shards") if isinstance(manifest, dict) else None
    manifest_present = bool(isinstance(shard_rows, list) and len(shard_rows) >= 2)
    inputs["cross_shard_manifest"] = _input_row(
        present=manifest_present,
        detail=(
            "distributed universe manifest declaring at least two shards; a "
            "monolithic run has no cross-shard interface"
        ),
        path=manifest_source,
    )
    inputs["seam_adjacency_for_run_pair"] = _input_row(
        present=manifest_present
        and _runs_are_seam_neighbors(shard_rows or [], left.name, right.name),
        detail=(
            "the two run ids must be declared seam neighbors of each other in "
            "the distributed manifest"
        ),
        path=manifest_source,
    )
    inputs["coarse_fine_refinement_pair"] = _input_row(
        present=False,
        detail=(
            "the verifier's refinement section needs a coarse/fine pair of the "
            "same seam with a declared Q_sr projection; no run pair with that "
            "structure exists"
        ),
        path=None,
    )

    missing = sorted(name for name, row in inputs.items() if not row["present"])
    blockers = list(missing)
    if not missing:
        blockers.append(MEASUREMENT_LANE_BLOCKER)

    return {
        "mode": H3_STITCH_PRODUCER_MODE,
        "target_schema": H3_PRIMITIVE_SCHEMA,
        "target_source_kind": H3_PRIMITIVE_SOURCE_KIND,
        "target_artifact_name": PRIMITIVES_ARTIFACT_NAME,
        "left_run_dir": str(left),
        "right_run_dir": str(right),
        "manifest_path": manifest_source,
        "artifact_emitted": False,
        "producer_fail_closed": True,
        "inputs": inputs,
        "missing_inputs": missing,
        "blockers": blockers,
        "claim_boundary": (
            "Fail-closed producer lane for oph_h3_worldline_stitch_primitives_v1. "
            "It emits no primitives artifact unless every required cross-shard "
            "input is measured, and the measurement lane itself is not "
            "implemented; the report makes the exact input gaps machine-visible. "
            "Nothing here is a synthetic stand-in for measured interface "
            "evidence."
        ),
    }


def write_h3_worldline_stitch_producer_report(
    left_run_dir: Path,
    right_run_dir: Path,
    out_dir: Path,
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Write the missing-inputs report; never write the primitives artifact."""

    report = h3_worldline_stitch_producer_report(
        left_run_dir,
        right_run_dir,
        manifest_path=manifest_path,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / MISSING_INPUTS_REPORT_NAME).write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    return report


def _input_row(*, present: bool, detail: str, path: str | None) -> dict[str, Any]:
    return {"present": bool(present), "detail": detail, "path": path}


def _events_carry_support_node_ids(worldlines: list[Any]) -> bool:
    for worldline in worldlines:
        if not isinstance(worldline, dict):
            return False
        events = worldline.get("events")
        if not isinstance(events, list) or not events:
            return False
        for event in events:
            if not isinstance(event, dict):
                return False
            nodes = event.get("support_nodes", event.get("support_node_ids"))
            if not isinstance(nodes, list) or not nodes:
                return False
    return True


def _load_manifest(
    manifest_path: Path | None,
    left: Path,
    right: Path,
) -> tuple[dict[str, Any], str | None]:
    candidates: list[Path] = []
    if manifest_path is not None:
        candidates.append(Path(manifest_path))
    for run_dir in (left, right):
        candidates.append(run_dir.parent / _MANIFEST_NAME)
        candidates.append(run_dir.parent.parent / _MANIFEST_NAME)
    for candidate in candidates:
        payload = _read_json(candidate)
        if isinstance(payload, dict) and payload:
            return payload, str(candidate)
    return {}, str(candidates[0]) if candidates else None


def _runs_are_seam_neighbors(
    shard_rows: list[Any],
    left_id: str,
    right_id: str,
) -> bool:
    by_id: dict[str, dict[str, Any]] = {}
    for row in shard_rows:
        if isinstance(row, dict):
            shard_id = str(row.get("run_id") or row.get("shard_id") or "")
            if shard_id:
                by_id[shard_id] = row
    left_row = by_id.get(left_id)
    right_row = by_id.get(right_id)
    if left_row is None or right_row is None or left_id == right_id:
        return False
    left_index = left_row.get("shard_index")
    right_index = right_row.get("shard_index")
    left_neighbors = left_row.get("seam_neighbor_indices")
    right_neighbors = right_row.get("seam_neighbor_indices")
    if not isinstance(left_neighbors, list) or not isinstance(right_neighbors, list):
        return False
    return right_index in left_neighbors and left_index in right_neighbors


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
