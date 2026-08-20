"""Build compact display-only OPH visualizer handoff directories.

The builder deliberately exports selected display arrays and rows, never a
whole run directory. Every package carries source hashes, a claim boundary,
and renderer instructions, and must remain below 200 MB.
"""

from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from oph_fpe.core.icosahedral import (
    build_geodesic_icosahedral_tower,
    geodesic_icosahedral_patch_arrays,
)
from oph_fpe.defects import z6_defect_census as defect_census
from oph_fpe.defects import z6_family_readout as family_readout
from oph_fpe.defects.z6_a5_action import rotation_group
from oph_fpe.defects.z6_carrier_defects import base_carrier_spec, face_curvature
from oph_fpe.dimension import geometry as dimension_geometry
from oph_fpe.dimension import operators as dimension_operators
from oph_fpe.dimension import probe as dimension_probe
from oph_fpe.em import base_carrier, green, temporal
from oph_fpe.qm_observer import receipt as qm_receipt


SCHEMA = "oph.visualizer-handoff-suite.v1"
PACKAGE_SCHEMA = "oph.visualizer-handoff.v1"
MAX_PACKAGE_BYTES = 200_000_000
DEFAULT_RUN = Path("runs/e6_16k_dense_axiom_20260805")
DEFAULT_OUTPUT = Path("visualizer_handoffs/oph-headlines-2026-08-20")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {
            "numerator": value.numerator,
            "denominator": value.denominator,
            "display": float(value),
        }
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": os.path.relpath(path.resolve(), root.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.index("{") :])


def _read_csv(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows if limit is None else rows[:limit]


def _instructions(
    title: str,
    files: list[str],
    rendering: list[str],
    interactions: list[str],
    boundary: str,
) -> str:
    data_lines = "\n".join(f"- `{name}`" for name in files)
    render_lines = "\n".join(f"{index}. {line}" for index, line in enumerate(rendering, 1))
    interaction_lines = "\n".join(f"- {line}" for line in interactions)
    return f"""# {title}: display instructions

This directory contains display data only. It is not a complete simulator run.

## Data

{data_lines}

## Display

{render_lines}

## Interaction

{interaction_lines}

## Local serving

Serve the suite root with `python3 -m http.server 8000` and load the files with
relative URLs. JSON is directly browser-readable. Convert NPZ arrays to typed
arrays in a small preprocessing step or load them with an NPZ-capable browser
decoder; never reinterpret integer labels as floating physical coordinates.

## Claim boundary

{boundary}
"""


def _finish_package(
    package: Path,
    *,
    package_id: str,
    title: str,
    provenance: str,
    claim_boundary: str,
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    files = []
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(package)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    size = sum(row["bytes"] for row in files)
    if size > MAX_PACKAGE_BYTES:
        raise ValueError(f"{package_id} exceeds 200 MB: {size}")
    manifest = {
        "schema": PACKAGE_SCHEMA,
        "package_id": package_id,
        "title": title,
        "display_data_only": True,
        "whole_run_included": False,
        "provenance_status": provenance,
        "claim_boundary": claim_boundary,
        "maximum_package_bytes": MAX_PACKAGE_BYTES,
        "payload_bytes_excluding_manifest": size,
        "sources": source_rows,
        "files": files,
    }
    manifest_path = package / "manifest.json"
    manifest["total_bytes"] = size
    for _ in range(3):
        _write_json(manifest_path, manifest)
        total = size + manifest_path.stat().st_size
        if manifest["total_bytes"] == total:
            break
        manifest["total_bytes"] = total
    _write_json(manifest_path, manifest)
    manifest["manifest_sha256"] = _sha256(package / "manifest.json")
    manifest["total_bytes"] = size + manifest_path.stat().st_size
    if manifest["total_bytes"] > MAX_PACKAGE_BYTES:
        raise ValueError(f"{package_id} exceeds 200 MB after manifest")
    return manifest


def _carrier_packages(output: Path, run: Path, root: Path) -> list[dict[str, Any]]:
    manifests = []
    frame_path = run / "screen_evolution_frames.npz"
    gauge_path = run / "s3_gauge_state.npz"
    source_report_path = run / "source_dynamics_repair_record_observer_report.json"
    with np.load(frame_path) as frames, np.load(gauge_path) as gauge:
        patch_count = int(gauge["points"].shape[0])
        sample_ids = np.unique(np.linspace(0, patch_count - 1, 1024, dtype=np.int64))
        package = output / "s2-carrier-network-interactions"
        data_path = package / "data/carrier_network_frames.npz"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            data_path,
            cycles=frames["cycles"],
            patch_ids=sample_ids,
            points=gauge["points"][sample_ids],
            mismatch=frames["field__local_mismatch_density"][:, sample_ids],
            repair_load=frames["field__cumulative_repair_load"][:, sample_ids],
            modular_depth=frames["field__modular_depth"][:, sample_ids],
            record_signature=frames["field__record_signature"][:, sample_ids],
        )
    _, level_edges_left, level_edges_right = geodesic_icosahedral_patch_arrays(
        2, patch_basis="cells"
    )
    _write_json(
        package / "data/topology.json",
        {
            "schema": "oph.display.s2-carrier-network.v1",
            "full_run_patch_count": patch_count,
            "display_patch_count": len(sample_ids),
            "sampling": "deterministic evenly spaced patch ids",
            "level_2_cell_adjacency": {
                "left": level_edges_left,
                "right": level_edges_right,
            },
            "provenance": "computed run frames, display-downsampled",
        },
    )
    boundary = (
        "Computed finite screen dynamics from the legacy-chart 16k control run. "
        "Downsampling is display-only; no neutral spacetime or physical field is implied."
    )
    (package / "DISPLAY_INSTRUCTIONS.md").write_text(
        _instructions(
            "S2 carrier-network interactions",
            ["data/carrier_network_frames.npz", "data/topology.json"],
            [
                "Render `points` on a unit sphere and color by the selected frame field.",
                "Animate along `cycles`; use the same global color scale for every frame.",
                "Draw only nearest sampled or level-2 adjacency links to avoid a false full-run topology claim.",
            ],
            ["Cycle scrubber", "Field selector", "Patch selection with record signature"],
            boundary,
        ),
        encoding="utf-8",
    )
    manifests.append(
        _finish_package(
            package,
            package_id=package.name,
            title="S2 carrier-network interactions",
            provenance="computed",
            claim_boundary=boundary,
            source_rows=[_source(frame_path, root), _source(gauge_path, root)],
        )
    )

    source_report = _read_json(source_report_path)
    spec = base_carrier_spec()
    initial = defect_census.sample_stream(spec, "uniform_iid", 20260820, 1)[0]
    fixed, trace = defect_census.repair(spec, initial)
    state = list(initial)
    repair_frames = [
        {
            "step": 0,
            "seam_state": list(state),
            "face_curvature": face_curvature(spec, state),
        }
    ]
    for index, (seam, delta) in enumerate(trace, 1):
        state[seam] = (state[seam] + delta) % 6
        repair_frames.append(
            {
                "step": index,
                "changed_seam": seam,
                "delta_mod6": delta,
                "seam_state": list(state),
                "face_curvature": face_curvature(spec, state),
            }
        )
    package = output / "individual-carrier-repair-animation"
    _write_json(
        package / "data/carrier_repair.json",
        {
            "schema": "oph.display.individual-carrier-repair.v1",
            "ports": source_report["local_patch_architecture"]["port_direction_template"],
            "edges": list(zip(base_carrier.SEAM_LEFT, base_carrier.SEAM_RIGHT, strict=True)),
            "faces": base_carrier.ORIENTED_FACES,
            "repair_rule": defect_census.REPAIR_RULE,
            "initial_state": initial,
            "fixed_state": fixed,
            "frames": repair_frames,
            "provenance": "fresh deterministic finite repair replay",
        },
    )
    boundary = (
        "Exact finite Z/6 repair animation on one committed 12/30/20 carrier. "
        "Seam values and face curvature are combinatorial registers, not physical fields."
    )
    (package / "DISPLAY_INSTRUCTIONS.md").write_text(
        _instructions(
            "Individual carrier repair",
            ["data/carrier_repair.json"],
            [
                "Render the twelve ports at `ports`, thirty seams as edges, and twenty triangular faces.",
                "At each frame highlight `changed_seam`; color faces by circular distance of `face_curvature` from zero.",
                "Hold the final frame and mark the repair fixed point without calling it a physical vacuum.",
            ],
            ["Step/play control", "Port/seam/face inspection", "Initial/fixed comparison"],
            boundary,
        ),
        encoding="utf-8",
    )
    manifests.append(
        _finish_package(
            package,
            package_id=package.name,
            title="Individual carrier repair animation",
            provenance="computed",
            claim_boundary=boundary,
            source_rows=[_source(source_report_path, root)],
        )
    )
    return manifests


def _quantum_package(output: Path, root: Path) -> dict[str, Any]:
    package = output / "observer-quantum-born-collapse"
    receipt = qm_receipt.build_receipt()
    export = qm_receipt.build_export()
    _write_json(package / "data/QM_OBSERVER_RECEIPT.v1.json", receipt)
    _write_json(package / "data/QM_OBSERVER_VIZ.v1.json", export)
    boundary = receipt["boundary"]
    (package / "DISPLAY_INSTRUCTIONS.md").write_text(
        _instructions(
            "Observer-frame quantum weights and collapse",
            ["data/QM_OBSERVER_VIZ.v1.json", "data/QM_OBSERVER_RECEIPT.v1.json"],
            [
                "Use one panel per scenario: base contexts, collapse chains, then interference.",
                "Draw exact count ratios as stacked outcome bars; display fraction pairs before decimal formatting.",
                "Animate collapse as conditioning on the observer record: pre-state, selected event, Lueders-updated state, repeat read.",
                "Show direct and mediated interference branches on a shared count scale.",
            ],
            ["Context selector", "Outcome-conditioned branch selector", "Exact fraction/decimal toggle"],
            boundary,
        ),
        encoding="utf-8",
    )
    return _finish_package(
        package,
        package_id=package.name,
        title="Observer-frame quantum weights and collapse",
        provenance="computed",
        claim_boundary=boundary,
        source_rows=[
            _source(root / "oph_fpe/qm_observer/receipt.py", root),
            _source(root / "oph_fpe/quantum/phase_operation.py", root),
        ],
    )


def _depth_package(output: Path, root: Path) -> dict[str, Any]:
    package = output / "refinement-depth-emergence"
    _, graphs, incidences = dimension_geometry.build_tower_window(3)
    configs = [
        ("single_level_3", [3], 0.0, False, False),
        ("decoupled_0_to_3", [0, 1, 2, 3], 0.0, False, False),
        ("area_weighted_kappa_0p25", [0, 1, 2, 3], 0.25, False, False),
        ("area_weighted_kappa_1", [0, 1, 2, 3], 1.0, False, False),
        ("area_weighted_kappa_2", [0, 1, 2, 3], 2.0, False, False),
        ("matched_uniform_kappa_0p25", [0, 1, 2, 3], 0.25, False, True),
        ("matched_uniform_kappa_1", [0, 1, 2, 3], 1.0, False, True),
        ("matched_uniform_kappa_2", [0, 1, 2, 3], 2.0, False, True),
        ("all_ones_scale_control", [0, 1, 2, 3], 1.0, True, False),
    ]
    by_level = {graph.level: graph for graph in graphs}
    rows = []
    for index, (name, levels, kappa, static, uniform) in enumerate(configs):
        selected_graphs = [by_level[level] for level in levels]
        selected_incidence = [
            row
            for row in incidences
            if row.coarse_level in levels and row.fine_level in levels
        ]
        matrix, meta = dimension_operators.union_laplacian(
            selected_graphs,
            selected_incidence,
            kappa,
            static_control=static,
            uniform_fiber_weights=uniform,
        )
        expected = len(levels) if kappa == 0.0 and len(levels) > 1 else 1
        measured = dimension_probe.measure_operator(
            matrix,
            expected_components=expected,
            probe_seed=20260820 + index,
        )
        rows.append({"name": name, "levels": levels, "meta": meta, **measured})
    _write_json(
        package / "data/depth_emergence.json",
        {
            "schema": "oph.display.refinement-depth-emergence.v1",
            "evidential_status": "exploratory_non_evidential",
            "sigma_grid": dimension_probe.est.SIGMA_GRID,
            "levels": [
                {
                    "level": graph.level,
                    "cells": graph.cell_count,
                    "edges": graph.edge_count,
                }
                for graph in graphs
            ],
            "rows": rows,
            "provenance": "fresh bounded deterministic level-3 replay",
        },
    )
    boundary = (
        "Finite spectral statistics of a declared union operator. The carrier levels "
        "are S2 by construction; the observed union-depth trend is not a measurement "
        "of physical spatial dimension or a closed spacetime derivation."
    )
    (package / "DISPLAY_INSTRUCTIONS.md").write_text(
        _instructions(
            "Refinement-depth emergence",
            ["data/depth_emergence.json"],
            [
                "Plot every `d_s_curve` against log10 sigma with the guarded window emphasized.",
                "Use small multiples for single-level, decoupled, area-weighted, matched-scale, and all-ones controls.",
                "Add a second chart for `d_weyl` versus coupling; never draw a target line as an achieved physical dimension.",
            ],
            ["Configuration selection", "Spectral/Weyl view", "Guarded-window overlay"],
            boundary,
        ),
        encoding="utf-8",
    )
    return _finish_package(
        package,
        package_id=package.name,
        title="Refinement-depth emergence",
        provenance="computed",
        claim_boundary=boundary,
        source_rows=[
            _source(root / "oph_fpe/dimension/probe.py", root),
            _source(root / "oph_fpe/dimension/operators.py", root),
        ],
    )


def _repair_package(output: Path, run: Path, root: Path) -> dict[str, Any]:
    package = output / "repair-confluence-and-public-records"
    endpoint_path = root / "data/repair_closure/vertex12_a2_endpoint_commutator_receipt.json"
    bridge_path = root / "data/repair_closure/port_repair_propagation_bridge_receipt.json"
    mismatch_path = run / "mismatch_trace.csv"
    _write_json(
        package / "data/repair_confluence.json",
        {
            "schema": "oph.display.repair-confluence.v1",
            "endpoint": _read_json(endpoint_path),
            "repair_bridge": _read_json(bridge_path),
            "mismatch_trace": _read_csv(mismatch_path),
            "provenance": "exact receipts plus computed run trace",
        },
    )
    boundary = (
        "Finite repair confluence and record traces. The endpoint receipt remains "
        "conditional where source-oriented transport or physical propagation is open."
    )
    (package / "DISPLAY_INSTRUCTIONS.md").write_text(
        _instructions(
            "Repair confluence and public records",
            ["data/repair_confluence.json"],
            [
                "Plot mismatch count and repair activity over cycle.",
                "Render alternate event words as parallel lanes merging only where the exact endpoint receipt permits.",
                "Attach record/readback/checkpoint markers to their actual cycle rows.",
            ],
            ["Cycle scrubber", "Event-word comparison", "Receipt boundary drawer"],
            boundary,
        ),
        encoding="utf-8",
    )
    return _finish_package(
        package,
        package_id=package.name,
        title="Repair confluence and public records",
        provenance="computed",
        claim_boundary=boundary,
        source_rows=[
            _source(endpoint_path, root),
            _source(bridge_path, root),
            _source(mismatch_path, root),
        ],
    )


def _observer_packages(output: Path, run: Path, root: Path) -> list[dict[str, Any]]:
    manifests = []
    perspective_path = run / "observer_perspective_rows.csv"
    object_path = run / "observer_objects.jsonl"
    rows = _read_csv(perspective_path, limit=128)
    objects = []
    with object_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 128:
                break
            objects.append(json.loads(line))
    package = output / "observer-cameras"
    _write_json(
        package / "data/observer_cameras.json",
        {
            "schema": "oph.display.observer-cameras.v1",
            "observers": rows,
            "objects": objects,
            "selection": "first 128 canonical rows",
            "provenance": "computed legacy-chart control run",
        },
    )
    boundary = (
        "Observer-local visible records from a finite legacy-chart control run. "
        "A camera is a self-reading support/readback view, not an external neutral camera."
    )
    (package / "DISPLAY_INSTRUCTIONS.md").write_text(
        _instructions(
            "Observer cameras",
            ["data/observer_cameras.json"],
            [
                "Use one observer selector and render only that row's visible support and object packets.",
                "Keep global cycle solely as synchronization metadata; lead with observer-relative fields.",
                "Show support size, readback record, and visible entropy next to the camera field.",
            ],
            ["Observer selector", "Object selection", "Local/global frame comparison"],
            boundary,
        ),
        encoding="utf-8",
    )
    manifests.append(
        _finish_package(
            package,
            package_id=package.name,
            title="Observer cameras",
            provenance="computed",
            claim_boundary=boundary,
            source_rows=[_source(perspective_path, root), _source(object_path, root)],
        )
    )

    modular_path = run / "modular_response_kernel_cache.json"
    modular = _read_json(modular_path)
    package = output / "observer-modular-time"
    _write_json(
        package / "data/observer_modular_time.json",
        {
            "schema": "oph.display.observer-modular-time.v1",
            "observer_ids": modular.get("observer_ids", [])[:128],
            "feature_rows": modular.get("feature_rows", [])[:128],
            "caps": modular.get("caps", [])[:128],
            "time_count": modular.get("time_count"),
            "field_names": modular.get("field_names"),
            "response_summary": modular.get("response_summary"),
            "wrong_scale_controls": modular.get("wrong_scale_controls"),
            "provenance": "computed, display-selected first 128 observer rows",
        },
    )
    boundary = modular.get("claim_boundary", "Observer-local modular-time diagnostic only.")
    (package / "DISPLAY_INSTRUCTIONS.md").write_text(
        _instructions(
            "Observer modular time",
            ["data/observer_modular_time.json"],
            [
                "Render relative-time samples on the horizontal axis and observer-visible features vertically.",
                "Animate one selected observer; compare its response with wrong-scale controls on the same axes.",
                "Label the coordinate observer-local modular time, not universal external time.",
            ],
            ["Observer selector", "Feature selector", "Wrong-scale control overlay"],
            boundary,
        ),
        encoding="utf-8",
    )
    manifests.append(
        _finish_package(
            package,
            package_id=package.name,
            title="Observer modular time",
            provenance="computed",
            claim_boundary=boundary,
            source_rows=[_source(modular_path, root)],
        )
    )
    return manifests


def _defect_packages(output: Path, run: Path, root: Path) -> list[dict[str, Any]]:
    manifests = []
    census = defect_census.run_census()
    package = output / "defect-emergence"
    _write_json(package / "data/defect_census.json", census)
    boundary = census["statement"]
    (package / "DISPLAY_INSTRUCTIONS.md").write_text(
        _instructions(
            "Defect emergence",
            ["data/defect_census.json"],
            [
                "Start with ensemble members and animate repair into fixed sector classes.",
                "Map multiplicity to mark area and energy to vertical position; preserve exact sector labels in tooltips.",
                "Separate the vacuum row from nonzero classes and retain stream identity.",
            ],
            ["Stream selector", "Energy/multiplicity view", "Class inspection"],
            boundary,
        ),
        encoding="utf-8",
    )
    manifests.append(
        _finish_package(
            package,
            package_id=package.name,
            title="Defect emergence",
            provenance="computed",
            claim_boundary=boundary,
            source_rows=[_source(root / "oph_fpe/defects/z6_defect_census.py", root)],
        )
    )

    spec = base_carrier_spec()
    rotations = rotation_group(spec)
    readout = family_readout.build_readout(spec, rotations)
    weights = readout["weights"]
    grouped_classes = []
    for row in census["classes"]:
        grouped_classes.append(
            {
                "sector": row["sector"],
                "multiplicity": row["multiplicity"],
                "energy": row["energy"],
                "orbit_size": row["orbit_size"],
                "label_qtd_v2": list(
                    family_readout.sector_label_v2(spec, weights, tuple(row["sector"]))
                ),
            }
        )
    events_path = run / "organic_defect_population_report_worldline_events.csv"
    interaction_path = run / "defect_interaction_report.json"
    package = output / "defect-grouping-and-interactions"
    _write_json(
        package / "data/defect_groups.json",
        {
            "schema": "oph.display.defect-groups.v1",
            "readout_receipts": readout["receipts"],
            "classes": grouped_classes,
            "worldline_events": _read_csv(events_path, limit=2500),
            "interaction_summary": _read_json(interaction_path),
            "provenance": "fresh exact grouping plus selected computed worldline rows",
        },
    )
    boundary = (
        "Exact finite A5-orbit and Z/6×Z/3×Z/2 grouping diagnostics plus "
        "computed screen worldlines. Defect classes are not physical particles; "
        "the interaction and particle receipts remain fail-closed where shown."
    )
    (package / "DISPLAY_INSTRUCTIONS.md").write_text(
        _instructions(
            "Defect grouping and interactions",
            ["data/defect_groups.json"],
            [
                "Use a nested orbit → `(q,t,d)` → class layout; never merge the two non-equivalent family channels.",
                "Render worldline events as time-linked points using the supplied H3 display coordinates.",
                "Use interaction summary booleans as visible boundaries, not as hidden filters.",
            ],
            ["Orbit/group selector", "Worldline playhead", "Interaction candidate inspection"],
            boundary,
        ),
        encoding="utf-8",
    )
    manifests.append(
        _finish_package(
            package,
            package_id=package.name,
            title="Defect grouping and interactions",
            provenance="computed",
            claim_boundary=boundary,
            source_rows=[
                _source(root / "oph_fpe/defects/z6_family_readout.py", root),
                _source(events_path, root),
                _source(interaction_path, root),
            ],
        )
    )
    return manifests


def _spacetime_package(output: Path, run: Path, root: Path) -> dict[str, Any]:
    package = output / "observer-spacetime-emergence"
    audit_path = run / "neutral_3d_bulk_audit_report.json"
    h3_path = run / "defect_cluster_h3_report.json"
    agreement_path = run / "observer_consensus_report.json"
    _write_json(
        package / "data/spacetime_emergence.json",
        {
            "schema": "oph.display.observer-spacetime-emergence.v1",
            "neutral_bulk_audit": _read_json(audit_path),
            "h3_defect_cluster": _read_json(h3_path),
            "observer_consensus": _read_json(agreement_path),
            "provenance": "computed legacy-chart control receipts",
        },
    )
    boundary = (
        "Observer-facing H3/chart and agreement diagnostics are shown with the "
        "strict-neutral bulk blockers. H3 plus a modular clock is not by itself "
        "a derived neutral 3+1-dimensional event manifold."
    )
    (package / "DISPLAY_INSTRUCTIONS.md").write_text(
        _instructions(
            "Observer spacetime emergence",
            ["data/spacetime_emergence.json"],
            [
                "Use parallel lanes for observer chart, agreement transport, H3 fit, and neutral-bulk gate.",
                "Place blockers at the transition they prevent; do not hide them behind a final scene.",
                "Render H3 points only when their source rows are present and label them observer-facing coordinates.",
            ],
            ["Lane selection", "Observer/chart inspection", "Blocker visibility toggle that never changes status"],
            boundary,
        ),
        encoding="utf-8",
    )
    return _finish_package(
        package,
        package_id=package.name,
        title="Observer spacetime emergence",
        provenance="computed",
        claim_boundary=boundary,
        source_rows=[
            _source(audit_path, root),
            _source(h3_path, root),
            _source(agreement_path, root),
        ],
    )


def _symmetry_package(output: Path, run: Path, root: Path) -> dict[str, Any]:
    package = output / "a5-symmetry-and-sector-decomposition"
    tower = build_geodesic_icosahedral_tower(0)
    base = tower.levels[0]
    spec = base_carrier_spec()
    rotations = rotation_group(spec)
    _write_json(
        package / "data/a5_symmetry.json",
        {
            "schema": "oph.display.a5-symmetry.v1",
            "vertices": base.vertices,
            "edges": base.edges,
            "faces": base.faces,
            "rotations": rotations,
            "rotation_count": len(rotations),
            "s3_class_counts": _read_json(run / "s3_class_counts.json"),
            "sector_decomposition": [1, 3, 3, 5],
            "provenance": "exact carrier and exact derived rotation group",
        },
    )
    boundary = (
        "Exact finite icosahedral/A5 structure and declared sector decomposition. "
        "It does not select the Standard Model, particle masses, or physical couplings."
    )
    (package / "DISPLAY_INSTRUCTIONS.md").write_text(
        _instructions(
            "A5 symmetry and sector decomposition",
            ["data/a5_symmetry.json"],
            [
                "Render the exact icosahedron and animate each supplied vertex permutation as a rigid action.",
                "Display the `1 + 3 + 3′ + 5` sectors as distinct linked subspaces; keep the two triplets separate.",
                "Use S3 class counts as a separate run diagnostic, not an A5 representation dimension.",
            ],
            ["Browse all 60 actions", "Sector highlighting", "Permutation-cycle display"],
            boundary,
        ),
        encoding="utf-8",
    )
    return _finish_package(
        package,
        package_id=package.name,
        title="A5 symmetry and sector decomposition",
        provenance="computed",
        claim_boundary=boundary,
        source_rows=[
            _source(root / "oph_fpe/defects/z6_a5_action.py", root),
            _source(run / "s3_class_counts.json", root),
        ],
    )


def _em_package(output: Path, root: Path) -> dict[str, Any]:
    package = output / "finite-electromagnetic-response"
    load = [Fraction(0) for _ in range(base_carrier.PORTS)]
    load[0], load[3] = Fraction(1), Fraction(-1)
    potential = green.green_potential(load)
    flux = green.seam_flux(load)
    demo = temporal.demo_bundle(steps=16)
    _write_json(
        package / "data/electromagnetic_response.json",
        {
            "schema": "oph.display.finite-em-response.v1",
            "ports": base_carrier.PORTS,
            "edges": list(zip(base_carrier.SEAM_LEFT, base_carrier.SEAM_RIGHT, strict=True)),
            "faces": base_carrier.ORIENTED_FACES,
            "neutral_load": load,
            "green_potential": potential,
            "seam_flux": flux,
            "temporal_demo": demo,
            "provenance": "fresh exact-rational Green and leapfrog replay",
        },
    )
    boundary = (
        "Exact finite carrier Green and temporal Maxwell identities. Port loads, "
        "seam fluxes, and the step index are not yet identified with physical "
        "charge, electromagnetic fields, or physical time."
    )
    (package / "DISPLAY_INSTRUCTIONS.md").write_text(
        _instructions(
            "Finite electromagnetic response",
            ["data/electromagnetic_response.json"],
            [
                "Render port potential as node height/color and seam flux as signed arrows.",
                "Animate the exact temporal demo by step, showing electric seams and magnetic faces separately.",
                "Display exact fractions on selection and use decimals only for color mapping.",
            ],
            ["Static/temporal toggle", "Step control", "Potential/flux/curvature inspection"],
            boundary,
        ),
        encoding="utf-8",
    )
    return _finish_package(
        package,
        package_id=package.name,
        title="Finite electromagnetic response",
        provenance="computed",
        claim_boundary=boundary,
        source_rows=[
            _source(root / "oph_fpe/em/green.py", root),
            _source(root / "oph_fpe/em/temporal.py", root),
        ],
    )


def _cosmology_package(output: Path, run: Path, root: Path) -> dict[str, Any]:
    package = output / "screen-to-cosmology-diagnostics"
    observables_path = run / "cosmology_observables.json"
    freezeout_path = run / "freezeout_map_summary.json"
    harmonic_path = run / "harmonic_time_trace.npz"
    with np.load(harmonic_path) as harmonic:
        arrays = {key: harmonic[key] for key in harmonic.files}
        selected = {}
        for key, value in arrays.items():
            if value.ndim >= 2 and value.shape[-1] > 256:
                selected[key] = value[..., :256]
            else:
                selected[key] = value
        data_path = package / "data/harmonic_time_display.npz"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data_path, **selected)
    _write_json(
        package / "data/cosmology_screen.json",
        {
            "schema": "oph.display.screen-cosmology.v1",
            "observables": _read_json(observables_path),
            "freezeout": _read_json(freezeout_path),
            "harmonic_selection": "all time rows; first 256 coefficients where wider",
            "provenance": "computed legacy-chart control diagnostics",
        },
    )
    boundary = (
        "Finite screen and freezeout diagnostics from a legacy-chart control run. "
        "They are not a physical CMB prediction, and comparison data are not used "
        "to promote the displayed screen statistics."
    )
    (package / "DISPLAY_INSTRUCTIONS.md").write_text(
        _instructions(
            "Screen-to-cosmology diagnostics",
            ["data/cosmology_screen.json", "data/harmonic_time_display.npz"],
            [
                "Animate harmonic coefficients over the supplied time rows using a fixed coefficient order.",
                "Show freezeout maps and summary diagnostics in separate panes from any reference comparison.",
                "Watermark the scene finite screen diagnostic; do not label it observed sky or physical CMB.",
            ],
            ["Time control", "Field/coefficient selector", "Freezeout summary inspection"],
            boundary,
        ),
        encoding="utf-8",
    )
    return _finish_package(
        package,
        package_id=package.name,
        title="Screen-to-cosmology diagnostics",
        provenance="computed",
        claim_boundary=boundary,
        source_rows=[
            _source(observables_path, root),
            _source(freezeout_path, root),
            _source(harmonic_path, root),
        ],
    )


def _evidence_atlas_package(output: Path, run: Path, root: Path) -> dict[str, Any]:
    package = output / "theorem-paper-simulation-evidence-atlas"
    rer = root.parent / "reverse-engineering-reality"
    axioms = _read_json_yaml(rer / "claims/axiom_registry.yaml")
    claims = _read_json_yaml(rer / "claims/claim_registry.yaml")
    frozen = _read_json(rer / "claims/frozen_prediction_register.json")
    physical = _read_json(rer / "claims/physical_identification_registry.json")
    gravity = _read_json(rer / "claims/gravity_premise_ladder.json")
    lean_modules = []
    declaration = re.compile(r"^\s*(?:theorem|lemma)\s+([A-Za-z0-9_']+)", re.MULTILINE)
    for path in sorted((rer / "Lean").rglob("*.lean")):
        names = declaration.findall(path.read_text(encoding="utf-8"))
        lean_modules.append(
            {
                "path": str(path.relative_to(rer)),
                "theorem_or_lemma_count": len(names),
                "sample_declarations": names[:12],
                "sha256": _sha256(path),
            }
        )
    claim_rows = [
        {
            key: row.get(key)
            for key in (
                "claim_id",
                "statement",
                "owner_paper",
                "tier",
                "status",
                "claim_class",
                "evidence",
                "gates",
                "premise_dependencies",
            )
        }
        for row in claims["claims"]
    ]
    _write_json(
        package / "data/evidence_atlas.json",
        {
            "schema": "oph.display.theorem-evidence-atlas.v1",
            "theory_release": claims.get("release_id"),
            "axioms": axioms["axioms"],
            "claims": claim_rows,
            "lean_modules": lean_modules,
            "frozen_predictions": frozen,
            "physical_identifications": physical,
            "gravity_premise_ladder": gravity,
            "simulator_emergence_ladder": _read_json(run / "emergence_ladder_report.json"),
            "provenance": "canonical theory registries plus computed simulator ladder",
        },
    )
    boundary = (
        "This atlas visualizes canonical claims, theorem modules, evidence links, "
        "prediction custody, and open premise gates. Presence in the graph is not "
        "proof or physical attainment; renderer status must come from each row."
    )
    (package / "DISPLAY_INSTRUCTIONS.md").write_text(
        _instructions(
            "Theorem, paper, and simulation evidence atlas",
            ["data/evidence_atlas.json"],
            [
                "Build a layered graph: axiom → claim → Lean module/evidence → simulator receipt → physical gate.",
                "Style nodes by their explicit status and class; never infer closure from an edge.",
                "Offer paper, theorem, prediction, gravity, and simulator lanes as synchronized filters.",
            ],
            ["Claim search", "Lane/status filters", "Source and premise inspection"],
            boundary,
        ),
        encoding="utf-8",
    )
    return _finish_package(
        package,
        package_id=package.name,
        title="Theorem, paper, and simulation evidence atlas",
        provenance="computed",
        claim_boundary=boundary,
        source_rows=[
            _source(rer / "claims/axiom_registry.yaml", root),
            _source(rer / "claims/claim_registry.yaml", root),
            _source(run / "emergence_ladder_report.json", root),
        ],
    )


def build(output: Path = DEFAULT_OUTPUT, run: Path = DEFAULT_RUN) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    run = (root / run).resolve() if not run.is_absolute() else run.resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    output.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    manifests.extend(_carrier_packages(output, run, root))
    manifests.append(_quantum_package(output, root))
    manifests.append(_depth_package(output, root))
    manifests.append(_repair_package(output, run, root))
    manifests.extend(_observer_packages(output, run, root))
    manifests.extend(_defect_packages(output, run, root))
    manifests.append(_spacetime_package(output, run, root))
    manifests.append(_symmetry_package(output, run, root))
    manifests.append(_em_package(output, root))
    manifests.append(_cosmology_package(output, run, root))
    manifests.append(_evidence_atlas_package(output, run, root))
    suite = {
        "schema": SCHEMA,
        "display_data_only": True,
        "whole_run_included": False,
        "package_count": len(manifests),
        "maximum_package_bytes": MAX_PACKAGE_BYTES,
        "source_run": str(run.relative_to(root)),
        "packages": [
            {
                "package_id": row["package_id"],
                "title": row["title"],
                "total_bytes": row["total_bytes"],
                "manifest_sha256": row["manifest_sha256"],
                "claim_boundary": row["claim_boundary"],
            }
            for row in manifests
        ],
    }
    _write_json(output / "suite_manifest.json", suite)
    index_lines = [
        "# OPH headline visualizer handoffs",
        "",
        "Display-only, directory-based packages. No package contains a whole run; every package is below 200 MB and carries its own sources, hashes, claim boundary, and display instructions.",
        "",
    ]
    for row in manifests:
        index_lines.append(
            f"- `{row['package_id']}/` — {row['title']} ({row['total_bytes']} bytes)"
        )
    index_lines.extend(
        [
            "",
            "Serve this directory with `python3 -m http.server 8000`, then follow each package's `DISPLAY_INSTRUCTIONS.md`.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(index_lines), encoding="utf-8")
    return suite


if __name__ == "__main__":
    result = build()
    print(json.dumps(result, indent=2, sort_keys=True))
