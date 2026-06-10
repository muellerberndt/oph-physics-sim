from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.claims import (
    DECLARED_SHAPE_SUBSTRATE_WITNESS,
    DECLARED_SHAPE_SUBSTRATE_WITNESS_RECEIPT,
    SHAPE_LOOP_PARTICLE_RECEIPT,
    SHAPE_SETTLING_RECEIPT,
)
from oph_fpe.cosmology.cmb_compare import load_planck_tt_binned, cmb_lite_comparison_report
from oph_fpe.cosmology.shape_certificates import shape_cmb_certificate_inputs_report
from oph_fpe.cosmology.shape_projection import project_dodeca_to_screen, write_shape_projection_reports
from oph_fpe.evidence import RunBundle
from oph_fpe.microphysics.dodeca_cell import (
    DodecaCell,
    dodeca_cell_report,
    dodecahedral_cell,
    loop_mode_report,
)
from oph_fpe.microphysics.loop_particles import detect_loop_particles, shape_loop_particle_receipt
from oph_fpe.microphysics.shape_constants import DELTA_P, PHI, P_SHAPE, loop_detuning_phase
from oph_fpe.microphysics.three_way_vertex import scattering_receipt, three_way_scattering_matrix


@dataclass(frozen=True)
class ShapeState:
    amp: np.ndarray
    cycle: int = 0


@dataclass(frozen=True)
class ShapeArcTables:
    tail: np.ndarray
    head: np.ndarray
    reverse: np.ndarray
    incoming_arcs: np.ndarray
    outgoing_arcs: np.ndarray


def build_arc_tables(graph) -> ShapeArcTables:
    arcs: list[tuple[int, int]] = []
    arc_index: dict[tuple[int, int], int] = {}
    for u, v in graph.edges():
        left = (int(u), int(v))
        right = (int(v), int(u))
        arc_index[left] = len(arcs)
        arcs.append(left)
        arc_index[right] = len(arcs)
        arcs.append(right)

    node_count = graph.number_of_nodes()
    incoming = np.full((node_count, 3), -1, dtype=np.int32)
    outgoing = np.full((node_count, 3), -1, dtype=np.int32)
    for node in graph.nodes():
        neighbors = [int(value) for value in graph.neighbors(node)]
        if len(neighbors) != 3:
            raise ValueError(f"vertex {node} has degree {len(neighbors)}, expected 3")
        incoming[int(node), :] = [arc_index[(neighbor, int(node))] for neighbor in neighbors]
        outgoing[int(node), :] = [arc_index[(int(node), neighbor)] for neighbor in neighbors]

    tail = np.asarray([arc[0] for arc in arcs], dtype=np.int32)
    head = np.asarray([arc[1] for arc in arcs], dtype=np.int32)
    reverse = np.asarray([arc_index[(arc[1], arc[0])] for arc in arcs], dtype=np.int32)
    return ShapeArcTables(tail=tail, head=head, reverse=reverse, incoming_arcs=incoming, outgoing_arcs=outgoing)


def face_arc_indices(face: list[int], arc_index: dict[tuple[int, int], int]) -> list[int]:
    return [arc_index[(int(left), int(right))] for left, right in zip(face, face[1:] + face[:1])]


def scatter_step(state: ShapeState, tables: ShapeArcTables) -> ShapeState:
    matrix = three_way_scattering_matrix(dtype=np.complex128)
    incoming = state.amp[tables.incoming_arcs]
    outgoing = incoming @ matrix.T
    next_amp = np.zeros_like(state.amp)
    next_amp[tables.outgoing_arcs.reshape(-1)] = outgoing.reshape(-1)
    return ShapeState(amp=next_amp, cycle=state.cycle + 1)


def loop_phase(amp: np.ndarray, arc_ids: list[int], eps: float = 1.0e-12) -> complex:
    values = amp[np.asarray(arc_ids, dtype=np.int64)]
    phases = values / np.maximum(np.abs(values), eps)
    return complex(np.prod(phases))


def loop_strain(amp: np.ndarray, face_arc_ids: list[list[int]], target_phase: float) -> float:
    target = np.exp(1j * float(target_phase))
    total = 0.0
    for arc_ids in face_arc_ids:
        values = amp[np.asarray(arc_ids, dtype=np.int64)]
        weight = float(np.mean(np.abs(values) ** 2))
        residual = abs(loop_phase(amp, arc_ids) - target) ** 2
        total += weight * residual
    return float(total)


def apply_loop_repair(
    state: ShapeState,
    face_arc_ids: list[list[int]],
    target_phase: float,
    *,
    repair_rate: float,
) -> ShapeState:
    amp = state.amp.copy()
    target = np.exp(1j * float(target_phase))
    correction = np.zeros_like(amp)
    counts = np.zeros_like(amp.real)
    for arc_ids in face_arc_ids:
        phase = loop_phase(amp, arc_ids)
        residual_angle = float(np.angle(phase / target))
        local = np.exp(-1j * residual_angle / max(len(arc_ids), 1))
        for arc_id in arc_ids:
            correction[int(arc_id)] += local
            counts[int(arc_id)] += 1.0
    touched = counts > 0.0
    correction[touched] /= counts[touched]
    amp[touched] = (1.0 - repair_rate) * amp[touched] + repair_rate * amp[touched] * correction[touched]
    return ShapeState(amp=amp, cycle=state.cycle)


def run_shape_dodeca_trace(config: dict[str, Any], *, detuning_multiplier: float = 1.0) -> dict[str, Any]:
    seed = int(config.get("seed", 1))
    rng = np.random.default_rng(seed + int(round(1000 * detuning_multiplier)))
    cell = dodecahedral_cell()
    tables = build_arc_tables(cell.graph)
    arc_index = {(int(tables.tail[i]), int(tables.head[i])): int(i) for i in range(len(tables.tail))}
    face_arcs = [face_arc_indices(face, arc_index) for face in cell.faces]
    amp = rng.normal(size=len(tables.tail)) + 1j * rng.normal(size=len(tables.tail))
    amp = amp / max(float(np.sqrt(np.sum(np.abs(amp) ** 2))), 1.0e-12)
    state = ShapeState(amp=amp)
    cycles = int(config.get("cycles", 512))
    repair_rate = float(config.get("repair_rate", 0.02))
    total_detuning = float(config.get("loop_detuning_phase", loop_detuning_phase())) * float(detuning_multiplier)
    target_loop_phase = 2.0 * np.pi + total_detuning
    phi_trace: list[float] = []
    power_trace: list[float] = []
    particle_history: list[list[dict[str, Any]]] = []
    energy_threshold = float(config.get("particle_energy_threshold", 0.001))

    for _ in range(cycles):
        state = scatter_step(state, tables)
        state = apply_loop_repair(state, face_arcs, target_loop_phase, repair_rate=repair_rate)
        phi_trace.append(loop_strain(state.amp, face_arcs, target_loop_phase))
        power_trace.append(float(np.sum(np.abs(state.amp) ** 2)))
        particle_history.append(detect_loop_particles(state.amp, face_arcs, energy_threshold=energy_threshold))

    face_rows = _face_observable_rows(
        state.amp,
        face_arcs,
        target_loop_phase=target_loop_phase,
        phi_trace=phi_trace,
    )
    tracks = _track_loop_particles(particle_history)
    return {
        "cell": cell,
        "tables": tables,
        "face_arcs": face_arcs,
        "state": state,
        "delta_P": float(DELTA_P * detuning_multiplier),
        "detuning_multiplier": float(detuning_multiplier),
        "target_loop_phase": float(target_loop_phase),
        "repair_rate": repair_rate,
        "cycles": cycles,
        "phi_initial": float(phi_trace[0]) if phi_trace else None,
        "phi_final": float(phi_trace[-1]) if phi_trace else None,
        "phi_trace": phi_trace,
        "power_trace": power_trace,
        "minus_dphi_dt_mean": float(np.mean(-np.diff(phi_trace))) if len(phi_trace) > 1 else 0.0,
        "face_rows": face_rows,
        "particle_tracks": tracks,
    }


def run_shape_dodeca_smoke(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    config = dict(config)
    seed = int(config.get("seed", 1))
    run_id = str(config.get("run_id") or f"{config.get('name', 'shape_dodeca_smoke')}_{seed}_{int(time.time() * 1000)}")
    bundle = RunBundle(Path(out_dir), run_id)
    bundle.write_config(config)

    trace = run_shape_dodeca_trace(config)
    vertex_report = scattering_receipt()
    cell_report = dodeca_cell_report()
    mode_report = loop_mode_report(trace["cell"])
    settling_report = _settling_report(trace)
    particle_report = shape_loop_particle_receipt(
        trace["particle_tracks"],
        min_lifetime=int(config.get("particle_min_lifetime", 8)),
    )
    projection_report = write_shape_projection_reports(
        bundle.path,
        trace["face_rows"],
        ell_max=int(config.get("ell_max", 8)),
        seed=seed,
        n_jobs=int(config.get("n_jobs", 1)),
        cell=trace["cell"],
    )
    fields_matrix = _feature_matrix_from_face_rows(trace["face_rows"])
    primary_field = fields_matrix[:, 0] if fields_matrix.size else np.zeros(0, dtype=float)
    detuning_multipliers = [float(value) for value in config.get("certificate_detuning_multipliers", [0.5, 1.0, 1.5, 2.0])]
    certificate_runs = [
        _strip_trace_for_certificate(run_shape_dodeca_trace(config, detuning_multiplier=value))
        for value in detuning_multipliers
    ]
    cert_report = shape_cmb_certificate_inputs_report(
        certificate_runs,
        projection_report,
        field_matrix=fields_matrix,
        primary_field=primary_field,
    )

    reports = {
        "shape_vertex_scattering_report.json": vertex_report,
        "shape_dodeca_cell_report.json": cell_report,
        "shape_loop_mode_report.json": mode_report,
        "shape_settling_report.json": settling_report,
        "shape_particle_loop_report.json": particle_report,
        "shape_loop_particle_report.json": particle_report,
        "shape_cmb_certificate_inputs.json": cert_report,
    }
    for filename, report in reports.items():
        bundle.write_json(filename, report)
    _write_shape_settling_trace_csv(bundle.path / "shape_settling_trace.csv", trace)
    _write_shape_loop_particles_csv(bundle.path / "shape_loop_particles.csv", trace["particle_tracks"])
    _write_shape_screen_cl_csv(bundle.path / "shape_screen_cl.csv", projection_report)
    cmb_report = _maybe_write_cmb_lite(config, bundle.path)
    summary = _shape_summary(
        run_id=run_id,
        vertex_report=vertex_report,
        cell_report=cell_report,
        mode_report=mode_report,
        settling_report=settling_report,
        particle_report=particle_report,
        projection_report=projection_report,
        cert_report=cert_report,
        cmb_report=cmb_report,
    )
    bundle.write_json("shape_substrate_summary.json", summary)
    bundle.write_manifest(
        {
            "run_id": run_id,
            "name": config.get("name", "shape_dodeca_smoke"),
            "claim_level": DECLARED_SHAPE_SUBSTRATE_WITNESS,
            "claim_boundary": summary["claim_boundary"],
            "patch_count": 20,
            "face_count": 12,
            "arc_count": int(len(trace["tables"].tail)),
            "cycles": int(trace["cycles"]),
            "physical_cmb_prediction": False,
            "neutral_oph_bulk_claim": False,
            "declared_3d_substrate": True,
            "bundle_files": sorted(path.name for path in bundle.path.iterdir()),
        }
    )
    return {"run_id": run_id, "path": str(bundle.path), **summary}


def run_shape_ensemble(
    config: dict[str, Any],
    out_dir: Path,
    *,
    seeds: list[int],
    workers: int = 1,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(dict(config), str(out_dir), int(seed)) for seed in seeds]
    if int(workers) <= 1 or len(tasks) <= 1:
        rows = [_run_shape_seed(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=get_context("spawn")) as pool:
            rows = list(pool.map(_run_shape_seed, tasks))
    report = {
        "mode": "shape_dodeca_ensemble",
        "run_count": len(rows),
        "rows": rows,
        "vertex_scattering_receipt_count": sum(1 for row in rows if row.get("shape_vertex_scattering_receipt")),
        "dodeca_cell_receipt_count": sum(1 for row in rows if row.get("shape_dodeca_cell_receipt")),
        "shape_settling_receipt_count": sum(1 for row in rows if row.get("shape_settling_receipt")),
        "loop_particle_receipt_count": sum(1 for row in rows if row.get("shape_loop_particle_receipt")),
        "screen_projection_receipt_count": sum(1 for row in rows if row.get("shape_screen_projection_receipt")),
        "selector_elimination_target_input_count": sum(
            1 for row in rows if row.get("shape_selector_elimination_target_input_receipt")
        ),
        "cmb_certificate_input_count": sum(1 for row in rows if row.get("shape_cmb_certificate_input_receipt")),
        "mean_q_IR_candidate": _mean(row.get("shape_q_IR_candidate") for row in rows),
        "mean_ell_IR_candidate": _mean(row.get("shape_ell_IR_candidate") for row in rows),
        "mean_phi_drop_fraction": _mean(row.get("shape_phi_drop_fraction") for row in rows),
        "mean_loop_particle_count": _mean(row.get("shape_loop_particle_count") for row in rows),
        "physical_cmb_prediction": False,
        "claim_boundary": "Declared Shape substrate ensemble witness. Not neutral OPH bulk proof.",
    }
    (out_dir / "shape_ensemble_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _run_shape_seed(task: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
    config, out_dir, seed = task
    config = dict(config)
    config["seed"] = int(seed)
    return run_shape_dodeca_smoke(config, Path(out_dir))


def _settling_report(trace: dict[str, Any]) -> dict[str, Any]:
    phi_initial = float(trace.get("phi_initial") or 0.0)
    phi_final = float(trace.get("phi_final") or 0.0)
    drop_fraction = (phi_initial - phi_final) / max(abs(phi_initial), 1.0e-12)
    return {
        "receipt": SHAPE_SETTLING_RECEIPT,
        "receipt_name": SHAPE_SETTLING_RECEIPT,
        "passed": bool(phi_final < phi_initial),
        "phi_initial": phi_initial,
        "phi_final": phi_final,
        "phi_drop_fraction": float(drop_fraction),
        "phi_trace": trace.get("phi_trace", []),
        "minus_dphi_dt_mean": trace.get("minus_dphi_dt_mean"),
        "power_initial": trace.get("power_trace", [None])[0] if trace.get("power_trace") else None,
        "power_final": trace.get("power_trace", [None])[-1] if trace.get("power_trace") else None,
        "declared_repair": "declared Shape settling repair over pentagonal loop phase closure",
        "claim_level": DECLARED_SHAPE_SUBSTRATE_WITNESS,
        "neutral_oph_bulk_claim": False,
        "physical_cmb_prediction": False,
    }


def _write_shape_settling_trace_csv(path: Path, trace: dict[str, Any]) -> None:
    phi_trace = [float(value) for value in trace.get("phi_trace", [])]
    power_trace = [float(value) for value in trace.get("power_trace", [])]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cycle", "phi_loop_strain", "power", "minus_delta_phi"])
        writer.writeheader()
        for cycle, phi in enumerate(phi_trace):
            previous = phi_trace[cycle - 1] if cycle > 0 else phi
            power = power_trace[cycle] if cycle < len(power_trace) else ""
            writer.writerow(
                {
                    "cycle": int(cycle),
                    "phi_loop_strain": phi,
                    "power": power,
                    "minus_delta_phi": float(previous - phi),
                }
            )


def _write_shape_loop_particles_csv(path: Path, particle_tracks: list[dict[str, Any]]) -> None:
    fieldnames = [
        "track_id",
        "face_id",
        "mode",
        "particle_class",
        "birth_cycle",
        "death_cycle",
        "lifetime",
        "max_energy",
        "class_preserved",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for track_id, row in enumerate(particle_tracks):
            writer.writerow(
                {
                    "track_id": int(track_id),
                    "face_id": row.get("face_id", ""),
                    "mode": row.get("mode", ""),
                    "particle_class": row.get("particle_class", ""),
                    "birth_cycle": row.get("birth_cycle", ""),
                    "death_cycle": row.get("death_cycle", ""),
                    "lifetime": row.get("lifetime", ""),
                    "max_energy": row.get("max_energy", ""),
                    "class_preserved": row.get("class_preserved", ""),
                }
            )


def _write_shape_screen_cl_csv(path: Path, projection_report: dict[str, Any]) -> None:
    fieldnames = ["field", "ell", "C_ell", "D_ell", "peak_ell", "peak_D_ell", "low_ell_power_2_10"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for field_name, field_report in sorted((projection_report.get("fields") or {}).items()):
            for row in field_report.get("spectrum", []):
                writer.writerow(
                    {
                        "field": field_name,
                        "ell": row.get("ell", ""),
                        "C_ell": row.get("C_ell", ""),
                        "D_ell": row.get("D_ell", ""),
                        "peak_ell": field_report.get("peak_ell", ""),
                        "peak_D_ell": field_report.get("peak_D_ell", ""),
                        "low_ell_power_2_10": field_report.get("low_ell_power_2_10", ""),
                    }
                )


def _face_observable_rows(
    amp: np.ndarray,
    face_arcs: list[list[int]],
    *,
    target_loop_phase: float,
    phi_trace: list[float],
) -> list[dict[str, Any]]:
    particles = detect_loop_particles(amp, face_arcs, energy_threshold=0.0)
    particle_by_face = {int(row["face_id"]): row for row in particles}
    rows: list[dict[str, Any]] = []
    minus_dphi = float(np.mean(-np.diff(phi_trace))) if len(phi_trace) > 1 else 0.0
    for face_id, arc_ids in enumerate(face_arcs):
        values = amp[np.asarray(arc_ids, dtype=np.int64)]
        phase = loop_phase(amp, arc_ids)
        residual = abs(phase - np.exp(1j * target_loop_phase))
        particle = particle_by_face.get(face_id, {})
        rows.append(
            {
                "face_id": int(face_id),
                "mode_energy": float(particle.get("energy", np.mean(np.abs(values) ** 2))),
                "particle_density": float(1.0 if particle else 0.0),
                "phi_wave": float(residual),
                "minus_dphi_dt": minus_dphi,
                "phase_coherence": float(max(0.0, 1.0 - residual / 2.0)),
            }
        )
    return rows


def _track_loop_particles(history: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    active: dict[tuple[int, int], dict[str, Any]] = {}
    for cycle, particles in enumerate(history):
        current = {(int(row["face_id"]), int(row["mode"])): row for row in particles}
        for key, row in current.items():
            if key not in active:
                active[key] = {
                    "face_id": key[0],
                    "mode": key[1],
                    "particle_class": row["particle_class"],
                    "birth_cycle": int(cycle),
                    "death_cycle": int(cycle),
                    "lifetime": 1,
                    "max_energy": float(row["energy"]),
                    "class_preserved": True,
                }
            else:
                active[key]["death_cycle"] = int(cycle)
                active[key]["lifetime"] = int(active[key]["lifetime"]) + 1
                active[key]["max_energy"] = max(float(active[key]["max_energy"]), float(row["energy"]))
    return list(active.values())


def _feature_matrix_from_face_rows(face_rows: list[dict[str, Any]]) -> np.ndarray:
    names = ["loop_mode_energy", "loop_particle_density", "phi_wave", "minus_dphi_dt", "phase_coherence"]
    rows = []
    for row in face_rows:
        rows.append(
            [
                float(row.get("mode_energy", 0.0)),
                float(row.get("particle_density", 0.0)),
                float(row.get("phi_wave", 0.0)),
                float(row.get("minus_dphi_dt", 0.0)),
                float(row.get("phase_coherence", 0.0)),
            ]
        )
    matrix = np.asarray(rows, dtype=float)
    identity = np.ones((matrix.shape[0], 1), dtype=float)
    return np.hstack([matrix, identity]) if matrix.size else np.zeros((0, len(names) + 1), dtype=float)


def _strip_trace_for_certificate(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "delta_P": trace["delta_P"],
        "phi_trace": trace["phi_trace"],
        "detuning_multiplier": trace["detuning_multiplier"],
    }


def _maybe_write_cmb_lite(config: dict[str, Any], run_path: Path) -> dict[str, Any]:
    benchmark_path = config.get("benchmark_path")
    if not benchmark_path:
        return {}
    path = Path(benchmark_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    rows = load_planck_tt_binned(path)
    cl_report = json.loads((run_path / "cl_comparison_report.json").read_text(encoding="utf-8"))
    report = cmb_lite_comparison_report(
        cl_report,
        rows,
        benchmark_label=str(config.get("benchmark_label", "Planck2018_TT_binned")),
        source_url=config.get("source_url"),
    )
    (run_path / "cmb_lite_comparison_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _shape_summary(
    *,
    run_id: str,
    vertex_report: dict[str, Any],
    cell_report: dict[str, Any],
    mode_report: dict[str, Any],
    settling_report: dict[str, Any],
    particle_report: dict[str, Any],
    projection_report: dict[str, Any],
    cert_report: dict[str, Any],
    cmb_report: dict[str, Any],
) -> dict[str, Any]:
    best_field = cmb_report.get("best_positive_shape_field") or cmb_report.get("best_shape_field")
    best = (cmb_report.get("field_comparisons", {}) or {}).get(best_field, {}) if best_field else {}
    return {
        "mode": "declared_shape_substrate_witness",
        "run_id": run_id,
        "receipt": DECLARED_SHAPE_SUBSTRATE_WITNESS_RECEIPT,
        "receipt_name": DECLARED_SHAPE_SUBSTRATE_WITNESS_RECEIPT,
        "shape_vertex_scattering_receipt": bool(vertex_report.get("passed")),
        "shape_dodeca_cell_receipt": bool(cell_report.get("passed")),
        "shape_loop_mode_receipt": bool(mode_report.get("passed")),
        "shape_settling_receipt": bool(settling_report.get("passed")),
        "shape_loop_particle_receipt": bool(particle_report.get("passed")),
        "shape_screen_projection_receipt": bool(projection_report.get("passed")),
        "shape_selector_elimination_target_input_receipt": bool(
            cert_report.get("selector_elimination_target_input_receipt")
        ),
        "shape_cmb_certificate_input_receipt": bool(cert_report.get("passed")),
        "passed": bool(vertex_report.get("passed") and cell_report.get("passed") and projection_report.get("passed")),
        "shape_phi_drop_fraction": settling_report.get("phi_drop_fraction"),
        "shape_loop_particle_count": particle_report.get("persistent_loop_particle_count"),
        "shape_q_IR_candidate": cert_report.get("q_IR_candidate"),
        "shape_q_IR_runtime_zero_mode": cert_report.get("q_IR_runtime_zero_mode"),
        "shape_ell_IR_candidate": cert_report.get("ell_IR_candidate"),
        "shape_ell_IR_runtime_covariance_rank": (cert_report.get("ell_IR_runtime_covariance") or {}).get(
            "visible_covariance_rank"
        ),
        "shape_eta_R_candidate": cert_report.get("eta_R_candidate"),
        "shape_planck_lite_shape_correlation": best.get("shape_correlation"),
        "shape_planck_lite_normalized_rmse": best.get("normalized_rmse"),
        "best_shape_field": best_field,
        "claim_level": DECLARED_SHAPE_SUBSTRATE_WITNESS,
        "neutral_oph_bulk_claim": False,
        "physical_cmb_prediction": False,
        "declared_3d_substrate": True,
        "P_shape": P_SHAPE,
        "phi": PHI,
        "delta_P": DELTA_P,
        "claim_boundary": (
            "Declared Alex/Shape dodecahedral substrate witness. It emits useful loop, screen, and "
            "C_l diagnostic readouts, but it is not neutral OPH bulk emergence and not a physical CMB prediction."
        ),
    }


def _mean(values) -> float | None:
    rows = [float(value) for value in values if isinstance(value, (int, float)) and np.isfinite(float(value))]
    return float(np.mean(rows)) if rows else None
