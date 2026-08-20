"""Assemble the consensus-to-physics visualizer handoff from a completed run.

Copies exactly the data the 32-panel brief cites, downsampling the two oversized
sidecars, and writes a per-panel data index so the coding agent never has to guess
which file backs which visualization.

    python tools/build_consensus_to_physics_handoff.py <run_dir> <out_dir> \
        [--control-run <dir>] [--qm-dir <dir>]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools._consensus_to_physics_dirdocs import DIR_DOCS

# panel -> (data files it needs, relative to the handoff data/ root)
PANEL_DATA: dict[str, list[str]] = {
    "V01": ["derived/physics_payload.json"],
    "V02": ["derived/physics_payload.json", "run/mismatch_trace.csv"],
    "V03": ["run/finite_consensus_replay_report.json",
            "run/observer_checkpoint_restoration_report.json"],
    "V04": ["run/mismatch_trace.csv", "derived/physics_payload.json"],
    "V05": ["run/mismatch_trace.csv", "run/freezeout_map_summary.json",
            "run/freezeout_fields.npz"],
    "V06": ["derived/physics_payload.json", "derived/a5_symmetry.json"],
    "V07": ["derived/physics_payload.json", "receipts/port_gram_completion_bridge_receipt.json"],
    "V08": ["receipts/port_gram_completion_bridge_receipt.json"],
    "V09": ["receipts/port_load_metric_quotient_receipt.json",
            "receipts/seam_current_same_metric_scale_receipt.json"],
    "V10": ["receipts/port_repair_propagation_bridge_receipt.json"],
    "V12": ["run/observer_modular_experience_report.json", "run/observer_perspective_rows.csv",
            "timeline/observers_full_128.json", "timeline/observer_anatomy.json"],
    "V12b": ["screen/README.md", "screen/screen_points.csv",
             "screen/screen_frames_record_port_entropy_65536x48.bin",
             "screen/screen_frames_local_mismatch_density_65536x48.bin",
             "screen/screen_frames_modular_depth_65536x48.bin",
             "screen/screen_frames_cumulative_repair_load_65536x48.bin",
             "timeline/subjective_observer_camera_frames.csv",
             "timeline/subjective_observer_cameras.csv", "timeline/cameras_full_128.json",
             "run/harmonic_time_trace.npz", "run/mismatch_trace.csv"],
    "V13": ["run/mismatch_trace.csv", "derived/physics_payload.json"],
    "V14": ["run/mismatch_trace.csv", "control/mismatch_trace.csv"],
    "V15": ["run/mismatch_trace.csv", "run/freezeout_map_summary.json",
            "run/freezeout_fields.npz"],
    "V16": ["run/finite_repair_transition_matrix_report.json",
            "run/finite_repair_transition_rows.csv", "run/finite_repair_transition_matrix.npz"],
    "V17": ["derived/QM_OBSERVER_VIZ.v1.json", "derived/QM_OBSERVER_RECEIPT.v1.json"],
    "V18": ["run/central_record_born_report.json",
            "timeline/hilbert_space_observer_algebra.json"],
    "V19": ["derived/QM_OBSERVER_VIZ.v1.json"],
    "V20": ["derived/QM_OBSERVER_VIZ.v1.json"],
    "V21": ["derived/physics_payload.json", "derived/a5_symmetry.json",
            "run/array_holonomy_report.json",
            "run/s3_class_counts.json", "run/s3_gauge_state.npz"],
    "V22": ["derived/physics_payload.json"],
    "V23": ["derived/physics_payload.json"],
    "V24": ["run/defect_timeline_report.json", "run/defect_h3_worldlines_report.json",
            "timeline/proto_particle_worldlines.csv",
            "timeline/proto_particle_worldline_events.csv",
            "timeline/screen_cluster_tracks.csv",
            "timeline/organic_defect_population_trajectory.csv",
            "timeline/organic_defect_population_worldline_events.csv",
            "timeline/observer_proto_worldline_sightings_sample.csv"],
    "V24b": ["timeline/observer_proto_worldline_sightings_sample.csv",
             "timeline/proto_particle_worldlines.csv",
             "timeline/proto_particle_worldline_events.csv",
             "timeline/screen_cluster_tracks.csv", "timeline/cameras_full_128.json"],
    "V25": ["derived/electromagnetic_response.json"],
    "V26": ["run/yang_mills_gap_certificate_report.json",
            "timeline/yang_mills_su2_plaquette_trace.csv",
            "timeline/yang_mills_su2_wilson_loop_trace.csv",
            "timeline/yang_mills_su2_polyakov_loop_trace.csv",
            "timeline/yang_mills_su2_orientation_plaquettes.csv",
            "timeline/yang_mills_gap_promotion_gates.csv",
            "timeline/reference_vacuum_scalar_spectrum.csv",
            "timeline/reference_vacuum_u1_plaquette_trace.csv"],
    "V27": ["run/observer_consensus_report.json", "run/observer_perspective_rows.csv",
            "run/observer_population_report.json", "timeline/observers_full_128.json"],
    "V28": ["run/modular_response_h3_report.json", "run/conformal_h3_spatial_chart_report.json",
            "run/observer_chart_object_h3_report.json", "run/h3_objects.csv",
            "timeline/consensus_h3_objects.csv", "timeline/cameras_full_128.json",
            "timeline/subjective_observer_cameras.csv",
            "timeline/subjective_observer_camera_frames.csv"],
    "V29": ["timeline/emergent_curved_spacetime.json",
            "timeline/emergent_curved_spacetime_curvature_proxy.csv",
            "timeline/emergent_curved_spacetime_time_slices.csv",
            "timeline/emergent_curved_spacetime_continuous_field.csv",
            "run/array_holonomy_report.json"],
    "V29b": ["timeline/emergent_curved_spacetime.json",
             "timeline/emergent_curved_spacetime_curvature_proxy.csv",
             "timeline/emergent_curved_spacetime_continuous_field.csv",
             "timeline/emergent_curved_spacetime_time_slices.csv",
             "timeline/consensus_h3_objects.csv"],
    "V31": ["run/cl_comparison_report.json", "run/cmb_lite_comparison_report.json",
            "run/freezeout_map_summary.json", "run/harmonic_time_trace.npz",
            "timeline/cmb_screen_spectrum_rows.csv", "timeline/cmb_residual_rows.csv",
            "screen/screen_points.csv", "screen/README.md"],
    "V32": ["run/simulation_assumption_manifest.json", "run/config.yml"],
}

# observer-sky raw frames: all four fields, flat float32, 65536 x 48
SCREEN_BINS = ["cumulative_repair_load", "local_mismatch_density",
               "modular_depth", "record_port_entropy"]

MAX_SIGHTING_ROWS = 24000
MAX_CAMERA_FRAME_ROWS = 12288


def _listify(value):
    """numpy arrays and tuples -> plain JSON-serialisable lists."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_listify(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _copy(src: Path, dst: Path, missing: list[str]) -> bool:
    if not src.exists():
        missing.append(str(src))
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _subsample_csv(src: Path, dst: Path, max_rows: int, missing: list[str]) -> bool:
    """Keep the header and an evenly-spaced sample of rows, deterministically."""
    if not src.exists():
        missing.append(str(src))
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    if len(rows) > max_rows:
        step = len(rows) / max_rows
        rows = [rows[int(i * step)] for i in range(max_rows)]
    with dst.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--control-run", type=Path, default=None)
    ap.add_argument("--qm-dir", type=Path, default=None)
    ap.add_argument("--payload", type=Path, default=None)
    args = ap.parse_args()

    run, out = args.run, args.out
    tl = run / "universe_timeline"
    data = out / "data"
    missing: list[str] = []

    # ---- root reports and traces
    wanted_run = sorted({f for files in PANEL_DATA.values() for f in files
                         if f.startswith("run/")})
    for rel in wanted_run:
        _copy(run / rel.split("/", 1)[1], data / rel, missing)

    # ---- timeline sidecars
    wanted_tl = sorted({f for files in PANEL_DATA.values() for f in files
                        if f.startswith("timeline/")})
    for rel in wanted_tl:
        name = rel.split("/", 1)[1]
        if name == "observer_proto_worldline_sightings_sample.csv":
            _subsample_csv(tl / "observer_proto_worldline_sightings.csv",
                           data / rel, MAX_SIGHTING_ROWS, missing)
        elif name == "subjective_observer_camera_frames.csv":
            _subsample_csv(tl / name, data / rel, MAX_CAMERA_FRAME_ROWS, missing)
        else:
            _copy(tl / name, data / rel, missing)

    # ---- committed receipts (repo, not run)
    for rel in sorted({f for files in PANEL_DATA.values() for f in files
                       if f.startswith("receipts/")}):
        _copy(ROOT / "data/repair_closure" / rel.split("/", 1)[1], data / rel, missing)

    # ---- observer sky: raw screen frames
    screen = data / "screen"
    screen.mkdir(parents=True, exist_ok=True)
    frames = []
    for field in SCREEN_BINS:
        name = f"screen_frames_{field}_65536x48.bin"
        if _copy(tl / name, screen / name, missing):
            frames.append({"field": field, "file": f"screen/{name}",
                           "points": 65536, "frames": 48,
                           "dtype": "float32", "layout": "frame-major, row = one cycle"})
    _copy(tl / "screen_points.csv", screen / "screen_points.csv", missing)
    _copy(tl / "screen_full_65536.bin", screen / "screen_full_65536.bin", missing)
    (screen / "README.md").write_text(
        "# Screen frames\n\n"
        "`screen_frames_<field>_65536x48.bin` is flat little-endian float32: 48 frames of 65,536\n"
        "patches, frame-major. Read frame k as bytes [k*65536*4, (k+1)*65536*4).\n\n"
        "Each frame is **standardised per frame** (mean 0, standard deviation 1). They show\n"
        "pattern, not amplitude — amplitude lives in `run/mismatch_trace.csv`.\n\n"
        "`screen_points.csv` gives the direction of every patch on the observer's sphere. Join by\n"
        "row order: row i of the CSV is index i in the binary.\n\n"
        "Cycle indices for the 48 frames are in `run/harmonic_time_trace.npz` under `cycles`.\n\n"
        "See `DISPLAY_INSTRUCTIONS.md` in this directory for how to render it.\n")

    # ---- control run (V14 needs the size contrast)
    if args.control_run:
        _copy(args.control_run / "mismatch_trace.csv", data / "control/mismatch_trace.csv", missing)

    # ---- derived: exact blocks computed here, not read from the run
    from fractions import Fraction

    from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower
    from oph_fpe.defects.z6_a5_action import rotation_group
    from oph_fpe.defects.z6_carrier_defects import base_carrier_spec
    from oph_fpe.em import base_carrier, green, temporal

    dv = data / "derived"
    dv.mkdir(parents=True, exist_ok=True)

    load = [Fraction(0) for _ in range(base_carrier.PORTS)]
    load[0], load[3] = Fraction(1), Fraction(-1)
    (dv / "electromagnetic_response.json").write_text(json.dumps({
        "schema": "oph.display.finite-em-response.v1",
        "ports": base_carrier.PORTS,
        "edges": [list(e) for e in zip(base_carrier.SEAM_LEFT, base_carrier.SEAM_RIGHT,
                                       strict=True)],
        "unit_dipole_load": [str(x) for x in load],
        "green_potential": [str(x) for x in green.green_potential(load)],
        "seam_flux": [str(x) for x in green.seam_flux(load)],
        "temporal_bundle": temporal.demo_bundle(steps=16),
        "provenance": "exact finite carrier Green and temporal Maxwell identities",
    }, indent=2, default=str) + "\n")

    base = build_geodesic_icosahedral_tower(0).levels[0]
    rotations = rotation_group(base_carrier_spec())
    (dv / "a5_symmetry.json").write_text(json.dumps({
        "schema": "oph.display.a5-symmetry.v1",
        "vertices": _listify(base.vertices), "edges": _listify(base.edges),
        "faces": _listify(base.faces),
        "rotations": _listify(rotations), "rotation_count": len(rotations),
        "sector_decomposition": [1, 3, 5, 3],
        "note": "block dimensions in damping order: constant, slow(3), middle(5), fast(3')",
    }, indent=2) + "\n")

    if args.payload:
        _copy(args.payload, dv / "physics_payload.json", missing)

    if args.qm_dir:
        for n in ("QM_OBSERVER_VIZ.v1.json", "QM_OBSERVER_RECEIPT.v1.json"):
            _copy(args.qm_dir / n, data / "derived" / n, missing)

    # ---- per-directory display instructions
    for name, text in DIR_DOCS.items():
        d = data / name
        if d.is_dir():
            (d / "DISPLAY_INSTRUCTIONS.md").write_text(text)
        else:
            missing.append(f"{d}  (no directory for DISPLAY_INSTRUCTIONS)")

    index = {
        "schema": "oph.visualizer-handoff.panel-data-index.v1",
        "source_run": str(run),
        "panels": {k: sorted(v) for k, v in sorted(PANEL_DATA.items())},
        "screen_frames": frames,
        "subsampled": {
            "timeline/observer_proto_worldline_sightings_sample.csv":
                f"evenly spaced sample, at most {MAX_SIGHTING_ROWS} of 196608 rows",
        },
    }
    (data / "PANEL_DATA_INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    lines = ["# Panel data index", "",
             "Which file backs which visualization. Paths are relative to `data/`.", ""]
    for pid in sorted(PANEL_DATA):
        lines.append(f"**{pid}** — " + ", ".join(f"`{f}`" for f in sorted(PANEL_DATA[pid])))
        lines.append("")
    (out / "DATA_INDEX.md").write_text("\n".join(lines))

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"handoff data: {total/1e6:.1f} MB across "
          f"{sum(1 for p in out.rglob('*') if p.is_file())} files")
    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for m in missing:
            print("  ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
