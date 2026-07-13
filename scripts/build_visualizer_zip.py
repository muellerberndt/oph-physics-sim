#!/usr/bin/env python3
"""Assemble the web-coding-agent ZIP bundle from a completed OPH universe run.

Target ~200 MB, hard max 256 MB (fails the build if exceeded).
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

RUN_DIR = Path(sys.argv[1]).resolve()
OUT_ZIP = Path(sys.argv[2]).resolve()
REPO = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else Path.cwd()
HARD_MAX = 256_000_000
TARGET = 210_000_000

TL = RUN_DIR / "universe_timeline"
ROOT = "oph_visualizer_bundle"

entries: list[tuple[Path, str]] = []


def add(source: Path, arc: str, required: bool = True) -> None:
    if not source.exists():
        if required:
            raise FileNotFoundError(source)
        return
    entries.append((source, f"{ROOT}/{arc}"))


# --- payload + bundle-level docs -------------------------------------------
add(TL / "visualization_payload.json", "payload/visualization_payload.json")
add(TL / "visualization_export_manifest.json", "payload/visualization_export_manifest.json", required=False)
add(TL / "universe_timeline_summary.json", "payload/universe_timeline_summary.json")
add(TL / "oph_visualizer_pack_v2.tar.zst", "payload/oph_visualizer_pack_v2.tar.zst", required=False)

for doc in ("VISUALIZATION_INSTRUCTIONS.md", "WEB_CODING_AGENT_VISUALIZATION_BRIEF.md"):
    add(TL / doc, f"docs/{doc}")
for doc in (
    "oph_universe_timeline_visualization_payload_v1.schema.json",
    "oph_visualizer_pack_v2.schema.json",
    "SIMULATION_ASSUMPTION_POLICY.md",
    "VISUALIZATION_APP_AGENT_MANUAL.md",
    "SCALING_MILESTONE_ESTIMATES_2026-07-13.md",
    "BEST_OF_PUBLIC_DATA_COMPARISONS.md",
):
    add(REPO / "docs" / doc, f"docs/{doc}")

# --- sidecars ----------------------------------------------------------------
for pattern in ("*.csv", "*.bin", "observers_full_*.json", "cameras_full_*.json",
                "effective_string_theory.json", "emergent_curved_spacetime.json",
                "hilbert_space_observer_algebra.json", "observer_anatomy.json",
                "observer_cinema.json", "paper_accuracy.json",
                "visualization_render_data.json"):
    for path in sorted(TL.glob(pattern)):
        add(path, f"sidecars/{path.name}")

# --- run-level reports (context for the coding agent) -----------------------
RUN_REPORTS = [
    "manifest.json",
    "config.yml",
    "RUN_HIGHLIGHTS.md",
    "run_highlights.json",
    "AUTO_THEOREM_UNIVERSE_SUMMARY.json",
    "simulation_assumption_manifest.json",
    "bulk_proof_certificate_report.json",
    "observer_consensus_report.json",
    "observer_modular_experience_report.json",
    "cl_comparison_report.json",
    "cmb_lite_comparison_report.json",
    "freezeout_map_summary.json",
    "harmonic_time_trace_report.json",
    "organic_defect_population_report.json",
    "free_two_defect_dynamics_report.json",
    "two_defect_stress_contraction_assay_report.json",
    "visualization_defect_diagnostics_summary.json",
    "defect_timeline_report.json",
    "h3_objects.csv",
    "observer_perspective_rows.csv",
    "mismatch_trace.csv",
    "observer_agreement_report.json",
    "agreement_bulk_field_summary.json",
    "screen_parity_report.json",
    "defect_worldline_turning_report.json",
    "s3_class_counts.json",
    "strict_neutral_bulk_report.json",
    "neutral_3d_bulk_audit_report.json",
    "neutral_3d_bulk_audit_report.md",
    "einstein_bridge_manifest.json",
    "strict_neutral_bulk_frontier_report.json",
    "transition_scale_selection_report.json",
]
for name in RUN_REPORTS:
    add(RUN_DIR / name, f"run_reports/{name}", required=False)

# --- raw arrays for advanced rendering --------------------------------------
for name in (
    "freezeout_fields.npz",
    "screen_evolution_frames.npz",
    "harmonic_time_trace.npz",
    "s3_gauge_state.npz",
    "agreement_bulk_field.npz",
):
    add(RUN_DIR / name, f"data/{name}", required=False)

# Full observer-view rows (all observers, spectra + histograms + repair tensors)
add(RUN_DIR / "observer_views.jsonl", "data/observer_views.jsonl", required=False)

# Reference single-file viewers produced by the simulator (working examples)
for name in ("oph_realtime_viewer.html", "object_h3_bulk_viewer.html", "cmb_neutral_frontier_viewer.html"):
    add(RUN_DIR / name, f"reference_viewers/{name}", required=False)

# Additional context reports
for name in (
    "array_holonomy_report.json",
    "defect_h3_worldlines_report.json",
    "modular_response_h3_report.json",
    "observer_chart_object_h3_report.json",
    "defect_interaction_report.json",
    "record_family_h3_report.json",
):
    add(RUN_DIR / name, f"run_reports/{name}", required=False)

# README is provided by caller next to this script
readme = Path(sys.argv[0]).parent / "README_FOR_WEB_CODING_AGENT.md"
add(readme, "README_FOR_WEB_CODING_AGENT.md")

# --- write zip ---------------------------------------------------------------
OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
if OUT_ZIP.exists():
    OUT_ZIP.unlink()

total_raw = 0
with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    seen = set()
    for source, arc in entries:
        if arc in seen:
            continue
        seen.add(arc)
        zf.write(source, arc)
        total_raw += source.stat().st_size

size = OUT_ZIP.stat().st_size
report = {
    "zip": str(OUT_ZIP),
    "zip_bytes": size,
    "zip_mb": round(size / 1e6, 1),
    "raw_bytes": total_raw,
    "raw_mb": round(total_raw / 1e6, 1),
    "file_count": len(entries),
    "under_hard_max": size < HARD_MAX,
    "near_target": size <= TARGET,
}
print(json.dumps(report, indent=2))
if size >= HARD_MAX:
    OUT_ZIP.unlink()
    raise SystemExit(f"ZIP {size} bytes exceeds hard max {HARD_MAX}")
