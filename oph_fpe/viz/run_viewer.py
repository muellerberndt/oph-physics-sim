from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.core.graph import fibonacci_sphere_points


def write_run_viewer(
    run_path: Path,
    out_path: Path | None = None,
    *,
    max_screen_points: int = 6000,
    seed: int = 1,
) -> dict[str, Any]:
    run_path = Path(run_path)
    if not run_path.exists():
        raise FileNotFoundError(run_path)
    manifest = _read_json(run_path / "manifest.json")
    status = _read_json(run_path / "emergence_status_report.json")
    holonomy = _read_json(run_path / "array_holonomy_report.json")
    timeline = _read_json(run_path / "defect_timeline_report.json")
    defect_interaction = _read_json(run_path / "defect_interaction_report.json")
    particle_likeness = _read_json(run_path / "particle_likeness_report.json")
    record_h3 = _read_json(run_path / "record_family_h3_report.json")
    defect_h3 = _read_json(run_path / "defect_cluster_h3_report.json")
    modular_response_h3 = _read_json(run_path / "modular_response_h3_report.json")
    defect_h3_worldlines = _read_json(run_path / "defect_h3_worldlines_report.json")
    cl_report = _read_json(run_path / "cl_comparison_report.json")
    cmb_lite = _read_json(run_path / "cmb_lite_comparison_report.json")
    observer_views = _read_jsonl(run_path / "observer_views.jsonl", limit=256)
    trace = _read_trace(run_path / "mismatch_trace.csv")
    points, screen_field_name, screen_values = _screen_points_and_field(run_path, manifest, max_screen_points, seed)
    clusters = _cluster_payload(holonomy, timeline, max_count=512)
    h3_points = _h3_points(record_h3, defect_h3, modular_response_h3)
    h3_worldlines = _h3_worldline_payload(defect_h3_worldlines)

    html_text = _render_html(
        run_path=run_path,
        manifest=manifest,
        status=status,
        points=points,
        screen_field_name=screen_field_name,
        screen_values=screen_values,
        observer_views=observer_views,
        clusters=clusters,
        h3_points=h3_points,
        h3_worldlines=h3_worldlines,
        defect_interaction=defect_interaction,
        particle_likeness=particle_likeness,
        trace=trace,
        cl_report=cl_report,
        cmb_lite=cmb_lite,
    )
    destination = out_path or (run_path / "plots" / "oph_realtime_viewer.html")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html_text, encoding="utf-8")
    summary = {
        "viewer_path": str(destination),
        "run_path": str(run_path),
        "screen_point_count": int(points.shape[0]),
        "defect_cluster_count": len(clusters.get("points", [])),
        "defect_timeline_snapshots": len(clusters.get("snapshots", [])),
        "persistent_defect_worldlines": int(timeline.get("persistent_worldline_count", 0)) if timeline else 0,
        "h3_point_count": len(h3_points),
        "h3_worldline_count": len(h3_worldlines),
        "defect_screen_transport_proxy_count": int(defect_interaction.get("screen_transport_proxy_count", 0)) if defect_interaction else 0,
        "defect_fusion_candidate_count": int(defect_interaction.get("fusion_candidate_count", 0)) if defect_interaction else 0,
        "particle_likeness_worldline_count": int(particle_likeness.get("worldline_count", 0)) if particle_likeness else 0,
        "particle_like_count": int(particle_likeness.get("particle_like_count", 0)) if particle_likeness else 0,
        "bulk_3d_established": bool(status.get("bulk_3d_established", False)),
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "standalone receipt viewer for OPH screen, observer readouts, H3 support fits, defect "
            "clusters, and C_l diagnostics. It visualizes current receipts and gates; it does not "
            "upgrade failed bulk or particle gates."
        ),
    }
    (destination.parent / "oph_realtime_viewer_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def _render_html(
    *,
    run_path: Path,
    manifest: dict[str, Any],
    status: dict[str, Any],
    points: np.ndarray,
    screen_field_name: str,
    screen_values: np.ndarray,
    observer_views: list[dict[str, Any]],
    clusters: dict[str, Any],
    h3_points: list[dict[str, Any]],
    h3_worldlines: list[dict[str, Any]],
    defect_interaction: dict[str, Any],
    particle_likeness: dict[str, Any],
    trace: list[dict[str, float]],
    cl_report: dict[str, Any],
    cmb_lite: dict[str, Any],
) -> str:
    payload = {
        "runPath": str(run_path),
        "runId": manifest.get("run_id", run_path.name),
        "status": {
            "edgeSector": bool(status.get("edge_sector_heat_kernel_receipt", False)),
            "centralRecord": bool(status.get("central_record_born_receipt", False)),
            "checkpoint": bool(status.get("observer_checkpoint_restoration_receipt", False)),
            "supportVisibleLorentz": bool(status.get("support_visible_lorentz_3p1_kinematics_receipt")),
            "h3Chart": bool(status.get("conformal_h3_spatial_chart_receipt")),
            "recordBulk": bool(status.get("record_populated_h3_spatial_receipt")),
            "bulk3d": bool(status.get("bulk_3d_established")),
            "defectH3": bool(status.get("defect_cluster_h3_support_receipt")),
            "particleDiagnostic": bool(status.get("particle_likeness_diagnostic_written", False)),
            "particle": bool(status.get("particle_matter_receipt", False)),
        },
        "screenField": screen_field_name,
        "screen": _screen_payload(points, screen_values),
        "observers": _observer_payload(observer_views),
        "clusters": clusters,
        "h3": h3_points,
        "h3Worldlines": h3_worldlines,
        "defectInteraction": _defect_interaction_payload(defect_interaction),
        "particleLikeness": _particle_likeness_payload(particle_likeness),
        "trace": trace,
        "cl": _cl_payload(cl_report),
        "cmbLite": _cmb_lite_payload(cmb_lite),
        "claimBoundary": (
            "Diagnostic viewer only: the S2 panel is the support-visible screen regulator, the H3 "
            "panel is the canonical conformal chart plus support-fit samples, and C_l is a gated "
            "screen proxy unless the physical CMB gate passes."
        ),
    }
    data = json.dumps(payload, separators=(",", ":"), default=str)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OPH-FPE Receipt Viewer</title>
<style>
body {{ margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#101214; color:#e9edf0; }}
header {{ padding:16px 20px; border-bottom:1px solid #2b3035; background:#15191d; }}
h1 {{ margin:0 0 8px; font-size:20px; font-weight:650; }}
.sub {{ color:#a8b0b8; font-size:13px; }}
.gates {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }}
.gate {{ padding:5px 8px; border-radius:6px; font-size:12px; background:#2a2f35; color:#c7d0d8; }}
.pass {{ background:#173d2a; color:#baf0ce; }}
.fail {{ background:#472124; color:#ffcbc9; }}
main {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; padding:12px; }}
section {{ min-height:320px; background:#171b20; border:1px solid #2b3035; border-radius:8px; overflow:hidden; }}
section h2 {{ margin:0; padding:10px 12px; font-size:14px; border-bottom:1px solid #2b3035; color:#dbe4ea; }}
svg {{ width:100%; height:320px; display:block; background:#111417; }}
.note {{ padding:8px 12px 12px; color:#9da7af; font-size:12px; line-height:1.4; }}
.wide {{ grid-column:1 / -1; }}
@media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} .wide {{ grid-column:auto; }} }}
</style>
</head>
<body>
<header>
  <h1>OPH-FPE Receipt Viewer</h1>
  <div class="sub" id="runline"></div>
  <div class="gates" id="gates"></div>
</header>
<main>
  <section><h2>Support-Visible S2 Lattice</h2><svg id="screen"></svg><div class="note" id="screenNote"></div></section>
  <section><h2>Observer Perspectives And Defect Clusters</h2><svg id="observer"></svg><input id="timeSlider" type="range" min="0" max="0" value="0" style="width:calc(100% - 24px); margin:8px 12px 0;"><div class="note" id="observerNote"></div></section>
  <section><h2>Canonical H3 Support-Fit Samples</h2><svg id="h3"></svg><div class="note" id="h3Note"></div></section>
  <section><h2>Repair / Record Time Trace</h2><svg id="trace"></svg><div class="note">Discrete repair time from the run mismatch trace.</div></section>
  <section class="wide"><h2>Freezeout C_l Proxy And CMB-Lite Shape Comparison</h2><svg id="cl"></svg><div class="note" id="clNote"></div></section>
</main>
<script>
const DATA = {data};
const NS = "http://www.w3.org/2000/svg";
function el(name, attrs={{}}) {{ const e=document.createElementNS(NS,name); for (const [k,v] of Object.entries(attrs)) e.setAttribute(k,v); return e; }}
function clear(svg) {{ while(svg.firstChild) svg.removeChild(svg.firstChild); }}
function dims(svg) {{ const r=svg.getBoundingClientRect(); return [Math.max(320,r.width), Math.max(240,r.height || 320)]; }}
function color(v) {{ const x=Math.max(0, Math.min(1, v)); const r=Math.round(40+210*x); const g=Math.round(170-80*x); const b=Math.round(230-180*x); return `rgb(${{r}},${{g}},${{b}})`; }}
function projectS2(p,w,h) {{ const lon=Math.atan2(p[1],p[0]); const lat=Math.asin(Math.max(-1,Math.min(1,p[2]))); return [w*(0.5+lon/(2*Math.PI)), h*(0.5-lat/Math.PI)]; }}
function projectXY(p,w,h,pad=24) {{ return [pad+(w-2*pad)*(0.5+0.28*p[0]), pad+(h-2*pad)*(0.5-0.28*p[1])]; }}
function drawScatter(svg, pts, values, projector, radius=1.8) {{ clear(svg); const [w,h]=dims(svg); svg.setAttribute("viewBox",`0 0 ${{w}} ${{h}}`);
  svg.appendChild(el("rect",{{x:0,y:0,width:w,height:h,fill:"#111417"}}));
  pts.forEach((p,i)=>{{ const [x,y]=projector(p,w,h); svg.appendChild(el("circle",{{cx:x,cy:y,r:radius,fill:color(values[i] ?? 0.5),opacity:0.78}})); }});
}}
function drawH3Paths(svg) {{ const [w,h]=dims(svg); (DATA.h3Worldlines || []).forEach((line,idx)=>{{ const pts=line.points || []; if(pts.length < 2) return; const d=pts.map((p,i)=>{{ const [x,y]=projectXY(p,w,h); return `${{i?'L':'M'}}${{x.toFixed(1)}} ${{y.toFixed(1)}}`; }}).join(" "); svg.appendChild(el("path",{{d,fill:"none",stroke:idx%2?"#ff8a65":"#66d9ef","stroke-width":1.6,opacity:0.72}})); }}); }}
function drawLine(svg, rows, key, stroke) {{ if(!rows.length) return; const [w,h]=dims(svg); svg.setAttribute("viewBox",`0 0 ${{w}} ${{h}}`);
  const xs=rows.map(r=>r.cycle ?? r.ell); const ys=rows.map(r=>r[key] ?? 0); const xmin=Math.min(...xs), xmax=Math.max(...xs); const ymin=Math.min(...ys), ymax=Math.max(...ys);
  const path=rows.map((r,i)=>{{ const x=30+(w-50)*((xs[i]-xmin)/Math.max(xmax-xmin,1e-9)); const y=20+(h-45)*(1-(ys[i]-ymin)/Math.max(ymax-ymin,1e-9)); return `${{i?'L':'M'}}${{x.toFixed(1)}} ${{y.toFixed(1)}}`; }}).join(" ");
  svg.appendChild(el("path",{{d:path,fill:"none",stroke, "stroke-width":2}}));
}}
function drawObserverSnapshot(index=0) {{
  const snapshot = DATA.clusters.snapshots && DATA.clusters.snapshots.length ? DATA.clusters.snapshots[index] : null;
  const clusterPoints = snapshot ? snapshot.points : DATA.clusters.points;
  const clusterValues = snapshot ? snapshot.values : DATA.clusters.values;
  drawScatter(document.getElementById("observer"), DATA.observers.points.concat(clusterPoints), DATA.observers.values.concat(clusterValues), projectS2, 3.2);
  const cycle = snapshot ? ` cycle ${{snapshot.cycle}}` : " final";
  document.getElementById("observerNote").textContent = `${{DATA.observers.points.length}} observer readouts and ${{clusterPoints.length}} defect-cluster centroids on the screen at${{cycle}}. Persistent worldline precursors: ${{DATA.clusters.persistentWorldlines}}. Screen transport proxies: ${{DATA.defectInteraction.transportProxyCount}}. Particle-like defects: ${{DATA.particleLikeness.particleLikeCount}} / ${{DATA.particleLikeness.worldlineCount}}.`;
}}
function drawTrace() {{ const svg=document.getElementById("trace"); clear(svg); const [w,h]=dims(svg); svg.setAttribute("viewBox",`0 0 ${{w}} ${{h}}`); svg.appendChild(el("rect",{{x:0,y:0,width:w,height:h,fill:"#111417"}})); drawLine(svg, DATA.trace, "phi", "#66d9ef"); drawLine(svg, DATA.trace, "committed_fraction", "#b5f26d"); if(DATA.clusters.timelineTrace) drawLine(svg, DATA.clusters.timelineTrace, "cluster_count", "#ff8a65"); }}
function drawCL() {{ const svg=document.getElementById("cl"); clear(svg); const [w,h]=dims(svg); svg.setAttribute("viewBox",`0 0 ${{w}} ${{h}}`); svg.appendChild(el("rect",{{x:0,y:0,width:w,height:h,fill:"#111417"}})); drawLine(svg, DATA.cl.spectrum || [], "D_ell", "#f6c85f"); if(DATA.cl.control) drawLine(svg, DATA.cl.control, "D_ell", "#8f99a3"); }}
function gate(label, ok) {{ return `<span class="gate ${{ok?'pass':'fail'}}">${{label}}: ${{ok?'pass':'open'}}</span>`; }}
function init() {{
  document.getElementById("runline").textContent = `${{DATA.runId}} · ${{DATA.runPath}}`;
  document.getElementById("gates").innerHTML = gate("edge sector", DATA.status.edgeSector)+gate("central records", DATA.status.centralRecord)+gate("checkpoint", DATA.status.checkpoint)+gate("BW/KMS Lorentz", DATA.status.supportVisibleLorentz)+gate("H3 chart", DATA.status.h3Chart)+gate("record bulk", DATA.status.recordBulk)+gate("3D bulk", DATA.status.bulk3d)+gate("defect H3", DATA.status.defectH3)+gate("particle diag", DATA.status.particleDiagnostic)+gate("particles", DATA.status.particle);
  drawScatter(document.getElementById("screen"), DATA.screen.points, DATA.screen.values, projectS2, DATA.screen.points.length>3000?1.25:1.8);
  document.getElementById("screenNote").textContent = `Color field: ${{DATA.screenField}}. This is the S2 observer screen regulator, not an initialized 3D bulk lattice.`;
  const slider = document.getElementById("timeSlider");
  const snapshotCount = DATA.clusters.snapshots ? DATA.clusters.snapshots.length : 0;
  slider.max = Math.max(0, snapshotCount - 1);
  slider.style.display = snapshotCount > 1 ? "block" : "none";
  slider.oninput = () => drawObserverSnapshot(Number(slider.value));
  drawObserverSnapshot(0);
  drawScatter(document.getElementById("h3"), DATA.h3.points, DATA.h3.values, projectXY, 4);
  drawH3Paths(document.getElementById("h3"));
  document.getElementById("h3Note").textContent = `${{DATA.h3.points.length}} fitted H3 support samples and ${{DATA.h3Worldlines.length}} fitted defect-worldline paths. Localized/persistent/sector-stable screen defects: ${{DATA.particleLikeness.localizedCount}}/${{DATA.particleLikeness.persistentCount}}/${{DATA.particleLikeness.sectorStableCount}}. Fusion candidates: ${{DATA.defectInteraction.fusionCandidateCount}}. This panel is a receipt visualization, not a populated-bulk claim unless the gates pass.`;
  drawTrace(); drawCL();
  const cmp = DATA.cmbLite.best ? ` CMB-lite best shape field: ${{DATA.cmbLite.best}}; physical prediction: false.` : "";
  document.getElementById("clNote").textContent = `Gated screen C_l proxy. Yellow target spectrum, gray first control if present.${{cmp}}`;
}}
init();
</script>
</body>
</html>
"""


def _screen_points_and_field(
    run_path: Path,
    manifest: dict[str, Any],
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, str, np.ndarray]:
    npz_path = run_path / "freezeout_fields.npz"
    if npz_path.exists():
        with np.load(npz_path) as data:
            points = np.asarray(data["points"], dtype=float)
            field_name = "record_signature" if "record_signature" in data.files else next(
                (name for name in data.files if name not in {"points", "cell_area_planck", "cell_entropy"}),
                "uniform",
            )
            values = np.asarray(data[field_name], dtype=float) if field_name in data.files else np.zeros(points.shape[0])
    else:
        count = int(manifest.get("patch_count", 4096))
        points = fibonacci_sphere_points(count)
        field_name = "uniform"
        values = np.zeros(count, dtype=float)
    if points.shape[0] > int(max_points):
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(points.shape[0], size=int(max_points), replace=False))
        points = points[indices]
        values = values[indices]
    return points, field_name, _normalize(values)


def _screen_payload(points: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    return {
        "points": [[float(x) for x in row] for row in points],
        "values": [float(value) for value in values],
    }


def _observer_payload(observer_views: list[dict[str, Any]]) -> dict[str, Any]:
    points = [row.get("axis", [0.0, 0.0, 1.0]) for row in observer_views if row.get("axis")]
    values = _normalize(np.asarray([row.get("visible_signature_entropy", 0.0) for row in observer_views], dtype=float))
    return {
        "points": [[float(x) for x in row] for row in points],
        "values": [float(value) for value in values[: len(points)]],
    }


def _cluster_payload(holonomy: dict[str, Any], timeline: dict[str, Any], *, max_count: int) -> dict[str, Any]:
    final_clusters = _cluster_points(holonomy, max_count=max_count)
    snapshots = []
    for snapshot in list(timeline.get("snapshots", []))[:64] if timeline else []:
        rows = list(snapshot.get("clusters", []))[: int(max_count)]
        points = [row["point"] for row in _cluster_points({"clusters": rows}, max_count=max_count)]
        values = [row["value"] for row in _cluster_points({"clusters": rows}, max_count=max_count)]
        snapshots.append(
            {
                "cycle": int(snapshot.get("cycle", 0)),
                "cluster_count": int(snapshot.get("cluster_count", len(rows))),
                "points": points,
                "values": values,
            }
        )
    return {
        "points": [row["point"] for row in final_clusters],
        "values": [row["value"] for row in final_clusters],
        "snapshots": snapshots,
        "timelineTrace": [
            {"cycle": int(row.get("cycle", 0)), "cluster_count": int(row.get("cluster_count", 0))}
            for row in snapshots
        ],
        "persistentWorldlines": int(timeline.get("persistent_worldline_count", 0)) if timeline else 0,
    }


def _cluster_points(holonomy: dict[str, Any], *, max_count: int) -> list[dict[str, Any]]:
    clusters = list(holonomy.get("clusters", []))[: int(max_count)] if holonomy else []
    return [
        {
            "id": str(row.get("cluster_id", index)),
            "point": [float(value) for value in row.get("centroid", [0.0, 0.0, 1.0])],
            "value": 1.0 if row.get("class") == "threecycle" else 0.75,
            "class": row.get("class"),
            "support_node_count": row.get("support_node_count"),
        }
        for index, row in enumerate(clusters)
    ]


def _h3_points(record_h3: dict[str, Any], defect_h3: dict[str, Any], modular_response_h3: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reports = [
        ("record", record_h3, 0.35, 128),
        ("defect", defect_h3, 0.95, 128),
        ("modular_response_observer", modular_response_h3, 0.65, 256),
    ]
    for label, report, value, limit in reports:
        h3_fit = report.get("h3_fit", {}) if report else {}
        points = h3_fit.get("sample_fitted_h3_points", []) or h3_fit.get("fitted_h3_points", [])[: int(limit)]
        for index, point in enumerate(points):
            if len(point) >= 4:
                rows.append({"label": label, "point": [float(point[1]), float(point[2]), float(point[3])], "value": value})
            elif len(point) >= 3:
                rows.append({"label": label, "point": [float(point[0]), float(point[1]), float(point[2])], "value": value})
    return rows


def _h3_worldline_payload(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for worldline in list(report.get("worldlines", []))[:64] if report else []:
        points = [
            [float(value) for value in event.get("h3_spatial_point", [0.0, 0.0, 0.0])]
            for event in worldline.get("events", [])
            if event.get("h3_spatial_point") is not None
        ]
        if points:
            rows.append(
                {
                    "worldline_id": worldline.get("worldline_id"),
                    "points": points,
                    "observation_count": int(worldline.get("observation_count", len(points))),
                    "h3_path_length": float(worldline.get("h3_path_length", 0.0)),
                }
            )
    return rows


def _cl_payload(cl_report: dict[str, Any]) -> dict[str, Any]:
    if not cl_report:
        return {"spectrum": [], "control": []}
    fields = cl_report.get("fields", {})
    name = "record_signature" if "record_signature" in fields else next(iter(fields), None)
    if name is None:
        return {"spectrum": [], "control": []}
    controls = cl_report.get("controls", {}).get(name, {})
    control = next((row.get("spectrum", []) for row in controls.values() if row.get("spectrum")), [])
    return {
        "field": name,
        "spectrum": fields[name].get("spectrum", []),
        "control": control,
    }


def _cmb_lite_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "best": report.get("best_shape_field") if report else None,
        "physical": bool(report.get("physical_cmb_prediction", False)) if report else False,
    }


def _particle_likeness_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "worldlineCount": int(report.get("worldline_count", 0)) if report else 0,
        "localizedCount": int(report.get("localized_count", 0)) if report else 0,
        "persistentCount": int(report.get("persistent_count", 0)) if report else 0,
        "sectorStableCount": int(report.get("sector_stable_count", 0)) if report else 0,
        "particleLikeCount": int(report.get("particle_like_count", 0)) if report else 0,
        "particleMatterReceipt": bool(report.get("particle_matter_receipt", False)) if report else False,
    }


def _defect_interaction_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "transportProxyCount": int(report.get("screen_transport_proxy_count", 0)) if report else 0,
        "fusionCandidateCount": int(report.get("fusion_candidate_count", 0)) if report else 0,
        "fusionIdentityCandidateCount": int(report.get("fusion_identity_candidate_count", 0)) if report else 0,
        "scatteringTransitionTotal": int(report.get("scattering_transition_total", 0)) if report else 0,
        "interactionProxyReceipt": bool(report.get("interaction_proxy_receipt", False)) if report else False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= int(limit):
                break
            rows.append(json.loads(line))
    return rows


def _read_trace(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                if value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            rows.append(parsed)
        return rows


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < 1e-12:
        return np.full(values.shape, 0.5, dtype=float)
    return (values - min_value) / (max_value - min_value)


def _escape(text: str) -> str:
    return html.escape(text, quote=True)
