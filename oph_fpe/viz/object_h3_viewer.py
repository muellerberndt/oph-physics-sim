from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_object_h3_bulk_viewer(
    run_dir: Path,
    out_path: Path | None = None,
    *,
    max_objects: int = 512,
) -> dict[str, Any]:
    """Write a standalone object-H3 bulk viewer from exported object receipts."""

    run_path = Path(run_dir)
    destination = out_path or (run_path / "plots" / "object_h3_bulk_viewer.html")
    destination.parent.mkdir(parents=True, exist_ok=True)
    objects, source_path = _load_object_rows(run_path, max_objects=max_objects)
    object_report = _read_json(run_path / "observer_chart_object_h3_lineage_report.json") or _read_json(
        run_path / "observer_chart_object_h3_report.json"
    )
    bulk_report = _read_json(run_path / "bulk_proof_certificate_report.json")
    observer_overlap_links = _observer_overlap_links(objects)
    summary = {
        "mode": "object_h3_bulk_viewer",
        "viewer_path": str(destination),
        "run_path": str(run_path),
        "source_path": str(source_path) if source_path is not None else None,
        "object_count": len(objects),
        "reported_object_count": object_report.get("object_count"),
        "reported_localized_object_count": object_report.get("localized_object_count"),
        "reported_localized_not_boundary_object_count": object_report.get("localized_not_boundary_object_count"),
        "theorem_assisted_h3_bulk": bool(
            object_report.get("THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT", False)
            or bulk_report.get("bulk_3d_established_theorem_assisted", False)
        ),
        "observer_chart_bulk_population_receipt": bool(
            object_report.get("observer_chart_bulk_population_receipt", False)
        ),
        "strict_neutral_bulk": bool(bulk_report.get("strict_neutral_third_person_bulk_established", False)),
        "physical_cmb_prediction": False,
        "observer_overlap_link_count": len(observer_overlap_links),
        "fundamental_operation": (
            "Overlapping observations by observers. Object packets are consensus groupings of record/support "
            "events seen by overlapping observer samples; the H3 chart is a derived representation of those "
            "overlap-stabilized packets."
        ),
        "dot_semantics": (
            "Each rendered dot is an observer-consensus object/component exported from h3_objects.csv. "
            "The H3 spatial point is a derived chart coordinate from modular-response/record-family "
            "incidence, not a fundamental particle position."
        ),
        "color_encoding": (
            "Cyan/blue means lower h3_compactness for that object packet; amber means higher h3_compactness. "
            "Dot size scales with observer_count."
        ),
        "view_panels": [
            "H3 bulk chart: derived 3D spatial object cloud",
            "Observer overlap substrate: sampled observer IDs co-observing object packets",
            "Boundary shadow: normalized S2-like directions seen at the observer screen",
            "Hyperboloid readout: H3 embedding time versus spatial radius",
            "Record/support substrate: record signature, observer support, and component structure before geometric interpretation",
        ],
        "claim_boundary": (
            "Standalone object-H3 visualization. It renders exported observer-object H3 spatial points "
            "and receipt metadata only; it does not promote theorem-assisted H3 previews to strict "
            "neutral third-person bulk or physical cosmology."
        ),
    }
    destination.write_text(_html(summary, objects, observer_overlap_links), encoding="utf-8")
    (destination.parent / "object_h3_bulk_viewer_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def _load_object_rows(run_path: Path, *, max_objects: int) -> tuple[list[dict[str, Any]], Path | None]:
    csv_path = run_path / "h3_objects.csv"
    if csv_path.exists():
        objects: list[dict[str, Any]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                h3_point = _parse_hyperboloid(row.get("h3_point"))
                point = _parse_point(row.get("h3_spatial_point")) or _spatial_from_hyperboloid(h3_point)
                if point is None:
                    continue
                objects.append(
                    {
                        "id": row.get("object_id") or f"obj_{len(objects):04d}",
                        "x": point[0],
                        "y": point[1],
                        "z": point[2],
                        "t": h3_point[0] if h3_point else None,
                        "radial_depth": _radius(point),
                        "observer_count": _optional_float(row.get("observer_count")),
                        "parent_observer_count": _optional_float(row.get("parent_observer_count")),
                        "support_size": _optional_float(row.get("support_size")),
                        "h3_compactness": _optional_float(row.get("h3_compactness")),
                        "h3_compactness_normalized": _optional_float(row.get("h3_compactness_normalized")),
                        "s2_boundary_compactness": _optional_float(row.get("s2_boundary_compactness")),
                        "s2_boundary_compactness_normalized": _optional_float(
                            row.get("s2_boundary_compactness_normalized")
                        ),
                        "record_signature": _optional_float(row.get("record_signature")),
                        "component_index": _optional_float(row.get("component_index")),
                        "family_mode": row.get("family_mode"),
                        "record_family_id": row.get("record_family_id"),
                        "mean_observer_key_weight": _optional_float(row.get("mean_observer_key_weight")),
                        "observer_ids_sample": _parse_json_list(row.get("observer_ids_sample")),
                    }
                )
                if len(objects) >= max_objects:
                    break
        return objects, csv_path
    for name in ("observer_chart_object_h3_lineage_report.json", "observer_chart_object_h3_report.json"):
        report_path = run_path / name
        report = _read_json(report_path)
        rows = report.get("sample_objects")
        if not isinstance(rows, list):
            continue
        objects = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            h3_point = _parse_hyperboloid(row.get("h3_point"))
            point = _parse_point(row.get("h3_spatial_point")) or _spatial_from_hyperboloid(h3_point)
            if point is None:
                continue
            objects.append(
                {
                    "id": row.get("object_id") or f"obj_{len(objects):04d}",
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                    "t": h3_point[0] if h3_point else None,
                    "radial_depth": _radius(point),
                    "observer_count": _optional_float(row.get("observer_count")),
                    "parent_observer_count": _optional_float(row.get("parent_observer_count")),
                    "support_size": _optional_float(row.get("support_size")),
                    "h3_compactness": _optional_float(row.get("h3_compactness")),
                    "h3_compactness_normalized": _optional_float(row.get("h3_compactness_normalized")),
                    "s2_boundary_compactness": _optional_float(row.get("s2_boundary_compactness")),
                    "s2_boundary_compactness_normalized": _optional_float(row.get("s2_boundary_compactness_normalized")),
                    "record_signature": _optional_float(row.get("record_signature")),
                    "component_index": _optional_float(row.get("component_index")),
                    "family_mode": row.get("family_mode"),
                    "record_family_id": row.get("record_family_id"),
                    "mean_observer_key_weight": _optional_float(row.get("mean_observer_key_weight")),
                    "observer_ids_sample": _parse_json_list(row.get("observer_ids_sample")),
                }
            )
            if len(objects) >= max_objects:
                break
        if objects:
            return objects, report_path
    return [], None


def _parse_point(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def _parse_hyperboloid(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list):
        return None
    if len(value) >= 4:
        try:
            return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except (TypeError, ValueError):
            return None
    return None


def _spatial_from_hyperboloid(value: list[float] | None) -> list[float] | None:
    if value is None or len(value) < 4:
        return None
    return [value[1], value[2], value[3]]


def _radius(point: list[float]) -> float:
    return float((point[0] ** 2 + point[1] ** 2 + point[2] ** 2) ** 0.5)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _observer_overlap_links(objects: list[dict[str, Any]], *, max_links: int = 320) -> list[dict[str, Any]]:
    observer_sets: list[set[str]] = []
    for obj in objects:
        observer_sets.append({str(value) for value in obj.get("observer_ids_sample") or []})
    links: list[dict[str, Any]] = []
    for i, left in enumerate(observer_sets):
        if not left:
            continue
        for j in range(i + 1, len(observer_sets)):
            right = observer_sets[j]
            if not right:
                continue
            overlap = len(left.intersection(right))
            if overlap <= 0:
                continue
            denom = max(1, min(len(left), len(right)))
            links.append(
                {
                    "source": i,
                    "target": j,
                    "overlap": overlap,
                    "overlap_fraction": overlap / denom,
                }
            )
    links.sort(key=lambda row: (row["overlap"], row["overlap_fraction"]), reverse=True)
    return links[:max_links]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _html(summary: dict[str, Any], objects: list[dict[str, Any]], overlap_links: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {"summary": summary, "objects": objects, "overlapLinks": overlap_links},
        separators=(",", ":"),
        default=str,
    )
    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OPH Object-H3 Bulk Viewer</title>
<style>
:root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; min-height: 100vh; background: #101216; color: #eef2f8; display: grid; grid-template-rows: auto 1fr; }
header { padding: 14px 18px; border-bottom: 1px solid #2b3340; display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }
h1 { margin: 0; font-size: 18px; font-weight: 650; letter-spacing: 0; }
.metric { color: #aeb8c8; font-size: 13px; }
.metric b { color: #ffffff; font-weight: 650; }
main { padding: 14px; display: grid; grid-template-columns: 1.25fr 1fr; gap: 12px; }
.panel { min-height: 260px; border: 1px solid #2b3340; border-radius: 8px; background: #151a21; display: grid; grid-template-rows: auto 1fr auto; overflow: hidden; }
.wide { min-height: 420px; grid-row: span 2; }
.panel h2 { margin: 0; padding: 10px 12px 0; font-size: 14px; font-weight: 650; color: #f2f5fa; letter-spacing: 0; }
.panel p { margin: 0; padding: 0 12px 10px; color: #aeb8c8; font-size: 12px; line-height: 1.4; }
canvas { width: 100%; height: 100%; min-height: 190px; display: block; background: #12161d; }
.explainer { grid-column: 1 / -1; border: 1px solid #2b3340; border-radius: 8px; padding: 12px 14px; color: #c5cfdc; background: #151a21; font-size: 13px; line-height: 1.45; }
.legend { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
.swatch { display: inline-flex; align-items: center; gap: 6px; color: #aeb8c8; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } .wide { grid-row: auto; } .explainer { grid-column: auto; } }
</style>
</head>
<body>
<header>
  <h1>OPH Object-H3 Bulk Viewer</h1>
  <div class="metric">objects <b id="object-count">0</b></div>
  <div class="metric">observer overlaps <b id="overlap-count">0</b></div>
  <div class="metric">theorem-assisted H3 <b id="h3-gate">false</b></div>
  <div class="metric">strict neutral bulk <b id="neutral-gate">false</b></div>
  <div class="metric">physical CMB <b id="cmb-gate">false</b></div>
</header>
<main>
  <section class="panel wide">
    <h2>Observer overlap substrate</h2>
    <canvas id="overlap"></canvas>
    <p>Most fundamental panel here: links connect object packets with shared sampled observers. Consensus objects are built from these overlapping observations before any H3 chart is assigned.</p>
  </section>
  <section class="panel">
    <h2>H3 bulk chart</h2>
    <canvas id="bulk"></canvas>
    <p>Each dot is an observer-consensus object/component rendered at its derived H3 spatial coordinate.</p>
  </section>
  <section class="panel">
    <h2>Boundary shadow</h2>
    <canvas id="boundary"></canvas>
    <p>Normalized spatial directions show the S2-like screen shadow that the H3 object packet projects toward.</p>
  </section>
  <section class="panel">
    <h2>Hyperboloid readout</h2>
    <canvas id="hyperboloid"></canvas>
    <p>Embedding time versus spatial radius checks that the displayed cloud is an H3 chart readout, not raw screen data.</p>
  </section>
  <section class="panel">
    <h2>Record/support substrate</h2>
    <canvas id="support"></canvas>
    <p>Record signature and observer support expose the record-family substrate underneath the geometric visualization.</p>
  </section>
  <section class="explainer">
    <div><b>Fundamental operation:</b> Overlapping observations by observers. Object packets are consensus groupings of record/support events seen by overlapping observer samples; the H3 chart is derived after that overlap step.</div>
    <div><b>Dot semantics:</b> Each rendered dot is an observer-consensus object/component exported from h3_objects.csv, not a fundamental particle position.</div>
    <b>What are the blue dots?</b>
    The dots are observer-consensus object packets. Cyan/blue means lower H3 compactness for that packet; amber means higher H3 compactness. Dot size scales with observer count. This viewer deliberately shows the causal order: observer overlap substrate -> record/support packets -> boundary shadow/hyperboloid readout -> derived H3 bulk chart.
    <div class="legend">
      <span class="swatch"><span class="dot" style="background:#50e1d2"></span>lower H3 compactness</span>
      <span class="swatch"><span class="dot" style="background:#d2aa5a"></span>higher H3 compactness</span>
      <span class="swatch"><span class="dot" style="background:#8aa7ff"></span>observer-overlap link</span>
    </div>
  </section>
</main>
<script>
const DATA = __PAYLOAD__;
const canvases = {
  bulk: document.getElementById('bulk'),
  boundary: document.getElementById('boundary'),
  hyperboloid: document.getElementById('hyperboloid'),
  support: document.getElementById('support'),
  overlap: document.getElementById('overlap')
};
document.getElementById('object-count').textContent = DATA.summary.object_count;
document.getElementById('overlap-count').textContent = DATA.summary.observer_overlap_link_count;
document.getElementById('h3-gate').textContent = String(DATA.summary.theorem_assisted_h3_bulk);
document.getElementById('neutral-gate').textContent = String(DATA.summary.strict_neutral_bulk);
document.getElementById('cmb-gate').textContent = String(DATA.summary.physical_cmb_prediction);
function resize(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, rect.width, rect.height];
}
function colorFor(v) {
  const t = Math.max(0, Math.min(1, v || 0));
  const r = Math.round(80 + 130 * t);
  const g = Math.round(170 + 55 * (1 - t));
  const b = Math.round(210 - 120 * t);
  return `rgb(${{r}},${{g}},${{b}})`;
}
function clear(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#12161d';
  ctx.fillRect(0, 0, w, h);
}
function drawBulk(time) {
  const [ctx, w, h] = resize(canvases.bulk);
  clear(ctx, w, h);
  const points = DATA.objects || [];
  const angle = time * 0.00008;
  const ca = Math.cos(angle), sa = Math.sin(angle);
  const scale = Math.min(w, h) * 0.23;
  ctx.strokeStyle = '#293340';
  ctx.lineWidth = 1;
  for (let r = 1; r <= 3; r++) {
    ctx.beginPath();
    ctx.ellipse(w / 2, h / 2, scale * r * 0.55, scale * r * 0.22, 0, 0, Math.PI * 2);
    ctx.stroke();
  }
  const projected = points.map((p) => {
    const x = p.x * ca - p.z * sa;
    const z = p.x * sa + p.z * ca;
    const y = p.y;
    const depth = 1 / (1 + Math.max(-0.65, Math.min(0.9, z * 0.12)));
    return {...p, sx: w / 2 + x * scale * depth, sy: h / 2 - y * scale * depth, depth};
  }).sort((a, b) => a.depth - b.depth);
  for (const p of projected) {
    const observers = Math.max(1, Number(p.observer_count || 1));
    const radius = Math.max(2.2, Math.min(8, 1.5 + Math.sqrt(observers) * 0.8)) * p.depth;
    ctx.beginPath();
    ctx.fillStyle = colorFor(p.h3_compactness);
    ctx.globalAlpha = Math.max(0.45, Math.min(0.95, 0.52 + p.depth * 0.22));
    ctx.arc(p.sx, p.sy, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}
function drawBoundary() {
  const [ctx, w, h] = resize(canvases.boundary);
  clear(ctx, w, h);
  ctx.strokeStyle = '#334050';
  ctx.beginPath();
  ctx.arc(w / 2, h / 2, Math.min(w, h) * 0.38, 0, Math.PI * 2);
  ctx.stroke();
  for (const p of DATA.objects || []) {
    const r = Math.max(1e-9, Math.sqrt(p.x*p.x + p.y*p.y + p.z*p.z));
    const sx = w / 2 + (p.x / r) * Math.min(w, h) * 0.36;
    const sy = h / 2 - (p.y / r) * Math.min(w, h) * 0.36;
    ctx.fillStyle = colorFor(p.s2_boundary_compactness_normalized ?? p.h3_compactness);
    ctx.globalAlpha = 0.72;
    ctx.beginPath();
    ctx.arc(sx, sy, 3.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}
function extent(values, fallbackMax = 1) {
  const finite = values.filter((v) => Number.isFinite(v));
  if (!finite.length) return [0, fallbackMax];
  const lo = Math.min(...finite), hi = Math.max(...finite);
  return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
}
function drawHyperboloid() {
  const [ctx, w, h] = resize(canvases.hyperboloid);
  clear(ctx, w, h);
  const points = DATA.objects || [];
  const [r0, r1] = extent(points.map((p) => Number(p.radial_depth)), 1);
  const [t0, t1] = extent(points.map((p) => Number(p.t)), 2);
  ctx.strokeStyle = '#334050';
  ctx.beginPath();
  ctx.moveTo(42, h - 32); ctx.lineTo(w - 16, h - 32); ctx.moveTo(42, 16); ctx.lineTo(42, h - 32);
  ctx.stroke();
  for (const p of points) {
    const rr = Number(p.radial_depth), tt = Number(p.t);
    if (!Number.isFinite(rr) || !Number.isFinite(tt)) continue;
    const sx = 42 + ((rr - r0) / (r1 - r0)) * Math.max(1, w - 68);
    const sy = h - 32 - ((tt - t0) / (t1 - t0)) * Math.max(1, h - 52);
    ctx.fillStyle = colorFor(p.h3_compactness);
    ctx.beginPath();
    ctx.arc(sx, sy, 3.4, 0, Math.PI * 2);
    ctx.fill();
  }
}
function drawSupport() {
  const [ctx, w, h] = resize(canvases.support);
  clear(ctx, w, h);
  const points = DATA.objects || [];
  const [sig0, sig1] = extent(points.map((p) => Number(p.record_signature)), 1);
  const [obs0, obs1] = extent(points.map((p) => Number(p.observer_count)), 1);
  ctx.strokeStyle = '#334050';
  ctx.beginPath();
  ctx.moveTo(42, h - 32); ctx.lineTo(w - 16, h - 32); ctx.moveTo(42, 16); ctx.lineTo(42, h - 32);
  ctx.stroke();
  for (const p of points) {
    const sig = Number(p.record_signature), obs = Number(p.observer_count || 0);
    if (!Number.isFinite(sig)) continue;
    const sx = 42 + ((sig - sig0) / (sig1 - sig0)) * Math.max(1, w - 68);
    const sy = h - 32 - ((obs - obs0) / (obs1 - obs0)) * Math.max(1, h - 52);
    const support = Math.sqrt(Math.max(1, Number(p.support_size || 1)));
    ctx.fillStyle = colorFor((Number(p.component_index || 0) % 6) / 6);
    ctx.globalAlpha = 0.72;
    ctx.beginPath();
    ctx.arc(sx, sy, Math.max(2.5, Math.min(6.5, support * 0.22)), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}
function drawOverlap() {
  const [ctx, w, h] = resize(canvases.overlap);
  clear(ctx, w, h);
  const points = DATA.objects || [];
  const links = DATA.overlapLinks || [];
  const n = points.length;
  if (!n) return;
  const cx = w / 2, cy = h / 2, radius = Math.min(w, h) * 0.38;
  const pos = points.map((p, i) => {
    const a = -Math.PI / 2 + (i / Math.max(1, n)) * Math.PI * 2;
    return {x: cx + Math.cos(a) * radius, y: cy + Math.sin(a) * radius};
  });
  for (const link of links.slice(0, 220)) {
    const a = pos[link.source], b = pos[link.target];
    if (!a || !b) continue;
    ctx.strokeStyle = '#8aa7ff';
    ctx.globalAlpha = Math.min(0.5, 0.08 + Number(link.overlap_fraction || 0) * 0.65);
    ctx.lineWidth = Math.max(0.4, Math.min(3, Number(link.overlap || 1) * 0.55));
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.quadraticCurveTo(cx, cy, b.x, b.y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  for (let i = 0; i < n; i++) {
    const p = points[i], xy = pos[i];
    const observers = Math.max(1, Number(p.observer_count || 1));
    ctx.fillStyle = colorFor(p.h3_compactness);
    ctx.beginPath();
    ctx.arc(xy.x, xy.y, Math.max(2.2, Math.min(7, 1.4 + Math.sqrt(observers) * 0.7)), 0, Math.PI * 2);
    ctx.fill();
  }
}
function draw(time) {
  drawOverlap();
  drawBulk(time);
  drawBoundary();
  drawHyperboloid();
  drawSupport();
  requestAnimationFrame(draw);
}
window.addEventListener('resize', () => draw(performance.now()));
requestAnimationFrame(draw);
</script>
</body>
</html>
"""
    return template.replace("__PAYLOAD__", payload)
