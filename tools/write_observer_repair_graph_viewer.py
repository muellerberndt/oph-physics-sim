#!/usr/bin/env python3
"""Write a standalone observer-overlap repair graph viewer.

The input is an OPH universe timeline `visualization_payload.json`. The output is
an offline HTML file that shows observers as nodes, overlap supports as edges,
and a repair-progress slider. Edge color, width, and labels are driven by a
frame-local closure-error proxy derived from each overlap repair trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _edge_error(link: dict[str, Any], frame_index: int) -> float:
    trajectory = link.get("repairTrajectory") or []
    if trajectory:
        point = trajectory[min(frame_index, len(trajectory) - 1)]
        committed = _float(point.get("committedFraction"), 0.0)
        repair_load = _float(point.get("repairLoadMean"), 0.0)
    else:
        committed = _float(link.get("overlapCommittedFraction"), 1.0)
        repair_load = _float(link.get("overlapRepairLoadMean"), 0.0)
    signature_gap = max(0.0, 1.0 - _float(link.get("signatureSimilarity"), 1.0))
    closure_gap = max(0.0, 1.0 - committed)
    return max(0.0, min(1.0, 0.72 * closure_gap + 0.20 * repair_load + 0.08 * signature_gap))


def _pick_slice(
    observers: list[dict[str, Any]],
    links: list[dict[str, Any]],
    *,
    max_nodes: int,
    max_edges: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observer_ids = {int(row["observerId"]) for row in observers if "observerId" in row}
    valid_links = [
        link
        for link in links
        if int(link.get("source", -1)) in observer_ids and int(link.get("target", -1)) in observer_ids
    ]
    degree: Counter[int] = Counter()
    for link in valid_links:
        degree[int(link["source"])] += 1
        degree[int(link["target"])] += 1

    ranked_ids = [
        node_id
        for node_id, _ in sorted(
            degree.items(),
            key=lambda item: (-item[1], item[0]),
        )[:max_nodes]
    ]
    selected = set(ranked_ids)
    selected_nodes = [row for row in observers if int(row.get("observerId", -1)) in selected]
    selected_links = [
        link
        for link in valid_links
        if int(link["source"]) in selected and int(link["target"]) in selected
    ]
    selected_links.sort(
        key=lambda link: (
            -_float(link.get("jaccard"), 0.0),
            -_float(link.get("overlapPatchCount"), 0.0),
            int(link.get("source", 0)),
            int(link.get("target", 0)),
        )
    )
    return selected_nodes, selected_links[:max_edges]


def _viewer_payload(source_payload: dict[str, Any], *, max_nodes: int, max_edges: int) -> dict[str, Any]:
    observer_time = source_payload.get("observerModularTime") or {}
    observers = observer_time.get("observers") or []
    links = observer_time.get("overlapLinks") or []
    time_frames = observer_time.get("timeFrames") or []
    nodes, edges = _pick_slice(observers, links, max_nodes=max_nodes, max_edges=max_edges)
    frame_count = max(
        len(time_frames),
        max((len(link.get("repairTrajectory") or []) for link in edges), default=0),
        1,
    )

    node_payload = []
    for row in nodes:
        axis = row.get("axis") or [0.0, 0.0, 0.0]
        x = _float(axis[0])
        y = _float(axis[1])
        z = _float(axis[2])
        node_payload.append(
            {
                "id": int(row["observerId"]),
                "x": x,
                "y": z,
                "depth": y,
                "supportPatchCount": int(row.get("supportPatchCount") or 0),
                "entropy": _float(row.get("visibleSignatureEntropy"), 0.0),
                "modularDepthMean": _float(row.get("modularDepthMean"), 0.0),
                "dominantRecordSignature": row.get("dominantRecordSignature"),
                "dominantObjectPacket": row.get("dominantObjectPacket"),
            }
        )

    edge_payload = []
    for index, link in enumerate(edges):
        errors = [_edge_error(link, frame_index) for frame_index in range(frame_count)]
        trajectory = link.get("repairTrajectory") or []
        edge_payload.append(
            {
                "id": f"e{index}",
                "source": int(link["source"]),
                "target": int(link["target"]),
                "overlapPatchCount": int(link.get("overlapPatchCount") or 0),
                "jaccard": _float(link.get("jaccard"), 0.0),
                "signatureSimilarity": _float(link.get("signatureSimilarity"), 0.0),
                "initialError": errors[0],
                "finalError": errors[-1],
                "errors": errors,
                "committed": [
                    _float((trajectory[min(i, len(trajectory) - 1)] if trajectory else {}).get("committedFraction"), 0.0)
                    for i in range(frame_count)
                ],
            }
        )

    frame_payload = []
    for index in range(frame_count):
        row = time_frames[min(index, len(time_frames) - 1)] if time_frames else {}
        values = [edge["errors"][index] for edge in edge_payload]
        mean_error = sum(values) / len(values) if values else 0.0
        max_error = max(values) if values else 0.0
        frame_payload.append(
            {
                "index": index,
                "relativeTime": _float(row.get("relativeTime"), index / max(frame_count - 1, 1)),
                "cycle": _float(row.get("cycle"), float(index)),
                "phi": _float(row.get("phi"), 0.0),
                "committedFraction": _float(row.get("committedFraction"), 0.0),
                "mismatchEdges": _float(row.get("mismatchEdges"), 0.0),
                "meanEdgeError": mean_error,
                "maxEdgeError": max_error,
            }
        )

    return {
        "schema": "oph_observer_repair_graph_viewer_v1",
        "title": "Observer Overlap Repair Graph",
        "claimBoundary": observer_time.get("claimBoundary"),
        "source": observer_time.get("source"),
        "nodeCount": len(node_payload),
        "edgeCount": len(edge_payload),
        "frameCount": frame_count,
        "nodes": node_payload,
        "edges": edge_payload,
        "frames": frame_payload,
        "notes": [
            "Observers are bounded self-reading patches with local records, support, and readback.",
            "Edges are observer-overlap supports, not decorative graph lines.",
            "The displayed error is a frame-local closure-error proxy from overlap committed fraction, repair load, and static signature gap.",
        ],
    }


def _html(data: dict[str, Any]) -> str:
    embedded = json.dumps(data, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OPH Observer Overlap Repair Graph</title>
  <style>
    :root {{
      --bg: #071013;
      --panel: rgba(11, 28, 32, 0.86);
      --ink: #eef8f6;
      --muted: #9ebbb5;
      --line: rgba(183, 224, 216, 0.22);
      --good: #63e6be;
      --warn: #ffd166;
      --bad: #ff5a73;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 20% 10%, rgba(60, 185, 160, 0.24), transparent 32rem),
        radial-gradient(circle at 80% 20%, rgba(255, 209, 102, 0.12), transparent 28rem),
        linear-gradient(135deg, #061013, #101521 58%, #080b12);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: 100vh;
    }}
    #stage {{
      width: 100%;
      height: 100vh;
      display: block;
    }}
    aside {{
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 24px;
      backdrop-filter: blur(18px);
      overflow: auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }}
    p {{ color: var(--muted); line-height: 1.45; }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      margin: 18px 0;
    }}
    .stat {{
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.04);
    }}
    .stat b {{
      display: block;
      font-size: 21px;
      letter-spacing: -0.04em;
    }}
    .stat span {{
      color: var(--muted);
      font-size: 12px;
    }}
    input[type="range"] {{ width: 100%; accent-color: var(--good); }}
    button {{
      border: 1px solid var(--line);
      color: var(--ink);
      background: rgba(99, 230, 190, 0.14);
      padding: 9px 13px;
      border-radius: 999px;
      cursor: pointer;
    }}
    .controls {{ display: flex; gap: 10px; align-items: center; margin: 14px 0 8px; }}
    .legend {{ display: grid; gap: 8px; margin-top: 18px; }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 13px; }}
    .swatch {{ width: 26px; height: 4px; border-radius: 999px; }}
    #edgeReadout {{
      white-space: pre-wrap;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: #d8f4ee;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      min-height: 110px;
      background: rgba(0, 0, 0, 0.2);
    }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      #stage {{ height: 68vh; }}
      aside {{ border-left: 0; border-top: 1px solid var(--line); }}
    }}
  </style>
</head>
<body>
<main>
  <svg id="stage" role="img" aria-label="Observer overlap repair graph"></svg>
  <aside>
    <h1>Observer overlap repair graph</h1>
    <p>Observers are self-reading patches. Edges are shared overlap supports. The slider shows how overlap closure error falls as records commit and repair propagates.</p>
    <div class="controls">
      <button id="play">play</button>
      <span id="frameLabel"></span>
    </div>
    <input id="slider" type="range" min="0" max="0" value="0" step="1" />
    <div class="stat-grid">
      <div class="stat"><b id="meanError">0</b><span>mean edge error</span></div>
      <div class="stat"><b id="maxError">0</b><span>max edge error</span></div>
      <div class="stat"><b id="phi">0</b><span>global phi</span></div>
      <div class="stat"><b id="committed">0%</b><span>committed fraction</span></div>
    </div>
    <div id="edgeReadout">Hover an edge.</div>
    <div class="legend">
      <div class="legend-row"><span class="swatch" style="background: var(--bad)"></span> high overlap error</div>
      <div class="legend-row"><span class="swatch" style="background: var(--warn)"></span> partial repair</div>
      <div class="legend-row"><span class="swatch" style="background: var(--good)"></span> closed overlap</div>
    </div>
    <p>{data.get("claimBoundary") or ""}</p>
  </aside>
</main>
<script>
const DATA = {embedded};
const svg = document.getElementById('stage');
const slider = document.getElementById('slider');
const play = document.getElementById('play');
const label = document.getElementById('frameLabel');
const readout = document.getElementById('edgeReadout');
const stats = {{
  meanError: document.getElementById('meanError'),
  maxError: document.getElementById('maxError'),
  phi: document.getElementById('phi'),
  committed: document.getElementById('committed')
}};
slider.max = Math.max(0, DATA.frameCount - 1);

const NS = 'http://www.w3.org/2000/svg';
const nodeById = new Map(DATA.nodes.map(n => [n.id, n]));
const edgeEls = [];
const labelEls = [];
const nodeEls = [];
let playing = false;
let timer = null;

function color(error) {{
  const e = Math.max(0, Math.min(1, error));
  if (e < 0.38) return `rgb(${{Math.round(99 + e * 280)}}, ${{Math.round(230 - e * 40)}}, ${{Math.round(190 - e * 110)}})`;
  if (e < 0.68) return '#ffd166';
  return '#ff5a73';
}}
function fmt(x, digits = 3) {{
  if (!Number.isFinite(x)) return '0';
  if (Math.abs(x) >= 1000) return Math.round(x).toLocaleString();
  return x.toFixed(digits);
}}
function layout() {{
  const rect = svg.getBoundingClientRect();
  const w = rect.width || 900;
  const h = rect.height || 700;
  const cx = w * 0.5;
  const cy = h * 0.5;
  const r = Math.min(w, h) * 0.42;
  for (const n of DATA.nodes) {{
    const depthScale = 0.82 + 0.18 * ((n.depth || 0) + 1) / 2;
    n.sx = cx + n.x * r * depthScale;
    n.sy = cy - n.y * r * depthScale;
  }}
  render(Number(slider.value || 0));
}}
function line(x1, y1, x2, y2, cls) {{
  const el = document.createElementNS(NS, 'line');
  el.setAttribute('x1', x1); el.setAttribute('y1', y1);
  el.setAttribute('x2', x2); el.setAttribute('y2', y2);
  if (cls) el.setAttribute('class', cls);
  svg.appendChild(el);
  return el;
}}
function circle(x, y, r) {{
  const el = document.createElementNS(NS, 'circle');
  el.setAttribute('cx', x); el.setAttribute('cy', y); el.setAttribute('r', r);
  svg.appendChild(el);
  return el;
}}
function text(x, y, value) {{
  const el = document.createElementNS(NS, 'text');
  el.setAttribute('x', x); el.setAttribute('y', y);
  el.textContent = value;
  svg.appendChild(el);
  return el;
}}
function build() {{
  svg.innerHTML = '';
  const defs = document.createElementNS(NS, 'defs');
  defs.innerHTML = '<filter id="glow"><feGaussianBlur stdDeviation="2.4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>';
  svg.appendChild(defs);
  for (const edge of DATA.edges) {{
    const a = nodeById.get(edge.source);
    const b = nodeById.get(edge.target);
    const el = line(a.sx, a.sy, b.sx, b.sy);
    el.style.strokeLinecap = 'round';
    el.style.cursor = 'pointer';
    el.addEventListener('mouseenter', () => {{
      const i = Number(slider.value || 0);
      readout.textContent =
        `observer ${{edge.source}} <-> observer ${{edge.target}}\\n` +
        `error: ${{fmt(edge.errors[i], 4)}}\\n` +
        `committed: ${{fmt(edge.committed[i] * 100, 1)}}%\\n` +
        `overlap patches: ${{edge.overlapPatchCount}}\\n` +
        `jaccard: ${{fmt(edge.jaccard, 3)}}\\n` +
        `signature similarity: ${{fmt(edge.signatureSimilarity, 3)}}`;
    }});
    edgeEls.push({{ el, edge }});
    const t = text((a.sx + b.sx) / 2, (a.sy + b.sy) / 2, '');
    t.setAttribute('text-anchor', 'middle');
    t.setAttribute('font-size', '9');
    t.setAttribute('fill', 'rgba(238,248,246,0.74)');
    t.setAttribute('paint-order', 'stroke');
    t.setAttribute('stroke', 'rgba(7,16,19,0.9)');
    t.setAttribute('stroke-width', '3');
    labelEls.push({{ el: t, edge }});
  }}
  for (const node of DATA.nodes) {{
    const c = circle(node.sx, node.sy, 5 + Math.min(7, node.supportPatchCount / 16));
    c.style.fill = '#d8fff4';
    c.style.stroke = 'rgba(7,16,19,0.92)';
    c.style.strokeWidth = '2';
    c.style.filter = 'url(#glow)';
    nodeEls.push({{ el: c, node }});
  }}
}}
function render(i) {{
  const frame = DATA.frames[i] || DATA.frames[0];
  label.textContent = `frame ${{i + 1}} / ${{DATA.frameCount}}`;
  stats.meanError.textContent = fmt(frame.meanEdgeError, 4);
  stats.maxError.textContent = fmt(frame.maxEdgeError, 4);
  stats.phi.textContent = fmt(frame.phi, 0);
  stats.committed.textContent = `${{fmt(frame.committedFraction * 100, 1)}}%`;
  for (const item of edgeEls) {{
    const e = item.edge.errors[i] ?? item.edge.finalError;
    item.el.style.stroke = color(e);
    item.el.style.strokeWidth = String(0.6 + 7.5 * e);
    item.el.style.opacity = String(0.18 + 0.76 * Math.min(1, e + 0.08));
  }}
  for (const item of labelEls) {{
    const e = item.edge.errors[i] ?? item.edge.finalError;
    item.el.textContent = e > 0.16 ? fmt(e, 2) : '';
  }}
}}
slider.addEventListener('input', () => render(Number(slider.value || 0)));
play.addEventListener('click', () => {{
  playing = !playing;
  play.textContent = playing ? 'pause' : 'play';
  if (timer) clearInterval(timer);
  if (playing) {{
    timer = setInterval(() => {{
      const next = (Number(slider.value || 0) + 1) % DATA.frameCount;
      slider.value = String(next);
      render(next);
    }}, 150);
  }}
}});
window.addEventListener('resize', layout);
layout();
build();
render(0);
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-nodes", default=96, type=int)
    parser.add_argument("--max-edges", default=220, type=int)
    args = parser.parse_args()

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    data = _viewer_payload(payload, max_nodes=args.max_nodes, max_edges=args.max_edges)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_html(data), encoding="utf-8")
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "viewer": str(args.out),
                "schema": data["schema"],
                "node_count": data["nodeCount"],
                "edge_count": data["edgeCount"],
                "frame_count": data["frameCount"],
                "source": data.get("source"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json.loads(summary_path.read_text()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
