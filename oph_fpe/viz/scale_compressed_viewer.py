from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


def write_scale_compressed_viewer(run_dir: Path, out_path: Path | None = None) -> dict[str, Any]:
    """Write a compact HTML viewer for the scale-compressed repair branch."""

    run_dir = Path(run_dir)
    report = _read_json(run_dir / "scale_compressed_repair_report.json")
    camb_report_path = _first_existing(
        run_dir / "scale_compressed_cmb_camb_report.json",
        run_dir / "camb_transfer" / "scale_compressed_cmb_camb_report.json",
    )
    camb_bins_path = _first_existing(
        run_dir / "scale_compressed_cmb_tt_bins.csv",
        run_dir / "camb_transfer" / "scale_compressed_cmb_tt_bins.csv",
    )
    camb = _read_json(camb_report_path) if camb_report_path is not None else {}
    objects = _read_rows(run_dir / "scale_compressed_h3_objects.csv")
    particles = _read_rows(run_dir / "scale_compressed_particles.csv")
    rounds = _read_rows(run_dir / "scale_compressed_repair_rounds.csv")
    screen_cl = _read_rows(run_dir / "scale_compressed_screen_cl.csv")
    camb_bins = _read_rows(camb_bins_path) if camb_bins_path is not None else []

    if not report:
        raise FileNotFoundError(run_dir / "scale_compressed_repair_report.json")

    destination = out_path or (run_dir / "plots" / "scale_compressed_repair_viewer.html")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _render_html(
            run_dir=run_dir,
            report=report,
            camb=camb,
            objects=objects,
            particles=particles,
            rounds=rounds,
            screen_cl=screen_cl,
            camb_bins=camb_bins,
        ),
        encoding="utf-8",
    )
    summary = {
        "mode": "scale_compressed_repair_viewer",
        "viewer_path": str(destination),
        "run_dir": str(run_dir),
        "object_count": len(objects),
        "particle_observation_count": len(particles),
        "logical_round_count": len(rounds),
        "screen_cl_row_count": len(screen_cl),
        "camb_bin_count": len(camb_bins),
        "camb_report_path": str(camb_report_path) if camb_report_path is not None else None,
        "camb_bins_path": str(camb_bins_path) if camb_bins_path is not None else None,
        "scale_compressed_operator_receipt": bool(report.get("scale_compressed_operator_receipt", False)),
        "populated_h3_preview_receipt": bool(
            ((report.get("h3_preview") or {}).get("populated_h3_preview_receipt", False))
        ),
        "measurement_comparable_cmb_curve": bool(camb.get("measurement_comparable_cmb_curve", False)),
        "physical_cmb_prediction": False,
        "strict_neutral_bulk": False,
        "claim_boundary": (
            "Visualization of the scale-compressed repair branch, H3 object preview, particle preview, "
            "screen C_l scaffold, and CAMB TT transfer if present. It does not upgrade strict neutral "
            "bulk, production-particle, or physical-CMB gates."
        ),
    }
    (destination.parent / "scale_compressed_repair_viewer_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def _render_html(
    *,
    run_dir: Path,
    report: dict[str, Any],
    camb: dict[str, Any],
    objects: list[dict[str, str]],
    particles: list[dict[str, str]],
    rounds: list[dict[str, str]],
    screen_cl: list[dict[str, str]],
    camb_bins: list[dict[str, str]],
) -> str:
    readouts = report.get("cmb_parameter_readouts", {}) or {}
    h3 = report.get("h3_preview", {}) or {}
    particle = report.get("particle_preview", {}) or {}
    camb_comp = camb.get("comparison", {}) if isinstance(camb, dict) else {}
    payload = {
        "runDir": str(run_dir),
        "readouts": readouts,
        "h3": h3,
        "particle": particle,
        "camb": {
            "available": bool(camb),
            "measurementComparable": bool(camb.get("measurement_comparable_cmb_curve", False)),
            "physicalPrediction": False,
            "irShapeCorrelation": _nested(camb_comp, "scale_compressed_ir_kernel", "shape_correlation"),
            "irChi2PerBin": _nested(camb_comp, "scale_compressed_ir_kernel", "amplitude_fit_chi2_per_bin"),
            "lcdmShapeCorrelation": _nested(camb_comp, "camb_lcdm_powerlaw", "shape_correlation"),
            "claimBoundary": camb.get("claim_boundary"),
        },
        "objects": [
            {
                "id": row.get("object_id"),
                "x": _float(row.get("h3_x")),
                "y": _float(row.get("h3_y")),
                "z": _float(row.get("h3_z")),
                "response": _float(row.get("mean_cap_response")),
            }
            for row in objects
        ],
        "particles": [
            {
                "id": row.get("particle_id"),
                "round": int(_float(row.get("logical_round"), 0.0)),
                "x": _float(row.get("h3_x")),
                "y": _float(row.get("h3_y")),
                "z": _float(row.get("h3_z")),
                "sector": row.get("sector_class"),
            }
            for row in particles
        ],
        "rounds": [
            {
                "round": int(_float(row.get("logical_round"), 0.0)),
                "capacity": _float(row.get("capacity_ratio")),
                "mismatch": _float(row.get("mismatch_residual")),
                "release": _float(row.get("record_release_weight")),
            }
            for row in rounds
        ],
        "screenCl": _thin_series(
            [
                {"ell": _float(row.get("ell")), "D": _float(row.get("D_ell"))}
                for row in screen_cl
                if row.get("ell") is not None
            ],
            max_rows=900,
        ),
        "cambBins": [
            {
                "ell": _float(row.get("ell")),
                "observed": _float(row.get("observed_D_ell")),
                "compressed": _float(row.get("scale_compressed_ir_kernel_D_ell")),
                "lcdm": _float(row.get("camb_lcdm_powerlaw_D_ell")),
            }
            for row in camb_bins
            if row.get("ell") is not None
        ],
        "claimBoundary": report.get("claim_boundary"),
    }
    data = json.dumps(payload, separators=(",", ":"), default=str)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OPH Scale-Compressed Repair Viewer</title>
<style>
body {{ margin:0; background:#101214; color:#e8edf2; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
header {{ padding:16px 20px; background:#15191d; border-bottom:1px solid #2b3035; }}
h1 {{ margin:0 0 8px; font-size:20px; }}
.sub {{ color:#aab3bd; font-size:13px; line-height:1.4; }}
.metrics {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }}
.metric {{ background:#242a30; border:1px solid #333b43; border-radius:6px; padding:6px 8px; font-size:12px; }}
.pass {{ background:#173a2a; color:#b9f0ce; }}
.open {{ background:#472124; color:#ffcbc9; }}
main {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; padding:12px; }}
section {{ background:#171b20; border:1px solid #2b3035; border-radius:8px; overflow:hidden; min-height:320px; }}
section h2 {{ margin:0; padding:10px 12px; font-size:14px; border-bottom:1px solid #2b3035; }}
svg {{ display:block; width:100%; height:320px; background:#111417; }}
.note {{ padding:8px 12px 12px; color:#aab3bd; font-size:12px; line-height:1.4; }}
.wide {{ grid-column:1/-1; }}
@media (max-width:900px) {{ main {{ grid-template-columns:1fr; }} .wide {{ grid-column:auto; }} }}
</style>
</head>
<body>
<header>
  <h1>OPH Scale-Compressed Repair Viewer</h1>
  <div class="sub" id="subtitle"></div>
  <div class="metrics" id="metrics"></div>
</header>
<main>
  <section><h2>H3 Object Preview</h2><svg id="objects"></svg><div class="note" id="objectsNote"></div></section>
  <section><h2>Particle Preview</h2><svg id="particles"></svg><input id="roundSlider" type="range" min="0" max="24" value="0" style="width:calc(100% - 24px);margin:8px 12px 0;"><div class="note" id="particleNote"></div></section>
  <section><h2>Repair Round Trace</h2><svg id="rounds"></svg><div class="note">Capacity grows by the logical scale round; mismatch contracts by the local repair derivative.</div></section>
  <section><h2>Screen C_l Scaffold</h2><svg id="screenCl"></svg><div class="note">Screen scalar scaffold before acoustic Boltzmann transfer.</div></section>
  <section class="wide"><h2>CAMB TT Transfer Against Planck Binned TT</h2><svg id="camb"></svg><div class="note" id="cambNote"></div></section>
</main>
<script>
const DATA = {data};
const NS = "http://www.w3.org/2000/svg";
function el(n,a={{}}){{const e=document.createElementNS(NS,n);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);return e;}}
function clear(s){{while(s.firstChild)s.removeChild(s.firstChild);}}
function dims(s){{const r=s.getBoundingClientRect();return [Math.max(320,r.width),Math.max(240,r.height||320)];}}
function scale(vals,pad,extent){{const min=Math.min(...vals),max=Math.max(...vals);return v=>pad+(extent-2*pad)*(v-min)/Math.max(max-min,1e-12);}}
function yscale(vals,pad,extent){{const min=Math.min(...vals),max=Math.max(...vals);return v=>pad+(extent-2*pad)*(1-(v-min)/Math.max(max-min,1e-12));}}
function color(v){{const x=Math.max(0,Math.min(1,v||0));return `rgb(${{Math.round(60+190*x)}},${{Math.round(180-80*x)}},${{Math.round(230-160*x)}})`;}}
function drawObjects(){{const svg=document.getElementById('objects');clear(svg);const[w,h]=dims(svg);svg.setAttribute('viewBox',`0 0 ${{w}} ${{h}}`);svg.appendChild(el('rect',{{x:0,y:0,width:w,height:h,fill:'#111417'}}));const pts=DATA.objects;if(!pts.length)return;const sx=scale(pts.map(p=>p.x),28,w),sy=yscale(pts.map(p=>p.y),24,h);pts.forEach(p=>svg.appendChild(el('circle',{{cx:sx(p.x),cy:sy(p.y),r:4,fill:color(p.response),opacity:.82}})));document.getElementById('objectsNote').textContent=`${{pts.length}} H3 preview objects from support-visible cap profiles. Strict neutral bulk remains false.`;}}
function particleGroups(){{const g={{}};(DATA.particles||[]).forEach(p=>{{(g[p.id] ||= []).push(p);}});return g;}}
function drawParticles(round=0){{const svg=document.getElementById('particles');clear(svg);const[w,h]=dims(svg);svg.setAttribute('viewBox',`0 0 ${{w}} ${{h}}`);svg.appendChild(el('rect',{{x:0,y:0,width:w,height:h,fill:'#111417'}}));const groups=particleGroups();const all=DATA.particles;if(!all.length)return;const sx=scale(all.map(p=>p.x),28,w),sy=yscale(all.map(p=>p.y),24,h);Object.values(groups).forEach((pts,idx)=>{{pts.sort((a,b)=>a.round-b.round);const d=pts.map((p,i)=>`${{i?'L':'M'}}${{sx(p.x).toFixed(1)}} ${{sy(p.y).toFixed(1)}}`).join(' ');svg.appendChild(el('path',{{d,fill:'none',stroke:idx%2?'#ff8a65':'#66d9ef','stroke-width':1.5,opacity:.55}}));const cur=pts.reduce((best,p)=>Math.abs(p.round-round)<Math.abs(best.round-round)?p:best,pts[0]);svg.appendChild(el('circle',{{cx:sx(cur.x),cy:sy(cur.y),r:5,fill:idx%2?'#ff8a65':'#66d9ef',opacity:.95}}));}});document.getElementById('particleNote').textContent=`Logical round ${{round}}. ${{Object.keys(groups).length}} scale-compressed particle/worldline previews; production particle gate remains false.`;}}
function drawLine(svg,rows,xkey,ykey,stroke){{if(!rows.length)return;const[w,h]=dims(svg);const xs=rows.map(r=>r[xkey]),ys=rows.map(r=>r[ykey]);const sx=scale(xs,32,w),sy=yscale(ys,24,h);const d=rows.map((r,i)=>`${{i?'L':'M'}}${{sx(r[xkey]).toFixed(1)}} ${{sy(r[ykey]).toFixed(1)}}`).join(' ');svg.appendChild(el('path',{{d,fill:'none',stroke,'stroke-width':2,opacity:.85}}));}}
function drawRounds(){{const svg=document.getElementById('rounds');clear(svg);const[w,h]=dims(svg);svg.setAttribute('viewBox',`0 0 ${{w}} ${{h}}`);svg.appendChild(el('rect',{{x:0,y:0,width:w,height:h,fill:'#111417'}}));const rows=DATA.rounds.map(r=>({{round:r.round,logCapacity:Math.log10(Math.max(r.capacity,1e-300)),negLogMismatch:-Math.log10(Math.max(r.mismatch,1e-300)),release:r.release}}));drawLine(svg,rows,'round','logCapacity','#b5f26d');drawLine(svg,rows,'round','negLogMismatch','#66d9ef');}}
function drawScreenCl(){{const svg=document.getElementById('screenCl');clear(svg);const[w,h]=dims(svg);svg.setAttribute('viewBox',`0 0 ${{w}} ${{h}}`);svg.appendChild(el('rect',{{x:0,y:0,width:w,height:h,fill:'#111417'}}));drawLine(svg,DATA.screenCl,'ell','D','#f6c85f');}}
function drawCamb(){{const svg=document.getElementById('camb');clear(svg);const[w,h]=dims(svg);svg.setAttribute('viewBox',`0 0 ${{w}} ${{h}}`);svg.appendChild(el('rect',{{x:0,y:0,width:w,height:h,fill:'#111417'}}));if(!DATA.cambBins.length){{document.getElementById('cambNote').textContent='CAMB transfer file not present.';return;}}drawLine(svg,DATA.cambBins,'ell','observed','#f6c85f');drawLine(svg,DATA.cambBins,'ell','compressed','#66d9ef');drawLine(svg,DATA.cambBins,'ell','lcdm','#8f99a3');document.getElementById('cambNote').textContent=`Planck binned TT vs scale-compressed CAMB IR curve. Shape correlation: ${{DATA.camb.irShapeCorrelation}}; chi2/bin: ${{DATA.camb.irChi2PerBin}}. Physical CMB prediction remains false.`;}}
function init(){{document.getElementById('subtitle').textContent=DATA.runDir;const m=document.getElementById('metrics');const r=DATA.readouts;m.innerHTML=[`n_s ${{r.n_s}}`,`eta_R ${{r.eta_R}}`,`q_IR ${{r.q_IR}}`,`ell_IR ${{r.ell_IR}}`,`objects ${{DATA.h3.object_count}}`,`particles ${{DATA.particle.particle_worldline_count}}`,DATA.camb.measurementComparable?'CAMB comparable':'CAMB open','physical CMB false','strict bulk false'].map((x,i)=>`<span class="metric ${{i===6&&DATA.camb.measurementComparable?'pass':''}} ${{i>6?'open':''}}">${{x}}</span>`).join('');drawObjects();drawParticles(0);const slider=document.getElementById('roundSlider');slider.max=Math.max(...DATA.rounds.map(r=>r.round),24);slider.oninput=()=>drawParticles(Number(slider.value));drawRounds();drawScreenCl();drawCamb();}}
init();
</script>
</body>
</html>
"""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _thin_series(rows: list[dict[str, float]], *, max_rows: int) -> list[dict[str, float]]:
    if len(rows) <= max_rows:
        return rows
    step = max(1, len(rows) // max_rows)
    return rows[::step]


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)
