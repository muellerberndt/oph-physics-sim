from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_physics_problem_outputs_viewer(report: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """Write a compact visual atlas for the adjacent physics-problem contracts."""

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_render_html(report), encoding="utf-8")
    outputs = report.get("outputs") or {}
    return {
        "mode": "physics_problem_outputs_viewer",
        "viewer_path": str(destination),
        "output_count": len(outputs),
        "featured_demo": "homochirality",
        "physical_claim": False,
    }


def _render_html(report: dict[str, Any]) -> str:
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"), default=str).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OPH Physics Problem Atlas</title>
<style>
:root {{
  color-scheme: dark;
  --ink:#eef4ff; --muted:#9faec4; --panel:#111a28; --panel2:#162235;
  --line:#2a3b55; --cyan:#58d5e8; --blue:#78a9ff; --amber:#f4bd65;
  --green:#67d69d; --red:#ff7f87; --bg:#08101b;
}}
* {{ box-sizing:border-box }}
body {{ margin:0; background:radial-gradient(circle at 78% 0,#17243a 0,transparent 38%),var(--bg); color:var(--ink); font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif }}
main {{ width:min(1180px,calc(100% - 32px)); margin:0 auto; padding:42px 0 64px }}
h1,h2,h3,p {{ margin-top:0 }} h1 {{ font-size:clamp(30px,5vw,54px); line-height:1.04; max-width:780px; letter-spacing:-.035em }}
h2 {{ font-size:22px; letter-spacing:-.015em }} h3 {{ font-size:16px }}
.eyebrow {{ color:var(--cyan); text-transform:uppercase; letter-spacing:.16em; font-size:12px; font-weight:750 }}
.lede {{ color:var(--muted); font-size:17px; max-width:850px }}
.notice {{ border:1px solid #5b482a; background:#211a10; color:#f7d9a5; padding:12px 15px; border-radius:10px; max-width:900px }}
.legend {{ display:flex; gap:8px; flex-wrap:wrap; margin:22px 0 30px }}
.tag {{ display:inline-flex; align-items:center; gap:7px; padding:5px 9px; border:1px solid var(--line); border-radius:999px; font-size:12px; font-weight:700; letter-spacing:.02em }}
.tag:before {{ content:""; width:8px; height:8px; border-radius:50%; background:currentColor }}
.assumed {{ color:var(--amber) }} .derived {{ color:var(--blue) }} .validated {{ color:var(--green) }} .blocked {{ color:var(--red) }}
.pipeline {{ display:grid; grid-template-columns:repeat(5,1fr); gap:22px; margin:18px 0 38px }}
.stage {{ position:relative; min-height:116px; padding:15px; border:1px solid var(--line); border-radius:12px; background:linear-gradient(145deg,var(--panel2),var(--panel)) }}
.stage:not(:last-child):after {{ content:"→"; position:absolute; right:-18px; top:42px; color:var(--cyan); font-size:22px }}
.stage b {{ display:block; color:var(--cyan); margin-bottom:6px }} .stage span {{ color:var(--muted); font-size:13px }}
.featured {{ border:1px solid #355174; border-radius:18px; background:rgba(14,24,39,.91); padding:22px; box-shadow:0 24px 70px #0007 }}
.feature-grid {{ display:grid; grid-template-columns:minmax(250px,.78fr) minmax(430px,1.5fr); gap:24px }}
.controls {{ display:grid; gap:14px }} .control label {{ display:flex; justify-content:space-between; gap:15px; font-size:13px; color:var(--muted) }}
input[type=range] {{ width:100%; accent-color:var(--cyan) }}
.criterion {{ margin:16px 0; padding:13px; border-radius:10px; background:#0c1522; border:1px solid var(--line) }}
.criterion strong {{ font-size:18px }} .criterion.pass strong {{ color:var(--green) }} .criterion.fail strong {{ color:var(--amber) }}
.plot-card {{ border:1px solid var(--line); border-radius:12px; padding:12px; background:#09121e }}
.plot-title {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.1em; margin-bottom:8px }}
svg {{ width:100%; height:auto; display:block }} .axis {{ stroke:#42556f; stroke-width:1 }} .zero {{ stroke:#6f8097; stroke-dasharray:4 5 }}
.curve {{ fill:none; stroke:var(--cyan); stroke-width:2.5 }} .drift {{ fill:none; stroke:var(--blue); stroke-width:2 }}
.fixed {{ fill:var(--green); stroke:#092; stroke-width:1 }}
.mini-chain {{ display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-top:18px }} .mini-chain div {{ padding:9px; border-radius:8px; background:#111d2d; border:1px solid var(--line); font-size:12px }}
.mini-chain b {{ display:block; color:var(--cyan); margin-bottom:4px }} .mini-chain span {{ color:var(--muted) }}
.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:20px }}
.list-card {{ border:1px solid var(--line); border-radius:12px; padding:15px; background:#0c1522 }} .list-card ul {{ margin:0; padding-left:19px; color:var(--muted) }}
.atlas {{ margin-top:38px }} .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:12px }}
table {{ width:100%; border-collapse:collapse; min-width:780px; background:#0d1724 }} th,td {{ padding:11px 12px; border-bottom:1px solid #24344b; text-align:left; vertical-align:top }}
th {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.1em; background:#111d2c }} td:first-child {{ font-weight:700 }}
.state {{ font-size:11px; font-weight:800; padding:4px 7px; border-radius:999px; white-space:nowrap; display:inline-block }}
.state.demo {{ color:#f7d69b; background:#382a14 }} .state.model {{ color:#a8c6ff; background:#172b4a }} .state.evidence {{ color:#9aebbf; background:#153528 }} .state.open {{ color:#ffadb2; background:#3e1d24 }}
footer {{ color:var(--muted); margin-top:28px; font-size:13px }} code {{ color:#b7d5ff }}
@media (max-width:850px) {{ .pipeline,.mini-chain {{ grid-template-columns:1fr }} .stage:not(:last-child):after {{ content:"↓"; right:auto; left:18px; top:auto; bottom:-24px }} .feature-grid,.two-col {{ grid-template-columns:1fr }} }}
</style>
</head>
<body>
<main>
  <div class="eyebrow">OPH finite demonstrator</div>
  <h1>A physics-problem atlas with its assumptions left visible</h1>
  <p class="lede">The simulator shows how an observer-like patch becomes a public record. It separates stipulated demo physics from mathematics derived inside that demo and from evidence supplied by an experiment.</p>
  <p class="notice">A complete animation is not a physical proof. Amber items are chosen inputs. Blue items follow from those inputs. Green is reserved for checks backed by an artifact. Red marks the next physical gate.</p>
  <div class="legend" aria-label="Epistemic legend">
    <span class="tag assumed">Assumed for demo</span><span class="tag derived">Derived in model</span><span class="tag validated">Artifact checked</span><span class="tag blocked">Physical gate open</span>
  </div>

  <section aria-labelledby="pipeline-title">
    <h2 id="pipeline-title">The common OPH chain</h2>
    <div class="pipeline">
      <div class="stage"><b>Bounded patch</b><span>Local state and a declared physical or software boundary.</span></div>
      <div class="stage"><b>Readback</b><span>A quantity by which the patch reads its present record.</span></div>
      <div class="stage"><b>Persistent record</b><span>State that survives long enough to affect a later update.</span></div>
      <div class="stage"><b>Repair or feedback</b><span>A rule that changes later state using the readback.</span></div>
      <div class="stage"><b>Public receipt</b><span>A reproducible output with provenance and an explicit claim limit.</span></div>
    </div>
  </section>

  <section class="featured" aria-labelledby="hom-title">
    <div class="eyebrow">Featured visual: reduced chemical kinetics</div>
    <h2 id="hom-title">Homochirality as record-branch selection</h2>
    <p class="lede">Move the rates to see the racemic branch change stability. The picture is a Frank-type normal form. It illustrates OPH readback and repair; it does not choose the terrestrial hand or identify the historical chemistry.</p>
    <div class="feature-grid">
      <div>
        <div class="controls">
          <div class="control"><label for="e0">Initial excess <output id="e0v"></output></label><input id="e0" type="range" min="-0.20" max="0.20" step="0.002" value="0.01"></div>
          <div class="control"><label for="kappa">Amplification κ <output id="kv"></output></label><input id="kappa" type="range" min="0" max="3" step="0.02" value="1"></div>
          <div class="control"><label for="mu">Erasure μ <output id="muv"></output></label><input id="mu" type="range" min="0" max="1.5" step="0.01" value="0.15"></div>
          <div class="control"><label for="bias">Signed source h <output id="hv"></output></label><input id="bias" type="range" min="-0.35" max="0.35" step="0.005" value="0"></div>
        </div>
        <div class="criterion" id="criterion"><div>Model branch criterion</div><strong id="criterionText"></strong><div id="criterionDetail"></div></div>
        <span class="tag assumed">Rates are provisional</span>
        <span class="tag derived">Threshold is algebraic</span>
        <span class="tag blocked">Chemistry and global fixation open</span>
      </div>
      <div>
        <div class="plot-card"><div class="plot-title">Enantiomeric excess e(t)</div><svg id="trajectory" viewBox="0 0 680 250" role="img" aria-label="Homochirality trajectory"></svg></div>
        <div class="plot-card" style="margin-top:12px"><div class="plot-title">Phase portrait: de/dt across the physical interval</div><svg id="phase" viewBox="0 0 680 210" role="img" aria-label="Homochirality phase portrait"></svg></div>
      </div>
    </div>
    <div class="mini-chain" id="homChain"></div>
    <div class="two-col"><div class="list-card"><h3>Derived inside this normal form</h3><ul id="derivedList"></ul></div><div class="list-card"><h3>Not derived by this panel</h3><ul id="blockedList"></ul></div></div>
  </section>

  <section class="atlas" aria-labelledby="atlas-title">
    <h2 id="atlas-title">All registered problem contracts</h2>
    <p class="lede">One compact table replaces a wall of panels. Open gates remain part of the result.</p>
    <div class="table-wrap"><table><thead><tr><th>Problem</th><th>Role</th><th>Status</th><th>Physical claim</th><th>Boundary</th></tr></thead><tbody id="problemRows"></tbody></table></div>
  </section>
  <footer>Generated from <code>physics_problem_outputs_report.json</code>. No experimental or material claim is promoted by this viewer.</footer>
</main>
<script>
const DATA={payload};
const NS="http://www.w3.org/2000/svg";
const hom=(DATA.outputs||{{}}).homochirality||{{}};
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));

function line(svg,x1,y1,x2,y2,cls) {{ const e=document.createElementNS(NS,"line"); e.setAttribute("x1",x1);e.setAttribute("y1",y1);e.setAttribute("x2",x2);e.setAttribute("y2",y2);e.setAttribute("class",cls);svg.appendChild(e); }}
function path(svg,points,cls) {{ const e=document.createElementNS(NS,"path");e.setAttribute("d",points.map((p,i)=>(i?"L":"M")+p[0].toFixed(2)+","+p[1].toFixed(2)).join(" "));e.setAttribute("class",cls);svg.appendChild(e); }}
function dot(svg,x,y) {{ const e=document.createElementNS(NS,"circle");e.setAttribute("cx",x);e.setAttribute("cy",y);e.setAttribute("r",4);e.setAttribute("class","fixed");svg.appendChild(e); }}
function drift(e,k,mu,h) {{ return (1-e*e)*(k*e+h)-2*mu*e; }}
function step(e,dt,k,mu,h) {{ const a=drift(e,k,mu,h),b=drift(e+dt*a/2,k,mu,h),c=drift(e+dt*b/2,k,mu,h),d=drift(e+dt*c,k,mu,h); return Math.max(-1,Math.min(1,e+dt*(a+2*b+2*c+d)/6)); }}
function draw() {{
  const e0=+$('e0').value,k=+$('kappa').value,mu=+$('mu').value,h=+$('bias').value;
  $('e0v').value=e0.toFixed(3);$('kv').value=k.toFixed(2);$('muv').value=mu.toFixed(2);$('hv').value=h.toFixed(3);
  const margin=k-2*mu, pass=margin>0;
  $('criterion').className='criterion '+(pass?'pass':'fail');
  $('criterionText').textContent=pass?'κ > 2μ: two chiral branches in the unbiased model':'κ ≤ 2μ: racemic branch stable when h=0';
  $('criterionDetail').textContent=`margin κ-2μ = ${{margin.toFixed(3)}} demo-time⁻¹`;
  const T=12,n=300,dt=T/(n-1),vals=[];let e=e0;for(let i=0;i<n;i++){{vals.push([i*dt,e]);e=step(e,dt,k,mu,h)}}
  const svg=$('trajectory');svg.replaceChildren();line(svg,42,120,666,120,'zero');line(svg,42,12,42,230,'axis');line(svg,42,230,666,230,'axis');
  path(svg,vals.map(([t,v])=>[42+t/T*624,120-v*108]),'curve');
  const ps=[],raw=[];for(let i=0;i<=240;i++){{const x=-1+2*i/240,y=drift(x,k,mu,h);raw.push(y);ps.push([x,y])}}const ymax=Math.max(.05,...raw.map(Math.abs));
  const phase=$('phase');phase.replaceChildren();line(phase,42,100,666,100,'zero');line(phase,354,12,354,188,'zero');line(phase,42,188,666,188,'axis');
  path(phase,ps.map(([x,y])=>[42+(x+1)/2*624,100-y/ymax*80]),'drift');
  if(Math.abs(h)<1e-9){{dot(phase,354,100);if(pass){{const b=Math.sqrt(1-2*mu/k);dot(phase,42+(b+1)/2*624,100);dot(phase,42+(-b+1)/2*624,100)}}}}
}}
['e0','kappa','mu','bias'].forEach(id=>$(id).addEventListener('input',draw));

function renderHom() {{
  const chain=hom.ophChain||[];$('homChain').innerHTML=chain.map(x=>`<div><b>${{esc(x.stage)}}</b><span>${{esc(x.value)}}</span></div>`).join('');
  $('derivedList').innerHTML=(hom.derivedWithinModel||[]).map(x=>`<li>${{esc(x)}}</li>`).join('');
  $('blockedList').innerHTML=(hom.blockedPhysicalClaims||[]).map(x=>`<li>${{esc(x)}}</li>`).join('');
}}
function roleFor(key,x) {{
  if(key==='homochirality')return ['model','model demonstrator'];
  if(x.physicalClaim===true||x.materialClaim===true)return ['evidence','physical candidate'];
  if(x.computed===true)return ['model','computed diagnostic'];
  if(String(x.status||'').includes('source_only'))return ['demo','declared source ansatz'];
  return ['open','contract or open gate'];
}}
function boundary(x) {{ return x.claimBoundary||x.claim_boundary||(x.blockers&&x.blockers[0])||'No physical promotion boundary supplied.'; }}
function renderTable() {{
  const rows=Object.entries(DATA.outputs||{{}}).map(([key,x])=>{{const [cls,role]=roleFor(key,x);return `<tr><td>${{esc(key.replaceAll('_',' '))}}</td><td><span class="state ${{cls}}">${{esc(role)}}</span></td><td>${{esc(x.status||x.claim||x.strongestAllowedClaim||'open')}}</td><td>${{x.physicalClaim===true||x.materialClaim===true?'yes':'no'}}</td><td>${{esc(boundary(x))}}</td></tr>`}});
  $('problemRows').innerHTML=rows.join('');
}}
renderHom();renderTable();draw();
</script>
</body>
</html>
"""
