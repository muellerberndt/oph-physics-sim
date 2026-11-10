// The Space tab, section "Force carriers on the same cone". Computed in the browser:
//  - the small-field waves of the continuous links of worker GL (gauge_wave.js, a port of repair-law/work/GL/s3_waves.py)
//    beside a quantum of the CF-1 tick (cf1_engine.js, Routing);
//  - GL's partner test (side by side, as in GL) and a head-on version of it with the same rule;
//  - the phase of the two-tick period over the zone for the CF-1 routing and for T3's member B (cone_symbol.js).
// Python numbers for comparison are read from /rl/GL/s3_waves.json and /rl/T3/cones.json. Checks: node space_gauge_check.mjs.
import { Routing, branchCoin } from './cf1_engine.js';
import { GaugeField, packetRun, partnerEffect, predictedVelocity, slope } from './gauge_wave.js';
import * as CS from './cone_symbol.js';

const PI = Math.PI;
const AMBER = [255, 180, 74], VIOLET = [169, 149, 255], PINK = [255, 100, 128], TEAL = [63, 224, 208];
const css = (c) => `rgb(${c.join(',')})`;
const F = { inited: false, started: false, eps: 0.4, k3: 0, ref: null, t3: null, beside: null, head: {}, vel: null, table: [], cones: null, anim: null, words: null, slices: {} };

const STYLE = `
.gx-jump{display:inline-block;margin-top:6px;font-size:12px;color:var(--teal);border:1px solid #1f4d55;border-radius:9px;padding:2px 9px;cursor:pointer;text-decoration:none}
.gx-jump:hover{border-color:var(--teal)}
.gx-sec{max-width:1100px;margin:34px auto 10px;padding-top:18px;border-top:1px solid var(--edge)}
.gx-sec h2{margin:0 0 4px;font-size:22px}
.gx-sec h3{margin:0 0 4px;font-size:16px;color:var(--text)}
.gx-lead{font-size:14px;color:#dfe6f5;margin:4px 0 10px;line-height:1.5;max-width:1000px}
.gx-claims{display:grid;gap:5px;margin:10px 0 6px}
.gx-claim{display:grid;grid-template-columns:205px 1fr;gap:10px;align-items:baseline;font-size:13.5px;color:var(--soft);line-height:1.4}
.gx-tag{justify-self:start;font-size:11px;padding:1px 8px;border-radius:9px;white-space:nowrap}
.gx-tag.d{background:rgba(95,227,161,.15);color:#8ff0bf}.gx-tag.c{background:rgba(255,180,74,.15);color:#ffd08a}.gx-tag.o{background:rgba(255,100,128,.16);color:#ff9db0}
.gx-block{margin:24px 0 8px}
.gx-text{font-size:13.5px;color:var(--soft);margin:2px 0 8px;line-height:1.45;max-width:1000px}
.gx-cv{width:100%;display:block;border:1px solid var(--edge);border-radius:12px;background:#060a14}
.gx-ctl{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:13px;color:var(--muted);margin:8px 0}
.gx-ctl input[type=range]{flex:1;min-width:160px;max-width:420px}
.gx-maps{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.gx-maps figure{margin:0}.gx-maps figcaption{font-size:12.5px;color:var(--soft);margin-top:4px}
.gx-maps canvas{aspect-ratio:1/1;max-width:440px;margin:0 auto}
.gx-tab{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 4px}
.gx-tab th{color:var(--muted);text-align:left;font-weight:600;font-size:12px;text-transform:none;letter-spacing:0;padding:6px 8px;border-bottom:1px solid var(--edge);background:#101a2c}
.gx-tab td{padding:6px 8px;border-bottom:1px solid #16213a;color:var(--soft);vertical-align:top}
.gx-tab td.n{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;white-space:nowrap}
.gx-tab td.ref{color:var(--muted)}
.gx-lg{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;color:var(--soft);margin:6px 0}
.gx-lg i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.gx-lg i.dash{border-radius:0;height:0;border-top:2px dashed #ffffff;width:16px;vertical-align:3px;background:none}
.gx-note{color:var(--faint);font-size:12px;margin-top:6px;line-height:1.45}
.gx-live{font-size:13px;color:var(--soft);margin:6px 0;min-height:1.4em}
@media (max-width:900px){.gx-maps{grid-template-columns:1fr}.gx-claim{grid-template-columns:1fr}}
`;

const tag = (k, label) => `<span class="gx-tag ${k}">${label}</span>`;
const claim = (k, label, text) => `<div class="gx-claim">${tag(k, label)}<span>${text}</span></div>`;
const CL = 'declared classical link action';
function html() {
  return `<h2>Force carriers on the same cone</h2>
<p class="gx-lead">Port-phase covariance on the chosen five-state code supports the Lie algebra of (SU(3) × SU(2) × U(1))/Z6. Continuous seam variables, protection/control, a trace-form electric term and plaquette action are additional choices. Their classical small-field modes can be labelled by eight colour, three weak and one hypercharge direction. Below that declared field rule runs beside a separate matter walk; the two are not coupled.</p>
<div class="gx-claims">
${claim('c', CL, 'For the stated isotropic trace-form action, the linearized continuous-time symbol has 24 transverse modes with ω² = k̂², k̂² = Σ 4 sin²(k<sub>i</sub>/2), and 12 longitudinal zero modes. The implemented finite-step dispersion differs. Covariance alone does not select this action or equal couplings.')}
${claim('c', CL, 'A packet moves 0.951499 cells per tick. The dispersion of the field step predicts 0.951654. Gluon, W and B give the same value (GL).')}
${claim('c', CL, 'The reported classical evolution has a Gauss residual near 10⁻¹⁷. This floating-point residual is a replay check, distinct from an exact Ward identity.')}
${claim('c', CL, 'A partner packet that does not commute changes a packet in proportion to ε². A partner that commutes changes nothing (GL).')}
${claim('d', 'finite walk symbol', 'The routing tick of CF-1 has 32 zero-phase cones in each arrow block, 16 of each handedness (T3).')}
${claim('c', 'conditional on a declared seam weight', 'T3\'s member B leaves a single Dirac cone in the whole zone. Its lazy seam step has the unequal weights cos 3π/32 and sin 3π/32 (T3).')}
${claim('o', 'open', 'No law yet writes the continuous port phases. The record law allows them, and reading them as a seam register is a declared step (GL).')}
</div>

<div class="gx-block"><h3>A gauge packet beside a charged quantum</h3>
<p class="gx-text">Left: classical electric-field energy in a colour-direction packet, time running down. Right: the CF-1 walk without rest phase, started in the same cell. The dashed line is one cell per tick. The quantum moves exactly on it. The packet falls slightly behind, because the field step slows waves of this wave number. Only long waves reach one cell per tick.</p>
<canvas class="gx-cv" id="gx-beside"></canvas>
<div class="gx-lg"><span><i style="background:${css(AMBER)}"></i>electric field energy of the packet</span><span><i style="background:${css(TEAL)}"></i>where the quantum would be found</span><span><i class="dash"></i>one cell per tick</span></div>
<div class="gx-ctl"><button id="gx-beside-run">replay</button><span id="gx-beside-t"></span></div>
<table class="gx-tab" id="gx-vel"></table>
<div class="gx-note">Same set-up as GL's run: a line of 48 cells, plane packets with wave number 0.6 and width 4 cells, amplitude 10⁻³, two field steps of 1/2 per tick, speed fitted over ticks 5 to 24. The packets are uniform across the line, so one cell across gives the same numbers as GL's 48 × 4 × 4 box. The quantum and the packet are not coupled here.</div></div>

<div class="gx-block"><h3>Classical packets: commuting and noncommuting controls</h3>
<p class="gx-text">Two packets in crossed polarizations run into each other. Top row: gluon λ1 against gluon λ2, which do not commute. Bottom row: gluon λ1 against B, which commute. The right panels show what the partner did to the λ1 packet: the difference from a run without the partner, magnified four times. This head-on version runs the declared classical rule, without quantum scattering probabilities. GL's own test sends the two packets side by side; it is recomputed in the table below.</p>
<canvas class="gx-cv" id="gx-scat"></canvas>
<div class="gx-lg"><span><i style="background:${css(AMBER)}"></i>gluon λ1</span><span><i style="background:${css(VIOLET)}"></i>partner: gluon λ2 (top), B (bottom)</span><span><i style="background:${css(PINK)}"></i>change of the λ1 packet caused by the partner, × 4</span></div>
<div class="gx-ctl"><span>amplitude ε</span><select id="gx-eps"><option value="0.05">0.05</option><option value="0.1">0.1</option><option value="0.2">0.2</option><option value="0.4" selected>0.4</option></select><button id="gx-scat-run">replay</button><span id="gx-scat-t"></span></div>
<div class="gx-live" id="gx-scat-num"></div>
<table class="gx-tab" id="gx-gl"></table>
<div class="gx-note">Table: GL's partner test, run here. Both packets start in the same place and move together for 16 ticks; the entry is the relative change of the main packet's field. Each cell gives the value computed here, then GL's Python value.</div></div>

<div class="gx-block"><h3>One Dirac cone instead of 32</h3>
<p class="gx-text">The phase per period of the two-tick product, over a slice of the zone of wave numbers. Teal marks phase zero, where a cone sits. Pink marks phase π. Left: the routing tick of CF-1, sub-step S<sub>i</sub>(p) = exp(−i p<sub>i</sub> α<sub>i</sub>), period S₁S₂S₃S₃S₂S₁; both arrow blocks have this same map. Right: T3's member B at φ = 5π/16, whose lazy sub-step has the exact symbol B<sub>i</sub>(p) = [cos²φ cos p<sub>i</sub> + sin²φ] − i cos φ sin p<sub>i</sub> α<sub>i</sub> − i sin φ cos φ (1 − cos p<sub>i</sub>) β, period B₁B₂B₃B₃B₂B₁. No mass angle in either.</p>
<div class="gx-maps"><figure><canvas class="gx-cv" id="gx-map-a"></canvas><figcaption id="gx-cap-a">routing tick of CF-1</figcaption></figure><figure><canvas class="gx-cv" id="gx-map-b"></canvas><figcaption id="gx-cap-b">member B</figcaption></figure></div>
<div class="gx-ctl"><span>third wave number p₃</span><input type="range" id="gx-k3" min="-64" max="64" step="1" value="0"><b id="gx-k3-val" style="color:var(--text);min-width:64px">0</b><button id="gx-k3-0">0</button><button id="gx-k3-h">π/2</button><button id="gx-k3-p">π</button></div>
<canvas class="gx-cv" id="gx-line"></canvas>
<div class="gx-lg"><span><i style="background:${css(TEAL)}"></i>routing tick of CF-1</span><span><i style="background:${css(AMBER)}"></i>member B</span><span>phase per period along p₁ at p₂ = 0, the dashed line on the maps</span></div>
<table class="gx-tab" id="gx-cones"></table>
<div class="gx-note">Counted here over the full zone: the low points of the phase on a 32³ grid, each refined, a cone accepted where 1 − cos Ω < 10⁻¹¹ (T3 used 48³ and two grid shifts). Handedness: the sign of the Jacobian of the arrow-block symbol at each node. A finite-range tick cannot have net handedness in a block (T3, theorem), so member B's cone is a Dirac cone, not a single Weyl cone.</div></div>
<p class="gx-note" style="font-size:12.5px">These classical link waves and the free walk use separately declared clocks and actions. They do not supply a full quantum gauge vacuum, confinement, an interacting continuum or particle masses. The spatial spin pointer and occupation statistics are declared, distinct from the code's internal weak doublet.</p>`;
}

// ---- canvases -----------------------------------------------------------------------------------------------------
function setup(cv, cssH) {
  const dpr = Math.min(2, window.devicePixelRatio || 1), W = Math.max(50, cv.clientWidth || cv.parentNode.clientWidth || 600), H = cssH === null ? W : cssH;
  cv.style.height = H + 'px'; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H); g.fillStyle = '#060a14'; g.fillRect(0, 0, W, H);
  return { g, W, H };
}
// a space-time picture: layers [{rows, col, gain, norm}], rows[t][x]; returns an offscreen canvas of L x (T + 1) pixels
function stImage(layers, L, T, tShow) {
  const c = document.createElement('canvas'); c.width = L; c.height = T + 1;
  const g = c.getContext('2d'), img = g.createImageData(L, T + 1), d = img.data;
  for (let k = 0; k < d.length; k += 4) { d[k] = 6; d[k + 1] = 10; d[k + 2] = 20; d[k + 3] = 255; }
  for (const ly of layers) {
    let mx = ly.norm; if (!mx) { mx = 0; for (const r of ly.rows) for (const v of r) mx = Math.max(mx, v); }
    for (let t = 0; t <= Math.min(T, tShow); t++) { const r = ly.rows[t]; if (!r) continue;
      for (let x = 0; x < L; x++) { const u = Math.min(1, (ly.gain || 1) * Math.sqrt(Math.max(0, r[x]) / (mx || 1))); if (u < 0.01) continue; const o = 4 * (t * L + x); d[o] = Math.min(255, d[o] + ly.col[0] * u); d[o + 1] = Math.min(255, d[o + 1] + ly.col[1] * u); d[o + 2] = Math.min(255, d[o + 2] + ly.col[2] * u); } }
  }
  g.putImageData(img, 0, 0); return c;
}
function panel(g, img, x, y, w, h, title, L, T, cone) {
  g.imageSmoothingEnabled = false; g.drawImage(img, x, y, w, h);
  g.strokeStyle = 'rgba(147,163,192,.35)'; g.lineWidth = 1; g.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  if (cone) { g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip(); g.strokeStyle = 'rgba(255,255,255,.92)'; g.lineWidth = 1.6; g.setLineDash([6, 6]); g.beginPath();
    for (const [x0, dir] of cone) { g.moveTo(x + w * (x0 + 0.5) / L, y); g.lineTo(x + w * (x0 + 0.5 + dir * (T + 1)) / L, y + h); }
    g.stroke(); g.restore(); g.setLineDash([]); }
  g.font = '12px ui-monospace,Menlo,monospace'; const tw = g.measureText(title).width;
  g.fillStyle = 'rgba(6,10,20,.78)'; g.fillRect(x + 4, y + 4, tw + 9, 17); g.fillStyle = 'rgba(230,236,250,.88)'; g.fillText(title, x + 8, y + 16);
}
function axisNote(g, x, y, text) { g.fillStyle = 'rgba(147,163,192,.75)'; g.font = '11px ui-monospace,Menlo,monospace'; g.fillText(text, x, y); }

// ---- 1. a gauge packet beside a charged quantum ------------------------------------------------------------------
const BES = { L: 48, T: 30 };
function besideCompute() {
  const f = new GaugeField(BES.L, 1, 1); f.addPacket(0, 1, 1e-3);
  const q = new Routing(BES.L, 1, 1); q.addPacket([12, 0, 0], 4, [0.6, 0, 0], branchCoin([0.6, 0, 0], 0)); q.normalize();
  const gr = [f.profile()], qr = [q.density(new Float64Array(BES.L))], gx = [], qx = [];
  for (let t = 1; t <= BES.T; t++) { f.tick(); q.tickAltHalf(0); gr.push(f.profile()); qr.push(q.density(new Float64Array(BES.L))); gx.push(f.centroid()); qx.push(q.meanPosition(0)); }
  const ts = Array.from({ length: 20 }, (_, i) => i + 5);
  F.beside = { gr, qr, vq: slope(ts, ts.map((t) => qx[t - 1])), vg: slope(ts, ts.map((t) => gx[t - 1])) };
}
function besideDraw(tShow = BES.T) {
  const cv = document.getElementById('gx-beside'); if (!cv || !F.beside) return;
  const { g, W, H } = setup(cv, 300), gap = 16, pw = (W - gap - 2) / 2, ph = H - 30, { L, T } = BES;
  panel(g, stImage([{ rows: F.beside.gr, col: AMBER }], L, T, tShow), 1, 1, pw, ph, 'classical colour packet λ1, polarization y', L, T, [[12, 1]]);
  panel(g, stImage([{ rows: F.beside.qr, col: TEAL }], L, T, tShow), 1 + pw + gap, 1, pw, ph, 'separate CF-1 matter walk', L, T, [[12, 1]]);
  for (const x0 of [1, 1 + pw + gap]) { axisNote(g, x0 + 2, H - 10, `cell 0 … ${L}, tick 0 … ${T} downward`); }
  const el = document.getElementById('gx-beside-t'); if (el) el.textContent = `tick ${Math.min(tShow, T)}`;
}
function velTable() {
  const el = document.getElementById('gx-vel'); if (!el) return;
  const R = F.ref && F.ref.packets_small_amplitude, f6 = (x) => (x === undefined || x === null ? 'n/a' : x.toFixed(6));
  let h = '<tr><th>what moves</th><th>speed here, cells per tick</th><th>GL, Python</th><th>Gauss law after 24 ticks</th></tr>';
  if (F.vel) for (const r of F.vel) h += `<tr><td>${r.label}</td><td class="n">${f6(r.v)}</td><td class="n ref">${R ? f6(R[r.key].velocity_cells_per_tick) : 'n/a'}</td><td class="n">${r.gauss.toExponential(1)}</td></tr>`;
  else h += '<tr><td colspan="4">computing …</td></tr>';
  h += `<tr><td>prediction from the dispersion of the field step</td><td class="n">${f6(predictedVelocity(48))}</td><td class="n ref">${R ? f6(R[Object.keys(R)[0]].predicted_packet_weighted_group_velocity) : 'n/a'}</td><td></td></tr>`;
  if (F.beside) h += `<tr><td>charged quantum of the CF-1 tick, no rest phase</td><td class="n">${f6(F.beside.vq)}</td><td class="ref">exactly 1 along an axis</td><td></td></tr>`;
  el.innerHTML = h;
}
const VEL = [['gluon λ1, polarization y', 'gluon lambda_1, polarization 2', 0, 1], ['W1, polarization y', 'W1, polarization 2', 8, 1], ['B, polarization y', 'B_GUT, polarization 2', 11, 1], ['gluon λ8, polarization z', 'gluon lambda_8, polarization 3', 7, 2]];
function velCompute() { F.vel = VEL.map(([label, key, a, pol]) => { const r = packetRun(a, pol); return { label, key, v: r.velocity, gauss: r.gauss }; }); }

// ---- 2. head-on scattering and GL's table -------------------------------------------------------------------------
const HO = { L: 96, T: 50 };
function headCompute(eps) {
  if (F.head[eps]) return F.head[eps];
  const run = (partner) => {
    const f = new GaugeField(HO.L, 1, 1); f.addPacket(0, 1, eps, { centre: 24, dir: 1 }); if (partner !== null) f.addPacket(partner, 2, eps, { centre: 72, dir: -1 });
    const main = [f.profile([0])], part = [partner === null ? null : f.profile([partner])], field = [Float64Array.from({ length: HO.L }, (_, x) => f.E[1][12 * x])];
    for (let t = 1; t <= HO.T; t++) { f.tick(); main.push(f.profile([0])); if (partner !== null) part.push(f.profile([partner])); field.push(Float64Array.from({ length: HO.L }, (_, x) => f.E[1][12 * x])); }
    return { f, main, part, field };
  };
  const alone = run(null), out = { eps };
  for (const [key, partner] of [['nc', 1], ['c', 11]]) {
    const r = run(partner), diff = r.field.map((row, t) => row.map((v, x) => (v - alone.field[t][x]) ** 2));
    const A = alone.field[HO.T], B = r.field[HO.T]; let d = 0, n = 0; for (let x = 0; x < HO.L; x++) { d += (B[x] - A[x]) ** 2; n += A[x] ** 2; }
    const cm = (row) => { let s = 0, sx = 0; row.forEach((v, x) => { s += v; sx += v * x; }); return sx / s; };
    const ph = (f) => { let sr = 0, si = 0; f.forEach((v, x) => { sr += v * Math.cos(0.6 * x); si += v * Math.sin(0.6 * x); }); return Math.atan2(si, sr); };
    out[key] = { main: r.main, part: r.part, diff, change: Math.sqrt(d / n), delay: cm(alone.main[HO.T]) - cm(r.main[HO.T]), phase: ph(B) - ph(A), gauss: r.f.gauss(), norm: Math.max(...alone.main.map((row) => Math.max(...row))), dnorm: Math.max(...alone.field.map((row) => Math.max(...row.map((v) => v * v)))) };
  }
  return (F.head[eps] = out);
}
function scatDraw(tShow = HO.T) {
  const cv = document.getElementById('gx-scat'), h = F.head[F.eps]; if (!cv || !h) return;
  const { g, W, H } = setup(cv, 470), gap = 16, pw = (W - gap - 2) / 2, ph = (H - gap - 30) / 2, { L, T } = HO;
  [['nc', 'gluon λ1 against gluon λ2: they do not commute'], ['c', 'gluon λ1 against B: they commute']].forEach(([key, title], row) => {
    const r = h[key], y = 1 + row * (ph + gap);
    panel(g, stImage([{ rows: r.main, col: AMBER, norm: r.norm }, { rows: r.part, col: VIOLET, norm: r.norm }], L, T, tShow), 1, y, pw, ph, title, L, T, [[24, 1], [72, -1]]);
    panel(g, stImage([{ rows: r.diff, col: PINK, norm: r.dnorm, gain: 4 }], L, T, tShow), 1 + pw + gap, y, pw, ph, key === 'nc' ? 'change of the λ1 packet, × 4' : 'change of the λ1 packet, × 4: none', L, T, null);
  });
  axisNote(g, 3, H - 10, `cell 0 … ${L}, tick 0 … ${T} downward`);
  const el = document.getElementById('gx-scat-t'); if (el) el.textContent = `tick ${Math.min(tShow, T)}`;
  const n = h.nc, c = h.c, num = document.getElementById('gx-scat-num');
  if (num) num.innerHTML = `ε = ${F.eps}. Do not commute: the λ1 packet leaves the crossing changed by ${(100 * n.change).toFixed(2)} %, ${n.delay.toFixed(3)} cells behind a free packet, its wave phase shifted by ${n.phase.toFixed(3)}; change / ε² = ${(n.change / F.eps ** 2).toFixed(3)}. Commute: change ${c.change.toExponential(1)}, delay ${Math.abs(c.delay) < 5e-13 ? '0' : c.delay.toExponential(1)}. Gauss law at the end: ${Math.max(n.gauss, c.gauss).toExponential(1)}.`;
}
const GLT = [['gluon λ1 with gluon λ2', 'gluon lambda_1 (pol 2) with gluon lambda_2 (pol 3); commutator lambda_3', [0, 1], [1, 2], 6], ['W1 with W2', 'W1 (pol 2) with W2 (pol 3); commutator W3', [8, 1], [9, 2], 10],
  ['gluon λ1 with B', 'control: gluon lambda_1 (pol 2) with B (pol 3)', [0, 1], [11, 2], null], ['gluon λ1 with W1', 'control: gluon lambda_1 (pol 2) with W1 (pol 3)', [0, 1], [8, 2], null]];
const EPS = [0.05, 0.1, 0.2, 0.4];
function glTable() {
  const el = document.getElementById('gx-gl'); if (!el) return;
  const S = F.ref && F.ref.self_interaction, fmt = (x) => (x === undefined ? '…' : x < 1e-6 ? (x === 0 ? '0' : x.toExponential(1)) : x.toFixed(4));
  let h = `<tr><th>ε</th>${GLT.map((c, j) => `<th>${c[0]}${j < 2 ? ' (do not commute)' : ' (commute)'}</th>`).join('')}</tr>`;
  EPS.forEach((eps, i) => { h += `<tr><td class="n">${eps}</td>${GLT.map((c, j) => { const v = F.table[4 * j + i], ref = S ? S[c[1]]['eps=' + eps].relative_change_of_main_field : undefined; return `<td class="n">${fmt(v && v.change)} <span class="ref" style="color:var(--muted)">· ${ref === undefined ? 'n/a' : fmt(ref)}</span></td>`; }).join('')}</tr>`; });
  const done = F.table.filter(Boolean);
  if (done.length === 16) { const r = [0, 1, 2, 3].map((i) => F.table[i].change / EPS[i] ** 2); h += `<tr><td colspan="5" style="color:var(--muted)">change / ε² for gluon λ1 with gluon λ2: ${r.map((v) => v.toFixed(2)).join(', ')}. Largest Gauss defect in these 16 runs: ${Math.max(...done.map((x) => x.gauss)).toExponential(1)}.</td></tr>`; }
  el.innerHTML = h;
}

// ---- 3. cones of the two-tick period ------------------------------------------------------------------------------
function words() { if (!F.words) F.words = { a: CS.compile(CS.periodCF1(0)), b: CS.compile(CS.periodB(5 * PI / 16, 0)) }; return F.words; }
const k3val = () => F.k3 * PI / 64;
const piFrac = (k) => { if (k === 0) return '0'; const s = k < 0 ? '−' : '', a = Math.abs(k), gcd = (x, y) => (y ? gcd(y, x % y) : x), d = gcd(a, 64), n = a / d, m = 64 / d; return `${s}${n === 1 ? '' : n}π${m === 1 ? '' : '/' + m}`; };
// phase per period to colour: dark blue in the middle, teal towards phase 0 (white at a cone), pink towards phase pi,
// faint iso-lines every pi/8
function phaseRGB(om) {
  const z = Math.max(0, 1 - om / 0.9) ** 1.6, w = Math.max(0, 1 - (PI - om) / 0.9) ** 1.6, core = Math.max(0, 1 - om / 0.12), s = Math.sin(om), fr = (om / (PI / 8)) % 1, band = Math.min(fr, 1 - fr) < 0.05 ? 18 : 0;
  return [8 + 22 * s + band + 70 * z + 255 * w + 180 * core, 12 + 30 * s + band + 235 * z + 95 * w + 30 * core, 26 + 48 * s + band + 215 * z + 122 * w + 40 * core];
}
function mapDraw(id, word, which, lowres = false) {
  const cv = document.getElementById(id); if (!cv) return;
  const W0 = Math.min(440, cv.parentNode.clientWidth || 400), { g, W, H } = setup(cv, W0), n = lowres ? 72 : 180, key = `${which}:${F.k3}:${n}`;
  let om = F.slices[key]; if (!om) { om = CS.omegaSlice(word, n, k3val()); F.slices[key] = om; }
  const c = document.createElement('canvas'); c.width = n; c.height = n; const cg = c.getContext('2d'), img = cg.createImageData(n, n);
  for (let b = 0; b < n; b++) for (let a = 0; a < n; a++) { const [r, gg, bb] = phaseRGB(om[b * n + a]), o = 4 * ((n - 1 - b) * n + a); img.data[o] = Math.min(255, r); img.data[o + 1] = Math.min(255, gg); img.data[o + 2] = Math.min(255, bb); img.data[o + 3] = 255; }
  cg.putImageData(img, 0, 0);
  const m = 26, top = 10, S = Math.min(W - m - 10, H - top - 22); g.imageSmoothingEnabled = true; g.drawImage(c, m, top, S, S);
  const X = (p) => m + S * (p + PI) / (2 * PI), Y = (p) => top + S * (1 - (p + PI) / (2 * PI));
  g.strokeStyle = 'rgba(230,236,250,.45)'; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(m, Y(0)); g.lineTo(m + S, Y(0)); g.stroke(); g.setLineDash([]);
  // cones of this slice from the full-zone count
  let inSlice = 0;
  if (F.cones) for (const z of F.cones[which].zeros) { let d = Math.abs(z.p[2] - k3val()); d = Math.min(d, 2 * PI - d); if (d > 1e-3) continue; inSlice++;
    for (const sx of [0, 2 * PI, -2 * PI]) for (const sy of [0, 2 * PI, -2 * PI]) { const px = z.p[0] + sx, py = z.p[1] + sy; if (px < -PI - 1e-9 || px > PI + 1e-9 || py < -PI - 1e-9 || py > PI + 1e-9) continue; g.strokeStyle = '#ffffff'; g.lineWidth = 1.6; g.beginPath(); g.arc(X(px), Y(py), 7, 0, 2 * PI); g.stroke(); } }
  g.fillStyle = 'rgba(147,163,192,.85)'; g.font = '11px ui-monospace,Menlo,monospace'; const yb = top + S + 15;
  g.fillText('−π', m - 4, yb); g.fillText('π', m + S - 8, yb); g.fillText('p₁', m + S / 2 - 6, yb);
  g.fillText('π', 8, top + 10); g.fillText('p₂', 6, top + S / 2 + 4); g.fillText('−π', 2, top + S);
  return inSlice;
}
function mapsDraw(lowres = false) {
  const w = words(), na = mapDraw('gx-map-a', w.a, 'a', lowres), nb = mapDraw('gx-map-b', w.b, 'b', lowres);
  const v = document.getElementById('gx-k3-val'); if (v) v.textContent = piFrac(F.k3);
  const ca = document.getElementById('gx-cap-a'), cb = document.getElementById('gx-cap-b');
  if (ca) ca.innerHTML = `Routing tick of CF-1. Cones in this slice: <b>${F.cones ? na : '…'}</b> (32 in the whole zone).`;
  if (cb) cb.innerHTML = `Member B, φ = 5π/16. Cones in this slice: <b>${F.cones ? nb : '…'}</b> (1 in the whole zone, at p = 0).`;
  lineDraw();
}
function lineDraw() {
  const cv = document.getElementById('gx-line'); if (!cv) return;
  const { g, W, H } = setup(cv, 170), m = { l: 40, r: 12, t: 12, b: 22 }, n = 361, w = words();
  const X = (p) => m.l + (W - m.l - m.r) * (p + PI) / (2 * PI), Y = (o) => H - m.b - (H - m.t - m.b) * o / PI;
  g.strokeStyle = 'rgba(147,163,192,.3)'; g.lineWidth = 1; g.beginPath(); g.moveTo(m.l, m.t); g.lineTo(m.l, H - m.b); g.lineTo(W - m.r, H - m.b); g.stroke();
  g.setLineDash([3, 4]); g.beginPath(); g.moveTo(m.l, Y(PI)); g.lineTo(W - m.r, Y(PI)); g.moveTo(X(0), m.t); g.lineTo(X(0), H - m.b); g.stroke(); g.setLineDash([]);
  for (const [word, col] of [[w.a, TEAL], [w.b, AMBER]]) { g.strokeStyle = css(col); g.lineWidth = 2; g.beginPath(); for (let i = 0; i < n; i++) { const p = -PI + 2 * PI * i / (n - 1), o = CS.omega(word, [p, 0, k3val()]); if (i) g.lineTo(X(p), Y(o)); else g.moveTo(X(p), Y(o)); } g.stroke(); }
  g.fillStyle = 'rgba(147,163,192,.85)'; g.font = '11px ui-monospace,Menlo,monospace';
  g.fillText('π', 22, Y(PI) + 4); g.fillText('0', 26, H - m.b + 4); g.fillText('−π', m.l - 6, H - 6); g.fillText('0', X(0) - 3, H - 6); g.fillText('π', W - m.r - 8, H - 6); g.fillText('phase per period', m.l + 8, m.t + 10); g.fillText('p₁', X(PI / 2), H - 6);
}
function conesCompute() { const w = words(); F.cones = { a: CS.criticalPoints(w.a, 32), b: CS.criticalPoints(w.b, 32) }; const chi = (z) => [1, -1].map((c) => CS.blockChirality(CS.periodCF1(0), z.p, c).chirality); F.cones.hand = F.cones.a.zeros.map(chi); F.cones.speedB = CS.speedPerTick(CS.periodB()); }
function conesTable() {
  const el = document.getElementById('gx-cones'); if (!el) return;
  const T = F.t3 && F.t3.members, ra = T && T['W2 (CF-1)'].full, rb = T && T['B (recommended, phi = 5 pi/16)'].result_theta0, C = F.cones, f4 = (x) => (x === null || x === undefined ? 'none' : x.toFixed(4));
  const ref = (x) => `<span style="color:var(--muted)"> · T3 ${x}</span>`;
  let h = '<tr><th>over the whole zone</th><th>routing tick of CF-1</th><th>member B, φ = 5π/16</th></tr>';
  if (!C) { el.innerHTML = h + '<tr><td colspan="3">counting …</td></tr>'; return; }
  const plus = C.hand.filter((x) => x[0] > 0).length, minus = C.hand.filter((x) => x[0] < 0).length, plus2 = C.hand.filter((x) => x[1] > 0).length;
  h += `<tr><td>cones (phase zero)</td><td class="n">${C.a.zeros.length}${ref(ra ? ra.count_phase0 : 32)}</td><td class="n">${C.b.zeros.length}, at p = (${C.b.zeros[0] ? C.b.zeros[0].p.map((v) => Math.abs(v) < 1e-6 ? '0' : v.toFixed(3)).join(', ') : ''})${ref(rb ? rb.count_phase0 : 1)}</td></tr>`;
  h += `<tr><td>handedness of the cones, arrow block +1 (block −1)</td><td class="n">${plus} of sign +, ${minus} of sign − (${plus2} and ${C.hand.length - plus2}): net 0</td><td>one Dirac cone: both handedness signs at the same point, joined by the β term</td></tr>`;
  h += `<tr><td>points at phase π</td><td class="n">${C.a.pis.length}${ref(ra ? ra.count_phasepi : 32)}</td><td class="n">${C.b.pis.length}${ref(rb ? rb.count_phasepi : 0)}</td></tr>`;
  h += `<tr><td>lightest doubler, phase per period</td><td class="n">none: the doublers are cones</td><td class="n">${f4(C.b.lightest)}${ref(rb ? rb.lightest_doubler_phase_per_period.toFixed(4) : '0.7042')}</td></tr>`;
  h += `<tr><td>gap at phase π</td><td class="n">0</td><td class="n">${f4(C.b.gapAtPi)} = π/4${ref(rb ? rb.gap_at_pi_per_period.toFixed(4) : '0.7854')}</td></tr>`;
  h += `<tr><td>speed without mass, cells per tick</td><td class="n">1 along an axis</td><td class="n">${C.speedB.toFixed(4)} = cos φ${ref(rb ? rb.massless_speed_per_tick_100.toFixed(4) : '0.5556')}</td></tr>`;
  h += `<tr><td>continuum colour running (imported formula b₀ = 11 − 2 n<sub>f</sub>/3 for n<sub>f</sub> Dirac fundamentals)</td><td>Cones alone do not determine the physical fermion census.</td><td>A vectorlike Dirac cone is not a chiral Standard Model family; no beta coefficient is inferred here.</td></tr>`;
  el.innerHTML = h;
}

// ---- shell --------------------------------------------------------------------------------------------------------
function animate(kind) {
  const T = kind === 'beside' ? BES.T : HO.T, t0 = performance.now(), dur = kind === 'beside' ? 1800 : 2600;
  F.anim = F.anim || {}; const id = (F.anim[kind] = (F.anim[kind] || 0) + 1);
  const step = () => { if (F.anim[kind] !== id) return; const u = Math.min(1, (performance.now() - t0) / dur), t = Math.round(u * T);
    if (kind === 'beside') besideDraw(t); else scatDraw(t); if (u < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}
const later = (fn) => new Promise((res) => setTimeout(() => { fn(); res(); }, 0));
async function start() {
  if (F.started) return; F.started = true; const t0 = performance.now(), lap = {}, mark = (k) => { lap[k] = Math.round(performance.now() - t0); };
  const get = async (u) => { try { const r = await fetch(u); return r.ok ? await r.json() : null; } catch (e) { return null; } };
  get('/rl/GL/s3_waves.json').then((d) => { F.ref = d; velTable(); glTable(); });
  get('/rl/T3/cones.json').then((d) => { F.t3 = d; conesTable(); });
  await later(() => { besideCompute(); velTable(); animate('beside'); mark('beside'); });
  await later(() => { headCompute(F.eps); animate('scat'); mark('headOn'); });
  await later(() => { mapsDraw(true); conesTable(); mark('mapsLow'); });
  await later(() => { velCompute(); velTable(); mark('speeds'); });
  await later(() => { conesCompute(); conesTable(); mapsDraw(); mark('cones'); });
  for (let j = 0; j < GLT.length; j++) for (let i = 0; i < EPS.length; i++) await later(() => { const c = GLT[j]; F.table[4 * j + i] = partnerEffect(c[2], c[3], EPS[i], c[4]); glTable(); });
  mark('glTable'); F.timing = lap; window.__spaceForces.ready = true;
}
export function forcesInit() {
  if (F.inited) return; F.inited = true;
  if (!document.getElementById('gx-style')) { const st = document.createElement('style'); st.id = 'gx-style'; st.textContent = STYLE; document.head.appendChild(st); }
  const host = document.getElementById('sp-left') || document.getElementById('page-space'); if (!host) return;
  const sec = document.createElement('section'); sec.id = 'gx-forces'; sec.className = 'gx-sec'; sec.innerHTML = html(); host.appendChild(sec);
  const head = host.querySelector('.car-head');
  if (head) { const a = document.createElement('a'); a.className = 'gx-jump'; a.href = '#gx-forces'; a.textContent = 'Force carriers on the same cone: further down'; a.onclick = (e) => { e.preventDefault(); sec.scrollIntoView({ behavior: 'smooth', block: 'start' }); start(); }; head.appendChild(a); }
  document.getElementById('gx-beside-run').onclick = () => animate('beside');
  document.getElementById('gx-scat-run').onclick = () => animate('scat');
  document.getElementById('gx-eps').onchange = (e) => { F.eps = parseFloat(e.target.value); setTimeout(() => { headCompute(F.eps); animate('scat'); }, 0); };
  const k3 = document.getElementById('gx-k3'), setK = (k, low) => { F.k3 = k; k3.value = String(k); mapsDraw(low); };
  k3.oninput = () => setK(parseInt(k3.value), true); k3.onchange = () => setK(parseInt(k3.value), false);
  document.getElementById('gx-k3-0').onclick = () => setK(0, false); document.getElementById('gx-k3-h').onclick = () => setK(32, false); document.getElementById('gx-k3-p').onclick = () => setK(-64, false);
  velTable(); glTable(); conesTable();
  if ('IntersectionObserver' in window) { const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); start(); } }, { root: host === document.getElementById('sp-left') ? host : null, rootMargin: '600px' }); io.observe(sec); }
  else start();
  window.addEventListener('resize', () => forcesShown());
}
export function forcesShown() {
  if (!F.inited) return;
  if (F.beside) besideDraw(); if (F.head[F.eps]) scatDraw(); if (F.words) mapsDraw();
}
window.__spaceForces = { F, start, ready: false };
