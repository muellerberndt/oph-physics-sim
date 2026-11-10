// The Story tab: one particle from birth to mass to record, every step executed live.
//  1 Born     the committed rule of the big runs with its declared register, on one carrier (story_rule.js)
//  2 Mass     the routing tick of the candidate CF-1 (cf1_engine.js) with and without a rest angle
//  3 Recorded position detector and momentum apparatus (phase estimation of the translation) on the same tick
//  4 Mass     what the model fixes and what it does not, particle by particle
//  5 Leptons  how a charged lepton gets its mass: the native circulant, its balance and phase (story_lepton.js)
import { $ } from './app.js';
import { lkInit, lkShown } from './story_lepton.js';
import { PV, EDGES } from './walk_engine.js';
import { Carrier, OFACES } from './story_rule.js';
import { describeRegisterEvent, realClockAmplitude } from './story_space_evidence.js';
import { Routing, branchCoin } from './cf1_engine.js';

const COLS = { 1: '#ff7a45', 5: '#ff7a45', 4: '#ffb347', 2: '#ffb347', 3: '#3ee6a8' };
const RING = { 5: true, 2: true };
const KIND_COL = { birth: '#ffe45c', move: '#7dd3fc', annihilation: '#ff5f7a', fusion: '#b197ff', split: '#b197ff', exchange: '#c5d0e6' };
let inited = false;

// ---------------------------------------------------------------- 1. born
const B = { c: null, seed: 1, running: false, yaw: 0.6, pitch: 0.35, log: [], flash: null, quiet: 0, last: 0 };
function bornNew() { B.seed = 1 + Math.floor(Math.random() * 1e6); B.c = new Carrier(B.seed); B.log = []; B.flash = null; B.quiet = 0; bornDraw(); }
function bornStep() {
  const ev = B.c.runSweep();
  for (const e of ev) { if (e.kind !== 'move' || B.log.length < 3 || Math.random() < 0.25) B.log.unshift(e); }
  B.log = B.log.slice(0, 14);
  const big = ev.filter((e) => e.kind !== 'move'); B.flash = (big.length ? big[big.length - 1] : ev[ev.length - 1]) || null;
  B.quiet = ev.length ? 0 : B.quiet + 1;
}
function disturb() {         // move three load units from one port to another: a local quench like the start of the runs
  let a = Math.floor(Math.random() * 12), b = (a + 1 + Math.floor(Math.random() * 11)) % 12, n = Math.min(3, B.c.x[a]);
  B.c.x[a] -= n; B.c.x[b] += n; B.log.unshift({ kind: 'note', text: `${n} units of load moved from port ${a} to port ${b}` }); B.log = B.log.slice(0, 14); bornDraw();
}
function proj() {
  const cy = Math.cos(B.yaw), sy = Math.sin(B.yaw), cp = Math.cos(B.pitch), sp = Math.sin(B.pitch), S = 128;
  return PV.map(([x, y, z]) => { const x1 = cy * x + sy * z, z1 = -sy * x + cy * z, y1 = cp * y - sp * z1, z2 = sp * y + cp * z1; return [300 + S * x1, 240 - S * y1, z2]; });
}
function bornDraw() {
  const host = $('st-born-svg'); if (!host || !B.c) return;
  const Q = proj(), c = B.c; let h = '<defs><filter id="stglow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>';
  const depth = (ids) => ids.reduce((s, i) => s + Q[i][2], 0) / ids.length;
  const faces = OFACES.map((f, i) => ({ f, i, z: depth(f) })).sort((a, b) => a.z - b.z);
  for (const { f, i, z } of faces) {
    const k = c.cls[i], front = z > 0, pts = f.map((p) => Q[p][0].toFixed(1) + ',' + Q[p][1].toFixed(1)).join(' ');
    const hot = B.flash && B.flash.faces && B.flash.faces.includes(i);
    h += `<polygon points="${pts}" fill="${k ? COLS[k] : '#9fb4e0'}" fill-opacity="${k ? (front ? 0.22 : 0.07) : (front ? 0.04 : 0.012)}" stroke="${hot ? KIND_COL[B.flash.kind] : 'none'}" stroke-width="${hot ? 2.5 : 0}" stroke-opacity="0.9"/>`;
  }
  const seams = EDGES.map((e, i) => ({ e, i, z: depth(e) })).sort((a, b) => a.z - b.z);
  for (const { e, i, z } of seams) {
    const hot = B.flash && B.flash.seam === i, o = z > 0 ? 0.55 : 0.16;
    h += `<line x1="${Q[e[0]][0].toFixed(1)}" y1="${Q[e[0]][1].toFixed(1)}" x2="${Q[e[1]][0].toFixed(1)}" y2="${Q[e[1]][1].toFixed(1)}" stroke="${hot ? KIND_COL[B.flash.kind] : '#c5d0e6'}" stroke-opacity="${hot ? 1 : o}" stroke-width="${hot ? 3.5 : 1.1}"/>`;
  }
  for (const { f, i, z } of faces) {
    const k = c.cls[i]; if (!k) continue;
    const cx = (Q[f[0]][0] + Q[f[1]][0] + Q[f[2]][0]) / 3, cyy = (Q[f[0]][1] + Q[f[1]][1] + Q[f[2]][1]) / 3, o = z > 0 ? 1 : 0.35, r = z > 0 ? 11 : 7;
    h += RING[k] ? `<circle cx="${cx.toFixed(1)}" cy="${cyy.toFixed(1)}" r="${r}" fill="none" stroke="${COLS[k]}" stroke-width="4" stroke-opacity="${o}" filter="url(#stglow)"/>`
      : k === 3 ? `<circle cx="${cx.toFixed(1)}" cy="${cyy.toFixed(1)}" r="${r}" fill="${COLS[k]}" fill-opacity="${o}" stroke="#4cc3ff" stroke-width="3" stroke-opacity="${o}" filter="url(#stglow)"/>`
      : `<circle cx="${cx.toFixed(1)}" cy="${cyy.toFixed(1)}" r="${r}" fill="${COLS[k]}" fill-opacity="${o}" filter="url(#stglow)"/>`;
  }
  const ports = PV.map((_, p) => p).sort((a, b) => Q[a][2] - Q[b][2]);
  for (const p of ports) {
    const [x, y, z] = Q[p], o = z > 0 ? 1 : 0.4, n = c.x[p];
    h += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${(5 + 2.2 * n).toFixed(1)}" fill="#0b1220" stroke="#3fe0d0" stroke-opacity="${0.8 * o}" stroke-width="1.6"/>`;
    h += `<text x="${x.toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="middle" fill="#dfe6f5" fill-opacity="${o}" font-size="12" font-family="ui-monospace,Menlo,monospace">${n}</text>`;
  }
  host.innerHTML = h;
  const k = c.counts, list = B.log.map((e) => e.kind === 'note' ? `<div class="st-ev" style="--c:#93a3c0"><b>you</b>${e.text}</div>`
    : `<div class="st-ev" style="--c:${KIND_COL[e.kind]}"><b>sweep ${e.sweep + 1}</b>${describeRegisterEvent(e)}</div>`).join('');
  $('st-born-side').innerHTML = `<div class="st-counts">${[['birth', 'pairs born'], ['annihilation', 'annihilated'], ['move', 'moves'], ['fusion', 'fusions'], ['split', 'splits'], ['exchange', 'exchanges']].map(([key, lab]) => `<div style="--c:${KIND_COL[key]}"><b>${k[key]}</b><span>${lab}</span></div>`).join('')}</div>
    <div class="small" style="margin:8px 0 4px">sweep ${c.sweep} · ${c.defects()} defect faces on the carrier · loads ${c.balanced() ? 'settled' : 'still uneven'}</div>${list}`;
}

// ---------------------------------------------------------------- 2. mass
const M = { L: 320, T: 150, theta: 0.1, eng: [], rows: [[], []], t: 0, running: false };
function massNew() {
  M.eng = [0, M.theta].map(() => new Routing(M.L, 1, 1)); M.rows = [[], []]; M.t = 0; M.amp = [];
  const SQ = Math.SQRT1_2; for (const e of M.eng) { e.addPacket([M.L / 2, 0, 0], 5, [0, 0, 0], [[SQ, 0], [0, 0], [0, SQ], [0, 0]]); e.normalize(); }   // the rest state with phase exp(-i theta) per tick
  M.ref = [Float64Array.from(M.eng[1].re), Float64Array.from(M.eng[1].im)];
  massRecord(); massDraw();
}
function massRecord() { M.eng.forEach((e, j) => { const d = new Float64Array(M.L); e.density(d); M.rows[j].push(d); }); const e = M.eng[1]; let sr = 0, si = 0; for (let x = 0; x < M.L; x++) { sr += e.re[4 * x]; si += e.im[4 * x]; } M.amp.push([sr, si]); }
function massTick() { if (M.t >= M.T) return; M.eng[0].tickAltHalf(0); M.eng[1].tickAltHalf(M.theta / 2); M.t++; massRecord(); }
function waterfall(cv, rows, col) {
  const g = cv.getContext('2d'), W = cv.width, H = cv.height; g.fillStyle = '#04060d'; g.fillRect(0, 0, W, H);
  const img = g.createImageData(W, H), [r, gg, b] = col, dx = W / M.L, dy = H / M.T;
  rows.forEach((d, t) => { let mx = 0; for (const v of d) mx = Math.max(mx, v); for (let x = 0; x < M.L; x++) { const u = Math.sqrt(d[x] / (mx || 1)); if (u < 0.02) continue;
    for (let py = Math.floor(t * dy); py < Math.floor((t + 1) * dy); py++) for (let px = Math.floor(x * dx); px < Math.floor((x + 1) * dx); px++) { const o = 4 * (py * W + px); img.data[o] = r * u; img.data[o + 1] = gg * u; img.data[o + 2] = b * u; img.data[o + 3] = 255; } } });
  g.putImageData(img, 0, 0);
  g.strokeStyle = 'rgba(255,255,255,.35)'; g.setLineDash([5, 5]); g.beginPath(); g.moveTo(W / 2, 0); g.lineTo(W / 2 - M.T * dx, H); g.moveTo(W / 2, 0); g.lineTo(W / 2 + M.T * dx, H); g.stroke(); g.setLineDash([]);
  g.fillStyle = 'rgba(255,255,255,.6)'; g.font = '12px ui-monospace,Menlo,monospace'; g.fillText('time ↓   dashed: one cell per tick', 8, 16);
}
function massDraw() {
  waterfall($('st-mass-a'), M.rows[0], [63, 224, 208]); waterfall($('st-mass-b'), M.rows[1], [255, 228, 92]);
  const p = Math.sin(M.theta) ** 2, ov = M.amp[M.amp.length - 1] || [1, 0], a0 = M.amp[0] || [1, 0], ang = Math.atan2(ov[1], ov[0]) - Math.atan2(a0[1], a0[0]);
  const dial = $('st-dial'), g = dial.getContext('2d'), R = 52; g.clearRect(0, 0, 130, 130); g.strokeStyle = 'rgba(147,163,192,.4)'; g.lineWidth = 1.5; g.beginPath(); g.arc(65, 65, R, 0, 2 * Math.PI); g.stroke();
  g.strokeStyle = '#ffe45c'; g.lineWidth = 3; g.beginPath(); g.moveTo(65, 65); g.lineTo(65 + R * Math.cos(-ang), 65 + R * Math.sin(-ang)); g.stroke();
  $('st-mass-num').innerHTML = `<table class="t" style="width:100%"><tr><td>rest angle per tick</td><td class="num">${M.theta.toFixed(3)}</td></tr><tr><td>rest-coin flip probability from a definite record basis state</td><td class="num">${(100 * p).toFixed(2)} %</td></tr>
    <tr><td>zero-angle rest coin: flip probability</td><td class="num">0</td></tr><tr><td>tick</td><td class="num">${M.t}</td></tr><tr><td>its clock: phase of its part at rest</td><td class="num">${((((-ang) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)).toFixed(3)}</td></tr><tr><td>rest angle × ticks, modulo one turn</td><td class="num">${((M.theta * M.t) % (2 * Math.PI)).toFixed(3)}</td></tr></table>`;
}

// ---------------------------------------------------------------- 3. recorded
const R = { L: 256, theta: 0.1, eng: null, t: 0, running: false, track: false, det: 32, clock: 32, recs: [], hits: [], p0: 2 * Math.PI * 12 / 256, every: 20 };
function recNew() { R.eng = new Routing(R.L, 1, 1); R.eng.packetOnBranch(Math.round(R.p0 * R.L / (2 * Math.PI)), 1.2, R.theta); R.t = 0; R.recs = []; R.hits = []; R.shift = 40; shiftTo(R.shift); recDraw(); }
function shiftTo(c) {      // move the packet so that it starts at cell c (the branch packet is built at cell 0)
  const { re, im } = R.eng, n = re.length, nr = new Float64Array(n), ni = new Float64Array(n);
  for (let x = 0; x < R.L; x++) { const y = (x + c) % R.L; for (let k = 0; k < 4; k++) { nr[4 * y + k] = re[4 * x + k]; ni[4 * y + k] = im[4 * x + k]; } }
  R.eng.re.set(nr); R.eng.im.set(ni);
}
function posDist() { const d = new Float64Array(R.L); R.eng.density(d); return d; }
function momAmps() {     // discrete Fourier transform of the four components: amplitudes a[k][comp]
  const L = R.L, { re, im } = R.eng, ar = new Float64Array(4 * L), ai = new Float64Array(4 * L);
  for (let k = 0; k < L; k++) for (let x = 0; x < L; x++) { const ph = -2 * Math.PI * k * x / L, c = Math.cos(ph), s = Math.sin(ph);
    for (let q = 0; q < 4; q++) { const vr = re[4 * x + q], vi = im[4 * x + q]; if (vr === 0 && vi === 0) continue; ar[4 * k + q] += vr * c - vi * s; ai[4 * k + q] += vr * s + vi * c; } }
  const f = 1 / Math.sqrt(L); for (let i = 0; i < 4 * L; i++) { ar[i] *= f; ai[i] *= f; }
  return [ar, ai];
}
function inverse(ar, ai) {
  const L = R.L, re = new Float64Array(4 * L), im = new Float64Array(4 * L);
  for (let x = 0; x < L; x++) for (let k = 0; k < L; k++) { const ph = 2 * Math.PI * k * x / L, c = Math.cos(ph), s = Math.sin(ph);
    for (let q = 0; q < 4; q++) { const vr = ar[4 * k + q], vi = ai[4 * k + q]; if (vr === 0 && vi === 0) continue; re[4 * x + q] += vr * c - vi * s; im[4 * x + q] += vr * s + vi * c; } }
  const f = 1 / Math.sqrt(L); for (let i = 0; i < 4 * L; i++) { re[i] *= f; im[i] *= f; }
  R.eng.re.set(re); R.eng.im.set(im);
}
const kOf = (k) => 2 * Math.PI * (k < R.L / 2 ? k : k - R.L) / R.L;
function gClock(k, j) { return realClockAmplitude(kOf(k), j, R.clock); } // Complete translation-phase POVM with declared real Kraus amplitudes.
function recordPosition() {
  const d = posDist(), w = R.det, nb = Math.ceil(R.L / w), pb = new Float64Array(nb); for (let x = 0; x < R.L; x++) pb[Math.floor(x / w)] += d[x];
  let u = Math.random() * pb.reduce((s, v) => s + v, 0), b = 0; while (b < nb - 1 && u > pb[b]) { u -= pb[b]; b++; }
  for (let x = 0; x < R.L; x++) if (Math.floor(x / w) !== b) for (let q = 0; q < 4; q++) { R.eng.re[4 * x + q] = 0; R.eng.im[4 * x + q] = 0; }
  R.eng.normalize(); const cell = b * w + (w - 1) / 2; R.recs.unshift(`tick ${R.t}: position recorded, cells ${b * w} to ${Math.min(R.L, b * w + w) - 1} (probability ${(100 * pb[b]).toFixed(1)} %)`); R.hits.push([R.t, cell]);
}
function recordMomentum() {
  const [ar, ai] = momAmps(), W = new Float64Array(R.L); for (let k = 0; k < R.L; k++) for (let q = 0; q < 4; q++) W[k] += ar[4 * k + q] ** 2 + ai[4 * k + q] ** 2;
  const Mc = R.clock, P = new Float64Array(Mc); for (let j = 0; j < Mc; j++) for (let k = 0; k < R.L; k++) P[j] += gClock(k, j) ** 2 * W[k];
  let u = Math.random() * P.reduce((s, v) => s + v, 0), j = 0; while (j < Mc - 1 && u > P[j]) { u -= P[j]; j++; }
  for (let k = 0; k < R.L; k++) { const g = gClock(k, j); for (let q = 0; q < 4; q++) { ar[4 * k + q] *= g; ai[4 * k + q] *= g; } }
  inverse(ar, ai); R.eng.normalize(); let pj = 2 * Math.PI * j / Mc; if (pj > Math.PI) pj -= 2 * Math.PI;
  R.recs.unshift(`tick ${R.t}: translation phase recorded by a ${Mc}-outcome detector: ${pj.toFixed(3)} per cell (probability ${(100 * P[j]).toFixed(1)} %)`);
}
function recTick() { R.eng.tickAlt(R.theta); R.t++; if (R.track && R.t % R.det === 0) recordPosition(); }
function bars(cv, vals, col, xlab, mark) {
  const g = cv.getContext('2d'), W = cv.width, H = cv.height, n = vals.length; g.fillStyle = '#060a14'; g.fillRect(0, 0, W, H);
  const mx = Math.max(1e-12, ...vals); g.fillStyle = col; for (let i = 0; i < n; i++) { const h = (H - 22) * vals[i] / mx; g.fillRect(i * W / n, H - 16 - h, Math.max(1, W / n - 0.5), h); }
  g.fillStyle = 'rgba(255,255,255,.6)'; g.font = '12px ui-monospace,Menlo,monospace'; g.fillText(xlab, 8, H - 3); if (mark) mark(g, W, H);
}
function recDraw() {
  const d = posDist(); bars($('st-rec-x'), Array.from(d), '#ffe45c', 'where it would be found, cell by cell');
  const [ar, ai] = momAmps(), W = new Array(R.L).fill(0); for (let k = 0; k < R.L; k++) for (let q = 0; q < 4; q++) W[k] += ar[4 * k + q] ** 2 + ai[4 * k + q] ** 2;
  const order = Array.from({ length: R.L }, (_, i) => (i + R.L / 2) % R.L), Wd = order.map((k) => W[k]);
  bars($('st-rec-p'), Wd, '#b197ff', 'momentum, from −π to π per cell', (g, Wc, H) => { g.strokeStyle = 'rgba(255,255,255,.3)'; g.beginPath(); g.moveTo(Wc / 2, 0); g.lineTo(Wc / 2, H - 16); g.stroke(); });
  const cv = $('st-rec-t'), g = cv.getContext('2d'), Wt = cv.width, Ht = cv.height; g.fillStyle = '#060a14'; g.fillRect(0, 0, Wt, Ht);
  g.fillStyle = 'rgba(255,255,255,.6)'; g.font = '12px ui-monospace,Menlo,monospace'; g.fillText('recorded positions: cell across, tick down', 8, 14);
  const Tm = Math.max(190, R.t); g.fillStyle = '#ffe45c'; for (const [t, x] of R.hits) { g.beginPath(); g.arc(x / R.L * Wt, 20 + (Ht - 26) * t / Tm, 3.5, 0, 2 * Math.PI); g.fill(); }
  const E = Math.acos(Math.cos(R.theta) * Math.cos(R.p0)), vg = Math.cos(R.theta) * Math.sin(R.p0) / Math.sin(E);
  let fit = ''; if (R.hits.length >= 3) { const n = R.hits.length, mt = R.hits.reduce((s, h) => s + h[0], 0) / n, mx = R.hits.reduce((s, h) => s + h[1], 0) / n; let sxy = 0, sxx = 0; for (const [t, x] of R.hits) { sxy += (t - mt) * (x - mx); sxx += (t - mt) ** 2; } fit = (sxy / sxx).toFixed(3); }
  $('st-rec-num').innerHTML = `<table class="t" style="width:100%"><tr><td>tick</td><td class="num">${R.t}</td></tr><tr><td>speed from the recorded track</td><td class="num">${fit || '—'}</td></tr><tr><td>speed the tick predicts for the starting momentum</td><td class="num">${vg.toFixed(3)}</td></tr><tr><td>rest angle</td><td class="num">${R.theta}</td></tr></table>
    <div class="st-log">${R.recs.slice(0, 9).map((r) => `<div>${r}</div>`).join('')}</div>`;
}

// ---------------------------------------------------------------- 4. its mass
const MASS = [
  ['photon', 'assigned class 0', 'unbroken electromagnetic mode', 'massless in the specified classical gauge construction; the quantum continuum transfer is separate', '0', 'open'],
  ['gluons', 'assigned class 0', 'unbroken colour modes', 'massless classical gauge modes; confinement and hadron poles are separate calculations', '0', 'open'],
  ['W and Z', 'assigned class 0', 'chosen scalar background', 'background-dependent gauge mixing; the physical scale is calibrated', '80.4 and 91.2 GeV', 'open'],
  ['Higgs', 'candidate class 3', 'declared scalar dynamics', 'a scalar register or background does not alone fix a physical Higgs pole', '125.2 GeV', 'open'],
  ['up type: u, c, t', 'candidate classes 1 and 4', 'specified Yukawa operator Y', 'neutral-background benchmark m=|vY|; operator, family occupations and unit remain inputs', '2.16 MeV · 1.27 GeV · 172.6 GeV', 'open'],
  ['down type: d, s, b', 'candidate classes 1 and 4', 'specified Yukawa operator Y', 'same neutral-background benchmark; a full representation is more than its centre class', '4.70 MeV · 93.5 MeV · 4.18 GeV', 'open'],
  ['charged lepton: e, μ, τ', 'candidate classes 3 and 0', 'one-tick reading or a specified Hamiltonian', 'FR one-tick ratios are conditional; the reused second tick is scalar. See the exact clock comparison below.', '0.511 MeV · 105.7 MeV · 1.777 GeV', 'open'],
  ['neutrino: physical target', 'candidate classes 3 and 0', 'specified neutral Yukawa sector', 'no mass spectrum or independent generation construction follows from the centre classes', 'absolute masses unresolved', 'open'],
];

// ---------------------------------------------------------------- shell
function loop() {
  requestAnimationFrame(loop);
  if (!$('page-story').classList.contains('on')) return;
  const now = performance.now();
  if (B.running && now - B.last > 420) { B.last = now; bornStep(); if (B.quiet > 25) bornNew(); bornDraw(); }
  B.yaw += 0.003; if (!B.running) bornDraw();
  if (M.running) { for (let i = 0; i < 2; i++) massTick(); massDraw(); if (M.t >= M.T) { M.running = false; $('st-mass-run').textContent = '▶ run'; } }
  if (R.running) { recTick(); if (R.t % 2 === 0) recDraw(); if (R.t > 190) { R.running = false; $('st-rec-run').textContent = '▶ run'; } }
}
export function storyShown() {
  if (inited) { bornDraw(); massDraw(); recDraw(); lkShown(); return; }
  inited = true; lkInit();
  $('st-born-new').onclick = () => bornNew(); $('st-born-step').onclick = () => { bornStep(); bornDraw(); };
  $('st-born-run').onclick = () => { B.running = !B.running; $('st-born-run').textContent = B.running ? '❚❚ pause' : '▶ run'; };
  $('st-born-kick').onclick = () => disturb();
  $('st-mass-theta').onchange = (e) => { M.theta = parseFloat(e.target.value); massNew(); };
  $('st-mass-run').onclick = () => { if (M.t >= M.T) massNew(); M.running = !M.running; $('st-mass-run').textContent = M.running ? '❚❚ pause' : '▶ run'; };
  $('st-mass-reset').onclick = () => { M.running = false; $('st-mass-run').textContent = '▶ run'; massNew(); };
  $('st-rec-run').onclick = () => { R.running = !R.running; $('st-rec-run').textContent = R.running ? '❚❚ pause' : '▶ run'; };
  $('st-rec-reset').onclick = () => { R.running = false; $('st-rec-run').textContent = '▶ run'; recNew(); };
  $('st-rec-pos').onclick = () => { recordPosition(); recDraw(); }; $('st-rec-mom').onclick = () => { recordMomentum(); recDraw(); };
  $('st-rec-track').onchange = (e) => { R.track = e.target.checked; };
  $('st-rec-det').onchange = (e) => { R.det = parseInt(e.target.value); }; $('st-rec-clock').onchange = (e) => { R.clock = parseInt(e.target.value); };
  $('st-mass-table').innerHTML = '<tr><th>particle target</th><th>candidate centre class</th><th>declared mechanism</th><th>exact scope and remaining input</th><th>comparison targets</th></tr>' +
    MASS.map((r) => `<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td>${r[2]}</td><td><span class="cos-tag ${r[5]}">${r[3]}</span></td><td class="mono">${r[4]}</td></tr>`).join('');
  const svg = $('st-born-svg'); let drag = null;
  svg.addEventListener('pointerdown', (e) => { drag = [e.clientX, e.clientY, B.yaw, B.pitch]; svg.setPointerCapture(e.pointerId); });
  svg.addEventListener('pointermove', (e) => { if (!drag) return; B.yaw = drag[2] + (e.clientX - drag[0]) * 0.008; B.pitch = Math.max(-1.4, Math.min(1.4, drag[3] + (e.clientY - drag[1]) * 0.008)); bornDraw(); });
  svg.addEventListener('pointerup', () => { drag = null; });
  bornNew(); massNew(); recNew();
  B.running = true; $('st-born-run').textContent = '❚❚ pause'; M.running = true; $('st-mass-run').textContent = '❚❚ pause';
  R.track = true; $('st-rec-track').checked = true; R.running = true; $('st-rec-run').textContent = '❚❚ pause';
  const q = new URLSearchParams(location.search);
  if (q.has('stsweeps')) { for (let i = 0; i < parseInt(q.get('stsweeps')); i++) bornStep(); bornDraw(); }
  if (q.has('stmass')) { for (let i = 0; i < parseInt(q.get('stmass')); i++) massTick(); massDraw(); }
  if (q.has('sttrack')) { R.track = true; $('st-rec-track').checked = true; for (let i = 0; i < parseInt(q.get('sttrack')); i++) recTick(); recDraw(); }
  loop();
}
window.__story = { B, M, R };
