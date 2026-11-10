// The Carrier tab: the public coherent programs of the theory repository on one twelve-port carrier.
// Engine and checks: walk_engine.js, carrier_check.mjs. Nothing here is fitted or tuned; every number is computed.
import { $ } from './app.js';
import { Walk, PV, EDGES, FACES, FACE_EDGES, ADDR, addrIndex, edgeIndex, regionOf, reachable, faceFlux, committedStep, bit } from './walk_engine.js';

const COL = { scalar: '#3fe0d0', vector: '#ff7a45', spinor: '#b197ff', flux: '#ffe45c', neg: '#ff5f7a', committed: '#8f7cff' };
const SCEN = {
  wave: { title: 'A wave instead of a relaxation', fibre: 1,
    text: 'One load quantum starts at a port. The committed rule spreads it out and never brings it back. The coherent walk keeps the removed disagreement in a record that the observer owns, and the quantum returns: every relaxation factor λ of the committed rule is now a frequency, cos ω = λ.' },
  types: { title: 'Two types of quantum', fibre: 5,
    text: 'The quantum carries five internal levels: three vector levels and two spinor levels. The walk never mixes the two kinds, so they are two conserved types. A flipped link (red) is invisible to the vector type and changes the sign of the spinor type. Click a seam to flip its link and watch the two types part.' },
  defect: { title: 'A defect that the wave moves', fibre: 5,
    text: 'The links are now a register of thirty bits. A flipped link puts flux on its two faces (yellow). Inside the region around port 5 the flux cannot be repaired away while the boundary is held, and it moves when the wave is at an end of an interior seam. The wave in turn feels where the flux sits.' },
  record: { title: 'Recording changes the result', fibre: 5,
    text: 'A spinor quantum is prepared so that after one tick it sits at ports 5 and 6 at once. At port 5 it moves the flux, at port 6 it does not, so the wave and the flux are tied together. Reversing the coupling brings the wave back with certainty. Reading where the flux is first, and then reversing, brings it back half of the time. Everything stays inside the gauge-invariant sector.' },
};
const C = { scen: 'wave', port: 0, level: 'mix', word: 1, running: false, yaw: 0.5, pitch: 0.35, hist: [], w: null, com: null, region: null, checks: null, timer: 0 };

function levelsOf() { if (SCEN[C.scen].fibre === 1) return [[0, 1]]; if (C.level === 'mix') return [[0, Math.SQRT1_2], [3, Math.SQRT1_2]]; return [[parseInt(C.level), 1]]; }
function buildRecord() {
  C.region = regionOf(5);
  const words = reachable(1, C.region.interior), g = words.indexOf(1);
  let w = new Walk({ fibre: 5, words, interior: C.region.interior });
  w.embed(g, 5, 3, Math.SQRT1_2); w.embed(g, 6, 3, Math.SQRT1_2); w.unstep();
  w = w.gaussProject(5, 1); w.normalize();
  C.w = w; C.psi = new Walk({ fibre: 5, words: [1] }); C.psi.embed(0, 5, 3, Math.SQRT1_2); C.psi.embed(0, 6, 3, Math.SQRT1_2);
  C.stage = 0; C.mix = null; C.ret = null; C.rows = null; C.port = 5; C.com = new Float64Array(12);
  C.hist = []; record(); draw();
}
function recordAction(what) {
  if (what === 'prep') return buildRecord();
  if (what === 'tick' && C.stage === 0) { C.w.tick(); C.stage = 1; record(); }
  else if (what === 'rev' && C.stage === 1) { C.w.unmoves(); C.ret = C.w.returnProbability(C.psi, 5); C.stage = 2; record(); }
  else if (what === 'look' && C.stage === 1) {
    C.rows = []; C.mix = []; C.ret = 0;
    for (const br of C.w.faceBranches(C.region.faces)) {
      if (br.p < 1e-14) continue;
      br.state.unmoves(); const r = br.state.returnProbability(C.psi, 5);
      C.rows.push({ face: FACES[br.face].join('·'), p: br.p, yes: r, no: br.p - r }); C.mix.push(br.state); C.ret += r;
    }
    C.stage = 3;
  }
  draw();
}
function pops() {
  if (!C.mix) return C.w.populations();
  const out = { port: new Float64Array(12), type: new Float64Array(2), link: new Float64Array(C.w.NG), portType: new Float64Array(24) };
  for (const s of C.mix) { const P = s.populations(); for (const k of Object.keys(out)) for (let i = 0; i < out[k].length; i++) out[k][i] += P[k][i]; }
  return out;
}
function build() {
  if (C.scen === 'record') return buildRecord();
  C.mix = null;
  const sc = SCEN[C.scen];
  if (C.scen === 'defect') {
    C.region = regionOf(5);
    const words = reachable(1, C.region.interior);
    C.w = new Walk({ fibre: 5, words, interior: C.region.interior });
    for (const [j, a] of levelsOf()) C.w.embed(words.indexOf(1), C.port, j, a);
  } else {
    C.region = null;
    C.w = new Walk({ fibre: sc.fibre, words: [C.scen === 'types' ? C.word : 0] });
    for (const [j, a] of levelsOf()) C.w.embed(0, C.port, j, a);
  }
  C.w.normalize();
  C.com = new Float64Array(12); C.com[C.port] = 1;
  C.hist = []; record(); draw();
}
function record() {
  const P = pops();
  const row = { t: C.w.ticks, ret: P.port[C.port], com: C.com[C.port], v: P.portType[C.port * 2], s: P.portType[C.port * 2 + 1] };
  if (C.region) { const fp = C.w.faceProbabilities(C.region.faces); row.faces = C.region.faces.map((f) => fp[f]); }
  C.hist.push(row); if (C.hist.length > 400) C.hist.shift();
}
function step() { C.w.tick(); C.com = committedStep(C.com); record(); }

// ---------------------------------------------------------------- drawing
function project() {
  const cy = Math.cos(C.yaw), sy = Math.sin(C.yaw), cp = Math.cos(C.pitch), sp = Math.sin(C.pitch), S = 148;
  return PV.map(([x, y, z]) => { const x1 = cy * x + sy * z, z1 = -sy * x + cy * z, y1 = cp * y - sp * z1, z2 = sp * y + cp * z1; return [380 + S * x1, 300 - S * y1, z2]; });
}
function draw() {
  const host = $('car-svg'); if (!host || !C.w) return;
  const Q = project(), P = pops(), sc = SCEN[C.scen], fib5 = sc.fibre === 5;
  const flux = new Float64Array(20);
  if (C.scen === 'defect' || C.scen === 'record') { for (let g = 0; g < C.w.NG; g++) { const fl = faceFlux(C.w.words[g]); for (let f = 0; f < 20; f++) if (fl[f]) flux[f] += P.link[g]; } }
  else if (C.scen === 'types') { const fl = faceFlux(C.word); for (let f = 0; f < 20; f++) flux[f] = fl[f]; }
  const depth = (ids) => ids.reduce((s, i) => s + Q[i][2], 0) / ids.length;
  let h = '<defs><filter id="glow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>';
  const faces = FACES.map((f, i) => ({ f, i, z: depth(f) })).sort((a, b) => a.z - b.z);
  for (const { f, i, z } of faces) {
    const front = z > 0, pts = f.map((p) => Q[p][0].toFixed(1) + ',' + Q[p][1].toFixed(1)).join(' ');
    const inRegion = C.region && C.region.faces.includes(i);
    const fill = flux[i] > 1e-4 ? COL.flux : (inRegion ? '#4cc3ff' : '#9fb4e0');
    const op = flux[i] > 1e-4 ? (front ? 0.14 + 0.6 * flux[i] : 0.05 + 0.2 * flux[i]) : (inRegion ? (front ? 0.10 : 0.04) : (front ? 0.045 : 0.015));
    h += `<polygon points="${pts}" fill="${fill}" fill-opacity="${op.toFixed(3)}" stroke="none"/>`;
  }
  const seams = EDGES.map((e, i) => ({ e, i, z: depth(e) })).sort((a, b) => a.z - b.z);
  for (const { e, i, z } of seams) {
    const neg = C.scen === 'types' && bit(C.word, i), o = z > 0 ? 0.75 : 0.22;
    const bd = C.region && C.region.boundary.includes(i);
    h += `<line x1="${Q[e[0]][0].toFixed(1)}" y1="${Q[e[0]][1].toFixed(1)}" x2="${Q[e[1]][0].toFixed(1)}" y2="${Q[e[1]][1].toFixed(1)}" stroke="${neg ? COL.neg : bd ? '#4cc3ff' : '#c5d0e6'}" stroke-opacity="${o}" stroke-width="${neg ? 3.2 : bd ? 2.2 : 1.2}" ${neg ? 'stroke-dasharray="7 5"' : ''}/>`;
    if (C.scen === 'types') h += `<line class="car-hit" data-e="${i}" x1="${Q[e[0]][0].toFixed(1)}" y1="${Q[e[0]][1].toFixed(1)}" x2="${Q[e[1]][0].toFixed(1)}" y2="${Q[e[1]][1].toFixed(1)}" stroke="transparent" stroke-width="16" style="cursor:pointer"/>`;
  }
  const ports = PV.map((_, p) => p).sort((a, b) => Q[a][2] - Q[b][2]);
  for (const p of ports) {
    const [x, y, z] = Q[p], o = z > 0 ? 1 : 0.42, rmax = 46;
    h += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4" fill="#0b1220" stroke="#c5d0e6" stroke-opacity="${0.7 * o}" stroke-width="1.2"/>`;
    if (!fib5) { const r = rmax * Math.sqrt(P.port[p]); if (r > 0.6) h += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${COL.scalar}" fill-opacity="${0.55 * o}" stroke="${COL.scalar}" stroke-opacity="${o}" filter="url(#glow)"/>`; }
    else {
      const rv = rmax * Math.sqrt(P.portType[p * 2]), rs = rmax * Math.sqrt(P.portType[p * 2 + 1]);
      if (rv > 0.6) h += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${rv.toFixed(1)}" fill="${COL.vector}" fill-opacity="${0.42 * o}" stroke="${COL.vector}" stroke-opacity="${o}" filter="url(#glow)"/>`;
      if (rs > 0.6) h += `<rect x="${(x - rs).toFixed(1)}" y="${(y - rs).toFixed(1)}" width="${(2 * rs).toFixed(1)}" height="${(2 * rs).toFixed(1)}" rx="5" fill="${COL.spinor}" fill-opacity="${0.20 * o}" stroke="${COL.spinor}" stroke-opacity="${o}" stroke-width="2.2" transform="rotate(45 ${x.toFixed(1)} ${y.toFixed(1)})" filter="url(#glow)"/>`;
    }
    h += `<text x="${(x + 9).toFixed(1)}" y="${(y - 9).toFixed(1)}" fill="#c5d0e6" fill-opacity="${0.85 * o}" font-size="12" font-family="ui-monospace,Menlo,monospace">${p}</text>`;
  }
  host.innerHTML = h;
  host.querySelectorAll('.car-hit').forEach((el) => el.onclick = () => { C.word = (C.word ^ (1 << parseInt(el.dataset.e))) >>> 0; build(); });
  plot(); numbers(P, flux);
}
function plot() {
  const cv = $('car-plot'); if (!cv) return;
  const W = cv.width = cv.clientWidth * 2, H = cv.height = cv.clientHeight * 2, g = cv.getContext('2d');
  g.clearRect(0, 0, W, H); g.lineWidth = 2; g.font = '22px ui-monospace,Menlo,monospace';
  const n = Math.max(60, C.hist.length), X = (i) => 60 + (W - 80) * i / (n - 1), Y = (v) => H - 44 - (H - 70) * v;
  g.strokeStyle = 'rgba(197,208,230,.25)'; g.fillStyle = '#93a3c0';
  for (const v of [0, 0.5, 1]) { g.beginPath(); g.moveTo(60, Y(v)); g.lineTo(W - 20, Y(v)); g.stroke(); g.fillText(v.toFixed(1), 8, Y(v) + 8); }
  const line = (key, col, dash) => { g.strokeStyle = col; g.setLineDash(dash || []); g.beginPath(); C.hist.forEach((r, i) => { const v = typeof key === 'function' ? key(r) : r[key]; i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)); }); g.stroke(); g.setLineDash([]); };
  let legend;
  if (C.scen === 'wave') { line('com', COL.committed, [10, 8]); line('ret', COL.scalar); legend = [['coherent walk', COL.scalar], ['committed rule', COL.committed]]; }
  else if (C.scen === 'types') { line('v', COL.vector); line('s', COL.spinor); legend = [['vector type at the starting port', COL.vector], ['spinor type at the starting port', COL.spinor]]; }
  else if (C.scen === 'record') { legend = [['one tick, then the choice on the right', '#93a3c0']]; }
  else { const cols = ['#ffe45c', '#ffb347', '#ff7a45', '#ff5fb0', '#b197ff']; C.region.faces.forEach((f, i) => line((r) => r.faces[i], cols[i])); legend = C.region.faces.map((f, i) => ['flux on face ' + FACES[f].join('·'), cols[i]]); }
  let x = 70; for (const [name, col] of legend) { g.fillStyle = col; g.fillRect(x, 14, 22, 6); g.fillStyle = '#c5d0e6'; g.fillText(name, x + 30, 24); x += 44 + g.measureText(name).width; }
  g.fillStyle = '#93a3c0'; g.fillText('walk steps →', W - 190, H - 10);
}
function numbers(P, flux) {
  const el = $('car-numbers'); if (!el) return;
  const row = (a, b) => `<div class="row"><span>${a}</span><b>${b}</b></div>`;
  if (C.scen === 'record') {
    const stage = ['prepared: the wave is one tick away from ports 5 and 6', 'after one tick: wave at ports 5 and 6, flux on two faces at once', 'reversed without a record', 'flux read, then reversed'][C.stage];
    let h = row('stage', stage) + row('wave at port 5 · port 6', P.port[5].toFixed(4) + ' · ' + P.port[6].toFixed(4));
    h += row('flux on face 0·1·5 · face 4·5·11', flux[FACES.findIndex((f) => f.join() === '0,1,5')].toFixed(4) + ' · ' + flux[FACES.findIndex((f) => f.join() === '4,5,11')].toFixed(4));
    if (C.ret !== null) h += row('<b>the wave comes back</b>', `<span style="color:${C.stage === 2 ? 'var(--green,#5fe3a1)' : 'var(--amber,#ffb44a)'};font-size:16px">${C.ret.toFixed(6)}</span>`);
    if (C.rows) for (const r of C.rows) h += row('record: flux on face ' + r.face, 'back ' + r.yes.toFixed(4) + ' · not back ' + r.no.toFixed(4));
    if (C.stage === 1) h += '<div class="note">Now choose: reverse the coupling as it is, or read the flux first.</div>';
    el.innerHTML = h; return;
  }
  let h = row('walk steps', C.w.ticks) + row('at the starting port', P.port[C.port].toFixed(6)) + row('a correction is on record', C.w.recordWeight().toFixed(6) + (C.w.ticks === 1 ? ' · after one tick this is 1 − cos ω' : ''));
  if (C.scen === 'wave') h += row('committed rule, same number of attempts', C.com[C.port].toFixed(6)) + row('farthest port', P.port[[...PV.keys()].find((q) => PV[q].every((v, i) => Math.abs(v + PV[C.port][i]) < 1e-9))].toFixed(6));
  else {
    h += row(`<span style="color:${COL.vector}">●</span> vector type, charge −2 per level`, P.type[0].toFixed(6) + ' · conserved') + row(`<span style="color:${COL.spinor}">◆</span> spinor type, charge +3 per level`, P.type[1].toFixed(6) + ' · conserved');
    if (C.scen === 'types') { const n = EDGES.filter((_, i) => bit(C.word, i)).length, nf = flux.reduce((a, b) => a + b, 0); h += row('flipped links', n) + row('faces with flux', nf); }
    else { const inside = C.region.faces.reduce((s, f) => s + flux[f], 0); h += row('flux inside the region around port 5', inside.toFixed(6) + ' · odd, protected') + row('link words in superposition', [...P.link].filter((v) => v > 1e-9).length + ' of ' + C.w.NG); }
  }
  el.innerHTML = h;
}
function checks() {
  if (C.checks) return C.checks;
  const out = [], add = (name, v, exact, f) => out.push({ name, v, exact, ok: Math.abs(v - f) < 1e-12 });
  let w = new Walk({ fibre: 1 }); w.embed(0, 0, 0); w.tick(); add('a correction in the first tick, from one port', w.recordWeight(), '1/12', 1 / 12); w.tick(); add('back at the port after two steps', w.populations().port[0], '643/900', 643 / 900);
  const amp = (word, j) => { const x = new Walk({ fibre: 5, words: [word] }); x.embed(0, 0, j); x.tick(); x.tick(); x.tick(); return x.overlap(0, 0, j)[0]; };
  add('one flipped link, spinor type, three-step return amplitude', amp(1, 3) - amp(0, 3), '−1/6750', -1 / 6750);
  add('the same for the vector type', amp(1, 0) - amp(0, 0), '0', 0);
  const R = regionOf(5), words = reachable(1, R.interior);
  const rec = (g0) => { const x = new Walk({ fibre: 5, words, interior: R.interior }); const g = words.indexOf(g0); x.embed(g, 0, 3, Math.SQRT1_2); x.embed(g, 5, 3, Math.SQRT1_2); x.tick(); return x.basisProbability(0, 3, 0, addrIndex(0, 5)); };
  add('wave record with the flux on face 0·1·5', rec(1), '1/120', 1 / 120); add('the same record with the flux on face 0·5·11', rec(3), '0', 0);
  const x = new Walk({ fibre: 5, words, interior: R.interior }); x.embed(words.indexOf(1), 5, 3); x.unstep(); x.tick();
  add('a wave arriving at port 5 moves the flux to face 4·5·11', x.faceProbabilities(R.faces)[FACES.findIndex((f) => f.join() === '4,5,11')], '1', 1);
  {
    const psi = new Walk({ fibre: 5, words: [1] }); psi.embed(0, 5, 3, Math.SQRT1_2); psi.embed(0, 6, 3, Math.SQRT1_2);
    const prep = () => { let y = new Walk({ fibre: 5, words, interior: R.interior }); const g = words.indexOf(1); y.embed(g, 5, 3, Math.SQRT1_2); y.embed(g, 6, 3, Math.SQRT1_2); y.unstep(); y = y.gaussProject(5, 1); y.normalize(); return y; };
    const a = prep(); a.tick(); a.unmoves(); add('reversing the coupling brings the wave back', a.returnProbability(psi, 5), '1', 1);
    const b = prep(); b.tick(); let ret = 0; for (const br of b.faceBranches(R.faces)) if (br.p > 1e-14) { br.state.unmoves(); ret += br.state.returnProbability(psi, 5); }
    add('the same after reading where the flux is', ret, '1/2', 0.5);
  }
  return (C.checks = out);
}
function side() {
  const sc = SCEN[C.scen], fib5 = sc.fibre === 5;
  $('car-scen-text').textContent = sc.text;
  $('car-level-row').style.display = fib5 ? 'flex' : 'none';
  $('car-rec').style.display = C.scen === 'record' ? 'block' : 'none'; $('car-std').style.display = C.scen === 'record' ? 'none' : 'block';
  $('car-legend').innerHTML = fib5
    ? `<span class="lg"><span style="color:${COL.vector};font-size:20px">●</span><span><b style="color:${COL.vector}">vector type</b><br><span class="small">three levels · charge −2 each · blind to a flipped link · best guess: the colour index of matter</span></span></span>
       <span class="lg"><span style="color:${COL.spinor};font-size:20px">◆</span><span><b style="color:${COL.spinor}">spinor type</b><br><span class="small">two levels · charge +3 each · a flipped link changes its sign · best guess: the weak doublet index of matter</span></span></span>
       <span class="lg"><span style="color:${COL.flux};font-size:20px">▲</span><span><b style="color:${COL.flux}">flux</b><br><span class="small">best guess: a flux of the weak centre · a face whose three links multiply to −1${(C.scen === 'defect' || C.scen === 'record') ? ' · brightness is its probability' : ''}</span></span></span>`
    : `<span class="lg"><span style="color:${COL.scalar};font-size:20px">●</span><span><b style="color:${COL.scalar}">load quantum</b><br><span class="small">disc area is the probability to find it at the port</span></span></span>`;
  $('car-checks').innerHTML = checks().map((c) => `<div class="row"><span>${c.name}</span><b style="color:${c.ok ? 'var(--green,#5fe3a1)' : '#ff6480'}">${c.exact} ${c.ok ? '✓' : '✗ ' + c.v.toPrecision(8)}</b></div>`).join('');
}
function run(on) {
  C.running = on; $('car-run').textContent = on ? '❚❚ pause' : '▶ run';
  clearInterval(C.timer);
  if (on) C.timer = setInterval(() => { step(); draw(); }, parseInt($('car-speed').value));
}
let inited = false;
export function carrierShown() {
  if (!inited) {
    inited = true;
    $('car-scen').onchange = (e) => { C.scen = e.target.value; C.word = 1; run(false); if (C.scen === 'defect') { C.port = 6; $('car-port').value = '6'; C.level = '3'; $('car-level').value = '3'; } else if (C.scen === 'types') { C.level = 'mix'; $('car-level').value = 'mix'; C.port = 0; $('car-port').value = '0'; } side(); build(); };
    $('car-port').onchange = (e) => { C.port = parseInt(e.target.value); build(); };
    $('car-level').onchange = (e) => { C.level = e.target.value; build(); };
    for (const k of ['prep', 'tick', 'rev', 'look']) $('car-rec-' + k).onclick = () => recordAction(k);
    $('car-reset').onclick = () => { run(false); build(); };
    $('car-step').onclick = () => { run(false); step(); draw(); };
    $('car-run').onclick = () => run(!C.running);
    $('car-speed').onchange = () => run(C.running);
    const svg = $('car-svg'); let drag = null;
    svg.addEventListener('pointerdown', (e) => { if (e.target.classList.contains('car-hit')) return; drag = [e.clientX, e.clientY, C.yaw, C.pitch]; svg.setPointerCapture(e.pointerId); });
    svg.addEventListener('pointermove', (e) => { if (!drag) return; C.yaw = drag[2] + (e.clientX - drag[0]) * 0.008; C.pitch = Math.max(-1.4, Math.min(1.4, drag[3] + (e.clientY - drag[1]) * 0.008)); draw(); });
    svg.addEventListener('pointerup', () => { drag = null; });
    const q = new URLSearchParams(location.search);
    if (q.has('carscen') && SCEN[q.get('carscen')]) { C.scen = q.get('carscen'); $('car-scen').value = C.scen; if (C.scen === 'defect') { C.port = 6; $('car-port').value = '6'; C.level = '3'; $('car-level').value = '3'; } }
    if (q.has('carword')) C.word = parseInt(q.get('carword')) >>> 0;
    if (q.has('carport')) { C.port = parseInt(q.get('carport')); $('car-port').value = String(C.port); }
    side(); build();
    if (q.has('carsteps') && C.scen !== 'record') { for (let i = 0; i < parseInt(q.get('carsteps')); i++) step(); draw(); }
    if (q.has('caract')) for (const k of q.get('caract').split(',')) recordAction(k);
  } else draw();
}
window.__carrier = { C, step, draw, build };
