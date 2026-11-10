// The Space tab: quanta moving through a cubic family of carriers, run in the browser with the routing tick of the
// candidate program CF-1 (repair-law/candidates/CF1.md). Engine: cf1_engine.js, checked by `node cf1_check.mjs`.
// Rest angles are imported into this declared walk from one-carrier calculations
// (repair-law/work/DIRAC/defect_phases.json) or the declared face-count angle of the audit's routed flux pair.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { $ } from './app.js';
import { Routing, branchCoin } from './cf1_engine.js';
import { forcesInit, forcesShown } from './space_forces.js';   // section "Force carriers on the same cone" (GL, T3)

// rest angle of a pattern on a carrier that holds a flux pair: REST[alpha in sixths][face separation - 1]
const REST = { 1: [0.042383, 0.051014, 0.056207, 0.058279, 0.059322], 2: [0.075113, 0.092788, 0.106416, 0.113080, 0.116921], 3: [0.087847, 0.110554, 0.132363, 0.147787, 0.171237] };
const SHARES = [0.241, 0.196, 0.127, 0.196, 0.241];             // shares of the flux classes 1..5 over run L10 (Particles tab, census)
const CLS = [
  { key: 'qd', name: 'quark-doublet candidate', who: 'q6 = 1', q6: 1, col: '#ff7a45', charged: true },
  { key: 'qs', name: 'quark-singlet candidate', who: 'q6 = 2', q6: 2, col: '#ff5f7a', charged: false },
  { key: 'ld', name: 'lepton-doublet candidate', who: 'q6 = 3', q6: 3, col: '#b197ff', charged: true },
  { key: 'ls', name: 'neutral-centre candidate', who: 'q6 = 0; charge not resolved', q6: 0, col: '#3fe0d0', charged: false },
];
const FLUXCOL = ['#ffe45c', '#7dd3fc', '#f0abfc', '#7dd3fc', '#ffe45c'];
const alphaOf = (q6, k) => { const a = ((k * q6) % 6 + 6) % 6; return Math.min(a, 6 - a); };
const restAngle = (q6, k, d) => { const a = alphaOf(q6, k); return a === 0 || k === 0 ? 0 : REST[a][d - 1]; };
const BLANK = [[1, 0], [0, 0], [0, 0], [0, 0]];

const EXP = {
  cone: { title: '1 · Light and heavy',
    text: 'Three prepared wave packets start with zero mean momentum. The massless walk has axial speed one cell per tick; the finite three-dimensional dispersion is direction dependent. The two chosen rest angles change spreading. The flux-pair curve uses a declared protected code and routing memory, not motion of a classical-run defect.' },
  race: { title: '2 · The species in a flux background',
    text: 'Every carrier is assigned the same static flux background. Four centre-charge candidates receive different imported rest phases and group velocities. A candidate with zero phase has axial speed one. Centre charge alone does not identify a quark or lepton.' },
  gas: { title: '3 · A gas of flux gives a rest phase',
    text: 'The chosen density and rounded L10 class shares prepare a static random background. The walk applies the imported local rest rotations coherently. A short-time record-occupation diagnostic is compared with the sample mean angle; it is not an interacting mass or a detector history from L10.' },
  line: { title: '4 · A flux line between carriers',
    text: 'A prescribed Z2 link background carries a line of centre flux. Odd weak-centre charge reads its link signs; even charge does not. The interference pattern tests this background coupling. The internal weak-doublet label is distinct from the declared spatial spin pointer.' },
  pairs: { title: '6 · Coherence in static centre flux',
    text: 'This experiment prepares random Z3 links at a density motivated by the run. A rigid opposite-charge pair reads u(x) times the conjugate of u(x+s). At zero separation these factors cancel exactly; separated paths lose overlap with the free packet. This is a static-background coherence test, not a bound meson, quantum confinement or proof that charged states cannot travel.' },
  reg: { title: '7 · The mass step with its register',
    text: 'This declared register compensates the charge moved by the handedness-changing step. The two supplied register preparations produce different coherent phases and spreading. The retained constraint is tested in that finite construction. A common register phase is a background choice, not a derived Higgs vacuum or a theorem that all physical masses need this preparation.' },
  fluxobj: { title: '5 · The protected defect itself moves',
    text: 'The audit moves a complete flux pair from carrier to carrier and routes it with the same tick, using two lanes and a declared routing memory. Its rest angle is twice the spacing, from the count of its two flipped faces. Position is read by asking which carrier holds the pair, momentum from the translation of the same code. A second construction of the theory session lets a charged quantum carry a flux pair as cargo: then the pattern gives the charges and the pair gives a conserved label and a rest phase.' },
};
const S = { exp: 'cone', k: 3, d: 5, dens: 0.2, a: 12, running: false, quanta: [], shape: [36, 36, 36], t: 0, hist: [], flux: null, timer: 0, note: '' };
let renderer, scene, camera, controls, points, fluxPts, cage, sprite, inited = false, posBuf, colBuf, runs = null;

function hex(c) { const v = parseInt(c.slice(1), 16); return [(v >> 16) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255]; }
function slab(eng, cx, width, p0, coin) {      // packet along axis 0, uniform across the other two
  const [n1, n2, n3] = eng.n;
  for (let a = 0; a < n1; a++) { let d = a - cx; d = ((d % n1) + n1) % n1; if (d > n1 / 2) d -= n1; const g = Math.exp(-d * d / (2 * width * width)); if (g < 1e-10) continue;
    const cr = Math.cos(p0 * d) * g, ci = Math.sin(p0 * d) * g;
    for (let b = 0; b < n2; b++) for (let c = 0; c < n3; c++) { const o = 4 * ((a * n2 + b) * n3 + c); for (let k = 0; k < 4; k++) { eng.re[o + k] += cr * coin[k][0] - ci * coin[k][1]; eng.im[o + k] += cr * coin[k][1] + ci * coin[k][0]; } } }
  eng.normalize();
}
function disc(eng, c, width, coin) {             // packet at rest, uniform along the third axis: the picture is a cut through the box
  const [n1, n2, n3] = eng.n;
  for (let a = 0; a < n1; a++) for (let b = 0; b < n2; b++) { const g = Math.exp(-((a - c[0]) ** 2 + (b - c[1]) ** 2) / (2 * width * width)); if (g < 1e-10) continue;
    for (let k = 0; k < n3; k++) { const o = 4 * ((a * n2 + b) * n3 + k); for (let j = 0; j < 4; j++) { eng.re[o + j] += g * coin[j][0]; eng.im[o + j] += g * coin[j][1]; } } }
  eng.normalize();
}
function planar(q, c, radius) {                   // mean in-plane distance from the start and the probability inside a radius
  const [n1, n2, n3] = q.eng.n; let s = 0, sr = 0, inside = 0;
  for (let a = 0; a < n1; a++) for (let b = 0; b < n2; b++) { const r = Math.hypot(a - c[0], b - c[1]); let w = 0; for (let k = 0; k < n3; k++) w += q.rho[(a * n2 + b) * n3 + k]; s += w; sr += w * r; if (r <= radius) inside += w; }
  return { mean: sr / s, inside: inside / s };
}
function quantum(label, col, theta, kind = 'Y') { const eng = new Routing(...S.shape), half = typeof theta === 'number' ? theta / 2 : Float64Array.from(theta, (x) => x / 2); return { label, col, rgb: hex(col), eng, theta, half, kind, rho: new Float32Array(eng.N), series: [] }; }

function build() {
  S.t = 0; S.hist = []; S.flux = null; S.note = '';
  const e = S.exp;
  if (e === 'reg') {
    S.shape = [256, 1, 1]; S.regStep = 0;
    const fake = (label, col) => ({ label, col, rgb: hex(col), fake: true, rho: new Float32Array(256) });
    S.quanta = [fake('common-phase register: coherent rest rotation', '#ffe45c'), fake('definite-state register: sector-averaged spreading', '#ff5f7a')];
    if (!S.regData) { fetch('/rl/SIM/cf1/runs/R8_register_line.json').then((r) => r.json()).then((d) => { S.regData = d; if (S.exp === 'reg') { geometry(); draw(); } }).catch(() => {}); }
  } else if (e === 'cone') {
    S.shape = [72, 72, 2]; const c = [36, 36, 0];
    S.quanta = [quantum('declared walk, zero rest angle', '#3fe0d0', 0),
      quantum('q6 = 3 candidate on class-3 flux, separation 5 (0.1712)', '#b197ff', REST[3][4]),
      quantum('protected flux pair, the audit’s two-lane route, spacing π/12 (0.5236)', '#ffe45c', Math.PI / 6, 'Z')];
    for (const q of S.quanta) { disc(q.eng, c, 3.2, BLANK); }
    S.centre = c;
  } else if (e === 'race') {
    S.shape = [192, 2, 2]; const p0 = 2 * Math.PI * 6 / 192;
    S.quanta = CLS.map((cl) => { const th = restAngle(cl.q6, S.k, S.d), q = quantum(`${cl.name} (${cl.who})`, cl.col, th); slab(q.eng, 30, 10, p0, branchCoin([p0, 0, 0], th)); q.p0 = p0; q.x0 = q.eng.meanPosition(0); return q; });
  } else if (e === 'gas') {
    S.shape = [26, 26, 26]; const N = 26 ** 3; let seed = 11; const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
    S.flux = new Uint8Array(N);
    for (let c = 0; c < N; c++) if (rnd() < S.dens) { let u = rnd(), k = 0; while (k < 4 && u > SHARES[k]) { u -= SHARES[k]; k++; } S.flux[c] = k + 1; }
    S.quanta = CLS.map((cl) => { const th = new Float64Array(N); let mean = 0; for (let c = 0; c < N; c++) { th[c] = restAngle(cl.q6, S.flux[c], S.d); mean += th[c]; }
      const q = quantum(`${cl.name} (${cl.who})`, cl.col, th); q.eng.addPlaneWave([0, 0, 0], BLANK); q.eng.normalize(); q.mean = mean / N;
      q.law = S.dens * SHARES.reduce((s, sh, i) => s + sh * restAngle(cl.q6, i + 1, S.d), 0); return q; });
  } else if (e === 'line') {
    S.shape = [96, 64, 2]; const [n1, n2, n3] = S.shape, x0 = 47, y0 = 31, p0 = 2 * Math.PI * 10 / 96;
    const links = [0, 1, 2].map(() => new Float64Array(n1 * n2 * n3).fill(1));
    for (let a = x0 + 1; a < n1; a++) for (let c = 0; c < n3; c++) links[1][(a * n2 + y0) * n3 + c] = -1;       // a sheet of flipped links: flux through the plaquettes on the line x = 47.5
    S.quanta = [quantum('even weak-centre charge', '#3fe0d0', 0), quantum('odd weak-centre charge', '#b197ff', 0)];
    S.quanta[1].eng.links = links;
    for (const q of S.quanta) { q.eng.addPacket([16, 31.5, 0], 6, [p0, 0, 0], branchCoin([p0, 0, 0], 0)); q.eng.normalize(); }
    S.line = [x0 + 0.5, y0 + 0.5];
  } else if (e === 'pairs') {
    S.shape = [96, 48, 2]; const [n1, n2, n3] = S.shape, N = n1 * n2 * n3, p0 = 2 * Math.PI * 8 / 96, q = 0.2676 / 12;
    let seed = 20261110; const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
    const ur = [0, 1, 2].map(() => new Float64Array(N).fill(1)), ui = [0, 1, 2].map(() => new Float64Array(N));
    for (let i = 0; i < 3; i++) for (let c = 0; c < N; c++) if (rnd() < q) { ur[i][c] = -0.5; ui[i][c] = (rnd() < 0.5 ? 1 : -1) * Math.sqrt(3) / 2; }
    const pair = (sep) => {            // link variable of a rigid pair: u(x) conj(u(x + sep e_2))
      if (sep < 0) return [ur, ui]; if (sep === 0) return [null, null];
      const zr = [0, 1, 2].map(() => new Float64Array(N)), zi = [0, 1, 2].map(() => new Float64Array(N));
      for (let i = 0; i < 3; i++) for (let a = 0; a < n1; a++) for (let b = 0; b < n2; b++) for (let c = 0; c < n3; c++) { const x = (a * n2 + b) * n3 + c, y = (a * n2 + (b + sep) % n2) * n3 + c;
        zr[i][x] = ur[i][x] * ur[i][y] + ui[i][x] * ui[i][y]; zi[i][x] = ui[i][x] * ur[i][y] - ur[i][x] * ui[i][y]; }
      return [zr, zi]; };
    S.quanta = [['one Z3-charge probe', '#ff7a45', -1], ['opposite-charge rigid pair, 4 cells apart', '#ffb199', 4], ['opposite-charge rigid pair, 1 cell apart', '#ffe45c', 1], ['coincident opposite charges: link factors cancel', '#3fe0d0', 0]].map(([label, col, sep]) => {
      const qn = quantum(label, col, 0), [zr, zi] = pair(sep); qn.eng.links = zr; qn.eng.linksIm = zi;
      qn.eng.addPacket([14, 24, 0], 5, [p0, 0, 0], branchCoin([p0, 0, 0], 0)); qn.eng.normalize(); return qn; });
    S.start = [14, 24];
  } else if (e === 'fluxobj') {
    S.shape = [256, 2, 2]; const a = Math.PI / S.a, th = 2 * a, pphys = 1.5, p0 = 2 * Math.PI * Math.round(a * pphys * 256 / (2 * Math.PI)) / 256;
    const q = quantum(`protected flux pair, spacing π/${S.a}`, '#ffe45c', th, 'Z'); slab(q.eng, 40, 7, p0, branchCoin([p0, 0, 0], th, 'Z')); q.p0 = p0; q.a = a; q.x0 = q.eng.meanPosition(0);
    const m = quantum('a quantum without rest phase, same start', '#3fe0d0', 0); slab(m.eng, 40, 7, p0, branchCoin([p0, 0, 0], 0)); m.p0 = p0; m.x0 = 40;
    S.quanta = [q, m];
  }
  geometry(); record(); draw();
}
function tick(n = 1) {
  if (S.exp === 'reg') { S.regStep = Math.min(3, S.regStep + 1); S.t = [0, 20, 40, 60][S.regStep]; return; }
  for (let i = 0; i < n; i++) { for (const q of S.quanta) q.eng.tickAltHalf(q.half, q.kind); S.t++; record(); }
}
function unwrap(x, x0, L) { let d = x - x0; d = ((d % L) + L) % L; if (d > L * 0.75) d -= L; return d; }
function record() {
  if (S.exp === 'reg') return;
  const row = { t: S.t, v: [] };
  for (const q of S.quanta) {
    if (S.exp === 'cone') { q.eng.density(q.rho); row.v.push(planar(q, S.centre, 5).inside); }
    else if (S.exp === 'race' || S.exp === 'fluxobj') row.v.push(unwrap(q.eng.meanPosition(0), q.x0, S.shape[0]));
    else if (S.exp === 'gas') row.v.push(q.eng.recordProbability());
    else if (S.exp === 'pairs') { q.eng.density(q.rho); const [n1, n2, n3] = S.shape, xc = S.start[0] + S.t; let w = 0;
      for (let a = 0; a < n1; a++) { if (Math.abs(a - xc) > 12) continue; for (let b = 0; b < n2; b++) { if (Math.abs(b - S.start[1]) > 12) continue; for (let c = 0; c < n3; c++) w += q.rho[(a * n2 + b) * n3 + c]; } }
      row.v.push(w); }
    else row.v.push(0);
  }
  S.hist.push(row);
}

// ---- drawing ------------------------------------------------------------------------------------------------------
// layouts: 'boxes' one box per quantum side by side; 'lanes' one strip per quantum (packets uniform across the box);
// 'single' one box (the flux gas)
const LAYOUT = { cone: 'boxes', line: 'boxes', pairs: 'boxes', race: 'lanes', fluxobj: 'lanes', reg: 'lanes', gas: 'single' };
const LANE_ROWS = 7;
function geometry() {
  const [n1, n2, n3] = S.shape, N = n1 * n2 * n3, lay = LAYOUT[S.exp], nq = S.quanta.length, vertical = S.exp === 'line';
  const sc = 2 / Math.max(n1, n2, n3), gap = vertical ? n2 * sc + 0.16 : n1 * sc + 0.3;
  const off = (j) => (j - (nq - 1) / 2) * gap, grid2 = S.exp === 'pairs';
  const offs = (j) => (lay !== 'boxes' ? [0, 0] : grid2 ? [((j % 2) - 0.5) * (n1 * sc + 0.2), (0.5 - Math.floor(j / 2)) * (n2 * sc + 0.2)] : vertical ? [0, -off(j)] : [off(j), 0]);
  let P;
  if (lay === 'lanes') {
    P = new Float32Array(3 * nq * n1 * LANE_ROWS); const lsc = 4.6 / n1;
    for (let j = 0; j < nq; j++) for (let a = 0; a < n1; a++) for (let r = 0; r < LANE_ROWS; r++) { const i = (j * n1 + a) * LANE_ROWS + r; P[3 * i] = (a - n1 / 2 + 0.5) * lsc; P[3 * i + 1] = ((nq - 1) / 2 - j) * 0.42 + (r - (LANE_ROWS - 1) / 2) * 0.03; P[3 * i + 2] = 0; }
  } else {
    const copies = lay === 'boxes' ? nq : 1; P = new Float32Array(3 * copies * N);
    for (let j = 0; j < copies; j++) for (let a = 0; a < n1; a++) for (let b = 0; b < n2; b++) for (let c = 0; c < n3; c++) { const i = j * N + (a * n2 + b) * n3 + c;
      P[3 * i] = (a - n1 / 2 + 0.5) * sc + offs(j)[0]; P[3 * i + 1] = (b - n2 / 2 + 0.5) * sc + offs(j)[1]; P[3 * i + 2] = (c - n3 / 2 + 0.5) * sc; }
  }
  posBuf = P; colBuf = new Float32Array(P.length);
  if (points) { scene.remove(points); points.geometry.dispose(); }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(posBuf, 3)); g.setAttribute('color', new THREE.BufferAttribute(colBuf, 3));
  const size = lay === 'lanes' ? 0.085 : n3 <= 2 ? 2.4 * sc : 1.7 * sc;
  points = new THREE.Points(g, new THREE.PointsMaterial({ size, map: sprite, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true }));
  points.frustumCulled = false; scene.add(points);
  if (fluxPts) { scene.remove(fluxPts); fluxPts.geometry.dispose(); fluxPts = null; }
  if (S.flux) { const FP = [], C = []; for (let i = 0; i < N; i++) if (S.flux[i]) { FP.push(posBuf[3 * i], posBuf[3 * i + 1], posBuf[3 * i + 2]); C.push(...hex(FLUXCOL[S.flux[i] - 1])); }
    const fg = new THREE.BufferGeometry(); fg.setAttribute('position', new THREE.Float32BufferAttribute(FP, 3)); fg.setAttribute('color', new THREE.Float32BufferAttribute(C, 3));
    fluxPts = new THREE.Points(fg, new THREE.PointsMaterial({ size: 1.3 * sc, map: sprite, vertexColors: true, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false })); scene.add(fluxPts); }
  // frames of the boxes, the flux line, and a block of carriers: icosahedra that share the six frame seams with their neighbours
  if (cage) { scene.remove(cage); cage.geometry.dispose(); }
  const L = [], box = (cx, cy, hx, hy, hz) => { for (const u of [-1, 1]) for (const v of [-1, 1]) { L.push(cx - hx, cy + u * hy, v * hz, cx + hx, cy + u * hy, v * hz); L.push(cx + u * hx, cy - hy, v * hz, cx + u * hx, cy + hy, v * hz); L.push(cx + u * hx, cy + v * hy, -hz, cx + u * hx, cy + v * hy, hz); } };
  if (lay === 'lanes') { for (let j = 0; j < nq; j++) box(0, ((nq - 1) / 2 - j) * 0.42, 2.3, 0.13, 0); }
  else { const copies = lay === 'boxes' ? nq : 1; for (let j = 0; j < copies; j++) box(offs(j)[0], offs(j)[1], n1 * sc / 2, n2 * sc / 2, n3 * sc / 2); }
  if (lay !== 'lanes' && $('sp-carriers') && $('sp-carriers').checked) {
    const ph = (1 + Math.sqrt(5)) / 2, V = []; for (const s1 of [-1, 1]) for (const s2 of [-1, 1]) V.push([0, s1, s2 * ph], [s1, s2 * ph, 0], [s1 * ph, 0, s2]);
    const E = []; for (let i = 0; i < 12; i++) for (let j = i + 1; j < 12; j++) if (Math.abs(Math.hypot(V[i][0] - V[j][0], V[i][1] - V[j][1], V[i][2] - V[j][2]) - 2) < 1e-9) E.push([i, j]);
    const r = sc / (2 * ph), B = Math.min(1, Math.floor((Math.min(n1, n2, n3) - 1) / 2)), [cx, cy] = offs(0);
    const base = [-(n1 / 2 - 0.5) * sc + cx, -(n2 / 2 - 0.5) * sc + cy, -(n3 / 2 - 0.5) * sc];          // the carriers of the corner cells of the first box
    for (let a = 0; a <= 2 * B; a++) for (let b = 0; b <= 2 * B; b++) for (let c = 0; c <= Math.min(2 * B, n3 - 1); c++) for (const [i, j] of E)
      L.push(base[0] + a * sc + V[i][0] * r, base[1] + b * sc + V[i][1] * r, base[2] + c * sc + V[i][2] * r, base[0] + a * sc + V[j][0] * r, base[1] + b * sc + V[j][1] * r, base[2] + c * sc + V[j][2] * r);
  }
  const cg = new THREE.BufferGeometry(); cg.setAttribute('position', new THREE.Float32BufferAttribute(L, 3));
  cage = new THREE.LineSegments(cg, new THREE.LineBasicMaterial({ color: '#7f97c9', transparent: true, opacity: 0.4 })); scene.add(cage);
  if (S.exp === 'line') { const [lx, ly] = S.line, LP = []; for (let j = 0; j < nq; j++) LP.push((lx - n1 / 2 + 0.5) * sc, (ly - n2 / 2 + 0.5) * sc - off(j), -0.1, (lx - n1 / 2 + 0.5) * sc, (ly - n2 / 2 + 0.5) * sc - off(j), 0.1);
    const fg = new THREE.BufferGeometry(); fg.setAttribute('position', new THREE.Float32BufferAttribute(LP, 3)); fluxPts = new THREE.LineSegments(fg, new THREE.LineBasicMaterial({ color: '#ffe45c' })); scene.add(fluxPts); }
  const cam = { cone: [0, 0, 6.6], reg: [0, 0, 4.4], pairs: [0, 0, 5.4], line: [0, -0.2, 4.6], race: [0, 0, 4.4], fluxobj: [0, 0, 4.4], gas: [2.0, 1.4, 2.5] }[S.exp];
  camera.position.set(...cam); controls.target.set(0, 0, 0); controls.update();
}
function colours() {
  const [n1, n2, n3] = S.shape, N = n1 * n2 * n3, lay = LAYOUT[S.exp]; colBuf.fill(0);
  S.quanta.forEach((q, j) => {
    if (q.fake) { const d = S.regData ? (j === 0 ? S.regData.common_phase : S.regData.number_states_bit).density[String([0, 20, 40, 60][S.regStep || 0])] : null; for (let a = 0; a < 256; a++) q.rho[a] = d ? d[a] : 0; }
    else q.eng.density(q.rho);
    if (lay === 'single') return;
    const [r, g, b] = q.rgb;
    if (lay === 'lanes') {
      const prof = new Float64Array(n1); let mx = 0; for (let a = 0; a < n1; a++) { let s = 0; for (let k = 0; k < n2 * n3; k++) s += q.rho[a * n2 * n3 + k]; prof[a] = s; if (s > mx) mx = s; }
      for (let a = 0; a < n1; a++) { const w = Math.pow(prof[a] / (mx || 1), 0.6); for (let rr = 0; rr < LANE_ROWS; rr++) { const i = (j * n1 + a) * LANE_ROWS + rr; colBuf[3 * i] = r * w; colBuf[3 * i + 1] = g * w; colBuf[3 * i + 2] = b * w; } }
      return;
    }
    let mx = 0; for (let i = 0; i < N; i++) if (q.rho[i] > mx) mx = q.rho[i];
    const inv = 1 / (mx || 1), gain = n3 <= 2 ? 0.8 : Math.min(1, 5.5 / n3);
    for (let i = 0; i < N; i++) { const u = q.rho[i] * inv; if (u < 1e-3) continue; const w = Math.min(1, Math.pow(u, 0.6)) * gain, o = 3 * (j * N + i); colBuf[o] = r * w; colBuf[o + 1] = g * w; colBuf[o + 2] = b * w; }
  });
  points.geometry.attributes.color.needsUpdate = true; points.visible = lay !== 'single';
}
function plot() {
  const cv = $('sp-plot'), r = cv.getBoundingClientRect(); if (!r.width) return;
  const dpr = window.devicePixelRatio || 1; cv.width = r.width * dpr; cv.height = r.height * dpr;
  const g = cv.getContext('2d'); g.scale(dpr, dpr); const W = r.width, H = r.height, m = { l: 52, r: 14, t: 22, b: 26 };
  g.clearRect(0, 0, W, H); g.font = '11px ui-monospace,Menlo,monospace';
  let series = [], xl = 'ticks', title = '', ymin = 0, ymax = 1, xmax = Math.max(40, S.t);
  if (S.exp === 'cone') { title = 'probability still within 5 cells of the start'; series = S.quanta.map((q, j) => ({ col: q.col, pts: S.hist.map((h) => [h.t, h.v[j]]) })); }
  else if (S.exp === 'race' || S.exp === 'fluxobj') { title = 'distance travelled, in cells (dashed: one cell per tick)'; series = S.quanta.map((q, j) => ({ col: q.col, pts: S.hist.map((h) => [h.t, h.v[j]]) })); ymax = Math.max(20, S.t); series.push({ col: '#93a3c0', dash: true, pts: [[0, 0], [xmax, xmax]] }); }
  else if (S.exp === 'reg') { title = 'distance travelled, in cells (dashed: one cell per tick)'; xmax = 60; ymax = 60;
    if (S.regData) series = [S.regData.common_phase, S.regData.number_states_bit].map((d, j) => ({ col: S.quanta[j].col, pts: d.mean_position.map((x, t) => [t, x]) }));
    series.push({ col: '#93a3c0', dash: true, pts: [[0, 0], [60, 60]] }); }
  else if (S.exp === 'pairs') { title = 'share of the packet still where a free quantum would be'; series = S.quanta.map((q, j) => ({ col: q.col, pts: S.hist.map((h) => [h.t, h.v[j]]) })); xmax = Math.max(60, S.t); }
  else if (S.exp === 'gas') { title = 'probability that the record of the quantum is written (sin² of rest phase × ticks)'; series = S.quanta.map((q, j) => ({ col: q.col, pts: S.hist.map((h) => [h.t, h.v[j]]) })); }
  else if (S.exp === 'line') { title = 'where the quantum is found across the line (summed along the flight direction)'; xl = 'cells across'; xmax = S.shape[1];
    series = S.quanta.map((q) => { const [n1, n2, n3] = S.shape, p = new Float64Array(n2); for (let a = 0; a < n1; a++) for (let b = 0; b < n2; b++) for (let c = 0; c < n3; c++) p[b] += q.rho[(a * n2 + b) * n3 + c]; return { col: q.col, pts: Array.from(p, (v, b) => [b, v]) }; });
    ymax = Math.max(1e-9, ...series.flatMap((s) => s.pts.map((p) => p[1]))) * 1.1; series.push({ col: '#ffe45c', dash: true, pts: [[S.line[1], 0], [S.line[1], ymax]] }); }
  const X = (x) => m.l + (W - m.l - m.r) * x / xmax, Y = (y) => H - m.b - (H - m.t - m.b) * (y - ymin) / (ymax - ymin);
  g.strokeStyle = 'rgba(147,163,192,.35)'; g.lineWidth = 1; g.beginPath(); g.moveTo(m.l, m.t); g.lineTo(m.l, H - m.b); g.lineTo(W - m.r, H - m.b); g.stroke();
  g.fillStyle = '#93a3c0'; g.fillText(title, m.l + 4, 14); g.fillText(xl, W - m.r - 60, H - 8); g.fillText(ymax < 10 ? ymax.toPrecision(2) : ymax.toFixed(0), 6, m.t + 4); g.fillText('0', 34, H - m.b + 3); g.fillText(String(Math.round(xmax)), W - m.r - 22, H - m.b + 13);
  for (const s of series) { g.strokeStyle = s.col; g.lineWidth = s.dash ? 1 : 2; g.setLineDash(s.dash ? [5, 5] : []); g.beginPath(); s.pts.forEach(([x, y], i) => { const px = X(x), py = Y(Math.max(ymin, Math.min(ymax, y))); if (i) g.lineTo(px, py); else g.moveTo(px, py); }); g.stroke(); }
  g.setLineDash([]);
}
const f4 = (x) => (Math.abs(x) < 5e-5 ? '0' : x.toFixed(4));
function numbers() {
  let h = `<div class="small" style="margin-bottom:6px">tick ${S.t} · lattice ${S.shape.join(' × ')} carriers</div><table class="t" style="width:100%"><tr>`;
  const sw = (q) => `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${q.col};margin-right:6px"></span>`;
  if (S.exp === 'cone') {
    h += '<th>quantum</th><th>rest phase</th><th>mean distance</th><th>still within 5 cells</th></tr>';
    for (const q of S.quanta) { const r = planar(q, S.centre, 5); h += `<tr><td>${sw(q)}${q.label}</td><td>${f4(q.theta)}</td><td>${r.mean.toFixed(1)}</td><td>${(100 * r.inside).toFixed(1)} %</td></tr>`; }
    h += `</table><div class="note">The picture is a cut: the packets are uniform along the third axis. A free shell grows by one cell per tick along an axis. Inverse rest angle of the middle walk: ${(1 / REST[3][4]).toFixed(1)} cells; of the routed flux code: ${(6 / Math.PI).toFixed(1)} cells.</div>`;
  } else if (S.exp === 'race') {
    h += '<th>class of patterns</th><th>rest phase</th><th>speed measured</th><th>dE/dp</th></tr>';
    S.quanta.forEach((q, j) => { const d = S.hist.length ? S.hist[S.hist.length - 1].v[j] : 0, v = S.t ? d / S.t : 0, E = Math.acos(Math.cos(q.theta) * Math.cos(q.p0)), vg = E < 1e-12 ? 1 : Math.cos(q.theta) * Math.sin(q.p0) / Math.sin(E);
      h += `<tr><td>${sw(q)}${q.label}</td><td>${f4(q.theta)}</td><td>${S.t ? v.toFixed(4) : ''}</td><td>${vg.toFixed(4)}</td></tr>`; });
    h += `</table><div class="note">Flux class ${S.k} at face separation ${S.d} on every carrier. Wave number of the packets ${S.quanta[0].p0.toFixed(4)} per cell. The rest phase is the angle by which the carrier turns the record of a resting quantum in one tick.</div>`;
  } else if (S.exp === 'gas') {
    h += '<th>class of patterns</th><th>sample mean angle</th><th>short-time inferred angle</th></tr>';
    S.quanta.forEach((q, j) => { const T = Math.min(S.t, 8), pr = S.hist[T] ? S.hist[T].v[j] : 0, th = T ? Math.asin(Math.sqrt(Math.min(1, pr))) / T : 0; h += `<tr><td>${sw(q)}${q.label}</td><td>${f4(q.mean)}</td><td>${T ? f4(th) : ''}</td></tr>`; });
    const n = S.flux.reduce((s, v) => s + (v ? 1 : 0), 0);
    h += `</table><div class="note">${n} of ${S.flux.length} carriers hold a flux pair (dots, coloured by class). The diagnostic is asin(sqrt(record occupation))/T after ${Math.min(S.t, 8)} ticks. It uses a short-time principal branch; scattering and later oscillations can change it.</div>`;
  } else if (S.exp === 'line') {
    h += '<th>pattern</th><th>probability in the strip |y − y_line| ≤ 2</th></tr>';
    for (const q of S.quanta) { const [n1, n2, n3] = S.shape; let s = 0; for (let a = 0; a < n1; a++) for (let b = 0; b < n2; b++) if (Math.abs(b - S.line[1]) <= 2) for (let c = 0; c < n3; c++) s += q.rho[(a * n2 + b) * n3 + c]; h += `<tr><td>${sw(q)}${q.label}</td><td>${(100 * s).toFixed(2)} %</td></tr>`; }
    h += '</table><div class="note">The flux line is the yellow mark. The two quanta start with the same packet. Only the sign of the links differs, and only one of them reads it. Run about 55 ticks.</div>';
  } else if (S.exp === 'reg') {
    h += '<th>state of the register</th><th>travelled in 60 ticks</th><th>width</th><th>right-handed share, start and end</th></tr>';
    if (S.regData) [S.regData.common_phase, S.regData.number_states_bit].forEach((d, j) => { h += `<tr><td>${sw(S.quanta[j])}${S.quanta[j].label}</td><td>${d.final_displacement.toFixed(1)} cells</td><td>${d.final_width.toFixed(1)}</td><td>${(100 * d.P_right_handed[0]).toFixed(1)} % and ${(100 * d.final_P_right_handed).toFixed(1)} %</td></tr>`; });
    h += `</table><div class="note">Line of 256 carriers, rest angle 0.1579, the same packet in both rows; press step to move through ticks 0, 20, 40 and 60. The common-phase row uses the supplied coherent evolution. The definite-state row is an average over 2,000 sampled register sectors, as the field session computed it; the method agrees with the exact calculation on small lines to 6 × 10⁻¹⁶. With the register the tick commutes with the gauge constraint exactly; without it the defect is 0.1755.</div>`;
  } else if (S.exp === 'pairs') {
    h += '<th>object</th><th>still on the free path</th></tr>';
    const last = S.hist[S.hist.length - 1];
    S.quanta.forEach((q, j) => { h += `<tr><td>${sw(q)}${q.label}</td><td>${(100 * last.v[j] / (last.v[3] || 1)).toFixed(1)} % of the free one</td></tr>`; });
    h += '</table><div class="note">Independent Z3 flux is placed on 0.2676/12 ≈ 2.23 percent of links. This is a dilute density matching choice, not a replay of the run. A rigid pair reads one link times the conjugate link at its fixed displacement. Run about 60 ticks. The plotted quantity is probability in a moving window relative to the free control; it is neither total survival nor a string energy.</div>';
  } else if (S.exp === 'fluxobj') {
    h += '<th>object</th><th>rest angle</th><th>speed measured</th><th>dE/dp of the tick</th><th>joint-scaling Dirac comparison</th></tr>';
    S.quanta.forEach((q, j) => { const d = S.hist.length ? S.hist[S.hist.length - 1].v[j] : 0, v = S.t ? d / S.t : 0, E = Math.acos(Math.cos(q.theta) * Math.cos(q.p0)), vg = E < 1e-12 ? 1 : Math.cos(q.theta) * Math.sin(q.p0) / Math.sin(E);
      const pp = q.a ? q.p0 / q.a : 0; h += `<tr><td>${sw(q)}${q.label}</td><td>${f4(q.theta)}</td><td>${S.t ? v.toFixed(4) : ''}</td><td>${vg.toFixed(4)}</td><td>${q.a ? (pp / Math.hypot(pp, 2)).toFixed(4) : '1.0000'}</td></tr>`; });
    const q = S.quanta[0], pp = q.p0 / q.a, E = Math.acos(Math.cos(q.theta) * Math.cos(q.p0)) / q.a;
    h += `</table><div class="note">Scaled wave number ${pp.toFixed(3)}, frequency ${E.toFixed(4)} against √(p² + 4) = ${Math.hypot(pp, 2).toFixed(4)}: ${(100 * (E / Math.hypot(pp, 2) - 1)).toFixed(2)} % off at this spacing. Both p and the rest angle 2a shrink with spacing a. The axial dispersion error is O(a²) at fixed p/a; the finite packet and momentum menu are separate approximations. The internal pair code is carried unchanged.</div>`;
  }
  $('sp-numbers').innerHTML = h;
}
function legend() {
  $('sp-legend').innerHTML = S.quanta.map((q) => `<span class="lg"><i style="background:${q.col}"></i>${q.label}</span>`).join('') + (S.exp === 'gas' ? '<span class="lg"><i style="background:#ffe45c"></i>flux classes 1, 5</span><span class="lg"><i style="background:#7dd3fc"></i>classes 2, 4</span><span class="lg"><i style="background:#f0abfc"></i>class 3</span>' : '');
}
function draw() { if (!renderer) return; colours(); renderer.render(scene, camera); plot(); numbers(); legend(); }
function side() {
  $('sp-text').textContent = EXP[S.exp].text;
  $('sp-bg').style.display = S.exp === 'race' || S.exp === 'gas' ? '' : 'none'; $('sp-k-row').style.display = S.exp === 'race' ? '' : 'none';
  $('sp-dens-row').style.display = S.exp === 'gas' ? '' : 'none'; $('sp-a-row').style.display = S.exp === 'fluxobj' ? '' : 'none';
}
function run(on) { S.running = on; $('sp-run').textContent = on ? '❚❚ pause' : '▶ run'; }
function resize() { const cv = $('sp-gl'), r = cv.getBoundingClientRect(); if (!r.width || !renderer) return; renderer.setSize(r.width, r.height, false); camera.aspect = r.width / r.height; camera.updateProjectionMatrix(); }
function loop() {
  requestAnimationFrame(loop);
  if (!$('page-space').classList.contains('on')) return;
  if (S.running && S.exp === 'reg') { const now = performance.now(); if (now - (S.lastReg || 0) > 800) { S.lastReg = now; tick(1); colours(); plot(); numbers(); if (S.regStep >= 3) run(false); } }
  else if (S.running) { const per = parseInt($('sp-speed').value); tick(per); const lim = { cone: 28, race: 140, gas: 60, line: 60, fluxobj: 190, pairs: 62, reg: 60 }[S.exp]; if (S.t >= lim) run(false); colours(); plot(); numbers(); }
  controls.update(); renderer.render(scene, camera);
}
async function atScale() {
  const get = async (u) => { try { const r = await fetch(u); return r.ok ? await r.json() : null; } catch (e) { return null; } };
  const [runsJ, v128, v256, conf, h2] = await Promise.all([get('/rl/SIM/cf1/cf1_runs.json'), get('/rl/SIM/cf1/runs/V_validation_128.json'), get('/rl/SIM/cf1/runs/V_validation_256.json'), get('/rl/SIM/cf1/runs/conformance.json'), get('/rl/SIM/cf1/runs/H2_coherence_48.json')]);
  runs = runsJ; const el = $('sp-scale'); let h = '<table class="t" style="width:100%"><tr><th>run</th><th>result</th></tr>';
  if (conf) { const n = Object.keys(conf.items).length, ok = Object.values(conf.items).filter((x) => x.pass).length; h += `<tr><td>conformance suite</td><td>${ok} of ${n} checks pass; 200 ticks forward and back on 32³: ${conf.items.d1.value.toExponential(1)}; browser against Python: ${conf.items.j1 ? conf.items.j1.value.toExponential(1) : 'n/a'}</td></tr>`; }
  for (const [name, v] of [['2.1 million carriers (128³)', v128], ['16.8 million carriers (256³)', v256]]) {
    if (!v) continue; const b = v.blocks;
    if (b.q1) h += `<tr><td>${name}: forward and back</td><td>64 ticks each way return the state to ${b.q1.return_error_64_forward_64_inverse.toExponential(1)}; norm drift ${b.q1.norm_drift_64_ticks.toExponential(1)}</td></tr>`;
    if (b.q2_6) { const r = b.q2_6.series[b.q2_6.series.length - 1]; h += `<tr><td>${name}: low-momentum packet</td><td>after ${r.t} ticks the shell is at ${r.radius_axis_100.toFixed(1)} cells along an axis, ${r.radius_diagonal_110.toFixed(1)} along a face diagonal, ${r.radius_diagonal_111.toFixed(1)} along a space diagonal</td></tr>`; }
    if (b.q3_0878 && b.q3_1712 && b.q2_6) { const r0 = b.q2_6.series[b.q2_6.series.length - 1], r1 = b.q3_0878.series[b.q3_0878.series.length - 1], r2 = b.q3_1712.series[b.q3_1712.series.length - 1]; h += `<tr><td>${name}: rest phases</td><td>mean radius ${r0.mean_radius.toFixed(1)} without rest phase, ${r1.mean_radius.toFixed(1)} at 0.0878, ${r2.mean_radius.toFixed(1)} at 0.1712</td></tr>`; }
    if (b.q4c_qd) h += `<tr><td>${name}: static flux sample using run proportions</td><td>rest phase ${b.q4c_qd.effective_phase.toFixed(5)} against density × angle ${b.q4c_qd.mean_angle_in_this_sample.toFixed(5)} for the q6 = 1 row; ${b.q4c_qs.effective_phase.toFixed(5)} against ${b.q4c_qs.mean_angle_in_this_sample.toFixed(5)} for the q6 = 2 row; 0 for a neutral species</td></tr>`;
  }
  if (h2) h += `<tr><td>link-flux gas, ${h2.lattice.join(' × ')}</td><td>fitted coherence length: ${h2.species.map((s) => s.name.replace(' and neutrino', '') + ' ' + (s.mean_free_path_cells ? s.mean_free_path_cells.toFixed(0) + ' carriers' : 'not resolved')).join('; ')}</td></tr>`;
  if (runs && runs.defect) { const def = runs.defect.rows; h += `<tr><td>one carrier with flux</td><td>scattered ${def.slice(0, 2).map((r) => r.scattered_probability.toExponential(1)).join(' and ')}; no trapping</td></tr>`; }
  h += '</table><div class="note">Label of the large runs: candidate CF-1, free tick in a static background, validation. These are separate static-background experiments, not the classical load dataset. Packets fill all four body-centred sublattices on purpose; the pictures show the sum.</div>';
  el.innerHTML = h;
}
export function spaceShown() {
  if (!inited) {
    inited = true;
    const cv = $('sp-gl');
    renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true, preserveDrawingBuffer: true }); renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    scene = new THREE.Scene(); scene.background = new THREE.Color('#04060d');
    camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50); controls = new OrbitControls(camera, cv); controls.enableDamping = true;
    const c = document.createElement('canvas'); c.width = c.height = 64; const g = c.getContext('2d'), gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.35, 'rgba(255,255,255,.55)'); gr.addColorStop(1, 'rgba(255,255,255,0)'); g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
    sprite = new THREE.CanvasTexture(c);
    $('sp-exp').innerHTML = Object.entries(EXP).map(([k, v]) => `<option value="${k}">${v.title}</option>`).join('');
    $('sp-exp').onchange = (e) => { S.exp = e.target.value; run(false); side(); build(); };
    $('sp-k').onchange = (e) => { S.k = parseInt(e.target.value); run(false); build(); };
    $('sp-d').onchange = (e) => { S.d = parseInt(e.target.value); run(false); build(); };
    $('sp-dens').onchange = (e) => { S.dens = parseFloat(e.target.value); run(false); build(); };
    $('sp-a').onchange = (e) => { S.a = parseInt(e.target.value); run(false); build(); };
    $('sp-carriers').onchange = () => { geometry(); draw(); };
    $('sp-reset').onclick = () => { run(false); build(); };
    window.__spaceTick = (n) => { for (let i = 0; i < n; i++) tick(1); draw(); };
    $('sp-step').onclick = () => { run(false); tick(1); draw(); };
    $('sp-run').onclick = () => run(!S.running);
    window.addEventListener('resize', () => { resize(); draw(); });
    const q = new URLSearchParams(location.search);
    if (q.has('spexp') && EXP[q.get('spexp')]) { S.exp = q.get('spexp'); $('sp-exp').value = S.exp; }
    if (q.has('spk')) { S.k = parseInt(q.get('spk')); $('sp-k').value = String(S.k); }
    if (q.has('spd')) { S.d = parseInt(q.get('spd')); $('sp-d').value = String(S.d); }
    if (q.has('spdens')) { S.dens = parseFloat(q.get('spdens')); $('sp-dens').value = String(S.dens); }
    resize(); side(); build();
    if (q.has('spticks')) { tick(parseInt(q.get('spticks'))); draw(); }
    atScale(); loop(); forcesInit();
  } else { resize(); draw(); forcesShown(); }
}
window.__space = { S, tick, draw, build };
