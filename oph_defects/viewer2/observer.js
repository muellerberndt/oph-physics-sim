// One observer: its own clock, its record tape and readouts, the history of its neighbourhood on the
// slices of synchronised time, and the declared wave reading of its patch (the 1+3 scalar description).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadNpz } from './npz.js';
import { $, S, FAM, fmt, row, classInfo } from './app.js';

let O = null, mode = 'records', tick = 0, playing = false, hooked = false, loadedFor = '';
let renderer, scene, camera, controls, group, nowPlane;
const cutDefects = new Map();
let wave = null;

function setup() {
  if (renderer) return;
  const canvas = $('obsgl');
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  scene = new THREE.Scene(); scene.background = new THREE.Color('#090d16');
  camera = new THREE.PerspectiveCamera(40, 1, 0.01, 500);
  camera.position.set(26, -34, 24); camera.up.set(0, 0, 1);
  controls = new OrbitControls(camera, canvas); controls.enableDamping = true; controls.target.set(0, 0, 8);
  group = new THREE.Group(); scene.add(group);
  window.addEventListener('resize', resize);
  (function loop() { requestAnimationFrame(loop); if ($('page-observer').classList.contains('on')) { controls.update(); renderer.render(scene, camera); } })();
}
function resize() {
  if (!renderer) return;
  const c = $('obsgl'), w = c.clientWidth, h = c.clientHeight;
  if (!w) return;
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); drawPlots();
}
function clearGroup() { while (group.children.length) { const c = group.children.pop(); c.geometry.dispose(); c.material.dispose(); } }
function points(P, C, size, opacity = 1) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3)); g.setAttribute('color', new THREE.Float32BufferAttribute(C, 3));
  const p = new THREE.Points(g, new THREE.PointsMaterial({ size, vertexColors: true, transparent: true, opacity, depthWrite: false, sizeAttenuation: true }));
  p.frustumCulled = false; group.add(p); return p;
}
function line(P, color, opacity = 1) {
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
  const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity })); group.add(l); return l;
}
const HEIGHT = 22;           // drawing height of the whole tape

// ------------------------------------------------------------------ geometry of the patch
function prepare() {
  const m = O.patch_cells.length, sp = S.meta.geometry.cell_spacing, fr = O.frame;
  O.m = m; O.uv = new Float32Array(2 * m);
  for (let c = 0; c < m; c++) {
    const x = O.patch_cell_xyz[3 * c], y = O.patch_cell_xyz[3 * c + 1], z = O.patch_cell_xyz[3 * c + 2];
    O.uv[2 * c] = (x * fr[0] + y * fr[3] + z * fr[6]) / sp; O.uv[2 * c + 1] = (x * fr[1] + y * fr[4] + z * fr[7]) / sp;
  }
  O.n = O.tape_sweep.length;
  O.radius = 0; for (let c = 0; c < m; c++) O.radius = Math.max(O.radius, Math.hypot(O.uv[2 * c], O.uv[2 * c + 1]));
  // cell loads on every slice, as deviations from the slice mean over the patch
  const K = O.cut_depth.length, P = 12 * m;
  O.K = K; O.cellDev = new Float32Array(K * m); O.cutSd = new Float32Array(K);
  for (let j = 0; j < K; j++) {
    let mean = 0;
    for (let c = 0; c < m; c++) { let s = 0; for (let p = 0; p < 12; p++) s += O.cut_patch_loads[j * P + 12 * c + p]; O.cellDev[j * m + c] = s; mean += s; }
    mean /= m; let sd = 0;
    for (let c = 0; c < m; c++) { O.cellDev[j * m + c] -= mean; sd += O.cellDev[j * m + c] ** 2; }
    O.cutSd[j] = Math.sqrt(sd / m) || 1;
  }
  tick = O.n; $('obsTick').max = O.n; $('obsTick').value = tick;
}
const zOfTick = (t) => HEIGHT * t / Math.max(1, O.n);
function tangent(x, y, z) {
  const fr = O.frame, sp = S.meta.geometry.cell_spacing;
  return [(x * fr[0] + y * fr[3] + z * fr[6]) / sp, (x * fr[1] + y * fr[4] + z * fr[7]) / sp, x * fr[2] + y * fr[5] + z * fr[8]];
}

// ------------------------------------------------------------------ records view: the neighbourhood on every slice
async function buildRecords() {
  clearGroup();
  const m = O.m, K = O.K, P = [], C = [];
  const stride = Math.max(1, Math.floor(K / 70));
  for (let j = 0; j < K; j += stride) {
    if (O.cut_depth[j] < 0) continue;
    const z = zOfTick(O.cut_ticks[j]), sd = O.cutSd[j];
    for (let c = 0; c < m; c++) {
      const t = Math.max(-1, Math.min(1, O.cellDev[j * m + c] / (2.5 * sd)));
      if (Math.abs(t) < 0.45) continue;
      P.push(O.uv[2 * c], O.uv[2 * c + 1], z);
      if (t > 0) C.push(0.95, 0.62, 0.20); else C.push(0.25, 0.55, 0.98);
    }
  }
  points(new Float32Array(P), new Float32Array(C), 0.34, 0.5);
  line(new Float32Array([0, 0, 0, 0, 0, HEIGHT]), '#ffffff', 0.9);                       // the observer's own history
  const ring = []; for (let a = 0; a <= 64; a++) ring.push(O.radius * Math.cos(a / 32 * Math.PI), O.radius * Math.sin(a / 32 * Math.PI), 0);
  line(new Float32Array(ring), '#3fe0d0', 0.5);
  nowPlane = new THREE.Mesh(new THREE.RingGeometry(0, O.radius, 64), new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false }));
  group.add(nowPlane); placeNow();
  // defects of the screen tier inside the patch, slice by slice: their worldlines (precomputed in view/observer_defects.npz)
  const key = S.name + ':' + O.carrier[0];
  if (!cutDefects.has(S.name)) {
    try { cutDefects.set(S.name, await loadNpz(S.base + 'view/observer_defects.npz')); } catch (e) { cutDefects.set(S.name, null); }
  }
  const D = cutDefects.get(S.name), c0 = O.carrier[0];
  if (D && D['o' + c0 + '_cut'] && loadedFor === key && mode === 'records') {
    const cut = D['o' + c0 + '_cut'], cls = D['o' + c0 + '_cls'], xyz = D['o' + c0 + '_xyz'];
    const DP = new Float32Array(3 * cut.length), DC = new Float32Array(3 * cut.length);
    for (let i = 0; i < cut.length; i++) {
      const t = tangent(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
      DP[3 * i] = t[0]; DP[3 * i + 1] = t[1]; DP[3 * i + 2] = zOfTick(O.cut_ticks[cut[i]]);
      const k = ((cls[i] % 6) + 6) % 6 || 1, c = FAM[k].rgb;
      DC[3 * i] = c.r; DC[3 * i + 1] = c.g; DC[3 * i + 2] = c.b;
    }
    points(DP, DC, 0.55, 0.95);
  }
}
function placeNow() { if (nowPlane) nowPlane.position.z = zOfTick(tick); }

// ------------------------------------------------------------------ wave reading: the declared 1+3 scalar description
const KMODES = 6, RSAMPLES = 14;
function buildWave() {
  // leapfrog integration of d^2 phi / dN^2 = -(L / M + m_k^2) phi on the patch port graph, in observer ticks N.
  // L is the Laplacian of the seams inside the patch, M the number of seams of the whole screen (one tick advances
  // the admitted coherent continuation cos(N arccos K), K = I - L / (2M)); m_k^2 = k^2 / (630 * 16^level) is the
  // declared radial mode k of the product interval (0, pi). Source: a unit kick at the observer's twelve ports at r = pi/2.
  const m = O.m, P = 12 * m, sm = O.patch_seams, ns = sm.length / 3, M = S.meta.geometry.seams, level = S.meta.geometry.level;
  const dt = Math.max(1, Math.floor(0.5 * Math.sqrt(M / 6)));
  const steps = Math.max(2, Math.ceil(O.n / dt));
  const frames = new Float32Array((steps + 1) * KMODES * m);
  for (let k = 1; k <= KMODES; k++) {
    const amp = Math.sin(k * Math.PI / 2);
    if (Math.abs(amp) < 1e-9) continue;
    const mk2 = k * k / (630 * Math.pow(16, level));
    let prev = new Float64Array(P), cur = new Float64Array(P), next = new Float64Array(P), lap = new Float64Array(P);
    for (let p = 0; p < 12; p++) cur[p] = amp * dt * (6 / 33);  // phi(dt) = dt * velocity, phi(0) = 0
    for (let s = 1; s <= steps; s++) {
      for (let c = 0; c < m; c++) { let a = 0; for (let p = 0; p < 12; p++) a += cur[12 * c + p]; frames[(s * KMODES + (k - 1)) * m + c] = a / 12; }
      lap.fill(0);
      for (let e = 0; e < ns; e++) { const a = sm[3 * e], b = sm[3 * e + 1], d = cur[a] - cur[b]; lap[a] += d; lap[b] -= d; }
      const f = dt * dt * (6 / 33) ** 2;                        // the walk steps once per incident event at a glued port (6 seams); the carrier counts 33 ticks for 6 such events
      for (let p = 0; p < P; p++) next[p] = 2 * cur[p] - prev[p] - f * (lap[p] / M + mk2 * cur[p]);
      const t = prev; prev = cur; cur = next; next = t;
    }
  }
  let peak = 1e-12; for (let i = 0; i < frames.length; i++) peak = Math.max(peak, Math.abs(frames[i]));
  wave = { dt, steps, frames, peak };
}
function showWave() {
  clearGroup();
  if (!wave) buildWave();
  const m = O.m, s = Math.max(1, Math.min(wave.steps, Math.round(tick / wave.dt)));
  const P = [], C = [];
  const L = HEIGHT * 0.8;
  let localPeak = 1e-12;
  const val = new Float32Array(RSAMPLES * m);
  for (let j = 0; j < RSAMPLES; j++) {
    const r = (j + 0.5) / RSAMPLES * Math.PI;
    for (let c = 0; c < m; c++) {
      let a = 0; for (let k = 1; k <= KMODES; k++) a += Math.sin(k * r) * wave.frames[(s * KMODES + (k - 1)) * m + c];
      val[j * m + c] = a; localPeak = Math.max(localPeak, Math.abs(a));
    }
  }
  for (let j = 0; j < RSAMPLES; j++) for (let c = 0; c < m; c++) {
    const t = val[j * m + c] / localPeak;
    if (Math.abs(t) < 0.06) continue;
    P.push(O.uv[2 * c], O.uv[2 * c + 1], (j + 0.5) / RSAMPLES * L);
    const a = Math.min(1, Math.abs(t) * 1.6);
    if (t > 0) C.push(0.25 + 0.75 * a, 0.2 + 0.5 * a, 0.1); else C.push(0.1, 0.25 + 0.4 * a, 0.3 + 0.7 * a);
  }
  points(new Float32Array(P), new Float32Array(C), 0.5, 0.85);
  line(new Float32Array([0, 0, 0, 0, 0, L]), '#ffffff', 0.5);
  for (const z of [0, L]) { const ring = []; for (let a = 0; a <= 64; a++) ring.push(O.radius * Math.cos(a / 32 * Math.PI), O.radius * Math.sin(a / 32 * Math.PI), z); line(new Float32Array(ring), '#3fe0d0', 0.4); }
  const src = new THREE.Mesh(new THREE.SphereGeometry(0.35, 16, 12), new THREE.MeshBasicMaterial({ color: '#ffffff' })); src.position.set(0, 0, L / 2); group.add(src);
}

// ------------------------------------------------------------------ plots and panel
function drawPlots() {
  const cv = $('obsPlot'); if (!O || !cv.clientWidth) return;
  const g = cv.getContext('2d');
  const W = cv.width = cv.clientWidth * devicePixelRatio, H = cv.height = cv.clientHeight * devicePixelRatio;
  g.fillStyle = '#0b1120'; g.fillRect(0, 0, W, H);
  const n = O.n, pw = W / 3;
  const panel = (i, title, series, colors) => {
    const x0 = i * pw + 12, x1 = (i + 1) * pw - 12, y0 = 22 * devicePixelRatio, y1 = H - 8;
    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (let t = 0; t <= n; t += Math.max(1, Math.floor(n / 600))) { const v = s(t); lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
    g.strokeStyle = '#22304d'; g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    series.forEach((s, k) => {
      g.strokeStyle = colors[k]; g.lineWidth = 1.4 * devicePixelRatio; g.beginPath();
      for (let t = 0, first = true; t <= n; t += Math.max(1, Math.floor(n / 600))) { const x = x0 + (x1 - x0) * t / n, y = y1 - (y1 - y0) * (s(t) - lo) / (hi - lo); first ? g.moveTo(x, y) : g.lineTo(x, y); first = false; }
      g.stroke();
    });
    const xt = x0 + (x1 - x0) * tick / Math.max(1, n);
    g.strokeStyle = '#fff'; g.lineWidth = devicePixelRatio; g.beginPath(); g.moveTo(xt, y0); g.lineTo(xt, y1); g.stroke();
    g.fillStyle = '#93a3c0'; g.font = `${11.5 * devicePixelRatio}px Helvetica, Arial`; g.fillText(title + `   [${lo.toFixed(1)}, ${hi.toFixed(1)}]`, x0, 14 * devicePixelRatio);
  };
  const D = O.displacement, Z = O.axial;
  const d0 = [D[0], D[1], D[2]];
  panel(0, 'load-weighted position x(n) − x(0), three components', [0, 1, 2].map((k) => (t) => D[3 * t + k] - d0[k]), ['#ffb44a', '#3fe0d0', '#a995ff']);
  panel(1, 'axial circulation Z(n) of the committed currents', [0, 1, 2].map((k) => (t) => Z[3 * t + k]), ['#ffb44a', '#3fe0d0', '#a995ff']);
  const rate = O.n ? O.tape_depth[O.n - 1] / O.n : 1;
  panel(2, 'synchronised depth minus its mean rate (clock offset against the screen)', [(t) => (t ? O.tape_depth[t - 1] - rate * t : 0)], ['#5fe3a1']);
}
function updatePanel() {
  const ob = S.meta.observers.find((x) => x.carrier === O.carrier[0]);
  const i = Math.max(0, Math.min(O.n - 1, tick - 1));
  const sweep = O.n ? O.tape_sweep[i] : 0, depth = O.n ? O.tape_depth[i] : 0;
  const loads = Array.from(O.loads_by_tick.slice(12 * tick, 12 * tick + 12));
  const lvl = S.meta.geometry.level;
  const waveTime = tick * (6 / 33) / (Math.sqrt(630) * Math.pow(4, lvl));     // one walk step per incident event at a port of degree 6
  let recent = '';
  for (let j = Math.max(0, tick - 6); j < tick; j++) recent += `<div class="row"><span>tick ${j + 1} · ${O.tape_slot[j] < 30 ? 'own seam ' + O.tape_slot[j] : 'glue seam ' + (O.tape_slot[j] - 30)}</span><b>coin ${O.tape_coin[j]} · current ${O.tape_current[j]}</b></div>`;
  $('obs-side').innerHTML = `
    <div class="k">Observer</div>
    <div class="card">${row('carrier', fmt(O.carrier[0]))}${row('role', ob ? ob.role : '')}${row('patch', fmt(O.m) + ' carriers within ' + Math.max(...O.patch_dist) + ' glue steps')}
      ${row('records on its tape', fmt(O.n) + ' (33 seams, every attempt counts)')}</div>
    <div class="k">Clock</div>
    <div class="card">${row('own tick', fmt(tick) + ' of ' + fmt(O.n))}${row('synchronised depth of that record', fmt(depth))}${row('solver sweep, for orientation', sweep)}
      ${row('wave time of the declared reading', waveTime.toExponential(3))}
      <div class="note">The own tick is the count of repair attempts on the 33 seams of this carrier. Slices of equal causal depth are the consensus slices; the sweep is a solver counter.</div></div>
    <div class="k">View</div>
    <div class="card"><div style="display:flex;gap:6px;margin-bottom:8px"><button id="mRec" class="${mode === 'records' ? 'on' : ''}">Records</button><button id="mWave" class="${mode === 'wave' ? 'on' : ''}">Wave reading</button></div>
      <div class="note">${mode === 'records'
        ? 'Derived from the run. Horizontal: the neighbourhood of the observer in units of one carrier. Vertical: its own ticks. Each layer is one slice of synchronised time: orange cells carry more load than the patch mean, blue cells less; bright dots are screen defects, so a column of dots is a worldline. The white plane is the selected tick.'
        : 'Declared reading, not produced by passive repair: the coherent continuation of the native transfer operator advanced one step per own tick, on the seams of the patch, times a radial interval (0, π) with ' + KMODES + ' sine modes. Shown: the response to a unit kick at the observer (white dot) at mid radius. Horizontal: screen, vertical: the declared radial coordinate. The front is the cone of this wave equation; it supplies no independent propagation speed.'}</div></div>
    <div class="k">Loads of its twelve ports at this tick</div>
    <div class="card"><div style="display:flex;gap:4px;align-items:flex-end;height:54px">${loads.map((v, p) => `<div title="port ${p}: ${v}" style="flex:1;background:#3fe0d0;opacity:.85;height:${8 + 9 * v}px;border-radius:3px 3px 0 0"></div>`).join('')}</div>
      <div class="note">ports 0 to 11 · the tape alone reproduces these loads from the initial reading (checked when the run was written)</div></div>
    <div class="k">Last records</div><div class="card">${recent || '<span class="small">the preparation</span>'}</div>
    <div class="k">Readouts</div>
    <div class="card">${row('position readout x(n)', [0, 1, 2].map((k) => O.displacement[3 * tick + k].toFixed(2)).join(', '))}
      ${row('axial circulation Z(n)', [0, 1, 2].map((k) => O.axial[3 * tick + k].toFixed(2)).join(', '))}
      ${row('|Z|² / n', tick ? ((O.axial[3 * tick] ** 2 + O.axial[3 * tick + 1] ** 2 + O.axial[3 * tick + 2] ** 2) / tick).toFixed(4) : '–')}
      <div class="note">x(n) = Σ n_p u_p over the twelve port vectors; each record moves it by j (u_b − u_a). Z(n) = Σ (û_a × û_b) j over the carrier's own seams: a face loop restores the loads and still advances Z.</div></div>`;
  $('mRec').onclick = () => setMode('records'); $('mWave').onclick = () => setMode('wave');
  $('obsLabel').textContent = `own tick ${fmt(tick)} · depth ${fmt(depth)} · sweep ${sweep}`;
}
function setMode(mo) { mode = mo; if (mode === 'records') buildRecords(); else showWave(); updatePanel(); }
function setTick(t) {
  tick = Math.max(0, Math.min(O.n, t)); $('obsTick').value = tick;
  if (mode === 'records') placeNow(); else showWave();
  updatePanel(); drawPlots();
}
async function loadObserver(carrier) {
  setup();
  loadedFor = S.name + ':' + carrier;
  O = await loadNpz(`${S.base}observers/obs_${carrier}.npz`);
  wave = null; prepare(); resize();
  if (mode === 'records') buildRecords(); else showWave();
  updatePanel(); drawPlots();
}
export async function initObserver() {
  const sel = $('obsSel'); sel.innerHTML = '';
  for (const ob of S.meta.observers) { const o = document.createElement('option'); o.value = ob.carrier; o.textContent = `carrier ${ob.carrier}: ${ob.role}`; sel.appendChild(o); }
  O = null; loadedFor = '';
  if (!hooked) {
    hooked = true;
    sel.onchange = () => loadObserver(parseInt(sel.value));
    $('obsTick').addEventListener('input', (e) => setTick(parseInt(e.target.value)));
    $('obsPlay').onclick = () => { playing = !playing; $('obsPlay').textContent = playing ? '❚❚ pause' : '▶ play'; if (playing) { if (tick >= O.n) tick = 0; play(); } };
  }
  if ($('page-observer').classList.contains('on') && S.meta.observers.length) await loadObserver(parseInt(sel.value));
  window.__observerLoad = async (c) => { await loadObserver(c ?? parseInt(sel.value)); };
  window.__observerMode = (mo, t) => { setMode(mo); if (t !== undefined) setTick(t); };
}
export function observerShown() {
  const sel = $('obsSel');
  if (!S.meta.observers.length) { $('obs-side').innerHTML = '<div class="card">This run stored no observer files.</div>'; return; }
  if (!O) loadObserver(parseInt(sel.value)); else resize();
}
function play() {
  if (!playing || !O) return;
  if (tick >= O.n) { playing = false; $('obsPlay').textContent = '▶ play'; return; }
  setTick(tick + Math.max(1, Math.floor(O.n / 400))); setTimeout(play, 60);
}
