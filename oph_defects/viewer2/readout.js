// The Readout tab: one observer's patch of the screen with its repair network (left: the run's own data) beside the
// separately prepared free Dirac packets (right), solved on a finite momentum grid.
// Source IDs and centre classes select illustrative packets; no interacting field transfer is asserted.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadNpz } from './npz.js';
import { $, S, fmt } from './app.js';
import { PV, EDGES } from './walk_engine.js';
import { phases } from './cf1_engine.js';
import { createReadoutInspector } from './readout_inspector.js';

// ---------------------------------------------------------------- particles of the classes (charges: k = 6Y mod 6)
const CL = {
  1: { short: 'qL', name: 'left-quark-type candidate', color: '#ff7a45', shape: 'disc', hand: 'left', what: 'centre class compatible with a left-quark doublet; the class alone does not determine the full representation',
       q: '+2/3 (up) or −1/3 (down)', Y: '+1/6', colour: 'triplet', weak: 'doublet' },
  5: { short: 'q̄L', name: 'left-antiquark-type candidate', color: '#ff7a45', shape: 'ring', hand: 'right', what: 'the conjugate centre class of the left-quark candidate; a full antiparticle assignment requires its field dynamics',
       q: '−2/3 or +1/3', Y: '−1/6', colour: 'antitriplet', weak: 'doublet' },
  4: { short: 'qR', name: 'right-quark-type candidate', color: '#ffb347', shape: 'disc', hand: 'right', what: 'centre class compatible with right-quark singlets; it does not distinguish up from down',
       q: '+2/3 (up) or −1/3 (down)', Y: '+2/3 or −1/3', colour: 'triplet', weak: 'singlet' },
  2: { short: 'q̄R', name: 'right-antiquark-type candidate', color: '#ffb347', shape: 'ring', hand: 'left', what: 'the conjugate centre class of the right-quark candidate; a full antiparticle assignment requires its field dynamics',
       q: '−2/3 or +1/3', Y: '−2/3 or +1/3', colour: 'antitriplet', weak: 'singlet' },
  3: { short: 'ℓL', name: 'lepton-doublet-type candidate', color: '#3ee6a8', rim: '#4cc3ff', shape: 'duo', hand: 'left',
       what: 'a centre class shared by a lepton doublet, its conjugate and a Higgs-type doublet; it does not identify spin or particle versus antiparticle',
       q: '−1 or 0 (antiparticle +1 or 0)', Y: '−1/2 (antiparticle +1/2)', colour: 'none', weak: 'doublet' },
};
const ORDER = [1, 5, 4, 2, 3];
// units and rest angles: repair-law/work/SIM/cf1/runs/MASS_CHECK.json (frozen flux of the level-10 run as background)
const U = { tick: 1.4710399122721842e-23, cell: 4.410066711161824, hbar: 6.582119569e-22,
  theta: { qL: 0.03507760752054648, qR: 0.03233349142518348, lL: 0.011420331136484984, lR: 0 } };
const thetaOf = (k) => (k === 1 || k === 5 ? U.theta.qL : k === 4 || k === 2 ? U.theta.qR : U.theta.lL);
const mevOf = (th) => th * U.hbar / U.tick;
// box: n^3 points, one per h cells; display: one slice of the run advances the box by `slice` time units (h ticks each)
const F = { n: 32, h: 16, slice: 2, recEvery: 1.5, sigma: 1.5, sigmaD: 0.9, pmag: 0.55, cap: 9, size: 10, secPerSlice: 2.4 };

let inited = false, playing = true, speed = 1, showLabels = true, showTracks = true, showSpin = true;
let inspector = null;
let O = null, D = null, K = 0, slice = 0, frac = 0, t = 0, loadedFor = '', lastNow = 0;
let worker = null, workerReady = false, busy = false, seq = 0, frames = 0;
const t0Load = performance.now();
const parts = new Map();          // id -> particle in the box
const flashes = [];               // record, birth and death flashes in the box
const seamFlash = [];             // the observer's own repairs: [edge, hue, t0]
let renderer, scene, camera, controls, volTex, volMesh, boxGroup, patchDisc, obsGroup;
const glyphTex = {};

// ---------------------------------------------------------------- small helpers
const rgbOf = (hex) => { const c = new THREE.Color(hex); return [c.r, c.g, c.b]; };
function hash01(id, salt) { let h = (id * 2654435761 + salt * 40503) >>> 0; h ^= h >>> 15; h = Math.imul(h, 2246822519) >>> 0; h ^= h >>> 13; return (h >>> 0) / 4294967296; }
const toWorld = (g) => ((g + 0.5) / F.n - 0.5) * F.size;
function spinAlong(nv, sign) {            // two-component spinor along the unit vector nv (sign +1) or against it (-1)
  const th = Math.acos(Math.max(-1, Math.min(1, nv[2]))), ph = Math.atan2(nv[1], nv[0]), c = Math.cos(th / 2), s = Math.sin(th / 2);
  return sign > 0 ? [[c, 0], [s * Math.cos(ph), s * Math.sin(ph)]] : [[s, 0], [-c * Math.cos(ph), -c * Math.sin(ph)]];
}
function chiOf(cls, dir) {                // prepared handedness: left = X_record -1 with spin against the motion, right = +1 with it
  const right = CL[cls].hand === 'right', r = Math.SQRT1_2, rec = right ? [r, r] : [r, -r], sp = spinAlong(dir, right ? 1 : -1);
  const out = [];
  for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) out.push([rec[a] * sp[b][0], rec[a] * sp[b][1]]);
  return out;
}
function glyph(cls) {
  if (glyphTex[cls]) return glyphTex[cls];
  const st = CL[cls], c = document.createElement('canvas'); c.width = c.height = 128; const g = c.getContext('2d');
  g.shadowColor = st.color; g.shadowBlur = 26; g.lineWidth = 15;
  if (st.shape === 'ring') { g.strokeStyle = st.color; g.beginPath(); g.arc(64, 64, 36, 0, 2 * Math.PI); g.stroke(); }
  else if (st.shape === 'duo') { g.fillStyle = st.color; g.beginPath(); g.arc(64, 64, 25, 0, 2 * Math.PI); g.fill(); g.strokeStyle = st.rim; g.lineWidth = 9; g.beginPath(); g.arc(64, 64, 41, 0, 2 * Math.PI); g.stroke(); }
  else { g.fillStyle = st.color; g.beginPath(); g.arc(64, 64, 36, 0, 2 * Math.PI); g.fill(); }
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; glyphTex[cls] = tex; return tex;
}
let ringTex = null;
function ringTexture() {
  if (ringTex) return ringTex;
  const c = document.createElement('canvas'); c.width = c.height = 128; const g = c.getContext('2d');
  g.strokeStyle = '#ffffff'; g.lineWidth = 6; g.shadowColor = '#ffffff'; g.shadowBlur = 16; g.beginPath(); g.arc(64, 64, 48, 0, 2 * Math.PI); g.stroke();
  ringTex = new THREE.CanvasTexture(c); return ringTex;
}

// ---------------------------------------------------------------- the box (Three.js)
const VERT = `varying vec3 vPos;
void main() { vPos = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const FRAG = `precision highp float;
precision highp sampler3D;
uniform sampler3D uVol; uniform vec3 uCam; uniform float uGain; uniform int uSteps;
varying vec3 vPos;
void main() {
  vec3 d = normalize(vPos - uCam);
  vec3 inv = 1.0 / (d + vec3(1e-7));
  vec3 t0 = (vec3(-0.5) - uCam) * inv, t1 = (vec3(0.5) - uCam) * inv;
  vec3 a = min(t0, t1), b = max(t0, t1);
  float tin = max(max(max(a.x, a.y), a.z), 0.0), tout = min(min(b.x, b.y), b.z);
  if (tout <= tin) discard;
  float dt = (tout - tin) / float(uSteps);
  vec3 col = vec3(0.0);
  for (int i = 0; i < 200; i++) {
    if (i >= uSteps) break;
    vec3 p = uCam + d * (tin + (float(i) + 0.5) * dt) + 0.5;
    vec4 s = texture(uVol, p);
    col += s.rgb * (0.35 + s.a);
  }
  gl_FragColor = vec4(col * uGain * dt, 1.0);
}`;
function setupGL() {
  const canvas = $('roGl');
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  scene = new THREE.Scene(); scene.background = new THREE.Color('#04070f');
  camera = new THREE.PerspectiveCamera(36, 1, 0.05, 200); camera.up.set(0, 0, 1); camera.position.set(9.6, -11.2, 7.4);
  controls = new OrbitControls(camera, canvas); controls.enableDamping = true; controls.target.set(0, 0, 0);
  controls.autoRotate = true; controls.autoRotateSpeed = 0.35;
  canvas.addEventListener('pointerdown', () => { controls.autoRotate = false; });
  const N = F.n ** 3;
  volTex = new THREE.Data3DTexture(new Uint8Array(4 * N), F.n, F.n, F.n);
  volTex.format = THREE.RGBAFormat; volTex.type = THREE.UnsignedByteType; volTex.minFilter = volTex.magFilter = THREE.LinearFilter; volTex.unpackAlignment = 1; volTex.needsUpdate = true;
  const mat = new THREE.ShaderMaterial({ uniforms: { uVol: { value: volTex }, uCam: { value: new THREE.Vector3() }, uGain: { value: 2.6 }, uSteps: { value: 120 } },
    vertexShader: VERT, fragmentShader: FRAG, side: THREE.BackSide, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  volMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mat); volMesh.scale.setScalar(F.size); volMesh.renderOrder = 1; scene.add(volMesh);
  boxGroup = new THREE.Group(); scene.add(boxGroup);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(F.size, F.size, F.size)), new THREE.LineBasicMaterial({ color: '#30466f', transparent: true, opacity: 0.75 }));
  boxGroup.add(edges);
  // faint grid in the observer's plane
  const grid = new THREE.GridHelper(F.size, 16, '#1b2a48', '#132037'); grid.rotation.x = Math.PI / 2; grid.material.transparent = true; grid.material.opacity = 0.35; boxGroup.add(grid);
  // the observer: its carrier, an icosahedron whose three perpendicular two-fold axes are the axes of the box
  obsGroup = new THREE.Group(); scene.add(obsGroup);
  const sc = 0.32, pts = [];
  for (const [a, b] of EDGES) pts.push(...PV[a].map((v) => v * sc), ...PV[b].map((v) => v * sc));
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(new Array(pts.length).fill(0.85), 3));
  obsGroup.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95 })));
  obsGroup.userData.edges = g;
  const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringTexture(), color: '#ffffff', transparent: true, opacity: 0.35, depthWrite: false, blending: THREE.AdditiveBlending }));
  glow.scale.setScalar(1.6); obsGroup.add(glow);
  const axCol = ['#6fb1ff', '#a995ff', '#5fe3a1'];
  for (let i = 0; i < 3; i++) {
    const dir = new THREE.Vector3(i === 0 ? 1 : 0, i === 1 ? 1 : 0, i === 2 ? 1 : 0);
    obsGroup.add(new THREE.ArrowHelper(dir, new THREE.Vector3(0, 0, 0), 1.7, axCol[i], 0.26, 0.13));
  }
  window.addEventListener('resize', resize);
}
function resize() {
  if (!renderer) return;
  const c = $('roGl'), w = c.clientWidth, h = c.clientHeight;
  if (w && h) { renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
  const s = $('roScreen'); if (s.clientWidth) { s.width = s.clientWidth * devicePixelRatio; s.height = s.clientHeight * devicePixelRatio; }
  const k = $('roClocks'); if (k && k.clientWidth) { k.width = k.clientWidth * devicePixelRatio; k.height = k.clientHeight * devicePixelRatio; }
  if ($('roTransfer')) drawTransfer();
}

// ---------------------------------------------------------------- the worker
function startWorker() {
  worker = new Worker(new URL('./readout_worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'ready') { workerReady = true; return; }
    if (m.type !== 'frame') return;
    busy = false;
    if (m.seq !== seq) return;                                  // a reset happened meanwhile
    volTex.image.data = m.rgba; volTex.needsUpdate = true; frames++;
    if (frames === 1) console.log(`readout: first field frame, ${m.stats.length} particles, ${(performance.now() - t0Load).toFixed(0)} ms after load`);
    for (const st of m.stats) { const p = parts.get(st.id); if (p) p.stats = st; }
    for (const r of m.records) {
      const p = parts.get(r.id); if (!p) continue;
      p.recs.push({ ...r, x: [...r.x] }); if (p.recs.length > 18) p.recs.shift();
      p.transcript.push({ ...r, x: [...r.x], samplerDraws: [...r.samplerDraws] });
      p.from = p.cur.slice(); p.to = r.x.map(toWorld); p.moveT0 = performance.now();
      if (p.to.some((v, i) => Math.abs(v - p.from[i]) > F.size / 2)) p.from = p.to.slice();   // across the periodic boundary
      addFlash(p.to, CL[p.cls].color, 0.7, 1.1);
      p.lastRec = r.t; p.nrec++; if (r.convertedNegativeBand) p.nconverted++;
    }
  };
  worker.postMessage({ type: 'setup', n: F.n });
}
function requestFrame() {
  if (!workerReady || busy) return;
  const due = [];
  for (const p of parts.values()) if (t >= p.nextRec) { due.push(p.id); p.nextRec = t + F.recEvery; }
  busy = true; worker.postMessage({ type: 'frame', t, records: due, sigmaD: F.sigmaD, seq });
}

// ---------------------------------------------------------------- particles in the box
function makeParticle(d, dir, tNow, flash) {
  const s = (F.n / 2 - 2) / Math.max(1, O.radius);
  const g0 = [F.n / 2 + d.u * s, F.n / 2 + d.v * s, F.n / 2];
  const st = CL[d.cls], m = thetaOf(d.cls) * F.h, p0 = dir.map((v) => v * F.pmag);
  const preparation = { id: d.id, m, x0: g0, p0, sigma: F.sigma, chi: chiOf(d.cls, dir), t: tNow, color: rgbOf(st.color) };
  worker.postMessage({ type: 'add', parts: [preparation] });
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: glyph(d.cls), transparent: true, depthWrite: false }));
  sprite.scale.setScalar(0.62); sprite.renderOrder = 3; scene.add(sprite);
  const tg = new THREE.BufferGeometry(); tg.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(3 * 20), 3)); tg.setDrawRange(0, 0);
  const track = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: st.color, transparent: true, opacity: 0.55 })); scene.add(track);
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 0.8, new THREE.Color(st.color).lerp(new THREE.Color('#ffffff'), 0.45), 0.2, 0.12); scene.add(arrow);
  const lab = document.createElement('div'); lab.className = 'ro-lab'; lab.style.color = st.color;
  lab.innerHTML = `${st.short}<span>candidate</span>`; lab.title = `${st.name}; centre class ${d.cls}`; lab.onclick = () => showInfo(d.id); $('roLabels').appendChild(lab);
  lab.setAttribute('role', 'button'); lab.tabIndex = 0; lab.setAttribute('aria-label', `Inspect defect ${d.id}, ${st.name}`);
  lab.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); showInfo(d.id); } });
  const w = g0.map(toWorld);
  const p = { id: d.id, cls: d.cls, born: tNow, g0, dir, preparation, sprite, track, arrow, lab, recs: [], transcript: [], cur: w.slice(), from: w.slice(), to: w.slice(), moveT0: 0,
    nextRec: tNow + 0.5 + hash01(d.id, 5) * F.recEvery, stats: null, lastRec: null, nrec: 0, nconverted: 0 };
  parts.set(d.id, p);
  if (flash) addFlash(w, '#ffe45c', 1.0, 1.6);
}
function dropParticle(id, flash) {
  const p = parts.get(id); if (!p) return;
  if (flash) addFlash(p.cur, '#ff5f7a', 1.0, 1.6);
  scene.remove(p.sprite); scene.remove(p.track); scene.remove(p.arrow); p.sprite.material.dispose(); p.track.geometry.dispose(); p.lab.remove();
  parts.delete(id); worker.postMessage({ type: 'remove', ids: [id] });
}
function addFlash(pos, color, dur, size) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringTexture(), color, transparent: true, opacity: 1, depthWrite: false, blending: THREE.AdditiveBlending }));
  s.position.set(...pos); s.scale.setScalar(0.2); s.renderOrder = 4; scene.add(s); flashes.push({ s, t0: performance.now(), dur: dur * 1000, size });
}
function dirFor(d) { const v = [hash01(d.id, 1) - 0.5, hash01(d.id, 2) - 0.5, (hash01(d.id, 3) - 0.5) * 0.8]; const l = Math.hypot(...v) || 1; return v.map((x) => x / l); }
// which defects of slice j the box follows: the ones it already holds, then others in a fixed pseudo-random order,
// so that the followed ones spread over the whole patch
function chooseIds(j) {
  const list = O.frames[j] || [], keep = list.filter((d) => parts.has(d.id)), rest = list.filter((d) => !parts.has(d.id)).sort((a, b) => hash01(a.id, 11) - hash01(b.id, 11));
  return keep.concat(rest).slice(0, F.cap);
}
function resetBox(j) {
  for (const id of [...parts.keys()]) dropParticle(id, false);
  seq++; busy = false; t = 0; frac = 0; slice = j;
  if (worker) worker.postMessage({ type: 'clear' });
  for (const d of chooseIds(j)) makeParticle(d, dirFor(d), 0, false);
}
function stepSlice(j) {
  const prev = new Set((O.frames[slice] || []).map((d) => d.id));
  slice = j; frac = 0;
  const now = new Map((O.frames[j] || []).map((d) => [d.id, d]));
  for (const id of [...parts.keys()]) if (!now.has(id)) dropParticle(id, true);
  const chosen = chooseIds(j), born = chosen.filter((d) => !parts.has(d.id)), dirs = new Map();
  for (let a = 0; a < born.length; a++) {                     // a new pair of conjugate classes close together leaves back to back
    if (dirs.has(born[a].id)) continue; dirs.set(born[a].id, dirFor(born[a]));
    if (prev.has(born[a].id)) continue;
    for (let b = a + 1; b < born.length; b++) {
      if (dirs.has(born[b].id) || (born[a].cls + born[b].cls) % 6 !== 0) continue;
      if (Math.hypot(born[a].u - born[b].u, born[a].v - born[b].v) > 2.5) continue;
      dirs.set(born[b].id, dirs.get(born[a].id).map((x) => -x)); break;
    }
  }
  for (const d of born) makeParticle(d, dirs.get(d.id), t, !prev.has(d.id));
  updateLegend();
}

// ---------------------------------------------------------------- the screen panel (2D)
function prepareScreen() {
  const m = O.patch_cells.length, sp = S.meta.geometry.cell_spacing, fr = O.frame;
  O.m = m; O.uv = new Float32Array(2 * m); O.radius = 0;
  const tan = (x, y, z) => [(x * fr[0] + y * fr[3] + z * fr[6]) / sp, (x * fr[1] + y * fr[4] + z * fr[7]) / sp];
  for (let c = 0; c < m; c++) { const [u, v] = tan(O.patch_cell_xyz[3 * c], O.patch_cell_xyz[3 * c + 1], O.patch_cell_xyz[3 * c + 2]); O.uv[2 * c] = u; O.uv[2 * c + 1] = v; O.radius = Math.max(O.radius, Math.hypot(u, v)); }
  const seen = new Set(), links = [];
  for (let e = 0; e < O.patch_seams.length / 3; e++) {
    const a = Math.floor(O.patch_seams[3 * e] / 12), b = Math.floor(O.patch_seams[3 * e + 1] / 12);
    if (a === b || a >= m || b >= m) continue; const key = Math.min(a, b) * m + Math.max(a, b); if (seen.has(key)) continue; seen.add(key); links.push(a, b);
  }
  O.links = links;
  K = O.cut_depth.length; const P = 12 * m;
  O.dev = new Float32Array(K * m); O.act = new Float32Array(K * m); O.actMax = new Float32Array(K);
  for (let j = 0; j < K; j++) {
    let mean = 0;
    for (let c = 0; c < m; c++) { let s = 0; for (let q = 0; q < 12; q++) s += O.cut_patch_loads[j * P + 12 * c + q]; O.dev[j * m + c] = s; mean += s; }
    mean /= m; let sd = 0; for (let c = 0; c < m; c++) { O.dev[j * m + c] -= mean; sd += O.dev[j * m + c] ** 2; } sd = Math.sqrt(sd / m) || 1;
    for (let c = 0; c < m; c++) O.dev[j * m + c] = Math.max(-1, Math.min(1, O.dev[j * m + c] / (2.2 * sd)));
    if (j) { let mx = 0; for (let c = 0; c < m; c++) { let s = 0; for (let q = 0; q < 12; q++) s += Math.abs(O.cut_patch_loads[j * P + 12 * c + q] - O.cut_patch_loads[(j - 1) * P + 12 * c + q]); O.act[j * m + c] = s; mx = Math.max(mx, s); } O.actMax[j] = mx || 1; }
  }
  // defects inside the patch, slice by slice
  O.frames = Array.from({ length: K }, () => []);
  const c0 = O.carrier[0];
  if (D && D['o' + c0 + '_cut']) {
    const cut = D['o' + c0 + '_cut'], cls = D['o' + c0 + '_cls'], ids = D['o' + c0 + '_id'], xyz = D['o' + c0 + '_xyz'];
    for (let i = 0; i < cut.length; i++) {
      const k = ((cls[i] % 6) + 6) % 6; if (!CL[k] || cut[i] >= K) continue;
      const [u, v] = tan(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
      O.frames[cut[i]].push({ id: ids[i], cls: k, u, v });
    }
  }
  O.byId = O.frames.map((f) => new Map(f.map((d) => [d.id, d])));
  // the observer's own records between consecutive slices
  O.tick = (j) => { let v = -1; for (let q = Math.min(j, K - 1); q >= 0 && v < 0; q--) v = O.cut_ticks[q]; return Math.max(0, v); };
}
function drawScreen(now) {
  const cv = $('roScreen'); if (!O?.links || !O?.uv || !O?.dev || !O?.act || !O?.frames || !cv.width) return;
  const g = cv.getContext('2d'), W = cv.width, H = cv.height, dpr = devicePixelRatio, m = O.m;
  const bg = g.createRadialGradient(W / 2, H / 2, 10, W / 2, H / 2, Math.max(W, H) * 0.7); bg.addColorStop(0, '#0d1834'); bg.addColorStop(1, '#03050b');
  g.fillStyle = bg; g.fillRect(0, 0, W, H);
  const sc = (Math.min(W, H) / 2 - 34 * dpr) / O.radius, cx = W / 2, cy = H / 2 + 6 * dpr;
  const X = (u) => cx + u * sc, Y = (v) => cy - v * sc;
  const j0 = slice, j1 = Math.min(K - 1, slice + 1), f = frac;
  // patch rim
  g.strokeStyle = 'rgba(63,224,208,.22)'; g.lineWidth = 1.2 * dpr; g.beginPath(); g.arc(cx, cy, O.radius * sc + 10 * dpr, 0, 2 * Math.PI); g.stroke();
  // glue seams
  g.strokeStyle = 'rgba(120,150,215,.14)'; g.lineWidth = dpr; g.beginPath();
  for (let e = 0; e < O.links.length; e += 2) { const a = O.links[e], b = O.links[e + 1]; g.moveTo(X(O.uv[2 * a]), Y(O.uv[2 * a + 1])); g.lineTo(X(O.uv[2 * b]), Y(O.uv[2 * b + 1])); }
  g.stroke();
  // carriers, coloured by load against the patch mean
  for (let c = 0; c < m; c++) {
    const v = (1 - f) * O.dev[j0 * m + c] + f * O.dev[j1 * m + c], a = Math.abs(v);
    g.fillStyle = v > 0 ? `rgba(255,157,74,${0.18 + 0.62 * a})` : `rgba(77,141,255,${0.18 + 0.62 * a})`;
    g.beginPath(); g.arc(X(O.uv[2 * c]), Y(O.uv[2 * c + 1]), (1.6 + 3.2 * a) * dpr, 0, 2 * Math.PI); g.fill();
  }
  // repairs on the way to the next slice: expanding rings
  if (j1 > j0) for (let c = 0; c < m; c++) {
    const a = O.act[j1 * m + c] / O.actMax[j1]; if (a < 0.08) continue;
    const ph = (f + hash01(c, 9)) % 1, r = (2 + 11 * ph) * dpr;
    g.strokeStyle = `rgba(160,240,255,${(1 - ph) * Math.min(1, 1.6 * a) * 0.75})`; g.lineWidth = 1.2 * dpr;
    g.beginPath(); g.arc(X(O.uv[2 * c]), Y(O.uv[2 * c + 1]), r, 0, 2 * Math.PI); g.stroke();
  }
  // the observer's carrier at the centre, its seams lit by its own repair records
  const tilt = 0.5, yaw = now / 9000, cyw = Math.cos(yaw), syw = Math.sin(yaw), ct = Math.cos(tilt), st = Math.sin(tilt), isc = 11 * dpr;
  const Q = PV.map(([x, y, z]) => { const x1 = cyw * x + syw * y, y1 = -syw * x + cyw * y; return [cx + isc * x1, cy - isc * (ct * z - st * y1)]; });
  const halo = g.createRadialGradient(cx, cy, 0, cx, cy, 34 * dpr); halo.addColorStop(0, 'rgba(255,255,255,.28)'); halo.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = halo; g.beginPath(); g.arc(cx, cy, 34 * dpr, 0, 2 * Math.PI); g.fill();
  const lit = new Map(); const tnow = performance.now();
  for (let i = seamFlash.length - 1; i >= 0; i--) { const [e, hue, t0] = seamFlash[i], a = 1 - (tnow - t0) / 450; if (a <= 0) { seamFlash.splice(i, 1); continue; } if (!lit.has(e) || lit.get(e)[1] < a) lit.set(e, [hue, a]); }
  EDGES.forEach(([a, b], e) => {
    const L = lit.get(e); g.strokeStyle = L ? `hsla(${L[0]},95%,65%,${0.35 + 0.65 * L[1]})` : 'rgba(230,236,250,.45)'; g.lineWidth = (L ? 2.4 : 1) * dpr;
    g.beginPath(); g.moveTo(Q[a][0], Q[a][1]); g.lineTo(Q[b][0], Q[b][1]); g.stroke();
  });
  // defects of the patch
  const A = O.byId[j0], B = O.byId[j1], inBox = new Set(parts.keys());
  const drawGlyph = (cls, x, y, alpha, rad) => {
    const st = CL[cls]; g.globalAlpha = alpha; g.shadowColor = st.color; g.shadowBlur = 12 * dpr;
    if (st.shape === 'ring') { g.strokeStyle = st.color; g.lineWidth = 2.6 * dpr; g.beginPath(); g.arc(x, y, rad, 0, 2 * Math.PI); g.stroke(); }
    else if (st.shape === 'duo') { g.fillStyle = st.color; g.beginPath(); g.arc(x, y, rad * 0.62, 0, 2 * Math.PI); g.fill(); g.strokeStyle = st.rim; g.lineWidth = 2 * dpr; g.beginPath(); g.arc(x, y, rad * 1.05, 0, 2 * Math.PI); g.stroke(); }
    else { g.fillStyle = st.color; g.beginPath(); g.arc(x, y, rad, 0, 2 * Math.PI); g.fill(); }
    g.shadowBlur = 0; g.globalAlpha = 1;
  };
  const rad = 5.2 * dpr;
  g.font = `${11 * dpr}px Inter, Helvetica, Arial`;
  for (const [id, d] of A) {
    const e = B.get(id), u = e ? (1 - f) * d.u + f * e.u : d.u, v = e ? (1 - f) * d.v + f * e.v : d.v, x = X(u), y = Y(v);
    const alpha = e || j1 === j0 ? 1 : 1 - f;
    drawGlyph(d.cls, x, y, alpha, rad);
    if (!e && j1 > j0) { g.strokeStyle = `rgba(255,95,122,${f})`; g.lineWidth = 1.6 * dpr; g.beginPath(); g.arc(x, y, rad + 10 * f * dpr, 0, 2 * Math.PI); g.stroke(); }
    if (inBox.has(id)) { g.strokeStyle = 'rgba(255,255,255,.55)'; g.lineWidth = dpr; g.beginPath(); g.arc(x, y, rad + 4 * dpr, 0, 2 * Math.PI); g.stroke(); }
    if (inspector?.selectedId === id) { g.strokeStyle = '#ffffff'; g.lineWidth = 2*dpr; g.beginPath(); g.arc(x,y,rad+8*dpr,0,2*Math.PI); g.stroke(); g.fillStyle='#ffffff'; g.fillText('#'+id,x+rad+5*dpr,y-10*dpr); }
    if (showLabels) { g.globalAlpha = alpha; g.fillStyle = CL[d.cls].color; g.fillText(CL[d.cls].short, x + rad + 4 * dpr, y + 4 * dpr); g.globalAlpha = 1; }
  }
  if (j1 > j0) for (const [id, e] of B) {
    if (A.has(id)) continue; const x = X(e.u), y = Y(e.v);
    drawGlyph(e.cls, x, y, f, rad);
    g.strokeStyle = `rgba(255,228,92,${1 - f})`; g.lineWidth = 1.8 * dpr; g.beginPath(); g.arc(x, y, rad + 12 * (1 - f) * dpr, 0, 2 * Math.PI); g.stroke();
  }
  // heading
  g.fillStyle = 'rgba(147,163,192,.95)'; g.font = `600 ${11 * dpr}px Inter, Helvetica, Arial`;
  g.fillText('THE SCREEN · REPAIR NETWORK', 14 * dpr, 20 * dpr);
  g.font = `${11.5 * dpr}px Inter, Helvetica, Arial`; g.fillStyle = 'rgba(197,208,230,.9)';
  g.fillText(`${S.name} · observer ${O.carrier[0]} · ${fmt(m)} carriers · slice ${slice + 1} of ${K} · its tick ${fmt(O.tick(slice))}`, 14 * dpr, 37 * dpr);
  g.fillText(`${(O.frames[slice] || []).length} defects in the patch`, 14 * dpr, H - 14 * dpr);
}
function fireSeams(j, fFrom, fTo) {               // the observer's records between slice j and j + 1 that fall in (fFrom, fTo]
  if (!O || j + 1 >= K) return;
  const a = O.tick(j), b = O.tick(j + 1); if (b <= a) return;
  const i0 = a + Math.floor(fFrom * (b - a)), i1 = a + Math.floor(fTo * (b - a)), now = performance.now();
  for (let i = i0; i < i1 && i < O.tape_slot.length; i++) { const s = O.tape_slot[i]; if (s < 30) seamFlash.push([s, (60 * Math.abs(O.tape_current[i])) % 360, now]); }
  if (seamFlash.length > 400) seamFlash.splice(0, seamFlash.length - 400);
}

// ---------------------------------------------------------------- frame loop
function advance(dt) {
  if (!O || !K) return;
  const df = dt * speed / F.secPerSlice, f0 = frac;
  frac += df; t += df * F.slice;
  fireSeams(slice, f0, Math.min(1, frac));
  if (frac >= 1) {
    if (slice + 1 < K - 1) stepSlice(slice + 1);
    else { frac = 1; playing = false; $('roPlay').textContent = '▶ play'; }
  }
  $('roSlice').value = slice;
}
function render3D(now) {
  if (!renderer) return;
  volMesh.material.uniforms.uCam.value.copy(volMesh.worldToLocal(camera.position.clone()));
  const host = $('roGl'), w = host.clientWidth, h = host.clientHeight, v = new THREE.Vector3();
  for (const p of parts.values()) {
    const k = Math.min(1, (now - p.moveT0) / 450), e = k * k * (3 - 2 * k);
    p.cur = p.from.map((a, i) => a + (p.to[i] - a) * e);
    p.sprite.position.set(...p.cur);
    // track of recorded points, broken where it crosses the periodic boundary
    const pos = p.track.geometry.attributes.position; let nseg = 0, prev = null;
    for (const r of p.recs.slice(-19)) { const q = r.x.map(toWorld); if (prev && q.some((a, i) => Math.abs(a - prev[i]) > F.size / 2)) nseg = 0; pos.setXYZ(nseg++, ...q); prev = q; }
    if (nseg) pos.setXYZ(nseg++, ...p.cur);
    pos.needsUpdate = true; p.track.geometry.setDrawRange(0, showTracks ? nseg : 0); p.track.geometry.computeBoundingSphere();
    const sv = p.stats ? p.stats.spin : null, sl = sv ? Math.hypot(...sv) : 0;
    p.arrow.visible = showSpin && sl > 0.05;
    if (p.arrow.visible) { p.arrow.position.set(...p.cur); p.arrow.setDirection(new THREE.Vector3(sv[0] / sl, sv[1] / sl, sv[2] / sl)); p.arrow.setLength(0.35 + 0.75 * sl, 0.2, 0.12); }
    v.set(...p.cur).project(camera);
    const hidden = !showLabels || v.z > 1;
    p.lab.style.display = hidden ? 'none' : '';
    if (!hidden) { p.lab.style.left = ((v.x + 1) / 2 * w) + 'px'; p.lab.style.top = ((1 - v.y) / 2 * h) + 'px'; }
  }
  for (let i = flashes.length - 1; i >= 0; i--) {
    const fl = flashes[i], a = (now - fl.t0) / fl.dur;
    if (a >= 1) { scene.remove(fl.s); fl.s.material.dispose(); flashes.splice(i, 1); continue; }
    fl.s.scale.setScalar(0.25 + fl.size * a); fl.s.material.opacity = 1 - a;
  }
  // the observer's carrier: seams lit by its own records, as on the screen
  const col = obsGroup.userData.edges.attributes.color, lit = new Map(), tnow = performance.now();
  for (const [e, hue, t0] of seamFlash) { const a = 1 - (tnow - t0) / 450; if (a > 0 && (!lit.has(e) || lit.get(e)[1] < a)) lit.set(e, [hue, a]); }
  const c = new THREE.Color();
  for (let e = 0; e < EDGES.length; e++) {
    const L = lit.get(e); if (L) c.setHSL(L[0] / 360, 0.95, 0.5 + 0.2 * L[1]); else c.setRGB(0.55, 0.6, 0.7);
    for (const q of [2 * e, 2 * e + 1]) col.setXYZ(q, c.r, c.g, c.b);
  }
  col.needsUpdate = true;
  controls.update(); renderer.render(scene, camera);
  const sec = t * F.h * U.tick;
  $('roTime').innerHTML = `field time <b>${fmt(Math.round(t * F.h))}</b> ticks = <b>${sec.toExponential(2)}</b> s`;
}
function loop(now) {
  requestAnimationFrame(loop);
  if (!$('page-readout').classList.contains('on')) { lastNow = now; return; }
  const dt = Math.min(0.1, (now - (lastNow || now)) / 1000); lastNow = now;
  if (O && playing) advance(dt);
  if (O) requestFrame();
  drawScreen(now); render3D(now); drawClocks(now);
  inspector?.update(now);
}

// ---------------------------------------------------------------- side panels
function showInfo(id) {
  const p = parts.get(id), box = $('roInfo'); if (!p) { box.style.display = 'none'; return; }
  inspector?.select(id);
  const st = CL[p.cls], th = thetaOf(p.cls), s = p.stats, last = p.recs[p.recs.length - 1];
  const fm = (g) => ((g - F.n / 2) * F.h * U.cell).toFixed(0);
  box.innerHTML = `<button class="ro-x" id="roInfoX">×</button>
    <div class="ro-ih" style="color:${st.color}">${st.short} <span>${st.name}</span></div>
    <div class="small" style="margin:2px 0 8px;color:var(--soft)">${st.what}</div>
    <table class="t ro-t">
      <tr><td>class of its flux</td><td>${p.cls}</td></tr>
      <tr><td>candidate electric charge</td><td>${st.q}</td></tr><tr><td>candidate hypercharge</td><td>${st.Y}</td></tr>
      <tr><td>colour</td><td>${st.colour}</td></tr><tr><td>weak isospin</td><td>${st.weak}</td></tr>
      <tr><td>prepared Dirac spin</td><td>½${s ? ` · now ⟨σ⟩ = (${s.spin.map((x) => x.toFixed(2)).join(', ')})` : ''}</td></tr>
      <tr><td>handedness, prepared</td><td>${st.hand}${s ? ` · now ⟨γ5⟩ = ${s.chirality.toFixed(2)}` : ''}</td></tr>
      <tr><td>rest angle per tick</td><td>${th.toFixed(4)}</td></tr><tr><td>model rest energy</td><td>${mevOf(th).toFixed(3)} MeV</td></tr>
      <tr><td>records so far</td><td>${p.nrec}</td></tr>
      <tr><td>last recorded position</td><td>${last ? last.x.map(fm).join(', ') + ' fm' : 'not yet recorded'}</td></tr></table>
    <div class="note">Position records use the observer’s axes 1, 2, 3. This family label is a candidate assignment. The model rest energy uses the displayed rest angle and supplied unit scale; it is not an identified Standard Model mass. Follow this source ID in the inspector below.</div>`;
  box.style.display = 'block'; $('roInfoX').onclick = () => { box.style.display = 'none'; };
}
function updateLegend() {
  const counts = {}; for (const d of (O && O.frames[slice]) || []) counts[d.cls] = (counts[d.cls] || 0) + 1;
  const inBox = {}; for (const p of parts.values()) inBox[p.cls] = (inBox[p.cls] || 0) + 1;
  $('ro-legend').innerHTML = ORDER.map((k) => {
    const st = CL[k];
    return `<span class="chip" style="--c:${st.color}"><i class="${st.shape === 'ring' ? 'ringi' : ''}" ${st.shape === 'duo' ? `style="box-shadow:0 0 0 2px ${st.rim},0 0 10px ${st.color}"` : ''}></i><b>${st.short}</b> ${st.name} <em>${counts[k] || 0} in the patch · ${inBox[k] || 0} in the box</em></span>`;
  }).join('');
}
const CLOCKS = [['qL', U.theta.qL, '#ff7a45'], ['qR', U.theta.qR, '#ffb347'], ['ℓL', U.theta.lL, '#3ee6a8'], ['massless', 0, '#c5d0e6']];
let clockTick = 0, clockLast = 0;
function drawClocks(now) {
  const cv = $('roClocks'); if (!cv || !cv.width) return;
  const g = cv.getContext('2d'), W = cv.width, H = cv.height, dpr = devicePixelRatio;
  if (clockLast) clockTick += Math.min(0.1, (now - clockLast) / 1000) * 40; clockLast = now;
  g.clearRect(0, 0, W, H);
  const n = CLOCKS.length, cw = W / n, r = Math.min(cw * 0.32, H * 0.3);
  CLOCKS.forEach(([name, th, color], i) => {
    const x = cw * (i + 0.5), y = H * 0.4, ang = th * clockTick;
    g.strokeStyle = 'rgba(147,163,192,.35)'; g.lineWidth = 1.5 * dpr; g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.stroke();
    for (let q = 0; q < 12; q++) { const a = q * Math.PI / 6; g.beginPath(); g.moveTo(x + 0.86 * r * Math.sin(a), y - 0.86 * r * Math.cos(a)); g.lineTo(x + r * Math.sin(a), y - r * Math.cos(a)); g.stroke(); }
    g.strokeStyle = color; g.lineWidth = 3 * dpr; g.shadowColor = color; g.shadowBlur = 10 * dpr;
    g.beginPath(); g.moveTo(x, y); g.lineTo(x + 0.82 * r * Math.sin(ang), y - 0.82 * r * Math.cos(ang)); g.stroke(); g.shadowBlur = 0;
    g.fillStyle = color; g.font = `700 ${12 * dpr}px Inter, Helvetica, Arial`; g.textAlign = 'center';
    g.fillText(name, x, y + r + 17 * dpr);
    g.fillStyle = 'rgba(197,208,230,.85)'; g.font = `${10.5 * dpr}px Inter, Helvetica, Arial`;
    g.fillText(th ? `θ ${th.toFixed(4)} · ${mevOf(th).toFixed(3)} MeV` : 'θ = 0: no clock at rest', x, y + r + 32 * dpr);
    g.fillText(th ? `one turn per ${Math.round(2 * Math.PI / th)} ticks` : 'moves at one cell per tick', x, y + r + 46 * dpr);
    g.textAlign = 'left';
  });
  g.fillStyle = 'rgba(147,163,192,.8)'; g.font = `${10.5 * dpr}px Inter, Helvetica, Arial`; g.fillText(`tick ${Math.floor(clockTick)}`, 8 * dpr, 14 * dpr);
}
function continuumTable() {
  const rows = [], dirs = [[1, 0, 0], [0.6, 0.8, 0], [1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)]], th = U.theta.qL;
  for (const pm of [0.4, 0.2, Math.PI / F.h, 0.1, 0.05]) {
    let worst = 0;
    for (const d of dirs) { const p = d.map((x) => x * pm), ew = phases(p, th, 'Y', true)[0], E = Math.hypot(pm, th); worst = Math.max(worst, Math.abs(ew - E) / E); }
    rows.push(`<tr${Math.abs(pm - Math.PI / F.h) < 1e-9 ? ' class="hl"' : ''}><td>${pm.toFixed(3)}${Math.abs(pm - Math.PI / F.h) < 1e-9 ? ' (one-axis grid edge)' : ''}</td><td class="num">${(100 * worst).toFixed(3)} %</td></tr>`);
  }
  return `<table class="t ro-t"><tr><th>momentum per cell</th><th>walk energy against √(p² + m²)</th></tr>${rows.join('')}</table>`;
}
function buildCards() {
  const qm = mevOf(U.theta.qL).toFixed(2), qr = mevOf(U.theta.qR).toFixed(2), lm = mevOf(U.theta.lL).toFixed(3), hbarTau = (U.hbar / U.tick).toFixed(2);
  const rows = [
    ['Candidate family', 'The run records a Z6 centre-flux class k. Under the proposed gauge representation, k = 6Y mod 6. Distinct representations can share this remainder.', 'quark-type classes 1 and 4, conjugates 5 and 2; class 3 is shared by lepton and Higgs candidates'],
    ['Electric charge', 'A full chosen representation supplies Y and weak level T3, so Q = T3 + Y. The centre class alone supplies neither value uniquely.', 'the table uses the candidate Standard Model weights, not a charge measurement in this load run'],
    ['Colour', 'The residue k mod 3 is a colour-centre label. The separate five-level quantum construction supplies full colour currents.', 'centre-neutral pairs and triples are possible; their neutrality alone does not prove confinement or hadron binding'],
    ['Weak isospin', 'The residue k mod 2 supplies centre parity. Doublet weights require the separate quantum representation.', 'a candidate doublet or singlet; the passive defect record does not measure all weak currents'],
    ['Spin and statistics', 'This viewer prepares a four-component Dirac spinor with rotation generator σ/2. The quantum source has an independently declared CAR occupation algebra.', 'spin expectation of the prepared packet; one-packet motion does not test exchange statistics or prove the run defect has spin ½'],
    ['Handedness', 'In this Dirac program γ5 = X_record. It commutes with the kinetic matrices and anticommutes with the mass matrix.', 'mass mixes the two chiral components; a massive positive-energy packet need not remain in its initially chosen chiral state'],
    ['Conjugation', 'Reversing flux orientation sends class k to 6 − k. The classical register has local pair-annihilation rules.', 'conjugate centre labels; relativistic antiparticle scattering is a separate test'],
    ['Space', 'The packet program declares three cubic routing axes. The icosahedron supplies the displayed orientation.', 'three-dimensional packet coordinates; the cubic program is not obtained merely by drawing three carrier axes'],
    ['Time', 'The finite Dirac Hamiltonian generates the packet clock. Playback advances it by a chosen number of ticks per source slice.', 'field time and the load-run repair clock remain separate; a modular clock identification needs its own reference-state proof'],
    ['Position', 'A complete Gaussian detector acts on the finite field and retains both of its energy-band outcomes.', 'a recorded detector centre, with a conditional state and finite spatial resolution'],
    ['Momentum', 'Fourier modes diagonalize translations of this same finite field. Its distribution is calculated from the packet amplitudes.', 'p = ħk; position and momentum are alternative settings, each with a distribution'],
    ['Rest energy', 'The specified Dirac mass angle θ gives the zero-momentum energy in the selected branch.', 'E0 = ħθ/τ; the displayed MeV scale uses electron calibration and is not a mass prediction'],
    ['Generation', 'A centre class does not distinguish electron, muon and tau, nor the three quark generations.', 'no generation or family mass hierarchy is extracted from this display'],
  ];
  $('ro-cards').innerHTML = `
    <div class="card ro-card ro-wide"><div class="k">What each property is</div>
      <table class="t ro-t ro-dict"><tr><th>Property</th><th>In the construction</th><th>What the observer reads</th></tr>
      ${rows.map((r) => `<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('')}</table></div>
    <div class="card ro-card"><div class="k">Rest mass</div>
      <canvas id="roClocks"></canvas>
      <p><b>What it is.</b> In this declared Dirac program, a nonzero mass angle rotates the two record components and mixes chirality. The corresponding rest frequency is θ per tick. For a blank record at zero momentum, a single mass pulse gives written-record probability sin²θ. At nonzero momentum the massless Dirac dispersion has group speed c; this is a property of the field equation used here.</p>
      <p><b>How it is calculated.</b> At rest the field goes as e<sup>−iθt</sup>, so the turning rate is an energy: E₀ = ħθ/τ, with τ the duration of one tick. One tick is τ = ħ/E<sub>cell</sub>, the inverse of the energy of one cell, so rest mass is m = E₀/c² = θE<sub>cell</sub>/c²; here E<sub>cell</sub> = ħ/τ = ${hbarTau} MeV. The benchmark interprets frozen level-10 flux as a supplied Higgs-type background and assigns θ = ${U.theta.qL.toFixed(4)} for left-handed quarks, ${U.theta.qR.toFixed(4)} for right-handed quarks, ${U.theta.lL.toFixed(4)} for left-handed leptons and 0 for the right-singlet class. These separate class values do not yet form the left/right components of one interacting massive lepton. With τ = ${U.tick.toExponential(3)} s, set by the electron, that is ${qm} MeV, ${qr} MeV and ${lm} MeV. The lepton-scale agreement fixes the clock; it is a calibration check, not an independent prediction.</p>
      <p class="note">One angle per class is insufficient for the observed particle spectrum. Full field identification, the generation dynamics and an independent comparison of masses remain required.</p></div>
    <div class="card ro-card"><div class="k">From the tick to the field equation</div>
      <ol class="ro-steps">
        <li>One tick: ψ → M(θ/2) S M(θ/2) ψ with S = (Q₃R)(Q₂R)(Q₁R), the order of the axes alternating.</li>
        <li>On a wave e<sup>ip·x</sup> each step Q<sub>i</sub>R multiplies by e<sup>−ip<sub>i</sub>α<sub>i</sub></sup>, α<sub>i</sub> = X<sub>record</sub> ⊗ σ<sub>i</sub>; the mass step by e<sup>−iθβ/2</sup>, β = Y<sub>record</sub> ⊗ 1.</li>
        <li>α₁, α₂, α₃ and β anticommute and square to one: they are Dirac matrices.</li>
        <li>The half-step, alternating-axis two-tick product is symmetric. If both p and θ scale as ε, it agrees with e<sup>−2i(α·p + θβ)</sup> up to O(ε³). Holding θ fixed while only shrinking p is a different limit.</li>
        <li>For wavelengths long against a cell: iħ∂<sub>t</sub>ψ = (−iħc α·∇ + βmc²)ψ, the Dirac equation, with c = one cell per tick and mc² = ħθ/τ. Its energies obey E² = (pc)² + (mc²)², the relation of special relativity.</li>
        <li>Supplied link phases can represent a gauge background. Dynamical gauge evolution, Gauss-constrained preparation and its interacting field limit need additional operators and proofs.</li></ol>
      ${continuumTable()}
      <p class="small">The box evolves the retained finite Fourier modes analytically, up to floating-point error, one grid point per ${F.h} cells: ${F.n} points = ${(F.n * F.h * U.cell / 1000).toFixed(2)} pm across, with periodic walls. Check (readout_check.mjs): one prepared long-wavelength packet has walk/Dirac fidelity about 0.997 after 24 ticks. This is a finite comparison, not equality or an interacting continuum proof. The tick also has cones away from p = 0; the box shows the selected low-momentum cone. <a href="readout_limit.md" target="_blank" rel="noopener">Finite routing bound and scope ↗</a></p></div>
    <div class="card ro-card ro-full"><div class="k">How charged and neutral matter start to move on the quantum source</div>
      <div class="ro-two"><canvas id="roTransfer"></canvas>
      <div><p>The audit's interacting source keeps every quantum link and the quantum Higgs. On it, a charged quantum hops to the next cell at first order in time: the chance to find it there grows as t². Its link records the charge it carried.</p>
      <p>The neutral composite of a lepton and the Higgs has no hopping term of its own. It still moves, through the links, but only at fifth order: the amplitude is A(t) = −(151/3840) t⁵ plus a bounded remainder, so the chance grows as t¹⁰. The tested spin-flip transfer is nonzero. The tested same-spin transfer is excluded by that source symmetry; this is a channel selection rule.</p>
      <p>Why fifth order: the source has no hopping term for the Higgs memory. The Higgs quantum moves only by turning into a fermion pair, so the hop splits by channel: −2i h³ (31 y_e² + 60 y_d² + 60 y_u²). The leading transfer is a kinetic matrix element. The onsite symmetry audit excludes a particular Majorana/spin-flip term, but does not establish an all-orders massless dispersion or identify a measured neutrino.</p>
      <p class="small">Charged: two-cell source, ‖R H ψ‖² = 3/4 (two_cell/PROPAGATION_PROOF.md). Neutral: three-dimensional source with 119 link states per link (peter_weyl_links/PROOF.md); channel split: repair-law/reports/NU3.md; mass-claim scope: the consolidated quantum-source audit. Leading terms in source time units; the proofs bound the full evolution at short times.</p></div></div></div>`;
  resize(); drawTransfer();
}
function drawTransfer() {
  const cv = $('roTransfer'); if (!cv || !cv.clientWidth) return;
  const dpr = devicePixelRatio, W = cv.width = cv.clientWidth * dpr, H = cv.height = cv.clientHeight * dpr, g = cv.getContext('2d');
  const x0 = 54 * dpr, x1 = W - 14 * dpr, y0 = 14 * dpr, y1 = H - 40 * dpr;
  const lt0 = -2, lt1 = 0.5, lp0 = -24, lp1 = 0.5;                       // log10 t and log10 P ranges
  const X = (lt) => x0 + (x1 - x0) * (lt - lt0) / (lt1 - lt0), Y = (lp) => y1 - (y1 - y0) * (lp - lp0) / (lp1 - lp0);
  g.clearRect(0, 0, W, H); g.font = `${11 * dpr}px Inter, Helvetica, Arial`; g.fillStyle = '#8090ae'; g.strokeStyle = 'rgba(147,163,192,.16)'; g.lineWidth = dpr;
  for (let lp = -24; lp <= 0; lp += 4) { g.beginPath(); g.moveTo(x0, Y(lp)); g.lineTo(x1, Y(lp)); g.stroke(); g.textAlign = 'right'; g.fillText(lp ? `10${String(lp).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d])}` : '1', x0 - 6 * dpr, Y(lp) + 4 * dpr); }
  for (const lt of [-2, -1, 0]) { g.beginPath(); g.moveTo(X(lt), y0); g.lineTo(X(lt), y1); g.stroke(); g.textAlign = 'center'; g.fillText(['0.01', '0.1', '1'][lt + 2], X(lt), y1 + 15 * dpr); }
  g.fillText('time, source units', (x0 + x1) / 2, y1 + 31 * dpr);
  g.save(); g.translate(13 * dpr, (y0 + y1) / 2); g.rotate(-Math.PI / 2); g.fillText('chance to have moved one cell', 0, 0); g.restore();
  const curve = (f, color, label, at) => {
    g.strokeStyle = color; g.lineWidth = 2.4 * dpr; g.beginPath(); let first = true;
    for (let k = 0; k <= 200; k++) { const lt = lt0 + (lt1 - lt0) * k / 200, lp = f(lt); if (lp < lp0 - 1) continue; const y = Y(Math.min(lp1, lp)); first ? g.moveTo(X(lt), y) : g.lineTo(X(lt), y); first = false; }
    g.stroke(); g.fillStyle = color; g.textAlign = 'left'; g.font = `600 ${12 * dpr}px Inter, Helvetica, Arial`; g.fillText(label, X(at) + 6 * dpr, Y(f(at)) - 6 * dpr); g.font = `${11 * dpr}px Inter, Helvetica, Arial`;
  };
  curve((lt) => Math.log10(0.75) + 2 * lt, '#ffb347', 'charged quantum: (3/4) t²', -1.6);
  curve((lt) => 2 * Math.log10(151 / 3840) + 10 * lt, '#3ee6a8', 'neutral composite: (151/3840)² t¹⁰', -0.95);
}

// ---------------------------------------------------------------- loading and controls
async function loadObserver(carrier) {
  loadedFor = S.name + ':' + carrier;
  $('roStatus').textContent = 'loading the observer…';
  O = await loadNpz(`${S.base}observers/obs_${carrier}.npz`);
  try { D = await loadNpz(S.base + 'view/observer_defects.npz'); } catch (e) { D = null; }
  prepareScreen();
  $('roSlice').max = Math.max(0, K - 2);
  const j0 = Math.min(K - 2, Math.max(0, O.frames.findIndex((f) => f.length > 0)));
  resetBox(j0); $('roSlice').value = j0; updateLegend();
  $('roStatus').textContent = '';
}
export function readoutShown() {
  if (!inited) {
    inited = true; setupGL(); startWorker(); buildCards();
    inspector = createReadoutInspector(
      () => ({ run: S.name, observer: O, slice, frac, time: t, parts, classes: CL, units: U, grid: F }),
      id => showInfo(id),
      () => { playing = false; $('roPlay').textContent = '▶ play'; });
    $('roPlay').onclick = () => { playing = !playing; $('roPlay').textContent = playing ? '❚❚ pause' : '▶ play'; };
    $('roSlice').addEventListener('input', (e) => { resetBox(parseInt(e.target.value)); updateLegend(); });
    $('roSpeed').onchange = (e) => { speed = parseFloat(e.target.value); };
    $('roLab').onchange = (e) => { showLabels = e.target.checked; };
    $('roTrk').onchange = (e) => { showTracks = e.target.checked; };
    $('roSpin').onchange = (e) => { showSpin = e.target.checked; };
    $('roObs').onchange = (e) => loadObserver(parseInt(e.target.value));
    requestAnimationFrame(loop);
    window.__readout = { step: (n) => { for (let i = 0; i < n; i++) advance(0.1); }, parts, state: () => ({ slice, frac, t, n: parts.size, frames, workerReady, busy }) };
  }
  const sel = $('roObs');
  if (!S.meta || !S.meta.observers || !S.meta.observers.length) { $('roStatus').textContent = 'This run stored no observer files.'; return; }
  if (sel.dataset.run !== S.name) {
    sel.innerHTML = S.meta.observers.map((ob) => `<option value="${ob.carrier}">observer ${ob.carrier}: ${ob.role}</option>`).join('');
    sel.dataset.run = S.name;
  }
  resize();
  if (loadedFor.split(':')[0] !== S.name) loadObserver(parseInt(sel.value));
}
