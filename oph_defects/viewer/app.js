// OPH defect catalogue viewer: streams snapshots of a Z6 run and shows every
// protected defect on the refined carrier, colored by family, with a card
// per identity drawn from the complete catalogue.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const FAMILY = {
  1: { name: 'A', label: 'elementary twist', color: '#ffb44a', anti: false },
  5: { name: 'A', label: 'elementary twist, conjugate', color: '#ffb44a', anti: true },
  2: { name: 'B', label: 'double twist', color: '#a995ff', anti: false },
  4: { name: 'B', label: 'double twist, conjugate', color: '#a995ff', anti: true },
  3: { name: 'C', label: 'half-turn twist, self-conjugate', color: '#3fe0d0', anti: false },
};
// Best-justified reading, a hypothesis: the class is the hypercharge label q = 6Y mod 6 of the
// matter the twist encloses (book ch. 28: the dial part of a twist is its charge). In the Standard
// Model the Z6 congruence 2t + 3d + q = 0 mod 6 makes colour triality t and weak duality d
// functions of q mod 6: odd q is a weak doublet, q not 0 mod 3 is coloured.
const GUESS = {
  1: 'coloured weak doublet: the quark doublet Q_L (u_L, d_L); class 5 is its antiparticle',
  5: 'coloured weak doublet: the anti-quark doublet; class 1 is its antiparticle',
  2: 'coloured weak singlet: a right-handed quark conjugate (u^c has q = -4, d^c has q = 2; both read 2 mod 6)',
  4: 'coloured weak singlet: a right-handed quark (u_R or d_R); conjugate of class 2',
  3: 'colourless weak doublet: the lepton doublet L, its antiparticle, or the Higgs doublet (all read 3 mod 6)',
};
const GUESS_NOTE = 'A Z6 register reads hypercharge only mod 6, so it cannot tell u^c from d^c or L from the Higgs; a flat face (class 0) is where colourless singlets such as e^c, neutrinos and the gauge fields would sit. An integer winding register would lift the class to the full label 6Y.';
const CAUSE = ['alive', 'annihilation', 'fusion', 'split', 'exchange'];
// Hypercharge labels q = 6Y of the five matter blocks. A flux of class k gives block q the
// phase exp(2 pi i k q / 6); the block notices the defect only when k*q is not 0 mod 6.
const BLOCKS = [['Q_L', 1], ['u^c', -4], ['d^c', 2], ['L', -3], ['e^c', 6]];
function visibility(k) {
  const sees = [], blind = [];
  for (const [name, q] of BLOCKS) ((((k * q) % 6) + 6) % 6 ? sees : blind).push(name);
  return { sees, blind };
}

let run, level, n, baseFaces, baseVerts, snaps = [], snapCache = new Map(), cat = null;
let tickIndex = 0, playing = false, selected = -1, follow = false, view = '3d';

// ------------------------------------------------------------ geometry
function decodeFace(f) {
  const nn = n * n;
  const b = Math.floor(f / nn), local = f - b * nn;
  const p = Math.floor(local / n), q = local - p * n;
  const up = p + q <= n - 1;
  return { b, i: up ? p : n - 1 - p, j: up ? q : n - 1 - q, up };
}
function facePos(f, out) {
  const { b, i, j, up } = decodeFace(f);
  const [a, bb, c] = baseFaces[b];
  const A = baseVerts[a], B = baseVerts[bb], C = baseVerts[c];
  const ci = (up ? i + 1 / 3 : i + 2 / 3) / n, cj = (up ? j + 1 / 3 : j + 2 / 3) / n;
  const x = A[0] * (1 - ci - cj) + B[0] * ci + C[0] * cj;
  const y = A[1] * (1 - ci - cj) + B[1] * ci + C[1] * cj;
  const z = A[2] * (1 - ci - cj) + B[2] * ci + C[2] * cj;
  const m = Math.hypot(x, y, z);
  out[0] = x / m; out[1] = y / m; out[2] = z / m;
  return out;
}
function lonlat(p) { return [Math.atan2(p[1], p[0]), Math.asin(Math.max(-1, Math.min(1, p[2])))]; }

// ------------------------------------------------------------ data
async function gunzip(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error('fetch failed ' + url);
  const ds = new DecompressionStream('gzip');
  const buf = await new Response(r.body.pipeThrough(ds)).arrayBuffer();
  return buf;
}
let showIntra = false;
async function loadSnapshot(k) {
  const key = showIntra && snaps[k].intra_file ? k + 0.5 : k;
  if (snapCache.has(key)) return snapCache.get(key);
  let snap = await decodeSnapshot(snaps[k].file, snaps[k].tick);
  if (showIntra && snaps[k].intra_file) {
    const extra = await decodeSnapshot(snaps[k].intra_file, snaps[k].tick);
    snap = mergeSnapshots(snap, extra);
  }
  snapCache.set(key, snap);
  if (snapCache.size > 400) snapCache.delete(snapCache.keys().next().value);
  return snap;
}
function mergeSnapshots(a, b) {
  const count = a.count + b.count;
  const faces = new Int32Array(count); faces.set(a.faces); faces.set(b.faces, a.count);
  const cls = new Uint8Array(count); cls.set(a.cls); cls.set(b.cls, a.count);
  const ids = new Int32Array(count); ids.set(a.ids); ids.set(b.ids, a.count);
  const pos = new Float32Array(3 * count); pos.set(a.pos); pos.set(b.pos, 3 * a.count);
  const byId = new Map(a.byId); for (const [id, i] of b.byId) byId.set(id, i + a.count);
  return { tick: a.tick, count, faces, cls, ids, pos, byId };
}
async function decodeSnapshot(file, tick) {
  const buf = await gunzip('/run/' + file);
  const dv = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf.slice(0, 4)));
  const count = dv.getUint32(8, true);
  let o = 12;
  const faces = new Int32Array(buf.slice(o, o + 4 * count)); o += 4 * count;
  const cls = new Uint8Array(buf.slice(o, o + count)); o += count;
  const ids = new Int32Array(buf.slice(o, o + 4 * count)); o += 4 * count;
  let pos;
  if (magic === 'Z6SX') pos = new Float32Array(buf.slice(o, o + 12 * count));
  else {
    pos = new Float32Array(3 * count);
    const tmp = [0, 0, 0];
    for (let i = 0; i < count; i++) { facePos(faces[i], tmp); pos[3 * i] = tmp[0]; pos[3 * i + 1] = tmp[1]; pos[3 * i + 2] = tmp[2]; }
  }
  const byId = new Map();
  for (let i = 0; i < count; i++) byId.set(ids[i], i);
  const snap = { tick, count, faces, cls, ids, pos, byId };
  if (run.geometry.coords) { if (!facePosCache) facePosCache = new Map(); for (let i = 0; i < count; i++) if (!facePosCache.has(faces[i])) facePosCache.set(faces[i], [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]]); }
  return snap;
}
async function loadCatalogue() {
  const buf = await gunzip('/run/catalogue.bin.gz');
  const dv = new DataView(buf);
  const count = dv.getUint32(4, true);
  let o = 8;
  const take = (ctor, bytes) => { const a = new ctor(buf.slice(o, o + bytes * count)); o += bytes * count; return a; };
  const c = {};
  c.cls = take(Uint8Array, 1); c.birth = take(Int32Array, 4); c.death = take(Int32Array, 4); c.bface = take(Int32Array, 4);
  c.lface = take(Int32Array, 4); c.hops = take(Int32Array, 4); c.partner = take(Int32Array, 4); c.p1 = take(Int32Array, 4);
  c.p2 = take(Int32Array, 4); c.cause = take(Uint8Array, 1); c.c1 = take(Int32Array, 4); c.c2 = take(Int32Array, 4);
  c.count = count;
  c.lifetime = new Int32Array(count);
  if (run.geometry.coords) {
    try {
      const xb = await gunzip('/run/catalogue_xyz.bin.gz');
      const n = new DataView(xb).getUint32(4, true);
      c.bxyz = new Float32Array(xb.slice(8, 8 + 12 * n)); c.lxyz = new Float32Array(xb.slice(8 + 12 * n, 8 + 24 * n));
    } catch (e) { c.bxyz = null; }
  }
  const last = run.config.ticks;
  for (let i = 0; i < count; i++) c.lifetime[i] = (c.death[i] < 0 ? last : c.death[i]) - c.birth[i];
  return c;
}

// ------------------------------------------------------------ 3D scene
const canvas = $('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
const scene = new THREE.Scene();
scene.background = new THREE.Color('#090d16');
const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 50);
camera.position.set(0, 0, 3.1);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.minDistance = 1.08; controls.maxDistance = 8;
let pointsFilled, pointsRing, trail, highlight;

function spriteTexture(ring) {
  const c = document.createElement('canvas'); c.width = c.height = 64;
  const g = c.getContext('2d');
  g.clearRect(0, 0, 64, 64);
  g.beginPath(); g.arc(32, 32, ring ? 24 : 26, 0, Math.PI * 2);
  if (ring) { g.lineWidth = 9; g.strokeStyle = '#fff'; g.stroke(); }
  else { g.fillStyle = '#fff'; g.fill(); }
  const t = new THREE.CanvasTexture(c); t.needsUpdate = true; return t;
}
function buildSphere(displayLevel) {
  // the carrier at a display level: dark surface plus faint seams
  const nn = 1 << displayLevel;
  const verts = [], idx = [];
  const key = new Map();
  const vid = (b, i, j) => {
    const [a, bb, c] = baseFaces[b];
    const A = baseVerts[a], B = baseVerts[bb], C = baseVerts[c];
    const x = A[0] * (nn - i - j) + B[0] * i + C[0] * j, y = A[1] * (nn - i - j) + B[1] * i + C[1] * j, z = A[2] * (nn - i - j) + B[2] * i + C[2] * j;
    const k = [x.toFixed(6), y.toFixed(6), z.toFixed(6)].join(',');
    if (key.has(k)) return key.get(k);
    const m = Math.hypot(x, y, z); verts.push(x / m, y / m, z / m);
    key.set(k, verts.length / 3 - 1); return key.get(k);
  };
  for (let b = 0; b < 20; b++) for (let i = 0; i < nn; i++) for (let j = 0; j < nn - i; j++) {
    idx.push(vid(b, i, j), vid(b, i + 1, j), vid(b, i, j + 1));
    if (i + j < nn - 1) idx.push(vid(b, i + 1, j), vid(b, i + 1, j + 1), vid(b, i, j + 1));
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  geo.setIndex(idx); geo.computeVertexNormals();
  const surf = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: '#0f1a2c', roughness: 0.95, metalness: 0.0, transparent: true, opacity: 0.96 }));
  surf.scale.setScalar(0.992);
  scene.add(surf);
  const wire = new THREE.LineSegments(new THREE.WireframeGeometry(geo), new THREE.LineBasicMaterial({ color: '#3fe0d0', transparent: true, opacity: 0.09 }));
  wire.scale.setScalar(0.996);
  scene.add(wire);
  // the twenty base seams, brighter
  const bverts = [];
  const seen = new Set();
  for (const [a, b, c] of baseFaces) for (const [u, v] of [[a, b], [b, c], [c, a]]) {
    const k = Math.min(u, v) + ':' + Math.max(u, v); if (seen.has(k)) continue; seen.add(k);
    const P = baseVerts[u], Q = baseVerts[v];
    for (let s = 0; s < 24; s++) {
      const t0 = s / 24, t1 = (s + 1) / 24;
      for (const t of [t0, t1]) { const x = P[0] * (1 - t) + Q[0] * t, y = P[1] * (1 - t) + Q[1] * t, z = P[2] * (1 - t) + Q[2] * t; const m = Math.hypot(x, y, z); bverts.push(x / m * 1.001, y / m * 1.001, z / m * 1.001); }
    }
  }
  const bg = new THREE.BufferGeometry(); bg.setAttribute('position', new THREE.Float32BufferAttribute(bverts, 3));
  scene.add(new THREE.LineSegments(bg, new THREE.LineBasicMaterial({ color: '#3fe0d0', transparent: true, opacity: 0.35 })));
  scene.add(new THREE.AmbientLight('#ffffff', 0.55));
  const l = new THREE.DirectionalLight('#dfe8ff', 1.1); l.position.set(2, 3, 4); scene.add(l);
}
function buildPoints() {
  const mk = (ring) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(0), 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(0), 3));
    const m = new THREE.PointsMaterial({ size: 0.028, vertexColors: true, map: spriteTexture(ring), transparent: true, alphaTest: 0.3, sizeAttenuation: true, depthWrite: false });
    const p = new THREE.Points(g, m); p.frustumCulled = false; scene.add(p); return p;
  };
  pointsFilled = mk(false); pointsRing = mk(true);
  const hg = new THREE.RingGeometry(0.02, 0.028, 32);
  highlight = new THREE.Mesh(hg, new THREE.MeshBasicMaterial({ color: '#ffffff', side: THREE.DoubleSide, transparent: true, opacity: 0.95 }));
  highlight.visible = false; scene.add(highlight);
  trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.8 }));
  trail.visible = false; scene.add(trail);
}
let bulkGroup = null, bulkSphere = null;
const BULK_LAYERS = 24;
async function showBulk(k) {
  // the consensus bulk: every observer's records glued; depth is record lag
  if (!bulkGroup) {
    bulkGroup = new THREE.Group(); scene.add(bulkGroup);
    bulkSphere = new THREE.Mesh(new THREE.SphereGeometry(0.985, 64, 48),
      new THREE.MeshBasicMaterial({ color: '#0f1a2c', transparent: true, opacity: 0.18, side: THREE.BackSide, depthWrite: false }));
    scene.add(bulkSphere);
  }
  while (bulkGroup.children.length) { const c = bulkGroup.children.pop(); c.geometry.dispose(); c.material.dispose(); }
  const lo = Math.max(0, k - BULK_LAYERS);
  for (let j = k - 1; j >= lo; j--) {
    const s = await loadSnapshot(j);
    const lag = (k - j) / BULK_LAYERS;
    const r = 0.98 - 0.86 * lag;
    const P = new Float32Array(3 * s.count), Cc = new Float32Array(3 * s.count);
    for (let i = 0; i < s.count; i++) {
      P[3 * i] = s.pos[3 * i] * r; P[3 * i + 1] = s.pos[3 * i + 1] * r; P[3 * i + 2] = s.pos[3 * i + 2] * r;
      const col = COLORS[s.cls[i]]; Cc[3 * i] = col.r; Cc[3 * i + 1] = col.g; Cc[3 * i + 2] = col.b;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(Cc, 3));
    const m = new THREE.PointsMaterial({ size: 0.012 * parseFloat($('sizeSel').value) * Math.min(1, Math.sqrt(30000 / Math.max(1, s.count))), vertexColors: true, transparent: true, opacity: 0.85 - 0.7 * lag, depthWrite: false, sizeAttenuation: true });
    const pts = new THREE.Points(g, m); pts.frustumCulled = false; bulkGroup.add(pts);
  }
}
function setBulkVisible(on) {
  if (bulkGroup) bulkGroup.visible = on;
  if (bulkSphere) bulkSphere.visible = on;
  scene.traverse((o) => { if (o.isMesh && o !== bulkSphere && o !== highlight && o.material && o.material.color && o.geometry.index) o.material.opacity = on ? 0.08 : 0.96; });
}
function colorOf(k) { return new THREE.Color(FAMILY[k].color); }
const COLORS = {}; for (const k of [1, 2, 3, 4, 5]) COLORS[k] = colorOf(k);

function showSnapshot(snap) {
  const fill = [], fillC = [], ring = [], ringC = [];
  const sz = parseFloat($('sizeSel').value);
  for (let i = 0; i < snap.count; i++) {
    const k = snap.cls[i]; const col = COLORS[k]; const r = 1.004;
    const arrP = FAMILY[k].anti ? ring : fill, arrC = FAMILY[k].anti ? ringC : fillC;
    arrP.push(snap.pos[3 * i] * r, snap.pos[3 * i + 1] * r, snap.pos[3 * i + 2] * r);
    arrC.push(col.r, col.g, col.b);
  }
  for (const [pts, P, C] of [[pointsFilled, fill, fillC], [pointsRing, ring, ringC]]) {
    pts.geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(P), 3));
    pts.geometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(C), 3));
    pts.geometry.computeBoundingSphere();
    pts.material.size = 0.03 * sz * Math.pow(2, (7 - level) * 0.5) * Math.min(1, Math.sqrt(30000 / Math.max(1, snap.count)));
  }
  updateHighlight(snap);
  drawMap(snap);
  drawObserver(snap);
  updateNow(snap);
  if (view === 'bulk') showBulk(tickIndex);
}
function updateHighlight(snap) {
  if (selected < 0 || !snap.byId.has(selected)) { highlight.visible = false; return; }
  const i = snap.byId.get(selected);
  const p = new THREE.Vector3(snap.pos[3 * i], snap.pos[3 * i + 1], snap.pos[3 * i + 2]);
  highlight.position.copy(p.clone().multiplyScalar(1.006));
  highlight.lookAt(p.clone().multiplyScalar(2));
  const s = 1.3 * Math.pow(2, Math.max(0, 7 - level) * 0.5);
  highlight.scale.setScalar(s);
  highlight.visible = true;
  if (follow) {
    const d = camera.position.length();
    camera.position.copy(p.clone().multiplyScalar(d));
    controls.update();
  }
}
async function drawTrail(id) {
  // worldline from the snapshots loaded so far plus those around the lifetime
  if (!cat) return;
  const b = cat.birth[id], d = cat.death[id] < 0 ? run.config.ticks : cat.death[id];
  const pts = [];
  const lo = Math.max(0, tickIndex - 60), hi = Math.min(snaps.length - 1, tickIndex + 60);
  for (let k = lo; k <= hi; k++) {
    if (snaps[k].tick < b || snaps[k].tick > d) continue;
    const s = await loadSnapshot(k);
    if (s.byId.has(id)) { const i = s.byId.get(id); pts.push(s.pos[3 * i] * 1.006, s.pos[3 * i + 1] * 1.006, s.pos[3 * i + 2] * 1.006); }
  }
  trail.geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(pts), 3));
  trail.visible = pts.length >= 6 && selected === id;
}

// ------------------------------------------------------------ picking
const ray = new THREE.Raycaster();
ray.params.Points.threshold = 0.012;
canvas.addEventListener('pointerdown', (ev) => { canvas._down = [ev.clientX, ev.clientY]; });
canvas.addEventListener('pointerup', async (ev) => {
  if (!canvas._down || Math.hypot(ev.clientX - canvas._down[0], ev.clientY - canvas._down[1]) > 4) return;
  const rect = canvas.getBoundingClientRect();
  const m = new THREE.Vector2(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
  ray.setFromCamera(m, camera);
  ray.params.Points.threshold = 0.012 * Math.pow(2, Math.max(0, 7 - level) * 0.5) * camera.position.length() / 3;
  const hits = ray.intersectObjects([pointsFilled, pointsRing]);
  if (!hits.length) return;
  // nearest to the camera, and in front of the sphere
  hits.sort((a, b) => a.distance - b.distance);
  const hit = hits[0];
  const snap = await loadSnapshot(tickIndex);
  const isRing = hit.object === pointsRing;
  // map the point index back to the snapshot row
  let row = -1, seen = -1;
  for (let i = 0; i < snap.count; i++) { if (FAMILY[snap.cls[i]].anti === isRing) { seen++; if (seen === hit.index) { row = i; break; } } }
  if (row >= 0) select(snap.ids[row]);
});

// ------------------------------------------------------------ map view
const map = $('map');
let mapSnap = null;
function drawMap(snap) {
  mapSnap = snap;
  if (view !== 'map') return;
  const g = map.getContext('2d');
  const W = map.width = map.clientWidth * devicePixelRatio, H = map.height = map.clientHeight * devicePixelRatio;
  g.fillStyle = '#090d16'; g.fillRect(0, 0, W, H);
  const X = (lon) => (lon / Math.PI + 1) / 2 * W, Y = (lat) => (1 - (lat / (Math.PI / 2) + 1) / 2) * H;
  // base seams as great-circle polylines
  g.strokeStyle = 'rgba(63,224,208,.35)'; g.lineWidth = 1 * devicePixelRatio;
  const seen = new Set();
  for (const [a, b, c] of baseFaces) for (const [u, v] of [[a, b], [b, c], [c, a]]) {
    const k = Math.min(u, v) + ':' + Math.max(u, v); if (seen.has(k)) continue; seen.add(k);
    const P = baseVerts[u], Q = baseVerts[v];
    g.beginPath(); let prev = null;
    for (let s = 0; s <= 32; s++) {
      const t = s / 32; const x = P[0] * (1 - t) + Q[0] * t, y = P[1] * (1 - t) + Q[1] * t, z = P[2] * (1 - t) + Q[2] * t; const m = Math.hypot(x, y, z);
      const [lo, la] = lonlat([x / m, y / m, z / m]); const px = X(lo), py = Y(la);
      if (prev && Math.abs(px - prev) > W / 2) { g.stroke(); g.beginPath(); g.moveTo(px, py); } else if (s === 0) g.moveTo(px, py); else g.lineTo(px, py);
      prev = px;
    }
    g.stroke();
  }
  g.strokeStyle = 'rgba(34,48,77,.9)';
  for (let k = -2; k <= 2; k++) { const y = Y(k * Math.PI / 6); g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke(); }
  for (let k = -5; k <= 6; k++) { const x = X(k * Math.PI / 6); g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); }
  const r = Math.max(1.0, Math.min(2.6 * parseFloat($('sizeSel').value), 2.4 * parseFloat($('sizeSel').value) * Math.sqrt(20000 / Math.max(1, snap.count)))) * devicePixelRatio;
  for (let i = 0; i < snap.count; i++) {
    const [lo, la] = lonlat([snap.pos[3 * i], snap.pos[3 * i + 1], snap.pos[3 * i + 2]]);
    const k = snap.cls[i]; const f = FAMILY[k];
    g.beginPath(); g.arc(X(lo), Y(la), r, 0, Math.PI * 2);
    if (f.anti) { g.lineWidth = 1.4 * devicePixelRatio; g.strokeStyle = f.color; g.stroke(); } else { g.fillStyle = f.color; g.fill(); }
    if (snap.ids[i] === selected) { g.beginPath(); g.arc(X(lo), Y(la), r * 2.6, 0, Math.PI * 2); g.lineWidth = 1.6 * devicePixelRatio; g.strokeStyle = '#fff'; g.stroke(); }
  }
}
map.addEventListener('click', (ev) => {
  if (!mapSnap) return;
  const rect = map.getBoundingClientRect();
  const W = rect.width, H = rect.height;
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  let best = -1, bd = 1e9;
  for (let i = 0; i < mapSnap.count; i++) {
    const [lo, la] = lonlat([mapSnap.pos[3 * i], mapSnap.pos[3 * i + 1], mapSnap.pos[3 * i + 2]]);
    const px = (lo / Math.PI + 1) / 2 * W, py = (1 - (la / (Math.PI / 2) + 1) / 2) * H;
    const d = Math.hypot(px - mx, py - my); if (d < bd) { bd = d; best = i; }
  }
  if (best >= 0 && bd < 14) select(mapSnap.ids[best]);
});

// ------------------------------------------------------------ observer view
const obs = $('obs');
let observerIndex = 0, yaw = 0, obsSnap = null;
function observerPoint() {
  if (observerIndex < 12) return baseVerts[observerIndex];
  // the base vertex nearest to the selected defect, read at its last face
  if (selected >= 0 && cat) { const p = [0, 0, 0]; catPos(selected, 'l', p); let best = 0, bd = -2; baseVerts.forEach((v, i) => { const d = v[0] * p[0] + v[1] * p[1] + v[2] * p[2]; if (d > bd) { bd = d; best = i; } }); return baseVerts[best]; }
  return baseVerts[0];
}
function tangentBasis(o) {
  const up = Math.abs(o[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const e = [up[1] * o[2] - up[2] * o[1], up[2] * o[0] - up[0] * o[2], up[0] * o[1] - up[1] * o[0]];
  const m = Math.hypot(...e); e[0] /= m; e[1] /= m; e[2] /= m;
  const nn = [o[1] * e[2] - o[2] * e[1], o[2] * e[0] - o[0] * e[2], o[0] * e[1] - o[1] * e[0]];
  return [e, nn];
}
async function drawObserver(snap) {
  obsSnap = snap;
  if (view !== 'obs') return;
  const g = obs.getContext('2d');
  const W = obs.width = obs.clientWidth * devicePixelRatio, H = obs.height = obs.clientHeight * devicePixelRatio;
  g.fillStyle = '#090d16'; g.fillRect(0, 0, W, H);
  const o = observerPoint(); const [e, nn] = tangentBasis(o);
  const Rf = Math.sqrt(run.geometry.faces / (4 * Math.PI));      // sphere radius in face widths
  const horizonFaces = Math.min(snap.tick, 24 * run.config.snapshot_every);   // one face per tick
  const camH = 24, f = 0.9 * H;                                   // camera height in faces, focal length
  const horizonY = H * 0.42;
  g.fillStyle = '#0c1424'; g.fillRect(0, horizonY, W, H - horizonY);
  g.strokeStyle = 'rgba(63,224,208,.25)'; g.lineWidth = devicePixelRatio;
  for (let d = 25; d <= horizonFaces; d += 25) { const y = horizonY + f * camH / d; if (y > H) continue; g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke(); }
  g.fillStyle = '#62718f'; g.font = `${12 * devicePixelRatio}px Helvetica, Arial`;
  g.fillText(`observer at base vertex ${observerIndex < 12 ? observerIndex : 'nearest to #' + selected} · horizon ${horizonFaces} faces (= ticks) · depth is record lag · yaw ${yaw}°`, 12 * devicePixelRatio, 18 * devicePixelRatio);
  const cy = Math.cos(yaw * Math.PI / 180), sy = Math.sin(yaw * Math.PI / 180);
  // retarded positions: a defect at distance d is shown where it was d ticks ago
  const items = [];
  const needed = new Map();
  for (let i = 0; i < snap.count; i++) {
    const p = [snap.pos[3 * i], snap.pos[3 * i + 1], snap.pos[3 * i + 2]];
    const th = Math.acos(Math.max(-1, Math.min(1, o[0] * p[0] + o[1] * p[1] + o[2] * p[2])));
    const d = th * Rf;
    if (d > horizonFaces) continue;
    const k = nearestSnapshot(snap.tick - d);
    if (!needed.has(k)) needed.set(k, []);
    needed.get(k).push(i);
  }
  const records = [];
  for (const [k, rows] of needed) {
    const past = await loadSnapshot(k);
    for (const i of rows) {
      const id = snap.ids[i];
      const j = past.byId.get(id);
      if (j === undefined) continue;
      const p = [past.pos[3 * j], past.pos[3 * j + 1], past.pos[3 * j + 2]];
      const th = Math.acos(Math.max(-1, Math.min(1, o[0] * p[0] + o[1] * p[1] + o[2] * p[2])));
      const d = Math.max(0.5, th * Rf);
      const ex = p[0] * e[0] + p[1] * e[1] + p[2] * e[2], ny = p[0] * nn[0] + p[1] * nn[1] + p[2] * nn[2];
      const az = Math.atan2(ex, ny);
      const x = d * Math.sin(az), z = d * Math.cos(az);
      const xr = x * cy - z * sy, zr = x * sy + z * cy;        // yaw
      if (d <= 2.5) records.push({ id, k: snap.cls[i], d });
      if (zr <= 0.6) continue;
      items.push({ id, k: snap.cls[i], u: W / 2 + f * xr / zr, v: horizonY + f * camH / zr, zr });
    }
  }
  items.sort((a, b) => b.zr - a.zr);
  for (const it of items) {
    const fam = FAMILY[it.k]; const r = Math.max(1.6, Math.min(14, 120 / it.zr)) * devicePixelRatio;
    if (it.u < -20 || it.u > W + 20) continue;
    g.beginPath(); g.arc(it.u, it.v, r, 0, Math.PI * 2);
    if (fam.anti) { g.lineWidth = Math.max(1, r * 0.35); g.strokeStyle = fam.color; g.stroke(); } else { g.fillStyle = fam.color; g.fill(); }
    if (it.id === selected) { g.beginPath(); g.arc(it.u, it.v, r * 2.2 + 4, 0, Math.PI * 2); g.lineWidth = 1.6 * devicePixelRatio; g.strokeStyle = '#fff'; g.stroke(); }
  }
  obs._items = items;
  const rec = $('records');
  rec.innerHTML = records.length
    ? records.sort((a, b) => a.d - b.d).slice(0, 12).map((r) => `<div class="row"><span>tick ${snap.tick} · <a href="#" data-id="${r.id}" style="color:#6fb1ff">#${r.id}</a> · class ${r.k}</span><b>${r.d < 1.5 ? 'on an adjacent face' : 'two faces away'}</b></div>`).join('')
    : '<span style="color:var(--faint)">no defect on the faces next to this observer at this tick</span>';
  rec.querySelectorAll('a[data-id]').forEach((a) => a.onclick = (ev) => { ev.preventDefault(); select(parseInt(a.dataset.id)); });
}
obs.addEventListener('click', (ev) => {
  if (!obs._items) return;
  const rect = obs.getBoundingClientRect();
  const mx = (ev.clientX - rect.left) * devicePixelRatio, my = (ev.clientY - rect.top) * devicePixelRatio;
  let best = null, bd = 1e9;
  for (const it of obs._items) { const d = Math.hypot(it.u - mx, it.v - my); if (d < bd) { bd = d; best = it; } }
  if (best && bd < 16 * devicePixelRatio) select(best.id);
});

// ------------------------------------------------------------ side panel
function fmtFace(f) {
  if (run.geometry.coords) {
    const intra = run.geometry.intra_faces;
    return f < intra ? `inside carrier ${Math.floor(f / 20)}, template face ${f % 20}` : `screen vertex ${f - intra} (between carriers)`;
  }
  const d = decodeFace(f); return `patch ${d.b}, cell (${d.i}, ${d.j}) ${d.up ? 'up' : 'down'}`;
}
let facePosCache = null;
function catPos(i, which, out) {
  // position of a catalogue entry's birth ('b') or last ('l') face
  if (!run.geometry.coords) return facePos(which === 'b' ? cat.bface[i] : cat.lface[i], out);
  const arr = which === 'b' ? cat.bxyz : cat.lxyz;
  if (arr) { out[0] = arr[3 * i]; out[1] = arr[3 * i + 1]; out[2] = arr[3 * i + 2]; return out; }
  return facePosAny(which === 'b' ? cat.bface[i] : cat.lface[i], out);
}
function facePosAny(f, out) {
  if (!run.geometry.coords) return facePos(f, out);
  // positions of catalogue faces come from the snapshots seen so far
  const p = facePosCache && facePosCache.get(f);
  if (p) { out[0] = p[0]; out[1] = p[1]; out[2] = p[2]; return out; }
  out[0] = 0; out[1] = 0; out[2] = 1; return out;
}
function row(k, v) { return `<div class="row"><span>${k}</span><b>${v}</b></div>`; }
function idLink(i) { return i < 0 ? '–' : `<a href="#" data-id="${i}" style="color:#6fb1ff">#${i}</a>`; }
function select(id) {
  selected = id; follow = false; $('followBtn').classList.remove('on');
  renderCard();
  loadSnapshot(tickIndex).then((s) => { updateHighlight(s); drawMap(s); drawObserver(s); });
  drawTrail(id);
}
function renderCard() {
  const el = $('selected');
  if (selected < 0 || !cat) { el.innerHTML = '<span style="color:var(--faint)">click a defect, or pick one from the list below</span>'; return; }
  const i = selected; const k = cat.cls[i]; const f = FAMILY[k];
  const alive = cat.death[i] < 0;
  const p0 = [0, 0, 0], p1 = [0, 0, 0]; catPos(i, 'b', p0); catPos(i, 'l', p1);
  const ang = Math.acos(Math.max(-1, Math.min(1, p0[0] * p1[0] + p0[1] * p1[1] + p0[2] * p1[2])));
  const circumference = 2 * Math.PI;
  const displacementFaces = ang / (circumference / (Math.sqrt(run.geometry.faces / (4 * Math.PI)) * 2 * Math.PI)) ;
  el.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
      <span class="${f.anti ? 'ring' : 'dot'}" style="background:${f.anti ? 'transparent' : f.color};border-color:${f.color}"></span>
      <div><div style="font-weight:700;font-size:16px">Defect #${i}</div><div style="color:var(--muted);font-size:12.5px">family ${f.name} · class ${k} · ${f.label}</div></div>
    </div>
    ${run.geometry.coords ? row('tier', cat.bface[i] < run.geometry.intra_faces ? 'internal twist of one carrier' : 'screen twist between carriers') : ''}
    ${row('status', alive ? '<span style="color:var(--green)">alive at the end</span>' : CAUSE[cat.cause[i]] + ' at tick ' + cat.death[i])}
    ${row('born', 'tick ' + cat.birth[i] + (cat.p1[i] >= 0 ? ' from ' + idLink(cat.p1[i]) + (cat.p2[i] >= 0 ? ' + ' + idLink(cat.p2[i]) : '') : ' as a conjugate pair'))}
    ${row('lifetime', cat.lifetime[i] + ' ticks')}
    ${row('hops', cat.hops[i] + ' (' + (cat.hops[i] / Math.max(1, cat.lifetime[i])).toFixed(3) + ' per tick)')}
    ${row('net displacement', (ang * 180 / Math.PI).toFixed(2) + '° of arc')}
    ${row('birth face', fmtFace(cat.bface[i]))}
    ${row('last face', fmtFace(cat.lface[i]))}
    <div class="row" style="display:block"><span>best-justified reading</span><div style="color:var(--text);font-weight:600;margin-top:2px">${GUESS[k]}</div><div style="color:var(--faint);font-size:12px;margin-top:3px">${GUESS_NOTE}</div></div>
    ${row('noticed by', visibility(k).sees.join(', ') || 'no block')}
    ${row('invisible to', visibility(k).blind.join(', ') || 'no block')}
    ${row('partner', idLink(cat.partner[i]))}
    ${row('children', (cat.c1[i] >= 0 ? idLink(cat.c1[i]) : '–') + (cat.c2[i] >= 0 ? ', ' + idLink(cat.c2[i]) : ''))}
    <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">
      <button data-jump="${cat.birth[i]}">jump to birth</button>
      ${alive ? '' : `<button data-jump="${Math.max(0, cat.death[i] - 1)}">jump to end</button>`}
      <button data-jump="${Math.floor((cat.birth[i] + (alive ? run.config.ticks : cat.death[i])) / 2)}">midlife</button>
    </div>`;
  el.querySelectorAll('a[data-id]').forEach((a) => a.onclick = (e) => { e.preventDefault(); select(parseInt(a.dataset.id)); });
  el.querySelectorAll('button[data-jump]').forEach((b) => b.onclick = () => setTick(nearestSnapshot(parseInt(b.dataset.jump))));
}
function updateNow(snap) {
  const st = run.stats;
  const sweep = run.geometry.coords ? Math.floor(snap.tick / run.config.chunks_per_sweep) : snap.tick;
  const c = run.geometry.coords ? st.counts_intra[sweep].map((v, i) => v + st.counts_macro[sweep][i]) : st.counts[snap.tick];
  const fam = { A: c[1] + c[5], B: c[2] + c[4], C: c[3] };
  const T = run.geometry.coords ? st.T_mean[sweep] : st.temperature[snap.tick];
  const at = run.geometry.coords ? sweep : snap.tick;
  const phase = run.geometry.coords ? (st.descents[sweep] > 0 ? `<span style="color:var(--amber)">settling: ${st.descents[sweep].toLocaleString()} balancing moves this sweep</span>` : '<span style="color:var(--teal)">balanced class: only tie swaps move units</span>') : '';
  $('now').innerHTML = (run.geometry.coords ? row('sweep', sweep + ' of ' + run.config.sweeps) + row('phase', phase) + row('descents this sweep', st.descents[sweep].toLocaleString()) + row('tie swaps this sweep', st.swaps[sweep].toLocaleString()) : row('tick', snap.tick + ' of ' + run.config.ticks))
    + row('register temperature', T.toFixed(3))
    + row('defects shown', snap.count.toLocaleString() + (run.geometry.coords && !showIntra ? ' (screen tier)' : '')) + row('family A / B / C', `${fam.A.toLocaleString()} / ${fam.B.toLocaleString()} / ${fam.C.toLocaleString()}`)
    + (run.geometry.coords ? row('screen / internal', `${st.counts_macro[sweep].slice(1).reduce((a, b) => a + b, 0).toLocaleString()} / ${st.counts_intra[sweep].slice(1).reduce((a, b) => a + b, 0).toLocaleString()}`) : '')
    + row('births', st.births[at]) + row('moves', st.moves[at])
    + row('annihilations', st.annihilations[at]) + row('fusions / splits', st.fusions[at] + ' / ' + st.splits[at]);
  $('ticklabel').textContent = 'tick ' + snap.tick;
}
function renderLegend() {
  const items = [[1, 'Family A: class 1 (filled) and class 5 (ring), conjugates · reading: quark doublet Q_L and its antiparticle'],
    [2, 'Family B: class 2 and class 4, conjugates · reading: right-handed quark conjugates and quarks'],
    [3, 'Family C: class 3, its own conjugate · reading: lepton doublet L, anti-L or Higgs']];
  $('legend').innerHTML = items.map(([k, t]) => `<div><span class="dot" style="background:${FAMILY[k].color};flex:none"></span>${t}</div>`).join('')
    + `<div style="color:var(--faint);font-size:12px;margin-top:6px">A defect is a face whose seam elements sum to a nonzero class mod 6. Rings are antiparticles: the pair that cancels. The readings take the class as the hypercharge label 6Y mod 6 and use the Standard Model congruence 2t + 3d + q = 0 mod 6; they are a hypothesis, not an identification.</div>`;
}
function renderList(mode) {
  if (!cat) return;
  const N = cat.count, idx = [];
  for (let i = 0; i < N; i++) idx.push(i);
  let title = 'longest lived';
  if (/^\d+$/.test(mode)) { const i = parseInt(mode); if (i < N) select(i); return; }
  if (mode.startsWith('hop')) { idx.sort((a, b) => cat.hops[b] - cat.hops[a]); title = 'most hops'; }
  else if (mode.startsWith('fus')) { const f = idx.filter((i) => cat.p2[i] >= 0); f.sort((a, b) => cat.lifetime[b] - cat.lifetime[a]); idx.length = 0; idx.push(...f); title = 'born by fusion or exchange'; }
  else if (mode.startsWith('alive')) { const f = idx.filter((i) => cat.death[i] < 0); f.sort((a, b) => cat.lifetime[b] - cat.lifetime[a]); idx.length = 0; idx.push(...f); title = 'alive at the end'; }
  else idx.sort((a, b) => cat.lifetime[b] - cat.lifetime[a]);
  const rows = idx.slice(0, 40).map((i) => `<tr data-id="${i}"><td><span class="${FAMILY[cat.cls[i]].anti ? 'ring' : 'dot'}" style="background:${FAMILY[cat.cls[i]].anti ? 'transparent' : FAMILY[cat.cls[i]].color};border-color:${FAMILY[cat.cls[i]].color};width:9px;height:9px"></span> #${i}</td><td>${FAMILY[cat.cls[i]].name}${cat.cls[i]}</td><td class="c">${cat.birth[i]}</td><td class="c">${cat.lifetime[i]}</td><td class="c">${cat.hops[i]}</td></tr>`).join('');
  $('list').innerHTML = `<div style="color:var(--faint);font-size:12px;margin:6px 0 2px">${title} · ${idx.length.toLocaleString()} of ${N.toLocaleString()} identities</div><table><tr><th>id</th><th>class</th><th>born</th><th>life</th><th>hops</th></tr>${rows}</table>`;
  $('list').querySelectorAll('tr[data-id]').forEach((tr) => tr.onclick = () => select(parseInt(tr.dataset.id)));
}
function renderSummary() {
  const N = cat.count, fam = { A: [], B: [], C: [] };
  for (let i = 0; i < N; i++) fam[FAMILY[cat.cls[i]].name].push(i);
  const med = (arr, f) => { if (!arr.length) return 0; const v = arr.map(f).sort((a, b) => a - b); return v[Math.floor(v.length / 2)]; };
  let html = row('identities', N.toLocaleString()) + row('alive at the end', Array.from({ length: N }, (_, i) => i).filter((i) => cat.death[i] < 0).length.toLocaleString());
  for (const f of ['A', 'B', 'C']) {
    const ids = fam[f]; if (!ids.length) { html += row('family ' + f, 'none'); continue; }
    const alive = ids.filter((i) => cat.death[i] < 0).length;
    html += row('family ' + f, `${ids.length.toLocaleString()} ever · ${alive.toLocaleString()} alive · median life ${med(ids, (i) => cat.lifetime[i])} · ${med(ids, (i) => cat.hops[i] / Math.max(1, cat.lifetime[i])).toFixed(3)} hops/tick`);
  }
  const ann = Array.from({ length: N }, (_, i) => i).filter((i) => cat.cause[i] === 1);
  const withSib = ann.filter((i) => cat.p1[i] < 0 && cat.partner[i] >= 0 && cat.cause[cat.partner[i]] === 1 && cat.partner[cat.partner[i]] === i).length;
  html += row('annihilations', ann.length.toLocaleString() + (ann.length ? ` (${(100 * withSib / ann.length).toFixed(0)}% with the birth partner)` : ''));
  $('summary').innerHTML = html;
}
function renderAbout() {
  const g = run.geometry, c = run.config;
  if (g.coords) {
    $('about').innerHTML = `Exact federation at level ${g.level}: ${g.carriers.toLocaleString()} twelve-port carriers glued by port pairs on the geodesic tower, ${g.ports.toLocaleString()} ports, ${g.seams.toLocaleString()} seams (${g.inter_seams.toLocaleString()} between carriers), ${g.screen_vertices.toLocaleString()} screen vertices.<br><br>
    ${c.law}. Cost per class 0..5: ${c.cost.join(', ')}. ${c.temperature} (T0 = ${c.T0}). Seed ${c.seed}; ${c.schedule}; ${c.chunks_per_sweep} record chunks per sweep.<br><br>${c.status}.<br><br>${run.identities.toLocaleString()} identities over ${c.sweeps} sweeps; ${snaps.length} snapshots; run time ${run.runtime_seconds}s.`;
    return;
  }
  $('about').innerHTML = `Level ${g.level} refinement of the certified twelve-port carrier: ${g.observers.toLocaleString()} observers, ${g.seams.toLocaleString()} seams, ${g.faces.toLocaleString()} faces.<br><br>
  Law: ${c.law}. Cost per class 0..5: ${c.cost.join(', ')} (${c.cost_name}). Proposal fraction ${c.proposal_fraction} per tick. Temperature knots ${c.temperature_knots.map((k) => k[0] + ':' + k[1]).join(', ')}. Seed ${c.seed}.<br><br>
  ${c.status}.<br><br>${run.identities.toLocaleString()} identities over ${c.ticks} ticks; ${snaps.length} snapshots; run time ${run.runtime_seconds}s.`;
}

// ------------------------------------------------------------ timeline
const tl = $('tl');
function drawTimeline() {
  const g = tl.getContext('2d');
  const W = tl.width = tl.clientWidth * devicePixelRatio, H = tl.height = tl.clientHeight * devicePixelRatio;
  g.fillStyle = '#0b1120'; g.fillRect(0, 0, W, H);
  const fedRun = !!run.geometry.coords;
  const T = fedRun ? run.config.sweeps : run.config.ticks;
  const cnt = fedRun ? run.stats.counts_intra.map((v, i) => v.map((x, j) => x + run.stats.counts_macro[i][j])) : run.stats.counts;
  const temps = fedRun ? run.stats.T_mean : run.stats.temperature;
  let max = 1; for (let t = 0; t < T; t++) max = Math.max(max, cnt[t][1] + cnt[t][5]);
  const X = (t) => t / (T - 1) * (W - 20) + 10, Y = (v) => H - 14 - v / max * (H - 34);
  // temperature
  g.strokeStyle = 'rgba(147,163,192,.5)'; g.lineWidth = devicePixelRatio; g.beginPath();
  let tmax = Math.max(...temps, 1e-9);
  for (let t = 0; t < T; t++) { const y = H - 14 - temps[t] / tmax * (H - 34); t ? g.lineTo(X(t), y) : g.moveTo(X(t), y); }
  g.stroke();
  for (const [sel, col] of [[(c) => c[1] + c[5], FAMILY[1].color], [(c) => (c[2] + c[4]) * 10, FAMILY[2].color], [(c) => c[3] * 10, FAMILY[3].color]]) {
    g.strokeStyle = col; g.lineWidth = 1.6 * devicePixelRatio; g.beginPath();
    for (let t = 0; t < T; t++) { const y = Y(sel(cnt[t])); t ? g.lineTo(X(t), y) : g.moveTo(X(t), y); }
    g.stroke();
  }
  g.fillStyle = '#93a3c0'; g.font = `${11 * devicePixelRatio}px Helvetica, Arial`;
  g.fillText(`family A count (max ${max.toLocaleString()})  ·  B and C ×10  ·  grey: temperature`, 12, 12 * devicePixelRatio);
  const x = X(fedRun ? Math.floor(snaps[tickIndex].tick / run.config.chunks_per_sweep) : snaps[tickIndex].tick);
  g.strokeStyle = '#ffffff'; g.lineWidth = devicePixelRatio; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke();
}
tl.addEventListener('click', (ev) => {
  const rect = tl.getBoundingClientRect();
  const total = run.geometry.coords ? run.config.sweeps : run.config.ticks;
  let t = Math.round((ev.clientX - rect.left - 10 / devicePixelRatio) / (rect.width - 20 / devicePixelRatio) * (total - 1));
  if (run.geometry.coords) t *= run.config.chunks_per_sweep;
  setTick(nearestSnapshot(Math.max(0, Math.min(run.config.ticks - 1, t))));
});
function nearestSnapshot(tick) { let best = 0; for (let k = 0; k < snaps.length; k++) if (Math.abs(snaps[k].tick - tick) < Math.abs(snaps[best].tick - tick)) best = k; return best; }
async function setTick(k) {
  tickIndex = Math.max(0, Math.min(snaps.length - 1, k));
  $('tick').value = tickIndex;
  const snap = await loadSnapshot(tickIndex);
  showSnapshot(snap); drawTimeline();
  if (tickIndex + 1 < snaps.length) loadSnapshot(tickIndex + 1);
}

// ------------------------------------------------------------ controls
$('tick').addEventListener('input', (e) => setTick(parseInt(e.target.value)));
$('play').onclick = () => { playing = !playing; $('play').textContent = playing ? '❚❚ pause' : '▶ play'; if (playing) step(); };
async function step() {
  if (!playing) return;
  const sp = parseInt($('speed').value);
  if (tickIndex + sp >= snaps.length) { playing = false; $('play').textContent = '▶ play'; return; }
  await setTick(tickIndex + sp);
  setTimeout(step, 120);
}
function setView(v) {
  view = v;
  for (const [id, name] of [['view3d', '3d'], ['viewmap', 'map'], ['viewobs', 'obs'], ['viewbulk', 'bulk']]) $(id).classList.toggle('on', name === v);
  $('gl').style.display = (v === '3d' || v === 'bulk') ? '' : 'none';
  setBulkVisible(v === 'bulk');
  if (v === 'bulk') { showBulk(tickIndex); controls.minDistance = 0.3; } else controls.minDistance = 1.08;
  map.style.display = v === 'map' ? 'block' : 'none';
  obs.style.display = v === 'obs' ? 'block' : 'none';
  for (const id of ['obsSel', 'yaw']) $(id).style.display = v === 'obs' ? '' : 'none';
  for (const id of ['recordsK', 'records']) $(id).style.display = v === 'obs' ? '' : 'none';
  $('hint').textContent = v === '3d' ? 'drag to orbit · wheel to zoom · click a defect for its card · space plays · arrows step'
    : v === 'map' ? 'equal-angle map of the carrier · teal arcs are the twenty base patches · click a defect'
    : v === 'bulk' ? 'the consensus bulk: all observers\' records glued · surface = now, deeper = older records (the third direction is record lag) · zoom in to enter'
    : 'what one observer can have received: each defect drawn where it was when its signal left, one face per tick · nearer is lower · click a defect';
  if (v === 'map' && mapSnap) drawMap(mapSnap);
  if (v === 'obs' && obsSnap) drawObserver(obsSnap);
}
$('view3d').onclick = () => setView('3d');
$('viewmap').onclick = () => setView('map');
$('viewobs').onclick = () => setView('obs');
$('viewbulk').onclick = () => setView('bulk');
$('obsSel').onchange = (e) => { observerIndex = parseInt(e.target.value); if (obsSnap) drawObserver(obsSnap); };
$('yaw').oninput = (e) => { yaw = parseInt(e.target.value); if (obsSnap) drawObserver(obsSnap); };
$('followBtn').onclick = () => { follow = !follow; $('followBtn').classList.toggle('on', follow); };
$('search').addEventListener('change', (e) => renderList(e.target.value.trim().toLowerCase()));
$('sizeSel').onchange = () => loadSnapshot(tickIndex).then(showSnapshot);
$('intraChk').onchange = (e) => { showIntra = e.target.checked; loadSnapshot(tickIndex).then(showSnapshot); };
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') { e.preventDefault(); $('play').click(); }
  if (e.code === 'ArrowRight') setTick(tickIndex + 1);
  if (e.code === 'ArrowLeft') setTick(tickIndex - 1);
});
window.addEventListener('resize', resize);
function resize() {
  const r = $('stage').getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height; camera.updateProjectionMatrix();
  if (run) { drawTimeline(); if (mapSnap) drawMap(mapSnap); }
}
function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }

// ------------------------------------------------------------ boot
(async function boot() {
  run = await (await fetch('/run/run.json')).json();
  level = run.geometry.level; n = 1 << level;
  const PHI = (1 + Math.sqrt(5)) / 2;
  const ICO = [[-1, PHI, 0], [1, PHI, 0], [-1, -PHI, 0], [1, -PHI, 0], [0, -1, PHI], [0, 1, PHI], [0, -1, -PHI], [0, 1, -PHI], [PHI, 0, -1], [PHI, 0, 1], [-PHI, 0, -1], [-PHI, 0, 1]].map((v) => { const m = Math.hypot(...v); return v.map((x) => x / m); });
  const ICO_FACES = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  baseFaces = run.geometry.base_faces || ICO_FACES; baseVerts = run.geometry.base_vertices || ICO;
  snaps = run.snapshots;
  $('tick').max = snaps.length - 1;
  $('runmeta').textContent = run.geometry.coords
    ? `${run.geometry.observers.toLocaleString()} carriers · ${run.geometry.ports.toLocaleString()} ports · ${run.geometry.seams.toLocaleString()} seams · ${run.config.sweeps} sweeps · law C`
    : `${run.geometry.observers.toLocaleString()} observers · ${run.geometry.faces.toLocaleString()} faces · ${run.config.ticks} ticks · seed ${run.config.seed}`;
  if (run.geometry.coords) $('intraLbl').style.display = '';
  buildSphere(Math.min(6, level));
  buildPoints();
  resize(); animate();
  renderLegend(); renderAbout();
  const sel = $('obsSel');
  for (let i = 0; i < 12; i++) { const o = document.createElement('option'); o.value = i; o.textContent = 'observer at base vertex ' + i; sel.appendChild(o); }
  const near = document.createElement('option'); near.value = 12; near.textContent = 'observer nearest the selected defect'; sel.appendChild(near);
  const q = new URLSearchParams(location.search);
  if (q.has('observer')) { observerIndex = parseInt(q.get('observer')); sel.value = observerIndex; }
  if (q.has('yaw')) { yaw = parseInt(q.get('yaw')); $('yaw').value = yaw; }
  if (['map', 'obs', 'bulk'].includes(q.get('view'))) setView(q.get('view'));
  await setTick(q.has('tick') ? nearestSnapshot(parseInt(q.get('tick'))) : 0);
  $('loading').remove();
  cat = await loadCatalogue();
  renderList('longest');
  renderSummary();
  if (q.has('id')) select(parseInt(q.get('id')));
})();
