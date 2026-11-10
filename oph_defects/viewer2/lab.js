// Detector lab: one load quantum on a federation under the native record law and two coherent continuations.
// The mathematics is in oph_defects/observer_lab.py (same operators, checked there against exact references).
import { loadNpz } from './npz.js';
import { $ } from './app.js';

const LMAX = 12;                       // upper bound of the port Laplacian spectrum (twice the largest degree)
const DT = 0.25;                       // step of the wave walk in units T, where the frequency is sqrt(lambda)
const KMAX = 2.5, KN = 33;             // momentum analyser grid, wave vectors per mesh edge
const FRONT = 0.373;                   // influence front of the native repair, mesh edges per sweep (measured, levels 4 and 5)
const LAWS = [
  { id: 'native', title: 'Native record law', color: [169, 149, 255], note: 'what the exact runs execute: every attempt next to the quantum tests one end of its seam' },
  { id: 'transfer', title: 'Transfer continuation', color: [255, 180, 74], note: 'declared: the unitary group with the generator of the native transfer, E = λ / 2' },
  { id: 'wave', title: 'Admitted wave', color: [63, 224, 208], note: 'declared: positive-frequency part of the coherent walk of the paper, ω = √λ' },
];
const G = { ready: false };
const W = {};
const ST = { running: false, s: 0, prepared: false, log: [] };

// ------------------------------------------------------------------ operators
function applyL(x, y) {
  const a = G.a, b = G.b, d = G.deg, P = G.P, M = G.M;
  for (let p = 0; p < P; p++) y[p] = d[p] * x[p];
  for (let s = 0; s < M; s++) { const i = a[s], j = b[s]; y[i] -= x[j]; y[j] -= x[i]; }
}
function norm2(re, im) { let t = 0; for (let p = 0; p < re.length; p++) t += re[p] * re[p] + (im ? im[p] * im[p] : 0); return t; }

function besselJ(a, K) {               // J_0(a) .. J_K(a) by downward recurrence
  const out = new Float64Array(K + 1);
  const top = Math.max(K, a);
  const m = 2 * Math.ceil((top + 30 + Math.sqrt(40 * (top + 1))) / 2);
  let jp = 0, j = 1e-30, sum = 0;
  for (let k = m; k > 0; k--) {
    const jm = (2 * k / a) * j - jp;
    jp = j; j = jm;
    if (k - 1 <= K) out[k - 1] = j;
    if ((k - 1) % 2 === 0) sum += (k - 1 === 0 ? 1 : 2) * j;
    if (Math.abs(j) > 1e250) { j *= 1e-250; jp *= 1e-250; sum *= 1e-250; for (let i = k - 1; i <= K; i++) out[i] *= 1e-250; }
  }
  for (let k = 0; k <= K; k++) out[k] /= sum;
  return out;
}

function transferAdvance(st, ds) {     // psi <- exp(-i ds L / 2) psi by a Chebyshev expansion
  const P = G.P, a = 3 * ds;           // -i ds L / 2 = -i a (X + 1), X = L / 6 - 1
  const K = Math.ceil(a + 10 * Math.cbrt(a) + 12);
  const J = besselJ(a, K);
  const { t0r, t0i, t1r, t1i, ar, ai, r0, r1 } = W;
  t0r.set(st.re); t0i.set(st.im);
  applyL(t0r, r0); applyL(t0i, r1);
  for (let p = 0; p < P; p++) { t1r[p] = r0[p] / 6 - t0r[p]; t1i[p] = r1[p] / 6 - t0i[p]; }
  // coefficients (2 - delta_k0) (-i)^k J_k
  for (let p = 0; p < P; p++) { ar[p] = J[0] * t0r[p] + 2 * J[1] * t1i[p]; ai[p] = J[0] * t0i[p] - 2 * J[1] * t1r[p]; }
  let ar_ = t0r, ai_ = t0i, br = t1r, bi = t1i;
  for (let k = 2; k <= K; k++) {
    applyL(br, r0); applyL(bi, r1);
    const c = 2 * J[k], ph = k & 3;
    for (let p = 0; p < P; p++) {
      const nr = r0[p] / 3 - 2 * br[p] - ar_[p], ni = r1[p] / 3 - 2 * bi[p] - ai_[p];
      ar_[p] = nr; ai_[p] = ni;
      if (ph === 0) { ar[p] += c * nr; ai[p] += c * ni; }
      else if (ph === 1) { ar[p] += c * ni; ai[p] -= c * nr; }
      else if (ph === 2) { ar[p] -= c * nr; ai[p] -= c * ni; }
      else { ar[p] -= c * ni; ai[p] += c * nr; }
    }
    const tr = ar_, ti = ai_; ar_ = br; ai_ = bi; br = tr; bi = ti;
  }
  const cs = Math.cos(a), sn = Math.sin(a);
  for (let p = 0; p < P; p++) { st.re[p] = ar[p] * cs + ai[p] * sn; st.im[p] = ai[p] * cs - ar[p] * sn; }
}

function jacobiEigen(A, n) {           // symmetric n x n (row-major Float64Array): eigenvalues and eigenvectors (columns of V)
  const V = new Float64Array(n * n); for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i * n + j] * A[i * n + j];
    if (off < 1e-26) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      const apq = A[p * n + q]; if (Math.abs(apq) < 1e-300) continue;
      const th = (A[q * n + q] - A[p * n + p]) / (2 * apq);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = A[k * n + p], akq = A[k * n + q]; A[k * n + p] = c * akp - s * akq; A[k * n + q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = A[p * n + k], aqk = A[q * n + k]; A[p * n + k] = c * apk - s * aqk; A[q * n + k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k * n + p], vkq = V[k * n + q]; V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq; }
    }
  }
  const vals = new Float64Array(n); for (let i = 0; i < n; i++) vals[i] = A[i * n + i];
  return { vals, V };
}

function lanczosApply(v, outRe, outIm, f) {   // out = f(L) v for a real vector v and a complex function f of the eigenvalue
  const P = G.P, m = G.krylov, Q = W.Q, q = W.q, w = W.r0;
  const alpha = new Float64Array(m), beta = new Float64Array(m);
  const b0 = Math.sqrt(norm2(v));
  outRe.fill(0); outIm.fill(0);
  if (b0 < 1e-300) return;
  for (let p = 0; p < P; p++) q[p] = v[p] / b0;
  let k = m;
  for (let j = 0; j < m; j++) {
    Q.set(q, j * P);
    applyL(q, w);
    let al = 0; for (let p = 0; p < P; p++) al += q[p] * w[p];
    alpha[j] = al;
    const o1 = (j - 1) * P, bj = j ? beta[j - 1] : 0;
    for (let p = 0; p < P; p++) w[p] -= al * q[p] + (j ? bj * Q[o1 + p] : 0);
    for (let i = 0; i <= j; i++) { const o = i * P; let d = 0; for (let p = 0; p < P; p++) d += Q[o + p] * w[p]; for (let p = 0; p < P; p++) w[p] -= d * Q[o + p]; }
    const bt = Math.sqrt(norm2(w)); beta[j] = bt;
    if (bt < 1e-12) { k = j + 1; break; }
    for (let p = 0; p < P; p++) q[p] = w[p] / bt;
  }
  const A = new Float64Array(k * k);
  for (let i = 0; i < k; i++) { A[i * k + i] = alpha[i]; if (i + 1 < k) { A[i * k + i + 1] = beta[i]; A[(i + 1) * k + i] = beta[i]; } }
  const { vals, V } = jacobiEigen(A, k);
  for (let j = 0; j < k; j++) {
    let cr = 0, ci = 0;
    for (let i = 0; i < k; i++) { const [fr, fi] = f(Math.max(vals[i], 0)); const u = V[j * k + i] * V[i]; cr += u * fr; ci += u * fi; }   // V[0*k+i] = V[i]
    const o = j * P; cr *= b0; ci *= b0;
    for (let p = 0; p < P; p++) { outRe[p] += cr * Q[o + p]; outIm[p] += ci * Q[o + p]; }
  }
}

function waveStart(st) {               // previous = psi, current = exp(-i arccos(1 - h L / 2)) psi: one exact step of the walk
  const P = G.P, h = DT * DT;
  const f = (x) => { const th = Math.acos(Math.max(-1, Math.min(1, 1 - h * x / 2))); return [Math.cos(th), -Math.sin(th)]; };
  st.pre.set(st.re); st.pim.set(st.im);
  lanczosApply(st.pre, W.ar, W.ai, f);          // f(L) re
  lanczosApply(st.pim, W.t0r, W.t0i, f);        // f(L) im
  for (let p = 0; p < P; p++) { st.re[p] = W.ar[p] - W.t0i[p]; st.im[p] = W.ai[p] + W.t0r[p]; }
}
function eventDriven() { const v = $('lab-clock').value; return v === 'poisson' || v === 'permuted'; }
function waveInit(st) {                // after a preparation or a read: continue psi as a positive-frequency wave
  if (!eventDriven()) { waveStart(st); return; }
  // seam field E = i U^T A^(-1/2) psi with U = B^T / sqrt 2, A = L / 2
  const P = G.P, M = G.M, a = G.a, b = G.b, f = (x) => [x > 1e-9 ? Math.sqrt(2 / x) : 0, 0], r2 = Math.SQRT1_2;
  let mr = 0, mi = 0; for (let p = 0; p < P; p++) { mr += st.re[p]; mi += st.im[p]; } mr /= P; mi /= P;
  for (let p = 0; p < P; p++) { W.pr[p] = st.re[p] - mr; W.pi[p] = st.im[p] - mi; }
  lanczosApply(W.pr, W.ar, W.ai, f);            // g_re in W.ar
  lanczosApply(W.pi, W.t0r, W.t0i, f);          // g_im in W.t0r
  for (let s = 0; s < M; s++) { st.er[s] = -(W.t0r[a[s]] - W.t0r[b[s]]) * r2; st.ei[s] = (W.ar[a[s]] - W.ar[b[s]]) * r2; }
}
function waveEvents(st, sweeps, phi, poisson) {
  const M = G.M, a = G.a, b = G.b, re = st.re, im = st.im, er = st.er, ei = st.ei, c = Math.cos(phi), sn = Math.sin(phi), r2 = Math.SQRT1_2, order = st.order;
  let n = Math.round(M * sweeps);
  while (n > 0) {
    const m = Math.min(n, M);
    if (!poisson) for (let i = M - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0, t = order[i]; order[i] = order[j]; order[j] = t; }
    for (let t = 0; t < m; t++) {
      const e = poisson ? (Math.random() * M) | 0 : order[t], i = a[e], j = b[e];
      let beta = (re[i] - re[j]) * r2, E = er[e], d = ((-sn * E + c * beta) - beta) * r2;
      er[e] = c * E + sn * beta; re[i] += d; re[j] -= d;
      beta = (im[i] - im[j]) * r2; E = ei[e]; d = ((-sn * E + c * beta) - beta) * r2;
      ei[e] = c * E + sn * beta; im[i] += d; im[j] -= d;
    }
    n -= m;
  }
}
function waveStep(st) {
  const P = G.P, h = DT * DT, r0 = W.r0, r1 = W.r1;
  applyL(st.re, r0); applyL(st.im, r1);
  for (let p = 0; p < P; p++) {
    const nr = 2 * st.re[p] - st.pre[p] - h * r0[p], ni = 2 * st.im[p] - st.pim[p] - h * r1[p];
    st.pre[p] = st.re[p]; st.pim[p] = st.im[p]; st.re[p] = nr; st.im[p] = ni;
  }
  st.T += DT;
}

function nativeAdvance(st, ds) {
  const P = G.P, d = G.deg, r0 = W.r0;
  const sub = Math.max(1, Math.ceil(ds / 0.25)), h = ds / sub;
  for (let k = 0; k < sub; k++) {
    applyL(st.p, r0);
    for (let p = 0; p < P; p++) st.p[p] -= 0.5 * h * r0[p];
    if (!st.clicked) {                 // the observer reads its own seams: a hop into it is a click
      let got = 0; for (const q of G.obsPorts) got += st.p[q];
      if (st.click + got >= st.u) {
        let pick = Math.random() * got, at = G.obsPorts[0];
        for (const q of G.obsPorts) { pick -= st.p[q]; if (pick <= 0) { at = q; break; } }
        st.p.fill(0); st.p[at] = 1; st.cr.fill(0); st.ci.fill(0);
        st.clicked = true; st.clickSweep = ST.s + (k + 1) * h; st.click = 1;
        say(`native law: the observer recorded the quantum on one of its own seams at sweep ${st.clickSweep.toFixed(1)}`);
      } else { st.click += got; for (const q of G.obsPorts) st.p[q] = 0; }
    }
  }
  for (let p = 0; p < P; p++) { const f = Math.exp(-0.5 * ds * d[p]); st.cr[p] *= f; st.ci[p] *= f; }
}

// ------------------------------------------------------------------ readouts
function cellLaw(st, id, out) {        // probability per carrier
  const N = G.N; let tot = 0;
  for (let c = 0; c < N; c++) {
    let t = 0; const o = 12 * c;
    if (id === 'native') for (let i = 0; i < 12; i++) t += st.p[o + i];
    else for (let i = 0; i < 12; i++) t += st.re[o + i] * st.re[o + i] + st.im[o + i] * st.im[o + i];
    out[c] = t; tot += t;
  }
  return tot;
}
function positionStats(cell, tot) {
  const N = G.N, x = G.x, y = G.y; let mx = 0, my = 0;
  for (let c = 0; c < N; c++) { mx += cell[c] * x[c]; my += cell[c] * y[c]; }
  mx /= tot; my /= tot; let v = 0;
  for (let c = 0; c < N; c++) v += cell[c] * ((x[c] - mx) ** 2 + (y[c] - my) ** 2);
  return { mx, my, v: v / tot };
}
function momentumLaw(st, id, out) {    // Born law of the plane-wave analyser over the view cells, on the KN x KN grid
  const A = G.view, nA = A.length, cr = W.cr, ci = W.ci, gr = W.gr, gi = W.gi;
  const re = id === 'native' ? st.cr : st.re, im = id === 'native' ? st.ci : st.im, s12 = 1 / Math.sqrt(12);
  for (let n = 0; n < nA; n++) { const o = 12 * A[n]; let a = 0, b = 0; for (let i = 0; i < 12; i++) { a += re[o + i]; b += im[o + i]; } cr[n] = a * s12; ci[n] = b * s12; }
  let total = 0;
  for (let j = 0; j < KN; j++) {
    const cy = G.cy, sy = G.sy, oj = j * nA;
    for (let n = 0; n < nA; n++) { gr[n] = cr[n] * cy[oj + n] + ci[n] * sy[oj + n]; gi[n] = ci[n] * cy[oj + n] - cr[n] * sy[oj + n]; }
    for (let i = 0; i < KN; i++) {
      const cx = G.cx, sx = G.sx, oi = i * nA; let a = 0, b = 0;
      for (let n = 0; n < nA; n++) { a += gr[n] * cx[oi + n] + gi[n] * sx[oi + n]; b += gi[n] * cx[oi + n] - gr[n] * sx[oi + n]; }
      const v = a * a + b * b; out[i * KN + j] = v; total += v;
    }
  }
  let coherent = 1;
  if (id === 'native') { coherent = norm2(st.cr, st.ci) / Math.max(1e-300, st.mass); }
  const flat = (1 - coherent) / (KN * KN);
  let mx = 0, my = 0;
  for (let i = 0; i < KN; i++) for (let j = 0; j < KN; j++) {
    const v = (total > 0 ? out[i * KN + j] / total * coherent : 0) + flat; out[i * KN + j] = v;
    mx += v * G.ks[i]; my += v * G.ks[j];
  }
  let va = 0; for (let i = 0; i < KN; i++) for (let j = 0; j < KN; j++) va += out[i * KN + j] * ((G.ks[i] - mx) ** 2 + (G.ks[j] - my) ** 2);
  return { mx, my, v: va, coherent };
}
function detectorMass(st, id) {
  let t = 0; const m = G.detPorts;
  if (id === 'native') for (let i = 0; i < m.length; i++) t += st.p[m[i]];
  else for (let i = 0; i < m.length; i++) t += st.re[m[i]] ** 2 + st.im[m[i]] ** 2;
  return t;
}

// ------------------------------------------------------------------ preparation and measurements
function say(t) { ST.log.unshift(t); ST.log = ST.log.slice(0, 9); $('lab-log').innerHTML = ST.log.map((l) => `<div>${l}</div>`).join(''); }

function fillPacket(re, im, parts) {   // parts: [{x0, y0, w, kx, ky}], constant on the twelve ports of a carrier
  const N = G.N; re.fill(0); im.fill(0);
  for (let c = 0; c < N; c++) {
    let a = 0, b = 0;
    for (const q of parts) {
      const dx = G.x[c] - q.x0, dy = G.y[c] - q.y0, e = Math.exp(-(dx * dx + dy * dy) / (4 * q.w * q.w));
      if (e < 1e-12) continue;
      const ph = q.kx * G.x[c] + q.ky * G.y[c]; a += e * Math.cos(ph); b += e * Math.sin(ph);
    }
    const o = 12 * c; for (let i = 0; i < 12; i++) { re[o + i] = a; im[o + i] = b; }
  }
  const s = 1 / Math.sqrt(norm2(re, im)); for (let p = 0; p < re.length; p++) { re[p] *= s; im[p] *= s; }
}
function setStates(re, im) {
  const P = G.P, n = ST.native, t = ST.transfer, w = ST.wave;
  for (let p = 0; p < P; p++) n.p[p] = re[p] * re[p] + im[p] * im[p];
  n.cr.set(re); n.ci.set(im);
  if (!n.clicked) { let own = 0; for (const q of G.obsPorts) { own += n.p[q]; n.p[q] = 0; n.cr[q] = 0; n.ci[q] = 0; } if (own > 0) { const s = 1 / (1 - own); for (let p = 0; p < P; p++) n.p[p] *= s; } }
  t.re.set(re); t.im.set(im);
  w.re.set(re); w.im.set(im); waveInit(w);
}
function prepare() {
  const dist = parseFloat($('lab-dist').value), w = parseFloat($('lab-width').value), k = parseFloat($('lab-k').value), two = $('lab-two').checked;
  const parts = [{ x0: -dist, y0: 0, w, kx: k, ky: 0 }];
  if (two) parts.push({ x0: dist, y0: 0, w, kx: -k, ky: 0 });
  ST.parts = parts; ST.s = 0; ST.waveDebt = 0;
  Object.assign(ST.native, { click: 0, clicked: false, u: Math.random(), clickSweep: null });
  ST.wave.T = 0;
  fillPacket(W.pr, W.pi, parts);
  setStates(W.pr, W.pi);
  ST.prepared = true; ST.frame = 0; ST.kref = 0;
  momentumLaw(ST.transfer, 'transfer', W.klaw.transfer); for (let i = 0; i < KN * KN; i++) if (W.klaw.transfer[i] > ST.kref) ST.kref = W.klaw.transfer[i];
  say(`prepared ${two ? 'two packets' : 'one packet'} ${dist} edges from the observer, width ${w}, wave number ${k} per edge`);
  refresh(true);
}
function readPosition() {
  if (!ST.prepared) return;
  const u = Math.random(), res = [];
  for (const law of LAWS) {
    const st = ST[law.id], P = G.P, inside = new Uint8Array(P); for (const q of G.detPorts) inside[q] = 1;
    const tot = law.id === 'native' ? st.p.reduce((a, b) => a + b, 0) : norm2(st.re, st.im);
    const pin = detectorMass(st, law.id) / tot, hit = u < pin;
    if (law.id === 'native') {
      let m = 0; for (let p = 0; p < P; p++) { if (!!inside[p] !== hit) { st.p[p] = 0; st.cr[p] = 0; st.ci[p] = 0; } m += st.p[p]; }
      const s = 1 / m, sq = Math.sqrt(s); for (let p = 0; p < P; p++) { st.p[p] *= s; st.cr[p] *= sq; st.ci[p] *= sq; }
      st.click = 0; st.u = Math.random(); st.clicked = hit && st.clicked;
    } else {
      for (let p = 0; p < P; p++) if (!!inside[p] !== hit) { st.re[p] = 0; st.im[p] = 0; }
      const s = 1 / Math.sqrt(norm2(st.re, st.im)); for (let p = 0; p < P; p++) { st.re[p] *= s; st.im[p] *= s; }
      if (law.id === 'wave') waveInit(st);
    }
    res.push(`${law.title.toLowerCase()}: ${hit ? 'found' : 'not found'} (probability ${(100 * pin).toFixed(1)}%)`);
  }
  say('position detector read · ' + res.join(' · '));
  refresh(true);
}
function readMomentum() {
  if (!ST.prepared) return;
  const u = Math.random(), res = [], win = Math.min(5, G.viewR / 2.2);
  for (const law of LAWS) {
    const st = ST[law.id], lawK = W.klaw[law.id];
    if (law.id === 'native') st.mass = st.p.reduce((a, b) => a + b, 0);
    momentumLaw(st, law.id, lawK);
    let acc = 0, pick = KN * KN - 1; for (let i = 0; i < KN * KN; i++) { acc += lawK[i]; if (u < acc) { pick = i; break; } }
    const kx = G.ks[Math.floor(pick / KN)], ky = G.ks[pick % KN];
    const cell = W.cell[law.id], tot = cellLaw(st, law.id, cell), ps = positionStats(cell, tot);
    fillPacket(W.pr, W.pi, [{ x0: ps.mx, y0: ps.my, w: win, kx, ky }]);
    if (law.id === 'native') { for (let p = 0; p < G.P; p++) st.p[p] = W.pr[p] ** 2 + W.pi[p] ** 2; st.cr.set(W.pr); st.ci.set(W.pi); st.click = 0; st.u = Math.random(); st.clicked = false; }
    else { st.re.set(W.pr); st.im.set(W.pi); if (law.id === 'wave') waveInit(st); }
    res.push(`${law.title.toLowerCase()}: k = (${kx.toFixed(2)}, ${ky.toFixed(2)})`);
  }
  say('momentum analyser read · ' + res.join(' · '));
  refresh(true);
}

// ------------------------------------------------------------------ time stepping
function advance() {
  const ds = parseFloat($('lab-speed').value);
  nativeAdvance(ST.native, ds);
  transferAdvance(ST.transfer, ds);
  const clock = $('lab-clock').value, phi = parseFloat($('lab-phi').value);
  if (eventDriven()) { waveEvents(ST.wave, ds, phi, clock === 'poisson'); ST.wave.T += phi * Math.SQRT1_2 * ds; }
  else {
    // T gained per sweep: phi / sqrt 2 on the regular clock (frequency phi sqrt(lambda / 2)), 6 / sqrt M on the paper clock
    ST.waveDebt += clock === 'paper' ? 6 * ds / Math.sqrt(G.M) : phi * Math.SQRT1_2 * ds;
    let n = 0; while (ST.waveDebt >= DT && n < 400) { waveStep(ST.wave); ST.waveDebt -= DT; n++; }
    if (n === 400) ST.waveDebt = 0;
  }
  ST.s += ds; ST.frame++;
}

// ------------------------------------------------------------------ drawing
function lut(rgb) {
  const t = new Array(256);
  for (let i = 0; i < 256; i++) {
    const v = i / 255, a = Math.min(1, v * 1.6), b = Math.max(0, (v - 0.62) / 0.38);
    const c = rgb.map((ch) => Math.round(14 + (ch - 14) * a + (255 - ch) * b * 0.8));
    t[i] = `rgb(${c[0]},${c[1]},${c[2]})`;
  }
  return t;
}
function drawPanel(law, cell, tot, stats, kstats) {
  const cv = $('lab-cv-' + law.id), ctx = cv.getContext('2d'), Wd = cv.width, H = cv.height, sc = Wd / (2 * G.viewR), cx = Wd / 2, cy = H / 2;
  ctx.fillStyle = '#070b13'; ctx.fillRect(0, 0, Wd, H);
  let mx = 0; for (const c of G.view) if (cell[c] > mx) mx = cell[c];
  const ref = Math.max(mx, 1e-12), L = law.lut, T = G.tri;
  for (const c of G.view) {
    const v = Math.sqrt(cell[c] / ref); if (v < 0.03) continue;
    ctx.fillStyle = L[Math.min(255, Math.floor(v * 255))];
    const o = 6 * c;
    ctx.beginPath(); ctx.moveTo(cx + T[o] * sc, cy - T[o + 1] * sc); ctx.lineTo(cx + T[o + 2] * sc, cy - T[o + 3] * sc); ctx.lineTo(cx + T[o + 4] * sc, cy - T[o + 5] * sc); ctx.closePath(); ctx.fill();
  }
  ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(255,255,255,.28)'; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.arc(cx, cy, G.detR * sc, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = 'rgba(147,163,192,.32)';
  for (const q of ST.parts || []) { const r = FRONT * ST.s * sc; if (r < Wd) { ctx.beginPath(); ctx.arc(cx + q.x0 * sc, cy - q.y0 * sc, r, 0, 2 * Math.PI); ctx.stroke(); } }
  ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(cx, cy, 3.2, 0, 2 * Math.PI); ctx.fill();
  ctx.strokeStyle = '#fff'; ctx.beginPath(); ctx.arc(cx, cy, 7, 0, 2 * Math.PI); ctx.stroke();
  ctx.strokeStyle = '#62718f'; ctx.beginPath(); ctx.moveTo(12, H - 14); ctx.lineTo(12 + sc, H - 14); ctx.stroke();
  ctx.fillStyle = '#93a3c0'; ctx.font = '11px Helvetica'; ctx.fillText('1 mesh edge', 12, H - 20);
  // momentum panel
  const kc = $('lab-k-' + law.id), kx = kc.getContext('2d'), n = kc.width / KN, lawK = W.klaw[law.id];
  let km = ST.kref || 0; for (let i = 0; i < KN * KN; i++) if (lawK[i] > km) km = lawK[i];
  kx.fillStyle = '#070b13'; kx.fillRect(0, 0, kc.width, kc.height);
  for (let i = 0; i < KN; i++) for (let j = 0; j < KN; j++) {
    const v = Math.sqrt(lawK[i * KN + j] / Math.max(km, 1e-12)); if (v < 0.03) continue;
    kx.fillStyle = L[Math.min(255, Math.floor(v * 255))]; kx.fillRect(i * n, kc.height - (j + 1) * n, n + 0.5, n + 0.5);
  }
  kx.strokeStyle = 'rgba(255,255,255,.25)'; kx.beginPath(); kx.moveTo(kc.width / 2, 0); kx.lineTo(kc.width / 2, kc.height); kx.moveTo(0, kc.height / 2); kx.lineTo(kc.width, kc.height / 2); kx.stroke();
  const st = ST[law.id], dx = Math.sqrt(stats.v), dk = Math.sqrt(kstats.v);
  const rows = [['position ⟨x⟩, ⟨y⟩', `${stats.mx.toFixed(2)}, ${stats.my.toFixed(2)} edges`], ['spread Δx', `${dx.toFixed(2)} edges`],
    ['momentum ⟨k⟩', `${kstats.mx.toFixed(2)}, ${kstats.my.toFixed(2)} per edge`], ['spread Δk', `${dk.toFixed(2)} per edge`],
    ['Δx · Δk (bound 1)', `<b style="color:${dx * dk < 1.5 ? 'var(--green)' : 'var(--soft)'}">${(dx * dk).toFixed(2)}</b>`],
    ['in the detector now', `${(100 * detectorMass(st, law.id) / tot).toFixed(2)}%`]];
  if (law.id === 'native') rows.push(['coherent weight left', kstats.coherent < 1e-4 ? kstats.coherent.toExponential(1) : kstats.coherent.toFixed(4)],
    ['observer click so far', st.clicked ? `clicked at sweep ${st.clickSweep.toFixed(1)}` : `${(100 * st.click).toFixed(2)}%`]);
  if (law.id === 'transfer') rows.push(['clock', `${ST.s.toFixed(1)} sweeps`]);
  if (law.id === 'wave') rows.push(['clock', eventDriven() ? `${Math.round(ST.s * G.M).toLocaleString('en-US')} seam events` : `T = ${st.T.toFixed(2)} · ${(st.T * Math.sqrt(G.M)).toFixed(0)} walk steps of the paper`]);
  $('lab-st-' + law.id).innerHTML = rows.map(([k, v]) => `<div class="row"><span>${k}</span><b>${v}</b></div>`).join('');
}
function refresh(force) {
  if (!ST.prepared) return;
  const doK = force || ST.frame % 4 === 0;
  for (const law of LAWS) {
    const st = ST[law.id], cell = W.cell[law.id], tot = cellLaw(st, law.id, cell);
    if (law.id === 'native') st.mass = tot;
    const stats = positionStats(cell, tot);
    if (doK) W.kst[law.id] = momentumLaw(st, law.id, W.klaw[law.id]);
    drawPanel(law, cell, tot, stats, W.kst[law.id]);
  }
  $('lab-clockline').textContent = `${ST.s.toFixed(1)} sweeps since preparation · ${(33 * ST.s).toFixed(0)} own ticks of the observer · influence front at ${(FRONT * ST.s).toFixed(1)} edges`;
}
function loop() {
  requestAnimationFrame(loop);
  if (!ST.running || !$('page-detector').classList.contains('on')) return;
  advance(); refresh(false);
  if (ST.auto && ST.frame >= ST.auto) { ST.running = false; $('lab-run').textContent = '▶ run'; if (ST.autoRead === 'pos') readPosition(); if (ST.autoRead === 'mom') readMomentum(); ST.auto = 0; window.__labDone = true; }
}

// ------------------------------------------------------------------ setup
async function loadLab(level) {
  $('lab-status').textContent = 'loading the federation…';
  const z = await loadNpz(`/lab/L${level}.npz`);
  const num = z.numbers;
  Object.assign(G, { level, a: z.a, b: z.b, M: z.a.length, P: z.deg.length, N: z.deg.length / 12, deg: Float64Array.from(z.deg),
    obsPorts: Array.from(z.observer_ports), obsEnds: Array.from(z.observer_outside_ends), radius: num[5], sigma: num[6], density: num[7] });
  G.x = new Float64Array(G.N); G.y = new Float64Array(G.N); G.tri = z.triangles;
  for (let c = 0; c < G.N; c++) { G.x[c] = z.chart[2 * c]; G.y[c] = z.chart[2 * c + 1]; }
  G.viewR = Math.min(11, 0.55 * Math.PI * G.radius);
  G.view = []; for (let c = 0; c < G.N; c++) if (Math.abs(G.x[c]) <= G.viewR && Math.abs(G.y[c]) <= G.viewR) G.view.push(c);
  G.krylov = level <= 4 ? 36 : 22;
  const P = G.P, f64 = () => new Float64Array(P);
  Object.assign(W, { r0: f64(), r1: f64(), t0r: f64(), t0i: f64(), t1r: f64(), t1i: f64(), ar: f64(), ai: f64(), pr: f64(), pi: f64(), q: f64(),
    Q: new Float64Array(G.krylov * P), cell: {}, klaw: {}, kst: {} });
  ST.native = { p: f64(), cr: f64(), ci: f64(), click: 0, clicked: false, u: 1, mass: 1 };
  ST.transfer = { re: f64(), im: f64() };
  ST.wave = { re: f64(), im: f64(), pre: f64(), pim: f64(), T: 0, er: new Float64Array(G.M), ei: new Float64Array(G.M), order: new Int32Array(G.M) };
  for (let i = 0; i < G.M; i++) ST.wave.order[i] = i;
  for (const law of LAWS) { W.cell[law.id] = new Float64Array(G.N); W.klaw[law.id] = new Float64Array(KN * KN); law.lut = lut(law.color); }
  const nA = G.view.length; G.ks = new Float64Array(KN);
  for (let i = 0; i < KN; i++) G.ks[i] = -KMAX + 2 * KMAX * i / (KN - 1);
  for (const key of ['cx', 'sx', 'cy', 'sy']) G[key] = new Float32Array(KN * nA);
  for (let i = 0; i < KN; i++) for (let n = 0; n < nA; n++) {
    const c = G.view[n]; G.cx[i * nA + n] = Math.cos(G.ks[i] * G.x[c]); G.sx[i * nA + n] = Math.sin(G.ks[i] * G.x[c]);
    G.cy[i * nA + n] = Math.cos(G.ks[i] * G.y[c]); G.sy[i * nA + n] = Math.sin(G.ks[i] * G.y[c]);
  }
  Object.assign(W, { cr: new Float64Array(nA), ci: new Float64Array(nA), gr: new Float64Array(nA), gi: new Float64Array(nA) });
  $('lab-dist').max = Math.max(3, Math.floor(G.viewR - 2)); if (parseFloat($('lab-dist').value) > G.viewR - 2) $('lab-dist').value = Math.floor(G.viewR - 2);
  setDetector();
  const c = Math.sqrt(G.sigma / G.density), m = G.density / G.sigma;
  $('lab-numbers').innerHTML = [['federation', `level ${level}: ${(G.N).toLocaleString('en-US')} observers, ${P.toLocaleString('en-US')} ports`],
    ['sphere radius', `${G.radius.toFixed(1)} mesh edges`], ['native diffusion constant', `${(0.5 / m).toFixed(5)} edge² per sweep`],
    ['native coherence decay', '5 to 6 per sweep'], ['influence front, the signal limit', `${FRONT} edges per sweep`],
    ['transfer continuation', `E = k² / (2m), m = ${m.toFixed(1)}`], ['admitted wave, paper clock', `${(6 * c / Math.sqrt(G.M)).toFixed(5)} edges per sweep`],
    ['admitted wave, seam events', `${(c * Math.SQRT1_2).toFixed(4)} × pulse angle, edges per sweep`],
    ['wave under the declared schedule', 'scattered at about 0.25 × angle² per sweep']].map(([k, v]) => `<div class="row"><span>${k}</span><b>${v}</b></div>`).join('');
  G.ready = true; ST.prepared = false;
  $('lab-status').textContent = '';
}
function setDetector() {
  G.detR = parseFloat($('lab-det').value);
  const ports = []; for (let c = 0; c < G.N; c++) if (G.x[c] * G.x[c] + G.y[c] * G.y[c] <= G.detR * G.detR) for (let i = 0; i < 12; i++) ports.push(12 * c + i);
  G.detPorts = Int32Array.from(ports);
}
export async function labShown() {
  if (G.loading) { await G.loading; return; }
  if (!G.ready) { G.loading = loadLab(parseInt($('lab-level').value)); try { await G.loading; prepare(); } finally { G.loading = null; } }
  for (const law of LAWS) { const cv = $('lab-cv-' + law.id); const r = cv.getBoundingClientRect(); if (r.width && cv.width !== Math.round(r.width)) { cv.width = Math.round(r.width); cv.height = Math.round(r.width); } }
  refresh(true);
}
export function initLab() {
  const host = $('lab-panels');
  host.innerHTML = LAWS.map((l) => `<div class="lab-col"><div class="k" style="color:rgb(${l.color.join(',')})">${l.title}</div><div class="note" style="min-height:34px;margin:0 0 6px">${l.note}</div>
    <canvas id="lab-cv-${l.id}" width="380" height="380"></canvas><div class="lab-k"><canvas id="lab-k-${l.id}" width="132" height="132" title="what the momentum analyser would record: wave vectors from -${KMAX} to ${KMAX} per mesh edge"></canvas>
    <div class="card" id="lab-st-${l.id}" style="flex:1;margin:0"></div></div></div>`).join('');
  $('lab-prepare').onclick = prepare;
  $('lab-run').onclick = () => { if (!ST.prepared) return; ST.running = !ST.running; $('lab-run').textContent = ST.running ? '❚❚ pause' : '▶ run'; };
  $('lab-step').onclick = () => { if (ST.prepared) { advance(); refresh(true); } };
  $('lab-readx').onclick = readPosition;
  $('lab-readk').onclick = readMomentum;
  $('lab-det').oninput = () => { if (G.ready) { setDetector(); refresh(true); } };
  $('lab-level').onchange = async () => { ST.running = false; $('lab-run').textContent = '▶ run'; G.ready = false; await labShown(); };
  $('lab-clock').onchange = () => { if (G.ready) prepare(); };
  $('lab-phi').onchange = () => { if (G.ready) prepare(); };
  for (const id of ['lab-dist', 'lab-width', 'lab-k']) $(id).oninput = () => { $(id + '-v').textContent = $(id).value; };
  window.addEventListener('resize', () => { if ($('page-detector').classList.contains('on')) labShown(); });
  loop();
  window.__lab = { G, ST, prepare, readPosition, readMomentum, advance, refresh,
    auto: (frames, read) => { ST.auto = frames; ST.autoRead = read || ''; ST.running = true; } };
}
