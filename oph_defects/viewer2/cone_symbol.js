// Symbols of the two-tick period of worker T3 (repair-law/work/T3/t3sym.py, report reports/T3.md) in JavaScript.
// Checks against T3's numbers: node space_gauge_check.mjs.
//
// Coin index 2r + s (record r, spin s in the frame basis), alpha_i = X_r sigma_i, beta = Y_r, as in cf1_engine.js.
// A word lists tokens in time order (first applied first):
//   ['S', i, e]    routing sub-step S_i = Q_i R (e = +1) or its inverse; symbol cos p_i - i e sin p_i alpha_i
//   ['M', a]       record rotation exp(-i a beta) (mass step M1)
//   ['L', i, phi]  lazy sub-step of member B: seam reflector with weights cos(eta), sin(eta), eta = pi/4 - phi/2, then R;
//                  symbol cos(phi) (cos p_i - i sin p_i alpha_i) - i sin(phi) beta
// Every token is c0 - i c1 alpha_i - i c2 beta. The phase Omega per period follows from cos Omega = Re tr W / 4.
// alpha_i and beta are signed permutation matrices: row k has one entry, at column PERM[m][k], value VAL[m][k] (re, im).
const PERM = [[3, 2, 1, 0], [3, 2, 1, 0], [2, 3, 0, 1], [2, 3, 0, 1]];
const VAL = [
  [[1, 0], [1, 0], [1, 0], [1, 0]],          // alpha_1 = X (x) sigma_x
  [[0, -1], [0, 1], [0, -1], [0, 1]],        // alpha_2 = X (x) sigma_y
  [[1, 0], [-1, 0], [1, 0], [-1, 0]],        // alpha_3 = X (x) sigma_z
  [[0, -1], [0, -1], [0, 1], [0, 1]],        // beta = Y (x) 1
];
export const S = (i, e = 1) => ['S', i, e];
export const M = (a) => ['M', a];
export const L = (i, phi) => ['L', i, phi];
const rev = (w) => w.slice().reverse();
// the CF-1 period in the half-step frame: M(th/2) S1 S2 S3 M(th) S3 S2 S1 M(th/2)
export function periodCF1(theta = 0) { const x = [S(0), S(1), S(2)]; return [M(theta / 2), ...x, M(theta), ...rev(x), M(theta / 2)]; }
// member B: B_i = M(-phi/2) L_i(phi) M(-phi/2); P = M(th/2) B1 B2 B3 M(th) B3 B2 B1 M(th/2)
export function periodB(phi = 5 * Math.PI / 16, theta = 0) { const h = []; for (let i = 0; i < 3; i++) h.push(M(-phi / 2), L(i, phi), M(-phi / 2)); return [M(theta / 2), ...h, M(theta), ...rev(h), M(theta / 2)]; }
// merge neighbouring record rotations (they commute with each other): M(a) M(b) = M(a + b)
export function compile(word) { const out = []; for (const t of word) { const last = out[out.length - 1]; if (t[0] === 'M' && last && last[0] === 'M') out[out.length - 1] = ['M', last[1] + t[1]]; else out.push(t.slice()); } return out.filter((t) => !(t[0] === 'M' && t[1] === 0)); }

const _a = { re: new Float64Array(16), im: new Float64Array(16) }, _b = { re: new Float64Array(16), im: new Float64Array(16) };
// left multiplication Y = (c0 - i c1 alpha_m - i c2 beta) X
function leftMul(c0, c1, m, c2, X, Y) {
  for (let r = 0; r < 4; r++) {
    const ka = PERM[m][r], [var_, vai] = VAL[m][r], kb = PERM[3][r], [vbr, vbi] = VAL[3][r];
    for (let c = 0; c < 4; c++) {
      const o = 4 * r + c, xa = 4 * ka + c, xb = 4 * kb + c;
      // -i c (v z) = c (Im(v z) - i Re(v z))
      const azr = var_ * X.re[xa] - vai * X.im[xa], azi = var_ * X.im[xa] + vai * X.re[xa];
      const bzr = vbr * X.re[xb] - vbi * X.im[xb], bzi = vbr * X.im[xb] + vbi * X.re[xb];
      Y.re[o] = c0 * X.re[o] + c1 * azi + c2 * bzi;
      Y.im[o] = c0 * X.im[o] - c1 * azr - c2 * bzr;
    }
  }
}
function run(word, p) {          // the product in a work buffer (no copy)
  let X = _a, Y = _b; X.re.fill(0); X.im.fill(0); for (let d = 0; d < 4; d++) X.re[5 * d] = 1;
  for (const t of word) {
    if (t[0] === 'S') leftMul(Math.cos(p[t[1]]), t[2] * Math.sin(p[t[1]]), t[1], 0, X, Y);
    else if (t[0] === 'M') leftMul(Math.cos(t[1]), 0, 0, Math.sin(t[1]), X, Y);
    else { const cp = Math.cos(t[2]); leftMul(cp * Math.cos(p[t[1]]), cp * Math.sin(p[t[1]]), t[1], Math.sin(t[2]), X, Y); }
    const z = X; X = Y; Y = z;
  }
  return X;
}
// the 4 x 4 symbol of a word at wave vector p: { re, im } row major
export function symbol(word, p) { const X = run(word, p); return { re: Float64Array.from(X.re), im: Float64Array.from(X.im) }; }
export function cosOmega(word, p) { const W = run(word, p); return (W.re[0] + W.re[5] + W.re[10] + W.re[15]) / 4; }
export const omega = (word, p) => Math.acos(Math.max(-1, Math.min(1, cosOmega(word, p))));

// phase per period on an n x n grid of (p1, p2) in [-pi, pi)^2 at fixed p3 (row index p2, column index p1)
export function omegaSlice(word, n, p3) {
  const out = new Float32Array(n * n), p = [0, 0, p3];
  for (let b = 0; b < n; b++) for (let a = 0; a < n; a++) { p[0] = -Math.PI + 2 * Math.PI * a / n; p[1] = -Math.PI + 2 * Math.PI * b / n; out[b * n + a] = omega(word, p); }
  return out;
}

// Nelder-Mead in three dimensions
function nelderMead(f, x0, step = 0.05, iters = 600) {
  let P = [x0.slice()]; for (let k = 0; k < 3; k++) { const x = x0.slice(); x[k] += step; P.push(x); }
  let F = P.map(f);
  for (let it = 0; it < iters; it++) {
    const ord = [0, 1, 2, 3].sort((i, j) => F[i] - F[j]); P = ord.map((i) => P[i]); F = ord.map((i) => F[i]);
    if (F[3] - F[0] < 1e-17 && it > 50) break;
    const c = [0, 1, 2].map((k) => (P[0][k] + P[1][k] + P[2][k]) / 3), at = (t) => c.map((v, k) => v + t * (P[3][k] - v));
    const xr = at(-1), fr = f(xr);
    if (fr < F[0]) { const xe = at(-2), fe = f(xe); if (fe < fr) { P[3] = xe; F[3] = fe; } else { P[3] = xr; F[3] = fr; } }
    else if (fr < F[2]) { P[3] = xr; F[3] = fr; }
    else { const xc = at(fr < F[3] ? -0.5 : 0.5), fc = f(xc); if (fc < Math.min(fr, F[3])) { P[3] = xc; F[3] = fc; } else { for (let i = 1; i < 4; i++) { P[i] = P[i].map((v, k) => P[0][k] + 0.5 * (v - P[0][k])); F[i] = f(P[i]); } } }
  }
  const i = F.indexOf(Math.min(...F)); return [P[i], F[i]];
}
const wrap = (x) => x.map((v) => ((v + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI);
const dist = (x, y) => Math.hypot(...x.map((v, k) => { let d = (v - y[k]) % (2 * Math.PI); if (d > Math.PI) d -= 2 * Math.PI; if (d < -Math.PI) d += 2 * Math.PI; return d; }));
// Zero-phase points (Omega = 0) and phase-pi points of a period over the full zone, as T3 counts them: discrete local
// minima of Omega and of pi - Omega on an n^3 grid, refined, zero accepted at 1 - |cos Omega| < 1e-11. Also the
// lightest nonzero minimum (the lightest doubler) and the gap at pi.
export function criticalPoints(word, n = 32, thr = 0.35, nlow = 10) {
  const g = new Float64Array(n * n * n), idx = (a, b, c) => ((((a % n) + n) % n) * n + (((b % n) + n) % n)) * n + (((c % n) + n) % n), pt = (a, b, c) => [a, b, c].map((v) => -Math.PI + 2 * Math.PI * v / n);
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) for (let c = 0; c < n; c++) g[idx(a, b, c)] = omega(word, pt(a, b, c));
  const out = {};
  for (const kind of ['zero', 'pi']) {
    const f = (v) => (kind === 'zero' ? v : Math.PI - v), seeds = [];
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) for (let c = 0; c < n; c++) {
      const v = f(g[idx(a, b, c)]); let min = true;
      for (let da = -1; da <= 1 && min; da++) for (let db = -1; db <= 1 && min; db++) for (let dc = -1; dc <= 1; dc++) { if (!da && !db && !dc) continue; if (f(g[idx(a + da, b + db, c + dc)]) < v) { min = false; break; } }
      if (min) seeds.push([v, pt(a, b, c)]);
    }
    seeds.sort((x, y) => x[0] - y[0]);
    const keep = seeds.filter((s, j) => s[0] < thr || j < nlow), found = [];
    const obj = kind === 'zero' ? (q) => 1 - cosOmega(word, q) : (q) => 1 + cosOmega(word, q);
    for (const [, x0] of keep) {
      let [x, fv] = nelderMead(obj, x0, 0.04); [x, fv] = nelderMead(obj, x, 0.002);
      x = wrap(x); let val = f(omega(word, x)); if (fv < 1e-11) val = 0;
      if (!found.some((y) => dist(x, y.p) < 1e-4)) found.push({ p: x, value: val });
    }
    found.sort((x, y) => x.value - y.value); out[kind] = found;
  }
  const zeros = out.zero.filter((x) => x.value === 0), pis = out.pi.filter((x) => x.value === 0), others = out.zero.filter((x) => x.value > 0);
  return { zeros, pis, lightest: others.length ? others[0].value : null, lightestAt: others.length ? others[0].p : null, gapAtPi: out.pi.length ? out.pi[0].value : null };
}
// massless speed in cells per tick along an axis: Omega(eps e_1) / eps per period, halved
export function speedPerTick(word, eps = 1e-5) { const o0 = omega(word, [0, 0, 0]), o = omega(word, [eps, 0, 0]); return Math.sqrt(Math.max(o * o - o0 * o0, 0)) / eps / 2; }

// chirality of a node in an arrow block (chi = +1, -1): U = V^dag W V, U(p* + q) = e^{-i eps0} exp(-i (J q).sigma);
// chirality = sign det J (t3sym.node_jacobian)
export function blockChirality(word, p, chi, h = 1e-6) {
  const s2 = Math.SQRT1_2, v = chi > 0 ? [s2, s2] : [s2, -s2];            // arrow state |+-> = (|0> +- |1>)/sqrt2 of the record
  const block = (q) => {          // 2 x 2 block: U[s][s'] = sum_{r r'} v_r v_r' W[(r,s),(r',s')]
    const W = symbol(word, q), re = [0, 0, 0, 0], im = [0, 0, 0, 0];
    for (let s = 0; s < 2; s++) for (let t = 0; t < 2; t++) for (let r = 0; r < 2; r++) for (let r2 = 0; r2 < 2; r2++) { const o = 4 * (2 * r + s) + 2 * r2 + t; re[2 * s + t] += v[r] * v[r2] * W.re[o]; im[2 * s + t] += v[r] * v[r2] * W.im[o]; }
    return [re, im];
  };
  // U = a0 - i a.sigma: a0 = Re tr U / 2, a_j = Re(i tr(U sigma_j)) / 2
  const su2 = ([re, im]) => [(re[0] + re[3]) / 2, [-(im[1] + im[2]) / 2, (re[2] - re[1]) / 2, -(im[0] - im[3]) / 2]];
  const [a0] = su2(block(p)), s0 = a0 > 0 ? 1 : -1, J = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let j = 0; j < 3; j++) { const pp = p.slice(), pm = p.slice(); pp[j] += h; pm[j] -= h; const ap = su2(block(pp))[1], am = su2(block(pm))[1]; for (let k = 0; k < 3; k++) J[k][j] = s0 * (ap[k] - am[k]) / (2 * h); }
  const det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
  return { chirality: Math.sign(det), det, phase: s0 > 0 ? 0 : Math.PI };
}
