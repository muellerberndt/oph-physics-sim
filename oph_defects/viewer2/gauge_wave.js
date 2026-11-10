// Small-field waves of the continuous links of worker GL (repair-law/work/GL/s3_waves.py, report reports/GL.md) in
// JavaScript. Checks against GL's numbers: node space_gauge_check.mjs.
//
// Links take values in G = S(U(3) x U(2)) on the five-level code and are stored as two blocks: colour 3 x 3 (code
// indices 0..2) and weak 2 x 2 (indices 3..4). Electric fields have twelve components along GL's basis TBASIS
// (gluons lambda_1, lambda_2, lambda_4, lambda_5, lambda_6, lambda_7, lambda_3, lambda_8 over 2; W1, W2, W3 = sigma/2;
// B_GUT = sqrt(3/5) Y; tr T_a T_b = delta/2). Classical Hamiltonian lattice field: code-trace plaquette action
// (10/g^2) sum (1 - Re tr5 U_P / 5), Casimir electric term (g^2/2) sum E^2, leapfrog with step 1/2, two field steps
// per tick: U <- exp(i tau E.T) U, E <- E + tau F, after one half kick.
export const TAU = 0.5;
const NB = 26;                                   // doubles per link: colour block 18, weak block 8 (re, im interleaved)
const off = (r, c) => (r < 3 ? 2 * (3 * r + c) : 18 + 2 * (2 * (r - 3) + (c - 3)));

// TBASIS as sparse lists [row, col, re, im]
export const GEN = (() => {
  const g = [];
  for (const [i, j] of [[0, 1], [0, 2], [1, 2]]) { g.push([[i, j, 0.5, 0], [j, i, 0.5, 0]]); g.push([[i, j, 0, -0.5], [j, i, 0, 0.5]]); }
  g.push([[0, 0, 0.5, 0], [1, 1, -0.5, 0]]);
  const r8 = 1 / (2 * Math.sqrt(3)); g.push([[0, 0, r8, 0], [1, 1, r8, 0], [2, 2, -2 * r8, 0]]);
  g.push([[3, 4, 0.5, 0], [4, 3, 0.5, 0]]); g.push([[3, 4, 0, -0.5], [4, 3, 0, 0.5]]); g.push([[3, 3, 0.5, 0], [4, 4, -0.5, 0]]);
  const b = Math.sqrt(3 / 5); g.push([[0, 0, -b / 3, 0], [1, 1, -b / 3, 0], [2, 2, -b / 3, 0], [3, 3, b / 2, 0], [4, 4, b / 2, 0]]);
  return g.map((ent) => ent.map(([r, c, re, im]) => ({ r, c, re, im, o: off(r, c), t: off(c, r) })));
})();
export const GEN_NAMES = ['gluon λ1', 'gluon λ2', 'gluon λ4', 'gluon λ5', 'gluon λ6', 'gluon λ7', 'gluon λ3', 'gluon λ8', 'W1', 'W2', 'W3', 'B'];

// C = op(A) op(B) on both blocks; op = conjugate transpose when the flag is set. C must not alias A or B.
function mm(A, a, ta, B, b, tb, C, c) {
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let sr = 0, si = 0;
    for (let k = 0; k < 3; k++) {
      const x = a + (ta ? 2 * (3 * k + i) : 2 * (3 * i + k)), y = b + (tb ? 2 * (3 * j + k) : 2 * (3 * k + j));
      const ar = A[x], ai = ta ? -A[x + 1] : A[x + 1], br = B[y], bi = tb ? -B[y + 1] : B[y + 1];
      sr += ar * br - ai * bi; si += ar * bi + ai * br;
    }
    C[c + 2 * (3 * i + j)] = sr; C[c + 2 * (3 * i + j) + 1] = si;
  }
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
    let sr = 0, si = 0;
    for (let k = 0; k < 2; k++) {
      const x = a + 18 + (ta ? 2 * (2 * k + i) : 2 * (2 * i + k)), y = b + 18 + (tb ? 2 * (2 * j + k) : 2 * (2 * k + j));
      const ar = A[x], ai = ta ? -A[x + 1] : A[x + 1], br = B[y], bi = tb ? -B[y + 1] : B[y + 1];
      sr += ar * br - ai * bi; si += ar * bi + ai * br;
    }
    C[c + 18 + 2 * (2 * i + j)] = sr; C[c + 18 + 2 * (2 * i + j) + 1] = si;
  }
}
function setIdentity(A, a) { for (let k = 0; k < NB; k++) A[a + k] = 0; A[a] = A[a + 8] = A[a + 16] = A[a + 18] = A[a + 24] = 1; }
// exp(i H) of the Hermitian block matrix H (colour block by scaled Taylor series, weak block in closed form)
const _y = new Float64Array(18), _p = new Float64Array(18), _q = new Float64Array(18), _s = new Float64Array(18);
function mm3(A, B, C) { for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { let sr = 0, si = 0; for (let k = 0; k < 3; k++) { const x = 2 * (3 * i + k), y = 2 * (3 * k + j); sr += A[x] * B[y] - A[x + 1] * B[y + 1]; si += A[x] * B[y + 1] + A[x + 1] * B[y]; } C[2 * (3 * i + j)] = sr; C[2 * (3 * i + j) + 1] = si; } }
function expiH(H, X) {
  // colour: Y = i H / 2^s, exp(Y) = sum Y^k / k!, squared s times
  let nrm = 0; for (let k = 0; k < 18; k++) nrm += H[k] * H[k];
  nrm = Math.sqrt(nrm); let s = 0; while (nrm / 2 ** s > 0.25) s++;
  const f = 1 / 2 ** s; for (let k = 0; k < 18; k += 2) { _y[k] = -H[k + 1] * f; _y[k + 1] = H[k] * f; }
  _s.fill(0); _s[0] = _s[8] = _s[16] = 1; _p.fill(0); _p[0] = _p[8] = _p[16] = 1;
  for (let n = 1; n < 30; n++) {
    mm3(_p, _y, _q); let m = 0; for (let k = 0; k < 18; k++) { _p[k] = _q[k] / n; _s[k] += _p[k]; m = Math.max(m, Math.abs(_p[k])); }
    if (m < 1e-19) break;
  }
  for (let k = 0; k < s; k++) { mm3(_s, _s, _q); _s.set(_q); }
  for (let k = 0; k < 18; k++) X[k] = _s[k];
  // weak: H2 = h0 + h.sigma
  const h00 = H[18], h11 = H[24], h0 = (h00 + h11) / 2, hz = (h00 - h11) / 2, hx = H[20], hy = -H[21], r = Math.hypot(hx, hy, hz);
  const c = Math.cos(r), sn = r < 1e-300 ? 1 : Math.sin(r) / r, er = Math.cos(h0), ei = Math.sin(h0);
  // cos r + i sn (h.sigma): [[c + i sn hz, i sn (hx - i hy)], [i sn (hx + i hy), c - i sn hz]]
  const m = [[c, sn * hz], [sn * hy, sn * hx], [-sn * hy, sn * hx], [c, -sn * hz]];
  for (let k = 0; k < 4; k++) { const [mr, mi] = m[k]; X[18 + 2 * k] = er * mr - ei * mi; X[19 + 2 * k] = er * mi + ei * mr; }
}
const traceOf = (a, M) => { let re = 0, im = 0; for (const e of GEN[a]) { const mr = M[e.t], mi = M[e.t + 1]; re += e.re * mr - e.im * mi; im += e.re * mi + e.im * mr; } return [re, im]; };   // tr(T_a M)

export class GaugeField {
  constructor(L0, L1 = 1, L2 = 1, g2 = 1) {
    this.n = [L0, L1, L2]; const N = this.N = L0 * L1 * L2; this.g2 = g2; this.t = 0; this.started = false;
    this.U = [0, 1, 2].map(() => { const u = new Float64Array(N * NB); for (let x = 0; x < N; x++) setIdentity(u, x * NB); return u; });
    this.E = [0, 1, 2].map(() => new Float64Array(N * 12)); this.F = [0, 1, 2].map(() => new Float64Array(N * 12));
    this.fwd = [new Int32Array(N), new Int32Array(N), new Int32Array(N)]; this.bwd = [new Int32Array(N), new Int32Array(N), new Int32Array(N)];
    for (let a = 0; a < L0; a++) for (let b = 0; b < L1; b++) for (let c = 0; c < L2; c++) {
      const i = (a * L1 + b) * L2 + c;
      this.fwd[0][i] = (((a + 1) % L0) * L1 + b) * L2 + c; this.bwd[0][i] = (((a + L0 - 1) % L0) * L1 + b) * L2 + c;
      this.fwd[1][i] = (a * L1 + (b + 1) % L1) * L2 + c; this.bwd[1][i] = (a * L1 + (b + L1 - 1) % L1) * L2 + c;
      this.fwd[2][i] = (a * L1 + b) * L2 + (c + 1) % L2; this.bwd[2][i] = (a * L1 + b) * L2 + (c + L2 - 1) % L2;
    }
    this.w1 = new Float64Array(NB); this.w2 = new Float64Array(NB); this.w3 = new Float64Array(NB); this.ws = new Float64Array(NB); this.wh = new Float64Array(NB);
  }
  // F_i^a = -(2/g^2) Im tr(T_a U_i S_i), S_i the six staples of the link
  force(F = this.F) {
    const { U, N, fwd, bwd, g2, w1, w2, w3, ws } = this;
    for (let i = 0; i < 3; i++) for (let x = 0; x < N; x++) {
      ws.fill(0);
      for (let j = 0; j < 3; j++) {
        if (j === i) continue;
        const xi = fwd[i][x], xj = fwd[j][x], xmj = bwd[j][x], ximj = bwd[j][xi];
        mm(U[j], xi * NB, false, U[i], xj * NB, true, w1, 0); mm(w1, 0, false, U[j], x * NB, true, w2, 0);
        for (let k = 0; k < NB; k++) ws[k] += w2[k];
        mm(U[j], ximj * NB, true, U[i], xmj * NB, true, w1, 0); mm(w1, 0, false, U[j], xmj * NB, false, w2, 0);
        for (let k = 0; k < NB; k++) ws[k] += w2[k];
      }
      mm(U[i], x * NB, false, ws, 0, false, w3, 0);
      for (let a = 0; a < 12; a++) F[i][12 * x + a] = -(2 / g2) * traceOf(a, w3)[1];
    }
    return F;
  }
  // the half kick that puts E at t + tau/2 (once, before the first tick)
  start() { if (this.started) return; const F = this.force(); for (let i = 0; i < 3; i++) for (let k = 0; k < F[i].length; k++) this.E[i][k] += 0.5 * TAU * F[i][k]; this.started = true; }
  step() {
    const { U, E, N, wh, w1 } = this;
    for (let i = 0; i < 3; i++) for (let x = 0; x < N; x++) {
      wh.fill(0);
      for (let a = 0; a < 12; a++) { const e = TAU * E[i][12 * x + a]; if (e === 0) continue; for (const g of GEN[a]) { wh[g.o] += e * g.re; wh[g.o + 1] += e * g.im; } }
      expiH(wh, w1); mm(w1, 0, false, U[i], x * NB, false, this.w2, 0); U[i].set(this.w2, x * NB);
    }
    const F = this.force(); for (let i = 0; i < 3; i++) for (let k = 0; k < F[i].length; k++) E[i][k] += TAU * F[i][k];
  }
  tick() { this.start(); this.step(); this.step(); this.t++; }
  // a packet along axis 0 in generator a and polarization axis pol: U_pol <- exp(i eps A T_a) U_pol, E_pol^a += eps dA/dt,
  // A the positive-frequency (dir = +1, right-moving) or negative-frequency (dir = -1) part of env cos(k0 x)
  addPacket(a, pol, eps, { k0 = 0.6, width = 4, centre = null, dir = 1 } = {}) {
    const [A, Adot] = packetProfile(this.n[0], k0, width, centre === null ? this.n[0] / 4 : centre, dir), [L0, L1, L2] = this.n, H = this.wh, X = this.w1;
    for (let x0 = 0; x0 < L0; x0++) {
      H.fill(0); for (const g of GEN[a]) { H[g.o] += eps * A[x0] * g.re; H[g.o + 1] += eps * A[x0] * g.im; }
      expiH(H, X);
      for (let r = 0; r < L1 * L2; r++) { const x = x0 * L1 * L2 + r; mm(X, 0, false, this.U[pol], x * NB, false, this.w2, 0); this.U[pol].set(this.w2, x * NB); this.E[pol][12 * x + a] += eps * Adot[x0]; }
    }
  }
  // [electric, magnetic] energy; the plaquette term 1 - Re tr5(U_P)/5 is evaluated as |U_P - 1|^2 / 10 (equal for a
  // unitary U_P), which avoids the cancellation of 1 - Re tr / 5 at small amplitude
  energy() {
    const { U, E, N, fwd, g2, w1, w2, w3 } = this; let el = 0, mag = 0;
    for (let i = 0; i < 3; i++) for (let k = 0; k < E[i].length; k++) el += E[i][k] * E[i][k];
    for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) for (let x = 0; x < N; x++) {
      mm(U[i], x * NB, false, U[j], fwd[i][x] * NB, false, w1, 0); mm(w1, 0, false, U[i], fwd[j][x] * NB, true, w2, 0); mm(w2, 0, false, U[j], x * NB, true, w3, 0);
      let d = 0; for (let k = 0; k < NB; k += 2) { const re = w3[k] - (k === 0 || k === 8 || k === 16 || k === 18 || k === 24 ? 1 : 0); d += re * re + w3[k + 1] * w3[k + 1]; }
      mag += d / 10 * 10 / g2;
    }
    return [el * g2 / 2, mag];
  }
  // largest component of the Gauss constraint sum_i (E_i(x) - U_i(x - e_i)^dag E_i U_i(x - e_i)) over sites and generators
  gauss() {
    const { U, E, N, bwd, wh, w1, w2 } = this, G = new Float64Array(N * 12);
    for (let i = 0; i < 3; i++) for (let x = 0; x < N; x++) {
      for (let a = 0; a < 12; a++) G[12 * x + a] += E[i][12 * x + a];
      const y = bwd[i][x]; wh.fill(0);
      for (let a = 0; a < 12; a++) { const e = E[i][12 * y + a]; if (e === 0) continue; for (const g of GEN[a]) { wh[g.o] += e * g.re; wh[g.o + 1] += e * g.im; } }
      mm(U[i], y * NB, true, wh, 0, false, w1, 0); mm(w1, 0, false, U[i], y * NB, false, w2, 0);
      for (let a = 0; a < 12; a++) G[12 * x + a] -= 2 * traceOf(a, w2)[0];
    }
    let m = 0; for (const v of G) m = Math.max(m, Math.abs(v)); return m;
  }
  // electric energy per cell along axis 0, summed over the other axes; optionally only the listed generators
  profile(gens = null) {
    const [L0, L1, L2] = this.n, out = new Float64Array(L0), list = gens || [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    for (let i = 0; i < 3; i++) for (let x = 0; x < this.N; x++) { const a0 = Math.floor(x / (L1 * L2)); let s = 0; for (const a of list) { const e = this.E[i][12 * x + a]; s += e * e; } out[a0] += s; }
    return out;
  }
  // circular mean of the electric energy along axis 0 (GL's centroid)
  centroid() { const w = this.profile(), L = w.length; let sr = 0, si = 0; for (let x = 0; x < L; x++) { sr += w[x] * Math.cos(2 * Math.PI * x / L); si += w[x] * Math.sin(2 * Math.PI * x / L); } let a = Math.atan2(si, sr); if (a < 0) a += 2 * Math.PI; return a * L / (2 * Math.PI); }
  // largest |2 Re tr(K T_a)|, U_P = exp(i K) on the (2,3) plaquettes: the commutator field strength
  fieldStrength23(a) {
    const { U, N, fwd, w1, w2, w3 } = this; let best = 0;
    for (let x = 0; x < N; x++) {
      mm(U[1], x * NB, false, U[2], fwd[1][x] * NB, false, w1, 0); mm(w1, 0, false, U[1], fwd[2][x] * NB, true, w2, 0); mm(w2, 0, false, U[2], x * NB, true, w3, 0);
      const L = logU(w3); let re = 0; for (const e of GEN[a]) re += e.re * L[e.t + 1] + e.im * L[e.t];     // tr(T_a L) = i tr(T_a K): Re tr(K T_a) = Im tr(T_a L)
      best = Math.max(best, Math.abs(2 * re));
    }
    return best;
  }
}
// log of a unitary block matrix near 1 (series in U - 1), both blocks
function logU(P) {
  const out = new Float64Array(NB);
  for (const [o, d] of [[0, 3], [18, 2]]) {
    const X = new Float64Array(2 * d * d), T = new Float64Array(2 * d * d), Q = new Float64Array(2 * d * d);
    for (let k = 0; k < 2 * d * d; k++) X[k] = P[o + k]; for (let i = 0; i < d; i++) X[2 * (d * i + i)] -= 1;
    T.set(X);
    for (let n = 1; n < 80; n++) {
      let m = 0; for (let k = 0; k < 2 * d * d; k++) { out[o + k] += (n % 2 ? 1 : -1) * T[k] / n; m = Math.max(m, Math.abs(T[k])); }
      if (m < 1e-20) break;
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) { let sr = 0, si = 0; for (let k = 0; k < d; k++) { const x = 2 * (d * i + k), y = 2 * (d * k + j); sr += T[x] * X[y] - T[x + 1] * X[y + 1]; si += T[x] * X[y + 1] + T[x + 1] * X[y]; } Q[2 * (d * i + j)] = sr; Q[2 * (d * i + j) + 1] = si; }
      T.set(Q);
    }
  }
  return out;
}
// GL's packet: A = Re ifft(analytic part of fft(env cos(k0 x))), dA/dt = Re ifft(-i omega_k A_k) with the dispersion of
// the field step, omega = arccos(1 - 2 tau^2 sin^2(k/2)) / tau
export function packetProfile(L, k0 = 0.6, width = 4, centre = L / 4, dir = 1) {
  const ks = Array.from({ length: L }, (_, k) => 2 * Math.PI * (k < L / 2 ? k : k - L) / L);
  const a0 = Array.from({ length: L }, (_, x) => Math.exp(-((x - centre) ** 2) / (2 * width * width)) * Math.cos(k0 * x));
  const Ar = new Float64Array(L), Ai = new Float64Array(L);
  for (let k = 0; k < L; k++) { let sr = 0, si = 0; for (let x = 0; x < L; x++) { const ph = -2 * Math.PI * k * x / L; sr += a0[x] * Math.cos(ph); si += a0[x] * Math.sin(ph); } const keep = dir > 0 ? ks[k] > 0 : ks[k] < 0, f = keep ? 2 : ks[k] === 0 ? 1 : 0; Ar[k] = f * sr; Ai[k] = f * si; }
  const om = ks.map((k) => Math.acos(Math.max(-1, Math.min(1, 1 - 2 * TAU * TAU * Math.sin(k / 2) ** 2))) / TAU);
  const A = new Float64Array(L), Ad = new Float64Array(L);
  for (let x = 0; x < L; x++) { let s1 = 0, s2 = 0; for (let k = 0; k < L; k++) { const ph = 2 * Math.PI * k * x / L, c = Math.cos(ph), s = Math.sin(ph); s1 += Ar[k] * c - Ai[k] * s; s2 += om[k] * (Ai[k] * c + Ar[k] * s); } A[x] = s1 / L; Ad[x] = s2 / L; }   // Re((-i om)(Ar + i Ai) e^{i ph}) = om (Ai c + Ar s)
  return [A, Ad];
}
// GL's prediction: group velocity of the field step weighted by the electric energy |A_k|^2 omega_k^2 of the packet
export function predictedVelocity(L, k0 = 0.6, width = 4, centre = L / 4) {
  let num = 0, den = 0;
  for (let k = 1; k < L / 2; k++) {
    const q = 2 * Math.PI * k / L; let sr = 0, si = 0;
    for (let x = 0; x < L; x++) { const v = Math.exp(-((x - centre) ** 2) / (2 * width * width)) * Math.cos(k0 * x), ph = -q * x; sr += v * Math.cos(ph); si += v * Math.sin(ph); }
    const om = Math.acos(1 - 2 * TAU * TAU * Math.sin(q / 2) ** 2) / TAU, w = (sr * sr + si * si) * om * om, vg = Math.cos(q / 2) / Math.sqrt(1 - TAU * TAU * Math.sin(q / 2) ** 2);
    num += w * vg; den += w;
  }
  return num / den;
}
export function slope(ts, ys) { const n = ts.length, mt = ts.reduce((s, v) => s + v, 0) / n, my = ys.reduce((s, v) => s + v, 0) / n; let sxy = 0, sxx = 0; for (let i = 0; i < n; i++) { sxy += (ts[i] - mt) * (ys[i] - my); sxx += (ts[i] - mt) ** 2; } return sxy / sxx; }
// GL's packet run: velocity (fit over ticks 5..ticks), Gauss law before and after, relative energy drift
export function packetRun(a, pol, { L = [48, 1, 1], ticks = 24, eps = 1e-3, onTick = null } = {}) {
  const f = new GaugeField(...L); f.addPacket(a, pol, eps);
  const g0 = f.gauss(), e0 = f.energy().reduce((s, v) => s + v, 0), traj = [];
  for (let t = 1; t <= ticks; t++) { f.tick(); traj.push(f.centroid()); if (onTick) onTick(f, t); }
  const ts = traj.map((_, i) => i + 1).slice(4), v = slope(ts, traj.slice(4));
  return { velocity: v, gauss0: g0, gauss: f.gauss(), drift: Math.abs(f.energy().reduce((s, x) => s + x, 0) - e0) / e0, field: f };
}
// GL's partner test: relative change of the main packet's field after 16 ticks caused by a partner packet of equal
// amplitude (both right-moving, same place), and the commutator field strength at the start
export function partnerEffect(main, partner, eps, commDir = null, { L = [48, 1, 1], ticks = 16 } = {}) {
  const alone = new GaugeField(...L); alone.addPacket(main[0], main[1], eps); for (let t = 0; t < ticks; t++) alone.tick();
  const both = new GaugeField(...L); both.addPacket(main[0], main[1], eps); both.addPacket(partner[0], partner[1], eps);
  const fs = commDir === null ? 0 : both.fieldStrength23(commDir);
  for (let t = 0; t < ticks; t++) both.tick();
  let d = 0, n = 0; const A = alone.E[main[1]], B = both.E[main[1]];
  for (let x = 0; x < alone.N; x++) { const u = A[12 * x + main[0]], w = B[12 * x + main[0]]; d += (w - u) ** 2; n += u * u; }
  return { change: Math.sqrt(d / n), fieldStrength: fs, gauss: both.gauss() };
}
