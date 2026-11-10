// Executable version of the routing tick of the candidate program CF-1 (repair-law/candidates/CF1.md).
// Reference engine: repair-law/work/DIRAC/routing.py (theory session). Checks: node cf1_check.mjs.
//
// State of one quantum of one species: psi[cell, record, spin]; record 0 is blank; spin in the basis of the cubic frame.
// One tick:  mass(theta) . (Q_3 R)(Q_2 R)(Q_1 R), the order of the three axes alternating from tick to tick.
//   R    reflection about the blank record
//   Q_i  seam involution of the public type between the carriers x and x + e_i, oriented by the spin along axis i
//   Q_i R = P+ T_i + P- T_i*, with alpha_i = X_record sigma_i   (shift() uses this form, shiftPublic() the other)
//   mass exp(-i theta Y_record) (option M1) or exp(-i theta Z_record)
// Component index k = 2 * record + spin.
const SQ = Math.SQRT1_2;
// columns of the spin eigenbases: V[i][comp] = [[re, im] of spin up, [re, im] of spin down], comp 0 is spin +1 along axis i
const V = [
  [[[SQ, 0], [SQ, 0]], [[SQ, 0], [-SQ, 0]]],
  [[[SQ, 0], [0, SQ]], [[SQ, 0], [0, -SQ]]],
  [[[1, 0], [0, 0]], [[0, 0], [1, 0]]],
];

export class Routing {
  constructor(n1, n2 = 1, n3 = 1) {
    this.n = [n1, n2, n3]; const N = this.N = n1 * n2 * n3;
    this.re = new Float64Array(4 * N); this.im = new Float64Array(4 * N);
    this.xr = new Float64Array(4 * N); this.xi = new Float64Array(4 * N);
    this.or = new Float64Array(4 * N); this.oi = new Float64Array(4 * N);
    this.fwd = [new Int32Array(N), new Int32Array(N), new Int32Array(N)];
    this.bwd = [new Int32Array(N), new Int32Array(N), new Int32Array(N)];
    for (let a = 0; a < n1; a++) for (let b = 0; b < n2; b++) for (let c = 0; c < n3; c++) {
      const i = (a * n2 + b) * n3 + c;
      this.fwd[0][i] = (((a + 1) % n1) * n2 + b) * n3 + c; this.bwd[0][i] = (((a + n1 - 1) % n1) * n2 + b) * n3 + c;
      this.fwd[1][i] = (a * n2 + (b + 1) % n2) * n3 + c; this.bwd[1][i] = (a * n2 + (b + n2 - 1) % n2) * n3 + c;
      this.fwd[2][i] = (a * n2 + b) * n3 + (c + 1) % n3; this.bwd[2][i] = (a * n2 + b) * n3 + (c + n3 - 1) % n3;
    }
    this.t = 0;
    this.linksIm = null;        // imaginary parts when the links carry phases instead of signs (same layout)
    this.links = null;          // static link signs of the lattice gauge field: links[i][x] on the link from x to x + e_i (charged species only)
  }
  cell(a, b, c) { const [n1, n2, n3] = this.n; return ((((a % n1) + n1) % n1) * n2 + (((b % n2) + n2) % n2)) * n3 + (((c % n3) + n3) % n3); }
  clear() { this.re.fill(0); this.im.fill(0); this.t = 0; }
  copyFrom(o) { this.re.set(o.re); this.im.set(o.im); this.t = o.t; }
  norm2() { let s = 0; const { re, im } = this; for (let j = 0; j < re.length; j++) s += re[j] * re[j] + im[j] * im[j]; return s; }
  normalize() { const f = 1 / Math.sqrt(this.norm2()); const { re, im } = this; for (let j = 0; j < re.length; j++) { re[j] *= f; im[j] *= f; } }

  // ---- the tick -------------------------------------------------------------------------------------------------
  // alpha_i psi into (xr, xi)
  _alpha(i) {
    const { re, im, xr, xi, N } = this;
    for (let c = 0, o = 0; c < N; c++, o += 4) {
      if (i === 0) { xr[o] = re[o + 3]; xi[o] = im[o + 3]; xr[o + 1] = re[o + 2]; xi[o + 1] = im[o + 2]; xr[o + 2] = re[o + 1]; xi[o + 2] = im[o + 1]; xr[o + 3] = re[o]; xi[o + 3] = im[o]; }
      else if (i === 1) { xr[o] = im[o + 3]; xi[o] = -re[o + 3]; xr[o + 1] = -im[o + 2]; xi[o + 1] = re[o + 2]; xr[o + 2] = im[o + 1]; xi[o + 2] = -re[o + 1]; xr[o + 3] = -im[o]; xi[o + 3] = re[o]; }
      else { xr[o] = re[o + 2]; xi[o] = im[o + 2]; xr[o + 1] = -re[o + 3]; xi[o + 1] = -im[o + 3]; xr[o + 2] = re[o]; xi[o + 2] = im[o]; xr[o + 3] = -re[o + 1]; xi[o + 3] = -im[o + 1]; }
    }
  }
  // S_i = P+ T_i + P- T_i*, the translations carrying the link variable they cross: the amplitude that arrives from
  // x + e_i carries z(x), the one from x - e_i carries conj(z(x - e_i))
  shift(i) {
    this._alpha(i);
    const { re, im, xr, xi, or, oi, N } = this, f = this.fwd[i], b = this.bwd[i], z = this.links ? this.links[i] : null, zi = this.linksIm ? this.linksIm[i] : null;
    for (let c = 0; c < N; c++) {
      const o = 4 * c, m = 4 * b[c], p = 4 * f[c], zm = z ? 0.5 * z[b[c]] : 0.5, zp = z ? 0.5 * z[c] : 0.5;
      if (!zi) {
        for (let k = 0; k < 4; k++) {
          or[o + k] = zm * (re[m + k] + xr[m + k]) + zp * (re[p + k] - xr[p + k]);
          oi[o + k] = zm * (im[m + k] + xi[m + k]) + zp * (im[p + k] - xi[p + k]);
        }
      } else {
        const wm = 0.5 * zi[b[c]], wp = 0.5 * zi[c];
        for (let k = 0; k < 4; k++) {
          const ar = re[m + k] + xr[m + k], ai = im[m + k] + xi[m + k], br = re[p + k] - xr[p + k], bi = im[p + k] - xi[p + k];
          or[o + k] = zm * ar + wm * ai + zp * br - wp * bi;
          oi[o + k] = zm * ai - wm * ar + zp * bi + wp * br;
        }
      }
    }
    this.re = or; this.im = oi; this.or = re; this.oi = im;
  }
  recordReflection() { const { re, im, N } = this; for (let c = 0, o = 0; c < N; c++, o += 4) { re[o + 2] = -re[o + 2]; im[o + 2] = -im[o + 2]; re[o + 3] = -re[o + 3]; im[o + 3] = -im[o + 3]; } }
  // Q_i: the public reflector v = (a0 - b0 - a1 - b1)/sqrt2 across every link of axis i; (a, b) = (x, x + e_i) on the
  // component of spin +1 along the axis and (x + e_i, x) on the component of spin -1
  seamInvolution(i) {
    const { re, im, N } = this, f = this.fwd[i];
    // components along the two spin eigenvectors: phi[comp][cell][record]
    const pr = [new Float64Array(2 * N), new Float64Array(2 * N)], pi = [new Float64Array(2 * N), new Float64Array(2 * N)];
    for (let comp = 0; comp < 2; comp++) for (let c = 0; c < N; c++) for (let r = 0; r < 2; r++) {
      let sr = 0, si = 0;
      for (let s = 0; s < 2; s++) { const [vr, vi] = V[i][comp][s], a = re[4 * c + 2 * r + s], b = im[4 * c + 2 * r + s]; sr += vr * a + vi * b; si += vr * b - vi * a; }   // conj(V) psi
      pr[comp][2 * c + r] = sr; pi[comp][2 * c + r] = si;
    }
    const qr = [Float64Array.from(pr[0]), Float64Array.from(pr[1])], qi = [Float64Array.from(pi[0]), Float64Array.from(pi[1])];
    const z = this.links ? this.links[i] : null;
    for (let comp = 0; comp < 2; comp++) for (let c = 0; c < N; c++) {
      // the amplitude at d = x + e_i is transported to x by the link sign before the reflector acts, and back after it
      const d = f[c], zz = z ? z[c] : 1, a = comp === 0 ? c : d, b = comp === 0 ? d : c, za = comp === 0 ? 1 : zz, zb = comp === 0 ? zz : 1;
      const cr = (za * pr[comp][2 * a] - zb * pr[comp][2 * b] - za * pr[comp][2 * a + 1] - zb * pr[comp][2 * b + 1]) / 2;
      const ci = (za * pi[comp][2 * a] - zb * pi[comp][2 * b] - za * pi[comp][2 * a + 1] - zb * pi[comp][2 * b + 1]) / 2;
      qr[comp][2 * a] -= za * cr; qi[comp][2 * a] -= za * ci; qr[comp][2 * a + 1] += za * cr; qi[comp][2 * a + 1] += za * ci;
      qr[comp][2 * b] += zb * cr; qi[comp][2 * b] += zb * ci; qr[comp][2 * b + 1] += zb * cr; qi[comp][2 * b + 1] += zb * ci;
    }
    for (let c = 0; c < N; c++) for (let r = 0; r < 2; r++) for (let s = 0; s < 2; s++) {
      let sr = 0, si = 0;
      for (let comp = 0; comp < 2; comp++) { const [vr, vi] = V[i][comp][s], a = qr[comp][2 * c + r], b = qi[comp][2 * c + r]; sr += vr * a - vi * b; si += vr * b + vi * a; }
      re[4 * c + 2 * r + s] = sr; im[4 * c + 2 * r + s] = si;
    }
  }
  shiftPublic(i) { this.recordReflection(); this.seamInvolution(i); }
  // theta: a number, or one angle per cell
  mass(theta, kind = 'Y') {
    const { re, im, N } = this, field = typeof theta !== 'number';
    if (!field && theta === 0) return;
    let c0 = field ? 1 : Math.cos(theta), s0 = field ? 0 : Math.sin(theta);
    for (let c = 0, o = 0; c < N; c++, o += 4) {
      if (field) { const th = theta[c]; if (th === 0) continue; c0 = Math.cos(th); s0 = Math.sin(th); }
      for (let s = 0; s < 2; s++) {
        const ar = re[o + s], ai = im[o + s], br = re[o + 2 + s], bi = im[o + 2 + s];
        if (kind === 'Y') { re[o + s] = c0 * ar - s0 * br; im[o + s] = c0 * ai - s0 * bi; re[o + 2 + s] = s0 * ar + c0 * br; im[o + 2 + s] = s0 * ai + c0 * bi; }
        else { re[o + s] = c0 * ar + s0 * ai; im[o + s] = c0 * ai - s0 * ar; re[o + 2 + s] = c0 * br - s0 * bi; im[o + 2 + s] = c0 * bi + s0 * br; }
      }
    }
  }
  // one tick; reverse: axes in the order 3, 2, 1. tickAlt alternates the order from tick to tick (the executable version)
  tick(theta = 0, kind = 'Y', reverse = false) {
    if (reverse) { this.shift(2); this.shift(1); this.shift(0); } else { this.shift(0); this.shift(1); this.shift(2); }
    this.mass(theta, kind); this.t++;
  }
  tickAlt(theta = 0, kind = 'Y') { this.tick(theta, kind, (this.t & 1) === 1); }
  // the tick in the half-step frame M^(1/2) S M^(1/2): half the mass rotation before the three shifts, half after. Same
  // phases as tickAlt; record statistics and the rest position of a packet are exact only here. half: theta / 2 (number or field)
  tickAltHalf(half = 0, kind = 'Y') {
    const reverse = (this.t & 1) === 1; this.mass(half, kind);
    if (reverse) { this.shift(2); this.shift(1); this.shift(0); } else { this.shift(0); this.shift(1); this.shift(2); }
    this.mass(half, kind); this.t++;
  }

  // ---- preparations ---------------------------------------------------------------------------------------------
  momentum(m) { return [0, 1, 2].map((i) => 2 * Math.PI * m[i] / this.n[i]); }
  // coin: four complex amplitudes [[re, im], ...] in the order (record, spin) = (0,up), (0,down), (1,up), (1,down)
  addPlaneWave(p, coin, weight = 1) {
    const [n1, n2, n3] = this.n, { re, im } = this;
    for (let a = 0; a < n1; a++) for (let b = 0; b < n2; b++) for (let c = 0; c < n3; c++) {
      const ph = p[0] * a + p[1] * b + p[2] * c, cr = Math.cos(ph) * weight, ci = Math.sin(ph) * weight, o = 4 * ((a * n2 + b) * n3 + c);
      for (let k = 0; k < 4; k++) { re[o + k] += cr * coin[k][0] - ci * coin[k][1]; im[o + k] += cr * coin[k][1] + ci * coin[k][0]; }
    }
  }
  // Gaussian packet around a cell with a mean wave vector and one coin
  addPacket(centre, width, p, coin, weight = 1) {
    const [n1, n2, n3] = this.n, { re, im } = this, w2 = 2 * width * width;
    const d = (x, c, n) => { let u = x - c; if (n > 1) { u = ((u % n) + n) % n; if (u > n / 2) u -= n; } return u; };
    for (let a = 0; a < n1; a++) for (let b = 0; b < n2; b++) for (let c = 0; c < n3; c++) {
      const da = d(a, centre[0], n1), db = n2 > 1 ? d(b, centre[1], n2) : 0, dc = n3 > 1 ? d(c, centre[2], n3) : 0;
      const g = Math.exp(-(da * da + db * db + dc * dc) / w2) * weight; if (g < 1e-12) continue;
      const ph = p[0] * da + p[1] * db + p[2] * dc, cr = Math.cos(ph) * g, ci = Math.sin(ph) * g, o = 4 * ((a * n2 + b) * n3 + c);
      for (let k = 0; k < 4; k++) { re[o + k] += cr * coin[k][0] - ci * coin[k][1]; im[o + k] += cr * coin[k][1] + ci * coin[k][0]; }
    }
  }
  // packet along axis 0 built from plane waves, each with the coin of the positive branch at its own momentum
  packetOnBranch(m0, width, theta, kind = 'Y') {
    const L = this.n[0]; this.clear();
    for (let m = 0; m < L; m++) {
      let d = ((m - m0 + L / 2) % L + L) % L - L / 2; const w = Math.exp(-d * d / (2 * width * width)); if (w < 1e-12) continue;
      const p = this.momentum([m, 0, 0]); this.addPlaneWave(p, branchCoin(p, theta, kind), w);
    }
    this.normalize();
  }

  // ---- readouts -------------------------------------------------------------------------------------------------
  recordProbability() { let s = 0; const { re, im, N } = this; for (let c = 0, o = 0; c < N; c++, o += 4) s += re[o + 2] ** 2 + im[o + 2] ** 2 + re[o + 3] ** 2 + im[o + 3] ** 2; return s / this.norm2(); }
  density(out) { const { re, im, N } = this; for (let c = 0, o = 0; c < N; c++, o += 4) { let s = 0; for (let k = 0; k < 4; k++) s += re[o + k] ** 2 + im[o + k] ** 2; out[c] = s; } return out; }
  // circular mean of the position along one axis, in cells
  meanPosition(axis = 0) {
    const [n1, n2, n3] = this.n, L = this.n[axis], { re, im } = this; let sr = 0, si = 0;
    for (let a = 0; a < n1; a++) for (let b = 0; b < n2; b++) for (let c = 0; c < n3; c++) {
      const o = 4 * ((a * n2 + b) * n3 + c); let w = 0; for (let k = 0; k < 4; k++) w += re[o + k] ** 2 + im[o + k] ** 2;
      const x = [a, b, c][axis], ph = 2 * Math.PI * x / L; sr += w * Math.cos(ph); si += w * Math.sin(ph);
    }
    let ang = Math.atan2(si, sr); if (ang < 0) ang += 2 * Math.PI; return ang * L / (2 * Math.PI);
  }
  // probability weighted mean distance from a cell, and the probability inside a radius
  radial(centre, radius = 0) {
    const [n1, n2, n3] = this.n, { re, im } = this; let s = 0, sr = 0, inside = 0;
    const d = (x, c, n) => { let u = ((x - c) % n + n) % n; if (u > n / 2) u -= n; return u; };
    for (let a = 0; a < n1; a++) for (let b = 0; b < n2; b++) for (let c = 0; c < n3; c++) {
      const o = 4 * ((a * n2 + b) * n3 + c); let w = 0; for (let k = 0; k < 4; k++) w += re[o + k] ** 2 + im[o + k] ** 2;
      const r = Math.hypot(d(a, centre[0], n1), d(b, centre[1], n2), d(c, centre[2], n3)); s += w; sr += w * r; if (r <= radius) inside += w;
    }
    return { mean: sr / s, inside: inside / s };
  }
  overlap(ore, oim) { let sr = 0, si = 0; const { re, im } = this; for (let j = 0; j < re.length; j++) { sr += ore[j] * re[j] + oim[j] * im[j]; si += ore[j] * im[j] - oim[j] * re[j]; } return [sr, si]; }
}

// ---- the symbol of the tick at a wave vector (4 x 4, row major, [re, im] arrays) ----------------------------------
const ALPHA = (() => {
  const mats = []; const sig = [[[0, 0], [1, 0], [1, 0], [0, 0]], [[0, 0], [0, -1], [0, 1], [0, 0]], [[1, 0], [0, 0], [0, 0], [-1, 0]]];
  for (let i = 0; i < 3; i++) {       // X_record (x) sigma_i : element [(r,s),(r',s')] = [r != r'] sigma_i[s][s']
    const re = new Float64Array(16), im = new Float64Array(16);
    for (let r = 0; r < 2; r++) for (let s = 0; s < 2; s++) for (let s2 = 0; s2 < 2; s2++) { const row = 2 * r + s, col = 2 * (1 - r) + s2; re[4 * row + col] = sig[i][2 * s + s2][0]; im[4 * row + col] = sig[i][2 * s + s2][1]; }
    mats.push({ re, im });
  }
  return mats;
})();
function mul(A, B) { const re = new Float64Array(16), im = new Float64Array(16); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let sr = 0, si = 0; for (let k = 0; k < 4; k++) { sr += A.re[4 * i + k] * B.re[4 * k + j] - A.im[4 * i + k] * B.im[4 * k + j]; si += A.re[4 * i + k] * B.im[4 * k + j] + A.im[4 * i + k] * B.re[4 * k + j]; } re[4 * i + j] = sr; im[4 * i + j] = si; } return { re, im }; }
function eye() { const re = new Float64Array(16), im = new Float64Array(16); for (let i = 0; i < 4; i++) re[5 * i] = 1; return { re, im }; }
export function symbol(p, theta = 0, kind = 'Y', reverse = false) {
  let U = eye();
  for (const i of (reverse ? [2, 1, 0] : [0, 1, 2])) {
    const c = Math.cos(p[i]), s = Math.sin(p[i]), F = eye();
    for (let j = 0; j < 16; j++) { F.re[j] = c * F.re[j] + s * ALPHA[i].im[j]; F.im[j] = -s * ALPHA[i].re[j]; }      // cos p - i sin p alpha_i
    U = mul(F, U);
  }
  const c = Math.cos(theta), s = Math.sin(theta), M = eye();
  if (kind === 'Y') { for (let sp = 0; sp < 2; sp++) { M.re[4 * sp + sp] = c; M.re[4 * (2 + sp) + 2 + sp] = c; M.re[4 * sp + 2 + sp] = -s; M.re[4 * (2 + sp) + sp] = s; } }
  else { for (let sp = 0; sp < 2; sp++) { M.re[4 * sp + sp] = c; M.im[4 * sp + sp] = -s; M.re[4 * (2 + sp) + 2 + sp] = c; M.im[4 * (2 + sp) + 2 + sp] = s; } }
  return mul(M, U);
}
// the two positive phases of a 4 x 4 unitary whose eigenvalues are exp(+-i E1), exp(+-i E2)
function pairPhases(U) {
  let t1 = 0, t2 = 0; const U2 = mul(U, U);
  for (let i = 0; i < 4; i++) { t1 += U.re[5 * i]; t2 += U2.re[5 * i]; }
  t1 /= 2; t2 /= 2;
  const d2 = t2 + 2 - t1 * t1, disc = d2 < 1e-11 ? 0 : Math.sqrt(d2), clamp = (x) => Math.max(-1, Math.min(1, x));
  return [Math.acos(clamp((t1 + disc) / 2)), Math.acos(clamp((t1 - disc) / 2))];
}
// phases per tick at wave vector p: [E1, E2] with E1 <= E2 (the two helicities). alternating: the executable order.
export function phases(p, theta = 0, kind = 'Y', alternating = true) {
  if (!alternating) return pairPhases(symbol(p, theta, kind));
  const two = pairPhases(mul(symbol(p, theta, kind, true), symbol(p, theta, kind, false)));
  return two.map((e) => e / 2);
}
// coin of the branch with phase exp(-i E), E > 0, for a wave vector along axis 0 (cos E = cos theta cos p):
// the projector (U - e^{iE}) / (e^{-iE} - e^{iE}) applied to the blank spin-up state
export function branchCoin(p, theta, kind = 'Y') {
  const U = symbol(p, theta, kind), E = Math.acos(Math.cos(theta) * Math.cos(p[0]));
  if (E < 1e-12) return [[1, 0], [0, 0], [0, 0], [0, 0]];
  const s = Math.sin(E), res = []; let n = 0;
  for (let k = 0; k < 4; k++) {
    let ar = U.re[4 * k], ai = U.im[4 * k]; if (k === 0) { ar -= Math.cos(E); ai -= Math.sin(E); }
    res.push([-ai / (2 * s), ar / (2 * s)]); n += (ai * ai + ar * ar) / (4 * s * s);       // z / (-2 i s) = (-Im z + i Re z) / (2 s)
  }
  n = Math.sqrt(n); return res.map(([a, b]) => [a / n, b / n]);
}
