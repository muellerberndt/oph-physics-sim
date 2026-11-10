// The observer's continuum field (Readout tab). No DOM: readout_worker.js runs it, readout_check.mjs checks it.
//
// Continuum limit of the CF-1 routing tick (repair-law/work/SIM/cf1/continuum_check.mjs): on a plane wave e^{ip.x},
// two ticks in the half-step frame act as exp(-2i (alpha.p + theta beta)) up to O(p^3), with the Dirac matrices
//   alpha_i = X_record (x) sigma_i,   beta = Y_record (x) 1.
// For wavelengths long against one cell the quantum therefore obeys the free Dirac equation
//   i d(psi)/dt = (alpha.p + m beta) psi,   c = one cell per tick,   m = theta per tick.
// Here that equation is solved exactly in a periodic box of n^3 points, one point per h cells: momenta in units of
// 1/(h cells), the mass m = theta h, the time in units of h ticks. Each particle is its own field; the expected
// density of the joint state is the sum of the particle densities. The state keeps the particle branch (projector
// (1 + H/E)/2), also after a record. Spinor component k = 2 * record + spin.
import { makeFFT3 } from './fft3.js';

export class ContinuumField {
  constructor(n = 32) {
    this.n = n; const N = this.N = n ** 3; this.fft = makeFFT3(n);
    const K = new Float64Array(n); for (let j = 0; j < n; j++) K[j] = 2 * Math.PI * (j < n / 2 ? j : j - n) / n;
    this.kx = new Float64Array(N); this.ky = new Float64Array(N); this.kz = new Float64Array(N); this.k2 = new Float64Array(N);
    for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const i = (z * n + y) * n + x; this.kx[i] = K[x]; this.ky[i] = K[y]; this.kz[i] = K[z]; this.k2[i] = K[x] ** 2 + K[y] ** 2 + K[z] ** 2;
    }
    this.wr = [0, 1, 2, 3].map(() => new Float64Array(N)); this.wi = [0, 1, 2, 3].map(() => new Float64Array(N));
    this.cosE = new Float64Array(N); this.sinE = new Float64Array(N);
    this.rho = new Float64Array(N); this.acc = new Float32Array(3 * N);
    this.parts = new Map();
  }
  // H psi at one momentum: (H psi)_record0 = (sigma.p - i m) psi_record1, (H psi)_record1 = (sigma.p + i m) psi_record0,
  // sigma.p (u, v) = (pz u + (px - i py) v, (px + i py) u - pz v). v, out: 8 numbers (re, im of the four components).
  static applyH(px, py, pz, m, v, out) {
    const [a0r, a0i, a1r, a1i, b0r, b0i, b1r, b1i] = v;
    out[0] = pz * b0r + px * b1r + py * b1i + m * b0i; out[1] = pz * b0i + px * b1i - py * b1r - m * b0r;
    out[2] = px * b0r - py * b0i - pz * b1r + m * b1i; out[3] = px * b0i + py * b0r - pz * b1i - m * b1r;
    out[4] = pz * a0r + px * a1r + py * a1i - m * a0i; out[5] = pz * a0i + px * a1i - py * a1r + m * a0r;
    out[6] = px * a0r - py * a0i - pz * a1r - m * a1i; out[7] = px * a0i + py * a0r - pz * a1i + m * a1r;
    return out;
  }
  // keep the particle branch: v <- (v + H v / E) / 2 at every momentum
  projectPlus(sr, si, m) {
    const { kx, ky, kz, k2, N } = this, v = new Float64Array(8), h = new Float64Array(8);
    for (let i = 0; i < N; i++) {
      const E = Math.sqrt(k2[i] + m * m); if (E < 1e-12) continue;
      for (let k = 0; k < 4; k++) { v[2 * k] = sr[k][i]; v[2 * k + 1] = si[k][i]; }
      ContinuumField.applyH(kx[i], ky[i], kz[i], m, v, h);
      const f = 0.5 / E;
      for (let k = 0; k < 4; k++) { sr[k][i] = 0.5 * v[2 * k] + f * h[2 * k]; si[k][i] = 0.5 * v[2 * k + 1] + f * h[2 * k + 1]; }
    }
  }
  normalizeSpectral(sr, si) {          // position-space norm one: sum over modes of |psi^|^2 = N
    let s = 0; for (let k = 0; k < 4; k++) for (let i = 0; i < this.N; i++) s += sr[k][i] ** 2 + si[k][i] ** 2;
    const f = Math.sqrt(this.N / s); for (let k = 0; k < 4; k++) for (let i = 0; i < this.N; i++) { sr[k][i] *= f; si[k][i] *= f; }
  }
  disp(u, c) { const n = this.n; let v = u - c; v -= n * Math.round(v / n); return v; }
  // a particle: Gaussian of rms width sigma per axis around x0 with mean momentum p0 and spinor chi ([re, im] x 4)
  add({ id, m, x0, p0, sigma, chi, t, color }) {
    const { n, N } = this, sr = [0, 1, 2, 3].map(() => new Float64Array(N)), si = [0, 1, 2, 3].map(() => new Float64Array(N));
    for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const dx = this.disp(x, x0[0]), dy = this.disp(y, x0[1]), dz = this.disp(z, x0[2]);
      const g = Math.exp(-(dx * dx + dy * dy + dz * dz) / (4 * sigma * sigma)); if (g < 1e-9) continue;
      const ph = p0[0] * dx + p0[1] * dy + p0[2] * dz, cr = g * Math.cos(ph), ci = g * Math.sin(ph), i = (z * n + y) * n + x;
      for (let k = 0; k < 4; k++) { sr[k][i] = cr * chi[k][0] - ci * chi[k][1]; si[k][i] = cr * chi[k][1] + ci * chi[k][0]; }
    }
    for (let k = 0; k < 4; k++) this.fft(sr[k], si[k], -1);
    this.projectPlus(sr, si, m); this.normalizeSpectral(sr, si);
    const E = new Float64Array(N); for (let i = 0; i < N; i++) E[i] = Math.sqrt(this.k2[i] + m * m);
    this.parts.set(id, { id, m, sr, si, E, tb: t, ref: 0, color: color || [1, 1, 1] });
  }
  // the spinor field of one particle at time t, in position space (this.wr, this.wi)
  evolve(P, t) {
    const { N, cosE, sinE, wr, wi } = this, dt = t - P.tb;
    for (let i = 0; i < N; i++) { const a = P.E[i] * dt; cosE[i] = Math.cos(a); sinE[i] = Math.sin(a); }
    for (let k = 0; k < 4; k++) {
      const sr = P.sr[k], si = P.si[k], r = wr[k], im = wi[k];
      for (let i = 0; i < N; i++) { r[i] = sr[i] * cosE[i] + si[i] * sinE[i]; im[i] = si[i] * cosE[i] - sr[i] * sinE[i]; }   // e^{-iE dt}
      this.fft(r, im, 1);
    }
  }
  // readouts of the position-space field: density, mean position (circular, per axis), spin <Sigma>, chirality <X_record>
  readouts() {
    const { n, N, wr, wi, rho } = this; let tot = 0, peak = 0;
    const cs = [0, 0, 0, 0, 0, 0], sp = [0, 0, 0]; let chi = 0;
    for (let i = 0; i < N; i++) {
      const a0r = wr[0][i], a0i = wi[0][i], a1r = wr[1][i], a1i = wi[1][i], b0r = wr[2][i], b0i = wi[2][i], b1r = wr[3][i], b1i = wi[3][i];
      const d = a0r * a0r + a0i * a0i + a1r * a1r + a1i * a1i + b0r * b0r + b0i * b0i + b1r * b1r + b1i * b1i;
      rho[i] = d; tot += d; if (d > peak) peak = d;
      sp[0] += 2 * (a0r * a1r + a0i * a1i + b0r * b1r + b0i * b1i);                 // 2 Re(psi*_s0 psi_s1), both records
      sp[1] += 2 * (a0r * a1i - a0i * a1r + b0r * b1i - b0i * b1r);                 // 2 Im(psi*_s0 psi_s1)
      sp[2] += a0r * a0r + a0i * a0i + b0r * b0r + b0i * b0i - (a1r * a1r + a1i * a1i + b1r * b1r + b1i * b1i);
      chi += 2 * (a0r * b0r + a0i * b0i + a1r * b1r + a1i * b1i);                   // 2 Re(psi*_record0 psi_record1)
      const x = i % n, y = ((i / n) | 0) % n, z = (i / (n * n)) | 0, w = 2 * Math.PI / n;
      cs[0] += d * Math.cos(w * x); cs[1] += d * Math.sin(w * x); cs[2] += d * Math.cos(w * y); cs[3] += d * Math.sin(w * y); cs[4] += d * Math.cos(w * z); cs[5] += d * Math.sin(w * z);
    }
    const ang = (c, s) => ((Math.atan2(s, c) / (2 * Math.PI)) * n + n) % n;
    return { norm: tot, peak, mean: [ang(cs[0], cs[1]), ang(cs[2], cs[3]), ang(cs[4], cs[5])], spin: sp.map((v) => v / tot), chirality: chi / tot };
  }
  // The former density-sampled Gaussian reset omitted input-dependent
  // projection outcomes. Require the complete Born instrument explicitly.
  record() {
    throw new Error('Use BornReadoutField from readout_instrument.js for complete detector records');
  }
  // one frame at field time t: RGBA volume (x fastest), readouts per particle, records for the ids in recordIds
  frame(t, recordIds = new Set(), sigmaD = 1, rand = Math.random) {
    const { N, rho, acc } = this; acc.fill(0);
    const stats = [], records = [];
    for (const P of this.parts.values()) {
      this.evolve(P, t);
      const r = this.readouts(); if (!P.ref) P.ref = r.peak;
      stats.push({ id: P.id, mean: r.mean, spin: r.spin, chirality: r.chirality, norm: r.norm, peak: r.peak / P.ref });
      const inv = 1 / P.ref, [cr, cg, cb] = P.color;
      for (let i = 0; i < N; i++) {
        const u = rho[i] * inv; if (u < 2e-3) continue;
        const v = Math.pow(Math.min(1.8, u), 0.7); acc[3 * i] += cr * v; acc[3 * i + 1] += cg * v; acc[3 * i + 2] += cb * v;
      }
      if (recordIds.has(P.id)) records.push({ id: P.id, t, ...this.record(P, t, sigmaD, rand()) });
    }
    const rgba = new Uint8Array(4 * N);
    for (let i = 0; i < N; i++) {
      const r = Math.min(1, acc[3 * i]), g = Math.min(1, acc[3 * i + 1]), b = Math.min(1, acc[3 * i + 2]);
      rgba[4 * i] = 255 * r; rgba[4 * i + 1] = 255 * g; rgba[4 * i + 2] = 255 * b; rgba[4 * i + 3] = 255 * Math.max(r, g, b);
    }
    return { rgba, stats, records };
  }
}
