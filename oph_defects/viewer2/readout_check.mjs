// Checks of the Readout tab's continuum field (readout_field.js). Run: node readout_check.mjs
//  1  H(p)^2 = (p^2 + m^2) 1 (Dirac algebra of alpha_i = X_record sigma_i, beta = Y_record)
//  2  the kept branch: H psi^ = E psi^ at every momentum after the projector
//  3  unitarity: the norm stays one in time
//  4  a packet moves at the group velocity p/E of the Dirac equation
//  5  the box agrees with the executable routing tick (cf1_engine.js) for a long-wavelength packet: same spinor field
//     after 24 ticks of the lattice walk and 24 time units of the continuum equation (h = 1 cell per grid point)
//  6  the complete instrument retains a positive joint record probability and normalized output
import { ContinuumField } from './readout_field.js';
import { BornReadoutField } from './readout_instrument.js';
import { Routing } from './cf1_engine.js';

let fails = 0;
const ok = (name, cond, info) => { console.log(`${cond ? 'pass' : 'FAIL'}  ${name}  ${info}`); if (!cond) fails++; };
let seed = 12345; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

// 1
{
  let worst = 0; const v = new Float64Array(8), h = new Float64Array(8), h2 = new Float64Array(8);
  for (let t = 0; t < 200; t++) {
    const p = [rnd() * 4 - 2, rnd() * 4 - 2, rnd() * 4 - 2], m = rnd() * 2;
    for (let k = 0; k < 8; k++) v[k] = rnd() - 0.5;
    ContinuumField.applyH(p[0], p[1], p[2], m, v, h); ContinuumField.applyH(p[0], p[1], p[2], m, h, h2);
    const E2 = p[0] ** 2 + p[1] ** 2 + p[2] ** 2 + m * m;
    for (let k = 0; k < 8; k++) worst = Math.max(worst, Math.abs(h2[k] - E2 * v[k]));
  }
  ok('H^2 = p^2 + m^2', worst < 1e-12, `max deviation ${worst.toExponential(2)}`);
}
// 2, 3, 4
{
  const n = 32, F = new ContinuumField(n), m = 0.56, p0 = [0.6, 0, 0], chi = [[0.5, 0], [0.5, 0], [-0.5, 0], [-0.5, 0]];
  F.add({ id: 1, m, x0: [8, 16, 16], p0, sigma: 3, chi, t: 0 });
  const P = F.parts.get(1); let worst = 0; const v = new Float64Array(8), h = new Float64Array(8);
  for (let i = 0; i < F.N; i += 7) {
    for (let k = 0; k < 4; k++) { v[2 * k] = P.sr[k][i]; v[2 * k + 1] = P.si[k][i]; }
    ContinuumField.applyH(F.kx[i], F.ky[i], F.kz[i], m, v, h);
    for (let k = 0; k < 8; k++) worst = Math.max(worst, Math.abs(h[k] - P.E[i] * v[k]));
  }
  ok('particle branch: H psi = E psi', worst < 1e-10, `max deviation ${worst.toExponential(2)}`);
  const xs = [], norms = [];
  for (const t of [0, 4, 8, 12]) { F.evolve(P, t); const r = F.readouts(); xs.push(r.mean[0]); norms.push(r.norm); }
  ok('unitary in time', norms.every((x) => Math.abs(x - 1) < 1e-10), `norms ${norms.map((x) => x.toFixed(12)).join(', ')}`);
  // for a particle-branch state d<x>/dt = <p/E>, the group velocity averaged over its own momentum distribution
  let wsum = 0, vsum = 0;
  for (let i = 0; i < F.N; i++) { let w = 0; for (let k = 0; k < 4; k++) w += P.sr[k][i] ** 2 + P.si[k][i] ** 2; wsum += w; vsum += w * F.kx[i] / P.E[i]; }
  const v0 = vsum / wsum, vm = (xs[3] - xs[0]) / 12;
  ok('packet moves at <p/E>', Math.abs(vm - v0) < 0.01, `measured ${vm.toFixed(4)} per unit time, <p/E> ${v0.toFixed(4)} (p/E at the mean momentum ${(0.6 / Math.sqrt(0.36 + m * m)).toFixed(4)})`);
}
// 5
{
  const n = 64, F = new ContinuumField(n), theta = 0.05, p0 = [0.12, 0.05, 0], chi = [[0.5, 0], [0.5, 0], [0.5, 0], [0.5, 0]];
  F.add({ id: 1, m: theta, x0: [32, 32, 32], p0, sigma: 5, chi, t: 0 });
  const P = F.parts.get(1); F.evolve(P, 0);
  const W = new Routing(n, n, n);
  // the walk's cell (a, b, c) is (x, y, z) here: a runs slowest in the engine, x fastest in the box
  const idxW = (x, y, z) => (x * n + y) * n + z, idxF = (x, y, z) => (z * n + y) * n + x;
  for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = idxF(x, y, z), o = 4 * idxW(x, y, z);
    for (let k = 0; k < 4; k++) { W.re[o + k] = F.wr[k][i]; W.im[o + k] = F.wi[k][i]; }
  }
  const T = 24;
  for (let t = 0; t < T; t++) W.tickAltHalf(theta / 2);
  F.evolve(P, T);
  let sr = 0, si = 0, nw = 0, nf = 0, l1 = 0;
  for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = idxF(x, y, z), o = 4 * idxW(x, y, z); let dw = 0, df = 0;
    for (let k = 0; k < 4; k++) {
      const ar = W.re[o + k], ai = W.im[o + k], br = F.wr[k][i], bi = F.wi[k][i];
      sr += ar * br + ai * bi; si += ar * bi - ai * br; dw += ar * ar + ai * ai; df += br * br + bi * bi;
    }
    nw += dw; nf += df; l1 += Math.abs(dw - df);
  }
  const fid = (sr * sr + si * si) / (nw * nf);
  ok('finite long-wavelength walk/Dirac comparison', fid > 0.995, `fidelity ${fid.toFixed(5)} after ${T} ticks, density L1 difference ${(l1 / 2).toFixed(4)}`);
}
// 6
{
  const n = 32, F = new BornReadoutField(n), m = 0.18;
  F.add({ id: 7, m, x0: [16, 16, 16], p0: [0, 0.4, 0], sigma: 2, chi: [[0.5, 0], [0.5, 0], [-0.5, 0], [-0.5, 0]], t: 0 });
  const P = F.parts.get(7); F.evolve(P, 3); F.readouts();
  const rec = F.record(P, 3, 0.9, 0.5);
  F.evolve(P, 3); const r = F.readouts();
  const d = [0, 1, 2].map((a) => F.disp(r.mean[a], rec.x[a]));
  // the window has rms radius 0.9 sqrt(3); the prior density across it pulls the posterior mean towards its denser side
  ok('complete record: normalised conditional output and positive joint probability', Math.abs(r.norm - 1) < 1e-10 && rec.probability > 0 && rec.probability <= 1, `norm ${r.norm.toFixed(12)}, mean minus record ${d.map((x) => x.toFixed(3)).join(', ')}`);
}
console.log(fails ? `${fails} check(s) failed` : 'all checks pass');
process.exit(fails ? 1 : 0);
